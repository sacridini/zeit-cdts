# SPDX-License-Identifier: GPL-3.0-or-later
"""Co-registration of each date of a cube to a reference image: ``zeit.coregister`` and ``zeit.coregister_local``.

A port of the global ``COREG`` and of ``COREG_LOCAL`` of AROSICS 1.14.0 (Scheffler, Hollstein, Hofmann and Bochow
2017, *Remote Sensing* 9(7):676; Copyright (C) 2017-2026 Daniel Scheffler and the GFZ Helmholtz Centre for
Geosciences, Potsdam; Apache-2.0, https://git.gfz-potsdam.de/danschef/arosics). The shifts are estimated by the
compiled core (``zeit._core.coreg``, ``rust/zeit-core/src/coreg.rs`` and ``coreg_local.rs``); the global one moves
the cube with the warp kernel of ``load_raster(..., like=)`` (GDAL's resampling), the local one warps it with GDAL
(rasterio) from the tie points.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from rasterio.crs import CRS

#: The columns of the table of shifts.
SHIFT_COLUMNS = ("x_shift_px", "y_shift_px", "x_shift_map", "y_shift_map", "reliability", "ssim_before",
                 "ssim_after", "success", "reason", "window_x0", "window_y0", "window_x1", "window_y1",
                 "fft_size", "applied")


def _pick_band(da: xr.DataArray, band: Any, what: str) -> xr.DataArray:
    """One band of ``da``: ``band`` as a label of its ``band`` coordinate, else as a position; the first by default."""
    if "band" not in da.dims:
        if band is not None and not (isinstance(band, (int, np.integer)) and band == 0):
            raise ValueError(f"{what} has no band dimension, so band={band!r} cannot be chosen")
        return da
    if band is None:
        return da.isel(band=0, drop=True)
    if "band" in da.coords and band in da["band"].values:
        return da.sel(band=band, drop=True)
    if isinstance(band, (int, np.integer)) and -da.sizes["band"] <= band < da.sizes["band"]:
        return da.isel(band=int(band), drop=True)
    raise ValueError(f"{what} has no band {band!r}; its bands are {[str(b) for b in da['band'].values]}")


def _nodata_values(nodata: Any, reference: xr.DataArray, cube: xr.DataArray) -> Tuple[Optional[float], Optional[float]]:
    """NoData of the reference and of the cube (None: only NaN is missing)."""
    def stored(da):
        v = da.rio.nodata
        return None if v is None or np.isnan(v) else float(v)

    if isinstance(nodata, str):
        if nodata != "auto":
            raise ValueError(f"nodata must be 'auto', a number, None or a pair of them, got {nodata!r}")
        return stored(reference), stored(cube)
    if isinstance(nodata, (tuple, list)):
        if len(nodata) != 2:
            raise ValueError("nodata as a pair is (reference, cube)")
        return tuple(None if v is None or np.isnan(v) else float(v) for v in nodata)
    if nodata is None or np.isnan(nodata):
        return None, None
    return float(nodata), float(nodata)


def _prepare(values: np.ndarray, nodata: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """(float32 values with 0 or NoData where missing, validity, the fill value)."""
    v = np.asarray(values)
    ok = np.isfinite(v)
    if nodata is not None:
        ok &= v != nodata
    fill = 0.0 if nodata is None else float(nodata)
    out = np.where(ok, v, fill).astype(np.float32)
    return np.ascontiguousarray(out), np.ascontiguousarray(ok), fill


def _grids(cube: xr.DataArray, reference: xr.DataArray):
    """The intersection of the two grids: ``(ref rows, ref cols, cube rows, cube cols, transform of the reference)``,
    as slices; ValueError when the CRS, the cells or the alignment differ or the images do not overlap."""
    from ._warp import transform_of

    if cube.rio.crs is None or reference.rio.crs is None:
        raise ValueError("the cube and the reference need a CRS (rio.write_crs)")
    crs_c, crs_r = CRS.from_user_input(cube.rio.crs), CRS.from_user_input(reference.rio.crs)
    if crs_c != crs_r:
        raise ValueError(f"the cube ({crs_c.to_string()}) and the reference ({crs_r.to_string()}) are in different "
                         f"CRSs; reproject one of them first (AROSICS does not handle different projections)")
    if crs_c.is_geographic:
        raise ValueError("coregister needs a projected CRS (the shifts are in map units and pixels of a metric grid)")
    tc, tr = transform_of(cube), transform_of(reference)
    if tc.b or tc.d or tr.b or tr.d:
        raise ValueError("coregister does not handle rotated grids")
    if abs(tc.a - tr.a) > 1e-9 * abs(tr.a) or abs(tc.e - tr.e) > 1e-9 * abs(tr.e):
        raise ValueError(f"the cube's cells ({tc.a} x {tc.e}) and the reference's ({tr.a} x {tr.e}) differ: "
                         f"coregister needs both on the same cell size (resample the reference to the cube's "
                         f"grid first, for instance with zeit.load_raster(..., like=cube))")
    col_off, row_off = (tc.c - tr.c) / tr.a, (tc.f - tr.f) / tr.e
    if abs(col_off - round(col_off)) > 1e-4 or abs(row_off - round(row_off)) > 1e-4:
        raise ValueError(f"the cube's grid is {col_off:.4f} x {row_off:.4f} cells from the reference's, not a whole "
                         f"number: coregister needs the two on the same pixel grid (resample first)")
    col_off, row_off = int(round(col_off)), int(round(row_off))
    ch, cw = cube.sizes["y"], cube.sizes["x"]
    rh, rw = reference.sizes["y"], reference.sizes["x"]
    r0, r1 = max(0, row_off), min(rh, row_off + ch)
    c0, c1 = max(0, col_off), min(rw, col_off + cw)
    if r1 <= r0 or c1 <= c0:
        raise ValueError("the cube and the reference do not overlap")
    return (slice(r0, r1), slice(c0, c1), slice(r0 - row_off, r1 - row_off), slice(c0 - col_off, c1 - col_off), tr, tc)


def coregister(
    cube: xr.DataArray,
    reference: xr.DataArray,
    *,
    window_size: int = 256,
    window_position: Optional[Tuple[float, float]] = None,
    max_shift: float = 5,
    max_iter: int = 5,
    band: Any = None,
    ref_band: Any = None,
    resampling: str = "cubic",
    nodata: Union[str, float, None, tuple] = "auto",
    min_reliability: Optional[float] = None,
    n_jobs: int = -1,
) -> Tuple[xr.DataArray, pd.DataFrame]:
    """Each date of a cube moved to fit a reference image: AROSICS' global co-registration.

    For every date, the shift of one band to the reference's is found by the phase correlation of a
    window of ``window_size`` pixels at the centre of what the two images have in common (the cross power spectrum
    of the two windows, whose peak is their shift; its side maxima give the part of a pixel), as AROSICS'
    ``COREG`` does it (Scheffler et al. 2017, *Remote Sensing* 9(7):676; the compiled core is a port of
    ``arosics/CoReg.py`` 1.14.0, Apache-2.0). The integer shift is checked by moving the reference's window by
    it and correlating again (up to ``max_iter`` times); a shift above ``max_shift`` pixels, or with no clear
    match, is a failure. Every band of the date is then moved by the shift (all bands share it) with the warp
    kernel of :func:`zeit.load_raster` (``like=``), GDAL's resampling, onto the cube's own grid.

    The cube and the reference must be on the same pixel grid: the same projected CRS, the same cell size, the
    cells a whole number apart (their extents may differ: only what they share is matched). AROSICS resamples
    the window of the other image when they are not; here a reference on another grid is resampled first
    (``zeit.load_raster(..., like=cube)``).

    A date whose shift is not found (no valid data in common, no match, a shift above ``max_shift``...) is left as it
    is and flagged in the table (``success`` False, ``reason``); so is one below ``min_reliability``.

    Where it differs from AROSICS: the overlap is that of the two images' valid pixels (NaN, and the NoData value),
    not of footprint polygons, so a hole in it (a cloud masked as NaN) is avoided by the window instead of
    filled; and the aligned images are resampled onto the cube's grid, where AROSICS (``align_grids=False``) only
    moves the origin of the geotransform. See ``benchmarks/vs_originals/arosics.py`` for the comparison.

    Parameters
    ----------
    cube
        ``(time, band, y, x)``, ``(band, y, x)``, ``(time, y, x)`` or ``(y, x)`` in a projected CRS. A lazy cube is
        read a date at a time and the result is lazy.
    reference
        ``(band, y, x)`` or ``(y, x)`` in the same CRS and on the same pixel grid.
    window_size
        The matching window's side in pixels (AROSICS' ``ws``; 256). It is cut down to the overlap and the FFT
        takes the largest power of 2 inside it.
    window_position
        The window's centre as ``(x, y)`` map coordinates (AROSICS' ``wp``); by default the centroid of the
        overlap of the valid data.
    max_shift
        The largest shift accepted, in pixels (5).
    max_iter
        Tries to validate the integer shift (5).
    band, ref_band
        The band of the cube and of the reference to match, by label or position (the first by default).
    resampling
        The resampling that moves the images (``"cubic"``, ``"bilinear"``, ``"nearest"``, ``"lanczos"``...: those of
        :func:`zeit.load_raster`). A shift of a whole number of pixels is a plain copy of the cells.
    nodata
        Missing values besides NaN: ``"auto"`` (each image's raster NoData), a number for both, None, or a pair
        ``(reference, cube)``.
    min_reliability
        Treat a shift whose reliability (AROSICS', 0-100) is below this as not found. None: take every shift
        AROSICS' checks accept, as its ``COREG`` does.
    n_jobs
        Dates run at once: the processors zeit may use by default (``-1``).

    Returns
    -------
    aligned : xarray.DataArray
        The cube on its own grid with each date moved by its shift (the dates that failed as they were).
    shifts : pandas.DataFrame
        One row per date (index ``time``; a single row for a cube without dates) with the columns
        ``x_shift_px, y_shift_px`` (pixels: the amount the date was moved, ``x`` to the right and ``y`` down),
        ``x_shift_map, y_shift_map`` (map units; ``y`` is negative for a shift down on a north-up grid),
        ``reliability`` (0-100), ``ssim_before, ssim_after`` (structural similarity of the window to the
        reference's before and after the shift), ``success``, ``reason`` (why it failed), the matching window
        ``window_x0, window_y0, window_x1, window_y1`` (cells of the cube, ``[x0, x1)``), ``fft_size`` (the side
        of the FFT window of the spectrum the shift is read from, smaller than the matching window's largest power
        of 2 after an integer shift) and ``applied`` (the date was moved).

    Examples
    --------
    >>> aligned, shifts = zeit.coregister(cube, reference, band="nir", ref_band="nir")
    >>> shifts[["x_shift_px", "y_shift_px", "reliability", "success"]]
    """
    from . import _core
    from ._threads import resolve
    from ._warp import RESAMPLING_NAMES, Grid, _grid, to_grid, transform_of

    if not isinstance(cube, xr.DataArray) or not isinstance(reference, xr.DataArray):
        raise TypeError("coregister needs two xarray.DataArray (the cube and the reference)")
    for da, what in ((cube, "the cube"), (reference, "the reference")):
        if not {"y", "x"} <= set(da.dims):
            raise ValueError(f"{what} needs y and x dimensions")
    extra = [d for d in cube.dims if d not in ("time", "band", "y", "x")]
    if extra:
        raise ValueError(f"the cube has dimensions {extra}; coregister takes (time, band, y, x)")
    extra = [d for d in reference.dims if d not in ("band", "y", "x")]
    if extra:
        raise ValueError(f"the reference has dimensions {extra}; it must be (band, y, x) or (y, x)")
    if resampling not in RESAMPLING_NAMES:
        raise ValueError(f"unknown resampling {resampling!r}; use one of {', '.join(RESAMPLING_NAMES)}")
    if not (max_shift > 0) or int(max_iter) < 1 or int(window_size) < 16:
        raise ValueError("max_shift must be positive, max_iter at least 1 and window_size at least 16")

    rows_r, cols_r, rows_c, cols_c, t_ref, t_cube = _grids(cube, reference)
    nd_ref, nd_cube = _nodata_values(nodata, reference, cube)

    ref2d = _pick_band(reference, ref_band, "the reference").isel(y=rows_r, x=cols_r)
    match = _pick_band(cube, band, "the cube").isel(y=rows_c, x=cols_c)
    ref_data, ref_ok, _ = _prepare(ref2d.transpose("y", "x").values, nd_ref)
    if not ref_ok.any():
        raise ValueError("the reference has no valid data where the cube is")

    wpos = None
    if window_position is not None:
        wx, wy = (float(v) for v in window_position)
        wpos = ((wx - t_ref.c) / t_ref.a - cols_r.start, (wy - t_ref.f) / t_ref.e - rows_r.start)
        h, w = ref_data.shape
        if not (0 <= wpos[0] < w and 0 <= wpos[1] < h):
            raise ValueError(f"window_position {window_position} is outside the overlap of the cube and the reference")

    has_time = "time" in cube.dims
    n = cube.sizes["time"] if has_time else 1
    integer = None
    if np.issubdtype(cube.dtype, np.integer):
        info = np.iinfo(cube.dtype)
        integer = (float(info.min), float(info.max))
    threads = max(1, min(resolve(n_jobs), n))

    def one(k: int) -> dict:
        plane = match.isel(time=k) if has_time else match
        data, ok, fill = _prepare(plane.transpose("y", "x").values, nd_cube)
        return _core.coreg.coregister(ref_data, ref_ok, data, ok, int(window_size), wpos, float(max_shift),
                                      int(max_iter), integer, nd_cube, float(fill),
                                      (abs(t_cube.a), abs(t_cube.e)))

    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(one, range(n)))
    else:
        results = [one(k) for k in range(n)]

    c0, r0 = cols_c.start, rows_c.start
    rows = []
    for res in results:
        ok = bool(res["success"])
        reason = res["reason"]
        if ok and min_reliability is not None and not res["reliability"] >= float(min_reliability):
            ok, reason = False, f"reliability {res['reliability']:.1f} below min_reliability {min_reliability}"
        wx0, wy0, wx1, wy1 = res["window"]
        has_window = wx1 > wx0
        rows.append({
            "x_shift_px": res["x_shift"], "y_shift_px": res["y_shift"],
            "x_shift_map": res["x_shift"] * t_cube.a, "y_shift_map": res["y_shift"] * t_cube.e,
            "reliability": res["reliability"], "ssim_before": res["ssim_before"], "ssim_after": res["ssim_after"],
            "success": ok, "reason": None if ok else reason,
            "window_x0": c0 + wx0 if has_window else -1, "window_y0": r0 + wy0 if has_window else -1,
            "window_x1": c0 + wx1 if has_window else -1, "window_y1": r0 + wy1 if has_window else -1,
            "fft_size": res["fft_size"], "applied": ok,
        })
    index = pd.Index(cube["time"].values, name="time") if has_time else pd.RangeIndex(1, name="time")
    shifts = pd.DataFrame(rows, index=index, columns=list(SHIFT_COLUMNS))
    for col in ("window_x0", "window_y0", "window_x1", "window_y1", "fft_size"):
        shifts[col] = shifts[col].astype("int64")
    shifts["success"] = shifts["success"].astype(bool)
    shifts["applied"] = shifts["applied"].astype(bool)

    if not shifts["applied"].any():
        return cube.copy(), shifts

    grid: Grid = _grid(t_cube, (cube.sizes["y"], cube.sizes["x"]), CRS.from_user_input(cube.rio.crs))

    def moved(plane: xr.DataArray, dx: float, dy: float) -> xr.DataArray:
        # the content moves by (dx, dy) cells: the cells' coordinates move, and the warp takes it back to the grid
        mx, my = dx * t_cube.a, dy * t_cube.e
        shifted = plane.assign_coords(x=plane["x"].values + mx, y=plane["y"].values + my)
        shifted = shifted.rio.write_transform(t_cube @ t_cube.__class__.translation(dx, dy))
        out = to_grid(shifted, grid, resampling)
        return out.assign_coords(x=cube["x"].values, y=cube["y"].values)

    planes = []
    for k in range(n):
        plane = cube.isel(time=k) if has_time else cube
        if shifts["applied"].iloc[k]:
            plane = moved(plane, float(shifts["x_shift_px"].iloc[k]), float(shifts["y_shift_px"].iloc[k]))
        planes.append(plane)
    if has_time:
        aligned = xr.concat(planes, dim="time", coords="minimal", compat="override", join="override")
        aligned = aligned.assign_coords(time=cube["time"]).transpose(*cube.dims)
    else:
        aligned = planes[0].transpose(*cube.dims)
    aligned.name = cube.name
    aligned.attrs = {k: v for k, v in cube.attrs.items()}
    aligned.attrs["coregister"] = (
        f"{int(shifts['applied'].sum())} of {n} dates moved to the reference by the global shift of AROSICS "
        f"(Scheffler et al. 2017, Apache-2.0), resampling={resampling}")
    if cube.rio.crs is not None:
        aligned = aligned.rio.write_crs(cube.rio.crs)
    return aligned, shifts


#: The columns of the table of tie points.
TIE_POINT_COLUMNS = ("time", "point_id", "col", "row", "x", "y", "x_shift_px", "y_shift_px", "x_shift_map",
                     "y_shift_map", "abs_shift", "angle", "reliability", "ssim_before", "ssim_after", "ssim_improved",
                     "success", "reason", "win_size_x", "win_size_y", "fft_size", "outlier_reliability",
                     "outlier_ssim", "outlier_ransac", "outlier", "used", "date_applied")

#: Most tie points GDAL's thin plate spline is given (AROSICS: "a GDAL bottleneck").
MAX_GCPS = 7000


def _grid_points(size: int, res: float) -> np.ndarray:
    """AROSICS' ``np.arange(0, size + grid_res, grid_res)``: the corners of the grid's cells, from the first one to
    the one past the last."""
    return np.arange(0, size + res, res)


class _GdalOptionNotice:
    """A logging filter that drops GDAL's notice that the warper does not support the transformer's options."""

    def filter(self, record) -> bool:
        msg = record.getMessage()
        return not ("does not support option" in msg and ("SRC_METHOD" in msg or "MAX_GCP_ORDER" in msg))


def _warp_gcps(plane: np.ndarray, gcps: list, crs: CRS, transform: Any, method: str, order: Optional[int],
               resampling: str, nodata: Optional[float], threads: int) -> np.ndarray:
    """``plane`` (``(y, x)`` or ``(band, y, x)``) warped onto its own grid by GDAL from ground control points given as
    ``(col, row, x, y)`` (a point of the image and the map position it has to take)."""
    import logging

    from rasterio.control import GroundControlPoint
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    # rasterio hands the options to the transformer, which takes them, and to the warper, which says it does not
    # know them
    log = logging.getLogger("rasterio._env")
    if not any(isinstance(f, _GdalOptionNotice) for f in log.filters):
        log.addFilter(_GdalOptionNotice())

    g = [GroundControlPoint(row=r, col=c, x=x, y=y, z=0.0) for c, r, x, y in gcps]
    extra = {"SRC_METHOD": "GCP_TPS"} if method == "tps" else {"SRC_METHOD": "GCP_POLYNOMIAL"}
    if method == "polynomial" and order is not None:
        extra["MAX_GCP_ORDER"] = str(int(order))
    floating = np.issubdtype(plane.dtype, np.floating)
    nd = nodata if nodata is not None else (np.nan if floating else None)
    out = np.empty_like(plane)
    reproject(plane, out, gcps=g, src_crs=crs, src_nodata=nd, dst_transform=transform, dst_crs=crs, dst_nodata=nd,
              resampling=getattr(Resampling, resampling), num_threads=threads, **extra)
    return out


def coregister_local(
    cube: xr.DataArray,
    reference: xr.DataArray,
    *,
    grid_res: float = 200,
    window_size: Union[int, Tuple[int, int]] = (256, 256),
    max_shift: float = 5,
    max_iter: int = 5,
    min_reliability: float = 60,
    tie_point_filter_level: int = 3,
    rs_max_outlier: float = 10,
    rs_tolerance: float = 2.5,
    rs_random_state: Optional[int] = 0,
    warp: str = "auto",
    order: Optional[int] = None,
    min_points: int = 5,
    band: Any = None,
    ref_band: Any = None,
    resampling: str = "cubic",
    nodata: Union[str, float, None, tuple] = "auto",
    arosics_compat: bool = False,
    n_jobs: int = -1,
) -> Tuple[xr.DataArray, pd.DataFrame]:
    """Each date of a cube warped to fit a reference image by its local shifts: AROSICS' ``COREG_LOCAL``.

    For every date, the shift of one band to the reference's is found at the points of a grid (the phase
    correlation of a window of ``window_size`` pixels centred on each point, as :func:`zeit.coregister` does for one
    window); the points whose shift is not reliable are filtered out (a low reliability, no gain in the structural
    similarity, and a RANSAC fit of an affine model to the shifts); and what remains are ground control points that
    warp each date onto the cube's grid with GDAL: a thin plate spline, as AROSICS does, or a polynomial. It
    corrects shifts that vary across the scene (a misregistration of the viewing geometry, of a bad orthorectification)
    where :func:`zeit.coregister` corrects one shift for the whole date. The compiled core
    (``rust/zeit-core/src/coreg_local.rs``) is a port of ``arosics/CoReg_local.py`` and
    ``arosics/Tie_Point_Grid.py`` 1.14.0 (Scheffler et al. 2017, *Remote Sensing* 9(7):676; Apache-2.0), RANSAC
    included: scikit-image's, with numpy's random generator reproduced, so the same ``rs_random_state`` flags the
    same points.

    As in :func:`zeit.coregister`, the cube and the reference must be on the same pixel grid (the same projected
    CRS and cell size, the cells a whole number apart), and the overlap is that of the images' valid pixels. The
    grid of points is AROSICS': the corners of the cube's cells ``0, grid_res, 2 * grid_res, ...`` in each
    direction, up to one step past the image; a point is used where it is on data of both images, and its window is
    cut down to the overlap as in the global case.

    Parameters
    ----------
    cube
        ``(time, band, y, x)``, ``(band, y, x)``, ``(time, y, x)`` or ``(y, x)`` in a projected CRS. A lazy cube is
        read a date at a time; the result is in memory.
    reference
        ``(band, y, x)`` or ``(y, x)`` in the same CRS and on the same pixel grid.
    grid_res
        The spacing of the tie points in pixels (AROSICS' ``grid_res``; 200).
    window_size
        The matching window's side in pixels, an int or ``(x, y)`` of equal values (AROSICS' ``window_size``).
    max_shift, max_iter
        The largest shift accepted in pixels, and the tries to validate the integer shift (as :func:`coregister`).
    min_reliability
        Tie points below this reliability (0-100) are flagged (AROSICS' ``min_reliability``, 60).
    tie_point_filter_level
        AROSICS' ``tieP_filter_level``: 0 no filter, 1 reliability, 2 plus the structural similarity (the point's
        window must not become less similar to the reference's by the shift), 3 plus RANSAC (default).
    rs_max_outlier, rs_tolerance
        RANSAC's expected outliers in percent (10) and the tolerance of that percent (2.5): the residual threshold
        is searched until the outliers are within ``rs_max_outlier`` +/- ``rs_tolerance``.
    rs_random_state
        RANSAC's seed (AROSICS' ``rs_random_state``, 0); ``None`` takes one from the clock.
    warp
        ``"tps"`` (GDAL's thin plate spline, which AROSICS always uses), ``"polynomial"`` (GDAL's polynomial of
        ``order``), or ``"auto"`` (``"tps"``).
    order
        The polynomial's order (1 to 3) for ``warp="polynomial"``; ``None``: GDAL's choice from the number of points.
    min_points
        A date with fewer valid tie points than this (AROSICS' ``min_points_local_corr``, 5) is left as it is and
        flagged (AROSICS then corrects with the mean shift of the points).
    band, ref_band
        The band of the cube and of the reference to match, by label or position (the first by default). Every
        band of a date is warped with the same points.
    resampling
        GDAL's resampling of the warp (``"cubic"``, ``"bilinear"``, ``"nearest"``, ``"lanczos"``...).
    nodata
        Missing values besides NaN: ``"auto"`` (each image's raster NoData), a number for both, None, or a pair
        ``(reference, cube)``.
    arosics_compat
        Reproduce a bug of AROSICS 1.14.0, to compare with it: when the reliability or SSIM filter has flagged points,
        RANSAC's flags are put on the wrong rows of its table (a pandas index misalignment), so some of the points it
        found to be outliers stay and as many good ones are dropped. ``False`` (default): each flag is on its point.
    n_jobs
        Threads: the dates, and the points of each, run on them (``-1``: the processors zeit may use by default).

    Returns
    -------
    aligned : xarray.DataArray
        The cube on its own grid with each date warped (dates with too few valid tie points as they were).
    tie_points : pandas.DataFrame
        One row per tie point of every date: ``time``, ``point_id`` (its position in the grid), ``col``/``row``
        (the position in the cube's pixel corners), ``x``/``y`` (map), ``x_shift_px``/``y_shift_px`` and
        ``x_shift_map``/``y_shift_map`` (the amount the date has to be moved there, ``x`` right and ``y`` down in
        pixels; NaN where the match failed), ``abs_shift`` (length in map units) and ``angle`` (degrees from north),
        ``reliability``, ``ssim_before``/``ssim_after``/``ssim_improved``, ``success``/``reason``,
        ``win_size_x``/``win_size_y`` (the matching window), ``fft_size``, the flags of the filters
        ``outlier_reliability``, ``outlier_ssim``, ``outlier_ransac`` and their union ``outlier``, ``used`` (the point
        is a control point of the warp) and ``date_applied``. ``tie_points.attrs["dates"]`` is a table with one
        row per date: the points placed, matched, used, whether the date was warped and why not.

    Examples
    --------
    >>> aligned, points = zeit.coregister_local(cube, reference, grid_res=100, window_size=128)
    >>> points.groupby("time")["used"].sum()
    """
    from . import _core
    from ._threads import resolve
    from ._warp import RESAMPLING_NAMES

    if not isinstance(cube, xr.DataArray) or not isinstance(reference, xr.DataArray):
        raise TypeError("coregister_local needs two xarray.DataArray (the cube and the reference)")
    for da, what in ((cube, "the cube"), (reference, "the reference")):
        if not {"y", "x"} <= set(da.dims):
            raise ValueError(f"{what} needs y and x dimensions")
    extra = [d for d in cube.dims if d not in ("time", "band", "y", "x")]
    if extra:
        raise ValueError(f"the cube has dimensions {extra}; coregister_local takes (time, band, y, x)")
    extra = [d for d in reference.dims if d not in ("band", "y", "x")]
    if extra:
        raise ValueError(f"the reference has dimensions {extra}; it must be (band, y, x) or (y, x)")
    if resampling not in RESAMPLING_NAMES:
        raise ValueError(f"unknown resampling {resampling!r}; use one of {', '.join(RESAMPLING_NAMES)}")
    if warp not in ("auto", "tps", "polynomial"):
        raise ValueError(f"warp must be 'auto', 'tps' or 'polynomial', got {warp!r}")
    method = "tps" if warp == "auto" else warp
    if order is not None and (method != "polynomial" or int(order) not in (1, 2, 3)):
        raise ValueError("order (1, 2 or 3) is for warp='polynomial'")
    if isinstance(window_size, (tuple, list)):
        if len(window_size) != 2 or int(window_size[0]) != int(window_size[1]):
            raise ValueError("window_size must be one number or two equal ones: the matching window is square")
        window_size = window_size[0]
    window_size = int(window_size)
    if not (grid_res > 0):
        raise ValueError("grid_res must be positive")
    if not (max_shift > 0) or int(max_iter) < 1 or window_size < 16:
        raise ValueError("max_shift must be positive, max_iter at least 1 and window_size at least 16")
    if int(tie_point_filter_level) not in (0, 1, 2, 3):
        raise ValueError("tie_point_filter_level must be 0, 1, 2 or 3")
    if not (0 < rs_max_outlier < 100):
        raise ValueError("rs_max_outlier must be between 0 and 100 (percent)")
    if int(min_points) < 1:
        raise ValueError("min_points must be at least 1")

    rows_r, cols_r, rows_c, cols_c, t_ref, t_cube = _grids(cube, reference)
    nd_ref, nd_cube = _nodata_values(nodata, reference, cube)
    ref2d = _pick_band(reference, ref_band, "the reference").isel(y=rows_r, x=cols_r)
    match = _pick_band(cube, band, "the cube").isel(y=rows_c, x=cols_c)
    ref_data, ref_ok, _ = _prepare(ref2d.transpose("y", "x").values, nd_ref)
    if not ref_ok.any():
        raise ValueError("the reference has no valid data where the cube is")

    has_time = "time" in cube.dims
    n = cube.sizes["time"] if has_time else 1
    integer = None
    if np.issubdtype(cube.dtype, np.integer):
        info = np.iinfo(cube.dtype)
        integer = (float(info.min), float(info.max))
    threads = max(1, resolve(n_jobs))
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    gx, gy = np.meshgrid(_grid_points(nx, grid_res), _grid_points(ny, grid_res))
    points = np.ascontiguousarray(np.column_stack([gx.ravel(), gy.ravel()]), dtype=np.float64)
    geo = (float(t_cube.c), float(t_cube.a), float(t_cube.f), float(t_cube.e))
    offset = (float(cols_c.start), float(rows_c.start))
    seed = None if rs_random_state is None else int(rs_random_state)

    def one(k: int) -> dict:
        plane = match.isel(time=k) if has_time else match
        data, ok, fill = _prepare(plane.transpose("y", "x").values, nd_cube)
        return _core.coreg.tie_points(ref_data, ref_ok, data, ok, points, offset, geo, window_size, float(max_shift),
                                      int(max_iter), integer, nd_cube, float(fill), int(tie_point_filter_level),
                                      float(min_reliability), float(rs_max_outlier), float(rs_tolerance), 15, seed,
                                      bool(arosics_compat), int(threads))

    if threads > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=min(threads, n)) as pool:
            results = list(pool.map(one, range(n)))
    else:
        results = [one(k) for k in range(n)]

    times = cube["time"].values if has_time else np.zeros(1, dtype=np.int64)
    frames, dates, gcps_of = [], [], []
    for k, res in enumerate(results):
        idx = np.asarray(res["index"], dtype=np.int64)
        success = np.asarray(res["success"], dtype=bool)
        outlier = np.asarray(res["outlier"], dtype=bool)
        used = success & ~outlier
        frame = pd.DataFrame({
            "time": np.repeat(times[k:k + 1], len(idx)), "point_id": idx,
            "col": points[idx, 0] if len(idx) else np.zeros(0), "row": points[idx, 1] if len(idx) else np.zeros(0),
            "x": res["x_map"], "y": res["y_map"], "x_shift_px": res["x_shift_px"], "y_shift_px": res["y_shift_px"],
            "x_shift_map": res["x_shift_map"], "y_shift_map": res["y_shift_map"], "abs_shift": res["abs_shift"],
            "angle": res["angle"], "reliability": res["reliability"], "ssim_before": res["ssim_before"],
            "ssim_after": res["ssim_after"], "ssim_improved": np.asarray(res["ssim_improved"], dtype=bool),
            "success": success, "reason": res["reason"],
            "win_size_x": [w[2] - w[0] for w in res["window"]], "win_size_y": [w[3] - w[1] for w in res["window"]],
            "fft_size": res["fft_size"], "outlier_reliability": np.asarray(res["l1"], dtype=bool),
            "outlier_ssim": np.asarray(res["l2"], dtype=bool), "outlier_ransac": np.asarray(res["l3"], dtype=bool),
            "outlier": outlier, "used": used,
        })
        n_used = int(used.sum())
        applied = n_used >= int(min_points)
        if applied:
            why = None
        elif len(idx) == 0:
            why = "no tie point on data of both images"
        elif not success.any():
            why = "no tie point matched"
        else:
            why = f"only {n_used} valid tie point(s), fewer than min_points={int(min_points)}"
        frame["date_applied"] = applied
        for col in ("win_size_x", "win_size_y", "fft_size"):
            frame[col] = frame[col].astype("int64")
        frames.append(frame)
        dates.append({"time": times[k], "points": len(idx), "matched": int(success.sum()), "used": n_used,
                      "applied": applied, "reason": why})
        gcps_of.append(frame[used] if applied else None)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(TIE_POINT_COLUMNS))
    table = table[list(TIE_POINT_COLUMNS)]
    table.attrs["dates"] = pd.DataFrame(dates).set_index("time")

    if not any(g is not None for g in gcps_of):
        return cube.copy(), table

    crs = CRS.from_user_input(cube.rio.crs)
    mid = nd_cube

    def warped(k: int) -> xr.DataArray:
        plane = cube.isel(time=k) if has_time else cube
        used_pts = gcps_of[k]
        if used_pts is None:
            return plane
        if len(used_pts) > MAX_GCPS:
            used_pts = used_pts.iloc[np.linspace(0, len(used_pts) - 1, MAX_GCPS).round().astype(int)]
        gcps = list(zip(used_pts["col"], used_pts["row"], used_pts["x"] + used_pts["x_shift_map"],
                        used_pts["y"] + used_pts["y_shift_map"]))
        lead = [d for d in plane.dims if d not in ("y", "x")]
        p = plane.transpose(*lead, "y", "x")
        values = np.asarray(p.values)
        if mid is not None and np.issubdtype(values.dtype, np.floating):
            values = np.where(np.isfinite(values), values, mid).astype(values.dtype)
        out = _warp_gcps(values, gcps, crs, t_cube, method, order, resampling, mid, 1)
        return p.copy(data=out).transpose(*plane.dims)

    if threads > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=min(threads, n)) as pool:
            planes = list(pool.map(warped, range(n)))
    else:
        planes = [warped(k) for k in range(n)]
    if has_time:
        aligned = xr.concat(planes, dim="time", coords="minimal", compat="override", join="override")
        aligned = aligned.assign_coords(time=cube["time"]).transpose(*cube.dims)
    else:
        aligned = planes[0].transpose(*cube.dims)
    aligned.name = cube.name
    aligned.attrs = {k: v for k, v in cube.attrs.items()}
    n_applied = sum(g is not None for g in gcps_of)
    aligned.attrs["coregister_local"] = (
        f"{n_applied} of {n} dates warped to the reference with the tie points of AROSICS' COREG_LOCAL "
        f"(Scheffler et al. 2017, Apache-2.0), warp={method}, resampling={resampling}")
    aligned = aligned.rio.write_crs(cube.rio.crs)
    return aligned, table
