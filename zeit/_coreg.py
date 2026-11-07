# SPDX-License-Identifier: GPL-3.0-or-later
"""Global co-registration of each date of a cube to a reference image: ``zeit.coregister``.

A port of the global ``COREG`` of AROSICS 1.14.0 (Scheffler, Hollstein, Hofmann and Bochow 2017, *Remote Sensing*
9(7):676; Copyright (C) 2017-2026 Daniel Scheffler and the GFZ Helmholtz Centre for Geosciences, Potsdam;
Apache-2.0, https://git.gfz-potsdam.de/danschef/arosics). The shift is estimated by the compiled core
(``zeit._core.coreg``, ``rust/zeit-core/src/coreg.rs``); the cube is moved by it with the warp kernel of
``load_raster(..., like=)`` (GDAL's resampling).
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
                                      int(max_iter), integer, nd_cube, float(fill))

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
