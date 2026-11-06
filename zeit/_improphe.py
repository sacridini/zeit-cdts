"""ImproPhe, sharpening a coarse field to the grid of a finer sensor: ``zeit.improphe``.

FORCE's CF-ImproPhe (a continuous field such as a 500 m phenology metric of each year, sharpened with the fine
scenes of that year) and L2-ImproPhe (the bands of a coarser sensor, such as Landsat at 30 m, sharpened with the fine
scenes of a finer one, such as Sentinel-2 at 10 m). The method is Frantz, Stellmes, Roeder, Udelhoven, Mader and Hill
(2016), IEEE TGRS 54(7):4153-4164, doi 10.1109/TGRS.2016.2537929; the implementation is a port of FORCE 3.11
(David Frantz, GPL-3.0-or-later), see ``zeit_core::improphe`` for the method and where it departs from FORCE.
"""

import warnings
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ._tsa import _CUM

#: FORCE's default seasonal windows (``SEASONAL_WINDOW``): four windows of the year.
DEFAULT_SEASONS = (1, 91, 181, 271, 365)


def _resolution(fine: xr.DataArray) -> float:
    """The pixel size of the fine cube (map units); it must be square and regular."""
    steps = []
    for dim in ("x", "y"):
        v = np.asarray(fine[dim].values, dtype=float)
        if v.size < 2:
            raise ValueError(f"fine needs a {dim!r} coordinate with at least two values to know the pixel size")
        d = np.abs(np.diff(v))
        if not np.allclose(d, d[0], rtol=1e-6, atol=0):
            raise ValueError(f"the {dim!r} coordinate of fine is not regularly spaced")
        steps.append(float(d[0]))
    if not np.isclose(steps[0], steps[1], rtol=1e-6):
        raise ValueError(f"the pixels of fine are not square ({steps[0]} x {steps[1]}): ImproPhe's radii are in pixels")
    return steps[0]


def _radius(size: float, res: float, what: str, minimum: int) -> int:
    """FORCE's ``(int)(size / RESOLUTION)``; a size that is not a multiple of the pixel size is warned about (FORCE
    refuses it)."""
    r = float(size) / res
    if abs(r - round(r)) > 5e-3:
        warnings.warn(f"{what}={size} is not a multiple of the pixel size {res:g} (FORCE refuses this); the radius is "
                      f"{int(r + 1e-9)} pixels", stacklevel=3)
    k = int(np.floor(r + 1e-9))
    if k < minimum:
        raise ValueError(f"{what}={size} is {r:.2f} pixels of {res:g}: the radius must be at least {minimum} pixel(s)")
    return k


def _missing(da: xr.DataArray, nodata: Any, fine: bool) -> list:
    """The values that mean "missing" besides NaN: the raster's NoData (``"auto"``; for the fine integer cube
    without one, 0, as ``tasseled_cap``), a number, or none."""
    if isinstance(nodata, str) and nodata != "auto":
        raise ValueError(f"nodata must be 'auto', a number or None, got {nodata!r}")
    if nodata == "auto":
        stored = da.rio.nodata
        if stored is not None and not np.isnan(stored):
            return [stored]
        return [0] if fine and np.issubdtype(da.dtype, np.integer) else []
    return [] if nodata is None else [nodata]


def _float32(values: np.ndarray, missing: Sequence[float]) -> np.ndarray:
    out = np.array(values, dtype=np.float32, order="C")        # always a copy: the masking is in place
    for m in missing:
        out[out == np.float32(m)] = np.nan
    return out


def _year_values(index: pd.Index) -> np.ndarray:
    if np.issubdtype(index.dtype, np.datetime64):
        return np.asarray(pd.DatetimeIndex(index).year)
    return np.asarray(index).astype(int)


def _layout(coarse: xr.DataArray, by: Optional[str]) -> Tuple[Optional[str], str, np.ndarray, Optional[str]]:
    """How to read `coarse`: ``(by, layer dim, layer values, field dim)``. The layers are the years (CF) or the
    scenes (L2); the field dimension, if any, holds the fields or bands."""
    others = [d for d in coarse.dims if d not in ("y", "x")]
    if not {"y", "x"} <= set(coarse.dims):
        raise ValueError("coarse needs y and x dimensions")
    if by == "auto":
        time_is_date = "time" in others and np.issubdtype(coarse["time"].dtype, np.datetime64)
        if "year" in others:
            by = "year"
        elif time_is_date:
            years = _year_values(coarse.indexes["time"])
            if len(set(years.tolist())) < len(years):
                by = None
            else:
                raise ValueError("coarse has one date per year: say by='year' if it is a field with one layer per year "
                                 "(CF-ImproPhe), or by=None if the dates are scenes of a sensor (L2-ImproPhe)")
        elif "time" not in others and "band" in others and np.issubdtype(coarse["band"].dtype, np.integer) \
                and 1900 <= int(coarse["band"].min()) and int(coarse["band"].max()) <= 2300:
            by = "year"
        else:
            raise ValueError("cannot tell what coarse is: pass by='year' (a field with a layer per year, dimension "
                             "'year', 'time' or 'band' of years) or by=None (scenes along 'time', bands along 'band')")
    if by == "year":
        for name in ("year", "time", "band"):
            if name in others:
                layer = name
                break
        else:
            raise ValueError("by='year': coarse needs a 'year' dimension (or 'time' with one date per year, or 'band' "
                             "named by the years)")
        years = _year_values(coarse.indexes[layer])
        if len(set(years.tolist())) != len(years):
            raise ValueError(f"by='year': the years of coarse's {layer!r} are not unique: {years.tolist()}")
        rest = [d for d in others if d != layer]
        values = years
    elif by is None:
        if "time" not in others:
            raise ValueError("by=None: coarse needs a 'time' dimension, the scenes to sharpen")
        layer, rest = "time", [d for d in others if d != "time"]
        values = np.asarray(coarse["time"].values)
    else:
        raise ValueError(f"by must be 'year', None or 'auto', got {by!r}")
    if len(rest) > 1:
        raise ValueError(f"coarse has more dimensions than {layer!r}, one more and y, x: {list(coarse.dims)}")
    return by, layer, values, (rest[0] if rest else None)


def _on_grid(coarse: xr.DataArray, fine: xr.DataArray, values: np.ndarray, res: float) -> np.ndarray:
    """`values` ``(n, ny, nx)`` of coarse on the grid of fine: as they are if the grids are the same, else
    nearest neighbour as FORCE's GDAL read does."""
    same = coarse.sizes["y"] == fine.sizes["y"] and coarse.sizes["x"] == fine.sizes["x"]
    if same:
        tol = 1e-3 * res
        same = (np.allclose(coarse["y"].values, fine["y"].values, rtol=0, atol=tol)
                and np.allclose(coarse["x"].values, fine["x"].values, rtol=0, atol=tol))
    if same:
        return values
    import rioxarray  # noqa: F401 - registers .rio
    from rasterio.enums import Resampling

    crs_c, crs_f = coarse.rio.crs, fine.rio.crs
    if crs_c is None or crs_f is None:
        raise ValueError("coarse is not on the grid of fine and one of them has no CRS: write_crs() on both, or give "
                         "coarse on the grid of fine")
    if crs_c != crs_f:
        raise ValueError(f"coarse ({crs_c}) and fine ({crs_f}) are in different CRS: reproject one first; ImproPhe "
                         "needs the same CRS, as FORCE's datacube has")
    tmp = xr.DataArray(values, dims=("band", "y", "x"),
                       coords={"band": np.arange(values.shape[0]), "y": coarse["y"].values, "x": coarse["x"].values})
    tmp = tmp.rio.write_crs(crs_c).rio.write_nodata(np.nan)
    ref = fine.isel(time=0, band=0, drop=True)
    return np.ascontiguousarray(tmp.rio.reproject_match(ref, resampling=Resampling.nearest).values, dtype=np.float32)


def improphe(coarse: Any, fine: Any, *, by: Any = "auto", seasons: Sequence[int] = DEFAULT_SEASONS,
             kernel_size: float = 2500.0, texture_size: float = 330.0, pc_variance: float = 0.975,
             force: bool = False, pca: Optional[str] = None, nodata: Any = "auto", chunks: Any = None, n_jobs: int = -1) -> xr.DataArray:
    """Sharpen a coarse field to the pixels of a finer sensor (FORCE's CF-ImproPhe and L2-ImproPhe).

    Each pixel's prediction is the weighted mean of the coarse values at the pixels of its neighbourhood that look
    like it in the fine data. The fine scenes are averaged in seasonal windows of the year, reduced by principal
    components to 97.5% of their variance and standardized; the neighbours of a pixel are those inside a circle of
    radius `kernel_size` whose components differ from the pixel's by a mean absolute difference under one of
    0.025, 0.05, 0.1, 0.2 and 0.4 (the smallest that leaves at least 0.5% of the circle's pixels, the pixel itself
    counting); the weight of a neighbour is the inverse of the product of three factors, its spectral difference,
    its fine texture (the largest standard deviation of the components in a window of radius `texture_size`) and
    its coarse texture, each scaled over the neighbours to ``1 + exp(25 x - 7.5)`` with x between 0 and 1. A pixel
    without a neighbour (no data in the fine components) keeps the coarse value. The method is Frantz et al. (2016,
    IEEE TGRS 54(7):4153-4164, doi 10.1109/TGRS.2016.2537929); this is a port of FORCE 3.11 (David Frantz,
    GPL-3.0-or-later).

    Parameters
    ----------
    coarse
        What to sharpen, on any grid of the same CRS (nearest neighbour onto the grid of `fine`, as FORCE's read
        does; see ``rioxarray``'s ``reproject_match``). With ``by="year"`` (CF-ImproPhe) a continuous field with a
        layer for each year, ``(year, y, x)`` or ``(year, field, y, x)``: the layer's dimension is ``year``, or
        ``time`` with one date per year, or ``band`` named by the years; the extra dimension holds several fields,
        e.g. the phenology metrics of ``zeit.phenology``. Each year is predicted with the fine scenes of that year
        (a window crossing the new year, as ``seasons=(330, 60, 330)``, takes its days of the neighbouring years as
        FORCE's ``average_season`` does). With ``by=None`` (L2-ImproPhe) scenes of a coarser sensor
        ``(time, band, y, x)`` (or ``(time, y, x)``), each predicted with the one seasonal mean of all the fine
        dates. NaN, and a NoData value (`nodata`), are missing: those pixels come out NaN and are not neighbours.
    fine
        The fine reflectance ``(time, band, y, x)`` with a datetime ``time`` coordinate (NaN where missing; any scale:
        the distances are on standardized components; with ``force=True`` the seasonal means are cut to integers,
        which is FORCE's only for the integer scale FORCE reads, reflectance times 10000). Its grid is the output's.
        Cloudy dates should be NaN-masked first. In memory or dask (loaded: see Notes).
    by
        ``"year"``, ``None`` or ``"auto"`` (a ``year`` dimension means ``"year"``; a ``time`` with several dates in a
        year means ``None``; anything else is asked).
    seasons
        Day-of-year limits of the seasonal windows, ``n + 1`` numbers for ``n`` windows (FORCE's ``SEASONAL_WINDOW``,
        default ``1 91 181 271 365``), in FORCE's 365-day calendar (29 February is day 60, with 1 March). A limit
        larger than the next one makes a window that crosses the new year. A window with no scene at a pixel takes
        the mean of all the scenes.
    kernel_size, texture_size
        The radii of the prediction circle and of the texture window, in map units (FORCE's ``KERNEL_SIZE`` and
        ``KERNEL_TEXT``), turned into pixels of `fine` by truncation; they should be multiples of the pixel size
        (FORCE refuses others; here a warning). The defaults, 2500 and 330, are for 10 m pixels (radii of 250 and
        33 pixels). The cost per pixel grows with the area of the circle: measured on a 297 x 297 tile with 6 threads
        of a desktop (the neighbourhood step alone, one year or scene), about 8e5 pixels/s for a radius of 10
        pixels and 6e4 for 50 (a sixth of that on one thread); a circle cut by the edge of a small tile is cheaper
        than a whole one, so a large tile costs more per pixel, and the default radius of 250 pixels has 25 times
        the area of 50: expect minutes for a 1000 x 1000 tile on 8 threads. Pass a smaller `kernel_size` for a
        quick look.
    pc_variance
        Fraction of the variance the components keep (FORCE: 0.975).
    force
        ``False`` (default): the method as the paper describes it: the principal components, float seasonal means,
        the last window including day 365. ``True``: FORCE's arithmetic to the letter, to reproduce its output: the
        means truncated towards zero (FORCE keeps 16-bit integers: only right for reflectance times 10000), the upper
        limit of every window excluded (day 365 falls in no window of the default seasons), and FORCE's components
        (``pca="force"``). With it the result truncated to integers is FORCE's in every value of the comparison in
        the docs, against FORCE built on GSL's reference CBLAS; FORCE linked to OpenBLAS differs in the last bits of
        the eigenvectors, and so in a few thousandths of the values (more where the seasonal means are collinear).
    pca
        The components whose distances choose the neighbours; by default as `force` says. ``"principal"``: the
        principal components (standardized). ``"force"``: FORCE's: its ``pca`` projects the data on the transposed
        eigenvector matrix (``gsl_blas_dgemm(NoTrans, Trans, ...)``), the ROWS of the matrix of GSL 2.7's
        eigenvectors, a rotation that is not the principal axes and whose directions depend on the signs GSL gives
        the eigenvectors (so ``gsl_eigen_symmv`` is ported literally). On the synthetic data of the comparison in
        the docs neither is better throughout: FORCE's was up to 2.8 times better in one year of a CF field and
        somewhat worse in others, and with float means instead of FORCE's integers (other eigenvector signs) the
        same projection lost what it had gained in one year and gained in another.
    nodata
        Values that mean missing besides NaN: ``"auto"`` (the raster's NoData of each input; 0 for an integer `fine`
        without one, as ``tasseled_cap``), a number for both, or ``None``.
    chunks
        For inputs read from disk (``zeit.load_raster``): ``None`` in memory (dask inputs are loaded in any case).
    n_jobs
        Threads (``-1``: the processors zeit may use by default).

    Returns
    -------
    xarray.DataArray
        float32, the layers of `coarse` on the grid and coordinates of `fine`: ``(year, [field], y, x)`` with
        ``by="year"``, ``(time, [band], y, x)`` with ``by=None``. ``attrs["improphe"]`` says the variant, the
        radius and the texture radius in pixels, ``min_neighbors``, the number of components (per year, or one for
        the series), the seasons and ``force``. A year (CF) or the whole series (L2) with no fine
        data gives the coarse values themselves (FORCE copies them too, but then stops for the other years; here the
        others are still computed).

    Notes
    -----
    The whole extent is one block: the components' statistics come from all the pixels, as FORCE's from a chunk
    with its margin of the neighbouring tiles, and the circle reaches `kernel_size` beyond any block; the cube, the
    coarse field and the result are in memory (a dask input is computed first, a tile at a time is the caller's).
    Pixels within `kernel_size` of the edge see fewer neighbours. FORCE's resmerge (Sentinel-2 20 m to 10 m) and
    STARFM are not part of this.

    Examples
    --------
    >>> sharp = zeit.improphe(lsp_500m, sentinel2, by="year", kernel_size=500, texture_size=60)   # CF, 10 m pixels
    >>> l8_10m = zeit.improphe(landsat, sentinel2, by=None, kernel_size=300, texture_size=60)       # L2
    """
    from . import _core
    from ._embeddings import refuse
    from ._load import load_raster
    from ._threads import resolve

    refuse(coarse, "zeit.improphe")
    refuse(fine, "zeit.improphe")
    fine = fine if isinstance(fine, xr.DataArray) else load_raster(fine, chunks=chunks)
    coarse = coarse if isinstance(coarse, xr.DataArray) else load_raster(coarse, chunks=chunks)
    if not {"time", "band", "y", "x"} <= set(fine.dims) or fine.ndim != 4:
        raise ValueError(f"fine must be (time, band, y, x), got {fine.dims}")
    if pca is None:
        pca = "force" if force else "principal"
    if pca not in ("force", "principal"):
        raise ValueError(f"pca must be 'force' or 'principal', got {pca!r}")
    force_pca = pca == "force"
    if not (0.0 < pc_variance <= 1.0):
        raise ValueError(f"pc_variance must be in (0, 1], got {pc_variance}")
    dwin = [int(s) for s in seasons]
    if len(dwin) < 2 or any(s < 1 or s > 365 for s in dwin) or [int(s) for s in seasons] != list(seasons):
        raise ValueError("seasons must be at least two integer days of the year in 1..365")
    if "time" not in fine.coords or not np.issubdtype(fine["time"].dtype, np.datetime64):
        raise ValueError("fine needs a datetime 'time' coordinate: the seasonal windows are days of the year")
    times = pd.DatetimeIndex(fine["time"].values)
    if times.isna().any():
        raise ValueError("fine has a missing date in its 'time' coordinate")
    res = _resolution(fine)
    h = _radius(kernel_size, res, "kernel_size", 1)
    ksd = _radius(texture_size, res, "texture_size", 0)
    mode, layer, layer_values, field = _layout(coarse, by)

    # fine to float32 with NaN for missing, (time, band, y, x)
    x4 = _float32(fine.transpose("time", "band", "y", "x").values, _missing(fine, nodata, True))
    doy = (_CUM[times.month - 1] + times.day).astype(np.int32).tolist()
    years = times.year.astype(np.int32).tolist()

    # coarse to (layer, field, y, x) float32 on fine's grid
    order = (layer,) + ((field,) if field else ()) + ("y", "x")
    raw = _float32(coarse.transpose(*order).values, _missing(coarse, nodata, False))
    n_layer = raw.shape[0]
    n_field = raw.shape[1] if field else 1
    raw = raw.reshape(n_layer * n_field, coarse.sizes["y"], coarse.sizes["x"])
    mr = _on_grid(coarse, fine, raw, res).reshape(n_layer, n_field, fine.sizes["y"], fine.sizes["x"])
    mr = np.ascontiguousarray(mr, dtype=np.float32)

    threads = resolve(n_jobs)
    out = np.empty_like(mr)
    components = []
    min_var = float(pc_variance)
    if mode == "year":
        for i, year in enumerate(layer_values.tolist()):
            prep = _core.improphe.prepare(x4, doy, years, dwin, int(year), h, ksd, min_var, bool(force), threads,
                                          force_pca)
            if prep is None:
                out[i] = mr[i]
                components.append(0)
                continue
            pcs, tex = prep[0], prep[1]
            out[i] = _core.improphe.predict(pcs, tex, mr[i], h, ksd, threads)
            components.append(int(pcs.shape[0]))
    else:
        prep = _core.improphe.prepare(x4, doy, years, dwin, None, h, ksd, min_var, bool(force), threads, force_pca)
        if prep is None:
            out[...] = mr
            components.append(0)
        else:
            pcs, tex = prep[0], prep[1]
            for i in range(n_layer):
                out[i] = _core.improphe.predict(pcs, tex, mr[i], h, ksd, threads)
            components.append(int(pcs.shape[0]))
    del x4

    out_layer = "year" if mode == "year" else "time"
    dims = (out_layer,) + (((field,) if field else ())) + ("y", "x")
    coords = {k: v for k, v in fine.coords.items() if set(v.dims) <= {"y", "x"}}
    coords[out_layer] = layer_values
    if field and field in coarse.coords:
        coords[field] = coarse[field].values
    result = xr.DataArray(out if field else out[:, 0], dims=dims, coords=coords, name=coarse.name)
    result.attrs = dict(coarse.attrs)
    ncomp = (", ".join(f"{y}: {c}" for y, c in zip(layer_values.tolist(), components)) if mode == "year"
             else str(components[0]))
    result.attrs["improphe"] = (
        f"{'CF-ImproPhe' if mode == 'year' else 'L2-ImproPhe'}; radius {h} px, texture radius {ksd} px, "
        f"min_neighbors {int(_core.improphe.min_neighbors(h))}; components ({pc_variance} of the variance) {ncomp}; "
        f"seasons {' '.join(map(str, dwin))}; pca={pca}, force={bool(force)}")
    if fine.rio.crs is not None:
        result = result.rio.write_crs(fine.rio.crs)
    return result
