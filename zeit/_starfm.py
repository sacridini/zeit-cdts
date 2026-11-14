"""STARFM spatiotemporal fusion: ``zeit.starfm``.

The Spatial and Temporal Adaptive Reflectance Fusion Model (Gao, Masek, Schwaller and Hall 2006, IEEE TGRS
44:2207-2218) as FORCE 3.11's streamlined ``starfm()`` has it (``resmerge-ll.c``, David Frantz, GPL-3.0-or-later; its
STARFM part is adopted from the USDA Agricultural Research Service's version, Feng Gao, 2005-2014), for any number of
base pairs, prediction dates and bands. Computed by the compiled core (``zeit._core.resmerge.starfm_batch``).
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import xarray as xr


def _check_cube(name: str, cube: Any) -> xr.DataArray:
    if not isinstance(cube, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray (time, [band], y, x), got {type(cube).__name__}")
    dims = set(cube.dims)
    if not {"time", "y", "x"} <= dims or not dims <= {"time", "band", "y", "x"}:
        raise ValueError(f"{name} must have the dimensions (time, [band], y, x), got {tuple(cube.dims)}")
    return cube


def starfm(
    fine: xr.DataArray,
    coarse: xr.DataArray,
    *,
    radius: int,
    target: Optional[Sequence[Any]] = None,
    nclass: float = 40.0,
    unc_fine: float = 0.002,
    unc_coarse: float = 0.002,
    scale: Union[str, float] = "auto",
    nodata: Union[float, str, None] = "auto",
    n_jobs: int = -1,
) -> xr.DataArray:
    """STARFM: fine-resolution images at the dates where only a coarse product exists.

    STARFM (Gao et al. 2006) predicts the fine image (Landsat, Sentinel-2) of a date for which only the coarse one
    (MODIS, VIIRS, or any daily product) is available, from the pairs of fine and coarse images of dates where both
    are. For each pixel it looks in the window of ``(2 radius + 1)`` pixels around it, in each pair, for the pixels
    that are spectrally similar to it in the fine image (the absolute difference to the pixel at most
    ``2 std / nclass``, ``std`` the standard deviation of that fine image) and whose coarse values changed little
    between the pair and the prediction date, or are close to their fine values, and takes the weighted mean of
    ``fine + (coarse at the date - coarse at the pair)``, the weights the inverse of the temporal difference, the
    spectral difference and the distance. Where there is no such pixel, the coarse value is kept.

    This is FORCE 3.11's ``starfm()`` (``resmerge-ll.c``; David Frantz, GPL-3.0-or-later), which is adopted from the
    USDA-ARS version (Feng Gao, 2005-2014; STARFM is released for further development). FORCE uses it for the
    Sentinel-2 20 m to 10 m merge, with two pairs, a radius of 1 pixel, 40 classes and uncertainties of 0.002; here
    it takes any number of pairs, any radius and any of those parameters, with FORCE's arithmetic: with the same
    parameters the values equal those of FORCE's C function (see the benchmarks).

    Requirements of the inputs. STARFM works pixel by pixel, so ``coarse`` must be on the grid of ``fine``: the
    coarse product resampled to the fine pixels (``zeit.load_raster(path, like=fine)``, or ``rioxarray``'s
    ``reproject_match``), with the same bands in the same order and the same unit and scale (reflectance times
    1 or times 10000). A grid, shape or band mismatch raises a ``ValueError``. The thresholds below (``unc_*``) are
    in reflectance, 0 to 1.

    Pairs and targets. The pairs are the dates present in both ``fine`` and ``coarse`` (an exact match on
    ``time``); a date of ``fine`` that ``coarse`` does not have is not used. ``target`` are the dates to predict:
    by default every date of ``coarse`` that is not a pair. Every target is predicted from **all** the pairs at
    once (the candidates of every pair compete in the weighted mean; FORCE's own code has two). Each band is
    independent of the others, as in FORCE.

    Missing data. A pixel with NaN (or ``nodata``) in any of the pairs' fine or coarse image, or in the coarse image
    of the target, in a band, is "off" for that band and target, as FORCE's off pixels: not a candidate and not
    predicted (NaN in the result). The standard deviation that sets the similarity threshold is that of the pixels
    that are not off.

    Parameters
    ----------
    fine
        Fine-resolution images ``(time, y, x)`` or ``(time, band, y, x)`` at the pair dates (one or more). Held in
        memory (a lazy cube is computed).
    coarse
        The coarse product on the same grid, ``(time, y, x)`` or ``(time, band, y, x)``, covering the pair dates
        and the dates to predict.
    radius
        Half-width of the window in pixels (the window is ``2 radius + 1`` pixels wide); required. FORCE's own value
        is 1, right for 20 m to 10 m and not for MODIS (500 m) to Landsat (30 m), where the window has to span
        several coarse pixels (one 500 m pixel is about 17 Landsat pixels across). There is no default: the window
        of Gao et al. (2006) depends on the pair of sensors, and Zeit does not guess it.
    target
        The dates to predict (values of ``coarse.time``); default: the dates of ``coarse`` that are not pairs.
    nclass
        Number of spectral classes; the similarity threshold is ``2 std / nclass``. FORCE's 40.
    unc_fine, unc_coarse
        Uncertainties of the fine and coarse images, reflectance. FORCE's 0.002 (the ARS value). Pixels whose
        weight product is under the combined uncertainty get the largest weight.
    scale
        What the reflectance is multiplied by in the data: ``"auto"`` (10000 for integers or values above 2, else
        1; ``fine`` and ``coarse`` must agree) or a number. The result is in the input's scale.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or
        ``None``.
    n_jobs
        Threads: the processors zeit may use by default (``-1``). The result does not depend on it.

    Returns
    -------
    xarray.DataArray
        float32 ``(time, [band], y, x)``, with ``time`` the predicted dates, and the coordinates and CRS of
        ``fine``; NaN where there is no prediction.

    Examples
    --------
    >>> landsat = zeit.load_raster("landsat_pairs.tif", dates=[...])         # (time, band, y, x), 30 m
    >>> modis = zeit.load_raster("modis_daily.tif", dates=[...], like=landsat)  # resampled to the Landsat grid
    >>> daily = zeit.starfm(landsat, modis, radius=25)                       # every MODIS date that is not a pair
    """
    from ._bandpass import _scale_of
    from ._series_api import _missing_as_nan
    from . import _core
    from ._threads import resolve

    fine = _check_cube("fine", fine)
    coarse = _check_cube("coarse", coarse)
    if not float(radius).is_integer() or radius < 0:
        raise ValueError(f"radius must be a non-negative integer number of pixels, got {radius!r}")
    if not nclass > 0:
        raise ValueError(f"nclass must be positive, got {nclass!r}")
    if unc_fine < 0 or unc_coarse < 0:
        raise ValueError("unc_fine and unc_coarse must not be negative")

    hint = "; resample the coarse product to the fine grid (zeit.load_raster(path, like=fine))"
    if ("band" in fine.dims) != ("band" in coarse.dims):
        raise ValueError("fine and coarse must both have a band dimension or neither" + hint)
    for d in ("y", "x"):
        if fine.sizes[d] != coarse.sizes[d]:
            raise ValueError(f"coarse has {coarse.sizes[d]} {d} values and fine {fine.sizes[d]}: the coarse "
                             f"product must be on the fine grid" + hint)
        if d in fine.coords and d in coarse.coords:
            a, b = np.asarray(fine[d].values, float), np.asarray(coarse[d].values, float)
            step = np.abs(np.diff(a)).mean() if a.size > 1 else 1.0
            if not np.allclose(a, b, rtol=0, atol=1e-6 * max(step, 1e-12)):
                raise ValueError(f"the {d} coordinates of coarse and fine differ: the coarse product must be on "
                                 f"the fine grid" + hint)
    has_band = "band" in fine.dims
    if has_band:
        fb, cb = list(fine["band"].values), list(coarse["band"].values)
        if fb != cb:
            raise ValueError(f"fine has the bands {fb} and coarse {cb}: they must be the same, in the same order")
    order = ("time", "band", "y", "x") if has_band else ("time", "y", "x")

    ftime, ctime = np.asarray(fine["time"].values), np.asarray(coarse["time"].values)
    in_coarse = np.isin(ftime, ctime)
    if not in_coarse.any():
        raise ValueError("no date of fine is in coarse: STARFM needs at least one pair (fine and coarse images of "
                         "the same date, matched exactly on time)")
    pair_t = ftime[in_coarse]
    if target is None:
        tmask = ~np.isin(ctime, pair_t)
        tdates = ctime[tmask]
        if tdates.size == 0:
            raise ValueError("every date of coarse is a pair: there is nothing to predict (pass target=)")
    else:
        tdates = np.atleast_1d(np.asarray(target, dtype=ctime.dtype))
        missing = tdates[~np.isin(tdates, ctime)]
        if missing.size:
            raise ValueError(f"target dates {list(missing)} are not in coarse.time")
    factor = _scale_of(fine, scale)
    if scale == "auto" and _scale_of(coarse, scale) != factor:
        raise ValueError("fine and coarse seem to be on different scales (reflectance 0-1 and times 10000); "
                         "convert one or pass scale=")

    pairs_f = _missing_as_nan(fine.isel(time=np.flatnonzero(in_coarse)), nodata).transpose(*order)
    tsel = [int(np.flatnonzero(ctime == t)[0]) for t in pair_t] + [int(np.flatnonzero(ctime == t)[0]) for t in tdates]
    cube_c = _missing_as_nan(coarse.isel(time=tsel), nodata).transpose(*order)
    fa = np.asarray(pairs_f.values, dtype=np.float64)
    ca = np.asarray(cube_c.values, dtype=np.float64)
    if factor != 1.0:
        fa, ca = fa / factor, ca / factor
    fa = fa if has_band else fa[:, None]
    ca = ca if has_band else ca[:, None]
    fa = np.ascontiguousarray(fa, dtype=np.float32)
    ca = np.ascontiguousarray(ca, dtype=np.float32)

    pred = _core.resmerge.starfm_batch(fa, ca, int(radius), float(nclass), float(unc_coarse), float(unc_fine),
                                       resolve(n_jobs))
    if factor != 1.0:
        pred = (pred.astype(np.float64) * factor).astype(np.float32)
    if not has_band:
        pred = pred[:, 0]

    coords = {k: v for k, v in fine.coords.items() if "time" not in v.dims}
    coords["time"] = ("time", ctime[[int(np.flatnonzero(ctime == t)[0]) for t in tdates]])
    out = xr.DataArray(pred, dims=order, coords=coords, name=fine.name).transpose(
        *[d for d in fine.dims if d in order])
    out.attrs = dict(fine.attrs)
    out.attrs["starfm"] = (
        f"STARFM (Gao et al. 2006) from {len(pair_t)} pair(s): radius {int(radius)}, nclass {nclass:g}, uncertainty "
        f"fine {unc_fine:g} coarse {unc_coarse:g}; FORCE 3.11's starfm(), resmerge-ll.c (David Frantz, "
        f"GPL-3.0-or-later), adopted from the USDA-ARS version (Feng Gao)")
    if fine.rio.crs is not None:
        out = out.rio.write_crs(fine.rio.crs)
    if fine.rio.nodata is not None and not np.issubdtype(fine.dtype, np.integer):
        out = out.rio.write_nodata(fine.rio.nodata, encoded=False)
    return out
