"""Sentinel-2 20 m to 10 m resolution merge: ``zeit.sentinel2_resmerge``.

FORCE's resolution merge of the Level 2 processing system (``resmerge-ll.c``, FORCE 3.11, David Frantz,
GPL-3.0-or-later): the six 20 m bands (red edge 1-3, B8A, SWIR1, SWIR2) predicted at 10 m from the 10 m green, red
and broad NIR, by a moving-window regression (``method="regression"``) or by the ImproPhe weighted mean
(``method="improphe"``). Computed by the compiled core (``zeit._core.resmerge``).
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import xarray as xr

from ._specadjust import TARGET_BANDS, _TARGET_NAMES, _find

#: The bands the 10 m ones predict, and the 10 m ones that predict.
PREDICTORS = ("green", "red", "nir")
PREDICTED = ("rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22")
METHODS = ("regression", "improphe")


def _clear_array(clear: Any, cube: xr.DataArray, order: tuple) -> Optional[np.ndarray]:
    """`clear` as a boolean array of the cube's shape without the bands, in `order` (time, y, x), or None."""
    if clear is None:
        return None
    ref = cube.isel(band=0, drop=True)
    if not isinstance(clear, xr.DataArray):
        values = np.asarray(clear, dtype=bool)
        try:
            return np.broadcast_to(values, ref.transpose(*order).shape)
        except ValueError:
            raise ValueError(f"clear has shape {values.shape}, which does not broadcast to the cube's "
                             f"{ref.transpose(*order).shape} (time, y, x)") from None
    extra = [d for d in clear.dims if d not in ref.dims]
    if extra:
        raise ValueError(f"clear has dimensions {extra} that the cube does not have")
    for d in clear.dims:
        if clear.sizes[d] != ref.sizes[d]:
            raise ValueError(f"clear has {clear.sizes[d]} values along {d!r}, the cube {ref.sizes[d]}")
    missing = [d for d in order if d not in clear.dims]
    c = clear.expand_dims(missing) if missing else clear
    return np.broadcast_to(np.asarray(c.transpose(*order).values, dtype=bool), ref.transpose(*order).shape)


def sentinel2_resmerge(
    cube: xr.DataArray,
    *,
    method: str = "regression",
    clear: Any = None,
    force: bool = False,
    bands: Optional[Dict[str, Any]] = None,
    scale: Union[str, float] = "auto",
    nodata: Union[float, str, None] = "auto",
    n_jobs: int = -1,
) -> xr.DataArray:
    """The 20 m bands of Sentinel-2 at 10 m: FORCE's resolution merge.

    Sentinel-2 has ten bands at 10 m or 20 m (the red edge B05-B07, B8A and the SWIR B11 and B12 at 20 m). On the
    10 m grid each 20 m pixel is four equal ones: FORCE's Level 2 processing reads the 20 m bands into the 10 m
    buffer with GDAL's ``RasterIO``, whose resampling is the nearest neighbour, and this function takes a cube
    that is already so (``zeit.build_time_series`` with the 10 m grid, or a resampling of the 20 m bands by
    nearest neighbour). The merge then predicts each pixel's six 20 m bands from its 10 m green (B03), red (B04)
    and broad NIR (B08), by one of two methods.

    ``"regression"`` (FORCE's ``RES_MERGE = REGRESSION``) fits, for each pixel and each 20 m band, the least
    squares line ``y = c0 + c1 green + c2 red + c3 NIR`` to the 5 x 5 pixels around it (GSL's
    ``gsl_multifit_linear``: the SVD of the column-balanced matrix, singular values at or below ``epsilon`` times
    the largest dropped) and evaluates it at the pixel. The pixel is in its own window; pixels with no data are
    left out of it and a window of fewer than 25 pixels is filled with rows ``(1, 0, 0, 0)`` with ``y = 0`` (as
    FORCE does, which pulls the fit down at the borders and next to gaps); clouds and shadows are kept in it.

    ``"improphe"`` (``RES_MERGE = IMPROPHE``) is ImproPhe's weighted mean in a spectral instead of a temporal
    setup: the pixels of the disc of radius 2 that are not cloud, shadow or no data and whose mean absolute
    difference of green, red and NIR (reflectance) to the pixel is under one of 0.0025, 0.005, 0.01, 0.02, 0.025
    are the neighbours; the smallest of those cutoffs with at least 3 neighbours is used (the last if none), and
    each band is the mean of the neighbours weighted by ``1 / (1 + exp(25 x - 7.5))``, ``x`` the rescaled
    difference. Faster than the regression.

    Pixels with a missing value (NaN or ``nodata``) in any of the nine bands are FORCE's "off": left out of the
    windows and not predicted. Pixels that are not ``clear`` are not predicted either (the regression still reads
    them as neighbours, ImproPhe does not). Pixels not predicted keep their values.

    Differences from FORCE. FORCE's regression writes each prediction into the band before the next pixel's
    window reads it (row-major with one thread, racy with several): with ``force=False`` every window reads the
    original values and the pixels run in parallel; the output is float32 and not truncated. ``force=True``
    is FORCE's arithmetic, truncated integers (``(short)(v * 10000)``) and, for ``"regression"``, the in-place
    row-major sweep on one thread; its values equal those of FORCE's code run on the same scene (see
    ``benchmarks/vs_originals/force_resmerge.py``).

    Parameters
    ----------
    cube
        Reflectance ``(time, band, y, x)`` or ``(band, y, x)`` (each date on its own), all bands on the 10 m
        grid, with the Sentinel-2 bands under Earth Search's names (``green red nir rededge1 rededge2
        rededge3 nir08 swir16 swir22``; ``nir`` is B08 and ``nir08`` B8A) or under ``B03``... ``B12``. Other
        bands (blue...) pass through unchanged. Held in memory (a lazy cube is computed).
    method
        ``"regression"`` (default) or ``"improphe"``.
    clear
        Optional boolean array broadcastable to ``(time, y, x)``: True where the pixel is not cloud or shadow.
        By default every pixel with data is.
    force
        FORCE's arithmetic (see above). The values are taken as reflectance times 10000 (data in another
        ``scale`` are converted and rounded) and the result is float32 holding integers there.
    bands
        The cube's band of each of the nine roles above, when its names are not the usual ones.
    scale
        What the reflectance is multiplied by in the data: ``"auto"`` (10000 for integers or values above 2,
        else 1), or a number. The distance thresholds of ``"improphe"`` are in reflectance and the regression's
        coefficients are not scale-free, so a cube in 0-1 and one in 0-10000 give the same.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or
        ``None``.
    n_jobs
        Threads: the processors zeit may use by default (``-1``). ``force=True`` with ``"regression"`` runs on one.

    Returns
    -------
    xarray.DataArray
        As ``cube`` (float32), with the six bands predicted.

    Examples
    --------
    >>> s2 = zeit.build_time_series(collection="sentinel-2-l2a", resolution=10, ...)
    >>> sharp = zeit.sentinel2_resmerge(s2, method="regression", clear=cloud_free)
    """
    from ._bandpass import _scale_of
    from ._series_api import _missing_as_nan
    from . import _core
    from ._threads import resolve

    if not isinstance(cube, xr.DataArray):
        raise TypeError("sentinel2_resmerge needs an xarray.DataArray (time, band, y, x) or (band, y, x)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    if "band" not in cube.dims or not {"y", "x"} <= set(cube.dims):
        raise ValueError("sentinel2_resmerge needs a cube with band, y and x dimensions")
    roles = PREDICTORS + PREDICTED
    names = {r: _TARGET_NAMES[r] for r in roles}
    unknown = [k for k in (bands or {}) if k not in roles and k not in TARGET_BANDS]
    if unknown:
        raise ValueError(f"bands=: unknown role {unknown}; roles are {list(roles)}")
    available = [str(b) for b in cube.band.values]
    try:
        found = _find(cube, bands, names, roles, "the Sentinel-2 resolution merge")
    except ValueError:
        lower = [b.lower() for b in available]
        lacking = [r for r in roles if not any(n.lower() in lower for n in names[r])
                   and str((bands or {}).get(r)) not in available]
        raise ValueError(f"the cube {available} has no band for the resolution merge: {lacking or list(roles)} "
                         f"(green, red and nir are the 10 m bands that predict; rededge1-3, nir08, swir16 and "
                         f"swir22 are the 20 m bands predicted); pass bands= (role -> band)") from None
    pred_idx = [found[r] for r in PREDICTORS]
    targ_idx = [found[r] for r in PREDICTED]
    factor = _scale_of(cube, scale)

    has_time = "time" in cube.dims
    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = _missing_as_nan(cube, nodata).transpose(*order)
    arr = np.asarray(work.values, dtype=np.float32)
    if not has_time:
        arr = arr[None]
    mask_order = (("time",) if has_time else ()) + ("y", "x")
    clear_arr = _clear_array(clear, cube, mask_order)
    if clear_arr is not None and not has_time:
        clear_arr = clear_arr[None]
    threads = resolve(n_jobs)

    if force:
        dn = 10000.0 / factor        # the data in FORCE's units
        denom = 10000.0
        if dn != 1.0:
            arr = np.rint(arr.astype(np.float64) * dn).astype(np.float32)
    else:
        dn = 1.0
        denom = factor

    out = arr.copy()
    for k in range(arr.shape[0]):
        x = arr[k]
        green, red, nir = (np.ascontiguousarray(x[i]) for i in pred_idx)
        targets = np.ascontiguousarray(x[targ_idx])
        off = ~(np.isfinite(green) & np.isfinite(red) & np.isfinite(nir) & np.isfinite(targets).all(axis=0))
        bad = np.zeros_like(off) if clear_arr is None else ~np.ascontiguousarray(clear_arr[k])
        pred = _core.resmerge.resmerge_batch(green, red, nir, targets, off, bad, method, float(denom), bool(force),
                                             threads)
        out[k][targ_idx] = pred
    if force and dn != 1.0:
        out = (out.astype(np.float64) / dn).astype(np.float32)
    if not has_time:
        out = out[0]

    result = xr.DataArray(out, dims=order, coords=work.coords, name=cube.name).transpose(*cube.dims)
    result.attrs = dict(cube.attrs)
    result.attrs["sentinel2_resmerge"] = (
        f"{method}{' (FORCE arithmetic)' if force else ''}: {', '.join(PREDICTED)} predicted at 10 m from "
        f"green, red and nir; FORCE's resolution merge, resmerge-ll.c (David Frantz, GPL-3.0-or-later)")
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    if cube.rio.nodata is not None and not np.issubdtype(cube.dtype, np.integer):
        result = result.rio.write_nodata(cube.rio.nodata, encoded=False)
    return result
