"""Sentinel-2 parallax cloud test and cloud buffer: ``zeit.cloud_displacement_index``, ``zeit.parallax_clouds`` and
``zeit.buffer_mask``.

FORCE's cloud detection for Sentinel-2 (``cloud-ll.c``, FORCE 3.11, David Frantz, GPL-3.0-or-later): the parallax
step that separates the potential cloud pixels into clouds and bright surfaces by the Cloud Displacement Index
(Frantz, Haß, Uhl, Stoffels and Hill 2018, Remote Sensing of Environment 215:471-481), and FORCE's buffer of a binary
mask (``imagefuns-cl.c``, used for the cloud buffer). Computed by the compiled core (``zeit._core.parallax``).
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import xarray as xr

from ._specadjust import TARGET_BANDS, _TARGET_NAMES, _find

#: The bands of the parallax test: red edge 3 (B07), broad NIR (B08) and NIR (B8A).
PARALLAX_ROLES = ("rededge3", "nir", "nir08")
#: The bands of FORCE's bright pixels (the mean of blue, green and red above 0.15).
VISIBLE_ROLES = ("blue", "green", "red")
#: The bands the variability probability reads besides the visible ones: NIR (B8A) and SWIR1 (B11).
VARIABILITY_ROLES = ("nir08", "swir16")


def _broadcast(values: Any, cube: xr.DataArray, order: tuple, name: str, dtype: Any) -> np.ndarray:
    """`values` as an array of `dtype` of the cube's shape without the bands, in `order` (time, y, x)."""
    ref = cube.isel(band=0, drop=True)
    target = ref.transpose(*order).shape
    if not isinstance(values, xr.DataArray):
        arr = np.asarray(values, dtype=dtype)
        try:
            return np.broadcast_to(arr, target)
        except ValueError:
            raise ValueError(f"{name} has shape {arr.shape}, which does not broadcast to the cube's {target} "
                             f"(time, y, x)") from None
    extra = [d for d in values.dims if d not in ref.dims]
    if extra:
        raise ValueError(f"{name} has dimensions {extra} that the cube does not have")
    for d in values.dims:
        if values.sizes[d] != ref.sizes[d]:
            raise ValueError(f"{name} has {values.sizes[d]} values along {d!r}, the cube {ref.sizes[d]}")
    missing = [d for d in order if d not in values.dims]
    v = values.expand_dims(missing) if missing else values
    return np.broadcast_to(np.asarray(v.transpose(*order).values, dtype=dtype), target)


def _variability(vis: np.ndarray, nir: np.ndarray, swir: np.ndarray, saturated: np.ndarray) -> np.ndarray:
    """Zhu and Woodcock's (2012) variability probability, ``1 - max(|NDVI|, |NDSI|, whiteness)`` in [0, 1].

    `vis` is ``(3, y, x)`` blue, green, red in reflectance. NDVI is ``(NIR - red) / (NIR + red)``, NDSI is
    ``(green - SWIR1) / (green + SWIR1)`` and the whiteness the sum over the visible bands of the distance to
    their mean, relative to it. Where the visible bands are saturated their values are too low: a positive NDVI
    and a negative NDSI are then set to 0 (the paper's adjustment, which it gave for Landsat 7's saturated
    visible bands); `saturated` says where. A zero denominator gives an index of 0.
    """
    blue, green, red = (v.astype(np.float32) for v in vis)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(nir + red != 0, (nir - red) / (nir + red), 0).astype(np.float32)
        ndsi = np.where(green + swir != 0, (green - swir) / (green + swir), 0).astype(np.float32)
        mean = (blue + green + red) / np.float32(3.0)
        white = np.where(mean != 0, (np.abs(blue - mean) + np.abs(green - mean) + np.abs(red - mean)) / mean, 0)
    ndvi = np.where(saturated & (ndvi > 0), 0, np.abs(ndvi))
    ndsi = np.where(saturated & (ndsi < 0), 0, np.abs(ndsi))
    return np.clip(1 - np.maximum(np.maximum(ndvi, ndsi), white), 0, 1).astype(np.float32)


def _prepare(cube, potential, variability, saturated, bright, bands, scale, nodata, need_var, need_bright):
    """The per-date inputs of the core: ``(work, has_time, order, mask_order, ints, off, pcp, var, bright,
    saturated)``, `ints` the three parallax bands as int16 ``(time, 3, y, x)`` (FORCE's units)."""
    from ._bandpass import _scale_of
    from ._series_api import _missing_as_nan

    if not isinstance(cube, xr.DataArray):
        raise TypeError("the parallax test needs an xarray.DataArray (time, band, y, x) or (band, y, x)")
    if "band" not in cube.dims or not {"y", "x"} <= set(cube.dims):
        raise ValueError("the parallax test needs a cube with band, y and x dimensions")
    unknown = [k for k in (bands or {}) if k not in TARGET_BANDS]
    if unknown:
        raise ValueError(f"bands=: unknown role {unknown}; roles are {list(TARGET_BANDS)}")
    if cube.sizes["y"] < 2 or cube.sizes["x"] < 2:
        raise ValueError("the parallax test needs at least 2 x 2 pixels")
    available = [str(b) for b in cube.band.values]
    roles = list(PARALLAX_ROLES)
    if need_bright:
        roles += [r for r in VISIBLE_ROLES if r not in roles]
    if need_var:
        roles += [r for r in VISIBLE_ROLES + VARIABILITY_ROLES if r not in roles]
    try:
        found = _find(cube, bands, _TARGET_NAMES, roles, "the Sentinel-2 parallax cloud test")
    except ValueError as err:
        if str(err).startswith("bands=:"):
            raise
        lower = [b.lower() for b in available]
        lacking = [r for r in roles if not any(n.lower() in lower for n in _TARGET_NAMES[r])
                   and str((bands or {}).get(r)) not in available]
        hints = []
        if need_bright:
            hints.append("pass bright= (a boolean mask) to go without blue, green and red")
        if need_var:
            hints.append("pass variability= to go without them for the variability probability")
        raise ValueError(f"the cube {available} has no band for the parallax test: {lacking or roles} "
                         f"(rededge3 = B07, nir = B08 and nir08 = B8A are the bands of the test"
                         f"{'; ' + '; '.join(hints) if hints else ''}); pass bands= (role -> band)") from None
    factor = _scale_of(cube, scale)
    has_time = "time" in cube.dims
    order = (("time",) if has_time else ()) + ("band", "y", "x")
    mask_order = (("time",) if has_time else ()) + ("y", "x")
    work = _missing_as_nan(cube, nodata).transpose(*order)

    def take(rs):
        a = np.asarray(work.isel(band=[found[r] for r in rs]).values, dtype=np.float32)
        return a if has_time else a[None]

    core = take(PARALLAX_ROLES)                                  # (time, 3, y, x)
    off = ~np.isfinite(core).all(axis=1)

    def dn(a):          # FORCE's units: reflectance times 10000, as int16 (non-finite -> 0)
        if factor == 10000.0:
            v = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            v = np.nan_to_num(a.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0) * (10000.0 / factor)
        np.rint(v, out=v)
        np.clip(v, -32768, 32767, out=v)
        return v.astype(np.int16)

    ints = dn(core)

    def mask(v, name, dtype=bool):
        m = _broadcast(v, cube, mask_order, name, dtype)
        return m if has_time else m[None]

    pcp = mask(potential, "potential")
    sat = mask(saturated, "saturated") if saturated is not None else np.zeros(off.shape, dtype=bool)
    if bright is not None:
        brt = mask(bright, "bright")
    elif need_bright:
        visible = take(VISIBLE_ROLES)                                     # (time, 3, y, x)
        brt = np.isfinite(visible).all(axis=1) & (dn(visible).astype(np.int32).sum(axis=1) > 4500)  # mean > 1500
    else:
        brt = np.ones(off.shape, dtype=bool)
    if variability is not None:
        var = mask(variability, "variability", np.float32)
    else:
        var = None
    if variability is None and need_var:
        vis = take(VISIBLE_ROLES) / np.float32(factor)
        nir8, swir = (take([r])[:, 0] / np.float32(factor) for r in VARIABILITY_ROLES)
        var = np.stack([_variability(vis[k], nir8[k], swir[k], sat[k]) for k in range(vis.shape[0])])
    if var is not None:
        var = np.clip(np.rint(np.nan_to_num(var.astype(np.float64)) * 10000.0), -32768, 32767).astype(np.int16)
    else:
        var = np.zeros(off.shape, dtype=np.int16)
    return work, has_time, order, mask_order, ints, off, np.ascontiguousarray(pcp), var, brt, sat


def _coords_20m(work: xr.DataArray) -> Dict[str, Any]:
    """Coordinates of the 20 m grid: the centre of each block of 2 x 2 pixels of the 10 m one."""
    out = {}
    for d in ("y", "x"):
        n = work.sizes[d] // 2
        if d in work.coords and np.issubdtype(work[d].dtype, np.number):
            out[d] = np.asarray(work[d].values[:2 * n], dtype=np.float64).reshape(n, 2).mean(axis=1)
    return out


def _result(work, arr, dims_tail, extra_coords, name, attrs):
    coords = {c: v for c, v in work.coords.items() if set(v.dims) <= {"time"}}
    coords.update(extra_coords)
    result = xr.DataArray(arr, dims=dims_tail, coords=coords, name=name)
    result.attrs = attrs
    return result


def cloud_displacement_index(
    cube: xr.DataArray,
    *,
    potential: Any = None,
    bands: Optional[Dict[str, Any]] = None,
    scale: Union[str, float] = "auto",
    nodata: Union[float, str, None] = "auto",
    n_jobs: int = -1,
) -> xr.DataArray:
    """The Cloud Displacement Index of Sentinel-2: the parallax signature of high clouds.

    The detectors of the 10 m bands look at the ground one after the other, so a high object is seen at slightly
    different places in different bands, while a surface object is at the same place in all of them. The red
    edge 3 (B07) and the NIR (B8A) are taken close in time, the broad NIR (B08) is not: the ratio ``B07 / B8A``
    varies little at a cloud and the ratio ``B08 / B8A`` (with B08 smoothed by a 5 x 5 Gaussian kernel of
    sigma 0.5) much, a bright surface changes both alike. On the 20 m grid (each block is 2 x 2 pixels of the 10 m
    one, the means of the pixels that have data) the CDI is ``(var_x - var_y) / (var_x + var_y)``, the variances
    of those two ratios over the 7 x 7 blocks around: close to -1 at a cloud, around 0 or above for a surface.
    This is the CDI of FORCE's ``cloud_parallax`` (Frantz, Haß, Uhl, Stoffels and Hill 2018, Remote Sensing of
    Environment 215:471-481, FORCE 3.11, David Frantz, GPL-3.0-or-later), bit for bit.

    The CDI is given on the 20 m half-resolution grid it is computed on (FORCE does not bring it back to 10 m; the
    clouds of ``parallax_clouds`` are). A block has one only if it has a potential cloud pixel and, among the 7 x 7
    blocks around it, at least three that have one (the ratios of the others are left out of the variances); else
    it is NaN (FORCE's -9999). An odd last row or column of the cube is not part of any block.

    Parameters
    ----------
    cube
        Reflectance ``(time, band, y, x)`` or ``(band, y, x)`` on the 10 m grid with the bands rededge3 (B07), nir
        (B08) and nir08 (B8A) under Earth Search's names or ``B07``, ``B08``, ``B8A`` (20 m bands as four equal
        pixels, as for ``sentinel2_resmerge``). Held in memory.
    potential
        Boolean array broadcastable to ``(time, y, x)``: the potential cloud pixels (Fmask's, or any cloud
        product). By default every pixel with data is.
    bands
        The cube's band of each role (``rededge3``, ``nir``, ``nir08``), when its names are not the usual ones.
    scale
        What the reflectance is multiplied by in the data: ``"auto"`` (10000 for integers or values above 2, else
        1), or a number. The values are taken to FORCE's units (times 10000, as integers) for the computation.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or
        ``None``. A pixel with a missing value in one of the three bands is FORCE's "off".
    n_jobs
        Threads: the processors zeit may use by default (``-1``).

    Returns
    -------
    xarray.DataArray
        float32 ``(time, y // 2, x // 2)`` (or ``(y // 2, x // 2)``), the coordinates those of the 20 m grid.
    """
    from . import _core
    from ._threads import resolve

    pot = potential if potential is not None else np.ones((), dtype=bool)
    work, has_time, order, mask_order, ints, off, pcp, var, brt, sat = _prepare(
        cube, pot, None, None, None, bands, scale, nodata, need_var=False, need_bright=False)
    threads = resolve(n_jobs)
    out = []
    for k in range(ints.shape[0]):
        re3, bnir, nir = (np.ascontiguousarray(ints[k][i]) for i in range(3))
        _, cdi = _core.parallax.parallax_batch(re3, bnir, nir, np.ascontiguousarray(var[k]), np.ascontiguousarray(pcp[k]),
                                               np.ascontiguousarray(off[k]), np.ascontiguousarray(brt[k]),
                                               np.ascontiguousarray(sat[k]), False, threads)
        out.append(cdi)
    arr = np.stack(out)
    arr = np.where(arr == -9999.0, np.float32(np.nan), arr).astype(np.float32)
    if not has_time:
        arr = arr[0]
    attrs = dict(cube.attrs)
    attrs["cloud_displacement_index"] = ("on the 20 m grid; FORCE's cloud_parallax, cloud-ll.c (David Frantz, "
                                         "GPL-3.0-or-later); Frantz et al. 2018, RSE 215:471-481")
    return _result(work, arr, mask_order, _coords_20m(work), "cdi", attrs)


def parallax_clouds(
    cube: xr.DataArray,
    potential: Any,
    *,
    variability: Any = None,
    saturated: Any = None,
    bright: Any = None,
    bands: Optional[Dict[str, Any]] = None,
    scale: Union[str, float] = "auto",
    nodata: Union[float, str, None] = "auto",
    return_cdi: bool = False,
    n_jobs: int = -1,
) -> Any:
    """Sentinel-2 clouds from the potential cloud pixels by the parallax test: FORCE's ``cloud_parallax``.

    Of the potential cloud pixels, those that are clouds and not bright surfaces (snow, built-up areas, bare
    soil), by the Cloud Displacement Index (see ``cloud_displacement_index``; Frantz, Haß, Uhl, Stoffels and Hill
    2018, Remote Sensing of Environment 215:471-481). On the 20 m grid, the blocks with a CDI of -0.5 or less, or
    below 0 where the minimum of the variability probability over the 15 x 15 blocks around is above 0.8, are
    cloud seeds if so is every block of their 3 x 3 neighbourhood (a CDI and either of the two conditions; the
    seeds are eroded by a block); the clouds then grow into the 8-connected blocks with a CDI in [-1, -0.25). Each
    block is brought back to four 10 m pixels. As FORCE does at the end, saturated pixels are added and every
    cloud pixel that is not bright is removed.

    The equal of FORCE's code in every value (see ``benchmarks/vs_originals/force_parallax.py``): FORCE's
    ``short`` units (the reflectance times 10000 as int16), its ``float`` and ``double``, the order of its sums.
    There is no ``force=`` switch because there is nothing to choose: its threads do not race, and the units
    are those the thresholds are in.

    What of FORCE's Level 2 chain this takes as input rather than computing, since it is Fmask's (Zhu and
    Woodcock 2012; Zhu, Wang and Woodcock 2015) and not Frantz's: the potential cloud pixels (``potential``), the
    variability probability (``variability``, computed from Zhu and Woodcock's formula when not given), the
    bright pixels (``bright``, the mean of blue, green and red above 0.15 when not given) and the saturated ones
    (``saturated``). FORCE's clear-sky and clear-sky-land pixels, their counts and the cloud cover are not read
    by ``cloud_parallax`` and are not inputs.

    Parameters
    ----------
    cube
        Reflectance ``(time, band, y, x)`` or ``(band, y, x)`` on the 10 m grid with the bands rededge3 (B07), nir
        (B08) and nir08 (B8A) under Earth Search's names or ``B07``... (and blue, green and red unless ``bright``
        is given; and swir16 for the computed variability). Held in memory.
    potential
        Boolean array broadcastable to ``(time, y, x)``: the potential cloud pixels, e.g. from a cloud product.
    variability
        Optional float array broadcastable to ``(time, y, x)``, Fmask's variability probability in [0, 1]
        (FORCE keeps it times 10000 as an int16; it is rounded to that). By default
        ``1 - max(|NDVI|, |NDSI|, whiteness)`` of Zhu and Woodcock (2012), with ``NDVI = (B8A - red) / (B8A + red)``,
        ``NDSI = (green - SWIR1) / (green + SWIR1)`` and the whiteness of the visible bands (the sum of their
        distances to their mean, relative to the mean), clipped to [0, 1]. B8A is the NIR here as in FORCE, B11
        the SWIR1. Where ``saturated`` a positive NDVI and a negative NDSI are set to 0, the paper's adjustment
        for pixels whose visible bands saturate (which the paper gave for Landsat 7; Sentinel-2's saturation is
        rare and flagged in the quality bands, so the pixels come from those).
    saturated
        Optional boolean array broadcastable to ``(time, y, x)``: saturated pixels (FORCE's QAI saturation
        flag), which are cloud if bright. By default none.
    bright
        Optional boolean array broadcastable to ``(time, y, x)``. By default the mean of the three visible bands
        is above 0.15 (FORCE's 1500).
    bands, scale, nodata, n_jobs
        As in ``cloud_displacement_index``. Pixels with a missing value in one of the three bands of the test
        are never cloud.
    return_cdi
        Also return the CDI.

    Returns
    -------
    xarray.DataArray
        Boolean ``(time, y, x)`` (or ``(y, x)``), True for cloud; with ``return_cdi=True`` a tuple of that and
        the CDI on the 20 m grid (NaN where it has none).

    Examples
    --------
    >>> pcp = potential_cloud_mask(s2)          # a mask of Fmask's potential cloud pixels, or a cloud product
    >>> cloud = zeit.parallax_clouds(s2, pcp)
    >>> cloud = zeit.buffer_mask(cloud, 300)    # FORCE's cloud buffer, 300 m
    """
    from . import _core
    from ._threads import resolve

    work, has_time, order, mask_order, ints, off, pcp, var, brt, sat = _prepare(
        cube, potential, variability, saturated, bright, bands, scale, nodata,
        need_var=variability is None, need_bright=bright is None)
    threads = resolve(n_jobs)
    clouds, cdis = [], []
    for k in range(ints.shape[0]):
        re3, bnir, nir = (np.ascontiguousarray(ints[k][i]) for i in range(3))
        c, cdi = _core.parallax.parallax_batch(re3, bnir, nir, np.ascontiguousarray(var[k]),
                                               np.ascontiguousarray(pcp[k]), np.ascontiguousarray(off[k]),
                                               np.ascontiguousarray(brt[k]), np.ascontiguousarray(sat[k]), True,
                                               threads)
        clouds.append(c)
        cdis.append(cdi)
    cloud = np.stack(clouds)
    attrs = dict(cube.attrs)
    attrs["parallax_clouds"] = ("FORCE's cloud_parallax, cloud-ll.c (David Frantz, GPL-3.0-or-later); Frantz et al. "
                                "2018, RSE 215:471-481")
    res = _result(work, cloud if has_time else cloud[0], mask_order, {c: work[c].values for c in ("y", "x")
                  if c in work.coords}, "cloud", attrs)
    if cube.rio.crs is not None:
        res = res.rio.write_crs(cube.rio.crs)
    if not return_cdi:
        return res
    cdi = np.stack(cdis)
    cdi = np.where(cdi == -9999.0, np.float32(np.nan), cdi).astype(np.float32)
    cdi_da = _result(work, cdi if has_time else cdi[0], mask_order, _coords_20m(work), "cdi", dict(attrs))
    return res, cdi_da


def _resolution_of(mask: Any, resolution: Optional[float]) -> float:
    if resolution is not None:
        if not resolution > 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        return float(resolution)
    if isinstance(mask, xr.DataArray) and "x" in mask.coords and mask.sizes.get("x", 0) > 1:
        return float(abs(mask.x.values[1] - mask.x.values[0]))
    raise ValueError("units='m' needs the pixel size: pass resolution= (m) or a DataArray with an x coordinate")


def buffer_mask(
    mask: Any,
    distance: float,
    *,
    units: str = "m",
    resolution: Optional[float] = None,
    force: bool = False,
    n_jobs: int = -1,
) -> Any:
    """Buffers the True pixels of a boolean mask: FORCE's cloud buffer (``buffer_`` of ``imagefuns-cl.c``).

    Every True pixel grows to a disc, drawn by the midpoint circle algorithm (the pixels of the integer radius
    ``r`` around it as that algorithm gives them: the axis points at distance ``r`` are in, and so is ``(r, 1)``,
    a little fatter than the true disc: 89 pixels for ``r = 5``).
    In FORCE's Level 2 processing the radius in pixels is ``CLOUD_BUFFER / resolution`` (300 m by default; 40 and
    20 m for shadows and snow) truncated to an integer: with ``units="m"`` ``r = int(distance / resolution)``,
    the resolution from the ``x`` coordinate of a DataArray or ``resolution=``; with ``units="px"`` ``r`` is
    ``distance``, an integer. A radius of 0 changes nothing.

    FORCE's code only buffers the pixels at least ``r`` from the image's borders, and then replaces the mask by
    what was drawn: a True pixel within ``r`` of an edge is lost, and a cloud at the border gets no buffer.
    ``force=True`` does that (the values equal FORCE's, see ``benchmarks/vs_originals/force_parallax.py``); the
    default keeps every True pixel and draws its disc clipped at the image's edge, the same everywhere else.

    Parameters
    ----------
    mask
        Boolean ``(y, x)`` or ``(time, y, x)``, a DataArray or array; each date is buffered on its own.
    distance
        The buffer distance, in ``units``.
    units
        ``"m"`` or ``"px"``.
    resolution
        The pixel size in metres, when ``units="m"`` and the mask is not a DataArray with an ``x`` coordinate.
    force
        FORCE's behaviour at the borders (see above).
    n_jobs
        Threads: the processors zeit may use by default (``-1``).

    Returns
    -------
    Boolean, as ``mask``.
    """
    from . import _core
    from ._threads import resolve

    if units not in ("m", "px"):
        raise ValueError(f"units must be 'm' or 'px', got {units!r}")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError(f"distance must be a number of at least 0, got {distance}")
    if units == "px":
        if int(distance) != distance:
            raise ValueError(f"units='px' needs a whole number of pixels, got {distance}")
        r = int(distance)
    else:
        r = int(np.float32(distance) / _resolution_of(mask, resolution))
    is_da = isinstance(mask, xr.DataArray)
    values = np.asarray(mask.values if is_da else mask)
    if values.dtype != bool:
        if values.dtype.kind not in "iuf" or not np.isin(values[np.isfinite(values)], (0, 1)).all():
            raise ValueError("mask must be boolean (or 0/1)")
        values = values != 0
    if values.ndim not in (2, 3):
        raise ValueError(f"mask must be (y, x) or (time, y, x), got {values.ndim} dimensions")
    if is_da and values.ndim != mask.ndim:
        raise ValueError("mask has an unexpected shape")
    if is_da and mask.dims[-2:] != ("y", "x"):
        raise ValueError(f"a DataArray mask needs the dimensions (..., y, x), got {mask.dims}")
    threads = resolve(n_jobs)
    stack = values if values.ndim == 3 else values[None]
    out = np.stack([_core.parallax.buffer_batch(np.ascontiguousarray(m), r, bool(force), threads) for m in stack])
    if values.ndim == 2:
        out = out[0]
    if not is_da:
        return out
    result = mask.copy(data=out)
    result.attrs = dict(mask.attrs)
    result.attrs["buffer_mask"] = (f"{r} px midpoint-circle buffer{' (FORCE borders)' if force else ''}; FORCE's "
                                   f"buffer_, imagefuns-cl.c (David Frantz, GPL-3.0-or-later)")
    return result
