"""Post-processing of maps and classifications: minimum mapping unit, majority and Bayesian
filters, sits' smoothing of class probabilities, a sieve of small patches and the removal of short
excursions from class series.

Each takes a ``DataArray`` (or a numpy array, or a raster path) and gives back the same
kind, georeferencing and attributes kept.
"""
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter

from ._threads import resolve


def _as_map(data: Any):
    """(values, rebuild): the 2-D values of a map, and a function giving back a result like
    the input (a DataArray with its coordinates, or a numpy array)."""
    import os

    import xarray as xr

    if isinstance(data, (str, os.PathLike)):
        from ._load import load_raster

        data = load_raster(data)
    if isinstance(data, xr.DataArray):
        template = data

        def rebuild(values, dims=None, keep_attrs=True):
            dims = dims or template.dims
            coords = {c: v for c, v in template.coords.items() if set(v.dims) <= set(dims)}
            out = xr.DataArray(values, dims=dims, coords=coords, name=template.name,
                               attrs=dict(template.attrs) if keep_attrs else {})
            if "y" in dims and template.rio.crs is not None:
                out = out.rio.write_crs(template.rio.crs)
            return out

        return np.asarray(data.values), rebuild, data
    return np.asarray(data), (lambda values, dims=None, keep_attrs=True: values), None


def apply_mmu_filter(data: Any, mmu_pixels: int = 11, *, nodata: Any = "auto") -> Any:
    """Minimum Mapping Unit: remove patches smaller than ``mmu_pixels`` pixels.

    A patch is a group of connected (4-neighbour) pixels holding a value (not NoData), whatever
    the value: the islands of a map of disturbances (LandTrendr's year of loss, a magnitude), or
    of any map with NoData around its features. Its pixels become NoData. Reduces
    "salt and pepper" noise.

    data: a map (``DataArray``, numpy array or raster path).
    mmu_pixels: patches with fewer pixels are removed.
    nodata: the value of "no feature": ``"auto"`` (the map's NoData, else ``0``; NaN for
        floats without one), or a number.

    Returns the filtered map, of the input's kind (a path gives a DataArray).
    """
    values, rebuild, da = _as_map(data)
    if values.ndim != 2:
        raise ValueError(f"apply_mmu_filter takes a single map (y, x), got shape {values.shape}")
    if nodata == "auto":
        stored = da.rio.nodata if da is not None else None
        nodata = stored if stored is not None else (np.nan if np.issubdtype(values.dtype, np.floating)
                                                    and np.isnan(values).any() else 0)
    empty = np.isnan(values) if (isinstance(nodata, float) and np.isnan(nodata)) else (values == nodata)
    labeled, num_features = ndimage.label(~empty)
    sizes = ndimage.sum(~empty, labeled, range(num_features + 1))
    remove = (sizes < mmu_pixels)[labeled] & ~empty
    filtered = np.copy(values)
    filtered[remove] = nodata
    return rebuild(filtered)


def _classes_in(values: np.ndarray, da, nodata: Any):
    """(float values with NaN where missing, the NoData to put back or None)."""
    if nodata == "auto":
        nodata = da.rio.nodata if da is not None and hasattr(da, "rio") else None
    out = values.astype(np.float64)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        out[values == nodata] = np.nan
    return out, nodata


def _classes_out(values: np.ndarray, dtype, nodata: Any) -> np.ndarray:
    """The filtered classes back in the input's dtype, NoData where missing."""
    missing = np.isnan(values)
    if np.issubdtype(dtype, np.floating):
        out = values.astype(dtype)
        if nodata is not None:
            out[missing] = nodata
        return out
    out = np.where(missing, 0 if nodata is None else nodata, values)
    return out.astype(dtype)


def apply_majority_filter(image: Any, size: int = 3, *, iterations: int = 1, nodata: Any = "auto",
                          n_jobs: int = -1) -> Any:
    """Majority (mode) filter: every pixel takes the most common class of its ``size`` x ``size``
    window (cut at the edges), its own when it is among the most common (else the smallest of
    them), ``iterations`` times or until nothing changes. NoData pixels stay and do not vote.

    image: a class map (``DataArray``, numpy array or raster path). Returns the same kind.
    size: the window, odd (3 = 3 x 3).
    iterations: how many times to apply it.
    nodata: ``"auto"`` (the map's NoData, NaN for floats), a number, or ``None``.
    """
    from ._core import postproc

    values, rebuild, da = _as_map(image)
    if values.ndim != 2:
        raise ValueError(f"apply_majority_filter takes a single map (y, x), got shape {values.shape}")
    classes, nodata = _classes_in(values, da, nodata)
    out = postproc.majority(classes, int(size), int(iterations), resolve(n_jobs))
    return rebuild(_classes_out(out, values.dtype, nodata))


def mmu_sieve(data: Any, min_size: int = 11, *, connectivity: int = 8, nodata: Any = "auto") -> Any:
    """Minimum mapping unit of a class map by a sieve: each patch (pixels of one class connected
    ``connectivity``-wise) of fewer than ``min_size`` pixels is merged, the smallest first, into
    the neighbouring patch with which it shares the longest border (on a tie, the largest), and
    takes its class. Unlike :func:`apply_mmu_filter`, the small patches are not left empty: the
    map stays complete, as with GDAL's sieve.

    data: a class map (``DataArray``, numpy array or raster path).
    min_size: patches with fewer pixels are merged (11 Landsat pixels ~ 1 ha).
    connectivity: 4 or 8 (diagonal pixels in the same patch).
    nodata: ``"auto"`` (the map's NoData, NaN for floats), a number, or ``None``. NoData stays,
        and a patch surrounded by NoData only is kept.

    Returns the map of the input's kind.
    """
    from ._core import postproc

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    values, rebuild, da = _as_map(data)
    if values.ndim != 2:
        raise ValueError(f"mmu_sieve takes a single map (y, x), got shape {values.shape}")
    classes, nodata = _classes_in(values, da, nodata)
    out = postproc.sieve(classes, int(min_size), connectivity == 8)
    return rebuild(_classes_out(out, values.dtype, nodata))


def smooth_probs(probs: Any, window_size: int = 9, neigh_fraction: float = 0.5,
                 smoothness: Any = 20.0, *, n_jobs: int = -1) -> Any:
    """Bayesian smoothing of class probabilities, as sits' ``sits_smooth`` (Camara et al.):
    each class's logit, ``log(p / (sum - p))``, is pulled towards the mean of the highest
    ``neigh_fraction`` of the logits around it (a ``window_size`` window, mirrored at the edges),
    the more as the neighbourhood varies little against ``smoothness``:
    ``w = s2 / (s2 + smoothness)``, ``x' = w x + (1 - w) m``. A pixel confident against
    consistent neighbours keeps its class; noise among uncertain pixels goes. The result equals
    sits' to 1e-15, except that a pixel missing (NaN) stays missing where sits gives it its
    neighbours' mean.

    probs: ``(class, y, x)`` probabilities in [0, 1] (e.g. ``zeit.classify(...,
        probability=True).probability``), a ``DataArray``, numpy array or raster path.
    window_size: the neighbourhood, odd (sits' default 9).
    neigh_fraction: the fraction of the neighbourhood's highest logits that make its mean.
    smoothness: the prior's variance, one number or one per class: higher smooths more.

    Returns the smoothed probabilities, of the input's kind (they no longer sum to one, as
    sits'): the class of each pixel is their ``argmax`` over ``class``.
    """
    from ._core import postproc

    values, rebuild, _ = _as_map(probs)
    if values.ndim != 3:
        raise ValueError(f"smooth_probs takes probabilities (class, y, x), got shape {values.shape}")
    if np.nanmax(values) > 1 or np.nanmin(values) < 0:
        raise ValueError("probabilities must be in [0, 1] (divide integer ones by their scale, "
                         "e.g. 10000 for sits')")
    smooth = np.atleast_1d(np.asarray(smoothness, dtype=np.float64))
    if smooth.size == 1:
        smooth = np.repeat(smooth, values.shape[0])
    if smooth.shape != (values.shape[0],):
        raise ValueError(f"one smoothness per class: {smooth.size} for {values.shape[0]} classes")
    if (smooth <= 0).any():
        raise ValueError("smoothness must be positive")
    out = postproc.smooth_probs(values.astype(np.float64), int(window_size), np.ascontiguousarray(smooth),
                                float(neigh_fraction), resolve(n_jobs))
    if np.issubdtype(values.dtype, np.floating):
        out = out.astype(values.dtype)
    return rebuild(out)


def desawtooth_classes(data: Any, max_length: int = 1, *, nodata: Any = "auto", n_jobs: int = -1) -> Any:
    """Temporal consistency of class series: in each pixel's series, a run of at most
    ``max_length`` dates of one class between two runs of another, the same before and after
    (forest, pasture, forest), takes that class (forest, forest, forest). The runs are those of
    the series given: forest, pasture, crop, pasture, forest with ``max_length=1`` only changes
    the crop. NoData dates stay and do not break a run.

    data: classes ``(time, y, x)`` (a ``DataArray`` with ``time``, a numpy array: time first)
        or one series.
    max_length: the longest excursion removed, in dates.
    nodata: ``"auto"`` (the map's NoData, NaN for floats), a number, or ``None``.

    Returns the series of the input's kind.
    """
    import xarray as xr

    from ._core import postproc

    if isinstance(data, xr.DataArray) and "time" in data.dims and data.dims[0] != "time":
        order = data.dims
        out = desawtooth_classes(data.transpose("time", ...), max_length, nodata=nodata, n_jobs=n_jobs)
        return out.transpose(*order)
    values, rebuild, da = _as_map(data)
    if values.ndim == 0:
        raise ValueError("desawtooth_classes takes a series or a cube (time, y, x)")
    classes, nodata = _classes_in(values, da, nodata)
    t = classes.shape[0]
    flat = np.ascontiguousarray(classes.reshape(t, -1).T)
    out = postproc.desawtooth(flat, int(max_length), resolve(n_jobs)).T.reshape(classes.shape)
    return rebuild(_classes_out(out, values.dtype, nodata))


def apply_bayesian_filter(probs: Any, window_size: int = 3) -> Any:
    """Bayesian smoothing of class probabilities: each pixel's probability of a class is
    multiplied by the class's mean probability around it, and the pixel takes the class with
    the highest product. Unlike a majority filter, it weighs the model's confidence.

    probs: ``(class, y, x)`` probabilities or scores (e.g. ``zeit.classify(...,
        probability=True).probability``), a ``DataArray`` or numpy array.

    Returns the winning class per pixel ``(y, x)``: its index (numpy), or, for a DataArray
    whose first dim has coordinates, the class itself. :func:`smooth_probs` is sits' smoothing.
    """
    values, rebuild, da = _as_map(probs)
    C, Y, X = values.shape
    smoothed_probs = np.zeros_like(values)

    for c in range(C):
        smoothed_probs[c] = uniform_filter(values[c], size=window_size)

    # Bayesian update: P(class|neighbor) is proportional to P(class) * P_neighbor(class)
    updated_probs = values * smoothed_probs
    winner = np.argmax(updated_probs, axis=0)
    if da is None:
        return winner
    first = da.dims[0]
    if first in da.coords:
        winner = np.asarray(da.coords[first].values)[winner]
    return rebuild(winner, dims=da.dims[1:], keep_attrs=False)
