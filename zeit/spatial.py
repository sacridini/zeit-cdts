"""Spatial post-processing of maps: minimum mapping unit, majority and Bayesian filters.

Each takes a ``DataArray`` (or a numpy array, or a raster path) and gives back the same
kind, georeferencing and attributes kept.
"""
from typing import Any

import numpy as np
from scipy import ndimage, stats
from scipy.ndimage import generic_filter, uniform_filter


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


def apply_majority_filter(image: Any, size: int = 3) -> Any:
    """Majority (mode) filter: every pixel takes the most common value of its ``size`` x
    ``size`` window, regularising a class map (like sits' smoothing of class maps).

    image: a class map (``DataArray``, numpy array or raster path). Returns the same kind.
    """
    values, rebuild, _ = _as_map(image)

    def _mode_func(window):
        # Return the most common value in the window
        return stats.mode(window, axis=None, keepdims=False).mode

    # generic_filter applies the function to a moving window
    return rebuild(generic_filter(values, _mode_func, size=size))


def apply_bayesian_filter(probs: Any, window_size: int = 3) -> Any:
    """Bayesian smoothing of class probabilities: each pixel's probability of a class is
    multiplied by the class's mean probability around it, and the pixel takes the class with
    the highest product. Unlike a majority filter, it weighs the model's confidence.

    probs: ``(class, y, x)`` probabilities or scores (e.g. ``zeit.classify(...,
        probability=True).probability``), a ``DataArray`` or numpy array.

    Returns the winning class per pixel ``(y, x)``: its index (numpy), or, for a DataArray
    whose first dim has coordinates, the class itself.
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
