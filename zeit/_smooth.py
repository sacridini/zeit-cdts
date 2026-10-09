"""``zeit.smooth``: smooth time series (Whittaker or Savitzky-Golay), keeping the cube.

``apply_savgol_filter`` and ``apply_whittaker_filter`` are the numpy engines of before.
"""
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter

_METHODS = ("whittaker", "savgol")


def smooth(
    data: Any,
    *,
    method: str = "whittaker",
    lmbda: float = 10.0,
    weights: Any = None,
    window: int = 5,
    polyorder: int = 2,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> Any:
    """Smooth time series along ``time``, keeping their dates, georeferencing and type.

    Parameters
    ----------
    data
        A cube with a ``time`` dim (``(time, y, x)``, ``(time, band, y, x)``...), numpy- or
        dask-backed, or anything ``load_raster`` reads; or one pixel's series, a
        ``pandas.Series`` indexed by dates or a 1-D ``DataArray`` with ``time``.
    method
        ``"whittaker"`` (default): the Whittaker smoother (Eilers 2003), a penalised
        least-squares fit that follows the data more or less closely (``lmbda``); dates
        may be unevenly spaced (the penalty uses the time between them) and missing
        observations (NaN, NoData) are filled by the curve. ``"savgol"``: the
        Savitzky-Golay filter (a local polynomial of ``polyorder`` over ``window`` dates),
        for evenly spaced series; missing observations are first interpolated linearly
        in time.
    lmbda
        Whittaker: the smoothness, larger is smoother (``10`` is light for a 16-day NDVI
        series, ``1000`` strong). Its meaning does not depend on the spacing: the time
        steps are measured in median steps of the series.
    weights
        Whittaker: a weight per observation, same shape as ``data`` (e.g. ``1`` clear,
        ``0.2`` hazy, ``0`` cloudy); ``None``: every valid observation weighs 1.
    window, polyorder
        Savitzky-Golay: the window length in dates (odd) and the polynomial's order.
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``: ``"auto"`` (the
        raster's NoData; ``0`` for integer data without one), a number, or ``None``.
    chunks
        Inputs read from disk: ``None`` reads into memory; ``"auto"`` or a dict keeps the
        result lazy.
    n_jobs
        Whittaker: CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.DataArray (or pandas.Series for a Series)
        The same dims, coordinates and georeferencing. Floats keep their type; integer
        data (e.g. NDVI x 10000 in Int16) is rounded back to its type when it has a
        NoData value (kept for the pixels with no observation at all), else returned as
        float32. ``attrs["smoothing"]`` records the method. A lazy cube stays lazy.

    Examples
    --------
    >>> ndvi = zeit.load_raster("S2_ndvi.tif")                  # (time, y, x), cloudy dates as NaN
    >>> smooth = zeit.smooth(ndvi, lmbda=100)                   # gaps filled by the curve
    >>> zeit.plot(ndvi, fit=smooth)                              # click a pixel: raw and smoothed
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if method == "savgol" and (window % 2 == 0 or window <= polyorder):
        raise ValueError(f"savgol: window must be odd and larger than polyorder, got {window} and {polyorder}")
    series = isinstance(data, pd.Series)
    if series:
        data = xr.DataArray(data.to_numpy(dtype=np.float64), dims=("time",),
                            coords={"time": pd.to_datetime(data.index)}, name=data.name)
    if isinstance(data, xr.DataArray) and "y" not in data.dims and "x" not in data.dims:
        cube = data
    else:
        from ._load import load_raster

        kwargs = {}
        if not isinstance(data, (xr.DataArray, xr.Dataset, np.ndarray)) and not hasattr(data, "dask"):
            kwargs["chunks"] = chunks
        cube = load_raster(data, **kwargs)
        if chunks is not None and cube.chunks is None and isinstance(data, xr.DataArray):
            cube = cube.chunk(chunks)
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("smooth needs a time dim with dates (load the data with zeit.load_raster(..., dates=...))")
    if cube.sizes["time"] < 3:
        raise ValueError(f"smooth needs at least 3 dates, got {cube.sizes['time']}")
    if method == "savgol" and cube.sizes["time"] < window:
        raise ValueError(f"savgol: {cube.sizes['time']} dates, fewer than window={window}")

    from ._lt import _missing_values

    sentinels = _missing_values(cube, nodata) if "y" in cube.dims else ([] if nodata in ("auto", None) else [nodata])
    times = pd.DatetimeIndex(cube.time.values)
    days = ((times.asi8 - times.asi8[0]) / (86400 * 10**9)).astype(np.float64)
    if not (np.diff(days) > 0).all():
        raise ValueError("smooth needs increasing, distinct dates")
    if weights is not None:
        weights = xr.DataArray(weights, dims=cube.dims) if not isinstance(weights, xr.DataArray) else weights
        weights = weights.transpose(*cube.dims)

    dtype = np.dtype(cube.dtype)
    stored = cube.rio.nodata if "y" in cube.dims else None
    integer = np.issubdtype(dtype, np.integer)
    keep_int = integer and stored is not None and not np.isnan(stored)
    work = np.dtype(np.float32) if dtype in (np.float32, np.float16) or integer else np.dtype(np.float64)

    def _run(values: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        for value in sentinels:
            values = np.where(values == value, np.nan, values)
        flat = np.ascontiguousarray(values.reshape(-1, values.shape[-1]))
        if method == "whittaker":
            from . import _core

            wflat = np.empty((0, 0)) if w is None else np.ascontiguousarray(
                np.asarray(w, dtype=np.float64).reshape(flat.shape))
            out = np.asarray(_core.smooth.whittaker_batch(flat, days, wflat, float(lmbda), n_jobs))
        else:
            out = _savgol_rows(flat, days, window, polyorder)
        out = out.reshape(values.shape)
        if keep_int:
            info = np.iinfo(dtype)
            filled = np.where(np.isnan(out), stored, np.clip(np.round(out), info.min, info.max))
            return filled.astype(dtype)
        return out.astype(work)

    others = [d for d in cube.dims if d != "time"]
    args = [cube.transpose(*others, "time")]
    core = [["time"]]
    if weights is not None:
        args.append(weights.transpose(*others, "time"))
        core.append(["time"])
    lazy = cube.chunks is not None
    if lazy:
        args = [a.chunk({"time": -1}) for a in args]
    out = xr.apply_ufunc(_run, *args, input_core_dims=core, output_core_dims=[["time"]],
                         dask="parallelized" if lazy else "forbidden",
                         output_dtypes=[dtype if keep_int else work], keep_attrs=True)
    out = out.transpose(*cube.dims).assign_coords(time=cube.time)
    out.name = cube.name
    out.attrs["smoothing"] = (f"Whittaker (lambda={lmbda:g})" if method == "whittaker"
                              else f"Savitzky-Golay (window={window}, polyorder={polyorder})")
    if "y" in cube.dims and cube.rio.crs is not None:
        out = out.rio.write_crs(cube.rio.crs)
        if stored is not None:
            out = out.rio.write_nodata(stored if keep_int else np.nan, encoded=False)
    if series:
        return out.to_series()
    return out


def _savgol_rows(flat: np.ndarray, days: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay along each row of (pixels, time), the gaps interpolated linearly in
    time first; rows without observations stay NaN."""
    out = np.full(flat.shape, np.nan)
    valid = np.isfinite(flat)
    full = valid.all(axis=1)
    if full.any():
        out[full] = savgol_filter(flat[full], window_length=window, polyorder=polyorder, axis=1)
    for p in np.flatnonzero(~full & valid.any(axis=1)):
        row = np.interp(days, days[valid[p]], flat[p, valid[p]])
        out[p] = savgol_filter(row, window_length=window, polyorder=polyorder)
    return out


def apply_savgol_filter(cube: "np.ndarray", window_length: int = 5, polyorder: int = 2, axis: int = 0) -> "np.ndarray":
    """
    Applies a Savitzky-Golay filter to smooth a time-series cube.
    This removes minor temporal noise and regularizes the trajectory before AI/classification.
    
    Args:
        cube (np.ndarray): The data cube. Expected shape (Time, Bands, H, W) or (Time, H, W).
        window_length (int): The length of the filter window (must be an odd integer).
        polyorder (int): The order of the polynomial used to fit the samples.
        axis (int): The temporal axis.
        
    Returns:
        np.ndarray: Smoothed cube with the same shape.
    """
    if cube.shape[axis] < window_length:
        raise ValueError(f"Time dimension ({cube.shape[axis]}) is smaller than window_length ({window_length}).")
        
    # Apply Savitzky-Golay filter along the time axis
    smoothed = savgol_filter(cube, window_length=window_length, polyorder=polyorder, axis=axis)
    
    # Optional: preserve NoData values (assuming 0 is NoData in original)
    smoothed[cube == 0] = 0
    
    return smoothed

import scipy.sparse as sp
from scipy.sparse.linalg import splu

def apply_whittaker_filter(cube: "np.ndarray", lmbd: float = 10.0, axis: int = 0, weights: "np.ndarray" = None) -> "np.ndarray":
    """
    Applies a Whittaker smoother along the time axis.
    This is often superior to Savitzky-Golay for NDVI/EVI because it penalizes roughness directly 
    and handles missing/cloudy data elegantly when weights are provided (0 for cloud, 1 for clear).
    
    Args:
        cube: 3D numpy array [Time, Y, X].
        lmbd: Smoothing parameter (larger = smoother curve).
        axis: The temporal axis.
        weights: Optional 3D array of the same shape with weights for each observation.
    """
    if axis != 0:
        cube = np.swapaxes(cube, 0, axis)
        if weights is not None:
            weights = np.swapaxes(weights, 0, axis)
            
    T, Y, X = cube.shape
    smoothed = np.zeros_like(cube)
    
    # Pre-build difference matrix for length T
    E = sp.eye(T, format='csc')
    D = E[2:] - 2 * E[1:-1] + E[:-2]
    D_T_D = D.T @ D
    
    for y in range(Y):
        for x in range(X):
            ts = cube[:, y, x]
            if weights is None:
                w = np.ones(T)
            else:
                w = weights[:, y, x]
                
            W = sp.spdiags(w, 0, T, T)
            A = W + lmbd * D_T_D
            try:
                # Solve A * z = W * ts
                z = splu(A.tocsc()).solve(w * ts)
                smoothed[:, y, x] = z
            except RuntimeError:
                # Fallback if matrix is perfectly singular (rare)
                smoothed[:, y, x] = ts
                
    if axis != 0:
        smoothed = np.swapaxes(smoothed, 0, axis)
        
    return smoothed
