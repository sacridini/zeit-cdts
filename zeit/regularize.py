from typing import Any

import numpy as np
import xarray as xr


def regularize_time_series(cube: Any, freq: str = '16D', method: str = 'median') -> xr.DataArray:
    """
    Regularize a time series to a fixed step: one composite per period, by median or medoid.

    - cube: a cube with a ``time`` dim, ``(time, y, x)`` or ``(time, band, y, x)`` (in memory
      or dask), or anything ``load_raster`` reads.
    - freq: the step, a pandas frequency (``'16D'``, ``'MS'``, ``'1YS'``...).
    - method: ``'median'`` (per band, each band its own median) or ``'medoid'`` (the
      observation closest to the median across bands, so the bands of a composite come
      from one date; with a single band, the observation closest to the median). Only
      observations with every band are candidates for the medoid. The medoid runs in C++,
      every pixel in parallel.

    Missing values (NaN, and the cube's NoData) are left out. Periods without observations,
    and pixels without one in a period, are NaN (the NoData value in an integer cube that
    has one). Georeferencing and NoData are kept.
    """
    if method not in ("median", "medoid"):
        raise ValueError(f"Unknown method {method}. Use 'median' or 'medoid'.")
    if not isinstance(cube, xr.DataArray):
        from ._load import load_raster
        cube = load_raster(cube)
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("regularize_time_series needs a time dim with dates")
    spatial = "y" in cube.dims and "x" in cube.dims
    crs = cube.rio.crs if spatial else None
    nodata = cube.rio.nodata if spatial else None
    dtype = cube.dtype
    if nodata is not None and not np.isnan(nodata):
        work = cube.where(cube != nodata)          # NoData out of the medians (floats from here)
    else:
        nodata = None
        work = cube if np.issubdtype(dtype, np.floating) else cube.astype(np.float64)

    if method == 'median':
        out = work.resample(time=freq).median(dim='time')
    else:
        out = _medoid(work, freq)

    if np.issubdtype(dtype, np.integer) and nodata is not None:
        out = out.fillna(nodata).astype(dtype)     # back to the cube's type, gaps as NoData
    elif np.issubdtype(dtype, np.floating):
        out = out.astype(dtype)
    if crs is not None and out.rio.crs is None:
        out = out.rio.write_crs(crs)
    if nodata is not None and out.rio.nodata is None:
        out = out.rio.write_nodata(nodata, encoded=False)
    return out


def _medoid(cube: xr.DataArray, freq: str) -> xr.DataArray:
    """The medoid of each period by ``_core.utils.compute_medoid`` (C++, OpenMP over rows):
    per pixel, the median of each band over the period's valid values, then the observation
    with every band whose distance to it is smallest (the first one on a tie); NaN without one."""
    from . import _core

    single = "band" not in cube.dims
    work = cube.expand_dims(band=["value"], axis=1) if single else cube
    others = [d for d in work.dims if d not in ("time", "band")]
    if work.chunks is not None:
        work = work.chunk({"time": -1, "band": -1})

    def _block(values: np.ndarray) -> np.ndarray:
        # (..., time, band), the other dims first; the engine takes (rows, cols, time, band)
        lead = values.shape[:-2]
        flat = np.ascontiguousarray(values.reshape((-1, 1) + values.shape[-2:]), dtype=np.float64)
        out = _core.utils.compute_medoid(flat, np.nan)   # only NaN is missing (NoData is NaN by now)
        return np.asarray(out).reshape(lead + (values.shape[-1],))

    def _period(group: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(_block, group, input_core_dims=[["time", "band"]], output_core_dims=[["band"]],
                              dask="parallelized", output_dtypes=[np.float64])

    out = work.resample(time=freq).map(_period)
    out = out.transpose("time", "band", *others)
    return out.isel(band=0, drop=True) if single else out
