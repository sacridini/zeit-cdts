"""Time series analysis of a cube, per pixel: ``zeit.temporal_metrics`` (statistics over the whole
series or by year, quarter, month, week or day of the year) and ``zeit.standardize``.

The statistics are those of FORCE's higher-level TSA module (STM and fold, Frantz 2019, Remote
Sensing 11(9):1124), computed by the compiled core (``zeit._core.tsa``). Input and output follow
``zeit.mann_kendall``: a ``(time, y, x)`` cube (in memory or dask), a raster on disk, a numpy
stack with ``dates``, or one pixel's series; in-memory input gives an in-memory result and dask
input stays lazy.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._threads import resolve

_BY = ("year", "quarter", "month", "week", "doy")
# Days before each month in a year of 365 days: February 29th and March 1st share day 60, as in
# FORCE's calendar, so the same calendar day falls on the same day of the year in every year.
_CUM = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def _periods(times: pd.DatetimeIndex, by: str):
    """The period of each date as an index into ``coord`` (-1 for a missing date), and ``coord``:
    the years from the first to the last, 1-4, 1-12, 1-52 (``min(ceil(doy / 7), 52)``) or 1-365."""
    ok = ~times.isna()
    t = times[ok]
    if by == "year":
        lo, hi = int(t.year.min()), int(t.year.max())
        coord, value = np.arange(lo, hi + 1), np.zeros(len(times), dtype=np.int64)
        value[ok] = t.year - lo
    else:
        doy = _CUM[t.month - 1] + t.day
        number = {"quarter": (t.month - 1) // 3 + 1, "month": t.month, "doy": doy,
                  "week": np.minimum(-(-doy // 7), 52)}[by]
        coord = np.arange(1, {"quarter": 4, "month": 12, "week": 52, "doy": 365}[by] + 1)
        value = np.zeros(len(times), dtype=np.int64)
        value[ok] = np.asarray(number) - 1
    value[~ok] = -1
    return value.astype(np.int32), coord


def _work_dtype(cube: xr.DataArray) -> np.dtype:
    return np.dtype(np.float32) if cube.dtype in (np.float32, np.float16) else np.dtype(np.float64)


def _flat(block: np.ndarray) -> np.ndarray:
    """A ``(time, y, x)`` block as ``(pixels, time)`` float64."""
    return np.ascontiguousarray(block.reshape(block.shape[0], -1).T, dtype=np.float64)


def _metrics_block(block, names, ddof, n_jobs, group, n_groups, dtype):
    from . import _core

    t, ny, nx = block.shape
    shape = (len(names),) + ((n_groups,) if group is not None else ()) + (ny, nx)
    if block.size == 0:
        return np.full(shape, np.nan, dtype=dtype)
    values = _flat(block)
    if group is None:
        out = _core.tsa.stm_batch(values, names, ddof, n_jobs)  # (pixels, metrics)
        out = out.T
    else:
        out = _core.tsa.fold_batch(values, group, n_groups, names, ddof, n_jobs)  # (pixels, metrics, groups)
        out = out.transpose(1, 2, 0)
    return np.ascontiguousarray(out, dtype=dtype).reshape(shape)


def temporal_metrics(data: Any, metrics: Sequence[str] = ("min", "p25", "median", "p75", "max", "mean", "std"), *,
                     by: Optional[str] = None, ddof: int = 1, dates: Optional[Sequence[Any]] = None,
                     band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto",
                     chunks: Any = None, n_jobs: int = -1) -> xr.DataArray:
    """Statistics of each pixel's series, over all of it or by period (FORCE's STM and fold).

    Parameters
    ----------
    data
        A ``(time, y, x)`` cube (in memory or dask), anything ``zeit.load_raster`` reads, a
        ``(time, band, y, x)`` cube or Dataset with ``band``, a numpy ``(time, y, x)`` with
        ``dates``, or one pixel's series (list/1-D array with ``dates``, or a ``pandas.Series``
        indexed by date). Any spacing; NaN marks a missing observation.
    metrics
        ``min``, ``max``, ``range`` (max - min), ``iqr`` (p75 - p25), ``mean``, ``std``,
        ``skew`` (M3 / sigma^3, sigma the population deviation: FORCE's SKW), ``kurtosis``
        (excess, M4 / sigma^4 - 3: FORCE's KRT / 1000), ``count`` (valid observations),
        ``median`` and ``pNN`` for any integer percentile 0-100 (R-7, as
        ``np.nanpercentile``).
    by
        ``None``: the statistics of the whole series. ``"year"``, ``"quarter"``, ``"month"``,
        ``"week"`` or ``"doy"``: those of each period (the dates are needed). The week is
        ``min(ceil(doy / 7), 52)`` and the day of the year counts 365 days, 29 February sharing
        day 60 with 1 March, so that a calendar day is the same day of the year in every year
        (FORCE's calendar). Periods: the years from the first to the last date, 1-4, 1-12, 1-52,
        1-365, all of them, the empty ones NaN (0 for ``count``).
    ddof
        Degrees of freedom of ``std``: the divisor is ``n - ddof`` (1: the sample deviation, as
        FORCE's STD; 0: the population's). A series with ``n <= ddof`` gives NaN; so do ``skew``
        and ``kurtosis`` of a series with fewer than 2 distinct values (FORCE leaves it undefined).
    dates, band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.DataArray
        ``(metric, y, x)``, or ``(metric, <by>, y, x)`` with ``by``, with the names of the
        metrics in ``metric``. Float32 for float32 input, else float64. Pixels without a valid
        observation are NaN (0 for ``count``).

    Examples
    --------
    >>> stm = zeit.temporal_metrics(ndvi, ["p10", "median", "p90"])            # (metric, y, x)
    >>> yearly = zeit.temporal_metrics(ndvi, ["mean", "max"], by="year")       # (metric, year, y, x)
    """
    from ._embeddings import refuse
    from ._series_api import _as_dask, _series_cube
    from .indices import parse_metrics

    refuse(data, "zeit.temporal_metrics")
    if by is not None and by not in _BY:
        raise ValueError(f"by must be None or one of {_BY}, got {by!r}")
    names = parse_metrics(metrics)
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, need_dates=by is not None, nodata=nodata)
    group, coord = (None, None) if by is None else _periods(pd.DatetimeIndex(cube.time.values), by)
    n_groups = 0 if coord is None else len(coord)
    threads, dtype, ddof = resolve(n_jobs), _work_dtype(cube), float(ddof)
    lazy = cube.chunks is not None

    if lazy:
        import dask.array as da

        arr, _ = _as_dask(cube)
        new_axis = [0] if by is None else [0, 1]
        chunks_out = ((len(names),),) + (() if by is None else ((n_groups,),)) + arr.chunks[1:]
        out = da.map_blocks(_metrics_block, arr, names, ddof, threads, group, n_groups, dtype,
                            dtype=dtype, drop_axis=[0], new_axis=new_axis, chunks=chunks_out,
                            meta=np.empty((0,) * len(chunks_out), dtype))
    else:
        out = _metrics_block(np.asarray(cube.values), names, ddof, threads, group, n_groups, dtype)

    dims = ("metric",) + (() if by is None else (by,)) + ("y", "x")
    coords = {"metric": names}
    if by is not None:
        coords[by] = coord
    if not pixel:
        for name in ("y", "x", "spatial_ref"):
            if name in cube.coords:
                coords[name] = cube.coords[name]
    result = xr.DataArray(out, dims=dims, coords=coords, name="temporal_metrics",
                          attrs={"algorithm": "temporal_metrics", "by": "none" if by is None else by,
                                 "ddof": int(ddof)})
    if pixel:
        return result.squeeze(("y", "x"), drop=True)
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    return result


def _standardize_block(block, method, ddof, n_jobs, dtype):
    from . import _core

    t, ny, nx = block.shape
    if block.size == 0:
        return np.full(block.shape, np.nan, dtype=dtype)
    out = _core.tsa.standardize_batch(_flat(block), method, ddof, n_jobs)  # (pixels, time)
    return np.ascontiguousarray(out.T, dtype=dtype).reshape(t, ny, nx)


def standardize(data: Any, method: str = "normalize", *, ddof: int = 1, dates: Optional[Sequence[Any]] = None,
                band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
                n_jobs: int = -1) -> Any:
    """Standardizes each pixel's series along time (FORCE's STANDARDIZE).

    Parameters
    ----------
    data
        As for ``zeit.temporal_metrics``; the dates are not needed.
    method
        ``"normalize"`` (default): ``(x - mean) / std``, the z-score; ``"center"``:
        ``x - mean``. The mean and the deviation are those of the valid observations.
    ddof
        Degrees of freedom of the deviation (``n - ddof``; 1: the sample's, as FORCE).
    dates, band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.DataArray (or pandas.Series for a Series)
        The cube with the same dims, coordinates and georeferencing; missing observations stay
        NaN. A series with fewer than 2 valid observations is left as it is; a constant series
        gives NaN when normalized. Float32 for float32 input, else float64.
    """
    from ._embeddings import refuse
    from ._series_api import _as_dask, _series_cube

    refuse(data, "zeit.standardize")
    if str(method).lower() not in ("normalize", "center"):
        raise ValueError(f"method must be 'normalize' or 'center', got {method!r}")
    method = str(method).lower()
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, need_dates=False, nodata=nodata)
    threads, dtype, ddof = resolve(n_jobs), _work_dtype(cube), float(ddof)
    if cube.chunks is not None:
        import dask.array as da

        arr, _ = _as_dask(cube)
        out = da.map_blocks(_standardize_block, arr, method, ddof, threads, dtype, dtype=dtype,
                            meta=np.empty((0, 0, 0), dtype))
    else:
        out = _standardize_block(np.asarray(cube.values), method, ddof, threads, dtype)
    result = cube.copy(data=out)
    result.attrs = {**cube.attrs, "standardization": method}
    if pixel:
        values = np.asarray(result.values).reshape(-1)
        if isinstance(data, pd.Series):
            return pd.Series(values, index=data.index, name=data.name)
        result = result.squeeze(("y", "x"), drop=True)
    return result
