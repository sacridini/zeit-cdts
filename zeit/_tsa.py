"""Time series analysis of a cube, per pixel: ``zeit.temporal_metrics`` (statistics over the whole
series or by year, quarter, month, week or day of the year), ``zeit.standardize`` and
``zeit.interpolate`` (linear, moving average, RBF ensemble or robust harmonic fit, to a regular
grid or to any dates), ``zeit.linear_trend`` and ``zeit.cat`` (the linear trend of a series and its
greatest change, with the trend before and after it) and ``zeit.clear_observations`` (the clear
observations in windows of months and the gaps between them).

The statistics and the interpolation are those of FORCE's higher-level TSA module (STM, fold and
interpolation, Frantz 2019, Remote Sensing 11(9):1124), computed by the compiled core (``zeit._core.tsa``). Input and output follow
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
    """A ``(time, y, x)`` block as ``(pixels, time)``: float32 stays float32 (the core reads it as it is and works in
    float64), anything else is float64."""
    dtype = np.float32 if block.dtype == np.float32 else np.float64
    return np.ascontiguousarray(block.reshape(block.shape[0], -1).T, dtype=dtype)


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


def _along(data: Any, dim: Optional[str]):
    """The series axis of ``data``: the data with that axis renamed ``time`` and stripped of its
    coordinate, and the name and values of the original axis (``(None, None)`` for the usual
    ``time`` axis, left as it is). A ``metric`` axis of length 1 is squeezed; with more, the
    caller must pick one. With ``dim=None`` the axis is ``time``, else the only axis that is not
    ``y``, ``x``, ``band`` or ``metric``."""
    if isinstance(data, xr.Dataset):
        names = [v for v in data.data_vars if data[v].ndim >= 3]
        if len(names) != 1:
            return data, None, None
        data = data[names[0]]
    if not isinstance(data, xr.DataArray):
        if dim not in (None, "time"):
            raise ValueError(f"dim={dim!r} needs an xarray DataArray, got {type(data).__name__}")
        return data, None, None
    if "metric" in data.dims:
        if data.sizes["metric"] != 1:
            raise ValueError(f"the data has {data.sizes['metric']} metrics {list(data.metric.values)}: "
                             "choose one with .sel(metric=...)")
        data = data.isel(metric=0, drop=True)
    if dim is None:
        if "time" in data.dims:
            return data, None, None
        others = [d for d in data.dims if d not in ("y", "x", "band", "metric")]
        if len(others) != 1:
            return data, None, None
        dim = others[0]
    if dim == "time":
        return data, None, None
    if dim not in data.dims:
        raise ValueError(f"dim={dim!r} is not a dimension of the data {data.dims}")
    if "time" in data.dims:
        raise ValueError(f"the data has a 'time' dimension as well as {dim!r}: drop or select one of them")
    coord = np.asarray(data[dim].values) if dim in data.coords else np.arange(data.sizes[dim])
    return data.rename({dim: "time"}).drop_vars("time", errors="ignore"), dim, coord


def _restore(result: xr.DataArray, dim: Optional[str], coord: Any) -> xr.DataArray:
    """``result`` with its ``time`` axis named and labelled as the axis ``_along`` took it from."""
    if dim is None:
        return result
    return result.rename(time=dim).assign_coords({dim: coord})


def _standardize_block(block, method, ddof, n_jobs, dtype):
    from . import _core

    t, ny, nx = block.shape
    if block.size == 0:
        return np.full(block.shape, np.nan, dtype=dtype)
    out = _core.tsa.standardize_batch(_flat(block), method, ddof, n_jobs)  # (pixels, time)
    return np.ascontiguousarray(out.T, dtype=dtype).reshape(t, ny, nx)


def standardize(data: Any, method: str = "normalize", *, ddof: int = 1, dim: Optional[str] = None,
                dates: Optional[Sequence[Any]] = None, band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
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
    dim
        The axis to standardize along: ``"time"`` (default when it exists) or another axis, e.g.
        ``"year"`` of ``zeit.temporal_metrics(..., by="year")`` (the fold of FORCE's
        ``STANDARDIZE_FOLD``). By default the only axis that is not ``y``, ``x``, ``band`` or
        ``metric``. A ``metric`` axis of length 1 is dropped; with more, choose one with
        ``.sel(metric=...)``.
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
    data, axis, labels = _along(data, dim)
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
    return _restore(result, axis, labels)


_METHODS = ("rbf", "linear", "moving", "harmonic")
_DAY_NS = 86400 * 10**9


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.values.astype("datetime64[ns]").astype(np.int64))


def _interp_block(block, t_in, t_out, kwargs, n_jobs, dtype):
    from . import _core

    t, ny, nx = block.shape
    n_out = len(t_out)
    if block.size == 0 or n_out == 0:
        return np.full((n_out, ny, nx), np.nan, dtype=dtype)
    out = _core.tsa.interp_batch(_flat(block), t_in, t_out, n_jobs=n_jobs, **kwargs)  # (pixels, n_out)
    return np.ascontiguousarray(out.T, dtype=dtype).reshape(n_out, ny, nx)


def interpolate(data: Any, method: str = "rbf", *, step: int = 16, start: Any = None, end: Any = None,
                times: Optional[Sequence[Any]] = None, sigma: Sequence[float] = (8, 16, 32), cutoff: float = 0.95,
                window: float = 16, modes: int = 3, trend: bool = True, fit: Optional[Sequence[Any]] = None,
                period: float = 365.25, dates: Optional[Sequence[Any]] = None, band: Union[str, int, None] = None,
                nodata: Union[float, str, None] = "auto", chunks: Any = None, n_jobs: int = -1) -> Any:
    """Interpolates each pixel's series to a regular grid of dates, or to any dates (FORCE's TSI).

    Parameters
    ----------
    data
        As for ``zeit.temporal_metrics``; the dates are needed and must not decrease. NaN marks a
        missing observation (so do the NoData value and, for integer data without one, 0: see
        ``nodata``).
    method
        ``"rbf"`` (default): an ensemble of Gaussian kernels of standard deviations ``sigma``
        (days), each cut where its mass on either side of the date reaches ``cutoff / 2`` and
        weighted by the data it finds around each date (FORCE's RBF, after Rabe's);
        ``"linear"``: straight lines between the valid observations on either side, constant
        beyond the first and last; ``"moving"``: the mean of the valid observations within
        ``window`` days (closed, |t - x| <= ``window``); ``"harmonic"``:
        ``a0 [+ a1 t] + sum_m b_m cos(2 pi m t / period) + c_m sin(...)`` for ``m`` up to ``modes``
        (1-3), fit to the valid observations within ``fit`` by iteratively reweighted least squares
        with the bisquare weights (tuning 4.685, MAD scale: the algorithm of GSL's
        ``gsl_multifit_robust``), and predicted at every new date, also beyond the fit.
    step, start, end
        The regular grid, as FORCE's: from ``start`` (default the first date, at midnight) every
        ``step`` days, ``floor((end - start + 1) / step)`` dates, ``end`` being the last date by
        default. No date is made after ``end``.
    times
        Any dates for the result, instead of the grid (``step``, ``start`` and ``end`` are then
        not allowed).
    sigma, cutoff
        ``rbf``: the standard deviations in days, increasing, and the share of each kernel's mass
        kept (0-1). A date with no valid observation within the largest kernel is NaN.
    window
        ``moving``: the half-width in days.
    modes, trend, fit, period
        ``harmonic``: the number of harmonics, an offset-and-slope term, the fit window ``(first,
        last)`` as dates (inclusive; default all the series), and the period in days (the default
        365.25 is the real year; FORCE counts 365 days on a calendar without leap days). A pixel
        with fewer valid observations in ``fit`` than coefficients is NaN. The time is centered
        and scaled inside, which does not change the fit.
    dates, band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.DataArray (or pandas.Series for a Series)
        ``(time, y, x)`` with the new dates, the coordinates and georeferencing of the input and
        ``attrs`` with the method and its parameters. Float32 for float32 input, else float64;
        the values are not rounded.

    Examples
    --------
    >>> tsi = zeit.interpolate(ndvi, "rbf", step=16)                              # FORCE's default grid
    >>> daily = zeit.interpolate(ndvi, "linear", times=pd.date_range("2020", "2021", freq="D"))
    >>> nrt = zeit.interpolate(ndvi, "harmonic", fit=("2018-01-01", "2020-12-31"), end="2022-12-31")
    """
    from ._embeddings import refuse
    from ._series_api import _as_dask, _series_cube

    refuse(data, "zeit.interpolate")
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, need_dates=True, nodata=nodata)
    index = pd.DatetimeIndex(cube.time.values)
    if len(index) == 0 or index.isna().any():
        raise ValueError("interpolate needs a date for every step")
    if (np.diff(_ns(index)) < 0).any():
        raise ValueError("interpolate needs dates that do not decrease")

    if times is not None:
        if start is not None or end is not None:
            raise ValueError("pass either times= or start=/end=/step=, not both")
        new = pd.DatetimeIndex(pd.to_datetime(np.asarray(times)))
        if new.isna().any():
            raise ValueError("times has a missing date")
    else:
        step = int(step)
        if step < 1:
            raise ValueError(f"step must be a positive number of days, got {step}")
        first = pd.Timestamp(start) if start is not None else index[0]
        last = pd.Timestamp(end) if end is not None else index[-1]
        first, last = first.tz_localize(None).normalize(), last.tz_localize(None).normalize()
        n = (int((last - first).days) + 1) // step
        if n < 1:
            raise ValueError(f"no date in the grid: start {first.date()}, end {last.date()}, step {step}")
        new = first + pd.to_timedelta(np.arange(n) * step, unit="D")

    origin = _ns(index)[0]
    t_in = (_ns(index) - origin) / _DAY_NS
    t_out = (_ns(new) - origin) / _DAY_NS
    params = {"method": method}
    if method == "rbf":
        sig = [float(x) for x in sigma]
        if not sig or any(x <= 0 for x in sig) or any(b < a for a, b in zip(sig, sig[1:])):
            raise ValueError(f"sigma must be positive and increasing, got {tuple(sigma)}")
        if not 0 < cutoff <= 1:
            raise ValueError(f"cutoff must be in (0, 1], got {cutoff}")
        params.update(sigma=sig, cutoff=float(cutoff))
    elif method == "moving":
        if window < 0:
            raise ValueError(f"window must not be negative, got {window}")
        params.update(window=float(window))
    elif method == "harmonic":
        if modes not in (1, 2, 3):
            raise ValueError(f"modes must be 1, 2 or 3, got {modes}")
        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        lo, hi = -np.inf, np.inf
        if fit is not None:
            if len(fit) != 2:
                raise ValueError("fit must be (first date, last date)")
            f0, f1 = (_ns(pd.DatetimeIndex([pd.Timestamp(x)]))[0] for x in fit)
            lo, hi = (f0 - origin) / _DAY_NS, (f1 - origin) / _DAY_NS
            if hi < lo:
                raise ValueError("fit must be (first date, last date), in order")
        params.update(modes=int(modes), trend=bool(trend), fit_lo=float(lo), fit_hi=float(hi), period=float(period))
    attrs = {k: v for k, v in params.items() if k not in ("fit_lo", "fit_hi")}
    if method == "harmonic" and fit is not None:
        attrs["fit"] = [str(pd.Timestamp(x).date()) for x in fit]
    if times is None:
        attrs["step"] = step

    threads, dtype = resolve(n_jobs), _work_dtype(cube)
    if cube.chunks is not None:
        import dask.array as da

        arr, _ = _as_dask(cube)
        chunks_out = ((len(new),),) + arr.chunks[1:]
        out = da.map_blocks(_interp_block, arr, t_in, t_out, params, threads, dtype, dtype=dtype, chunks=chunks_out,
                            meta=np.empty((0, 0, 0), dtype))
    else:
        out = _interp_block(np.asarray(cube.values), t_in, t_out, params, threads, dtype)

    coords = {"time": new}
    if not pixel:
        for name in ("y", "x", "spatial_ref"):
            if name in cube.coords:
                coords[name] = cube.coords[name]
    result = xr.DataArray(out, dims=("time", "y", "x"), coords=coords, name=cube.name,
                          attrs={**cube.attrs, "algorithm": "interpolate", **attrs})
    if pixel:
        if isinstance(data, pd.Series):
            return pd.Series(np.asarray(result.values).reshape(-1), index=new, name=data.name)
        return result.squeeze(("y", "x"), drop=True)
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    return result


def _cso_block(block, t, windows, names, n_jobs, dtype):
    from . import _core

    _, ny, nx = block.shape
    if block.size == 0:
        return np.full((len(names), len(windows), ny, nx), np.nan, dtype=dtype)
    out = _core.tsa.cso_batch(_flat(block), t, windows, names, n_jobs)  # (pixels, metrics, windows)
    return np.ascontiguousarray(out.transpose(1, 2, 0), dtype=dtype).reshape(len(names), len(windows), ny, nx)


def _boolean_as_nan(data: Any) -> Any:
    """A boolean cube (True: a clear observation) as float, NaN where False; other data as it is."""
    if isinstance(data, (list, tuple)):
        arr = np.asarray(data)
        return arr if arr.dtype != bool else np.where(arr, 1.0, np.nan)
    if isinstance(data, np.ndarray) and data.dtype == bool:
        return np.where(data, 1.0, np.nan)
    if isinstance(data, (pd.Series, xr.DataArray)) and data.dtype == bool:
        return data.where(data)
    return data


def clear_observations(data: Any, *, months: int = 1, metrics: Sequence[str] = ("count", "mean", "max"),
                       start: Any = None, end: Any = None, dates: Optional[Sequence[Any]] = None,
                       band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto",
                       chunks: Any = None, n_jobs: int = -1) -> xr.DataArray:
    """The clear-sky observations of each pixel in windows of months, and the gaps between them
    (FORCE's CSO: NUM, MIN, MAX, RNG, IQR, AVG, STD, SKW, KRT and the quantiles).

    An observation is valid where the value is finite (NaN, and the NoData value, are cloud or
    missing), or True in a boolean cube. In each window the intervals are, in days: from the start
    of the window to the first valid observation, between consecutive valid observations, and from
    the last one to the end of the window (the first day of the next window). Observations on the
    same day make no interval. A window without a valid observation has one interval, as long as
    the window, and a count of 0.

    Parameters
    ----------
    data
        A ``(time, y, x)`` cube (in memory or dask) of any values, whose valid observations are the
        finite ones, or a boolean cube (True: clear); anything ``zeit.load_raster`` reads, a numpy
        ``(time, y, x)`` with ``dates``, or one pixel's series. The dates are needed and must not
        decrease.
    months
        The length of a window in months, 1 to 12.
    metrics
        ``count`` (the valid observations), and of the intervals in days: ``min``, ``max``,
        ``range``, ``iqr``, ``mean``, ``std``, ``skew``, ``kurtosis``, ``median`` and ``pNN``
        (as for ``zeit.temporal_metrics``). ``std`` has ``k - 1`` degrees of freedom for ``k``
        intervals (NaN for one).
    start, end
        The windows begin on the first day of the month of ``start`` (default the first date)
        and go on until the one that holds ``end`` (default the last date), which is included
        whole: observations of that window after ``end``, and of the first before ``start``, are
        counted. FORCE leaves out the last window when it starts in the month of its range's end
        (a range ending in December loses the December window, with 1 month windows); here every
        window is made.
    dates, band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.DataArray
        ``(metric, window, y, x)``, with ``window`` the first day of each window and ``window_end``
        (along ``window``) its end (exclusive). Float32 for float32 input, else float64; the
        values are not rounded.

        Unlike FORCE, which writes int16 and mixes the number of observations ``n`` with the number
        of intervals ``k`` (the standard deviation divides by ``n - 1``, the skewness and kurtosis
        by ``n``, the quantiles are taken over the first ``n`` of the ``k`` intervals and, with
        observations on the same day, read stale memory), every metric but ``count`` is that of the
        same ``k`` intervals. ``count``, ``min``, ``max``, ``range`` and ``mean`` equal FORCE's
        (once truncated); the others differ where ``n != k``. In a window without a valid
        observation FORCE writes 0 for ``std``, ``skew``, ``kurtosis`` and the quantiles; here
        ``std``, ``skew`` and ``kurtosis`` are NaN and the quantiles the length of the window.

    Examples
    --------
    >>> cso = zeit.clear_observations(ndvi, months=3, metrics=["count", "max"])   # (metric, window, y, x)
    >>> longest_gap = zeit.clear_observations(ndvi.notnull(), metrics=["max"])    # a boolean cube works too
    """
    from ._embeddings import refuse
    from ._series_api import _as_dask, _series_cube
    from .indices import parse_metrics

    refuse(data, "zeit.clear_observations")
    if isinstance(months, bool) or int(months) != months or not 1 <= int(months) <= 12:
        raise ValueError(f"months must be an integer from 1 to 12, got {months!r}")
    months = int(months)
    names = parse_metrics(metrics)
    cube, pixel = _series_cube(_boolean_as_nan(data), dates=dates, band=band, chunks=chunks, need_dates=True,
                               nodata=nodata)
    index = pd.DatetimeIndex(cube.time.values)
    if len(index) == 0 or index.isna().any():
        raise ValueError("clear_observations needs a date for every step")
    if (np.diff(_ns(index)) < 0).any():
        raise ValueError("clear_observations needs dates that do not decrease")
    first = pd.Timestamp(start) if start is not None else index[0]
    last = pd.Timestamp(end) if end is not None else index[-1]
    first, last = first.tz_localize(None).normalize(), last.tz_localize(None).normalize()
    if last < first:
        raise ValueError(f"end {last.date()} is before start {first.date()}")
    month0 = first.year * 12 + first.month - 1
    n_windows = (last.year * 12 + last.month - 1 - month0) // months + 1
    bounds = month0 + months * np.arange(n_windows + 1)
    edges = pd.DatetimeIndex([pd.Timestamp(int(b) // 12, int(b) % 12 + 1, 1) for b in bounds])
    t = _ns(index) / _DAY_NS
    windows = np.stack([_ns(edges[:-1]), _ns(edges[1:])], axis=1) / _DAY_NS

    threads, dtype = resolve(n_jobs), _work_dtype(cube)
    if cube.chunks is not None:
        import dask.array as da

        arr, _ = _as_dask(cube)
        chunks_out = ((len(names),), (n_windows,)) + arr.chunks[1:]
        out = da.map_blocks(_cso_block, arr, t, windows, names, threads, dtype, dtype=dtype, drop_axis=[0],
                            new_axis=[0, 1], chunks=chunks_out, meta=np.empty((0,) * 4, dtype))
    else:
        out = _cso_block(np.asarray(cube.values), t, windows, names, threads, dtype)

    coords = {"metric": names, "window": edges[:-1], "window_end": ("window", edges[1:])}
    if not pixel:
        for name in ("y", "x", "spatial_ref"):
            if name in cube.coords:
                coords[name] = cube.coords[name]
    result = xr.DataArray(out, dims=("metric", "window", "y", "x"), coords=coords, name="clear_observations",
                          attrs={"algorithm": "clear_observations", "months": months})
    if pixel:
        return result.squeeze(("y", "x"), drop=True)
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    return result


_TAILS = ("two", "left", "right")
TREND_NAMES = ["mean", "offset", "slope", "pct_gain", "abs_gain", "rsq", "sig", "rmse", "mae", "maxe", "num", "len"]
CAT_NAMES = (["change", "loss", "change_index"]
             + [f"{part}_{name}" for part in ("total", "before", "after") for name in TREND_NAMES])


def _trend_block(block, conf, tail, n_jobs, dtype):
    from . import _core

    t, ny, nx = block.shape
    if block.size == 0:
        return np.full((len(TREND_NAMES), ny, nx), np.nan, dtype=dtype)
    out = _core.tsa.trend_batch(_flat(block), conf, tail, n_jobs)  # (pixels, 12)
    return np.ascontiguousarray(out.T, dtype=dtype).reshape(len(TREND_NAMES), ny, nx)


def _cat_block(block, penalty, conf, tail, n_jobs, dtype):
    from . import _core

    t, ny, nx = block.shape
    if block.size == 0:
        return np.full((len(CAT_NAMES), ny, nx), np.nan, dtype=dtype)
    out = _core.tsa.cat_batch(_flat(block), penalty, conf, tail, n_jobs)  # (pixels, 39)
    return np.ascontiguousarray(out.T, dtype=dtype).reshape(len(CAT_NAMES), ny, nx)


def _check_trend(conf: float, tail: str) -> str:
    if not 0 < conf < 1:
        raise ValueError(f"conf must be between 0 and 1, got {conf}")
    tail = str(tail).lower()
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
    return tail


def _run_block(fn, cube, args, names, dtype):
    """``fn(block, *args, dtype)`` on a (time, y, x) cube, lazily on dask: ``(len(names), y, x)``."""
    from ._series_api import _as_dask

    if cube.chunks is None:
        return fn(np.asarray(cube.values), *args, dtype), False
    import dask.array as da

    arr, _ = _as_dask(cube)
    chunks_out = ((len(names),),) + arr.chunks[1:]
    out = da.map_blocks(fn, arr, *args, dtype, dtype=dtype, drop_axis=[0], new_axis=[0], chunks=chunks_out,
                        meta=np.empty((0, 0, 0), dtype))
    return out, True


def linear_trend(data: Any, *, conf: float = 0.95, tail: str = "two", dim: Optional[str] = None,
                 band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
                 n_jobs: int = -1) -> xr.Dataset:
    """The linear trend of each pixel's series (FORCE's TSA trend: TRY, TRQ, TRM, TRW and TRD).

    A least-squares line on the position of each step along ``dim`` (x = 0, 1, 2... counting the
    missing steps too, so the slope is per step: per year for a yearly series), with a t-test of
    the slope. A pixel with fewer than 3 valid steps is NaN.

    Parameters
    ----------
    data
        A ``(time, y, x)`` cube (in memory or dask), anything ``zeit.load_raster`` reads, a numpy
        ``(time, y, x)``, one pixel's series, or a series along another axis such as the
        ``zeit.temporal_metrics(..., by="year")`` result ``(metric, year, y, x)``. NaN marks a
        missing observation.
    conf
        Confidence level of the test of the slope (0.95: significance level 0.05).
    tail
        ``"two"`` (default), ``"left"`` (is the slope significantly negative?) or ``"right"``.
        The critical value is the exact quantile of Student's t with ``num - 2`` degrees of freedom
        (FORCE iterates Hill's approximation to a tolerance of 1e-4, so ``sig`` can differ from it
        where the t-value is that close to the limit).
    dim
        The axis of the series: ``"time"`` (default when it exists), or another, e.g. ``"year"``.
        By default the only axis that is not ``y``, ``x``, ``band`` or ``metric``. A ``metric`` axis
        of length 1 is dropped; with more, choose one with ``.sel(metric=...)``.
    band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        ``mean``, ``offset`` (the fit at the first step), ``slope`` (per step), ``pct_gain`` (the
        slope times the number of steps as a percentage of the offset, 0 unless the offset is
        positive), ``abs_gain`` (the slope times the steps), ``rsq`` (0-1; FORCE writes it times
        10000), ``sig`` (-1, 0 or 1), ``rmse``, ``mae``, ``maxe`` (the largest absolute residual),
        ``num`` (valid steps) and ``len`` (steps).

    Examples
    --------
    >>> yearly = zeit.temporal_metrics(ndvi, ["mean"], by="year")
    >>> trend = zeit.linear_trend(yearly)                    # slope in NDVI per year
    """
    from ._embeddings import refuse
    from ._series_api import _metrics_dataset, _series_cube

    refuse(data, "zeit.linear_trend")
    tail = _check_trend(conf, tail)
    data, _, _ = _along(data, dim)
    cube, pixel = _series_cube(data, dates=None, band=band, chunks=chunks, need_dates=False, nodata=nodata)
    dtype = _work_dtype(cube)
    out, lazy = _run_block(_trend_block, cube, (float(conf), tail, resolve(n_jobs)), TREND_NAMES, dtype)
    return _metrics_dataset(out, TREND_NAMES, cube, pixel, lazy,
                            dict(algorithm="linear_trend", conf=float(conf), tail=tail))


def cat(data: Any, *, penalty: bool = False, conf: float = 0.95, tail: str = "two", dim: Optional[str] = None,
        band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
        n_jobs: int = -1) -> xr.Dataset:
    """The greatest change of each pixel's series and the trend around it (FORCE's CAT: CAY, CAQ, CAM...).

    The change is the greatest loss between a valid step and the valid one before it (the previous
    value less this one: positive for a loss; a series that only rises keeps its smallest rise). The
    series is then split at that step and the linear trend of ``zeit.linear_trend`` (x the step along
    ``dim``) is fit to the whole series (``total_``), to the steps before the change (``before_``:
    from the first up to two before the change step, as FORCE defines it) and from the change step
    on (``after_``). A part without valid steps is NaN; one with fewer than 3 has only its ``mean``,
    ``num`` and ``len``. ``offset`` is the fit at the first step of the part.

    Parameters
    ----------
    data, conf, tail, dim, band, nodata, chunks, n_jobs
        As for ``zeit.linear_trend``.
    penalty
        Weight each loss by what the series has lost by the next valid step (FORCE's
        ``CHANGE_PENALTY``): the loss times (the value before it less the next value) / 10000, which
        discounts a dip the series recovers from; the 10000 is FORCE's scale of values times 10000,
        so for other scales it only matters through the sign. A loss without a valid step after it
        is not considered. FORCE compares the penalized value of each candidate with the
        *unpenalized* change of the best one so far, mixing two scales; here the penalized values are
        compared with each other, and the winner's unpenalized change is what is returned (as in
        FORCE). The two can pick different steps.

    Returns
    -------
    xarray.Dataset
        ``change`` (the greatest loss), ``loss`` (the change as a percentage of ``total_offset``,
        0 unless that is positive), ``change_time`` (the coordinate of ``dim`` at the change: the
        date, the year, the month..., NaN or NaT when there is none) and the twelve variables of
        ``zeit.linear_trend`` for each part, prefixed ``total_``, ``before_`` and ``after_``. A pixel
        with fewer than 2 valid steps is NaN.

    Examples
    --------
    >>> yearly = zeit.temporal_metrics(ndvi, ["mean"], by="year")
    >>> ch = zeit.cat(yearly)                                # ch.change_time: the year of the greatest loss
    """
    from ._embeddings import refuse
    from ._series_api import _metrics_dataset, _series_cube

    refuse(data, "zeit.cat")
    tail = _check_trend(conf, tail)
    data, axis, labels = _along(data, dim)
    cube, pixel = _series_cube(data, dates=None, band=band, chunks=chunks, need_dates=False, nodata=nodata)
    if axis is None and "time" in cube.coords:
        labels = np.asarray(cube.time.values)
    n = cube.sizes["time"]
    dtype = _work_dtype(cube)
    out, lazy = _run_block(_cat_block, cube, (bool(penalty), float(conf), tail, resolve(n_jobs)), CAT_NAMES, dtype)
    ds = _metrics_dataset(out, CAT_NAMES, cube, pixel, lazy, dict(
        algorithm="cat", penalty=int(bool(penalty)), conf=float(conf), tail=tail))

    if labels is None:
        labels = np.arange(n)
    labels = np.asarray(labels)
    if np.issubdtype(labels.dtype, np.datetime64):
        labels, empty = labels.astype("datetime64[ns]"), np.datetime64("NaT", "ns")
    elif np.issubdtype(labels.dtype, np.number) or np.issubdtype(labels.dtype, np.bool_):
        labels, empty = labels.astype(np.float64), np.nan
    else:
        labels, empty = labels.astype(object), None

    def _take(idx):
        idx = np.asarray(idx, dtype=np.float64)
        ok = np.isfinite(idx)
        picked = labels[np.where(ok, idx, 0).astype(np.int64)]
        return np.where(ok, picked, empty)

    ds["change_time"] = xr.apply_ufunc(_take, ds["change_index"], dask="parallelized",
                                       output_dtypes=[labels.dtype])
    ds = ds.drop_vars("change_index")
    return ds[["change", "loss", "change_time"] + CAT_NAMES[3:]]
