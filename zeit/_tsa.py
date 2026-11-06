"""Time series analysis of a cube, per pixel: ``zeit.temporal_metrics`` (statistics over the whole
series or by year, quarter, month, week or day of the year), ``zeit.standardize`` and
``zeit.interpolate`` (linear, moving average, RBF ensemble or robust harmonic fit, to a regular
grid or to any dates).

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
