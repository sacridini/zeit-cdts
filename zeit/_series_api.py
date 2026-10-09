"""BFAST, Mann-Kendall and phenology: one function each, for every kind of input.

Each takes a pixel's series, a numpy stack, a georeferenced ``(time, y, x)`` cube (in
memory or dask) or a raster on disk, reads the time axis from the cube's dates, and
returns an ``xarray.Dataset`` with one variable per metric, georeferenced when the input
is. In-memory input gives an in-memory result; dask input stays lazy.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._time import fractional_years, to_datetime_index


# ---------------------------------------------------------------------------
# Input and output
# ---------------------------------------------------------------------------

def _missing_as_nan(cube: xr.DataArray, nodata: Any) -> xr.DataArray:
    """Float cube with NaN for missing observations: the raster's NoData (``"auto"``; for
    integer data without one, 0, as Earth Engine exports masked pixels), a number, or None."""
    if isinstance(nodata, str) and nodata != "auto":
        raise ValueError(f"nodata must be 'auto', a number or None, got {nodata!r}")
    values = []
    if nodata == "auto":
        stored = cube.rio.nodata
        if stored is not None and not np.isnan(stored):
            values = [stored]
        elif np.issubdtype(cube.dtype, np.integer):
            values = [0]
    elif nodata is not None:
        values = [nodata]
    out = cube if np.issubdtype(cube.dtype, np.floating) else cube.astype(np.float64)
    for value in values:
        out = out.where(cube != value)
    return out


def _series_cube(data: Any, *, dates: Any, band: Any, chunks: Any, need_dates: bool = True,
                 nodata: Any = "auto"):
    """The input as a (time, y, x) cube (dates in ``time`` when known) and True for one pixel."""
    from ._load import load_raster

    if isinstance(data, pd.Series):
        if dates is None and isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        return _missing_as_nan(_pixel(data.to_numpy(dtype=float), dates, need_dates), nodata), True
    if isinstance(data, (list, tuple)) or (isinstance(data, np.ndarray) and data.ndim == 1):
        return _missing_as_nan(_pixel(np.asarray(data, dtype=float), dates, need_dates), nodata), True
    if isinstance(data, xr.DataArray) and data.ndim == 1:
        if dates is None and "time" in data.coords:
            dates = data.time.values
        return _missing_as_nan(_pixel(np.asarray(data.values, dtype=float), dates, need_dates), nodata), True

    kwargs = {}
    if not isinstance(data, (xr.DataArray, xr.Dataset, np.ndarray)) and not hasattr(data, "dask"):
        kwargs["chunks"] = chunks
    if isinstance(data, xr.Dataset):
        if band is None:
            names = [v for v in data.data_vars if data[v].ndim >= 3]
            if len(names) != 1:
                raise ValueError(f"the Dataset has the variables {names}; choose one with band=")
            band = names[0]
        data, band = data[band], None
    cube = load_raster(data, dates=dates, **kwargs)
    if chunks is not None and cube.chunks is None and isinstance(data, (xr.DataArray, np.ndarray)):
        cube = cube.chunk(chunks)
    if "band" in cube.dims and "time" in cube.dims:
        if band is None:
            if cube.sizes["band"] != 1:
                raise ValueError(f"the cube has {cube.sizes['band']} bands {list(np.atleast_1d(cube.band.values))}; "
                                 "choose the index with band=")
            cube = cube.isel(band=0, drop=True)
        else:
            cube = cube.sel(band=band, drop=True)
    if "time" not in cube.dims and "band" in cube.dims and not need_dates:
        cube = cube.drop_vars("band", errors="ignore").rename(band="time")
    if "time" not in cube.dims:
        raise ValueError("no dates in the data: pass dates= (or load it with zeit.load_raster(..., dates=...))")
    if need_dates and "time" not in cube.coords:
        raise ValueError("the time axis has no dates: pass dates=")
    extra = [d for d in cube.dims if d not in ("time", "y", "x")]
    if extra:
        raise ValueError(f"expected one index (time, y, x); the cube also has {extra}")
    return _missing_as_nan(cube.transpose("time", "y", "x"), nodata), False


def _pixel(values: np.ndarray, dates: Any, need_dates: bool) -> xr.DataArray:
    coords = {}
    if dates is not None:
        times = to_datetime_index(dates)
        if len(times) != len(values):
            raise ValueError(f"{len(times)} dates for {len(values)} values")
        coords["time"] = times
    elif need_dates:
        raise ValueError("one pixel's series needs dates= (or a pandas Series indexed by dates)")
    return xr.DataArray(values.reshape(-1, 1, 1), dims=("time", "y", "x"), coords=coords)


def _as_dask(cube: xr.DataArray):
    import dask.array as da

    if cube.chunks is not None:
        return cube.data.rechunk({0: -1}), True
    return da.from_array(np.asarray(cube.values), chunks=(-1, "auto", "auto")), False


def _metrics_dataset(out: Any, names: List[str], cube: xr.DataArray, pixel: bool, lazy: bool,
                     attrs: Dict[str, Any], extra_dims: Tuple[str, ...] = (), extra_coords: Optional[dict] = None
                     ) -> xr.Dataset:
    if not lazy:
        out = np.asarray(out.compute() if hasattr(out, "compute") else out)
    dims = extra_dims + ("y", "x")
    variables = {name: (dims, out[i]) for i, name in enumerate(names)}
    coords = dict(extra_coords or {})
    if not pixel:
        for name in ("y", "x", "spatial_ref"):
            if name in cube.coords:
                coords[name] = cube.coords[name]
    ds = xr.Dataset(variables, coords=coords, attrs={k: v for k, v in attrs.items() if v is not None})
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    if cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
    return ds


def _regular_time(cube: xr.DataArray, start_time: Optional[float], frequency: Optional[int]) -> Tuple[float, int]:
    """``ts``-style regular time axis (``start_time + i / frequency``) from the cube's dates."""
    if frequency is None or start_time is None:
        if "time" not in cube.coords:
            raise ValueError("no dates in the data: pass dates=, or start_time= and frequency=")
        times = pd.DatetimeIndex(cube.time.values)
        if frequency is None:
            if len(times) < 2:
                raise ValueError("a single date: pass frequency=")
            step = np.median(np.diff(times.values).astype("timedelta64[s]").astype(float)) / 86400.0
            frequency = max(1, int(round(365.25 / step)))
        if start_time is None:
            first = float(fractional_years(times[:1])[0])
            year = int(np.floor(first))
            start_time = year + round((first - year) * frequency) / frequency
    return float(start_time), int(frequency)


def _as_time(value: Any) -> float:
    """A date or a decimal year, as a decimal year."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    return float(fractional_years(to_datetime_index([value]))[0])


def _jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        import os
        return max(1, (os.cpu_count() or 2) - 1)
    return int(n_jobs)


# ---------------------------------------------------------------------------
# BFAST family
# ---------------------------------------------------------------------------

_COMMON_DOC = """    data
        A ``(time, y, x)`` cube (in memory or dask), anything ``zeit.load_raster`` reads, a
        ``(time, band, y, x)`` cube or Dataset with ``band``, a numpy ``(time, y, x)`` with
        ``dates``, or one pixel's series (list/1-D array with ``dates``, or a ``pandas.Series``
        indexed by date). One observation per regular time step (e.g. 16-day composites);
        NaN marks a missing observation.
    dates
        Date of each time step, when the data has none.
    start_time, frequency
        The regular time axis of R's ``ts`` (``start_time + i / frequency``, decimal years).
        By default taken from the dates: ``frequency`` from their median spacing (23 for 16-day
        data, 12 monthly, 1 annual...) and ``start_time`` from the first date.
    band
        Index to use in a ``(time, band, y, x)`` cube or a Dataset.
    nodata
        Value marking a missing observation besides NaN: ``"auto"`` (default) the raster's
        NoData, or ``0`` for integer data without one; a number; or ``None``.
    chunks
        Inputs read from disk: ``None`` reads them into memory; ``"auto"`` or a dict keeps
        them lazy.
    n_jobs
        CPU threads (``-1``: all but one).
"""


def bfast_monitor(data: Any, monitor_start: Any, *, dates: Optional[Sequence[Any]] = None,
                  start_time: Optional[float] = None, frequency: Optional[int] = None, order: int = 3,
                  h: float = 0.25, period: int = 10, alpha: float = 0.05, min_valid: int = 10,
                  band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
                  n_jobs: int = -1) -> xr.Dataset:
    """Near-real-time disturbance monitoring (bfastmonitor, Verbesselt et al. 2012).

    Fits a trend + harmonic model on the history before ``monitor_start`` and flags the
    first date after it where the OLS-MOSUM process crosses its boundary.

    Parameters
    ----------
    monitor_start
        Start of the monitoring period: a date (``"2019-01-01"``) or a decimal year (2019.0).
    order, h, period, alpha, min_valid
        As in R's bfastmonitor (harmonic order; MOSUM window 0.25/0.5/1.0 of the history;
        monitoring period 2..10; significance; minimum valid history observations).
    <common>
    Returns
    -------
    xarray.Dataset
        ``breakpoint`` (decimal year of the break, NaN if none), ``breakpoint_idx``,
        ``magnitude``, ``sigma``, ``n_history``, ``has_break``, ``valid``.
    """
    from ._bfast import BFM_METRIC_NAMES, run_bfast_monitor_dask

    # With start_time and frequency given, the series needs no dates.
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, nodata=nodata,
                               need_dates=start_time is None or frequency is None)
    start_time, frequency = _regular_time(cube, start_time, frequency)
    arr, lazy = _as_dask(cube)
    out = run_bfast_monitor_dask(arr, start_time=start_time, monitor_start_time=_as_time(monitor_start),
                                 frequency=frequency, order=order, h=h, period=period, alpha=alpha,
                                 min_valid=min_valid, n_jobs=_jobs(n_jobs))
    return _metrics_dataset(out, BFM_METRIC_NAMES, cube, pixel, lazy, dict(
        algorithm="bfastmonitor", start_time=start_time, frequency=frequency,
        monitor_start=_as_time(monitor_start), order=order, h=h, period=period, alpha=alpha, min_valid=min_valid))


def bfast_lite(data: Any, *, dates: Optional[Sequence[Any]] = None, start_time: Optional[float] = None,
               frequency: Optional[int] = None, order: int = 3, h: float = 0.15, max_breaks: int = 5,
               min_valid: int = 20, band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto",
               chunks: Any = None, n_jobs: int = -1) -> xr.Dataset:
    """Multiple-breakpoint detection in one pass (bfastlite, Masiliūnas et al. 2021).

    Parameters
    ----------
    order, h, min_valid
        Harmonic order; minimum segment size as a fraction of the series; minimum valid
        observations.
    max_breaks
        Number of breakpoint indices reported per pixel.
    <common>
    Returns
    -------
    xarray.Dataset
        ``n_breaks``, ``rss``, ``lwz``, ``n_valid``, ``valid`` and ``breakpoint_idx_1`` ..
        ``breakpoint_idx_{max_breaks}`` (0-based indices into the pixel's valid observations,
        NaN past ``n_breaks``).
    """
    from ._bfast import bfl_metric_names, run_bfast_lite_dask

    # With start_time and frequency given, the series needs no dates.
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, nodata=nodata,
                               need_dates=start_time is None or frequency is None)
    start_time, frequency = _regular_time(cube, start_time, frequency)
    arr, lazy = _as_dask(cube)
    out = run_bfast_lite_dask(arr, start_time=start_time, frequency=frequency, order=order, h=h,
                              max_breaks_output=max_breaks, min_valid=min_valid, n_jobs=_jobs(n_jobs))
    return _metrics_dataset(out, bfl_metric_names(max_breaks), cube, pixel, lazy, dict(
        algorithm="bfastlite", start_time=start_time, frequency=frequency, order=order, h=h,
        max_breaks=max_breaks, min_valid=min_valid))


def bfast(data: Any, *, dates: Optional[Sequence[Any]] = None, start_time: Optional[float] = None,
          frequency: Optional[int] = None, order: int = 3, h: float = 0.15, max_breaks_trend: int = 5,
          max_breaks_season: int = 5, max_iter: int = 10, level: float = 0.05, min_valid: int = 20,
          band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
                  n_jobs: int = -1) -> xr.Dataset:
    """Classic iterative trend + season break detection (bfast, Verbesselt et al. 2010).

    Parameters
    ----------
    order, h, max_iter, level, min_valid
        Harmonic order; minimum segment size as a fraction of the valid observations;
        trend/season re-estimation iterations; significance of the structural-stability
        pre-check; minimum valid observations.
    max_breaks_trend, max_breaks_season
        Number of trend and season breakpoint indices reported per pixel.
    <common>
    Returns
    -------
    xarray.Dataset
        ``n_trend_breaks``, ``n_season_breaks``, ``magnitude``, ``break_time`` (decimal year of the
        largest trend break), ``n_iter``, ``n_valid``, ``valid``, ``trend_breakpoint_idx_*`` and
        ``season_breakpoint_idx_*`` (0-based indices into the valid observations).
    """
    from ._bfast import bf_metric_names, run_bfast_dask

    # With start_time and frequency given, the series needs no dates.
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, nodata=nodata,
                               need_dates=start_time is None or frequency is None)
    start_time, frequency = _regular_time(cube, start_time, frequency)
    arr, lazy = _as_dask(cube)
    out = run_bfast_dask(arr, start_time=start_time, frequency=frequency, order=order, h=h,
                         max_breaks_trend=max_breaks_trend, max_breaks_season=max_breaks_season,
                         max_iter=max_iter, level=level, min_valid=min_valid, n_jobs=_jobs(n_jobs))
    names = bf_metric_names(max_breaks_trend, max_breaks_season)
    names = ["break_time" if n == "time" else n for n in names]  # "time" would clash with the time coordinate
    return _metrics_dataset(out, names, cube, pixel, lazy, dict(
        algorithm="bfast", start_time=start_time, frequency=frequency, order=order, h=h,
        max_breaks_trend=max_breaks_trend, max_breaks_season=max_breaks_season, max_iter=max_iter,
        level=level, min_valid=min_valid))


for _fn in (bfast_monitor, bfast_lite, bfast):
    _fn.__doc__ = _fn.__doc__.replace("    <common>\n", _COMMON_DOC + "\n")


# ---------------------------------------------------------------------------
# Mann-Kendall
# ---------------------------------------------------------------------------

def mann_kendall(data: Any, *, method: str = "hamed_rao", alpha: float = 0.05, lag: Optional[int] = None,
                 period: int = 1, min_valid: int = 4, dates: Optional[Sequence[Any]] = None,
                 band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
                  n_jobs: int = -1) -> xr.Dataset:
    """Mann-Kendall trend test and Theil-Sen slope, per pixel.

    Parameters
    ----------
    data
        As for the BFAST functions; dates are not needed (the test runs on the order of the
        observations).
    method
        ``"hamed_rao"`` (default, autocorrelation-corrected, for annual composites),
        ``"original"``, ``"yue_wang"`` or ``"seasonal"`` (pools ``period`` season slots,
        e.g. 23 for 16-day data).
    alpha, lag, period, min_valid
        Significance level; lags of the autocorrelation correction; season slots for
        ``"seasonal"``; minimum valid observations.
    dates, band, nodata, chunks, n_jobs
        As for the BFAST functions.

    Returns
    -------
    xarray.Dataset
        ``trend`` (1 increasing, -1 decreasing, 0 none), ``h``, ``p``, ``z``, ``tau``, ``s``,
        ``var_s``, ``slope`` (per time step; per ``period`` cycle for ``"seasonal"``) and
        ``intercept``.
    """
    from .trend import MK_METRIC_NAMES, run_mann_kendall_dask

    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, need_dates=False, nodata=nodata)
    arr, lazy = _as_dask(cube)
    out = run_mann_kendall_dask(arr, method=method, alpha=alpha, lag=lag, period=period, min_valid=min_valid,
                                n_jobs=_jobs(n_jobs))
    return _metrics_dataset(out, MK_METRIC_NAMES, cube, pixel, lazy, dict(
        algorithm="Mann-Kendall", method=method, alpha=alpha, lag=lag, period=period, min_valid=min_valid))


# ---------------------------------------------------------------------------
# Phenology
# ---------------------------------------------------------------------------

PHENOLOGY_METRICS = [
    "TRS2.sos", "TRS2.eos", "TRS5.sos", "TRS5.eos", "TRS6.sos", "TRS6.eos",
    "DER.sos", "DER.pos", "DER.eos",
    "UD", "SD", "DD", "RD",
    "Greenup", "Maturity", "Senescence", "Dormancy",
    "LOS", "POP",
    "R2", "RMSE",
]


def _enum(value: Union[str, int], enum: Any, what: str) -> int:
    if isinstance(value, str):
        members = {k.lower(): int(v) for k, v in enum.__members__.items()}
        if value.lower() not in members:
            raise ValueError(f"{what} must be one of {sorted(members)}, got {value!r}")
        return members[value.lower()]
    return int(value)


def phenology(data: Any, *, curve: Union[str, int] = "beck", method: Union[str, int] = "threshold",
              weights: Any = None, dates: Optional[Sequence[Any]] = None, annual: bool = True,
              max_seasons: Optional[int] = None, whittaker_lambda: float = 10.0, apply_whittaker: bool = True,
              apply_hants: bool = False, hants_frequencies: int = 3, hants_threshold: float = 0.1,
              min_season_length: int = 0, min_amplitude: float = 0.0, min_pixel_amplitude: float = 0.1,
              season_retry: bool = True, band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto",
              chunks: Any = None, n_jobs: int = -1) -> xr.Dataset:
    """Land surface phenology: fit a seasonal curve and extract its transition dates.

    Parameters
    ----------
    data
        As for the BFAST functions: a vegetation index series with dates (any spacing).
    curve
        Fitted curve: ``"beck"`` (default), ``"elmore"``, ``"gu"``, ``"klos"``, ``"zhang"``,
        ``"ag"`` or ``"dl"``.
    method
        Extraction method: ``"threshold"`` (default), ``"derivative"``, ``"gu"`` or
        ``"klosterman"``; every metric family is computed either way.
    weights
        Per-observation reliability weights in [0, 1], aligned with the data (e.g. from
        ``zeit.qc_sentinel2_scl``).
    annual
        ``True`` (default): one value per calendar year, dates as day of year; ``False``:
        one value per detected season, dates as days since January 1st of the first year.
    max_seasons
        Seasons (or years, with ``annual``) per pixel; by default the number of years.
    whittaker_lambda, apply_whittaker, apply_hants, hants_frequencies, hants_threshold,
    min_season_length, min_amplitude, min_pixel_amplitude, season_retry
        Smoothing and season-detection settings, as in the phenology tutorial.
    dates, band, nodata, chunks, n_jobs
        As for the BFAST functions.

    Returns
    -------
    xarray.Dataset
        One variable per metric (``TRS2.sos`` ... ``POP``, ``R2``, ``RMSE``), with dims
        ``(year, y, x)`` (``annual``) or ``(season, y, x)``.
    """
    from ._core.phenology import CurveType, ExtractionMethod
    from ._phenology import run_phenology_dask

    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, nodata=nodata)
    times = pd.DatetimeIndex(cube.time.values)
    base_year = int(times.year.min())
    n_years = int(times.year.max()) - base_year + 1
    seasons = int(max_seasons) if max_seasons is not None else n_years
    # Days since January 1st of the first year, 1-based: the C++ core's day numbering.
    day_numbers = ((times - pd.Timestamp(base_year, 1, 1)).days + 1).to_numpy(dtype=np.float64)

    arr, lazy = _as_dask(cube)
    weights_arr = None
    if weights is not None:
        import dask.array as da
        w = weights.data if isinstance(weights, xr.DataArray) else np.asarray(weights)
        if pixel:
            w = np.asarray(w).reshape(-1, 1, 1)
        weights_arr = w if hasattr(w, "dask") else da.from_array(w, chunks=arr.chunks)
        weights_arr = weights_arr.rechunk(arr.chunks)
    out = run_phenology_dask(arr, dates=day_numbers, curve_type=_enum(curve, CurveType, "curve"),
                             extraction_method=_enum(method, ExtractionMethod, "method"), max_seasons=seasons,
                             whittaker_lambda=whittaker_lambda, apply_whittaker=apply_whittaker,
                             apply_hants=apply_hants, hants_frequencies=hants_frequencies,
                             hants_threshold=hants_threshold, min_season_length=min_season_length,
                             min_amplitude=min_amplitude, min_pixel_amplitude=min_pixel_amplitude,
                             return_annual=annual, base_year=base_year, n_jobs=_jobs(n_jobs),
                             weights=weights_arr, season_retry=season_retry)
    dim = "year" if annual else "season"
    coord = np.arange(base_year, base_year + seasons) if annual else np.arange(1, seasons + 1)
    return _metrics_dataset(out, PHENOLOGY_METRICS, cube, pixel, lazy, dict(
        algorithm="phenology", curve=str(curve), method=str(method), base_year=base_year, annual=int(annual)),
        extra_dims=(dim,), extra_coords={dim: coord})
