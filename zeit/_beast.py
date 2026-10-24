"""``zeit.beast``: BEAST, the Bayesian estimator of abrupt change, seasonal change and trend
(Zhao et al. 2019), a port of Rbeast's C code. The engine is ``_core.beast``, in Rust.
"""

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._load import sorted_by_time
from ._series_api import _series_cube
from ._time import fractional_years

def _pair(value: Sequence[int], name: str) -> Tuple[int, int]:
    v = tuple(int(x) for x in value)
    if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
        raise ValueError(f"{name} must be (min, max) with 0 <= min <= max, got {value!r}")
    return v


def beast(
    data: Any,
    *,
    season: str = "harmonic",
    period: Optional[float] = None,
    freq: Optional[float] = None,
    deltat: Optional[float] = None,
    start: Optional[float] = None,
    times: Any = None,
    scp_minmax: Sequence[int] = (0, 10),
    sorder_minmax: Sequence[int] = (0, 5),
    tcp_minmax: Sequence[int] = (0, 10),
    torder_minmax: Sequence[int] = (0, 1),
    sseg_minlength: Optional[int] = None,
    tseg_minlength: Optional[int] = None,
    samples: int = 8000,
    burnin: int = 200,
    chains: int = 3,
    thin: int = 5,
    seed: int = 42,
    prec_value: float = 1.5,
    prec_type: str = "componentwise",
    ci: bool = True,
    max_missing_rate: float = 0.75,
    dates: Any = None,
    band: Union[str, int, None] = None,
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """BEAST (Zhao et al. 2019): the changepoints of the season and of the trend of a series, and
    how sure of them to be, by Bayesian model averaging.

    Each series is decomposed into a piecewise harmonic season and a piecewise linear (or
    polynomial) trend. Instead of choosing one model, reversible-jump MCMC samples models of every
    number and position of changepoints and harmonic orders, in proportion to how well they fit
    and to their prior; the outputs average over them: the probability of a changepoint at each
    date (``trend_cp_occ``, ``season_cp_occ``), of each number of changepoints, the season and the
    trend with their 95% credible intervals, the trend's slope and the probability that it rises,
    the season's amplitude. The most probable changepoints are then listed (``trend_cp``) with their
    probabilities, the jump of the component there and their intervals.

    A port of Rbeast 1.0.2's C code (Kaiguang Zhao, GPL >= 2), its ``beast()`` on one series with
    the generic kernels (``cputype='sse'``): the same seed gives the same chain and the same numbers,
    to the last bit. Each pixel is fitted as ``beast()`` fits one series (from the seed, whatever
    the other pixels or the threads), not as ``beast123()``, which carries its random stream from
    one pixel to the next. Unlike Rbeast, the slope's interval is the mean over the chains, in the
    slope's units (Rbeast sums it over the chains, per time step), and ``ncp_mode``, ``ncp_median``
    and the percentiles are given with one chain too (Rbeast leaves them 0).

    An irregular series (dates missing, several in a step) is first put on a regular grid of step
    ``deltat``, each step the mean of the values in it, as ``beast.irreg()`` does.

    Parameters
    ----------
    data
        One index: a ``(time, y, x)`` cube, a raster, a Dataset with ``band=``, or one pixel's series
        (``pd.Series`` with dates, a 1-D DataArray, or values with ``dates=`` or ``times=``, or
        alone: a regular series from ``start`` every ``deltat``).
    season
        ``'harmonic'`` (a season of sines and cosines) or ``'none'`` (a trend alone).
    period
        The period of the season, in the units of the times: decimal years for dates (1, the
        default with dates), else needed for a harmonic season (12 for a monthly series counted
        in months...).
    freq
        The values per period instead (``period = freq * deltat``), Rbeast's older argument.
    deltat
        The step of the regular grid. Default: the step of the series when it is regular; else, for
        dates, the nearest of a day, half a month, a month and a week to the mean spacing (as
        Rbeast), and for ``times``, the period divided by the number of whole steps the mean
        spacing fits in it; without times, 1.
    start
        The time of the first value of the grid (default: the first time; 1 without times).
    times
        Numeric times of the values (any unit; ``period`` and ``deltat`` in the same), instead of
        dates.
    scp_minmax, tcp_minmax
        The least and most changepoints of the season and of the trend ((0, 10)).
    sorder_minmax
        The least and most harmonics of the season ((0, 5): from 1 to half the period less one).
    torder_minmax
        The least and most degree of the trend's polynomials ((0, 1): flat or linear segments).
    sseg_minlength, tseg_minlength
        The shortest segment of the season and of the trend, in steps (default: half a period, or
        3 without a season).
    samples, burnin, chains, thin
        The MCMC: the samples kept per chain (8000), the iterations left out first (200), the
        chains (3), and every how many iterations a sample is kept (5).
    seed
        The seed of the random stream (42; Rbeast's ``mcmc.seed``: the same seed, the same result).
    prec_value, prec_type
        The prior precision of the coefficients (1.5), and its model: ``'componentwise'`` (one for
        the season, one for the trend), ``'uniform'`` (one for all) or ``'constant'`` (fixed).
    ci
        The 95% credible intervals of the season, the trend and the slope (Rbeast's ``dump.ci``;
        they draw from the random stream, so the chain differs with and without them).
    max_missing_rate
        A series with more missing values than this fraction is not fitted (NaN; 0.75).
    dates, band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        On the regular ``time`` grid (decimal years for dates):

        - ``trend``, ``trend_sd``, ``trend_ci`` (``bound``: lower, upper); ``trend_slope``,
          ``trend_slope_sd``, ``trend_slope_ci``, ``trend_slope_pos_prob`` (the probability that
          the slope is positive) and ``trend_slope_zero_prob``; ``trend_order`` (the mean degree);
        - ``trend_cp_occ``: the probability of a trend changepoint at each date;
        - ``trend_ncp`` (the mean number of changepoints), ``trend_ncp_mode``,
          ``trend_ncp_median``, ``trend_ncp_pct10``, ``trend_ncp_pct90`` and ``trend_ncp_prob``
          (``tncp``: 0, 1, ... changepoints);
        - ``trend_cp`` (``tcp``: the changepoints found, most probable first, NaN after the last),
          ``trend_cp_prob``, ``trend_cp_abrupt`` (the jump of the trend there) and ``trend_cp_ci``;
        - the same for the season (``season``, ``season_cp_occ``...), with ``season_amp`` and
          ``season_amp_sd`` (the amplitude) and ``season_order`` (the mean number of harmonics), on
          ``sncp`` and ``scp``;
        - ``data`` (the series on the grid), ``marg_lik``, ``sig2`` (the noise variance), ``r2``
          and ``rmse`` of the fit.

    Examples
    --------
    >>> res = zeit.beast(ndvi_cube, deltat=1 / 23)               # 16-day composites
    >>> res.trend_ncp.plot()                                     # how many trend changes
    >>> res.trend_cp.isel(tcp=0).plot()                          # the most probable one

    References
    ----------
    Zhao, K., Wulder, M. A., Hu, T., Bright, R., Wu, Q., Qin, H., Li, Y., Toman, E., Mallick, B.,
    Zhang, X., & Brown, M. (2019). Detecting change-point, trend, and seasonality in satellite time
    series data to track abrupt changes and nonlinear dynamics: A Bayesian ensemble algorithm.
    Remote Sensing of Environment, 232, 111181.
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.beast")
    if season not in ("harmonic", "none"):
        raise ValueError(f"season must be 'harmonic' or 'none', got {season!r}")
    numeric = times is not None
    cube, pixel = _series_cube(data, dates=None if numeric else dates, band=band, chunks=chunks,
                               need_dates=False, nodata=nodata)
    if numeric:
        t = np.asarray(times, dtype=np.float64).ravel()
        if t.size != cube.sizes["time"]:
            raise ValueError(f"{t.size} times for {cube.sizes['time']} values")
    elif "time" in cube.coords and np.issubdtype(cube.time.dtype, np.datetime64):
        cube = sorted_by_time(cube)
        t = np.asarray(fractional_years(pd.DatetimeIndex(cube.time.values)), dtype=np.float64)
    else:
        t = None

    dt = float("nan") if deltat is None else float(deltat)
    if deltat is not None and not dt > 0:
        raise ValueError(f"deltat must be positive, got {deltat!r}")
    if period is None and freq is not None:
        if deltat is None:
            raise ValueError("freq needs deltat (period = freq * deltat)")
        period = float(freq) * dt
    if season == "harmonic":
        if period is None:
            if t is None or numeric:
                raise ValueError("a harmonic season needs period= (in the units of the times)")
            period = 1.0
        if not float(period) > 0:
            raise ValueError(f"period must be positive, got {period!r}")
    kw = dict(start=float("nan") if start is None else float(start), dt=dt, season=season,
              period=float("nan") if period is None else float(period),
              scp=_pair(scp_minmax, "scp_minmax"), sorder=_pair(sorder_minmax, "sorder_minmax"),
              tcp=_pair(tcp_minmax, "tcp_minmax"), torder=_pair(torder_minmax, "torder_minmax"),
              sseg_min=None if sseg_minlength is None else int(sseg_minlength),
              tseg_min=None if tseg_minlength is None else int(tseg_minlength),
              samples=int(samples), burnin=int(burnin), chains=int(chains), thin=int(thin), seed=int(seed),
              prec_value=float(prec_value), prec_type=str(prec_type), ci=bool(ci),
              max_missing_rate=float(max_missing_rate), decimal_years=t is not None and not numeric)
    for name in ("samples", "chains", "thin"):
        if kw[name] < 1:
            raise ValueError(f"{name} must be at least 1, got {kw[name]!r}")
    if kw["seed"] < 0:
        raise ValueError(f"seed must not be negative, got {seed!r}")
    n_raw = cube.sizes["time"]
    fields, (n, t0, step) = _core.beast.beast_layout(n_raw, t, **kw)
    width = sum(int(np.prod(shape)) for _, _, shape in fields)
    grid = t0 + step * np.arange(n, dtype=np.float64)
    attrs = dict(algorithm="BEAST", season=season, period=kw["period"], deltat=step, start=t0,
                 samples=kw["samples"], burnin=kw["burnin"], chains=kw["chains"], thin=kw["thin"],
                 seed=kw["seed"], prec_type=kw["prec_type"], prec_value=kw["prec_value"],
                 time_units="decimal years" if (t is not None and not numeric) else "as given")

    series = cube.transpose("y", "x", "time")
    jobs = int(n_jobs)

    def _block(v):   # (..., time) -> (..., width)
        shape = v.shape[:-1]
        flat = np.ascontiguousarray(v.reshape(-1, v.shape[-1]), dtype=np.float64)
        out = _core.beast.beast_batch(flat, t, n_jobs=jobs, **kw)
        return np.asarray(out).reshape(shape + (width,))

    if series.chunks is not None:
        series = series.chunk({"time": -1})
    out = xr.apply_ufunc(_block, series, input_core_dims=[["time"]], output_core_dims=[["row"]],
                         dask="parallelized", output_dtypes=[np.float32],
                         dask_gufunc_kwargs={"output_sizes": {"row": width}})

    ds = xr.Dataset()
    for name, off, shape in fields:
        size = int(np.prod(shape))
        part = out.isel(row=slice(off, off + size))
        comp = "t" if name.startswith("trend") else "s"
        if not shape:
            ds[name] = part.isel(row=0, drop=True)
            continue
        if name.endswith("_ncp_prob"):
            dim = comp + "ncp"
        elif name.endswith(("_cp", "_cp_prob", "_cp_abrupt", "_cp_ci")):
            dim = comp + "cp"
        else:
            dim = "time"
        if len(shape) == 1:
            ds[name] = part.rename(row=dim).transpose(dim, "y", "x")
            continue
        # Two rows: the core gives the intervals of the components as (upper, lower) and those of
        # the changepoints as (lower, upper); here both are (bound: lower, upper).
        inner = shape[1]
        first = part.isel(row=slice(0, inner)).rename(row=dim)
        second = part.isel(row=slice(inner, 2 * inner)).rename(row=dim)
        lower, upper = (first, second) if dim != "time" else (second, first)
        ds[name] = xr.concat([lower, upper], dim="bound").transpose("bound", dim, "y", "x")
    coords = {"time": grid}
    if "bound" in ds.dims:
        coords["bound"] = ["lower", "upper"]
    for dim in ("tncp", "sncp", "tcp", "scp"):
        if dim in ds.dims:
            coords[dim] = np.arange(ds.sizes[dim])
    ds = ds.assign_coords(coords)
    ds.attrs = attrs
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    for name in ("y", "x", "spatial_ref"):
        if name in cube.coords:
            ds = ds.assign_coords({name: cube.coords[name]})
    crs = getattr(cube.rio, "crs", None) if hasattr(cube, "rio") else None
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds
