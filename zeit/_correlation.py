"""``zeit.lagged_correlation`` and ``zeit.fdr``: the correlation of every pixel's series with a driver (a map
series or one series) at several lags, with a significance that accounts for the series' autocorrelation, and the
false discovery rate. The engine is ``_core.correlation``, in Rust.
"""

import re
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._eof import _crs_of, days_of
from ._load import sorted_by_time
from ._series_api import _series_cube
from ._threads import resolve

_ANOMALY = {"none": None, "mean": "mean", "month": "month", "harmonic": "harmonic"}
_MATCH = {"exact": "exact", "nearest": "nearest", "month": "month", "same_month": "month", "year": "year",
          "same_year": "year"}


def fdr(p: Any, alpha: float = 0.10) -> Tuple[np.ndarray, float, np.ndarray]:
    """Benjamini and Hochberg's (1995) false discovery rate procedure.

    Of the ``M`` p-values that are not NaN, the ones at or below the threshold ``p*`` (the largest ``p_(i)``
    with ``p_(i) <= i alpha / M``) are significant: among them, at most the fraction ``alpha`` is expected to be
    false discoveries when the tests are independent or positively dependent. Wilks (2016) recommends it for
    maps of tests, with ``alpha`` twice the level the single tests would have (a global ``0.10`` for ``0.05``).

    Parameters
    ----------
    p
        The p-values (any shape; a ``DataArray`` gives its coordinates back); NaN are left out.
    alpha
        The false discovery rate.

    Returns
    -------
    q, threshold, significant
        ``q``: the adjusted p-values (the smallest rate at which each one is significant), the shape of ``p``,
        NaN where ``p`` is; ``threshold``: ``p*`` (0 when nothing is significant); ``significant``: bool, the
        shape of ``p``.

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. JRSS B, 57, 289-300.
    Wilks, D. S. (2016). "The stippling shows statistically significant grid points": how research results are
    routinely overstated and overinterpreted. BAMS, 97, 2263-2273.
    """
    from . import _core

    arr = np.asarray(p, dtype=np.float64)
    q, star = _core.correlation.fdr(np.ascontiguousarray(arr.reshape(-1)), float(alpha))
    sig = np.isfinite(q) & (q <= alpha)
    q, sig = q.reshape(arr.shape), sig.reshape(arr.shape)
    if isinstance(p, xr.DataArray):
        q = p.copy(data=q)
        sig = p.copy(data=sig)
    return q, float(star), sig


def _days_of_tolerance(tol: Any) -> float:
    """A tolerance in days: a number (days), ``"8D"``, ``"12h"``, ``"2W"`` or a timedelta."""
    if isinstance(tol, (int, float, np.integer, np.floating)):
        return float(tol)
    if isinstance(tol, str):
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([dDwWhH])\s*", tol)
        if m is None:
            raise ValueError(f"tolerance must be a number of days or a string like '8D', got {tol!r}")
        return float(m.group(1)) * {"d": 1.0, "w": 7.0, "h": 1.0 / 24.0}[m.group(2).lower()]
    return float(np.timedelta64(tol, "ns") / np.timedelta64(1, "D"))


def _month_grid(days: np.ndarray) -> np.ndarray:
    """The months (year * 12 + month - 1 as ``datetime64[M]`` counts) of the dates."""
    return days.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64)


def _mid_month(months: np.ndarray) -> np.ndarray:
    """The 15th of each month (``datetime64[M]`` counts) as days since 1970-01-01."""
    return (months.astype("datetime64[M]").astype("datetime64[D]") + np.timedelta64(14, "D")).astype(np.int64)


def _anomalies(values: np.ndarray, days: np.ndarray, anomaly: Optional[str], jobs: int) -> np.ndarray:
    from . import _core

    if anomaly is None:
        return values
    return _core.anomaly.anomalies(values, days, method=anomaly, n_jobs=jobs)


def _driver(y: Any, x_cube: xr.DataArray, nodata: Any):
    """y as (time, y, x) on x's grid, or a series: (values (T,) or (T, N), time index, is_series)."""
    nt, ny, nx = x_cube.shape
    if isinstance(y, pd.Series):
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("a series y needs a DatetimeIndex")
        return y.to_numpy(dtype=np.float64), y.index.values, True
    if isinstance(y, xr.DataArray) and y.ndim == 1:
        if "time" not in y.coords:
            raise ValueError("a 1-D y needs a time coordinate")
        return np.asarray(y.values, dtype=np.float64), y.time.values, True
    cube, pixel = _series_cube(y, dates=None, band=None, chunks=None, need_dates=True, nodata=nodata)
    if pixel:
        return cube.values.reshape(-1).astype(np.float64), cube.time.values, True
    cube = sorted_by_time(cube).transpose("time", "y", "x")
    if cube.shape[1:] != (ny, nx):
        raise ValueError(f"y is on a grid of {cube.shape[1:]} pixels, x on {(ny, nx)}: put y on x's grid first "
                         "(zeit.load_raster(..., like=x) or reproject)")
    return np.asarray(cube.values, dtype=np.float64).reshape(cube.shape[0], ny * nx), cube.time.values, False


def _monthly(values: np.ndarray, days: np.ndarray, first: int, count: int) -> np.ndarray:
    """Monthly means on a contiguous grid of ``count`` months from month ``first``, NaN for the missing."""
    from . import _core

    v2 = values.reshape(len(days), -1)
    months, means = _core.anomaly.monthly_means(np.ascontiguousarray(v2), days)
    out = np.full((count, v2.shape[1]), np.nan)
    out[months - 1970 * 12 - first] = means      # the core counts months as year * 12 + month - 1
    return out


def lagged_correlation(
    x: Any,
    y: Any,
    lags: Union[int, Sequence[int]] = 3,
    *,
    step: str = "month",
    match: str = "nearest",
    tolerance: Any = "8D",
    anomaly: str = "month",
    method: str = "pearson",
    best: str = "absolute",
    min_pairs: int = 10,
    effective_n: bool = True,
    sidak: bool = True,
    alpha_fdr: float = 0.10,
    band: Union[str, int, None] = None,
    nodata: Any = "auto",
    n_jobs: int = -1,
) -> xr.Dataset:
    """The correlation of every pixel's series ``x`` with a driver ``y`` at several lags.

    At lag ``k`` the pairs are ``(x_t, y_{t-k})``: ``y`` leads ``x`` by ``k`` steps, and a driver that acts two
    months later shows its strongest correlation at lag 2. The pairs are the dates where both are valid. The
    significance is a two-sided Student t test with the *effective* number of pairs
    ``n (1 - rx ry) / (1 + rx ry)`` (Bretherton et al. 1999), ``rx`` and ``ry`` the lag-1 autocorrelations of
    the series, because the consecutive values of a series are not independent; the p-value of the best lag is
    adjusted for having tried the lags (Sidak), and the best lags of all the pixels are tested together by the
    false discovery rate (:func:`zeit.fdr`).

    Parameters
    ----------
    x
        The series of every pixel: a ``(time, y, x)`` cube, a raster, or a Dataset (with ``band=``).
    y
        The driver: a ``(time, y, x)`` cube on the grid of ``x`` (each pixel's series against its own), or one
        series (a ``pandas.Series`` with a ``DatetimeIndex``, or a ``DataArray`` of ``time``), used for every
        pixel: a climate index, a regional mean.
    lags
        An integer ``K`` for the lags ``0, 1, ..., K``, or the list of lags (steps; negative: ``x`` leads).
    step
        ``"month"`` (default): both are averaged to calendar months first (``y`` and ``x`` of any frequency,
        monthly climate indices against 16-day indices) and the lags are months. ``"dates"``: ``y`` is matched to
        the dates of ``x`` by ``match`` and the lags are observations of ``x``.
    match, tolerance
        For ``step="dates"``: ``"nearest"`` (the nearest date of ``y`` within ``tolerance``, default 8 days),
        ``"exact"``, ``"month"`` (a date of the same calendar month) or ``"year"``.
    anomaly
        Taken out of both before correlating: ``"month"`` (default; the monthly climatology), ``"harmonic"``,
        ``"mean"`` or ``"none"`` (the raw values; trends and seasonality then count).
    method
        ``"pearson"`` or ``"spearman"`` (ranks, ties averaged).
    best
        The best lag: ``"absolute"`` (the largest ``|r|``), ``"positive"`` or ``"negative"``.
    min_pairs
        Fewer pairs give NaN.
    effective_n
        ``False`` takes the pairs as independent (the p-values are then too small for smooth series).
    sidak
        Adjust ``best_p`` for the number of lags tried.
    alpha_fdr
        The false discovery rate of ``significant`` (default 0.10, Wilks 2016).
    band, nodata, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        - ``r``, ``p``, ``n`` (pairs) and ``n_eff`` (effective pairs), each ``(lag, y, x)``;
        - ``best_r``, ``best_lag`` (``-1`` where no lag has a correlation) and ``best_p`` (Sidak-adjusted),
          ``q`` (the false discovery rate adjusted p) and ``significant`` (``q <= alpha_fdr``), each ``(y, x)``;
        - attribute ``p_threshold``: the p-value up to which ``best_p`` is significant.

    Examples
    --------
    >>> ndvi = zeit.load_raster("ndvi_16d.tif")
    >>> oni = pd.read_csv("oni.csv", index_col=0, parse_dates=True)["oni"]      # monthly
    >>> lc = zeit.lagged_correlation(ndvi, oni, lags=6)                          # NDVI follows ENSO by ...
    >>> lc.best_lag.where(lc.significant).zeit.plot()

    References
    ----------
    Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., & Blade, I. (1999). The effective number
    of spatial degrees of freedom of a time-varying field. Journal of Climate, 12, 1990-2004.
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. JRSS B, 57, 289-300.
    Wilks, D. S. (2016). BAMS, 97, 2263-2273.
    """
    from ._embeddings import refuse
    from . import _core

    refuse(x, "zeit.lagged_correlation")
    if step not in ("month", "dates"):
        raise ValueError(f"step must be 'month' or 'dates', got {step!r}")
    if anomaly not in _ANOMALY:
        raise ValueError(f"anomaly must be one of {sorted(_ANOMALY)}, got {anomaly!r}")
    if match not in _MATCH:
        raise ValueError(f"match must be one of {sorted(_MATCH)}, got {match!r}")
    if method not in ("pearson", "spearman") or best not in ("absolute", "positive", "negative"):
        raise ValueError("method is 'pearson' or 'spearman', best 'absolute', 'positive' or 'negative'")
    lag_list = list(range(int(lags) + 1)) if np.isscalar(lags) else [int(k) for k in lags]
    if not lag_list or (np.isscalar(lags) and int(lags) < 0):
        raise ValueError(f"lags must be a number >= 0 or a list of lags, got {lags!r}")
    if not 0.0 < float(alpha_fdr) < 1.0:
        raise ValueError(f"alpha_fdr must be in (0, 1), got {alpha_fdr!r}")
    jobs = resolve(n_jobs)
    cube, _ = _series_cube(x, dates=None, band=band, chunks=None, need_dates=True, nodata=nodata)
    cube = sorted_by_time(cube).transpose("time", "y", "x")
    nt, ny, nx = cube.shape
    xv = np.ascontiguousarray(np.asarray(cube.values, dtype=np.float64)).reshape(nt, ny * nx)
    xd = days_of(cube.time.values)
    yv, ytime, series = _driver(y, cube, nodata)
    yd = days_of(ytime)
    order = np.argsort(yd, kind="stable")
    if not np.all(order == np.arange(len(order))):
        yd, yv = yd[order], yv[order]
    if len(yd) != yv.shape[0]:
        raise ValueError(f"y has {len(yd)} dates for {yv.shape[0]} values")
    an = _ANOMALY[anomaly]
    y2 = yv.reshape(len(yd), -1)

    if step == "month":
        mx, my = _month_grid(xd), _month_grid(yd)
        first, last = int(min(mx.min(), my.min())), int(max(mx.max(), my.max()))
        count = last - first + 1
        mid = _mid_month(np.arange(first, last + 1))
        xs = _anomalies(_monthly(xv, xd, first, count), mid, an, jobs)
        ys = _anomalies(_monthly(y2, yd, first, count), mid, an, jobs)
        rows = np.arange(count)
        index = np.array([np.where((rows - k >= 0) & (rows - k < count), rows - k, -1) for k in lag_list],
                         dtype=np.int32)
        used = count
    else:
        tol = _days_of_tolerance(tolerance) if match == "nearest" else 0.0
        xs = _anomalies(xv, xd, an, jobs)
        ys = _anomalies(np.ascontiguousarray(y2), yd, an, jobs)
        index = _core.correlation.match_index(xd, yd, lag_list, matching=_MATCH[match], tolerance=float(tol),
                                              unit="steps")
        used = nt
    y_arg = np.ascontiguousarray(ys.reshape(-1)) if series else np.ascontiguousarray(ys)
    r = _core.correlation.lagged(np.ascontiguousarray(xs), y_arg, np.ascontiguousarray(index), method=method,
                                 min_pairs=int(min_pairs), effective_n=bool(effective_n), best=best,
                                 sidak=bool(sidak), n_jobs=jobs)
    q, star, sig = fdr(r["best_p"], alpha_fdr)
    nl = len(lag_list)
    coords = {c: cube.coords[c] for c in ("y", "x") if c in cube.coords}
    coords["lag"] = np.asarray(lag_list)
    f32 = np.float32
    ds = xr.Dataset(
        {
            "r": (("lag", "y", "x"), r["r"].reshape(nl, ny, nx).astype(f32)),
            "p": (("lag", "y", "x"), r["p"].reshape(nl, ny, nx).astype(f32)),
            "n": (("lag", "y", "x"), r["n"].reshape(nl, ny, nx).astype(np.int32)),
            "n_eff": (("lag", "y", "x"), r["n_eff"].reshape(nl, ny, nx).astype(f32)),
            "best_r": (("y", "x"), r["best_r"].reshape(ny, nx).astype(f32)),
            "best_lag": (("y", "x"), r["best_lag"].reshape(ny, nx)),
            "best_p": (("y", "x"), r["best_p"].reshape(ny, nx).astype(f32)),
            "q": (("y", "x"), q.reshape(ny, nx).astype(f32)),
            "significant": (("y", "x"), sig.reshape(ny, nx)),
        },
        coords=coords,
    )
    ds["best_lag"].attrs.update(_FillValue=-1, long_name=f"lag of the best correlation ({step})")
    ds.attrs = dict(algorithm="lagged correlation", step=step, match=match if step == "dates" else "",
                    anomaly=anomaly, method=method, best=best, effective_n=int(bool(effective_n)),
                    sidak=int(bool(sidak)), alpha_fdr=float(alpha_fdr), p_threshold=float(star),
                    min_pairs=int(min_pairs), steps=int(used))
    crs = _crs_of(cube)
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds
