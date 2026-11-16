"""Lagged correlation (zeit.lagged_correlation) and the false discovery rate (zeit.fdr): a driver that acts two
months later, a cube or a 1-D driver, months or dates, missing values, Spearman, the null maps, the errors."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

NT, NY, NX = 240, 5, 6
PHI = 0.7


def _ar1(rng, n, size=None):
    shape = (n,) if size is None else (n, size)
    e = rng.normal(size=shape)
    out = np.zeros(shape)
    for t in range(1, n):
        out[t] = PHI * out[t - 1] + e[t]
    return out


def _months(n=NT):
    return pd.DatetimeIndex((np.datetime64("1990-01", "M") + np.arange(n)).astype("datetime64[D]") + np.timedelta64(14, "D"))


def _driven(seed=0, lag=2, coef=0.8):
    """y AR(1) (one series, and a cube of independent ones) and x_t = coef y_{t-lag} + noise."""
    rng = np.random.default_rng(seed)
    y = _ar1(rng, NT)
    x = np.full((NT, NY * NX), np.nan)
    x[lag:] = coef * y[:-lag, None] + rng.normal(size=(NT - lag, NY * NX))
    x[:lag] = rng.normal(size=(lag, NY * NX))
    coords = {"time": _months(), "y": np.arange(NY, dtype=float), "x": np.arange(NX, dtype=float)}
    xda = xr.DataArray(x.reshape(NT, NY, NX), dims=("time", "y", "x"), coords=coords)
    return xda, pd.Series(y, index=_months(), name="y"), y


def _expected_r(coef=0.8):
    var_y = 1.0 / (1.0 - PHI ** 2)
    return coef * np.sqrt(var_y) / np.sqrt(coef ** 2 * var_y + 1.0)


def test_best_lag_two_with_a_series_driver():
    x, y, _ = _driven()
    ds = zeit.lagged_correlation(x, y, lags=4, anomaly="none", n_jobs=2)
    assert ds.r.dims == ("lag", "y", "x") and ds.best_lag.dims == ("y", "x")
    np.testing.assert_array_equal(ds.lag.values, [0, 1, 2, 3, 4])
    assert (ds.best_lag.values == 2).all() and ds.best_lag.dtype == np.int16
    np.testing.assert_allclose(ds.best_r.values, ds.r.sel(lag=2).values, rtol=1e-6)
    assert abs(float(ds.best_r.mean()) - _expected_r()) < 0.05
    assert (ds.r.sel(lag=0).values < ds.best_r.values).all()
    assert ds.n.dtype == np.int32 and (ds.n.sel(lag=2).values == NT - 2).all()
    assert (ds.n_eff.values <= ds.n.values + 1e-3).all()
    assert (ds.n_eff.sel(lag=2).values < NT - 2).all()          # both series are autocorrelated
    assert ds.significant.values.all() and ds.significant.dtype == bool
    assert 0 < ds.attrs["p_threshold"] <= 1
    q, star, sig = zeit.fdr(ds.best_p.values.astype(float), 0.10)
    np.testing.assert_allclose(ds.attrs["p_threshold"], star)
    assert (ds.q.values <= 0.10).all()
    np.testing.assert_array_equal(ds.x.values, x.x.values)


def test_sidak_makes_best_p_larger_and_effective_n_makes_p_larger():
    x, y, _ = _driven(seed=1, coef=0.15)
    on = zeit.lagged_correlation(x, y, lags=3, anomaly="none", sidak=True)
    off = zeit.lagged_correlation(x, y, lags=3, anomaly="none", sidak=False)
    assert (on.best_p.values >= off.best_p.values - 1e-7).all()
    np.testing.assert_allclose(on.best_p.values, 1 - (1 - off.best_p.values.astype(float)) ** 4, rtol=1e-5, atol=1e-7)
    iid = zeit.lagged_correlation(x, y, lags=3, anomaly="none", effective_n=False)
    np.testing.assert_allclose(iid.n_eff.values, iid.n.values, rtol=1e-6)
    assert (iid.p.values <= on.p.values + 1e-7).all()


def test_cube_driver_and_dataarray_series():
    x, y, y_raw = _driven(seed=2)
    rng = np.random.default_rng(9)
    ycube = xr.DataArray(_ar1(rng, NT, NY * NX).reshape(NT, NY, NX), dims=("time", "y", "x"), coords=x.coords)
    xv = x.values.copy()
    xv[2:] = 0.8 * ycube.values[:-2] + rng.normal(size=(NT - 2, NY, NX))
    ds = zeit.lagged_correlation(x.copy(data=xv), ycube, lags=3, anomaly="none")
    assert (ds.best_lag.values == 2).all()
    assert abs(float(ds.best_r.mean()) - _expected_r()) < 0.05
    da = xr.DataArray(y.values, dims="time", coords={"time": y.index.values})
    a = zeit.lagged_correlation(x, y, lags=3, anomaly="none")
    b = zeit.lagged_correlation(x, da, lags=3, anomaly="none")
    np.testing.assert_array_equal(a.r.values, b.r.values)
    # a y that is one pixel's series per pixel: the cube of copies gives the same as the series
    copies = xr.DataArray(np.repeat(y.values[:, None, None], NY * NX, axis=1).reshape(NT, NY, NX),
                          dims=("time", "y", "x"), coords=x.coords)
    c = zeit.lagged_correlation(x, copies, lags=3, anomaly="none")
    np.testing.assert_allclose(a.r.values, c.r.values, rtol=1e-6)


def test_month_anomalies_and_a_seasonal_cycle():
    x, y, _ = _driven(seed=3)
    season = 5.0 * np.sin(2 * np.pi * np.arange(NT) / 12.0)
    sx = x + xr.DataArray(season, dims="time", coords={"time": x.time})
    sy = y + season
    raw = zeit.lagged_correlation(sx, sy, lags=4, anomaly="none")
    anom = zeit.lagged_correlation(sx, sy, lags=4, anomaly="month")
    assert abs(float(anom.best_r.mean()) - _expected_r()) < 0.07
    assert (anom.best_lag.values == 2).all()
    assert float(raw.r.sel(lag=0).mean()) > 0.9                    # the shared cycle dominates the raw series
    harm = zeit.lagged_correlation(sx, sy, lags=4, anomaly="harmonic")
    assert (harm.best_lag.values == 2).all()


def test_sixteen_day_x_against_a_monthly_driver():
    rng = np.random.default_rng(4)
    months = _months(120)
    y = pd.Series(_ar1(rng, 120), index=months)
    days = pd.date_range("1990-01-01", months[-1], freq="16D")
    ym = pd.Series(y.values, index=months.to_period("M"))
    x = np.array([[0.8 * ym.get(pd.Period(d, "M") - 2, np.nan) + rng.normal() for _ in range(6)] for d in days])
    da = xr.DataArray(x.reshape(len(days), 2, 3), dims=("time", "y", "x"),
                      coords={"time": days, "y": [0.0, 1.0], "x": [0.0, 1.0, 2.0]})
    ds = zeit.lagged_correlation(da, y, lags=4, anomaly="none", min_pairs=10)
    assert (ds.best_lag.values == 2).all()
    assert (ds.n.sel(lag=2).values > 100).all() and (ds.n.sel(lag=2).values <= 118).all()
    ds2 = zeit.lagged_correlation(da, y, lags=[2, 0], anomaly="none")
    np.testing.assert_array_equal(ds2.lag.values, [2, 0])
    np.testing.assert_allclose(ds2.r.sel(lag=2).values, ds.r.sel(lag=2).values, rtol=1e-6)


def test_step_dates_matching():
    x, y, _ = _driven(seed=5)
    exact = zeit.lagged_correlation(x, y, lags=4, step="dates", match="exact", anomaly="none")
    assert exact.attrs["step"] == "dates" and (exact.best_lag.values == 2).all()
    month = zeit.lagged_correlation(x, y, lags=4, anomaly="none")
    np.testing.assert_allclose(exact.r.values, month.r.values, rtol=1e-5, atol=1e-6)
    shifted = pd.Series(y.values, index=y.index - np.timedelta64(3, "D"))
    near = zeit.lagged_correlation(x, shifted, lags=4, step="dates", match="nearest", tolerance="8D",
                                   anomaly="none")
    np.testing.assert_allclose(near.r.values, exact.r.values, rtol=1e-6)
    none = zeit.lagged_correlation(x, shifted, lags=4, step="dates", match="exact", anomaly="none")
    assert (none.best_lag.values == -1).all() and np.isnan(none.best_r.values).all()
    assert not none.significant.values.any()
    mm = zeit.lagged_correlation(x, shifted, lags=4, step="dates", match="month", anomaly="none")
    np.testing.assert_allclose(mm.r.values, exact.r.values, rtol=1e-6)
    tight = zeit.lagged_correlation(x, shifted, lags=4, step="dates", match="nearest", tolerance="2D",
                                    anomaly="none")
    assert (tight.best_lag.values == -1).all()


def test_missing_pixels_dates_and_min_pairs():
    x, y, _ = _driven(seed=6)
    v = x.values.copy()
    v[:, 0, 0] = np.nan
    v[10:60, 1, 1] = np.nan
    v[:, 2, 2] = np.nan
    v[:8, 2, 2] = 1.0                                              # fewer than min_pairs
    ds = zeit.lagged_correlation(x.copy(data=v), y, lags=3, anomaly="none")
    assert ds.best_lag.values[0, 0] == -1 and np.isnan(ds.best_r.values[0, 0]) and np.isnan(ds.q.values[0, 0])
    assert not bool(ds.significant.values[0, 0]) and np.isnan(ds.r.values[:, 0, 0]).all()
    assert ds.best_lag.values[2, 2] == -1
    assert ds.n.values[2, 1, 1] == NT - 2 - 50 and ds.best_lag.values[1, 1] == 2
    yn = y.copy()
    yn.iloc[100:130] = np.nan
    both = zeit.lagged_correlation(x, yn, lags=3, anomaly="none")
    assert (both.n.sel(lag=2).values == NT - 2 - 30).all()
    few = zeit.lagged_correlation(x, y, lags=3, anomaly="none", min_pairs=1000)
    assert np.isnan(few.r.values).all()


def test_spearman_uses_ranks():
    x, y, _ = _driven(seed=7)
    a = zeit.lagged_correlation(x, y, lags=3, anomaly="none", method="spearman")
    b = zeit.lagged_correlation(x, np.exp(y), lags=3, anomaly="none", method="spearman")   # monotone: same ranks
    np.testing.assert_allclose(a.r.values, b.r.values, rtol=1e-6, atol=1e-7)
    p = zeit.lagged_correlation(x, y, lags=3, anomaly="none")
    assert (a.best_lag.values == 2).all() and not np.array_equal(a.r.values, p.r.values)
    j, i = 2, 3
    from scipy import stats
    expected = stats.spearmanr(x.values[2:, j, i], y.values[:-2])[0]
    np.testing.assert_allclose(a.r.sel(lag=2).values[j, i], expected, atol=1e-6)


def test_best_negative_and_positive():
    x, y, _ = _driven(seed=8)
    neg = zeit.lagged_correlation(-x, y, lags=4, anomaly="none", best="negative")
    assert (neg.best_lag.values == 2).all() and (neg.best_r.values < 0).all()
    pos = zeit.lagged_correlation(-x, y, lags=4, anomaly="none", best="positive")
    assert (pos.best_lag.values != 2).any()
    assert (pos.best_r.values >= neg.best_r.values).all()


def test_null_maps_have_few_false_positives():
    rng = np.random.default_rng(11)
    ny, nx = 20, 20
    x = xr.DataArray(rng.normal(size=(NT, ny, nx)), dims=("time", "y", "x"),
                     coords={"time": _months(), "y": np.arange(ny, dtype=float), "x": np.arange(nx, dtype=float)})
    y = pd.Series(_ar1(rng, NT), index=_months())
    ds = zeit.lagged_correlation(x, y, lags=3, anomaly="none", alpha_fdr=0.10)
    assert float(ds.significant.mean()) <= 0.10
    assert (ds.q.values >= ds.best_p.values - 1e-6).all()


def test_fdr_benjamini_hochberg_1995_example():
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719,
         0.6528, 0.7590, 1.000]
    q, star, sig = zeit.fdr(np.array(p), alpha=0.05)
    assert star == 0.0095 and sig.sum() == 4 and sig[:4].all()
    assert q[3] == pytest.approx(0.0095 * 15 / 4) and q[-1] == 1.0
    assert np.all(np.diff(q) >= -1e-12)
    shuffled = np.random.default_rng(0).permutation(len(p))
    q2, star2, sig2 = zeit.fdr(np.array(p)[shuffled], 0.05)
    assert star2 == star and sig2.sum() == 4
    nanned = np.array(p + [np.nan]).reshape(4, 4)
    q3, star3, sig3 = zeit.fdr(nanned, 0.05)
    assert q3.shape == (4, 4) and np.isnan(q3[3, 3]) and not sig3[3, 3] and star3 == star
    da = xr.DataArray(nanned, dims=("y", "x"), coords={"y": range(4), "x": range(4)})
    qd, _, sd = zeit.fdr(da, 0.05)
    assert isinstance(qd, xr.DataArray) and qd.dims == ("y", "x") and sd.dtype == bool
    q4, star4, sig4 = zeit.fdr(np.array([0.5, 0.9, np.nan]), 0.05)
    assert star4 == 0.0 and not sig4.any()
    with pytest.raises(ValueError, match="alpha"):
        zeit.fdr(p, alpha=1.5)


def test_core_modules_and_months_lags():
    from zeit import _core

    x = np.array([np.datetime64("2001-03-31"), np.datetime64("2001-05-15")]).astype("datetime64[D]").astype(np.int64)
    y = np.array([np.datetime64("2001-02-28"), np.datetime64("2001-04-15")]).astype("datetime64[D]").astype(np.int64)
    idx = _core.correlation.match_index(x, y, [1], matching="exact", unit="months")
    assert idx.tolist() == [[0, 1]]                                  # 31 Mar - 1 month = 28 Feb (clamped), 15 May -> 15 Apr
    steps = _core.correlation.match_index(x, y, [0, 1], matching="nearest", tolerance=40.0)
    assert steps.shape == (2, 2) and steps[1, 0] == -1
    months, means = _core.anomaly.monthly_means(np.array([[1.0], [3.0], [5.0]]), np.array([0, 5, 90]))
    assert months.tolist() == [1970 * 12, 1970 * 12 + 3] and means[:, 0].tolist() == [2.0, 5.0]


def test_errors():
    x, y, _ = _driven()
    with pytest.raises(ValueError, match="step"):
        zeit.lagged_correlation(x, y, step="weeks")
    with pytest.raises(ValueError, match="anomaly"):
        zeit.lagged_correlation(x, y, anomaly="bogus")
    with pytest.raises(ValueError, match="match"):
        zeit.lagged_correlation(x, y, match="bogus")
    with pytest.raises(ValueError, match="method"):
        zeit.lagged_correlation(x, y, method="kendall")
    with pytest.raises(ValueError, match="lags"):
        zeit.lagged_correlation(x, y, lags=-1)
    with pytest.raises(ValueError, match="alpha_fdr"):
        zeit.lagged_correlation(x, y, alpha_fdr=0.0)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        zeit.lagged_correlation(x, pd.Series(np.arange(NT, dtype=float)))
    other = xr.DataArray(np.zeros((NT, 3, 3)), dims=("time", "y", "x"),
                         coords={"time": _months(), "y": np.arange(3.0), "x": np.arange(3.0)})
    with pytest.raises(ValueError, match="grid"):
        zeit.lagged_correlation(x, other)
