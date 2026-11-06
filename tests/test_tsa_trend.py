"""zeit.linear_trend and zeit.cat (FORCE's TSA trend and change), and zeit.standardize along another axis."""
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import zeit
from zeit._tsa import CAT_NAMES, TREND_NAMES

NAN = np.nan


def _cube(values, dates=None):
    n, ny, nx = values.shape
    dates = pd.date_range("2018-01-01", periods=n, freq="YS") if dates is None else dates
    return xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": pd.DatetimeIndex(dates), "y": np.arange(ny)[::-1] + 0.5,
                                "x": np.arange(nx) + 0.5}).rio.write_crs(4326)


def _one(series):
    """One pixel's series, yearly dates (a pandas Series)."""
    values = np.asarray(series, dtype=float)
    return pd.Series(values, index=pd.date_range("2018-01-01", periods=len(values), freq="YS"))


@pytest.fixture(scope="module")
def noisy():
    """12 steps, (y=5, x=6): a trend plus noise, about a fifth missing; pixels with 2, 3 and 0 valid steps."""
    rng = np.random.default_rng(3)
    slope = rng.normal(0.0, 0.03, (1, 5, 6))
    v = 0.5 + slope * np.arange(12)[:, None, None] + rng.normal(0, 0.05, (12, 5, 6))
    v[rng.random(v.shape) < 0.2] = NAN
    v[:, 0, 0] = NAN
    v[2:, 0, 1] = NAN
    v[3:, 0, 2] = NAN
    v[:, 0, 2][[0, 1, 2]] = [0.4, 0.5, 0.45]
    return _cube(v)


# ---- linear_trend -------------------------------------------------------------------------------------------

def test_trend_matches_scipy_linregress_and_the_t_quantile(noisy):
    out = zeit.linear_trend(noisy, n_jobs=2)
    assert list(out.data_vars) == TREND_NAMES
    assert out.rio.crs.to_epsg() == 4326 and (out.y.values == noisy.y.values).all()
    values = noisy.values
    n_checked = n_sig = 0
    for j in range(5):
        for i in range(6):
            y = values[:, j, i]
            ok = ~np.isnan(y)
            if ok.sum() < 3:
                assert all(np.isnan(out[name].values[j, i]) for name in TREND_NAMES), (j, i)
                continue
            x = np.arange(12)[ok]
            r = stats.linregress(x, y[ok])
            got = {name: float(out[name].values[j, i]) for name in TREND_NAMES}
            assert got["slope"] == pytest.approx(r.slope, rel=1e-9, abs=1e-12)
            assert got["offset"] == pytest.approx(r.intercept, rel=1e-9, abs=1e-12)
            assert got["rsq"] == pytest.approx(r.rvalue ** 2, rel=1e-9, abs=1e-12)
            assert got["mean"] == pytest.approx(y[ok].mean(), rel=1e-12)
            resid = y[ok] - (r.intercept + r.slope * x)
            assert got["rmse"] == pytest.approx(np.sqrt((resid ** 2).mean()), rel=1e-9)
            assert got["mae"] == pytest.approx(np.abs(resid).mean(), rel=1e-9)
            assert got["maxe"] == pytest.approx(np.abs(resid).max(), rel=1e-9)
            assert (got["num"], got["len"]) == (ok.sum(), 12)
            assert got["abs_gain"] == pytest.approx(r.slope * 12, rel=1e-9, abs=1e-12)
            if r.intercept > 0:
                assert got["pct_gain"] == pytest.approx(r.slope * 12 / r.intercept * 100, rel=1e-9, abs=1e-9)
            else:
                assert got["pct_gain"] == 0
            # the critical value of scipy's t quantile, df = k - 2
            t = r.slope / r.stderr
            crit = stats.t.ppf(1 - 0.05 / 2, ok.sum() - 2)
            assert got["sig"] == (1 if t >= crit else -1 if t <= -crit else 0), (j, i, t, crit)
            n_checked += 1
            n_sig += got["sig"] != 0
    assert n_checked >= 20 and 0 < n_sig < n_checked


@pytest.mark.parametrize("tail", ["two", "left", "right"])
@pytest.mark.parametrize("conf", [0.9, 0.99])
def test_significance_by_tail_and_level(noisy, tail, conf):
    out = zeit.linear_trend(noisy, conf=conf, tail=tail)
    values = noisy.values
    alpha = 1 - conf
    n_sig = 0
    for j in range(5):
        for i in range(6):
            y = values[:, j, i]
            ok = ~np.isnan(y)
            if ok.sum() < 3:
                continue
            r = stats.linregress(np.arange(12)[ok], y[ok])
            t, df = r.slope / r.stderr, ok.sum() - 2
            want = {"two": (1 if t >= stats.t.ppf(1 - alpha / 2, df) else -1 if t <= -stats.t.ppf(1 - alpha / 2, df)
                            else 0),
                    "left": -1 if t <= stats.t.ppf(alpha, df) else 0,
                    "right": 1 if t >= stats.t.ppf(1 - alpha, df) else 0}[tail]
            assert out.sig.values[j, i] == want, (tail, conf, j, i)
            n_sig += want != 0
    assert n_sig > 0


def test_critical_value_at_the_limit():
    # a series whose t-value is just under / over the exact critical value of df = 3 (3.1824463)
    x = np.arange(5)
    e = np.array([1.0, -1.0, 0.5, -0.5, 0.0])

    def t_of(scale):
        y = x + scale * e
        r = stats.linregress(x, y)
        return r.slope / r.stderr

    lo, hi = 0.0, 2.0
    for _ in range(200):  # the scale that gives t = the critical value
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if t_of(mid) > stats.t.ppf(0.975, 3) else (lo, mid)
    crit = stats.t.ppf(0.975, 3)
    for scale, sig in ((lo * 0.999, 1), (hi * 1.001, 0)):
        assert t_of(scale) > crit - 0.01 and t_of(scale) < crit + 0.01
        out = zeit.linear_trend(_one(x + scale * e))
        assert float(out.sig) == sig, (scale, t_of(scale), crit)


def test_perfect_line_constant_and_short_series():
    out = zeit.linear_trend(_one(2.0 + 3.0 * np.arange(6)))
    assert float(out.slope) == pytest.approx(3.0) and float(out.offset) == pytest.approx(2.0)
    assert float(out.rsq) == pytest.approx(1.0) and float(out.sig) == 1 and float(out.rmse) < 1e-9
    assert float(out.pct_gain) == pytest.approx(900.0) and float(out.abs_gain) == pytest.approx(18.0)
    const = zeit.linear_trend(_one([4.0, 4.0, 4.0, 4.0]))
    assert float(const.slope) == 0 and float(const.sig) == 0 and float(const.rmse) == 0 and np.isnan(float(const.rsq))
    short = zeit.linear_trend(_one([1.0, NAN, 2.0, NAN]))
    assert all(np.isnan(float(short[name])) for name in TREND_NAMES)
    # the position counts the missing steps: gaps do not close up
    gap = zeit.linear_trend(_one([0.0, NAN, 2.0, NAN, 4.0, 5.0]))
    assert float(gap.slope) == pytest.approx(1.0) and (float(gap["num"]), float(gap["len"])) == (4.0, 6.0)


def test_trend_errors_and_pixel_input():
    with pytest.raises(ValueError, match="conf"):
        zeit.linear_trend(_one([1, 2, 3, 4.0]), conf=1.0)
    with pytest.raises(ValueError, match="tail"):
        zeit.linear_trend(_one([1, 2, 3, 4.0]), tail="both")
    with pytest.raises(ValueError, match="dim"):
        zeit.linear_trend(_one([1, 2, 3, 4.0]), dim="year")
    out = zeit.linear_trend([1.0, 2.5, 2.9, 4.2, 5.0])  # a list: one pixel
    assert out.slope.shape == () and float(out.slope) == pytest.approx(stats.linregress(range(5), [1, 2.5, 2.9, 4.2, 5]).slope)


# ---- cat ----------------------------------------------------------------------------------------------------

def _cat(series, **kw):
    out = zeit.cat(_one(series), **kw)
    return {name: float(out[name]) if name != "change_time" else out[name].values for name in out.data_vars}


def test_cat_names_and_a_clear_drop():
    out = zeit.cat(_one([10, 10, 10, 4, 4, 4.0]))
    assert list(out.data_vars) == ["change", "loss", "change_time"] + CAT_NAMES[3:]
    r = _cat([10, 10, 10, 4, 4, 4.0])
    assert r["change"] == 6.0 and pd.Timestamp(r["change_time"]) == pd.Timestamp("2021-01-01")
    # total: the line over all six; loss: the change relative to its offset
    line = stats.linregress(np.arange(6), [10, 10, 10, 4, 4, 4.0])
    assert r["total_slope"] == pytest.approx(line.slope) and r["total_offset"] == pytest.approx(line.intercept)
    assert r["loss"] == pytest.approx(6.0 / line.intercept * 100)
    assert (r["total_num"], r["total_len"]) == (6, 6)
    # before is [0, f_change - 1): two steps, so only mean, num, len
    assert (r["before_mean"], r["before_num"], r["before_len"]) == (10.0, 2.0, 2.0)
    assert all(np.isnan(r[f"before_{n}"]) for n in ("offset", "slope", "rsq", "sig", "rmse", "maxe"))
    # after is [3, 6)
    assert (r["after_mean"], r["after_offset"], r["after_slope"], r["after_num"], r["after_len"]) == (4.0, 4.0, 0.0, 3.0, 3.0)
    assert r["after_sig"] == 0 and r["after_rmse"] == 0


def test_cat_parts_against_linregress():
    y = np.array([9.0, 8.1, 7.0, 6.2, 1.0, 2.1, 2.9, 4.2, 5.0])
    r = _cat(y)
    assert r["change"] == pytest.approx(5.2) and pd.Timestamp(r["change_time"]).year == 2018 + 4
    before = stats.linregress(np.arange(3), y[:3])
    after = stats.linregress(np.arange(4, 9), y[4:])
    assert r["before_slope"] == pytest.approx(before.slope) and r["before_offset"] == pytest.approx(before.intercept)
    assert r["before_rsq"] == pytest.approx(before.rvalue ** 2)
    assert r["before_pct_gain"] == pytest.approx(before.slope * 3 / before.intercept * 100)
    assert r["after_slope"] == pytest.approx(after.slope)
    # the offset of a part is the fit at its first step: x = 4 for the part after the change
    assert r["after_offset"] == pytest.approx(after.intercept + after.slope * 4)
    assert r["after_abs_gain"] == pytest.approx(after.slope * 5)
    assert r["after_sig"] == 1 and (r["after_num"], r["after_len"]) == (5, 5)


def test_cat_only_increases_ties_gaps_and_short_series():
    r = _cat([1.0, 2.0, 4.0, 8.0])  # only rises: the smallest rise is the "greatest change"
    assert r["change"] == -1.0 and pd.Timestamp(r["change_time"]).year == 2019
    assert np.isnan(r["before_num"]) and r["after_len"] == 3.0
    r = _cat([5.0, 3.0, 1.0])  # a tie keeps the first
    assert r["change"] == 2.0 and pd.Timestamp(r["change_time"]).year == 2019
    r = _cat([5.0, NAN, 3.0, NAN, 3.5])  # the loss at step 2 is against step 0
    assert r["change"] == 2.0 and pd.Timestamp(r["change_time"]).year == 2020
    assert (r["total_num"], r["total_len"]) == (3.0, 5.0)
    r = _cat([5.0, NAN, 3.0, NAN])  # two valid steps: a change, a total with mean/num/len only, no loss
    assert r["change"] == 2.0 and r["loss"] == 0.0 and r["total_num"] == 2.0 and np.isnan(r["total_slope"])
    # an after part with two valid steps
    r = _cat([9.0, 8.0, 7.0, 1.0, 2.0])
    assert (r["after_mean"], r["after_num"]) == (1.5, 2.0) and np.isnan(r["after_slope"])
    for series in ([NAN, 2.0, NAN], [NAN, NAN, NAN], [3.0]):
        r = _cat(series)
        assert np.isnat(r["change_time"]) and all(np.isnan(v) for k, v in r.items() if k != "change_time")


def test_cat_loss_is_zero_without_a_positive_offset():
    r = _cat([-1.0, -2.0, -6.0, -7.0, -8.0])
    assert r["change"] == 4.0 and r["total_offset"] < 0 and r["loss"] == 0.0


def _force_literal(y):
    """FORCE's loop as written, with CHANGE_PENALTY: the penalized candidate against the best one's real change."""
    change, found = -32768.0, None
    for f in range(1, len(y)):
        if np.isnan(y[f]):
            continue
        pre = [g for g in range(f) if not np.isnan(y[g])]
        if not pre:
            continue
        real = y[pre[-1]] - y[f]
        now = real
        if now > 0:
            post = [g for g in range(f + 1, len(y)) if not np.isnan(y[g])]
            if not post:
                continue
            now *= (y[pre[-1]] - y[post[0]]) / 1e4
        if now > change:
            change, found = real, f
    return found, change


def test_penalty_weights_a_loss_and_corrects_forces_comparison():
    # a dip of 7000 that recovers, and a lasting loss of 5000
    y = [8000.0, 1000.0, 7000.0, 2000.0, 0.0]
    plain = _cat(y)
    assert plain["change"] == 7000.0 and pd.Timestamp(plain["change_time"]).year == 2019
    pen = _cat(y, penalty=True)
    # penalized 7000 x 1000 / 1e4 = 700 against 5000 x 7000 / 1e4 = 3500: the lasting loss, its real change written
    assert pen["change"] == 5000.0 and pd.Timestamp(pen["change_time"]).year == 2021
    # FORCE compares 3500 with the dip's raw 7000 and keeps the dip: the deliberate difference
    assert _force_literal(np.array(y)) == (1, 7000.0)
    # where the literal behavior agrees with the corrected one, the results are the same
    y2 = [9000.0, 8500.0, 8000.0, 3000.0, 2900.0, 2800.0]
    assert _force_literal(np.array(y2))[0] == 3 and pd.Timestamp(_cat(y2, penalty=True)["change_time"]).year == 2021
    # a loss with no valid step after it is not considered; if all are such, no change
    y3 = [10000.0, 9000.0, 9500.0, 0.0]
    r = _cat(y3, penalty=True)
    assert r["change"] == 1000.0 and pd.Timestamp(r["change_time"]).year == 2019
    assert np.isnan(_cat([5.0, 1.0], penalty=True)["change"])


def test_cat_over_a_cube_matches_pixel_by_pixel(noisy):
    out = zeit.cat(noisy, n_jobs=2)
    values = noisy.values
    for j in range(5):
        for i in range(6):
            one = zeit.cat(_one(values[:, j, i]), n_jobs=1)
            for name in out.data_vars:
                a, b = out[name].values[j, i], one[name].values
                assert (np.isnat(a) and np.isnat(b)) if name == "change_time" and np.isnat(a) else \
                    (a == b or (np.isnan(a) and np.isnan(b))), (name, j, i)


# ---- another axis: temporal_metrics(by="year") --------------------------------------------------------------

@pytest.fixture(scope="module")
def yearly_source():
    """Six years of 16-day data with a drop of the level in 2021 (and noise), (y=3, x=4)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="16D")
    level = np.where(dates.year >= 2021, 0.3, 0.7)[:, None, None] - 0.01 * (dates.year - 2018).values[:, None, None]
    v = level + rng.normal(0, 0.02, (len(dates), 3, 4))
    v[rng.random(v.shape) < 0.2] = NAN
    return _cube(v, dates)


def test_dim_year_from_temporal_metrics(yearly_source):
    yearly = zeit.temporal_metrics(yearly_source, ["mean"], by="year")
    assert yearly.dims == ("metric", "year", "y", "x")
    tr = zeit.linear_trend(yearly)  # dim found by itself, metric squeezed
    assert tr.slope.dims == ("y", "x") and "year" not in tr.dims and "metric" not in tr.coords
    means = yearly.values[0]  # (year, y, x)
    for j, i in ((0, 0), (1, 2), (2, 3)):
        r = stats.linregress(np.arange(6), means[:, j, i])
        assert float(tr.slope[j, i]) == pytest.approx(r.slope) and float(tr.rsq[j, i]) == pytest.approx(r.rvalue ** 2)
    assert (tr["num"].values == 6).all() and (tr["len"].values == 6).all()
    assert (zeit.linear_trend(yearly, dim="year").slope.values == tr.slope.values).all()

    ch = zeit.cat(yearly)
    assert (ch.change_time.values == 2021).all() and ch.change_time.dtype == np.float64
    assert (ch.change.values > 0.35).all()
    # the same through the accessor and with the dates of a time cube
    assert (yearly.zeit.cat().change_time.values == 2021).all()
    assert (yearly_source.zeit.linear_trend().num.values > 0).all()
    daily = zeit.cat(yearly_source)
    assert str(daily.change_time.dtype).startswith("datetime64")


def test_dim_metric_must_be_chosen_when_there_are_several(yearly_source):
    two = zeit.temporal_metrics(yearly_source, ["mean", "max"], by="year")
    with pytest.raises(ValueError, match=r"\.sel\(metric="):
        zeit.linear_trend(two)
    with pytest.raises(ValueError, match=r"\.sel\(metric="):
        zeit.standardize(two)
    chosen = zeit.cat(two.sel(metric="max"))
    assert (chosen.change_time.values == 2021).all()
    other = zeit.temporal_metrics(yearly_source, ["mean"], by="month")  # (metric, month, y, x): months as the series
    assert zeit.linear_trend(other).slope.dims == ("y", "x")
    assert "month" not in zeit.linear_trend(other, dim="month").dims
    with pytest.raises(ValueError, match="not a dimension"):
        zeit.cat(other, dim="year")


# ---- standardize(dim=) --------------------------------------------------------------------------------------

def test_standardize_along_the_years(yearly_source):
    yearly = zeit.temporal_metrics(yearly_source, ["mean"], by="year")
    z = zeit.standardize(yearly, "normalize")  # the fold is standardized along year, as FORCE's STANDARDIZE_FOLD
    assert z.dims == ("year", "y", "x") and (z.year.values == np.arange(2018, 2024)).all()
    v = yearly.values[0]
    want = (v - v.mean(0)) / v.std(0, ddof=1)
    np.testing.assert_allclose(z.values, want, rtol=1e-10, atol=1e-12)
    c = zeit.standardize(yearly, "center", dim="year")
    np.testing.assert_allclose(c.values, v - v.mean(0), atol=1e-12)
    assert z.rio.crs.to_epsg() == 4326
    # a plain time cube is as before
    t = zeit.standardize(yearly_source, "center")
    assert t.dims == ("time", "y", "x")
    with pytest.raises(ValueError, match="not a dimension"):
        zeit.standardize(yearly, dim="quarter")


# ---- dask ---------------------------------------------------------------------------------------------------

def test_dask_gives_the_same_as_memory(noisy, yearly_source):
    with dask.config.set(scheduler="threads", num_workers=4):
        chunked = noisy.chunk({"time": 5, "y": 2, "x": 3})
        for fn, kw in ((zeit.linear_trend, {}), (zeit.cat, {}), (zeit.cat, {"penalty": True})):
            lazy = fn(chunked, n_jobs=2, **kw)
            assert isinstance(lazy[list(lazy.data_vars)[0]].data, da.Array)
            eager = fn(noisy, n_jobs=2, **kw)
            for name in eager.data_vars:
                a, b = lazy[name].values, eager[name].values
                if name == "change_time":
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_array_equal(a, b, err_msg=name)
        via = zeit.cat(noisy, chunks={"y": 2, "x": 3})
        assert isinstance(via.change.data, da.Array)
        np.testing.assert_array_equal(via.change.values, zeit.cat(noisy).change.values)
        # another axis, lazily
        yearly = zeit.temporal_metrics(yearly_source, ["mean"], by="year")
        lazy = zeit.cat(yearly.chunk({"y": 2, "x": 2}))
        np.testing.assert_array_equal(lazy.change_time.values, zeit.cat(yearly).change_time.values)
        z = zeit.standardize(yearly.chunk({"y": 2, "x": 2}))
        np.testing.assert_array_equal(z.values, zeit.standardize(yearly).values)
