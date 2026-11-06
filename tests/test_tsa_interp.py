"""zeit.interpolate: linear, moving average, RBF ensemble and robust harmonic fit to a grid or to any dates."""
import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

DAYS = 86400 * 10**9


def _cube(values, dates, crs=True):
    ny, nx = values.shape[1:]
    cube = xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": pd.DatetimeIndex(dates), "y": np.arange(ny)[::-1] + 0.5,
                                "x": np.arange(nx) + 0.5})
    return cube.rio.write_crs(4326) if crs else cube


def _days(index, origin):
    return (pd.DatetimeIndex(index).values.astype("datetime64[ns]").astype(np.int64) - origin.value) / DAYS


@pytest.fixture(scope="module")
def gappy():
    """230 dates every 8 days (about 5 years), (y=3, x=4), a quarter missing; one pixel empty, one with one value."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2018-01-01", periods=230, freq="8D")
    doy = np.arange(230) * 8.0
    base = 0.5 + 0.3 * np.sin(2 * np.pi * doy / 365.25)[:, None, None]
    v = base + 0.05 * rng.normal(size=(230, 3, 4))
    v[rng.random(v.shape) < 0.25] = np.nan
    v[:, 0, 0] = np.nan
    v[:, 0, 1] = np.nan
    v[40, 0, 1] = 0.7
    return _cube(v, dates)


def _grid(cube, step=16):
    first = pd.Timestamp(cube.time.values[0]).normalize()
    last = pd.Timestamp(cube.time.values[-1]).normalize()
    n = ((last - first).days + 1) // step
    return first + pd.to_timedelta(np.arange(n) * step, unit="D")


def test_grid_is_forces(gappy):
    out = zeit.interpolate(gappy, "linear", step=16)
    grid = _grid(gappy)
    assert out.dims == ("time", "y", "x") and (out.time.values == grid.values).all()
    assert out.rio.crs == gappy.rio.crs and out.attrs["method"] == "linear" and out.attrs["step"] == 16
    assert out.shape == (len(grid), 3, 4)
    # start and end: the last date of the grid is at most `end`
    cut = zeit.interpolate(gappy, "linear", step=30, start="2018-03-01", end="2019-03-01")
    assert cut.time.values[0] == np.datetime64("2018-03-01") and len(cut.time) == (366 // 30)
    assert cut.time.values[-1] <= np.datetime64("2019-03-01")


def test_linear_matches_np_interp_with_constant_edges(gappy):
    t = pd.DatetimeIndex(gappy.time.values)
    out_t = pd.date_range("2017-12-01", "2022-08-01", freq="11D")  # before the start, within and after the end
    out = zeit.interpolate(gappy, "linear", times=out_t)
    x_in, x_out = _days(t, t[0]), _days(out_t, t[0])
    for iy, ix in [(1, 1), (2, 3), (0, 2)]:
        y = gappy.values[:, iy, ix]
        ok = ~np.isnan(y)
        want = np.interp(x_out, x_in[ok], y[ok])  # np.interp is constant beyond the ends
        np.testing.assert_allclose(out.values[:, iy, ix], want, rtol=0, atol=1e-12)
    assert np.isnan(out.values[:, 0, 0]).all()  # no valid observation
    assert (out.values[:, 0, 1] == 0.7).all()  # a single one is constant


def test_moving_mean_by_hand(gappy):
    t = pd.DatetimeIndex(gappy.time.values)
    grid = _grid(gappy, 16)
    out = zeit.interpolate(gappy, "moving", window=16, step=16)
    x_in, x_out = _days(t, t[0]), _days(grid, t[0])
    y = gappy.values[:, 2, 2]
    for i, x in enumerate(x_out):
        sel = (np.abs(x_in - x) <= 16) & ~np.isnan(y)  # the window is closed
        want = y[sel].mean() if sel.any() else np.nan
        np.testing.assert_allclose(out.values[i, 2, 2], want, rtol=0, atol=1e-12, equal_nan=True)
    # closed: an observation exactly `window` days away counts
    s = pd.Series([1.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-17"]))
    got = zeit.interpolate(s, "moving", window=16, times=["2020-01-01"])
    assert got.iloc[0] == 2.0
    assert zeit.interpolate(s, "moving", window=15, times=["2020-01-01"]).iloc[0] == 1.0


def _rbf_by_kernel(x_in, y, x, sigma, cutoff):
    """FORCE's loop for each kernel, in numpy: the cut where the float32 sum of the half kernel passes cutoff / 2,
    the centre counted twice, each kernel's weighted mean weighted by its weighted share of the window."""
    num = den = 0.0
    hbin = int(4 * sigma[-1])
    ok = ~np.isnan(y)
    for s in sigma:
        d = np.arange(0, hbin + 1)
        k = (1 / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * d * d / (s * s))).astype(np.float32)
        csum = np.cumsum(k, dtype=np.float32)  # float32 as FORCE's `float sum`
        inside = csum.astype(np.float64) <= cutoff / 2
        max_ce = d[inside][-1] if inside.any() else 0
        w = 2 * k[inside].astype(np.float64).sum()
        sel = ok & (np.abs(x_in - x) <= max_ce)
        kern = 1 / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * (x_in[sel] - x) ** 2 / (s * s))
        if kern.sum() > 0:
            num += (y[sel] * kern).sum() / w
            den += kern.sum() / w
    return num / den if den > 0 else np.nan


@pytest.mark.parametrize("sigma,cutoff", [((8, 16, 32), 0.95), ((5, 10), 0.8)])
def test_rbf_matches_the_kernel_loop(gappy, sigma, cutoff):
    t = pd.DatetimeIndex(gappy.time.values)
    out_t = pd.date_range("2017-12-01", "2022-09-01", freq="16D")
    out = zeit.interpolate(gappy, "rbf", sigma=sigma, cutoff=cutoff, times=out_t)
    x_in, x_out = _days(t, t[0]), _days(out_t, t[0])
    for iy, ix in [(1, 1), (2, 3), (0, 2), (0, 1), (0, 0)]:
        y = gappy.values[:, iy, ix]
        want = np.array([_rbf_by_kernel(x_in, y, x, sigma, cutoff) for x in x_out])
        np.testing.assert_allclose(out.values[:, iy, ix], want, rtol=1e-9, atol=1e-12)
    assert np.isnan(out.values[:, 0, 0]).all()
    # dates at a fraction of a day do not use the table of integer days, and agree with it at a whole one
    half = out_t + pd.to_timedelta(12, unit="h")
    a = zeit.interpolate(gappy, "rbf", sigma=sigma, cutoff=cutoff, times=half).values[:, 2, 3]
    x_half = x_out + 0.5
    y = gappy.values[:, 2, 3]
    want = np.array([_rbf_by_kernel(x_in, y, x, sigma, cutoff) for x in x_half])
    np.testing.assert_allclose(a, want, rtol=1e-9, atol=1e-12)


def _design(x, modes=3, trend=True, period=365.25):
    cols = [np.ones_like(x)] + ([x / 1000.0] if trend else [])
    for m in range(1, modes + 1):
        cols += [np.cos(2 * np.pi * m * x / period), np.sin(2 * np.pi * m * x / period)]
    return np.stack(cols, 1)


def _curve(x):
    w = 2 * np.pi / 365.25
    return 0.4 + 0.08 * x / 365.25 + 0.3 * np.sin(w * x) + 0.1 * np.cos(2 * w * x) - 0.05 * np.sin(3 * w * x)


@pytest.mark.parametrize("modes,trend", [(3, True), (2, False), (1, True)])
def test_harmonic_without_outliers_is_the_least_squares(modes, trend):
    dates = pd.date_range("2018-01-01", periods=230, freq="8D")
    x = _days(dates, dates[0])
    rng = np.random.default_rng(5)
    truth = _design(x, modes, trend) @ rng.normal(size=_design(x, modes, trend).shape[1])  # inside the model
    y = truth[:, None, None] * np.ones((1, 2, 2))
    y[rng.random(y.shape) < 0.2] = np.nan
    cube = _cube(y, dates)
    out_t = pd.date_range("2017-06-01", "2024-01-01", freq="20D")  # extrapolates on both sides
    out = zeit.interpolate(cube, "harmonic", modes=modes, trend=trend, times=out_t)
    x_out = _days(out_t, dates[0])
    for iy, ix in [(0, 0), (1, 1)]:
        v = y[:, iy, ix]
        ok = ~np.isnan(v)
        coef = np.linalg.lstsq(_design(x[ok], modes, trend), v[ok], rcond=None)[0]
        np.testing.assert_allclose(out.values[:, iy, ix], _design(x_out, modes, trend) @ coef, rtol=0, atol=1e-8)
    coef = np.linalg.lstsq(_design(x, modes, trend), truth, rcond=None)[0]  # exact data: recovered beyond the series
    np.testing.assert_allclose(out.values[:, 0, 0], _design(x_out, modes, trend) @ coef, rtol=0, atol=1e-8)


def test_harmonic_is_robust_to_outliers_and_honours_fit():
    dates = pd.date_range("2018-01-01", periods=230, freq="8D")
    x = _days(dates, dates[0])
    rng = np.random.default_rng(6)
    y = _curve(x) + 0.01 * rng.normal(size=230)
    bad = np.zeros(230, bool)
    bad[::9] = True
    y[bad] -= 0.5  # clouds: far below
    cube = _cube(y[:, None, None] * np.ones((1, 1, 2)), dates)
    out = zeit.interpolate(cube, "harmonic", times=dates)
    robust = np.abs(out.values[:, 0, 0] - _curve(x)).max()
    ols = np.linalg.lstsq(_design(x), y, rcond=None)[0]
    plain = np.abs(_design(x) @ ols - _curve(x)).max()
    assert robust < 0.05 < plain  # the bisquare fit ignores them; the OLS does not
    # the fit window: only the first 3 years are used, and the rest is predicted
    part = zeit.interpolate(cube, "harmonic", times=dates, fit=("2018-01-01", "2020-12-31"))
    inside = x <= (pd.Timestamp("2020-12-31") - dates[0]).days
    assert part.attrs["fit"] == ["2018-01-01", "2020-12-31"]
    assert np.abs(part.values[:, 0, 0] - _curve(x)).max() < 0.1
    full = out.values[:, 0, 0]
    assert not np.allclose(part.values[inside, 0, 0], full[inside], atol=1e-6)
    # a window with fewer observations than the 8 coefficients is NaN
    short = zeit.interpolate(cube, "harmonic", times=dates[:3], fit=("2018-01-01", "2018-02-01"))
    assert np.isnan(short.values).all()
    statsmodels = pytest.importorskip("statsmodels.api")
    from statsmodels.robust.norms import TukeyBiweight

    fit = statsmodels.RLM(y, _design(x), M=TukeyBiweight(4.685)).fit()
    # the scale estimate differs (statsmodels: normalized MAD of all residuals), so close, not equal
    np.testing.assert_allclose(out.values[:, 0, 0], _design(x) @ fit.params, atol=0.03)


def test_harmonic_period_and_time_origin():
    """A period of 365 days with real dates, and the fit is the same wherever the time starts."""
    dates = pd.date_range("2019-03-05", periods=200, freq="9D")
    x = _days(dates, dates[0])
    y = (0.5 + 0.2 * np.sin(2 * np.pi * x / 365.0 + 1.0))[:, None, None]
    cube = _cube(y, dates)
    out = zeit.interpolate(cube, "harmonic", period=365, times=dates)
    np.testing.assert_allclose(out.values[:, 0, 0], y[:, 0, 0], atol=1e-9)
    shifted = zeit.interpolate(_cube(y, dates + pd.to_timedelta(1000, unit="D")), "harmonic", period=365,
                               times=dates + pd.to_timedelta(1000, unit="D"))
    np.testing.assert_allclose(out.values, shifted.values, atol=1e-9)


def test_missing_values_and_nodata(gappy):
    out = zeit.interpolate(gappy, "rbf")
    assert np.isnan(out.values[:, 0, 0]).all()
    assert np.isfinite(out.values[:, 0, 1]).any()  # one observation reaches the dates around it
    far = out.values[:, 0, 1]
    assert np.isnan(far).any()  # but not those beyond the largest kernel
    # NoData values are missing observations
    v = gappy.values.copy()
    v[np.isnan(v)] = -9999
    ints = gappy.copy(data=v)
    ints = ints.rio.write_nodata(-9999)
    np.testing.assert_allclose(zeit.interpolate(ints, "rbf").values, out.values, equal_nan=True, atol=1e-12)


def test_series_input_and_dtype(gappy):
    s = pd.Series(gappy.values[:, 2, 3], index=pd.DatetimeIndex(gappy.time.values), name="ndvi")
    got = zeit.interpolate(s, "linear", step=16)
    assert isinstance(got, pd.Series) and got.name == "ndvi" and (got.index == _grid(gappy)).all()
    cube = zeit.interpolate(gappy, "linear", step=16)
    np.testing.assert_allclose(got.values, cube.values[:, 2, 3], atol=1e-12)
    # numpy with dates; float32 stays float32
    arr = zeit.interpolate(gappy.values[:, 2, 3].astype(np.float32), "linear", dates=gappy.time.values, step=16)
    assert arr.dims == ("time",)
    np.testing.assert_allclose(arr.values, got.values, rtol=1e-6)
    f32 = zeit.interpolate(gappy.astype(np.float32), "rbf")
    assert f32.dtype == np.float32
    np.testing.assert_allclose(f32.values, zeit.interpolate(gappy, "rbf").values, rtol=1e-5, atol=1e-6)


def test_times_replace_the_grid(gappy):
    dates = pd.DatetimeIndex([pd.Timestamp("2018-02-01"), pd.Timestamp("2019-07-04 12:00"), pd.Timestamp("2021-01-01")])
    out = zeit.interpolate(gappy, "linear", times=dates)
    assert (out.time.values == dates.values).all()
    both = zeit.interpolate(gappy, "linear", times=pd.date_range("2018-01-01", periods=3, freq="D"))
    assert both.shape[0] == 3
    with pytest.raises(ValueError, match="either times"):
        zeit.interpolate(gappy, "linear", times=dates, end="2020-01-01")


def test_errors(gappy):
    with pytest.raises(ValueError, match="method"):
        zeit.interpolate(gappy, "spline")
    with pytest.raises(ValueError, match="sigma"):
        zeit.interpolate(gappy, "rbf", sigma=(16, 8))
    with pytest.raises(ValueError, match="cutoff"):
        zeit.interpolate(gappy, "rbf", cutoff=1.5)
    with pytest.raises(ValueError, match="modes"):
        zeit.interpolate(gappy, "harmonic", modes=4)
    with pytest.raises(ValueError, match="step"):
        zeit.interpolate(gappy, "linear", step=0)
    with pytest.raises(ValueError, match="no date"):
        zeit.interpolate(gappy, "linear", step=16, start="2022-01-01", end="2021-01-01")
    backwards = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-03-01", "2020-02-01", "2020-01-01"]))
    with pytest.raises(ValueError, match="do not decrease"):
        zeit.interpolate(backwards, "linear")
    with pytest.raises(ValueError, match="dates"):
        zeit.interpolate(gappy.values, "linear")


@pytest.mark.parametrize("method,kw", [("linear", {}), ("moving", {}), ("rbf", {}),
                                       ("harmonic", {"fit": ("2018-01-01", "2020-12-31")})])
def test_dask_gives_the_same_as_memory(gappy, method, kw):
    mem = zeit.interpolate(gappy, method, n_jobs=2, **kw)
    lazy = zeit.interpolate(gappy.chunk({"y": 2, "x": 2}), method, n_jobs=1, **kw)
    assert lazy.chunks is not None and lazy.shape == mem.shape
    with dask.config.set(scheduler="threads", num_workers=4):
        got = lazy.compute()
    np.testing.assert_allclose(got.values, mem.values, equal_nan=True, atol=1e-12)
    assert (got.time.values == mem.time.values).all() and got.rio.crs == mem.rio.crs


def test_accessor(gappy):
    a = gappy.zeit.interpolate("moving", window=10, step=20)
    b = zeit.interpolate(gappy, "moving", window=10, step=20)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.attrs["window"] == 10.0
