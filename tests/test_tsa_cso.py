"""zeit.clear_observations (FORCE's CSO): the clear observations in windows of months and the gaps between them."""
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import zeit

NAN = np.nan


def _cube(values, dates):
    n, ny, nx = values.shape
    return xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": pd.DatetimeIndex(dates), "y": np.arange(ny)[::-1] + 0.5,
                                "x": np.arange(nx) + 0.5}).rio.write_crs(4326)


def _one(days, valid=None):
    """One pixel's series on the given dates (a pandas Series): 1.0 where valid, NaN elsewhere."""
    ok = np.ones(len(days), bool) if valid is None else np.asarray(valid, bool)
    return pd.Series(np.where(ok, 1.0, NAN), index=pd.DatetimeIndex(days))


# Three windows of the leap year 2020: January 31 days, February 29, March 31.
HAND = ["2020-01-03", "2020-01-10", "2020-01-10", "2020-01-20", "2020-03-01", "2020-03-31"]
ALL = ["count", "min", "max", "range", "mean", "std", "skew", "kurtosis", "median", "p25", "iqr"]


def _stats(gaps, n):
    g = np.asarray(gaps, dtype=float)
    q25, q50, q75 = np.percentile(g, [25, 50, 75])
    return {"count": n, "min": g.min(), "max": g.max(), "range": g.max() - g.min(), "mean": g.mean(),
            "std": g.std(ddof=1) if len(g) > 1 else NAN,
            "skew": stats.skew(g) if g.std() > 0 else NAN,
            "kurtosis": stats.kurtosis(g) if g.std() > 0 else NAN,
            "median": q50, "p25": q25, "iqr": q75 - q25}


def test_intervals_by_hand_with_a_gap_a_duplicate_and_the_start_of_a_window():
    out = zeit.clear_observations(_one(HAND), metrics=["count", "min", "max", "mean"], end="2020-03-31")
    assert out.dims == ("metric", "window")
    assert list(out.metric.values) == ["count", "min", "max", "mean"]
    assert list(out.window.values) == list(pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    assert list(out.window_end.values) == list(pd.to_datetime(["2020-02-01", "2020-03-01", "2020-04-01"]))
    # January: intervals 2, 7, (the same day: none), 10 and 12 to the end; count 4
    # February: nothing, one interval of the 29 days; count 0
    # March: the 1st is on the start (counted, no interval), then 30 and 1 to the end; count 2
    assert out.sel(metric="count").values.tolist() == [4, 0, 2]
    assert out.sel(metric="min").values.tolist() == [2, 29, 1]
    assert out.sel(metric="max").values.tolist() == [12, 29, 30]
    np.testing.assert_allclose(out.sel(metric="mean").values, [31 / 4, 29, 31 / 2])


def test_every_metric_of_the_hand_made_windows_against_numpy():
    out = zeit.clear_observations(_one(HAND), metrics=ALL, end="2020-03-31")
    for j, (gaps, n) in enumerate([([2, 7, 10, 12], 4), ([29], 0), ([30, 1], 2)]):
        want = _stats(gaps, n)
        for name in ALL:
            got = float(out.sel(metric=name).values[j])
            assert got == pytest.approx(want[name], nan_ok=True, rel=1e-12), (j, name)


def test_a_window_without_observations_is_one_interval_the_length_of_the_window():
    out = zeit.clear_observations(_one(["2020-01-05", "2020-04-20"]), months=1, metrics=["count", "min", "max", "std"],
                                  end="2020-04-30")
    feb, mar = out.isel(window=1), out.isel(window=2)
    for w, days in ((feb, 29), (mar, 31)):
        assert w.sel(metric="count").item() == 0 and w.sel(metric="min").item() == days
        assert w.sel(metric="max").item() == days and np.isnan(w.sel(metric="std").item())
    # an observation that is not valid does not count either
    out = zeit.clear_observations(_one(["2020-01-05", "2020-01-25"], [True, False]), metrics=["count", "max"])
    assert out.sel(metric="count").item() == 1 and out.sel(metric="max").item() == 27


def test_nan_the_nodata_value_and_a_boolean_cube_are_the_same_observations():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", "2020-04-30", freq="3D")
    v = rng.random((len(dates), 4, 5))
    v[rng.random(v.shape) < 0.4] = NAN
    v[:, 0, 0] = NAN
    cube = _cube(v, dates)
    want = zeit.clear_observations(cube, metrics=ALL, n_jobs=2)
    boolean = zeit.clear_observations(cube.notnull(), metrics=ALL, n_jobs=2)
    xr.testing.assert_identical(boolean, want)
    filled = _cube(np.where(np.isnan(v), -9999.0, v), dates)
    xr.testing.assert_allclose(zeit.clear_observations(filled, metrics=ALL, nodata=-9999), want)
    # an integer 0/1 mask: 0 is the NoData of an integer cube
    ints = _cube((~np.isnan(v)).astype(np.uint8), dates)
    xr.testing.assert_allclose(zeit.clear_observations(ints, metrics=ALL), want)
    assert want.sel(metric="count").values[:, 0, 0].tolist() == [0, 0, 0, 0]
    assert want.dims == ("metric", "window", "y", "x")
    assert want.rio.crs.to_epsg() == 4326 and (want.y.values == cube.y.values).all()


def test_the_same_day_twice_counts_twice_but_makes_one_interval():
    once = zeit.clear_observations(_one(["2020-02-05", "2020-02-20"]), metrics=["count", "max", "mean"],
                                   start="2020-02-01", end="2020-02-29")
    twice = zeit.clear_observations(_one(["2020-02-05", "2020-02-05", "2020-02-20"]),
                                    metrics=["count", "max", "mean"], start="2020-02-01", end="2020-02-29")
    assert once.sel(metric="count").item() == 2 and twice.sel(metric="count").item() == 3
    assert once.sel(metric=["max", "mean"]).values.tolist() == twice.sel(metric=["max", "mean"]).values.tolist()


@pytest.fixture(scope="module")
def noisy():
    """An irregular series from November 2019 to March 2021 (the leap February among it), (y=4, x=5), about
    half of the observations missing, a pixel with none and one with all."""
    rng = np.random.default_rng(7)
    days = np.unique(rng.integers(0, 500, 150))
    dates = pd.Timestamp("2019-11-14") + pd.to_timedelta(days, unit="D")
    v = np.ones((len(dates), 4, 5))
    v[rng.random(v.shape) < 0.5] = NAN
    v[:, 0, 0] = NAN
    v[:, 0, 1] = 1.0
    return _cube(v, dates)


def _windows(first, last, months):
    month0 = first.year * 12 + first.month - 1
    n = (last.year * 12 + last.month - 1 - month0) // months + 1
    return [pd.Timestamp(int(b) // 12, int(b) % 12 + 1, 1) for b in month0 + months * np.arange(n + 1)]


@pytest.mark.parametrize("months", [1, 3, 5, 12])
def test_every_metric_matches_numpy_on_the_intervals_made_by_hand(noisy, months):
    out = zeit.clear_observations(noisy, months=months, metrics=ALL, n_jobs=2)
    dates = pd.DatetimeIndex(noisy.time.values)
    edges = _windows(dates[0], dates[-1], months)
    assert out.sizes["window"] == len(edges) - 1
    assert list(out.window.values) == list(pd.DatetimeIndex(edges[:-1]))
    assert list(out.window_end.values) == list(pd.DatetimeIndex(edges[1:]))
    values = noisy.values
    checked = 0
    for j in range(4):
        for i in range(5):
            for w, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                days = [d for d, ok in zip(dates, np.isfinite(values[:, j, i])) if ok and a <= d < b]
                gaps, left = [], a
                for d in days:
                    if d > left:
                        gaps.append((d - left).days)
                        left = d
                gaps.append((b - left).days)
                want = _stats(gaps, len(days))
                for name in ALL:
                    got = float(out.sel(metric=name).values[w, j, i])
                    assert got == pytest.approx(want[name], nan_ok=True, rel=1e-9, abs=1e-9), (months, w, j, i, name)
                checked += 1
    assert checked == out.sizes["window"] * 20
    # the windows of a pixel tile their span: the intervals add up to the days of the window
    assert out.sel(metric="mean").values.shape[1:] == (4, 5)


def test_the_february_of_a_leap_year_has_29_days_and_the_quarter_91():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    nothing = _cube(np.full((len(dates), 1, 1), NAN), dates)
    monthly = zeit.clear_observations(nothing, months=1, metrics=["max"])
    assert monthly.sizes["window"] == 12   # the December window too (FORCE loses it)
    assert monthly.values[0, :, 0, 0].tolist() == [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    quarterly = zeit.clear_observations(nothing, months=3, metrics=["max"])
    assert quarterly.values[0, :, 0, 0].tolist() == [91, 91, 92, 92]
    assert quarterly.window.dt.month.values.tolist() == [1, 4, 7, 10]
    assert zeit.clear_observations(nothing, months=12, metrics=["max"]).values[0, :, 0, 0].tolist() == [366]
    # a window across the February of a leap year, from November
    across = zeit.clear_observations(_one(["2019-11-20", "2020-03-10"]), months=5, metrics=["max", "count"],
                                     start="2019-11-30", end="2020-03-31")
    assert across.window.dt.month.values.tolist() == [11]
    # the five months (152 days) cut by the 20 November and the 10 March: 19, 111 and 22 days
    assert across.sel(metric="max").item() == 111 and across.sel(metric="count").item() == 2
    assert zeit.clear_observations(_one(["2019-11-20", "2020-03-10"]), months=5, metrics=["mean"], start="2019-11-30",
                                   end="2020-03-31").item() == pytest.approx(152 / 3)


def test_start_and_end_choose_the_windows():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="5D")
    cube = _cube(np.ones((len(dates), 2, 2)), dates)
    out = zeit.clear_observations(cube, start="2020-03-17", end="2020-06-02", metrics=["count"])
    assert out.window.dt.strftime("%Y-%m-%d").values.tolist() == ["2020-03-01", "2020-04-01", "2020-05-01",
                                                                  "2020-06-01"]
    full = zeit.clear_observations(cube, metrics=["count"])
    assert full.sizes["window"] == 12 and int(full.sum()) == 4 * len(dates)
    with pytest.raises(ValueError, match="before start"):
        zeit.clear_observations(cube, start="2020-06-01", end="2020-05-01")


def test_dask_gives_the_same_result_as_numpy(noisy):
    want = zeit.clear_observations(noisy, months=3, metrics=ALL, n_jobs=2)
    lazy = noisy.chunk({"y": 2, "x": 3})
    got = zeit.clear_observations(lazy, months=3, metrics=ALL, n_jobs=2)
    assert isinstance(got.data, da.Array) and got.dims == want.dims
    xr.testing.assert_identical(got.compute(), want)
    boolean = zeit.clear_observations(lazy.notnull(), months=3, metrics=ALL, n_jobs=2)
    xr.testing.assert_identical(boolean.compute(), want)


def test_float32_and_the_accessor(noisy):
    f32 = noisy.astype(np.float32)
    out = zeit.clear_observations(f32, metrics=["count", "mean"], n_jobs=2)
    assert out.dtype == np.float32
    via = f32.zeit.clear_observations(metrics=["count", "mean"], n_jobs=2)
    xr.testing.assert_identical(via, out)
    assert out.attrs["algorithm"] == "clear_observations" and out.attrs["months"] == 1


def test_a_numpy_stack_with_dates_and_a_series():
    dates = pd.date_range("2020-01-01", periods=40, freq="4D")
    stack = np.ones((40, 2, 3))
    stack[5:20, 0, 0] = NAN
    out = zeit.clear_observations(stack, dates=dates, metrics=["count", "max"])
    assert out.dims == ("metric", "window", "y", "x") and out.sizes["window"] == 6
    series = zeit.clear_observations(pd.Series(stack[:, 0, 0], index=dates), metrics=["count", "max"])
    assert series.dims == ("metric", "window")
    np.testing.assert_array_equal(series.values, out.values[:, :, 0, 0])
    listed = zeit.clear_observations([True, False, True], dates=pd.date_range("2020-01-01", periods=3, freq="10D"),
                                     metrics=["count"])
    assert listed.values.tolist() == [[2]]


def test_input_errors():
    cube = _cube(np.ones((3, 2, 2)), pd.date_range("2020-01-01", periods=3, freq="10D"))
    for months in (0, 13, 1.5, True):
        with pytest.raises(ValueError, match="months"):
            zeit.clear_observations(cube, months=months)
    with pytest.raises(ValueError, match="Unknown metric"):
        zeit.clear_observations(cube, metrics=["variance"])
    with pytest.raises(ValueError, match="do not decrease"):
        zeit.clear_observations(pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-03-01", "2020-02-01", "2020-01-01"])))
    with pytest.raises(ValueError, match="dates"):
        zeit.clear_observations(np.ones((3, 2, 2)))
