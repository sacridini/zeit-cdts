"""zeit.temporal_metrics (the series' statistics and their folds) and zeit.standardize."""
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import zeit
from zeit._tsa import _periods
from zeit.indices import parse_metrics

ALL = ["min", "max", "range", "iqr", "mean", "std", "skew", "kurtosis", "count", "median", "p10", "p99"]


def _cube(values, dates, crs=True):
    ny, nx = values.shape[1:]
    cube = xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": pd.DatetimeIndex(dates), "y": np.arange(ny)[::-1] + 0.5,
                                "x": np.arange(nx) + 0.5})
    return cube.rio.write_crs(4326) if crs else cube


@pytest.fixture(scope="module")
def noisy():
    """40 dates, (y=6, x=7), about a third missing; one pixel empty, one with a single value."""
    rng = np.random.default_rng(1)
    v = rng.normal(0.5, 0.2, (40, 6, 7)) + rng.exponential(0.1, (40, 6, 7))
    v[rng.random(v.shape) < 0.3] = np.nan
    v[:, 0, 0] = np.nan
    v[:, 0, 1] = np.nan
    v[7, 0, 1] = 0.25
    return _cube(v, pd.date_range("2020-01-01", periods=40, freq="9D"))


def _reference(series, name, ddof):
    """What numpy and scipy give for one pixel's valid values."""
    x = series[~np.isnan(series)]
    if name == "count":
        return float(len(x))
    if len(x) == 0:
        return np.nan
    if name == "std":
        return np.nanstd(x, ddof=ddof) if len(x) > ddof else np.nan
    if name == "skew":
        return stats.skew(x, bias=True) if len(x) > 1 and x.std() > 0 else np.nan
    if name == "kurtosis":
        return stats.kurtosis(x, bias=True) if len(x) > 1 and x.std() > 0 else np.nan
    return {
        "min": np.min, "max": np.max, "range": np.ptp, "mean": np.mean, "median": np.median,
        "iqr": lambda a: np.percentile(a, 75) - np.percentile(a, 25),
        "p10": lambda a: np.percentile(a, 10), "p99": lambda a: np.percentile(a, 99),
    }[name](x)


@pytest.mark.parametrize("ddof", [0, 1])
def test_stm_matches_numpy_and_scipy(noisy, ddof):
    out = zeit.temporal_metrics(noisy, ALL, ddof=ddof, n_jobs=2)
    assert out.dims == ("metric", "y", "x") and list(out.metric.values) == ALL
    assert out.dtype == np.float64 and out.rio.crs.to_epsg() == 4326
    assert (out.y.values == noisy.y.values).all()
    values = noisy.values
    for k, name in enumerate(ALL):
        ref = np.array([[_reference(values[:, j, i], name, ddof) for i in range(7)] for j in range(6)])
        np.testing.assert_allclose(out.values[k], ref, rtol=1e-10, atol=1e-12, err_msg=name)


def test_empty_pixel_single_value_and_float32(noisy):
    out = zeit.temporal_metrics(noisy, ["count", "min", "max", "std", "skew", "kurtosis", "median"], ddof=1)
    empty, one = out.isel(y=0, x=0), out.isel(y=0, x=1)
    assert empty.sel(metric="count") == 0 and np.isnan(empty.drop_sel(metric="count")).all()
    assert one.sel(metric="count") == 1 and one.sel(metric="min") == one.sel(metric="max") == 0.25
    assert one.sel(metric="median") == 0.25
    # a single value: no sample deviation (FORCE leaves it undefined), skew and kurtosis undefined
    assert np.isnan(one.sel(metric=["std", "skew", "kurtosis"])).all()
    assert zeit.temporal_metrics(noisy, "std", ddof=0).isel(y=0, x=1, metric=0) == 0.0
    f32 = zeit.temporal_metrics(noisy.astype("float32"), ["mean", "p25"])
    assert f32.dtype == np.float32


def test_nodata_integer_input_and_pixel_series():
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    ints = np.array([[0, 10, 20, 0, 30, 40]]).reshape(6, 1, 1).astype("int16")
    out = zeit.temporal_metrics(_cube(ints, dates), ["count", "mean"])  # 0 is the NoData of integer data
    assert out.values.ravel().tolist() == [4.0, 25.0]
    out = zeit.temporal_metrics(ints, ["count", "mean"], nodata=None, dates=dates)
    assert out.values.ravel().tolist() == [6.0, 100 / 6]
    series = pd.Series([1.0, np.nan, 3.0, 5.0], index=pd.date_range("2021-01-01", periods=4, freq="MS"))
    pixel = zeit.temporal_metrics(series, ["mean", "max", "count"])
    assert pixel.dims == ("metric",) and pixel.values.tolist() == [3.0, 5.0, 3.0]
    yearly = zeit.temporal_metrics(series, ["mean"], by="quarter")
    assert yearly.dims == ("metric", "quarter") and yearly.values[0, 0] == 2.0 and yearly.values[0, 1] == 5.0


# ---- folds -----------------------------------------------------------------

_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _calendar_doy(ts):
    """FORCE's day of the year: a year of 365 days; 29 February is 60, as is 1 March."""
    return sum(_DAYS[: ts.month - 1]) + ts.day


@pytest.fixture(scope="module")
def daily():
    """Every day of 2019-2021 (2020 is a leap year), the value of a pixel being its date's number."""
    dates = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    n = np.arange(len(dates), dtype=np.float64)
    v = np.stack([n, -n, np.where(n % 3 == 0, np.nan, n)], -1).reshape(len(dates), 1, 3)
    return _cube(v, dates), dates


def test_periods_calendar():
    dates = pd.DatetimeIndex(["2019-03-01", "2020-02-29", "2020-03-01", "2020-12-31", "2021-01-07", "2021-01-08"])
    idx, coord = _periods(dates, "doy")
    assert list(idx + 1) == [60, 60, 60, 365, 7, 8] and len(coord) == 365
    idx, coord = _periods(dates, "week")
    assert list(idx + 1) == [9, 9, 9, 52, 1, 2] and len(coord) == 52  # 364 and 365 are in week 52
    assert list(_periods(dates, "month")[0] + 1) == [3, 2, 3, 12, 1, 1]
    assert list(_periods(dates, "quarter")[0] + 1) == [1, 1, 1, 4, 1, 1]
    idx, coord = _periods(dates, "year")
    assert list(coord) == [2019, 2020, 2021] and list(idx) == [0, 1, 1, 1, 2, 2]


@pytest.mark.parametrize("by,size", [("year", 3), ("quarter", 4), ("month", 12), ("week", 52), ("doy", 365)])
def test_fold_matches_a_groupby(daily, by, size):
    cube, dates = daily
    out = zeit.temporal_metrics(cube, ["count", "mean", "min", "max", "p75"], by=by, n_jobs=2)
    assert out.dims == ("metric", by, "y", "x") and out.sizes[by] == size
    doy = np.array([_calendar_doy(t) for t in dates])
    key = {"year": dates.year, "quarter": dates.quarter, "month": dates.month, "doy": doy,
           "week": np.minimum(-(-doy // 7), 52)}[by]
    first = 2019 if by == "year" else 1
    for pixel in range(3):
        series = cube.values[:, 0, pixel]
        for g in range(size):
            x = series[(np.asarray(key) == g + first) & ~np.isnan(series)]
            got = out.isel(y=0, x=pixel, **{by: g}).values
            assert got[0] == len(x)
            if len(x):
                np.testing.assert_allclose(got[1:], [x.mean(), x.min(), x.max(), np.percentile(x, 75)], rtol=1e-12)
            else:
                assert np.isnan(got[1:]).all()
    assert (out[by].values == np.arange(first, first + size)).all()


def test_the_leap_day_shares_day_60_with_march_1st(daily):
    cube, _ = daily
    out = zeit.temporal_metrics(cube.isel(x=[0]), ["count", "max"], by="doy")
    day60 = out.isel(y=0, x=0).sel(doy=60)
    # 1 March of three years and 29 February 2020: four dates; 29 February is date number 424
    assert day60.sel(metric="count") == 4 and day60.sel(metric="max") == (pd.Timestamp("2021-03-01") - pd.Timestamp("2019-01-01")).days
    assert out.isel(y=0, x=0).sel(doy=365, metric="count") == 3  # 31 December: three, the 31st of 2020 included
    assert out.isel(y=0, x=0).sel(doy=59, metric="count") == 3  # 28 February


def test_fold_by_year_of_a_gappy_series():
    dates = pd.to_datetime(["2018-06-01", "2018-07-01", "2020-06-01", "2020-06-20"])
    v = np.array([1.0, 3.0, 5.0, np.nan]).reshape(4, 1, 1)
    out = zeit.temporal_metrics(_cube(v, dates), ["count", "mean", "std", "range"], by="year", ddof=1)
    assert list(out.year.values) == [2018, 2019, 2020]
    got = out.isel(y=0, x=0).values
    assert got[0].tolist() == [2, 0, 1]
    assert got[1][0] == 2.0 and np.isnan(got[1][1]) and got[1][2] == 5.0
    assert got[2][0] == pytest.approx(np.sqrt(2)) and np.isnan(got[2][1:]).all()
    assert got[3][0] == 2.0 and got[3][2] == 0.0


def test_errors_and_names(noisy):
    with pytest.raises(ValueError, match="by must be"):
        zeit.temporal_metrics(noisy, "mean", by="decade")
    with pytest.raises(ValueError, match="dates"):
        zeit.temporal_metrics(np.zeros((5, 2, 2)), "mean", by="year")
    with pytest.raises(ValueError, match="Unknown metric"):
        zeit.temporal_metrics(noisy, ["variance"])
    assert parse_metrics(["Range", "SKEW", "kurtosis", "P05", "p5"]) == ["range", "skew", "kurtosis", "p5"]
    with pytest.raises(ValueError):
        parse_metrics(["skew"], extended=False)
    # without dates a plain stack works for the whole series
    assert zeit.temporal_metrics(np.arange(10.0).reshape(10, 1, 1), ["mean"]).values.item() == 4.5


# ---- dask ------------------------------------------------------------------

def test_dask_gives_the_same_as_memory(noisy, daily):
    with dask.config.set(scheduler="threads", num_workers=4):
        chunked = noisy.chunk({"time": 10, "y": 3, "x": 4})
        for by, cube in ((None, chunked), ("month", chunked)):
            lazy = zeit.temporal_metrics(cube, ALL, by=by, n_jobs=2)
            assert isinstance(lazy.data, da.Array)
            ref = zeit.temporal_metrics(noisy, ALL, by=by, n_jobs=2)
            assert lazy.dims == ref.dims
            np.testing.assert_array_equal(lazy.values, ref.values)
            assert lazy.rio.crs.to_epsg() == 4326 and (lazy.x.values == ref.x.values).all()
        via_chunks = zeit.temporal_metrics(noisy, ["mean"], chunks={"y": 2, "x": 3})
        assert isinstance(via_chunks.data, da.Array)
        np.testing.assert_array_equal(via_chunks.values, zeit.temporal_metrics(noisy, ["mean"]).values)
        z = zeit.standardize(chunked, n_jobs=2)
        assert isinstance(z.data, da.Array)
        np.testing.assert_array_equal(z.values, zeit.standardize(noisy, n_jobs=2).values)


# ---- standardize -----------------------------------------------------------

def test_standardize_matches_scipy(noisy):
    values = noisy.values
    z = zeit.standardize(noisy)
    c = zeit.standardize(noisy, "center", n_jobs=2)
    assert z.dims == noisy.dims and (z.time.values == noisy.time.values).all() and z.rio.crs.to_epsg() == 4326
    for j in range(6):
        for i in range(7):
            x = values[:, j, i]
            ok = ~np.isnan(x)
            if ok.sum() < 2:
                np.testing.assert_array_equal(z.values[:, j, i], x)
                np.testing.assert_array_equal(c.values[:, j, i], x)
                continue
            np.testing.assert_allclose(z.values[ok, j, i], stats.zscore(x[ok], ddof=1), rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(c.values[ok, j, i], x[ok] - x[ok].mean(), rtol=1e-10, atol=1e-12)
            assert np.isnan(z.values[~ok, j, i]).all()
    np.testing.assert_allclose(zeit.standardize(noisy, ddof=0).values[:, 3, 3],
                               np.where(np.isnan(values[:, 3, 3]), np.nan,
                                        (values[:, 3, 3] - np.nanmean(values[:, 3, 3])) / np.nanstd(values[:, 3, 3])),
                               rtol=1e-10, equal_nan=True)


def test_standardize_edge_cases():
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    v = np.array([[5.0, 2.0, 2.0], [np.nan, 2.0, 1.0], [np.nan, 2.0, 3.0], [np.nan, 2.0, np.nan]]).reshape(4, 1, 3)
    cube = _cube(v, dates)
    z = zeit.standardize(cube)
    assert z.values[:, 0, 0].tolist()[:1] == [5.0] and np.isnan(z.values[1:, 0, 0]).all()  # one value: as it is
    assert np.isnan(z.values[:, 0, 1]).all()                                           # constant: undefined
    np.testing.assert_allclose(z.values[:, 0, 2], [0.0, -1.0, 1.0, np.nan])
    assert np.allclose(zeit.standardize(cube, "center").values[:, 0, 1], 0.0)
    series = pd.Series([1.0, 2.0, np.nan, 3.0], index=dates, name="ndvi")
    out = zeit.standardize(series)
    assert isinstance(out, pd.Series) and out.name == "ndvi" and (out.index == dates).all()
    np.testing.assert_allclose(out.dropna().values, [-1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="method"):
        zeit.standardize(cube, "zscore")


def test_accessor(noisy):
    out = noisy.zeit.temporal_metrics(["mean", "p90"], by="quarter")
    ref = zeit.temporal_metrics(noisy, ["mean", "p90"], by="quarter")
    np.testing.assert_array_equal(out.values, ref.values)
    np.testing.assert_array_equal(noisy.zeit.standardize("center").values, zeit.standardize(noisy, "center").values)
    assert "temporal_metrics" in zeit.__all__ and "standardize" in zeit.__all__
