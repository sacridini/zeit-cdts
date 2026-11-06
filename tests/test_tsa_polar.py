"""zeit.polar_metrics (FORCE's polar phenology metrics): the dates, values, integrals and rates of each year's season."""
import inspect

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._tsa import POLAR_NAMES

NAN = np.nan
# Every 5 days from 5 January 2021 to 31 December 2023: the days of the year 5, 10 ... 365 of three years
# without a leap day, so the date of a step is its day in FORCE's calendar of 365 days.
DATES = pd.date_range("2021-01-05", "2023-12-31", freq="5D")
DOY = np.tile(np.arange(5, 366, 5), 3)
YEAR = np.repeat([0, 1, 2], 73)
ALL = list(POLAR_NAMES)


def _one(values, dates=DATES):
    return pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(dates))


def _cube(pixels, dates=DATES, dtype=np.float64):
    """A (time, 2, 2) cube from four series."""
    values = np.stack([np.asarray(p, dtype=dtype) for p in pixels], axis=1).reshape(len(dates), 2, 2)
    return xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": pd.DatetimeIndex(dates), "y": [1.5, 0.5], "x": [0.5, 1.5]}
                        ).rio.write_crs(4326)


def _box(first=100, last=190, height=100.0):
    return np.where((DOY >= first) & (DOY <= last), height, 0.0)


def _triangle():
    k = (DOY - 100) // 5
    return np.where((DOY >= 100) & (DOY <= 140), 10.0 * (1 + np.minimum(k, 8 - k)), 0.0)


def test_a_symmetric_box_by_hand():
    # Symmetric about day 145: the long-term mean vector at 145, the start of the year at 327.5 -> 327, the
    # seasons start on day 330 and each belongs to the year it ends in; 2021 has none. The box has 19 steps of
    # 100: START (fraction >= 0.2) is the 4th, MID the 10th, END the 16th.
    out = zeit.polar_metrics(_one(_box()), ALL)
    assert out.dims == ("metric", "year")
    assert list(out.year.values) == [2021, 2022, 2023]
    assert list(out.metric.values) == ALL
    assert np.isnan(out.sel(year=2021)).all()
    want = {"DEM": 100, "DSS": 115, "DMS": 145, "DPS": 115, "DES": 175, "DLM": 330, "LTS": 230, "LGS": 60,
            "DPY": 327 - 365, "DPV": 0, "VEM": 100, "VSS": 100, "VMS": 100, "VPS": 100, "VES": 100, "VLM": 0,
            "VBL": 50, "VSA": 50, "VGA": 0, "VPA": 0, "VGM": 100, "VGV": 0, "IST": 4750 / 365, "IBL": 4750 / 365,
            "IBT": 9500 / 365, "IGS": 0, "IRR": 500 / 30, "IFR": 500 / 30, "RAR": 0, "RAF": 0, "RMR": 0, "RMF": 0}
    for year in (2022, 2023):
        got = out.sel(year=year)
        for name, value in want.items():
            assert float(got.sel(metric=name)) == pytest.approx(value, abs=1e-5), (year, name)
        # the mean vector of the 12 steps of days 115..170 points to their middle and has the length
        # sin(12 d / 2) / (12 sin(d / 2)) of 100, d = 2 pi 5 / 365
        d = 2 * np.pi * 5 / 365
        assert float(got.sel(metric="VAV")) == pytest.approx(100 * np.sin(6 * d) / (12 * np.sin(d / 2)), abs=0.01)
        assert float(got.sel(metric="DAV")) == pytest.approx(142.5, abs=1)
        assert float(got.sel(metric="DEV")) == pytest.approx(127.5, abs=1)
        assert float(got.sel(metric="DLV")) == pytest.approx(157.5, abs=1)


def test_an_asymmetric_season_by_hand_with_rates():
    # 10 20 30 40 50 40 30 20 10 on days 100..140: start of the year 302, seasons from day 305; the fractions
    # .04 .12 .24 .4 .6 .76 .88 .96 1 put START on the 3rd step (day 110, value 30), MID on the 5th (120, 50),
    # END on the 7th (130, 30). The growing season holds 30 40 50 40.
    out = zeit.polar_metrics(_one(_triangle()), ["DSS", "dms", "DES", "DPS", "VPS", "VGM", "VGV", "VSA", "RAR",
                                                 "RAF", "RMR", "RMF", "IGS", "IRR", "IFR", "DPY"])
    assert list(out.metric.values)[1] == "DMS"
    want = {"DSS": 110, "DMS": 120, "DES": 130, "DPS": 120, "VPS": 50, "VGM": 40, "VGV": np.sqrt(200 / 3),
            "VSA": 45,
            # early (fractions [0.2, 0.5)): two rises of 10 in 5 days, rate 2 per day, 60 a month
            "RAR": 60, "RMR": 60,
            # late ([0.5, 0.8)): the step of 50 (a rise) and the step of 40 (a fall of 10): the rates add up
            # to 2 over both steps (FORCE divides by all the steps of the window: 1), the maximum is 2
            "RAF": 30, "RMF": 60,
            "IGS": (10 + 20 + 10) * 5 / 365, "IRR": 250 / 30, "IFR": 250 / 30, "DPY": 302 - 365}
    for year in (2022, 2023):
        for name, value in want.items():
            assert float(out.sel(year=year, metric=name)) == pytest.approx(value, abs=1e-5), (year, name)


def test_adaptive_moves_the_start_of_the_year():
    # Boxes on days 100-190 in 2021 and 2022, 115-205 in 2023: the long-term start of the year is 332, the
    # seasons start on day 335. Season 1's own start is day 342, 10 days later: two whole steps of it
    # (days 335 and 340 of 2022) go to season 0, which then runs 10 days longer; season 0's own start is 327.
    v = _box() + np.where(YEAR == 2, _box(115, 205) - _box(), 0.0)
    fixed = zeit.polar_metrics(_one(v), ALL, adaptive=False)
    adapt = zeit.polar_metrics(_one(v), ALL, adaptive=True)
    for name in ("DEM", "DSS", "DMS", "DES", "DPS", "VBL", "IST", "DPY"):
        assert adapt.sel(metric=name).values[1:] == pytest.approx(fixed.sel(metric=name).values[1:])
    assert list(fixed.sel(metric="DSS").values[1:]) == [115, 130]
    assert list(fixed.sel(metric="DPY").values[1:]) == [332 - 365] * 2
    assert list(fixed.sel(metric="DLM").values[1:]) == [335, 335]
    assert list(adapt.sel(metric="DLM").values[1:]) == [345, 335]
    assert list(fixed.sel(metric="LTS").values[1:]) == [235, 220]
    assert list(adapt.sel(metric="LTS").values[1:]) == [245, 220]
    # the shift of each start: 327 - 332 and 342 - 332; none without adaptive
    assert list(adapt.sel(metric="DPV").values[1:]) == [-5, 10]
    assert list(fixed.sel(metric="DPV").values[1:]) == [0, 0]
    assert adapt.attrs["adaptive"] == 1 and fixed.attrs["adaptive"] == 0


def test_thresholds_move_the_dates():
    v = _one(_box())
    out = zeit.polar_metrics(v, ["DSS", "DMS", "DES"], start=0.1, mid=0.5, end=0.9)
    # 10% of 19 steps: the 2nd (day 105); 90%: the 18th (day 185)
    assert list(out.sel(year=2022).values) == [105, 145, 185]
    with pytest.raises(ValueError, match="between 0 and 1"):
        zeit.polar_metrics(v, ["DSS"], end=1.0)


def test_missing_steps_are_interpolated_and_empty_pixels_are_nan():
    base = _box()
    holes = base.copy()
    holes[[0, 3, 4, 75, 76, 100, 200, 218]] = NAN       # zeros and, at 100, a step inside the box
    holes[101] = np.inf
    cube = _cube([base, holes, np.full(len(DATES), NAN), np.zeros(len(DATES))])
    out = zeit.polar_metrics(cube, ["DSS", "VPS", "IST", "LTS"])
    assert out.dims == ("metric", "year", "y", "x")
    a, b = out.isel(y=0, x=0), out.isel(y=0, x=1)
    np.testing.assert_allclose(a.values, b.values, equal_nan=True)     # the same as without holes
    assert float(a.sel(metric="DSS", year=2022)) == 115
    assert np.isnan(out.isel(y=1, x=0)).all()                         # no valid step
    assert np.isnan(out.isel(y=1, x=1)).all()                         # a series of zeros: a season total of 0
    # the NoData value is a missing step too
    cube2 = cube.copy()
    cube2[100, 0, 1] = -9999
    out2 = zeit.polar_metrics(cube2, ["DSS", "VPS", "IST", "LTS"], nodata=-9999)
    np.testing.assert_allclose(out2.isel(y=0, x=1).values, a.values, equal_nan=True)


def _seasonal(start, steps=125):
    """A smooth 16-day seasonal series (peak in July) that is the same function of the day of the year wherever
    it starts: its long-term mean vector is at about day 184, so the phenological year starts at the turn of
    the year."""
    dates = pd.date_range(start, periods=steps, freq="16D")
    doy = np.minimum(dates.dayofyear, 365)
    return _one(0.16 + 0.52 * np.exp(-((doy - 190) / 50.0) ** 2), dates)


@pytest.mark.parametrize("start", ["2016-01-01", "2016-02-10", "2016-03-01", "2016-06-01", "2015-12-01"])
def test_a_series_that_does_not_start_on_1_january_measures_the_seasons_it_covers(start):
    # The start of the phenological year falls before the first step (February to June) or the series begins
    # in December: FORCE finds no season at all there (every metric 0), here every season whose start is in
    # the series is measured, and as in the series from 1 January.
    names = ["DSS", "DPS", "DES", "VPS", "VSS", "VES", "LGS", "IGS"]
    out = zeit.polar_metrics(_seasonal(start), names, n_jobs=2)
    ref = zeit.polar_metrics(_seasonal("2016-01-01"), names, n_jobs=2)
    years = [2017, 2018, 2019, 2020]               # fully inside every series
    assert set(years) <= set(out.year.values)
    for year in years:
        got, want = out.sel(year=year), ref.sel(year=year)
        assert np.isfinite(got.values).all(), (start, year, got.values)
        # the steps are on another grid: the same season to two steps
        for name, tol in (("DSS", 36), ("DPS", 36), ("DES", 36), ("LGS", 36), ("VPS", 0.03), ("VSS", 0.15),
                          ("VES", 0.15)):
            assert abs(float(got.sel(metric=name)) - float(want.sel(metric=name))) <= tol, (start, year, name)
    # the first band is the season that ends before the first start, with no steps of its own
    assert np.isnan(out.sel(year=out.year.values[0])).all()


def test_a_box_that_starts_after_its_first_start_of_the_year_is_as_the_whole_series():
    # The seasons start on day 330 (window 327..332): a series from day 335 has no step in the window of
    # the first year, so that season is NaN and the others are the whole series' (FORCE: every season lost).
    whole = zeit.polar_metrics(_one(_box()), ALL)
    cut = _one(_box())[DATES >= "2021-12-01"]
    out = zeit.polar_metrics(cut, ALL)
    assert list(out.year.values) == [2021, 2022, 2023]
    assert np.isnan(out.sel(year=[2021, 2022])).all()
    np.testing.assert_allclose(out.sel(year=2023).values, whole.sel(year=2023).values, equal_nan=True)


def test_negative_values_count_as_zero():
    a = zeit.polar_metrics(_one(_box()), ALL)
    b = zeit.polar_metrics(_one(_box() - np.where(_box() == 0, 50, 0)), ALL)
    np.testing.assert_allclose(b.values, a.values, equal_nan=True)


def test_an_irregular_series_is_refused_and_points_to_interpolate():
    v = _one(_box())
    with pytest.raises(ValueError, match=r"zeit\.interpolate"):
        zeit.polar_metrics(v.drop(v.index[50]))
    with pytest.raises(ValueError, match="regular"):
        zeit.polar_metrics(_one(np.arange(40.0), pd.date_range("2021-01-01", periods=40, freq="MS")))
    with pytest.raises(ValueError, match="at least 2 calendar years"):
        zeit.polar_metrics(_one(np.arange(73.0), DATES[:73]))
    with pytest.raises(ValueError, match="unknown polar metrics"):
        zeit.polar_metrics(v, ["XYZ"])


def test_float32_dask_and_the_accessor_agree_with_the_in_memory_float64_result():
    rng = np.random.default_rng(3)
    shift = rng.integers(0, 40, size=(4, 5))
    t = DOY[:, None, None]
    y = YEAR[:, None, None]
    values = np.floor(60 + 90 * np.exp(-(((t - (150 + shift + 3 * y)) / 40.0) ** 2)))
    values[rng.random(values.shape) < 0.03] = NAN
    cube = xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": DATES, "y": np.arange(4)[::-1] + 0.5, "x": np.arange(5) + 0.5})
    names = ["DSS", "DES", "VPS", "IGS", "RAR", "DEV", "LGV"]
    ref = zeit.polar_metrics(cube, names)
    assert ref.dtype == np.float64 and ref.dims == ("metric", "year", "y", "x")
    # float32 input: the same up to float32 rounding of the results
    f32 = zeit.polar_metrics(cube.astype(np.float32), names)
    assert f32.dtype == np.float32
    np.testing.assert_allclose(f32.values, ref.values, rtol=1e-5, atol=1e-4, equal_nan=True)
    lazy = zeit.polar_metrics(cube.chunk({"y": 2, "x": 3}), names)
    assert isinstance(lazy.data, da.Array)
    np.testing.assert_array_equal(lazy.compute().values, ref.values)
    lazy32 = zeit.polar_metrics(cube.astype(np.float32).chunk({"y": 3, "x": 2, "time": 100}), names)
    np.testing.assert_array_equal(lazy32.compute().values, f32.values)
    acc = cube.zeit.polar_metrics(names)
    np.testing.assert_array_equal(acc.values, ref.values)
    assert np.isfinite(ref.values).mean() > 0.4
    # threads do not change the result
    one = zeit.polar_metrics(cube, names, n_jobs=1)
    np.testing.assert_array_equal(one.values, ref.values)


def test_linear_trend_of_a_date_along_the_years_is_the_change_per_year():
    dates = pd.date_range("2021-01-05", periods=73 * 7, freq="5D")
    doy = np.asarray(dates.dayofyear)
    year = np.asarray(dates.year) - 2021
    # a season that begins 5 days later every year
    v = np.where((doy >= 100 + 5 * year) & (doy <= 190 + 5 * year), 100.0, 0.0)
    out = zeit.polar_metrics(_cube([v] * 4, dates), ["DSS", "VPS"], adaptive=False)
    ny = int(dates.year[-1]) - 2020
    assert out.sizes["year"] == ny
    dss = out.sel(metric="DSS")
    trend = zeit.linear_trend(dss, dim="year")
    series = dss.isel(y=0, x=0).values
    ok = np.isfinite(series)
    assert ok.sum() >= 5
    slope, offset = np.polyfit(np.arange(len(series))[ok], series[ok], 1)
    assert float(trend.slope.isel(y=0, x=0)) == pytest.approx(slope, abs=1e-6)
    assert float(trend.slope.isel(y=0, x=0)) == pytest.approx(5, abs=1.5)
    assert float(trend.num.isel(y=0, x=0)) == ok.sum()
    assert float(trend.len.isel(y=0, x=0)) == ny
    # a metric axis of length 1 is dropped by linear_trend, but more than one must be chosen
    with pytest.raises(ValueError, match="metrics"):
        zeit.linear_trend(out, dim="year")


def test_the_docstring_lists_the_39_metrics():
    doc = inspect.cleandoc(zeit.polar_metrics.__doc__)      # Python 3.13+ dedents docstrings itself
    assert len(ALL) == 39 and len(set(ALL)) == 39
    for name in ALL:
        assert f"\n{name} " in doc, name
