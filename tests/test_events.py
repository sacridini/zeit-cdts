"""extract_events on every change algorithm, and zeit.agreement between their maps."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

DATES = pd.date_range("2010-01-01", periods=23 * 8, freq="16D")
BREAK = 23 * 5   # DATES[115] = 2015-01-14 is the first date after the drop; DATES[114] = 2014-12-29


@pytest.fixture(scope="module")
def ndvi():
    """16-day NDVI-like series, (time, y=3, x=4): row 0 drops by 0.3 from DATES[BREAK] on."""
    rng = np.random.default_rng(0)
    t = np.arange(len(DATES))
    base = 0.5 + 0.2 * np.sin(2 * np.pi * t / 23)
    stack = np.stack([base + rng.normal(0, 0.02, len(t)) for _ in range(12)], -1).reshape(len(t), 3, 4)
    stack[BREAK:, 0, :] -= 0.3
    return xr.DataArray(stack.astype(np.float32), dims=("time", "y", "x"),
                        coords={"time": DATES, "y": [3.5, 2.5, 1.5], "x": [0.5, 1.5, 2.5, 3.5]}).rio.write_crs(4326)


@pytest.fixture(scope="module")
def reflectance():
    """Six Landsat-like bands (x 10000), 16-day, 2000-2011, (time, band, y=2, x=3): every
    other pixel changes on 2005-06-01 (NIR -1500, SWIR1 +1200, ...)."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("2000-01-01", "2011-12-31", freq="16D")
    season = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    base = {"blue": 400, "green": 700, "red": 500, "nir": 3500, "swir1": 1800, "swir2": 900}
    amp = {"blue": 30, "green": 50, "red": 80, "nir": 400, "swir1": 150, "swir2": 80}
    step = {"blue": 100, "green": 150, "red": 600, "nir": -1500, "swir1": 1200, "swir2": 900}
    data = np.zeros((len(dates), 6, 2, 3))
    for b, name in enumerate(base):
        for p in range(6):
            y = base[name] + amp[name] * season + rng.normal(0, 25, len(dates))
            if p % 2 == 0:
                y = y + step[name] * (dates >= pd.Timestamp("2005-06-01"))
            data[:, b, p // 3, p % 3] = y
    cube = xr.DataArray(data, dims=("time", "band", "y", "x"),
                        coords={"time": dates, "band": list(base), "y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]})
    return cube.rio.write_crs(32721), step


VARIABLES = ["yod", "date", "magnitude", "duration", "pre_val", "post_val", "rate", "dsnr"]


def _changed(ev):
    return ev.yod.values > 0


def test_ccdc_events_on_a_band(reflectance):
    cube, step = reflectance
    seg = zeit.ccdc(cube)
    ev = zeit.extract_events(seg, band="nir", event_type="loss")
    assert list(ev.data_vars) == VARIABLES
    assert ev.rio.crs.to_epsg() == 32721 and ev.attrs["algorithm"] == "CCDC" and ev.attrs["band"] == "nir"
    changed = np.array([[True, False, True], [False, True, False]])
    assert (_changed(ev) == changed).all()
    # the date of the break is the first observation after the change, yod the year of the last before
    first = cube.time.values[cube.time.values >= np.datetime64("2005-06-01")][0]
    assert (ev.date.values[changed] == first).all() and np.isnat(ev.date.values[~changed]).all()
    assert (ev.yod.values[changed] == 2005).all()
    np.testing.assert_allclose(ev.magnitude.values[changed], 1500, atol=30)
    np.testing.assert_allclose(ev.pre_val.values[changed] - ev.post_val.values[changed], ev.magnitude.values[changed],
                               rtol=1e-5)
    assert (ev.duration.values[changed] == 1).all() and (ev.dsnr.values[changed] > 20).all()
    assert ev.yod.rio.nodata == 0
    # NIR fell: as a gain there is nothing
    assert not _changed(zeit.extract_events(seg, band="nir", event_type="gain")).any()
    # lazy is the same
    lazy = zeit.extract_events(zeit.ccdc(cube.chunk({"x": 1})), band="nir", event_type="loss")
    xr.testing.assert_identical(lazy.compute(), ev)


def test_ccdc_events_without_a_band_measure_the_change_vector(reflectance):
    cube, step = reflectance
    seg = zeit.ccdc(cube)
    ev = zeit.extract_events(seg)
    vector = np.sqrt(sum(step[b] ** 2 for b in ("green", "red", "nir", "swir1", "swir2")))
    changed = _changed(ev)
    np.testing.assert_allclose(ev.magnitude.values[changed], vector, rtol=0.03)
    assert np.isnan(ev.pre_val.values).all() and ev.attrs["event_type"] == "any"
    with pytest.raises(ValueError, match="band="):
        zeit.extract_events(seg, event_type="loss")
    with pytest.raises(ValueError, match="not in the CCDC"):
        zeit.extract_events(seg, band="thermal")
    # pre_val_threshold: a loss needs the value before at or above it
    assert not _changed(zeit.extract_events(seg, band="nir", event_type="loss", pre_val_threshold=5000)).any()
    assert _changed(zeit.extract_events(seg, band="nir", event_type="loss", pre_val_threshold=3000)).sum() == 3


def test_bfast_monitor_events(ndvi):
    result = zeit.bfast_monitor(ndvi, "2014-01-01")
    ev = zeit.extract_events(result, event_type="loss")
    assert list(ev.data_vars) == VARIABLES and ev.attrs["algorithm"] == "bfastmonitor"
    assert _changed(ev)[0].all() and not _changed(ev)[1:].any()
    flagged = pd.DatetimeIndex(ev.date.values[0])
    assert (flagged >= DATES[BREAK - 6]).all() and (flagged <= DATES[BREAK + 3]).all()   # MOSUM noise: a bit early
    np.testing.assert_allclose(ev.magnitude.values[0], 0.3, atol=0.05)
    assert np.isnan(ev.pre_val.values).all() and (ev.dsnr.values[0] > 5).all()
    assert not _changed(zeit.extract_events(result, event_type="gain")).any()


@pytest.mark.parametrize("algorithm", ["bfast_lite", "bfast"])
def test_bfast_lite_and_bfast_events(ndvi, algorithm):
    result = getattr(zeit, algorithm)(ndvi)
    ev = zeit.extract_events(result, event_type="loss")
    assert list(ev.data_vars) == VARIABLES and ev.attrs["algorithm"] == algorithm
    assert _changed(ev)[0].all()
    # the break's index is the last observation before the drop (2014-12-29), the date the
    # first after it (2015-01-14)
    assert (ev.yod.values[0] == 2014).all()
    assert (pd.DatetimeIndex(ev.date.values[0]) == DATES[BREAK]).all()
    np.testing.assert_allclose(ev.magnitude.values[0], 0.3, atol=0.05)
    assert (ev.duration.values[0] == 1).all()


def test_bfast_lite_with_gaps_and_the_newest_break():
    """Indices count the missing observations too, so dates stay right with clouds."""
    rng = np.random.default_rng(3)
    t = np.arange(len(DATES))
    y = 0.5 + 0.2 * np.sin(2 * np.pi * t / 23) + rng.normal(0, 0.01, len(t))
    y[60:] -= 0.25
    y[130:] += 0.2
    y[::7] = np.nan
    series = pd.Series(y, index=DATES)
    result = zeit.bfast_lite(series)
    assert int(result.n_breaks) == 2
    greatest = zeit.extract_events(result)
    newest = zeit.extract_events(result, sort_by="newest")
    assert pd.Timestamp(greatest.date.values) == DATES[60] and float(greatest.magnitude) == pytest.approx(0.25, abs=0.03)
    assert pd.Timestamp(newest.date.values) == DATES[130] and float(newest.magnitude) == pytest.approx(0.2, abs=0.03)
    assert zeit.extract_events(result, event_type="gain").date.values == np.datetime64(DATES[130])
    # the viewer draws the breaks on those dates
    from zeit._plot._data import Frames
    from zeit._plot._fit import overlays, pixel_series

    cube = xr.DataArray(y.reshape(-1, 1, 1), dims=("time", "y", "x"), coords={"time": DATES, "y": [0.5], "x": [0.5]})
    s = pixel_series(Frames(cube), 0, 0)
    marks = [o["x"] for o in overlays(zeit.bfast_lite(cube), s, shape=(1, 1)) if o["kind"] == "vline"]
    assert marks == [s["x"][59], s["x"][129]]


def test_errors(ndvi):
    with pytest.raises(ValueError, match="dsnr"):
        zeit.extract_events(zeit.bfast(ndvi), sort_by="dsnr")
    old = zeit.bfast_lite(ndvi).drop_vars([f"magnitude_{k}" for k in range(1, 6)])
    with pytest.raises(ValueError, match="older than 0.48"):
        zeit.extract_events(old)
    with pytest.raises(ValueError, match="extract_events takes"):
        zeit.extract_events(zeit.mann_kendall(ndvi))
    with pytest.raises(ValueError, match="event_type"):
        zeit.extract_events(zeit.bfast_lite(ndvi), event_type="drop")
    with pytest.raises(ValueError, match="band="):
        zeit.extract_events(zeit.bfast_lite(ndvi), band="nir")


def test_every_algorithm_gives_the_same_maps(ndvi):
    """LandTrendr on annual composites and BFAST on the 16-day series: the same variables,
    the same year."""
    annual = ndvi.resample(time="YS").median()
    lt = zeit.extract_events(zeit.landtrendr(annual))
    lite = zeit.extract_events(zeit.bfast_lite(ndvi), event_type="loss")
    assert list(lt.data_vars) == list(lite.data_vars)
    for name in lt.data_vars:
        assert lt[name].dtype == lite[name].dtype, name
    assert (lite.yod.values[0] == 2014).all() and (lt.yod.values[0] > 0).all()


def _yod(values):
    return xr.DataArray(np.array(values, dtype=np.uint16), dims=("y", "x"),
                        coords={"y": [1.5, 0.5], "x": [0.5, 1.5, 2.5]}).rio.write_crs(4326)


def test_agreement():
    a = _yod([[2005, 2005, 0], [2010, 2001, 0]])
    b = _yod([[2005, 2006, 0], [2012, 0, 0]])
    c = _yod([[2004, 2009, 0], [2011, 0, 1999]])
    agree = zeit.agreement({"lt": a, "ccdc": b, "bfast": c}, tolerance=1)
    assert agree.attrs["maps"] == ["lt", "ccdc", "bfast"] and agree.rio.crs.to_epsg() == 4326
    # [0, 0]: 2004, 2005 and 2005 all agree within a year; 2005 is the year most maps give
    assert agree.yod.values.tolist() == [[2005, 2005, 0], [2011, 2001, 1999]]
    assert agree.n_detected.values.tolist() == [[3, 3, 0], [3, 1, 1]]
    assert agree.n_agree.values.tolist() == [[3, 2, 0], [3, 1, 1]]
    np.testing.assert_array_equal(agree.spread.values, [[1, 4, np.nan], [2, 0, 0]])
    assert agree.agrees.sel(map="bfast").values.tolist() == [[1, 0, 0], [1, 0, 1]]
    exact = zeit.agreement(a, b, c, tolerance=0)
    assert exact.yod.values.tolist()[0] == [2005, 2005, 0] and exact.n_agree.values.tolist()[1] == [1, 1, 1]
    # the earliest year wins a tie
    assert exact.yod.values[1, 0] == 2010
    with pytest.raises(ValueError, match="grid"):
        zeit.agreement(a, a.assign_coords(x=a.x + 30))
    with pytest.raises(ValueError, match="two or more"):
        zeit.agreement(a)


def test_agreement_of_extracted_events(ndvi):
    lite = zeit.extract_events(zeit.bfast_lite(ndvi), event_type="loss")
    classic = zeit.extract_events(zeit.bfast(ndvi), event_type="loss")
    agree = zeit.agreement(lite, classic)
    assert agree.attrs["maps"] == ["bfast_lite", "bfast"]
    assert (agree.n_agree.values[0] == 2).all() and (agree.yod.values[0] == 2014).all()
    lazy = zeit.agreement(zeit.extract_events(zeit.bfast_lite(ndvi.chunk({"x": 2})), event_type="loss"), classic)
    xr.testing.assert_identical(lazy.compute(), agree)


def test_events_as_features_of_a_classifier(ndvi):
    """An ensemble in the LCMS way: the events of several algorithms as the features of a
    classifier trained on reference points."""
    import geopandas as gpd

    lite = zeit.extract_events(zeit.bfast_lite(ndvi), event_type="loss")
    monitor = zeit.extract_events(zeit.bfast_monitor(ndvi, "2014-01-01"), event_type="loss")
    features = xr.merge([ev[["magnitude", "dsnr"]].fillna(0).rename({v: f"{name}_{v}" for v in ("magnitude", "dsnr")})
                         for name, ev in (("lite", lite), ("monitor", monitor))], compat="override")
    xs, ys = np.meshgrid(ndvi.x.values, ndvi.y.values)
    points = gpd.GeoDataFrame({"label": np.where(ys.ravel() == 3.5, "loss", "stable")},
                              geometry=gpd.points_from_xy(xs.ravel(), ys.ravel()), crs=4326)
    model = zeit.train_classifier(features, points, label="label")
    out = zeit.classify(features, model)
    assert out.label.values.tolist()[0] == [1, 1, 1, 1] or out.attrs.get("flag_meanings")


def test_the_viewer_draws_a_break_event_on_its_date(ndvi):
    from zeit._plot._data import Frames
    from zeit._plot._fit import overlays, pixel_series

    ev = zeit.extract_events(zeit.bfast_lite(ndvi), event_type="loss")
    s = pixel_series(Frames(ndvi), 0, 0)
    marks = [o for o in overlays(ev, s, shape=(3, 4)) if o["kind"] == "vline"]
    assert len(marks) == 1 and marks[0]["x"] == s["x"][BREAK]
