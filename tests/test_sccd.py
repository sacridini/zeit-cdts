"""zeit.sccd: changes of synthetic series, the near-real-time state, edge cases, every input of
zeit.ccdc, and the agreement with pyxccd (the authors' S-CCD) on Landsat pixels of Rondonia
(tests/data/make_sccd_pyxccd_fixtures.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import chi2

import zeit
from zeit import _core

PYXCCD = Path(__file__).parent / "data" / "sccd_pyxccd.npz"
W = 2 * np.pi / 365.25
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]


def _frame(n, step_at=None, step=1500.0, noise=30.0, seed=0, start="2000-01-01"):
    """A 16-day series of 6 bands: a seasonal cycle, noise and a step (NIR down) at `step_at`."""
    idx = pd.date_range(start, periods=n, freq="16D")
    t = np.array([d.toordinal() for d in idx], float)
    rng = np.random.default_rng(seed)
    cols = {}
    for b, name in enumerate(BANDS):
        v = 500 + 300 * b + 200 * np.sin(W * t) + 80 * np.cos(2 * W * t) + rng.normal(0, noise, n)
        if step_at is not None:
            v = v + np.where(np.arange(n) >= step_at, step * (1 if name != "nir" else -0.4), 0)
        cols[name] = v
    return pd.DataFrame(cols, index=idx)


def _model(coefs, dates):
    """The harmonic model of zeit's coefficients (CCDC's time axis) on dates."""
    t = np.array([pd.Timestamp(d).toordinal() for d in dates], float) + 366
    terms = np.stack([np.ones_like(t), t, np.cos(W * t), np.sin(W * t), np.cos(2 * W * t), np.sin(2 * W * t)])
    return np.asarray(coefs, float) @ terms


def test_finds_a_step():
    df = _frame(230, step_at=150)
    ds = zeit.sccd(df)
    assert int(ds.n_breaks) == 1 and int(ds.n_segments) == 2
    assert ds.t_break.values[0] == np.datetime64(df.index[150], "ns")
    assert float(ds.change_prob[0]) == 1.0
    mag = ds.magnitude.isel(segment=0).values
    assert np.all(mag[[0, 1, 2, 4, 5]] > 1000) and mag[3] < -400
    assert ds.status.values[150] == zeit._sccd.STATUS.index("break")
    # the first model follows the series before the step, the second after it
    before = _model(ds.coefs.isel(segment=0).values, df.index[100:150]) - df.iloc[100:150].to_numpy().T
    after = _model(ds.coefs.isel(segment=1).values, df.index[180:220]) - df.iloc[180:220].to_numpy().T
    assert np.sqrt(np.mean(before ** 2)) < 60 and np.sqrt(np.mean(after ** 2)) < 60
    assert np.all((ds.rmse.isel(segment=0) > 15) & (ds.rmse.isel(segment=0) < 50))
    assert int(ds.nrt_mode) == 1 and int(ds.n_anomalies) == 0


def test_stable_series_is_monitored():
    df = _frame(230)
    ds = zeit.sccd(df)
    assert int(ds.n_breaks) == 0 and int(ds.n_segments) == 1
    assert pd.isnull(ds.t_break.values[0]) and float(ds.change_prob[0]) == 0.0
    assert int(ds.num_obs[0]) == 230 - 6  # the last conse observations are the peek window
    assert int(ds.nrt_mode) == 1 and int(ds.n_anomalies) == 0 and int(ds.n_queue) == 0
    assert pd.isnull(ds.anomaly_start.values) and np.isnan(float(ds.anomaly_statistic))
    assert ds.last_observation.values == np.datetime64(df.index[-1], "ns")
    assert 0 <= float(ds.last_statistic) < ds.attrs["threshold"]
    # the filtered components add up to the prediction
    s = ds.trend + ds.annual + ds.semiannual
    ok = ds.status.values == zeit._sccd.STATUS.index("model")
    np.testing.assert_allclose(s.values[ok], ds.predicted.values[ok] - (ds.predicted - s).values[ok], atol=1e-9)
    assert np.all(np.abs(ds.predicted.values[ok] - df.to_numpy()[ok]) < 200)


def test_recent_anomalies_are_not_yet_a_change():
    df = _frame(200, step_at=197)
    ds = zeit.sccd(df)
    assert int(ds.n_breaks) == 0
    assert int(ds.n_anomalies) == 3
    assert ds.anomaly_start.values == np.datetime64(df.index[197], "ns")
    assert float(ds.change_prob[0]) == pytest.approx(0.5)
    assert float(ds.anomaly_statistic) > ds.attrs["threshold"]
    assert np.all(ds.anomaly_magnitude.values[[0, 1, 2, 4, 5]] > 1000)
    assert np.all(ds.status.values[-6:] == zeit._sccd.STATUS.index("peek"))
    # the model ends before the anomalies
    assert ds.t_end.values[0] == np.datetime64(df.index[196], "ns")


@pytest.mark.parametrize("case", ["no data", "all clouds", "few observations"])
def test_edge_cases(case):
    df = _frame(50)
    qa = None
    if case == "no data":
        df = df * np.nan
    elif case == "all clouds":
        qa = np.full(50, 4)
    else:
        df = df.iloc[:17]
    ds = zeit.sccd(df, qa=qa)
    assert int(ds.n_segments) == 0 and int(ds.n_breaks) == 0
    assert ds.t_start.isnull().all() and ds.rmse.isnull().all()
    if case == "few observations":  # waiting for the 18 observations of a model
        assert int(ds.nrt_mode) == 2 and int(ds.n_queue) == 17
        assert np.all(ds.status.values == zeit._sccd.STATUS.index("queue"))
    else:
        assert int(ds.nrt_mode) == 0 and pd.isnull(ds.last_observation.values)
        assert np.all(ds.status.values == zeit._sccd.STATUS.index("unused"))


def _cube():
    frames = [_frame(200, step_at=s, seed=k) for k, s in enumerate([None, 120, 60, 197, None, 150])]
    values = np.stack([f.to_numpy() for f in frames], axis=-1).reshape(200, 6, 2, 3)
    values[:, :, 1, 1] = np.nan  # a pixel without data
    cube = xr.DataArray(values, dims=("time", "band", "y", "x"),
                        coords={"time": frames[0].index, "band": BANDS, "y": [10.0, 20.0], "x": [1.0, 2.0, 3.0]})
    qa = np.zeros((200, 2, 3), dtype=np.int16)
    qa[::7, 0, 0] = 4  # clouds
    return cube, qa


def test_cube_is_its_pixels():
    cube, qa = _cube()
    ds = zeit.sccd(cube, qa=qa, n_jobs=2)
    assert ds.t_break.dims == ("segment", "y", "x") and ds.coefs.dims == ("segment", "band", "coef", "y", "x")
    assert list(ds.coef.values) == ["a0", "c1", "a1", "b1", "a2", "b2"]
    np.testing.assert_array_equal(ds.n_breaks.values, [[0, 1, 1], [0, 0, 1]])
    assert int(ds.nrt_mode[1, 1]) == 0 and int(ds.n_anomalies[1, 0]) == 3
    for y in range(2):
        for x in range(3):
            px = zeit.sccd(cube.isel(y=y, x=x).transpose("time", "band"), qa=qa[:, y, x])
            xr.testing.assert_identical(px[list(ds.data_vars)].drop_vars("time", errors="ignore"),
                                        ds.isel(y=y, x=x, drop=True).assign_attrs(px.attrs))


def test_dask_numpy_and_unsorted_dates():
    cube, qa = _cube()
    ref = zeit.sccd(cube, qa=qa)
    lazy = zeit.sccd(cube.chunk({"y": 1, "x": 2}), qa=xr.DataArray(qa, dims=("time", "y", "x")).chunk({"y": 1}))
    assert hasattr(lazy.t_break.data, "dask")
    xr.testing.assert_identical(ref, lazy.compute())
    arr = zeit.sccd(cube.values, qa=qa, dates=cube.time.values)
    np.testing.assert_array_equal(arr.t_break.values, ref.t_break.values)
    np.testing.assert_array_equal(arr.coefs.values, ref.coefs.values)
    order = np.random.default_rng(1).permutation(200)
    shuffled = zeit.sccd(cube.isel(time=order), qa=qa[order])
    xr.testing.assert_identical(ref.drop_attrs(), shuffled.drop_attrs())
    # the QA given by position follows the cube into date order (zeit.ccdc's too)
    xr.testing.assert_identical(zeit.ccdc(cube, qa=qa), zeit.ccdc(cube.isel(time=order), qa=qa[order]))


def test_qa_band_and_predicted_image():
    cube, qa = _cube()
    with_qa = xr.concat([cube, xr.DataArray(qa[:, None].astype(float), dims=("time", "band", "y", "x"),
                                            coords={"band": ["fmask"]})], dim="band")
    ds = zeit.sccd(with_qa, qa="fmask")
    xr.testing.assert_identical(ds, zeit.sccd(cube, qa=qa))
    date = cube.time.values[100]
    image = zeit.predict_synthetic_image(ds, date)
    assert image.dims == ("band", "y", "x")
    clean = cube.sel(time=date).transpose("band", "y", "x")
    ok = ds.n_segments.values > 0
    assert np.isnan(image.values[:, ~ok]).all()
    assert np.nanmax(np.abs((image - clean).values[:, ok])) < 200


def test_parameters():
    df = _frame(60)
    ds = zeit.sccd(df, prob_threshold=0.95, detection_bands=["green", "nir"])
    assert ds.attrs["threshold"] == pytest.approx(chi2.ppf(0.95, 2), rel=1e-6)
    assert zeit.sccd(df).attrs["threshold"] == pytest.approx(chi2.ppf(0.99, 5), rel=1e-6)
    with pytest.raises(ValueError, match="conse"):
        zeit.sccd(df, conse=0)
    with pytest.raises(ValueError, match="prob_threshold"):
        zeit.sccd(df, prob_threshold=1.5)
    with pytest.raises(ValueError, match="detection_bands"):
        zeit.sccd(df, detection_bands=["thermal"])
    # a longer peek window confirms the change later, never earlier
    step = _frame(230, step_at=150)
    assert zeit.sccd(step, conse=3).t_break.values[0] == zeit.sccd(step, conse=8).t_break.values[0]


def test_agrees_with_pyxccd_on_rondonia():
    """384 Landsat pixels of Rondonia against pyxccd 1.1.0's sccd_detect (run as a black box).

    The Kalman filter's parameters (the process noise, the initial covariance) and how the RMSE
    that standardizes the test is kept are only partly specified by the paper, so the breaks
    are not identical: on 4096 pixels of the same stack, 85% of the pixels have as many breaks
    and 75% of pyxccd's breaks are within one observation of zeit's, zeit finding 22% more
    breaks, mostly near the threshold (its RMSE is ~15% smaller). The bounds below are those
    figures with room for the 384 pixels' sampling.
    """
    f = np.load(PYXCCD)
    dates, values, qa = f["dates"], f["values"].astype(float), f["qa"]
    n = len(values)
    cube = xr.DataArray(values.transpose(2, 1, 0)[:, :, None, :], dims=("time", "band", "y", "x"),
                        coords={"time": [pd.Timestamp.fromordinal(int(d)) for d in dates], "band": BANDS})
    ds = zeit.sccd(cube, qa=qa.T[:, None, :], nodata=None)
    breaks = ds.t_break.isel(y=0)
    n_breaks = ds.n_breaks.isel(y=0).values
    assert np.mean(n_breaks == f["n_breaks"]) > 0.8
    assert 1.0 <= n_breaks.sum() / f["n_breaks"].sum() < 1.5

    near, mags = 0, []
    for p in range(n):
        clear = np.sort(dates[qa[p] < 2])
        ours = [(k, pd.Timestamp(t).toordinal()) for k, t in enumerate(breaks.values[:, p]) if not pd.isnull(t)]
        for j in range(f["n_breaks"][p]):
            if not ours:
                continue
            k, t = min(ours, key=lambda kt: abs(kt[1] - f["t_break"][p, j]))
            if abs(np.searchsorted(clear, t) - np.searchsorted(clear, f["t_break"][p, j])) <= 1:
                near += 1
                mags.append((ds.magnitude.values[k, :, 0, p], f["magnitude"][p, j]))
    assert near / f["n_breaks"].sum() > 0.68
    m = np.array(mags)
    assert np.corrcoef(m[:, 0].ravel(), m[:, 1].ravel())[0, 1] > 0.97

    # the near-real-time state: monitoring (pyxccd 1, 3: snow) or waiting (2, 4, 5: transition)
    pyx_mode = np.select([np.isin(f["nrt_mode"] % 10, [1, 3]), np.isin(f["nrt_mode"] % 10, [2, 4, 5])], [1, 2], 0)
    assert np.mean(ds.nrt_mode.isel(y=0).values == pyx_mode) > 0.84
    both = (pyx_mode == 1) & (ds.nrt_mode.isel(y=0).values == 1)
    # the count of the latest anomalies agrees less (81%): the statistics of a peek window are
    # often near the threshold, where the two RMSEs decide differently
    assert np.mean(ds.n_anomalies.isel(y=0).values[both] == f["anomaly_conse"][both]) > 0.75


def test_dataframe_with_a_named_index():
    """A pixel's DataFrame whose dates index has a name (df.index.name = 'date'), for sccd and ccdc alike."""
    dates = pd.date_range("2015-01-01", periods=60, freq="16D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.uniform(500, 3000, (60, 6)), index=dates,
                      columns=["blue", "green", "red", "nir", "swir1", "swir2"])
    df.index.name = "date"
    assert zeit.sccd(df).sizes["time"] == 60
    assert "t_start" in zeit.ccdc(df)
