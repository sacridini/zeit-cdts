"""zeit.plot's pixel inspector: a pixel's series and what each algorithm made of it."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._plot._data import prepare
from zeit._plot._fit import overlays, pixel_series
from zeit._plot._session import Session

YEARS = pd.date_range("2000-01-01", periods=20, freq="YS")


@pytest.fixture(scope="module")
def cube():
    """(time, y, x) NDVI x 10000: every pixel drops by 3000 in 2008."""
    rng = np.random.default_rng(0)
    base = np.full((20, 6, 8), 8000.0) + rng.normal(0, 80, (20, 6, 8))
    base[8:] -= 3000
    return xr.DataArray(base.astype(np.float32), dims=("time", "y", "x"), name="ndvi",
                        coords={"time": YEARS, "y": np.arange(6)[::-1] * 30.0 + 15, "x": np.arange(8) * 30.0 + 15}
                        ).rio.write_crs(32721)


def _series(cube, col=3, row=2):
    frames = prepare(cube)[0]
    return frames, pixel_series(frames, col, row)


def test_pixel_series(cube):
    frames, s = _series(cube)
    assert s["is_time"] and len(s["x"]) == 20
    assert s["x"][0] == pd.Timestamp("2000-01-01").value / 1e6
    np.testing.assert_allclose(s["series"][0]["y"], cube.values[:, 2, 3], rtol=1e-6)
    assert s["world"] == [105.0, 105.0] and s["cell"] == [3, 2]


def test_landtrendr_and_events_overlays(cube):
    frames, s = _series(cube)
    lt = zeit.landtrendr(cube)
    (line,) = [o for o in overlays(lt, s, shape=(6, 8)) if o["kind"] == "line"]
    n = int(lt.n_vertices.values[2, 3])
    assert len(line["x"]) == n and line["markers"]
    assert line["y"][0] == pytest.approx(float(lt.vertex_value.values[0, 2, 3]))
    events = zeit.extract_events(lt)
    (span,) = overlays(events, s, shape=(6, 8))
    assert span["kind"] == "span" and span["label"].startswith("event 2007")


def test_matched_by_coordinates(cube):
    frames, s = _series(cube)
    part = zeit.landtrendr(cube.isel(y=slice(1, 4), x=slice(2, 6)))   # a window of the grid
    assert overlays(part, s, shape=(6, 8))                               # found by coordinates
    _, outside = _series(cube, col=7, row=5)
    assert overlays(part, outside, shape=(6, 8)) == []                    # off the result's grid


def test_bfast_and_mann_kendall_overlays():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2010-01-01", periods=23 * 8, freq="16D")
    t = np.arange(len(dates))
    values = (0.5 + 0.2 * np.sin(2 * np.pi * t / 23))[:, None, None] + rng.normal(0, 0.02, (len(t), 4, 3))
    values[23 * 5:] -= 0.3
    cube = xr.DataArray(values.astype(np.float32), dims=("time", "y", "x"), name="ndvi",
                        coords={"time": dates, "y": [2.5, 1.5, 0.5, -0.5], "x": [0.5, 1.5, 2.5]})
    frames, s = _series(cube, col=1, row=1)
    marks = overlays(zeit.bfast_monitor(cube, "2014-01-01"), s, shape=(4, 3))
    labels = [o["label"] for o in marks]
    assert "bfastmonitor break" in labels and "monitoring starts" in labels
    lite = overlays(zeit.bfast_lite(cube), s, shape=(4, 3))
    assert lite and all(o["kind"] == "vline" for o in lite)
    assert pd.Timestamp(lite[0]["x"], unit="ms").year in (2014, 2015)
    mk = overlays(zeit.mann_kendall(cube), s, shape=(4, 3))
    assert mk[0]["kind"] == "line" and "decreasing" in mk[0]["label"] and len(mk[0]["y"]) == len(dates)


def test_ccdc_overlays():
    import os
    inputs = np.load(os.path.join(os.path.dirname(__file__), "data", "ccdc_matlab_parity_inputs.npz"))
    dates = inputs["two_breaks__dates"]
    bands = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]
    data = inputs["two_breaks__bands"].astype(float).T[:, :, None, None]
    times = pd.DatetimeIndex([pd.Timestamp.fromordinal(int(d)) for d in dates])
    cube = xr.DataArray(data, dims=("time", "band", "y", "x"), coords={"time": times, "band": bands,
                                                                       "y": [0.5], "x": [0.5]})
    qa = inputs["two_breaks__qa"].astype(int)[:, None, None]
    seg = zeit.ccdc(cube, qa=qa, detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4], thermal_band=6)
    frames, s = _series(cube.sel(band="nir").rename("nir"), col=0, row=0)
    out = overlays(seg, s, shape=(1, 1))
    lines = [o for o in out if o["kind"] == "line"]
    breaks = [o for o in out if o["kind"] == "vline"]
    assert len(lines) == int(seg.n_segments.values[0, 0]) and lines[0]["label"] == "CCDC model (nir)"
    assert len(breaks) == int(seg.t_break.notnull().sum())
    # the model goes through the observations: residuals of the stable first segment are small
    obs = np.array(s["series"][0]["y"], dtype=float)
    fit = np.interp(s["x"], lines[0]["x"], lines[0]["y"], left=np.nan, right=np.nan)
    inside = np.isfinite(fit) & (inputs["two_breaks__qa"] < 2)
    assert np.nanmedian(np.abs(obs[inside] - fit[inside])) < 400


def test_session_pixel_request_and_static_pixel_plot(cube):
    lt = zeit.landtrendr(cube)
    session = Session(cube, fit=lt)
    meta, _ = session.handle({"type": "meta"})
    assert meta["has_fit"]
    content, buffers = session.handle({"type": "pixel", "x": 3, "y": 2})
    assert buffers == [] and content["cell_top"] == [3, 2]
    assert any(o["kind"] == "line" for o in content["overlays"])
    up = Session(cube.isel(y=slice(None, None, -1)), fit=lt)            # y increasing: top row is the last
    flipped, _ = up.handle({"type": "pixel", "x": 3, "y": 2})
    assert flipped["series"][0]["y"] == content["series"][0]["y"]
    fig = zeit.plot(cube, fit=lt, pixel=(105.0, 105.0), static=True)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "LandTrendr fit" in labels
