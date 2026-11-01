"""Best available pixel composites: zeit-core's bap (the score, the distance to clouds) through cube.py."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import ndimage

import zeit.cube as cube_mod
from zeit._core import bap
from zeit.cube import (
    BAP_CLEAR, BAP_CLOUD, BAP_NO_DATA, _bap_block, _bap_cube, _bap_opts, _qa_states, _scene_dates,
    _season_middle, _sensor_scores, build_annual_composites,
)

CLEAR = 1 << 6      # Landsat QA_PIXEL's clear bit
CLOUD = 1 << 3      # its cloud bit


def test_distance_to_matches_scipy():
    rng = np.random.default_rng(0)
    for p in (0.0005, 0.02, 0.4):
        m = rng.random((157, 203)) < p
        np.testing.assert_allclose(bap.distance_to(m), ndimage.distance_transform_edt(~m), atol=1e-4)
    capped = bap.distance_to(np.eye(30, dtype=bool), cap=3.0)
    assert capped.max() == 3.0
    assert (bap.distance_to(np.zeros((4, 4), bool), cap=9.0) == 9.0).all()


def test_qa_states():
    qa = np.array([CLEAR, 0, 1, CLOUD, 1 << 4, 1 << 1, 21824, 1 << 5], dtype="uint16")
    assert _qa_states(qa, "qa_pixel").tolist() == [2, 0, 0, 1, 1, 1, 2, 0]
    scl = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype="uint16")
    assert _qa_states(scl, "SCL").tolist() == [0, 0, 0, 1, 2, 2, 2, 2, 1, 1, 1, 2]
    fmask = np.array([0, 2, 4, 8, 16, 32, 255], dtype="uint16")
    assert _qa_states(fmask, "Fmask").tolist() == [2, 1, 1, 1, 2, 2, 0]
    assert (_qa_states(scl, None) == BAP_CLEAR).all()


def test_scene_scores_and_season():
    times = pd.to_datetime(["2002-07-01", "2010-07-01", "2010-07-01"]).values
    assert _sensor_scores(["landsat-7", "landsat-7", "landsat-5"], times).tolist() == [1.0, 0.5, 1.0]
    assert _season_middle(("06-01", "09-30")) == (152 + 273) / 2
    assert _season_middle(("12-01", "01-31")) == pytest.approx((335 + 31 + 365) / 2 - 365)


def _reference(states, doy, sensor, cover, o):
    """The best date of each pixel in numpy and scipy: the score as zeit-core's bap documents it."""
    t = states.shape[0]
    scores = np.full(states.shape, -np.inf)
    for i in range(t):
        clouds = states[i] == BAP_CLOUD
        d = ndimage.distance_transform_edt(~clouds) if clouds.any() else np.full(clouds.shape, o["cloud_max"])
        d = np.minimum(d, o["cloud_max"])
        cs = np.clip((d - o["cloud_min"]) / (o["cloud_max"] - o["cloud_min"]), 0, 1)
        dd = abs(doy[i] - o["target_doy"]) % 365.25
        ds = np.exp(-0.5 * (min(dd, 365.25 - dd) / o["doy_sigma"]) ** 2)
        s = o["w_doy"] * ds + o["w_sensor"] * sensor[i] + o["w_cover"] * cover[i] + o["w_cloud"] * cs
        scores[i] = np.where(states[i] == BAP_CLEAR, s, -np.inf)
    best = np.argmax(scores, axis=0)
    best[np.isinf(scores.max(axis=0))] = -1
    return best, scores.max(axis=0)


def _raw(rng, t=7, ny=23, nx=19):
    """(time, band, y, x) uint16: bands [b1, b2, qa_pixel], clouds in blobs, a fill corner, a band missing."""
    raw = rng.integers(1, 30000, size=(t, 3, ny, nx)).astype("uint16")
    raw[:, 2] = CLEAR
    for i in range(t):
        for _ in range(3):
            y, x, r = rng.integers(0, ny), rng.integers(0, nx), rng.integers(1, 4)
            yy, xx = np.ogrid[:ny, :nx]
            raw[i, 2][(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = CLOUD
    raw[0, :, :5, :5] = 0                   # outside scene 0
    raw[1, 0, -3:, :] = 0                   # band b1 missing in scene 1's last rows
    raw[:, 2, 0, -1] = CLOUD                # a pixel cloudy in every scene
    return raw


def test_bap_block_picks_the_best_scores():
    rng = np.random.default_rng(1)
    raw = _raw(rng)
    t = raw.shape[0]
    doy = np.array([150, 170, 190, 196, 200, 230, 260], dtype=float)
    sensor = np.array([1, 1, 0.5, 1, 1, 1, 1], dtype=float)
    cover = np.linspace(1, 0.4, t)
    o = _bap_opts(196, 20, (1, 6), None)        # pixels here: cloud distances 1 to 6
    scale = np.full((t, 2), 1e-4, dtype="float32")
    offset = np.zeros((t, 2), dtype="float32")
    offset[4:] = -0.1
    got = _bap_block(raw, [0, 1], 2, "qa_pixel", scale, offset, doy, sensor, cover, o, True, (0, 0, 0, 0))
    assert got.shape == (5, 23, 19) and got.dtype == np.float32

    states = _qa_states(raw[:, 2], "qa_pixel")
    states[((raw[:, :2] == 0).any(axis=1)) & (states == BAP_CLEAR)] = BAP_NO_DATA
    best, score = _reference(states, doy, sensor, cover, o)
    has = best >= 0
    assert not has[0, -1]
    np.testing.assert_array_equal(np.where(has, doy[np.maximum(best, 0)], np.nan), got[2])
    np.testing.assert_allclose(got[3][has], score[has], rtol=1e-6)
    np.testing.assert_array_equal(got[4], (states == BAP_CLEAR).sum(0))
    for k in range(2):
        v = np.take_along_axis(raw[:, k], np.maximum(best, 0)[None], 0)[0] * 1e-4 + offset[np.maximum(best, 0), k]
        np.testing.assert_allclose(got[k][has], v[has], rtol=1e-6, atol=1e-7)
        assert np.isnan(got[k][~has]).all()
    # the bands of a pixel come from one date
    assert set(np.unique(best[has])) <= set(range(t))


def _stub_cube(raw, chunks):
    t, nb, ny, nx = raw.shape
    return xr.DataArray(
        da.from_array(raw, chunks=chunks),
        dims=("time", "band", "y", "x"),
        coords={
            "time": pd.date_range("2016-06-01", periods=t, freq="11D"),
            "band": ["b1", "b2", "qa_pixel"],
            "y": np.arange(ny) * -30.0, "x": np.arange(nx) * 30.0,
            "platform": ("time", ["landsat-7", "landsat-8"] * (t // 2) + ["landsat-8"] * (t % 2)),
            "eo:cloud_cover": ("time", np.linspace(0, 60, t)),
            "scale": (("time", "band"), np.full((t, nb), 1e-4, dtype="float32")),
            "offset": (("time", "band"), np.zeros((t, nb), dtype="float32")),
            "epsg": 32720,
        },
        attrs={"crs": "epsg:32720", "resolution": 30},
    )


def test_bap_cube_is_the_same_whatever_the_chunks():
    rng = np.random.default_rng(2)
    raw = _raw(rng, t=8, ny=37, nx=29)
    o = _bap_opts(190, 25, (30, 150), None)     # 1 to 5 pixels of 30 m: a margin of 5
    whole = _bap_cube(_stub_cube(raw, (1, 1, 37, 29)), ["b1", "b2"], "qa_pixel", o, True).compute()
    for chunks in ((1, 1, 6, 7), (1, 1, 10, 29), (1, 1, 37, 5)):
        got = _bap_cube(_stub_cube(raw, chunks), ["b1", "b2"], "qa_pixel", o, True).compute()
        np.testing.assert_array_equal(np.isnan(got), np.isnan(whole))
        np.testing.assert_allclose(got[np.isfinite(got)], whole[np.isfinite(whole)], rtol=0)
    # the scenes' scores: Landsat 7 in 2016 is after its SLC failed, the cover from eo:cloud_cover
    doy, sensor, cover = _scene_dates(_stub_cube(raw, (1, 1, 37, 29)))
    assert sensor.tolist() == [0.5, 1.0] * 4
    np.testing.assert_allclose(cover, 1 - np.linspace(0, 0.6, 8))
    assert doy[0] == pd.Timestamp("2016-06-01").dayofyear


def test_build_annual_composites_bap(monkeypatch):
    rng = np.random.default_rng(3)
    raws = {2016: _raw(rng, t=6, ny=21, nx=17)}

    def fake_build_time_series(**kw):
        year = int(kw["start_date"][:4])
        if year not in raws:
            raise ValueError("No images found for the given criteria.")
        assert kw["bands"] == ["b1", "b2", "qa_pixel"]
        return _stub_cube(raws[year], (1, 1, 8, 8))

    monkeypatch.setattr(cube_mod, "build_time_series", fake_build_time_series)
    out = build_annual_composites(collection="landsat-c2-l2", bbox=[0, 0, 1, 1], start_year=2016, end_year=2017,
                                  bands=["b1", "b2"], season=("06-01", "09-30"), method="bap",
                                  cloud_distance=(30, 120), bap_info=True)
    assert out.band.values.tolist() == ["b1", "b2", "bap_doy", "bap_score", "bap_clear"]
    vals = out.compute().values
    assert np.isnan(vals[1]).all()
    doy = vals[0, 2]
    times = pd.date_range("2016-06-01", periods=6, freq="11D").dayofyear.values
    assert set(np.unique(doy[np.isfinite(doy)])) <= set(times.astype(float))
    # the target day is the season's middle (Aug 1st, day 212.5): the scenes near it win the most
    assert np.nanmedian(doy) >= 170
    # the plain bands without the info
    out = build_annual_composites(collection="landsat-c2-l2", bbox=[0, 0, 1, 1], start_year=2016, end_year=2016,
                                  bands=["b1", "b2"], method="bap", weights={"sensor": 0})
    assert out.band.values.tolist() == ["b1", "b2"]
    with pytest.raises(ValueError, match="weights"):
        build_annual_composites(bbox=[0, 0, 1, 1], bands=["b1"], method="bap", weights={"haze": 1})
    with pytest.raises(ValueError, match="cloud_distance"):
        build_annual_composites(bbox=[0, 0, 1, 1], bands=["b1"], method="bap", cloud_distance=(100, 10))
