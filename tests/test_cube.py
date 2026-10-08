from types import SimpleNamespace

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.errors import RasterioIOError
from stackstac.nodata_reader import exception_matches

import zeit.cube as cube_mod
from zeit.cube import (
    ERRORS_AS_NODATA, _clear_mask, _composite_block, _cube_scale_offset, _item_scale_offset,
    _nanmedian_time, _qa_band_for, _rescale, _stac_band_map, _stm_block,
    build_annual_composites, build_spectral_temporal_metrics, build_time_series,
)

CLEAR = 1 << 6  # Landsat QA_PIXEL clear bit


def test_nanmedian_time_matches_numpy():
    rng = np.random.default_rng(0)
    for t in (1, 2, 7, 8):
        a = rng.random((t, 130, 9)).astype("float32")
        a[a < 0.4] = np.nan
        a[:, 0, 0] = np.nan  # all-NaN pixel
        with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
            ref = np.nanmedian(a, axis=0)
        got = _nanmedian_time(a)
        np.testing.assert_allclose(got, ref, rtol=1e-6)
        assert np.isnan(got[0, 0])


def test_clear_mask():
    qa = np.array([CLEAR, 0, CLEAR | 8, 21824], dtype="uint16")  # 21824: L8 clear land
    assert _clear_mask(qa, "qa_pixel").tolist() == [True, False, True, True]
    scl = np.array([4, 8, 9, 3, 6, 11], dtype="uint16")
    assert _clear_mask(scl, "scl").tolist() == [True, False, False, False, True, True]


def _raw_block(rng, t=6, ny=5, nx=4):
    """(time, band, y, x) uint16 with bands [b1, b2, qa]."""
    raw = rng.integers(1, 30000, size=(t, 3, ny, nx)).astype("uint16")
    raw[:, 2] = CLEAR
    raw[1, 2, 0, :] = 0           # cloudy row in scene 1
    raw[2, 0, :, 0] = 0           # nodata column in band 0 of scene 2
    raw[:, 2, 4, 3] = 0           # pixel cloudy in every scene
    return raw


def _reference(raw, scale, offset):
    clear = (raw[:, 2] & CLEAR) > 0
    vals = []
    for k in range(2):
        v = raw[:, k].astype("float64")
        v[~clear | (raw[:, k] == 0)] = np.nan
        vals.append(v * scale[:, k, None, None] + offset[:, k, None, None])
    return np.stack(vals, axis=1)  # (t, 2, y, x)


def test_composite_block_median_masks_and_rescales_per_scene():
    rng = np.random.default_rng(1)
    raw = _raw_block(rng)
    scale = np.full((6, 2), 1e-4, dtype="float32")
    offset = np.zeros((6, 2), dtype="float32")
    offset[3:] = -0.1  # e.g. Sentinel-2 processing baseline 04.00 from scene 3 on
    got = _composite_block(raw, [0, 1], 2, "qa_pixel", scale, offset, "median")
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ref = np.nanmedian(_reference(raw, scale, offset), axis=0)
    assert got.shape == (2, 5, 4) and got.dtype == np.float32
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert np.isnan(got[:, 4, 3]).all()


def test_composite_block_medoid_returns_a_real_observation():
    rng = np.random.default_rng(2)
    raw = _raw_block(rng)
    scale = np.ones((6, 2), dtype="float32")
    offset = np.zeros((6, 2), dtype="float32")
    got = _composite_block(raw, [0, 1], 2, "qa_pixel", scale, offset, "medoid")
    vals = _reference(raw, scale, offset)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        med = np.nanmedian(vals, axis=0)
    for y in range(5):
        for x in range(4):
            if np.isnan(med[:, y, x]).all():
                assert np.isnan(got[:, y, x]).all()
                continue
            d = np.nansum((vals[:, :, y, x] - med[:, y, x]) ** 2, axis=1)
            d[np.isnan(vals[:, :, y, x]).all(axis=1)] = np.inf
            np.testing.assert_allclose(got[:, y, x], vals[np.argmin(d), :, y, x], rtol=1e-6)


def _stub_cube(raw, scale=1.0, offset=0.0):
    """Raw cube as `build_time_series(dtype="uint16")` returns it (scale/offset coords)."""
    t, nb, ny, nx = raw.shape
    return xr.DataArray(
        da.from_array(raw, chunks=(1, 1, 4, 4)),
        dims=("time", "band", "y", "x"),
        coords={
            "time": pd.date_range("2016-06-01", periods=t, freq="10D"),
            "id": ("time", [f"scene{i}" for i in range(t)]),
            "band": ["b1", "b2", "qa_pixel"][:nb],
            "y": np.arange(ny) * -30.0, "x": np.arange(nx) * 30.0,
            "scale": (("time", "band"), np.broadcast_to(np.float32(scale), (t, nb)).copy()),
            "offset": (("time", "band"), np.broadcast_to(np.float32(offset), (t, nb)).copy()),
            "epsg": 3035,
        },
        attrs={"crs": "epsg:3035", "resolution": 30},
    )


def _item(item_id, **assets):
    """Minimal stand-in for a pystac Item: id + assets with raster:bands."""
    return SimpleNamespace(id=item_id, assets={
        name: SimpleNamespace(extra_fields={"raster:bands": rb} if rb is not None else {})
        for name, rb in assets.items()
    })


def test_item_scale_offset_per_scene_and_band():
    cube = _stub_cube(np.ones((3, 3, 2, 2), dtype="uint16"))
    items = [
        # Listed out of order: matched to the cube by id, not position.
        _item("scene2", b1=[{"scale": 1e-4, "offset": -0.1}], b2=[{"scale": 1e-4, "offset": -0.1}], qa_pixel=[{"unit": "bit"}]),
        _item("scene0", b1=[{"scale": 2.75e-5, "offset": -0.2}], b2=[{"scale": 2.0}], qa_pixel=None),
        _item("scene1", b1=[{"scale": 1e-4}]),  # b2 and qa missing from this item
    ]
    s, o = _item_scale_offset(cube, items)
    np.testing.assert_allclose(s, [[2.75e-5, 2.0, 1.0], [1e-4, 1.0, 1.0], [1e-4, 1e-4, 1.0]], rtol=1e-6)
    np.testing.assert_allclose(o, [[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.1, -0.1, 0.0]], rtol=1e-6)


def test_item_scale_offset_sentinel2_without_raster_bands():
    """Planetary Computer Sentinel-2: factors from the processing baseline."""
    cube = _stub_cube(np.ones((2, 3, 2, 2), dtype="uint16")).assign_coords(band=["B04", "B8A", "SCL"])
    items = [
        SimpleNamespace(id="scene0", properties={"s2:processing_baseline": "03.01"},
                        assets={b: SimpleNamespace(extra_fields={}) for b in ("B04", "B8A", "SCL")}),
        SimpleNamespace(id="scene1", properties={"s2:processing_baseline": "05.11"},
                        assets={b: SimpleNamespace(extra_fields={}) for b in ("B04", "B8A", "SCL")}),
    ]
    s, o = _item_scale_offset(cube, items)
    np.testing.assert_allclose(s, [[1e-4, 1e-4, 1.0], [1e-4, 1e-4, 1.0]], rtol=1e-6)
    np.testing.assert_allclose(o, [[0.0, 0.0, 0.0], [-0.1, -0.1, 0.0]], rtol=1e-6)


def test_qa_band_and_band_map_per_provider():
    assert _qa_band_for("sentinel-2-l2a", "planetary_computer") == "SCL"
    assert _qa_band_for("sentinel-2-l2a", "earth_search") == "scl"
    assert _qa_band_for("landsat-c2-l2", "planetary_computer") == "qa_pixel"
    assert _clear_mask(np.array([4, 9], dtype="uint16"), "SCL").tolist() == [True, False]
    assert _stac_band_map("sentinel-2-l2a", "planetary_computer")["nir"] == "B08"
    assert _stac_band_map("sentinel-2-l2a", "earth_search")["nir"] == "nir"
    assert _stac_band_map("landsat-c2-l2")["swir2"] == "swir22"


def test_cube_scale_offset_reads_coords():
    cube = _stub_cube(np.ones((2, 3, 2, 2), dtype="uint16"), scale=1e-4, offset=-0.1)
    s, o = _cube_scale_offset(cube, ["b2"])
    assert s.shape == (2, 1)
    np.testing.assert_allclose(s, 1e-4)
    np.testing.assert_allclose(o, -0.1)
    s, o = _cube_scale_offset(cube.drop_vars(["scale", "offset"]), ["b1", "b2"])
    np.testing.assert_allclose(s, 1.0)
    np.testing.assert_allclose(o, 0.0)


def test_rescale_keeps_float32_and_handles_duplicate_times():
    raw = np.full((3, 3, 2, 2), 10000, dtype="float32")
    raw[0, 0, 0, 0] = np.nan
    cube = _stub_cube(raw)
    cube = cube.assign_coords(time=[cube.time.values[0]] * 3)  # duplicate timestamps
    scale = np.array([[2.75e-5, 1e-4, 1.0]] * 3, dtype="float32")
    offset = np.array([[-0.2, -0.1 * i, 0.0] for i in range(3)], dtype="float32")
    out = _rescale(cube, scale, offset).compute()
    assert out.dtype == np.float32 and out.attrs == cube.attrs
    np.testing.assert_allclose(out.sel(band="b1").values[1], 10000 * 2.75e-5 - 0.2, rtol=1e-6)
    np.testing.assert_allclose(out.sel(band="b2").values[:, 0, 0], [1.0, 0.9, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(out.sel(band="qa_pixel").values, raw[:, 2])
    assert np.isnan(out.values[0, 0, 0, 0])


def test_errors_as_nodata_patterns():
    assert exception_matches(RasterioIOError("HTTP response code: 404"), ERRORS_AS_NODATA)
    assert exception_matches(RasterioIOError("x.tif: TIFFReadEncodedTile() failed. IReadBlock failed at X offset 1"), ERRORS_AS_NODATA)
    # Access problems must surface instead of silently becoming an all-NaN cube.
    assert not exception_matches(RasterioIOError("InvalidCredentials: No valid AWS credentials found."), ERRORS_AS_NODATA)
    assert not exception_matches(RasterioIOError("AccessDenied: Anonymous users cannot invoke requests against Requester Pays buckets."), ERRORS_AS_NODATA)


def test_integer_dtype_rejects_in_place_cloud_mask():
    with pytest.raises(ValueError, match="float dtype"):
        build_time_series(bbox=[0, 0, 1, 1], apply_cloud_mask=True, dtype="uint16")


def test_build_annual_composites(monkeypatch):
    rng = np.random.default_rng(3)
    raws = {2016: _raw_block(rng, t=6, ny=6, nx=7), 2018: _raw_block(rng, t=3, ny=6, nx=7)}
    calls = []

    def fake_build_time_series(**kw):
        calls.append(kw)
        year = int(kw["start_date"][:4])
        if year not in raws:
            raise ValueError("No images found for the given criteria.")
        assert kw["dtype"] == "uint16" and kw["apply_cloud_mask"] is False
        assert kw["bands"] == ["b1", "b2", "qa_pixel"]
        return _stub_cube(raws[year], scale=1e-4)

    monkeypatch.setattr(cube_mod, "build_time_series", fake_build_time_series)
    out = build_annual_composites(collection="landsat-c2-l2", bbox=[0, 0, 1, 1],
                                  start_year=2016, end_year=2018, bands=["b1", "b2"],
                                  season=("06-01", "09-30"))
    assert out.dims == ("time", "band", "y", "x") and out.shape == (3, 2, 6, 7)
    assert out.year.values.tolist() == [2016, 2017, 2018]
    assert out.time.dt.year.values.tolist() == [2016, 2017, 2018]
    assert calls[0]["start_date"] == "2016-06-01" and calls[0]["end_date"] == "2016-09-30"
    assert int(out.epsg) == 3035

    vals = out.compute().values
    scale = np.full((6, 2), 1e-4, dtype="float32")
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ref = np.nanmedian(_reference(raws[2016], scale, np.zeros_like(scale)), axis=0)
    np.testing.assert_allclose(vals[0], ref, rtol=1e-5)
    assert np.isnan(vals[1]).all()          # 2017 had no scenes
    assert np.isfinite(vals[2]).any()


def test_build_annual_composites_season_wrapping_and_validation(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return _stub_cube(np.full((2, 3, 4, 4), CLEAR, dtype="uint16"))

    monkeypatch.setattr(cube_mod, "build_time_series", fake)
    build_annual_composites(bbox=[0, 0, 1, 1], start_year=2020, end_year=2020,
                            bands=["b1"], season=("12-01", "02-28"))
    assert calls[0]["start_date"] == "2020-12-01" and calls[0]["end_date"] == "2021-02-28"

    with pytest.raises(ValueError, match="bbox or vector_path"):
        build_annual_composites(bands=["b1"])
    with pytest.raises(ValueError, match="method"):
        build_annual_composites(bbox=[0, 0, 1, 1], bands=["b1"], method="mean")


def test_stm_block_indices_per_observation_then_metrics():
    rng = np.random.default_rng(4)
    raw = _raw_block(rng, t=9)                      # bands [b1, b2, qa]
    scale = np.full((9, 2), 1e-4, dtype="float32")
    offset = np.zeros((9, 2), dtype="float32")
    offset[5:] = -0.1
    layers = [("NDVI", {"nir": "b2", "red": "b1"}), ("b2", "b2")]
    band_pos = {"b1": (0, 0), "b2": (1, 1)}
    metrics = ["median", "p10", "count"]
    got = _stm_block(raw, layers, band_pos, 2, "qa_pixel", scale, offset, metrics)
    assert got.shape == (6, 5, 4) and got.dtype == np.float32

    vals = _reference(raw, scale, offset)            # (t, 2, y, x) reflectance, NaN masked
    with np.errstate(all="ignore"):
        ndvi = (vals[:, 1] - vals[:, 0]) / (vals[:, 1] + vals[:, 0])
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        for k, a in enumerate((ndvi, vals[:, 1])):
            np.testing.assert_allclose(got[3 * k], np.nanmedian(a, 0), rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(got[3 * k + 1], np.nanpercentile(a, 10, 0), rtol=1e-4, atol=1e-6)
            np.testing.assert_array_equal(got[3 * k + 2], np.isfinite(a).sum(0))
    assert np.isnan(got[[0, 1, 3, 4], 4, 3]).all()   # cloudy in every scene
    assert (got[[2, 5], 4, 3] == 0).all()


def test_build_spectral_temporal_metrics(monkeypatch):
    rng = np.random.default_rng(5)
    raws = {2019: _raw_block(rng, t=8, ny=6, nx=7)}
    calls = []

    def fake_build_time_series(**kw):
        calls.append(kw)
        year = int(kw["start_date"][:4])
        if year not in raws:
            raise ValueError("No images found for the given criteria.")
        assert kw["dtype"] == "uint16" and kw["apply_cloud_mask"] is False
        return _stub_cube(raws[year], scale=1e-4)

    monkeypatch.setattr(cube_mod, "build_time_series", fake_build_time_series)
    out = build_spectral_temporal_metrics(
        collection="landsat-c2-l2", bbox=[0, 0, 1, 1], start_year=2019, end_year=2020,
        indices=["ndvi", "b2"], metrics=["median", "P90"], band_map={"red": "b1", "nir": "b2"},
    )
    assert calls[0]["bands"] == ["b2", "b1", "qa_pixel"]
    assert calls[0]["start_date"] == "2019-01-01" and calls[0]["end_date"] == "2019-12-31"
    assert out.dims == ("time", "band", "y", "x") and out.shape == (2, 4, 6, 7)
    assert out.band.values.tolist() == ["NDVI_median", "NDVI_p90", "b2_median", "b2_p90"]
    assert out.year.values.tolist() == [2019, 2020]

    vals = out.compute().values
    scale = np.full((8, 2), 1e-4, dtype="float32")
    ref = _reference(raws[2019], scale, np.zeros_like(scale))
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ndvi = (ref[:, 1] - ref[:, 0]) / (ref[:, 1] + ref[:, 0])
        np.testing.assert_allclose(vals[0, 0], np.nanmedian(ndvi, 0), rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(vals[0, 3], np.nanpercentile(ref[:, 1], 90, 0), rtol=1e-4)
    assert np.isnan(vals[1]).all()


def test_build_spectral_temporal_metrics_validation():
    with pytest.raises(ValueError, match="indices"):
        build_spectral_temporal_metrics(bbox=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="metric"):
        build_spectral_temporal_metrics(bbox=[0, 0, 1, 1], indices=["NDVI"], metrics=["variance"])
    with pytest.raises(ValueError, match="band_map"):
        build_spectral_temporal_metrics(collection="my-sensor", bbox=[0, 0, 1, 1], indices=["NDVI"])
