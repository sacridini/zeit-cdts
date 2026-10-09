import numpy as np

from zeit.qc import qc_modis_summary, qc_modis_state, qc_sentinel2_scl


def test_qc_modis_summary_levels():
    qa = np.array([0, 1, 2, 3, 99])
    w = qc_modis_summary(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    # good, marginal, snow/ice, cloudy, fill/unknown
    np.testing.assert_allclose(w, [1.0, 0.5, 0.2, 0.2, 0.0])


def test_qc_modis_summary_custom_levels():
    qa = np.array([0, 1, 2])
    w = qc_modis_summary(qa, wmin=0.1, wmid=0.4, wmax=0.9)
    np.testing.assert_allclose(w, [0.9, 0.4, 0.1])


def test_qc_modis_state_good_pixel_is_clear_no_snow_low_aerosol():
    # cloud state bits 0-1 = 00 (clear), aerosol bits 6-7 = 01 (low), no snow bit 12
    qa = np.array([1 << 6], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [1.0])


def test_qc_modis_state_snow_pixel_is_bad():
    # snow bit (bit 12) set
    qa = np.array([1 << 12], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [0.2])


def test_qc_modis_state_cloudy_pixel_is_bad():
    # cloud state bits 0-1 = 01 (cloudy)
    qa = np.array([0b01], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [0.2])


def test_qc_modis_state_high_aerosol_is_bad():
    # aerosol bits 6-7 = 11 (high), clear cloud state
    qa = np.array([0b11 << 6], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [0.2])


def test_qc_sentinel2_scl_classes():
    # 4=vegetation(good), 8=cloud medium prob(mid), 3=cloud shadow(bad), 11=snow(bad)
    scl = np.array([4, 8, 3, 11, 6])
    w = qc_sentinel2_scl(scl, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [1.0, 0.5, 0.2, 0.2, 1.0])


# --- QA cubes -----------------------------------------------------------------------

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

import zeit  # noqa: E402


def _scl_cube(nodata=0):
    rng = np.random.default_rng(0)
    t = from_origin(500000.0, 4500000.0, 20.0, 20.0)
    scl = rng.choice([0, 3, 4, 5, 8, 9], size=(6, 8, 10)).astype(np.uint8)
    da = xr.DataArray(scl, dims=("time", "y", "x"),
                      coords={"time": pd.date_range("2022-01-01", periods=6, freq="16D"),
                              "y": t.f + t.e * (np.arange(8) + 0.5), "x": t.c + t.a * (np.arange(10) + 0.5)},
                      name="scl").rio.write_crs("EPSG:32722")
    return da.rio.write_nodata(nodata, encoded=False) if nodata is not None else da


def test_a_qa_cube_gives_a_weights_cube():
    scl = _scl_cube()
    w = zeit.qc_sentinel2_scl(scl)
    assert isinstance(w, xr.DataArray) and w.dims == scl.dims and w.name == "weights"
    assert (w.time.values == scl.time.values).all() and (w.x.values == scl.x.values).all()
    assert w.rio.crs == "EPSG:32722"
    expected = qc_sentinel2_scl(scl.values)
    expected[scl.values == 0] = 0.0      # the raster's NoData: excluded
    np.testing.assert_allclose(w.values, expected)
    lazy = zeit.qc_sentinel2_scl(scl.chunk({"time": 2}))
    assert lazy.chunks is not None
    np.testing.assert_allclose(lazy.values, w.values)
    without = zeit.qc_sentinel2_scl(_scl_cube(nodata=None))
    np.testing.assert_allclose(without.values, qc_sentinel2_scl(scl.values))  # phenofit's wmin for 0


def test_a_qa_raster_path_and_nan(tmp_path):
    scl = _scl_cube()
    path = tmp_path / "scl.tif"
    zeit.save_raster(scl, path)
    np.testing.assert_allclose(zeit.qc_sentinel2_scl(path).values, zeit.qc_sentinel2_scl(scl).values)
    qa = np.array([0.0, np.nan, 1.0])
    np.testing.assert_allclose(qc_modis_summary(qa), [1.0, 0.0, 0.5])
    np.testing.assert_allclose(qc_modis_state(np.array([np.nan, float(1 << 6)])), [0.0, 1.0])


def test_weights_go_straight_into_smooth():
    scl = _scl_cube()
    rng = np.random.default_rng(1)
    ndvi = xr.DataArray(rng.normal(0.6, 0.1, scl.shape), dims=scl.dims, coords=scl.coords).rio.write_crs("EPSG:32722")
    w = zeit.qc_sentinel2_scl(scl)
    out = zeit.smooth(ndvi, weights=w, lmbda=5)
    ref = zeit.smooth(ndvi, weights=w.values, lmbda=5)
    np.testing.assert_allclose(out.values, ref.values)
