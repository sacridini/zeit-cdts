"""zeit.qc: QA bands to observation weights (MOD13 SummaryQA, MOD09 State QA, Sentinel-2 SCL)."""
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit.qc import qc_modis_state, qc_modis_summary, qc_sentinel2_scl

WMIN, WMID, WMAX = 0.2, 0.5, 1.0


def check(func, values, expected, dtype=None, **kw):
    out = func(np.array(values, dtype=dtype), **kw)
    np.testing.assert_array_equal(out, np.array(expected, dtype=float))
    assert out.dtype.kind == "f" and out.shape == np.shape(values)


# ---- MOD13 SummaryQA ----

def test_summary_codes():
    expected = [0, 1.0, 0.5, 0.2, 0.2]
    check(qc_modis_summary, [-1, 0, 1, 2, 3], expected, np.int8)
    check(qc_modis_summary, [255, 0, 1, 2, 3], expected, np.uint8)
    check(qc_modis_summary, [4, 100, -2, 254], [0, 0, 0, 0], np.int16)
    check(qc_modis_summary, [0.0, 1.0, 2.0, 3.0, np.nan, -1.0, 1.5], [1.0, 0.5, 0.2, 0.2, 0, 0, 0])


def test_summary_keeps_the_shape_and_the_weights_given():
    codes = np.array([[0, 1, 2], [3, 255, 0]], dtype=np.uint8)
    out = qc_modis_summary(codes)
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, [[1.0, 0.5, 0.2], [0.2, 0, 1.0]])
    out = qc_modis_summary(np.array([0, 1, 2, 3]), wmin=0.1, wmid=0.4, wmax=0.9)
    np.testing.assert_array_equal(out, [0.9, 0.4, 0.1, 0.1])


# ---- MOD09 State QA ----

CLEAN = 0b0000_0000_0100_0000  # low aerosol, nothing else


@pytest.mark.parametrize("value, weight", [
    (0b0000_0000_0000_0000, WMID),   # aerosol climatology
    (0b0000_0000_0100_0000, WMAX),   # low aerosol
    (0b0000_0000_1000_0000, WMAX),   # average aerosol
    (0b0000_0000_1100_0000, WMIN),   # high aerosol
    (0b0000_0000_0100_0001, WMIN),   # cloudy
    (0b0000_0000_0100_0010, WMIN),   # mixed
    (0b0000_0000_0100_0011, WMAX),   # not set, assumed clear
    (0b0000_0000_0100_0100, WMIN),   # cloud shadow
    (0b0000_0100_0100_0000, WMIN),   # internal cloud
    (0b0000_0001_0100_0000, WMID),   # small cirrus
    (0b0000_0010_0100_0000, WMIN),   # average cirrus
    (0b0000_0011_0100_0000, WMIN),   # high cirrus
    (0b0010_0000_0100_0000, WMID),   # adjacent to cloud
    (0b0001_0000_0100_0000, WMIN),   # MOD35 snow
    (0b1000_0000_0100_0000, WMIN),   # internal snow
    (0b0000_1000_0100_0000, WMIN),   # fire
    (0b0100_0000_0100_0000, WMAX),   # salt pan: ignored
    (0b0000_0000_0100_1000, WMAX),   # land: ignored
    (0b0010_0001_0100_0000, WMID),   # adjacent + small cirrus
    (0b0001_0010_1100_0000, WMIN),   # snow + average cirrus + high aerosol
    (0b0000_0010_0100_0100, WMIN),   # average cirrus + shadow
    (65535, 0.0),
])
def test_state_bit_combinations(value, weight):
    assert qc_modis_state(np.array([value], dtype=np.uint16))[0] == weight


def test_state_other_dtypes():
    assert qc_modis_state(np.array([-1], dtype=np.int16))[0] == 0
    assert qc_modis_state(np.array([-1], dtype=np.int32))[0] == 0
    assert qc_modis_state(np.array([70000], dtype=np.int32))[0] == 0
    assert qc_modis_state(np.array([np.nan]))[0] == 0
    assert qc_modis_state(np.array([64.0]))[0] == WMAX
    assert qc_modis_state(np.array([64.5]))[0] == 0
    assert qc_modis_state(np.array([64], dtype=np.int16))[0] == WMAX
    # a uint16 pattern stored in an int16 reads as unsigned: 0b1000_0000_0100_0000 is internal snow
    assert qc_modis_state(np.array([-32768 + 64], dtype=np.int16))[0] == WMIN
    assert qc_modis_state(np.array([-32769]))[0] == 0


def naive_state_weight(v, wmin=WMIN, wmid=WMID, wmax=WMAX):
    """One value decoded the long way, bit by bit."""
    bit = lambda i: (v >> i) & 1
    if v == 65535:
        return 0.0
    w = wmax
    cloud = bit(0) + 2 * bit(1)
    if cloud in (1, 2) or bit(2) or bit(10):
        w = min(w, wmin)
    cirrus = bit(8) + 2 * bit(9)
    w = min(w, {0: wmax, 1: wmid, 2: wmin, 3: wmin}[cirrus])
    aerosol = bit(6) + 2 * bit(7)
    w = min(w, {0: wmid, 1: wmax, 2: wmax, 3: wmin}[aerosol])
    if bit(13):
        w = min(w, wmid)
    if bit(11) or bit(12) or bit(15):
        w = min(w, wmin)
    return w


def test_state_sweep_of_every_uint16():
    values = np.arange(65536, dtype=np.uint16)
    out = qc_modis_state(values)
    expected = np.array([naive_state_weight(int(v)) for v in values])
    np.testing.assert_array_equal(out, expected)
    assert set(np.unique(out)) <= {0.0, WMIN, WMID, WMAX}
    # zero only for the fill value
    assert (out == 0).sum() == 1 and out[65535] == 0
    # turning on a bad bit never raises the weight
    for bit in (2, 10, 11, 12, 13, 15):
        on = values | (1 << bit)
        assert (qc_modis_state(on.astype(np.uint16)) <= out + 1e-12)[values != 65535].all()


# ---- Sentinel-2 SCL ----

def test_scl_codes():
    check(qc_sentinel2_scl, list(range(12)), [0, 0, 0.2, 0.2, 1.0, 1.0, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2], np.uint8)
    check(qc_sentinel2_scl, [12, 255], [0, 0], np.uint8)
    check(qc_sentinel2_scl, [-1], [0], np.int16)
    check(qc_sentinel2_scl, [4.0, 4.5, np.nan, 9.0], [1.0, 0, 0, 0.2])
    check(qc_sentinel2_scl, [4, 6, 11, 9], [0.9, 0.4, 0.1, 0.1], wmin=0.1, wmid=0.4, wmax=0.9)


def test_bad_arguments():
    with pytest.raises(ValueError):
        qc_modis_summary(np.array([0]), wmin=0.6, wmid=0.5)
    with pytest.raises(ValueError):
        qc_sentinel2_scl(np.array([4]), wmin=-0.1)
    with pytest.raises(ValueError):
        qc_modis_state(np.array([64]), wmax=0.3)
    with pytest.raises(TypeError):
        qc_sentinel2_scl(np.array([True, False]))
    with pytest.raises(TypeError):
        qc_modis_summary(np.array(["a", "b"]))


# ---- DataArray, raster path, NoData ----

def scl_cube() -> xr.DataArray:
    tr = from_origin(500000, 9000000, 30, 30)
    codes = np.array([[[4, 9, 6], [3, 5, 11]], [[8, 4, 4], [10, 7, 0]]], dtype=np.uint8)
    return xr.DataArray(
        codes, dims=("time", "y", "x"), name="scl",
        coords={"time": pd.to_datetime(["2020-01-01", "2020-01-06"]),
                "y": tr.f - 30 * (np.arange(2) + 0.5), "x": tr.c + 30 * (np.arange(3) + 0.5)},
        attrs={"scale_factor": 1.0, "add_offset": 0.0, "units": "class"},
    ).rio.write_crs("EPSG:32722").rio.write_transform(tr)


def test_dataarray_keeps_dims_coords_and_georeferencing():
    da = scl_cube()
    out = qc_sentinel2_scl(da)
    assert isinstance(out, xr.DataArray) and out.dims == da.dims and out.dtype == np.float64
    xr.testing.assert_equal(out.coords.to_dataset(), da.coords.to_dataset())
    assert out.rio.crs == da.rio.crs and out.rio.transform() == da.rio.transform()
    for key in ("scale_factor", "add_offset", "_FillValue", "nodata"):
        assert key not in out.attrs and key not in out.encoding
    assert not np.isnan(out.values).any()
    np.testing.assert_array_equal(out.values, qc_sentinel2_scl(da.values))


def test_dataarray_with_nan():
    da = xr.DataArray(np.array([[4.0, np.nan, 6.0, 9.0]]), dims=("y", "x"))
    out = qc_sentinel2_scl(da)
    np.testing.assert_array_equal(out.values, [[1.0, 0, 0.5, 0.2]])


def test_dataarray_declared_nodata_wins():
    da = xr.DataArray(np.array([0, 255, 1], dtype=np.uint8), dims="x", attrs={"_FillValue": 255})
    np.testing.assert_array_equal(qc_modis_summary(da).values, [1.0, 0, 0.5])
    da = xr.DataArray(np.array([0, 1], dtype=np.uint8), dims="x", attrs={"_FillValue": 0})
    np.testing.assert_array_equal(qc_modis_summary(da).values, [0, 0.5])
    da = xr.DataArray(np.array([0, 1], dtype=np.uint8), dims="x", attrs={"nodata": 0})
    np.testing.assert_array_equal(qc_modis_summary(da).values, [0, 0.5])


def test_lazy_dataarray_stays_lazy():
    pytest.importorskip("dask")
    da = scl_cube().chunk({"time": 1})
    out = qc_sentinel2_scl(da)
    assert out.chunks is not None
    np.testing.assert_array_equal(out.values, qc_sentinel2_scl(scl_cube().values))


@pytest.mark.parametrize("as_path", [str, lambda p: p])
def test_raster_path_with_nodata(tmp_path, as_path):
    tr = from_origin(500000, 9000000, 30, 30)
    codes = np.array([[[0, 1, 255], [2, 3, 0]], [[255, 0, 1], [3, 2, 255]]], dtype=np.uint8)
    path = tmp_path / "qa.tif"
    with rasterio.open(path, "w", driver="GTiff", height=2, width=3, count=2, dtype="uint8",
                       crs="EPSG:32722", transform=tr, nodata=255) as dst:
        dst.write(codes)
    out = qc_modis_summary(as_path(path))
    assert isinstance(out, xr.DataArray) and out.shape == (2, 2, 3)
    expected = np.array([[[1.0, 0.5, 0], [0.2, 0.2, 1.0]], [[0, 1.0, 0.5], [0.2, 0.2, 0]]])
    np.testing.assert_array_equal(out.values, expected)
    assert out.rio.crs == rasterio.crs.CRS.from_epsg(32722)
    assert out.rio.transform() == tr
    assert not np.isnan(out.values).any()
    # NoData equal to a good code: the declared NoData prevails
    path0 = tmp_path / "qa0.tif"
    with rasterio.open(path0, "w", driver="GTiff", height=1, width=3, count=1, dtype="uint8",
                       crs="EPSG:32722", transform=tr, nodata=0) as dst:
        dst.write(np.array([[[0, 1, 3]]], dtype=np.uint8))
    np.testing.assert_array_equal(qc_modis_summary(str(path0)).values.ravel(), [0, 0.5, 0.2])


# ---- Weights feeding the smoother and the phenology ----

def test_weights_feed_smooth_and_phenology():
    dates = pd.date_range("2020-01-01", periods=46, freq="8D")
    rng = np.random.default_rng(3)
    doy = dates.dayofyear.to_numpy()
    base = 0.5 + 0.3 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    values = (base[:, None, None] + rng.normal(0, 0.02, (46, 2, 2))).astype(np.float32)
    tr = from_origin(500000, 9000000, 30, 30)
    ndvi = xr.DataArray(
        values, dims=("time", "y", "x"), name="ndvi",
        coords={"time": dates, "y": tr.f - 30 * (np.arange(2) + 0.5), "x": tr.c + 30 * (np.arange(2) + 0.5)},
    ).rio.write_crs("EPSG:32722").rio.write_transform(tr)
    scl = xr.zeros_like(ndvi, dtype=np.uint8) + 4
    scl[10:13] = 9
    ndvi[10:13] = 0.0  # clouds read as a drop
    weights = qc_sentinel2_scl(scl)
    assert weights.dims == ndvi.dims
    smooth = zeit.smooth(ndvi, lmbda=10, weights=weights)
    plain = zeit.smooth(ndvi, lmbda=10)
    truth = base[11]
    assert abs(float(smooth[11, 0, 0]) - truth) < abs(float(plain[11, 0, 0]) - truth)
    pheno = zeit.phenology(ndvi, weights=weights)
    assert pheno is not None
