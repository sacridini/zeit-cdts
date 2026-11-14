"""zeit.qa_mask / zeit.qa_to_fmask with the "force" rule (FORCE's QAI, default SCREEN_QAI)."""
import numpy as np
import pytest
import xarray as xr

import zeit

# (name, code, usable in qa_mask, Fmask code), one field of the QAI alone
CASES = [
    ("clear", 0, True, 0),
    ("nodata", 1 << 0, False, 255),
    ("cloud_buffer", 1 << 1, False, 4),
    ("cloud_opaque", 2 << 1, False, 4),
    ("cloud_cirrus", 3 << 1, False, 4),
    ("shadow", 1 << 3, False, 2),
    ("snow", 1 << 4, False, 3),
    ("water", 1 << 5, True, 1),
    ("aod_interp", 1 << 6, True, 0),
    ("aod_high", 2 << 6, True, 0),
    ("aod_fill", 3 << 6, True, 0),
    ("subzero", 1 << 8, False, 0),
    ("saturation", 1 << 9, False, 0),
    ("low_sun", 1 << 10, True, 0),
    ("illum_moderate", 1 << 11, True, 0),
    ("illum_poor", 2 << 11, True, 0),
    ("illum_deep", 3 << 11, True, 0),
    ("slope", 1 << 13, True, 0),
    ("wvp_fill", 1 << 14, True, 0),
    # combinations
    ("water_buffer", (1 << 5) | (1 << 1), False, 4),
    ("shadow_snow", (1 << 3) | (1 << 4), False, 2),
    ("snow_water", (1 << 4) | (1 << 5), False, 3),
    ("nodata_water", 1 | (1 << 5), False, 255),
    ("nodata_cloud", 1 | (2 << 1), False, 255),
    ("water_aod_high_poor", (1 << 5) | (2 << 6) | (2 << 11), True, 1),
    ("water_subzero", (1 << 5) | (1 << 8), False, 1),
    ("all_benign", (1 << 5) | (3 << 6) | (1 << 10) | (3 << 11) | (1 << 13) | (1 << 14), True, 1),
]
CODES = np.array([c[1] for c in CASES], dtype=np.int16)
USABLE = np.array([c[2] for c in CASES])
FMASK = np.array([c[3] for c in CASES], dtype=np.uint8)


def _da(values, chunks=None):
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": np.arange(values.shape[0]), "x": np.arange(values.shape[1])})
    return da.chunk(chunks) if chunks else da


@pytest.mark.parametrize("name,code,usable,fmask", CASES, ids=[c[0] for c in CASES])
def test_each_case(name, code, usable, fmask):
    a = np.array([code], dtype=np.int16)
    assert bool(zeit.qa_mask(a, "force")[0]) is usable
    assert int(zeit.qa_to_fmask(a, "force")[0]) == fmask


def test_numpy_all():
    np.testing.assert_array_equal(zeit.qa_mask(CODES, "force"), USABLE)
    out = zeit.qa_to_fmask(CODES, "force")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, FMASK)


@pytest.mark.parametrize("chunks", [None, 4])
def test_dataarray_and_dask(chunks):
    da = _da(CODES.reshape(-1, 1), {"y": chunks, "x": 1} if chunks else None)
    m = zeit.qa_mask(da, "force")
    f = zeit.qa_to_fmask(da, "force")
    assert (m.chunks is not None) == bool(chunks)
    assert (f.chunks is not None) == bool(chunks)
    np.testing.assert_array_equal(m.values.ravel(), USABLE)
    np.testing.assert_array_equal(f.values.ravel(), FMASK)
    assert f.dtype == np.uint8


def test_nodata_nan_negative():
    values = np.array([0, -9999, -1, 32, 1], dtype=np.int16)
    np.testing.assert_array_equal(zeit.qa_mask(values, "force"), [True, False, False, True, False])
    np.testing.assert_array_equal(zeit.qa_to_fmask(values, "force"), [0, 255, 255, 1, 255])
    fl = np.array([0.0, np.nan, -3.0, 16.0, 32.0])
    np.testing.assert_array_equal(zeit.qa_mask(fl, "force"), [True, False, False, False, True])
    np.testing.assert_array_equal(zeit.qa_to_fmask(fl, "force"), [0, 255, 255, 3, 1])
    # a declared nodata that is a valid-looking code (0) is masked
    da = _da(np.array([[0, 32, 8]], dtype=np.int16)).rio.write_nodata(0)
    np.testing.assert_array_equal(zeit.qa_mask(da, "force").values, [[False, True, False]])
    np.testing.assert_array_equal(zeit.qa_to_fmask(da, "force").values, [[255, 1, 2]])
    # a float DataArray with NaN, as after masked=True
    db = _da(np.array([[np.nan, 32.0, 2.0]]))
    np.testing.assert_array_equal(zeit.qa_mask(db, "force").values, [[False, True, False]])
    np.testing.assert_array_equal(zeit.qa_to_fmask(db, "force").values, [[255, 1, 4]])


def test_rule_name_normalisation():
    for name in ("force", "FORCE", "Force"):
        np.testing.assert_array_equal(zeit.qa_mask(CODES, name), USABLE)
        np.testing.assert_array_equal(zeit.qa_to_fmask(CODES, name), FMASK)


def test_path(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    p = str(tmp_path / "qai.tif")
    data = CODES[None, None, :].astype(np.int16)
    with rasterio.open(p, "w", driver="GTiff", height=1, width=data.shape[2], count=1, dtype="int16",
                       crs="EPSG:32722", transform=from_origin(0, 0, 30, 30), nodata=-32768) as dst:
        dst.write(data)
    np.testing.assert_array_equal(zeit.qa_mask(p, "force").values.ravel(), USABLE)
    np.testing.assert_array_equal(zeit.qa_to_fmask(p, "force").values.ravel(), FMASK)
