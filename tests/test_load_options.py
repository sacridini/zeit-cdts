"""load_raster(scale=, qa=, bands=, window=, pixel=), zeit.qa_mask, zeit.nd, ZonalTables.to_csv."""
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit

RNG = np.random.default_rng(11)
H, W = 12, 16
TRANSFORM = from_origin(500000.0, 8000000.0, 30.0, 30.0)
DATES = ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"]


def write(path, data, descriptions=None, scales=None, offsets=None, nodata=None, crs="EPSG:32722",
          transform=TRANSFORM):
    data = np.asarray(data)
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                       dtype=data.dtype, crs=crs, transform=transform if crs else None, nodata=nodata) as dst:
        dst.write(data)
        if descriptions:
            dst.descriptions = descriptions
        if scales:
            dst.scales = scales
        if offsets:
            dst.offsets = offsets
    return str(path)


@pytest.fixture
def landsat(tmp_path):
    """One 6-band file a date (blue, green, red, nir, swir1, fmask), reflectance as Int16 x 10000 with a scale."""
    folder = tmp_path / "landsat"
    folder.mkdir()
    refl = RNG.integers(500, 6000, (len(DATES), 5, H, W)).astype("int16")
    refl[0, 0, 0, 0] = -9999
    fmask = RNG.choice([0, 1, 2, 3, 4], size=(len(DATES), 1, H, W), p=[0.5, 0.1, 0.15, 0.1, 0.15]).astype("int16")
    for k, d in enumerate(DATES):
        data = np.concatenate([refl[k], fmask[k]])
        write(folder / f"LC08_{d.replace('-', '')}_SR.tif", data,
              descriptions=["blue", "green", "red", "nir", "swir1", "fmask"], nodata=-9999,
              scales=[1e-4] * 5 + [1.0], offsets=[0.0] * 6)
    return folder, refl, fmask


# --- ZonalTables.to_csv ------------------------------------------------------------------------------------

def test_zonal_tables_to_csv_same_bytes(tmp_path):
    long = pd.DataFrame({"id": ["0", "1"], "mean": [1.5, np.nan]})
    wide = pd.DataFrame({"id": ["0", "1"], "pixels": [4, 0]})
    tables = zeit.ZonalTables(long, wide, None)
    paths = tables.to_csv(tmp_path / "new" / "out")
    assert [p.name for p in paths] == ["zonal_long.csv", "zonal_wide.csv"]
    long.to_csv(tmp_path / "ref_long.csv", index=False)
    wide.to_csv(tmp_path / "ref_wide.csv", index=False)
    out = tmp_path / "new" / "out"
    assert (out / "zonal_long.csv").read_bytes() == (tmp_path / "ref_long.csv").read_bytes()
    assert (out / "zonal_wide.csv").read_bytes() == (tmp_path / "ref_wide.csv").read_bytes()
    assert not (out / "transitions.csv").exists()
    trans = pd.DataFrame({"id": ["0"], "from": [1], "to": [2]})
    paths = zeit.ZonalTables(long, wide, trans).to_csv(out)
    assert [p.name for p in paths][-1] == "transitions.csv"
    trans.to_csv(tmp_path / "ref_t.csv", index=False)
    assert (out / "transitions.csv").read_bytes() == (tmp_path / "ref_t.csv").read_bytes()


# --- scale=True --------------------------------------------------------------------------------------------

def test_scale_applies_scale_and_offset_after_nodata(tmp_path):
    raw = RNG.integers(100, 9000, (3, H, W)).astype("int16")
    raw[1, 2, 3] = -9999
    path = write(tmp_path / "s.tif", raw, descriptions=["2020", "2021", "2022"], nodata=-9999,
                 scales=[2.5e-5] * 3, offsets=[-0.2] * 3)
    plain = zeit.load_raster(path)
    assert plain.dtype == np.int16 and int(plain.values[1, 2, 3]) == -9999
    scaled = zeit.load_raster(path, scale=True)
    expected = np.where(raw == -9999, np.nan, raw.astype("float32") * 2.5e-5 - 0.2)
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled.values, expected, rtol=1e-6, atol=1e-7)
    assert np.isnan(scaled.values[1, 2, 3]) and np.isfinite(scaled.values).sum() == raw.size - 1
    assert "scale_factor" not in scaled.attrs and "add_offset" not in scaled.attrs
    assert scaled.rio.crs == plain.rio.crs and scaled.dims == plain.dims


def test_scale_without_a_scale_keeps_the_type(tmp_path):
    raw = RNG.integers(0, 100, (3, H, W)).astype("int16")
    path = write(tmp_path / "n.tif", raw, descriptions=["2020", "2021", "2022"])
    out = zeit.load_raster(path, scale=True)
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out.values, raw)


def test_scale_per_band_and_selected_band(tmp_path):
    raw = RNG.integers(100, 900, (3, H, W)).astype("int16")
    path = write(tmp_path / "p.tif", raw, descriptions=["a", "b", "c"], scales=[1.0, 0.5, 2.0],
                 offsets=[0.0, 1.0, -1.0])
    out = zeit.load_raster(path, scale=True, band=[3, 2], dates=["2020-01-01", "2020-02-01"])
    np.testing.assert_allclose(out.values[0], raw[2] * 2.0 - 1.0)
    np.testing.assert_allclose(out.values[1], raw[1] * 0.5 + 1.0)
    one = zeit.load_raster(path, scale=True, band=2)
    np.testing.assert_allclose(one.values, raw[1] * 0.5 + 1.0)


def test_scale_in_a_folder_each_file_its_own(tmp_path):
    for k, (s, o) in enumerate([(0.1, 0.0), (0.2, 1.0)]):
        write(tmp_path / f"v_{2000 + k}.tif", np.full((1, 4, 5), 10, dtype="int16"), scales=[s], offsets=[o])
    out = zeit.load_raster(str(tmp_path), scale=True)
    assert out.dims == ("time", "y", "x")
    assert np.allclose(out.values[0], 1.0) and np.allclose(out.values[1], 3.0)


def test_scale_needs_masked(tmp_path):
    path = write(tmp_path / "m.tif", np.ones((2, 4, 5), dtype="int16"), descriptions=["2020", "2021"])
    with pytest.raises(ValueError, match="scale=True"):
        zeit.load_raster(path, scale=True, masked=False)


# --- qa_mask -------------------------------------------------------------------------------------------------

def test_qa_mask_rules():
    codes = np.arange(0, 256, dtype=np.int64)
    expected = {
        "fmask": np.isin(codes, [0, 1]),
        "landsat_c2": (codes & 0x3B) == 0,
        "nonzero": codes != 0,
        "scl": (codes >= 4) & (codes <= 6),
        "hls_fmask": (codes != 255) & ((codes & 0x1E) == 0),
    }
    for rule, want in expected.items():
        got = zeit.qa_mask(codes, rule)
        assert got.dtype == bool
        np.testing.assert_array_equal(got, want, err_msg=rule)
        np.testing.assert_array_equal(zeit.qa_mask(xr.DataArray(codes, dims="x"), rule).values, want)
    assert zeit.qa_mask(np.array([0.0]), "Landsat-C2")[0]


def test_qa_mask_nan_negative_and_nodata():
    q = np.array([np.nan, -1.0, 0.0, 1.9, 4.0, 255.0])
    assert list(zeit.qa_mask(q, "fmask")) == [False, False, True, True, False, False]
    assert list(zeit.qa_mask(q, "nonzero")) == [False, False, False, True, True, True]
    da = xr.DataArray(np.array([[0, 1, 255]], dtype="uint8"), dims=("y", "x")).rio.write_nodata(255)
    assert list(zeit.qa_mask(da, "nonzero").values[0]) == [False, True, False]
    with pytest.raises(ValueError, match="unknown QA rule"):
        zeit.qa_mask(q, "cloudy")


def test_qa_mask_lazy_on_dask():
    da = xr.DataArray(np.arange(20.0).reshape(4, 5), dims=("y", "x")).chunk({"y": 2})
    out = zeit.qa_mask(da, "scl")
    assert out.chunks is not None
    np.testing.assert_array_equal(out.values, (da.values >= 4) & (da.values <= 6))


# --- bands= and qa= (compared with the xarray the Janus scripts write today) ----------------------------------------

def janus_style(folder, rule_test, bands):
    """What Janus emits today: a read of each band, the scale by hand, the nested where() for the quality band."""
    src = str(folder)
    region = dict(masked=True)
    cube = xr.concat([zeit.load_raster(src, band=n, **region) for n in bands.values()], dim="band")
    cube = cube.assign_coords(band=list(bands)).transpose("time", "band", "y", "x")
    cube = cube * 1e-4
    qa = zeit.load_raster(src, band=6, **region)
    return cube.where(rule_test(qa)), qa


def test_bands_scale_and_qa_equal_the_janus_script(landsat):
    folder, refl, fmask = landsat
    bands = {"blue": 1, "red": 3, "nir": 4}
    new = zeit.load_raster(str(folder), bands=bands, scale=True, qa=6, qa_rule="fmask")
    old, qa = janus_style(folder, lambda q: q.isin([0, 1]), bands)
    assert new.dims == ("time", "band", "y", "x") == old.dims
    assert list(new.band.values) == ["blue", "red", "nir"]
    assert list(new.time.values) == list(old.time.values)
    assert new.dtype == old.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(new.values), np.isnan(old.values))
    np.testing.assert_allclose(new.values, old.values, rtol=1e-6, equal_nan=True)
    # and against numpy straight from the arrays
    expected = refl[:, [0, 2, 3]].astype("float32") * 1e-4
    expected = np.where(np.isin(fmask, [0, 1]), expected, np.nan)
    expected[0, 0, 0, 0] = np.nan  # the NoData of the blue band
    np.testing.assert_allclose(new.values, expected, rtol=1e-6, equal_nan=True)
    assert new.rio.crs == old.rio.crs
    assert np.isnan(new.values).any() and np.isfinite(new.values).any()


def test_qa_from_another_source_and_one_band(landsat, tmp_path):
    folder, refl, fmask = landsat
    qdir = tmp_path / "qa"
    qdir.mkdir()
    for k, d in enumerate(DATES):
        write(qdir / f"QA_{d.replace('-', '')}.tif", fmask[k], nodata=255)
    by_band = zeit.load_raster(str(folder), band=4, scale=True, qa=6, qa_rule="fmask")
    by_source = zeit.load_raster(str(folder), band=4, scale=True, qa=str(qdir), qa_rule="fmask")
    assert by_band.dims == ("time", "y", "x")
    np.testing.assert_array_equal(by_band.values, by_source.values)
    assert np.isfinite(by_band.values).sum() == int(np.isin(fmask[:, 0], [0, 1]).sum()) - int(
        (refl[:, 3] == -9999).sum())


def test_qa_errors(landsat):
    folder, _, _ = landsat
    with pytest.raises(ValueError, match="qa_rule"):
        zeit.load_raster(str(folder), band=4, qa=6)
    with pytest.raises(ValueError, match="needs qa"):
        zeit.load_raster(str(folder), band=4, qa_rule="fmask")
    with pytest.raises(ValueError, match="unknown qa_rule"):
        zeit.load_raster(str(folder), band=4, qa=6, qa_rule="x")
    with pytest.raises(ValueError, match="not both"):
        zeit.load_raster(str(folder), band=4, bands={"nir": 4})


def test_qa_on_a_window_and_a_pixel(landsat):
    folder, refl, fmask = landsat
    full = zeit.load_raster(str(folder), band=4, scale=True, qa=6, qa_rule="fmask")
    win = zeit.load_raster(str(folder), band=4, scale=True, qa=6, qa_rule="fmask", window=(3, 2, 9, 7))
    np.testing.assert_array_equal(win.values, full.values[:, 2:7, 3:9])
    px = zeit.load_raster(str(folder), band=4, scale=True, qa=6, qa_rule="fmask", pixel=(5, 4))
    np.testing.assert_array_equal(px.values, full.values[:, 4, 5])


def test_bands_of_a_stack_file_and_a_dataset(tmp_path):
    raw = RNG.integers(1, 100, (3, H, W)).astype("int16")
    path = write(tmp_path / "x.tif", raw, descriptions=["2020", "2021", "2022"])
    cube = zeit.load_raster(path, bands={"a": 1, "b": 2})
    assert cube.dims == ("time", "band", "y", "x") or cube.dims == ("band", "y", "x")
    ds = xr.Dataset({"B04": (("time", "y", "x"), raw.astype("float32")), "B08": (("time", "y", "x"), raw + 1.0)},
                    coords={"time": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])})
    ds = ds.rio.write_crs("EPSG:32722")
    out = zeit.load_raster(ds, bands={"red": "B04", "nir": "B08"})
    assert out.dims == ("time", "band", "y", "x") and list(out.band.values) == ["red", "nir"]
    np.testing.assert_allclose(out.sel(band="nir").values, raw + 1.0)


# --- nd / normalized_difference -------------------------------------------------------------------------------

def test_nd_and_normalized_difference():
    a = np.array([[0.4, 0.0, 0.2]])
    b = np.array([[0.2, 0.0, 0.6]])
    np.testing.assert_allclose(zeit.nd(a, b), [[1 / 3, np.nan, -0.5]], equal_nan=True)
    ints = zeit.nd(np.array([60000], dtype="uint16"), np.array([65000], dtype="uint16"))
    assert ints.dtype == np.float32 and ints[0] == pytest.approx(-5000 / 125000)
    cube = xr.DataArray(RNG.random((3, 3, 4, 5)).astype("float32"), dims=("time", "band", "y", "x"),
                        coords={"band": ["blue", "red", "B08"]})
    cube.values[:, 1, 0, 0] = 0
    cube.values[:, 2, 0, 0] = 0
    ndvi = zeit.normalized_difference(cube, "nir", "red")
    nir, red = cube.sel(band="B08"), cube.sel(band="red")
    assert ndvi.dims == ("time", "y", "x") and np.isnan(ndvi.values[:, 0, 0]).all()
    np.testing.assert_allclose(ndvi.values[:, 1:], ((nir - red) / (nir + red)).values[:, 1:], rtol=1e-6)
    # the same value compute_indices gives for NDVI
    same = zeit.compute_indices(cube, ["NDVI"]).sel(band="NDVI")
    np.testing.assert_allclose(ndvi.values, same.values, rtol=1e-6, equal_nan=True)
    np.testing.assert_allclose(zeit.normalized_difference(cube, "B08", "red").values, ndvi.values, equal_nan=True)
    with pytest.raises(ValueError, match="swir1"):
        zeit.normalized_difference(cube, "nir", "swir1")


# --- window= and pixel= ------------------------------------------------------------------------------------------

def test_window_equals_isel_and_moves_the_transform(tmp_path):
    raw = RNG.random((5, H, W)).astype("float32")
    path = write(tmp_path / "w.tif", raw, descriptions=[str(y) for y in range(2000, 2005)])
    full = zeit.load_raster(path)
    win = zeit.load_raster(path, window=(3, 2, 10, 9))
    assert win.dims == full.dims and win.sizes["x"] == 7 and win.sizes["y"] == 7
    np.testing.assert_array_equal(win.values, full.values[:, 2:9, 3:10])
    np.testing.assert_array_equal(win.x.values, full.x.values[3:10])
    assert win.rio.crs == full.rio.crs
    t = win.rio.transform()
    assert (t.c, t.f, t.a, t.e) == (500000.0 + 3 * 30, 8000000.0 - 2 * 30, 30.0, -30.0)
    # a window that does not fit
    with pytest.raises(ValueError, match="outside the raster"):
        zeit.load_raster(path, window=(0, 0, W + 1, 3))
    with pytest.raises(ValueError, match="empty"):
        zeit.load_raster(path, window=(4, 4, 4, 8))
    with pytest.raises(ValueError, match="not with clip"):
        zeit.load_raster(path, window=(0, 0, 2, 2), clip=(0, 0, 1, 1))
    with pytest.raises(ValueError, match="not both"):
        zeit.load_raster(path, window=(0, 0, 2, 2), pixel=(0, 0))


def test_window_without_a_crs_and_with_chunks(tmp_path):
    raw = RNG.random((4, H, W)).astype("float32")
    path = write(tmp_path / "nocrs.tif", raw, descriptions=["2000", "2001", "2002", "2003"], crs=None)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = zeit.load_raster(path)
        win = zeit.load_raster(path, window=(2, 1, 6, 5))
        lazy = zeit.load_raster(path, window=(2, 1, 6, 5), chunks="auto")
    assert full.rio.crs is None
    np.testing.assert_array_equal(win.values, full.values[:, 1:5, 2:6])
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, win.values)


def test_window_and_pixel_of_a_folder_and_a_dataarray(landsat):
    folder, refl, _ = landsat
    full = zeit.load_raster(str(folder), band=3, scale=True)
    win = zeit.load_raster(str(folder), band=3, scale=True, window=(1, 1, 5, 4))
    np.testing.assert_array_equal(win.values, full.values[:, 1:4, 1:5])
    px = zeit.load_raster(str(folder), band=3, scale=True, pixel=(7, 6))
    assert px.dims == ("time",)
    np.testing.assert_array_equal(px.values, full.values[:, 6, 7])
    assert float(px.x) == float(full.x[7]) and float(px.y) == float(full.y[6])
    from_da = zeit.load_raster(full, pixel=(7, 6))
    np.testing.assert_array_equal(from_da.values, px.values)
    from_da_w = zeit.load_raster(full, window=(1, 1, 5, 4))
    t = from_da_w.rio.transform()
    assert (t.c, t.f) == (500000.0 + 30, 8000000.0 - 30)
    with pytest.raises(ValueError, match="outside the raster"):
        zeit.load_raster(str(folder), band=3, pixel=(W, 0))


def test_pixel_of_a_stack_and_several_bands(tmp_path):
    raw = RNG.random((6, H, W)).astype("float32")
    path = write(tmp_path / "stack.tif", raw, descriptions=[f"{y}" for y in range(2000, 2006)])
    full = zeit.load_raster(path)
    px = zeit.load_raster(path, pixel=(4, 3))
    assert px.dims == ("time",) and px.sizes["time"] == 6
    np.testing.assert_array_equal(px.values, full.values[:, 3, 4])
    np.testing.assert_array_equal(px.time.values, full.time.values)


def test_pixel_of_a_multiband_dated_file(tmp_path):
    names = [f"{d}_{b}" for d in ("2020-01-15", "2020-02-15") for b in ("red", "nir")]
    raw = RNG.random((4, H, W)).astype("float32")
    path = write(tmp_path / "tb.tif", raw, descriptions=names)
    full = zeit.load_raster(path)
    assert full.dims == ("time", "band", "y", "x")
    px = zeit.load_raster(path, pixel=(2, 5))
    assert px.dims == ("time", "band")
    np.testing.assert_array_equal(px.values, full.values[:, :, 5, 2])
