import json
import os

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit.io import load_raster, save_raster

TRANSFORM = from_origin(-63.0, -10.0, 0.001, 0.001)


def write_stack(path, array, descriptions=None, nodata=None, tags=None, crs="EPSG:4326"):
    count = array.shape[0]
    profile = dict(driver="GTiff", height=array.shape[1], width=array.shape[2], count=count,
                   dtype=array.dtype.name, crs=crs, transform=TRANSFORM)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        for i, desc in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, desc)
        if tags:
            dst.update_tags(**tags)
    return str(path)


@pytest.fixture
def ndvi(tmp_path):
    rng = np.random.default_rng(0)
    array = rng.integers(2000, 9000, size=(5, 20, 30), dtype=np.int16)
    path = write_stack(tmp_path / "ndvi.tif", array, [f"yr{y}" for y in range(1985, 1990)])
    return path, array


def test_exported_at_top_level():
    assert zeit.load_raster is load_raster


def test_band_descriptions_give_the_time_axis(ndvi):
    path, array = ndvi
    cube = load_raster(path)
    assert cube.dims == ("time", "y", "x")
    assert cube.time.dt.year.values.tolist() == [1985, 1986, 1987, 1988, 1989]
    assert cube.dtype == np.int16
    assert cube.rio.crs.to_epsg() == 4326
    assert cube.rio.transform() == TRANSFORM
    np.testing.assert_array_equal(cube.values, array)
    assert cube.chunks is None


@pytest.mark.parametrize("labels, expected", [
    (["1990", "1991", "1992"], ["1990-01-01", "1991-01-01", "1992-01-01"]),
    (["year_2000", "year_2001", "year_2002"], ["2000-01-01", "2001-01-01", "2002-01-01"]),
    (["2020-01-15", "2020-02-16", "2020-03-01"], ["2020-01-15", "2020-02-16", "2020-03-01"]),
    (["20200115", "20200216", "20200301"], ["2020-01-15", "2020-02-16", "2020-03-01"]),
    (["2020_01_15", "2020_02_16", "2020_03_01"], ["2020-01-15", "2020-02-16", "2020-03-01"]),
    (["NDVI_2020-01-15", "NDVI_2020-02-16", "NDVI_2020-03-01"], None),  # a date_band stack
])
def test_description_layouts(tmp_path, labels, expected):
    path = write_stack(tmp_path / "s.tif", np.ones((3, 4, 5), dtype=np.float32), labels)
    cube = load_raster(path)
    if expected is None:
        assert cube.dims == ("time", "band", "y", "x")
        assert cube.band.values.tolist() == ["NDVI"]
    else:
        assert cube.dims == ("time", "y", "x")
        assert [str(t)[:10] for t in cube.time.values] == expected


def test_date_band_descriptions_give_a_4d_cube(tmp_path):
    labels = [f"{d}_{b}" for d in ("2021-01-01", "2021-02-01") for b in ("red", "nir")]
    array = np.arange(4 * 3 * 3, dtype=np.int16).reshape(4, 3, 3)
    cube = load_raster(write_stack(tmp_path / "s.tif", array, labels))
    assert cube.dims == ("time", "band", "y", "x")
    assert cube.band.values.tolist() == ["red", "nir"]   # the file's order, not alphabetical
    np.testing.assert_array_equal(cube.sel(time="2021-02-01", band="nir").values, array[3])
    np.testing.assert_array_equal(cube.sel(time="2021-01-01", band="red").values, array[0])


def test_start_year_and_dates(tmp_path):
    path = write_stack(tmp_path / "plain.tif", np.ones((4, 3, 3), dtype=np.int16))
    assert load_raster(path).dims == ("band", "y", "x")  # no dates: the algorithms ask for them
    assert load_raster(path, start_year=2001).time.dt.year.values.tolist() == [2001, 2002, 2003, 2004]
    cube = load_raster(path, dates=["2020-05-01", "2020-06-01", "2020-07-01", "2020-08-01"])
    assert cube.time.dt.month.values.tolist() == [5, 6, 7, 8]
    assert load_raster(path, dates=[1990, 1991, 1992, 1993]).time.dt.year.values[0] == 1990
    with pytest.raises(ValueError, match="dates for"):
        load_raster(path, dates=[1990, 1991])
    with pytest.raises(ValueError, match="not both"):
        load_raster(path, dates=[1, 2, 3, 4], start_year=1990)


def test_dates_override_descriptions(ndvi):
    path, _ = ndvi
    assert load_raster(path, start_year=2000).time.dt.year.values[0] == 2000


def test_time_tag_and_sidecar_csv(tmp_path):
    times = ["2019-03-01", "2019-04-01", "2019-05-01"]
    path = write_stack(tmp_path / "tagged.tif", np.ones((3, 3, 3), dtype=np.int16),
                       tags={"ZEIT_TIME": json.dumps(times)})
    assert [str(t)[:10] for t in load_raster(path).time.values] == times

    path = write_stack(tmp_path / "old.tif", np.ones((3, 3, 3), dtype=np.int16))
    pd.DataFrame({"Date": times}).to_csv(tmp_path / "old_dates.csv", index=False)
    assert [str(t)[:10] for t in load_raster(path).time.values] == times


def test_band_selection(ndvi):
    path, array = ndvi
    cube = load_raster(path, band=[2, 3])
    assert cube.time.dt.year.values.tolist() == [1986, 1987]
    single = load_raster(path, band=1)
    assert single.dims == ("y", "x")
    np.testing.assert_array_equal(single.values, array[0])


def test_lazy_chunks_and_clip(ndvi):
    path, array = ndvi
    lazy = load_raster(path, chunks="auto")
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, array)

    x0, y1 = -63.0, -10.0
    part = load_raster(path, clip=(x0, y1 - 0.0099, x0 + 0.0099, y1))
    assert part.sizes["y"] == 10 and part.sizes["x"] == 10
    np.testing.assert_array_equal(part.values, array[:, :10, :10])


def test_masked(tmp_path):
    array = np.array([[[1.0, -9999.0], [3.0, 4.0]]] * 3, dtype=np.float32)
    path = write_stack(tmp_path / "f.tif", array, nodata=-9999)
    cube = load_raster(path, start_year=2000)
    assert np.isnan(cube.values[:, 0, 1]).all()
    assert load_raster(path, start_year=2000, masked=False).values[0, 0, 1] == -9999

    ints = write_stack(tmp_path / "i.tif", array.astype(np.int16), nodata=-9999)
    kept = load_raster(ints, start_year=2000)
    assert kept.dtype == np.int16 and kept.rio.nodata == -9999
    floats = load_raster(ints, start_year=2000, masked=True)
    assert np.issubdtype(floats.dtype, np.floating) and np.isnan(floats.values[0, 0, 1])


def test_folder_of_single_date_files(tmp_path):
    folder = tmp_path / "series"
    folder.mkdir()
    for i, d in enumerate(["20220301", "20220101", "20220201"]):
        write_stack(folder / f"LC08_{d}_ndvi.tif", np.full((1, 3, 3), i, dtype=np.int16))
    cube = load_raster(str(folder))
    assert cube.dims == ("time", "y", "x")
    assert cube.time.dt.month.values.tolist() == [1, 2, 3]  # sorted by date
    assert cube.isel(time=0).values[0, 0] == 1
    assert load_raster(str(folder / "*.tif")).sizes["time"] == 3
    files = sorted(str(p) for p in folder.iterdir())
    assert load_raster(files).sizes["time"] == 3


def test_folder_with_pattern_and_bands(tmp_path):
    for d in ("20220101", "20220115"):
        for b in ("B02", "B03"):
            write_stack(tmp_path / f"S2_{d}_{b}.tif", np.ones((1, 3, 3), dtype=np.uint16))
    cube = load_raster(str(tmp_path), pattern=r"_(?P<date>\d{8})_(?P<band>B\d{2})\.tif$")
    assert cube.dims == ("time", "band", "y", "x")
    assert cube.band.values.tolist() == ["B02", "B03"]
    assert cube.sizes["time"] == 2


def test_folder_errors(tmp_path):
    write_stack(tmp_path / "a.tif", np.ones((1, 3, 3), dtype=np.int16))
    write_stack(tmp_path / "b.tif", np.ones((1, 3, 3), dtype=np.int16))
    with pytest.raises(ValueError, match="no date found"):
        load_raster(str(tmp_path))
    assert load_raster(str(tmp_path), start_year=2010).time.dt.year.values.tolist() == [2010, 2011]

    other = tmp_path / "grid"
    other.mkdir()
    write_stack(other / "x_2001.tif", np.ones((1, 3, 3), dtype=np.int16))
    write_stack(other / "x_2002.tif", np.ones((1, 4, 3), dtype=np.int16))
    with pytest.raises(ValueError, match="grid"):
        load_raster(str(other))
    with pytest.raises(FileNotFoundError):
        load_raster(str(tmp_path / "missing.tif"))


def test_xarray_and_numpy_inputs(ndvi):
    path, array = ndvi
    reference = load_raster(path)

    da = xr.DataArray(array, dims=("t", "lat", "lon"), coords={"t": [2000, 2001, 2002, 2003, 2004]})
    cube = load_raster(da)
    assert cube.dims == ("time", "y", "x")
    assert cube.time.dt.year.values[0] == 2000

    cube = load_raster(array, start_year=1985, like=path)
    assert cube.dims == ("time", "y", "x")
    assert cube.rio.crs.to_epsg() == 4326
    np.testing.assert_allclose(cube.x.values, reference.x.values)

    ds = xr.Dataset({"ndvi": reference})
    assert load_raster(ds).dims == ("time", "y", "x")

    with pytest.raises(ValueError, match="cells"):
        load_raster(array[:, :5], start_year=1985, like=path)


def test_zarr_and_netcdf_round_trip(ndvi, tmp_path):
    path, array = ndvi
    cube = load_raster(path)
    store = str(tmp_path / "cube.zarr")
    cube.to_dataset(name="data").to_zarr(store, mode="w")
    back = load_raster(store)
    assert back.dims == ("time", "y", "x")
    assert back.time.dt.year.values.tolist() == cube.time.dt.year.values.tolist()
    np.testing.assert_array_equal(back.values, array)

    nc = str(tmp_path / "cube.nc")
    try:
        cube.to_dataset(name="data").to_netcdf(nc)
    except (ImportError, ValueError):
        pytest.skip("no netCDF engine installed")
    assert load_raster(nc).dims == ("time", "y", "x")


def test_validate_warnings(tmp_path):
    array = np.random.rand(2, 5, 5).astype(np.float32)
    path = write_stack(tmp_path / "small.tif", array)
    with pytest.warns(UserWarning) as record:
        load_raster(path, validate="landtrendr")
    messages = [str(w.message) for w in record]
    assert any("annual time series" in m for m in messages)
    assert any("unscaled" in m for m in messages)
    with pytest.warns(UserWarning, match="dense time series"):
        load_raster(path, validate="ccdc")
    with pytest.raises(ValueError, match="validate"):
        load_raster(path, validate="bogus")


def test_save_then_load_keeps_data(tmp_path):
    array = np.random.randint(0, 10000, size=(10, 50, 50), dtype=np.int16)
    file_path = str(tmp_path / "dummy_stack.tif")
    save_raster(array, file_path)
    cube = load_raster(file_path)
    assert cube.shape == (10, 50, 50)
    np.testing.assert_array_equal(cube.values, array)


# ---------------------------------------------------------------------------
# save_raster
# ---------------------------------------------------------------------------

def test_save_returns_path_and_adds_extension(ndvi, tmp_path):
    path, _ = ndvi
    cube = load_raster(path)
    out = save_raster(cube.isel(time=0), tmp_path / "out" / "first")
    assert out == tmp_path / "out" / "first.tif" and out.exists()
    assert zeit.save_raster is save_raster


def test_round_trip_keeps_dates_georef_and_type(ndvi, tmp_path):
    path, array = ndvi
    cube = load_raster(path)
    out = save_raster(cube, tmp_path / "copy.tif")
    with rasterio.open(out) as src:
        assert src.descriptions == ("1985", "1986", "1987", "1988", "1989")
        assert src.transform == TRANSFORM and src.crs.to_epsg() == 4326
        assert src.dtypes[0] == "int16"
        assert "ZEIT_TIME" in src.tags()
    back = load_raster(out)
    assert back.dims == ("time", "y", "x")
    assert back.time.values.tolist() == cube.time.values.tolist()
    np.testing.assert_array_equal(back.values, array)


def test_sub_annual_dates_use_iso_descriptions(tmp_path):
    times = pd.to_datetime(["2021-01-05", "2021-02-10"])
    da = xr.DataArray(np.ones((2, 3, 3), dtype=np.float32), dims=("time", "y", "x"),
                      coords={"time": times, "y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]}).rio.write_crs(3857)
    out = save_raster(da, tmp_path / "s.tif")
    with rasterio.open(out) as src:
        assert src.descriptions == ("2021-01-05", "2021-02-10")
        assert src.transform == from_origin(0, 3, 1, 1)
    assert load_raster(out).time.values.tolist() == da.time.values.tolist()


def test_4d_cube_writes_date_band_descriptions(tmp_path):
    times = pd.to_datetime(["2021-01-01", "2021-02-01"])
    data = np.arange(2 * 2 * 3 * 3, dtype=np.int16).reshape(2, 2, 3, 3)
    da = xr.DataArray(data, dims=("time", "band", "y", "x"),
                      coords={"time": times, "band": ["red", "nir"], "y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]})
    out = save_raster(da.rio.write_crs(4326), tmp_path / "s.tif")
    with rasterio.open(out) as src:
        assert src.descriptions == ("2021-01-01_red", "2021-01-01_nir", "2021-02-01_red", "2021-02-01_nir")
    back = load_raster(out)
    assert back.dims == ("time", "band", "y", "x")
    np.testing.assert_array_equal(back.sel(band="nir").values, data[:, 1])


def test_numpy_with_like_and_explicit_georef(ndvi, tmp_path):
    path, array = ndvi
    cube = load_raster(path)
    for like in (path, cube):
        out = save_raster(array[0], tmp_path / "m.tif", like=like)
        with rasterio.open(out) as src:
            assert src.transform == TRANSFORM and src.crs.to_epsg() == 4326
    out = save_raster(array[0], tmp_path / "e.tif", crs="EPSG:32721", transform=from_origin(0, 0, 30, 30))
    with rasterio.open(out) as src:
        assert src.crs.to_epsg() == 32721 and src.transform.a == 30
    with pytest.raises(ValueError, match="cells"):
        save_raster(array[0, :5], tmp_path / "bad.tif", like=path)


def test_reference_cube_is_a_deprecated_alias(ndvi, tmp_path):
    path, array = ndvi
    with pytest.warns(DeprecationWarning, match="like="):
        out = save_raster(array[0], tmp_path / "m.tif", reference_cube=load_raster(path))
    with rasterio.open(out) as src:
        assert src.transform == TRANSFORM


def test_no_georeferencing_warns(tmp_path):
    with pytest.warns(UserWarning, match="no georeferencing"):
        save_raster(np.ones((3, 3), dtype=np.uint8), tmp_path / "plain.tif")


def test_nodata_and_types(tmp_path):
    georef = dict(crs="EPSG:4326", transform=TRANSFORM)
    floats = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    with rasterio.open(save_raster(floats, tmp_path / "nan.tif", **georef)) as src:
        assert np.isnan(src.nodata)
    with rasterio.open(save_raster(floats, tmp_path / "fill.tif", nodata=-1, **georef)) as src:
        assert src.nodata == -1 and src.read(1)[0, 1] == -1
    with rasterio.open(save_raster(floats, tmp_path / "int.tif", dtype="int16", nodata=0, **georef)) as src:
        assert src.dtypes[0] == "int16" and src.read(1)[0, 1] == 0
    with rasterio.open(save_raster(floats > 2, tmp_path / "bool.tif", **georef)) as src:
        assert src.dtypes[0] == "uint8"
    with rasterio.open(save_raster(np.array([[1, 300]], dtype=np.int64).repeat(2, 0), tmp_path / "i64.tif", **georef)) as src:
        assert src.dtypes[0] == "uint16"
    with pytest.raises(ValueError, match="NaN"):
        save_raster(np.ones((2, 2), dtype=np.int16), tmp_path / "x.tif", nodata=np.nan, **georef)

    masked = load_raster(write_stack(tmp_path / "m.tif", np.array([[[1.0, -9999.0]]] * 2, dtype=np.float32), nodata=-9999),
                         start_year=2000)
    with rasterio.open(save_raster(masked, tmp_path / "m2.tif")) as src:
        assert src.nodata == -9999 and src.read(1)[0, 1] == -9999


def test_dict_and_dataset_to_folder_or_stack(ndvi, tmp_path):
    path, array = ndvi
    cube = load_raster(path)
    events = {"yod": array[0].astype(np.uint16), "magnitude": array[1].astype(np.float32), "scalar": 3}
    folder = save_raster(events, tmp_path / "events", like=cube)
    assert folder.is_dir()
    assert sorted(p.name for p in folder.iterdir()) == ["magnitude.tif", "yod.tif"]
    with rasterio.open(folder / "yod.tif") as src:
        assert src.dtypes[0] == "uint16" and src.transform == TRANSFORM

    stack = save_raster(events, tmp_path / "events.tif", like=cube)
    with rasterio.open(stack) as src:
        assert src.count == 2 and src.descriptions == ("yod", "magnitude")
        assert src.dtypes[0] == "float32"

    ds = xr.Dataset({"a": cube.isel(time=0, drop=True), "b": cube.isel(time=1, drop=True)})
    folder = save_raster(ds, tmp_path / "ds")
    with rasterio.open(folder / "b.tif") as src:
        assert src.transform == TRANSFORM
        np.testing.assert_array_equal(src.read(1), array[1])


def test_dask_written_block_by_block(ndvi, tmp_path):
    path, array = ndvi
    lazy = load_raster(path, chunks={"time": -1, "y": 7, "x": -1})
    out = save_raster(lazy, tmp_path / "lazy.tif")
    np.testing.assert_array_equal(load_raster(out).values, array)


def test_cog_and_other_formats(ndvi, tmp_path):
    path, array = ndvi
    cube = load_raster(path)
    cog = save_raster(cube, tmp_path / "cog.tif", driver="COG")
    assert cog.exists() and not list(tmp_path.glob("*__tmp__*"))
    np.testing.assert_array_equal(load_raster(cog).values, array)

    store = save_raster(cube, tmp_path / "cube.zarr")
    back = load_raster(store)
    assert back.time.values.tolist() == cube.time.values.tolist()
    np.testing.assert_array_equal(back.values, array)

    img = save_raster(cube.isel(time=0), tmp_path / "map.img")
    with rasterio.open(img) as src:
        assert src.driver == "HFA"
