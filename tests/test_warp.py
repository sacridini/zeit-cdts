"""load_raster(..., like=): the warp ported from landschaft, every date on its own.

Against GDAL's warp of each date alone (exact transformations: bit for bit, for every
method and type), the default lattice of coordinates close to it, the same bits in memory,
lazily, from a file and with any number of threads; the crop shortcut; load_raster's grid,
dims, dates and NoData; folders of rasters on different grids; QA bands.
"""

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import Resampling, calculate_default_transform
from rasterio.warp import reproject as gdal_reproject

import zeit
from zeit import _gdaltransform, _warp

SRC_CRS = "EPSG:32633"
TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)
T, H, W = 4, 120, 160
needs_gdal = pytest.mark.skipif(_gdaltransform.open_library() is None, reason="rasterio's GDAL library not found")


def cube(values, transform=TRANSFORM, crs=SRC_CRS, nodata=None, bands=None) -> xr.DataArray:
    values = np.asarray(values)
    h, w = values.shape[-2:]
    coords = {"time": pd.date_range("2020-01-01", periods=values.shape[0], freq="MS"),
              "y": transform.f + transform.e * (np.arange(h) + 0.5),
              "x": transform.c + transform.a * (np.arange(w) + 0.5)}
    dims = ("time", "y", "x")
    if bands is not None:
        dims = ("time", "band", "y", "x")
        coords["band"] = bands
    da = xr.DataArray(values, dims=dims, coords=coords, name="ndvi")
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    return da.rio.write_nodata(nodata, encoded=False) if nodata is not None else da


def cloudy(dtype=np.float32, nodata=None, seed=0) -> np.ndarray:
    """A series whose dates have clouds (NoData) in different places."""
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        v = rng.normal(5000, 1500, (T, H, W)).astype(dtype)
        hole = np.nan if nodata is None else nodata
    else:
        info = np.iinfo(dtype)
        v = rng.integers(max(info.min, -1000), min(int(info.max), 1000) + 1, (T, H, W)).astype(dtype)
        hole = nodata
    if hole is not None:
        for t in range(T):
            v[t, rng.random((H, W)) < 0.05] = hole
            v[t, 10 + 20 * t:40 + 20 * t, 30:90] = hole
    return v


def target(crs="EPSG:3035", res=20.0) -> _warp.Grid:
    t, w, h = calculate_default_transform(SRC_CRS, crs, W, H, *array_bounds(H, W, TRANSFORM), resolution=res)
    return _warp.Grid(t, (h, w), CRS.from_user_input(crs), t.c + t.a * (np.arange(w) + 0.5),
                      t.f + t.e * (np.arange(h) + 0.5))


def gdal_date(values2d, transform, grid, method, src_nodata, dst_nodata, dtype):
    """GDAL's warp of one date alone, the whole destination in one chunk."""
    ref = np.full(grid.shape, dst_nodata, dtype=dtype)
    gdal_reproject(values2d, ref, src_transform=transform, src_crs=CRS.from_user_input(SRC_CRS),
                   src_nodata=src_nodata, dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=dst_nodata,
                   resampling=Resampling[method], tolerance=0.0, warp_mem_limit=4096)
    return ref


def same(a, b) -> np.ndarray:
    a, b = np.asarray(a), np.asarray(b)
    if np.issubdtype(a.dtype, np.floating):
        return (a == b) | (np.isnan(a) & np.isnan(b))
    return a == b


@pytest.fixture
def small_blocks(monkeypatch):
    """Lazy tasks of 64 cells a side, so that a small raster is split in many."""
    monkeypatch.setattr(_warp, "_BLOCK", 64)


# --- values: GDAL's, date by date --------------------------------------------------


@needs_gdal
@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode",
                                    "min", "max", "med", "q1", "q3", "rms"])
def test_every_date_is_gdals_warp_of_it_alone(method):
    da = cube(cloudy())
    grid = target()
    out = _warp.to_grid(da, grid, method, tolerance=0)
    assert out.dims == ("time", "y", "x") and out.shape == (T, *grid.shape)
    for t in range(T):
        ref = gdal_date(da.values[t], TRANSFORM, grid, method, np.nan, np.nan, np.float32)
        assert same(out.values[t], ref).all(), (method, t)


@needs_gdal
@pytest.mark.parametrize("dtype,nodata", [(np.uint8, 0), (np.uint8, None), (np.int16, -9999), (np.uint16, 0),
                                          (np.int32, -1), (np.float64, -9999.0)])
@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average", "mode"])
def test_every_type_matches_gdal(dtype, nodata, method):
    da = cube(cloudy(dtype, nodata), nodata=nodata)
    grid = target(res=45.0)
    out = _warp.to_grid(da, grid, method, tolerance=0)
    src_nodata, dst_nodata, work = _warp._nodata(da)
    assert out.dtype == dtype and out.rio.nodata == dst_nodata
    for t in range(T):
        ref = gdal_date(da.values[t], TRANSFORM, grid, method, src_nodata, dst_nodata, work)
        assert same(out.values[t], ref).all(), (method, t)


def test_a_cloud_of_one_date_stays_out_of_the_others():
    """GDAL's unified mask (a cell is valid when any band is) would let the NaN of one date
    into the average of another; here every date averages its own clear cells."""
    v = np.full((2, 40, 40), 100.0, dtype=np.float32)
    v[1, :, :] = 200.0
    v[0, 0, 0] = np.nan  # a cloud in the first date only
    da = cube(v)
    t = TRANSFORM
    coarse = from_origin(t.c, t.f, 4 * t.a, 4 * t.a)  # 4 x 4 cells per cell
    grid = _warp.Grid(coarse, (10, 10), CRS.from_user_input(SRC_CRS), coarse.c + coarse.a * (np.arange(10) + 0.5),
                      coarse.f + coarse.e * (np.arange(10) + 0.5))
    out = _warp.to_grid(da, grid, "average")
    assert (out.values[0] == 100.0).all() and (out.values[1] == 200.0).all()


def test_the_default_tolerance_stays_close_to_exact():
    da = cube(cloudy())
    grid = target()
    exact = _warp.to_grid(da, grid, "bilinear", tolerance=0).values
    approx = _warp.to_grid(da, grid, "bilinear").values
    both = ~np.isnan(exact) & ~np.isnan(approx)
    assert both.mean() > 0.5
    assert np.abs(exact[both] - approx[both]).max() < 2.0  # NDVI x 10000: well under a unit of the index


def test_without_gdals_library_pyproj_gives_the_coordinates(monkeypatch):
    da = cube(cloudy())
    grid = target()
    with_gdal = _warp.to_grid(da, grid, "nearest").values
    monkeypatch.setattr(_gdaltransform, "_searched", True)
    monkeypatch.setattr(_gdaltransform, "_lib", None)
    without = _warp.to_grid(da, grid, "nearest").values
    assert same(with_gdal, without).mean() > 0.999


# --- the same bits however it runs -------------------------------------------------


@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic", "average", "mode"])
@pytest.mark.parametrize("tolerance", [0.125, 0.0])
def test_lazy_gives_the_bits_in_memory(small_blocks, method, tolerance):
    da = cube(cloudy())
    grid = target()
    eager = _warp.to_grid(da, grid, method, tolerance=tolerance)
    lazy = _warp.to_grid(da.chunk({"time": 3, "y": 64, "x": 64}), grid, method, tolerance=tolerance)
    assert lazy.chunks is not None and len(lazy.chunks[0]) == 2 and len(lazy.chunks[1]) > 1
    assert same(lazy.values, eager.values).all()


def test_threads_do_not_change_results():
    da = cube(cloudy())
    grid = target()
    one = _warp.to_grid(da, grid, "cubic", n_threads=1).values
    many = _warp.to_grid(da, grid, "cubic", n_threads=4).values
    assert same(one, many).all()


def test_files_are_read_by_window(tmp_path, small_blocks):
    da = cube(cloudy())
    path = tmp_path / "ndvi.tif"
    zeit.save_raster(da, path)
    grid = target()
    eager = _warp.to_grid(da, grid, "bilinear").values
    from_file = zeit.load_raster(path, like=_grid_raster(grid), resampling="bilinear")
    lazy = zeit.load_raster(path, like=_grid_raster(grid), resampling="bilinear", chunks={"time": 2})
    assert lazy.chunks is not None
    assert same(from_file.values, eager).all() and same(lazy.values, eager).all()


def _grid_raster(grid: _warp.Grid) -> xr.DataArray:
    """A (y, x) DataArray on a grid (a reference for like=)."""
    da = xr.DataArray(np.zeros(grid.shape, np.uint8), dims=("y", "x"), coords={"y": grid.y, "x": grid.x})
    return da.rio.write_crs(grid.crs).rio.write_transform(grid.transform)


# --- cropping and padding without the kernel ---------------------------------------


@pytest.mark.parametrize("dtype,nodata", [(np.float32, None), (np.int16, -9999), (np.uint8, None)])
@pytest.mark.parametrize("offset", [(5, -7), (-10, 20), (0, 0)])
def test_a_whole_number_of_cells_apart_is_a_crop(dtype, nodata, offset):
    da = cube(cloudy(dtype, nodata), nodata=nodata)
    row, col = offset
    t = TRANSFORM
    shifted = from_origin(t.c + col * t.a, t.f + row * t.e, t.a, -t.e)
    shape = (H - 15, W + 12)
    grid = _warp.Grid(shifted, shape, CRS.from_user_input(SRC_CRS), shifted.c + t.a * (np.arange(shape[1]) + 0.5),
                      shifted.f + t.e * (np.arange(shape[0]) + 0.5))
    assert _warp._crop_offsets(da, grid) == (row, col)
    out = _warp.to_grid(da, grid, "bilinear")
    lazy = _warp.to_grid(da.chunk({"time": 1}), grid, "bilinear")
    assert out.dtype == dtype and same(lazy.values, out.values).all()
    # GDAL takes the nearest cell for a translation of whole cells, whatever the method
    src_nodata, dst_nodata, work = _warp._nodata(da)
    for k in range(T):
        ref = gdal_date(da.values[k], TRANSFORM, grid, "nearest", src_nodata, dst_nodata, work)
        assert same(out.values[k], ref).all()


def test_the_same_grid_is_returned_as_it_is():
    da = cube(cloudy())
    out = zeit.load_raster(da, like=da.isel(time=0))
    assert same(out.values, da.values).all() and (out.x.values == da.x.values).all()


# --- load_raster(..., like=) -------------------------------------------------------


def test_load_raster_takes_the_grid_dates_and_nodata(tmp_path):
    da = cube(cloudy(np.int16, -9999), nodata=-9999)
    path = tmp_path / "ndvi.tif"
    zeit.save_raster(da, path)
    grid = target()
    ref = _grid_raster(grid)
    ref_path = tmp_path / "ref.tif"
    zeit.save_raster(ref, ref_path)
    for like in (ref, ref_path, str(ref_path)):
        out = zeit.load_raster(path, like=like)
        assert out.dims == ("time", "y", "x") and out.shape == (T, *grid.shape)
        assert (out.time.values == da.time.values).all()
        assert (out.x.values == ref.x.values).all() and (out.y.values == ref.y.values).all()
        assert out.rio.crs == grid.crs and out.rio.transform().almost_equals(grid.transform)
        assert out.dtype == np.int16 and out.rio.nodata == -9999  # nearest for integers, NoData kept
        assert set(np.unique(out.values)) <= set(np.unique(da.values))
    floats = zeit.load_raster(path, like=ref, masked=True)
    assert floats.dtype == np.float32 and np.isnan(floats.values).any() and not (floats.values == -9999).any()


def test_clip_applies_on_the_grid_of_like(tmp_path):
    da = cube(cloudy())
    grid = target()
    ref = _grid_raster(grid)
    x0, y0, x1, y1 = ref.rio.bounds()
    box = (x0, y0, (x0 + x1) / 2, (y0 + y1) / 2)
    out = zeit.load_raster(da, like=ref, clip=box)
    assert out.rio.crs == grid.crs and out.sizes["x"] < grid.shape[1] and out.sizes["y"] < grid.shape[0]


def test_a_folder_on_different_grids_goes_onto_like(tmp_path):
    folder = tmp_path / "scenes"
    folder.mkdir()
    v = cloudy()
    for k, when in enumerate(["20200105", "20200121", "20200206"]):
        da = cube(v[k:k + 1]).isel(time=0)
        if k == 1:  # a scene delivered in another CRS
            da = da.rio.reproject("EPSG:4326")
        zeit.save_raster(da, folder / f"S_{when}_ndvi.tif")
    with pytest.raises(ValueError, match="like="):
        zeit.load_raster(folder)
    ref = folder / "S_20200105_ndvi.tif"
    out = zeit.load_raster(folder, like=ref)
    assert out.dims == ("time", "y", "x") and out.shape == (3, H, W)
    assert list(out.time.dt.day.values) == [5, 21, 6]
    assert same(out.values[0], v[0]).all()  # the reference's own grid: unchanged
    assert np.isfinite(out.values[1]).mean() > 0.8
    lazy = zeit.load_raster(folder, like=ref, chunks="auto")
    assert lazy.chunks is not None and same(lazy.values, out.values).all()


def test_qa_bands_take_the_nearest_cell():
    rng = np.random.default_rng(4)
    v = rng.normal(0.5, 0.1, (T, 2, H, W)).astype(np.float32)
    v[:, 1] = rng.choice([1.0, 2.0, 8.0, 64.0], size=(T, H, W))  # bit flags stored as floats
    da = cube(v, bands=["red", "qa_pixel"])
    out = zeit.load_raster(da, like=_grid_raster(target()))
    assert list(out.band.values) == ["red", "qa_pixel"]
    qa = out.sel(band="qa_pixel").values
    assert set(np.unique(qa[~np.isnan(qa)])) <= {1.0, 2.0, 8.0, 64.0}
    red = out.sel(band="red").values
    assert not set(np.unique(red[~np.isnan(red)])) <= set(np.unique(v[:, 0]))  # interpolated


def test_numpy_with_like_is_georeferenced_as_before():
    da = cube(cloudy())
    out = zeit.load_raster(np.asarray(da.values), dates=da.time.values, like=da)
    assert (out.x.values == da.x.values).all() and out.rio.crs == da.rio.crs


# --- load_raster(..., crs=, res=) --------------------------------------------------

# The grids gdalwarp 3.12 makes for the test cube (EPSG:32633, 160 x 120 cells of 30 m).
GDALWARP_GRIDS = [
    ({"crs": "EPSG:3035"}, (29.972668434973983, 0.0, 4745530.23880969, 0.0, -29.972668434973983, 1964700.0095445109),
     (130, 168)),                                                                    # -t_srs EPSG:3035
    ({"crs": 3035, "res": 25}, (25.0, 0.0, 4745530.23880969, 0.0, -25.0, 1964700.0095445109), (156, 201)),
    ({"res": 45}, (45.0, 0.0, 500000.0, 0.0, -45.0, 4500000.0), (80, 107)),         # -tr 45 45
    ({"res": (100, 70)}, (100.0, 0.0, 500000.0, 0.0, -70.0, 4500000.0), (51, 48)),  # -tr 100 70
]


@pytest.mark.parametrize("kwargs,transform,shape", GDALWARP_GRIDS)
def test_crs_and_res_make_gdalwarps_grid(tmp_path, kwargs, transform, shape):
    da = cube(cloudy())
    path = tmp_path / "ndvi.tif"
    zeit.save_raster(da, path)
    for source in (da, path):
        out = zeit.load_raster(source, **kwargs)
        assert out.dims == ("time", "y", "x") and (out.sizes["y"], out.sizes["x"]) == shape
        assert out.rio.transform().almost_equals(rasterio.Affine(*transform))
        assert out.rio.crs == CRS.from_user_input(kwargs.get("crs", SRC_CRS))
        assert (out.time.values == da.time.values).all()


def test_crs_and_res_warp_like_the_grid_they_make():
    da = cube(cloudy())
    out = zeit.load_raster(da, crs="EPSG:3035", res=25, resampling="average")
    on_grid = zeit.load_raster(da, like=out.isel(time=0), resampling="average")
    assert same(out.values, on_grid.values).all()
    coarse = zeit.load_raster(da, res=60, resampling="average")  # 2 x 2 cells: their mean
    clear = da.values.reshape(T, H // 2, 2, W // 2, 2)
    n = np.isfinite(clear).sum(axis=(2, 4))
    expected = np.where(n > 0, np.nansum(clear, axis=(2, 4)) / np.maximum(n, 1), np.nan)
    assert np.allclose(coarse.values, expected, rtol=1e-6, equal_nan=True)


def test_its_own_crs_keeps_the_grid():
    da = cube(cloudy())
    out = zeit.load_raster(da, crs=SRC_CRS)
    assert same(out.values, da.values).all() and (out.x.values == da.x.values).all()


def test_crs_puts_a_folder_on_a_grid_that_covers_every_file(tmp_path):
    folder = tmp_path / "scenes"
    folder.mkdir()
    v = cloudy()
    zeit.save_raster(cube(v[:1]).isel(time=0), folder / "S_20200105_ndvi.tif")
    other = cube(v[1:2], transform=from_origin(500900.0, 4499100.0, 20.0, 20.0)).isel(time=0)
    zeit.save_raster(other.rio.reproject("EPSG:4326"), folder / "S_20200121_ndvi.tif")
    out = zeit.load_raster(folder, crs=SRC_CRS)
    assert out.rio.crs == CRS.from_user_input(SRC_CRS) and out.sizes["time"] == 2
    x0, y0, x1, y1 = out.rio.bounds()
    assert x0 <= 500000.0 and y1 >= 4500000.0 and x1 >= 500900.0 + 20 * W and y0 <= 4499100.0 - 20 * H
    assert abs(out.rio.resolution()[0]) < 30.0  # the finest of the files', as gdalwarp
    for t in range(2):
        assert np.isfinite(out.values[t]).any()
    fixed = zeit.load_raster(folder, crs=SRC_CRS, res=30, chunks="auto")
    assert fixed.chunks is not None and fixed.rio.resolution() == (30.0, -30.0)


def test_errors():
    da = cube(cloudy())
    with pytest.raises(ValueError, match="not both"):
        zeit.load_raster(da, like=da, crs="EPSG:3035")
    with pytest.raises(ValueError, match="numpy"):
        zeit.load_raster(np.asarray(da.values), dates=da.time.values, res=60)
    with pytest.raises(ValueError, match="positive"):
        zeit.load_raster(da, res=0)
    ref = _grid_raster(target())
    with pytest.raises(ValueError, match="resampling"):
        zeit.load_raster(da, like=ref, resampling="sum")
    with pytest.raises(ValueError, match="no CRS"):
        zeit.load_raster(da, like=xr.DataArray(np.zeros((3, 3)), dims=("y", "x"),
                                               coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]}))
    bare = xr.DataArray(np.zeros((2, 3, 3)), dims=("time", "y", "x"),
                        coords={"time": pd.date_range("2020", periods=2), "y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]})
    with pytest.raises(ValueError, match="no CRS"):
        zeit.load_raster(bare, like=ref)
    with pytest.raises(ValueError, match="no CRS"):
        zeit.load_raster(bare, crs="EPSG:3035")
    with pytest.raises(ValueError, match="no CRS"):
        zeit.load_raster(bare, res=2)
