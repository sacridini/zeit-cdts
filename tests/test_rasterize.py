"""zeit.rasterize and zeit.pixel_area: a layer burned on a grid, the area of each pixel."""
import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from shapely.geometry import box

import zeit


def grid(ny=10, nx=10, crs="EPSG:32722", res=10.0, x0=500000.0, ytop=8000100.0, fill=0.0):
    x = x0 + res * (np.arange(nx) + 0.5)
    y = ytop - res * (np.arange(ny) + 0.5)
    da = xr.DataArray(np.full((ny, nx), fill), dims=("y", "x"), coords={"y": y, "x": x})
    return da.rio.write_crs(crs) if crs else da


def zones(crs="EPSG:32722"):
    # columns 2-4 and rows 2-4 (9 pixels); a feature without geometry; columns 7-8, rows 6-7, touching columns 6-9 and rows 5-8
    geoms = [box(500020, 8000050, 500050, 8000080), None, box(500066, 8000016, 500094, 8000044)]
    return gpd.GeoDataFrame({"crop": ["soy", "corn", "Soy bean"], "n": [4, 9, 4], "x": [0.5, 1.5, np.nan],
                             "none": [None, None, None]}, geometry=geoms, crs=crs)


def test_burn_one_is_uint8_on_the_grid():
    like = grid()
    r = zeit.rasterize(zones(), like)
    assert r.dtype == np.uint8 and r.dims == ("y", "x")
    assert r.sum() == 9 + 4  # centres inside: 3 x 3 and 2 x 2
    assert np.array_equal(r.x, like.x) and np.array_equal(r.y, like.y)
    assert r.rio.crs == like.rio.crs and r.rio.transform() == like.rio.transform()
    assert r.rio.nodata is None
    assert r.values[2:5, 2:5].all() and r.values[6:8, 7:9].all()


def test_all_touched_burns_what_the_feature_touches():
    like = grid()
    inside, touched = zeit.rasterize(zones(), like), zeit.rasterize(zones(), like, all_touched=True)
    assert touched.sum() > inside.sum()
    assert ((touched.values - inside.values) >= 0).all()
    # the second box touches 4 x 4 pixels
    assert touched.values[5:9, 6:10].all()


def test_burn_id_counts_every_feature_read():
    r = zeit.rasterize(zones(), grid(), "id")
    assert r.dtype == np.int32 and r.rio.nodata == 0
    assert set(np.unique(r.values)) == {0, 1, 3}  # the feature without geometry is the 2nd
    assert (r.values == 1).sum() == 9


def test_burn_text_makes_classes_in_sorted_order_with_names():
    r = zeit.rasterize(zones(), grid(), "crop")
    assert r.dtype == np.uint8 and r.rio.nodata == 0
    names = dict(zip(r.attrs["flag_values"], r.attrs["flag_meanings"].split()))
    assert names == {1: "Soy_bean", 2: "soy"}  # sorted: capitals first; "corn" has no geometry
    assert set(np.unique(r.values)) == {0, 1, 2}
    assert r.values[3, 3] == 2 and r.values[7, 7] == 1


def test_burn_number_attributes_keep_type_and_nodata():
    ints = zeit.rasterize(zones(), grid(), "n")
    assert ints.dtype == np.int32 and ints.rio.nodata == -2**31
    assert ints.values[3, 3] == 4 and ints.values[0, 0] == -2**31
    reals = zeit.rasterize(zones(), grid(), "x")
    assert reals.dtype == np.float64 and np.isnan(reals.rio.nodata)
    assert reals.values[3, 3] == 0.5 and np.isnan(reals.values[0, 0])
    assert np.isnan(reals.values[7, 7])  # a null value is left out
    with pytest.raises(KeyError):
        zeit.rasterize(zones(), grid(), "missing")


def test_zones_in_another_crs_and_from_a_file(tmp_path):
    like = grid()
    z = zones()
    path = tmp_path / "z.gpkg"
    z.to_file(path, layer="z", driver="GPKG")
    want = zeit.rasterize(z, like, "id")
    assert np.array_equal(zeit.rasterize(z.to_crs(4326), like, "id").values, want.values)
    assert np.array_equal(zeit.rasterize(path, like, "id", layer="z").values, want.values)
    assert np.array_equal(zeit.rasterize(path, like, "n", layer="z").values, zeit.rasterize(z, like, "n").values)


def test_grid_from_a_path_and_save_raster_roundtrip(tmp_path):
    like = grid(fill=1.0)
    zeit.save_raster(like, tmp_path / "like.tif")
    r = zeit.rasterize(zones(), tmp_path / "like.tif", "crop")
    out = zeit.save_raster(r, tmp_path / "r.tif")
    with rasterio.open(out) as src:
        assert src.dtypes[0] == "uint8" and src.nodata == 0
        assert src.transform == like.rio.transform() and src.crs == like.rio.crs
        assert np.array_equal(src.read(1), r.values)


def test_no_feature_gives_an_empty_grid():
    r = zeit.rasterize(zones().iloc[[1]], grid(), "id")
    assert not r.values.any()


def test_pixel_area_projected_geographic_and_without_crs():
    like = grid()
    a = zeit.pixel_area(like)
    assert a.dims == ("y", "x") and a.attrs["units"] == "m2"
    assert abs(a.values.mean() / 100 - 1) < 2e-3  # UTM scale at 0.9996 at its centre
    # an equal-area projection: the transform's area
    ea = zeit.pixel_area(grid(crs="EPSG:6933", res=30.0))
    assert np.all(ea.values == 900.0)
    # no CRS: grid units squared
    assert np.all(zeit.pixel_area(grid(crs=None, res=2.0)).values == 4.0)
    # degrees: the sphere's area of the row within 1 % (the ellipsoid differs), shrinking toward the poles
    geo = grid(ny=40, nx=6, crs="EPSG:4326", res=0.5, x0=-60.0, ytop=70.0)
    g = zeit.pixel_area(geo)
    lat = np.radians(geo.y.values)
    sphere = 6371007.18 ** 2 * np.radians(0.5) ** 2 * np.cos(lat)
    assert np.all(np.abs(g.values[:, 0] / sphere - 1) < 1e-2)
    assert np.all(g.values == g.values[:, :1])
    assert g.values[0, 0] < g.values[-1, 0]
    # every pixel measured, or interpolated from the nodes: the same within 1e-4
    exact = zeit.pixel_area(geo, samples=None)
    assert np.abs(g.values / exact.values - 1).max() < 1e-4


def test_pixel_area_from_a_path_and_a_dataset(tmp_path):
    like = grid(crs="EPSG:4326", res=0.01, x0=-50.0, ytop=-20.0)
    zeit.save_raster(like, tmp_path / "g.tif")
    a, b = zeit.pixel_area(tmp_path / "g.tif"), zeit.pixel_area(xr.Dataset({"v": like}))
    assert np.allclose(a.values, b.values) and np.allclose(a.values, zeit.pixel_area(like).values)
