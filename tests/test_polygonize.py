"""zeit.polygonize: a map's patches as polygons."""
import numpy as np
import pytest
import xarray as xr

import zeit


def grid(a, crs="EPSG:32722", res=30.0, x0=500000.0, ytop=8000000.0):
    a = np.asarray(a, dtype=float)
    ny, nx = a.shape
    da = xr.DataArray(a, dims=("y", "x"), coords={"y": ytop - res * (np.arange(ny) + 0.5),
                                                   "x": x0 + res * (np.arange(nx) + 0.5)})
    return da.rio.write_crs(crs) if crs else da


MAP = [[1, 1, 2, 2, np.nan, 3],
       [1, 2, 2, 2, np.nan, 3],
       [3, 3, 1, 2, 2, 3],
       [3, 1, 1, 1, 2, 3],
       [3, 3, 3, 1, 1, 1]]


def test_patches_values_pixels_and_columns():
    p = zeit.polygonize(grid(MAP))
    assert list(p.columns) == ["value", "area_ha", "perimeter_m", "pixels", "geometry"]
    assert p.crs == grid(MAP).rio.crs
    assert p.pixels.sum() == np.isfinite(np.array(MAP)).sum()  # every valued pixel in a patch; NaN is none
    assert len(p) == 5 and sorted(p.value) == [1, 1, 2, 3, 3]
    assert sorted(p.pixels) == [3, 4, 6, 7, 8]
    # 900 m2 a pixel, ellipsoid area within 0.2 % of the planar
    assert np.allclose(p.area_ha, p.pixels * 0.09, rtol=2e-3)
    assert (p.perimeter_m > 0).all()


def test_connectivity_joins_diagonals():
    diagonal = grid([[1, 2], [2, 1]])
    four, eight = zeit.polygonize(diagonal, connectivity=4), zeit.polygonize(diagonal, connectivity=8)
    assert (len(four), len(eight)) == (4, 2)
    assert four.pixels.sum() == eight.pixels.sum()
    with pytest.raises(ValueError):
        zeit.polygonize(grid(MAP), connectivity=6)


def test_classes_and_years_columns():
    p = zeit.polygonize(grid([[2000, 2000.5], [2001, 2001.25]]), classes={2000: "a", 2001: "b"}, years=True)
    assert list(p.columns) == ["value", "class", "date", "area_ha", "perimeter_m", "pixels", "geometry"]
    by = dict(zip(p.value, zip(p["class"], p.date)))
    assert by[2000.0] == ("a", "2000")
    assert by[2001.0] == ("b", "2001")
    assert by[2000.5][1] == "2000-07-02"  # 0.5 of the 366 days of 2000
    assert by[2001.25][1] == "2001-04-02"  # 0.25 of 365 days = 91 days
    assert np.isnan(by[2000.5][0]) or by[2000.5][0] is None


def test_sieve_merges_small_patches_and_leaves_a_lone_one_out():
    a = np.ones((8, 8))
    a[2, 2] = 2            # one pixel inside the 1s: merged
    a[:, 7] = 3
    a[0, 0] = np.nan
    p = zeit.polygonize(grid(a), min_pixels=2)
    assert sorted(p.value) == [1.0, 3.0]
    assert p.pixels.sum() == 63
    # the same minimum in hectares: 900 m2 a pixel, 2 pixels = 0.18 ha
    q = zeit.polygonize(grid(a), min_area_ha=0.18)
    assert q.equals(p)
    lone = np.full((4, 4), np.nan)
    lone[1, 1] = 5
    lone[3, 3] = 6
    assert len(zeit.polygonize(grid(lone), min_pixels=2)) == 0  # no neighbour to join: left out
    assert len(zeit.polygonize(grid(lone))) == 2


def test_mean_of_another_map_and_simplify():
    a = np.array([[1, 1, 2], [1, 1, 2], [3, 3, 3.0]])
    other = grid(np.arange(9.0).reshape(3, 3))
    other[0, 0] = np.nan
    p = zeit.polygonize(grid(a), mean_of=other, mean_name="m")
    assert list(p.columns)[-2:] == ["m", "geometry"]
    got = dict(zip(p.value, p.m))
    assert got[1.0] == pytest.approx(np.mean([1, 3, 4]))  # the NaN is left out
    assert got[2.0] == pytest.approx(np.mean([2, 5]))
    assert got[3.0] == pytest.approx(np.mean([6, 7, 8]))
    staircase = np.triu(np.ones((20, 20)))
    plain, simple = zeit.polygonize(grid(staircase)), zeit.polygonize(grid(staircase), simplify=100.0)
    assert simple.geometry.apply(lambda g: len(g.exterior.coords)).sum() < plain.geometry.apply(
        lambda g: len(g.exterior.coords)).sum()
    assert list(simple.pixels) == list(plain.pixels)


def test_geographic_crs_measures_on_the_ellipsoid_and_no_crs_is_planar():
    a = np.array([[1, 1], [1, 1.0]])
    geo = zeit.polygonize(grid(a, crs="EPSG:4326", res=0.001, x0=-50.0, ytop=-20.0))
    assert geo.pixels[0] == 4
    expect = 4 * (0.001 * 111_000) ** 2 * np.cos(np.radians(20)) / 1e4
    assert geo.area_ha[0] == pytest.approx(expect, rel=2e-2)
    flat = zeit.polygonize(grid(a, crs=None, res=10.0, x0=0.0, ytop=20.0))
    assert flat.area_ha[0] == pytest.approx(4 * 100 / 1e4) and flat.perimeter_m[0] == pytest.approx(80.0)
    assert flat.crs is None


def test_reads_a_file_band(tmp_path):
    da = xr.concat([grid(MAP), grid(np.ones((5, 6)))], dim="band").assign_coords(band=[1, 2])
    zeit.save_raster(da, tmp_path / "m.tif")
    assert len(zeit.polygonize(tmp_path / "m.tif", band=2)) == 1
    assert len(zeit.polygonize(tmp_path / "m.tif", band=1)) == 5
