"""load_raster(crs=, res=): the grid and the values of gdalwarp's default (GDAL 3.12, ``-t_srs -tr [-tap]``, ``-et`` 0.125).

A 14 x 10 raster of 30 m cells (UTM 23S) warped by ``gdalwarp`` to EPSG:4326; the grid (size and transform) and the
float32 values it wrote are kept as they were written. They need the two things the first versions of the warp
lacked: the size of the grid (gdalwarp takes the extent from the points it samples, rounds the cells of the
suggested extent and drops a border row or column that falls outside the source altogether, which it does for
``-tap``), and the coordinates of GDAL's approximate transformer (``GDALApproxTransform``, one row at a time),
which move bilinear and cubic values by a thousandth in most cells and the nearest cell in some.
"""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

import zeit
from zeit import _warp

W, H = 14, 10
T = from_origin(610000.0, 7410000.0, 30.0, 30.0)
DATA = [
    [0, 8, 16, 24, 32, 40, 8, 16, 24, 32, 40, 48, 16, 24],
    [14, 23, 32, 41, 10, 14, 23, 32, 41, 10, 14, 23, 32, 41],
    [28, 38, 8, 13, 23, 28, 38, 48, 13, 23, 28, 38, 48, 53],
    [42, 13, 19, 30, 36, 42, 13, 19, 30, 36, 42, 53, 19, 30],
    [16, 28, 35, 42, 9, 16, 28, 35, 42, 49, 16, 28, 35, 42],
    [30, 38, 46, 14, 22, 30, 38, 46, 14, 22, 30, 38, 46, 54],
    [44, 13, 22, 31, 40, 44, 13, 22, 31, 40, 44, 53, 22, 31],
    [18, 28, 38, 43, 13, 18, 28, 38, 43, 53, 18, 28, 38, 43],
    [32, 43, 49, 20, 26, 32, 43, 49, 20, 26, 32, 43, 49, 60],
    [46, 18, 25, 32, 39, 46, 18, 25, 32, 39, 46, 58, 25, 32],
]

# what gdalwarp wrote (see the arguments of each case)
REFERENCES = {
    "bilinear": dict(res=0.0004, method="bilinear", tap=False, shape=(7, 10), gt=(0.0004, 0.0, -43.923350492524555, 0.0, -0.0004, -23.41612685354806), values=[
        [6.88299274, 17.0029678, 26.8537025, 27.4862747, 20.8556995, 21.2310085, 27.5316162, 29.0548782, 34.1895485, 25.5538273],
        [24.9955635, 24.4959564, 22.5666389, 20.2060013, 27.5094643, 35.9805527, 22.5365925, 22.2006092, 33.4098549, 43.3739471],
        [29.6846085, 20.8158855, 27.8575993, 29.5229206, 26.0541134, 26.4253368, 33.6336823, 35.2640381, 38.852169, 31.1657181],
        [26.4146023, 37.1492004, 29.0526352, 19.6630611, 29.1726971, 36.9355087, 29.5552101, 28.4703598, 34.2416496, 43.9712334],
        [31.597559, 24.9362755, 30.5305042, 32.4981194, 27.3582497, 27.8684063, 34.7518311, 37.7110214, 40.1037102, 32.897419],
        [28.1167488, 39.6349144, 33.1753044, 22.4707165, 31.1501637, 40.2680283, 33.9243279, 31.0156498, 36.2216682, 47.3583298],
        [37.5174065, 27.0030975, 30.3146667, 37.5009804, 32.5423698, 28.5847034, 31.9064884, 40.3799896, 48.3777161, 33.1919098],
    ]),
    "cubic": dict(res=0.0004, method="cubic", tap=False, shape=(7, 10), gt=(0.0004, 0.0, -43.923350492524555, 0.0, -0.0004, -23.41612685354806), values=[
        [4.47445679, 16.7227802, 28.36763, 28.6849747, 19.6639919, 19.3001652, 28.2943192, 29.2620544, 34.9967804, 23.9069672],
        [26.1766777, 24.9950943, 21.7087002, 18.7475834, 28.448595, 39.2916908, 20.9197693, 20.0029411, 33.1030998, 45.6883354],
        [30.9494991, 18.394392, 28.5195694, 31.7248154, 25.4148979, 23.7413502, 35.627327, 37.3264122, 40.1700974, 29.0244389],
        [25.2819042, 39.0272675, 29.2702942, 17.1282578, 29.9782257, 39.0061264, 28.8491001, 27.3199997, 33.1396332, 45.6892052],
        [32.5092316, 22.4247837, 30.8146515, 35.0637512, 26.4948444, 25.1080685, 35.7424583, 39.3582802, 41.1894302, 30.7838879],
        [26.7381229, 41.0809937, 33.93573, 20.0335693, 31.5327129, 42.0772972, 34.1713867, 30.1984024, 34.9091797, 48.7124023],
        [39.2465286, 25.8880348, 29.3958035, 40.3573303, 32.8991737, 26.5732517, 30.8182297, 41.9599991, 51.4660645, 31.8016376],
    ]),
    "nearest_tap_trimmed": dict(res=0.000308, method="nearest", tap=True, shape=(9, 13), gt=(0.000308, 0.0, -43.923264, 0.0, -0.000308, -23.416008), values=[
        [0, 8, 16, 24, 40, 8, 16, 24, 32, 40, 48, 16, 24],
        [14, 23, 32, 41, 14, 23, 32, 41, 10, 14, 23, 32, 41],
        [28, 38, 8, 13, 23, 38, 48, 13, 23, 28, 38, 48, 53],
        [42, 13, 19, 30, 36, 13, 19, 30, 36, 42, 53, 19, 30],
        [16, 28, 35, 42, 9, 28, 35, 42, 49, 16, 28, 35, 42],
        [30, 38, 46, 14, 22, 38, 46, 14, 22, 30, 38, 46, 54],
        [44, 13, 22, 31, 40, 13, 22, 31, 40, 44, 53, 22, 31],
        [18, 28, 49, 20, 26, 43, 49, 20, 26, 32, 43, 49, 60],
        [46, 18, 25, 32, 39, 46, 25, 32, 39, 46, 58, 25, 32],
    ])
}


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("gdalwarp") / "source.tif"
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=1, dtype="float32", crs="EPSG:32723",
                       transform=T) as f:
        f.write(np.array(DATA, dtype="float32"), 1)
    return path


def call(source, ref, **more):
    kw = dict(crs="EPSG:4326", res=ref["res"], resampling=ref["method"])
    if ref["tap"]:
        kw["tap"] = True
    return zeit.load_raster(source, masked=False, **{**kw, **more})


@pytest.mark.parametrize("name", list(REFERENCES))
def test_the_grid_and_the_values_are_gdalwarps_default(source, name):
    ref = REFERENCES[name]
    out = call(source, ref)
    assert out.shape == ref["shape"]
    assert np.allclose(tuple(out.rio.transform())[:6], ref["gt"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(out.values, np.array(ref["values"], dtype="float32"))


@pytest.mark.parametrize("name", list(REFERENCES))
def test_a_lazy_cube_gives_the_same_values(source, name):
    ref = REFERENCES[name]
    lazy = call(source, ref, chunks={"y": 3, "x": 4})
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, np.array(ref["values"], dtype="float32"))


def test_the_suggested_grid_has_gdalwarps_cells():
    """Landsat-like 48 x 36 cells of 30 m: gdalwarp's 0.0004 degree grid is 35 x 24 cells (the extent of the
    sampled points, not the one rounded to the suggested cells, gives 36 x 24)."""
    t = from_origin(610000.0, 7410000.0, 30.0, 30.0)
    grid = _warp.suggested_grid([(t, (36, 48), CRS.from_epsg(32723))], crs="EPSG:4326", res=0.0004)
    assert grid.shape == (24, 35)
    assert grid.transform.c == pytest.approx(-43.923350492524555, abs=1e-12)
    assert grid.transform.f == pytest.approx(-23.416057471572852, abs=1e-12)

