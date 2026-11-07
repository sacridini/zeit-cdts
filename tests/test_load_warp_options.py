"""load_raster(..., extent=, tap=, nodata=, dtype=, cutline=): the grid and values of gdalwarp's options.

Each case is a small raster warped by ``gdalwarp`` (GDAL 3.12, the options Janus' ``WarpOptions`` makes: ``-te``,
``-tr -tap``, ``-dstnodata``, ``-ot``, ``-cutline -crop_to_cutline``, a layer's grid with ``-te -ts``) and the grid
(size, transform), type, NoData and values it wrote, kept here as they were written.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

import zeit

ROWS, COLS = 12, 16
T = from_origin(600000.0, 7400000.0, 30.0, 30.0)
CRS = "EPSG:32723"
EXTENT = (600100.5, 7399700.7, 600400.2, 7399900.1)
POLYGON = Polygon([(600060, 7399950), (600390, 7399930), (600420, 7399700), (600200, 7399650), (600040, 7399750),
                   (600060, 7399950)])
LL = dict(crs="EPSG:4326", res=0.0004)

IN_LONLAT = {"tap_ll", "dstnodata", "fill0", "dstnodata_u8", "like_cut", "like_plain"}

# what gdalwarp wrote for the options of each case (see ``arguments``)
REFERENCES = {
    "fit": dict(shape=(7, 10), gt=(29.969999999995345, 0.0, 600100.5, 0.0, -28.485714285634458, 7399900.1), dtype="int16", nodata=-9999.0, values=[
        [78, -9999, -9999, -9999, 96, 14, 22, 23, 31, 32],
        [98, -9999, -9999, -9999, 23, 32, 34, 43, 45, 54],
        [21, 31, 34, 37, 40, 50, 53, 56, 66, 69],
        [41, 45, 49, 53, 57, 68, 72, 76, 80, 84],
        [54, 59, 64, 69, 74, 79, 84, 89, 94, 99],
        [74, 80, 86, 92, 91, 97, 13, 19, 25, 31],
        [94, 94, 11, 18, 18, 25, 32, 39, 39, 46]]),
    "tap": dict(shape=(9, 13), gt=(25.0, 0.0, 600100.0, 0.0, -25.0, 7399925.0), dtype="float32", nodata=None, values=[
        [7.375, 7.6625, 8.037499, 8.631249, 8.840972, 7.6048613, 6.035417, 6.1916666, 2.1916666, 2.0416667, 2.3875, 2.8840277, 3.3270833],
        [8.150001, 6.9750004, 7.225, 7.7375, 7.9729166, 5.55625, 2.2625, 2.775, 3.025, 3.5375, 3.875, 4.2708335, 4.68125],
        [7.4125, 1.4958333, 2.2041667, 2.4312499, 2.604861, 3.070139, 3.6520834, 3.8791668, 4.5875, 4.8583336, 5.4208336, 5.8618054, 6.0354166],
        [3.0375, 3.2041667, 3.4958334, 3.74375, 3.9868057, 4.521528, 5.172917, 5.4208336, 5.7125, 6.4416666, 6.8791666, 7.146528, 7.389583],
        [3.9875, 4.43125, 4.74375, 5.05625, 5.36875, 5.9729166, 6.69375, 7.00625, 7.31875, 7.7625, 8.11875, 8.43125, 8.74375],
        [5.2229166, 5.6048613, 5.9868054, 6.36875, 6.7506943, 7.2541666, 7.8062496, 8.188194, 8.570139, 8.952083, 9.334027, 7.528472, 4.8479166],
        [6.61875, 7.070139, 7.521528, 7.9729166, 8.254167, 8.584028, 8.097917, 5.7368054, 6.1881943, 6.639583, 7.090972, 5.354861, 2.6708333],
        [8.2625, 8.464583, 7.110417, 7.44375, 7.55625, 7.7854166, 6.61875, 2.0770833, 2.5979168, 2.9875, 3.4645834, 3.9854167, 4.33125],
        [9.025, 8.441667, 2.1083333, 2.00625, 2.1881945, 2.4868054, 3.0770833, 3.6625001, 4.204167, 4.3083334, 4.7041664, 5.2701387, 5.6854167]]),
    "tap_all": dict(shape=(10, 13), gt=(37.0, 0.0, 599992.0, 0.0, -37.0, 7400000.0), dtype="int16", nodata=-9999.0, values=[
        [10, 15, 20, 30, 35, 40, 45, 50, 60, 65, 70, 75, 85],
        [27, 33, 39, 51, 57, 63, 62, 68, 80, 86, 92, 98, 13],
        [61, 69, 77, -9999, -9999, -9999, 96, 14, 23, 31, 32, 40, 49],
        [78, 87, 89, -9999, -9999, -9999, 23, 32, 43, 45, 54, 56, 67],
        [95, 15, 18, 31, 34, 37, 40, 50, 56, 66, 69, 72, 85],
        [22, 33, 37, 45, 49, 53, 57, 68, 76, 80, 84, 88, 13],
        [56, 62, 68, 80, 86, 92, 91, 97, 19, 25, 31, 37, 42],
        [73, 80, 87, 94, 11, 18, 18, 25, 39, 39, 46, 53, 60],
        [90, 98, 16, 25, 26, 34, 35, 43, 52, 60, 61, 69, 78],
        [17, 26, 28, 39, 48, 50, 52, 61, 72, 74, 83, 85, 96]]),
    "tap_ll": dict(shape=(7, 10), gt=(0.00045, 0.0, -44.02035, 0.0, -0.00045, -23.5071), dtype="uint8", nodata=None, values=[
        [15, 20, 30, 40, 45, 55, 60, 70, 75, 85],
        [51, 58, 65, 79, 79, 93, 10, 17, 24, 31],
        [87, 89, 10, 21, 23, 34, 43, 54, 56, 67],
        [15, 18, 31, 37, 40, 53, 56, 69, 72, 85],
        [44, 49, 59, 64, 74, 84, 89, 99, 14, 24],
        [80, 87, 94, 11, 18, 32, 39, 46, 53, 60],
        [98, 16, 25, 26, 35, 51, 52, 61, 69, 78]]),
    "dstnodata": dict(shape=(8, 12), gt=(0.0004, 0.0, -44.020549940453954, 0.0, -0.0004, -23.507085389133973), dtype="int16", nodata=-1.0, values=[
        [10, 20, 25, 30, 40, 45, 50, 60, 65, 70, 80, 85],
        [44, 58, 65, 65, 79, 79, 86, 10, 10, 17, 24, 31],
        [61, 77, 78, -1, -1, 96, 14, 23, 31, 32, 41, 49],
        [95, 18, 21, 31, 37, 40, 50, 56, 66, 69, 75, 85],
        [22, 33, 41, 45, 53, 57, 68, 76, 80, 84, 92, 13],
        [56, 62, 74, 80, 92, 91, 97, 19, 25, 31, 36, 42],
        [73, 80, 94, 94, 18, 18, 25, 39, 39, 46, 53, 60],
        [90, 98, 17, 25, 50, 52, 61, 72, 74, 83, 87, 96]]),
    "fill0": dict(shape=(8, 12), gt=(0.0004, 0.0, -44.020549940453954, 0.0, -0.0004, -23.507085389133973), dtype="uint8", nodata=None, values=[
        [10, 20, 25, 30, 40, 45, 50, 60, 65, 70, 80, 85],
        [44, 58, 65, 65, 79, 79, 86, 10, 10, 17, 24, 31],
        [61, 77, 78, 86, 95, 96, 14, 23, 31, 32, 41, 49],
        [95, 18, 21, 31, 37, 40, 50, 56, 66, 69, 75, 85],
        [22, 33, 41, 45, 53, 57, 68, 76, 80, 84, 92, 13],
        [56, 62, 74, 80, 92, 91, 97, 19, 25, 31, 36, 42],
        [73, 80, 94, 94, 18, 18, 25, 39, 39, 46, 53, 60],
        [90, 98, 17, 25, 50, 52, 61, 72, 74, 83, 87, 96]]),
    "dstnodata_u8": dict(shape=(8, 12), gt=(0.0004, 0.0, -44.020549940453954, 0.0, -0.0004, -23.507085389133973), dtype="uint8", nodata=255.0, values=[
        [10, 20, 25, 30, 40, 45, 50, 60, 65, 70, 80, 85],
        [44, 58, 65, 65, 79, 79, 86, 10, 10, 17, 24, 31],
        [61, 77, 78, 86, 95, 96, 14, 23, 31, 32, 41, 49],
        [95, 18, 21, 31, 37, 40, 50, 56, 66, 69, 75, 85],
        [22, 33, 41, 45, 53, 57, 68, 76, 80, 84, 92, 13],
        [56, 62, 74, 80, 92, 91, 97, 19, 25, 31, 36, 42],
        [73, 80, 94, 94, 18, 18, 25, 39, 39, 46, 53, 60],
        [90, 98, 17, 25, 50, 52, 61, 72, 74, 83, 87, 96]]),
    "ot": dict(shape=(7, 9), gt=(30.0, 0.0, 600030.0, 0.0, -28.571428571428573, 7399900.0), dtype="int16", nodata=None, values=[
        [-2, -1, -1, 0, 1, 1, 2, 2, -2],
        [0, 1, 1, 2, 2, -2, -2, -1, -1],
        [2, 2, -2, -2, -1, -1, 0, 1, 1],
        [-2, -1, -1, 0, 1, 1, 2, 2, -2],
        [0, 1, 1, 2, 2, -2, -2, -1, -1],
        [2, 2, -2, -2, -1, -1, 0, 1, 1],
        [-2, -1, -1, 0, 1, 1, 2, 2, -2]]),
    "ot_float": dict(shape=(8, 11), gt=(45.0, 0.0, 600000.0, 0.0, -45.0, 7400000.0), dtype="float32", nodata=-9999.0, values=[
        [18.358631, 25.853495, 33.771507, 41.978493, 49.021507, 55.478493, 63.396507, 71.60349, 79.52151, 82.712364, 62.11895],
        [42.723545, 52.099163, 59.166252, 64.80779, 73.18452, 73.72999, 63.455196, 37.284348, 45.34528, 48.048985, 30.370968],
        [67.81217, 77.34528, 81.66724, -9999, -9999, 42.49582, 31.430107, 32.048985, 39.210274, 45.16607, 51.80645],
        [66.3783, 43.346474, 44.44355, 35.750896, 37.927803, 43.00956, 50.212666, 58.827957, 66.09797, 70.27479, 71.03226],
        [35.367725, 37.890083, 44.369175, 51.33453, 57.55436, 67.112305, 74.887695, 81.85305, 78.33214, 60.391876, 30.634409],
        [54.354496, 62.41099, 70.663086, 72.29988, 75.92234, 76.7945, 44.983273, 48.44803, 46.70012, 32.41099, 37.16129],
        [79.44312, 69.28793, 66.74911, 40.069294, 30.930704, 36.323776, 38.12067, 43.80645, 51.267624, 57.40263, 62.951614],
        [47.558037, 33.91398, 30.336021, 38.16129, 44.130375, 51.78629, 60.08871, 67.91398, 75.58602, 80.918015, 87.997986]]),
    "cut": dict(shape=(9, 12), gt=(30.0, 0.0, 600060.0, 0.0, -30.0, 7399940.0), dtype="int16", nodata=-9999.0, values=[
        [58, 65, 65, 72, 79, 79, 86, 93, 10, 10, 17, -9999],
        [77, 78, -9999, -9999, -9999, 96, 14, 22, 23, 31, 32, -9999],
        [89, 98, -9999, -9999, -9999, 23, 32, 34, 43, 45, 54, -9999],
        [18, 21, 31, 34, 37, 40, 50, 53, 56, 66, 69, -9999],
        [37, 41, 45, 49, 53, 57, 68, 72, 76, 80, 84, 88],
        [49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, 14],
        [68, 74, 80, 86, 92, 91, 97, 13, 19, 25, 31, 37],
        [-9999, 94, 94, 11, 18, 18, 25, 32, 39, 39, 46, 53],
        [-9999, -9999, -9999, 26, 34, 35, 43, 51, 52, 60, -9999, -9999]]),
    "cut_tap": dict(shape=(12, 16), gt=(25.0, 0.0, 600025.0, 0.0, -25.0, 7399950.0), dtype="int16", nodata=-1.0, values=[
        [-1, 58, 58, 65, 65, 72, 79, 79, 79, 86, 93, 10, 10, 17, 17, -1],
        [-1, 58, 58, 65, 65, 72, 79, 79, 79, 86, 93, 10, 10, 17, 17, -1],
        [-1, 77, 77, 78, -1, -1, -1, 96, 96, 14, 22, 23, 31, 32, 32, -1],
        [-1, 89, 89, 98, -1, -1, -1, 23, 23, 32, 34, 43, 45, 54, 54, -1],
        [-1, 18, 18, 21, 31, 34, 37, 40, 40, 50, 53, 56, 66, 69, 69, -1],
        [-1, 37, 37, 41, 45, 49, 53, 57, 57, 68, 72, 76, 80, 84, 84, 88],
        [44, 49, 49, 54, 59, 64, 69, 74, 74, 79, 84, 89, 94, 99, 99, 14],
        [44, 49, 49, 54, 59, 64, 69, 74, 74, 79, 84, 89, 94, 99, 99, 14],
        [-1, 68, 68, 74, 80, 86, 92, 91, 91, 97, 13, 19, 25, 31, 31, 37],
        [-1, -1, -1, 94, 94, 11, 18, 18, 18, 25, 32, 39, 39, 46, 46, 53],
        [-1, -1, -1, -1, -1, 26, 34, 35, 35, 43, 51, 52, 60, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 50, -1, -1, -1, -1, -1, -1, -1, -1, -1]]),
    "cut_u8": dict(shape=(15, 19), gt=(20.0, 0.0, 600040.0, 0.0, -20.0, 7399950.0), dtype="uint8", nodata=None, values=[
        [0, 58, 65, 65, 65, 72, 72, 79, 79, 79, 86, 93, 93, 10, 10, 10, 17, 0, 0],
        [0, 58, 65, 65, 65, 72, 72, 79, 79, 79, 86, 93, 93, 10, 10, 10, 17, 0, 0],
        [0, 77, 78, 78, 86, 87, 87, 95, 96, 96, 14, 22, 22, 23, 31, 31, 32, 0, 0],
        [0, 89, 98, 98, 10, 19, 19, 21, 23, 23, 32, 34, 34, 43, 45, 45, 54, 0, 0],
        [0, 89, 98, 98, 10, 19, 19, 21, 23, 23, 32, 34, 34, 43, 45, 45, 54, 0, 0],
        [0, 18, 21, 21, 31, 34, 34, 37, 40, 40, 50, 53, 53, 56, 66, 66, 69, 0, 0],
        [0, 37, 41, 41, 45, 49, 49, 53, 57, 57, 68, 72, 72, 76, 80, 80, 84, 88, 88],
        [0, 37, 41, 41, 45, 49, 49, 53, 57, 57, 68, 72, 72, 76, 80, 80, 84, 88, 88],
        [44, 49, 54, 54, 59, 64, 64, 69, 74, 74, 79, 84, 84, 89, 94, 94, 99, 14, 14],
        [0, 68, 74, 74, 80, 86, 86, 92, 91, 91, 97, 13, 13, 19, 25, 25, 31, 37, 37],
        [0, 68, 74, 74, 80, 86, 86, 92, 91, 91, 97, 13, 13, 19, 25, 25, 31, 37, 37],
        [0, 0, 94, 94, 94, 11, 11, 18, 18, 18, 25, 32, 32, 39, 39, 39, 46, 53, 53],
        [0, 0, 0, 0, 0, 26, 26, 34, 35, 35, 43, 51, 51, 52, 60, 60, 0, 0, 0],
        [0, 0, 0, 0, 0, 26, 26, 34, 35, 35, 43, 51, 51, 52, 60, 60, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    "like_cut": dict(shape=(9, 11), gt=(0.00039999999999971374, 0.0, -44.0207, 0.0, -0.0003999999999998573, -23.5071), dtype="uint8", nodata=255.0, values=[
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        [255, 255, 58, 65, 72, 79, 86, 93, 10, 17, 255],
        [255, 255, 77, 86, 87, 95, 14, 22, 31, 32, 255],
        [255, 255, 18, 31, 34, 37, 50, 53, 66, 69, 255],
        [255, 255, 37, 45, 49, 53, 68, 72, 80, 84, 88],
        [255, 255, 68, 80, 86, 92, 97, 13, 25, 31, 37],
        [255, 255, 255, 94, 11, 18, 25, 32, 39, 46, 53],
        [255, 255, 255, 255, 255, 50, 255, 255, 255, 255, 255],
        [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]]),
    "like_plain": dict(shape=(9, 11), gt=(0.00039999999999971374, 0.0, -44.0207, 0.0, -0.0003999999999998573, -23.5071), dtype="int16", nodata=-9999.0, values=[
        [15, 21, 28, 35, 43, 49, 55, 63, 70, 77, 84],
        [37, 44, 54, 59, 67, 73, 77, 69, 36, 42, 49],
        [64, 72, 80, -9999, -9999, -9999, 40, 29, 31, 38, 44],
        [87, 54, 50, 42, 34, 35, 40, 47, 55, 63, 67],
        [35, 34, 39, 45, 51, 57, 67, 74, 80, 83, 69],
        [48, 54, 62, 69, 77, 83, 84, 50, 51, 54, 30],
        [74, 78, 72, 75, 34, 31, 35, 36, 41, 47, 55],
        [53, 53, 25, 31, 38, 43, 50, 58, 65, 72, 78],
        [-9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999, -9999]]),
}


def write(path, values, dtype, nodata, transform=T, crs=CRS):
    values = np.asarray(values)
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(values.astype(dtype), 1)
    return path


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    import geopandas as gpd

    d = tmp_path_factory.mktemp("warp_options")
    r, c = np.mgrid[0:ROWS, 0:COLS]
    base = (r * 17 + c * 5 + (r * c) % 7) % 90 + 10
    i16 = base.copy()
    i16[3:5, 4:7] = -9999
    cut = gpd.GeoDataFrame(geometry=[POLYGON], crs=CRS)
    cut.to_file(d / "cut.geojson", driver="GeoJSON")
    return {
        "i16": write(d / "i16.tif", i16, "int16", -9999),
        "u8": write(d / "u8.tif", base, "uint8", None),
        "f32": write(d / "f32.tif", base / 10.0 + 0.25, "float32", None),
        "halves": write(d / "halves.tif", ((r * 3 + c) % 9 - 4) * 0.5, "float32", None),
        "grid": write(d / "grid.tif", np.zeros((9, 11)), "uint8", None, from_origin(-44.0207, -23.5071, 0.0004, 0.0004),
                      "EPSG:4326"),
        "cut": d / "cut.geojson",
        "gdf": cut,
    }


def arguments(data, name):
    """The load_raster call Janus' script makes for each case."""
    cut = str(data["cut"])
    return {
        "fit": ("i16", dict(extent=EXTENT, nodata="source")),
        "tap": ("f32", dict(res=25, tap=True, extent=EXTENT, resampling="bilinear", nodata="source")),
        "tap_all": ("i16", dict(res=37, tap=True, nodata="source")),
        "tap_ll": ("u8", dict(tap=True, nodata="source", crs="EPSG:4326", res=0.00045)),
        "dstnodata": ("i16", dict(nodata=-1, **LL)),
        "fill0": ("u8", dict(nodata="source", **LL)),
        "dstnodata_u8": ("u8", dict(nodata=255, **LL)),
        "ot": ("halves", dict(extent=(600030.0, 7399700.0, 600300.0, 7399900.0), dtype="int16", nodata="source",
                              resampling="nearest")),
        "ot_float": ("i16", dict(dtype="float32", nodata=-9999, res=45, resampling="bilinear")),
        "cut": ("i16", dict(cutline=cut, nodata="source")),
        "cut_tap": ("i16", dict(res=25, tap=True, nodata=-1, cutline=cut)),
        "cut_u8": ("u8", dict(res=20, cutline=cut, nodata="source")),
        "like_cut": ("u8", dict(like=str(data["grid"]), cutline=cut, nodata=255, resampling="nearest")),
        "like_plain": ("i16", dict(like=str(data["grid"]), resampling="bilinear", nodata="source")),
    }[name]


def load(data, name, **more):
    source, kw = arguments(data, name)
    return zeit.load_raster(data[source], masked=False, **{**kw, **more})


@pytest.mark.parametrize("name", list(REFERENCES))
def test_grid_type_nodata_and_values_are_gdalwarps(data, name):
    ref = REFERENCES[name]
    out = load(data, name)
    assert out.shape == ref["shape"]
    assert np.allclose(tuple(out.rio.transform())[:6], ref["gt"], rtol=0, atol=1e-12)
    assert out.dtype == np.dtype(ref["dtype"])
    nodata = out.rio.nodata
    assert (nodata is None) == (ref["nodata"] is None) and (nodata is None or float(nodata) == ref["nodata"])
    assert out.rio.crs == rasterio.crs.CRS.from_user_input("EPSG:4326" if name in IN_LONLAT else CRS)
    np.testing.assert_array_equal(out.values, np.array(ref["values"], dtype=ref["dtype"]))


@pytest.mark.parametrize("name", ["fill0", "dstnodata", "ot_float", "cut_tap", "cut_u8", "like_cut", "tap"])
def test_a_lazy_cube_gives_the_same_values(data, name):
    assert load(data, name, chunks={"y": 5, "x": 6}).chunks is not None
    np.testing.assert_array_equal(load(data, name, chunks={"y": 5, "x": 6}).values, load(data, name).values)


def test_saved_files_have_the_nodata_and_type_gdalwarp_writes(data, tmp_path):
    for name in ("cut_tap", "dstnodata_u8", "fill0", "ot"):
        source, kw = arguments(data, name)
        path = zeit.save_raster(zeit.load_raster(data[source], **kw), tmp_path / f"{name}.tif")
        ref = REFERENCES[name]
        with rasterio.open(path) as f:
            assert f.dtypes[0] == ref["dtype"] and f.nodata == ref["nodata"]
            assert (f.height, f.width) == ref["shape"]
            np.testing.assert_array_equal(f.read(1), np.array(ref["values"], dtype=ref["dtype"]))


def test_a_float_result_masks_its_nodata_as_load_raster_does(data):
    source, kw = arguments(data, "ot_float")
    out = zeit.load_raster(data[source], **kw)  # masked="auto": floats get NaN
    ref = np.array(REFERENCES["ot_float"]["values"], dtype="float32")
    np.testing.assert_array_equal(np.isnan(out.values), ref == -9999)
    np.testing.assert_array_equal(out.values[ref != -9999], ref[ref != -9999])
    assert out.rio.encoded_nodata == -9999


def test_the_defaults_are_unchanged(data):
    # no nodata=: a raster without NoData gets the type's maximum where no data covers, as before
    wide = dict(extent=(599940.0, 7399640.0, 600540.0, 7400060.0), res=30)
    out = zeit.load_raster(data["u8"], masked=False, **wide)
    assert out.rio.nodata == 255 and (out.values[0] == 255).all() and (out.values == 255).sum() == 20 * 14 - ROWS * COLS
    plain = zeit.load_raster(data["u8"], masked=False, nodata="source", **wide)
    assert plain.rio.nodata is None and (plain.values[0] == 0).all() and not (plain.values == 255).any()
    # the options change nothing for the cells they do not touch
    assert zeit.load_raster(data["i16"], masked=False).equals(zeit.load_raster(data["i16"], masked=False, nodata=-9999))
    assert zeit.load_raster(data["f32"], dtype="float64").dtype == np.float64


def test_a_geodataframe_a_file_and_geojson_cut_the_same(data):
    kw = dict(res=25, nodata=-1)
    a = zeit.load_raster(data["i16"], cutline=data["cut"], masked=False, **kw)
    b = zeit.load_raster(data["i16"], cutline=data["gdf"], masked=False, **kw)
    c = zeit.load_raster(data["i16"], cutline=POLYGON, masked=False, **kw)  # no CRS: the result's
    d = zeit.load_raster(data["i16"], cutline=POLYGON.__geo_interface__, masked=False, **kw)
    e = zeit.load_raster(data["i16"], cutline=(str(data["cut"]), "cut"), masked=False, **kw)
    for other in (b, c, d, e):
        np.testing.assert_array_equal(a.values, other.values)
        assert a.rio.transform() == other.rio.transform()
    # in another CRS the polygon is transformed to the data's
    ll = data["gdf"].to_crs("EPSG:4326")
    f = zeit.load_raster(data["i16"], cutline=ll, masked=False, **kw)
    assert f.shape == a.shape and (f.values != a.values).mean() < 0.05


def test_a_folder_is_warped_file_by_file_onto_one_grid(data, tmp_path):
    for day, shift in (("20200101", 0), ("20200102", 3)):
        with rasterio.open(data["u8"]) as src:
            values = np.roll(src.read(1), shift, axis=1)
        write(tmp_path / f"scene_{day}.tif", values, "uint8", None)
    kw = dict(res=20, cutline=data["cut"], nodata=0, dtype="int16", masked=False)
    cube = zeit.load_raster(tmp_path, **kw)
    assert cube.dims == ("time", "y", "x") and cube.dtype == np.int16
    for k, day in enumerate(("20200101", "20200102")):
        one = zeit.load_raster(tmp_path / f"scene_{day}.tif", **kw)
        np.testing.assert_array_equal(cube.values[k], one.values)
    assert cube.rio.nodata == 0


def test_a_crop_by_whole_cells_and_a_tap_on_the_grid_copy_the_cells(data):
    i16 = zeit.load_raster(data["i16"], masked=False)
    crop = zeit.load_raster(data["i16"], masked=False, extent=(600120.0, 7399760.0, 600330.0, 7399940.0))
    np.testing.assert_array_equal(crop.values, i16.values[2:8, 4:11])
    # tap: the edges to the nearest multiples of the cell (the top, 7400000, to 7400010: a third of a cell away)
    again = zeit.load_raster(data["i16"], masked=False, res=30, tap=True)
    np.testing.assert_array_equal(again.values, i16.values)
    assert tuple(again.rio.transform())[:6] == (30.0, 0.0, 600000.0, 0.0, -30.0, 7400010.0)


def test_values_equal_to_the_destination_nodata_move_one_step_as_gdal_does(data):
    out = zeit.load_raster(data["u8"], masked=False, nodata=40, res=30, tap=True)
    src = zeit.load_raster(data["u8"], masked=False).values
    assert (src == 40).any() and not (out.values == 40).any()
    np.testing.assert_array_equal(out.values[src == 40], 39)


def test_rounding_and_clamping_to_the_type(data, tmp_path):
    values = np.array([[-1.5, -0.5, 0.5, 1.5], [300.4, -40.0, 254.5, 255.5]])
    path = write(tmp_path / "v.tif", values, "float32", None)
    out = zeit.load_raster(path, masked=False, dtype="uint8", nodata="source")
    np.testing.assert_array_equal(out.values, [[0, 0, 1, 2], [255, 0, 255, 255]])
    out = zeit.load_raster(path, masked=False, dtype="int16", nodata="source")
    np.testing.assert_array_equal(out.values, [[-2, -1, 1, 2], [300, -40, 255, 256]])


def test_errors(data):
    f = data["i16"]
    with pytest.raises(ValueError, match="tap=True needs res"):
        zeit.load_raster(f, tap=True)
    with pytest.raises(ValueError, match="extent= or cutline="):
        zeit.load_raster(f, extent=EXTENT, cutline=POLYGON)
    with pytest.raises(ValueError, match="does not fit the type"):
        zeit.load_raster(f, nodata=-1, dtype="uint8")
    with pytest.raises(ValueError, match="does not fit the type"):
        zeit.load_raster(f, nodata=0.5, dtype="int16")
    with pytest.raises(ValueError, match="nodata must be"):
        zeit.load_raster(f, nodata="gdal")
    with pytest.raises(ValueError, match="need res="):
        zeit.load_raster(f, crs="EPSG:4326", extent=(-44.02, -23.52, -44.01, -23.51))
    with pytest.raises(ValueError, match="not both"):
        zeit.load_raster(f, like=data["grid"], extent=EXTENT)
    with pytest.raises(ValueError, match="own pixels"):
        zeit.load_raster(f, window=(0, 0, 4, 4), cutline=POLYGON)
    with pytest.raises(ValueError, match="extent="):
        zeit.load_raster(f, extent=(1, 2, 3))
    with pytest.raises(ValueError, match="empty"):
        zeit.load_raster(f, extent=(600100.0, 7399900.0, 600000.0, 7399700.0))
    with pytest.raises(ValueError, match="georeferenced"):
        zeit.load_raster(np.zeros((2, 3, 3)), nodata=0)
    with pytest.raises(ValueError, match="real number type"):
        zeit.load_raster(f, dtype="complex64")
    with pytest.raises(ValueError, match="smaller than a cell"):
        zeit.load_raster(f, cutline=Polygon([(600001, 7399999), (600005, 7399999), (600005, 7399995)]))
