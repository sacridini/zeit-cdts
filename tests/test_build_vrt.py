import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import zeit


def _tif(path, array, x0, y0, res, nodata=None, crs="EPSG:32723", descriptions=None):
    array = np.asarray(array)
    array = array[None] if array.ndim == 2 else array
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1], count=array.shape[0],
                       dtype=array.dtype, crs=crs, transform=from_origin(x0, y0, res, res), nodata=nodata) as dst:
        dst.write(array)
        if descriptions:
            dst.descriptions = descriptions
    return str(path)


def test_tiles_are_mosaicked_the_later_one_over_the_earlier_except_nodata(tmp_path):
    left = np.arange(1, 25, dtype="float32").reshape(4, 6)
    right = np.full((4, 6), 100.0, dtype="float32")
    right[0, 0] = 0                                                   # NoData: the left tile shows through
    a = _tif(tmp_path / "a.tif", left, 0, 4, 1.0, nodata=0)
    b = _tif(tmp_path / "b.tif", right, 4, 4, 1.0, nodata=0)         # overlaps columns 4-5
    vrt = zeit.build_vrt([a, b], tmp_path / "out" / "m.vrt")
    with rasterio.open(vrt) as src:
        assert (src.width, src.height, src.count, src.nodata) == (10, 4, 1, 0)
        assert tuple(src.transform) == tuple(from_origin(0, 4, 1.0, 1.0))
        got = src.read(1)
    expect = np.zeros((4, 10), dtype="float32")
    expect[:, :6] = left
    expect[:, 4:] = right
    expect[0, 4] = left[0, 4]
    assert np.array_equal(got, expect)


def test_extent_res_and_resampling_follow_gdal_buildvrt(tmp_path):
    a = _tif(tmp_path / "a.tif", np.arange(36, dtype="int16").reshape(6, 6), 0, 6, 1.0)
    vrt = zeit.build_vrt([a], tmp_path / "m.vrt", extent=(1.0, 1.0, 5.0, 5.0), res=0.5)
    with rasterio.open(vrt) as src:
        assert (src.width, src.height) == (8, 8) and src.nodata is None
        assert src.bounds == (1.0, 1.0, 5.0, 5.0)
        got = src.read(1)
    base = np.arange(36, dtype="int16").reshape(6, 6)[1:5, 1:5]      # rows 1-4 from the top, columns 1-4
    assert np.array_equal(got, np.repeat(np.repeat(base, 2, 0), 2, 1))   # nearest: every pixel twice


def test_files_as_bands_and_tiles_per_band_with_names_and_wavelengths(tmp_path):
    red = np.full((2, 4), 1, dtype="uint16"); nir = np.full((2, 4), 2, dtype="uint16")
    r1, r2 = _tif(tmp_path / "r1.tif", red, 0, 2, 1.0), _tif(tmp_path / "r2.tif", red + 10, 4, 2, 1.0)
    n1, n2 = _tif(tmp_path / "n1.tif", nir, 0, 2, 1.0), _tif(tmp_path / "n2.tif", nir + 10, 4, 2, 1.0)
    vrt = zeit.build_vrt([[r1, r2], [n1, n2]], tmp_path / "s.vrt", band_names=["B04", "B08"],
                         wavelengths={"B04": 664.6, "B08": 832.8})
    with rasterio.open(vrt) as src:
        assert src.count == 2 and src.descriptions == ("B04", "B08")
        assert src.tags(1)["wavelength"] == "664.6" and src.tags(2)["wavelength_units"] == "nm"
        assert np.array_equal(src.read(1)[0], [1, 1, 1, 1, 11, 11, 11, 11])
        assert np.array_equal(src.read(2)[0], [2, 2, 2, 2, 12, 12, 12, 12])
    flat = zeit.build_vrt([r1, n1], tmp_path / "f.vrt", separate=True, band_names=["red", "nir"])
    with rasterio.open(flat) as src:
        assert src.count == 2 and src.descriptions == ("red", "nir") and src.tags(1) == {}
        assert src.read(2).max() == 2


def test_bands_of_the_files_are_kept_and_the_names_default_to_the_first_file(tmp_path):
    data = np.arange(24, dtype="float32").reshape(3, 2, 4)
    a = _tif(tmp_path / "a.tif", data, 0, 2, 1.0, descriptions=("Blue", "Green", "Red"))
    b = _tif(tmp_path / "b.tif", data + 100, 4, 2, 1.0, descriptions=("Blue", "Green", "Red"))
    with rasterio.open(zeit.build_vrt([a, b], tmp_path / "m.vrt")) as src:
        assert src.count == 3 and src.descriptions == ("Blue", "Green", "Red")
        assert np.array_equal(src.read(), np.concatenate([data, data + 100], axis=2))


def test_load_raster_reads_a_vrt_and_zero_fills_what_no_tile_covers(tmp_path):
    a = _tif(tmp_path / "a.tif", np.ones((2, 2), "uint8"), 0, 2, 1.0)
    b = _tif(tmp_path / "b.tif", np.full((2, 2), 2, "uint8"), 4, 2, 1.0)    # columns 2-3 uncovered
    got = zeit.load_raster(zeit.build_vrt([a, b], tmp_path / "m.vrt"))
    assert np.asarray(got).tolist() == [[1, 1, 0, 0, 2, 2], [1, 1, 0, 0, 2, 2]]


def test_build_vrt_refuses_what_it_cannot_mosaic(tmp_path):
    a = _tif(tmp_path / "a.tif", np.ones((2, 2), "uint8"), 0, 2, 1.0)
    b = _tif(tmp_path / "b.tif", np.ones((2, 2), "uint8"), 0, 2, 1.0, crs="EPSG:32722")
    c = _tif(tmp_path / "c.tif", np.ones((2, 2, 2), "uint8"), 0, 2, 1.0)
    with pytest.raises(ValueError, match="CRS"):
        zeit.build_vrt([a, b], tmp_path / "x.vrt")
    with pytest.raises(ValueError, match="different numbers of bands"):
        zeit.build_vrt([a, c], tmp_path / "x.vrt")
    with pytest.raises(ValueError):
        zeit.build_vrt([], tmp_path / "x.vrt")
    with pytest.raises(ValueError):
        zeit.build_vrt([a], tmp_path / "x.vrt", resampling="sharpen")
    with pytest.raises(ValueError):
        zeit.build_vrt([a, [a]], tmp_path / "x.vrt")
