"""zeit.texture against scikit-image's graycomatrix / graycoprops and NumPy's window statistics."""

import numpy as np
import pytest
import xarray as xr

import zeit

skimage_feature = pytest.importorskip("skimage.feature")

ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
SKIMAGE = {
    "contrast": "contrast",
    "dissimilarity": "dissimilarity",
    "homogeneity": "homogeneity",
    "asm": "ASM",
    "energy": "energy",
    "correlation": "correlation",
    "glcm_mean": "mean",
    "glcm_variance": "variance",
}


def _reference(img, y, x, window, levels, distance, prop):
    h = window // 2
    patch = img[max(0, y - h): y + h + 1, max(0, x - h): x + h + 1]
    glcm = skimage_feature.graycomatrix(patch, [distance], ANGLES, levels=levels, symmetric=True, normed=True)
    if prop == "entropy":
        p = glcm[:, :, 0, :]
        return float(np.mean([-(q[q > 0] * np.log2(q[q > 0])).sum() for q in np.moveaxis(p, -1, 0)]))
    return float(skimage_feature.graycoprops(glcm, SKIMAGE[prop])[0].mean())


@pytest.mark.parametrize("window,distance", [(5, 1), (7, 2)])
def test_glcm_properties_are_scikit_images(window, distance):
    rng = np.random.default_rng(3)
    levels = 8
    img = rng.integers(0, levels, (20, 23)).astype(np.uint8)
    props = list(SKIMAGE) + ["entropy"]
    t = zeit.texture(img.astype(float), features=props, window=window, levels=levels, distance=distance,
                     lo=0, hi=levels, n_jobs=1)
    for y, x in [(10, 11), (0, 0), (19, 22), (3, 17)]:
        for k, prop in enumerate(props):
            ref = _reference(img, y, x, window, levels, distance, prop)
            assert t.values[k, y, x] == pytest.approx(ref, abs=1e-12), (prop, y, x)


def test_window_statistics_and_gradient():
    rng = np.random.default_rng(5)
    img = rng.normal(size=(15, 16))
    img[4, 5] = np.nan
    t = zeit.texture(img, features=["mean", "std", "gradient"], window=3, n_jobs=1)
    y, x = 7, 8
    win = img[y - 1: y + 2, x - 1: x + 2]
    assert t.sel(feature="mean").values[y, x] == pytest.approx(np.nanmean(win), abs=1e-12)
    assert t.sel(feature="std").values[y, x] == pytest.approx(np.nanstd(win), abs=1e-12)
    from scipy import ndimage

    g = np.hypot(ndimage.sobel(img, axis=1, mode="nearest"), ndimage.sobel(img, axis=0, mode="nearest"))
    assert t.sel(feature="gradient").values[y, x] == pytest.approx(g[y - 1: y + 2, x - 1: x + 2].mean(), abs=1e-12)


def test_a_cube_is_summarized_and_coordinates_kept():
    rng = np.random.default_rng(7)
    cube = xr.DataArray(rng.random((4, 12, 10)), dims=("time", "y", "x"),
                        coords={"y": np.arange(12) * -10.0, "x": np.arange(10) * 10.0})
    t = zeit.texture(cube, features="contrast", summary="max", n_jobs=1)
    assert t.dims == ("feature", "y", "x")
    assert (t.y.values == cube.y.values).all()
    direct = zeit.texture(cube.max("time"), features="contrast", n_jobs=1)
    np.testing.assert_array_equal(t.values, direct.values)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="odd"):
        zeit.texture(np.zeros((5, 5)), window=4)
    with pytest.raises(ValueError, match="unknown texture feature"):
        zeit.texture(np.zeros((5, 5)), features="roughness")
