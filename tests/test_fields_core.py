"""zeit._fields_core (phase 24a) pixel for pixel against scikit-image and scipy.

Matched versions: scikit-image 0.26.0, scipy 1.18, numpy 2.5 (skimage's morphology and
remove_small_objects changed between versions: the parameters used here follow 0.26).
"""

import inspect
import warnings

import numpy as np
import pytest

pytest.importorskip("skimage")
from scipy import ndimage as ndi  # noqa: E402
from scipy import spatial  # noqa: E402
from skimage import exposure, filters, measure, morphology, segmentation  # noqa: E402

from zeit import _fields_core as fc  # noqa: E402


def _markers(rng, shape, n, nlabels):
    m = np.zeros(shape, dtype=np.int32)
    idx = rng.choice(m.size, size=n, replace=False)
    m.ravel()[idx] = rng.integers(1, nlabels + 1, size=n)
    return m


def _blob_mask(rng, shape):
    return ndi.gaussian_filter(rng.random(shape), 2) > 0.5


def _images(rng, shape):
    return {
        "random": rng.random(shape),
        "quantized": rng.integers(0, 4, shape).astype(np.float64),
        "plateau": np.zeros(shape),
        "blocks": np.kron(rng.integers(0, 3, (shape[0] // 5 + 1, shape[1] // 5 + 1)), np.ones((5, 5)))[
            : shape[0], : shape[1]
        ],
        "uint8": rng.integers(0, 6, shape).astype(np.uint8),
        "float32": rng.random(shape).astype(np.float32),
        "int16neg": rng.integers(-3, 3, shape).astype(np.int16),
    }


# ---------------------------------------------------------------- watershed


@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_watershed_equals_skimage(connectivity, use_mask, seed):
    rng = np.random.default_rng(seed)
    shape = (37, 53)
    for name, img in _images(rng, shape).items():
        for n, nl in ((1, 1), (6, 3), (40, 40), (150, 5)):
            markers = _markers(rng, shape, n, nl)
            mask = _blob_mask(rng, shape) if use_mask else None
            want = segmentation.watershed(img, markers, connectivity=connectivity, mask=mask)
            got = fc.watershed(img, markers, connectivity, mask)
            assert got.dtype == np.int32
            assert np.array_equal(got, want), (name, n, nl)


def test_watershed_markers_on_a_mask_edge_and_outside_it():
    shape = (9, 11)
    mask = np.zeros(shape, bool)
    mask[2:7, 3:9] = True
    markers = np.zeros(shape, np.int32)
    markers[2, 3] = 1  # corner of the mask
    markers[6, 8] = 2  # opposite corner
    markers[0, 0] = 3  # outside: dropped
    img = np.zeros(shape)
    for conn in (1, 2):
        want = segmentation.watershed(img, markers, connectivity=conn, mask=mask)
        assert np.array_equal(fc.watershed(img, markers, conn, mask), want)
        assert 3 not in want


def test_watershed_ties_with_many_markers_on_a_plateau():
    # equal value and age 0 markers: the heap layout decides, so it must be the same heap
    rng = np.random.default_rng(7)
    shape = (60, 60)
    img = np.round(rng.random(shape), 1)
    markers = _markers(rng, shape, 300, 300)
    for conn in (1, 2):
        assert np.array_equal(
            fc.watershed(img, markers, conn), segmentation.watershed(img, markers, connectivity=conn)
        )


# --------------------------------------------------------------- morphology


def _fp(kind):
    return {
        "disk1": fc.footprint_disk(1),
        "disk3": fc.footprint_disk(3),
        "disk4": fc.footprint_disk(4),
        "sq3": fc.footprint_square(3),
        "sq5": fc.footprint_square(5),
        "rect": np.ones((3, 5), bool),
    }[kind]


@pytest.mark.parametrize("kind", ["disk1", "disk3", "disk4", "sq3", "sq5", "rect"])
def test_footprints_equal_skimage(kind):
    if kind.startswith("disk"):
        assert np.array_equal(_fp(kind), morphology.disk(int(kind[4:])).astype(bool))


@pytest.mark.parametrize("op", ["dilation", "erosion", "opening", "closing", "white_tophat", "black_tophat"])
@pytest.mark.parametrize("kind", ["disk1", "disk3", "sq3", "sq5", "rect"])
def test_grey_morphology_equals_skimage(op, kind):
    rng = np.random.default_rng(3)
    fp = _fp(kind)
    for name, img in _images(rng, (17, 23)).items():
        want = getattr(morphology, op)(img, fp.astype(np.uint8))
        got = getattr(fc, op)(img, fp)
        assert got.dtype == want.dtype, (name, got.dtype, want.dtype)
        assert np.array_equal(got, want), name


@pytest.mark.parametrize("op", ["dilation", "erosion", "white_tophat", "black_tophat"])
def test_default_footprint_is_the_cross(op):
    img = np.random.default_rng(4).random((20, 21))
    assert np.array_equal(getattr(fc, op)(img), getattr(morphology, op)(img))


@pytest.mark.parametrize("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])
@pytest.mark.parametrize("op", ["dilation", "erosion", "closing", "black_tophat"])
def test_border_modes(op, mode):
    img = np.random.default_rng(5).random((12, 15))
    fp = fc.footprint_disk(2)
    cval = 0.37
    want = getattr(morphology, op)(img, fp.astype(np.uint8), mode=mode, cval=cval)
    got = getattr(fc, op)(img, fp, mode=mode, cval=cval)
    assert np.array_equal(got, want)


def test_footprint_larger_than_the_image():
    img = np.random.default_rng(6).random((3, 4))
    fp = fc.footprint_disk(3)
    for op in ("dilation", "erosion", "black_tophat"):
        assert np.array_equal(getattr(fc, op)(img, fp), getattr(morphology, op)(img, fp.astype(np.uint8)))


def test_notebook_gradient_chain():
    # black_tophat with a flat 3x3 square, minus half the image, then a default dilation
    rng = np.random.default_rng(8)
    img = exposure.rescale_intensity(rng.integers(0, 4000, (40, 50)).astype(np.uint16), out_range=(0, 1))
    sq = np.ones((3, 3), np.uint8)
    want = morphology.dilation(morphology.black_tophat(img, sq) - 0.5 * img)
    got = fc.dilation(fc.black_tophat(img, fc.footprint_square(3)) - 0.5 * img)
    assert np.array_equal(got, want)


# ---------------------------------------------------------- rescale_intensity


@pytest.mark.parametrize(
    "img",
    [
        np.random.default_rng(1).random((30, 31)) * 7000,
        np.random.default_rng(2).normal(size=(30, 31)),
        np.random.default_rng(3).random((30, 31)).astype(np.float32) * 100,
        np.random.default_rng(4).integers(0, 4000, (30, 31)).astype(np.uint16),
        np.random.default_rng(5).integers(-500, 500, (30, 31)).astype(np.int16),
    ],
    ids=["f64", "f64neg", "f32", "u16", "i16"],
)
def test_percentile_rescale_equals_skimage(img):
    p = np.percentile(img, (2, 98))
    assert np.array_equal(fc.percentiles(img, (2, 98)), np.asarray(p, dtype=np.float64))
    want = exposure.rescale_intensity(img, in_range=(p[0], p[1]))
    got = fc.rescale_intensity_percentile(img, 2, 98)
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)
    want = exposure.rescale_intensity(img, in_range=(p[0], p[1]), out_range=(0, 1))
    got = fc.rescale_intensity(img, (p[0], p[1]), (0, 1))
    assert got.dtype == want.dtype
    assert np.array_equal(got, want)


def test_rescale_constant_range_and_percentiles_at_the_ends():
    img = np.full((4, 5), 7.0)
    assert np.array_equal(
        fc.rescale_intensity(img, (7.0, 7.0), (0, 1)), exposure.rescale_intensity(img, in_range=(7, 7), out_range=(0, 1))
    )
    v = np.random.default_rng(9).random(101)
    qs = (0, 0.5, 33.3, 50, 99.99, 100)
    assert np.array_equal(fc.percentiles(v, qs), np.percentile(v, qs))


# ------------------------------------------- hysteresis, boundaries, labels


@pytest.mark.parametrize("seed", range(4))
def test_hysteresis_equals_skimage(seed):
    rng = np.random.default_rng(seed)
    img = ndi.gaussian_filter(rng.random((60, 70)), 2)
    img = (img - img.min()) / (img.max() - img.min())
    for lo, hi in ((0.3, 0.6), (0.5, 0.5), (0.7, 0.4), (0.0, 0.9)):
        want = filters.apply_hysteresis_threshold(img, lo, hi)
        assert np.array_equal(fc.hysteresis_threshold(img, lo, hi), want)


@pytest.mark.parametrize("connectivity", [1, 2, 4])
def test_find_boundaries_equals_skimage(connectivity):
    rng = np.random.default_rng(10)
    labels = segmentation.watershed(rng.random((40, 45)), _markers(rng, (40, 45), 30, 30))
    labels[5:9, 5:9] = 0
    want = segmentation.find_boundaries(labels, connectivity=connectivity)
    assert np.array_equal(fc.find_boundaries(labels, connectivity), want)
    one = np.ones((6, 6), np.int32)
    assert not fc.find_boundaries(one, connectivity).any()


@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("seed", range(3))
def test_label_equals_skimage_and_scipy(connectivity, seed):
    rng = np.random.default_rng(seed)
    shape = (50, 60)
    mask = ndi.gaussian_filter(rng.random(shape), 1.5) > 0.5
    got, n = fc.label(mask, connectivity)
    want, wn = measure.label(mask, connectivity=connectivity, return_num=True)
    assert n == wn and np.array_equal(got, want)
    got, n = fc.ndi_label(mask, connectivity)
    want, wn = ndi.label(mask, ndi.generate_binary_structure(2, connectivity))
    assert n == wn and np.array_equal(got, want)
    ints = rng.integers(0, 3, shape)
    got, n = fc.label(ints, connectivity)
    want, wn = measure.label(ints, connectivity=connectivity, return_num=True)
    assert n == wn and np.array_equal(got, want)
    got, n = fc.label(ints)
    want, wn = measure.label(ints, return_num=True)
    assert n == wn and np.array_equal(got, want)
    got, n = fc.ndi_label(ints, connectivity)
    want, wn = ndi.label(ints, ndi.generate_binary_structure(2, connectivity))
    assert n == wn and np.array_equal(got, want)


@pytest.mark.parametrize("offset", [1, 5])
def test_relabel_sequential_equals_skimage(offset):
    rng = np.random.default_rng(11)
    lab = rng.choice([0, 3, 8, 42, 99, 1000], size=(20, 25)).astype(np.int32)
    got, inv = fc.relabel_sequential(lab, offset)
    want, _, winv = segmentation.relabel_sequential(lab, offset=offset)
    assert np.array_equal(got, want)
    assert got.dtype == np.int32
    assert np.array_equal(inv, np.unique(lab[lab > 0]))
    assert np.array_equal(winv[np.arange(offset, offset + len(inv))], inv)
    with pytest.raises(ValueError):
        fc.relabel_sequential(np.array([[-1, 0]]), 1)


def _skimage_remove_small(ar, size, connectivity):
    """skimage's remove_small_objects with the objects of ``size`` pixels or fewer removed."""
    params = inspect.signature(morphology.remove_small_objects).parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "max_size" in params:
            return morphology.remove_small_objects(ar, max_size=size, connectivity=connectivity)
        return morphology.remove_small_objects(ar, min_size=size + 1, connectivity=connectivity)


@pytest.mark.parametrize("connectivity", [1, 2])
@pytest.mark.parametrize("size", [0, 1, 5, 30])
def test_remove_small_objects_equals_skimage(connectivity, size):
    rng = np.random.default_rng(12)
    mask = ndi.gaussian_filter(rng.random((50, 60)), 1.2) > 0.52
    got = fc.remove_small_objects(mask, max_size=size, connectivity=connectivity)
    assert np.array_equal(got, _skimage_remove_small(mask, size, connectivity))
    lab = measure.label(mask, connectivity=connectivity)
    got = fc.remove_small_objects(lab, max_size=size)
    assert got.dtype == lab.dtype
    assert np.array_equal(got, _skimage_remove_small(lab, size, connectivity))
    # the older min_size: objects smaller than it
    assert np.array_equal(fc.remove_small_objects(lab, size + 1), got)


# ------------------------------------------------------- trim_close_points


def _notebook_trim_close_points(points, distance):
    """The notebook's function (Baumann, 2020; ported with the author's permission), verbatim."""
    x, y = points
    tree = spatial.KDTree(np.array([x, y]).T)
    pairs = tree.query_pairs(distance)
    remove_indices = []
    for pair in pairs:
        if pair[0] in remove_indices:
            continue
        if pair[1] in remove_indices:
            continue
        else:
            remove_indices.append(pair[1])
    keep_indices = np.setdiff1d(np.arange(len(x)), remove_indices)
    return np.array([x[keep_indices], y[keep_indices]]), pairs


@pytest.mark.parametrize("seed", range(5))
def test_trim_close_points_follows_the_notebook_for_its_pair_order(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.random((2, 1000)) * 500
    want, pairs = _notebook_trim_close_points((x, y), 20)
    got = fc.trim_close_points((x, y), 20, pairs=list(pairs))
    assert got.shape == want.shape and np.array_equal(got, want)


@pytest.mark.parametrize("seed", range(3))
def test_close_pairs_equal_query_pairs_and_default_order_is_sorted(seed):
    rng = np.random.default_rng(seed)
    x, y = rng.random((2, 800)) * 300
    ref = sorted(spatial.KDTree(np.array([x, y]).T).query_pairs(15))
    assert np.array_equal(fc.close_pairs((x, y), 15), np.asarray(ref).reshape(-1, 2))
    got = fc.trim_close_points((x, y), 15)
    again = fc.trim_close_points((x, y), 15, pairs=ref)
    assert np.array_equal(got, again)
    # kept points are all farther than the distance
    d = spatial.distance.pdist(got.T)
    assert (d > 15).all()


def test_close_pairs_on_a_lattice_with_points_exactly_at_the_distance():
    gx, gy = np.meshgrid(np.arange(15.0), np.arange(15.0))
    x, y = gx.ravel() * 5, gy.ravel() * 5
    ref = sorted(spatial.KDTree(np.array([x, y]).T).query_pairs(5.0))
    assert np.array_equal(fc.close_pairs((x, y), 5.0), np.asarray(ref).reshape(-1, 2))
