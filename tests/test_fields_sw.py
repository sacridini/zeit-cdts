"""zeit._fields_core (phase 24b): the Monte Carlo stochastic watershed against the notebook's loop.

The kNN graph against scikit-learn's ``kneighbors_graph``, the colouring against pyamg's
``vertex_coloring`` (same colours, not just a valid colouring) and the boundary counts against
``benchmarks/fields/notebook_reference.py`` fed with the very same markers.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("pyamg")
from pyamg.graph import vertex_coloring  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.neighbors import kneighbors_graph  # noqa: E402

from zeit import _fields_core as fc  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fields"))
import notebook_reference as nb  # noqa: E402


def _sets(row, indptr, indices):
    return set(indices[indptr[row]:indptr[row + 1]].tolist())


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n,k", [(2, 1), (15, 20), (21, 20), (22, 20), (60, 5), (300, 20), (1000, 20)])
def test_knn_graph_equals_sklearn(seed, n, k):
    rng = np.random.default_rng(seed)
    pts = rng.random((2, n)) * [[200.0], [150.0]]
    indptr, indices = fc.knn_graph(pts, k)
    kk = min(k, n - 1)
    ref = kneighbors_graph(pts.T, kk)
    assert len(indptr) == n + 1
    for i in range(n):
        assert _sets(i, indptr, indices) == set(ref.indices[ref.indptr[i]:ref.indptr[i + 1]].tolist())


def test_knn_graph_tiny():
    indptr, indices = fc.knn_graph(np.zeros((2, 1)), 3)
    assert indptr.tolist() == [0, 0] and len(indices) == 0
    indptr, indices = fc.knn_graph(np.zeros((2, 0)), 3)
    assert indptr.tolist() == [0]


@pytest.mark.parametrize("seed", range(30))
def test_vertex_coloring_equals_pyamg_on_knn(seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(20, 1001))
    k = int(rng.choice([3, 8, 20]))
    pts = rng.random((2, n)) * 100
    mat = kneighbors_graph(pts.T, k)
    ref = vertex_coloring(mat)
    got = fc.vertex_coloring_mis(mat.indptr, mat.indices)
    assert np.array_equal(got, ref)
    # Zeit's own graph gives the same
    assert np.array_equal(fc.vertex_coloring_mis(*fc.knn_graph(pts, k)), ref)


def test_vertex_coloring_equals_pyamg_on_other_graphs():
    rng = np.random.default_rng(7)
    for n, density in [(1, 0.0), (5, 0.0), (40, 0.05), (200, 0.03), (200, 0.2)]:
        a = sparse.random(n, n, density=density, random_state=int(rng.integers(1 << 30)), format="csr")
        a.data[:] = 1.0
        for g in (
            a,                                   # directed, may hold a diagonal
            (a + a.T).tocsr(),                   # symmetric
            (a + sparse.identity(n)).tocsr(),    # with a full diagonal
        ):
            g = g.tocsr()
            g.sort_indices()
            assert np.array_equal(fc.vertex_coloring_mis(g.indptr, g.indices), vertex_coloring(g))
    # empty rows in the middle of a directed graph
    g = sparse.csr_array(np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0]], dtype=float))
    assert np.array_equal(fc.vertex_coloring_mis(g.indptr, g.indices), vertex_coloring(g))


def _field_image(rng, shape):
    """Blocks of different levels plus noise (uint16 like a band of digital numbers)."""
    r, c = shape
    blocks = np.zeros(shape)
    for i, (r0, c0) in enumerate([(0, 0), (0, c // 3), (0, 2 * c // 3)]):
        blocks[r0:r // 2, c0:c0 + c // 3] = 1500 + 900 * i
    blocks[r // 2:, : c // 2] = 3500
    blocks[r // 2:, c // 2:] = 2200
    img = blocks + rng.normal(0, 60, shape)
    return np.clip(img, 0, 65535).astype(np.uint16)


def _problem(shape, seed):
    rng = np.random.default_rng(seed)
    img = _field_image(rng, shape)
    grad = nb.gradient(nb.stretch(img))
    mask = np.ones(shape, dtype=bool)
    mask[shape[0] // 3: shape[0] // 3 + shape[0] // 4, shape[1] // 4:] = False  # cuts through the fields
    return grad, mask


@pytest.mark.parametrize(
    "shape,seed,n_real,markers,dist",
    [((64, 64), 0, 12, 200, 6), ((128, 96), 1, 15, 300, 8), ((200, 200), 2, 10, 500, 10)],
)
def test_counts_equal_the_notebook(shape, seed, n_real, markers, dist):
    grad, mask = _problem(shape, seed)
    sets = nb.draw_markers(shape, n_real, markers, dist, seed=seed)
    ref = nb.boundary_frequency(grad, mask.astype(int), markers=sets)
    got = fc.stochastic_watershed_counts(grad, mask, marker_sets=sets, n_jobs=1)
    assert got.dtype == np.uint32
    assert np.array_equal(got, ref.astype(np.uint32))
    assert got.max() > 0


def test_counts_equal_the_notebook_other_settings():
    shape = (96, 96)
    grad, mask = _problem(shape, 5)
    sets = nb.draw_markers(shape, 12, 400, 7, seed=5)
    ref = nb.boundary_frequency(grad, mask, markers=sets, n_neighbors=8, disk_radius=2)
    got = fc.stochastic_watershed_counts(grad, mask, sets, n_neighbors=8, disk_radius=2, n_jobs=2)
    assert np.array_equal(got, ref.astype(np.uint32))
    # no mask: the notebook with an all-ones mask
    ref = nb.boundary_frequency(grad, np.ones(shape, dtype=int), markers=sets)
    assert np.array_equal(fc.stochastic_watershed_counts(grad, None, sets, n_jobs=2), ref.astype(np.uint32))


def test_counts_do_not_depend_on_threads():
    shape = (80, 70)
    grad, mask = _problem(shape, 3)
    sets = nb.draw_markers(shape, 16, 250, 8, seed=3)
    one = fc.stochastic_watershed_counts(grad, mask, sets, n_jobs=1)
    four = fc.stochastic_watershed_counts(grad, mask, sets, n_jobs=4)
    assert np.array_equal(one, four)


def test_own_draws_reproducible_and_independent_of_threads():
    shape = (80, 70)
    grad, mask = _problem(shape, 4)
    kw = dict(realisations=14, markers=250, min_distance=8, seed=11)
    a = fc.stochastic_watershed_counts(grad, mask, n_jobs=1, **kw)
    b = fc.stochastic_watershed_counts(grad, mask, n_jobs=4, **kw)
    c = fc.stochastic_watershed_counts(grad, mask, n_jobs=3, **kw)
    assert np.array_equal(a, b) and np.array_equal(a, c)
    other = fc.stochastic_watershed_counts(grad, mask, n_jobs=2, **{**kw, "seed": 12})
    assert not np.array_equal(a, other)
    assert a.max() <= 14 and a.max() > 0
    # the draws are numpy's: thin them here with Zeit's trim and feed them as sets
    rng = np.random.default_rng(11)
    sets = []
    for _ in range(14):
        p = rng.random((2, 250)) * np.array([[shape[0]], [shape[1]]])
        sets.append(fc.trim_close_points(p, 8))
    assert np.array_equal(fc.stochastic_watershed_counts(grad, mask, sets, n_jobs=2), a)


def test_unique_labels_and_errors():
    shape = (64, 64)
    grad, mask = _problem(shape, 6)
    got = fc.stochastic_watershed_counts(
        grad, mask, realisations=8, markers=120, min_distance=5, coloring="unique", seed=1, n_jobs=2)
    assert got.dtype == np.uint32 and got.max() <= 8 and got.max() > 0
    # every marker its own label: reference with distinct labels via skimage
    from skimage import morphology, segmentation
    sets = nb.draw_markers(shape, 5, 100, 5, seed=2)
    ref = np.zeros(shape, dtype=np.float64)
    for m in sets:
        xk, yk = m
        mk = np.zeros(shape, dtype=int)
        mk[xk.astype(int), yk.astype(int)] = np.arange(1, xk.size + 1)
        mk = morphology.dilation(mk, morphology.disk(3))
        ref += segmentation.find_boundaries(segmentation.watershed(grad, mk, mask=mask), connectivity=4)
    got = fc.stochastic_watershed_counts(grad, mask, sets, coloring="unique", n_jobs=1)
    assert np.array_equal(got, ref.astype(np.uint32))
    with pytest.raises(ValueError):
        fc.stochastic_watershed_counts(grad, mask, sets, coloring="other")
    with pytest.raises(ValueError):
        fc.stochastic_watershed_counts(grad, mask, [np.array([[1000.0], [1.0]])], n_jobs=1)
    with pytest.raises(ValueError):
        fc.stochastic_watershed_counts(grad, mask[:10], sets)


def test_same_pixel_twice_last_wins():
    shape = (40, 40)
    grad, mask = _problem(shape, 8)
    rng = np.random.default_rng(8)
    sets = [rng.random((2, 900)) * 40 for _ in range(6)]  # dense: many points share a pixel
    assert any(len(np.unique(s.astype(int), axis=1).T) < s.shape[1] for s in sets)
    ref = nb.boundary_frequency(grad, mask, markers=sets, n_neighbors=10)
    got = fc.stochastic_watershed_counts(grad, mask, sets, n_neighbors=10, n_jobs=2)
    assert np.array_equal(got, ref.astype(np.uint32))
