"""zeit._fields_core (phase 24b): the exact boundary probability of the stochastic watershed
(Kruskal + union-find) against a pure-Python evaluation of its formula, and against a Monte Carlo of
the same marker model (minimum-spanning-forest cut) that converges to it."""

import time

import numpy as np
import pytest

from zeit import _fields_core as fc


BOUND = 0.35  # Pearson bound against the skimage-style Monte Carlo, set from the measured values


def _field_image(rows, cols, seed, noise=0.05):
    """Blocks of constant value plus noise; the gradient is the max-abs difference to the neighbours."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 4, size=(4, 4)).astype(float)
    img = np.kron(blocks, np.ones((rows // 4, cols // 4)))
    img = img + rng.normal(0, noise, img.shape)
    g = np.zeros_like(img)
    dy = np.abs(np.diff(img, axis=0))
    dx = np.abs(np.diff(img, axis=1))
    g[:-1] = np.maximum(g[:-1], dy)
    g[1:] = np.maximum(g[1:], dy)
    g[:, :-1] = np.maximum(g[:, :-1], dx)
    g[:, 1:] = np.maximum(g[:, 1:], dx)
    return g


def _reference(g, mask, markers, prob, return_tree=False, replay=True):
    """Pure-Python Kruskal in the documented edge order: per-edge (h, v) values of prob(a, b, n)."""
    rows, cols = g.shape
    m = np.ones(g.shape, bool) if mask is None else mask
    n = int(m.sum())
    edges = []
    for r in range(rows):
        for c in range(cols - 1):
            if m[r, c] and m[r, c + 1]:
                edges.append((max(g[r, c], g[r, c + 1]), r * (cols - 1) + c, ("h", r, c), (r * cols + c, r * cols + c + 1)))
    nh = rows * (cols - 1)
    for r in range(rows - 1):
        for c in range(cols):
            if m[r, c] and m[r + 1, c]:
                edges.append((max(g[r, c], g[r + 1, c]), nh + r * cols + c, ("v", r, c), (r * cols + c, (r + 1) * cols + c)))
    edges.sort(key=lambda e: (e[0], e[1]))
    parent = list(range(rows * cols))
    size = [1] * (rows * cols)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h = np.zeros((rows, cols - 1))
    v = np.zeros((rows - 1, cols))
    th = np.zeros((rows, cols - 1), bool)  # True: the edge is in the spanning forest
    tv = np.zeros((rows - 1, cols), bool)
    merges = []  # (p, q, probability) in order
    for _, _, (kind, r, c), (p, q) in edges:
        a, b = find(p), find(q)
        if a == b and not replay:
            continue
        if a == b:
            # a cycle edge takes the maximum of P over the spanning-forest path from p to q (brute
            # force: depth-first search over the forest edges merged so far)
            adj = {}
            for mp, mq, mv in merges:
                adj.setdefault(mp, []).append((mq, mv))
                adj.setdefault(mq, []).append((mp, mv))
            best = {p: 0.0}
            stack = [p]
            while stack:
                x = stack.pop()
                for y, mv in adj.get(x, []):
                    if y not in best:
                        best[y] = max(best[x], mv)
                        stack.append(y)
            val = best[q]
        else:
            val = prob(size[a], size[b], n, markers)
            merges.append((p, q, val))
            (th if kind == "h" else tv)[r, c] = True
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
        (h if kind == "h" else v)[r, c] = val
    return (h, v, th, tv) if return_tree else (h, v)


def _naive(a, b, n, N):
    return 1 - (1 - a / n) ** N - (1 - b / n) ** N + (1 - (a + b) / n) ** N


def _mp(a, b, n, N):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80
    a, b, n = mp.mpf(a), mp.mpf(b), mp.mpf(n)
    return float(1 - (1 - a / n) ** N - (1 - b / n) ** N + (1 - (a + b) / n) ** N)


def _pixel_from_edges(h, v):
    rows, cols = v.shape[0] + 1, h.shape[1] + 1
    out = np.zeros((rows, cols))
    out[:, :-1] = np.maximum(out[:, :-1], h)
    out[:, 1:] = np.maximum(out[:, 1:], h)
    out[:-1] = np.maximum(out[:-1], v)
    out[1:] = np.maximum(out[1:], v)
    return out


@pytest.mark.parametrize("shape", [(3, 4), (2, 5), (1, 7), (5, 1), (6, 6)])
@pytest.mark.parametrize("N", [0, 1, 2, 3, 7.5])
def test_formula_against_pure_python(shape, N):
    rng = np.random.default_rng(shape[0] * 10 + shape[1])
    g = rng.integers(0, 4, size=shape).astype(float)  # many ties
    mask = None
    if shape == (6, 6):
        mask = rng.random(shape) < 0.8
    pixel, h, v = fc.exact_boundary_probability(g, mask, N, return_edges=True)
    rh, rv = _reference(g, mask, N, lambda a, b, n, N: _naive(a, b, n, N) if N > 0 else 0.0)
    assert h.shape == rh.shape and v.shape == rv.shape
    np.testing.assert_allclose(h, rh, atol=1e-12, rtol=0)
    np.testing.assert_allclose(v, rv, atol=1e-12, rtol=0)
    np.testing.assert_allclose(pixel, _pixel_from_edges(h, v), atol=0, rtol=0)
    if mask is not None:
        assert (pixel[~mask] == 0).all()


def test_stable_for_large_n_and_tiny_components():
    # 60 x 60, N = 1000: many edges join components of a few pixels out of 3600, where the naive
    # inclusion-exclusion loses most digits; compare with an 80-digit evaluation
    pytest.importorskip("mpmath")
    rng = np.random.default_rng(3)
    g = rng.random((60, 60))
    _, h, v = fc.exact_boundary_probability(g, None, 1000, return_edges=True)
    rh, rv, th, tv = _reference(g, None, 1000, _mp, return_tree=True, replay=False)
    for got, ref, tree in ((h, rh, th), (v, rv, tv)):  # the spanning-forest edges (cycle edges: see above)
        np.testing.assert_allclose(got[tree], ref[tree], rtol=1e-11, atol=1e-300)
    assert h.min() >= 0 and h.max() <= 1
    # a strip of 3000 pixels, N = 1e5: the first edge joins two single pixels, P ~ N(N-1)/n^2 ~ 1e-3
    # of the cancelling terms
    g3 = np.zeros((1, 3000))
    g3[0] = np.arange(3000)
    _, h3, _ = fc.exact_boundary_probability(g3, None, 1e5, return_edges=True)
    ref = _mp(1, 1, 3000, int(1e5))
    assert abs(h3[0, 0] - ref) <= 1e-12 * ref


def test_edges_outside_mask_and_nan():
    g = np.arange(12.0).reshape(3, 4)
    mask = np.ones((3, 4), bool)
    mask[:, 2] = False  # two parts
    pixel, h, v = fc.exact_boundary_probability(g, mask, 5, return_edges=True)
    assert (h[:, 1:] == 0).all() and (v[:, 2] == 0).all() and (pixel[:, 2] == 0).all()
    assert pixel[:, [0, 1, 3]].max() > 0
    g[0, 0] = np.nan
    with pytest.raises(ValueError):
        fc.exact_boundary_probability(g, None, 5)
    mask[0, 0] = False  # NaN outside the mask is ignored
    fc.exact_boundary_probability(g, mask, 5)
    with pytest.raises(ValueError):
        fc.exact_boundary_probability(g, None, -1)


def _cases():
    rng = np.random.default_rng(11)
    rand = rng.random((24, 24))
    field = _field_image(32, 32, 5)
    masked = _field_image(32, 32, 6)
    mask2 = np.ones((32, 32), bool)
    mask2[:, 14:18] = False  # two parts
    mask2[:6, :6] = False
    ties = rng.integers(0, 3, size=(24, 24)).astype(float)  # plateaus
    return {"random": (rand, None), "field": (field, None), "two parts": (masked, mask2),
            "ties": (ties, None)}


@pytest.mark.parametrize("name", ["random", "field", "two parts", "ties"])
@pytest.mark.parametrize("N", [3, 10, 40])
def test_monte_carlo_against_the_exact_probability(name, N):
    """On spanning-forest edges the cut frequency converges to P; on cycle edges P (the
    path maximum of P over the spanning-forest path between their ends) is a lower bound of it, and
    the gap is reported."""
    g, mask = _cases()[name]
    R = 20000
    _, ph, pv = fc.exact_boundary_probability(g, mask, N, return_edges=True, n_jobs=2)
    fh, fv = fc.msf_cut_frequency(g, mask, N, R, seed=100 + N, n_jobs=4)
    _, _, th, tv = _reference(g, mask, N, _naive, return_tree=True, replay=False)
    worst, tree_d = 0.0, []
    for p, f, tree in ((ph, fh, th), (pv, fv, tv)):
        sigma = np.sqrt(p * (1 - p) / R)
        tol = 5 * sigma + 1e-3
        assert (np.abs(f - p)[tree] <= tol[tree]).all(), np.abs(f - p)[tree].max()
        assert (f - p >= -tol)[~tree].all(), (f - p)[~tree].min()
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sigma > 0, np.abs(f - p) / sigma, 0)
        worst = max(worst, float(z[tree].max()))
        tree_d.append(np.abs(f - p)[tree])
    m = np.ones(g.shape, bool) if mask is None else mask
    cyc_h = ~th & (m[:, :-1] & m[:, 1:])  # cycle edges inside the mask
    cyc_v = ~tv & (m[:-1] & m[1:])
    gap = np.concatenate([(fh - ph)[cyc_h], (fv - pv)[cyc_v]])
    pix = fc.exact_boundary_probability(g, mask, N, n_jobs=1)
    fpix = _pixel_from_edges(fh, fv)
    pgap = fpix - pix
    assert pgap.min() > -0.03
    print(f"\n  [{name} N={N}] tree: max|MC-P|/sigma = {worst:.2f}, mean |MC-P| = "
          f"{np.concatenate(tree_d).mean():.2e}; cycle edges: max MC-P = {gap.max():.3f}, "
          f"mean = {gap.mean():.3f}; pixel map: max {pgap.max():.3f}, mean {pgap.mean():.3f}")


def test_independent_of_n_jobs():
    g = _field_image(32, 32, 9)
    a = fc.exact_boundary_probability(g, None, 25, return_edges=True, n_jobs=1)
    b = fc.exact_boundary_probability(g, None, 25, return_edges=True, n_jobs=4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    c = fc.msf_cut_frequency(g, None, 25, 300, seed=1, n_jobs=1)
    d = fc.msf_cut_frequency(g, None, 25, 300, seed=1, n_jobs=4)
    for x, y in zip(c, d):
        np.testing.assert_array_equal(x, y)
    e = fc.msf_cut_frequency(g, None, 25, 300, seed=2, n_jobs=1)
    assert not np.array_equal(c[0], e[0])


def test_degenerate_inputs():
    g = np.ones((3, 3))
    nomask = np.zeros((3, 3), bool)
    pixel = fc.exact_boundary_probability(g, nomask, 10)
    assert pixel.shape == (3, 3) and (pixel == 0).all()
    h, v = fc.msf_cut_frequency(g, nomask, 10, 5, seed=0)
    assert (h == 0).all() and (v == 0).all()
    h, v = fc.msf_cut_frequency(g, None, 0, 5, seed=0)
    assert (h == 0).all() and (v == 0).all()
    assert fc.exact_boundary_probability(np.ones((1, 1)), None, 4).shape == (1, 1)


def test_against_the_skimage_style_monte_carlo():
    """A sanity check, not parity: the priority-flood watershed with thick 3x3 boundaries and the
    path-maximum bound with edge maxima differ by design, but on a field-like image both map the
    same boundaries."""
    pytest.importorskip("skimage")
    g = _field_image(48, 48, 4, noise=0.03)
    for N in (20, 100):
        counts = fc.stochastic_watershed_counts(g, None, coloring="unique", min_distance=0, disk_radius=0,
                                                markers=N, realisations=1500, seed=7, n_jobs=4)
        sk = counts / 1500
        exact = fc.exact_boundary_probability(g, None, N, n_jobs=1)
        r = float(np.corrcoef(sk.ravel(), exact.ravel())[0, 1])
        fh, fv = fc.msf_cut_frequency(g, None, N, 1500, seed=1, n_jobs=4)
        mc = _pixel_from_edges(fh, fv)
        r_cut = float(np.corrcoef(sk.ravel(), mc.ravel())[0, 1])
        r_ex = float(np.corrcoef(exact.ravel(), mc.ravel())[0, 1])
        print(f"\n  N={N}: Pearson(skimage MC, exact) = {r:.3f}; (skimage MC, cut-forest MC) = {r_cut:.3f}; "
              f"(exact, cut-forest MC) = {r_ex:.3f}")
        # Measured: 0.46 / 0.46 (skimage MC vs exact, N=20 / 100), 0.48 / 0.51 (skimage MC vs cut-forest
        # MC), 0.99 / 0.97 (exact vs cut-forest MC). The exact map is a lower bound of the cut-forest
        # one and tracks it closely; both differ from the flooding by design (thick boundaries, the
        # flooding's markers also grow). The bounds leave room for the Monte Carlo noise.
        assert r > BOUND
        assert r_cut > 0.3
        assert r_ex > 0.9


def test_timing_1000x1000():
    g = np.random.default_rng(0).random((1000, 1000))
    out = []
    for jobs in (1, 8):
        t = time.perf_counter()
        fc.exact_boundary_probability(g, None, 1000, n_jobs=jobs)
        out.append(time.perf_counter() - t)
    print(f"\n  exact 1000x1000: n_jobs=1 {out[0]:.2f} s, n_jobs=8 {out[1]:.2f} s")
