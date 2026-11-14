"""The building blocks of field-boundary delineation by the stochastic watershed (phase 24a), private.

Thin wrappers of ``zeit._core.fields`` that give the same arrays as scikit-image 0.26 / scipy 1.18
(``skimage.segmentation.watershed``, ``skimage.morphology`` grey morphology, ``rescale_intensity``,
``apply_hysteresis_threshold``, ``find_boundaries``, ``label``, ``relabel_sequential``,
``remove_small_objects``) and the notebook's ``trim_close_points``. The public API (``zeit.fields``)
comes later and builds on these.
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from . import _core
from ._threads import resolve

_F = _core.fields


def _f64_2d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("expected a 2-D array")
    return np.ascontiguousarray(a, dtype=np.float64)


def _i32_2d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("expected a 2-D array")
    return np.ascontiguousarray(a, dtype=np.int32)


def _bool_2d(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("expected a 2-D array")
    return np.ascontiguousarray(a, dtype=bool)


def watershed(image, markers, connectivity: int = 1, mask=None) -> np.ndarray:
    """``skimage.segmentation.watershed(image, markers, connectivity=, mask=)`` as int32 labels."""
    img = _f64_2d(image)
    mk = _i32_2d(markers)
    if mk.shape != img.shape:
        raise ValueError("markers must have the shape of image")
    m = None
    if mask is not None:
        m = _bool_2d(mask)
        if m.shape != img.shape:
            raise ValueError("mask must have the shape of image")
    return _F.watershed(img, mk, int(connectivity), m)


def footprint_disk(radius: int) -> np.ndarray:
    """``skimage.morphology.disk(radius)`` as a boolean array."""
    r = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(r, r)
    return (x**2 + y**2) <= radius**2


def footprint_square(size: int) -> np.ndarray:
    """``skimage.morphology.footprint_rectangle((size, size))`` (``size`` odd) as a boolean array."""
    return np.ones((size, size), dtype=bool)


#: The default footprint of scikit-image's grey morphology: the 3x3 cross.
FOOTPRINT_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _morph(op: str, image, footprint, mode: str, cval: float) -> np.ndarray:
    image = np.asarray(image)
    fp = FOOTPRINT_CROSS if footprint is None else np.ascontiguousarray(footprint, dtype=bool)
    out = _F.grey_morphology(op, _f64_2d(image), fp, mode, float(cval))
    return out.astype(image.dtype, copy=False)


def dilation(image, footprint=None, *, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """``skimage.morphology.dilation`` (footprint of odd size, default the 3x3 cross), image dtype kept."""
    return _morph("dilation", image, footprint, mode, cval)


def erosion(image, footprint=None, *, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """``skimage.morphology.erosion``."""
    return _morph("erosion", image, footprint, mode, cval)


def opening(image, footprint=None, *, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """``skimage.morphology.opening``."""
    return _morph("opening", image, footprint, mode, cval)


def closing(image, footprint=None, *, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """``skimage.morphology.closing``."""
    return _morph("closing", image, footprint, mode, cval)


def white_tophat(image, footprint=None, *, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """``skimage.morphology.white_tophat``."""
    return _morph("white_tophat", image, footprint, mode, cval)


def black_tophat(image, footprint=None, *, mode: str = "reflect", cval: float = 0.0) -> np.ndarray:
    """``skimage.morphology.black_tophat``."""
    return _morph("black_tophat", image, footprint, mode, cval)


def percentiles(values, q: Sequence[float]) -> np.ndarray:
    """``np.percentile(values, q)`` for a sequence ``q`` (linear), float64 (a float32 input gets numpy's
    float32 neighbour difference)."""
    values = np.asarray(values)
    if values.dtype == np.float32:
        return np.asarray(_F.percentiles_f32(np.ascontiguousarray(values.ravel()), [float(x) for x in q]))
    v = np.ascontiguousarray(np.asarray(values).ravel(), dtype=np.float64)
    return np.asarray(_F.percentiles(v, [float(x) for x in q]))


def _dtype_range(dtype: np.dtype) -> Tuple[float, float]:
    if dtype.kind == "f":
        return -1.0, 1.0
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    raise TypeError(f"unsupported dtype {dtype}")


def rescale_intensity(image, in_range: Tuple[float, float], out_range: Optional[Tuple[float, float]] = None):
    """``skimage.exposure.rescale_intensity(image, in_range=, out_range=)``, ``out_range=None``
    being its ``"dtype"`` (the dtype's range, from 0 when ``in_range`` starts at or above 0)."""
    image = np.asarray(image)
    imin, imax = float(in_range[0]), float(in_range[1])
    if out_range is None:
        omin, omax = _dtype_range(image.dtype)
        if imin >= 0:
            omin = 0.0
    else:
        omin, omax = float(out_range[0]), float(out_range[1])
    flat = image.ravel()
    if image.dtype == np.float32:
        out = _F.rescale_intensity_f32(np.ascontiguousarray(flat), imin, imax, omin, omax)
        return out.reshape(image.shape)
    out = _F.rescale_intensity(np.ascontiguousarray(flat, dtype=np.float64), imin, imax, omin, omax)
    out = out.reshape(image.shape)
    if out_range is None and image.dtype.kind in "iu":
        return out.astype(image.dtype)
    return out


def rescale_intensity_percentile(image, low: float = 2, high: float = 98, out_range=None):
    """``rescale_intensity(image, in_range=tuple(np.percentile(image, (low, high))))``."""
    p = percentiles(image, (low, high))
    return rescale_intensity(image, (float(p[0]), float(p[1])), out_range)


def hysteresis_threshold(image, low: float, high: float) -> np.ndarray:
    """``skimage.filters.apply_hysteresis_threshold``."""
    return _F.hysteresis_threshold(_f64_2d(image), float(low), float(high))


def find_boundaries(labels, connectivity: int = 1) -> np.ndarray:
    """``skimage.segmentation.find_boundaries(labels, connectivity, mode="thick")``; the connectivity is
    clamped to 2 (so 4, as the notebook passes, is the 8-neighbourhood)."""
    return _F.find_boundaries(_i32_2d(labels), int(connectivity))


def label(image, connectivity: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """``skimage.measure.label(image, connectivity=, return_num=True)`` (background 0): a boolean image
    gives its components, an integer one the components of equal value; connectivity None is the full
    one (2)."""
    image = np.asarray(image)
    conn = 2 if connectivity is None else int(connectivity)
    if image.dtype == bool:
        return _F.label_ints(_i32_2d(image), conn, False)
    return _F.label_ints(_i32_2d(image), conn, True)


def ndi_label(image, connectivity: int = 1) -> Tuple[np.ndarray, int]:
    """``scipy.ndimage.label(image, generate_binary_structure(2, connectivity))``: the components of
    the non-zero values (scipy's default is 4-connected)."""
    return _F.label_ints(_i32_2d(np.asarray(image) != 0), int(connectivity), False)


def relabel_sequential(labels, offset: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """``skimage.segmentation.relabel_sequential``: ``(relabelled int32, inverse)`` where
    ``inverse[k]`` is the old value of label ``offset + k`` (scikit-image's inverse map starts with
    the 0 -> 0 entry, this one does not)."""
    out, inv = _F.relabel_sequential(_i32_2d(labels), int(offset))
    return out, np.asarray(inv, dtype=np.int32)


def remove_small_objects(ar, min_size: Optional[int] = None, *, max_size: Optional[int] = None,
                         connectivity: int = 1) -> np.ndarray:
    """``skimage.morphology.remove_small_objects``: with ``max_size`` (scikit-image 0.26) the objects of
    that many pixels or fewer are removed, with ``min_size`` (the older) those smaller than it. A boolean
    ``ar`` is taken by components (``connectivity``), an integer one by label value."""
    if (min_size is None) == (max_size is None):
        raise ValueError("give either min_size or max_size")
    if max_size is None:
        max_size = max(int(min_size) - 1, 0)
    ar = np.asarray(ar)
    if ar.dtype == bool:
        return _F.remove_small_objects_mask(np.ascontiguousarray(ar), int(max_size), int(connectivity))
    if ar.dtype.kind not in "iu":
        raise TypeError("only integer or boolean arrays are supported")
    return _F.remove_small_labels(_i32_2d(ar), int(max_size)).astype(ar.dtype, copy=False)


def close_pairs(points, distance: float) -> np.ndarray:
    """The pairs ``(i, j)``, i < j, of ``points`` (2, n) within ``distance`` (like ``KDTree.query_pairs``),
    sorted by (i, j), as an array (m, 2)."""
    x, y = (np.ascontiguousarray(p, dtype=np.float64) for p in points)
    return np.asarray(_F.close_pairs(x, y, float(distance)), dtype=np.int64).reshape(-1, 2)


def trim_close_points(points, distance: float, pairs=None) -> np.ndarray:
    """The notebook's ``trim_close_points``: the points (2, n) kept by the greedy thinning so that all are
    farther than ``distance`` from each other, as the array (2, k). The kept set depends on the order
    of the close pairs the greedy loop walks: with ``pairs`` (a sequence of ``(i, j)``, i < j, e.g.
    ``list(KDTree(points.T).query_pairs(distance))``, the notebook's order) it follows them, without it
    the pairs are sorted by (i, j)."""
    x = np.ascontiguousarray(points[0], dtype=np.float64)
    y = np.ascontiguousarray(points[1], dtype=np.float64)
    if pairs is None:
        keep = _F.trim_close_points(x, y, float(distance))
    else:
        keep = _F.trim_close_points_pairs(len(x), [(int(i), int(j)) for i, j in pairs])
    keep = np.asarray(keep, dtype=np.intp)
    return np.array([np.asarray(points[0])[keep], np.asarray(points[1])[keep]])


def knn_graph(points, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """The k nearest other points of each point of ``points`` (2, n) as CSR ``(indptr, indices)``, the
    neighbour sets of ``sklearn.neighbors.kneighbors_graph(points.T, k)`` (each row nearest first).
    A ``k`` above n - 1 takes the n - 1 others (scikit-learn raises)."""
    x = np.ascontiguousarray(points[0], dtype=np.float64)
    y = np.ascontiguousarray(points[1], dtype=np.float64)
    indptr, indices = _F.knn_graph(x, y, int(k))
    return indptr.astype(np.int64), indices.astype(np.int64)


def vertex_coloring_mis(indptr, indices) -> np.ndarray:
    """``pyamg.graph.vertex_coloring(G, method="MIS")`` of the CSR graph ``(indptr, indices)``: the same
    colours (from 0), the graph read as given (not symmetrised, the diagonal ignored)."""
    ip = np.asarray(indptr)
    ix = np.asarray(indices)
    if ip.ndim != 1 or ix.ndim != 1 or len(ip) < 1:
        raise ValueError("indptr and indices must be 1-D, indptr with n + 1 entries")
    if len(ip) and (ip.min() < 0 or (len(ix) and ix.min() < 0)):
        raise ValueError("indptr and indices must not be negative")
    return _F.vertex_coloring_mis(ip.astype(np.uint32), ix.astype(np.uint32))


def stochastic_watershed_counts(gradient, mask=None, marker_sets=None, *, realisations: int = 500,
                                markers: int = 1000, min_distance: float = 20, coloring: str = "knn",
                                n_neighbors: int = 20, disk_radius: int = 3, seed=None,
                                n_jobs: Optional[int] = None) -> np.ndarray:
    """The Monte Carlo of the stochastic watershed (the MappingDemo notebook's loop): for how many of
    the realisations each pixel of ``gradient`` is on a boundary, as uint32 counts.

    Each realisation colours its marker points (``coloring="knn"``: pyamg's colouring of their
    ``n_neighbors``-nearest-neighbour graph, as the notebook; ``"unique"``: every point its own label),
    writes the labels at ``int(x), int(y)``, dilates them by a disk of ``disk_radius``, floods
    ``gradient`` inside ``mask`` (``None``: everywhere) and counts ``find_boundaries(connectivity=4)``.

    ``marker_sets``: a list of (2, k) float arrays (x = row, y = col), the notebook's convention, used
    as they are (no thinning); ``realisations``, ``markers``, ``min_distance`` and ``seed`` are then
    ignored. Without it each of ``realisations`` sets is ``markers`` uniform points drawn from
    ``numpy.random.default_rng(seed)`` (all up front, in order), thinned to ``min_distance`` (0: no
    thinning), so the result depends on ``seed`` and not on ``n_jobs``.
    """
    grad = _f64_2d(gradient)
    m = None
    if mask is not None:
        m = _bool_2d(mask)
        if m.shape != grad.shape:
            raise ValueError("mask must have the shape of gradient")
    if coloring not in ("knn", "unique"):
        raise ValueError("coloring must be 'knn' or 'unique'")
    if marker_sets is not None:
        sets = [np.ascontiguousarray(s, dtype=np.float64) for s in marker_sets]
        if any(s.ndim != 2 or s.shape[0] != 2 for s in sets):
            raise ValueError("each marker set must have shape (2, k)")
        distance = 0.0
    else:
        if realisations < 0 or markers < 0:
            raise ValueError("realisations and markers must not be negative")
        rng = np.random.default_rng(seed)
        scale = np.array([grad.shape[0], grad.shape[1]], dtype=np.float64)[:, None]
        sets = [rng.random((2, int(markers))) * scale for _ in range(int(realisations))]
        distance = float(min_distance)
    return _F.stochastic_counts(grad, m, sets, distance, coloring, int(n_neighbors), int(disk_radius),
                                resolve(n_jobs))


def exact_boundary_probability(gradient, mask=None, markers: float = 1000, *, return_edges: bool = False,
                               n_jobs: Optional[int] = None):
    """The exact boundary probability of the stochastic watershed with ``markers`` uniform one-pixel
    markers (drawn with replacement among the mask pixels): float64 (y, x), the maximum over the
    4-neighbour edges touching each pixel of the probability that the minimum-spanning-forest cut
    separates the edge's two sides (0 outside ``mask``). The graph weights an edge ``max(g[p], g[q])``;
    NaN in ``gradient`` inside ``mask`` is an error. With ``return_edges`` it also gives the per-edge
    probabilities ``(h, v)``: ``h`` (y, x - 1) between ``(r, c)`` and ``(r, c + 1)``, ``v`` (y - 1, x)
    between ``(r, c)`` and ``(r + 1, c)`` (0 for edges not inside the mask)."""
    grad = _f64_2d(gradient)
    m = None
    if mask is not None:
        m = _bool_2d(mask)
        if m.shape != grad.shape:
            raise ValueError("mask must have the shape of gradient")
    pixel, h, v = _F.exact_boundary_probability(grad, m, float(markers), resolve(n_jobs))
    return (pixel, h, v) if return_edges else pixel


def msf_cut_frequency(gradient, mask=None, markers: int = 1000, realisations: int = 1000, seed=None,
                      n_jobs: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo of the model of :func:`exact_boundary_probability`: ``realisations`` draws of
    ``markers`` uniform pixels among the mask pixels (with replacement, all up front, in order, from
    ``numpy.random.default_rng(seed)``), each cutting the minimum spanning forest where both Kruskal
    components hold a marker. Returns the per-edge cut frequencies ``(h, v)`` (shapes as there), an
    unbiased estimate of the exact per-edge probabilities; the result does not depend on ``n_jobs``."""
    grad = _f64_2d(gradient)
    m = None
    if mask is not None:
        m = _bool_2d(mask)
        if m.shape != grad.shape:
            raise ValueError("mask must have the shape of gradient")
    if realisations < 0 or markers < 0:
        raise ValueError("realisations and markers must not be negative")
    realisations, markers = int(realisations), int(markers)
    inside = np.arange(grad.size, dtype=np.uint32) if m is None else np.flatnonzero(m.ravel()).astype(np.uint32)
    rng = np.random.default_rng(seed)
    if len(inside) == 0:
        draws = np.zeros((realisations, 0), dtype=np.uint32)
    else:
        draws = np.empty((realisations, markers), dtype=np.uint32)
        for i in range(realisations):
            draws[i] = inside[rng.integers(0, len(inside), size=markers)]
    h, v = _F.msf_cut_counts(grad, draws, m, resolve(n_jobs))
    scale = 1.0 / realisations if realisations else 0.0
    return h * scale, v * scale
