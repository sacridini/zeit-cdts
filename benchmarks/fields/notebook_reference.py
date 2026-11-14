"""Stochastic-watershed field boundaries: the MappingDemo notebook as functions.

Source: the notebook "MappingDemo" (Baumann, August 2020; inspired by Emmanuelle
Gouillart's segmentation tutorial); the author authorised porting the code.
``notebook_original.py`` keeps its code cells verbatim; the cell numbers quoted
in the docstrings below refer to the ``# %% [cell N]`` markers in that file.

This module keeps the notebook's operations, order and default values. The
only changes are API names for the installed library versions (see "Version
notes") and that every random draw can be injected, so that another
implementation (Zeit's port) can be compared with the very same markers:
``draw_markers`` makes the per-realisation point arrays from a seed exactly as
the notebook would, ``save_markers``/``load_markers`` store them in an npz.

Arrays follow the notebook's conventions: images are (rows, cols); a marker
set is a float array of shape (2, k) holding (x, y) = (row, col) coordinates
(x is scaled by shape[0], y by shape[1], and the pixel is ``astype(int)``).

Version notes (behaviour that depends on the library versions)
--------------------------------------------------------------
* skimage 0.26 deprecated ``remove_small_objects(min_size=)`` for ``max_size=``
  (objects of size <= max_size are removed, so min_size=50 is max_size=49).
  ``_remove_small_objects`` picks the keyword; the result is the same.
* skimage ``dilation``/``find_boundaries`` take ``footprint`` (older versions:
  ``selem``); the notebook passes it positionally, which works in both.
* ``find_boundaries(seg, connectivity=4)`` on a 2-D image: the structuring
  element is ``generate_binary_structure(2, 4)``, and scipy clips the
  connectivity to the rank, so it is the full 3 x 3 square (the same as
  connectivity=2, 8-connected), not a 4-neighbour cross. The mode is the
  default 'thick'.
* ``rescale_intensity(..., out_range=(0, 1))`` returns float64 in skimage 0.26
  for uint16 input (checked); ``ndimage.black_tophat(x, 3)`` is a flat 3 x 3
  square footprint (size=3), not a disk. In both, the dtype of the input
  matters: ``gradient`` must get the stretched float image.
* ``morphology.dilation(img)`` with no footprint is the 3 x 3 cross
  (connectivity 1) in the versions checked.
* ``np.int`` (cell 12) was removed in numpy 1.24: ``int`` is used.
* pyamg ``vertex_coloring``: the notebook calls it with the default
  ``method='MIS'`` (checked with pyamg 5.4.1); so does this module.
* scipy ``KDTree.query_pairs`` returns a set of tuples; ``trim_close_points``
  iterates it, so the thinning depends on that iteration order (stable for a
  given scipy, not specified).
* The notebook draws from the global ``np.random`` state; here a
  ``numpy.random.RandomState(seed)`` gives the same stream as
  ``np.random.seed(seed)``. The two random relabelings (cells 15 and 17,
  ``np.random.permutation``) only shuffle label ids; they are off by default
  (``rng=None``) so that outputs are comparable.
"""

import inspect

import numpy as np
from scipy import ndimage, spatial
from skimage import filters, measure, morphology, segmentation
from skimage.exposure import rescale_intensity
from sklearn.neighbors import kneighbors_graph

try:
    from pyamg.graph import vertex_coloring
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "notebook_reference needs pyamg (pyamg.graph.vertex_coloring, the "
        "notebook's colouring); install it in this environment"
    ) from exc


# --------------------------------------------------------------------------
# Step I, variation 2 (cell 8): stretch the histogram
# --------------------------------------------------------------------------
def stretch(img):
    """Stretch between the 2nd and 98th percentiles to [0, 1] (cell 8, StretchHisto)."""
    p2, p98 = np.percentile(img, (2, 98))
    return rescale_intensity(img, in_range=(p2, p98), out_range=(0, 1))


# --------------------------------------------------------------------------
# Step II (cell 9): gradient image
# --------------------------------------------------------------------------
def gradient(img):
    """Gradient of a stretched image (cell 9).

    ``black_tophat(img, 3) - 0.5 * img``, then a grey dilation with the
    default (cross) footprint "to remove some holes". ``img`` is the output of
    ``stretch``.
    """
    hat = ndimage.black_tophat(img, 3)
    hat_lc = hat - 0.5 * img
    return morphology.dilation(hat_lc)


# --------------------------------------------------------------------------
# Step III (cell 10): marker thinning, copied verbatim
# --------------------------------------------------------------------------
def trim_close_points(points, distance):  # =10):
    """
    Greedy method to remove some points so that
    all points are separated by a distance greater
    than ``distance``.

    points : array of shape (2, n_points)
        Coordinates of 2-D points

    distance : float
        Minimal distance between points
    """
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
    return np.array([x[keep_indices], y[keep_indices]])


# --------------------------------------------------------------------------
# Step IV (cell 12): random markers
# --------------------------------------------------------------------------
def random_markers(shape, n, min_distance, rng):
    """One realisation's markers: ``n`` uniform points thinned to ``min_distance``.

    From the loop of cell 12: ``x, y = np.random.random((2, n_markers))``,
    ``x *= shape[0]``, ``y *= shape[1]``, then ``trim_close_points((x, y), 20)``.
    ``rng`` is a ``numpy.random.RandomState`` (or anything with ``random``).
    Returns a float array of shape (2, k).
    """
    x, y = rng.random((2, n))
    x *= shape[0]
    y *= shape[1]
    return trim_close_points((x, y), min_distance)


def draw_markers(shape, realisations=500, markers=1000, min_distance=20, seed=0):
    """The marker sets of ``realisations`` consecutive loop iterations (cell 12).

    Same draws as ``np.random.seed(seed)`` followed by the notebook's loop.
    Returns a list of (2, k) float arrays.
    """
    rng = np.random.RandomState(seed)
    return [random_markers(shape, markers, min_distance, rng) for _ in range(realisations)]


def save_markers(path, marker_sets):
    """Save a list of (2, k) marker arrays to an npz (concatenated + offsets)."""
    sizes = np.array([m.shape[1] for m in marker_sets], dtype=np.int64)
    points = np.concatenate([np.asarray(m, dtype=np.float64) for m in marker_sets], axis=1)
    np.savez(path, points=points, sizes=sizes)


def load_markers(path):
    """Inverse of ``save_markers``: a list of (2, k) float arrays."""
    with np.load(path) as z:
        points, sizes = z["points"], z["sizes"]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return [points[:, offsets[i]:offsets[i + 1]] for i in range(len(sizes))]


# --------------------------------------------------------------------------
# Step IV (cell 12): one realisation
# --------------------------------------------------------------------------
def realisation(gradient, markers, mask, n_neighbors=20, disk_radius=3):
    """One watershed realisation on a gradient image (cell 12, loop body).

    ``markers`` is a (2, k) array of (x, y) points. The points are coloured
    with ``vertex_coloring`` on their ``n_neighbors``-nearest-neighbour graph,
    written into an int image as colour + 1, dilated by ``disk(disk_radius)``
    (grey dilation, so the larger colour wins where discs overlap) and used as
    watershed markers inside ``mask`` (nonzero = valid; the notebook passes
    ``np.where(arr <= 2000, 0, 1)``). Returns the label image.
    """
    xk, yk = markers
    mat = kneighbors_graph(np.array([xk, yk]).T, n_neighbors)
    colors = vertex_coloring(mat)
    markers_rw = np.zeros(gradient.shape, dtype=int)
    markers_rw[xk.astype(int), yk.astype(int)] = colors + 1
    markers_rw = morphology.dilation(markers_rw, morphology.disk(disk_radius))
    return segmentation.watershed(gradient, markers_rw, mask=mask)


# --------------------------------------------------------------------------
# Step V (cell 13): count the segmentations' boundaries
# --------------------------------------------------------------------------
def boundary_frequency(gradient, mask, realisations=500, markers=1000, seed=0,
                       return_markers=False, min_distance=20, n_neighbors=20,
                       disk_radius=3):
    """Number of realisations in which each pixel is on a boundary (cells 12-13).

    ``markers`` is either the number of uniform points per realisation (drawn
    with ``draw_markers`` from ``seed``) or an explicit list of (2, k) arrays,
    one per realisation (then ``realisations`` is taken from its length and
    ``seed`` is unused). Returns the raw counts as the notebook's
    ``boundaries_2000`` (float array, 0..realisations): ``hysteresis_regions``
    needs the same ``realisations``. With ``return_markers`` it returns
    ``(counts, marker_sets)``.
    """
    if isinstance(markers, (int, np.integer)):
        marker_sets = draw_markers(gradient.shape, realisations, markers, min_distance, seed)
    else:
        marker_sets = list(markers)
    counts = np.zeros_like(gradient)
    for m in marker_sets:
        seg = realisation(gradient, m, mask, n_neighbors, disk_radius)
        counts += segmentation.find_boundaries(seg, connectivity=4)
    if return_markers:
        return counts, marker_sets
    return counts


# --------------------------------------------------------------------------
# Step VI (cell 15): hysteresis, labels of the complement
# --------------------------------------------------------------------------
def _remove_small_objects(labels, min_size):
    """``remove_small_objects`` with ``min_size`` semantics in any skimage."""
    if "max_size" in inspect.signature(morphology.remove_small_objects).parameters:
        return morphology.remove_small_objects(labels, max_size=min_size - 1)
    return morphology.remove_small_objects(labels, min_size=min_size)


def hysteresis_regions(freq, low=0.3, high=0.6, min_size=50, realisations=500, rng=None):
    """Hysteresis on the boundary counts and labels of the non-boundary (cell 15).

    ``freq`` is the output of ``boundary_frequency`` (counts). Thresholds are
    ``low * realisations`` and ``high * realisations`` (skimage's
    ``apply_hysteresis_threshold``: strictly greater). The complement of the
    hysteresis mask is labelled with 4-connectivity (``label(..., background=0,
    connectivity=1)``); then ``remove_small_objects(regions + 1, min_size)``:
    the +1 makes the label-0 pixels (the boundaries) a label of their own, so
    the whole boundary network is one object that is kept when large, and
    removed objects become 0. Then ``relabel_sequential``. If ``rng`` (a
    RandomState) is given the labels are shuffled as in the notebook.
    Returns ``(hyst, regions_clean)``.
    """
    low_t = low * realisations
    high_t = high * realisations
    hyst = filters.apply_hysteresis_threshold(freq, low_t, high_t)
    regions = measure.label(np.logical_not(hyst), background=0, connectivity=1)
    regions_clean = _remove_small_objects(regions + 1, min_size)
    regions_clean, _, _ = segmentation.relabel_sequential(regions_clean)
    if rng is not None:
        regions_clean = rng.permutation(regions_clean.max() + 1)[regions_clean]
    return hyst, regions_clean


# --------------------------------------------------------------------------
# Step VII (cell 17): final watershed on the original image
# --------------------------------------------------------------------------
def final_watershed(img, regions, mask, rng=None):
    """Final watershed on the original (unstretched) image (cell 17).

    ``watershed(img, regions)`` with no mask (the notebook has the mask
    commented out), then pixels outside ``mask`` (``arr <= 2000``) are set to
    0. With ``rng`` the labels are shuffled as in the notebook. ``mask`` is
    True where the pixel is kept (``img > threshold``).
    """
    final = segmentation.watershed(img, regions)
    if rng is not None:
        final = rng.permutation(final.max() + 1)[final]
    return np.where(mask, final, 0)


# --------------------------------------------------------------------------
# Whole method
# --------------------------------------------------------------------------
def pipeline(img, mask_threshold=2000, realisations=500, markers=1000, min_distance=20,
             low=0.3, high=0.6, min_size=50, seed=0, n_neighbors=20, disk_radius=3,
             return_all=False):
    """Cells 8-17 for one band image (the B5 medoid, raw digital numbers).

    ``markers`` is an int or a list of (2, k) arrays (see ``boundary_frequency``).
    Mask: ``img > mask_threshold`` in the marker watersheds and for zeroing the
    final labels. Returns the final label image, or with ``return_all`` a dict
    of ``stretched, gradient, counts, markers, hyst, regions, final``.
    """
    stretched = stretch(img)
    grad = gradient(stretched)
    mask = np.where(img <= mask_threshold, 0, 1)
    counts, marker_sets = boundary_frequency(
        grad, mask, realisations, markers, seed, True, min_distance, n_neighbors, disk_radius)
    hyst, regions = hysteresis_regions(
        counts, low, high, min_size, realisations=len(marker_sets))
    final = final_watershed(img, regions, img > mask_threshold)
    if return_all:
        return dict(stretched=stretched, gradient=grad, counts=counts, markers=marker_sets,
                    hyst=hyst, regions=regions, final=final)
    return final
