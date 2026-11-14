"""``zeit.fields``: field boundaries by the stochastic watershed (Angulo & Jeulin 2007), georeferenced.

The method is the MappingDemo notebook's (Baumann 2020, after E. Gouillart's segmentation tutorial):
a gradient image, many watersheds from random markers, the frequency with which each pixel falls on a
boundary, a hysteresis threshold on that frequency, the labels of the complement, and a final watershed
that grows those labels. The building blocks (``zeit._fields_core``) are equal value for value to
scikit-image / scipy; this module only orchestrates them and keeps the georeferencing.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from . import _fields_core as _fc

#: Sobel's smoothing and edge weights, as ``skimage.filters.sobel`` (normalised: weights sum to 1).
_SOBEL_EDGE = np.array([1.0, 0.0, -1.0])
_SOBEL_SMOOTH = np.array([1.0, 2.0, 1.0]) / 4

#: The notebook's values (cells 8-17): the defaults of ``fields`` (``preset="notebook"`` adds final and the mask).
_NOTEBOOK = dict(
    gradient="tophat", method="montecarlo", realizations=500, markers=1000, min_distance=20,
    marker_radius=3, n_neighbors=20, coloring="knn", low=0.3, high=0.6, min_size=50, final="image",
)
_NOTEBOOK_MASK_THRESHOLD = 2000


# --------------------------------------------------------------------------------------
# Inputs and outputs
# --------------------------------------------------------------------------------------
def _as_cube(data: Any) -> xr.DataArray:
    """``data`` as a DataArray with ``y`` and ``x`` dims (arrays: 2-D (y, x), 3-D (band, y, x), 4-D
    (time, band, y, x))."""
    if isinstance(data, xr.DataArray):
        cube = data.compute()
    elif isinstance(data, np.ndarray):
        dims = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}.get(data.ndim)
        if dims is None:
            raise ValueError(f"expected a 2-D to 4-D array, got shape {data.shape}")
        cube = xr.DataArray(data, dims=dims)
    else:
        from ._load import load_raster

        cube = load_raster(data)
    if "y" not in cube.dims or "x" not in cube.dims:
        raise ValueError(f"expected y and x dims, got {cube.dims}")
    return cube


def _select(cube: xr.DataArray, band: Any, time: Any) -> xr.DataArray:
    """Select ``band`` / ``time`` by label (a position when the dim has no coordinate)."""
    for name, value in (("band", band), ("time", time)):
        if value is None:
            continue
        if name not in cube.dims:
            raise ValueError(f"{name}= given but the data has no {name!r} dim (dims: {cube.dims})")
        try:
            cube = cube.sel({name: value})
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"{name}={value!r} not found in the data's {name!r} coordinate") from exc
    return cube


def _layers(cube: xr.DataArray) -> Tuple[xr.DataArray, List[str]]:
    """The cube with its length-1 non-spatial dims squeezed, and the remaining feature dims."""
    extra = [d for d in cube.dims if d not in ("y", "x") and cube.sizes[d] == 1]
    if extra:
        cube = cube.squeeze(extra, drop=True)
    features = [d for d in cube.dims if d not in ("y", "x")]
    return cube.transpose(*features, "y", "x"), features


def _image2d(cube: xr.DataArray, what: str) -> np.ndarray:
    cube, features = _layers(cube)
    if features:
        raise ValueError(
            f"{what} needs one 2-D image, got dims {tuple(cube.dims)}: select a layer with band= and/or time= "
            "(or use gradient='multitemporal')")
    values = np.asarray(cube.values)
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{what} needs a numeric image, got {values.dtype}")
    return values


def _spatial_coords(template: Optional[xr.DataArray]) -> dict:
    """The coordinates of ``template`` that live on (y, x), plus the CRS scalar (``spatial_ref``)."""
    if template is None:
        return {}
    out = {}
    for name, c in template.coords.items():
        if (c.dims and set(c.dims) <= {"y", "x"}) or name == "spatial_ref":
            out[name] = c
    return out


def _wrap(values: np.ndarray, template: Optional[xr.DataArray], name: Optional[str] = None,
          attrs: Optional[dict] = None) -> xr.DataArray:
    return xr.DataArray(values, dims=("y", "x"), coords=_spatial_coords(template), name=name,
                        attrs=attrs or {})


def _template(*candidates: Any) -> Optional[xr.DataArray]:
    """The first DataArray with y/x coordinates among ``candidates`` (or None)."""
    for c in candidates:
        if isinstance(c, xr.DataArray) and "y" in c.coords and "x" in c.coords:
            return c
    return None


def _array2d(a: Any, what: str, dtype=None) -> np.ndarray:
    if isinstance(a, xr.DataArray):
        a = a.compute()
        if "y" in a.dims and "x" in a.dims:
            a = a.transpose(..., "y", "x")
        a = a.values
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{what} must be 2-D, got shape {a.shape}")
    return a if dtype is None else a.astype(dtype, copy=False)


# --------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------
def _stretch(img: np.ndarray) -> np.ndarray:
    """The 2nd / 98th percentile stretch to [0, 1] (the notebook's cell 8)."""
    if img.dtype.kind == "f" and not np.isfinite(img).all():
        raise ValueError("the image has NaN or infinite values: mask or fill them first")
    return _fc.rescale_intensity_percentile(img, 2, 98, out_range=(0, 1))


def _sobel(img: np.ndarray) -> np.ndarray:
    """``skimage.filters.sobel`` (reflect border, magnitude / sqrt(2)); NaN spreads to the neighbours."""
    from scipy import ndimage as ndi

    img = np.asarray(img)
    if img.dtype.kind != "f":
        img = img.astype(np.float64)
    kernels = (
        _SOBEL_EDGE.reshape(3, 1) * _SOBEL_SMOOTH.reshape(1, 3),
        _SOBEL_EDGE.reshape(1, 3) * _SOBEL_SMOOTH.reshape(3, 1),
    )
    out = np.zeros(img.shape, dtype=img.dtype)
    for k in kernels:
        c = ndi.convolve(img, k, mode="reflect")
        c *= c
        out += c
    return np.sqrt(out) / np.sqrt(2.0, dtype=out.dtype)


def _tophat_gradient(stretched: np.ndarray) -> np.ndarray:
    """The notebook's cell 9: ``black_tophat(img, 3x3 square) - 0.5 * img``, then a grey dilation."""
    hat = _fc.black_tophat(stretched, _fc.footprint_square(3))
    return _fc.dilation(hat - 0.5 * stretched)


def _multitemporal(cube: xr.DataArray, combine: str) -> np.ndarray:
    if combine not in ("max", "sum"):
        raise ValueError("combine must be 'max' or 'sum'")
    cube, features = _layers(cube)
    h, w = cube.sizes["y"], cube.sizes["x"]
    layers = np.asarray(cube.values, dtype=np.float64).reshape(-1, h, w)
    if not features:  # one image: a single layer
        layers = layers.reshape(1, h, w)
    acc = None
    n_ok = np.zeros((h, w), dtype=np.int64)
    for layer in layers:
        finite = np.isfinite(layer)
        if not finite.any():
            continue
        z = np.where(finite, layer, np.nan)
        mean, std = np.nanmean(z), np.nanstd(z)
        z = (z - mean) / std if std > 0 else np.where(finite, 0.0, np.nan)
        g = _sobel(z)
        ok = np.isfinite(g)
        n_ok += ok
        g = np.where(ok, g, -np.inf if combine == "max" else 0.0)
        if acc is None:
            acc = g
        elif combine == "max":
            acc = np.maximum(acc, g)
        else:
            acc = acc + g
    if acc is None:
        return np.full((h, w), np.nan)
    if combine == "sum":
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = acc / n_ok
    return np.where(n_ok > 0, acc, np.nan)


def _gradient_of(cube: xr.DataArray, method: str, combine: str) -> np.ndarray:
    if method == "tophat":
        return _tophat_gradient(_stretch(_image2d(cube, "gradient='tophat'")))
    if method == "sobel":
        return _sobel(_stretch(_image2d(cube, "gradient='sobel'")))
    if method == "multitemporal":
        return _multitemporal(cube, combine)
    raise ValueError(f"unknown gradient method {method!r}: use 'tophat', 'sobel' or 'multitemporal'")


def edge_gradient(
    data: Any,
    method: str = "tophat",
    *,
    band: Any = None,
    time: Any = None,
    combine: str = "max",
    n_jobs: Optional[int] = None,
) -> xr.DataArray:
    """An edge-strength image to flood: the gradient the stochastic watershed of ``zeit.fields`` runs on.

    Parameters
    ----------
    data
        A map ``(y, x)``, a stack ``(band, y, x)``, a cube ``(time, y, x)`` or ``(time, band, y, x)``
        (a DataArray, a NumPy array, or anything ``load_raster`` reads). Dims of length 1 are squeezed.
    method
        ``"tophat"``: the MappingDemo notebook's (Baumann 2020): the image stretched between its 2nd and
        98th percentiles to [0, 1], ``black_tophat(img, 3x3 square) - 0.5 * img``, then a grey dilation
        with the 3x3 cross. ``"sobel"``: the Sobel magnitude of the stretched image (equal to
        ``skimage.filters.sobel``: normalised kernels, reflected border, divided by sqrt(2)).
        ``"multitemporal"``: every layer (each time and band) is standardised (z-score, NaN-aware) and
        Sobel'd, and the layers are combined by ``combine``.
        ``"tophat"`` and ``"sobel"`` need one 2-D image (and finite values).
    band, time
        Select one band and/or one date by label (a position where the dim has no coordinate), so
        that a cube becomes the 2-D image ``"tophat"`` and ``"sobel"`` need.
    combine
        For ``"multitemporal"``: ``"max"`` (the strongest edge of any layer) or ``"sum"``, the sum
        divided by the number of layers with a value at the pixel, that is, the mean.
    n_jobs
        Accepted for symmetry with the other functions; the operations here are single-threaded.

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` float, with the coordinates and CRS of the input.

    Examples
    --------
    >>> swir = zeit.load_raster("B5_medoid_2019.tif")                       # (y, x)
    >>> grad = zeit.edge_gradient(swir)                                       # the notebook's
    >>> grad = zeit.edge_gradient(cube, "multitemporal", combine="max")      # (time, band, y, x)
    """
    cube = _select(_as_cube(data), band, time)
    grad = _gradient_of(cube, method, combine)
    return _wrap(grad, _template(data if isinstance(data, xr.DataArray) else None, cube), "gradient",
                 {"long_name": f"edge gradient ({method})"})


# --------------------------------------------------------------------------------------
# Watershed
# --------------------------------------------------------------------------------------
def watershed(
    elevation: Any,
    markers: Any,
    *,
    mask: Any = None,
    connectivity: int = 1,
) -> xr.DataArray:
    """Marker-controlled watershed, equal to ``skimage.segmentation.watershed``.

    Parameters
    ----------
    elevation
        The ``(y, x)`` image to flood (low values first).
    markers
        ``(y, x)`` integers: the label of each seed, 0 where unlabelled.
    mask
        ``(y, x)`` boolean, True where flooding is allowed (``None``: everywhere); pixels outside get 0.
    connectivity
        1 (the 4 neighbours) or 2 (the 8 neighbours).

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` int32 labels, with the coordinates of ``elevation`` (or of ``markers``).
    """
    el = _array2d(elevation, "elevation")
    mk = _array2d(markers, "markers")
    m = None if mask is None else _array2d(mask, "mask").astype(bool)
    out = _fc.watershed(el, mk, connectivity=connectivity, mask=m)
    return _wrap(out, _template(elevation, markers), "labels")


# --------------------------------------------------------------------------------------
# Stochastic watershed
# --------------------------------------------------------------------------------------
def _marker_sets(markers: Any) -> Optional[List[np.ndarray]]:
    """The explicit marker sets of ``markers`` (a list of (2, k) arrays, x = row, y = col), or None."""
    if isinstance(markers, np.ndarray) and markers.ndim == 3:
        markers = list(markers)
    if isinstance(markers, (list, tuple)):
        sets = [np.asarray(s, dtype=np.float64) for s in markers]
        if not sets:
            raise ValueError("markers: an empty list of marker sets")
        if any(s.ndim != 2 or s.shape[0] != 2 for s in sets):
            raise ValueError("markers: each marker set must be an array of shape (2, k) (row, col)")
        return sets
    return None


def _resolve_markers(markers: Any, marker_density: Optional[float], shape: Tuple[int, int]):
    """``(markers, density)``: ``markers`` as a number or a list of sets (None: the notebook's 1000), or,
    from ``marker_density`` (per million pixels of the whole grid), the count it stands for."""
    if marker_density is None:
        return (_NOTEBOOK["markers"] if markers is None else markers), None
    if markers is not None:
        raise ValueError("give markers or marker_density, not both")
    density = float(marker_density)
    if not np.isfinite(density) or density <= 0:
        raise ValueError("marker_density must be positive (markers per million pixels)")
    return max(1, int(round(density * shape[0] * shape[1] / 1e6))), density


def _check_stochastic(method: str, coloring: str) -> None:
    if method not in ("montecarlo", "exact"):
        raise ValueError(f"method must be 'montecarlo' or 'exact', got {method!r}")
    if coloring not in ("knn", "unique"):
        raise ValueError(f"coloring must be 'knn' or 'unique', got {coloring!r}")


def _boundary(grad: np.ndarray, mask: Optional[np.ndarray], *, method, realizations, markers, min_distance,
              marker_radius, n_neighbors, coloring, random_state, n_jobs):
    """``(probability float64, counts or None, realizations or None)`` of the stochastic watershed."""
    _check_stochastic(method, coloring)
    sets = _marker_sets(markers)
    if method == "exact":
        if sets is not None:
            raise ValueError("method='exact' draws its own uniform markers: markers must be a number")
        n = float(markers)
        if n < 0:
            raise ValueError("markers must not be negative")
        return _fc.exact_boundary_probability(grad, mask, n, n_jobs=n_jobs), None, None
    if sets is None:
        if int(realizations) < 1:
            raise ValueError("realizations must be at least 1")
        if int(markers) < 1:
            raise ValueError("markers must be at least 1")
        r = int(realizations)
    else:
        r = len(sets)
    counts = _fc.stochastic_watershed_counts(
        grad, mask, sets, realisations=r, markers=int(markers) if sets is None else 0,
        min_distance=min_distance, coloring=coloring, n_neighbors=n_neighbors, disk_radius=marker_radius,
        seed=random_state, n_jobs=n_jobs)
    return counts.astype(np.float64) / r, counts, r


def stochastic_watershed(
    gradient: Any,
    *,
    method: str = "montecarlo",
    realizations: int = 500,
    markers: Union[None, int, Sequence[np.ndarray]] = None,
    min_distance: float = 20,
    marker_radius: int = 3,
    n_neighbors: int = 20,
    coloring: str = "knn",
    mask: Any = None,
    marker_density: Optional[float] = None,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    return_counts: bool = False,
):
    """The probability that each pixel lies on a watershed boundary (Angulo & Jeulin 2007).

    ``method="montecarlo"`` is the MappingDemo notebook's loop (Baumann 2020): in each realisation,
    ``markers`` uniform random points are thinned to ``min_distance`` pixels apart, coloured on their
    ``n_neighbors``-nearest-neighbour graph (pyamg's maximal-independent-set colouring, colour for colour),
    dilated by a disk of ``marker_radius`` and flooded on ``gradient`` inside ``mask``; a pixel counts
    when it is on a boundary of the segmentation (``find_boundaries``, mode ``thick``, 8-connected,
    as the notebook). The realisations run in parallel and are summed without being kept.
    ``method="exact"`` is the exact probability (Meyer & Stawiaski 2010; Malmberg & Luengo Hendriks
    2014) of a different marker model: ``markers`` uniform markers of one pixel each, drawn with
    replacement among the mask pixels, no minimum distance, no disk, no colouring (``min_distance``,
    ``marker_radius``, ``n_neighbors`` and ``coloring`` are ignored). It costs about a pass of
    union-find instead of ``realizations`` watersheds, and equals no Monte Carlo of the first model.

    Parameters
    ----------
    gradient
        ``(y, x)`` edge strength (see ``zeit.edge_gradient``).
    method
        ``"montecarlo"`` or ``"exact"``.
    realizations
        Monte Carlo: how many watersheds (500 in the notebook).
    markers
        How many uniform random markers per realisation (``None``: 1000, the notebook's); for ``"exact"``,
        the number of markers of the model. For ``"montecarlo"`` it may also be a list of ``(2, k)`` float
        arrays, one per realisation, with (row, col) coordinates (the notebook's ``x``, ``y``, before
        ``astype(int)``): they are used as given, without thinning, and ``realizations``,
        ``min_distance`` and ``random_state`` are ignored.
    min_distance
        Pixels between markers after thinning (20 in the notebook; 0: no thinning).
    marker_radius
        Radius of the disk each marker is dilated to (3 in the notebook).
    n_neighbors
        Neighbours of the colouring graph (20 in the notebook).
    coloring
        ``"knn"`` (the notebook's: the markers are coloured so that near ones differ, 5 to 10 labels
        in all) or ``"unique"`` (every marker its own label).
    mask
        ``(y, x)`` boolean (True = flood), inside every watershed; ``None``: everywhere.
    marker_density
        Markers per million pixels of the grid (``rows * cols``, the whole grid, not only the mask), instead
        of ``markers``: the count is ``max(1, round(marker_density * rows * cols / 1e6))``, so the same value
        means the same density on windows of any size. Not together with ``markers``.
    random_state
        Seed of the Monte Carlo (``numpy.random.default_rng``); the result does not depend on ``n_jobs``.
    n_jobs
        CPU threads (``None`` or ``-1``: all but one).
    return_counts
        Also return the raw counts (Monte Carlo only).

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` float32 in [0, 1] (Monte Carlo: counts / realizations; ``attrs["realizations"]`` holds
        the number). With ``return_counts``, a pair ``(probability, counts)``, counts uint32 ``(y, x)``.

    Examples
    --------
    >>> grad = zeit.edge_gradient(swir)
    >>> prob = zeit.stochastic_watershed(grad, realizations=200, markers=1000, random_state=0)
    >>> exact = zeit.stochastic_watershed(grad, method="exact", markers=1000)
    """
    grad = _array2d(gradient, "gradient", np.float64)
    m = None if mask is None else _array2d(mask, "mask").astype(bool)
    if m is not None and m.shape != grad.shape:
        raise ValueError("mask must have the shape of gradient")
    if return_counts and method != "montecarlo":
        raise ValueError("return_counts needs method='montecarlo'")
    markers, density = _resolve_markers(markers, marker_density, grad.shape)
    prob, counts, r = _boundary(
        grad, m, method=method, realizations=realizations, markers=markers, min_distance=min_distance,
        marker_radius=marker_radius, n_neighbors=n_neighbors, coloring=coloring,
        random_state=random_state, n_jobs=n_jobs)
    tmpl = _template(gradient)
    attrs = {"long_name": "boundary probability", "method": method}
    if density is not None:
        attrs["markers"] = int(markers)
        attrs["marker_density"] = density
    if r is not None:
        attrs["realizations"] = r
    out = _wrap(prob.astype(np.float32), tmpl, "boundary", attrs)
    if return_counts:
        return out, _wrap(counts, tmpl, "counts", {"long_name": "boundary count", "realizations": r})
    return out


# --------------------------------------------------------------------------------------
# The whole method
# --------------------------------------------------------------------------------------
def fields(
    data: Any,
    *,
    gradient: Any = "tophat",
    method: str = "montecarlo",
    realizations: int = 500,
    markers: Union[None, int, Sequence[np.ndarray]] = None,
    min_distance: float = 20,
    marker_radius: int = 3,
    n_neighbors: int = 20,
    coloring: str = "knn",
    low: float = 0.3,
    high: float = 0.6,
    min_size: int = 50,
    final: Optional[str] = None,
    mask: Any = None,
    mask_threshold: Optional[float] = None,
    band: Any = None,
    time: Any = None,
    preset: Optional[str] = None,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = None,
    marker_density: Optional[float] = None,
    mask_quantile: Optional[float] = None,
) -> xr.Dataset:
    """Delineate fields by the stochastic watershed (Angulo & Jeulin 2007), as in the MappingDemo notebook.

    The steps are those of the notebook (Baumann 2020, after E. Gouillart's segmentation tutorial),
    whose values are the defaults: (1) a gradient image (``gradient``); (2) the probability that each
    pixel lies on a boundary of a watershed from random markers (``zeit.stochastic_watershed``), with the
    ``mask`` inside every watershed; (3) a hysteresis threshold on that probability (``low``, ``high``),
    whose complement is labelled (4-connected) with the boundary network as one label of its own;
    objects below ``min_size`` pixels are dropped; (4) a final watershed from those labels on the image
    (or the gradient), without the mask; (5) the pixels outside the mask are set to 0. The notebook's
    random relabelings of the label ids are not applied.

    Parameters
    ----------
    data
        The image: a map ``(y, x)``, a stack ``(band, y, x)``, a cube ``(time, y, x)`` or
        ``(time, band, y, x)`` (DataArray, NumPy array, or anything ``load_raster`` reads). Dims of
        length 1 are squeezed.
    gradient
        ``"tophat"`` (the notebook's), ``"sobel"`` or ``"multitemporal"`` (see ``zeit.edge_gradient``), or
        a ready ``(y, x)`` gradient (DataArray or array, e.g. a learned edge map).
    method
        ``"montecarlo"`` (the notebook's) or ``"exact"`` (uniform one-pixel markers; ``min_distance``,
        ``marker_radius``, ``n_neighbors`` and ``coloring`` are ignored). **The hysteresis thresholds are
        not calibrated for the exact probability**: ``low`` and ``high`` are applied to it as given and
        must be tuned (this is the validation of phase 24e).
    realizations, markers, min_distance, marker_radius, n_neighbors, coloring
        The Monte Carlo of ``zeit.stochastic_watershed`` (the notebook's values). ``markers`` may be a
        list of ``(2, k)`` (row, col) arrays, one per realisation, used as given (the number of
        realisations is then their count): this is how the notebook is reproduced exactly. ``markers=None``
        means 1000, the notebook's.
    marker_density
        Markers per million pixels of the grid (``rows * cols``, the whole grid, not only the mask), instead
        of ``markers`` (an error to give both): the count is ``max(1, round(marker_density * rows * cols /
        1e6))``. ``markers`` is a count per image, so the same value is a different density on windows of
        different sizes; the sensitivity analysis (phase 24e) found F rising up to about 400 to 1000 markers
        per million pixels and flat above. Works with ``"montecarlo"`` and ``"exact"``; ``attrs`` holds the
        resolved ``markers`` and the ``marker_density``.
    low, high
        The hysteresis, as fractions of the realisations (Monte Carlo: the integer counts are compared,
        strictly greater, with ``low * realizations`` and ``high * realizations``) or of the probability.
    min_size
        Objects of fewer pixels are removed before the final watershed (the boundary network
        counts as one object).
    final
        ``"image"``: the final watershed floods the image (the notebook's); ``"gradient"``: the
        gradient. ``None``: ``"image"`` for one 2-D image (after ``band`` / ``time``), else
        ``"gradient"``. ``"image"`` on a multi-layer input is an error.
    mask
        ``(y, x)`` boolean, True where the pixel is kept (e.g. not forest).
    mask_threshold
        Keep the pixels where the image is greater than this, in the image's own units (needs one 2-D image).
        Not together with ``mask``.
    mask_quantile
        A relative mask, in percent (0 to 100): keep the pixels where the image is strictly greater than its own
        ``mask_quantile``-th percentile over the finite pixels (``numpy.percentile``, linear). An absolute
        threshold does not transfer between regions (phase 24e); this one adapts to the image. Needs one 2-D
        image; not together with ``mask`` or ``mask_threshold``. ``attrs`` holds ``mask_quantile`` and the
        ``mask_threshold`` it resolved to.
    band, time
        Select one band and/or date by label (see ``zeit.edge_gradient``).
    preset
        ``"notebook"``: the notebook's values, which are the defaults above, plus ``final="image"``
        (unless ``final`` is given) and ``mask_threshold=2000`` (unless ``mask``, ``mask_threshold`` or
        ``mask_quantile`` is given). 2000 is the notebook's scale (Landsat surface reflectance times 10000, SWIR1). An
        argument passed explicitly is kept, so ``preset="notebook", low=0.4`` departs from the
        notebook in ``low`` only.
    random_state
        Seed of the Monte Carlo; the result does not depend on ``n_jobs``.
    n_jobs
        CPU threads (``None`` or ``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        - ``labels (y, x)``: int32 field labels, 1..k, 0 outside the mask;
        - ``boundary (y, x)``: float32 boundary probability, 0 to 1;
        - ``gradient (y, x)``: float32 gradient.

        Georeferenced like the input; the parameters are in ``attrs`` (``realizations`` is the number of
        watersheds actually run; ``boundary_label`` is 1 when the boundary network survived
        ``min_size`` and remained in the labels, as one label that the final watershed grew like the
        notebook does, else 0: drop it before counting fields).

    Examples
    --------
    >>> swir = zeit.load_raster("B5_medoid_2019.tif")                    # raw digital numbers
    >>> res = zeit.fields(swir, preset="notebook", random_state=0)
    >>> res["labels"].max()
    >>> res = zeit.fields(cube, gradient="multitemporal", mask=~forest, method="exact", markers=500)
    """
    if preset is not None:
        if preset != "notebook":
            raise ValueError(f"unknown preset {preset!r}: only 'notebook' exists")
        if final is None:
            final = _NOTEBOOK["final"]
        if mask is None and mask_threshold is None and mask_quantile is None:
            mask_threshold = _NOTEBOOK_MASK_THRESHOLD
    _check_stochastic(method, coloring)
    if final not in (None, "image", "gradient"):
        raise ValueError(f"final must be 'image' or 'gradient', got {final!r}")
    if not (0 <= low <= high <= 1):
        raise ValueError("low and high must satisfy 0 <= low <= high <= 1")
    if min_size < 0:
        raise ValueError("min_size must not be negative")
    if mask is not None and mask_threshold is not None:
        raise ValueError("give mask or mask_threshold, not both")
    if mask_quantile is not None:
        if mask is not None or mask_threshold is not None:
            raise ValueError("give only one of mask, mask_threshold and mask_quantile")
        if not (0 <= mask_quantile <= 100):
            raise ValueError("mask_quantile must be between 0 and 100 (percent)")
    if marker_density is not None and markers is not None:
        raise ValueError("give markers or marker_density, not both")

    cube = _select(_as_cube(data), band, time)
    cube, features = _layers(cube)
    single = not features
    if final is None:
        final = "image" if single else "gradient"
    if final == "image" and not single:
        raise ValueError(
            f"final='image' needs one 2-D image, got dims {tuple(cube.dims)}: select a layer with band= / time= "
            "or use final='gradient'")
    img = _image2d(cube, "the image") if single else None
    shape = (cube.sizes["y"], cube.sizes["x"])

    if isinstance(gradient, str):
        grad = _gradient_of(cube, gradient, "max")
    else:
        grad = _array2d(gradient, "gradient", np.float64)
        if grad.shape != shape:
            raise ValueError(f"gradient has shape {grad.shape}, the data {shape}")
    grad = np.asarray(grad, dtype=np.float64)

    if mask is not None:
        keep = _array2d(mask, "mask").astype(bool)
        if keep.shape != shape:
            raise ValueError(f"mask has shape {keep.shape}, the data {shape}")
    elif mask_threshold is not None:
        if img is None:
            raise ValueError("mask_threshold needs one 2-D image: select a layer with band= / time=, or pass mask=")
        keep = img > mask_threshold
    elif mask_quantile is not None:
        if img is None:
            raise ValueError("mask_quantile needs one 2-D image: select a layer with band= / time=, or pass mask=")
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            raise ValueError("mask_quantile: the image has no finite pixels")
        mask_threshold = float(np.percentile(finite, mask_quantile))
        keep = img > mask_threshold
    else:
        keep = None

    markers, density = _resolve_markers(markers, marker_density, shape)
    prob, counts, r = _boundary(
        grad, keep, method=method, realizations=realizations, markers=markers, min_distance=min_distance,
        marker_radius=marker_radius, n_neighbors=n_neighbors, coloring=coloring,
        random_state=random_state, n_jobs=n_jobs)

    # Hysteresis: the integer counts against low * R and high * R (not divided first: ties).
    if counts is not None:
        hyst = _fc.hysteresis_threshold(counts.astype(np.float64), low * r, high * r)
    else:
        hyst = _fc.hysteresis_threshold(prob, low, high)
    regions, _ = _fc.label(~hyst, connectivity=1)
    regions = _fc.remove_small_objects(regions + 1, min_size)
    network_kept = bool((regions == 1).any())  # label 1 is the boundary network until relabelled
    regions, _ = _fc.relabel_sequential(regions)

    surface = img if final == "image" else grad
    lab = _fc.watershed(surface, regions, connectivity=1)
    if keep is not None:
        lab = np.where(keep, lab, 0).astype(np.int32)
    # The network is label 1 of ``regions`` (the smallest, so it stays 1 through every relabelling).
    boundary_label = int(network_kept and (lab == 1).any())
    lab, _ = _fc.relabel_sequential(lab)

    tmpl = _template(data if isinstance(data, xr.DataArray) else None, cube, gradient)
    attrs = {
        "algorithm": "stochastic watershed", "gradient": gradient if isinstance(gradient, str) else "given",
        "method": method, "markers": int(markers) if _marker_sets(markers) is None else "given",
        "min_distance": float(min_distance), "marker_radius": int(marker_radius),
        "n_neighbors": int(n_neighbors), "coloring": coloring, "low": float(low), "high": float(high),
        "min_size": int(min_size), "final": final, "boundary_label": boundary_label,
    }
    if r is not None:
        attrs["realizations"] = r
    if density is not None:
        attrs["marker_density"] = density
    if mask_threshold is not None:
        attrs["mask_threshold"] = float(mask_threshold)
    if mask_quantile is not None:
        attrs["mask_quantile"] = float(mask_quantile)
    if preset is not None:
        attrs["preset"] = preset
    if random_state is not None:
        attrs["random_state"] = int(random_state)
    return xr.Dataset(
        {
            "labels": _wrap(lab, tmpl, None, {"long_name": "field label"}),
            "boundary": _wrap(prob.astype(np.float32), tmpl, None, {"long_name": "boundary probability"}),
            "gradient": _wrap(grad.astype(np.float32), tmpl, None, {"long_name": "edge gradient"}),
        },
        attrs=attrs,
    )


# --------------------------------------------------------------------------------------
# Polygons
# --------------------------------------------------------------------------------------
def _georef(labels: xr.DataArray, transform: Any, crs: Any):
    """The transform and CRS of ``labels`` (as ``zeit.snic``'s result gives them), unless given."""
    if transform is None and labels.sizes.get("x", 0) > 1 and labels.sizes.get("y", 0) > 1:
        from ._warp import transform_of

        transform = transform_of(labels)
    if crs is None:
        try:
            crs = labels.rio.crs
        except Exception:  # no rioxarray metadata at all
            crs = None
    return transform, crs


def _measure(geoms: list, crs: Any):
    """``(area_ha, perimeter_m, kind)`` of each geometry, ``kind`` 'planar', 'geodesic' or None (no CRS)."""
    n = len(geoms)
    if crs is None:
        return np.full(n, np.nan), np.full(n, np.nan), None
    from pyproj import CRS

    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        from pyproj import Geod

        ell = crs.ellipsoid
        geod = Geod(a=ell.semi_major_metre, b=ell.semi_minor_metre)

        def one(poly):
            a, p = geod.geometry_area_perimeter(type(poly)(poly.exterior))
            a = abs(a)
            for ring in poly.interiors:
                ai, pi = geod.geometry_area_perimeter(type(poly)(ring))
                a -= abs(ai)
                p += pi
            return a, p

        areas, perims = np.zeros(n), np.zeros(n)
        for i, g in enumerate(geoms):
            for part in getattr(g, "geoms", [g]):
                a, p = one(part)
                areas[i] += a
                perims[i] += p
        return areas / 1e4, perims, "geodesic"
    if not crs.is_projected:
        raise ValueError(f"cannot measure areas in a CRS that is neither projected nor geographic: {crs.name}")
    factor = float(crs.axis_info[0].unit_conversion_factor)  # metres per CRS unit (0.3048 for feet)
    if not (factor > 0) or crs.axis_info[0].unit_name in ("degree", "grad", "radian"):
        raise ValueError(f"the CRS {crs.name} has no linear unit in metres or a multiple of it")
    areas = np.array([g.area for g in geoms]) * factor ** 2 / 1e4
    perims = np.array([g.length for g in geoms]) * factor
    return areas, perims, "planar"


def fields_to_polygons(result, *, transform=None, crs=None, include_boundary: bool = False):
    """Polygonise the field labels of ``zeit.fields`` into a GeoDataFrame, one row per field.

    The columns are ``field`` (the label), ``n_pixels``, ``area_ha``, ``perimeter_m``, ``compactness``
    (Polsby-Popper, ``4 pi A / P^2``: 1 for a circle) and ``geometry`` (a MultiPolygon where a label
    is in several pieces). Label 0 (outside the mask) is skipped, and so is the boundary network that
    the notebook's method keeps as one label (``result.attrs["boundary_label"]``), unless
    ``include_boundary``.

    Area and perimeter follow the CRS: planar in a projected CRS (its unit converted to metres: feet,
    kilometres...), geodesic on the CRS's ellipsoid in a geographic one (x = longitude, y = latitude;
    holes subtracted). Without a CRS, ``area_ha`` and ``perimeter_m`` are NaN (and a warning says so):
    ``compactness`` is still computed, on the geometry in pixel (or map) units.

    Parameters
    ----------
    result
        The Dataset of ``zeit.fields``, or its ``labels`` DataArray (or a ``(y, x)`` integer array,
        for which ``transform`` and ``crs`` are needed to be georeferenced).
    transform, crs
        Affine transform and CRS to use instead of the ones of ``result`` (identity: pixel coordinates).
    include_boundary
        Keep the boundary network's polygon (the label ``attrs["boundary_label"]``).

    Returns
    -------
    geopandas.GeoDataFrame

    Examples
    --------
    >>> res = zeit.fields(swir, preset="notebook", random_state=0)
    >>> polygons = zeit.fields_to_polygons(res)
    >>> polygons.to_file("fields.gpkg")
    """
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union

    if isinstance(result, xr.Dataset):
        if "labels" not in result:
            raise ValueError("the Dataset has no 'labels' variable: pass the result of zeit.fields")
        labels, boundary = result["labels"], int(result.attrs.get("boundary_label", 0))
    elif isinstance(result, xr.DataArray):
        labels, boundary = result, int(result.attrs.get("boundary_label", 0))
    else:
        labels, boundary = xr.DataArray(np.asarray(result), dims=("y", "x")), 0
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got dims {labels.dims}")
    transform, crs = _georef(labels, transform, crs)
    transform = Affine.identity() if transform is None else transform
    values = np.asarray(labels.transpose("y", "x").values)
    if values.dtype.kind not in "iu":
        raise TypeError(f"labels must be integers, got {values.dtype}")
    values = values.astype(np.int32)

    keep = values > 0
    if boundary and not include_boundary:
        keep &= values != boundary
    geoms: dict = {}
    for geom, value in features.shapes(values, mask=keep, transform=transform, connectivity=4):
        geoms.setdefault(int(value), []).append(shape(geom))
    ids = np.array(sorted(geoms), dtype=np.int64)
    shapes = [unary_union(geoms[i]) if len(geoms[i]) > 1 else geoms[i][0] for i in ids]

    if crs is None:
        import warnings

        warnings.warn("fields_to_polygons: no CRS, so area_ha and perimeter_m are NaN (compactness is "
                      "computed in the units of the transform); pass crs= to measure", UserWarning, stacklevel=2)
    area_ha, perimeter_m, _ = _measure(shapes, crs)
    if crs is None:
        area = np.array([g.area for g in shapes])
        perim = np.array([g.length for g in shapes])
    else:
        area, perim = area_ha * 1e4, perimeter_m
    with np.errstate(invalid="ignore", divide="ignore"):
        compactness = 4 * np.pi * area / perim ** 2
    n_pixels = np.bincount(values[keep], minlength=int(ids.max()) + 1 if len(ids) else 1)[ids]
    return gpd.GeoDataFrame(
        {"field": ids, "n_pixels": n_pixels, "area_ha": area_ha, "perimeter_m": perimeter_m,
         "compactness": compactness},
        geometry=shapes, crs=crs)
