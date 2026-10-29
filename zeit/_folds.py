"""``zeit.spatial_folds`` and ``zeit.autocorrelation_range``: cross-validation folds that keep the test points away
from the training points, as blockCV's (Valavi et al. 2019, doi:10.1111/2041-210X.13107).

Points near each other are alike (spatial autocorrelation): a random k-fold validation tests a model on points next
to its training points and says it is better than it is where it will be applied. Spatial folds hold out whole
regions: blocks of a grid (squares or hexagons), a point and its neighbours within a buffer, clusters of the
points (in space or in the predictors), or kNNDM's folds (Linnenbrink et al. 2024, doi:10.5194/gmd-17-5897-2024),
whose distances from test to training points match those from the area to predict to the training points. They
choose a model and calibrate its area of applicability
(``zeit.train_di``); the accuracy of a map is still that of a probability sample (``zeit.accuracy``; Wadoux et al.
2021). The size of a block comes from the range of the predictors' spatial autocorrelation
(``zeit.autocorrelation_range``: an exponential semivariogram fitted to a sample of pixels, in the core).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._threads import resolve

METHODS = ("blocks", "buffer", "clusters", "knndm")


@dataclass
class SpatialFolds:
    """Cross-validation folds of sample points (``zeit.spatial_folds``); a scikit-learn splitter.

    ``folds``: each point's fold (0 to k-1; -1: in none, never held out but used for training); ``exclude``: for a
    buffer, the points each one's training leaves out (its neighbours); ``blocks``: the blocks (polygons, with their
    fold) for ``method="blocks"``; ``points``: the samples, in the CRS the distances were measured in. For kNNDM,
    ``W`` (the statistic the folds minimise), ``q`` (the clusters merged into them) and ``distances``: ``Gij``
    (prediction points to the nearest training point), ``Gj`` (each training point to its nearest other) and
    ``Gjstar`` (to the nearest of another fold).
    """

    method: str
    folds: np.ndarray
    points: Any = field(repr=False)
    size: Optional[float] = None
    exclude: Optional[List[np.ndarray]] = field(default=None, repr=False)
    blocks: Any = field(default=None, repr=False)
    W: Optional[float] = None
    q: Optional[int] = None
    distances: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)

    @property
    def k(self) -> int:
        return int(len(np.unique(self.folds[self.folds >= 0])))

    def split(self, X: Any = None, y: Any = None, groups: Any = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """(train, test) indices of each fold, as scikit-learn's ``cross_val_predict`` and ``GridSearchCV`` take
        them; for a buffer, one point at a time without its neighbours in the training."""
        if X is not None and len(X) != len(self.folds):
            raise ValueError(f"{len(X)} rows for folds of {len(self.folds)} points")
        everyone = np.arange(len(self.folds))
        for f in np.unique(self.folds[self.folds >= 0]):
            test = np.flatnonzero(self.folds == f)
            out = self.folds == f
            if self.exclude is not None:
                for i in test:
                    out[self.exclude[i]] = True
            yield everyone[~out], test

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.k

    def subset(self, keep: Any) -> "SpatialFolds":
        """The folds of the points ``keep`` says (a boolean per point), a buffer's neighbours among them."""
        import dataclasses

        keep = np.asarray(keep, dtype=bool)
        if keep.size != len(self.folds):
            raise ValueError(f"{keep.size} values to subset folds of {len(self.folds)} points")
        exclude = None
        if self.exclude is not None:
            new = np.cumsum(keep) - 1
            exclude = [new[e[keep[e]]] for e, k in zip(self.exclude, keep) if k]
        return dataclasses.replace(self, folds=self.folds[keep], points=self.points[keep].reset_index(drop=True),
                                   exclude=exclude)

    def __len__(self) -> int:
        return len(self.folds)

    def __repr__(self) -> str:
        sizes = np.unique(self.folds[self.folds >= 0], return_counts=True)[1]
        what = f"{self.k} folds of {sizes.min()}-{sizes.max()} points" if self.method != "buffer" else (
            f"leave-one-out, {np.mean([len(e) for e in self.exclude]):.3g} neighbours left out on average")
        none = int((self.folds < 0).sum())
        size = "" if self.size is None else f", size {self.size:.6g}"
        w = "" if self.W is None else f", W {self.W:.6g} from {self.q} clusters"
        return (f"SpatialFolds({self.method}: {len(self.folds)} points, {what}{size}{w}"
                + (f", {none} in none" if none else "") + ")")

    def plot(self, ax: Any = None, kind: str = "points", **kwargs: Any) -> Any:
        """The points colored by fold (and the blocks); with ``kind="distances"``, kNNDM's distributions of the
        nearest-neighbour distances (Gij, Gj and Gj*). Returns the matplotlib axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6) if kind == "points" else (6, 4))
        if kind == "distances":
            if self.distances is None:
                raise ValueError("kind='distances' is for method='knndm'")
            labels = {"Gij": "prediction to training", "Gj": "leave-one-out", "Gjstar": "these folds"}
            for key, label in labels.items():
                d = np.sort(self.distances[key][np.isfinite(self.distances[key])])
                ax.step(d, np.arange(1, d.size + 1) / d.size, where="post", label=f"{key}: {label}")
            ax.set_xlabel("distance to the nearest training point")
            ax.set_ylabel("cumulative share")
            ax.set_title(f"kNNDM: W {self.W:.4g}")
            ax.legend()
            return ax
        if kind != "points":
            raise ValueError(f"kind must be 'points' or 'distances', got {kind!r}")
        if self.blocks is not None:
            self.blocks.plot(ax=ax, column="fold", categorical=True, cmap="tab10", alpha=0.25, edgecolor="0.4",
                             linewidth=0.5)
        xy = _xy(self.points)
        shown = self.folds >= 0
        ax.scatter(xy[shown, 0], xy[shown, 1], c=self.folds[shown], cmap="tab10", s=kwargs.pop("s", 8), **kwargs)
        if (~shown).any():
            ax.scatter(xy[~shown, 0], xy[~shown, 1], c="0.6", marker="x", s=8)
        ax.set_aspect("equal")
        ax.set_title(f"{self.method}: {self.k if self.method != 'buffer' else len(self.folds)} folds")
        return ax


def _xy(points) -> np.ndarray:
    return np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])


def _samples(samples: Any, crs: Any = None):
    """The samples as points in a CRS whose distances are in metres (``crs`` if given, else theirs; a geographic one
    replaced by its UTM zone)."""
    import geopandas as gpd

    if not isinstance(samples, gpd.GeoDataFrame):
        samples = gpd.read_file(samples)
    if not (samples.geometry.geom_type == "Point").all():
        raise ValueError("samples must be points (take polygons' representative_point() first)")
    if crs is not None and samples.crs is not None and samples.crs != crs:
        samples = samples.to_crs(crs)
    if samples.crs is not None and samples.crs.is_geographic:
        samples = samples.to_crs(samples.estimate_utm_crs())
    return samples.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------- blocks

_SQRT3 = np.sqrt(3.0)


def _hexagons(xy: np.ndarray, size: float, origin: np.ndarray) -> np.ndarray:
    """Each point's hexagon (axial coordinates q, r) in a grid of pointy-top hexagons whose neighbouring centres are
    ``size`` apart (the hexagons' width between opposite sides), from rounding the cube coordinates."""
    radius = size / _SQRT3
    px, py = (xy - origin).T
    q = (_SQRT3 / 3 * px - py / 3) / radius
    r = (2.0 / 3 * py) / radius
    s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    fix_q = (dq > dr) & (dq > ds)
    fix_r = ~fix_q & (dr > ds)
    rq = np.where(fix_q, -rr - rs, rq)
    rr = np.where(fix_r, -rq - rs, rr)
    return np.column_stack([rq, rr]).astype(np.int64)


def _hexagon_polygon(q: int, r: int, size: float, origin: np.ndarray):
    from shapely.geometry import Polygon

    radius = size / _SQRT3
    cx = origin[0] + radius * _SQRT3 * (q + r / 2)
    cy = origin[1] + radius * 1.5 * r
    angles = np.deg2rad(30 + 60 * np.arange(6))
    return Polygon(np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)]))


def _imbalance(folds: np.ndarray, k: int, classes: Optional[np.ndarray]) -> float:
    """How far the folds are from holding a k-th of the points (of each class), as a chi-square."""
    if classes is None:
        expected = len(folds) / k
        return float(((np.bincount(folds, minlength=k) - expected) ** 2).sum() / expected)
    total = 0.0
    for c in np.unique(classes):
        sel = classes == c
        expected = sel.sum() / k
        total += ((np.bincount(folds[sel], minlength=k) - expected) ** 2).sum() / expected
    return float(total)


def _blocks(points, xy: np.ndarray, k: int, size: float, shape: str, selection: str, classes: Optional[np.ndarray],
            iterations: int, rng: np.random.Generator):
    import geopandas as gpd
    from shapely.geometry import box

    origin = xy.min(axis=0)
    if shape == "hexagon":
        cells = _hexagons(xy, size, origin)
        order = (1, 0)  # rows (r), then along them (q)
    elif shape == "square":
        cells = np.floor((xy - origin) / size).astype(np.int64)[:, ::-1]  # (row, col)
        order = (0, 1)
    else:
        raise ValueError(f"shape must be 'hexagon' or 'square', got {shape!r}")
    ids, block = np.unique(cells, axis=0, return_inverse=True)
    block = block.ravel()
    nb = len(ids)
    if nb < k:
        raise ValueError(f"only {nb} blocks of size {size:.6g} hold points, fewer than the {k} folds: give a smaller "
                         "size or fewer folds")
    if selection == "random":
        best, best_score = None, np.inf
        for _ in range(max(int(iterations), 1)):
            of_block = rng.permutation(np.arange(nb) % k)
            score = _imbalance(of_block[block], k, classes)
            if score < best_score:
                best, best_score = of_block, score
        of_block = best
    elif selection == "systematic":
        rank = np.lexsort((ids[:, order[1]], ids[:, order[0]]))
        of_block = np.empty(nb, dtype=np.int64)
        of_block[rank] = np.arange(nb) % k
    elif selection == "checkerboard":
        if shape != "square":
            raise ValueError("selection='checkerboard' needs shape='square'")
        if k != 2:
            raise ValueError("selection='checkerboard' gives 2 folds: k=2")
        of_block = (ids.sum(axis=1) % 2).astype(np.int64)
    else:
        raise ValueError(f"selection must be 'random', 'systematic' or 'checkerboard', got {selection!r}")
    if shape == "hexagon":
        polygons = [_hexagon_polygon(q, r, size, origin) for q, r in ids]
    else:
        polygons = [box(origin[0] + c * size, origin[1] + r * size, origin[0] + (c + 1) * size,
                        origin[1] + (r + 1) * size) for r, c in ids]
    blocks = gpd.GeoDataFrame({"fold": of_block, "points": np.bincount(block, minlength=nb)}, geometry=polygons,
                              crs=points.crs)
    return of_block[block].astype(np.int64), blocks


# ------------------------------------------------------------------------------------------------- kNNDM

def _prediction_cells(feats, samplesize: int, crs: Any):
    """About ``samplesize`` cells of ``feats`` with every feature, on a regular grid: their coordinates (in
    ``crs``) and values."""
    values = np.asarray(feats.values, dtype=np.float64)
    good = np.isfinite(values).all(axis=0)
    ny, nx = good.shape
    step = max(int(np.sqrt(good.sum() / max(samplesize, 1))), 1)
    rows, cols = np.meshgrid(np.arange(step // 2, ny, step), np.arange(step // 2, nx, step), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    keep = good[rows, cols]
    rows, cols = rows[keep], cols[keep]
    x = np.asarray(feats.x.values, dtype=np.float64)[cols]
    y = np.asarray(feats.y.values, dtype=np.float64)[rows]
    if crs is not None and feats.rio.crs is not None and feats.rio.crs != crs:
        import geopandas as gpd

        pts = gpd.GeoSeries(gpd.points_from_xy(x, y), crs=feats.rio.crs).to_crs(crs)
        x, y = pts.x.to_numpy(), pts.y.to_numpy()
    return np.column_stack([x, y]), values[:, rows, cols].T


def _merge(labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Clusters merged into k folds of about equal size: the largest first (at random among equal sizes), each to
    the fold with the fewest points."""
    sizes = np.bincount(labels)
    order = rng.permutation(sizes.size)
    order = order[np.argsort(-sizes[order], kind="stable")]
    fold_of = np.empty(sizes.size, dtype=np.int64)
    counts = np.zeros(k, dtype=np.int64)
    for c in order:
        f = int(np.argmin(counts))
        fold_of[c] = f
        counts[f] += sizes[c]
    return fold_of[labels]


def _knndm(train: np.ndarray, pred: np.ndarray, k: int, clustering: str, maxp: float, candidates: int,
           rng: np.random.Generator, seed: int, n_jobs: int):
    """kNNDM's folds of the points ``train`` (n, d) for the prediction points ``pred`` (m, d): (folds, W, q,
    distances)."""
    from ._core import knndm as core

    n = len(train)
    threads = resolve(n_jobs)
    train = np.ascontiguousarray(train, dtype=np.float64)
    gij = core.nearest(train, np.ascontiguousarray(pred, dtype=np.float64), threads)
    gj = core.nearest_other(train, np.arange(n, dtype=np.int64), threads)
    qs = np.unique(np.round(np.geomspace(k, n, max(int(candidates), 2))).astype(int))
    if clustering == "hierarchical":
        from scipy.cluster.hierarchy import fcluster, linkage

        tree = linkage(train, method="ward")

        def clusters(q):
            return fcluster(tree, t=q, criterion="maxclust") - 1
    elif clustering == "kmeans":
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits

        def clusters(q):
            if q >= n:
                return np.arange(n)
            with threadpool_limits(threads):
                return KMeans(n_clusters=int(q), n_init=1, random_state=seed).fit_predict(train)
    else:
        raise ValueError(f"clustering must be 'hierarchical' or 'kmeans', got {clustering!r}")
    best = None
    for q in qs:
        labels = np.unique(clusters(q), return_inverse=True)[1].ravel()
        folds = _merge(labels, k, rng)
        if np.bincount(folds, minlength=k).max() > maxp * n:
            continue
        gjstar = core.nearest_other(train, folds, threads)
        w = core.wasserstein(gjstar, gij)
        if best is None or w < best[1]:
            best = (folds, w, int(q), gjstar)
    if best is None:
        raise ValueError(f"every configuration has a fold with more than maxp={maxp} of the points: the points are "
                         "too clustered for these folds; raise maxp or lower k")
    folds, w, q, gjstar = best
    return folds, float(w), q, {"Gij": gij, "Gj": gj, "Gjstar": gjstar}


# ------------------------------------------------------------------------------------------------- the API

def as_folds(folds: Any, points, kept: np.ndarray, *, data: Any = None, label: Optional[str] = None) -> SpatialFolds:
    """``folds`` of every sample point (a ``SpatialFolds``, labels, a column of ``points``, or the name of a
    method of ``spatial_folds``, run with ``data``) as the ``SpatialFolds`` of the points ``kept``."""
    from ._aoa import _folds

    n = len(points)
    if isinstance(folds, str) and folds in METHODS:
        # on the points kept only: those left out would make folds of their own
        return spatial_folds(points[kept].reset_index(drop=True), method=folds, data=data,
                             label=label if folds == "blocks" else None)
    if isinstance(folds, str):
        if folds not in points.columns:
            raise ValueError(f"folds={folds!r} is neither a method of spatial_folds {METHODS} nor a column of the "
                             f"samples ({list(points.columns)})")
        folds = points[folds].to_numpy()
    if not isinstance(folds, SpatialFolds):
        labels = _folds(folds, len(np.asarray(folds, dtype=object).ravel()))
        if labels.size == n:
            return SpatialFolds("labels", labels[kept], points[kept].reset_index(drop=True))
        if labels.size == kept.sum():
            return SpatialFolds("labels", labels, points[kept].reset_index(drop=True))
        raise ValueError(f"{labels.size} folds for {n} samples ({kept.sum()} on a pixel with every feature)")
    if len(folds) == n:
        return folds.subset(kept)
    if len(folds) == kept.sum():
        return folds
    raise ValueError(f"folds of {len(folds)} points for {n} samples ({kept.sum()} on a pixel with every feature)")


def cross_validate(model: Any, x: np.ndarray, y: np.ndarray, folds: SpatialFolds, index: Any = None) -> pd.DataFrame:
    """Each point's prediction by a copy of ``model`` fitted without its fold (and, for a buffer, its neighbours):
    a ``DataFrame`` of ``observed``, ``predicted`` (missing for a point in no fold) and ``fold``."""
    import copy

    try:
        from sklearn.base import clone

        fresh = lambda: clone(model)  # noqa: E731
        fresh()
    except TypeError:
        fresh = lambda: copy.deepcopy(model)  # noqa: E731
    predicted = np.full(len(y), None, dtype=object)
    for train, test in folds.split(x):
        if len(train) == 0:
            continue
        predicted[test] = fresh().fit(x[train], y[train]).predict(x[test])
    out = pd.DataFrame({"observed": y, "predicted": predicted, "fold": folds.folds}, index=index)
    if not pd.isna(predicted).any():
        out["predicted"] = out["predicted"].astype(np.asarray(y).dtype)
    return out


def spatial_folds(samples: Any, *, method: str = "blocks", k: int = 5, size: Optional[float] = None,
                  shape: str = "hexagon", selection: str = "random", label: Optional[str] = None,
                  iterations: int = 100, space: str = "geographic", data: Any = None,
                  prediction_points: Any = None, samplesize: int = 1000, clustering: str = "hierarchical",
                  maxp: float = 0.5, candidates: int = 100, seed: int = 0, n_jobs: int = -1) -> SpatialFolds:
    """Cross-validation folds of sample points that keep the test points away from the training points, as
    blockCV's ``cv_spatial``, ``cv_buffer`` and ``cv_cluster``, and CAST's ``knndm``.

    samples: the points (a ``GeoDataFrame`` or a vector file). Distances are in their CRS's units (a geographic CRS
        is replaced by its UTM zone; with ``data``, the points are taken to its CRS first).
    method:
        ``"blocks"``: the points in blocks of a grid (``shape``: ``"hexagon"`` or ``"square"``, ``size`` across:
        the side of a square, the distance between neighbouring hexagons' centres), the blocks in ``k`` folds
        (``selection``: ``"random"``, ``"systematic"`` in rows, or ``"checkerboard"``, squares and k = 2). At
        random, the most balanced of ``iterations`` draws: the folds closest to a k-th of the points, and of each
        class with ``label``.
        ``"buffer"``: leave-one-out, each point tested without its neighbours within ``size`` in the training.
        ``"clusters"``: ``k`` clusters of the points by k-means: of their coordinates, or of the predictors
        (``space="feature"``, their values in ``data`` at the points, standardised).
        ``"knndm"``: kNNDM (Linnenbrink et al. 2024): the points in q clusters (``clustering``: Ward's
        ``"hierarchical"`` or ``"kmeans"``), merged into ``k`` folds of about equal size, for up to ``candidates``
        values of q from k to the number of points (q = n: a random k-fold), none with a fold above ``maxp`` of the
        points; the folds kept are those whose distances from each point to the nearest point of another fold (Gj*)
        are distributed most like those from the area to predict to the nearest point (Gij), by the W statistic
        (the integral of the absolute difference of their distribution functions). The area is ``prediction_points``
        or a regular sample of about ``samplesize`` cells of ``data`` with every predictor; in
        ``space="feature"``, their predictors (standardised by the points').
    size: the blocks' size or the buffer's radius, in the CRS's units; for blocks, None takes the median range of
        the spatial autocorrelation of ``data``'s bands (``zeit.autocorrelation_range``).
    label: the column of ``samples`` with the classes, to balance them across random blocks.
    data: the predictors (anything ``zeit.classify`` takes): for ``size=None``, ``space="feature"`` and kNNDM's area.
    prediction_points: kNNDM's prediction points, instead of a sample of ``data`` (geographic space).

    Returns a ``SpatialFolds``: ``.folds`` (each point's fold), a scikit-learn splitter (``split``,
    ``get_n_splits``) for ``cross_val_predict`` or ``GridSearchCV``, accepted by ``zeit.train_di`` and ``zeit.aoa``
    as ``folds``; ``.plot()``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    feats = None
    if data is not None:
        from ._classify_api import features

        feats = features(data)
    points = _samples(samples, None if feats is None else feats.rio.crs)
    n = len(points)
    if n < 2:
        raise ValueError("at least two sample points are needed")
    xy = _xy(points)
    rng = np.random.default_rng(seed)
    k = int(k)
    if method != "buffer" and not 2 <= k <= n:
        raise ValueError(f"k must be between 2 and the {n} points, got {k}")
    classes = None
    if label is not None:
        if label not in points.columns:
            raise ValueError(f"samples have no column {label!r}; they have {list(points.columns)}")
        classes = pd.factorize(points[label])[0]

    if method == "blocks":
        if size is None:
            if feats is None:
                raise ValueError("give size= (in the CRS's units), or data= to take it from the range of its "
                                 "spatial autocorrelation")
            size = float(autocorrelation_range(feats, seed=seed, n_jobs=n_jobs).attrs["range"])
        if not size > 0:
            raise ValueError(f"size must be positive, got {size}")
        folds, blocks = _blocks(points, xy, k, float(size), shape, selection, classes, iterations, rng)
        return SpatialFolds("blocks", folds, points, size=float(size), blocks=blocks)

    if method == "buffer":
        from scipy.spatial import cKDTree

        if size is None or not size > 0:
            raise ValueError("method='buffer' needs size= (the buffer's radius, in the CRS's units)")
        near = cKDTree(xy).query_ball_point(xy, r=float(size), workers=resolve(n_jobs))
        exclude = [np.array(sorted(set(row) - {i}), dtype=np.int64) for i, row in enumerate(near)]
        return SpatialFolds("buffer", np.arange(n, dtype=np.int64), points, size=float(size), exclude=exclude)

    if method == "knndm":
        folds = np.full(n, -1, dtype=np.int64)
        if space == "geographic":
            train, kept = xy, np.ones(n, dtype=bool)
            if prediction_points is not None:
                pred = _xy(_samples(prediction_points, points.crs))
            elif feats is not None:
                pred = _prediction_cells(feats, samplesize, points.crs)[0]
            else:
                raise ValueError("method='knndm' needs the area to predict: data= or prediction_points=")
        elif space == "feature":
            if feats is None:
                raise ValueError("space='feature' needs data= (the predictors at the points and in the area)")
            from ._classify_api import _values_at

            train, kept = _values_at(feats, points)
            pred = _prediction_cells(feats, samplesize, None)[1]
            mean, sd = train.mean(axis=0), train.std(axis=0, ddof=1)
            sd = np.where(sd > 0, sd, 1)
            train, pred = (train - mean) / sd, (pred - mean) / sd
        else:
            raise ValueError(f"space must be 'geographic' or 'feature', got {space!r}")
        if kept.sum() < k:
            raise ValueError(f"only {kept.sum()} points have every predictor, fewer than the {k} folds")
        if len(pred) == 0:
            raise ValueError("the area to predict has no point")
        f, w, q, distances = _knndm(train, pred, k, clustering, float(maxp), candidates, rng, seed, n_jobs)
        folds[kept] = f
        return SpatialFolds("knndm", folds, points, W=w, q=q, distances=distances)

    # clusters
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    folds = np.full(n, -1, dtype=np.int64)
    if space == "geographic":
        values, kept = xy, np.ones(n, dtype=bool)
    elif space == "feature":
        if feats is None:
            raise ValueError("space='feature' needs data= (the predictors at the points)")
        from ._classify_api import _values_at

        values, kept = _values_at(feats, points)
        sd = values.std(axis=0)
        values = (values - values.mean(axis=0)) / np.where(sd > 0, sd, 1)
        if kept.sum() < k:
            raise ValueError(f"only {kept.sum()} points have every predictor, fewer than the {k} folds")
    else:
        raise ValueError(f"space must be 'geographic' or 'feature', got {space!r}")
    with threadpool_limits(resolve(n_jobs)):
        folds[kept] = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(values)
    return SpatialFolds("clusters", folds, points)


def autocorrelation_range(data: Any, *, sample: int = 5000, bins: int = 15, cutoff: Optional[float] = None,
                          seed: int = 0, n_jobs: int = -1) -> pd.DataFrame:
    """The range of each band's spatial autocorrelation, as blockCV's ``cv_spatial_autocor``: the empirical
    semivariogram of ``sample`` pixels with a value, in ``bins`` bands of distance up to ``cutoff`` (default a third
    of the map's diagonal, as gstat), and an exponential model fitted to it (in the core).

    data: anything ``zeit.classify`` takes; a geographic CRS's pixels are measured in their UTM zone.

    Returns a ``DataFrame`` by band: ``range`` (the practical range, where the model reaches 95% of its sill above
    the nugget: 3 ``scale``), ``scale``, ``nugget``, ``sill`` (above the nugget) and ``pixels``; in ``attrs``, the
    median ``range`` (the blocks' size of ``zeit.spatial_folds``) and each band's semivariogram (``variograms``: a
    ``DataFrame`` of ``distance``, ``gamma`` and ``pairs``).
    """
    from ._classify_api import features
    from ._core import variogram as core

    feats = features(data)
    ny, nx = feats.sizes["y"], feats.sizes["x"]
    xs = np.asarray(feats.x.values, dtype=np.float64)
    ys = np.asarray(feats.y.values, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()
    crs = feats.rio.crs
    if crs is not None and crs.is_geographic:
        import geopandas as gpd

        pts = gpd.GeoSeries(gpd.points_from_xy(gx, gy), crs=crs)
        pts = pts.to_crs(pts.estimate_utm_crs())
        gx, gy = pts.x.to_numpy(), pts.y.to_numpy()
    if cutoff is None:
        cutoff = float(np.hypot(gx.max() - gx.min(), gy.max() - gy.min()) / 3)
    rng = np.random.default_rng(seed)
    rows, variograms = [], {}
    values = np.asarray(feats.values, dtype=np.float64).reshape(feats.sizes["feature"], ny * nx)
    for name, v in zip([str(f) for f in feats.feature.values], values):
        good = np.flatnonzero(np.isfinite(v))
        pick = good if good.size <= sample else rng.choice(good, size=int(sample), replace=False)
        r = core.fit(gx[pick], gy[pick], v[pick], bins=int(bins), cutoff=float(cutoff), n_jobs=resolve(n_jobs))
        variograms[name] = pd.DataFrame({"distance": r["distance"], "gamma": r["gamma"], "pairs": r["pairs"]})
        rows.append({"band": name, "range": r["range"], "scale": r["scale"], "nugget": r["nugget"],
                     "sill": r["sill"], "pixels": int(pick.size)})
    out = pd.DataFrame(rows).set_index("band").astype({"range": float, "scale": float, "nugget": float,
                                                        "sill": float})
    out.attrs["range"] = float(np.nanmedian(out["range"])) if out["range"].notna().any() else float("nan")
    out.attrs["variograms"] = variograms
    return out
