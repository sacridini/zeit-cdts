"""``zeit.spatial_folds`` and ``zeit.autocorrelation_range``: cross-validation folds that keep the test points away
from the training points, as blockCV's (Valavi et al. 2019, doi:10.1111/2041-210X.13107).

Points near each other are alike (spatial autocorrelation): a random k-fold validation tests a model on points next
to its training points and says it is better than it is where it will be applied. Spatial folds hold out whole
regions: blocks of a grid (squares or hexagons), a point and its neighbours within a buffer, or clusters of the
points (in space or in the predictors). They choose a model and calibrate its area of applicability
(``zeit.train_di``); the accuracy of a map is still that of a probability sample (``zeit.accuracy``; Wadoux et al.
2021). The size of a block comes from the range of the predictors' spatial autocorrelation
(``zeit.autocorrelation_range``: an exponential semivariogram fitted to a sample of pixels, in the core).
"""

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._threads import resolve

METHODS = ("blocks", "buffer", "clusters")


@dataclass
class SpatialFolds:
    """Cross-validation folds of sample points (``zeit.spatial_folds``); a scikit-learn splitter.

    ``folds``: each point's fold (0 to k-1; -1: in none, never held out but used for training); ``exclude``: for a
    buffer, the points each one's training leaves out (its neighbours); ``blocks``: the blocks (polygons, with their
    fold) for ``method="blocks"``; ``points``: the samples, in the CRS the distances were measured in.
    """

    method: str
    folds: np.ndarray
    points: Any = field(repr=False)
    size: Optional[float] = None
    exclude: Optional[List[np.ndarray]] = field(default=None, repr=False)
    blocks: Any = field(default=None, repr=False)

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

    def __len__(self) -> int:
        return len(self.folds)

    def __repr__(self) -> str:
        sizes = np.bincount(self.folds[self.folds >= 0])
        what = f"{self.k} folds of {sizes.min()}-{sizes.max()} points" if self.method != "buffer" else (
            f"leave-one-out, {np.mean([len(e) for e in self.exclude]):.3g} neighbours left out on average")
        none = int((self.folds < 0).sum())
        size = "" if self.size is None else f", size {self.size:.6g}"
        return (f"SpatialFolds({self.method}: {len(self.folds)} points, {what}{size}"
                + (f", {none} in none" if none else "") + ")")

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """The points colored by fold (and the blocks); returns the matplotlib axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
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


# ------------------------------------------------------------------------------------------------- the API

def spatial_folds(samples: Any, *, method: str = "blocks", k: int = 5, size: Optional[float] = None,
                  shape: str = "hexagon", selection: str = "random", label: Optional[str] = None,
                  iterations: int = 100, space: str = "geographic", data: Any = None, seed: int = 0,
                  n_jobs: int = -1) -> SpatialFolds:
    """Cross-validation folds of sample points that keep the test points away from the training points, as
    blockCV's ``cv_spatial``, ``cv_buffer`` and ``cv_cluster``.

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
    size: the blocks' size or the buffer's radius, in the CRS's units; for blocks, None takes the median range of
        the spatial autocorrelation of ``data``'s bands (``zeit.autocorrelation_range``).
    label: the column of ``samples`` with the classes, to balance them across random blocks.
    data: the predictors (anything ``zeit.classify`` takes): for ``size=None`` and ``space="feature"``.

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
