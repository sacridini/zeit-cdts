"""``zeit.train_di`` and ``zeit.aoa``: the area of applicability of a model (Meyer & Pebesma 2021), as CAST's
``trainDI`` and ``aoa``.

The predictors are scaled by the training data's means and standard deviations and weighted (by their importance).
A training point's dissimilarity index (DI) is its distance to the nearest training point of another
cross-validation fold over the mean distance between training points; the threshold is the upper whisker of those
(75% quantile + 1.5 IQR, at most their largest). A new point's DI is its distance to the nearest training point over
the same mean: it is in the area of applicability (AOA) when its DI is not above the threshold, and its local point
density (LPD) is how many training points are that near.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import xarray as xr

from ._threads import resolve


@dataclass
class TrainDI:
    """What ``zeit.train_di`` keeps of the training data (CAST's ``trainDI``)."""

    features: List[str]
    center: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    train: np.ndarray = field(repr=False)
    mean_distance: float
    train_di: np.ndarray = field(repr=False)
    threshold: float

    def __repr__(self) -> str:
        n, p = self.train.shape
        return (f"TrainDI({n} training points, {p} predictors, mean distance {self.mean_distance:.6g}, "
                f"threshold {self.threshold:.6g})")


def _table(train: Any, features: Optional[List[str]] = None):
    """(values (n, p) float64, the predictors' names) of a DataFrame or an array."""
    import pandas as pd

    if isinstance(train, pd.DataFrame):
        names = [str(c) for c in train.columns]
        values = train.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(train, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError(f"the training data must be (points, predictors), got shape {values.shape}")
        names = list(features) if features is not None else [f"v{j + 1}" for j in range(values.shape[1])]
    return values, names


def _weights(weights: Any, model: Any, names: List[str]) -> np.ndarray:
    if weights is None and model is not None and hasattr(model, "feature_importances_"):
        weights = "importance"
    if weights is None:
        return np.ones(len(names))
    if isinstance(weights, str):
        if weights != "importance":
            raise ValueError(f"weights must be None, 'importance', a dict or a list, got {weights!r}")
        if model is None or not hasattr(model, "feature_importances_"):
            raise ValueError("weights='importance' needs a model with feature_importances_")
        return np.asarray(model.feature_importances_, dtype=np.float64)
    if isinstance(weights, dict):
        missing = [n for n in names if n not in weights]
        if missing:
            raise ValueError(f"no weight for {missing[:5]}")
        return np.array([float(weights[n]) for n in names])
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != len(names):
        raise ValueError(f"{w.size} weights for {len(names)} predictors")
    return w


def _folds(folds: Any, n: int) -> Optional[np.ndarray]:
    """Each point's fold as integers (negative: in none), from labels of any kind (NaN / None: none)."""
    if folds is None:
        return None
    labels = np.asarray(folds, dtype=object).ravel()
    if labels.size != n:
        raise ValueError(f"{labels.size} folds for {n} training points")
    out = np.full(n, -1, dtype=np.int64)
    seen = {}
    for i, v in enumerate(labels):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[i] = seen.setdefault(v, len(seen))
    return out


def train_di(train: Any = None, *, model: Any = None, weights: Any = None, folds: Any = None,
             n_jobs: int = -1) -> TrainDI:
    """The dissimilarity of the training data to itself and the AOA's threshold (CAST's ``trainDI``).

    train: the training data, ``(points, predictors)``: a DataFrame (the columns named) or an array; or none with
        ``model`` from ``zeit.train_classifier`` (its training data, ``zeit_train_``). Points with a missing value
        are left out.
    model: a model from ``zeit.train_classifier``: its training data and, unless ``weights`` says otherwise, its
        ``feature_importances_`` as the weights (CAST weights by the model's variable importance).
    weights: the predictors' weights: ``None`` (the model's importance if it has one, else all 1), ``"importance"``,
        a dict by name or a list. Negative weights count as 0.
    folds: each training point's cross-validation fold (any labels; None/NaN: in none, its DI missing): a point's
        DI is then its distance to the nearest point of another fold, as the model's validation saw it. Without
        folds, its nearest other point (spatially clustered samples then give a threshold too strict: give the
        clusters as folds).
    """
    features = None
    if train is None:
        if model is None or not hasattr(model, "zeit_train_"):
            raise ValueError("give the training data, or a model from zeit.train_classifier")
        train = model.zeit_train_
        features = list(model.zeit_features_)
    values, names = _table(train, features)
    w = _weights(weights, model, names)
    f = _folds(folds, len(values))
    keep = np.isfinite(values).all(axis=1)
    if f is not None:
        f = f[keep]
    values = values[keep]
    from ._core import aoa as core

    r = core.train_di(np.ascontiguousarray(values), w, f, resolve(n_jobs))
    return TrainDI(features=names, center=r["center"], scale=r["scale"], weight=r["weight"], train=r["train"],
                   mean_distance=float(r["mean_distance"]), train_di=r["train_di"], threshold=float(r["threshold"]))


def aoa(data: Any, train: Any = None, *, model: Any = None, weights: Any = None, folds: Any = None,
        lpd: bool = False, date: Any = None, n_jobs: int = -1) -> Any:
    """The area of applicability of a model (Meyer & Pebesma 2021; CAST's ``aoa``): where the data are like the
    training data, by the dissimilarity index of each pixel to them.

    data: what the model is applied to: anything ``zeit.classify`` takes (a cube, a stack, a Dataset of maps, CCDC
        segments with ``date``), its features matched to the training's by name; or a table of new points
        ``(points, predictors)`` (a DataFrame, or an array in the training's order).
    train: the training data (see ``zeit.train_di``), or a ``TrainDI`` already computed; none with ``model``.
    model, weights, folds: as ``zeit.train_di``.
    lpd: also the local point density: how many training points are within the threshold's dissimilarity.

    Returns, for a map, an ``xarray.Dataset`` georeferenced as the data: ``DI (y, x)``; ``AOA (y, x)``, 1 inside,
    0 outside, 255 where a feature is missing; ``LPD (y, x)`` with ``lpd=True``; its ``threshold`` and
    ``mean_distance`` in the attributes. For a table: a DataFrame of ``DI``, ``AOA`` (NaN where missing) and
    ``LPD``. The DI equals CAST 1.0.4's to 1e-13 (Euclidean distances; CAST's Mahalanobis option is not ported).
    """
    import pandas as pd

    td = train if isinstance(train, TrainDI) else train_di(train, model=model, weights=weights, folds=folds,
                                                            n_jobs=n_jobs)
    from ._core import aoa as core

    def run(values: np.ndarray):
        d, l = core.di(np.ascontiguousarray(values, dtype=np.float64), td.center, td.scale, td.weight, td.train,
                       td.mean_distance, td.threshold, bool(lpd), resolve(n_jobs))
        return d, (l if lpd else None)

    attrs = {"threshold": td.threshold, "mean_distance": td.mean_distance}
    if isinstance(data, pd.DataFrame) or (isinstance(data, np.ndarray) and data.ndim <= 2):
        if isinstance(data, pd.DataFrame):
            missing = [n for n in td.features if n not in data.columns]
            if missing:
                raise ValueError(f"the data lack predictors of the training: {missing[:5]}")
            values = data[td.features].to_numpy(dtype=np.float64)
            index = data.index
        else:
            values = np.asarray(data, dtype=np.float64).reshape(len(data), -1)
            index = None
        d, l = run(values)
        out = pd.DataFrame({"DI": d, "AOA": np.where(np.isnan(d), np.nan, (d <= td.threshold).astype(float))},
                           index=index)
        if l is not None:
            out["LPD"] = np.where(np.isnan(d), np.nan, l)
        out.attrs.update(attrs)
        return out

    from ._classify_api import features as _features

    feats = _features(data, date=date)
    names = [str(f) for f in feats.feature.values]
    if names != td.features:
        missing = [f for f in td.features if f not in names]
        if missing:
            raise ValueError(f"the data lack features of the training: {missing[:5]}")
        feats = feats.sel(feature=td.features)
    p, ny, nx = feats.shape
    values = np.asarray(feats.values, dtype=np.float64).reshape(p, -1).T
    d, l = run(values)
    coords = {c: feats.coords[c] for c in ("y", "x", "spatial_ref") if c in feats.coords}
    di_map = d.reshape(ny, nx).astype(np.float32)
    inside = np.where(np.isnan(d), 255, d <= td.threshold).astype(np.uint8).reshape(ny, nx)
    variables = {
        "DI": xr.DataArray(di_map, dims=("y", "x"), attrs={"long_name": "dissimilarity index"}),
        "AOA": xr.DataArray(inside, dims=("y", "x"),
                            attrs={"long_name": "area of applicability", "flag_values": [0, 1],
                                   "flag_meanings": "outside inside"}),
    }
    if l is not None:
        variables["LPD"] = xr.DataArray(np.where(np.isnan(d), np.nan, l).reshape(ny, nx).astype(np.float32),
                                        dims=("y", "x"), attrs={"long_name": "local point density"})
    ds = xr.Dataset(variables, coords=coords, attrs=attrs)
    if feats.rio.crs is not None:
        ds = ds.rio.write_crs(feats.rio.crs)
        ds["AOA"] = ds["AOA"].rio.write_nodata(255, encoded=False)
    return ds
