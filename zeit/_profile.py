"""``zeit.error_profile``: how a model's error grows with the dissimilarity of the data to its training data, as
CAST's ``errorProfiles`` (Meyer & Pebesma 2021), and the map of the error expected where it is applied.

The training points held out by a cross-validation are each predicted by a model fitted without their fold; their
dissimilarity index (DI, ``zeit.train_di``) is their distance to the nearest training point of another fold. Those
within the AOA's threshold, sorted by DI, give the error metric (RMSE, MAE, or the accuracy for classes) in a moving
window of ``window`` points, at the DI of the window's middle point; a curve that only grows (only falls, for the
accuracy) is fitted to those (a monotone P-spline, in the core) or a line. Applied to the DI of a map, it gives the
error to expect in each pixel inside the AOA.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ._threads import resolve

# each metric, and whether it grows with the error
METRICS = {"rmse": True, "mae": True, "accuracy": False}


def _metric(name: str, observed: np.ndarray, predicted: np.ndarray) -> float:
    if name == "accuracy":
        return float(np.mean(observed == predicted))
    error = observed.astype(np.float64) - predicted.astype(np.float64)
    return float(np.sqrt(np.mean(error ** 2))) if name == "rmse" else float(np.mean(np.abs(error)))


@dataclass
class ErrorProfile:
    """A model's error against the DI (``zeit.error_profile``): ``points`` (the moving windows' ``DI`` and
    ``metric``), the AOA's ``threshold``, and the ``curve`` fitted to them (call the profile on DI values for it).
    """

    metric: str
    window: int
    calibration: str
    threshold: float
    points: pd.DataFrame = field(repr=False)
    curve: Dict[str, Any] = field(repr=False)

    def __call__(self, di: Any) -> np.ndarray:
        """The curve at the DI values ``di`` (an array of any shape; NaN at NaN)."""
        di = np.asarray(di, dtype=np.float64)
        flat = np.ascontiguousarray(di.ravel())
        if self.calibration == "linear":
            out = self.curve["intercept"] + self.curve["slope"] * flat
        else:
            from ._core import profile as core

            out = core.predict(self.curve["lo"], self.curve["hi"], np.asarray(self.curve["coef"]), flat, resolve(-1))
        return np.asarray(out).reshape(di.shape)

    def predict(self, data: Any, *, outside: bool = False) -> Any:
        """The error expected in each pixel (or point) of a ``zeit.aoa`` result: a ``(y, x)`` ``DataArray``
        georeferenced as it (a ``Series`` for a table of points), missing outside the AOA (where the
        cross-validation says nothing) unless ``outside=True`` (the curve goes on as a line beyond its last
        point)."""
        name = f"expected_{self.metric}"
        if isinstance(data, pd.DataFrame):
            values = pd.Series(self(data["DI"].to_numpy()), index=data.index, name=name)
            if not outside and "AOA" in data:
                values = values.where(data["AOA"] == 1)
            return values
        if isinstance(data, xr.Dataset):
            di = data["DI"]
            values = self(di.values).astype(np.float32)
            if not outside and "AOA" in data:
                values = np.where(data["AOA"].values == 1, values, np.nan).astype(np.float32)
            out = xr.DataArray(values, dims=di.dims, coords=di.coords, name=name,
                               attrs={"long_name": f"expected {self.metric} (error profile by the DI)"})
            if data.rio.crs is not None:
                out = out.rio.write_crs(data.rio.crs)
            return out
        return self(data)

    def to_dict(self) -> Dict[str, Any]:
        curve = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in self.curve.items()}
        return {"zeit_error_profile": 1, "metric": self.metric, "window": self.window,
                "calibration": self.calibration, "threshold": self.threshold, "curve": curve,
                "points": {"DI": self.points["DI"].tolist(), "metric": self.points["metric"].tolist()}}

    def save(self, path: Any) -> Path:
        """Writes the profile as JSON (``ErrorProfile.load`` reads it back)."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: Any) -> "ErrorProfile":
        d = json.loads(Path(path).read_text())
        if d.get("zeit_error_profile") != 1:
            raise ValueError(f"{path} is not an error profile saved by zeit")
        return cls(metric=d["metric"], window=int(d["window"]), calibration=d["calibration"],
                   threshold=float(d["threshold"]), points=pd.DataFrame(d["points"]), curve=d["curve"])

    def plot(self, ax: Any = None) -> Any:
        """The windows' metric against their DI, the curve and the AOA's threshold; returns the matplotlib axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(self.points["DI"], self.points["metric"], s=8, color="0.5", label=f"windows of {self.window}")
        grid = np.linspace(0, max(self.threshold, float(self.points["DI"].max())), 200)
        ax.plot(grid, self(grid), color="C0", label=self.calibration)
        ax.axvline(self.threshold, color="C3", linestyle="--", linewidth=1, label="AOA threshold")
        ax.set_xlabel("dissimilarity index (DI)")
        ax.set_ylabel(self.metric)
        ax.legend()
        return ax

    def __repr__(self) -> str:
        how = (f"edf {self.curve['edf']:.3g}" if self.calibration == "monotone"
               else f"slope {self.curve['slope']:.4g}")
        return (f"ErrorProfile({self.metric} by the DI: {len(self.points)} windows of {self.window} points, "
                f"{self.calibration} ({how}), threshold {self.threshold:.6g})")


def error_profile(model: Any = None, *, train: Any = None, observed: Any = None, predicted: Any = None,
                  folds: Any = None, weights: Any = None, distance: str = "euclidean", metric: Optional[str] = None,
                  window: int = 5, calibration: str = "monotone", k: int = 6, n_jobs: int = -1) -> ErrorProfile:
    """How a model's error grows with the dissimilarity of the data to its training data, as CAST's
    ``errorProfiles`` (Meyer & Pebesma 2021).

    model: a model from ``zeit.train_classifier(..., folds=...)``: its cross-validation (``zeit_cv_``), its training
        data and folds for the DI. Or, for a model of any kind (a network of Janus, say), ``train`` (the training
        predictors, points x predictors), ``observed`` and ``predicted`` (each training point's value and its
        prediction by a model fitted without its fold) and ``folds`` (as ``zeit.train_di``).
    weights, distance: as ``zeit.train_di``.
    metric: ``"rmse"`` or ``"mae"`` (values), ``"accuracy"`` (classes); default ``"accuracy"`` for a classifier or
        observations that are not numbers, else ``"rmse"``.
    window: the moving window's points.
    calibration: ``"monotone"`` (a P-spline that only grows, or only falls for the accuracy; CAST's ``scam``) or
        ``"linear"`` (CAST's ``lm``). k: the monotone curve's B-splines (CAST's ``k = 6``).

    Returns an ``ErrorProfile``: ``points`` (the windows' DI and metric), ``threshold``, the curve (call it on DI
    values), ``.predict(zeit.aoa(...))`` (the map of the error expected inside the AOA), ``.save()`` /
    ``ErrorProfile.load()`` (JSON) and ``.plot()``.
    """
    from ._aoa import _table, train_di

    if metric is not None and metric not in METRICS:
        raise ValueError(f"metric must be one of {tuple(METRICS)}, got {metric!r}")
    if calibration not in ("monotone", "linear"):
        raise ValueError(f"calibration must be 'monotone' or 'linear', got {calibration!r}")
    window = int(window)
    if window < 2:
        raise ValueError(f"window must be at least 2 points, got {window}")
    if model is not None and train is None:
        cv = getattr(model, "zeit_cv_", None)
        if cv is None:
            raise ValueError("the model has no cross-validation: train it with zeit.train_classifier(..., folds=...),"
                             " or give train=, observed=, predicted= and folds=")
        observed, predicted = cv["observed"].to_numpy(), cv["predicted"].to_numpy()
        td = train_di(model=model, weights=weights, folds=folds, distance=distance, n_jobs=n_jobs)
        if metric is None:
            metric = "accuracy" if hasattr(model, "classes_") else "rmse"
    else:
        if train is None or observed is None or predicted is None:
            raise ValueError("give a model from zeit.train_classifier(..., folds=...), or train=, observed= and "
                             "predicted=")
        values, _ = _table(train)
        keep = np.isfinite(values).all(axis=1)
        observed, predicted = np.asarray(observed), np.asarray(predicted)
        if not len(observed) == len(predicted) == len(values):
            raise ValueError(f"{len(observed)} observed and {len(predicted)} predicted values for "
                             f"{len(values)} training points")
        observed, predicted = observed[keep], predicted[keep]
        td = train_di(train, model=model, weights=weights, folds=folds, distance=distance, n_jobs=n_jobs)
        if metric is None:
            metric = "rmse" if np.issubdtype(observed.dtype, np.number) and not np.issubdtype(
                observed.dtype, np.integer) else "accuracy"
    di = np.asarray(td.train_di, dtype=np.float64)
    known = np.isfinite(di) & ~pd.isna(predicted) & (di <= td.threshold)
    order = np.flatnonzero(known)[np.argsort(di[known], kind="stable")]
    if order.size < window + 3:
        raise ValueError(f"{order.size} cross-validated points within the AOA's threshold: too few for windows of "
                         f"{window}")
    o, p, d = observed[order], predicted[order], di[order]
    middle = (window - 1) // 2
    rows = [(d[i + middle], _metric(metric, o[i:i + window], p[i:i + window])) for i in range(order.size - window + 1)]
    points = pd.DataFrame(rows, columns=["DI", "metric"])
    if calibration == "linear":
        slope, intercept = np.polyfit(points["DI"], points["metric"], 1)
        curve = {"intercept": float(intercept), "slope": float(slope)}
    else:
        from ._core import profile as core

        fit = core.fit(points["DI"].to_numpy(), points["metric"].to_numpy(), METRICS[metric], int(k))
        if fit is None:
            raise ValueError("the windows have fewer than 4 distinct DI values: too few for a curve")
        curve = {"lo": float(fit["lo"]), "hi": float(fit["hi"]), "coef": np.asarray(fit["coef"]),
                 "lambda": float(fit["lambda"]), "edf": float(fit["edf"])}
    return ErrorProfile(metric=metric, window=window, calibration=calibration, threshold=float(td.threshold),
                        points=points, curve=curve)
