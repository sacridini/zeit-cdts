"""From a cube to a map with the deep learning models: ``samples``, ``train``, ``predict``.

The same path as ``zeit.train_classifier``/``zeit.classify``: labelled points or polygons and
a cube give a ``SampleSet`` (pixels, or patches around the samples), ``train`` fits one of
the ``zeit.ai`` models on it with a ready-made loop, and ``predict`` classifies every pixel
of a cube into a georeferenced map. What prediction needs to check (band names, dates,
classes, the normalization, the patch size) travels with the model in ``zeit_meta_``, and
``save``/``load`` keep it with the weights.
"""

import copy
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from .._embeddings import check_same, embedding_meta

IGNORE = -1  # label of an unlabelled pixel in a patch (CrossEntropyLoss's ignore_index)


# ---------------------------------------------------------------------------
# The cube
# ---------------------------------------------------------------------------

def _cube(data: Any, nodata: Any = "auto", chunks: Any = None) -> Tuple[xr.DataArray, Optional[pd.DatetimeIndex]]:
    """The input as a ``(time, band, y, x)`` float cube with NaN for missing values, and its
    dates (``None`` for a pair of images, whose two steps have no dates)."""
    from .._load import load_raster
    from .._lt import _missing_values

    if isinstance(data, (tuple, list)):
        if len(data) != 2:
            raise ValueError(f"a pair of images (before, after) has two items, got {len(data)}")
        pair = [_one_date(d, nodata, chunks) for d in data]
        if pair[0].sizes["y"] != pair[1].sizes["y"] or pair[0].sizes["x"] != pair[1].sizes["x"] or \
                not np.allclose(pair[0].x.values, pair[1].x.values) or \
                not np.allclose(pair[0].y.values, pair[1].y.values):
            raise ValueError("the two images are on different grids; load the second with "
                             "zeit.load_raster(..., like=first)")
        if list(pair[0].band.values) != list(pair[1].band.values):
            raise ValueError("the two images have different bands")
        pair[1] = pair[1].assign_coords(x=pair[0].x, y=pair[0].y)
        cube = xr.concat(pair, dim="time", coords="minimal", compat="override", join="override")
        cube = cube.transpose("time", "band", "y", "x")
        crs = pair[0].rio.crs
        return (cube.rio.write_crs(crs) if crs is not None else cube), None
    da = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if "time" not in da.dims:
        raise ValueError("the zeit.ai models need a time series: a cube with a time dim "
                         "(or a pair of images, (before, after), for the Siamese detector)")
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"the cube needs y and x dims, got {da.dims}")
    if "band" not in da.dims:
        da = da.expand_dims(band=[str(da.name or "value")], axis=1)
    da = da.transpose("time", "band", "y", "x")
    crs = da.rio.crs
    out = _as_float(da, _missing_values(da, nodata))
    out = out.rio.write_crs(crs) if crs is not None and out.rio.crs is None else out
    return out, pd.DatetimeIndex(da.time.values)


def _one_date(data: Any, nodata: Any, chunks: Any) -> xr.DataArray:
    from .._load import load_raster
    from .._lt import _missing_values

    da = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError("each image of a pair is one date (a map or a stack of bands)")
        da = da.isel(time=0)
    if "band" not in da.dims:
        da = da.expand_dims(band=[str(da.name or "value")])
    da = da.transpose("band", "y", "x")
    crs = da.rio.crs
    out = _as_float(da, _missing_values(da, nodata))
    return out.rio.write_crs(crs) if crs is not None and out.rio.crs is None else out


def _as_float(da: xr.DataArray, missing: list) -> xr.DataArray:
    out = da if da.dtype == np.float32 else da.astype(np.float32)
    if missing:
        out = out.where(~da.isin(missing))
    return out


def _positions(times: Optional[pd.DatetimeIndex], n: int) -> List[float]:
    """Days since the first date (a pair of images: 0, 1)."""
    if times is None:
        return [float(i) for i in range(n)]
    days = (times - times[0]).total_seconds() / 86400.0
    return [float(d) for d in days]


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

class SampleSet(Dataset):
    """Labelled samples of a cube, ready for a PyTorch ``DataLoader``.

    Each item is a dict with ``x`` (normalized, missing values as 0: ``(time, band)`` for a
    pixel, ``(time, band, patch, patch)`` for a patch), ``y`` (the class index, or a
    ``(patch, patch)`` mask with ``-1`` where there is no label) and ``positions`` (days
    since the first date). ``train`` and ``val`` are the two parts of the split, with the
    same normalization (computed on ``train``). ``meta`` is what ``zeit.ai.train`` records
    in the model for ``zeit.ai.predict``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, is_val: np.ndarray, meta: Dict[str, Any],
                 indices: Optional[np.ndarray] = None):
        self.X = X
        self.y = y
        self.is_val = is_val
        self.meta = meta
        self.indices = np.arange(len(X)) if indices is None else indices
        self._low = np.asarray(meta["norm_low"], dtype=np.float32)
        self._range = np.asarray(meta["norm_range"], dtype=np.float32)
        self._positions = torch.tensor(meta["positions"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        k = self.indices[i]
        return {"x": torch.from_numpy(_normalize(self.X[k], self._low, self._range, patch=self.patch is not None)),
                "y": torch.as_tensor(self.y[k], dtype=torch.long),
                "positions": self._positions}

    @property
    def train(self) -> "SampleSet":
        return SampleSet(self.X, self.y, self.is_val, self.meta, self.indices[~self.is_val[self.indices]])

    @property
    def val(self) -> "SampleSet":
        return SampleSet(self.X, self.y, self.is_val, self.meta, self.indices[self.is_val[self.indices]])

    @property
    def classes(self) -> List[str]:
        return list(self.meta["classes"])

    @property
    def patch(self) -> Optional[int]:
        return self.meta["patch"]

    def __repr__(self) -> str:
        kind = f"patches of {self.patch}" if self.patch else "pixels"
        return (f"SampleSet({len(self)} {kind}: {len(self.train)} train, {len(self.val)} val; "
                f"{len(self.meta['times'] or [None, None])} dates x {len(self.meta['bands'])} bands; "
                f"classes {self.classes})")


def _normalize(x: np.ndarray, low: np.ndarray, rng: np.ndarray, patch: bool) -> np.ndarray:
    """(x - q02) / (q98 - q02) per band (the band axis is 1 of (time, band[, y, x]) and
    -1 of (n, time, band)), missing values as 0."""
    shape = (1, -1, 1, 1) if patch else (1, -1)
    out = (x - low.reshape(shape)) / rng.reshape(shape)
    return np.nan_to_num(out, nan=0.0).astype(np.float32, copy=False)


def samples(
    data: Any,
    samples: Any,
    *,
    label: str = "class",
    patch: Optional[int] = None,
    split: float = 0.2,
    split_by: str = "block",
    block_size: int = 64,
    seed: int = 42,
    nodata: Any = "auto",
) -> SampleSet:
    """Labelled samples of a cube, for ``zeit.ai.train``.

    Parameters
    ----------
    data
        A ``(time, y, x)`` or ``(time, band, y, x)`` cube (in memory or dask), anything
        ``zeit.load_raster`` reads, or a pair of images ``(before, after)`` for the Siamese
        change detector.
    samples
        Points or polygons with their class: a ``GeoDataFrame`` or a vector file,
        reprojected when in another CRS.
    label
        The column holding the class.
    patch
        ``None`` (default): one sample per pixel, ``(time, band)``, for the pixel models
        (``TempCNN``, ``LightTAE``); a polygon gives every pixel it covers. A size in
        pixels: a ``(time, band, patch, patch)`` window around each sample for the patch
        models (``UTAE``, ``SiameseChangeDetector``, ``GeoFoundationViT``), with every
        labelled pixel in it (points and polygons rasterized) and ``-1`` elsewhere.
    split
        Share of the samples kept for validation (``0``: none).
    split_by
        ``"block"`` (default): whole blocks of ``block_size`` pixels go to validation, so
        that it measures how the model does away from what it was trained on (neighbouring
        pixels are alike: random pixels on both sides would measure that). ``"random"``:
        samples drawn at random.
    block_size
        Side of the validation blocks, in pixels.
    seed
        Seed of the split.
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``.

    Returns
    -------
    SampleSet
        A PyTorch ``Dataset`` (``.train`` and ``.val`` its parts). The bands are
        normalized by their 2% and 98% quantiles in the training samples (as ``sits``
        does) and missing values become 0.

    Examples
    --------
    >>> s = zeit.ai.samples(cube, "points.gpkg", label="class")
    >>> s.train, s.val
    """
    import geopandas as gpd

    from .._classify_api import _points

    if split_by not in ("block", "random"):
        raise ValueError(f"split_by must be 'block' or 'random', got {split_by!r}")
    if not 0 <= split < 1:
        raise ValueError(f"split must be in [0, 1), got {split!r}")
    if patch is not None and int(patch) < 2:
        raise ValueError(f"patch must be at least 2 pixels (or None), got {patch!r}")
    cube, times = _cube(data, nodata)
    if isinstance(samples, gpd.GeoDataFrame):
        geoms = samples
    else:
        geoms = gpd.read_file(samples)
    if label not in geoms.columns:
        raise ValueError(f"samples have no column {label!r}; they have {list(geoms.columns)}")
    if geoms.crs is not None and cube.rio.crs is not None and geoms.crs != cube.rio.crs:
        geoms = geoms.to_crs(cube.rio.crs)
    geoms = geoms[geoms[label].notna() & ~geoms.geometry.is_empty]
    names = sorted({str(v) for v in geoms[label]})
    if len(names) < 2:
        raise ValueError(f"the samples need at least two classes, got {names}")
    ids = np.array([names.index(str(v)) for v in geoms[label]], dtype=np.int64)
    from .._warp import transform_of

    transform = transform_of(cube)
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    labels = _rasterize(geoms.geometry, ids, transform, (ny, nx))

    if patch is None:
        rows, cols = np.nonzero(labels >= 0)
        if not len(rows):
            raise ValueError("no sample falls inside the data")
        values = cube.isel(y=xr.DataArray(rows, dims="sample"), x=xr.DataArray(cols, dims="sample"))
        X = np.asarray(values.transpose("sample", "time", "band").values, dtype=np.float32)
        y = labels[rows, cols].astype(np.int64)
        keep = np.isfinite(X).any(axis=(1, 2))
        X, y, rows, cols = X[keep], y[keep], rows[keep], cols[keep]
    else:
        p = int(patch)
        origins = sorted({o for g in geoms.geometry for o in _origins(g, transform, p, (ny, nx))})
        if not origins:
            raise ValueError("no sample falls inside the data")
        rows = np.array([r + p // 2 for r, _ in origins])
        cols = np.array([c + p // 2 for _, c in origins])
        X = np.stack([_window(cube, r, c, p) for r, c in origins])
        y = np.stack([_window_labels(labels, r, c, p) for r, c in origins])
    if not len(X):
        raise ValueError("no sample has data")

    is_val = _split(rows, cols, split, split_by, block_size, seed)
    train_values = X[~is_val] if (~is_val).any() else X
    by_band = np.moveaxis(train_values, 2, 0).reshape(train_values.shape[2], -1)
    low = np.nanquantile(by_band, 0.02, axis=1) if np.isfinite(by_band).any() else np.zeros(len(by_band))
    high = np.nanquantile(by_band, 0.98, axis=1) if np.isfinite(by_band).any() else np.ones(len(by_band))
    rng = np.where(high > low, high - low, 1.0)
    meta = {
        "bands": [str(b) for b in cube.band.values],
        "times": None if times is None else [t.isoformat() for t in times],
        "positions": _positions(times, cube.sizes["time"]),
        "classes": names,
        "norm_low": [float(v) for v in low],
        "norm_range": [float(v) for v in rng],
        "patch": None if patch is None else int(patch),
        "label": label,
        "embedding": embedding_meta(data),
    }
    return SampleSet(X, y, is_val, meta)


def _rasterize(geometries, ids: np.ndarray, transform, shape) -> np.ndarray:
    """Class index of every cell a geometry covers (points: their cell), -1 elsewhere."""
    from rasterio.features import rasterize

    out = np.full(shape, IGNORE, dtype=np.int64)
    polys = [(g, int(i)) for g, i in zip(geometries, ids) if g.geom_type != "Point"]
    if polys:
        burnt = rasterize(polys, out_shape=shape, transform=transform, fill=IGNORE, dtype="int32")
        out = np.where(burnt >= 0, burnt, out)
    for g, i in zip(geometries, ids):
        if g.geom_type == "Point":
            r, c = _center(g, transform)
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                out[r, c] = int(i)
    return out


def _origins(geometry, transform, p: int, shape) -> List[Tuple[int, int]]:
    """Top-left pixels of the windows of a sample: one centred on a point or a small
    polygon; a polygon larger than a window is tiled by windows that cover it."""
    ny, nx = shape
    r, c = _center(geometry, transform)
    if geometry.geom_type != "Point":
        x0, y0, x1, y1 = geometry.bounds
        c0, r0 = ~transform * (x0, y1)
        c1, r1 = ~transform * (x1, y0)
        r0, r1 = sorted((int(np.floor(r0)), int(np.ceil(r1))))
        c0, c1 = sorted((int(np.floor(c0)), int(np.ceil(c1))))
        r0, r1, c0, c1 = max(r0, 0), min(r1, ny), max(c0, 0), min(c1, nx)
        if r1 - r0 > p or c1 - c0 > p:
            rows = _Windows._starts(r1 - r0, p, p) if r1 - r0 > p else [(r0 + r1) // 2 - p // 2 - r0]
            cols = _Windows._starts(c1 - c0, p, p) if c1 - c0 > p else [(c0 + c1) // 2 - p // 2 - c0]
            return [(r0 + a, c0 + b) for a in rows for b in cols]
    if not (0 <= r < ny and 0 <= c < nx):
        return []
    return [(r - p // 2, c - p // 2)]


def _center(geometry, transform) -> Tuple[int, int]:
    point = geometry if geometry.geom_type == "Point" else geometry.representative_point()
    col, row = ~transform * (point.x, point.y)
    return int(np.floor(row)), int(np.floor(col))


def _window(cube: xr.DataArray, r0: int, c0: int, p: int) -> np.ndarray:
    """(time, band, p, p) from (r0, c0), NaN outside the cube."""
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    out = np.full((cube.sizes["time"], cube.sizes["band"], p, p), np.nan, dtype=np.float32)
    rs, re, cs, ce = max(r0, 0), min(r0 + p, ny), max(c0, 0), min(c0 + p, nx)
    if rs < re and cs < ce:
        out[:, :, rs - r0:re - r0, cs - c0:ce - c0] = cube.isel(y=slice(rs, re), x=slice(cs, ce)).values
    return out


def _window_labels(labels: np.ndarray, r0: int, c0: int, p: int) -> np.ndarray:
    ny, nx = labels.shape
    out = np.full((p, p), IGNORE, dtype=np.int64)
    rs, re, cs, ce = max(r0, 0), min(r0 + p, ny), max(c0, 0), min(c0 + p, nx)
    if rs < re and cs < ce:
        out[rs - r0:re - r0, cs - c0:ce - c0] = labels[rs:re, cs:ce]
    return out


def _split(rows: np.ndarray, cols: np.ndarray, split: float, split_by: str, block_size: int, seed: int) -> np.ndarray:
    n = len(rows)
    is_val = np.zeros(n, dtype=bool)
    if split <= 0 or n < 2:
        return is_val
    rng = np.random.default_rng(seed)
    if split_by == "random":
        is_val[rng.choice(n, max(1, int(round(split * n))), replace=False)] = True
        return is_val
    blocks = (rows // int(block_size)) * 1_000_003 + cols // int(block_size)
    unique = rng.permutation(np.unique(blocks))
    target = split * n
    taken = 0
    for b in unique:
        if taken >= target:
            break
        members = blocks == b
        if taken + members.sum() > target * 1.5 and taken > 0:
            continue
        is_val |= members
        taken += members.sum()
    if is_val.all():   # one block holds everything: fall back to random samples
        is_val[:] = False
        is_val[rng.choice(n, max(1, int(round(split * n))), replace=False)] = True
    return is_val


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------

class _Spec:
    """How zeit builds a model from the samples and calls it."""

    def __init__(self, kind: str, build: Callable[[Dict[str, Any]], Dict[str, Any]],
                 forward: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
                 fixed_times: bool, check: Optional[Callable[[Dict[str, Any], nn.Module], None]] = None):
        self.kind = kind            # "pixel" or "patch"
        self.build = build          # meta -> the keyword arguments zeit fills in
        self.forward = forward      # (model, x, positions) -> logits
        self.fixed_times = fixed_times
        self.check = check


def _specs() -> Dict[str, _Spec]:
    def n(meta, what):
        return {"bands": len(meta["bands"]), "times": len(meta["positions"]), "classes": len(meta["classes"])}[what]

    def utae_check(meta, model):
        halvings = 2 ** (model.n_stages - 1)
        if meta["patch"] % halvings:
            raise ValueError(f"this UTAE halves the patch {model.n_stages - 1} times: use a patch that is a "
                             f"multiple of {halvings}, got {meta['patch']}")

    def siamese_check(meta, model):
        if len(meta["positions"]) != 2:
            raise ValueError("SiameseChangeDetector takes a pair of images: samples of (before, after)")
        if meta["patch"] % 2:
            raise ValueError(f"SiameseChangeDetector halves the patch once: use an even patch, got {meta['patch']}")

    return {
        "TempCNN": _Spec(
            "pixel", lambda m: {"in_channels": n(m, "bands"), "n_times": n(m, "times"), "num_classes": n(m, "classes")},
            lambda model, x, pos: model(x.permute(0, 2, 1)), fixed_times=True),
        "LightTAE": _Spec(
            "pixel", lambda m: {"n_bands": n(m, "bands"), "day_offsets": list(m["positions"]),
                                "n_labels": n(m, "classes")},
            lambda model, x, pos: model(x), fixed_times=True),
        "UTAE": _Spec(
            "patch", lambda m: {"input_dim": n(m, "bands"), "out_conv": [32, n(m, "classes")]},
            lambda model, x, pos: model(x, batch_positions=pos), fixed_times=False, check=utae_check),
        "SiameseChangeDetector": _Spec(
            "patch", lambda m: {"in_channels": n(m, "bands"), "num_classes": n(m, "classes")},
            lambda model, x, pos: model(x[:, 0], x[:, 1]), fixed_times=True, check=siamese_check),
        "GeoFoundationViT": _Spec(
            "patch", lambda m: {"num_classes": n(m, "classes")},
            lambda model, x, pos: model(x.permute(0, 2, 1, 3, 4)), fixed_times=True),
    }


def _model_class(name: str):
    from . import foundation, siamese, tempcnn, utae

    for module in (tempcnn, utae, siamese, foundation):
        if hasattr(module, name):
            return getattr(module, name)
    raise ValueError(f"unknown zeit.ai model {name!r}")


def _spec_of(model: Any) -> Tuple[str, _Spec]:
    specs = _specs()
    name = model.__name__ if isinstance(model, type) else type(model).__name__
    if name not in specs:
        raise ValueError(f"zeit.ai.train and predict work with {', '.join(specs)}; got {name}. "
                         "For another model, use the SampleSet with your own training loop.")
    return name, specs[name]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _loss(loss: Union[str, Callable], gamma: float = 2.0) -> Callable:
    if callable(loss):
        return loss
    if loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=IGNORE)
    if loss == "focal":
        def focal(logits, target):
            ce = F.cross_entropy(logits, target, ignore_index=IGNORE, reduction="none")
            valid = target != IGNORE
            pt = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce)[valid].mean()
        return focal
    raise ValueError(f"loss must be 'ce', 'focal' or a function (logits, target) -> loss, got {loss!r}")


def _batch(batch: Dict[str, torch.Tensor], device: torch.device):
    x = batch["x"].to(device)
    pos = batch["positions"].to(device)
    return x, pos, batch["y"].to(device)


# Steps before early stopping may end training: validation runs with the running statistics
# of batch normalization, which need tens of steps to settle (momentum 0.1: 0.9 ** 50 < 1%);
# until then the validation loss can rise while the model learns.
_MIN_STEPS = 50


def train(
    model: Any,
    samples: SampleSet,
    *,
    epochs: int = 50,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    loss: Union[str, Callable] = "ce",
    patience: Optional[int] = 10,
    device: str = "auto",
    seed: int = 42,
    verbose: bool = False,
    **model_kwargs: Any,
) -> nn.Module:
    """Train a ``zeit.ai`` model on a ``SampleSet``.

    Parameters
    ----------
    model
        A model class (``TempCNN``, ``LightTAE``, ``UTAE``, ``SiameseChangeDetector``,
        ``GeoFoundationViT``), built with the number of bands, dates and classes of the
        samples (and ``model_kwargs``), or an instance already built.
    samples
        From ``zeit.ai.samples``: pixels for the pixel models, patches for the others.
    epochs, batch_size, lr, weight_decay
        Adam on ``batch_size`` samples at a time (default 64 pixels, or 8 patches: a patch
        holds many labelled pixels, and few patches per step leave more steps per epoch),
        for at most ``epochs`` passes.
    loss
        ``"ce"`` (cross-entropy, default), ``"focal"`` (focal loss, for imbalanced classes)
        or a function ``(logits, target) -> loss`` (``target`` is ``-1`` where unlabelled).
    patience
        Stop after this many epochs without a lower validation loss (once the model has
        taken 50 steps: batch normalization's statistics, which validation runs with, need
        them), and keep the best epoch's weights (``None``: train every epoch, keep the
        last).
    device
        ``"auto"`` (the GPU if there is one), ``"cpu"``, ``"cuda"``...
    seed
        Seed of the initial weights and of the order of the samples.
    verbose
        Print each epoch's losses and validation accuracy.
    **model_kwargs
        For a model class: its own arguments (e.g. ``dropout_rate``), next to those zeit
        fills in.

    Returns
    -------
    The trained model in evaluation mode, with ``zeit_meta_`` (what ``zeit.ai.predict``
    checks: bands, dates, classes, normalization, patch, the model's class and arguments)
    and ``zeit_history_`` (per epoch: ``train_loss``, ``val_loss``, ``val_accuracy``).

    Examples
    --------
    >>> s = zeit.ai.samples(cube, "points.gpkg", label="class")
    >>> model = zeit.ai.train(zeit.ai.TempCNN, s, epochs=50)
    >>> classes = zeit.ai.predict(model, cube)
    """
    name, spec = _spec_of(model)
    meta = dict(samples.meta)
    if (spec.kind == "patch") != (meta["patch"] is not None):
        want = "patches (zeit.ai.samples(..., patch=64))" if spec.kind == "patch" else "pixels (patch=None)"
        raise ValueError(f"{name} trains on {want}")
    if batch_size is None:
        batch_size = 64 if spec.kind == "pixel" else 8
    torch.manual_seed(seed)
    if isinstance(model, type):
        kwargs = {**spec.build(meta), **model_kwargs}
        model = model(**kwargs)
    else:
        if model_kwargs:
            raise ValueError("model arguments are for a model class, not an instance")
        kwargs = None
    if spec.check is not None:
        spec.check(meta, model)
    dev = _device(device)
    model = model.to(dev)
    criterion = _loss(loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_set, val_set = samples.train, samples.val
    if not len(train_set):
        raise ValueError("no training samples")
    generator = torch.Generator().manual_seed(seed)
    # drop only a last batch of one sample, which batch normalization cannot train on
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator,
                        drop_last=len(train_set) % batch_size == 1 and len(train_set) > 1)
    history: List[Dict[str, float]] = []
    best, best_state, waited, steps = np.inf, None, 0, 0
    for epoch in range(int(epochs)):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            x, pos, y = _batch(batch, dev)
            optimizer.zero_grad()
            out = criterion(spec.forward(model, x, pos), y)
            out.backward()
            optimizer.step()
            steps += 1
            total += out.item() * len(y)
            count += len(y)
        row = {"epoch": epoch + 1, "train_loss": total / max(count, 1)}
        if len(val_set):
            row.update(_evaluate(model, spec, val_set, criterion, batch_size, dev))
        history.append(row)
        if verbose:
            print(", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        watched = row.get("val_loss", row["train_loss"])
        if watched < best - 1e-7:
            best, waited = watched, 0
            if patience is not None:
                best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if patience is not None and waited >= patience and steps >= _MIN_STEPS:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    watched = [r.get("val_loss", r["train_loss"]) for r in history]
    best_epoch = int(np.argmin(watched)) + 1 if watched and best_state is not None else len(history)
    meta.update(model=name, model_kwargs=kwargs, best_epoch=best_epoch)
    model.zeit_meta_ = meta
    model.zeit_history_ = history
    return model


@torch.no_grad()
def _evaluate(model, spec, dataset, criterion, batch_size, dev) -> Dict[str, float]:
    model.eval()
    total, count, right, labelled = 0.0, 0, 0, 0
    for batch in DataLoader(dataset, batch_size=batch_size):
        x, pos, y = _batch(batch, dev)
        logits = spec.forward(model, x, pos)
        total += float(criterion(logits, y)) * len(y)
        count += len(y)
        valid = y != IGNORE
        right += int((logits.argmax(1) == y)[valid].sum())
        labelled += int(valid.sum())
    return {"val_loss": total / max(count, 1), "val_accuracy": right / max(labelled, 1)}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_MODEL_LOCK = threading.Lock()  # dask computes blocks in threads; the model runs one block at a time


def predict(
    model: nn.Module,
    data: Any,
    *,
    batch_size: int = 256,
    overlap: float = 0.25,
    probability: bool = False,
    device: str = "auto",
    nodata: Any = "auto",
    chunks: Any = None,
) -> xr.Dataset:
    """Classify every pixel of a cube with a model trained by ``zeit.ai.train``.

    Parameters
    ----------
    model
        A trained model with ``zeit_meta_`` (``zeit.ai.train`` or ``zeit.ai.load``).
    data
        The cube to classify, as in ``zeit.ai.samples`` (or a pair of images for the
        Siamese detector). Its bands are taken by name and normalized as the training
        samples were. ``TempCNN``, ``LightTAE`` and the Siamese and ViT models need as
        many dates as they were trained on; ``UTAE`` takes any dates (its positions come
        from them). A lazy cube stays lazy, computed block by block.
    batch_size
        Pixels (or windows) through the model at a time.
    overlap
        Patch models: the share of a window its neighbours overlap. The image is covered
        by windows of the training patch size; each pixel's class probabilities are the
        average of the windows over it, weighted down towards the windows' edges, so the
        windows leave no seams.
    probability
        Also return each class's probability.
    device
        ``"auto"`` (the GPU if there is one), ``"cpu"``, ``"cuda"``...
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``.
    chunks
        A raster path: read lazily with these chunks.

    Returns
    -------
    xarray.Dataset
        As ``zeit.classify``'s: ``label (y, x)`` (``1`` for the first class, ``0`` where the
        pixel has no data at any date; the names in ``flag_meanings``) and, with
        ``probability=True``, ``probability (class, y, x)``. Georeferenced as the input.
    """
    meta = getattr(model, "zeit_meta_", None)
    if meta is None:
        raise ValueError("this model has no zeit_meta_: train it with zeit.ai.train (or load it with zeit.ai.load)")
    name, spec = _spec_of(model)
    check_same(meta.get("embedding"), data, name)
    cube, times = _cube(data, nodata, chunks)
    cube = _bands(cube, meta["bands"])
    n_times = cube.sizes["time"]
    if spec.fixed_times and n_times != len(meta["positions"]):
        raise ValueError(f"{name} was trained on {len(meta['positions'])} dates; this cube has {n_times}. "
                         "It needs the same number of dates (e.g. the same composites in another year).")
    positions = torch.tensor(_positions(times, n_times) if not spec.fixed_times or times is None
                             else meta["positions"], dtype=torch.float32)
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")
    dev = _device(device)
    model = model.to(dev).eval()
    low = np.asarray(meta["norm_low"], dtype=np.float32)
    rng = np.asarray(meta["norm_range"], dtype=np.float32)
    k = len(meta["classes"])
    run = _Runner(model, spec, positions, low, rng, k, batch_size, dev)

    arr = cube.data
    lazy = hasattr(arr, "dask")
    if spec.kind == "pixel":
        if lazy:
            import dask.array as da

            arr = arr.rechunk({0: -1, 1: -1})
            out = da.map_blocks(run.pixels, arr, dtype=np.float32, drop_axis=[0],
                                chunks=((k + 1,),) + arr.chunks[2:])
        else:
            out = run.pixels(np.asarray(arr))
    else:
        p = int(meta["patch"])
        grid = _Windows(cube.sizes["y"], cube.sizes["x"], p, overlap)
        if lazy:
            out = _lazy_patches(arr.rechunk({0: -1, 1: -1}), run, grid, k)
        else:
            out = run.patches(np.asarray(arr), grid, 0, 0, (0, cube.sizes["y"]), (0, cube.sizes["x"]))
    return _result(out, cube, meta["classes"], probability, type(model).__name__)


def _bands(cube: xr.DataArray, bands: Sequence[str]) -> xr.DataArray:
    have = [str(b) for b in cube.band.values]
    if have == list(bands):
        return cube
    missing = [b for b in bands if b not in have]
    if missing:
        if len(have) == len(bands) and len(bands) == 1:
            return cube   # one band: its name may differ (a file named differently)
        raise ValueError(f"the cube lacks bands the model was trained on: {missing} (it has {have})")
    return cube.sel(band=list(bands))


class _Runner:
    """The model on numpy blocks: normalization, batches, softmax."""

    def __init__(self, model, spec, positions, low, rng, k, batch_size, dev):
        self.model, self.spec, self.positions = model, spec, positions
        self.low, self.rng, self.k, self.batch_size, self.dev = low, rng, k, batch_size, dev

    @torch.no_grad()
    def _probs(self, x: np.ndarray) -> np.ndarray:
        """Softmax of the model on normalized samples (n, ...): (n, k[, h, w])."""
        out = []
        with _MODEL_LOCK:
            for i in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(x[i:i + self.batch_size]).to(self.dev)
                pos = self.positions.to(self.dev).unsqueeze(0).expand(len(xb), -1)
                out.append(torch.softmax(self.spec.forward(self.model, xb, pos), dim=1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.k), dtype=np.float32)

    def pixels(self, block: np.ndarray) -> np.ndarray:
        t, b, h, w = block.shape
        series = np.moveaxis(block.reshape(t, b, h * w), -1, 0)           # (n, time, band)
        has = np.isfinite(series).any(axis=(1, 2))
        out = np.zeros((self.k + 1, h * w), dtype=np.float32)
        out[1:] = np.nan
        if has.any():
            probs = self._probs(_normalize(series[has], self.low, self.rng, patch=False))
            out[0, has] = probs.argmax(axis=1) + 1
            out[1:, has] = probs.T
        return out.reshape(self.k + 1, h, w)

    def patches(self, region: np.ndarray, grid: "_Windows", r_off: int, c_off: int,
                core_rows: Tuple[int, int], core_cols: Tuple[int, int]) -> np.ndarray:
        """Label and probabilities of the core rows/cols (image coordinates) from ``region``
        (the image from (r_off, c_off), covering every window over the core)."""
        p = grid.p
        r0, r1 = core_rows
        c0, c1 = core_cols
        acc = np.zeros((self.k, r1 - r0, c1 - c0), dtype=np.float64)
        wsum = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
        windows = [(wr, wc) for wr in grid.starts_y if wr < r1 and wr + p > r0
                   for wc in grid.starts_x if wc < c1 and wc + p > c0]
        weight = grid.weight
        for i in range(0, len(windows), self.batch_size):
            chunk = windows[i:i + self.batch_size]
            x = np.stack([self._cut(region, wr - r_off, wc - c_off, p) for wr, wc in chunk])
            probs = self._probs(_normalize(x, self.low, self.rng, patch=True))
            for (wr, wc), pr in zip(chunk, probs):
                ys, ye = max(wr, r0), min(wr + p, r1)
                xs, xe = max(wc, c0), min(wc + p, c1)
                w = weight[ys - wr:ye - wr, xs - wc:xe - wc]
                acc[:, ys - r0:ye - r0, xs - c0:xe - c0] += pr[:, ys - wr:ye - wr, xs - wc:xe - wc] * w
                wsum[ys - r0:ye - r0, xs - c0:xe - c0] += w
        probs = (acc / np.maximum(wsum, 1e-12)).astype(np.float32)
        core = region[:, :, r0 - r_off:r1 - r_off, c0 - c_off:c1 - c_off]
        has = np.isfinite(core).any(axis=(0, 1))
        out = np.zeros((self.k + 1, r1 - r0, c1 - c0), dtype=np.float32)
        out[0] = np.where(has, probs.argmax(axis=0) + 1, 0)
        out[1:] = np.where(has, probs, np.nan)
        return out

    @staticmethod
    def _cut(region: np.ndarray, r: int, c: int, p: int) -> np.ndarray:
        t, b, h, w = region.shape
        out = np.full((t, b, p, p), np.nan, dtype=np.float32)
        rs, re, cs, ce = max(r, 0), min(r + p, h), max(c, 0), min(c + p, w)
        out[:, :, rs - r:re - r, cs - c:ce - c] = region[:, :, rs:re, cs:ce]
        return out


class _Windows:
    """The windows that cover an image: starts every ``p * (1 - overlap)`` pixels from the
    top-left corner, the last one against the far edge; weights that fall towards a
    window's edges (a pyramid, never 0)."""

    def __init__(self, h: int, w: int, p: int, overlap: float):
        self.p = p
        step = max(1, int(round(p * (1 - overlap))))
        self.starts_y = self._starts(h, p, step)
        self.starts_x = self._starts(w, p, step)
        ramp = np.minimum(np.arange(1, p + 1), np.arange(p, 0, -1)).astype(np.float64)
        self.weight = np.outer(ramp, ramp) / ramp.max() ** 2

    @staticmethod
    def _starts(n: int, p: int, step: int) -> List[int]:
        if n <= p:
            return [0]
        starts = list(range(0, n - p + 1, step))
        if starts[-1] != n - p:
            starts.append(n - p)
        return starts


def _lazy_patches(arr, run: _Runner, grid: _Windows, k: int):
    """Patch prediction block by block: each block of the output reads its own rows and
    columns plus a window's width around them, and takes the windows over it from the
    image-wide grid, so the result is the one in memory."""
    import dask
    import dask.array as da

    p = grid.p
    h, w = arr.shape[2], arr.shape[3]
    row_edges = np.cumsum((0,) + arr.chunks[2])
    col_edges = np.cumsum((0,) + arr.chunks[3])
    blocks = []
    for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
        line = []
        for c0, c1 in zip(col_edges[:-1], col_edges[1:]):
            er0, er1 = max(0, r0 - p), min(h, r1 + p)
            ec0, ec1 = max(0, c0 - p), min(w, c1 + p)
            region = arr[:, :, er0:er1, ec0:ec1]
            part = dask.delayed(run.patches)(region, grid, int(er0), int(ec0), (int(r0), int(r1)), (int(c0), int(c1)))
            line.append(da.from_delayed(part, shape=(k + 1, int(r1 - r0), int(c1 - c0)), dtype=np.float32))
        blocks.append(line)
    return da.block(blocks)


def _result(out, cube: xr.DataArray, classes: Sequence[str], probability: bool, algorithm: str) -> xr.Dataset:
    k = len(classes)
    coords = {c: cube.coords[c] for c in ("y", "x", "spatial_ref") if c in cube.coords}
    coords["class"] = list(classes)
    label_dtype = np.uint8 if k < 255 else np.uint16
    label = xr.DataArray(out[0].astype(label_dtype), dims=("y", "x"),
                         attrs={"long_name": "class", "flag_values": list(range(1, k + 1)),
                                "flag_meanings": " ".join(str(c).replace(" ", "_") for c in classes)})
    variables = {"label": label}
    if probability:
        variables["probability"] = (("class", "y", "x"), out[1:])
    ds = xr.Dataset(variables, coords=coords, attrs={"algorithm": algorithm})
    if cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
        ds["label"] = ds["label"].rio.write_nodata(0, encoded=False)
    return ds


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def save(model: nn.Module, path: Any) -> None:
    """Save a model trained by ``zeit.ai.train``: its weights, ``zeit_meta_`` and history.

    ``zeit.ai.load(path)`` builds it again (a model built by ``train`` from its class; for
    an instance you built, ``load`` takes one to put the weights in).
    """
    meta = getattr(model, "zeit_meta_", None)
    if meta is None:
        raise ValueError("this model has no zeit_meta_: train it with zeit.ai.train first")
    torch.save({"format": "zeit.ai/1", "meta": meta, "history": getattr(model, "zeit_history_", []),
                "state_dict": model.state_dict()}, path)


def load(path: Any, model: Optional[nn.Module] = None, device: str = "cpu") -> nn.Module:
    """Load a model saved by ``zeit.ai.save``, ready for ``zeit.ai.predict``.

    Parameters
    ----------
    path
        The file ``save`` wrote.
    model
        An instance to put the weights in, for a model that was not built by ``train``
        from its class (``train`` knows the arguments of the models it builds).
    device
        Where to load the weights.
    """
    saved = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != "zeit.ai/1":
        raise ValueError(f"{path} was not written by zeit.ai.save")
    meta = saved["meta"]
    if model is None:
        if meta.get("model_kwargs") is None:
            raise ValueError("this model was trained from an instance; pass one like it as model=")
        model = _model_class(meta["model"])(**meta["model_kwargs"])
    model.load_state_dict(saved["state_dict"])
    model.zeit_meta_ = meta
    model.zeit_history_ = saved.get("history", [])
    return model.to(device).eval()
