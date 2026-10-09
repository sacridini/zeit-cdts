"""``zeit.train_classifier`` and ``zeit.classify``: supervised classification of any result.

The features of a pixel are what it holds off ``y``/``x``: the bands of a map, the dates
(and bands) of a cube, the variables of a Dataset, or the CCDC model of the segment that
covers a date (its coefficients, with the intercept at that date, and its RMSE, as CCDC
classifications do). A scikit-learn model is trained on the features at sample points and
predicts every pixel; the result is a georeferenced class map.
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import xarray as xr


def features(data: Any, *, date: Any = None) -> xr.DataArray:
    """The features of every pixel as a ``(feature, y, x)`` DataArray, named in ``feature``.

    ``data``: a ``DataArray`` (``(y, x)``, ``(band, y, x)``, ``(time[, band], y, x)``...), a
    ``Dataset`` (each map a feature; a variable with another dim gives one feature per
    value of it), a ``zeit.ccdc`` result with ``date``, or anything ``load_raster`` reads.
    """
    if isinstance(data, xr.Dataset) and "coefs" in data and "t_start" in data:
        if date is None:
            raise ValueError("CCDC segments: give date= (the features are the model of the segment on it)")
        return _ccdc_features(data, date)
    if date is not None:
        raise ValueError("date= is for CCDC segments")
    if isinstance(data, xr.Dataset):
        parts = []
        for name, var in data.data_vars.items():
            if not {"y", "x"} <= set(var.dims):
                continue
            parts.append(_flatten(var, prefix=str(name)))
        if not parts:
            raise ValueError("the Dataset has no maps to use as features")
        out = xr.concat(parts, dim="feature")
        if data.rio.crs is not None and out.rio.crs is None:
            out = out.rio.write_crs(data.rio.crs)
        return out
    if not isinstance(data, xr.DataArray):
        from ._load import load_raster

        data = load_raster(data)
    return _flatten(data, prefix=None)


def _flatten(da: xr.DataArray, prefix: Optional[str]) -> xr.DataArray:
    """(..., y, x) -> (feature, y, x), the feature named by the coordinates of the other dims."""
    others = [d for d in da.dims if d not in ("y", "x")]
    crs = da.rio.crs if "y" in da.dims else None
    if not others:
        out = da.expand_dims(feature=[prefix or str(da.name or "value")])
    else:
        labels: List[List[str]] = []
        for d in others:
            if d in da.coords:
                values = da.coords[d].values
                if np.issubdtype(values.dtype, np.datetime64):
                    labels.append(list(pd.DatetimeIndex(values).strftime("%Y-%m-%d")))
                else:
                    labels.append([str(v) for v in values])
            else:
                labels.append([str(i) for i in range(da.sizes[d])])
        names = labels[0]
        for more in labels[1:]:
            names = [f"{a}_{b}" for a in names for b in more]
        if prefix is not None:
            names = [f"{prefix}_{n}" for n in names]
        stacked = da.transpose(*others, "y", "x")
        values = stacked.data.reshape((-1,) + stacked.shape[-2:])
        coords = {c: stacked.coords[c] for c in ("y", "x", "spatial_ref") if c in stacked.coords}
        coords["feature"] = names
        out = xr.DataArray(values, dims=("feature", "y", "x"), coords=coords)
    out = out.transpose("feature", "y", "x")
    out = out.drop_vars([c for c in out.coords if c not in ("feature", "y", "x", "spatial_ref")])
    if crs is not None and out.rio.crs is None:
        out = out.rio.write_crs(crs)
    return out


def _ccdc_features(segments: xr.Dataset, date: Any) -> xr.DataArray:
    """Each pixel's CCDC model on ``date``: the coefficients of every band (the intercept
    moved to that date, a0 + c1 t, as CCDC classifications use it) and the RMSE."""
    from ._ccdc_api import _DATENUM_OFFSET, segment_at

    stamp, chosen = segment_at(segments, date)
    t = float(stamp.toordinal() + _DATENUM_OFFSET)
    coefs = (segments.coefs.fillna(0) * chosen).sum("segment")
    rmse = (segments.rmse.fillna(0) * chosen).sum("segment")
    coefs = coefs.where(coefs.coef != "a0", coefs.sel(coef="a0") + coefs.sel(coef="c1") * t)
    has = segments.n_segments > 0
    parts = [_flatten(coefs.where(has).transpose("band", "coef", "y", "x"), prefix=None),
             _flatten(rmse.where(has).transpose("band", "y", "x"), prefix=None)]
    parts[1] = parts[1].assign_coords(feature=[f"{n}_rmse" for n in parts[1].feature.values])
    out = xr.concat(parts, dim="feature")
    if segments.rio.crs is not None:
        out = out.rio.write_crs(segments.rio.crs)
    return out


def train_classifier(data: Any, samples: Any, *, label: str = "class", model: Any = None, date: Any = None) -> Any:
    """Train a classifier on the features of ``data`` at sample points.

    Parameters
    ----------
    data
        What to classify (see ``zeit.classify``): a cube, a map or stack, a Dataset of
        maps (e.g. phenology or LandTrendr metrics), a ``zeit.ccdc`` result with ``date``,
        or a raster path.
    samples
        Points with their class: a ``GeoDataFrame`` or a vector file. Points in another
        CRS are reprojected; points outside the data or on a pixel with a missing feature
        are left out.
    label
        The column of ``samples`` holding the class (names or numbers).
    model
        A scikit-learn classifier (anything with ``fit``/``predict``); default a random
        forest of 100 trees (``random_state=42``).
    date
        CCDC segments: the date whose models are the features.

    Returns
    -------
    The fitted model, with the feature names in ``zeit_features_`` (``zeit.classify``
    checks them).

    Examples
    --------
    >>> stack = zeit.load_raster("s2_2022.tif")                        # (time, band, y, x)
    >>> rf = zeit.train_classifier(stack, "samples.gpkg", label="class")
    >>> classes = zeit.classify(stack, rf)
    """
    feats = features(data, date=date)
    points = _points(samples, feats)
    if label not in points.columns:
        raise ValueError(f"samples have no column {label!r}; they have {list(points.columns)}")
    x, kept = _values_at(feats, points)
    y = points[label].to_numpy()[kept]
    if len(y) == 0:
        raise ValueError("no sample falls on a pixel with every feature")
    if model is None:
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x, y)
    model.zeit_features_ = [str(f) for f in feats.feature.values]
    return model


def classify(data: Any, model: Any, *, date: Any = None, probability: bool = False) -> xr.Dataset:
    """Classify every pixel with a trained model.

    Parameters
    ----------
    data
        The features: a cube (each date and band a feature), a map or stack, a Dataset of
        maps, a ``zeit.ccdc`` result with ``date``, or a raster path. A lazy cube stays
        lazy.
    model
        A fitted classifier, e.g. from ``zeit.train_classifier``; trained elsewhere, its
        inputs must be the features in their order (``zeit.classify`` reorders them by name
        when the model has ``zeit_features_``). A deep learning model trained by
        ``zeit.ai.train`` goes to ``zeit.ai.predict``.
    date
        CCDC segments: the date whose models are the features.
    probability
        Also return each class's probability (``predict_proba``).

    Returns
    -------
    xarray.Dataset
        - ``label (y, x)``: ``1`` for the first class of ``model.classes_``, ``2`` for the
          second... (``0``: a missing feature), the names in its ``flag_meanings``;
        - ``probability (class, y, x)`` with ``probability=True``.

        Georeferenced as the input; the ``class`` coordinate holds the classes.
    """
    if hasattr(model, "zeit_meta_") and hasattr(model, "state_dict"):   # a zeit.ai model
        if date is not None:
            raise ValueError("date= is for CCDC segments; a zeit.ai model classifies the cube")
        from .ai.pipeline import predict

        return predict(model, data, probability=probability)
    feats = features(data, date=date)
    names = [str(f) for f in feats.feature.values]
    expected = getattr(model, "zeit_features_", None)
    if expected is not None and list(expected) != names:
        missing = [f for f in expected if f not in names]
        if missing:
            raise ValueError(f"the data lacks features the model was trained on: {missing[:5]}")
        feats = feats.sel(feature=list(expected))
    classes = [str(c) for c in getattr(model, "classes_", [])]
    k = len(classes)
    with_p = bool(probability)
    if with_p and not hasattr(model, "predict_proba"):
        raise ValueError("this model has no predict_proba")
    lookup = {c: i + 1 for i, c in enumerate(getattr(model, "classes_", []))}

    def _block(block: np.ndarray) -> np.ndarray:
        n_feat, rows, cols = block.shape
        out = np.zeros((1 + (k if with_p else 0), rows, cols), dtype=np.float32)
        if with_p:
            out[1:] = np.nan
        flat = block.reshape(n_feat, -1).T.astype(np.float64)
        ok = np.isfinite(flat).all(axis=1)
        if ok.any():
            pred = model.predict(flat[ok])
            labels = np.zeros(flat.shape[0], dtype=np.float32)
            labels[ok] = [lookup.get(v, 0) for v in pred]
            out[0] = labels.reshape(rows, cols)
            if with_p:
                proba = np.full((flat.shape[0], k), np.nan, dtype=np.float32)
                proba[ok] = model.predict_proba(flat[ok])
                out[1:] = proba.T.reshape(k, rows, cols)
        return out

    if feats.chunks is not None:
        import dask.array as da

        arr = feats.data.rechunk({0: -1})
        out = da.map_blocks(_block, arr, dtype=np.float32, chunks=((1 + (k if with_p else 0),),) + arr.chunks[1:])
    else:
        out = _block(np.asarray(feats.values))
    label_dtype = np.uint8 if k < 255 else np.uint16
    coords = {c: feats.coords[c] for c in ("y", "x", "spatial_ref") if c in feats.coords}
    coords["class"] = classes
    label = xr.DataArray(out[0].astype(label_dtype), dims=("y", "x"),
                         attrs={"long_name": "class", "flag_values": list(range(1, k + 1)),
                                "flag_meanings": " ".join(c.replace(" ", "_") for c in classes)})
    variables = {"label": label}
    if with_p:
        variables["probability"] = (("class", "y", "x"), out[1:])
    ds = xr.Dataset(variables, coords=coords, attrs={"algorithm": type(model).__name__})
    if feats.rio.crs is not None:
        ds = ds.rio.write_crs(feats.rio.crs)
        ds["label"] = ds["label"].rio.write_nodata(0, encoded=False)
    return ds


def _points(samples: Any, feats: xr.DataArray):
    import geopandas as gpd

    if not isinstance(samples, gpd.GeoDataFrame):
        samples = gpd.read_file(samples)
    if not (samples.geometry.geom_type == "Point").all():
        raise ValueError("samples must be points (take polygons' representative_point() first)")
    crs = feats.rio.crs
    if crs is not None and samples.crs is not None and samples.crs != crs:
        samples = samples.to_crs(crs)
    return samples


def _values_at(feats: xr.DataArray, points) -> tuple:
    """(features at the points (n, feature), which points were kept)."""
    from ._warp import transform_of

    t = transform_of(feats)
    cols, rows = ~t * (points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    inside = (rows >= 0) & (rows < feats.sizes["y"]) & (cols >= 0) & (cols < feats.sizes["x"])
    idx = np.flatnonzero(inside)
    values = feats.isel(y=xr.DataArray(rows[idx], dims="sample"), x=xr.DataArray(cols[idx], dims="sample"))
    values = np.asarray(values.transpose("sample", "feature").values, dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    kept = np.zeros(len(points), dtype=bool)
    kept[idx[finite]] = True
    return values[finite], kept
