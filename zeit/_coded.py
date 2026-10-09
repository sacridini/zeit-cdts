"""``zeit.coded``: Continuous Degradation Detection (CODED; Bullock, Woodcock & Olofsson 2020).

CODED monitors forests for degradation (a disturbance after which the land is still forest:
selective logging, understory fire) and deforestation (after which it is not) with the NDFI of
Souza et al. (2005). Per pixel (the engine is ``_core.coded``, in C++):

1. the fractions of green vegetation, non-photosynthetic vegetation, soil, shade and cloud of
   every observation (``zeit.unmix``), cloudy ones masked, and their NDFI;
2. a model of the NDFI (constant and an annual harmonic) over a training period, with its RMSE;
3. monitoring after it: ``consec`` observations in a row whose residual is beyond ``thresh``
   RMSEs below the model are a change; after each one, a new model of the next ``min_years``;
4. the land cover before and after each change, from the models: with training points, a
   random forest on the models' coefficients (the NDFI's and the fractions'), as CODED does;
   without, forest where the model's mean NDFI is at least ``forest_ndfi`` (0.5). A change that
   leaves forest is degradation, one that does not is deforestation.

The result's ``strata`` map (forest, non-forest, degradation, deforestation) is what CODED
recommends using it for: the strata of a sample (``zeit.sampling_design``) from which
``zeit.accuracy`` estimates the area of degradation.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._time import fractional_years, to_datetime_index

FEATURES = ["gv", "npv", "soil", "shade"]
STRATA = ["forest", "non_forest", "degradation", "deforestation", "disturbance"]
TYPES = {1: "degradation", 2: "deforestation", 3: "disturbance"}


def _layout(n_features: int, max_events: int):
    header = 3 + 1 + 1 + 3 * n_features + 1
    per_event = 3 + 3 + 1 + 3 * n_features
    return header, per_event


def coded(
    data: Any,
    *,
    start: Any = None,
    train_years: float = 3.0,
    consec: int = 3,
    thresh: float = 3.0,
    min_years: float = 3.0,
    min_obs: int = 6,
    direction: str = "loss",
    max_events: int = 3,
    training: Any = None,
    label: str = "label",
    forest_label: Any = None,
    forest_ndfi: float = 0.5,
    endmembers: Any = "souza2005",
    bands: Optional[Sequence[str]] = None,
    scale: Union[str, float] = "auto",
    cloud_threshold: Optional[float] = 0.05,
    seed: int = 42,
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """Continuous Degradation Detection (CODED, Bullock et al. 2020): degradation and
    deforestation from the NDFI time series.

    Parameters
    ----------
    data
        A Landsat-like reflectance cube ``(time, band, y, x)`` with blue, green, red, NIR,
        SWIR1 and SWIR2 (found by name, or ``bands=``), or the result of ``zeit.unmix`` (with
        ``ndfi``, ``gv``, ``npv``, ``soil``, ``shade``), or anything ``zeit.load_raster``
        reads. Every observation counts: no compositing.
    start
        Start of the monitoring (a date or year); the training period is the ``train_years``
        before it. Default: the training period starts with the series.
    train_years, consec, thresh, min_years, min_obs
        CODED's parameters (defaults: its documented example): years of the training period;
        consecutive observations beyond the threshold that make a change; the threshold, in
        RMSEs of the model; years after a change modelled as its new land cover, and the
        least time between changes; least observations to fit a model.
    direction
        ``"loss"`` (NDFI drops, CODED's disturbances) or ``"both"``.
    max_events
        Changes kept per pixel.
    training
        Points with the land cover in ``label`` (``GeoDataFrame`` or vector file), for the
        random forest that tells forest from the rest, from the models of the training
        period at the points. They should describe the land cover of that period.
    label, forest_label
        Column of ``training`` with the class, and the class that is forest (default:
        ``"forest"`` or ``1``, whichever the labels have).
    forest_ndfi
        Without ``training``: forest where the model's mean NDFI is at least this. The NDFI
        is near 1 in closed forest and below 0 on bare soil and pasture (Souza et al. 2005);
        a forest degraded a few years ago sits in between, hence the 0.5. Training points
        (a random forest, as in CODED) adapt to the forests at hand, such as open woodlands
        whose NDFI is naturally lower.
    endmembers, bands, scale, cloud_threshold
        The unmixing (see ``zeit.unmix``); observations whose cloud fraction is above
        ``cloud_threshold`` are left out (CODED: 0.05).
    seed
        Seed of the random forest.
    nodata, chunks, n_jobs
        As elsewhere in zeit.

    Returns
    -------
    xarray.Dataset
        - ``t_change (event, y, x)``: date of each change (its first observation beyond the
          threshold); ``t_before``: the observation before it; ``ndfi_change``: the mean NDFI
          residual of the change's observations (negative: the NDFI fell);
        - ``type (event, y, x)``: 1 degradation (forest after the change), 2 deforestation, 3
          disturbance (too few observations after it to tell), 0 none; ``n_events``;
        - ``forest (y, x)``: forest in the training period; ``ndfi_mean``, ``rmse``: the
          training model's mean NDFI and RMSE; ``post_ndfi (event, y, x)``: the mean NDFI of
          the model after each change;
        - ``strata (y, x)``: 1 forest, 2 non-forest (no change), 3 degradation, 4
          deforestation, 5 disturbance (by the first change), with its ``flag_meanings``:
          ready for ``zeit.sampling_design`` and ``zeit.accuracy``.

        ``zeit.extract_events`` takes it (one change per pixel, the same maps as for the
        other algorithms).

    Examples
    --------
    >>> result = zeit.coded(landsat, start=2000, training="land_cover_2000.gpkg")
    >>> result.strata.zeit.plot()
    >>> design = zeit.sampling_design(result.strata, expected_ua={"degradation": 0.6})
    """
    from ._embeddings import refuse

    refuse(data, "zeit.coded")
    from ._core.coded import Params, coded_batch, coded_size
    from ._sma import unmix

    if direction not in ("loss", "both"):
        raise ValueError(f"direction must be 'loss' or 'both', got {direction!r}")
    if isinstance(data, xr.Dataset) and "ndfi" in data:
        fractions = data
    else:
        fractions = unmix(data, endmembers, bands=bands, scale=scale, cloud_threshold=cloud_threshold,
                          nodata=nodata, chunks=chunks, n_jobs=n_jobs)
    missing = [v for v in ["ndfi"] + FEATURES if v not in fractions]
    if missing:
        raise ValueError(f"CODED needs the NDFI and the fractions {FEATURES}; missing: {missing}")
    if "time" not in fractions.dims:
        raise ValueError("CODED needs a time series: the cube has no time dimension")
    fractions = fractions.sortby("time")
    times = pd.DatetimeIndex(fractions.time.values)
    t = np.asarray(fractional_years(times), dtype=np.float64)
    if start is None:
        train_start = float(t[0])
    else:
        first = float(start) if isinstance(start, (int, float, np.integer, np.floating)) and not isinstance(
            start, bool) else float(fractional_years(to_datetime_index([start]))[0])
        train_start = first - float(train_years)
    params = Params()
    params.train_start, params.train_years = train_start, float(train_years)
    params.consec, params.thresh, params.min_years = int(consec), float(thresh), float(min_years)
    params.min_obs, params.loss_only, params.max_events = int(min_obs), direction == "loss", int(max_events)
    nf = len(FEATURES)
    size = coded_size(nf, int(max_events))

    ndfi = fractions["ndfi"].transpose("time", "y", "x")
    feats = xr.concat([fractions[v] for v in FEATURES], dim="feature").transpose("feature", "time", "y", "x")

    def _block(nd, fe):   # (..., time) and (..., feature, time) -> (..., row)
        shape = nd.shape[:-1]
        y = np.ascontiguousarray(nd.reshape(-1, nd.shape[-1]), dtype=np.float64)
        f = np.ascontiguousarray(fe.reshape((-1,) + fe.shape[-2:]), dtype=np.float64)
        return coded_batch(t, y, f, params, n_jobs=n_jobs).reshape(shape + (size,))

    if ndfi.chunks is not None:
        ndfi = ndfi.chunk({"time": -1})
        feats = feats.chunk({"feature": -1, "time": -1})
    out = xr.apply_ufunc(_block, ndfi, feats, input_core_dims=[["time"], ["feature", "time"]],
                         output_core_dims=[["row"]], dask="parallelized", output_dtypes=[np.float64],
                         dask_gufunc_kwargs={"output_sizes": {"row": size}}).transpose("row", "y", "x")

    ds = _to_dataset(out, fractions, nf, int(max_events))
    ds = _classify(ds, training, label, forest_label, forest_ndfi, seed)
    ds.attrs.update(algorithm="CODED", train_start=train_start, train_years=float(train_years), consec=int(consec),
                    thresh=float(thresh), min_years=float(min_years), min_obs=int(min_obs), direction=direction,
                    forest="random forest on training points" if training is not None
                    else f"model mean NDFI >= {forest_ndfi}")
    if fractions.rio.crs is not None:
        ds = ds.rio.write_crs(fractions.rio.crs)
    return ds


def _decimal_to_dates(values):
    from .metrics import _days_to_dates, _decimal_year_days
    return _days_to_dates(_decimal_year_days(values))


def _to_dataset(out, fractions: xr.Dataset, nf: int, max_events: int) -> xr.Dataset:
    header, per_event = _layout(nf, max_events)
    coords = {k: fractions.coords[k] for k in ("y", "x", "spatial_ref") if k in fractions.coords}
    da = out.drop_vars([c for c in out.coords if c not in coords]).assign_coords(coords)
    da = da.drop_vars("row", errors="ignore")
    events = xr.concat([da.isel(row=slice(header + e * per_event, header + (e + 1) * per_event))
                        for e in range(max_events)], dim="event")
    events = events.assign_coords(event=np.arange(1, max_events + 1))
    feature_names = [f"{v}_{c}" for v in FEATURES for c in ("c", "a", "b")]
    train = {"ndfi_c": da.isel(row=0), "ndfi_a": da.isel(row=1), "ndfi_b": da.isel(row=2), "rmse": da.isel(row=3)}
    for k, name in enumerate(feature_names):
        train[name] = da.isel(row=5 + k)
    post = {"ndfi_c": events.isel(row=3), "ndfi_a": events.isel(row=4), "ndfi_b": events.isel(row=5),
            "rmse": events.isel(row=6)}
    for k, name in enumerate(feature_names):
        post[name] = events.isel(row=7 + k)
    ds = xr.Dataset({
        "t_change": _decimal_to_dates(events.isel(row=0)),
        "t_before": _decimal_to_dates(events.isel(row=1)),
        "ndfi_change": events.isel(row=2).astype(np.float32),
        "n_events": da.isel(row=header - 1).fillna(0).astype(np.uint8),
        "n_train": da.isel(row=4).fillna(0).astype(np.uint16),
        "ndfi_mean": train["ndfi_c"].astype(np.float32),
        "rmse": train["rmse"].astype(np.float32),
        "post_ndfi": post["ndfi_c"].astype(np.float32),
    })
    ds = ds.drop_vars("row", errors="ignore")
    # The models, for the classification (dropped afterwards).
    ds = ds.assign({f"_train_{k}": v.drop_vars("row", errors="ignore") for k, v in train.items()})
    ds = ds.assign({f"_post_{k}": v.drop_vars("row", errors="ignore") for k, v in post.items()})
    return ds


def _model_features(ds: xr.Dataset, prefix: str) -> list:
    names = ["ndfi_c", "ndfi_a", "ndfi_b", "rmse"] + [f"{v}_c" for v in FEATURES]
    return [ds[f"{prefix}{n}"] for n in names]


def _classify(ds: xr.Dataset, training: Any, label: str, forest_label: Any, forest_ndfi: float,
              seed: int) -> xr.Dataset:
    has_post = ds["_post_ndfi_c"].notnull()
    happened = ds["t_change"].notnull()
    if training is None:
        forest = ds["_train_ndfi_c"] >= forest_ndfi
        post_forest = ds["_post_ndfi_c"] >= forest_ndfi
    else:
        model, forest_code = _train_forest(ds, training, label, forest_label, seed)

        def _predict(*columns):
            x = np.stack([np.asarray(c, dtype=np.float64) for c in columns], axis=-1)
            shape = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            ok = np.isfinite(x).all(axis=1)
            out = np.zeros(len(x), dtype=np.int16)
            if ok.any():
                out[ok] = model.predict(x[ok]).astype(np.int16)
            return out.reshape(shape)

        def predict(prefix):
            return xr.apply_ufunc(_predict, *_model_features(ds, prefix), dask="parallelized",
                                  output_dtypes=[np.int16])

        forest = predict("_train_") == forest_code
        post_forest = predict("_post_") == forest_code
    kind = xr.where(happened & has_post & post_forest, 1, xr.where(happened & has_post, 2, xr.where(happened, 3, 0)))
    ds["type"] = kind.astype(np.uint8)
    ds["forest"] = (forest & ds["_train_ndfi_c"].notnull()).astype(np.uint8)
    first = ds["type"].isel(event=0, drop=True)
    strata = xr.where(first > 0, first + 2, xr.where(ds["forest"] == 1, 1, 2))
    strata = xr.where(ds["_train_ndfi_c"].notnull() | (first > 0), strata, 0).astype(np.uint8)
    ds["strata"] = strata.assign_attrs(flag_values=[1, 2, 3, 4, 5], flag_meanings=" ".join(STRATA),
                                       long_name="CODED strata")
    ds["type"].attrs.update(flag_values=[1, 2, 3], flag_meanings="degradation deforestation disturbance")
    ds = ds.drop_vars([v for v in ds.data_vars if str(v).startswith("_")])
    ds["strata"] = ds["strata"].rio.write_nodata(0) if "x" in ds.dims else ds["strata"]
    return ds


def _train_forest(ds: xr.Dataset, training: Any, label: str, forest_label: Any, seed: int):
    """A random forest of land cover on the training-period models at the training points."""
    import geopandas as gpd
    from sklearn.ensemble import RandomForestClassifier

    from ._accuracy import _to_crs, _values_at

    points = training if isinstance(training, gpd.GeoDataFrame) else gpd.read_file(training)
    if label not in points.columns:
        raise ValueError(f"the training points have no {label!r} column")
    columns = _model_features(ds, "_train_")
    reference = columns[0]
    if reference.rio.crs is None and ds.rio.crs is not None:
        reference = reference.rio.write_crs(ds.rio.crs)
    pts = _to_crs(points, reference)
    from ._warp import transform_of
    t = transform_of(reference)
    cols, rows = ~t * (pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy())
    rows, cols = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    inside = (rows >= 0) & (rows < reference.sizes["y"]) & (cols >= 0) & (cols < reference.sizes["x"])
    if not inside.any():
        raise ValueError("no training point falls on the data")
    yi, xi = xr.DataArray(rows[inside], dims="p"), xr.DataArray(cols[inside], dims="p")
    x = np.stack([np.asarray(c.isel(y=yi, x=xi).values, dtype=np.float64) for c in columns], axis=1)
    labels = points.loc[inside, label].to_numpy()
    ok = np.isfinite(x).all(axis=1)
    if ok.sum() < 2:
        raise ValueError("fewer than two training points have a model of the training period")
    classes = list(pd.unique(labels[ok]))
    if forest_label is None:
        for candidate in ("forest", "Forest", 1, "1"):
            if candidate in classes:
                forest_label = candidate
                break
        else:
            raise ValueError(f"which class is forest? pass forest_label= (classes: {classes})")
    if forest_label not in classes:
        raise ValueError(f"forest_label {forest_label!r} is not among the training classes {classes}")
    codes = {c: i + 1 for i, c in enumerate(classes)}
    y = np.array([codes[c] for c in labels[ok]])
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(x[ok], y)
    return model, codes[forest_label]
