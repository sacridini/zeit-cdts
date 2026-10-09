"""``zeit.som`` and ``zeit.clean_samples``: Self-Organizing Maps of any cube, one call.

The features of a pixel are what it holds off ``y``/``x``, as in ``zeit.train_classifier``
(the dates of an index, the dates and bands of a cube, the maps of a Dataset). A SOM is
trained on a sample of the pixels by the engine (``zeit._som.SOM``, a bit-for-bit port of
MiniSom) and every pixel goes to its best-matching neuron; the result is a georeferenced
cluster map with the neurons' prototypes, which read as typical trajectories.
"""

from typing import Any, Optional

import numpy as np
import xarray as xr

_NEIGHBORHOODS = ("gaussian", "mexican_hat", "bubble", "triangle")
_DECAYS = ("linear_decay_to_zero", "asymptotic_decay", "inverse_decay_to_zero")


def _feature_cube(data: Any, nodata: Any, chunks: Any):
    """The input as a cube (DataArray or Dataset) with its missing values as NaN."""
    from ._load import load_raster
    from ._lt import _missing_values

    if isinstance(data, np.ndarray):
        dims = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}.get(data.ndim)
        if dims is None:
            raise ValueError(f"som takes 2-D to 4-D arrays, got shape {data.shape}")
        data = xr.DataArray(data, dims=dims)
    elif not isinstance(data, (xr.DataArray, xr.Dataset)):
        data = load_raster(data, chunks=chunks)
    if isinstance(data, xr.Dataset):
        out = {}
        for name, var in data.data_vars.items():
            if not {"y", "x"} <= set(var.dims):
                continue
            # a Dataset's maps: "auto" takes only the NoData stored with them (0 is a value
            # of e.g. n_breaks), never the 0 of an integer cube
            if nodata is None:
                missing = []
            elif isinstance(nodata, str):
                stored = var.rio.nodata
                missing = [stored] if stored is not None and not np.isnan(stored) else []
            else:
                missing = [nodata]
            out[name] = _as_nan(var, missing)
        ds = xr.Dataset(out, attrs=data.attrs)
        return ds.rio.write_crs(data.rio.crs) if data.rio.crs is not None else ds
    return _as_nan(data, _missing_values(data, nodata))


def _as_nan(da: xr.DataArray, missing: list) -> xr.DataArray:
    if not missing and np.issubdtype(da.dtype, np.floating):
        return da
    crs = da.rio.crs if "y" in da.dims and "x" in da.dims else None
    out = da.astype(np.float32 if da.dtype.itemsize <= 2 else np.float64)
    if missing:
        out = out.where(~da.isin(missing))
    return out.rio.write_crs(crs) if crs is not None and out.rio.crs is None else out


def _engine(x: int, y: int, n_features: int, *, sigma, learning_rate, decay, neighborhood, topology, seed):
    from ._som import SOM

    if decay not in _DECAYS:
        raise ValueError(f"decay must be one of {', '.join(_DECAYS)}, got {decay!r}")
    if neighborhood not in _NEIGHBORHOODS:
        raise ValueError(f"neighborhood must be one of {', '.join(_NEIGHBORHOODS)}, got {neighborhood!r}")
    if int(x) < 1 or int(y) < 1:
        raise ValueError(f"the grid needs at least one neuron a side, got x={x}, y={y}")
    return SOM(int(x), int(y), n_features, sigma=sigma, learning_rate=learning_rate, decay_function=decay,
               neighborhood_function=neighborhood, topology=topology, random_seed=seed)


def _train(engine, samples: np.ndarray, *, init: Optional[str], num_iters: Optional[int], algorithm: str,
           n_jobs: int) -> None:
    if init == "pca" and samples.shape[1] >= 2 and len(samples) >= 2:
        engine.pca_weights_init(samples)
    elif init in ("pca", "random"):
        engine.random_weights_init(samples)
    elif init is not None:
        raise ValueError(f"init must be 'pca', 'random' or None, got {init!r}")
    if algorithm == "batch":
        engine.train(samples, 20 if num_iters is None else num_iters, n_jobs=n_jobs, algorithm="batch")
    elif algorithm == "online":
        engine.train(samples, 20 * len(samples) if num_iters is None else num_iters, algorithm="online")
    else:
        raise ValueError(f"algorithm must be 'batch' or 'online', got {algorithm!r}")


def som(
    data: Any,
    *,
    x: int = 3,
    y: int = 3,
    sample: Optional[int] = 50_000,
    num_iters: Optional[int] = None,
    algorithm: str = "online",
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    decay: str = "linear_decay_to_zero",
    neighborhood: str = "gaussian",
    topology: str = "rectangular",
    init: Optional[str] = "random",
    seed: int = 42,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """Cluster the pixels of a map or a cube with a Self-Organizing Map (Kohonen).

    Every value a pixel holds off its ``y``/``x`` is a feature (each date of an index,
    each band of each date, each map of a Dataset), so pixels share a neuron when their
    whole trajectories are alike. The SOM is trained on a random sample of the pixels,
    then every pixel goes to its best-matching neuron. The engine is a bit-for-bit port of
    MiniSom (online and Batch SOM), in C++ with OpenMP.

    Parameters
    ----------
    data
        A map ``(y, x)``, a stack ``(band, y, x)``, a cube ``(time, y, x)`` or
        ``(time, band, y, x)`` (in memory or dask), a Dataset of maps (e.g. phenology
        metrics), or anything ``load_raster`` reads. The features should share a scale
        (one index, or reflectances): the distance between pixels adds them up as they are.
    x, y
        The neurons of the grid: ``x * y`` clusters. Small grids (2 x 2, 3 x 3) cluster;
        larger ones (10 x 10 and up) explore the data.
    sample
        Pixels to train on, drawn at random among those with every feature (reproducible
        by ``seed``); ``None`` trains on all of them.
    num_iters
        ``algorithm="online"``: single-sample updates (default 20 passes over the sample).
        ``"batch"``: passes over the sample (default 20).
    algorithm
        ``"online"`` (default; the classic sample-by-sample update) or ``"batch"`` (Batch
        SOM, parallel). On small grids the batch update can leave neurons empty (a
        neuron that wins no pixel is pulled onto its neighbour's mean, and stays there);
        the online one does not, and is about as fast on a sample of 50,000.
    sigma, learning_rate, neighborhood, topology
        The SOM's: initial neighbourhood radius in grid units, initial learning rate,
        ``"gaussian"``, ``"mexican_hat"``, ``"bubble"`` or ``"triangle"``, and
        ``"rectangular"`` or ``"hexagonal"``.
    decay
        Online: how the learning rate falls during training. ``"linear_decay_to_zero"``
        (default) ends at 0, so the prototypes settle; MiniSom's default,
        ``"asymptotic_decay"``, ends at a third of it, and the last samples seen still
        pull the prototypes (on 30,000 Rondonia trajectories: a mean distance 10%
        larger, prototypes up to twice as far from their pixels' mean).
        ``"inverse_decay_to_zero"`` is the third choice.
    init
        Initial prototypes: ``"random"`` (default; pixels of the sample), ``"pca"``
        (spanning the first two principal components of the sample; poor on a grid one
        neuron wide, and random with a single feature) or ``None`` (MiniSom's random unit
        vectors).
    seed
        Seed of the sample, the initial weights and the sample order.
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``: ``"auto"``, a
        number or ``None``. Pixels with a missing feature are left out of the training
        and get no neuron.
    chunks
        A raster path: read lazily with these chunks (see ``load_raster``).
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        - ``label (y, x)``: the neuron of every pixel, ``1`` to ``x * y`` (``0``: a missing
          feature), the grid positions ``i_j`` in its ``flag_meanings``;
        - ``distance (y, x)``: the distance from the pixel to its neuron's prototype
          (large where the map represents the pixel poorly);
        - ``prototypes (neuron, ...)``: each neuron's prototype, with the cube's dims and
          coordinates (``prototypes.sel(neuron=3)`` is a series over ``time``);
        - ``n_pixels (neuron)``: the pixels of each neuron.

        Georeferenced as the input, with the grid position of each neuron in the ``i``
        and ``j`` coordinates and the sample's mean distance in ``quantization_error``.
        A lazy cube stays lazy. ``zeit.plot(cube, fit=result)`` draws a pixel's
        prototype over its series.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")     # (time, y, x)
    >>> clusters = zeit.som(ndvi, x=2, y=2, sample=30_000)
    >>> clusters.prototypes.sel(neuron=1).plot()                  # a typical trajectory
    >>> zeit.plot(ndvi, fit=clusters)
    """
    from . import _core
    from ._classify_api import features

    cube = _feature_cube(data, nodata, chunks)
    feats = features(cube)
    n_features = feats.sizes["feature"]
    ny, nx = feats.sizes["y"], feats.sizes["x"]

    valid = np.asarray(np.isfinite(feats).all("feature").values).ravel()
    idx = np.flatnonzero(valid)
    if not len(idx):
        raise ValueError("no pixel has every feature")
    if sample is not None:
        if int(sample) < 1:
            raise ValueError(f"sample must be at least 1 (or None), got {sample!r}")
        if len(idx) > int(sample):
            idx = np.sort(np.random.default_rng(seed).choice(idx, int(sample), replace=False))
    rows, cols = np.divmod(idx, nx)
    picked = feats.isel(y=xr.DataArray(rows, dims="sample"), x=xr.DataArray(cols, dims="sample"))
    samples = np.ascontiguousarray(picked.transpose("sample", "feature").values, dtype=np.float64)

    engine = _engine(x, y, n_features, sigma=sigma, learning_rate=learning_rate, decay=decay,
                     neighborhood=neighborhood, topology=topology, seed=seed)
    _train(engine, samples, init=init, num_iters=num_iters, algorithm=algorithm, n_jobs=n_jobs)
    weights = np.ascontiguousarray(engine.weights, dtype=np.float64)
    codebook = weights.reshape(-1, n_features)
    n = codebook.shape[0]

    def _block(block: np.ndarray) -> np.ndarray:
        k, r, c = block.shape
        flat = block.reshape(k, -1).T.astype(np.float64)
        ok = np.isfinite(flat).all(axis=1)
        out = np.zeros((2, r * c), dtype=np.float32)
        out[1] = np.nan
        if ok.any():
            good = np.ascontiguousarray(flat[ok])
            bmu = np.asarray(_core.som.predict_bmus(good, weights, n_jobs))
            out[0, ok] = bmu + 1
            out[1, ok] = np.linalg.norm(good - codebook[bmu], axis=1)
        return out.reshape(2, r, c)

    if feats.chunks is not None:
        import dask.array as dsa

        arr = feats.data.rechunk({0: -1})
        out = dsa.map_blocks(_block, arr, dtype=np.float32, chunks=((2,),) + arr.chunks[1:])
        counts = dsa.bincount(out[0].ravel().astype(np.int64), minlength=n + 1)[1:]
    else:
        out = _block(np.asarray(feats.values))
        counts = np.bincount(out[0].ravel().astype(np.int64), minlength=n + 1)[1:]
    label_dtype = np.uint8 if n < 255 else (np.uint16 if n < 65535 else np.uint32)

    grid_i, grid_j = np.divmod(np.arange(n), int(y))   # MiniSom's flat index i * y + j
    names = [f"{i}_{j}" for i, j in zip(grid_i, grid_j)]
    coords = {c: feats.coords[c] for c in ("y", "x", "spatial_ref") if c in feats.coords}
    coords.update(neuron=np.arange(1, n + 1), i=("neuron", grid_i), j=("neuron", grid_j))
    if isinstance(cube, xr.DataArray):
        others = [d for d in cube.dims if d not in ("y", "x")]
        proto_dims = ("neuron", *others) if others else ("neuron", "feature")
        shape = [cube.sizes[d] for d in others] if others else [n_features]
        for d in others:
            if d in cube.coords and cube.coords[d].dims == (d,):
                coords[d] = cube.coords[d].values
        if not others:
            coords["feature"] = feats.feature.values
    else:
        proto_dims, shape = ("neuron", "feature"), [n_features]
        coords["feature"] = feats.feature.values
    label = xr.DataArray(out[0].astype(label_dtype), dims=("y", "x"),
                         attrs={"long_name": "SOM neuron", "flag_values": list(range(1, n + 1)),
                                "flag_meanings": " ".join(names)})
    ds = xr.Dataset(
        {
            "label": label,
            "distance": (("y", "x"), out[1]),
            "prototypes": (proto_dims, codebook.reshape(n, *shape)),
            "n_pixels": ("neuron", counts),
        },
        coords=coords,
        attrs={"algorithm": "SOM", "grid": f"{int(x)}x{int(y)}", "topology": topology,
               "training_algorithm": algorithm, "decay": decay, "n_samples": int(len(samples)),
               "quantization_error": float(engine.quantization_error(samples, n_jobs))},
    )
    if feats.rio.crs is not None:
        ds = ds.rio.write_crs(feats.rio.crs)
        ds["label"] = ds["label"].rio.write_nodata(0, encoded=False)
    return ds


def clean_samples(
    data: Any,
    samples: Any,
    *,
    label: str = "class",
    x: Optional[int] = None,
    y: Optional[int] = None,
    num_iters: Optional[int] = None,
    algorithm: str = "online",
    sigma: float = 1.0,
    learning_rate: float = 0.5,
    decay: str = "linear_decay_to_zero",
    neighborhood: str = "gaussian",
    topology: str = "rectangular",
    init: Optional[str] = "random",
    seed: int = 42,
    date: Any = None,
    nodata: Any = "auto",
    n_jobs: int = -1,
):
    """Check labelled samples with a SOM before training a classifier.

    A SOM is trained on the features of the samples (as ``zeit.train_classifier`` reads
    them); a sample whose class differs from the majority class of its neuron is
    suspicious, as in ``sits_som_clean_samples`` of the R ``sits`` package.

    Parameters
    ----------
    data
        What the classifier will be trained on (see ``zeit.train_classifier``).
    samples
        Points with their class: a ``GeoDataFrame`` or a vector file, reprojected when
        in another CRS.
    label
        The column of ``samples`` holding the class.
    x, y
        The neurons of the grid; by default a square grid of about ``5 * sqrt(n)``
        neurons for ``n`` samples (Vesanto's rule), so that each neuron gets a few.
    num_iters, algorithm, sigma, learning_rate, decay, neighborhood, topology, init, seed, nodata, n_jobs
        As in ``zeit.som``.
    date
        CCDC segments: the date whose models are the features.

    Returns
    -------
    geopandas.GeoDataFrame
        The samples as given, with:

        - ``neuron``: the sample's neuron (``0``: outside the data or a missing feature);
        - ``neuron_class``: the majority class of that neuron's samples;
        - ``purity``: the share of the neuron's samples in that class;
        - ``keep``: the sample's class is its neuron's (``False`` also without a neuron).

    Examples
    --------
    >>> checked = zeit.clean_samples(stack, "samples.gpkg", label="class")
    >>> rf = zeit.train_classifier(stack, checked[checked.keep], label="class")
    """
    import geopandas as gpd

    from ._classify_api import _points, _values_at, features

    cube = _feature_cube(data, nodata, None) if date is None else data
    feats = features(cube, date=date)
    original = samples if isinstance(samples, gpd.GeoDataFrame) else gpd.read_file(samples)
    points = _points(original, feats)
    if label not in points.columns:
        raise ValueError(f"samples have no column {label!r}; they have {list(points.columns)}")
    values, kept = _values_at(feats, points)
    if len(values) == 0:
        raise ValueError("no sample falls on a pixel with every feature")
    if x is None or y is None:
        side = max(2, int(round(np.sqrt(5 * np.sqrt(len(values))))))
        x = side if x is None else x
        y = side if y is None else y
    engine = _engine(x, y, values.shape[1], sigma=sigma, learning_rate=learning_rate, decay=decay,
                     neighborhood=neighborhood, topology=topology, seed=seed)
    values = np.ascontiguousarray(values, dtype=np.float64)
    _train(engine, values, init=init, num_iters=num_iters, algorithm=algorithm, n_jobs=n_jobs)
    winners = np.asarray(engine.predict(values, n_jobs=n_jobs))

    classes = points[label].to_numpy()[kept]
    names, ids = np.unique(classes, return_inverse=True)
    counts = np.zeros((int(x) * int(y), len(names)), dtype=np.int64)
    np.add.at(counts, (winners, ids), 1)
    majority = counts.argmax(axis=1)            # the first class on ties
    purity = counts.max(axis=1) / np.maximum(counts.sum(axis=1), 1)

    out = original.copy()
    neuron = np.zeros(len(out), dtype=np.int64)
    neuron[kept] = winners + 1
    neuron_class = np.full(len(out), None, dtype=object)
    neuron_class[kept] = names[majority[winners]]
    share = np.full(len(out), np.nan)
    share[kept] = purity[winners]
    keep = np.zeros(len(out), dtype=bool)
    keep[kept] = ids == majority[winners]
    out["neuron"] = neuron
    out["neuron_class"] = neuron_class
    out["purity"] = share
    out["keep"] = keep
    return out
