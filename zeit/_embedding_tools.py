"""What to do with a cube of embeddings besides classifying it: ``zeit.similarity`` (where
else does this look like these samples?), ``zeit.embedding_change`` (how far did each pixel
move from one year to the next?, which ``extract_events`` turns into events) and the principal
components ``zeit.plot`` shows an embedding cube through.

None of it depends on the product: a ``(time, band, y, x)`` cube of any embeddings will do.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr


_METRICS = ("cosine", "euclidean")


def _as_cube(emb: Any, name: str) -> xr.DataArray:
    if not isinstance(emb, xr.DataArray):
        from ._load import load_raster

        emb = load_raster(emb, chunks="auto")
    if "band" not in emb.dims or "y" not in emb.dims or "x" not in emb.dims:
        raise ValueError(f"{name} takes a cube of embeddings, (time, band, y, x) or (band, y, x) "
                         f"(e.g. from zeit.load_embeddings), got dims {emb.dims}")
    extra = [d for d in emb.dims if d not in ("time", "band", "y", "x")]
    if extra:
        raise ValueError(f"{name}: unexpected dims {extra}")
    return emb


def _keep(emb: xr.DataArray) -> Dict[str, Any]:
    return {k: v for k, v in emb.attrs.items() if str(k).startswith("embedding_")}


def _check_metric(metric: str) -> str:
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")
    return metric


def _distance(a: xr.DataArray, b: xr.DataArray, metric: str) -> xr.DataArray:
    """Per pixel distance between two embeddings along ``band``: ``1 - cos`` or Euclidean."""
    if metric == "euclidean":
        return np.sqrt(((a - b) ** 2).sum("band", skipna=False))
    dot = (a * b).sum("band", skipna=False)
    norm = np.sqrt((a * a).sum("band", skipna=False) * (b * b).sum("band", skipna=False))
    return 1 - dot / norm.where(norm > 0)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def similarity(emb: Any, ref: Any, *, by: Optional[str] = None, year: Any = None,
               metric: str = "cosine") -> xr.DataArray:
    """How much each pixel looks like a reference: "find more places like these".

    Parameters
    ----------
    emb
        Embeddings, ``(time, band, y, x)`` or ``(band, y, x)`` (``zeit.load_embeddings``).
    ref
        What to look for: points or polygons (a ``GeoDataFrame`` or a vector file, in any
        CRS), whose embeddings are averaged (every cell of a polygon); or one embedding
        (a sequence of as many values as ``band``).
    by
        A column of ``ref``: one reference (and one map) per value of it, e.g. the class of
        each sample.
    year
        The year the samples' embeddings are taken from (default: the last of the cube).
        Every year of the cube is then compared with it, which shows where it appears over
        time (new mines, new fields).
    metric
        ``"cosine"`` (default): the cosine similarity, 1 for the same direction; or
        ``"euclidean"``: the distance (0 for the same embedding).

    Returns
    -------
    xarray.DataArray
        ``(time, y, x)`` (or ``(y, x)``) float32, with a ``class`` dim first when ``by`` is
        given; lazy when the cube is.

    Examples
    --------
    >>> emb = zeit.load_embeddings("aoi.gpkg", source="alphaearth", years=range(2018, 2025))
    >>> mines = zeit.similarity(emb, "known_mines.gpkg", year=2024)
    >>> zeit.plot(mines)                                   # where it looks like a mine, every year
    """
    da = _as_cube(emb, "similarity")
    metric = _check_metric(metric)
    references = _references(da, ref, by, year)
    maps = []
    for label, vector in references.items():
        v = xr.DataArray(np.asarray(vector, dtype=np.float32), dims="band", coords={"band": da.band.values})
        if metric == "cosine":
            out = 1 - _distance(da, v, "cosine")
        else:
            out = _distance(da, v, "euclidean")
        maps.append(out.astype(np.float32))
    if by is not None:
        out = xr.concat(maps, dim=pd.Index([str(k) for k in references], name="class"))
    else:
        out = maps[0]
    out.name = "similarity"
    out.attrs = {"metric": metric, **_keep(da)}
    if da.rio.crs is not None:
        out = out.rio.write_crs(da.rio.crs)
    return out.rio.write_nodata(np.nan, encoded=False)


def _references(da: xr.DataArray, ref: Any, by: Optional[str], year: Any) -> Dict[Any, np.ndarray]:
    import geopandas as gpd

    nb = da.sizes["band"]
    if not isinstance(ref, (str, gpd.GeoDataFrame, gpd.GeoSeries)) and not hasattr(ref, "__fspath__"):
        vector = np.asarray(ref.values if isinstance(ref, xr.DataArray) else ref, dtype=np.float64).ravel()
        if vector.size != nb:
            raise ValueError(f"ref: a reference embedding has {nb} values (one per band), got {vector.size}")
        if by is not None:
            raise ValueError("by= is a column of reference samples")
        return {None: vector}
    samples = gpd.read_file(ref) if not isinstance(ref, (gpd.GeoDataFrame, gpd.GeoSeries)) else ref
    if isinstance(samples, gpd.GeoSeries):
        samples = gpd.GeoDataFrame(geometry=samples)
    if samples.empty:
        raise ValueError("ref has no samples")
    if da.rio.crs is not None and samples.crs is not None and samples.crs != da.rio.crs:
        samples = samples.to_crs(da.rio.crs)
    base = da
    if "time" in da.dims:
        if year is None:
            base = da.isel(time=-1)
        else:
            stamp = pd.Timestamp(int(year), 1, 1) if np.isscalar(year) and str(year).isdigit() else pd.Timestamp(year)
            match = np.flatnonzero(pd.DatetimeIndex(da.time.values).year == stamp.year)
            if not len(match):
                raise ValueError(f"year={year}: the cube has {sorted(set(pd.DatetimeIndex(da.time.values).year))}")
            base = da.isel(time=int(match[0]))
    elif year is not None:
        raise ValueError("year= picks a year of a (time, band, y, x) cube; this one has a single year")
    if by is not None and by not in samples.columns:
        raise ValueError(f"ref has no column {by!r}; it has {list(samples.columns)}")
    groups = samples.groupby(by, sort=True) if by is not None else [(None, samples)]
    out = {}
    for label, group in groups:
        values = _values_in(base, group)
        if not len(values):
            raise ValueError(f"no reference sample{'' if label is None else f' of {by}={label!r}'} falls on a "
                             "pixel with an embedding")
        out[label] = values.mean(axis=0)
    return out


def _values_in(base: xr.DataArray, samples) -> np.ndarray:
    """The embeddings (n, band) of the cells under points and inside polygons."""
    from ._warp import transform_of

    t = transform_of(base)
    points = samples[samples.geometry.geom_type == "Point"]
    rows: List[np.ndarray] = []
    if len(points):
        cols, rws = ~t * (points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
        rws, cols = np.floor(rws).astype(int), np.floor(cols).astype(int)
        inside = (rws >= 0) & (rws < base.sizes["y"]) & (cols >= 0) & (cols < base.sizes["x"])
        if inside.any():
            picked = base.isel(y=xr.DataArray(rws[inside], dims="sample"), x=xr.DataArray(cols[inside], dims="sample"))
            rows.append(np.asarray(picked.transpose("sample", "band").values, dtype=np.float64))
    polygons = samples[samples.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    for geom in polygons.geometry:
        try:
            part = base.rio.clip([geom], drop=True)
        except Exception:  # noqa: BLE001 - a polygon outside the cube (rioxarray's NoDataInBounds)
            continue
        values = np.asarray(part.transpose("y", "x", "band").values, dtype=np.float64).reshape(-1, base.sizes["band"])
        rows.append(values)
    if not rows:
        return np.empty((0, base.sizes["band"]))
    values = np.concatenate(rows)
    return values[np.isfinite(values).all(axis=1)]


# ---------------------------------------------------------------------------
# Change from year to year
# ---------------------------------------------------------------------------

def embedding_change(emb: Any, *, metric: str = "cosine", baseline: Any = "previous") -> xr.Dataset:
    """How far each pixel's embedding moved from one year to another.

    Parameters
    ----------
    emb
        Embeddings ``(time, band, y, x)`` with at least two years (``zeit.load_embeddings``).
    metric
        ``"cosine"`` (default): ``1 - cos`` of the angle between the two embeddings (0 for
        the same direction, 2 for opposite ones); or ``"euclidean"``.
    baseline
        ``"previous"`` (default): each year against the year before it; ``"first"``: each
        year against the first; or a year to compare every other year with.

    Returns
    -------
    xarray.Dataset
        - ``distance (time, y, x)``: the distance of each year (from the second on, or every
          year but the baseline's) to its baseline;
        - ``noise (y, x)``: the median of a pixel's distances, its usual movement from year
          to year (one change does not move a median).

        ``extract_events`` turns it (with ``baseline="previous"``) into one event per pixel,
        in the schema of the other change algorithms: ``magnitude`` the distance, ``dsnr``
        the distance over the noise, ``yod`` the year before the first embedding that shows
        the change and ``date`` January 1 of that one. An embedding summarises a year, so a
        change in the middle of a year shows partly in it and partly in the next: compare
        with other algorithms with ``zeit.agreement(..., tolerance=1)``.

    Examples
    --------
    >>> emb = zeit.load_embeddings("aoi.gpkg", source="tessera", years=range(2017, 2026))
    >>> change = zeit.embedding_change(emb)
    >>> events = zeit.extract_events(change, min_magnitude=0.2)   # or sort_by="dsnr"
    """
    da = _as_cube(emb, "embedding_change")
    metric = _check_metric(metric)
    if "time" not in da.dims or da.sizes["time"] < 2:
        raise ValueError("embedding_change needs a (time, band, y, x) cube with at least two years")
    times = pd.DatetimeIndex(da.time.values)
    if isinstance(baseline, str) and baseline == "previous":
        after = da.isel(time=slice(1, None))
        before = da.isel(time=slice(None, -1)).assign_coords(time=after.time.values)
    elif isinstance(baseline, str) and baseline == "first":
        after = da.isel(time=slice(1, None))
        before = da.isel(time=0, drop=True)
    else:
        try:
            year = int(baseline)
        except (TypeError, ValueError):
            raise ValueError(f"baseline must be 'previous', 'first' or a year, got {baseline!r}") from None
        match = np.flatnonzero(times.year == year)
        if not len(match):
            raise ValueError(f"baseline={year}: the cube has the years {list(times.year)}")
        before = da.isel(time=int(match[0]), drop=True)
        after = da.isel(time=[i for i in range(len(times)) if i != match[0]])
    distance = _distance(after, before, metric).astype(np.float32)
    data = distance.data
    if hasattr(data, "dask"):
        distance = distance.chunk({"time": -1})
    noise = distance.median("time").astype(np.float32)
    coords = {k: da.coords[k] for k in ("y", "x", "spatial_ref") if k in da.coords}
    ds = xr.Dataset({"distance": distance.drop_vars([c for c in distance.coords if c not in ("time", "y", "x")]),
                     "noise": noise.drop_vars([c for c in noise.coords if c not in ("y", "x")])},
                    attrs={"algorithm": "embedding_change", "metric": metric,
                           "baseline": baseline if isinstance(baseline, str) else int(baseline), **_keep(da)})
    ds = ds.assign_coords(coords)
    if da.rio.crs is not None:
        ds = ds.rio.write_crs(da.rio.crs)
    return ds


# ---------------------------------------------------------------------------
# Principal components, for zeit.plot
# ---------------------------------------------------------------------------

def pca_fit(x: np.ndarray, *, k: int = 6, sample: int = 40_000, seed: int = 0) -> Dict[str, np.ndarray]:
    """Principal components of embeddings ``x`` (n, band): the first ``k`` ``weights``
    (k, band) with a fixed sign, the ``mean``, the 2 and 98 % percentiles of each component
    (``lo``, ``hi``) and the fraction of the variance each one explains (``explained``)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3:
        raise ValueError("the cube has too few pixels with an embedding to find its principal components")
    if len(x) > sample:
        x = x[np.random.default_rng(seed).choice(len(x), sample, replace=False)]
    mean = x.mean(axis=0)
    _, sv, vt = np.linalg.svd(x - mean, full_matrices=False)
    k = min(k, x.shape[1], vt.shape[0])
    comps = vt[:k]
    comps *= np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])[:, None]   # a fixed sign
    projected = (x - mean) @ comps.T
    variance = sv ** 2
    return {"weights": comps, "mean": mean, "lo": np.percentile(projected, 2, axis=0),
            "hi": np.percentile(projected, 98, axis=0),
            "explained": variance[:k] / variance.sum() if variance.sum() > 0 else np.zeros(k)}


def pca_rgb(emb: xr.DataArray, *, sample: int = 40_000, seed: int = 0, fit: Optional[Dict] = None) -> xr.DataArray:
    """The first three principal components of the embeddings, stretched to 0..1, as a
    ``red``/``green``/``blue`` band axis. The components and their stretch are fitted once,
    on pixels of every year, and applied to all of them: the same colour is the same
    embedding in every year, so paging through the years shows change. ``fit`` (from
    ``pca_fit``) skips the fitting."""
    da = _as_cube(emb, "pca_rgb")
    if fit is None:
        fit = pca_fit(_sample(da, sample), sample=sample, seed=seed)
    k = min(3, len(fit["weights"]))
    comps, mean, lo, hi = fit["weights"][:k], fit["mean"], fit["lo"][:k], fit["hi"][:k]
    span = np.where(hi > lo, hi - lo, 1.0)
    names = ["red", "green", "blue"][:k]
    w = xr.DataArray(comps.astype(np.float32), dims=("band_rgb", "band"),
                     coords={"band_rgb": names, "band": da.band.values})
    centred = da - xr.DataArray(mean.astype(np.float32), dims="band", coords={"band": da.band.values})
    out = xr.dot(centred, w, dim="band")
    out = (out - xr.DataArray(lo.astype(np.float32), dims="band_rgb", coords={"band_rgb": names})) \
        / xr.DataArray(span.astype(np.float32), dims="band_rgb", coords={"band_rgb": names})
    out = out.clip(0, 1).rename(band_rgb="band")
    order = [d for d in ("time", "band", "y", "x") if d in out.dims]
    out = out.transpose(*order).astype(np.float32)
    out.name = "embeddings (principal components)"
    out.attrs = _keep(da)
    if da.rio.crs is not None:
        out = out.rio.write_crs(da.rio.crs)
    return out


def _sample(da: xr.DataArray, n: int) -> np.ndarray:
    """Up to about ``n`` pixels (n, band) with an embedding, spread over the years and the grid."""
    import math

    times = da.sizes.get("time", 1)
    picks = np.unique(np.linspace(0, times - 1, min(times, 6)).astype(int)) if "time" in da.dims else [None]
    per = max(1, n // len(picks))
    step = max(1, int(math.ceil(math.sqrt(da.sizes["y"] * da.sizes["x"] / per))))
    rows = []
    for i in picks:
        part = da if i is None else da.isel(time=int(i))
        values = np.asarray(part.isel(y=slice(None, None, step), x=slice(None, None, step))
                            .transpose("y", "x", "band").values, dtype=np.float64).reshape(-1, da.sizes["band"])
        rows.append(values[np.isfinite(values).all(axis=1)])
    return np.concatenate(rows) if rows else np.empty((0, da.sizes["band"]))
