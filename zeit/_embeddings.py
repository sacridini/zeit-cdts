"""``zeit.load_embeddings``: the yearly embeddings of Earth observation foundation models as a
zeit cube.

Two products are global, open and pixel-wise: TESSERA (Feng et al. 2025; 128 dimensions from
Sentinel-1 and Sentinel-2) and Google's AlphaEarth Foundations Satellite Embedding (64
dimensions). Both are one vector per 10 m pixel and year, which zeit holds like any other
cube, ``(time, band, y, x)``: classification, sampling, segmentation and clustering take it
as it is. One adapter per product reads it (``_tessera``, ``_alphaearth``); everything after
that does not know which product the cube came from. What it came from travels in its
attributes, so that a model trained on one product (or version) is not applied to another.
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

#: Attributes that say which embeddings a cube holds; two cubes are the same space only when
#: they all agree.
IDENTITY = ("embedding_source", "embedding_version", "embedding_variant")
#: GDAL metadata tag where save_raster stores them (with the rest of the embedding_* attributes).
TAG = "ZEIT_EMBEDDING"
SOURCES = ("tessera", "alphaearth")
_NAMES = {"tessera": "TESSERA", "alphaearth": "AlphaEarth"}
_RES = 10.0  # metres: the cells of both products
_DEG = 111319.49079327357  # metres per degree at the equator


def load_embeddings(
    region: Any = None,
    *,
    source: str,
    years: Any = None,
    like: Any = None,
    crs: Any = None,
    res: Any = None,
    resampling: str = "auto",
    version: Optional[str] = None,
    variant: Optional[str] = None,
    depth: Optional[int] = None,
    backend: Optional[str] = None,
    store: Any = None,
    chunks: Any = "auto",
    cache_dir: Any = None,
) -> xr.DataArray:
    """Read the yearly embeddings of a foundation model over a region, as a cube.

    Parameters
    ----------
    region
        Where: bounds ``(west, south, east, north)`` in longitude and latitude, a
        ``GeoDataFrame``/``GeoSeries`` or a vector file (any CRS), or shapely geometries
        in longitude and latitude. Cells outside polygons are NaN. Points (e.g. training
        samples) read the box around them. Optional with ``like``, whose extent is used.
    source
        ``"tessera"`` (TESSERA, 128 dimensions, Sentinel-1 and -2; through the
        ``geotessera`` library) or ``"alphaearth"`` (Google's AlphaEarth Foundations
        Satellite Embedding, 64 dimensions; its open copy on Source Cooperative, or Earth
        Engine with ``backend="gee"``).
    years
        A year, a list or a range (default: every year the product has, 2017-2025).
    like
        A raster (path, ``DataArray`` or ``Dataset``) whose grid the result takes, as in
        ``load_raster``, e.g. a Landsat cube to classify with the embeddings as features.
    crs, res
        Without ``like``: the CRS (and cell size, default 10 m) of the result. By default
        the native grid of the product: a region within one UTM zone keeps the zone's CRS
        and the cells the embeddings were made on; a region across zones needs ``crs``.
    resampling
        With ``like``, ``crs`` or ``res``: ``"auto"`` (default) takes the nearest cell
        (interpolating embeddings makes vectors the model never produced), and the mean of
        the cells (``"average"``) when the target cells are at least 1.5 times larger, the
        way embeddings are usually taken to a coarser scale. Or any method of
        ``load_raster``.
    version, variant
        TESSERA: the dataset version (default ``"v1.1"``, the complete global run) and its
        variant (default: the version's own). Embeddings of different versions or variants
        are different spaces; the cube records them.
    depth
        TESSERA v2 stores: the first ``depth`` dimensions only (a Matryoshka prefix, which
        those stores are trained for and publish as arrays of their own).
    backend
        AlphaEarth: ``"source.coop"`` (default; open COGs, no account) or ``"gee"`` (the
        ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` collection, downloaded with an Earth
        Engine account into ``cache_dir``).
    store
        A copy of the product to read instead of the public one: TESSERA, the URL or path of
        a Zarr store (or a ``zarr`` store object); AlphaEarth, a URL or folder with
        ``aef_index.parquet`` and the COGs in its ``<year>/<zone>/`` layout.
    chunks
        ``"auto"`` (default): a lazy cube, read block by block (one year and every
        dimension of 512 x 512 cells for TESSERA, 1024 x 1024 for AlphaEarth, on the
        product's own blocks). ``None``: read it all into memory. A dict of chunk sizes.
    cache_dir
        Where downloads are kept: the AlphaEarth file index (78 MB, refreshed monthly) and
        the Earth Engine downloads. Default ``~/.cache/zeit/embeddings``.

    Returns
    -------
    xarray.DataArray
        ``(time, band, y, x)`` float32, ``time`` on January 1 of each year, ``band`` the
        dimensions (``"A00"``, ``"A01"``...), NaN where there is no embedding (water in
        AlphaEarth, gaps), georeferenced through ``.rio``. Its attributes say which
        embeddings it holds (``embedding_source``, ``embedding_version``,
        ``embedding_variant``, the model, licence and attribution); ``save_raster`` keeps
        them, ``train_classifier`` and ``zeit.ai.train`` record them in the model, and
        ``classify``/``zeit.ai.predict`` refuse a cube of other embeddings.

    Notes
    -----
    Embeddings summarise a whole year: there is no seasonal signal nor physical unit in
    them, and the algorithms that fit one (LandTrendr, CCDC, BFAST, phenology, TWDTW,
    indices, unmixing...) refuse such a cube. What they are for: ``train_classifier`` and
    ``classify`` (few labelled points go a long way), ``similarity``, ``embedding_change``
    with ``extract_events``, ``som``, ``snic``, ``zeit.ai``; ``zeit.plot`` shows them
    through the first three principal components.

    TESSERA's south-of-the-equator data is stored in the northern UTM CRS (negative
    northings); a region entirely south of the equator is given in the southern UTM CRS
    (the same cells, 10,000 km apart in northing), as Landsat and AlphaEarth are.

    Training points read the whole blocks they fall on: when the same region is then
    classified, ``.persist()`` the cube (or ``chunks=None``) to read it once.

    Examples
    --------
    >>> emb = zeit.load_embeddings("aoi.gpkg", source="tessera", years=range(2018, 2025))
    >>> rf = zeit.train_classifier(emb.sel(time="2024"), "samples.gpkg")
    >>> classes = zeit.classify(emb.sel(time="2024"), rf)
    >>> aef = zeit.load_embeddings((-63.0, -10.0, -62.9, -9.9), source="alphaearth", years=2024)
    >>> on_landsat = zeit.load_embeddings(source="tessera", like=landsat, years=2020)  # 30 m, averaged
    """
    if not isinstance(source, str) or source.lower() not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    source = source.lower()
    from ._warp import RESAMPLING_NAMES

    if resampling != "auto" and resampling not in RESAMPLING_NAMES:
        raise ValueError(f"unknown resampling {resampling!r}; use 'auto' or one of {', '.join(RESAMPLING_NAMES)}")
    if region is None and like is None:
        raise ValueError("give the region to read (bounds in longitude and latitude, geometries or a vector file), "
                         "or like= a raster whose extent to read")
    if like is not None and (crs is not None or res is not None):
        raise ValueError("give like= or crs=/res=, not both: like= already sets the CRS and the cells")
    if source == "tessera" and backend is not None:
        raise ValueError("backend= is for AlphaEarth; TESSERA is read from its Zarr store")
    if source == "alphaearth":
        for name, value in (("version", version), ("variant", variant), ("depth", depth)):
            if value is not None:
                raise ValueError(f"{name}= is for TESSERA; AlphaEarth has one version of 64 dimensions")
    if chunks is not None and chunks != "auto" and not isinstance(chunks, dict):
        raise ValueError(f"chunks must be 'auto', None or a dict of chunk sizes, got {chunks!r}")

    from ._load import grid_of

    grid = grid_of(like) if like is not None else None
    bbox, shapes = _region(region, grid)
    wanted = _years(years)
    cache = Path(os.path.expanduser(os.fspath(cache_dir))) if cache_dir is not None else _default_cache()

    if source == "tessera":
        from . import _tessera

        parts, meta = _tessera.parts(bbox, wanted, version=version, variant=variant, depth=depth, store=store,
                                     cache_dir=cache)
    else:
        from . import _alphaearth

        parts, meta = _alphaearth.parts(bbox, wanted, backend=backend, store=store, cache_dir=cache, crs=crs)

    da = _assemble(parts, grid, crs, res, resampling, bbox)
    if shapes is not None:
        from ._load import _clip

        da = _clip(da, shapes)
    da.name = "embeddings"
    da.attrs = {k: v for k, v in da.attrs.items() if k == "_FillValue"}
    da.attrs.update(meta)
    if chunks is None:
        da = da.load()
    elif isinstance(chunks, dict):
        da = da.chunk(chunks)
    return da


# ---------------------------------------------------------------------------
# What a cube holds
# ---------------------------------------------------------------------------

def embedding_meta(data: Any) -> Optional[Dict[str, str]]:
    """The ``embedding_*`` identity of a cube or Dataset, or None when it holds no embeddings
    (or lost its attributes on the way, e.g. through arithmetic)."""
    attrs = getattr(data, "attrs", None)
    if not isinstance(attrs, dict) or not attrs.get("embedding_source"):
        return None
    return {k: str(attrs.get(k, "")) for k in IDENTITY}


def describe(meta: Dict[str, str]) -> str:
    name = _NAMES.get(meta.get("embedding_source", ""), meta.get("embedding_source", "?"))
    extra = [v for v in (meta.get("embedding_version"), meta.get("embedding_variant")) if v]
    return f"{name} ({', '.join(extra)})" if extra else name


def check_same(trained: Optional[Dict[str, str]], data: Any, what: str) -> None:
    """Refuse ``data`` when it holds other embeddings than the model was trained on."""
    if not trained:
        return
    meta = embedding_meta(data)
    if meta is not None and meta != dict(trained):
        raise ValueError(f"{what} was trained on {describe(trained)} embeddings; this cube holds "
                         f"{describe(meta)}. Embeddings of different products, versions or variants are "
                         "different spaces: train on the embeddings you classify.")


def refuse(data: Any, name: str) -> None:
    """Stop an algorithm that fits a physical, seasonal signal from running on embeddings."""
    meta = embedding_meta(data) if isinstance(data, (xr.DataArray, xr.Dataset)) else None
    if meta is not None:
        raise ValueError(f"{name}: the cube holds {describe(meta)} embeddings, yearly summaries with no "
                         f"physical unit nor seasonal signal for {name} to fit. Use them with "
                         "zeit.train_classifier/classify, zeit.similarity, zeit.embedding_change, zeit.som or "
                         "zeit.snic.")


def tags_of(data: Any) -> Dict[str, str]:
    """The GDAL metadata tag ``save_raster`` writes for a cube of embeddings (empty otherwise)."""
    attrs = getattr(data, "attrs", None)
    if not isinstance(attrs, dict) or not attrs.get("embedding_source"):
        return {}
    keep = {k: v for k, v in attrs.items() if str(k).startswith("embedding_") and isinstance(v, (str, int, float))}
    return {TAG: json.dumps(keep)}


def attrs_from_tag(tag: Optional[str]) -> Dict[str, Any]:
    if not tag:
        return {}
    try:
        value = json.loads(tag)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


# ---------------------------------------------------------------------------
# Region, years, grid
# ---------------------------------------------------------------------------

def _default_cache() -> Path:
    return Path.home() / ".cache" / "zeit" / "embeddings"


def _years(years: Any) -> Optional[List[int]]:
    if years is None:
        return None
    values = [years] if np.isscalar(years) else list(years)
    out = []
    for v in values:
        if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.strip().isdigit()):
            out.append(int(v))
        else:
            out.append(pd.Timestamp(v).year)
    if not out:
        raise ValueError("years is empty")
    return sorted(set(out))


def _region(region: Any, grid: Any) -> Tuple[Tuple[float, float, float, float], Any]:
    """(bounds in longitude and latitude, the polygons to clip to in a GeoSeries or None)."""
    import geopandas as gpd
    from rasterio.warp import transform_bounds

    if region is None:
        left, top = grid.transform.c, grid.transform.f
        right = left + grid.transform.a * grid.shape[1]
        bottom = top + grid.transform.e * grid.shape[0]
        bbox = transform_bounds(grid.crs, "EPSG:4326", min(left, right), min(top, bottom), max(left, right),
                                max(top, bottom), densify_pts=21)
        return _check_bbox(bbox), None
    if isinstance(region, (tuple, list)) and len(region) == 4 and all(np.isscalar(v) for v in region):
        return _check_bbox(tuple(float(v) for v in region)), None
    if isinstance(region, (str, os.PathLike)):
        region = gpd.read_file(region)
    if isinstance(region, (gpd.GeoDataFrame, gpd.GeoSeries)):
        series = region.geometry if isinstance(region, gpd.GeoDataFrame) else region
        if series.crs is None:
            raise ValueError("the region has no CRS; set it (e.g. gdf.set_crs(4326)) or give bounds in longitude and "
                             "latitude")
        series = gpd.GeoSeries(series.to_crs(4326).values, crs=4326)
    else:
        from shapely.geometry import shape

        geoms = region if isinstance(region, (list, tuple)) else [region]
        geoms = [g if hasattr(g, "geom_type") else shape(g) for g in geoms]
        series = gpd.GeoSeries(geoms, crs=4326)
    series = series[~series.is_empty & series.notna()]
    if series.empty:
        raise ValueError("the region is empty")
    w, s, e, n = series.total_bounds
    polygons = series.geom_type.isin(["Polygon", "MultiPolygon"])
    if not polygons.any():   # points or lines: the box around them, a cell further each way
        pad = _RES / _DEG
        return _check_bbox((w - pad, s - pad, e + pad, n + pad)), None
    return _check_bbox((w, s, e, n)), series


def _check_bbox(bbox) -> Tuple[float, float, float, float]:
    w, s, e, n = (float(v) for v in bbox)
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"region bounds {bbox} are not (west, south, east, north) in longitude and latitude; "
                         "for bounds in another CRS give a GeoDataFrame or a vector file (regions across the "
                         "antimeridian: read each side on its own)")
    return w, s, e, n


def window_of(bbox, crs, transform, shape) -> Optional[Tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) of the cells of a grid (``transform``, ``shape``) that cover a
    longitude/latitude box, or None when the box misses it."""
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    t = transform
    col0 = int(math.floor((left - t.c) / t.a + 1e-6))
    col1 = int(math.ceil((right - t.c) / t.a - 1e-6))
    row0 = int(math.floor((top - t.f) / t.e + 1e-6))
    row1 = int(math.ceil((bottom - t.f) / t.e - 1e-6))
    h, w = shape
    row0, row1, col0, col1 = max(0, row0), min(h, row1), max(0, col0), min(w, col1)
    if row1 <= row0 or col1 <= col0:
        return None
    return row0, row1, col0, col1


def aligned_chunks(start: int, length: int, size: int) -> Tuple[int, ...]:
    """Chunks of ``length`` cells from ``start`` whose edges fall on multiples of ``size``
    (the product's own blocks), so that no block of the store is read by two chunks."""
    edges = [start] + list(range((start // size + 1) * size, start + length, size)) + [start + length]
    return tuple(int(b - a) for a, b in zip(edges[:-1], edges[1:]) if b > a)


def cube(data, *, times: Sequence[int], bands: Sequence[str], transform, crs, row0: int, col0: int,
         y_shift: float = 0.0) -> xr.DataArray:
    """A ``(time, band, y, x)`` cube of the cells ``row0..``, ``col0..`` of a grid."""
    from rasterio.transform import Affine

    h, w = data.shape[-2:]
    t = transform
    x = t.c + t.a * (col0 + np.arange(w) + 0.5)
    y = t.f + y_shift + t.e * (row0 + np.arange(h) + 0.5)
    da = xr.DataArray(data, dims=("time", "band", "y", "x"),
                      coords={"time": pd.DatetimeIndex([pd.Timestamp(int(v), 1, 1) for v in times]),
                              "band": list(bands), "y": y, "x": x})
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(Affine(t.a, 0.0, t.c + t.a * col0, 0.0, t.e, t.f + y_shift + t.e * row0))
    return da.rio.write_nodata(np.nan, encoded=False)


def _assemble(parts: List[xr.DataArray], grid: Any, crs: Any, res: Any, resampling: str, bbox) -> xr.DataArray:
    """The parts (one per UTM zone) as one cube: as they are, or on a grid."""
    from rasterio.crs import CRS

    from ._warp import to_grid

    if grid is None and crs is None and res is None:
        crss = sorted({CRS.from_user_input(p.rio.crs).to_string() for p in parts})
        if len(crss) > 1:
            raise ValueError(f"the region spans {len(crss)} UTM zones ({', '.join(crss)}); give crs= (e.g. one of "
                             "them, or an equal-area CRS) or like= to put them on one grid")
        if len(parts) == 1:
            return parts[0]
    if grid is None:
        target = CRS.from_user_input(crs if crs is not None else parts[0].rio.crs)
        grid = _grid_for(bbox, target, res)
    method = _method(resampling, grid)
    out = None
    for part in parts:
        warped = to_grid(part, grid, method)
        out = warped if out is None else out.fillna(warped)
    return out.rio.write_nodata(np.nan, encoded=False)


def _grid_for(bbox, crs, res):
    """Cells of ``res`` (default 10 m, or its size in degrees for a geographic CRS) over the
    region in ``crs``, on multiples of the cell size (as the products' own grids are)."""
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds

    from ._warp import _grid

    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    if res is None:
        rx = ry = _RES / _DEG if crs.is_geographic else _RES
    else:
        rx, ry = (res, res) if np.isscalar(res) else tuple(res)
        rx, ry = float(rx), float(ry)
        if not (rx > 0 and ry > 0):
            raise ValueError(f"res must be positive, got {res!r}")
    x0 = math.floor(left / rx + 1e-6) * rx
    y0 = math.ceil(top / ry - 1e-6) * ry
    w = max(1, math.ceil((right - x0) / rx - 1e-6))
    h = max(1, math.ceil((y0 - bottom) / ry - 1e-6))
    return _grid(Affine(rx, 0.0, x0, 0.0, -ry, y0), (h, w), crs)


def _method(resampling: str, grid) -> str:
    if resampling != "auto":
        return resampling
    t = grid.transform
    size = math.sqrt(abs(t.a * t.e))
    if grid.crs.is_geographic:
        size *= _DEG
    return "average" if size >= 1.5 * _RES else "nearest"
