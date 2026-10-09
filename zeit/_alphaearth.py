"""Google's AlphaEarth Foundations Satellite Embedding for ``zeit.load_embeddings``.

The open copy on Source Cooperative holds, per year and UTM zone (``2024/20S/``), COGs of
8192 x 8192 cells and 64 int8 bands, indexed by ``aef_index.parquet`` (each file's UTM and
longitude/latitude bounds). The files are "bottom-up": the first row is the southern one and
the y resolution positive, so each block is read and flipped (what the ``.vrt`` published
next to each file does through a warp). Values map to [-1, 1] as ``sign(v) * (v / 127.5)**2``
and -128 marks a pixel without an embedding; each embedding has length 1. The same product is
the ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` collection of Earth Engine (``backend="gee"``).
"""

import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._embeddings import BLOCKS, aligned_chunks, cube, window_of

# The bucket's own endpoint rather than the data.source.coop gateway, which drops requests
# under the load of a region read (as geotessera found for TESSERA's copy there).
ROOT = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual"
INDEX = "aef_index.parquet"
COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
ATTRIBUTION = "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind."
_REFRESH_DAYS = 30
_CHUNK = 1024      # cells a side of a block: the files' own blocks
_BAND_CHUNK = 16   # bands of a block: each band is a request of its own, and blocks run in parallel
_COLUMNS = ["path", "year", "utm_zone", "crs", "utm_west", "utm_south", "utm_east", "utm_north",
            "wgs84_west", "wgs84_south", "wgs84_east", "wgs84_north"]
_GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff,.tif",
                 GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="1", VSI_CACHE="TRUE",
                 GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES")


def _meta(backend: str) -> Dict[str, Any]:
    return {"embedding_source": "alphaearth", "embedding_version": "1", "embedding_variant": "annual",
            "embedding_model": "AlphaEarth Foundations", "embedding_dimensions": 64,
            "embedding_license": "CC-BY-4.0", "embedding_attribution": ATTRIBUTION,
            "embedding_backend": backend,
            "embedding_citation": "Brown et al. (2025), AlphaEarth Foundations: An embedding field model for "
                                  "accurate and efficient global mapping from sparse label data, arXiv:2507.22291"}


def parts(bbox, years: Optional[List[int]], *, backend: Optional[str], store: Any, cache_dir: Path,
          crs: Any) -> Tuple[List[Any], Dict[str, Any]]:
    backend = "source.coop" if backend is None else str(backend).lower()
    if backend == "gee":
        if store is not None:
            raise ValueError("store= is a copy of the COGs; backend='gee' reads Earth Engine")
        return [_gee(bbox, years, cache_dir, crs)], _meta("gee")
    if backend != "source.coop":
        raise ValueError(f"backend must be 'source.coop' or 'gee', got {backend!r}")
    root = ROOT if store is None else os.fspath(store).rstrip("/\\")
    index = _index(root, cache_dir)
    available = sorted(int(y) for y in index.year.unique())
    wanted = available if years is None else years
    missing = [y for y in wanted if y not in available]
    if missing:
        raise ValueError(f"AlphaEarth has the years {available}, not {missing}")
    w, s, e, n = bbox
    hit = index[index.year.isin(wanted) & (index.wgs84_west < e) & (index.wgs84_east > w)
                & (index.wgs84_south < n) & (index.wgs84_north > s)]
    out = []
    for zone, files in hit.groupby("utm_zone", sort=True):
        part = _zone(root, files, bbox, wanted)
        if part is not None:
            out.append(part)
    if not out:
        raise ValueError(f"no AlphaEarth embeddings cover the region {tuple(round(v, 5) for v in bbox)}")
    return out, _meta("source.coop")


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

def _is_remote(root: str) -> bool:
    return root.startswith(("http://", "https://"))


def _index(root: str, cache_dir: Path) -> pd.DataFrame:
    """The file index: ``rel`` (the file's path under the root) and its zone, CRS and bounds."""
    if _is_remote(root):
        path = cache_dir / "alphaearth" / INDEX
        stale = not path.exists() or time.time() - path.stat().st_mtime > _REFRESH_DAYS * 86400
        if stale:
            try:
                _download(f"{root}/{INDEX}", path)
            except Exception:
                if not path.exists():
                    raise
                # offline: the copy we have still lists the files
    else:
        path = Path(root) / INDEX
        if not path.exists():
            raise FileNotFoundError(f"store={root!r}: no {INDEX} in it (a copy of the AlphaEarth COGs keeps the "
                                    "index at its root, with the files in <year>/<zone>/)")
    return _read_index(str(path), path.stat().st_mtime)


@lru_cache(maxsize=4)
def _read_index(path: str, mtime: float) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=_COLUMNS).to_pandas()
    table["rel"] = [p.split("/annual/", 1)[1] if "/annual/" in p else p for p in table.pop("path")]
    table["year"] = table["year"].astype(int)
    return table


def _download(url: str, path: Path) -> None:
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(partial, "wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
    os.replace(partial, path)


# ---------------------------------------------------------------------------
# A zone's files as one lazy cube
# ---------------------------------------------------------------------------

def _zone(root: str, files: pd.DataFrame, bbox, years: List[int]):
    import dask.array as dsa
    from dask.base import tokenize
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    crs = CRS.from_user_input(files.crs.iloc[0])
    # The files of a zone lie on one grid of 10 m cells; the part covers those the region touches.
    left, top = float(files.utm_west.min()), float(files.utm_north.max())
    width = int(round((float(files.utm_east.max()) - left) / 10.0))
    height = int(round((top - float(files.utm_south.min())) / 10.0))
    transform = Affine(10.0, 0.0, left, 0.0, -10.0, top)
    window = window_of(bbox, crs, transform, (height, width))
    if window is None:
        return None
    row0, row1, col0, col1 = window
    by_year = {int(y): [(_location(root, r.rel), int(round((top - r.utm_north) / 10.0)),
                         int(round((r.utm_west - left) / 10.0))) for r in group.itertuples()]
               for y, group in files.groupby("year")}
    source = _Files([by_year.get(y, []) for y in years], window)
    chunks = ((1,) * len(years), _band_chunks(64), aligned_chunks(row0, row1 - row0, _CHUNK),
              aligned_chunks(col0, col1 - col0, _CHUNK))
    name = "alphaearth-" + tokenize(root, str(files.utm_zone.iloc[0]), window, years)
    data = dsa.from_array(source, chunks=chunks, name=name, lock=False, asarray=True, fancy=False,
                          meta=np.empty((0, 0, 0, 0), dtype=np.float32))
    return cube(data, times=years, bands=[f"A{i:02d}" for i in range(64)], transform=transform, crs=crs,
                row0=row0, col0=col0)


def _band_chunks(n: int) -> Tuple[int, ...]:
    return tuple(min(_BAND_CHUNK, n - i) for i in range(0, n, _BAND_CHUNK))


def _location(root: str, rel: str) -> str:
    if _is_remote(root):
        return f"/vsicurl/{root}/{rel}"
    return os.path.join(root, *rel.split("/"))


def dequantise(values: np.ndarray) -> np.ndarray:
    """int8 values as the embedding's [-1, 1]; NaN for -128 (no embedding)."""
    v = np.asarray(values).astype(np.float32)
    out = np.sign(v) * (v / np.float32(127.5)) ** 2
    out[np.asarray(values) == -128] = np.nan
    return out


class _Files:
    """The cells ``window`` of a zone's grid, from the files of each year (``(path, row, col)``:
    the file and where its top-left cell is on the grid), as a float32 ``(time, band, y, x)``
    array that dask reads a block at a time."""

    ndim = 4
    dtype = np.dtype(np.float32)

    def __init__(self, files: List[List[Tuple[str, int, int]]], window):
        self.files = files
        row0, row1, col0, col1 = window
        self.row0, self.col0 = row0, col0
        self.shape = (len(files), 64, row1 - row0, col1 - col0)

    def __getitem__(self, key):
        import rasterio

        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        (t0, t1, ts), (b0, b1, bs), (y0, y1, ys), (x0, x1, xs) = (k.indices(n) for k, n in zip(key, self.shape))
        nt, nb = len(range(t0, t1, ts)), max(0, b1 - b0)
        h, w = max(0, y1 - y0), max(0, x1 - x0)
        out = np.full((nt, nb, h, w), np.nan, dtype=np.float32)
        top, left = self.row0 + y0, self.col0 + x0
        if nb and h and w:
            with rasterio.Env(**_GDAL_ENV):
                for i, t in enumerate(range(t0, t1, ts)):
                    for path, frow, fcol in self.files[t]:
                        r0, r1 = max(top, frow), top + h
                        c0, c1 = max(left, fcol), left + w
                        if r0 >= r1 or c0 >= c1:
                            continue
                        key = (path, b0, b1, r0, r1, c0, c1)
                        cached = BLOCKS.get(key)
                        if cached is None:
                            cached = (_read(path, frow, fcol, r0, r1, c0, c1, b0, b1),)
                            BLOCKS.put(key, cached)
                        values = cached[0]
                        if values.size:
                            rr, cc = values.shape[1:]
                            out[i, :, r0 - top:r0 - top + rr, c0 - left:c0 - left + cc] = dequantise(values)
        if bs != 1 or ys != 1 or xs != 1:
            out = out[:, ::bs, ::ys, ::xs]
        return out


def _read(path: str, frow: int, fcol: int, r0: int, r1: int, c0: int, c1: int, b0: int, b1: int) -> np.ndarray:
    """Bands ``b0..b1`` of the cells ``r0..r1``, ``c0..c1`` of the zone's grid that a file
    (its top-left cell at ``frow``, ``fcol``) holds, top-down, int8 as stored; an empty array
    when the file holds none of them."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as src:
        fh, fw = src.height, src.width
        r1, c1 = min(r1, frow + fh), min(c1, fcol + fw)
        if r1 <= r0 or c1 <= c0:
            return np.empty((b1 - b0, 0, 0), dtype=np.int8)
        up = src.transform.e > 0   # bottom-up: file row 0 is the southern one
        rows = (fh - (r1 - frow), fh - (r0 - frow)) if up else (r0 - frow, r1 - frow)
        values = src.read(list(range(b0 + 1, b1 + 1)), window=Window(c0 - fcol, rows[0], c1 - c0, rows[1] - rows[0]))
    return values[:, ::-1, :] if up else values


# ---------------------------------------------------------------------------
# Earth Engine
# ---------------------------------------------------------------------------

def _gee(bbox, years: Optional[List[int]], cache_dir: Path, crs: Any):
    """Each year of the collection over the region, downloaded once into ``cache_dir``, in the
    UTM zone of the region's centre (or ``crs``) at 10 m."""
    import xarray as xr
    from dask.base import tokenize
    from rasterio.crs import CRS

    from ._load import load_raster

    if years is None:
        raise ValueError("backend='gee': give years= (the collection has 2017 onwards)")
    w, s, e, n = bbox
    if crs is None:
        zone = min(60, int(((w + e) / 2 + 180) // 6) + 1)
        crs = CRS.from_epsg((32600 if (s + n) / 2 >= 0 else 32700) + zone)
    crs = CRS.from_user_input(crs)
    folder = cache_dir / "alphaearth-gee"
    layers = []
    for year in years:
        path = folder / f"alphaearth_{year}_{tokenize(tuple(round(v, 7) for v in bbox), crs.to_string())}.tif"
        if not path.exists():
            folder.mkdir(parents=True, exist_ok=True)
            if _gee_download(year, bbox, crs.to_string(), str(path)) is None or not path.exists():
                raise RuntimeError(f"the Earth Engine download of AlphaEarth {year} failed")
        layer = load_raster(str(path), chunks={"y": _CHUNK, "x": _CHUNK})
        if "band" not in layer.dims:
            layer = layer.expand_dims(band=1)
        layer = layer.assign_coords(band=[f"A{i:02d}" for i in range(layer.sizes["band"])])
        layers.append(layer.astype(np.float32))
    da = xr.concat(layers, dim=pd.Index(pd.DatetimeIndex([pd.Timestamp(y, 1, 1) for y in years]), name="time"),
                   coords="minimal", compat="override", join="override")
    da = da.transpose("time", "band", "y", "x").where(np.isfinite(da))
    da = da.rio.write_crs(crs).rio.write_transform(layers[0].rio.transform())
    return da.rio.write_nodata(np.nan, encoded=False)


def _gee_download(year: int, bbox, crs: str, path: str) -> Optional[str]:
    """Download a year of the collection over ``bbox`` (an Earth Engine session must be up:
    ``zeit.gee.auth.initialize_gee``)."""
    import ee

    from .gee.downloader import download_gee_image

    roi = ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)
    image = (ee.ImageCollection(COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
             .filterBounds(roi).mosaic().toFloat())
    return download_gee_image(image, roi, path, scale=10, crs=crs)
