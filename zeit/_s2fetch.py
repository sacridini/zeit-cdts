# SPDX-License-Identifier: GPL-3.0-or-later
"""Finding and fetching what ``zeit.level2`` reads: ``zeit.find_l1c``, ``zeit.download_l1c`` and ``zeit.download_dem``.

* ``find_l1c`` searches the ``sentinel-2-l1c`` collection of Earth Search (STAC) and lists the products.
* ``download_l1c`` downloads the files of a product that the correction reads (the two XML files and the 13 band files, in
  the layout of the ``.SAFE``) from the public bucket ``gcp-public-data-sentinel-2`` of Google Cloud Storage, which keeps the
  products in their original layout and needs no credential. (The assets of Earth Search point to another bucket, on S3,
  without ``MTD_MSIL1C.xml``.) Every file is written to ``<name>.part`` and renamed, its size and MD5 are checked against
  the listing of the bucket, and one that is already there with the right MD5 is not downloaded again.
* ``download_dem`` mosaics the Copernicus GLO-30 tiles (``cop-dem-glo-30`` of Earth Search, public COGs) that cover a scene,
  in the geometry of the source and in EPSG:4326, to one GeoTIFF.
"""

import base64
import datetime
import hashlib
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

#: The generic user agent of every request.
USER_AGENT = "zeit-python (+https://github.com/sacridini/zeit)"

_BUCKET = "gcp-public-data-sentinel-2"
_LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{_BUCKET}/o"
_FILE_URL = f"https://storage.googleapis.com/{_BUCKET}/"
_DONE = ".zeit-complete"
_L1C_COLLECTION = "sentinel-2-l1c"
_DEM_COLLECTION = "cop-dem-glo-30"
_MAX_JOBS = 8
_TRIES = 5
_TRANSIENT_STATUS = (408, 429, 500, 502, 503, 504)
#: Margin around the footprint of the scene, in degrees (6 pixels of the finest GLO-30 tile): the bilinear warp reads neighbours.
_DEM_MARGIN = 6.0 / 3600.0
DEM_NODATA = -32767

_PRODUCT = re.compile(r"^S2[A-Z]_MSIL1C_\d{8}T\d{6}_N\d{4}_R\d{3}_T(\d{2})([A-Z])([A-Z]{2})_\d{8}T\d{6}\.SAFE$")
_BANDS = ("01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12")
_WANTED = re.compile(r"^(?:MTD_MSIL1C\.xml|GRANULE/[^/]+/MTD_TL\.xml|GRANULE/[^/]+/IMG_DATA/[^/]+_B(?:" + "|".join(_BANDS) + r")\.jp2)$")
_TILE = re.compile(r"^(?:MGRS-)?T?(\d{2})([C-HJ-NP-X])([A-HJ-NP-Z]{2})$")


class _Transient(Exception):
    """An answer worth asking again (a busy server)."""


class ChecksumError(RuntimeError):
    """A downloaded file whose size or MD5 is not the one the bucket lists."""


# --- the HTTP layer (the tests replace ``_send``) -----------------------------------------------------------------------


def _send(url: str, params: Optional[Dict[str, Any]] = None, stream: bool = False):
    import requests

    return requests.get(url, params=params, stream=stream, timeout=60, headers={"User-Agent": USER_AGENT})


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _with_retry(fn: Callable[[], Any], what: str) -> Any:
    """``fn()``, asked again with a growing wait when the network fails or the server is busy; ``what`` names it in the error."""
    import requests

    from ._network import connection_failed, offline_message

    retriable = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError, _Transient, ChecksumError)
    for attempt in range(_TRIES):
        try:
            return fn()
        except retriable as e:
            if attempt == _TRIES - 1:
                if isinstance(e, ChecksumError):
                    raise
                if connection_failed(e):
                    raise ConnectionError(offline_message(what)) from e
                raise RuntimeError(f"{what}: failed {_TRIES} times, the last with: {e}") from e
            _sleep(min(2.0 ** attempt, 30.0))


def _check(response: Any, url: str) -> Any:
    code = response.status_code
    if code in _TRANSIENT_STATUS:
        response.close()
        raise _Transient(f"HTTP {code}")
    if code >= 400:
        response.close()
        raise RuntimeError(f"{url}: HTTP {code}")
    return response


def _get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    def go():
        r = _check(_send(url, params), url)
        try:
            return r.json()
        finally:
            r.close()

    return _with_retry(go, url)


def _md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode("ascii")


def _fetch_file(url: str, dest: Path, size: int, md5: str) -> bool:
    """Download ``url`` to ``dest`` through ``dest.part``, checking size and MD5 (base64, as the bucket lists it); False when
    ``dest`` was already there with that MD5."""
    if dest.is_file() and dest.stat().st_size == size and _md5_b64(dest) == md5:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def go():
        r = _check(_send(url, None, True), url)
        h = hashlib.md5()
        n = 0
        try:
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
        finally:
            r.close()
        got = base64.b64encode(h.digest()).decode("ascii")
        if n != size or got != md5:
            raise ChecksumError(f"{url}: downloaded {n} bytes with MD5 {got}, the bucket lists {size} bytes with MD5 {md5}")

    try:
        _with_retry(go, url)
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return True


def _run_all(jobs: Sequence[Callable[[], Any]], n_jobs: int) -> List[Any]:
    n = _n_jobs(n_jobs)
    if n == 1 or len(jobs) <= 1:
        return [j() for j in jobs]
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(j) for j in jobs]
        try:
            return [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise


def _n_jobs(n_jobs: int) -> int:
    if int(n_jobs) < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs}")
    return min(int(n_jobs), _MAX_JOBS)


# --- find_l1c -----------------------------------------------------------------------------------------------------------


def _stac_client(url: str):
    from pystac_client import Client

    return Client.open(url, headers={"User-Agent": USER_AGENT})


def _stamp(value: Any, end: bool) -> Optional[str]:
    """``value`` (a date string, ``date`` or ``datetime``) as an RFC 3339 instant; a bare date is the start or the end of its day."""
    import pandas as pd

    if value is None:
        return None
    ts = pd.Timestamp(value)
    if isinstance(value, str):
        date_only = len(value.strip()) <= 10
    else:
        date_only = not isinstance(value, datetime.datetime)     # a date (a Timestamp is a datetime)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if date_only and end:
        ts = ts.replace(hour=23, minute=59, second=59, microsecond=0)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tile_code(tile: str) -> str:
    m = _TILE.match(str(tile).strip().upper())
    if not m:
        raise ValueError(f"{tile!r} is not an MGRS tile such as '23LKC' (or 'T23LKC')")
    return "".join(m.groups())


def find_l1c(tile: Optional[str] = None, bbox: Optional[Sequence[float]] = None, start: Any = None, end: Any = None,
             max_cloud: float = 100.0, limit: Optional[int] = None):
    """Search Sentinel-2 Level 1C products in the ``sentinel-2-l1c`` collection of Earth Search.

    Exactly one of ``tile`` (an MGRS tile, ``"23LKC"`` or ``"T23LKC"``) and ``bbox`` (``(lon_min, lat_min, lon_max, lat_max)``,
    degrees) is needed. ``start`` and ``end`` are dates (``"2023-07-28"``, a ``date`` or a ``datetime``; a bare ``end`` date
    includes its whole day), either one optional. ``max_cloud`` is the largest cloud cover (percent) and ``limit`` the largest
    number of products (the most recent ones when there are more).

    Returns a ``pandas.DataFrame`` sorted by date with the columns ``product`` (the name of the ``.SAFE``, what
    ``download_l1c`` and ``level2`` take), ``datetime`` (UTC), ``tile``, ``cloud_cover`` (percent), ``baseline`` (processing
    baseline, ``"05.09"``) and ``platform``.
    """
    import pandas as pd

    from .cube import STAC_CATALOGS

    if (tile is None) == (bbox is None):
        raise ValueError("find_l1c: give exactly one of tile= and bbox=")
    kw: Dict[str, Any] = dict(collections=[_L1C_COLLECTION])
    query: Dict[str, Any] = {}
    if tile is not None:
        query["grid:code"] = {"eq": "MGRS-" + _tile_code(tile)}
    else:
        b = [float(v) for v in bbox]
        if len(b) != 4 or not (b[0] < b[2] and b[1] < b[3]):
            raise ValueError("find_l1c: bbox must be (lon_min, lat_min, lon_max, lat_max) with min < max")
        kw["bbox"] = b
    if max_cloud < 100:
        query["eo:cloud_cover"] = {"lte": float(max_cloud)}
    if query:
        kw["query"] = query
    t0, t1 = _stamp(start, False), _stamp(end, True)
    if t0 or t1:
        kw["datetime"] = f"{t0 or '..'}/{t1 or '..'}"
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("find_l1c: limit must be at least 1")
        kw["max_items"] = int(limit)
        kw["limit"] = min(int(limit), 100)
        kw["sortby"] = "-properties.datetime"       # the most recent N, whatever the default order of the server
    search = _stac_client(STAC_CATALOGS["earth_search"]).search(**kw)
    rows = []
    for item in search.items():
        p = item.properties
        uri = p.get("s2:product_uri")
        if not uri:
            raise ValueError(f"find_l1c: the item {item.id} has no s2:product_uri")
        if not str(uri).endswith(".SAFE"):
            uri = f"{uri}.SAFE"
        grid = str(p.get("grid:code") or "")
        rows.append(dict(product=str(uri), datetime=pd.Timestamp(item.datetime or p.get("datetime")),
                         tile=grid.replace("MGRS-", "") or _tile_of(str(uri)),
                         cloud_cover=p.get("eo:cloud_cover"), baseline=p.get("s2:processing_baseline"),
                         platform=p.get("platform")))
    cols = ["product", "datetime", "tile", "cloud_cover", "baseline", "platform"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    return df


def _tile_of(name: str) -> str:
    m = _PRODUCT.match(name)
    return "".join(m.groups()) if m else ""


# --- download_l1c -------------------------------------------------------------------------------------------------------


def _list_objects(prefix: str) -> List[Dict[str, Any]]:
    """The objects of the bucket under ``prefix`` (``name``, ``size`` and ``md5Hash``), page after page, without the
    ``*_$folder$`` placeholders."""
    params: Dict[str, Any] = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken", "maxResults": 1000}
    out: List[Dict[str, Any]] = []
    while True:
        page = _get_json(_LIST_URL, dict(params))
        out.extend(o for o in page.get("items", []) if not o["name"].endswith("$folder$"))
        token = page.get("nextPageToken")
        if not token:
            return out
        params["pageToken"] = token


def _product_name(product: Any) -> str:
    if hasattr(product, "to_dict") and not isinstance(product, str):      # a row of find_l1c
        product = product["product"]
    name = str(product).strip()
    name = name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not name.endswith(".SAFE"):
        name += ".SAFE"
    if not _PRODUCT.match(name):
        raise ValueError(f"{name!r} is not the name of a Sentinel-2 L1C product (such as "
                         "S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149.SAFE)")
    return name


def _download_product(name: str, dest: Path, n_jobs: int) -> Path:
    root = dest / name
    if (root / _DONE).is_file():
        return root
    zone, band, square = _PRODUCT.match(name).groups()
    prefix = f"tiles/{zone}/{band}/{square}/{name}/"
    objects = _list_objects(prefix)
    if not objects:
        raise ValueError(f"product {name} not found in the bucket {_BUCKET} (looked under {prefix}); "
                         "it may be too old or too new for the bucket, or reprocessed under another name")
    files = [(o["name"][len(prefix):], o) for o in objects if _WANTED.match(o["name"][len(prefix):])]
    rel = [r for r, _ in files]
    granules = {r.split("/")[1] for r in rel if r.startswith("GRANULE/")}
    bands = [r for r in rel if r.startswith("GRANULE/") and r.endswith(".jp2")]
    if "MTD_MSIL1C.xml" not in rel or len(granules) != 1 or len(bands) != len(_BANDS) or not any(r.endswith("/MTD_TL.xml") for r in rel):
        raise ValueError(f"product {name}: the bucket does not list MTD_MSIL1C.xml, one MTD_TL.xml and the 13 band files "
                         f"(found {len(rel)} of the needed files)")
    jobs = [(lambda r=r, o=o: _fetch_file(_FILE_URL + o["name"], root / r, int(o["size"]), o["md5Hash"])) for r, o in files]
    root.mkdir(parents=True, exist_ok=True)
    _run_all(jobs, n_jobs)
    (root / _DONE).write_text("")
    return root


def download_l1c(product: Any, dest: Any = None, n_jobs: int = 4):
    """Download a Sentinel-2 Level 1C product to a local ``.SAFE`` directory that ``zeit.level2`` reads.

    ``product`` is the name of the ``.SAFE`` (``"S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149.SAFE"``), or a row
    or a ``DataFrame`` of ``find_l1c`` (a ``DataFrame`` downloads all and returns the list). The files come from the public
    bucket ``gcp-public-data-sentinel-2`` of Google Cloud Storage, which needs no credential: only what ``level2`` reads (the two
    XML files and the 13 band files, not the true-colour image), in the layout of the product. Each file goes to ``<name>.part``
    and is renamed once its size and MD5 match the listing of the bucket; a file already there with the right MD5 is not
    downloaded again, so an interrupted download resumes. ``dest`` is the folder that gets the ``.SAFE`` (default: the ``l1c``
    cache, ``ZEIT_CACHE_DIR/l1c``); ``n_jobs`` is the number of simultaneous downloads (at most 8).

    Returns the ``pathlib.Path`` of the ``.SAFE`` directory. Raises ``ValueError`` for a product the bucket does not have.
    """
    from ._cache import cache_dir

    _n_jobs(n_jobs)
    target = Path(os.path.expanduser(os.fspath(dest))) if dest is not None else cache_dir("l1c")
    if hasattr(product, "iterrows"):
        return [_download_product(_product_name(n), target, n_jobs) for n in product["product"]]
    return _download_product(_product_name(product), target, n_jobs)


# --- download_dem -------------------------------------------------------------------------------------------------------


def _anonymous(href: str) -> str:
    """The HTTPS URL that reads ``href`` without credentials (``s3://bucket/key`` is public on ``bucket.s3.amazonaws.com``)."""
    m = re.match(r"^s3://([^/]+)/(.+)$", href)
    return f"https://{m.group(1)}.s3.amazonaws.com/{m.group(2)}" if m else href


def _dem_sources(bounds: Tuple[float, float, float, float]) -> List[str]:
    """The URLs of the GLO-30 tiles that touch ``bounds`` (west, south, east, north), by the ``cop-dem-glo-30`` collection of Earth Search."""
    from .cube import STAC_CATALOGS

    search = _stac_client(STAC_CATALOGS["earth_search"]).search(collections=[_DEM_COLLECTION], bbox=list(bounds))
    urls = []
    for item in search.items():
        asset = item.assets.get("data") or next(iter(item.assets.values()))
        urls.append(_anonymous(asset.href))
    return sorted(set(urls))


def _read_window(url: str, bounds: Tuple[float, float, float, float]):
    """The window of the tile at ``url`` that covers ``bounds`` as ``(float32 array, transform)``, or None if it has none."""
    import numpy as np
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.windows import Window, from_bounds

    def go():
        with rasterio.Env(GDAL_HTTP_USER_AGENT=USER_AGENT, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2",
                          GDAL_HTTP_TIMEOUT="60"):
            with rasterio.open(url) as d:
                raw = from_bounds(*bounds, transform=d.transform)
                eps = 1e-6      # whole cells that touch the box, clipped to the tile
                c0, r0 = max(int(math.floor(raw.col_off + eps)), 0), max(int(math.floor(raw.row_off + eps)), 0)
                c1 = min(int(math.ceil(raw.col_off + raw.width - eps)), d.width)
                r1 = min(int(math.ceil(raw.row_off + raw.height - eps)), d.height)
                if c1 <= c0 or r1 <= r0:
                    return None
                win = Window(c0, r0, c1 - c0, r1 - r0)
                a = d.read(1, window=win).astype(np.float32)
                if d.nodata is not None:
                    a[(a == d.nodata) | np.isnan(a)] = DEM_NODATA
                a[~np.isfinite(a)] = DEM_NODATA
                return a, d.window_transform(win)

    try:
        return _with_retry(lambda: _retry_io(go, RasterioIOError), url)
    except RuntimeError as e:
        raise RuntimeError(f"could not read the DEM tile {url}: {e}") from e


def _retry_io(fn: Callable[[], Any], error: type) -> Any:
    try:
        return fn()
    except error as e:
        raise _Transient(str(e)) from e


def _mosaic(pieces: Sequence[Tuple[Any, Any]]):
    """The pieces ``(array, transform)`` (all EPSG:4326) on one grid of the finest cell of the pieces, aligned to the finest
    piece, ``DEM_NODATA`` where no piece has data. Each is copied by nearest neighbour in its own geometry, so no tile is
    resampled onto another one's grid but onto the finest cell (the 1.5 and 2 ratios of the tiles beyond 50 degrees of latitude)."""
    import numpy as np
    from affine import Affine

    ref = min((t for _, t in pieces), key=lambda t: (abs(t.a), abs(t.e)))
    xres = min(abs(t.a) for _, t in pieces)
    yres = min(abs(t.e) for _, t in pieces)
    west = min(t.c for _, t in pieces)
    north = max(t.f for _, t in pieces)
    east = max(t.c + t.a * a.shape[1] for a, t in pieces)
    south = min(t.f + t.e * a.shape[0] for a, t in pieces)
    eps = 1e-6
    west = ref.c + math.floor((west - ref.c) / xres + eps) * xres
    north = ref.f - math.floor((ref.f - north) / yres + eps) * yres
    width = int(math.ceil((east - west) / xres - eps))
    height = int(math.ceil((north - south) / yres - eps))
    transform = Affine(xres, 0.0, west, 0.0, -yres, north)
    out = np.full((height, width), DEM_NODATA, dtype=np.float32)
    cx = west + xres * (np.arange(width) + 0.5)
    cy = north - yres * (np.arange(height) + 0.5)
    # coarser pieces first, so that the finest one wins where they overlap; the cell of the piece that holds the centre of each
    # cell of the mosaic is copied (nearest neighbour), and a piece's own no-data never blanks what another one gave
    for a, t in sorted(pieces, key=lambda p: -abs(p[1].a)):
        cols = np.floor((cx - t.c) / t.a + eps).astype(np.int64)
        rows = np.floor((cy - t.f) / t.e + eps).astype(np.int64)
        ok_c = np.flatnonzero((cols >= 0) & (cols < a.shape[1]))
        ok_r = np.flatnonzero((rows >= 0) & (rows < a.shape[0]))
        if ok_c.size == 0 or ok_r.size == 0:
            continue
        src = a[np.ix_(rows[ok_r], cols[ok_c])]
        dst = out[np.ix_(ok_r, ok_c)]
        out[np.ix_(ok_r, ok_c)] = np.where(src != DEM_NODATA, src, dst)
    return out, transform


def _scene_bounds(path: Path) -> Tuple[float, float, float, float]:
    """The footprint of a local ``.SAFE`` (its 10 m band grid) in degrees, the UTM edges densified."""
    from rasterio.warp import transform_bounds

    from ._level2 import Safe, _band_grid

    g = _band_grid(Safe(path))
    h, w = g.shape
    t = g.transform
    xs = (t.c, t.c + t.a * w)
    ys = (t.f, t.f + t.e * h)
    return transform_bounds(g.crs, "EPSG:4326", min(xs), min(ys), max(xs), max(ys), densify_pts=101)


def _dem_target(dest: Any, name: str) -> Path:
    from ._cache import cache_dir

    if dest is None:
        return cache_dir("dem") / name
    p = Path(os.path.expanduser(os.fspath(dest)))
    return p if p.suffix.lower() in (".tif", ".tiff") else p / name


def download_dem(area: Any, dest: Any = None, n_jobs: int = 4):
    """A Copernicus GLO-30 DEM covering a scene or a box, as one GeoTIFF in EPSG:4326 (what ``level2(dem=)`` takes).

    ``area`` is a local Sentinel-2 ``.SAFE`` directory (the footprint of the scene) or ``(lon_min, lat_min, lon_max, lat_max)``
    in degrees. The tiles come from the ``cop-dem-glo-30`` collection of Earth Search (public COGs, read anonymously, only the
    windows that are needed) and are mosaicked in the geometry of the source, never resampled one by one onto the grid of the
    scene: the area is the footprint plus a margin of 6 arc-seconds (the bilinear warp reads neighbours). Where tiles of
    different resolution meet (the GLO-30 widens its cells in longitude beyond 50 degrees of latitude) the mosaic has the
    finest cell. Where there is no tile (the sea) the value is ``-32767``, ``level2``'s no-data.

    ``dest`` is a folder (default: the ``dem`` cache) or a ``.tif`` path; a file already there is reused. ``n_jobs`` is the
    number of simultaneous reads (at most 8). Returns the ``pathlib.Path`` of the GeoTIFF.

    Raises ``ValueError`` for a scene that crosses the antimeridian (not handled) and for an area without any tile (all sea:
    use ``topography=False`` in ``level2``).
    """
    import numpy as np
    import rasterio
    from rasterio.crs import CRS

    n = _n_jobs(n_jobs)
    if isinstance(area, (str, os.PathLike)):
        path = Path(os.path.expanduser(os.fspath(area)))
        if not path.is_dir():
            raise ValueError(f"download_dem: {area} is not a directory; give a .SAFE directory or (lon_min, lat_min, lon_max, lat_max)")
        w, s, e, nn = _scene_bounds(path)
        name = f"cop_glo30_{path.name[:-5] if path.name.endswith('.SAFE') else path.name}.tif"
    else:
        vals = [float(v) for v in area]
        if len(vals) != 4:
            raise ValueError("download_dem: area must be a .SAFE directory or (lon_min, lat_min, lon_max, lat_max)")
        w, s, e, nn = vals
        name = "cop_glo30_{:.4f}_{:.4f}_{:.4f}_{:.4f}.tif".format(w, s, e, nn)
    if w > e or w < -180.0 or e > 180.0:
        raise ValueError(f"download_dem: the area ({w:.4f}, {s:.4f}, {e:.4f}, {nn:.4f}) crosses the antimeridian or leaves "
                         "[-180, 180]; that is not handled: split the area in two")
    if not (-90.0 <= s < nn <= 90.0) or not w < e:
        raise ValueError(f"download_dem: invalid area ({w}, {s}, {e}, {nn}): need lon_min < lon_max and -90 <= lat_min < lat_max <= 90")
    out = _dem_target(dest, name)
    if out.is_file():
        return out
    bounds = (max(w - _DEM_MARGIN, -180.0), max(s - _DEM_MARGIN, -90.0), min(e + _DEM_MARGIN, 180.0), min(nn + _DEM_MARGIN, 90.0))
    urls = _dem_sources(bounds)
    pieces = [p for p in _run_all([(lambda u=u: _read_window(u, bounds)) for u in urls], n) if p is not None]
    if not pieces:
        raise ValueError(f"download_dem: no Copernicus GLO-30 tile covers ({w:.4f}, {s:.4f}, {e:.4f}, {nn:.4f}) (the sea has none); "
                         "there is no terrain to correct: use topography=False in level2")
    data, transform = _mosaic(pieces)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    try:
        with rasterio.Env(GDAL_NUM_THREADS=str(min(n, _MAX_JOBS))):
            with rasterio.open(part, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype="float32",
                               crs=CRS.from_epsg(4326), transform=transform, nodata=DEM_NODATA, tiled=True, blockxsize=256,
                               blockysize=256, compress="deflate", predictor=3) as d:
                d.write(data, 1)
        os.replace(part, out)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return out
