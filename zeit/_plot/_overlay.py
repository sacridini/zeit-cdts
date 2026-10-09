"""What goes under and over the maps: a web basemap and vector outlines.

Basemap tiles are Web Mercator; the data can be in any CRS. Instead of reprojecting every
frame, the viewer gets two small control grids (cell -> lon/lat and lon/lat -> cell) and
places each tile by interpolating in them, which is exact to well under a pixel at tile
scale. Static plots fetch the tiles here, mosaic them and warp the mosaic to the data's CRS.
"""

import hashlib
import io
import math
import os
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._data import Frames

PROVIDERS = {
    "satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community", 19),
    "osm": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", "© OpenStreetMap contributors", 19),
    "light": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
              "Tiles © Esri — Esri, DeLorme, NAVTEQ", 16),
    "dark": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
             "Tiles © Esri — Esri, DeLorme, NAVTEQ", 16),
    "topo": ("https://a.tile.opentopomap.org/{z}/{x}/{y}.png", "© OpenStreetMap contributors, SRTM | © OpenTopoMap", 17),
}
PROVIDERS["esri"] = PROVIDERS["satellite"]
USER_AGENT = "zeit-cdts (https://github.com/sacridini/zeit-cdts)"


def basemap_provider(basemap: Any) -> Optional[Dict[str, Any]]:
    """{"url", "attribution", "max_zoom"} for ``basemap``: a name (``"satellite"``, ``"osm"``,
    ``"light"``, ``"dark"``, ``"topo"``), an xyzservices provider or its name
    (``"Esri.WorldImagery"``), or a ``{z}/{x}/{y}`` URL template."""
    if basemap is None or basemap is False:
        return None
    if basemap is True:
        basemap = "satellite"
    if isinstance(basemap, str) and basemap.lower() in PROVIDERS:
        url, attribution, max_zoom = PROVIDERS[basemap.lower()]
        return {"url": url, "attribution": attribution, "max_zoom": max_zoom}
    if isinstance(basemap, str) and "{z}" in basemap:
        return {"url": basemap, "attribution": "", "max_zoom": 19}
    provider = basemap
    if isinstance(basemap, str):
        try:
            import xyzservices.providers as xyz
        except ImportError as err:
            raise ValueError(f"unknown basemap {basemap!r}; use one of {sorted(PROVIDERS)} or a URL template") from err
        try:
            provider = xyz.query_name(basemap)
        except ValueError as err:
            raise ValueError(f"unknown basemap {basemap!r}; use one of {sorted(PROVIDERS)}, an xyzservices name "
                             "or a URL template") from err
    if hasattr(provider, "build_url"):
        return {"url": provider.build_url(), "attribution": provider.get("attribution", ""),
                "max_zoom": int(provider.get("max_zoom", 19))}
    raise ValueError(f"unknown basemap {basemap!r}")


# ---------------------------------------------------------------------------
# Geometry: cells (full resolution, top row first) <-> lon/lat
# ---------------------------------------------------------------------------

def _cell_transform(frames: Frames, flip: bool):
    """(left, top, dx, dy) mapping top-first full-resolution cells to the data's CRS."""
    left, right, bottom, top = frames.extent()
    left, right = min(left, right), max(left, right)
    bottom, top = min(bottom, top), max(bottom, top)
    return left, top, (right - left) / frames.width, (top - bottom) / frames.height


def geo_grids(frames: Frames, crs: Any, n: int = 17) -> Optional[Dict[str, Any]]:
    """Control grids between top-first cells and lon/lat, for placing basemap tiles."""
    from rasterio.warp import transform as warp

    if crs is None:
        return None
    left, top, dx, dy = _cell_transform(frames, False)
    W, H = frames.width, frames.height
    # forward: a regular grid of cells -> lon/lat
    cx, cy = np.meshgrid(np.linspace(0, W, n), np.linspace(0, H, n))
    lon, lat = warp(crs, "EPSG:4326", (left + cx * dx).ravel().tolist(), (top - cy * dy).ravel().tolist())
    lon, lat = np.asarray(lon), np.asarray(lat)
    # inverse: a regular lon/lat lattice over the data's footprint (with a margin) -> cells
    pad_lon = (lon.max() - lon.min()) * 0.75 + 1e-6
    pad_lat = (lat.max() - lat.min()) * 0.75 + 1e-6
    lon0, lon1 = max(-180.0, lon.min() - pad_lon), min(180.0, lon.max() + pad_lon)
    lat0, lat1 = max(-85.0, lat.min() - pad_lat), min(85.0, lat.max() + pad_lat)
    m = 2 * n - 1
    glon, glat = np.meshgrid(np.linspace(lon0, lon1, m), np.linspace(lat1, lat0, m))
    xs, ys = warp("EPSG:4326", crs, glon.ravel().tolist(), glat.ravel().tolist())
    icx = (np.asarray(xs) - left) / dx
    icy = (top - np.asarray(ys)) / dy
    return {
        "forward": {"nx": n, "ny": n, "x0": 0.0, "x1": float(W), "y0": 0.0, "y1": float(H),
                    "a": _flat(lon), "b": _flat(lat)},
        "inverse": {"nx": m, "ny": m, "x0": lon0, "x1": lon1, "y0": lat1, "y1": lat0,
                    "a": _flat(icx), "b": _flat(icy)},
    }


def _flat(values: np.ndarray) -> List[float]:
    return [float(v) if np.isfinite(v) else 0.0 for v in np.asarray(values, dtype=float).ravel()]


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def read_vectors(vector: Any, crs: Any):
    """A GeoSeries of the geometries in ``vector`` (path, GeoDataFrame, GeoSeries, shapely
    geometry or a list of them), in ``crs``."""
    import geopandas as gpd

    if isinstance(vector, (str, os.PathLike)):
        geoms = gpd.read_file(vector).geometry
    elif hasattr(vector, "geometry") and hasattr(vector, "crs"):
        geoms = vector.geometry if hasattr(vector, "columns") else vector
    else:
        items = vector if isinstance(vector, (list, tuple)) else [vector]
        geoms = gpd.GeoSeries(list(items), crs=crs)
    geoms = geoms[~geoms.is_empty & geoms.notna()]
    if crs is not None and geoms.crs is not None and geoms.crs != crs:
        geoms = geoms.to_crs(crs)
    return geoms


def vector_paths(frames: Frames, vector: Any, crs: Any, step: int) -> Tuple[Dict[str, Any], bytes]:
    """Outlines as top-first full-resolution cell coordinates: (content, float32 xy buffer).

    ``content["paths"]`` lists (offset, length, closed) of each path in the buffer; points
    are paths of length 1."""
    left, top, dx, dy = _cell_transform(frames, False)
    geoms = read_vectors(vector, crs).simplify(min(abs(dx), abs(dy)) * step / 2)
    coords: List[np.ndarray] = []
    paths: List[List[int]] = []
    offset = 0

    def add(xy: np.ndarray, closed: bool) -> None:
        nonlocal offset
        xy = np.asarray(xy, dtype=float)[:, :2]
        cells = np.column_stack([(xy[:, 0] - left) / dx, (top - xy[:, 1]) / dy]).astype(np.float32)
        coords.append(cells.ravel())
        paths.append([offset, len(cells), int(closed)])
        offset += len(cells)

    def walk(geom) -> None:
        kind = geom.geom_type
        if kind == "Polygon":
            add(geom.exterior.coords, True)
            for ring in geom.interiors:
                add(ring.coords, True)
        elif kind in ("LineString", "LinearRing"):
            add(geom.coords, kind == "LinearRing")
        elif kind == "Point":
            add([geom.coords[0]], False)
        elif hasattr(geom, "geoms"):
            for part in geom.geoms:
                walk(part)

    for geom in geoms:
        walk(geom)
    buffer = np.concatenate(coords).astype(np.float32).tobytes() if coords else b""
    return {"paths": paths, "count": len(paths)}, buffer


# ---------------------------------------------------------------------------
# Static basemap (matplotlib)
# ---------------------------------------------------------------------------

def _tile_xy(lon: float, lat: float, z: int) -> Tuple[float, float]:
    n = 2 ** z
    lat = max(min(lat, 85.0511), -85.0511)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _fetch(url: str) -> Optional[bytes]:
    cache = os.path.join(os.path.expanduser("~"), ".cache", "zeit", "tiles")
    path = os.path.join(cache, hashlib.sha1(url.encode()).hexdigest() + ".img")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read()
    except Exception:  # noqa: BLE001 - offline or tile missing: the map is drawn without it
        return None
    os.makedirs(cache, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def static_basemap(frames: Frames, crs: Any, basemap: Any, width_px: int = 1024):
    """(rgb uint8 image, (left, right, bottom, top)) of the basemap behind the frames, in
    the data's CRS; None when offline or not georeferenced."""
    from PIL import Image
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    provider = basemap_provider(basemap)
    if provider is None or crs is None:
        return None
    left, right, bottom, top = frames.extent()
    left, right, bottom, top = min(left, right), max(left, right), min(bottom, top), max(bottom, top)
    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", left, bottom, right, top)
    span = max(lon1 - lon0, 1e-9)
    z = int(max(0, min(provider["max_zoom"], round(math.log2(360.0 * width_px / (span * 256.0))))))
    tx0, ty0 = _tile_xy(lon0, lat1, z)
    tx1, ty1 = _tile_xy(lon1, lat0, z)
    xs, ys = range(int(tx0), int(tx1) + 1), range(int(ty0), int(ty1) + 1)
    if len(xs) * len(ys) > 64:
        return None
    mosaic = np.zeros((len(ys) * 256, len(xs) * 256, 3), dtype=np.uint8)
    got = 0
    for j, ty in enumerate(ys):
        for i, tx in enumerate(xs):
            data = _fetch(provider["url"].format(z=z, x=tx, y=ty, s="a", r=""))
            if data is None:
                continue
            tile = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = tile[:256, :256]
            got += 1
    if not got:
        return None
    world = 20037508.342789244
    size = 2 * world / 2 ** z
    src_transform = from_bounds(-world + xs[0] * size, world - (ys[-1] + 1) * size,
                                -world + (xs[-1] + 1) * size, world - ys[0] * size, mosaic.shape[1], mosaic.shape[0])
    out_w = width_px
    out_h = max(1, int(round(width_px * (top - bottom) / (right - left))))
    dst = np.zeros((3, out_h, out_w), dtype=np.uint8)
    for band in range(3):
        reproject(mosaic[..., band], dst[band], src_transform=src_transform, src_crs="EPSG:3857",
                  dst_transform=from_bounds(left, bottom, right, top, out_w, out_h), dst_crs=crs,
                  resampling=Resampling.bilinear)
    return np.moveaxis(dst, 0, -1), (left, right, bottom, top), provider["attribution"]
