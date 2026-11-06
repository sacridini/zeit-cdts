"""Sentinel-2's sun and view angles of every pixel, ``zeit.sentinel2_angles``, from a tile's ``MTD_TL.xml``.

ESA gives each tile's angles as 23 x 23 grids with a 5 km step (``Geometric_Info/Tile_Angles``): the sun's zenith
and azimuth, and the view zenith and azimuth of every band and detector (a detector's grid is NaN outside its
swath). The grids' origin is the tile's upper left corner, ``ULX`` and ``ULY`` of ``Tile_Geocoding``, and they run
east (columns) and south (rows) from it. This module fuses the detectors of a band into one grid and interpolates
it to the pixels of a cube, for ``zeit.nbar``. Written from ESA's product description (the XML itself) and
Roy et al. (2017, RSE 199:25-38, Sec. 3.2), who derive the angles of each pixel "by bilinear interpolation from the
5 km grid values", as the HLS User Guide (Sec. 6.3) does for HLS S30 with the second red edge band's view angles.
"""
import datetime
import functools
import os
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

#: Band names and the ``bandId`` ESA numbers them with in ``Viewing_Incidence_Angles_Grids``.
_BAND_IDS = {"B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6, "B08": 7, "B8A": 8, "B09": 9,
             "B10": 10, "B11": 11, "B12": 12}
#: The names of Earth Search's assets (and of the roles of ``zeit.nbar``).
_ALIASES = {"coastal": "B01", "blue": "B02", "green": "B03", "red": "B04", "rededge1": "B05", "rededge2": "B06",
            "rededge3": "B07", "nir": "B08", "nir08": "B8A", "nir09": "B09", "cirrus": "B10", "swir16": "B11",
            "swir22": "B12"}
#: The metadata assets of the STAC catalogs that carry Sentinel-2 L2A: Earth Search, Planetary Computer.
_ASSET_KEYS = ("granule_metadata", "granule-metadata")
#: Rows of the strips the angles are computed in (memory and threads).
_STRIP = 512


def _band_id(band: Any) -> Optional[int]:
    """ESA's ``bandId`` of a band name (``B06``, ``b8a``, ``rededge2``, an integer id), None if it is not one."""
    if isinstance(band, (int, np.integer)) and not isinstance(band, bool):
        return int(band) if 0 <= band <= 12 else None
    name = str(band)
    key = _ALIASES.get(name.lower(), name).upper()
    if key in _BAND_IDS:
        return _BAND_IDS[key]
    if len(key) == 2 and key[0] == "B" and key[1].isdigit() and "B0" + key[1] in _BAND_IDS:
        return _BAND_IDS["B0" + key[1]]
    return None


# --- reading the XML ----------------------------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _kids(element: ET.Element, name: str) -> List[ET.Element]:
    return [c for c in element if _local(c.tag) == name]


def _first(element: ET.Element, name: str, what: str) -> ET.Element:
    """The first descendant called `name`."""
    for c in element.iter():
        if _local(c.tag) == name:
            return c
    raise ValueError(f"MTD_TL.xml: no {name} in {what}: is it a Sentinel-2 tile metadata file?")


def _grid(element: ET.Element, what: str) -> Tuple[np.ndarray, float, float]:
    """A ``Zenith`` or ``Azimuth`` element: its values ``(rows, columns)`` and its column and row steps in metres."""
    col = float(_first(element, "COL_STEP", what).text)
    row = float(_first(element, "ROW_STEP", what).text)
    rows = [np.array(v.text.split(), dtype=np.float64) for v in element.iter() if _local(v.tag) == "VALUES"]
    if not rows or any(r.size != rows[0].size for r in rows):
        raise ValueError(f"MTD_TL.xml: the {what} grid has no rows or ragged ones")
    return np.stack(rows), col, row


class _Tile:
    """What the XML says of one tile: the geocoding and the angle grids (sun, and each band's detectors)."""

    def __init__(self, name: str, crs: str, ulx: float, uly: float, step: Tuple[float, float], sun: np.ndarray,
                 view: Dict[int, List[np.ndarray]], sensing: Optional[datetime.datetime]) -> None:
        self.name, self.crs, self.ulx, self.uly, self.step = name, crs, ulx, uly, step
        self.sun = sun          # (2, rows, columns): zenith, azimuth
        self.view = view        # band id -> [(2, rows, columns) of each detector]
        self.sensing = sensing


def _parse(text: Any, name: str) -> _Tile:
    root = ET.fromstring(text)
    geo = _first(root, "Tile_Geocoding", name)
    crs = _first(geo, "HORIZONTAL_CS_CODE", name).text.strip()
    positions = [p for p in geo.iter() if _local(p.tag) == "Geoposition"]
    if not positions:
        raise ValueError(f"MTD_TL.xml {name}: no Geoposition in Tile_Geocoding")
    pos = next((p for p in positions if p.get("resolution") == "10"), positions[0])
    ulx, uly = float(_first(pos, "ULX", name).text), float(_first(pos, "ULY", name).text)
    angles = _first(root, "Tile_Angles", name)

    sun_el = _first(angles, "Sun_Angles_Grid", name)
    zen, cs, rs = _grid(_first(sun_el, "Zenith", name), "sun zenith")
    azi, cs2, rs2 = _grid(_first(sun_el, "Azimuth", name), "sun azimuth")
    if zen.shape != azi.shape or (cs, rs) != (cs2, rs2):
        raise ValueError(f"MTD_TL.xml {name}: the sun's zenith and azimuth grids differ")
    sun = np.stack([zen, azi])
    view: Dict[int, List[np.ndarray]] = {}
    for g in _kids(angles, "Viewing_Incidence_Angles_Grids"):
        z, c1, r1 = _grid(_first(g, "Zenith", name), "view zenith")
        a, _, _ = _grid(_first(g, "Azimuth", name), "view azimuth")
        if z.shape != zen.shape or a.shape != zen.shape or (c1, r1) != (cs, rs):
            raise ValueError(f"MTD_TL.xml {name}: a view grid is not the sun grid's shape and step "
                             f"({zen.shape}, {cs} x {rs} m)")
        view.setdefault(int(g.get("bandId")), []).append(np.stack([z, a]))
    if not view:
        raise ValueError(f"MTD_TL.xml {name}: no Viewing_Incidence_Angles_Grids")
    sensing = None
    for c in root.iter():
        if _local(c.tag) == "SENSING_TIME" and c.text:
            try:
                sensing = datetime.datetime.fromisoformat(c.text.strip().replace("Z", "+00:00"))
            except ValueError:
                pass
            break
    return _Tile(name, crs, ulx, uly, (cs, rs), sun, view, sensing)


# --- finding the XML ----------------------------------------------------------------------------------------------

def _sign(href: str) -> str:
    """A Planetary Computer URL signed (if the package is there; others untouched)."""
    if "blob.core.windows.net" in href and "sig=" not in href:
        try:
            import planetary_computer
        except ImportError:
            return href
        return planetary_computer.sign(href)
    return href


def _href_of(item: Any) -> Optional[str]:
    assets = item.get("assets") if isinstance(item, dict) else getattr(item, "assets", None)
    if assets is None:
        return None
    for key in _ASSET_KEYS:
        if key in assets:
            a = assets[key]
            return a["href"] if isinstance(a, dict) else a.href
    return None


def _read(source: Any) -> Tuple[Any, str]:
    """The text of an MTD_TL.xml given as a STAC item (pystac or dict), a path, a URL or the XML itself, and a name."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "<bytes>"
    if not isinstance(source, (str, os.PathLike)):
        href = _href_of(source)
        if href is None:
            raise ValueError(f"sentinel2_angles: no {' or '.join(_ASSET_KEYS)} asset in the item (Earth Search and "
                             "Planetary Computer's sentinel-2-l2a items have one); pass the MTD_TL.xml's path or "
                             "URL")
        source = href
    s = os.fspath(source)
    if s.lstrip().startswith("<"):
        return s, "<xml>"
    if s.lower().startswith(("http://", "https://")):
        with urllib.request.urlopen(_sign(s), timeout=60) as r:
            return r.read(), s
    with open(s, "rb") as f:
        return f.read(), s


# --- fusing and interpolating -------------------------------------------------------------------------------------

def _fields(grids: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The detectors' ``(2, rows, columns)`` grids as one: ``(3, rows, columns)`` zenith, sin and cos of the
    azimuth, each zero where the cell has no detector, and the ``(rows, columns)`` 0/1 validity.

    A cell covered by several detectors (the swaths overlap) is the mean of their zeniths and the circular mean of
    their azimuths; a cell none covers is invalid. The azimuth's sine and cosine, which are what is interpolated,
    are normalized to a unit vector.
    """
    g = np.stack(grids)                      # (detectors, 2, rows, cols)
    z, a = g[:, 0], np.radians(g[:, 1])
    ok = np.isfinite(z) & np.isfinite(a)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        zen = np.where(ok, z, 0.0).sum(axis=0) / n
        s = np.where(ok, np.sin(np.where(ok, a, 0.0)), 0.0).sum(axis=0) / n
        c = np.where(ok, np.cos(np.where(ok, a, 0.0)), 0.0).sum(axis=0) / n
    norm = np.hypot(s, c)
    valid = (n > 0) & (norm > 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        s, c = s / norm, c / norm
    f = np.stack([zen, s, c])
    f[:, ~valid] = 0.0
    return f.astype(np.float32), valid.astype(np.float32)


class _Prepared:
    """A tile reduced to what one call needs: its geocoding and the fused grids of the sun and the view bands."""

    def __init__(self, tile: _Tile, band_ids: Sequence[int]) -> None:
        self.name, self.crs, self.ulx, self.uly, self.step = tile.name, tile.crs, tile.ulx, tile.uly, tile.step
        self.sensing = tile.sensing
        self.layers = [_fields([tile.sun])]
        for b in band_ids:
            if b not in tile.view:
                raise ValueError(f"MTD_TL.xml {tile.name}: no view angles for band id {b}")
            self.layers.append(_fields(tile.view[b]))


def _axis(pos: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell, weight of the next node and whether the position is inside the grid, along one axis."""
    ok = (pos >= 0) & (pos <= n - 1)
    cell = np.clip(np.floor(pos), 0, n - 2).astype(np.intp)
    return cell, np.clip(pos - cell, 0.0, 1.0).astype(np.float32), ok


def _segments(cell: np.ndarray):
    """The runs of equal cells of `cell` as (cell, rows), the rows a slice where they are contiguous."""
    for c in np.unique(cell):
        rows = np.flatnonzero(cell == c)
        if rows[-1] - rows[0] + 1 == rows.size:
            yield int(c), slice(int(rows[0]), int(rows[-1]) + 1)
        else:
            yield int(c), rows


def _interp_separable(f: np.ndarray, v: np.ndarray, yy: Tuple, xx: Tuple) -> np.ndarray:
    """The fields `f` (k, rows, cols) interpolated bilinearly at the pixels of the axis-aligned lattice `yy` x `xx`
    (each ``_axis``'s output), over the corners that are valid: ``(k, ny, nx)``, NaN where no corner is."""
    (iy, wy, oky), (ix, wx, okx) = yy, xx
    fx = f[:, :, ix] * (1 - wx) + f[:, :, ix + 1] * wx          # (k, rows, nx)
    vx = None if v.min() > 0 else v[:, ix] * (1 - wx) + v[:, ix + 1] * wx
    k, ny, nx = f.shape[0], iy.size, ix.size
    out = np.empty((k, ny, nx), dtype=np.float32)
    den = None if vx is None else np.empty((ny, nx), dtype=np.float32)
    w = wy[:, None]
    for c, rows in _segments(iy):
        wr = w[rows]
        for j in range(k):
            out[j, rows] = fx[j, c] + wr * (fx[j, c + 1] - fx[j, c])
        if den is not None:
            den[rows] = vx[c] + wr * (vx[c + 1] - vx[c])
    if den is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            out /= den
        out[:, den <= 0] = np.nan
    out[:, ~oky, :] = np.nan
    out[:, :, ~okx] = np.nan
    return out


def _interp_general(f: np.ndarray, v: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """As ``_interp_separable`` at pixels given by their fractional grid row and column (2-D arrays)."""
    n_r, n_c = v.shape
    (iy, wy, oky), (ix, wx, okx) = _axis(rows, n_r), _axis(cols, n_c)
    num = np.zeros((f.shape[0],) + rows.shape, dtype=np.float32)
    den = np.zeros(rows.shape, dtype=np.float32)
    for dy in (0, 1):
        for dx in (0, 1):
            w = (wy if dy else 1 - wy) * (wx if dx else 1 - wx)
            r, c = iy + dy, ix + dx
            den += w * v[r, c]
            num += w * f[:, r, c]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[:, (den <= 0) | ~oky | ~okx] = np.nan
    return out


def _angles_of(fields: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zenith and azimuth (degrees, 0-360) of interpolated ``(3, ...)`` zenith, sine and cosine."""
    az = np.degrees(np.arctan2(fields[1], fields[2]))
    np.add(az, 360.0, out=az, where=az < 0)
    return fields[0], az.astype(np.float32, copy=False)


@functools.lru_cache(maxsize=16)
def _transformer(src: str, dst: str):
    from pyproj import Transformer

    return Transformer.from_crs(src, dst, always_xy=True)


def _tile_block(prep: _Prepared, like_crs: Optional[str], same_crs: bool, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The angles ``(2 + 2 views, ny, nx)`` (SZA, SAA, then each view's VZA, VAA) of one tile at the pixel centres
    `y` x `x` (in `like_crs`), NaN outside the tile's grid or where no detector covers."""
    out = np.empty((2 * len(prep.layers),) + (y.size, x.size), dtype=np.float32)
    dx, dy = prep.step
    if same_crs:
        xx = _axis((x - prep.ulx) / dx, prep.layers[0][1].shape[1])
        yy = _axis((prep.uly - y) / dy, prep.layers[0][1].shape[0])
        for k, (f, v) in enumerate(prep.layers):
            z, a = _angles_of(_interp_separable(f, v, yy, xx))
            out[2 * k], out[2 * k + 1] = z, a
        return out
    gx, gy = np.meshgrid(x, y)
    tx, ty = _transformer(like_crs, prep.crs).transform(gx, gy)
    cols, rows = (tx - prep.ulx) / dx, (prep.uly - ty) / dy
    for k, (f, v) in enumerate(prep.layers):
        z, a = _angles_of(_interp_general(f, v, rows.astype(np.float32), cols.astype(np.float32)))
        out[2 * k], out[2 * k + 1] = z, a
    return out


def _date_block(preps: Sequence[_Prepared], like_crs: Optional[str], same: Sequence[bool], y: np.ndarray,
                x: np.ndarray) -> np.ndarray:
    """The angles of a date at `y` x `x`; with several tiles, the first that has a view angle at a pixel."""
    out = _tile_block(preps[0], like_crs, same[0], y, x)
    for p, s in zip(preps[1:], same[1:]):
        missing = np.isnan(out[2])
        if not missing.any():
            break
        more = _tile_block(p, like_crs, s, y, x)
        out[:, missing] = more[:, missing]
    return out


def _lazy_block(_block, preps, like_crs, same, y, x, block_info=None):
    (a, b), (c, d) = block_info[0]["array-location"][1:]
    return _date_block(preps, like_crs, same, y[a:b], x[c:d])


# --- the public function ------------------------------------------------------------------------------------------

def _entries(source: Any) -> List[Any]:
    """`source` as one entry per date: a list of dates, each a source or a list of sources (tiles of the date)."""
    items = getattr(source, "items", None)
    if isinstance(items, list):          # a pystac ItemCollection
        source = items
    if isinstance(source, (list, tuple)):
        return list(source)
    return [source]


def sentinel2_angles(
    source: Any,
    like: Any,
    *,
    view_band: str = "B06",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """The sun and view angles of every pixel of a Sentinel-2 cube, from the tiles' ``MTD_TL.xml``, for
    ``zeit.nbar``.

    ESA gives each tile's angles as 23 x 23 grids with a 5 km step (the sun's zenith and azimuth, and the view
    zenith and azimuth of every band and detector), starting at the tile's upper left corner (``ULX``, ``ULY``)
    and running east and south. A detector's grid is NaN where its swath does not reach; where two swaths overlap
    a cell has two values. Here the detectors of a band are fused first, cell by cell: the mean of the zeniths and
    the circular mean of the azimuths (sine and cosine) of the detectors that cover the cell, so a cell is valid if
    any does. Each angle is then interpolated bilinearly to the pixel centres (``Roy et al. 2017``, RSE
    199:25-38, Sec. 3.2), azimuths through their sine and cosine so that 0/360 is no seam, and over the valid
    corners of the cell only (the weights of the valid corners, rescaled): a pixel between the last valid node of a
    swath and the next, invalid one takes the valid corner's value (flat, within the 5 km cell), and a pixel with no
    valid corner, or outside the grid, is NaN. The pixels of the cube may be in another CRS than the tile's
    (UTM zone of a neighbouring tile, or geographic): their centres are projected to the tile's.

    Parameters
    ----------
    source
        The tile metadata of each date of ``like``: a STAC item (a ``pystac`` Item or its dict) of Earth Search
        (asset ``granule_metadata``) or Planetary Computer (``granule-metadata``, signed with ``planetary_computer``
        if it is installed), a path or URL of an ``MTD_TL.xml``, or the XML text. A list gives one per date, in the
        order of ``like.time`` (an entry may itself be a list of the tiles that make that date's mosaic: a pixel
        takes the angles of the first tile that has them). A single source serves a ``like`` with one date or
        none. A source whose sensing time is more than a day from the date of ``like`` is an error (the dates
        were mismatched).
    like
        The cube (``xarray.DataArray``) the angles must match: its CRS, ``y`` and ``x`` pixel centres and ``time``.
        With ``view_band="each"`` it also gives the bands.
    view_band
        The band whose view angles every band takes. The default, ``"B06"`` (the second red edge band,
        ``"rededge2"``), is HLS's choice (User Guide Sec. 6.3): one view geometry for all bands. ``"each"`` gives
        each band of ``like`` its own view angles (the detectors' offsets between bands are up to a fraction of a
        degree), as variables ``VZA`` and ``VAA`` with a ``band`` dimension (the bands of ``like`` that are
        Sentinel-2 bands), which ``zeit.nbar`` uses band by band. Any other band name, ``"B08"``, ``"red"``...,
        takes that band for all.
    chunks
        ``None`` computes now in memory unless ``like`` is a dask array, in which case the angles are lazy with
        its chunks over ``y`` and ``x``; a dict or ``"auto"`` makes them lazy with those chunks. In memory a
        10980 x 10980 tile is four float32 images of 480 MB per date: prefer lazy for many dates.
    n_jobs
        Threads of the in-memory computation (strips of rows): the processors zeit may use by default (``-1``).

    Returns
    -------
    xarray.Dataset
        ``SZA``, ``SAA``, ``VZA``, ``VAA`` in degrees (float32; azimuths 0-360, clockwise from north, as ESA
        gives them), ``(time, y, x)`` (``(y, x)`` for a cube without ``time``), on exactly the coordinates of
        ``like``, ready for ``zeit.nbar(like, angles)``.

    Examples
    --------
    >>> angles = zeit.sentinel2_angles(items, cube)           # items: the STAC items of the cube's dates
    >>> cube_nbar = zeit.nbar(cube, angles)
    """
    from ._threads import resolve

    if not isinstance(like, (xr.DataArray, xr.Dataset)):
        raise ValueError("like= must be the cube (an xarray DataArray) whose grid the angles must match")
    if not {"y", "x"} <= set(like.dims) or "y" not in like.coords or "x" not in like.coords:
        raise ValueError("like= needs y and x dimensions with coordinates (the pixel centres)")
    if like.rio.crs is None:
        raise ValueError("like= has no CRS: the pixel centres cannot be placed on the tile")
    has_time = "time" in like.dims
    n_dates = like.sizes["time"] if has_time else 1

    each = isinstance(view_band, str) and view_band.lower() == "each"
    if each:
        if "band" not in like.dims:
            raise ValueError("view_band='each' needs a like with a band dimension")
        bands = [b for b in like.band.values if _band_id(b) is not None]
        if not bands:
            raise ValueError(f"none of like's bands {[str(b) for b in like.band.values]} is a Sentinel-2 band")
        ids = [_band_id(b) for b in bands]
    else:
        vid = _band_id(view_band)
        if vid is None:
            raise ValueError(f"view_band= {view_band!r} is not a Sentinel-2 band (B01-B12, B8A, or 'each')")
        bands, ids = [], [vid]

    entries = _entries(source)
    if len(entries) != n_dates:
        raise ValueError(f"{len(entries)} sources for the {n_dates} date(s) of like: one MTD_TL.xml (or item) per "
                         "date, in the order of like.time")
    like_crs = like.rio.crs.to_wkt()
    from pyproj import CRS

    tile_cache: Dict[Any, _Prepared] = {}
    per_date: List[List[_Prepared]] = []
    times = like.time.values if has_time else [None]
    for entry, t in zip(entries, times):
        preps = []
        for s in (list(entry) if isinstance(entry, (list, tuple)) else [entry]):
            key = id(s) if not isinstance(s, (str, os.PathLike)) else str(s)
            if key not in tile_cache:
                text, name = _read(s)
                tile_cache[key] = _Prepared(_parse(text, name), ids)
            p = tile_cache[key]
            if t is not None and p.sensing is not None and np.issubdtype(np.asarray(t).dtype, np.datetime64):
                got = np.datetime64(p.sensing.astimezone(datetime.timezone.utc).replace(tzinfo=None), "s")
                if abs(got - np.asarray(t).astype("datetime64[s]")) > np.timedelta64(1, "D"):
                    raise ValueError(f"the tile {p.name} was sensed on {p.sensing:%Y-%m-%d}, the date of like is "
                                     f"{str(np.asarray(t).astype('datetime64[D]'))}: sources must be in the order "
                                     "of like.time")
            preps.append(p)
        per_date.append(preps)
    same = {}
    for preps in per_date:
        for p in preps:
            if p.crs not in same:
                same[p.crs] = CRS.from_user_input(like_crs).equals(CRS.from_user_input(p.crs), ignore_axis_order=True)

    x = np.asarray(like.x.values, dtype=np.float64)
    y = np.asarray(like.y.values, dtype=np.float64)
    nl = 2 * (1 + len(ids))
    lazy = chunks is not None or (like.chunks is not None if isinstance(like, xr.DataArray) else False)
    if lazy:
        import dask.array as da

        if chunks is None:
            ch = dict(zip(like.dims, like.chunks))
            ycs, xcs = ch["y"], ch["x"]
        else:
            probe = da.zeros((y.size, x.size), chunks=chunks if not isinstance(chunks, dict)
                             else (chunks.get("y", "auto"), chunks.get("x", "auto")), dtype=np.float32)
            ycs, xcs = probe.chunks
        dummy = da.zeros((nl, y.size, x.size), chunks=((nl,), ycs, xcs), dtype=np.float32)
        dates = [da.map_blocks(_lazy_block, dummy, [p for p in preps], like_crs, [same[p.crs] for p in preps], y, x,
                               dtype=np.float32, meta=np.empty((0, 0, 0), np.float32))
                 for preps in per_date]
        stack = da.stack(dates)                                  # (time, nl, y, x)
    else:
        stack = np.empty((n_dates, nl, y.size, x.size), dtype=np.float32)
        threads = resolve(n_jobs)
        for k, preps in enumerate(per_date):
            flags = [same[p.crs] for p in preps]
            spans = [(a, min(a + _STRIP, y.size)) for a in range(0, y.size, _STRIP)]

            def work(span, k=k, preps=preps, flags=flags):
                a, b = span
                stack[k, :, a:b] = _date_block(preps, like_crs, flags, y[a:b], x)

            if threads > 1 and len(spans) > 1:
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    list(pool.map(work, spans))
            else:
                for s in spans:
                    work(s)

    dims = ("time", "y", "x") if has_time else ("y", "x")
    coords = {k: like.coords[k] for k in ("time", "y", "x") if k in like.coords}
    take = (lambda i: stack[:, i]) if has_time else (lambda i: stack[0, i])
    long = {"SZA": "solar zenith angle", "SAA": "solar azimuth angle", "VZA": "view zenith angle",
            "VAA": "view azimuth angle"}
    data = {}
    for name, i in (("SZA", 0), ("SAA", 1)):
        data[name] = xr.DataArray(take(i), dims=dims, coords=coords)
    if each:
        for name, off in (("VZA", 2), ("VAA", 3)):
            arr = [take(off + 2 * j) for j in range(len(ids))]
            arr = da.stack(arr, axis=1 if has_time else 0) if lazy else np.stack(arr, axis=1 if has_time else 0)
            d = ("time", "band", "y", "x") if has_time else ("band", "y", "x")
            data[name] = xr.DataArray(arr, dims=d, coords={**coords, "band": bands})
    else:
        data["VZA"] = xr.DataArray(take(2), dims=dims, coords=coords)
        data["VAA"] = xr.DataArray(take(3), dims=dims, coords=coords)
    for name, v in data.items():
        v.attrs = {"long_name": long[name], "units": "degree"}
    ds = xr.Dataset(data)
    ds = ds.rio.write_crs(like.rio.crs)
    names = sorted({p.name for preps in per_date for p in preps})
    ds.attrs = {"sentinel2_angles": "ESA 5 km grids (MTD_TL.xml) fused over detectors by the mean per cell, "
                                    "bilinear to the pixel centres over valid corners, azimuths through sin/cos; "
                                    f"view_band={view_band}",
                "sources": "; ".join(names)}
    return ds
