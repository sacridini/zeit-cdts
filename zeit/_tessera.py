"""The TESSERA embeddings (geotessera.org) for ``zeit.load_embeddings``.

The published store is Zarr v3 with a group per UTM zone (``utm20``...), each on the grid
the embeddings were made on: ``embeddings (time, band, y, x)`` int8 and ``scales (time, y,
x)`` float32, one scale per pixel (a non-finite scale marks a pixel without an embedding:
water, or a gap), with the zone's CRS and transform in its attributes (the ``proj:`` and
``spatial:`` Zarr conventions, ``geoemb:`` for the embeddings). The ``geotessera`` library
knows where each version and variant is published and opens the store; zeit reads the
window of a region from it a block at a time, and dequantises each block as it is read.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._embeddings import BLOCKS, aligned_chunks, cube, window_of

DEFAULT_VERSION = "v1.1"
_CHUNK = 512   # cells a side of a block: whole 32 x 32 inner chunks, within the 4096 x 4096 shards
_SOUTH = 10_000_000.0   # false northing of the southern UTM zones


def _geotessera():
    try:
        import geotessera  # noqa: F401
        from geotessera.registry import dataset_for_location, zarr_store_url
        from geotessera.store import zarr_store
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("source='tessera' finds and opens the TESSERA store through the geotessera library, "
                          "which zeit installs on Python 3.12 or newer (the versions geotessera supports); it is "
                          "missing here (on Python 3.12+: pip install geotessera). A local copy of a store reads "
                          "without it: store='path/to/store'.") from e
    return zarr_store_url, dataset_for_location, zarr_store


def _open(version: Optional[str], variant: Optional[str], store: Any, cache_dir) -> Tuple[Any, Optional[Any], str]:
    """(the zarr store, geotessera's Dataset for it when it is a published one, a name for messages)."""
    if store is not None and not isinstance(store, (str, os.PathLike)):
        return store, None, "the given store"
    location = os.fspath(store) if store is not None else None
    if location is not None and os.path.exists(location):
        from zarr.storage import LocalStore

        return LocalStore(location, read_only=True), None, location
    zarr_store_url, dataset_for_location, zarr_store = _geotessera()
    if location is None:
        location = zarr_store_url(version or DEFAULT_VERSION, variant)
    if location.rstrip("/").endswith(".icechunk"):
        raise ValueError(f"TESSERA {version or DEFAULT_VERSION} {variant or ''} is published only as an Icechunk "
                         "repository; zeit reads the Zarr stores (the default v1.1 is one)")
    return zarr_store(location), dataset_for_location(location), location


def _identity(root_attrs: Dict[str, Any], dataset: Any, version: Optional[str], variant: Optional[str]):
    """(version, variant) of a store: geotessera's for a published store; otherwise the model of
    the store and the run it was copied from (so a mirror of a published store matches it),
    unless version=/variant= say otherwise."""
    if dataset is not None:
        return str(dataset.version), str(dataset.variant)
    model = str(root_attrs.get("geoemb:model", ""))
    found_version = model.rstrip("/").rsplit("/", 1)[-1] if model else ""
    source = str(root_attrs.get("geotessera:source_store", ""))
    found_variant = os.path.splitext(source.rstrip("/").rsplit("/", 1)[-1])[0] if source else ""
    if version is not None:
        found_version = version[1:] if version.lower().startswith("v") else version
    return found_version, variant if variant is not None else found_variant


def _zones(bbox) -> List[int]:
    w, _, e, _ = bbox
    first = min(60, int((w + 180) // 6) + 1)
    last = min(60, int((e - 1e-9 + 180) // 6) + 1)
    return list(range(first, last + 1))


def parts(bbox, years: Optional[List[int]], *, version: Optional[str], variant: Optional[str],
          depth: Optional[int], store: Any, cache_dir) -> Tuple[List[Any], Dict[str, Any]]:
    """The embeddings over ``bbox``, one lazy cube per UTM zone it touches, and their identity."""
    import zarr

    location, dataset, name = _open(version, variant, store, cache_dir)
    root = zarr.open_group(location, mode="r")
    attrs = dict(root.attrs)
    n = int(attrs.get("geoemb:dimensions", 128))
    depths = {int(d["dimensions"]): str(d["array"]) for d in attrs.get("geoemb:depths", [])} or {n: "embeddings"}
    depths.setdefault(n, "embeddings")
    if depth is None:
        depth = n
    if int(depth) not in depths:
        raise ValueError(f"depth={depth}: this store has the depths {sorted(depths)}"
                         + ("; Matryoshka prefixes come with the v2 stores" if len(depths) == 1 else ""))
    array = depths[int(depth)]
    groups = set(root.group_keys())
    out = []
    available: set = set()
    for zone in _zones(bbox):
        group = f"utm{zone:02d}"
        if group not in groups:
            continue
        part, zone_years = _zone(root[group], array, bbox, years, zone)
        available.update(zone_years)
        if part is not None:
            out.append(part)
    if not out:
        if years is not None and available and not set(years) <= available:
            raise ValueError(f"TESSERA ({name}) has the years {sorted(available)}, not {sorted(set(years) - available)}")
        raise ValueError(f"no TESSERA embeddings cover the region {tuple(round(v, 5) for v in bbox)} in {name}")
    found_version, found_variant = _identity(attrs, dataset, version, variant)
    meta = {"embedding_source": "tessera", "embedding_version": found_version, "embedding_variant": found_variant,
            "embedding_model": str(attrs.get("geoemb:model", "")),
            "embedding_build": str(attrs.get("geoemb:build_version", "")),
            "embedding_dimensions": int(depth),
            "embedding_license": "CC0-1.0",
            "embedding_citation": "Feng et al. (2025), TESSERA: Temporal Embeddings of Surface Spectra for Earth "
                                  "Representation and Analysis, arXiv:2506.20380"}
    return out, meta


def _zone(group, array: str, bbox, years: Optional[List[int]], zone: int):
    """(the lazy cube of the zone's window over ``bbox``, or None; the years the zone has)."""
    import dask.array as dsa
    from dask.base import tokenize
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    attrs = dict(group.attrs)
    stored = [int(v) for v in np.asarray(group["time"][:])]
    wanted = stored if years is None else years
    missing = [y for y in wanted if y not in stored]
    if missing:
        raise ValueError(f"TESSERA has the years {stored}, not {missing}")
    emb, scales = group[array], group["scales"]
    a, b, c, d, e, f = (float(v) for v in attrs["spatial:transform"][:6])
    transform = Affine(a, b, c, d, e, f)
    crs = CRS.from_user_input(attrs.get("proj:code") or 32600 + zone)
    window = window_of(bbox, crs, transform, emb.shape[-2:])
    if window is None:
        return None, stored
    row0, row1, col0, col1 = window
    # South of the equator the store keeps the northern zone's CRS, with negative northings;
    # a region entirely south of it is given in the southern zone's CRS, the same cells.
    shift = 0.0
    epsg = crs.to_epsg()
    if bbox[3] <= 0 and epsg is not None and 32601 <= epsg <= 32660:
        crs, shift = CRS.from_epsg(epsg + 100), _SOUTH
    name = "tessera-" + tokenize(str(getattr(emb, "store_path", "")), str(emb.store), array, zone, window, wanted)
    source = _Window(emb, scales, [stored.index(y) for y in wanted], window, name)
    chunks = ((1,) * len(wanted), (source.shape[1],), aligned_chunks(row0, row1 - row0, _CHUNK),
              aligned_chunks(col0, col1 - col0, _CHUNK))
    data = dsa.from_array(source, chunks=chunks, name=name, lock=False, asarray=True, fancy=False,
                          meta=np.empty((0, 0, 0, 0), dtype=np.float32))
    bands = [f"A{i:02d}" for i in range(source.shape[1])]
    return cube(data, times=wanted, bands=bands, transform=transform, crs=crs, row0=row0, col0=col0,
                y_shift=shift), stored


def dequantise(values: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """int8 ``(band, y, x)`` and their scales ``(y, x)`` as float32; NaN where the scale is not
    finite (no embedding)."""
    s = np.asarray(scales, dtype=np.float32)
    s = np.where(np.isfinite(s), s, np.float32(np.nan))
    return np.asarray(values).astype(np.float32) * s[None]


class _Window:
    """The cells ``window`` of a zone's arrays at some of its time steps, seen as a float32
    ``(time, band, y, x)`` array that dask reads a block at a time."""

    ndim = 4
    dtype = np.dtype(np.float32)

    def __init__(self, emb, scales, steps: List[int], window, token: str = ""):
        self.emb, self.scales, self.steps, self.token = emb, scales, list(steps), token
        row0, row1, col0, col1 = window
        self.row0, self.col0 = row0, col0
        self.shape = (len(self.steps), int(emb.shape[1]), row1 - row0, col1 - col0)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (4 - len(key))
        (t0, t1, ts), (b0, b1, bs), (y0, y1, ys), (x0, x1, xs) = (k.indices(n) for k, n in zip(key, self.shape))
        rows = slice(self.row0 + y0, self.row0 + max(y0, y1))
        cols = slice(self.col0 + x0, self.col0 + max(x0, x1))
        blocks = []
        for i in range(t0, t1, ts):
            t = self.steps[i]
            key = (self.token, t, b0, b1, rows.start, rows.stop, cols.start, cols.stop)
            raw = BLOCKS.get(key)
            if raw is None:
                raw = (np.asarray(self.emb[t, b0:max(b0, b1), rows, cols]), np.asarray(self.scales[t, rows, cols]))
                BLOCKS.put(key, raw)
            blocks.append(dequantise(*raw))
        if not blocks:
            return np.empty((0, max(0, b1 - b0), max(0, y1 - y0), max(0, x1 - x0)), dtype=np.float32)
        out = np.stack(blocks)
        if bs != 1 or ys != 1 or xs != 1:
            out = out[:, ::bs, ::ys, ::xs]
        return out
