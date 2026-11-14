"""``zeit.fields_tiled``: ``zeit.fields`` over rasters too large for one call (phase 24g).

The raster is cut into square tiles. For each tile the boundary probability is computed on the tile's
*core* plus an overlap all around it (so the watershed sees what lies beyond the seam) and only the core is
kept (pass 1, ``boundary.tif``). The labels are then made the same way from the probability and the image
of core plus overlap (pass 2: hysteresis, labelling, ``min_size`` and the final watershed, the very code of
``zeit.fields``) and only the core is kept. The labels of different tiles are joined across the seams where
the extended labellings of both tiles say two neighbouring pixels belong to one field (a union-find over the
tiles' label ids), and the polygons are made block by block and dissolved across the blocks by field.

Nothing as large as the raster is ever held in memory: windows are read and written one at a time, and the
stitching keeps the strips along the seams, one id per tile label and the union-find.
"""

import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from . import _fields as _f

#: What ``fields_tiled`` accepts of ``fields``' options (the others are errors that say why).
_OPTIONS = ("gradient", "method", "realizations", "marker_density", "min_distance", "marker_radius", "n_neighbors",
            "coloring", "low", "high", "min_size", "final", "mask_threshold", "band", "combine")
#: Layers above this many pixels are sampled for the statistics of the gradient (below: read whole, exactly).
_SAMPLE_PIXELS = 16_000_000
_SAMPLE_BLOCK = 256
_SIDES = ("top", "bottom", "left", "right")


# --------------------------------------------------------------------------------------
# The input, by window
# --------------------------------------------------------------------------------------
class _Source:
    """The raster read by window: a path (``load_raster`` with ``window=``) or a DataArray."""

    def __init__(self, src: Any):
        self.src = src
        if isinstance(src, xr.DataArray):
            self.cube = _f._as_cube(src)
            self.height, self.width = self.cube.sizes["y"], self.cube.sizes["x"]
            from ._warp import transform_of

            self.transform = transform_of(self.cube)
            try:
                self.crs = self.cube.rio.crs
            except Exception:
                self.crs = None
        else:
            import rasterio

            self.cube = None
            with rasterio.open(str(src)) as ds:
                self.height, self.width = ds.height, ds.width
                self.transform, self.crs = ds.transform, ds.crs

    def read(self, c0: int, r0: int, c1: int, r1: int) -> xr.DataArray:
        if self.cube is not None:
            return self.cube.isel(y=slice(r0, r1), x=slice(c0, c1))
        from ._load import load_raster

        return load_raster(str(self.src), window=(c0, r0, c1, r1))

    def sample(self, band: Any) -> xr.DataArray:
        """The pixels to take the statistics of the gradient from, as a cube of one row (the whole raster
        when it is small, else a regular grid of blocks): the percentiles and the means and standard
        deviations are the whole raster's up to the sampling."""
        h, w = self.height, self.width
        if h * w <= _SAMPLE_PIXELS:
            return _f._select(self.read(0, 0, w, h), band, None)
        b = _SAMPLE_BLOCK
        n_blocks = max(1, _SAMPLE_PIXELS // (b * b))
        side = max(1, int(np.ceil(np.sqrt(n_blocks))))
        rows = np.linspace(0, max(h - b, 0), side).astype(int)
        cols = np.linspace(0, max(w - b, 0), side).astype(int)
        parts, template = [], None
        for r0 in rows:
            for c0 in cols:
                blk = _f._select(self.read(int(c0), int(r0), int(min(w, c0 + b)), int(min(h, r0 + b))), band, None)
                template = blk if template is None else template
                v = np.asarray(blk.transpose(..., "y", "x").values)
                parts.append(v.reshape(*v.shape[:-2], 1, -1))
        values = np.concatenate(parts, axis=-1)
        t = template.transpose(..., "y", "x")
        coords = {k: c for k, c in t.coords.items() if "y" not in c.dims and "x" not in c.dims}
        return xr.DataArray(values, dims=t.dims, coords=coords)


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
def _resolve_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None or n_jobs < 1:
        return max(1, (os.cpu_count() or 2) - 1)
    return int(n_jobs)


def _check_options(opts: Dict[str, Any]) -> Dict[str, Any]:
    for name, why in (
            ("markers", "a count per image means a different density on every tile: give marker_density (markers per "
                        "million pixels, the same on every tile)"),
            ("preset", "the notebook's preset sets a marker count and a global mask: pass its values one by one"),
            ("mask", "a mask array is the size of the raster: use mask_threshold"),
            ("mask_quantile", "a percentile of the whole raster is not known tile by tile: use mask_threshold"),
            ("time", "the raster has bands, not dates: select one with band=")):
        if name in opts:
            raise ValueError(f"fields_tiled: {name}= is not supported: {why}")
    unknown = sorted(set(opts) - set(_OPTIONS))
    if unknown:
        raise TypeError(f"fields_tiled got unexpected keyword arguments {unknown}")
    o = dict(method="montecarlo", min_distance=20, marker_radius=3, n_neighbors=20, coloring="knn", high=0.6,
             combine="mean")
    o.update({k: v for k, v in opts.items() if v is not None or k in ("gradient", "final", "band")})
    o["realizations"] = _f._DEFAULTS["realizations"] if opts.get("realizations") is None else opts["realizations"]
    o["low"] = _f._DEFAULTS["low"] if opts.get("low") is None else opts["low"]
    o["min_size"] = _f._DEFAULTS["min_size"] if opts.get("min_size") is None else opts["min_size"]
    o["marker_density"] = (_f._DEFAULTS["marker_density"] if opts.get("marker_density") is None
                           else float(opts["marker_density"]))
    _f._check_stochastic(o["method"], o["coloring"])
    if o["combine"] not in ("mean", "max"):
        raise ValueError(f"combine must be 'mean' or 'max', got {o['combine']!r}")
    if o.get("final") not in (None, "image", "gradient"):
        raise ValueError(f"final must be 'image' or 'gradient', got {o['final']!r}")
    if not (0 <= o["low"] <= o["high"] <= 1):
        raise ValueError("low and high must satisfy 0 <= low <= high <= 1")
    if o["min_size"] < 0:
        raise ValueError("min_size must not be negative")
    if not np.isfinite(o["marker_density"]) or o["marker_density"] <= 0:
        raise ValueError("marker_density must be positive (markers per million pixels)")
    return o


# --------------------------------------------------------------------------------------
# One tile
# --------------------------------------------------------------------------------------
class _Tile:
    """The grid cell ``(r, c)``: its core window and the extended one (``(col0, row0, col1, row1)``)."""

    def __init__(self, r: int, c: int, tile: int, overlap: int, height: int, width: int):
        self.r, self.c = r, c
        self.core = (c * tile, r * tile, min(width, (c + 1) * tile), min(height, (r + 1) * tile))
        c0, r0, c1, r1 = self.core
        self.ext = (max(0, c0 - overlap), max(0, r0 - overlap), min(width, c1 + overlap), min(height, r1 + overlap))

    def core_in_ext(self) -> Tuple[slice, slice]:
        c0, r0, c1, r1 = self.core
        e0, f0, _, _ = self.ext
        return slice(r0 - f0, r1 - f0), slice(c0 - e0, c1 - e0)


class _Job:
    """Everything the tiles share."""

    def __init__(self, source: _Source, opts: Dict[str, Any], stats: Any, single: bool, final: str, gradient: str):
        self.source, self.o, self.stats, self.single, self.final, self.gradient = source, opts, stats, single, final, gradient

    def prepare(self, tile: _Tile):
        """``(grad float64, img or None, keep or None)`` of the tile's extended window."""
        o = self.o
        cube, _ = _f._layers(_f._select(self.source.read(*tile.ext), o.get("band"), None))
        grad = np.asarray(_f._gradient_of(cube, self.gradient, o["combine"], self.stats), dtype=np.float64)
        img = _f._image2d(cube, "the image") if self.single else None
        keep = None
        if o.get("mask_threshold") is not None:
            if img is None:
                raise ValueError("mask_threshold needs one 2-D image: select a layer with band=")
            keep = img > o["mask_threshold"]
        return grad, img, _f._flood_mask(grad, keep)


def _probability(job: _Job, tile: _Tile, prepared, seed, n_threads: int):
    """The core of the tile's boundary probability (float32) and the number of realisations (or None), from
    ``job.prepare(tile)``."""
    o = job.o
    grad, _, keep = prepared
    markers, _ = _f._resolve_markers(None, o["marker_density"], grad.shape)
    prob, _, r = _f._boundary(
        grad, keep, method=o["method"], realizations=o["realizations"], markers=markers,
        min_distance=o["min_distance"], marker_radius=o["marker_radius"], n_neighbors=o["n_neighbors"],
        coloring=o["coloring"], random_state=seed, n_jobs=n_threads)
    return prob[tile.core_in_ext()].astype(np.float32), r


def _tile_labels(job: _Job, tile: _Tile, prob: np.ndarray, realizations: Optional[int]):
    """``(labels of the extended window, boundary_label)`` from the extended window's probability."""
    o = job.o
    grad, img, keep = job.prepare(tile)
    counts = None
    if realizations is not None:  # the file holds counts / R: the integer counts are compared, as in fields
        counts = np.rint(prob.astype(np.float64) * realizations)
    return _f._labels_from_boundary(
        prob.astype(np.float64), counts, realizations, img if job.final == "image" else grad, keep,
        low=o["low"], high=o["high"], min_size=o["min_size"])


# --------------------------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------------------------
def _gid(local: np.ndarray, offset: int, network: bool) -> np.ndarray:
    """The ids of a tile's labels among all tiles: the network is 1, every other label ``offset + label``."""
    g = np.where(local > 0, local.astype(np.int64) + offset, 0)
    if network:
        g[local == 1] = 1
    return g


def _strip(local: np.ndarray, side: str, outer: bool, core: Tuple[slice, slice]) -> np.ndarray:
    """The core's edge row or column ``side`` (or the one just outside it, in the extended window)."""
    rs, cs = core
    if side == "top":
        return local[rs.start - outer, cs]
    if side == "bottom":
        return local[rs.stop - 1 + outer, cs]
    if side == "left":
        return local[rs, cs.start - outer]
    return local[rs, cs.stop - 1 + outer]


def _seam_edges(a: Dict, b: Dict, side_a: str, side_b: str) -> Tuple[np.ndarray, np.ndarray]:
    """The pairs of ids to union across the seam between tile ``a`` and its neighbour ``b`` (on ``a``'s
    ``side_a``): two facing pixels, neither of the network, are one field when either tile's extended
    labelling gives them the same label."""
    ga = _gid(a["core"][side_a], a["offset"], a["network"])
    gb = _gid(b["core"][side_b], b["offset"], b["network"])
    # a's labelling of b's facing pixels, and b's of a's: their ids in the tile's own numbering
    va = _gid(a["outer"][side_a], a["offset"], a["network"])
    vb = _gid(b["outer"][side_b], b["offset"], b["network"])
    same = ((va == ga) | (vb == gb)) & (ga > 1) & (gb > 1) & (va > 1) & (vb > 1)
    return ga[same], gb[same]


def _components(n: int, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(u) == 0:
        return np.arange(n)
    graph = coo_matrix((np.ones(len(u), dtype=np.int8), (u, v)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


# --------------------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------------------
def _profile(source: _Source, dtype: str, nodata: Any, tile: int) -> dict:
    block = 512 if tile >= 512 else 256
    return dict(driver="GTiff", dtype=dtype, count=1, width=source.width, height=source.height,
                crs=source.crs, transform=source.transform, nodata=nodata, tiled=True, blockxsize=block,
                blockysize=block, compress="deflate", BIGTIFF="IF_SAFER")


def _to_cog(tmp: str, path: str, tags: Dict[str, Any], resampling: str) -> None:
    import rasterio
    import rasterio.shutil

    if tags:
        with rasterio.open(tmp, "r+") as ds:
            ds.update_tags(**{k: str(v) for k, v in tags.items()})
    rasterio.shutil.copy(tmp, path, driver="COG", compress="deflate", overview_resampling=resampling)


def _window(w: Tuple[int, int, int, int]):
    from rasterio.windows import Window

    return Window(w[0], w[1], w[2] - w[0], w[3] - w[1])


# --------------------------------------------------------------------------------------
# The polygons
# --------------------------------------------------------------------------------------
def _polygons_of(labels_path: str, tiles: List[_Tile], grid: Tuple[int, int], boundary_label: int, source: _Source,
                 out_path: str) -> int:
    """Polygonise the labels block by block (the tiles' cores) and dissolve across the blocks by field.
    The pixel corners are whole numbers while the blocks are cut (so the pieces of a field meet exactly) and
    are put on the map after the union. Returns the number of polygons written."""
    import rasterio
    from rasterio.transform import Affine
    from shapely import affinity
    from shapely.ops import unary_union

    t = source.transform
    matrix = [t.a, t.b, t.d, t.e, t.c, t.f]
    rows, cols = grid
    by_rc = {(t_.r, t_.c): t_ for t_ in tiles}
    held: Dict[int, list] = {}
    held_px: Dict[int, int] = {}
    written = 0
    first = True

    def emit(ids, shapes, n_pixels):
        nonlocal written, first
        if len(ids) == 0:
            return
        shapes = [affinity.affine_transform(g, matrix) for g in shapes]
        gdf = _f._polygon_table(np.asarray(ids, dtype=np.int64), shapes, np.asarray(n_pixels), source.crs)
        gdf.to_file(out_path, driver="GPKG", mode="w" if first else "a")
        first = False
        written += len(gdf)

    with rasterio.open(labels_path) as ds:
        for tl in tiles:
            c0, r0, c1, r1 = tl.core
            lab = ds.read(1, window=_window(tl.core))
            keep = lab > 0
            if boundary_label:
                keep &= lab != boundary_label
            if not keep.any():
                continue
            ids, shapes = _f._polygonise(lab, keep, Affine.translation(c0, r0))
            n_px = np.bincount(lab[keep], minlength=int(ids.max()) + 1)[ids]
            edges = []
            if (tl.r - 1, tl.c) in by_rc:
                edges.append(lab[0])
            if (tl.r + 1, tl.c) in by_rc:
                edges.append(lab[-1])
            if (tl.r, tl.c - 1) in by_rc:
                edges.append(lab[:, 0])
            if (tl.r, tl.c + 1) in by_rc:
                edges.append(lab[:, -1])
            is_open = np.isin(ids, np.unique(np.concatenate(edges))) if edges else np.zeros(len(ids), dtype=bool)
            closed = ~is_open
            emit(ids[closed], [s for s, k in zip(shapes, closed) if k], n_px[closed])
            for i, s, n in zip(ids[is_open], [s for s, k in zip(shapes, is_open) if k], n_px[is_open]):
                held.setdefault(int(i), []).append(s)
                held_px[int(i)] = held_px.get(int(i), 0) + int(n)
    if held:
        keys = sorted(held)
        emit(keys, [unary_union(held[k]) if len(held[k]) > 1 else held[k][0] for k in keys],
             [held_px[k] for k in keys])
    if first:  # no field at all: an empty layer, with the columns
        gdf = _f._polygon_table(np.zeros(0, dtype=np.int64), [], np.zeros(0, dtype=np.int64), source.crs)
        gdf.to_file(out_path, driver="GPKG")
    return written


# --------------------------------------------------------------------------------------
# The function
# --------------------------------------------------------------------------------------
def fields_tiled(
    src: Any,
    out: Any,
    *,
    tile: int = 512,
    overlap: Optional[int] = None,
    polygons: bool = True,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **fields_kwargs,
) -> Dict[str, Any]:
    """``zeit.fields`` on a raster too large for one call, by overlapping tiles, written to a folder.

    The raster is cut into ``tile`` x ``tile`` cores. Pass 1 computes the boundary probability of each core
    from its window enlarged by ``overlap`` pixels on every side and keeps the core (``boundary.tif``). Pass 2
    makes the labels of each core from the probability and the image of the same enlarged window, with the code
    of ``fields`` (hysteresis, labelling, ``min_size``, the final watershed), and keeps the core. The labels of
    neighbouring tiles are then joined across the seams (see Notes), renumbered 1..k and written
    (``labels.tif``), and the polygons are made block by block and dissolved across the blocks
    (``fields.gpkg``, the columns of ``zeit.fields_to_polygons``). Only windows are in memory, never the
    raster.

    Parameters
    ----------
    src
        The raster: a path (GeoTIFF, VRT... anything GDAL reads, one band or several: with several, the
        multitemporal gradient and the ``"gradient"`` final surface are the defaults, as in ``fields``), read
        by window; or a DataArray (``(y, x)`` or ``(band, y, x)``), for tests.
    out
        Output folder (created): ``labels.tif`` (int32 COG), ``boundary.tif`` (float32 COG, the probability)
        and, with ``polygons``, ``fields.gpkg``.
    tile
        Pixels along a side of a core (512 by default). Measured on a 1.8 Mpx window with 8 threads, the cost
        per million pixels of window falls with smaller tiles (4.6, 5.3, 7.2 and 9.9 s for 256, 512, 1024 and
        2048), which outweighs the larger share of overlap: about 3.4 h per 1.2e9 pixels at 512 against 3.9 h
        at 2048 (``benchmarks/fields/tiled/``).
    overlap
        Pixels of window around the core. The default is ``max(64, 4 * min_distance)`` (80 pixels with
        ``min_distance=20``): the markers are about ``min_distance`` apart and a watershed basin reaches a few
        markers away, so the probability near a seam is the whole raster's up to the stochastic noise. For a
        field larger than half the overlap the labels of the two tiles are still joined (see Notes).
    polygons
        Also write ``fields.gpkg``.
    n_jobs
        Threads in all (``None``: all but one). Pass 1 (the Monte Carlo, nearly all the time) runs the tiles one
        after the other with every thread in the Rust call; pass 2 runs ``min(n_jobs, tiles)`` tiles at a time.
    random_state
        Seed. The tile ``(row, col)`` draws from ``SeedSequence(random_state, spawn_key=(row, col))``, so the
        result depends on neither ``n_jobs`` nor the order of the tiles (but it does on ``tile``).
    **fields_kwargs
        ``gradient``, ``method``, ``realizations``, ``marker_density``, ``min_distance``, ``marker_radius``,
        ``n_neighbors``, ``coloring``, ``low``, ``high``, ``min_size``, ``final``, ``mask_threshold``, ``band``,
        ``combine``, with the meaning and the defaults of ``zeit.fields``. ``markers`` (a count per image) is
        refused: the density ``marker_density`` is the same on every tile. So are ``mask`` (a raster-sized
        array), ``mask_quantile`` (a percentile of the whole raster), ``preset`` and ``time``.

    Returns
    -------
    dict
        ``labels``, ``boundary`` and ``polygons`` (paths; the last None without ``polygons``), ``n_fields``
        (labels other than the network), ``boundary_label`` (1 when the network is in the labels, as one id,
        1; else 0), ``tiles`` ``(rows, cols)``, ``overlap``, ``n_polygons`` and ``window_pixels`` (the pixels of all the
        extended windows: what the Monte Carlo ran on).

    Notes
    -----
    - **The gradient is the whole raster's.** The stretch percentiles of ``"tophat"`` and ``"sobel"`` and the
      mean and standard deviation of each layer of ``"multitemporal"`` are those of the whole raster (of a
      regular sample of 16 Mpx per layer when it is larger), not of each tile.
    - **The seams.** Two pixels facing each other across a seam, neither the network nor 0, are given one
      id when the extended labelling of either tile gives them the same label (the tile that sees both
      pixels and found them connected). The ids are joined through chains of such pairs, so a field of any
      size across several tiles becomes one label. The network (the label ``fields`` keeps for the pixels of
      the hysteresis) is one id, 1, in all tiles.
    - **What a tile cannot see.** The hysteresis joins a pixel of probability above ``low`` to one above ``high``
      along any path, and ``min_size`` counts the pixels of an object: both look only inside the extended
      window, so a boundary line that is connected to a strong one only beyond ``overlap`` pixels of the core,
      or an object cut by the window, can differ from the whole raster's (near the window's outer edge, outside
      the core, for the second). The tiles' labels are equal to ``fields``' as a partition up to the Monte
      Carlo's noise and these edge effects.

    Examples
    --------
    >>> res = zeit.fields_tiled("chaco_2020.vrt", "out/", random_state=0, n_jobs=8)
    >>> res["labels"], res["n_fields"]
    """
    tile = int(tile)
    if tile < 1:
        raise ValueError("tile must be at least 1")
    opts = _check_options(dict(fields_kwargs))
    source = _Source(src)
    if overlap is None:
        overlap = max(64, int(np.ceil(4 * opts["min_distance"])))
    overlap = int(overlap)
    if overlap < 1:
        raise ValueError("overlap must be at least 1 pixel")
    n_jobs = _resolve_jobs(n_jobs)

    sample = source.sample(opts.get("band"))
    cube, features = _f._layers(sample)
    single = not features
    final = opts.get("final") or ("image" if single else "gradient")
    if final == "image" and not single:
        raise ValueError("final='image' needs one 2-D image: select a layer with band= or use final='gradient'")
    gradient = opts.get("gradient") or ("sobel" if single else "multitemporal")
    if not isinstance(gradient, str):
        raise ValueError("fields_tiled: gradient must be 'tophat', 'sobel' or 'multitemporal' (a ready gradient "
                         "is the size of the raster)")
    if gradient == "multitemporal":
        stats = _f._multitemporal_stats(cube)
    else:
        stats = _f._stretch_bounds(_f._image2d(cube, f"gradient={gradient!r}").reshape(1, -1))
    job = _Job(source, opts, stats, single, final, gradient)

    rows, cols = -(-source.height // tile), -(-source.width // tile)
    tiles = [_Tile(r, c, tile, overlap, source.height, source.width) for r in range(rows) for c in range(cols)]
    workers = max(1, min(n_jobs, len(tiles)))
    base = np.random.SeedSequence(random_state)

    out_dir = os.fspath(out)
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, ".fields_tiled_tmp")
    os.makedirs(work, exist_ok=True)
    try:
        return _run(job, tiles, (rows, cols), tile, overlap, workers, n_jobs, base, out_dir, work, polygons)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run(job: _Job, tiles: List[_Tile], grid: Tuple[int, int], tile: int, overlap: int, workers: int, n_jobs: int,
         base: np.random.SeedSequence, out_dir: str, work: str, polygons: bool) -> Dict[str, Any]:
    import rasterio

    source, o = job.source, job.o
    lock = threading.Lock()
    prob_tmp = os.path.join(work, "boundary.tif")
    local_tmp = os.path.join(work, "local.tif")
    final_tmp = os.path.join(work, "labels.tif")
    reals: List[Optional[int]] = []

    # Pass 1: the probability of each core, from the extended window. The tiles go one after the other with
    # every thread in the Rust call (the realisations are what costs; a few tiles on as many workers would
    # leave threads idle at the end); the next tile's window is read and its gradient made meanwhile.
    with rasterio.open(prob_tmp, "w", **_profile(source, "float32", None, tile)) as ds,             ThreadPoolExecutor(1) as ahead:
        nxt = ahead.submit(job.prepare, tiles[0])
        for k, tl in enumerate(tiles):
            prepared = nxt.result()
            if k + 1 < len(tiles):
                nxt = ahead.submit(job.prepare, tiles[k + 1])
            seed = np.random.SeedSequence(base.entropy, spawn_key=(tl.r, tl.c))
            prob, r = _probability(job, tl, prepared, seed, n_jobs)
            del prepared
            ds.write(prob, 1, window=_window(tl.core))
            reals.append(r)
    realizations = reals[0]

    # Pass 2: the labels of each core, from the extended window of the probability and the image.
    info: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with rasterio.open(local_tmp, "w", **_profile(source, "int32", 0, tile)) as ds:
        def pass2(tl: _Tile):
            with rasterio.open(prob_tmp) as pf:
                prob = pf.read(1, window=_window(tl.ext))
            lab, network = _tile_labels(job, tl, prob, realizations)
            core = tl.core_in_ext()
            core_lab = lab[core]
            strips = {"core": {s: _strip(lab, s, False, core).copy() for s in _SIDES},
                      "outer": {}, "network": bool(network), "nmax": int(lab.max()),
                      "present": np.unique(core_lab)}
            for s in _SIDES:  # only the sides with a neighbouring tile have an outer strip
                has = {"top": tl.r > 0, "bottom": tl.r < grid[0] - 1, "left": tl.c > 0,
                       "right": tl.c < grid[1] - 1}[s]
                if has:
                    strips["outer"][s] = _strip(lab, s, True, core).copy()
            with lock:
                ds.write(core_lab, 1, window=_window(tl.core))
                info[(tl.r, tl.c)] = strips

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(pass2, tiles))

    # Offsets: tile k's labels are numbered from 1 + (the labels of the tiles before it).
    offset = 1
    for tl in tiles:
        inf = info[(tl.r, tl.c)]
        inf["offset"] = offset
        offset += inf["nmax"] + 1
    n_ids = offset + 1

    # Stitch: union the ids across the seams.
    us: List[np.ndarray] = []
    vs: List[np.ndarray] = []
    for tl in tiles:
        a = info[(tl.r, tl.c)]
        for side_a, side_b, nb in (("right", "left", (tl.r, tl.c + 1)), ("bottom", "top", (tl.r + 1, tl.c))):
            if nb in info:
                u, v = _seam_edges(a, info[nb], side_a, side_b)
                us.append(u)
                vs.append(v)
    comp = _components(n_ids, np.concatenate(us) if us else np.zeros(0, dtype=np.int64),
                       np.concatenate(vs) if vs else np.zeros(0, dtype=np.int64))

    # The used ids and their final numbers: the network 1, the others 2.. (1.. without a network) in the
    # order of their smallest provisional id.
    used = np.zeros(n_ids, dtype=bool)
    for tl in tiles:
        inf = info[(tl.r, tl.c)]
        present = inf["present"]
        used[_gid(present[present > 0], inf["offset"], inf["network"])] = True
    network_used = bool(used[1])
    used[1] = False
    ids = np.flatnonzero(used)
    k = int(comp.max()) + 1 if len(comp) else 0
    first = np.full(k, np.iinfo(np.int64).max)
    np.minimum.at(first, comp[ids], ids)
    comps = np.unique(comp[ids])
    rank = np.zeros(k, dtype=np.int64)
    rank[comps[np.argsort(first[comps])]] = np.arange(len(comps)) + (2 if network_used else 1)
    lut = np.zeros(n_ids, dtype=np.int32)
    lut[ids] = rank[comp[ids]]
    if network_used:
        lut[1] = 1
    n_fields = len(comps)

    # Rewrite the labels with the final ids.
    with rasterio.open(local_tmp) as src_ds, rasterio.open(final_tmp, "w", **_profile(source, "int32", 0, tile)) as ds:
        for tl in tiles:
            local = src_ds.read(1, window=_window(tl.core))
            inf = info[(tl.r, tl.c)]
            ds.write(lut[_gid(local, inf["offset"], inf["network"])].astype(np.int32), 1, window=_window(tl.core))

    tags = {"algorithm": "stochastic watershed", "tiled": f"{grid[0]}x{grid[1]}", "tile": tile, "overlap": overlap,
            "boundary_label": int(network_used), "method": o["method"], "gradient": job.gradient,
            "final": job.final, "low": o["low"], "high": o["high"], "min_size": o["min_size"],
            "marker_density": o["marker_density"], "min_distance": o["min_distance"]}
    if realizations is not None:
        tags["realizations"] = realizations
    labels_path = os.path.join(out_dir, "labels.tif")
    boundary_path = os.path.join(out_dir, "boundary.tif")
    _to_cog(final_tmp, labels_path, tags, "nearest")
    _to_cog(prob_tmp, boundary_path, {k_: v for k_, v in tags.items() if k_ != "boundary_label"}, "average")

    result = {"labels": labels_path, "boundary": boundary_path, "polygons": None, "n_fields": n_fields,
              "boundary_label": int(network_used), "tiles": grid, "overlap": overlap, "n_polygons": 0,
              "window_pixels": sum((t_.ext[2] - t_.ext[0]) * (t_.ext[3] - t_.ext[1]) for t_ in tiles)}
    if polygons:
        gpkg = os.path.join(out_dir, "fields.gpkg")
        if os.path.exists(gpkg):
            os.remove(gpkg)
        result["polygons"] = gpkg
        result["n_polygons"] = _polygons_of(labels_path, tiles, grid, int(network_used), source, gpkg)
    return result
