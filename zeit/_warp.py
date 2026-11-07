# SPDX-License-Identifier: GPL-3.0-or-later
"""The warp of ``load_raster(..., like=)``: a cube onto the grid of another raster.

From landschaft 1.35.0 (``_reproject.py``, by the same author), the raster part. Rasters are
warped by a port of GDAL's warp kernel (``_core.warp``, from GDAL 3.12's gdalwarpkernel.cpp)
on source coordinates that do not depend on how the work is split: the source coordinates of
the destination's cells come from a lattice anchored at its origin, exact (PROJ) every 16
cells, refined down to every cell where interpolating them would be off by more than the
tolerance, interpolated bilinearly in between; the kernel scale is GDAL's for the whole
destination in one chunk. In memory, lazily or from a file, with any number of threads,
every cell gets the same bits. tolerance=0 transforms every cell exactly: GDAL's values for
the whole destination at once.

What zeit adds to landschaft's: every plane of a cube (every date, every band) is warped
with a validity mask of its own (GDAL's unified mask would let a cloud of one date into the
interpolation of the others'), which is GDAL warping each date alone; the coordinates are
still computed once for all of them. A lazy cube is split in tasks along its dates as well
as its rows and columns. A grid that only crops or pads the source (the same CRS and cells,
a whole number of cells apart) is taken by indexing, without the kernel. QA bands are
always taken from the nearest cell.
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple, Optional, Tuple

import numpy as np
import xarray as xr
from rasterio.crs import CRS

# GDAL's GDALResampleAlg numbers of the methods the kernel runs; the area methods take
# the source cells a destination cell covers (from its corners), the others sample at
# cell centres. GDAL's filter radius of the interpolating ones.
_METHODS = {"nearest": 0, "bilinear": 1, "cubic": 2, "cubic_spline": 3, "lanczos": 4, "average": 5, "mode": 6,
            "max": 8, "min": 9, "med": 10, "q1": 11, "q3": 12, "rms": 14}
RESAMPLING_NAMES = tuple(_METHODS)
_AREA_METHODS = {"average", "mode", "max", "min", "med", "q1", "q3", "rms"}
_FILTER_RADIUS = {"bilinear": 1, "cubic": 2, "cubic_spline": 2, "lanczos": 3}
_AFFINE, _LATTICE, _DENSE, _EXACT = 0, 1, 2, 3  # WarpCoords::Kind
_CELL = 16  # samples a side of a lattice cell (kWarpCell)
_LEVELS = 4  # refinements of a lattice cell: nodes every 16, 8, 4, 2 or 1 samples
_DENSE_POINTS = 1 << 22  # exact coordinates computed at once with tolerance=0
_STEP_COUNT = 21  # GDAL's DEFAULT_STEP_COUNT (sample points along each edge)
_BLOCK = 1024  # cells a side of the tasks of a lazy result (a multiple of it)

#: Band names of quality/mask bands, always taken from the nearest cell (their values are
#: bit flags or classes, never to be interpolated).
QA_BANDS = {"qa", "qa_pixel", "pixel_qa", "qa_radsat", "bqa", "fmask", "scl", "qa60", "msk_cldprb", "mask",
            "cloud_mask", "state_1km", "sur_refl_state_500m", "summaryqa", "detailedqa"}


class Grid(NamedTuple):
    """A target grid: affine transform, (height, width), CRS and the x, y cell-centre
    coordinates the result takes."""
    transform: Any
    shape: Tuple[int, int]
    crs: CRS
    x: np.ndarray
    y: np.ndarray


def resolve_threads(n_threads: Optional[int]) -> int:
    """``None``, 0 or negative: as ``n_jobs=-1`` (see zeit._threads)."""
    from ._threads import resolve

    return resolve(n_threads)


def transform_of(da: xr.DataArray):
    """The affine transform of a DataArray, from its coordinates: rioxarray's cached
    transform keeps the old resolution after a strided slice such as ``da[::2, ::2]``."""
    cached = da.rio.transform()
    if da.sizes.get("x", 0) > 1 and da.sizes.get("y", 0) > 1:
        fresh = da.rio.transform(recalc=True)
        tol = 1e-6 * max(abs(fresh.a), abs(fresh.e))
        if any(abs(p - q) > tol for p, q in zip(cached[:6], fresh[:6])):
            return fresh  # stale: keep the cached one otherwise, it is exact
    return cached


def _nodata(data: xr.DataArray):
    """(source NoData, destination NoData, the type the warp works in)."""
    dtype = data.dtype.newbyteorder("=")  # the kernel reads the machine's byte order
    nodata = data.rio.nodata
    if dtype == np.bool_:
        dtype = np.dtype(np.uint8)
    elif dtype == np.float16:
        dtype = np.dtype(np.float32)
    if np.issubdtype(dtype, np.floating):
        value = np.nan if nodata is None or np.isnan(nodata) else float(nodata)
        return value, value, dtype
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f"load_raster(like=): cannot warp values of type {dtype}")
    if nodata is not None and not np.isnan(nodata):
        return nodata, nodata, dtype
    info = np.iinfo(dtype)
    return None, (info.max if info.min == 0 else info.min), dtype


class _Plan(NamedTuple):
    """What a warp does with values and NoData: the NoData of the source for the kernel (None: no validity
    mask), the kernel's destination NoData, the value of the cells no source cell covers, the type the kernel
    works in, the type of the result and its NoData (None: none)."""
    src_nodata: Any
    kernel_nodata: Optional[float]
    fill: float
    work: np.dtype
    dtype: np.dtype
    nodata: Any

    @property
    def sentinel(self) -> bool:
        """No destination NoData (``nodata="source"`` of a raster without one): the cells outside are 0. The
        kernel moves a value equal to its destination NoData (GDAL's AvoidNoData), which GDAL does only with
        one, so it is given a value that no cell of the window it warps can take (see ``_Warp.free``) and the
        cells that hold it are set to ``fill`` after."""
        return self.nodata is None


def _default_nodata(dtype: np.dtype) -> float:
    if np.issubdtype(dtype, np.floating):
        return np.nan
    info = np.iinfo(dtype)
    return info.max if info.min == 0 else info.min


def _fits(value: float, dtype: np.dtype) -> bool:
    if np.issubdtype(dtype, np.floating):
        return True
    if np.isnan(value) or value != int(value):
        return False
    info = np.iinfo(dtype)
    return info.min <= value <= info.max


def _plan(data: xr.DataArray, nodata: Any, dtype: Any) -> _Plan:
    """The plan of a warp of ``data`` with ``load_raster``'s ``nodata=`` and ``dtype=``.

    ``nodata=None``: the NoData of the source, and for a raster without one NaN (floats) or the type's
    extreme (integers), written as the NoData of the result. ``"source"``: gdalwarp without ``-dstnodata``:
    the source's NoData if it has one, else none, the cells outside set to 0. A number: ``-dstnodata``. The
    kernel works in the union of the source's type and the result's (GDAL's working type)."""
    src, dst, work = _nodata(data)
    out = work if dtype is None else np.dtype(dtype).newbyteorder("=")
    if out == np.bool_:
        out = np.dtype(np.uint8)
    if not (np.issubdtype(out, np.integer) or np.issubdtype(out, np.floating)):
        raise TypeError(f"dtype= takes a number type, got {out}")
    if nodata is None:
        if dtype is None:
            return _Plan(src, dst, dst, work, work, dst)
        # the type changes: the source's NoData if it has one, else the new type's default
        raw = data.rio.nodata
        value = float(raw) if raw is not None and not (np.isnan(raw) and not np.issubdtype(out, np.floating)) \
            else _default_nodata(out)
        src_nd = src
    else:
        raw = data.rio.nodata
        src_nd = None if raw is None else float(raw)
        if isinstance(nodata, str):
            if nodata != "source":
                raise ValueError(f"nodata must be None, 'source' or a number, got {nodata!r}")
            value = src_nd
        else:
            value = float(nodata)
    if value is not None and not _fits(value, out):
        raise ValueError(f"the NoData {value!r} does not fit the type {out}")
    kind = work if dtype is None else np.promote_types(work, out)
    return _Plan(src_nd, value, 0.0 if value is None else value, kind, out, value)


def _free_values(values, count: int, banned=()):
    """``count`` integers the type of ``values`` holds that no value of it is, past its range (None: no room)."""
    info = np.iinfo(values.dtype)
    lo, hi = int(values.min()), int(values.max())
    found = []
    for v in range(hi + 1, min(hi + 1 + count + len(banned), info.max + 1)):
        if v not in banned and len(found) < count:
            found.append(v)
    for v in range(lo - 1, max(lo - 1 - count - len(banned), info.min - 1), -1):
        if v not in banned and len(found) < count:
            found.append(v)
    return found if len(found) == count else None


def _where(inside, out, fill):
    if hasattr(out, "dask"):
        import dask.array as dsa

        return dsa.where(inside, out, fill)
    return np.where(inside, out, fill)


def _round_back(values, dtype):
    """Floats the kernel made of integers, as the kernel rounds an integer result (ClampRoundAndAvoidNoData)."""
    info = np.iinfo(dtype)
    return np.clip(np.floor(values + 0.5), info.min, info.max).astype(dtype)


def _convert(values, dtype):
    """``values`` in ``dtype`` as GDAL writes a working type into a band: floats to integers rounded half away
    from zero and clamped (NaN is 0), integers clamped."""
    dtype = np.dtype(dtype)
    if values.dtype == dtype:
        return values
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if np.issubdtype(values.dtype, np.floating):
            v = values.astype(np.float64)
            v = np.where(np.isnan(v), 0.0, np.where(v < 0, np.ceil(v - 0.5), np.floor(v + 0.5)))
            return np.clip(v, info.min, info.max).astype(dtype)
        return np.clip(values, info.min, info.max).astype(dtype)
    return values.astype(dtype)


def _converted(values, dtype):
    dtype = np.dtype(dtype)
    if values.dtype == dtype:
        return values
    return values.map_blocks(_convert, dtype, dtype=dtype) if hasattr(values, "dask") else _convert(values, dtype)


# --- a cutline ---------------------------------------------------------------------


class Cutline(NamedTuple):
    """The polygons that cut a result (gdalwarp ``-cutline``): shapely geometries and their CRS (None: the
    destination's)."""
    geometries: tuple
    crs: Any

    def in_crs(self, crs: CRS) -> list:
        """The geometries in ``crs`` (their vertices transformed, as OGR does)."""
        if self.crs is None or CRS.from_user_input(self.crs.to_wkt()) == crs:
            return list(self.geometries)
        import shapely
        from pyproj import Transformer

        tr = Transformer.from_crs(self.crs.to_wkt(), crs.to_wkt(), always_xy=True)
        return [shapely.transform(g, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
                for g in self.geometries]

    def bounds(self, crs: CRS) -> Tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of the geometries in ``crs``."""
        boxes = np.array([g.bounds for g in self.in_crs(crs)])
        return float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())

    def mask(self, grid: Grid) -> np.ndarray:
        """True for the cells of ``grid`` whose centre is inside the polygons (GDAL's rasterization)."""
        from rasterio.features import rasterize

        burned = rasterize([(g, 1) for g in self.in_crs(grid.crs)], out_shape=tuple(grid.shape),
                           transform=grid.transform, fill=0, all_touched=False, dtype="uint8")
        return burned.astype(bool)


def cutline_of(spec: Any) -> Cutline:
    """A ``Cutline`` from what ``load_raster(cutline=)`` takes: a vector file (``path``, or ``(path, layer)``), a
    GeoDataFrame or GeoSeries, a shapely geometry or a list of them, or GeoJSON-like objects."""
    import os

    layer = None
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], (str, os.PathLike)):
        spec, layer = spec
    crs = None
    if isinstance(spec, (str, os.PathLike)):
        import geopandas as gpd

        spec = gpd.read_file(spec, layer=layer)
    if hasattr(spec, "geometry") and hasattr(spec, "crs"):  # GeoDataFrame, GeoSeries
        crs = spec.crs
        geoms = list(spec.geometry)
    else:
        from shapely.geometry import shape

        items = list(spec) if isinstance(spec, (list, tuple)) else [spec]
        geoms = []
        for item in items:
            if isinstance(item, dict) and item.get("type") == "FeatureCollection":
                geoms += [shape(f["geometry"]) for f in item["features"]]
            elif isinstance(item, dict) and item.get("type") == "Feature":
                geoms.append(shape(item["geometry"]))
            elif isinstance(item, dict):
                geoms.append(shape(item))
            else:
                geoms.append(item)
    geoms = tuple(g for g in geoms if g is not None and not g.is_empty)
    if not geoms:
        raise ValueError("cutline=: no geometry in it")
    return Cutline(geoms, None if crs is None else (crs if hasattr(crs, "to_wkt") else CRS.from_user_input(crs)))


# --- source coordinates of destination points -------------------------------------


def _geotransform(t) -> tuple:
    return (t.c, t.a, t.b, t.f, t.d, t.e)


def _inv_geotransform(gt) -> tuple:
    """GDALInvGeoTransform of a grid without rotation."""
    return (-gt[0] / gt[1], 1.0 / gt[1], 0.0, -gt[3] / gt[5], 0.0, 1.0 / gt[5])


def _center_long(crs: CRS, t, shape):
    """GDAL's CENTER_LONG of a geographic source (InsertCenterLong): the middle of its
    longitudes, as "%g" writes it; the longitudes transformed to it wrap around it."""
    from pyproj import CRS as ProjCRS

    pc = ProjCRS.from_user_input(crs.to_wkt())
    if not pc.is_geographic or abs(pc.axis_info[0].unit_conversion_factor - math.pi / 180) > 1e-9:
        return None
    gt = _geotransform(t)
    h, w = shape
    corners = (gt[0] + 0 * gt[1] + 0 * gt[2], gt[0] + w * gt[1] + 0 * gt[2], gt[0] + 0 * gt[1] + h * gt[2],
               gt[0] + w * gt[1] + h * gt[2])
    lo, hi = min(corners), max(corners)
    if hi - lo > 360.0:
        return None
    return float("%g" % ((hi + lo) / 2.0))


def _is_geographic(crs: CRS) -> bool:
    from pyproj import CRS as ProjCRS

    return ProjCRS.from_user_input(crs.to_wkt()).is_geographic


class _Points:
    """The source pixel coordinates (column, row) of destination points given in destination
    pixel coordinates, by GDAL's own transformer (GDALGenImgProjTransform, from the GDAL
    rasterio loads, one per thread) or, without it, as GDAL computes them with pyproj's
    PROJ: destination geotransform, reprojection (longitudes wrapped around a geographic
    source's centre), inverse source geotransform. Failed points are infinite. ``same``:
    the same CRS, which GDAL does not reproject (but for a geographic one)."""

    def __init__(self, src_t, src_crs: CRS, src_shape, dst_t, dst_crs: CRS, dst_shape):
        from ._gdaltransform import GdalTransformer, open_library

        self._args = (src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape)
        self.gt = _geotransform(dst_t)
        self.src_gt = _geotransform(src_t)
        self.inv = _inv_geotransform(self.src_gt)
        self.src_crs, self.dst_crs = src_crs, dst_crs
        self.same = src_crs == dst_crs and not _is_geographic(src_crs)
        self.center = None if self.same else _center_long(src_crs, src_t, src_shape)
        self._local = threading.local()
        lib = None if self.same else open_library()
        self.gdal = GdalTransformer(lib, src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape) if lib else None

    def __getstate__(self):  # (dask's processes) GDAL's transformer is made again where unpickled
        return self._args

    def __setstate__(self, args):
        self.__init__(*args)

    def _transformer(self, forward: bool = False):
        name = "forward" if forward else "transformer"
        tr = getattr(self._local, name, None)
        if tr is None:
            from pyproj import Transformer

            a, b = (self.src_crs, self.dst_crs) if forward else (self.dst_crs, self.src_crs)
            tr = Transformer.from_crs(a.to_wkt(), b.to_wkt(), always_xy=True)
            setattr(self._local, name, tr)
        return tr

    def forward(self, x, y):
        """Source pixel coordinates to the destination's (GDAL's transformer the other way)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if self.gdal is not None:
            return self.gdal(x, y, dst_to_src=False)
        src_gt = self.src_gt
        dinv = _inv_geotransform(self.gt)
        gx = src_gt[0] + x * src_gt[1] + y * src_gt[2]
        gy = src_gt[3] + x * src_gt[4] + y * src_gt[5]
        if not self.same:
            gx, gy = self._transformer(forward=True).transform(gx, gy, errcheck=False)
            gx, gy = np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)
        return dinv[0] + gx * dinv[1] + gy * dinv[2], dinv[3] + gx * dinv[4] + gy * dinv[5]

    def special_points(self) -> list:
        """The poles (longitude 0, latitude +-89.9999) in destination pixel coordinates, as
        GDALTransformLonLatToDestGenImgProjTransformer gives them, with its bug fixed: from a
        geographic source GDAL (3.12, still in its master) skips the reprojection to the
        destination, and so misses a pole in the destination."""
        if self.same:
            return []
        from pyproj import CRS as ProjCRS
        from pyproj import Transformer

        src = ProjCRS.from_user_input(self.src_crs.to_wkt())
        out = []
        for lat in (-89.9999, 89.9999):
            gx, gy = 0.0, lat
            if not src.is_geographic:
                try:
                    tr = Transformer.from_crs(src.geodetic_crs, src, always_xy=True)
                    gx, gy = tr.transform(gx, gy, errcheck=True)
                except Exception:  # noqa: BLE001, S112 - outside the projection's domain
                    continue
            px = self.inv[0] + gx * self.inv[1] + gy * self.inv[2]
            py = self.inv[3] + gx * self.inv[4] + gy * self.inv[5]
            fx, fy = self.forward([px], [py])
            if np.isfinite(fx[0]) and np.isfinite(fy[0]):
                out.append((float(fx[0]), float(fy[0])))
        return out

    def handles(self, n: int):
        """(address of GDAL's transformer, n transformers of their own as an array) for the kernel,
        or (0, empty) without GDAL's; give them back with release()."""
        if self.gdal is None:
            return 0, np.empty(0, dtype=np.uintp)
        return self.gdal.address, np.array(self.gdal.take(n), dtype=np.uintp)

    def release(self, args) -> None:
        if self.gdal is not None and len(args):
            self.gdal.release([int(a) for a in args])

    def __call__(self, u, v):
        if self.gdal is not None:
            return self.gdal(u, v)
        gt, inv = self.gt, self.inv
        gx = gt[0] + u * gt[1] + v * gt[2]
        gy = gt[3] + u * gt[4] + v * gt[5]
        if not self.same:
            gx, gy = self._transformer().transform(gx, gy, errcheck=False)
            gx = np.asarray(gx, dtype=np.float64)
            gy = np.asarray(gy, dtype=np.float64)
            if self.center is not None:  # OGRProjCT's target wrap (x != HUGE_VAL and y != HUGE_VAL)
                ok = (gx != np.inf) & (gy != np.inf)
                low = ok & (gx < self.center - 180.0)
                high = ok & ~low & (gx > self.center + 180)
                gx = np.where(low, gx + 360.0, np.where(high, gx - 360.0, gx))
        return inv[0] + gx * inv[1] + gy * inv[2], inv[3] + gx * inv[4] + gy * inv[5]

    def many(self, u, v, n_threads: int, out=None):
        """__call__ on large flat arrays, in parallel (the transformations release the GIL);
        into out ((n, 2), x and y) when given."""
        n = u.size
        k = max(1, min(n_threads, n // 65536))
        if out is None:
            if k == 1:
                return self(u, v)
            out = np.empty((n, 2))
            split = True
        else:
            split = False
        cut = np.linspace(0, n, k + 1).astype(np.int64)

        def run(i):
            a, b = cut[i], cut[i + 1]
            out[a:b, 0], out[a:b, 1] = self(u[a:b], v[a:b])

        if k == 1:
            run(0)
        else:
            with ThreadPoolExecutor(k) as pool:
                list(pool.map(run, range(k)))
        return (out[:, 0].copy(), out[:, 1].copy()) if split else out

    def grid(self, cols, rows, offset: float, n_threads: int):
        """Coordinates of the samples (cols x rows, sample indices): (len(rows), len(cols), 2)."""
        u = np.broadcast_to(np.asarray(cols, dtype=np.float64)[None, :] + offset, (len(rows), len(cols)))
        v = np.broadcast_to(np.asarray(rows, dtype=np.float64)[:, None] + offset, (len(rows), len(cols)))
        out = np.empty((len(rows) * len(cols), 2))
        self.many(np.ascontiguousarray(u).ravel(), np.ascontiguousarray(v).ravel(), n_threads, out)
        return out.reshape(len(rows), len(cols), 2)


def _translation(points: _Points, src_t, dst_t) -> bool:
    """GDALTransformIsTranslationOnPixelBoundaries: the same CRS (not geographic, which GDAL
    reprojects anyway), cell size and a whole number of cells apart; GDAL then takes the
    nearest cell whatever the method."""
    if not points.same:
        return False
    s, d = _geotransform(src_t), _geotransform(dst_t)
    if not (s[1] == d[1] and s[5] == d[5] and s[2] == d[2] and s[4] == d[4]):
        return False
    inv = points.inv
    ox = inv[0] + d[0] * inv[1] + d[3] * inv[2]
    oy = inv[3] + d[0] * inv[4] + d[3] * inv[5]
    return abs(ox - round(ox)) <= 1e-6 and abs(oy - round(oy)) <= 1e-6


def _kernel_scales(points: _Points, src_shape, dst_shape, method: str, translation: bool) -> tuple:
    """GDAL's dfXScale, dfYScale for the whole destination in one chunk: the source window of
    GDALWarpOperation::ComputeSourceWindow (points along the edges; a grid when some fail or
    a pole is in the destination, widened by the source points that land in it), then
    GDALWarpKernel::PerformWarp's rules."""
    h, w = src_shape
    dh, dw = dst_shape
    step = 1.0 / (_STEP_COUNT - 1)

    def transform(u, v):
        x, y = points(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        return x, y

    def edges(every_cell):
        u, v = [], []
        if every_cell:
            for ix in range(dw + 1):
                u += [ix, ix]
                v += [0, dh]
            for iy in range(1, dh):
                u += [0, dw]
                v += [iy, iy]
        else:
            ratio = 0.0
            while ratio <= 1.0 + step * 0.5:
                u += [ratio * dw, ratio * dw, 0, dw]
                v += [0, dh, ratio * dh, ratio * dh]
                ratio += step
        return u, v

    def ratios(n):
        return [0.5 / n if i == 0 else ((i - 1) * step if i <= _STEP_COUNT else 1 - 0.5 / n)
                for i in range(_STEP_COUNT + 2)]

    def grid():
        u, v = [], []
        for ry in ratios(dw):  # (GDAL takes the x size for both)
            for rx in ratios(dw):
                u.append(rx * dw)
                v.append(ry * dh)
        return u, v

    def sample(u, v):
        x, y = transform(u, v)
        ok = np.isfinite(x) & np.isfinite(y)
        return x, y, ok

    cx, cy = transform([0, dw, 0, dw], [0, 0, dh, dh])
    every = not (np.isfinite(cx).all() and np.isfinite(cy).all())
    x, y, ok = sample(*edges(every))
    use_grid = False
    if any(0 <= px <= dw and 0 <= py <= dh for px, py in points.special_points()):
        use_grid = True
        x, y, ok = sample(*grid())
    if not use_grid and not ok.all():
        use_grid = True
        x, y, ok = sample(*grid())
    failed = int((~ok).sum())
    if failed > x.size - 5:
        return 1.0, 1.0
    min_x, max_x, min_y, max_y = x[ok].min(), x[ok].max(), y[ok].min(), y[ok].max()
    if use_grid:  # ComputeSourceWindowStartingFromSource
        sx = np.array([rx * w for _ in ratios(h) for rx in ratios(w)])
        sy = np.array([ry * h for ry in ratios(h) for _ in ratios(w)])
        fx, fy = points.forward(sx, sy)
        inside = np.isfinite(fx) & np.isfinite(fy) & (fx >= 0) & (fx <= dw) & (fy >= 0) & (fy <= dh)
        if inside.any():
            min_x, max_x = min(min_x, sx[inside].min()), max(max_x, sx[inside].max())
            min_y, max_y = min(min_y, sy[inside].min()), max(max_y, sy[inside].max())
    if min_x > w or max_x < 0 or min_y > h or max_y < 0:
        return 1.0, 1.0

    def snap(v):
        r = float(np.round(v))
        return r if abs(r - v) < 1e-6 else float(v)

    min_x, max_x, min_y, max_y = snap(min_x), snap(max_x), snap(min_y), snap(max_y)
    res = 0 if translation else _FILTER_RADIUS.get(method, 0)

    def side(lo, hi, n_dst, n_src):
        scale0 = max(1e-3, n_dst / (hi - lo))
        radius = math.ceil(res / scale0) if scale0 < 0.95 else res
        if failed > 0:
            radius += 10
        lo_c = int(max(0.0, lo))
        hi_c = int(min(math.ceil(hi), float(n_src)))
        raw = max(0.0, min(float(n_src - lo_c), hi - lo))
        if hi_c - lo_c > 0.9 * n_src:
            size = n_src
        else:
            off = max(0, min(lo_c - radius, n_src))
            size = max(0, min(n_src - off, hi_c - off + radius))
        extra = size - raw
        # PerformWarp
        scale = n_dst / (size - extra)
        if size >= n_dst and size <= n_dst + extra:
            scale = 1.0
        if scale < 1.0:
            reciprocal = 1.0 / scale
            whole = int(reciprocal + 0.5)
            if abs(reciprocal - whole) < 0.05:
                scale = 1.0 / whole
        return scale

    xscale = side(min_x, max_x, dw, w)
    yscale = side(min_y, max_y, dh, h)
    if yscale / xscale > 100:
        # A wide source wrapped around the antimeridian: the mean scale at a few points.
        nx, ny = min(10, dw), min(10, dh)
        u, v = [], []
        for iy in range(ny):
            for ix in range(nx):
                fx = 0.0 if nx == 1 else float(ix) * dw / (nx - 1)
                fy = 0.0 if ny == 1 else float(iy) * dh / (ny - 1)
                u += [fx, fx - 1 if ix == nx - 1 else fx + 1, fx]
                v += [fy, fy, fy - 1 if iy == ny - 1 else fy + 1]
        px, py = transform(u, v)
        scales = []
        for i in range(0, px.size, 3):
            if all(np.isfinite(px[i:i + 3])) and all(np.isfinite(py[i:i + 3])):
                scales.append(1.0 / max(abs(px[i + 1] - px[i]), abs(px[i + 2] - px[i])))
        scales.sort()
        if scales:
            top = scales[-1]
            kept = [s for s in scales if s > top / 10]
            total = 0.0
            for s in kept:
                total += s
            xscale = total / len(kept)
    return float(xscale), float(yscale)


class _Lattice:
    """The source coordinates of the destination samples at the nodes of a lattice anchored
    at the destination's origin: cells of 16 x 16 samples with nodes every 16, 8, 4, 2 or 1
    samples, the coarsest whose interpolation is off by at most ``tolerance`` source cells
    (|dx| + |dy|, as GDAL's approximate transformer measures it) at the midpoints checked
    (``_core.warp.lattice_checks``); cells with points the transformation fails on go down
    to every sample, cells well outside the source stay coarse. Samples (i, j) are the
    destination points (i + offset, j + offset). Every cell depends on itself only: the
    lattice of some of the cells (rows, cols: [first, end) cell ranges) has their values in
    the lattice of all."""

    def __init__(self, points: _Points, rows, cols, offset: float, tolerance: float, src_shape, n_threads: int):
        from . import _core

        checks = _core.warp.lattice_checks
        h, w = src_shape
        (cy0, cy1), (cx0, cx1) = rows, cols
        cx, cy = cx1 - cx0, cy1 - cy0
        self.cells_x, self.cells_y, self.cell_x0, self.cell_y0 = cx, cy, cx0, cy0
        half = _CELL // 2
        g = points.grid((2 * cx0 + np.arange(2 * cx + 1)) * half, (2 * cy0 + np.arange(2 * cy + 1)) * half, offset,
                        n_threads)
        err, bad, outside, box = checks(g[None], h, w, n_threads)
        need = (((err[0] > tolerance) | bad[0]) & ~outside[0]).ravel()
        self.grid0 = np.ascontiguousarray(g[::2, ::2])
        level = np.zeros(cy * cx, dtype=np.int8)
        bbox = box[0].reshape(cy * cx, 4).copy()  # xmin, xmax, ymin, ymax of the finite nodes of every cell
        node_off = np.zeros(cy * cx, dtype=np.int64)
        blocks = []
        used = 0
        idx = np.flatnonzero(need)
        for lev in range(1, _LEVELS + 1):
            if idx.size == 0:
                break
            step = _CELL >> lev
            m = _CELL // step
            k = np.arange(2 * m + 1) * (step // 2) if step > 1 else np.arange(m + 1)
            ccx, ccy = idx % cx + cx0, idx // cx + cy0
            u = np.broadcast_to(ccx[:, None, None] * _CELL + k[None, None, :], (idx.size, k.size, k.size))
            v = np.broadcast_to(ccy[:, None, None] * _CELL + k[None, :, None], (idx.size, k.size, k.size))
            gl = np.empty((idx.size, k.size, k.size, 2))
            points.many(u.ravel() + offset, v.ravel() + offset, n_threads, gl.reshape(-1, 2))
            if step > 1:
                e, b, o, bx = checks(gl, h, w, n_threads)
                more = (((e > tolerance) | b) & ~o).any(axis=(1, 2))
                nodes = gl[:, ::2, ::2]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    cell_box = np.stack([np.nanmin(bx[..., 0], axis=(1, 2)), np.nanmax(bx[..., 1], axis=(1, 2)),
                                         np.nanmin(bx[..., 2], axis=(1, 2)), np.nanmax(bx[..., 3], axis=(1, 2))], -1)
            else:
                more = np.zeros(idx.size, dtype=bool)
                nodes = gl
                cell_box = _finite_box(gl.reshape(idx.size, -1, 2))
            done = idx[~more]
            if done.size:
                flat = np.ascontiguousarray(nodes[~more]).reshape(done.size, -1, 2)
                level[done] = lev
                bbox[done] = cell_box[~more]
                node_off[done] = used + np.arange(done.size) * flat.shape[1]
                used += flat.size // 2
                blocks.append(flat.reshape(-1, 2))
            idx = idx[more]
        self.level, self.node_off = level, node_off
        self.nodes = np.concatenate(blocks) if blocks else np.empty((0, 2))
        self.bbox = bbox.reshape(cy, cx, 4)

    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the samples of the lattice's cells; None if the
        transformation fails on all of them."""
        b = self.bbox
        if not np.isfinite(b).any():
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanmin(b[..., 0]), np.nanmax(b[..., 1]), np.nanmin(b[..., 2]), np.nanmax(b[..., 3]))


def _finite_box(flat):
    """xmin, xmax, ymin, ymax of the finite points of every row of flat (k, n, 2); NaN for none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fin = np.isfinite(flat).all(-1)
        xs = np.where(fin, flat[..., 0], np.nan)
        ys = np.where(fin, flat[..., 1], np.nan)
        return np.stack([np.nanmin(xs, axis=1), np.nanmax(xs, axis=1), np.nanmin(ys, axis=1),
                         np.nanmax(ys, axis=1)], -1)


class _Warp:
    """A warp by the kernel: the whole destination in one call (in memory), or tasks of a
    lazy result reading the source window they see. The source coordinates of every
    destination sample are fixed beforehand (lattice, exact or affine), so any split gives
    the same bits. Every plane (date, band) has a validity mask of its own."""

    def __init__(self, src_shape, src_t, src_crs, dst_t, dst_shape, dst_crs, method, src_nodata, dst_nodata, dtype,
                 tolerance, n_threads, fill=None, keep=None):
        """``dst_nodata`` None: no destination NoData, the cells outside are ``fill``. ``keep``: (rows, columns)
        of the source, True for the cells that take part (a cutline); the others are left out as NoData is."""
        self.src_shape, self.shape = src_shape, dst_shape
        self.src_nodata, self.dst_nodata, self.dtype = src_nodata, dst_nodata, dtype
        self.fill = dst_nodata if fill is None else fill
        self.keep = keep
        self.tolerance = tolerance
        self.n_threads = n_threads
        self.points = _Points(src_t, src_crs, src_shape, dst_t, dst_crs, dst_shape)
        translation = _translation(self.points, src_t, dst_t)
        if translation:
            method = "nearest"
        self.method = method
        self.area = method in _AREA_METHODS
        self.offset = 0.0 if self.area else 0.5
        self.extra = 1 if self.area else 0
        self.scales = _kernel_scales(self.points, src_shape, dst_shape, method, translation)
        if self.points.same:
            self.kind = _AFFINE
        elif tolerance == 0:
            self.kind = _EXACT if self.points.gdal is not None else _DENSE
        else:
            self.kind = _LATTICE
        radius = _FILTER_RADIUS.get(method, 0)
        xs, ys = self.scales
        rx = math.ceil(radius / xs) if xs < 1.0 else radius
        ry = math.ceil(radius / ys) if ys < 1.0 else radius
        margin = 2 if self.kind in (_DENSE, _EXACT) else 0  # exact points stray from the lattice's
        self.pad = (ry + 2 + margin, rx + 2 + margin)

    def lattice(self, r0, c0, rh, cw, n_threads: int) -> _Lattice:
        """The lattice of the cells holding the samples of a destination region (built with
        the default tolerance for the windows of exact coordinates)."""
        r1, c1 = r0 + rh + self.extra, c0 + cw + self.extra  # samples, exclusive
        tol = self.tolerance if self.kind == _LATTICE else 0.125
        return _Lattice(self.points, (r0 // _CELL, (r1 - 1) // _CELL + 1), (c0 // _CELL, (c1 - 1) // _CELL + 1),
                        self.offset, tol, self.src_shape, n_threads)

    def window(self, r0, c0, rh, cw, n_threads: int):
        """The source cells (row0, row1, col0, col1) a destination region needs; None if none."""
        r1, c1 = r0 + rh + self.extra, c0 + cw + self.extra
        if self.kind == _AFFINE:
            x, y = self.points(np.array([c0, c1 - 1, c0, c1 - 1], dtype=np.float64) + self.offset,
                               np.array([r0, r0, r1 - 1, r1 - 1], dtype=np.float64) + self.offset)
            box = (x.min(), x.max(), y.min(), y.max())
        else:
            box = self.lattice(r0, c0, rh, cw, n_threads).bounds()
            if box is None:
                return None
        h, w = self.src_shape
        pr, pc = self.pad
        a, b = max(0, math.floor(box[2]) - pr), min(h, math.ceil(box[3]) + pr)
        c, d = max(0, math.floor(box[0]) - pc), min(w, math.ceil(box[1]) + pc)
        if b <= a or d <= c:
            return None
        return a, b, c, d

    def _coords_args(self, coords, handles) -> tuple:
        """The coordinate arguments of _core.warp.warp and _core.warp.coords."""
        lat = coords if isinstance(coords, _Lattice) else None
        dense = coords if isinstance(coords, tuple) else None
        address, args = handles
        return (self.kind, self.offset, self.points.gt, self.points.inv,
                lat.level if lat is not None else np.empty(0, np.int8),
                lat.grid0 if lat is not None else np.empty((0, 2)),
                lat.node_off if lat is not None else np.empty(0, np.int64),
                lat.nodes if lat is not None else np.empty((0, 2)),
                (lat.cells_x, lat.cell_x0, lat.cell_y0) if lat is not None else (1, 0, 0),
                dense[0] if dense is not None else np.empty((0, 0, 2)), dense[1] if dense is not None else 0,
                dense[2] if dense is not None else 0, address, args)

    def _call(self, src, origin, r0, c0, rh, cw, coords, handles, n_threads: int):
        """The kernel on a window of the source. Cells left out by a cutline, and cells outside when there is no
        destination NoData, are marked with values the window does not hold (+-inf for floats, for integers
        values past its range; in a window with no room for them, warped as floats and rounded back)."""
        h, w = self.src_shape
        src_nd, dst_nd = self.src_nodata, self.dst_nodata
        keep = None
        if self.keep is not None:
            keep = self.keep[origin[0]:origin[0] + src.shape[1], origin[1]:origin[1] + src.shape[2]]
            if keep.all():
                keep = None
        back = None  # the type to give the result back in, after a warp in floats
        marker = src_nd
        needs_marker = keep is not None and src_nd is None
        if needs_marker or dst_nd is None:
            if np.issubdtype(src.dtype, np.floating):
                marker = -np.inf if needs_marker else src_nd
                dst_nd = np.inf if dst_nd is None else dst_nd
            else:
                need = int(needs_marker) + int(dst_nd is None)
                free = _free_values(src, need, () if src_nd is None else (int(src_nd),))
                if free is None:
                    back, src = src.dtype, src.astype(np.float64)
                    free = [-np.inf, np.inf][2 - need:]
                values = iter(free)
                marker = next(values) if needs_marker else src_nd
                dst_nd = next(values) if dst_nd is None else dst_nd
        if keep is not None:
            src = np.where(keep[None], src, np.asarray(marker, dtype=src.dtype))
            src_nd = marker
        from . import _core

        has_nodata = src_nd is not None
        (kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense, dense_row0, dense_col0, address,
         args) = self._coords_args(coords, handles)
        out = _core.warp.warp(src, _METHODS[self.method], (h, w), origin, self.scales, has_nodata,
                              float(src_nd) if has_nodata else 0.0, float(dst_nd if dst_nd is not None else self.fill),
                              (r0, c0, rh, cw), kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense,
                              dense_row0, dense_col0, address, args, per_band=True, n_threads=n_threads)
        if self.dst_nodata is None:  # no destination NoData: the cells outside are not the marker but the fill
            out[out == dst_nd] = self.fill
        if back is not None:
            out = _round_back(out, back)
        return out

    def _handles(self, n: int):
        """GDAL's transformer for the kernel (n threads), but for the same CRS (never needed)."""
        return self.points.handles(n) if self.kind != _AFFINE else (0, np.empty(0, dtype=np.uintp))

    def sample_coords(self, r0, c0, nr, nc, n_threads: int = 1) -> np.ndarray:
        """The source coordinates the kernel takes for samples rows r0.., columns c0..: (nr, nc, 2)."""
        from . import _core

        lat = self.lattice(r0, c0, nr - self.extra, nc - self.extra, n_threads) if self.kind == _LATTICE else None
        dense = (self.points.grid(np.arange(c0, c0 + nc), np.arange(r0, r0 + nr), self.offset, n_threads), r0, c0) \
            if self.kind == _DENSE else None
        handles = self._handles(1)
        try:
            return _core.warp.coords(*self._coords_args(lat if lat is not None else dense, handles),
                                     (r0, c0, nr, nc))
        finally:
            self.points.release(handles[1])

    def region(self, src, origin, r0, c0, rh, cw, n_threads: int) -> np.ndarray:
        """The destination rows r0.., columns c0.. of every plane from src ((planes, rows,
        columns), the source cells from origin on), on n_threads threads."""
        src = np.ascontiguousarray(src)
        handles = self._handles(n_threads)
        try:
            if self.kind in (_AFFINE, _EXACT):
                return self._call(src, origin, r0, c0, rh, cw, None, handles, n_threads)
            if self.kind == _LATTICE:
                return self._call(src, origin, r0, c0, rh, cw, self.lattice(r0, c0, rh, cw, n_threads), handles,
                                  n_threads)
            out = np.empty((src.shape[0], rh, cw), dtype=self.dtype)
            rows = max(1, _DENSE_POINTS // (cw + self.extra))
            for a in range(r0, r0 + rh, rows):
                nr = min(rows, r0 + rh - a)
                g = self.points.grid(np.arange(c0, c0 + cw + self.extra), np.arange(a, a + nr + self.extra),
                                     self.offset, n_threads)
                out[:, a - r0:a - r0 + nr] = self._call(src, origin, a, c0, nr, cw, (g, a, c0), handles, n_threads)
            return out
        finally:
            self.points.release(handles[1])

    def in_memory(self, src) -> np.ndarray:
        """The whole destination from src ((planes, rows, columns), an array or anything that
        reads a window by slicing, such as a lazily loaded file): only the window the
        destination sees is read."""
        dh, dw = self.shape
        n = src.shape[0]
        win = self.window(0, 0, dh, dw, self.n_threads)
        if win is None:
            return np.full((n, dh, dw), self.fill, dtype=self.dtype)
        a, b, c, d = win
        part = np.asarray(src[:, a:b, c:d]).astype(self.dtype, copy=False)
        return self.region(part, (a, c), 0, 0, dh, dw, self.n_threads)

    def task(self, src, origin, r0, c0, rh, cw) -> np.ndarray:
        return self.region(np.asarray(src).astype(self.dtype, copy=False), origin, r0, c0, rh, cw, 1)

    def lazy(self, src, task: int):
        """A dask array of tasks of a group of planes (the source's chunks of them) and
        ``task`` cells a side, each reading the source window it sees (from the lattice of
        its cells, which the task builds again)."""
        import dask
        import dask.array as dsa

        dh, dw = self.shape
        run = dask.delayed(self.task, pure=True)
        windows = {}
        for r0 in range(0, dh, task):
            for c0 in range(0, dw, task):
                rh, cw = min(task, dh - r0), min(task, dw - c0)
                windows[r0, c0] = (rh, cw, self.window(r0, c0, rh, cw, self.n_threads))
        groups = []
        p0 = 0
        for size in src.chunks[0]:
            p1 = p0 + size
            rows = []
            for r0 in range(0, dh, task):
                row = []
                for c0 in range(0, dw, task):
                    rh, cw, win = windows[r0, c0]
                    if win is None:
                        row.append(dsa.full((size, rh, cw), self.fill, dtype=self.dtype,
                                            chunks=(size, rh, cw)))
                        continue
                    a, b, c, d = win
                    part = run(src[p0:p1, a:b, c:d], (a, c), r0, c0, rh, cw)
                    row.append(dsa.from_delayed(part, shape=(size, rh, cw), dtype=self.dtype))
                rows.append(row)
            groups.append(rows)
            p0 = p1
        return dsa.block(groups)


def _task_size(data) -> int:
    """Cells a side of the tasks of a lazy result: whole blocks, about the source's chunks."""
    side = max(data.chunks[-2][0], data.chunks[-1][0])
    return _BLOCK * min(4, max(1, round(side / _BLOCK)))


# --- a grid for a CRS and a resolution ------------------------------------------------


def suggested_grid(sources, crs: Any = None, res: Any = None, extent: Any = None, tap: bool = False,
                   cutline: Optional[Cutline] = None) -> Grid:
    """The grid gdalwarp makes for ``sources`` (``(transform, (height, width), crs)`` each)
    with ``-t_srs crs -tr res``: the extent that covers all of them in ``crs`` (GDAL's
    suggested output of each), cells of ``res`` (a number or ``(x, y)``) from its top-left
    corner, or GDAL's suggested resolution, the finest of the sources'. Without ``crs`` the
    sources' own; one source in its own CRS and resolution keeps its grid as it is.

    ``extent`` ``(xmin, ymin, xmax, ymax)`` in ``crs`` (``-te``): that extent instead; without ``res``
    the cells of the source (the same CRS only), fitted to the extent as gdalwarp does (a whole
    number of them, a little larger or smaller). ``tap`` (``-tap``, needs ``res``): the extent widened
    to whole cells from the origin of the coordinates. ``cutline`` (``-cutline -crop_to_cutline``):
    the extent of the polygons, which gdalwarp takes over ``-te``; without ``res`` in the sources' CRS,
    their cells entirely inside it."""
    from math import ceil, floor

    from rasterio.transform import Affine, array_bounds
    from rasterio.warp import calculate_default_transform, transform_bounds

    if not sources:
        raise ValueError("crs=/res=: no raster to take the extent from")
    if crs is None and any(s[2] is None for s in sources):
        raise ValueError("res=: the data has no CRS; pass crs= as well")
    if any(s[2] is None for s in sources):
        raise ValueError("crs=: the data has no CRS, so it cannot be reprojected")
    if tap and res is None:
        raise ValueError("tap=True needs res= (the cells to align to), as gdalwarp -tap needs -tr")
    if extent is not None and cutline is not None:
        raise ValueError("give extent= or cutline=, not both: gdalwarp crops to the cutline and ignores -te")
    dst = CRS.from_user_input(crs if crs is not None else sources[0][2])
    if res is not None:
        rx, ry = (res, res) if np.isscalar(res) else tuple(res)
        if not (float(rx) > 0 and float(ry) > 0):
            raise ValueError(f"res must be positive, got {res!r}")
        rx, ry = float(rx), float(ry)
    if extent is not None:
        try:
            left, bottom, right, top = (float(v) for v in extent)
        except (TypeError, ValueError):
            raise ValueError(f"extent= takes (xmin, ymin, xmax, ymax), got {extent!r}") from None
        if not (right > left and top > bottom):
            raise ValueError(f"extent= is empty: {tuple(extent)!r}")

    boxes = []
    true_boxes = []  # the extents of the sources, not rounded to cells
    same = True
    for t, (h, w), src_crs in sources:
        if t.b or t.d:
            raise ValueError("crs=/res=: rotated grids are not supported")
        src_crs = CRS.from_user_input(src_crs)
        if src_crs == dst:
            left_, bottom_, right_, top_ = array_bounds(h, w, t)
            boxes.append((left_, bottom_, right_, top_, abs(t.a), abs(t.e)))
            true_boxes.append((left_, bottom_, right_, top_))
            if len(sources) == 1 and res is None and extent is None and cutline is None:
                return _grid(t, (h, w), dst)
        else:
            same = False
            st, sw, sh = calculate_default_transform(src_crs, dst, w, h, *array_bounds(h, w, t))
            boxes.append((st.c, st.f + st.e * sh, st.c + st.a * sw, st.f, st.a, -st.e))
            true_boxes.append(transform_bounds(src_crs, dst, *array_bounds(h, w, t), densify_pts=21))
    fit = False
    if cutline is not None or extent is not None:
        if cutline is not None:
            left, bottom, right, top = cutline.bounds(dst)
        if res is None:
            if not same:
                raise ValueError("extent=/cutline= with another crs= need res= (gdalwarp's cell size for them "
                                 "is not reproduced)")
            rx, ry = min(b[4] for b in boxes), min(b[5] for b in boxes)
            t0 = sources[0][0]
            if cutline is not None:
                # the cells of the source entirely inside the cutline's extent
                if any(tuple(s[0])[:6] != tuple(t0)[:6] for s in sources):
                    raise ValueError("cutline= without res= needs the files on one grid")
                eps = 1e-6
                c0, c1 = ceil((left - t0.c) / t0.a - eps), floor((right - t0.c) / t0.a + eps)
                r0, r1 = ceil((top - t0.f) / t0.e - eps), floor((bottom - t0.f) / t0.e + eps)
                if c1 <= c0 or r1 <= r0:
                    raise ValueError("the cutline is smaller than a cell")
                return _grid(Affine(t0.a, 0.0, t0.c + c0 * t0.a, 0.0, t0.e, t0.f + r0 * t0.e), (r1 - r0, c1 - c0), dst)
            fit = True
    else:
        left, bottom = min(b[0] for b in boxes), min(b[1] for b in boxes)
        right, top = max(b[2] for b in boxes), max(b[3] for b in boxes)
        if res is None:
            rx, ry = min(b[4] for b in boxes), min(b[5] for b in boxes)
    from_south = False
    if tap and (extent is not None or cutline is not None):  # (-te, -crop_to_cutline: whole cells around it)
        def near(v):  # GDAL's ROUND_IF_CLOSE
            r = round(v)
            return float(r) if abs(v - r) < 1e-6 else v

        left, right = floor(near(left / rx)) * rx, ceil(near(right / rx)) * rx
        bottom, top = floor(near(bottom / ry)) * ry, ceil(near(top / ry)) * ry
    elif tap:  # the extent of the data: its west and south edges to the nearest multiples of the cell (halves go down)
        left, bottom = min(b[0] for b in true_boxes), min(b[1] for b in true_boxes)
        right, top = max(b[2] for b in true_boxes), max(b[3] for b in true_boxes)
        left, bottom = ceil(left / rx - 0.5) * rx, ceil(bottom / ry - 0.5) * ry
        from_south = True
    # gdalwarp's count of cells (the extent rounded to whole cells from the top-left corner)
    nx = max(1, int((right - left + rx / 2) / rx))
    ny = max(1, int((top - bottom + ry / 2) / ry))
    if from_south:  # (the cells are counted up from the aligned south edge)
        top = bottom + ny * ry
    if fit:  # an extent without cells: gdalwarp fits the cells to it
        rx, ry = (right - left) / nx, (top - bottom) / ny
    return _grid(Affine(rx, 0.0, left, 0.0, -ry, top), (ny, nx), dst)


def _grid(t, shape, crs: CRS) -> Grid:
    h, w = shape
    return Grid(t, (int(h), int(w)), crs, t.c + t.a * (np.arange(w) + 0.5), t.f + t.e * (np.arange(h) + 0.5))


# --- onto a grid ----------------------------------------------------------------------


def same_grid(da: xr.DataArray, grid: Grid) -> bool:
    return (CRS.from_user_input(da.rio.crs) == grid.crs and (da.sizes["y"], da.sizes["x"]) == tuple(grid.shape)
            and transform_of(da).almost_equals(grid.transform))


def _crop_offsets(da: xr.DataArray, grid: Grid):
    """(row, column) of the grid's first cell in the source when the grid only crops or pads
    it (the same CRS and cells, a whole number of cells apart); else None."""
    s, d = transform_of(da), grid.transform
    if s.b or s.d or d.b or d.d or CRS.from_user_input(da.rio.crs) != grid.crs:
        return None
    if abs(s.a - d.a) > 1e-9 * abs(s.a) or abs(s.e - d.e) > 1e-9 * abs(s.e):
        return None
    col, row = (d.c - s.c) / s.a, (d.f - s.f) / s.e
    if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
        return None
    return int(round(row)), int(round(col))


def _crop(values, offsets, src_shape, dst_shape, fill, dtype, avoid: bool = False, src_nodata: Any = None):
    """values ((planes, rows, columns): dask, or anything read by slicing) cut or padded to
    the grid with ``fill``; ``avoid``: the copied cells equal to ``fill`` move one step
    (GDAL's AvoidNoData, for a raster without NoData of its own); the cells equal to ``src_nodata``
    become ``fill``."""
    row, col = offsets
    h, w = src_shape
    dh, dw = dst_shape
    r0, r1 = max(0, row), min(h, row + dh)
    c0, c1 = max(0, col), min(w, col + dw)
    lazy = hasattr(values, "dask")
    if r1 <= r0 or c1 <= c0:
        if lazy:
            import dask.array as dsa

            return dsa.full((values.shape[0], dh, dw), fill, dtype=dtype, chunks=(values.chunks[0], dh, dw))
        return np.full((values.shape[0], dh, dw), fill, dtype=dtype)
    part = values[:, r0:r1, c0:c1]
    part = part.astype(dtype) if lazy else np.asarray(part).astype(dtype, copy=False)  # (_Planes reads it)
    same_nodata = src_nodata is None or src_nodata == fill or (np.isnan(src_nodata) and np.isnan(fill))
    if avoid or not same_nodata:
        part = part.map_blocks(_recode, src_nodata, fill, avoid, dtype=dtype) if lazy \
            else _recode(part, src_nodata, fill, avoid)
    pad = ((0, 0), (r0 - row, row + dh - r1), (c0 - col, col + dw - c1))
    if not any(p for side in pad for p in side):
        return part
    if lazy:
        import dask.array as dsa

        return dsa.pad(part, pad, mode="constant", constant_values=fill)
    return np.pad(part, pad, mode="constant", constant_values=fill)


def _avoid_nodata(values, nodata):
    """GDAL's AvoidNoData on copied cells: a value equal to the destination NoData moves one
    step (an integer raster without NoData gets one)."""
    if np.issubdtype(values.dtype, np.floating):
        if np.isnan(nodata) or not (values == nodata).any():
            return values
        top = np.finfo(values.dtype).max
        moved = np.nextafter(values, values.dtype.type(0 if nodata == top else top))
        return np.where(values == nodata, moved, values)
    info = np.iinfo(values.dtype)
    moved = info.min + 1 if nodata == info.min else nodata - 1
    return values if not (values == nodata).any() else \
        np.where(values == nodata, np.asarray(moved, dtype=values.dtype), values)


def _recode(values, src_nodata, nodata, avoid: bool):
    """The copied cells as GDAL's kernel writes them: those equal to the source's NoData become the
    destination's, the others equal to it move one step when ``avoid``."""
    invalid = None
    if src_nodata is not None:
        invalid = np.isnan(values) if np.isnan(src_nodata) else values == src_nodata
    if avoid:
        values = _avoid_nodata(values, nodata)
    if invalid is not None and invalid.any():
        values = np.where(invalid, np.asarray(nodata, dtype=values.dtype), values)
    return values


def _method(da: xr.DataArray, resampling: str) -> str:
    if resampling != "auto":
        return resampling
    return "bilinear" if np.issubdtype(da.dtype, np.floating) else "nearest"


def to_grid(da: xr.DataArray, grid: Grid, resampling: str = "auto", *, tolerance: float = 0.125,
            n_threads: Optional[int] = None, nodata: Any = None, dtype: Any = None,
            cutline: Optional[Cutline] = None) -> xr.DataArray:
    """``da`` (``(..., y, x)``, georeferenced) on ``grid``: its CRS, cells and the grid's x and
    y coordinates. The dims before ``y``/``x`` are kept, every plane warped on its own;
    a lazy cube stays lazy. ``resampling="auto"``: the nearest cell for integers, bilinear
    for floats, the nearest cell for QA bands (``QA_BANDS``) whatever their type.
    ``nodata`` (None, ``"source"`` or a number), ``dtype`` and ``cutline`` (a ``Cutline``): see
    ``load_raster``."""
    if da.rio.crs is None:
        raise ValueError("load_raster(like=): the data has no CRS, so it cannot be put on another grid")
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError("load_raster(like=): the data needs y and x dimensions")
    if resampling != "auto" and resampling not in _METHODS:
        raise ValueError(f"unknown resampling {resampling!r}; use 'auto' or one of {', '.join(RESAMPLING_NAMES)}")
    if not tolerance >= 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")
    if same_grid(da, grid) and (nodata is None or nodata == "source") and dtype is None and cutline is None:
        return da.assign_coords(x=grid.x, y=grid.y)
    if resampling == "auto" and "band" in da.dims and da.sizes["band"] > 1:
        names = [str(b).lower() for b in da["band"].values] if "band" in da.coords else []
        qa = [i for i, name in enumerate(names) if name in QA_BANDS]
        if qa and len(qa) < len(names):
            other = [i for i in range(len(names)) if i not in qa]
            parts = [_to_grid(da.isel(band=qa), grid, "nearest", tolerance, n_threads, nodata, dtype, cutline),
                     _to_grid(da.isel(band=other), grid, _method(da, "auto"), tolerance, n_threads, nodata, dtype,
                              cutline)]
            out = xr.concat(parts, dim="band", coords="minimal", compat="override", join="override")
            return out.isel(band=np.argsort(qa + other))
        if qa:
            return _to_grid(da, grid, "nearest", tolerance, n_threads, nodata, dtype, cutline)
    return _to_grid(da, grid, _method(da, resampling), tolerance, n_threads, nodata, dtype, cutline)


def _to_grid(da: xr.DataArray, grid: Grid, method: str, tolerance: float, n_threads: Optional[int], nodata: Any = None,
             dtype: Any = None, cutline: Optional[Cutline] = None) -> xr.DataArray:
    src_t = transform_of(da)
    if src_t.b or src_t.d or grid.transform.b or grid.transform.d:
        raise ValueError("load_raster(like=): rotated grids are not supported")
    src_crs = CRS.from_user_input(da.rio.crs)
    plan = _plan(da, nodata, dtype)
    encoded = da.rio.encoded_nodata
    attrs = {k: v for k, v in da.attrs.items() if k != "_FillValue"}
    lead = [d for d in da.dims if d not in ("y", "x")]
    data = da.transpose(*lead, "y", "x")
    lead_shape = tuple(data.sizes[d] for d in lead)
    n = int(np.prod(lead_shape)) if lead else 1
    h, w = data.sizes["y"], data.sizes["x"]
    dh, dw = grid.shape
    # GDAL applies a cutline to the cells of the source: those outside the polygons take no part in the warp
    keep = None if cutline is None else cutline.mask(Grid(src_t, (h, w), src_crs, None, None))
    if hasattr(data.data, "dask"):
        values = data.data.reshape((n, h, w))
    else:
        values = _Planes(data, n)  # a file read lazily is read a window at a time
    offsets = _crop_offsets(data, grid)
    if offsets is not None:
        sn = plan.src_nodata  # (valid cells never equal the NoData of the source, which is the result's)
        shared = sn is not None and (sn == plan.fill or (np.isnan(sn) and np.isnan(plan.fill)))
        out = _crop(values, offsets, (h, w), (dh, dw), plan.fill, plan.work, avoid=not plan.sentinel and not shared,
                    src_nodata=sn)
        if keep is not None:
            inside = _crop(keep[None], offsets, (h, w), (dh, dw), False, np.dtype(bool))
            fill = np.asarray(plan.fill, dtype=plan.work)
            out = _where(inside, out, fill)
    else:
        warp = _Warp((h, w), src_t, src_crs, grid.transform, (dh, dw), grid.crs, method, plan.src_nodata,
                     plan.kernel_nodata, plan.work, float(tolerance), resolve_threads(n_threads), fill=plan.fill,
                     keep=keep)
        out = warp.lazy(values, _task_size(values)) if hasattr(values, "dask") else warp.in_memory(values)
    out = _converted(out, plan.dtype)
    out = out.reshape(lead_shape + (dh, dw))
    coords = {name: c for name, c in data.coords.items()
              if "y" not in c.dims and "x" not in c.dims and name not in ("spatial_ref", "x", "y")}
    coords.update({"y": grid.y, "x": grid.x})
    result = xr.DataArray(out, dims=(*lead, "y", "x"), coords=coords, name=da.name, attrs=attrs)
    result.rio.write_crs(grid.crs, inplace=True)
    result.rio.write_transform(grid.transform, inplace=True)
    if encoded is not None and np.issubdtype(plan.dtype, np.floating):
        result.rio.write_nodata(encoded, encoded=True, inplace=True)
    elif plan.nodata is not None:
        result.rio.write_nodata(plan.nodata, encoded=False, inplace=True)
    return result


class _Planes:
    """A cube not in dask (in memory, or a file xarray reads lazily), ``(..., y, x)``, seen as
    (planes, rows, columns) and read a window at a time."""

    def __init__(self, da: xr.DataArray, n: int):
        self.da = da
        self.shape = (n, da.sizes["y"], da.sizes["x"])

    def __getitem__(self, key):
        planes, rows, cols = key
        block = np.asarray(self.da.isel(y=rows, x=cols).values)
        return block.reshape((self.shape[0],) + block.shape[-2:])[planes]
