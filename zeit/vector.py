"""Zones and vectors: the area of each pixel, a vector layer burned on a grid, a map's patches as polygons and
the statistics and areas of a map (or a series of them) in each zone of a layer.

The functions take what the rest of zeit takes: a ``DataArray`` or a raster path, a ``GeoDataFrame`` or a path of
a vector file. Janus' Python history writes its zonal, rasterize and polygonize runs as calls to them.
"""

import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["pixel_area", "rasterize", "polygonize", "zonal_stats", "zonal_areas", "ZonalTables"]

#: Map projections (WKT1 names, lower case) whose map area is the ellipsoid's.
_EQUAL_AREA = ("equal_area", "albers", "mollweide", "sinusoidal", "eckert_iv", "eckert_vi", "homolosine",
               "equal_earth", "quartic_authalic")
_STATS = ("n", "mean", "std", "min", "p10", "p25", "median", "p75", "p90", "max")


# ---------------------------------------------------------------------------
# Grids and areas
# ---------------------------------------------------------------------------

def _pyproj_crs(crs: Any):
    """A pyproj CRS from a rasterio/rioxarray CRS, a pyproj one, a text or an EPSG code (None: none)."""
    import pyproj

    if crs is None:
        return None
    if isinstance(crs, pyproj.CRS):
        return crs
    if hasattr(crs, "to_wkt"):
        return pyproj.CRS.from_wkt(crs.to_wkt())
    return pyproj.CRS.from_user_input(crs)


def _grid_of(like: Any):
    """(pyproj CRS or None, transform, (rows, columns)) of the grid of a raster path, ``DataArray`` or ``Dataset``."""
    if isinstance(like, (str, os.PathLike)):
        import rasterio

        with rasterio.open(like) as src:
            return _pyproj_crs(src.crs), src.transform, (src.height, src.width)
    from ._warp import transform_of

    if isinstance(like, xr.Dataset):
        like = like[list(like.data_vars)[0]]
    if not isinstance(like, xr.DataArray):
        raise TypeError("the grid is a raster path, a DataArray or a Dataset")
    return _pyproj_crs(like.rio.crs), transform_of(like), (like.sizes["y"], like.sizes["x"])


def _area_kind(crs) -> str:
    """``"planar"`` (no CRS: the grid units squared), ``"equal"`` (an equal-area projection: the same area for
    every pixel) or ``"geodesic"`` (longitudes and latitudes, or another projection, whose scale varies)."""
    if crs is None or not (crs.is_geographic or crs.is_projected):
        return "planar"
    if crs.is_geographic:
        return "geodesic"
    try:
        method = re.search(r'PROJECTION\["([^"]+)"', crs.to_wkt("WKT1_GDAL")).group(1).lower()
    except Exception:  # noqa: BLE001 - a CRS WKT1 cannot say
        return "geodesic"
    return "equal" if any(m in method for m in _EQUAL_AREA) else "geodesic"


def _nominal_area(crs, t) -> float:
    """The transform's pixel area, in m2 when the CRS is projected (in degrees2 when geographic)."""
    area = abs(t.a * t.e - t.b * t.d)
    if crs is not None and crs.is_projected:
        try:
            area *= crs.axis_info[0].unit_conversion_factor ** 2
        except Exception:  # noqa: BLE001 - CRS without linear units
            pass
    return float(area)


def _area_samples(crs, t, c0: int, r0: int, c1: int, r1: int, samples: Optional[int] = 128):
    """The geodesic area (m2) of the pixels of columns ``c0:c1`` and rows ``r0:r1`` at the nodes of a grid of at
    most ``samples`` + 1 a side (every pixel for ``samples=None``): (rows, columns, areas), the first two relative
    to ``(r0, c0)``."""
    from pyproj import Transformer

    step = 1 if samples is None else max(-(-max(c1 - c0, r1 - r0) // samples), 1)
    xs, ys = (np.unique([*range(0, n, step), n - 1]) for n in (c1 - c0, r1 - r0))
    nodes = np.broadcast_arrays(c0 + xs[None, :], r0 + ys[:, None])
    corners = [t * (nodes[0] + dx, nodes[1] + dy) for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
    lon, lat = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True).transform(
        np.stack([x for x, _ in corners], -1), np.stack([y for _, y in corners], -1))
    geod = crs.get_geod()
    areas = np.array([[abs(geod.polygon_area_perimeter(lon[j, i], lat[j, i])[0]) for i in range(len(xs))]
                      for j in range(len(ys))])
    return ys, xs, areas


def _interpolate(sampled, rows, cols) -> np.ndarray:
    """The sampled areas bilinearly interpolated at pixels (``rows`` and ``cols`` relative to the samples' origin)."""
    from scipy.interpolate import RegularGridInterpolator

    ys, xs, areas = sampled
    if areas.size == 1:
        return np.full(len(rows), areas[0, 0])
    if len(ys) == 1:  # one row of samples
        return np.interp(cols, xs, areas[0])
    if len(xs) == 1:
        return np.interp(rows, ys, areas[:, 0])
    return RegularGridInterpolator((ys, xs), areas)(np.column_stack([rows, cols]))


def _centre_area(crs, t, ny: int, nx: int) -> float:
    """The area (m2) of the pixel at the grid's centre."""
    if _area_kind(crs) == "geodesic":
        try:
            return float(_area_samples(crs, t, nx // 2, ny // 2, nx // 2 + 1, ny // 2 + 1)[2][0, 0])
        except Exception:  # noqa: BLE001 - no transformation to the ellipsoid
            pass
    return _nominal_area(crs, t)


def pixel_area(da: Any, *, samples: Optional[int] = 128) -> xr.DataArray:
    """The area of each pixel of a grid, in m2.

    On a geographic CRS and on a projection that does not preserve areas it is the geodesic area of the pixel on
    the CRS's ellipsoid (Karney's algorithm, as GDAL's ``OGR_G_GeodesicArea``), measured at the nodes of a grid of
    at most ``samples`` + 1 a side and interpolated between them; on an equal-area projection it is the transform's
    area, the same for every pixel; without a CRS the transform's, in the grid units squared.

    Parameters
    ----------
    da
        A raster path, ``DataArray`` or ``Dataset`` (only its grid is used).
    samples
        Nodes a side of the measured grid (default 128: at most 129 x 129 measures however large the grid);
        ``None`` measures every pixel.

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` float64, the grid's coordinates, ``units`` ``"m2"``.
    """
    crs, t, (ny, nx) = _grid_of(da)
    kind = _area_kind(crs)
    area = np.empty((ny, nx))
    sampled = None
    if kind == "geodesic":
        try:
            sampled = _area_samples(crs, t, 0, 0, nx, ny, samples)
        except Exception:  # noqa: BLE001 - no transformation to the ellipsoid
            kind = "planar"
    if sampled is None:
        area[:] = _nominal_area(crs, t)
    else:
        cols = np.arange(nx)
        step = max(1, 2_000_000 // max(nx, 1))
        for r in range(0, ny, step):
            rows = np.arange(r, min(r + step, ny))
            grid = np.broadcast_arrays(rows[:, None], cols[None, :])
            area[rows] = _interpolate(sampled, grid[0].ravel(), grid[1].ravel()).reshape(len(rows), nx)
    out = xr.DataArray(area, dims=("y", "x"), name="pixel_area", attrs={"units": "m2", "long_name": "pixel area"})
    out = out.assign_coords(x=_centres(t, nx, 0), y=_centres(t, ny, 1))
    if crs is not None:
        out = out.rio.write_crs(crs.to_wkt())
    return out.rio.write_transform(t)


def _centres(t, n: int, axis: int) -> np.ndarray:
    """The coordinates of the centres of ``n`` pixels along x (``axis`` 0) or y (1) of a north-up transform."""
    i = np.arange(n) + 0.5
    return t.c + t.a * i if axis == 0 else t.f + t.e * i


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def _zones(vector: Any, layer: Optional[str], crs):
    """The layer as a ``GeoDataFrame`` in ``crs`` (a path is read with the feature ids as the index)."""
    import geopandas as gpd

    if isinstance(vector, (str, os.PathLike)):
        zones = gpd.read_file(vector, layer=layer, fid_as_index=True)
    elif isinstance(vector, gpd.GeoDataFrame):
        zones = vector
    elif isinstance(vector, gpd.GeoSeries):
        zones = gpd.GeoDataFrame(geometry=vector)
    else:
        raise TypeError("the zones are a vector file path, a GeoDataFrame or a GeoSeries")
    if crs is not None and zones.crs is not None:
        zones = zones.to_crs(crs)
    return zones


def _field_kind(vector: Any, layer: Optional[str], zones, attribute: str) -> str:
    """``"int"``, ``"real"`` or ``"text"``: the type of the attribute (the file's own when the layer is a file,
    so that integers with a null stay integers)."""
    if isinstance(vector, (str, os.PathLike)):
        try:
            import pyogrio

            kind = str(pyogrio.read_info(vector, layer=layer)["dtypes"][list(pyogrio.read_info(
                vector, layer=layer)["fields"]).index(attribute)])
            if kind.startswith(("int", "uint")):
                return "int"
            return "real" if kind.startswith("float") else "text"
        except Exception:  # noqa: BLE001 - fall back on the frame's types
            pass
    column = zones[attribute]
    if pd.api.types.is_bool_dtype(column) or pd.api.types.is_integer_dtype(column):
        return "int"
    return "real" if pd.api.types.is_float_dtype(column) else "text"


def rasterize(vector: Any, like: Any, burn: Union[int, float, str] = 1, *, layer: Optional[str] = None,
              all_touched: bool = False, name: Optional[str] = None) -> xr.DataArray:
    """Burn a vector layer on the grid of a raster.

    The features, in the grid's CRS and in reading order, are burned by GDAL (rasterio's ``rasterize``: a later
    feature over an earlier one; a pixel is burned when its centre is inside the feature, or when the feature
    touches it with ``all_touched``).

    Parameters
    ----------
    vector
        A vector file path or a ``GeoDataFrame``.
    like
        The grid: a raster path, ``DataArray`` or ``Dataset``.
    burn
        ``1`` (default): ``uint8`` 1 in the features, 0 elsewhere; another number: that value (``uint8`` between 0
        and 255, else ``int32`` or ``float64``); ``"id"``: the feature's number (1, 2... counting every feature read)
        as ``int32``, 0 elsewhere; or an attribute: its value as it is (an integer field as ``int32`` with no data
        ``-2**31``, a real one as ``float64`` with no data NaN) or, for a text, a class numbered from 1 in sorted
        order (``uint8``, ``int32`` from 255 classes; 0 is no data) whose names are in the ``attrs``
        ``flag_values`` and ``flag_meanings``. Features without geometry, or without a value, are left out.
    layer
        The layer of a multi-layer file.
    all_touched
        Burn every pixel a feature touches.
    name
        The result's name.

    Returns
    -------
    xarray.DataArray
        ``(y, x)`` on the grid of ``like`` (its CRS and transform; the no data in ``.rio.nodata``).
    """
    from rasterio.features import rasterize as burn_shapes

    crs, t, shape = _grid_of(like)
    zones = _zones(vector, layer, crs)
    has = zones.geometry.notna()
    fill, nodata, attrs = 0, None, {}
    if isinstance(burn, str) and burn == "id":
        zones = zones.assign(_burn=range(1, len(zones) + 1))[has]
        values, dtype, nodata = zones["_burn"], "int32", 0
        long_name = "feature number"
    elif isinstance(burn, str):
        if burn not in zones.columns:
            raise KeyError(f"rasterize: no attribute {burn!r} in the layer")
        kind = _field_kind(vector, layer, zones, burn)
        zones = zones[has & zones[burn].notna()]
        long_name = burn
        if kind == "real":
            values, dtype, fill, nodata = zones[burn], "float64", float("nan"), float("nan")
        elif kind == "int":
            values, dtype, fill, nodata = zones[burn], "int32", -2**31, -2**31
        else:
            classes = sorted(zones[burn].unique())
            dtype = "uint8" if len(classes) < 255 else "int32"
            values, nodata = zones[burn].map({c: i for i, c in enumerate(classes, 1)}), 0
            attrs = {"flag_values": list(range(1, len(classes) + 1)),
                     "flag_meanings": " ".join(str(c).replace(" ", "_") for c in classes)}
    else:
        zones = zones[has]
        if burn == 1:
            dtype = "uint8"
        elif float(burn).is_integer() and 0 <= burn < 256:
            dtype = "uint8"
        elif float(burn).is_integer():
            dtype = "int32"
        else:
            dtype = "float64"
        values, long_name = pd.Series(burn, index=zones.index), "burned"
    pairs = list(zip(zones.geometry, values))
    if pairs:
        burned = burn_shapes(pairs, shape, transform=t, fill=fill, dtype=dtype, all_touched=all_touched)
    else:
        burned = np.full(shape, fill, dtype=dtype)
    out = xr.DataArray(burned, dims=("y", "x"), name=name or long_name.replace(" ", "_"),
                       coords={"y": _centres(t, shape[0], 1), "x": _centres(t, shape[1], 0)},
                       attrs={"long_name": long_name, **attrs})
    if crs is not None:
        out = out.rio.write_crs(crs.to_wkt())
    out = out.rio.write_transform(t)
    if nodata is not None:
        out = out.rio.write_nodata(nodata)
    return out


# ---------------------------------------------------------------------------
# Polygonize
# ---------------------------------------------------------------------------

def _as_map(source: Any, band: Any = None, like: Any = None) -> xr.DataArray:
    """A single map as a 2-D ``DataArray``, no data as NaN: a path read with ``load_raster`` (``band`` the 1-based
    band), or a ``DataArray`` with its length-1 dimensions dropped."""
    from ._load import load_raster

    if isinstance(source, (str, os.PathLike)) or like is not None:
        kw = {"like": like} if like is not None else {}
        if band is not None:
            kw["band"] = band
        da = load_raster(source, masked=True, **kw)
    else:
        da = source
    if isinstance(da, xr.Dataset):
        da = da[list(da.data_vars)[0]]
    if not isinstance(da, xr.DataArray):
        raise TypeError("a map is a raster path or a DataArray")
    extra = [d for d in da.dims if d not in ("y", "x")]
    da = da.squeeze(extra, drop=True) if all(da.sizes[d] == 1 for d in extra) else da
    if da.ndim != 2:
        raise ValueError(f"expected a 2-D map, got dims {da.dims}")
    return da


def _date_text(v: float) -> str:
    """A decimal year as a date (YYYY-MM-DD), a whole year as itself."""
    import calendar
    import datetime

    year = int(v)
    days = 366 if calendar.isleap(year) else 365
    if v == year:
        return str(year)
    return (datetime.date(year, 1, 1) + datetime.timedelta(days=int((v - year) * days))).isoformat()


def polygonize(raster: Any, *, band: Optional[int] = None, connectivity: int = 4, min_area_ha: float = 0.0,
               min_pixels: Optional[int] = None, simplify: float = 0.0, classes: Optional[Dict[Any, str]] = None,
               years: bool = False, mean_of: Any = None, mean_band: Optional[int] = None,
               mean_name: str = "mean"):
    """The patches of a map (pixels of one value connected 4- or 8-wise) as polygons.

    The patches come from GDAL's polygonizer (rasterio's ``shapes``). Each has its ``value``, its ``area_ha`` and
    ``perimeter_m`` (measured on the CRS's ellipsoid; the geometry's own without a CRS) and its ``pixels``.

    Parameters
    ----------
    raster
        A raster path or a 2-D ``DataArray``; NaN (no data) is not a patch.
    band
        The band of a file (1-based).
    connectivity
        4 or 8.
    min_area_ha, min_pixels
        A minimum patch, in hectares (turned into pixels with the area of the pixel at the grid's centre) or in
        pixels: smaller patches are merged by ``mmu_sieve`` into the neighbour they share the longest border with,
        and a patch with no neighbour to join is left out.
    simplify
        Simplify the polygons to this tolerance in metres (degrees at 111,320 m each in a geographic CRS), keeping
        the topology.
    classes
        Class names by value: adds a ``class`` column.
    years
        The values are decimal years: adds a ``date`` column (``YYYY-MM-DD``; a whole year as itself).
    mean_of, mean_band, mean_name
        Another map (path or ``DataArray``; ``mean_band`` of a file) put on this grid, whose mean in each
        patch is added as the ``mean_name`` column.

    Returns
    -------
    geopandas.GeoDataFrame
        ``value``, ``class``, ``date``, ``area_ha``, ``perimeter_m``, ``pixels``, the mean and ``geometry``
        (those asked for), in the raster's CRS.
    """
    import geopandas as gpd
    import shapely
    from rasterio.features import rasterize as burn_shapes
    from rasterio.features import shapes
    from shapely.geometry import shape

    from ._load import load_raster
    from .spatial import mmu_sieve

    if connectivity not in (4, 8):
        raise ValueError("polygonize: connectivity is 4 or 8")
    raster = _as_map(raster, band)
    crs = _pyproj_crs(raster.rio.crs)
    from ._warp import transform_of

    t = transform_of(raster)
    if min_pixels is None and min_area_ha > 0:
        pixel = _centre_area(crs, t, *raster.shape)
        min_pixels = math.ceil(min_area_ha * 1e4 / pixel) if pixel > 0 else 0
    sieve = bool(min_pixels) and min_pixels >= 2
    if sieve:
        raster = mmu_sieve(raster, min_pixels, connectivity=connectivity)
    found = list(shapes(raster.values.astype("float32"), mask=np.isfinite(raster.values),
                        connectivity=connectivity, transform=t))
    patches = gpd.GeoDataFrame({"value": [v for _, v in found]}, geometry=[shape(g) for g, _ in found],
                               crs=raster.rio.crs)
    patches["pixels"] = (shapely.area(np.asarray(patches.geometry.values)) / abs(t.a * t.e)).round().astype(int)
    cols = ["value"]
    if mean_of is not None:
        other = _as_map(mean_of, mean_band, like=raster) if isinstance(mean_of, (str, os.PathLike)) \
            else load_raster(mean_of, like=raster, masked=True)
        ids = burn_shapes(zip(patches.geometry, range(1, len(patches) + 1)), raster.shape, transform=t,
                          dtype="int32")
        patches[mean_name] = pd.Series(np.asarray(other.values).ravel().astype("float64")).groupby(
            ids.ravel()).mean().reindex(range(1, len(patches) + 1)).values
    if sieve:
        patches = patches[patches.pixels >= min_pixels].copy()  # the rest had no neighbour to join
    if simplify > 0:
        tolerance = simplify / 111_320.0 if crs is not None and crs.is_geographic else simplify
        patches.geometry = patches.simplify(tolerance, preserve_topology=True)
    if classes:
        patches["class"] = patches.value.map(dict(classes))
        cols.append("class")
    if years:
        patches["date"] = patches.value.map(_date_text)
        cols.append("date")
    if crs is not None:
        geod, lonlat = patches.crs.get_geod(), patches.to_crs(patches.crs.geodetic_crs).geometry
        patches["area_ha"] = [abs(geod.geometry_area_perimeter(g)[0]) / 1e4 for g in lonlat]
        patches["perimeter_m"] = [geod.geometry_length(g.boundary) for g in lonlat]
    else:
        patches["area_ha"], patches["perimeter_m"] = patches.geometry.values.area / 1e4, patches.length
    cols += ["area_ha", "perimeter_m", "pixels"] + ([mean_name] if mean_of is not None else []) + ["geometry"]
    return patches[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Zonal statistics and areas
# ---------------------------------------------------------------------------

class ZonalTables(NamedTuple):
    """The tables of ``zonal_stats`` and ``zonal_areas``."""

    #: One row per zone and map (statistics), or per zone, map and class (or year).
    long: pd.DataFrame
    #: One row per zone: its pixels and one column per map (one statistic) or per class (hectares).
    wide: pd.DataFrame
    #: One row per zone and transition between two maps (``zonal_areas(transitions=)``), else None.
    transitions: Optional[pd.DataFrame]

    def to_csv(self, folder: Any) -> List[Path]:
        """Write the tables as CSV files in ``folder`` (created if needed).

        ``zonal_long.csv`` and ``zonal_wide.csv``, and ``transitions.csv`` when there are
        transitions, each as ``DataFrame.to_csv(path, index=False)`` writes it.

        Parameters
        ----------
        folder
            The folder to write into.

        Returns
        -------
        The paths written, in that order.
        """
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        files = [("zonal_long.csv", self.long), ("zonal_wide.csv", self.wide)]
        if self.transitions is not None:
            files.append(("transitions.csv", self.transitions))
        paths = []
        for name, table in files:
            table.to_csv(out / name, index=False)
            paths.append(out / name)
        return paths


def _column(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(s)).strip("_")


def _maps(rasters: Any, bands: Any, labels: Optional[Sequence[str]], window) -> Tuple[List[xr.DataArray], List[str]]:
    """The maps (2-D, no data as NaN) and their labels. An item is a path, a ``DataArray`` (2-D, or ``(time|band,
    y, x)`` whose layers are maps) or a ``(path, band[, label])`` tuple."""
    if isinstance(rasters, (str, os.PathLike, xr.DataArray, xr.Dataset)) or (
            isinstance(rasters, tuple) and len(rasters) in (2, 3) and isinstance(rasters[1], (int, np.integer))
            and isinstance(rasters[0], (str, os.PathLike))):
        rasters = [rasters]
    bands = [bands] * len(rasters) if bands is None or np.isscalar(bands) else list(bands)
    maps, names = [], []
    for item, band in zip(rasters, bands):
        label = None
        if isinstance(item, tuple):
            item, band, *rest = item
            label = rest[0] if rest else None
        if isinstance(item, xr.Dataset):
            item = item[list(item.data_vars)[0]]
        if isinstance(item, xr.DataArray) and item.ndim == 3 and band is None:
            lead = [d for d in item.dims if d not in ("y", "x")][0]
            for k in range(item.sizes[lead]):
                maps.append(item.isel({lead: k}, drop=False))
                names.append(str(item[lead].values[k]) if lead in item.coords else str(k + 1))
            continue
        da = _as_map(item, band)
        maps.append(da)
        names.append(label if label is not None else Path(str(item)).stem
                     if isinstance(item, (str, os.PathLike)) else str(da.name if da.name is not None else len(maps)))
    if labels is not None:
        if len(labels) != len(maps):
            raise ValueError(f"zonal: {len(labels)} labels for {len(maps)} maps")
        names = [str(label) for label in labels]
    if window is not None:
        x0, y0, x1, y1 = window
        maps = [m.isel(y=slice(y0, y1), x=slice(x0, x1)) for m in maps]
    if len({m.shape for m in maps}) > 1:
        raise ValueError("zonal: the maps are not on the same grid (put them on one with load_raster(like=))")
    return maps, names


class _Burned(NamedTuple):
    zones: Any
    ids: np.ndarray
    maps: List[xr.DataArray]
    labels: List[str]
    t: Any
    shape: Tuple[int, int]
    extents: List[Any]
    cells: List[np.ndarray]
    zone: np.ndarray
    flat: np.ndarray
    crs: Any


def _burn_zones(zones: Any, rasters: Any, *, bands, labels, layer, id_field, all_touched, window) -> _Burned:
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    maps, names = _maps(rasters, bands, labels, window)
    crs = _pyproj_crs(maps[0].rio.crs)
    zones = _zones(zones, layer, crs)
    ids = (zones.index.astype(str).values if id_field is None
           else zones[id_field].fillna("").astype(str).values)
    t = maps[0].rio.transform(recalc=True)
    H, W = maps[0].shape

    def burn(g):
        """A zone's extent on the grid (a cell around) and its cells' positions, the zone burned alone on it."""
        if g is None or g.is_empty:
            return None, np.zeros(0, int)
        (c0, r0), (c1, r1) = ~t * (g.bounds[0], g.bounds[3]), ~t * (g.bounds[2], g.bounds[1])
        c0, r0 = max(int(np.floor(c0)) - 1, 0), max(int(np.floor(r0)) - 1, 0)
        c1, r1 = min(int(np.ceil(c1)) + 1, W), min(int(np.ceil(r1)) + 1, H)
        if c1 <= c0 or r1 <= r0:
            return None, np.zeros(0, int)
        inside = geometry_mask([g], (r1 - r0, c1 - c0), t * Affine.translation(c0, r0), all_touched=all_touched,
                               invert=True)
        rows, cols = np.nonzero(inside)
        return (c0, r0, c1, r1), (rows + r0) * W + cols + c0

    burned = [burn(g) for g in zones.geometry]
    cells = [c for _, c in burned]
    zone = np.repeat(np.arange(len(zones)), [len(c) for c in cells])
    flat = np.concatenate(cells) if cells else np.zeros(0, int)
    return _Burned(zones, ids, maps, names, t, (H, W), [e for e, _ in burned if e], cells, zone, flat, crs)


def _cell_areas(b: _Burned) -> np.ndarray:
    """The area (m2) of each burned cell: the CRS's (see ``pixel_area``), measured over the zones' extent."""
    H, W = b.shape
    kind = _area_kind(b.crs)
    if kind == "geodesic" and b.extents:
        try:
            c0, r0, c1, r1 = (f([e[i] for e in b.extents]) for f, i in ((min, 0), (min, 1), (max, 2), (max, 3)))
            sampled = _area_samples(b.crs, b.t, c0, r0, c1, r1)
            return _interpolate(sampled, b.flat // W - r0, b.flat % W - c0)
        except Exception:  # noqa: BLE001 - no transformation to the ellipsoid
            kind = "planar"
    return np.full(len(b.flat), _nominal_area(b.crs, b.t))


def zonal_stats(zones: Any, rasters: Any, *, bands: Any = None, labels: Optional[Sequence[str]] = None,
                layer: Optional[str] = None, id_field: Optional[str] = None, all_touched: bool = False,
                stat: str = "mean", window: Optional[Tuple[int, int, int, int]] = None) -> ZonalTables:
    """The statistics of continuous maps in each zone of a layer.

    Each feature is a zone; its pixels are those whose centre is inside it (also those it touches with
    ``all_touched``; a pixel in several features counts in each) on the maps' grid. In each map, the zone's
    values (no data left out) give ``n``, ``mean``, ``std`` (population), ``min``, ``p10``, ``p25``, ``median``,
    ``p75``, ``p90`` (linear interpolation) and ``max``.

    Parameters
    ----------
    zones
        A vector file path or a ``GeoDataFrame`` (reprojected to the maps' CRS).
    rasters
        What to measure: a map or a list of maps on one grid, each a raster path, a ``DataArray`` (2-D, or with
        a leading ``time``/``band`` dimension whose layers are maps) or a ``(path, band[, label])`` tuple.
    bands
        The band (1-based) of each file (one for all, or a list).
    labels
        A label for each map (default: the file's name, the tuple's label, the layer's coordinate).
    layer
        The layer of a multi-layer file.
    id_field
        The attribute that names the zones in the ``id`` column (default: the feature id).
    all_touched
        Also the pixels a zone touches.
    stat
        The statistic in the wide table (one of ``n``, ``mean``, ``std``, ``min``, ``p10``, ``p25``, ``median``,
        ``p75``, ``p90``, ``max``).
    window
        Pixel window ``(x0, y0, x1, y1)`` of the maps to work in.

    Returns
    -------
    ZonalTables
        ``long``: ``id``, ``fid``, ``date`` (the map's label) and the ten statistics, a row per zone and map;
        ``wide``: ``id``, ``fid``, ``pixels`` (the zone's pixels, valid or not) and a ``<stat>_<label>`` column
        per map; ``transitions`` is None.
    """
    if stat not in _STATS:
        raise ValueError(f"zonal_stats: stat is one of {', '.join(_STATS)}")
    b = _burn_zones(zones, rasters, bands=bands, labels=labels, layer=layer, id_field=id_field,
                    all_touched=all_touched, window=window)
    long: List[pd.DataFrame] = []
    wide = pd.DataFrame({"id": b.ids, "fid": b.zones.index, "pixels": [len(c) for c in b.cells]})
    n = len(b.zones)
    for label, raster in zip(b.labels, b.maps):
        x = pd.Series(raster.values.ravel()[b.flat].astype("float32").astype("float64")).groupby(b.zone)
        s = x.agg(n="count", mean="mean", std=lambda v: v.std(ddof=0), min="min", p10=lambda v: v.quantile(0.1),
                  p25=lambda v: v.quantile(0.25), median="median", p75=lambda v: v.quantile(0.75),
                  p90=lambda v: v.quantile(0.9), max="max").reindex(range(n))
        s["n"] = s["n"].fillna(0).astype(int)
        long.append(pd.concat([pd.DataFrame({"id": b.ids, "fid": b.zones.index, "date": label}),
                               s.reset_index(drop=True)], axis=1))
        wide[f"{stat}_{_column(label)}"] = s[stat].values
    return ZonalTables(pd.concat(long), wide, None)


def zonal_areas(zones: Any, rasters: Any, *, kind: str = "classes", names: Optional[Dict[int, str]] = None,
                bands: Any = None, labels: Optional[Sequence[str]] = None, layer: Optional[str] = None,
                id_field: Optional[str] = None, all_touched: bool = False,
                transitions: Optional[Tuple[int, int]] = None,
                window: Optional[Tuple[int, int, int, int]] = None) -> ZonalTables:
    """The pixels and area of each class (or year) of class maps in each zone of a layer, and the transitions.

    The zones and their pixels are those of ``zonal_stats``. Each pixel counts with its true area (see
    ``pixel_area``: geodesic on the ellipsoid, measured over the zones' extent), in the class of its value (the
    value rounded, halves away from zero) or, for ``kind="years"``, in its year (the value floored). Pixels with
    no data in a map are left out of it.

    Parameters
    ----------
    zones, rasters, bands, labels, layer, id_field, all_touched, window
        As in ``zonal_stats``.
    kind
        ``"classes"`` (default) or ``"years"`` (maps whose values are years).
    names
        Class names by value, used in the ``class`` column and in the wide table's ``ha_<name>`` columns.
    transitions
        ``(a, b)``, the positions of two maps: the area of every change of class from the first to the second.

    Returns
    -------
    ZonalTables
        ``long``: ``id``, ``fid``, ``date`` (the map's label), ``value``, ``class`` (or ``year``), ``pixels``,
        ``area_m2``, ``area_ha`` and ``percent`` of the zone's counted area, a row per zone, map and class;
        ``wide``: ``id``, ``fid``, ``pixels`` and an ``ha_<class>`` column per class (``_<label>`` after it with
        several maps); ``transitions``: ``id``, ``fid``, ``from``, ``from_class``, ``to``, ``to_class``,
        ``pixels``, ``area_m2``, ``area_ha``, or None.
    """
    if kind not in ("classes", "years"):
        raise ValueError('zonal_areas: kind is "classes" or "years"')
    names = dict(names or {})
    years = kind == "years"
    b = _burn_zones(zones, rasters, bands=bands, labels=labels, layer=layer, id_field=id_field,
                    all_touched=all_touched, window=window)
    area = _cell_areas(b)
    ids, zs, zone, flat, maps = b.ids, b.zones, b.zone, b.flat, b.maps
    n = len(zs)
    key = ((lambda v: np.floor(v).astype(np.int64)) if years else
           (lambda v: np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)))
    name = (lambda v: str(v)) if years else (lambda v: names.get(v, str(v)))

    def count(*layers):
        """The pixels and the area (m2) of each zone's cells by the keys of the maps' values."""
        x = [m.values.ravel()[flat] for m in layers]
        ok = np.all([np.isfinite(v) for v in x], axis=0) if x else np.ones(len(flat), bool)
        frame = pd.DataFrame({"zone": zone[ok], **{i: key(v[ok]) for i, v in enumerate(x)}, "area_m2": area[ok]})
        return frame.groupby(["zone", *range(len(x))]).area_m2.agg(["count", "sum"]).set_axis(
            ["pixels", "area_m2"], axis=1).reset_index()

    counts = [count(m) for m in maps]
    values = sorted({v for c in counts for v in c[0]})
    long: List[pd.DataFrame] = []
    wide = pd.DataFrame({"id": ids, "fid": zs.index, "pixels": [len(c) for c in b.cells]})
    for label, c in zip(b.labels, counts):
        c = c.rename(columns={0: "value"})
        c.insert(0, "id", ids[c.zone])
        c.insert(1, "fid", zs.index[c.zone])
        c.insert(2, "date", label)
        c.insert(5, "year" if years else "class", c.value.map(name))
        c["area_ha"], c["percent"] = c.area_m2 / 1e4, 100 * c.area_m2 / c.groupby("zone").area_m2.transform("sum")
        long.append(c.drop(columns="zone").round({"area_m2": 3, "area_ha": 6, "percent": 6}))
        ha = c.pivot_table(index="zone", columns="value", values="area_ha", aggfunc="sum").reindex(
            index=range(n), columns=values).fillna(0)
        suffix = "" if len(maps) == 1 else "_" + _column(label)
        ha.columns = [f"ha_{str(v) if years else _column(name(v))}" + suffix for v in values]
        wide = wide.join(ha.set_index(wide.index))
    move = None
    if transitions is not None:
        a, c_ = transitions
        move = count(maps[a], maps[c_]).rename(columns={0: "from", 1: "to"})
        move.insert(0, "id", ids[move.zone])
        move.insert(1, "fid", zs.index[move.zone])
        move.insert(4, "from_class", move["from"].map(name))
        move.insert(6, "to_class", move["to"].map(name))
        move["area_ha"] = move.area_m2 / 1e4
        move = move.drop(columns="zone").round({"area_m2": 3, "area_ha": 6})
    return ZonalTables(pd.concat(long) if long else pd.DataFrame(), wide, move)
