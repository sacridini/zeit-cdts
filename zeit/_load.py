"""``load_raster``: any time series zeit works with, as one georeferenced cube.

The cube is an ``xarray.DataArray`` with dims ``(time, y, x)`` (one index) or
``(time, band, y, x)`` (several spectral bands), a ``datetime64`` ``time``
coordinate and its CRS, transform and NoData through ``.rio``.
"""

import glob
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ._time import dates_from_labels, find_date, to_datetime_index

#: GDAL metadata tag where save_raster stores the dates of the bands.
TIME_TAG = "ZEIT_TIME"
RASTER_EXTENSIONS = (".tif", ".tiff", ".vrt", ".img", ".jp2", ".hdf", ".h5", ".asc")
_DIM_ALIASES = {"lat": "y", "latitude": "y", "lon": "x", "long": "x", "longitude": "x",
                "bands": "band", "variable": "band", "date": "time", "t": "time"}
_VALIDATE = {"landtrendr": 3, "ccdc": 12, "cold": 12}


def load_raster(
    source: Any,
    *,
    dates: Optional[Iterable[Any]] = None,
    start_year: Optional[int] = None,
    band: Union[int, str, Sequence[Union[int, str]], None] = None,
    chunks: Any = None,
    clip: Any = None,
    masked: Union[bool, str] = "auto",
    pattern: Optional[str] = None,
    date_format: Optional[str] = None,
    recursive: bool = False,
    like: Any = None,
    validate: Optional[str] = None,
) -> xr.DataArray:
    """Read a time series (or a single map) as a georeferenced cube.

    Parameters
    ----------
    source
        What to read:

        - a raster file (GeoTIFF or anything GDAL reads) with one band per date;
        - a folder, a glob pattern (``"tiles/*.tif"``) or a list of rasters with one
          date each, the date taken from the file name;
        - a Zarr store (``.zarr``) or a NetCDF file (``.nc``);
        - an ``xarray.DataArray`` or ``Dataset`` (e.g. from ``build_time_series``);
        - a numpy array ``(time, y, x)`` or ``(time, band, y, x)``, with ``dates`` or
          ``start_year`` and ``like`` for the georeferencing.
    dates
        Date of each time step: strings, datetimes, years (``1985``) or fractional
        years (``2020.5``). Overrides any date found in the data.
    start_year
        Annual series: the year of the first time step (``1985`` -> 1985, 1986, ...).
    band
        Bands to read. In a raster file: 1-based band numbers (``1`` or ``[1, 2]``);
        in a folder: the band of each file; in a cube with a ``band`` dim or a
        Dataset: band or variable names.
    chunks
        ``None`` (default) loads the data into memory; ``"auto"`` or a dict of chunk
        sizes returns a lazy dask-backed cube, read block by block.
    clip
        Read only a region: bounds ``(xmin, ymin, xmax, ymax)`` in the raster's CRS,
        or geometries (GeoDataFrame, GeoSeries, shapely or GeoJSON-like); cells outside
        the geometries become NoData.
    masked
        NoData handling. ``"auto"`` (default): NoData becomes NaN in floating-point
        rasters, while integer rasters (e.g. NDVI x 10000 in Int16) keep their values and
        type, with the NoData value in ``.rio.nodata``. ``True``: NaN in every raster
        (integers become floats). ``False``: values as stored.
    pattern
        Folder/glob/list only: regular expression with a named group ``date`` (and
        optionally ``band``) matched against each file name, e.g.
        ``r"_(?P<date>\\d{8})_(?P<band>B\\d{2})\\.tif$"``. Without it, the first date in
        the name is used.
    date_format
        ``strptime`` format of the dates in file names or band descriptions
        (e.g. ``"%Y%m%d"``). Without it the common layouts are recognised.
    recursive
        Folder only: also search sub-folders.
    like
        numpy input only: a raster (path or DataArray) on the same grid, whose
        coordinates and CRS georeference the array.
    validate
        ``"landtrendr"``, ``"ccdc"`` or ``"cold"``: warn when the series looks unfit for
        that algorithm (too few dates, values that do not look scaled).

    Returns
    -------
    xarray.DataArray
        ``(time, y, x)`` or ``(time, band, y, x)``. Without dates the band axis is
        named ``band`` and the algorithms ask for ``years``/``dates``. A single
        band is returned as ``(y, x)``.

    Notes
    -----
    The dates of a raster file come, in order, from ``dates``/``start_year``, from the
    dates ``save_raster`` writes into the file, from the band descriptions (``yr1985``,
    ``1985``, ``2020-01-15``, ``20200115``, or ``2020-01-15_red`` for a time x band
    stack) and from a ``<name>_dates.csv`` file next to it.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")     # bands yr1985 ... yr2024
    >>> ndvi.dims, ndvi.time.dt.year.values[[0, -1]]
    (('time', 'y', 'x'), array([1985, 2024]))
    >>> ndvi = zeit.load_raster("ndvi_stack.tif", start_year=1985, chunks="auto")
    >>> s2 = zeit.load_raster("S2/", pattern=r"_(?P<date>\\d{8})_(?P<band>B\\d{2})\\.tif$")
    """
    if masked not in ("auto", True, False):
        raise ValueError(f"masked must be 'auto', True or False, got {masked!r}")
    if validate is not None and validate.lower() not in _VALIDATE:
        raise ValueError(f"validate must be one of {sorted(_VALIDATE)}, got {validate!r}")
    if dates is not None and start_year is not None:
        raise ValueError("give dates or start_year, not both")

    if isinstance(source, np.ndarray) or (hasattr(source, "dask") and not isinstance(source, (xr.DataArray, xr.Dataset))):
        da = _from_array(source, like)
    elif isinstance(source, (xr.DataArray, xr.Dataset)):
        da = _from_xarray(source, band)
        band = None
    else:
        files = _expand(source, recursive)
        if len(files) == 1 and not _is_collection(source):
            da = _open_one(files[0], band=band, chunks=chunks, date_format=date_format)
        else:
            da = _open_many(files, band=band, chunks=chunks, pattern=pattern, date_format=date_format,
                            dates=dates, start_year=start_year)
        band = None

    if band is not None and "band" in da.dims:
        da = da.sel(band=band)
    da = _normalize(da, dates=dates, start_year=start_year)
    if clip is not None:
        da = _clip(da, clip)
    da = _mask(da, masked)
    if chunks is None and not isinstance(source, (xr.DataArray, xr.Dataset)):
        da = da.load()
    elif chunks is not None and da.chunks is None:
        da = da.chunk(chunks)
    if validate is not None:
        _validate(da, validate.lower())
    return da


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _is_collection(source: Any) -> bool:
    if isinstance(source, (list, tuple)):
        return True
    text = os.fspath(source)
    return os.path.isdir(text) and not _is_store(text) or any(c in text for c in "*?[")


def _is_store(path: str) -> bool:
    return path.rstrip("/\\").lower().endswith(".zarr")


def _expand(source: Any, recursive: bool) -> List[str]:
    if isinstance(source, (list, tuple)):
        files: List[str] = []
        for item in source:
            files.extend(_expand(item, recursive))
        return files
    path = os.path.expanduser(os.fspath(source))
    if _is_store(path) or os.path.isfile(path) or path.startswith(("/vsi", "http://", "https://", "s3://", "gs://")):
        return [path]
    if os.path.isdir(path):
        walk = Path(path).rglob("*") if recursive else Path(path).glob("*")
        found = sorted(str(p) for p in walk if p.suffix.lower() in RASTER_EXTENSIONS and p.is_file())
        if not found:
            raise FileNotFoundError(f"no raster files ({', '.join(RASTER_EXTENSIONS)}) in {path}")
        return found
    if any(c in path for c in "*?["):
        found = sorted(glob.glob(path, recursive=recursive))
        if not found:
            raise FileNotFoundError(f"no files match {path}")
        return found
    raise FileNotFoundError(f"file not found: {path}")


def _open_one(path: str, *, band: Any, chunks: Any, date_format: Optional[str]) -> xr.DataArray:
    lower = path.rstrip("/\\").lower()
    if _is_store(lower) or lower.endswith((".nc", ".nc4", ".netcdf")):
        engine = "zarr" if _is_store(lower) else None
        ds = xr.open_dataset(path, engine=engine, chunks=chunks if chunks is not None else {},
                             decode_coords="all")
        return _from_xarray(ds, band)

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)
        tag = src.tags().get(TIME_TAG)
    da = rioxarray.open_rasterio(path, masked=False, chunks=_file_chunks(chunks))
    da.attrs.pop("long_name", None)
    times: Optional[pd.DatetimeIndex] = None
    names: Optional[List[str]] = None
    if tag:
        times = pd.DatetimeIndex(json.loads(tag))
        if len(times) != da.sizes["band"] or not times.is_unique:
            times = None
    if times is None:
        if date_format is not None:
            found = [find_date(d, date_format) if d else None for d in descriptions]
            parsed = (pd.DatetimeIndex([f[0] for f in found]), None) if all(found) else None
        else:
            parsed = dates_from_labels(descriptions)
        if parsed is not None:
            times, names = parsed
    if times is None:
        times = _sidecar_dates(path, da.sizes["band"])
    if times is not None:
        labels = _time_labels(times, names)
        da = da.drop_vars("band").assign_coords(labels if names is not None else {"band": labels})
    elif all(descriptions) and len(set(descriptions)) == len(descriptions):
        da = da.assign_coords(band=descriptions)

    if band is not None:
        idx = [band] if np.isscalar(band) else list(band)
        if any(not isinstance(b, (int, np.integer)) for b in idx):
            raise ValueError(f"band of a raster file: 1-based band numbers, got {band!r}")
        da = da.isel(band=[int(b) - 1 for b in idx])
        if np.isscalar(band):
            da = da.squeeze("band", drop=False)
    da.attrs["source"] = path
    return da


def _file_chunks(chunks: Any, single: bool = False) -> Any:
    """Chunks given for the cube's dims, as the dims of a raster file: (band, y, x)."""
    if not isinstance(chunks, dict):
        return chunks
    out = {"band" if k in ("time", "band") else k: v for k, v in chunks.items() if k in ("time", "band", "y", "x")}
    if single:
        out.pop("band", None)
    return out


def _time_labels(times: pd.DatetimeIndex, names: Optional[List[str]]) -> pd.Index:
    """Band coordinate of a raster whose bands are dates; _normalize turns it into a
    time axis (a MultiIndex of (time, band) when the bands are date_band pairs)."""
    if names is None:
        return pd.Index(times, name="band")
    return _multi(pd.MultiIndex.from_arrays([times, names], names=("time", "spectral")))


def _multi(index: pd.MultiIndex) -> xr.Coordinates:
    return xr.Coordinates.from_pandas_multiindex(index, "band")


def _sidecar_dates(path: str, count: int) -> Optional[pd.DatetimeIndex]:
    csv = os.path.splitext(path)[0] + "_dates.csv"
    if not os.path.isfile(csv):
        return None
    try:
        table = pd.read_csv(csv)
        times = pd.DatetimeIndex(pd.to_datetime(table["Date"]))
    except (KeyError, ValueError):
        return None
    return times if len(times) == count else None


def _open_many(files: List[str], *, band: Any, chunks: Any, pattern: Optional[str],
               date_format: Optional[str], dates: Any, start_year: Optional[int]) -> xr.DataArray:
    import re

    regex = re.compile(pattern) if pattern else None
    records = []
    for path in files:
        name = os.path.basename(path.rstrip("/\\"))
        when, spectral = None, None
        if regex is not None:
            match = regex.search(name)
            if match is None:
                continue
            groups = match.groupdict()
            if "date" in groups:
                found = find_date(groups["date"], date_format)
                when = found[0] if found else None
            spectral = groups.get("band")
        else:
            found = find_date(os.path.splitext(name)[0], date_format)
            when = found[0] if found else None
        records.append((path, when, spectral))
    if not records:
        raise ValueError(f"no file name matches the pattern {pattern!r}")

    explicit = dates is not None or start_year is not None
    missing = [os.path.basename(p) for p, w, _ in records if w is None]
    if missing and not explicit:
        raise ValueError("no date found in the names of " + ", ".join(missing[:5])
                         + (" ..." if len(missing) > 5 else "")
                         + "; pass pattern= with a (?P<date>...) group, or dates=/start_year=")
    if explicit:
        records = [(p, None, s) for p, _, s in records]  # dates come from the arguments, in file order

    layers = []
    reference = None
    for path, when, spectral in records:
        layer = rioxarray.open_rasterio(path, masked=False, chunks=_file_chunks(chunks, single=True))
        if layer.sizes["band"] > 1:
            if band is None:
                raise ValueError(f"{os.path.basename(path)} has {layer.sizes['band']} bands; "
                                 "files of a series need one band each, or give band=N")
            layer = layer.isel(band=int(band) - 1)
        else:
            layer = layer.isel(band=0)
        layer = layer.drop_vars("band", errors="ignore")
        layer.attrs.pop("long_name", None)
        grid = (layer.shape, layer.rio.transform(), layer.rio.crs)
        if reference is None:
            reference = (grid, path)
        elif grid != reference[0]:
            raise ValueError(f"{os.path.basename(path)} is not on the grid of {os.path.basename(reference[1])}: "
                             "every file of a series needs the same CRS, size and transform")
        layers.append(layer)

    if explicit:
        cube = xr.concat(layers, dim="band", coords="minimal", compat="override", join="override")
        return cube
    times = pd.DatetimeIndex([w for _, w, _ in records])
    if any(s is not None for _, _, s in records):
        if any(s is None for _, _, s in records):
            raise ValueError("the pattern's band group matched only some files")
        index = pd.MultiIndex.from_arrays([times, [s for _, _, s in records]], names=("time", "spectral"))
        if not index.is_unique:
            raise ValueError("two files have the same date and band")
        cube = xr.concat(layers, dim="band", coords="minimal", compat="override", join="override")
        return cube.drop_vars("band", errors="ignore").assign_coords(_multi(index))
    if not times.is_unique:
        raise ValueError("two files have the same date; use pattern= with a band group for multi-band series")
    cube = xr.concat(layers, dim="band", coords="minimal", compat="override", join="override")
    return cube.assign_coords(band=pd.Index(times, name="band"))


def _from_xarray(obj: Union[xr.DataArray, xr.Dataset], band: Any) -> xr.DataArray:
    if isinstance(obj, xr.Dataset):
        names = [v for v in obj.data_vars if obj[v].ndim >= 2]
        if band is not None and not isinstance(band, (int, np.integer)):
            names = [band] if isinstance(band, str) else list(band)
        if not names:
            raise ValueError("the Dataset has no gridded variables")
        if len(names) == 1:
            da = obj[names[0]]
        else:
            da = obj[names].to_array("band")
        crs = obj.rio.crs if obj.rio.crs is not None else None
        if crs is not None and da.rio.crs is None:
            da = da.rio.write_crs(crs)
        return da
    return obj


def _from_array(array: Any, like: Any) -> xr.DataArray:
    dims = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}.get(array.ndim)
    if dims is None:
        raise ValueError(f"a numpy cube has 2, 3 or 4 dimensions, got shape {array.shape}")
    da = xr.DataArray(array, dims=dims)
    if like is not None:
        ref = load_raster(like, chunks="auto") if not isinstance(like, xr.DataArray) else like
        if (ref.sizes["y"], ref.sizes["x"]) != array.shape[-2:]:
            raise ValueError(f"like= has {ref.sizes['y']}x{ref.sizes['x']} cells, the array {array.shape[-2:]}")
        da = da.assign_coords(y=ref.y.values, x=ref.x.values)
        if ref.rio.crs is not None:
            da = da.rio.write_crs(ref.rio.crs)
        da = da.rio.write_transform(ref.rio.transform())
    return da


# ---------------------------------------------------------------------------
# Normalisation to (time, [band,] y, x)
# ---------------------------------------------------------------------------

def _normalize(da: xr.DataArray, *, dates: Any, start_year: Optional[int]) -> xr.DataArray:
    rename = {d: _DIM_ALIASES[d.lower()] for d in da.dims if d.lower() in _DIM_ALIASES
              and _DIM_ALIASES[d.lower()] not in da.dims}
    if rename:
        da = da.rename(rename)
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"a raster needs y and x dimensions, got {da.dims}")

    # A band axis of dates (or of (date, band) pairs) becomes the time axis.
    if "time" not in da.dims and "band" in da.dims:
        coord = da.indexes.get("band")
        if isinstance(coord, pd.MultiIndex) and list(coord.names) == ["time", "spectral"]:
            da = da.unstack("band").rename(spectral="band")
        elif isinstance(coord, pd.DatetimeIndex):
            da = da.rename(band="time")
        elif "spectral" in da.dims:
            da = da.rename(band="time", spectral="band")
        elif dates is None and start_year is None and coord is not None and coord.dtype == object:
            parsed = dates_from_labels(list(coord))
            if parsed is not None and parsed[1] is None:
                da = da.assign_coords(band=parsed[0]).rename(band="time")
    elif "spectral" in da.dims:
        da = da.rename(spectral="band")

    if dates is not None or start_year is not None:
        axis = "time" if "time" in da.dims else "band" if "band" in da.dims else None
        if axis is None:
            raise ValueError("dates/start_year given, but the data has a single time step (y, x)")
        n = da.sizes[axis]
        times = (to_datetime_index(dates) if dates is not None
                 else pd.DatetimeIndex([pd.Timestamp(int(start_year) + i, 1, 1) for i in range(n)]))
        if len(times) != n:
            raise ValueError(f"{len(times)} dates for {n} time steps")
        if axis == "band":
            da = da.drop_vars("band", errors="ignore").rename(band="time")
        da = da.assign_coords(time=times)
    elif "time" in da.dims and "time" in da.coords and not np.issubdtype(da.time.dtype, np.datetime64):
        da = da.assign_coords(time=to_datetime_index(da.time.values))

    if "time" in da.dims and "time" in da.coords:
        da = da.sortby("time")
    order = [d for d in ("time", "band", "y", "x") if d in da.dims]
    others = [d for d in da.dims if d not in order]
    da = da.transpose(*others, *order)

    # A single map: (y, x), keeping its date (if any) as a scalar coordinate.
    for dim in ("band", "time"):
        if dim in da.dims and da.sizes[dim] == 1 and da.ndim == 3:
            da = da.squeeze(dim, drop=not (dim == "time" and "time" in da.coords))
    if da.rio.crs is None:
        crs = da.attrs.get("crs")
        if crs is not None:
            try:
                da = da.rio.write_crs(crs)
            except Exception:  # noqa: BLE001 - an unreadable crs attribute just stays as it is
                pass
    return da


def _clip(da: xr.DataArray, region: Any) -> xr.DataArray:
    if isinstance(region, (tuple, list)) and len(region) == 4 and all(np.isscalar(v) for v in region):
        return da.rio.clip_box(*region)
    geoms = region
    crs = None
    if hasattr(region, "geometry") and hasattr(region, "crs"):  # GeoDataFrame / GeoSeries
        crs = region.crs
        geoms = list(region.geometry)
    elif hasattr(region, "__geo_interface__"):
        geoms = [region]
    return da.rio.clip(geoms, crs=crs if crs is not None else da.rio.crs, drop=True)


def _mask(da: xr.DataArray, masked: Union[bool, str]) -> xr.DataArray:
    if masked is False:
        return da
    nodata = da.rio.nodata
    is_float = np.issubdtype(da.dtype, np.floating)
    if masked == "auto" and not is_float:
        return da
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        return da
    if not is_float:
        da = da.astype(np.float64 if da.dtype.itemsize >= 4 else np.float32)
    out = da.where(da != nodata)
    return out.rio.write_nodata(nodata, encoded=True)


def _validate(da: xr.DataArray, algorithm: str) -> None:
    label = {"landtrendr": "LandTrendr", "ccdc": "CCDC/COLD", "cold": "CCDC/COLD"}[algorithm]
    steps = da.sizes.get("time", da.sizes.get("band", 1)) if da.ndim > 2 else 1
    if steps < _VALIDATE[algorithm]:
        kind = "an annual time series" if algorithm == "landtrendr" else "a dense time series"
        warnings.warn(f"{label}: expected {kind}, but only {steps} time steps were found.", stacklevel=3)
    if da.chunks is None and np.issubdtype(da.dtype, np.floating):
        values = da.values
        if values.size and np.isfinite(values).any() and np.nanmax(np.abs(values)) <= 10.0:
            warnings.warn(f"{label}: values look like unscaled floats; the algorithm expects values scaled "
                          "by a factor (e.g. 10000).", stacklevel=3)
