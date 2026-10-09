"""``save_raster``: write any map, stack or result of zeit so that GIS software and
``load_raster`` read it back as it was (georeferencing, NoData, band names, dates)."""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
import rasterio.shutil
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from rasterio.transform import Affine
from rasterio.windows import Window

from ._load import RASTER_EXTENSIONS, TIME_TAG

_XARRAY_EXTENSIONS = (".nc", ".nc4", ".zarr")


def save_raster(
    data: Any,
    path: Union[str, os.PathLike],
    *,
    like: Any = None,
    crs: Any = None,
    transform: Optional[Affine] = None,
    nodata: Optional[float] = None,
    dtype: Any = None,
    band_names: Optional[Sequence[str]] = None,
    compress: str = "deflate",
    driver: Optional[str] = None,
    reference_cube: Any = None,
) -> Path:
    """Write a map, a stack or a result of zeit.

    Parameters
    ----------
    data
        What to write:

        - a 2-D ``(y, x)``, 3-D ``(band|time, y, x)`` or 4-D ``(time, band, y, x)``
          ``xarray.DataArray``, georeferenced through ``.rio`` or its ``x``/``y``
          coordinates;
        - a numpy or dask array of the same shapes, georeferenced by ``like`` (or
          ``crs`` and ``transform``);
        - an ``xarray.Dataset`` or a ``dict`` of maps, e.g. the result of
          ``extract_events``: ``path`` without an extension becomes a folder with one
          GeoTIFF per variable; a ``.tif`` path one multi-band file named by variable.
    path
        Output file (or folder, see above). The format follows the extension
        (``.tif``, ``.img``, ``.nc``, ``.zarr``...); without one, ``.tif`` is added.
        Missing folders are created.
    like
        A raster (path or DataArray) on the same grid whose CRS and transform
        georeference ``data``, e.g. a numpy result computed from a loaded cube.
    crs, transform
        Coordinate reference system and affine transform; they override those of
        ``data`` and ``like``.
    nodata
        NoData value; overrides the one of ``data``. NaN in floating-point data are
        written as this value. Floats with NaN and no NoData get NaN as NoData.
    dtype
        Data type to write (default: the data's). ``bool`` is written as ``uint8`` and
        64-bit integers as the smallest integer type that holds them. Floats cast to
        integers get ``nodata`` (or the type's maximum) where they are NaN.
    band_names
        Band descriptions. By default: the dates of a ``time`` axis (``1985`` for annual
        series, ``2020-01-15`` otherwise, ``2020-01-15_red`` for a time x band cube), or
        the labels of a ``band`` axis.
    compress
        GeoTIFF compression: ``"deflate"`` (default), ``"lzw"``, ``"zstd"``, ``None``...
    driver
        GDAL driver, when the extension is not enough; ``"COG"`` writes a
        Cloud-Optimised GeoTIFF.
    reference_cube
        Deprecated name of ``like``.

    Returns
    -------
    pathlib.Path
        The file (or folder) written.

    Notes
    -----
    The dates of a ``time`` axis are also stored in the ``ZEIT_TIME`` metadata tag,
    so ``load_raster`` reads the file back with its ``time`` coordinate. Large GeoTIFFs
    are tiled, compressed with the predictor that suits the type and become BigTIFF when
    needed; dask arrays are written block by block, never whole in memory.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
    >>> zeit.save_raster(ndvi.sel(time="2020"), "ndvi_2020.tif")
    >>> zeit.save_raster(events, "lt_rondonia")            # dict/Dataset: one file per map
    >>> zeit.save_raster(numpy_map, "map.tif", like=ndvi)  # numpy: georeferenced by like
    """
    if reference_cube is not None:
        warnings.warn("save_raster(reference_cube=) is deprecated; use like=", DeprecationWarning, stacklevel=2)
        like = reference_cube if like is None else like
    path = Path(os.path.expanduser(os.fspath(path)))
    options = dict(like=like, crs=crs, transform=transform, nodata=nodata, dtype=dtype,
                   compress=compress, driver=driver)

    if isinstance(data, (xr.Dataset, Mapping)):
        layers = _layers(data)
        if path.suffix.lower() in _XARRAY_EXTENSIONS:
            return _write_xarray(xr.Dataset({k: _as_dataarray(v) for k, v in layers.items()}), path)
        if path.suffix:
            stacked, names = _stack_layers(layers)
            return _write(stacked, path, band_names=band_names or names, **options)
        path.mkdir(parents=True, exist_ok=True)
        for name, layer in layers.items():
            _write(layer, path / f"{name}.tif", band_names=None, **options)
        return path

    if not path.suffix:
        path = path.with_suffix(".tif")
    if path.suffix.lower() in _XARRAY_EXTENSIONS:
        return _write_xarray(_as_dataarray(data), path)
    return _write(data, path, band_names=band_names, **options)


# ---------------------------------------------------------------------------
# Datasets and dicts
# ---------------------------------------------------------------------------

def _layers(data: Union[xr.Dataset, Mapping]) -> Dict[str, Any]:
    """The gridded variables of a Dataset or dict, in order."""
    items = data.data_vars.items() if isinstance(data, xr.Dataset) else data.items()
    layers = {}
    for name, value in items:
        if value is None or np.ndim(value) < 2:
            continue
        layers[str(name)] = value
    if not layers:
        raise ValueError("nothing to write: no 2-D or 3-D maps in the data")
    crs = data.rio.crs if isinstance(data, xr.Dataset) else None
    if crs is not None:
        layers = {k: (v.rio.write_crs(crs) if isinstance(v, xr.DataArray) and v.rio.crs is None else v)
                  for k, v in layers.items()}
    return layers


def _stack_layers(layers: Dict[str, Any]) -> Tuple[Any, List[str]]:
    """Maps of a result as one (band, y, x) stack named by variable."""
    arrays, names = [], []
    for name, layer in layers.items():
        if np.ndim(layer) == 2:
            arrays.append(layer)
            names.append(name)
        else:
            n = layer.shape[0]
            for i in range(n):
                arrays.append(layer[i])
                names.append(f"{name}_{i + 1}")
    shapes = {tuple(np.shape(a)) for a in arrays}
    if len(shapes) > 1:
        raise ValueError(f"the maps have different shapes {sorted(shapes)}; write them to a folder instead")
    common = np.result_type(*[np.asarray(a).dtype if not isinstance(a, xr.DataArray) else a.dtype for a in arrays])
    first = next((a for a in arrays if isinstance(a, xr.DataArray)), None)
    if first is not None:
        stacked = xr.concat([(a if isinstance(a, xr.DataArray) else first.copy(data=np.asarray(a))).astype(common)
                             .drop_vars([c for c in first.coords if c not in ("x", "y", "spatial_ref")], errors="ignore")
                             for a in arrays], dim="band", coords="minimal", compat="override")
        stacked = stacked.drop_vars("band", errors="ignore")
        if first.rio.crs is not None and stacked.rio.crs is None:
            stacked = stacked.rio.write_crs(first.rio.crs)
    else:
        stacked = np.stack([np.asarray(a).astype(common) for a in arrays])
    return stacked, names


def _as_dataarray(data: Any) -> xr.DataArray:
    if isinstance(data, xr.DataArray):
        return data
    dims = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}[np.ndim(data)]
    return xr.DataArray(data, dims=dims)


def _write_xarray(obj: Union[xr.DataArray, xr.Dataset], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = obj.to_dataset(name=obj.name or "data") if isinstance(obj, xr.DataArray) else obj
    if path.suffix.lower() == ".zarr":
        ds.to_zarr(path, mode="w")
    else:
        ds.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# One raster
# ---------------------------------------------------------------------------

def _write(data: Any, path: Path, *, like: Any, crs: Any, transform: Optional[Affine], nodata: Optional[float],
           dtype: Any, band_names: Optional[Sequence[str]], compress: Optional[str], driver: Optional[str]) -> Path:
    da = data if isinstance(data, xr.DataArray) else None
    if da is not None:
        da = _spatial_last(da)
    ndim = np.ndim(data)
    if ndim not in (2, 3, 4):
        raise ValueError(f"a raster has 2, 3 or 4 dimensions, got shape {np.shape(data)}")

    # Band names and dates, before the cube is flattened to (band, y, x).
    names, times = _band_labels(da)
    if band_names is not None:
        names = list(band_names)
    array = da.data if da is not None else data
    if ndim == 4:
        array = array.reshape((array.shape[0] * array.shape[1],) + tuple(array.shape[2:]))
    elif ndim == 2:
        array = array[None, ...]
    count, height, width = array.shape
    if names is not None and len(names) != count:
        raise ValueError(f"{len(names)} band names for {count} bands")

    crs_, transform_ = _georef(da, like, (height, width))
    crs_ = crs if crs is not None else crs_
    transform_ = transform if transform is not None else transform_
    if transform_ is None:
        warnings.warn(f"{path.name}: no georeferencing (pass like=, or crs= and transform=); "
                      "the raster is written in pixel coordinates.", UserWarning, stacklevel=3)

    if nodata is None and da is not None:
        nodata = da.rio.encoded_nodata if da.rio.encoded_nodata is not None else da.rio.nodata
    out_dtype, nodata = _out_dtype(array, dtype, nodata)

    path.parent.mkdir(parents=True, exist_ok=True)
    drv = driver or _driver_for(path)
    final_driver = drv
    target = path
    if drv.upper() == "COG":  # COG is a copy-only driver: write a GeoTIFF, then copy it
        drv = "GTiff"
        target = path.with_name(path.stem + ".__tmp__.tif")
    profile = dict(driver=drv, height=height, width=width, count=count, dtype=out_dtype)
    if crs_ is not None:
        profile["crs"] = crs_
    if transform_ is not None:
        profile["transform"] = transform_
    if nodata is not None:
        profile["nodata"] = nodata
    if drv == "GTiff":
        profile.update(_gtiff_options(out_dtype, compress, height, width))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(target, "w", **profile) as dst:
            for window, block in _blocks(array, height):
                dst.write(_cast(block, out_dtype, nodata), window=window)
            if names is not None:
                for i, name in enumerate(names, start=1):
                    dst.set_band_description(i, str(name))
            if times is not None and ndim == 3:
                dst.update_tags(**{TIME_TAG: json.dumps([t.isoformat() for t in times])})
    if final_driver.upper() == "COG":
        rasterio.shutil.copy(target, path, driver="COG", compress=compress or "NONE")
        rasterio.shutil.delete(target)
    return path


def _spatial_last(da: xr.DataArray) -> xr.DataArray:
    rename = {d: n for d, n in (("lat", "y"), ("latitude", "y"), ("lon", "x"), ("longitude", "x"))
              if d in da.dims and n not in da.dims}
    if rename:
        da = da.rename(rename)
    if "y" in da.dims and "x" in da.dims:
        lead = [d for d in ("time", "band") if d in da.dims]
        rest = [d for d in da.dims if d not in (*lead, "y", "x")]
        da = da.transpose(*rest, *lead, "y", "x")
    return da


def _date_label(times: pd.DatetimeIndex) -> List[str]:
    if (times.month == 1).all() and (times.day == 1).all() and (times.normalize() == times).all():
        return [str(t.year) for t in times]
    return [t.strftime("%Y-%m-%d") for t in times]


def _band_labels(da: Optional[xr.DataArray]) -> Tuple[Optional[List[str]], Optional[pd.DatetimeIndex]]:
    """Band descriptions (and the dates of a time axis) of a DataArray."""
    if da is None or da.ndim == 2:
        return None, None
    lead = da.dims[:-2]
    has_time = "time" in lead and "time" in da.coords and np.issubdtype(da.time.dtype, np.datetime64)
    times = pd.DatetimeIndex(da.time.values) if has_time else None
    if da.ndim == 4:
        first = _date_label(times) if times is not None else [str(v) for v in _coord(da, lead[0])]
        second = [str(v) for v in _coord(da, lead[1])]
        return [f"{a}_{b}" for a in first for b in second], None
    if times is not None:
        return _date_label(times), times
    values = _coord(da, lead[0])
    if values is not None and values.dtype.kind in "OU":
        return [str(v) for v in values], None
    return None, None


def _coord(da: xr.DataArray, dim: str) -> Optional[np.ndarray]:
    return da[dim].values if dim in da.coords else np.arange(1, da.sizes[dim] + 1)


def _georef(da: Optional[xr.DataArray], like: Any, shape: Tuple[int, int]) -> Tuple[Any, Optional[Affine]]:
    crs, transform = None, None
    if da is not None:
        crs, transform = _georef_of(da)
    if like is not None and (crs is None or transform is None):
        if isinstance(like, xr.DataArray):
            ref_crs, ref_transform = _georef_of(like)
            ref_shape = (like.sizes.get("y"), like.sizes.get("x"))
        else:
            with rasterio.open(os.path.expanduser(os.fspath(like))) as src:
                ref_crs, ref_transform, ref_shape = src.crs, src.transform, (src.height, src.width)
        if tuple(ref_shape) != tuple(shape):
            raise ValueError(f"like= is {ref_shape[0]}x{ref_shape[1]} cells, the data {shape[0]}x{shape[1]}")
        crs = crs if crs is not None else ref_crs
        transform = transform if transform is not None else ref_transform
    return crs, transform


def _georef_of(da: xr.DataArray) -> Tuple[Any, Optional[Affine]]:
    crs = da.rio.crs
    if crs is None and da.attrs.get("crs") is not None:  # e.g. stackstac cubes
        crs = da.attrs["crs"]
    transform = None
    if "x" in da.coords and "y" in da.coords and da.sizes.get("x", 0) > 1 and da.sizes.get("y", 0) > 1:
        transform = da.rio.transform(recalc=True)
        # The transform stored with the data is exact; the one recomputed from the cell
        # centres carries float noise. Keep the stored one when both agree (it is stale
        # after a plain .isel, which the recomputed one catches).
        stored = da.rio.transform()
        if np.allclose(tuple(stored)[:6], tuple(transform)[:6], rtol=1e-9, atol=1e-9 * max(abs(transform.a), 1e-12)):
            transform = stored
    elif da.rio.crs is not None or "spatial_ref" in da.coords:
        candidate = da.rio.transform()
        transform = None if candidate == Affine.identity() else candidate
    return crs, transform


def _driver_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff", ""):
        return "GTiff"
    try:
        from rasterio.drivers import driver_from_extension
        return driver_from_extension(path)
    except (ImportError, ValueError):
        if ext in RASTER_EXTENSIONS:
            return "GTiff"
        raise ValueError(f"no GDAL driver for {ext!r}; pass driver=")


def _gtiff_options(dtype: str, compress: Optional[str], height: int, width: int) -> Dict[str, Any]:
    options: Dict[str, Any] = {"BIGTIFF": "IF_SAFER"}
    if compress:
        options["compress"] = compress
        if compress.lower() in ("deflate", "lzw", "zstd"):
            options["predictor"] = 3 if np.dtype(dtype).kind == "f" else 2
    if height > 256 or width > 256:
        options.update(tiled=True, blockxsize=256, blockysize=256)
    return options


def _out_dtype(array: Any, dtype: Any, nodata: Optional[float]) -> Tuple[str, Optional[float]]:
    src = np.dtype(array.dtype)
    if dtype is not None:
        out = np.dtype(dtype)
    elif src == bool:
        out = np.dtype("uint8")
    elif src.kind in "iu" and src.itemsize == 8:
        low, high = (int(v) for v in (np.min(array), np.max(array))) if array.size else (0, 0)
        if nodata is not None and np.isfinite(nodata):
            low, high = min(low, int(nodata)), max(high, int(nodata))
        out = np.result_type(np.min_scalar_type(low), np.min_scalar_type(high))
        if out.itemsize < 1 or out.kind not in "iu":
            out = np.dtype("int64")
    elif src == np.float16:
        out = np.dtype("float32")
    else:
        out = src

    if out.kind == "f":
        if nodata is None and _has_nan(array):
            nodata = float("nan")
    elif src.kind == "f" and nodata is None and _has_nan(array):
        nodata = float(np.iinfo(out).max)
    if nodata is not None and out.kind in "iu" and np.isnan(nodata):
        raise ValueError(f"nodata=NaN cannot be written in an integer raster ({out.name})")
    return out.name, nodata


def _has_nan(array: Any) -> bool:
    if np.dtype(array.dtype).kind != "f":
        return False
    found = np.isnan(array).any()
    return bool(found.compute() if hasattr(found, "compute") else found)


def _cast(block: np.ndarray, dtype: str, nodata: Optional[float]) -> np.ndarray:
    block = np.asarray(block)
    if block.dtype.kind == "f" and nodata is not None and not np.isnan(nodata):
        block = np.where(np.isnan(block), nodata, block)
    return block.astype(dtype, copy=False)


def _blocks(array: Any, height: int):
    """(window, block) pairs: the whole array, or row strips of a dask array."""
    if not hasattr(array, "chunks") or not hasattr(array, "compute") or array.chunks is None:
        yield None, np.asarray(array)
        return
    row = 0
    for size in array.chunks[-2]:
        strip = array[..., row:row + size, :].compute()
        yield Window(0, row, array.shape[-1], size), strip
        row += size
