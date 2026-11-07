"""``load_raster``: any time series zeit works with, as one georeferenced cube.

The cube is an ``xarray.DataArray`` with dims ``(time, y, x)`` (one index) or
``(time, band, y, x)`` (several spectral bands), a ``datetime64`` ``time``
coordinate and its CRS, transform and NoData through ``.rio``.
"""

import atexit
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
from .qc import QA_RULES, qa_mask

#: GDAL metadata tag where save_raster stores the dates of the bands.
TIME_TAG = "ZEIT_TIME"
RASTER_EXTENSIONS = (".tif", ".tiff", ".vrt", ".img", ".jp2", ".hdf", ".h5", ".asc")
_DIM_ALIASES = {"lat": "y", "latitude": "y", "lon": "x", "long": "x", "longitude": "x",
                "bands": "band", "variable": "band", "date": "time", "t": "time"}
_VALIDATE = {"landtrendr": 3, "ccdc": 12, "cold": 12}


def _close_files() -> None:
    """Close the files xarray keeps open for lazy rasters (load_raster(..., chunks=)) before the
    interpreter shuts down, which otherwise ends with an "Error in sys.excepthook" message."""
    try:
        from xarray.backends.file_manager import FILE_CACHE
        FILE_CACHE.clear()
    except Exception:  # noqa: BLE001 - at exit, nothing to report
        pass


atexit.register(_close_files)


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
    crs: Any = None,
    res: Any = None,
    resampling: str = "auto",
    validate: Optional[str] = None,
    scale: bool = False,
    qa: Any = None,
    qa_rule: Optional[str] = None,
    bands: Optional[Dict[str, Union[int, str]]] = None,
    window: Optional[Tuple[int, int, int, int]] = None,
    pixel: Optional[Tuple[int, int]] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    tap: bool = False,
    nodata: Any = None,
    dtype: Any = None,
    cutline: Any = None,
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
        A reference raster (path, ``DataArray`` or ``Dataset``) whose grid the result
        takes: its CRS, cells, extent and ``x``/``y`` coordinates, so that the two line
        up cell by cell. The data is reprojected and resampled onto it (each date on its
        own, with its own NoData); cells outside the data are NoData. A file is read
        only in the window the grid sees, and a lazy cube stays lazy. Folders of
        rasters on different grids are put on it file by file. With a numpy array:
        a raster on the same grid, whose coordinates and CRS georeference the array.
    crs
        Without a reference raster: the CRS to reproject to (``"EPSG:4326"``, ``32722``,
        WKT or a ``pyproj``/rasterio CRS), as ``gdalwarp -t_srs`` does. The extent is the
        one that covers the data in that CRS (all the files of a folder), and the
        resolution GDAL's suggestion for it unless ``res`` is given.
    res
        Without a reference raster: the cell size, a number or ``(x, y)``, in the units of
        ``crs`` (or of the data's CRS), as ``gdalwarp -tr``: cells laid from the top-left
        corner of the extent. Use ``like`` to line up with an existing grid.
    resampling
        With ``like``, ``crs`` or ``res``: ``"auto"`` (default) takes the nearest cell for integer rasters
        (classes, QA flags, indices scaled to integers keep values that were observed)
        and interpolates floats bilinearly; QA bands (``qa``, ``fmask``, ``scl``...)
        always take the nearest cell. Or a GDAL method: ``"nearest"``, ``"bilinear"``,
        ``"cubic"``, ``"cubic_spline"``, ``"lanczos"``, ``"average"``, ``"mode"``,
        ``"min"``, ``"max"``, ``"med"``, ``"q1"``, ``"q3"``, ``"rms"``.
    validate
        ``"landtrendr"``, ``"ccdc"`` or ``"cold"``: warn when the series looks unfit for
        that algorithm (too few dates, values that do not look scaled).
    scale
        ``True``: apply the ``scale_factor`` and ``add_offset`` the raster files store
        for each band (GDAL's scale and offset, ``value * scale + offset``), after NoData
        became NaN. Bands without them are left as they are, so a file with none keeps its
        type. Files of a folder each use their own. Ignored for NetCDF/Zarr stores and
        ``DataArray`` sources, which are already decoded. Needs ``masked`` ``"auto"`` or
        ``True``.
    qa
        A quality band, so that the series comes back masked (NaN where the observation is
        not usable, by ``qa_rule``): a band number of the same files (``qa=7`` for Landsat
        files with the bands of a date in one file, read like ``band=7``), or another
        source (a raster, folder, glob, list or ``DataArray``) with the same dates and
        grid, read with the same ``clip``, ``window``/``pixel``, ``like``... as the data.
        The quality band is read with the nearest cell. Integer data becomes float.
    qa_rule
        With ``qa``: how the quality band marks usable observations, one of ``"fmask"``,
        ``"landsat_c2"``, ``"nonzero"``, ``"scl"`` or ``"hls_fmask"`` (see `zeit.qa_mask`).
    bands
        Bands by role, ``{"blue": 1, "green": 2, "red": 3, "nir": 4}``: name -> band (the
        number of the band in each file, as ``band=``). Returns a ``(time, band, y, x)``
        cube whose ``band`` coordinate holds the names, in the order of the dict. Give
        ``band`` or ``bands``, not both.
    window
        Read only a window of pixels, ``(col0, row0, col1, row1)``: the first column and row
        and the end (exclusive) of the raster's own grid, with no need for a CRS. The result
        keeps the cells' coordinates and transform. A file is read in the window only. Not
        with ``clip``, ``like``, ``crs`` or ``res``.
    pixel
        Read one pixel, ``(col, row)`` (``x`` first, as in ``window``), as a ``(time,)`` series
        (``(time, band)`` with several bands) with the pixel's ``x`` and ``y`` as scalar
        coordinates. Only that pixel is read from each file, so it is quick in a stack of many
        gigabytes. The same restrictions as ``window``.
    extent
        Without a reference raster: the extent of the grid, ``(xmin, ymin, xmax, ymax)`` in the
        CRS of the result (``crs`` or the data's), as ``gdalwarp -te``. With ``res`` the cells of
        that size are laid from its top-left corner; without it, the cells of the data (same CRS
        only), fitted to the extent as gdalwarp does: a whole number of them, a little larger or
        smaller. With ``crs`` and ``extent``, ``res`` is needed. An extent that is a whole number
        of cells apart from the data's crops it with no resampling.
    tap
        ``True``: the extent widened to whole cells counted from the origin of the coordinates, as
        ``gdalwarp -tap``. Needs ``res``.
    nodata
        What the cells no data covers (outside the area, or the ``cutline``) hold, and the result's
        NoData. ``None`` (default): the data's NoData, else NaN for floats and the type's extreme
        for integers (the result's NoData). ``"source"``: as gdalwarp without ``-dstnodata``: the
        data's NoData, and when it has none those cells are 0 and the result has no NoData (so
        its zeros are values). A number: ``gdalwarp -dstnodata``; the data's own NoData cells
        become it, and a value equal to it is moved one step. An error where it does not fit
        ``dtype``.
    dtype
        The type of the result (``"float32"``, ``"int16"``, ``np.uint8``...), as ``gdalwarp -ot``:
        the warp is done in the wider of the data's type and this one, then rounded (half away
        from zero) and clamped to it. Without ``nodata`` the data's NoData is kept, or the type's
        default.
    cutline
        Cells whose centre is outside the polygons get the ``nodata`` fill, as ``gdalwarp -cutline``:
        a vector file (``"zones.gpkg"``, or ``("zones.gpkg", "layer")``), a GeoDataFrame or
        GeoSeries, shapely geometries or GeoJSON-like objects (in the CRS of the result when they
        have none). Without ``like``, the grid is also cropped to them, as ``-crop_to_cutline``: their
        extent (so ``extent`` is not allowed), and without ``res`` the cells of the data entirely
        inside it (same CRS only).

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
    stack) and from a ``<name>_dates.csv`` file next to it (its ``Date`` column). Dates
    in the tag or the CSV that are fewer than the bands but divide them are one for
    every few bands, date after date: 125 dates for 875 bands give a ``(time, band,
    y, x)`` cube of 7 bands, named as the file names them or 1 to 7.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")     # bands yr1985 ... yr2024
    >>> ndvi.dims, ndvi.time.dt.year.values[[0, -1]]
    (('time', 'y', 'x'), array([1985, 2024]))
    >>> ndvi = zeit.load_raster("ndvi_stack.tif", start_year=1985, chunks="auto")
    >>> s2 = zeit.load_raster("S2/", pattern=r"_(?P<date>\\d{8})_(?P<band>B\\d{2})\\.tif$")

    On the grid of another raster (e.g. Landsat at 30 m onto a Sentinel-2 grid at 10 m):

    >>> s2 = zeit.load_raster("s2_ndvi.tif")
    >>> landsat = zeit.load_raster("landsat_ndvi.tif", like=s2)
    >>> bool((landsat.x == s2.x).all())
    True

    In another CRS or at another resolution, without a reference raster:

    >>> albers = zeit.load_raster("ndvi_stack.tif", crs="EPSG:5880", res=30)
    >>> coarse = zeit.load_raster("ndvi_stack.tif", res=250, resampling="average")

    Surface reflectance by role, scaled and masked by its quality band, then an index:

    >>> cube = zeit.load_raster("landsat/", bands={"red": 3, "nir": 4}, qa=7, qa_rule="fmask", scale=True)
    >>> ndvi = zeit.normalized_difference(cube, "nir", "red")

    One pixel of a large stack, or a window of its pixels:

    >>> series = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif", pixel=(120, 80))
    >>> block = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif", window=(100, 60, 140, 100))
    """
    if masked not in ("auto", True, False):
        raise ValueError(f"masked must be 'auto', True or False, got {masked!r}")
    if validate is not None and validate.lower() not in _VALIDATE:
        raise ValueError(f"validate must be one of {sorted(_VALIDATE)}, got {validate!r}")
    if scale and masked is False:
        raise ValueError("scale=True needs NoData as NaN (masked='auto' or True): "
                         "the scale would change the NoData value")
    rule = None
    if qa is not None:
        if qa_rule is None:
            raise ValueError(f"qa= needs qa_rule= (one of {', '.join(QA_RULES)})")
        rule = str(qa_rule).lower().replace("-", "_")
        if rule not in QA_RULES:
            raise ValueError(f"unknown qa_rule {qa_rule!r}; use one of {', '.join(QA_RULES)}")
    elif qa_rule is not None:
        raise ValueError("qa_rule= needs qa= (the quality band)")
    win, point = _parse_window(window, pixel)
    if win is not None and (clip is not None or like is not None or crs is not None or res is not None
                            or extent is not None or tap or cutline is not None):
        raise ValueError("window=/pixel= read the raster's own pixels: not with clip=, like=, crs=, res=, extent=, "
                         "tap= or cutline=")
    if nodata is not None and not isinstance(nodata, str):
        try:
            nodata = float(nodata)
        except (TypeError, ValueError):
            raise ValueError(f"nodata must be None, 'source' or a number, got {nodata!r}") from None
    elif isinstance(nodata, str) and nodata != "source":
        raise ValueError(f"nodata must be None, 'source' or a number, got {nodata!r}")
    if dtype is not None:
        try:
            dtype = np.dtype(dtype)
        except TypeError:
            raise ValueError(f"dtype must be a numpy type, got {dtype!r}") from None
        if not (np.issubdtype(dtype, np.number) or dtype == np.bool_) or np.issubdtype(dtype, np.complexfloating):
            raise ValueError(f"dtype must be a real number type, got {dtype}")
    if extent is not None:
        try:
            extent = tuple(float(v) for v in extent)
        except (TypeError, ValueError):
            extent = ()
        if len(extent) != 4:
            raise ValueError("extent= takes (xmin, ymin, xmax, ymax)")
    if cutline is not None:
        from ._warp import cutline_of
        cutline = cutline_of(cutline)
        if extent is not None:
            raise ValueError("give extent= or cutline=, not both: gdalwarp crops to the cutline and ignores -te")
    common = dict(dates=dates, start_year=start_year, chunks=chunks, clip=clip, masked=masked, pattern=pattern,
                  date_format=date_format, recursive=recursive, like=like, crs=crs, res=res,
                  resampling=resampling, scale=scale, window=win, point=point, extent=extent, tap=bool(tap),
                  nodata=nodata, dtype=dtype, cutline=cutline)
    if bands is not None:
        if band is not None:
            raise ValueError("give band= or bands=, not both")
        da = _read_bands(source, bands, common)
    else:
        da = _read(source, band=band, **common)
    if qa is not None:
        da = _apply_qa(da, source, qa, rule, common)
    if validate is not None:
        _validate(da, validate.lower())
    return da


def _read(
    source: Any,
    *,
    dates: Any,
    start_year: Optional[int],
    band: Any,
    chunks: Any,
    clip: Any,
    masked: Union[bool, str],
    pattern: Optional[str],
    date_format: Optional[str],
    recursive: bool,
    like: Any,
    crs: Any,
    res: Any,
    resampling: str,
    scale: bool,
    window: Optional[Tuple[int, int, int, int]],
    point: bool,
    extent: Any = None,
    tap: bool = False,
    nodata: Any = None,
    dtype: Any = None,
    cutline: Any = None,
) -> xr.DataArray:
    """``load_raster`` without ``bands``, ``qa`` and ``validate``."""
    if dates is not None and start_year is not None:
        raise ValueError("give dates or start_year, not both")
    from ._warp import RESAMPLING_NAMES
    if resampling != "auto" and resampling not in RESAMPLING_NAMES:
        raise ValueError(f"unknown resampling {resampling!r}; use 'auto' or one of {', '.join(RESAMPLING_NAMES)}")
    is_array = isinstance(source, np.ndarray) or (hasattr(source, "dask")
                                                  and not isinstance(source, (xr.DataArray, xr.Dataset)))
    target = crs is not None or res is not None or extent is not None or tap
    if target and like is not None:
        raise ValueError("give like= or crs=/res=/extent=/tap=, not both: like= already sets the CRS and the cells")
    if (target or nodata is not None or dtype is not None or cutline is not None) and is_array:
        raise ValueError("crs=/res=/extent=/tap=/nodata=/dtype=/cutline= need a georeferenced source; a numpy array "
                         "has no grid (georeference it with like= first)")
    # nodata=, dtype= and cutline= without another grid: the data's own grid, converted
    target = target or (like is None and (nodata is not None or dtype is not None or cutline is not None))
    warp_kw = dict(nodata=nodata, dtype=dtype, cutline=cutline)
    grid = grid_of(like) if like is not None and not is_array else None
    early = False  # the window is already cut (a raster file is read in the window only)

    if is_array:
        da = _from_array(source, like)
    elif isinstance(source, (xr.DataArray, xr.Dataset)):
        da = _from_xarray(source, band)
        band = None
    else:
        files = _expand(source, recursive)
        if len(files) == 1 and not _is_collection(source):
            da = _open_one(files[0], band=band, chunks=chunks, date_format=date_format, scale=scale,
                           window=window)
            early = not _is_xarray_file(files[0])
        else:
            da = _open_many(files, band=band, chunks=chunks, pattern=pattern, date_format=date_format,
                            dates=dates, start_year=start_year, grid=grid, resampling=resampling,
                            target=(crs, res, extent, tap, cutline) if target else None, scale=scale,
                            window=window, warp_kw=warp_kw)
            early = True
            target = False  # already on the grid: the folder's extent is that of all its files
        band = None

    if band is not None and "band" in da.dims:
        da = da.sel(band=band)
    da = _normalize(da, dates=dates, start_year=start_year)
    if window is not None and not early:
        da = _cut(da, window)
    if target:
        grid = _target_grid(da, crs, res, extent, tap, cutline)
    if grid is not None:
        from ._warp import to_grid
        da = to_grid(da, grid, resampling, **warp_kw)
    if clip is not None:
        da = _clip(da, clip)
    da = _mask(da, masked)
    if point:
        da = da.isel(y=0, x=0)
    if chunks is None and not isinstance(source, (xr.DataArray, xr.Dataset)):
        da = da.load()
    elif chunks is not None and da.chunks is None:
        da = da.chunk(chunks)
    return da


# ---------------------------------------------------------------------------
# Windows, scale and offset, bands by role, quality bands
# ---------------------------------------------------------------------------

def _parse_window(window: Any, pixel: Any) -> Tuple[Optional[Tuple[int, int, int, int]], bool]:
    """``(window, point)``: the window ``(col0, row0, col1, row1)`` to read (``pixel=(col, row)`` is the window
    of one cell) and whether the result is a single pixel."""
    if window is not None and pixel is not None:
        raise ValueError("give window= or pixel=, not both")

    def whole(values: Any, n: int, name: str) -> Tuple[int, ...]:
        try:
            items = tuple(values)
            out = tuple(int(v) for v in items)
        except (TypeError, ValueError):
            raise ValueError(f"{name} takes {n} whole numbers, got {values!r}") from None
        if len(out) != n or any(a != b for a, b in zip(out, items)) or any(v < 0 for v in out):
            raise ValueError(f"{name} takes {n} whole numbers of 0 or more, got {values!r}")
        return out

    if pixel is not None:
        col, row = whole(pixel, 2, "pixel= (col, row)")
        return (col, row, col + 1, row + 1), True
    if window is not None:
        col0, row0, col1, row1 = whole(window, 4, "window= (col0, row0, col1, row1)")
        if col1 <= col0 or row1 <= row0:
            raise ValueError(f"window= is empty: {tuple(window)!r} (the end is exclusive)")
        return (col0, row0, col1, row1), False
    return None, False


def _cut(da: xr.DataArray, window: Tuple[int, int, int, int]) -> xr.DataArray:
    """The window of pixels ``(col0, row0, col1, row1)`` of a cube, with its coordinates and transform."""
    from affine import Affine

    from ._warp import transform_of

    col0, row0, col1, row1 = window
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"window=/pixel= need y and x dimensions, got {da.dims}")
    if col1 > da.sizes["x"] or row1 > da.sizes["y"]:
        raise ValueError(f"the window {tuple(window)} is outside the raster ({da.sizes['x']} columns x "
                         f"{da.sizes['y']} rows)")
    try:
        transform = transform_of(da)
    except Exception:  # noqa: BLE001 - no georeferencing to move
        transform = None
    out = da.isel(x=slice(col0, col1), y=slice(row0, row1))
    if transform is not None:
        out = out.rio.write_transform(transform * Affine.translation(col0, row0))
    return out


def _scale_offsets(src: Any) -> Tuple[np.ndarray, np.ndarray]:
    """The scale and offset of each band of an open rasterio dataset (1 and 0 where it has none)."""
    scales = np.array([1.0 if s is None or not np.isfinite(s) else s for s in src.scales], dtype=np.float64)
    offsets = np.array([0.0 if o is None or not np.isfinite(o) else o for o in src.offsets], dtype=np.float64)
    return scales, offsets


def _apply_scale(da: xr.DataArray, scales: Any, offsets: Any) -> xr.DataArray:
    """``da * scale + offset`` with NoData as NaN, where a scale or an offset changes something (else ``da`` as it
    is). ``scales`` and ``offsets`` hold one value, or one for each band of ``da``."""
    scales = np.atleast_1d(np.asarray(scales, dtype=np.float64))
    offsets = np.atleast_1d(np.asarray(offsets, dtype=np.float64))
    if (scales == 1.0).all() and (offsets == 0.0).all():
        return da
    da = _mask(da, True)  # the NoData value is not a measurement: it must not be scaled
    if not np.issubdtype(da.dtype, np.floating):
        da = da.astype(np.float64 if da.dtype.itemsize >= 4 else np.float32)
    with xr.set_options(keep_attrs=True):
        if "band" in da.dims and da.sizes["band"] == len(scales) and not (
                (scales == scales[0]).all() and (offsets == offsets[0]).all()):
            da = (da * xr.DataArray(scales.astype(da.dtype), dims="band")
                  + xr.DataArray(offsets.astype(da.dtype), dims="band"))
        else:
            da = da * float(scales[0]) + float(offsets[0])
    for key in ("scale_factor", "add_offset"):
        da.attrs.pop(key, None)
        da.encoding.pop(key, None)
    return da.rio.write_nodata(np.nan, encoded=True)


def _read_bands(source: Any, bands: Any, common: Dict[str, Any]) -> xr.DataArray:
    """``load_raster(..., bands={name: band})``: a ``(time, band, y, x)`` cube of the named bands."""
    if not isinstance(bands, dict) or not bands:
        raise ValueError("bands= takes a dict of band roles to bands, e.g. {'red': 3, 'nir': 4}")
    names = [str(k) for k in bands]
    if len(set(names)) != len(names):
        raise ValueError(f"bands= has a name twice: {names}")
    layers = [_read(source, band=b, **common).drop_vars("band", errors="ignore") for b in bands.values()]
    try:
        cube = xr.concat(layers, dim=pd.Index(names, name="band"), coords="minimal", compat="override",
                         join="exact")
    except ValueError as err:
        raise ValueError(f"the bands do not share dates and grid: {err}") from None
    cube.attrs = dict(layers[0].attrs)
    return cube.transpose(*[d for d in ("time", "band", "y", "x") if d in cube.dims])


def _apply_qa(da: xr.DataArray, source: Any, qa: Any, rule: str, common: Dict[str, Any]) -> xr.DataArray:
    """The cube where the quality band ``qa`` (a band of ``source``, or another source) keeps the observation."""
    in_source = isinstance(qa, (int, np.integer)) or (
        isinstance(qa, str) and isinstance(source, (xr.DataArray, xr.Dataset)))
    kw = dict(common, masked=True, resampling="nearest", nodata=None, dtype=None, cutline=None)
    if (common["like"] is not None or common["crs"] is not None or common["res"] is not None
            or common["extent"] is not None or common["tap"] or common["cutline"] is not None):
        # the data is on its grid already: the quality band is put on the same one
        kw.update(like=da, crs=None, res=None, clip=None, extent=None, tap=False)
    q = _read(source if in_source else qa, band=qa if in_source else None, **kw)
    mask = qa_mask(q, rule)
    for dim in ("time", "y", "x"):
        if dim in mask.dims:
            if dim not in da.dims or mask.sizes[dim] != da.sizes[dim]:
                raise ValueError(f"the quality band has {dim} {mask.sizes[dim]}, the data "
                                 f"{da.sizes.get(dim, 'none')}: they must share dates and grid")
            if dim == "time" and not np.array_equal(mask["time"].values, da["time"].values):
                raise ValueError("the quality band and the data have different dates")
            mask = mask.assign_coords({dim: da[dim].values})
    keep = mask.drop_vars([c for c in mask.coords if c not in mask.dims], errors="ignore")
    out = da.where(keep)
    out.attrs = dict(da.attrs)
    return out


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


def _is_xarray_file(path: str) -> bool:
    """A Zarr store or a NetCDF file (opened by xarray, not by rasterio)."""
    lower = path.rstrip("/\\").lower()
    return _is_store(lower) or lower.endswith((".nc", ".nc4", ".netcdf"))


def _open_rio(path: str, chunks: Any, *, quiet: bool) -> xr.DataArray:
    """``rioxarray.open_rasterio`` of the raw values; ``quiet`` drops rioxarray's "Scales differ across bands"
    and "Offsets differ across bands" warnings (its ``scale_factor`` attribute is not used), and only them."""
    with warnings.catch_warnings():
        if quiet:
            warnings.filterwarnings("ignore", message=r"(Scales|Offsets) differ across bands")
        return rioxarray.open_rasterio(path, masked=False, chunks=chunks)


def _open_one(path: str, *, band: Any, chunks: Any, date_format: Optional[str], scale: bool = False,
              window: Optional[Tuple[int, int, int, int]] = None) -> xr.DataArray:
    lower = path.rstrip("/\\").lower()
    if _is_xarray_file(path):
        engine = "zarr" if _is_store(lower) else None
        ds = xr.open_dataset(path, engine=engine, chunks=chunks if chunks is not None else {},
                             decode_coords="all")
        return _from_xarray(ds, band)

    with rasterio.open(path) as src:
        descriptions = list(src.descriptions)
        tags = src.tags()
        scales, offsets = _scale_offsets(src) if scale else (None, None)
    tag = tags.get(TIME_TAG)
    # With scale= the scales are applied per band here; with one band picked the other bands' scales are
    # of no interest: rioxarray's "Scales differ across bands" does not apply to either.
    da = _open_rio(path, _file_chunks(chunks), quiet=bool(scale) or band is not None)
    if window is not None:  # lazily indexed: only the window is read from the file
        da = _cut(da, window)
    da.attrs.pop("long_name", None)
    count = da.sizes["band"]
    times: Optional[pd.DatetimeIndex] = None
    names: Optional[List[Any]] = None
    tagged = pd.DatetimeIndex(json.loads(tag)) if tag else None
    if tagged is not None and len(tagged) == count and tagged.is_unique:
        times = tagged
    if times is None:
        if date_format is not None:
            found = [find_date(d, date_format) if d else None for d in descriptions]
            parsed = (pd.DatetimeIndex([f[0] for f in found]), None) if all(found) else None
        else:
            parsed = dates_from_labels(descriptions)
        if parsed is not None:
            times, names = parsed
    # several bands a date, date after date: the tag's or the side-car file's dates, one for every few bands
    if times is None and tagged is not None:
        times, names = _dates_of_bands(tagged, count, descriptions)
    if times is None:
        found = _sidecar_dates(path)
        if found is not None:
            times, names = _dates_of_bands(found, count, descriptions)
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
        if scale:
            scales, offsets = scales[[int(b) - 1 for b in idx]], offsets[[int(b) - 1 for b in idx]]
        if np.isscalar(band):
            da = da.squeeze("band", drop=False)
    if scale:
        da = _apply_scale(da, scales, offsets)
    da.attrs["source"] = path
    from ._embeddings import TAG, attrs_from_tag

    da.attrs.update(attrs_from_tag(tags.get(TAG)))
    if da.name is None:
        da.name = os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
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


def _sidecar_dates(path: str) -> Optional[pd.DatetimeIndex]:
    """The dates of a ``<name>_dates.csv`` next to the file (its ``Date`` column; the name in any case)."""
    folder, name = os.path.split(path)
    wanted = os.path.splitext(name)[0].lower() + "_dates.csv"
    try:
        csv = next((os.path.join(folder, f) for f in os.listdir(folder or ".") if f.lower() == wanted), None)
    except OSError:
        return None
    if csv is None or not os.path.isfile(csv):
        return None
    try:
        table = pd.read_csv(csv)
        column = next((c for c in table.columns if str(c).strip().lower() == "date"), None)
        return None if column is None else pd.DatetimeIndex(pd.to_datetime(table[column]))
    except (ValueError, TypeError):
        return None


def _dates_of_bands(times: pd.DatetimeIndex, count: int, descriptions: Sequence[Optional[str]]
                    ) -> Tuple[Optional[pd.DatetimeIndex], Optional[List[Any]]]:
    """``(times, names)`` of ``count`` bands from dates: one a band (names ``None``), or one for every
    ``count / len(times)`` bands, date after date (a CCDC stack of 125 dates x 7 bands in 875 bands), each
    band's date and name. The names of a date's bands are their descriptions when every date repeats them,
    else 1, 2, ..."""
    if len(times) == count:
        return (times, None) if times.is_unique else (None, None)
    if len(times) < 2 or count % len(times) or not times.is_unique:
        return None, None
    per = count // len(times)
    first = [d or "" for d in descriptions[:per]]
    named = all(first) and len(set(first)) == per and all(
        (descriptions[b] or "") == first[b % per] for b in range(count))
    names: List[Any] = first if named else list(range(1, per + 1))
    return times.repeat(per), names * len(times)


def _open_many(files: List[str], *, band: Any, chunks: Any, pattern: Optional[str],
               date_format: Optional[str], dates: Any, start_year: Optional[int], grid: Any = None,
               resampling: str = "auto", target: Optional[Tuple[Any, ...]] = None, scale: bool = False,
               window: Optional[Tuple[int, int, int, int]] = None,
               warp_kw: Optional[Dict[str, Any]] = None) -> xr.DataArray:
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
    if target is not None:  # crs=/res=: a grid that covers every file
        from ._warp import suggested_grid
        headers = []
        for path, _, _ in records:
            with rasterio.open(path) as src:
                headers.append((src.transform, (src.height, src.width), src.crs))
        grid = suggested_grid(headers, *target)

    layers = []
    reference = None
    for path, when, spectral in records:
        layer = _open_rio(path, _file_chunks(chunks, single=True), quiet=True)  # one band is picked from the file
        if window is not None:
            layer = _cut(layer, window)
        if layer.sizes["band"] > 1:
            if band is None:
                raise ValueError(f"{os.path.basename(path)} has {layer.sizes['band']} bands; "
                                 "files of a series need one band each, or give band=N")
            layer = layer.isel(band=int(band) - 1)
            at = int(band) - 1
        else:
            layer = layer.isel(band=0)
            at = 0
        if scale:
            with rasterio.open(path) as src:
                scales, offsets = _scale_offsets(src)
            layer = _apply_scale(layer, scales[at], offsets[at])
        layer = layer.drop_vars("band", errors="ignore")
        layer.attrs.pop("long_name", None)
        if grid is not None:  # every file onto the reference grid
            from ._warp import to_grid
            layers.append(to_grid(layer, grid, resampling, **(warp_kw or {})))
            continue
        here = (layer.shape, layer.rio.transform(), layer.rio.crs)
        if reference is None:
            reference = (here, path)
        elif here != reference[0]:
            raise ValueError(f"{os.path.basename(path)} is not on the grid of {os.path.basename(reference[1])}: "
                             "every file of a series needs the same CRS, size and transform, "
                             "or pass like= to put them on one grid")
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


def grid_of(like: Any):
    """The grid (``_warp.Grid``) of a reference raster: a DataArray or Dataset, or a path
    (a raster file is only opened for its grid; anything else load_raster reads is read
    lazily)."""
    from rasterio.crs import CRS

    from ._warp import Grid, transform_of

    if isinstance(like, (str, os.PathLike)):
        path = os.path.expanduser(os.fspath(like))
        if os.path.isfile(path) and not path.lower().endswith((".nc", ".nc4", ".netcdf")):
            with rasterio.open(path) as src:
                t, (h, w), crs = src.transform, (src.height, src.width), src.crs
            if crs is None:
                raise ValueError(f"like={like!r} has no CRS")
            x = t.c + t.a * (np.arange(w) + 0.5)
            y = t.f + t.e * (np.arange(h) + 0.5)
            return Grid(t, (h, w), CRS.from_user_input(crs), x, y)
        like = load_raster(like, chunks="auto")
    if not isinstance(like, (xr.DataArray, xr.Dataset)):
        raise TypeError(f"like= takes a raster path, a DataArray or a Dataset, got {type(like).__name__}")
    ref = like
    rename = {d: _DIM_ALIASES[str(d).lower()] for d in ref.dims if str(d).lower() in _DIM_ALIASES
              and _DIM_ALIASES[str(d).lower()] in ("x", "y") and _DIM_ALIASES[str(d).lower()] not in ref.dims}
    if rename:
        ref = ref.rename(rename)
    if "y" not in ref.dims or "x" not in ref.dims:
        raise ValueError("like= needs a raster with y and x dimensions")
    if ref.rio.crs is None:
        raise ValueError("like= has no CRS")
    if isinstance(ref, xr.Dataset):
        ref = next(ref[v] for v in ref.data_vars if {"y", "x"} <= set(ref[v].dims))
    t = transform_of(ref)
    if t.b or t.d:
        raise ValueError("like=: rotated grids are not supported")
    return Grid(t, (ref.sizes["y"], ref.sizes["x"]), CRS.from_user_input(ref.rio.crs), ref["x"].values,
                ref["y"].values)


def _target_grid(da: xr.DataArray, crs: Any, res: Any, extent: Any = None, tap: bool = False, cutline: Any = None):
    """The grid of ``load_raster(..., crs=, res=, extent=, tap=, cutline=)`` for a cube (``_warp.suggested_grid``)."""
    from ._warp import suggested_grid, transform_of

    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError("crs=/res=: the data needs y and x dimensions")
    return suggested_grid([(transform_of(da), (da.sizes["y"], da.sizes["x"]), da.rio.crs)], crs, res, extent, tap,
                          cutline)


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
            order = list(dict.fromkeys(coord.get_level_values("spectral")))  # unstack sorts; keep the file's order
            da = da.unstack("band").rename(spectral="band").sel(band=order)
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

    da = sorted_by_time(da)
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


def sorted_by_time(da: Any) -> Any:
    """The cube in the order of its dates; as it is (not copied) when it already is."""
    if "time" in da.indexes and not da.indexes["time"].is_monotonic_increasing:
        return da.sortby("time")
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
    steps = da.sizes.get("time", da.sizes.get("band", 1)) if da.ndim > 2 or "time" in da.dims else 1
    if steps < _VALIDATE[algorithm]:
        kind = "an annual time series" if algorithm == "landtrendr" else "a dense time series"
        warnings.warn(f"{label}: expected {kind}, but only {steps} time steps were found.", stacklevel=3)
    if da.chunks is None and np.issubdtype(da.dtype, np.floating):
        values = da.values
        if values.size and np.isfinite(values).any() and np.nanmax(np.abs(values)) <= 10.0:
            warnings.warn(f"{label}: values look like unscaled floats; the algorithm expects values scaled "
                          "by a factor (e.g. 10000).", stacklevel=3)
