"""``build_vrt``: a GDAL virtual raster (VRT) of tiles and bands, written with rasterio only."""

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import escape

import numpy as np
import rasterio

_RESAMPLING = ("nearest", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode", "min", "max", "med", "q1", "q3")


class _Source:
    """What a VRT needs of one file."""

    def __init__(self, path: str):
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.transform is None or not ds.transform.is_rectilinear:
                raise ValueError(f"{path}: build_vrt needs georeferenced, north-up rasters")
            self.path = str(Path(path).absolute())
            self.width, self.height, self.count = ds.width, ds.height, ds.count
            t = ds.transform
            self.ulx, self.uly, self.rx, self.ry = t.c, t.f, t.a, -t.e
            self.lrx, self.lry = self.ulx + self.width * self.rx, self.uly - self.height * self.ry
            self.crs = ds.crs
            self.dtypes, self.nodata = list(ds.dtypes), list(ds.nodatavals)
            self.descriptions = [d or "" for d in ds.descriptions]
            self.block = ds.block_shapes[0]
            self.wkt = ds.crs.to_wkt(version="WKT1_GDAL") if ds.crs.is_projected or ds.crs.is_geographic else ds.crs.to_wkt()


def build_vrt(
    sources: Sequence[Any],
    path: Union[str, os.PathLike],
    *,
    separate: bool = False,
    extent: Optional[Tuple[float, float, float, float]] = None,
    res: Union[float, Tuple[float, float], None] = None,
    resampling: str = "nearest",
    band_names: Optional[Sequence[str]] = None,
    wavelengths: Optional[Mapping[str, float]] = None,
) -> Path:
    """Write a VRT: tiles mosaicked and bands stacked, with no pixel copied.

    A VRT is a small XML file that GDAL, rasterio, QGIS and ``zeit.load_raster`` read as a
    raster. It is what ``gdal.BuildVRT`` writes, without the GDAL Python bindings.

    Parameters
    ----------
    sources
        What the VRT is made of, in one of three forms:

        - a list of files (the tiles of a scene): mosaicked, each band of the output made
          of the same band of every file;
        - a list of files with ``separate=True``: each file is one band of the output (its
          first band), stacked;
        - a list of lists of files: one output band per inner list, made of the first band
          of its files mosaicked (the tiles of each band of a date).

        Where tiles overlap, the later file wins, except where it holds NoData. Pixels no
        tile covers are NoData, or 0 for bands without it. The files must share one CRS.
    path
        The ``.vrt`` file to write (missing folders are created). The source files are
        referred to by absolute path.
    separate
        See ``sources``.
    extent
        ``(xmin, ymin, xmax, ymax)`` of the VRT; ``None``: the union of the files.
    res
        The VRT's pixel, one number or ``(xres, yres)``; ``None``: the finest pixel of the
        files. Files of another pixel size are resampled on reading.
    resampling
        How a source of another pixel size or alignment is read: ``'nearest'`` (default),
        ``'bilinear'``, ``'cubic'``, ``'cubicspline'``, ``'lanczos'``, ``'average'``,
        ``'mode'``...
    band_names
        The description of each band of the output (shown by QGIS and read by
        ``load_raster``); by default the first file's.
    wavelengths
        ``{band name: nanometres}``, written as the band's ``wavelength`` and
        ``wavelength_units`` metadata for the names in ``band_names``.

    Returns
    -------
    pathlib.Path
        The VRT file.

    Notes
    -----
    A band has a NoData value when every file of it has one (the first file's). The VRT keeps
    the type of the first file.

    Examples
    --------
    >>> zeit.build_vrt([["B04_a.tif", "B04_b.tif"], ["B08_a.tif", "B08_b.tif"]], "scene.vrt",
    ...                band_names=["B04", "B08"], wavelengths={"B04": 664.6, "B08": 832.8})
    """
    if resampling not in _RESAMPLING:
        raise ValueError(f"resampling must be one of {_RESAMPLING}, got {resampling!r}")
    if len(sources) == 0:
        raise ValueError("build_vrt needs at least one file")
    nested = all(isinstance(s, (list, tuple)) for s in sources)
    if not nested and any(isinstance(s, (list, tuple)) for s in sources):
        raise ValueError("sources: files, or lists of files, not both")
    cache: Dict[str, _Source] = {}

    def read(f: Any) -> _Source:
        key = str(f)
        if key not in cache:
            cache[key] = _Source(key)
        return cache[key]

    if nested:
        bands = [[(read(f), 1) for f in group] for group in sources]
    elif separate:
        bands = [[(read(f), 1)] for f in sources]
    else:
        files = [read(f) for f in sources]
        counts = {s.count for s in files}
        if len(counts) > 1:
            raise ValueError(f"the files have different numbers of bands ({sorted(counts)}); use separate=True")
        bands = [[(s, b) for s in files] for b in range(1, files[0].count + 1)]
    if any(len(group) == 0 for group in bands):
        raise ValueError("an empty list of files")
    every = [s for group in bands for s, _ in group]
    if len({s.crs.to_wkt() for s in every}) > 1:
        raise ValueError("the files are in different CRS; reproject them first (zeit.load_raster(..., crs=...))")

    if res is None:
        rx, ry = min(s.rx for s in every), min(s.ry for s in every)
    elif np.ndim(res) == 0:
        rx = ry = float(res)
    else:
        rx, ry = (float(v) for v in res)
    if extent is None:
        xmin, ymin = min(s.ulx for s in every), min(s.lry for s in every)
        xmax, ymax = max(s.lrx for s in every), max(s.uly for s in every)
    else:
        xmin, ymin, xmax, ymax = (float(v) for v in extent)
        if not (xmax > xmin and ymax > ymin):
            raise ValueError("extent is (xmin, ymin, xmax, ymax)")
    width, height = int(0.5 + (xmax - xmin) / rx), int(0.5 + (ymax - ymin) / ry)
    if width < 1 or height < 1:
        raise ValueError("the extent is smaller than one pixel")
    names = list(band_names) if band_names is not None else None
    if names is not None and len(names) < len(bands):
        names = names + [""] * (len(bands) - len(names))

    out = [f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">',
           f'  <SRS dataAxisToSRSAxisMapping="1,2">{escape(every[0].wkt)}</SRS>',
           f"  <GeoTransform>{xmin!r}, {rx!r}, 0.0, {ymax!r}, 0.0, {-ry!r}</GeoTransform>"]
    for k, group in enumerate(bands, 1):
        first = group[0][0]
        dtype = _vrt_type(first.dtypes[group[0][1] - 1])
        nodata = [s.nodata[b - 1] for s, b in group]
        has_nodata = all(v is not None for v in nodata)
        out.append(f'  <VRTRasterBand dataType="{dtype}" band="{k}">')
        name = names[k - 1] if names is not None else first.descriptions[group[0][1] - 1]
        if wavelengths and name in wavelengths:
            out += ["    <Metadata>", f'      <MDI key="wavelength">{wavelengths[name]!r}</MDI>',
                    '      <MDI key="wavelength_units">nm</MDI>', "    </Metadata>"]
        if name:
            out.append(f"    <Description>{escape(name)}</Description>")
        if has_nodata:
            out.append(f"    <NoDataValue>{_number(nodata[0])}</NoDataValue>")
        for (s, b), nd in zip(group, nodata):
            window = _window(s, xmin, ymin, xmax, ymax, rx, ry, width, height)
            if window is None:
                continue
            tag = "ComplexSource" if has_nodata else "SimpleSource"
            out += [f'    <{tag} resampling="{resampling}">',
                    f'      <SourceFilename relativeToVRT="0">{escape(s.path)}</SourceFilename>',
                    f"      <SourceBand>{b}</SourceBand>",
                    f'      <SourceProperties RasterXSize="{s.width}" RasterYSize="{s.height}" '
                    f'DataType="{_vrt_type(s.dtypes[b - 1])}" BlockXSize="{s.block[1]}" BlockYSize="{s.block[0]}" />',
                    '      <SrcRect xOff="{}" yOff="{}" xSize="{}" ySize="{}" />'.format(*map(_number, window[:4])),
                    '      <DstRect xOff="{}" yOff="{}" xSize="{}" ySize="{}" />'.format(*map(_number, window[4:]))]
            if has_nodata:
                out.append(f"      <NODATA>{_number(nd)}</NODATA>")
            out.append(f"    </{tag}>")
        out.append("  </VRTRasterBand>")
    out.append("</VRTDataset>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def _window(s: _Source, xmin: float, ymin: float, xmax: float, ymax: float, rx: float, ry: float,
            width: int, height: int) -> Optional[Tuple[float, ...]]:
    """The part of ``s`` that falls in the VRT and where: (src x, y, w, h, dst x, y, w, h) in pixels,
    None when it does not touch the VRT."""
    if s.ulx >= xmax or s.lrx <= xmin or s.lry >= ymax or s.uly <= ymin:
        return None
    if s.ulx < xmin:
        sx, dx = (xmin - s.ulx) / s.rx, 0.0
    else:
        sx, dx = 0.0, (s.ulx - xmin) / rx
    if s.uly > ymax:
        sy, dy = (s.uly - ymax) / s.ry, 0.0
    else:
        sy, dy = 0.0, (ymax - s.uly) / ry
    sw, sh = s.width - max(sx, 0.0), s.height - max(sy, 0.0)
    dw, dh = sw * s.rx / rx, sh * s.ry / ry
    if dx + dw > width:
        dw = width - dx
        sw = dw * rx / s.rx
    if dy + dh > height:
        dh = height - dy
        sh = dh * ry / s.ry
    if dw <= 0 or dh <= 0:
        return None
    return sx, sy, sw, sh, dx, dy, dw, dh


def _number(v: float) -> str:
    """A number for the XML: whole numbers without a point, the rest at full precision."""
    v = float(v)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    if abs(v - round(v)) < 1e-9 * max(1.0, abs(v)):
        return str(int(round(v)))
    return repr(v)


def _vrt_type(dtype: str) -> str:
    names = {"uint8": "Byte", "int8": "Int8", "uint16": "UInt16", "int16": "Int16", "uint32": "UInt32",
             "int32": "Int32", "uint64": "UInt64", "int64": "Int64", "float32": "Float32", "float64": "Float64",
             "complex64": "CFloat32", "complex128": "CFloat64"}
    return names[dtype]
