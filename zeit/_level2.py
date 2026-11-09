# SPDX-License-Identifier: GPL-3.0-or-later
"""The atmospheric correction of a Sentinel-2 Level 1C scene: ``zeit.level2`` (the engine, ``_level2_arrays``, gives arrays).

What FORCE's ``force-l2ps`` does for one image, on a local ``.SAFE`` directory (``zeit._core.l2scene``, a port of FORCE 3.11.00
by David Frantz, GPL-3.0-or-later). This module reads what the core does not:

* the two XML files and the 13 band files (anything GDAL reads: the JPEG 2000 of the product, or GeoTIFFs under the same
  names), the 20 and 60 m bands put on the 10 m grid by nearest neighbour as ``read_level1`` does;
* the geography: the longitude and latitude of the corners of the coarse grid (5000 m) that the correction needs, from the
  projection of the bands (GDAL/PROJ through rasterio);
* the DEM: any raster, put on the grid of the scene by ``load_raster(like=)``, bilinear by default, with ``DEM_NODATA`` where
  it has no data, which is how FORCE's ``GDALWarpOperation`` hands it to the topography.

The products stay on the grid of the scene (10 m). The cloud, shadow, snow and water flags come from Zeit's own Fmask 4.7
(``zeit._core.fmask_run``, ``fmask_flags``: run at 20 m on the bands decimated from the 10 m grid, with the angles of the
``MTD_TL.xml``, the DEM and the optional GSWO) unless they are given (``flags=``) or switched off (``fmask=False``).
"""

import xml.etree.ElementTree  # noqa: F401  (before rasterio: on Windows a DLL clash otherwise)
import xml.etree.ElementTree as ET

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from . import _core

#: The band files of the product in the order of FORCE (``orig_band`` of ``meta-ll.c``).
BAND_NAMES = ("01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12")
DEM_NODATA = -32767
#: The bits of the QAI the cloud detection sets: the cloud state, shadow, snow and water.
FLAG_MASK = 0b111110

_SWITCHES = ("domulti", "doenv", "doatmo", "dotopo", "dobrdf", "erase_cloud", "doaod", "odst", "ovzn", "ohot", "oaod", "owvp")


class Safe:
    """The files of a Level 1C ``.SAFE`` directory that the correction reads."""

    def __init__(self, path: Any):
        self.path = Path(path)
        self.product = self.path / "MTD_MSIL1C.xml"
        granules = sorted(g for g in (self.path / "GRANULE").iterdir() if g.is_dir()) if (self.path / "GRANULE").is_dir() else []
        if not self.product.is_file() or len(granules) != 1:
            raise ValueError(f"{path}: not a Sentinel-2 L1C .SAFE directory (MTD_MSIL1C.xml and one GRANULE/<name>/ expected)")
        self.granule = granules[0]
        self.tile = self.granule / "MTD_TL.xml"
        if not self.tile.is_file():
            raise ValueError(f"{self.granule}: no MTD_TL.xml")
        self.bands = []
        for name in BAND_NAMES:
            found = sorted(glob.glob(str(self.granule / "IMG_DATA" / f"*_B{name}.jp2")) + glob.glob(str(self.granule / "IMG_DATA" / f"*_B{name}.tif")))
            if not found:
                raise ValueError(f"{self.granule / 'IMG_DATA'}: no band file *_B{name}.jp2")
            self.bands.append(found[0])

    def describe(self) -> Dict[str, Any]:
        """The metadata the core reads (``_core.l2scene.describe``): grid, date, corners and read windows."""
        return _core.l2scene.describe(self.product.read_bytes(), self.tile.read_bytes(), self.granule.name)


def read_bands(safe: Safe, desc: Dict[str, Any], n_jobs: int = 8) -> np.ndarray:
    """The 13 bands as ``uint16`` ``(13, nrows, ncols)`` on the grid of the finest band: each is read in its window and
    resampled to the grid by GDAL's nearest neighbour (``read_level1``). ``n_jobs`` caps the threads of the JPEG 2000
    decoder, which would otherwise take every core."""
    nr, nc = desc["nrows"], desc["ncols"]
    windows = {res: win for res, win in desc["windows"]}
    out = np.empty((len(BAND_NAMES), nr, nc), dtype=np.uint16)
    with rasterio.Env(GDAL_NUM_THREADS=int(n_jobs)):
        for b, f in enumerate(safe.bands):
            with rasterio.open(f) as d:
                res = abs(d.transform.a)
                xoff, yoff, nx, ny = windows[min(windows, key=lambda r: abs(r - res))]
                out[b] = d.read(1, window=Window(xoff, yoff, nx, ny), out_shape=(nr, nc), resampling=Resampling.nearest)
    return out


def geo_tables(safe: Safe, desc: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude (degrees, ``float64``, row by row) of the corners ``desc["geo_grid"]`` lists, from the
    projection of the first band file."""
    from rasterio.warp import transform

    ulx, uly, res, nx, ny = desc["geo_grid"]
    with rasterio.open(safe.bands[0]) as d:
        crs = d.crs
    j, i = np.meshgrid(np.arange(nx), np.arange(ny))
    x = ulx + j.ravel() * res
    y = uly + i.ravel() * -res
    lon, lat = transform(crs, "EPSG:4326", x.tolist(), y.tolist())
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _band_grid(safe: Safe, resolution: Optional[float] = None, desc: Optional[Dict[str, Any]] = None):
    """The grid (``_warp.Grid``) of the band file whose resolution is closest to ``resolution`` (the finest if None); with
    ``desc``, cropped to the read window of that resolution (the image of the correction), else the whole file."""
    from affine import Affine

    from ._load import grid_of
    from ._warp import Grid

    best = None
    for f in safe.bands:
        with rasterio.open(f) as d:
            res = abs(d.transform.a)
        key = res if resolution is None else abs(res - resolution)
        if best is None or key < best[0]:
            best = (key, res, f)
    g = grid_of(best[2])
    if desc is None:
        return g
    windows = {r: w for r, w in desc["windows"]}
    xoff, yoff, nx, ny = windows[min(windows, key=lambda r: abs(r - best[1]))]
    t = g.transform * Affine.translation(xoff, yoff)
    return Grid(t, (ny, nx), g.crs, t.c + t.a * (np.arange(nx) + 0.5), t.f + t.e * (np.arange(ny) + 0.5))


def dem_on_grid(dem: Any, safe: Safe, *, resampling: str = "bilinear", nodata: int = DEM_NODATA,
                resolution: Optional[float] = None, desc: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """The DEM on the grid of the scene, ``float32`` ``(nrows, ncols)`` in metres, ``nodata`` where it has no data: the
    warp of ``load_raster(like=)`` (the port of GDAL's) onto the grid of the 10 m band, with the exact coordinates
    (``tolerance=0``, the ``GDALWarpOperation`` of FORCE has an error threshold of 0), ``nodata`` as the source's and the
    destination's no-data value, ``float32``.

    ``resolution`` picks the grid of the band of that resolution (``20.0``: B11, for the Fmask) instead of the finest
    one, and ``desc`` crops it to the read window of the scene (both None: the whole 10 m file, as always)."""
    from ._load import load_raster
    from ._warp import to_grid

    src = load_raster(dem, masked=False)
    if src.rio.nodata is None:
        src = src.rio.write_nodata(nodata)
    da = to_grid(src, _band_grid(safe, resolution, desc), resampling, tolerance=0, nodata=nodata, dtype="float32")
    return np.ascontiguousarray(da.values.reshape(da.shape[-2], da.shape[-1]), dtype=np.float32)


def _gswo_on_grid(gswo: Any, safe: Safe, desc: Dict[str, Any]) -> np.ndarray:
    """The Global Surface Water Occurrence raster (percent) on the 20 m grid by nearest neighbour, ``uint8``, 0 where
    it has no data."""
    from ._load import load_raster
    from ._warp import to_grid

    src = load_raster(gswo, masked=False)
    if src.rio.nodata is None:
        src = src.rio.write_nodata(255)
    da = to_grid(src, _band_grid(safe, 20.0, desc), "nearest", tolerance=0, nodata=0, dtype="uint8")
    return np.ascontiguousarray(da.values.reshape(da.shape[-2], da.shape[-1]), dtype=np.uint8)


def prepare(safe: Safe, desc: Dict[str, Any], dn: np.ndarray, n_jobs: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """The initial QAI ``(nrows, ncols)`` and the TOA reflectance ``(13, nrows, ncols)`` (``int16``, x 10000) the correction
    starts from (``bounds_level1`` and ``convert_level1``)."""
    out = _core.l2scene.prepare(safe.product.read_bytes(), safe.tile.read_bytes(), safe.granule.name,
                                np.ascontiguousarray(dn).ravel(), int(n_jobs))
    shape = (desc["nrows"], desc["ncols"])
    return out["qai"].reshape(shape), out["toa"].reshape((len(BAND_NAMES),) + shape)


#: Indices in ``BAND_NAMES`` of the bands the Fmask reads, in the argument order of ``dn_to_toa20``: B02, B03, B04 (10 m),
#: B8A, B11, B12 (20 m), B10 (60 m), B07 (20 m), B08 (10 m).
_FMASK_BANDS = (1, 2, 3, 8, 11, 12, 10, 6, 7)
_FMASK_STEP = (1, 1, 1, 2, 2, 2, 6, 2, 1)


def _mean_sun_angle(tile_xml: Path) -> Tuple[float, float]:
    """``Mean_Sun_Angle`` of ``MTD_TL.xml``: the zenith and azimuth (degrees) of the whole scene."""
    root = ET.parse(str(tile_xml)).getroot()
    node = next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "Mean_Sun_Angle"), None)
    if node is None:
        raise ValueError(f"{tile_xml}: no Mean_Sun_Angle")
    val = {e.tag.rsplit("}", 1)[-1]: float(e.text) for e in node}
    return val["ZENITH_ANGLE"], val["AZIMUTH_ANGLE"]


def _view_angles_20m(safe: Safe, desc: Dict[str, Any], rows: int, cols: int, n_jobs: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """The view zenith and azimuth of B8A on the 20 m grid in hundredths of a degree (``float32``, what ``run_fmask`` reads),
    from the 5 km grids of ``MTD_TL.xml`` (``sentinel2_angles``), and the fraction of pixels no detector covered, which take
    the mean of the others (a NaN would poison the mean the shadow matching takes)."""
    import xarray as xr
    import rioxarray  # noqa: F401  (the .rio accessor)

    from ._s2angles import sentinel2_angles

    g = _band_grid(safe, 20.0, desc)
    if g.shape != (rows, cols):
        raise ValueError(f"the 20 m grid of the window is {g.shape}, not {(rows, cols)}")
    like = xr.DataArray(np.zeros((rows, cols), np.uint8), dims=("y", "x"), coords={"y": g.y, "x": g.x}).rio.write_crs(g.crs)
    ang = sentinel2_angles(str(safe.tile), like, view_band="B8A", n_jobs=n_jobs)
    out = []
    missing = 0.0
    for name in ("VZA", "VAA"):
        a = np.ascontiguousarray(ang[name].values, dtype=np.float32)
        bad = ~np.isfinite(a)
        if bad.all():
            raise ValueError("MTD_TL.xml: no view angle covers the scene")
        missing = max(missing, float(bad.mean()))
        if bad.any():
            a[bad] = np.float32(a[~bad].mean())
        a *= np.float32(100.0)
        out.append(a)
    return out[0], out[1], missing


def fmask_flags(
    safe: Safe,
    desc: Dict[str, Any],
    dn: np.ndarray,
    *,
    dem: Any = None,
    gswo: Any = None,
    compat: bool = False,
    n_jobs: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The cloud, shadow, snow and water bits of the QAI (``int16`` ``(nrows, ncols)``, 10 m) from Zeit's own Fmask 4.7
    (``zeit._core.fmask_run``), and a dict of what it did.

    ``dn`` is the ``(13, nrows, ncols)`` ``uint16`` of ``read_bands``. The Fmask runs at 20 m, on the grid of the window of the
    scene: the 10 m bands (B02, B03, B04, B08) go in whole, the 20 m ones (B8A, B11, B12, B07) by ``[::2, ::2]`` and B10 (60 m)
    by ``[::6, ::6]`` (so ``nrows`` and ``ncols`` must be multiples of 6, as the 10980 of a tile are). The radiometry is that
    of ``nd2toarbt_msi2.m`` from processing baseline 04.00 on (``(DN + RADIO_ADD_OFFSET) / QUANTIFICATION``, the offsets
    negative as the XML gives them) and the old one (the DN itself) before. The sun is the ``Mean_Sun_Angle`` of the
    ``MTD_TL.xml`` (one pair for the scene, as the Fmask uses) and the view angles those of B8A per pixel, in hundredths of a
    degree. ``dem`` (a raster, or an array on the 20 m grid or the 10 m one, in metres) gives the DEM, its slope and aspect;
    ``DEM_NODATA`` is NaN to the Fmask. ``gswo`` (a raster or a ``uint8`` array of the 20 m grid, percent) is optional. The
    cloud buffer of the output is 300 m and cirrus is never flagged (as in FORCE).
    """
    fr = _core.fmask_run
    nr, nc = desc["nrows"], desc["ncols"]
    if dn.shape != (len(BAND_NAMES), nr, nc):
        raise ValueError(f"dn has shape {dn.shape}, expected {(len(BAND_NAMES), nr, nc)}")
    if nr % 6 or nc % 6:
        raise ValueError(f"the Fmask needs the scene's rows and columns to be multiples of 6, got {nr} x {nc}")
    rows, cols = nr // 2, nc // 2
    bands = [np.ascontiguousarray(dn[b, ::k, ::k]).ravel() for b, k in zip(_FMASK_BANDS, _FMASK_STEP)]

    baseline = desc.get("baseline")
    try:
        new = float(str(baseline).strip()) >= 4.0
    except ValueError:
        new = any(desc["radd"])
    if new:
        radd = np.asarray(desc["radd"], dtype=np.float64)
        offsets = np.ascontiguousarray(radd[list(_FMASK_BANDS)], dtype=np.float64)
        toa = fr.dn_to_toa20(rows, cols, *bands, compat, offsets, float(desc["dn_scale"]))
    else:
        toa = fr.dn_to_toa20(rows, cols, *bands, compat)

    zen, azi = _mean_sun_angle(safe.tile)
    vza, vaa, vmiss = _view_angles_20m(safe, desc, rows, cols, n_jobs)

    kw: Dict[str, Any] = {}
    if dem is not None:
        if isinstance(dem, np.ndarray):
            d = np.asarray(dem, dtype=np.float32)
            if d.shape == (nr, nc):
                d = d[::2, ::2]
            if d.shape != (rows, cols):
                raise ValueError(f"dem has shape {d.shape}, expected the 20 m grid {(rows, cols)} or the 10 m one {(nr, nc)}")
        else:
            d = dem_on_grid(dem, safe, resolution=20.0, desc=desc)
        d = np.where(d == DEM_NODATA, np.float32(np.nan), d).astype(np.float32)
        slope, aspect = fr.slope_aspect_deg(np.ascontiguousarray(d).ravel(), rows, cols, 20.0)
        kw.update(dem=np.ascontiguousarray(d).ravel(), slope=slope, aspect=aspect)
    if gswo is not None:
        g = np.asarray(gswo, dtype=np.uint8) if isinstance(gswo, np.ndarray) else _gswo_on_grid(gswo, safe, desc)
        if g.shape != (rows, cols):
            raise ValueError(f"gswo has shape {g.shape}, expected the 20 m grid {(rows, cols)}")
        kw.update(gswo=np.ascontiguousarray(g).ravel())

    r = fr.run_fmask(rows, cols, toa["blue"], toa["green"], toa["red"], toa["nir"], toa["swir1"], toa["swir2"], toa["cirrus"],
                     toa["satu_blue"], toa["satu_green"], toa["satu_red"], vza.ravel(), vaa.ravel(), zen, azi, compat, False,
                     b7=toa["b7"], b8=toa["b8"], **kw)
    q = fr.to_qai(rows, cols, 2, 20.0, r["mask"], r["pcloud"], r["water"], r["psnow"], r["pshadow"], 300.0, int(n_jobs))
    flags = np.ascontiguousarray(q, dtype=np.int16).reshape(nr, nc)

    obs = np.asarray(r["mask"]).astype(bool)
    n_obs = max(int(obs.sum()), 1)
    frac = lambda a: float(np.asarray(a).astype(bool)[obs].sum()) / n_obs
    info = dict(
        sum_clr=r["sum_clr"], few_clear=r["few_clear"], match_skipped=r["match_skipped"], clr_pct=r["clr_pct"],
        observed=float(obs.mean()), cloud=frac(r["pcloud"]), shadow=frac(r["pshadow"]), snow=frac(r["psnow"]), water=frac(r["water"]),
        baseline=baseline, route="baseline04" if new else "legacy", sun_zenith=zen, sun_azimuth=azi,
        view_band="B8A", view_missing=vmiss, dem=dem is not None, gswo=gswo is not None, compat=bool(compat),
        grid=(rows, cols), cloud_buffer_m=300.0,
    )
    return flags, info


def _level2_arrays(
    safe: Any,
    *,
    dem: Any = None,
    flags: Optional[np.ndarray] = None,
    dobrdf: bool = True,
    dotopo: bool = True,
    doaod: bool = True,
    doenv: bool = True,
    domulti: bool = True,
    erase_cloud: bool = False,
    dem_resampling: str = "bilinear",
    n_jobs: int = 8,
    geography: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    sun: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    brdf_table: Optional[np.ndarray] = None,
    bands: Optional[np.ndarray] = None,
    dem_array: Optional[np.ndarray] = None,
    fmask: bool = True,
    gswo: Any = None,
    fmask_compat: bool = False,
) -> Dict[str, Any]:
    """The Level 2 products of the scene: ``boa`` ``(10, nrows, ncols)`` ``int16`` (reflectance x 10000, ``-9999`` no data),
    ``qai``, ``dst``, ``vzn``, ``hot``, ``aod``, ``wvp`` ``(nrows, ncols)`` ``int16``, the scene's water vapour and AOD, the
    sun angles the correction used and the time of the core in ms.

    ``flags``: the cloud, shadow, snow and water bits of the QAI per pixel (``int16``, only ``FLAG_MASK`` is read). Without
    them (the default) Zeit's own Fmask 4.7 makes them (``fmask_flags``: at 20 m, with the DEM and ``gswo`` if given, ``fmask_compat``
    the ``compat`` of the port) and ``out["fmask"]`` says what it did; ``fmask=False`` runs no cloud detection, as before. With
    ``flags`` the Fmask does not run. ``geography`` replaces the longitude and latitude of the corners, ``sun`` the sun zenith and azimuth (degrees) of the
    coarse cells, ``brdf_table`` the BRDF factor ``(13, cells)`` of every band and coarse cell; ``bands`` and ``dem_array`` replace the reading of the files (a scene read before).
    """
    s = safe if isinstance(safe, Safe) else Safe(safe)
    desc = s.describe()
    nr, nc = desc["nrows"], desc["ncols"]
    dn = read_bands(s, desc, n_jobs) if bands is None else bands
    if dem_array is None and dem is not None:
        dem_array = dem_on_grid(dem, s, resampling=dem_resampling, desc=desc)
    info = None
    if flags is None and fmask:
        # the raster is warped straight to the 20 m grid; an array given instead is decimated
        flags, info = fmask_flags(s, desc, dn, dem=dem if dem is not None and not isinstance(dem, np.ndarray) else dem_array,
                                  gswo=gswo, compat=fmask_compat, n_jobs=n_jobs)
    lon, lat = geography if geography is not None else geo_tables(s, desc)
    f = np.zeros(0, dtype=np.int16) if flags is None else np.ascontiguousarray(flags, dtype=np.int16).ravel()
    d = np.zeros(0, dtype=np.float32) if dem_array is None else np.ascontiguousarray(dem_array, dtype=np.float32).ravel()
    params = dict(domulti=domulti, doenv=doenv, doatmo=True, dotopo=dotopo, dobrdf=dobrdf,
                  erase_cloud=erase_cloud, doaod=doaod, odst=True, ovzn=True, ohot=True, oaod=True, owvp=True,
                  dem_nodata=DEM_NODATA, buffer_nodata=False, n_jobs=int(n_jobs))
    args = (s.product.read_bytes(), s.tile.read_bytes(), s.granule.name, np.ascontiguousarray(dn).ravel(), f, d,
            np.ascontiguousarray(lon, dtype=np.float64), np.ascontiguousarray(lat, dtype=np.float64), params)
    kw = {}
    if sun is not None:
        kw.update(sun_zen=np.ascontiguousarray(sun[0], dtype=np.float32), sun_azi=np.ascontiguousarray(sun[1], dtype=np.float32))
    if brdf_table is not None:
        kw.update(brdf=np.ascontiguousarray(brdf_table, dtype=np.float32).ravel())
    out = _core.l2scene.run(*args, **kw)
    shape = (nr, nc)
    for k in ("qai", "dst", "vzn", "hot", "aod", "wvp"):
        if k in out:
            out[k] = out[k].reshape(shape)
    out["boa"] = out["boa"].reshape((-1,) + shape)
    out["desc"] = desc
    if info is not None:
        out["fmask"] = info
    return out


#: The bands of ``boa`` in the order of the product (``boa_domain`` of FORCE for Sentinel-2).
BOA_BANDS = ("BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2")
BOA_NODATA = -9999
_PROVENANCE = ("Sentinel-2 L1C to Level 2: the atmospheric correction of FORCE's force-l2ps (Frantz 2019, Remote Sensing "
               "11:1124; FORCE 3.11.00, David Frantz, GPL-3.0-or-later) ported to the Zeit core, with the cloud, shadow, snow and "
               "water flags of Zeit's Fmask 4.7 port (Qiu, Zhu and Woodcock 2019, Remote Sensing of Environment 231:111205) "
               "unless flags were given or the Fmask was switched off")


def _plain(value: Any) -> Any:
    """``value`` in types an attribute can serialize (``bool``, ``int``, ``float``, ``str``, lists of them); None if it has none."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    return str(value)


def level2(
    safe: Any,
    dem: Any = None,
    *,
    brdf: bool = True,
    topography: bool = True,
    aod: bool = True,
    adjacency: bool = True,
    multiple_scattering: bool = True,
    erase_cloud: bool = False,
    flags: Optional[np.ndarray] = None,
    fmask: bool = True,
    gswo: Any = None,
    fmask_compat: bool = False,
    dem_resampling: str = "bilinear",
    n_jobs: int = 8,
):
    """The atmospheric correction of a Sentinel-2 Level 1C scene to surface reflectance: FORCE's ``force-l2ps``.

    Runs on a local ``.SAFE`` directory (or on the name of a product, which ``zeit.download_l1c`` fetches first; ``zeit.find_l1c``
    lists them) the correction FORCE 3.11.00 (Frantz 2019, *Remote Sensing* 11:1124; David Frantz,
    GPL-3.0-or-later) does for one image, in the Zeit core: the water vapour and gases, the aerosol optical depth estimated from
    dark targets, the adjacency effect, the topographic and the BRDF corrections, and the products on the grid of the scene
    (10 m). The cloud, shadow, snow and water flags come from Zeit's Fmask 4.7 (Qiu, Zhu and Woodcock 2019, *Remote Sensing
    of Environment* 231:111205), run at 20 m with the DEM, unless ``flags`` gives them or ``fmask=False``. It reads the 13 band
    files (JPEG 2000 or GeoTIFF) and the two XML files; it does not merge the 20 m bands nor coregister, and it is only for
    Sentinel-2.

    Parameters
    ----------
    safe
        The ``.SAFE`` directory of a Level 1C product (``MTD_MSIL1C.xml`` and one ``GRANULE/<name>/`` with the ``IMG_DATA``), or
        the name of a product (a string that is not an existing directory and ends with ``.SAFE`` or starts with ``S2``, such as
        the ``product`` column of ``zeit.find_l1c``): it is downloaded to the ``l1c`` cache by ``zeit.download_l1c``.
    dem
        The digital elevation model, in metres: a raster path, or a DataArray, of any grid and CRS (it is put on the 10 m
        grid of the scene, with ``dem_resampling``), or ``"glo30"``: the Copernicus GLO-30 DEM of the scene, made by
        ``zeit.download_dem``. Needed for ``topography=True``; also used by the atmosphere (elevation) and
        the Fmask when given.
    brdf
        Normalise the BRDF to nadir view and the sun at the zenith of the scene (FORCE's ``DO_BRDF``).
    topography
        Correct the illumination of the terrain (FORCE's ``DO_TOPO``); needs ``dem``.
    aod
        Estimate the aerosol optical depth from the dark targets of the scene (FORCE's ``DO_AOD``); False uses the fallback
        spectrum.
    adjacency
        The environmental weighting of the adjacency effect (FORCE's ``ADJACENCY_EFFECT``).
    multiple_scattering
        Multiple scattering (FORCE's ``MULTI_SCATTERING``); False is single scattering.
    erase_cloud
        Set the cloud pixels to no data (FORCE's ``ERASE_CLOUD``).
    flags
        The cloud, shadow, snow and water bits of the QAI per pixel, ``int16`` ``(nrows, ncols)`` on the 10 m grid (only the
        bits 1 to 5 are read). With them the Fmask does not run.
    fmask
        Run Zeit's Fmask when ``flags`` is None; False runs no cloud detection (no cloud flag, and the distance to cloud is
        32767 in the whole image).
    gswo
        Optional Global Surface Water Occurrence raster (percent) for the Fmask, put on the 20 m grid by nearest neighbour.
    fmask_compat
        ``True`` reproduces three oddities of the original Fmask 4.7 code; the default corrects them.
    dem_resampling
        The resampling of the DEM onto the grid of the scene: ``"bilinear"`` (FORCE's default), ``"nearest"``, ``"cubic"``...
    n_jobs
        Threads of the correction, the Fmask and the JPEG 2000 decoder.

    Returns
    -------
    xarray.Dataset
        On the grid of the scene, with the CRS and the transform of the bands (``x`` and ``y`` are the pixel centres):
        ``boa`` ``(band, y, x)`` ``int16``, the reflectance times 10000 with ``_FillValue`` -9999 (``band`` is BLUE, GREEN,
        RED, REDEDGE1, REDEDGE2, REDEDGE3, BROADNIR, NIR, SWIR1, SWIR2), and ``qai`` (FORCE's quality bits), ``dst`` (distance
        to cloud, shadow or snow), ``vzn`` (view zenith), ``hot`` (haze optimized transformation), ``aod`` (aerosol optical
        depth) and ``wvp`` (water vapour), ``(y, x)`` ``int16`` in FORCE's units. ``attrs`` has the sensing time
        (``datetime``, UTC), ``sensor``, ``baseline``, ``aod_green`` (the optical depth of the scene in B03), ``aod_map``
        (False when the fallback spectrum was used), ``water_vapor_cm`` (the mean of the ``wvp`` product, NaN if none),
        ``n_targets`` (valid water, shadow and vegetation dark targets), the Fmask's summary as ``fmask_*`` keys, the time of
        the core (``core_seconds``) and ``provenance``. ``.zeit.save(path)`` and ``.zeit.plot()`` work on it.

    Raises
    ------
    ValueError
        ``topography=True`` without ``dem``.

    Notes
    -----
    Equal to FORCE 3.11.00 in every pixel of ``boa``, ``qai``, ``aod``, ``wvp``, ``hot``, ``dst`` and ``vzn`` when the cloud
    flags, the sun, the geography and the BRDF factor are FORCE's (see the benchmark). Zeit computes them itself (its own sun,
    Fmask and NBAR c-factor as BRDF factor), and then it differs a little (see
    ``docs/benchmarks/vs-originals.md``). The products stay on the 10 m grid of the scene: FORCE's reprojection to the datacube
    is not done.

    Examples
    --------
    >>> l2 = zeit.level2("S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149.SAFE", dem="cop_glo30.tif")
    >>> l2.boa.sel(band="RED")                      # reflectance x 10000
    >>> l2.zeit.save("l2_20230728")                 # one GeoTIFF per variable
    >>> l2 = zeit.level2(safe, brdf=False, topography=False)   # no DEM, no BRDF
    >>> product = zeit.find_l1c(tile="23LKC", start="2023-07-28", end="2023-07-28")["product"][0]
    >>> l2 = zeit.level2(product, dem="glo30")      # downloads the product and the DEM first
    """
    import xarray as xr
    import rioxarray  # noqa: F401  (the .rio accessor)
    from affine import Affine

    if topography and dem is None:
        raise ValueError("zeit.level2: topography=True needs the DEM: pass dem= (a raster in metres), dem=\"glo30\" or topography=False")
    # a bare name only: a local path that does not exist is a mistake to report (by Safe), not an 800 MB download
    if isinstance(safe, str) and not os.path.isdir(safe) and os.path.basename(safe) == safe and (safe.endswith(".SAFE") or safe.startswith("S2")):
        from ._s2fetch import download_l1c

        safe = download_l1c(safe)
    if isinstance(dem, str) and dem.lower() == "glo30":
        from ._s2fetch import download_dem

        dem = download_dem(safe.path if isinstance(safe, Safe) else safe)
    out = _level2_arrays(safe, dem=dem, flags=flags, dobrdf=brdf, dotopo=topography, doaod=aod, doenv=adjacency,
                         domulti=multiple_scattering, erase_cloud=erase_cloud, dem_resampling=dem_resampling,
                         n_jobs=n_jobs, fmask=fmask, gswo=gswo, fmask_compat=fmask_compat)
    desc = out["desc"]
    nr, nc = desc["nrows"], desc["ncols"]
    res = float(desc["res"])
    x = desc["ulx"] + res * (np.arange(nc) + 0.5)
    y = desc["uly"] - res * (np.arange(nr) + 0.5)
    coords = {"y": y, "x": x}
    boa = np.asarray(out["boa"], dtype=np.int16)
    if boa.shape[0] != len(BOA_BANDS):
        raise ValueError(f"the correction gave {boa.shape[0]} bands, expected {len(BOA_BANDS)}")
    data = {"boa": xr.DataArray(boa, dims=("band", "y", "x"), coords={"band": list(BOA_BANDS), **coords})}
    for k in ("qai", "dst", "vzn", "hot", "aod", "wvp"):
        if k in out:
            data[k] = xr.DataArray(np.asarray(out[k], dtype=np.int16), dims=("y", "x"), coords=coords)
    ds = xr.Dataset(data)
    ds = ds.rio.write_crs(f"EPSG:{int(desc['epsg'])}")
    ds = ds.rio.write_transform(Affine(res, 0.0, desc["ulx"], 0.0, -res, desc["uly"]))
    ds["boa"] = ds["boa"].rio.write_nodata(BOA_NODATA)
    ds["boa"].attrs["long_name"] = "bottom of atmosphere reflectance x 10000"

    wvp = np.asarray(out["wvp"]) if "wvp" in out else np.zeros(0, np.int16)
    wvp = wvp[wvp != BOA_NODATA]
    aod_scene = np.asarray(out["aod_scene"], dtype=np.float64)
    yr, mo, dy, hh, mi, ss = desc["date"]
    attrs: Dict[str, Any] = dict(
        datetime=f"{int(yr):04d}-{int(mo):02d}-{int(dy):02d}T{int(hh):02d}:{int(mi):02d}:{int(ss):02d}Z",
        sensor=str(desc["sensor"]), baseline=str(desc["baseline"]),
        aod_green=float(aod_scene[2]) if aod_scene.size > 2 else float("nan"), aod_map=bool(out["aodmap"]),
        water_vapor_cm=float(wvp.mean()) / 1000.0 if wvp.size else float("nan"),
        n_targets=[int(v) for v in out["n_targets"]], core_seconds=float(out["ms"]) / 1e3,
    )
    for k, v in (out.get("fmask") or {}).items():
        v = _plain(v)
        if v is not None:
            attrs[f"fmask_{k}"] = v
    attrs["provenance"] = _PROVENANCE
    ds.attrs.update(attrs)
    return ds
