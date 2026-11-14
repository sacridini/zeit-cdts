"""``load_level2_series`` and ``level2_series``: the scenes of ``run_series`` as one lazy ``(time, band, y, x)`` cube.

The cube is what ``zeit.temporal_metrics``, ``zeit.regularize_time_series`` and ``zeit.ccdc`` read as they are:
reflectance x 10000 as ``float32`` with NaN where there is no usable observation."""

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ._level2 import BOA_BANDS, BOA_NODATA
from ._level2_series import JOURNAL, MARKER, _read_json, run_series
from ._load import load_raster
from .qc import qa_mask

_MASKS = ("force", None)
_NUMERIC_ATTRS = ("aod_green", "water_vapor_cm")


def _num(value: Any) -> Optional[float]:
    """``value`` as a float if it is a number (the marker stores a NaN or an infinity as a string), else None."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
        return float(value)
    return None


def _utc_naive(text: Any) -> pd.Timestamp:
    t = pd.Timestamp(text)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.tz_localize(None)


def _scene_name(product: Any) -> str:
    base = os.fspath(product).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return base[:-len(".SAFE")] if base.endswith(".SAFE") else base


def _scenes(out_dir: Path, products: Any) -> List[Dict[str, Any]]:
    """The scenes of ``out_dir`` with a readable marker (or the ``products`` asked for): ``name``, ``folder``, ``marker``."""
    if not out_dir.is_dir():
        raise ValueError(f"{out_dir} has no level2 scene (it is not a folder)")
    if products is None:
        folders = [p for p in sorted(out_dir.iterdir())
                   if p.is_dir() and not p.name.startswith(".") and p.name != "dem"]
    else:
        if isinstance(products, (str, os.PathLike)):
            products = [products]
        folders = []
        for product in products:
            name = _scene_name(product)
            folder = out_dir / name
            if not (folder / MARKER).is_file() or _read_json(folder / MARKER) is None:
                raise KeyError(f"{name}: no level2 scene of that name in {out_dir}")
            folders.append(folder)
    scenes = []
    for folder in folders:
        marker = _read_json(folder / MARKER)
        if not isinstance(marker, dict) or "datetime" not in marker:
            continue
        scenes.append(dict(name=folder.name, folder=folder, marker=marker))
    if not scenes:
        raise ValueError(f"{out_dir} has no level2 scene (no folder with a {MARKER})")
    return scenes


def _header(path: Path) -> Dict[str, Any]:
    with rasterio.open(path) as src:
        return dict(crs=None if src.crs is None else src.crs.to_wkt(), transform=tuple(src.transform)[:6],
                    shape=(src.height, src.width), count=src.count)


def _check(scenes: List[Dict[str, Any]], like: Any) -> None:
    for s in scenes:
        for tif in ("boa.tif", "qai.tif"):
            if not (s["folder"] / tif).is_file():
                raise FileNotFoundError(f"{s['folder'] / tif} is missing")
        boa, qai = _header(s["folder"] / "boa.tif"), _header(s["folder"] / "qai.tif")
        if boa["count"] != len(BOA_BANDS):
            raise ValueError(f"{s['folder'] / 'boa.tif'} has {boa['count']} bands, a level2 boa has {len(BOA_BANDS)}")
        s["grid"] = (boa["crs"], boa["transform"], boa["shape"])
        s["qai_grid"] = (qai["crs"], qai["transform"], qai["shape"])
    if like is not None:
        return
    count = Counter(s["grid"] for s in scenes)
    best = max(count.values())
    reference = next(s["grid"] for s in scenes if count[s["grid"]] == best)
    odd = [s["name"] for s in scenes if s["grid"] != reference or s["qai_grid"] != s["grid"]]
    if odd:
        raise ValueError("the scenes are not on one grid (same CRS, transform and shape; use like= to put them on one): "
                         + ", ".join(odd))


def _scene(s: Dict[str, Any], mask: Optional[str], like: Any, resampling: str, chunks: Any) -> xr.DataArray:
    boa = load_raster(s["folder"] / "boa.tif", chunks=chunks, like=like, resampling=resampling, masked="auto")
    boa = boa.assign_coords(band=list(BOA_BANDS))
    valid = boa != BOA_NODATA
    if mask is not None:
        qai = load_raster(s["folder"] / "qai.tif", chunks=chunks, like=like, resampling="nearest", masked="auto")
        if qai.chunks is not None and boa.chunks is not None:
            qai = qai.chunk({d: boa.chunksizes[d] for d in ("y", "x")})
        keep = qa_mask(qai, mask)
        keep = keep.drop_vars([c for c in keep.coords if c not in keep.dims], errors="ignore")
        keep = keep.assign_coords(y=boa["y"].values, x=boa["x"].values)
        valid = valid & keep
    out = boa.astype("float32").where(valid)
    out.attrs = {}
    return out


def load_level2_series(out_dir: Any, *, mask: Optional[str] = "force", like: Any = None, resampling: str = "auto",
                       chunks: Any = "auto", products: Any = None) -> xr.DataArray:
    """The scenes written by ``zeit.level2_series`` as one lazy ``(time, band, y, x)`` reflectance cube.

    Opens the scenes already in ``out_dir`` (what ``zeit.level2_series`` or a previous run left there) without computing a
    pixel: the cube is ``float32`` reflectance times 10000, NaN where the scene has no data and, by default, where FORCE's
    default screening of the quality bits (``zeit.qa_mask(qai, "force")``) drops the observation. It goes as it is into
    ``zeit.temporal_metrics``, ``zeit.regularize_time_series`` and ``zeit.ccdc``.

    Parameters
    ----------
    out_dir
        The folder of ``zeit.level2_series``: every sub-folder with a ``.zeit-level2.json`` marker is a scene (folders
        starting with a dot, ``dem`` and folders without a marker are ignored).
    mask
        ``"force"`` (default): NaN also where ``zeit.qa_mask(qai, "force")`` is False, i.e. FORCE's default ``SCREEN_QAI``:
        no data, cloud (buffer, opaque, cirrus), cloud shadow, snow, sub-zero and saturation. ``None``: only the fill
        value (-9999) becomes NaN. Anything else is an error.
    like
        A reference raster (path or ``DataArray``): every scene is put on its grid by ``zeit.load_raster(like=...)``. ``boa``
        takes ``resampling``, the ``qai`` always the nearest cell (its bits cannot be interpolated), and the mask is applied
        after the warp. Without it all the scenes must share one grid (CRS, transform and shape).
    resampling
        With ``like``: ``"auto"`` (nearest, as ``boa`` is an integer product), or a GDAL method as in ``zeit.load_raster``.
    chunks
        Dask chunks of the cube, as in ``zeit.load_raster`` (``"auto"`` or a dict of sizes of ``y``/``x``). The cube is
        always lazy; one scene is one chunk in time.
    products
        Names of the scenes to take (with or without ``.SAFE``), as the ``product`` column of ``zeit.find_l1c``; a name
        that is not in ``out_dir`` is a ``KeyError``. Default: all of them.

    Returns
    -------
    xarray.DataArray
        ``boa`` ``(time, band, y, x)`` ``float32``, dask-backed, with ``time`` the sensing time (``datetime64[ns]``, naive UTC)
        in order, ``band`` BLUE, GREEN, RED, REDEDGE1, REDEDGE2, REDEDGE3, BROADNIR, NIR, SWIR1, SWIR2, the CRS and transform
        of the scenes (``.rio``), and per-scene coordinates along ``time``: ``product``, ``platform``, ``baseline``,
        ``sensor`` and the numbers ``aod_green``, ``water_vapor_cm`` and ``fmask_*`` of the markers (NaN where a scene has
        none). ``attrs`` has ``tile`` and ``source`` (``"zeit.level2"``). NaN is its NoData, so ``.zeit.save`` and ``.plot()``
        work.

    Raises
    ------
    ValueError
        ``out_dir`` has no scene; ``mask`` is not ``"force"`` or None; the scenes are not on one grid and there is no ``like``.
    KeyError
        A name of ``products`` is not a scene of ``out_dir``.

    Notes
    -----
    The mask is lazy and costs one more read (``qai.tif``, 2 bytes a pixel) per scene. A full scene is 10980 x 10980 pixels
    and the cube is 4 bytes x 10 bands a pixel and date: take a window with ``like`` (a small raster) or work chunk by chunk.

    Examples
    --------
    >>> cube = zeit.load_level2_series("l2_23LKC")
    >>> cube.dims, cube.sizes["time"]
    (('time', 'band', 'y', 'x'), 38)
    >>> metrics = zeit.temporal_metrics(cube.sel(band=["RED", "NIR"]), ...)
    >>> clear = zeit.load_level2_series("l2_23LKC", mask=None)     # only the fill is NaN
    """
    if mask not in _MASKS:
        raise ValueError(f"mask must be 'force' or None, got {mask!r}")
    if chunks is None:
        chunks = "auto"
    out_dir = Path(os.path.expanduser(os.fspath(out_dir)))
    scenes = _scenes(out_dir, products)
    _check(scenes, like)
    times = pd.DatetimeIndex([_utc_naive(s["marker"]["datetime"]) for s in scenes]).as_unit("ns")
    order = np.argsort(times.values, kind="stable")
    scenes = [scenes[i] for i in order]
    times = times[order]
    layers = [_scene(s, mask, like, resampling, chunks) for s in scenes]
    cube = xr.concat(layers, dim=pd.Index(times, name="time"), coords="minimal", compat="override", join="override")
    cube = cube.transpose("time", "band", "y", "x").assign_coords(band=list(BOA_BANDS))
    markers = [s["marker"] for s in scenes]
    extra: Dict[str, Any] = {}
    for key in ("product", "platform", "baseline"):
        extra[key] = ("time", [str(m.get(key, "")) for m in markers])
    sensors = [str((m.get("attrs") or {}).get("sensor", "")) for m in markers]
    if any(sensors):
        extra["sensor"] = ("time", sensors)
    keys = list(_NUMERIC_ATTRS) + sorted({k for m in markers for k in (m.get("attrs") or {}) if str(k).startswith("fmask_")})
    for key in keys:
        raw = [(m.get("attrs") or {}).get(key) for m in markers]
        values = [_num(v) for v in raw]
        if any(v is not None for v in values) and all(v is not None or r is None for v, r in zip(values, raw)):
            extra[key] = ("time", np.array([np.nan if v is None else v for v in values], dtype="float64"))
    cube = cube.assign_coords(extra)
    tiles = sorted({str(m["tile"]) for m in markers if m.get("tile")})
    cube.name = "boa"
    cube.attrs = {"source": "zeit.level2"}
    if tiles:
        cube.attrs["tile"] = tiles[0] if len(tiles) == 1 else tiles
    if cube.rio.crs is None:
        first = rasterio.open(scenes[0]["folder"] / "boa.tif")
        with first:
            cube = cube.rio.write_crs(first.crs)
    return cube.rio.write_nodata(np.nan, encoded=True)


def level2_series(tile: Optional[str] = None, start: Any = None, end: Any = None, out_dir: Any = None, *,
                  max_cloud: float = 100.0, products: Any = None, dem: Any = "glo30",
                  level2_kw: Optional[Dict[str, Any]] = None, mask: Optional[str] = "force", like: Any = None,
                  resampling: str = "auto", chunks: Any = "auto", keep_safe: bool = False, safe_dir: Any = None,
                  n_jobs: int = 8, verbose: bool = True) -> xr.DataArray:
    """Sentinel-2 L1C scenes of a tile to surface reflectance, one scene at a time, and the series as one lazy cube.

    The whole chain in one call: ``zeit.find_l1c`` (or the ``products`` given), ``zeit.download_l1c`` and, once for the
    tile, ``zeit.download_dem``, ``zeit.level2`` and ``.zeit.save`` for each scene, then ``zeit.load_level2_series`` over
    the scenes of this run. What is written for a scene is exactly what ``zeit.level2(...).zeit.save(folder)`` writes
    (``boa.tif``, ``qai.tif``, ``dst.tif``...) plus a marker file that makes the scene "done".

    Parameters
    ----------
    tile, start, end
        The MGRS tile (``"23LKC"``) and the dates to search with ``zeit.find_l1c``; not needed with ``products``.
    out_dir
        Where the scenes go, one folder each (named as the product without ``.SAFE``), with the DEM in ``dem/`` and the
        journal ``level2_series.json``. Required.
    max_cloud
        Maximum cloud cover (percent) of the search.
    products
        Instead of searching: product names (or paths of local ``.SAFE`` folders) or a table with a ``product`` column.
    dem
        ``"glo30"`` (default: the Copernicus GLO-30 DEM of the tile, downloaded once), the path of a DEM raster, or None
        with ``level2_kw={"topography": False}``.
    level2_kw
        Options of ``zeit.level2`` (``brdf``, ``aod``, ``fmask``, ``merge``...), not ``safe``, ``dem`` or ``n_jobs``. A folder
        written with other options is an error.
    mask, like, resampling, chunks
        As in ``zeit.load_level2_series``.
    keep_safe
        Keep the downloaded ``.SAFE`` folders (default: deleted once the scene is done).
    safe_dir
        Where the ``.SAFE`` are downloaded (default ``out_dir/.safe``).
    n_jobs
        Threads of each scene's correction (the downloads use at most 4).
    verbose
        Print one line per scene.

    Returns
    -------
    xarray.DataArray
        The cube of ``zeit.load_level2_series`` for the scenes that are done (this run or an earlier one); the scenes that
        failed and the duplicates (an older processing of the same acquisition) are not in it. ``attrs["failed"]`` lists the
        products that failed and ``attrs["journal"]`` is the path of the journal.

    Raises
    ------
    RuntimeError
        No scene succeeded (the message points to the journal).

    Notes
    -----
    A scene is processed at a time and needs about 22 to 29 GiB of RAM at 10 m for a full tile; a disk for about the size of
    the ``boa.tif`` and ``qai.tif`` of every scene (a warning tells when the free space looks short after the first scene).
    The call can be interrupted and repeated: a scene with its marker is skipped, one interrupted is redone from scratch and
    a failed one is tried again. The pixels are not read until the cube is used.

    Only FORCE's default screening of the QAI is applied by ``mask="force"``: no data, cloud (buffer, opaque, cirrus), cloud
    shadow, snow, sub-zero and saturation; water, aerosol, low sun and illumination flags do not mask. Use
    ``mask=None`` and ``zeit.qa_mask`` on the ``qai.tif`` files for another screening.

    Examples
    --------
    >>> cube = zeit.level2_series("23LKC", "2022-01-01", "2023-12-31", "l2_23LKC", max_cloud=60)
    >>> cube.attrs["failed"]                          # [] when every scene worked
    >>> metrics = zeit.temporal_metrics(cube.sel(band=["RED", "NIR"]), ...)
    >>> cube = zeit.load_level2_series("l2_23LKC")    # later: reopen the same scenes
    """
    if mask not in _MASKS:
        raise ValueError(f"mask must be 'force' or None, got {mask!r}")
    table = run_series(tile, start, end, out_dir, max_cloud=max_cloud, products=products, dem=dem, level2_kw=level2_kw,
                       keep_safe=keep_safe, safe_dir=safe_dir, n_jobs=n_jobs, verbose=verbose)
    out_dir = Path(os.path.expanduser(os.fspath(out_dir)))
    journal = out_dir / JOURNAL
    ok = [str(p) for p, s in zip(table["product"], table["status"]) if s in ("done", "skipped")]
    failed = [str(p) for p, s in zip(table["product"], table["status"]) if s == "failed"]
    if not ok:
        raise RuntimeError(f"level2_series: no scene succeeded ({len(failed)} failed); see {journal}")
    cube = load_level2_series(out_dir, products=ok, mask=mask, like=like, resampling=resampling, chunks=chunks)
    cube.attrs["failed"] = failed
    cube.attrs["journal"] = str(journal)
    return cube
