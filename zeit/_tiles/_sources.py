"""Tiles that every computer fetches or builds for itself: a recipe per tile instead of a
file. A recipe is data (a kind and its options, checked against a fixed list), never code.

- ``tile_urls``: rasters on a web server or in a bucket (``https://``, ``s3://``...).
- ``stac_tiles``: annual composites from a STAC catalog (``build_annual_composites``).
- ``gee_tiles``: annual medoid composites from Earth Engine (``download_gee_timeseries``).

The machine that processes a tile builds it into its cache, in a process of its own
(``python -m zeit._tiles._sources <recipe.json> <out.tif>``), and keeps it for the next run.
By default an index is written as int16 (the index times 10000, NoData -32768): a quarter
of float64, half of float32, and what LandTrendr reads directly.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

KINDS = ("raster", "stac_annual", "gee_annual")
URL = re.compile(r"^(https?|s3|gs|az|abfs|oss)://", re.I)
STAC_OPTIONS = {"source", "collection", "start_year", "end_year", "season", "bands", "cloud_cover_max", "method",
                "apply_cloud_mask", "resolution", "epsg", "chunksize"}
DTYPES = ("int16", "int32", "float32")


class TileSources:
    """Tiles described by recipes, for ``zeit.run_tiles``: each computer builds the tiles
    it processes (see ``tile_urls``, ``stac_tiles``, ``gee_tiles``)."""

    def __init__(self, recipes: Dict[str, dict]):
        if not recipes:
            raise ValueError("no tiles")
        for name, recipe in recipes.items():
            if not re.fullmatch(r"[\w.-]+", name):
                raise ValueError(f"bad tile name {name!r}: letters, digits, '.', '-' and '_' only")
            check_recipe(recipe)
        self.recipes = dict(recipes)

    def __len__(self) -> int:
        return len(self.recipes)

    def __iter__(self):
        return iter(self.recipes.items())

    def __repr__(self) -> str:
        kinds = sorted({r["kind"] for r in self.recipes.values()})
        return f"TileSources({len(self)} tiles, {', '.join(kinds)})"


def _name(item: Any, i: int) -> str:
    if isinstance(item, str):
        stem = item.rstrip("/").split("/")[-1].split("?")[0] if URL.match(item) else item
        stem = re.sub(r"\.(tif|tiff)$", "", stem, flags=re.I)
        return re.sub(r"[^\w.-]", "_", stem) or f"tile_{i}"
    return f"tile_{i}"


def _named(tiles: Union[Dict[str, Any], Iterable]) -> Dict[str, Any]:
    if isinstance(tiles, dict):
        return {re.sub(r"[^\w.-]", "_", str(k)): v for k, v in tiles.items()}
    out: Dict[str, Any] = {}
    for i, item in enumerate(tiles):
        name = _name(item, i)
        if name in out:
            raise ValueError(f"two tiles named {name}: name them with a dict {{name: tile}}")
        out[name] = item
    return out


def _compact_args(index: Optional[str], scale: float, dtype: Optional[str]) -> dict:
    dtype = dtype or "float32"
    if dtype not in DTYPES:
        raise ValueError(f"dtype: one of {DTYPES}")
    return {"index": index, "scale": float(scale), "dtype": dtype}


def tile_urls(urls: Union[Dict[str, str], Iterable[str]]) -> TileSources:
    """Rasters each computer downloads for itself: ``https://`` (a plain download) or a
    bucket GDAL reads (``s3://``, ``gs://``, ``az://``, with the usual credentials).

    ``urls``: a list (each tile named after its file) or a dict ``{name: url}``."""
    recipes = {}
    for name, url in _named(urls).items():
        recipes[name] = {"kind": "raster", "url": str(url)}
    return TileSources(recipes)


def url_list(inputs):
    """A ``.txt`` file of URLs (one per line, ``#`` comments) as ``tile_urls``; other inputs as
    they are."""
    if isinstance(inputs, (str, Path)) and str(inputs).lower().endswith(".txt") and Path(inputs).is_file():
        urls = [line.strip() for line in Path(inputs).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        return tile_urls(urls)
    return inputs


def stac_tiles(tiles, *, index: Optional[str] = None, scale: float = 10000.0, dtype: Optional[str] = "int16",
               **options) -> TileSources:
    """Annual composites from a STAC catalog, built by each computer for the tiles it
    processes (``zeit.build_annual_composites``), optionally reduced to one index.

    ``tiles``: Sentinel-2 tile ids (``"23KPQ"``), ``(min_lon, min_lat, max_lon, max_lat)``
    boxes, vector or raster files (their extent), or a dict ``{name: one of these}``.
    ``index``: ``"NBR"``, ``"NDVI"``... (``zeit.compute_indices``); None keeps the bands.
    ``scale``, ``dtype``: values are written as ``value * scale`` in ``dtype`` (NoData its
    minimum); ``dtype="float32"`` writes them as they are.
    ``**options``: of ``build_annual_composites``: ``source``, ``collection``,
    ``start_year``, ``end_year``, ``season``, ``bands``, ``cloud_cover_max``, ``method``,
    ``apply_cloud_mask``, ``resolution``, ``epsg``, ``chunksize``. Without ``epsg``, each tile
    is built in its own UTM zone (so that ``resolution``, 30 by default, is in metres)."""
    from ..gee.roi import roi_bounds

    unknown = sorted(set(options) - STAC_OPTIONS)
    if unknown:
        raise TypeError(f"stac_tiles: no option {', '.join(unknown)}; options: {', '.join(sorted(STAC_OPTIONS))}")
    if not options.get("bands"):
        raise ValueError("stac_tiles needs the bands to composite, e.g. bands=['nir08', 'swir22'] for NBR")
    recipes = {}
    for name, tile in _named(tiles).items():
        bbox = [float(v) for v in roi_bounds(tile)]
        tile_options = dict(options)
        tile_options.setdefault("epsg", utm_epsg((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
        recipes[name] = {"kind": "stac_annual", "bbox": bbox, "options": _jsonable(tile_options),
                         **_compact_args(index, scale, dtype)}
    return TileSources(recipes)


def utm_epsg(lon: float, lat: float) -> int:
    zone = min(60, int((lon + 180) // 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def gee_tiles(tiles, *, start_year: int, end_year: int, index: Optional[str] = "NBR", project: Optional[str] = None,
              method: str = "auto", scale: float = 10000.0, dtype: Optional[str] = "int16") -> TileSources:
    """Annual medoid composites from Earth Engine, downloaded by each computer for the tiles
    it processes (``zeit.gee.download_gee_timeseries``, composite_type="annual"). Every
    computer needs Earth Engine credentials (``earthengine authenticate``).

    ``tiles``: Landsat WRS-2 ids (``"217/076"``), Sentinel-2 ids, boxes or a dict
    ``{name: tile}``. ``index``: one index (``"NBR"``), or None for the six SR bands.
    ``scale``, ``dtype``: as in ``stac_tiles``."""
    recipes = {}
    for name, tile in _named(tiles).items():
        roi = tile if isinstance(tile, str) else [float(v) for v in tile]
        recipes[name] = {"kind": "gee_annual", "roi": roi, "start_year": int(start_year), "end_year": int(end_year),
                         "project": project, "method": method, **_compact_args(index, scale, dtype)}
    return TileSources(recipes)


def _jsonable(value):
    return json.loads(json.dumps(value))


def check_recipe(recipe: Any) -> None:
    """Only the known kinds, with options of the right types: what a helper agrees to build."""
    if not isinstance(recipe, dict) or recipe.get("kind") not in KINDS:
        raise ValueError(f"unknown tile recipe: {str(recipe)[:200]}")
    kind = recipe["kind"]
    if kind == "raster":
        if set(recipe) != {"kind", "url"} or not isinstance(recipe["url"], str) or not URL.match(recipe["url"]):
            raise ValueError("a raster recipe takes one URL (https://, s3://, gs://, az://...)")
        return
    if recipe.get("dtype") not in DTYPES or not isinstance(recipe.get("scale"), (int, float)):
        raise ValueError("bad dtype or scale in a tile recipe")
    if recipe.get("index") is not None and not re.fullmatch(r"\w+", str(recipe["index"])):
        raise ValueError("bad index in a tile recipe")
    if kind == "stac_annual":
        if set(recipe) != {"kind", "bbox", "options", "index", "scale", "dtype"}:
            raise ValueError("bad stac_annual recipe")
        if not (isinstance(recipe["bbox"], list) and len(recipe["bbox"]) == 4
                and all(isinstance(v, (int, float)) for v in recipe["bbox"])):
            raise ValueError("bad bbox in a tile recipe")
        if not isinstance(recipe["options"], dict) or set(recipe["options"]) - STAC_OPTIONS:
            raise ValueError("bad options in a tile recipe")
    else:
        if set(recipe) != {"kind", "roi", "start_year", "end_year", "project", "method", "index", "scale", "dtype"}:
            raise ValueError("bad gee_annual recipe")
        roi = recipe["roi"]
        if not (isinstance(roi, str) and re.fullmatch(r"[\w/ ]+", roi)
                or isinstance(roi, list) and len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi)):
            raise ValueError("bad roi in a tile recipe")
        if recipe["method"] not in ("auto", "direct", "drive"):
            raise ValueError("bad method in a tile recipe")


def recipe_key(recipe: dict) -> str:
    from ._group import version

    text = json.dumps({"recipe": recipe, "zeit": version()}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------- building

def _compact(da, scale: float, dtype: str):
    """(DataArray to write, its NoData)."""
    import numpy as np

    if dtype == "float32":
        return da.astype("float32"), None
    info = np.iinfo(dtype)
    nodata = int(info.min)
    values = (da * scale).round().clip(info.min + 1, info.max)
    return values.fillna(nodata).astype(dtype), nodata


def _download(url: str, dest: Path) -> None:
    import shutil
    import urllib.request

    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)


def build(recipe: dict, out: Union[str, Path], cores: int = 0) -> None:
    """Build one tile into ``out`` (a GeoTIFF)."""
    check_recipe(recipe)
    out = Path(out)
    kind = recipe["kind"]
    if kind == "raster":
        url = recipe["url"]
        if url.lower().startswith(("http://", "https://")):
            _download(url, out)
        else:
            import rasterio.shutil

            rasterio.shutil.copy(url, str(out), driver="GTiff", COMPRESS="ZSTD", PREDICTOR="YES", TILED="YES",
                                 BIGTIFF="IF_SAFER")
        return

    from .._save import save_raster

    if kind == "stac_annual":
        import dask

        from ..cube import build_annual_composites
        from ..indices import compute_indices

        cube = build_annual_composites(bbox=recipe["bbox"], **recipe["options"])
        if recipe["index"]:
            cube = compute_indices(cube, [recipe["index"]]).sel(band=recipe["index"], drop=True)
        data, nodata = _compact(cube, recipe["scale"], recipe["dtype"])
        with dask.config.set(num_workers=cores or None):
            save_raster(data, out, nodata=nodata)
        return

    # gee_annual
    import pandas as pd
    import rioxarray
    import xarray as xr

    from ..gee import download_gee_timeseries

    with tempfile.TemporaryDirectory(prefix="zeit-gee-", dir=out.parent) as tmp:
        start, end = recipe["start_year"], recipe["end_year"]
        download_gee_timeseries(recipe["roi"], f"{start}-01-01", f"{end}-12-31", tmp, method=recipe["method"],
                                composite_type="annual", indices=[recipe["index"]] if recipe["index"] else None,
                                project=recipe["project"])
        layers, times = [], []
        for year in range(start, end + 1):
            path = Path(tmp) / f"landsat_medoid_{year}.tif"
            if not path.exists():
                continue
            da = rioxarray.open_rasterio(path, masked=True).load()
            names = [d or f"b{i + 1}" for i, d in enumerate(da.attrs.get("long_name") or ())]
            if recipe["index"]:
                da = da.isel(band=0, drop=True)
            elif names and len(names) == da.sizes["band"]:
                da = da.assign_coords(band=list(names))
            layers.append(da)
            times.append(pd.Timestamp(f"{year}-01-01"))
        if not layers:
            raise RuntimeError(f"Earth Engine returned no composite for {recipe['roi']}")
        cube = xr.concat(layers, dim=pd.DatetimeIndex(times, name="time"))
        data, nodata = _compact(cube, recipe["scale"], recipe["dtype"])
        save_raster(data, out, nodata=nodata)


def run_build(recipe: dict, out: Union[str, Path], cores: int, cancel: Optional[threading.Event] = None,
              log_path: Optional[Union[str, Path]] = None) -> Tuple[bool, float, str]:
    """Build a tile in a process of its own: (built, seconds, the end of its output when not)."""
    from ._tasks import subprocess_env

    out = Path(out)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recipe, f)
        recipe_path = f.name
    log_path = Path(log_path) if log_path else out.with_suffix(".build.log")
    start = time.monotonic()
    try:
        with open(log_path, "wb") as log:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen([sys.executable, "-m", "zeit._tiles._sources", recipe_path, str(out), str(cores)],
                                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=subprocess_env(cores), creationflags=flags)
            try:
                while True:
                    try:
                        code = proc.wait(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        if cancel is not None and cancel.is_set():
                            proc.kill()
                            proc.wait()
                            return False, time.monotonic() - start, "cancelled"
            except BaseException:
                proc.kill()
                proc.wait()
                raise
    finally:
        os.unlink(recipe_path)
    seconds = time.monotonic() - start
    if code == 0 and out.exists():
        return True, seconds, ""
    try:
        tail = Path(log_path).read_text(errors="replace").strip().splitlines()[-5:]
    except OSError:
        tail = []
    return False, seconds, "\n".join(tail) or f"exit code {code}"


if __name__ == "__main__":
    with open(sys.argv[1]) as _f:
        _recipe = json.load(_f)
    try:
        build(_recipe, sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    except Exception as _e:
        print(f"Error building the tile: {type(_e).__name__}: {_e}")
        sys.exit(1)
