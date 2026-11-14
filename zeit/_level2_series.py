"""``run_series``: ``zeit.level2`` over a series of Sentinel-2 L1C scenes, one scene at a time, resumable.

Only the orchestration of public functions: ``find_l1c`` -> ``download_l1c`` -> ``download_dem`` (once) ->
``level2`` -> ``.zeit.save`` scene by scene. What is written for a scene is exactly what
``level2(safe, dem=..., **level2_kw).zeit.save(folder)`` writes, plus a marker file that makes the scene
"done" (the journal ``level2_series.json`` is only informational, as the one of ``run_tiles``)."""

import hashlib
import inspect
import json
import os
import re
import shutil
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import _level2, _s2fetch

MARKER = ".zeit-level2.json"
JOURNAL = "level2_series.json"
_COLUMNS = ["product", "datetime", "tile", "baseline", "platform", "status", "path", "error", "seconds"]
_NAME = re.compile(r"^(S2[A-Z])_MSIL1C_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})(?:\.SAFE)?$")
_FORBIDDEN_KW = ("safe", "dem", "n_jobs")


def parse_name(name: Any) -> Dict[str, Any]:
    """The fields of a standard L1C product name (or path to one): ``product`` (with ``.SAFE``), ``platform``
    (``S2B``), ``datetime`` (UTC sensing time), ``baseline`` (``"05.09"``), ``n`` (the baseline as an int),
    ``orbit``, ``tile`` (``23LKC``) and ``generated`` (the processing time, the last field)."""
    base = os.fspath(name).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    m = _NAME.match(base.strip())
    if not m:
        raise ValueError(f"{base!r} is not the name of a Sentinel-2 L1C product (such as "
                         "S2B_MSIL1C_20230715T131239_N0509_R138_T23LKC_20230715T145717.SAFE)")
    platform, sensed, n, orbit, tile, generated = m.groups()
    return dict(product=base.strip() if base.strip().endswith(".SAFE") else base.strip() + ".SAFE",
                platform=platform, datetime=pd.Timestamp(datetime.strptime(sensed, "%Y%m%dT%H%M%S"), tz="UTC"),
                baseline=f"{n[:2]}.{n[2:]}", n=int(n), orbit=orbit, tile=tile, generated=generated)


# --- inputs ------------------------------------------------------------------------------------------------------------


def _is_local(item: Any) -> bool:
    return isinstance(item, os.PathLike) or (isinstance(item, str) and os.path.isdir(os.path.expanduser(item)))


def _table(products: Any) -> pd.DataFrame:
    """One row per product with the fields of its name, plus the local path (or None)."""
    rows = []
    if isinstance(products, pd.DataFrame):
        if "product" not in products.columns:
            raise ValueError("products: a DataFrame needs a 'product' column (as find_l1c gives)")
        items = [(r["product"], r) for _, r in products.iterrows()]
    else:
        if isinstance(products, (str, os.PathLike)):
            products = [products]
        items = [(p, None) for p in products]
    for item, row in items:
        local = Path(os.path.expanduser(os.fspath(item))) if _is_local(item) else None
        if local is not None and not local.is_dir():
            raise FileNotFoundError(f"products: {local} is not a directory")
        info = parse_name(local.name if local is not None else item)
        if row is not None:
            for key in ("datetime", "tile", "baseline", "platform"):
                if key in row.index and row[key] is not None and not pd.isna(row[key]):
                    value = row[key]
                    if key == "datetime":
                        value = pd.Timestamp(value)
                        value = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
                    elif key == "tile":
                        value = _s2fetch._tile_code(str(value))
                    info["shown_" + key] = value
        info["local"] = local
        rows.append(info)
    return pd.DataFrame(rows)


def _check_level2_kw(level2_kw: Dict[str, Any]) -> None:
    params = inspect.signature(_level2.level2).parameters
    open_ended = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    for key in level2_kw:
        if key in _FORBIDDEN_KW:
            raise TypeError(f"level2_kw: {key!r} is set by the series itself, not in level2_kw")
        if key not in params and not open_ended:
            raise TypeError(f"level2_kw: {key!r} is not a parameter of zeit.level2")


def _canon(value: Any) -> Any:
    if isinstance(value, np.ndarray) or hasattr(value, "values") and hasattr(value, "dims"):
        arr = np.ascontiguousarray(np.asarray(getattr(value, "values", value)))
        return {"array": [str(arr.dtype), list(arr.shape), hashlib.sha256(arr.tobytes()).hexdigest()]}
    if isinstance(value, dict):
        return {str(k): _canon(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _digest(level2_kw: Dict[str, Any], dem_source: str) -> str:
    text = json.dumps({"level2_kw": _canon(level2_kw), "dem": dem_source}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return str(value)


# --- files -------------------------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- the series --------------------------------------------------------------------------------------------------------


def run_series(tile: Optional[str] = None, start: Any = None, end: Any = None, out_dir: Any = None, *,
               max_cloud: float = 100.0, products: Any = None, dem: Any = "glo30",
               level2_kw: Optional[Dict[str, Any]] = None, keep_safe: bool = False, safe_dir: Any = None,
               n_jobs: int = 8, verbose: bool = True) -> pd.DataFrame:
    """``zeit.level2`` for every scene of a series, one scene at a time; see the module docstring.

    Returns a DataFrame with one row per product (duplicates included), sorted by date: ``product``, ``datetime``,
    ``tile``, ``baseline``, ``platform``, ``status`` (``done``, ``skipped``, ``failed`` or ``duplicate``), ``path`` (the
    scene's folder), ``error`` and ``seconds``."""
    if out_dir is None:
        raise ValueError("run_series: out_dir is required")
    level2_kw = dict(level2_kw or {})
    _check_level2_kw(level2_kw)
    if dem is None:
        dem_source = "none"
        if level2_kw.get("topography", True):
            raise ValueError("run_series: dem=None with topography=True (the default): give a DEM (dem='glo30' or a raster "
                             "path) or level2_kw={'topography': False}")
    elif isinstance(dem, (str, os.PathLike)):
        glo30 = isinstance(dem, str) and dem == "glo30"
        dem_source = "glo30" if glo30 else os.fspath(dem)
        if not glo30 and not os.path.isfile(os.path.expanduser(dem_source)):
            raise FileNotFoundError(f"run_series: the DEM {dem_source} does not exist")
    else:
        raise TypeError("run_series: dem is 'glo30', a raster path or None")

    if products is None:
        if tile is None or start is None or end is None:
            raise ValueError("run_series: give tile, start and end (or products=)")
        found = _s2fetch.find_l1c(tile=tile, start=start, end=end, max_cloud=max_cloud)
        table = _table(found) if len(found) else pd.DataFrame()
    else:
        table = _table(products)
    out_dir = Path(os.path.expanduser(os.fspath(out_dir)))
    if len(table):
        shown_tile = table["shown_tile"] if "shown_tile" in table else table["tile"]
        tiles = sorted({str(t) for t in shown_tile.where(shown_tile.notna(), table["tile"])})
        if len(tiles) > 1:
            raise ValueError(f"run_series: the products span several tiles ({', '.join(tiles)}); the DEM is per tile, "
                             "run one series per tile")
        if tile is not None and products is not None:
            want = _s2fetch._tile_code(tile)
            if tiles[0] != want:
                raise ValueError(f"run_series: the products are of tile {tiles[0]}, not {want}")

    result = _rows(table)
    scenes = [i for i in result.index if result.at[i, "status"] != "duplicate"]
    digest = _digest(level2_kw, dem_source)
    folders = {i: out_dir / result.at[i, "product"][:-len(".SAFE")] for i in result.index}
    resumed = {}
    for i in scenes:
        marker = _read_json(folders[i] / MARKER) if folders[i].is_dir() else None
        if marker is not None and marker.get("params_digest") != digest:
            raise FileExistsError(f"{folders[i]} was written with other options (level2_kw or DEM); choose another out_dir")
        resumed[i] = marker is not None

    out_dir.mkdir(parents=True, exist_ok=True)
    old = _read_json(out_dir / JOURNAL) or {}
    state = {"started": _now(), "finished": None, "done": [], "failed": {}, "dem": old.get("dem"), "scenes": []}
    dem_path: Optional[str] = None
    if dem_source == "glo30":
        if state["dem"] and os.path.isfile(state["dem"]):
            dem_path = state["dem"]
    elif dem is not None:
        dem_path = os.path.expanduser(dem_source)
        state["dem"] = dem_path

    def journal() -> None:
        state["scenes"] = result[["product", "status"]].assign(
            datetime=result["datetime"].astype(str)).to_dict("records")
        try:
            _write_json(out_dir / JOURNAL, state)
        except OSError:
            pass

    def say(text: str) -> None:
        if verbose:
            print(text, flush=True)

    for i in result.index:
        if result.at[i, "status"] == "duplicate":
            say(f"{result.at[i, 'product'][:-5]} duplicate of a newer processing, not processed")
    journal()
    safe_root = Path(os.path.expanduser(os.fspath(safe_dir))) if safe_dir is not None else out_dir / ".safe"
    total, first_size, warned = len(scenes), None, False
    try:
        for k, i in enumerate(scenes, start=1):
            name = result.at[i, "product"][:-len(".SAFE")]
            tag = f"[{k}/{total}] {name}"
            folder, part = folders[i], out_dir / (".part-" + name)
            if resumed[i]:
                result.at[i, "status"], result.at[i, "path"] = "skipped", str(folder)
                state["done"].append(name)
                say(f"{tag} skipped")
                journal()
                continue
            local = table.at[i, "local"] if "local" in table else None
            downloaded: Optional[Path] = None
            t0 = time.time()
            try:
                for stale in (part, folder):     # an interrupted write: redone from scratch
                    if stale.exists():
                        shutil.rmtree(stale, ignore_errors=True)
                if local is not None:
                    safe = Path(local)
                else:
                    safe = downloaded = Path(_s2fetch.download_l1c(result.at[i, "product"], dest=safe_root,
                                                                   n_jobs=min(n_jobs, 4)))
                if dem_source == "glo30" and dem_path is None:
                    dem_path = os.fspath(_s2fetch.download_dem(safe, dest=out_dir / "dem", n_jobs=min(n_jobs, 4)))
                    state["dem"] = dem_path
                ds = _level2.level2(safe, dem=dem_path, n_jobs=n_jobs, **level2_kw)
                ds.zeit.save(part)
                seconds = time.time() - t0
                _write_json(part / MARKER, dict(
                    params_digest=digest, params=dict(level2_kw=_json_safe(level2_kw), dem=dem_source),
                    product=result.at[i, "product"], datetime=str(result.at[i, "datetime"]),
                    tile=result.at[i, "tile"], baseline=result.at[i, "baseline"], platform=result.at[i, "platform"],
                    attrs=_json_safe(dict(ds.attrs)), seconds=round(seconds, 3), zeit_version=_version()))
                os.replace(part, folder)
            except Exception as e:
                shutil.rmtree(part, ignore_errors=True)
                tb = traceback.format_exc().strip().splitlines()[-6:]
                message = f"{type(e).__name__}: {e}"
                result.at[i, "status"], result.at[i, "error"] = "failed", message
                state["failed"][name] = {"error": message, "traceback": tb, "at": _now()}
                say(f"{tag} failed: {message}")
                journal()
                continue
            except BaseException:
                shutil.rmtree(part, ignore_errors=True)
                state["finished"] = _now()
                journal()
                raise
            result.at[i, "status"], result.at[i, "path"], result.at[i, "seconds"] = "done", str(folder), seconds
            state["done"].append(name)
            state["failed"].pop(name, None)
            if downloaded is not None and not keep_safe:
                shutil.rmtree(downloaded, ignore_errors=True)
            say(f"{tag} done in {seconds:.0f} s")
            journal()
            if first_size is None:
                first_size = _size(folder) + (_size(downloaded) if downloaded is not None and keep_safe else 0)
            remaining = sum(1 for j in scenes[k:] if not resumed[j])
            if not warned and remaining:
                free = shutil.disk_usage(out_dir).free
                if free < first_size * remaining:
                    warned = True
                    warnings.warn(f"run_series: the first scene took {first_size / 2**30:.1f} GiB and {remaining} remain "
                                  f"(about {first_size * remaining / 2**30:.1f} GiB), but {free / 2**30:.1f} GiB are free "
                                  f"in {out_dir}", UserWarning, stacklevel=2)
    finally:
        state["finished"] = _now()
        journal()
    failed = [result.at[i, "product"] for i in result.index if result.at[i, "status"] == "failed"]
    if failed:
        warnings.warn(f"run_series: {len(failed)} of {total} scenes failed (run again to retry them; see {out_dir / JOURNAL}): "
                      + ", ".join(failed), UserWarning, stacklevel=2)
    return result[_COLUMNS].reset_index(drop=True)


def _version() -> str:
    from ._tiles._group import version
    return version()


def _rows(table: pd.DataFrame) -> pd.DataFrame:
    """The result frame: sorted by date, duplicates (same acquisition, other processing) marked."""
    if not len(table):
        return pd.DataFrame(columns=_COLUMNS)
    table = table.copy()
    key = table[["platform", "datetime", "orbit", "tile"]].astype(str).agg("|".join, axis=1)
    best = {}
    for i in table.index:
        rank = (table.at[i, "n"], table.at[i, "generated"])
        k = key[i]
        if k not in best or rank > best[k][0]:
            best[k] = (rank, i)
    keep = {i for _, i in best.values()}
    out = pd.DataFrame({
        "product": table["product"],
        "datetime": table["shown_datetime"] if "shown_datetime" in table else table["datetime"],
        "tile": table["shown_tile"] if "shown_tile" in table else table["tile"],
        "baseline": table["shown_baseline"] if "shown_baseline" in table else table["baseline"],
        "platform": table["shown_platform"] if "shown_platform" in table else table["platform"],
        "status": ["pending" if i in keep else "duplicate" for i in table.index],
        "path": None, "error": None, "seconds": np.nan})
    for col in ("datetime", "tile", "baseline", "platform"):
        if f"shown_{col}" in table:
            out[col] = table[f"shown_{col}"].where(table[f"shown_{col}"].notna(), table[col])
    return out.sort_values("datetime", kind="stable")
