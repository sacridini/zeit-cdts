"""The work of a tile run: which commands run per tile, their options, the plan (the tiles,
what is already done), running one tile through the zeit command line, and moving its
outputs around as a tar archive.

Every tile, on this computer or on a helper, runs as ``python -m zeit.cli <command> <tile>
<output dir> <options> --jobs <cores>`` in its own process: a crash costs one tile, and a
helper only ever runs zeit's own commands.
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# The commands that read one raster and write into an output folder.
TILE_COMMANDS = ("landtrendr", "ccdc", "bfast-monitor", "bfast-lite", "bfast", "mann-kendall", "phenology",
                 "smooth", "tmask", "twdtw", "snic", "classify", "som", "coded", "bandpass-adjust")
# Options naming one file used by every tile: sent to the helpers along with the task.
SHARED_FILES = {"ccdc": ("dates_file",), "twdtw": ("patterns",), "classify": ("model",), "coded": ("training",)}
# Options naming a file that belongs to one raster: they make no sense for many tiles.
PER_TILE_FILES = {"phenology": ("weights",), "smooth": ("weights",), "classify": ("samples",)}
# The tile run's own options, never forwarded to the command.
RUN_OPTIONS = {"help", "tiles", "helpers", "no_local", "local_cores", "port", "jobs"}

SUFFIXES = (".tif", ".tiff")
MARKER = ".zeit-tile.json"
PARTIAL = ".zeit-partial"
_PROBE = 1 << 20


def command_name(command: str) -> str:
    name = command.replace("_", "-")
    if name not in TILE_COMMANDS:
        raise ValueError(f"{command!r} does not run per tile; one of: {', '.join(TILE_COMMANDS)}")
    return name


def _subparser(command: str) -> argparse.ArgumentParser:
    from .. import cli

    return cli.build_parser().zeit_commands[command]


def option_values(command: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Python keyword arguments (``max_segments=6``) as the command's option values."""
    sub = _subparser(command)
    dests = {a.dest for a in sub._actions if a.option_strings} - RUN_OPTIONS
    unknown = sorted(set(params) - dests)
    if unknown:
        raise TypeError(f"{command} has no option {', '.join(unknown)}; its options: {', '.join(sorted(dests))}")
    return dict(params)


def forward_args(command: str, values: Dict[str, Any]) -> Tuple[List[str], List[dict]]:
    """The command line options for ``values`` (option dest -> value; defaults left out),
    and the shared files apart: [{"option": "--patterns", "path": ...}]."""
    sub = _subparser(command)
    args: List[str] = []
    shared: List[dict] = []
    for action in sub._actions:
        if not action.option_strings or action.dest in RUN_OPTIONS or action.dest not in values:
            continue
        value = values[action.dest]
        if value == action.default:
            continue
        option = max(action.option_strings, key=len)
        if action.dest in PER_TILE_FILES.get(command, ()):
            raise ValueError(f"{option} names a file of one raster; it cannot be used with many tiles")
        if action.dest in SHARED_FILES.get(command, ()):
            path = Path(value)
            if not path.is_file():
                raise FileNotFoundError(f"{option}: no file {value}")
            shared.append({"option": option, "path": str(path.resolve())})
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                args.append(option)
        elif isinstance(action, argparse._StoreFalseAction):
            if not value:
                args.append(option)
        elif action.nargs in ("+", "*") or isinstance(action.nargs, int):
            args.append(option)
            args.extend(str(v) for v in (value if isinstance(value, (list, tuple)) else [value]))
        else:
            args.extend([option, str(value)])
    check_args(command, args + [a for s in shared for a in (s["option"], s["path"])])
    return args, shared


def check_args(command: str, args: Sequence[str], *, received: bool = False) -> None:
    """Parse the options as the command would: raises ValueError instead of exiting.
    ``received``: options from a coordinator, which must not name files (those come as
    shared files) nor start a tile run of their own."""
    import contextlib
    import io

    from .. import cli

    if command not in TILE_COMMANDS:
        raise ValueError(f"{command!r} does not run per tile")
    parser = cli.build_parser()
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            ns = parser.parse_args([command, "input.tif", "output", *args])
    except SystemExit:
        message = err.getvalue().strip().splitlines()
        raise ValueError(f"{command}: {message[-1] if message else 'invalid options'}") from None
    if received:
        if ns.tiles or ns.helpers or ns.no_local or ns.local_cores is not None:
            raise ValueError("a tile's options cannot start a tile run")
        for dest in SHARED_FILES.get(command, ()) + PER_TILE_FILES.get(command, ()):
            if getattr(ns, dest, None) is not None:
                raise ValueError(f"a tile's options cannot name files ({dest})")


def shared_files(path: str) -> List[str]:
    """A shared file and its sidecars (a shapefile's .dbf, .shx, .prj, ...)."""
    p = Path(path)
    if p.suffix.lower() != ".shp":
        return [str(p)]
    return sorted(str(q) for q in p.parent.glob(glob.escape(p.stem) + ".*") if q.is_file())


def list_tiles(inputs: Union[str, os.PathLike, Iterable]) -> List[Path]:
    """A folder (its .tif files), a glob pattern, a file, or a list of them."""
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    files: List[Path] = []
    for item in inputs:
        text = str(item)
        path = Path(text)
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
            if not found:
                raise FileNotFoundError(f"no .tif files in {path}")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        elif glob.has_magic(text):
            found = sorted(Path(p) for p in glob.glob(text) if Path(p).is_file())
            if not found:
                raise FileNotFoundError(f"no files match {text}")
            files.extend(found)
        else:
            raise FileNotFoundError(f"no file or folder {text}")
    stems: Dict[str, Path] = {}
    for f in files:
        if f.stem in stems and stems[f.stem].resolve() != f.resolve():
            raise ValueError(f"two tiles named {f.stem}: {stems[f.stem]} and {f} (their outputs would collide)")
        stems[f.stem] = f
    return list({f.resolve(): f for f in files}.values())


def probe(path: Union[str, Path], size: Optional[int] = None) -> str:
    """A cheap content check: SHA-256 of the first and last MiB and the size."""
    size = os.path.getsize(path) if size is None else size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_PROBE))
        if size > _PROBE:
            f.seek(max(_PROBE, size - _PROBE))
            h.update(f.read(_PROBE))
    return h.hexdigest()


def network_path(path: Path) -> str:
    """The path as other computers may see it: a mapped Windows drive as its UNC path."""
    text = str(path.resolve())
    if text.startswith("\\\\?\\UNC\\"):
        text = "\\\\" + text[8:]
    if sys.platform == "win32" and len(text) > 1 and text[1] == ":":
        try:
            import ctypes

            buf = ctypes.create_unicode_buffer(1024)
            n = ctypes.c_ulong(len(buf))
            if ctypes.windll.mpr.WNetGetConnectionW(text[:2], buf, ctypes.byref(n)) == 0:
                text = buf.value.rstrip("\\") + text[2:]
        except Exception:
            pass
    return text


@dataclass
class Tile:
    id: int
    path: Path
    stem: str
    size: int
    mtime_ns: int
    probe: str = ""
    shared_path: str = ""
    recipe: Optional[dict] = None       # a tile each computer builds for itself (see _sources)
    compressed: Optional[bool] = None   # known once checked, for the transfers

    @property
    def key(self) -> str:
        """Names the tile in a helper's cache: changes whenever the file (or recipe) does."""
        if self.recipe is not None:
            from ._sources import recipe_key

            return recipe_key(self.recipe)
        return hashlib.sha256(f"{self.path.name}|{self.size}|{self.mtime_ns}|{self.probe}".encode()).hexdigest()[:32]


@dataclass
class Plan:
    command: str
    args: List[str]
    shared: List[dict]
    output_dir: Path
    tiles: List[Tile]
    done: List[Tile] = field(default_factory=list)
    digest: str = ""
    recipes: bool = False               # the tiles are built by whoever processes them


def params_digest(command: str, args: Sequence[str], shared: Sequence[dict]) -> str:
    files = []
    for s in shared:
        st = os.stat(s["path"])
        files.append([s["option"], Path(s["path"]).name, st.st_size, probe(s["path"], st.st_size)])
    text = json.dumps({"command": command, "args": list(args), "shared": files}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def make_plan(command: str, inputs, output_dir, values: Dict[str, Any]) -> Plan:
    from ._sources import TileSources

    command = command_name(command)
    args, shared = forward_args(command, values)
    output_dir = Path(output_dir)
    digest = params_digest(command, args, shared)
    if isinstance(inputs, TileSources):
        # Sizes are unknown until built: every tile weighs the same.
        tiles = [Tile(i, Path(f"{name}.tif"), name, 1, 0, recipe=recipe) for i, (name, recipe) in enumerate(inputs)]
    else:
        tiles = []
        for i, path in enumerate(list_tiles(inputs)):
            st = path.stat()
            tiles.append(Tile(i, path, path.stem, st.st_size, st.st_mtime_ns))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = Plan(command, args, shared, output_dir, [], digest=digest, recipes=isinstance(inputs, TileSources))
    conflicts = []
    for tile in tiles:
        target = output_dir / tile.stem
        if target.exists():
            marker = _read_marker(target)
            if (marker is not None and marker.get("params") == digest
                    and marker.get("source") == (tile.key if tile.recipe is not None else None)):
                plan.done.append(tile)
                continue
            conflicts.append(tile.stem)
            continue
        plan.tiles.append(tile)
    if conflicts:
        shown = ", ".join(conflicts[:5]) + (" ..." if len(conflicts) > 5 else "")
        raise FileExistsError(f"{output_dir} already has results of other options (or not from a tile run) for "
                              f"{len(conflicts)} tile(s): {shown}. Use another output folder, or delete them.")
    plan.tiles.sort(key=lambda t: t.size, reverse=True)    # the largest first
    return plan


def _read_marker(folder: Path) -> Optional[dict]:
    try:
        return json.loads((folder / MARKER).read_text())
    except (OSError, ValueError):
        return None


def finalize(plan: Plan, tile: Tile, folder: Path, info: dict) -> bool:
    """Move a finished tile's outputs into place, atomically; False if already there."""
    target = plan.output_dir / tile.stem
    if target.exists():
        return False
    folder.mkdir(parents=True, exist_ok=True)
    source = {"source": tile.key, "recipe": tile.recipe} if tile.recipe is not None else {"tile": str(tile.path)}
    (folder / MARKER).write_text(json.dumps({"params": plan.digest, "command": plan.command, "args": plan.args,
                                             **source, **info}, indent=1))
    os.replace(folder, target)
    return True


def partial_dir(plan: Plan, name: str) -> Path:
    path = plan.output_dir / PARTIAL / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_partial(plan: Plan) -> None:
    import shutil

    shutil.rmtree(plan.output_dir / PARTIAL, ignore_errors=True)


def subprocess_env(cores: int) -> Dict[str, str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(cores)
    env["PYTHONUNBUFFERED"] = "1"
    if sys.platform == "darwin":
        # torch and zeit's extension each bring an OpenMP runtime (see the parallel tutorial).
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    return env


def execute(command: str, tile_path: Union[str, Path], out_dir: Union[str, Path], args: Sequence[str],
            shared: Sequence[Tuple[str, str]], cores: int, cancel: Optional[threading.Event] = None,
            log_path: Optional[Union[str, Path]] = None) -> Tuple[bool, float, str]:
    """Run one tile: (finished, seconds, the end of its output when it failed)."""
    cmd = [sys.executable, "-m", "zeit.cli", command, str(tile_path), str(out_dir), *args]
    for option, path in shared:
        cmd.extend([option, str(path)])
    cmd.extend(["--jobs", str(cores)])
    log_path = Path(log_path) if log_path else Path(out_dir).with_name(Path(out_dir).name + ".log")
    start = time.monotonic()
    with open(log_path, "wb") as log:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
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
        except BaseException:      # Ctrl+C: never leave the command running
            proc.kill()
            proc.wait()
            raise
    seconds = time.monotonic() - start
    if code == 0:
        return True, seconds, ""
    try:
        tail = Path(log_path).read_text(errors="replace").strip().splitlines()[-5:]
    except OSError:
        tail = []
    return False, seconds, "\n".join(tail) or f"exit code {code}"


def is_compressed(path: Union[str, Path]) -> bool:
    import rasterio

    try:
        with rasterio.open(path) as src:
            return src.compression is not None
    except Exception:
        return True       # not a raster GDAL reads: send it as it is


def compress_copy(src: Union[str, Path], dst: Union[str, Path], threads: int = 2) -> None:
    """A lossless copy for the network: ZSTD (DEFLATE where GDAL lacks it) with the
    predictor of the data type, every band, description, tag and NoData kept."""
    import rasterio
    import rasterio.shutil

    with rasterio.open(src) as s:
        predictor = 3 if s.dtypes[0].startswith("float") else 2
    options = dict(driver="GTiff", PREDICTOR=predictor, TILED="YES", BLOCKXSIZE=512, BLOCKYSIZE=512,
                   BIGTIFF="IF_SAFER", NUM_THREADS=str(threads))
    try:
        rasterio.shutil.copy(str(src), str(dst), COMPRESS="ZSTD", ZSTD_LEVEL=1, **options)
    except Exception:
        rasterio.shutil.copy(str(src), str(dst), COMPRESS="DEFLATE", ZLEVEL=1, **options)


def check_placed(folder: Path) -> bool:
    """Results a helper wrote into the output folder: regular files only, no marker."""
    if not folder.is_dir() or folder.is_symlink():
        return False
    for path in folder.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()) or path.name == MARKER:
            return False
    return True


def pack(folder: Union[str, Path], archive: Union[str, Path]) -> int:
    """The files of an output folder as an (uncompressed: GeoTIFFs already are) tar."""
    folder = Path(folder)
    with tarfile.open(archive, "w") as tar:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=path.relative_to(folder).as_posix(), recursive=False)
    return os.path.getsize(archive)


def unpack(archive, dest: Union[str, Path]) -> None:
    """Extract a helper's outputs: regular files only, never outside ``dest``."""
    dest = Path(dest)
    root = dest.resolve()
    with tarfile.open(fileobj=archive, mode="r|") if hasattr(archive, "read") else tarfile.open(archive) as tar:
        for member in tar:
            name = member.name
            parts = Path(name).parts
            if (not member.isfile() or name.startswith(("/", "\\")) or ".." in parts or ":" in name
                    or not parts or Path(name).name == MARKER):
                raise ValueError(f"refusing the archive member {name!r}")
            target = (dest / name).resolve()
            if root not in target.parents:
                raise ValueError(f"refusing the archive member {name!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            with open(target, "wb") as f:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} TB"


def format_seconds(s: float) -> str:
    if s < 9.95:
        return f"{s:.1f} s"
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s" if s < 600 else f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"
