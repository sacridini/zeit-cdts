"""What the runs of an output folder did, kept in it (``<output>/.zeit-tiles/``) so that
``zeit tiles status`` can tell, across the tasks of a job array, the tiles done, failed (with
their error and log) and still missing, and which tasks to submit again.

- ``tiles.json``: the run's command, options and every tile, in the order of the split;
- ``split.json``: how a job array splits them (see ``_shard``);
- ``runs/<run>.json``: one per run (a task of an array, a process): its tiles, those it
  finished, those that failed and why, and whether it was stopped;
- ``logs/<tile>.log``: the output of a tile that failed here.
"""

import json
import os
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ._shard import STATE, Shard, read_split

MANIFEST = "tiles.json"
RUNS = "runs"
LOGS = "logs"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _write_json(path: Path, data: dict) -> None:
    """Atomically: a reader never sees half a file, and runs writing the same file at once
    leave one whole copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}")
    part.write_text(json.dumps(data, indent=1))
    os.replace(part, path)


def write_manifest(plan) -> None:
    """Every tile of the plan's inputs, in the order the split follows."""
    _write_json(plan.output_dir / STATE / MANIFEST, {"command": plan.command, "args": plan.args,
                                                       "params": plan.digest, "tiles": plan.names})


class Journal:
    """The record of one run, rewritten as tiles finish."""

    def __init__(self, plan, shard: Optional[Shard] = None):
        self.folder = plan.output_dir / STATE
        self.path = self.folder / RUNS / f"{plan.run}.json"
        self._lock = threading.Lock()
        self.record = {"run": plan.run, "host": socket.gethostname(), "pid": os.getpid(), "started": _now(),
                       "command": plan.command, "params": plan.digest,
                       "shard": [shard.index, shard.count] if shard is not None else None,
                       "tiles": [t.stem for t in plan.tiles], "skipped": [t.stem for t in plan.done],
                       "done": [], "failed": {}, "stopped": None, "ended": None}
        self._save()

    def _save(self) -> None:
        try:
            _write_json(self.path, self.record)
        except OSError:
            pass      # the record helps; it must never stop the tiles

    def done(self, stem: str) -> None:
        with self._lock:
            self.record["done"].append(stem)
            self.record["failed"].pop(stem, None)
            self._save()

    def failed(self, stem: str, error: str, log: Optional[Path] = None) -> None:
        with self._lock:
            entry = {"error": error.strip()[-4000:], "at": _now()}
            if log is not None and Path(log).is_file():
                kept = self.folder / LOGS / f"{stem}.log"
                try:
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(log, kept)
                    entry["log"] = str(kept)
                except OSError:
                    pass
            self.record["failed"][stem] = entry
            self._save()

    def end(self, stopped: Optional[str] = None) -> None:
        with self._lock:
            self.record["stopped"] = stopped
            self.record["ended"] = _now()
            self._save()


@dataclass
class Status:
    """The tiles of an output folder, across every run that wrote into it."""
    output_dir: Path
    command: str
    tiles: List[str]
    done: List[str] = field(default_factory=list)
    failed: Dict[str, dict] = field(default_factory=dict)      # tile -> error, run, log
    missing: List[str] = field(default_factory=list)
    running: Dict[str, str] = field(default_factory=dict)      # tile -> run, of runs not ended
    split: Optional[dict] = None

    def tasks(self, stems: List[str]) -> List[int]:
        """The array task ids that process ``stems`` (with the split the folder keeps)."""
        if not self.split:
            return []
        position = {name: i for i, name in enumerate(self.tiles)}
        count, first, step = self.split["count"], self.split["first"], self.split["step"]
        return sorted({first + (position[s] % count) * step for s in stems if s in position})


def ranges(ids: List[int]) -> str:
    """``[3, 4, 5, 9]`` -> ``"3-5,9"``, as Slurm's --array takes them."""
    out: List[str] = []
    for i in sorted(set(ids)):
        if out:
            last = out[-1].split("-")
            if int(last[-1]) + 1 == i:
                out[-1] = f"{last[0]}-{i}"
                continue
        out.append(str(i))
    return ",".join(out)


def _marker_params(folder: Path) -> Optional[str]:
    from ._tasks import MARKER

    try:
        return json.loads((folder / MARKER).read_text()).get("params")
    except (OSError, ValueError, AttributeError):
        return None


def read_status(output_dir: Union[str, Path]) -> Status:
    """What the runs of ``output_dir`` did. FileNotFoundError when no tile run wrote there."""
    output_dir = Path(output_dir)
    try:
        manifest = json.loads((output_dir / STATE / MANIFEST).read_text())
    except (OSError, ValueError):
        raise FileNotFoundError(f"{output_dir} has no record of a tile run ({STATE}/{MANIFEST})") from None
    status = Status(output_dir, manifest.get("command", "?"), list(manifest.get("tiles", [])),
                    split=read_split(output_dir))
    params = manifest.get("params")
    records = []
    for path in sorted((output_dir / STATE / RUNS).glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    records.sort(key=lambda r: str(r.get("started", "")))
    failures: Dict[str, Tuple[str, dict]] = {}
    for r in records:
        for stem, entry in (r.get("failed") or {}).items():
            failures[stem] = (str(r.get("run")), entry)
        for stem in r.get("done") or []:
            failures.pop(stem, None)
        if r.get("ended") is None:
            for stem in r.get("tiles") or []:
                if stem not in (r.get("done") or []) and stem not in (r.get("failed") or {}):
                    status.running[stem] = str(r.get("run"))
    for stem in status.tiles:
        if _marker_params(output_dir / stem) == params:
            status.done.append(stem)
            status.running.pop(stem, None)
        elif stem in failures:
            run, entry = failures[stem]
            status.failed[stem] = {**entry, "run": run}
            status.running.pop(stem, None)
        else:
            status.missing.append(stem)
    return status


def describe(status: Status, *, script: Optional[str] = None, shown: int = 10) -> str:
    """What ``zeit tiles status`` prints."""
    n = len(status.tiles)
    lines = [f"{status.command} in {status.output_dir}: {n} tiles, {len(status.done)} done, "
             f"{len(status.failed)} failed, {len(status.missing)} not done"
             + (f" ({len(status.running)} of them in runs that have not ended)" if status.running else "")]
    for stem, entry in list(status.failed.items())[:shown]:
        error = (entry.get("error") or "").strip().splitlines()
        lines.append(f"  failed: {stem} (run {entry.get('run')}): {error[-1] if error else 'no message'}"
                     + (f"\n          log: {entry['log']}" if entry.get("log") else ""))
    if len(status.failed) > shown:
        lines.append(f"  ... and {len(status.failed) - shown} more failed")
    missing = [s for s in status.missing if s not in status.running]
    if missing:
        more = f" ... and {len(missing) - shown} more" if len(missing) > shown else ""
        lines.append(f"  not started: {', '.join(missing[:shown])}{more}")
    if status.running:
        names = sorted(status.running)
        more = f" ... and {len(names) - shown} more" if len(names) > shown else ""
        lines.append(f"  in runs that have not ended (still running, or killed): {', '.join(names[:shown])}{more}")
    left = list(status.failed) + status.missing
    if not left:
        lines.append("Every tile is done.")
    elif status.split:
        tasks = ranges(status.tasks(left))
        target = script or "<the job script>"
        lines.append(f"To process the {len(left)} left: sbatch --array={tasks} {target}")
    else:
        lines.append(f"To process the {len(left)} left: run the same command again (finished tiles are kept).")
    return "\n".join(lines)
