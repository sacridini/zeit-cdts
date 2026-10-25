"""``zeit slurm``: the job script that splits a folder of tiles among the tasks of a Slurm job
array, each task running the same command (``zeit <command> <tiles> <output> --tiles``), which
takes its part of the tiles from ``SLURM_ARRAY_TASK_ID`` and every processor of the task."""

import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

from ._shard import STATE, read_split, write_split

MAX_ARRAY = 1000          # Slurm's default MaxArraySize is 1001: ids 0 to 1000
SIGNAL_SECONDS = 120      # SIGTERM this long before the time limit: time to drop the tile and requeue

# The peak memory of a task: the process (Python, zeit, GDAL), in GB, plus the blocks computed at
# once, in bytes per value of the input (pixels x bands). The commands read a tile in blocks of
# --chunk-size (512) and write it one row of blocks at a time, so a task holds the blocks of one
# row, at most one per processor: the width of the tile counts, not its height. Measured with
# `zeit landtrendr` on 2048 x 2048 tiles of 40 years and `zeit ccdc` on 1024 x 1024 tiles of 200
# dates of 6 bands, with 2 to 16 processors; the other commands need less than CCDC.
MEMORY = {"landtrendr": (0.8, 21), "ccdc": (0.8, 32)}
MEMORY_OTHER = MEMORY["ccdc"]
UNKNOWN_BANDS = 300        # a tile not read yet (built by each task): a long series


class JobScript(NamedTuple):
    path: Path
    text: str
    tiles: int
    tasks: int


def suggest_memory(command: str, cpus: int, chunk_size: int = 512, width: Optional[int] = None,
                   bands: Optional[int] = None) -> str:
    """``--mem`` for a task of ``cpus`` processors running ``command`` on tiles of ``width``
    columns and ``bands`` bands (see MEMORY), with a quarter more, in whole GB."""
    base, per_value = MEMORY.get(command, MEMORY_OTHER)
    blocks = cpus if width is None else min(cpus, math.ceil(width / chunk_size))
    gb = base + blocks * chunk_size ** 2 * (bands or UNKNOWN_BANDS) * per_value / 2 ** 30
    return f"{max(2, math.ceil(1.25 * gb))}G"


def tile_shape(path) -> Tuple[Optional[int], Optional[int]]:
    """(columns, bands) of a raster, (None, None) when it cannot be read."""
    try:
        import rasterio

        with rasterio.open(path) as src:
            return src.width, src.count
    except Exception:  # noqa: BLE001 - a guess, then
        return None, None


def write_script(command: str, inputs, output_dir, values: dict, *, cpus: int, mem: Optional[str] = None,
                 time: Optional[str] = None, partition: Optional[str] = None, account: Optional[str] = None,
                 tasks: Optional[int] = None, max_running: Optional[int] = None, sbatch: Sequence[str] = (),
                 setup: Sequence[str] = (), script: Optional[str] = None) -> JobScript:
    """Plan the tiles, keep the split in the output folder and write the job script."""
    from ._sources import url_list
    from ._tasks import forward_args, make_plan

    if cpus < 1:
        raise ValueError("--cpus must be at least 1")
    output_dir = Path(output_dir).resolve()
    plan = make_plan(command, url_list(inputs), output_dir, values)
    n = len(plan.names)
    if n == 0:
        raise ValueError("no tiles")
    tasks = min(n, MAX_ARRAY, tasks if tasks else n)
    if tasks < 1:
        raise ValueError("--tasks must be at least 1")
    args, shared = forward_args(command, values)
    path = Path(script).resolve() if script else output_dir / STATE / f"{command}.sbatch"
    logs = output_dir / STATE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    old = read_split(output_dir)
    if old is not None and (old["count"], old["first"], old["step"]) != (tasks, 0, 1):
        print(f"! {output_dir} was split among {old['count']} tasks before; now among {tasks}. Tasks of the earlier "
              "split still running would take other tiles than this script's.", file=sys.stderr)
    write_split(output_dir, tasks, 0, 1, script=str(path))
    note = []
    if not mem:
        widest = max(plan.tiles + plan.done, key=lambda t: t.size)
        width, bands = (None, None) if plan.recipes else tile_shape(widest.path)
        chunk = int(values.get("chunk_size") or 512)
        mem = suggest_memory(command, cpus, chunk, width, bands)
        blocks = cpus if width is None else min(cpus, math.ceil(width / chunk))
        note = [f"# --mem: suggested for {blocks} blocks of {chunk} x {chunk} pixels at once (a row of the "
                + (f"widest tile, {width} columns of {bands} bands)" if width else "tile, one per processor)")
                + "; a smaller --chunk-size, or fewer --cpus, needs less."]
    per_task = math.ceil(n / tasks)
    source = os.path.abspath(inputs) if isinstance(inputs, (str, os.PathLike)) else str(inputs)
    line = [sys.executable, "-m", "zeit.cli", command, source, str(output_dir), "--tiles", "--requeue", *args]
    for s in shared:
        line += [s["option"], s["path"]]
    array = f"0-{tasks - 1}" + (f"%{max_running}" if max_running else "")
    head = [
        "#!/bin/bash",
        f"# Written by `zeit slurm`: {command} on {n} tiles of {source}, "
        + ("one per task of the array." if per_task == 1 else f"{per_task} or fewer per task of the array."),
        f"#   submit it:    sbatch {path}",
        f"#   follow it:    zeit tiles status {output_dir}",
        "# Each task takes its tiles from SLURM_ARRAY_TASK_ID and every processor it is given; finished tiles",
        "# are kept, and a task stopped at its time limit requeues itself to continue.",
        *note,
        f"#SBATCH --job-name=zeit-{command}",
        f"#SBATCH --array={array}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ]
    if time:
        head.append(f"#SBATCH --time={time}")
    if partition:
        head.append(f"#SBATCH --partition={partition}")
    if account:
        head.append(f"#SBATCH --account={account}")
    head += [
        f"#SBATCH --output={logs / 'slurm-%A_%a.out'}",
        f"#SBATCH --signal=B:TERM@{SIGNAL_SECONDS}",
        "#SBATCH --requeue",
    ]
    head += [f"#SBATCH {extra}" for extra in sbatch]
    body = [""] + list(setup) + ([""] if setup else []) + ["exec " + " ".join(shlex.quote(str(a)) for a in line)]
    text = "\n".join(head + body) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return JobScript(path, text, n, tasks)


def submit(path: Path) -> str:
    """``sbatch <path>``: what it answered (``Submitted batch job 123456``)."""
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is not here: run `zeit slurm --submit` on the cluster's login node, or submit "
                           f"the script there with: sbatch {path}")
    done = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"sbatch failed: {(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


def requeue() -> Optional[str]:
    """Requeue this Slurm job (or array task), to run again from the start of the queue: the
    job id requeued, or None outside a Slurm job. RuntimeError when scontrol refuses."""
    env = os.environ
    if env.get("SLURM_ARRAY_JOB_ID") and env.get("SLURM_ARRAY_TASK_ID"):
        job = f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    elif env.get("SLURM_JOB_ID"):
        job = env["SLURM_JOB_ID"]
    else:
        return None
    done = subprocess.run(["scontrol", "requeue", job], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f"scontrol requeue {job} failed: {(done.stderr or done.stdout).strip()}")
    return job


def split_args(argv: List[str]):
    """``zeit slurm``'s own options apart from the command's: (namespace, the command's options)."""
    import argparse

    p = argparse.ArgumentParser(prog="zeit slurm", description="Write the Slurm job script that splits a folder of "
                                "tiles among the tasks of a job array (and submit it with --submit). The options "
                                "after the output folder that are not zeit slurm's go to the command.")
    add_arguments(p)
    return p.parse_known_args(argv)


def add_arguments(p) -> None:
    p.add_argument("command", help="The command run on every tile: landtrendr, ccdc, mann-kendall, ...")
    p.add_argument("input", help="The folder of tiles (or a quoted glob, or a .txt of URLs)")
    p.add_argument("output_dir", help="Where each tile's outputs go, <output_dir>/<tile name>/")
    p.add_argument("--cpus", type=int, default=16, help="Processors of each task, --cpus-per-task (default: 16)")
    p.add_argument("--mem", default=None, help="Memory of each task, as Slurm takes it (32G); default: suggested from "
                   "the command and --cpus")
    p.add_argument("--time", default=None, help="Time limit of each task (2:00:00, 1-00:00:00); default: the "
                   "partition's")
    p.add_argument("--partition", default=None, help="The partition (queue)")
    p.add_argument("--account", default=None, help="The account to charge")
    p.add_argument("--tasks", type=int, default=None, help="Tasks of the array (default: one per tile, at most 1000)")
    p.add_argument("--max-running", type=int, default=None, help="Tasks running at once (--array=0-49%%N)")
    p.add_argument("--sbatch", action="append", default=[], metavar="OPTION", help="Another #SBATCH line, e.g. "
                   "--sbatch=--qos=long (repeatable)")
    p.add_argument("--setup", action="append", default=[], metavar="LINE", help="A line run before zeit, e.g. "
                   "--setup 'module load gdal' (repeatable)")
    p.add_argument("--script", default=None, help="Where to write the script (default: "
                   "<output_dir>/.zeit-tiles/<command>.sbatch)")
    p.add_argument("--submit", action="store_true", help="Submit it with sbatch")
