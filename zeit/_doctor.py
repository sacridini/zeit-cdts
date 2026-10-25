"""``zeit doctor``: what zeit finds on this computer, the first thing to look at on a cluster
and to paste in an issue: the threads and where their number comes from, the job, the caches
and whether they can be written, the network, the versions."""

import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# What zeit downloads from, by what needs it.
SERVICES: Tuple[Tuple[str, str], ...] = (
    ("STAC, Planetary Computer", "https://planetarycomputer.microsoft.com/api/stac/v1"),
    ("STAC, Earth Search", "https://earth-search.aws.element84.com/v1"),
    ("Earth Engine", "https://earthengine-highvolume.googleapis.com"),
    ("Hugging Face (zeit.ai weights)", "https://huggingface.co"),
    ("TESSERA embeddings (Source Cooperative)", "https://data.source.coop"),
    ("AlphaEarth embeddings (Source Cooperative)", "https://s3.us-west-2.amazonaws.com"),
)

VARIABLES = ("ZEIT_NUM_THREADS", "OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK", "ZEIT_CACHE_DIR", "ZEIT_WEIGHTS",
             "XDG_CACHE_HOME")


def _versions() -> List[str]:
    from ._tiles._group import version

    libc = platform.libc_ver()
    system = f"{platform.system()} {platform.machine()}" + (f", {libc[0]} {libc[1]}" if libc[0] else "")
    lines = [f"zeit {version()} (Python {platform.python_version()}, {system})",
             f"  installed in {os.path.dirname(os.path.abspath(__file__))}"]
    try:
        from . import _core

        lines.append(f"  compiled core: {os.path.basename(_core.__file__)}")
    except ImportError as e:
        lines.append(f"  ! the compiled core does not load: {e}")
    return lines


def _threads() -> List[str]:
    from ._threads import detect

    found = detect()
    lines = [f"Threads (n_jobs=-1): {found.describe()}"]
    given = [f"{v}={os.environ[v]}" for v in VARIABLES[:3] if os.environ.get(v)]
    if given:
        lines.append("  " + ", ".join(given))
    if found.variable == "OMP_NUM_THREADS" and found.job is not None and found.all < found.available:
        lines.append(f"  ! OMP_NUM_THREADS={found.all} leaves {found.available - found.all} of this job's processors "
                     f"idle (cluster modules often set it for MPI programs): set ZEIT_NUM_THREADS="
                     f"{found.available}, which zeit reads first")
    return lines


def _job() -> List[str]:
    from ._tiles._shard import array_task

    found = []
    if os.environ.get("SLURM_JOB_ID"):
        found.append(f"Slurm job {os.environ['SLURM_JOB_ID']}"
                     + (f" on {os.environ['SLURMD_NODENAME']}" if os.environ.get("SLURMD_NODENAME") else ""))
    elif os.environ.get("PBS_JOBID"):
        found.append(f"PBS job {os.environ['PBS_JOBID']}")
    elif os.environ.get("LSB_JOBID"):
        found.append(f"LSF job {os.environ['LSB_JOBID']}")
    if not found:
        return ["Job: none (not inside Slurm, PBS or LSF)"]
    try:
        task = array_task()
    except ValueError as e:
        found.append(f"! {e}")
    else:
        if task is not None:
            step = f" step {task.step}" if task.step != 1 else ""
            found.append(f"array task {task.task} of {task.count} ({task.first}-{task.last}{step})")
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS")
    if gpus:
        found.append(f"GPUs {gpus}")
    return ["Job: " + "; ".join(found)]


def _caches() -> List[str]:
    from ._cache import cache_root, caches, writable

    root = cache_root()
    origin = "ZEIT_CACHE_DIR" if os.environ.get("ZEIT_CACHE_DIR") else (
        "XDG_CACHE_HOME" if os.environ.get("XDG_CACHE_HOME") else "default")
    lines = [f"Caches: {root} ({origin})"]
    for name, path in caches().items():
        state = "writable" if writable(path) else "NOT WRITABLE"
        where = "" if path == root / name else "  (where an older zeit kept it)"
        lines.append(f"  {name:<11} {path}  {state}{where}")
    if os.environ.get("ZEIT_WEIGHTS"):
        lines.append(f"  ZEIT_WEIGHTS={os.environ['ZEIT_WEIGHTS']} (copies of the weights, read first)")
    if not writable(root) and not os.environ.get("ZEIT_CACHE_DIR"):
        lines.append("  ! set ZEIT_CACHE_DIR to a folder that can be written (on a cluster: the scratch space)")
    return lines


def _network(timeout: float) -> List[str]:
    from ._network import OFFLINE_HINT, probe

    with ThreadPoolExecutor(len(SERVICES)) as pool:
        answers = list(pool.map(lambda s: probe(s[1], timeout), SERVICES))
    lines = ["Network:"]
    for (name, url), problem in zip(SERVICES, answers):
        host = url.split("/")[2]
        lines.append(f"  {name:<44} {'ok' if problem is None else 'NO: ' + problem}  ({host})")
    if all(a is not None for a in answers):
        lines.append(f"  ! {OFFLINE_HINT}: `zeit fetch` downloads weights, embeddings and tiles.")
    return lines


def _extras() -> List[str]:
    import importlib.metadata
    import importlib.util

    lines = []
    if importlib.util.find_spec("torch") is None:
        lines.append("zeit.ai: torch is not installed (pip install zeit-cdts[ai])")
    else:
        lines.append(f"zeit.ai: torch {importlib.metadata.version('torch')}")
    credentials = os.path.join(os.path.expanduser("~"), ".config", "earthengine", "credentials")
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        lines.append(f"Earth Engine: a service account (GOOGLE_APPLICATION_CREDENTIALS="
                     f"{os.environ['GOOGLE_APPLICATION_CREDENTIALS']})")
    elif os.path.exists(credentials):
        lines.append(f"Earth Engine: credentials in {credentials}")
    else:
        lines.append("Earth Engine: no credentials (earthengine authenticate, on a computer with a browser)")
    return lines


def report(network: bool = True, timeout: float = 5.0) -> str:
    """What ``zeit doctor`` prints."""
    lines = _versions() + _threads() + _job() + _caches()
    if network:
        lines += _network(timeout)
    lines += _extras()
    return "\n".join(lines)


def doctor(network: bool = True) -> None:
    """Print what zeit finds on this computer: the threads it uses and where that number comes
    from, the batch job (Slurm, PBS, LSF) it runs in, its caches and whether they can be
    written, whether the services it downloads from answer, and the versions. Also
    ``zeit doctor`` on the command line."""
    print(report(network=network), file=sys.stdout, flush=True)
