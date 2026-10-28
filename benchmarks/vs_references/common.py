"""What the vs_references benchmarks share. They re-time Zeit's side of the speed comparisons of
docs/benchmarks/performance.md (taken with the C++ core of zeit 1.x, then called cdts) with the Rust core, on the
same workloads and the same machine, with the protocol of the original scripts of the cdts_vs_originals suite. The
reference tools (R, GDL, Octave, snic.c) are not run again: their stored timings stay; the reference outputs stored
by that suite are used to check that Zeit still agrees with them. Only minisom (pure Python, measured on another
machine) is run again.

Every script imports this module first: it sets the thread variables before numpy and zeit are loaded.

- At most 8 threads: ZEIT_NUM_THREADS=8, the "multi-core" runs pass n_jobs=8; numpy's OpenBLAS keeps to one.
- Out of Windows' power throttling (EcoQoS), one-thread runs pinned to logical CPU 0.
- Before every timed step the machine must have been quiet for 10 s (no cargo/rustc process, under 25% load from
  other processes), and a step during which a build started is thrown away and run again (``timed``). A watchdog
  process (watch.py) samples the machine twice a second for this.
- Results: benchmarks/results/vs-references-rust.json, one entry per row of the tables (``record``).

The suite's data come from CDTS_VS_ORIGINALS (default: ../cdts_vs_originals next to this repository).
"""
import atexit
import ctypes
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

THREADS = 8
os.environ["ZEIT_NUM_THREADS"] = str(THREADS)
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINALS = Path(os.environ.get("CDTS_VS_ORIGINALS", REPO.parent / "cdts_vs_originals")).resolve()
DATA = ORIGINALS / "data"
RESULTS = REPO / "benchmarks" / "results" / "vs-references-rust.json"

QUIET_SECONDS = 10.0
QUIET_LOAD = 25.0


# ---------------------------------------------------------------- the machine

def no_power_throttling():
    """Takes this process out of Windows' power throttling (as benchmarks/bench_core.py does)."""
    if sys.platform != "win32":
        return False
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 1 | 4, 0)  # execution speed and timer resolution: not throttled
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def set_affinity(cpus):
    """Restricts the process to `cpus` (all with None), as benchmarks/bench_core.py does."""
    every = list(range(os.cpu_count()))
    cpus = every if cpus is None else cpus
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
        return True
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        mask = sum(1 << c for c in cpus)
        return bool(k32.SetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.c_size_t(mask)))
    return False


def pin(n_jobs):
    """One thread: logical CPU 0; more: every CPU."""
    set_affinity([0] if n_jobs == 1 else None)


_watch = {}


def start():
    """Call at the start of every timing process: power throttling off, the watchdog started."""
    no_power_throttling()
    if "proc" in _watch:
        return
    log = Path(tempfile.gettempdir()) / f"zeit-vs-references-watch-{os.getpid()}.log"
    log.write_text("")
    proc = subprocess.Popen([sys.executable, str(HERE / "watch.py"), str(os.getpid()), str(log)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _watch.update(proc=proc, log=log)
    atexit.register(_stop)
    time.sleep(1.5)


def _stop():
    proc = _watch.get("proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
    try:
        _watch["log"].unlink()
    except Exception:
        pass


def _samples(since):
    out = []
    for line in _watch["log"].read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and float(parts[0]) >= since:
            out.append((float(parts[0]), int(parts[1]), float(parts[2])))
    return out


def quiet():
    """Waits until, for the last 10 s, no cargo/rustc process existed and other processes kept the load under 25%."""
    t_wait = time.time()
    said = 0.0
    while True:
        s = _samples(time.time() - QUIET_SECONDS)
        if len(s) >= 0.8 * QUIET_SECONDS / 0.5 and not any(c for _, c, _ in s) and max(l for *_, l in s) < QUIET_LOAD:
            return time.time() - t_wait
        if time.time() - said > 60 and time.time() - t_wait > 5:
            last = s[-1] if s else None
            print(f"   (waiting for a quiet machine: {time.time() - t_wait:.0f} s; last sample {last})", flush=True)
            said = time.time()
        time.sleep(1.0)


def disturbed(t0, t1):
    """Whether a build ran, or other processes loaded the machine (mean over 25%), between t0 and t1."""
    time.sleep(1.0)  # the sample that covers t1
    s = [x for x in _samples(t0 - 0.5) if x[0] <= t1 + 1.0]
    if any(c for _, c, _ in s):
        return "a cargo/rustc process ran"
    if s and sum(l for *_, l in s) / len(s) > QUIET_LOAD:
        return f"other processes loaded the machine ({sum(l for *_, l in s) / len(s):.0f}%)"
    return None


def spin(seconds=0.5):
    """Keeps the processor busy a moment: after the idle wait of quiet() the core is at a low clock, and a short
    cold pass (40 calls of 0.1 ms) came out 2-3 times slower for it. The original scripts ran theirs right after
    other Python work; this restores that without running the measured code."""
    t = time.perf_counter()
    while time.perf_counter() - t < seconds:
        pass


def timed(fn, label=""):
    """fn() on a quiet machine, again until no build disturbed it; returns fn's result."""
    while True:
        quiet()
        spin()
        t0 = time.time()
        out = fn()
        t1 = time.time()
        why = disturbed(t0, t1)
        if why is None:
            return out
        print(f"   {label}: discarded ({why}), timing again", flush=True)


# ---------------------------------------------------------------- timing helpers

def per_unit_ms(fn, n):
    """One pass of fn (n units), in ms per unit."""
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) / n * 1000.0


def warm_best(fn, repeat=3):
    """The best of `repeat` runs of fn after a warm-up, in s (the warm-up included in the quiet window)."""
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def once(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- what was published, what is kept

def core_build():
    """The compiled core these timings measure (file, size, modification time) and the repository's commit."""
    import zeit._core as core
    p = Path(core.__file__)
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = ""
    st = p.stat()
    return {"file": p.name, "bytes": st.st_size,
            "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "commit": commit}


def record(entry):
    """Keeps one row of the tables (replacing the row of the same table and name)."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"results": []}
    data["description"] = (
        "Zeit's side of docs/benchmarks/performance.md re-timed with the Rust core (benchmarks/vs_references). "
        "published_cpp_*: the C++ core of zeit 1.x as published (i5-13600K, except SOM: Ryzen 7 7730U laptop); "
        "rust_*: this run; reference_*: the stored timings of the reference tools (not run again, except minisom). "
        "Multi-core: the published numbers used 19-20 threads (15 for SOM), these runs 8 (ZEIT_NUM_THREADS=8). "
        "ratio_rust_cpp < 1: Rust faster.")
    data["machine"] = {"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count(),
                       "python": sys.version.split()[0], "threads_multi": THREADS, "single_thread_cpu": 0}
    entry = dict(entry)
    entry.setdefault("core", core_build())
    entry.setdefault("date", datetime.date.today().isoformat())
    data["results"] = [r for r in data["results"]
                       if not (r["table"] == entry["table"] and r["row"] == entry["row"])]
    data["results"].append(entry)
    order = {"single-core": 0, "multi-core": 1}
    data["results"].sort(key=lambda r: (order.get(r["table"], 2), r.get("order", 99), r["row"]))
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=1))
    r1 = entry.get("rust_1t")
    r8 = entry.get("rust_8t")
    print(f"-> {entry['table']:<11} {entry['row']:<28} published C++ {entry.get('published_cpp_1t')} | "
          f"Rust 1T {r1:.4g}" + (f" | Rust 8T {r8:.4g}" if r8 is not None else "") + f" {entry.get('unit', '')}",
          flush=True)


def ratio(a, b):
    return None if a is None or b is None else a / b


def read_csv_rows(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stored_row(results_csv, row_id):
    """A row of one of the suite's results/*.csv."""
    for r in read_csv_rows(ORIGINALS / "results" / results_csv):
        if r["id"] == row_id:
            return r
    raise KeyError(f"{row_id} not in {results_csv}")
