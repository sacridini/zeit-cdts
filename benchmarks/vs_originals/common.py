"""What the benchmarks against the original implementations share: running the original (an R script timing its
own call with ``system.time``, so that loading R and its packages is not counted), timing Zeit's call (the best of a
few runs, after a warm-up), and keeping the results in benchmarks/results/vs-originals.json (one entry per method and
size, the machine's description with them).

Each method's script (``python benchmarks/vs_originals/<method>.py``) writes its inputs to a temporary folder, runs
the original and Zeit on them, checks that they agree and prints the timings.
"""
import ctypes
import datetime
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "vs-originals.json"
RSCRIPT = os.environ.get("RSCRIPT", r"C:\Program Files\R\R-4.4.2\bin\Rscript.exe" if os.name == "nt" else "Rscript")


def workdir(name):
    d = Path(tempfile.gettempdir()) / f"zeit-vs-{name}"
    d.mkdir(exist_ok=True)
    return d


def no_power_throttling(pid=None):
    """Takes a process (this one by default) out of Windows' power throttling; nothing elsewhere.

    Windows 11 slows the processes it deems in the background (power throttling, "EcoQoS": on a hybrid processor,
    the efficiency cores at a low clock) after a few seconds of work, and a benchmark run from a terminal is one:
    zeit's threads ran up to 2.4 times slower for it.
    """
    if os.name != "nt":
        return
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    h = k.GetCurrentProcess() if pid is None else k.OpenProcess(0x0200, False, pid)  # PROCESS_SET_INFORMATION
    s = State(1, 1 | 4, 0)  # execution speed and timer resolution: not throttled
    if not k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)):  # ProcessPowerThrottling
        raise OSError(ctypes.get_last_error(), "SetProcessInformation")
    if pid is not None:
        k.CloseHandle(h)


def run_r(script, *args):
    """Runs an R script of this folder, taken out of power throttling as soon as it starts (before its timed call);
    it prints ``SECONDS <s>`` for the timed call (and anything else)."""
    p = subprocess.Popen([RSCRIPT, str(HERE / script), *map(str, args)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    no_power_throttling(p.pid)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{err[-3000:]}")
    for line in out.splitlines():
        if line.startswith("SECONDS "):
            return float(line.split()[1])
    raise RuntimeError(f"{script} printed no SECONDS line:\n{out[-2000:]}")


def threads_label(n_jobs):
    """Zeit's threads for an n_jobs (-1: all the processors but one, as zeit's core)."""
    return n_jobs if n_jobs > 0 else max((os.cpu_count() or 2) - 1, 1)


def best_of(fn, repeat=3):
    fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def record(method, original, size, t_original, t_zeit, threads, note=""):
    """Keeps a result and prints it."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {"results": []}
    data["machine"] = {"platform": platform.platform(), "processor": platform.processor(),
                       "cpus": os.cpu_count()}
    data["results"] = [r for r in data["results"] if not (r["method"] == method and r["size"] == size
                                                          and r["original"] == original and r["threads"] == threads)]
    data["results"].append({"method": method, "original": original, "size": size, "original_s": t_original,
                            "zeit_s": t_zeit, "speedup": t_original / t_zeit, "threads": threads, "note": note,
                            "date": datetime.date.today().isoformat()})
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=1))
    print(f"{method:<22} {size:<26} {original:<28} {t_original:9.3f} s   zeit {t_zeit:9.4f} s ({threads} threads)"
          f"   x{t_original / t_zeit:,.0f}")
