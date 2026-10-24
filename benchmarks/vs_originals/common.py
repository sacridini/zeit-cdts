"""What the benchmarks against the original implementations share: running the original (an R script timing its
own call with ``system.time``, so that loading R and its packages is not counted), timing Zeit's call (the best of a
few runs, after a warm-up), and keeping the results in benchmarks/results/vs-originals.json (one entry per method and
size, the machine's description with them).

Each method's script (``python benchmarks/vs_originals/<method>.py``) writes its inputs to a temporary folder, runs
the original and Zeit on them, checks that they agree and prints the timings.
"""
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


def run_r(script, *args):
    """Runs an R script of this folder; it prints ``SECONDS <s>`` for the timed call (and anything else)."""
    out = subprocess.run([RSCRIPT, str(HERE / script), *map(str, args)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{out.stderr[-3000:]}")
    for line in out.stdout.splitlines():
        if line.startswith("SECONDS "):
            return float(line.split()[1])
    raise RuntimeError(f"{script} printed no SECONDS line:\n{out.stdout[-2000:]}")


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
