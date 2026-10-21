"""Timings of the compiled core, for either backend.

Every workload calls the core the way zeit's wrappers do, on deterministic
synthetic data, with one thread and with all of them. The result is a JSON file
that ``--compare`` puts side by side with another (C++ before and after a change,
or C++ against Rust). Several backends are measured in turns in one process.

    python benchmarks/bench_core.py --backend cpp --out benchmarks/results/cpp.json
    python benchmarks/bench_core.py --backend cpp rust --only mann_kendall
    python benchmarks/bench_core.py --compare benchmarks/results/old.json benchmarks/results/new.json:rust
"""
import argparse
import datetime
import importlib
import json
import os
import platform
import sys
import time

import numpy as np

WORKLOADS = {}


def workload(name, module):
    def register(build):
        WORKLOADS[name] = (module, build)
        return build
    return register


def seasonal(rng, t, n, period=365.25, amp=0.2, base=0.5, noise=0.03):
    """(n, len(t)) series: a harmonic, a trend, noise and an abrupt drop in half of them."""
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    y = base + amp * np.sin(2 * np.pi * t[None, :] / period + phase) + rng.normal(0, noise, (n, len(t)))
    y += rng.uniform(-1e-5, 1e-5, (n, 1)) * (t[None, :] - t[0])
    at = rng.integers(len(t) // 3, 2 * len(t) // 3, n)
    drop = (np.arange(len(t))[None, :] >= at[:, None]) & (rng.random((n, 1)) < 0.5)
    return y - 0.25 * drop


@workload("landtrendr", "landtrendr")
def _(core, rng, n_jobs):
    years = np.arange(1985, 2025, dtype=np.int32)
    v = seasonal(rng, years.astype(float), 150 * 150, period=7.3, amp=0.05, base=0.7)
    v = np.ascontiguousarray(v.reshape(150, 150, len(years)))
    p = core.landtrendr.LandTrendrParams()
    return lambda: core.landtrendr.fit_trajectory_batch(v, years, p, -9999.0, n_jobs)


@workload("ccdc", "ccdc")
def _(core, rng, n_jobs):
    dates = np.arange(730120, 730120 + 20 * 365, 16, dtype=np.int32)
    n = 16 * 16
    bands = np.stack([seasonal(rng, dates.astype(float), n, amp=0.08, base=0.25, noise=0.01) * 10000
                      for _ in range(6)], axis=1)
    qa = np.where(rng.random((n, len(dates))) < 0.2, 4, 0).astype(np.int32)
    v = np.ascontiguousarray(bands.reshape(16, 16, 6, len(dates)))
    qa = np.ascontiguousarray(qa.reshape(16, 16, len(dates)))
    p = core.ccdc.CCDCParams()
    return lambda: core.ccdc.fit_ccdc_batch(v, qa, dates, p, 6, True, n_jobs)


@workload("twdtw", "twdtw")
def _(core, rng, n_jobs):
    dates = np.arange(0, 365 * 2, 16, dtype=np.int32)
    v = np.ascontiguousarray(seasonal(rng, dates.astype(float), 100 * 100).reshape(100, 100, len(dates)))
    pat_dates = np.arange(0, 365, 16, dtype=np.int32)
    pat = 0.5 + 0.2 * np.sin(2 * np.pi * pat_dates / 365.25)
    p = core.twdtw.TWDTWParams()
    return lambda: core.twdtw.fit_twdtw_batch(v, dates, pat, pat_dates, p, float("inf"), n_jobs)


@workload("som_batch", "som")
def _(core, rng, n_jobs):
    data = np.ascontiguousarray(rng.normal(size=(20000, 8)))
    w0 = rng.normal(size=(10, 10, 8))
    xx, yy = np.meshgrid(np.arange(10.0), np.arange(10.0))

    def run():
        w = w0.copy()
        # learning_rate, sigma, lr_decay, sigma_decay, neighborhood: MiniSom's defaults
        core.som.train_batch(w, data, 20, 0.5, 1.0, 0, 0, 0, xx, yy, n_jobs=n_jobs)
    return run


@workload("phenology", "phenology")
def _(core, rng, n_jobs):
    dates = np.arange(0, 365 * 3, 8, dtype=float)
    v = np.ascontiguousarray(seasonal(rng, dates, 300, amp=0.25, base=0.45, noise=0.02))
    return lambda: core.phenology.fit_phenology_batch(v, dates, 0, 0, n_jobs=n_jobs)


@workload("mann_kendall", "mannkendall")
def _(core, rng, n_jobs):
    v = np.ascontiguousarray(seasonal(rng, np.arange(40.0), 50000, period=9.0))
    return lambda: core.mannkendall.fit_mann_kendall_batch(v, 1, 0.05, -1, 1, 4, n_jobs)


@workload("bfast_monitor", "bfastmonitor")
def _(core, rng, n_jobs):
    v = np.ascontiguousarray(seasonal(rng, np.arange(23 * 8.0), 20000, period=23.0))
    return lambda: core.bfastmonitor.fit_bfast_monitor_batch(v, 2000.0, 2005.0, 23, 3, 0.25, 10, 0.05, 10, n_jobs)


@workload("bfast_lite", "bfastlite")
def _(core, rng, n_jobs):
    v = np.ascontiguousarray(seasonal(rng, np.arange(23 * 6.0), 200, period=23.0))
    return lambda: core.bfastlite.fit_bfast_lite_batch(v, 2000.0, 23, 3, 0.15, 5, 20, n_jobs)


@workload("bfast", "bfast")
def _(core, rng, n_jobs):
    v = np.ascontiguousarray(seasonal(rng, np.arange(23 * 6.0), 200, period=23.0))
    return lambda: core.bfast.fit_bfast_batch(v, 2000.0, 23, 3, 0.15, 5, 5, 10, 0.05, 20, n_jobs)


@workload("snic", "snic")
def _(core, rng, n_jobs):
    y, x = np.mgrid[0:1024, 0:1024] / 1024.0
    data = np.stack([np.sin(6 * x + k) * np.cos(4 * y - k) for k in range(4)]) + rng.normal(0, 0.05, (4, 1024, 1024))
    data = np.ascontiguousarray(data)
    g = np.arange(8, 1024, 16)
    seeds = np.array([(r, c) for r in g for c in g], dtype=np.int32)
    return lambda: core.snic.snic_segment(data, seeds, 10.0, 256, 256, n_jobs)


@workload("sma", "sma")
def _(core, rng, n_jobs):
    em = np.abs(rng.normal(0.3, 0.1, (4, 6)))
    fr = rng.dirichlet(np.ones(4), 200000)
    v = np.ascontiguousarray(fr @ em + rng.normal(0, 0.005, (200000, 6)))
    return lambda: core.sma.unmix_batch(v, em, True, True, 1000.0, n_jobs)


@workload("coded", "coded")
def _(core, rng, n_jobs):
    t = np.sort(2000 + rng.random(260) * 12)
    ndfi = np.ascontiguousarray(0.85 + 0.03 * np.sin(2 * np.pi * t)[None, :] + rng.normal(0, 0.02, (20000, len(t))))
    ndfi[:, 150:] -= 0.4 * (rng.random((20000, 1)) < 0.3)
    feats = np.zeros((20000, 0, len(t)))
    p = core.coded.Params()
    p.train_start, p.train_years = 2000.0, 3.0
    return lambda: core.coded.coded_batch(t, ndfi, feats, p, n_jobs)


@workload("tmask", "tmask")
def _(core, rng, n_jobs):
    days = np.arange(730120, 730120 + 6 * 365, 8, dtype=float)
    g = seasonal(rng, days, 10000, amp=0.03, base=0.08, noise=0.005) * 10000
    s = seasonal(rng, days, 10000, amp=0.05, base=0.2, noise=0.005) * 10000
    cloud = rng.random(g.shape) < 0.1
    g[cloud] += 3000
    s[cloud] += 2000
    g, s = np.ascontiguousarray(g), np.ascontiguousarray(s)
    return lambda: core.tmask.tmask_batch(g, s, days, 10000.0, 5, 0.04, 0.04, n_jobs)


@workload("whittaker", "smooth")
def _(core, rng, n_jobs):
    x = np.arange(0, 365 * 3, 5, dtype=float)
    v = np.ascontiguousarray(seasonal(rng, x, 100000))
    return lambda: core.smooth.whittaker_batch(v, x, np.empty((0, 0)), 10.0, n_jobs)


@workload("medoid", "utils")
def _(core, rng, n_jobs):
    v = np.ascontiguousarray(rng.random((256, 256, 24, 6)))
    # compute_medoid has no n_jobs: it always uses OpenMP's default team
    return lambda: core.utils.compute_medoid(v, -9999.0)


def load_backend(name):
    if name == "cpp":
        return importlib.import_module("zeit._core")
    if name == "rust":
        return importlib.import_module("zeit._rust")
    raise SystemExit(f"unknown backend {name!r}")


def has(core, module):
    try:
        getattr(core, module)
        return True
    except AttributeError:
        return False


def best_time(fn, min_total=1.0, min_reps=3, max_reps=20):
    fn()  # warm-up: page faults, thread pool start, lazy tables
    times = []
    start = time.perf_counter()
    while len(times) < min_reps or (time.perf_counter() - start < min_total and len(times) < max_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), len(times)


def set_affinity(cpus):
    """Restrict the process (its threads, present and future) to `cpus`, or to all with None.

    One-thread timings on hybrid CPUs (performance and efficiency cores) otherwise
    depend on which kind of core the scheduler picks. Returns False where unsupported.
    """
    every = list(range(os.cpu_count()))
    cpus = every if cpus is None else cpus
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
        return True
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        mask = sum(1 << c for c in cpus)
        return bool(k32.SetProcessAffinityMask(ctypes.c_void_p(k32.GetCurrentProcess()), ctypes.c_size_t(mask)))
    return False


def run(args):
    cores = {b: load_backend(b) for b in args.backend}
    threads = {"1": 1, "all": -1}
    pinned = args.cpu >= 0
    results = {b: {} for b in cores}
    for name, (module, build) in WORKLOADS.items():
        if args.only and name not in args.only:
            continue
        present = [b for b, core in cores.items() if has(core, module)]
        for b in cores:
            if b not in present:
                print(f"{name:14s} (not in the {b} backend)")
        for label, n_jobs in threads.items():
            if not present or (name == "medoid" and label == "1"):
                continue
            fns = {b: build(cores[b], np.random.default_rng(42), n_jobs) for b in present}
            if pinned:
                pinned = set_affinity([args.cpu] if n_jobs == 1 else None)
            best = {}
            # the backends take turns, so a drift of the machine (heat, other
            # programs) falls on all of them alike
            for _ in range(args.rounds if len(present) > 1 else 1):
                for b in present:
                    t, _ = best_time(fns[b], args.min_time / max(1, args.rounds * len(present) // 2))
                    best[b] = min(t, best.get(b, t))
            for b in present:
                results[b].setdefault(name, {})[label] = best[b]
            line = "  ".join(f"{b} {best[b] * 1e3:9.2f} ms" for b in present)
            if len(present) > 1:
                line += "  " + "  ".join(f"{b}/{present[0]} {best[b] / best[present[0]]:5.2f}" for b in present[1:])
            print(f"{name:14s} threads={label:3s} {line}", flush=True)
    if pinned:
        set_affinity(None)
    out = {
        "backends": list(cores),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpus": os.cpu_count(), "python": sys.version.split()[0]},
        "single_thread_cpu": args.cpu if pinned else None,
        "results": results,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"-> {args.out}")


def load_results(spec):
    """The timings of FILE.json[:backend] (the file's first backend by default)."""
    path, _, backend = spec.partition(".json:")
    path = path if path.endswith(".json") else path + ".json"
    data = json.load(open(path))
    backend = backend or data["backends"][0]
    return f"{os.path.basename(path)[:-5]}:{backend}", data["results"][backend]


def compare(a_spec, b_spec):
    (la, a), (lb, b) = load_results(a_spec), load_results(b_spec)
    print(f"A = {la}\nB = {lb}\n")
    print(f"{'workload':14s} {'thr':>4s} {'A':>12s} {'B':>12s} {'B/A':>7s}")
    for name, ra in a.items():
        rb = b.get(name, {})
        for label, ta in ra.items():
            tb = rb.get(label)
            ratio = f"{tb / ta:7.2f}" if tb else "      -"
            tbs = f"{tb * 1e3:10.2f}ms" if tb else "           -"
            print(f"{name:14s} {label:>4s} {ta * 1e3:10.2f}ms {tbs} {ratio}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", nargs="+", default=["cpp"], choices=["cpp", "rust"],
                    help="one backend, or several measured in turns in the same process")
    ap.add_argument("--only", nargs="*", help="workloads to run (default: all)")
    ap.add_argument("--min-time", type=float, default=1.0, help="seconds of repetitions per timing")
    ap.add_argument("--rounds", type=int, default=3, help="turns of the backends, with several")
    ap.add_argument("--cpu", type=int, default=0,
                    help="logical CPU the one-thread timings run on (a performance core); -1 to leave it to the OS")
    ap.add_argument("--out", help="JSON file for the results")
    ap.add_argument("--compare", nargs=2, metavar=("A.json[:backend]", "B.json[:backend]"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
