"""Timings of the compiled core (zeit._core, in Rust).

Every workload calls the core the way zeit's wrappers do, on deterministic
synthetic data, with one thread and with ``--threads`` of them (all with -1;
ZEIT_NUM_THREADS is set to the same, so that the functions without an n_jobs
keep to it too). The result is a JSON file
that ``--compare`` puts side by side with another: before and after a change,
or against the C++ core of zeit 1.x, whose timings are kept in
benchmarks/results/ (the "cpp" backend of those files).

    python benchmarks/bench_core.py --out benchmarks/results/new.json
    python benchmarks/bench_core.py --threads 8 --out benchmarks/results/new.json
    python benchmarks/bench_core.py --only mann_kendall
    python benchmarks/bench_core.py --compare benchmarks/results/win-i5-13600k-rust.json:cpp benchmarks/results/new.json
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


@workload("som_predict", "som")
def _(core, rng, n_jobs):
    # the winners alone (the first phase of every batch iteration)
    data = np.ascontiguousarray(rng.normal(size=(200000, 8)))
    w = rng.normal(size=(10, 10, 8))
    return lambda: core.som.predict_bmus(data, w, n_jobs=n_jobs)


@workload("som_online", "som")
def _(core, rng, n_jobs):
    # sequential by nature: n_jobs is not used
    data = np.ascontiguousarray(rng.normal(size=(20000, 8)))
    w0 = rng.normal(size=(10, 10, 8))
    xx, yy = np.meshgrid(np.arange(10.0), np.arange(10.0))
    order = np.arange(20000, dtype=np.int64)

    def run():
        w = w0.copy()
        core.som.train_online(w, data, order, 20000, False, 0.5, 1.0, 0, 0, 0, xx, yy)
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
    # compute_medoid has no n_jobs: it uses every processor ZEIT_NUM_THREADS allows
    return lambda: core.utils.compute_medoid(v, -9999.0)


@workload("ar_trend", "parts")
def _(core, rng, n_jobs):
    # remotePARTS' fitAR, per pixel: REML of the AR(1) parameter, then its GLS
    t = np.arange(1, 31, dtype=float)
    v = seasonal(rng, t, 100000, period=7.0, amp=0.05, base=0.5, noise=0.05)
    v = np.ascontiguousarray(v + rng.normal(0, 0.002, (100000, 1)) * t[None, :])
    design = np.column_stack([np.ones_like(t), t])
    return lambda: core.parts.fit_ar_batch(v, design, n_jobs=n_jobs)


@workload("parts", "parts")
def _(core, rng, n_jobs):
    # the partitioned GLS of 20 partitions of 500 pixels, exponential correlation of a known range
    n = 10000
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    response = 0.01 * x + rng.normal(0, 0.1, n)
    design = np.column_stack([np.ones(n), x / 100])
    pm = core.parts.sample_partitions(n, 500, 20, 7)
    return lambda: core.parts.fit_parts(x, y, response, design, np.ones((n, 1)), pm, range=10.0, n_jobs=n_jobs)


@workload("recovery", "recovery")
def _(core, rng, n_jobs):
    n, years = 1000 * 1000, np.arange(2000, 2021, dtype=np.int32)
    after = years[None, :] - 2010
    pre, low, rate = rng.uniform(0.4, 0.7, (n, 1)), rng.uniform(0, 0.2, (n, 1)), rng.uniform(0.02, 0.15, (n, 1))
    v = np.ascontiguousarray(np.where(after < 0, pre + rng.normal(0, 0.01, (n, len(years))),
                                      np.minimum(low + rate * after, pre)))
    starts, troughs = np.full(n, 2009, np.int32), np.full(n, 2010, np.int32)
    return lambda: core.recovery.recovery_batch(v, years, starts, troughs, n_jobs=n_jobs)


@workload("ewmacd", "ewmacd")
def _(core, rng, n_jobs):
    days = np.sort(rng.choice(np.arange(0, 10 * 365), 120, replace=False))
    dates = np.datetime64("2000-01-01") + days.astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]").astype(int).astype(np.int32) + 1970
    doys = (dates - dates.astype("datetime64[Y]")).astype(float) + 1
    t = years + (doys - 1) / 365.25
    v = np.ascontiguousarray(seasonal(rng, days.astype(float), 20000, amp=0.15, base=0.55, noise=0.02))
    return lambda: core.ewmacd.ewmacd_batch(v, years, doys, t, 2000, 2004, n_jobs=n_jobs)


@workload("hotspots", "hotspots")
def _(core, rng, n_jobs):
    # emerging hot spots: Gi* of every cell and date (square window of radius 1, lag 1), Mann-Kendall of each series
    y, x = np.mgrid[0:300, 0:300] / 300.0
    trend = np.arange(12.0)[:, None, None] * 0.05 * np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.02)[None]
    v = np.ascontiguousarray(trend + rng.normal(0, 0.5, (12, 300, 300)))
    return lambda: core.hotspots.emerging(v, n_jobs=n_jobs)


@workload("beast", "beast")
def _(core, rng, n_jobs):
    # Rbeast's defaults (3 chains of 8000 samples) on 10 series of 6 years of 23 observations
    v = np.ascontiguousarray(seasonal(rng, np.arange(23 * 6.0), 10, period=23.0))
    return lambda: core.beast.beast_batch(v, start=1.0, dt=1.0, period=23.0, n_jobs=n_jobs)


@workload("sccd", "sccd")
def _(core, rng, n_jobs):
    dates = np.arange(730120, 730120 + 20 * 365, 16, dtype=np.int32)
    n = 32 * 32
    bands = np.stack([seasonal(rng, dates.astype(float), n, amp=0.08, base=0.25, noise=0.01) * 10000
                      for _ in range(6)], axis=1)
    qa = np.where(rng.random((n, len(dates))) < 0.2, 4, 0).astype(np.int32)
    v = np.ascontiguousarray(bands.reshape(32, 32, 6, len(dates)))
    qa = np.ascontiguousarray(qa.reshape(32, 32, len(dates)))
    return lambda: core.sccd.sccd_batch(v, qa, dates, n_jobs=n_jobs)


@workload("aoa_train_di", "aoa")
def _(core, rng, n_jobs):
    train = np.ascontiguousarray(rng.normal(size=(5000, 8)))
    folds = rng.integers(0, 10, 5000).astype(np.int64)
    return lambda: core.aoa.train_di(train, np.ones(8), folds, n_jobs)


@workload("aoa_di", "aoa")
def _(core, rng, n_jobs):
    # the DI of 100,000 pixels against 2000 training points (8 predictors: the k-d tree), and their LPD
    td = core.aoa.train_di(np.ascontiguousarray(rng.normal(size=(2000, 8))), np.ones(8), None, 1)
    data = np.ascontiguousarray(rng.normal(size=(100000, 8)) * 1.2)
    return lambda: core.aoa.di(data, td["center"], td["scale"], td["weight"], td["train"], td["mean_distance"],
                               td["threshold"], True, n_jobs)


def class_map(rng, side, classes=5):
    """A (side, side) map of classes in patches, a tenth of its pixels noise."""
    blocks = rng.integers(0, classes, (side // 16 + 1, side // 16 + 1))
    m = np.kron(blocks, np.ones((16, 16)))[:side, :side]
    noise = rng.random((side, side)) < 0.1
    m[noise] = rng.integers(0, classes, noise.sum())
    return np.ascontiguousarray(m.astype(np.float64))


@workload("smooth_probs", "postproc")
def _(core, rng, n_jobs):
    # sits' Bayesian smoothing: 4 classes, a 9 x 9 window
    p = rng.dirichlet(np.ones(4), (500, 500)).transpose(2, 0, 1)
    p = np.ascontiguousarray(p)
    return lambda: core.postproc.smooth_probs(p, 9, np.full(4, 20.0), 0.5, n_jobs)


@workload("majority", "postproc")
def _(core, rng, n_jobs):
    m = class_map(rng, 2000)
    return lambda: core.postproc.majority(m, 3, 1, n_jobs)


@workload("sieve", "postproc")
def _(core, rng, n_jobs):
    # sequential by nature: n_jobs is not used
    m = class_map(rng, 2000)
    return lambda: core.postproc.sieve(m, 11, True)


@workload("desawtooth", "postproc")
def _(core, rng, n_jobs):
    # temporal consistency of 1,000,000 class series of 20 dates
    runs = np.repeat(rng.integers(0, 4, (1000000, 5)), 4, axis=1).astype(np.float64)
    flip = rng.random(runs.shape) < 0.05
    runs[flip] = rng.integers(0, 4, flip.sum())
    runs = np.ascontiguousarray(runs)
    return lambda: core.postproc.desawtooth(runs, 1, n_jobs)


def _warp_args(src, method, scales, gt, kind, nodata, region, lattice=None):
    """The arguments of warp.warp as zeit/_warp.py passes them (per_band, without GDAL's transformer)."""
    if lattice is None:
        lattice = (np.empty(0, np.int8), np.empty((0, 2)), np.empty(0, np.int64), np.empty((0, 2)), (1, 0, 0))
    identity = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    offset = 0.0 if method >= 5 else 0.5
    return ((src, method, src.shape[1:], (0, 0), scales, nodata is not None, 0.0 if nodata is None else nodata, 0.0,
             region, kind, offset, gt, identity) + lattice + (np.empty((0, 0, 2)), 0, 0, 0, np.empty(0, np.uintp)))


def _warp_lattice(rng, cells_y, cells_x):
    """The lattice of a reprojection, its corner nodes on a smooth map, a tenth of its cells at level 2."""
    def at(u, v):
        return 5.0 + 1.05 * u + 0.08 * v + 2e-5 * u * v, 7.0 - 0.06 * u + 1.02 * v - 1e-5 * u * u

    c = np.arange(cells_x + 1) * 16 + 0.5
    r = np.arange(cells_y + 1) * 16 + 0.5
    grid0 = np.ascontiguousarray(np.stack(at(c[None, :], r[:, None]), axis=-1))
    level = np.where(rng.random(cells_y * cells_x) < 0.1, 2, 0).astype(np.int8)
    node_off = np.zeros(level.size, np.int64)
    blocks = []
    for k in np.flatnonzero(level):
        cy, cx = divmod(k, cells_x)
        steps = np.arange(0, 17, 4) + 0.5
        blocks.append(np.stack(at(cx * 16 + steps[None, :], cy * 16 + steps[:, None]), axis=-1).reshape(-1, 2))
        node_off[k] = 25 * (len(blocks) - 1)
    return level, grid0, node_off, np.ascontiguousarray(np.concatenate(blocks)), (cells_x, 0, 0)


@workload("warp_bilinear", "warp")
def _(core, rng, n_jobs):
    src = rng.normal(1000, 300, (4, 1200, 1200)).astype(np.float32)
    src[rng.random(src.shape) < 0.1] = np.nan
    args = _warp_args(src, 1, (0.95, 0.98), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0), 1, float("nan"), (0, 0, 1024, 1024),
                      _warp_lattice(rng, 64, 64))
    return lambda: core.warp.warp(*args, per_band=True, n_threads=n_jobs)


@workload("warp_lanczos", "warp")
def _(core, rng, n_jobs):
    src = np.ascontiguousarray(rng.integers(0, 256, (3, 1100, 1100), dtype=np.uint8))
    args = _warp_args(src, 4, (1.0, 1.0), (40.0, 0.96, 0.12, 20.0, -0.1, 0.97), 0, None, (0, 0, 1000, 1000))
    return lambda: core.warp.warp(*args, per_band=True, n_threads=n_jobs)


@workload("warp_average", "warp")
def _(core, rng, n_jobs):
    src = np.ascontiguousarray(rng.integers(-2000, 8000, (3, 2000, 2000), dtype=np.int16))
    src[:, rng.random(src.shape[1:]) < 0.05] = -9999
    args = _warp_args(src, 5, (1 / 2.7, 1 / 2.7), (3.0, 2.7, 0.02, 1.0, -0.03, 2.7), 0, -9999.0, (0, 0, 700, 700))
    return lambda: core.warp.warp(*args, per_band=True, n_threads=n_jobs)


def load_backend(name):
    if name == "rust":
        return importlib.import_module("zeit._core")
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


def no_power_throttling():
    """Takes this process out of Windows' power throttling ("EcoQoS"), where it is supported.

    Windows 11 moves a process it deems in the background, as a benchmark run from a
    terminal is, to the efficiency cores at a low clock after a few seconds of work:
    one-thread timings came out up to 2.4 times slower for it.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 1 | 4, 0)  # execution speed and timer resolution: not throttled
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s)))


def run(args):
    no_power_throttling()
    cores = {b: load_backend(b) for b in args.backend}
    threads = {"1": 1, ("all" if args.threads <= 0 else str(args.threads)): args.threads}
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
            sequential = name in ("som_online", "sieve")
            if not present or (name == "medoid" and n_jobs == 1) or (sequential and n_jobs != 1):
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
        "threads": args.threads,
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
    print(f"{'workload':14s} {'thr':>6s} {'A':>12s} {'B':>12s} {'B/A':>7s}")
    for name, ra in a.items():
        rb = b.get(name, {})
        for label, ta in ra.items():
            tb = rb.get(label)
            if tb is None and label != "1":
                # the many-thread timings of two runs with different thread counts ("all", "8")
                other = [k for k in rb if k != "1"]
                tb = rb[other[0]] if other else None
                label = f"{label}/{other[0]}" if other else label
            ratio = f"{tb / ta:7.2f}" if tb else "      -"
            tbs = f"{tb * 1e3:10.2f}ms" if tb else "           -"
            print(f"{name:14s} {label:>6s} {ta * 1e3:10.2f}ms {tbs} {ratio}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", nargs="+", default=["rust"], choices=["rust"],
                    help="the core measured (the name its timings take in the JSON file)")
    ap.add_argument("--only", nargs="*", help="workloads to run (default: all)")
    ap.add_argument("--threads", type=int, default=-1,
                    help="threads of the many-thread timings (-1: all the processors zeit would use)")
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
        if args.threads > 0:
            os.environ["ZEIT_NUM_THREADS"] = str(args.threads)
        run(args)


if __name__ == "__main__":
    main()
