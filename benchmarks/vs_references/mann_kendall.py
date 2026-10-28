"""Mann-Kendall with the Rust core, on the workloads of cdts_vs_originals' compare_mann_kendall.py (single core)
and parallel_mann_kendall.py (multi-core).

Single core: 6 kinds of synthetic series x 15 seeds = 90 series per test (original, Hamed-Rao, Yue-Wang), the
per-call time of _core.mannkendall.mk_test_single, each call timed alone and interleaved with pymannkendall's
call on the same series, as the original script did (pymannkendall is run for the agreement check only: its
stored timing stays the reference). Also a warm best of 3 (each run 10 passes of Zeit's 90 calls in a row).
Multi-core: fit_mann_kendall_batch (Hamed-Rao) on 5,000 series of 30 observations, n_jobs=1 then n_jobs=8, one
run each as the original (cold), and the warm best of 3.

    python benchmarks/vs_references/mann_kendall.py
"""
import common  # first: thread variables

import time

import numpy as np
import pymannkendall as pmk

from zeit._core.mannkendall import MKMethod, fit_mann_kendall_batch, mk_test_single
from common import THREADS, once, pin, record, stored_row, timed, warm_best

TREND_CODE = {"increasing": 1, "decreasing": -1, "no trend": 0}
KINDS = ["trend", "decreasing", "no_trend", "ties", "autocorrelated", "with_nans"]
N_SEEDS = 15
VARIANTS = [("original", int(MKMethod.ORIGINAL), pmk.original_test, "mann_kendall_original"),
            ("hamed_rao", int(MKMethod.HAMED_RAO), pmk.hamed_rao_modification_test, "mann_kendall_hamed_rao"),
            ("yue_wang", int(MKMethod.YUE_WANG), pmk.yue_wang_modification_test, "mann_kendall_yue_wang")]
PUBLISHED_MEAN = 0.021  # performance.md, the mean of the three tests


def make_series(kind, rng):
    if kind == "trend":
        return np.arange(30) * 0.4 + rng.normal(0, 0.3, 30)
    if kind == "decreasing":
        return -np.arange(25) * 0.3 + rng.normal(0, 0.2, 25)
    if kind == "no_trend":
        return rng.normal(0, 1.0, 40)
    if kind == "ties":
        return np.repeat(np.arange(10), 3).astype(float) + rng.normal(0, 0.01, 30)
    if kind == "autocorrelated":
        return np.cumsum(rng.normal(0, 1, 50)) * 0.1
    if kind == "with_nans":
        y = np.arange(30) * 0.4 + rng.normal(0, 0.3, 30)
        y[5] = np.nan
        y[17] = np.nan
        return y


def series_of(seed_base):
    return [make_series(kind, np.random.RandomState(seed_base + s)) for kind in KINDS for s in range(N_SEEDS)]


def interleaved(series, code, ref_fn):
    """The original protocol: per series, pymannkendall's call then Zeit's, each timed."""
    z_times, r_times, out, ref = [], [], [], []
    for y in series:
        t0 = time.perf_counter()
        r = ref_fn(y)
        t1 = time.perf_counter()
        r_times.append((t1 - t0) * 1000.0)
        t0 = time.perf_counter()
        z = mk_test_single(list(y), method=code)
        t1 = time.perf_counter()
        z_times.append((t1 - t0) * 1000.0)
        out.append(z)
        ref.append(r)
    return float(np.mean(z_times)), float(np.mean(r_times)), out, ref


def agreement(out, ref):
    trend = np.mean([int(z[0]) == TREND_CODE[r.trend] for z, r in zip(out, ref)])
    dp = [abs(r.p - z[2]) for z, r in zip(out, ref) if not (np.isnan(r.p) and np.isnan(z[2]))]
    ds = [abs(r.slope - z[7]) for z, r in zip(out, ref) if not (np.isnan(r.slope) and np.isnan(z[7]))]
    return {"trend_direction_exact_match_rate": float(trend), "mean_abs_diff_p_value": float(np.mean(dp)),
            "mean_abs_diff_slope": float(np.mean(ds))}


def single_core():
    pin(1)
    means, means_warm, refs_stored, refs_fresh = [], [], [], []
    for i, (name, code, ref_fn, row_id) in enumerate(VARIANTS):
        series = series_of(1000 * (i + 1))
        n = len(series)
        z_ms, r_ms, out, ref = timed(lambda: interleaved(series, code, ref_fn), name)

        def calls():  # 10 passes of the 90 calls: one pass (~1 ms) is within the timer's jitter
            for _ in range(10):
                for y in series:
                    mk_test_single(list(y), method=code)
        warm_ms = timed(lambda: warm_best(calls), name) / (10 * n) * 1000.0
        stored = stored_row("mann_kendall.csv", row_id)
        agree = agreement(out, ref)
        changed = {k: (float(stored["primary_metric_value"]) if k.startswith("trend") else
                       float(stored["secondary_metric_value"]))
                   for k in ("trend_direction_exact_match_rate", "mean_abs_diff_p_value")}
        ref_stored = float(stored["reference_time_ms_mean"])
        pub = float(stored["cdts_time_ms_mean"])
        record({"table": "single-core", "row": f"Mann-Kendall ({name})", "order": 6,
                "workload": f"{n} series of 25-50 obs (6 kinds x 15 seeds), mk_test_single per call", "unit": "ms/call",
                "published_cpp_1t": pub, "rust_1t": z_ms, "rust_1t_warm_best3": warm_ms,
                "ratio_rust_cpp_1t": z_ms / pub, "reference_1t": ref_stored, "reference_tool": f"pymannkendall {ref_fn.__name__}",
                "speedup_vs_reference_1t": ref_stored / z_ms, "published_speedup": float(stored["speedup_x"]),
                "agreement": agree, "agreement_stored": changed,
                "notes": f"published value: results/mann_kendall.csv ({row_id}); pymannkendall re-run here only for "
                         f"the agreement check, its fresh time {r_ms:.4f} ms/call (stored {ref_stored} kept)."})
        means.append(z_ms)
        means_warm.append(warm_ms)
        refs_stored.append(ref_stored)
        refs_fresh.append(r_ms)
    m, mr = float(np.mean(means)), float(np.mean(refs_stored))
    record({"table": "single-core", "row": "Mann-Kendall", "order": 6,
            "workload": "30-obs series (3 tests): mean per call over the 3 x 90 series", "unit": "ms/call",
            "published_cpp_1t": PUBLISHED_MEAN, "rust_1t": m, "rust_1t_warm_best3": float(np.mean(means_warm)),
            "ratio_rust_cpp_1t": m / PUBLISHED_MEAN, "reference_1t": round(mr, 2), "reference_tool": "Python pymannkendall",
            "speedup_vs_reference_1t": mr / m, "published_speedup": 36.4,
            "notes": "the table's row: the mean of the three tests (published 0.021 ms vs 0.77 ms; per-test rows "
                     f"above). pymannkendall re-run for agreement: {np.mean(refs_fresh):.3f} ms/call fresh."})


def multi_core():
    rng = np.random.RandomState(0)
    values = np.zeros((5000, 30))
    for i in range(5000):
        values[i] = np.arange(30) * 0.1 * (1 if i % 2 == 0 else -1) + rng.normal(0, 0.5, 30)
    method = int(MKMethod.HAMED_RAO)
    res = {}

    def run(n_jobs):
        res[n_jobs] = fit_mann_kendall_batch(values_array=values, method=method, n_jobs=n_jobs)

    pin(1)
    t1 = timed(lambda: once(lambda: run(1)), "1T") * 1000
    pin(THREADS)
    t8 = timed(lambda: once(lambda: run(THREADS)), "8T") * 1000
    pin(1)
    w1 = timed(lambda: warm_best(lambda: run(1)), "1T warm") * 1000
    pin(THREADS)
    w10 = timed(lambda: warm_best(lambda: run(THREADS)), "8T warm") * 1000
    pin(None)
    same = bool(np.array_equal(res[1], res[THREADS], equal_nan=True))
    st = {r["tool"]: r for r in common.read_csv_rows(common.ORIGINALS / "results" / "parallel_mann_kendall.csv")}
    c, p = st["cdts"], st["Python_pymannkendall"]
    ref1, refm = float(p["time_ms_single_thread"]), float(p["time_ms_parallel"])
    record({"table": "multi-core", "row": "Mann-Kendall", "order": 5,
            "workload": "5,000 series x 30 obs, fit_mann_kendall_batch (Hamed-Rao)", "unit": "ms (batch)",
            "published_cpp_1t": float(c["time_ms_single_thread"]), "published_cpp_multi": float(c["time_ms_parallel"]),
            "published_threads": int(c["n_workers_parallel"]),
            "rust_1t": t1, "rust_8t": t8, "rust_1t_warm_best3": w1, "rust_8t_warm_best3": w10,
            "ratio_rust_cpp_1t": t1 / float(c["time_ms_single_thread"]),
            "ratio_rust8_cpp19": t8 / float(c["time_ms_parallel"]),
            "reference_1t": ref1, "reference_multi": refm, "reference_workers": int(p["n_workers_parallel"]),
            "reference_tool": "pymannkendall, multiprocessing.Pool(19)",
            "speedup_vs_reference_1t": ref1 / t1, "speedup_vs_reference_multi": refm / t8,
            "agreement": {"outputs_1t_equal_8t": same},
            "notes": "rust_1t / rust_8t: one cold run each, n_jobs=1 first (the original protocol); warm best of 3 "
                     "also given. Published multi-core: n_jobs=-1 = 19 threads; here 8."})


if __name__ == "__main__":
    common.start()
    single_core()
    multi_core()
