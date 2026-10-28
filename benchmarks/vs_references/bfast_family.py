"""BFAST Monitor, BFAST Lite and classic BFAST with the Rust core, on the workloads of cdts_vs_originals'
compare_bfast_monitor.py, compare_bfast_lite.py, compare_bfast.py (single core) and parallel_bfast_family.py
(multi-core).

Single core: the 40 series each comparison wrote for R (data/bfast*_series.csv, read back here, so Zeit gets
exactly what R got), _core.bfast*.bfast* per series (the list built inside the timed call, as there), the mean
per-call time of one pass (the original protocol) and of the warm best of 3 runs (10 passes each for BFAST
Monitor); the agreement with R's stored outputs (data/bfast*_r_results.csv) computed as the original scripts did.
Multi-core: fit_bfast_monitor_batch on 20,000 series of 150 obs, fit_bfast_lite_batch on 2,000, fit_bfast_batch on
400 of 300 (the same synthetic series and seeds), n_jobs=1 then n_jobs=8, one cold run each as the original, and
the warm best of 3.

    python benchmarks/vs_references/bfast_family.py [monitor] [lite] [classic]
"""
import common  # first: thread variables

import sys

import numpy as np
import pandas as pd

from zeit._core.bfast import bfast, fit_bfast_batch
from zeit._core.bfastlite import bfast_lite, fit_bfast_lite_batch
from zeit._core.bfastmonitor import bfast_monitor, fit_bfast_monitor_batch
from common import DATA, THREADS, once, pin, record, stored_row, timed, warm_best

FREQ = 23
NAN = float("nan")


def read_series(name):
    df = pd.read_csv(DATA / name, na_values=["NA"], keep_default_na=False)
    cols = [c for c in df.columns if c.startswith("v")]
    rows = []
    for _, r in df.iterrows():
        y = r[cols[:int(r["n"])]].to_numpy(dtype=float)
        rows.append((r.to_dict(), y))  # a dict, as the original's scenario dicts (pandas rows are slow)
    return rows


def as_list(y):
    """What the original scripts passed (built inside the timed call, as there)."""
    return [float(v) if not np.isnan(v) else float("nan") for v in y]


def per_call(fn, series):
    """(mean ms per call of one pass, outputs), each call timed alone as the original scripts did."""
    import time
    out, times = [], []
    for s in series:
        t0 = time.perf_counter()
        out.append(fn(s))
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), out


def single(row_name, order, csv_name, r_csv, stored_id, call, compare, workload, passes=1):
    series = read_series(csv_name)
    pin(1)
    ms, out = timed(lambda: per_call(call, series), row_name)
    # warm: the best of 3 runs of `passes` passes (BFAST Monitor's pass, ~3 ms, is within the timer's jitter)
    warm = timed(lambda: warm_best(lambda: [call(s) for _ in range(passes) for s in series]),
                 row_name) / (passes * len(series)) * 1000.0
    ref = pd.read_csv(DATA / r_csv)
    agree, stored_agree = compare(series, out, ref)
    st = stored_row(f"{stored_id}.csv", stored_id)
    pub, ref_ms = float(st["cdts_time_ms_mean"]), float(st["reference_time_ms_mean"])
    record({"table": "single-core", "row": row_name, "order": order, "workload": workload, "unit": "ms/call",
            "published_cpp_1t": pub, "rust_1t": ms, "rust_1t_warm_best3": warm, "ratio_rust_cpp_1t": ms / pub,
            "reference_1t": ref_ms, "reference_tool": st["reference_tool"], "speedup_vs_reference_1t": ref_ms / ms,
            "published_speedup": float(st["speedup_x"]), "agreement": agree, "agreement_stored": stored_agree,
            "notes": f"published and reference: results/{stored_id}.csv; agreement against data/{r_csv}."})


# ---------------------------------------------------------------- the three comparisons

def monitor_call(s):
    r, y = s
    freq = int(r["frequency"])
    try:
        res = bfast_monitor(as_list(y), start_time=2000.0, monitor_start_time=2000.0 + int(r["monitor_start_idx"]) / freq, frequency=freq,
                            h=float(r["h"]), period=int(r["period"]), alpha=float(r["alpha"]))
    except RuntimeError:
        return None
    return res


def monitor_compare(series, out, ref):
    rows = []
    for (r, _), res in zip(series, out):
        ok = res is not None and res.valid == 1.0
        rows.append(dict(id=r["id"], valid=int(ok), has_break=int(res.has_break) if ok else np.nan,
                         magnitude=res.magnitude if ok else np.nan))
    m = pd.DataFrame(rows).merge(ref, on="id", suffixes=("_z", "_r"))
    both = m[(m.valid_z == 1) & (m.valid_r == 1)]
    bb = both[(both.has_break_z == 1) & (both.has_break_r == 1)]
    st = stored_row("bfast_monitor.csv", "bfast_monitor")
    return ({"has_break_agreement_rate": float((both.has_break_z == both.has_break_r).mean()),
             "magnitude_mean_abs_diff": float(np.mean(np.abs(bb.magnitude_z - bb.magnitude_r))),
             "magnitude_correlation": float(np.corrcoef(bb.magnitude_z, bb.magnitude_r)[0, 1]),
             "n_both_break": int(len(bb))},
            {"has_break_agreement_rate": float(st["primary_metric_value"]),
             "magnitude_mean_abs_diff": float(st["secondary_metric_value"])})


def lite_call(s):
    r, y = s
    return bfast_lite(as_list(y), start_time=2000.0, frequency=int(r["frequency"]), h=float(r["h"]))


def lite_compare(series, out, ref):
    rows = []
    for (r, _), res in zip(series, out):
        ok = res.valid == 1.0
        bp1 = res.breakpoint_idx[0] if ok and len(res.breakpoint_idx) else np.nan
        rows.append(dict(id=r["id"], valid=int(ok), n_breaks=int(res.n_breaks) if ok else np.nan, bp1=bp1))
    m = pd.DataFrame(rows).merge(ref, on="id", suffixes=("_z", "_r"))
    both = m[(m.valid_z == 1) & (m.valid_r == 1)]
    bb = both[(both.n_breaks_z > 0) & (both.n_breaks_r > 0)]
    st = stored_row("bfast_lite.csv", "bfast_lite")
    return ({"n_breaks_exact_match_rate": float((both.n_breaks_z == both.n_breaks_r).mean()),
             "first_breakpoint_idx_exact_match_rate": float((bb.bp1_z == bb.bp1_r).mean()),
             "breakpoint_idx_mean_abs_diff": float(np.mean(np.abs(bb.bp1_z - bb.bp1_r))), "n_both_break": int(len(bb))},
            {"n_breaks_exact_match_rate": float(st["primary_metric_value"]),
             "first_breakpoint_idx_exact_match_rate": float(st["secondary_metric_value"])})


def classic_call(s):
    r, y = s
    return bfast(as_list(y), start_time=2000.0, frequency=int(r["frequency"]), level=float(r["level"]))


def classic_compare(series, out, ref):
    rows = []
    for (r, _), res in zip(series, out):
        ok = res.valid == 1.0
        rows.append(dict(id=r["id"], valid=int(ok),
                         n_trend_breaks=int(res.n_trend_breaks) if ok else np.nan,
                         trend_bp1=(res.trend_breakpoint_idx[0] if ok and len(res.trend_breakpoint_idx) else np.nan),
                         n_season_breaks=int(res.n_season_breaks) if ok else np.nan,
                         magnitude=res.magnitude if ok else np.nan, niter=int(res.n_iter) if ok else np.nan))
    m = pd.DataFrame(rows).merge(ref, on="id", suffixes=("_z", "_r"))
    both = m[(m.valid_z == 1) & (m.valid_r == 1)]
    bt = both[(both.n_trend_breaks_z > 0) & (both.n_trend_breaks_r > 0)]
    st = stored_row("bfast.csv", "bfast")
    return ({"n_trend_breaks_exact_match_rate": float((both.n_trend_breaks_z == both.n_trend_breaks_r).mean()),
             "magnitude_mean_abs_diff": float(np.mean(np.abs(bt.magnitude_z - bt.magnitude_r))),
             "n_season_breaks_exact_match_rate": float((both.n_season_breaks_z == both.n_season_breaks_r).mean()),
             "n_iter_exact_match_rate": float((both.niter_z == both.niter_r).mean()),
             "magnitude_correlation": float(np.corrcoef(bt.magnitude_z, bt.magnitude_r)[0, 1]),
             "trend_breakpoint_idx_exact_match_rate": float((bt.trend_bp1_z == bt.trend_bp1_r).mean()),
             "n_both_trend_break": int(len(bt))},
            {"n_trend_breaks_exact_match_rate": float(st["primary_metric_value"]),
             "magnitude_mean_abs_diff": float(st["secondary_metric_value"]),
             "notes": "n_season_breaks 1.0, n_iter 0.875, magnitude_correlation 0.998122, trend_bp exact 0.75 on 20"})


# ---------------------------------------------------------------- multi-core

def make_series(n, seed, break_at=None, break_mag=1.0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 0.5 + 0.0003 * t + 0.1 * np.cos(2 * np.pi * t / FREQ) + rng.normal(0, 0.02, n)
    if break_at is not None:
        y[break_at:] += break_mag
    return y


def multi(row_name, order, csv_name, workload, run):
    res = {}

    def go(n_jobs):
        res[n_jobs] = run(n_jobs)

    pin(1)
    t1 = timed(lambda: once(lambda: go(1)), "1T") * 1000
    pin(THREADS)
    t8 = timed(lambda: once(lambda: go(THREADS)), "8T") * 1000
    pin(1)
    w1 = timed(lambda: warm_best(lambda: go(1)), "1T warm") * 1000
    pin(THREADS)
    w10 = timed(lambda: warm_best(lambda: go(THREADS)), "8T warm") * 1000
    pin(None)
    a, b = res[1], res[THREADS]
    same = all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b)) if isinstance(a, tuple) else \
        bool(np.array_equal(a, b, equal_nan=True))
    st = {r["tool"]: r for r in common.read_csv_rows(common.ORIGINALS / "results" / csv_name)}
    c = st["cdts"]
    r = next(v for k, v in st.items() if k != "cdts")
    p1, pm = float(c["time_ms_single_thread"]), float(c["time_ms_parallel"])
    ref1, refm = float(r["time_ms_single_thread"]), float(r["time_ms_parallel"])
    record({"table": "multi-core", "row": row_name, "order": order, "workload": workload, "unit": "ms (batch)",
            "published_cpp_1t": p1, "published_cpp_multi": pm, "published_threads": int(c["n_workers_parallel"]),
            "rust_1t": t1, "rust_8t": t8, "rust_1t_warm_best3": w1, "rust_8t_warm_best3": w10,
            "ratio_rust_cpp_1t": t1 / p1, "ratio_rust8_cpp19": t8 / pm,
            "reference_1t": ref1, "reference_multi": refm, "reference_workers": int(r["n_workers_parallel"]),
            "reference_tool": "R bfast, PSOCK cluster (19 workers, setup excluded)",
            "speedup_vs_reference_1t": ref1 / t1, "speedup_vs_reference_multi": refm / t8,
            "agreement": {"outputs_1t_equal_8t": same},
            "notes": f"published and reference: results/{csv_name}. rust_1t / rust_8t: one cold run each, n_jobs=1 "
                     "first (the original protocol); warm best of 3 also given. Published: n_jobs=-1 = 19 threads; "
                     "here 8."})


def run_monitor():
    single("BFAST Monitor", 7, "bfast_monitor_series.csv", "bfast_monitor_r_results.csv", "bfast_monitor",
           monitor_call, monitor_compare, "40 series of 150 obs (23/yr), history 'all', bfast_monitor per call",
           passes=10)
    v = np.stack([make_series(150, seed=s, break_at=110 if s % 2 == 0 else None) for s in range(20000)])
    multi("BFAST Monitor", 6, "parallel_bfast_monitor.csv", "20,000 series x 150 obs, fit_bfast_monitor_batch",
          lambda n: fit_bfast_monitor_batch(values_array=v, start_time=2000.0, monitor_start_time=2000.0 + 90 / FREQ,
                                            frequency=FREQ, n_jobs=n))


def run_lite():
    single("BFAST Lite", 9, "bfast_lite_series.csv", "bfast_lite_r_results.csv", "bfast_lite",
           lite_call, lite_compare, "40 series of 150 obs (23/yr), LWZ breaks, bfast_lite per call")
    v = np.stack([make_series(150, seed=s, break_at=80 if s % 2 == 0 else None) for s in range(2000)])
    multi("BFAST Lite", 7, "parallel_bfast_lite.csv", "2,000 series x 150 obs, fit_bfast_lite_batch",
          lambda n: fit_bfast_lite_batch(values_array=v, start_time=2000.0, frequency=FREQ, n_jobs=n))


def run_classic():
    single("BFAST (classic)", 8, "bfast_series.csv", "bfast_r_results.csv", "bfast",
           classic_call, classic_compare, "40 series of 300 obs (23/yr), iterative STL + breakpoints, bfast per call")
    v = np.stack([make_series(300, seed=s, break_at=150 if s % 2 == 0 else None) for s in range(400)])
    multi("BFAST (classic)", 8, "parallel_bfast.csv", "400 series x 300 obs, fit_bfast_batch",
          lambda n: fit_bfast_batch(values_array=v, start_time=2000.0, frequency=FREQ, n_jobs=n))


if __name__ == "__main__":
    common.start()
    which = sys.argv[1:] or ["monitor", "lite", "classic"]
    for w in which:
        {"monitor": run_monitor, "lite": run_lite, "classic": run_classic}[w]()
