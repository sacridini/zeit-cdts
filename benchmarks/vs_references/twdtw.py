"""TWDTW with the Rust core, on the workloads of cdts_vs_originals' compare_twdtw.py (single core) and
parallel_twdtw.py (multi-core).

Single core: the 45 labelled synthetic series and 3 class patterns of data/twdtw (alpha 0.1, beta 0.05, gamma 50,
max_time_warp 365), zeit._twdtw.run_twdtw for each (series, pattern), the mean time per pair of one pass (the
original protocol: each series' 3 calls timed together) and of the warm best of 3 runs (10 passes each); the
agreement with R twdtw's stored classifications and distances (data/twdtw/r_twdtw_results.csv).
Multi-core: run_twdtw_batch on 3,000 series of 40 steps against 1 pattern (the same synthetic batch and seed),
n_jobs=1 then n_jobs=8, one cold run each as the original, and the warm best of 3.

    python benchmarks/vs_references/twdtw.py
"""
import common  # first: thread variables

import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from zeit._twdtw import run_twdtw, run_twdtw_batch
from common import DATA, THREADS, once, pin, record, stored_row, timed, warm_best

ALPHA, BETA, GAMMA = 0.1, 0.05, 50.0


def single_core():
    d = DATA / "twdtw"
    patterns_csv = pd.read_csv(d / "patterns.csv")
    series_csv = pd.read_csv(d / "series.csv")
    meta = pd.read_csv(d / "series_meta.csv")
    r_res = pd.read_csv(d / "r_twdtw_results.csv")
    labels = sorted(patterns_csv["label"].unique().tolist())
    patterns = {}
    for lb in labels:
        sub = patterns_csv[patterns_csv["label"] == lb].sort_values("doy")
        patterns[lb] = (sub["value"].to_numpy(dtype=np.float64), sub["doy"].to_numpy(dtype=np.int32))
    ids = sorted(series_csv["series_id"].unique().tolist())
    series = []
    for sid in ids:
        sub = series_csv[series_csv["series_id"] == sid].sort_values("doy")
        series.append((sid, sub["value"].to_numpy(dtype=np.float64), sub["doy"].to_numpy(dtype=np.int32)))

    def dists(y, t):
        return {lb: run_twdtw(y, t, pv, pd_, alpha=ALPHA, beta=BETA, gamma=GAMMA, max_time_warp=365,
                              subsequence_matching=False) for lb, (pv, pd_) in patterns.items()}

    def one_pass():
        rows, times = [], []
        for sid, y, t in series:
            t0 = time.perf_counter()
            dd = dists(y, t)
            times.append(time.perf_counter() - t0)
            rows.append({"series_id": sid, "z_best": min(dd, key=dd.get), **{f"z_{lb}": dd[lb] for lb in labels}})
        return float(np.mean(times)) / len(labels) * 1000.0, pd.DataFrame(rows)

    pin(1)
    ms, out = timed(one_pass, "TWDTW")
    warm = timed(lambda: warm_best(lambda: [dists(y, t) for _ in range(10) for _, y, t in series]),
                 "TWDTW warm") / (10 * len(series) * len(labels)) * 1000.0
    m = out.merge(r_res, on="series_id").merge(meta, on="series_id")
    zf = np.concatenate([m[f"z_{lb}"].to_numpy() for lb in labels])
    rf = np.concatenate([m[f"dist_{lb}"].to_numpy() for lb in labels])
    ok = np.isfinite(zf) & np.isfinite(rf)
    agree = {"classification_agreement_vs_R": float((m.z_best == m.best_label).mean()),
             "pearson_correlation_raw_distances": float(pearsonr(zf[ok], rf[ok])[0]),
             "accuracy_vs_truth": float((m.z_best == m.true_label).mean())}
    st = stored_row("twdtw.csv", "twdtw_classification_vs_dtwSat")
    pub, ref = float(st["cdts_time_ms_mean"]), float(st["reference_time_ms_mean"])
    record({"table": "single-core", "row": "TWDTW", "order": 4,
            "workload": f"{len(series)} series x {len(labels)} patterns (doy, 1 band), run_twdtw per pair",
            "unit": "ms/call (series-pattern pair)", "published_cpp_1t": pub, "rust_1t": ms,
            "rust_1t_warm_best3": warm, "ratio_rust_cpp_1t": ms / pub, "reference_1t": ref,
            "reference_tool": st["reference_tool"], "speedup_vs_reference_1t": ref / ms,
            "published_speedup": float(st["speedup_x"]), "agreement": agree,
            "agreement_stored": {"classification_agreement_vs_R": float(st["primary_metric_value"]),
                                 "pearson_correlation_raw_distances": float(st["secondary_metric_value"]),
                                 "accuracy_vs_truth": 1.0},
            "notes": "published and reference: results/twdtw.csv; agreement against data/twdtw/r_twdtw_results.csv. "
                     "The time includes the Python wrapper's list conversions, as the original's did."})


def multi_core():
    doy = np.arange(1, 362, 9, dtype=np.int32)
    tt = np.linspace(0, 2 * np.pi, len(doy))
    pattern = 0.2 + 0.6 * np.exp(-((tt - np.pi) ** 2) / (2 * 0.8 ** 2))
    rng = np.random.RandomState(0)
    values = np.zeros((3000, len(doy)))
    for i in range(3000):
        noise = rng.normal(0, 0.05, len(doy))
        shift = rng.randint(-2, 3)
        values[i] = np.roll(pattern, shift) + noise
    v4, p2 = values[:, None, :, None], pattern[:, None]
    stored_series = pd.read_csv(DATA / "twdtw_parallel" / "series.csv")
    same_input = bool(np.allclose(stored_series["value"].to_numpy().reshape(3000, len(doy)), values, rtol=0, atol=1e-15))
    res = {}

    def go(n):
        res[n] = run_twdtw_batch(v4, doy, p2, doy, n_jobs=n)

    pin(1)
    t1 = timed(lambda: once(lambda: go(1)), "1T") * 1000
    pin(THREADS)
    t8 = timed(lambda: once(lambda: go(THREADS)), "8T") * 1000
    pin(1)
    w1 = timed(lambda: warm_best(lambda: go(1)), "1T warm") * 1000
    pin(THREADS)
    w10 = timed(lambda: warm_best(lambda: go(THREADS)), "8T warm") * 1000
    pin(None)
    st = {r["tool"]: r for r in common.read_csv_rows(common.ORIGINALS / "results" / "parallel_twdtw.csv")}
    c, r = st["cdts"], st["R_twdtw_base_parallel"]
    p1, pm = float(c["time_ms_single_thread"]), float(c["time_ms_parallel"])
    ref1, refm = float(r["time_ms_single_thread"]), float(r["time_ms_parallel"])
    record({"table": "multi-core", "row": "TWDTW", "order": 9,
            "workload": "3,000 series x 40 steps x 1 pattern, run_twdtw_batch", "unit": "ms (batch)",
            "published_cpp_1t": p1, "published_cpp_multi": pm, "published_threads": int(c["n_workers_parallel"]),
            "rust_1t": t1, "rust_8t": t8, "rust_1t_warm_best3": w1, "rust_8t_warm_best3": w10,
            "ratio_rust_cpp_1t": t1 / p1, "ratio_rust8_cpp19": t8 / pm,
            "reference_1t": ref1, "reference_multi": refm, "reference_workers": int(r["n_workers_parallel"]),
            "reference_tool": "R twdtw, PSOCK cluster (19 workers, setup excluded)",
            "speedup_vs_reference_1t": ref1 / t1, "speedup_vs_reference_multi": refm / t8,
            "agreement": {"outputs_1t_equal_8t": bool(np.array_equal(res[1], res[THREADS], equal_nan=True)),
                          "input_equals_stored_series_csv": same_input},
            "notes": "published and reference: results/parallel_twdtw.csv. rust_1t / rust_8t: one cold run each, "
                     "n_jobs=1 first (the original protocol); warm best of 3 also given. Published: n_jobs=-1 = 19 "
                     "threads; here 8."})


if __name__ == "__main__":
    common.start()
    single_core()
    multi_core()
