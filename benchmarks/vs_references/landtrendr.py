"""LandTrendr with the Rust core, on the workloads of cdts_vs_originals' compare_landtrendr.py (single core) and of
scratch/benchmark_cdts_tile.py (the full Landsat tile 215_067, multi-core).

Single core: the 30 synthetic 30-year scenarios (x1000, max_segments 6, best_model_proportion 0.75, ...; the same
generator and seed), zeit._landtrendr.run_landtrendr per series, the mean per-call time of one pass (the original
protocol) and the warm best of 3 runs (10 passes each). Agreement: the vertex years and integer-truncated values
against the GDL outputs stored in data/landtrendr/comparison_detail.csv, and the same for the broad battery of 300
series (comparison_detail_broad.csv).
Tile (--tile): the 41 yearly medoid GeoTIFFs (band 1, 1985-2025, 7478 x 7319 = 54.7M px) in blocks of 1000 rows,
run_landtrendr_batch with the GEE run's parameters (modifier -1) on n_jobs=8 then n_jobs=1 per block (the
original ran its 20 threads then 1), I/O excluded; the two outputs must be identical.

    python benchmarks/vs_references/landtrendr.py
    python benchmarks/vs_references/landtrendr.py --tile C:/Users/eduar/Documents/215_067 [--blocks N]
"""
import common  # first: thread variables

import argparse
import ast
import glob
import os
import time

import numpy as np
import pandas as pd

from zeit._landtrendr import run_landtrendr, run_landtrendr_batch
from common import DATA, THREADS, pin, record, stored_row, timed, warm_best

SCALE = 1000.0
YEARS = np.arange(1990, 2020)
N = len(YEARS)
BASELINE = 0.70
CDTS_KW = dict(max_segments=6, pval_threshold=0.05, recovery_threshold=0.25, spike_threshold=0.9,
               best_model_proportion=0.75, vertex_count_overshoot=3, min_observations_needed=6, modifier=1.0)
BROAD_YEARS = np.arange(1985, 2021)


def make_scenarios(seed=42):
    """compare_landtrendr.py's make_scenarios, unchanged."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(10):
        noise = rng.normal(0, 0.015, N)
        out.append((f"stable_{i}", np.clip(BASELINE + noise, 0.05, 0.95)))
    for i in range(10):
        dist_year_idx = rng.integers(6, 20)
        magnitude = rng.uniform(0.20, 0.40)
        recovery_len = rng.integers(3, 9)
        noise = rng.normal(0, 0.012, N)
        vals = np.full(N, BASELINE)
        vals[dist_year_idx] = BASELINE - magnitude
        end_recover = min(dist_year_idx + recovery_len, N - 1)
        if end_recover > dist_year_idx:
            vals[dist_year_idx:end_recover + 1] = np.linspace(BASELINE - magnitude, BASELINE,
                                                              end_recover - dist_year_idx + 1)
        if end_recover + 1 < N:
            vals[end_recover + 1:] = BASELINE
        out.append((f"dist_recover_{i}", np.clip(vals + noise, 0.05, 0.95)))
    for i in range(10):
        dist_year_idx = rng.integers(6, 22)
        magnitude = rng.uniform(0.25, 0.45)
        noise = rng.normal(0, 0.012, N)
        vals = np.full(N, BASELINE)
        vals[dist_year_idx:] = BASELINE - magnitude
        out.append((f"dist_permanent_{i}", np.clip(vals + noise, 0.05, 0.95)))
    return [{"id": i, "values": np.round(v * SCALE, 4), "params": dict(CDTS_KW)} for i, v in out]


def make_broad_rows(seed=11, reps=25):
    """compare_landtrendr.py's make_broad_rows, unchanged."""
    rng = np.random.default_rng(seed)
    n = len(BROAD_YEARS)
    rows = []
    kinds = ["stable", "trend", "drop_recover", "drop_perm", "gain", "two_events"]
    settings = [(6, 0.75, 0.9, 0.25, 0.05), (6, 1.25, 0.9, 0.25, 0.05), (4, 0.75, 1.0, 0.25, 0.05),
                (5, 0.75, 0.75, 0.5, 0.1), (6, 1.0, 0.9, 1.0, 0.05)]
    k = 0
    for kind in kinds:
        for rep in range(reps):
            base = rng.uniform(550, 800)
            noise = rng.uniform(8, 30)
            v = np.full(n, base)
            if kind == "trend":
                v = base + rng.uniform(-6, 6) * np.arange(n)
            elif kind in ("drop_recover", "drop_perm", "two_events"):
                t = rng.integers(4, n - 6)
                mag = rng.uniform(150, 450)
                v[t:] -= mag
                if kind != "drop_perm":
                    length = rng.integers(3, 15)
                    e = min(t + length, n - 1)
                    v[t:e + 1] = np.linspace(base - mag, base, e - t + 1)
                    v[e + 1:] = base
                if kind == "two_events":
                    t2 = rng.integers(max(t + 5, 6), n - 2) if t + 5 < n - 3 else n - 3
                    v[t2:] -= rng.uniform(100, 300)
            elif kind == "gain":
                t = rng.integers(4, n - 6)
                v[t:] += rng.uniform(100, 300) * np.minimum(1, (np.arange(n - t) + 1) / rng.integers(1, 8))
            v = v + rng.normal(0, noise, n)
            if rng.random() < 0.25:
                v[rng.integers(1, n - 1)] -= rng.uniform(100, 300)
            if rng.random() < 0.3:
                miss = rng.choice(n, rng.integers(1, 5), replace=False)
                if rng.random() < 0.4:
                    miss = np.append(miss, 0)
                if rng.random() < 0.4:
                    miss = np.append(miss, n - 1)
                v[miss] = np.nan
            v = np.round(v, 4)
            for modifier in (1.0, -1.0):
                ms, bmp, spike, rthr, pv = settings[k % len(settings)]
                rows.append(dict(id=f"{kind}_{rep}_m{int(modifier)}", values=v.copy(), params=dict(
                    max_segments=ms, modifier=modifier, best_model_proportion=bmp, spike_threshold=spike,
                    recovery_threshold=rthr, pval_threshold=pv, vertex_count_overshoot=3, min_observations_needed=6)))
                k += 1
    return rows


def run_all(rows, years):
    return {r["id"]: run_landtrendr(years, r["values"], **r["params"]) for r in rows}


def compare(rows, out, detail_csv):
    gdl = pd.read_csv(DATA / "landtrendr" / detail_csv)
    gdl = {r.id: (ast.literal_eval(r.gdl_years), ast.literal_eval(r.gdl_values)) for r in gdl.itertuples()}
    years_ok = values_ok = 0
    for r in rows:
        m = r["params"]["modifier"]
        cy = [v["year"] for v in out[r["id"]]]
        cv = [float(np.trunc(v["value"] * m)) for v in out[r["id"]]]
        gy, gv = gdl[r["id"]]
        if cy == list(gy):
            years_ok += 1
            values_ok += int(np.max(np.abs(np.array(cv) - np.array(gv))) <= 1.0)
    return {"vertex_years_exact_match_rate": years_ok / len(rows), "vertex_values_match_rate_trunc_tol1": values_ok / len(rows),
            "n": len(rows)}


def input_check(rows, input_csv):
    df = pd.read_csv(DATA / "landtrendr" / input_csv)
    ycols = [c for c in df.columns if c.isdigit()]
    stored = df.set_index("id")[ycols].to_numpy(dtype=float)
    mine = np.stack([r["values"] for r in rows])
    return bool(np.allclose(stored, mine, rtol=0, atol=6e-5, equal_nan=True))


def single_core():
    rows = make_scenarios()
    broad = make_broad_rows()
    pin(1)

    def one_pass():
        t0 = time.perf_counter()
        out = run_all(rows, YEARS)
        return (time.perf_counter() - t0) / len(rows) * 1000.0, out

    ms, out = timed(one_pass, "LandTrendr 30")
    warm = timed(lambda: warm_best(lambda: [run_all(rows, YEARS) for _ in range(10)]), "warm") / (10 * len(rows)) * 1000
    pin(None)
    agree = compare(rows, out, "comparison_detail.csv")
    agree_broad = compare(broad, run_all(broad, BROAD_YEARS), "comparison_detail_broad.csv")
    st = stored_row("landtrendr.csv", "landtrendr_vs_original_idl")
    stb = stored_row("landtrendr.csv", "landtrendr_vs_original_idl_broad")
    pub, ref = float(st["cdts_time_ms_mean"]), float(st["reference_time_ms_mean"])
    record({"table": "single-core", "row": "LandTrendr", "order": 2,
            "workload": "30 synthetic 30-year series (x1000), run_landtrendr per series", "unit": "ms/call",
            "published_cpp_1t": pub, "rust_1t": ms, "rust_1t_warm_best3": warm, "ratio_rust_cpp_1t": ms / pub,
            "reference_1t": ref, "reference_tool": "original LandTrendr-2012 IDL under GDL 1.1.2 (in-interpreter, warm)",
            "speedup_vs_reference_1t": ref / ms, "published_speedup": float(st["speedup_x"]),
            "agreement": {"scenarios30": agree, "broad300": agree_broad,
                          "inputs_equal_stored_csv": input_check(rows, "scenarios30_input.csv")
                          and input_check(broad, "broad_input.csv")},
            "agreement_stored": {"scenarios30": {"vertex_years_exact_match_rate": float(st["primary_metric_value"]),
                                                 "vertex_values_match_rate_trunc_tol1": float(st["secondary_metric_value"])},
                                 "broad300": {"vertex_years_exact_match_rate": float(stb["primary_metric_value"]),
                                              "vertex_values_match_rate_trunc_tol1": float(stb["secondary_metric_value"])}},
            "notes": "published and reference: results/landtrendr.csv; one pass of the 30 series (the original "
                     "protocol), warm best of 3 runs of 10 passes also given. Broad battery (300 series): "
                     f"published C++ {stb['cdts_time_ms_mean']} ms/call, GDL {stb['reference_time_ms_mean']}."})


LT_PARAMS = dict(max_segments=6, pval_threshold=0.05, recovery_threshold=0.25, prevent_fast_recovery=True,
                 spike_threshold=0.9, best_model_proportion=0.75, vertex_count_overshoot=3, min_observations_needed=6,
                 modifier=-1.0)


def tile(data_dir, max_blocks=None, block_rows=1000):
    import rasterio
    from rasterio.windows import Window
    files = sorted(glob.glob(os.path.join(data_dir, "landsat_medoid_*.tif")))
    years = np.arange(1985, 1985 + len(files), dtype=np.int32)
    srcs = [rasterio.open(f) for f in files]
    width, height = srcs[0].width, srcs[0].height
    print(f"tile: {width} x {height} = {width * height:,} px, {len(files)} years", flush=True)
    tot = {"io": 0.0, "multi": 0.0, "single": 0.0, "pixels": 0, "vertices": 0}
    identical = True
    for b, row in enumerate(range(0, height, block_rows)):
        if max_blocks is not None and b >= max_blocks:
            break
        nr = min(block_rows, height - row)
        t0 = time.perf_counter()
        stack = np.empty((nr, width, len(files)), dtype=np.float64)
        for k, src in enumerate(srcs):
            stack[:, :, k] = src.read(1, window=Window(0, row, width, nr))
        tot["io"] += time.perf_counter() - t0
        res = {}

        def go(n):
            t = time.perf_counter()
            res[n] = run_landtrendr_batch(years=years, values=stack, n_jobs=n, **LT_PARAMS)
            return time.perf_counter() - t

        pin(THREADS)
        t_multi = timed(lambda: go(THREADS), f"block {b} 8T")
        pin(1)
        t_single = timed(lambda: go(1), f"block {b} 1T")
        pin(None)
        same = all(np.array_equal(x, y) for x, y in zip(res[1], res[THREADS]))
        identical &= same
        tot["multi"] += t_multi
        tot["single"] += t_single
        tot["pixels"] += nr * width
        tot["vertices"] += int(res[1][1].sum())
        print(f"  block {b + 1}: rows {row}-{row + nr - 1} | 8T {t_multi:6.2f} s | 1T {t_single:7.2f} s | "
              f"{t_single / t_multi:4.2f}x | identical {same}", flush=True)
        del res, stack
    for s in srcs:
        s.close()
    full = tot["pixels"] == width * height
    pub = {r["id"]: r for r in common.read_csv_rows(common.ORIGINALS / "results" / "parallel_landtrendr.csv")}
    p = pub["parallel_landtrendr_tile_215_067"]
    p1, pm = float(p["time_ms_single_thread"]) / 1000, float(p["time_ms_parallel"]) / 1000
    print(f"total {tot['pixels']:,} px: 8T {tot['multi']:.2f} s, 1T {tot['single']:.2f} s, I/O {tot['io']:.1f} s",
          flush=True)
    if not full:
        print("(not the whole tile: nothing recorded)")
        return
    record({"table": "multi-core", "row": "LandTrendr (tile)", "order": 1,
            "workload": f"tile 215_067, {tot['pixels']:,} px x {len(files)} yrs, run_landtrendr_batch, I/O excluded",
            "unit": "s", "published_cpp_1t": p1, "published_cpp_multi": pm, "published_threads": 20,
            "rust_1t": tot["single"], "rust_8t": tot["multi"], "ratio_rust_cpp_1t": tot["single"] / p1,
            "ratio_rust8_cpp19": tot["multi"] / pm, "rust_px_per_s_1t": tot["pixels"] / tot["single"],
            "rust_px_per_s_8t": tot["pixels"] / tot["multi"], "io_s": tot["io"],
            "agreement": {"outputs_1t_equal_8t_every_block": identical, "total_vertices": tot["vertices"]},
            "notes": "published: scratch/cdts_benchmark_results.json (cdts 0.18.0, 20 threads). Same blocks and "
                     "order (multi-thread then single per block); no reference tool on this row (GEE's 1,396 s was "
                     "another tile on Google's cluster)."})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", help="folder of the 41 landsat_medoid_YYYY.tif of tile 215_067 (runs the tile only)")
    ap.add_argument("--blocks", type=int, help="only the first N blocks of 1000 rows (a trial; nothing recorded)")
    a = ap.parse_args()
    common.start()
    if a.tile:
        tile(a.tile, a.blocks)
    else:
        single_core()
