"""CCDC with the Rust core, on the workloads of cdts_vs_originals' compare_ccdc.py (single core) and
benchmark_ccdc_tile.py (the full Rondonia stack, multi-core).

Single core: the 200 synthetic Landsat-like pixels of the synthetic battery (16-day grid 2000-2019, the same
generator and seed), zeit._ccdc.run_ccdc per pixel with the original's defaults (conse 6, change probability 0.99,
8 coefficients, detection bands Green..SWIR2, Tmask Green/SWIR1, a constant thermal band), the mean per-pixel
time of one pass (the original protocol) and the warm best of 3. Agreement: every model against the Octave
outputs stored in data/ccdc_compare/synthetic_octave_rec_cg.csv, and the same for the 150 real pixels
(real_octave_rec_cg.csv), as compare_ccdc.py decides a full match.
Stack (--stack): the 1690 x 1675 px x 125 dates stack (6 bands of C2 DNs + QA_PIXEL) in blocks of 256 rows, DN to
SR x 10000 and QA to Fmask per block, run_ccdc_batch (10 segments) on n_jobs=8 then n_jobs=1 (the original ran its
20 threads then 1), I/O and conversion excluded; the two outputs must be identical.

    python benchmarks/vs_references/ccdc.py
    python benchmarks/vs_references/ccdc.py --stack C:/Users/eduar/Downloads/CCDC_Stack_Rondonia.tif \\
        --dates C:/Users/eduar/Downloads/CCDC_Stack_Rondonia_Dates.csv
"""
import common  # first: thread variables

import argparse
import time
from datetime import date

import numpy as np
import pandas as pd

from zeit._ccdc import run_ccdc, run_ccdc_batch
from common import DATA, THREADS, pin, record, stored_row, timed, warm_best

CONSE, CHANGE_PROB, NUM_C = 6, 0.99, 8
THERMAL_RAW = 2932
DATENUM = 366


def synthetic_battery(n=200, seed=2024):
    """compare_ccdc.py's synthetic_battery, unchanged."""
    rng = np.random.default_rng(seed)
    d0 = date(2000, 1, 1).toordinal()
    ordinals = np.arange(d0, date(2019, 12, 31).toordinal(), 16)
    T = len(ordinals)
    w = 2 * np.pi / 365.25
    base = np.array([450, 750, 550, 3000, 1800, 900], float)
    delta = np.array([150, 250, 450, -1300, 900, 700], float)
    optical = np.zeros((T, n, 6))
    fmask = np.zeros((T, n), int)
    kinds = []
    for p in range(n):
        kind = ["stable", "one_break", "two_breaks", "gradual", "dirty", "one_break",
                "cloudy", "snowy"][p % 8] if p >= 16 else ["cloudy", "snowy"][p % 2]
        kinds.append(kind)
        lvl = base * rng.uniform(0.8, 1.2, 6)
        amp = lvl * rng.uniform(0.05, 0.25, 6)
        phase = rng.uniform(0, 2 * np.pi)
        t = ordinals.astype(float)
        y = lvl[None, :] + amp[None, :] * np.cos(w * t[:, None] + phase)
        if kind in ("one_break", "two_breaks", "dirty"):
            tb = rng.integers(T // 4, 3 * T // 4)
            y[tb:] += delta * rng.uniform(0.6, 1.2)
            if kind == "two_breaks":
                tb2 = min(T - 20, tb + rng.integers(60, 150))
                y[tb2:] -= delta * rng.uniform(0.5, 1.0)
        if kind == "gradual":
            y += (t[:, None] - t[0]) / 365.25 * delta[None, :] * 0.03
        y += rng.normal(0, 1, y.shape) * lvl * rng.uniform(0.02, 0.05)
        q = np.zeros(T, int)
        q[rng.random(T) < 0.15] = 4
        q[rng.random(T) < 0.04] = 2
        q[rng.random(T) < 0.12] = 255
        if kind == "dirty":
            bad = (rng.random(T) < 0.06) & (q == 0)
            y[bad] += np.array([2500, 2500, 2500, 2000, 1500, 1000]) * rng.uniform(0.6, 1.2)
            bad2 = (rng.random(T) < 0.03) & (q == 0)
            y[bad2] *= 0.3
        if kind == "cloudy":
            q[(q == 0) & (rng.random(T) < 0.8)] = 4
        if kind == "snowy":
            s = (q == 0) & (rng.random(T) < 0.85)
            q[s] = 3
            y[s] = [6000, 6200, 6300, 6100, 900, 700]
        y[q == 4] = np.array([3000, 3200, 3400, 4000, 3800, 3000])
        optical[:, p, :] = y
        fmask[:, p] = q
    optical = np.clip(np.round(optical), -32000, 32000).astype(np.int16)
    return ordinals, optical, fmask, kinds


def real_battery():
    src = DATA / "ccdc_real_stack"
    dm = pd.read_csv(src / "date_manifest.csv")
    ordinals = np.array([date(y, 1, 1).toordinal() + d - 1 for y, d in zip(dm.year, dm.doy)])
    dn = np.load(src / "optical_i16.npy").astype(float)
    fmask = np.load(src / "fmask.npy").astype(int)
    sr = np.round(dn * 0.275 - 2000.0)
    sr[fmask == 255] = 0
    return ordinals, np.clip(sr, -32000, 32000).astype(np.int16), fmask, ["real"] * dn.shape[1]


def run_all(ordinals, optical, fmask):
    segs = {}
    for p in range(optical.shape[1]):
        bands = np.concatenate([optical[:, p, :].astype(float),
                                np.full((optical.shape[0], 1), THERMAL_RAW * 10.0 - 27315.0)], axis=1).T
        out = run_ccdc(ordinals, bands, fmask[:, p], conseq_anom=CONSE, chi2_prob_threshold=CHANGE_PROB,
                       tmax_cg_prob_threshold=1 - 1e-6, detection_bands=[1, 2, 3, 4, 5], num_c=NUM_C,
                       tmask_bands=[1, 4], thermal_band=6)
        # as compare_ccdc.py's run_cdts, inside its timing
        segs[p] = [dict(t_start=int(s["t_start"]), t_end=int(s["t_end"]), t_break=int(s["t_break"]),
                        change_prob=s.get("change_prob", np.nan), category=s.get("category", -1),
                        num_obs=s.get("num_obs", -1), rmse=np.array(s["rmse"]),
                        magnitude=np.array(s["magnitude"]), coefs=np.array(s["coefs"])) for s in out]
    return segs


def octave_models(csv):
    df = pd.read_csv(DATA / "ccdc_compare" / csv)
    segs = {}
    for _, r in df.iterrows():
        segs.setdefault(int(r.pos) - 1, []).append(dict(
            t_start=int(round(r.t_start)) - DATENUM, t_end=int(round(r.t_end)) - DATENUM,
            t_break=(int(round(r.t_break)) - DATENUM) if r.t_break > 0 else 0,
            change_prob=float(r.change_prob), category=int(r.category), num_obs=int(r.num_obs),
            coefs=np.array([[r[f"coef{b}_{c}"] for c in range(1, 9)] for b in range(1, 8)])))
    return segs


def compare(n, ref, got):
    """compare_ccdc.py's full-match rule: identical dates, category, num_obs, change_prob; coefficients to 1e-3."""
    dates_ok = full_ok = 0
    max_rel = 0.0
    for p in range(n):
        r, g = ref.get(p, []), got.get(p, [])
        key = lambda s: (s["t_start"], s["t_end"], s["t_break"])
        if [key(s) for s in r] != [key(s) for s in g]:
            continue
        dates_ok += 1
        if not r:
            full_ok += 1
            continue
        cr = np.concatenate([s["coefs"][:6].ravel() for s in r])
        cg = np.concatenate([np.pad(s["coefs"][:6, :8], ((0, 0), (0, max(0, 8 - s["coefs"].shape[1])))).ravel()
                             for s in g])
        rel = float(np.max(np.abs(cr - cg) / np.maximum(np.abs(cr), 1.0)))
        max_rel = max(max_rel, rel)
        meta = all(a["category"] == b["category"] and a["num_obs"] == b["num_obs"]
                   and abs(a["change_prob"] - b["change_prob"]) < 1e-9 for a, b in zip(r, g))
        full_ok += int(meta and rel < 1e-3)
    return {"model_dates_exact_match_rate": dates_ok / n, "full_model_match_rate": full_ok / n,
            "max_coef_rel_diff": max_rel, "n": n}


def single_core():
    ordinals, optical, fmask, _ = synthetic_battery()
    n = optical.shape[1]
    pin(1)

    def one_pass():
        t0 = time.perf_counter()
        out = run_all(ordinals, optical, fmask)
        return (time.perf_counter() - t0) / n * 1000.0, out

    ms, out = timed(one_pass, "CCDC synthetic")
    warm = timed(lambda: warm_best(lambda: run_all(ordinals, optical, fmask)), "warm") / n * 1000.0
    ro, rop, rf, _ = real_battery()
    t0 = time.perf_counter()
    out_real = run_all(ro, rop, rf)
    real_ms = (time.perf_counter() - t0) / rop.shape[1] * 1000.0
    pin(None)
    agree = {"synthetic200": compare(n, octave_models("synthetic_octave_rec_cg.csv"), out),
             "real150": compare(rop.shape[1], octave_models("real_octave_rec_cg.csv"), out_real)}
    st = stored_row("ccdc.csv", "ccdc_vs_original_matlab")
    sr = stored_row("ccdc.csv", "ccdc_vs_original_matlab_real")
    pub, ref = float(st["cdts_time_ms_mean"]), float(st["reference_time_ms_mean"])
    record({"table": "single-core", "row": "CCDC", "order": 3,
            "workload": f"{n} synthetic Landsat-like pixels (7 bands, 2000-2019 16-day), run_ccdc per pixel",
            "unit": "ms/px", "published_cpp_1t": pub, "rust_1t": ms, "rust_1t_warm_best3": warm,
            "ratio_rust_cpp_1t": ms / pub, "reference_1t": ref,
            "reference_tool": "original CCDC MATLAB under GNU Octave 11.3 (in-interpreter, warm)",
            "speedup_vs_reference_1t": ref / ms, "published_speedup": float(st["speedup_x"]),
            "real150": {"published_cpp_1t": float(sr["cdts_time_ms_mean"]), "rust_1t": real_ms,
                        "reference_1t": float(sr["reference_time_ms_mean"]),
                        "speedup_vs_reference_1t": float(sr["reference_time_ms_mean"]) / real_ms},
            "agreement": agree,
            "agreement_stored": {"synthetic200": {"model_dates_exact_match_rate": float(st["primary_metric_value"]),
                                                  "full_model_match_rate": float(st["secondary_metric_value"])},
                                 "real150": {"model_dates_exact_match_rate": float(sr["primary_metric_value"]),
                                             "full_model_match_rate": float(sr["secondary_metric_value"])}},
            "notes": "published and reference: results/ccdc.csv (synthetic row; the real 150 px row in real150, "
                     "timed by one pass after the synthetic ones). One pass, as compare_ccdc.py; warm best of 3 "
                     "also given."})


def qa_to_fmask(qa):
    q = np.nan_to_num(qa, nan=1).astype(np.int64)
    f = np.zeros(q.shape, dtype=np.int32)
    f[(q & (1 << 7)) != 0] = 1
    f[(q & (1 << 4)) != 0] = 2
    f[(q & (1 << 5)) != 0] = 3
    f[(q & (1 << 3)) != 0] = 4
    f[(q & 1) != 0] = 255
    f[q == 0] = 255
    return f


KW = dict(max_segments=10, return_coefs=True, conseq_anom=6, chi2_prob_threshold=0.99,
          tmax_cg_prob_threshold=0.999999, num_c=8, detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4])


def stack(raster, dates_csv, max_blocks=None, block_rows=256):
    import rasterio
    from rasterio.windows import Window
    dates = pd.read_csv(dates_csv)["Ordinal_Day"].to_numpy(np.int32)
    tot = dict(io=0.0, prep=0.0, multi=0.0, single=0.0, pixels=0, segments=0, breaks=0)
    identical = True
    with rasterio.open(raster) as src:
        H, W = src.height, src.width
        print(f"stack: {W} x {H} = {W * H:,} px, {len(dates)} dates", flush=True)
        for b, row in enumerate(range(0, H, block_rows)):
            if max_blocks is not None and b >= max_blocks:
                break
            nr = min(block_rows, H - row)
            t0 = time.perf_counter()
            data = src.read(window=Window(0, row, W, nr))
            tot["io"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            data = data.reshape(len(dates), 7, nr, W)
            opt = data[:, :6]
            fm = qa_to_fmask(data[:, 6])
            missing = (fm == 255) | ~np.all(np.isfinite(opt) & (opt != 0), axis=1)
            sr = np.where(missing[:, None], np.nan, np.round(opt * 0.275 - 2000.0))
            fm[missing] = 255
            values = np.ascontiguousarray(sr.transpose(2, 3, 1, 0))
            qa = np.ascontiguousarray(fm.transpose(1, 2, 0))
            del data, opt, sr
            tot["prep"] += time.perf_counter() - t0
            res = {}

            def go(n):
                t = time.perf_counter()
                res[n] = run_ccdc_batch(dates, values, qa, n_jobs=n, **KW)
                return time.perf_counter() - t

            pin(THREADS)
            t_multi = timed(lambda: go(THREADS), f"block {b} 8T")
            pin(1)
            t_single = timed(lambda: go(1), f"block {b} 1T")
            pin(None)
            (sm, cm), (ss, cs) = res[THREADS], res[1]
            same = bool(np.array_equal(cm, cs) and np.array_equal(sm, ss, equal_nan=True))
            identical &= same
            tot["multi"] += t_multi
            tot["single"] += t_single
            tot["pixels"] += nr * W
            tot["segments"] += int(cs.sum())
            tot["breaks"] += int((ss[:, :, 2] > 0).sum())
            print(f"  rows {row}-{row + nr - 1}: 8T {t_multi:6.2f} s | 1T {t_single:7.2f} s | "
                  f"{t_single / t_multi:4.2f}x | identical {same}", flush=True)
            del res, values, qa
    print(f"total {tot['pixels']:,} px: 8T {tot['multi']:.2f} s, 1T {tot['single']:.2f} s, "
          f"{tot['segments']:,} models, {tot['breaks']:,} breaks", flush=True)
    if tot["pixels"] != H * W:
        print("(not the whole stack: nothing recorded)")
        return
    import json
    old = json.loads((DATA / "ccdc_compare" / "tile_benchmark.json").read_text())
    record({"table": "multi-core", "row": "CCDC (stack)", "order": 2,
            "workload": f"Rondonia stack, {tot['pixels']:,} px x {len(dates)} dates, run_ccdc_batch, I/O excluded",
            "unit": "s", "published_cpp_1t": old["single"], "published_cpp_multi": old["multi"],
            "published_threads": old["threads"], "rust_1t": tot["single"], "rust_8t": tot["multi"],
            "ratio_rust_cpp_1t": tot["single"] / old["single"], "ratio_rust8_cpp19": tot["multi"] / old["multi"],
            "rust_px_per_s_1t": tot["pixels"] / tot["single"], "rust_px_per_s_8t": tot["pixels"] / tot["multi"],
            "io_s": tot["io"], "prep_s": tot["prep"],
            "agreement": {"outputs_1t_equal_8t_every_block": identical, "models": tot["segments"],
                          "breaks": tot["breaks"], "stored_models": old["segments"], "stored_breaks": old["breaks"]},
            "notes": "published: data/ccdc_compare/tile_benchmark.json (cdts 0.19.0, 20 threads). Same blocks and "
                     "order (multi-thread then single per block); no reference tool on this row."})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", help="CCDC_Stack_Rondonia.tif (runs the stack only)")
    ap.add_argument("--dates", help="CCDC_Stack_Rondonia_Dates.csv")
    ap.add_argument("--blocks", type=int, help="only the first N blocks of 256 rows (a trial; nothing recorded)")
    a = ap.parse_args()
    common.start()
    if a.stack:
        stack(a.stack, a.dates, a.blocks)
    else:
        single_core()
