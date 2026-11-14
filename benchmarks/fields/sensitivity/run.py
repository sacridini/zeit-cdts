"""Runner of the sensitivity studies of field boundaries (phase 24e). Resumable: finished rows are skipped.

    python run.py convergence                 # R in {25..2000} x 3 seeds, notebook parameters, Monte Carlo
    python run.py exact                       # method="exact" (N markers) and a low/high grid
    python run.py oat --R 200                 # one factor at a time around the notebook defaults
    python run.py ... --groups chaco,lem --max-per-group 10   # a subset

Results: ``ZEIT_FIELDS_DATA/sensitivity/results.jsonl`` (one row per case x parameter set x seed, appended as
it goes), the stage-1 maps in ``.../cache``, the masked shares in ``.../mask_shares.csv``. One process at a
time; stage 1 runs on ``--n-jobs`` threads (default 8), everything else is single-threaded.
"""
import pyexpat  # noqa: F401  (first, so rasterio's expat is the conda one)
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("GDAL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import cases as C
import pipeline as P

RESULTS = C.OUT / "results.jsonl"
CONV_R = (25, 50, 100, 200, 500, 1000, 2000)
CONV_SEEDS = (0, 1, 2)
BASE1, BASE2 = P.S1(), P.S2()

# One-factor-at-a-time levels (the notebook's value is in each list).
OAT_S1 = {
    "markers": (250, 500, 1000, 2000, 4000),
    "min_distance": (0, 5, 10, 20, 40),
    "marker_radius": (0, 1, 3, 5),
    "coloring": ("knn", "unique"),
    "gradient": ("tophat", "sobel", "multitemporal"),
    "mask_threshold": (None, 1500, 2000, 2500),
}
OAT_S2 = {
    "low": (0.1, 0.2, 0.3, 0.4, 0.5),
    "high": (0.4, 0.5, 0.6, 0.7, 0.8),
    "min_size": (0, 10, 25, 50, 100, 200),
    "final": ("image", "gradient"),
}
EXACT_LOW = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)   # the first two added after the first pass
EXACT_HIGH = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
EXACT_MARKERS = (500, 1000, 2000)
EXACT_DENSITIES = (30, 100, 300, 1000, 3000)      # markers per Mpx
EXACT_DENSITY_LOW = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
EXACT_DENSITY_HIGH = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def done_keys():
    done = set()
    if RESULTS.exists():
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:   # a half-written last line after a crash
                    continue
                done.add((r["case"], r["s1_key"], r["s2_key"]))
    return done


def row_base(case, s1, s2, b):
    d = {"case": case.id, "mosaic": case.name, "group": case.group, "ref": case.ref, "kind": case.kind,
         "s1_key": s1.key(), "s2_key": s2.key(), "seed": s1.seed if s1.method == "montecarlo" else -1}
    d.update({f"s1_{k}": v for k, v in asdict(s1).items() if k != "seed"})
    d.update({f"s2_{k}": v for k, v in asdict(s2).items()})
    d["s1_seconds"] = b.seconds
    d["realizations"] = b.r
    return d


def run_config(cs, s1, s2s, done, out, n_jobs, tag=None):
    """Stage 1 once per case, then every ``s2`` in ``s2s`` (skipping the finished ones)."""
    t0 = time.perf_counter()
    n_new = 0
    for case in cs:
        todo = [s2 for s2 in s2s if (case.id, s1.key(), s2.key()) not in done]
        if not todo:
            continue
        b = P.stage1(case, s1, n_jobs=n_jobs)
        for s2 in todo:
            t = time.perf_counter()
            lab, bl = P.stage2(case, s1, b, s2)
            resp = P.evaluate(case, lab, bl)
            row = row_base(case, s1, s2, b)
            if tag is not None:
                row["tag"] = tag
            row.update(resp)
            row["s2_seconds"] = time.perf_counter() - t
            out.write(json.dumps(row, default=lambda o: None if o is None else float(o)) + "\n")
            done.add((case.id, s1.key(), s2.key()))
            n_new += 1
        out.flush()
    if n_new:
        log(f"  {s1.key()}: {n_new} rows in {time.perf_counter() - t0:.0f}s")


def dedupe(items):
    seen, out = set(), []
    for x in items:
        if x.key() not in seen:
            seen.add(x.key())
            out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["convergence", "exact", "exactdensity", "oat", "masks"])
    ap.add_argument("--R", type=int, default=200, help="realizations of the OAT study")
    ap.add_argument("--seeds", type=int, default=2, help="seeds of the OAT study")
    ap.add_argument("--groups", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-per-group", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)

    cs = C.load_cases(groups=set(args.groups.split(",")) if args.groups else None,
                      only=set(args.only.split(",")) if args.only else None, max_per_group=args.max_per_group)
    log(f"{len(cs)} cases, study {args.study}")
    if args.study == "masks":
        import pandas as pd

        rows = [r for c in cs for r in C.mask_shares(c)]
        pd.DataFrame(rows).to_csv(C.OUT / "mask_shares.csv", index=False)
        log("mask_shares.csv written")
        return

    done = done_keys()
    with open(RESULTS, "a", encoding="utf-8") as out:
        if args.study == "convergence":
            for seed in CONV_SEEDS:
                for R in CONV_R:
                    run_config(cs, replace(BASE1, realizations=R, seed=seed), [BASE2], done, out, args.n_jobs)
        elif args.study == "exact":
            grid = [P.S2(low=lo, high=hi) for lo in EXACT_LOW for hi in EXACT_HIGH if lo <= hi]
            for m in EXACT_MARKERS:
                run_config(cs, replace(BASE1, method="exact", markers=m), grid, done, out, args.n_jobs)
        elif args.study == "exactdensity":
            # The marker count of a window is not comparable across windows (24 kpx to 1.8 Mpx): the exact
            # probability with N = density x pixels, density in markers per Mpx.
            grid = [P.S2(low=lo, high=hi) for lo in EXACT_DENSITY_LOW for hi in EXACT_DENSITY_HIGH if lo <= hi]
            for d in EXACT_DENSITIES:
                for case in cs:
                    n = max(2, int(round(d * case.valid.size / 1e6)))
                    run_config([case], replace(BASE1, method="exact", markers=n), grid, done, out, args.n_jobs,
                               tag=f"exact density {d}/Mpx")
                log(f"  density {d}/Mpx done")
        else:
            seeds = range(args.seeds)
            for seed in seeds:
                base = replace(BASE1, realizations=args.R, seed=seed)
                s1s = [base]
                for name, levels in OAT_S1.items():
                    s1s += [replace(base, **{name: v}) for v in levels]
                s1s = dedupe(s1s)
                s2s = dedupe([BASE2] + [replace(BASE2, **{n: v}) for n, lv in OAT_S2.items() for v in lv])
                log(f"seed {seed}: {len(s1s)} stage-1 configurations")
                for s1 in s1s:
                    # the stage-2 factors only on the base maps
                    run_config(cs, s1, s2s if s1.key() == base.key() else [BASE2], done, out, args.n_jobs)
    log("done")


if __name__ == "__main__":
    main()
