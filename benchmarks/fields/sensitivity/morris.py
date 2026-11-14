"""Morris elementary effects of the field-boundary pipeline (phase 24e, task B). Resumable.

    python morris.py run --r 15 --n-jobs 8          # r trajectories x (k + 1) points x all the cases
    python morris.py run --r 15 --max-traj 3        # only the first 3 (to time it)

Design: Morris' trajectories on a grid of ``p = 6`` levels (``delta = 0.6``), ``k = 8`` factors in the unit cube,
the start and the order and the directions drawn from ``default_rng(SEED + traj)``; the seed of the Monte Carlo of
a trajectory is its index (one seed along a trajectory, so a step's effect is not seed noise). No SALib in the env.

Factors (unit position ``u`` in [0, 1] -> value):

    density      markers per Mpx, log scale 30 .. 3000 (markers of a window = max(1, round(density x pixels / 1e6)))
    min_distance 8 .. 40 pixels (integer)
    radius       marker_radius 0 .. 5 (integer)
    low          0.1 .. 0.5; high = max(0.6, low + 0.1)
    min_size     0 .. 200 pixels, linear in log1p (integer)
    gradient     tophat | sobel | multitemporal (u in thirds: the 6 grid levels fall two on each)
    mask_kind    none | relative | absolute 2000 (thirds)
    mask_q       10 .. 50 (integer): with ``relative``, the mask keeps swir16 > the q-th percentile of the window
                 (valid pixels); a dummy factor otherwise (its elementary effect is exactly 0 there)

Fixed: R = 200, method montecarlo, coloring knn, final image.
"""
import pyexpat  # noqa: F401
import argparse
import json
import os
import sys
import time
from dataclasses import replace

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("GDAL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import cases as C
import pipeline as P

SEED = 24500
R_FIXED = 200
FACTORS = ["density", "min_distance", "radius", "low", "min_size", "gradient", "mask_kind", "mask_q"]
GRADIENTS = ("tophat", "sobel", "multitemporal")
MASKS = ("none", "relative", "absolute")
LEVELS = 6
DELTA = LEVELS / (2 * (LEVELS - 1))


def design(r: int, seed: int = SEED):
    """(r, k + 1, k) unit-cube points, the factor moved at each step (r, k) and the signed step (r, k)."""
    k = len(FACTORS)
    grid = np.arange(LEVELS) / (LEVELS - 1)
    pts = np.zeros((r, k + 1, k))
    moved = np.zeros((r, k), int)
    signed = np.zeros((r, k))
    for t in range(r):
        rng = np.random.default_rng(seed + t)
        x = rng.choice(grid, size=k)
        order = rng.permutation(k)
        pts[t, 0] = x
        for s, f in enumerate(order):
            up, down = x[f] + DELTA <= 1 + 1e-9, x[f] - DELTA >= -1e-9
            sign = DELTA if (up and (not down or rng.random() < 0.5)) else -DELTA
            x = x.copy()
            x[f] = np.round(x[f] + sign, 10)
            pts[t, s + 1] = x
            moved[t, s] = f
            signed[t, s] = sign
    return pts, moved, signed


def third(u: float) -> int:
    return min(int(u * 3 + 1e-9), 2)


_META = {}


def meta_of(pixels: int, swir_valid: np.ndarray) -> dict:
    """What ``decode`` needs of a window: its pixels and the percentiles of its valid swir16 (q = 10 .. 50)."""
    return {"pixels": int(pixels), "q": {str(q): int(np.percentile(swir_valid, q)) for q in range(10, 51)}}


def meta_of_case(case) -> dict:
    if case.name not in _META:
        _META[case.name] = meta_of(case.valid.size, case.bands[C.SWIR][case.valid])
    return _META[case.name]


def case_meta() -> dict:
    """The meta of every mosaic window (cached in ``case_meta.json``; reads the mosaics, not the references)."""
    p = C.OUT / "case_meta.json"
    if p.exists():
        return json.loads(p.read_text())
    out = {}
    for m in C.index():
        arr, valid, _, _ = C._read(f"{m['name']}_{m['year']}")
        out[m["name"]] = meta_of(valid.size, arr[C.SWIR][valid])
    p.write_text(json.dumps(out))
    return out


def decode(u, meta: dict, traj: int):
    """(S1, S2) of the unit point ``u`` for a window with ``meta`` (the markers and a relative mask depend on it)."""
    dens = float(np.exp(np.log(30) + u[0] * (np.log(3000) - np.log(30))))
    n = max(1, int(round(dens * meta["pixels"] / 1e6)))
    kind = MASKS[third(u[6])]
    q = int(round(10 + 40 * u[7]))
    thr = {"none": None, "absolute": 2000.0, "relative": float(meta["q"][str(q)])}[kind]
    s1 = P.S1(gradient=GRADIENTS[third(u[5])], realizations=R_FIXED, markers=n,
              min_distance=float(round(8 + 32 * u[1])), marker_radius=int(round(5 * u[2])), coloring="knn",
              mask_threshold=thr, seed=traj)
    low = round(0.1 + 0.4 * u[3], 4)
    s2 = P.S2(low=low, high=round(max(0.6, low + 0.1), 4), min_size=int(round(np.expm1(u[4] * np.log1p(200)))),
              final="image")
    return s1, s2


def main():
    import run as RUN

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--r", type=int, default=15)
    ap.add_argument("--max-traj", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--groups", default=None)
    args = ap.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)
    cs = C.load_cases(groups=set(args.groups.split(",")) if args.groups else None)
    RUN.log(f"{len(cs)} cases, Morris r={args.r}, k={len(FACTORS)}")
    pts, moved, _ = design(args.r)
    done = RUN.done_keys()
    t_all = time.perf_counter()
    with open(RUN.RESULTS, "a", encoding="utf-8") as out:
        for t in range(args.max_traj or args.r):
            t0 = time.perf_counter()
            n_new = 0
            for s in range(pts.shape[1]):
                for case in cs:
                    s1, s2 = decode(pts[t, s], meta_of_case(case), t)
                    if (case.id, s1.key(), s2.key()) in done:
                        continue
                    b = P.stage1(case, s1, n_jobs=args.n_jobs)
                    tt = time.perf_counter()
                    lab, bl = P.stage2(case, s1, b, s2)
                    resp = P.evaluate(case, lab, bl)
                    row = RUN.row_base(case, s1, s2, b)
                    row.update({"study": "morris", "traj": t, "step": s})
                    row.update(resp)
                    row["s2_seconds"] = time.perf_counter() - tt
                    out.write(json.dumps(row, default=lambda o: None if o is None else float(o)) + "\n")
                    done.add((case.id, s1.key(), s2.key()))
                    n_new += 1
                out.flush()
            RUN.log(f"trajectory {t}: {n_new} rows in {time.perf_counter() - t0:.0f}s "
                    f"(total {(time.perf_counter() - t_all) / 60:.1f} min)")
    RUN.log("done")


if __name__ == "__main__":
    main()
