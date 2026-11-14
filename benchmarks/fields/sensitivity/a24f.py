"""24f, task A: the grid of the extended spatial cross-validation. Resumable (one CSV per stage-1 configuration).

    python a24f.py run [--n-jobs 8] [--threads 6]     # 64 stage-1 maps x 62 mosaics, then stage 2 on each
    python a24f.py timing                              # one configuration, timed

Stage 1 (R = 200, seed 0, min_distance 20, radius 3, knn): gradient {tophat, sobel, multitemporal (max),
multimean (mean)} x marker density {300, 600, 1000, 2000} per Mpx x mask {none, q20, q35, abs2000}.
Stage 2 (the cheap responses of ``fast.py``: F and recall at 1 px, area-weighted D, crossing share):
low {0.1 .. 0.7} (high = max(0.6, low + 0.1)) x min_size {0, 2, 5, 10, 25, 50} (multimean: {0, 5, 25}).
Output: ``sensitivity/f24/A/<grad>_d<dens>_<mask>.csv``.
"""
import argparse
import sys
import time

import pandas as pd

import s24f as S
import cases as C
import pipeline as P

DIR = S.OUT24 / "A"


def path_for(grad, dens, mask):
    return DIR / f"{grad}_d{dens}_{mask}.csv"


def configs():
    return [(g, d, m) for g in S.GRADS for d in S.DENS for m in S.MASKS]


def run_one(cs, grad, dens, mask, n_jobs, threads):
    t0 = time.perf_counter()
    bs, keeps = [], []
    for c in cs:
        s1 = S.s1_for(c, grad, dens, mask)
        bs.append(P.stage1(c, s1, n_jobs=n_jobs))
        keeps.append(c.keep(s1.mask_threshold))
    t1 = time.perf_counter()
    sizes = S.sizes_for(grad)
    cfg = {"grad": grad, "dens": dens, "mask": mask}

    def work(i):
        return S.case_rows_mc(cs[i], keeps[i], bs[i], cfg, S.LOWS, sizes)

    rows = [r for rr in S.pmap(work, range(len(cs)), threads) for r in rr]
    DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    p = path_for(grad, dens, mask)
    df.to_csv(p.with_name(p.stem + ".tmp"), index=False)
    p.with_name(p.stem + ".tmp").replace(p)
    S.log(f"{grad} d{dens} {mask}: stage 1 {t1 - t0:.0f}s, stage 2 {time.perf_counter() - t1:.0f}s, {len(df)} rows")


LOWS_LOW = (0.02, 0.05)


def lows(cs, n_jobs, threads):
    """The edge check: low 0.02 and 0.05 (below the task's grid) on every configuration, from the cached maps."""
    for grad, dens, mask in configs():
        p = DIR / f"lows_{grad}_d{dens}_{mask}.csv"
        if p.exists():
            continue
        bs, keeps = [], []
        for c in cs:
            s1 = S.s1_for(c, grad, dens, mask)
            bs.append(P.stage1(c, s1, n_jobs=n_jobs))
            keeps.append(c.keep(s1.mask_threshold))
        cfg = {"grad": grad, "dens": dens, "mask": mask}
        sizes = S.sizes_for(grad)
        rows = [r for rr in S.pmap(lambda i: S.case_rows_mc(cs[i], keeps[i], bs[i], cfg, LOWS_LOW, sizes),
                                   range(len(cs)), threads) for r in rr]
        pd.DataFrame(rows).to_csv(p.with_name(p.stem + ".tmp"), index=False)
        p.with_name(p.stem + ".tmp").replace(p)
        S.log(f"lows {grad} d{dens} {mask}")
    S.log("done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "timing", "lows"])
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=S.THREADS)
    a = ap.parse_args()
    cs = C.load_cases()
    S.log(f"{len(cs)} cases")
    if a.cmd == "lows":
        return lows(cs, a.n_jobs, a.threads)
    S.preload_gradients(cs)
    S.log("gradients ready")
    todo = configs() if a.cmd == "run" else [("sobel", 1000, "q20")]
    for grad, dens, mask in todo:
        if a.cmd == "run" and path_for(grad, dens, mask).exists():
            continue
        run_one(cs, grad, dens, mask, a.n_jobs, a.threads)
    S.log("done")


if __name__ == "__main__":
    sys.exit(main())
