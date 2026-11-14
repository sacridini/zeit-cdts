"""24f, task B: the Monte Carlo with noise added to the gradient in every realisation.

    python b24f.py check  --cfg proposed       # s = 0 reproduces the counts of the normal path, case by case
    python b24f.py run    --cfg proposed --s 0,0.05,0.1,0.2,0.4

Scheme (an approximation, not a transcription: Bernander et al. 2013, "Improving the stochastic watershed", Pattern Recognition
Letters 34, was not consulted): in each of the R = 200 realisations, i.i.d. Gaussian noise of standard deviation
``s * std(gradient over the mask)`` is added to the gradient (the std is taken once, over the mask's pixels), and the marker draw
is the normal path's (the seed's uniform sets thinned to ``min_distance``, here made in Python with ``trim_close_points`` and
given through ``marker_sets=``, so that s = 0 equals the normal counts). Nothing else is perturbed. One call of
``stochastic_watershed_counts`` per realisation (``realisations=1``, ``n_jobs=1``), the counts summed; the noise of realisation
i of a case is drawn from ``default_rng([24600, case index, i])``. Chunks of 25 realisations run in a thread pool.

Configurations: ``notebook`` (tophat, abs2000, 1000 markers, low 0.3, min_size 50) and ``proposed`` (``--spec`` or the file
``f24/proposed.txt`` written by the report of task A: ``grad,dens,mask,low,ms``).
Output: ``sensitivity/f24/B/<cfg>_s<s>.csv``: per case, the cheap responses over low {0.1 .. 0.7} at the configuration's
min_size, and ``pkg_*`` / ``cc_*`` at its own low (``extras.py``).
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

import s24f as S
import cases as C
import pipeline as P
from extras import pkg_metrics
from zeit import _fields_core as _fc

DIR = S.OUT24 / "B"
NOISE_SEED = 24600
CHUNK = 25


def cfg_of(name: str, spec: str = None):
    """(grad, dens, mask, low, ms) of ``notebook`` / ``proposed`` / a spec string; dens 'n1000' = 1000 per window."""
    if name == "notebook":
        return ("tophat", "n1000", "abs2000", 0.3, 50)
    spec = spec or (S.OUT24 / "proposed.txt").read_text().strip()
    g, d, m, lo, ms = spec.split(",")
    return (g, float(d), m, float(lo), int(ms))


def s1_of(case, cfg):
    g, d, m, _, _ = cfg
    if d == "n1000":
        from dataclasses import replace

        return replace(S.BASE, gradient=g, markers=1000, mask_threshold=S.mask_thr(case, m), seed=0)
    return S.s1_for(case, g, d, m)


def marker_sets(case, s1: P.S1, lo: int, hi: int):
    """The thinned sets of realisations ``lo .. hi`` of the normal path (seed, uniform draws up front, in order)."""
    rng = np.random.default_rng(s1.seed)
    scale = np.array(case.shape, dtype=np.float64)[:, None]
    sets = [rng.random((2, int(s1.markers))) * scale for _ in range(s1.realizations)]
    return [_fc.trim_close_points(s, s1.min_distance) if s1.min_distance > 0 else s for s in sets[lo:hi]]


def noisy_counts(case, ci, grad, keep, s1, s, lo, hi):
    sd = float(np.std(grad[keep])) if keep is not None else float(np.nanstd(grad))
    sets = marker_sets(case, s1, lo, hi)
    total = np.zeros(case.shape, np.uint32)
    for k, i in enumerate(range(lo, hi)):
        g = grad
        if s > 0:
            g = grad + s * sd * np.random.default_rng([NOISE_SEED, ci, i]).standard_normal(grad.shape)
        total += _fc.stochastic_watershed_counts(g, keep, [sets[k]], coloring=s1.coloring, n_neighbors=20,
                                                 disk_radius=s1.marker_radius, n_jobs=1)
    return total


def counts_for(cs, cfg, s, threads):
    tasks = [(ci, lo) for ci in range(len(cs)) for lo in range(0, S.R, CHUNK)]
    s1s = [s1_of(c, cfg) for c in cs]
    keeps = [c.keep(x.mask_threshold) for c, x in zip(cs, s1s)]

    def work(t):
        ci, lo = t
        return noisy_counts(cs[ci], ci, P.gradient_of(cs[ci], cfg[0]), keeps[ci], s1s[ci], s, lo, min(S.R, lo + CHUNK))

    outs = S.pmap(work, tasks, threads)
    counts = [np.zeros(c.shape, np.uint32) for c in cs]
    for (ci, _), o in zip(tasks, outs):
        counts[ci] += o
    return counts, s1s, keeps


def check(cs, cfg, threads):
    counts, s1s, _ = counts_for(cs, cfg, 0.0, threads)
    bad = 0
    for c, n, s1 in zip(cs, counts, s1s):
        ref = P.stage1(c, s1, n_jobs=8).counts
        if not np.array_equal(ref, n):
            bad += 1
            S.log(f"  DIFFERS {c.id}: {int((ref != n).sum())} pixels")
    S.log(f"s = 0 check, {cfg}: {len(cs) - bad} of {len(cs)} cases equal the normal path's counts")
    return bad


def run(cs, name, cfg, svals, threads):
    DIR.mkdir(parents=True, exist_ok=True)
    g, d, m, low, ms = cfg
    for s in svals:
        p = DIR / f"{name}_s{s:g}.csv"
        if p.exists():
            continue
        t0 = time.perf_counter()
        counts, s1s, keeps = counts_for(cs, cfg, s, threads)
        t1 = time.perf_counter()

        class B:  # the Boundary the stage-2 helpers need
            pass

        def work(i):
            b = B()
            b.counts, b.r = counts[i], S.R
            c = cs[i]
            cf = {"grad": g, "dens": d if d == "n1000" else float(d), "mask": m, "s": s}
            rows = S.case_rows_mc(c, keeps[i], b, cf, S.LOWS, (ms,))
            pred = S.hyst_pred(c, keeps[i], S.mc_regions(b.counts, b.r, low, S.high_for(low)), ms)
            ex = pkg_metrics(c, pred)
            for r in rows:
                if r["low"] == low:
                    r.update(ex)
            return rows

        rows = [r for rr in S.pmap(work, range(len(cs)), threads) for r in rr]
        pd.DataFrame(rows).to_csv(p, index=False)
        S.log(f"{name} s={s:g}: counts {t1 - t0:.0f}s, stage 2 {time.perf_counter() - t1:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--cfg", default="proposed")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--s", default="0,0.05,0.1,0.2,0.4")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    cs = C.load_cases()
    S.preload_gradients(cs, tuple(sorted({"tophat", cfg_of(a.cfg, a.spec)[0]})))
    cfg = cfg_of(a.cfg, a.spec)
    if a.cmd == "check":
        sys.exit(1 if check(cs, cfg, a.threads) else 0)
    run(cs, a.cfg, cfg, [float(x) for x in a.s.split(",")], a.threads)
    S.log("done")


if __name__ == "__main__":
    main()
