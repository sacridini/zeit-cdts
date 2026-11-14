"""24f, task C: the hierarchy cut instead of the hysteresis. Resumable (one CSV per method x stage-1 configuration).

    python c24f.py run --grads sobel,multitemporal [--threads 6]

From a map of edge values, cut every 4-neighbour edge whose value is above ``t`` and label the components of what is left
(a partition with no boundary network); then ``min_size`` and the final watershed on the image as usual.

* ``cut-exact``: the per-edge probabilities of ``zeit._fields_core.exact_boundary_probability(..., return_edges=True)`` (the
  uniform one-pixel marker model with ``markers`` = the density's count; a different model from the Monte Carlo);
* ``cut-mc``: the Monte Carlo map of task A (same maps, from the stage-1 cache): an edge's value is the maximum of its two
  pixels' frequencies (counts / R).

``t`` on a log grid (exact: 0.001 .. 0.3; Monte Carlo: 0.005 .. 0.6, the frequency quantum being 1 / 200), ``min_size`` in
{2, 5, 10, 25} (on the Monte Carlo map every boundary pixel is cut off on all sides, a singleton the minimum size removes;
the final watershed gives its pixels back). Output: ``sensitivity/f24/C/<method>_<grad>_d<dens>_<mask>.csv``; the column
``grad`` is ``<gradient>|cut-exact`` or ``<gradient>|cut-mc`` and ``low`` is ``t``.
"""
import argparse
import sys
import time

import pandas as pd

import s24f as S
import cases as C
import pipeline as P
from fast import fast_eval

DIR = S.OUT24 / "C"
T_EXACT = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
T_MC = (0.005, 0.01, 0.03, 0.1, 0.3, 0.6)
SIZES = (2, 5, 10, 25)


def path_for(method, grad, dens, mask):
    return DIR / f"{method}_{grad}_d{dens}_{mask}.csv"


def case_rows(case, keep, h, v, cfg, ts, sizes=SIZES):
    rows = []
    for t in ts:
        comps = S.cut_components(h, v, keep, t)
        for ms in sizes:
            rows.append(S.row(case, fast_eval(case, S.cut_pred(case, keep, comps, ms)), **cfg, low=t, ms=ms))
    return rows


def run_one(cs, method, grad, dens, mask, n_jobs, threads):
    t0 = time.perf_counter()
    s1s = [S.s1_for(c, grad, dens, mask) for c in cs]
    keeps = [c.keep(s.mask_threshold) for c, s in zip(cs, s1s)]
    edges = []
    for c, s1 in zip(cs, s1s):
        if method == "cut-exact":
            edges.append(S.exact_edges(c, s1, n_jobs))
        else:
            b = P.stage1(c, s1, n_jobs=n_jobs)
            edges.append(S.edge_values_mc(b.counts, b.r))
    t1 = time.perf_counter()
    ts = T_EXACT if method == "cut-exact" else T_MC
    cfg = {"grad": f"{grad}|{method}", "dens": dens, "mask": mask}
    rows = [r for rr in S.pmap(lambda i: case_rows(cs[i], keeps[i], *edges[i], cfg, ts), range(len(cs)), threads)
            for r in rr]
    DIR.mkdir(parents=True, exist_ok=True)
    p = path_for(method, grad, dens, mask)
    pd.DataFrame(rows).to_csv(p.with_name(p.stem + ".tmp"), index=False)
    p.with_name(p.stem + ".tmp").replace(p)
    S.log(f"{method} {grad} d{dens} {mask}: maps {t1 - t0:.0f}s, stage 2 {time.perf_counter() - t1:.0f}s, {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--grads", default="sobel,multitemporal")
    ap.add_argument("--methods", default="cut-mc,cut-exact")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=S.THREADS)
    a = ap.parse_args()
    cs = C.load_cases()
    S.preload_gradients(cs, tuple(a.grads.split(",")))
    for method in a.methods.split(","):
        for grad in a.grads.split(","):
            for dens in S.DENS:
                for mask in S.MASKS:
                    if not path_for(method, grad, dens, mask).exists():
                        run_one(cs, method, grad, dens, mask, a.n_jobs, a.threads)
    S.log("done")


if __name__ == "__main__":
    sys.exit(main())
