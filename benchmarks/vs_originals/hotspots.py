"""zeit.emerging_hotspots against sfdep's emerging_hotspot_analysis (0.2.5): the same cube of values on a grid
(queen neighbours, a time lag of 1), sfdep with its default 199 permutations and with 9 (beyond
20 x 20 cells, 9 only: 50 x 50 took sfdep 13 minutes with 199). Their Gi* agree to 1e-13.

    python benchmarks/vs_originals/hotspots.py [SIDE ...]
"""
import sys

import numpy as np

import zeit
from common import best_of, record, run_r, threads_label, workdir


def main(sides):
    d = workdir("hotspots")
    nt = 12
    for side in sides:
        rng = np.random.default_rng(side)
        v = rng.normal(size=(nt, side, side))
        v[:, side // 4: side // 2, side // 4: side // 2] += np.linspace(0, 3, nt)[:, None, None]
        path = d / f"values_{side}.txt"
        np.savetxt(path, v.ravel(), fmt="%.17g")
        # sfdep's p-values are by permutation (its default 199, the most of its time); Zeit's are the normal's
        for nsim in (199, 9) if side <= 20 else (9,):
            run_side(v, side, nt, nsim, run_r("hotspots.R", path, side, side, nt, nsim),
                     np.loadtxt(d / f"values_{side}_gi.txt").reshape(nt, side, side))


def run_side(v, side, nt, nsim, t_r, gi_r):
    for threads in (1, -1):
        res = {}
        t_z = best_of(lambda: res.update(r=zeit.emerging_hotspots(v, radius=1, lag=1, trend="original",
                                                                  n_jobs=threads)))
        gi = zeit._core.hotspots.gi_star(v, radius=1, lag=1)
        err = float(np.abs(gi - gi_r).max())
        assert err < 1e-10, err
        record("emerging_hotspots", f"sfdep 0.2.5 (R), {nsim} perm.", f"{side}x{side} cells x {nt} dates", t_r,
               t_z, threads_label(threads), f"Gi* max diff {err:.1e}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [20, 50, 100])
