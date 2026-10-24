"""zeit's EWMACD (what zeit.ewmacd runs: _core.ewmacd.ewmacd_batch) against Brooks' EWMACD v1.3.0 R script per
pixel (its function EWMACD.pixel.for.rasterEngine, as its rasterEngine calls it): the same irregular dates
2005-2012, seasonal pixels with noise, clouds and changes, harmonics 2, lambda 0.3, persistence 1. The flags are
the same. The script has no licence stated: give its path, it is run, never copied.

    python benchmarks/vs_originals/ewmacd.py "path/to/EWMACD v1.3.0.r"
"""
import sys

import numpy as np

from zeit import _core
from common import best_of, record, run_r, threads_label, workdir


def series(n, rng):
    years, doys = [], []
    for y in range(2005, 2013):
        d = int(rng.integers(1, 21))
        while d <= 365:
            if rng.random() < 0.6:
                years.append(y)
                doys.append(d)
            d += int(rng.integers(10, 41))
    years, doys = np.array(years, dtype=np.int32), np.array(doys, dtype=np.float64)
    t = years + doys / 365.0
    th = 2 * np.pi * doys / 365
    v = np.empty((n, len(t)))
    for k in range(n):
        base, a1, b1 = rng.uniform(300, 700), rng.uniform(-150, 150), rng.uniform(-150, 150)
        noise = rng.uniform(5, 40)
        p = base + a1 * np.sin(th) + b1 * np.cos(th) + rng.normal(0, noise, len(t))
        if k % 3 == 1:
            t0 = 2009 + rng.uniform(0, 2.5)
            p -= np.where(t >= t0, 300 * np.maximum(0, 1 - (t - t0) / 3), 0)
        cloud = rng.random(len(t)) < 0.06
        p[cloud] -= rng.uniform(200, 500, cloud.sum())
        v[k] = np.round(p)
    return years, doys, t, v


def main(script):
    d = workdir("ewmacd")
    for n in (1000, 10000):
        years, doys, t, v = series(n, np.random.default_rng(n))
        vp, dp = d / f"values_{n}.csv", d / f"dates_{n}.csv"
        np.savetxt(vp, v, delimiter=",", fmt="%.17g")
        np.savetxt(dp, np.column_stack([years, doys]), delimiter=",", fmt="%d")
        t_r = run_r("ewmacd.R", script, vp, dp)
        ref = np.loadtxt(d / f"values_{n}_flags.csv", delimiter=",")
        for threads in (1, -1):
            out = {}
            t_z = best_of(lambda: out.update(r=_core.ewmacd.ewmacd_batch(v, years, doys, t, 2005, 2009, harmonics=2,
                                                                         lmbda=0.3, persistence=1, n_jobs=threads)))
            flags = out["r"][0]
            same = float(np.mean(np.isclose(flags, ref, equal_nan=True)))
            assert same == 1.0, same
            record("ewmacd", "Brooks' EWMACD v1.3.0 (R)", f"{n} pixels x {len(t)} dates", t_r, t_z,
                   threads_label(threads), "the same flags")


if __name__ == "__main__":
    main(sys.argv[1])
