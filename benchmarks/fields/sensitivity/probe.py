"""Timing probe: one stage 1 (R realizations) and one stage 2 + evaluation per case; prints the extrapolation."""
import pyexpat  # noqa: F401  (first, so rasterio's expat is the conda one)
import os
import sys
import time

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("GDAL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import cases as C
import pipeline as P


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    t0 = time.perf_counter()
    cs = C.load_cases()
    print(f"loaded {len(cs)} cases in {time.perf_counter() - t0:.1f}s")
    tot1 = tot2 = 0.0
    px = 0
    per = {}
    for c in cs:
        s1 = P.S1(realizations=R, seed=99)
        t = time.perf_counter()
        b = P.stage1(c, s1, use_cache=False)
        t1 = time.perf_counter() - t
        t = time.perf_counter()
        lab, bl = P.stage2(c, s1, b, P.S2())
        r = P.evaluate(c, lab, bl)
        t2 = time.perf_counter() - t
        n = c.valid.size
        tot1 += t1
        tot2 += t2
        px += n
        g = per.setdefault((c.group, c.kind), [0, 0.0, 0.0])
        g[0] += n; g[1] += t1; g[2] += t2
        print(f"{c.id:36s} {n/1e6:6.2f} Mpx stage1 {t1:6.2f}s  stage2+eval {t2:6.2f}s  F1={r.get('f_t1', np.nan):.3f}")
    print(f"R={R}: total stage1 {tot1:.1f}s ({tot1 / px * 1e6 / R * 1000:.1f} ms per realization per Mpx), "
          f"stage2+eval {tot2:.1f}s over {px/1e6:.2f} Mpx")
    for k, v in per.items():
        print(k, f"{v[0]/1e6:.2f} Mpx, s1 {v[1]:.1f}s, s2 {v[2]:.1f}s")


if __name__ == "__main__":
    main()
