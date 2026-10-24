"""zeit.train_di + zeit.aoa against CAST's trainDI + aoa (1.0.4): the same training points (clusters, 8 predictors,
or 64 as the series of a deep learning model) and new points, with the local point density and without. Their DI
agree to 1e-13.

    python benchmarks/vs_originals/aoa.py
"""
import numpy as np

import zeit
from common import best_of, record, run_r, threads_label, workdir

SIZES = [(500, 10_000, False, 8), (500, 100_000, False, 8), (2000, 100_000, False, 8), (500, 10_000, True, 8),
         (2000, 100_000, False, 64)]


def main():
    d = workdir("aoa")
    for n, m, lpd, p in SIZES:
        rng = np.random.default_rng(n + m)
        centres = rng.normal(0, 4, (6, p))
        train = centres[rng.integers(0, 6, n)] + rng.normal(size=(n, p))
        new = rng.normal(0, 5, (m, p))
        tp, npth = d / f"train_{n}_{p}.csv", d / f"new_{n}_{m}_{p}.csv"
        np.savetxt(tp, train, delimiter=",", fmt="%.17g")
        np.savetxt(npth, new, delimiter=",", fmt="%.17g")
        t_r = run_r("aoa.R", tp, npth, "TRUE" if lpd else "FALSE")
        di_r = np.loadtxt(d / f"new_{n}_{m}_{p}_di.txt")
        for threads in (1, -1):
            out = {}

            def ours():
                td = zeit.train_di(train, n_jobs=threads)
                out["r"] = zeit.aoa(new, td, lpd=lpd, n_jobs=threads)

            t_z = best_of(ours)
            err = float(np.abs(out["r"]["DI"].to_numpy() - di_r).max())
            assert err < 1e-9, err
            record("aoa", "CAST 1.0.4 (R)", f"{n} train x {m} new{' + LPD' if lpd else ''}" + ("" if p == 8 else f", {p} pred."), t_r, t_z,
                   threads_label(threads), f"DI max diff {err:.1e}")


if __name__ == "__main__":
    main()
