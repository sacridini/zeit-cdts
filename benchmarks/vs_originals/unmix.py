"""zeit.unmix against RStoolbox's mesma (RStoolbox 1.0.2.2, its NNLS): 4 endmembers, 6 bands, the fractions
non-negative and free to sum to anything (mesma's solver; its sum_to_one divides afterwards). mesma's solver stops at a
tolerance, Zeit's active set is exact: the fractions agree to 1e-3 (a median 7e-5) and Zeit's residual is never larger
than mesma's (smaller by up to 5e-7).

    python benchmarks/vs_originals/unmix.py
"""
import numpy as np

from zeit import _core
from common import best_of, record, run_r, threads_label, workdir

E = np.array([[0.02, 0.05, 0.03, 0.45, 0.22, 0.10],    # vegetation
              [0.10, 0.14, 0.20, 0.26, 0.38, 0.33],    # soil
              [0.08, 0.06, 0.04, 0.02, 0.01, 0.01],    # water
              [0.05, 0.07, 0.10, 0.20, 0.30, 0.18]])   # non-photosynthetic vegetation


def main():
    d = workdir("unmix")
    k, nb = E.shape
    np.savetxt(d / "em.csv", E, delimiter=",", fmt="%.17g")
    for side in (200, 1000):
        rng = np.random.default_rng(side)
        f = rng.dirichlet(np.ones(k), side * side)
        pixels = f @ E + rng.normal(0, 0.005, (side * side, nb))
        path = d / f"img_{side}.bin"
        pixels.astype(np.float64).tofile(path)
        t_r = run_r("unmix.R", path, side, nb, d / "em.csv")
        ref = np.fromfile(d / f"img_{side}_out.bin").reshape(-1, k + 1)
        for threads in (1, -1):
            out = {}
            t_z = best_of(lambda: out.update(r=_core.sma.unmix_batch(pixels, E, sum_to_one=False, nonneg=True,
                                                                      n_jobs=threads)))
            got = np.asarray(out["r"])
            err = float(np.abs(got[:, :k] - ref[:, :k]).max())
            better = float((got[:, k] - ref[:, k]).max())
            assert err < 1e-3 and better < 1e-9, (err, better)
            record("unmix", "RStoolbox 1.0.2.2 (R)", f"{side}x{side} px x {nb} bands x {k} em.", t_r, t_z,
                   threads_label(threads), f"max diff {err:.1e} (fractions); RMSE never larger than mesma's")


if __name__ == "__main__":
    main()
