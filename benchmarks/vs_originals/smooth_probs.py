"""zeit.smooth_probs against sits' Bayesian smoothing (sits 1.5.4, the function sits_smooth applies to each block):
the same probabilities of 4 classes, a 7 x 7 window. Their results agree to 1e-15 (sits' own call: the block's
smoothing, without its reading and writing of files).

    python benchmarks/vs_originals/smooth_probs.py
"""
import numpy as np

import zeit
from common import best_of, record, run_r, threads_label, workdir


def main():
    d = workdir("smooth")
    k, window = 4, 7
    for side in (200, 1000):
        rng = np.random.default_rng(side)
        scores = rng.normal(size=(k, side, side))
        for c in range(k):
            yy, xx = np.mgrid[:side, :side]
            cy, cx = rng.uniform(0, side, 2)
            scores[c] += 3 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (side / 4) ** 2))
        probs = np.exp(scores) / np.exp(scores).sum(axis=0)
        probs = np.round(probs * 10000) / 10000
        path = d / f"probs_{side}.csv"
        np.savetxt(path, probs.reshape(k, -1).T, delimiter=",", fmt="%.17g")
        t_r = run_r("smooth_probs.R", path, side, side, window, 20)
        ref = np.loadtxt(d / f"probs_{side}_out.csv", delimiter=",").T.reshape(k, side, side)
        for threads in (1, -1):
            out = {}
            t_z = best_of(lambda: out.update(r=zeit.smooth_probs(probs, window_size=window, smoothness=20,
                                                                 n_jobs=threads)))
            err = float(np.abs(np.asarray(out["r"]) - ref).max())
            assert err < 1e-12, err
            record("smooth_probs", "sits 1.5.4 (R)", f"{side}x{side} px x {k} classes", t_r, t_z,
                   threads_label(threads), f"max diff {err:.1e}")


if __name__ == "__main__":
    main()
