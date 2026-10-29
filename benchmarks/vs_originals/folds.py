"""zeit's spatial cross-validation against its originals (folds.R): the range of a map's autocorrelation against
blockCV's cv_spatial_autocor (4.0.1), kNNDM against CAST's knndm (1.0.4), the error profile against CAST's
errorProfiles (with scam 1.2.23), and the Mahalanobis distance of trainDI and aoa. Zeit on 1 thread and 8.

    python benchmarks/vs_originals/folds.py [autocor] [knndm] [profile] [mahalanobis]

(all four by default; CAST's Mahalanobis aoa of 100,000 points takes about 15 minutes).
"""
import sys

import numpy as np
import pandas as pd

import zeit
from common import best_of, record, run_r, workdir

THREADS = (1, 8)


def smooth_map(n, window, seed, pixel=100.0):
    """White noise averaged over window x window pixels (2 bands): correlated up to window pixels apart."""
    import xarray as xr

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(2, n + window - 1, n + window - 1))
    c = np.cumsum(np.cumsum(np.pad(noise, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
    w = window
    smooth = (c[:, w:, w:] - c[:, :-w, w:] - c[:, w:, :-w] + c[:, :-w, :-w]) / w ** 2
    da = xr.DataArray(smooth, dims=("band", "y", "x"),
                      coords={"band": ["b1", "b2"], "y": 500_000.0 - pixel * (np.arange(n) + 0.5),
                              "x": 300_000.0 + pixel * (np.arange(n) + 0.5)})
    return da.rio.write_crs("EPSG:32722")


def autocor(d):
    for window in (3, 7, 15):
        da = smooth_map(300, window, seed=window)
        path = d / f"map_{window}.tif"
        da.rio.to_raster(path)
        t_r = run_r("folds.R", "autocor", path)
        theirs = np.loadtxt(d / f"map_{window}_range.txt")
        for threads in THREADS:
            out = {}

            def ours():
                out["r"] = zeit.autocorrelation_range(da, n_jobs=threads)

            t_z = best_of(ours)
            ratio = out["r"]["range"].to_numpy() / theirs
            record("autocorrelation_range", "blockCV 4.0.1 (R)", f"300x300 px, range {window * 100} m", t_r, t_z,
                   threads, f"range / blockCV's {ratio.min():.2f}-{ratio.max():.2f}")


def knndm(d):
    from zeit._core import knndm as core
    import geopandas as gpd

    for n in (1000, 5000):
        prefix = d / f"knndm_{n}"
        t_r = run_r("folds.R", "knndm", n, prefix)
        train = pd.read_csv(f"{prefix}_train.csv").to_numpy()
        pred = pd.read_csv(f"{prefix}_pred.csv").to_numpy()
        cast_folds = np.loadtxt(f"{prefix}_folds.txt").astype(np.int64)
        w_cast = float(np.loadtxt(f"{prefix}_W.txt"))
        # CAST's own folds measured by zeit: its W
        w_check = core.wasserstein(core.nearest_other(train, cast_folds), core.nearest(train, pred))
        assert abs(w_check / w_cast - 1) < 1e-9, (w_check, w_cast)
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(train[:, 0], train[:, 1]))
        pp = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pred[:, 0], pred[:, 1]))
        for threads in THREADS:
            out = {}

            def ours():
                out["f"] = zeit.spatial_folds(pts, method="knndm", k=5, prediction_points=pp, n_jobs=threads)

            t_z = best_of(ours, repeat=2)
            record("knndm", "CAST 1.0.4 (R)", f"{n} clustered points, k=5", t_r, t_z, threads,
                   f"W {out['f'].W:.3f} (CAST {w_cast:.3f}; CAST's W by zeit equal)")


def profile(d):
    rng = np.random.default_rng(3)
    n = 2000
    x = pd.DataFrame({"a": np.concatenate([rng.normal(size=1400), rng.normal(3, 1, 600)]),
                      "b": rng.normal(size=n), "c": rng.uniform(size=n)})
    y = 2 * x.a + x.b ** 2 + rng.normal(size=n) * (0.5 + np.abs(x.a))
    folds = rng.permutation(np.arange(n) % 5) + 1
    x.to_csv(d / "profile.csv", index=False)
    np.savetxt(d / "profile_y.txt", y)
    np.savetxt(d / "profile_folds.txt", folds, fmt="%d")
    t_r = run_r("folds.R", "profile", d / "profile.csv", d / "profile_y.txt", d / "profile_folds.txt")
    pred = np.loadtxt(d / "profile_pred.txt")
    weight = np.loadtxt(d / "profile_weight.txt")
    windows = pd.read_csv(d / "profile_windows.csv")
    curve = pd.read_csv(d / "profile_curve.csv")
    for threads in THREADS:
        out = {}

        def ours():
            out["p"] = zeit.error_profile(train=x, observed=y.to_numpy(), predicted=pred, folds=folds,
                                          weights=weight, n_jobs=threads)

        t_z = best_of(ours)
        p = out["p"]
        diff = float(np.abs(p.points["metric"].to_numpy() - windows["metric"].to_numpy()).max())
        assert diff < 1e-9, diff
        gap = float(np.abs(p(curve["DI"].to_numpy()) - curve["fit"].to_numpy()).max() / np.ptp(windows["metric"]))
        record("error_profile", "CAST 1.0.4 + scam (R)", f"{n} points, 5 folds", t_r, t_z, threads,
               f"windows max diff {diff:.1e}; curve within {gap:.1%} of the metric's range")


def mahalanobis(d):
    for n, m in ((500, 10_000), (2000, 100_000)):
        rng = np.random.default_rng(n + m)
        centres = rng.normal(0, 4, (6, 8))
        train = centres[rng.integers(0, 6, n)] + rng.normal(size=(n, 8))
        train[:, 1] = 0.7 * train[:, 0] + 0.3 * train[:, 1]
        new = rng.normal(0, 5, (m, 8))
        tp, npth = d / f"md_train_{n}.csv", d / f"md_new_{n}_{m}.csv"
        np.savetxt(tp, train, delimiter=",", fmt="%.17g")
        np.savetxt(npth, new, delimiter=",", fmt="%.17g")
        t_r = run_r("folds.R", "md", tp, npth)
        di_r = np.loadtxt(d / f"md_new_{n}_{m}_di.txt")
        for threads in THREADS:
            out = {}

            def ours():
                td = zeit.train_di(train, distance="mahalanobis", n_jobs=threads)
                out["r"] = zeit.aoa(new, td, n_jobs=threads)

            t_z = best_of(ours)
            err = float(np.abs(out["r"]["DI"].to_numpy() - di_r).max())
            assert err < 1e-9, err
            record("aoa (Mahalanobis)", "CAST 1.0.4 (R)", f"{n} train x {m} new, 8 pred.", t_r, t_z, threads,
                   f"DI max diff {err:.1e}")


def main():
    d = workdir("folds")
    parts = {"autocor": autocor, "knndm": knndm, "profile": profile, "mahalanobis": mahalanobis}
    for name in sys.argv[1:] or parts:
        parts[name](d)


if __name__ == "__main__":
    main()
