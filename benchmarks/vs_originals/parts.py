"""zeit.ar_trend and zeit.parts against remotePARTS (1.0.4, R; Ives et al. 2021): the AR(1) trend of every pixel's
annual series (fitAR_map) and the partitioned GLS of the slopes against a mean (fitGLS_partition, exponential
correlation of a given range on the grid's distances, the nugget by maximum likelihood), on a grid of series with
spatially correlated AR(1) errors and a slope that changes across the map. The AR fits agree to the optimizers'
tolerance; the GLS is checked on remotePARTS' own slopes and partitions (Zeit's timing samples its own partitions
of the same size, from the slopes of zeit.ar_trend): every partition's estimate, SE and nugget and the overall
estimate agree to rounding. The overall SE does not quite: remotePARTS crosses the partitions after the first without
their nugget (fitGLS_partition drops their inverse Cholesky factor after the first pair and recomputes it from V
alone), Zeit crosses them all with it, so their t statistics are a little apart where the nugget is not 0.

    python benchmarks/vs_originals/parts.py [SIDE ...]
"""
import os
import sys

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

import zeit
from common import best_of, no_power_throttling, record, run_r, workdir

NT, RANGE = 30, 8.0
THREADS = int(os.environ.get("ZEIT_BENCH_THREADS", "8"))


def series(side, rng):
    """(pixels, years): a slope that changes across the map, AR(1) errors of spatially correlated innovations."""
    def field():
        f = gaussian_filter(rng.normal(size=(side, side)), 3.0, mode="wrap")
        return (f / f.std() + 0.5 * rng.normal(size=(side, side))).ravel()

    yy, xx = np.mgrid[0:side, 0:side] / side
    slope = (0.002 + 0.004 * np.sin(3 * xx) * np.cos(2 * yy)).ravel()
    e = np.empty((side * side, NT))
    e[:, 0] = field()
    for t in range(1, NT):
        e[:, t] = 0.4 * e[:, t - 1] + field()
    return 0.5 + slope[:, None] * np.arange(1, NT + 1)[None, :] + 0.02 * e


def main(sides):
    no_power_throttling()
    d = workdir("parts")
    for side in sides:
        n = side * side
        partsize = min(1500, n // 5)
        npart = n // partsize
        v = series(side, np.random.default_rng(side))
        path = d / f"series_{side}.bin"
        v.tofile(path)
        t_gls = run_r("parts.R", path, side, NT, RANGE, partsize, npart)
        t_ar = float((d / f"series_{side}_ar_seconds.txt").read_text())
        ar = np.fromfile(d / f"series_{side}_ar.bin").reshape(3, n)
        pm = np.fromfile(d / f"series_{side}_pm.bin", dtype=np.int32).reshape(npart, partsize)
        overall = np.fromfile(d / f"series_{side}_parts.bin")  # estimate, SE, t, p
        cube = xr.DataArray(v.T.reshape(NT, side, side), dims=("time", "y", "x"),
                            coords={"y": np.arange(side, dtype=float), "x": np.arange(side, dtype=float)})
        size = f"{side}x{side} px x {NT} years"

        for threads in (1, THREADS):
            res = {}
            t_z = best_of(lambda: res.update(r=zeit.ar_trend(cube, n_jobs=threads)))
            slope = res["r"].slope.values.ravel().astype(float)
            rho = res["r"].rho.values.ravel().astype(float)
            err_slope = float(np.max(np.abs(slope - ar[0]) / np.abs(ar[0]).max()))
            err_rho = float(np.max(np.abs(rho - ar[2])))
            assert err_slope < 1e-3 and err_rho < 1e-2, (err_slope, err_rho)
            record("ar_trend", "remotePARTS 1.0.4 fitAR_map (R)", size, t_ar, t_z, threads,
                   f"slope max diff {err_slope:.1e} of the largest, rho max diff {err_rho:.1e}")

        # the GLS on remotePARTS' slopes and partitions
        k = np.arange(n)
        got = zeit._core.parts.fit_parts(
            (k % side).astype(float), (k // side).astype(float), ar[0], np.ones((n, 1)), np.ones((n, 1)),
            pm.astype(np.int64), correlation="exponential", range=RANGE, nugget=None, ncross=6, n_jobs=THREADS)
        part = np.fromfile(d / f"series_{side}_part.bin").reshape(3, npart)  # estimates, SEs, nuggets
        mine = np.stack([got["part_coefficients"][:, 0], got["part_se"][:, 0], got["part_nugget"]])
        err_part = float(np.max(np.abs(mine - part) / np.abs(part)))
        err_est = abs(got["coefficients"][0] - overall[0]) / abs(overall[0])
        assert err_part < 1e-6 and err_est < 1e-6, (err_part, err_est)
        # remotePARTS crosses the partitions after the first without their nugget, Zeit with it: the overall SE
        # (and t) differ by as much as the nugget changes the correlation between partitions
        err_t = abs(got["t"][0] - overall[2]) / abs(overall[2])
        trend = zeit.ar_trend(cube, n_jobs=THREADS)
        for threads in (1, THREADS):
            t_z = best_of(lambda: zeit.parts(cube, trend=trend, range=RANGE, partsize=partsize, npart=npart,
                                             n_jobs=threads))
            record("parts", "remotePARTS 1.0.4 fitGLS_partition (R)", f"{size}, {npart} x {partsize}", t_gls, t_z,
                   threads, f"partitions' estimates, SEs and nuggets max rel. diff {err_part:.1e}, overall estimate "
                   f"{err_est:.1e}; overall t {100 * err_t:.2g}% apart (cross-partition nugget)")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [50, 100])
