"""zeit.recovery_metrics against RecoverMetricsR (0.1.0, R; GPL-3, run as a black box): RRI, R80P, YrYr and the
years to recovery of an annual series (21 years), every pixel disturbed in 2010 (Zeit's start 2009, its trough 2010)
and regrowing steadily. RecoverMetricsR takes the best of the 5 years after the trough where Zeit (Frazier et al.
2018) takes years 4 and 5, and counts the years to recovery from the year after it: on steady regrowths they are the
same and agree to float32's precision (Zeit's outputs).

    python benchmarks/vs_originals/recovery.py
"""
import numpy as np
import pandas as pd
import xarray as xr

import zeit
from common import best_of, record, run_r, threads_label, workdir

FIRST, NT, EVENT = 2000, 21, 2010


def series(n, rng):
    years = np.arange(FIRST, FIRST + NT)
    pre = rng.uniform(0.4, 0.7, n)[:, None]
    low = rng.uniform(0.0, 0.2, n)[:, None]
    rate = rng.uniform(0.02, 0.15, n)[:, None]
    after = years[None, :] - EVENT
    v = np.where(after < 0, pre, np.minimum(low + rate * after, pre))
    return v + np.where(after < 0, rng.normal(0, 0.01, (n, NT)), 0)


def main():
    d = workdir("recovery")
    for side in (200, 1000):
        rng = np.random.default_rng(side)
        v = series(side * side, rng)
        path = d / f"series_{side}.bin"
        v.tofile(path)
        t_r = run_r("recovery.R", path, side, FIRST, NT, EVENT)
        ref = np.fromfile(d / f"series_{side}_out.bin").reshape(-1, 4)
        cube = xr.DataArray(v.T.reshape(NT, side, side), dims=("time", "y", "x"),
                            coords={"time": pd.to_datetime([f"{y}-07-01" for y in range(FIRST, FIRST + NT)])})
        for threads in (1, -1):
            out = {}
            t_z = best_of(lambda: out.update(r=zeit.recovery_metrics(cube, EVENT - 1, n_jobs=threads)))
            r = out["r"]
            got = np.stack([np.asarray(r[k]).ravel() for k in ("rri", "r80p", "yryr", "y2r")], axis=1)
            with np.errstate(invalid="ignore"):
                err = float(np.nanmax(np.abs(got - ref) / np.maximum(np.abs(ref), 1)))
            assert np.array_equal(np.isnan(got), np.isnan(ref)) and err < 1e-6, err
            record("recovery_metrics", "RecoverMetricsR 0.1.0 (R)", f"{side}x{side} px x {NT} years", t_r, t_z,
                   threads_label(threads), f"max rel. diff {err:.1e} (rri, r80p, yryr, y2r)")


if __name__ == "__main__":
    main()
