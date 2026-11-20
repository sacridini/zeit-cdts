"""zeit.fuse_estarfm and zeit.fuse_starfm against the reference programs, as black boxes (outputs only).

* ESTARFM: Xiaolin Zhu's Python program (no licence: run locally from ``$ZHU_ESTARFM``, never copied or committed, its
  bundled test data never committed either), in the conda environment ``estarfm-ref`` (not zeit's), answered
  through ``fusion_estarfm_ref.py``.
* STARFM: Feng Gao's program (USDA-ARS, v1.2.2, public domain) could not be downloaded without a registration form,
  so the reference is ``zeit.starfm`` (FORCE's variant, adopted from the ARS program) with matched parameters.

Datasets: ``A`` a synthetic series (the test of the unit tests, 128 x 128, 6 fine dates, 46 coarse 8-day dates, the
coarse series already on the fine grid, as the references need); ``B`` Zhu's bundled test data (400 x 400, TM and
MODIS of 2001-05-24, 07-11, 08-12 at the BOREAS site; predicts 07-11 from the other two).

    python benchmarks/vs_originals/fusion.py A|B [--threads 4]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import zeit
from common import best_of, no_power_throttling, record, workdir

HERE = Path(__file__).resolve().parent
ZHU = Path(os.environ.get("ZHU_ESTARFM", r"C:\Users\eduar\Documents\dev\fusion-refs\estarfm_py\py_ESTARFM_1105"))
ENV = os.environ.get("ZHU_ENV", "estarfm-ref")
FINE_BANDS = ["green", "red", "nir08"]
COARSE_BANDS = ["sur_refl_b04", "sur_refl_b01", "sur_refl_b02"]
RES = 30.0
CRS = "EPSG:32633"


def cube(images, days, bands, res=RES):
    n_y, n_x = images[0].shape[-2:]
    c = xr.DataArray(np.stack(images).astype(np.float32), dims=("time", "band", "y", "x"),
                     coords={"time": pd.to_datetime(days), "band": bands,
                             "y": 4000000.0 - res * (np.arange(n_y) + 0.5), "x": 500000.0 + res * (np.arange(n_x) + 0.5)})
    return c.rio.write_crs(CRS)


def read_envi(path, bands=3, n=400):
    """BSQ file of the ENVI header next to it (data type 2 = int16, 4 = float32; little endian)."""
    kind = "<i2"
    for line in Path(str(path) + ".hdr").read_text().splitlines():
        if line.startswith("data type"):
            kind = {"2": "<i2", "4": "<f4", "12": "<u2"}[line.split("=")[1].strip()]
    return np.fromfile(path, dtype=kind).reshape(bands, n, n)


def write_envi(path, dn):
    """int16 BSQ with a minimal ENVI header (what the program reads)."""
    b, h, w = dn.shape
    Path(path).write_bytes(dn.astype("<i2").tobytes())
    Path(str(path) + ".hdr").write_text(
        f"ENVI\nsamples = {w}\nlines   = {h}\nbands   = {b}\nheader offset = 0\nfile type = ENVI Standard\n"
        "data type = 2\ninterleave = bsq\nbyte order = 0\n")


def run_zhu(work, fine1, coarse1, fine2, coarse2, coarsep, half_window, num_class):
    """Runs Zhu's program on ENVI files; returns (output DN array, seconds)."""
    work = Path(work)
    (work / "temp").mkdir(exist_ok=True)
    (work / "params.yaml").write_text(
        f"w: {half_window}\nnum_class: {num_class}\nDN_min: 0.0\nDN_max: 10000.0\nbackground: -9999\npatch_long: 500\n")
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = ["conda", "run", "-n", ENV, "python", str(HERE / "fusion_estarfm_ref.py"), str(ZHU), str(work / "params.yaml"),
           str(work / "temp"), *(str(work / n) for n in (fine1, coarse1, fine2, coarse2, coarsep))]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = work / f"{coarsep}_ESTARFM"
    if r.returncode or not out.exists():
        raise RuntimeError(f"Zhu's program failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return read_envi(out, bands=len(FINE_BANDS), n=read_size(work / fine1)), dt


def read_size(path):
    for line in Path(str(path) + ".hdr").read_text().splitlines():
        if line.startswith("lines"):
            return int(line.split("=")[1])
    raise ValueError(path)


def agreement(a, b, border, label):
    """Per band: max, mean |difference|, RMSE (reflectance x 10000) and the share within 1 DN, inside the border."""
    a = np.asarray(a, np.float64) * 10000.0
    b = np.asarray(b, np.float64) * 10000.0
    sl = (slice(None), slice(border, -border or None), slice(border, -border or None))
    d = np.abs(a[sl] - b[sl])
    ok = np.isfinite(d)
    print(f"  {label}")
    for i in range(d.shape[0]):
        x = d[i][ok[i]]
        if not x.size:
            continue
        print(f"    band {i}: max {x.max():8.2f} DN  mean {x.mean():7.3f}  rmse {np.sqrt((x ** 2).mean()):7.3f}  "
              f"within 1 DN {100.0 * (x <= 1.0).mean():6.2f}%  (n {x.size})")
    return float(100.0 * (d[ok] <= 1.0).mean())


def accuracy(pred, truth, label):
    """RMSE, bias, r and AAD per band (reflectance) and for NDVI, over the pixels both have."""
    print(f"  {label}")
    names = ["green", "red", "nir"]
    for i, name in enumerate(names):
        p, t = pred[i].ravel().astype(np.float64), truth[i].ravel().astype(np.float64)
        ok = np.isfinite(p) & np.isfinite(t)
        p, t = p[ok], t[ok]
        print(f"    {name:<6} rmse {np.sqrt(((p - t) ** 2).mean()):.4f}  bias {np.mean(p - t):+.4f}  "
              f"r {np.corrcoef(p, t)[0, 1]:.4f}  aad {np.mean(np.abs(p - t)):.4f}")
    nd = lambda x: (x[2] - x[1]) / (x[2] + x[1])
    p, t = nd(pred).ravel(), nd(truth).ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    print(f"    NDVI   rmse {np.sqrt(((p[ok] - t[ok]) ** 2).mean()):.4f}  bias {np.mean(p[ok] - t[ok]):+.4f}  "
          f"r {np.corrcoef(p[ok], t[ok])[0, 1]:.4f}")


CLASS = np.array([[0.10, 0.18, 0.25], [0.03, 0.02, 0.01], [0.06, 0.05, 0.35], [0.08, 0.07, 0.28]])


def synthetic(n=128, k=16, seed=3):
    """The series of the unit test: 4 classes, two with a seasonal curve, 6 fine dates (days 24, 88, ..., 336) and 46
    coarse 8-day dates; the coarse = block mean x 1.03 + 0.005 + noise 0.003, returned on the fine grid (nearest)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n / 96.0)
    c = np.zeros((n, n), dtype=int)
    c[(xx - 30) ** 2 + (yy - 30) ** 2 < 14 ** 2] = 1
    c[(xx >= 50) & (xx < 90) & (yy >= 10) & (yy < 60) & (c == 0)] = 2
    c[((xx - 30) ** 2 + (yy - 70) ** 2 < 18 ** 2) & (c == 0)] = 3
    tex = rng.normal(0, 0.004, (3, n, n))
    lg = lambda v: 1 / (1 + np.exp(-v))
    season = lambda t, on, off, kk: lg(kk * (t - on)) * (1 - lg(kk * (t - off)))

    def truth(t):
        out = CLASS[c].transpose(2, 0, 1).copy()
        s2, s3 = season(t, 110, 270, 0.07), season(t, 150, 300, 0.05)
        out[1][c == 2] = 0.20 - 0.15 * s2
        out[2][c == 2] = 0.15 + 0.30 * s2
        out[0][c == 2] = 0.05 + 0.04 * s2
        out[1][c == 3] = 0.22 - 0.16 * s3
        out[2][c == 3] = 0.18 + 0.22 * s3
        out[0][c == 3] = 0.06 + 0.03 * s3
        return (out + tex).astype(np.float32)

    fine_days = np.array([24, 88, 152, 208, 272, 336])
    coarse_days = np.arange(46) * 8
    fine = np.stack([truth(d) for d in fine_days])
    coarse = []
    for d in coarse_days:
        blk = truth(d).reshape(3, n // k, k, n // k, k).mean(axis=(2, 4)) * 1.03 + 0.005
        blk = blk + rng.normal(0, 0.003, blk.shape)
        coarse.append(np.repeat(np.repeat(blk, k, axis=1), k, axis=2))
    base = np.datetime64("2023-01-01")
    return (fine, np.stack(coarse), [str(base + int(d)) for d in fine_days], [str(base + int(d)) for d in coarse_days])


def dataset_a(threads):
    fine, coarse, fdays, cdays = synthetic()
    n = fine.shape[-1]
    fc, cc = cube(list(fine), fdays, FINE_BANDS), cube(list(coarse), cdays, COARSE_BANDS)
    truth = fine[2]
    d = workdir("fusion-a")
    print("Dataset A: synthetic series, 128 x 128, coarse on the fine grid; predict day 152 without it (pairs 88, 208)")
    day = str(np.datetime64("2023-01-01") + 152)
    kw = dict(include_fine_dates=False, dates=[day], exclude_dates=[day], radius=10 * RES, n_jobs=threads)
    ours = zeit.fuse_estarfm(fc, cc, **kw).values[0]
    f = lambda a: np.round(a * 10000).astype(np.int16)
    for i, nme in ((1, "tm1"), (3, "tm2")):
        write_envi(d / nme, f(fine[i]))
    for nme, day in (("m1", 88), ("m2", 208), ("mp", 152)):
        write_envi(d / nme, f(coarse[day // 8]))
    theirs, t_o = run_zhu(d, "tm1", "m1", "tm2", "m2", "mp", 10, 4.0)
    theirs = theirs / 10000.0
    agreement(ours, theirs, 10, "zeit.fuse_estarfm vs Zhu (w=10), border excluded")
    accuracy(ours, truth, "zeit.fuse_estarfm vs the truth")
    accuracy(theirs, truth, "Zhu vs the truth")
    print(f"  Zhu {t_o:.1f} s")


def dataset_b(threads):
    src = ZHU / "test data"
    d = workdir("fusion-b")
    for n in ("tm0524", "tm0812", "tm0711", "modis0524", "modis0812", "modis0711"):
        (d / n).write_bytes((src / n).read_bytes())
        (d / (n + ".hdr")).write_bytes((src / (n + ".hdr")).read_bytes())
    dn = {n: read_envi(d / n) for n in ("tm0524", "tm0812", "tm0711", "modis0524", "modis0812", "modis0711")}
    f = lambda a: np.where(a <= -9999, np.nan, a / 10000.0).astype(np.float32)
    fine = cube([f(dn["tm0524"]), f(dn["tm0812"])], ["2001-05-24", "2001-08-12"], FINE_BANDS)
    coarse = cube([f(dn[k]) for k in ("modis0524", "modis0711", "modis0812")], ["2001-05-24", "2001-07-11", "2001-08-12"],
                  COARSE_BANDS)
    truth = f(dn["tm0711"])
    print("Dataset B: Zhu's bundled data, 400 x 400, predict 2001-07-11 from 05-24 and 08-12")
    kw = dict(include_fine_dates=False, dates=["2001-07-11"], radius=25 * RES, n_jobs=threads)
    est = {}
    t_z = best_of(lambda: est.update(v=zeit.fuse_estarfm(fine, coarse, **kw).values[0]), repeat=2)
    ours = est["v"]
    t_z1 = best_of(lambda: zeit.fuse_estarfm(fine, coarse, **{**kw, "n_jobs": 1}), repeat=1)
    t_s1 = best_of(lambda: zeit.fuse_starfm(fine, coarse, **{**kw, "n_jobs": 1}), repeat=1)
    t_s4 = best_of(lambda: zeit.fuse_starfm(fine, coarse, **kw), repeat=1)
    print(f"  zeit.fuse_estarfm 1 thread {t_z1:.2f} s, {threads} threads {t_z:.2f} s; zeit.fuse_starfm 1 thread {t_s1:.2f} s, "
          f"{threads} threads {t_s4:.2f} s")
    zhu_path = d / "modis0711_ESTARFM"
    if os.environ.get("REUSE") and zhu_path.exists():
        theirs, t_o = read_envi(zhu_path), float("nan")
    else:
        theirs, t_o = run_zhu(d, "tm0524", "modis0524", "tm0812", "modis0812", "modis0711", 25, 4.0)
    theirs = np.where(theirs <= -9999, np.nan, theirs / 10000.0)
    print(f"  Zhu's program: {t_o:.1f} s (one process); zeit.fuse_estarfm: {t_z:.2f} s ({threads} threads)")
    agreement(ours, theirs, 25, "zeit (defaults) vs Zhu, border of 25 pixels excluded")
    accuracy(ours, truth, "zeit.fuse_estarfm vs the real TM of 07-11")
    accuracy(theirs, truth, "Zhu's ESTARFM vs the real TM of 07-11")
    for name, opts in (("v_scope=cell", dict(v_scope="cell")), ("alpha=0.01", dict(alpha=0.01)),
                       ("alpha=0.2", dict(alpha=0.2)), ("sigma=window", dict(sigma="window")),
                       ("resampling=nearest", dict(resampling="nearest"))):
        o = zeit.fuse_estarfm(fine, coarse, **{**kw, **opts}).values[0]
        agreement(o, theirs, 25, f"zeit {name} vs Zhu")
    s = zeit.fuse_starfm(fine, coarse, classes=4, **kw).values[0]
    accuracy(s, truth, "zeit.fuse_starfm (classes=4) vs the real TM of 07-11")
    # STARFM against zeit.starfm (FORCE's variant of the ARS program), the coarse series on the fine grid
    print("  STARFM: zeit.fuse_starfm vs zeit.starfm (FORCE's variant), radius 25, pairs 05-24 and 08-12")
    pairs = cube([f(dn["tm0524"]), f(dn["tm0812"])], ["2001-05-24", "2001-08-12"], FINE_BANDS)
    c3 = cube([f(dn[k]) for k in ("modis0524", "modis0711", "modis0812")], ["2001-05-24", "2001-07-11", "2001-08-12"],
              FINE_BANDS)
    for m in (4, 40):
        force = zeit.starfm(pairs, c3, radius=25, nclass=m, unc_fine=0.002, unc_coarse=0.002,
                            target=["2001-07-11"], n_jobs=threads).values[0]
        accuracy(force, truth, f"zeit.starfm (FORCE variant) nclass={m} vs the real TM")
        for name, opts in (("defaults", {}), ("filter=either", dict(filter="either")),
                           ("sigma=window", dict(sigma="window")), ("weighting=log", dict(weighting="log")),
                           ("pure=tiers", dict(pure="tiers")), ("coarse_uncertainty=0.002", dict(coarse_uncertainty=0.002))):
            o = zeit.fuse_starfm(fine, coarse, classes=m, **{**kw, **opts}).values[0]
            agreement(o, force, 25, f"fuse_starfm classes={m} {name} vs the FORCE variant")
    np.save(d / "ours.npy", ours)
    if t_o == t_o:          # not when the output of an earlier run was reused
        record("estarfm", "Zhu's ESTARFM (Python)", "400x400 px x 3 bands, w=25", t_o, t_z, threads,
               "Zhu's bundled test data, not redistributed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["A", "B"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    no_power_throttling()
    {"A": dataset_a, "B": dataset_b}[a.dataset](a.threads)


if __name__ == "__main__":
    main()
