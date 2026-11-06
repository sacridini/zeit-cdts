"""``zeit.improphe`` against FORCE 3.11.00's CF-ImproPhe (CFIMP) and L2-ImproPhe (L2IMP) on the same synthetic data.

The FORCE side (``force_improphe.sh``, WSL) wrote the cube and its outputs to ``workdir("force_improphe")``; the cube is
the Sentinel-2A BOA at 10 m (the fine data), Landsat 8 BOA at 30 m (L2's coarse scenes) and a field of the year's peak NDVI
at 30 m (CF's coarse), all made from one truth map by ``force_improphe_cube.py``. FORCE ran every configuration twice, with
OpenBLAS (A: how the reference build is linked) and with GSL's reference CBLAS (B: ``LD_PRELOAD=libgslcblas``, which is what
Zeit's port of ``gsl_eigen_symmv`` follows)::

    python benchmarks/vs_originals/force_improphe.py --part cf,l2 [--jobs 6]

Zeit's input is built from the files FORCE read: the fine BOA with the observations FORCE's QAI screening drops set to NaN
(``SCREEN_QAI`` default: off, any cloud state, shadow), the coarse scenes the same way, the field with its NoData as NaN,
each on its own grid: Zeit resamples by nearest neighbour as FORCE's GDAL read does (checked here to equal the replication).
``zeit.improphe(..., force=True)`` truncated toward zero is compared with FORCE's int16 ``_IMP`` files over the pixels Zeit
predicts (the others are NaN; what FORCE writes there is listed), per layer, in the share of equal values, of values within
1, and the largest difference. Zeit's default (the paper's principal components, float means) is compared with FORCE only
as statistics, and everything with the truth: the RMSE of FORCE, of Zeit as FORCE, of Zeit's default, of FORCE's
components with float means (``pca="force"``) and of nearest-neighbour upsampling.
"""
import argparse
import datetime
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from common import no_power_throttling, workdir

DATA = workdir("force_improphe")
NODATA = -9999
WINDOWS = {"w4": (1, 91, 181, 271, 365), "w3": (1, 121, 241, 365)}
KSD_M = 30.0   # KERNEL_TEXT of every run: ksd = 3 pixels


def _grid(ds):
    t = ds.transform
    ny, nx = ds.height, ds.width
    xs = t.c + (np.arange(nx) + 0.5) * t.a
    ys = t.f + (np.arange(ny) + 0.5) * t.e
    return xs, ys


def _dataarray(values, dims, coords, crs):
    import rioxarray  # noqa: F401
    import xarray as xr

    da = xr.DataArray(values, dims=dims, coords=coords)
    return da.rio.write_crs(crs)


def load_fine():
    """The screened Sentinel-2A cube (time, band, y, x) float32, NaN where FORCE's QAI screening drops the observation."""
    files = sorted((DATA / "ard" / "X0000_Y0000").glob("*_LEVEL2_SEN2A_BOA.tif"))
    vals, dates = [], []
    for f in files:
        with rasterio.open(f) as ds:
            a = ds.read().astype(np.float32)
            xs, ys = _grid(ds)
            crs = ds.crs
        with rasterio.open(str(f).replace("_BOA.tif", "_QAI.tif")) as ds:
            q = ds.read(1)
        a[:, (q & 0b1111) != 0] = np.nan          # off, cloud state 1-3, shadow
        a[a == NODATA] = np.nan
        vals.append(a)
        dates.append(pd.to_datetime(f.name[:8], format="%Y%m%d"))
    return _dataarray(np.stack(vals), ("time", "band", "y", "x"),
                      {"time": pd.DatetimeIndex(dates), "band": np.arange(vals[0].shape[0]), "y": ys, "x": xs}, crs)


def load_l8():
    files = sorted((DATA / "ard" / "X0000_Y0000").glob("*_LEVEL2_LND08_BOA.tif"))
    vals, dates = [], []
    for f in files:
        with rasterio.open(f) as ds:
            a = ds.read().astype(np.float32)
            xs, ys = _grid(ds)
            crs = ds.crs
        with rasterio.open(str(f).replace("_BOA.tif", "_QAI.tif")) as ds:
            q = ds.read(1)
        a[:, (q & 0b1111) != 0] = np.nan
        a[a == NODATA] = np.nan
        vals.append(a)
        dates.append(pd.to_datetime(f.name[:8], format="%Y%m%d"))
    return _dataarray(np.stack(vals), ("time", "band", "y", "x"),
                      {"time": pd.DatetimeIndex(dates), "band": np.arange(vals[0].shape[0]), "y": ys, "x": xs}, crs), files


def load_field(years=(2018, 2019)):
    f = DATA / "coarse" / "X0000_Y0000" / "FIELD.tif"
    with rasterio.open(f) as ds:
        a = ds.read().astype(np.float32)
        xs, ys = _grid(ds)
        crs = ds.crs
    a[a == NODATA] = np.nan
    return _dataarray(a, ("year", "y", "x"), {"year": np.array(years), "y": ys, "x": xs}, crs)


def force_tif(name, pattern):
    hits = sorted((DATA / "out" / name / "X0000_Y0000").glob(pattern))
    return hits


def has_run(name):
    return bool(force_tif(name, "*_IMP.tif"))


def read(path):
    with rasterio.open(path) as ds:
        return ds.read().astype(np.int32)


def trunc16(x):
    """``(short)x`` as FORCE writes it: toward zero (NaN stays NaN, returned as NODATA)."""
    return np.where(np.isfinite(x), np.trunc(np.nan_to_num(x)), NODATA).astype(np.int32)


def agree(got, want, mask):
    """(n, % equal, % within 1, max |dif|) over ``mask``."""
    d = np.abs(got.astype(np.int64) - want.astype(np.int64))[mask]
    n = d.size
    if n == 0:
        return 0, 100.0, 100.0, 0
    return n, 100.0 * (d == 0).mean(), 100.0 * (d <= 1).mean(), int(d.max())


def stats_vs(got, want, mask):
    """Statistics of a float ``got`` against FORCE's ``want`` (the default mode is not meant to equal it)."""
    g, w = got[mask].astype(np.float64), want[mask].astype(np.float64)
    d = np.abs(g - w)
    rel = d / np.maximum(np.abs(w), 1.0)
    return dict(corr=float(np.corrcoef(g, w)[0, 1]), within1=100.0 * (d <= 1).mean(),
                within1pct=100.0 * (rel <= 0.01).mean(), within5pct=100.0 * (rel <= 0.05).mean())


def rmse(a, truth, mask):
    return float(np.sqrt(np.mean((a[mask].astype(np.float64) - truth[mask].astype(np.float64)) ** 2)))


def nn_check():
    """Zeit's nearest-neighbour regridding of the 30 m field equals the replication (the integer factor)."""
    import zeit._improphe as imp

    fine = load_fine()
    fld = load_field()
    n_rep = fine.sizes["y"] // fld.sizes["y"]
    got = imp._on_grid(fld, fine, fld.values, 10.0)
    want = np.repeat(np.repeat(fld.values, n_rep, 1), n_rep, 2)
    ok = np.array_equal(np.nan_to_num(got, nan=-1), np.nan_to_num(want, nan=-1))
    print(f"nearest neighbour of the field ({n_rep}x): Zeit's reproject_match equals the replication of each pixel: {ok}")
    return ok


def run_cf(jobs, fine, results, only_default=False):
    import zeit

    fld = load_field()
    truth = np.load(DATA / "truth.npz")["peak_ndvi"] * 10000.0                       # (2, n, n)
    nn = np.repeat(np.repeat(fld.values, fine.sizes["y"] // fld.sizes["y"], 1), fine.sizes["x"] // fld.sizes["x"], 2)
    for h in (10, 20):
        for w in ("w4", "w3"):
            seasons = WINDOWS[w]
            if not has_run(f"cf_h{h}_{w}_A"):
                continue
            kw = dict(by="year", seasons=seasons, kernel_size=h * 10.0, texture_size=KSD_M, nodata=None, n_jobs=jobs)
            zf = zeit.improphe(fld, fine, force=True, **kw)
            zd = zeit.improphe(fld, fine, **kw)
            zn = zeit.improphe(fld, fine, pca="force", **kw)
            print(f"\n== CF-ImproPhe, h = {h} px, seasons {' '.join(map(str, seasons))}; components: {zf.attrs['improphe']}")
            zt = trunc16(zf.values)
            finite = np.isfinite(zf.values)
            for blas in ("A", "B"):
                name = f"cf_h{h}_{w}_{blas}"
                hits = force_tif(name, "*_IMP.tif")
                if len(hits) != 1:
                    print(f"  {name}: no FORCE output")
                    continue
                want = read(hits[0])                                                # (2, n, n)
                for i, year in enumerate((2018, 2019)):
                    n, eq, w1, mx = agree(zt[i], want[i], finite[i])
                    print(f"  {name} {year}: {n:>7} px  equal {eq:8.4f} %  within 1 {w1:8.4f} %  max |dif| {mx}")
                    results.append(dict(part="cf", name=name, layer=year, n=n, equal=eq, within1=w1, maxdif=mx))
                if blas == "A" and w == "w4":
                    out = want[0][~finite[0]]
                    vals, cnt = np.unique(out, return_counts=True)
                    print(f"  FORCE's 2018 values at the {out.size} pixels Zeit leaves NaN (no coarse value): "
                          f"{dict(zip(vals[:6].tolist(), cnt[:6].tolist()))}{' ...' if vals.size > 6 else ''}")
            if w == "w4":
                name = f"cf_h{h}_w4_B"
                want = read(force_tif(name, "*_IMP.tif")[0]).astype(np.float64)
                for i, year in enumerate((2018, 2019)):
                    m = finite[i] & (want[i] != NODATA) & (want[i] != 0)
                    mt = m & np.isfinite(truth[i])
                    st = stats_vs(zd.values[i], want[i], m)
                    print(f"  default mode vs FORCE {year}: corr {st['corr']:.5f}, within 1 {st['within1']:.2f} %, "
                          f"within 1 % {st['within1pct']:.2f} %, within 5 % {st['within5pct']:.2f} %")
                    print(f"  RMSE to the truth (peak NDVI x 10000) {year}: FORCE {rmse(want[i], truth[i], mt):.1f}, "
                          f"Zeit force {rmse(zf.values[i], truth[i], mt):.1f}, Zeit default {rmse(zd.values[i], truth[i], mt):.1f}, "
                          f"FORCE's components with float means {rmse(zn.values[i], truth[i], mt):.1f}, "
                          f"nearest neighbour {rmse(nn[i], truth[i], mt):.1f}")
                    results.append(dict(part="cf-default", h=h, layer=year, **st,
                                        rmse_force=rmse(want[i], truth[i], mt), rmse_zeit_force=rmse(zf.values[i], truth[i], mt),
                                        rmse_zeit=rmse(zd.values[i], truth[i], mt),
                                        rmse_force_pca_float=rmse(zn.values[i], truth[i], mt), rmse_nn=rmse(nn[i], truth[i], mt)))


def run_l2(jobs, fine, results):
    import zeit

    l8, files = load_l8()
    n = fine.sizes["y"]
    tr = np.load(DATA / "truth.npz")["l8_truth"].astype(np.float64)                    # (nt, 6, n, n)
    nn_all = np.repeat(np.repeat(l8.values, n // l8.sizes["y"], 2), n // l8.sizes["x"], 3)
    for h in (10, 20):
        for w in ("w4", "w3"):
            seasons = WINDOWS[w]
            if not has_run(f"l2_h{h}_{w}_A"):
                continue
            kw = dict(by=None, seasons=seasons, kernel_size=h * 10.0, texture_size=KSD_M, nodata=None, n_jobs=jobs)
            zf = zeit.improphe(l8, fine, force=True, **kw)
            zd = zeit.improphe(l8, fine, **kw)
            zn = zeit.improphe(l8, fine, pca="force", **kw)
            print(f"\n== L2-ImproPhe, h = {h} px, seasons {' '.join(map(str, seasons))}; components: {zf.attrs['improphe']}")
            zv = zf.values                                                                # (nt, 6, n, n)
            zt = trunc16(zv)
            finite = np.isfinite(zv)
            for blas in ("A", "B"):
                name = f"l2_h{h}_{w}_{blas}"
                want = np.stack([read(p) for p in
                                 (force_tif(name, Path(f).name.replace("_BOA.tif", "_IMP.tif"))[0] for f in files)])
                tot = [0, 0, 0, 0]
                worst = 0
                eq_all = w1_all = 0.0
                n_all = 0
                per_band = []
                for b in range(6):
                    nn_, eq, w1, mx = agree(zt[:, b], want[:, b], finite[:, b])
                    per_band.append((nn_, eq, w1, mx))
                    n_all += nn_
                    eq_all += eq * nn_
                    w1_all += w1 * nn_
                    worst = max(worst, mx)
                print(f"  {name}: {n_all:>8} values  equal {eq_all / n_all:8.4f} %  within 1 {w1_all / n_all:8.4f} %  "
                      f"max |dif| {worst}   (bands: " + ", ".join(f"{p[1]:.3f}/{p[2]:.3f}" for p in per_band) + ")")
                results.append(dict(part="l2", name=name, n=n_all, equal=eq_all / n_all, within1=w1_all / n_all, maxdif=worst))
                if blas == "A" and w == "w4":
                    out = want[:, 0][~finite[:, 0]]
                    vals, cnt = np.unique(out, return_counts=True)
                    top = np.argsort(-cnt)[:4]
                    print(f"  FORCE's values at the {out.size} band-1 pixels Zeit leaves NaN (screened or NoData): "
                          f"most frequent {dict(zip(vals[top].tolist(), cnt[top].tolist()))}")
            if w == "w4":
                want = np.stack([read(p) for p in
                                 (force_tif(f"l2_h{h}_w4_B", Path(f).name.replace("_BOA.tif", "_IMP.tif"))[0] for f in files)])
                m = finite & (want != NODATA)
                mt = m
                st = stats_vs(zd.values, want, m)
                print(f"  default mode vs FORCE: corr {st['corr']:.5f}, within 1 {st['within1']:.2f} %, within 1 % "
                      f"{st['within1pct']:.2f} %, within 5 % {st['within5pct']:.2f} %")
                r = dict(force=rmse(want, tr, mt), zeit_force=rmse(zv, tr, mt), zeit=rmse(zd.values, tr, mt),
                         force_pca_float=rmse(zn.values, tr, mt), nn=rmse(nn_all, tr, mt))
                print(f"  RMSE to the truth (reflectance x 10000): FORCE {r['force']:.1f}, Zeit force {r['zeit_force']:.1f}, "
                      f"Zeit default {r['zeit']:.1f}, FORCE's components with float means {r['force_pca_float']:.1f}, nearest neighbour {r['nn']:.1f}")
                results.append(dict(part="l2-default", h=h, **st, **{f"rmse_{k}": v for k, v in r.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="cf,l2")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--json", default=None)
    ap.add_argument("--data", default=None, help="another folder with the cube and FORCE's outputs")
    a = ap.parse_args()
    global DATA
    if a.data:
        DATA = Path(a.data)
    no_power_throttling()
    parts = a.part.split(",")
    fine = load_fine()
    print(f"fine: {fine.shape} (time, band, y, x); {int(np.isnan(fine.values[:, 0]).all(0).sum())} pixels without any observation")
    results = []
    ok = nn_check()
    if "cf" in parts:
        run_cf(a.jobs, fine, results)
    if "l2" in parts:
        run_l2(a.jobs, fine, results)
    if a.json:
        import json

        Path(a.json).write_text(json.dumps(results, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
