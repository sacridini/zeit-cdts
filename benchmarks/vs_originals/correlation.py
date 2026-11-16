"""zeit.lagged_correlation and zeit.fdr against xarray 2026.9 (xr.corr of the series with the driver shifted by each lag),
scipy 1.18 (pearsonr, spearmanr, the Student t distribution, false_discovery_control) and statsmodels 0.15
(fdrcorrection).

The pipeline is Zeit's and then the references': a 16-day series is averaged to months (Zeit's ``monthly_means``,
xarray's ``resample("MS").mean()``), the monthly climatology is taken out (Zeit's anomalies, xarray's groupby), and the
anomalies are correlated with the driver shifted by 0 to 6 months. What is compared:

- r and the number of pairs per lag and pixel (``xr.corr``), at float32 precision (Zeit's output), and in float64
  through ``_core.correlation.lagged`` (to 1e-12);
- the best lag, and its r;
- the p-values with ``effective_n=False`` (``scipy.stats.pearsonr`` on a sample of pixels, the t formula of scipy
  for all) and with the effective number of pairs (n (1 - rx ry) / (1 + rx ry), Bretherton et al. 1999, coded in
  numpy, and scipy's t distribution with the non-integer degrees of freedom);
- Spearman (``scipy.stats.spearmanr`` on a sample);
- the false discovery rate (``scipy.stats.false_discovery_control`` and ``statsmodels`` ``fdrcorrection``).

Two cubes: a synthetic one (16-day dates, pixels driven by a monthly AR(1) series at lags of 0 to 3 months, a share of
them by nothing, gaps) and the monthly EVI of eof.py (the driver: the mean of a block of the cube far from the rest).

    python benchmarks/vs_originals/correlation.py [--sizes 316x316] [--no-real] [--no-timing]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import statsmodels.stats.multitest
import xarray as xr

import eof_data
import zeit
from common import best_of, record, threads_label

LAGS = 6
JOBS = (1, 8)
MIN_PAIRS = 10


# ---------------------------------------------------------------- the data


def sixteen_day(first_year, years):
    """Dates every 16 days from January 1st, 23 a year (the grid of the MODIS composites)."""
    return pd.DatetimeIndex([pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=16 * k)
                             for y in range(first_year, first_year + years) for k in range(23)])


def synthetic(ny, nx, seed=0, years=25):
    """x (time, y, x) every 16 days and the driver y, a monthly series (pandas). 60 % of the pixels follow y at their
    own lag (0 to 3 months; correlation about 0.4 to 0.6), the others are noise; AR(1) noise (0.5 for x, 0.6 for y);
    a seasonal cycle on both; 10 % of the values missing, and 1 % of the pixels nearly empty."""
    rng = np.random.default_rng(seed)
    dates = sixteen_day(2001, years)
    months = pd.date_range("2000-07-01", periods=12 * years + 6, freq="MS")     # 6 months before the first date
    e = rng.normal(size=len(months))
    y = scipy.signal.lfilter([1.0], [1.0, -0.6], e) + 1.5 * np.cos(2 * np.pi * (months.month.values - 1) / 12)
    n = ny * nx
    lag = rng.integers(0, 4, size=n)
    driven = rng.random(n) < 0.6
    month_of = (dates.year.values - 2000) * 12 + dates.month.values - 7             # index in `months` of each date
    ydrv = y[month_of[:, None] - lag[None, :]] * driven[None, :]
    noise = scipy.signal.lfilter([1.0], [1.0, -0.5], rng.normal(size=(len(dates), n)), axis=0)
    season = 3.0 * np.cos(2 * np.pi * dates.dayofyear.values[:, None] / 365.25 + rng.uniform(0, 6.28, size=(1, n)))
    x = 10.0 + season + 0.5 * ydrv + noise
    x[rng.random(x.shape) < 0.10] = np.nan
    sparse = rng.random(n) < 0.01
    x[:, sparse] = np.where(rng.random((len(dates), sparse.sum())) < 0.05, x[:, sparse], np.nan)
    cube = xr.DataArray(x.reshape(len(dates), ny, nx), dims=("time", "y", "x"), coords={
        "time": dates, "y": 3000000 - 250.0 * np.arange(ny), "x": 4000000 + 250.0 * np.arange(nx)}).rio.write_crs(
        "EPSG:3035")
    series = pd.Series(y[6:], index=months[6:] + pd.Timedelta(days=14), name="driver")   # from 2001-01, mid-month
    return cube, series


def real():
    """The monthly EVI cube and the mean of a 120 x 120 block far from the others as the driver."""
    cube = eof_data.real_monthly()
    block = cube.isel(y=slice(520, 640), x=slice(520, 640))
    drv = block.mean(("y", "x")).to_series()
    drv.name = "driver"
    return cube, drv


# ---------------------------------------------------------------- the references


def anomalies_xarray(cube, series):
    """The monthly means and monthly-climatology anomalies of the cube and of the driver, by xarray (float64)."""
    xm = cube.astype("float64").resample(time="MS").mean()
    ym = xr.DataArray(series.to_numpy(np.float64), dims="time", coords={"time": series.index}).resample(time="MS").mean()
    ym = ym.reindex(time=xm.time)               # the same months (the pairs are where both are valid anyway)
    return eof_data.xarray_anomalies(xm), eof_data.xarray_anomalies(ym)


def xarray_lags(xa, ya, lags=LAGS):
    """r and the number of pairs of xr.corr(x, y shifted by each lag): ``(lag, y, x)`` arrays."""
    r, n = [], []
    for k in range(lags + 1):
        ys = ya.shift(time=k)
        r.append(xr.corr(xa, ys, dim="time").values)
        n.append((xa.notnull() & ys.notnull()).sum("time").values)
    return np.stack(r), np.stack(n)


def p_from_r(r, n):
    """The two-sided p of Pearson's r with n pairs (Student t, n - 2 degrees of freedom), as scipy."""
    with np.errstate(all="ignore"):
        t = r * np.sqrt((n - 2) / (1 - r * r))
    return 2 * scipy.stats.t.sf(np.abs(t), n - 2)


def lag1(v):
    """Correlation of v[t] with v[t + 1] over the consecutive pairs that are both valid (Bretherton et al. 1999)."""
    a, b = v[:-1], v[1:]
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 else 0.0


def effective_p(x, y, k, method="pearson"):
    """r, pairs, effective pairs and p of one pixel's series x against y shifted by k (the paper's formula, scipy's
    t distribution with the non-integer degrees of freedom): the independent route to Zeit's."""
    ys = np.full_like(y, np.nan)
    ys[k:] = y[:len(y) - k] if k else y
    ok = np.isfinite(x) & np.isfinite(ys)
    n = int(ok.sum())
    if n < max(MIN_PAIRS, 3):
        return np.nan, n, np.nan, np.nan
    if method == "pearson":
        r = float(np.corrcoef(x[ok], ys[ok])[0, 1])
    else:
        r = float(scipy.stats.spearmanr(x[ok], ys[ok])[0])
    prod = min(max(lag1(x) * lag1(ys), 0.0), 0.99)
    n_eff = n * (1 - prod) / (1 + prod)
    if n_eff <= 2:
        return r, n, np.nan, np.nan
    df = n_eff - 2
    p = 2 * scipy.stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)
    return r, n, n_eff, float(p)


def same(a, b):
    """Max |a - b| over the values that are finite in both, and whether the NaN are the same."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    return (float(np.abs(a - b)[fa & fb].max()) if (fa & fb).any() else 0.0), bool((fa == fb).all())


# ---------------------------------------------------------------- the checks


def core_f64(xa, ya, method="pearson", effective=True, jobs=4):
    """``_core.correlation.lagged`` on the references' anomalies (float64), the lags the rows shifted."""
    t = xa.sizes["time"]
    xs = np.ascontiguousarray(xa.values.reshape(t, -1))
    rows = np.arange(t)
    index = np.array([np.where(rows - k >= 0, rows - k, -1) for k in range(LAGS + 1)], dtype=np.int32)
    return zeit._core.correlation.lagged(xs, np.ascontiguousarray(ya.values), index, method=method, min_pairs=MIN_PAIRS,
                                         effective_n=effective, best="absolute", sidak=True, n_jobs=jobs)


def check(name, cube, series, sample=1500):
    print(f"{name}: {cube.sizes['time']} dates x {cube.sizes['y']}x{cube.sizes['x']} pixels, "
          f"{float(cube.notnull().mean()):.1%} valid")
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    xa, ya = anomalies_xarray(cube, series)
    t = xa.sizes["time"]
    # Zeit's own anomalies of the monthly series against xarray's
    ds = zeit.lagged_correlation(cube, series, lags=LAGS, n_jobs=8)
    r_ref, n_ref = xarray_lags(xa, ya)
    r_ref = np.where(n_ref >= MIN_PAIRS, r_ref, np.nan)
    # 1. the public result: r, n, best lag, in float32
    r_diff, r_nan = same(ds.r.values, r_ref.astype(np.float32).reshape(LAGS + 1, ny, nx))
    n_diff = int(np.abs(ds.n.values.astype(np.int64) - n_ref.reshape(LAGS + 1, ny, nx)).max())
    n_ok = np.where(n_ref >= MIN_PAIRS, n_ref, n_ref)        # pairs are reported also where r is NaN
    absr = np.where(np.isnan(r_ref), -1.0, np.abs(r_ref))
    best_ref = np.where((absr >= 0).any(axis=0), absr.argmax(axis=0), -1)
    best_diff = int((ds.best_lag.values != best_ref.reshape(ny, nx)).sum())
    best_r_ref = np.take_along_axis(np.nan_to_num(r_ref, nan=np.nan), np.maximum(best_ref, 0)[None], axis=0)[0]
    best_r_ref = np.where(best_ref >= 0, best_r_ref, np.nan)
    br_diff, br_nan = same(ds.best_r.values, best_r_ref.reshape(ny, nx).astype(np.float32))
    print(f"  r (float32): max diff {r_diff:.1e} (NaN the same: {r_nan}); pairs: max diff {n_diff}; best lag: "
          f"{best_diff} of {best_ref.size} pixels differ; best r max diff {br_diff:.1e}")
    assert r_diff < 2e-7 and r_nan and n_diff == 0 and best_diff == 0 and br_diff < 2e-7 and br_nan
    # 2. float64 through the core: r to 1e-12; the p-values of independent pairs against scipy's t
    core = core_f64(xa, ya, effective=False)
    cr = np.stack([core["r"][k * ny * nx:(k + 1) * ny * nx] for k in range(LAGS + 1)]) if core["r"].ndim == 1 else \
        core["r"].reshape(LAGS + 1, -1)
    cp = core["p"].reshape(LAGS + 1, -1)
    d64, nan64 = same(cr, r_ref.reshape(LAGS + 1, -1))
    p_ref = p_from_r(r_ref.reshape(LAGS + 1, -1), n_ref.reshape(LAGS + 1, -1).astype(float))
    dp, nanp = same(cp, p_ref)
    rel = float(np.nanmax(np.abs(cp - p_ref) / np.maximum(p_ref, 1e-300)))
    print(f"  float64 core: r max diff {d64:.1e}; p (effective_n=False) against scipy.stats.t.sf of the t statistic: "
          f"max diff {dp:.1e}, max relative {rel:.1e}")
    assert d64 < 1e-12 and nan64 and dp < 1e-12 and nanp
    # 3. a sample of pixels: scipy.stats.pearsonr and spearmanr, and the effective n by the paper's formula
    rng = np.random.default_rng(0)
    pick = rng.choice(ny * nx, size=min(sample, ny * nx), replace=False)
    xs = xa.values.reshape(t, -1)[:, pick]
    yv = ya.values
    worst = dict(pearson_r=0.0, pearson_p=0.0, spearman_r=0.0, spearman_p=0.0, neff=0.0, p_eff=0.0)
    core_eff = core_f64(xa, ya, effective=True)
    core_sp = core_f64(xa, ya, method="spearman", effective=False)
    ce = {k: core_eff[k].reshape(LAGS + 1, -1) for k in ("r", "p", "n_eff")}
    cs = {k: core_sp[k].reshape(LAGS + 1, -1) for k in ("r", "p")}
    for j, px in enumerate(pick):
        for k in range(LAGS + 1):
            ys = np.full(t, np.nan)
            ys[k:] = yv[:t - k]
            ok = np.isfinite(xs[:, j]) & np.isfinite(ys)
            if ok.sum() < MIN_PAIRS:
                continue
            pr = scipy.stats.pearsonr(xs[ok, j], ys[ok])
            worst["pearson_r"] = max(worst["pearson_r"], abs(pr.statistic - cr[k, px]))
            worst["pearson_p"] = max(worst["pearson_p"], abs(pr.pvalue - cp[k, px]))
            sr = scipy.stats.spearmanr(xs[ok, j], ys[ok])
            worst["spearman_r"] = max(worst["spearman_r"], abs(sr.statistic - cs["r"][k, px]))
            worst["spearman_p"] = max(worst["spearman_p"], abs(sr.pvalue - cs["p"][k, px]))
            r, n, n_eff, p = effective_p(xs[:, j], yv, k)
            if np.isfinite(p):
                worst["neff"] = max(worst["neff"], abs(n_eff - ce["n_eff"][k, px]) / n_eff)
                worst["p_eff"] = max(worst["p_eff"], abs(p - ce["p"][k, px]) / max(p, 1e-300))
    print(f"  {len(pick)} pixels x {LAGS + 1} lags: pearsonr r {worst['pearson_r']:.1e}, p {worst['pearson_p']:.1e}; "
          f"spearmanr r {worst['spearman_r']:.1e}, p {worst['spearman_p']:.1e}; effective n (relative) "
          f"{worst['neff']:.1e}, its p (relative) {worst['p_eff']:.1e}")
    assert worst["pearson_r"] < 1e-12 and worst["pearson_p"] < 1e-10 and worst["spearman_r"] < 1e-12
    assert worst["spearman_p"] < 1e-10 and worst["neff"] < 1e-10 and worst["p_eff"] < 1e-8
    # 4. the false discovery rate on Zeit's best p (the Sidak-adjusted p of the best lag)
    bp = core_eff["best_p"]
    q, star, sig = zeit.fdr(bp, 0.10)
    ok = np.isfinite(bp)
    q_sp = np.full_like(bp, np.nan)
    q_sp[ok] = scipy.stats.false_discovery_control(bp[ok], method="bh")
    rej, q_sm = statsmodels.stats.multitest.fdrcorrection(bp[ok], alpha=0.10, method="indep")
    dq = float(np.nanmax(np.abs(q - q_sp)))
    dq_sm = float(np.abs(q[ok] - q_sm).max())
    same_sig = bool((sig[ok] == rej).all() and (sig[ok] == (q_sp[ok] <= 0.10)).all())
    ds_q_diff = float(np.nanmax(np.abs(ds.q.values.reshape(-1) - q_sp))) if ok.all() else \
        same(ds.q.values.reshape(-1), q_sp.astype(np.float32))[0]
    print(f"  FDR on {int(ok.sum())} best p: q against scipy {dq:.1e}, against statsmodels {dq_sm:.1e}; the same "
          f"{int(sig.sum())} significant at 0.10: {same_sig}; the public q (float32) max diff {ds_q_diff:.1e}; "
          f"{float(ds.significant.values.mean()):.1%} of the pixels significant")
    assert dq < 1e-12 and dq_sm < 1e-12 and same_sig and ds_q_diff < 1e-6
    # the public p and best_p: float32 of the core's
    pd_, pn = same(ds.best_p.values.reshape(-1), core_eff["best_p"].astype(np.float32))
    assert pn and pd_ < 1e-6
    return xa, ya


def drift_check(cube, series):
    """The p-values with the effective n are larger than the independent ones: how much (the point of the option)."""
    ds_e = zeit.lagged_correlation(cube, series, lags=LAGS, effective_n=True, n_jobs=8)
    ds_i = zeit.lagged_correlation(cube, series, lags=LAGS, effective_n=False, n_jobs=8)
    print(f"  significant at FDR 0.10 with the effective n: {float(ds_e.significant.mean()):.1%}; with n: "
          f"{float(ds_i.significant.mean()):.1%}")
    assert ds_i.significant.mean() >= ds_e.significant.mean()


# ---------------------------------------------------------------- the times


def timing(name, cube, series, xa, ya, size):
    """Zeit's call (the monthly means, the anomalies, 7 lags, r, n, n_eff, p, the Sidak p, the FDR) with 1 and 8
    threads, against xr.corr of the anomalies per lag (r only: no p, no effective n), the same plus the t-test's
    p from scipy (``effective_n=False``), and a loop of scipy's pearsonr and spearmanr over a sample of pixels."""
    small = cube.size < 3e7
    reps = 3 if small else 1

    def once(f):
        import time

        t0 = time.perf_counter()
        f()
        return time.perf_counter() - t0

    def ref_r():
        xarray_lags(xa, ya)

    def ref_rp():
        r, n = xarray_lags(xa, ya)
        p_from_r(r, n.astype(float))

    t_r = min(once(ref_r) for _ in range(reps))
    t_rp = min(once(ref_rp) for _ in range(reps))
    print(f"{name} timing: xr.corr x {LAGS + 1} lags {t_r:.2f} s; with n and the t p-values {t_rp:.2f} s")
    for jobs in JOBS:
        t_z = best_of(lambda: zeit.lagged_correlation(cube, series, lags=LAGS, n_jobs=jobs), 3 if small else 2)
        note = ("Zeit's whole call: monthly means, anomalies, r, n, n_eff, p and the Sidak p of the best lag, the FDR; "
                "xarray's: xr.corr of the anomalies (given) per lag")
        record("lagged_correlation", "xarray 2026.9 xr.corr (r only)", size, t_r, t_z, threads_label(jobs), note)
        t_z0 = best_of(lambda: zeit.lagged_correlation(cube, series, lags=LAGS, effective_n=False, n_jobs=jobs),
                       3 if small else 2)
        record("lagged_correlation", "xarray xr.corr + scipy t p", size, t_rp, t_z0, threads_label(jobs),
               "effective_n=False; the references: xr.corr, the pairs and scipy.stats.t.sf of the t statistic")
    # a sample of pixels, the scipy loops
    rng = np.random.default_rng(1)
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    pick = rng.choice(ny * nx, size=2000, replace=False)
    flat = cube.values.reshape(cube.sizes["time"], -1)[:, pick]
    sub = xr.DataArray(flat.reshape(cube.sizes["time"], 1, -1), dims=("time", "y", "x"),
                       coords={"time": cube.time.values, "y": [0.0], "x": np.arange(2000.0)})
    xs = xa.values.reshape(xa.sizes["time"], -1)[:, pick]
    yv = ya.values
    tt = xa.sizes["time"]
    pairs = []
    for k in range(LAGS + 1):
        ys = np.full(tt, np.nan)
        ys[k:] = yv[:tt - k]
        pairs.append(ys)

    def loop(fn):
        for j in range(2000):
            for ys in pairs:
                ok = np.isfinite(xs[:, j]) & np.isfinite(ys)
                if ok.sum() >= MIN_PAIRS:
                    fn(xs[ok, j], ys[ok])

    for method, fn, label in (("pearson", scipy.stats.pearsonr, "scipy pearsonr loop"),
                              ("spearman", scipy.stats.spearmanr, "scipy spearmanr loop")):
        t_ref = once(lambda: loop(fn))
        for jobs in JOBS:
            t_z = best_of(lambda: zeit.lagged_correlation(sub, series, lags=LAGS, method=method, effective_n=False,
                                                          n_jobs=jobs), 3)
            record(f"lagged_correlation ({method})", f"scipy.stats {label.split()[1]} loop", size + ", 2000 px sample",
                   t_ref, t_z, threads_label(jobs),
                   f"2000 pixels x {LAGS + 1} lags, effective_n=False; the loop on the anomalies (given), Zeit's call "
                   f"with the monthly means and anomalies")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=["316x316"])
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    print("zeit from", zeit.__file__)
    for s in args.sizes:
        ny, nx = (int(v) for v in s.split("x"))
        cube, series = synthetic(ny, nx)
        xa, ya = check(f"synthetic ({s})", cube, series)
        drift_check(cube, series)
        if not args.no_timing:
            timing("synthetic", cube, series, xa, ya, f"{cube.sizes['time']} dates x {ny}x{nx} px (synthetic)")
    if not args.no_real:
        cube, series = real()
        xa, ya = check("real EVI", cube, series)
        drift_check(cube, series)
        if not args.no_timing:
            timing("real EVI", cube, series, xa, ya,
                   f"{cube.sizes['time']} months x {cube.sizes['y']}x{cube.sizes['x']} px (EVI)")


if __name__ == "__main__":
    main()
