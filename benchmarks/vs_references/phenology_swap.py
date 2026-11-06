"""Runs the phenology on fixed inputs and saves everything it returns, to compare two builds of the Rust core.

Made for the swap of the phenology core (the phenofit-derived module, then the clean-room zeit-core/src/phenology):
run it with the old extension (``--out phenology_before.npz``), rebuild, run it again (``--out phenology_after.npz``)
and compare with ``--compare before.npz after.npz [--report report.md]``.

Inputs (all generated here, no random state outside the seeds):
  unit   the synthetic series of tests/test_phenology.py through the compiled core: a clean Beck-like year pair, a
         flat series, a spike with and without its weight, min_season_length 0/45/720 days, the season detector's
         strict and retried runs, and the HANTS + derivative case of the accessor test
  cube   2000 pixels, 5 years of 16-day composites (23 a year), 1000 with one season a year and 1000 with two,
         down-pointing clouds (flagged in the weights for 70% of them), 3% gaps (a zero with weight 0); the seven curves, each with
         annual=True and with annual=False, max_seasons=3 (zeit.phenology); plus Beck without weights, and with HANTS
         instead of Whittaker
  real   the EVI raster of cdts_vs_originals (29 x 22 pixels, 575 dates, 2001-2025; skipped if absent): the seven
         curves, annual, and Beck with annual=False
At most 4 threads (n_jobs=4).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ZEIT_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import xarray as xr  # noqa: E402
import zeit  # noqa: E402,F401
import zeit.xarray_api  # noqa: E402,F401
from zeit import _core  # noqa: E402
from zeit._series_api import PHENOLOGY_METRICS  # noqa: E402

CurveType = _core.phenology.CurveType
fit = _core.phenology.fit_phenology_batch
debug_split = _core.phenology.debug_split_seasons
CURVES = ["beck", "elmore", "gu", "klos", "zhang", "ag", "dl"]
N_JOBS = 4
N_METRICS = 21
EVI = REPO.parent / "cdts_vs_originals" / "data" / "phenology" / "EVI_Series_ID_218_2001_2025.tif"


def synthetic_evi_curve(time_steps=46):
    """tests/test_phenology.py::_synthetic_evi_curve."""
    dates = np.array([t * 16 + (t // 23) * 365 for t in range(time_steps)], dtype=np.float64)
    doy = dates % 365
    growth = (np.sin((doy / 365.0) * np.pi - np.pi / 2) + 1) / 2.0
    return dates, 0.2 + growth * 0.6


def unit_cases():
    out = {}
    dates, clean = synthetic_evi_curve()
    kw = dict(dates_array=dates, curve_type=int(CurveType.BECK), extraction_method=0, max_seasons=2,
              min_season_length=3, min_amplitude=0.05, n_jobs=1)
    out["unit__clean"] = fit(values_array=clean.reshape(1, -1), **kw)
    out["unit__flat"] = fit(values_array=np.full((1, dates.size), 0.3), **kw)
    noisy = clean.copy()
    noisy[5] = 0.98
    w = np.ones((1, clean.size))
    w[0, 5] = 0.02
    out["unit__spike_unweighted"] = fit(values_array=noisy.reshape(1, -1), **kw)
    out["unit__spike_weighted"] = fit(values_array=noisy.reshape(1, -1), weights_array=w, **kw)
    for c in CURVES:
        out[f"unit__clean_{c}"] = fit(values_array=clean.reshape(1, -1), **{**kw, "curve_type": int(getattr(CurveType, c.upper()))})
    # tests/test_phenology.py::test_min_season_length_is_calendar_days_not_observation_count
    d5 = 1.0 + 16.0 * np.arange(23 * 5)
    g = 0.2 + 0.3 * np.maximum(0.0, np.sin(2 * np.pi * (d5 - 80.0) / 365.0))
    for msl in (0, 45, 720):
        out[f"unit__msl{msl}"] = fit(values_array=g[None, :], dates_array=d5, curve_type=int(CurveType.BECK),
                                     extraction_method=0, max_seasons=5, apply_whittaker=True, whittaker_lambda=5.0,
                                     min_amplitude=0.1, min_pixel_amplitude=0.1, min_season_length=msl, n_jobs=1)
    y = [1.0] * 3 + [0.0, 1.0, 0.75, 1.0, 0.75, 1.0, 0.0] + [1.0] * 3
    for retry in (False, True):
        out[f"unit__retry{int(retry)}"] = np.array(
            debug_split(y, min_season_length=0, min_amplitude=0.0, rtrough_max=0.6, r_min_filter=0.02,
                        retry_on_empty=retry), float).reshape(-1, 3)
    for retry in (False, True):
        out[f"unit__split_clean{int(retry)}"] = np.array(
            debug_split(list(clean), min_season_length=3, min_amplitude=0.05, rtrough_max=0.6, r_min_filter=0.02,
                        retry_on_empty=retry), float).reshape(-1, 3)
    # the accessor test with HANTS and the derivative method (2 years, 16-day composites, noise sd 0.05)
    time_steps, rows, cols = 46, 2, 2
    dts = pd.DatetimeIndex(np.concatenate([pd.date_range(f"{2001 + k}-01-01", periods=23, freq="16D")
                                           for k in range(2)]))
    np.random.seed(42)
    data = np.zeros((time_steps, rows, cols))
    for t in range(time_steps):
        growth = (np.sin(((dts[t].dayofyear - 1) / 365.0) * np.pi - np.pi / 2) + 1) / 2.0
        data[t] = 0.2 + growth * 0.6 + np.random.normal(0, 0.05, (rows, cols))
    da_ = xr.DataArray(data, dims=["time", "y", "x"], coords={"time": dts, "y": np.arange(rows), "x": np.arange(cols)})
    res = da_.zeit.phenology(curve="beck", method="derivative", max_seasons=2, apply_whittaker=False, apply_hants=True,
                             hants_frequencies=3, min_season_length=3, min_amplitude=0.1, n_jobs=1).compute()
    out["unit__hants_accessor"] = np.stack([res[m].values for m in PHENOLOGY_METRICS])
    return out


def cube_data(seed=11, n=2000, years=5):
    """Pixels of one season a year (first half) and two (second half); returns the (time, y, x) values, weights,
    dates and the true season dates (day of year of the 50% rise and fall; NaN for the second season of
    one-season pixels)."""
    rng = np.random.default_rng(seed)
    doy = np.concatenate([1.0 + 16.0 * np.arange(23) for _ in range(years)])
    day = doy + np.repeat(np.arange(years) * 365.0, 23)
    dates = np.concatenate([pd.date_range(f"{2001 + k}-01-01", periods=23, freq="16D") for k in range(years)])
    sig = lambda u: 1.0 / (1.0 + np.exp(-u))  # noqa: E731
    y = np.zeros((n, doy.size))
    truth = np.full((n, 4), np.nan)
    for p in range(n):
        base = rng.uniform(0.08, 0.3)
        amp = rng.uniform(0.3, 0.6)
        r1, r2 = rng.uniform(6, 14, 2)
        if p < n // 2:
            s = rng.uniform(100, 160)
            e = s + rng.uniform(110, 170)
            f = base + amp * (sig((doy - s) / r1) - sig((doy - e) / r2))
            truth[p, :2] = (s, e)
        else:
            s1 = rng.uniform(30, 60)
            e1 = s1 + rng.uniform(60, 90)
            s2 = e1 + rng.uniform(45, 70)
            e2 = s2 + rng.uniform(60, 90)
            a2 = amp * rng.uniform(0.6, 1.0)
            f = base + amp * (sig((doy - s1) / r1) - sig((doy - e1) / r2)) + a2 * (sig((doy - s2) / r1) - sig((doy - e2) / r2))
            truth[p] = (s1, e1, s2, e2)
        y[p] = f + rng.normal(0, 0.02, doy.size)
    cloud = rng.random(y.shape) < 0.12
    y[cloud] *= rng.uniform(0.2, 0.7, cloud.sum())
    w = np.ones_like(y)
    w[cloud & (rng.random(y.shape) < 0.7)] = 0.1
    gap = rng.random(y.shape) < 0.03
    y[gap] = 0.0  # the old core fails a whole pixel on a NaN: a gap is a zero with no weight
    w[gap] = 0.0
    cube = y.T.reshape(doy.size, 40, 50)
    wcube = w.T.reshape(doy.size, 40, 50)
    return cube, wcube, dates, truth


def run_cube(cube, wcube, dates, curve, annual, max_seasons=None, weights=True, **kw):
    da_ = xr.DataArray(cube, dims=["time", "y", "x"],
                       coords={"time": dates, "y": np.arange(cube.shape[1]), "x": np.arange(cube.shape[2])})
    t0 = time.perf_counter()
    res = da_.zeit.phenology(curve=curve, weights=wcube if weights else None, annual=annual, max_seasons=max_seasons,
                             nodata=None, n_jobs=N_JOBS, **kw).compute()
    dt = time.perf_counter() - t0
    return np.stack([res[m].values for m in PHENOLOGY_METRICS]), dt


def real_cube():
    import rioxarray
    ds = rioxarray.open_rasterio(EVI)
    cube = ds.where(ds >= -1).interpolate_na(dim="band", method="linear").fillna(0)
    dates = pd.date_range(start="2001-01-01", periods=cube.shape[0], freq="16D")
    return cube.values.astype(np.float64), dates


def collect():
    out = {}
    t0 = time.time()
    out.update(unit_cases())
    cube, wcube, dates, truth = cube_data()
    out["truth"] = truth
    for c in CURVES:
        for annual in (True, False):
            key = f"cube__{c}__{'annual' if annual else 'seasons'}"
            res, dt = run_cube(cube, wcube, dates, c, annual, max_seasons=None if annual else 3)
            out[key], out["time__" + key] = res, np.array(dt)
            print(f"{key}: {dt:.1f} s")
    for key, kw in (("cube__beck_noweights__annual", dict(weights=False)),
                    ("cube__beck_hants__annual", dict(apply_whittaker=False, apply_hants=True))):
        res, dt = run_cube(cube, wcube, dates, "beck", True, **kw)
        out[key], out["time__" + key] = res, np.array(dt)
        print(f"{key}: {dt:.1f} s")
    # the same gaps as NaN (the old core gives no season to a pixel with a NaN)
    res, dt = run_cube(np.where(wcube == 0, np.nan, cube), wcube, dates, "beck", True)
    out["cube__beck_nan__annual"], out["time__cube__beck_nan__annual"] = res, np.array(dt)
    print(f"cube__beck_nan__annual: {dt:.1f} s")
    if EVI.exists():
        from zeit._phenology import run_phenology_dask
        import dask.array as da
        vals, rdates = real_cube()
        doy = np.array([d.timetuple().tm_yday + (d.year - 2001) * 365 for d in rdates], dtype=np.float64)
        for c, annual, ms in [(c, True, 25) for c in CURVES] + [("beck", False, 3)]:
            key = f"real__{c}__{'annual' if annual else 'seasons'}"
            t1 = time.perf_counter()
            res = run_phenology_dask(da.from_array(vals), dates=doy, curve_type=int(getattr(CurveType, c.upper())),
                                     max_seasons=ms, whittaker_lambda=10.0, apply_whittaker=True,
                                     min_season_length=7, min_amplitude=0.0, min_pixel_amplitude=0.1,
                                     return_annual=annual, base_year=2001, n_jobs=N_JOBS).compute(scheduler="threads", num_workers=1)
            dt = time.perf_counter() - t1
            out[key], out["time__" + key] = np.asarray(res), np.array(dt)
            print(f"{key}: {dt:.1f} s")
    print(f"ran in {time.time() - t0:.1f} s")
    return out


# ---------------------------------------------------------------- comparison

DATE_METRICS = [i for i in range(19) if i != 17]


def _q(x):
    if x.size == 0:
        return "n/a"
    return f"{np.median(x):.3g} / {np.percentile(x, 90):.3g} / {np.percentile(x, 99):.3g}"


def n_seasons(a):
    """Seasons per pixel of an output (21, S, ...): the slots with a POP (or a LOS)."""
    ok = np.isfinite(a[18]) | np.isfinite(a[17])
    return ok.reshape(ok.shape[0], -1).sum(0)


def season_axis_is_first(a):
    return True  # cube and real outputs are (21, S, y, x); the unit ones (21, pixels, S) are handled apart


def compare_pair(b, a, lines, truth_ok=False):
    """b, a: (21, S, y, x) before and after."""
    S = b.shape[1]
    nb, na = n_seasons(b), n_seasons(a)
    same = nb == na
    lines.append(f"- pixels with the same number of seasons: {same.sum()} / {same.size} ({100 * same.mean():.2f}%); "
                 f"mean seasons before {nb.mean():.3f}, after {na.mean():.3f}")
    # the pixels of the same number of seasons, slot by slot (annual slots are the years; the others are in time order)
    bf = b.reshape(N_METRICS, S, -1)
    af = a.reshape(N_METRICS, S, -1)
    sel = same
    lines += ["", "| metric | cells with a value in both | NaN agreement | abs diff median / p90 / p99 (days) |",
              "|---|---|---|---|"]
    names = PHENOLOGY_METRICS
    for m in range(N_METRICS):
        x, y = bf[m][:, sel], af[m][:, sel]
        fb, fa = np.isfinite(x), np.isfinite(y)
        both = fb & fa
        agree = (fb == fa).mean() if fb.size else float("nan")
        d = np.abs(x[both] - y[both])
        unit = "" if m < 19 else " (no unit)" if m == 19 else " (value units)"
        lines.append(f"| {names[m]}{unit} | {both.sum()} | {100 * agree:.2f}% | {_q(d)} |")
    lines.append("")
    for m, nm in ((19, "R2"), (20, "RMSE")):
        x, y = bf[m][:, sel], af[m][:, sel]
        lines.append(f"- {nm}: median before {np.nanmedian(x):.4g}, after {np.nanmedian(y):.4g}; "
                     f"mean before {np.nanmean(x):.4g}, after {np.nanmean(y):.4g}")
    return same


def truth_error(res, truth, lines, label):
    """TRS5 rise and fall against the true 50% crossings (day of year), pixels of one season a year, annual output
    (slot = year; the first year is compared)."""
    n1 = truth.shape[0] // 2
    out = []
    for nm, col in (("TRS5.sos", 0), ("TRS5.eos", 1)):
        v = res[PHENOLOGY_METRICS.index(nm)][:, :].reshape(res.shape[1], -1)[:, :n1]  # (years, pixels)
        d = np.abs(v - truth[:n1, col][None, :])
        d = d[np.isfinite(d)]
        out.append(f"{nm} {_q(d)} (n={d.size})")
    lines.append(f"- {label}, error against the true 50% crossings, one-season pixels (median / p90 / p99, days): "
                 + "; ".join(out))


def compare(before, after, report):
    B, A = np.load(before), np.load(after)
    lines = ["# Phenology swap: before (phenofit-derived) vs after (clean-room phenology)", ""]
    # timings
    tkeys = sorted(k for k in B.files if k.startswith("time__") and k in A.files)
    lines += ["## Time (n_jobs=4)", "", "| case | before (s) | after (s) | ratio |", "|---|---|---|---|"]
    for k in tkeys:
        tb, ta = float(B[k]), float(A[k])
        lines.append(f"| {k[6:]} | {tb:.2f} | {ta:.2f} | {ta / tb:.2f} |")
    if tkeys:
        tb, ta = sum(float(B[k]) for k in tkeys), sum(float(A[k]) for k in tkeys)
        lines.append(f"| total | {tb:.2f} | {ta:.2f} | {ta / tb:.2f} |")
    lines.append("")
    # accuracy against the truth (cube)
    if "truth" in B.files:
        lines += ["## Error against the truth (cube, annual output)", ""]
        for c in CURVES:
            k = f"cube__{c}__annual"
            truth_error(B[k], B["truth"], lines, f"{c} before")
            truth_error(A[k], A["truth"], lines, f"{c} after")
        lines.append("")
    # unit cases
    lines += ["## Unit cases (core output (21, pixels, seasons))", ""]
    for k in sorted(x for x in B.files if x.startswith("unit__")):
        b, a = B[k], A[k]
        if b.ndim == 2:  # debug_split_seasons: (seasons, 3)
            lines.append(f"- {k}: before {b.astype(int).tolist()}, after {a.astype(int).tolist()}")
            continue
        if k == "unit__hants_accessor":
            lines.append(f"- {k}: seasons per pixel before {n_seasons(b).tolist()}, after {n_seasons(a).tolist()}; "
                         f"LOS before {np.nanmean(b[17]):.1f}, after {np.nanmean(a[17]):.1f}")
            continue
        nb, na = int(np.isfinite(b[18]).sum()), int(np.isfinite(a[18]).sum())
        def first(x, m):
            v = x[m, 0]
            return "/".join("nan" if not np.isfinite(t) else f"{t:.3g}" for t in v)
        lines.append(f"- {k}: seasons before {nb}, after {na}; TRS5.sos before {first(b, 2)}, after {first(a, 2)}; "
                     f"TRS5.eos {first(b, 3)} vs {first(a, 3)}; POP {first(b, 18)} vs {first(a, 18)}; "
                     f"R2 {first(b, 19)} vs {first(a, 19)}; RMSE {first(b, 20)} vs {first(a, 20)}")
    lines.append("")
    for k in sorted(x for x in B.files if x.startswith(("cube__", "real__"))):
        if k not in A.files:
            continue
        lines += [f"## {k}", ""]
        compare_pair(B[k], A[k], lines)
        lines.append("")
    Path(report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.report or "phenology_swap_report.md")
    else:
        np.savez_compressed(a.out, **collect())
        print("saved", a.out)


if __name__ == "__main__":
    main()
