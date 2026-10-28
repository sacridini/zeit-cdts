"""Phenology with the Rust core on the real EVI raster of cdts_vs_originals (data/phenology/
EVI_Series_ID_218_2001_2025.tif: 29 x 22 = 638 pixels, 575 16-day dates, 2001-2025), as its
benchmark_phenology_real_timing_cdts.py (single core) and parallel_phenology.py (multi-core) timed cdts.

The same preparation (values below -1 masked, linear interpolation over time, 0 for what is left) and call: what the
cdts accessor ran, zeit._phenology.run_phenology_dask on the whole cube as one block (Beck curve, 25 seasons,
Whittaker lambda 10, min season length 7, annual output), wrapped in a DataArray and computed. Single core: one
n_jobs=1 run (the original protocol), plus the warm best of 3; multi-core: n_jobs=1 then n_jobs=8, one run each
(each with the raster loaded afresh, as parallel_phenology.py), plus the warm best of 3.
Agreement: the three curves (Beck, Elmore, Gu) in the long format of generate_cdts_phenology_reference.py, compared
with the C++ core's stored output (data/phenology/cdts_reference_long.csv) and, as compare_phenology_real.py, with
R phenofit's (phenofit_reference_long.csv).

    python benchmarks/vs_references/phenology.py
"""
import common  # first: thread variables

import time

import dask.array as da
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from zeit._core.phenology import CurveType
from zeit._phenology import run_phenology_dask
from zeit._series_api import PHENOLOGY_METRICS
from common import DATA, THREADS, pin, record, stored_row, timed, warm_best

RASTER = DATA / "phenology" / "EVI_Series_ID_218_2001_2025.tif"
CURVES = {"beck": CurveType.BECK, "elmore": CurveType.ELMORE, "gu": CurveType.GU}
REF_METRICS = PHENOLOGY_METRICS[:17]


def load():
    ds = rioxarray.open_rasterio(RASTER)
    cube = ds.where(ds >= -1).interpolate_na(dim="band", method="linear").fillna(0)
    dates = pd.date_range(start="2001-01-01", periods=cube.shape[0], freq="16D")
    doy = np.array([d.timetuple().tm_yday + (d.year - 2001) * 365 for d in dates], dtype=np.float64)
    return cube, doy


def run(cube, doy, n_jobs, curve=CurveType.BECK):
    out = run_phenology_dask(da.from_array(cube.data), dates=doy, curve_type=int(curve), max_seasons=25,
                             whittaker_lambda=10.0, apply_whittaker=True, min_season_length=7, min_amplitude=0.0,
                             min_pixel_amplitude=0.1, return_annual=True, base_year=2001, n_jobs=n_jobs)
    arr = xr.DataArray(out, dims=["metric", "year", "y", "x"],
                       coords={"metric": PHENOLOGY_METRICS, "year": np.arange(2001, 2026),
                               "y": cube.coords["y"], "x": cube.coords["x"]})
    return arr.compute(scheduler="threads", num_workers=1)


def fresh(n_jobs):
    """parallel_phenology.py's run(): the raster loaded and prepared, then the timed call."""
    cube, doy = load()
    t0 = time.perf_counter()
    run(cube, doy, n_jobs)
    return time.perf_counter() - t0


def long_format(cube, doy):
    n_y, n_x = cube.shape[1:]
    pid = (np.arange(n_y)[:, None] * n_x + np.arange(n_x)[None, :]) + 1
    rows = []
    for name, ct in CURVES.items():
        m = run(cube, doy, THREADS, ct)
        for metric in REF_METRICS:
            a = m.sel(metric=metric).values
            for yi in range(a.shape[0]):
                s = a[yi]
                ys, xs = np.where(~np.isnan(s) & (s > 0))
                rows += [(int(p), name, 2001 + yi, metric, float(v)) for p, v in zip(pid[ys, xs], s[ys, xs])]
    return pd.DataFrame(rows, columns=["pixel_id", "curve_type", "year", "metric", "value_doy"])


def agreement(cube, doy):
    z = long_format(cube, doy)
    keys = ["pixel_id", "curve_type", "year", "metric"]
    cpp = pd.read_csv(DATA / "phenology" / "cdts_reference_long.csv")
    j = z.merge(cpp, on=keys, how="outer", suffixes=("_rust", "_cpp"), indicator=True)
    both = j[j._merge == "both"]
    vs_cpp = {"rows_rust": int(len(z)), "rows_cpp": int(len(cpp)), "rows_only_rust": int((j._merge == "left_only").sum()),
              "rows_only_cpp": int((j._merge == "right_only").sum()),
              "values_identical_rate": float((both.value_doy_rust == both.value_doy_cpp).mean()),
              "max_abs_diff_days": float((both.value_doy_rust - both.value_doy_cpp).abs().max())}
    ph = pd.read_csv(DATA / "phenology" / "phenofit_reference_long.csv")
    ph.columns = [c.strip('"') for c in ph.columns]
    for c in ("curve_type", "metric"):
        ph[c] = ph[c].astype(str).str.strip('"')
    m = z.merge(ph, on=keys, suffixes=("_z", "_p"))
    diff = (m.value_doy_z - m.value_doy_p).abs()
    vs_phenofit = {}
    for ct in CURVES:
        d = diff[m.curve_type == ct]
        st = stored_row("phenology.csv", f"phenology_{ct}_vs_phenofit")
        vs_phenofit[ct] = {"n": int(len(d)), "MAE_days_all_metrics": round(float(d.mean()), 3),
                           "within_15_days_pct": round(float((d <= 15).mean() * 100), 2),
                           "stored_n": int(st["n_scenarios"]), "stored_MAE": float(st["primary_metric_value"]),
                           "stored_within_15": float(st["secondary_metric_value"])}
    sos = diff[(m.curve_type == "beck") & (m.metric == "TRS5.sos")]
    vs_phenofit["beck_TRS5.sos_MAE"] = round(float(sos.mean()), 3)
    return {"vs_cpp_output": vs_cpp, "vs_phenofit": vs_phenofit}


def main():
    cube, doy = load()
    n_px = cube.shape[1] * cube.shape[2]
    pin(1)
    t_single = timed(lambda: fresh(1), "1T")
    w1 = timed(lambda: warm_best(lambda: run(cube, doy, 1)), "1T warm")
    t_par1 = timed(lambda: fresh(1), "parallel 1T")
    pin(THREADS)
    t_par8 = timed(lambda: fresh(THREADS), "parallel 8T")
    w10 = timed(lambda: warm_best(lambda: run(cube, doy, THREADS)), "8T warm")
    pin(None)
    agree = agreement(cube, doy)

    st = stored_row("phenology.csv", "phenology_beck_vs_phenofit")
    pub, ref = float(st["cdts_time_ms_mean"]), float(st["reference_time_ms_mean"])
    ms = t_single / n_px * 1000
    record({"table": "single-core", "row": "Phenology", "order": 1,
            "workload": f"real EVI raster, {n_px} px x 575 dates, Beck, n_jobs=1", "unit": "ms/px",
            "published_cpp_1t": pub, "rust_1t": ms, "rust_1t_warm_best3": w1 / n_px * 1000,
            "ratio_rust_cpp_1t": ms / pub, "reference_1t": ref, "reference_tool": st["reference_tool"],
            "speedup_vs_reference_1t": ref / ms, "published_speedup": float(st["speedup_x"]), "agreement": agree,
            "notes": "published and reference: results/phenology.csv (beck). One run, as "
                     "benchmark_phenology_real_timing_cdts.py; warm best of 3 also given."})
    pr = {r["tool"]: r for r in common.read_csv_rows(common.ORIGINALS / "results" / "parallel_phenology.csv")}
    c, r = pr["cdts"], pr["R_phenofit_doParallel"]
    p1, pm = float(c["time_ms_single_thread"]), float(c["time_ms_parallel"])
    ref1, refm = float(r["time_ms_single_thread"]), float(r["time_ms_parallel"])
    t1, t8 = t_par1 * 1000, t_par8 * 1000
    record({"table": "multi-core", "row": "Phenology", "order": 4,
            "workload": f"real EVI raster, {n_px} px x 25 yrs (575 dates), Beck", "unit": "ms",
            "published_cpp_1t": p1, "published_cpp_multi": pm, "published_threads": int(c["n_workers_parallel"]),
            "rust_1t": t1, "rust_8t": t8, "rust_1t_warm_best3": w1 * 1000, "rust_8t_warm_best3": w10 * 1000,
            "ratio_rust_cpp_1t": t1 / p1, "ratio_rust8_cpp19": t8 / pm,
            "reference_1t": ref1, "reference_multi": refm, "reference_workers": int(r["n_workers_parallel"]),
            "reference_tool": "R phenofit, doParallel (19 workers)",
            "speedup_vs_reference_1t": ref1 / t1, "speedup_vs_reference_multi": refm / t8,
            "notes": "published and reference: results/parallel_phenology.csv. One run each, n_jobs=1 first, the "
                     "raster loaded afresh each time (parallel_phenology.py); warm best of 3 also given. Published: "
                     "n_jobs=-1 = 19 threads; here 8."})


if __name__ == "__main__":
    common.start()
    main()
