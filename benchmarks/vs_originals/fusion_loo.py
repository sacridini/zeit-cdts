"""Leave-one-date-out accuracy of the fusion on real Landsat and MODIS (the npz of ``fusion_real.py``): every Landsat
date is predicted from the others (``exclude_dates=[d]``) and compared with the observation of that date.

Methods: ``zeit.fuse_starfm`` and ``zeit.fuse_estarfm`` (settings as below), ``zeit.starfm`` (FORCE's variant, the
stand-in for the ARS program, with the coarse series on the fine grid), baselines: the nearest Landsat date, the linear
interpolation in time of the two bracketing Landsat dates, and the coarse image resampled. With ``--zhu N`` Zhu's
ESTARFM Python program (black box, local only, see ``fusion.py``) is run on the first N dates it can predict.

    python benchmarks/vs_originals/fusion_loo.py SITE [--zhu N] [--threads 4] [--radius 750]
"""
import argparse
import time

import numpy as np
import pandas as pd
import xarray as xr

import zeit
from common import no_power_throttling, workdir
from fusion import agreement, run_zhu, write_envi

ROLES = ["green", "red", "nir"]


def load(site):
    z = np.load(workdir("fusion-real") / f"{site}.npz", allow_pickle=False)
    days = pd.to_datetime([str(d) for d in z["days"]])
    x0, y0, x1, y1 = z["fine_bounds"]
    n = z["fine"].shape[-1]
    fine = xr.DataArray(z["fine"], dims=("time", "band", "y", "x"),
                        coords={"time": days, "band": ["green", "red", "nir08"],
                                "y": y1 - 30.0 * (np.arange(n) + 0.5), "x": x0 + 30.0 * (np.arange(n) + 0.5)}
                        ).rio.write_crs(f"EPSG:{int(z['fine_epsg'])}")
    a, b, c, d_, e, f = (float(v) for v in z["coarse_transform"])
    h, w = z["coarse"].shape[-2:]
    coarse = xr.DataArray(z["coarse"], dims=("time", "band", "y", "x"),
                          coords={"time": days, "band": ["sur_refl_b04", "sur_refl_b01", "sur_refl_b02"],
                                  "y": f + e * (np.arange(h) + 0.5), "x": c + a * (np.arange(w) + 0.5)}
                          ).rio.write_crs(str(z["coarse_crs"]))
    return fine, coarse


def fill(cube):
    """A copy of ``(band, y, x)`` with each missing pixel taken from its nearest valid neighbour."""
    from scipy.ndimage import distance_transform_edt

    out = np.array(cube, dtype=np.float32)
    for b in range(out.shape[0]):
        bad = ~np.isfinite(out[b])
        if bad.any():
            idx = distance_transform_edt(bad, return_distances=False, return_indices=True)
            out[b] = out[b][tuple(idx)]
    return out


def stats(pred, truth):
    """Per band and NDVI: rmse, bias, r, aad over the pixels both have; ``(4, 4)``."""
    rows = []
    nd = lambda x: (x[2] - x[1]) / (x[2] + x[1])
    arrays = [(pred[i], truth[i]) for i in range(3)] + [(nd(pred), nd(truth))]
    for p, t in arrays:
        p, t = p.ravel().astype(np.float64), t.ravel().astype(np.float64)
        ok = np.isfinite(p) & np.isfinite(t)
        if ok.sum() < 10:
            rows.append([np.nan] * 4)
            continue
        p, t = p[ok], t[ok]
        rows.append([np.sqrt(np.mean((p - t) ** 2)), np.mean(p - t), np.corrcoef(p, t)[0, 1], np.mean(np.abs(p - t))])
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--radius", type=float, default=750.0)
    ap.add_argument("--zhu", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0, help="also leave out this many dates on each side of the predicted one")
    ap.add_argument("--fast", action="store_true", help="without the slow per-cell variant")
    ap.add_argument("--only", nargs="*", help="run only these methods (names as in the table)")
    a = ap.parse_args()
    no_power_throttling()
    fine, coarse = load(a.site)
    days = [str(t)[:10] for t in fine.time.values]
    # the coarse series on the fine grid, bilinear: what the programs that need one grid are given
    ref = fine.isel(time=0, drop=True)
    on_grid = xr.concat([coarse.isel(time=k).rio.reproject_match(ref, resampling=1, nodata=np.nan)
                         for k in range(coarse.sizes["time"])], dim="time").assign_coords(time=coarse.time)
    def starfm(**kw):
        return lambda d, ex: zeit.fuse_starfm(fine, coarse, dates=[d], include_fine_dates=False, exclude_dates=ex,
                                         radius=a.radius, n_jobs=a.threads, **kw)

    def estarfm(**kw):
        return lambda d, ex: zeit.fuse_estarfm(fine, coarse, dates=[d], include_fine_dates=False, exclude_dates=ex,
                                          radius=a.radius, n_jobs=a.threads, **kw)

    methods = {
        "starfm": starfm(),
        "starfm m=4": starfm(classes=4.0),
        "starfm either": starfm(filter="either"),
        "starfm log": starfm(weighting="log"),
        "estarfm": estarfm(),
        "estarfm V=1": estarfm(v_min_change=1.0),
        "estarfm vmax 2": estarfm(v_max=2.0),
        "estarfm m=8": estarfm(classes=8.0),
        "estarfm cell": estarfm(v_scope="cell"),
    }
    if a.fast:
        methods.pop("estarfm cell")
    if a.only:
        methods = {k: v for k, v in methods.items() if k in a.only}
    table, times = {}, {}
    zhu_done = 0
    for i, d in enumerate(days):
        truth = fine.isel(time=i).values
        out = [j for j in range(len(days)) if abs(j - i) <= a.skip]        # the date and its `skip` neighbours on each side
        others = [j for j in range(len(days)) if j not in out]
        ex = [days[j] for j in out]
        before = [j for j in others if j < i]
        after = [j for j in others if j > i]
        base = {}
        near = min(others, key=lambda j: abs((pd.Timestamp(days[j]) - pd.Timestamp(d)).days))
        base["nearest date"] = fine.isel(time=near).values
        if before and after:
            j0, j1 = before[-1], after[0]
            w = (pd.Timestamp(d) - pd.Timestamp(days[j0])).days / (pd.Timestamp(days[j1]) - pd.Timestamp(days[j0])).days
            base["linear in time"] = (1 - w) * fine.isel(time=j0).values + w * fine.isel(time=j1).values
        base["coarse"] = on_grid.isel(time=i).values
        res = {k: v for k, v in base.items()}
        for name, fn in methods.items():
            t0 = time.perf_counter()
            try:
                res[name] = fn(d, ex).values[0]
            except ValueError as e:
                print(f"  {d} {name}: {e}")
                continue
            times.setdefault(name, []).append(time.perf_counter() - t0)
        ok_frac = lambda arr: float(np.isfinite(arr.values).mean())
        if a.zhu and before and after and zhu_done < a.zhu and all(
                ok_frac(x) >= 0.99 for x in (fine.isel(time=before[-1]), fine.isel(time=after[0]), on_grid.isel(time=before[-1]),
                                             on_grid.isel(time=after[0]), on_grid.isel(time=i))):
            # Zhu's program is given images with the few missing pixels (under 1%) filled from their nearest valid
            # neighbour: how it treats a fill value in the data was not tested
            zhu_done += 1
            j0, j1 = before[-1], after[0]
            zd = workdir("fusion-zhu")
            dn = lambda arr: np.round(fill(arr) * 10000).astype(np.int16)
            for nme, arr in (("f1", fine.isel(time=j0)), ("f2", fine.isel(time=j1)), ("m1", on_grid.isel(time=j0)),
                             ("m2", on_grid.isel(time=j1)), ("mp", on_grid.isel(time=i))):
                write_envi(zd / nme, dn(arr.values))
            theirs, t_z = run_zhu(zd, "f1", "m1", "f2", "m2", "mp", int(round(a.radius / 30.0)), 4.0)
            theirs = np.where(theirs <= -9999, np.nan, theirs / 10000.0)
            res["Zhu ESTARFM"] = theirs
            times.setdefault("Zhu ESTARFM", []).append(t_z)
            own = methods["estarfm"](d, ex).values[0]
            agreement(own, theirs, int(round(a.radius / 30.0)), f"{d}: zeit.fuse_estarfm vs Zhu on the same inputs")
        gap = min(abs((pd.Timestamp(days[j]) - pd.Timestamp(d)).days) for j in others)
        for name, pred in res.items():
            table.setdefault(name, []).append((d, gap, stats(pred, truth)))
        print(d, f"(nearest pair {gap} d)", {k: f"{v[-1][2][1, 0]:.4f}" for k, v in table.items()})
    skip = f", {a.skip} neighbouring date(s) on each side left out too" if a.skip else ""
    print(f"\n{a.site}: leave-one-date-out, {len(days)} dates, radius {a.radius:g} m{skip} (mean over dates)")
    print(f"  {'method':<16}" + "".join(f"{n:>14}" for n in ("green rmse", "red rmse", "nir rmse", "NDVI rmse", "NDVI bias", "NDVI r", "s/date", "dates")))
    for name, rows in table.items():
        s = np.nanmean(np.stack([r[2] for r in rows]), axis=0)
        t = np.mean(times[name]) if name in times else float("nan")
        print(f"  {name:<16}{s[0, 0]:>14.4f}{s[1, 0]:>14.4f}{s[2, 0]:>14.4f}{s[3, 0]:>14.4f}{s[3, 1]:>14.4f}{s[3, 2]:>14.3f}{t:>14.1f}{len(rows):>8d}")
    pd.to_pickle(table, workdir("fusion-real") / f"{a.site}_loo.pkl")


if __name__ == "__main__":
    main()
