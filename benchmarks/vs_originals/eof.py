"""zeit.eof against xeofs 3.0.4 (MIT) and eofs 2.0.0 (GPL-3.0): the same anomalies of a cube, decomposed by each.

Zeit's anomalies (the monthly climatology, its ``_core.anomaly``) are what the references receive, as the matrix Zeit
decomposes itself (the pixels and dates it keeps, each column centred over its valid values, the gaps 0), so that
what is compared is the decomposition: the explained variance, the maps and the PCs, North's errors (eofs'
``northTest``), and the varimax rotation (xeofs' ``EOFRotator``). The data are a synthetic monthly cube with three
known modes (projected and geographic, the latter with cos(latitude) weights) and the monthly means of a real 16-day
EVI series (EVI_Series_ID_EU_150-X026-Y027_2001_2025.tif: 300 months, 677 x 680 pixels, EPSG:3035, many gaps).

    python benchmarks/vs_originals/eof.py [--sizes 120x316x316 240x707x707] [--no-real]

(``TxNYxNX``: dates and the synthetic cube's grid.) The references run with 8 BLAS threads, Zeit with 1 and with 8.
"""
import argparse
import time
import warnings

import numpy as np

import eof_data
import zeit
from common import best_of, record, threads_label

NM = 10
JOBS = (1, 8)


def time_once(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def kept_matrix(cube, ds, jobs=8):
    """The matrix Zeit decomposes: Zeit's monthly anomalies of the kept dates and pixels, each column centred over
    its valid values, the gaps 0 (``(T, N)``), with the cube's index of the kept pixels."""
    t = cube.sizes["time"]
    values = np.ascontiguousarray(cube.values.astype(np.float64).reshape(t, -1))
    days = zeit._eof.days_of(cube.time.values)
    an = zeit._core.anomaly.anomalies(values, days, method="month", n_jobs=jobs)
    pix = np.isfinite(ds.eof.values[0].reshape(-1))
    dates = ~ds.pc_filled.values.astype(bool)
    a = an[dates][:, pix]
    a = a - np.nanmean(a, axis=0)
    return np.nan_to_num(a, nan=0.0), pix, dates, an


def cosine(a, b):
    """|cos| of each row of a with the same row of b."""
    num = np.abs((a * b).sum(axis=1))
    return num / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def run_eofs(a, weights=None):
    import eofs.standard

    s = eofs.standard.Eof(a, weights=weights)
    e = s.eofs(neofs=NM, eofscaling=2)
    p = s.pcs(npcs=NM, pcscaling=1)
    return dict(eof=e, pc=p.T, ratio=s.varianceFraction(NM), eigen=s.eigenvalues(NM),
                north=s.northTest(NM, vfscaled=False))


def xeofs_input(a, grid=None):
    """``a`` (T, N) as a DataArray; with ``grid=(ny, nx)``, a (time, lat, lon) one from 70 to 10 N (for use_coslat)."""
    import xarray as xr

    t = np.arange(a.shape[0])
    if grid is None:
        return xr.DataArray(a, dims=("time", "pixel"), coords={"time": t, "pixel": np.arange(a.shape[1])})
    return xr.DataArray(a.reshape(a.shape[0], *grid), dims=("time", "lat", "lon"),
                        coords={"time": t, "lat": np.linspace(70, 10, grid[0]), "lon": np.linspace(0, 60, grid[1])})


def run_xeofs(a, solver="auto", grid=None, **kw):
    import xeofs

    warnings.simplefilter("ignore", FutureWarning)
    if solver == "full":            # numpy's SVD with its default full_matrices=True would make a (N, N) matrix
        kw["solver_kwargs"] = {"full_matrices": False}
    m = xeofs.single.EOF(n_modes=NM, center=True, solver=solver, **kw)
    m.fit(xeofs_input(a, grid), dim="time")
    comp = m.components()
    if grid is not None:            # xeofs gives the grid back with its coordinates sorted: put the cube's order back
        comp = comp.transpose("mode", "lat", "lon").sel(lat=np.linspace(70, 10, grid[0]), lon=np.linspace(0, 60, grid[1]))
    return m, dict(eof=comp.values.reshape(NM, -1), pc=m.scores().transpose("mode", "time").values,
                   ratio=m.explained_variance_ratio().values, eigen=m.explained_variance().values)


def leading(ds):
    """How many of the first modes North's test separates from their neighbours (at least 1)."""
    sep = ds.separable.values.astype(bool)
    return max(int(np.argmin(sep)) if not sep.all() else len(sep), 1)


def agreement(ds, kept, ref, name, scaled):
    """The numbers of one reference against Zeit's result: the explained variance, the cosines of the maps and PCs,
    the maps' and PCs' values (``scaled``: the reference is in Zeit's ``units`` scaling) and North's errors, over
    the 10 modes and over the ``lead`` first, those North's test separates (the others, noise, are degenerate: their
    directions are not defined to better than the gaps of their eigenvalues allow)."""
    lead = leading(ds)
    z_eof = ds.eof.values.reshape(NM, -1)[:, kept]
    z_pc = ds.pc.values[:NM][:, ~ds.pc_filled.values.astype(bool)]
    out = dict(name=name, lead=lead)
    out["ratio_diff"] = float(np.abs(ds.variance.values - ref["ratio"]).max())
    rel = np.abs(ds.eigenvalue.values - ref["eigen"]) / ds.eigenvalue.values
    out["eigen_rel"], out["eigen_rel_lead"] = float(rel.max()), float(rel[:lead].max())
    out["cos_eof"], out["cos_pc"] = cosine(z_eof, ref["eof"]), cosine(z_pc, ref["pc"])
    if scaled:
        sign = np.sign((z_eof * ref["eof"]).sum(axis=1))[:, None]
        out["eof_rel"] = float(np.abs(z_eof - sign * ref["eof"])[:lead].max() / np.abs(z_eof[:lead]).max())
        out["pc_abs"] = float(np.abs(z_pc - sign * ref["pc"])[:lead].max())
    if "north" in ref:
        rel = np.abs(ds.north_error.values - ref["north"]) / ds.north_error.values
        out["north_rel"], out["north_rel_lead"] = float(rel.max()), float(rel[:lead].max())
    return out


def show(r):
    c, p, lead = r["cos_eof"], r["cos_pc"], r["lead"]
    print(f"  {r['name']:<28} ratio diff {r['ratio_diff']:.1e}   eigenvalue rel {r['eigen_rel']:.1e} "
          f"(first {lead}: {r['eigen_rel_lead']:.1e})   1-min|cos| EOF {1 - c.min():.1e} PC {1 - p.min():.1e} "
          f"(first {lead}: {1 - c[:lead].min():.1e}, {1 - p[:lead].min():.1e})"
          + (f"   values EOF {r['eof_rel']:.1e} PC {r['pc_abs']:.1e}" if "eof_rel" in r else "")
          + (f"   North {r['north_rel']:.1e} (first {lead}: {r['north_rel_lead']:.1e})" if "north_rel" in r else ""))


def check_anomalies(cube):
    """Zeit's monthly anomalies against xarray's groupby (in float64: a float32 cube's mean is rounded to float32)."""
    cube = cube.astype("float64")
    t = cube.sizes["time"]
    values = np.ascontiguousarray(cube.values.astype(np.float64).reshape(t, -1))
    an = zeit._core.anomaly.anomalies(values, zeit._eof.days_of(cube.time.values), method="month", n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # pixels with no value at all
        ref = eof_data.xarray_anomalies(cube).values.reshape(t, -1)
    both = np.isfinite(an) & np.isfinite(ref)
    assert (np.isfinite(an) == np.isfinite(ref)).all()
    err = float(np.abs(an - ref)[both].max())
    print(f"  anomalies, Zeit's monthly climatology against xarray's groupby: max diff {err:.1e}")
    assert err < 1e-9, err
    return err


def accuracy_synthetic(months=240, ny=100, nx=100):
    """Known modes, projected CRS (equal weights): Zeit exact and randomized against both references."""
    cube, (u, v) = eof_data.synthetic(months, ny, nx, seed=1)
    print(f"synthetic {months} dates x {ny}x{nx} pixels (EPSG:3035, equal weights)")
    check_anomalies(cube)
    ds = zeit.eof(cube, n_modes=NM, weights="none", solver="exact", n_jobs=2, rotate=3)
    a, pix, dates, _ = kept_matrix(cube, ds)
    # the truth: the modes put in
    cos_true_pc = [float(np.abs(np.corrcoef(ds.pc.values[k], u[k])[0, 1])) for k in range(3)]
    cos_true_eof = [float(np.abs(np.corrcoef(ds.eof.values[k].ravel(), v[k].ravel())[0, 1])) for k in range(3)]
    print(f"  Zeit against the modes put in: |cos| of the PCs {np.round(cos_true_pc, 4)}, of the EOFs "
          f"{np.round(cos_true_eof, 4)}; first 3 modes' variance {np.round(ds.variance.values[:3], 4)}")
    assert min(cos_true_pc) > 0.99 and min(cos_true_eof) > 0.99
    assert ds.separable.values[:3].all()
    res = {}
    res["eofs"] = agreement(ds, pix, run_eofs(a), "eofs (exact)", True)
    m, rx = run_xeofs(a, solver="full")
    res["xeofs"] = agreement(ds, pix, rx, "xeofs (solver='full')", False)
    for r in res.values():
        show(r)
    assert res["eofs"]["ratio_diff"] < 1e-8 and res["xeofs"]["ratio_diff"] < 1e-8
    assert res["eofs"]["cos_eof"].min() > 1 - 1e-6 and res["eofs"]["cos_pc"].min() > 1 - 1e-6
    assert res["eofs"]["eof_rel"] < 1e-6 and res["eofs"]["north_rel"] < 1e-8
    assert res["xeofs"]["cos_eof"].min() > 1 - 1e-6
    _, rx2 = run_xeofs(a, solver="auto")
    r2 = agreement(ds, pix, rx2, "xeofs (solver='auto')", False)
    show(r2)
    rnd = zeit.eof(cube, n_modes=NM, weights="none", solver="randomized", n_jobs=2)
    rr = agreement(rnd, pix, run_eofs(a), "Zeit randomized vs eofs", True)
    show(rr)
    assert rr["ratio_diff"] < 1e-4, rr["ratio_diff"]
    assert rr["cos_eof"][:rr["lead"]].min() > 1 - 1e-6 and rr["cos_pc"][:rr["lead"]].min() > 1 - 1e-6
    assert r2["ratio_diff"] < 1e-4 and r2["cos_eof"][:r2["lead"]].min() > 1 - 1e-4
    rot_check(ds, m, "synthetic", ny, nx)
    return res


def rot_check(ds, model, label, ny, nx, kept=None):
    """The varimax rotation of the first 3 modes: Zeit's against xeofs' EOFRotator (both with Kaiser's row
    normalization; xeofs' scores carry the variance, Zeit's have unit variance: a convention)."""
    import xeofs

    mr = ds.eof_rotated.sizes["mode_rotated"]
    rot = xeofs.single.EOFRotator(n_modes=mr, power=1)
    rot.fit(model)
    pix = np.isfinite(ds.eof.values[0].reshape(-1)) if kept is None else kept
    zr = ds.eof_rotated.values.reshape(mr, -1)[:, pix]
    xr_ = rot.components().values.reshape(mr, -1)
    cos = cosine(zr, xr_)
    vz = ds.variance_rotated.values
    vx = rot.explained_variance_ratio().values
    zp = ds.pc_rotated.values[:, ~ds.pc_filled.values.astype(bool)]
    cp = cosine(zp, rot.scores().transpose("mode", "time").values)
    d = float(np.abs(vz - vx).max())
    print(f"  varimax of {mr} modes, Zeit against xeofs' EOFRotator ({label}): rotated variance {np.round(vz, 5)} "
          f"vs {np.round(vx, 5)} (max diff {d:.1e}); 1-min cos maps {1 - cos.min():.1e}, PCs {1 - cp.min():.1e}")
    assert d < 1e-6 and cos.min() > 1 - 1e-6 and cp.min() > 1 - 1e-6
    return d


def accuracy_varimax(months=240, ny=60, nx=90):
    """Two patches mixed by their correlated time series: the EOFs are their sum and difference, the varimax rotation
    of 2 modes gives them back; Zeit's rotation against the truth and against xeofs' EOFRotator."""
    cube, p = eof_data.two_patches(months, ny, nx)
    print(f"two patches, {months} dates x {ny}x{nx} pixels (PC correlation 0.5): Kaiser-normalized varimax of 2 modes")
    ds = zeit.eof(cube, n_modes=NM, weights="none", solver="exact", n_jobs=2, rotate=2)
    a, pix, dates, _ = kept_matrix(cube, ds)
    cos = np.abs(np.corrcoef(ds.eof_rotated.values.reshape(2, -1), p.reshape(2, -1))[:2, 2:])
    unrot = np.abs(np.corrcoef(ds.eof.values[:2].reshape(2, -1), p.reshape(2, -1))[:2, 2:])
    best = max(cos[0, 0] * cos[1, 1], cos[0, 1] * cos[1, 0])
    print(f"  |corr| of the first two EOFs with the patches {np.round(unrot, 3).tolist()}, of the rotated "
          f"{np.round(cos, 3).tolist()}; rotated variance {np.round(ds.variance_rotated.values, 4)} of "
          f"{np.round(ds.variance.values[:2], 4)}")
    assert unrot.max() < 0.9 and best > 0.95 ** 2      # orthogonal factors cannot be the correlated patches' exactly
    m, _ = run_xeofs(a, solver="full")
    rot_check(ds, m, "two patches", ny, nx)


def accuracy_geographic(months=120, ny=60, nx=80):
    """A geographic grid (10 to 70 degrees north), cos(latitude) weights on all three sides."""
    cube, _ = eof_data.synthetic(months, ny, nx, geographic=True, seed=2)
    print(f"synthetic {months} dates x {ny}x{nx} pixels (EPSG:4326, 10-70 N, cos(latitude) weights)")
    ds = zeit.eof(cube, n_modes=NM, weights="coslat", solver="exact", n_jobs=2)
    a, pix, dates, _ = kept_matrix(cube, ds)
    coslat = np.cos(np.deg2rad(cube.y.values))[:, None] * np.ones((1, nx))
    w = coslat.reshape(-1)[pix]
    eofs_ref = run_eofs(a.reshape(months, ny, nx), weights=np.sqrt(coslat))
    eofs_ref["eof"] = eofs_ref["eof"].reshape(NM, -1)
    # eofs' EOFs are in the data's units with the weights removed, as Zeit's "units"
    z_eof = ds.eof.values.reshape(NM, -1)
    out_e = agreement_flat(ds, z_eof, eofs_ref, "eofs (weights=sqrt(coslat))", w, scale=float(coslat.mean()))
    _, rx = run_xeofs(a, solver="full", grid=(ny, nx), use_coslat=True)
    out_x = agreement_flat(ds, z_eof, rx, "xeofs (use_coslat=True)", w)
    for r in (out_e, out_x):
        show(r)
    assert out_e["ratio_diff"] < 1e-8 and out_x["ratio_diff"] < 1e-8
    assert out_e["cos_eof"].min() > 1 - 1e-6 and out_x["cos_eof"].min() > 1 - 1e-6
    assert out_e["eof_rel"] < 1e-6 and out_e["eigen_rel"] < 1e-8
    return out_e, out_x


def agreement_flat(ds, z_eof, ref, name, w, scale=None):
    """``agreement`` for the weighted case. Zeit's ``units`` maps have the weights removed (data units per standard
    deviation of the PC); eofs and xeofs return the maps of the weighted data, so the cosine is taken with Zeit's
    maps weighted (``sqrt(w)``, w the area weights of mean 1). eofs' weights are sqrt(cos(latitude)) without
    normalization: its eigenvalues and maps are Zeit's times ``scale`` (the mean of cos(latitude)) and its square
    root, which is checked when ``scale`` is given."""
    sw = np.sqrt(w / w.mean())[None, :]
    out = dict(name=name, lead=leading(ds))
    out["ratio_diff"] = float(np.abs(ds.variance.values - ref["ratio"]).max())
    out["cos_eof"] = cosine(z_eof * sw, ref["eof"])
    out["cos_pc"] = cosine(ds.pc.values, ref["pc"])
    if scale is not None:
        q = ref["eigen"] / ds.eigenvalue.values / scale
        out["eigen_rel"] = out["eigen_rel_lead"] = float(np.abs(q - 1).max())
        sign = np.sign((z_eof * sw * ref["eof"]).sum(axis=1))[:, None]
        out["eof_rel"] = float(np.abs(z_eof * sw * np.sqrt(scale) - sign * ref["eof"]).max() / np.abs(ref["eof"]).max())
        out["pc_abs"] = float(np.abs(ds.pc.values - sign * ref["pc"]).max())
    else:
        out["eigen_rel"] = out["eigen_rel_lead"] = float("nan")
    return out


def timing(label, cube, size):
    """Zeit's eof (the whole call: anomalies, weights, solver, maps) with 1 and 8 threads against the references' fit
    and extraction of the 10 modes, on Zeit's anomalies (the references' time does not count the anomalies): eofs
    (the full SVD), xeofs with solver='full' (the same, exact) and xeofs' default (solver='auto': randomized)."""
    ds = zeit.eof(cube, n_modes=NM, weights="none", solver="exact", n_jobs=8)
    a, pix, dates, _ = kept_matrix(cube, ds)
    print(f"{label}: {a.shape[0]} dates x {a.shape[1]} pixels kept of {cube.sizes['time']} x "
          f"{cube.sizes['y'] * cube.sizes['x']}")
    small = a.size < 3e7
    runs = {"eofs 2.0.0 (full SVD)": lambda: run_eofs(a),
            "xeofs 3.0.4 (solver='full')": lambda: run_xeofs(a, solver="full")[1],
            "xeofs 3.0.4 (solver='auto', randomized)": lambda: run_xeofs(a, solver="auto")[1]}
    times, refs = {}, {}
    for name, fn in runs.items():           # the references once, the best of 3 on the small sizes
        t, refs[name] = time_once(fn)
        if small:
            t = min([t] + [time_once(fn)[0] for _ in range(2)])
        times[name] = t
        print(f"  {name}: {t:.2f} s")
    pairs = {"exact": ("eofs 2.0.0 (full SVD)", "xeofs 3.0.4 (solver='full')"),
             "randomized": ("xeofs 3.0.4 (solver='auto', randomized)", "eofs 2.0.0 (full SVD)")}
    for solver, originals in pairs.items():
        for jobs in JOBS:
            res = {}
            t_z = best_of(lambda: res.update(r=zeit.eof(cube, n_modes=NM, weights="none", solver=solver,
                                                        n_jobs=jobs)), 3 if small else 1)
            ag = agreement(res["r"], pix, refs["eofs 2.0.0 (full SVD)"], "eofs", True)
            tol = 1e-8 if solver == "exact" else 2e-3
            assert ag["ratio_diff"] < tol, (solver, ag["ratio_diff"])
            note = (f"{a.shape[0]} x {a.shape[1]} px kept; explained variance (10 modes) max diff against eofs "
                    f"{ag['ratio_diff']:.1e}; Zeit's whole call (anomalies, filtering, maps) against the "
                    f"references' decomposition of the anomalies; the references with 8 BLAS threads")
            for original in originals:
                record(f"eof ({solver})", original, size, times[original], t_z, threads_label(jobs), note)


def accuracy_real(cube):
    """The real EVI cube: Zeit (exact, randomized) against eofs and xeofs on what Zeit decomposes."""
    print(f"real EVI: {cube.shape}, {float(np.isfinite(cube.values).mean()):.1%} of the months valid")
    window = cube.isel(y=slice(250, 450), x=slice(250, 450))
    check_anomalies(window.where(window.notnull().all("time")))      # the pixels with no gap
    ds = zeit.eof(cube, n_modes=NM, weights="none", solver="exact", n_jobs=8, rotate=3)
    a, pix, dates, _ = kept_matrix(cube, ds)
    print(f"  Zeit keeps {int(pix.sum())} of {pix.size} pixels and {int(dates.sum())} of {dates.size} dates; "
          f"variance of the 5 first modes {np.round(ds.variance.values[:5], 4)}; separable {ds.separable.values[:6]}")
    r1 = agreement(ds, pix, run_eofs(a), "eofs (exact)", True)
    m, rx = run_xeofs(a, solver="full")
    r2 = agreement(ds, pix, rx, "xeofs (solver='full')", False)
    rnd = zeit.eof(cube, n_modes=NM, weights="none", solver="randomized", power_iterations=2, n_jobs=8)
    r3 = agreement(rnd, pix, run_eofs(a), "Zeit randomized vs eofs", True)
    for r in (r1, r2, r3):
        show(r)
    # the randomized sketch (Halko et al.: 10 more columns; 2 power passes here, 4 by default) is not exact on a spectrum that decays as
    # slowly as this real cube's: its error by mode, and with more power passes
    err = {2: np.abs(rnd.variance.values - ds.variance.values)}
    for q in (4, 8):
        rq = zeit.eof(cube, n_modes=NM, weights="none", solver="randomized", power_iterations=q, n_jobs=8)
        err[q] = np.abs(rq.variance.values - ds.variance.values)
    for q, v in err.items():
        print(f"  randomized, {q} power passes: |variance - exact| by mode " + " ".join(f"{x:.0e}" for x in v))
    assert err[8].max() < err[4].max() < err[2].max()
    assert r1["ratio_diff"] < 1e-8 and r2["ratio_diff"] < 1e-8 and r3["ratio_diff"] < 2e-3
    assert r3["eigen_rel_lead"] < 1e-5 and r3["cos_eof"][0] > 1 - 1e-6 and r3["cos_pc"][0] > 1 - 1e-6
    sep = ds.separable.values.astype(bool)
    assert r1["cos_eof"][sep].min() > 1 - 1e-6 and r2["cos_eof"][sep].min() > 1 - 1e-6
    assert r1["north_rel"] < 1e-8
    rot_check(ds, m, "real EVI", cube.sizes["y"], cube.sizes["x"], pix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=["120x316x316", "240x707x707"])
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-accuracy", action="store_true")
    args = ap.parse_args()
    print("zeit from", zeit.__file__)
    if not args.no_accuracy:
        accuracy_synthetic()
        accuracy_varimax()
        accuracy_geographic()
    if not args.no_real:
        cube = eof_data.real_monthly()
        if not args.no_accuracy:
            accuracy_real(cube)
        if not args.no_timing:
            timing("real EVI", cube, f"{cube.sizes['time']} months x {cube.sizes['y']}x{cube.sizes['x']} px (EVI)")
    if not args.no_timing:
        for s in args.sizes:
            t, ny, nx = (int(v) for v in s.split("x"))
            cube, _ = eof_data.synthetic(t, ny, nx, seed=3)
            timing(f"synthetic {s}", cube, f"{t} months x {ny}x{nx} px (synthetic)")


if __name__ == "__main__":
    main()
