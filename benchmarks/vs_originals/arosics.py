"""zeit.coregister against AROSICS' global COREG (1.14.0, Scheffler et al. 2017, Apache-2.0) on pairs of GeoTIFFs written
here (AROSICS runs in its own conda environment, ``arosics-ref``, through ``arosics_driver.py``; both read the same
files):

* synthetic scenes (value noise) with known shifts, whole and fractional pixels, with and without a no-data margin,
  odd sizes, a small image whose window AROSICS cuts down, a diagonal no-data edge, a featureless scene;
* a natural image (scikit-image's ``camera``), a real content with no satellite scene to use (the repository has
  none), moved by a known subpixel shift.

For each pair: the shifts, the integer part, the reliability, the structural similarity before and after, the
matching window and the FFT size of both, the largest difference of the two aligned images, and the times
(AROSICS: ``COREG(...)`` + ``calculate_spatial_shifts()``; Zeit: ``zeit.coregister`` on the same files, and the core's
estimate alone).

    python benchmarks/vs_originals/arosics.py        # AROSICS_ENV=...\\envs\\arosics-ref if it is elsewhere

and ``zeit.coregister_local`` against AROSICS' ``COREG_LOCAL`` (tie points grid, the filters, the warp by the GCPs),
on synthetic pairs whose target is the reference moved by a smooth field of shifts (with patches that are outliers, a
no-data margin and diagonal edge, odd sizes, a fine grid) and on the natural image warped by a field. The tie point
table of each is compared point by point (same grid positions, shifts, reliability, SSIM, the flags of each filter)
and the warped images; the times are AROSICS' ``COREG_LOCAL(...)`` + ``calculate_spatial_shifts()`` (1 and 4 CPUs)
against the core's tie points. AROSICS takes minutes, so it is a run of its own for each CPU count:

    python benchmarks/vs_originals/arosics.py local run 1
    python benchmarks/vs_originals/arosics.py local run 4
    python benchmarks/vs_originals/arosics.py local compare
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import zeit  # first: it sets up the DLL search path that rasterio needs on Windows
import rasterio  # noqa: E402
import rioxarray  # noqa: E402,F401
from common import HERE, best_of, no_power_throttling, record, workdir
from zeit import _core
from zeit._coreg import _prepare

ENV = Path(os.environ.get("AROSICS_ENV", r"C:\Users\eduar\miniforge3\envs\arosics-ref"))
RES = 30.0
X0, Y0 = 500000.0, 4600000.0


class Scene:
    """Value noise of 5 octaves, continuous (a scene at fractional coordinates is exact)."""

    def __init__(self, seed, size):
        rng = np.random.default_rng(seed)
        self.lat = [rng.uniform(-1, 1, (size // (2 << o) + 8, size // (2 << o) + 8)) for o in range(5)]

    def image(self, ny, nx, move=(0.0, 0.0)):
        r, c = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return self.at(c + 0.5 - move[0], r + 0.5 - move[1])

    def at(self, x, y):
        """The scene at pixel coordinates (x, y) (arrays)."""
        v = 0.0
        for o, lat in enumerate(self.lat):
            period = float(2 << o)
            u, w = x / period, y / period
            iu, iw = np.floor(u).astype(int), np.floor(w).astype(int)
            fu, fw = u - iu, w - iw
            su, sw = fu * fu * (3 - 2 * fu), fw * fw * (3 - 2 * fw)
            top = lat[iw, iu] * (1 - su) + lat[iw, iu + 1] * su
            bot = lat[iw + 1, iu] * (1 - su) + lat[iw + 1, iu + 1] * su
            v = v + (top * (1 - sw) + bot * sw) * np.sqrt(period)
        return 3000.0 + 300.0 * v


def write_tif(path, arr, nodata, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:32633", transform=rasterio.Affine(RES, 0, X0, 0, -RES, Y0), nodata=nodata) as d:
        d.write(arr.astype(dtype), 1)


def synthetic_pairs(folder):
    """[(name, spec, expected shift or None)]: the files written."""
    pairs = []

    def add(name, ny, nx, shift, nodata=-9999.0, dtype="float32", margin=None, diagonal=False, flat=False, ws=256,
            seed=3, ref_margin=None):
        sc = Scene(seed, max(ny, nx) + 64)
        ref = sc.image(ny, nx)
        tgt = sc.image(ny, nx, move=(-shift[0], -shift[1]))
        if flat:
            ref = np.full_like(ref, 1000.0)
            tgt = np.full_like(tgt, 1000.0)
        if dtype == "uint16":
            ref, tgt = ref + 6000, tgt + 6000
        valid = np.ones((ny, nx), bool)
        if margin:
            valid[:, :margin[0]] = False
            valid[ny - margin[1]:, :] = False
        if diagonal:
            r, c = np.mgrid[0:ny, 0:nx]
            valid &= (r + 0.6 * c) > 0.3 * ny
        for a in (ref, tgt):
            a[~valid] = nodata
        r = valid if ref_margin is None else valid.copy()
        write_tif(folder / f"{name}_ref.tif", ref, nodata, dtype)
        write_tif(folder / f"{name}_tgt.tif", tgt, nodata, dtype)
        pairs.append({"name": name, "ref": str(folder / f"{name}_ref.tif"), "tgt": str(folder / f"{name}_tgt.tif"),
                      "nodata": nodata, "ws": ws, "expected": list(shift), "size": f"{nx}x{ny}"})

    add("synth_int", 1024, 1024, (3, -2))
    add("synth_sub", 1024, 1024, (0.4, 1.7))
    add("synth_neg", 1024, 1024, (-4.25, 3.1), seed=5)
    add("synth_zero", 1024, 1024, (0, 0), seed=6)
    add("synth_margin_u16", 1024, 1024, (2.5, -1.2), nodata=0, dtype="uint16", margin=(300, 200), seed=7)
    add("synth_odd", 1001, 777, (1.6, 0.7), seed=8)
    add("synth_small", 200, 200, (1.3, -0.8), seed=9)
    add("synth_ws128", 700, 500, (-2.2, 2.2), ws=128, seed=10)
    add("synth_diag_nodata", 1024, 1024, (2.0, 1.0), diagonal=True, seed=11)
    add("synth_far", 1024, 1024, (9, 0), seed=12)
    add("synth_flat", 1024, 1024, (0, 0), flat=True)
    add("synth_4096", 4096, 4096, (1.5, -2.5), seed=13)
    return pairs


def run_arosics(pairs, cpus, repeat=3):
    spec = {"pairs": [dict(p, cpus=cpus, aligned=str(Path(p["ref"]).with_name(p["name"] + f"_aros{cpus}.npy")))
                      for p in pairs], "repeat": repeat}
    w = workdir("arosics")
    (w / "spec.json").write_text(json.dumps(spec))
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS=str(cpus), OPENBLAS_NUM_THREADS=str(cpus),
               MKL_NUM_THREADS=str(cpus), GDAL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    e = str(ENV)
    env["PATH"] = os.pathsep.join([e, e + r"\Library\mingw-w64\bin", e + r"\Library\usr\bin", e + r"\Library\bin",
                                   e + r"\Scripts", env["PATH"]])
    p = subprocess.Popen([str(ENV / "python.exe"), str(HERE / "arosics_driver.py"), "run", str(w / "spec.json"),
                          str(w / "result.json")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True)
    no_power_throttling(p.pid)
    out, err = p.communicate()
    if p.returncode:
        raise RuntimeError(f"arosics_driver failed:\n{err[-3000:]}")
    return {r["name"]: r for r in json.loads((w / "result.json").read_text())}, spec


def natural_pair(folder):
    e = str(ENV)
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env["PATH"] = os.pathsep.join([e, e + r"\Library\mingw-w64\bin", e + r"\Library\usr\bin", e + r"\Library\bin",
                                   e + r"\Scripts", env["PATH"]])
    subprocess.run([str(ENV / "python.exe"), str(HERE / "arosics_driver.py"), "natural", str(folder)], env=env,
                   check=True, stdin=subprocess.DEVNULL)
    s = json.loads((folder / "camera_shift.json").read_text())
    return {"name": "natural_camera", "ref": str(folder / "camera_ref.tif"), "tgt": str(folder / "camera_tgt.tif"),
            "nodata": 0, "ws": 256, "expected": [s["dx"], s["dy"]], "size": "512x512"}


def zeit_side(p, n_jobs):
    ref = rioxarray.open_rasterio(p["ref"]).squeeze("band", drop=True)
    tgt = rioxarray.open_rasterio(p["tgt"]).squeeze("band", drop=True)
    aligned, shifts = zeit.coregister(tgt, ref, window_size=p["ws"], n_jobs=n_jobs)
    t = best_of(lambda: zeit.coregister(tgt, ref, window_size=p["ws"], n_jobs=n_jobs))
    rd, rok, _ = _prepare(ref.values, p["nodata"])
    td, tok, fill = _prepare(tgt.values, p["nodata"])
    wsz = p["ws"]
    integer = None
    if np.issubdtype(tgt.dtype, np.integer):
        info = np.iinfo(tgt.dtype)
        integer = (float(info.min), float(info.max))
    t_core = best_of(lambda: _core.coreg.coregister(rd, rok, td, tok, wsz, None, 5.0, 5, integer, float(p["nodata"]), float(fill)))
    return shifts.iloc[0], aligned, t, t_core


def compare(pairs, aros, spec):
    rows = []
    for p in pairs:
        a = aros[p["name"]]
        z, aligned, t_z, t_core = zeit_side(p, 1)
        row = {"name": p["name"], "size": p["size"], "expected": p["expected"], "arosics_ok": a["success"],
               "zeit_ok": bool(z.success), "t_arosics": a["t_estimate"], "t_zeit": t_z, "t_core": t_core}
        zw = [int(z.window_x0), int(z.window_y0), int(z.window_x1), int(z.window_y1)]
        row["window_zeit"], row["window_arosics"] = zw, a.get("window")
        row["fft_zeit"], row["fft_arosics"] = int(z.fft_size), a.get("fft_win_size_YX")
        if a["success"] and z.success:
            row["dx"] = float(z.x_shift_px) - a["x_shift_px"]
            row["dy"] = float(z.y_shift_px) - a["y_shift_px"]
            row["shift_zeit"] = [float(z.x_shift_px), float(z.y_shift_px)]
            row["shift_arosics"] = [a["x_shift_px"], a["y_shift_px"]]
            row["int_equal"] = (round(float(z.x_shift_px) - a["x_shift_px"]) == 0
                                and abs(float(z.x_shift_px) - a["x_shift_px"]) < 0.5)
            row["rel"] = [float(z.reliability), a["reliability"]]
            row["ssim_before"] = [float(z.ssim_before), a["ssim_before"]]
            row["ssim_after"] = [float(z.ssim_after), a["ssim_after"]]
            ap = Path(spec["pairs"][[q["name"] for q in spec["pairs"]].index(p["name"])]["aligned"])
            if ap.exists():
                other = np.load(ap).astype(np.float64)
                mine = aligned.values.astype(np.float64)
                if other.shape == mine.shape:
                    ok = np.isfinite(mine) & (mine != p["nodata"]) & (other != p["nodata"])
                    inner = np.zeros_like(ok)
                    inner[8:-8, 8:-8] = True
                    ok &= inner
                    row["aligned_max_diff"] = float(np.abs(mine[ok] - other[ok]).max()) if ok.any() else None
                    row["aligned_mean_diff"] = float(np.abs(mine[ok] - other[ok]).mean()) if ok.any() else None
                else:
                    row["aligned_shape"] = [list(mine.shape), list(other.shape)]
        else:
            row["reason_zeit"] = z.reason
            row["reason_arosics"] = a.get("error")
        rows.append(row)
    return rows


# --- COREG_LOCAL -------------------------------------------------------------------------------------------


def field_a(x, y):
    """A smooth field of shifts (pixels) with an affine part and a sinusoid, up to about 3 pixels."""
    return 1.0 + 0.0015 * x + 1.0 * np.sin(y / 200.0), -0.8 + 0.001 * y + 0.7 * np.sin(x / 230.0)


def field_b(x, y):
    """A gentler one, up to about 1.5 pixels (room for the shifts of a displaced patch below max_shift)."""
    return 0.5 + 0.0008 * x + 0.5 * np.sin(y / 180.0), -0.4 + 0.0006 * y + 0.4 * np.sin(x / 210.0)


def local_pairs(folder):
    """The synthetic pairs of the local comparison: a target that is the reference's scene at the positions a smooth
    field moves it from (the shift to find at p is the field at p)."""
    pairs = []

    def add(name, ny, nx, field, ws, grid_res, nodata=-9999.0, dtype="float32", margin=None, diagonal=False, seed=3,
            patches=False):
        sc = Scene(seed, max(ny, nx) + 128)
        r, c = np.mgrid[0:ny, 0:nx]
        ref = sc.at(c + 0.5, r + 0.5)
        dx, dy = field(c + 0.5, r + 0.5)
        tgt = sc.at(c + 0.5 + dx, r + 0.5 + dy)
        if patches:
            # a patch of another texture, a featureless one, and one moved 3 pixels more than the field (a good
            # match that does not fit the model)
            other = Scene(seed + 100, max(ny, nx) + 128)
            a, b = int(ny * 0.20), int(nx * 0.30)
            tgt[a:a + 300, b:b + 300] = other.at(c + 0.5, r + 0.5)[a:a + 300, b:b + 300]
            a, b = int(ny * 0.55), int(nx * 0.65)
            tgt[a:a + 300, b:b + 300] = 1500.0
            a, b = int(ny * 0.62), int(nx * 0.12)
            tgt[a:a + 300, b:b + 300] = sc.at(c + 0.5 + dx + 3.0, r + 0.5 + dy - 2.5)[a:a + 300, b:b + 300]
        if dtype == "uint16":
            ref, tgt = ref + 6000, tgt + 6000
        valid = np.ones((ny, nx), bool)
        if margin:
            valid[:, :margin[0]] = False
            valid[ny - margin[1]:, :] = False
        if diagonal:
            valid &= (r + 0.6 * c) > 0.3 * ny
        for arr in (ref, tgt):
            arr[~valid] = nodata
        write_tif(folder / f"{name}_ref.tif", ref, nodata, dtype)
        write_tif(folder / f"{name}_tgt.tif", tgt, nodata, dtype)
        pairs.append({"name": name, "ref": str(folder / f"{name}_ref.tif"), "tgt": str(folder / f"{name}_tgt.tif"),
                      "nodata": nodata, "ws": ws, "grid_res": grid_res, "size": f"{nx}x{ny}"})

    add("local_smooth", 1200, 1400, field_a, 256, 256)
    add("local_patches", 1200, 1400, field_b, 256, 200, patches=True, seed=4)
    add("local_margin_u16", 1024, 1100, field_a, 256, 200, nodata=0, dtype="uint16", margin=(300, 150),
        diagonal=True, seed=5)
    add("local_odd", 1001, 777, field_a, 128, 100, seed=6)
    add("local_ws128_fine", 700, 900, field_b, 128, 64, seed=7)
    add("local_2048", 2048, 2048, field_a, 256, 200, seed=8)
    return pairs


def natural_local_pair(folder):
    e = str(ENV)
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    env["PATH"] = os.pathsep.join([e, e + r"\Library\mingw-w64\bin", e + r"\Library\usr\bin", e + r"\Library\bin",
                                   e + r"\Scripts", env["PATH"]])
    subprocess.run([str(ENV / "python.exe"), str(HERE / "arosics_driver.py"), "natural_local", str(folder)], env=env,
                   check=True, stdin=subprocess.DEVNULL)
    return {"name": "natural_camera_local", "ref": str(folder / "camera_local_ref.tif"),
            "tgt": str(folder / "camera_local_tgt.tif"), "nodata": 0, "ws": 128, "grid_res": 64, "size": "512x512"}


def run_arosics_local(pairs, cpus):
    w = workdir("arosics")
    spec = {"pairs": [dict(p, cpus=cpus, aligned=str(w / f"{p['name']}_local{cpus}.npy"),
                           table=str(w / f"{p['name']}_local{cpus}.json")) for p in pairs], "repeat": 3}
    (w / f"local_spec{cpus}.json").write_text(json.dumps(spec))
    env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS=str(cpus), OPENBLAS_NUM_THREADS=str(cpus),
               MKL_NUM_THREADS=str(cpus), GDAL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    e = str(ENV)
    env["PATH"] = os.pathsep.join([e, e + r"\Library\mingw-w64\bin", e + r"\Library\usr\bin", e + r"\Library\bin",
                                   e + r"\Scripts", env["PATH"]])
    p = subprocess.Popen([str(ENV / "python.exe"), str(HERE / "arosics_driver.py"), "local",
                          str(w / f"local_spec{cpus}.json"), str(w / f"local_result{cpus}.json")], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    no_power_throttling(p.pid)
    out, err = p.communicate()
    if p.returncode:
        raise RuntimeError(f"arosics_driver failed:\n{err[-3000:]}")
    return spec


def zeit_local(p, n_jobs, repeat=2):
    """(aligned, tie points, seconds of zeit.coregister_local, seconds of the tie points alone) on the files."""
    from zeit._coreg import _grid_points

    ref = rioxarray.open_rasterio(p["ref"]).squeeze("band", drop=True)
    tgt = rioxarray.open_rasterio(p["tgt"]).squeeze("band", drop=True)
    # arosics_compat: AROSICS' misassigned RANSAC flags too, to compare the points one by one
    kw = dict(grid_res=p["grid_res"], window_size=p["ws"], n_jobs=n_jobs, arosics_compat=True)
    aligned, points = zeit.coregister_local(tgt, ref, **kw)
    t = best_of(lambda: zeit.coregister_local(tgt, ref, **kw), repeat)
    rd, rok, _ = _prepare(ref.values, p["nodata"])
    td, tok, fill = _prepare(tgt.values, p["nodata"])
    integer = None
    if np.issubdtype(tgt.dtype, np.integer):
        info = np.iinfo(tgt.dtype)
        integer = (float(info.min), float(info.max))
    ny, nx = td.shape
    gx, gy = np.meshgrid(_grid_points(nx, p["grid_res"]), _grid_points(ny, p["grid_res"]))
    pts = np.ascontiguousarray(np.column_stack([gx.ravel(), gy.ravel()]), dtype=np.float64)
    geo = (X0, RES, Y0, -RES)
    t_core = best_of(lambda: _core.coreg.tie_points(rd, rok, td, tok, pts, (0.0, 0.0), geo, p["ws"], 5.0, 5, integer,
                                                    float(p["nodata"]), float(fill), 3, 60.0, 10.0, 2.5, 15, 0,
                                                    True, n_jobs), repeat)
    return aligned, points, t, t_core


def zeit_local_fixed(p):
    """The tie points with each RANSAC flag on its own point (the default, not AROSICS' misassignment)."""
    ref = rioxarray.open_rasterio(p["ref"]).squeeze("band", drop=True)
    tgt = rioxarray.open_rasterio(p["tgt"]).squeeze("band", drop=True)
    return zeit.coregister_local(tgt, ref, grid_res=p["grid_res"], window_size=p["ws"], n_jobs=1)


def local_compare(pairs, a_res, spec, n_jobs=1):
    """One row per pair: the tie points and the warped image of zeit against AROSICS' (run with the CPUs of `spec`)."""
    import pandas as pd

    rows = []
    for p in pairs:
        a = a_res[p["name"]]
        sp = spec["pairs"][[q["name"] for q in spec["pairs"]].index(p["name"])]
        aligned, points, t_z, t_core = zeit_local(p, n_jobs)
        row = {"name": p["name"], "size": p["size"], "grid_res": p["grid_res"], "ws": p["ws"],
               "t_arosics": a["t_estimate"], "t_arosics_correct": a.get("t_correct"), "t_zeit": t_z,
               "t_core": t_core, "arosics_ok": a["success"], "n_gcps_arosics": a.get("n_gcps"),
               "n_used_zeit": int(points.used.sum()), "zeit_ok": bool(points.date_applied.all()) if len(points) else False}
        tab = pd.DataFrame(json.loads(Path(sp["table"]).read_text()))
        row["points_arosics"], row["points_zeit"] = len(tab), len(points)
        m = tab.merge(points, left_on="POINT_ID", right_on="point_id", how="outer", indicator=True)
        row["points_common"] = int((m["_merge"] == "both").sum())
        both = m[m["_merge"] == "both"]
        a_ok = both["LAST_ERR"].isna()
        z_ok = both["success"].astype(bool)
        row["success_mismatch"] = int((a_ok != z_ok).sum())
        row["matched"] = [int(a_ok.sum()), int(z_ok.sum())]
        c = both[a_ok & z_ok]
        row["n_compared"] = len(c)
        if len(c):
            def mx(a_col, z_col):
                return float(np.abs(c[a_col].astype(float) - c[z_col].astype(float)).max())

            row["d_shift_px"] = max(mx("X_SHIFT_PX", "x_shift_px"), mx("Y_SHIFT_PX", "y_shift_px"))
            row["d_shift_map"] = max(mx("X_SHIFT_M", "x_shift_map"), mx("Y_SHIFT_M", "y_shift_map"))
            row["d_abs_angle"] = [mx("ABS_SHIFT", "abs_shift"), mx("ANGLE", "angle")]
            row["d_reliability"] = mx("RELIABILITY", "reliability")
            row["d_ssim"] = [mx("SSIM_BEFORE", "ssim_before"), mx("SSIM_AFTER", "ssim_after")]
            row["win_equal"] = int(((c["X_WIN_SIZE"] == c["win_size_x"]) & (c["Y_WIN_SIZE"] == c["win_size_y"])).sum())
            row["ssim_improved_equal"] = int((c["SSIM_IMPROVED"].astype(bool) == c["ssim_improved"].astype(bool)).sum())
            flags = {}
            for name, z in (("L1_OUTLIER", "outlier_reliability"), ("L2_OUTLIER", "outlier_ssim"),
                            ("L3_OUTLIER", "outlier_ransac"), ("OUTLIER", "outlier")):
                av = c[name].map(lambda v: v is True) if name in c else np.zeros(len(c), bool)
                flags[name] = [int((av.values != c[z].astype(bool).values).sum()), int(av.sum()), int(c[z].sum())]
            row["flag_mismatch"] = flags  # name -> [points that differ, flagged by AROSICS, flagged by zeit]
            # the default (no quirk): where its RANSAC flags differ from AROSICS' misassigned ones, and whether the
            # points it flags are those RANSAC found (they are the same count; the planted outliers it catches)
            _, fixed = zeit_local_fixed(p)
            f = fixed.set_index("point_id").loc[c["point_id"].astype(int)]
            av = c["L3_OUTLIER"].map(lambda v: v is True).values
            row["ransac_fixed_vs_arosics"] = [int((av != f["outlier_ransac"].values).sum()), int(av.sum()),
                                              int(f["outlier_ransac"].sum())]
        ap = Path(sp["aligned"])
        if ap.exists():
            other = np.load(ap).astype(np.float64)
            mine = aligned.values.astype(np.float64)
            if other.shape == mine.shape:
                ok = np.isfinite(mine) & (mine != p["nodata"]) & (other != p["nodata"])
                inner = np.zeros_like(ok)
                inner[8:-8, 8:-8] = True
                ok &= inner
                diff = np.abs(mine[ok] - other[ok])
                row["warp_max_diff"] = float(diff.max()) if diff.size else None
                row["warp_mean_diff"] = float(diff.mean()) if diff.size else None
                row["warp_frac_gt1"] = float((diff > 1.0).mean()) if diff.size else None
                row["warp_valid_equal"] = float((((mine != p["nodata"]) & np.isfinite(mine)) ==
                                                 (other != p["nodata"]))[inner].mean())
            else:
                row["warp_shape"] = [list(mine.shape), list(other.shape)]
        rows.append(row)
    return rows


def main_local(argv):
    """``arosics.py local run 1`` / ``local run 4`` (AROSICS with that many CPUs, the pairs written first), then
    ``local compare`` (zeit on the same files, the report and the timings)."""
    no_power_throttling()
    folder = workdir("arosics")
    if argv[0] == "run":
        pairs = local_pairs(folder) + [natural_local_pair(folder)]
        (folder / "local_pairs.json").write_text(json.dumps(pairs))
        run_arosics_local(pairs, int(argv[1]))
        return
    pairs = json.loads((folder / "local_pairs.json").read_text())
    out = {}
    for cpus in (1, 4):
        spec = json.loads((folder / f"local_spec{cpus}.json").read_text())
        a_res = {r["name"]: r for r in json.loads((folder / f"local_result{cpus}.json").read_text())}
        out[cpus] = local_compare(pairs, a_res, spec, n_jobs=cpus)
    (folder / "local_report.json").write_text(json.dumps(out, indent=1))
    print(f"load: {os.getloadavg() if hasattr(os, 'getloadavg') else 'n/a'}")
    print(f"{'pair':<22}{'points A/Z':<12}{'ok A/Z':<10}{'dshift px':>10}{'dmap m':>10}{'drel':>9}{'dssim b/a':>16}"
          f"{'win=':>6}{'impr=':>7}{'GCPs A/Z':>10}")
    for r in out[1]:
        if "d_shift_px" in r:
            print(f"{r['name']:<22}{r['points_arosics']:>4}/{r['points_zeit']:<7}{r['matched'][0]:>4}/{r['matched'][1]:<5}"
                  f"{r['d_shift_px']:>10.1e}{r['d_shift_map']:>10.1e}{r['d_reliability']:>9.1e}"
                  f"{r['d_ssim'][0]:>8.0e}/{r['d_ssim'][1]:<7.0e}{r['win_equal']:>3}/{r['n_compared']:<3}"
                  f"{r['ssim_improved_equal']:>3}/{r['n_compared']:<3}{r['n_gcps_arosics']:>5}/{r['n_used_zeit']:<4}")
        else:
            print(f"{r['name']:<22}{r['points_arosics']:>4}/{r['points_zeit']:<7}  no point matched in both")
    print()
    print("flags (points that differ / flagged by AROSICS / flagged by zeit):")
    for r in out[1]:
        f = r.get("flag_mismatch")
        if f:
            print(f"{r['name']:<22}" + "  ".join(f"{k[:2] if k != 'OUTLIER' else 'OUT'} {v[0]}/{v[1]}/{v[2]}"
                                                  for k, v in f.items()) + f"   success mismatches {r['success_mismatch']}"
                  f"   RANSAC without the quirk: {r['ransac_fixed_vs_arosics'][0]} points differ")
    print()
    print(f"{'pair':<22}{'warp max':>10}{'warp mean':>11}{'>1':>8}{'valid=':>8}")
    for r in out[1]:
        if "warp_max_diff" in r:
            print(f"{r['name']:<22}{r['warp_max_diff']:>10.3g}{r['warp_mean_diff']:>11.3g}{r['warp_frac_gt1']:>8.4f}"
                  f"{r['warp_valid_equal']:>8.4f}")
    print()
    print(f"{'pair':<22}{'AROSICS 1':>11}{'AROSICS 4':>11}{'zeit 1':>10}{'zeit 4':>10}{'core 1':>10}{'core 4':>10}"
          f"{'A correct':>11}")
    for r1, r4 in zip(out[1], out[4]):
        print(f"{r1['name']:<22}{r1['t_arosics']:>10.2f}s{r4['t_arosics']:>10.2f}s{r1['t_zeit']:>9.3f}s{r4['t_zeit']:>9.3f}s"
              f"{r1['t_core']:>9.3f}s{r4['t_core']:>9.3f}s{(r1['t_arosics_correct'] or float('nan')):>10.2f}s")
    for r1, r4 in zip(out[1], out[4]):
        for r, cpus in ((r1, 1), (r4, 4)):
            record("coregister_local", "AROSICS 1.14.0 COREG_LOCAL", f"{r['size']} {r['name']}", r["t_arosics"],
                   r["t_core"], cpus, f"tie points only ({r['points_zeit']} points, grid_res={r['grid_res']}, "
                   f"ws={r['ws']}; AROSICS CPUs={cpus}, one run, COREG_LOCAL(...) + calculate_spatial_shifts(), from "
                   f"the files); zeit: the core's call on arrays, {cpus} threads; zeit.coregister_local with the "
                   f"warp: {r['t_zeit']:.3f} s, AROSICS' correct_shifts: {r['t_arosics_correct'] or float('nan'):.2f} s")


def main():
    no_power_throttling()
    folder = workdir("arosics")
    pairs = synthetic_pairs(folder) + [natural_pair(folder)]
    out = {"load": os.getloadavg() if hasattr(os, "getloadavg") else None}
    for cpus in (1, 4):
        aros, spec = run_arosics(pairs, cpus)
        out[f"cpus{cpus}"] = compare(pairs, aros, spec)
    rows = out["cpus1"]
    print(f"{'pair':<20}{'ok A/Z':<9}{'dx':>11}{'dy':>11}  {'zeit shift':<22}{'arosics shift':<22}{'rel Z/A':<15}"
          f"{'ssim after Z/A':<20}{'window equal':<13}{'aligned max':>12}")
    for r in rows:
        if "dx" in r:
            print(f"{r['name']:<20}{str(r['arosics_ok'])[0]}/{str(r['zeit_ok'])[0]:<7}{r['dx']:>11.2e}{r['dy']:>11.2e}  "
                  f"({r['shift_zeit'][0]:.5f}, {r['shift_zeit'][1]:.5f})  "
                  f"({r['shift_arosics'][0]:.5f}, {r['shift_arosics'][1]:.5f})  "
                  f"{r['rel'][0]:6.2f}/{r['rel'][1]:6.2f}  {r['ssim_after'][0]:.5f}/{r['ssim_after'][1]:.5f}  "
                  f"{str(r['window_zeit'] == r['window_arosics']):<13}{r.get('aligned_max_diff', float('nan')):>12.3g}")
        else:
            print(f"{r['name']:<20}{str(r['arosics_ok'])[0]}/{str(r['zeit_ok'])[0]:<7}  zeit: {r.get('reason_zeit')} | "
                  f"arosics: {r.get('reason_arosics')}")
    print()
    print(f"{'pair':<20}{'AROSICS 1 cpu':>14}{'AROSICS 4 cpus':>16}{'zeit coregister':>17}{'zeit core only':>16}")
    for r1, r4 in zip(out["cpus1"], out["cpus4"]):
        print(f"{r1['name']:<20}{r1['t_arosics']:>13.3f}s{r4['t_arosics']:>15.3f}s{r1['t_zeit']:>16.4f}s"
              f"{r1['t_core']:>15.4f}s")
    for r1, r4 in zip(out["cpus1"], out["cpus4"]):
        if r1["name"] in ("synth_sub", "synth_4096", "natural_camera"):
            # the shift estimate against the estimate: AROSICS' COREG(...) + calculate_spatial_shifts() (the
            # footprints, the overlap, the windows and the correlation, from the files) against the core's call on the
            # arrays; the whole coregister() of a date (the files read by rioxarray apart) moves the whole image too
            for r, cpus in ((r1, 1), (r4, 4)):
                record("coregister", "AROSICS 1.14.0 COREG", r["size"] + " " + r["name"], r["t_arosics"], r["t_core"],
                       cpus, f"estimate only (AROSICS CPUs={cpus}); zeit's core call on arrays, one thread (a date is "
                       f"one task); zeit.coregister of the date with the aligned image: {r1['t_zeit']:.3f} s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        main_local(sys.argv[2:])
    else:
        main()
