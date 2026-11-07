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
        x, y = c + 0.5 - move[0], r + 0.5 - move[1]
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
    main()
