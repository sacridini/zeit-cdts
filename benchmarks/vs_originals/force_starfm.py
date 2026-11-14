"""zeit.starfm against FORCE 3.11.00's own starfm() (resmerge-ll.c), on a synthetic multi-date scene.

FORCE's function is copied verbatim into a harness (force_resmerge/make_starfm_harness.sh; only its hard-coded
two pairs, nclass 40 and uncertainty 0.002 are replaced by globals, so that the general path can be compared) and
run in WSL once per prediction date. The scene is a 200 x 200 reflectance field of classes with texture; the coarse
image of each date is the fine one averaged over 4 x 4 blocks and blurred, plus a temporal change; some pixels have
no data (NaN in a pair or in a target, which are FORCE's off pixels). zeit.starfm must equal FORCE's output in
every value (float32), at several radii, numbers of pairs and parameters.

    wsl -e bash -lc 'bash /mnt/c/.../force_resmerge/make_starfm_harness.sh'
    $env:PYTHONPATH = <repo>; python benchmarks/vs_originals/force_starfm.py [--size 200] [--time]
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent


def _wsl(path):
    p = Path(path).resolve()
    return f"/mnt/{p.drive[0].lower()}" + p.as_posix()[2:]


def make_scene(size, npair, ntarget, seed=5):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cls = (np.sin(xx / 17.0) + np.cos(yy / 23.0) > 0).astype(float) + 0.5 * ((xx // 40 + yy // 40) % 2)
    def fine_at(phase):
        return (0.08 + 0.10 * cls + 0.03 * np.sin(xx / 9.0 + phase) + 0.004 * rng.standard_normal((size, size)))
    def coarse_of(f, change):
        b = 4
        m = f[:size - size % b, :size - size % b].reshape(size // b, b, size // b, b).mean(axis=(1, 3))
        c = np.repeat(np.repeat(m, b, 0), b, 1)
        c = np.pad(c, ((0, size - c.shape[0]), (0, size - c.shape[1])), mode="edge")
        k = np.array([1, 4, 6, 4, 1], float) / 16
        for ax in (0, 1):
            c = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 2, mode="edge"), k, "valid"), ax, c)
        return c + change + 0.002 * rng.standard_normal(c.shape)
    fines = [fine_at(0.7 * k) for k in range(npair)]
    pair_coarse = [coarse_of(f, 0.0) for f in fines]
    targets = [coarse_of(fine_at(0.7 * npair + k), 0.01 * (k + 1) + 0.02 * np.sin(xx / 30.0)) for k in range(ntarget)]
    fine = np.stack(fines).astype(np.float32)
    coarse = np.stack(pair_coarse + targets).astype(np.float32)
    # off pixels: a patch, scattered ones in a pair's fine image and in one coarse target
    fine[0, 20:30, 40:55] = np.nan
    fine[-1, rng.integers(0, size, 60), rng.integers(0, size, 60)] = np.nan
    coarse[npair, rng.integers(0, size, 80), rng.integers(0, size, 80)] = np.nan
    coarse[0, 100:104, 100:160] = np.nan
    return fine, coarse


def run_force(fine, coarse, radius, nclass, unc_c, unc_f, out_dir):
    npair = fine.shape[0]
    nt = coarse.shape[0] - npair
    size = fine.shape[1]
    off = np.zeros((nt, size, size), np.uint8)
    for t in range(nt):
        o = np.isnan(coarse[npair + t]) | np.isnan(fine).any(axis=0) | np.isnan(coarse[:npair]).any(axis=0)
        off[t] = o
    clean = lambda a: np.nan_to_num(a, nan=-7.0)   # the values at off pixels are never read by FORCE's code
    inp, outp = Path(out_dir) / "in.bin", Path(out_dir) / "out.bin"
    with open(inp, "wb") as f:
        np.array([size, size, npair, nt], np.int32).tofile(f)
        clean(fine).astype("<f4").tofile(f)
        clean(coarse).astype("<f4").tofile(f)
        off.tofile(f)
    exe = "$HOME/force-ref-work/starfm/starfm_harness"
    cmd = f"OMP_NUM_THREADS=1 {exe} {_wsl(inp)} {_wsl(outp)} {radius} {nclass} {unc_c} {unc_f}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"harness failed: {r.stderr}")
    return np.fromfile(outp, "<f4").reshape(nt, size, size), off.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--time", action="store_true", help="time the calls (the machine must be quiet)")
    args = ap.parse_args()
    import zeit
    print(f"zeit from {zeit.__file__}")
    cases = [  # npair, ntarget, radius, nclass, unc_coarse, unc_fine
        (2, 3, 1, 40, 0.002, 0.002),
        (2, 3, 5, 40, 0.002, 0.002),
        (2, 3, 12, 40, 0.002, 0.002),
        (1, 2, 5, 40, 0.002, 0.002),
        (3, 2, 5, 40, 0.002, 0.002),
        (2, 2, 5, 20, 0.01, 0.004),
    ]
    total = bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for npair, nt, r, nclass, uc, uf in cases:
            fine, coarse = make_scene(args.size, npair, nt)
            ref, off = run_force(fine, coarse, r, nclass, uc, uf, tmp)
            times = np.arange(npair + nt).astype("timedelta64[D]") + np.datetime64("2020-01-01")
            fx = xr.DataArray(fine, dims=("time", "y", "x"), coords={"time": times[:npair]})
            cx = xr.DataArray(coarse, dims=("time", "y", "x"), coords={"time": times})
            t0 = time.perf_counter()
            out = zeit.starfm(fx, cx, radius=r, nclass=nclass, unc_coarse=uc, unc_fine=uf, scale=1, n_jobs=1)
            dt = time.perf_counter() - t0
            z = out.values
            assert z.shape == ref.shape and z.dtype == np.float32
            # FORCE writes 0 at the off pixels, Zeit NaN
            same_off = bool((np.isnan(z) == off).all() and (ref[off] == 0).all())
            on = ~off
            differ = int((z[on].view(np.uint32) != ref[on].view(np.uint32)).sum())
            maxd = float(np.abs(z[on] - ref[on]).max())
            total += int(on.sum())
            bad += differ + (0 if same_off else 1)
            print(f"pairs {npair} targets {nt} radius {r:2d} nclass {nclass} unc {uc}/{uf}: {int(on.sum())} values "
                  f"compared, {differ} differ (max |d| {maxd:.3g}), {int(off.sum())} off pixels "
                  f"{'agree' if same_off else 'DISAGREE'}" + (f", zeit {dt * 1e3:.0f} ms" if args.time else ""))
    print(f"total {total} values, {bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
