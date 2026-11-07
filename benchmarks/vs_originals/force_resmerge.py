"""``zeit.sentinel2_resmerge`` against FORCE 3.11.00's resolution merge (``resmerge-ll.c``) on the same synthetic scene.

FORCE runs the merge only inside ``force-l2ps``, so there is no FORCE program to feed an ARD tile to: the C harness in
``force_resmerge/`` (``make_harness.sh``, WSL) is FORCE's ``resolution_merge_1`` (regression) and ``resolution_merge_2``
(ImproPhe-like), ``improphe_resmerge``, ``rescale_weight_resmerge`` and ``distance_kernel`` copied verbatim behind shims
that replace the brick and QAI accessors with plain arrays, linked statically to the real GSL 2.7 of the conda env
``force-ref`` (libgsl.a and the reference libgslcblas.a: conda-forge's ``libgslcblas.so.0`` is OpenBLAS). It runs with one
OpenMP thread, where FORCE's regression is deterministic (row-major, each prediction written back before the next pixel)::

    python benchmarks/vs_originals/force_resmerge.py [--size 500] [--reps 3]

The scene is made here: a 10 m Sentinel-2 scene (int16, reflectance times 10000) of land covers with noise in each band,
the 20 m bands as 2 x 2 blocks (the nearest neighbour copies FORCE's GDAL read makes), with no-data pixels (a stripe and
scattered pixels), clouds, shadows, a patch of constant spectra (a rank deficient fit) and a border of zeros. Zeit gets it as
float32 with NaN at the "off" pixels and ``clear`` from the cloud and shadow masks. ``force=True`` is compared with FORCE's
int16 output per band over every pixel that is not off (the predicted ones and those left alone): the share of equal values,
of values within 1, and the largest difference. ``force=False`` (no truncation; no feedback for the regression) is compared
with ``force=True`` as the difference it makes. Times: FORCE's harness (gcc -O2, 1 and 4 OpenMP threads; with 4 threads
its regression is racy, only the time is of use) and Zeit (release build, Windows), best of ``--reps``.
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_resmerge")
BANDS = ["green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
PRED = BANDS[3:]


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def make_scene(size, seed=7):
    """The 10 m scene: int16 bands (9, ny, nx) in the order BANDS, and the masks off, cloud, shadow (bool)."""
    rng = np.random.default_rng(seed)
    ny = nx = size
    # land covers: smooth random classes (blocky blobs), each a reflectance spectrum (x 10000) in the nine bands
    coarse = rng.integers(0, 6, size=(ny // 40 + 2, nx // 40 + 2))
    cover = np.kron(coarse, np.ones((40, 40), int))[:ny, :nx]
    spectra = np.array([
        [350, 300, 150, 600, 1500, 2000, 2200, 400, 200],        # water
        [600, 500, 4200, 1200, 2800, 3600, 3800, 1900, 900],     # vegetation
        [900, 1100, 1500, 1400, 1700, 1900, 2000, 2800, 2600],   # bare soil
        [1500, 1700, 2000, 1900, 2100, 2200, 2300, 3300, 3100],  # urban
        [500, 450, 3300, 900, 2200, 2800, 3000, 1500, 700],      # crop
        [2000, 2100, 2300, 2200, 2400, 2500, 2500, 2000, 1800],  # bright
    ], float)
    base = spectra[cover]                                            # (ny, nx, 9)
    illum = 1 + 0.15 * np.sin(np.arange(ny)[:, None] / 23.0) * np.cos(np.arange(nx)[None, :] / 31.0)
    x = base * illum[..., None] * (1 + 0.05 * rng.standard_normal((ny, nx, 9)))
    # the 20 m bands are 2 x 2 blocks of one value
    for b in range(3, 9):
        blocks = x[::2, ::2, b]
        x[:, :, b] = np.kron(blocks, np.ones((2, 2)))[:ny, :nx]
    # a patch of constant spectra (rank deficient fits) and a patch of zeros (saturated dark)
    x[60:90, 60:90, :] = spectra[1]
    x[150:170, 100:120, :] = 0.0
    x = np.clip(np.round(x), 0, 10000).astype(np.int16)
    off = np.zeros((ny, nx), bool)
    off[:, 200:203] = True                                           # a stripe
    off[rng.random((ny, nx)) < 0.01] = True                          # scattered pixels
    off[0:3, :] |= rng.random((3, nx)) < 0.3
    cloud = np.zeros((ny, nx), bool)
    yy, xx = np.mgrid[:ny, :nx]
    for cy, cx, r in ((ny // 4, nx // 3, 25), (ny * 3 // 4, nx * 2 // 3, 18)):
        cloud |= (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
    shadow = np.roll(cloud, (7, 11), axis=(0, 1)) & ~cloud
    shadow &= rng.random((ny, nx)) < 0.9
    x[off] = -9999                                                   # what FORCE would hold there: never read
    return x.transpose(2, 0, 1), off, cloud, shadow


def write_scene(path, x, off, cloud, shadow):
    ny, nx = off.shape
    with open(path, "wb") as f:
        f.write(np.array([nx, ny], np.int32).tobytes())
        f.write(np.ascontiguousarray(x, np.int16).tobytes())
        for m in (off, cloud, shadow):
            f.write(m.astype(np.uint8).tobytes())


def run_force(scene, method, threads, out):
    """FORCE's harness: the time (ms) of the merge; the output is written to `out`."""
    exe = "$HOME/force-ref-work/resmerge/resmerge_harness"
    cmd = f"OMP_NUM_THREADS={threads} {exe} {_wsl(scene)} {_wsl(out)} {method}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed: {r.stderr}")
    return float(r.stdout.strip().splitlines()[-1])


def zeit_cube(x, off):
    import zeit  # noqa: F401

    ny, nx = off.shape
    a = x.astype(np.float32)
    a[:, off] = np.nan
    names = ["green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
    return xr.DataArray(a, dims=("band", "y", "x"), coords={"band": names, "y": np.arange(ny) * -10.0, "x": np.arange(nx) * 10.0})


def best(fn, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return min(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    no_power_throttling()
    import zeit

    x, off, cloud, shadow = make_scene(args.size)
    scene = DATA / "scene.bin"
    write_scene(scene, x, off, cloud, shadow)
    npix = off.size
    predicted = ~(off | cloud | shadow)
    print(f"scene {args.size} x {args.size}: {npix} pixels, {off.sum()} off, {cloud.sum()} cloud, {shadow.sum()} shadow, "
          f"{predicted.sum()} predicted")
    cube = zeit_cube(x, off)
    clear = xr.DataArray(~(cloud | shadow), dims=("y", "x"))
    for method, name in ((1, "regression"), (2, "improphe")):
        out = DATA / f"force_{method}.bin"
        run_force(scene, method, 1, out)
        ref = np.fromfile(out, np.int16).reshape(6, args.size, args.size)
        forced = zeit.sentinel2_resmerge(cube, method=name, clear=clear, force=True, n_jobs=1)
        free = zeit.sentinel2_resmerge(cube, method=name, clear=clear, force=False, n_jobs=1)
        print(f"\n== {name} ==  FORCE (int16) vs zeit force=True, over the {(~off).sum()} pixels with data")
        print(f"{'band':10s} {'identical':>10s} {'within 1':>9s} {'max |d|':>8s}   predicted pixels: identical")
        keep = ~off
        for k, band in enumerate(PRED):
            z = forced.sel(band=band).values[keep]
            f = ref[k][keep].astype(np.float32)
            d = np.abs(z - f)
            zp = forced.sel(band=band).values[predicted]
            fp = ref[k][predicted].astype(np.float32)
            print(f"{band:10s} {100 * (d == 0).mean():9.4f}% {100 * (d <= 1).mean():8.4f}% {d.max():8.1f}   "
                  f"{100 * (zp == fp).mean():9.4f}%")
        print(f"zeit force=False vs force=True (predicted pixels): per band share within 1 / max |d| / mean |d|")
        for k, band in enumerate(PRED):
            a = free.sel(band=band).values[predicted]
            b = forced.sel(band=band).values[predicted]
            d = np.abs(a - b)
            print(f"{band:10s} {100 * (d <= 1).mean():8.3f}% {d.max():9.2f} {d.mean():9.4f}")
        # the input is not touched where a pixel is not predicted
        for k, band in enumerate(PRED):
            z = forced.sel(band=band).values
            assert np.array_equal(z[~predicted & ~off], x[3 + k][~predicted & ~off].astype(np.float32))

    print("\ntimes (ms, best of %d)" % args.reps)
    print(f"{'':24s} {'1 thread':>10s} {'4 threads':>10s}")
    for method, name in ((1, "regression"), (2, "improphe")):
        out = DATA / "tmp.bin"
        f1 = min(run_force(scene, method, 1, out) for _ in range(args.reps))
        f4 = min(run_force(scene, method, 4, out) for _ in range(args.reps))
        z = {}
        for force in (False, True):
            for n in (1, 4):
                z[(force, n)] = best(lambda: zeit.sentinel2_resmerge(cube, method=name, clear=clear, force=force, n_jobs=n),
                                     args.reps)
        print(f"FORCE {name:18s} {f1:10.0f} {f4:10.0f}")
        print(f"zeit  {name:18s} {z[(False, 1)]:10.0f} {z[(False, 4)]:10.0f}   (force=False)")
        print(f"zeit  {name:18s} {z[(True, 1)]:10.0f} {z[(True, 4)]:10.0f}   (force=True)")


if __name__ == "__main__":
    main()
