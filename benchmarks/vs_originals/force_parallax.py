"""``zeit.parallax_clouds``, ``zeit.cloud_displacement_index`` and ``zeit.buffer_mask`` against FORCE 3.11.00's
``cloud_parallax`` and ``buffer_`` (``cloud-ll.c``, ``imagefuns-cl.c``) on the same synthetic scene.

FORCE runs the parallax test only inside ``force-l2ps``, so there is no FORCE program to feed an ARD tile to: the C
harness in ``force_parallax/`` (``make_harness.sh``, WSL) is FORCE's ``cloud_parallax``, ``gauss_kernel``,
``covar_recurrence``, ``variance``, the queue and ``buffer_`` copied verbatim behind shims that replace the brick and QAI
accessors with plain arrays (one line is added to ``cloud_parallax``: it hands the CDI to the harness)::

    python benchmarks/vs_originals/force_parallax.py [--size 512] [--big 2048] [--reps 3]

The scene is made here, at 10 m (int16, reflectance times 10000): a land surface whose red edge 3 and NIR (B8A) vary
smoothly and independently; clouds, which are textured objects seen at the same place in B07 and B8A and 6 pixels to
the side in B08 (the parallax); bright built-up patches (textured, seen in the same place in all bands, with ratios
that change as much in both); a patch of bright surface with a 1 pixel offset and a high variability probability (the
second way into the cloud class, CDI < 0 and variability > 0.8); a variability field; potential cloud pixels around all
of those and scattered; saturated pixels; bright pixels (the gate); and pixels that are off (a stripe, scattered ones,
a block). Zeit gets the same arrays (float32 with NaN at the off pixels, the masks as boolean arrays, the variability
as float). The cloud masks and the CDI of both are compared (the CDI bit for bit, NaN for FORCE's -9999), then the buffer
of FORCE's mask by 30 and 15 pixels (300 m at 10 and 20 m) and the buffer of random masks of several radii.
Times: FORCE's harness (gcc -O2, 1 and 4 OpenMP threads) and Zeit (release build, Windows), best of ``--reps``, on the
scene of ``--big`` pixels a side.
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_parallax")
BANDS = ["rededge3", "nir", "nir08"]


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def make_scene(size, seed=11):
    """re3, bnir, nir, var (int16), pcp, off, brt, sat (bool), all (size, size)."""
    rng = np.random.default_rng(seed)
    n = size
    yy, xx = np.mgrid[:n, :n]

    def smooth(s, scale=1.0):
        f = gaussian_filter(rng.standard_normal((n, n)), s)
        return scale * f / f.std()

    re3 = 2500 * (1 + 0.15 * smooth(25))
    nir = 3000 * (1 + 0.15 * smooth(30))
    bnir = 3200 * (1 + 0.15 * smooth(35))
    var = 6500 + 1500 * smooth(40)
    pcp = rng.random((n, n)) < 0.01

    def shifted(a, mask, dx):       # what B08 sees of an object: the object moved by dx pixels
        return np.roll(a, dx, axis=1), np.roll(mask, dx, axis=1)

    # clouds: textured, high, displaced in B08
    tex = smooth(1.5, 700) + 300 * rng.standard_normal((n, n))
    for cy, cx, r in ((0.25, 0.3, 0.09), (0.7, 0.65, 0.12), (0.35, 0.75, 0.06), (0.8, 0.2, 0.05)):
        m = (yy - cy * n) ** 2 + (xx - cx * n) ** 2 < (r * n) ** 2
        val = 6500 + tex
        re3 = np.where(m, 0.95 * val + 100 * rng.standard_normal((n, n)), re3)
        nir = np.where(m, val, nir)
        sv, sm = shifted(val, m, 6)
        bnir = np.where(sm, sv, bnir)
        var = np.where(m, 5000 + 800 * smooth(5), var)
        pcp |= m | sm
    # built-up: aligned in every band, the ratios change alike
    for y0, x0, h, w in ((0.1, 0.55, 0.12, 0.15), (0.55, 0.1, 0.1, 0.12), (0.85, 0.5, 0.1, 0.2)):
        sl = (slice(int(y0 * n), int((y0 + h) * n)), slice(int(x0 * n), int((x0 + w) * n)))
        val = 4500 + 600 * rng.standard_normal((n, n))
        nir[sl] = val[sl]
        re3[sl] = (0.75 * val + 150 * rng.standard_normal((n, n)))[sl]
        bnir[sl] = (1.1 * val + 150 * rng.standard_normal((n, n)))[sl]
        var[sl] = 4000
        pcp[sl] = True
    # a bright surface with a high variability probability and a 1 pixel offset in B08 (CDI < 0, variability > 0.8)
    m = (np.abs(yy - 0.5 * n) < 0.1 * n) & (np.abs(xx - 0.4 * n) < 0.12 * n)
    val = 5500 + 500 * rng.standard_normal((n, n))
    nir = np.where(m, val, nir)
    re3 = np.where(m, val * 0.98, re3)
    sv, sm = shifted(val, m, 1)
    bnir = np.where(sm, sv, bnir)
    var = np.where(m | sm, 9300, var)
    pcp |= m
    off = np.zeros((n, n), bool)
    off[:, int(0.9 * n):int(0.9 * n) + 3] = True
    off[rng.random((n, n)) < 0.01] = True
    off[int(0.3 * n):int(0.3 * n) + 4, int(0.3 * n):int(0.3 * n) + 5] = True
    off[0:2, :] |= rng.random((2, n)) < 0.4
    sat = rng.random((n, n)) < 0.003
    brt = smooth(60) < 1.3
    clip = lambda a: np.clip(np.round(a), -32768, 32767).astype(np.int16)  # noqa: E731
    out = [clip(a) for a in (re3, bnir, nir, var)]
    for a in out[:3]:
        a[off] = -9999
    brt &= ~off
    return (*out, pcp, off, brt, sat)


def write_scene(path, re3, bnir, nir, var, pcp, off, brt, sat):
    ny, nx = off.shape
    with open(path, "wb") as f:
        f.write(np.array([nx, ny], np.int32).tobytes())
        for a in (re3, bnir, nir, var):
            f.write(np.ascontiguousarray(a, np.int16).tobytes())
        for m in (pcp, off, brt, sat):
            f.write(m.astype(np.uint8).tobytes())


def run_harness(threads, *args):
    exe = "$HOME/force-ref-work/parallax/parallax_harness"
    cmd = f"OMP_NUM_THREADS={threads} {exe} " + " ".join(_wsl(a) if isinstance(a, Path) else str(a) for a in args)
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed: {r.stderr}")
    return float(r.stdout.strip().splitlines()[-1])


def zeit_cube(re3, bnir, nir, off):
    n = off.shape[0]
    a = np.stack([re3, bnir, nir]).astype(np.float32)
    a[:, off] = np.nan
    return xr.DataArray(a, dims=("band", "y", "x"), coords={"band": BANDS, "y": np.arange(n) * -10.0,
                                                           "x": np.arange(n) * 10.0})


def best(fn, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return min(t)


def compare_buffer(zeit, mask, r, tag):
    ny, nx = mask.shape
    src = DATA / "buf_in.bin"
    out = DATA / "buf_out.bin"
    with open(src, "wb") as f:
        f.write(np.array([nx, ny], np.int32).tobytes())
        f.write(mask.astype(np.uint8).tobytes())
    run_harness(1, "buffer", src, out, r)
    ref = np.fromfile(out, np.uint8).reshape(ny, nx).astype(bool)
    forced = zeit.buffer_mask(mask, r, units="px", force=True, n_jobs=1)
    free = zeit.buffer_mask(mask, r, units="px", force=False, n_jobs=1)
    d = int((forced != ref).sum())
    near = np.zeros_like(mask)
    near[:r + 1] = near[-r - 1:] = True
    near[:, :r + 1] = near[:, -r - 1:] = True
    print(f"  {tag:28s} r={r:3d}  true in: {int(mask.sum()):7d}  FORCE: {int(ref.sum()):7d}  force=True differs at "
          f"{d} px ({100 * (forced == ref).mean():.4f}% identical);  force=False differs from FORCE at "
          f"{int((free != ref).sum())} px, {int(((free != ref) & ~near).sum())} of them away from the edge")
    return d


def compare_scene(zeit, arrays):
    """(cloud pixels of FORCE, pixels where the masks differ, blocks where the CDI differs bit for bit)."""
    re3, bnir, nir, var, pcp, off, brt, sat = arrays
    n = off.shape[0]
    scene, out, cdi_out = DATA / "scene2.bin", DATA / "cld2.bin", DATA / "cdi2.bin"
    write_scene(scene, *arrays)
    run_harness(1, "parallax", scene, out, cdi_out)
    ref = np.fromfile(out, np.uint8).reshape(n, n).astype(bool)
    ref_cdi = np.fromfile(cdi_out, np.float32).reshape(n // 2, n // 2)
    cloud, cdi = zeit.parallax_clouds(zeit_cube(re3, bnir, nir, off), pcp, variability=var / 10000.0, saturated=sat,
                                      bright=brt, scale=10000, return_cdi=True, n_jobs=1)
    ref_cdi = np.where(ref_cdi == -9999.0, np.float32(np.nan), ref_cdi)
    cdi = cdi.values
    same = (cdi.view(np.uint32) == ref_cdi.view(np.uint32)) | (np.isnan(cdi) & np.isnan(ref_cdi))
    return int(ref.sum()), int((cloud.values != ref).sum()), int((~same).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--big", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-times", action="store_true")
    args = ap.parse_args()
    no_power_throttling()
    import zeit
    from zeit import _core

    arrays = make_scene(args.size)
    re3, bnir, nir, var, pcp, off, brt, sat = arrays
    scene, out, cdi_out = DATA / "scene.bin", DATA / "cld.bin", DATA / "cdi.bin"
    write_scene(scene, *arrays)
    n = args.size
    print(f"scene {n} x {n}: {off.sum()} off, {pcp.sum()} potential cloud, {sat.sum()} saturated, {brt.sum()} bright")
    run_harness(1, "parallax", scene, out, cdi_out)
    ref = np.fromfile(out, np.uint8).reshape(n, n).astype(bool)
    ref_cdi = np.fromfile(cdi_out, np.float32).reshape(n // 2, n // 2)

    cube = zeit_cube(re3, bnir, nir, off)
    cloud, cdi = zeit.parallax_clouds(cube, pcp, variability=var / 10000.0, saturated=sat, bright=brt, scale=10000,
                                      return_cdi=True, n_jobs=1)
    cloud, cdi = cloud.values, cdi.values
    ref_cdi_nan = np.where(ref_cdi == -9999.0, np.float32(np.nan), ref_cdi)
    same_cdi = (cdi.view(np.uint32) == ref_cdi_nan.view(np.uint32)) | (np.isnan(cdi) & np.isnan(ref_cdi_nan))
    print(f"\nFORCE: {int(ref.sum())} cloud pixels; CDI on {int((ref_cdi != -9999).sum())} of {ref_cdi.size} blocks, "
          f"{int(((ref_cdi < -0.5) & (ref_cdi != -9999)).sum())} at or below -0.5, {int((ref_cdi > 0).sum())} above 0")
    print(f"cloud mask  zeit vs FORCE: {100 * (cloud == ref).mean():.4f}% identical ({int((cloud != ref).sum())} "
          f"pixels differ)")
    print(f"CDI         zeit vs FORCE: {100 * same_cdi.mean():.4f}% bit for bit ({int((~same_cdi).sum())} blocks differ)")
    only = zeit.cloud_displacement_index(cube, potential=pcp, scale=10000, n_jobs=1).values
    print(f"cloud_displacement_index equals the CDI of parallax_clouds: {np.array_equal(only, cdi, equal_nan=True)}")
    c4 = zeit.parallax_clouds(cube, pcp, variability=var / 10000.0, saturated=sat, bright=brt, scale=10000,
                              n_jobs=4).values
    print(f"4 threads equal 1 thread: {np.array_equal(c4, cloud)}")

    print("\nother scenes (seed, variability scaled): FORCE cloud pixels / pixels differing / CDI blocks differing")
    for seed, vscale in ((12, 1.0), (13, 1.0), (14, 0.5), (15, 1.15)):
        a = list(make_scene(args.size, seed=seed))
        a[3] = np.clip(np.round(a[3] * vscale), -32768, 32767).astype(np.int16)
        print(f"  seed {seed}, variability x {vscale}: {compare_scene(zeit, a)}")

    print("\nbuffer (zeit force=True vs FORCE's buffer_; force=False vs FORCE)")
    rng = np.random.default_rng(3)
    bad = compare_buffer(zeit, ref, 30, "FORCE's cloud mask")
    bad += compare_buffer(zeit, ref, 15, "FORCE's cloud mask")
    for r in (1, 2, 3, 7, 12, 30):
        m = rng.random((n, n)) < (0.002 if r > 3 else 0.01)
        bad += compare_buffer(zeit, m, r, "random mask")
    m = np.zeros((n, n), bool)
    m[:40, :60] = True
    m[-30:, 100:130] = True
    bad += compare_buffer(zeit, m, 5, "blocks at the edges")
    print(f"\ntotal pixels differing from FORCE: cloud {int((cloud != ref).sum())}, CDI {int((~same_cdi).sum())}, "
          f"buffer {bad}")

    if args.skip_times:
        return
    # times, on the big scene
    big = make_scene(args.big, seed=5)
    write_scene(scene, *big)
    bre3, bbnir, bnir_, bvar, bpcp, boff, bbrt, bsat = big
    bcube = zeit_cube(bre3, bbnir, bnir_, boff)
    bvarf = bvar / 10000.0
    mask = ref if args.big == n else np.kron(ref, np.ones((args.big // n + 1,) * 2, bool))[:args.big, :args.big]
    bmask = DATA / "bmask.bin"
    with open(bmask, "wb") as f:
        f.write(np.array([mask.shape[1], mask.shape[0]], np.int32).tobytes())
        f.write(mask.astype(np.uint8).tobytes())
    print(f"\ntimes on {args.big} x {args.big} (ms, best of {args.reps}); zeit end to end (numpy conversions included) "
          f"and the core alone")
    print(f"{'':32s} {'1 thread':>10s} {'4 threads':>10s}")
    f1 = min(run_harness(1, "parallax", scene, out, cdi_out) for _ in range(args.reps))
    f4 = min(run_harness(4, "parallax", scene, out, cdi_out) for _ in range(args.reps))
    print(f"{'FORCE cloud_parallax':32s} {f1:10.0f} {f4:10.0f}")
    z = {k: best(lambda k=k: zeit.parallax_clouds(bcube, bpcp, variability=bvarf, saturated=bsat, bright=bbrt,
                                                  scale=10000, n_jobs=k), args.reps) for k in (1, 4)}
    print(f"{'zeit parallax_clouds':32s} {z[1]:10.0f} {z[4]:10.0f}")
    c = {}
    for k in (1, 4):
        a = (np.ascontiguousarray(x) for x in (bre3, bbnir, bnir_, bvar))
        a = list(a)
        c[k] = best(lambda k=k: _core.parallax.parallax_batch(a[0], a[1], a[2], a[3], bpcp, boff, bbrt, bsat, True, k),
                    args.reps)
    print(f"{'zeit core (parallax_batch)':32s} {c[1]:10.0f} {c[4]:10.0f}")
    b1 = min(run_harness(1, "buffer", bmask, out, 30) for _ in range(args.reps))
    b4 = min(run_harness(4, "buffer", bmask, out, 30) for _ in range(args.reps))
    zb = {k: best(lambda k=k: zeit.buffer_mask(mask, 30, units="px", force=True, n_jobs=k), args.reps) for k in (1, 4)}
    print(f"{'FORCE buffer_ r=30':32s} {b1:10.0f} {b4:10.0f}")
    print(f"{'zeit buffer_mask r=30':32s} {zb[1]:10.0f} {zb[4]:10.0f}")


if __name__ == "__main__":
    main()
