"""The topography of zeit-core (``zeit_core::topo``, a port of FORCE 3.11.00's ``topo-ll.c``: ``compile_topography`` with the
ocean detection, the smoothing of the DEM, slope and aspect, the elevation statistics with the binned DEM, the illumination
angle and the sky view factor; ``cfactor_topography``; ``average_elevation_cell``) against FORCE's own functions on
synthetic scenes.

The C side is ``force_topo/`` (``make_harness.sh``, WSL): FORCE's topo-ll.c compiled as it is (whole, but for its first
include, behind shims for the brick, atc and parameter structures), with radtran-ll.c; the warp of the DEM file by GDAL is a
shim that hands over the DEM given on the grid of the scene: both sides get the same DEM. Run with ``OMP_NUM_THREADS=1`` (the
average elevation of ``stats_topography`` is a sum of doubles per thread, so its last bits depend on the number of threads;
Zeit's is FORCE with one thread, see the notes of ``zeit_core::topo``). Zeit's side is ``zeit._core.topo`` (Windows, UCRT
libm; the harness runs on glibc 2.39: ``atan``, ``atan2``, ``sqrt``, ``cos``, ``sin``)::

    python benchmarks/vs_originals/force_topo.py [--scale 1.0] [--skip-build]

Scenes: a flat DEM (FORCE with no ``FILE_DEM``, and a DEM of zeros), a ramp, a cone and a valley with the sun low (deep
shadow, poor and moderate illumination), mountains of sinusoids with noise, a DEM with a frame of no-data, holes, an ocean (0
m) with its no-data coast, a QAI with off pixels and flags already set, grids of 5, 10, 20, 30 and 250 m, coarse cells of
different sizes (3 km, as FORCE's default, and finer), a sun that changes over the coarse grid, and DEMs out of the range of
the atmosphere (-500 m to 9000 m: FORCE fails). The C-factor is run on scenes with a SWIR1 and SWIR2 made from the terrain and
three land covers. Prints, per scene and field, the number of values compared, how many are equal and how many differ (the
integers exactly; the elevation statistics in float32 ulps). Exits with 1 if anything differs by more than 1 unit.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_topo")
EXE = "$HOME/force-ref-work/topo/topo_harness"
F32 = np.float32
ND = -32767


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_topo' / 'make_harness.sh')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness build failed: {r.stdout}{r.stderr}")


def ordered(a):
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def smooth_field(rng, ny, nx, scale, n=7):
    """A smooth random field in -1..1: sinusoids of random direction and phase, wavelengths around ``scale`` pixels."""
    ii, jj = np.mgrid[0:ny, 0:nx].astype(np.float64)
    f = np.zeros((ny, nx))
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        k = 2 * np.pi / (scale * rng.uniform(0.5, 2.0))
        f += rng.uniform(0.3, 1.0) * np.sin(k * (np.cos(a) * jj + np.sin(a) * ii) + rng.uniform(0, 2 * np.pi))
    return f / np.abs(f).max()


def make_scene(rng, nx, ny, res, cs, surface, flat=False, do_c=True, sun=(40.0, 150.0), qai_kind="clean"):
    """The inputs of one scene as the binding and the harness take them; ``cs`` is the coarse cell in pixels."""
    gres = res * cs
    gnx, gny = -(-nx // cs), -(-ny // cs)
    gn, npix = gnx * gny, nx * ny
    ii, jj = np.mgrid[0:ny, 0:nx].astype(np.float64)
    ci, cj = ny / 2, nx / 2
    r = np.hypot(ii - ci, jj - cj)
    hill = 1500.0 * 30.0 / res  # a slope of 30 m per 30 m at most
    if surface == "flat":
        dem = np.zeros((ny, nx))
    elif surface == "zeros":
        dem = np.zeros((ny, nx))
    elif surface == "ramp":
        dem = 100.0 + 0.37 * res * jj + 0.11 * res * ii
    elif surface == "cone":
        dem = np.maximum(0.0, 1800.0 - 0.9 * res * r)
    elif surface == "valley":
        dem = 200.0 + 0.6 * res * np.minimum(r, 0.35 * nx)
        dem += 40.0 * np.sin(jj / 11.0)
    elif surface in ("mountains", "ocean", "holes"):
        dem = 900.0 + 700.0 * smooth_field(rng, ny, nx, 80.0 * 30.0 / res) + 180.0 * smooth_field(rng, ny, nx, 15.0 * 30.0 / res)
        dem += rng.normal(0, 0.4 + 0.02 * res, dem.shape)
        if surface == "ocean":
            dem = dem - 800.0
            sea = dem < 0
            dem[sea] = 0.0
        dem = np.maximum(dem, 0.0)
    elif surface == "high":
        dem = 3000.0 + 6500.0 * smooth_field(rng, ny, nx, 60.0) ** 2
    elif surface == "low":
        dem = -700.0 * (0.5 + 0.5 * smooth_field(rng, ny, nx, 60.0))
    else:
        raise ValueError(surface)
    dem = dem.astype(F32)
    if surface == "ocean":
        # the sea is 0 m, its coast is no-data, a lake
        coast = (dem == 0)
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(coast, 2)
        win = sliding_window_view(pad, (5, 5)).any(axis=(2, 3))
        dem[win & ~coast & (rng.random(dem.shape) < 0.7)] = ND
    if surface == "holes":
        for _ in range(12):
            i0, j0 = rng.integers(0, ny - 8), rng.integers(0, nx - 8)
            dem[i0:i0 + rng.integers(1, 7), j0:j0 + rng.integers(1, 7)] = ND
        dem[:, :3] = ND
        dem[-2:, :] = ND
        dem[rng.random(dem.shape) < 0.003] = ND
    qai = np.zeros((ny, nx), np.int16)
    if qai_kind in ("frame", "dirty"):
        qai[:2, :] |= 1
        qai[:, -3:] |= 1
        qai[ny // 3:ny // 3 + 5, nx // 4:nx // 4 + 9] |= 1
        qai[rng.random(qai.shape) < 0.002] |= 1
    if qai_kind == "dirty":  # flags of other stages: cloud, shadow, water, and the bits of the slope and the illumination
        qai |= (rng.integers(0, 4, qai.shape) << 1).astype(np.int16)
        qai |= (rng.integers(0, 2, qai.shape) << 3).astype(np.int16)
        qai |= (rng.integers(0, 2, qai.shape) << 5).astype(np.int16)
        qai |= (rng.integers(0, 4, qai.shape) << 11).astype(np.int16)
        qai |= (rng.integers(0, 2, qai.shape) << 13).astype(np.int16)
    if qai_kind == "alloff":
        qai[:] = 1
    gj, gi = np.meshgrid(np.arange(gnx), np.arange(gny))
    zen = np.radians(sun[0] + 6.0 * (gi / max(gny - 1, 1)) - 4.0 * (gj / max(gnx - 1, 1))).ravel()
    azi = np.radians(sun[1] + 25.0 * (gj / max(gnx - 1, 1)) - 15.0 * (gi / max(gny - 1, 1))).ravel()
    sc = dict(nx=nx, ny=ny, res=float(res), gnx=gnx, gny=gny, gres=float(gres), dem_nodata=ND, flat=int(flat), do_c=int(do_c),
              sun_zen=zen.astype(F32), sun_azi=azi.astype(F32), sun_czen=np.cos(zen).astype(F32),
              sun_szen=np.sin(zen).astype(F32), dem=dem.ravel(), qai=qai.ravel())
    if do_c:
        domain = ["GREEN", "SWIR2", "RED", "SWIR1"]
        gy, gx = np.gradient(dem.astype(np.float64), res)
        # an approximate cosine of the illumination angle, to make the SWIR2 reflectance follow it
        cosi = np.clip((np.cos(zen.mean()) + np.sin(zen.mean()) * (np.cos(np.radians(sun[1])) * gy - np.sin(np.radians(sun[1])) * gx))
                       / np.sqrt(1 + gx ** 2 + gy ** 2), 0.02, 1)
        cover = (smooth_field(rng, ny, nx, 40.0 * 30.0 / res) * 1.5).round().clip(-1, 1)  # three land covers
        sw2 = 0.12 + 0.05 * cover + (0.10 + 0.06 * cover) * cosi + rng.normal(0, 0.004, cosi.shape)
        sw1 = sw2 * (1.4 + 0.3 * cover) + 0.02
        toa = np.zeros((4, ny, nx), np.int16)
        toa[0] = (0.05 * 10000 + rng.normal(0, 30, cosi.shape)).astype(np.int16)
        toa[1] = np.round(sw2 * 10000).astype(np.int16)
        toa[2] = (0.04 * 10000 + rng.normal(0, 30, cosi.shape)).astype(np.int16)
        toa[3] = np.round(sw1 * 10000).astype(np.int16)
        z = np.arange(256)[:, None]
        rho_p = (0.002 + 0.00004 * (255 - z) * (0.5 + rng.random((1, gn))) + rng.random((256, gn)) * 0.003).astype(F32)
        tss = (0.55 + 0.0012 * (255 - z) + 0.1 * rng.random((256, gn))).astype(F32)
        tsd = (0.80 + 0.0005 * (255 - z) + 0.1 * rng.random((256, gn))).astype(F32)
        sc.update(domain=domain, toa=toa.ravel(), rho_p=rho_p.ravel(), tss=tss.ravel(), tsd=tsd.ravel())
    return sc


def write_scene(path, sc):
    b = bytearray()
    i32 = lambda *v: b.extend(np.array(v, "<i4").tobytes())  # noqa: E731
    f64 = lambda *v: b.extend(np.array(v, "<f8").tobytes())  # noqa: E731
    i32(sc["nx"], sc["ny"])
    f64(sc["res"])
    i32(sc["gnx"], sc["gny"])
    f64(sc["gres"])
    i32(sc["dem_nodata"], sc["flat"], sc["do_c"])
    for k in ("sun_zen", "sun_azi", "sun_czen", "sun_szen", "dem"):
        b.extend(np.ascontiguousarray(sc[k], "<f4").tobytes())
    b.extend(np.ascontiguousarray(sc["qai"], "<i2").tobytes())
    if sc["do_c"]:
        i32(len(sc["domain"]))
        for d in sc["domain"]:
            b.extend(d.encode().ljust(16, b"\0"))
        b.extend(np.ascontiguousarray(sc["toa"], "<i2").tobytes())
        for k in ("rho_p", "tss", "tsd"):
            b.extend(np.ascontiguousarray(sc[k], "<f4").tobytes())
    Path(path).write_bytes(bytes(b))


def parse(raw, sc):
    npix, gn = sc["nx"] * sc["ny"], sc["gnx"] * sc["gny"]
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    o = {"ret": int(take("<i4", 1)[0]), "stats": take("<f4", 4), "cnum": int(take("<i4", 1)[0]), "qai": take("<i2", npix)}
    if o["ret"] == 0:
        o["dem"] = take("u1", npix)
        o["slope"] = take("<u2", npix)
        o["aspect"] = take("<u2", npix)
        o["ill"] = take("<i2", npix)
        o["sky"] = take("<u2", npix)
        if sc["do_c"]:
            o["cnull"] = int(take("<i4", 1)[0])
            o["cf"] = take("<u2", npix)
        o["xy_dem"] = take("<i4", gn)
    o["ms"] = take("<f8", 2)
    return o


def run_force(name, sc, threads=1):
    fin, fout = DATA / f"{name}.in.bin", DATA / f"{name}.out.bin"
    write_scene(fin, sc)
    cmd = f"OMP_NUM_THREADS={threads} {EXE} scene {_wsl(fin)} {_wsl(fout)}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed: {r.stdout}{r.stderr}")
    return parse(fout.read_bytes(), sc)


def run_zeit(sc):
    from zeit import _core

    d = {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in sc.items()}
    d["flat"], d["do_c"] = bool(sc["flat"]), bool(sc["do_c"])
    raw, _, _ = _core.topo.scene(d)
    return parse(bytes(raw), sc)


def compare(title, sc, c, z):
    """Prints one line per field; returns (bad, values, equal)."""
    bad, tv, te = False, 0, 0
    shown = [title]

    def line(field, n, same, mx, note=""):
        nonlocal bad, tv, te
        tv += n
        te += same
        flag = ""
        if mx > 1:
            bad = True
            flag = "  <-- MORE THAN 1"
        print(f"{shown[0]:42s} {field:18s} {n:9d} {same:12d} {n - same:8d} {mx:6d}{note}{flag}")
        shown[0] = ""

    line("return code", 1, int(c["ret"] == z["ret"]), 0 if c["ret"] == z["ret"] else 9, f" (FORCE {c['ret']}, Zeit {z['ret']})")
    st_c, st_z = ordered(c["stats"]), ordered(z["stats"])
    d = np.abs(st_c - st_z)
    line("dem stats", 4, int((d == 0).sum()), int(d.max()), f" (avg {c['stats'][0]:.6f} km)")
    line("QAI", len(c["qai"]), int((c["qai"] == z["qai"]).sum()), 0 if (c["qai"] == z["qai"]).all() else 9)
    if c["ret"] != 0 or z["ret"] != 0:
        return bad, tv, te
    for k in ("dem", "slope", "aspect", "ill", "sky", "cf", "xy_dem"):
        if k not in c:
            continue
        a, b = c[k].astype(np.int64), z[k].astype(np.int64)
        dd = np.abs(a - b)
        extra = ""
        if k == "slope":
            extra = f" (max {c[k].max() / 10000 * 57.2958:.1f} deg, {int((c['qai'] >> 13 & 1).sum())} flagged)"
        if k == "ill":
            extra = f" (min {c[k].min()}, QAI illumination 3/2/1: " + "/".join(
                str(int(((c['qai'] >> 11 & 3) == v).sum())) for v in (3, 2, 1)) + ")"
        if k == "cf":
            extra = f" ({int((c[k] > 0).sum())} > 0, max {c[k].max()})"
        line(k, a.size, int((dd == 0).sum()), int(dd.max()), extra)
    return bad, tv, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the area of the large scene")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    no_power_throttling()
    rng = np.random.default_rng(23)
    if not args.skip_build:
        build_harness()
    big = int(round(900 * np.sqrt(args.scale)))
    scenes = [
        ("flat, no DEM file, 10 m", dict(nx=300, ny=240, res=10, cs=60, surface="flat", flat=True)),
        ("DEM of zeros, 10 m, frame QAI", dict(nx=300, ny=240, res=10, cs=60, surface="zeros", qai_kind="frame")),
        ("ramp, 10 m", dict(nx=400, ny=300, res=10, cs=100, surface="ramp")),
        ("cone, 10 m, 3 km cells, low sun", dict(nx=500, ny=400, res=10, cs=300, surface="cone", sun=(62.0, 120.0))),
        ("valley, 20 m, dirty QAI", dict(nx=300, ny=260, res=20, cs=150, surface="valley", sun=(55.0, 200.0), qai_kind="dirty")),
        ("mountains, 30 m (no smoothing), 3 km", dict(nx=400, ny=400, res=30, cs=100, surface="mountains", sun=(35.0, 160.0),
                                                      qai_kind="frame")),
        ("mountains, 5 m (k = 6)", dict(nx=220, ny=200, res=5, cs=200, surface="mountains", sun=(48.0, 110.0))),
        ("mountains, 250 m", dict(nx=110, ny=100, res=250, cs=12, surface="mountains", sun=(30.0, 170.0))),
        ("no-data holes and frame, 20 m", dict(nx=300, ny=260, res=20, cs=50, surface="holes", qai_kind="dirty")),
        ("ocean and its no-data coast, 10 m", dict(nx=320, ny=280, res=10, cs=80, surface="ocean", sun=(50.0, 140.0))),
        ("all off", dict(nx=60, ny=50, res=30, cs=20, surface="mountains", qai_kind="alloff")),
        ("DEM above 9 km (FORCE fails)", dict(nx=120, ny=100, res=30, cs=40, surface="high")),
        ("DEM below -500 m (FORCE fails)", dict(nx=120, ny=100, res=30, cs=40, surface="low")),
        (f"large {big}x{int(big * 0.9)}, 10 m, 3 km cells", dict(nx=big, ny=int(big * 0.9), res=10, cs=300, surface="mountains",
                                                             sun=(42.0, 135.0), qai_kind="frame")),
    ]
    print(f"{'scene':42s} {'field':18s} {'values':>9s} {'equal':>12s} {'differ':>8s} {'max':>6s}")
    bad = False
    tot = np.zeros(2, np.int64)
    for k, (title, kw) in enumerate(scenes):
        sc = make_scene(rng, **kw)
        c = run_force(f"scene{k}", sc)
        z = run_zeit(sc)
        b, tv, te = compare(title, sc, c, z)
        bad |= b
        tot += (tv, te)
        print(f"{'':42s} FORCE {c['ms'][0]:.1f} ms compile_topography + {c['ms'][1]:.1f} ms cfactor, Zeit {z['ms'][0]:.1f} + "
              f"{z['ms'][1]:.1f} ms")
        if k == len(scenes) - 1:  # FORCE with 4 threads: the sum of the elevation statistics per thread
            c4 = run_force(f"scene{k}_t4", sc, threads=4)
            same = all((c[f] == c4[f]).all() for f in ("slope", "aspect", "ill", "sky", "dem", "cf", "qai"))
            print(f"{'':42s} FORCE with 4 threads vs 1: every field equal: {same}; dem.avg {c['stats'][0]!r} vs "
                  f"{c4['stats'][0]!r} ({int(abs(ordered(c['stats'][:1]) - ordered(c4['stats'][:1]))[0])} ulp)")
    print(f"{'total (bit for bit)':42s} {'':18s} {tot[0]:9d} {tot[1]:12d} {tot[0] - tot[1]:8d}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
