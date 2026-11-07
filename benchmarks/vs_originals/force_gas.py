"""The water vapour and gas transmittance of zeit-core (``zeit_core::gas``, a port of FORCE 3.11.00's ``gas-ll.c``:
``wvp_transmitt_lut``, ``water_vapor`` with its minimizer ``fun_wvp`` and GSL's ``nmsimplex2``, ``gas_transmittance`` and
``water_vapor_from_lut``) against FORCE's own functions on synthetic scenes and LUT files.

The C side is ``force_gas/`` (``make_harness.sh``, WSL): FORCE's gas-ll.c compiled as it is (but for its first include,
behind shims for the brick, atc, meta and parameter structures), with radtran-ll.c and table-ll.c, against the real GSL
2.7 and its reference CBLAS, linked statically; run with ``OMP_NUM_THREADS=1`` (FORCE's retrieval carries the last
estimate of each OpenMP thread to its next pixel, so its output depends on the number of threads; Zeit's is FORCE with
one thread, see the notes of ``zeit_core::gas``). Zeit's side is ``zeit._core.gas`` (Windows, UCRT libm; the harness runs
on glibc 2.39)::

    python benchmarks/vs_originals/force_gas.py [--scale 1.0] [--skip-build]

Scenes: Sentinel-2 (13 bands of the RSR table, NIR = B8A, vapour = B9) with a smooth water vapour field (0.4 to 5 cm, a
patch of 9 cm that fails), a sun and view geometry on a coarse grid (with cells without data at one edge), an elevation
(the 256 bins of the path reflectance, transmittance and albedo), TOA reflectances from the forward model plus noise,
and a QAI with off pixels, shadow, water (including whole blocks), clouds and other flags. Prints, per scene and
function, the number of values compared, how many are equal bit for bit and how many differ (the 16-bit outputs in
digits of 1/1000 cm and 1/10000, the LUT in float32 ulps). Exits with 1 if anything differs by more than 1 float32 ulp
(the LUT) or at all (the 16-bit outputs). Then the water vapour LUT files (``water_vapor_from_lut``) on a few hundred
scenarios, and the retrieval of FORCE with 4 threads against its own single-thread result, for the size of that effect.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_gas")
EXE = "$HOME/force-ref-work/gas/gas_harness"
F32 = np.float32
RSR0 = 30  # RSR_START_SEN2A
NB = 13
B_NIR, B_VAP = 8, 9  # B8A, B9


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_gas' / 'make_harness.sh')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness build failed: {r.stdout}{r.stderr}")


def ordered(a):
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp_compare(a, b):
    """(values, bit-equal, 1 ulp, more, max ulp) of float32 arrays, NaN == NaN."""
    a, b = np.asarray(a, F32).ravel(), np.asarray(b, F32).ravel()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.abs(ordered(a) - ordered(b))
    d = np.where(nan_a & nan_b, 0, d)
    d = np.where(nan_a ^ nan_b, 1 << 40, d)
    return a.size, int((d == 0).sum()), int((d == 1).sum()), int((d > 1).sum()), int(d.max()) if a.size else 0


def make_scene(rng, nx, ny, gres, kind="normal"):
    """The inputs of one scene (arrays as the binding and the harness take them)."""
    from zeit import _core

    res = 10.0
    gnx, gny = int(np.ceil(nx * res / gres)), int(np.ceil(ny * res / gres))
    gn, npix = gnx * gny, nx * ny
    nodata = -9999
    gj, gi = np.meshgrid(np.arange(gnx), np.arange(gny))
    gi, gj = gi.ravel(), gj.ravel()
    sun = (0.55 + 0.2 * gi / max(gny - 1, 1) + 0.02 * rng.random(gn)).astype(F32)
    view = (0.92 + 0.07 * (gj / max(gnx - 1, 1)) + 0.005 * rng.random(gn)).astype(F32)
    if gnx > 3:
        view[gj >= gnx - 1] = nodata  # outside the swath
    valid = view != nodata
    cosszen = np.array([sun[valid].min(), sun[valid].max()], F32)
    cosvzen = np.array([view[valid].min(), view[valid].max()], F32)
    if kind == "lowsun":  # a long air mass: the LUT's cosine range reaches low values
        sun = (0.25 + 0.25 * gi / max(gny - 1, 1)).astype(F32)
        cosszen = np.array([sun.min(), sun.max()], F32)

    def U(lo, hi, *shape):
        return rng.uniform(lo, hi, shape).astype(F32)

    tso, tvo = U(0.93, 1.0, NB, gn), U(0.93, 1.0, NB, gn)
    z = np.arange(256, dtype=F32)[:, None, None]
    rho0 = U(0.01, 0.12, 1, NB, gn)
    rho_p = (rho0 * (1 - 0.0012 * z) * (1 + 0.05 * rng.random((256, NB, gn)))).astype(F32)
    t = (U(0.80, 0.95, 1, NB, gn) - 0.00025 * z + 0.01 * rng.random((256, NB, gn))).astype(F32)
    s = U(0.04, 0.12, 1, NB, gn) * np.ones((256, 1, 1), F32) + 0.002 * rng.random((256, NB, gn)).astype(F32)
    s = s.astype(F32)

    # per pixel: block (60 m) indices, coarse cell, elevation bin
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    g = (np.floor(ii * res / gres).astype(int) * gnx + np.floor(jj * res / gres).astype(int)).ravel()
    dem = np.clip(40 + 3 * ii + 2 * jj + rng.integers(0, 12, (ny, nx)), 0, 255).astype(np.uint8).ravel()
    # the retrieval's truth: smooth, per 60 m block
    w_block = 0.4 + 4.6 * (0.5 + 0.5 * np.sin(ii[::6, ::6] / 23.0) * np.cos(jj[::6, ::6] / 31.0))
    if nx >= 60 and ny >= 60:
        w_block[2:4, 3:5] = 9.0  # fails (above 7 cm)
    w = np.repeat(np.repeat(w_block, 6, 0), 6, 1)[:ny, :nx].ravel().astype(F32)
    ms, mv = sun[g], view[g]
    mv = np.where(mv == nodata, F32(0.97), mv).astype(F32)
    zi = dem.astype(int)
    toa = np.zeros((NB, npix), np.int16)
    sr_nir = U(0.12, 0.5, npix)
    sr_vap = (sr_nir * (1 + 0.04 * (rng.random(npix) - 0.5))).astype(F32)
    for b, sr in ((B_NIR, sr_nir), (B_VAP, sr_vap)):
        tsw = _core.radtran.radtran_eval("wvp_transmitt", w, ms, np.full(npix, RSR0 + b, F32))[0]
        tvw = _core.radtran.radtran_eval("wvp_transmitt", w, mv, np.full(npix, RSR0 + b, F32))[0]
        tgw = tsw * tvw * tso[b][g] * tvo[b][g]
        tmp = sr * t[zi, b, g] / (1 - s[zi, b, g] * sr)
        toa[b] = np.round(10000 * (rho_p[zi, b, g] + tgw * tmp) + rng.normal(0, 8, npix)).astype(np.int16)
    for b in range(NB):
        if b not in (B_NIR, B_VAP):
            toa[b] = rng.integers(100, 4000, npix)
    # a few pixels of the reference band under 0.15, one of the vapour band at nonsense (a cloud edge)
    low = rng.random(npix) < 0.03
    toa[B_NIR][low] = rng.integers(0, 1500, low.sum())
    cl = rng.random(npix) < 0.01
    toa[B_VAP][cl] = rng.integers(0, 12000, cl.sum())

    # QAI: random flags in the bits the functions do not read, off at the frame, shadow, water (patches and noise)
    flag = lambda p: (rng.random(npix) < p).astype(np.int32)  # noqa: E731
    qai = (rng.integers(0, 2, npix) << 1 | rng.integers(0, 2, npix) << 2 | flag(0.2) << 4
           | rng.integers(0, 4, npix) << 6 | flag(0.1) << 8 | flag(0.1) << 13).astype(np.int16)
    qai = qai.reshape(ny, nx)
    qai[(rng.random((ny, nx)) < 0.05)] |= 1 << 3  # shadow
    qai[(rng.random((ny, nx)) < 0.05)] |= 1 << 5  # water
    qai[:, -3:] |= 1  # off
    if ny >= 24 and nx >= 24:
        qai[6:18, 12:24] |= 1 << 5  # two whole blocks of water
        qai[6:12, 6:12] |= 1  # a block off
    qai = qai.ravel()
    if kind == "allwater":
        qai = (qai | (1 << 5)).astype(np.int16)
    # nothing in the (random) QAI sets bit 14 before the fill: FORCE clears it first, so set it for a few
    qai[rng.random(npix) < 0.02] |= 1 << 14
    return dict(nx=nx, ny=ny, res=res, gnx=gnx, gny=gny, gres=float(gres), nodata_view=nodata, b_nir=B_NIR,
                b_vapor=B_VAP, rsr_band=(RSR0 + np.arange(NB)).astype(np.int32), cosszen=cosszen, cosvzen=cosvzen,
                sun_czen=sun, view_czen=view, tso=tso.ravel(), tvo=tvo.ravel(), rho_p=rho_p.ravel(), t=t.ravel(),
                s=s.ravel(), toa=toa.ravel(), qai=qai.astype(np.int16), dem=dem)


def write_scene(path, sc):
    with open(path, "wb") as f:
        a = lambda x, dt: f.write(np.ascontiguousarray(x, dt).tobytes())  # noqa: E731
        a([NB, sc["nx"], sc["ny"]], np.int32)
        a([sc["res"]], np.float64)
        a([sc["gnx"], sc["gny"]], np.int32)
        a([sc["gres"]], np.float64)
        a([sc["nodata_view"], sc["b_nir"], sc["b_vapor"]], np.int32)
        a(sc["rsr_band"], np.int32)
        a(sc["cosszen"], F32)
        a(sc["cosvzen"], F32)
        a(sc["sun_czen"], F32)
        a(sc["view_czen"], F32)
        a(sc["tso"], F32)
        a(sc["tvo"], F32)
        gn = sc["gnx"] * sc["gny"]
        rho = sc["rho_p"].reshape(256, NB * gn)
        t = sc["t"].reshape(256, NB * gn)
        s = sc["s"].reshape(256, NB * gn)
        for z in range(256):
            a(rho[z], F32)
            a(t[z], F32)
            a(s[z], F32)
        a(sc["toa"], np.int16)
        a(sc["qai"], np.int16)
        a(sc["dem"], np.uint8)


def run_force_scene(name, sc, threads=1):
    fin, fout = DATA / f"{name}.in", DATA / f"{name}.out"
    write_scene(fin, sc)
    cmd = f"OMP_NUM_THREADS={threads} {EXE} scene {_wsl(fin)} {_wsl(fout)}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed on {name}: {r.stdout}{r.stderr}")
    npix, nx = sc["nx"] * sc["ny"], 0
    raw = fout.read_bytes()
    nlut = int(np.frombuffer(raw[:4], np.int32)[0])
    off = 4
    lut = np.frombuffer(raw, F32, nlut, off)
    off += 4 * nlut
    wvp = np.frombuffer(raw, np.int16, npix, off)
    off += 2 * npix
    qai = np.frombuffer(raw, np.int16, npix, off)
    off += 2 * npix
    tg = np.frombuffer(raw, np.int16, NB * npix, off)
    ms = [float(x) for x in r.stdout.split()[-2:]]
    return lut, wvp, qai, tg, ms


def run_zeit_scene(sc, n_jobs=1):
    import time

    from zeit import _core

    a = {k: v for k, v in sc.items()}
    t0 = time.perf_counter()
    out = _core.gas.water_vapor_scene(a["nx"], a["ny"], a["res"], a["gnx"], a["gny"], a["gres"], a["nodata_view"],
                                      a["b_nir"], a["b_vapor"], a["rsr_band"], a["cosszen"], a["cosvzen"],
                                      a["sun_czen"], a["view_czen"], a["tso"], a["tvo"], a["rho_p"], a["t"], a["s"],
                                      a["toa"], a["qai"], a["dem"], n_jobs)
    return out, (time.perf_counter() - t0) * 1e3


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.size, int((a == b).sum()), int((a != b).sum())


def lut_files(root, rng):
    """Synthetic LUT files in FORCE's format: climatologies for months 1-12 and dailies for some days."""
    root.mkdir(exist_ok=True)
    for f in root.glob("WVP_*.txt"):
        f.unlink()
    sites = [(float(x), float(y)) for x, y in zip(rng.uniform(-10, 10, 60), rng.uniform(40, 55, 60))]
    for m in range(1, 13):
        if m == 7:
            continue  # a month without climatology
        with open(root / f"WVP_0000-{m:02d}-00.txt", "w") as f:
            for x, y in sites:
                v = 9999 if rng.random() < 0.1 else round(float(rng.uniform(0.3, 4.5)), 3)
                f.write(f"{x:.2f} {y:.2f} {v} {int(rng.integers(1, 90))}\n")
    for (y, m, d) in [(2020, 3, 14), (2020, 3, 15), (2021, 6, 1), (2021, 7, 4), (2022, 12, 31)]:
        with open(root / f"WVP_{y:04d}-{m:02d}-{d:02d}.txt", "w") as f:
            for x, yy in sites:
                r = rng.random()
                v = 9999 if r < 0.25 else (-1.5 if r < 0.3 else round(float(rng.uniform(0.2, 6.0)), 3))
                src = "TBD" if v == 9999 else rng.choice(["MOD", "MYD"])
                sep = "  " if rng.random() < 0.2 else " "  # doubled blanks, strtok skips them
                tail = " # comment" if rng.random() < 0.1 else ""
                f.write(f"{x:.2f}{sep}{yy:.2f} {v} {src}{tail}\r\n" if rng.random() < 0.3 else
                        f"{x:.2f}{sep}{yy:.2f} {v} {src}{tail}\n")
    # a file with long values and a LUT without trailing newline
    with open(root / "WVP_2021-09-09.txt", "w") as f:
        f.write("1.0 45.0 2.5 MOD\n2.0 46.0 3.5 MYD")
    return sites


def check_lut_files(rng, scale):
    from zeit import _core

    root = DATA / "lut"
    sites = lut_files(root, rng)
    dates = [(2020, 3, 14), (2020, 3, 15), (2020, 3, 16), (2021, 6, 1), (2021, 7, 4), (2021, 9, 9), (2022, 12, 31),
             (2019, 1, 1), (2021, 8, 8)]
    n = int(300 * scale)
    scen = []
    for k in range(n):
        y, m, d = dates[k % len(dates)]
        if rng.random() < 0.7:
            x0, y0 = sites[int(rng.integers(0, len(sites)))]
            lon, lat = x0 + rng.uniform(-1.2, 1.2), y0 + rng.uniform(-1.2, 1.2)
        else:
            lon, lat = rng.uniform(-30, 30), rng.uniform(20, 70)
        scen.append((str(root) if k % 41 else "NULL", float(lon), float(lat), y, m, d, float(rng.uniform(0.5, 3)),
                     bool(rng.random() < 0.3)))
    # the dir as the harness sees it, with a trailing-blank-free path
    lines = [f"{_wsl(root) if s[0] != 'NULL' else 'NULL'} {s[1]!r} {s[2]!r} {s[3]} {s[4]} {s[5]} {s[6]!r} {int(s[7])}"
             for s in scen]
    fin = DATA / "lut_scenarios.txt"
    fin.write_text("\n".join(lines) + "\n")
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"{EXE} lut < {_wsl(fin)}"], capture_output=True, text=True)
    ref = []
    for ln in r.stdout.splitlines():
        try:
            ref.append(F32(float(ln)))
        except ValueError:
            pass  # a message of FORCE's
    ref = np.array(ref, F32)
    out = np.array(_core.gas.water_vapor_from_lut(scen), F32)
    ok = ref.size == out.size
    same = int((ref.view(np.uint32) == out.view(np.uint32)).sum()) if ok else 0
    neg = int((ref < 0).sum())
    print(f"{'water_vapor_from_lut':26s} {'scenarios':>10s} {len(scen):9d} {same:12d} {len(scen) - same:8d}   "
          f"(of which {neg} -1: no value or strict)")
    return ok and same == len(scen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of LUT scenarios and the big scene")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    no_power_throttling()
    rng = np.random.default_rng(23)
    if not args.skip_build:
        build_harness()
    big = int(round(300 * np.sqrt(args.scale) / 6)) * 6
    scenes = [("small 36x30, 120 m cells", (36, 30, 120.0, "normal")),
              ("medium 120x96, 120 m cells", (120, 96, 120.0, "normal")),
              ("medium 120x96, 60 m cells", (120, 96, 60.0, "normal")),
              ("low sun 102x72, 300 m cells", (102, 72, 300.0, "lowsun")),
              (f"large {big}x{big - 60}, 300 m cells", (big, big - 60, 300.0, "normal")),
              ("all water 12x12 (no retrieval)", (12, 12, 60.0, "allwater"))]
    print(f"{'scene / function':34s} {'values':>9s} {'bit for bit':>12s} {'differ':>8s} {'max':>8s}")
    bad = False
    tot = np.zeros(2, np.int64)
    for k, (title, (nx, ny, gres, kind)) in enumerate(scenes):
        sc = make_scene(rng, nx, ny, gres, kind)
        lut_c, wvp_c, qai_c, tg_c, ms_c = run_force_scene(f"scene{k}", sc)
        (lut_z, wvp_z, qai_z, tg_z, w_avg, n), ms_z = run_zeit_scene(sc)
        n_valid = int((wvp_c > 0).sum())
        n_, e, one, more, mx = ulp_compare(lut_c, lut_z)
        print(f"{title + ': LUT':34s} {n_:9d} {e:12d} {one + more:8d} {mx:6d}ulp")
        bad |= more > 0
        tot += (n_, e)
        for nm, a, b in (("wvp", wvp_c, wvp_z), ("qai", qai_c, qai_z), ("gas transmittance", tg_c, tg_z)):
            n_, e, d = exact(a, b)
            mx = int(np.abs(a.astype(int) - b.astype(int)).max())
            print(f"{title + ': ' + nm:34s} {n_:9d} {e:12d} {d:8d} {mx:8d}")
            bad |= d > 0
            tot += (n_, e)
        print(f"{'':34s} wvp > 0 in {n_valid} of {len(wvp_c)} pixels, {n} blocks retrieved, mean {w_avg:.4f} cm; "
              f"FORCE {ms_c[0]:.1f} ms + {ms_c[1]:.1f} ms, Zeit {ms_z:.1f} ms (all, with the LUT and the conversions)")
        if k == 4:  # FORCE with 4 threads: the effect of the warm start by thread
            _, wvp4, _, _, _ = run_force_scene(f"scene{k}_t4", sc, threads=4)
            d = np.abs(wvp4.astype(int) - wvp_c.astype(int))
            print(f"{'':34s} FORCE with 4 threads vs its 1 thread: {int((d > 0).sum())} of {len(d)} pixels differ, "
                  f"largest {int(d.max())}/1000 cm, mean {d[d > 0].mean() if (d > 0).any() else 0:.1f}/1000 cm")
    ok = check_lut_files(rng, args.scale)
    bad |= not ok
    print(f"{'total (scenes, bit for bit)':34s} {tot[0]:9d} {tot[1]:12d} {tot[0] - tot[1]:8d}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
