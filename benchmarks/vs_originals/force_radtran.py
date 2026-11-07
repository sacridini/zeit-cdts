"""The radiative transfer pieces of zeit-core (``zeit_core::radtran``, a port of FORCE 3.11.00's ``radtran-ll.c``,
``table-ll.c`` (``wavelength``, ``E0``), ``gas-ll.c`` (``ozone_amount``) and ``sunview-ll.c`` (``localtime2gmt``,
``standard_sunzenith``)) against FORCE's own C functions on a large grid of inputs. The sun position (``sunpos`` and
the zenith in ``standard_sunzenith``) is not a port: it is ``zeit_core::solpos``, a clean-room implementation from
Meeus and NOAA's documentation, so these two are reported as a difference in degrees, not bit for bit.

The C side is ``force_radtran/`` (``make_harness.sh``, WSL): FORCE's radtran-ll.c, table-ll.c, sun-cl.c and date-cl.c
compiled as they are with FORCE's flags (gcc -O3, no -ffast-math), the three functions of gas-ll.c and sunview-ll.c that
cannot be compiled alone copied verbatim; it evaluates a function on float32 columns read from a file. Zeit's side is
``zeit._core.radtran.radtran_eval`` (Windows, UCRT libm; the harness runs on glibc 2.39). Inputs are float32 (integers as
whole floats), the outputs float32; the comparison is of the bits, and of the distance in float32 ulps::

    python benchmarks/vs_originals/force_radtran.py [--scale 1.0]

Prints, per function, the number of values compared, how many are equal bit for bit (NaN equal to NaN), how many differ
by 1 ulp, by more, and the largest distance. Exits with 1 if anything but the sun position differs by more than 1 ulp. Then, for
the sun position, the mean and maximum difference in degrees (zenith by day and by night, azimuth by day).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_radtran")
EXE = "$HOME/force-ref-work/radtran/radtran_harness"
F32 = np.float32
# not bit for bit: Zeit's sun position is a clean-room one (Meeus and NOAA's documentation), FORCE's is another method
SUN_FUNCTIONS = ("sunpos", "standard_sunzenith")


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_radtran' / 'make_harness.sh')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness build failed: {r.stdout}{r.stderr}")


def run_force(name, cols, nout):
    n = len(cols[0])
    fin, fout = DATA / f"{name}.in", DATA / f"{name}.out"
    with open(fin, "wb") as f:
        f.write(np.array([n], np.int32).tobytes())
        for c in cols:
            f.write(np.ascontiguousarray(c, F32).tobytes())
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"{EXE} {name} {_wsl(fin)} {_wsl(fout)}"], capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed on {name}: {r.stderr}")
    return np.fromfile(fout, F32).reshape(nout, n), float(r.stdout.strip().splitlines()[-1])


def ordered(a):
    """float32 bits as integers that increase with the value (-0 and +0 are 1 apart, as ulps: treated as equal)."""
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def compare(name, a, b):
    """(values, bit-equal, 1 ulp, more, max ulp) of two float32 arrays; NaN == NaN; inf == inf."""
    a, b = np.asarray(a, F32).ravel(), np.asarray(b, F32).ravel()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    same = (a.view(np.uint32) == b.view(np.uint32)) | (nan_a & nan_b)
    d = np.abs(ordered(a) - ordered(b))
    d = np.where(nan_a & nan_b, 0, d)
    d = np.where(nan_a ^ nan_b, 1 << 40, d)
    exact = same | (d == 0)
    one = (~exact) & (d == 1)
    more = d > 1
    return a.size, int(exact.sum()), int(one.sum()), int(more.sum()), int(d.max()) if a.size else 0


def make_grids(scale, seed=7):
    rng = np.random.default_rng(seed)
    n = int(200_000 * scale)
    U = lambda lo, hi, m=n: rng.uniform(lo, hi, m).astype(F32)  # noqa: E731
    G = {}

    wvl = np.concatenate([U(0.3, 2.6, n), np.linspace(0.41, 2.4, 1991, dtype=F32)])
    G["molecular_optical_depth"] = ([wvl], 1)
    z, hp = U(-0.5, 6.0), U(0.3, 4.0)
    G["aod_elev_factor"] = ([z, hp], 1)
    G["mod_elev_factor"] = ([np.concatenate([z, np.zeros(1, F32), np.linspace(-1, 9, 5001, dtype=F32)])], 1)
    G["mod_elev_scale"] = ([U(0, 0.6), U(0.3, 1.2), U(0.3, 1.2)], 1)
    G["aod_elev_scale"] = ([U(0, 2), U(0.3, 1.2), U(0.3, 1.2)], 1)
    G["optical_depth"] = ([U(0, 2), U(0, 0.6)], 1)

    aod, mod = U(0.0, 2.0), U(0.0, 0.6)
    od = (aod + mod).astype(F32)
    ms, mv = U(0.1, 1.0), U(0.1, 1.0)
    G["scatt_transmitt"] = ([aod, mod, od, ms, mv], 7)
    multi = (rng.random(n) < 0.5).astype(F32)
    aod2 = U(0.001, 2.0)
    od2 = (aod2 + mod).astype(F32)
    G["path_ref"] = ([multi, U(0.5, 1.0), aod2, mod, od2, U(0, 3), U(0.75, 1.5), U(0, 1), U(0, 1), ms, mv], 1)
    G["sphere_albedo"] = ([aod, mod, od], 1)
    G["env_weight_aerosol"] = ([U(0, 30)], 1)
    G["env_weight_molecular"] = ([U(0, 30)], 1)
    G["env_weight"] = ([U(0.001, 2), U(0.001, 0.6), U(0, 1), U(0, 1)], 1)
    G["backscatter"] = ([ms, mv, U(0, 2 * np.pi), U(0, 2 * np.pi)], 1)
    G["phase_molecular"] = ([U(-1, 1)], 1)
    G["phase_aerosol"] = ([U(-1, 1), U(0, 1), U(0, 1), U(1.0, 2.0), U(1.0, 2.0), U(0, 1), U(0, 1)], 1)

    # every band, on a grid of paths
    nb = 56
    w = np.concatenate([np.linspace(0, 8, 33), [0.01, 0.1, 6.999, 7.0, 7.01]]).astype(F32)
    m = np.linspace(0.25, 1.0, 17).astype(F32)
    bb, ww, mm = (a.ravel().astype(F32) for a in np.meshgrid(np.arange(nb), w, m, indexing="ij"))
    G["wvp_transmitt"] = ([ww, mm, bb], 1)
    o = np.concatenate([np.linspace(0.0, 0.8, 33), [0.001, 0.3, 0.345]]).astype(F32)
    bb, oo, mm = (a.ravel().astype(F32) for a in np.meshgrid(np.arange(nb), o, m, indexing="ij"))
    G["ozone_transmitt"] = ([oo, mm, bb], 1)
    G["wavelength"] = ([np.arange(nb, dtype=F32)], 1)
    G["e0"] = ([np.arange(nb, dtype=F32)], 1)

    G["gas_transmitt"] = ([U(0, 1), U(0, 1), U(0.8, 1), U(0.8, 1)], 1)
    G["fresnel_reflection"] = ([np.concatenate([U(0, np.pi / 2), np.zeros(1, F32), [np.pi / 2], [1e-4]]).astype(F32)], 1)
    G["illumin"] = ([U(0, 1), U(0, 1), U(0, 1), U(0, 1), U(0, 2 * np.pi), U(0, 2 * np.pi)], 1)
    G["c_factor_emp"] = ([np.concatenate([U(-1, 3), np.zeros(3, F32)]), np.concatenate([U(-1, 3), [0, 1, -1]]).astype(F32)], 1)
    G["c_factor_com"] = ([U(0, 1), U(0, 5), U(0.1, 1)], 1)
    G["f_factor"] = ([U(0, 1), np.concatenate([U(0, 1, n - 1), [0]]).astype(F32)], 1)

    G["ozone_amount"] = ([np.concatenate([U(-180, 180), [0, 0, 10, -10]]).astype(F32),
                          np.concatenate([U(-90, 90), [0, 1e-3, 0, 0]]).astype(F32),
                          rng.integers(1, 366, n + 4).astype(F32)], 1)

    year = rng.integers(1984, 2031, n).astype(F32)
    doy = rng.integers(1, 366, n).astype(F32)
    G["localtime2gmt"] = ([year, doy, U(-3, 30), U(-180, 180)], 7)
    lat = U(-80, 80)
    lon = U(-180, 180)
    G["standard_sunzenith"] = ([year, doy, rng.integers(-12, 13, n).astype(F32) * (rng.random(n) < 0.2), lat, lon], 1)
    G["sunpos"] = ([U(-90, 90), U(-180, 180), year, doy, rng.integers(0, 24, n).astype(F32),
                    rng.integers(0, 60, n).astype(F32), rng.integers(0, 60, n).astype(F32),
                    rng.integers(-12, 13, n).astype(F32) * (rng.random(n) < 0.2)], 2)
    # beyond about 82 degrees of latitude the overpass time of standard_sunzenith is NaN, and the C turns it into INT_MIN
    # hours (x86-64 cvttsd2si), which Zeit reproduces
    la = np.concatenate([np.linspace(80, 90, 201), -np.linspace(80, 90, 201)]).astype(F32)
    k = len(la)
    G["standard_sunzenith@polar"] = ([np.full(k, 2021, F32), np.full(k, 172, F32), np.zeros(k, F32), la,
                                      np.linspace(-180, 180, k).astype(F32)], 1)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of random inputs")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    no_power_throttling()
    from zeit import _core

    if not args.skip_build:
        build_harness()
    grids = make_grids(args.scale)
    print(f"{'function':26s} {'inputs':>8s} {'values':>9s} {'bit for bit':>12s} {'1 ulp':>8s} {'> 1 ulp':>8s} {'max ulp':>8s}")
    tot = np.zeros(4, np.int64)
    bad = False
    sun = []
    for name, (cols, nout) in grids.items():
        cols = [np.ascontiguousarray(c, F32) for c in cols]
        fname = name.split("@")[0]
        ref, _ = run_force(fname, cols, nout)
        out = _core.radtran.radtran_eval(fname, *cols)
        if fname in SUN_FUNCTIONS:
            sun.append((name, ref, out))
            continue
        res = [compare(name, ref[k], out[k]) for k in range(nout)]
        n, e, one, more = (sum(r[i] for r in res) for i in range(4))
        mx = max(r[4] for r in res)
        tot += (n, e, one, more)
        bad |= more > 0
        print(f"{name:26s} {len(cols[0]):8d} {n:9d} {e:12d} {one:8d} {more:8d} {mx:8d}")
        if more:
            k = next(k for k in range(nout) if res[k][3])
            d = np.abs(ordered(ref[k]) - ordered(out[k])) > 1
            d &= ~(np.isnan(ref[k]) & np.isnan(out[k]))
            idx = np.flatnonzero(d)[:3]
            for i in idx:
                print(f"    e.g. output {k}: inputs {[float(c[i]) for c in cols]} FORCE {ref[k][i]!r} zeit {out[k][i]!r}")
    print(f"{'total (bit for bit)':26s} {'':8s} {tot[0]:9d} {tot[1]:12d} {tot[2]:8d} {tot[3]:8d}")
    print("\nsun position (clean-room, not bit for bit): difference to FORCE in degrees")
    print(f"{'function / output':30s} {'subset':>6s} {'values':>9s} {'mean':>10s} {'max':>10s} {'> 0.01':>8s}")
    for name, ref, out in sun:
        zen_ref = ref[0]
        for k, what in enumerate(("zenith", "azimuth")[:len(ref)]):
            d = np.abs(ref[k].astype(np.float64) - out[k].astype(np.float64))
            if k == 1:  # azimuth wraps around 360
                d = np.minimum(d, 360.0 - d)
            both_nan = np.isnan(ref[k]) & np.isnan(out[k])
            for sub, m in (("day", zen_ref < 90), ("night", zen_ref >= 90)):
                m = m & ~both_nan & np.isfinite(d)
                if not m.any():
                    continue
                if k == 1 and sub == "night":
                    continue
                print(f"{name + ' ' + what:30s} {sub:>6s} {int(m.sum()):9d} {d[m].mean():10.6f} {d[m].max():10.6f} "
                      f"{int((d[m] > 0.01).sum()):8d}")
            nan_mis = int((np.isnan(ref[k]) ^ np.isnan(out[k])).sum())
            if nan_mis:
                print(f"{name + ' ' + what:30s} NaN in only one of the two: {nan_mis}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
