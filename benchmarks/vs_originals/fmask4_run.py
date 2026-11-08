"""The end-to-end Fmask 4.7 step of the port (``zeit_core::fmask::run``: the DN to TOA math, the chain of ``autoFmask.m`` for
Sentinel-2 and the QAI flags) against the original .m files run in GNU Octave 11.3 (image 2.20, statistics) on synthetic
scenes of a fixed seed: with and without a DEM (with and without slope and aspect, with NaN in the DEM), with and without
GSWO, a normal scene (clouds, their shadows, thin cloud, cirrus, snow, urban surface, water, no data), a scene almost all
cloud (no more than 40000 clear pixels: the ``pshadow = ~pcloud`` of ``autoFmask.m:221``), a scene of more than 90 % of cloud
(``MatchCloudShadow`` skips the matching and sets the shadow where there is none), the snow scene.

The Octave side is ``fmask4/fm_run.m``: a replica, calling the real .m files in the order and with the arguments of
``autoFmask.m`` 135-251, from the TOA bands on (it reads no files of the product), and a literal replica of the DN to TOA math
of ``nd2toarbt_msi2.m`` / ``nd2toarbt_msi.m`` on arrays. Shims: those of ``fmask4.py`` (``imresize`` box, ``strel('disk')``)
and of ``fmask4_shadow.py`` (``bwlabeln``, ``rng``, ``randperm``). Rust is run with ``compat = True``, which is the original
(the two oddities of the scenes with little clear sky and the single-band fill of the missing pixels). The DEM, slope and
aspect are the same on both sides. Run::

    python benchmarks/vs_originals/fmask4_run.py [--octave PATH] [--scenes N,A,...] [--timing]

``cs_final`` and every mask must agree exactly (exit status 1 otherwise). With ``compat = False`` the harness checks that the
cloud, snow and water are the same and that the shadow is gone exactly in the two oddities. ``--timing`` times Rust on
5490 x 5490. Run with ``OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8``.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import no_power_throttling, workdir

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zeit import _core  # noqa: E402

import fmask4  # noqa: E402  (the shims of Octave)
import fmask4_detect as det  # noqa: E402  (the scenes)
from fmask4_shadow import Cases, Report, rd  # noqa: E402

fr = _core.fmask_run
HERE = Path(__file__).resolve().parent
DATA = workdir("fmask4_run")
FMASK_SRC = r"C:\Users\eduar\Documents\dev\Fmask4"
OCTAVE = r"C:\Users\eduar\AppData\Local\Programs\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe"
F32 = np.float32
ZEN, AZI = 35.0, 150.0

# name: (detect scene, extra) ; the detect scenes are those of fmask4_detect.SCENES, plus the ones below
SCENES = {
    "N": dict(base="D", shadows=True, topo=True),                    # 1000 x 1000, DEM, GSWO
    "A": dict(base="A", shadows=True, topo=True, dem_nan=True),      # DEM with NaN, GSWO
    "A_nt": dict(base="A", shadows=True, topo=False),                # DEM, no slope and aspect
    "B": dict(base="B", shadows=True, topo=False),                   # no DEM, no GSWO
    "G": dict(base="G", shadows=True, topo=False),                   # no DEM, GSWO of low occurrence
    "F": dict(base="F", shadows=True, topo=True),                    # DEM, GSWO all zero
    "C": dict(base="C", shadows=False, topo=True),                   # 75 % cloud: sum_clr <= 40000
    "E": dict(base="E", shadows=True, topo=False),                   # 60 % snow
    "V": dict(base=None, shadows=True, topo=False, big=dict(rows=1000, cols=1000, seed=21, cloud=0.93, thin=0.0, cirr=0.0, snow=0.0, urban=0.01, water=0.02, dem=False, gs=None)),  # > 90 % cloud
    "W": dict(base=None, shadows=True, topo=True, big=dict(rows=1000, cols=1000, seed=22, cloud=0.14, thin=0.06, cirr=0.04, snow=0.05, urban=0.06, water=0.10, dem=True, gs="good", tall_dem=True)),
}
DEFAULT = "N,A,A_nt,B,G,F,C,E,V"


def shift(m, dy, dx):
    out = np.zeros_like(m)
    rows, cols = m.shape
    ys, yd = (slice(max(-dy, 0), rows - max(dy, 0)), slice(max(dy, 0), rows - max(-dy, 0)))
    xs, xd = (slice(max(-dx, 0), cols - max(dx, 0)), slice(max(dx, 0), cols - max(-dx, 0)))
    out[yd, xd] = m[ys, xs]
    return out


def add_shadows(sc):
    """Dark patches where the thick clouds' shadows fall: the cloud mask moved along the sun (to the north-west for
    azimuth 150), by 30 pixels in the left half of the scene and 55 in the right half (two heights)."""
    rows, cols = sc["blue"].shape
    cl = (sc["blue"] > 3300) & (sc["swir1"] > 2400) & (sc["blue"] > -9999)
    mask = sc["blue"] > -9999
    tazi = np.radians(AZI - 90)
    sh = np.zeros_like(cl)
    for lo, hi, i in ((0, cols // 2, 30), (cols // 2, cols, 55)):
        part = np.zeros_like(cl)
        part[:, lo:hi] = cl[:, lo:hi]
        sh |= shift(part, -int(round(i * np.sin(tazi))), -int(round(i * np.cos(tazi))))
    sh &= ~cl & mask
    for k, f in (("nir", 0.45), ("swir1", 0.4), ("swir2", 0.4), ("blue", 0.6), ("green", 0.6), ("red", 0.6)):
        sc[k] = np.where(sh, sc[k] * F32(f), sc[k]).astype(F32)
    return int(sh.sum())


def make_run_scene(name):
    p = dict(SCENES[name])
    if p.get("big"):
        b = dict(p["big"])
        sc = det.make_scene(b.pop("rows"), b.pop("cols"), b.pop("seed"), **b)
    else:
        sc = det.scene_of(p["base"])
    rows, cols = sc["blue"].shape
    sc["nshadow"] = add_shadows(sc) if p["shadows"] else 0
    sc["slope"] = sc["aspect"] = None
    if sc["dem"] is not None and p["topo"]:
        gy, gx = np.gradient(sc["dem"].astype(np.float64), 20.0)
        sc["slope"] = np.degrees(np.arctan(np.hypot(gx, gy))).astype(F32)
        sc["aspect"] = (np.degrees(np.arctan2(-gx, gy)) % 360.0).astype(F32)
    if sc["dem"] is not None and p.get("dem_nan"):
        sc["dem"] = sc["dem"].copy()
        sc["dem"][10:14, 20:26] = np.nan
        sc["dem"][rows - 40:rows - 30, 100:110] = np.nan
    u = np.linspace(0, 1, cols)[None, :] * np.ones((rows, 1))
    v = np.linspace(0, 1, rows)[:, None] * np.ones((1, cols))
    sc["vza"] = np.round(100 * (2.5 + 6.0 * u + 0.8 * v)).astype(F32)
    sc["vaa"] = np.round(100 * (95 + 25 * v - 8 * u)).astype(F32)
    return sc


def write_scene(c, name, sc):
    rows, cols = sc["blue"].shape
    for k in ("blue", "green", "red", "nir", "swir1", "swir2", "cirrus", "sb", "sg", "sr", "b7", "b8", "vza", "vaa"):
        c.put(name, k, sc[k])
    hasdem, hasgs, hastopo = sc["dem"] is not None, sc["gs"] is not None, sc["slope"] is not None
    if hasdem:
        c.put(name, "dem", sc["dem"])
    if hasgs:
        c.put(name, "gs", sc["gs"])
    if hastopo:
        c.put(name, "slope", sc["slope"])
        c.put(name, "aspect", sc["aspect"])
    c.lines.append(f"scene {name} {rows} {cols} {int(hasdem)} {int(hasgs)} {int(hastopo)} {ZEN!r} {AZI!r}")


def flat(a):
    return np.ascontiguousarray(a).ravel()


def run_rust(sc, compat, keep=True):
    rows, cols = sc["blue"].shape
    o = lambda k: None if sc.get(k) is None else flat(sc[k])
    t0 = time.perf_counter()
    r = fr.run_fmask(rows, cols, flat(sc["blue"]), flat(sc["green"]), flat(sc["red"]), flat(sc["nir"]), flat(sc["swir1"]), flat(sc["swir2"]), flat(sc["cirrus"]),
                     flat(sc["sb"]), flat(sc["sg"]), flat(sc["sr"]), flat(sc["vza"]), flat(sc["vaa"]), ZEN, AZI, compat, keep,
                     b7=flat(sc["b7"]), b8=flat(sc["b8"]), dem=o("dem"), slope=o("slope"), aspect=o("aspect"), gswo=o("gs"))
    return r, time.perf_counter() - t0


def compare_scene(d, name, sc, rep, tim):
    rows, cols = sc["blue"].shape
    err = Path(d) / f"{name}.err"
    if err.exists():
        rep.note(f"[{name}] Octave error", 0, err.read_text(), True)
        return
    r, dt = run_rust(sc, True)
    tim[name] = dt
    g = lambda ext, dt_=np.uint8: rd(d, name, ext, dt_)
    sum_clr = g("sumclr", np.float64)[0]
    path = g("path")[0]
    rep.note(f"[{name}] sum_clr / path", 1, f"Octave {sum_clr:.0f} clear pixels, {'matching' if path else 'few clear pixels (pshadow = ~pcloud)'}; Rust {r['sum_clr']}, few_clear {r['few_clear']}", r["sum_clr"] != sum_clr or r["few_clear"] != (path == 0))
    npx = rows * cols
    skipped_o = None
    if path:
        sim = g("sim", np.float64)
        skipped_o = bool(sim.size == 1 and sim[0] == -1)
        rep.note(f"[{name}] MatchCloudShadow skipped", 1, f"Octave {skipped_o}, Rust {r['match_skipped']}", skipped_o != r["match_skipped"])
    rep.note(f"[{name}] scene", npx, f"{int(g('mask').sum())} observed, {int(g('pcloud0').sum())} cloud, {int(g('pshadow').sum())} shadow (after buffer), {int(g('psnow').sum())} snow; {sc['nshadow']} dark shadow pixels put")
    pairs = [("mask", "mask"), ("psnow0", "psnow0"), ("water", "water"), ("waterall", "water_all"), ("idplcd", "idplcd"), ("cloud", "cloud"), ("pfpl", "pfpl"),
             ("pcloud_eroded", "pcloud_eroded")]
    if g("idlndempty")[0] == 0:
        pairs.append(("idlnd", "idused"))
    if path:
        pairs += [("pshadow_pot", "pshadow_potential"), ("match_cloud", "match_cloud"), ("match_shadow", "match_shadow")]
    pairs += [("pcloud0", "pcloud"), ("pshadow", "pshadow"), ("psnow", "psnow"), ("water", "water")]
    for ox, rk in pairs:
        rep.exact(f"[{name}] {ox}", r[rk], g(ox))
    # the buffered cloud is not an output of Rust (it is only stacked): the stack is the proof
    rep.exact(f"[{name}] cs_final", r["cs_final"], g("cs_final"))
    clr = 100 * (1 - g("pcloud").sum() / g("mask").sum())
    rep.note(f"[{name}] clr_pct", 1, f"Octave {clr!r}, Rust {r['clr_pct']!r}", abs(clr - r["clr_pct"]) > 1e-12)
    if path and "topo" in r:
        rep.note(f"[{name}] SCS+C applied", 1, f"c = {r['topo'][0]:.4g} (NIR), {r['topo'][1]:.4g} (SWIR)")
    # compat = False: the same cloud, snow and water; the shadow gone in the oddities
    n, _ = run_rust(sc, False, keep=False)
    for k in ("pcloud", "psnow", "water", "mask"):
        rep.exact(f"[{name}] compat=False {k} equals compat=True", n[k], r[k])
    odd = r["few_clear"] or r["match_skipped"]
    if odd:
        rep.note(f"[{name}] compat=False has no shadow", int(n["pshadow"].sum()), "(the oddity)", n["pshadow"].any() or (n["cs_final"] == 2).any())
        under = np.where(n["psnow"] == 1, 3, (n["water"] == 1).astype(np.uint8)).astype(np.uint8)
        rep.exact(f"[{name}] compat=False cs_final = compat=True with the shadow replaced by the class below", n["cs_final"], np.where(r["cs_final"] == 2, under, r["cs_final"]))
    else:
        rep.exact(f"[{name}] compat=False cs_final equals compat=True (no oddity)", n["cs_final"], r["cs_final"])
    return r


# ---------------------------------------------------------------------------------------------------------------
# the DN to TOA math
# ---------------------------------------------------------------------------------------------------------------
OFFS = {1: -1000.0, 2: -900.0, 3: -1000.0, 8: -500.0, 11: -1000.0, 12: -1000.0, 10: -50.0, 6: -1.0, 7: -1000.0}
BAND_IDS = [1, 2, 3, 8, 11, 12, 10, 6, 7]
NAT = [2, 2, 2, 1, 1, 1, 1 / 3, 1, 2]


def build_toa_cases(c, rng):
    cases = {}
    for tid, (rows, cols, kind, q) in {"t1": (60, 90, "b04", 10000.0), "t2": (90, 60, "legacy", 10000.0), "t3": (60, 60, "b04", 5000.0), "t4": (90, 90, "b04z", 10000.0)}.items():
        dns = []
        for k in range(9):
            r_, c_ = round(rows * NAT[k]), round(cols * NAT[k])
            hi = 4000 if k else 4000
            dn = rng.integers(1, hi, (r_, c_)).astype(np.uint16)
            dn[rng.random((r_, c_)) < 0.01] = 0
            dn[r_ // 4: r_ // 4 + 3 * round(NAT[k] + 0.99), c_ // 3: c_ // 3 + 2] = 0       # a block of no data
            if k < 3:
                dn[:2, :2] = 65535
                dn[r_ - 2:, c_ - 2:] = 65535
                dn[r_ // 2, c_ // 2] = 65535
            if k == 3:
                dn[1:3, 4:6] = 0
            if k == 6:
                dn[2, 3] = 0
            dns.append(dn)
            c.put(tid, f"d{k + 1}", dn)
        cases[tid] = (rows, cols, kind, q, dns)
        c.lines.append(f"toa {tid} {rows} {cols} {kind} {q!r}")
    return cases


def compare_toa(d, tid, case, rep):
    rows, cols, kind, q, dns = case
    err = Path(d) / f"{tid}.err"
    if err.exists():
        rep.note(f"[{tid}] Octave error", 0, err.read_text(), True)
        return
    offs = None if kind == "legacy" else np.array([OFFS[i] for i in BAND_IDS])
    if kind == "b04z":
        offs = np.zeros(9)
    for compat in (True, False):
        t = fr.dn_to_toa20(rows, cols, *[flat(x) for x in dns], compat, offs, q)
        if compat:
            for k, key in enumerate(("blue", "green", "red", "nir", "swir1", "swir2", "cirrus", "b7", "b8")):
                want = rd(d, tid, f"t{k + 1}", F32)
                rep.exact(f"[{tid}] {kind} q={q:g} band {k + 1} ({key}), {int(np.isnan(want).sum())} NaN, {int((want == -9999).sum())} at -9999", np.where(np.isnan(t[key]), -12345, t[key]), np.where(np.isnan(want), -12345, want))
            for k, key in enumerate(("satu_blue", "satu_green", "satu_red")):
                want = rd(d, tid, ("sb", "sg", "sr")[k], np.uint8)
                rep.exact(f"[{tid}] {key} ({int(want.sum())} flagged)", t[key], want)
            ref = t
        else:
            # without compat: the pixel with no data in any band is -9999 in every band; elsewhere the same
            miss = np.zeros(rows * cols, bool)
            for key in ("blue", "green", "red", "nir", "swir1", "swir2", "cirrus", "b7", "b8"):
                miss |= np.isnan(ref[key]) | (ref[key] == -9999)
            ok = True
            for key in ("blue", "green", "red", "nir", "swir1", "swir2", "cirrus", "b7", "b8"):
                ok &= bool((t[key][miss] == -9999).all()) and not np.isnan(t[key]).any() and bool(np.array_equal(t[key][~miss], ref[key][~miss]))
            rep.note(f"[{tid}] compat=False: -9999 in all bands where any is missing ({int(miss.sum())} px), the rest as compat", rows * cols, "ok" if ok else "DIFFERS", not ok)


# ---------------------------------------------------------------------------------------------------------------
def tile_scene(size=5490, tile=610):
    reps = size // tile
    sc = det.make_scene(tile, tile, 31, cloud=0.12, thin=0.08, cirr=0.06, snow=0.05, urban=0.05, water=0.08, dem=True, gs="good")
    add_shadows(sc)
    t = lambda a: np.ascontiguousarray(np.tile(a, (reps, reps)))
    out = {k: t(v) for k, v in sc.items() if isinstance(v, np.ndarray)}
    out["gs"] = t(sc["gs"])
    gy, gx = np.gradient(out["dem"].astype(np.float64), 20.0)
    out["slope"] = np.degrees(np.arctan(np.hypot(gx, gy))).astype(F32)
    out["aspect"] = (np.degrees(np.arctan2(-gx, gy)) % 360.0).astype(F32)
    n = out["blue"].shape[0]
    u = np.linspace(0, 1, n)[None, :] * np.ones((n, 1))
    v = np.linspace(0, 1, n)[:, None] * np.ones((1, n))
    out["vza"] = np.round(100 * (2.5 + 6.0 * u + 0.8 * v)).astype(F32)
    out["vaa"] = np.round(100 * (95 + 25 * v - 8 * u)).astype(F32)
    return out


def timing():
    n = 5490
    print(f"\n{n} x {n} timings (RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}):")
    t0 = time.perf_counter()
    sc = tile_scene(n)
    print(f"  (scene made in {time.perf_counter() - t0:.1f} s)")
    for compat in (False,):
        best = None
        for _ in range(2):
            r, dt = run_rust(sc, compat, keep=False)
            best = dt if best is None else min(best, dt)
        print(f"  run_fmask (DEM, slope, aspect, GSWO; compat={compat}) {best:7.2f} s: sum_clr {r['sum_clr']}, cloud {int(r['pcloud'].sum())} px, shadow {int(r['pshadow'].sum())} px, skipped {r['match_skipped']}")
    for f in (1, 2):
        t0 = time.perf_counter()
        q = fr.to_qai(n, n, f, 20.0, r["mask"], r["pcloud"], r["water"], r["psnow"], r["pshadow"], 300.0, 8)
        print(f"  to_qai {20 // f} m ({q.size} px) {time.perf_counter() - t0:7.2f} s; cloud state 2: {int(((q >> 1) & 3 == 2).sum())}, state 1: {int(((q >> 1) & 3 == 1).sum())}")
    # the DN to TOA
    rng = np.random.default_rng(5)
    dn = [rng.integers(1, 5000, (round(n * f), round(n * f))).astype(np.uint16) for f in NAT]
    offs = np.array([OFFS[i] for i in BAND_IDS])
    t0 = time.perf_counter()
    fr.dn_to_toa20(n, n, *[flat(x) for x in dn], False, offs, 10000.0)
    print(f"  dn_to_toa20 {time.perf_counter() - t0:7.2f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave", default=OCTAVE)
    ap.add_argument("--fmask-src", default=FMASK_SRC)
    ap.add_argument("--scenes", default=DEFAULT, help="of " + ",".join(SCENES) + "; '' for none")
    ap.add_argument("--no-toa", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    a = ap.parse_args()
    no_power_throttling()
    print(f"zeit._core: {_core.__file__}")
    if a.timing_only:
        timing()
        return 0
    for f in DATA.glob("*"):
        if f.is_file():
            f.unlink()
    c = Cases(DATA)
    rng = np.random.default_rng(20261108)
    toa = {} if a.no_toa else build_toa_cases(c, rng)
    names = [x for x in a.scenes.split(",") if x]
    scenes = {}
    for nm in names:
        t0 = time.perf_counter()
        scenes[nm] = make_run_scene(nm)
        write_scene(c, nm, scenes[nm])
        print(f"scene {nm}: made in {time.perf_counter() - t0:.1f} s, {scenes[nm]['blue'].shape}")
    c.write()
    shims = fmask4.make_shims(DATA, a.octave)
    for f in (HERE / "fmask4" / "shims_shadow").glob("*.m"):
        shutil.copy(f, shims)
    env = dict(os.environ, FMASK_DIR=str(DATA), FMASK_SHIMS=str(shims), FMASK_SRC=str(a.fmask_src), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    t0 = time.perf_counter()
    r = subprocess.run([a.octave, "--no-gui", str(HERE / "fmask4" / "fm_run.m")], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"octave failed:\n{r.stdout}\n{r.stderr}")
    print(f"Octave ran {len(c.lines)} cases in {time.perf_counter() - t0:.1f} s")
    for l in r.stdout.splitlines():
        if l.startswith(("  scene", "ERROR")):
            print(l)
    rep = Report()
    rep.bad += sum(l.startswith("ERROR") for l in r.stdout.splitlines())
    for tid, case in toa.items():
        compare_toa(DATA, tid, case, rep)
    tim = {}
    for nm in names:
        compare_scene(DATA, nm, scenes[nm], rep, tim)
    rep.show()
    nv = sum(x[1] for x in rep.rows)
    print(f"\n{len(rep.rows)} comparisons, {nv} values; {rep.bad} comparisons with differences")
    for k, v in tim.items():
        print(f"  Rust run_fmask {k:6s} {v:7.2f} s")
    if a.timing:
        timing()
    return 1 if rep.bad else 0


if __name__ == "__main__":
    sys.exit(main())
