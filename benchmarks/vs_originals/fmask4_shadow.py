"""The cloud shadow step of the Fmask 4.7 port (``zeit_core::fmask::shadow``: ``DetectPotentialCloudShadow``,
``stratiedSampleHanlder``, ``ProjectDEM2Plane``, ``getRealCloudPositionS2``, ``MatchCloudShadow`` and ``BufferMasks``, the
Sentinel-2 route) against the original .m files run in GNU Octave 11.3 (image 2.20, statistics), on synthetic scenes of a
fixed seed (clouds of several sizes with the shadow the sun and the height project, over land and water, with and without
a DEM, a cloud on the border, a scene with more than 90 % of cloud).

The Octave side is ``fmask4/fm_shadow.m`` (the .m files of Fmask are NOT edited) with the shims of
``fmask4/shims_shadow/``: ``bwlabeln`` (Octave numbers the objects differently from MATLAB's order of the first pixel by
columns, and in MatchCloudShadow the order of the objects changes the result), ``rng`` and ``randperm`` (MATLAB's generator
cannot be reproduced: both sides use the same 64-bit LCG with a partial Fisher-Yates shuffle, ``fmask_randperm``; a
deviation from MATLAB, as the sun of phase 23g was). ``strel('square')`` and ``knnsearch``/``pdist2`` run as they are
(``strel('disk')`` and ``imresize`` are not used by these functions). The values the .m keeps inside (the ``c`` of the
SCS+C, the samples, and at every match ``record_base_h``, ``record_thresh`` and ``record_base_h_near``) are recorded by
copies of the two .m files made here with one added line each (``fm_rec_*``), whose outputs must equal the originals'. The
inputs of MatchCloudShadow are the Octave outputs of DetectPotentialCloudShadow (so the two steps are compared apart)::

    python benchmarks/vs_originals/fmask4_shadow.py [--octave PATH] [--scenes A,B,C,D,E] [--huge] [--timing]

Masks (potential shadow, matched shadow, buffers) and the integer values must agree exactly (exit status 1 otherwise);
the continuous ones are reported (the largest difference in ulps, or relative). ``--timing`` times the Rust functions on
5490 x 5490. Run with ``OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8``.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import no_power_throttling, workdir

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zeit import _core  # noqa: E402

sh = _core.fmask_shadow
HERE = Path(__file__).resolve().parent
DATA = workdir("fmask4_shadow")
FMASK_SRC = r"C:\Users\eduar\Documents\dev\Fmask4"
OCTAVE = r"C:\Users\eduar\AppData\Local\Programs\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe"
F32 = np.float32


# ---------------------------------------------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------------------------------------------
def ordered(a):
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulps(a, b):
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    na, nb = np.isnan(a), np.isnan(b)
    bad = int((na != nb).sum())
    ok = ~(na | nb)
    d = np.abs(ordered(a)[ok] - ordered(b)[ok])
    return (int(d.max()) if d.size else 0), int((d > 0).sum()), bad


def ulps64(a, b):
    a, b = np.ascontiguousarray(a, np.float64), np.ascontiguousarray(b, np.float64)
    ia, ib = a.view(np.int64), b.view(np.int64)
    oa = np.where(ia < 0, -(ia & 0x7FFFFFFFFFFFFFFF), ia)
    ob = np.where(ib < 0, -(ib & 0x7FFFFFFFFFFFFFFF), ib)
    d = np.abs(oa - ob)
    return (int(d.max()) if d.size else 0), int((d > 0).sum())


class Report:
    def __init__(self):
        self.rows, self.bad = [], 0

    def exact(self, name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        nd = int((a.ravel() != b.ravel()).sum()) if a.size == b.size else a.size + 1
        self.rows.append((name, a.size, f"{nd} differ" if nd else "equal"))
        self.bad += nd > 0

    def cont32(self, name, a, b, tol=0):
        mx, nd, nan = ulps(a, b)
        self.rows.append((name, np.asarray(a).size, f"max {mx} ulp, {nd} differ, {nan} NaN mismatch"))
        self.bad += (mx > tol) or nan > 0

    def cont64(self, name, a, b, tol=0):
        mx, nd = ulps64(a, b)
        self.rows.append((name, np.asarray(a).size, f"max {mx} ulp (double), {nd} differ"))
        self.bad += mx > tol

    def note(self, name, n, text, bad=False):
        self.rows.append((name, n, text))
        self.bad += bool(bad)

    def show(self):
        for name, n, t in self.rows:
            print(f"{name:46s} {n:>10d}  {t}")


class Cases:
    def __init__(self, d):
        self.d, self.lines = Path(d), []

    def put(self, cid, ext, arr):
        np.ascontiguousarray(arr).tofile(self.d / f"{cid}.{ext}")

    def add(self, op, cid, rows, cols, args=()):
        self.lines.append(f"{op} {cid} {rows} {cols} " + " ".join(str(a) for a in args))

    def write(self):
        (self.d / "cases.txt").write_text("\n".join(self.lines) + "\n")


def rd(d, cid, ext, dtype):
    return np.fromfile(Path(d) / f"{cid}.{ext}", dtype)


# ---------------------------------------------------------------------------------------------------------------
# the Octave side: shims, instrumented copies, run
# ---------------------------------------------------------------------------------------------------------------
def make_shims(d, src):
    """The folder of the shims: ours, and the copies of two .m files with a recorder line added (the originals are untouched)."""
    s = Path(d) / "shims"
    s.mkdir(exist_ok=True)
    for f in (HERE / "fmask4" / "shims_shadow").glob("*.m"):
        (s / f.name).write_bytes(f.read_bytes())
    t = (Path(src) / "MatchCloudShadow.m").read_bytes().decode("latin-1")
    a = "similar_num(cloud_type)=record_thresh;"
    assert t.count(a) == 1
    t = t.replace(a, a + "\n                fm_rec_match(cloud_type_cur, cloud_type, record_base_h, record_thresh, record_base_h_near);")
    t, n = re.subn(r"(function \[[^\]]*\] = MatchCloudShadow)\(", r"\1_rec(", t, count=1)
    assert n == 1
    (s / "MatchCloudShadow_rec.m").write_bytes(t.encode("latin-1"))
    t = (Path(src) / "DetectPotentialCloudShadow.m").read_bytes().decode("latin-1")
    c = "c=c_fitted(1,2)/c_fitted(1,1);"
    assert t.count(c) == 2
    t = t.replace(c, c + " fm_rec_c(c);")
    b = "cos_sita_samples=cos_sita_exclude_cloud(samples_ids);"
    assert t.count(b) == 1
    t = t.replace(b, b + " fm_rec_ids(samples_ids);")
    t, n = re.subn(r"(function masker_shadow = DetectPotentialCloudShadow)\(", r"\1_rec(", t, count=1)
    assert n == 1
    (s / "DetectPotentialCloudShadow_rec.m").write_bytes(t.encode("latin-1"))
    return s


def run_octave(d, octave, shims, src):
    env = dict(os.environ, FMASK_DIR=str(d), FMASK_SHIMS=str(shims), FMASK_SRC=str(src), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    r = subprocess.run([octave, "--no-gui", str(HERE / "fmask4" / "fm_shadow.m")], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"octave failed:\n{r.stdout}\n{r.stderr}")
    errs = [l for l in r.stdout.splitlines() if l.startswith(("ERROR", "ERRDIFF"))]
    times = {l.split()[1]: float(l.split()[2]) for l in r.stdout.splitlines() if l.endswith(" s") and not l.startswith(" ")}
    return times, errs, r.stdout


# ---------------------------------------------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------------------------------------------
def build_prims(c, rng):
    meta = {}
    # the colon operator, in the patterns of the .m: Min:step:Max (double without a DEM, single with one) and min:0.1:max
    for k in range(250):
        elev = np.radians(rng.uniform(15, 75))
        step = 2 * 20 * np.tan(elev)
        b = 200.0 + (0.0 if k % 2 else float(F32(rng.uniform(0, 1500))))
        lim = 12000.0 + (b - 200.0)
        if k % 5 == 0:
            step = float(rng.choice([10.0, 0.1, 17.0, 33.333]))
        c.put(f"rd{k}", "args", np.array([b, step, lim]))
        c.add("range", f"rd{k}", 1, 1, ["double", repr(b), repr(float(step)), repr(lim)])
        b32, l32 = float(F32(b)), float(F32(lim))
        c.add("range", f"rs{k}", 1, 1, ["single", repr(b32), repr(float(step)), repr(l32)])
        meta[f"rd{k}"] = (b, step, lim)
        meta[f"rs{k}"] = (b32, step, l32)
    for k in range(200):
        lo = float(F32(rng.uniform(-0.2, 0.7)))
        hi = float(F32(lo + rng.uniform(0.05, 1.3)))
        c.add("range", f"rc{k}", 1, 1, ["single", repr(lo), repr(0.1), repr(hi)])
        meta[f"rc{k}"] = (lo, 0.1, hi)
    # randperm
    for n, k in [(1, 1), (5, 5), (10, 3), (1000, 1000), (50000, 40000), (300000, 300000)]:
        c.add("randperm", f"rp_{n}_{k}", 1, 1, [n, k])
    # the stratified sampling
    for i, (n, lo, hi, tot) in enumerate([(30000, 0.3, 0.99, 40000.0), (200000, -0.1, 0.95, 40000.0), (5000, 0.2, 0.7, 40000.0), (60000, 0.5, 0.52, 1000.0), (120000, 0.1, 1.0, 40000.0)]):
        v = rng.uniform(lo, hi, n).astype(F32)
        if i == 3:
            v[:10] = F32(0.9)
        c.put(f"st{i}", "in", v)
        c.add("strat", f"st{i}", n, 1, [repr(float(v.min())), repr(float(v.max())), repr(tot), repr(0.1)])
    # polyfit
    for i, n in enumerate([2, 3, 100, 40000]):
        x = rng.uniform(0.2, 1.0, n).astype(F32).astype(np.float64)
        y = (3000 * x + 400 + rng.normal(0, 60, n)).astype(F32).astype(np.float64)
        c.put(f"pf{i}", "in", x)
        c.put(f"pf{i}", "in2", y)
        c.add("polyfit", f"pf{i}", n, 1)
    # getRealCloudPositionS2: the heights as single (DEM) and as double, view angles of the S2 swath
    for i, (cls, h) in enumerate([("single", 3123.4), ("double", 3123.4), ("single", 250.0), ("double", 11950.7), ("single", 1.0e-3)]):
        n = 2000
        x = rng.integers(1, 5491, n).astype(np.float64)
        y = rng.integers(1, 5491, n).astype(np.float64)
        vza = float(F32(rng.uniform(0, 10) * np.pi / 180))
        vaa = float(F32(rng.uniform(70, 300) * np.pi / 180))
        c.put(f"rl{i}", "in", x)
        c.put(f"rl{i}", "in2", y)
        c.add("realpos", f"rl{i}", n, 1, [repr(h), cls, repr(vza), repr(vaa)])
    # ProjectDEM2Plane
    for i, (hasdem, azi, elev) in enumerate([(0, 150.0, 55.0), (1, 150.0, 55.0), (1, 215.0, 38.0), (1, 179.9, 70.0), (1, 180.0, 20.0)]):
        r_, c_ = 60 + 7 * i, 80 + 5 * i
        d = (smooth(rng, r_, c_, 6, 400) + 500).astype(F32)
        d[3, 4] = F32(2400)
        d[10, 12] = F32(-3000)
        c.put(f"pl{i}", "dem", d)
        eb = float(np.radians(elev))
        tb = float(np.radians(azi - 90))
        base = float(np.percentile(d, 0.5)) if hasdem else 0.0
        c.add("plane", f"pl{i}", r_, c_, [hasdem, repr(float(F32(base))), repr(eb), repr(tb), repr(azi), 40])
    # BufferMasks
    for i, (pc, ps, pn) in enumerate([(3, 3, 3), (0, 3, 1), (2, 0, 0), (4, 1, 2), (0, 0, 0)]):
        r_, c_ = 90, 110
        for ext in ("in", "in2", "in3"):
            c.put(f"bf{i}", ext, (rng.random((r_, c_)) < 0.02).astype(np.uint8))
        c.add("buffer", f"bf{i}", r_, c_, [pc, ps, pn])
    # knnsearch (cityblock, k = 14) on centroids with ties
    for i, n in enumerate([14, 15, 30, 50, 51, 80, 200]):
        pts = rng.integers(0, 12, (n, 2)).astype(np.float64)
        pts = pts + (rng.random((n, 2)) < 0.3) * 0.5
        c.put(f"kn{i}", "in", pts)
        c.add("knn", f"kn{i}", n, 2, [repr(float(rng.integers(0, 12))), repr(float(rng.integers(0, 12))), 14])
    return meta


def smooth(rng, rows, cols, sigma, scale=1.0):
    f = ndimage.gaussian_filter(rng.standard_normal((rows, cols)), sigma, mode="reflect")
    return ((f - f.mean()) / f.std() * scale).astype(np.float64)


def compare_prims(d, rep, meta, rng_unused=None):
    L = [l.split() for l in (Path(d) / "cases.txt").read_text().splitlines()]
    n_rng = {"double": [0, 0], "single": [0, 0], "cos": [0, 0]}
    for t in L:
        op, cid, rows, cols, args = t[0], t[1], int(t[2]), int(t[3]), t[4:]
        if op == "range":
            want = rd(d, cid, "out", np.float32 if args[0] == "single" else np.float64)
            b, i, l = float(args[1]), float(args[2]), float(args[3])
            kind = "cos" if cid.startswith("rc") else args[0]
            if args[0] == "single":
                assert rd(d, cid, "out2", np.uint8)[0] == 1
                got = sh.colon_f32(F32(b), F32(i), F32(l))
            else:
                got = sh.colon_f64(b, i, l)
            ok = got.size == want.size and np.array_equal(got, want)
            n_rng[kind][0] += 1
            n_rng[kind][1] += not ok
            if not ok:
                rep.note(f"range {cid} {args}", got.size, f"Rust {got.size} elements, Octave {want.size}", True)
        elif op == "randperm":
            n, k = int(args[0]), int(args[1])
            want = rd(d, cid, "out", np.float64).astype(np.int64) - 1
            rep.exact(f"randperm({n}) first {k} (Octave shim vs Rust)", sh.randperm(n, k), want)
        elif op == "strat":
            x = rd(d, cid, "in", F32)
            want = rd(d, cid, "out", np.float64).astype(np.int64) - 1
            got = sh.stratied_sample(x, F32(float(args[0])), F32(float(args[1])), float(args[2]), float(args[3]))
            rep.exact(f"stratiedSampleHanlder {cid} ({x.size} values, {want.size} samples)", got, want)
        elif op == "polyfit":
            x, y = rd(d, cid, "in", np.float64), rd(d, cid, "in2", np.float64)
            want = rd(d, cid, "out", np.float64)
            p1, p2 = sh.polyfit1(x, y)
            m1, n1 = ulps64([p1, p2], want)
            rel = np.abs(np.array([p1, p2]) - want) / np.abs(want)
            rep.note(f"polyfit degree 1, {x.size} points", 2, f"max {m1} ulp, max rel {rel.max():.2e}; c = p2/p1: Rust {p2 / p1!r}, Octave {want[1] / want[0]!r}")
            rep.bad += rel.max() > 1e-9
        elif op == "realpos":
            x, y = rd(d, cid, "in", np.float64), rd(d, cid, "in2", np.float64)
            wx, wy = rd(d, cid, "out", F32), rd(d, cid, "out2", F32)
            assert rd(d, cid, "out3", np.uint8).tolist() == [1, 1]
            h, cls, vz, va = float(args[0]), args[1], F32(float(args[2])), F32(float(args[3]))
            gx, gy = sh.real_position(x, y, F32(h), vz, va, 20.0)
            rep.cont32(f"getRealCloudPositionS2 {cid} (h {cls}) x", gx, wx)
            rep.cont32(f"getRealCloudPositionS2 {cid} (h {cls}) y", gy, wy)
        elif op == "plane":
            hasdem, base, eb, tb, azi, de = int(args[0]), float(args[1]), float(args[2]), float(args[3]), float(args[4]), int(args[5])
            dem = rd(d, cid, "dem", F32) if hasdem else None
            (pr, pc), gr, gc = sh.project_dem_to_plane(rows, cols, dem, base, eb, tb, azi, de, 20.0)
            rep.exact(f"ProjectDEM2Plane {cid} (dem {hasdem}, azi {azi}) rows", gr, rd(d, cid, "out", np.uint16))
            rep.exact(f"ProjectDEM2Plane {cid} (dem {hasdem}, azi {azi}) cols", gc, rd(d, cid, "out2", np.uint16))
        elif op == "buffer":
            a, b, cc = (rd(d, cid, e, np.uint8) for e in ("in", "in2", "in3"))
            x, y, z = sh.buffer_masks(rows, cols, a, int(args[0]), b, int(args[1]), cc, int(args[2]))
            rep.exact(f"BufferMasks {cid} {args}", np.concatenate([x, y, z]), np.concatenate([rd(d, cid, "out", np.uint8), rd(d, cid, "out2", np.uint8), rd(d, cid, "out3", np.uint8)]))
        elif op == "knn":
            pts = rd(d, cid, "in", np.float64)
            wi = rd(d, cid, "out", np.float64).astype(np.int64) - 1
            wd = rd(d, cid, "out2", np.float64)
            gi, gd = sh.knn_cityblock(pts, float(args[0]), float(args[1]), int(args[2]))
            ties = int((np.diff(wd) == 0).sum())
            same_d = np.array_equal(gd, wd)
            same_i = np.array_equal(gi, wi)
            n = rows
            rep.note(f"knnsearch cityblock k=14, {n} points ({ties} tied pairs in the answer)", 14,
                     f"distances {'equal' if same_d else 'DIFFER'}, indices {'equal' if same_i else 'differ (order of ties: Octave kd-tree for > 50 points)'}", bad=(not same_d) or (not same_i and n <= 50))
    for k, (n, bad) in n_rng.items():
        label = {"double": "Min:step:Max, double", "single": "Min:step:Max, single", "cos": "min:0.1:max, single"}[k]
        rep.note(f"colon operator {label}: {n} ranges", n, f"{bad} differ")
    return len(L)


# ---------------------------------------------------------------------------------------------------------------
# the scenes
# ---------------------------------------------------------------------------------------------------------------
def ellipses(rows, cols, cy, cx, ry, rx):
    y0, y1 = max(int(cy - ry) - 1, 0), min(int(cy + ry) + 2, rows)
    x0, x1 = max(int(cx - rx) - 1, 0), min(int(cx + rx) + 2, cols)
    if y0 >= y1 or x0 >= x1:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return (y0, y1, x0, x1), (((yy + 1 - cy) / ry) ** 2 + ((xx + 1 - cx) / rx) ** 2 <= 1.0)


def paint(m, rows, cols, parts):
    for cy, cx, ry, rx in parts:
        e = ellipses(rows, cols, cy, cx, ry, rx)
        if e is not None:
            (y0, y1, x0, x1), b = e
            m[y0:y1, x0:x1] |= b


def gen_scene(seed, rows, cols, zen, azi, ncl, dem=False, border=True, frac=False, rmin=6, rmax=40, big_cloud=None, nodata=True, layers=(1800.0, 4200.0, 7500.0)):
    """A Sentinel-2 20 m scene: land (smooth NIR/SWIR), water bodies, clouds of several sizes in three layers with the
    shadow the sun (zen, azi) and the height project, the parallax of the view angle included, and a corner without data."""
    rng = np.random.default_rng(seed)
    elev = np.radians(90 - zen)
    tazi = np.radians(azi - 90)
    t = {}
    # view angles in hundredths of a degree (VZA 2-9 degrees across the swath, VAA about 100 degrees)
    u = np.linspace(0, 1, cols)[None, :] * np.ones((rows, 1))
    v = np.linspace(0, 1, rows)[:, None] * np.ones((1, cols))
    vza = np.round(100 * (2.5 + 6.0 * u + 0.8 * v)).astype(F32)
    vaa = np.round(100 * (95 + 25 * v - 8 * u)).astype(F32)
    nir0 = 3200 + smooth(rng, rows, cols, 30, 450)
    swir0 = 2200 + smooth(rng, rows, cols, 25, 300) + 0.4 * (nir0 - 3200)
    if dem:
        z = np.clip(smooth(rng, rows, cols, 35, 260) + 450, 0, None)
        t["dem"] = z.astype(F32)
        gy, gx = np.gradient(z, 20.0)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0
        t["slope"], t["aspect"] = slope.astype(F32), aspect.astype(F32)
        cos_i = np.cos(np.radians(zen)) * np.cos(np.radians(slope)) + np.sin(np.radians(zen)) * np.sin(np.radians(slope)) * np.cos(np.radians(azi - aspect))
        illum = cos_i / np.cos(np.radians(zen))
        nir0 = nir0 * (0.7 + 0.3 * illum)
        swir0 = swir0 * (0.7 + 0.3 * illum)
    nir = nir0 + rng.normal(0, 25, (rows, cols))
    swir = swir0 + rng.normal(0, 20, (rows, cols))
    # water: the highest of a smooth field
    wf = smooth(rng, rows, cols, 18, 1.0)
    water = wf > np.quantile(wf, 0.96)
    nir[water] = 350 + rng.normal(0, 20, water.sum())
    swir[water] = 120 + rng.normal(0, 10, water.sum())
    # clouds, in three layers of height
    layers = np.array(layers)
    cloud = np.zeros((rows, cols), bool)
    shadow = np.zeros((rows, cols), bool)
    meta = []
    specs = []
    if big_cloud is not None:
        specs.append(big_cloud)
    for k in range(ncl):
        specs.append(None)
    for k, sp in enumerate(specs):
        for attempt in range(200):
            h = float(layers[rng.integers(0, len(layers))] + rng.normal(0, 120))
            r = float(rng.uniform(rmin, rmax))
            if sp is not None:
                cy, cx, r, h = sp
                parts_off = [(0.0, 0.0, 1.0, 1.0)]
            else:
                parts_off = [(0.0, 0.0, 1.0, rng.uniform(0.8, 1.4)), (rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8), rng.uniform(0.5, 0.9), rng.uniform(0.5, 1.0))]
                # the shadow lands inside the scene: place the shadow and move it back to the cloud
                sy, sx = rng.uniform(0.05 * rows, 0.95 * rows), rng.uniform(0.05 * cols, 0.95 * cols)
                i_xy = h / (20 * np.tan(elev))
                iy, ix = min(int(sy), rows - 1), min(int(sx), cols - 1)
                dist = h * np.tan(np.radians(vza[iy, ix] / 100)) / 20
                a = np.pi / 2 - np.radians(vaa[iy, ix] / 100)
                px, py = dist * np.cos(a), -dist * np.sin(a)
                cy, cx = sy - py + i_xy * np.sin(tazi), sx - px + i_xy * np.cos(tazi)
                if not (0.01 * rows < cy < rows - 1 and 0.01 * cols < cx < cols - 1):
                    continue
                if border and k == 1:
                    cy, cx = rows - 2.0, float(cx)  # a cloud on the border (its shadow may be out of the scene)
            break
        else:
            continue
        i_xy = h / (20 * np.tan(elev))
        iy, ix = min(max(int(cy), 0), rows - 1), min(max(int(cx), 0), cols - 1)
        dist = h * np.tan(np.radians(vza[iy, ix] / 100)) / 20
        a = np.pi / 2 - np.radians(vaa[iy, ix] / 100)
        px, py = dist * np.cos(a), -dist * np.sin(a)
        scy, scx = cy + py - i_xy * np.sin(tazi), cx + px - i_xy * np.cos(tazi)
        cparts = [(cy + oy * r, cx + ox * r, r * sy_ * 1.0, r * sx_) for oy, ox, sy_, sx_ in parts_off]
        sparts = [(scy + oy * r, scx + ox * r, r * sy_ * 1.0, r * sx_) for oy, ox, sy_, sx_ in parts_off]
        paint(cloud, rows, cols, cparts)
        paint(shadow, rows, cols, sparts)
        meta.append((cy, cx, r, h))
    nir[shadow] *= 0.45
    swir[shadow] *= 0.40
    nir[cloud] = 6500 + rng.normal(0, 150, cloud.sum())
    swir[cloud] = 4200 + rng.normal(0, 120, cloud.sum())
    mask = np.ones((rows, cols), bool)
    if nodata:
        rr, cc = np.mgrid[:rows, :cols]
        mask &= ~(cc / cols + (rows - rr) / rows * 0.55 > 1.5)        # a wedge in the corner
        mask &= ~((rr > 0.3 * rows) & (rr < 0.3 * rows + 4))            # a stripe of missing lines
    cloud &= mask
    # the potential cloud of the input: the cloud with a few chips taken out and some isolated pixels and pairs added
    pcloud = cloud.copy()
    chips = rng.random((rows, cols)) < 0.0005
    pcloud &= ~ndimage.binary_dilation(chips, iterations=1)
    spk = (rng.random((rows, cols)) < 0.0004) & mask & ~water
    pcloud |= spk
    pcloud |= np.roll(spk, 1, axis=1) & mask
    nir_i = np.where(mask, nir, -9999)
    swir_i = np.where(mask, swir, -9999)
    if not frac:
        nir_i, swir_i = np.where(mask, np.round(nir), -9999), np.where(mask, np.round(swir), -9999)
    clear = mask & ~ndimage.binary_dilation(pcloud, iterations=2) & ~water
    t.update(nir=nir_i.astype(F32), swir=swir_i.astype(F32), clear=clear.astype(np.uint8), mask=mask.astype(np.uint8), cloud=pcloud.astype(np.uint8),
             waterall=water.astype(np.uint8), vza=vza, vaa=vaa, snow=(smooth(rng, rows, cols, 12, 1.0) > 1.7).astype(np.uint8),
             zen=zen, azi=azi, ptm=float(clear.sum()), clouds=meta)
    t["true_shadow"] = shadow & mask
    return t


# name: (seed, rows, cols, zen, azi, ncl, dem, extra)
SCENES = {
    "A": dict(seed=11, rows=300, cols=400, zen=35.0, azi=150.0, ncl=7, dem=False, rmin=6, rmax=28),
    "B": dict(seed=12, rows=400, cols=400, zen=48.0, azi=215.0, ncl=9, dem=True, rmin=7, rmax=30),
    "C": dict(seed=13, rows=1000, cols=1000, zen=33.0, azi=160.0, ncl=44, dem=False, frac=True, rmin=5, rmax=45),
    "D": dict(seed=14, rows=1000, cols=1000, zen=42.0, azi=140.0, ncl=40, dem=True, rmin=5, rmax=45),
    "F": dict(seed=17, rows=1000, cols=1000, zen=37.0, azi=145.0, ncl=45, dem=False, rmin=6, rmax=40, layers=(3000.0,)),   # one layer: the heights of the neighbours predict
    "G": dict(seed=18, rows=800, cols=800, zen=44.0, azi=205.0, ncl=40, dem=True, rmin=6, rmax=36, layers=(2500.0, 2900.0)),
    "E": dict(seed=15, rows=250, cols=250, zen=40.0, azi=150.0, ncl=1, dem=False, rmin=100, rmax=110),  # cloud above 90 %
    "H": dict(seed=16, rows=1300, cols=1300, zen=30.0, azi=155.0, ncl=6, dem=False, rmin=10, rmax=40, big_cloud=(650.0, 650.0, 1.0, 2500.0)),
}
# scene variants: id -> (scene, is_shadow_water, ptm override)
VARIANTS = {
    "A": ("A", False, None),
    "A_w": ("A", True, None),
    "A_ptm": ("A", False, 30000.0),
    "B": ("B", False, None),
    "C": ("C", False, None),
    "D": ("D", False, None),
    "E": ("E", False, None),
    "F": ("F", False, None),
    "G": ("G", False, None),
    "E_r": ("E", False, 1.0e6),
    "H": ("H", False, None),
}


def make_scene(name):
    p = dict(SCENES[name])
    if name == "E":
        s = gen_scene(**{**p, "ncl": 1, "nodata": False})
        # make the cloud cover about 93 % of the observations
        rows, cols = p["rows"], p["cols"]
        s["cloud"] = np.ones((rows, cols), np.uint8)
        s["cloud"][: int(0.07 * rows), :] = 0
        s["clear"] = (1 - s["cloud"]).astype(np.uint8) * s["mask"]
        s["ptm"] = float(s["clear"].sum())
        return s
    if name == "H":
        bc = p.pop("big_cloud")
        rows = p["rows"]
        s = gen_scene(**p, big_cloud=None, nodata=False)  # (a stripe of missing lines would cut the cloud in two)
        return add_big_cloud(s, p, bc)
    return gen_scene(**p)


def add_big_cloud(s, p, bc):
    """H: a cloud of more than 1,000,000 pixels (an ellipse of 580 x 600 radii) over the scene, its shadow, and the small clouds."""
    rows, cols = p["rows"], p["cols"]
    rng = np.random.default_rng(99)
    cy, cx, _, h = bc
    elev = np.radians(90 - p["zen"])
    tazi = np.radians(p["azi"] - 90)
    cloud = np.zeros((rows, cols), bool)
    paint(cloud, rows, cols, [(cy, cx, 575.0, 600.0)])
    i_xy = h / (20 * np.tan(elev))
    shadow = np.zeros((rows, cols), bool)
    paint(shadow, rows, cols, [(cy - i_xy * np.sin(tazi), cx - i_xy * np.cos(tazi), 575.0, 600.0)])
    nir, swir = s["nir"].copy(), s["swir"].copy()
    m = s["mask"] > 0
    nir[shadow & m] *= F32(0.45)
    swir[shadow & m] *= F32(0.4)
    nir[cloud & m] = F32(6500)
    swir[cloud & m] = F32(4200)
    s["nir"], s["swir"] = np.round(nir), np.round(swir)
    pc = (s["cloud"] > 0) | (cloud & m)
    s["cloud"] = pc.astype(np.uint8)
    wa = s["waterall"] > 0
    s["clear"] = (m & ~ndimage.binary_dilation(pc, iterations=2) & ~wa).astype(np.uint8)
    s["ptm"] = float(s["clear"].sum())
    s["nbig"] = int((cloud & m).sum())
    return s


def write_scene(c, vid, scene, s, isw, ptm):
    rows, cols = s["nir"].shape
    for k in ("nir", "swir", "clear", "mask", "cloud", "waterall", "snow", "vza", "vaa"):
        c.put(vid, k, s[k])
    hasdem = "dem" in s
    hastopo = "slope" in s
    if hasdem:
        c.put(vid, "dem", s["dem"])
    if hastopo:
        c.put(vid, "slope", s["slope"])
        c.put(vid, "aspect", s["aspect"])
    c.add("shadow", vid, rows, cols, [repr(s["zen"]), repr(s["azi"]), int(hasdem), int(hastopo), repr(ptm), int(isw)])


def compare_scene(d, vid, s, isw, ptm, rep, tim):
    rows, cols = s["nir"].shape
    n = rows * cols
    err = Path(d) / f"{vid}.err"
    if err.exists():
        rep.note(f"scene {vid}: Octave error", 0, err.read_text(), True)
        return
    flat = lambda a: np.ascontiguousarray(a).ravel()
    hasdem, hastopo = "dem" in s, "slope" in s
    # stage 1
    t0 = time.perf_counter()
    ps, info = sh.detect_potential_cloud_shadow(rows, cols, flat(s["nir"]), flat(s["swir"]), flat(s["clear"]), flat(s["mask"]), s["zen"], s["azi"],
                                                flat(s["slope"]) if hastopo else None, flat(s["aspect"]) if hastopo else None)
    tim[f"{vid} DetectPotentialCloudShadow"] = time.perf_counter() - t0
    wps = rd(d, vid, "ps", np.uint8)
    rep.exact(f"[{vid}] pshadow (DetectPotentialCloudShadow), {int((wps == 1).sum())} potential", ps, wps)
    if hastopo:
        wc = rd(d, vid, "c", np.float64)
        rel = np.abs(info[:2] - wc) / np.abs(wc)
        mx, nd = ulps64(info[:2], wc)
        rep.note(f"[{vid}] c of the SCS+C (NIR, SWIR)", 2, f"Rust {info[0]!r}, {info[1]!r}; Octave {wc[0]!r}, {wc[1]!r}; max rel {rel.max():.2e}, {mx} ulp", rel.max() > 1e-9)
        rep.note(f"[{vid}]   samples of the stratified sampling", 1, f"{int(info[4])} in Rust, {rd(d, vid, 'ids', np.float64).size} in Octave", int(info[4]) != rd(d, vid, 'ids', np.float64).size)
    # stage 2 on the Octave potential shadow
    t0 = time.perf_counter()
    skipped, num, sim, cl, shd, ev = sh.match_cloud_shadow(rows, cols, flat(s["mask"]), flat(s["cloud"]), wps, flat(s["waterall"]), flat(s["vza"]), flat(s["vaa"]),
                                                          s["zen"], s["azi"], ptm, isw, flat(s["dem"]) if hasdem else None)
    tim[f"{vid} MatchCloudShadow"] = time.perf_counter() - t0
    wsim = rd(d, vid, "sim", np.float64)
    wcl, wsh = rd(d, vid, "cl", np.uint8), rd(d, vid, "sh", np.uint8)
    rep.exact(f"[{vid}] matched cloud (skipped {skipped}, {num} clouds)", cl, wcl)
    rep.exact(f"[{vid}] matched shadow, {int(wsh.sum())} px", shd, wsh)
    rep.exact(f"[{vid}] similar_num ({wsim.size} values)", sim, wsim)
    wrec = rd(d, vid, "rec", np.float64).reshape(-1, 5)
    ev = np.asarray(ev).reshape(-1, 5)
    rep.exact(f"[{vid}] matches: number", [len(ev)], [wrec.shape[0]])
    if len(ev) == wrec.shape[0] and len(ev):
        for j, nm in ((0, "cloud_type_cur"), (1, "cloud_type"), (2, "record_base_h"), (3, "record_thresh"), (4, "record_base_h_near")):
            if j < 2:
                rep.exact(f"[{vid}]   {nm}", ev[:, j], wrec[:, j])
            else:
                rep.cont64(f"[{vid}]   {nm}", ev[:, j], wrec[:, j])
        if (ev[:, 4] > 0).any():
            rep.note(f"[{vid}]   matches with a height predicted by the neighbours", int((ev[:, 4] > 0).sum()), "(record_base_h_near > 0)")
    # buffers
    x, y, z = sh.buffer_masks(rows, cols, wcl, 3, wsh, 3, flat(s["snow"]), 3)
    rep.exact(f"[{vid}] BufferMasks cloud/shadow/snow", np.concatenate([x, y, z]), np.concatenate([rd(d, vid, "bc", np.uint8), rd(d, vid, "bs", np.uint8), rd(d, vid, "bn", np.uint8)]))
    # the whole chain in Rust (its own potential shadow into the match)
    if (ps == wps).all():
        rep.note(f"[{vid}] true shadow of the synthetic clouds found", int(s["true_shadow"].sum()),
                 f"{(shd.reshape(rows, cols).astype(bool) & s['true_shadow']).sum() / max(s['true_shadow'].sum(), 1):.2f} of it matched")


# ---------------------------------------------------------------------------------------------------------------
# the timing
# ---------------------------------------------------------------------------------------------------------------
def timing():
    n = 5490
    print(f"\n5490 x 5490 timings (RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}):")
    t0 = time.perf_counter()
    s = gen_scene(seed=77, rows=n, cols=n, zen=38.0, azi=152.0, ncl=300, dem=True, rmin=15, rmax=170)
    print(f"  (scene made in {time.perf_counter() - t0:.1f} s: {int(s['cloud'].sum())} cloud pixels, ptm {s['ptm']:.0f})")
    flat = lambda a: np.ascontiguousarray(a).ravel()

    def t(name, fn):
        t0 = time.perf_counter()
        r = fn()
        print(f"  {name:48s} {time.perf_counter() - t0:7.2f} s")
        return r

    ps, _ = t("DetectPotentialCloudShadow, no topo", lambda: sh.detect_potential_cloud_shadow(n, n, flat(s["nir"]), flat(s["swir"]), flat(s["clear"]), flat(s["mask"]), s["zen"], s["azi"]))
    ps, info = t("DetectPotentialCloudShadow, SCS+C", lambda: sh.detect_potential_cloud_shadow(n, n, flat(s["nir"]), flat(s["swir"]), flat(s["clear"]), flat(s["mask"]), s["zen"], s["azi"], flat(s["slope"]), flat(s["aspect"])))
    t("stratied_sample (30M values)", lambda: sh.stratied_sample(np.random.default_rng(0).uniform(0.3, 0.99, 30_000_000).astype(F32), F32(0.3), F32(0.99), 40000.0, 0.1))
    for nm, dem in (("no DEM", None), ("DEM", flat(s["dem"]))):
        r = t(f"MatchCloudShadow, {nm}", lambda: sh.match_cloud_shadow(n, n, flat(s["mask"]), flat(s["cloud"]), ps, flat(s["waterall"]), flat(s["vza"]), flat(s["vaa"]), s["zen"], s["azi"], s["ptm"], False, dem))
        print(f"    ({r[1]} clouds, {len(r[5]) // 5} matched, {int(r[4].sum())} shadow pixels)")
    t("BufferMasks 3/3/3", lambda: sh.buffer_masks(n, n, flat(s["cloud"]), 3, ps, 3, flat(s["snow"]), 3))


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave", default=OCTAVE)
    ap.add_argument("--fmask-src", default=FMASK_SRC, help="the Fmask 4.7 repository (its .m files are run as they are)")
    ap.add_argument("--scenes", default="A,A_w,A_ptm,B,E,E_r,C,D,F,G", help="the variants of VARIANTS; '' for none (H is the scene with a cloud of > 1,000,000 pixels)")
    ap.add_argument("--no-prims", action="store_true")
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
    rng = np.random.default_rng(20261108)
    c = Cases(DATA)
    meta = {} if a.no_prims else build_prims(c, rng)
    names = [x for x in a.scenes.split(",") if x]
    scenes = {}
    for vid in names:
        sc, isw, ptm_o = VARIANTS[vid]
        if sc not in scenes:
            t0 = time.perf_counter()
            scenes[sc] = make_scene(sc)
            print(f"scene {sc}: made in {time.perf_counter() - t0:.1f} s, {scenes[sc]['nir'].shape}, {len(scenes[sc]['clouds']) if 'clouds' in scenes[sc] else 0} clouds, ptm {scenes[sc]['ptm']:.0f}")
        ptm = scenes[sc]["ptm"] if ptm_o is None else ptm_o
        write_scene(c, vid, sc, scenes[sc], isw, ptm)
    c.write()
    shims = make_shims(DATA, a.fmask_src)
    t0 = time.perf_counter()
    times, errs, out = run_octave(DATA, a.octave, shims, a.fmask_src)
    print(f"Octave ran {len(c.lines)} cases in {time.perf_counter() - t0:.1f} s")
    for l in out.splitlines():
        if l.startswith("  scene"):
            print(l)
    for e in errs:
        print(e)
    rep = Report()
    rep.bad += len(errs)
    if not a.no_prims:
        compare_prims(DATA, rep, meta)
    tim = {}
    for vid in names:
        sc, isw, ptm_o = VARIANTS[vid]
        ptm = scenes[sc]["ptm"] if ptm_o is None else ptm_o
        compare_scene(DATA, vid, scenes[sc], isw, ptm, rep, tim)
    rep.show()
    nv = sum(r[1] for r in rep.rows)
    print(f"\n{len(rep.rows)} comparisons, {nv} values; {rep.bad} comparisons with differences")
    for k, v in tim.items():
        print(f"  Rust {k:40s} {v:7.2f} s")
    if a.timing:
        timing()
    return 1 if rep.bad else 0


if __name__ == "__main__":
    sys.exit(main())
