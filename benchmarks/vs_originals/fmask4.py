"""The MATLAB primitives of the Fmask 4.7 port (``zeit_core::fmask::mlab``: ``prctile``, ``imresize`` by 2 and 3, ``stdfilt``,
``imerode``/``imdilate`` with ``strel`` square, disk and arbitrary, ``bwlabel``, ``bwareaopen``, ``regionprops``, ``imfill``,
``imfilter`` 3 x 3 and ``round``) against GNU Octave 11.3 (image 2.20, statistics) on pseudo-random arrays of a fixed seed.

The Octave side is ``fmask4/driver.m`` with the shims of ``fmask4/shims/`` (Fmask's .m files are not touched, and none of
them is used here): ``imresize`` (Octave's ``'box'`` does not average), ``fmask_disk`` (the ``strel('disk', r)`` of MATLAB,
N = 4, by periodic lines, written apart from the Rust one), and a copy of Octave's ``@strel`` class that this script makes
in the temporary folder, patched so that ``strel('disk', r)`` returns that element (Octave only takes N = 0). Inputs and
outputs go through files in a temporary folder outside the repository. Zeit's side is ``zeit._core.fmask``::

    python benchmarks/vs_originals/fmask4.py [--octave PATH] [--timing]

Integers, labels and masks must agree exactly (the exit status is 1 if not); the continuous values are reported as the
largest difference in float32 ulps (the single results) or relative (the double ``stdfilt``). ``--timing`` also times the
Rust functions on 5490 x 5490 (a Sentinel-2 scene at 20 m). Run with ``OMP_NUM_THREADS=8 RAYON_NUM_THREADS=8``.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from common import no_power_throttling, workdir
import fmask4_detect as det

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from zeit import _core  # noqa: E402

fm = _core.fmask
HERE = Path(__file__).resolve().parent
DATA = workdir("fmask4")
FMASK_SRC = r"C:\Users\eduar\Documents\dev\Fmask4"
OCTAVE = r"C:\Users\eduar\AppData\Local\Programs\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe"
F32 = np.float32


def ordered(a):
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulps(a, b):
    """The largest difference in float32 ulps; NaN must be NaN on both sides (otherwise a huge number)."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    na, nb = np.isnan(a), np.isnan(b)
    bad = int((na != nb).sum())
    ok = ~(na | nb)
    d = np.abs(ordered(a)[ok] - ordered(b)[ok])
    return (int(d.max()) if d.size else 0), int((d > 0).sum()), bad


class Cases:
    """The cases of one run: the files written for Octave and the lines of cases.txt."""

    def __init__(self, d):
        self.d, self.lines = Path(d), []

    def add(self, op, cid, arr, rows, cols, args=(), mask=None):
        arr.tofile(self.d / f"{cid}.in")
        if mask is not None:
            mask.astype(np.uint8).tofile(self.d / f"{cid}.mask")
        self.lines.append(f"{op} {cid} {rows} {cols} " + " ".join(str(a) for a in args))

    def write(self):
        (self.d / "cases.txt").write_text("\n".join(self.lines) + "\n")


def make_shims(d, octave):
    """The folder of the shims: ours, and the @strel class of the installed image package patched for strel('disk', r)."""
    s = Path(d) / "shims"
    if s.exists():
        shutil.rmtree(s)
    s.mkdir()
    for f in (HERE / "fmask4" / "shims").glob("*.m"):
        shutil.copy(f, s)
    root = Path(octave).parent.parent
    src = glob.glob(str(root / "share" / "octave" / "packages" / "image-*" / "@strel"))[0]
    shutil.copytree(src, s / "@strel")
    f = s / "@strel" / "strel.m"
    t = f.read_text(encoding="utf8")
    a = 'error ("strel: N for disk shape not yet implemented, use N of 0");'
    b = "      SE.nhood = r <= radius;"
    assert a in t and b in t
    t = t.replace(a, "n = -1;").replace(b, "      if (n == -1)\n        SE.nhood = fmask_disk (radius);\n      else\n        SE.nhood = r <= radius;\n      endif")
    f.write_text(t, encoding="utf8")
    return s


def run_octave(d, octave, shims, src=None):
    env = dict(os.environ, FMASK_DIR=str(d), FMASK_SHIMS=str(shims), OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8")
    if src:
        env["FMASK_SRC"] = str(src)
    r = subprocess.run([octave, "--no-gui", str(HERE / "fmask4" / "driver.m")], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"octave failed:\n{r.stdout}\n{r.stderr}")
    return {l.split()[1]: float(l.split()[2]) for l in r.stdout.splitlines() if l.endswith(" s")}


def blobs(rng, rows, cols, sigma, thr):
    """A binary image of objects of different sizes (some touching the borders): a smoothed noise above a threshold."""
    f = ndimage.gaussian_filter(rng.standard_normal((rows, cols)), sigma)
    f = (f - f.mean()) / f.std()
    return (f > thr).astype(np.uint8)


def smooth(rng, rows, cols, sigma, scale=1.0):
    f = ndimage.gaussian_filter(rng.standard_normal((rows, cols)), sigma)
    return ((f - f.mean()) / f.std() * scale).astype(F32)


def build_cases(d):
    rng = np.random.default_rng(20261108)
    c = Cases(d)
    meta = {}
    # prctile
    ps = [0, 0.001, 0.01, 1, 3.3, 17.5, 25, 50, 82.5, 90, 99.99, 100]
    for n in [1, 2, 3, 10, 101, 1000, 250000, 3_000_000]:
        v = np.round(rng.standard_normal(n) * 1000).astype(F32) / F32(8)  # duplicates
        v[rng.random(n) < 0.1] = np.nan
        if n <= 3:
            v[:] = rng.standard_normal(n).astype(F32)
        c.add("prctile", f"prctile_{n}", v, n, 1, ps)
        meta[f"prctile_{n}"] = ps
    # imresize
    for r_, c_ in [(40, 60), (500, 700)]:
        x = (rng.random((r_, c_)) * 10000).astype(F32)
        x[rng.random((r_, c_)) < 0.001] = np.nan
        c.add("resize_half", f"half_{r_}", x, r_, c_)
    x = np.arange(1, 17, dtype=F32).reshape(4, 4)
    c.add("resize_native_box", "native_box", x, 4, 4)
    for r_, c_ in [(37, 41), (200, 300)]:
        c.add("rep3", f"rep3_{r_}", (rng.random((r_, c_)) * 100).astype(F32), r_, c_)
    # stdfilt: reflectance-like integers, flat blocks and NaN
    for (r_, c_), ks in {(200, 300): [3, 7, 21], (300, 300): [101], (260, 270): [501]}.items():
        x = np.round(np.abs(smooth(rng, r_, c_, 6, 1500) + 3000 + rng.standard_normal((r_, c_)) * 50)).astype(F32)
        x[20:80, 30:90] = 1234  # a flat block
        for k in ks:
            c.add("stdfilt", f"std_{r_}_{k}", x, r_, c_, [k])
    x = (rng.random((40, 50)) * 100).astype(F32)
    x[10, 10] = np.nan
    x[30, 45] = np.nan
    c.add("stdfilt", "std_nan", x, 40, 50, [7])
    x = (rng.random((5, 4)) * 100).astype(F32)
    c.add("stdfilt", "std_tiny", x, 5, 4, [7])
    # morphology
    arb = rng.random((5, 7)) < 0.5
    ses = [("square", 3, 3), ("square", 5, 5), ("square", 11, 11), ("arb", 5, 7)] + [("disk", r, r) for r in (1, 2, 3, 5, 6, 10, 15)]
    for si, (kind, a, b) in enumerate(ses):
        m = arb if kind == "arb" else None
        for ii, (r_, c_, sg, thr) in enumerate([(150, 200, 3, 0.3), (160, 180, 1.5, 1.0), (240, 250, 6, 0.0)]):
            img = blobs(rng, r_, c_, sg, thr)
            img[rng.random((r_, c_)) < 0.003] = 1  # isolated pixels
            for op in ("erode", "dilate"):
                c.add("morph_bin", f"mb_{op}_{si}_{ii}", img, r_, c_, [op, kind, a, b], m)
        g = (smooth(rng, 150, 200, 3, 100) + 500).astype(F32)
        for op in ("erode", "dilate"):
            c.add("morph_gray", f"mg_{op}_{si}", g, 150, 200, [op, kind, a, b], m)
    # strel disk
    for r in list(range(0, 21)) + [25, 30]:
        if r > 0:
            c.add("strel_disk", f"disk_{r}", np.zeros(1, np.uint8), 1, 1, [r])
    # bwlabel, regionprops, bwareaopen
    for i, (r_, c_, sg, thr) in enumerate([(60, 80, 1.0, 0.5), (200, 300, 2.0, 0.6), (300, 300, 1.0, 1.6), (101, 77, 0.7, 0.2), (3, 3, 1, 0.5)]):
        img = blobs(rng, r_, c_, sg, thr)
        c.add("bwlabel", f"lab_{i}", img, r_, c_)
    c.add("bwlabel", "lab_zero", np.zeros(20, np.uint8), 4, 5)
    c.add("bwlabel", "lab_one", np.ones(20, np.uint8), 4, 5)
    for mn in (1, 2, 3, 10, 50):
        c.add("bwareaopen", f"bwao_{mn}", blobs(rng, 200, 300, 1.5, 0.7), 200, 300, [mn])
    # imfill: continuous, integer plateaus, and a basin that leaks diagonally
    for i, (r_, c_) in enumerate([(100, 120), (257, 301)]):
        c.add("imfill", f"fill_c{i}", (smooth(rng, r_, c_, 4, 1.0) * 1000).astype(F32), r_, c_)
        c.add("imfill", f"fill_i{i}", rng.integers(0, 6, (r_, c_)).astype(F32), r_, c_)
        c.add("imfill", f"fill_s{i}", (np.round(smooth(rng, r_, c_, 2, 3)) ).astype(F32), r_, c_)
    diag = np.array([[9, 9, 9, 9, 9], [9, 1, 9, 9, 9], [9, 9, 1, 9, 9], [9, 9, 9, 1, 9], [9, 9, 9, 9, 0]], F32)
    c.add("imfill", "fill_diag", diag, 5, 5)
    # imfilter 3 x 3: the four templates of EnhanceLine
    tpls = {"v": [-1, 2, -1, -1, 2, -1, -1, 2, -1], "h": [-1, -1, -1, 2, 2, 2, -1, -1, -1], "d1": [2, -1, -1, -1, 2, -1, -1, -1, 2],
            "d2": [-1, -1, 2, -1, 2, -1, 2, -1, -1]}
    img = (np.abs(smooth(rng, 200, 300, 2, 0.1)) + rng.random((200, 300)) * 0.05).astype(F32)
    for name, t in tpls.items():
        c.add("imfilter", f"filt_{name}", img, 200, 300, [repr(v / 6) for v in t])
    img2 = (rng.random((101, 103)) * 10000).astype(F32)
    c.add("imfilter", "filt_big", img2, 101, 103, [repr(v / 6) for v in tpls["d1"]])
    # round
    x = np.concatenate([np.arange(-50, 51) + 0.5, np.arange(-50, 51) - 0.5, (rng.random(500) - 0.5) * 200, [0.49999999999999994, 2.5, -2.5, 4.5]])
    c.add("round", "round", x.astype(np.float64), 1, x.size)
    c.write()
    return meta


def rd(d, cid, ext, dtype, shape=None):
    a = np.fromfile(Path(d) / f"{cid}.{ext}", dtype)
    return a if shape is None else a.reshape(shape)


class Report:
    def __init__(self):
        self.rows, self.bad = [], 0

    def exact(self, name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        n = a.size
        nd = int((a.ravel() != b.ravel()).sum()) if a.size == b.size else n + 1
        self.rows.append((name, n, f"{nd} differ" if nd else "equal"))
        self.bad += nd > 0

    def cont(self, name, a, b, tol=0):
        mx, nd, nan = ulps(a, b)
        self.rows.append((name, np.asarray(a).size, f"max {mx} ulp, {nd} differ, {nan} NaN mismatch"))
        self.bad += (mx > tol) or nan > 0

    def note(self, name, n, text):
        self.rows.append((name, n, text))

    def show(self):
        for name, n, t in self.rows:
            print(f"{name:34s} {n:>10d}  {t}")


def disk_mask(r):
    s, m = fm.strel_disk(r)
    return s, m


def compare(d, rep):
    L = [l.split() for l in (Path(d) / "cases.txt").read_text().splitlines()]
    total = 0
    for t in L:
        op, cid, rows, cols = t[0], t[1], int(t[2]), int(t[3])
        args = t[4:]
        shp = (rows, cols)
        if op == "prctile":
            v = rd(d, cid, "in", F32)
            want = rd(d, cid, "out", F32)
            got = np.array([fm.prctile(v, float(p)) for p in args], F32)
            rep.cont(f"prctile n={v.size}", got, want)
        elif op == "resize_half":
            x = rd(d, cid, "in", F32)
            want = rd(d, cid, "out", F32)
            got = fm.resize_box_half(x, rows, cols)
            rep.cont(f"resize 2x2 mean {rows}x{cols}", got, want)
        elif op == "resize_native_box":
            x = rd(d, cid, "in", F32)
            nat = rd(d, cid, "out", F32)
            got = fm.resize_box_half(x, rows, cols)
            rep.note("Octave native box (4x4 of 1..16)", 4, f"Octave {nat.tolist()}, MATLAB/Rust {got.tolist()}")
        elif op == "rep3":
            x = rd(d, cid, "in", F32)
            rep.exact(f"replicate 3 {rows}x{cols}", fm.resize_replicate3(x, rows, cols), rd(d, cid, "out", F32))
        elif op == "stdfilt":
            x = rd(d, cid, "in", F32)
            want = rd(d, cid, "out", np.float64)
            got = fm.stdfilt(x, rows, cols, int(args[0]))
            nanm = int((np.isnan(got) != np.isnan(want)).sum())
            ok = ~np.isnan(want)
            ad = np.abs(got[ok] - want[ok])
            big = want[ok] > 1.0
            rel = (ad[big] / want[ok][big]).max() if big.any() else 0.0
            flat = (want[ok] <= 1.0)
            rep.note(f"stdfilt {rows}x{cols} k={args[0]}", got.size,
                     f"NaN mismatch {nanm}; std>1: max rel {rel:.2e}; std<=1 ({int(flat.sum())} px): max abs {ad[flat].max() if flat.any() else 0:.2e}; "
                     f"Octave class double: {bool(rd(d, cid, 'out2', np.uint8)[0])}")
            rep.bad += nanm > 0 or rel > 1e-9
        elif op in ("morph_bin", "morph_gray"):
            kind, a, b = args[1], int(args[2]), int(args[3])
            erode = args[0] == "erode"
            if op == "morph_bin":
                x = rd(d, cid, "in", np.uint8)
                want = rd(d, cid, "out", np.uint8)
                f = fm.morph_bin
            else:
                x = rd(d, cid, "in", F32)
                want = rd(d, cid, "out", F32)
                f = fm.morph_gray
            if kind == "arb":
                m = rd(d, cid, "mask", np.uint8)
                got = f(x, rows, cols, erode, m, a, b)
            elif kind == "square":
                got = f(x, rows, cols, erode, None, 0, 0, a)  # the fast square
                got2 = f(x, rows, cols, erode, np.ones(a * a, np.uint8), a, a)  # the general path
                assert (got == got2).all()
            else:
                s, m = disk_mask(a)
                got = f(x, rows, cols, erode, m, s, s)
            if op == "morph_bin":
                rep.exact(f"{cid} ({kind} {a})", got, want)
            else:
                rep.exact(f"{cid} ({kind} {a})", got, want)
        elif op == "strel_disk":
            r = int(args[0])
            s, m = disk_mask(r)
            want = rd(d, cid, "out", np.uint8)
            rep.exact(f"strel disk r={r} ({s}x{s}, {int(m.sum())} nbrs)", m, want)
            if r < 3:
                rep.exact(f"  Octave exact disk r={r}", m, rd(d, cid, "out2", np.uint8))
        elif op == "bwlabel":
            x = rd(d, cid, "in", np.uint8)
            l, n = fm.bwlabel(x, rows, cols)
            n1, n2 = rd(d, cid, "out3", np.int32)
            wl, wl2 = rd(d, cid, "out", np.int32), rd(d, cid, "out2", np.int32)
            area, cen, pix = fm.regionprops(l, rows, cols, n)
            # Octave's numbers are not in the order of the first pixel by columns (bwlabeln and bwlabel agree with
            # each other): the numbering is renamed to that order here, which leaves the partition (the objects,
            # their 8-connectivity) and the regionprops values to be compared; the raw numbering is counted
            raw = int((l != wl).sum())
            perm = np.zeros(n1 + 1, np.int64)
            seen = 0
            for v in wl.reshape(rows, cols).T.ravel():
                if v and perm[v] == 0:
                    seen += 1
                    perm[v] = seen
            canon = perm[wl]
            rep.note(f"bwlabeln {cid}: Octave numbering", wl.size, f"{raw} pixels numbered differently from column order ({n} objects)")
            rep.exact(f"bwlabeln {cid} (renamed)", l, canon)
            rep.exact(f"bwlabel  {cid} (== bwlabeln)", wl2, wl)
            rep.exact(f"  count {cid}", [n, n], [n1, n2])
            oa = rd(d, cid, "area", np.float64).astype(np.int64)
            oc = rd(d, cid, "cen", np.float64).reshape(-1, 2)
            op_ = rd(d, cid, "pix", np.float64).astype(np.int64).reshape(-1, 2)
            starts = np.concatenate([[0], np.cumsum(oa)])
            order = np.argsort(perm[1:])  # order[k] = Octave label (0-based) that is k + 1 in column order
            ra = oa[order]
            rc = oc[order] if n else oc
            rp = np.concatenate([op_[starts[i]:starts[i + 1]] for i in order]) if n else op_
            rep.exact(f"  regionprops Area {cid}", area, ra)
            rd_ = np.abs(cen.reshape(-1, 2) - rc).max() if n else 0.0
            rep.note(f"  regionprops Centroid {cid}", cen.size, f"max abs diff {rd_:.2e} (the order of the sums)")
            rep.bad += rd_ > 1e-9
            rep.exact(f"  regionprops PixelList {cid}", pix.reshape(-1, 2), rp)
        elif op == "bwareaopen":
            x = rd(d, cid, "in", np.uint8)
            rep.exact(f"bwareaopen min={args[0]}", fm.bwareaopen(x, rows, cols, int(args[0])), rd(d, cid, "out", np.uint8))
        elif op == "imfill":
            x = rd(d, cid, "in", F32)
            got = fm.imfill_gray(x, rows, cols)
            o1, o4, o8 = rd(d, cid, "out", F32), rd(d, cid, "out2", F32), rd(d, cid, "out3", F32)
            # Octave's imfill goes through 1 - x in single (imcomplement), so every pixel is 1 - (1 - x): the same
            # round trip applied to the result must give Octave's
            rt = F32(1) - (F32(1) - got)
            mx, nd, _ = ulps(got, o4)
            rep.note(f"imfill {cid} vs imfill(x,4), raw", got.size, f"max {mx} ulp, {nd} differ")
            rep.exact(f"imfill {cid} vs imfill(x,4), after 1-(1-x)", rt, o4)
            rep.note(f"  Octave default == conn 4: {np.array_equal(o1, o4)}; == conn 8: {np.array_equal(o1, o8)}", got.size,
                     f"Rust(4) round trip vs conn 8: {int((rt != o8).sum())} differ")
            rep.bad += not np.array_equal(o1, o4)
        elif op == "imfilter":
            x = rd(d, cid, "in", F32)
            tp = np.array([float(a) for a in args])
            got = fm.imfilter3(x, rows, cols, list(tp.astype(F32)))
            tm = tp.reshape(3, 3)
            scale = ndimage.correlate(np.abs(x.reshape(rows, cols)).astype(np.float64), np.abs(tm), mode="constant").ravel()
            for name, w in (("single kernel", rd(d, cid, "out", F32)), ("double kernel", rd(d, cid, "out2", F32))):
                err = np.abs(got.astype(np.float64) - w.astype(np.float64)) / (np.finfo(F32).eps * np.maximum(scale, 1e-300))
                mx, nd, _ = ulps(got, w)
                rep.note(f"imfilter {cid} ({name} in Octave)", got.size,
                         f"{nd} differ; max {err.max():.2f} eps of sum|t*x| (raw {mx} ulp of the result: cancellation)")
                rep.bad += err.max() > 8
        elif op == "round":
            x = rd(d, cid, "in", np.float64)
            got = np.array([fm.round_half_away(float(v)) for v in x])
            rep.exact("round", got, rd(d, cid, "out", np.float64))
        total += 1
    return total


def timing():
    """Rust on 5490 x 5490 (Sentinel-2 at 20 m)."""
    rng = np.random.default_rng(1)
    n = 5490
    print(f"\n5490 x 5490 timings (RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}):")
    base = ndimage.gaussian_filter(rng.standard_normal((n // 6, n // 6)), 4)
    big = ndimage.zoom(base, 6, order=1)[:n, :n]
    f = np.ascontiguousarray(((big - big.mean()) / big.std() * 1000 + 3000 + rng.standard_normal((n, n)) * 30).astype(F32)).ravel()
    b = np.ascontiguousarray((big > 0.4).astype(np.uint8)).ravel()
    b[rng.random(b.size) < 0.001] = 1
    ff = f.copy()
    ff[rng.random(ff.size) < 0.001] = np.nan
    s15, m15 = fm.strel_disk(15)
    ones = np.ones(21 * 21, np.uint8)

    def t(name, fn):
        t0 = time.perf_counter()
        r = fn()
        print(f"  {name:36s} {time.perf_counter() - t0:7.2f} s")
        return r

    t("prctile 0.001 (30M, NaN)", lambda: fm.prctile(ff, 0.001))
    t("resize 2x2 mean (5490 -> 2745)", lambda: fm.resize_box_half(f, n, n))
    t("replicate 3 (1830 -> 5490)", lambda: fm.resize_replicate3(f[: 1830 * 1830], 1830, 1830))
    t("stdfilt k=7", lambda: fm.stdfilt(f, n, n, 7))
    t("stdfilt k=501", lambda: fm.stdfilt(ff, n, n, 501))
    t("erode bin disk r=15 (29x29)", lambda: fm.morph_bin(b, n, n, True, m15, s15, s15))
    t("dilate bin disk r=15", lambda: fm.morph_bin(b, n, n, False, m15, s15, s15))
    t("erode gray disk r=15", lambda: fm.morph_gray(f, n, n, True, m15, s15, s15))
    t("dilate bin square 101", lambda: fm.morph_bin(b, n, n, False, None, 0, 0, 101))
    t("erode gray square 21 (fast)", lambda: fm.morph_gray(f, n, n, True, None, 0, 0, 21))
    t("erode gray 21x21 ones (general)", lambda: fm.morph_gray(f, n, n, True, ones, 21, 21))
    l, k = t("bwlabel 8", lambda: fm.bwlabel(b, n, n))
    print(f"    ({k} components)")
    t("bwareaopen 10", lambda: fm.bwareaopen(b, n, n, 10))
    t("regionprops", lambda: fm.regionprops(l, n, n, k))
    t("imfill (gray, 4)", lambda: fm.imfill_gray(f, n, n))
    t("imfilter 3x3", lambda: fm.imfilter3(f, n, n, [-1 / 6, 2 / 6, -1 / 6] * 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octave", default=OCTAVE)
    ap.add_argument("--fmask-src", default=FMASK_SRC, help="the Fmask 4.7 repository (its .m files are run as they are)")
    ap.add_argument("--scenes", default="A,B,C,F,G,E,H,T", help="the scenes of the potential pixels (fmask4_detect.SCENES); '' for none")
    ap.add_argument("--big", action="store_true", help="also the 1000 x 1000 scene D (Octave takes minutes in DetectAbsSnow)")
    ap.add_argument("--no-prims", action="store_true", help="skip the MATLAB primitives")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--timing-only", action="store_true")
    a = ap.parse_args()
    no_power_throttling()
    if a.timing_only:
        timing()
        det.timing_scene()
        return 0
    for f in DATA.glob("*"):
        if f.is_file():
            f.unlink()
    names = [x for x in a.scenes.split(",") if x] + (["D"] if a.big else [])
    if a.no_prims:
        c = Cases(DATA)
    else:
        build_cases(DATA)
        c = Cases(DATA)
        c.lines = (DATA / "cases.txt").read_text().splitlines()
    for nm in names:
        det.write_scene(c, nm)
    c.write()
    shims = make_shims(DATA, a.octave)
    t0 = time.perf_counter()
    times = run_octave(DATA, a.octave, shims, a.fmask_src)
    print(f"Octave ran {len(times)} cases in {time.perf_counter() - t0:.1f} s")
    rep = Report()
    if not a.no_prims:
        compare(DATA, rep)
    for nm in names:
        det.compare_scene(DATA, nm, rep)
        print(f"  scene {nm}: Octave {times.get(nm, float('nan')):.1f} s")
    if names:
        base = "A" if "A" in names else names[0]
        lines, ids = det.build_direct(DATA, base)
        (DATA / "cases.txt").write_text("\n".join(lines) + "\n")
        run_octave(DATA, a.octave, shims, a.fmask_src)
        det.compare_direct(DATA, ids, base, rep)
    rep.show()
    nv = sum(r[1] for r in rep.rows)
    print(f"\n{len(rep.rows)} comparisons, {nv} values; {rep.bad} comparisons with differences")
    if a.timing:
        timing()
        det.timing_scene()
    return 1 if rep.bad else 0


if __name__ == "__main__":
    sys.exit(main())
