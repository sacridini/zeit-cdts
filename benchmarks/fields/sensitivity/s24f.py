"""Shared pieces of phase 24f: the configurations, stage 1 with the density and the relative mask, a threaded stage 2
that returns the cheap responses, the hierarchy cut (task C) and the noisy Monte Carlo (task B).

A configuration is ``(grad, dens, mask)`` for stage 1 and ``(low, ms)`` (or ``(t, ms)`` for the cut) for stage 2:

* ``grad``: tophat | sobel | multitemporal (combine max) | multimean (combine mean);
* ``dens``: markers per Mpx of the window (``max(1, round(dens * pixels / 1e6))`` markers);
* ``mask``: ``none`` | ``q20`` | ``q35`` (swir16 > that percentile of the window's valid pixels, the package's
  ``mask_quantile``) | ``abs2000`` (swir16 > 2000, the notebook's);
* the Monte Carlo is R = 200, min_distance 20, radius 3, knn colouring, seed 0; final watershed on the image.
"""
import pyexpat  # noqa: F401
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("GDAL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cases as C
import pipeline as P
from fast import fast_eval
from zeit import _fields_core as _fc

OUT24 = C.OUT / "f24"
R = 200
GRADS = ("tophat", "sobel", "multitemporal", "multimean")
DENS = (300, 600, 1000, 2000)
MASKS = ("none", "q20", "q35", "abs2000")
LOWS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SIZES = (0, 2, 5, 10, 25, 50)
SIZES_MEAN = (0, 5, 25)          # the multimean gradient is run on a reduced min_size grid (cost)
BASE = P.S1(realizations=R, markers=1000, min_distance=20, marker_radius=3, coloring="knn", seed=0)
THREADS = 6


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def high_for(low: float) -> float:
    return round(max(0.6, low + 0.1), 4)


def mask_thr(case, mask: str):
    if mask == "none":
        return None
    if mask == "abs2000":
        return 2000.0
    q = float(mask[1:])
    return float(np.percentile(case.bands[C.SWIR][case.valid], q))


def n_markers(case, dens: float) -> int:
    return max(1, int(round(dens * case.valid.size / 1e6)))


def s1_for(case, grad: str, dens: float, mask: str, seed: int = 0) -> P.S1:
    return replace(BASE, gradient=grad, markers=n_markers(case, dens), mask_threshold=mask_thr(case, mask), seed=seed)


def sizes_for(grad: str):
    return SIZES_MEAN if grad == "multimean" else SIZES


def preload_gradients(cs, names=GRADS):
    """Every gradient of every case, once (6 Mpx x 4 gradients: small), and ``P.gradient_of`` served from it."""
    store = {}
    for c in cs:
        for g in names:
            store[(c.name, g)] = None
    for c in cs:
        for g in names:
            if store[(c.name, g)] is None:
                store[(c.name, g)] = P.gradient_of(c, g)
        P._GRAD.clear()
    P.gradient_of = lambda case, name: store[(case.name, name)]
    return store


# ------------------------------------------------------------------ stage 2
def _finish(case, keep, regions_with_zero):
    """Final watershed on the image from the labels (0 = free), the mask and the boundary network left out."""
    lab = _fc.watershed(case.img, regions_with_zero, connectivity=1)
    if keep is not None:
        lab = np.where(keep, lab, 0).astype(np.int32)
    return lab


def mc_regions(counts, r, low, high):
    """Hysteresis on the counts and the label of its complement; the network is label 1, as in ``P.stage2``."""
    hyst = _fc.hysteresis_threshold(counts.astype(np.float64), low * r, high * r)
    regions, _ = _fc.label(~hyst, connectivity=1)
    return regions + 1


def hyst_pred(case, keep, regions, ms):
    """``P.stage2`` from the labelled complement: min_size, the final watershed, the network set to 0 (int64)."""
    reg = _fc.remove_small_objects(regions, ms)
    network_kept = bool((reg == 1).any())
    reg, _ = _fc.relabel_sequential(reg)
    lab = _finish(case, keep, reg)
    bl = int(network_kept and (lab == 1).any())
    lab, _ = _fc.relabel_sequential(lab)
    pred = lab.astype(np.int64)
    if bl:
        pred = np.where(pred == bl, 0, pred)
    return pred


def cut_components(h, v, keep, t):
    """Components of the pixel graph keeping the 4-neighbour edges with value <= t (inside ``keep``): int32, 0 outside.

    The graph is drawn on a doubled grid (pixels at even, edges at odd positions) and labelled 4-connected."""
    if keep is None:
        keep = np.ones((h.shape[0], h.shape[1] + 1), bool)
    ny, nx = keep.shape
    g = np.zeros((2 * ny - 1, 2 * nx - 1), bool)
    g[::2, ::2] = keep
    g[::2, 1::2] = (h <= t) & keep[:, :-1] & keep[:, 1:]
    g[1::2, ::2] = (v <= t) & keep[:-1] & keep[1:]
    lab, _ = _fc.label(g, connectivity=1)
    return lab[::2, ::2]


def cut_pred(case, keep, comps, ms):
    """Hierarchy cut -> ``pred``: ``min_size`` and the final watershed as usual (no boundary network)."""
    reg = _fc.remove_small_objects(comps, ms) if ms > 0 else comps
    reg, _ = _fc.relabel_sequential(reg)
    lab = _finish(case, keep, reg)
    lab, _ = _fc.relabel_sequential(lab)
    return lab.astype(np.int64)


def edge_values_mc(counts, r):
    """An edge's value on the Monte Carlo map: the maximum of its two pixels' frequencies (h: (y, x-1), v: (y-1, x))."""
    f = counts.astype(np.float64) / r
    return np.maximum(f[:, :-1], f[:, 1:]), np.maximum(f[:-1], f[1:])


def exact_edges(case, s1: P.S1, n_jobs: int = 8):
    """The per-edge exact probabilities ``(h, v)`` (cached as float32 beside the stage-1 maps)."""
    t = "none" if s1.mask_threshold is None else f"{s1.mask_threshold:g}"
    p = C.OUT / "cache" / case.name / f"edges_{s1.gradient}_m{s1.markers}_t{t}.npz"
    if p.exists():
        z = np.load(p)
        return z["h"].astype(np.float64), z["v"].astype(np.float64)
    keep = case.keep(s1.mask_threshold)
    grad = P.gradient_of(case, s1.gradient)
    bad = ~np.isfinite(grad) & (keep if keep is not None else True)
    if bad.any():   # the exact method refuses NaN inside the mask (valid pixels next to NoData on the multitemporal
        grad = np.where(bad, np.nanmax(grad[np.isfinite(grad)]), grad)   # gradients): an edge there, the maximum
    _, h, v = _fc.exact_boundary_probability(grad, keep, float(s1.markers), return_edges=True, n_jobs=n_jobs)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, h=h.astype(np.float32), v=v.astype(np.float32))
    tmp.replace(p)
    return np.asarray(h, np.float64), np.asarray(v, np.float64)


# ------------------------------------------------------------------ rows
def fprop(rec, cross):
    """The score of a property reference: harmonic mean of the recall and the share not crossing (as cv.py)."""
    nc = 1 - cross
    return 2 * rec * nc / (rec + nc) if (rec + nc) > 0 else np.nan


def row(case, resp, **cfg):
    f = resp["f"]
    if case.kind == "property":
        f = fprop(resp["rec"], resp["cross_share"]) if not np.isnan(resp.get("cross_share", np.nan)) else np.nan
    return {"case": case.id, "mosaic": case.name, "group": case.group, "kind": case.kind, "ref": case.ref,
            "region_pixels": int(case.regions[1].sum()), **cfg, "f": f, "rec": resp["rec"], "d_w": resp["d_w"],
            "cross_share": resp.get("cross_share", np.nan), "n_labels": resp["n_labels"]}


def pmap(fn, items, threads=THREADS):
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(fn, items))


def case_rows_mc(case, keep, b, cfg: dict, lows, sizes):
    rows = []
    for low in lows:
        regions = mc_regions(b.counts, b.r, low, high_for(low))
        for ms in sizes:
            pred = hyst_pred(case, keep, regions, ms)
            rows.append(row(case, fast_eval(case, pred), **cfg, low=low, ms=ms))
    return rows
