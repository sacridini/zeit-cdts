"""The two stages of ``zeit.fields`` and the responses of the sensitivity analysis.

Stage 1 (expensive): the boundary map, the Monte Carlo counts (or the exact probability), from a gradient, a
marker model, a mask and a seed. It is cached on disk.

Stage 2 (cheap): the hysteresis, ``min_size``, the final watershed and the mask, exactly as ``zeit.fields``
does them (``test_pipeline.py`` checks that stage 1 + stage 2 equals ``zeit.fields``), then the accuracy.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

import zeit
from zeit import _fields_core as _fc
from zeit._fields import _boundary
from zeit._segacc import _boundary_pixels, _hit_share

from cases import OUT, TOLERANCES, Case
from matched import variants


@dataclass(frozen=True)
class S1:
    """The stage-1 factors (the notebook's values by default)."""
    gradient: str = "tophat"          # tophat | sobel | multitemporal
    method: str = "montecarlo"        # montecarlo | exact
    realizations: int = 500
    markers: int = 1000
    min_distance: float = 20
    marker_radius: int = 3
    coloring: str = "knn"
    mask_threshold: Optional[float] = 2000
    seed: int = 0

    def key(self) -> str:
        t = "none" if self.mask_threshold is None else f"{self.mask_threshold:g}"
        if self.method == "exact":
            return f"{self.gradient}_exact_m{self.markers}_t{t}"
        return (f"{self.gradient}_mc_R{self.realizations}_m{self.markers}_d{self.min_distance:g}_"
                f"r{self.marker_radius}_{self.coloring}_t{t}_s{self.seed}")


@dataclass(frozen=True)
class S2:
    """The stage-2 factors (the notebook's values by default)."""
    low: float = 0.3
    high: float = 0.6
    min_size: int = 50
    final: str = "image"              # image | gradient

    def key(self) -> str:
        return f"lo{self.low:g}_hi{self.high:g}_ms{self.min_size}_{self.final}"


@dataclass
class Boundary:
    counts: Optional[np.ndarray]      # uint32 counts (Monte Carlo)
    prob: np.ndarray                  # float64 probability
    r: Optional[int]
    seconds: float
    cached: bool = False


def cache_path(case: Case, s1: S1):
    return OUT / "cache" / case.name / f"{s1.key()}.npz"


_GRAD = {}


def gradient_of(case: Case, name: str) -> np.ndarray:
    key = (case.name, name)
    if key not in _GRAD:
        if len(_GRAD) > 8:
            _GRAD.clear()
        if name == "multitemporal":
            g = zeit.edge_gradient(case.stack, "multitemporal", combine="max").values
        else:
            g = zeit.edge_gradient(case.img, name).values
        _GRAD[key] = np.asarray(g, np.float64)
    return _GRAD[key]


def stage1(case: Case, s1: S1, n_jobs: int = 8, use_cache: bool = True) -> Boundary:
    """The boundary map of ``case`` under ``s1`` (read from the cache when it is there)."""
    p = cache_path(case, s1)
    if use_cache and p.exists():
        z = np.load(p)
        if s1.method == "exact":
            return Boundary(None, z["prob"].astype(np.float64), None, float(z["seconds"]), True)
        counts = z["counts"].astype(np.uint32)
        r = int(z["r"])
        return Boundary(counts, counts.astype(np.float64) / r, r, float(z["seconds"]), True)
    grad = gradient_of(case, s1.gradient)
    keep = case.keep(s1.mask_threshold)
    t0 = time.perf_counter()
    prob, counts, r = _boundary(
        grad, keep, method=s1.method, realizations=s1.realizations, markers=s1.markers,
        min_distance=s1.min_distance, marker_radius=s1.marker_radius, n_neighbors=20, coloring=s1.coloring,
        random_state=s1.seed, n_jobs=n_jobs)
    sec = time.perf_counter() - t0
    if not use_cache:
        return Boundary(counts, prob, r, sec)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    if counts is not None:
        np.savez_compressed(tmp, counts=counts.astype(np.uint16), r=r, seconds=sec)
    else:
        np.savez_compressed(tmp, prob=prob.astype(np.float32), seconds=sec)
    tmp.replace(p)
    return Boundary(counts, prob, r, sec)


def stage2(case: Case, s1: S1, b: Boundary, s2: S2):
    """The labels and ``boundary_label`` from a boundary map: the post-processing of ``zeit.fields``."""
    grad = gradient_of(case, s1.gradient)
    keep = case.keep(s1.mask_threshold)
    if b.counts is not None:
        hyst = _fc.hysteresis_threshold(b.counts.astype(np.float64), s2.low * b.r, s2.high * b.r)
    else:
        hyst = _fc.hysteresis_threshold(b.prob, s2.low, s2.high)
    regions, _ = _fc.label(~hyst, connectivity=1)
    regions = _fc.remove_small_objects(regions + 1, s2.min_size)
    network_kept = bool((regions == 1).any())
    regions, _ = _fc.relabel_sequential(regions)
    surface = case.img if s2.final == "image" else grad
    lab = _fc.watershed(surface, regions, connectivity=1)
    if keep is not None:
        lab = np.where(keep, lab, 0).astype(np.int32)
    boundary_label = int(network_kept and (lab == 1).any())
    lab, _ = _fc.relabel_sequential(lab)
    return lab, boundary_label


_BREF = {}


def _ref_boundary(case: Case, tol):
    key = (case.id, tol)
    if key not in _BREF:
        _BREF[key] = _boundary_pixels(case.ref_labels, case.regions[tol])
    return _BREF[key]


def evaluate(case: Case, lab: np.ndarray, boundary_label: int) -> dict:
    """The responses of one segmentation against the case's reference (suffix ``_t0``..``_t2``: tolerances)."""
    pred = lab.astype(np.int64)
    if boundary_label:
        pred = np.where(pred == boundary_label, 0, pred)
    out = {}
    for tol in TOLERANCES:
        region = case.regions[tol]
        bp = _boundary_pixels(pred, region)
        br = _ref_boundary(case, tol)
        rec = _hit_share(br, bp, tol)
        if case.kind == "fields":
            prec = _hit_share(bp, br, tol)
            if np.isnan(prec) or np.isnan(rec):
                f = np.nan
            else:
                f = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        else:
            prec = f = np.nan
        out.update({f"prec_t{tol}": prec, f"rec_t{tol}": rec, f"f_t{tol}": f})
    acc = zeit.segmentation_accuracy(pred, case.ref_labels, kind=case.kind, tolerance=1, region=case.regions[1],
                                     transform=case.transform, crs=case.crs)
    if case.kind == "fields":
        o, t = acc.objects, acc.size_tests
        out.update({"os": o.over_seg, "us": o.under_seg, "d": o.d, "os_w": o.over_seg_w, "us_w": o.under_seg_w,
                    "d_w": o.d_w, "matched": o.matched_fraction, "iou_fraction": o.iou_fraction,
                    "wasserstein": t.wasserstein, "ks": t.ks, "n_ref": int(o.n_ref), "n_pred": int(o.n_pred),
                    "n_unmatched": int(o.n_unmatched)})
        out.update(variants(case, pred))      # matched_clip, matched_in, n_in (see matched.py)
    else:
        c = acc.crossing
        out.update({"cross_share": c.share, "cross_area_share": c.area_share, "n_pred": int(c.n_pred),
                    "n_crossing": int(c.n_crossing)})
    out["region_pixels"] = int(case.regions[1].sum())
    out["n_labels"] = int(len(np.unique(pred[pred > 0])))
    return out
