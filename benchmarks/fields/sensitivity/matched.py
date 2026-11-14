"""One-to-one matching at IoU >= 0.5 and its corrected variants (harness only; the package is not touched).

``segmentation_accuracy`` counts a predicted object in full (the whole grid) and the reference only inside the
region. When the region is smaller than the window (the Trazo chip footprints) a predicted field that runs out of
the footprints has a large |p| and a low IoU with a reference field that is complete inside. Variants:

* ``matched``: as ``zeit.segmentation_accuracy`` (|p| in full).
* ``matched_clip``: the predicted objects clipped to the region too (the reference is only known there).
* ``matched_in``: ``matched_clip``, counted only over the reference fields that are fully inside the region and
  do not touch the window's edge (a field cut by either is not a field of the reference).
"""

import numpy as np


_INSIDE = {}


def inside_ids(case) -> np.ndarray:
    """Ids of the reference fields fully inside the region (tolerance 1) and away from the window's edge."""
    if case.id not in _INSIDE:
        _INSIDE[case.id] = _inside_ids(case)
    return _INSIDE[case.id]


def _inside_ids(case) -> np.ndarray:
    ref, region = case.ref_labels, case.regions[1]
    ids = np.unique(ref[ref > 0])
    if len(ids) == 0:
        return ids
    out_ids = np.unique(ref[(ref > 0) & ~region])
    edge = np.concatenate([ref[0], ref[-1], ref[:, 0], ref[:, -1]])
    bad = np.union1d(out_ids, edge[edge > 0])
    return np.setdiff1d(ids, bad)


def match(pred: np.ndarray, ref: np.ndarray, region: np.ndarray, clip: bool, thr: float = 0.5):
    """(ids of the reference objects, their one-to-one IoU, best IoU of each) by descending IoU.

    The reference is taken inside ``region``; the predicted objects in full or, with ``clip``, inside it.
    """
    p_all = np.where(region, pred, 0) if clip else pred
    rr = np.where(region, ref, 0)
    r_ids, r_n = np.unique(rr[rr > 0], return_counts=True)
    p_flat = p_all[p_all > 0]
    p_ids, p_n = np.unique(p_flat, return_counts=True)
    m = (rr > 0) & (p_all > 0)
    if not m.any():
        return r_ids, np.zeros(len(r_ids)), np.zeros(len(r_ids))
    code = rr[m].astype(np.int64) * (int(p_all.max()) + 1) + p_all[m]
    cu, cn = np.unique(code, return_counts=True)
    r = cu // (int(p_all.max()) + 1)
    p = cu % (int(p_all.max()) + 1)
    rn = r_n[np.searchsorted(r_ids, r)]
    pn = p_n[np.searchsorted(p_ids, p)]
    iou = cn / (rn + pn - cn)
    order = np.lexsort((p, r, -iou))
    used_r, used_p, pair = set(), set(), {}
    for i in order:
        a, b = r[i], p[i]
        if a in used_r or b in used_p:
            continue
        used_r.add(a)
        used_p.add(b)
        pair[a] = iou[i]
    one = np.array([pair.get(a, 0.0) for a in r_ids])
    best = np.zeros(len(r_ids))
    np.maximum.at(best, np.searchsorted(r_ids, r), iou)
    return r_ids, one, best


def variants(case, pred: np.ndarray, thr: float = 0.5) -> dict:
    """``matched_clip``, ``matched_in`` and the number of reference fields behind ``matched_in``."""
    region = case.regions[1]
    ids, one, _ = match(pred, case.ref_labels, region, clip=True)
    if len(ids) == 0:
        return {"matched_clip": np.nan, "matched_in": np.nan, "n_in": 0}
    keep = np.isin(ids, inside_ids(case))
    return {"matched_clip": float((one >= thr).mean()),
            "matched_in": float((one[keep] >= thr).mean()) if keep.any() else np.nan,
            "n_in": int(keep.sum())}
