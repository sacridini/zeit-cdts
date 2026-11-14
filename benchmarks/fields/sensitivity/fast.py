"""The cheap responses of the 24f search, equal to ``pipeline.evaluate`` for the ones they give.

``pipeline.evaluate`` calls ``zeit.segmentation_accuracy`` (pandas, size tests, three tolerances) and costs about 70 ms per
case; the search over thousands of settings only needs the boundary F / recall at 1 px, the area-weighted Clinton D and the
share of crossing objects. ``test_fast.py`` checks them against ``pipeline.evaluate`` (equal to 1e-12).
The expensive extras (``matched``, with the package's ``clip`` / ``complete_only``) are computed only at the selected
settings, by ``zeit.segmentation_accuracy`` itself.
"""
import numpy as np

from zeit._segacc import _boundary_pixels

MIN_SHARE = 0.1     # segmentation_accuracy's default for the crossing share


def _near1(b: np.ndarray) -> np.ndarray:
    """Pixels within Euclidean distance 1 of ``b`` (itself and its 4 neighbours): equals ``edt(~b) <= 1``."""
    out = b.copy()
    out[:-1] |= b[1:]
    out[1:] |= b[:-1]
    out[:, :-1] |= b[:, 1:]
    out[:, 1:] |= b[:, :-1]
    return out


def hit_share1(a: np.ndarray, b: np.ndarray) -> float:
    n = int(a.sum())
    if n == 0:
        return float("nan")
    if not b.any():
        return 0.0
    return float((a & _near1(b)).sum()) / n


_REF = {}


def _ref_cache(case):
    if case.id not in _REF:
        region = case.regions[1]
        ref_eff = np.where(region, case.ref_labels, 0)
        sizes = np.bincount(ref_eff.ravel())
        _REF[case.id] = (_boundary_pixels(case.ref_labels, region), ref_eff, sizes, ref_eff > 0)
    return _REF[case.id]


def _pairs(ref_eff, has, pred):
    """(r, p, inter) of the overlaps with p > 0, over the reference pixels inside the region."""
    r, p = ref_eff[has], pred[has]
    ok = p > 0
    r, p = r[ok].astype(np.int64), p[ok].astype(np.int64)
    if len(r) == 0:
        z = np.zeros(0, np.int64)
        return z, z, z
    base = int(p.max()) + 1
    cu, cn = np.unique(r * base + p, return_counts=True)
    return cu // base, cu % base, cn


def fast_eval(case, pred: np.ndarray) -> dict:
    """``f`` (boundary F at 1 px, fields), ``rec`` (recall at 1 px), ``d_w`` (fields) or ``cross_share`` (property).

    ``pred`` is the label image with the boundary network already set to 0, int64."""
    region = case.regions[1]
    br, ref_eff, ref_sizes, has = _ref_cache(case)
    bp = _boundary_pixels(pred, region)
    rec = hit_share1(br, bp)
    out = {"rec": rec, "n_labels": int(len(np.unique(pred[pred > 0])))}
    r, p, inter = _pairs(ref_eff, has, pred)
    pred_sizes = np.bincount(pred.ravel())
    if case.kind == "fields":
        prec = hit_share1(bp, br)
        if np.isnan(prec) or np.isnan(rec):
            f = np.nan
        else:
            f = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        out["f"] = f
        ids = np.flatnonzero(ref_sizes)
        ids = ids[ids > 0]
        if len(ids) == 0:
            out["d_w"] = np.nan
            return out
        # best match: the largest overlap, the lowest label on a tie
        order = np.lexsort((p, -inter, r))
        r_o, p_o, i_o = r[order], p[order], inter[order]
        first = np.ones(len(r_o), bool)
        first[1:] = r_o[1:] != r_o[:-1]
        best_p = np.zeros(len(ref_sizes), np.int64)
        best_i = np.zeros(len(ref_sizes), np.int64)
        best_p[r_o[first]] = p_o[first]
        best_i[r_o[first]] = i_o[first]
        rs = ref_sizes[ids].astype(float)
        bp_, bi = best_p[ids], best_i[ids].astype(float)
        over = np.where(bp_ > 0, 1 - bi / rs, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            under = np.where(bp_ > 0, 1 - bi / pred_sizes[bp_].astype(float).clip(min=1), 1.0)
        d = np.sqrt((over ** 2 + under ** 2) / 2)
        out["d_w"] = float(np.average(d, weights=rs))
    else:
        out["f"] = np.nan
        out["d_w"] = np.nan
        if len(r) == 0:
            out["cross_share"] = np.nan
            return out
        order = np.lexsort((-inter, p))
        p_o, i_o = p[order], inter[order]
        first = np.ones(len(p_o), bool)
        first[1:] = p_o[1:] != p_o[:-1]
        start = np.flatnonzero(first)
        n = len(start)
        count = np.diff(np.append(start, len(p_o)))
        second = np.where(count > 1, i_o[np.minimum(start + 1, len(p_o) - 1)], 0)
        cross = (count > 1) & (second >= MIN_SHARE * pred_sizes[p_o[start]])
        out["cross_share"] = float(cross.sum()) / n
    return out
