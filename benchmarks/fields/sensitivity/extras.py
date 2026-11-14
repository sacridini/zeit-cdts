"""The expensive responses, computed only at selected settings: ``zeit.segmentation_accuracy`` with the package's ``clip`` and
``complete_only`` (24f), next to the harness's own ``matched.py`` variants, to check that they agree."""
import numpy as np

import zeit
from matched import variants


def pkg_metrics(case, pred: np.ndarray) -> dict:
    """``pred`` has the boundary network already at 0. Fields: the package's metrics unclipped, and with clip=True,
    complete_only=True (``cc_*``), plus ``matched.py``'s ``matched_clip`` / ``matched_in``; property: the crossing share
    unclipped and with clip=True."""
    kw = dict(kind=case.kind, tolerance=1, region=case.regions[1], transform=case.transform, crs=case.crs)
    out = {}
    if case.kind == "fields":
        a = zeit.segmentation_accuracy(pred, case.ref_labels, **kw)
        c = zeit.segmentation_accuracy(pred, case.ref_labels, clip=True, complete_only=True, **kw)
        o, oc = a.objects, c.objects
        out.update({"matched_pkg": o.matched_fraction, "iou_pkg": o.iou_fraction, "d_w_pkg": o.d_w, "n_ref_pkg": o.n_ref,
                    "cc_matched": oc.matched_fraction, "cc_iou": oc.iou_fraction, "cc_d_w": oc.d_w,
                    "cc_under_w": oc.under_seg_w, "cc_over_w": oc.over_seg_w, "cc_n_ref": int(oc.n_ref),
                    "cc_n_ref_all": int(oc.get("n_ref_all", oc.n_ref))})
        v = variants(case, pred)
        out.update({"mp_matched_clip": v["matched_clip"], "mp_matched_in": v["matched_in"], "mp_n_in": v["n_in"]})
    else:
        a = zeit.segmentation_accuracy(pred, case.ref_labels, **kw)
        c = zeit.segmentation_accuracy(pred, case.ref_labels, clip=True, **kw)
        out.update({"cross_pkg": a.crossing.share, "cc_cross": c.crossing.share})
    return out
