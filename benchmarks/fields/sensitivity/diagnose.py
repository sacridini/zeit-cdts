"""Why is ``matched`` (one to one at IoU >= 0.5) so low? Diagnosis on cached notebook-parameter maps (R=200, seed 0).

    python diagnose.py            # writes sensitivity/matched_diagnosis.csv and .md

Per case: the IoU of each reference field, how many reference fields the window or the region cuts, whether the
failures are splits or merges, how much of the predicted objects lies outside the region, how much of the reference
the mask or the boundary network eats, and ``matched`` against the corrected variants of ``matched.py``.
"""
import pyexpat  # noqa: F401
import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cases as C
import matched as M
import pipeline as P

PICK = ["trazo_chaco_012","trazo_chaco_008", "trazo_chaco_016", "trazo_chaco_024", "trazo_chaco_032",
        "trazo_chaco_040", "trazo_humidchaco_007", "lem_1"]


def one_case(case):
    s1, s2 = P.S1(realizations=200, seed=0), P.S2()
    b = P.stage1(case, s1)
    lab, bl = P.stage2(case, s1, b, s2)
    pred = lab.astype(np.int64)
    if bl:
        pred = np.where(pred == bl, 0, pred)
    ev = P.evaluate(case, lab, bl)
    import zeit

    acc = zeit.segmentation_accuracy(pred, case.ref_labels, kind="fields", tolerance=1, region=case.regions[1],
                                     transform=case.transform, crs=case.crs)
    tab = acc.object_table
    region, ref = case.regions[1], case.ref_labels
    ids_all = np.unique(ref[ref > 0])
    out_ids = np.unique(ref[(ref > 0) & ~region])
    edge = np.concatenate([ref[0], ref[-1], ref[:, 0], ref[:, -1]])
    edge_ids = np.unique(edge[edge > 0])
    inside = M.inside_ids(case)
    # predicted objects: share of the best match outside the region
    p_full = pd.Series(pred[pred > 0]).value_counts()
    p_reg = pd.Series(pred[region & (pred > 0)]).value_counts()
    leak = 1 - (p_reg.reindex(tab["best"]).fillna(0).to_numpy() / p_full.reindex(tab["best"]).to_numpy())
    leak = leak[tab["best"].to_numpy() > 0]
    iou = tab["iou"].to_numpy()
    low = iou < 0.5
    split = low & (tab["over_seg"].to_numpy() > tab["under_seg"].to_numpy())
    merge = low & ~split
    ref_px = region & (ref > 0)
    ids, one, best = M.match(pred, ref, region, clip=False)
    idc, onec, bestc = M.match(pred, ref, region, clip=True)
    kin = np.isin(idc, inside)
    row = {
        "case": case.id, "group": case.group, "window_px": ref.size, "region_share": float(region.mean()),
        "n_ref_total": len(ids_all), "n_ref_in_region": len(tab),
        "cut_by_region": len(np.intersect1d(out_ids, ids_all)), "touch_window_edge": len(edge_ids),
        "fully_inside": len(inside),
        "ref_px_median": float(np.median(tab["pixels"])),
        "pred_px_median_full": float(np.median(tab.loc[tab["best"] > 0, "pred_pixels"])),
        "pred_leak_median": float(np.median(leak)) if len(leak) else np.nan,
        "iou_q25": float(np.quantile(iou, 0.25)), "iou_median": float(np.median(iou)),
        "iou_q75": float(np.quantile(iou, 0.75)),
        "share_iou_lt_05": float(low.mean()), "share_split": float(split.mean()), "share_merge": float(merge.mean()),
        "over_seg": float(tab["over_seg"].mean()), "under_seg": float(tab["under_seg"].mean()),
        "ref_px_pred_zero": float((pred[ref_px] == 0).mean()),
        "ref_px_masked": float((lab[ref_px] == 0).mean()),
        "matched": ev["matched"], "iou_fraction": ev["iou_fraction"], "f_t1": ev["f_t1"],
        "matched_clip": float((onec >= 0.5).mean()), "matched_in": float((onec[kin] >= 0.5).mean()) if kin.any() else np.nan,
        "iou_fraction_clip": float((bestc >= 0.5).mean()),
        "iou_fraction_in": float((bestc[kin] >= 0.5).mean()) if kin.any() else np.nan,
        "iou_median_in": float(np.median(tab.loc[np.isin(tab.index, inside), "iou"])) if kin.any() else np.nan,
        "n_in": int(kin.sum()),
    }
    # (a) the boundary network's pixels given to the nearest object; (b) the predicted objects cut to the
    # reference's fields (what is left once the unmapped land around them is ignored); (c) both
    filled = pred
    if bl:
        from scipy.ndimage import distance_transform_edt

        free = lab == bl
        idx = distance_transform_edt(pred == 0, return_distances=False, return_indices=True)
        filled = np.where(free, pred[tuple(idx)], pred)
    fg = ref > 0
    for name, p in (("fill", filled), ("fg", np.where(fg, pred, 0)), ("fillfg", np.where(fg, filled, 0))):
        i2, o2, _ = M.match(p, ref, region, clip=True)
        k2 = np.isin(i2, inside)
        row[f"matched_{name}"] = float((o2 >= 0.5).mean())
        row[f"matched_in_{name}"] = float((o2[k2] >= 0.5).mean()) if k2.any() else np.nan
    row["ref_share_of_region"] = float(fg[region].mean())
    row["n_pred_in_region"] = int(len(np.unique(pred[region & (pred > 0)])))
    row["boundary_label_share_region"] = float((lab[region] == bl).mean()) if bl else 0.0
    # how many "touching" reference fields share a straight edge with another reference field (chip seams)
    return row, tab


def main():
    cs = C.load_cases(only=set(PICK))
    rows = []
    for c in cs:
        r, _ = one_case(c)
        rows.append(r)
        print(c.id, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "case"}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT / "matched_diagnosis.csv", index=False)
    show = ["case", "n_ref_in_region", "cut_by_region", "fully_inside", "ref_share_of_region", "ref_px_median",
            "pred_px_median_full", "pred_leak_median", "iou_median", "share_split", "share_merge",
            "ref_px_pred_zero", "matched", "matched_clip", "matched_in", "matched_fill", "matched_fg",
            "matched_fillfg", "matched_in_fillfg", "f_t1"]
    import warnings  # noqa: F401

    from summary import md

    (C.OUT / "matched_diagnosis.md").write_text(md(df[show], "{:.3f}"), encoding="utf-8")


if __name__ == "__main__":
    main()
