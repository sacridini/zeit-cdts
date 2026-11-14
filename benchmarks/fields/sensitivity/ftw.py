"""24f task 5: the Fields of The World global predicted polygons scored like ``zeit.fields``.

    python ftw.py check     # (a) / (b) of the harness equal a direct zeit.fields call, label for label (2 cases)
    python ftw.py run       # every case x source -> ftw/ftw_rows.csv  (resumable by case)
    python ftw.py report    # ../report_ftw.md

Sources (``src``):

* ``ftw2024``, ``ftw2025``: the window's predicted polygons of that year rasterised onto the window grid (the polygons
  reprojected to the window's CRS, ``zeit._segacc._rasterise``, all_touched=False); pixels outside every polygon are 0;
* ``ftw2024_c69`` / ``ftw2025_c69``: the same with confidence >= 69 (polygons with no confidence are dropped), only in
  the groups where the confidence layer exists;
* each of them also with the suffix ``_fill``: the 0 background filled by the nearest label (a Voronoi partition of the
  window), to see what the 0 background does to the boundary and object metrics;
* ``new``: ``zeit.fields`` at the package defaults (multitemporal gradient combined by the mean on the 6-band stack, R 200,
  1000 markers per Mpx, low 0.1, min_size 10, no mask, final on the gradient as the package does for a stack);
* ``new_mask``: the same with the notebook's forest mask swir16 > 2000 (Chaco groups);
* ``notebook``: ``preset="notebook"`` on swir16 (tophat, R 500, 1000 markers, mask 2000, low 0.3, min_size 50, final image).

The responses are those of ``pipeline.evaluate`` (boundary F / precision / recall at 1 px, Clinton D_w, os_w, us_w,
Wasserstein, crossing share) plus the package's ``clip=True, complete_only=True`` matched fraction (``cc_*``).
"""
import pyexpat  # noqa: F401
import os
import sys
import time

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("GDAL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cases as C
import pipeline as P
import s24f as S
from extras import pkg_metrics

FTW = C.DATA / "ftw_predictions.gpkg"
OUTF = C.OUT / "ftw"
ROWS = OUTF / "ftw_rows.csv"
YEARS = (2024, 2025)
MASK_GROUPS = ("chaco", "humidchaco")
S2_NEW = P.S2(low=0.1, high=0.6, min_size=10, final="gradient")
S2_NB = P.S2()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ FTW
def load_ftw():
    import geopandas as gpd

    g = gpd.read_file(FTW)
    return g[["window", "group", "year", "confidence", "geometry"]]


def ftw_labels(case, ftw, year, conf_min=None):
    """The window's polygons of ``year`` (and confidence >= conf_min) on the window grid; also the polygon count."""
    sub = ftw[(ftw["window"] == case.name) & (ftw["year"] == year)]
    if conf_min is not None:
        sub = sub[sub["confidence"] >= conf_min]
    n = int(len(sub))
    if n == 0:
        return np.zeros(case.shape, np.int64), 0
    sub = sub.to_crs(case.crs)
    return C._rasterise(sub, case.shape, case.transform, case.crs), n


def fill_nearest(lab):
    """The 0 background replaced by the nearest label (Euclidean); unchanged when there is no label."""
    from scipy.ndimage import distance_transform_edt

    if not (lab > 0).any():
        return lab
    idx = distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
    return lab[idx[0], idx[1]]


# ------------------------------------------------------------------ zeit.fields through the harness
def s1_new(case, mask=False):
    return S.s1_for(case, "multimean", 1000, "abs2000" if mask else "none")


def s1_notebook():
    return P.S1(gradient="tophat", realizations=500, markers=1000, mask_threshold=2000, seed=0)


def zeit_labels(case, which):
    """``(labels with the boundary network at 0, boundary_label)`` of a zeit.fields configuration."""
    if which == "new":
        s1, s2 = s1_new(case), S2_NEW
    elif which == "new_mask":
        s1, s2 = s1_new(case, True), S2_NEW
    else:
        s1, s2 = s1_notebook(), S2_NB
    b = P.stage1(case, s1)
    lab, bl = P.stage2(case, s1, b, s2)
    return lab, bl


def respond(case, lab, bl=0):
    r = P.evaluate(case, lab, bl)
    pred = lab.astype(np.int64)
    if bl:
        pred = np.where(pred == bl, 0, pred)
    r.update(pkg_metrics(case, pred))
    if case.kind == "property":
        r["f_prop"] = S.fprop(r["rec_t1"], r["cross_share"]) if not np.isnan(r["cross_share"]) else np.nan
    return r


# ------------------------------------------------------------------ check
def check():
    import zeit

    cs = C.load_cases(only={"trazo_chaco_005", "sorriso_1"})
    for c in cs:
        t0 = time.perf_counter()
        for which in ("new", "new_mask", "notebook"):
            lab, bl = zeit_labels(c, which)
            if which == "notebook":
                res = zeit.fields(c.img, preset="notebook", random_state=0, n_jobs=8)
            elif which == "new":
                res = zeit.fields(c.stack, random_state=0, n_jobs=8)
            else:
                res = zeit.fields(c.stack, mask=c.keep(2000), random_state=0, n_jobs=8)
            same = np.array_equal(res["labels"].values, lab) and int(res.attrs["boundary_label"]) == bl
            n_diff = int((res["labels"].values != lab).sum())
            log(f"{c.id} {which}: equal={same} pixels_different={n_diff} of {lab.size} "
                f"(harness {lab.max()} labels, package {int(res['labels'].values.max())}); {time.perf_counter() - t0:.0f} s")


# ------------------------------------------------------------------ run
def run():
    OUTF.mkdir(parents=True, exist_ok=True)
    cs = C.load_cases()
    ftw = load_ftw()
    have = set()
    old = pd.DataFrame()
    if ROWS.exists():
        old = pd.read_csv(ROWS)
        have = set(old["case"])
    has_conf = {g: bool(ftw.loc[ftw["group"] == g, "confidence"].notna().any()) for g in ftw["group"].unique()}
    log("confidence present by group:", has_conf)
    rows = []
    for k, c in enumerate(cs):
        if c.id in have:
            continue
        t0 = time.perf_counter()
        base = {"case": c.id, "mosaic": c.name, "group": c.group, "kind": c.kind, "ref": c.ref,
                "region_pixels": int(c.regions[1].sum())}
        for y in YEARS:
            variants = [("all", None)] + ([("c69", 69)] if has_conf.get(c.group) else [])
            for tag, cm in variants:
                lab, n = ftw_labels(c, ftw, y, cm)
                cover = float((lab > 0).mean())
                for fill in (False, True):
                    src = f"ftw{y}" + ("" if cm is None else "_c69") + ("_fill" if fill else "")
                    pred = fill_nearest(lab) if fill else lab
                    rows.append({**base, "src": src, "n_polygons": n, "cover": cover, **respond(c, pred.astype(np.int64))})
        for which in ("new", "notebook") + (("new_mask",) if c.group in MASK_GROUPS else ()):
            lab, bl = zeit_labels(c, which)
            rows.append({**base, "src": which, "n_polygons": np.nan, "cover": np.nan, **respond(c, lab, bl)})
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(ROWS, index=False)
        log(f"[{k + 1}/{len(cs)}] {c.id}: {time.perf_counter() - t0:.1f} s")
    log("done")


# ------------------------------------------------------------------ report
def _wavg(x, col, w="region_pixels"):
    x = x.dropna(subset=[col])
    return float(np.average(x[col], weights=x[w])) if len(x) and x[w].sum() > 0 else np.nan


def _pooled_matched(x):
    x = x.dropna(subset=["cc_matched"])
    n = x["cc_n_ref"].sum()
    return float((x["cc_matched"] * x["cc_n_ref"]).sum() / n) if n > 0 else np.nan


def report():
    d = pd.read_csv(ROWS)
    # sorriso has two references (CAR properties, pivots): 24f's "sorriso" group is the pivots
    d["group"] = np.where(d["group"] == "sorriso", "sorriso_" + d["ref"], d["group"])
    # a window with no FTW polygon predicts no field: its boundary F is 0 (precision stays undefined), not missing
    gap = d["src"].str.startswith("ftw") & (d["n_polygons"] == 0)
    d.loc[gap & d["f_t1"].isna() & d["rec_t1"].notna(), "f_t1"] = 0.0
    d.loc[gap & d["f_prop"].isna() & d["rec_t1"].notna(), "f_prop"] = 0.0
    d["gap"] = gap
    nm = {"ftw2024": "FTW 2024", "ftw2025": "FTW 2025", "ftw2024_c69": "FTW 2024 conf>=69",
          "ftw2025_c69": "FTW 2025 conf>=69", "new": "zeit.fields new", "new_mask": "zeit.fields new + mask 2000",
          "notebook": "notebook"}
    nm.update({k + "_fill": v + " (background filled)" for k, v in list(nm.items()) if k.startswith("ftw")})
    order = [k for k in nm if k in set(d["src"])]
    groups = [g for g in ("chaco", "humidchaco", "lem", "sorriso_pivots", "porto_murtinho", "sorriso_car") if g in set(d["group"])]
    main5 = [g for g in groups if g != "sorriso_car"]
    out = ["# Fields of The World predictions against zeit.fields (phase 24f, task 5)", "",
           "62 Landsat mosaic windows (64 cases) of the harness; the references and the mosaics are 2019 (Sorriso), 2020 (Chaco, "
           "LEM) and 2022 (Humid Chaco), the FTW polygons 2024 and 2025. Scores as in 24f: boundary F, precision, recall at 1 px "
           "(fields references), D_w (Clinton, area-weighted, lower is better), over- and under-segmentation (os_w, us_w), the "
           "Wasserstein distance of the size distributions, `matched` with `clip=True, complete_only=True` pooled over the group's "
           "complete reference fields (`cc_matched`); CAR (property) groups: recall at 1 px, the share of crossing objects and "
           "their harmonic mean `F(rec x non-crossing)`; `sorriso_pivots` is 24f's Sorriso group, `sorriso_car` is shown apart and left out of "
           "the means. A window with no FTW polygon (a coverage gap) counts as a prediction of no field: F 0, D_w 1, recall 0 (the "
           "`gaps` column counts them; without this rule the FTW means would be taken over the covered windows only). A group's value is the mean of its windows weighted by the region's pixels "
           "(as in 24f); `n` is the number of windows with a defined value. The FTW 0 background (no polygon) is label 0, "
           "so field/background edges are boundaries on both sides; `(background filled)` rows fill it by the nearest label.", ""]
    for g in groups:
        x = d[d["group"] == g]
        prop = x["kind"].iloc[0] == "property"
        out += [f"## {g} ({x['mosaic'].nunique()} windows, {x['ref'].iloc[0]} reference)", ""]
        if prop:
            cols = [("rec_t1", "recall"), ("cross_share", "crossing"), ("f_prop", "F(rec x non-cross)")]
        else:
            cols = [("f_t1", "F"), ("prec_t1", "precision"), ("rec_t1", "recall"), ("d_w", "D_w"), ("os_w", "os_w"),
                    ("us_w", "us_w"), ("wasserstein", "Wasserstein"), ("cc_matched", "cc_matched")]
        out += ["| source | n | " + " | ".join(h for _, h in cols) + " | polygons/window | covered share | windows without FTW |",
                "|---|---|" + "---|" * (len(cols) + 3)]
        for s in order:
            y = x[x["src"] == s]
            if not len(y):
                continue
            vals = []
            for c, _ in cols:
                v = _pooled_matched(y) if c == "cc_matched" else _wavg(y, c)
                vals.append(f"{v:.3f}")
            n = int(y[cols[0][0]].notna().sum())
            npol = y["n_polygons"].mean()
            cov = y["cover"].mean()
            out.append(f"| {nm[s]} | {n} | " + " | ".join(vals) + (
                f" | {npol:.0f} | {cov:.2f} | {int(y['gap'].sum())} |" if not np.isnan(npol) else " | | | |"))
        out.append("")
        score = "f_prop" if prop else "f_t1"
        out += [f"Per-case spread of {'F(rec x non-crossing)' if prop else 'F'} and of {'recall' if prop else 'D_w'} "
                f"(median, interquartile range, number of windows):", "",
                f"| source | median {score} | IQR | n | median {'rec_t1' if prop else 'D_w'} | IQR |", "|---|---|---|---|---|---|"]
        sc2 = "rec_t1" if prop else "d_w"
        for s in order:
            y = x[x["src"] == s]
            if not len(y):
                continue
            a, b = y[score].dropna(), y[sc2].dropna()
            q = lambda v: (f"{v.quantile(.25):.3f} to {v.quantile(.75):.3f}" if len(v) else "")
            out.append(f"| {nm[s]} | {a.median():.3f} | {q(a)} | {len(a)} | {b.median():.3f} | {q(b)} |")
        out.append("")
    # all groups together, mean of the group means (24f's "mean" column)
    out += ["## Mean over the five groups of 24f (F; Porto Murtinho: F(rec x non-crossing); D_w over the four fields groups)", "",
            "| source | mean score | mean D_w (4 groups) |", "|---|---|---|"]
    for s in order:
        fs, ds = [], []
        for g in main5:
            y = d[(d["group"] == g) & (d["src"] == s)]
            if not len(y):
                continue
            prop = y["kind"].iloc[0] == "property"
            fs.append(_wavg(y, "f_prop" if prop else "f_t1"))
            if not prop:
                ds.append(_wavg(y, "d_w"))
        if fs:
            out.append(f"| {nm[s]} | {np.nanmean(fs):.3f} ({len(fs)} groups) | {np.nanmean(ds):.3f} |")
    out.append("")
    out += ["## Caveats", "",
            "* Year: the references and mosaics are 2019 (Sorriso), 2020 (Chaco, LEM) and 2022 (Humid Chaco); the FTW polygons are 2024 "
            "and 2025, so fields (and the Chaco clearings) have changed in between: the FTW scores are biased down, and the two FTW "
            "years differ by up to 0.1 of F in the same window.",
            "* Resolution: FTW predicts on 10 m Sentinel-2; rasterised on the 30 m grid, small fields vanish (a polygon under about "
            "one pixel may rasterise to nothing) or merge, and its thin gaps between neighbours become single pixels.",
            "* Coverage gaps: the windows without a polygon (the `windows without FTW` column) count as F 0, D_w 1; the confidence "
            "layer exists only in the Chaco groups (about 89 to 91 % of the polygons there), so the conf >= 69 rows are Chaco and "
            "Humid Chaco only and drop the polygons without confidence; the 5-group mean has two groups for those rows and is "
            "not comparable with the others.",
            "* Background: label 0 outside every polygon makes every field edge a boundary against 0 on the prediction side, but "
            "only the part of the reference boundary inside the region counts, and the references (Trazo footprints, FTW 2020 "
            "fields, CAR properties) have their own 0 background; filling it by the nearest label removes those edges and "
            "creates boundaries in the middle of the gaps: it lowers F (precision and recall) in Chaco, Humid Chaco and LEM, so "
            "the 0 background does not favour FTW on the boundary metrics (it helps it, if anything); the fill trades over- for "
            "under-segmentation (os_w down, us_w up).",
            "* Small groups: LEM is one window, Sorriso two, Porto Murtinho two; per-case spread is shown but not meaningful there. "
            "The LEM reference comes from the Fields of The World dataset (2020), whose labels may overlap the model's training data (not checked).",
            ""]
    (C.OUT / "report_ftw.md").write_text("\n".join(out), encoding="utf-8")
    log("wrote", C.OUT / "report_ftw.md")


if __name__ == "__main__":
    {"check": check, "run": run, "report": report}[sys.argv[1]]()
