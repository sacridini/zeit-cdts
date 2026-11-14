"""Threshold choice by spatial cross-validation (phase 24e, task C).

    python cv.py run [--n-jobs 8]      # stage-1 maps (R=200, notebook markers) for 2 gradients x 5 masks, then the
                                       # stage-2 grid low x min_size on each: rows of results.jsonl tagged "cv"
    python cv.py analyze               # cv_results.csv, cv_report.md and plots/cv_heldout.png

Grid: gradient {tophat, sobel}, mask {none, relative q = 20 / 35 / 50 (swir16 > the window's q-th percentile of valid
pixels), absolute 2000}, low {0.05 .. 0.5}, min_size {0 .. 200}, high = max(0.6, low + 0.1), final = image; stage 1
at R = 200, 1000 markers, min_distance 20, marker_radius 3, knn colouring, seed 0 (the notebook's values).

Scores of a window set: ``f_t1`` (boundary F at 1 px) for fields references; for a property reference (CAR) the
harmonic mean of ``rec_t1`` and ``1 - cross_share`` (``fprop``; recall alone is maximised by cutting everything);
the second objective is ``d_w`` (area-weighted Clinton D, lower is better), fields references only. A group's score
is the pixel-weighted mean of its windows; the selection objective of several groups is the mean of their scores
(a group, not a window, is the unit, as chaco has 45 windows and lem one).

Folds: leave one group out (chaco, humidchaco, lem, sorriso, porto_murtinho; sorriso scored on its pivots,
porto_murtinho on the CAR), and within chaco ``zeit.spatial_folds`` of the 45 window centroids (k-means clusters
and hexagonal blocks, k = 5) against random folds.
"""
import pyexpat  # noqa: F401
import argparse
import json
import os
import sys
import time
from dataclasses import replace

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("GDAL_NUM_THREADS", "4")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cases as C
import morris as MO
import pipeline as P

GRADS = ("tophat", "sobel")
MASKS = ("none", "rel20", "rel35", "rel50", "abs2000")
LOWS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
SIZES = (0, 10, 25, 50, 100, 200)
DEFAULT = ("tophat", "abs2000", 0.3, 50)
BASE = P.S1(realizations=200, markers=1000, min_distance=20, marker_radius=3, coloring="knn", seed=0)
GROUPS = ("chaco", "humidchaco", "lem", "sorriso", "porto_murtinho")


def s1_for(meta: dict, grad: str, mask: str) -> P.S1:
    thr = {"none": None, "abs2000": 2000.0}.get(mask)
    if mask.startswith("rel"):
        thr = float(meta["q"][mask[3:]])
    return replace(BASE, gradient=grad, mask_threshold=thr)


def s2_grid():
    return [P.S2(low=lo, high=round(max(0.6, lo + 0.1), 4), min_size=ms) for lo in LOWS for ms in SIZES]


def run(n_jobs: int):
    import run as RUN

    cs = C.load_cases()
    RUN.log(f"{len(cs)} cases; {len(GRADS) * len(MASKS)} stage-1 maps x {len(s2_grid())} stage-2 settings")
    done = RUN.done_keys()
    with open(RUN.RESULTS, "a", encoding="utf-8") as out:
        for grad in GRADS:
            for mask in MASKS:
                t0 = time.perf_counter()
                n = 0
                for case in cs:
                    s1 = s1_for(MO.meta_of_case(case), grad, mask)
                    todo = [s2 for s2 in s2_grid() if (case.id, s1.key(), s2.key()) not in done]
                    if not todo:
                        continue
                    b = P.stage1(case, s1, n_jobs=n_jobs)
                    for s2 in todo:
                        tt = time.perf_counter()
                        lab, bl = P.stage2(case, s1, b, s2)
                        resp = P.evaluate(case, lab, bl)
                        row = RUN.row_base(case, s1, s2, b)
                        row.update({"study": "cv", "cv_gradient": grad, "cv_mask": mask})
                        row.update(resp)
                        row["s2_seconds"] = time.perf_counter() - tt
                        out.write(json.dumps(row, default=lambda o: None if o is None else float(o)) + "\n")
                        done.add((case.id, s1.key(), s2.key()))
                        n += 1
                    out.flush()
                RUN.log(f"  {grad} {mask}: {n} rows in {time.perf_counter() - t0:.0f}s")
    RUN.log("done")


# ------------------------------------------------------------------ analysis
def long_table() -> pd.DataFrame:
    """One row per window x (gradient, mask, low, min_size) of the grid, with the scores."""
    import summary as S

    df = S.load()
    meta = MO.case_meta()
    key2cfg = {}
    for name, m in meta.items():
        for grad in GRADS:
            for mask in MASKS:
                key2cfg[(name, s1_for(m, grad, mask).key())] = (grad, mask)
    s2cfg = {s2.key(): (s2.low, s2.min_size) for s2 in s2_grid()}
    df = df[df["s2_key"].isin(s2cfg)].copy()
    cfg = [key2cfg.get((n, k)) for n, k in zip(df["mosaic"], df["s1_key"])]
    ok = np.array([c is not None for c in cfg])
    df = df[ok].copy()
    cfg = [c for c in cfg if c is not None]
    df["grad"] = [c[0] for c in cfg]
    df["mask"] = [c[1] for c in cfg]
    df["low"] = [s2cfg[k][0] for k in df["s2_key"]]
    df["ms"] = [s2cfg[k][1] for k in df["s2_key"]]
    rec, cs = df["rec_t1"], df["cross_share"] if "cross_share" in df else np.nan
    prop = df["kind"] == "property"
    fp = 2 * rec * (1 - cs) / (rec + (1 - cs))
    df["f"] = np.where(prop, fp, df["f_t1"])
    df["d"] = np.where(prop, np.nan, df["d_w"])
    return df[["case", "mosaic", "group", "gk", "kind", "grad", "mask", "low", "ms", "region_pixels", "f", "d"]]


def _wmean(g: pd.DataFrame, col: str) -> float:
    ok = g[col].notna()
    return float(np.average(g.loc[ok, col], weights=g.loc[ok, "region_pixels"])) if ok.any() else np.nan


def score_cube(t: pd.DataFrame, col: str):
    """(unit -> config -> score) for the pixel-weighted mean of ``col`` over the rows of each unit."""
    cfgs = ["grad", "mask", "low", "ms"]
    out = {}
    for unit, g in t.groupby("unit"):
        s = g.groupby(cfgs).apply(lambda x: _wmean(x, col), include_groups=False)
        out[unit] = s
    return pd.DataFrame(out)        # rows: config tuples, columns: units


def _fmt(cfg) -> str:
    g, m, lo, ms = cfg
    return f"{g}/{m}/low {lo:g}/min_size {ms}"


def wmean_rows(v: pd.DataFrame, w: np.ndarray) -> pd.Series:
    """Weighted mean over the columns of each row, leaving out NaN."""
    a = v.to_numpy(float)
    ok = ~np.isnan(a)
    den = (ok * w).sum(axis=1)
    num = (np.where(ok, a, 0) * w).sum(axis=1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), index=v.index).dropna()


def pick(scores: pd.DataFrame, units, sign: int, grad_only=None, weights=None):
    """The config with the best (weighted) mean score over ``units`` (sign +1 maximise, -1 minimise)."""
    w = np.array([1.0 if weights is None else weights[u] for u in units])
    s = wmean_rows(scores[list(units)], w)
    if grad_only is not None:
        s = s[[c[0] == grad_only for c in s.index]]
    return (s * sign).idxmax()


def evaluate_folds(scores: pd.DataFrame, folds: dict, weights: dict, sign: int, label: str, fixed_grad=None,
                   weight_train=False):
    """Per fold: the config chosen on the others, its held-out score, the default's and the in-sample optimum's.

    ``folds`` maps a fold name to its units; the others are the training units. Held-out score of a fold with
    several units: weighted mean (``weights``)."""
    rows = []
    allu = [u for us in folds.values() for u in us]
    for name, us in folds.items():
        train = [u for u in allu if u not in us]
        sel = pick(scores, train, sign, fixed_grad, weights if weight_train else None)
        w = np.array([weights[u] for u in us], float)

        def held(cfg):
            v = scores.loc[[cfg], us].to_numpy(float)[0]
            ok = ~np.isnan(v)
            return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan

        ins = wmean_rows(scores[us], w)
        if fixed_grad is not None:
            ins = ins[[c[0] == fixed_grad for c in ins.index]]
        best = (ins * sign).idxmax()
        rows.append({"scheme": label, "fold": name, "n_units": len(us), "selected": _fmt(sel), "selected_cfg": sel,
                     "held_out": held(sel), "default": held(DEFAULT), "in_sample": held(best),
                     "in_sample_cfg": _fmt(best)})
    return pd.DataFrame(rows)


def window_centroids(names):
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import Point

    wins = C.windows()
    pts = []
    for n in names:
        w = wins[n]
        x, y = (w["bounds"][0] + w["bounds"][2]) / 2, (w["bounds"][1] + w["bounds"][3]) / 2
        lon, lat = Transformer.from_crs(w["epsg"], 4326, always_xy=True).transform(x, y)
        pts.append(Point(lon, lat))
    return gpd.GeoDataFrame({"name": list(names)}, geometry=pts, crs=4326)


def analyze():
    import warnings

    warnings.filterwarnings("ignore")
    t = long_table()
    if t.empty:
        raise SystemExit("no cv rows: run `python cv.py run` first")
    t.to_csv(C.OUT / "cv_long.csv", index=False)
    results = []
    # ---- leave one group out
    lg = t.copy()
    lg["unit"] = lg["group"]
    # sorriso: the pivots (fields); its CAR windows are left out of the group score
    lg = lg[~((lg["group"] == "sorriso") & (lg["kind"] == "property"))]
    wts = lg.groupby("unit")["region_pixels"].sum().to_dict()
    for col, sign, label, units in (("f", 1, "F", GROUPS), ("d", -1, "D", GROUPS[:4])):
        sc = score_cube(lg, col)
        folds = {u: [u] for u in units}
        for fixed in (None, "tophat"):
            r = evaluate_folds(sc.loc[:, list(units)], folds, wts, sign,
                               f"LOGO {label}" + ("" if fixed is None else " (tophat only)"), fixed)
            results.append(r)
        pooled = pick(sc, units, sign)
        print(f"pooled optimum ({label}):", _fmt(pooled))
    # ---- within chaco
    ch = t[t["group"] == "chaco"].copy()
    names = sorted(ch["mosaic"].unique())
    ch["unit"] = ch["mosaic"]
    wts_w = ch.groupby("unit")["region_pixels"].first().to_dict()
    import zeit

    pts = window_centroids(names)
    nfold = 5
    for col, sign, label in (("f", 1, "F"), ("d", -1, "D")):
        sc = score_cube(ch, col)
        schemes = {}
        for meth in ("clusters", "blocks"):
            kw = dict(method=meth, k=nfold, seed=0, n_jobs=2)
            if meth == "blocks":
                xy = pts.to_crs(pts.estimate_utm_crs()).geometry
                size = 0.35 * max(xy.x.max() - xy.x.min(), xy.y.max() - xy.y.min())
                kw.update(size=float(size), iterations=200)
            f = zeit.spatial_folds(pts, **kw)
            schemes[meth] = np.asarray(f.folds)
        rng = np.random.default_rng(0)
        rf = np.arange(len(names)) % nfold
        rng.shuffle(rf)
        schemes["random"] = rf
        for meth, fl in schemes.items():
            folds = {f"{meth} {i}": [names[j] for j in np.where(fl == i)[0]] for i in np.unique(fl)}
            r = evaluate_folds(sc, folds, wts_w, sign, f"chaco {meth} {label}", weight_train=True)
            results.append(r)
    res = pd.concat(results, ignore_index=True)
    res.drop(columns="selected_cfg").to_csv(C.OUT / "cv_results.csv", index=False)
    cv_plot(res)
    (C.OUT / "cv_report.md").write_text(cv_markdown(res), encoding="utf-8")
    print("cv_results.csv, cv_report.md, plots/cv_heldout.png written")


def cv_markdown(res: pd.DataFrame) -> str:
    import summary as S

    out = []
    for scheme, g in res.groupby("scheme", sort=False):
        show = g[["fold", "n_units", "selected", "held_out", "default", "in_sample"]].copy()
        mean = {"fold": "mean", "n_units": "", "selected": "", "held_out": g["held_out"].mean(),
                "default": g["default"].mean(), "in_sample": g["in_sample"].mean()}
        show = pd.concat([show, pd.DataFrame([mean])], ignore_index=True)
        out.append(f"**{scheme}**\n\n" + S.md(show, "{:.3f}"))
    return "\n\n".join(out)


def cv_plot(res: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = list(dict.fromkeys(res["scheme"]))
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    for ax, scheme in zip(axes.ravel(), schemes):
        g = res[res["scheme"] == scheme]
        x = np.arange(len(g))
        w = 0.27
        ax.bar(x - w, g["default"], w, label="notebook defaults")
        ax.bar(x, g["held_out"], w, label="chosen on the other folds")
        ax.bar(x + w, g["in_sample"], w, label="in-sample optimum")
        ax.set_xticks(x)
        ax.set_xticklabels([f.split(" ")[-1] if scheme.startswith("chaco") else f for f in g["fold"]], fontsize=7,
                           rotation=30)
        ax.set_title(scheme, fontsize=8)
    axes[0, 0].legend(fontsize=6)
    for ax in axes.ravel()[len(schemes):]:
        ax.axis("off")
    fig.suptitle("held-out score (F at 1 px; D: lower is better) by fold")
    fig.tight_layout()
    d = C.OUT / "plots"
    d.mkdir(exist_ok=True)
    fig.savefig(d / "cv_heldout.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyze"])
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()
    run(a.n_jobs) if a.cmd == "run" else analyze()
