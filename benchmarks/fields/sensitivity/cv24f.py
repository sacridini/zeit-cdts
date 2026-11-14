"""24f, tasks A and C: the extended spatial cross-validation on the grids of ``a24f.py`` / ``c24f.py``.

    python cv24f.py A            # f24/A_report.md, f24/A_*.csv
    python cv24f.py C            # f24/C_report.md, f24/C_*.csv  (needs c24f.py's output and A's)

A configuration is the tuple ``(grad, dens, mask, low, ms)`` (task C: ``low`` is the cut's ``t`` and ``grad`` carries
the method, e.g. ``sobel|cut-mc``). Scores of a window set: ``f`` (boundary F at 1 px for fields; for a property reference,
the harmonic mean of the recall and ``1 - cross_share``, as in 24e) or ``d_w`` (lower is better); the score of a group is the
pixel-weighted mean of its windows; the selection objective of several groups is the plain mean of their scores.
Folds as in 24e: leave one group out (chaco, humidchaco, lem, sorriso on its pivots, porto_murtinho on the CAR; D has the
four fields groups) and, within chaco, ``zeit.spatial_folds`` of the 45 window centroids (k-means clusters and hexagonal
blocks, k = 5).

Reference configurations kept out of the selection pool but scored on every fold: the notebook's (tophat, abs2000, low 0.3,
min_size 50, 1000 markers per window) and the 24e choice (sobel, none, low 0.1, min_size 0, 1000 markers per window), read from
24e's ``cv_long.csv`` (same R = 200, seed 0).
"""
import pyexpat  # noqa: F401
import sys

import numpy as np
import pandas as pd

import cases as C
import s24f as S

CFG = ["grad", "dens", "mask", "low", "ms"]
GROUPS = ("chaco", "humidchaco", "lem", "sorriso", "porto_murtinho")
NOTEBOOK = ("tophat", "n1000", "abs2000", 0.3, 50)
E24 = ("sobel", "n1000", "none", 0.1, 0)
REFS = (NOTEBOOK, E24)


def load_dir(sub: str) -> pd.DataFrame:
    d = S.OUT24 / sub
    parts = [pd.read_csv(p) for p in sorted(d.glob("*.csv"))]
    return pd.concat(parts, ignore_index=True)


def load_refs(full: bool = False) -> pd.DataFrame:
    """24e's rows (1000 markers per window, R = 200, seed 0, dens = ``n1000``): the two reference configurations, or with
    ``full`` its whole grid (tophat / sobel x 5 masks x 6 lows x 6 min_sizes), to repeat 24e's own selection."""
    t = pd.read_csv(C.OUT / "cv_long.csv")
    if full:
        s = t.copy()
    else:
        parts = [t[(t["grad"] == g) & (t["mask"] == m) & (t["low"] == lo) & (t["ms"] == ms)]
                 for g, m, lo, ms in ((NOTEBOOK[0], NOTEBOOK[2], NOTEBOOK[3], NOTEBOOK[4]),
                                      (E24[0], E24[2], E24[3], E24[4]))]
        s = pd.concat(parts)
    s = s.copy()
    s["dens"] = "n1000"
    return s.rename(columns={"d": "d_w"})


def units_table(t: pd.DataFrame) -> pd.DataFrame:
    """The A-style rows with ``unit`` = group (sorriso's CAR windows left out, as in 24e)."""
    lg = t[~((t["group"] == "sorriso") & (t["kind"] == "property"))].copy()
    lg["unit"] = lg["group"]
    return lg


def score_cube(t: pd.DataFrame, col: str) -> pd.DataFrame:
    """Rows: configurations; columns: units; value: the pixel-weighted mean of ``col`` over the unit's windows."""
    d = t[CFG + ["unit", "region_pixels", col]].dropna(subset=[col]).copy()
    d["w"] = d["region_pixels"].astype(float)
    d["wx"] = d["w"] * d[col]
    g = d.groupby(CFG + ["unit"], sort=False)[["w", "wx"]].sum()
    s = (g["wx"] / g["w"]).unstack("unit")
    return s


def wmean(scores: pd.DataFrame, units, weights=None) -> pd.Series:
    a = scores[list(units)].to_numpy(float)
    w = np.ones(len(units)) if weights is None else np.array([weights[u] for u in units], float)
    ok = ~np.isnan(a)
    den = (ok * w).sum(axis=1)
    num = (np.where(ok, a, 0) * w).sum(axis=1)
    # a configuration needs a score in every unit it is averaged over (the reduced multimean grid has gaps)
    full = ok.all(axis=1)
    return pd.Series(np.where(full, num / np.where(den > 0, den, 1), np.nan), index=scores.index).dropna()


def fmt(cfg) -> str:
    g, d, m, lo, ms = cfg
    dd = d if isinstance(d, str) else f"{int(d)}/Mpx"
    return f"{g}/{dd}/{m}/low {lo:g}/min_size {ms}"


def pool_of(scores: pd.DataFrame, restrict=None) -> pd.DataFrame:
    """The configurations a selection may pick: the density grid only (24e's rows, ``dens`` a string, are references),
    unless ``restrict`` names ``n1000`` (then 24e's whole grid, the references included, to repeat its selection)."""
    idx = scores.index
    own = bool(restrict) and "dens" in restrict and any(isinstance(d, str) for d in restrict["dens"])
    keep = np.array([own or (c not in REFS and not isinstance(c[1], str)) for c in idx])
    if restrict:
        for k, vals in restrict.items():
            pos = CFG.index(k)
            keep &= np.array([c[pos] in vals for c in idx])
    return scores[keep]


def pick_mean(scores, units, sign, weights=None, restrict=None):
    s = wmean(pool_of(scores, restrict), units, weights)
    return (s * sign).idxmax()


def pick_minimax(scores, units, sign, restrict=None):
    """Minimax regret over ``units``: the configuration whose worst shortfall from the unit's best is the smallest
    (ties by the mean shortfall)."""
    p = pool_of(scores, restrict)[list(units)]
    p = p.dropna()
    best = (p * sign).max(axis=0) / 1.0
    regret = best - (p * sign)
    key = pd.DataFrame({"max": regret.max(axis=1), "mean": regret.mean(axis=1)})
    return key.sort_values(["max", "mean"]).index[0]


def pick_rank(scores, units, sign, restrict=None):
    """The best mean rank over ``units`` (rank 1 = best score in the unit)."""
    p = pool_of(scores, restrict)[list(units)].dropna()
    ranks = (-(p * sign)).rank(axis=0, method="min")
    return ranks.mean(axis=1).sort_values().index[0]


def held(scores, cfg, us, weights):
    v = scores.loc[[cfg], list(us)].to_numpy(float)[0]
    w = np.array([weights[u] for u in us], float)
    ok = ~np.isnan(v)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan


def evaluate_folds(scores, folds, weights, sign, label, rule="mean", restrict=None, weight_train=False):
    allu = [u for us in folds.values() for u in us]
    rows = []
    for name, us in folds.items():
        train = [u for u in allu if u not in us]
        if rule == "mean":
            sel = pick_mean(scores, train, sign, weights if weight_train else None, restrict)
        elif rule == "minimax":
            sel = pick_minimax(scores, train, sign, restrict)
        else:
            sel = pick_rank(scores, train, sign, restrict)
        w = np.array([weights[u] for u in us], float)
        ins = wmean(pool_of(scores, restrict), us, weights)
        best = (ins * sign).idxmax()
        rows.append({"scheme": label, "fold": name, "n_units": len(us), "selected": fmt(sel), "selected_cfg": sel,
                     "held_out": held(scores, sel, us, weights), "notebook": held(scores, NOTEBOOK, us, weights),
                     "e24": held(scores, E24, us, weights), "in_sample": held(scores, best, us, weights),
                     "in_sample_cfg": fmt(best)})
    return pd.DataFrame(rows)


def chaco_folds(names):
    import zeit
    import cv as CV

    pts = CV.window_centroids(names)
    out = {}
    for meth in ("clusters", "blocks"):
        kw = dict(method=meth, k=5, seed=0, n_jobs=2)
        if meth == "blocks":
            xy = pts.to_crs(pts.estimate_utm_crs()).geometry
            kw.update(size=float(0.35 * max(xy.x.max() - xy.x.min(), xy.y.max() - xy.y.min())), iterations=200)
        f = np.asarray(zeit.spatial_folds(pts, **kw).folds)
        out[meth] = {f"{meth} {i}": [names[j] for j in np.where(f == i)[0]] for i in np.unique(f)}
    return out


def md(df: pd.DataFrame, fmt_f="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt_f.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                         else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def with_mean(g: pd.DataFrame) -> pd.DataFrame:
    row = {"fold": "mean", "n_units": "", "selected": "", **{c: g[c].mean() for c in ("held_out", "notebook", "e24", "in_sample")}}
    return pd.concat([g[["fold", "n_units", "selected", "held_out", "notebook", "e24", "in_sample"]],
                      pd.DataFrame([row])], ignore_index=True)


def analyse(t: pd.DataFrame, tag: str, extra_units=True, restricts=None):
    """All the folds on ``t`` (A- or C-style rows plus the reference rows); returns (results, scores)."""
    import warnings

    warnings.filterwarnings("ignore")
    lg = units_table(t)
    wts = lg.groupby("unit")["region_pixels"].sum().to_dict()
    results, cubes = [], {}
    for col, sign, lab, units in (("f", 1, "F", GROUPS), ("d_w", -1, "D", GROUPS[:4])):
        sc = score_cube(lg, col)
        cubes[lab] = sc
        folds = {u: [u] for u in units}
        for rule in ("mean", "minimax", "rank"):
            results.append(evaluate_folds(sc[list(units)], folds, wts, sign, f"LOGO {lab} [{rule}]", rule))
        for name, restrict in (restricts or {}).items():
            if wmean(pool_of(sc[list(units)], restrict), units).empty:
                continue
            results.append(evaluate_folds(sc[list(units)], folds, wts, sign, f"LOGO {lab} [mean; {name}]", "mean",
                                          restrict))
    ch = t[t["group"] == "chaco"].copy()
    names = sorted(ch["mosaic"].unique())
    ch["unit"] = ch["mosaic"]
    wts_w = ch.groupby("unit")["region_pixels"].first().to_dict()
    schemes = chaco_folds(names)
    for col, sign, lab in (("f", 1, "F"), ("d_w", -1, "D")):
        sc = score_cube(ch, col)
        cubes["chaco " + lab] = sc
        for meth, folds in schemes.items():
            # windows with no score at all (no reference inside the region) are left out of the units
            folds = {k: [u for u in us if u in sc.columns] for k, us in folds.items()}
            results.append(evaluate_folds(sc, folds, wts_w, sign, f"chaco {meth} {lab} [mean]", "mean",
                                          weight_train=True))
    return pd.concat(results, ignore_index=True), cubes


def propose(cubes: dict, sign_f=1):
    """The proposed single configuration on the five groups' F (minimax regret) and its group scores."""
    sc = cubes["F"][list(GROUPS)]
    out = {"minimax": pick_minimax(sc, GROUPS, 1), "rank": pick_rank(sc, GROUPS, 1), "mean": pick_mean(sc, GROUPS, 1)}
    return out


# ------------------------------------------------------------------ task A
ABLATIONS = {"24e grid (1000 markers per window)": {"dens": ["n1000"]}}
for _g in S.GRADS:
    ABLATIONS[f"gradient={_g}"] = {"grad": [_g]}
for _m in S.MASKS:
    ABLATIONS[f"mask={_m}"] = {"mask": [_m]}
for _d in S.DENS:
    ABLATIONS[f"density={_d}"] = {"dens": [_d]}


NN = "\n\n"


def cfg_str(cfg) -> str:
    g, d, m, lo, ms = cfg
    return f"{g},{d if isinstance(d, str) else int(d)},{m},{lo:g},{int(ms)}"


def per_group_table(cubes, cfgs: dict, lab: str) -> pd.DataFrame:
    sc = cubes[lab]
    units = GROUPS if lab == "F" else GROUPS[:4]
    rows = []
    for name, cfg in cfgs.items():
        r = {"config": name, "params": fmt(cfg)}
        for u in units:
            r[u] = float(sc.loc[[cfg], u].iloc[0]) if cfg in sc.index else np.nan
        r["mean"] = float(np.nanmean([r[u] for u in units]))
        rows.append(r)
    return pd.DataFrame(rows)


def main_A(ext=False):
    """``ext``: also the lows 0.02 and 0.05 (below the task's grid, the edge check); outputs get the prefix ``Aext``."""
    import json

    out = S.OUT24
    a = load_dir("A")
    if not ext:
        a = a[a["low"] >= 0.1 - 1e-9]
    t = pd.concat([a, load_refs(full=True)], ignore_index=True)
    tag = "Aext" if ext else "A"
    print(len(t), "rows;", t[["grad", "dens", "mask"]].drop_duplicates().shape[0], "stage-1 configurations")
    res, cubes = analyse(t, "A", restricts=ABLATIONS)
    res.drop(columns="selected_cfg").to_csv(out / f"{tag}_folds.csv", index=False)
    prop = propose(cubes)
    best = {}
    for lab in ("F", "D"):
        sc = cubes[lab]
        units = GROUPS if lab == "F" else GROUPS[:4]
        sign = 1 if lab == "F" else -1
        p = pool_of(sc)
        best[lab] = {u: (p[u].max() if sign == 1 else p[u].min()) for u in units}
    sob = pick_minimax(cubes["F"][list(GROUPS)], GROUPS, 1, {"grad": ["sobel"]})
    cfgs = {"notebook defaults": NOTEBOOK, "24e choice": E24, "proposed (minimax regret)": prop["minimax"],
            "single band (sobel), minimax regret": sob,
            "mean-rank rule": prop["rank"], "best mean of groups": prop["mean"]}
    tabs = {lab: per_group_table(cubes, cfgs, lab) for lab in ("F", "D")}
    for lab, tb in tabs.items():
        tb.loc[len(tb)] = {"config": "best of the group (in-sample)", "params": "", **best[lab],
                           "mean": float(np.mean(list(best[lab].values())))}
        tb.to_csv(out / f"{tag}_proposal_{lab}.csv", index=False)
    (out / ("proposed_ext.txt" if ext else "proposed.txt")).write_text(cfg_str(prop["minimax"]))
    sel = {"notebook": cfg_str(NOTEBOOK), "e24": cfg_str(E24), "proposed": cfg_str(prop["minimax"]),
           "sobel": cfg_str(sob), "mean_rule": cfg_str(prop["mean"]), "rank_rule": cfg_str(prop["rank"])}
    folds = res[(res["scheme"] == "LOGO F [mean]")]
    for _, r in folds.iterrows():
        sel[f"logo_{r['fold']}"] = cfg_str(r["selected_cfg"])
    (out / f"{tag}_selected.json").write_text(json.dumps(sel, indent=1))
    (out / f"{tag}_tables.md").write_text(render_A(res, tabs, prop, cubes), encoding="utf-8")
    print((out / f"{tag}_tables.md").read_text(encoding="utf-8")[:6000])


def render_A(res, tabs, prop, cubes) -> str:
    parts = []
    for scheme, g in res.groupby("scheme", sort=False):
        parts.append(f"**{scheme}**\n\n" + md(with_mean(g)))
    parts.append("**Proposal, F per group (selected on all five groups)**\n\n" + md(tabs["F"]))
    parts.append("**Proposal, D per group (lower is better)**\n\n" + md(tabs["D"]))
    return "\n\n".join(parts)


# ------------------------------------------------------------------ task C
def main_C(grads=("sobel", "multitemporal")):
    import json

    out = S.OUT24
    a = load_dir("A")
    a = a[a["grad"].isin(grads)]
    c = load_dir("C")
    t = pd.concat([a, c, load_refs()], ignore_index=True)
    methods = {"hysteresis": list(grads), "cut-mc": [f"{g}|cut-mc" for g in grads],
               "cut-exact": [f"{g}|cut-exact" for g in grads]}
    restricts = {f"method={k}": {"grad": v} for k, v in methods.items() if k in set(
        ["hysteresis"] + [m for m in ("cut-mc", "cut-exact") if any(x.endswith(m) for x in c["grad"].unique())])}
    res, cubes = analyse(t, "C", restricts=restricts)
    res.drop(columns="selected_cfg").to_csv(out / "C_folds.csv", index=False)
    cfgs = {"notebook defaults": NOTEBOOK, "24e choice": E24}
    sel = {"notebook": cfg_str(NOTEBOOK), "e24": cfg_str(E24)}
    for k, v in methods.items():
        if not any(g in set(t["grad"]) for g in v):
            continue
        for lab_rule, fn in (("minimax", pick_minimax),):
            cfg = fn(cubes["F"][list(GROUPS)], GROUPS, 1, {"grad": v})
            cfgs[f"best {k} ({lab_rule})"] = cfg
            sel[f"{k}"] = cfg_str(cfg)
    best = {}
    for lab in ("F", "D"):
        units = GROUPS if lab == "F" else GROUPS[:4]
        sign = 1 if lab == "F" else -1
        p = pool_of(cubes[lab])
        best[lab] = {u: (p[u].max() if sign == 1 else p[u].min()) for u in units}
    tabs = {lab: per_group_table(cubes, cfgs, lab) for lab in ("F", "D")}
    for lab, tb in tabs.items():
        tb.loc[len(tb)] = {"config": "best of the group (in-sample, any method)", "params": "", **best[lab],
                           "mean": float(np.mean(list(best[lab].values())))}
        tb.to_csv(out / f"C_proposal_{lab}.csv", index=False)
    (out / "C_selected.json").write_text(json.dumps(sel, indent=1))
    parts = []
    for scheme, g in res.groupby("scheme", sort=False):
        parts.append(f"**{scheme}**" + NN + md(with_mean(g)))
    parts.append("**Best configuration of each method on the five groups, F**" + NN + md(tabs["F"]))
    parts.append("**the same, D**" + NN + md(tabs["D"]))
    (out / "C_tables.md").write_text(NN.join(parts), encoding="utf-8")
    print(NN.join(parts)[:5000])


if __name__ == "__main__":
    {"A": main_A, "Aext": lambda: main_A(True), "C": main_C}[sys.argv[1]]()
