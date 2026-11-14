"""Tables (markdown) and plots of the sensitivity study from ``results.jsonl`` and the stage-1 cache.

    python summary.py tidy          # results.csv (one row per case x parameter set x seed)
    python summary.py maps          # convergence of the boundary map itself (needs the convergence cache)
    python summary.py tables        # markdown in sensitivity/report.md, PNGs in sensitivity/plots/
    python summary.py report24e     # report_24e.md (+ morris_effects.csv, plots/morris_mu_sigma.png): the short account
    python summary.py all
"""
import pyexpat  # noqa: F401
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import cases as C

OUT = C.OUT
BASE = dict(s1_gradient="tophat", s1_method="montecarlo", s1_markers=1000, s1_min_distance=20.0,
            s1_marker_radius=3, s1_coloring="knn", s1_mask_threshold=2000.0, s2_low=0.3, s2_high=0.6,
            s2_min_size=50, s2_final="image")
RESP_FIELDS = ["f_t0", "f_t1", "f_t2", "prec_t1", "rec_t1", "os_w", "us_w", "d_w", "d", "matched", "iou_fraction",
               "wasserstein", "ks", "n_pred", "n_ref"]
RESP_PROP = ["rec_t0", "rec_t1", "rec_t2", "cross_share", "cross_area_share", "n_pred"]
SUM_COLS = ("n_pred", "n_ref", "n_crossing", "n_unmatched", "n_labels")


def load() -> pd.DataFrame:
    rows = []
    with open(OUT / "results.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    df = pd.DataFrame(rows)
    df["gk"] = df["group"] + ":" + df["ref"]
    return df.drop_duplicates(["case", "s1_key", "s2_key"], keep="last")


def tidy():
    df = load()
    df.to_csv(OUT / "results.csv", index=False)
    print(len(df), "rows ->", OUT / "results.csv")


def aggregate(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Per gk, parameter set and seed: weighted means (weight = region pixels, NaN left out; counts summed)."""
    keys = ["gk", "s1_key", "s2_key", "seed"]
    out = []
    for k, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, k))
        row["n_cases"] = len(g)
        for c in cols:
            if c not in g:
                continue
            v = g[c].to_numpy(float)
            if c in SUM_COLS:
                row[c] = np.nansum(v)
                continue
            ok = ~np.isnan(v)
            row[c] = float(np.average(v[ok], weights=g["region_pixels"].to_numpy(float)[ok])) if ok.any() else np.nan
            row[c + "_ncase"] = int(ok.sum())
        row["s1_seconds"] = g["s1_seconds"].sum()
        row["s2_seconds"] = g["s2_seconds"].sum()
        for c in g.columns:
            if c.startswith("s1_") and c not in row and c not in ("s1_seconds",) or c.startswith("s2_") and c not in row:
                row[c] = g[c].iloc[0]
        out.append(row)
    return pd.DataFrame(out)


def is_base(df: pd.DataFrame, free=()):
    m = pd.Series(True, index=df.index)
    for k, v in BASE.items():
        if k in free:
            continue
        col = df[k]
        if isinstance(v, str):
            m &= col == v
        else:
            m &= (col.astype(float) - v).abs() < 1e-9
    return m


def md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (floatfmt.format(v) if abs(v) < 1e5 else f"{v:.0f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def resp_for(gk: str):
    return RESP_PROP if gk.endswith(":car") else RESP_FIELDS


# ------------------------------------------------------------------ the boundary map itself
def maps():
    """mean |p_R - p_ref| and the share of pixels whose hysteresis decision differs, with p_ref the mean of the
    other seeds' R = 2000 maps (leave one out), per group."""
    from zeit import _fields_core as _fc

    seeds, Rs = (0, 1, 2), (25, 50, 100, 200, 500, 1000, 2000)
    cs = {c.name: c for c in C.load_cases()}
    rows = []
    for name, c in cs.items():
        prob = {}
        for s in seeds:
            for R in Rs:
                p = C.OUT / "cache" / name / f"tophat_mc_R{R}_m1000_d20_r3_knn_t2000_s{s}.npz"
                if not p.exists():
                    break
                z = np.load(p)
                prob[(R, s)] = z["counts"].astype(np.float64) / int(z["r"])
        if len(prob) < len(seeds) * len(Rs):
            continue
        keep = c.keep(2000)
        region = c.regions[1] if keep is None else c.regions[1]
        for s in seeds:
            ref = np.mean([prob[(2000, o)] for o in seeds if o != s], axis=0)
            href = _fc.hysteresis_threshold(ref, 0.3, 0.6)
            for R in Rs:
                p = prob[(R, s)]
                h = _fc.hysteresis_threshold(p, 0.3, 0.6)
                diff = h != href
                rows.append({
                    "mosaic": name, "group": c.group, "R": R, "seed": s, "pixels": p.size,
                    "region_pixels": int(region.sum()),
                    "mae": float(np.abs(p - ref).mean()), "mae_region": float(np.abs(p - ref)[region].mean()),
                    "mae_active": float(np.abs(p - ref)[(p > 0) | (ref > 0)].mean()),
                    "hyst_diff_share": float(diff.mean()), "hyst_diff_region": float(diff[region].mean()),
                    "hyst_jaccard_dist": float(diff.sum() / max((h | href).sum(), 1)),
                    "hyst_pixels_ref": int(href.sum()),
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "convergence_maps.csv", index=False)
    print(len(df), "rows -> convergence_maps.csv")


def maps_table(df: pd.DataFrame) -> str:
    out = []
    for g, d in df.groupby("group"):
        t = d.groupby("R").apply(lambda x: pd.Series({
            "mae": np.average(x["mae"], weights=x["pixels"]),
            "mae_region": np.average(x["mae_region"], weights=x["region_pixels"].clip(lower=1)),
            "hyst_diff_share": np.average(x["hyst_diff_share"], weights=x["pixels"]),
            "hyst_diff_region": np.average(x["hyst_diff_region"], weights=x["region_pixels"].clip(lower=1)),
            "jaccard_dist": np.average(x["hyst_jaccard_dist"], weights=x["hyst_pixels_ref"].clip(lower=1)),
        })).reset_index()
        out.append(f"**{g}**\n\n" + md(t, "{:.4f}"))
    return "\n\n".join(out)


# ------------------------------------------------------------------ tables
def seed_stats(agg: pd.DataFrame, cols, by):
    g = agg.groupby(by, dropna=False)
    mean = g[cols].mean()
    sd = g[cols].std(ddof=1)
    return mean, sd


def convergence_tables(df: pd.DataFrame) -> str:
    sub = df[is_base(df) & (df["s1_method"] == "montecarlo")]
    sub = sub[sub["s1_realizations"].isin([25, 50, 100, 200, 500, 1000, 2000])]
    out = []
    for gk, d in sub.groupby("gk"):
        cols = [c for c in resp_for(gk) if c in d]
        agg = aggregate(d, cols)
        agg["R"] = agg["s1_realizations"]
        mean, sd = seed_stats(agg, cols, "R")
        t = mean.copy()
        for c in cols:
            if c in ("n_pred", "n_ref"):
                continue
            t[c + "_sd"] = sd[c]
        keep = [c for c in cols if not c.endswith("_t0") and not c.endswith("_t2")][:7]
        cols_show = []
        for c in keep:
            cols_show += [c] + ([c + "_sd"] if c + "_sd" in t else [])
        out.append(f"**{gk}** ({int(agg['n_cases'].max())} cases, seeds {sorted(agg['seed'].unique())})\n\n"
                   + md(t[cols_show].reset_index(), "{:.4f}"))
    return "\n\n".join(out)


def oat_tables(df: pd.DataFrame, R: int):
    """One row per factor level and group: the responses (mean over seeds) and the effect sizes."""
    sub = df[(df["s1_method"] == "montecarlo") & (df["s1_realizations"] == R)].copy()
    factors = {
        "s1_markers": "markers", "s1_min_distance": "min_distance", "s1_marker_radius": "marker_radius",
        "s1_coloring": "coloring", "s1_gradient": "gradient", "s1_mask_threshold": "mask_threshold",
        "s2_low": "low", "s2_high": "high", "s2_min_size": "min_size", "s2_final": "final"}
    tables, effects = {}, []
    for gk, d in sub.groupby("gk"):
        cols = [c for c in resp_for(gk) if c in d]
        rows = []
        for col, name in factors.items():
            m = is_base(d, free=(col,))
            for lvl, g in d[m].groupby(d.loc[m, col].astype(str)):
                agg = aggregate(g, cols)
                mean = agg[cols].mean()
                sd = agg[cols].std(ddof=1) if len(agg) > 1 else pd.Series(np.nan, index=cols)
                r = {"factor": name, "level": lvl, "seeds": len(agg)}
                r.update(mean.to_dict())
                r.update({c + "_sd": sd[c] for c in cols})
                rows.append(r)
        t = pd.DataFrame(rows)
        tables[gk] = t
        for name, g in t.groupby("factor"):
            e = {"gk": gk, "factor": name}
            for c in cols:
                if c in ("n_pred", "n_ref"):
                    continue
                e[c + "_range"] = float(g[c].max() - g[c].min())
                e[c + "_best"] = g.loc[g[c].idxmax(), "level"] if c.startswith(("f_", "matched", "rec_")) else (
                    g.loc[g[c].idxmin(), "level"])
            e["seed_sd_f"] = float(np.nanmean(g["f_t1_sd"])) if "f_t1_sd" in g else np.nan
            e["seed_sd_rec"] = float(np.nanmean(g["rec_t1_sd"])) if "rec_t1_sd" in g else np.nan
            effects.append(e)
    return tables, pd.DataFrame(effects)


def exact_density_tables(df: pd.DataFrame) -> str:
    """The exact probability with N = density x pixels: per group and density, the best (low, high) by F or D."""
    if "tag" not in df:
        return "(no density rows)"
    sub = df[df["tag"].notna()].copy()
    if sub.empty:
        return "(no density rows)"
    sub["s1_key"] = sub["tag"]
    out = []
    for gk, d in sub.groupby("gk"):
        cols = [c for c in resp_for(gk) if c in d]
        agg = aggregate(d, cols)
        agg["density"] = agg["s1_key"].str.extract(r"density (\d+)")[0].astype(float)
        agg["low"], agg["high"] = agg["s2_low"], agg["s2_high"]
        show = ["density", "low", "high"] + [c for c in ("f_t1", "os_w", "us_w", "d_w", "matched", "wasserstein",
                                                          "n_pred", "rec_t1", "cross_share") if c in agg]
        key, asc = ("f_t1", False) if "f_t1" in agg and agg["f_t1"].notna().any() else ("rec_t1", False)
        best = agg.sort_values(key, ascending=asc).groupby("density").head(1).sort_values("density")
        out.append(f"**{gk}** best (low, high) by {key} at each density\n\n" + md(best[show], "{:.3f}"))
        if key == "f_t1":
            bd = agg.sort_values("d_w").groupby("density").head(1).sort_values("density")
            out.append(f"**{gk}** best (low, high) by d_w at each density\n\n" + md(bd[show], "{:.3f}"))
    return "\n\n".join(out)


def exact_tables(df: pd.DataFrame) -> str:
    sub = df[df["s1_method"] == "exact"]
    if "tag" in sub:
        sub = sub[sub["tag"].isna()]
    if sub.empty:
        return "(no exact rows)"
    out = []
    for gk, d in sub.groupby("gk"):
        cols = [c for c in resp_for(gk) if c in d]
        agg = aggregate(d, cols)
        agg["N"] = agg["s1_markers"]
        agg["low"], agg["high"] = agg["s2_low"], agg["s2_high"]
        show = ["N", "low", "high"] + [c for c in ("f_t1", "os_w", "us_w", "d_w", "matched", "wasserstein", "n_pred",
                                                    "rec_t1", "cross_share") if c in agg]
        if "f_t1" in agg and agg["f_t1"].notna().any():
            by_f = agg.groupby(["N", "low", "high"], as_index=False)[[c for c in show if c not in ("N", "low", "high")]].mean()
            out.append(f"**{gk}** top 6 by f_t1\n\n" + md(by_f.sort_values("f_t1", ascending=False).head(6)[show], "{:.3f}"))
            out.append(f"**{gk}** top 6 by d_w (lowest)\n\n" + md(by_f.sort_values("d_w").head(6)[show], "{:.3f}"))
            out.append(f"**{gk}** top 6 by matched\n\n" + md(by_f.sort_values("matched", ascending=False).head(6)[show], "{:.3f}"))
        else:
            by_f = agg.groupby(["N", "low", "high"], as_index=False)[[c for c in show if c not in ("N", "low", "high")]].mean()
            out.append(f"**{gk}** top 6 by rec_t1\n\n" + md(by_f.sort_values("rec_t1", ascending=False).head(6)[show], "{:.3f}"))
    return "\n\n".join(out)


def plots(df: pd.DataFrame, R: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = OUT / "plots"
    pdir.mkdir(exist_ok=True)
    # convergence
    sub = df[is_base(df) & (df["s1_method"] == "montecarlo")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, resp in zip(axes, ("f_t1", "d_w", "matched")):
        for gk, d in sub.groupby("gk"):
            if resp not in d or d[resp].notna().sum() == 0:
                continue
            agg = aggregate(d, [resp])
            agg["R"] = agg["s1_realizations"]
            mean, sd = seed_stats(agg, [resp], "R")
            ax.errorbar(mean.index, mean[resp], sd[resp], marker="o", ms=3, capsize=2, label=gk)
        ax.set_xscale("log")
        ax.set_xlabel("realizations")
        ax.set_ylabel(resp)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(pdir / "convergence.png", dpi=110)
    plt.close(fig)
    # OAT, F by factor
    tables, _ = oat_tables(df, R)
    names = ["markers", "min_distance", "marker_radius", "coloring", "gradient", "mask_threshold", "low", "high",
             "min_size", "final"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6))
    for ax, name in zip(axes.ravel(), names):
        for gk, t in tables.items():
            col = "rec_t1" if gk.endswith(":car") else "f_t1"
            g = t[t.factor == name]
            if g.empty:
                continue
            ax.plot(range(len(g)), g[col], marker="o", ms=3, label=gk)
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels(g["level"], fontsize=6, rotation=30)
        ax.set_title(name, fontsize=8)
    axes[0, 0].legend(fontsize=5)
    fig.suptitle("boundary F at 1 px (recall for CAR groups) against each factor, others at the notebook's values")
    fig.tight_layout()
    fig.savefig(pdir / "oat_f.png", dpi=110)
    plt.close(fig)


def report(R: int):
    df = load()
    parts = ["# Sensitivity of field boundaries (24e)\n"]
    parts.append(f"{len(df)} result rows, {df['case'].nunique()} cases.\n")
    if (OUT / "convergence_maps.csv").exists():
        parts.append("## Convergence of the boundary map (leave-one-seed-out reference = mean of the other two R=2000 maps)\n")
        parts.append(maps_table(pd.read_csv(OUT / "convergence_maps.csv")))
    parts.append("\n## Convergence of the responses (notebook parameters, mean over seeds, sd between seeds)\n")
    parts.append(convergence_tables(df))
    parts.append("\n## Exact probability (low/high grid, top rows)\n")
    parts.append(exact_tables(df))
    parts.append("\n## Exact probability with N proportional to the window's pixels (markers per Mpx)\n")
    parts.append(exact_density_tables(df))
    tables, effects = oat_tables(df, R)
    parts.append(f"\n## One factor at a time (R = {R})\n")
    for gk, t in tables.items():
        cols = [c for c in t.columns if not c.endswith("_sd") and c not in ("seeds",)]
        keep = [c for c in ("factor", "level", "f_t1", "os_w", "us_w", "d_w", "matched", "wasserstein", "n_pred",
                            "rec_t1", "cross_share") if c in cols]
        parts.append(f"**{gk}**\n\n" + md(t[keep]))
    parts.append("\n## Effect of each factor (range of the response over its levels)\n")
    for gk, e in effects.groupby("gk"):
        cols = ["factor"] + [c for c in e.columns if c.endswith("_range") and c.split("_range")[0] in
                             ("f_t1", "d_w", "matched", "wasserstein", "os_w", "us_w", "rec_t1", "cross_share")]
        cols += ["seed_sd_f"] if "f_t1_range" in e.columns else ["seed_sd_rec"]
        parts.append(f"**{gk}**\n\n" + md(e[cols].dropna(axis=1, how="all"), "{:.3f}"))
    (OUT / "report.md").write_text("\n\n".join(parts), encoding="utf-8")
    plots(df, R)
    print("report.md and plots written")


# ------------------------------------------------------------------ Morris (task B)
MORRIS_FIELDS = ["f_t1", "d_w", "matched", "matched_in", "wasserstein"]
MORRIS_PROP = ["rec_t1", "cross_share"]


def morris_effects(df: pd.DataFrame, r: int, nboot: int = 2000) -> pd.DataFrame:
    """mu, mu*, sigma of the elementary effects and a bootstrap CI of mu* (over trajectories), per gk and response."""
    import morris as MO

    meta = MO.case_meta()
    pts, moved, signed = MO.design(r)
    idx = df.drop_duplicates(["case", "s1_key", "s2_key"]).set_index(["case", "s1_key", "s2_key"])
    cases = df.drop_duplicates("case")[["case", "mosaic", "gk"]]
    rng = np.random.default_rng(0)
    rows = []
    k = len(MO.FACTORS)
    for gk, cs in cases.groupby("gk"):
        resp = MORRIS_PROP if gk.endswith(":car") else MORRIS_FIELDS
        Y = np.full((r, k + 1, len(resp)), np.nan)
        for t in range(r):
            for s in range(k + 1):
                vals, w, ok = [], [], True
                for case, name in zip(cs["case"], cs["mosaic"]):
                    s1, s2 = MO.decode(pts[t, s], meta[name], t)
                    key = (case, s1.key(), s2.key())
                    if key not in idx.index:
                        ok = False
                        break
                    row = idx.loc[key]
                    vals.append([row.get(c, np.nan) for c in resp])
                    w.append(row["region_pixels"])
                if not ok:
                    continue
                v, w = np.array(vals, float), np.array(w, float)
                for j in range(len(resp)):
                    good = ~np.isnan(v[:, j])
                    if good.any():
                        Y[t, s, j] = np.average(v[good, j], weights=w[good])
        for j, c in enumerate(resp):
            ee = np.full((r, k), np.nan)
            for t in range(r):
                for s in range(k):
                    ee[t, moved[t, s]] = (Y[t, s + 1, j] - Y[t, s, j]) / signed[t, s]
            for f, name in enumerate(MO.FACTORS):
                e = ee[:, f]
                e = e[~np.isnan(e)]
                if len(e) < 3:
                    continue
                boot = [np.abs(rng.choice(e, len(e))).mean() for _ in range(nboot)]
                rows.append({"gk": gk, "response": c, "factor": name, "n_traj": len(e), "mu": e.mean(),
                             "mu_star": np.abs(e).mean(), "sigma": e.std(ddof=1),
                             "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "morris_effects.csv", index=False)
    return out


def morris_tables(eff: pd.DataFrame) -> str:
    parts = []
    for gk, d in eff.groupby("gk"):
        prim = "rec_t1" if gk.endswith(":car") else "f_t1"
        t = d[d.response == prim].sort_values("mu_star", ascending=False)
        t = t.assign(rank=range(1, len(t) + 1))
        t["mu* [95% CI]"] = [f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(t.mu_star, t.ci_lo, t.ci_hi)]
        show = t[["rank", "factor", "mu* [95% CI]", "mu", "sigma"]]
        parts.append(f"**{gk}**, response {prim} ({int(t.n_traj.max())} trajectories)\n\n" + md(show, "{:.3f}"))
        if not gk.endswith(":car"):
            w = d.pivot(index="factor", columns="response", values="mu_star")
            w = w.loc[t.factor, [c for c in MORRIS_FIELDS if c in w]]
            parts.append("mu* of the other responses\n\n" + md(w.reset_index(), "{:.3f}"))
    return "\n\n".join(parts)


def morris_plot(eff: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gks = list(dict.fromkeys(eff["gk"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, gk in zip(axes.ravel(), gks):
        prim = "rec_t1" if gk.endswith(":car") else "f_t1"
        d = eff[(eff.gk == gk) & (eff.response == prim)]
        ax.errorbar(d.mu_star, d.sigma, xerr=[d.mu_star - d.ci_lo, d.ci_hi - d.mu_star], fmt="o", ms=4, capsize=2)
        for _, r in d.iterrows():
            ax.annotate(r.factor, (r.mu_star, r.sigma), fontsize=7, xytext=(3, 3), textcoords="offset points")
        m = max(d.mu_star.max(), d.sigma.max()) * 1.05
        ax.plot([0, m], [0, m], "k:", lw=0.6)
        ax.set_title(f"{gk}: {prim}", fontsize=9)
        ax.set_xlabel("mu*")
        ax.set_ylabel("sigma")
    for ax in axes.ravel()[len(gks):]:
        ax.axis("off")
    fig.suptitle("Morris: mu* (bars: 95% bootstrap CI over trajectories) against sigma; dotted: sigma = mu*")
    fig.tight_layout()
    (OUT / "plots").mkdir(exist_ok=True)
    fig.savefig(OUT / "plots" / "morris_mu_sigma.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ report_24e.md
def conv_compact(df: pd.DataFrame) -> str:
    sub = df[is_base(df) & (df["s1_method"] == "montecarlo")]
    if "study" in sub:
        sub = sub[sub["study"].isna()]
    sub = sub[sub["s1_realizations"].isin([25, 50, 100, 200, 500, 1000, 2000])]
    rows = []
    for gk, d in sub.groupby("gk"):
        cols = ["rec_t1", "cross_share"] if gk.endswith(":car") else ["f_t1", "d_w", "matched"]
        agg = aggregate(d, cols)
        agg["R"] = agg["s1_realizations"]
        mean, sd = seed_stats(agg, cols, "R")
        for R in mean.index:
            r = {"gk": gk, "R": int(R)}
            for c in cols:
                r[c] = mean.loc[R, c]
                r[c + " sd(seeds)"] = sd.loc[R, c]
            rows.append(r)
    t = pd.DataFrame(rows)
    f = t[~t.gk.str.endswith(":car")][["gk", "R", "f_t1", "f_t1 sd(seeds)", "d_w", "d_w sd(seeds)", "matched",
                                       "matched sd(seeds)"]]
    c = t[t.gk.str.endswith(":car")][["gk", "R", "rec_t1", "rec_t1 sd(seeds)", "cross_share", "cross_share sd(seeds)"]]
    return md(f, "{:.4f}") + "\n\n" + md(c, "{:.4f}")


def oat_compact(df: pd.DataFrame, R: int) -> str:
    _, eff = oat_tables(df, R)
    rows = {}
    for _, e in eff.iterrows():
        col = "rec_t1_range" if e.gk.endswith(":car") else "f_t1_range"
        rows.setdefault(e.factor, {})[e.gk] = e[col]
    t = pd.DataFrame(rows).T
    sd = {gk: float(g["seed_sd_rec" if gk.endswith(":car") else "seed_sd_f"].mean()) for gk, g in eff.groupby("gk")}
    order = list(t.assign(m=t.max(axis=1)).sort_values("m", ascending=False).index)
    t = t.loc[order]
    t.loc["(sd between seeds)"] = pd.Series(sd)
    return ("Range of F at 1 px (recall for the CAR groups) over each factor's levels, the others at the notebook's "
            "values, R = %d, mean over 2 seeds; the last row is the sd of F between seeds.\n\n" % R
            + md(t.reset_index().rename(columns={"index": "factor"}), "{:.3f}"))


def matched_table() -> str:
    p = OUT / "matched_diagnosis.csv"
    if not p.exists():
        return "(run diagnose.py)"
    d = pd.read_csv(p)
    show = ["case", "n_ref_in_region", "cut_by_region", "fully_inside", "ref_px_median", "pred_px_median_full",
            "share_split", "share_merge", "ref_px_pred_zero", "matched", "matched_clip", "matched_in", "matched_fillfg",
            "f_t1"]
    return md(d[show], "{:.3f}")


def report_24e(R: int = 200, r: int = 20):
    df = load()

    def text(name):
        p = OUT / name
        return p.read_text(encoding="utf-8") if p.exists() else f"({name} missing)"

    parts = ["# Field boundaries by the stochastic watershed: sensitivity, 24e\n"]
    parts.append("## 1. Convergence (notebook parameters; mean and sd between 3 seeds)\n")
    parts.append(conv_compact(df))
    parts.append("\n## 2. One factor at a time\n")
    parts.append(oat_compact(df, R))
    parts.append("\n## 3. Why `matched` is low\n")
    parts.append(text("matched_conclusion.md"))
    parts.append(matched_table())
    parts.append("\n## 4. Morris elementary effects\n")
    parts.append(f"{r} trajectories x 9 points x 64 windows, R = 200, 6-level grid (delta 0.6), the Monte Carlo seed "
                 "tied to the trajectory; mu* in units of the response per unit of the factor's range (log scale for "
                 "markers per Mpx 30-3000, min_size 0-200). `mask_q` only acts when the mask kind is relative (a dummy "
                 "otherwise); `gradient` and `mask kind` are discrete (6 levels, two on each). `density` is mostly the "
                 "low end: on a 24 kpx window 30-300/Mpx is 1-7 markers. Groups are pixel-weighted means of their "
                 "windows; CAR groups are scored by recall.\n")
    if "study" in df and (df["study"] == "morris").any():
        eff = morris_effects(df, r)
        parts.append(morris_tables(eff))
        morris_plot(eff)
    parts.append("\n## 5. Spatial cross-validation of the thresholds\n")
    parts.append(text("cv_report.md"))
    parts.append("\n## 6. Recommended defaults and what 24f should attack\n")
    parts.append(text("recommend.md"))
    (OUT / "report_24e.md").write_text("\n\n".join(parts), encoding="utf-8")
    print("report_24e.md written")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if cmd in ("tidy", "all"):
        tidy()
    if cmd in ("maps", "all"):
        maps()
    if cmd in ("tables", "all"):
        report(R)
    if cmd in ("report24e", "all"):
        report_24e(R)
