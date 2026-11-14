"""24f: the tables of task B and of the extras (package ``clip`` / ``complete_only``), and the assembly of report_24f.md.

    python report24f.py B           # f24/B_tables.md
    python report24f.py extras A    # f24/A_extras_tables.md  (also for C)
    python report24f.py report      # report_24f.md from the pieces
"""
import pyexpat  # noqa: F401
import json
import sys

import numpy as np
import pandas as pd

import cases as C
import s24f as S
from cv24f import GROUPS, md

OUT = S.OUT24


def wavg(g, col, w):
    ok = g[col].notna() & (g[w] > 0)
    return float(np.average(g.loc[ok, col], weights=g.loc[ok, w])) if ok.any() else np.nan


def group_scores(df: pd.DataFrame) -> dict:
    """F per group (pixel-weighted; sorriso's CAR left out), D per fields group, and the package's matched pooled by
    reference fields (unclipped, and clipped + complete only)."""
    df = df[~((df["group"] == "sorriso") & (df["kind"] == "property"))]
    out = {}
    for g in GROUPS:
        x = df[df["group"] == g]
        out[("F", g)] = wavg(x, "f", "region_pixels")
        if g != "porto_murtinho":
            out[("D", g)] = wavg(x, "d_w", "region_pixels")
            if "cc_matched" in x and x["cc_matched"].notna().any():
                y = x[x["kind"] == "fields"]
                out[("cc_matched", g)] = wavg(y, "cc_matched", "cc_n_ref")
                out[("matched", g)] = wavg(y, "matched_pkg", "n_ref_pkg")
    return out


def b_tables() -> str:
    parts = []
    from b24f import cfg_of

    for name in ("notebook", "proposed"):
        files = sorted((OUT / "B").glob(f"{name}_s*.csv"), key=lambda p: float(p.stem.split("_s")[1]))
        if not files:
            continue
        cfg = cfg_of(name)
        rows, rows_best = [], []
        for p in files:
            s = float(p.stem.split("_s")[1])
            df = pd.read_csv(p)
            own = df[np.isclose(df["low"], cfg[3])]
            sc = group_scores(own)
            r = {"s": s}
            for k in ("F", "D", "cc_matched"):
                vals = [sc.get((k, g), np.nan) for g in GROUPS]
                for g, v in zip(GROUPS, vals):
                    if not np.isnan(v):
                        r[f"{k} {g}"] = v
                r[f"{k} mean"] = float(np.nanmean(vals))
            best = max(((low, np.nanmean([group_scores(d)[("F", g)] for g in GROUPS]))
                        for low, d in df.groupby("low")), key=lambda x: x[1])
            r["best low"], r["F mean at best low"] = best
            rows.append(r)
        t = pd.DataFrame(rows)
        parts.append(f"**{name}** ({cfg[0]}, {cfg[1]}, {cfg[2]}, low {cfg[3]}, min_size {cfg[4]})\n\n" + md(t))
    return "\n\n".join(parts)


def extras_tables(part: str) -> str:
    x = pd.read_csv(OUT / f"{part}_extras.csv")
    sel = json.loads((OUT / f"{part}_selected.json").read_text())
    names = {v: k for k, v in sel.items()}
    rows = []
    for cfg, g in x.groupby("cfg", sort=False):
        sc = group_scores(g.assign(f=np.nan, d_w=g.get("cc_d_w"), region_pixels=g["region_pixels"]))
        r = {"config": names.get(cfg, cfg), "spec": cfg}
        for k in ("matched", "cc_matched"):
            for grp in GROUPS[:4]:
                r[f"{k} {grp}"] = sc.get((k, grp), np.nan)
        rows.append(r)
    t = pd.DataFrame(rows)
    # agreement of the package's clip + complete_only with matched.py's matched_in
    f = x[x["kind"] == "fields"].dropna(subset=["cc_matched", "mp_matched_in"])
    diff = (f["cc_matched"] - f["mp_matched_in"]).abs()
    note = (f"package (clip + complete_only) against matched.py's matched_in, {len(f)} case x configuration pairs: "
            f"max |difference| {diff.max():.2e}, {int((diff > 1e-9).sum())} differ; n_ref of the package equals n_in in "
            f"{int((f['cc_n_ref'] == f['mp_n_in']).sum())} of {len(f)}.")
    return md(t) + "\n\n" + note


if __name__ == "__main__":
    if sys.argv[1] == "B":
        (OUT / "B_tables.md").write_text(b_tables(), encoding="utf-8")
        print(b_tables())
    elif sys.argv[1] == "extras":
        s = extras_tables(sys.argv[2])
        (OUT / f"{sys.argv[2]}_extras_tables.md").write_text(s, encoding="utf-8")
        print(s)


# ------------------------------------------------------------------ the report
def _read(name):
    p = OUT / name
    return p.read_text(encoding="utf-8").strip() if p.exists() else f"(missing: {name})"


def _compact(folds: pd.DataFrame, schemes: dict, mean_only=False) -> str:
    """One row per fold; the columns: notebook, 24e fixed, then the held-out of each scheme named in ``schemes``."""
    base = folds[folds["scheme"] == schemes["new, mean rule"]]
    t = pd.DataFrame({"fold": base["fold"].values, "notebook": base["notebook"].values, "24e choice": base["e24"].values})
    for label, sch in schemes.items():
        g = folds[folds["scheme"] == sch]
        t[label] = g["held_out"].values
    t["in-sample optimum"] = base["in_sample"].values
    t.loc[len(t)] = {"fold": "mean", **{c: t[c].mean() for c in t.columns[1:]}}
    return md(t)


def _selected(folds: pd.DataFrame, scheme: str) -> str:
    g = folds[folds["scheme"] == scheme]
    return md(g[["fold", "selected", "held_out", "in_sample_cfg"]].rename(
        columns={"selected": "chosen on the others", "in_sample_cfg": "in-sample optimum"}))


def _ablation(folds: pd.DataFrame) -> str:
    rows = []
    for lab in ("F", "D"):
        for sch, g in folds.groupby("scheme", sort=False):
            if sch.startswith(f"LOGO {lab} [mean"):
                name = sch[len(f"LOGO {lab} [mean"):].strip(" ;]") or "unrestricted"
                rows.append({"score": lab, "pool": name, "held-out mean": g["held_out"].mean(),
                             "in-sample mean": g["in_sample"].mean()})
    return md(pd.DataFrame(rows))


def report():
    fa = pd.read_csv(OUT / "A_folds.csv")
    fe = pd.read_csv(OUT / "Aext_folds.csv")
    fc = pd.read_csv(OUT / "C_folds.csv")
    S_ = {"24e grid (its own selection)": "LOGO {} [mean; 24e grid (1000 markers per window)]",
          "new, mean rule": "LOGO {} [mean]", "new, minimax rule": "LOGO {} [minimax]", "new, rank rule": "LOGO {} [rank]"}
    parts = ["# Field boundaries: the candidate defaults, phase 24f (task 2)\n",
             _read("notes_design.md"), "\n## A. Extended spatial cross-validation\n", _read("notes_A.md")]
    for lab, name in (("F", "boundary F at 1 px (CAR groups: recall x non-crossing)"), ("D", "area-weighted Clinton D (lower is better)")):
        parts.append(f"\n### A.{1 if lab == 'F' else 2} Leave one group out, {name}\n")
        parts.append(_compact(fa, {k: v.format(lab) for k, v in S_.items()}))
        parts.append(f"\nParameters chosen on the other groups (mean rule), {lab}:\n")
        parts.append(_selected(fa, f"LOGO {lab} [mean]"))
        parts.append(f"\nParameters chosen on the other groups (minimax rule), {lab}:\n")
        parts.append(_selected(fa, f"LOGO {lab} [minimax]"))
    parts.append("\n### A.3 Within Chaco (45 windows, k = 5)\n")
    for lab in ("F", "D"):
        for meth in ("clusters", "blocks"):
            g = fa[fa["scheme"] == f"chaco {meth} {lab} [mean]"]
            t = g[["fold", "n_units", "notebook", "e24", "held_out", "in_sample", "selected"]].rename(
                columns={"e24": "24e choice", "held_out": "new, chosen on the others", "in_sample": "in-sample optimum"})
            t.loc[len(t)] = {"fold": "mean", "n_units": "", "notebook": g["notebook"].mean(), "24e choice": g["e24"].mean(),
                             "new, chosen on the others": g["held_out"].mean(), "in-sample optimum": g["in_sample"].mean(),
                             "selected": ""}
            parts.append(f"\n**{meth}, {lab}**\n\n" + md(t))
    parts.append("\n### A.4 What each factor is worth (leave one group out, pool restricted)\n")
    parts.append(_ablation(fa))
    parts.append("\n### A.5 The proposal\n")
    parts.append(_read("notes_proposal.md"))
    parts.append("\nF per group:\n\n" + md(pd.read_csv(OUT / "A_proposal_F.csv")))
    parts.append("\nD per group (lower is better):\n\n" + md(pd.read_csv(OUT / "A_proposal_D.csv")))
    parts.append("\nmatched (one to one, IoU >= 0.5) by the package: `matched` as the package counts it by default, `cc_matched` "
                 "with clip=True, complete_only=True (pooled over the group's reference fields); LOGO rows are the configuration "
                 "chosen without that group:\n\n" + _read("A_extras_report.md"))
    parts.append("\n### A.6 The edge of `low`\n")
    parts.append(_read("notes_ext.md"))
    parts.append("\n" + _compact(fe, {k: v.format("F") for k, v in S_.items() if "grid" not in k}))
    parts.append("\n## B. Monte Carlo with noise\n")
    parts.append(_read("notes_B.md"))
    parts.append("\n" + _read("B_tables.md"))
    parts.append("\n## C. Hierarchy cut instead of hysteresis\n")
    parts.append(_read("notes_C.md"))
    for lab in ("F", "D"):
        rows = []
        for sch, g in fc.groupby("scheme", sort=False):
            if sch.startswith(f"LOGO {lab} [mean"):
                rows.append({"pool": sch[len(f"LOGO {lab} ["):].rstrip("]"), "held-out mean": g["held_out"].mean(),
                             "in-sample mean": g["in_sample"].mean()})
        parts.append(f"\n**Leave one group out, {lab}: mean over the groups, by method**\n\n" + md(pd.DataFrame(rows)))
    parts.append("\n**Best configuration of each method (minimax on F), per group**\n\n" + _read("C_best_F.md"))
    parts.append("\n" + _read("C_best_D.md"))
    parts.append("\n" + _read("C_extras_report.md"))
    parts.append("\n## Recommended defaults\n")
    parts.append(_read("notes_recommend.md"))
    (OUT.parent / "report_24f.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("report_24f.md written")


if __name__ == "__main__" and sys.argv[1] == "report":
    report()


def pieces():
    """The small markdown pieces the report includes: the extras tables and the best-of-method tables of task C."""
    for part in ("A", "C"):
        p = OUT / f"{part}_extras.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p)
        sel = json.loads((OUT / f"{part}_selected.json").read_text())
        rows = []

        def entry(label, spec, groups_from=None):
            r = {"config": label, "spec": spec}
            for k in ("matched", "cc_matched"):
                for g in GROUPS[:4]:
                    cfg = spec if groups_from is None else sel[f"logo_{g}"]
                    sc = group_scores(x[x["cfg"] == cfg].assign(f=np.nan, d_w=np.nan))
                    r[f"{k} {g}"] = sc.get((k, g), np.nan)
            rows.append(r)

        for key, label in (("notebook", "notebook defaults"), ("e24", "24e choice"), ("proposed", "proposed"),
                           ("sobel", "single band (sobel)"), ("hysteresis", "best hysteresis"), ("cut-mc", "best cut-mc"),
                           ("cut-exact", "best cut-exact")):
            if key in sel:
                entry(label, sel[key])
        if part == "A":
            entry("held-out choice (LOGO, mean rule)", "", groups_from=True)
            for key in sel:
                if key.startswith("scan_"):
                    entry(key.replace("scan_", "scan: "), sel[key])
        t = pd.DataFrame(rows)
        f = x[x["kind"] == "fields"].dropna(subset=["cc_matched", "mp_matched_in"])
        diff = (f["cc_matched"] - f["mp_matched_in"]).abs()
        note = (f"\n\nThe package's clip=True + complete_only=True against matched.py's `matched_in` on {len(f)} case x "
                f"configuration pairs: max |difference| {diff.max():.1e}, {int((diff > 1e-9).sum())} differ; its `n_ref` equals "
                f"matched.py's `n_in` in {int((f['cc_n_ref'] == f['mp_n_in']).sum())} of {len(f)}.")
        (OUT / f"{part}_extras_report.md").write_text(md(t) + note, encoding="utf-8")
    for lab in ("F", "D"):
        p = OUT / f"C_proposal_{lab}.csv"
        if p.exists():
            (OUT / f"C_best_{lab}.md").write_text(md(pd.read_csv(p)), encoding="utf-8")


if __name__ == "__main__" and sys.argv[1] == "pieces":
    pieces()
