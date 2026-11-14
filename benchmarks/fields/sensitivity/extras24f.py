"""24f: the expensive responses (the package's ``clip`` / ``complete_only`` metrics) at the selected configurations.

    python extras24f.py A     # the configurations of f24/A_selected.json -> f24/A_extras.csv  (resumable by cfg)
    python extras24f.py C     # the configurations of f24/C_selected.json -> f24/C_extras.csv

A configuration string is ``grad,dens,mask,low,ms`` (``dens`` ``n1000`` = 1000 markers per window, the notebook's; a
``grad`` with ``|cut-mc`` / ``|cut-exact`` is task C's hierarchy cut and ``low`` is its ``t``).
"""
import json
import sys

import pandas as pd

import s24f as S
import cases as C
import pipeline as P
from b24f import s1_of
from extras import pkg_metrics


def parse(spec: str):
    g, d, m, lo, ms = spec.split(",")
    return g, (d if d.startswith("n") else float(d)), m, float(lo), int(ms)


def preds_for(cs, cfg):
    """The predictions of ``cfg`` on every case (the boundary network at 0)."""
    g, d, m, low, ms = cfg
    out = []
    method = None
    if "|" in g:
        g, method = g.split("|")
    for c in cs:
        s1 = s1_of(c, (g, d, m, low, ms))
        keep = c.keep(s1.mask_threshold)
        if method is None:
            b = P.stage1(c, s1)
            pred = S.hyst_pred(c, keep, S.mc_regions(b.counts, b.r, low, S.high_for(low)), ms)
        else:
            if method == "cut-exact":
                h, v = S.exact_edges(c, s1)
            else:
                b = P.stage1(c, s1)
                h, v = S.edge_values_mc(b.counts, b.r)
            pred = S.cut_pred(c, keep, S.cut_components(h, v, keep, low), ms)
        out.append(pred)
    return out


def main():
    part = sys.argv[1]
    cs = C.load_cases()
    S.preload_gradients(cs, S.GRADS)
    sel = json.loads((S.OUT24 / f"{part}_selected.json").read_text())
    p = S.OUT24 / f"{part}_extras.csv"
    old = pd.read_csv(p) if p.exists() else pd.DataFrame()
    have = set(old["cfg"]) if len(old) else set()
    rows = []
    for spec in dict.fromkeys(sel.values()):
        if spec in have:
            continue
        preds = preds_for(cs, parse(spec))
        for c, pred in zip(cs, preds):
            rows.append({"cfg": spec, "case": c.id, "mosaic": c.name, "group": c.group, "kind": c.kind,
                         "region_pixels": int(c.regions[1].sum()), **pkg_metrics(c, pred)})
        S.log(f"{spec}: done")
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
    S.log("done")


if __name__ == "__main__":
    main()
