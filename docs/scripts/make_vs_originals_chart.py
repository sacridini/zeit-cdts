"""The chart of docs/benchmarks/vs-originals.md, an inline SVG snippet, from
benchmarks/results/vs-originals.json (written by benchmarks/vs_originals/*.py).

    python docs/scripts/make_vs_originals_chart.py

It uses the helpers and the CSS classes of ``make_benchmark_charts.py`` (light and dark
themes, a ``data-tip`` on each row) and writes docs/assets/charts/vs_originals.svg, included
in the page with ``--8<-- "docs/assets/charts/vs_originals.svg"``.
"""

import json
import math
import sys
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_benchmark_charts import svg, tip, write  # noqa: E402

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "vs-originals.json"

# one row per method: its largest workload (two for the AOA: 8 and 64 predictors)
ROWS = [
    ("Emerging hot spots", "emerging_hotspots", "sfdep 0.2.5 (R), 9 perm.", "100x100 cells x 12 dates",
     "sfdep, 9 permutations", "100 × 100 cells × 12 dates"),
    ("Recovery metrics", "recovery_metrics", "RecoverMetricsR 0.1.0 (R)", "1000x1000 px x 21 years",
     "RecoverMetricsR", "1000 × 1000 px × 21 years"),
    ("EWMACD", "ewmacd", "Brooks' EWMACD v1.3.0 (R)", "10000 pixels x 71 dates",
     "Brooks' EWMACD v1.3.0 script", "10,000 px × 71 dates"),
    ("S-CCD", "sccd", "pyxccd 1.1.0 sccd_detect", "1000 pixels x 1598 dates (synthetic)",
     "pyxccd 1.1.0", "1000 px × 1598 dates, synthetic"),
    ("Spectral unmixing", "unmix", "RStoolbox 1.0.2.2 (R)", "1000x1000 px x 6 bands x 4 em.",
     "RStoolbox mesma", "1000 × 1000 px × 6 bands"),
    ("Bayesian smoothing", "smooth_probs", "sits 1.5.4 (R)", "1000x1000 px x 4 classes",
     "sits", "1000 × 1000 px × 4 classes"),
    ("Area of applicability", "aoa", "CAST 1.0.4 (R)", "2000 train x 100000 new",
     "CAST", "2000 × 100,000 points, 8 predictors"),
    ("Area of applicability", "aoa", "CAST 1.0.4 (R)", "2000 train x 100000 new, 64 pred.",
     "CAST", "2000 × 100,000 points, 64 predictors"),
]


def fmt_s(s):
    if s >= 60:
        return f"{s / 60:.1f} min"
    if s >= 1:
        return f"{s:.1f} s"
    return f"{s * 1000:.0f} ms" if s >= 0.01 else f"{s * 1000:.1f} ms"


def chart_vs_originals():
    data = json.loads(RESULTS.read_text())["results"]
    W, row_h, top = 720, 46, 56
    x0, x1 = 215, 520
    lo, hi = math.log10(0.003), math.log10(1000)

    def X(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    def find(method, original, size, threads):
        for r in data:
            if (r["method"], r["original"], r["size"], r["threads"]) == (method, original, size, threads):
                return r
        raise KeyError((method, original, size, threads))

    H = top + row_h * len(ROWS) + 18
    b = ['<text class="zc-head" x="0" y="20">Method · workload</text>',
         f'<text class="zc-head" x="{x0}" y="20">Time of the call (log scale)</text>',
         f'<text class="zc-head" x="{W}" y="20" text-anchor="end">Faster, 1 / 19 threads</text>']
    for v, label in [(0.01, "10 ms"), (0.1, "100 ms"), (1, "1 s"), (10, "10 s"), (100, "100 s")]:
        b.append(f'<line class="zc-grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 12}" y2="{H - 14}"/>')
        b.append(f'<text class="zc-tick" x="{X(v):.1f}" y="{top - 18}" text-anchor="middle">{label}</text>')
    for i, (name, method, original, size, ref, work) in enumerate(ROWS):
        one, many = find(method, original, size, 1), find(method, original, size, 19)
        r, z1, z19 = one["original_s"], one["zeit_s"], many["zeit_s"]
        y = top + i * row_h
        cy = y + row_h / 2
        text = (f"{name}, {work}: {ref} {fmt_s(r)}; Zeit {fmt_s(z1)} on 1 thread ({one['speedup']:,.0f}× faster), "
                f"{fmt_s(z19)} on 19 ({many['speedup']:,.0f}×). Agreement: {one['note']}.")
        b.append(f'<g class="zc-row" {tip(text)}>')
        b.append(f'<rect class="zc-hit" x="0" y="{y}" width="{W}" height="{row_h}"/>')
        b.append(f'<text class="zc-name" x="0" y="{cy - 3:.1f}">{escape(name)}</text>')
        b.append(f'<text class="zc-sub" x="0" y="{cy + 13:.1f}">{escape(work)}</text>')
        b.append(f'<line class="zc-link" x1="{X(min(z1, z19)):.1f}" x2="{X(r):.1f}" y1="{cy}" y2="{cy}"/>')
        b.append(f'<circle class="zc-dot zc-dot--ref" cx="{X(r):.1f}" cy="{cy}" r="6"/>')
        b.append(f'<circle class="zc-dot zc-bar--zeit-lo" cx="{X(z1):.1f}" cy="{cy}" r="6"/>')
        b.append(f'<circle class="zc-dot zc-dot--zeit" cx="{X(z19):.1f}" cy="{cy}" r="6"/>')
        b.append(f'<text class="zc-val zc-val--soft" x="{X(min(z1, z19)) - 11:.1f}" y="{cy + 4:.1f}" '
                 f'text-anchor="end">{fmt_s(min(z1, z19))}</text>')
        b.append(f'<text class="zc-val zc-val--soft" x="{X(r) + 11:.1f}" y="{cy + 4:.1f}">{fmt_s(r)}</text>')
        b.append(f'<text class="zc-val" x="{W}" y="{cy + 4:.1f}" text-anchor="end">'
                 f'{one["speedup"]:,.0f}× / {many["speedup"]:,.0f}×</text>')
        b.append("</g>")
    write("vs_originals", svg(W, H, "Time of each original implementation and of Zeit on 1 and 19 threads, "
                                    "on the largest workload of each benchmark", b))


if __name__ == "__main__":
    chart_vs_originals()
