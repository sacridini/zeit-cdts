"""Generate the benchmark charts of the documentation as inline SVG snippets.

The numbers are the ones in the tables of docs/benchmarks/ (each chart has its table
next to it). The SVGs carry no colours of their own: they use the classes of
docs/assets/benchmarks.css, so they follow the light and dark themes, and each mark
has a ``data-tip`` that docs/javascripts/charts.js shows on hover and focus.

    python docs/scripts/make_benchmark_charts.py

Snippets are written to docs/assets/charts/ and included in the pages with
``--8<-- "docs/assets/charts/<name>.svg"``.
"""

import math
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets" / "charts"

# ---------------------------------------------------------------------------
# Data (docs/benchmarks/index.md and performance.md)
# ---------------------------------------------------------------------------
# name, reference tool, single-core speedup (low, high), fidelity tier, fidelity detail
TIERS = {
    "same": ("Identical", "Same output as the reference: vertices, breaks, labels or weights"),
    "float": ("≤ 1e-7", "Same up to floating-point rounding"),
    "close": ("Close", "Same method, small numerical differences (curve fits, break search)"),
    "analog": ("Analog", "No reference with the same architecture; compared to a conceptual analog"),
}
SINGLE_CORE = [
    ("Phenology", "R phenofit", (244.5, None), "close", "Start of season within 3.7 days (MAE), 99.1% within 15 days"),
    ("LandTrendr", "IDL original (GDL)", (168.5, None), "same", "100% identical vertex years (330/330 series)"),
    ("CCDC", "MATLAB original (Octave)", (83.7, 105), "same", "100% model match: dates, categories, coefficients"),
    ("TWDTW", "R twdtw / dtwSat", (43.9, None), "float", "Distances identical to R within 1e-12"),
    ("SOM", "Python minisom", (37.2, 126), "same", "Bit-exact codebooks (max weight difference 0.0)"),
    ("Mann-Kendall", "Python pymannkendall", (33, 41), "same", "100% trend agreement, mean p-value difference 0.0"),
    ("BFAST Monitor", "R bfast::bfastmonitor", (15.8, None), "same", "100% break agreement, magnitude correlation 1.0"),
    ("TempCNN", "R sits (torch)", (11.1, None), "float", "Max absolute difference 5.6e-9"),
    ("LightTAE", "R sits (torch)", (5.8, None), "float", "Max absolute difference 8.9e-8"),
    ("BFAST", "R bfast::bfast", (4.3, None), "close", "87.5% same break count, 100% same positions"),
    ("BFAST Lite", "R bfast::bfastlite", (2.5, None), "close", "100% same break count, 80% same index"),
    ("SNIC", "C reference (snic.c)", (1.8, 3.5), "same", "Bitwise-identical segment labels"),
    ("U-TAE", "Official utae-paps", (1.08, None), "same", "Max absolute difference 0.0"),
    ("Siamese CNN", "Conceptual analog", (0.85, None), "analog", "100% accuracy on the respective change tasks"),
]

# name, workload, Zeit 20 threads (ms), reference multi-worker (ms), reference framework
FINISH_FIRST = [
    ("Mann-Kendall", "5,000 series", 7.29, 20109, "Python multiprocessing, 19 workers"),
    ("BFAST Monitor", "20,000 series", 14.54, 5003, "R PSOCK cluster, 19 workers"),
    ("Phenology", "638 px × 25 years", 341, 93504, "R foreach + doParallel, 19 workers"),
    ("TWDTW", "3,000 series", 13.35, 188.67, "R PSOCK cluster, 19 workers"),
    ("BFAST", "400 series", 909, 3335, "R PSOCK cluster, 19 workers"),
    ("BFAST Lite", "2,000 series", 2019, 4551, "R PSOCK cluster, 19 workers"),
    ("TempCNN", "batch of 512", 3.94, 7.78, "R torch intra-op threads"),
    ("LightTAE", "batch of 512", 14.61, 26.98, "R torch intra-op threads"),
    ("U-TAE", "batch of 32", 126.0, 124.67, "PyTorch intra-op threads"),
]

GEE = [
    ("Google Earth Engine", "batch workers (EECU)", 27213, 1396.39, 18308, "ref"),
    ("Zeit, 1 thread", "desktop PC", 60492, 904.77, 0, "zeit-lo"),
    ("Zeit, 20 threads", "desktop PC", 556007, 98.44, 0, "zeit"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_x(v):
    return f"{v:.0f}×" if v >= 20 else f"{v:g}×"


def fmt_ms(v):
    if v >= 1000:
        return f"{v / 1000:,.1f} s"
    return f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms"


def svg(width, height, label, body):
    lines = [f'<svg class="zc" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}" '
             f'xmlns="http://www.w3.org/2000/svg">']
    lines += [b for b in body if b.strip()]
    lines.append("</svg>")
    return "\n".join(lines)


def tip(text):
    return f'data-tip="{escape(text, quote=True)}" tabindex="0"'


def write(name, content):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.svg").write_text(content + "\n", encoding="utf-8")
    print(f"wrote docs/assets/charts/{name}.svg")


# ---------------------------------------------------------------------------
# 1. Single-core speedup and fidelity, one row per algorithm
# ---------------------------------------------------------------------------
def chart_overview():
    W, row_h, top = 720, 42, 56
    x0, x1 = 272, 640                  # plot area
    lo, hi = math.log10(0.5), math.log10(400)

    def X(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    H = top + row_h * len(SINGLE_CORE) + 18
    b = []
    # column headers
    b.append(f'<text class="zc-head" x="0" y="20">Algorithm · reference</text>')
    b.append(f'<text class="zc-head" x="170" y="20">Fidelity</text>')
    b.append(f'<text class="zc-head" x="{x0}" y="20">Speedup, one core (log scale)</text>')
    # grid
    for v in (1, 10, 100):
        b.append(f'<line class="zc-grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 12}" y2="{H - 14}"/>')
        b.append(f'<text class="zc-tick" x="{X(v):.1f}" y="{top - 18}" text-anchor="middle">{v}×</text>')
    for v in (2, 3, 5, 20, 30, 50, 200, 300):
        b.append(f'<line class="zc-grid zc-grid--minor" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 12}" '
                 f'y2="{H - 14}"/>')
    b.append(f'<line class="zc-base" x1="{X(1):.1f}" x2="{X(1):.1f}" y1="{top - 12}" y2="{H - 14}"/>')
    for i, (name, ref, (v, vhi), tier, detail) in enumerate(SINGLE_CORE):
        y = top + i * row_h
        cy = y + row_h / 2
        slower = v < 1
        speed = fmt_x(v) + (f"–{fmt_x(vhi)}" if vhi else "")
        text = (f"{name} vs {ref}: {speed} {'slower' if slower else 'faster'} on one core. "
                f"Fidelity: {TIERS[tier][0]}, {detail}.")
        b.append(f'<g class="zc-row" {tip(text)}>')
        b.append(f'<rect class="zc-hit" x="0" y="{y}" width="{W}" height="{row_h}"/>')
        b.append(f'<text class="zc-name" x="0" y="{cy - 3:.1f}">{escape(name)}</text>')
        b.append(f'<text class="zc-sub" x="0" y="{cy + 13:.1f}">{escape(ref)}</text>')
        b.append(f'<g class="zc-pill zc-pill--{tier}"><rect x="170" y="{cy - 11:.1f}" width="80" height="22" '
                 f'rx="11"/><text x="210" y="{cy + 4:.1f}" text-anchor="middle">{TIERS[tier][0]}</text></g>')
        a, e = (X(v), X(1)) if slower else (X(1), X(v))
        cls = "zc-bar--ref" if slower else "zc-bar--zeit"
        if vhi:
            b.append(f'<rect class="zc-bar zc-bar--range" x="{a:.1f}" y="{cy - 8:.1f}" '
                     f'width="{X(vhi) - a:.1f}" height="16" rx="4"/>')
        b.append(f'<rect class="zc-bar {cls}" x="{a:.1f}" y="{cy - 8:.1f}" width="{max(e - a, 2):.1f}" '
                 f'height="16" rx="4"/>')
        end = X(vhi or v) if not slower else X(1)
        b.append(f'<text class="zc-val" x="{end + 7:.1f}" y="{cy + 4:.1f}">{speed}</text>')
        b.append("</g>")
    write("speedup_overview", svg(W, H, "Single-core speedup of Zeit over each reference tool, with the "
                                        "fidelity of each port", b))


# ---------------------------------------------------------------------------
# 2. Who finishes first when both use every core
# ---------------------------------------------------------------------------
def chart_finish_first():
    W, row_h, top = 720, 42, 56
    x0, x1 = 190, 580
    lo, hi = math.log10(2), math.log10(200000)

    def X(v):
        return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)

    rows = sorted(FINISH_FIRST, key=lambda r: -r[3] / r[2])
    H = top + row_h * len(rows) + 18
    b = []
    b.append('<text class="zc-head" x="0" y="20">Algorithm · workload</text>')
    b.append(f'<text class="zc-head" x="{x0}" y="20">Wall-clock time, every core (log scale)</text>')
    b.append(f'<text class="zc-head" x="{W}" y="20" text-anchor="end">Zeit finishes</text>')
    for v, label in [(10, "10 ms"), (100, "100 ms"), (1000, "1 s"), (10000, "10 s"), (100000, "100 s")]:
        b.append(f'<line class="zc-grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 12}" y2="{H - 14}"/>')
        b.append(f'<text class="zc-tick" x="{X(v):.1f}" y="{top - 18}" text-anchor="middle">{label}</text>')
    for i, (name, work, z, r, fw) in enumerate(rows):
        y = top + i * row_h
        cy = y + row_h / 2
        ratio = r / z
        verdict = "tied" if 0.9 < ratio < 1.1 else (f"{ratio:,.0f}× sooner" if ratio >= 10 else f"{ratio:.1f}× sooner")
        text = (f"{name}, {work}: Zeit with 20 threads {fmt_ms(z)}; reference ({fw}) {fmt_ms(r)}. "
                f"{'Both run on the PyTorch backend: tied.' if verdict == 'tied' else 'Zeit finishes ' + verdict + '.'}")
        b.append(f'<g class="zc-row" {tip(text)}>')
        b.append(f'<rect class="zc-hit" x="0" y="{y}" width="{W}" height="{row_h}"/>')
        b.append(f'<text class="zc-name" x="0" y="{cy - 3:.1f}">{escape(name)}</text>')
        b.append(f'<text class="zc-sub" x="0" y="{cy + 13:.1f}">{escape(work)}</text>')
        b.append(f'<line class="zc-link" x1="{X(min(z, r)):.1f}" x2="{X(max(z, r)):.1f}" y1="{cy}" y2="{cy}"/>')
        b.append(f'<circle class="zc-dot zc-dot--ref" cx="{X(r):.1f}" cy="{cy}" r="6"/>')
        b.append(f'<circle class="zc-dot zc-dot--zeit" cx="{X(z):.1f}" cy="{cy}" r="6"/>')
        b.append(f'<text class="zc-val zc-val--soft" x="{X(z) - 11:.1f}" y="{cy + 4:.1f}" '
                 f'text-anchor="end">{fmt_ms(z)}</text>')
        b.append(f'<text class="zc-val zc-val--soft" x="{X(r) + 11:.1f}" y="{cy + 4:.1f}">{fmt_ms(r)}</text>')
        b.append(f'<text class="zc-val" x="{W}" y="{cy + 4:.1f}" text-anchor="end">{verdict}</text>')
        b.append("</g>")
    write("finish_first", svg(W, H, "Wall-clock time of Zeit with 20 threads and of each reference tool "
                                    "with its parallel framework", b))


# ---------------------------------------------------------------------------
# 3. Full Landsat tile: throughput, and turnaround including the queue
# ---------------------------------------------------------------------------
def chart_gee():
    W, row_h = 720, 44
    x0, x1 = 178, 620
    b = []
    rows = len(GEE)

    def panel(top, title, scale_max, ticks, value, label, tipf, queue=False):
        def X(v):
            return x0 + v / scale_max * (x1 - x0)

        b.append(f'<text class="zc-head" x="0" y="{top - 22}">{title}</text>')
        for v, t in ticks:
            b.append(f'<line class="zc-grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 8}" '
                     f'y2="{top + rows * row_h}"/>')
            b.append(f'<text class="zc-tick" x="{X(v):.1f}" y="{top + rows * row_h + 15}" '
                     f'text-anchor="middle">{t}</text>')
        for i, row in enumerate(GEE):
            name, sub, pxs, comp, q, kind = row
            cy = top + i * row_h + row_h / 2
            b.append(f'<g class="zc-row" {tip(tipf(row))}>')
            b.append(f'<rect class="zc-hit" x="0" y="{cy - row_h / 2}" width="{W}" height="{row_h}"/>')
            b.append(f'<text class="zc-name" x="0" y="{cy - 3:.1f}">{escape(name)}</text>')
            b.append(f'<text class="zc-sub" x="0" y="{cy + 13:.1f}">{escape(sub)}</text>')
            start = x0
            if queue and q:
                qm = q / 60
                b.append(f'<rect class="zc-bar zc-bar--queue" x="{x0}" y="{cy - 9}" width="{X(qm) - x0:.1f}" '
                         f'height="18" rx="4"/>')
                b.append(f'<text class="zc-inbar" x="{(x0 + X(qm)) / 2:.1f}" y="{cy + 4:.1f}" '
                         f'text-anchor="middle">queue: {qm / 60:.1f} h</text>')
                start = X(qm) + 2
            v = value(row)
            width = max(X(v) - x0, 3)
            b.append(f'<rect class="zc-bar zc-bar--{kind}" x="{start:.1f}" y="{cy - 9}" width="{width:.1f}" '
                     f'height="18" rx="4"/>')
            b.append(f'<text class="zc-val" x="{start + width + 7:.1f}" y="{cy + 4:.1f}">{label(row)}</text>')
            b.append("</g>")

    panel(40, "Compute throughput, pixels per second", 600000,
          [(0, "0"), (200000, "200k"), (400000, "400k"), (600000, "600k")],
          lambda r: r[2], lambda r: f"{r[2] / 1000:,.0f}k",
          lambda r: f"{r[0]}: {r[2]:,} pixels per second ({r[3]:,.0f} s of compute).")

    def total(r):
        t = (r[3] + r[4]) / 60
        return f"{t / 60:.2f} h" if t > 60 else f"{t:.1f} min"

    top2 = 40 + rows * row_h + 70
    panel(top2, "Time to the result, minutes: queue + compute", 360,
          [(v, str(v)) for v in (0, 60, 120, 180, 240, 300, 360)],
          lambda r: r[3] / 60, total,
          lambda r: f"{r[0]}: {r[3] / 60:.1f} min of compute"
                    + (f" after {r[4] / 3600:.2f} h in the queue" if r[4] else ", no queue")
                    + f": {total(r)} in all.", queue=True)
    H = top2 + rows * row_h + 24
    write("gee_tile", svg(W, H, "LandTrendr on a full Landsat tile: Zeit on a desktop against Google Earth "
                                "Engine", b))


if __name__ == "__main__":
    chart_overview()
    chart_finish_first()
    chart_gee()
