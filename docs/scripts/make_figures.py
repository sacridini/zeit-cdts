"""Generate the figures used in the Zeit documentation.

Every figure is produced by running Zeit itself, so the images double as a
smoke test of the documented API. Run from the repository root:

    python docs/scripts/make_figures.py            # every figure
    python docs/scripts/make_figures.py ccdc bfast_monitor   # a subset

Figures are written to ``docs/assets/figures/``.

Two kinds of inputs are used:

* **Synthetic series** (CCDC, BFAST family, Tmask, phenology, TWDTW,
  smoothing, TempCNN) are generated here with a fixed random seed and need
  nothing but Zeit.
* **Real Landsat data** (LandTrendr, Mann-Kendall, SNIC, SOM and the
  concept figures) use an annual NDVI stack (1985-2024, NDVI x 10000, one
  band per year) over a ~50 x 50 km area of Rondonia, Brazil, exported
  from LT-GEE (https://github.com/eMapR/LT-GEE) on Google Earth Engine. Point the
  ``ZEIT_DOCS_RONDONIA`` environment variable at that GeoTIFF; the
  real-data figures are skipped when it is not set. The same stack gives the
  layers of the home page (``hero_stack``) and the real sample of the accuracy
  tutorial.
* **Embeddings** (the embeddings tutorial) of AlphaEarth and TESSERA over a
  5 x 5 km box of that scene are downloaded once (about 2 GB, a minute or two)
  into ``build/docs_embeddings/``, or the folder ``ZEIT_DOCS_EMBEDDINGS`` points
  at. Their comparison with LandTrendr also needs ``ZEIT_DOCS_RONDONIA``.
* The **accuracy** figures use the numerical example of Olofsson et al. (2014),
  from ``tests/data/accuracy_sits.json``.

The benchmark charts are inline SVGs made by ``make_benchmark_charts.py``.

Charts are written as palette PNGs, image-like maps as WebP.

Optional: set ``ZEIT_DOCS_FONT_DIR`` to a folder with the Inter TTF files so
the figures use the same typeface as the site.
"""

import os
import sys
import glob
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

import zeit

OUT = Path(__file__).resolve().parents[1] / "assets" / "figures"
RNG_SEED = 7

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
INK = "#1d2939"        # primary text
INK2 = "#52514e"       # secondary text
MUTED = "#898781"      # axis ticks / labels
GRID = "#e6e5df"       # hairline gridlines
AXIS = "#c3c2b7"       # baseline
OBS = "#a8a8a3"        # raw observations (recessive)

# Categorical slots, fixed order (validated: blue, orange, aqua are CVD-safe
# for every pair; violet is only used as a 4th slot in facetted figures).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"

GREENS = LinearSegmentedColormap.from_list(
    "ndvi", ["#f4f1e4", "#c9dfa4", "#7fb069", "#3a7d44", "#1b4d2b"])
YEARS = LinearSegmentedColormap.from_list(
    "years", ["#fbe3a1", "#f5a45d", "#e0603f", "#a8325e", "#5b1f6b", "#231942"])
DIVERGING = LinearSegmentedColormap.from_list(
    "trend", ["#8c4a14", "#c98a4b", "#f0efec", "#5aa39a", "#0f5c55"])


def setup_style():
    font_dir = os.environ.get("ZEIT_DOCS_FONT_DIR")
    family = ["DejaVu Sans"]
    if font_dir:
        for f in glob.glob(os.path.join(font_dir, "**", "Inter-*.ttf"), recursive=True):
            font_manager.fontManager.addfont(f)
        family = ["Inter", "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": family,
        "font.size": 10.5,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 160,
    })


def save(fig, name, photo=False):
    """Charts are saved as palette PNGs; image-like maps (photo=True) as WebP."""
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.12, dpi=130 if photo else None)
    plt.close(fig)
    img = Image.open(png).convert("RGB")
    if photo:
        png.unlink()
        path = OUT / f"{name}.webp"
        img.save(path, quality=88, method=6)
    else:
        path = png
        img.quantize(256, method=Image.Quantize.MEDIANCUT,
                     dither=Image.Dither.NONE).save(path, optimize=True)
    print(f"  wrote {path.relative_to(OUT.parents[2])}  ({path.stat().st_size / 1024:.0f} KB)")


def map_axes(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


def colorbar(fig, im, ax, label, ticks=None, fmt=None):
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.03,
                      ticks=ticks, format=fmt)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=MUTED, labelcolor=INK2, length=0)
    cb.set_label(label, color=INK2, fontsize=9.5)
    return cb


def obs_scatter(ax, x, y, label="Observations", **kw):
    kw.setdefault("s", 14)
    ax.scatter(x, y, color=OBS, edgecolor="white", linewidth=0.6, zorder=2, label=label, **kw)


def break_line(ax, x, label=None):
    ax.axvline(x, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1, label=label)


def year_axis(ax, step=1):
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))


def legend_above(ax, ncol=4):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=ncol, borderaxespad=0)


def frac_year(ordinals):
    from datetime import date
    out = []
    for o in np.atleast_1d(ordinals):
        d = date.fromordinal(int(o))
        start = date(d.year, 1, 1).toordinal()
        out.append(d.year + (int(o) - start) / 365.25)
    return np.array(out)


# ---------------------------------------------------------------------------
# Real data: Rondonia annual NDVI, 1985-2024
# ---------------------------------------------------------------------------
_RONDONIA = {}


def rondonia():
    """Load the stack once and run LandTrendr on it (cached)."""
    if _RONDONIA:
        return _RONDONIA
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    if not path or not os.path.exists(path):
        return None
    import rasterio
    with rasterio.open(path) as src:
        stack = src.read().astype(np.float32)
    years = np.arange(1985, 1985 + stack.shape[0])
    valid = (stack != 0).all(axis=0)

    lt = zeit.landtrendr(stack, years=years, max_segments=6, nodata=0)   # 0 = masked year
    events = zeit.extract_events(lt, event_type="loss", sort_by="greatest", min_magnitude=2000)
    events = {k: v.values for k, v in events.items()}
    vertices = lt_vertex_stack(lt)
    rmse = lt.rmse.values
    _RONDONIA.update(stack=stack, years=years, valid=valid,
                     vertices=vertices, rmse=rmse, events=events)
    return _RONDONIA


def lt_vertex_stack(lt):
    """zeit.landtrendr's Dataset as a (2 * vertices, rows, cols) array: vertex years,
    then fitted values, 0 in unused slots."""
    return np.concatenate([lt.vertex_year.values.astype(np.float32),
                           np.nan_to_num(lt.vertex_value.values, nan=0.0)])


def fitted_trajectory(vertices, r, c, years):
    nv = vertices.shape[0] // 2
    vy = vertices[:nv, r, c]
    vv = vertices[nv:, r, c]
    keep = vy > 0
    return vy[keep], vv[keep], np.interp(years, vy[keep], vv[keep])


def _mmu(mask, min_pixels):
    """Drop connected patches smaller than min_pixels (what apply_mmu_filter does on files)."""
    from scipy import ndimage
    lab, _ = ndimage.label(mask)
    sizes = np.bincount(lab.ravel())
    return mask & (sizes[lab] >= min_pixels)


def _masked(a, valid):
    return np.where(valid, a, np.nan)


def fig_landtrendr_maps(d):
    s, y, valid, ev = d["stack"], d["years"], d["valid"], d["events"]
    # extract_events' "yod" is the vertex year where the loss segment starts,
    # i.e. the last year before the drop: +1 gives the first year it shows.
    yod = ev["yod"].astype(float) + 1
    keep = valid & (ev["yod"] > y[0]) & (ev["dsnr"] >= 3)
    yod = np.where(_mmu(keep, 11), yod, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.04})
    for ax, idx in zip(axes[:2], [0, -1]):
        im = ax.imshow(_masked(s[idx] / 10000, valid), cmap=GREENS, vmin=0.2, vmax=0.9,
                       interpolation="nearest")
        map_axes(ax, f"NDVI, {y[idx]}")
    colorbar(fig, im, axes[:2].tolist(), "NDVI", ticks=[0.2, 0.4, 0.6, 0.8])

    ax = axes[2]
    ax.imshow(_masked(s[-1] / 10000, valid), cmap="Greys", vmin=-0.5, vmax=3.0,
              interpolation="nearest")
    im = ax.imshow(yod, cmap=YEARS, vmin=1986, vmax=y[-1], interpolation="nearest")
    map_axes(ax, "Year of greatest vegetation loss (LandTrendr)")
    colorbar(fig, im, ax, "First year of loss", ticks=[1990, 2000, 2010, 2020])
    save(fig, "landtrendr_maps", photo=True)


HERO_YEARS = [1990, 2000, 2010, 2024]


def fig_hero_stack(d):
    """Home page: a stack of yearly NDVI images of one place, the LandTrendr map on top, and
    the time series of one pixel through the stack (an inline SVG the page styles)."""
    from PIL import Image
    s, y, v, valid, ev = d["stack"], d["years"], d["vertices"], d["valid"], d["events"]
    _, (r, c) = _pick_pixels(d)[0]
    h = 120
    r0, c0 = min(max(r - h, 0), s.shape[1] - 2 * h), min(max(c - h + 30, 0), s.shape[2] - 2 * h)
    sl = (slice(r0, r0 + 2 * h), slice(c0, c0 + 2 * h))
    out = OUT / "hero"
    out.mkdir(parents=True, exist_ok=True)

    def write(rgb, name):
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., :3])
        img = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
        img.save(out / f"{name}.webp", quality=86, method=6)

    for yr in HERO_YEARS:
        a = s[yr - y[0]][sl] / 10000
        rgb = GREENS(np.clip((a - 0.2) / 0.7, 0, 1))
        rgb[~valid[sl]] = (0.93, 0.93, 0.91, 1)
        write(rgb, f"ndvi_{yr}")
    yod = ev["yod"].astype(float) + 1
    keep = valid & (ev["yod"] > y[0]) & (ev["dsnr"] >= 3)
    yod = np.where(_mmu(keep, 11), yod, np.nan)[sl]
    grey = plt.get_cmap("Greys")(np.clip((s[-1][sl] / 10000 + 0.5) / 3.5, 0, 1))
    col = YEARS(np.clip((yod - 1986) / (y[-1] - 1986), 0, 1))
    rgb = np.where(np.isfinite(yod)[..., None], col, grey)
    write(rgb, "loss_year")
    px, py = (c - c0 + 0.5) / (2 * h) * 100, (r - r0 + 0.5) / (2 * h) * 100
    print(f"  pixel at left {px:.1f}%, top {py:.1f}%; loss year {int(ev['yod'][r, c]) + 1}")

    # the pixel's series, as an inline SVG snippet
    vy, vv, fit = fitted_trajectory(v, r, c, y)
    obs = s[:, r, c] / 10000
    W, H, L, R, T, B = 340, 150, 30, 10, 12, 24

    def X(t):
        return L + (t - 1984) / (2025 - 1984) * (W - L - R)

    def Y(val):
        return T + (1.0 - val) / (1.0 - 0.1) * (H - T - B)

    loss = int(ev["yod"][r, c]) + 1
    parts = [f'<svg class="zh-chart" viewBox="0 0 {W} {H}" role="img" aria-label="NDVI of the marked pixel, '
             f'1985 to 2024, with the LandTrendr fit: forest until {loss - 1}, cleared in {loss}" '
             'xmlns="http://www.w3.org/2000/svg">']
    for val in (0.2, 0.5, 0.8):
        parts.append(f'<line class="zh-grid" x1="{L}" x2="{W - R}" y1="{Y(val):.1f}" y2="{Y(val):.1f}"/>')
        parts.append(f'<text class="zh-tick" x="{L - 5}" y="{Y(val) + 3.5:.1f}" text-anchor="end">{val:.1f}</text>')
    for t in (1990, 2000, 2010, 2020):
        parts.append(f'<text class="zh-tick" x="{X(t):.1f}" y="{H - 6}" text-anchor="middle">{t}</text>')
    parts.append(f'<line class="zh-event" x1="{X(loss - 0.5):.1f}" x2="{X(loss - 0.5):.1f}" y1="{T}" '
                 f'y2="{H - B}"/>')
    parts.append(f'<text class="zh-label" x="{X(loss - 0.5) + 5:.1f}" y="{T + 9}">cleared in {loss}</text>')
    for t, o in zip(y, obs):
        if o > 0:
            parts.append(f'<circle class="zh-obs" cx="{X(t):.1f}" cy="{Y(o):.1f}" r="2.6"/>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{X(t):.1f},{Y(f / 10000):.1f}" for i, (t, f) in
                    enumerate(zip(vy, vv)))
    parts.append(f'<path class="zh-fit" d="{path}"/>')
    for t, f in zip(vy, vv):
        parts.append(f'<circle class="zh-vertex" cx="{X(t):.1f}" cy="{Y(f / 10000):.1f}" r="3.6"/>')
    parts.append("</svg>")
    (out / "pixel_chart.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(OUT.parents[2])} (layers and pixel_chart.svg)")


def _pick_pixels(d):
    v, y, ev, valid = d["vertices"], d["years"], d["events"], d["valid"]
    yod, mag, dur = ev["yod"], ev["magnitude"], ev["duration"]
    post, pre = ev["post_val"], ev["pre_val"]
    last = d["stack"][-1]
    rng = np.random.default_rng(RNG_SEED)

    def pick(mask):
        rr, cc = np.nonzero(mask & valid)
        i = rng.integers(len(rr))
        return rr[i], cc[i]

    abrupt = pick((yod >= 1998) & (yod <= 2004) & (dur == 1) & (mag > 4000) & (last < 7000)
                  & (ev["dsnr"] > 8))
    regrowth = pick((yod >= 1995) & (yod <= 2002) & (dur <= 2) & (mag > 3500) & (last > 7800)
                    & (ev["dsnr"] > 6))
    stable = pick((yod == 0) & (d["stack"].min(axis=0) > 7800))
    return [("Clear-cut, then pasture", abrupt), ("Clear-cut, then regrowth", regrowth),
            ("Undisturbed forest", stable)]


def fig_landtrendr_pixels(d):
    s, y, v = d["stack"], d["years"], d["vertices"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, (title, (r, c)) in zip(axes, _pick_pixels(d)):
        vy, vv, fit = fitted_trajectory(v, r, c, y)
        obs_scatter(ax, y, s[:, r, c] / 10000, s=18)
        ax.plot(y, fit / 10000, color=BLUE, label="LandTrendr fit")
        ax.scatter(vy, vv / 10000, s=42, color=BLUE, edgecolor="white", linewidth=1.5,
                   zorder=4, label="Vertices")
        ax.set_title(title)
        ax.set_ylim(0.1, 1.0)
        ax.set_xlim(1983, 2026)
        year_axis(ax, 10)
    axes[0].set_ylabel("NDVI")
    axes[0].legend(loc="lower left")
    save(fig, "landtrendr_pixels")


def fig_pixel_time_series(d):
    """Concept figure: a stack of images is a time series per pixel."""
    s, y, v, valid = d["stack"], d["years"], d["vertices"], d["valid"]
    _, (r, c) = _pick_pixels(d)[0]
    h = 110
    r0, c0 = max(r - h, 0), max(c - h, 0)
    crop = s[:, r0:r0 + 2 * h, c0:c0 + 2 * h]
    fig = plt.figure(figsize=(13.5, 3.7))
    left, right = fig.subfigures(1, 2, width_ratios=[1.2, 1], wspace=0.0)
    left.suptitle("A stack of yearly images…", x=0.02, ha="left", fontsize=11.5,
                  fontweight="semibold", color=INK)
    axes = left.subplots(1, 3, gridspec_kw={"wspace": 0.05})
    left.subplots_adjust(left=0.01, right=0.99)
    for ax, yr in zip(axes, [1990, 2000, 2020]):
        ax.imshow(crop[yr - y[0]] / 10000, cmap=GREENS, vmin=0.2, vmax=0.9, interpolation="nearest")
        ax.scatter([c - c0], [r - r0], s=80, facecolor="none", edgecolor=ORANGE, linewidth=2)
        map_axes(ax)
        ax.set_xlabel(str(yr), color=INK2)
    ax = right.subplots()
    right.subplots_adjust(left=0.12, bottom=0.14, top=0.86)
    for yr in [1990, 2000, 2020]:
        ax.axvline(yr, color="#f1efe8", linewidth=7, zorder=0)
    ax.plot(y, s[:, r, c] / 10000, color=OBS, linewidth=1.2, zorder=1)
    obs_scatter(ax, y, s[:, r, c] / 10000, s=22)
    ax.set_title("…is one time series per pixel (the circled one)")
    ax.set_ylim(0.1, 1.0)
    ax.set_ylabel("NDVI")
    year_axis(ax, 10)
    save(fig, "concept_pixel_time_series", photo=True)


def fig_mann_kendall(d):
    s, valid = d["stack"], d["valid"]
    mk = zeit.mann_kendall(np.where(valid, s / 10000, np.nan).astype(np.float32), method="hamed_rao")
    slope = mk.slope.values * 10  # per decade
    h = mk.h.values

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw={"wspace": 0.04})
    im = axes[0].imshow(_masked(slope, valid), cmap=DIVERGING, vmin=-0.2, vmax=0.2,
                        interpolation="nearest")
    map_axes(axes[0], "Theil-Sen slope, 1985–2024")
    colorbar(fig, im, axes[0], "NDVI change per decade", ticks=[-0.2, -0.1, 0, 0.1, 0.2])

    cls = np.full(slope.shape, np.nan)
    cls[valid & (h == 0)] = 1
    cls[valid & (h == 1) & (slope < 0)] = 0
    cls[valid & (h == 1) & (slope > 0)] = 2
    cmap = ListedColormap(["#a0612a", "#e4e2dc", "#237a70"])
    axes[1].imshow(cls, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    map_axes(axes[1], "Significant trends (p < 0.05)")
    share = [np.mean(cls[valid] == k) * 100 for k in range(3)]
    handles = [Line2D([], [], marker="s", linestyle="", markersize=10, color=col, label=lab)
               for col, lab in [("#a0612a", f"Browning  {share[0]:.0f}%"),
                                ("#e4e2dc", f"No trend  {share[1]:.0f}%"),
                                ("#237a70", f"Greening  {share[2]:.0f}%")]]
    axes[1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                   handletextpad=0.3, columnspacing=1.2)
    save(fig, "mann_kendall_map", photo=True)


def fig_snic(d):
    from skimage.segmentation import find_boundaries
    s, valid = d["stack"], d["valid"]
    r0, c0, n = 560, 560, 320
    crop = s[:, r0:r0 + n, c0:c0 + n] / 10000
    res = zeit.run_snic(crop.astype(np.float32), spacing=14, compactness=0.6)
    b = find_boundaries(res.labels, mode="inner")
    mean_img = res.means[res.labels][..., -1]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), gridspec_kw={"wspace": 0.04})
    axes[0].imshow(crop[-1], cmap=GREENS, vmin=0.2, vmax=0.9, interpolation="nearest")
    map_axes(axes[0], "NDVI 2024 (one of 40 layers)")
    rgb = GREENS(np.clip((crop[-1] - 0.2) / 0.7, 0, 1))
    rgb[b] = matplotlib.colors.to_rgba(ORANGE)
    axes[1].imshow(rgb, interpolation="nearest")
    map_axes(axes[1], f"{len(np.unique(res.labels))} SNIC superpixels")
    axes[2].imshow(mean_img, cmap=GREENS, vmin=0.2, vmax=0.9, interpolation="nearest")
    map_axes(axes[2], "Each segment filled with its mean")
    save(fig, "snic_segments", photo=True)


def fig_som(d):
    from zeit.ai import SOM
    s, y, valid = d["stack"], d["years"], d["valid"]
    r0, c0, n = 420, 420, 700
    crop = s[:, r0:r0 + n, c0:c0 + n] / 10000
    vmask = valid[r0:r0 + n, c0:c0 + n]
    X = crop.reshape(len(y), -1).T
    rng = np.random.default_rng(RNG_SEED)
    idx = np.flatnonzero(vmask.ravel())
    train = X[rng.choice(idx, 30000, replace=False)]
    som = SOM(x=2, y=2, input_len=len(y), sigma=0.8, learning_rate=0.5, random_seed=RNG_SEED)
    som.random_weights_init(train)
    som.train(train, num_iters=20, algorithm="batch")
    bmu = som.predict(X).astype(float)
    bmu[~vmask.ravel()] = np.nan
    bmu = bmu.reshape(n, n)
    protos = som.get_weights().reshape(-1, len(y))
    order = np.argsort(-protos.mean(axis=1))  # most forest-like first
    colors = [AQUA, BLUE, ORANGE, VIOLET]
    remap = np.full(4, 0)
    remap[order] = np.arange(4)
    cls = np.where(np.isnan(bmu), np.nan, remap[np.nan_to_num(bmu).astype(int)])

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cls, cmap=ListedColormap(colors), vmin=0, vmax=3, interpolation="nearest")
    map_axes(ax, "Pixels grouped by trajectory (2×2 batch SOM)")
    ax = fig.add_subplot(gs[0, 1])
    for k, j in enumerate(order):
        share = np.nanmean(cls == k) * 100
        ax.plot(y, protos[j], color=colors[k], label=f"Cluster {k + 1}  ({share:.0f}% of pixels)")
    ax.set_title("Prototype NDVI trajectory of each cluster")
    ax.set_ylabel("NDVI")
    ax.set_ylim(0.3, 0.95)
    ax.legend(loc="lower left")
    save(fig, "som_clusters", photo=True)


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------
def _landsat_dates(start_year, end_year, step=16, cloud_frac=0.35, seed=RNG_SEED):
    from datetime import date
    rng = np.random.default_rng(seed)
    d0, d1 = date(start_year, 1, 5).toordinal(), date(end_year, 12, 31).toordinal()
    dates = np.arange(d0, d1, step)
    return dates, rng.random(len(dates)) < cloud_frac


def _harmonic(t_year, a, b):
    w = 2 * np.pi * t_year
    return a * np.cos(w) + b * np.sin(w)


def fig_quickstart():
    """Runs the exact code shown in getting-started/quickstart.md."""
    # --- quickstart code (keep in sync with the page) ----------------------
    import numpy as np
    import zeit

    rng = np.random.default_rng(42)
    years = np.arange(1990, 2025)                      # 35 annual observations
    n_years, rows, cols = len(years), 120, 120

    # A forest (NDVI ~0.85) with noise...
    stack = 0.85 + rng.normal(0, 0.03, (n_years, rows, cols))
    # ...a field cleared in 2003 that stays as pasture,
    stack[years >= 2003, 20:60, 15:70] = 0.35 + rng.normal(0, 0.04, (np.sum(years >= 2003), 40, 55))
    # ...and a patch cleared in 2016 that starts to regrow.
    regrow = np.clip((years - 2016) * 0.06, 0, 0.4)[:, None, None]
    after = (years >= 2016)[:, None, None]
    stack[:, 70:105, 50:110] = np.where(after, 0.30 + regrow, stack[:, 70:105, 50:110])
    stack = (stack * 10000).astype(np.float32)         # NDVI x 10000, like most products

    lt = zeit.landtrendr(stack, years=years)
    loss = zeit.extract_events(lt, min_magnitude=1500)
    print(np.unique(loss["yod"]))
    # ------------------------------------------------------------------------

    vertices = lt_vertex_stack(lt)
    yod = loss["yod"].values.astype(float)
    yod[yod == 0] = np.nan
    fig = plt.figure(figsize=(12.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(stack[-1] / 10000, cmap=GREENS, vmin=0.2, vmax=0.9)
    map_axes(ax, "NDVI in 2024")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(np.full(yod.shape, 0.0), cmap="Greys", vmin=0, vmax=8)
    im = ax.imshow(yod, cmap=YEARS, vmin=1991, vmax=2024)
    map_axes(ax, 'loss["yod"]')
    colorbar(fig, im, ax, "Year of loss", ticks=[2000, 2010, 2020])
    ax = fig.add_subplot(gs[0, 2])
    for (r, c), col, lab in [((40, 40), BLUE, "Pixel in the 2003 clearing"),
                             ((90, 80), ORANGE, "Pixel in the 2016 clearing")]:
        vy, vv, fit = fitted_trajectory(vertices, r, c, years)
        ax.scatter(years, stack[:, r, c] / 10000, s=12, color=OBS, edgecolor="white",
                   linewidth=0.5, zorder=2)
        ax.plot(years, fit / 10000, color=col, label=lab)
    ax.set_title("Observations and LandTrendr fit")
    ax.set_ylabel("NDVI")
    ax.set_ylim(0.1, 1.0)
    year_axis(ax, 10)
    ax.legend(loc="lower left")
    save(fig, "quickstart_result")


def fig_ccdc():
    from zeit._ccdc import run_ccdc, predict
    rng = np.random.default_rng(RNG_SEED)
    dates, cloudy = _landsat_dates(2008, 2021)
    ty = frac_year(dates)
    brk = 2014.55
    after = ty >= brk
    # Blue, Green, Red, NIR, SWIR1, SWIR2 (reflectance x 10000):
    forest = np.array([250, 450, 300, 3100, 1300, 550])
    pasture = np.array([450, 750, 700, 2700, 2500, 1400])
    seas_f = np.array([20, 40, 30, 350, 120, 60])
    seas_p = np.array([60, 90, 180, 600, 450, 300])
    vals = np.zeros((6, len(dates)))
    for b in range(6):
        vals[b] = np.where(after, pasture[b] + _harmonic(ty, seas_p[b], -seas_p[b] * 0.4),
                           forest[b] + _harmonic(ty, seas_f[b], -seas_f[b] * 0.4))
        vals[b] += rng.normal(0, [25, 30, 35, 120, 80, 50][b], len(dates))
    qa = np.where(cloudy, 4, 0)
    models = run_ccdc(dates, vals, qa)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True, gridspec_kw={"hspace": 0.22})
    clear = ~cloudy
    for ax, b, name in zip(axes, [3, 4], ["NIR", "SWIR1"]):
        obs_scatter(ax, ty[clear], vals[b, clear] / 10000, label="Clear observations")
        for i, m in enumerate(models):
            tt = np.arange(m["t_start"], m["t_end"] + 1, 4)
            ax.plot(frac_year(tt), predict(np.asarray(m["coefs"])[b], tt) / 10000, color=BLUE,
                    label="Harmonic model" if i == 0 else None)
            if m["t_break"] > 0:
                break_line(ax, frac_year(m["t_break"])[0],
                           label="Detected break" if i == 0 else None)
        ax.set_ylabel(f"{name} reflectance")
        year_axis(ax)
    legend_above(axes[0], ncol=3)
    save(fig, "ccdc_fit")
    return models


def fig_tmask():
    from zeit.tmask import run_tmask_pixel
    rng = np.random.default_rng(RNG_SEED + 1)
    dates, _ = _landsat_dates(2016, 2019, cloud_frac=0)
    ty = frac_year(dates)
    green = 600 + _harmonic(ty, 90, -40) + rng.normal(0, 25, len(dates))
    swir = 1800 + _harmonic(ty, 250, -120) + rng.normal(0, 40, len(dates))
    cloud = rng.choice(len(dates), 7, replace=False)
    shadow = rng.choice(np.setdiff1d(np.arange(len(dates)), cloud), 5, replace=False)
    green[cloud] += rng.uniform(900, 2500, len(cloud))
    swir[cloud] += rng.uniform(300, 900, len(cloud))
    swir[shadow] -= rng.uniform(700, 1100, len(shadow))
    green[shadow] -= rng.uniform(80, 200, len(shadow))
    clear = run_tmask_pixel(dates - dates[0] + 1, green, swir)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"hspace": 0.25})
    for ax, band, name in zip(axes, [green, swir], ["Green", "SWIR1"]):
        ax.scatter(ty[clear], band[clear] / 10000, s=18, color=BLUE, edgecolor="white",
                   linewidth=0.6, label="Kept (clear)", zorder=3)
        ax.scatter(ty[~clear], band[~clear] / 10000, s=46, marker="X", color=ORANGE,
                   edgecolor="white", linewidth=0.6, label="Flagged by Tmask", zorder=4)
        ax.set_ylabel(f"{name} reflectance")
        year_axis(ax)
    legend_above(axes[0], ncol=2)
    save(fig, "tmask_flags")


def _ndvi_series(start, years, freq=23, seed=RNG_SEED, noise=0.03):
    rng = np.random.default_rng(seed)
    t = start + np.arange(years * freq) / freq
    base = 0.62 + _harmonic(t, -0.12, 0.05)
    return t, base + rng.normal(0, noise, len(t)), rng


def _ols_harmonic(t, y, order=3):
    cols = [np.ones_like(t), t]
    for k in range(1, order + 1):
        cols += [np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
    X = np.column_stack(cols)
    ok = np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return X @ beta


def fig_bfast_monitor():
    t, y, rng = _ndvi_series(2010, 10, seed=RNG_SEED + 2)
    brk = 2017.4
    y[t >= brk] -= 0.25
    y[rng.random(len(t)) < 0.12] = np.nan
    found = float(zeit.bfast_monitor(y, 2016.0, start_time=2010.0, frequency=23).breakpoint)

    hist = t < 2016.0
    fit_hist = _ols_harmonic(t[hist], y[hist])
    cols = [np.ones_like(t), t] + sum(([np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]
                                       for k in range(1, 4)), [])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X[hist][np.isfinite(y[hist])], y[hist][np.isfinite(y[hist])],
                               rcond=None)
    pred = X @ beta

    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.axvspan(2016.0, t[-1] + 0.1, color="#f3f2ee", zorder=0)
    ax.text(2010.1, 0.93, "History: the stable model is fitted here", color=INK2, fontsize=9.5)
    ax.text(2016.1, 0.93, "Monitoring: new data is tested against it", color=INK2, fontsize=9.5)
    obs_scatter(ax, t, y)
    ax.plot(t[hist], pred[hist], color=BLUE, label="Stable-history model")
    ax.plot(t[~hist], pred[~hist], color=BLUE, linestyle=(0, (2, 2)), linewidth=1.6,
            label="Model projected forward")
    break_line(ax, found, label=f"Break detected: {found:.2f}")
    ax.set_ylim(0.15, 0.99)
    ax.set_ylabel("NDVI")
    year_axis(ax)
    legend_above(ax, ncol=4)
    save(fig, "bfast_monitor")
    return found


def fig_bfast_lite():
    t, y, rng = _ndvi_series(2005, 16, seed=RNG_SEED + 3)
    y[(t >= 2010.3)] -= 0.22
    y[(t >= 2015.6)] += 0.12 + 0.03 * (t[t >= 2015.6] - 2015.6)
    out = zeit.bfast_lite(y, start_time=2005.0, frequency=23)
    idx = [int(out[f"breakpoint_idx_{i + 1}"]) for i in range(int(out.n_breaks))]

    fig, ax = plt.subplots(figsize=(12, 3.9))
    obs_scatter(ax, t, y)
    bounds = [0] + [i + 1 for i in idx] + [len(t)]
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        ax.plot(t[a:b], _ols_harmonic(t[a:b], y[a:b]), color=BLUE,
                label="Segment model (trend + season)" if k == 0 else None)
    for k, i in enumerate(idx):
        break_line(ax, t[i], label="Detected breaks" if k == 0 else None)
    ax.set_ylabel("NDVI")
    ax.set_ylim(0.1, 0.95)
    year_axis(ax, 2)
    legend_above(ax, ncol=3)
    save(fig, "bfast_lite")
    return t[idx]


def fig_bfast_classic():
    t, y, rng = _ndvi_series(2005, 13, seed=RNG_SEED + 4, noise=0.025)
    brk = 2011.5
    y[t >= brk] -= 0.18
    out = zeit.bfast(y, start_time=2005.0, frequency=23)
    ntb = int(out.n_trend_breaks)
    idx = [int(out[f"trend_breakpoint_idx_{i + 1}"]) for i in range(ntb)]

    # Decompose for display: season = harmonic fit on the whole series,
    # trend = piecewise linear on the deseasonalised series.
    bounds = [0] + [i + 1 for i in idx] + [len(t)]
    X = np.column_stack([np.cos(2 * np.pi * k * t) for k in (1, 2, 3)] +
                        [np.sin(2 * np.pi * k * t) for k in (1, 2, 3)])
    trend = np.zeros_like(y)
    for a, b in zip(bounds[:-1], bounds[1:]):
        trend[a:b] = np.polyval(np.polyfit(t[a:b], y[a:b], 1), t[a:b])
    beta, *_ = np.linalg.lstsq(X, y - trend, rcond=None)
    season = X @ beta
    remainder = y - trend - season

    fig, axes = plt.subplots(4, 1, figsize=(12, 7.4), sharex=True,
                             gridspec_kw={"hspace": 0.35, "height_ratios": [1.3, 1, 1, 0.8]})
    obs_scatter(axes[0], t, y, s=10)
    axes[0].set_title("Observed NDVI")
    axes[1].plot(t, trend, color=BLUE)
    axes[1].set_title("Trend component")
    axes[2].plot(t, season, color=AQUA)
    axes[2].set_title("Seasonal component")
    axes[3].vlines(t, 0, remainder, color=OBS, linewidth=1.2)
    axes[3].axhline(0, color=AXIS, linewidth=0.8)
    axes[3].set_title("Remainder")
    year_axis(axes[3], 2)
    for ax in axes[:2]:
        for i in idx:
            break_line(ax, t[i])
    axes[1].text(t[idx[0]] + 0.1, trend.max(), f"  trend break\n  magnitude {float(out.magnitude):.2f}",
                 color=INK2, fontsize=9, va="top")
    save(fig, "bfast_decomposition")


def fig_phenology():
    import pandas as pd
    rng = np.random.default_rng(RNG_SEED + 5)
    base = 2019
    doy = np.arange(1, 3 * 365, 8)
    t = doy / 365.0
    frac = t % 1

    def dl(x, sos, eos, lo=0.2, hi=0.82):
        return lo + (hi - lo) * (1 / (1 + np.exp(-(x - sos) / 0.025)) - 1 / (1 + np.exp(-(x - eos) / 0.03)))

    shift = np.floor(t).astype(int)
    sos = np.array([0.30, 0.34, 0.29])[shift]
    eos = np.array([0.72, 0.70, 0.74])[shift]
    truth = dl(frac, sos, eos)
    y = truth + rng.normal(0, 0.03, len(t))
    cloud = rng.random(len(t)) < 0.15
    y[cloud] -= rng.uniform(0.1, 0.35, cloud.sum())

    dates = pd.Timestamp(f"{base}-01-01") + pd.to_timedelta(doy - 1, unit="D")
    m = zeit.phenology(pd.Series(y, index=dates), curve="beck")   # one value per year, 2019-2021
    sm = zeit.smooth.apply_whittaker_filter(y[:, None, None], lmbd=15)[:, 0, 0]

    fig, ax = plt.subplots(figsize=(12, 4.0))
    x = base + t
    obs_scatter(ax, x, y)
    ax.plot(x, sm, color=BLUE, label="Whittaker-smoothed NDVI")
    labels = [("TRS5.sos", "SOS", AQUA), ("DER.pos", "Peak", VIOLET), ("TRS5.eos", "EOS", ORANGE)]
    for yi, yr in enumerate(m.year.values):
        for key, lab, col in labels:
            v = float(m[key].sel(year=yr))
            if np.isfinite(v) and v > 0:
                xv = yr + (v - 1) / 365.0
                ax.axvline(xv, color=col, linewidth=1.6, linestyle=(0, (4, 3)),
                           label=lab if yi == 0 else None)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("NDVI")
    ticks = [base + k * 0.5 for k in range(7)]
    ax.set_xticks(ticks, [("Jan " if k % 2 == 0 else "Jul ") + str(int(tk)) for k, tk in enumerate(ticks)])
    legend_above(ax, ncol=5)
    save(fig, "phenology_metrics")
    return m


def fig_twdtw():
    import pandas as pd

    import zeit
    from zeit._twdtw_api import match
    rng = np.random.default_rng(RNG_SEED + 6)
    pdates = np.arange(0, 365, 16)
    ft = pdates / 365

    def bump(center, width, lo, hi):
        return lo + (hi - lo) * np.exp(-0.5 * ((ft - center) / width) ** 2)

    patterns = {
        "Soybean": bump(0.12, 0.08, 0.25, 0.85),
        "Pasture": 0.5 + _harmonic(ft, 0.08, 0.1),
        "Forest": np.full_like(ft, 0.82) + _harmonic(ft, 0.02, 0.03),
    }
    # A soybean field planted ~3 weeks later than the reference pattern.
    query = np.interp(ft - 0.06, ft, patterns["Soybean"], period=1) + rng.normal(0, 0.03, len(ft))
    dates = pd.Timestamp("2021-01-01") + pd.to_timedelta(pdates, unit="D")
    result = zeit.twdtw(pd.Series(query, dates), {k: pd.Series(p, dates) for k, p in patterns.items()})
    dists = dict(zip(patterns, result.distances.values))
    path = match(result, query, dates, list(patterns).index("Soybean"))
    colors = {"Soybean": BLUE, "Pasture": ORANGE, "Forest": AQUA}

    fig = plt.figure(figsize=(13.5, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 0.8], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for k, p in patterns.items():
        ax.plot(pdates, p, color=colors[k], label=k)
    ax.set_title("Reference patterns")
    ax.set_ylabel("NDVI")
    ax.set_xlabel("Day of year")
    ax.legend(loc="lower right")
    ax.set_ylim(0.1, 1.0)

    ax = fig.add_subplot(gs[0, 1])
    offset = -0.0
    ax.plot(pdates, patterns["Soybean"], color=BLUE, label="Soybean pattern")
    ax.plot(pdates, query, color=INK2, linewidth=1.4, marker="o", markersize=4,
            markerfacecolor="white", label="Unknown pixel")
    for i, j in path[::1]:
        i, j = int(i), int(j)
        ax.plot([pdates[j], pdates[i]], [patterns["Soybean"][j], query[i]], color=AXIS,
                linewidth=0.8, zorder=0)
    ax.set_title("Time-weighted alignment")
    ax.set_xlabel("Day of year")
    ax.legend(loc="upper right")
    ax.set_ylim(0.1, 1.0)

    ax = fig.add_subplot(gs[0, 2])
    names = list(dists)
    vals = [dists[k] for k in names]
    ax.barh(names, vals, color=[colors[k] for k in names], height=0.36)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v:.2f}", va="center", color=INK2, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_title("TWDTW distance (lower = better)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(vals) * 1.35)
    save(fig, "twdtw_matching")
    return dists


def fig_smoothing():
    import pandas as pd

    import zeit
    rng = np.random.default_rng(RNG_SEED + 8)
    t = np.arange(0, 4, 1 / 23)
    truth = 0.55 + _harmonic(t, -0.2, 0.08)
    y = truth + rng.normal(0, 0.025, len(t))
    cloud = rng.random(len(t)) < 0.2
    y[cloud] -= rng.uniform(0.15, 0.4, cloud.sum())
    w = (~cloud).astype(float)
    series = pd.Series(y, pd.date_range("2020-01-01", periods=len(t), freq="16D"))
    sg = zeit.smooth(series, method="savgol", window=7, polyorder=2).to_numpy()
    wh = zeit.smooth(series, lmbda=10, weights=w).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = 2020 + t
    obs_scatter(ax, x[~cloud], y[~cloud], label="Clear observations")
    ax.scatter(x[cloud], y[cloud], s=30, marker="X", color=OBS, label="Cloud-contaminated")
    ax.plot(x, sg, color=ORANGE, label="Savitzky-Golay (unweighted)")
    ax.plot(x, wh, color=BLUE, label="Whittaker (QA-weighted)")
    ax.set_ylabel("NDVI")
    ax.set_ylim(0.0, 0.95)
    year_axis(ax)
    legend_above(ax, ncol=4)
    save(fig, "smoothing")


def fig_tempcnn():
    import torch
    from torch import nn
    from zeit.ai import TempCNN
    torch.manual_seed(RNG_SEED)
    rng = np.random.default_rng(RNG_SEED + 9)
    n_times, per_class = 23, 250
    t = np.arange(n_times) / n_times

    def make(kind, n):
        sh = rng.normal(0, 0.07, (n, 1))
        tt = t[None, :] - sh
        amp = rng.uniform(0.7, 1.15, (n, 1))
        if kind == 0:   # forest
            v = 0.8 + 0.03 * np.cos(2 * np.pi * tt)
        elif kind == 1:  # pasture
            v = 0.5 + 0.12 * np.cos(2 * np.pi * (tt - 0.1))
        elif kind == 2:  # single crop
            v = 0.25 + 0.6 * np.exp(-0.5 * ((tt - 0.25) / 0.08) ** 2)
        else:            # double crop: a weaker, variable second season
            second = rng.uniform(0.1, 0.45, (n, 1))
            v = (0.25 + 0.55 * np.exp(-0.5 * ((tt - 0.2) / 0.06) ** 2)
                 + second * np.exp(-0.5 * ((tt - 0.6) / 0.07) ** 2))
        v = v.mean(axis=1, keepdims=True) + amp * (v - v.mean(axis=1, keepdims=True))
        v = v + rng.normal(0, 0.07, v.shape)
        cloud = rng.random(v.shape) < 0.12
        return np.where(cloud, v - rng.uniform(0.1, 0.4, v.shape), v)

    names = ["Forest", "Pasture", "Single crop", "Double crop"]
    X = np.concatenate([make(k, per_class) for k in range(4)])[:, None, :].astype(np.float32)
    Y = np.repeat(np.arange(4), per_class)
    perm = rng.permutation(len(Y))
    X, Y = X[perm], Y[perm]
    ntr = int(0.7 * len(Y))
    Xtr, Ytr = torch.tensor(X[:ntr]), torch.tensor(Y[:ntr])
    Xte, Yte = torch.tensor(X[ntr:]), torch.tensor(Y[ntr:])

    model = TempCNN(in_channels=1, n_times=n_times, num_classes=4)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    hist_tr, hist_te = [], []
    for epoch in range(30):
        model.train()
        for i in range(0, ntr, 64):
            opt.zero_grad()
            loss = loss_fn(model(Xtr[i:i + 64]), Ytr[i:i + 64])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hist_tr.append((model(Xtr).argmax(1) == Ytr).float().mean().item())
            hist_te.append((model(Xte).argmax(1) == Yte).float().mean().item())
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    cm = np.zeros((4, 4), int)
    for a, b in zip(Yte.numpy(), pred):
        cm[a, b] += 1

    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.1, 1], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    cols = [AQUA, ORANGE, BLUE, VIOLET]
    for k in range(4):
        sel = X[Y == k][:25, 0]
        ax.plot(np.arange(n_times) * 16, sel.T, color=cols[k], alpha=0.12, linewidth=1)
        ax.plot(np.arange(n_times) * 16, sel.mean(0), color=cols[k], label=names[k])
    ax.set_title("Training samples")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("NDVI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)

    ax = fig.add_subplot(gs[0, 1])
    ep = np.arange(1, len(hist_tr) + 1)
    ax.plot(ep, hist_tr, color=BLUE, label="Train")
    ax.plot(ep, hist_te, color=ORANGE, label="Held-out")
    ax.set_ylim(0.2, 1.02)
    ax.legend(loc="lower right")
    ax.set_title("Overall accuracy per epoch")
    ax.set_xlabel("Epoch")

    ax = fig.add_subplot(gs[0, 2])
    ax.grid(False)
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap=LinearSegmentedColormap.from_list("b", ["#f5f8fd", "#104281"]),
              vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else INK2)
    short = ["For.", "Past.", "Single", "Double"]
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    ax.set_title(f"Held-out confusion (%)\nOverall accuracy {hist_te[-1] * 100:.1f}%", pad=10)
    save(fig, "tempcnn_training")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Accuracy & Area: Olofsson et al. (2014)'s example and a real stratified sample
# ---------------------------------------------------------------------------
OLOFSSON_NAMES = ["Deforestation", "Forest gain", "Stable forest", "Stable non-forest"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cfe0f5", "#86b2e8", "#2a78d6", "#123f7a"])


def _olofsson():
    """The map and the 640 labelled points of Olofsson et al. (2014), section 5, as in
    tests/test_accuracy.py: a 2500 x 4000 map of 30 m pixels and the paper's error matrix."""
    import json
    import xarray as xr
    from rasterio.transform import from_origin
    fixture = Path(__file__).resolve().parents[2] / "tests" / "data" / "accuracy_sits.json"
    case = json.loads(fixture.read_text(encoding="utf-8"))["olofsson2014"]
    names, matrix = case["labels"], np.array(case["matrix"])
    ny, nx = 2500, 4000
    values = np.repeat(np.arange(1, 5, dtype=np.uint8), case["area"]).reshape(ny, nx)
    m = xr.DataArray(values, dims=("y", "x"),
                     coords={"y": 9000000 - 15 - 30 * np.arange(ny), "x": 500000 + 15 + 30 * np.arange(nx)},
                     attrs={"flag_values": [1, 2, 3, 4], "flag_meanings": " ".join(names)})
    m = m.rio.write_crs(32722).rio.write_transform(from_origin(500000, 9000000, 30, 30))
    design = zeit.sampling_design(m, alloc={n: int(matrix[i].sum()) for i, n in enumerate(names)})
    pts = zeit.stratified_sample(m, design=design, seed=1)
    pts = pts.sort_values(["stratum", "row", "col"]).reset_index(drop=True)
    pts["ref"] = np.concatenate([np.repeat(names, matrix[i]) for i in range(4)])
    return m, pts, names


def fig_accuracy_sample(d):
    """Steps 1-2 on real data: a map of loss events and the stratified sample drawn on it."""
    import xarray as xr
    from rasterio.transform import from_origin
    s, y, valid, ev = d["stack"], d["years"], d["valid"], d["events"]
    r0, c0, n = 500, 700, 700
    keep = valid & (ev["yod"] >= 2015) & (ev["dsnr"] >= 3)     # loss in the last ten years
    change = _mmu(keep, 11)[r0:r0 + n, c0:c0 + n]
    ok = valid[r0:r0 + n, c0:c0 + n]
    codes = np.where(ok, change.astype(np.uint8) + 1, 0).astype(np.uint8)
    m = xr.DataArray(codes, dims=("y", "x"),
                     coords={"y": -np.arange(n) * 30.0 - 15, "x": np.arange(n) * 30.0 + 15},
                     attrs={"flag_values": [1, 2], "flag_meanings": "no_change change"})
    m = m.rio.write_crs(32720).rio.write_transform(from_origin(0, 0, 30, 30)).rio.write_nodata(0)
    ua = {"change": 0.7, "no_change": 0.95}
    design = zeit.sampling_design(m, expected_ua=ua, std_error=0.015)
    prop = zeit.sampling_design(m, expected_ua=ua, std_error=0.015, min_per_stratum=0).n.to_numpy()
    pts = zeit.stratified_sample(m, design=design, seed=3)
    print("  design:", design[["name", "weight", "n"]].to_dict("records"))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                                 gridspec_kw={"width_ratios": [1, 0.8], "wspace": 0.16})
    base = _masked(s[-1, r0:r0 + n, c0:c0 + n] / 10000, ok)
    ax.imshow(base, cmap="Greys", vmin=-0.6, vmax=2.6, interpolation="nearest")
    ax.imshow(np.where(change & ok, 1.0, np.nan), cmap=ListedColormap(["#e0603f"]), alpha=0.8,
              interpolation="nearest")
    for code, color, label in [(1, BLUE, "No change"), (2, INK, "Change")]:
        p = pts[pts.stratum == code]
        ax.scatter(p.col, p.row, s=34, color=color, edgecolor="white", linewidth=1.2, zorder=3,
                   label=f"{label} stratum: {len(p)} points")
    map_axes(ax, "Loss since 2015 mapped by LandTrendr (orange), and the sample")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, borderaxespad=0, handletextpad=0.2)

    w = design.weight.to_numpy()
    total = int(design.n.sum())
    names = ["No change", "Change"]
    yy = np.arange(2)
    bx.barh(yy - 0.2, w * 100, height=0.36, color="#d6d4cc", label="Share of the map's area")
    bx.barh(yy + 0.2, design.n / total * 100, height=0.36, color=BLUE, label="Share of the sample")
    for i in range(2):
        ha = design.pixels.iloc[i] * 0.09
        bx.text(w[i] * 100 + 1.5, i - 0.2, f"{w[i] * 100:.1f}%  ·  {ha:,.0f} ha", va="center",
                fontsize=9.5, color=INK2)
        bx.text(design.n.iloc[i] / total * 100 + 1.5, i + 0.2,
                f"{int(design.n.iloc[i])} points  (proportional: {int(prop[i])})", va="center",
                fontsize=9.5, color=INK)
    bx.set_yticks(yy, names)
    bx.set_xlim(0, 150)
    bx.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    bx.grid(axis="y", visible=False)
    bx.invert_yaxis()
    bx.set_title(f"Where the {total} points go")
    bx.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=1, borderaxespad=0)
    bx.set_box_aspect(1.0)
    save(fig, "accuracy_sample", photo=True)


def fig_accuracy_matrix():
    """The error matrix of Olofsson et al. (2014): sample counts and estimated area proportions."""
    m, pts, names = _olofsson()
    acc = zeit.accuracy(m, pts)
    counts = acc.counts.to_numpy()
    prop = acc.confusion.to_numpy() * 100
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"wspace": 0.3})
    short = ["Deforest.", "Gain", "Forest", "Non-forest"]
    panels = [(counts, "Sample counts $n_{ij}$", "{:.0f}", 60.0),
              (prop, r"Estimated share of the area $\hat p_{ij}$ (%)", "{:.2f}", 8.0)]
    for ax, (mat, title, fmt, vmax) in zip(axes, panels):
        ax.imshow(np.minimum(mat, vmax), cmap=SEQ_BLUE, vmin=0, vmax=vmax * 1.08)
        for i in range(4):
            for j in range(4):
                v = mat[i, j]
                dark = min(v, vmax) > vmax * 0.5
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=10.5,
                        color="white" if dark else INK, fontweight="semibold" if i == j else "normal")
        ax.set_xticks(range(4), short)
        ax.set_yticks(range(4), short)
        ax.set_xlabel("Reference class", color=INK2, labelpad=8)
        ax.set_ylabel("Map class", color=INK2)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, pad=44)
    n_i = counts.sum(axis=1)
    ax = axes[0]
    for i in range(4):
        ax.text(3.62, i, f"{n_i[i]}", va="center", fontsize=9.5, color=INK2)
    ax.text(3.62, -0.72, "$n_i$", va="center", fontsize=10, color=MUTED)
    ax.set_xlim(-0.5, 4.1)
    ax = axes[1]
    for i, c in enumerate(acc.users.index):
        ax.text(3.62, i, f"UA {acc.users.estimate[c]:.2f}", va="center", fontsize=9.5, color=INK2)
        ax.text(i, 3.85, f"PA {acc.producers.estimate[c]:.2f}", ha="center", va="center", fontsize=9.5,
                color=INK2)
    ax.text(3.62, 3.85, f"OA {acc.overall.estimate:.3f}", va="center", fontsize=9.5, color=INK,
            fontweight="semibold")
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.15, -0.5)
    axes[0].set_ylim(4.15, -0.5)
    save(fig, "accuracy_matrix")


def fig_accuracy_area():
    """Mapped area versus the error-adjusted estimate and its 95% interval, per class."""
    from matplotlib.ticker import MaxNLocator
    m, pts, names = _olofsson()
    acc = zeit.accuracy(m, pts)
    a = acc.area / 1000
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.9), gridspec_kw={"wspace": 0.3})
    for ax, c, label in zip(axes, a.index, OLOFSSON_NAMES):
        row = a.loc[c]
        ax.plot([row.ci_low, row.ci_high], [0, 0], color=BLUE, linewidth=7, alpha=0.22,
                solid_capstyle="round")
        ax.scatter([row.estimate], [0], s=70, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3,
                   label="Error-adjusted estimate, 95% CI")
        ax.scatter([row.mapped], [0], s=60, marker="D", facecolor="white", edgecolor=ORANGE, linewidth=2,
                   zorder=4, label="Mapped (pixel count)")
        lo, hi = min(row.ci_low, row.mapped), max(row.ci_high, row.mapped)
        ax.set_xlim(lo - 0.2 * (hi - lo), hi + 0.2 * (hi - lo))
        ax.set_ylim(-1, 1.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        ax.set_title(label)
        ax.text(row.estimate, 0.38, f"{row.estimate:,.1f} ± {row.ci:,.1f}", ha="center", fontsize=9.5,
                color=INK)
        ax.text(row.mapped, -0.6, f"{row.mapped:,.1f}", ha="center", fontsize=9.5, color=INK2)
        ax.set_xlabel("thousand ha", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.22), ncol=2, borderaxespad=0)
    save(fig, "accuracy_area")


def fig_accuracy_sample_size():
    """Sample size versus the target standard error of the overall accuracy (eq. 13)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    m, _, names = _olofsson()
    paper = dict(zip(names, [0.7, 0.6, 0.9, 0.95]))
    ses = np.linspace(0.005, 0.03, 60)
    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    for ua, color, label in [(paper, BLUE, "Expected user's accuracies of the paper: 0.7, 0.6, 0.9, 0.95"),
                             (0.75, ORANGE, "0.75 for every stratum (the default)")]:
        n = [zeit.sampling_design(m, expected_ua=ua, std_error=s, min_per_stratum=0).attrs["n"] for s in ses]
        ax.plot(ses, n, color=color, label=label)
    n01 = zeit.sampling_design(m, expected_ua=paper, std_error=0.01, min_per_stratum=0).attrs["n"]
    ax.scatter([0.01], [n01], s=60, color=BLUE, edgecolor="white", linewidth=1.5, zorder=4)
    ax.annotate(f"{n01} points for a standard error of 0.01", (0.01, n01), xytext=(12, 6),
                textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([100, 200, 500, 1000, 2000, 5000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Target standard error of the overall accuracy (std_error)")
    ax.set_ylabel("Sample size n")
    ax.set_title("Halving the standard error takes about four times the points")
    ax.legend(loc="upper right")
    save(fig, "accuracy_sample_size")


# ---------------------------------------------------------------------------
# Embeddings: AlphaEarth and TESSERA over a 5 x 5 km box of the Rondonia scene
# ---------------------------------------------------------------------------
# The box sits inside the LT-GEE NDVI stack, where it has the most loss since 2016, so the
# embeddings can be compared with LandTrendr. The cubes are downloaded once (about 100 s);
# point ZEIT_DOCS_EMBEDDINGS at a folder to keep them (alphaearth.nc, tessera.nc).
EMB_BOX = (-63.136, -10.133, -63.086, -10.083)
EMB_THRESHOLD = {"alphaearth": 0.2, "tessera": 0.3}
EMB_NAMES = {"alphaearth": "AlphaEarth (64 dims)", "tessera": "TESSERA (128 dims)"}
_EMB = {}


def embeddings():
    if _EMB:
        return _EMB
    import xarray as xr
    folder = Path(os.environ.get("ZEIT_DOCS_EMBEDDINGS", OUT.parents[2] / "build" / "docs_embeddings"))
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("alphaearth", "tessera"):
        path = folder / f"{name}.nc"
        if not path.exists():
            emb = zeit.load_embeddings(EMB_BOX, source=name, years=range(2017, 2026), chunks=None)
            emb.attrs = {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in emb.attrs.items()}
            emb.to_netcdf(path)
        ds = xr.open_dataset(path)
        da = ds["embeddings"].load()
        da.attrs.update(ds.attrs)
        _EMB[name] = da.rio.write_crs("EPSG:32720")
    return _EMB


def _lt_on_embedding_grid(like):
    """LandTrendr's year of greatest NDVI loss on the embeddings' 10 m grid (nearest)."""
    from rasterio.enums import Resampling
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    ndvi = zeit.load_raster(path, start_year=1985)
    x0, y0, x1, y1 = EMB_BOX
    ndvi = ndvi.rio.clip_box(x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01).load()
    ndvi = ndvi.where(ndvi != 0)
    ev = zeit.extract_events(zeit.landtrendr(ndvi), min_magnitude=2000)
    yod = ev.yod.where(ev.dsnr >= 3, 0).astype(np.float32).rio.write_nodata(np.nan, encoded=False)
    return yod.rio.reproject_match(like, resampling=Resampling.nearest)


def fig_embeddings_pca():
    """The first three principal components as RGB, the same fit for every year."""
    from zeit._embedding_tools import pca_rgb
    cubes = embeddings()
    years = [2018, 2021, 2025]          # TESSERA has no 2017 embedding for most of this box
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.3), gridspec_kw={"wspace": 0.04, "hspace": 0.1})
    for row, (name, emb) in zip(axes, cubes.items()):
        rgb = pca_rgb(emb).values
        for ax, yr in zip(row, years):
            img = np.moveaxis(rgb[yr - 2017], 0, -1)
            img = np.where(np.isfinite(img), img, 1.0)
            ax.imshow(img, interpolation="nearest")
            map_axes(ax, f"{EMB_NAMES[name].split(' (')[0]}, {yr}")
    save(fig, "embeddings_pca", photo=True)


def _forest_points(cubes, n=25, seed=RNG_SEED):
    """n pixels that stayed forest through 2017-2024 in the NDVI stack and moved little in
    the embeddings: the 'samples' a user would click."""
    import geopandas as gpd
    aef = cubes["alphaearth"]
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    from rasterio.enums import Resampling
    ndvi = zeit.load_raster(path, start_year=1985)
    x0, y0, x1, y1 = EMB_BOX
    ndvi = ndvi.rio.clip_box(x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01).load()
    low = ndvi.sel(time=slice("2010", "2024")).min("time").astype(np.float32)
    low = low.rio.reproject_match(aef.isel(time=0, band=0), resampling=Resampling.nearest).values
    still = zeit.embedding_change(aef).distance.max("time").values < 0.08
    rr, cc = np.nonzero((low > 8000) & still)
    pick = np.random.default_rng(seed).choice(len(rr), n, replace=False)
    xs, ys = aef.x.values[cc[pick]], aef.y.values[rr[pick]]
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=aef.rio.crs), rr[pick], cc[pick]


def fig_embeddings_similarity(d=None):
    """Cosine similarity to the mean embedding of 25 forest samples, in 2018 and 2025."""
    cubes = embeddings()
    aef = cubes["alphaearth"]
    pts, rr, cc = _forest_points(cubes)
    sim = zeit.similarity(aef, pts, year=2018)
    fig = plt.figure(figsize=(13.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.2)
    cmap = LinearSegmentedColormap.from_list("sim", ["#f4f1e4", "#c9dfa4", "#7fb069", "#3a7d44", "#1b4d2b"])
    for i, yr in enumerate([2018, 2025]):
        ax = fig.add_subplot(gs[i])
        im = ax.imshow(sim.sel(time=str(yr)).squeeze().values, cmap=cmap, vmin=0.2, vmax=1.0,
                       interpolation="nearest")
        if yr == 2018:
            ax.scatter(cc, rr, s=36, facecolor="none", edgecolor=ORANGE, linewidth=1.8, label="The 25 samples")
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.01), borderaxespad=0)
        map_axes(ax, f"Similarity to the forest samples, {yr}")
    cb = fig.colorbar(im, ax=fig.axes[:2], orientation="horizontal", fraction=0.05, pad=0.1, aspect=40)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=MUTED, labelcolor=INK2, length=0)
    cb.set_label("Cosine similarity (AlphaEarth)", color=INK2, fontsize=9.5)
    ax = fig.add_subplot(gs[2])
    lo = float(np.floor(np.nanmin(sim.values) * 20) / 20)
    bins = np.linspace(lo, 1.0, int(round((1 - lo) * 100)) + 1)
    for yr, color in [(2018, BLUE), (2025, ORANGE)]:
        v = sim.sel(time=str(yr)).values.ravel()
        h, _ = np.histogram(v[np.isfinite(v)], bins=bins)
        ax.stairs(h / h.sum() * 100, bins, color=color, linewidth=2, label=str(yr))
        print(f"  {yr}: share >= 0.8: {(v >= 0.8).mean():.1%}")
    ax.set_xlabel("Cosine similarity to the forest samples (AlphaEarth)")
    ax.set_ylabel("% of the box")
    ax.set_title("The forest peak shrinks as it is cleared")
    ax.legend(loc="upper center")
    ax.set_box_aspect(0.9)
    save(fig, "embeddings_similarity", photo=True)


def fig_embeddings_change(d=None):
    """Year of the largest change: AlphaEarth, TESSERA and LandTrendr on NDVI, same grid."""
    cubes = embeddings()
    like = cubes["alphaearth"].isel(time=0, band=0)
    maps = {}
    for name, emb in cubes.items():
        ev = zeit.extract_events(zeit.embedding_change(emb), min_magnitude=EMB_THRESHOLD[name])
        maps[name] = ev.yod.where(ev.yod > 0).values
    lt = _lt_on_embedding_grid(like).values
    maps["landtrendr"] = np.where((lt >= 2016) & (lt <= 2024), lt, np.nan)
    titles = {"alphaearth": "AlphaEarth: 1 − cos > 0.2", "tessera": "TESSERA: 1 − cos > 0.3",
              "landtrendr": "LandTrendr on Landsat NDVI"}
    cmap = LinearSegmentedColormap.from_list("yrs", ["#fbe3a1", "#f5a45d", "#e0603f", "#a8325e", "#5b1f6b",
                                                     "#231942"], N=9)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), gridspec_kw={"wspace": 0.04})
    base = cubes["alphaearth"].isel(time=0)
    for ax, (name, yod) in zip(axes, maps.items()):
        ax.imshow(np.ones(like.shape), cmap="Greys", vmin=0, vmax=12)
        im = ax.imshow(yod + 1, cmap=cmap, vmin=2016.5, vmax=2025.5, interpolation="nearest")
        map_axes(ax, titles[name])
    cb = colorbar(fig, im, axes.tolist(), "First year that shows the change", ticks=list(range(2017, 2026)))
    cb.ax.set_xticklabels([str(y) for y in range(2017, 2026)])
    a, t, l = (np.isfinite(maps[k]) for k in ("alphaearth", "tessera", "landtrendr"))
    print(f"  pixels: aef {a.sum()}, tessera {t.sum()}, lt {l.sum()}; "
          f"aef&lt {(a & l).sum()} within1 {((np.abs(maps['alphaearth'] - maps['landtrendr']) <= 1) & a & l).sum()}; "
          f"tes&lt {(t & l).sum()} within1 {((np.abs(maps['tessera'] - maps['landtrendr']) <= 1) & t & l).sum()}; "
          f"aef&tes {(a & t).sum()} within1 {((np.abs(maps['alphaearth'] - maps['tessera']) <= 1) & a & t).sum()}")
    save(fig, "embeddings_change", photo=True)


def fig_embeddings_distance():
    """How far pixels move in a year: the distribution, and one clearing."""
    cubes = embeddings()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 4.0), gridspec_kw={"wspace": 0.22})
    bins = np.logspace(np.log10(0.005), np.log10(1.2), 70)
    changes = {}
    for (name, emb), color in zip(cubes.items(), [BLUE, ORANGE]):
        ch = zeit.embedding_change(emb)
        changes[name] = ch
        d = ch.distance.values.ravel()
        d = d[np.isfinite(d)]
        h, _ = np.histogram(d, bins=bins)
        ax.stairs(h / h.sum() * 100, bins, color=color, linewidth=2, label=EMB_NAMES[name].split(" (")[0])
        med = np.median(d)
        ax.axvline(EMB_THRESHOLD[name], color=color, linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(EMB_THRESHOLD[name] * 1.06, ax.get_ylim()[1] * 0.0 + (5.2 if name == "alphaearth" else 4.4),
                f"threshold {EMB_THRESHOLD[name]}", fontsize=9, color=INK2)
        print(f"  {name}: median {med:.3f}, share above threshold {(d > EMB_THRESHOLD[name]).mean():.3%}")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"])
    ax.set_xlabel("Distance to the year before (1 − cos)")
    ax.set_ylabel("% of pixel-years")
    ax.set_title("Most pixels barely move; each product has its own scale")
    ax.legend(loc="upper left")
    # one pixel cleared in the middle of the record
    a = changes["alphaearth"].distance.values
    t = changes["tessera"].distance.values
    score = np.nan_to_num(a[3]) * 4 + np.nan_to_num(t[3]) - np.nan_to_num(np.delete(a, 3, 0)).max(0) * 4
    r, c = np.unravel_index(np.argmax(np.where(np.isfinite(t).all(0), score, -9)), score.shape)
    years = np.arange(2018, 2026)
    for d, color, name in [(a, BLUE, "AlphaEarth"), (t, ORANGE, "TESSERA")]:
        bx.plot(years, d[:, r, c], color=color, marker="o", markersize=5, label=name)
    bx.set_xlabel("Year (distance to the year before)")
    bx.set_ylabel("1 − cos")
    bx.set_title("One pixel cleared between 2020 and 2021")
    year_axis(bx, 1)
    bx.legend(loc="upper right")
    save(fig, "embeddings_distance")


def fig_embeddings_som():
    """Unsupervised: six SOM clusters of the 2025 embeddings, next to their colours."""
    from zeit._embedding_tools import pca_rgb
    aef = embeddings()["alphaearth"]
    year = aef.sel(time="2025").squeeze("time", drop=True)
    clusters = zeit.som(year, x=2, y=3, sample=20_000, seed=0) if "seed" in zeit.som.__code__.co_varnames \
        else zeit.som(year, x=2, y=3, sample=20_000)
    lab = np.asarray(clusters.label.values, dtype=float)
    rgb = np.moveaxis(pca_rgb(aef).sel(time="2025").squeeze().values, 0, -1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.9), gridspec_kw={"wspace": 0.04})
    ax.imshow(rgb, interpolation="nearest")
    map_axes(ax, "AlphaEarth 2025, three principal components")
    cmap = ListedColormap(["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e8b422", "#8a8780"])
    bx.imshow(lab, cmap=cmap, vmin=-0.5, vmax=5.5, interpolation="nearest")
    map_axes(bx, "zeit.som: six clusters, no labels")
    save(fig, "embeddings_som", photo=True)


FIGURES = {
    "hero_stack": (fig_hero_stack, True),
    "landtrendr_maps": (fig_landtrendr_maps, True),
    "landtrendr_pixels": (fig_landtrendr_pixels, True),
    "concept_pixel_time_series": (fig_pixel_time_series, True),
    "mann_kendall_map": (fig_mann_kendall, True),
    "snic_segments": (fig_snic, True),
    "som_clusters": (fig_som, True),
    "quickstart_result": (fig_quickstart, False),
    "ccdc_fit": (fig_ccdc, False),
    "tmask_flags": (fig_tmask, False),
    "bfast_monitor": (fig_bfast_monitor, False),
    "bfast_lite": (fig_bfast_lite, False),
    "bfast_decomposition": (fig_bfast_classic, False),
    "phenology_metrics": (fig_phenology, False),
    "twdtw_matching": (fig_twdtw, False),
    "smoothing": (fig_smoothing, False),
    "tempcnn_training": (fig_tempcnn, False),
    "accuracy_sample": (fig_accuracy_sample, True),
    "accuracy_matrix": (fig_accuracy_matrix, False),
    "accuracy_area": (fig_accuracy_area, False),
    "accuracy_sample_size": (fig_accuracy_sample_size, False),
    "embeddings_pca": (fig_embeddings_pca, False),
    "embeddings_similarity": (fig_embeddings_similarity, True),
    "embeddings_change": (fig_embeddings_change, True),
    "embeddings_distance": (fig_embeddings_distance, False),
    "embeddings_som": (fig_embeddings_som, False),
}

# ---------------------------------------------------------------------------
# Gallery thumbnails: a 16:9 crop of the most telling part of each figure.
THUMBS = {
    "landtrendr": ("landtrendr_maps.webp", (0.655, 0.0, 1.0, 0.8)),
    "ccdc": ("ccdc_fit.png", (0.0, 0.0, 1.0, 1.0)),
    "bfast_monitor": ("bfast_monitor.png", (0.0, 0.0, 1.0, 1.0)),
    "mann_kendall": ("mann_kendall_map.webp", (0.5, 0.0, 1.0, 0.86)),
    "phenology": ("phenology_metrics.png", (0.0, 0.0, 1.0, 1.0)),
    "twdtw": ("twdtw_matching.png", (0.0, 0.0, 0.7, 1.0)),
    "snic": ("snic_segments.webp", (0.33, 0.0, 0.67, 1.0)),
    "som": ("som_clusters.webp", (0.0, 0.0, 1.0, 1.0)),
    "tempcnn": ("tempcnn_training.png", (0.0, 0.0, 1.0, 1.0)),
    "tmask": ("tmask_flags.png", (0.0, 0.0, 1.0, 1.0)),
    "bfast": ("bfast_decomposition.png", (0.0, 0.0, 1.0, 0.62)),
    "bfast_lite": ("bfast_lite.png", (0.0, 0.0, 1.0, 1.0)),
    "smoothing": ("smoothing.png", (0.0, 0.0, 1.0, 1.0)),
    "landtrendr_pixels": ("landtrendr_pixels.png", (0.0, 0.0, 0.36, 1.0)),
}


def make_thumbs():
    from PIL import Image
    tw, th = 800, 450
    for name, (src, (x0, y0, x1, y1)) in THUMBS.items():
        path = OUT / src
        if not path.exists():
            continue
        im = Image.open(path).convert("RGB")
        w, h = im.size
        im = im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        im.thumbnail((tw - 40, th - 40), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), "white")
        canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
        (OUT / "thumbs").mkdir(exist_ok=True)
        canvas.save(OUT / "thumbs" / f"{name}.webp", quality=85, method=6)
    print("  wrote gallery thumbnails")


def main(names):
    setup_style()
    names = names or list(FIGURES)
    for name in names:
        fn, needs_data = FIGURES[name]
        print(name)
        if needs_data:
            d = rondonia()
            if d is None:
                print("  skipped (set ZEIT_DOCS_RONDONIA to the annual NDVI GeoTIFF)")
                continue
            fn(d)
        else:
            fn()
    make_thumbs()


if __name__ == "__main__":
    main(sys.argv[1:])
