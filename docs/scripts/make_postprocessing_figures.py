"""Generate the figures of the post-processing tutorial (docs/tutorials/postprocessing.md).

Every figure is produced by running Zeit (``zeit.train_classifier``, ``zeit.classify``,
``zeit.smooth_probs``, ``zeit.mmu_sieve``, ``zeit.apply_majority_filter``,
``zeit.desawtooth_classes``, ``zeit.aoa``), so the numbers quoted in the tutorial can be
checked by running this script. From the repository root:

    python docs/scripts/make_postprocessing_figures.py              # every figure
    python docs/scripts/make_postprocessing_figures.py aoa          # a subset

Inputs:

* **Real data** (``smoothing``, ``smoothness``, ``aoa``): the annual NDVI stack of
  Rondonia used by ``make_figures.py`` (1985-2024, NDVI x 10000, 0 = no data), from the
  ``ZEIT_DOCS_RONDONIA`` environment variable; these figures are skipped when it is not set.
  A 500 x 500 pixel window is labelled from its 40 years by LandTrendr (forest; cleared before
  2000, in 2000-2009, since 2010: the year of the greatest loss), 40 random points per class
  train a random forest on the NDVI of eight years, and the forest classifies the window (its
  probabilities are what is smoothed) and, for the area of applicability, the whole scene.
* **Synthetic series** (``desawtooth``): yearly class series drawn here with a fixed seed.

The style, the ``save`` helper and the thumbnail follow ``make_figures.py``.
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402

OUT = mf.OUT
SEED = 7
YEARS = np.arange(1985, 2025)
FEATURE_YEARS = [1990, 1995, 2000, 2005, 2010, 2015, 2020, 2024]
WINDOW = (450, 450, 500)                     # row, column and side of the classified window
CROP = (150, 150, 250)                       # the part of the window shown in the smoothing figure
CLASSES = ["Forest", "Cleared before 2000", "Cleared 2000-2009", "Cleared since 2010"]
CLASS_COLORS = ["#2f6b3a", "#e6d9ae", "#eb6834", "#7a2d5e"]
CLASS_CMAP = ListedColormap(["#ffffff"] + CLASS_COLORS)
DI_CMAP = LinearSegmentedColormap.from_list(
    "di", ["#f4f1e4", "#c9d6e8", "#86b2e8", "#2a78d6", "#123f7a", "#0b1f3d"])

_SCENE = {}


def scene():
    """The stack, the LandTrendr labels of the window, the forest and its classification (cached)."""
    if _SCENE:
        return _SCENE
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    if not path or not os.path.exists(path):
        return None
    import geopandas as gpd

    stack = zeit.load_raster(path).assign_coords(time=YEARS)
    r0, c0, n = WINDOW
    # the labels: LandTrendr's year of greatest loss over the 40 years
    win_raw = stack.isel(y=slice(r0, r0 + n), x=slice(c0, c0 + n))
    a = win_raw.values.astype(np.float32)
    lt = zeit.landtrendr(a, years=YEARS, max_segments=6, nodata=0)
    ev = zeit.extract_events(lt, event_type="loss", sort_by="greatest", min_magnitude=2000)
    yod = np.asarray(ev.yod.values, dtype=float)
    yod[yod == 0] = np.nan
    recent = a[-3:].mean(axis=0) / 10000
    ref = np.zeros((n, n), np.uint8)
    ref[np.isnan(yod) & (recent > 0.72)] = 1
    ref[(np.isnan(yod) & (recent <= 0.72)) | (yod < 2000)] = 2
    ref[(yod >= 2000) & (yod < 2010)] = 3
    ref[yod >= 2010] = 4
    ref[(a == 0).any(axis=0)] = 0

    # the predictors: NDVI of eight years, missing where the stack has 0
    sel = stack.sel(time=FEATURE_YEARS)
    feats = (sel.astype("float32") / 10000).where(sel != 0).rio.write_crs(stack.rio.crs)
    win = feats.isel(y=slice(r0, r0 + n), x=slice(c0, c0 + n))

    rng = np.random.default_rng(SEED)
    rows, cols, labels = [], [], []
    for k, name in enumerate(CLASSES, start=1):
        rr, cc = np.nonzero(ref == k)
        pick = rng.choice(len(rr), 40, replace=False)
        rows += list(rr[pick])
        cols += list(cc[pick])
        labels += [name] * 40
    points = gpd.GeoDataFrame({"class": labels}, crs=stack.rio.crs,
                              geometry=gpd.points_from_xy(win.x.values[cols], win.y.values[rows]))
    rf = zeit.train_classifier(win, points, label="class")
    land_cover = zeit.classify(win, rf, probability=True)
    # classify numbers the classes in the forest's (alphabetical) order: to CLASSES' codes
    code = np.array([0] + [CLASSES.index(c) + 1 for c in rf.classes_], dtype=np.uint8)
    _SCENE.update(stack=stack, feats=feats, win=win, ref=ref, rf=rf, land_cover=land_cover, code=code,
                  rows=np.array(rows), cols=np.array(cols), labels=np.array(labels))
    return _SCENE


def labels_of(d, probs):
    """The class (CLASSES' codes, 0 = missing) of smoothed probabilities."""
    lab = (probs.fillna(-1).argmax("class") + 1).where(d["land_cover"].label > 0, 0)
    return d["code"][lab.values]


def agreement(d, lab):
    ok = d["ref"] > 0
    return float((lab[ok] == d["ref"][ok]).mean())


def patches(lab):
    from scipy import ndimage
    return int(sum(ndimage.label(lab == k, structure=np.ones((3, 3)))[1] for k in range(1, 5)))


def fig_smoothing(d):
    """The per-pixel classes, smoothed a little and as sits does, then sieved."""
    lc = d["land_cover"]
    raw = d["code"][lc.label.values]
    light = labels_of(d, zeit.smooth_probs(lc.probability, window_size=9, smoothness=2))
    strong = labels_of(d, zeit.smooth_probs(lc.probability, window_size=9, smoothness=20))
    sieved = np.asarray(zeit.mmu_sieve(light, min_size=11, nodata=0))
    majority = np.asarray(zeit.apply_majority_filter(raw, size=3, nodata=0))
    for name, lab in [("per pixel", raw), ("majority 3x3", majority), ("smoothness 2", light),
                      ("smoothness 2 + sieve 11", sieved), ("smoothness 20", strong)]:
        print(f"  {name:<24} agreement with LandTrendr {100 * agreement(d, lab):.1f}%, "
              f"{patches(lab):,} patches")
    r, c, n = CROP
    panels = [(raw, "Random forest, per pixel"),
              (light, "smooth_probs, smoothness 2"),
              (sieved, "… then mmu_sieve, 11 pixels"),
              (strong, "smooth_probs, smoothness 20")]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.25), gridspec_kw={"wspace": 0.05})
    for ax, (lab, title) in zip(axes, panels):
        ax.imshow(lab[r:r + n, c:c + n], cmap=CLASS_CMAP, vmin=-0.5, vmax=4.5, interpolation="nearest")
        mf.map_axes(ax, title)
    handles = [Patch(color=col, label=name) for name, col in zip(CLASSES, CLASS_COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False)
    mf.save(fig, "postproc_smoothing", photo=True)


def fig_smoothness(d):
    """Agreement with the LandTrendr labels and number of patches against smoothness."""
    lc = d["land_cover"]
    raw = d["code"][lc.label.values]
    values = [0.5, 1, 2, 5, 10, 20, 50]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.28})
    for window, color in [(5, mf.AQUA), (9, mf.BLUE)]:
        agr, pat = [], []
        for s in values:
            lab = labels_of(d, zeit.smooth_probs(lc.probability, window_size=window, smoothness=s))
            agr.append(100 * agreement(d, lab))
            pat.append(patches(lab))
        print(f"  window {window}: agreement {np.round(agr, 1).tolist()}, patches {pat}")
        ax.plot(values, agr, "o-", color=color, label=f"window_size={window}")
        bx.plot(values, pat, "o-", color=color, label=f"window_size={window}")
    a0, p0 = 100 * agreement(d, raw), patches(raw)
    for axis, v, text in [(ax, a0, f"per pixel: {a0:.1f}%"), (bx, p0, f"per pixel: {p0:,}")]:
        axis.axhline(v, color=mf.INK2, lw=1, ls=(0, (4, 3)))
        axis.text(values[-1], v, text, ha="right", va="bottom", color=mf.INK2, fontsize=9)
    for axis in (ax, bx):
        axis.set_xscale("log")
        axis.set_xticks(values, [f"{v:g}" for v in values])
        axis.minorticks_off()
        axis.set_xlabel("smoothness (sits' default: 20)")
    ax.set_ylabel("agreement with LandTrendr's labels (%)")
    ax.set_title("A little smoothing agrees better, a lot less")
    ax.legend(loc="lower left")
    bx.set_yscale("log")
    bx.set_ylabel("patches (8-connected)")
    bx.set_title("Each step removes patches")
    mf.save(fig, "postproc_smoothness")


def fig_desawtooth():
    """Yearly class series before and after desawtooth_classes."""
    rows = ["FFFFFFPFFFFFFFFF",     # a year of pasture in the forest
            "FFFFFPPPPPFPPPPP",     # cleared, then a year of forest in the pasture
            "FFFFFFFPPFFFFFFF",     # two years of pasture
            "FFFFFFPPPPPPPPPP",     # cleared for good
            "PPPCPPPPPFFFFFFF",     # a year of crop, then regrowth
            "FFFFPFPFFFFFFFFF"]     # alternating years
    series = np.array([[" FPC".index(ch) for ch in r] for r in rows], dtype=np.uint8)
    nyear = series.shape[1]
    one = zeit.desawtooth_classes(series.T, max_length=1).T
    two = zeit.desawtooth_classes(series.T, max_length=2).T
    again = zeit.desawtooth_classes(one.T, max_length=1).T
    for r, a, b, c in zip(rows, one, two, again):
        print("  ", r, *("".join(" FPC"[v] for v in x) for x in (a, b, c)))

    cmap = ListedColormap([CLASS_COLORS[0], CLASS_COLORS[1], "#e8b422"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), gridspec_kw={"wspace": 0.06})
    years = 2009 + np.arange(nyear)
    for ax, s, title in [(axes[0], series, "Yearly classes"),
                         (axes[1], one, "desawtooth_classes, max_length=1"),
                         (axes[2], two, "max_length=2")]:
        ax.imshow(s, cmap=cmap, vmin=0.5, vmax=3.5, aspect="auto", interpolation="nearest")
        for i in range(1, len(rows)):
            ax.axhline(i - 0.5, color="white", lw=4)
        changed = np.argwhere(s != series)
        for i, j in changed:
            ax.add_patch(Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, fill=False, ec=mf.INK, lw=1.4))
        ax.set_xticks(np.arange(0, nyear, 5), years[::5])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title)
    axes[0].set_ylabel("six pixels")
    handles = [Patch(color=c, label=n) for n, c in zip(["Forest", "Pasture", "Crop"],
                                                       [CLASS_COLORS[0], CLASS_COLORS[1], "#e8b422"])]
    handles.append(Patch(fill=False, ec=mf.INK, lw=1.4, label="changed"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), frameon=False)
    mf.save(fig, "postproc_desawtooth")


def fig_aoa(d):
    """The dissimilarity index and the area of applicability of the forest over the whole scene."""
    feats, rf = d["feats"], d["rf"]
    res = zeit.aoa(feats, model=rf)
    di, inside = res.DI.values, res.AOA.values
    valid = inside != 255
    out = (inside == 0).sum() / valid.sum()
    print(f"  threshold {res.attrs['threshold']:.3f}, mean distance {res.attrs['mean_distance']:.3f}, "
          f"outside {100 * out:.1f}% of {valid.sum():,} pixels")
    r0, c0, n = WINDOW
    ok = valid[r0:r0 + n, c0:c0 + n]
    print(f"  outside within the training window: {100 * (inside[r0:r0 + n, c0:c0 + n] == 0).sum() / ok.sum():.1f}%")
    weights = dict(zip(rf.zeit_features_, np.round(rf.feature_importances_, 3)))
    print(f"  weights (the forest's importance): {weights}")
    with_lpd = zeit.aoa(feats.isel(y=slice(r0, r0 + n), x=slice(c0, c0 + n)), model=rf, lpd=True)
    lpd = with_lpd.LPD.values
    print(f"  LPD in the window: median {np.nanmedian(lpd):.0f}, max {np.nanmax(lpd):.0f} training points")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 6.9), gridspec_kw={"wspace": 0.05})
    vmax = 2 * res.attrs["threshold"]
    im = ax.imshow(np.where(valid, di, np.nan), cmap=DI_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    mf.map_axes(ax, "Dissimilarity index")
    cb = mf.colorbar(fig, im, ax, "distance to the nearest training point / their mean distance",
                     ticks=[0, res.attrs["threshold"], vmax], fmt="%.2f")
    cb.ax.axvline(res.attrs["threshold"], color=mf.ORANGE, lw=2)
    base = np.where(valid, inside.astype(float), np.nan)
    bx.imshow(base, cmap=ListedColormap([mf.ORANGE, "#e9e7df"]), vmin=-0.5, vmax=1.5, interpolation="nearest")
    mf.map_axes(bx, f"Area of applicability: {100 * out:.1f}% outside")
    for axis in (ax, bx):
        axis.add_patch(Rectangle((c0, r0), n, n, fill=False, ec=mf.INK, lw=1.2, ls=(0, (4, 2))))
        axis.scatter(c0 + d["cols"], r0 + d["rows"], s=5, color=mf.INK, linewidths=0, zorder=3)
    handles = [Patch(color="#e9e7df", label="inside"), Patch(color=mf.ORANGE, label="outside"),
               plt.Line2D([], [], marker="o", ls="", color=mf.INK, ms=4, label="training points (160)"),
               Patch(fill=False, ec=mf.INK, ls=(0, (4, 2)), label="window classified")]
    bx.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, borderaxespad=0)
    mf.save(fig, "postproc_aoa", photo=True)

    # the folds: training points in clusters, with and without their clusters as folds
    import geopandas as gpd
    rng = np.random.default_rng(3)
    win, ref = d["win"], d["ref"]
    rows, cols, sites = [], [], []
    for s in range(10):
        cr, cc = rng.integers(20, n - 20, 2)
        while len(rows) < 20 * (s + 1):
            r, c = cr + rng.integers(-15, 16), cc + rng.integers(-15, 16)
            if ref[r, c]:
                rows.append(r), cols.append(c), sites.append(s)
    rows, cols = np.array(rows), np.array(cols)
    pts = gpd.GeoDataFrame({"class": np.array(CLASSES)[ref[rows, cols] - 1], "site": sites}, crs=feats.rio.crs,
                           geometry=gpd.points_from_xy(win.x.values[cols], win.y.values[rows]))
    rf2 = zeit.train_classifier(win, pts, label="class")
    for folds in (None, pts["site"]):
        r = zeit.aoa(feats, model=rf2, folds=folds)
        v = r.AOA.values
        print(f"  clustered sample, folds {'by site' if folds is not None else 'none'}: threshold "
              f"{r.attrs['threshold']:.3f}, outside {100 * (v == 0).sum() / (v != 255).sum():.1f}%")


FIGURES = {
    "smoothing": (fig_smoothing, True),
    "smoothness": (fig_smoothness, True),
    "desawtooth": (fig_desawtooth, False),
    "aoa": (fig_aoa, True),
}


def make_thumb():
    from PIL import Image
    path = OUT / "postproc_smoothing.webp"
    if not path.exists():
        return
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((0, 0, int(0.505 * w), int(0.9 * h)))
    tw, th = 800, 450
    im.thumbnail((tw - 40, th - 40), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), "white")
    canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
    (OUT / "thumbs").mkdir(exist_ok=True)
    canvas.save(OUT / "thumbs" / "postprocessing.webp", quality=85, method=6)
    print("  wrote the gallery thumbnail")


def main(names):
    mf.setup_style()
    for name in names or list(FIGURES):
        fn, needs = FIGURES[name]
        print(name)
        if not needs:
            fn()
            continue
        d = scene()
        if d is None:
            print("  skipped (set ZEIT_DOCS_RONDONIA to the annual NDVI GeoTIFF)")
            continue
        fn(d)
    make_thumb()


if __name__ == "__main__":
    main(sys.argv[1:])
