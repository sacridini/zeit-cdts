"""Generate the figures of the Emerging Hot Spots and Spectral Unmixing tutorials
(docs/tutorials/hotspots.md, docs/tutorials/unmixing.md).

Every figure is produced by running ``zeit.emerging_hotspots`` and ``zeit.unmix``, so
the numbers quoted in the tutorials can be checked by running this script. From the
repository root:

    python docs/scripts/make_hotspots_unmix_figures.py                  # every figure
    python docs/scripts/make_hotspots_unmix_figures.py hotspots_map     # a subset

Inputs (the figures whose data is missing are skipped):

* **Hot spots**: the annual Landsat NDVI stack over Rondonia of the other tutorials
  (1985-2024, NDVI x 10000, one band per year, 0 = no data), exported from LT-GEE.
  Point ``ZEIT_DOCS_RONDONIA`` at the GeoTIFF. LandTrendr's fitted trajectories
  remove the year-to-year wobble of the composites; the share of each 10 x 10 pixel
  cell (about 300 m) below NDVI 0.7 is the variable whose hot spots are mapped.
* **Unmixing**: an EnMAP L2A hyperspectral scene (224 bands, 418-2446 nm, 30 m,
  reflectance x 10000) over the coast north of Groningen, the Netherlands, on
  10 August 2022. Point ``ZEIT_DOCS_ENMAP`` at the GeoTIFF. A 350 x 350 pixel window, with its 210 usable bands,
  (10.5 km) is unmixed into green vegetation, dry vegetation, soil and water, with
  endmembers taken from pure pixels of the image.

The style, the ``save`` helper and the thumbnails follow ``make_figures.py``.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402

OUT = mf.OUT
NUMBERS = {}


def note(key, value):
    NUMBERS[key] = value
    print(f"  {key} = {value}")


def best_time(fn, repeat=5):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# ---------------------------------------------------------------------------
# Emerging hot spots: the share of open land in 300 m cells, Rondonia 1985-2024
# ---------------------------------------------------------------------------
CELL = 10            # pixels per cell side: 10 x 30 m = about 300 m
OPEN_NDVI = 7000     # NDVI x 10000 below which a pixel is open land (pasture, crops, bare)

# the colours of the patterns: hot spots warm, cold spots cool, darker = more lasting
PATTERN_COLORS = {
    1: "#f2c94c",    # new hot spot
    2: "#e0603f",    # consecutive
    3: "#7a1020",    # intensifying
    4: "#b32b4a",    # persistent
    5: "#f5a45d",    # diminishing
    6: "#f6d7c3",    # sporadic
    7: "#a0612a",    # oscillating
    8: "#c9a99a",    # historical
    0: "#efeee9",    # no pattern
    -1: "#8fd3e8",   # new cold spot
    -2: "#2a78d6",   # consecutive
    -3: "#123f7a",   # intensifying
    -4: "#4a3aa7",   # persistent
    -5: "#86b2e8",   # diminishing
    -6: "#cfe0f5",   # sporadic
    -7: "#1baf7a",   # oscillating
    -8: "#9aa8b5",   # historical
}
OPEN_CMAP = LinearSegmentedColormap.from_list(
    "open", ["#1b4d2b", "#3a7d44", "#7fb069", "#c9dfa4", "#f4f1e4", "#e9c9a0", "#c98a4b"])

_HS = {}


def open_share(ndvi_like, valid):
    """The share (%) of each cell's valid pixels below OPEN_NDVI; NaN where less than
    half of the cell has data."""
    open_ = (ndvi_like < OPEN_NDVI).where(valid)
    share = open_.coarsen(y=CELL, x=CELL, boundary="trim").mean() * 100
    n = valid.coarsen(y=CELL, x=CELL, boundary="trim").sum()
    return share.where(n >= CELL * CELL / 2)


def hotspots_data():
    if _HS:
        return _HS
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    if not path or not os.path.exists(path):
        return None
    ndvi = zeit.load_raster(path)                                   # (time, y, x), int16, 0 = no data
    t = time.perf_counter()
    lt = zeit.landtrendr(ndvi, fitted=True)
    note("hotspots_landtrendr_s", round(time.perf_counter() - t, 1))
    fitted = lt.fitted                                              # NDVI with the yearly wobble removed
    valid = fitted.notnull()                                        # NaN where the stack has no data
    share = open_share(fitted, valid)                               # (time, cell_y, cell_x)
    raw = open_share(ndvi.where(ndvi != 0), (ndvi != 0))            # the same from the raw composites
    note("hotspots_cells", f"{share.sizes['y']} x {share.sizes['x']} x {share.sizes['time']}")
    note("hotspots_open_share_by_decade",
         {int(y): round(float(share.sel(time=str(y)).mean()), 1) for y in (1985, 1995, 2005, 2015, 2024)})

    ehs = zeit.emerging_hotspots(share)                              # radius=1, lag=1, reference="date"
    note("hotspots_time_ms", round(1000 * best_time(lambda: zeit.emerging_hotspots(share)), 1))
    data = share.notnull().any("time").values
    _HS.update(ndvi=ndvi, share=share, raw=raw, ehs=ehs, data=data)

    def counts(result):
        c = result.category.values[data]
        u, k = np.unique(c, return_counts=True)
        return {zeit.emerging_hotspots.PATTERNS[int(a)]: round(100 * b / c.size, 1) for a, b in zip(u, k)}

    note("hotspots_patterns_pct", counts(ehs))
    note("hotspots_patterns_raw_ndvi_pct", counts(zeit.emerging_hotspots(raw)))
    note("hotspots_patterns_lag0_pct", counts(zeit.emerging_hotspots(share, lag=0)))
    note("hotspots_patterns_cube_pct", counts(zeit.emerging_hotspots(share, reference="cube")))
    note("hotspots_patterns_radius2_pct", counts(zeit.emerging_hotspots(share, radius=2, shape="circle")))
    return _HS


def _pick_cells(d):
    """One cell of three telling patterns: the cell of that pattern whose Gi* trend is
    the strongest, among cells surrounded by the same pattern."""
    from scipy import ndimage
    ehs = d["ehs"]
    cat = ehs.category.values
    z = ehs.trend_z.values
    picks = []
    for code, sign in ((5, -1), (2, 1), (-3, -1)):
        same = ndimage.minimum_filter((cat == code).astype(int), size=3) == 1
        score = np.where(same, sign * z, -np.inf)
        if code == 2:   # a clearing of the 2000s: about half of the dates hot
            score = np.where(same, -np.abs(ehs.n_hot.values - 20), -np.inf)
        r, c = np.unravel_index(np.argmax(score), score.shape)
        picks.append((code, int(r), int(c)))
    return picks


def _pattern_legend(ax, cat, data, loc="center left", anchor=(1.02, 0.5)):
    names = zeit.emerging_hotspots.PATTERNS
    c = cat[data]
    handles = []
    for code in [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8, 0]:
        share = 100 * np.mean(c == code)
        if share == 0:
            continue
        label = f"{names[code]}  {share:.0f}%" if share >= 1 else f"{names[code]}  <1%"
        handles.append(Line2D([], [], marker="s", linestyle="", markersize=10,
                              color=PATTERN_COLORS[code], label=label))
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, handletextpad=0.4,
              labelspacing=0.45, fontsize=9.5)


def fig_hotspots_map():
    d = hotspots_data()
    if d is None:
        print("  skipped (set ZEIT_DOCS_RONDONIA to the annual NDVI GeoTIFF)")
        return
    share, ehs, data = d["share"], d["ehs"], d["data"]
    cat = ehs.category.values
    codes = sorted(PATTERN_COLORS)
    lut = np.full(256, np.nan)
    for i, code in enumerate(codes):
        lut[code % 256] = i
    idx = lut[cat.astype(np.int16) % 256]
    idx[~data] = np.nan
    cmap = ListedColormap([PATTERN_COLORS[c] for c in codes])

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), gridspec_kw={"wspace": 0.04})
    for ax, year in zip(axes[:2], (1985, 2024)):
        im = ax.imshow(share.sel(time=str(year)).squeeze().values, cmap=OPEN_CMAP, vmin=0, vmax=100,
                       interpolation="nearest")
        mf.map_axes(ax, f"Open land in each 300 m cell, {year}")
    mf.colorbar(fig, im, axes[:2].tolist(), "% of the cell below NDVI 0.7", ticks=[0, 25, 50, 75, 100])
    ax = axes[2]
    ax.imshow(idx, cmap=cmap, vmin=-0.5, vmax=len(codes) - 0.5, interpolation="nearest")
    mf.map_axes(ax, "Emerging hot spot patterns, 1985–2024")
    for k, (code, r, c) in enumerate(_pick_cells(d), 1):
        ax.plot(c, r, "o", markersize=17, markerfacecolor="white", markeredgecolor=mf.INK, markeredgewidth=1.2)
        ax.text(c, r, str(k), ha="center", va="center", fontsize=10, fontweight="bold", color=mf.INK)
    _pattern_legend(ax, cat, data)
    mf.save(fig, "hotspots_map", photo=True)


def fig_hotspots_series():
    d = hotspots_data()
    if d is None:
        print("  skipped (set ZEIT_DOCS_RONDONIA to the annual NDVI GeoTIFF)")
        return
    share, ehs = d["share"], d["ehs"]
    years = share.time.dt.year.values
    names = zeit.emerging_hotspots.PATTERNS
    picks = _pick_cells(d)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12, "wspace": 0.18})
    for k, (code, r, c) in enumerate(picks):
        s = share.values[:, r, c]
        gi = ehs.gi.values[:, r, c]
        p = ehs.p.values[:, r, c]
        top, bottom = axes[0, k], axes[1, k]
        top.fill_between(years, 0, s, color="#e9c9a0", linewidth=0)
        top.plot(years, s, color="#a0612a", linewidth=1.6)
        top.set_ylim(0, 100)
        top.set_title(f"{k + 1} · {names[code]}")
        if k == 0:
            top.set_ylabel("Open land (%)")
            bottom.set_ylabel("Gi* (z-score)")
        lim = max(4.0, np.nanmax(np.abs(gi)) * 1.1)
        bottom.axhspan(-1.96, 1.96, color="#f3f2ee", zorder=0)
        bottom.axhline(0, color=mf.AXIS, linewidth=0.8)
        bottom.plot(years, gi, color=mf.INK2, linewidth=1.2, zorder=1)
        hot = (p < 0.05) & (gi > 0)
        cold = (p < 0.05) & (gi < 0)
        ns = ~(hot | cold)
        bottom.scatter(years[hot], gi[hot], s=22, color=mf.ORANGE, zorder=3, label="hot spot (p < 0.05)")
        bottom.scatter(years[cold], gi[cold], s=22, color=mf.BLUE, zorder=3, label="cold spot (p < 0.05)")
        bottom.scatter(years[ns], gi[ns], s=18, facecolor="white", edgecolor=mf.OBS, zorder=3,
                       label="not significant")
        bottom.set_ylim(-lim, lim)
        tz, tp = float(ehs.trend_z.values[r, c]), float(ehs.trend_p.values[r, c])
        ptxt = "p < 0.001" if tp < 0.001 else f"p = {tp:.2f}"
        bottom.text(0.02, 0.04, f"trend of the Gi*: z = {tz:.1f}, {ptxt}", transform=bottom.transAxes,
                    fontsize=9, color=mf.INK2)
        mf.year_axis(bottom, 10)
    axes[1, 0].legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, borderaxespad=0)
    mf.save(fig, "hotspots_series")


# ---------------------------------------------------------------------------
# Spectral unmixing: EnMAP over Groningen, 10 August 2022
# ---------------------------------------------------------------------------
WINDOW = dict(y=slice(560, 910), x=slice(740, 1090))   # 350 x 350 pixels of the scene
# endmembers: the median spectrum of the 3 x 3 pixels around a pure pixel (row, col in the window)
PURE = {"green vegetation": (231, 198), "dry vegetation": (41, 227), "soil": (251, 255), "water": (49, 24)}
EM_COLORS = {"green vegetation": "#1baf7a", "dry vegetation": "#e8b422", "soil": "#8c4a14", "water": "#2a78d6"}
FRACTION = LinearSegmentedColormap.from_list("fraction", ["#f7f6f2", "#c9d6cf", "#7fa39a", "#3c6e6a", "#173f45"])
ERROR = LinearSegmentedColormap.from_list("error", ["#fbf6ec", "#f5c98a", "#e0603f", "#8f1d3f"])

_UM = {}


def wavelength(name):
    """'Band 45 (647.214 nm)' -> 647.214"""
    return float(str(name).split("(")[1].split()[0])


def enmap_data():
    if _UM:
        return _UM
    path = os.environ.get("ZEIT_DOCS_ENMAP")
    if not path or not os.path.exists(path):
        return None
    cube = zeit.load_raster(path, chunks="auto").isel(**WINDOW).load()     # (band, y, x) int16
    refl = cube.where(cube != -32768) / 10000                              # reflectance, NaN = no data
    empty = refl.isnull().all(("y", "x"))
    negative = (refl < 0).mean(("y", "x")) > 0.001
    good = refl.band.values[~(empty | negative).values]
    note("unmix_bands", f"{refl.sizes['band']} bands, {len(good)} kept; dropped "
         f"{[round(wavelength(b)) for b in refl.band.values[(empty | negative).values]]} nm")
    spectra = refl.sel(band=good)
    wl = np.array([wavelength(b) for b in good])
    spectra = spectra.assign_coords(band=wl)                                # bands named by wavelength (nm)

    def spectrum(r, c):
        return spectra.isel(y=slice(r - 1, r + 2), x=slice(c - 1, c + 2)).median(["y", "x"]).to_series()

    endmembers = pd.DataFrame({k: spectrum(r, c) for k, (r, c) in PURE.items()}).T
    for k, (r, c) in PURE.items():
        note(f"unmix_point_{k.replace(' ', '_')}",
             (round(float(spectra.x[c])), round(float(spectra.y[r]))))
    fr = zeit.unmix(spectra, endmembers)
    note("unmix_time_s", round(best_time(lambda: zeit.unmix(spectra, endmembers), 3), 3))
    note("unmix_time_1thread_s", round(best_time(lambda: zeit.unmix(spectra, endmembers, n_jobs=1), 3), 3))
    note("unmix_mean_fractions", {v: round(float(fr[v].mean()), 3) for v in endmembers.index})
    note("unmix_rmse_p50_p95", np.percentile(fr.rmse, [50, 95]).round(4).tolist())
    fr3 = zeit.unmix(spectra, endmembers.drop("dry vegetation"))
    note("unmix_rmse_3em_p50_p95", np.percentile(fr3.rmse, [50, 95]).round(4).tolist())
    dry = fr["dry vegetation"].values > 0.4
    note("unmix_rmse_dry_over_40pct_4em_3em", [round(float(np.nanmedian(fr.rmse.values[dry])), 4),
                                               round(float(np.nanmedian(fr3.rmse.values[dry])), 4)])
    for label, kw in (("non-negative", dict(sum_to_one=False)), ("free", dict(sum_to_one=False, nonneg=False))):
        f = zeit.unmix(spectra, endmembers, **kw)
        total = sum(f[v] for v in endmembers.index)
        note(f"unmix_{label}_rmse_p50", round(float(f.rmse.median()), 4))
        note(f"unmix_{label}_sum_p5_p50_p95", np.percentile(total, [5, 50, 95]).round(2).tolist())
        note(f"unmix_{label}_min_fraction", round(float(min(f[v].min() for v in endmembers.index)), 2))
    # the same with Landsat 8's six bands, simulated from the full spectra
    oli = {"blue": (452, 512), "green": (533, 590), "red": (636, 673), "nir": (851, 879),
           "swir1": (1566, 1651), "swir2": (2107, 2294)}
    full = refl.sel(band=good)
    wl_full = np.array([wavelength(b) for b in good])
    landsat = xr.concat([full.isel(band=np.flatnonzero((wl_full >= lo) & (wl_full <= hi))).mean("band")
                         for lo, hi in oli.values()], dim="band").assign_coords(band=list(oli))
    em_landsat = pd.DataFrame({k: landsat.isel(y=slice(r - 1, r + 2), x=slice(c - 1, c + 2))
                               .median(["y", "x"]).to_series() for k, (r, c) in PURE.items()}).T
    fl = zeit.unmix(landsat, em_landsat)
    note("unmix_landsat_vs_enmap_median_absdiff",
         {v: round(float(np.abs(fr[v] - fl[v]).median()), 3) for v in endmembers.index})
    note("unmix_landsat_vs_enmap_p90_absdiff",
         {v: round(float(np.abs(fr[v] - fl[v]).quantile(0.9)), 3) for v in endmembers.index})
    _UM.update(refl=refl, spectra=spectra, wl=wl, endmembers=endmembers, fr=fr)
    return _UM


def _stretch(x):
    lo, hi = np.nanpercentile(x, [2, 98])
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _false_colour(refl):
    def band(w):
        names = refl.band.values
        i = int(np.argmin([abs(wavelength(b) - w) for b in names]))
        return refl.isel(band=i).values
    rgb = np.dstack([_stretch(band(865)), _stretch(band(1650)), _stretch(band(660))])
    return np.nan_to_num(rgb)


def _style_cb(cb, label):
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, colors=mf.MUTED, labelcolor=mf.INK2, length=0)
    cb.set_label(label, color=mf.INK2, fontsize=9.5)


def fig_unmixing_maps():
    d = enmap_data()
    if d is None:
        print("  skipped (set ZEIT_DOCS_ENMAP to the EnMAP GeoTIFF)")
        return
    fr = d["fr"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.4), gridspec_kw={"wspace": 0.04, "hspace": 0.14})
    ax = axes[0, 0]
    ax.imshow(_false_colour(d["refl"]), interpolation="nearest")
    mf.map_axes(ax, "EnMAP, 10 Aug 2022 (NIR, SWIR, red)")
    for k, (name, (r, c)) in enumerate(PURE.items(), 1):
        for colour, width in (("white", 3.2), (EM_COLORS[name], 1.6)):
            ax.add_patch(Rectangle((c - 6, r - 6), 12, 12, fill=False, edgecolor=colour, linewidth=width))
        ax.text(c + 9, r, str(k), color="white", fontsize=10, fontweight="bold", va="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=mf.INK)])
    panels = [(axes[0, 1], "green vegetation", "1"), (axes[0, 2], "dry vegetation", "2"),
              (axes[1, 0], "soil", "3"), (axes[1, 1], "water", "4")]
    for ax, name, k in panels:
        im = ax.imshow(fr[name].values, cmap=FRACTION, vmin=0, vmax=1, interpolation="nearest")
        mf.map_axes(ax, f"{k} · fraction of {name}")
    cax = axes[1, 0].inset_axes([0.0, -0.09, 1.98, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    _style_cb(cb, "Fraction (0–1)")
    ax = axes[1, 2]
    im = ax.imshow(fr.rmse.values, cmap=ERROR, vmin=0, vmax=0.04, interpolation="nearest")
    mf.map_axes(ax, "RMSE of the rebuilt spectrum")
    cax = ax.inset_axes([0.06, -0.09, 0.94, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.01, 0.02, 0.03, 0.04])
    _style_cb(cb, "Reflectance (0.04 and above: darkest)")
    mf.save(fig, "unmixing_maps", photo=True)


def _broken(wl, values, gap=40):
    """Insert NaN where bands are missing, so the lines break over the absorption gaps."""
    wl, values = np.asarray(wl, float), np.asarray(values, float)
    cut = np.flatnonzero(np.diff(wl) > gap) + 1
    return np.insert(wl, cut, np.nan), np.insert(values, cut, np.nan)


def fig_unmixing_spectra():
    d = enmap_data()
    if d is None:
        print("  skipped (set ZEIT_DOCS_ENMAP to the EnMAP GeoTIFF)")
        return
    wl, em, fr, spectra = d["wl"], d["endmembers"], d["fr"], d["spectra"]
    names = list(em.index)
    # a mixed pixel: a third or more of two materials, well rebuilt
    f = np.stack([fr[n].values for n in names])
    mixed = (np.sort(f, axis=0)[-2] > 0.3) & (fr.rmse.values < np.nanpercentile(fr.rmse, 30)) & (f[3] < 0.15)
    rows, cols = np.nonzero(mixed)
    k = len(rows) // 2
    r, c = int(rows[k]), int(cols[k])
    obs = spectra.isel(y=r, x=c).values
    frac = f[:, r, c]
    rebuilt = frac @ em.to_numpy()
    note("unmix_mixed_pixel", {"row_col": (r, c), **{n: round(float(v), 2) for n, v in zip(names, frac)},
                               "rmse": round(float(fr.rmse.values[r, c]), 4)})

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"wspace": 0.16})
    ax = axes[0]
    for n in names:
        ax.plot(*_broken(wl, em.loc[n].values), color=EM_COLORS[n], label=n)
    ax.set_title("Endmembers: pure pixels of the image")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.plot(*_broken(wl, obs), color=mf.OBS, linewidth=4, label="observed")
    ax.plot(*_broken(wl, rebuilt), color=mf.INK, linewidth=1.4, linestyle=(0, (4, 2)), label="rebuilt from the fractions")
    parts = "\n".join(f"{100 * v:.0f}% {n}" for n, v in zip(names, frac) if v >= 0.005)
    ax.set_title("A mixed pixel")
    ax.text(0.97, 0.97, parts + f"\nRMSE {fr.rmse.values[r, c]:.3f}", transform=ax.transAxes, va="top",
            ha="right", fontsize=9.5, color=mf.INK2, linespacing=1.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim(0, 0.65)
    ax.legend(loc="upper left")
    mf.save(fig, "unmixing_spectra")


# ---------------------------------------------------------------------------
FIGURES = {
    "hotspots_map": fig_hotspots_map,
    "hotspots_series": fig_hotspots_series,
    "unmixing_maps": fig_unmixing_maps,
    "unmixing_spectra": fig_unmixing_spectra,
}

# gallery thumbnails: a 16:9 crop of the most telling part of a figure
THUMBS = {
    "hotspots": ("hotspots_map.webp", (0.56, 0.03, 1.0, 0.93)),
    "unmixing": ("unmixing_maps.webp", (0.0, 0.0, 1.0, 0.46)),
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
    mf.setup_style()
    for name in names or list(FIGURES):
        print(name)
        FIGURES[name]()
    make_thumbs()
    print(json.dumps(NUMBERS, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
