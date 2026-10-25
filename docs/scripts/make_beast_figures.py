"""Generate the figures of the BEAST tutorial (docs/tutorials/beast.md).

Every figure is produced by running ``zeit.beast`` (and ``zeit.landtrendr``, to compare), so the
numbers quoted in the tutorial can be checked by running this script. From the repository
root:

    python docs/scripts/make_beast_figures.py                # every figure
    python docs/scripts/make_beast_figures.py beast_pixels   # a subset

Inputs (the figures of a missing input are skipped):

* **Annual NDVI**: the stack of Rondonia that ``make_figures.py`` uses for LandTrendr
  (1985-2024, NDVI x 10000, one band per year, 0 = no data). Point ``ZEIT_DOCS_RONDONIA`` at
  that GeoTIFF. A 300 x 300 pixel window around the first pixel of the tutorial is mapped
  (``WINDOW``).
* **16-day EVI**: MODIS EVI every 16 days from January 2001 to December 2025 (575 dates) at
  250 m over western France, one GeoTIFF with a band per date named ``EVI_YYYY_MM_DD``, the
  series of Janus' BFAST guide. Point ``ZEIT_DOCS_EVI`` at it.

The timings are taken with this process out of Windows' power throttling, which otherwise
slows a long run started from a terminal (see docs/benchmarks/vs-originals.md).

The style, the ``save`` helper and the thumbnails follow ``make_figures.py``.
"""

import ctypes
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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402

OUT = mf.OUT
NUMBERS = {}

# rows and columns of the window mapped (300 x 300 pixels, 9 x 9 km), centred on pixel A
WINDOW = (slice(763, 1063), slice(156, 456))
# the two pixels of the annual stack the tutorial follows: (row, column) of the whole stack
PIXELS = {"A": (913, 306), "B": (966, 250)}
EVI_PIXEL = (153, 435)          # row, column of the EVI series (pin 1 of Janus' BFAST guide)
MAP_SAMPLES = dict(samples=2000, chains=2)   # the map: fewer samples, as Janus' defaults


def note(key, value):
    NUMBERS[key] = value
    print(f"  {key} = {value}")


def no_power_throttling():
    """This process out of Windows' power throttling (as benchmarks/vs_originals/beast.py)."""
    if os.name != "nt":
        return
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    s = State(1, 1 | 4, 0)
    k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))


def _timed(f, *a, repeat=1, **kw):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = f(*a, **kw)
        best = min(best, time.perf_counter() - t0)
    return out, best


# ---------------------------------------------------------------------------
_A = {}


def annual():
    """The annual NDVI stack, BEAST on its two pixels and on the window, and LandTrendr on the
    window (cached)."""
    if _A:
        return _A
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    if not path or not os.path.exists(path):
        return None
    ndvi = zeit.load_raster(path)                       # (40 years, y, x), NDVI x 10000
    years = ndvi.time.dt.year.values
    pix = {}
    for name, (r, c) in PIXELS.items():
        series = ndvi.isel(y=r, x=c)
        res, dt = _timed(zeit.beast, series, season="none", repeat=3)
        pix[name] = (series, res)
        note(f"time_pixel_{name}_s", round(dt, 3))
    win = ndvi.isel(y=WINDOW[0], x=WINDOW[1])
    note("window_valid", int((win.values != 0).all(axis=0).sum()))
    res, dt = _timed(zeit.beast, win, season="none", **MAP_SAMPLES)
    note("time_window_s", round(dt, 1))
    lt = zeit.landtrendr(win.values.astype(np.float32), years=years, max_segments=6, nodata=0)
    loss = zeit.extract_events(lt, event_type="loss", sort_by="greatest", min_magnitude=2000)
    _A.update(ndvi=ndvi, years=years, pix=pix, win=win, res=res, lt=lt, loss=loss)
    return _A


def _lt_fit(series, years):
    lt = zeit.landtrendr(series.values.astype(np.float32), years=years, max_segments=6, nodata=0)
    vy = lt.vertex_year.values
    vv = lt.vertex_value.values
    keep = vy > 0
    return vy[keep], vv[keep]


def fig_beast_pixels(d):
    years = d["years"]
    fig = plt.figure(figsize=(13.5, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=0.12, wspace=0.1)
    for k, name in enumerate(PIXELS):
        series, r = d["pix"][name]
        top = fig.add_subplot(gs[0, k])
        bot = fig.add_subplot(gs[1, k], sharex=top)
        t = r.time.values
        v = series.values / 1e4
        lo, hi = r.trend_ci.sel(bound="lower").values / 1e4, r.trend_ci.sel(bound="upper").values / 1e4
        top.fill_between(t, lo, hi, color=mf.BLUE, alpha=0.16, linewidth=0, zorder=1, label="Trend, 95% interval")
        top.plot(t, r.trend.values / 1e4, color=mf.BLUE, linewidth=2, zorder=3, label="BEAST trend (model average)")
        vy, vv = _lt_fit(series, years)
        top.plot(vy, vv / 1e4, color=mf.ORANGE, linewidth=1.4, linestyle=(0, (4, 2)), zorder=2,
                 label="LandTrendr (one model)")
        mf.obs_scatter(top, years, v, s=20, label="NDVI")
        cps = r.trend_cp.values
        prob = r.trend_cp_prob.values
        ok = np.isfinite(cps)
        for c, p in zip(cps[ok], prob[ok]):
            if p > 0.5:
                top.axvline(c, color=mf.INK2, linewidth=1.0, linestyle=(0, (1, 2)), zorder=0)
        occ = r.trend_cp_occ.values
        bot.bar(t, occ, width=0.7, color=mf.VIOLET, alpha=0.85, zorder=2)
        for c, p in zip(cps[ok][:4], prob[ok][:4]):
            if p < 0.1:
                continue
            i = int(np.argmin(np.abs(t - c)))
            bot.annotate(f"{int(c)}: {p:.0%}", (c, occ[i]), xytext=(0, 4), textcoords="offset points",
                         ha="center", fontsize=9, color=mf.INK)
        bot.set_ylim(0, 1.15)
        bot.set_yticks([0, 0.5, 1])
        top.set_ylim(0.45, 0.95)
        top.set_xlim(1983.5, 2025.5)
        mf.year_axis(bot, 5)
        plt.setp(top.get_xticklabels(), visible=False)
        ncp = r.trend_ncp_prob.values
        title = ("A: forest cleared in 1999" if name == "A" else "B: a forest left standing")
        top.set_title(title)
        # the probability of each number of changepoints, inset
        ins = top.inset_axes([0.70, 0.06, 0.28, 0.30] if name == "A" else [0.70, 0.06, 0.28, 0.30])
        n = np.arange(len(ncp))
        ins.bar(n[:7], ncp[:7], color=mf.INK2, width=0.7)
        ins.set_xticks(n[:7])
        ins.tick_params(labelsize=8)
        ins.set_yticks([])
        ins.set_ylim(0, max(0.6, ncp.max() * 1.15))
        ins.grid(False)
        ins.set_facecolor((1, 1, 1, 0.85))
        ins.set_title("P(number of changepoints)", fontsize=8.5, fontweight="normal", pad=3)
        note(f"pixel_{name}", dict(
            rowcol=PIXELS[name], ncp_mean=round(float(r.trend_ncp), 2), ncp_mode=int(r.trend_ncp_mode),
            ncp_prob=[round(float(x), 3) for x in ncp[:7]],
            cp=[float(c) for c in cps[ok]], cp_prob=[round(float(p), 3) for p in prob[ok]],
            cp_abrupt=[round(float(a)) for a in r.trend_cp_abrupt.values[ok]],
            cp_ci=[[round(float(a), 1) for a in r.trend_cp_ci.values[:, j]] for j in np.nonzero(ok)[0]],
            r2=round(float(r.r2), 3), rmse=round(float(r.rmse), 1),
            slope_end=round(float(r.trend_slope.values[-1]), 1),
            slope_end_pos=round(float(r.trend_slope_pos_prob.values[-1]), 3),
            lt_vertices=[int(x) for x in vy]))
        if k == 0:
            top.set_ylabel("NDVI")
            bot.set_ylabel("P(changepoint)")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=mf.INK2, linewidth=1.0, linestyle=(0, (1, 2))))
    labels.append("Changepoint, P > 0.5")
    handles.append(Patch(facecolor=mf.VIOLET, alpha=0.85))
    labels.append("Probability of a changepoint in the year")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=3, fontsize=9.5)
    mf.save(fig, "beast_pixels")


def fig_beast_maps(d):
    r, win, loss, years = d["res"], d["win"], d["loss"], d["years"]
    ncp = r.trend_ncp.values
    p0 = r.trend_cp_prob.isel(tcp=0).values
    cp0 = r.trend_cp.isel(tcp=0).values
    j0 = r.trend_cp_abrupt.isel(tcp=0).values / 1e4
    sure = p0 > 0.5
    note("map_ncp_pct", [round(float(x), 2) for x in np.nanpercentile(ncp, [10, 50, 90])])
    note("map_share_p0_gt_05", round(float(np.mean(sure)), 3))
    note("map_share_p0_gt_09", round(float(np.mean(p0 > 0.9)), 3))
    note("map_share_sure_negative", round(float(np.mean(j0[sure] < 0)), 3))
    yrs, cnt = np.unique(cp0[sure].astype(int), return_counts=True)
    top = np.argsort(cnt)[::-1][:5]
    note("map_top_years", {int(yrs[i]): round(float(cnt[i] / ncp.size), 3) for i in top})
    note("map_r2_median", round(float(np.nanmedian(r.r2.values)), 3))
    neg = sure & (j0 < 0)
    note("map_jump_drops_pct10_50_90", [round(float(x), 3) for x in np.percentile(j0[neg], [10, 50, 90])])
    rise = sure & (j0 > 0)
    note("map_rises", dict(share=round(float(rise.mean()), 3), jump_median=round(float(np.median(j0[rise])), 3)))

    # agreement with LandTrendr's greatest loss (yod, magnitude >= 0.2 NDVI)
    yod = loss.yod.values
    has_lt = yod > years[0]
    cps, prob, jump = r.trend_cp.values, r.trend_cp_prob.values, r.trend_cp_abrupt.values
    drop = (prob > 0.5) & (jump < -1000) & np.isfinite(cps)            # a likely drop of 0.1 NDVI or more
    has_b = drop.any(axis=0)
    near = (np.abs(cps - yod[None]) <= 1) & drop
    note("lt_share_loss", round(float(np.mean(has_lt)), 3))
    note("beast_share_drop", round(float(np.mean(has_b)), 3))
    note("lt_loss_with_beast_drop_within_1y", round(float(near.any(axis=0)[has_lt].mean()), 3))
    near_any = np.zeros_like(drop)
    for k in range(cps.shape[0]):
        near_any[k] = drop[k] & (np.abs(cps[k] - yod) <= 1) & has_lt
    note("beast_drop_with_lt_loss_within_1y", round(float(near_any.sum() / drop.sum()), 3))
    # by the duration of LandTrendr's loss: BEAST takes a slow decline as a slope, not a changepoint
    dur, mag = loss.duration.values, loss.magnitude.values
    found = near.any(axis=0)
    for name, m in [("1y", dur == 1), ("2y", dur == 2), ("3-4y", (dur >= 3) & (dur <= 4)), ("5y+", dur >= 5)]:
        m = m & has_lt
        note(f"lt_loss_{name}", dict(share_of_lt_loss=round(float(m.sum() / has_lt.sum()), 3),
                                     with_beast_drop=round(float(found[m].mean()), 3)))
    m = has_lt & (dur == 1) & (mag > 3000)
    note("lt_loss_1y_over_03", dict(n=int(m.sum()), with_beast_drop=round(float(found[m].mean()), 3)))

    last = win.isel(time=-1).values / 1e4
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.55), gridspec_kw={"wspace": 0.05})
    ax = axes[0]
    im = ax.imshow(last, cmap=mf.GREENS, vmin=0.4, vmax=0.9, interpolation="nearest")
    mf.map_axes(ax, f"NDVI, {int(years[-1])}")
    mf.colorbar(fig, im, ax, "NDVI", ticks=[0.4, 0.6, 0.8])
    ax = axes[1]
    im = ax.imshow(ncp, cmap="viridis", vmin=0, vmax=5, interpolation="nearest")
    mf.map_axes(ax, "Trend changepoints (mean)")
    mf.colorbar(fig, im, ax, "Number, averaged over the models", ticks=[0, 1, 2, 3, 4, 5])
    ax = axes[2]
    ax.imshow(np.ones_like(ncp), cmap=ListedColormap(["#e4e3dd"]), interpolation="nearest")
    im = ax.imshow(np.where(sure, cp0, np.nan), cmap=mf.YEARS, vmin=1986, vmax=2024, interpolation="nearest")
    mf.map_axes(ax, "Most probable changepoint")
    mf.colorbar(fig, im, ax, "Year (grey: none with P > 0.5)", ticks=[1990, 2000, 2010, 2020])
    ax = axes[3]
    ax.imshow(np.ones_like(ncp), cmap=ListedColormap(["#e4e3dd"]), interpolation="nearest")
    im = ax.imshow(np.where(sure, j0, np.nan), cmap=mf.DIVERGING, vmin=-0.3, vmax=0.3, interpolation="nearest")
    mf.map_axes(ax, "Its jump")
    mf.colorbar(fig, im, ax, "NDVI (grey: none with P > 0.5)", ticks=[-0.3, -0.15, 0, 0.15, 0.3])
    r0, c0 = WINDOW[0].start, WINDOW[1].start
    for a in axes:
        for name, (rr, cc) in PIXELS.items():
            a.scatter([cc - c0], [rr - r0], s=60, facecolor="none", edgecolor=mf.INK, linewidth=1.5, zorder=5)
            a.text(cc - c0 + 6, rr - r0 - 6, name, color=mf.INK, fontsize=10, fontweight="semibold", zorder=5,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    mf.save(fig, "beast_maps", photo=True)


# ---------------------------------------------------------------------------
def evi():
    path = os.environ.get("ZEIT_DOCS_EVI")
    if not path or not os.path.exists(path):
        return None
    import rasterio
    r, c = EVI_PIXEL
    with rasterio.open(path) as src:
        dates = pd.to_datetime([n.split("_", 1)[1].replace("_", "-") for n in src.descriptions])
        v = src.read(window=((r, r + 1), (c, c + 1)))[:, 0, 0].astype(np.float64)
    v[~np.isfinite(v)] = np.nan
    return pd.Series(v, index=dates)


def fig_beast_season(s):
    note("evi_dates", len(s))
    note("evi_missing", int(s.isna().sum()))
    r, dt = _timed(zeit.beast, s, deltat=1 / 23, repeat=3)
    note("time_evi_s", round(dt, 2))
    r3 = zeit.beast(s, deltat=1 / 23, tcp_minmax=(0, 3))
    t = r.time.values

    def cps(res, comp, n=6):
        c = res[f"{comp}_cp"].values
        ok = np.isfinite(c)
        return [(round(float(a), 2), round(float(p), 2), round(float(j), 3))
                for a, p, j in zip(c[ok], res[f"{comp}_cp_prob"].values[ok], res[f"{comp}_cp_abrupt"].values[ok])][:n]

    note("evi", dict(grid=int(r.sizes["time"]), deltat=float(r.attrs["deltat"]), r2=round(float(r.r2), 3),
                     trend_ncp=round(float(r.trend_ncp), 2), season_ncp=round(float(r.season_ncp), 2),
                     trend_cp=cps(r, "trend"), season_cp=cps(r, "season"),
                     season_order_2001_2012=round(float(r.season_order.values[t < 2013].mean()), 2),
                     season_order_2014_2025=round(float(r.season_order.values[t >= 2014].mean()), 2),
                     season_amp_2001_2012=round(float(r.season_amp.values[t < 2013].mean()), 4),
                     season_amp_2014_2025=round(float(r.season_amp.values[t >= 2014].mean()), 4)))
    note("evi_tcp3", dict(trend_ncp=round(float(r3.trend_ncp), 2), trend_cp=cps(r3, "trend"),
                          season_ncp=round(float(r3.season_ncp), 2), season_cp=cps(r3, "season", 3),
                          r2=round(float(r3.r2), 3)))
    s_pre = r.season.values[(t >= 2009) & (t < 2012)]
    s_post = r.season.values[(t >= 2015) & (t < 2018)]
    note("evi_season_range", dict(before=[round(float(s_pre.min()), 3), round(float(s_pre.max()), 3)],
                                  after=[round(float(s_post.min()), 3), round(float(s_post.max()), 3)]))

    fig, axes = plt.subplots(3, 1, figsize=(13.5, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1, 1], "hspace": 0.16})
    ax = axes[0]
    mf.obs_scatter(ax, t, r.data.values, s=11, label="EVI on the 16-day grid")
    ax.plot(t, r.trend.values + r.season.values, color=mf.BLUE, linewidth=1.3, zorder=3,
            label="BEAST fit (trend + season)")
    ax.set_ylabel("EVI")
    ax.set_ylim(0.05, 0.95)
    ax.set_title("A field of western France, MODIS EVI every 16 days")
    ax.legend(loc="upper right", ncol=2)

    def with_occ(ax, y, lo, hi, occ, colour, label):
        if lo is not None:
            ax.fill_between(t, lo, hi, color=colour, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(t, y, color=colour, linewidth=1.6, zorder=3, label=label)
        tw = ax.twinx()
        tw.bar(t, occ, width=1 / 23, color=mf.VIOLET, alpha=0.75, zorder=1)
        tw.set_ylim(0, 2.2)
        tw.set_yticks([0, 0.5, 1])
        tw.grid(False)
        tw.spines["right"].set_visible(True)
        tw.set_ylabel("P(changepoint)", color=mf.INK2)
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax = axes[1]
    with_occ(ax, r.trend.values, r.trend_ci.sel(bound="lower").values, r.trend_ci.sel(bound="upper").values,
             r.trend_cp_occ.values, mf.BLUE, "Trend, with its 95% interval")
    ax.set_ylabel("Trend")
    ax.legend(loc="upper right")
    ax = axes[2]
    with_occ(ax, r.season.values, None, None, r.season_cp_occ.values, mf.AQUA, "Season")
    ax.set_ylabel("Season")
    ax.legend(loc="upper right")
    mf.year_axis(ax, 2)
    ax.set_xlim(2000.7, 2026.2)
    fig.legend([Patch(facecolor=mf.VIOLET, alpha=0.75)], ["Probability of a changepoint at each date (right axis)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.06), fontsize=9.5)
    mf.save(fig, "beast_season")


# ---------------------------------------------------------------------------
FIGURES = {
    "beast_pixels": ("annual", fig_beast_pixels),
    "beast_maps": ("annual", fig_beast_maps),
    "beast_season": ("evi", fig_beast_season),
}

# Gallery thumbnail: a 16:9 crop of the most telling part of a figure, as in make_figures.py.
THUMBS = {
    "beast": ("beast_pixels.png", (0.0, 0.0, 0.505, 0.5)),
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
    print("  wrote the gallery thumbnails")


def main(names):
    no_power_throttling()
    mf.setup_style()
    inputs = {"annual": annual, "evi": evi}
    for name in names or list(FIGURES):
        kind, f = FIGURES[name]
        d = inputs[kind]()
        if d is None:
            print(f"{name}: skipped (set {'ZEIT_DOCS_RONDONIA' if kind == 'annual' else 'ZEIT_DOCS_EVI'})")
            continue
        print(name)
        f(d)
    make_thumbs()
    print(json.dumps(NUMBERS, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
