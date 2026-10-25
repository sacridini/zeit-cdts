"""Generate the figures of the S-CCD tutorial (docs/tutorials/sccd.md).

Every figure is produced by running ``zeit.sccd`` (and ``zeit.ccdc``, to compare), so the
numbers quoted in the tutorial can be checked by running this script. From the repository
root:

    python docs/scripts/make_sccd_figures.py                # every figure
    python docs/scripts/make_sccd_figures.py sccd_pixels    # a subset

Input (the figures are skipped without it): the Landsat Collection 2 Level-2 stack of
Rondonia used by the S-CCD tests and benchmarks, ``CCDC_Stack_Rondonia.tif``: 125 dates from
2020-02-04 to 2026-08-23, 7 bands each (Blue, Green, Red, NIR, SWIR1, SWIR2 as digital numbers,
then QA_PIXEL), 1690 x 1675 pixels in EPSG:4326, 0 = no data. Point ``ZEIT_DOCS_SCCD`` at the
GeoTIFF; its dates are read from ``CCDC_Stack_Rondonia_Dates.csv`` next to it (column
``Date``). A 16 x 16 km window is read (``WINDOW``); the stack interleaves two Landsat paths,
and the window lies on one of them, so the 61 dates of that path are kept.

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
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402

OUT = mf.OUT
NUMBERS = {}

WINDOW = (-63.4269, -10.2965, -63.2667, -10.1362)   # west, south, east, north (degrees): 596 x 596 pixels
BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
# The two pixels the tutorial follows (longitude, latitude of their centres): a forest cleared in
# July 2023, and a forest whose latest observations depart from its model.
PIXELS = {"cleared": (-63.278350, -10.272438), "alert": (-63.325155, -10.184206)}
STATE_NAMES = ["Stable", "Change in progress", "Waiting for a model after a change", "No model"]
STATE_COLOURS = ["#e4e3dd", mf.ORANGE, mf.BLUE, "#ffffff"]


def note(key, value):
    NUMBERS[key] = value
    print(f"  {key} = {value}")


def fmask(qa_pixel):
    """Landsat Collection 2 QA_PIXEL as the Fmask codes of zeit.ccdc and zeit.sccd:
    0 clear land, 1 water, 2 cloud shadow, 3 snow, 4 cloud, 255 no observation."""
    missing = ~np.isfinite(qa_pixel)
    q = np.nan_to_num(qa_pixel).astype(np.int64)
    cloud = (q & 0b1110) != 0                       # dilated cloud, cirrus, cloud
    code = np.full(q.shape, 4, dtype=np.uint8)      # not said to be clear: cloud
    code[(q & (1 << 6)) != 0] = 0                   # clear
    code[((q & (1 << 7)) != 0) & ~cloud] = 1        # water
    code[(q & (1 << 5)) != 0] = 3                   # snow
    code[(q & (1 << 4)) != 0] = 2                   # cloud shadow
    code[cloud] = 4
    code[missing | ((q & 1) != 0)] = 255            # fill
    return code


# ---------------------------------------------------------------------------
_D = {}


def data():
    """The window as a (time, band, y, x) cube of reflectance x 10000, its Fmask codes, and the
    results of zeit.sccd and zeit.ccdc on it (cached)."""
    if _D:
        return _D
    path = os.environ.get("ZEIT_DOCS_SCCD")
    if not path or not os.path.exists(path):
        return None
    csv = Path(path).with_name(Path(path).stem + "_Dates.csv")
    dates = pd.to_datetime(pd.read_csv(csv).Date)

    stack = zeit.load_raster(path, clip=WINDOW)                  # (875 layers, y, x), 0 -> NaN
    ny, nx = stack.sizes["y"], stack.sizes["x"]
    layers = stack.values.reshape(len(dates), 7, ny, nx)
    qa = fmask(layers[:, 6])
    keep = (qa != 255).any(axis=(1, 2))                          # the dates of this path
    cube = xr.DataArray(layers[keep, :6] * 0.275 - 2000, dims=("time", "band", "y", "x"),
                        coords={"time": dates[keep].values, "band": BANDS, "y": stack.y, "x": stack.x})
    cube = cube.rio.write_crs(stack.rio.crs)
    qa = qa[keep]
    note("window_pixels", f"{nx} x {ny}")
    note("window_dates", int(keep.sum()))
    note("clear_per_pixel_median", int(np.median((qa < 2).sum(axis=0))))

    t0 = time.perf_counter()
    seg = zeit.sccd(cube, qa=qa)
    t_sccd = time.perf_counter() - t0
    t0 = time.perf_counter()
    cc = zeit.ccdc(cube, qa=qa)
    t_ccdc = time.perf_counter() - t0
    note("time_sccd_s", round(t_sccd, 1))
    note("time_ccdc_s", round(t_ccdc, 1))
    _D.update(cube=cube, qa=qa, seg=seg, cc=cc)
    return _D


def _ndvi(nir, red):
    return (nir - red) / (nir + red)


def _state(seg):
    """1 stable, 2 change in progress, 3 waiting for a model after a change, 4 no model: Janus' map."""
    mode, na, ns = seg.nrt_mode.values, seg.n_anomalies.values, seg.n_segments.values
    return np.where(mode == 1, np.where(na > 0, 2, 1), np.where((mode == 2) & (ns > 0), 3, 4))


def _rowcol(d, name):
    lon, lat = PIXELS[name]
    cube = d["cube"]
    return int(np.abs(cube.y.values - lat).argmin()), int(np.abs(cube.x.values - lon).argmin())


def _pixel(d, name):
    r, c = _rowcol(d, name)
    return d["cube"].isel(y=r, x=c), d["qa"][:, r, c]


def _years(times):
    t = pd.DatetimeIndex(times)
    return (t.year + (t.dayofyear - 1) / 365.25).to_numpy()


# ---------------------------------------------------------------------------
# Status colours of a pixel's observations
STATUS_STYLE = {
    1: dict(color=mf.INK2, label="In the model"),
    2: dict(facecolor="white", edgecolor=mf.INK2, label="Outlier, left out"),
    3: dict(color=mf.ORANGE, label="First observation of a change"),
    4: dict(color=mf.VIOLET, label="The last 6, not in the model yet"),
    5: dict(color=mf.AQUA, label="Waiting for a new model"),
}


def _scatter_status(ax, t, v, status, s=22):
    for k, style in STATUS_STYLE.items():
        m = (status == k) & np.isfinite(v)
        if not m.any():
            continue
        kw = dict(style)
        label = kw.pop("label")
        if k == 2:
            ax.scatter(t[m], v[m], s=s, linewidth=1.1, zorder=3, label=label, **kw)
        else:
            ax.scatter(t[m], v[m], s=s if k != 3 else 46, edgecolor="white", linewidth=0.6, zorder=4 if k != 1 else 3,
                       marker="D" if k == 3 else "o", label=label, **kw)


def _segments_model(px, t, status):
    """The filtered model's NDVI at each date of each segment (the dates it was updated on, and
    the latest observations, which it predicts without them)."""
    p = px.predicted
    model = _ndvi(p.sel(band="nir").values, p.sel(band="red").values)
    days = pd.DatetimeIndex(px.time.values)
    out = []
    for k in range(int(px.n_segments)):
        t0, t1 = px.t_start.values[k], px.t_end.values[k]
        inside = (days >= t0) & (days <= t1) & (status == 1)
        out.append((t[inside], model[inside]))
    tail = status == 4
    return out, (t[tail], model[tail]), model


def fig_sccd_pixels(d):
    from scipy.stats import chi2
    threshold = float(d["seg"].attrs["threshold"])
    outlier = float(chi2.ppf(0.99999, 5))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.12, "wspace": 0.12})
    for k, name in enumerate(PIXELS):
        series, qa = _pixel(d, name)
        px = zeit.sccd(series, qa=qa)
        t = _years(px.time.values)
        status = px.status.values
        v = _ndvi(series.sel(band="nir").values, series.sel(band="red").values)
        top, bot = axes[0, k], axes[1, k]
        segs, (tt, tm), model = _segments_model(px, t, status)
        for i, (ts, ms) in enumerate(segs):
            top.plot(ts, ms, color=mf.BLUE, linewidth=1.6, zorder=2, label="Filtered model" if i == 0 else None)
        if len(tt):
            last = [t[status == 1][-1]] if (status == 1).any() else []
            lv = [model[status == 1][-1]] if last else []
            top.plot(np.r_[last, tt], np.r_[lv, tm], color=mf.BLUE, linewidth=1.4, linestyle=(0, (3, 2)), zorder=2,
                     label="Its prediction of the latest dates")
        _scatter_status(top, t, v, status)
        stat = px.statistic.values
        tested = np.isfinite(stat) & (stat > 0)
        bot.axhline(threshold, color=mf.ORANGE, linewidth=1.3, linestyle=(0, (4, 3)), zorder=1,
                    label=f"Change threshold ({threshold:.1f})")
        bot.axhline(outlier, color=mf.INK2, linewidth=1.0, linestyle=(0, (1, 2)), zorder=1,
                    label=f"Outlier threshold ({outlier:.1f})")
        _scatter_status(bot, t[tested], stat[tested], status[tested], s=18)
        bot.set_yscale("log")
        bot.set_ylim(0.5, 800)
        bot.yaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
        bot.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        bot.yaxis.set_minor_locator(NullLocator())
        for b in px.t_break.values:
            if not np.isnat(b):
                for ax in (top, bot):
                    mf.break_line(ax, _years([b])[0], label="Break" if ax is top else None)
        breaks = [str(b)[:10] for b in px.t_break.values if not np.isnat(b)]
        na, start = int(px.n_anomalies), px.anomaly_start.values
        if name == "cleared":
            b = pd.Timestamp(breaks[0])
            top.set_title(f"A: forest cleared, a break on {b.day} {b:%B %Y}")
        else:
            a = pd.Timestamp(start)
            top.set_title(f"B: a change in progress, {na} of 6 anomalies since {a.day} {a:%B %Y}")
        top.set_ylim(0, 1)
        top.set_xlim(2020.2, 2026.9)
        mf.year_axis(top, 1)
        note(f"sccd_pixel_{name}", dict(
            rowcol=_rowcol(d, name), breaks=breaks, n_segments=int(px.n_segments),
            nrt_mode=int(px.nrt_mode), n_anomalies=na, anomaly_start=str(start)[:10],
            anomaly_statistic=round(float(px.anomaly_statistic), 1) if na else None,
            magnitude_nir=[round(float(m)) for m in px.magnitude.sel(band="nir").values[:int(px.n_segments)]],
            anomaly_magnitude={b: round(float(px.anomaly_magnitude.sel(band=b))) for b in BANDS} if na else None,
            n_outliers=int((status == 2).sum()), n_clear=int((qa < 2).sum())))
    axes[0, 0].set_ylabel("NDVI")
    axes[1, 0].set_ylabel("Test statistic")
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", bbox_to_anchor=(0.5, 0.045),
               ncol=4, fontsize=9.5)
    mf.save(fig, "sccd_pixels")


def fig_sccd_maps(d):
    cube, seg, qa = d["cube"], d["seg"], d["qa"]
    state = _state(seg)
    n = state.size
    note("sccd_states", {STATE_NAMES[i]: round(float(np.mean(state == i + 1)), 3) for i in range(4)})
    note("sccd_n_breaks", {int(k): round(float(np.mean(seg.n_breaks.values == k)), 3) for k in range(4)})
    na = seg.n_anomalies.values
    note("sccd_in_progress_by_n", {int(k): int((na == k).sum()) for k in range(1, 7)})
    note("pixels", n)

    # the NDVI of the last date, the date of the latest break, and the monitoring state
    last = cube.isel(time=-1)
    ndvi = _ndvi(last.sel(band="nir").values, last.sel(band="red").values)
    ndvi = np.where(qa[-1] < 2, ndvi, np.nan)
    tb = seg.t_break.values
    latest = np.full(state.shape, np.nan)
    for k in range(tb.shape[0]):
        y = _years(tb[k].ravel()).reshape(state.shape)
        latest = np.where(np.isfinite(y), y, latest)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.3), gridspec_kw={"wspace": 0.06})
    ax = axes[0]
    im = ax.imshow(ndvi, cmap=mf.GREENS, vmin=0.2, vmax=0.9, interpolation="nearest")
    day = pd.Timestamp(str(last.time.values))
    mf.map_axes(ax, f"NDVI, {day.day} {day:%B %Y}")
    mf.colorbar(fig, im, ax, "NDVI", ticks=[0.2, 0.4, 0.6, 0.8])

    ax = axes[1]
    ax.imshow(np.ones_like(latest), cmap=ListedColormap(["#e4e3dd"]), interpolation="nearest")
    im = ax.imshow(latest, cmap=mf.YEARS, vmin=2021, vmax=2027, interpolation="nearest")
    mf.map_axes(ax, "Date of the latest confirmed break")
    mf.colorbar(fig, im, ax, "Decimal year (grey: none)", ticks=[2021, 2022, 2023, 2024, 2025, 2026, 2027])

    ax = axes[2]
    ax.imshow(state, cmap=ListedColormap(STATE_COLOURS), vmin=0.5, vmax=4.5, interpolation="nearest")
    mf.map_axes(ax, f"Monitoring state on {day.day} {day:%B %Y}")
    ax.legend(handles=[Patch(facecolor=c, edgecolor=mf.AXIS, linewidth=0.5, label=nm)
                       for c, nm in zip(STATE_COLOURS, STATE_NAMES)],
              loc="upper left", bbox_to_anchor=(0.0, -0.03), ncol=2, fontsize=9, handlelength=1.2,
              columnspacing=1.0)
    for a in axes:
        for k, name in enumerate(PIXELS):
            r, c = _rowcol(d, name)
            a.scatter([c], [r], s=70, facecolor="none", edgecolor=mf.INK, linewidth=1.6, zorder=5)
            a.text(c + 9, r - 9, "AB"[k], color=mf.INK, fontsize=10, fontweight="semibold", zorder=5,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    mf.save(fig, "sccd_maps", photo=True)


def _ccdc_curve(cc_px, band, k, days):
    c = cc_px.coefs.isel(segment=k).sel(band=band).values
    t = np.array([x.toordinal() + 366 for x in days], dtype=float)
    w = 2 * np.pi / 365.25
    v = c[0] + c[1] * t
    for h in range(1, 4):
        v = v + c[2 * h] * np.cos(h * w * t) + c[2 * h + 1] * np.sin(h * w * t)
    return v


def fig_sccd_vs_ccdc(d):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), sharey=True, gridspec_kw={"wspace": 0.06})
    for k, name in enumerate(PIXELS):
        ax = axes[k]
        series, qa = _pixel(d, name)
        px = zeit.sccd(series, qa=qa)
        cc = zeit.ccdc(series, qa=qa)
        t = _years(px.time.values)
        status = px.status.values
        v = _ndvi(series.sel(band="nir").values, series.sel(band="red").values)
        clear = (qa < 2) & np.isfinite(v)
        mf.obs_scatter(ax, t[clear], v[clear], s=16, label="Clear observations")
        for s in range(int(cc.n_segments)):
            days = pd.date_range(pd.Timestamp(cc.t_start.values[s]), pd.Timestamp(cc.t_end.values[s]), freq="D")
            nd = _ndvi(_ccdc_curve(cc, "nir", s, days), _ccdc_curve(cc, "red", s, days))
            ax.plot(_years(days), nd, color=mf.ORANGE, linewidth=1.3, alpha=0.9, zorder=2,
                    label="CCDC (refitted)" if s == 0 else None)
        segs, _, _ = _segments_model(px, t, status)
        for i, (ts, ms) in enumerate(segs):
            ax.plot(ts, ms, color=mf.BLUE, linewidth=1.6, zorder=3, label="S-CCD (filtered)" if i == 0 else None)
        for b in cc.t_break.values:
            if not np.isnat(b):
                ax.axvline(_years([b])[0], color=mf.ORANGE, linewidth=1.2, linestyle=(0, (1, 2)), zorder=1)
        for b in px.t_break.values:
            if not np.isnat(b):
                ax.axvline(_years([b])[0], color=mf.BLUE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.set_title("AB"[k] + (": the clearing" if name == "cleared" else ": the latest observations"))
        ax.set_ylim(0, 1)
        ax.set_xlim(2020.2, 2026.9)
        mf.year_axis(ax, 1)
        note(f"ccdc_pixel_{name}", [str(b)[:10] for b in cc.t_break.values if not np.isnat(b)])
    axes[0].set_ylabel("NDVI")
    h, lab = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color=mf.ORANGE, linestyle=(0, (1, 2))), Line2D([], [], color=mf.BLUE, linestyle=(0, (4, 3)))]
    lab += ["CCDC's breaks", "S-CCD's breaks"]
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=9.5)
    mf.save(fig, "sccd_vs_ccdc")

    # agreement over the window
    seg, cc = d["seg"], d["cc"]
    nb_s = seg.n_breaks.values
    nb_c = cc.t_break.notnull().sum("segment").values
    note("same_n_breaks", round(float(np.mean(nb_s == nb_c)), 3))
    note("share_with_break_sccd", round(float(np.mean(nb_s > 0)), 3))
    note("share_with_break_ccdc", round(float(np.mean(nb_c > 0)), 3))
    dates = d["cube"].time.values
    tail = dates[-6:]
    end_breaks = np.isin(cc.t_break.values, tail).any(axis=0)
    note("ccdc_breaks_in_last_6_dates", round(float(np.mean(end_breaks)), 3))
    note("ccdc_last6_and_sccd_anomalies", round(float(np.mean(seg.n_anomalies.values[end_breaks] > 0)), 3))
    # CCDC's breaks (before the last 6 dates) with one of S-CCD's within one clear observation
    idx = {t: i for i, t in enumerate(dates)}
    tb_c, tb_s = cc.t_break.values, seg.t_break.values
    clear_rank = np.cumsum(d["qa"] < 2, axis=0)
    hit = tot = 0
    rows, cols = np.nonzero(nb_c > 0)
    for r, c in zip(rows, cols):
        rank = clear_rank[:, r, c]
        s_ranks = [rank[idx[b]] for b in tb_s[:, r, c] if not np.isnat(b)]
        for b in tb_c[:, r, c]:
            if np.isnat(b) or b in tail:
                continue
            tot += 1
            hit += any(abs(rank[idx[b]] - x) <= 1 for x in s_ranks)
    note("ccdc_breaks_found_by_sccd_within_1", round(hit / max(tot, 1), 3))


# ---------------------------------------------------------------------------
FIGURES = {
    "sccd_pixels": fig_sccd_pixels,
    "sccd_maps": fig_sccd_maps,
    "sccd_vs_ccdc": fig_sccd_vs_ccdc,
}

# Gallery thumbnail: a 16:9 crop of the most telling part of a figure, as in make_figures.py.
THUMBS = {
    "sccd": ("sccd_pixels.png", (0.5, 0.0, 1.0, 0.48)),
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
    mf.setup_style()
    d = data()
    if d is None:
        print("skipped (set ZEIT_DOCS_SCCD)")
        return
    for name in names or list(FIGURES):
        print(name)
        FIGURES[name](d)
    make_thumbs()
    print(json.dumps(NUMBERS, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
