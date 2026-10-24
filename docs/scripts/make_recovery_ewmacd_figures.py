"""Generate the figures of the recovery and EWMACD tutorials (docs/tutorials/recovery.md
and docs/tutorials/ewmacd.md).

Every figure is produced by running ``zeit.recovery_metrics`` and ``zeit.ewmacd``, so the
numbers quoted in the tutorials can be checked by running this script. From the repository
root:

    python docs/scripts/make_recovery_ewmacd_figures.py                 # every figure
    python docs/scripts/make_recovery_ewmacd_figures.py ewmacd_pixels   # a subset

Inputs (the figures of a missing input are skipped):

* **Recovery**: the annual NDVI stack of Rondonia (1985-2024, NDVI x 10000, one band per
  year, 0 = no data) that ``make_figures.py`` uses for LandTrendr. Point
  ``ZEIT_DOCS_RONDONIA`` at that GeoTIFF.
* **EWMACD**: the sample data of Brooks et al. (2014), a 65 x 64 pixel stack of the
  tasseled cap angle index (x 1000) over loblolly pine stands in Alabama, 51 Landsat dates
  from 2005 to 2011 (Virginia Tech, doi:10.7294/W4WD3XHK). Point ``ZEIT_DOCS_EWMACD`` at a
  folder holding either the download as it comes ("Sample Data - Angle Index x1000
  Stack.tif" and "Temporal Distribution with DOY.csv") or one GeoTIFF per date named
  ``angle_YYYY-MM-DD.tif``.

The style, the ``save`` helper and the thumbnails follow ``make_figures.py``.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402

OUT = mf.OUT
NUMBERS = {}
SEED = 7

# Years to recovery: quick (light) to slow (dark); "not recovered" in a warm grey-brown.
Y2R = LinearSegmentedColormap.from_list("y2r", ["#c9ead9", "#7cc8a4", "#2f9c74", "#16674f", "#0b3a2f"])
NOT_RECOVERED = "#c98a4b"
NO_EVENT = "#e4e3dd"
STATE_COLOURS = ["#e4e3dd", mf.ORANGE, mf.AQUA, mf.BLUE]   # stable, disturbed, recovered, gain


def note(key, value):
    NUMBERS[key] = value
    print(f"  {key} = {value}")


# ---------------------------------------------------------------------------
# Recovery after disturbance: LandTrendr on the Rondonia NDVI stack
# ---------------------------------------------------------------------------
_REC = {}
REC_WINDOW = (900, 1500, 500, 1100)   # rows and columns of the 18 x 18 km window shown


def rondonia():
    """LandTrendr, its events and their recovery metrics, as in the tutorial (cached)."""
    if _REC:
        return _REC
    path = os.environ.get("ZEIT_DOCS_RONDONIA")
    if not path or not os.path.exists(path):
        return None
    ndvi = zeit.load_raster(path, validate="landtrendr")
    years = ndvi.time.dt.year.values
    lt = zeit.landtrendr(ndvi)
    loss = zeit.extract_events(lt, min_magnitude=2000)
    confident = (loss.yod > years[0]) & (loss.dsnr >= 3)
    rec = zeit.recovery_metrics(lt, loss.where(confident))
    _REC.update(ndvi=ndvi, years=years, lt=lt, loss=loss, rec=rec,
                valid=(ndvi.values != 0).all(axis=0))
    return _REC


def fig_recovery_maps(d):
    rec, years, valid = d["rec"], d["years"], d["valid"]
    last = int(years[-1])
    has = (rec.yod.values > 0) & valid
    # the metrics at the horizon need 5 years after the trough: keep events old enough
    seen = has & (rec.trough.values > 0) & (rec.trough.values <= last - 5)
    y2r = rec.y2r.values
    r80p = rec.r80p.values
    recovered = seen & np.isfinite(y2r)
    not_rec = seen & ~np.isfinite(y2r)
    note("recovery_events", int(seen.sum()))
    note("recovery_share_recovered", round(float(recovered.sum() / seen.sum()), 3))
    note("recovery_median_y2r", float(np.nanmedian(y2r[recovered])))
    note("recovery_median_r80p", round(float(np.nanmedian(r80p[seen])), 3))
    note("recovery_share_r80p_ge1", round(float(np.mean(r80p[seen] >= 1)), 3))

    r0, r1, c0, c1 = REC_WINDOW
    win = (slice(r0, r1), slice(c0, c1))
    note("recovery_window_share_recovered",
         round(float(recovered[win].sum() / max(seen[win].sum(), 1)), 3))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), gridspec_kw={"wspace": 0.1})
    base = np.where(valid, 1.0, np.nan)[win]
    grey = ListedColormap([NO_EVENT])

    ax = axes[0]
    im = ax.imshow(mf._masked(d["ndvi"].values[-1] / 1e4, valid)[win], cmap=mf.GREENS, vmin=0.2, vmax=0.9,
                   interpolation="nearest")
    mf.map_axes(ax, f"NDVI, {last}")
    mf.colorbar(fig, im, ax, "NDVI", ticks=[0.2, 0.4, 0.6, 0.8])

    ax = axes[1]
    ax.imshow(base, cmap=grey, interpolation="nearest")
    ax.imshow(np.where(not_rec, 1.0, np.nan)[win], cmap=ListedColormap([NOT_RECOVERED]),
              interpolation="nearest")
    im = ax.imshow(np.where(recovered, y2r, np.nan)[win], cmap=Y2R, vmin=1, vmax=20,
                   interpolation="nearest")
    mf.map_axes(ax, "Years to recovery (Y2R)")
    mf.colorbar(fig, im, ax, "Years from the trough to 80% of NDVI before the loss",
                ticks=[1, 5, 10, 15, 20])
    ax.legend(handles=[Patch(color=NOT_RECOVERED, label=f"Not recovered by {last}"),
                       Patch(color=NO_EVENT, label="No loss, or too recent")],
              loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2, fontsize=9, handlelength=1.2,
              columnspacing=1.0)

    ax = axes[2]
    ax.imshow(base, cmap=grey, interpolation="nearest")
    im = ax.imshow(np.where(seen, r80p, np.nan)[win], cmap=mf.DIVERGING, vmin=0.5, vmax=1.5,
                   interpolation="nearest")
    mf.map_axes(ax, "R80P, five years after the trough")
    mf.colorbar(fig, im, ax, "1 or more: recovered", ticks=[0.5, 0.75, 1.0, 1.25, 1.5])
    mf.save(fig, "recovery_maps", photo=True)


def _pick_recovery_pixels(d):
    rec, years, loss = d["rec"], d["years"], d["loss"]
    rng = np.random.default_rng(SEED)
    yod, y2r, mag = rec.yod.values, rec.y2r.values, rec.magnitude.values
    dur, dsnr = loss.duration.values, loss.dsnr.values
    last = d["ndvi"].values[-1]

    def pick(mask):
        rr, cc = np.nonzero(mask & d["valid"])
        i = rng.integers(len(rr))
        return int(rr[i]), int(cc[i])

    regrew = pick((yod >= 1995) & (yod <= 2002) & (dur == 1) & (mag > 3500) & (y2r >= 4)
                  & (y2r <= 8) & (dsnr > 6) & (last > 7800))
    pasture = pick((yod >= 1995) & (yod <= 2002) & (dur == 1) & (mag > 4000) & ~np.isfinite(y2r)
                   & (dsnr > 8) & (last < 6000))
    return [regrew, pasture]


def fig_recovery_pixels(d):
    ndvi, years = d["ndvi"], d["years"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3), sharey=True, gridspec_kw={"wspace": 0.06})
    for k, (ax, (r, c)) in enumerate(zip(axes, _pick_recovery_pixels(d))):
        pixel = ndvi[:, r, c]
        fit = zeit.landtrendr(pixel, fitted=True)
        m = zeit.recovery_metrics(fit, min_magnitude=2000)
        full = d["rec"].isel(y=r, x=c)
        assert int(m.yod) == int(full.yod) and np.allclose(float(m.r80p), float(full.r80p), equal_nan=True)
        pre, t0, dist = float(m.pre_val), int(m.trough), float(m.dist_val)
        start, target = int(m.yod), 0.8 * pre
        obs = pixel.values.astype(float)
        mf.obs_scatter(ax, years[obs > 0], obs[obs > 0] / 1e4, s=18)
        ax.plot(years, fit.fitted / 1e4, color=mf.BLUE, label="LandTrendr fit")
        ax.axhline(pre / 1e4, color=mf.INK2, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
        ax.axhline(target / 1e4, color=mf.ORANGE, linewidth=1.3, linestyle=(0, (2, 2)), zorder=1)
        ax.text(1984.5, pre / 1e4 + 0.012, "Before the loss (pre_val)", color=mf.INK2, fontsize=9, va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
        ax.text(1984.5, target / 1e4 + 0.012, "80% of it: recovered above", color=mf.ORANGE,
                fontsize=9, va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
        ax.axvspan(start, t0, color=mf.ORANGE, alpha=0.10, linewidth=0)
        ax.scatter([t0], [dist / 1e4], s=60, color=mf.ORANGE, edgecolor="white", linewidth=1.5, zorder=5,
                   label="Trough")
        h = t0 + 5
        post = float(m.post_val)
        ax.scatter([h], [post / 1e4], s=60, marker="D", color=mf.INK, edgecolor="white", linewidth=1.2,
                   zorder=5, label="post_val (years 4 and 5)")
        y2r = float(m.y2r)
        if np.isfinite(y2r):
            yr = t0 + int(y2r)
            ax.annotate("", xy=(yr, 0.28), xytext=(t0, 0.28),
                        arrowprops=dict(arrowstyle="<->", color=mf.AQUA, lw=1.6, shrinkA=0, shrinkB=0))
            ax.text((t0 + yr) / 2, 0.30, f"Y2R = {int(y2r)} years", color=mf.AQUA, ha="center",
                    va="bottom", fontsize=9.5, fontweight="semibold")
            ax.axvline(yr, color=mf.AQUA, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            title = f"Cleared in {start + 1}: Y2R {int(y2r)} years"
        else:
            title = f"Cleared in {start + 1}: not recovered"
        ax.set_title(f"{title}, R80P {float(m.r80p):.2f}, RRI {float(m.rri):.2f}")
        ax.set_ylim(0.2, 0.95)
        ax.set_xlim(1983, 2026)
        mf.year_axis(ax, 10)
        note(f"recovery_pixel{k}", dict(row=r, col=c, yod=start, trough=t0, pre_val=round(pre),
                                         dist_val=round(dist), post_val=round(post), y2r=y2r,
                                         r80p=round(float(m.r80p), 2), rri=round(float(m.rri), 2),
                                         ari=round(float(m.ari)), yryr=round(float(m.yryr))))
    axes[0].set_ylabel("NDVI")
    axes[0].legend(loc="lower right", fontsize=9)
    mf.save(fig, "recovery_pixels")


# ---------------------------------------------------------------------------
# EWMACD: the sample data of Brooks et al. (2014)
# ---------------------------------------------------------------------------
_EW = {}


def ewmacd_data():
    """The angle index cube (time, y, x) with its dates, and EWMACD's result (cached)."""
    if _EW:
        return _EW
    folder = os.environ.get("ZEIT_DOCS_EWMACD")
    if not folder or not os.path.isdir(folder):
        return None
    folder = Path(folder)
    if list(folder.glob("angle_*.tif")):
        cube = zeit.load_raster(str(folder / "angle_*.tif"))
    else:
        table = pd.read_csv(folder / "Temporal Distribution with DOY.csv")
        dates = pd.to_datetime(dict(year=table.Year, month=table.Month, day=table.Day))
        cube = zeit.load_raster(str(folder / "Sample Data - Angle Index x1000 Stack.tif"), dates=dates)
    res = zeit.ewmacd(cube, train_end=2009)
    _EW.update(cube=cube, res=res)
    return _EW


def _model_curve(px, t0, t1):
    """The harmonic model of one pixel's result, every day from t0 to t1."""
    days = pd.date_range(t0, t1, freq="D")
    w = 2 * np.pi * days.dayofyear.to_numpy() / 365.0
    c = px.coefficients.to_series()
    v = np.full(len(days), c["constant"])
    for k in range(1, int(px.attrs["order"]) + 1):
        v += c[f"sin{k}"] * np.sin(k * w) + c[f"cos{k}"] * np.cos(k * w)
    return np.asarray(mf.frac_year([x.toordinal() for x in days])), v


EWMACD_PIXELS = [(31, 42), (39, 50)]    # row, col: disturbed and recovered; harvested


def fig_ewmacd_pixels(d):
    cube = d["cube"]
    t = np.asarray(mf.frac_year([pd.Timestamp(x).toordinal() for x in cube.time.values]))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.12, "wspace": 0.14})
    for k, (r, c) in enumerate(EWMACD_PIXELS):
        px = zeit.ewmacd(cube.isel(y=r, x=c), train_end=2009)
        top, bot = axes[0, k], axes[1, k]
        obs = px.ewma.notnull().values | (t >= 2009)
        kept = px.ewma.notnull().values
        v = cube.isel(y=r, x=c).values.astype(float)
        tm, vm = _model_curve(px, "2005-01-01", "2011-12-31")
        for ax in (top, bot):
            ax.axvspan(2005, 2009, color=mf.GRID, alpha=0.6, linewidth=0, zorder=0)
        top.plot(tm, vm, color=mf.BLUE, linewidth=1.4, label="Harmonic model (training years)")
        mf.obs_scatter(top, t[kept], v[kept], s=22, label="Observations")
        top.scatter(t[~kept], v[~kept], s=22, facecolor="white", edgecolor=mf.OBS, linewidth=1.1, zorder=3,
                    label="Screened (X-bar)")
        z, lim, flags = px.ewma.values, px.limit.values, px.flags.values
        bot.fill_between(t[kept], -lim[kept], lim[kept], color=mf.BLUE, alpha=0.10, linewidth=0,
                         label="Control limits")
        bot.plot(t[kept], z[kept], color=mf.INK2, linewidth=1.3, zorder=2)
        bot.scatter(t[kept], z[kept], s=18, color=mf.INK2, edgecolor="white", linewidth=0.6, zorder=3,
                    label="EWMA of the residuals")
        out = kept & (flags < 0)
        bot.scatter(t[out], z[out], s=30, color=mf.ORANGE, edgecolor="white", linewidth=0.8, zorder=4,
                    label="Flag below 0 (disturbed)")
        bt, rt = float(px.break_time), float(px.recovery_time)
        for ax in (top, bot):
            mf.break_line(ax, bt, label="First disturbance" if ax is bot else None)
            if np.isfinite(rt):
                ax.axvline(rt, color=mf.AQUA, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1,
                           label="Recovered" if ax is bot else None)
        bot.axhline(0, color=mf.AXIS, linewidth=0.8, zorder=1)
        state = px.state.attrs["flag_meanings"].split()[int(px.state)]
        when = pd.Timestamp(str(cube.time.values[np.argmin(np.abs(t - bt))])).strftime("%B %Y")
        if np.isfinite(rt):
            back = pd.Timestamp(str(cube.time.values[np.argmin(np.abs(t - rt))])).strftime("%B %Y")
            top.set_title(f"Out of control from {when}, back in {back}")
        else:
            top.set_title(f"Out of control from {when}, still in October 2011")
        top.text(2005.1, 0.97, "Training years", transform=top.get_xaxis_transform(), color=mf.INK2,
                 fontsize=9, va="top")
        top.set_xlim(2004.9, 2012.0)
        mf.year_axis(top, 1)
        note(f"ewmacd_pixel{k}", dict(row=r, col=c, break_time=round(bt, 3), recovery_time=round(rt, 3),
                                       severity=int(px.severity), state=state, sigma=round(float(px.sigma), 1),
                                       r2=round(float(px.r2), 3), screened=int((~kept & (t < 2009)).sum())))
    axes[0, 0].set_ylabel("Angle index (x1000)")
    axes[1, 0].set_ylabel("EWMA of the residuals")
    handles = []
    for ax in (axes[0, 0], axes[1, 0]):
        h, lab = ax.get_legend_handles_labels()
        handles += list(zip(h, lab))
    fig.legend([h for h, _ in handles], [lab for _, lab in handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.055), ncol=4, fontsize=9.5)
    for ax in axes[1]:
        ax.set_ylim(-260, 90)
    mf.save(fig, "ewmacd_pixels")


def fig_ewmacd_maps(d):
    cube, res = d["cube"], d["res"]
    sev, st, bt = res.severity.values, res.state.values, res.break_time.values
    names = res.state.attrs["flag_meanings"].split()
    counts = {n: int((st == i).sum()) for i, n in enumerate(names)}
    note("ewmacd_states", counts)
    note("ewmacd_pixels", int(np.isfinite(st).sum()))
    note("ewmacd_share_break_spring_2009", round(float(np.mean((bt < 2009.3)[np.isfinite(bt)])), 3))
    note("ewmacd_share_sev_le3", round(float(np.mean(sev <= -3)), 3))
    yrs = pd.DatetimeIndex(cube.time.values).year
    flags = res.flags.values
    note("ewmacd_share_flag_le3_last", round(float(np.mean(flags[-1] <= -3)), 3))
    note("ewmacd_share_flag_le3_2009", round(float(np.mean((flags[yrs == 2009] <= -3).any(0))), 3))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), gridspec_kw={"wspace": 0.06})
    last = cube.isel(time=-1)
    ax = axes[0]
    im = ax.imshow(last.values, cmap=mf.GREENS, vmin=200, vmax=700, interpolation="nearest")
    day = pd.Timestamp(str(last.time.values))
    mf.map_axes(ax, f"Angle index, {day.day} {day:%B %Y}")
    mf.colorbar(fig, im, ax, "Tasseled cap angle (x1000)", ticks=[200, 400, 600])

    ax = axes[1]
    ax.imshow(np.ones_like(sev), cmap=ListedColormap([NO_EVENT]), interpolation="nearest")
    strong = sev <= -3
    im = ax.imshow(np.where(strong, bt, np.nan), cmap=mf.YEARS, vmin=2008.6, vmax=2012,
                   interpolation="nearest")
    mf.map_axes(ax, "First disturbance, severity 3 or more")
    mf.colorbar(fig, im, ax, "Date of the first negative flag", ticks=[2009, 2010, 2011, 2012])

    ax = axes[2]
    ax.imshow(st, cmap=ListedColormap(STATE_COLOURS), vmin=-0.5, vmax=3.5, interpolation="nearest")
    mf.map_axes(ax, "State at the last date (October 2011)")
    ax.legend(handles=[Patch(color=col, label=n) for col, n in zip(STATE_COLOURS, names)],
              loc="upper left", bbox_to_anchor=(0.0, -0.03), ncol=4, fontsize=9, handlelength=1.2,
              columnspacing=1.0)
    mf.save(fig, "ewmacd_maps", photo=True)


# ---------------------------------------------------------------------------
FIGURES = {
    "recovery_maps": (fig_recovery_maps, rondonia, "ZEIT_DOCS_RONDONIA"),
    "recovery_pixels": (fig_recovery_pixels, rondonia, "ZEIT_DOCS_RONDONIA"),
    "ewmacd_pixels": (fig_ewmacd_pixels, ewmacd_data, "ZEIT_DOCS_EWMACD"),
    "ewmacd_maps": (fig_ewmacd_maps, ewmacd_data, "ZEIT_DOCS_EWMACD"),
}

# Gallery thumbnails: a 16:9 crop of the most telling part of a figure, as in make_figures.py.
THUMBS = {
    "recovery": ("recovery_pixels.png", (0.0, 0.0, 0.5, 1.0)),
    "ewmacd": ("ewmacd_maps.webp", (0.345, 0.0, 1.0, 0.835)),
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
    for name in names or list(FIGURES):
        fn, load, env = FIGURES[name]
        print(name)
        d = load()
        if d is None:
            print(f"  skipped (set {env})")
            continue
        fn(d)
    make_thumbs()
    print(json.dumps(NUMBERS, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
