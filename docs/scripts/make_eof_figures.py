"""Generate the figures of the EOF and lagged correlation tutorial (docs/tutorials/eof.md).

Every figure is produced by running ``zeit.eof`` and ``zeit.lagged_correlation``, so the numbers quoted in the tutorial
can be checked by running this script. From the repository root:

    python docs/scripts/make_eof_figures.py                # every figure
    python docs/scripts/make_eof_figures.py eof_modes      # a subset

Input: a 16-day EVI series of 575 dates (2001-2025) over a 170 x 170 km area of central Europe, 677 x 680 pixels of
250 m (EPSG:3035; bands named ``EVI_2001_01_01`` ...), averaged to months here. Point ``ZEIT_DOCS_EVI`` at the GeoTIFF
(the figures are skipped when it is missing). The EOF of the tutorial is of the monthly means; the driver of the lagged
correlation is the mean EVI of a 30 km block of the area (a regional series, as a climate index would be).

The style, the ``save`` helper and the thumbnails follow ``make_figures.py``.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402

OUT = mf.OUT
NUMBERS = {}
RDBU = LinearSegmentedColormap.from_list("rdbu", ["#8c4a14", "#c98a4b", "#f0efec", "#5aa39a", "#0f5c55"])
_DATA = {}


def note(key, value):
    NUMBERS[key] = value
    print(f"  {key} = {value}")


def monthly():
    """The monthly means of the EVI series, ``(time, y, x)``."""
    if "evi" in _DATA:
        return _DATA["evi"]
    path = os.environ.get("ZEIT_DOCS_EVI")
    if not path or not os.path.exists(path):
        return None
    evi = zeit.load_raster(path)
    note("evi_shape", list(evi.shape))
    if "band" in evi.dims:                                    # one band per date: the dates are the time axis
        evi = evi.squeeze("band", drop=True)
    m = evi.resample(time="MS").mean()
    m = m.assign_coords(time=m.indexes["time"] + pd.Timedelta(days=14))           # mid-month
    _DATA["evi"] = m
    return m


def decomposition():
    if "eof" in _DATA:
        return _DATA["eof"]
    cube = monthly()
    if cube is None:
        return None
    t = time.perf_counter()
    ds = zeit.eof(cube, n_modes=10, weights="none", solver="exact", rotate=3, north_n="autocorrelation", n_jobs=8)
    note("eof_s", round(time.perf_counter() - t, 1))
    note("variance", [round(float(v), 4) for v in ds.variance.values])
    note("separable", [bool(v) for v in ds.separable.values])
    note("kept", [ds.attrs["n_pixels"], ds.attrs["n_dates"]])
    _DATA["eof"] = ds
    return ds


def years(ds):
    return pd.to_datetime(ds.time.values).year + (pd.to_datetime(ds.time.values).dayofyear - 1) / 365.25


def fig_eof_modes():
    ds = decomposition()
    if ds is None:
        return
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.28, wspace=0.12)
    x = years(ds)
    for k in range(3):
        ax = fig.add_subplot(gs[0, k])
        m = ds.eof.isel(mode=k).values
        lim = float(np.nanpercentile(np.abs(m), 99))
        im = ax.imshow(m, cmap=RDBU, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        v = float(ds.variance.values[k]) * 100
        ax.set_title(f"EOF {k + 1}: {v:.1f} % of the variance", fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03, shrink=0.8)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("EVI per standard deviation of the PC", fontsize=8.5)
        ax2 = fig.add_subplot(gs[1, k])
        ax2.plot(x, ds.pc.isel(mode=k).values, color=mf.BLUE, lw=1.2)
        ax2.axhline(0, color=mf.AXIS, lw=0.8)
        ax2.set_xlim(x.min(), x.max())
        ax2.set_title(f"PC {k + 1}", fontsize=10)
    mf.save(fig, "eof_modes", photo=True)


def fig_eof_spectrum():
    ds = decomposition()
    if ds is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw=dict(wspace=0.28))
    ax = axes[0]
    m = np.arange(1, 11)
    ev = ds.eigenvalue.values
    ax.errorbar(m, ev, yerr=ds.north_error.values, fmt="none", ecolor=mf.INK2, capsize=3, lw=1.2)
    sep = ds.separable.values.astype(bool)
    ax.scatter(m[sep], ev[sep], s=46, color=mf.BLUE, zorder=3, label="separable")
    ax.scatter(m[~sep], ev[~sep], s=46, facecolor="white", edgecolor=mf.ORANGE, lw=1.8, zorder=3,
               label="not separable from a neighbour")
    ax.set_yscale("log")
    ax.set_xticks(m)
    ax.set_xlabel("mode")
    ax.set_ylabel("eigenvalue (EVI²)")
    ax.set_title("North's rule of thumb")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.bar(m - 0.2, ds.variance.values * 100, width=0.4, color=mf.BLUE, label="EOF")
    vr = ds.variance_rotated.values * 100
    ax.bar(np.arange(1, len(vr) + 1) + 0.2, vr, width=0.4, color=mf.ORANGE, label="varimax of the first 3")
    ax.set_xticks(m)
    ax.set_xlabel("mode")
    ax.set_ylabel("% of the variance")
    ax.set_title("What the rotation redistributes")
    ax.legend(loc="upper right")
    mf.save(fig, "eof_spectrum")


def driver_block(cube):
    block = cube.isel(y=slice(60, 120), x=slice(60, 120))           # 60 x 60 pixels: 15 km
    s = block.mean(("y", "x")).to_series()
    s.name = "regional EVI"
    return s, block


def fig_lagcorr():
    cube = monthly()
    if cube is None:
        return
    drv, block = driver_block(cube)
    t = time.perf_counter()
    lc = zeit.lagged_correlation(cube, drv, lags=6, n_jobs=8)
    note("lagged_s", round(time.perf_counter() - t, 1))
    note("significant_share", round(float(lc.significant.mean()), 3))
    note("p_threshold", float(lc.attrs["p_threshold"]))
    lc0 = zeit.lagged_correlation(cube, drv, lags=6, effective_n=False, n_jobs=8)
    note("significant_share_independent_n", round(float(lc0.significant.mean()), 3))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), gridspec_kw=dict(wspace=0.08))
    r = lc.best_r.where(lc.significant)
    for ax, (da, cmap, vmin, vmax, label, title) in zip(axes, (
            (lc.best_r, RDBU, -1, 1, "r", "best correlation"),
            (lc.best_lag.where(lc.best_lag >= 0), "viridis", 0, 6, "lag (months)", "lag of the best correlation"),
            (r, RDBU, -1, 1, "r", "where it is significant (FDR 10 %)"))):
        im = ax.imshow(da.values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.03, shrink=0.8)
        cb.set_label(label, fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
    axes[0].add_patch(plt.Rectangle((60, 60), 60, 60, fill=False, ec=mf.INK, lw=1.2))
    mf.save(fig, "eof_lagcorr", photo=True)
    return lc


FIGURES = {
    "eof_modes": fig_eof_modes,
    "eof_spectrum": fig_eof_spectrum,
    "eof_lagcorr": fig_lagcorr,
}
THUMBS = {"eof": ("eof_modes.webp", (0.0, 0.0, 1.0, 0.6))}


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
