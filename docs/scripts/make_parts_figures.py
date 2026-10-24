"""Generate the figures of the PARTS tutorials (docs/tutorials/regional-trends/).

Every figure is produced by running ``zeit.ar_trend`` and ``zeit.parts`` (and, for
the empirical correlations, the core functions they call), so the numbers quoted in
the tutorials can be checked by running this script. From the repository root:

    python docs/scripts/make_parts_figures.py              # every figure
    python docs/scripts/make_parts_figures.py calibration  # a subset

Inputs:

* **Synthetic cubes** (``problem``, ``correlation``, ``partitions``, ``calibration``)
  are simulated here with a fixed seed: a trend (0 unless stated), AR(1) errors in
  time and innovations that are Gaussian fields with an exponential spatial
  correlation, exp(-d / range), plus a nugget. They need nothing but Zeit.
* **Real data** (``pixel``, ``maps``, ``correlation``, ``partitions``, ``classes``,
  ``tuning``): 41 annual Landsat medoid composites of a vegetation index, 1985-2025,
  one GeoTIFF per year (``landsat_medoid_YYYY.tif``, EPSG:4326, 30 m, 0 outside the
  data and over the sea), over north-eastern Brazil (Sergipe and Alagoas). Point
  ``ZEIT_DOCS_PARTS`` at that folder; the real-data figures are skipped when it is
  not set. Two cubes are made from it: a 21 x 21 km window at 30 m and the whole
  scene at 300 m, both reprojected to SIRGAS 2000 / UTM 24S (EPSG:31984).

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (style, colours, save)

import zeit  # noqa: E402
from zeit import _core  # noqa: E402

OUT = mf.OUT
NUMBERS = {}
SEED = 7
RANGE_PX = 15  # the simulated range, in 30 m pixels: 450 m

# the window of the worked example, in the source grid's pixels (x0, y0, x1, y1)
ORIGIN, STEP = (-38.416992114541813, -9.137483407006950), 0.000269494585236
WINDOW = (1000, 500, 1700, 1200)
UTM = "EPSG:31984"

RHO = LinearSegmentedColormap.from_list("rho", ["#f4f1e4", "#b9b3d9", "#6f5fb5", "#2e2270"])
SIG = ListedColormap(["#8c4a14", "#ecebe6", "#0f5c55"])


def note(key, value):
    NUMBERS[key] = value
    print(f"  {key} = {value}")


# ---------------------------------------------------------------------------
# Synthetic cubes with a known spatial correlation
# ---------------------------------------------------------------------------
def fields(rng, n, range_px, k):
    """k independent n x n Gaussian fields of unit variance whose correlation is
    exp(-d / range_px), d in pixels (circulant embedding on a 2n x 2n torus)."""
    m = 2 * n
    i = np.minimum(np.arange(m), m - np.arange(m))
    d = np.hypot(i[:, None], i[None, :])
    lam = np.clip(np.fft.fft2(np.exp(-d / range_px)).real, 0, None)
    out = []
    while len(out) < k:
        e = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
        f = np.fft.fft2(np.sqrt(lam / (m * m)) * e)
        out += [f.real[:n, :n], f.imag[:n, :n]]
    return np.array(out[:k])


def synthetic(seed, n=60, nt=30, slope=0.0, rho=0.3, range_px=RANGE_PX, nugget=0.2, sd=0.03, spacing=30.0):
    """A projected (time, y, x) cube: 0.5 + slope * t + sd * e, with e AR(1) in time
    (rho) and its innovations spatially correlated (range_px pixels, a nugget)."""
    rng = np.random.default_rng(seed)
    f = fields(rng, n, range_px, nt)
    innov = np.sqrt(1 - nugget) * f + np.sqrt(nugget) * rng.standard_normal(f.shape)
    e = np.empty_like(innov)
    e[0] = innov[0] / np.sqrt(1 - rho ** 2)
    for t in range(1, nt):
        e[t] = rho * e[t - 1] + innov[t]
    values = 0.5 + slope * np.arange(nt)[:, None, None] + sd * e
    cube = xr.DataArray(values, dims=("time", "y", "x"), coords={
        "time": pd.date_range("1995-01-01", periods=nt, freq="YS"),
        "y": 8_900_000.0 - spacing * np.arange(n), "x": 600_000.0 + spacing * np.arange(n)})
    return cube.rio.write_crs(UTM)


def naive_p(slopes):
    """The t-test of the mean of the pixels' slopes as if they were independent."""
    from scipy import stats
    s = np.asarray(slopes, dtype=float)
    s = s[np.isfinite(s)]
    return float(stats.ttest_1samp(s, 0.0).pvalue)


def residuals(cube, idx):
    """The AR(1) fit's residual series of the pixels idx (flat indices), as zeit.parts
    takes them to fit the spatial correlation."""
    nt = cube.sizes["time"]
    design = np.column_stack([np.ones(nt), np.arange(1, nt + 1, dtype=float)])
    series = cube.values.reshape(nt, -1)[:, idx].T
    return _core.parts.ar_residuals(np.ascontiguousarray(series, dtype=np.float64), design)


def pair_correlations(cube, n_sample, seed):
    """Distances and correlations of the residual series of every pair of a sample."""
    xs, ys = np.meshgrid(cube.x.values, cube.y.values)
    ok = np.flatnonzero(np.isfinite(cube.values).all(axis=0).ravel())
    idx = np.random.default_rng(seed).choice(ok, size=min(n_sample, ok.size), replace=False)
    r = residuals(cube, idx)
    c = np.corrcoef(r)
    x, y = xs.ravel()[idx], ys.ravel()[idx]
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    iu = np.triu_indices(len(idx), 1)
    return d[iu], c[iu]


def binned(d, c, edges):
    mid, mean = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() > 20:
            mid.append(d[m].mean())
            mean.append(c[m].mean())
    return np.array(mid), np.array(mean)


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------
_REAL = {}


def real():
    """The window (30 m) and the scene (300 m) of the annual series, their trends, and
    the scene's land mask and zones (cached)."""
    if _REAL:
        return _REAL
    folder = os.environ.get("ZEIT_DOCS_PARTS")
    if not folder:
        return None
    x0, y0, x1, y1 = WINDOW
    bounds = (ORIGIN[0] + x0 * STEP, ORIGIN[1] - y1 * STEP, ORIGIN[0] + x1 * STEP, ORIGIN[1] - y0 * STEP)
    t = time.time()
    geo = zeit.load_raster(folder, clip=bounds)              # lon/lat, 30 m
    win = zeit.load_raster(geo, crs=UTM, res=30)              # UTM 24S, 30 m
    print(f"  window {win.shape} read in {time.time() - t:.0f} s")
    t = time.time()
    scene = zeit.load_raster(folder, crs=UTM, res=300, resampling="average")
    scene = scene.where(scene != 0)                            # 0: outside the data
    print(f"  scene {scene.shape} read in {time.time() - t:.0f} s")
    from scipy import ndimage
    # the source's footprint on the UTM grid (its corners fall outside the rotated rectangle)
    ny, nx = 7319, 7478
    fp = xr.DataArray(np.ones((ny // 10, nx // 10), dtype="float32"), dims=("y", "x"), coords={
        "y": ORIGIN[1] - STEP * ny * (np.arange(ny // 10) + 0.5) / (ny // 10),
        "x": ORIGIN[0] + STEP * nx * (np.arange(nx // 10) + 0.5) / (nx // 10)})
    fp = fp.rio.write_crs("EPSG:4326").rio.write_nodata(np.nan)
    inside = fp.rio.reproject_match(scene.isel(time=0)).notnull().values
    mean = scene.mean("time")
    water = inside & ((mean < 0.05) | (scene.notnull().sum("time") < 30)).values
    labels, _ = ndimage.label(water)
    rows, cols = np.nonzero(water)
    k = np.argmax(cols + rows)                                 # the south-easternmost water: the sea
    sea = labels == labels[rows[k], cols[k]]
    dist = ndimage.distance_transform_edt(~sea, sampling=300.0) / 1000.0
    land = inside & ~water
    zones = np.where(land, np.where(dist < 40, 1, np.where(dist < 100, 2, 3)), np.nan)
    like = dict(dims=("y", "x"), coords={"y": scene.y, "x": scene.x})
    _REAL.update(
        win=win, ar_win=zeit.ar_trend(win),
        scene=scene, ar_scene=zeit.ar_trend(scene, min_valid=30),
        land=xr.DataArray(land, **like), zones=xr.DataArray(zones, **like),
        coast_km=xr.DataArray(np.where(land, dist, np.nan), **like))
    return _REAL


def parts_window(d, npart=100):
    key = f"res_win_{npart}"
    if key not in d:
        t = time.time()
        d[key] = zeit.parts(d["win"], trend=d["ar_win"], npart=npart)
        d[key + "_s"] = time.time() - t
    return d[key]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_problem():
    """A map with no trend at all: the pixels that test significant, and the tests."""
    cube = synthetic(SEED, n=100)
    ar = zeit.ar_trend(cube)
    s, p = ar.slope.values, ar.slope_p.values
    res = zeit.parts(cube, trend=ar, partsize=1000)
    up, down = np.mean((p < .05) & (s > 0)), np.mean((p < .05) & (s < 0))
    pn, pp = naive_p(s), float(res.table.loc["mean", "p"])
    note("problem_share_up", round(100 * up, 1))
    note("problem_share_down", round(100 * down, 1))
    note("problem_naive_p", f"{pn:.2g}")
    note("problem_parts_p", round(pp, 3))
    note("problem_parts_range_m", round(res.correlation["range"]))

    fig = plt.figure(figsize=(13.5, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.12)
    axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
    lim = np.nanpercentile(np.abs(s), 98)
    im = axes[0].imshow(s, cmap=mf.DIVERGING, vmin=-lim, vmax=lim)
    mf.map_axes(axes[0], "Each pixel's AR(1) trend")
    tick = float(f"{0.6 * lim:.1g}")
    mf.colorbar(fig, im, axes[0], "slope per year (the true trend is 0)", ticks=[-tick, 0, tick])
    cat = np.where(p < .05, np.sign(s), 0) + 1
    axes[1].imshow(cat, cmap=SIG, vmin=0, vmax=2, interpolation="nearest")
    mf.map_axes(axes[1], f"p < 0.05: {100 * (up + down):.1f}% of the pixels")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=SIG.colors[2], label=f"increasing, {100 * up:.1f}%"),
                            Patch(color=SIG.colors[0], label=f"decreasing, {100 * down:.1f}%")],
                   loc="upper left", bbox_to_anchor=(0, -0.02), ncol=1, handletextpad=0.4)

    ax = axes[2]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Is the mean trend of the map different from 0?")
    mant, expo = f"{pn:.1e}".split("e")
    rows = [(0.66, mf.ORANGE, "t-test of the 10,000 slopes,\nas if the pixels were independent",
             f"p = {mant} × 10$^{{{int(expo)}}}$", "\"yes\": wrong"),
            (0.2, mf.BLUE, "PARTS", f"p = {pp:.2f}", "\"no evidence\": right")]
    for yv, colr, name, ptxt, verdict in rows:
        ax.add_patch(plt.Rectangle((0, yv - 0.15), 1, 0.36, color=colr, alpha=0.1, lw=0))
        ax.add_patch(plt.Rectangle((0, yv - 0.15), 0.012, 0.36, color=colr, lw=0))
        ax.text(0.05, yv + 0.1, name, fontsize=10, color=mf.INK, va="center", linespacing=1.3)
        ax.text(0.05, yv - 0.07, ptxt, fontsize=15, color=colr, va="center", weight="bold")
        ax.text(0.97, yv - 0.07, verdict, fontsize=10, color=mf.INK2, va="center", ha="right")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2)
    mf.save(fig, "parts_problem", photo=True)


def fig_pixel(d):
    """A pixel's series, its OLS line and its AR(1) REML line, and the residuals."""
    cube, ar = d["win"], d["ar_win"]
    nt = cube.sizes["time"]
    t = np.arange(1, nt + 1, dtype=float)
    years = pd.DatetimeIndex(cube.time.values).year.values
    v = cube.values.reshape(nt, -1)
    ok = np.isfinite(v).all(axis=0)
    # OLS of every complete pixel
    tc = t - t.mean()
    b = (tc[:, None] * (v - v.mean(axis=0))).sum(axis=0) / (tc ** 2).sum()
    resid = v - v.mean(axis=0) - b * tc[:, None]
    se = np.sqrt((resid ** 2).sum(axis=0) / (nt - 2) / (tc ** 2).sum())
    from scipy import stats
    p_ols = 2 * stats.t.sf(np.abs(b / se), nt - 2)
    rho, p_ar = ar.rho.values.ravel(), ar.slope_p.values.ravel()
    # a pixel whose OLS test says "significant" and whose AR(1) test does not, with a
    # moderate autocorrelation: the most significant by OLS of them
    pick = ok & (p_ols < 0.01) & (p_ar > 0.05) & (rho > 0.4) & (rho < 0.6)
    i = np.flatnonzero(pick)[np.argmin(p_ols[pick])]
    y = v[:, i]
    a_ar = ar.intercept.values.ravel()[i]
    b_ar, se_ar = ar.slope.values.ravel()[i], ar.slope_se.values.ravel()[i]
    row, col = np.unravel_index(i, (cube.sizes["y"], cube.sizes["x"]))
    note("pixel_xy_utm", [float(cube.x[col]), float(cube.y[row])])
    note("pixel_ols", dict(slope=round(float(b[i]), 5), se=round(float(se[i]), 5), p=f"{p_ols[i]:.2g}"))
    note("pixel_ar", dict(slope=round(float(b_ar), 5), se=round(float(se_ar), 5), p=round(float(p_ar[i]), 3),
                          rho=round(float(rho[i]), 2)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    mf.obs_scatter(ax, years, y, s=22)
    ols_line = y.mean() + b[i] * tc
    ax.plot(years, ols_line, color=mf.ORANGE, lw=1.8, ls=(0, (5, 3)),
            label=f"OLS: {b[i]:+.4f} ± {se[i]:.4f} /yr, p = {p_ols[i]:.2g}")
    ax.plot(years, a_ar + b_ar * t, color=mf.BLUE, lw=2.2,
            label=f"AR(1) REML: {b_ar:+.4f} ± {se_ar:.4f} /yr, p = {p_ar[i]:.2f}")
    ax.set_ylabel("vegetation index")
    ax.set_title("One pixel, 41 years: same slope, different certainty")
    mf.legend_above(ax, ncol=1)
    ax = axes[1]
    r = y - (a_ar + b_ar * t)
    ax.bar(years, r, color=np.where(r > 0, mf.AQUA, mf.ORANGE), width=0.8)
    ax.axhline(0, color=mf.AXIS, lw=1)
    ax.set_title(f"Residuals come in runs: rho = {rho[i]:.2f}")
    ax.set_ylabel("observed − trend")
    fig.tight_layout()
    mf.save(fig, "parts_pixel_fit")


def fig_maps(d):
    """The window's per-pixel AR(1) trends: slope, significance, rho."""
    ar = d["ar_win"]
    s, p, rho = ar.slope.values, ar.slope_p.values, ar.rho.values
    n = np.isfinite(s).sum()
    up, down = np.nansum((p < .05) & (s > 0)) / n, np.nansum((p < .05) & (s < 0)) / n
    note("window_pixels", int(n))
    note("window_shape", list(ar.slope.shape))
    note("window_share_up", round(100 * up, 1))
    note("window_share_down", round(100 * down, 2))
    note("window_mean_slope", round(float(np.nanmean(s)), 5))
    note("window_median_rho", round(float(np.nanmedian(rho)), 2))
    one = zeit.ar_trend(d["win"].mean(("y", "x")).values)  # the series of the window's mean
    note("window_mean_series", dict(slope=round(float(one.slope), 5), se=round(float(one.slope_se), 5),
                                    p=round(float(one.slope_p), 3), rho=round(float(one.rho), 2)))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    im = axes[0].imshow(s, cmap=mf.DIVERGING, vmin=-0.006, vmax=0.006)
    mf.map_axes(axes[0], "Slope")
    mf.colorbar(fig, im, axes[0], "index per year", ticks=[-0.006, -0.003, 0, 0.003, 0.006])
    cat = np.where(np.isfinite(s), np.where(p < .05, np.sign(s), 0) + 1, np.nan)
    axes[1].imshow(cat, cmap=SIG, vmin=0, vmax=2, interpolation="nearest")
    mf.map_axes(axes[1], f"p < 0.05: {100 * up:.0f}% up, {100 * down:.1f}% down")
    im = axes[2].imshow(rho, cmap=RHO, vmin=-0.2, vmax=0.8)
    mf.map_axes(axes[2], "AR(1) rho")
    mf.colorbar(fig, im, axes[2], "temporal autocorrelation of the residuals")
    fig.tight_layout()
    mf.save(fig, "parts_trend_maps", photo=True)


def fig_correlation(d):
    """Correlation of the residual series against distance, and the fitted exp(-d/range)."""
    cube = synthetic(SEED, n=100)
    res = zeit.parts(cube, partsize=1000)
    dist, cor = pair_correlations(cube, 800, 1)
    true = 30.0 * RANGE_PX
    panels = [(dist, cor, res.correlation["range"], true, "m", f"Simulated: true range {true:.0f} m, nugget 0.2")]
    note("corr_synthetic_range_m", round(res.correlation["range"]))
    note("corr_synthetic_nugget", round(float(np.nanmean(res.nuggets)), 3))
    if d is not None:
        rw = parts_window(d)
        dw, cw = pair_correlations(d["win"], 800, 1)
        panels.append((dw / 1000, cw, rw.correlation["range"] / 1000, None, "km",
                       "Real: the 21 km window, 41 years"))
        note("corr_window_range_km", round(rw.correlation["range"] / 1000, 1))
        note("corr_window_nugget", round(float(np.nanmean(rw.nuggets)), 3))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.4), squeeze=False)
    for ax, (dd, cc, rng_fit, rng_true, unit, title) in zip(axes[0], panels):
        ax.hexbin(dd, cc, gridsize=60, cmap=LinearSegmentedColormap.from_list("h", ["#ffffff", "#cfcdc4", "#8d8b84"]),
                  mincnt=1, bins="log", linewidths=0)
        edges = np.linspace(0, dd.max(), 30)
        mx, my = binned(dd, cc, edges)
        ax.plot(mx, my, "o", color=mf.INK, ms=4.5, label="mean of the pairs at that distance")
        xs = np.linspace(0, dd.max(), 300)
        ax.plot(xs, np.exp(-xs / rng_fit), color=mf.BLUE, lw=2.4,
                label=f"fitted exp(−d / {rng_fit:.3g} {unit})")
        if rng_true:
            ax.plot(xs, 0.8 * np.exp(-xs / rng_true), color=mf.ORANGE, lw=1.6, ls=(0, (4, 3)),
                    label=f"true correlation, 0.8 exp(−d / {rng_true:.0f} m)")
        ax.axhline(0, color=mf.AXIS, lw=1)
        ax.set_ylim(-0.6, 1.05)
        ax.set_xlabel(f"distance between the two pixels ({unit})")
        ax.set_ylabel("correlation of their residual series")
        ax.set_title(title)
        ax.legend(loc="upper right")
    fig.tight_layout()
    mf.save(fig, "parts_correlation")


def fig_partitions(d):
    """Random partitions of the pixels, and the partitions' estimates combined."""
    n = 60
    pm = np.asarray(_core.parts.sample_partitions(n * n, 600, 6, 3))
    yy, xx = np.divmod(np.arange(n * n), n)
    has_real = d is not None
    fig, axes = plt.subplots(1, 2 if has_real else 1, figsize=(13, 4.8) if has_real else (6, 5), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.5]} if has_real else None)
    ax = axes[0][0]
    ax.scatter(xx, yy, s=3, color="#e6e5df", linewidths=0)
    for k, colr in zip(range(3), [mf.BLUE, mf.ORANGE, mf.AQUA]):
        ax.scatter(xx[pm[k]], yy[pm[k]], s=7, color=colr, linewidths=0, label=f"partition {k + 1}")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    mf.map_axes(ax, "3 of 6 random partitions of 600 pixels")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.13), ncol=3, handletextpad=0.2, markerscale=2)
    if has_real:
        res = parts_window(d)
        est = res.partitions["mean"].values
        m, se = float(res.table.loc["mean", "estimate"]), float(res.table.loc["mean", "se"])
        naive = est.std(ddof=1) / np.sqrt(len(est))
        note("partitions_window", dict(npart=res.npart, partsize=res.partsize, estimate=round(m, 5),
                                       se=round(se, 5), p=round(float(res.table.loc["mean", "p"]), 3),
                                       sd_partitions=round(float(est.std(ddof=1)), 5),
                                       naive_se=round(float(naive), 6), seconds=round(d["res_win_100_s"], 1)))
        ax = axes[0][1]
        ax.axvspan(m - 1.96 * se, m + 1.96 * se, color=mf.BLUE, alpha=0.13, lw=0,
                   label=f"PARTS 95% interval (SE {se:.4f})")
        ax.axvspan(m - 1.96 * naive, m + 1.96 * naive, color=mf.ORANGE, alpha=0.5, lw=0,
                   label=f"as if the partitions were independent (SE {naive:.5f})")
        jitter = np.random.default_rng(0).uniform(-0.35, 0.35, len(est))
        ax.scatter(est, jitter, s=16, color=mf.INK2, alpha=0.75, linewidths=0, label="one partition's GLS estimate")
        ax.axvline(m, color=mf.BLUE, lw=2)
        ax.axvline(0, color=mf.INK, lw=1, ls=(0, (4, 3)))
        ax.set_ylim(-1.2, 1.4)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("mean trend of the window (index per year)")
        ax.set_title(f"{res.npart} partitions of {res.partsize} pixels: the estimates and their mean")
        ax.legend(loc="upper left", fontsize=8.8)
    fig.tight_layout()
    mf.save(fig, "parts_partitions")


def fig_calibration():
    """Under no trend: p-values of the naive test and of PARTS, the share of significant
    pixels; with a trend: how often each detects it."""
    reps = 300
    rows = []
    t0 = time.time()
    for s in range(reps):
        cube = synthetic(10_000 + s)
        ar = zeit.ar_trend(cube)
        sl, p = ar.slope.values, ar.slope_p.values
        res = zeit.parts(cube, trend=ar, partsize=600)
        known = zeit.parts(cube, trend=ar, partsize=600, range=30.0 * RANGE_PX)
        rows.append((np.mean(p < .05), naive_p(sl), float(res.table.loc["mean", "p"]),
                     float(known.table.loc["mean", "p"]), res.correlation["range"]))
    a = np.array(rows)
    note("calibration_reps", reps)
    note("calibration_seconds", round(time.time() - t0))
    note("calibration_naive_fpr", round(float(np.mean(a[:, 1] < .05)), 3))
    note("calibration_parts_fpr", round(float(np.mean(a[:, 2] < .05)), 3))
    note("calibration_parts_fpr_01", round(float(np.mean(a[:, 2] < .01)), 3))
    note("calibration_parts_fpr_true_range", round(float(np.mean(a[:, 3] < .05)), 3))
    note("calibration_fitted_range_median_m", round(float(np.median(a[:, 4]))))
    note("calibration_share_mean", round(float(100 * a[:, 0].mean()), 1))
    note("calibration_share_q95", round(float(100 * np.quantile(a[:, 0], 0.95)), 1))
    note("calibration_share_max", round(float(100 * a[:, 0].max()), 1))
    slopes = [0.0001, 0.0002, 0.0003, 0.0005, 0.0008]
    power = {"naive": [float(np.mean(a[:, 1] < .05))], "parts": [float(np.mean(a[:, 2] < .05))]}
    for b in slopes:
        hits_n, hits_p = [], []
        for s in range(100):
            cube = synthetic(20_000 + s, slope=b)
            ar = zeit.ar_trend(cube)
            res = zeit.parts(cube, trend=ar, partsize=600)
            hits_n.append(naive_p(ar.slope.values) < .05)
            hits_p.append(float(res.table.loc["mean", "p"]) < .05)
        power["naive"].append(float(np.mean(hits_n)))
        power["parts"].append(float(np.mean(hits_p)))
    note("power_slopes", [0.0] + slopes)
    note("power", power)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    ax.hist(a[:, 1], bins=bins, color=mf.ORANGE, alpha=0.85, label="t-test of the pixels' slopes")
    ax.hist(a[:, 2], bins=bins, color=mf.BLUE, alpha=0.85, label="PARTS")
    ax.hist(a[:, 3], bins=bins, histtype="step", color=mf.INK, lw=1.4, label="PARTS, the true range given")
    ax.axhline(reps / 20, color=mf.INK2, lw=1, ls=(0, (4, 3)))
    ax.text(1.0, reps / 20, "uniform", ha="right", va="bottom", color=mf.INK2, fontsize=9)
    ax.set_xlabel("p-value of \"the mean trend is 0\"")
    ax.set_ylabel(f"maps (of {reps})")
    ax.set_title(f"No trend: p < 0.05 in {100 * np.mean(a[:, 1] < .05):.0f}% vs "
                 f"{100 * np.mean(a[:, 2] < .05):.0f}% of maps")
    ax.legend(loc="upper center")
    ax = axes[1]
    ax.hist(100 * a[:, 0], bins=30, color=mf.OBS)
    ax.axvline(5, color=mf.INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(5.3, ax.get_ylim()[1] * 0.92, "5% expected", color=mf.INK2, fontsize=9)
    ax.set_xlabel("pixels with a significant trend (%)")
    ax.set_ylabel("maps")
    ax.set_title("No trend: the share of significant pixels")
    ax = axes[2]
    xs = [0.0] + slopes
    xs = [1e4 * v for v in xs]
    ax.plot(xs, power["naive"], "o-", color=mf.ORANGE, label="t-test of the pixels' slopes")
    ax.plot(xs, power["parts"], "o-", color=mf.BLUE, label="PARTS")
    ax.axhline(0.05, color=mf.INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("true trend (10⁻⁴ per year; noise SD 0.03)")
    ax.set_ylabel("share of maps where p < 0.05")
    ax.set_title("A real trend: how often it is detected")
    ax.legend(loc="lower right")
    fig.tight_layout()
    mf.save(fig, "parts_calibration")


def fig_classes(d):
    """The scene's zones by distance to the sea, and their mean trends."""
    scene, ar, zones, land = d["scene"], d["ar_scene"], d["zones"], d["land"]
    t = time.time()
    res_mean = zeit.parts(scene, trend=ar, mask=land, npart=100)
    res = zeit.parts(scene, trend=ar, classes=zones, npart=100)
    res_cov = zeit.parts(scene, trend=ar, mask=land, covariates={"coast_km": d["coast_km"]}, npart=100)
    note("scene_seconds_3_runs", round(time.time() - t, 1))
    s, p = ar.slope.values, ar.slope_p.values
    note("scene_shape", list(s.shape))
    note("scene_mean", dict(pixels=res_mean.n_pixels, estimate=round(float(res_mean.table.loc["mean", "estimate"]), 6),
                            se=round(float(res_mean.table.loc["mean", "se"]), 6),
                            p=round(float(res_mean.table.loc["mean", "p"]), 3),
                            range_km=round(res_mean.correlation["range"] / 1000, 1),
                            nugget=round(float(np.nanmean(res_mean.nuggets)), 3)))
    names = {1: "Coastal\n< 40 km", 2: "Transition\n40–100 km", 3: "Interior\n> 100 km"}
    rows = []
    for k in (1, 2, 3):
        m = (zones.values == k) & np.isfinite(s)
        est = res.table.loc[f"class {float(k)}"]
        rows.append(dict(zone=k, pixels=int(m.sum()), estimate=float(est.estimate), se=float(est.se),
                         p=float(est.p), raw_mean=float(s[m].mean()),
                         naive_se=float(s[m].std(ddof=1) / np.sqrt(m.sum())),
                         up=float(np.mean((p[m] < .05) & (s[m] > 0))),
                         down=float(np.mean((p[m] < .05) & (s[m] < 0)))))
    note("classes", [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    note("classes_chisqr", dict(p=round(float(res.p_chisqr), 3), F=round(float(res.f), 3), df1=res.df1, df2=res.df2))
    note("covariate", dict(estimate=float(f"{res_cov.table.loc['coast_km', 'estimate']:.3g}"),
                           se=float(f"{res_cov.table.loc['coast_km', 'se']:.3g}"),
                           p=round(float(res_cov.table.loc["coast_km", "p"]), 3),
                           chisqr_p=round(float(res_cov.p_chisqr), 3)))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    zc = ListedColormap(["#c9e4de", "#7fb3a8", "#3d7a6f"])
    axes[0].imshow(np.where(land.values, zones.values, np.nan), cmap=zc, vmin=0.5, vmax=3.5, interpolation="nearest")
    mf.map_axes(axes[0], "Zones by distance to the sea (300 m pixels)")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=colr, label=names[k].replace("\n", " "))
                            for k, colr in zip((1, 2, 3), zc.colors)],
                   loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3, handletextpad=0.4)
    ax = axes[1]
    ypos = np.arange(3)[::-1]
    for yv, r in zip(ypos, rows):
        ax.errorbar(r["estimate"], yv + 0.12, xerr=1.96 * r["se"], fmt="o", color=mf.BLUE, ms=7, capsize=4,
                    lw=2, label="PARTS: GLS mean ± 95%" if yv == 2 else None)
        ax.errorbar(r["raw_mean"], yv - 0.12, xerr=1.96 * r["naive_se"], fmt="s", color=mf.ORANGE, ms=6,
                    capsize=4, lw=2, label="plain mean ± 95% as if independent" if yv == 2 else None)
        ax.text(1.0, yv, f"{100 * r['up']:.0f}% ↑  {100 * r['down']:.0f}% ↓",
                transform=ax.get_yaxis_transform(), ha="right", va="center", color=mf.INK2, fontsize=9.5)
    ax.axvline(0, color=mf.INK, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(ypos, [names[r["zone"]] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean trend 1985–2025 (index per year)")
    ax.set_title(f"Do the zones' trends differ? chi-square p = {res.p_chisqr:.2f}")
    ax.set_ylim(-0.5, 2.95)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    mf.save(fig, "parts_classes", photo=True)


def fig_tuning(d):
    """How the number and size of the partitions and the seed change the answer and the time."""
    win, ar = d["win"], d["ar_win"]
    rows = []
    for npart in (4, 8, 16, 32, 64, 128, 256):
        t = time.time()
        r = zeit.parts(win, trend=ar, npart=npart)
        rows.append(("npart", npart, float(r.table.loc["mean", "se"]), float(r.table.loc["mean", "p"]), time.time() - t))
    for size in (500, 1000, 1500, 2000, 3000):
        t = time.time()
        r = zeit.parts(win, trend=ar, partsize=size, npart=40)
        rows.append(("partsize", size, float(r.table.loc["mean", "se"]), float(r.table.loc["mean", "p"]), time.time() - t))
    syn = synthetic(SEED, n=100)
    syn_ar = zeit.ar_trend(syn)
    for size in (300, 500, 1000, 1500, 2000):
        r = zeit.parts(syn, trend=syn_ar, partsize=size, npart=min(10_000 // size, 20))
        rows.append(("partsize_sim", size, float(r.table.loc["mean", "se"]), float(r.table.loc["mean", "p"]), 0.0))
    for seed in range(10):
        r = zeit.parts(win, trend=ar, npart=50, seed=seed)
        rows.append(("seed", seed, float(r.table.loc["mean", "se"]), float(r.table.loc["mean", "p"]),
                     r.correlation["range"] / 1000))
    tab = pd.DataFrame(rows, columns=["what", "value", "se", "p", "extra"])
    note("tuning", {w: g.drop(columns="what").round(6).values.tolist() for w, g in tab.groupby("what")})
    note("cpu_count", os.cpu_count())

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    axes = axes.ravel()
    a = tab[tab.what == "npart"]
    ax = axes[0]
    ax.plot(a.value, a.se, "o-", color=mf.BLUE, label="SE of the mean trend")
    ax.set_xscale("log", base=2)
    ax.set_xticks(a.value, [str(int(v)) for v in a.value])
    ax.set_xlabel("partitions of 1500 pixels")
    ax.set_ylabel("standard error (index per year)")
    ax.set_ylim(0, a.se.max() * 1.25)
    ax2 = ax.twinx()
    ax2.plot(a.value, a.extra, "s--", color=mf.ORANGE, label="seconds")
    ax2.set_ylabel("seconds", color=mf.ORANGE)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("More partitions: the same SE, more time")
    b = tab[tab.what == "partsize"]
    ax = axes[1]
    ax.plot(b.value, b.extra, "o-", color=mf.ORANGE)
    ax.set_xlabel("pixels per partition (40 partitions)")
    ax.set_ylabel("seconds")
    ax.set_title("Bigger partitions: the time grows as size³")
    ax = axes[2]
    bs = tab[tab.what == "partsize_sim"]
    ref = lambda g: g.se / g.se[g.value == 1500].iloc[0]  # noqa: E731
    ax.plot(bs.value, ref(bs), "o-", color=mf.BLUE, label="simulated map (range 450 m, 3 km wide)")
    ax.plot(b.value, ref(b), "s-", color=mf.ORANGE, label="real window (correlation ~0.6 at every distance)")
    for _, r in b.iterrows():
        ax.annotate(f"p {r.p:.2f}", (r.value, r.se / b.se[b.value == 1500].iloc[0]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=mf.INK2)
    ax.axhline(1, color=mf.AXIS, lw=1)
    ax.set_xlabel("pixels per partition")
    ax.set_ylabel("SE relative to partitions of 1500")
    ax.set_title("The partition size should not matter, unless the model misfits")
    ax.legend(loc="upper left", fontsize=9)
    c = tab[tab.what == "seed"]
    ax = axes[3]
    ax.plot(c.value, c.p, "o", color=mf.BLUE, ms=7)
    ax.axhline(0.05, color=mf.INK2, lw=1, ls=(0, (4, 3)))
    ax.set_ylim(0, max(0.6, c.p.max() * 1.15))
    ax.set_xticks(c.value)
    ax.set_xlabel("seed (50 partitions of 1500)")
    ax.set_ylabel("p of the mean trend")
    ax.set_title(f"Another draw, another p: {c.p.min():.2f} to {c.p.max():.2f}")
    fig.tight_layout()
    mf.save(fig, "parts_tuning")


FIGURES = {
    "problem": (fig_problem, False),
    "calibration": (fig_calibration, False),
    "pixel": (fig_pixel, True),
    "maps": (fig_maps, True),
    "correlation": (fig_correlation, None),
    "partitions": (fig_partitions, None),
    "classes": (fig_classes, True),
    "tuning": (fig_tuning, True),
}


def make_thumb():
    from PIL import Image
    path = OUT / "parts_problem.webp"
    if not path.exists():
        return
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((0, 0, int(0.67 * w), h))
    tw, th = 800, 450
    im.thumbnail((tw - 40, th - 40), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), "white")
    canvas.paste(im, ((tw - im.width) // 2, (th - im.height) // 2))
    (OUT / "thumbs").mkdir(exist_ok=True)
    canvas.save(OUT / "thumbs" / "parts.webp", quality=85, method=6)
    print("  wrote the gallery thumbnail")


def main(names):
    mf.setup_style()
    for name in names or list(FIGURES):
        fn, needs = FIGURES[name]
        print(name)
        if needs is False:
            fn()
            continue
        d = real()
        if d is None and needs:
            print("  skipped (set ZEIT_DOCS_PARTS to the folder of the annual series)")
            continue
        fn(d)
    make_thumb()
    print(json.dumps(NUMBERS, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
