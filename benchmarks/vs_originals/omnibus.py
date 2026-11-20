"""zeit.omnibus against Mort Canty's sar_seqQ.py (CRC5Docker, MIT licence), run as a black box (``canty_run.py``; its
code is not read), on a synthetic Sentinel-1 series (500 x 500 pixels, 12 dates, VV and VH gamma speckle of 4.4 looks, three
blocks with known changes) and on a real one: RTC gamma0 of Planetary Computer's ``sentinel-1-rtc`` over Rondonia, one
relative orbit, 46 dates (``s1_rondonia.py``), with the equivalent number of looks estimated on intact forest.

    set CANTY_PYTHON=<python of an env with gdal, scipy, matplotlib>
    set CANTY_SRC=<CRC5Docker>/src
    set CANTY_LIB=<folder with libprov_means.so>        (Windows only)
    set S1_PYTHON=<python of an env with pystac-client, planetary-computer, odc-stac>   (for the real series)
    python benchmarks/vs_originals/omnibus.py [synthetic] [real] [speed]

What is found (see docs/benchmarks/vs-originals.md): with ``approximation="improved"`` the maps of the first change, the
last change and the number of changes are the same as Canty's in every pixel, and his per-interval map is ours; with
``"chi2"`` they are not. His input is the two-band file (VV, VH) of linear power of each date, diagonal dual-pol.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

import change_data as cd
import zeit
from common import best_of, no_power_throttling, record, workdir

THREADS = 4
ALPHA = 0.01


def read_series(files):
    return np.stack([rasterio.open(f).read() for f in files]).astype(np.float64)


def canty_maps(folder, files, stub, enl):
    """sar_seqQ.py on `files` at the significance ALPHA: the maps cmap, fmap, smap (uint8) and bmap (k-1 bands) and
    the time."""
    seconds, out = cd.canty("sar_seqQ.py", "-s", ALPHA, *files, stub, enl, threads=THREADS, timeout=7200)
    maps = {m: rasterio.open(Path(folder) / f"{stub}_{m}").read() for m in ("cmap", "fmap", "smap", "bmap")}
    return seconds, maps


def ours(series, dates, enl, approximation, valid=None):
    """Zeit's planes as Canty's maps: dates as indices (0 where nothing changed), the count, the direction at each
    date."""
    cube = xr.DataArray(series.astype(np.float32), dims=("time", "band", "y", "x"),
                        coords={"time": dates, "band": ["vv", "vh"]})
    out, per = zeit.omnibus(cube, enl=enl, alpha=ALPHA, approximation=approximation, per_date=True, n_jobs=THREADS)
    d = np.array(json.loads(out.attrs["omnibus"])["dates"]).astype(np.float32)
    idx = lambda band: np.where(np.isnan(out.sel(band=band).values), 0,
                                np.searchsorted(d, out.sel(band=band).values - 1e-6)).astype(np.int64)
    return {"smap": idx("first_change"), "cmap": idx("last_change"), "fmap": out.sel(band="n_changes").values.astype(np.int64),
            "bmap": np.nan_to_num(per.values[1:], nan=0).astype(np.int64)}, out


def agreement(mine, theirs, mask=None):
    mask = np.ones(theirs["fmap"][0].shape, bool) if mask is None else mask
    share = {}
    for m in ("smap", "cmap", "fmap"):
        share[m] = float((mine[m][mask] == theirs[m][0][mask]).mean())
    share["bmap"] = float((mine["bmap"][:, mask] == theirs["bmap"][:, mask]).mean())
    both = (mine["bmap"] > 0) & (theirs["bmap"] > 0)
    share["direction"] = float((mine["bmap"][:, mask][both[:, mask]] == theirs["bmap"][:, mask][both[:, mask]]).mean())
    return share


def show(label, share):
    print(f"  {label:<22} first {100 * share['smap']:8.4f} %  last {100 * share['cmap']:8.4f} %  count {100 * share['fmap']:8.4f} %  "
          f"per date {100 * share['bmap']:8.4f} %  direction where both {100 * share['direction']:8.4f} %")


def synthetic():
    d = workdir("omnibus-synthetic")
    size, k = 500, 12
    stub = "seq"
    if not (d / "s1_00.tif").exists():
        cd.sar_series(size, k=k, enl=4.4, path=str(d / "s1"))
    files = [d / f"s1_{i:02d}.tif" for i in range(k)]
    series = read_series(files)
    dates = pd.date_range("2021-06-01", periods=k, freq="12D")
    print(f"\nsynthetic: {size}x{size} px, {k} dates, VV and VH, 4.4 looks, alpha {ALPHA}")
    t_c, theirs = canty_maps(d, files, stub, 4.4)
    results = {}
    for approximation in ("improved", "chi2"):
        box = {}
        t = best_of(lambda: box.update(r=ours(series, dates, 4.4, approximation)), repeat=3)
        mine, out = box["r"]
        results[approximation] = (agreement(mine, theirs), t)
        show(f"zeit '{approximation}'", results[approximation][0])
    best, t_z = results["improved"]
    print(f"  times ({THREADS} threads): Canty {t_c:.1f} s, Zeit {t_z:.3f} s (maps from an array in memory)")
    assert min(best["smap"], best["cmap"], best["fmap"]) > 0.999 and best["bmap"] > 0.999, best
    note = (f"improved approximation: first/last change and count equal to Canty's in {100 * min(best['smap'], best['cmap'], best['fmap']):.3f} % "
            f"of the pixels, per-date map {100 * best['bmap']:.3f} %; the chi-square mode agrees in {100 * results['chi2'][0]['fmap']:.1f} % (count)")
    record("omnibus", "Canty sar_seqQ.py (MIT)", f"{size}x{size} px x {k} dates x VV+VH, synthetic", t_c, t_z, THREADS, note)


def real():
    folder = workdir("omnibus-real")
    if not (folder / "info.json").exists():
        py = os.environ.get("S1_PYTHON") or os.environ.get("CANTY_PYTHON")
        subprocess.run([py, "-I", str(Path(__file__).with_name("s1_rondonia.py")), str(folder), "400"], check=True,
                       env={**os.environ, "PYTHONPATH": ""})
    info = json.loads((folder / "info.json").read_text())
    files = sorted(folder.glob("s1_*.tif"))
    k = len(files)
    full = read_series(files)
    dates = pd.to_datetime(info["dates"])[:k]
    print(f"\nreal: Sentinel-1 RTC gamma0, Rondonia, relative orbit {info['orbit']} {info['state']}, {k} dates "
          f"{info['dates'][0]} to {info['dates'][-1]}, {full.shape[2]}x{full.shape[3]} px")
    valid_all = np.isfinite(full).all(axis=(0, 1)) & (full > 0).all(axis=(0, 1))
    print(f"  pixels valid (positive, finite) at every date: {100 * valid_all.mean():.1f} %")
    # the ENL on intact forest: the 15 x 15 block with the highest median VV and the lowest spatial variation
    med = np.nanmedian(full[:, 0], axis=0)
    best, where = (np.inf, None)
    for r in range(0, med.shape[0] - 15, 5):
        for c in range(0, med.shape[1] - 15, 5):
            blk = med[r:r + 15, c:c + 15]
            if np.isfinite(blk).all() and blk.mean() > np.nanpercentile(med, 70):
                cv = blk.std() / blk.mean()
                if cv < best:
                    best, where = cv, (r, r + 15, c, c + 15)
    cube = xr.DataArray(full.astype(np.float32), dims=("time", "band", "y", "x"), coords={"time": dates, "band": ["vv", "vh"]})
    est = zeit.estimate_enl(cube, region=where)
    enl = round(est["enl"]["vv"], 1)
    print(f"  ENL on the forest block {where}: VV {est['enl']['vv']:.2f}, VH {est['enl']['vh']:.2f} -> {enl} used for both")
    size = 300
    crop = (slice(50, 50 + size), slice(50, 50 + size))
    sub = full[:, :, crop[0], crop[1]].copy()
    valid = valid_all[crop]
    # Canty's script cannot take NaN or zeros: fill them with the pixel's median and leave them out of the comparison
    for b in range(2):
        med_t = np.nanmedian(np.where(sub[:, b] > 0, sub[:, b], np.nan), axis=0)
        bad = ~np.isfinite(sub[:, b]) | (sub[:, b] <= 0)
        sub[:, b] = np.where(bad, med_t[None], sub[:, b])
    sub = np.where(np.isfinite(sub), sub, np.nanmedian(sub))
    d = workdir("omnibus-real-crop")
    paths = []
    with rasterio.open(files[0]) as ref:
        transform = ref.window_transform(rasterio.windows.Window(50, 50, size, size))
        crs = ref.crs
    for i in range(k):
        p = d / f"s1_{i:02d}.tif"
        with rasterio.open(p, "w", driver="GTiff", count=2, dtype="float32", width=size, height=size, crs=crs,
                           transform=transform) as dst:
            dst.write(sub[i].astype("float32"))
        paths.append(p)
    print(f"  compared on a {size}x{size} crop ({100 * valid.mean():.1f} % of its pixels valid at every date): Canty's memory is "
          f"k*k*pixels*8 bytes ({k * k * size * size * 8 / 1e9:.1f} GB on disk)")
    t_c, theirs = canty_maps(d, paths, "seq", enl)
    res = {}
    for approximation in ("improved", "chi2"):
        box = {}
        t = best_of(lambda: box.update(r=ours(sub, dates, enl, approximation)), repeat=2)
        mine, out = box["r"]
        res[approximation] = (agreement(mine, theirs, valid), t)
        show(f"zeit '{approximation}'", res[approximation][0])
    best, t_z = res["improved"]
    changed = (theirs["fmap"][0][valid] > 0).mean()
    print(f"  changed pixels at alpha {ALPHA}: {100 * changed:.1f} %; times ({THREADS} threads): Canty {t_c:.1f} s, Zeit {t_z:.3f} s")
    assert min(best["smap"], best["cmap"], best["fmap"]) > 0.995 and best["bmap"] > 0.995, best
    note = (f"46 dates of Sentinel-1 RTC gamma0 (relative orbit {info['orbit']}), ENL {enl} estimated on forest; first/last change/count "
            f"equal in {100 * min(best['smap'], best['cmap'], best['fmap']):.2f} % of the pixels, per-date map {100 * best['bmap']:.2f} %")
    record("omnibus", "Canty sar_seqQ.py (MIT)", f"{size}x{size} px x {k} dates x VV+VH, Sentinel-1 RTC Rondonia", t_c, t_z,
           THREADS, note)
    # all the pixels of the fetched window, Zeit only
    t_full = best_of(lambda: ours(full, dates, enl, "improved"), repeat=2)
    print(f"  Zeit on the whole {full.shape[2]}x{full.shape[3]} window: {t_full:.3f} s")


def speed():
    """Zeit alone on a series Canty's memory cannot hold (k*k*pixels*8 bytes): 12 dates of 2000 x 2000 pixels."""
    rng = np.random.default_rng(0)
    size, k = 2000, 12
    series = (rng.gamma(4.4, 1 / 4.4, (k, 2, size, size)) * np.array([0.12, 0.03])[None, :, None, None]).astype(np.float32)
    cube = xr.DataArray(series, dims=("time", "band", "y", "x"),
                        coords={"time": pd.date_range("2021-06-01", periods=k, freq="12D"), "band": ["vv", "vh"]})
    t = best_of(lambda: zeit.omnibus(cube, n_jobs=THREADS), repeat=2)
    print(f"\nspeed: zeit.omnibus on {k} dates of {size}x{size} px, VV+VH: {t:.2f} s ({THREADS} threads); Canty's pvarray alone "
          f"would take {k * k * size * size * 8 / 1e9:.0f} GB")


if __name__ == "__main__":
    no_power_throttling()
    only = sys.argv[1:] or ["synthetic", "real", "speed"]
    if "synthetic" in only:
        synthetic()
    if "real" in only:
        real()
    if "speed" in only:
        speed()
