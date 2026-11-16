"""The cubes of the EOF and lagged correlation benchmarks (eof.py, correlation.py): a synthetic monthly cube with
known modes, and the monthly mean of a real 16-day EVI series (575 dates, 2001-2025, 677 x 680 pixels, EPSG:3035).
"""
import json
import json
import os
import warnings
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import workdir

EVI = Path(os.environ.get("ZEIT_BENCH_EVI", r"C:\Users\eduar\Downloads\EVI_Series_ID_EU_150-X026-Y027_2001_2025.tif"))
PERIODS = (29, 47, 61)       # months: none of them is the year, so the anomalies keep the modes
AMPLITUDES = (3.0, 1.5, 0.6)


def mid_months(first_year, months):
    """The 15th of ``months`` consecutive months from January of ``first_year``."""
    return pd.date_range(f"{first_year}-01-01", periods=months, freq="MS") + pd.Timedelta(days=14)


def patterns(ny, nx):
    """A dipole (odd in x), a monopole (even in both) and a checker (even in x, odd in y), each with a root mean
    square of 1 over the grid (3, ny, nx): orthogonal on the symmetric grid."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx), indexing="ij")
    dipole = np.tanh(4 * xx)
    monopole = np.exp(-(xx ** 2 + yy ** 2) / 0.3)
    checker = np.sign(np.cos(3 * np.pi * xx) * np.sin(2 * np.pi * yy)) * np.exp(-(xx ** 2 + yy ** 2) / 1.5)
    out = np.stack([dipole, monopole, checker])
    return out / np.sqrt((out ** 2).mean(axis=(1, 2), keepdims=True))


def synthetic(months, ny, nx, *, geographic=False, seed=0, noise=0.5):
    """A monthly cube ``(time, y, x)`` made of three modes (sinusoids of 29, 47 and 61 months times a dipole, a
    monopole and a checker), a seasonal cycle that differs from pixel to pixel, and noise. Returns the cube
    (EPSG:3035, or EPSG:4326 between 10 and 70 degrees north when ``geographic``) and the true modes
    ``(u (3, T), v (3, ny, nx))`` with unit variance PCs."""
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    u = np.stack([np.sqrt(2) * np.sin(2 * np.pi * t / p + ph) for p, ph in zip(PERIODS, (0.3, 1.1, 2.0))])
    u = u - u.mean(axis=1, keepdims=True)
    q, _ = np.linalg.qr(u.T)                         # Gram-Schmidt: the PCs are exactly uncorrelated
    u = (q * np.sign((q * u.T).sum(axis=0))).T * np.sqrt(months - 1)
    v = patterns(ny, nx)
    a = np.array(AMPLITUDES)[:, None, None]
    signal = np.einsum("kt,kyx->tyx", u, a * v)
    phase = rng.uniform(0, 2 * np.pi, size=(ny, nx))
    season = 5.0 * np.cos(2 * np.pi * t[:, None, None] / 12 + phase) + 20.0 + rng.normal(size=(ny, nx))
    cube = signal + season + noise * rng.normal(size=(months, ny, nx))
    if geographic:
        y, x = np.linspace(70, 10, ny), np.linspace(0, 60, nx)
    else:
        y, x = 3000000 - 250.0 * np.arange(ny), 4000000 + 250.0 * np.arange(nx)
    da = xr.DataArray(cube, dims=("time", "y", "x"), coords={"time": mid_months(2001, months), "y": y, "x": x})
    da = da.rio.write_crs("EPSG:4326" if geographic else "EPSG:3035")
    return da, (u, v)


def two_patches(months, ny, nx, seed=0, noise=0.4):
    """Two disjoint patches (the left and the right third of the grid) whose time series have a correlation of 0.5:
    the EOFs of the cube are mixtures of the two (their sum and difference), and a varimax rotation of the first two
    should give the patches back. Returns the cube and the patches ``(2, ny, nx)`` (unit root mean square)."""
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    u1 = np.sin(2 * np.pi * t / 29 + 0.3)
    u2 = np.sin(2 * np.pi * t / 47 + 1.1)
    u2 = u2 - (u2 @ u1) / (u1 @ u1) * u1
    u2 = 0.5 * u1 / u1.std() + np.sqrt(0.75) * u2 / u2.std()
    u = np.stack([u1 / u1.std(), u2 / u2.std()])
    p = np.zeros((2, ny, nx))
    p[0, :, : nx // 3] = 1.0
    p[1, :, 2 * nx // 3:] = 1.0
    p /= np.sqrt((p ** 2).mean(axis=(1, 2), keepdims=True))
    season = 5.0 * np.cos(2 * np.pi * t[:, None, None] / 12 + rng.uniform(0, 2 * np.pi, size=(ny, nx))) + 20.0
    cube = np.einsum("kt,kyx->tyx", u, 2.0 * p) + season + noise * rng.normal(size=(months, ny, nx))
    da = xr.DataArray(cube, dims=("time", "y", "x"), coords={
        "time": mid_months(2001, months), "y": 3000000 - 250.0 * np.arange(ny), "x": 4000000 + 250.0 * np.arange(nx)})
    return da.rio.write_crs("EPSG:3035"), p


def real_monthly():
    """The monthly means of the EVI series as a ``(time, y, x)`` DataArray (float32; NaN where a month has no
    valid date), kept in the temporary folder after the first time."""
    d = workdir("eof")
    cache = d / "evi_monthly.npy"
    if not cache.exists():
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(EVI) as src:
            dates = pd.to_datetime([s.removeprefix("EVI_") for s in src.descriptions], format="%Y_%m_%d")
            months = dates.to_period("M")
            first = months.min()
            codes = np.array([(m - first).n for m in months])
            nm = int(codes.max()) + 1
            out = np.full((nm, src.height, src.width), np.nan, dtype=np.float32)
            for r0 in range(0, src.height, 128):
                h = min(128, src.height - r0)
                blk = src.read(window=Window(0, r0, src.width, h)).astype(np.float32)
                blk[~np.isfinite(blk)] = np.nan
                for m in range(nm):
                    sel = blk[codes == m]
                    if len(sel):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)      # months with no valid date
                            out[m, r0:r0 + h] = np.nanmean(sel, axis=0)
            np.save(d / "evi_monthly.tmp.npy", out)
            os.replace(d / "evi_monthly.tmp.npy", cache)
            tr = src.transform
            meta = dict(first=str(first), c=tr.c, a=tr.a, f=tr.f, e=tr.e, crs=src.crs.to_string())
        (d / "evi_meta.json").write_text(json.dumps(meta))
    meta = json.loads((d / "evi_meta.json").read_text())
    cube = np.load(cache, mmap_mode="r")
    nm, ny, nx = cube.shape
    first_year = int(str(meta["first"])[:4])
    da = xr.DataArray(np.asarray(cube), dims=("time", "y", "x"),
                      coords={"time": mid_months(first_year, nm), "y": meta["f"] + meta["e"] * (np.arange(ny) + 0.5),
                              "x": meta["c"] + meta["a"] * (np.arange(nx) + 0.5)})
    return da.rio.write_crs(meta["crs"])


def xarray_anomalies(cube, min_per_month=2):
    """Monthly-climatology anomalies by xarray (all the years): the reference of Zeit's ``anomaly="month"``, which
    leaves a month NaN for a pixel with fewer than ``min_per_month`` values of that calendar month."""
    g = cube.groupby("time.month")
    return g - g.mean("time").where(g.count("time") >= min_per_month)
