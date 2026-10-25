"""
Example 27: Near-real-time monitoring with S-CCD

Builds a synthetic Landsat-like cube (no network needed): six bands of surface reflectance
x 10000, dates every 8 days from 2013 to 2023 (Landsat 8 and 9 together), 40% of them lost to
clouds (Fmask code 4). The grid has stable forest, forest cleared in mid-2018 and kept as
pasture, and forest cleared a few weeks before the last image. Then:

1. `zeit.sccd` starts a harmonic model on the first stable observations of every pixel and
   carries it forward with a Kalman filter, one update per new image;
2. the maps give the breaks, and the monitoring state at the last date: the forest cleared
   at the end is a change in progress, its latest observations anomalous but not yet `conse`
   (6) of them;
3. one pixel's observations are listed with what S-CCD made of each (`status`), and
   `zeit.ccdc` runs on the same cube for comparison.
"""
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

import zeit

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
FOREST = np.array([250, 500, 300, 3000, 1300, 500])     # reflectance x 10000
PASTURE = np.array([500, 800, 800, 2600, 2600, 1500])
SEASON = np.array([20, 40, 40, 300, 150, 80])            # amplitude of the yearly cycle


def build_cube(seed=27, ny=40, nx=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-04-15", "2023-11-30", freq="8D")
    t = (dates.year + (dates.dayofyear - 1) / 365.25).to_numpy()
    kind = np.zeros((ny, nx), dtype=int)                 # 0 stable forest
    kind[:, 20:40] = 1                                   # cleared in mid-2018
    kind[:20, 40:] = 2                                   # cleared in October 2023
    season = np.sin(2 * np.pi * (t - 0.25))[:, None] * SEASON
    curves = []
    for end, when in ((None, None), (PASTURE, 2018.5), (PASTURE, 2023.78)):
        c = FOREST + season
        if end is not None:
            c = np.where((t >= when)[:, None], end + 1.5 * season, c)
        curves.append(c)                                 # (time, band)
    data = np.stack(curves)[kind].transpose(2, 3, 0, 1)  # (time, band, y, x)
    data = data + rng.normal(0, 60, data.shape)
    qa = np.where(rng.random((len(t), ny, nx)) < 0.4, 4, 0).astype(np.uint8)
    coords = {"time": dates, "band": BANDS,
              "y": 8_900_000 - 15 - 30 * np.arange(ny), "x": 400_000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data.astype(np.float32), dims=("time", "band", "y", "x"), coords=coords)
    return cube.rio.write_crs(32720), qa, kind


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    cube, qa, kind = build_cube()
    print(f"{cube.sizes['time']} dates from {cube.time.dt.year.values[0]} to {cube.time.dt.year.values[-1]}, "
          f"{cube.sizes['y']} x {cube.sizes['x']} pixels")

    print("[1/3] S-CCD...")
    t0 = time.perf_counter()
    seg = zeit.sccd(cube, qa=qa)
    print(f"    {time.perf_counter() - t0:.2f} s")
    zeit.save_raster(seg[["n_breaks", "t_break", "nrt_mode", "n_anomalies", "anomaly_start"]],
                     os.path.join(out, "sccd"))

    print("[2/3] What the maps say for each kind of pixel...")
    for k, name in enumerate(["stable forest", "cleared 2018", "cleared Oct 2023"]):
        m = kind == k
        first = seg.t_break.isel(segment=0).values[m]
        broke = ~np.isnat(first)
        when = f"first break {pd.Series(first[broke]).median():%Y-%m-%d}" if broke.any() else "no break"
        anomalies = np.bincount(seg.n_anomalies.values[m], minlength=7)
        print(f"    {name:>16}: {when} ({broke.mean():.0%}), latest anomalies {dict(enumerate(anomalies.tolist()))}")
    alerts = (seg.nrt_mode == 1) & (seg.n_anomalies >= 3)
    print(f"    pixels with 3 or more anomalies at the last date: {int(alerts.sum())}")

    print("[3/3] One pixel cleared in October 2023, observation by observation...")
    px = zeit.sccd(cube.isel(y=5, x=50), qa=qa[:, 5, 50])
    names = np.array(zeit._sccd.STATUS)
    table = pd.DataFrame({
        "nir": cube.isel(y=5, x=50).sel(band="nir").values.round(),
        "nir_model": px.predicted.sel(band="nir").values.round(),
        "statistic": px.statistic.values.round(1),
        "status": names[px.status.values],
    }, index=px.time.values)
    print(table[px.status.values > 0].tail(10).to_string())
    print(f"    threshold {px.attrs['threshold']:.1f}; {int(px.n_anomalies)} anomalies since "
          f"{pd.Timestamp(px.anomaly_start.values):%Y-%m-%d}: a change in progress")

    t0 = time.perf_counter()
    cc = zeit.ccdc(cube, qa=qa)
    ccdc_break = cc.t_break.notnull().any("segment").values
    print(f"    CCDC on the same cube: {time.perf_counter() - t0:.2f} s, {int(ccdc_break.sum())} pixels with a "
          f"break (S-CCD: {int((seg.n_breaks > 0).sum())}), {int(ccdc_break[kind == 2].sum())} of them in the "
          f"October 2023 clearing, where CCDC dates the anomalies at the end as a break and S-CCD keeps them "
          f"as a change in progress")


if __name__ == "__main__":
    main()
