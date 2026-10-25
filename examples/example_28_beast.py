"""
Example 28: Bayesian changepoints with BEAST

Builds a synthetic 16-day NDVI cube (no network needed): 20 years, 2005 to 2024, 23 dates a
year, 5% of them missing, noise of 0.03. The grid has four kinds of pixels: a stable forest, a
forest cleared in mid-2014 (an abrupt drop of the trend), a forest slowly degraded from 2010
(a change of slope, no jump) and a field turned to a double crop in 2016 (a change of the
season, not of the trend). Then:

1. `zeit.beast` decomposes every pixel's series into a piecewise trend and a piecewise
   harmonic season, averaging over the models of a reversible-jump MCMC instead of choosing
   one;
2. the maps say, for each kind of pixel, how many changepoints the trend and the season most
   probably have (the mean of a long, flat series is inflated by changepoints of nearly no jump,
   each with a low probability: read the mode and the probabilities), the most probable trend
   and season changepoints, their probability and the trend's jump;
3. one pixel's changepoints are listed with their probability, their 95% interval of dates
   and the jump of the trend there, and the probability of each number of changepoints.
"""
import os
import time

import numpy as np
import pandas as pd
import xarray as xr

import zeit

KINDS = ["stable forest", "cleared mid-2014", "degraded from 2010", "double crop from 2016"]


def build_cube(seed=28, ny=24, nx=40):
    rng = np.random.default_rng(seed)
    years = np.arange(2005, 2025)
    dates = pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=16 * k) for y in years for k in range(23)])
    t = (dates.year + (dates.dayofyear - 1) / 365.0).to_numpy()
    one = 0.12 * np.sin(2 * np.pi * (t - 0.3))                     # one growing season a year
    two = 0.10 * np.sin(4 * np.pi * (t - 0.15))                    # two
    curves = [
        0.80 + one,
        np.where(t < 2014.5, 0.80, 0.45) + one,
        0.80 - 0.025 * np.clip(t - 2010, 0, None) + one,
        0.55 + np.where(t < 2016, one, two),
    ]
    kind = np.repeat(np.arange(4), nx // 4)[None, :].repeat(ny, axis=0)
    data = np.stack(curves)[kind].transpose(2, 0, 1)               # (time, y, x)
    data = data + rng.normal(0, 0.03, data.shape)
    data[rng.random(data.shape) < 0.05] = np.nan
    coords = {"time": dates, "y": 8_900_000 - 15 - 30 * np.arange(ny), "x": 400_000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data.astype(np.float32), dims=("time", "y", "x"), coords=coords)
    return cube.rio.write_crs(32720), kind


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    cube, kind = build_cube()
    print(f"{cube.sizes['time']} dates from {cube.time.dt.year.values[0]} to {cube.time.dt.year.values[-1]}, "
          f"{cube.sizes['y']} x {cube.sizes['x']} pixels")

    print("[1/3] BEAST on every pixel...")
    t0 = time.perf_counter()
    # 16-day composites: a grid of 23 steps a year; fewer samples than the default 8000 x 3 for a map
    res = zeit.beast(cube, deltat=1 / 23, samples=2000, chains=2)
    print(f"    {time.perf_counter() - t0:.1f} s for {kind.size} pixels")
    zeit.save_raster(res[["trend_ncp", "season_ncp", "r2"]], os.path.join(out, "beast"))

    print("[2/3] What the maps say for each kind of pixel...")
    cp0 = res.trend_cp.isel(tcp=0).values
    p0 = res.trend_cp_prob.isel(tcp=0).values
    jump = res.trend_cp_abrupt.isel(tcp=0).values
    scp0 = res.season_cp.isel(scp=0).values
    sp0 = res.season_cp_prob.isel(scp=0).values
    slope = res.trend_slope.isel(time=-1).values
    for k, name in enumerate(KINDS):
        m = kind == k
        print(f"    {name}:")
        print(f"        trend: {np.median(res.trend_ncp_mode.values[m]):.0f} changepoints most probably "
              f"(mean {np.median(res.trend_ncp.values[m]):.1f}); the most probable at {np.nanmedian(cp0[m]):.2f}, "
              f"P {np.nanmedian(p0[m]):.2f}, jump {np.nanmedian(jump[m]):+.3f}; slope at the end "
              f"{np.median(slope[m]):+.4f} a year")
        print(f"        season: {np.median(res.season_ncp_mode.values[m]):.0f} changepoints most probably; "
              f"the most probable at {np.nanmedian(scp0[m]):.2f}, P {np.nanmedian(sp0[m]):.2f}")

    print("[3/3] One cleared pixel...")
    px = zeit.beast(cube.isel(y=5, x=15), deltat=1 / 23)           # Rbeast's defaults: 8000 samples, 3 chains
    found = px.trend_cp.notnull().values
    table = pd.DataFrame({
        "changepoint": px.trend_cp.values[found].round(2),
        "probability": px.trend_cp_prob.values[found].round(3),
        "from": px.trend_cp_ci.sel(bound="lower").values[found].round(2),
        "to": px.trend_cp_ci.sel(bound="upper").values[found].round(2),
        "jump": px.trend_cp_abrupt.values[found].round(3),
    })
    print(table.to_string(index=False))
    prob = px.trend_ncp_prob.values
    print("    P(number of trend changepoints): "
          + ", ".join(f"{n}: {p:.2f}" for n, p in enumerate(prob) if p >= 0.01))
    print(f"    R2 {float(px.r2):.3f}; the same seed gives the same numbers, here and in Rbeast's beast() "
          f"(with cputype='sse' and dump.ci=TRUE)")


if __name__ == "__main__":
    main()
