"""
Example 24: Monitoring with EWMACD

Builds a synthetic Landsat-like NDVI series (no network needed): irregular dates from 2013
to 2023, about 16 days apart, with 40% of them lost to clouds (masked, as a QA band would).
The grid has stable forest, forest clear-cut in mid-2019 that regrows over three
years, forest thinned in 2020 (a small, lasting drop) and young forest that keeps growing.
Then:

1. `zeit.ewmacd` fits a harmonic model to the training years (2013-2016) of every pixel and
   follows the EWMA chart of its residuals from 2017 on;
2. the maps give the first disturbance, its severity, the date it recovered and the state at
   the last date;
3. one pixel's chart (the EWMA, its control limits and the flags) is printed.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import zeit


def build_cube(seed=24, ny=40, nx=60):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-05", "2023-12-31", freq="16D")
    dates = dates[rng.random(len(dates)) > 0.4]                       # cloudy dates are gone
    t = (dates.year + (dates.dayofyear - 1) / 365.25).to_numpy()
    season = 0.08 * np.sin(2 * np.pi * (t - 0.25))
    kind = np.zeros((ny, nx), dtype=int)                              # 0 stable forest
    kind[:20, 20:40] = 1                                              # clear-cut in mid-2019
    kind[20:, 20:40] = 2                                              # thinned in 2020
    kind[:, 40:] = 3                                                  # young forest, growing
    base = np.stack([
        0.80 + season,
        0.80 + season - np.where(t >= 2019.5, 0.45 * np.clip(1 - (t - 2019.5) / 3, 0, 1), 0),
        0.80 + season - np.where(t >= 2020.3, 0.06, 0),
        0.62 + season + 0.03 * np.clip(t - 2013, 0, None),
    ])                                                                # (4, time)
    data = base[kind].transpose(2, 0, 1) + rng.normal(0, 0.03, (len(t), ny, nx))
    coords = {"time": dates, "y": 8_900_000 - 15 - 30 * np.arange(ny), "x": 400_000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data.astype(np.float32), dims=("time", "y", "x"), coords=coords).rio.write_crs(32720)
    return cube, kind


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    ndvi, kind = build_cube()
    print(f"{ndvi.sizes['time']} dates from {ndvi.time.dt.year.values[0]} to {ndvi.time.dt.year.values[-1]}")

    print("[1/3] EWMACD, trained on 2013-2016...")
    res = zeit.ewmacd(ndvi, train_end=2017)
    zeit.save_raster(res[["break_time", "severity", "recovery_time", "state"]], os.path.join(out, "ewmacd"))

    print("[2/3] What the maps say for each kind of pixel...")
    states = res.state.attrs["flag_meanings"].split()
    for k, name in enumerate(["stable forest", "clear-cut 2019", "thinned 2020", "growing"]):
        m = kind == k
        state = np.bincount(res.state.values[m].astype(int), minlength=4)
        broke = np.isfinite(res.break_time.values[m])
        first = f"first disturbance {np.median(res.break_time.values[m][broke]):7.2f}" if broke.any() else "no disturbance"
        print(f"    {name:>14}: {first}, state {dict(zip(states, state.tolist()))}")
    calm = zeit.ewmacd(ndvi, train_end=2017, persistence=6)
    false = [float(np.isfinite(r.break_time.values[kind == 0]).mean()) for r in (res, calm)]
    print(f"    stable forest with a (false) signal: {false[0]:.0%} with persistence 3, {false[1]:.0%} with 6")

    print("[3/3] One clear-cut pixel's chart...")
    px = zeit.ewmacd(ndvi.isel(y=5, x=25), train_end=2017)
    chart = pd.DataFrame({"ewma": px.ewma, "limit": px.limit, "flag": px.flags}, index=px.time.values)
    print(chart.loc["2019-03":"2020-06"].round(3).to_string())
    print(f"    disturbed at {float(px.break_time):.2f}, recovered at {float(px.recovery_time):.2f}")


if __name__ == "__main__":
    main()
