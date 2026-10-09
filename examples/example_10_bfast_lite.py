"""
Example 10: BFAST Lite End-to-End (Single-Pass Multiple-Breakpoint Detection)

Loads a synthetic 16-day composite datacube (no network needed) where every
pixel has two injected structural breaks (e.g. a disturbance followed by
recovery/regrowth), runs the single-pass `bfastlite` segmentation via the
`.zeit` accessor (the same as `zeit.bfast_lite(cube)`; start_time and
frequency are read from the cube's dates), and saves the breakpoint maps
with `zeit.save_raster`.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import zeit

FREQUENCY = 23  # 16-day composites/year, from January 2010
MAX_BREAKS = 5


def build_synthetic_cube(n_years=8, rows=15, cols=15, seed=2):
    """
    Trend + harmonic series with two injected breaks per pixel: a drop
    (disturbance) around 1/3 of the series, and a partial recovery around
    2/3 - a classic disturbance-then-regrowth trajectory.
    """
    rng = np.random.RandomState(seed)
    n_time = n_years * FREQUENCY
    t = np.arange(n_time)
    break1 = n_time // 3
    break2 = (2 * n_time) // 3

    data = np.zeros((n_time, rows, cols), dtype=np.float64)
    for y in range(rows):
        for x in range(cols):
            base = 0.6 + rng.normal(0, 0.02)
            season = 0.1 * np.cos(2 * np.pi * t / FREQUENCY) + 0.05 * np.sin(2 * np.pi * t / FREQUENCY)
            series = base + season
            series[break1:] -= 0.3   # disturbance
            series[break2:] += 0.15  # partial recovery
            data[:, y, x] = series + rng.normal(0, 0.02, n_time)

    return data, break1, break2


def georeferenced_cube(data, dates):
    """A (time, y, x) cube on a fake 30 m UTM grid, as zeit.load_raster would return it."""
    _, rows, cols = data.shape
    cube = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={
            "time": dates,
            "y": 8800000.0 - 15.0 - 30.0 * np.arange(rows),
            "x": 500000.0 + 15.0 + 30.0 * np.arange(cols),
        },
    )
    return cube.rio.write_crs("EPSG:32721")


def main():
    print("Zeit Example 10: BFAST Lite (Single-Pass Multiple-Breakpoint Detection)")

    rows, cols = 15, 15
    print(f"\n[1/3] Generating synthetic cube ({rows}x{cols} px) with 2 injected breaks per pixel...")
    data, break1, break2 = build_synthetic_cube(rows=rows, cols=cols)
    print(f"    Injected breaks at observation indices {break1} (disturbance) and {break2} (recovery).")

    dates = pd.date_range("2010-01-01", periods=data.shape[0], freq="16D")
    cube = georeferenced_cube(data, dates)

    print("\n[2/3] Running bfastlite via the .zeit accessor...")
    result = cube.zeit.bfast_lite(h=0.15, max_breaks=MAX_BREAKS, n_jobs=-1)

    n_breaks = result.n_breaks.values
    print(f"    Mean number of breaks detected per pixel: {np.nanmean(n_breaks):.2f} (injected: 2).")
    bp1 = result.breakpoint_idx_1.values
    bp2 = result.breakpoint_idx_2.values
    print(f"    Median 1st breakpoint: {np.nanmedian(bp1):.1f} (injected {break1}); "
          f"median 2nd breakpoint: {np.nanmedian(bp2):.1f} (injected {break2}).")

    print("\n[3/3] Saving results with zeit.save_raster()...")
    out_tif = os.path.join("data", "bfast_lite_breaks.tif")
    zeit.save_raster(result, out_tif)  # one band per metric, named, georeferenced from the cube

    names = list(result.data_vars)
    print(f"\nDone! {len(names)}-band raster ({', '.join(names)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
