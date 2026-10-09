"""
Example 11: Classic BFAST End-to-End (Iterative Trend + Season Break Detection)

Loads a synthetic 16-day composite datacube (no network needed) where half
the pixels have an injected trend break (e.g. a land-cover change), runs the
classic iterative `bfast()` with `zeit.bfast` (STL seasonal seed +
alternating trend/season segmented regressions; start_time and frequency are
read from the cube's dates), and saves the breakpoint and magnitude maps with
`zeit.save_raster`.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import zeit

FREQUENCY = 23  # 16-day composites/year, from January 2000
MAX_BREAKS_TREND = 5
MAX_BREAKS_SEASON = 5


def build_synthetic_cube(n_years=13, rows=12, cols=12, seed=3):
    """
    Trend + harmonic series, > 2*frequency long (bfast's STL seasonal seed
    needs at least two full years). Half the pixels (x >= cols/2) get an
    abrupt trend-level shift injected mid-series.
    """
    rng = np.random.RandomState(seed)
    n_time = n_years * FREQUENCY
    t = np.arange(n_time)
    trend_break = n_time // 2

    data = np.zeros((n_time, rows, cols), dtype=np.float64)
    for y in range(rows):
        for x in range(cols):
            base = 0.5 + rng.normal(0, 0.02)
            season = 0.1 * np.cos(2 * np.pi * t / FREQUENCY) + 0.05 * np.sin(2 * np.pi * t / FREQUENCY)
            series = base + 0.0006 * t + season
            if x >= cols // 2:
                series = series.copy()
                series[trend_break:] += 0.4
            data[:, y, x] = series + rng.normal(0, 0.02, n_time)

    return data, trend_break


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
    print("Zeit Example 11: Classic BFAST (Iterative Trend + Season Break Detection)")

    rows, cols = 12, 12
    print(f"\n[1/3] Generating synthetic cube ({rows}x{cols} px, {FREQUENCY} obs/year)...")
    data, trend_break = build_synthetic_cube(rows=rows, cols=cols)
    print(f"    Trend break injected at observation index {trend_break} for x >= {cols // 2}.")

    dates = pd.date_range("2000-01-01", periods=data.shape[0], freq="16D")
    cube = georeferenced_cube(data, dates)

    print("\n[2/3] Running classic bfast() with zeit.bfast (this alternates trend/season "
          "segmented regressions until convergence)...")
    result = zeit.bfast(
        cube, h=0.15,
        max_breaks_trend=MAX_BREAKS_TREND, max_breaks_season=MAX_BREAKS_SEASON,
        n_jobs=-1,
    )

    n_trend_breaks = result.n_trend_breaks.values
    print(f"    Trend breaks detected on the disturbed half: "
          f"{np.nanmean(n_trend_breaks[:, cols // 2:]):.2f} per pixel (injected: 1).")
    print(f"    Trend breaks detected on the stable half: "
          f"{np.nanmean(n_trend_breaks[:, :cols // 2]):.2f} per pixel (injected: 0).")
    magnitude = result.magnitude.values
    print(f"    Mean magnitude on the disturbed half: {np.nanmean(magnitude[:, cols // 2:]):.3f} "
          f"(injected: 0.4).")
    print(f"    Median time of the largest trend jump on the disturbed half: "
          f"{np.nanmedian(result.break_time.values[:, cols // 2:]):.2f} (decimal year).")

    print("\n[3/3] Saving results with zeit.save_raster()...")
    out_tif = os.path.join("data", "bfast_classic_breaks.tif")
    zeit.save_raster(result, out_tif)  # one band per metric, named, georeferenced from the cube

    names = list(result.data_vars)
    print(f"\nDone! {len(names)}-band raster ({', '.join(names)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
