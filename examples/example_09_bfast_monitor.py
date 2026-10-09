"""
Example 09: BFAST Monitor End-to-End (Near-Real-Time Disturbance Monitoring)

Loads a synthetic 16-day composite datacube (no network needed) with a
stable history period and, for half the pixels, an abrupt disturbance
injected during the monitoring period. Runs `zeit.bfast_monitor` (the
regular time axis, start_time and frequency, is read from the cube's dates)
and saves the breakpoint/magnitude/sigma maps with `zeit.save_raster`.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import zeit

FREQUENCY = 23  # 16-day composites/year
START_TIME = 2015.0
MONITOR_START_TIME = 2019.0  # 4 years of history, then start monitoring


def build_synthetic_cube(n_years=6, rows=20, cols=20, seed=1):
    """
    Synthetic NDVI-like series: intercept + trend + one harmonic seasonal
    cycle, stable throughout history. Half the pixels (x >= cols/2) get an
    abrupt drop injected partway through the monitoring period (e.g. a
    clear-cut), the other half stays stable end to end.
    """
    rng = np.random.RandomState(seed)
    n_time = n_years * FREQUENCY
    t = np.arange(n_time)
    time_years = START_TIME + t / FREQUENCY

    disturbance_row = np.searchsorted(time_years, MONITOR_START_TIME + 0.5)  # ~mid-monitoring

    data = np.zeros((n_time, rows, cols), dtype=np.float64)
    for y in range(rows):
        for x in range(cols):
            base = 0.55 + rng.normal(0, 0.02)
            season = 0.12 * np.cos(2 * np.pi * t / FREQUENCY) + 0.06 * np.sin(2 * np.pi * t / FREQUENCY)
            series = base + season
            if x >= cols // 2:
                series = series.copy()
                series[disturbance_row:] -= 0.35  # abrupt disturbance (e.g. deforestation)
            data[:, y, x] = series + rng.normal(0, 0.015, n_time)

    return data, time_years, disturbance_row


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
    print("Zeit Example 09: BFAST Monitor (Near-Real-Time Monitoring)")

    rows, cols = 20, 20
    print(f"\n[1/3] Generating synthetic 16-day composite cube ({rows}x{cols} px, {FREQUENCY} obs/year)...")
    data, time_years, disturbance_row = build_synthetic_cube(rows=rows, cols=cols)
    print(f"    Disturbance injected at fractional year {time_years[disturbance_row]:.3f} "
          f"(observation index {disturbance_row}) for x >= {cols // 2}.")

    # One composite every 16 days from January 2015: zeit reads frequency=23 and start_time=2015.0
    # from these dates.
    dates = pd.date_range("2015-01-01", periods=data.shape[0], freq="16D")
    cube = georeferenced_cube(data, dates)

    print("\n[2/3] Running zeit.bfast_monitor...")
    result = zeit.bfast_monitor(
        cube,
        "2019-01-01",  # monitoring starts here (a decimal year, 2019.0, works too)
        h=0.25,
        period=10,
        alpha=0.05,
        n_jobs=-1,
    )
    print(f"    Time axis read from the dates: start_time={result.attrs['start_time']}, "
          f"frequency={result.attrs['frequency']}.")

    has_break = result.has_break.values
    print(f"    Breaks detected: {int(np.sum(has_break == 1))} pixels "
          f"(expected ~{rows * cols // 2}, the disturbed half).")
    breakpoint_idx = result.breakpoint_idx.values
    print(f"    Median detected breakpoint index on the disturbed half: "
          f"{np.nanmedian(breakpoint_idx[:, cols // 2:]):.1f} (injected at {disturbance_row}).")

    print("\n[3/3] Saving results with zeit.save_raster()...")
    out_tif = os.path.join("data", "bfast_monitor_breaks.tif")
    zeit.save_raster(result, out_tif)  # one band per metric, named, georeferenced from the cube

    names = list(result.data_vars)
    print(f"\nDone! {len(names)}-band raster ({', '.join(names)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
