"""
Example 08: Mann-Kendall / Theil-Sen Trend Detection End-to-End

Loads a synthetic annual NDVI datacube (no network needed), runs the
pixel-wise Mann-Kendall trend test + Theil-Sen slope estimator with
`zeit.mann_kendall`, and saves the resulting trend/slope/p-value maps
as a multi-band GeoTIFF with `zeit.save_raster`.
"""
import os
import numpy as np
import xarray as xr
import zeit


def build_synthetic_ndvi_cube(n_years=20, rows=40, cols=40, seed=0):
    """
    Fake annual NDVI composites (time, y, x): the left half of the image
    greens up over time (a real trend), the right half only has noise
    around a constant mean (no trend) - a simple ground truth to check the
    trend test against.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_years)
    data = np.zeros((n_years, rows, cols), dtype=np.float64)

    greening_slope = 0.01  # NDVI units per year
    for y in range(rows):
        for x in range(cols):
            base = 0.4 + rng.normal(0, 0.02)
            if x < cols // 2:
                series = base + greening_slope * t
            else:
                series = np.full(n_years, base)
            data[:, y, x] = series + rng.normal(0, 0.015, n_years)

    return data


def main():
    print("Zeit Example 08: Mann-Kendall / Theil-Sen Trend Detection")

    n_years, rows, cols = 20, 40, 40
    start_year = 2004
    print(f"\n[1/3] Generating synthetic annual NDVI cube ({n_years} years, {rows}x{cols} px)...")
    data = build_synthetic_ndvi_cube(n_years=n_years, rows=rows, cols=cols)

    cube = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={
            "time": np.arange(start_year, start_year + n_years),
            # A fake 30 m UTM grid, since this data has no real-world footprint
            "y": 8800000.0 - 15.0 - 30.0 * np.arange(rows),
            "x": 500000.0 + 15.0 + 30.0 * np.arange(cols),
        },
    ).rio.write_crs("EPSG:32721")

    print("\n[2/3] Running Mann-Kendall (hamed_rao, autocorrelation-corrected) with zeit.mann_kendall...")
    # One NDVI composite per year, so slope comes out directly in NDVI/year.
    result = zeit.mann_kendall(cube, method="hamed_rao", alpha=0.05, n_jobs=-1)

    trend = result.trend.values
    slope = result.slope.values
    n_greening = int(np.sum(trend == 1))
    n_no_trend = int(np.sum(trend == 0))
    print(f"    Detected 'increasing' trend in {n_greening} pixels (expected ~{rows * cols // 2}).")
    print(f"    Detected 'no trend' in {n_no_trend} pixels.")
    print(f"    Mean slope on the greening half: {np.nanmean(slope[:, :cols // 2]):.4f} NDVI/year "
          f"(injected: 0.01).")

    print("\n[3/3] Saving results with zeit.save_raster()...")
    out_tif = os.path.join("data", "mann_kendall_trend.tif")
    zeit.save_raster(result, out_tif)  # one band per metric, named, georeferenced from the cube

    names = list(result.data_vars)
    print(f"\nDone! {len(names)}-band raster ({', '.join(names)}) saved to {out_tif}")


if __name__ == "__main__":
    main()
