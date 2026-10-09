"""
Example 03: Phenology Extraction End-to-End

Builds a synthetic 16-day NDVI datacube over three years (no network needed),
runs `zeit.phenology` with the HANTS smoother and Beck's double logistic
curve, and prints the start, peak and end of season of one pixel per year.
The day numbering and the years are read from the cube's dates.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import zeit


def main():
    print("Zeit Phenofit: End-to-End Phenology Extraction Example")

    # 1. Simulate a multi-year vegetation time-series (e.g., NDVI for 3 years)
    # We will create a synthetic dataset for 10 pixels over 3 years (16-day composites)
    years = 3
    time_steps_per_year = 23
    total_time_steps = years * time_steps_per_year
    n_pixels = 10

    # One composite every 16 days from January 2019
    dates = pd.date_range("2019-01-01", periods=total_time_steps, freq="16D")

    # Create synthetic NDVI datacube (Time, Y, X) -> (69, 10, 1)
    # Using a sine wave to simulate seasonal growth, adding noise
    np.random.seed(42)
    cube_data = np.zeros((total_time_steps, n_pixels, 1))

    for t in range(total_time_steps):
        # A simple curve peaking around the middle of the year
        doy = dates[t].dayofyear
        growth = np.sin((doy / 365.0) * np.pi)  # 0 to 1

        # Add a baseline of 0.2 (soil) and peak of 0.8 (healthy veg)
        ndvi = 0.2 + growth * 0.6

        # Add some random noise
        cube_data[t, :, 0] = ndvi + np.random.normal(0, 0.05, n_pixels)

    print(f"Created synthetic datacube with shape: {cube_data.shape}")

    # 2. Convert to an xarray DataArray (Dask backed)
    # In a real scenario, this would be returned by zeit.build_time_series()
    da_cube = da.from_array(cube_data, chunks=(total_time_steps, 5, 1))
    xr_cube = xr.DataArray(
        da_cube, dims=["time", "y", "x"],
        # A fake 30 m UTM grid, since this data has no real-world footprint
        coords={"time": dates, "y": 8800000.0 - 15.0 - 30.0 * np.arange(n_pixels), "x": [500015.0]},
    ).rio.write_crs("EPSG:32721")

    # 3. Run Phenology Extraction
    # We will use the HANTS smoother and the double logistic (Beck) curve.
    print("\nRunning Phenology extraction (this will process via C++ / OpenMP)...")
    pheno_results = zeit.phenology(
        xr_cube,
        curve="beck",
        method="derivative",
        annual=True,                     # One value per calendar year (the default)

        # Smoothing Configuration
        apply_whittaker=False,           # Disable Whittaker
        apply_hants=True,                # Enable HANTS
        hants_frequencies=3,
        hants_threshold=0.1,

        # Fine-Grained Season Control
        min_season_length=90,            # A real season must last at least 90 days
        min_amplitude=0.2,               # A real season must have an NDVI jump of at least 0.2

        n_jobs=-1                        # Use all available CPU cores
    )

    # 4. Trigger computation (the cube is Dask-backed, so the result is lazy)
    # 21 variables (19 metrics + R2 + RMSE), each (year, y, x)
    result = pheno_results.compute()

    print(f"\nExtraction Complete! {len(result.data_vars)} metrics, dims {dict(result.sizes)}")

    # 5. Inspect Results
    for year in result.year.values:
        px = result.sel(year=year).isel(y=0, x=0)
        print(f"\nPixel 0, {year}:")
        print(f"  Start of Season (DER.sos): DOY {float(px['DER.sos']):.1f}")
        print(f"  Peak of Season  (DER.pos): DOY {float(px['DER.pos']):.1f}")
        print(f"  End of Season   (DER.eos): DOY {float(px['DER.eos']):.1f}")
        print(f"  Length of Season:          {float(px['DER.eos'] - px['DER.sos']):.1f} days")

    # 6. Save: one GeoTIFF per metric, one band per year
    out_dir = os.path.join("data", "phenology")
    zeit.save_raster(result, out_dir)
    print(f"\nSaved one GeoTIFF per metric to {out_dir}")


if __name__ == "__main__":
    main()
