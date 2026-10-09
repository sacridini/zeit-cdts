"""
Example 02: CCDC and COLD End-to-End
"""
import os
import numpy as np
from zeit import build_time_series, apply_tmask_stack, run_ccdc_array, predict_synthetic_image, save_raster

# Very small bounding box
bbox = [-55.01, -11.01, -55.00, -11.00]
print("Fetching STAC data...")
cube = build_time_series(bbox=bbox, start_date="2021-01-01", end_date="2022-12-31", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=15)

print("Downloading arrays for Tmask...")
green = np.nan_to_num(cube.isel(band=1).compute().values)
swir1 = np.nan_to_num(cube.isel(band=4).compute().values)
dates_julian = cube.time.dt.dayofyear.values

print("Running Tmask...")
clear_mask = apply_tmask_stack(dates_julian, green, swir1, scale_factor=10000.0)

print("Running COLD (CCDC)...")
full_array = np.nan_to_num(cube.compute().values)
full_array[:, ~clear_mask] = 0

coefs = run_ccdc_array(full_array, dates_julian, conseq_anom=6, chi_square_prob=0.99)

print("Generating Synthetic Image for Julian Day 200...")
synthetic = predict_synthetic_image(coefs, target_julian_day=200)

out_tif = os.path.join("data", "ccdc_synthetic.tif")
save_raster(synthetic, out_tif, like=cube, nodata=0)

print(f"Done! Output saved to {out_tif}")
