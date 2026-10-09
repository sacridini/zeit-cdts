"""
Example 02: CCDC and COLD End-to-End

Builds a small Sentinel-2 cube from a STAC catalog, runs CCDC on every pixel with
`zeit.ccdc`, predicts a cloud-free image for a date with `zeit.predict_synthetic_image`
and writes both with `zeit.save_raster`.
"""
import os
import zeit
from zeit import build_time_series

# Very small bounding box
bbox = [-55.01, -11.01, -55.00, -11.00]
print("Fetching STAC data...")
cube = build_time_series(bbox=bbox, start_date="2021-01-01", end_date="2022-12-31", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=15)

# (time, band, y, x) reflectance 0-1 -> x 10000, the scale CCDC's thresholds are defined on.
# NaN marks a missing observation; CCDC's internal Tmask screens the clouds left in the series.
cube = cube * 10000

print("Running COLD (CCDC)...")
segments = zeit.ccdc(cube, conseq_anom=6, chi2_prob_threshold=0.99).compute()
print(f"{int((segments.n_segments > 0).sum())} pixels modelled, "
      f"{int(segments.t_break.notnull().any('segment').sum())} with a break")

print("Generating a synthetic image for 2022-07-19...")
synthetic = zeit.predict_synthetic_image(segments, "2022-07-19")   # (band, y, x)

out_tif = os.path.join("data", "ccdc_synthetic.tif")
zeit.save_raster(synthetic, out_tif)
zeit.save_raster(segments, os.path.join("data", "ccdc_segments"))   # t_start.tif ... coefs.tif

print(f"Done! Output saved to {out_tif}")
