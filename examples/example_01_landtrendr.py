"""
Example 01: LandTrendr End-to-End
"""
import os
import numpy as np
from zeit import build_time_series, run_landtrendr_array, extract_events, save_raster
from zeit.smooth import apply_savgol_filter

if __name__ == '__main__':
    # Very small bounding box (0.01 degree) for ultra-fast execution
    bbox = [-55.01, -11.01, -55.00, -11.00]
    print("Fetching STAC data...")
    cube = build_time_series(bbox=bbox, start_date="2020-01-01", end_date="2022-12-31", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=15)

    swir1 = cube.isel(band=4).compute().values
    swir1 = np.nan_to_num(swir1)

    print("Smoothing time series...")
    swir1_smooth = apply_savgol_filter(swir1, window_length=5, polyorder=2, axis=0)

    print("Running LandTrendr...")
    years = cube.time.dt.year.values
    lt_vertices = run_landtrendr_array(years, swir1_smooth, max_segments=4, pval_threshold=0.05)

    print("Extracting greatest disturbance...")
    events = extract_events(lt_vertices, event_type="gain", sort_by="greatest")

    out_tif = os.path.join("data", "lt_disturbance_year.tif")
    save_raster(events['yod'], out_tif, like=cube, nodata=0)

    print(f"Done! Output saved to {out_tif}")
