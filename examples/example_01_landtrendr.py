"""
Example 01: LandTrendr End-to-End
"""
import os
from zeit import build_annual_composites, landtrendr, extract_events, save_raster

if __name__ == '__main__':
    # Very small bounding box (0.01 degree) for ultra-fast execution
    bbox = [-55.01, -11.01, -55.00, -11.00]
    print("Building annual Landsat composites (one per year, as LandTrendr expects)...")
    comp = build_annual_composites(source="planetary_computer", collection="landsat-c2-l2", bbox=bbox,
                                   start_year=2000, end_year=2024, bands=["swir16"], resolution=30, epsg=3857)
    swir1 = comp.sel(band="swir16")   # (time, y, x), dates on January 1st of each year

    print("Running LandTrendr...")
    # SWIR1 rises when vegetation is cleared, so look for rises of the index.
    lt = landtrendr(swir1, direction="gain", max_segments=4, pval_threshold=0.05)

    print("Extracting greatest disturbance...")
    events = extract_events(lt, sort_by="greatest")   # event_type follows direction ("gain")

    out_dir = os.path.join("data", "lt_disturbance")
    save_raster(events, out_dir)   # one GeoTIFF per metric: yod.tif, magnitude.tif, ...

    print(f"Done! Outputs saved to {out_dir}")
