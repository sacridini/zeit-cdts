"""
Example 14: QA-Band Decoding and CCDC Water Mask End-to-End

Loads synthetic MODIS "State QA" and Sentinel-2 "Scene Classification
Layer" (SCL) bands (no network needed) and decodes them into per-observation
reliability weights with `zeit.qc_modis_state` / `zeit.qc_sentinel2_scl`
(the same weights `zeit.phenology(..., weights=)` accepts). Also builds a
synthetic CCDC coefficient stack with a fake "lake" footprint and extracts a
persistent water mask from it with `zeit.extract_water_mask`. Saves
everything with `zeit.save_raster`.
"""
import os
import numpy as np
from rasterio.transform import from_origin
import zeit


def build_modis_state_qa(rows=30, cols=30, seed=6):
    """
    Fabricates a MODIS 500m State QA 16-bit flag field: bits 0-1 = cloud
    state, bits 6-7 = aerosol quantity, bit 12 = snow/ice. A diagonal band
    is "cloudy", a corner is "snow", the rest is clear/good aerosol.
    """
    rng = np.random.RandomState(seed)
    cloud_state = np.zeros((rows, cols), dtype=np.int64)   # 0 = clear
    aerosol = rng.choice([0, 1], size=(rows, cols))         # climatology/low
    snow = np.zeros((rows, cols), dtype=np.int64)

    yy, xx = np.mgrid[0:rows, 0:cols]
    cloud_state[np.abs(yy - xx) < 3] = 1  # a diagonal cloud band ("mixed")
    snow[(yy < rows // 5) & (xx < cols // 5)] = 1  # a snowy corner

    qa = (cloud_state & 0b11) | ((aerosol & 0b11) << 6) | ((snow & 0b1) << 12)
    return qa.astype(np.int64)


def build_sentinel2_scl(rows=30, cols=30, seed=7):
    """Fabricates a Sentinel-2 SCL layer: mostly vegetation (4), a cloud
    patch (9 = high-probability cloud) and a cloud-shadow strip (3)."""
    scl = np.full((rows, cols), 4, dtype=np.int64)  # vegetation
    yy, xx = np.mgrid[0:rows, 0:cols]
    scl[(yy > rows * 0.6) & (xx > cols * 0.6)] = 9    # cloud
    scl[(yy > rows * 0.6) & (xx > cols * 0.5) & (xx <= cols * 0.6)] = 3  # shadow fringe
    return scl


def build_fake_ccdc_water_scene(rows=30, cols=30, num_bands=6, green_idx=1, swir_idx=4):
    """
    A synthetic single-segment CCDC coefficient stack (shape:
    (1 segment, 3 + num_bands*7 params, rows, cols)) - only the Green and
    SWIR1 intercepts are populated, which is all `extract_water_mask` reads.
    A circular "lake" has high Green / very low SWIR1 reflectance; the rest
    of the scene looks like ordinary vegetated land.
    """
    params_per_seg = 3 + num_bands * 7
    coefs = np.zeros((1, params_per_seg, rows, cols), dtype=np.float64)

    green_intercept_idx = 4 + green_idx * 7
    swir_intercept_idx = 4 + swir_idx * 7

    coefs[0, green_intercept_idx] = 900.0   # land Green reflectance
    coefs[0, swir_intercept_idx] = 2200.0   # land SWIR1 reflectance

    yy, xx = np.mgrid[0:rows, 0:cols]
    lake = ((yy - rows * 0.35) ** 2 + (xx - cols * 0.65) ** 2) < (min(rows, cols) * 0.18) ** 2
    coefs[0, green_intercept_idx][lake] = 650.0
    coefs[0, swir_intercept_idx][lake] = 200.0

    return coefs, lake


def main():
    print("Zeit Example 14: QA-Band Decoding and CCDC Water Mask")

    rows, cols = 30, 30
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)

    print(f"\n[1/4] Decoding a synthetic MODIS State QA layer ({rows}x{cols} px)...")
    modis_qa = build_modis_state_qa(rows=rows, cols=cols)
    modis_weights = zeit.qc_modis_state(modis_qa)
    print(f"    Weight distribution: {np.mean(modis_weights == 1.0):.1%} good, "
          f"{np.mean(modis_weights == 0.5):.1%} marginal, {np.mean(modis_weights == 0.2):.1%} bad.")
    out_modis = os.path.join("data", "qc_modis_state_weights.tif")
    zeit.save_raster(modis_weights.astype("float32"), out_modis, crs="EPSG:32721", transform=transform, nodata=-1.0)
    print(f"    Saved -> {out_modis}")

    print(f"\n[2/4] Decoding a synthetic Sentinel-2 SCL layer ({rows}x{cols} px)...")
    scl = build_sentinel2_scl(rows=rows, cols=cols)
    scl_weights = zeit.qc_sentinel2_scl(scl)
    print(f"    Weight distribution: {np.mean(scl_weights == 1.0):.1%} good, "
          f"{np.mean(scl_weights == 0.5):.1%} marginal, {np.mean(scl_weights == 0.2):.1%} bad.")
    out_scl = os.path.join("data", "qc_sentinel2_scl_weights.tif")
    zeit.save_raster(scl_weights.astype("float32"), out_scl, crs="EPSG:32721", transform=transform, nodata=-1.0)
    print(f"    Saved -> {out_scl}")

    print(f"\n[3/4] Extracting a persistent water mask from a synthetic CCDC coefficient stack...")
    coefs, lake_truth = build_fake_ccdc_water_scene(rows=rows, cols=cols)
    water_mask = zeit.extract_water_mask(coefs, green_band_idx=1, swir_band_idx=4)
    hit_rate = (water_mask.astype(bool) == lake_truth).mean()
    print(f"    Agreement with the injected lake footprint: {hit_rate:.1%}.")

    print("\n[4/4] Saving results with zeit.save_raster()...")
    out_water = os.path.join("data", "ccdc_water_mask.tif")
    zeit.save_raster(water_mask.astype("uint8"), out_water, crs="EPSG:32721", transform=transform, nodata=255)
    print(f"    Saved -> {out_water}")

    print("\nDone!")


if __name__ == "__main__":
    main()
