"""
Example 04: U-TAE (U-Net with Temporal Attention), from a cube to a crop map

Builds a synthetic NDVI/red cube (no network needed) with fields of three kinds,
takes their polygons as labels (dense labels, as segmentation models are trained
on), trains a small U-TAE on 32 x 32 windows with `zeit.ai.train`, and maps the
whole cube with `zeit.ai.predict`, which slides overlapping windows over it.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import zeit
from zeit import ai

CLASSES = np.array(["early_crop", "late_crop", "forest"])


def build_scene(rows=96, cols=96, field=24, seed=4):
    """Square fields of 24 x 24 pixels, each one of three seasons; 20 dates."""
    rng = np.random.default_rng(seed)
    t = np.arange(20)
    seasons = [0.3 + 0.4 * np.sin(2 * np.pi * t / 20), 0.3 + 0.4 * np.sin(2 * np.pi * (t - 7) / 20),
               np.full(20, 0.8)]
    kinds = rng.integers(0, 3, (rows // field, cols // field))
    truth = np.kron(kinds, np.ones((field, field), dtype=int))
    ndvi = np.stack([seasons[k] for k in truth.ravel()], axis=1).reshape(20, rows, cols)
    ndvi = ndvi + rng.normal(0, 0.05, ndvi.shape)
    red = 0.2 - 0.1 * ndvi + rng.normal(0, 0.01, ndvi.shape)
    transform = from_origin(500000.0, 8800000.0, 10.0, 10.0)
    cube = xr.DataArray(np.stack([ndvi, red], axis=1).astype(np.float32), dims=("time", "band", "y", "x"), coords={
        "time": pd.date_range("2022-01-01", periods=20, freq="18D"), "band": ["ndvi", "red"],
        "y": transform.f - 10.0 * (np.arange(rows) + 0.5), "x": transform.c + 10.0 * (np.arange(cols) + 0.5)})
    fields = gpd.GeoDataFrame(
        {"class": [CLASSES[k] for k in kinds.ravel()]}, crs="EPSG:32721",
        geometry=[box(transform.c + 10.0 * field * j, transform.f - 10.0 * field * (i + 1),
                      transform.c + 10.0 * field * (j + 1), transform.f - 10.0 * field * i)
                  for i in range(kinds.shape[0]) for j in range(kinds.shape[1])])
    return cube.rio.write_crs("EPSG:32721"), truth, fields


def main():
    print("Zeit Example 04: U-TAE, from a cube to a crop map")

    print("\n[1/4] Building a synthetic cube of 16 fields and their polygons...")
    cube, truth, fields = build_scene()

    print("\n[2/4] Windows of 32 x 32 pixels around the fields (zeit.ai.samples)...")
    samples = ai.samples(cube, fields, label="class", patch=32, block_size=48)
    print(f"    {samples}")

    print("\n[3/4] Training a small U-TAE (zeit.ai.train)...")
    model = ai.train(ai.UTAE, samples, epochs=40, encoder_widths=[16, 16, 32, 32],
                     decoder_widths=[16, 16, 32, 32], d_model=32, n_head=4)
    print(f"    {len(model.zeit_history_)} epochs, best {model.zeit_meta_['best_epoch']}")

    print("\n[4/4] Mapping the cube with overlapping windows (zeit.ai.predict)...")
    crops = ai.predict(model, cube, overlap=0.5)
    names = np.array(model.zeit_meta_["classes"])[crops.label.values - 1]
    print(f"    agreement with the true map: {(names == CLASSES[truth]).mean():.1%}")
    out = zeit.save_raster(crops.label, os.path.join("data", "utae_crops.tif"))
    print(f"    crop map -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
