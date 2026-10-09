"""
Example 06: Geospatial Foundation Model (ViT), fine-tuned on a cube

Builds a synthetic cube of 3 dates and the 6 bands Prithvi-100M takes (no data
download needed), fine-tunes `GeoFoundationViT` on 224 x 224 windows with
`zeit.ai.train` and maps the cube with `zeit.ai.predict`, then cleans the map with
a majority filter. With network access the Prithvi-100M backbone is downloaded from
Hugging Face (and training is slow on a CPU); without it, the model falls back to a
small convolutional encoder, and the example still runs end to end.
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


def build_scene(size=256, seed=6):
    rng = np.random.default_rng(seed)
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    water = np.zeros((size, size), dtype=bool)
    water[:, : size // 3] = True
    land = np.array([0.04, 0.07, 0.06, 0.35, 0.20, 0.10])
    lake = np.array([0.06, 0.05, 0.03, 0.02, 0.01, 0.01])
    values = np.where(water[None, None], lake[None, :, None, None], land[None, :, None, None])
    values = values + rng.normal(0, 0.01, (3, 6, size, size))
    cube = xr.DataArray(values.astype(np.float32), dims=("time", "band", "y", "x"), coords={
        "time": pd.date_range("2022-05-01", periods=3, freq="MS"),
        "band": ["blue", "green", "red", "nir", "swir1", "swir2"],
        "y": transform.f - 30.0 * (np.arange(size) + 0.5), "x": transform.c + 30.0 * (np.arange(size) + 0.5)})
    edge = transform.c + 30.0 * (size // 3)
    labels = gpd.GeoDataFrame({"class": ["water", "land"]}, crs="EPSG:32721", geometry=[
        box(transform.c, transform.f - 30.0 * size, edge, transform.f),
        box(edge, transform.f - 30.0 * size, transform.c + 30.0 * size, transform.f)])
    return cube.rio.write_crs("EPSG:32721"), labels, water


def main():
    print("Zeit Example 06: Geospatial Foundation Model (ViT), fine-tuned on a cube")

    print("\n[1/3] Building a synthetic 3-date, 6-band cube and two polygons...")
    cube, labels, water = build_scene()

    print("\n[2/3] Fine-tuning GeoFoundationViT on 224 x 224 windows (zeit.ai.samples, train)...")
    samples = ai.samples(cube, labels, label="class", patch=224, split=0)
    print(f"    {samples}")
    model = ai.train(ai.GeoFoundationViT, samples, epochs=5, lr=1e-4)

    print("\n[3/3] Mapping the cube (zeit.ai.predict), a majority filter, and saving...")
    classes = ai.predict(model, cube)
    clean = zeit.apply_majority_filter(classes.label, size=3)
    found = np.array(model.zeit_meta_["classes"])[clean.values - 1] == "water"
    print(f"    agreement with the true lake: {(found == water).mean():.1%}")
    out = zeit.save_raster(clean, os.path.join("data", "vit_water.tif"))
    print(f"    water map -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
