"""
Example 05: TempCNN, from a cube to a land-cover map

Builds a synthetic NDVI/red cube (no network needed) of a year with three kinds of
fields (an early crop, a late crop and forest), samples labelled points from it,
trains a TempCNN with `zeit.ai.train` and classifies every pixel with
`zeit.ai.predict`, then saves the map and the model.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit import ai

CLASSES = np.array(["early_crop", "late_crop", "forest"])


def build_scene(rows=60, cols=90, seed=5):
    """24 semi-monthly dates; three vertical strips with different seasons; some clouds."""
    rng = np.random.default_rng(seed)
    t = np.arange(24)
    seasons = [0.3 + 0.4 * np.sin(2 * np.pi * t / 24), 0.3 + 0.4 * np.sin(2 * np.pi * (t - 8) / 24),
               np.full(24, 0.8)]
    truth = np.zeros((rows, cols), dtype=int)
    truth[:, cols // 3:2 * cols // 3] = 1
    truth[:, 2 * cols // 3:] = 2
    ndvi = np.stack([seasons[k] for k in truth.ravel()], axis=1).reshape(24, rows, cols)
    ndvi = ndvi + rng.normal(0, 0.05, ndvi.shape)
    red = 0.2 - 0.1 * ndvi + rng.normal(0, 0.01, ndvi.shape)
    values = np.stack([ndvi, red], axis=1).astype(np.float32)
    values[rng.random((24, 1, rows, cols)).repeat(2, axis=1) < 0.05] = np.nan   # clouds
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    cube = xr.DataArray(values, dims=("time", "band", "y", "x"), coords={
        "time": pd.date_range("2022-01-01", periods=24, freq="SMS"), "band": ["ndvi", "red"],
        "y": transform.f - 30.0 * (np.arange(rows) + 0.5), "x": transform.c + 30.0 * (np.arange(cols) + 0.5)})
    return cube.rio.write_crs("EPSG:32721"), truth


def main():
    print("Zeit Example 05: TempCNN, from a cube to a land-cover map")

    print("\n[1/4] Building a synthetic NDVI/red cube and 400 labelled points...")
    cube, truth = build_scene()
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, cube.sizes["y"], 400), rng.integers(0, cube.sizes["x"], 400)
    points = gpd.GeoDataFrame({"class": CLASSES[truth[rows, cols]]}, crs="EPSG:32721",
                              geometry=gpd.points_from_xy(cube.x.values[cols], cube.y.values[rows]))

    print("\n[2/4] Samples, with validation in spatial blocks (zeit.ai.samples)...")
    samples = ai.samples(cube, points, label="class", block_size=20)
    print(f"    {samples}")

    print("\n[3/4] Training a TempCNN (zeit.ai.train)...")
    model = ai.train(ai.TempCNN, samples, epochs=40)
    best = model.zeit_history_[model.zeit_meta_["best_epoch"] - 1]
    print(f"    best epoch {best['epoch']}: validation accuracy {best['val_accuracy']:.1%}")

    print("\n[4/4] Classifying every pixel (zeit.ai.predict) and saving...")
    classes = ai.predict(model, cube, probability=True)
    names = np.array(model.zeit_meta_["classes"])[classes.label.values - 1]
    print(f"    agreement with the true map: {(names == CLASSES[truth]).mean():.1%}")
    out_dir = os.path.join("data", "tempcnn")
    zeit.save_raster(classes, out_dir)
    ai.save(model, os.path.join("data", "tempcnn.pt"))
    print(f"    label.tif and probability.tif -> {out_dir}/; the model -> data/tempcnn.pt")

    print("\nDone!")


if __name__ == "__main__":
    main()
