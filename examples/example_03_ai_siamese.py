"""
Example 03: Siamese Change Detector, from two images to a change map

Builds a synthetic pair of 4-band images (no network needed), "before" and
"after" a clearing, labels what changed and what did not with two polygons,
trains a Siamese change detector with `zeit.ai.train` and maps the change with
`zeit.ai.predict`.
"""
import os

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import zeit
from zeit import ai


def build_pair(size=64, seed=3):
    rng = np.random.default_rng(seed)
    transform = from_origin(500000.0, 8800000.0, 10.0, 10.0)
    coords = {"band": ["blue", "green", "red", "nir"],
              "y": transform.f - 10.0 * (np.arange(size) + 0.5), "x": transform.c + 10.0 * (np.arange(size) + 0.5)}
    forest = np.array([0.03, 0.06, 0.04, 0.40])[:, None, None]
    before = forest + rng.normal(0, 0.01, (4, size, size))
    after = before + rng.normal(0, 0.01, (4, size, size))
    cleared = np.zeros((size, size), dtype=bool)
    cleared[14:38, 20:50] = True
    after[:, cleared] = (np.array([0.10, 0.14, 0.18, 0.25])[:, None] + rng.normal(0, 0.01, (4, cleared.sum())))
    pair = tuple(xr.DataArray(v.astype(np.float32), dims=("band", "y", "x"), coords=coords).rio.write_crs("EPSG:32721")
                 for v in (before, after))
    clearing = box(transform.c + 10.0 * 20, transform.f - 10.0 * 38, transform.c + 10.0 * 50, transform.f - 10.0 * 14)
    scene = box(transform.c, transform.f - 10.0 * size, transform.c + 10.0 * size, transform.f)
    labels = gpd.GeoDataFrame({"class": ["change", "same"]}, geometry=[clearing, scene.difference(clearing)],
                              crs="EPSG:32721")
    return pair, labels, cleared


def main():
    print("Zeit Example 03: Siamese Change Detector, from two images to a change map")

    print("\n[1/3] Building a synthetic pair of images around a clearing...")
    pair, labels, cleared = build_pair()

    print("\n[2/3] Windows of 16 x 16 over both polygons, and training (zeit.ai.samples, train)...")
    samples = ai.samples(pair, labels, label="class", patch=16, split=0)
    print(f"    {samples}")
    model = ai.train(ai.SiameseChangeDetector, samples, epochs=40, batch_size=4)

    print("\n[3/3] Mapping the change (zeit.ai.predict) and saving...")
    change = ai.predict(model, pair, probability=True)
    found = np.array(model.zeit_meta_["classes"])[change.label.values - 1] == "change"
    print(f"    agreement with the true clearing: {(found == cleared).mean():.1%}")
    out = zeit.save_raster(change, os.path.join("data", "siamese_change"))
    print(f"    label.tif and probability.tif -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
