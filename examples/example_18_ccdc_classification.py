"""
Example 18: Land Cover Classification End-to-End (train at points, classify the map)

Builds a synthetic georeferenced feature stack (no network needed - in a real
pipeline these would be the CCDC models of a date, `zeit.classify(segments,
model, date="2020-07-01")` reads them straight from a `zeit.ccdc` result, or
any cube, stack or Dataset of metrics), trains a Random Forest on labelled
sample points with `zeit.train_classifier`, classifies every pixel with
`zeit.classify`, and saves the class map and the class probabilities with
`zeit.save_raster`.
"""
import os
import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin
import zeit

N_FEATURES = 10  # e.g. intercept/slope/harmonic coefficients + RMSE, per a single band
CLASSES = ["forest", "pasture", "water"]


def class_centroid(class_id, n_features=N_FEATURES):
    rng = np.random.RandomState(100 + class_id)
    return rng.uniform(-1.0, 1.0, n_features) * (class_id + 1)


def build_synthetic_feature_stack(rows=50, cols=50, seed=13):
    """A georeferenced (band, y, x) stack of 3 spatial blocks, each with its
    class's centroid feature vector plus noise; and the true class of each pixel."""
    rng = np.random.RandomState(seed)
    third = cols // 3
    truth = np.zeros((rows, cols), dtype=np.uint8)
    truth[:, third:2 * third] = 1
    truth[:, 2 * third:] = 2

    stack = np.zeros((N_FEATURES, rows, cols), dtype=np.float32)
    for c in range(len(CLASSES)):
        mask = truth == c
        centroid = class_centroid(c)
        for f in range(N_FEATURES):
            stack[f][mask] = centroid[f] + rng.normal(0, 0.2, mask.sum())

    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    da = xr.DataArray(stack, dims=("band", "y", "x"), name="features",
                      coords={"band": [f"f{i}" for i in range(N_FEATURES)],
                              "y": transform.f - 30.0 * (np.arange(rows) + 0.5),
                              "x": transform.c + 30.0 * (np.arange(cols) + 0.5)})
    return da.rio.write_crs("EPSG:32721").rio.write_transform(transform), truth


def build_sample_points(stack, truth, n_per_class=20, seed=12):
    """Labelled points, as a field campaign would give them, in lon/lat."""
    rng = np.random.RandomState(seed)
    rows, cols, labels = [], [], []
    for c, name in enumerate(CLASSES):
        r, k = np.nonzero(truth == c)
        pick = rng.choice(len(r), n_per_class, replace=False)
        rows += list(r[pick])
        cols += list(k[pick])
        labels += [name] * n_per_class
    points = gpd.GeoDataFrame({"class": labels},
                              geometry=gpd.points_from_xy(stack.x.values[cols], stack.y.values[rows]),
                              crs=stack.rio.crs)
    return points.to_crs("EPSG:4326")   # zeit reprojects them to the stack's CRS


def main():
    print("Zeit Example 18: Land Cover Classification")

    rows, cols = 50, 50
    print(f"\n[1/4] Generating a synthetic feature stack ({N_FEATURES} CCDC-like features, {rows}x{cols} px)...")
    stack, truth = build_synthetic_feature_stack(rows=rows, cols=cols)

    print("\n[2/4] Training a Random Forest at labelled points (zeit.train_classifier)...")
    samples = build_sample_points(stack, truth)
    model = zeit.train_classifier(stack, samples, label="class")
    print(f"    {len(samples)} samples, classes {list(model.classes_)}, features {model.zeit_features_[:3]}...")

    print("\n[3/4] Classifying every pixel (zeit.classify)...")
    result = zeit.classify(stack, model, probability=True)
    names = list(result["class"].values)
    predicted = np.array(names)[result.label.values - 1]
    accuracy = (predicted == np.array(CLASSES)[truth]).mean()
    print(f"    Agreement between the classified map and the injected ground truth: {accuracy:.1%}")

    print("\n[4/4] Saving results with zeit.save_raster()...")
    out = zeit.save_raster(result, os.path.join("data", "land_cover"))
    print(f"    label.tif ({', '.join(f'{i + 1}={n}' for i, n in enumerate(names))}) and probability.tif -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
