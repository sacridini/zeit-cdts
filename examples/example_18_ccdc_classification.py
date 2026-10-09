"""
Example 18: CCDC-Based Land Cover Classification End-to-End

Loads a synthetic CCDC coefficient stack (no network needed - in a real
pipeline this would be `zeit.ccdc` coefficients written with `zeit.save_raster`,
e.g. `segments.coefs.isel(segment=0)`), trains a Random Forest
classifier on a handful of labeled training samples with
`zeit.train_ccdc_classifier`, applies it chunk-by-chunk to the full
coefficient GeoTIFF with `zeit.classify_ccdc_stack`, and loads the result
back with `zeit.load_raster` to check it. `classify_ccdc_stack` writes its
own output file directly (it's designed for coefficient stacks too large to
fit in memory), so this is the one example where saving happens inside the
Zeit function itself rather than via a separate `save_raster` call.
"""
import os
import numpy as np
from rasterio.transform import from_origin
import zeit
from zeit.io import load_raster

N_FEATURES = 10  # e.g. intercept/slope/harmonic coefficients + RMSE, per a single band


def class_centroid(class_id, n_features=N_FEATURES):
    rng = np.random.RandomState(100 + class_id)
    return rng.uniform(-1.0, 1.0, n_features) * (class_id + 1)


def build_training_samples(n_classes=3, n_per_class=60, seed=12):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in range(n_classes):
        centroid = class_centroid(c)
        X.append(centroid + rng.normal(0, 0.2, (n_per_class, N_FEATURES)))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def build_synthetic_coef_stack(rows=50, cols=50, n_classes=3, seed=13):
    """A fake full-scene CCDC coefficient stack: 3 spatial blocks, each with
    its class's centroid feature vector plus noise."""
    rng = np.random.RandomState(seed)
    third = cols // 3
    truth = np.zeros((rows, cols), dtype=np.uint8)
    truth[:, third:2 * third] = 1
    truth[:, 2 * third:] = 2

    stack = np.zeros((N_FEATURES, rows, cols), dtype=np.float32)
    for c in range(n_classes):
        mask = truth == c
        centroid = class_centroid(c)
        for f in range(N_FEATURES):
            stack[f][mask] = centroid[f] + rng.normal(0, 0.2, mask.sum())

    return stack, truth


def main():
    print("Zeit Example 18: CCDC-Based Land Cover Classification")

    print(f"\n[1/4] Generating {3*60} synthetic training samples ({N_FEATURES} CCDC-like features/class)...")
    X_train, y_train = build_training_samples()

    print("\n[2/4] Training the Random Forest classifier (zeit.train_ccdc_classifier)...")
    clf = zeit.train_ccdc_classifier(X_train, y_train, n_estimators=100)
    train_accuracy = clf.score(X_train, y_train)
    print(f"    Training-set accuracy: {train_accuracy:.1%}")

    rows, cols = 50, 50
    print(f"\n[3/4] Generating a synthetic full-scene CCDC coefficient stack ({rows}x{cols} px) and saving it...")
    coef_stack, truth = build_synthetic_coef_stack(rows=rows, cols=cols)
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    coef_path = os.path.join("data", "ccdc_coef_stack.tif")
    zeit.save_raster(coef_stack, coef_path, crs="EPSG:32721", transform=transform, nodata=0)

    print("\n[4/4] Classifying the full stack with zeit.classify_ccdc_stack() (chunked, writes its own output)...")
    class_path = os.path.join("data", "ccdc_land_cover_classification.tif")
    zeit.classify_ccdc_stack(clf, coef_path, class_path, chunk_size=32)

    predicted = load_raster(class_path).values[None]   # (1, rows, cols)
    accuracy = (predicted[0] == truth).mean()
    print(f"    Agreement between the classified map and the injected ground truth: {accuracy:.1%}")

    print(f"\nDone! Saved {coef_path} and {class_path}")


if __name__ == "__main__":
    main()
