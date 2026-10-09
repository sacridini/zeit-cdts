"""
Example 17: Spatial Post-Processing End-to-End (MMU, Majority, Bayesian Filters)

Loads two synthetic rasters (no network needed) that mirror what
`zeit`'s other tools typically hand off to spatial post-processing: a
LandTrendr-style disturbance-year map (0 = no disturbance) with a few noisy
single-pixel specks, and a "salt and pepper" multi-class classification map
- the kind a per-pixel classifier (TWDTW, Random Forest, SOM, ...)
typically produces. Cleans them up with a Minimum Mapping Unit (MMU) filter
(`zeit.apply_mmu_filter`), a majority/mode filter (`zeit.apply_majority_filter`)
and a probability-aware Bayesian smoothing filter (`zeit.apply_bayesian_filter`).
The filters take and return georeferenced maps (or numpy arrays); every stage
is saved with `zeit.save_raster`.
"""
import os
import numpy as np
from rasterio.transform import from_origin
import zeit
from zeit.spatial import apply_bayesian_filter
from zeit.io import load_raster


def build_disturbance_map(rows=60, cols=60, seed=10):
    """
    A LandTrendr-style "year of disturbance" map: 0 = no disturbance, a few
    real disturbance patches (>= 9 px, should survive MMU filtering) and a
    scatter of single/2-pixel noise specks (should be removed).
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((rows, cols), dtype=np.uint16)

    # Real disturbance patches: 5x5, 4x4 and 3x4 blocks (25, 16, 12 px - all >= mmu_pixels=9).
    data[10:15, 10:15] = 2015
    data[30:34, 30:34] = 2018
    data[45:48, 15:19] = 2020

    # Noise: isolated 1-2 pixel specks scattered around (< mmu_pixels=9).
    n_specks = 40
    speck_idx = rng.choice(rows * cols, size=n_specks, replace=False)
    data.flat[speck_idx] = rng.choice([2016, 2017, 2019, 2021], size=n_specks)

    return data


def build_noisy_classification(rows=60, cols=60, seed=11, n_classes=3):
    """Three coherent land-cover blocks, corrupted with salt-and-pepper
    speckle - a stand-in for a real pixel-wise classifier's raw output."""
    rng = np.random.RandomState(seed)
    third = cols // 3
    clean = np.zeros((rows, cols), dtype=np.uint8)
    clean[:, third:2 * third] = 1
    clean[:, 2 * third:] = 2

    noisy = clean.copy()
    speckle_idx = rng.choice(rows * cols, size=(rows * cols) // 8, replace=False)
    noisy.flat[speckle_idx] = rng.randint(0, n_classes, size=len(speckle_idx))
    return clean, noisy


def probs_from_labels(noisy, clean, n_classes=3, confidence_correct=0.8, confidence_wrong=0.4):
    """
    Fakes a per-class probability cube [Classes, Y, X] from hard labels,
    like a softmax classifier's output would look. Crucially, the predicted
    class gets a *lower* confidence exactly where the prediction is wrong
    (`confidence_wrong`) than where it's right (`confidence_correct`) -
    simulating a real classifier, which is typically genuinely less certain
    right where it errs. That confidence signal (which a plain majority
    filter ignores entirely) is what lets Bayesian smoothing recover from
    noise a same-size majority filter might not.
    """
    rows, cols = noisy.shape
    is_correct = noisy == clean
    own_confidence = np.where(is_correct, confidence_correct, confidence_wrong)

    probs = np.zeros((n_classes, rows, cols), dtype=np.float64)
    for c in range(n_classes):
        is_pred_c = noisy == c
        probs[c] = np.where(is_pred_c, own_confidence, (1.0 - own_confidence) / (n_classes - 1))
    return probs


def main():
    print("Zeit Example 17: Spatial Post-Processing (MMU, Majority, Bayesian Filters)")

    rows, cols = 60, 60
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)

    print(f"\n[1/4] Generating a synthetic LandTrendr-style disturbance map ({rows}x{cols} px) with noise specks...")
    disturbance = build_disturbance_map(rows=rows, cols=cols)
    n_disturbed_before = int((disturbance != 0).sum())
    print(f"    {n_disturbed_before} disturbed pixels total (3 real patches >= 9 px + ~40 single-pixel noise specks).")

    in_tif = os.path.join("data", "spatial_disturbance_map.tif")
    zeit.save_raster(disturbance, in_tif, crs="EPSG:32721", transform=transform, nodata=0)

    print("\n[2/4] Applying the Minimum Mapping Unit (MMU) filter (removes patches < 9 px)...")
    out_mmu = os.path.join("data", "spatial_mmu_filtered.tif")
    mmu_result = zeit.apply_mmu_filter(load_raster(in_tif), mmu_pixels=9)   # NoData (0) where removed
    zeit.save_raster(mmu_result, out_mmu)
    n_disturbed_after = int((mmu_result.values != 0).sum())
    print(f"    Disturbed pixels after MMU filtering: {n_disturbed_after} "
          f"(the 3 real patches total {5*5 + 4*4 + 3*4} px - the rest was noise, now removed).")

    print("\n[3/4] Generating a synthetic noisy classification raster and applying the majority (mode) filter...")
    clean, noisy = build_noisy_classification(rows=rows, cols=cols)
    noisy_accuracy = (noisy == clean).mean()
    print(f"    Raw agreement with ground truth: {noisy_accuracy:.1%}.")
    majority_result = zeit.apply_majority_filter(noisy, size=3).astype("uint8")
    majority_accuracy = (majority_result == clean).mean()
    print(f"    Agreement after majority filtering: {majority_accuracy:.1%}.")
    out_majority = os.path.join("data", "spatial_majority_filtered.tif")
    zeit.save_raster(majority_result, out_majority, crs="EPSG:32721", transform=transform, nodata=255)

    print("\n[4/4] Applying the Bayesian probability-smoothing filter to the same scene "
          "(using a fake classifier confidence that's realistically lower right where it's wrong)...")
    probs = probs_from_labels(noisy, clean)
    bayesian_result = apply_bayesian_filter(probs, window_size=3).astype("uint8")
    bayesian_accuracy = (bayesian_result == clean).mean()
    print(f"    Agreement after Bayesian filtering: {bayesian_accuracy:.1%}.")
    out_bayes = os.path.join("data", "spatial_bayesian_filtered.tif")
    zeit.save_raster(bayesian_result, out_bayes, crs="EPSG:32721", transform=transform, nodata=255)

    print(f"\nDone! Saved: {in_tif}, {out_mmu}, {out_majority}, {out_bayes}")


if __name__ == "__main__":
    main()
