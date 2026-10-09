"""
Example 16: Self-Organizing Map (SOM) End-to-End Unsupervised Clustering

Builds a synthetic 4-band reflectance scene (no network needed) with 3
spatially distinct land-cover-like clusters (water, vegetation, bare soil),
clusters its pixels with `zeit.som`, checks a set of labelled sample points
with `zeit.clean_samples` (some of them deliberately mislabelled), and saves
the cluster map with `zeit.save_raster`.
"""
import os
import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_origin
import zeit

TRANSFORM = from_origin(500000.0, 8800000.0, 30.0, 30.0)


def build_synthetic_scene(rows=40, cols=40, seed=9):
    """
    Three spectrally distinct classes arranged in vertical thirds. The
    REFLECTANCE always follows the true (clean) spatial class - only the
    training LABELS get some salt-and-pepper corruption, simulating
    mislabeled ground-truth samples (e.g. a GPS/digitizing error) whose
    spectra still clearly belong to their real class's SOM neuron. This is
    exactly the case `filter_noisy_samples` is meant to catch: a label that
    disagrees with the majority label among its neuron's other members.
    """
    rng = np.random.RandomState(seed)
    signatures = {
        0: np.array([0.05, 0.06, 0.04, 0.03]),   # water: low reflectance everywhere, esp. NIR/SWIR
        1: np.array([0.04, 0.09, 0.05, 0.45]),   # vegetation: high NIR
        2: np.array([0.25, 0.28, 0.30, 0.35]),   # bare soil: bright, flat spectrum
    }
    third = cols // 3
    labels = np.zeros((rows, cols), dtype=np.int32)
    labels[:, third:2 * third] = 1
    labels[:, 2 * third:] = 2

    data = np.zeros((rows, cols, 4), dtype=np.float64)
    for cls, sig in signatures.items():
        mask = labels == cls
        data[mask] = sig + rng.normal(0, 0.015, (mask.sum(), 4))

    # Corrupt only the LABELS of a few pixels (not their reflectance).
    noisy_labels = labels.copy()
    flip_idx = rng.choice(rows * cols, size=(rows * cols) // 25, replace=False)
    for idx in flip_idx:
        true_cls = labels.flat[idx]
        noisy_labels.flat[idx] = rng.choice([c for c in signatures if c != true_cls])

    return data, labels, noisy_labels


def main():
    print("Zeit Example 16: Self-Organizing Map (SOM) Unsupervised Clustering")

    rows, cols = 40, 40
    print(f"\n[1/3] Generating synthetic 4-band scene ({rows}x{cols} px, 3 land-cover clusters + noise)...")
    data, clean_labels, noisy_labels = build_synthetic_scene(rows=rows, cols=cols)
    scene = xr.DataArray(
        np.moveaxis(data, -1, 0), dims=("band", "y", "x"),
        coords={"band": ["blue", "green", "red", "nir"],
                "y": TRANSFORM.f + TRANSFORM.e * (np.arange(rows) + 0.5),
                "x": TRANSFORM.c + TRANSFORM.a * (np.arange(cols) + 0.5)},
    ).rio.write_crs("EPSG:32721")

    print("\n[2/3] Clustering every pixel with a 4x4 SOM (zeit.som)...")
    clusters = zeit.som(scene, x=4, y=4, sigma=1.5, sample=None)
    used = int((clusters.n_pixels > 0).sum())
    print(f"    {used} of {4 * 4} neurons hold pixels; mean distance to the prototypes: "
          f"{clusters.attrs['quantization_error']:.4f}")
    print(f"    Prototype of neuron 1 (blue, green, red, nir): {clusters.prototypes.sel(neuron=1).values.round(3)}")

    print("    Checking labelled sample points with zeit.clean_samples()...")
    names = np.array(["water", "vegetation", "bare_soil"])
    points = gpd.GeoDataFrame(
        {"class": names[noisy_labels.ravel()]},
        geometry=gpd.points_from_xy(np.tile(scene.x.values, rows), np.repeat(scene.y.values, cols)),
        crs="EPSG:32721",
    )
    checked = zeit.clean_samples(scene, points, label="class", x=4, y=4, sigma=1.5)
    true_noise = (noisy_labels != clean_labels).ravel()
    recall = (~checked.keep.to_numpy()[true_noise]).mean() if true_noise.any() else float("nan")
    print(f"    Flagged {int((~checked.keep).sum())}/{len(checked)} samples as suspicious; "
          f"recall on the actually-injected noise: {recall:.1%}.")

    print("\n[3/3] Saving results with zeit.save_raster()...")
    out_dir = os.path.join("data", "som")
    zeit.save_raster(clusters, out_dir)
    print(f"    Cluster map (label, 1-16) and distance to the prototypes -> {out_dir}/")
    out_points = os.path.join("data", "som_checked_samples.gpkg")
    checked.to_file(out_points)
    print(f"    Samples with neuron, neuron_class, purity and keep -> {out_points}")

    print("\nDone!")


if __name__ == "__main__":
    main()
