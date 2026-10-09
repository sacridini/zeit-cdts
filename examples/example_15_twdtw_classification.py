"""
Example 15: TWDTW End-to-End (Time-Weighted Dynamic Time Warping Classification)

Builds a synthetic single-season NDVI datacube (no network needed) where the
left half of the image follows a "soybean"-like early-peak phenological
curve and the right half a "corn"-like late-peak curve (both with noise and
a random phase jitter, as real fields would have), classifies every pixel
against two reference patterns with `zeit.twdtw`, and saves the class,
distance and per-pattern distance maps with `zeit.save_raster`.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.transform import from_origin
import zeit


def double_logistic(doy, sos, eos, peak_amplitude=0.7, base=0.15, steepness=0.1):
    """A simple double-logistic vegetation growth curve peaking between sos/eos."""
    mid = (sos + eos) / 2.0
    green_up = 1.0 / (1.0 + np.exp(-steepness * (doy - sos)))
    green_down = 1.0 / (1.0 + np.exp(steepness * (doy - eos)))
    return base + peak_amplitude * green_up * green_down / (green_up * green_down).max() * (doy < mid + 200)


DATES = pd.date_range("2022-01-01", "2022-12-31", freq="16D")


def build_patterns():
    """The patterns of a reference year, as series indexed by dates."""
    doy = DATES.dayofyear.to_numpy()
    return {"soybean": pd.Series(double_logistic(doy, sos=40, eos=130), DATES),
            "corn": pd.Series(double_logistic(doy, sos=110, eos=220), DATES)}


def build_synthetic_cube(rows=20, cols=20, seed=8):
    """A georeferenced (time, y, x) cube of the next season; truth: 1 soybean, 2 corn."""
    rng = np.random.RandomState(seed)
    patterns = build_patterns()
    doy = DATES.dayofyear.to_numpy()
    values = np.zeros((len(DATES), rows, cols), dtype=np.float64)
    truth = np.zeros((rows, cols), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            name = "soybean" if x < cols // 2 else "corn"
            truth[y, x] = 1 if name == "soybean" else 2
            curve = patterns[name].to_numpy()
            # random phase jitter (a few days) + noise, like real field-to-field variability
            jitter = rng.randint(-8, 9)
            shifted = np.interp(doy + jitter, doy, curve, left=curve[0], right=curve[-1])
            values[:, y, x] = shifted + rng.normal(0, 0.03, len(DATES))
    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    cube = xr.DataArray(values, dims=("time", "y", "x"), name="ndvi",
                        coords={"time": DATES + pd.DateOffset(years=1),   # the next season
                                "y": transform.f - 30.0 * (np.arange(rows) + 0.5),
                                "x": transform.c + 30.0 * (np.arange(cols) + 0.5)})
    return cube.rio.write_crs("EPSG:32721").rio.write_transform(transform), truth


def main():
    print("Zeit Example 15: TWDTW (Time-Weighted Dynamic Time Warping) Classification")

    rows, cols = 20, 20
    print(f"\n[1/3] Generating synthetic single-season NDVI cube ({rows}x{cols} px)...")
    cube, truth = build_synthetic_cube(rows=rows, cols=cols)
    patterns = build_patterns()
    print(f"    {cube.sizes['time']} composites/season, patterns: {list(patterns)}.")

    print("\n[2/3] Running zeit.twdtw() (C++/OpenMP TWDTW against both patterns)...")
    # The patterns are from 2022 and the cube from 2023: with cycle="year" (the default)
    # the dates are matched by day of year, so the season matches across years.
    classes = zeit.twdtw(cube, patterns, max_elapsed=60)

    accuracy = (classes.label.values == truth).mean()
    print(f"    Classes: {dict(zip(classes.label.attrs['flag_values'], map(str, classes.pattern.values)))}")
    print(f"    Agreement with the injected ground truth: {accuracy:.1%}")
    print(f"    Mean TWDTW distance to the winning class: {float(classes.distance.mean()):.3f}")

    print("\n[3/3] Saving results with zeit.save_raster()...")
    out = zeit.save_raster(classes, os.path.join("data", "twdtw"))
    print(f"    label.tif (1=soybean, 2=corn), distance.tif, distances.tif -> {out}")

    print("\nDone!")


if __name__ == "__main__":
    main()
