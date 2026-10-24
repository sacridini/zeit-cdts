"""
Example 25: Emerging Hot Spots of Deforestation

Builds a synthetic space-time cube (no network needed): the hectares cleared each year,
2013-2024, in a grid of 1 km cells. Most of the landscape clears a little every year, at
random, and five blocks of cells behave differently:

* a frontier that opened in 2019 and kept clearing every year since;
* a frontier active every year, clearing more and more;
* a frontier active every year at a steady rate;
* a cluster that cleared only in the last year;
* a protected area where nothing is cleared.

`zeit.emerging_hotspots` computes the Getis-Ord Gi* of every cell and year, tests the
trend of each cell's Gi* series, and names its pattern as ESRI's Emerging Hot Spot
Analysis does. The pattern map is saved as a GeoTIFF.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import zeit

BLOCKS = {
    "frontier since 2019": (slice(5, 13), slice(5, 13)),
    "frontier, accelerating": (slice(5, 13), slice(27, 35)),
    "frontier, steady": (slice(27, 35), slice(5, 13)),
    "cleared in 2024 only": (slice(27, 35), slice(27, 35)),
    "protected area": (slice(16, 24), slice(16, 24)),
}


def build_cube(seed=25, n=40):
    rng = np.random.default_rng(seed)
    years = np.arange(2013, 2025)
    ha = rng.gamma(2.0, 1.0, (len(years), n, n))                 # background: about 2 ha a year
    for t, year in enumerate(years):
        a, b = BLOCKS["frontier since 2019"]
        if year >= 2019:
            ha[t, a, b] += rng.gamma(6.0, 1.0, (8, 8))
        a, b = BLOCKS["frontier, accelerating"]
        ha[t, a, b] += rng.gamma(2.0 + 0.6 * t, 1.0, (8, 8))
        a, b = BLOCKS["frontier, steady"]
        ha[t, a, b] += rng.gamma(6.0, 1.0, (8, 8))
        a, b = BLOCKS["cleared in 2024 only"]
        if year == 2024:
            ha[t, a, b] += rng.gamma(8.0, 1.0, (8, 8))
        a, b = BLOCKS["protected area"]
        ha[t, a, b] = 0.0
    coords = {"time": pd.to_datetime([f"{y}-01-01" for y in years]),
              "y": 8_900_000 - 500 - 1000 * np.arange(n), "x": 400_000 + 500 + 1000 * np.arange(n)}
    return xr.DataArray(ha, dims=("time", "y", "x"), coords=coords).rio.write_crs(32720)


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    cube = build_cube()
    print(f"Cube: {cube.sizes['y']} x {cube.sizes['x']} cells, {cube.sizes['time']} years")

    ehs = zeit.emerging_hotspots(cube, radius=1, lag=1)          # queen neighbours, the year before
    whole = zeit.emerging_hotspots(cube, reference="cube")       # every year against the whole cube
    names = zeit.emerging_hotspots.PATTERNS

    print("\nPattern at the centre of each block, each year measured against that year (default)"
          "\nand against the whole cube:")
    for block, (a, b) in BLOCKS.items():
        r, c = (a.start + a.stop) // 2, (b.start + b.stop) // 2
        print(f"  {block:>24}: {names[int(ehs.category.values[r, c])]:<23}"
              f"{names[int(whole.category.values[r, c])]:<23}"
              f"hot {int(ehs.n_hot.values[r, c]):>2} of 12 years, trend of the Gi* z = "
              f"{float(ehs.trend_z.values[r, c]):+.1f}")
    print("  The steady frontier and the protected area do not change, but the other frontiers raise"
          "\n  each year's mean: against that year they look diminishing, against the whole cube persistent.")

    print("\nCells by pattern:")
    codes, counts = np.unique(ehs.category.values, return_counts=True)
    for code, count in zip(codes, counts):
        print(f"  {names[int(code)]:>24}: {count}")

    zeit.save_raster(ehs.category, os.path.join(out, "emerging_hotspots.tif"))
    print(f"\nSaved {os.path.join(out, 'emerging_hotspots.tif')}")


if __name__ == "__main__":
    main()
