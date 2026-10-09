"""
Example 21: Forest Degradation with CODED End-to-End

Builds a synthetic Landsat series (no network needed) by mixing known fractions of the
Souza et al. (2005) endmembers: intact forest, forest selectively logged in 2006 (the canopy
closes again within two years; the lighter cuts are hard to see), forest cleared for pasture
in 2008, and pasture. Then:

1. `zeit.unmix` turns every observation into fractions of green vegetation, non-photosynthetic
   vegetation, soil, shade and cloud, and their NDFI;
2. `zeit.coded` monitors the NDFI and tells degradation from deforestation;
3. its strata map is sampled (`zeit.stratified_sample`), the sample labelled from the known
   truth, and `zeit.accuracy` estimates the area of degradation.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import zeit
from zeit._sma import endmember_table


def build_scene(seed=21, ny=60, nx=80):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2002-01-01", "2013-12-31", freq="16D")
    years = dates.year + (dates.dayofyear - 1) / 365.25
    kind = np.zeros((ny, nx), dtype=int)               # 0 forest
    kind[5:25, 10:40] = 1                              # logged in mid-2006
    kind[35:55, 15:45] = 2                             # cleared in mid-2008
    kind[:, 60:] = 3                                   # pasture all along
    intensity = rng.uniform(0.1, 1.0, (ny, nx))[..., None]   # light logging is hard to see
    E = endmember_table("souza2005").to_numpy()
    data = np.full((len(dates), 6, ny, nx), np.nan, dtype=np.float32)
    for i, t in enumerate(years):
        season = 0.03 * np.sin(2 * np.pi * t)
        forest = np.array([0.80 + season, 0.12, 0.05 - season / 3, 0.03, 0.0])
        pasture = np.array([0.30 + 2 * season, 0.05, 0.35, 0.30, 0.0])
        damage = max(0.0, 1 - (t - 2006.5) / 2.0) * (t >= 2006.5) * np.array([-0.35, 0.05, 0.22, 0.08, 0])
        cleared = forest if t < 2008.5 else pasture
        f = np.stack([forest, forest, cleared, pasture])[kind]             # (y, x, 5)
        f = f + (kind == 1)[..., None] * intensity * damage
        f = np.clip(f + rng.normal(0, 0.01, f.shape) * np.array([1, 1, 1, 1, 0]), 0, None)
        f /= f.sum(-1, keepdims=True)
        cloudy = rng.random((ny, nx)) < 0.35                              # clouds (masked), as in Landsat
        img = (f @ E) * 10000 + rng.normal(0, 20, (ny, nx, 6))
        img[cloudy] = np.nan
        data[i] = np.moveaxis(img, -1, 0)
    coords = {"time": dates, "band": ["blue", "green", "red", "nir", "swir1", "swir2"],
              "y": 8_900_000 - 15 - 30 * np.arange(ny), "x": 400_000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data, dims=("time", "band", "y", "x"), coords=coords).rio.write_crs(32721)
    return cube, kind


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    cube, kind = build_scene()

    print("[1/3] Unmixing every observation...")
    fractions = zeit.unmix(cube, cloud_threshold=0.05)
    for name, k in (("forest", 0), ("logged", 1), ("cleared", 2), ("pasture", 3)):
        print(f"    median NDFI of the {name} pixels: {float(np.nanmedian(fractions.ndfi.values[:, kind == k])):.2f}")

    print("[2/3] CODED...")
    result = zeit.coded(fractions, start=2005)
    zeit.save_raster(result[["strata", "forest", "n_events"]], os.path.join(out, "coded.tif"))
    names = result.strata.attrs["flag_meanings"].split()
    for code, name in enumerate(names, start=1):
        print(f"    {name:>13}: {int((result.strata.values == code).sum())} pixels")

    print("[3/3] Sample, reference, accuracy...")
    points = zeit.stratified_sample(result.strata, n=200, min_per_stratum=30)
    truth = np.array(["forest", "degradation", "deforestation", "non_forest"])[kind]
    points["ref"] = truth[points.row, points.col]
    acc = zeit.accuracy(result.strata, points)
    print(acc)


if __name__ == "__main__":
    main()
