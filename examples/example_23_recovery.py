"""
Example 23: Recovery after Disturbance

Builds a synthetic annual NDVI stack (no network needed), 1990-2024, on a small grid:
intact forest, forest cleared in 2000 that regrows fast, forest cleared in 2000 that regrows
slowly, and forest cleared in 2005 and kept as pasture. Then:

1. `zeit.landtrendr` segments every pixel and `zeit.extract_events` picks its greatest loss;
2. `zeit.recovery_metrics` measures how each pixel came back after that loss: the years to
   recovery (Y2R), R80P, RRI, ARI and YrYr;
3. the same metrics are measured on the yearly values themselves, with the events given.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import zeit

YEARS = np.arange(1990, 2025)


def build_stack(seed=23, ny=40, nx=60):
    """NDVI x 10000, one band per year, with the year of each band in `time`."""
    rng = np.random.default_rng(seed)
    kind = np.zeros((ny, nx), dtype=int)       # 0 forest
    kind[:20, 20:40] = 1                       # cleared in 2000, back to forest in about 6 years
    kind[20:, 20:40] = 2                       # cleared in 2000, slow regrowth
    kind[:, 40:] = 3                           # cleared in 2005, pasture since
    data = np.empty((len(YEARS), ny, nx), dtype=np.float32)
    for i, year in enumerate(YEARS):
        forest = 8200.0
        fast = forest if year < 2000 else 4000 + 4200 * (1 - np.exp(-(year - 2000) / 3))
        slow = forest if year < 2000 else 4000 + 4200 * (1 - np.exp(-(year - 2000) / 12))
        pasture = forest if year < 2005 else 5000.0
        data[i] = np.array([forest, fast, slow, pasture])[kind] + rng.normal(0, 250, (ny, nx))
    coords = {"time": pd.to_datetime([f"{y}-01-01" for y in YEARS]),
              "y": 8_900_000 - 15 - 30 * np.arange(ny), "x": 400_000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data, dims=("time", "y", "x"), coords=coords).rio.write_crs(32720)
    return cube, kind


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    ndvi, kind = build_stack()
    names = ["forest", "fast regrowth", "slow regrowth", "pasture"]

    print("[1/3] LandTrendr and its greatest loss per pixel...")
    lt = zeit.landtrendr(ndvi, direction="loss")
    loss = zeit.extract_events(lt, min_magnitude=2000)
    print(f"    pixels with a loss: {int((loss.yod > 0).sum())} of {loss.yod.size}")

    print("[2/3] Recovery metrics on LandTrendr's fitted trajectory...")
    rec = zeit.recovery_metrics(lt, loss)
    for k, name in enumerate(names):
        m = rec.where(xr.DataArray(kind == k, dims=("y", "x")))
        n_loss = int((m.yod > 0).sum())
        if n_loss == 0:
            print(f"    {name:>13}: no loss")
            continue
        recovered = float(np.isfinite(m.y2r).sum() / n_loss)
        print(f"    {name:>13}: median Y2R {float(m.y2r.median()):4.1f} years, "
              f"R80P {float(m.r80p.median()):.2f}, RRI {float(m.rri.median()):.2f}, "
              f"recovered {recovered:.0%}")
    zeit.save_raster(rec[["y2r", "r80p", "rri"]], os.path.join(out, "recovery"))

    print("[3/3] The same metrics on the yearly values, after the same events...")
    raw = zeit.recovery_metrics(ndvi, loss)
    print(f"    median R80P of the fast regrowth: {float(raw.r80p.where(kind == 1).median()):.2f} "
          f"(on the yearly values) vs {float(rec.r80p.where(kind == 1).median()):.2f} (on the fit)")


if __name__ == "__main__":
    main()
