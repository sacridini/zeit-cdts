"""
Example 26: Spectral Unmixing with Endmembers of Your Own

Builds a synthetic Landsat series (no network needed): six annual images, 2019-2024, of
60 x 60 pixels in six bands (blue, green, red, NIR, SWIR1, SWIR2), each pixel a known mix
of green vegetation, soil and water plus sensor noise. A pasture is ploughed in 2021 and
grows back; a reservoir fills from 2022.

1. `zeit.unmix` with the three endmembers as a table: the fractions of every pixel and
   year, checked against the known ones, with the RMSE of the rebuilt spectra;
2. the same with fewer constraints (non-negative only, then free), to see what the
   constraints do;
3. the built-in endmembers of Souza et al. (2005) and their NDFI.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

import zeit

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
# reflectance (0-1) of each material in each band: rows endmembers, columns bands of the cube
ENDMEMBERS = pd.DataFrame(
    {"blue": [0.03, 0.10, 0.06], "green": [0.06, 0.14, 0.05], "red": [0.04, 0.18, 0.03],
     "nir": [0.40, 0.26, 0.02], "swir1": [0.20, 0.34, 0.01], "swir2": [0.09, 0.30, 0.005]},
    index=["vegetation", "soil", "water"])


def build_scene(seed=26, n=60):
    rng = np.random.default_rng(seed)
    years = np.arange(2019, 2025)
    E = ENDMEMBERS[BANDS].to_numpy()
    truth = np.empty((len(years), n, n, 3))
    base = rng.dirichlet([6, 2, 0.2], (n, n))                     # mostly vegetation, some soil
    for t, year in enumerate(years):
        f = base.copy()
        ploughed = max(0.0, 1 - 0.4 * (year - 2021)) if year >= 2021 else 0.0
        f[10:30, 10:40] = (1 - 0.85 * ploughed) * np.array([0.8, 0.2, 0]) + 0.85 * ploughed * np.array([0, 1, 0])
        if year >= 2022:                                          # the reservoir fills
            r = 6 + 4 * (year - 2022)
            yy, xx = np.ogrid[:n, :n]
            lake = (yy - 45) ** 2 + (xx - 45) ** 2 <= r ** 2
            f[lake] = [0.05, 0.05, 0.90]
        truth[t] = f
    refl = truth @ E + rng.normal(0, 0.004, (len(years), n, n, len(BANDS)))
    coords = {"time": pd.to_datetime([f"{y}-07-01" for y in years]), "band": BANDS,
              "y": 8_900_000 - 15 - 30 * np.arange(n), "x": 400_000 + 15 + 30 * np.arange(n)}
    cube = xr.DataArray(np.moveaxis(refl, -1, 1).astype(np.float32), dims=("time", "band", "y", "x"),
                        coords=coords).rio.write_crs(32721)
    return cube, truth


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    cube, truth = build_scene()
    print(f"Cube: {dict(cube.sizes)}")

    print("\n[1/3] Fully constrained (fractions >= 0, adding up to 1)")
    fr = zeit.unmix(cube, ENDMEMBERS)                             # vegetation, soil, water, rmse
    for k, name in enumerate(ENDMEMBERS.index):
        err = np.abs(fr[name].values - truth[..., k])
        print(f"  {name:>10}: mean fraction {float(fr[name].mean()):.3f}, "
              f"mean error against the truth {err.mean():.4f}")
    print(f"  RMSE of the rebuilt spectra: median {float(fr.rmse.median()):.4f} (the noise is 0.004)")
    field = fr.isel(y=slice(10, 30), x=slice(10, 40)).mean(["y", "x"])
    print("  The ploughed field, soil fraction by year:",
          " ".join(f"{int(y)}: {v:.2f}" for y, v in zip(field.time.dt.year.values, field.soil.values)))
    zeit.save_raster(fr[["vegetation", "soil", "water", "rmse"]].isel(time=-1),
                     os.path.join(out, "unmixing_2024.tif"))

    print("\n[2/3] Fewer constraints")
    for label, kw in (("non-negative", dict(sum_to_one=False)),
                      ("free", dict(sum_to_one=False, nonneg=False))):
        f = zeit.unmix(cube, ENDMEMBERS, **kw)
        total = f.vegetation + f.soil + f.water
        print(f"  {label:>12}: sum of the fractions {float(total.min()):.2f} to {float(total.max()):.2f}, "
              f"smallest fraction {float(min(f[v].min() for v in ENDMEMBERS.index)):+.3f}, "
              f"median RMSE {float(f.rmse.median()):.4f}")

    print("\n[3/3] Souza et al. (2005): green vegetation, NPV, soil, shade, cloud and the NDFI")
    souza = zeit.unmix(cube)                                      # endmembers="souza2005"
    print(f"  variables: {list(souza.data_vars)}")
    print(f"  NDFI of the vegetated pixels in 2019: {float(souza.ndfi.isel(time=0, y=slice(40, 60), x=slice(0, 20)).median()):.2f}")
    print(f"  median RMSE {float(souza.rmse.median()):.4f}: these endmembers are not the ones the scene was mixed from")
    print(f"\nSaved {os.path.join(out, 'unmixing_2024.tif')}")


if __name__ == "__main__":
    main()
