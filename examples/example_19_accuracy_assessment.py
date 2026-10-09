"""
Example 19: Accuracy Assessment and Area Estimation End-to-End

Builds a synthetic Landsat-like cube where two patches are cleared (no network
needed), maps the loss with LandTrendr, draws a stratified random sample of the
map with `zeit.stratified_sample`, labels it, and estimates the map's accuracy and
the error-adjusted area of loss with `zeit.accuracy` (Olofsson et al. 2014).

The labels would normally come from an interpreter looking at the imagery, with
`zeit.interpret`. Run this script with `--interpret` to label the points yourself;
without it, the script labels them from the synthetic truth (with some mistakes
of the map's to find), so it runs unattended.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import zeit


def build_synthetic_cube(seed=19):
    """Blue, green, red, NIR and SWIR1 for 15 years; two patches are cleared in 2008
    and 2012, and a third one only thins a little in 2010 (a change the map may miss)."""
    rng = np.random.default_rng(seed)
    years = pd.date_range("2005-07-01", periods=15, freq="12MS")
    ny, nx = 150, 200
    forest = np.array([0.03, 0.05, 0.03, 0.32, 0.14])
    bare = np.array([0.08, 0.11, 0.13, 0.22, 0.28])
    when = np.zeros((ny, nx), dtype=int)
    when[20:70, 30:100] = 2008
    when[90:130, 120:180] = 2012
    thinned = np.zeros((ny, nx), dtype=bool)
    thinned[100:140, 20:60] = True
    data = np.empty((len(years), 5, ny, nx), dtype=np.float32)
    for t, year in enumerate(years):
        img = forest[:, None, None] + rng.normal(0, 0.008, (5, ny, nx))
        cleared = (when > 0) & (year.year >= when)
        img[:, cleared] = bare[:, None] + rng.normal(0, 0.01, (5, int(cleared.sum())))
        if year.year >= 2010:
            img[:, thinned] += ((bare - forest) * 0.15)[:, None]
        data[t] = img
    coords = {"time": years, "band": ["blue", "green", "red", "nir", "swir1"],
              "y": 8_900_000 - 15 - 30 * np.arange(ny), "x": 400_000 + 15 + 30 * np.arange(nx)}
    cube = xr.DataArray(data, dims=("time", "band", "y", "x"), coords=coords).rio.write_crs(32721)
    truth = xr.DataArray(np.where(thinned & (when == 0), 2010, when), dims=("y", "x"),
                         coords={"y": coords["y"], "x": coords["x"]}).rio.write_crs(32721)
    return cube, truth


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    cube, truth = build_synthetic_cube()

    # 1. The map: the greatest NBR loss per pixel
    nir, swir = cube.sel(band="nir"), cube.sel(band="swir1")
    nbr = ((nir - swir) / (nir + swir) * 10000).rename("nbr")
    lt = zeit.landtrendr(nbr)
    loss = zeit.extract_events(lt, min_magnitude=1500)

    # 2. The sample: change is rare, so it gets its own stratum and at least 50 points
    design = zeit.sampling_design(loss, expected_ua={"change": 0.8, "no change": 0.95}, std_error=0.015)
    print(design, "\n")
    points = zeit.stratified_sample(loss, design=design)

    # 3. The reference labels
    if "--interpret" in sys.argv:
        session = zeit.interpret(cube, points, rgb=["swir1", "nir", "red"], series=nbr, map=loss, fit=lt,
                                 save=os.path.join(out, "example_19_reference.gpkg"))
        labelled = session.samples
    else:
        years = truth.values[points.row, points.col]
        points["ref"] = np.where(years > 0, "change", "no change")
        points["ref_date"] = [f"{y}-07-01" if y else None for y in years]
        labelled = points

    # 4. Accuracy and the error-adjusted area of loss
    acc = zeit.accuracy(loss, labelled, date_tolerance=1)
    print(acc)
    change = acc.area.loc["change"]
    print(f"\nmapped loss {change.mapped:,.0f} ha; estimated {change.estimate:,.0f} ± {change.ci:,.0f} ha "
          f"(the real area is {(truth.values > 0).sum() * 0.09:,.0f} ha)")
    labelled.to_file(os.path.join(out, "example_19_points.gpkg"))


if __name__ == "__main__":
    main()
