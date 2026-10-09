"""
Example 22: Foundation-Model Embeddings (TESSERA and AlphaEarth)

Reads the yearly embeddings of a 5 x 5 km box in Rondonia, Brazil (needs a network: a few
hundred MB), from AlphaEarth Foundations (open COGs, nothing to install) and, when the
`geotessera` library is installed (`pip install zeit-cdts[tessera]`, Python 3.12+), from
TESSERA. Then:

1. `zeit.plot` saves the first three principal components of 2017 and 2025 as a figure;
2. `zeit.som` clusters the pixels of 2025 without labels;
3. `zeit.similarity` maps, every year, how much each pixel looks like one chosen pixel;
4. `zeit.embedding_change` and `zeit.extract_events` date the changes from year to year, and
   `zeit.agreement` compares the two products' dates.
"""
import os

import numpy as np

import zeit

BOX = (-62.95, -9.95, -62.90, -9.90)
YEARS = range(2017, 2026)
OUT = "outputs/example_22"


def report(name, emb):
    events = zeit.extract_events(zeit.embedding_change(emb), min_magnitude=THRESHOLD[name])
    years, counts = np.unique(events.yod.values[events.yod.values > 0], return_counts=True)
    print(f"{name}: changes per year (yod) {dict(zip(years.tolist(), counts.tolist()))}")
    return events


# Each product moves on its own scale (TESSERA's pixels about 0.05-0.1 a year, AlphaEarth's 0.02).
THRESHOLD = {"alphaearth": 0.2, "tessera": 0.3}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cubes = {"alphaearth": zeit.load_embeddings(BOX, source="alphaearth", years=YEARS, chunks=None)}
    try:
        cubes["tessera"] = zeit.load_embeddings(BOX, source="tessera", years=YEARS, chunks=None)
    except ImportError as e:
        print(f"TESSERA skipped: {e}")
    for name, emb in cubes.items():
        print(f"{name}: {dict(emb.sizes)}, {emb.rio.crs}, {emb.attrs['embedding_license']}")
        zeit.save_raster(emb.sel(time="2025"), f"{OUT}/{name}_2025.tif")

    aef = cubes["alphaearth"]
    zeit.plot(aef, time=["2017", "2025"], static=True, save=f"{OUT}/alphaearth_pca.png")

    clusters = zeit.som(aef.sel(time="2025"), x=2, y=3, sample=20_000)
    print("SOM clusters of 2025:", np.bincount(clusters.label.values.ravel()))

    centre = aef.isel(time=-1, y=aef.sizes["y"] // 2, x=aef.sizes["x"] // 2)
    like_centre = zeit.similarity(aef, centre.values)
    print("pixels at least 0.9 similar to the centre, per year:",
          (like_centre > 0.9).sum(["y", "x"]).values.tolist())

    events = {name: report(name, emb) for name, emb in cubes.items()}
    if len(events) == 2:
        agree = zeit.agreement(events, tolerance=1)
        both = int((agree.n_detected == 2).sum())
        print(f"both products see a change in {both} pixels; they agree on its year (within one) in "
              f"{int((agree.n_agree == 2).sum())}")
        zeit.save_raster(agree, f"{OUT}/agreement")
