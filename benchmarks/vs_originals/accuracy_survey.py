"""zeit.transitions against its original (accuracy_survey.R): the area of each transition between two maps of
classes, by the direct, difference and ratio estimators of a stratified sample whose strata are not the
transitions, against terra (the mapped areas) and R's survey package (the estimates, svytotal and svyratio).

    python benchmarks/vs_originals/accuracy_survey.py

A made-up landscape of 2048 x 2048 pixels of 30 m: Forest, Pasture, Crop and Water at two dates (patches cleared,
cropped and regrown between them), two maps of the dates with errors, strata from an older alert map (its alerts, a
buffer of 2 pixels, the rest) and 250 + 200 + 150 points, the truth their reference. Both read the three rasters
from files; zeit's estimates (areas, standard errors, user's, producer's and overall accuracies) must agree with
survey's to 1e-9.
"""
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import binary_dilation, gaussian_filter

import zeit
from common import best_of, record, run_r, workdir

SIZE = 2048
SIZES = (250, 200, 150)


def write(path, a):
    prof = dict(driver="GTiff", width=SIZE, height=SIZE, count=1, dtype="uint8", crs="EPSG:32723", nodata=0,
                transform=from_origin(270000, 7500000, 30, 30), tiled=True, compress="deflate")
    with rasterio.open(path, "w", **prof) as d:
        d.write(a, 1)


def landscape(d):
    rng = np.random.default_rng(20141114)

    def patches(sigma, share):
        f = gaussian_filter(rng.standard_normal((SIZE, SIZE)), sigma)
        return f > np.quantile(f, 1 - share)

    f = gaussian_filter(rng.standard_normal((SIZE, SIZE)), 30)
    ref_from = (np.digitize(f, np.quantile(f, [0.45, 0.75, 0.95])) + 1).astype(np.uint8)
    ref_to = ref_from.copy()
    ref_to[patches(10, 0.06) & (ref_from == 1)] = 2
    ref_to[patches(8, 0.05) & (ref_from == 2)] = 3
    ref_to[patches(6, 0.03) & (ref_from == 3)] = 1

    def flip(a, p):
        k = rng.random(a.shape) < p
        out = a.copy()
        out[k] = ((a[k] - 1 + rng.integers(1, 4, k.sum())) % 4 + 1).astype(np.uint8)
        return out

    map_from = flip(ref_from, 0.06)
    changed = ref_from != ref_to
    map_to = map_from.copy()
    seen = changed & (rng.random(changed.shape) < 0.8)
    map_to[seen] = ref_to[seen]
    map_to[~changed] = flip(map_from, 0.015)[~changed]
    alert = (changed & (rng.random(changed.shape) < 0.7)) | (rng.random(changed.shape) < 0.01)
    strata = np.where(alert, 2, np.where(binary_dilation(alert, np.ones((5, 5), bool)), 3, 1)).astype(np.uint8)
    for name, a in (("strata", strata), ("map_from", map_from), ("map_to", map_to)):
        write(d / f"{name}.tif", a)
    cells = np.concatenate([rng.choice(np.flatnonzero(strata.ravel() == h), k, replace=False)
                            for h, k in zip((1, 2, 3), SIZES)])
    rows, cols = np.divmod(cells, SIZE)
    rf, rt = ref_from.ravel()[cells], ref_to.ravel()[cells]
    (d / "points.csv").write_text("row,col,ref_from,ref_to\n" + "".join(
        f"{r},{c},{a},{b}\n" for r, c, a, b in zip(rows, cols, rf, rt)))
    return rows, cols, rf, rt


def worst(ours, theirs):
    """The largest difference relative to max(|R's value|, 1) (accuracies near 0, areas in hectares); NaN as NA."""
    w = 0.0
    for a, b in zip(ours, theirs):
        a, b = float(a), np.nan if b is None else float(b)
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), (a, b)
            continue
        w = max(w, abs(a - b) / max(abs(b), 1.0))
    return w


def main():
    import geopandas as gpd

    d = workdir("accuracy_survey")
    rows, cols, rf, rt = landscape(d)
    with rasterio.open(d / "strata.tif") as s:
        x, y = s.transform * (cols + 0.5, rows + 0.5)
        crs = s.crs
    pts = gpd.GeoDataFrame({"ref_from": rf.astype(int), "ref_to": rt.astype(int)},
                           geometry=gpd.points_from_xy(x, y), crs=crs)
    for estimator in ("direct", "difference", "ratio"):
        t_r = run_r("accuracy_survey.R", d, estimator)
        theirs = json.loads((d / f"survey_{estimator}.json").read_text())
        out = {}

        def ours():
            out["t"] = zeit.transitions(str(d / "map_from.tif"), str(d / "map_to.tif"), pts,
                                        strata=str(d / "strata.tif"), estimator=estimator)

        t_z = best_of(ours)
        t = out["t"]
        a, b = [], []
        for k, v in theirs["classes"].items():
            a += [t.area.mapped[k], t.area.estimate[k], t.area.se[k], t.users.estimate[k], t.users.se[k],
                  t.producers.estimate[k], t.producers.se[k]]
            b += [v["mapped"], *v["area"], *v["users"], *v["producers"]]
        a += [t.overall["estimate"], t.overall["se"]]
        b += theirs["overall"]
        diff = worst(a, b)
        assert diff < 1e-9, diff
        record(f"transitions ({estimator})", f"survey {theirs['survey']} + terra (R)",
               f"{SIZE}x{SIZE} px, {sum(SIZES)} points", t_r, t_z, 1,
               f"{len(theirs['classes'])} transitions; largest difference {diff:.1e}")


if __name__ == "__main__":
    main()
