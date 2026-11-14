"""zeit.fields_tiled against zeit.fields on the whole of a real window (phase 24g).

Run with the zeit environment's Python from PowerShell, on a quiet machine, at most 8 threads:

    python benchmarks/fields/tiled/lem_tiled.py [--tile 512] [--jobs 8] [--src PATH]

It runs ``zeit.fields`` (the package defaults) on the whole LEM 2020 mosaic (1334 x 1334, six bands), then
``zeit.fields_tiled`` with ``tile`` (about 9 tiles) and the same defaults, then ``zeit.fields`` again with
another seed (the Monte Carlo's own noise), and reports the wall times, the partition agreement and the
boundary F (1 px) of each against the whole-image result, and the extrapolated time of 1.2e9 pixels.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

for k, v in (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
             ("GDAL_NUM_THREADS", "4")):
    os.environ.setdefault(k, v)

import pyexpat  # noqa: F401  (before rasterio, in Git Bash's PATH)
import numpy as np
import rasterio
import xarray as xr

import zeit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vs_originals"))
from common import no_power_throttling  # noqa: E402

DEFAULT_SRC = r"C:\Users\eduar\Documents\dev\zeit_data\fields\mosaics\lem_1_2020.tif"


def matched(a, b, thr):
    """The share of ``a``'s pixels in fields that have a field of ``b`` with IoU >= ``thr`` (1 to 1 for thr > 0.5)."""
    _, ai = np.unique(a.ravel(), return_inverse=True)
    _, bi = np.unique(b.ravel(), return_inverse=True)
    na, nb = ai.max() + 1, bi.max() + 1
    co = np.bincount(ai * nb + bi, minlength=na * nb).reshape(na, nb)
    sa, sb = co.sum(1), co.sum(0)
    ok = (co / (sa[:, None] + sb[None, :] - co)).max(1) >= thr
    return float((sa * ok).sum() / sa.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--scan", action="store_true", help="also time tiles of 256, 512, 1024 and 2048")
    args = ap.parse_args()
    if args.jobs > 8:
        raise SystemExit("at most 8 threads")
    no_power_throttling()

    cube = zeit.load_raster(args.src)
    h, w = cube.sizes["y"], cube.sizes["x"]
    mpx = h * w / 1e6
    print(f"{args.src}: {cube.sizes['band']} bands, {h} x {w} = {mpx:.2f} Mpx")

    t = time.perf_counter()
    whole = zeit.fields(cube, random_state=0, n_jobs=args.jobs)
    t_whole = time.perf_counter() - t
    ref = whole["labels"].values
    ref_da = whole["labels"].assign_attrs(boundary_label=whole.attrs["boundary_label"])

    with tempfile.TemporaryDirectory() as d:
        t = time.perf_counter()
        res = zeit.fields_tiled(args.src, d, tile=args.tile, random_state=0, n_jobs=args.jobs)
        t_tiled = time.perf_counter() - t
        with rasterio.open(res["labels"]) as ds:
            lab = ds.read(1)
        tiled_da = xr.DataArray(lab, dims=("y", "x"), coords=ref_da.coords,
                                attrs={"boundary_label": res["boundary_label"]})
        t = time.perf_counter()
        other = zeit.fields(cube, random_state=1, n_jobs=args.jobs)
        t_other = time.perf_counter() - t
        other_da = other["labels"].assign_attrs(boundary_label=other.attrs["boundary_label"])

        rows = {}
        for name, pred, labels in (("tiled", tiled_da, lab), ("whole, other seed", other_da, other["labels"].values)):
            acc = zeit.segmentation_accuracy(pred, ref_da, tolerance=1)
            rows[name] = (acc.boundary.f, acc.boundary.precision, acc.boundary.recall, matched(ref, labels, 0.9), matched(ref, labels, 0.5),
                          matched(labels, ref, 0.5))
        n_fields = {"whole": int(ref.max() - whole.attrs["boundary_label"]), "tiled": res["n_fields"],
                    "whole, other seed": int(other["labels"].values.max() - other.attrs["boundary_label"])}
        n_poly = res["n_polygons"]

    print(f"\nwall time, {args.jobs} threads: whole {t_whole:.1f} s ({t_whole / mpx:.2f} s/Mpx), "
          f"tiled {t_tiled:.1f} s ({t_tiled / mpx:.2f} s/Mpx; {res['tiles'][0]} x {res['tiles'][1]} tiles of "
          f"{args.tile}, overlap {res['overlap']}, polygons included), whole again {t_other:.1f} s")
    print(f"fields: {n_fields}; polygons of the tiled run: {n_poly}")
    print("\nagainst the whole-image result (seed 0)         F(1px)  prec   rec    IoU>=0.9  IoU>=0.5  (reverse)")
    for name, (f, p, r, m9, m5, m5r) in rows.items():
        print(f"  {name:<20s}                          {f:.3f}  {p:.3f}  {r:.3f}  {m9:.3f}     {m5:.3f}     {m5r:.3f}")
    # The cost is the pixels of the extended windows: 1.54 x the raster's here, and (1 + 2 overlap / tile)^2 = 1.34 x
    # in the middle of a raster of any size (the edge tiles have less window).
    factor_here = res["window_pixels"] / (h * w)
    factor_far = ((args.tile + 2 * res["overlap"]) / args.tile) ** 2
    secs = t_tiled / mpx * factor_far / factor_here * 1200.0
    print(f"\nextrapolation to 1.2e9 pixels with tile {args.tile} (windows {factor_far:.2f} x the raster), {args.jobs} "
          f"threads: {secs:,.0f} s = {secs / 3600:.1f} h ({secs / 1200:.2f} s/Mpx)")

    if args.scan:  # the wall time against the tile size (no polygons)
        print("\ntile  tiles  windows (Mpx)  seconds  s per window Mpx")
        for tile in (256, 512, 1024, 2048):
            with tempfile.TemporaryDirectory() as d:
                t = time.perf_counter()
                r = zeit.fields_tiled(args.src, d, tile=tile, random_state=0, n_jobs=args.jobs, polygons=False)
                dt = time.perf_counter() - t
            wm = r["window_pixels"] / 1e6
            print(f"{tile:5d}  {r['tiles'][0]} x {r['tiles'][1]}  {wm:9.2f}  {dt:9.1f}  {dt / wm:9.2f}")


if __name__ == "__main__":
    main()
