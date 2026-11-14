"""Generate the numbers and figures of the field-boundaries tutorial (docs/tutorials/fields.md).

The example compares the field-size distribution of 2000 with 2019 in one 30 x 30 km window of the
Paraguayan Dry Chaco. Every number in the tutorial comes from this script. From the repository root,
with the zeit environment (on Windows, from PowerShell, with the environment's folders first on PATH):

    python docs/scripts/make_fields_figures.py plan        # scenes and the bytes estimate; no download
    python docs/scripts/make_fields_figures.py download    # the two Landsat medoid mosaics (within --max-gb)
    python docs/scripts/make_fields_figures.py run         # fields, polygons, numbers and figures

Data (outside git) goes to ``ZEIT_FIELDS_DATA/tutorial`` (default
``C:/Users/eduar/Documents/dev/zeit_data/fields/tutorial``; see ``benchmarks/fields/validation/common.py``);
the figures go to ``docs/assets/figures/``.

The mosaics are built by ``benchmarks/fields/validation/mosaics.py`` (the same code and the same
procedure as the validation windows of the sensitivity analysis): Landsat Collection 2 Level-2 from the
Planetary Computer, a medoid over the six reflective bands of the clear observations of the dry season
(1 June to 30 September), surface reflectance x 10000 as int16, on Landsat's own 30 m lattice in the
scenes' UTM zone (EPSG:32620, the zone Landsat uses for this area, with the southern hemisphere's negative
northings). 2000: Landsat 5 TM and Landsat 7 ETM+ (before the SLC failure of 2003); 2019: Landsat 8.
Clear: QA_PIXEL with fill, dilated cloud, cirrus, cloud and cloud shadow at 0 and the clear bit at 1.

The random seed is fixed (``SEED``) and the result does not depend on ``n_jobs``.
"""

import pyexpat  # noqa: F401  (on Windows, load the environment's libexpat before rasterio can find Git's)
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "benchmarks" / "fields" / "validation"))

EPSG = 32620
# 30 x 30 km (1000 x 1000 px), edges on Landsat's lattice (15 mod 30 m), UTM 20 (negative northings)
BOUNDS = (749985, -2539995, 779985, -2509995)
YEARS = {2000: {"platforms": ["landsat-5", "landsat-7"], "start": "2000-06-01", "end": "2000-09-30"},
         2019: {"platforms": ["landsat-8"], "start": "2019-06-01", "end": "2019-09-30"}}
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
CLOUD = 40
SEED = 0
N_JOBS = 4
MASK_THRESHOLD = 2000       # swir16 x 10000: the notebook's forest threshold
BINS_HA = [0, 1, 5, 20, 100, 500, np.inf]


def data_dir():
    from common import data_dir as dd

    p = dd() / "tutorial"
    p.mkdir(parents=True, exist_ok=True)
    return p


def windows():
    import mosaics

    out = []
    for year, kw in YEARS.items():
        x0, y0, x1, y1 = BOUNDS
        out.append({"name": "chaco_boqueron", "group": "tutorial", "year": year, "epsg": EPSG,
                    "bounds": list(BOUNDS), "shape": [(y1 - y0) // 30, (x1 - x0) // 30],
                    "why": "tutorial window, Paraguayan Dry Chaco south-west of Filadelfia", **kw})
    assert mosaics.snap(x0, x1) == (x0, x1) and mosaics.snap(y0, y1) == (y0, y1), "bounds off Landsat's lattice"
    return out


# ------------------------------------------------------------------------------------------------ data

def plan(args):
    import mosaics

    tile_bytes, total = None, 0
    for w in windows():
        items = mosaics.search(w, CLOUD)
        if tile_bytes is None:
            tile_bytes = mosaics.calibrate(items)
        est = mosaics.estimate_bytes(w, len(items), tile_bytes)
        total += est
        plats = sorted({i.properties["platform"] for i in items})
        print(f"{w['year']}: {len(items)} scenes {plats}, estimate {est / 1e6:.0f} MB (x 1.6: {est * 1.6 / 1e6:.0f} MB)")
    print(f"estimate total {total / 1e9:.2f} GB (cap {args.max_gb} GB)")
    (data_dir() / "plan.json").write_text(json.dumps({"tile_bytes": tile_bytes, "total": total}))


def download(args):
    import mosaics

    mosaics.THREADS = 4     # dask workers reading the COGs (the machine's cap is shared)
    plan_json = json.loads((data_dir() / "plan.json").read_text())
    if plan_json["total"] * 1.6 / 1e9 > args.max_gb:
        sys.exit(f"estimate x 1.6 passes the {args.max_gb} GB cap: not downloading")
    spent = 0
    for w in windows():
        if (data_dir() / f"{w['name']}_{w['year']}.json").exists():
            print(f"{w['year']} exists")
            continue
        prov = mosaics.build(w, CLOUD, plan_json["tile_bytes"], out=data_dir())
        b = prov["bytes"]["network_received"] or prov["bytes"]["estimated"] or 0
        spent += b
        print(f"{w['year']}: scenes {prov['scenes']['n']}, valid {prov['valid_pct']}%, median clear "
              f"{prov['median_clear_observations']}, {prov['seconds']['total']} s, received {b / 1e6:.0f} MB "
              f"(estimate {(prov['bytes']['estimated'] or 0) / 1e6:.0f} MB); total {spent / 1e9:.2f} GB")


# ----------------------------------------------------------------------------------------------- analysis

def load_mosaic(year):
    """The mosaic as a (band, y, x) float32 DataArray in reflectance x 10000, NaN where there is no value."""
    import rasterio
    import xarray as xr

    with rasterio.open(data_dir() / f"chaco_boqueron_{year}.tif") as src:
        a = src.read().astype("float32")
        a[a == src.nodata] = np.nan
        t, crs = src.transform, src.crs
    da = xr.DataArray(a, dims=("band", "y", "x"),
                      coords={"band": BANDS, "y": t.f + t.e * (np.arange(a.shape[1]) + 0.5),
                              "x": t.c + t.a * (np.arange(a.shape[2]) + 0.5)})
    return da.rio.write_crs(crs)


def summarize(gdf, label_map):
    """Number of fields, quantiles of area (ha), the share of the window in fields and the area-weighted median
    (the area such that half of the fields' area is in objects at least that large)."""
    a = gdf["area_ha"].to_numpy()
    srt = np.sort(a)
    cum = np.cumsum(srt) / srt.sum()
    q = {f"p{p}": float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)}
    return {"n_fields": int(len(a)), "area_ha_quantiles": q, "mean_ha": float(a.mean()),
            "area_weighted_median_ha": float(srt[np.searchsorted(cum, 0.5)]),
            "share_of_window_in_fields_pct": float((label_map > 0).mean() * 100),
            "fields_area_ha": float(a.sum()), "largest_ha": float(a.max()),
            "bins_ha": {f"{lo:g}-{hi:g}": int(((a >= lo) & (a < hi)).sum())
                        for lo, hi in zip(BINS_HA[:-1], BINS_HA[1:])}}


def run(args):
    import zeit

    d = data_dir()
    out = {"window": {"epsg": EPSG, "bounds": list(BOUNDS), "random_state": SEED, "n_jobs": N_JOBS}}
    results = {}
    for year in YEARS:
        img = load_mosaic(year)
        swir = img.sel(band="swir16")
        valid = float(np.isfinite(swir).mean() * 100)
        forest = float((swir <= MASK_THRESHOLD).where(np.isfinite(swir)).mean() * 100)
        row = {"valid_pct": valid, "swir16_le_2000_pct_of_valid": forest}
        for variant in ("plain", "forest_mask", "forest_mask_dense"):
            t0 = time.time()
            # a stack gives the multitemporal gradient; the mask must be given as an array (mask_threshold
            # needs one 2-D image), and swir16 > 2000 is exactly the notebook's mask_threshold=2000.
            # "dense": twice the marker density, to see how much the sizes follow it
            kw = {} if variant == "plain" else {"mask": swir > MASK_THRESHOLD}
            if variant == "forest_mask_dense":
                kw["marker_density"] = 2000
            res = zeit.fields(img, random_state=SEED, n_jobs=N_JOBS, **kw)
            dt = time.time() - t0
            gdf = zeit.fields_to_polygons(res)
            lab = res["labels"].values
            if res.attrs.get("boundary_label") == 1:
                lab = np.where(lab == 1, 0, lab)
            row[variant] = {"seconds": round(dt, 1), "markers": int(res.attrs["markers"]),
                            "realizations": int(res.attrs["realizations"]), **summarize(gdf, lab)}
            results[(year, variant)] = (res, gdf)
            gdf.to_file(d / f"fields_{year}_{variant}.gpkg", driver="GPKG")
            print(year, variant, json.dumps(row[variant]))
        out[str(year)] = row
    (d / "numbers.json").write_text(json.dumps(out, indent=1))
    figures(results)


def figures(results):
    import make_figures as mf
    import matplotlib.pyplot as plt

    mf.setup_style()
    imgs = {y: load_mosaic(y) for y in YEARS}
    pooled = np.concatenate([np.stack([im.sel(band=b).values.ravel() for b in ("swir16", "nir08", "red")])
                             for im in imgs.values()], axis=1)
    lo, hi = np.percentile(pooled, [2, 98], axis=1)
    stretch = [(l, h) for l, h in zip(lo, hi)]
    x0, y0, x1, y1 = BOUNDS
    extent = [x0, x1, y0, y1]

    for year in YEARS:
        im = imgs[year]
        rgb = np.stack([np.clip((im.sel(band=b).values - l) / (h - l), 0, 1)
                        for b, (l, h) in zip(("swir16", "nir08", "red"), stretch)], axis=-1)
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
        axes[0].imshow(rgb, extent=extent)
        mf.map_axes(axes[0], f"{year}: SWIR1 / NIR / red")
        for ax, variant, title in ((axes[1], "plain", "No mask"), (axes[2], "forest_mask", "Forest mask")):
            res, gdf = results[(year, variant)]
            ax.imshow(rgb, extent=extent)
            gdf.boundary.plot(ax=ax, color="#ffd24a", linewidth=0.35)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_xlabel("")
            ax.set_ylabel("")
            mf.map_axes(ax, f"{title}: {len(gdf):,} fields")
        fig.subplots_adjust(wspace=0.03)
        mf.save(fig, f"fields_{year}", photo=True)

    # the size distributions of the forest-mask variant (the one a Chaco user wants)
    colors = {2000: mf.BLUE, 2019: mf.ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    edges = np.logspace(-1, 3, 25)
    for year in YEARS:
        a = results[(year, "forest_mask")][1]["area_ha"].to_numpy()
        axes[0].hist(a, bins=edges, histtype="step", color=colors[year], linewidth=2, label=f"{year} ({len(a):,})")
        s = np.sort(a)
        axes[1].plot(s, np.cumsum(s) / s.sum(), color=colors[year], label=str(year))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Field area (ha)")
    axes[0].set_ylabel("Number of fields")
    axes[0].set_title("How many fields of each size")
    axes[0].legend(title="year (fields)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Field area (ha)")
    axes[1].set_ylabel("Share of the fields' area in fields up to this size")
    axes[1].set_title("Where the area is")
    axes[1].legend()
    fig.tight_layout()
    mf.save(fig, "fields_sizes")

    # the gallery thumbnail: the 2019 map, as the other tutorials do
    from PIL import Image
    src = mf.OUT / "fields_2019.webp"
    (mf.OUT / "thumbs").mkdir(exist_ok=True)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im = im.crop((w // 3, 0, w, h))        # the two panels with boundaries
    canvas = Image.new("RGB", (800, 450), "white")
    im.thumbnail((760, 410), Image.LANCZOS)
    canvas.paste(im, ((800 - im.width) // 2, (450 - im.height) // 2))
    canvas.save(mf.OUT / "thumbs" / "fields.webp", quality=85, method=6)
    print("  wrote the gallery thumbnail")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "download", "run"])
    ap.add_argument("--max-gb", type=float, default=3.0)
    args = ap.parse_args()
    os.environ.setdefault("GDAL_NUM_THREADS", "4")
    {"plan": plan, "download": download, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
