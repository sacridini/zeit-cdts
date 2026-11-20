"""Landsat 8/9 (Collection 2 Level 2) and MODIS NBAR (MCD43A4, modis-43A4-061) of a small area, from the Planetary
Computer, cached as an npz for the leave-one-date-out experiment of ``fusion_loo.py``.

    python benchmarks/vs_originals/fusion_real.py SITE        # SITE: bahia | goias

Read-only network access (public STAC, signed COG reads of a window of 15 x 15 km); nothing is committed.
"""
import sys
import time
from pathlib import Path

import numpy as np

from common import workdir

SITES = {
    # mechanised agriculture, large fields (western Bahia, Brazil)
    "bahia": dict(lon=-45.8, lat=-12.1, year=2023),
    # heterogeneous: smallholder fields, pasture and cerrado remnants near Brasilia (Goias)
    "goias": dict(lon=-47.9, lat=-16.0, year=2023),
}
SIDE = 500           # fine pixels
RES = 30.0
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
LANDSAT_ASSETS = {"green": "green", "red": "red", "nir": "nir08"}
MODIS_BANDS = {"green": 4, "red": 1, "nir": 2}


def item_epsg(item):
    """The EPSG code of a Landsat item (``proj:epsg``, or ``proj:code`` "EPSG:32723" of the newer extension)."""
    p = item.properties
    if p.get("proj:epsg"):
        return int(p["proj:epsg"])
    code = p.get("proj:code") or item.assets["red"].extra_fields.get("proj:code")
    return int(str(code).split(":")[1])


def clear_mask(qa):
    """Landsat QA_PIXEL: fill, dilated cloud, cirrus, cloud and cloud shadow are not clear."""
    bad = (qa & 0b11111) != 0
    return ~bad


def main(site):
    import planetary_computer
    import pystac_client
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    cfg = SITES[site]
    cat = pystac_client.Client.open(STAC, modifier=planetary_computer.sign_inplace)
    d = 0.12
    bbox = [cfg["lon"] - d, cfg["lat"] - d, cfg["lon"] + d, cfg["lat"] + d]
    m_ = 0.085          # the MODIS window: the 15 km area and a margin of a few cells
    mbox = [cfg["lon"] - m_, cfg["lat"] - m_, cfg["lon"] + m_, cfg["lat"] + m_]
    items = cat.search(collections=["landsat-c2-l2"], bbox=bbox, datetime=f"{cfg['year']}-01-01/{cfg['year']}-12-31",
                       query={"eo:cloud_cover": {"lt": 80}, "platform": {"in": ["landsat-8", "landsat-9"]}}).item_collection()
    print(f"{site}: {len(items)} Landsat items")
    epsg = item_epsg(items[0])
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    cx, cy = tr.transform(cfg["lon"], cfg["lat"])
    x0 = np.floor((cx - SIDE * RES / 2) / RES) * RES
    y1 = np.ceil((cy + SIDE * RES / 2) / RES) * RES
    bounds = (x0, y1 - SIDE * RES, x0 + SIDE * RES, y1)
    fine, days = {}, {}
    for it in items:
        if item_epsg(it) != epsg:
            continue
        day = it.datetime.strftime("%Y-%m-%d")
        if day in fine:
            continue
        try:
            def read(key):
                with rasterio.open(it.assets[key].href) as src:
                    w = from_bounds(*bounds, transform=src.transform)
                    return src.read(1, window=w, out_shape=(SIDE, SIDE), boundless=True, fill_value=0, resampling=Resampling.nearest)
            qa = read("qa_pixel")
            bands = [read(a) for a in LANDSAT_ASSETS.values()]
        except Exception as e:                      # a scene that does not cover the window
            print("  skip", day, type(e).__name__, e)
            continue
        ok = clear_mask(qa) & np.all([b > 0 for b in bands], axis=0)
        frac = float(ok.mean())
        if frac < 0.8:
            print(f"  {day} {it.properties['platform']}: {100 * frac:.0f}% clear, skipped")
            continue
        refl = np.stack([b.astype(np.float32) * 2.75e-5 - 0.2 for b in bands])
        refl[:, ~ok] = np.nan
        fine[day] = refl
        print(f"  {day} {it.properties['platform']}: {100 * frac:.0f}% clear")
    days = sorted(fine)
    # MODIS NBAR of those dates
    coarse, c_transform, c_crs = {}, None, None
    for day in days:
        res = cat.search(collections=["modis-43A4-061"], bbox=bbox, datetime=day).item_collection()
        if not len(res):
            print("  no MCD43A4 on", day)
            continue
        it = None
        for cand in res:                    # the tile that holds the whole window (no mosaic is made)
            with rasterio.open(cand.assets["Nadir_Reflectance_Band1"].href) as src:
                t_ = Transformer.from_crs(4326, src.crs, always_xy=True)
                xs, ys = t_.transform([mbox[0], mbox[2]], [mbox[1], mbox[3]])
                b_ = src.bounds
                if min(xs) >= b_.left and max(xs) <= b_.right and min(ys) >= b_.bottom and max(ys) <= b_.top:
                    it = cand
                    break
        if it is None:
            print("  no MCD43A4 tile holds the window on", day)
            continue
        vals = []
        for role, n in MODIS_BANDS.items():
            with rasterio.open(it.assets[f"Nadir_Reflectance_Band{n}"].href) as src:
                t = Transformer.from_crs(4326, src.crs, always_xy=True)
                xs, ys = t.transform([mbox[0], mbox[2], mbox[0], mbox[2]], [mbox[1], mbox[1], mbox[3], mbox[3]])
                w = from_bounds(min(xs), min(ys), max(xs), max(ys), transform=src.transform).round_offsets().round_lengths()
                a = src.read(1, window=w).astype(np.float32)
                wt = src.window_transform(w)
                c_transform, c_crs = tuple(wt)[:6], src.crs.to_wkt()
            with rasterio.open(it.assets[f"BRDF_Albedo_Band_Mandatory_Quality_Band{n}"].href) as src:
                q = src.read(1, window=w)
            a = np.where((a == 32767) | (q > 1), np.nan, a * 1e-4)     # mandatory QA 0 and 1 (full and magnitude inversion)
            vals.append(a)
        coarse[day] = np.stack(vals).astype(np.float32)
    for d_ in days:
        if d_ in coarse:
            print(f"  MODIS {d_}: {100 * np.isfinite(coarse[d_]).mean():.0f}% valid")
    days = [d_ for d_ in days if d_ in coarse and np.isfinite(coarse[d_]).mean() >= 0.5]
    out = workdir("fusion-real") / f"{site}.npz"
    np.savez_compressed(out, days=np.array(days), fine=np.stack([fine[d_] for d_ in days]),
                        coarse=np.stack([coarse[d_] for d_ in days]), fine_bounds=np.array(bounds), fine_epsg=epsg,
                        coarse_transform=np.array(c_transform), coarse_crs=c_crs)
    print(f"{len(days)} dates with both -> {out}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main(sys.argv[1])
    print(f"{time.perf_counter() - t0:.0f} s")
