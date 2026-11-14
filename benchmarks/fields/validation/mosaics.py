"""Landsat annual medoid mosaics of the field-boundary test windows (the input of the 24e sensitivity analysis).

One Landsat Collection 2 Level-2 medoid per window and year, from the Planetary Computer's STAC
(``landsat-c2-l2``, credential-free, windowed COG reads), built with Zeit's own pieces
(``zeit.cube.build_time_series`` to read the raw digital numbers, ``zeit.cube._composite_block`` for the
medoid) and written with ``zeit.save_raster`` as a COG in the native UTM zone at 30 m:

- six reflective bands (blue, green, red, nir08, swir16, swir22) as int16 surface reflectance x 10000, the
  scale the MappingDemo notebook's ``B5 > 2000`` expects. Collection 2's DN to reflectance is
  ``DN * 0.0000275 - 0.2``, so ``value = round((DN * 0.0000275 - 0.2) * 10000)``; NoData -32768. The notebook's
  "B5" is SWIR1: ``swir16`` here (band 5 of the file), B6 of Landsat 8/9;
- clear observations: QA_PIXEL with the fill, dilated cloud, cirrus, cloud and cloud shadow bits all 0 and
  the clear bit set (Zeit's stock Landsat mask is the clear bit alone, which keeps the shadow);
- Landsat 8 and 9 only (the Landsat 7 SLC-off stripes and the Landsat 5 / 7 differences stay out); a window
  may carry ``platforms``, ``start`` and ``end`` instead (the field-boundary tutorial's 2000 mosaic is Landsat
  5 and 7 before the SLC failure, of the dry season; the band names and the QA_PIXEL bits used here are the
  same for Landsat 5, 7, 8 and 9 in Collection 2);
- a second file ``<name>_<year>_clear.tif`` (uint8): the clear observations of each pixel.

Beside each mosaic: ``<name>_<year>.json`` (provenance), ``<name>_<year>_quicklook.png`` (SWIR1 / NIR / red).

    python mosaics.py plan  [--cloud 40]                 # windows, scenes and the bytes estimate; no download
    python mosaics.py build [--only a,b | --group g,h]   # builds what is missing, within --max-gb
    python mosaics.py check                              # reads every mosaic back; writes index.json
    python mosaics.py smoke [name]                       # zeit.fields on one mosaic's SWIR1

Run from PowerShell, with the zeit environment's folders first on PATH (Git's libexpat breaks rasterio).
Data goes to ``ZEIT_FIELDS_DATA/mosaics`` (see ``common.data_dir``).
"""

import pyexpat  # noqa: F401  (on Windows, load the environment's libexpat before rasterio can find Git's)
import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

from common import data_dir, today, write_json

PIXEL = 30
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
QA = "qa_pixel"
PLATFORMS = ["landsat-8", "landsat-9"]
NODATA = -32768
CLOUD = 40                  # scene-level eo:cloud_cover below which a scene is read (pixels are masked anyway)
THREADS = 8                 # dask workers reading the COGs: the machine's cap
COLLECTION = "landsat-c2-l2"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TRAZO_YEARS = {"chaco": 2020, "humidchaco": 2022, "maparaguay": 2024}
HEADER_BYTES = 70_000       # GDAL_INGESTED_BYTES_AT_OPEN (64 KiB) plus a little, per band and scene

GROUPS = ["sorriso", "porto_murtinho", "lem", "chaco", "humidchaco", "maparaguay"]


def mosaics_dir():
    p = data_dir() / "mosaics"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------- the windows

def snap(lo, hi):
    """[lo, hi] widened to Landsat's pixel lattice (edges at 15 mod 30 m), so no resampling is needed."""
    a = math.floor((lo - 15) / PIXEL) * PIXEL + 15
    b = math.ceil((hi - 15) / PIXEL) * PIXEL + 15
    return int(a), int(b)


def utm_epsg(lon, lat):
    zone = min(60, int((lon + 180) // 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def grid_scores(layers, epsg, region, win, step=5000, inside=None):
    """Windows of ``win`` m over ``region`` (UTM bounds) with the number of points of each layer inside."""
    import pandas as pd
    from shapely.geometry import box

    pts = {k: g.to_crs(epsg).representative_point() for k, g in layers.items()}
    minx, miny, maxx, maxy = region
    rows = []
    for x in np.arange(minx, maxx - win + 1, step):
        for y in np.arange(miny, maxy - win + 1, step):
            b = box(x, y, x + win, y + win)
            if inside is not None and inside.intersection(b).area / b.area < 0.9:
                continue
            row = {"x": float(x), "y": float(y)}
            for k, p in pts.items():
                row[k] = int(p.within(b).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def pick(df, score, n, win, allow=None):
    """The ``n`` best windows of a grid that do not overlap each other."""
    chosen = []
    for _, r in df.sort_values(score, ascending=False).iterrows():
        if allow is not None and not allow(r):
            continue
        if all(abs(r.x - c.x) >= win or abs(r.y - c.y) >= win for c in chosen):
            chosen.append(r)
        if len(chosen) == n:
            break
    return chosen


def window(name, group, year, epsg, x, y, w, h, **info):
    x0, x1 = snap(x, x + w)
    y0, y1 = snap(y, y + h)
    return {"name": name, "group": group, "year": year, "epsg": epsg, "bounds": [x0, y0, x1, y1],
            "shape": [(y1 - y0) // PIXEL, (x1 - x0) // PIXEL], **info}


def windows():
    """Every window (name, group, year, UTM bounds) and why it was chosen. Deterministic from the data."""
    import geopandas as gpd

    d = data_dir()
    mun = gpd.read_file(d / "municipios.gpkg").set_index("key")
    out = []

    # Sorriso (MT): 20 km windows inside the municipality; CAR properties + 10 x pivots (the pivots are the rarer
    # and the clearer fields).
    car = gpd.read_file(d / "car_sorriso.gpkg")
    piv = gpd.read_file(d / "pivots_2019.gpkg")
    piv = piv[piv.key == "sorriso"]
    epsg = 32721
    m = gpd.GeoSeries([mun.loc["sorriso"].geometry], crs=4326).to_crs(epsg).iloc[0]
    df = grid_scores({"car": car, "pivots": piv}, epsg, m.bounds, 20000, inside=m)
    df["score"] = df.car + 10 * df.pivots
    for k, r in enumerate(pick(df, "score", 2, 20000), 1):
        out.append(window(f"sorriso_{k}", "sorriso", 2019, epsg, r.x, r.y, 20000, 20000,
                          why=f"CAR properties {int(r.car)} + pivots 2019 {int(r.pivots)} (score {int(r.score)}), "
                              f"the {k}{'st' if k == 1 else 'nd'} best non-overlapping 20 km window inside the municipality",
                          reference="CAR (car_sorriso), pivots 2019"))

    # Porto Murtinho (MS): the CAR properties, without the river's floodplain (the Pantanal): no Pantanal layer is
    # at hand, so the windows must be inland (centre east of 57.5 W), which is also where the properties are
    # pasture farms; checked on the quick looks.
    car = gpd.read_file(d / "car_porto_murtinho.gpkg")
    m = gpd.GeoSeries([mun.loc["porto_murtinho"].geometry], crs=4326).to_crs(epsg).iloc[0]
    df = grid_scores({"car": car}, epsg, m.bounds, 20000, inside=m)
    from pyproj import Transformer
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    df["lon"] = [to_ll.transform(x + 10000, y + 10000)[0] for x, y in zip(df.x, df.y)]
    for k, r in enumerate(pick(df, "car", 2, 20000, allow=lambda r: r.lon > -57.5), 1):
        out.append(window(f"porto_murtinho_{k}", "porto_murtinho", 2019, epsg, r.x, r.y, 20000, 20000,
                          why=f"CAR properties {int(r.car)}; centre {r.lon:.2f} E of the river (inland, not the Pantanal)",
                          reference="CAR (car_porto_murtinho)"))

    # Luis Eduardo Magalhaes (BA): the densest 40 km windows of the FTW fields (the FTW extent is 250 km tall,
    # far beyond the municipality).
    ftw = gpd.read_file(d / "ftw_brazil_2020.gpkg")
    epsg = 32723
    b = ftw.to_crs(epsg).total_bounds
    df = grid_scores({"ftw": ftw}, epsg, b, 40000)
    for k, r in enumerate(pick(df, "ftw", 2, 40000), 1):
        out.append(window(f"lem_{k}", "lem", 2020, epsg, r.x, r.y, 40000, 40000,
                          why=f"FTW fields {int(r.ftw)} of {len(ftw)}; the {k}{'st' if k == 1 else 'nd'} densest "
                              f"non-overlapping 40 km window",
                          reference="Fields of The World Brazil 2020 (ftw_brazil_2020)"))

    # Trazo: each cluster of chip footprints padded by 1 km, in its own UTM zone, the year of its folder.
    areas = gpd.read_file(d / "areas.gpkg")
    for r in areas[areas.kind == "trazo_cluster"].itertuples():
        folder = r.source.split()[1]
        c = r.geometry.representative_point()
        epsg = utm_epsg(c.x, c.y)
        g = gpd.GeoSeries([r.geometry], crs=4326).to_crs(epsg)
        x0, y0, x1, y1 = g.total_bounds
        out.append(window(r.area, folder, TRAZO_YEARS[folder], epsg, x0 - 1000, y0 - 1000, x1 - x0 + 2000,
                          y1 - y0 + 2000, why=f"{r.n_chips} chip footprint(s) of {folder}, padded by 1 km",
                          reference=f"Trazo {folder}"))
    return out


# ---------------------------------------------------------------------------------------- the STAC search

def lonlat_bbox(w):
    from pyproj import Transformer

    t = Transformer.from_crs(w["epsg"], 4326, always_xy=True)
    x0, y0, x1, y1 = w["bounds"]
    pts = [t.transform(x, y) for x in (x0, x1) for y in (y0, y1)]
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def period(w):
    """(start, end) dates of a window's mosaic: the calendar year, or ``w["start"]`` / ``w["end"]`` (the
    tutorial's dry season)."""
    return w.get("start", f"{w['year']}-01-01"), w.get("end", f"{w['year']}-12-31")


def search(w, cloud):
    """The Landsat items of the window's period (Landsat 8/9, or ``w["platforms"]``) (unsigned)."""
    import pystac_client

    start, end = period(w)
    cat = pystac_client.Client.open(STAC)
    s = cat.search(collections=[COLLECTION], bbox=lonlat_bbox(w), datetime=f"{start}/{end}",
                   query={"eo:cloud_cover": {"lt": cloud}, "platform": {"in": w.get("platforms", PLATFORMS)}})
    return list(s.items())


def calibrate(items):
    """Mean compressed bytes of a populated 256 x 256 tile of each asset, from the TIFF headers of a few items."""
    import fsspec
    import planetary_computer
    import tifffile

    per = {a: [] for a in BANDS + [QA]}
    pick_ = [items[i] for i in np.linspace(0, len(items) - 1, 3).astype(int)]
    for it in pick_:
        it = planetary_computer.sign(it)
        for a in per:
            with fsspec.open(it.assets[a].href, "rb") as f:
                page = tifffile.TiffFile(f).pages[0]
                counts = np.asarray(page.databytecounts)
                per[a].append(float(counts[counts > 0].mean()))
    return {a: float(np.mean(v)) for a, v in per.items()}


def estimate_bytes(w, n_scenes, tile_bytes):
    """Bytes to read ``n_scenes`` scenes of the window: the tiles it touches in each band, plus the headers.
    Tiles: (w / 256 + 1) x (h / 256 + 1) on average for a window at any offset (a scene covering the window only
    in part reads fewer, so it errs high)."""
    h, wd = w["shape"]
    tiles = (wd / 256 + 1) * (h / 256 + 1)
    per_scene = sum(tiles * tile_bytes[a] + HEADER_BYTES for a in tile_bytes)
    return n_scenes * per_scene


def net_bytes():
    """Bytes received by the machine's network adapters (``Get-NetAdapterStatistics``): everything on the
    network, so an upper bound of ours (a 1 MB range request counts 1.05 MB; ``netstat -e`` counts 6x that, as
    it adds the virtual interfaces)."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-NetAdapterStatistics | Measure-Object ReceivedBytes -Sum).Sum"],
                             capture_output=True, text=True, timeout=60).stdout
        return int(out.strip())
    except Exception:
        return None


# ------------------------------------------------------------------------------------------- the build

@contextlib.contextmanager
def zeit_shims(bounds, shadow=True, platforms=None):
    """Three things ``zeit.cube`` cannot be told (reported, not fixed): the platform of the scenes, the exact
    output grid (``snap_bounds`` and ``bounds`` of ``stackstac.stack``, so the mosaic sits on Landsat's own
    lattice and has exactly the requested size) and a Landsat mask that includes the cloud shadow. The mask
    bits are the same for Landsat 5, 7, 8 and 9 (bit 2, cirrus, is only set by 8 and 9)."""
    import stackstac
    import zeit.cube as zc

    platforms = platforms or PLATFORMS
    real_open = zc.Client.open
    real_stack = stackstac.stack

    def open_(*a, **k):
        cat = real_open(*a, **k)
        real_search = cat.search

        def search_(**kw):
            q = dict(kw.get("query") or {})
            q["platform"] = {"in": platforms}
            kw["query"] = q
            return real_search(**kw)

        cat.search = search_
        return cat

    def stack(items, **kw):
        kw.pop("bounds_latlon", None)
        kw.update(bounds=tuple(bounds), snap_bounds=False)
        return real_stack(items, **kw)

    def clear(qa, qa_band):
        qa = np.asarray(qa)
        if qa_band.lower() != QA:
            return real_clear(qa, qa_band)
        ok = (qa & (1 << 6)) > 0
        if shadow:   # fill 0, dilated cloud 1, cirrus 2, cloud 3, cloud shadow 4
            ok &= (qa & 0b11111) == 0
        return ok

    real_clear = zc._clear_mask
    with mock.patch.object(zc.Client, "open", staticmethod(open_)), \
            mock.patch.object(stackstac, "stack", stack), \
            mock.patch.object(zc, "_clear_mask", clear):
        yield zc


def quicklook(refl, path):
    """SWIR1 / NIR / red as RGB, 2-98 percent stretch, at most ~900 px on a side."""
    rgb = np.stack([refl[4], refl[3], refl[2]]).astype("float32")
    step = max(1, int(math.ceil(max(rgb.shape[1:]) / 900)))
    rgb = rgb[:, ::step, ::step]
    out = np.zeros(rgb.shape, dtype="uint8")
    for k in range(3):
        v = rgb[k]
        ok = np.isfinite(v)
        if ok.any():
            lo, hi = np.percentile(v[ok], [2, 98])
            out[k] = np.where(ok, np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255, 0).astype("uint8")
    img = np.moveaxis(out, 0, -1)
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except ImportError:
        import matplotlib.pyplot as plt

        plt.imsave(path, img)


def build(w, cloud, tile_bytes=None, out=None):
    """Build one mosaic into ``out`` (default ``mosaics_dir()``). Returns its provenance dict. A window may
    carry ``platforms`` (default Landsat 8/9) and ``start`` / ``end`` dates (default the calendar year)."""
    import dask
    import xarray as xr
    from affine import Affine

    import zeit
    from zeit._save import save_raster
    import zeit.cube as zc

    t0 = time.time()
    name, year, epsg = w["name"], w["year"], w["epsg"]
    x0, y0, x1, y1 = w["bounds"]
    out = Path(out) if out is not None else mosaics_dir()
    out.mkdir(parents=True, exist_ok=True)
    platforms = w.get("platforms", PLATFORMS)
    start, end = period(w)
    net0 = net_bytes()
    with zeit_shims(w["bounds"], platforms=platforms) as zc:
        cube = zc.build_time_series(
            source="planetary_computer", collection=COLLECTION, bbox=lonlat_bbox(w),
            start_date=start, end_date=end, cloud_cover_max=cloud,
            bands=BANDS + [QA], apply_cloud_mask=False, resolution=PIXEL, epsg=epsg, chunksize=2048,
            dtype="uint16")
        t = cube.sizes["time"]
        assert cube.sizes["y"] == w["shape"][0] and cube.sizes["x"] == w["shape"][1], (cube.shape, w["shape"])
        names = list(cube.band.values)
        with dask.config.set(scheduler="threads", num_workers=THREADS):
            raw = np.asarray(cube.data.compute())
        t_read = time.time() - t0
        band_idx = [names.index(b) for b in BANDS]
        qa_idx = names.index(QA)
        scale, offset = zc._cube_scale_offset(cube, BANDS)
        refl = zc._composite_block(raw, band_idx, qa_idx, QA, scale, offset, "medoid")
        clear = zc._clear_mask(raw[:, qa_idx], QA) & (raw[:, band_idx] != 0).all(axis=1)
    count = clear.sum(axis=0).astype("uint8")
    scene_clear = clear.mean(axis=(1, 2))
    net1 = net_bytes()

    valid = np.isfinite(refl).all(axis=0)
    data = np.where(np.isfinite(refl), np.rint(refl * 10000.0), NODATA).astype("int16")
    transform = Affine(PIXEL, 0, x0, 0, -PIXEL, y1)
    da = xr.DataArray(data, dims=("band", "y", "x"), coords={"band": BANDS})
    save_raster(da, out / f"{name}_{year}.tif", crs=f"EPSG:{epsg}", transform=transform, nodata=NODATA,
                band_names=BANDS, driver="COG")
    save_raster(xr.DataArray(count, dims=("y", "x")), out / f"{name}_{year}_clear.tif", crs=f"EPSG:{epsg}",
                transform=transform, nodata=0, driver="COG")
    quicklook(refl, out / f"{name}_{year}_quicklook.png")

    pctl = [0, 5, 25, 50, 75, 95, 100]
    ids = [str(v) for v in cube.coords["id"].values]
    times = [str(v)[:19] for v in cube.time.values]

    def coord(k):
        if k not in cube.coords:
            return [None] * t
        vals = np.broadcast_to(np.asarray(cube.coords[k].values, dtype=object), (t,))
        return [None if v is None else (v.item() if hasattr(v, "item") else v) for v in vals]
    sc = [{"id": i, "datetime": d, "platform": p, "wrs_path": a, "wrs_row": b, "cloud_cover": c,
           "clear_fraction_in_window": round(float(f), 4)}
          for i, d, p, a, b, c, f in zip(ids, times, coord("platform"), coord("landsat:wrs_path"),
                                         coord("landsat:wrs_row"), coord("eo:cloud_cover"), scene_clear)]
    est = None
    if tile_bytes:
        est = estimate_bytes(w, t, tile_bytes)
    prov = {
        "name": name, "group": w["group"], "year": year, "built": today(), "why": w.get("why"),
        "reference": w.get("reference"),
        "source": {"stac": STAC, "collection": COLLECTION, "signed": "planetary_computer",
                   "platforms": platforms, "datetime": f"{start}/{end}",
                   "scene_cloud_cover_lt": cloud, "windowed_cog_reads": True},
        "grid": {"crs": f"EPSG:{epsg}", "pixel_m": PIXEL, "bounds": [x0, y0, x1, y1], "shape": list(data.shape),
                 "lattice": "edges at 15 mod 30 m, Landsat's own (no resampling)"},
        "bands": {"order": BANDS, "note": "the notebook's B5 is swir16 (band 5 of the file)"},
        "values": {"dtype": "int16", "nodata": NODATA,
                   "conversion": "value = round((DN * 0.0000275 - 0.2) * 10000) (scale and offset from each item's "
                                 "raster:bands); surface reflectance x 10000",
                   "medoid": "the clear observation closest, over the six bands, to the per-band median of the "
                             "clear observations (zeit.cube._composite_block, method medoid)"},
        "mask": "QA_PIXEL: fill, dilated cloud, cirrus, cloud and cloud shadow bits 0 and the clear bit 1; "
                "a zero in any band is no observation",
        "scenes": {"n": t, "list": sc},
        "clear_observations": {"percentiles": dict(zip(map(str, pctl), map(int, np.percentile(count, pctl)))),
                               "pixels_with_none_pct": round(float((count == 0).mean() * 100), 3)},
        "valid_pct": round(float(valid.mean() * 100), 3),
        "median_clear_observations": float(np.median(count)),
        "seconds": {"read": round(t_read, 1), "total": round(time.time() - t0, 1)},
        "bytes": {"estimated": None if est is None else int(est),
                  "network_received": None if net0 is None or net1 is None else net1 - net0,
                  "note": "the machine's adapters' received bytes during the build (headers and TLS included): "
                          "an upper bound of ours"},
        "zeit": {"version": getattr(zeit, "__version__", None), "git": git_head()},
    }
    write_json(out / f"{name}_{year}.json", prov)
    return prov


def git_head():
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------- commands

def selected(args):
    ws = windows()
    if args.only:
        keep = set(args.only.split(","))
        ws = [w for w in ws if w["name"] in keep]
    if args.group:
        keep = set(args.group.split(","))
        ws = [w for w in ws if w["group"] in keep]
    return ws


def plan(args):
    ws = selected(args)
    write_json(mosaics_dir() / "windows.json", windows())
    tile_bytes = None
    rows, total = [], 0
    for w in ws:
        items = search(w, args.cloud)
        if not items:
            print(f"{w['name']}: no scenes")
            continue
        if tile_bytes is None:
            tile_bytes = calibrate(items)
            print("bytes per populated 256 x 256 tile:", {k: round(v / 1e3) for k, v in tile_bytes.items()}, "KB")
        n = len(items)
        est = estimate_bytes(w, n, tile_bytes)
        total += est
        rows.append({"name": w["name"], "group": w["group"], "year": w["year"], "shape": w["shape"], "scenes": n,
                     "bytes": int(est)})
        print(f"{w['name']:28s} {w['year']} {w['shape'][1]:5d}x{w['shape'][0]:<5d} scenes {n:3d} "
              f"est {est / 1e6:8.1f} MB   {w['why']}")
    by = {}
    for r in rows:
        by[r["group"]] = by.get(r["group"], 0) + r["bytes"]
    print("\nestimate by group (GB):", {k: round(v / 1e9, 2) for k, v in by.items()})
    print(f"estimate total: {total / 1e9:.2f} GB")
    plan_file = mosaics_dir() / "plan.json"
    old = json.loads(plan_file.read_text()) if plan_file.exists() else {"rows": {}}
    merged = old["rows"]
    for r in rows:   # a later plan of the same window (another --cloud) replaces the earlier one
        merged[r["name"]] = {**r, "cloud": args.cloud}
    write_json(plan_file, {"tile_bytes": tile_bytes or old.get("tile_bytes"), "rows": merged})


def build_cmd(args):
    ws = selected(args)
    plan_file = mosaics_dir() / "plan.json"
    plan_json = json.loads(plan_file.read_text()) if plan_file.exists() else {"tile_bytes": None, "rows": {}}
    tile_bytes = plan_json["tile_bytes"]
    # the estimate is low where the tiles are dense (1.1 to 1.7 x measured): the cap uses it x 1.6
    planned = {k: r["bytes"] * 1.6 for k, r in plan_json["rows"].items() if r.get("cloud") == args.cloud}
    spent = 0
    for w in ws:
        if (mosaics_dir() / f"{w['name']}_{w['year']}.json").exists():
            print(f"[{w['name']}] exists")
            continue
        # the cap is on what the whole project transferred: the bytes of the mosaics already built count too
        done_gb = args.extra_gb + sum((json.loads(p.read_text()).get("bytes") or {}).get("network_received") or 0
                                      for p in mosaics_dir().glob("*_20??.json")) / 1e9
        if w["name"] not in planned:
            print(f"[{w['name']}] skipped: no estimate for --cloud {args.cloud} (run plan first)")
            continue
        if done_gb + planned[w["name"]] / 1e9 > args.max_gb:
            print(f"[{w['name']}] skipped: {done_gb:.2f} GB transferred + its estimate "
                  f"{planned[w['name']] / 1e9:.2f} GB (x 1.6) would pass the {args.max_gb} GB cap")
            continue
        try:
            prov = build(w, args.cloud, tile_bytes)
        except Exception as e:
            print(f"[{w['name']}] FAILED {type(e).__name__}: {e}")
            continue
        b = prov["bytes"]["network_received"] or prov["bytes"]["estimated"] or 0
        spent += b
        print(f"[{w['name']}] {prov['grid']['shape']} scenes {prov['scenes']['n']} valid {prov['valid_pct']}% "
              f"median clear {prov['median_clear_observations']} {prov['seconds']['total']} s, "
              f"bytes {b / 1e6:.0f} MB (estimate {(prov['bytes']['estimated'] or 0) / 1e6:.0f} MB); "
              f"total {spent / 1e9:.2f} GB")


def check(args):
    """Read every mosaic back: shape, CRS, COG layout, % valid, clear observations; write ``index.json``."""
    import rasterio

    rows = []
    for jp in sorted(mosaics_dir().glob("*_20??.json")):
        prov = json.loads(jp.read_text())
        if "grid" not in prov:
            continue
        tif = jp.with_suffix(".tif")
        with rasterio.open(tif) as src:
            data = src.read(1)
            valid = float((data != src.nodata).mean() * 100)
            # the layout of a COG: tiled, with the IFDs in front (overviews are optional for these sizes)
            layout = src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT")
            row = {"name": prov["name"], "group": prov["group"], "year": prov["year"], "count": src.count,
                   "shape": [src.height, src.width], "crs": str(src.crs), "dtype": src.dtypes[0],
                   "nodata": src.nodata, "res": list(src.res), "layout": layout, "tiled": src.profile.get("tiled"),
                   "valid_pct": round(valid, 3), "scenes": prov["scenes"]["n"],
                   "clear_p50": prov["clear_observations"]["percentiles"]["50"],
                   "clear_p5": prov["clear_observations"]["percentiles"]["5"],
                   "clear_p95": prov["clear_observations"]["percentiles"]["95"],
                   "swir16_median": float(np.median(src.read(5)[src.read(5) != src.nodata])),
                   "cloud_lt": prov["source"]["scene_cloud_cover_lt"], "mb": round(prov["bytes"]["network_received"] / 1e6, 1)
                   if prov["bytes"]["network_received"] else None}
        rows.append(row)
        print(f"{row['name']:24s} {row['year']} {row['count']}x{row['shape'][0]}x{row['shape'][1]} {row['crs']} "
              f"{row['layout']} valid {row['valid_pct']:6.2f}% scenes {row['scenes']:3d} clear p5/p50/p95 "
              f"{row['clear_p5']}/{row['clear_p50']}/{row['clear_p95']} swir16 median {row['swir16_median']:.0f} "
              f"{row['mb']} MB")
    total = sum(r["mb"] or 0 for r in rows) / 1e3
    print(f"{len(rows)} mosaics; received {total:.2f} GB while building them")
    write_json(mosaics_dir() / "index.json", {"built": today(), "mosaics": rows, "received_gb": round(total, 3)})


def smoke(args):
    import rasterio

    import zeit

    name = args.name or "sorriso_1_2019"
    path = mosaics_dir() / f"{name}.tif"
    with rasterio.open(path) as src:
        swir1 = src.read(BANDS.index("swir16") + 1).astype("float32")
        nodata = src.nodata
        transform, crs = src.transform, src.crs
    swir1[swir1 == nodata] = np.nan
    import xarray as xr

    da = xr.DataArray(swir1, dims=("y", "x"),
                      coords={"y": transform.f + transform.e * (np.arange(swir1.shape[0]) + 0.5),
                              "x": transform.c + transform.a * (np.arange(swir1.shape[1]) + 0.5)})
    da = da.rio.write_crs(crs)
    t0 = time.time()
    res = zeit.fields(da, preset="notebook", realizations=50, random_state=0, n_jobs=8)
    dt = time.time() - t0
    labels = res["labels"].values
    n = len(np.unique(labels[labels > 0]))
    print(f"zeit.fields {name} {swir1.shape}: {dt:.1f} s, {n} fields, "
          f"{(labels > 0).mean() * 100:.1f}% of pixels labelled")
    write_json(mosaics_dir() / f"smoke_{name}.json", {"shape": list(swir1.shape), "seconds": round(dt, 2),
                                                      "fields": n, "realizations": 50, "random_state": 0,
                                                      "n_jobs": 8, "preset": "notebook"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "build", "check", "smoke"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--cloud", type=int, default=CLOUD)
    ap.add_argument("--only")
    ap.add_argument("--group")
    ap.add_argument("--max-gb", type=float, default=6.0)
    ap.add_argument("--extra-gb", type=float, default=0.0,
                    help="bytes transferred outside the written mosaics (failed runs, tests), counted in the cap")
    args = ap.parse_args()
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    {"plan": plan, "build": build_cmd, "check": check, "smoke": smoke}[args.cmd](args)


if __name__ == "__main__":
    main()
