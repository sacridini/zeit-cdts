"""Trazo annotations (WRI / ASU), the Chaco reference. CC-BY-4.0.

Grupp et al. (2026), Field Boundaries of South America. The README of the dataset
states that the annotations completely cover all fields within their chips.
Fetches, per folder: the annotation GeoPackage, "annotation info.txt", the chips
GeoParquet and the header of each chip's window_b GeoTIFF (for the chip extent).
Not the imagery or the label masks.
"""

import io
import time
import urllib.parse

import geopandas as gpd
import pandas as pd
import shapely
import tifffile

from common import Counter, data_dir, done, get, session, today, write_json

BASE = "https://data.source.coop/wri-data-lab/trazoannotations"
FOLDERS = {
    "chaco": "chacoAnnotations_2020.gpkg",
    "humidchaco": "humidchacoAnnotations_2022.gpkg",
    "maparaguay": "maparaguayAnnotations_2024.gpkg",
}


def fetch(s, url, dest):
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(get(url, s).content)
    print(f"  fetched {dest.name} ({dest.stat().st_size:,} bytes)")
    return True


def footprints(s, folder, aoi_ids):
    """Chip footprints (EPSG:4326) from the GeoTIFF headers of window_b (range reads).

    The chips parquet holds the field polygons per chip, not the chip extent; the
    extent of the 256 x 256 chip is the extent of its Sentinel-2 image.
    """
    rows = []
    for aoi in aoi_ids:
        url = f"{BASE}/{folder}/s2_images/window_b/{aoi}.tif"
        for k in range(5):
            try:
                r = s.get(url, headers={"Range": "bytes=0-16383"}, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(2 ** (k + 1))
        Counter.bytes += len(r.content)
        with tifffile.TiffFile(io.BytesIO(r.content)) as t:
            pg = t.pages[0]
            h, w = pg.shape[:2]
            sx, sy, _ = pg.tags["ModelPixelScaleTag"].value
            _, _, _, x0, y0, _ = pg.tags["ModelTiepointTag"].value
            gk = pg.geotiff_tags
            epsg = int(gk.get("ProjectedCSTypeGeoKey") or gk.get("GeographicTypeGeoKey"))
        rows.append({"aoi_id": aoi, "epsg": epsg,
                     "geometry": shapely.box(x0, y0 - h * sy, x0 + w * sx, y0)})
    out = []
    for epsg, g in gpd.GeoDataFrame(rows, geometry="geometry").groupby("epsg"):
        out.append(gpd.GeoDataFrame(g, geometry="geometry").set_crs(f"EPSG:{epsg}").to_crs(4326))
    return pd.concat(out, ignore_index=True).drop(columns="epsg")


def main():
    d = data_dir() / "trazo"
    s = session()
    prov = {}
    for folder, gpkg in FOLDERS.items():
        raw = d / "_raw" / folder
        fetch(s, f"{BASE}/{folder}/{gpkg}", raw / gpkg)
        fetch(s, f"{BASE}/{folder}/{urllib.parse.quote('annotation info.txt')}",
              raw / "annotation_info.txt")
        fetch(s, f"{BASE}/{folder}/chips_{folder}.parquet", raw / f"chips_{folder}.parquet")
        out = d / f"trazo_{folder}.gpkg"
        info = (raw / "annotation_info.txt").read_text(encoding="utf-8", errors="replace")
        ann = gpd.read_file(raw / gpkg)
        chips = gpd.read_parquet(raw / f"chips_{folder}.parquet")
        crs0 = str(ann.crs)
        if not out.exists():
            ann.to_crs(4326).to_file(out, driver="GPKG", layer="annotations")
            chips.to_crs(4326).to_file(out, driver="GPKG", layer="chips")
            fp = footprints(s, folder, list(chips["aoi_id"]))
            fp = fp.merge(chips.drop(columns="geometry"), on="aoi_id")
            fp.to_file(out, driver="GPKG", layer="chip_footprints")
        fp = gpd.read_file(out, layer="chip_footprints")
        a4 = ann.to_crs(4326)
        prov[folder] = {
            "gpkg": gpkg, "annotations": len(ann), "chips": len(chips), "crs": crs0,
            "columns": [c for c in ann.columns if c != "geometry"],
            "chip_columns": [c for c in chips.columns if c != "geometry"],
            "chips_geometry": "field polygons of the chip (not its extent)",
            "footprints": "256 x 256 px extent of the window_b GeoTIFF (EPSG:4326)",
            "chip_footprints": len(fp), "bounds": [float(v) for v in a4.total_bounds], "info": info.strip()}
        for c in ann.columns:
            if "year" in c.lower():
                prov[folder]["years"] = {str(k): int(v) for k, v in
                                         ann[c].value_counts(dropna=False).items()}
        print(f"  {folder}: {len(ann)} annotations, {len(chips)} chips, CRS {crs0}, "
              f"bounds {a4.total_bounds}\n    info: {info.strip()!r}")
    write_json(d / "trazo.json", {
        "base": BASE, "fetched": today(), "licence": "CC-BY-4.0",
        "cite": "Grupp et al. (2026), Field Boundaries of South America, WRI & ASU",
        "complete_within_chips": "README: all field boundary annotations completely "
                                 "cover all fields within their chips",
        "folders": prov})
    done("trazo")


if __name__ == "__main__":
    main()
