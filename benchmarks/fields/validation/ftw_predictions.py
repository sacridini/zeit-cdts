"""Fields of The World global predicted field polygons (PRUE) for the test windows.

Source: https://source.coop/ftw/global-data/predictions/vectors, one GeoParquet per
country (Brazil: per state), spatially sorted in row groups of 65,536 rows with `bbox`
statistics. Only the row groups whose bbox statistics meet a window are read (HTTP range
requests), never a whole file; the script aborts when that would exceed MAX_BYTES.

Windows: the built mosaics (mosaics/index.json) with their bounds from mosaics/windows.json.
Polygons are kept whole where they intersect a window (a polygon on two windows appears twice);
columns: window, group, year (of determination:datetime: 2024 or 2025), the confidence (0-100,
null outside the modeled-confidence layer; the page recommends >= 69) and the rest of the source.

    python ftw_predictions.py plan     # row groups and bytes, nothing downloaded
    python ftw_predictions.py fetch
"""

import json
import sys

import fsspec
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyproj
import shapely
from shapely.geometry import box
from shapely.ops import transform

from common import USER_AGENT, Counter, data_dir, done, today, write_json

BASE = ("https://data.source.coop/ftw/global-data/predictions/vectors/alpha/"
        "results-by-admin-conf/admin:country_code=")
CANDIDATES = [f"BR/BR_{s}.parquet" for s in ("MT", "MS", "BA", "GO", "TO", "PI", "MG", "MA", "PR", "SP")] + [
    "PY/Paraguay.parquet", "AR/Argentina.parquet", "BO/Bolivia.parquet"]
MAX_BYTES = 1.5e9
OUT = "ftw_predictions.gpkg"


def windows(d):
    """Built windows: name, group, year, polygon in EPSG:4326 (densified UTM rectangle)."""
    built = {m["name"] for m in json.loads((d / "mosaics" / "index.json").read_text())["mosaics"]}
    out = []
    for w in json.loads((d / "mosaics" / "windows.json").read_text()):
        if w["name"] not in built:
            continue
        t = pyproj.Transformer.from_crs(w["epsg"], 4326, always_xy=True).transform
        x0, y0, x1, y1 = w["bounds"]
        out.append({**w, "poly": transform(t, shapely.segmentize(box(x0, y0, x1, y1), 200.0))})
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "plan"
    d = data_dir()
    out = d / OUT
    if mode == "fetch" and out.exists():
        print(f"[ftw_predictions] exists: {out}")
        return
    wins = windows(d)
    print(f"{len(wins)} windows")
    fs = fsspec.filesystem("https", headers={"User-Agent": USER_AGENT})
    files, plan, total = {}, {}, 0
    for c in CANDIDATES:
        f = fs.open(BASE + c, block_size=1 << 20)
        pf = pq.ParquetFile(f)
        md = pf.metadata
        ix = {md.row_group(0).column(i).path_in_schema: i for i in range(md.num_columns)}
        groups, comp = set(), 0
        for g in range(md.num_row_groups):
            r = md.row_group(g)

            def st(n):
                return r.column(ix[n]).statistics

            bx = (st("bbox.xmin").min, st("bbox.ymin").min, st("bbox.xmax").max, st("bbox.ymax").max)
            if any(bx[0] <= w["poly"].bounds[2] and bx[2] >= w["poly"].bounds[0]
                   and bx[1] <= w["poly"].bounds[3] and bx[3] >= w["poly"].bounds[1] for w in wins):
                groups.add(g)
                comp += sum(r.column(i).total_compressed_size for i in range(md.num_columns))
        if groups:
            files[c] = (f, pf)
            plan[c] = {"row_groups": sorted(groups), "of": md.num_row_groups, "rows_total": md.num_rows,
                       "compressed_mb": round(comp / 1e6, 1)}
            total += comp
        print(f"  {c}: {len(groups)} of {md.num_row_groups} row groups, ~{comp / 1e6:.0f} MB")
    print(f"  total ~{total / 1e6:.0f} MB (cap {MAX_BYTES / 1e6:.0f} MB)")
    if total > MAX_BYTES:
        raise SystemExit("would exceed the cap")
    if mode != "fetch":
        return

    frames = []
    for c, (f, pf) in files.items():
        t = pf.read_row_groups(plan[c]["row_groups"])
        bb = t["bbox"].combine_chunks()
        xmin, ymin, xmax, ymax = (np.asarray(bb.field(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        year = np.asarray(t["determination:datetime"].to_pandas().dt.year)
        geom = None
        for w in wins:
            b = w["poly"].bounds
            m = (xmin <= b[2]) & (xmax >= b[0]) & (ymin <= b[3]) & (ymax >= b[1])
            idx = np.nonzero(m)[0]
            if not len(idx):
                continue
            if geom is None:
                geom = gpd.GeoSeries.from_wkb(t["geometry"].to_pandas(), crs="EPSG:4326")
            sub = geom.iloc[idx]
            keep = idx[np.asarray(sub.intersects(w["poly"]))]
            if not len(keep):
                continue
            df = t.take(pa.array(keep)).drop_columns(["geometry", "bbox"]).to_pandas()
            df["year"] = year[keep]
            df["window"], df["group"], df["source_file"] = w["name"], w["group"], c
            frames.append(gpd.GeoDataFrame(df, geometry=geom.iloc[keep].values, crs="EPSG:4326"))
        Counter.bytes += int(getattr(f.cache, "total_requested_bytes", 0))
        print(f"  {c}: read, {sum(len(x) for x in frames)} polygons so far")
    gdf = gpd.GeoDataFrame(__import__("pandas").concat(frames, ignore_index=True), crs="EPSG:4326")
    gdf["determination:datetime"] = gdf["determination:datetime"].dt.tz_localize(None)
    gdf.to_file(out, driver="GPKG", layer="ftw_predictions")
    summ = {}
    for g, s in gdf.groupby("group"):
        conf = s["confidence"]
        summ[g] = {"polygons": len(s), "by_year": {int(k): int(v) for k, v in s["year"].value_counts().items()},
                   "confidence_null": int(conf.isna().sum()),
                   "confidence_ge_69": int((conf >= 69).sum()),
                   "confidence_quantiles": {str(q): float(conf.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
                   if conf.notna().any() else None,
                   "windows": {k: int(v) for k, v in s["window"].value_counts().sort_index().items()}}
    write_json(d / "ftw_predictions.json", {
        "url": BASE, "fetched": today(), "licence": "CC-BY-4.0",
        "cite": ["Kerner et al. (2025), AAAI-2025, arXiv:2409.16252",
                 "Muhawenayo et al. (2026), PRUE, arXiv:2603.27101"],
        "filter": "polygons intersecting each window; confidence not filtered (recommended >= 69)",
        "files": plan, "bytes_requested": Counter.bytes, "groups": summ})
    for g, v in summ.items():
        print(g, v["polygons"], v["by_year"], "null", v["confidence_null"], ">=69", v["confidence_ge_69"],
              v["confidence_quantiles"])
    done("ftw_predictions")


if __name__ == "__main__":
    main()
