"""The test-area list as one GeoPackage (EPSG:4326), areas.gpkg.

- the three municipalities (IBGE boundaries): Porto Murtinho, Sorriso, LEM;
- the Chaco: one area per cluster of Trazo chip footprints (footprints that touch or
  overlap are one cluster), the cluster being the union of the footprints.
Columns: area, kind, source, n_chips, km2 (equal-area, EPSG:6933).
"""

import geopandas as gpd
import pandas as pd
import shapely

from common import data_dir, done, today, write_json

FOLDERS = ["chaco", "humidchaco", "maparaguay"]


def clusters(fp):
    """Connected components of overlapping or touching footprints."""
    n = len(fp)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree(fp.geometry.values)
    for i, g in enumerate(fp.geometry.values):
        for j in tree.query(g, predicate="intersects"):
            parent[find(int(j))] = find(i)
    return [find(i) for i in range(n)]


def main():
    d = data_dir()
    out = d / "areas.gpkg"
    if out.exists():
        print(f"[areas] exists: {out}")
        return
    rows = []
    muni = gpd.read_file(d / "municipios.gpkg")
    for _, r in muni.iterrows():
        rows.append({"area": r["key"], "kind": "municipality", "source": "IBGE",
                     "n_chips": 0, "geometry": r.geometry})
    for folder in FOLDERS:
        fp = gpd.read_file(d / "trazo" / f"trazo_{folder}.gpkg", layer="chip_footprints")
        fp["cl"] = clusters(fp)
        for k, (cl, g) in enumerate(sorted(fp.groupby("cl"), key=lambda t: t[1].total_bounds[0])):
            rows.append({"area": f"trazo_{folder}_{k:03d}", "kind": "trazo_cluster",
                         "source": f"Trazo {folder}", "n_chips": len(g),
                         "geometry": shapely.union_all(g.geometry.values)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["km2"] = gdf.to_crs(6933).area / 1e6
    gdf.to_file(out, driver="GPKG")
    tz = gdf[gdf["kind"] == "trazo_cluster"]
    summary = {f: {"clusters": int((tz["source"] == f"Trazo {f}").sum()),
                   "km2": [round(v, 1) for v in sorted(tz[tz["source"] == f"Trazo {f}"].km2)]}
               for f in FOLDERS}
    write_json(d / "areas.json", {"fetched_or_built": today(), "rows": len(gdf),
                                  "municipalities_km2": {r.area: round(r.km2) for r in gdf.itertuples()
                                                         if r.kind == "municipality"},
                                  "trazo_clusters": summary})
    print(gdf[["area", "kind", "n_chips", "km2"]].assign(km2=lambda x: x.km2.round(1)).to_string())
    done("areas")


if __name__ == "__main__":
    main()
