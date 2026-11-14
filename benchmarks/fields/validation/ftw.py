"""Fields of The World, Brazil: hand-labelled fields near Luis Eduardo Magalhaes.

Kerner et al. (2025), CC-BY-4.0. https://source.coop/kerner-lab/fields-of-the-world-brazil
"""

import geopandas as gpd

from common import data_dir, done, get, session, today, write_json

URL = ("https://data.source.coop/kerner-lab/fields-of-the-world-brazil/"
       "boundaries_brazil_2020.parquet")


def main():
    d = data_dir()
    out = d / "ftw_brazil_2020.gpkg"
    if out.exists():
        print(f"[ftw] exists: {out}")
        return
    raw = d / "_raw" / "boundaries_brazil_2020.parquet"
    raw.parent.mkdir(exist_ok=True)
    raw.write_bytes(get(URL, session()).content)
    gdf = gpd.read_parquet(raw)
    cols = [c for c in gdf.columns if c != "geometry"]
    crs0 = str(gdf.crs)
    gdf = gdf.to_crs(4326)
    gdf.to_file(out, driver="GPKG")
    write_json(d / "ftw_brazil_2020.json", {
        "url": URL, "fetched": today(), "licence": "CC-BY-4.0",
        "cite": "Kerner et al. (2025), AAAI 39(27)", "rows": len(gdf), "source_crs": crs0,
        "columns": cols, "bounds": [float(v) for v in gdf.total_bounds],
        "geometry_types": sorted(gdf.geom_type.unique())})
    print(f"  {len(gdf)} polygons, source CRS {crs0}, columns {cols}, bounds {gdf.total_bounds}")
    done("ftw")


if __name__ == "__main__":
    main()
