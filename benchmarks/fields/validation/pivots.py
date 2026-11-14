"""Central pivots of ANA / Embrapa Milho e Sorgo (2019 layer), per municipality.

Licence/citation: ANA & Embrapa Milho e Sorgo (2019), Levantamento da agricultura
irrigada por pivos centrais no Brasil (safra 2019).
"""

import geopandas as gpd

from common import MUNICIPALITIES, data_dir, done, get, session, today, write_json

URL = ("https://www.snirh.gov.br/arcgis/rest/services/SPR/"
       "Irrigada_por_Pivos_Mapiados/FeatureServer/0")
PAGE = 1000


def main():
    d = data_dir()
    out = d / "pivots_2019.gpkg"
    if out.exists():
        print(f"[pivots] exists: {out}")
        return
    s = session()
    frames, prov = [], {}
    for key, code in MUNICIPALITIES.items():
        feats, off = [], 0
        while True:
            r = get(f"{URL}/query", s, params={
                "where": f"CD_GEOCMU='{code}' AND ANO='2019'", "outFields": "*",
                "f": "geojson", "resultOffset": off, "resultRecordCount": PAGE,
                "outSR": 4326}).json()
            if "error" in r:
                raise SystemExit(f"{key}: {r['error']}")
            feats += r["features"]
            if len(r["features"]) < PAGE:
                break
            off += PAGE
        gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
        gdf["key"] = key
        frames.append(gdf)
        prov[key] = {"code": code, "count": len(gdf),
                     "columns": [c for c in gdf.columns if c != "geometry"]}
        print(f"  {key} ({code}): {len(gdf)} pivots")
    import pandas as pd
    allg = gpd.GeoDataFrame(pd.concat([f for f in frames if len(f)], ignore_index=True),
                            crs="EPSG:4326")
    allg.to_file(out, driver="GPKG")
    write_json(d / "pivots_2019.json", {
        "url": URL, "fetched": today(), "year": "2019",
        "cite": "ANA & Embrapa Milho e Sorgo (2019)", "rows": len(allg), "areas": prov})
    done("pivots")


if __name__ == "__main__":
    main()
