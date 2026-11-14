"""IBGE municipal boundaries of the test areas (EPSG:4326 GeoPackage).

Tries the IBGE API (servicodados.ibge.gov.br) first; when it does not answer,
falls back to the 2022 municipal mesh of IBGE's FTP (geoftp.ibge.gov.br), one
zip per state. Which one was used is written to the provenance JSON.
"""

import io
import zipfile

import geopandas as gpd
import pandas as pd
import requests

from common import MUNICIPALITIES, Counter, data_dir, done, get, session, today, write_json

API = "https://servicodados.ibge.gov.br/api"
FTP = ("https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/"
       "malhas_municipais/municipio_2022/UFs/{uf}/{uf}_Municipios_2022.zip")
UF = {"50": "MS", "51": "MT", "29": "BA"}


def from_api(s):
    frames, prov = [], []
    for key, code in MUNICIPALITIES.items():
        info = get(f"{API}/v1/localidades/municipios/{code}", s, tries=2, timeout=15).json()
        url = f"{API}/v3/malhas/municipios/{code}"
        r = get(url, s, params={"formato": "application/vnd.geo+json"}, tries=2, timeout=60)
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        gdf["name"] = info["nome"]
        gdf["uf"] = info["microrregiao"]["mesorregiao"]["UF"]["sigla"]
        gdf["key"], gdf["code"] = key, code
        frames.append(gdf[["key", "code", "name", "uf", "geometry"]])
        prov.append({"key": key, "url": url})
    return frames, prov


def from_ftp(s):
    frames, prov, cache = [], [], {}
    for key, code in MUNICIPALITIES.items():
        uf = UF[code[:2]]
        url = FTP.format(uf=uf)
        if uf not in cache:
            tmp = data_dir() / "_raw" / "ibge_2022" / uf
            if not tmp.exists():
                zipfile.ZipFile(io.BytesIO(get(url, s, timeout=300).content)).extractall(tmp)
            shp = next(tmp.glob("*.shp"))
            cache[uf] = gpd.read_file(shp, encoding="utf-8").to_crs(4326)
        g = cache[uf]
        sel = g[g["CD_MUN"] == code].copy()
        assert len(sel) == 1, (code, len(sel))
        sel["key"], sel["code"], sel["name"], sel["uf"] = key, code, sel["NM_MUN"].iloc[0], uf
        frames.append(sel[["key", "code", "name", "uf", "geometry"]])
        prov.append({"key": key, "url": url})
    return frames, prov


def main():
    out = data_dir() / "municipios.gpkg"
    if out.exists():
        print(f"[municipios] exists: {out}")
        return
    s = session()
    try:
        frames, prov = from_api(s)
        source = "IBGE API malhas v3"
    except requests.RequestException as e:
        print(f"  IBGE API unreachable ({type(e).__name__}); using the IBGE FTP 2022 mesh")
        Counter.bytes = 0
        frames, prov = from_ftp(s)
        source = "IBGE FTP municipio_2022 (API unreachable)"
    all_ = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    for p, (_, row) in zip(prov, all_.iterrows()):
        p.update(code=row["code"], name=row["name"], uf=row["uf"],
                 bounds=[float(v) for v in row.geometry.bounds])
        print(f"  {row['code']} {row['name']}/{row['uf']}: bounds {row.geometry.bounds}")
    all_.to_file(out, driver="GPKG")
    write_json(data_dir() / "municipios.json",
               {"source": source, "fetched": today(), "rows": len(all_), "items": prov})
    done("municipios")


if __name__ == "__main__":
    main()
