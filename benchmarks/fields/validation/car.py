"""CAR property polygons (AREA_IMOVEL) of Porto Murtinho and Sorriso.

Source: the source.coop GeoParquet mirror of SICAR (SFB). Only the row groups
whose bbox statistics meet the municipality are read (HTTP range requests), never
the whole file; the script aborts when that would exceed MAX_BYTES.
"""

import fsspec
import geopandas as gpd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import USER_AGENT, Counter, data_dir, done, today, write_json

URL = ("https://data.source.coop/tristangruppwri/cadastral/"
       "brazil-car-area-imovel/brazil_car_area_imovel.parquet")
MAX_BYTES = 400e6
# output key -> (municipality key in municipios.gpkg, cod_imovel prefix)
AREAS = {"porto_murtinho": "MS-5006903-", "sorriso": "MT-5107925-"}


def main():
    d = data_dir()
    todo = [k for k in AREAS if not (d / f"car_{k}.gpkg").exists()]
    if not todo:
        print("[car] all outputs exist")
        return
    muni = gpd.read_file(d / "municipios.gpkg").set_index("key")
    fs = fsspec.filesystem("https", headers={"User-Agent": USER_AGENT})
    prov = {}
    f = fs.open(URL, block_size=1 << 20)
    pf = pq.ParquetFile(f)
    md = pf.metadata
    names = [md.row_group(0).column(i).path_in_schema for i in range(md.num_columns)]
    ix = {n: i for i, n in enumerate(names)}
    print(f"  file: {md.num_rows:,} rows, {md.num_row_groups} row groups")
    for k in todo:
        x0, y0, x1, y1 = muni.loc[k].geometry.bounds
        groups, comp = [], 0
        for g in range(md.num_row_groups):
            r = md.row_group(g)

            def st(n):
                return r.column(ix[n]).statistics

            if (st("bbox.xmin").min <= x1 and st("bbox.xmax").max >= x0
                    and st("bbox.ymin").min <= y1 and st("bbox.ymax").max >= y0):
                groups.append(g)
                comp += sum(r.column(i).total_compressed_size for i in range(md.num_columns))
        print(f"  {k}: {len(groups)} row groups {groups}, ~{comp / 1e6:.0f} MB compressed")
        if comp > MAX_BYTES:
            raise SystemExit(f"{k}: would transfer {comp / 1e6:.0f} MB > {MAX_BYTES / 1e6:.0f} MB")
        t = pf.read_row_groups(groups)
        got = getattr(f.cache, "total_requested_bytes", comp)
        t = t.filter(pc.starts_with(t["cod_imovel"], AREAS[k]))
        cols = t.schema.names
        geom = gpd.GeoSeries.from_wkb(t["geometry"].to_pandas(), crs="EPSG:4326")
        df = t.drop_columns(["geometry", "bbox"]).to_pandas()
        gdf = gpd.GeoDataFrame(df, geometry=geom.values, crs="EPSG:4326")
        n_all = len(gdf)
        status = gdf["ind_status"].value_counts(dropna=False).to_dict()
        cond = gdf["des_condic"].value_counts(dropna=False).to_dict()
        cross = {f"{a}|{b}": int(n) for (a, b), n in
                 gdf.groupby(["ind_status", "des_condic"]).size().items()}
        # ind_status is present in the mirror: CA = cancelled (AT active, PE pending,
        # SU suspended). Dropping CA is the rule, and also the rows whose des_condic says
        # "Cancelado..." (e.g. "Cancelado por duplicidade" on a few AT/PE rows).
        gdf = gdf[(gdf["ind_status"] != "CA") & ~gdf["des_condic"].fillna("").str.startswith("Cancelado")]
        ndup = int(gdf["cod_imovel"].duplicated(keep=False).sum())
        gdf.to_file(d / f"car_{k}.gpkg", driver="GPKG")
        prov[k] = {"prefix": AREAS[k], "rows_with_prefix": n_all, "rows_kept": len(gdf),
                   "dropped_ind_status_CA": n_all - len(gdf), "status_x_condic": cross,
                   "ind_status_counts": status, "des_condic_counts": cond,
                   "rows_with_repeated_cod_imovel": ndup,
                   "row_groups": groups, "bytes_requested": int(got),
                   "columns": [c for c in cols if c not in ("geometry", "bbox")]}
        print(f"  {k}: {n_all} with prefix, {len(gdf)} kept after dropping ind_status CA; "
              f"ind_status {status}; repeated cod_imovel rows {ndup}; ~{got / 1e6:.0f} MB")
    write_json(d / "car_provenance.json" if len(todo) == len(AREAS)
               else d / f"car_provenance_{'_'.join(todo)}.json",
               {"url": URL, "fetched": today(), "snapshot": "SICAR 2026-07-16 (per research)",
                "file_rows": md.num_rows, "filter": "cod_imovel prefix; ind_status != CA; des_condic not Cancelado*",
                "areas": prov})
    Counter.bytes = int(got)  # the cache is shared: cumulative over both areas
    done("car")


if __name__ == "__main__":
    main()
