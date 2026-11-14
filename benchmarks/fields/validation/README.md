# Field-boundary validation data

Download scripts for the reference data used to validate `zeit.fields`
(segmentation accuracy against cadastral, pivot and hand-drawn boundaries).
Nothing here is committed: the data goes to `ZEIT_FIELDS_DATA` (default
`C:\Users\eduar\Documents\dev\zeit_data\fields`). Every script is idempotent
(skips what exists), prints the bytes it transferred and writes GeoPackages in
EPSG:4326 plus a small JSON of provenance (URL, date, counts, columns).
Requests are sequential, with the User-Agent `zeit-validation-script` and
retries with backoff.

Needs: geopandas, pyarrow, fsspec, aiohttp, requests, shapely, tifffile.

```
python benchmarks/fields/validation/municipios.py   # first: the others read it
python benchmarks/fields/validation/car.py
python benchmarks/fields/validation/pivots.py
python benchmarks/fields/validation/ftw.py
python benchmarks/fields/validation/trazo.py
python benchmarks/fields/validation/areas.py        # last: reads municipios + trazo
```

Run the scripts from any directory (they import `common.py` next to them).

## Sources

| Script | Data | Source | Licence / citation | Output |
|---|---|---|---|---|
| `municipios.py` | Municipal boundaries: Porto Murtinho (5006903), Sorriso (5107925), Luis Eduardo Magalhaes (2919553) | IBGE API `malhas/v3`; falls back to the IBGE FTP 2022 municipal mesh (`geoftp.ibge.gov.br`, state zips) when the API does not answer. `municipios.json` says which was used | IBGE, open data | `municipios.gpkg` |
| `car.py` | CAR rural properties (`AREA_IMOVEL`) | source.coop GeoParquet mirror of SICAR, `tristangruppwri/cadastral/brazil-car-area-imovel` (snapshot 2026-07-16, 8.45 M rows, 212 row groups, GeoParquet 1.1 with `bbox` column). Only the row groups whose bbox statistics meet the municipality are read (HTTP range requests), then `cod_imovel` prefix `MS-5006903-` / `MT-5107925-`; aborts above 400 MB | SICAR / Servico Florestal Brasileiro public data; check the mirror's terms before redistributing | `car_porto_murtinho.gpkg`, `car_sorriso.gpkg`, `car_provenance.json` |
| `pivots.py` | Central pivots, 2019 | ANA FeatureServer `SPR/Irrigada_por_Pivos_Mapiados/0`, `CD_GEOCMU` and `ANO='2019'`, paged by 1000 | Cite ANA & Embrapa Milho e Sorgo (2019) | `pivots_2019.gpkg` (column `key` = area) |
| `ftw.py` | Fields of The World, Brazil: hand-labelled fields around Luis Eduardo Magalhaes, 2020 | `kerner-lab/fields-of-the-world-brazil/boundaries_brazil_2020.parquet` | CC-BY-4.0, Kerner et al. (2025), AAAI 39(27) | `ftw_brazil_2020.gpkg` |
| `trazo.py` | Trazo annotations, Chaco: `chaco` (2020), `humidchaco` (2022), `maparaguay` (2024) | `wri-data-lab/trazoannotations/<folder>/`: annotation GPKG, `annotation info.txt`, `chips_<folder>.parquet`, and the header of each chip's `window_b` GeoTIFF (range read, for the chip extent) | CC-BY-4.0, Grupp et al. (2026), Field Boundaries of South America, WRI & ASU | `trazo/trazo_<folder>.gpkg` with layers `annotations`, `chips`, `chip_footprints`; `trazo/trazo.json` |
| `areas.py` | The test-area list | the above | | `areas.gpkg`, `areas.json` |

## Notes

- CAR: `ind_status` is present in the mirror (AT active, PE pending, SU
  suspended, CA cancelled). The scripts drop `ind_status == 'CA'` and the rows
  whose `des_condic` starts with "Cancelado" (a few AT/PE rows say "Cancelado por
  duplicidade": 8 in Porto Murtinho, which keeps 720 properties; Sorriso keeps
  2,769), and record the counts and the `ind_status` x `des_condic` table in
  `car_provenance.json`. `cod_imovel` repeats; the rows are kept and the
  count of repeated rows is recorded.
- CAR, manual alternative: the official download at
  `consultapublica.car.gov.br` is behind an image captcha and is not scripted.
- Trazo: the `chips_<folder>.parquet` geometry is the field polygons of each
  chip, not the chip extent; the chip footprints (256 x 256 px, Sentinel-2) are
  taken from the `window_b` GeoTIFF headers. The dataset README states that the
  annotations completely cover all fields within their chips, but the
  annotation GeoPackages also hold fields outside the chip footprints (about a
  third of the Chaco ones, by their representative point), whose completeness
  is not stated: evaluate inside the footprints only.
- `areas.py` clusters the Trazo footprints that touch or overlap (one area per
  cluster, the union of its footprints) and adds the three municipalities.
- Imagery is not downloaded here.
