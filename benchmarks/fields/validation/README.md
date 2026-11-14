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
- Imagery is not downloaded by the scripts above: see `mosaics.py` below.

## Landsat mosaics (`mosaics.py`)

Annual Landsat 8/9 Collection 2 Level-2 medoids of the test windows, the input of the sensitivity
analysis (24e). Source: the Planetary Computer STAC (`landsat-c2-l2`, credential-free, signed with
`planetary_computer`), windowed COG reads, built with Zeit's own pieces (`zeit.cube.build_time_series`
for the raw digital numbers, `zeit.cube._composite_block` for the medoid, `zeit.save_raster` for the COG).
Outputs go to `ZEIT_FIELDS_DATA/mosaics/`:

| File | What |
|---|---|
| `<name>_<year>.tif` | COG, native UTM zone, 30 m, on Landsat's own pixel lattice (edges at 15 mod 30 m, so nothing is resampled), 6 bands `blue green red nir08 swir16 swir22`, int16, NoData -32768 |
| `<name>_<year>_clear.tif` | uint8, the clear observations of each pixel |
| `<name>_<year>.json` | provenance: scenes (id, date, platform, path/row, cloud cover, clear fraction), quantiles of the clear observations, the conversion, the mask, seconds, bytes received |
| `<name>_<year>_quicklook.png` | SWIR1 / NIR / red, 2-98 percent stretch |
| `windows.json`, `plan.json`, `index.json` | the windows and why, the bytes estimates, the read-back check |

Values are surface reflectance x 10000, the scale of the notebook (`B5 > 2000`): Collection 2's
`reflectance = DN * 0.0000275 - 0.2`, so `value = round((DN * 0.0000275 - 0.2) * 10000)`. The notebook's
"B5" is SWIR1, `swir16` (band 5 of the file; B6 of Landsat 8/9). Clear observations: QA_PIXEL with the
fill, dilated cloud, cirrus, cloud and cloud-shadow bits at 0 and the clear bit at 1 (Zeit's stock Landsat
mask is the clear bit alone, which keeps the shadow; the script swaps it for this one inside the build);
a zero in any band is no observation. Landsat 8 and 9 only (no Landsat 7 SLC-off stripes). The medoid is
over the calendar year (the annotation year of each reference), of the scenes whose scene cloud cover is
below `--cloud` (40 for the Brazilian windows, 20 for Trazo, to fit the transfer budget); the pixels are
masked anyway.

```
python benchmarks/fields/validation/mosaics.py plan  --group sorriso,porto_murtinho,lem --cloud 40
python benchmarks/fields/validation/mosaics.py build --group sorriso,porto_murtinho --cloud 40
python benchmarks/fields/validation/mosaics.py build --only lem_1 --cloud 40
python benchmarks/fields/validation/mosaics.py plan  --group chaco,humidchaco,maparaguay --cloud 20
python benchmarks/fields/validation/mosaics.py build --group chaco,humidchaco --cloud 20
python benchmarks/fields/validation/mosaics.py check         # reads them back
python benchmarks/fields/validation/mosaics.py smoke sorriso_1_2019
```

`plan` searches the STAC and prints the scenes and the estimated bytes of every window before anything
is downloaded; `build` skips the windows already built and refuses one whose estimate (x 1.6, the
estimate being low where the tiles are dense) would pass `--max-gb` (6) with what was already received
(`--extra-gb` for transfers that left no JSON). Reads use 8 dask workers and `GDAL_NUM_THREADS=1`.
Run it from PowerShell with the zeit environment's folders first on `PATH`.

| Windows | Year | Why |
|---|---|---|
| `sorriso_1`, `sorriso_2`: 20 x 20 km inside the municipality | 2019 | the two best non-overlapping windows by CAR properties + 10 x pivots 2019 (pivots are rarer and are real fields); `sorriso_1` holds the town |
| `porto_murtinho_1`, `_2`: 20 x 20 km | 2019 | the CAR property count, centre east of 57.5 W (no Pantanal layer at hand: the floodplain and the town are on the river side). Hilly, much of it forest |
| `lem_1`: 40 x 40 km (`lem_2` planned, not built) | 2020 | the densest 40 km window of the FTW fields (650 of 1,854); the FTW extent is 250 km tall, far beyond the municipality. Calendar 2020: the FTW labels are dated 2020-03 to 2020-09 and the 2019/20 season ends with its harvest in 2020 |
| `trazo_<folder>_NNN`: each cluster of `areas.gpkg` padded by 1 km | `chaco` 2020, `humidchaco` 2022, `maparaguay` 2024 | the annotation year of the folder |

What has not been downloaded, for the transfer cap, is in `plan.json` with its estimate: `lem_2`, the
Trazo `humidchaco` clusters 011-013 and 015-025, and all of `maparaguay` (89 clusters, estimated 7.4 GB at
`--cloud 20`).
