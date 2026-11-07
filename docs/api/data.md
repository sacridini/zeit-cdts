# Data & I/O

<p class="lead">Build data cubes from cloud catalogs, Earth Engine or local files; read and write GeoTIFFs; composite to a regular time step; and turn QA bands into observation weights.</p>

## Building cubes

### `build_time_series` { .api }

<!-- sig: zeit.cube.build_time_series -->
```python
zeit.cube.build_time_series(
    source="earth_search", collection="sentinel-2-l2a", bbox=None,
    vector_path=None, tiles=None, start_date="2020-01-01",
    end_date="2020-12-31", cloud_cover_max=30, bands=None,
    apply_cloud_mask=False, resolution=None, epsg=4326,
    validate_items=False, access_token=None, chunksize=2048,
    dtype="float32",
)
```

Builds a lazy, Dask-backed `xarray.DataArray` from a STAC catalog. It searches the catalog, keeps the items matching the area, dates and cloud-cover limit, and stacks them into an aligned cube shaped `(time, band, y, x)`. Nothing is downloaded until the cube is computed. Also exported as `zeit.build_time_series`. Tutorial: [STAC Data Cubes](../tutorials/stac-downloads.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str` | `"earth_search"` | Catalog alias (`"earth_search"`, `"planetary_computer"`, `"brazil_data_cube"`) or any STAC API URL. |
| `collection` | `str` or `list` | `"sentinel-2-l2a"` | Collection ID, e.g. `"sentinel-2-l2a"`, `"landsat-c2-l2"`, `"modis-13Q1-061"`. Available collections depend on `source`. HLS (`"hls2-l30"`, `"hls2-s30"` on Planetary Computer) is read as reflectance, and `["hls2-l30", "hls2-s30"]` builds both and joins them into one series sorted by time (see [Landsat and Sentinel-2 in one series](preprocessing.md#landsat-and-sentinel-2-in-one-series)). |
| `bbox` | `list` | `None` | `[min_lon, min_lat, max_lon, max_lat]` in EPSG:4326. |
| `vector_path` | `str` | `None` | Vector file (Shapefile, GeoJSON) whose bounds define the area. |
| `tiles` | `list[str]` | `None` | Sentinel-2 MGRS tiles (`"22JFQ"`) or Landsat WRS-2 path/rows (`"215065"`). |
| `start_date`, `end_date` | `str` | `"2020-01-01"`, `"2020-12-31"` | Date range, `YYYY-MM-DD`. |
| `cloud_cover_max` | `int` | `30` | Maximum scene cloud cover, in percent (metadata filter). |
| `bands` | `list[str]` | `None` | Assets to load, e.g. `["red", "nir"]`. For HLS, band roles (`coastal`, `blue`, `green`, `red`, `nir`, `swir1`, `swir2`) are read from each product's own asset (the NIR is L30's `B05` and S30's `B8A`) and keep the role's name; with both products, roles are required. |
| `apply_cloud_mask` | `bool` | `False` | Also load the QA band (`scl` for Sentinel-2, `qa_pixel` for Landsat, `Fmask` for HLS: cloud, adjacent to cloud or shadow, and shadow) and mask clouds and shadows. Other collections warn that nothing was masked. |
| `resolution` | `float` | `None` | Output pixel size, in units of `epsg`. |
| `epsg` | `int` | `4326` | Output coordinate reference system. |
| `validate_items` | `bool` | `False` | Test each asset URL first and drop broken ones. |
| `access_token` | `str` | `None` | Token for catalogs that require one (e.g. Brazil Data Cube). |
| `chunksize` | `int` | `2048` | Spatial chunk size, in pixels. Large chunks mean fewer HTTP requests and fewer COG blocks read twice at chunk edges. |
| `dtype` | `str` | `"float32"` | Float types hold reflectance (scale and offset applied, NaN = nodata). Integer types (`"uint16"`) hold the raw digital numbers (0 = nodata) at half the memory, with per-scene `scale` and `offset` coordinates to recover reflectance; `apply_cloud_mask` must then stay `False`. |

</div>

**Returns** a lazy `xarray.DataArray` `(time, band, y, x)`.

Missing files (HTTP 404) and corrupted blocks become nodata. Access errors raise: Landsat on Earth Search lives in the requester-pays `s3://usgs-landsat` bucket and needs AWS credentials (the requester pays the transfer), while Planetary Computer serves the same Landsat files for free.

```python
import zeit

cube = zeit.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    bbox=[-48.0, -16.0, -47.9, -15.9],
    start_date="2021-01-01",
    end_date="2021-12-31",
    bands=["red", "nir"],
    apply_cloud_mask=True,
    resolution=10,
    epsg=32722,
)
```

### `build_annual_composites` { .api }

<!-- sig: zeit.cube.build_annual_composites -->
```python
zeit.cube.build_annual_composites(
    source="planetary_computer", collection="landsat-c2-l2",
    bbox=None, vector_path=None, start_year=1985, end_year=2024,
    season=('06-01', '09-30'), bands=None, cloud_cover_max=30,
    method="median", apply_cloud_mask=True, resolution=30, epsg=4326,
    access_token=None, chunksize=2048, target_doy=None, doy_sigma=30.0,
    cloud_distance=(30.0, 1500.0), weights=None, bap_info=False,
)
```

Builds one cloud-masked composite per year from a STAC catalog: the annual stack LandTrendr expects. Scenes are read as raw integers and reduced one spatial chunk at a time (QA mask, per-scene scale and offset, then median, medoid or best available pixel), so memory stays bounded by a few chunks however many scenes a year has, and every pixel is downloaded once. Also exported as `zeit.build_annual_composites`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str` | `"planetary_computer"` | Catalog alias or STAC API URL, as in `build_time_series`. |
| `collection` | `str` | `"landsat-c2-l2"` | Collection ID. The QA band is picked from it (`qa_pixel` for Landsat, `scl` for Sentinel-2). |
| `bbox` | `list` | `None` | `[min_lon, min_lat, max_lon, max_lat]` in EPSG:4326. `bbox` or `vector_path` is required so every year shares one grid. |
| `vector_path` | `str` | `None` | Vector file whose bounds define the area. |
| `start_year`, `end_year` | `int` | `1985`, `2024` | Inclusive range of years. |
| `season` | `tuple` | `("06-01", "09-30")` | `("MM-DD", "MM-DD")` window inside each year. A window that wraps the new year (`("12-01", "02-28")`) is labelled with its first year. |
| `bands` | `list[str]` | `None` | Bands to composite, e.g. `["nir08", "swir22"]` for NBR. Required. |
| `cloud_cover_max` | `int` | `30` | Maximum scene cloud cover, in percent. |
| `method` | `str` | `"median"` | `"median"` (per band), `"medoid"` (the real observation closest to the multi-band median) or `"bap"` (the best available pixel: each pixel's clear observation of the highest score, every band from that date; below). |
| `apply_cloud_mask` | `bool` | `True` | Mask clouds and shadows with the collection's QA band. |
| `target_doy` | `float` | `None` | `"bap"`: the day of the year the composite aims at (default: the middle of `season`). |
| `doy_sigma` | `float` | `30.0` | `"bap"`: days from the target at which the day's score falls to 0.61 (`exp(-0.5 (days / doy_sigma)²)`). |
| `cloud_distance` | `tuple` | `(30.0, 1500.0)` | `"bap"`: metres to the nearest cloud or shadow of the scene at which the cloud's score is 0 and 1 (linear between; a scene without clouds scores 1). |
| `weights` | `dict` | `None` | `"bap"`: the score's weights, `zeit.cube.BAP_WEIGHTS` by default: `doy` 0.3, `cloud` 0.5, `sensor` 0.1 (Landsat 7 after its scan line corrector failed scores 0.5, the others 1), `cover` 0.1 (1 − the scene's `eo:cloud_cover`). |
| `bap_info` | `bool` | `False` | `"bap"`: adds the bands `bap_doy` (the day of the year of each pixel's date), `bap_score` and `bap_clear` (its clear observations). |
| `resolution` | `float` | `30` | Output pixel size, in units of `epsg`. |
| `epsg` | `int` | `4326` | Output coordinate reference system. |
| `access_token` | `str` | `None` | Token for catalogs that require one. |
| `chunksize` | `int` | `2048` | Spatial chunk size, in pixels. Memory per chunk in flight is about `scenes × bands × chunksize² × 2` bytes. |

</div>

**Returns** a lazy float32 `xarray.DataArray` `(time, band, y, x)` of reflectance, with `time` on January 1st of each year and a `year` coordinate. Years without scenes are NaN, so the time axis has no gaps.

```python
import zeit

comp = zeit.build_annual_composites(
    source="planetary_computer",
    collection="landsat-c2-l2",
    bbox=[9.63, 50.92, 12.06, 52.43],
    start_year=1985,
    end_year=2024,
    bands=["nir08", "swir22"],
    epsg=3035,
)
nbr = zeit.compute_indices(comp, ["NBR"]).sel(band="NBR")
zeit.save_raster(nbr, "nbr_1985_2024.tif")   # one band per year, ready for LandTrendr
```

The best available pixel (`method="bap"`) scores each clear observation the way Griffiths et al. (2013) and White et al. (2014) do: a weighted sum of its day of the year near a target day, its distance to the nearest cloud or shadow (the exact Euclidean distance, in zeit's core), its sensor and its scene's cloud cover; each pixel takes the observation of the highest score (the earlier date of equal ones), so its bands come from one date, as the medoid's do, but chosen for its quality rather than for being typical. Each chunk is read with the margin of the cloud distance around it, so the composite does not depend on the chunks. On 60 dates of 1024 × 1024 pixels the score and the choice take 2.7 s on one thread and 0.85 s on four, against 5.9 s for the same score in numpy and scipy (the same date at every pixel; `benchmarks/vs_originals/bap.py`).

```python
comp = zeit.build_annual_composites(
    bbox=[-63.05, -9.95, -62.95, -9.85], start_year=2023, end_year=2023, season=("06-01", "09-30"),
    bands=["red", "nir08", "swir16"], method="bap", epsg=32720, resolution=30, cloud_cover_max=70,
    bap_info=True,
)
comp.sel(band="bap_doy")   # the date each pixel came from (day 211 nearly everywhere here: 2 of 33 scenes)
```

### `build_spectral_temporal_metrics` { .api }

<!-- sig: zeit.cube.build_spectral_temporal_metrics -->
```python
zeit.cube.build_spectral_temporal_metrics(
    source="planetary_computer", collection="landsat-c2-l2",
    bbox=None, vector_path=None, start_year=1985, end_year=2024,
    season=('01-01', '12-31'), indices=None,
    metrics=('median', 'p10', 'p25', 'p75', 'p90', 'std'),
    band_map=None, cloud_cover_max=30, apply_cloud_mask=True,
    resolution=30, epsg=4326, access_token=None, chunksize=1024,
)
```

Builds per-year spectral temporal metrics (STMs) from a STAC catalog: the median, percentiles, spread and so on of each index over every clear observation of a season. Each index is computed on every observation first, and the statistics are taken over time afterwards (the median NDVI, not the NDVI of the median bands). Scenes are read as raw integers and reduced one spatial chunk at a time, as in `build_annual_composites`, so the raw bands pass through memory once and only the metrics are kept. Also exported as `zeit.build_spectral_temporal_metrics`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | `str` | `"planetary_computer"` | Catalog alias or STAC API URL, as in `build_time_series`. |
| `collection` | `str` | `"landsat-c2-l2"` | Collection ID. Picks the QA band and the default asset of each band role. |
| `bbox` | `list` | `None` | `[min_lon, min_lat, max_lon, max_lat]` in EPSG:4326. `bbox` or `vector_path` is required. |
| `vector_path` | `str` | `None` | Vector file whose bounds define the area. |
| `start_year`, `end_year` | `int` | `1985`, `2024` | Inclusive range of years. |
| `season` | `tuple` | `("01-01", "12-31")` | `("MM-DD", "MM-DD")` window inside each year; wraps the new year as in `build_annual_composites`. |
| `indices` | `list[str]` | `None` | What to summarize: indices (`NDVI`, `EVI`, `SAVI`, `kNDVI`, `NBR`, `NDMI`, `NDWI`, `MNDWI`, `NDFI`, `TCB`, `TCG`, `TCW`, `TCD`), band roles (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`) or asset names. Required. |
| `metrics` | `list[str]` | `('median', 'p10', 'p25', 'p75', 'p90', 'std')` | `median`, `mean`, `std` (population), `min`, `max`, `iqr` (p75 − p25), `count` (clear observations) or any integer percentile such as `p5`. |
| `band_map` | `dict` | `None` | Asset of each band role, e.g. `{"nir": "B8A"}`. Known for Landsat Collection 2 and Sentinel-2 L2A on Planetary Computer and Earth Search; needed for other collections. |
| `cloud_cover_max` | `int` | `30` | Maximum scene cloud cover, in percent. |
| `apply_cloud_mask` | `bool` | `True` | Mask clouds and shadows with the collection's QA band. |
| `resolution` | `float` | `30` | Output pixel size, in units of `epsg`. |
| `epsg` | `int` | `4326` | Output coordinate reference system. |
| `access_token` | `str` | `None` | Token for catalogs that require one. |
| `chunksize` | `int` | `1024` | Spatial chunk size, in pixels. Memory per chunk in flight is about `scenes × chunksize² × (2 bytes per band loaded + 4 per band or index used)`. |

</div>

**Returns** a lazy float32 `xarray.DataArray` `(time, band, y, x)` with one band per index and metric, named `"<index>_<metric>"` (`"NDVI_p10"`), `time` on January 1st of each year and a `year` coordinate. Years without scenes are NaN.

```python
import zeit

stm = zeit.build_spectral_temporal_metrics(
    source="planetary_computer",
    collection="sentinel-2-l2a",
    bbox=[-47.95, -15.85, -47.90, -15.80],
    start_year=2021,
    end_year=2024,
    indices=["NDVI", "NBR"],
    metrics=["median", "p10", "p90", "iqr"],
    resolution=20,
    epsg=32723,
)
zeit.save_raster(stm.sel(year=2024), "stm_2024.tif")   # 8 bands: NDVI_median ... NBR_iqr
```

### `compute_indices` { .api }

<!-- sig: zeit.indices.compute_indices -->
```python
zeit.indices.compute_indices(cube, indices, band_map=None)
```

Computes spectral indices from a reflectance cube, lazily, so a cube from `build_time_series` or `build_annual_composites` can be saved as indices without ever writing its bands. Each band role is found by its usual asset names (`red`, `B04`, `SR_B4`...), so the same call works for Landsat and Sentinel-2 from any of the built-in catalogs. Also exported as `zeit.compute_indices`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `xr.DataArray` | required | Float reflectance cube with a `band` dimension. Raw integer cubes are rejected. |
| `indices` | `list[str]` | required | `NDVI`, `EVI`, `SAVI`, `kNDVI`, `NBR`, `NDMI`, `NDWI`, `MNDWI`, `NDFI` (any case; the NDFI unmixes the six Landsat bands, see [`unmix`](preprocessing.md#unmix)), or the Tasseled Cap `TCB` (brightness), `TCG` (greenness), `TCW` (wetness) and `TCD` (FORCE's disturbance, `TCB - TCG - TCW`), all with Crist's (1985) matrix in reflectance units, see [`tasseled_cap`](preprocessing.md#tasseled_cap) (for Sentinel-2 pass `band_map={"nir": "nir08"}`, FORCE's NIR). |
| `band_map` | `dict` | `None` | Band to read a role from when the default doesn't fit, e.g. `{"nir": "B8A"}`. |

</div>

**Returns** a DataArray with the cube's dimensions whose `band` coordinate lists the indices. Divisions by zero become NaN.

```python
cube = zeit.build_time_series(
    source="planetary_computer", collection="sentinel-2-l2a",
    bbox=[-47.95, -15.85, -47.90, -15.80],
    start_date="2024-01-01", end_date="2024-12-31",
    bands=["B02", "B04", "B08", "B12"], apply_cloud_mask=True,
    resolution=10, epsg=32723,
)
idx = zeit.compute_indices(cube, ["NDVI", "EVI", "NBR"])   # (time, 3, y, x), still lazy
```

### `nd` { .api }

<!-- sig: zeit.indices.nd -->
```python
zeit.indices.nd(a, b)
zeit.indices.normalized_difference(cube, a, b, band_map=None)
```

The normalized difference `(a - b) / (a + b)`, NaN where `a + b` is 0: the formula of NDVI (`nd(nir, red)`), NDWI, NBR and the other normalized differences of `compute_indices`. `nd` takes two numpy arrays, scalars or `DataArray` objects (lazy on Dask; integers are computed as floats, so unsigned values cannot wrap around). `normalized_difference` takes two bands of a cube with a `band` dimension, by name or by role (`"nir"`, `"red"`, `"swir1"`...), found by their usual asset names or `band_map` as in `compute_indices`, and returns the cube without `band`. Both are exported at the top level.

```python
ndvi = zeit.normalized_difference(cube, "nir", "red")          # (time, band, y, x) -> (time, y, x)
ndwi = zeit.nd(cube.sel(band="green"), cube.sel(band="nir"))
```

### `download_gee_timeseries` { .api }

<!-- sig: zeit.gee.download_gee_timeseries -->
```python
zeit.gee.download_gee_timeseries(
    roi, start_date, end_date, out_dir, method="auto",
    composite_type="annual", indices=None, project=None,
    metrics=('median', 'p10', 'p25', 'p75', 'p90', 'std'), bands=None,
)
```

Builds harmonised Landsat 5/7/8/9 composites on Google Earth Engine and downloads one GeoTIFF per composite. By default (`method="auto"`) it uses a concurrent tiled direct download and falls back to a Google Drive export for very large images or ones that hit Earth Engine's interactive compute limits. Tutorial: [Google Earth Engine](../tutorials/gee-downloads.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `roi` | `str`, `list`, file path, GeoDataFrame or `ee.Geometry` | required | A Landsat WRS-2 path/row (`"217/076"`), a Sentinel-2 tile (`"23KPQ"`), a bbox `[min_lon, min_lat, max_lon, max_lat]`, a vector or raster file, a GeoDataFrame / shapely geometry, or an `ee.Geometry`. Local inputs are read offline; their bounding box is downloaded. |
| `start_date`, `end_date` | `str` | required | Date range, `YYYY-MM-DD`. |
| `out_dir` | `str` | required | Output folder (for `"drive"`, used to name the exports). |
| `method` | `str` | `"auto"` | `"auto"` (direct, Drive when needed), or force `"direct"` / `"drive"`. |
| `composite_type` | `str` | `"annual"` | `"annual"` medoid composites (LandTrendr), `"dense"` (every observation, CCDC) or `"stm"` (spectral temporal metrics per year). |
| `indices` | `list` | `None` | Indices (`"NDVI"`, `"EVI"`, `"SAVI"`, `"kNDVI"`, `"NBR"`, `"NDMI"`, `"NDWI"`, `"MNDWI"`, `"NDFI"`, `"TCB"`, `"TCG"`, `"TCW"`, `"TCD"`) and SR bands (`"SR_B2"` … `"SR_B7"`) to export, as in `build_spectral_temporal_metrics`. Indices are computed on surface reflectance; SR bands are exported as Collection 2 digital numbers. Default: the six reflective bands. |
| `project` | `str` | `None` | Google Cloud project used to initialise Earth Engine. Recommended. |
| `metrics` | `list` | `('median', 'p10', 'p25', 'p75', 'p90', 'std')` | With `composite_type="stm"`: `median`, `mean`, `std`, `min`, `max`, `iqr`, `count` or a percentile such as `"p10"`. |
| `bands` | `list` | `None` | Deprecated name of `indices` (warns). |

</div>

```python
from zeit.gee import download_gee_timeseries

download_gee_timeseries(
    roi="217/076",                      # WRS-2 path/row; or a bbox, .shp, .gpkg, .tif ...
    start_date="1985-01-01", end_date="2025-12-31",
    out_dir="./gee_data", composite_type="annual",
    indices=["NBR"], project="my-gcp-project",
)

# Spectral temporal metrics, reduced on Earth Engine: landsat_stm_2015.tif ... (8 bands each)
download_gee_timeseries(
    roi="217/076", start_date="2015-01-01", end_date="2024-12-31",
    out_dir="./gee_stm", composite_type="stm",
    indices=["NDVI", "NBR"], metrics=["median", "p10", "p90", "iqr"],
    project="my-gcp-project",
)
```

Masked pixels are written as `-inf` (float outputs), the files' NoData value: `load_raster` reads them as `NaN`, which LandTrendr treats as a missing year. See the [worked example](../tutorials/gee-downloads.md#worked-example-a-full-landsat-tile-19852025-ready-for-landtrendr).

### `download_gee_image` { .api }

<!-- sig: zeit.gee.downloader.download_gee_image -->
```python
zeit.gee.downloader.download_gee_image(
    image, roi, out_filename, method="auto", scale=30, tile_size=None,
    sub_tile_workers=16, crs="EPSG:4326", max_tile_mb=None,
    max_direct_mb=4096, max_retries=5, base_backoff=5.0,
)
```

Downloads one `ee.Image` to a local GeoTIFF by the fastest route that works. This is what `download_gee_timeseries` calls for each composite; use it directly for images you build yourself or to tune the download. The direct route fixes one pixel grid for the whole ROI and fetches it as tiles with concurrent `ee.data.computePixels` calls, writing each tile into its window of the output file (no temporary tiles, no mosaicking). Concurrency adapts to `HTTP 429` responses.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `ee.Image` | required | The image to download. |
| `roi` | `ee.Geometry` | required | Region whose bounding box is downloaded. Build it with `resolve_roi`. |
| `out_filename` | `str` | required | Output GeoTIFF. Never left partially written. |
| `method` | `str` | `"auto"` | `"auto"`, `"direct"` or `"drive"`. |
| `scale` | `float` | `30` | Pixel size in metres (converted to degrees at the equator for a geographic CRS, as Earth Engine does). |
| `tile_size` | `float` | `None` | Deprecated and ignored. |
| `sub_tile_workers` | `int` | `16` | Upper bound on concurrent requests. Starts at 4 and adapts to the account's limit (about 40 on a standard tier, about 2 in Restricted Mode). |
| `crs` | `str` | `"EPSG:4326"` | Output CRS, e.g. `"EPSG:32723"`. |
| `max_tile_mb` | `float` | `None` | Cap on one tile request, in MB (at most 32, the default). |
| `max_direct_mb` | `float` | `4096` | With `"auto"`, larger images (raw size) go to a Drive export. |
| `max_retries`, `base_backoff` | | `5`, `5.0` | Retry policy for network or server errors. Throttled requests get short jittered retries instead. |

</div>

**Returns** the output path, or `None` if the download failed (nothing is written in that case).

```python
from zeit.gee.auth import initialize_gee
from zeit.gee.roi import resolve_roi
from zeit.gee.harmonization import get_harmonized_collection
from zeit.gee.composites import create_annual_medoid
from zeit.gee.downloader import download_gee_image

initialize_gee(project="my-gcp-project")
roi = resolve_roi("data/study_area.gpkg")       # or "217/076", "23KPQ", a bbox ...
col = get_harmonized_collection(roi, "1985-01-01", "2025-12-31")

for year in range(1985, 2026):
    img = create_annual_medoid(col, year)
    img = img.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR").toFloat()
    download_gee_image(img, roi, f"nbr_{year}.tif", crs="EPSG:32723", sub_tile_workers=32)
```

`initialize_gee(project=None, *, key_file=None)` uses the credentials `earthengine authenticate` saved, and asks to authenticate when there are none, in a terminal or a notebook. In a batch job, with no one to answer, it raises a `RuntimeError` that says what to do instead of waiting: authenticate once on the login node (the credentials go to `~/.config/earthengine`, which compute nodes usually see), or use a service account, whose JSON key `key_file` or the variable `GOOGLE_APPLICATION_CREDENTIALS` names (its project is the default `project`). Without internet access it says so, and that `zeit fetch tiles` builds the tiles of `gee_tiles` beforehand.

### `resolve_roi` { .api }

<!-- sig: zeit.gee.roi.resolve_roi -->
```python
zeit.gee.roi.resolve_roi(roi)
```

Turns any supported area description into the `ee.Geometry` the Earth Engine functions need, so your code never builds Earth Engine objects. Local inputs are reduced to their bounding box.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `roi` | see below | required | Area description. |

</div>

| Input | Example | Result |
| :--- | :--- | :--- |
| WRS-2 path/row | `"217/076"`, `"217_076"`, `"217076"` | Tile footprint, looked up from a Landsat Collection 2 scene |
| Sentinel-2 MGRS tile | `"23KPQ"`, `"T23KPQ"` | 109.8 km tile computed offline (within ~50 m of real footprints). Tiles crossing 180° raise `ValueError` |
| Bounding box | `[-43.6, -23.1, -43.1, -22.6]` | That box (lon/lat) |
| Vector file | `.shp`, `.gpkg`, `.geojson`, `.kml` | Extent of all features, reprojected to lon/lat |
| Raster file | `reference.tif` | Raster extent, reprojected to lon/lat |
| GeoDataFrame / GeoSeries | `geopandas.read_file(...)` | Extent, reprojected (no CRS: lon/lat assumed, with a warning) |
| shapely geometry | `box(...)` | Its bounds (assumed lon/lat) |
| `ee.Geometry` | | Passed through |

Sentinel-1 has no fixed tiling grid, so describe Sentinel-1 areas with any of the other inputs. Two offline helpers live in the same module: `zeit.gee.roi.roi_bounds(roi)` returns the lon/lat bounding box of a local input or a Sentinel-2 tile, and `zeit.gee.roi.s2_tile_utm_bounds("23KPQ")` returns a tile's exact UTM box, e.g. `("EPSG:32723", (600000, 7390200, 709800, 7500000))`.

## Compositing

### `regularize_time_series` { .api }

<!-- sig: zeit.regularize.regularize_time_series -->
```python
zeit.regularize.regularize_time_series(
    cube, freq="16D", method="median", fill=None, max_gap=None, lmbda=10.0,
)
```

Composites an irregular cube to a fixed time step (16-day, monthly, yearly) with the median, the medoid or the clearest observation of each window, and can fill the periods without a value. Missing values (NaN and the cube's NoData) are left out; periods without observations, and pixels without one in a period, are NaN (an integer cube with NoData keeps its type, with NoData there). Georeferencing and NoData are kept. Stays lazy. Also exported as `zeit.regularize_time_series`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `DataArray` or path | required | Cube with a `time` dimension, `(time, y, x)` or `(time, band, y, x)`, or anything `load_raster` reads. |
| `freq` | `str` | `"16D"` | pandas frequency: `"16D"`, `"1MS"`, `"1YS"`, … |
| `method` | `str` | `"median"` | `"median"` (per band), `"clearest"` or `"medoid"`. `"clearest"` ranks the dates of a period by the share of the cube's pixels that have every band (the clearest first, the earlier date on a tie) and gives each pixel the observation, with every band, of the first date that has one: a composite is a mosaic of whole observations, of the clearest date where it has data. `"medoid"` is (the real observation closest to the multi-band median, so a composite's bands come from one date; with one band, the observation closest to the median). Only observations with every band are candidates, and on a tie the earliest wins. The medoid runs in Rust, every pixel in parallel: on 60 dates of 4 bands and 300 × 300 pixels, 0.15 s against 1.0 s for xarray. |
| `fill` | `str` | `None` | What to do with the periods without a value: `None` leaves them NaN; `"linear"` interpolates linearly in time between the values before and after (a series' ends stay NaN); `"whittaker"` fills the same gaps with a Whittaker curve through the values (`lmbda`) and keeps the values. |
| `max_gap` | `float` | `None` | With `fill`, the largest gap filled, in days between the value before and the one after; larger gaps stay NaN. `None` fills any. |
| `lmbda` | `float` | `10.0` | `fill="whittaker"`: the curve's smoothness, as in [`smooth`](preprocessing.md#smooth). |

</div>

```python
cube_16d = zeit.regularize_time_series(cube, freq="16D", method="medoid")
clear = zeit.regularize_time_series(cube, "MS", method="clearest", fill="linear", max_gap=60)
```

## Reading and writing rasters

### `load_raster` { .api }

<!-- sig: zeit.io.load_raster -->
```python
zeit.io.load_raster(
    source, *, dates=None, start_year=None, band=None, chunks=None,
    clip=None, masked="auto", pattern=None, date_format=None,
    recursive=False, like=None, crs=None, res=None, resampling="auto",
    validate=None, scale=False, qa=None, qa_rule=None, bands=None,
    window=None, pixel=None, extent=None, tap=False, nodata=None, dtype=None,
    cutline=None,
)
```

Reads a time series, or a single map, as one georeferenced cube: an `xarray.DataArray` with dims `(time, y, x)` (one index) or `(time, band, y, x)` (several spectral bands), a `datetime64` `time` coordinate, and its CRS, transform and NoData through `.rio`. The georeferencing and the dates travel with the data, so the algorithms and `save_raster` need nothing else. Also exported as `zeit.load_raster`.

It reads:

| Source | Example | Where the dates come from |
| :--- | :--- | :--- |
| A raster file with one band per date | `"LT_Stack_NDVI_Rondonia.tif"` | `dates`/`start_year`; the dates `save_raster` writes into the file; the band descriptions (`yr1985`, `1985`, `2020-01-15`, `20200115`, or `2020-01-15_red` for a time × band stack); a `<name>_dates.csv` next to it |
| A raster file with several bands per date, date after date | `"CCDC_Stack_Rondonia.tif"` (875 bands) | The band descriptions (`2020-01-15_red`); the dates `save_raster` writes into the file; a `<name>_dates.csv` next to it with one date for every few bands (125 dates: 7 bands each, named as the file names them or 1 to 7) |
| A folder, a glob pattern or a list of single-date rasters | `"tiles/*.tif"` | The first date in each file name, or `pattern=` |
| A Zarr store or a NetCDF file | `"cube.zarr"` | Its `time` coordinate |
| An `xarray.DataArray` or `Dataset` | the output of `build_time_series` | Its `time` coordinate (`lat`/`lon` are renamed `y`/`x`) |
| A numpy array `(time, y, x)` or `(time, band, y, x)` | | `dates`/`start_year`; `like=` georeferences it |

Without dates, the band axis keeps the name `band` and the algorithms ask for `years`/`dates`. A single band is returned as `(y, x)`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | path, list, `DataArray`, `Dataset` or `ndarray` | required | What to read (see the table above). |
| `dates` | list | `None` | Date of each time step: strings, datetimes, years (`1985`) or fractional years (`2020.5`). Overrides the dates found in the data. |
| `start_year` | `int` | `None` | Annual series: year of the first time step. |
| `band` | `int`, `str` or list | `None` | Bands to read: 1-based band numbers in a raster file, band or variable names in a cube or Dataset. |
| `chunks` | `"auto"`, `dict` | `None` | `None` loads the data into memory; anything else returns a lazy dask-backed cube, read block by block. |
| `clip` | bounds or geometries | `None` | Read only a region: `(xmin, ymin, xmax, ymax)` in the raster's CRS, or polygons (GeoDataFrame, GeoSeries, shapely). |
| `masked` | `"auto"`, `bool` | `"auto"` | `"auto"`: NoData becomes NaN in float rasters, while integer rasters (e.g. NDVI × 10000 in Int16) keep their values and type. `True`: NaN everywhere (integers become floats). `False`: values as stored. |
| `pattern` | `str` | `None` | Several files: regular expression with a named group `date` and optionally `band`, matched against each file name. |
| `date_format` | `str` | `None` | `strptime` format of the dates in file names or band descriptions (e.g. `"%Y%m%d"`). |
| `recursive` | `bool` | `False` | Folders: also search sub-folders. |
| `like` | path, `DataArray` or `Dataset` | `None` | A reference raster whose grid the result takes (CRS, cells, extent and `x`/`y` coordinates), so that the two line up cell by cell; see [On the grid of another raster](#on-the-grid-of-another-raster). numpy input: a raster on the same grid whose coordinates and CRS georeference the array. |
| `crs` | `str`, `int`, `CRS` | `None` | Without a reference raster: the CRS to reproject to, as `gdalwarp -t_srs`; see [In another CRS or at another resolution](#in-another-crs-or-at-another-resolution). |
| `res` | number or `(x, y)` | `None` | Without a reference raster: the cell size, in the units of `crs` (or of the data's CRS), as `gdalwarp -tr`. |
| `resampling` | `str` | `"auto"` | With `like`, `crs` or `res`: `"auto"` takes the nearest cell for integer rasters and interpolates floats bilinearly (QA bands always take the nearest cell); or a GDAL method: `"nearest"`, `"bilinear"`, `"cubic"`, `"cubic_spline"`, `"lanczos"`, `"average"`, `"mode"`, `"min"`, `"max"`, `"med"`, `"q1"`, `"q3"`, `"rms"`. |
| `validate` | `str` | `None` | `"landtrendr"`, `"ccdc"` or `"cold"`: warn when the series looks unfit for that algorithm (too few dates, values that do not look scaled). |
| `scale` | `bool` | `False` | Apply the `scale_factor` and `add_offset` each band stores (`value * scale + offset`, after NoData became NaN). Bands without them stay as they are (an Int16 file with no scale keeps its type); the files of a folder each use their own. Not for NetCDF/Zarr stores or `DataArray` sources, already decoded. Needs `masked` `"auto"` or `True`. Landsat files with a quality band (scales that differ by band) load without rioxarray's "Scales differ across bands" warning, which does not apply when Zeit reads the scales itself (`scale=True`) or only one `band` is read. |
| `qa` | band, path, glob, list or `DataArray` | `None` | A quality band: the cube comes back with NaN where the observation is not usable by `qa_rule`. A band number reads it from the same files, like `band=` (`qa=7`: the Fmask of Landsat files of a date); anything else is another source with the same dates and grid, read with the same `clip`, `window`/`pixel`, `like`... (nearest cell). Integer data becomes float. |
| `qa_rule` | `str` | `None` | With `qa`: `"fmask"`, `"landsat_c2"`, `"nonzero"`, `"scl"` or `"hls_fmask"`, see [`qa_mask`](#qa_mask). |
| `bands` | `dict` | `None` | Bands by role, `{"blue": 1, "red": 3, "nir": 4}`: name to band (as `band=`). Gives a `(time, band, y, x)` cube whose `band` coordinate holds the names, in the order of the dict. Not with `band`. |
| `window` | `(col0, row0, col1, row1)` | `None` | Read only a window of the raster's own pixels (the end is exclusive), no CRS needed. The cells keep their coordinates and transform. A file is read in the window only. Not with `clip`, `like`, `crs` or `res`. |
| `pixel` | `(col, row)` | `None` | Read one pixel, as a `(time,)` series (`(time, band)` with several bands) with scalar `x` and `y` coordinates. Only that pixel is read from each file, so it takes a fraction of a second in a stack of many gigabytes. Same restrictions as `window`. |
| `extent` | `(xmin, ymin, xmax, ymax)` | `None` | Without a reference raster: the extent of the grid, in the CRS of the result, as `gdalwarp -te`. With `res` the cells of that size are laid from its top-left corner; without it, the cells of the data (same CRS only), fitted to the extent as gdalwarp does (a whole number of them, a little larger or smaller). With another `crs`, `res` is needed. An extent a whole number of cells from the data's only crops it. Not with `cutline`. |
| `tap` | `bool` | `False` | `gdalwarp -tap`: the extent aligned to whole cells counted from the origin of the coordinates. Needs `res`. |
| `nodata` | `None`, `"source"` or number | `None` | What the cells no data covers hold, and the NoData of the result. `None`: the data's NoData, else NaN (floats) or the type's extreme (integers). `"source"`: as gdalwarp without `-dstnodata`: the data's NoData, and when it has none those cells are 0 and the result has no NoData. A number: `gdalwarp -dstnodata` (the data's NoData cells become it; a value equal to it moves one step). |
| `dtype` | numpy type | `None` | The type of the result, as `gdalwarp -ot`: the warp is done in the wider of the data's type and this one, then rounded (half away from zero) and clamped to it. |
| `cutline` | path, `(path, layer)`, `GeoDataFrame`, geometries | `None` | As `gdalwarp -cutline -crop_to_cutline`: the cells of the data outside the polygons take no part in the warp (the result there is `nodata`), and without `like` the grid is cropped to their extent (and, without `res`, to the cells of the data entirely inside it). Geometries without a CRS are in the result's. Not with `extent`. |

</div>

**Returns** `xarray.DataArray`.

```python
ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")      # bands yr1985 ... yr2024
ndvi.dims                                                 # ('time', 'y', 'x')
ndvi.time.dt.year.values[[0, -1]]                         # array([1985, 2024])

ndvi = zeit.load_raster("ndvi_stack.tif", start_year=1985, chunks="auto")   # lazy
s2 = zeit.load_raster("S2/", pattern=r"_(?P<date>\d{8})_(?P<band>B\d{2})\.tif$")   # (time, band, y, x)
```

#### Bands by role, scale, quality band and pixel windows

The options below replace the lines of xarray that a script would otherwise write after reading.

```python
# Landsat files with the bands of a date in each (blue ... swir2, Fmask), reflectance x 10000 with a GDAL scale
cube = zeit.load_raster("landsat/", bands={"blue": 1, "red": 3, "nir": 4}, scale=True,
                        qa=7, qa_rule="fmask")          # (time, band, y, x), NaN where Fmask refuses
ndvi = zeit.normalized_difference(cube, "nir", "red")    # (time, y, x)

stack = "LT_Stack_NDVI_Rondonia.tif"                     # 40 bands, 20 GB-class stacks work the same
series = zeit.load_raster(stack, pixel=(120, 80))        # (time,), reads that pixel only
block = zeit.load_raster(stack, window=(100, 60, 140, 100))   # (time, 40, 40)
```

`window` and `pixel` count the columns and rows of the file's own grid (column first), so they also work on series with no CRS, where `clip=` cannot. Quality bands take the pixel or window too, and `bands=` reads each band with the same options.

#### On the grid of another raster

`like=` puts the data on the grid of a reference raster, reprojecting and resampling it: Landsat at 30 m onto a Sentinel-2 grid at 10 m, a land-cover map onto the grid of a cube, or a folder of scenes from different orbits or UTM zones onto one grid.

```python
s2 = zeit.load_raster("S2_ndvi.tif")
landsat = zeit.load_raster("landsat_ndvi.tif", like=s2)              # bilinear for floats
classes = zeit.load_raster("mapbiomas_2020.tif", like=s2)           # nearest cell for integers
scenes = zeit.load_raster("scenes/", like="S2_ndvi.tif", chunks="auto")   # each file onto the grid
bool((landsat.x == s2.x).all())                                     # True: they line up
```

- **Every date on its own.** Each date (and band) is resampled with its own NoData, as GDAL warping that date alone: a cloud in one date never leaks into the interpolation of another. (GDAL's warp of a multiband raster takes a cell as valid when *any* band is, so a cloudy cell of one date enters the bilinear or average of that date and spreads its NoData.)
- **GDAL's resampling, faster.** The kernels are GDAL's (`gdalwarpkernel.cpp`, ported to Rust from landschaft's C++ port), run in parallel; the source coordinates of the cells are computed once for every date. On a cube of 240 dates of 1000 × 1000 cells (UTM to EPSG:3035), nearest takes 0.5 s, bilinear 0.7 s and average 0.9 s, against 2.4 s, 22.6 s and 6.4 s for `rioxarray`'s `reproject_match`.
- **The same values however it runs**: in memory, lazily (`chunks=`) or from a file, with any number of threads. A file is read only in the window the grid sees, and a lazy cube stays lazy.
- **No warp when it is not needed**: a grid with the same CRS and cells, a whole number of cells apart, is only a crop (and padding with NoData) of the data.
- Cells outside the data are NoData (NaN for floats; integer rasters keep their NoData value, or get the largest value of unsigned types and the smallest of signed ones).

#### The options of `gdalwarp`

`extent=`, `tap=`, `nodata=`, `dtype=` and `cutline=` add `gdalwarp`'s `-te`, `-tap`, `-dstnodata`, `-ot` and `-cutline` to `crs=`, `res=` and `like=`, with the same grid, type, NoData and values (nearest cell: the same cells; bilinear and cubic: the same within 1e-5 of the values; see below). The defaults are unchanged: `nodata=None` still gives NaN or the type's extreme outside the data, where gdalwarp writes 0; pass `nodata="source"` for gdalwarp's.

```python
# Janus' "Reproject, resample, clip" with aligned pixels, a NoData value and a cutline
zeit.save_raster(zeit.load_raster("ndvi.tif", crs="EPSG:4326", res=0.0003, tap=True, nodata=-9999,
                                  cutline="zones.gpkg", resampling="bilinear"), "ndvi_zone.tif")
```

Notes on how close it is to `gdalwarp`:

- A reprojection places the cells with the coordinates of gdalwarp's approximate transformer (`-et 0.125`, GDAL's `GDALApproxTransform`: exact at the ends and middle of a row of the result, a line between them where it is within the tolerance), so every method (the area ones from the corners of their cells, as GDAL approximates them) gives the same cells and values as `gdalwarp`'s default, bit for bit, for a result that gdalwarp warps in one piece (it splits one that needs more than 64 MB of working memory, or whose source window is mostly outside the source, and approximates each piece in its own rows; then a few cells, 0.01 % to 1 %, take a neighbouring cell).
- `cutline` is applied as gdalwarp does, to the cells of the source: those outside the polygons do not take part in the interpolation (cells whose nearest source cells are outside are NoData).
- `crs` with `extent` or `cutline` needs `res`; and `cutline` with `extent` is an error, because gdalwarp's `-crop_to_cutline` replaces `-te`. `tap` without a `extent`/`cutline` widens the extent of the data to whole cells from the origin of the coordinates, as gdalwarp does. Without `extent`, gdalwarp drops a border row or column of the result that falls entirely outside the source (GDAL 3.8 and later); so does Zeit.

#### In another CRS or at another resolution

Without a reference raster, `crs=` and `res=` make the grid `gdalwarp -t_srs crs -tr res` would: the extent that covers the data in that CRS, and either cells of `res` laid from its top-left corner or GDAL's suggested resolution. A folder's grid covers all of its files, at the finest of their resolutions, as gdalwarp does with several inputs. The warp is the same as with `like=`, so everything above applies.

```python
albers = zeit.load_raster("ndvi_stack.tif", crs="EPSG:5880", res=30)    # reprojected, 30 m cells
coarse = zeit.load_raster("ndvi_stack.tif", res=250, resampling="average")   # same CRS, 250 m
scenes = zeit.load_raster("scenes/", crs=32722, chunks="auto")          # two UTM zones onto one
```

The cells start at the corner of the data's extent, so two rasters loaded this way do not necessarily line up: to combine sources, load the first with `crs=`/`res=` and the others with `like=` it.

### `save_raster` { .api }

<!-- sig: zeit.io.save_raster -->
```python
zeit.io.save_raster(
    data, path, *, like=None, crs=None, transform=None, nodata=None,
    dtype=None, band_names=None, compress="deflate", driver=None,
    reference_cube=None, categories=None, colors=None,
)
```

Writes a map, a stack or a result so that GIS software and `load_raster` read it back as it was: georeferencing, NoData, band names and dates. Returns the path written. Also exported as `zeit.save_raster`.

- A `DataArray` brings its own georeferencing (`.rio`, or its `x`/`y` coordinates); a numpy or dask array takes it from `like=` (a raster or a path), or from `crs` and `transform`.
- A `time` axis becomes the band descriptions (`1985`, … for annual series, `2020-01-15`, … otherwise) and the `ZEIT_TIME` metadata tag, so `load_raster` gets the `time` coordinate back. A `(time, band, y, x)` cube is written as `2020-01-15_red`, `2020-01-15_nir`, …
- More leading dims become one band per combination, named by joining their labels: CCDC's `coefs (segment, band, coef, y, x)` is written as `1_blue_a0`, `1_blue_c1`, … Dates (`datetime64`, such as CCDC's `t_break`) are written as decimal years, NaN where there is none.
- A `Dataset` or a `dict` of maps (e.g. the result of `extract_events`, `bfast_monitor` or `mann_kendall`) becomes a folder with one GeoTIFF per variable when `path` has no extension, or one multi-band file named by variable when it ends in `.tif`. Variables with a leading dim get one band per label: each phenology metric `(year, y, x)` is written with bands `2010`, `2011`, …
- Floats with NaN get NaN as NoData (or `nodata`, which then replaces the NaN); `bool` is written as `uint8` and 64-bit integers as the smallest integer type that holds them.
- GeoTIFFs are compressed with the predictor that suits the type, tiled when larger than 256 × 256 and BigTIFF when needed; dask arrays are written block by block. The format follows the extension (`.tif`, `.img`, `.nc`, `.zarr`, …) or `driver` (`"COG"` for a Cloud-Optimised GeoTIFF). Missing folders are created.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset`, `dict`, numpy or dask array | required | What to write: `(y, x)`, `(band\|time, y, x)`, `(time, band, y, x)` or more leading dims. |
| `path` | `str` or `Path` | required | Output file, or folder for a `Dataset`/`dict`. Without an extension, `.tif` is added. |
| `like` | path or `DataArray` | `None` | A raster on the same grid whose CRS and transform georeference `data`. |
| `crs` | `str`, `int`, `CRS` | `None` | Coordinate reference system; overrides that of `data` and `like`. |
| `transform` | `Affine` | `None` | Affine transform; overrides that of `data` and `like`. |
| `nodata` | `float` | `None` | NoData value; overrides that of `data`. NaN in floats are written as this value. |
| `dtype` | `str` | `None` | Data type to write (default: the data's). Floats cast to integers get `nodata` (or the type's maximum) where they are NaN. |
| `band_names` | list of `str` | `None` | Band descriptions; by default the dates of a `time` axis or the labels of a `band` axis. |
| `compress` | `str` | `"deflate"` | GeoTIFF compression: `"deflate"`, `"lzw"`, `"zstd"`, `None`… |
| `driver` | `str` | `None` | GDAL driver when the extension is not enough; `"COG"` writes a Cloud-Optimised GeoTIFF. |
| `reference_cube` | path or `DataArray` | `None` | Deprecated name of `like`. |
| `categories` | `dict` | `None` | The classes of a classified map, `{value: name}`, written as the first band's category names (GDAL's `SetCategoryNames`; for a GeoTIFF, in the `.aux.xml` beside it), where QGIS and Janus read them. Values are non-negative integers; a value without a name gets an empty one. One map only, not a `dict` or `Dataset`. |
| `colors` | `dict` | `None` | The classes' colours, `{value: colour}`, with the colour as `"#RRGGBB"`, `"#RRGGBBAA"` or an `(r, g, b[, a])` tuple of 0-255, written as the first band's colour table. Needs a `uint8` or `uint16` map and a GeoTIFF or COG. A GeoTIFF palette keeps no per-class alpha. |

</div>

**Returns** `pathlib.Path`.

```python
ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
zeit.save_raster(ndvi.sel(time="2020"), "ndvi_2020.tif")       # georeferencing from the cube
zeit.save_raster(loss, "lt_rondonia")                          # Dataset of maps: one GeoTIFF each
zeit.save_raster(yod_array, "year_of_loss.tif", like=ndvi, nodata=0)     # numpy map
zeit.save_raster(lc, "lc.tif", nodata=0, categories={1: "forest", 2: "water"}, colors={1: "#2ca02c", 2: "#1f77b4"})
```

### `build_vrt` { .api }

<!-- sig: zeit._vrt.build_vrt -->
```python
zeit._vrt.build_vrt(
    sources, path, *, separate=False, extent=None, res=None,
    resampling="nearest", band_names=None, wavelengths=None,
)
```

Writes a VRT, a small XML file that GDAL, rasterio, QGIS and `load_raster` read as a raster, of tiles mosaicked and bands stacked, with no pixel copied. It is what `gdal.BuildVRT` writes (the same grid, the same sources' windows, the same values), without GDAL's Python bindings. Returns the path written. Also exported as `zeit.build_vrt`.

- `sources` is a list of files (the tiles of a scene, mosaicked; the bands of the files are kept), a list of files with `separate=True` (each file is one band, stacked), or a list of lists of files (one output band per inner list, made of the tiles of that band).
- Where tiles overlap the later file wins, except where it holds NoData. Pixels no tile covers are NoData, or 0 for a band without one. A band has a NoData value when every file of it has one. The files must share a CRS and be north-up.
- The source files are referred to by absolute path.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `sources` | list of paths | required | Files, or lists of files, as above. |
| `path` | `str` or `Path` | required | The `.vrt` to write. Missing folders are created. |
| `separate` | `bool` | `False` | A flat list of files: each is one band (its first), stacked. |
| `extent` | `(xmin, ymin, xmax, ymax)` | `None` | The VRT's extent; `None`: the union of the files. |
| `res` | `float` or `(xres, yres)` | `None` | The VRT's pixel; `None`: the finest pixel of the files. |
| `resampling` | `str` | `"nearest"` | How a source of another pixel size or alignment is read: `"nearest"`, `"bilinear"`, `"cubic"`, `"average"`, `"mode"`, … |
| `band_names` | list of `str` | `None` | The bands' descriptions; by default the first file's. |
| `wavelengths` | `dict` | `None` | `{band name: nm}`, written as the `wavelength` and `wavelength_units` metadata of the bands with that name. |

</div>

**Returns** `pathlib.Path`.

```python
vrt = zeit.build_vrt([["B04_a.tif", "B04_b.tif"], ["B08_a.tif", "B08_b.tif"]], "scene.vrt",
                     extent=(600000, 7396400, 606000, 7400000), res=30,
                     band_names=["B04", "B08"], wavelengths={"B04": 664.6, "B08": 832.8})
cube = zeit.load_raster(vrt)
```

### `get_georef` { .api }

<!-- sig: zeit.io.get_georef -->
```python
zeit.io.get_georef(reference_cube)
```

Extracts `{"crs": ..., "transform": ...}` from a rasterio dataset or an xarray object. Handy for `snic_to_polygons` or custom writers. Also exported as `zeit.get_georef`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `reference_cube` | rasterio dataset or `xr.DataArray` | required | Object to read the georeferencing from. |

</div>

## Quality bands to weights

Decoders written from the products' documentation (the MODIS MOD13 and MOD09 user guides, ESA's description of the Sentinel-2 SCL), with weights of Zeit's own. Each turns a sensor's QA band into per-observation reliability weights in `[0, 1]` for the `weights=` of [`zeit.smooth`](preprocessing.md#smooth) and [`zeit.phenology`](time-series.md#phenology). A QA cube (a `DataArray`, or a raster path) gives a weights cube with its dims, dates and georeferencing, lazy if it was, ready to pass along with the data it describes; the raster's NoData (and NaN) get the weight `0`. A numpy array gives a numpy array. All three are exported at the top level.

```python
cube = zeit.build_time_series(..., bands=["red", "nir", "scl"])
ndvi = zeit.compute_indices(cube, ["NDVI"]).sel(band="NDVI")
weights = zeit.qc_sentinel2_scl(cube.sel(band="scl"))     # (time, y, x), on the cube's grid
pheno = zeit.phenology(ndvi, weights=weights)
```

### `qa_mask` { .api }

<!-- sig: zeit.qc.qa_mask -->
```python
zeit.qc.qa_mask(qa, rule)
```

The observations a quality band keeps, as a boolean "usable" mask (the rules Janus applies to a quality band). A code counts when it is a number of 0 or more (fractions are cut); NaN, negative values and the raster's NoData are never usable. Exported as `zeit.qa_mask`; `load_raster(..., qa=, qa_rule=)` applies it while reading.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array, `DataArray` or path | required | The quality band (lazy if the `DataArray` is). |
| `rule` | `str` | required | `"fmask"`: 0 clear land and 1 water usable (2 shadow, 3 snow, 4 cloud, 255 fill masked). `"landsat_c2"`: QA_PIXEL bits 0 fill, 1 dilated cloud, 3 cloud, 4 shadow and 5 snow clear (`code & 0x3B == 0`). `"nonzero"`: any code but 0. `"scl"`: Sentinel-2 SCL 4 vegetation, 5 bare soil, 6 water. `"hls_fmask"`: HLS v2.0 Fmask bits 1 to 4 (cloud, adjacent, shadow, snow or ice) clear, and not 255. |

</div>

**Returns** a boolean array or `DataArray` with the input's dims and coordinates.

```python
usable = zeit.qa_mask(cube.sel(band="fmask"), "fmask")
clear = cube.sel(band="nir").where(usable)
```

### `qa_to_fmask` { .api }

<!-- sig: zeit.qc.qa_to_fmask -->
```python
zeit.qc.qa_to_fmask(qa, rule)
```

The Fmask codes of a quality band: `0` clear, `1` water, `2` shadow, `3` snow, `4` cloud, `255` no data, the codes the `qa=` of [`ccdc`](change-detection.md#ccdc) and [`sccd`](change-detection.md#sccd) read. Same rule names as [`qa_mask`](#qa_mask) (and `fmask <= 1` is exactly its mask). A code counts when it is a number of 0 or more (fractions are cut); NaN, negative values and the raster's NoData are `255`. Exported as `zeit.qa_to_fmask`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array, `DataArray` or path | required | The quality band (lazy if the `DataArray` is). |
| `rule` | `str` | required | `"fmask"`: the codes as they are (anything but 0-4 and 255 becomes 255). `"landsat_c2"`: QA_PIXEL bit 0 (fill) 255; bits 1 and 3 (dilated cloud, cloud) 4; bit 4 (shadow) 2; bit 5 (snow) 3; bit 7 (water) 1; else 0 (the first that applies, in this order). `"scl"`: Sentinel-2 SCL 4 and 5 are 0, 6 (water) 1, 2 and 3 (shadows) 2, 11 (snow or ice) 3, 7 to 10 (unclassified, clouds, cirrus) 4, the rest 255. `"hls_fmask"`: 255 is 255; bits 1 and 2 (cloud, adjacent) 4; bit 3 (shadow) 2; bit 4 (snow or ice) 3; bit 5 (water) 1; else 0. `"nonzero"`: 0 is 4, any other code 0. |

</div>

**Returns** a `uint8` array or `DataArray` (named `fmask`) with the input's dims and coordinates.

```python
qa = zeit.load_raster("landsat/", band=7, masked=True, pixel=(896, 186))   # QA_PIXEL
segments = zeit.ccdc(bands_cube, qa=zeit.qa_to_fmask(qa, "landsat_c2"), scale="auto")
```

### `qc_sentinel2_scl` { .api }

<!-- sig: zeit.qc.qc_sentinel2_scl -->
```python
zeit.qc.qc_sentinel2_scl(scl, wmin=0.2, wmid=0.5, wmax=1.0)
```

Sentinel-2 L2A Scene Classification Layer: vegetation and bare soil get `wmax`; water and unclassified get `wmid`; shadows, medium- and high-probability cloud, thin cirrus and snow get `wmin`; no data, saturated or defective pixels and codes outside 0-11 get `0`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `scl` | array, `DataArray` or path | required | SCL values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>

### `qc_modis_summary` { .api }

<!-- sig: zeit.qc.qc_modis_summary -->
```python
zeit.qc.qc_modis_summary(qa, wmin=0.2, wmid=0.5, wmax=1.0)
```

MOD13 "SummaryQA" / pixel reliability: `0` good → `wmax`, `1` marginal → `wmid`, `2` snow and `3` cloudy → `wmin`, fill → `0`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array, `DataArray` or path | required | SummaryQA values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>

```python
from zeit.qc import qc_modis_summary

weights = qc_modis_summary(qa_cube)          # a DataArray like qa_cube
```

### `qc_modis_state` { .api }

<!-- sig: zeit.qc.qc_modis_state -->
```python
zeit.qc.qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
```

MOD09 500 m 16-bit "State QA": decodes cloud state, cloud shadow, aerosol quantity and snow/ice bits.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `qa` | array, `DataArray` or path | required | State QA values, any shape. |
| `wmin`, `wmid`, `wmax` | `float` | `0.2`, `0.5`, `1.0` | Weights for bad, doubtful and good observations. |

</div>
