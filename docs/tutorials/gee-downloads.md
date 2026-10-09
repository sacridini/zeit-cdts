# Google Earth Engine

<p class="lead">Let Google's servers do the heavy preparation (Landsat sensor harmonisation, cloud masking and compositing) and download a ready-to-use time series as GeoTIFFs. This is the quickest way to get the annual stacks LandTrendr needs.</p>

<div class="glance" markdown>
<div><span class="k">Does</span><span class="v">Harmonise Landsat 5/7/8/9, mask clouds, composite, download</span></div>
<div><span class="k">Area</span><span class="v">WRS-2 path/row, Sentinel-2 tile, bbox, vector or raster file</span></div>
<div><span class="k">Output</span><span class="v">One GeoTIFF per composite, on a shared pixel grid</span></div>
<div><span class="k">Needs</span><span class="v">An Earth Engine account and a Google Cloud project</span></div>
</div>

## Authentication

Earth Engine requires the user to authenticate their machine with Google Cloud Project (GCP) credentials that have API access enabled. Every time you attempt a download, the `download_gee_timeseries` function handles this initialization. If your credentials expire or do not exist, a browser window will open prompting you to log in.

!!! note
    Make sure the Google account has access to Earth Engine, and pass your Google Cloud project with `project="your-project"`.

## Choosing the area (`roi`)

You never need to build Earth Engine objects yourself. The `roi` argument accepts local, offline inputs, and `zeit` converts them internally:

```python
roi = "217/076"                              # a Landsat WRS-2 path/row: the tile's footprint
roi = "23KPQ"                                # a Sentinel-2 (MGRS) tile id, computed offline ('T23KPQ' also works)
roi = [-43.6, -23.1, -43.1, -22.6]           # a bounding box [min_lon, min_lat, max_lon, max_lat]
roi = "data/study_area.shp"                  # a vector file: Shapefile, GeoPackage, GeoJSON, KML...
roi = "data/reference_image.tif"             # a raster file: its extent
roi = geopandas.read_file("area.gpkg")       # an in-memory GeoDataFrame (or a shapely geometry)
```

Files in any CRS are reprojected to lon/lat. For vector and raster inputs, the download covers their **bounding box**. To keep only the pixels inside a polygon, mask the result locally afterwards, for example with `rasterio.mask`.

**Sentinel-2 tiles** follow the MGRS grid: 109.8 km × 109.8 km squares in the tile's UTM zone, overlapping their neighbours by 9.8 km. `zeit` computes the footprint from the id alone, with no Earth Engine lookup. The calculation matches real Sentinel-2 scene footprints to within about 50 m. Tiles that straddle the 180° meridian can't be expressed as a lon/lat box and raise an error; pass a bbox for one side instead.

**Landsat WRS-2 path/rows** are looked up from a Landsat Collection 2 scene on that path/row, which takes one quick Earth Engine query.

!!! note "What about Sentinel-1?"
    Sentinel-1 has no fixed tiling grid equivalent to WRS-2 or MGRS. Its *relative orbit* (1–175) and pass direction (ascending/descending) identify a ground track, not an area. The GRD *slices* in Earth Engine are not cut at the same places from one acquisition to the next. ESA's stable *burst IDs* exist only for SLC products, which Earth Engine doesn't provide. For a Sentinel-1 area, use a Sentinel-2 tile id, a bounding box or a vector file.

## Downloading: `method="auto"` (default)

For most areas you don't need to choose a method. With the default `method='auto'`, `zeit` plans each image before downloading it and picks the fastest route that will work:

- **Direct tiled download** for images up to `max_direct_mb` (4 GB raw by default). `zeit` fixes a single pixel grid for the whole region, splits it into tiles sized in *bytes* rather than degrees, and fetches them concurrently with `ee.data.computePixels`. Each tile is written straight into its place in the output GeoTIFF, so tiles line up exactly with no seams and there is no mosaicking step. Tiles are split in space first; deep stacks such as a `'dense'` time series with hundreds of bands are also split by band, so they stay under Earth Engine's per-request limits (32 MB, 1024 bands).
- **Google Drive export** (see [Large areas](#large-areas-export-to-google-drive)) for images above that size, or when a tile hits an Earth Engine *interactive* compute limit (user memory limit, computation timeout), which retrying the same request cannot fix.

Concurrency adapts to your account. Earth Engine limits concurrent interactive requests per account (about 40 on a standard tier, only 2–3 for a project in noncommercial *Restricted Mode*). `zeit` starts with 4 concurrent requests, ramps up while requests succeed, and halves on every `HTTP 429`, so it settles just under whatever limit your account actually has.

```python
from zeit.gee import download_gee_timeseries

# Bounding box [min_lon, min_lat, max_lon, max_lat]
my_roi = [-47.95, -15.85, -47.85, -15.75]

download_gee_timeseries(
    roi=my_roi,
    start_date='2010-01-01',
    end_date='2020-12-31',
    out_dir='./gee_direct_data',
    method='auto',             # default: direct tiled download, Drive export when needed
    composite_type='annual',   # Generates LandTrendr-style Annual Medoid Composites
    project='my-gcp-project'   # Replace with your Google Cloud Project ID
)
```

To download a single `ee.Image` you built yourself, use `download_gee_image` directly. It accepts the same `method` and exposes the tuning knobs:

```python
from zeit.gee.downloader import download_gee_image

download_gee_image(
    image, roi, 'out.tif',
    scale=30,
    crs='EPSG:4326',           # or a projected CRS such as 'EPSG:32723'
    sub_tile_workers=16,       # upper bound on concurrent requests (adapts downward on 429)
    max_tile_mb=None,          # cap per tile request (default 32 MB); tiles are sized on the fly
)
```

Masked pixels are written with the GeoTIFF nodata value Earth Engine uses for the output type (`-inf` for float, the type minimum for signed integers, `0` for unsigned). You can force either route with `method='direct'` or `method='drive'`.

## Indices and spectral temporal metrics

Earth Engine computes whatever you ask on its own servers, so you can download indices instead of bands, and per-year statistics instead of every observation.

**Indices instead of bands.** Pass index names in `indices`: `NDVI`, `EVI`, `SAVI`, `kNDVI`, `NBR`, `NDMI`, `NDWI` and `MNDWI`, alone or mixed with SR bands. They are computed on surface reflectance (the Collection 2 digital numbers scaled by 2.75e-05 and offset by −0.2), on the medoid for `'annual'` and on every observation for `'dense'`. One index is one sixth of the six-band download.

**Spectral temporal metrics** (`composite_type='stm'`) summarize a whole year in a few bands per index: each index is computed on every clear observation, then reduced over the year on Earth Engine with one combined reducer. Choose the statistics with `metrics`: `median`, `mean`, `std`, `min`, `max`, `iqr` (p75 − p25), `count` (clear observations) or any percentile such as `p10`.

```python
from zeit.gee import download_gee_timeseries

download_gee_timeseries(
    roi="23KPQ",
    start_date="2015-01-01",
    end_date="2024-12-31",
    out_dir="./stm",
    composite_type="stm",
    indices=["NDVI", "NBR", "SR_B6"],           # indices and SR bands
    metrics=["median", "p10", "p90", "iqr"],
    project="my-gcp-project",
)
# -> ./stm/landsat_stm_2015.tif ... landsat_stm_2024.tif,
#    12 bands each: NDVI_median, NDVI_p10, NDVI_p90, NDVI_iqr, NBR_median, ...
```

Each year is the calendar year inside `start_date`–`end_date`. To summarize one season, run one call per year with that season's dates. The band names and order match `zeit.build_spectral_temporal_metrics` (the STAC version), so models trained on one source can be applied to the other. Both take what to summarize in `indices`. Earth Engine's percentile reducer is not guaranteed to interpolate exactly like NumPy, so percentiles can differ slightly from the STAC version.

To build metrics on a collection of your own, use `zeit.gee.compute_stm(collection, indices, metrics)`. It returns the `ee.Image` to pass to `download_gee_image`.

The `indices` argument was called `bands` up to zeit 0.25. `bands=` still works, with a deprecation warning.

!!! note "Indices from zeit 0.25 and earlier"
    zeit 0.25 and earlier computed indices on the raw Collection 2 digital numbers, which carry a large offset (reflectance 0 is DN 7273). That compressed normalized differences towards zero (an NDVI of 0.71 came out as 0.33) and made EVI meaningless. Download index composites made with those versions again.

## Large areas: export to Google Drive

For state-level or national-scale analyses, downloading data directly over the internet in real-time might fail due to API payload limits or simply take too long.

In these cases, pass `method='drive'`. The `zeit` package will set up everything and dispatch a Task directly to Google's servers. Google will silently process and save the final file in the cloud inside your **Google Drive**, under the `Zeit_Downloads` folder.

```python
from zeit.gee import download_gee_timeseries

# Example: Bounding box of a larger region
state_roi = [-53.11, -25.31, -44.15, -19.78]

download_gee_timeseries(
    roi=state_roi, 
    start_date='1985-01-01',
    end_date='2022-12-31', 
    out_dir='./data',  # Used only to name the files in Drive for this method
    method='drive',    # Initiates asynchronous export
    composite_type='annual',
    project='my-gcp-project'
)

# The terminal will display a message similar to:
# [landsat_medoid_1985] Task sent to Google Drive (Task ID: ABCD123456).
```

## Worked example: a full Landsat tile, 1985–2025, ready for LandTrendr

This is the complete path from "I have a WRS-2 path/row" to LandTrendr disturbance maps, all through `zeit`. It uses no Earth Engine objects in your code and needs no local satellite data.

LandTrendr works on **one spectral index per year**. NBR is the standard choice for forest loss. The pipeline has three steps: download the annual composites as NBR, read the yearly files as one cube, then run LandTrendr out-of-core and write the event maps.

```python
import zeit
from zeit.gee import download_gee_timeseries

# 1. Annual harmonized Landsat 5/7/8/9 medoid composites -> NBR, one file per year
download_gee_timeseries(
    roi="217/076",                   # WRS-2 path/row (Rio de Janeiro); or a bbox / .shp / .gpkg / .tif
    start_date="1985-01-01",
    end_date="2025-12-31",
    out_dir="./lt_rj/annual",
    composite_type="annual",
    indices=["NBR"],                 # index computed on the medoid, which is selected on all 6 SR bands
    project="my-gcp-project",
)
# -> ./lt_rj/annual/landsat_medoid_1985.tif ... landsat_medoid_2025.tif

# 2. The yearly files as one (time, y, x) cube: the years come from the file names,
#    masked pixels (-inf) become NaN, and chunks= keeps it lazy (the tile is ~10 GB)
nbr = zeit.load_raster("./lt_rj/annual", chunks="auto")

# 3. LandTrendr and the event maps, computed block by block while they are written
lt = zeit.landtrendr(nbr, direction="loss", max_segments=6)   # an NBR drop is a disturbance
loss = zeit.extract_events(lt)
zeit.save_raster(loss, "./lt_rj/results")                     # yod.tif, magnitude.tif, ...
```

!!! warning "Three details that matter for LandTrendr"
    1. **`-inf` → `NaN`.** Downloaded GeoTIFFs mark masked pixels (clouds, no scene that year) with the nodata value `-inf`. `load_raster` reads them as `NaN`, which LandTrendr treats as a missing year. If you read the files another way, convert them yourself.
    2. **`nodata`.** For float data like this NBR, `zeit.landtrendr`'s default (`nodata="auto"`) treats only NaN and the raster's NoData as missing, so pixels whose NBR is exactly 0 are kept. For integer stacks without a NoData value, `0` is treated as missing; pass `nodata=None` if 0 is a valid value there.
    3. **2012 has gaps.** Landsat 5 stopped in November 2011 and Landsat 8 started in April 2013, so 2012 relies on Landsat 7 SLC-off alone and has striping gaps. LandTrendr tolerates missing years (`min_observations_needed=6` by default), so this is expected, not an error.

**Resuming.** A download never leaves a partial file at its final path, so re-running the same call is safe. For fine-grained control, loop over the years yourself with `download_gee_image` and skip years whose file already exists. The composites come from `zeit.gee.composites.create_annual_medoid` on the collection from `zeit.gee.harmonization.get_harmonized_collection`.

**Tile extent.** For `roi="217/076"`, `zeit` looks up the tile's footprint from a Landsat Collection 2 scene and downloads its bounding box, about 2.2° × 2.0° (~8,070 × 7,330 px at 30 m). Scenes from neighbouring paths/rows that overlap the box are also used, which fills the box corners and adds observations in the overlap zones.

## How long will a download take?

Time is dominated by Earth Engine's server-side compute, not by your connection or CPU. Two factors decide it: how many pixels you request, and **how many concurrent requests your account is allowed**.

| Account state | Concurrent interactive requests |
|:---|:---:|
| Standard tier (paid or noncommercial within quota) | about 40 |
| Noncommercial project in **Restricted Mode** (quota exceeded) | about 2 |

`zeit` discovers the limit on its own. It starts with 4 concurrent requests, ramps up to `sub_tile_workers` while requests succeed, and backs off on every `HTTP 429`. You'll see `EE throttled concurrency: finished at 2/16 concurrent requests` when the account is limited. When Earth Engine initializes, it prints a warning if your project is in Restricted Mode.

Times below are for annual medoid composites on WRS-2 217/076 (year 2020). The rows marked *measured* were timed on a Restricted Mode account; everything else is extrapolated from them. Years before 1999 have only Landsat 5 and compute faster, so treat totals as orders of magnitude:

| Area | Content | Restricted Mode (~2 concurrent) | Standard tier (estimated, 16 concurrent) |
|:---|:---|:---:|:---:|
| 0.5° × 0.5° (~3.5 M px) | 1 index (NBR) | ~27 s / year *(measured)* → **~20 min** for 41 years | ~3–5 min |
| 0.5° × 0.5° | 6 SR bands | ~77 s / year *(measured)* → **~50 min** | ~8–15 min |
| Sentinel-2 tile `23KPQ` (~15 M px) | 1 index (NBR) | ~2.5 min / year *(measured)* → **~1.5 h** | ~15–25 min |
| Full WRS-2 tile `217/076` (~59 M px) | 1 index (NBR) | ~13 min / year *(measured)* → **~6–9 h** | ~1–2 h |
| Full WRS-2 tile | 6 SR bands | ~35 min / year → **~1 day** | ~3–6 h |

The standard-tier column assumes Earth Engine throughput scales 4–8× from 2 to 16 concurrent requests. It was not measured.

!!! tip "Making large downloads faster"
    - **Download the index, not the bands.** A single NBR band is ~3× faster than 6 SR bands, and it is all LandTrendr needs.
    - **Standard-tier accounts can raise concurrency.** Call `download_gee_image` directly with `sub_tile_workers=32` (up to ~40). `download_gee_timeseries` uses the default of 16.
    - **Don't run several downloads in parallel.** They all share the account-wide concurrency limit and just throttle each other.
    - **For a full tile on a restricted account, consider the Drive export.** `download_gee_timeseries_drive` submits all years at once as batch tasks, which Earth Engine computes in parallel server-side. Turnaround then depends on the batch queue, from tens of minutes to hours.

Once downloaded, the local part is fast. LandTrendr on a full tile with 41 years runs in about 1.6 minutes on a 20-thread desktop (see [Performance & Scaling](../benchmarks/performance.md)).

## Output format

Each image arrives as **one GeoTIFF**. The tiles it is fetched in are written directly into their place in that file, so there is nothing to mosaic afterwards. `download_gee_timeseries` writes one such file per year. Stacking the years into a single multi-band file for LandTrendr is the one manual step (see the [worked example](#worked-example-a-full-landsat-tile-19852025-ready-for-landtrendr)).

Every file written by the direct download has:

- **The same pixel grid for every year.** The grid is fixed from the ROI and snapped to multiples of the resolution, so the annual files stack without resampling. The default CRS is EPSG:4326 with 30 m converted to degrees at the equator, as Earth Engine does. Pass `crs="EPSG:32723"` (for example) to `download_gee_image` for a metric grid.
- **Named bands** (`SR_B2` … `SR_B7`, or the index names) stored as band descriptions.
- **Nodata** equal to the value Earth Engine uses for masked pixels: `-inf` for float, the type minimum for signed integers, `0` for unsigned.
- A tiled, LZW-compressed GeoTIFF (BigTIFF when needed), written atomically: it appears at its final path only once every tile has arrived.

## What happens on the server

When using `composite_type='annual'` (the current `zeit` default for LandTrendr integration):

1. **Sensor Fusion:** The function fetches Landsat 5, 7, 8, and 9 collections (Surface Reflectance Collection 2).
2. **Harmonization:** Values from Landsat 8 and 9 (OLI) are mathematically converted to their ETM+ equivalents using coefficients from Roy et al. (2016) (see [References](#references)). This ensures a perfect time series, free from sensor biases.
3. **Cloud Masking:** The `QA_PIXEL` quality assurance band is used to filter out dense clouds and shadows across all images.
4. **Medoid Compositing:** Instead of a simple median, we apply the Medoid geometric strategy to find the actual real-world pixel that best represents the season (a standard approach in *eMapR/LandTrendr* workflows).
5. **Download planning:** The ROI's bounding box gets one pixel grid, and one request asks for the band names and data types. The image is then fetched as tiles with `ee.data.computePixels`, each sized on the fly as remaining work divided by the concurrency Earth Engine currently allows (between 4 and 32 MB). Tiles are split in space first, so a composite is never recomputed per band. Only deep stacks such as a `'dense'` series are also split by band, to respect the 1024-band request limit.
6. **Automatic fallback:** With `method='auto'`, images larger than `max_direct_mb` (4 GB raw) go to a Drive export. So does any image whose tiles hit an Earth Engine *interactive* compute limit (user memory limit, computation timeout), which retrying the same request cannot fix.

## References

- Roy, D. P., Kovalskyy, V., Zhang, H. K., Vermote, E. F., Yan, L., Kumar, S. S., & Egorov, A. (2016). Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity. **Remote Sensing of Environment**, 185, 57–70. [https://doi.org/10.1016/j.rse.2015.12.024](https://doi.org/10.1016/j.rse.2015.12.024)
