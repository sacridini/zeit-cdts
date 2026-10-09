# Embeddings

<p class="lead">The yearly embeddings of Earth observation foundation models, TESSERA and Google's AlphaEarth, as zeit cubes: one vector per 10 m pixel and year that classification, sampling, segmentation and clustering take as they are; plus similarity search and change from year to year.</p>

Both products summarise a whole year of imagery in a vector per pixel, made by a model trained on a great many pixels; what they are made for, in both papers, is mapping from few labelled samples with simple classifiers. zeit reads them as a `(time, band, y, x)` cube, `band` the dimensions (`A00`, `A01`...), and records in its attributes which embeddings it holds; [`save_raster`](data.md#save_raster) keeps that, [`train_classifier`](post-processing.md#train_classifier) and `zeit.ai.train` record it in the model, and [`classify`](post-processing.md#classify) and `zeit.ai.predict` refuse a cube of other embeddings (another product, version or variant is another space). Tutorial: [Embeddings (TESSERA, AlphaEarth)](../tutorials/embeddings.md).

| | TESSERA | AlphaEarth Foundations |
| :--- | :--- | :--- |
| Dimensions | 128 | 64 (each embedding has length 1) |
| Cells, years | 10 m, 2017–2025 (v1.1) | 10 m, 2017–2025 |
| Made from | Sentinel-1 and Sentinel-2 | Sentinel-1, Sentinel-2, Landsat and more |
| Read from | the TESSERA Zarr store, through `geotessera` (`pip install zeit-cdts[tessera]`, Python 3.12+) | open COGs on Source Cooperative (no account), or Earth Engine |
| Licence | CC0 1.0 | CC-BY 4.0: "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind." |
| Cite | Feng et al. (2025), arXiv:2506.20380 | Brown et al. (2025), arXiv:2507.22291 |

!!! note "What not to run on embeddings"
    An embedding has no physical unit and no seasonal signal: the algorithms that fit one (LandTrendr, CCDC, BFAST, Mann-Kendall, phenology, TWDTW, smoothing, Tmask, indices, unmixing, CODED, `regularize_time_series`) refuse a cube of embeddings with a message saying what to use instead.

## Reading

### `load_embeddings` { .api }

<!-- sig: zeit.load_embeddings -->
```python
zeit.load_embeddings(
    region=None, source, years=None, like=None, crs=None, res=None,
    resampling="auto", version=None, variant=None, depth=None,
    backend=None, store=None, chunks="auto", cache_dir=None,
)
```

Reads the embeddings of a product over a region as a georeferenced cube, lazily by default (one year and every dimension of 512 x 512 cells for TESSERA, 1024 x 1024 for AlphaEarth, on the products' own blocks). A region within one UTM zone keeps the zone's CRS and the cells the embeddings were made on, so the two products line up cell by cell; south of the equator that is the southern UTM CRS (TESSERA stores it in the northern one, with negative northings: the same cells). Also exported as `zeit.load_embeddings`; CLI: `zeit embeddings`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `region` | bounds, `GeoDataFrame`, path or geometries | `None` | `(west, south, east, north)` in longitude and latitude, a `GeoDataFrame`/`GeoSeries` or vector file in any CRS, or shapely geometries in longitude and latitude. Cells outside polygons are NaN; points (e.g. training samples) read the box around them. Optional with `like`. |
| `source` | `str` | required | `"tessera"` or `"alphaearth"`. |
| `years` | `int`, list or range | `None` | The years (default: all the product has). |
| `like` | path, `DataArray` or `Dataset` | `None` | A raster whose grid the result takes (its extent, if `region` is not given), as in `load_raster`. |
| `crs`, `res` | | `None` | Without `like`: CRS and cell size of the result (default 10 m). Needed for a region across UTM zones. |
| `resampling` | `str` | `"auto"` | With `like`, `crs` or `res`: `"auto"` takes the nearest cell (interpolated embeddings are vectors the model never made), and the mean of the cells (`"average"`) when the target cells are at least 1.5 times larger. Or any method of `load_raster`. |
| `version`, `variant` | `str` | `None` | TESSERA: dataset version (default `"v1.1"`, the complete global run) and variant (default: the version's). |
| `depth` | `int` | `None` | TESSERA v2: only the first `depth` dimensions (a Matryoshka prefix, published as an array of its own). |
| `backend` | `str` | `None` | AlphaEarth: `"source.coop"` (default, open COGs) or `"gee"` (the `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` collection, with an Earth Engine session up, downloaded into `cache_dir`). |
| `store` | URL, path or zarr store | `None` | A copy of the product to read: TESSERA, a Zarr store (read without `geotessera` when local); AlphaEarth, a folder or URL with `aef_index.parquet` and the COGs in `<year>/<zone>/`. |
| `chunks` | `"auto"`, `None` or `dict` | `"auto"` | `None` reads it all into memory. |
| `cache_dir` | path | `None` | Where downloads are kept: the AlphaEarth index (78 MB, refreshed monthly) and Earth Engine downloads. Default `~/.cache/zeit/embeddings`. |

</div>

**Returns** an `xarray.DataArray` `(time, band, y, x)` float32 named `embeddings`, NaN where there is no embedding, with attributes `embedding_source`, `embedding_version`, `embedding_variant` (what a model checks), `embedding_model`, `embedding_dimensions`, `embedding_license` and the citation (and attribution, for AlphaEarth).

```python
import zeit

emb = zeit.load_embeddings("aoi.gpkg", source="tessera", years=range(2018, 2025))
aef = zeit.load_embeddings((-63.0, -10.0, -62.9, -9.9), source="alphaearth", years=2024)
on_landsat = zeit.load_embeddings(source="alphaearth", like=landsat_cube, years=2020)   # 30 m, averaged
```

Training points read the whole blocks they fall on. When the same region is classified next, `.persist()` the cube (or read it with `chunks=None`, or save it) so that it is read once.

## Similarity

### `similarity` { .api }

<!-- sig: zeit.similarity -->
```python
zeit.similarity(emb, ref, by=None, year=None, metric="cosine")
```

How much each pixel looks like a reference: "find more places like these". The reference is the mean embedding of some samples (every cell of a polygon) in one year, compared with every year of the cube, which shows where it appears over time. Also exported as `zeit.similarity`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `emb` | `DataArray` | required | Embeddings `(time, band, y, x)` or `(band, y, x)`. |
| `ref` | `GeoDataFrame`, path or sequence | required | Points or polygons (any CRS), or one embedding (as many values as `band`). |
| `by` | `str` | `None` | A column of `ref`: one reference, and one map, per value (e.g. each class). |
| `year` | `int` | `None` | The year the samples' embeddings are taken from (default: the last). |
| `metric` | `str` | `"cosine"` | `"cosine"`: cosine similarity (1 for the same direction); `"euclidean"`: the distance. |

</div>

**Returns** an `xarray.DataArray` `(time, y, x)` float32 (with a `class` dim first with `by`), lazy when the cube is.

```python
mines = zeit.similarity(emb, "known_mines.gpkg", year=2024)
zeit.plot(mines)                                            # where it looks like a mine, year by year
```

## Change

### `embedding_change` { .api }

<!-- sig: zeit.embedding_change -->
```python
zeit.embedding_change(emb, metric="cosine", baseline="previous")
```

How far each pixel's embedding moved from one year to another. [`extract_events`](change-detection.md#extract_events) turns it into one event per pixel in the schema of the other change algorithms, so [`agreement`](change-detection.md#agreement) compares it with LandTrendr, CCDC, BFAST or CODED and the [validation](validation.md) functions take it as it is. Also exported as `zeit.embedding_change`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `emb` | `DataArray` | required | Embeddings `(time, band, y, x)` with at least two years. |
| `metric` | `str` | `"cosine"` | `"cosine"`: `1 - cos` of the angle between the embeddings (0 the same direction, 2 opposite); or `"euclidean"`. |
| `baseline` | `str` or `int` | `"previous"` | `"previous"`: each year against the year before it; `"first"`: against the first; or a year to compare every other year with. |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Meaning |
| :--- | :--- |
| `distance` | `(time, y, x)`: the distance of each year to its baseline. |
| `noise` | `(y, x)`: the median of the pixel's distances, its usual movement from year to year. |

With `baseline="previous"`, `extract_events` gives `magnitude` (the distance), `dsnr` (the distance over the noise), `yod` (the year before the first embedding that shows the change) and `date` (January 1 of that one); events have no direction (`event_type="any"`). An embedding summarises a year, so a change in the middle of one shows partly in it and partly in the next: compare with other algorithms with `tolerance=1`.

Each product has its own scale of distances: year to year, TESSERA's pixels move a cosine distance of about 0.05–0.1 and AlphaEarth's about 0.02, so a threshold for one is not one for the other. `min_magnitude` (e.g. 0.3 for TESSERA, 0.2 for AlphaEarth, for clearing of forest) or `sort_by="dsnr"` with a threshold on `dsnr` keep the real changes.

```python
change = zeit.embedding_change(emb)
events = zeit.extract_events(change, min_magnitude=0.3)
zeit.agreement(events, zeit.extract_events(zeit.landtrendr(ndvi)), tolerance=1)
```
