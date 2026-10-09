# Embeddings (TESSERA, AlphaEarth)

<p class="lead">Foundation models for Earth observation turn a whole year of satellite images into one vector per pixel: an embedding. Two products are global, open and made at 10 m every year since 2017, TESSERA and Google's AlphaEarth Foundations. zeit reads them as a cube like any other, so the classifiers, the clustering, the segmentation and the validation of zeit work on them as they are, and adds what is particular to embeddings: looking at them, searching for places like a sample, and following each pixel's embedding from year to year.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">What is where, from few samples? Where else looks like this? Where did the land change, year to year?</span></div>
<div><span class="k">Input</span><span class="v">A region and years; the embeddings are downloaded</span></div>
<div><span class="k">Output</span><span class="v">A <code>(time, band, y, x)</code> cube of 128 (TESSERA) or 64 (AlphaEarth) dimensions</span></div>
<div><span class="k">Reference</span><span class="v">Feng et al. (2025), arXiv:2506.20380; Brown et al. (2025), arXiv:2507.22291</span></div>
</div>

## Two products, one cube

| | TESSERA | AlphaEarth Foundations |
| :--- | :--- | :--- |
| Dimensions | 128 | 64 (each embedding has length 1) |
| Cells, years | 10 m, 2017–2025 | 10 m, 2017–2025 |
| Made from | Sentinel-1 and Sentinel-2 | Sentinel-1, Sentinel-2, Landsat and more |
| Stored as | Zarr, int8 times a scale per pixel, a group per UTM zone | COGs, int8 on a fixed curve, per year and UTM zone |
| Read through | `geotessera`, installed with zeit (Python 3.12+) | nothing: free open COGs on Source Cooperative (default); or Earth Engine |
| Licence | CC0 1.0 (attribution requested) | CC-BY 4.0, attribution required |

Both are read with one function, and give the same thing: a georeferenced `(time, band, y, x)` cube, `time` on January 1 of each year, `band` the dimensions (`A00`, `A01`...), NaN where there is no embedding. A region within a UTM zone keeps the zone's CRS and the cells the embeddings were made on, so the two products line up cell by cell:

```python
import zeit

emb = zeit.load_embeddings((-62.95, -9.95, -62.90, -9.90), source="tessera", years=range(2017, 2026))
aef = zeit.load_embeddings((-62.95, -9.95, -62.90, -9.90), source="alphaearth", years=range(2017, 2026))
emb.rio.transform() == aef.rio.transform()      # True: the same 554 x 549 cells of 10 m, EPSG:32720
```

The cube is lazy: the blocks are read when computed, a year and every dimension at a time, and dequantised as they arrive (over this 5 x 5 km box, all nine years took 43 s for TESSERA and 70 s for AlphaEarth from a laptop in Europe). `chunks=None` reads it all at once; `zeit.save_raster(emb, "emb.tif")` keeps a copy that `zeit.load_raster` reads back, still knowing what it holds.

Regions come as bounds in longitude and latitude, a vector file or a `GeoDataFrame` (cells outside its polygons become NaN), or the grid of another raster:

```python
emb = zeit.load_embeddings("municipality.gpkg", source="tessera", years=2024)
on_landsat = zeit.load_embeddings(source="alphaearth", like=landsat, years=range(2018, 2025))   # 30 m, averaged
```

With `like=` (or `crs=`/`res=`) the embeddings are put on that grid: by the nearest cell, or by the mean of the cells when the new ones are at least 1.5 times larger. A region across UTM zones needs `crs=`.

### What the cube remembers

The cube's attributes say which embeddings it holds: product, version and variant (TESSERA has had several inference runs, and each is a space of its own), the licence and the citation. A model trained with `zeit.train_classifier` or `zeit.ai.train` records them, and `zeit.classify`/`zeit.ai.predict` refuse a cube of other embeddings rather than produce a map from vectors it was never trained on. The algorithms that fit a physical, seasonal signal (LandTrendr, CCDC, BFAST, phenology, TWDTW, indices...) refuse a cube of embeddings too: there is nothing in a yearly summary for them to fit.

## Look

```python
zeit.plot(emb)
```

Dozens of dimensions cannot be shown at once, so `zeit.plot` shows the first three principal components as red, green and blue. The components are fitted once, on pixels of every year, and applied to all of them: a colour means the same embedding in 2017 and in 2025, and paging through the years shows the land changing (fields, clearings, regrowth). `zeit.plot(emb, band="A07")` shows one dimension.

## Classify from few samples

The embeddings were made for this. Each dimension of a year is a feature:

```python
year = emb.sel(time="2024").persist()                     # read once: training reads whole blocks
rf = zeit.train_classifier(year, "samples.gpkg", label="class")
classes = zeit.classify(year, rf)
```

Then validate it as any map ([Accuracy & Area](accuracy.md)):

```python
design = zeit.sampling_design(classes.label, expected_ua=0.8)
points = zeit.stratified_sample(classes.label, design=design)
```

Several years at once (`emb.sel(time=slice("2022", "2024"))`) give the classifier how each pixel changed, and `zeit.som`/`zeit.snic` cluster or segment the embeddings without labels.

## Find more places like these

```python
mines = zeit.similarity(emb, "known_mines.gpkg", year=2024)    # (time, y, x): cosine similarity
per_class = zeit.similarity(emb, "samples.gpkg", by="class")    # one map per class
```

The reference is the mean embedding of the samples (every cell of a polygon) in one year, and every year of the cube is compared with it: where something like it appears, and when.

## Change from year to year

```python
change = zeit.embedding_change(emb)                   # distance of each year to the year before
events = zeit.extract_events(change, min_magnitude=0.3)
```

`embedding_change` measures how far each pixel's embedding moved (`1 - cos` of the angle between two years), and `extract_events` keeps the largest move of each pixel in the schema of the other change algorithms: `yod` the year before the first embedding that shows the change, `date`, `magnitude`, and `dsnr` (the move over the pixel's usual one). An embedding summarises a year, so a clearing in the middle of a year shows partly in it and partly in the next.

Each product has its own scale. Over the box above, the median pixel moved 0.05–0.1 a year in TESSERA and 0.02 in AlphaEarth, so the thresholds differ (0.3 and 0.2 here). With them, the two products found change in 6,459 of the same pixels, and agreed on its year (within one) in 6,326. The same comparison works against the classic algorithms:

```python
agree = zeit.agreement({
    "tessera": zeit.extract_events(zeit.embedding_change(emb), min_magnitude=0.3),
    "alphaearth": zeit.extract_events(zeit.embedding_change(aef), min_magnitude=0.2),
    "landtrendr": zeit.extract_events(zeit.landtrendr(ndvi), min_magnitude=2000),
}, tolerance=1)
```

## Earth Engine, local copies, versions

- AlphaEarth comes by default from its open copy on Source Cooperative: free, no account, no quota. `backend="gee"` reads it from the `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` collection of Earth Engine instead (start a session with `zeit.gee.auth.initialize_gee()`), which counts against the account's quota; each year is downloaded once into `cache_dir`.
- `store=` reads a copy of either product: a TESSERA Zarr store (a local one needs no `geotessera`), or a folder with AlphaEarth's `aef_index.parquet` and its COGs.
- `version=`/`variant=` choose a TESSERA run (default v1.1, the complete global one) and `depth=` the first dimensions of a v2 store (Matryoshka embeddings, trained to work truncated).
- From the shell: `zeit embeddings ./out --source tessera --bbox -63 -10 -62.9 -9.9 --years 2018-2024` ([CLI](../cli.md)).

## Cite

TESSERA: Feng et al. (2025), *TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis*, arXiv:2506.20380 (embeddings and weights CC0; the authors ask to be cited). AlphaEarth: Brown et al. (2025), *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data*, arXiv:2507.22291; the dataset is CC-BY 4.0 and requires the attribution "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind." Both are in the cube's attributes.
