---
title: Zeit — Change detection and time series for satellite imagery
hide:
  - navigation
  - toc
  - path
---

<div class="zeit-hero" markdown>

<div markdown>

<img class="zeit-hero__logo" src="assets/logo/zeit-lockup-light.svg" alt="Zeit">

<h1>Find <span>when</span> and <span>where</span> the landscape changed.</h1>

<p class="lead">Zeit is an ultra-fast Python library for analysing satellite image time series. Reference algorithms such as LandTrendr, CCDC and BFAST are re-engineered in parallel Rust and validated against their original implementations. Use them to detect deforestation, fires, regrowth, crop cycles and long-term trends. Full scenes run on a laptop using every core, and the same code scales out in parallel to your own cluster or the cloud with Dask.</p>

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[See what it can do](#what-you-can-do-with-zeit){ .md-button }
[:fontawesome-brands-github: GitHub](https://github.com/sacridini/zeit-cdts){ .md-button }

<div class="zeit-hero__install" markdown>

```bash
pip install zeit-cdts
```

</div>

</div>

<div class="zeit-stack" style="--px-x:37.7%; --px-y:50.2%" role="img" aria-label="Four yearly NDVI images of the same place in Rondônia (1990, 2000, 2010, 2024) stacked in time, with the LandTrendr map of the year of loss on top, and the NDVI time series of one pixel through the stack">
<div class="zeit-stack__scene">
<div class="zeit-stack__layer" style="--i:0"><img src="assets/figures/hero/ndvi_1990.webp" alt="" width="480" height="480"><span class="zeit-stack__tag">1990</span><i class="zeit-stack__px"></i></div>
<div class="zeit-stack__layer" style="--i:1"><img src="assets/figures/hero/ndvi_2000.webp" alt="" width="480" height="480"><span class="zeit-stack__tag">2000</span><i class="zeit-stack__px"></i></div>
<div class="zeit-stack__layer" style="--i:2"><img src="assets/figures/hero/ndvi_2010.webp" alt="" width="480" height="480"><span class="zeit-stack__tag">2010</span><i class="zeit-stack__px"></i></div>
<div class="zeit-stack__layer" style="--i:3"><img src="assets/figures/hero/ndvi_2024.webp" alt="" width="480" height="480"><span class="zeit-stack__tag">2024</span><i class="zeit-stack__px"></i></div>
<div class="zeit-stack__layer zeit-stack__layer--result" style="--i:4"><img src="assets/figures/hero/loss_year.webp" alt="" width="480" height="480"><span class="zeit-stack__tag">Year of loss</span><i class="zeit-stack__px"></i></div>
</div>
<div class="zeit-stack__card">
<div class="zeit-stack__card-head"><span class="zeit-stack__dot"></span>One pixel, 40 years of Landsat NDVI</div>
--8<-- "docs/assets/figures/hero/pixel_chart.svg"
<div class="zeit-stack__card-foot"><span><i class="k-obs"></i>observed</span><span><i class="k-fit"></i>LandTrendr fit</span></div>
</div>
</div>

</div>

<p class="zeit-hero__note"><strong>The stack:</strong> 40 years of Landsat NDVI over Rondônia, Brazil, one image per year. Zeit reads each pixel through the stack as a time series and finds when it changed; the map on top is LandTrendr's year of the largest loss, for 2.8 million pixels in about 5 seconds on a desktop CPU. <em>Data: annual Landsat NDVI composites exported from <a href="https://github.com/eMapR/LT-GEE">LT-GEE</a>.</em></p>

## What is Zeit?

Every satellite pixel has a history. Stack images of the same place over time and each pixel becomes a **time series**: a line that stays flat while a forest stands, drops when it is cleared, and climbs back as it regrows. Most land-change questions come down to reading those lines. Did something change? When? How much? Was it sudden or gradual?

<figure markdown>
  ![Three NDVI snapshots of the same area in 1990, 2000 and 2020, and the NDVI time series of one pixel that was cleared in 2003](assets/figures/concept_pixel_time_series.webp)
  <figcaption>The circled pixel was forest until 2002, then was cleared. Seen as a time series, the event is obvious. Zeit finds events like this in every pixel of an image. <em>Data: annual Landsat NDVI composites exported from <a href="https://github.com/eMapR/LT-GEE">LT-GEE</a> on Google Earth Engine.</em></figcaption>
</figure>

Zeit gives you the standard scientific methods for reading these time series, plus the plumbing around them. It can pull imagery from cloud catalogs, mask clouds, fill gaps, run the analysis in parallel, and write maps back out as GeoTIFF or Zarr.

<span class="kicker">Gallery</span>

## What you can do with Zeit

Every image below was produced by Zeit itself. Click one to open its tutorial.

<div class="gallery" markdown>

<a class="tile" href="tutorials/landtrendr/">
  <img src="assets/figures/thumbs/landtrendr.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Change detection</span><span class="tile-title">Map forest loss and regrowth</span><span class="tile-text">LandTrendr turns yearly images into maps of when, how much and how fast change happened.</span></span>
</a>

<a class="tile" href="tutorials/recovery/">
  <img src="assets/figures/thumbs/recovery.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Change detection</span><span class="tile-title">Measure how land recovers</span><span class="tile-text">Years to recovery, R80P and the relative recovery after each disturbance, from LandTrendr or any annual series.</span></span>
</a>

<a class="tile" href="tutorials/ccdc/">
  <img src="assets/figures/thumbs/ccdc.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Change detection</span><span class="tile-title">Detect change in dense time series</span><span class="tile-text">CCDC models each pixel's seasonal cycle and flags the moment it breaks.</span></span>
</a>

<a class="tile" href="tutorials/beast/">
  <img src="assets/figures/thumbs/beast.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Change detection</span><span class="tile-title">Weigh every possible change</span><span class="tile-text">BEAST averages thousands of models of the trend and the season, and says how probable each changepoint is.</span></span>
</a>

<a class="tile" href="tutorials/sccd/">
  <img src="assets/figures/thumbs/sccd.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Monitoring</span><span class="tile-title">Watch every new image</span><span class="tile-text">S-CCD carries CCDC's models forward with a Kalman filter and flags changes still in progress.</span></span>
</a>

<a class="tile" href="tutorials/bfast_monitor/">
  <img src="assets/figures/thumbs/bfast_monitor.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Monitoring</span><span class="tile-title">Raise near-real-time alerts</span><span class="tile-text">BFAST Monitor tests each new image against a stable history.</span></span>
</a>

<a class="tile" href="tutorials/ewmacd/">
  <img src="assets/figures/thumbs/ewmacd.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Monitoring</span><span class="tile-title">Monitor irregular, cloudy series</span><span class="tile-text">EWMACD follows every observation with control charts and dates disturbances, gains and recoveries.</span></span>
</a>

<a class="tile" href="tutorials/mann_kendall/">
  <img src="assets/figures/thumbs/mann_kendall.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Trends</span><span class="tile-title">Map greening and browning</span><span class="tile-text">Mann-Kendall and Theil-Sen give a robust, significance-tested trend per pixel.</span></span>
</a>

<a class="tile" href="tutorials/regional-trends/">
  <img src="assets/figures/thumbs/parts.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Trends</span><span class="tile-title">Test a region's trend</span><span class="tile-text">PARTS asks whether a region, or each land-cover class, is really trending, accounting for correlated pixels.</span></span>
</a>

<a class="tile" href="tutorials/hotspots/">
  <img src="assets/figures/thumbs/hotspots.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Space-time patterns</span><span class="tile-title">Find emerging hot spots</span><span class="tile-text">Where values cluster high or low, and how that changed: new, intensifying, persistent or sporadic hot spots.</span></span>
</a>

<a class="tile" href="tutorials/phenology/">
  <img src="assets/figures/thumbs/phenology.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Phenology</span><span class="tile-title">Extract crop and vegetation calendars</span><span class="tile-text">Start, peak and end of season, plus 16 more metrics, for every year.</span></span>
</a>

<a class="tile" href="tutorials/twdtw/">
  <img src="assets/figures/thumbs/twdtw.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Classification</span><span class="tile-title">Classify by temporal signature</span><span class="tile-text">TWDTW matches each pixel to reference patterns, even when seasons shift.</span></span>
</a>

<a class="tile" href="tutorials/unmixing/">
  <img src="assets/figures/thumbs/unmixing.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Spectral mixtures</span><span class="tile-title">Unmix pixels into materials</span><span class="tile-text">The fraction of vegetation, soil, water or any endmember in every pixel and date, fully constrained.</span></span>
</a>

<a class="tile" href="tutorials/snic/">
  <img src="assets/figures/thumbs/snic.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Segmentation</span><span class="tile-title">Group pixels into objects</span><span class="tile-text">SNIC superpixels group pixels whose whole trajectories are similar.</span></span>
</a>

<a class="tile" href="tutorials/som/">
  <img src="assets/figures/thumbs/som.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Clustering</span><span class="tile-title">Discover patterns without labels</span><span class="tile-text">Self-organizing maps (a bit-exact, much faster port of <code>minisom</code>) cluster millions of trajectories into a few prototypes.</span></span>
</a>

<a class="tile" href="tutorials/tempcnn/">
  <img src="assets/figures/thumbs/tempcnn.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Deep learning</span><span class="tile-title">Train land-cover classifiers</span><span class="tile-text">PyTorch models (TempCNN, LightTAE, U-TAE) that are weight-compatible with R <code>sits</code>.</span></span>
</a>

<a class="tile" href="tutorials/postprocessing/">
  <img src="assets/figures/thumbs/postprocessing.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">After classifying</span><span class="tile-title">Clean maps and know where they hold</span><span class="tile-text">Bayesian smoothing, a minimum mapping unit, and the area of applicability: where the model's samples say nothing.</span></span>
</a>

</div>

<span class="kicker">Workflow</span>

## How a Zeit analysis fits together

<div class="steps" markdown>

<div markdown>
**Get the data**

Stream a cube from a STAC catalog, Earth Engine or local GeoTIFFs.

`build_time_series` · `download_gee_timeseries`
</div>

<div markdown>
**Clean it**

Mask clouds, composite to a regular time step, smooth noise.

`tmask` · `regularize_time_series` · `smooth`
</div>

<div markdown>
**Analyse every pixel**

Run change detection, trends, phenology or a classifier.

`landtrendr` · `ccdc`
</div>

<div markdown>
**Map the result**

Turn per-pixel output into maps, look at them, and write GeoTIFF or Zarr.

`extract_events` · `plot` · `save_raster`
</div>

</div>

The same analysis in three styles. Pick the one that fits your data:

=== "Raster file"

    ```python
    import zeit

    ndvi = zeit.load_raster("ndvi_1985_2024.tif", start_year=1985)   # (time, y, x), georeferenced

    lt = zeit.landtrendr(ndvi)                            # looks for NDVI drops (direction="loss")
    loss = zeit.extract_events(lt, min_magnitude=2000)    # greatest loss per pixel

    zeit.plot(ndvi, fit=lt)                               # page through the years, click a pixel to see its fit
    zeit.save_raster(loss, "lt_results")                  # one GeoTIFF per map: yod.tif, magnitude.tif, ...
    ```

=== "Xarray / Dask cube"

    ```python
    import zeit  # registers the .zeit accessor on xarray objects

    cube = zeit.build_time_series(
        source="earth_search", collection="sentinel-2-l2a",
        bbox=[-63.2, -10.2, -63.0, -10.0],
        start_date="2019-01-01", end_date="2024-12-31",
        bands=["red", "nir"], apply_cloud_mask=True,
    )
    ndvi = (cube.sel(band="nir") - cube.sel(band="red")) / (cube.sel(band="nir") + cube.sel(band="red"))
    ndvi_16d = zeit.regularize_time_series(ndvi, freq="16D", method="median")

    trend = ndvi_16d.zeit.mann_kendall(method="seasonal", period=23)
    trend.to_zarr("ndvi_trend.zarr", consolidated=True)   # computed chunk by chunk, in parallel
    ```

=== "Command line"

    ```bash
    # No Python needed: every core algorithm is also a CLI subcommand.
    zeit landtrendr ndvi_1985_2024.tif results/ --start-year 1985 --event-type loss --min-mag 2000
    zeit mmu-filter results/lt_event_yod.tif results/lt_event_yod_clean.tif --mmu-pixels 11
    ```

<span class="kicker">Why Zeit</span>

## Built for trustworthy results at scale

<div class="stats" markdown>
<div><span class="n">100%</span><span class="l">vertex-for-vertex agreement with the original LandTrendr IDL code, and model-for-model with CCDC MATLAB</span></div>
<div><span class="n">232×</span><span class="l">faster than the reference LandTrendr on one core, with more from its threads</span></div>
<div><span class="n">19×</span><span class="l">higher throughput than Google Earth Engine on a full Landsat tile, with no queue</span></div>
<div><span class="n">0</span><span class="l">compilers needed. Pre-built wheels for Windows, macOS and Linux</span></div>
</div>

<div class="grid cards two" markdown>

-   :material-scale-balance:{ .lg .middle } **Faithful to the originals**

    ---

    Each algorithm is ported from its reference implementation (IDL, MATLAB, R) and tested against it, for its results and for its speed. You get the published method, not an approximation.

    [:octicons-arrow-right-24: Fidelity reports](benchmarks/fidelity.md) · [:octicons-arrow-right-24: Against the originals](benchmarks/vs-originals.md)

-   :material-lightning-bolt:{ .lg .middle } **Fast by default**

    ---

    The per-pixel work runs in Rust, on every core and outside the Python GIL. A laptop can process scenes that used to need a cloud platform.

    [:octicons-arrow-right-24: Performance](benchmarks/performance.md)

-   :material-server-network:{ .lg .middle } **Scales without rewrites**

    ---

    The `.zeit` xarray accessor maps every algorithm over Dask chunks, so the same script runs on one machine or a cluster.

    [:octicons-arrow-right-24: Scaling up](tutorials/parallel-cloud-processing.md)

-   :material-puzzle-outline:{ .lg .middle } **One toolbox, end to end**

    ---

    Data access, cloud masking, smoothing, unmixing, change detection, monitoring, recovery, trends, hot spots, phenology, segmentation, deep learning, post-processing and plotting all use the same array conventions.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## Where to go next

<div class="grid cards two" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } **New to Zeit?**

    ---

    Install it and produce your first change map in five minutes, then learn the few concepts every tutorial builds on.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md) · [Core concepts](getting-started/concepts.md)

-   :material-map-search-outline:{ .lg .middle } **Not sure which method to use?**

    ---

    A short guide that maps common questions ("when was this cleared?", "is it getting greener?") to the right algorithm.

    [:octicons-arrow-right-24: Choosing an algorithm](getting-started/choosing-an-algorithm.md)

-   :material-book-open-variant:{ .lg .middle } **Ready to go deeper?**

    ---

    Step-by-step tutorials for every algorithm, each with real outputs, parameter guidance and references.

    [:octicons-arrow-right-24: User guide](tutorials/index.md)

-   :material-api:{ .lg .middle } **Looking something up?**

    ---

    Every public function, with signatures, parameters and a runnable example.

    [:octicons-arrow-right-24: API reference](api/index.md) · [CLI](cli.md)

</div>
