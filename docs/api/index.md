# API Reference

<p class="lead">Every public function and class, with its signature, parameters and an example. Signatures on these pages are generated from the code, so they always match the installed version.</p>

!!! tip "How to import"
    `import zeit` gives you the most used functions at the top level (`zeit.landtrendr`, `zeit.save_raster`, …) and registers the `.zeit` xarray accessor. Everything else lives in submodules (`zeit.twdtw`, `zeit.segmentation`, `zeit.ai`, …), shown in each signature.

<div class="grid cards" markdown>

-   :material-database-arrow-down-outline:{ .lg .middle } **[Data & I/O](data.md)**

    ---

    STAC and Earth Engine cubes, local files, compositing, GeoTIFF I/O, QA weights.

-   :material-auto-fix:{ .lg .middle } **[Pre-processing](preprocessing.md)**

    ---

    Tmask cloud masking, Whittaker and Savitzky-Golay smoothing, spike removal.

-   :material-chart-timeline-variant-shimmer:{ .lg .middle } **[Change Detection](change-detection.md)**

    ---

    LandTrendr, CCDC, BFAST, BFAST Monitor, BFAST Lite and event extraction.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **[Time-Series Analysis](time-series.md)**

    ---

    Mann-Kendall trends, phenology, TWDTW, SNIC segmentation, SOM clustering.

-   :material-filter-outline:{ .lg .middle } **[Post-processing](post-processing.md)**

    ---

    Spatial filters, CCDC classification, water masks, validation dashboard.

-   :material-map-search-outline:{ .lg .middle } **[Visualisation](plot.md)**

    ---

    `zeit.plot`: a fast time-series viewer, pixel inspector, basemaps and static figures.

-   :material-brain:{ .lg .middle } **[Deep Learning](ai.md)**

    ---

    TempCNN, LightTAE, U-TAE, Siamese and foundation models; datasets and losses.

-   :material-grid:{ .lg .middle } **[Xarray Accessor](xarray.md)**

    ---

    `DataArray.zeit.*`: every algorithm on lazy Dask cubes, and Zarr output.

-   :material-console:{ .lg .middle } **[Command Line](../cli.md)**

    ---

    The core algorithms as `zeit` subcommands, for scripts and HPC jobs.

</div>

## All functions

<div class="api-index" markdown>

| Function | What it does |
| :--- | :--- |
| **Data & I/O** | |
| [`build_time_series`](data.md#build_time_series) | Lazy cube from a STAC catalog |
| [`build_annual_composites`](data.md#build_annual_composites) | Cloud-masked annual composites from a STAC catalog (LandTrendr input) |
| [`download_gee_timeseries`](data.md#download_gee_timeseries) | Harmonised Landsat composites from Earth Engine |
| [`download_gee_image`](data.md#download_gee_image) / [`resolve_roi`](data.md#resolve_roi) | Download any `ee.Image`; turn tile ids, files or boxes into an area |
| [`regularize_time_series`](data.md#regularize_time_series) | Median or medoid composites at a fixed step |
| [`load_raster`](data.md#load_raster) / [`save_raster`](data.md#save_raster) | Read any time series as a georeferenced cube / write GeoTIFFs |
| [`get_georef`](data.md#get_georef) | CRS and transform of a raster or cube |
| [`qc_sentinel2_scl`](data.md#qc_sentinel2_scl), [`qc_modis_summary`](data.md#qc_modis_summary), [`qc_modis_state`](data.md#qc_modis_state) | QA bands to observation weights |
| **Pre-processing** | |
| [`tmask`](preprocessing.md#tmask) | Time-series cloud and shadow detection of a cube |
| [`smooth`](preprocessing.md#smooth) | Whittaker (uneven dates, gaps, weights) or Savitzky-Golay smoothing of a cube or series |
| [`desawtooth`](preprocessing.md#desawtooth) | LandTrendr spike removal |
| **Change detection** | |
| [`landtrendr`](change-detection.md#landtrendr) | LandTrendr on a file, cube, array or single pixel |
| [`extract_events`](change-detection.md#extract_events) | LandTrendr vertices to event maps |
| [`apply_vertices`](change-detection.md#apply_vertices) | Fit another band to the same vertices |
| [`ccdc`](change-detection.md#ccdc) | CCDC on a file, cube, array or single pixel |
| [`predict_synthetic_image`](change-detection.md#predict_synthetic_image) | Evaluate CCDC models on any date |
| [`bfast_monitor`](change-detection.md#bfast_monitor) | Near-real-time monitoring, on a file, cube, array or single pixel |
| [`bfast_lite`](change-detection.md#bfast_lite) | Optimal multiple breakpoints |
| [`bfast`](change-detection.md#bfast) | Trend and seasonal breaks |
| **Time-series analysis** | |
| [`mann_kendall`](time-series.md#mann_kendall) | Trend test and Theil-Sen slope |
| [`phenology`](time-series.md#phenology) | 19 phenology metrics per year or season |
| [`twdtw`](time-series.md#twdtw) | Classification by time-weighted DTW, on a file, cube or single pixel |
| [`snic`](time-series.md#snic) / [`snic_to_polygons`](time-series.md#snic_to_polygons) / [`snic_grid`](time-series.md#snic_grid) | Superpixel segmentation of a map or cube, and its polygons |
| [`som`](time-series.md#som) | Self-organizing map clusters of a map or cube, with each neuron's prototype |
| [`clean_samples`](time-series.md#clean_samples) | Flag labelled samples whose class is not their SOM neuron's |
| [`SOM`](time-series.md#som_1) | The SOM engine, online and batch (bit-exact `minisom` port) |
| **Post-processing** | |
| [`apply_mmu_filter`](post-processing.md#apply_mmu_filter) | Remove patches below a minimum size |
| [`apply_majority_filter`](post-processing.md#apply_majority_filter) / [`apply_bayesian_filter`](post-processing.md#apply_bayesian_filter) | Smooth class maps |
| [`train_classifier`](post-processing.md#train_classifier) / [`classify`](post-processing.md#classify) | Train at sample points, classify any cube, metrics or CCDC models |
| [`extract_water_mask`](post-processing.md#extract_water_mask) | Water mask from CCDC |
| [`generate_landtrendr_accuracy_dashboard`](post-processing.md#generate_landtrendr_accuracy_dashboard) | Interactive validation page |
| **Visualisation** | |
| [`plot`](plot.md#plot) | Any cube, map, result or pixel: interactive viewer in a notebook or window, or a matplotlib figure |
| **Deep learning** (`zeit.ai`) | |
| [`ai.samples`](ai.md#samples), [`ai.train`](ai.md#train), [`ai.predict`](ai.md#predict) | From a cube and labelled samples to a map with a deep learning model |
| [`ai.save`](ai.md#save), [`ai.load`](ai.md#load) | Keep a trained model with what prediction checks |
| [`TempCNN`](ai.md#tempcnn), [`LightTAE`](ai.md#lighttae), [`LTAE`](ai.md#ltae) | Pixel time-series classifiers |
| [`UTAE`](ai.md#utae) | Spatio-temporal patch segmentation |
| [`SiameseChangeDetector`](ai.md#siamesechangedetector) | Two-date change detection |
| [`GeoFoundationViT`](ai.md#geofoundationvit) | Foundation-model fine-tuning |
| [`STACCubeDataset`](ai.md#staccubedataset) | Windows that cover a lazy cube, for a loop of your own |
| [`FocalLoss`](ai.md#focalloss), [`TverskyLoss`](ai.md#tverskyloss), [`ContrastiveSiameseLoss`](ai.md#contrastivesiameseloss) | Losses for imbalanced data |

</div>
