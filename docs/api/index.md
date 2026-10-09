# API Reference

<p class="lead">Every public function and class, with its signature, parameters and an example. Signatures on these pages are generated from the code, so they always match the installed version.</p>

!!! tip "How to import"
    `import zeit` gives you the most used functions at the top level (`zeit.run_landtrendr_array`, `zeit.save_raster`, …) and registers the `.zeit` xarray accessor. Everything else lives in submodules (`zeit.bfast`, `zeit.trend`, `zeit.ai`, …), shown in each signature.

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
| [`build_local_cube`](data.md#build_local_cube) | Lazy cube from a folder of GeoTIFFs |
| [`download_gee_timeseries`](data.md#download_gee_timeseries) | Harmonised Landsat composites from Earth Engine |
| [`download_gee_image`](data.md#download_gee_image) / [`resolve_roi`](data.md#resolve_roi) | Download any `ee.Image`; turn tile ids, files or boxes into an area |
| [`regularize_time_series`](data.md#regularize_time_series) | Median or medoid composites at a fixed step |
| [`load_raster`](data.md#load_raster) / [`save_raster`](data.md#save_raster) | Read any time series as a georeferenced cube / write GeoTIFFs |
| [`get_georef`](data.md#get_georef) | CRS and transform of a raster or cube |
| [`qc_sentinel2_scl`](data.md#qc_sentinel2_scl), [`qc_modis_summary`](data.md#qc_modis_summary), [`qc_modis_state`](data.md#qc_modis_state) | QA bands to observation weights |
| **Pre-processing** | |
| [`apply_tmask_stack`](preprocessing.md#apply_tmask_stack) / [`run_tmask_pixel`](preprocessing.md#run_tmask_pixel) | Time-series cloud and shadow detection |
| [`apply_whittaker_filter`](preprocessing.md#apply_whittaker_filter) | Weighted Whittaker smoothing |
| [`apply_savgol_filter`](preprocessing.md#apply_savgol_filter) | Savitzky-Golay smoothing |
| [`desawtooth`](preprocessing.md#desawtooth) | LandTrendr spike removal |
| **Change detection** | |
| [`run_landtrendr_array`](change-detection.md#run_landtrendr_array) / [`run_landtrendr_image`](change-detection.md#run_landtrendr_image) / [`run_landtrendr`](change-detection.md#run_landtrendr) | LandTrendr: array, file, pixel |
| [`extract_events`](change-detection.md#extract_events) | LandTrendr vertices to event maps |
| [`apply_vertices`](change-detection.md#apply_vertices) | Fit another band to the same vertices |
| [`run_ccdc_array`](change-detection.md#run_ccdc_array) / [`run_ccdc_image`](change-detection.md#run_ccdc_image) / [`run_ccdc`](change-detection.md#run_ccdc) | CCDC: array, file, pixel |
| [`predict_synthetic_image`](change-detection.md#predict_synthetic_image) / [`predict`](change-detection.md#predict) | Evaluate CCDC models on any date |
| [`run_bfast_monitor_dask`](change-detection.md#run_bfast_monitor_dask) | Near-real-time monitoring |
| [`run_bfast_lite_dask`](change-detection.md#run_bfast_lite_dask) | Optimal multiple breakpoints |
| [`run_bfast_dask`](change-detection.md#run_bfast_dask) | Trend and seasonal breaks |
| [`run_bfast_*_image`](change-detection.md#geotiff-versions) | BFAST family on GeoTIFFs |
| **Time-series analysis** | |
| [`run_mann_kendall_dask`](time-series.md#run_mann_kendall_dask) / [`run_mann_kendall_image`](time-series.md#run_mann_kendall_image) | Trend test and Theil-Sen slope |
| [`run_phenology_dask`](time-series.md#run_phenology_dask) | 19 phenology metrics per season |
| [`classify_twdtw`](time-series.md#classify_twdtw) / [`run_twdtw`](time-series.md#run_twdtw) / [`run_twdtw_batch`](time-series.md#run_twdtw_batch) | Time-weighted DTW |
| [`run_snic`](time-series.md#run_snic) / [`snic_to_polygons`](time-series.md#snic_to_polygons) / [`snic_grid`](time-series.md#snic_grid) | Superpixel segmentation |
| [`SOM`](time-series.md#som) | Self-organizing maps, online and batch (bit-exact `minisom` port) |
| **Post-processing** | |
| [`apply_mmu_filter`](post-processing.md#apply_mmu_filter) | Remove patches below a minimum size |
| [`apply_majority_filter`](post-processing.md#apply_majority_filter) / [`apply_bayesian_filter`](post-processing.md#apply_bayesian_filter) | Smooth class maps |
| [`train_ccdc_classifier`](post-processing.md#train_ccdc_classifier) / [`classify_ccdc_stack`](post-processing.md#classify_ccdc_stack) | Classify CCDC coefficients |
| [`extract_water_mask`](post-processing.md#extract_water_mask) | Water mask from CCDC |
| [`generate_landtrendr_accuracy_dashboard`](post-processing.md#generate_landtrendr_accuracy_dashboard) | Interactive validation page |
| **Deep learning** (`zeit.ai`) | |
| [`TempCNN`](ai.md#tempcnn), [`LightTAE`](ai.md#lighttae), [`LTAE`](ai.md#ltae) | Pixel time-series classifiers |
| [`UTAE`](ai.md#utae) | Spatio-temporal patch segmentation |
| [`SiameseChangeDetector`](ai.md#siamesechangedetector) | Two-date change detection |
| [`GeoFoundationViT`](ai.md#geofoundationvit) | Foundation-model fine-tuning |
| [`STACCubeDataset`](ai.md#staccubedataset) | Patches from a lazy cube |
| [`FocalLoss`](ai.md#focalloss), [`TverskyLoss`](ai.md#tverskyloss), [`ContrastiveSiameseLoss`](ai.md#contrastivesiameseloss) | Losses for imbalanced data |

</div>
