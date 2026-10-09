<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sacridini/zeit-cdts/refs/heads/main/docs/assets/logo-wide-dark.png">
    <img src="https://raw.githubusercontent.com/sacridini/zeit-cdts/refs/heads/main/docs/assets/logo-wide.png" alt="Zeit Logo" width="420">
  </picture>
</p>

# Zeit: Change Detection and Time Series for Python

[![Build Wheels](https://github.com/sacridini/zeit-cdts/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/sacridini/zeit-cdts/actions/workflows/build_wheels.yml)
[![Tests](https://github.com/sacridini/zeit-cdts/actions/workflows/tests.yml/badge.svg)](https://github.com/sacridini/zeit-cdts/actions/workflows/tests.yml)
[![Docs](https://github.com/sacridini/zeit-cdts/actions/workflows/docs.yml/badge.svg)](https://github.com/sacridini/zeit-cdts/actions/workflows/docs.yml)
[![PyPI version](https://badge.fury.io/py/zeit-cdts.svg)](https://badge.fury.io/py/zeit-cdts)

**Zeit** is a high-performance Python package for Earth Observation (EO) data cube processing and time series analysis. It bridges the gap between modern cloud-native data formats (STAC, Xarray, Dask) and state-of-the-art pixel-based trajectory algorithms (TWDTW, CCDC, LandTrendr). 

Built with highly optimized C++ extensions (OpenMP and Eigen SIMD) bound to Python via `pybind11`, Zeit is designed to handle massive multi-spectral satellite image time series efficiently while keeping memory footprints strictly bounded.

---

## Key Capabilities

- **ARD Data Cube Ingestion:** Fetch cloud-native STAC catalogs (via MGRS/WRS tiles or Bounding Boxes) or parse local TIFF directories into lazy Dask-backed `xarray` Datacubes.
- **Semantic Cloud Masking:** Automated extraction and translation of Quality Assessment (QA) bands for Landsat and Sentinel-2 directly inside the query pipeline, plus a temporal **Tmask** harmonic baseline model to catch clouds/shadows missed by the native QA mask.
- **Temporal Regularization:** Mathematical composite generation (e.g., Medoid, Median) to align irregular satellite acquisitions into uniform time steps (crucial for Deep Learning and DTW).
- **High-Performance C++ Algorithms:**
  - **TWDTW** (Time-Weighted Dynamic Time Warping): Highly optimized with LB_Keogh lower bounding, early abandonment, Sakoe-Chiba constraints, and multivariate Eigen vectorization.
  - **Batch SOM** (Self-Organizing Maps): Unsupervised multi-threaded clustering of massive spectral-temporal arrays.
  - **CCDC / COLD**: Continuous Change Detection and Classification via robust harmonic modeling.
  - **LandTrendr**: Trajectory-based disturbance and recovery detection.
  - **BFAST, BFAST Monitor & BFAST Lite**: Classic iterative trend+season break detection, near-real-time monitoring, and single-pass multiple-breakpoint detection (ported from R's `bfast`/`strucchangeRcpp`).
  - **Phenology Extraction**: 19 simultaneous phenological metrics from optimized curve-fitting models (Beck, Elmore, Gu, Zhang, Asymmetric Gaussian, Double Logistic), with QA-based per-observation weighting.
  - **Mann-Kendall / Theil-Sen**: Pixel-wise non-parametric trend test and slope estimation for detecting statistically significant greening/browning trends.
  - **SNIC Segmentation**: Superpixel segmentation of images and whole time series cubes (one segment = similar trajectories), matching the original SNIC's labels pixel for pixel, with tile-parallel processing for large scenes.
- **Deep Learning (`zeit.ai`):** Pre-built PyTorch architectures tailored for spatio-temporal Earth Observation (U-TAE, TempCNN, Siamese Networks), plus wrappers for Geospatial Foundation Models (ViT).
- **Visualisation (`zeit.plot`):** One function to look at any cube, map or result: an interactive viewer in Jupyter (or its own window from a script) that pages through dense time series at the display's frame rate, with a click-a-pixel inspector showing each algorithm's fit, satellite basemaps and vector outlines; or a matplotlib figure for reports.
- **Command-Line Interface:** Every core algorithm is also available as a `zeit` subcommand, for running change detection on GeoTIFF stacks from bash scripts, cron jobs, or HPC environments without writing Python.

---

## Installation

```bash
pip install zeit-cdts
```
*(Note: Wheels are provided for Windows, Linux, and macOS. macOS runs in single-threaded mode by default due to Apple Clang lacking OpenMP).*

For `zeit.plot` (matplotlib and anywidget), install the `plot` extra: `pip install zeit-cdts[plot]`.

**For macOS users who want C++ OpenMP multi-threading:**
Apple's default Clang compiler disables OpenMP. To achieve maximum performance and enable multi-threading, you must install the `libomp` library and manually export the compilation flags *before* forcing a local compilation:

```bash
brew install libomp
export CFLAGS="-I$(brew --prefix libomp)/include"
export CXXFLAGS="-I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -lomp"
pip install --no-binary zeit-cdts zeit-cdts
```

---

## Quickstart: load, run, export

Every algorithm is one function that understands its input, and the dates and georeferencing travel with the data:

```python
import zeit

ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")   # (time, y, x) cube, years from the band names
lt = zeit.landtrendr(ndvi)                               # or zeit.ccdc, zeit.bfast_monitor, zeit.mann_kendall, zeit.phenology...
loss = zeit.extract_events(lt)                           # greatest NDVI loss per pixel
zeit.save_raster(loss, "lt_rondonia")                    # one georeferenced GeoTIFF per metric
zeit.plot(ndvi, fit=lt)                                  # look: page through the years, click a pixel to see its fit
```

Coming from 0.26 or earlier? See [Upgrading to the one-function API](https://sacridini.github.io/zeit-cdts/getting-started/migrating/).

## Visualisation

`zeit.plot` shows any cube, map, result `Dataset` or pixel's series. In a notebook (JupyterLab, VS Code, Colab) it returns an interactive viewer; in a script it opens the same viewer in its own window; with `static=True` or `save=` it draws a matplotlib figure.

```python
ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
zeit.plot(ndvi)                                          # space plays the 40 years, wheel zooms, hover reads values
zeit.plot(ndvi, fit=zeit.landtrendr(ndvi))               # click a pixel: its series and LandTrendr's segments
zeit.plot(zeit.extract_events(zeit.landtrendr(ndvi)), basemap="satellite", vector="aoi.gpkg")
zeit.plot(ndvi, time=["1985", "2000", "2024"], save="ndvi.png")   # a figure for a report
```

It is fast because each frame travels to the browser once, as one byte per cell, and is coloured by the GPU: paging through time then runs at the display's refresh rate without touching Python, and the cube is only ever read at screen resolution. See the [plotting tutorial](https://sacridini.github.io/zeit-cdts/tutorials/plotting/).

## Cloud-Native ARD Cubes (STAC)

Fetch lazy evaluated, Dask-backed analysis-ready data cubes directly from STAC providers (e.g., Earth Search, Planetary Computer, Brazil Data Cube).

```python
import zeit

# Build a lazy DataArray using MGRS/WRS tiles or Bounding Boxes
cube = zeit.build_time_series(
    source="earth_search",
    collection="sentinel-2-l2a",
    tiles=["22JFQ"], # Sentinel-2 MGRS or Landsat WRS-2 (e.g., "215065")
    start_date="2022-01-01",
    end_date="2022-12-31",
    bands=["red", "green", "blue", "nir"],
    apply_cloud_mask=True # Automatically fetches QA band and natively masks clouds/shadows
)

print(cube) # Returns an xarray.DataArray (Time, Band, Y, X)
```

## Temporal Regularization

Algorithms like TWDTW, SOM, and Deep Learning expect temporally aligned data. `zeit` natively regularizes irregular STAC acquisitions.

```python
from zeit import regularize_time_series

# Aggregate observations into 16-day Medoid composites 
# (Maintains xarray lazy evaluation via Dask graphs)
cube_16d = regularize_time_series(cube, freq="16D", method="medoid")
```

## Change Detection (LandTrendr & CCDC)

Continuous structural monitoring using robust breakpoint and harmonic regression models directly on xarray Datacubes (`zeit.landtrendr(cube)`, `zeit.ccdc(cube)`, or pandas-like accessors such as `cube.zeit.ccdc(...)`).

### LandTrendr (Trajectory-based Disturbance)
Identify structural breakpoints in time-series (e.g., detecting exactly when deforestation occurred). One function, `zeit.landtrendr`, takes a raster file, an in-memory or Dask cube, a NumPy stack or a single pixel's series, and scales LandTrendr to massive datasets using C++ OpenMP and Dask.

```python
import zeit

# 1. An annual index stack: one band per year (years read from band names such as yr1985)
ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")   # georeferenced (time, y, x) cube

# 2. Segment every pixel. direction="loss" (the default) looks for drops of the index
#    (NDVI/NBR vegetation loss); use direction="gain" for indices that rise (e.g. SWIR).
lt = zeit.landtrendr(ndvi, max_segments=6)
# xarray.Dataset: vertex_year, vertex_value (vertex, y, x), n_vertices, rmse (y, x)
# Dask cubes stay lazy; zeit.landtrendr("big.tif", chunks="auto") works block by block.

# 3. Analyze disturbances (e.g., finding the biggest drop in NDVI)
loss = zeit.extract_events(
    lt,
    sort_by="greatest",     # Get the segment with the largest magnitude
    min_magnitude=1500      # Optional noise filter (NDVI x 10000)
)

# Georeferenced maps, ready to be exported to GeoTIFF
yod_map = loss["yod"]         # Year of Disturbance (YOD)
mag_map = loss["magnitude"]   # Magnitude of the disturbance
dur_map = loss["duration"]    # How many years the disturbance took
zeit.save_raster(loss, "lt_rondonia")   # one GeoTIFF per map: yod.tif, magnitude.tif, ...
```

### CCDC / COLD (Harmonic Modeling)
Extracts harmonic coefficients (Intercept, Slopes, Sine, Cosine) and detects intra-annual changes by fitting mathematical curves to multi-spectral data. Like LandTrendr, one function, `zeit.ccdc`, takes a raster file, an in-memory or Dask cube, a NumPy stack or a single pixel's series.

```python
import zeit
from zeit.classify import train_ccdc_classifier, classify_ccdc_stack

# 1. A dense multi-band stack: surface reflectance x 10000 (Blue, Green, Red, NIR, SWIR1, SWIR2)
#    plus a band of Fmask codes (0 clear, 1 water, 2 shadow, 3 snow, 4 cloud, 255 fill)
cube = zeit.load_raster("landsat_sr.tif")   # (time, band, y, x); dates from band names 2020-01-15_blue ...

# 2. Run CCDC on every pixel; the QA band is named, and left out of the spectral bands
segments = zeit.ccdc(cube, qa="fmask", max_segments=6)
# xarray.Dataset: t_start, t_end, t_break (segment, y, x), n_segments (y, x),
# rmse (segment, band, y, x), coefs (segment, band, coef, y, x)
# Dask cubes stay lazy; zeit.ccdc("big.tif", qa="fmask", chunks="auto") works block by block.

first_break = segments.t_break.isel(segment=0)   # date of the first change (NaT = none)
zeit.save_raster(segments, "ccdc_out")           # one GeoTIFF per variable, dates as decimal years

# 3. Generate Synthetic Images (Harmonic Reconstruction)
# Predict what the surface should look like on any arbitrary date without clouds!
synthetic_image = zeit.predict_synthetic_image(segments, "2020-07-15")   # (band, y, x)

# 4. Land Cover Classification using the Harmonic Coefficients
# The coefficients of the first segment as a feature stack: bands blue_a0 ... swir2_b3
zeit.save_raster(segments.coefs.isel(segment=0).fillna(0), "output/ccdc_coefs.tif")

# Train a Random Forest using harmonic coefficients as features
rf_model = train_ccdc_classifier(
    X_train=training_coefs, # Your extracted training samples
    y_train=training_labels, 
    n_estimators=100
)

# Classify the entire coefficient stack into a categorical land cover map block-by-block
# (Handles memory efficiently by reading/writing chunks)
classify_ccdc_stack(
    clf=rf_model,
    coef_stack_path="output/ccdc_coefs.tif",
    output_path="output/land_cover_map.tif",
    chunk_size=512
)
```

## Phenology Extraction

Extract 19 simultaneous phenological metrics (Gu, Zhang, Thresholds, Derivatives, LOS, POP) across massive datasets using optimized C++ curve-fitting models (Beck, Elmore, Gu, Zhang, Asymmetric Gaussian, Double Logistic) over Dask clusters.

> The smoothing, curve-fitting, and metric-extraction methodology is based on the R package [`phenofit`](https://github.com/eco-hydro/phenofit) (Kong *et al.*, 2022, *Methods in Ecology and Evolution*, [doi:10.1111/2041-210X.13870](https://doi.org/10.1111/2041-210X.13870)), reimplemented in C++/Eigen/OpenMP. See the [Phenology tutorial](https://sacridini.github.io/zeit-cdts/tutorials/phenology/#7-references) for the full reference list and a real-world walkthrough.

```python
import zeit

# cube_16d: (time, y, x) DataArray of 16-day NDVI composites; the day numbering
# and the years are read from its dates
pheno = zeit.phenology(
    cube_16d,
    curve="beck",                   # Beck's double logistic
    annual=False,                   # one value per detected season, not per calendar year
    max_seasons=2,                  # up to 2 growing seasons per pixel

    # Smoothing Configuration
    apply_whittaker=False,          # Turn off Whittaker
    apply_hants=True,               # Use HANTS (Fourier-based) instead
    hants_frequencies=3,

    # Fine-Grained Season Control
    min_season_length=90,           # Ignore noisy peaks shorter than 90 days
    min_amplitude=0.2,              # Ignore seasons with less than 0.2 NDVI growth

    n_jobs=-1                       # C++ multithreading (leave cores for OS)
)

# An xarray.Dataset: 19 metrics + R2 + RMSE, each (season, y, x).
# Lazy for a Dask cube: computed by .compute() or while save_raster writes it.
greenup_map = pheno["Greenup"].sel(season=1)    # Zhang's Greenup, first season
zeit.save_raster(pheno, "output/phenology")      # one GeoTIFF per metric
```

**Down-weighting cloud/snow-contaminated observations:** `zeit.qc` decodes a sensor's QA/QC band into per-observation reliability weights in `[0, 1]` (ported from phenofit's `qcFUN.R`), which feed the Whittaker/HANTS smoothing and the iterative curve fit instead of trusting every observation equally:

```python
from zeit.qc import qc_modis_summary

# qa_cube: (time, y, x) MOD13 SummaryQA band, aligned with cube_16d
weights = qc_modis_summary(qa_cube)  # 0=good, 1=marginal, 2=snow/ice, 3=cloudy -> [1.0, 0.5, 0.2, 0.2]

pheno = zeit.phenology(
    cube_16d,
    weights=weights,       # down-weights unreliable observations during smoothing/fitting
    season_retry=True,     # relax the trough threshold once if a pixel finds no season at all
)
```

## Trend Analysis (Mann-Kendall)

Pixel-wise Mann-Kendall trend test + Theil-Sen slope, ported from [`pymannkendall`](https://github.com/mmhs013/pymannkendall) (Hussain & Mahmud, 2019) to a C++/OpenMP backend, with the same Dask distribution strategy as Phenology Extraction. Useful for "is there a statistically significant greening/browning trend at this pixel?" questions on multi-year composite stacks. See the [Mann-Kendall tutorial](https://sacridini.github.io/zeit-cdts/tutorials/mann_kendall/) for the full method comparison (autocorrelation-corrected variants, seasonal test) and a real-world walkthrough.

```python
# annual_ndvi: (time, y, x) DataArray, one max-NDVI composite per year (or a GeoTIFF path)
result = zeit.mann_kendall(
    annual_ndvi,
    method="hamed_rao",  # autocorrelation-corrected (recommended for annual composites)
    alpha=0.05,
)

slope_map = result.slope                      # NDVI change per year
significant = result.h == 1.0                 # statistically significant at alpha=0.05
declining = (result.trend == -1) & significant
zeit.save_raster(result, "output/mann_kendall")   # one GeoTIFF per metric
```

## Change Monitoring (BFAST Monitor)

Pixel-wise near-real-time structural change monitoring, ported from the R package [`bfast`](https://github.com/bfast2/bfast) (Verbesselt *et al.*) to a C++/OpenMP backend, with the same Dask distribution strategy as Mann-Kendall. Unlike LandTrendr/CCDC (retrospective, whole-series segmentation), `bfastmonitor` fits a trend+harmonic model on a stable history period and asks "is a disturbance happening *right now*, in the most recent observations?" — verified bit-for-bit-scale accurate against R's `bfastmonitor()`. See the [BFAST Monitor tutorial](https://sacridini.github.io/zeit-cdts/tutorials/bfast_monitor/) for the full method background and scope (only `type="OLS-MOSUM"` + `history="all"` are ported so far).

```python
# ndvi_16d: (time, y, x) DataArray of 16-day composites from 2010 (or a GeoTIFF path);
# start_time=2010.0 and frequency=23 are read from its dates
result = zeit.bfast_monitor(ndvi_16d, "2022-01-01")   # monitor everything from 2022 onward

disturbed = result.has_break == 1.0
break_time = result.breakpoint  # fractional-year time of the first detected break
```

## Change Detection (BFAST Lite)

Pixel-wise, single-pass multiple-breakpoint detection, ported from the R package `bfast`'s `bfastlite()` and its `strucchangeRcpp` dependency's `breakpoints()` (the Bai & Perron optimal multiple-breakpoint dynamic program) to a C++/OpenMP backend. Unlike `bfastmonitor` above (single break, near-real-time), this retrospectively segments the *whole* series into the optimal number of pieces (via the LWZ model-selection criterion) — no STL decomposition needed. Verified exactly against R's `bfastlite()` across 6 scenarios (~3.1x faster single-threaded than R; `n_jobs=-1` adds a further ~5.5x on top of that by reserving one CPU core and parallelizing across the rest — a smaller gap than `bfastmonitor`'s, since this workload is genuinely CPU-bound dynamic programming on both sides, not dominated by R's per-call overhead). See the [BFAST Lite tutorial](https://sacridini.github.io/zeit-cdts/tutorials/bfast_lite/) for the full method background, scope, and validation details (including two real numerical bugs caught and fixed along the way).

```python
# ndvi_16d: (time, y, x) DataArray of 16-day composites
result = zeit.bfast_lite(ndvi_16d, max_breaks=5)

n_breaks = result.n_breaks
first_break_idx = result.breakpoint_idx_1  # NaN where n_breaks == 0
```

## Time-Series Classification (TWDTW)

Classify pixels by their likeness to a few reference series, one per class, with Time-Weighted Dynamic Time Warping. The distances are those of the R package twdtw (checked to 1e-12), computed in C++ with OpenMP; a pattern of one year matches the same season of any year, and cloudy dates are left out of each pixel's series.

```python
import pandas as pd
import zeit

# Patterns: a typical series per class, indexed by dates (a DataFrame for several bands)
patterns = {"Forest": forest_ndvi, "Soy": soy_ndvi, "Pasture": pasture_ndvi}

ndvi = zeit.load_raster("S2_ndvi_2022.tif")          # (time, y, x)
classes = zeit.twdtw(ndvi, patterns)                  # label, distance, distances

# Pixels that matched poorly with every pattern: unknown (0)
known = classes.label.where(classes.distance < 15, 0)
classes.label.zeit.plot()                             # legend: Forest, Soy, Pasture
zeit.plot(ndvi, fit=classes)                          # click a pixel: its series and best match
```

## Unsupervised Clustering (SOM)

Unsupervised classification and dimensionality reduction of time series with a C++ port of Python `minisom` (online and Batch SOM) that reproduces its results bit-for-bit, 30-190x faster.

```python
from zeit.ai import SOM

# Flatten cube to (Pixels, Features)
X_train = cube_16d.values.reshape(-1, cube_16d.shape[2] * cube_16d.shape[3])

# Train a 10x10 SOM grid (Batch SOM, 20 passes over the data, OpenMP-parallel)
som = SOM(x=10, y=10, input_len=X_train.shape[1], sigma=1.5)
som.random_weights_init(X_train)
som.train(X_train, num_iters=20, algorithm="batch", n_jobs=-1)

# Predict Best Matching Units (BMUs) for new data
bmus = som.predict(X_train, n_jobs=-1)
```

## Pre and Post-Processing

Before classifying, it is highly recommended to smooth temporal trajectories. After classifying, pixel-based maps often suffer from noise. Zeit provides fast functions to regularize your data in both dimensions:

```python
import zeit
from zeit import apply_majority_filter, apply_mmu_filter, save_raster

# Temporal smoothing: Whittaker (uneven dates, NaN gaps filled, per-observation weights)...
smoothed = zeit.smooth(cube, lmbda=10.0, weights=clear_sky_weights)

# ...or Savitzky-Golay, for evenly spaced series
smoothed = zeit.smooth(cube, method="savgol", window=5, polyorder=2)

# Spatial Regularization (Mode filter)
regularized_map = apply_majority_filter(classified_map, size=3)
save_raster(regularized_map, "results/classified_regularized.tif", like=cube)

# Minimum Mapping Unit (MMU): operates on a GeoTIFF on disk, not an in-memory array
# Erase isolated patches smaller than 11 pixels
apply_mmu_filter(
    input_path="results/classified_regularized.tif",
    output_path="results/classified_final.tif",
    mmu_pixels=11,
)
```

## Exporting Geospatial Data

`save_raster` writes any map, stack or result back to disk with its georeferencing, NoData, band names and dates, so GIS software and `load_raster` read it back as it was.

```python
from zeit import save_raster

save_raster(final_map, "output/land_cover.tif", like=cube, nodata=255)   # numpy: CRS and transform from cube
save_raster(ndvi_cube, "output/ndvi.tif")      # a DataArray brings its own georeferencing and dates
save_raster(events, "output/loss")             # a dict/Dataset of maps: one GeoTIFF per map
```

## Command-Line Interface

Every core algorithm is also available as a `zeit` subcommand, so you can run change detection directly on GeoTIFF stacks from bash scripts, cron jobs, or HPC batch systems without writing any Python:

```bash
zeit landtrendr input_stack.tif output_dir/ --max-segments 6 --jobs -1   # years from the band names, or --start-year
```

See the [CLI reference](https://sacridini.github.io/zeit-cdts/cli/) for the full list of subcommands and options.

---

## Architecture & Threading Safety

Zeit safely blends Python-based distributed workflows (Dask) with highly parallel C++ routines:
- **OpenMP CPU Scaling**: All C++ algorithms expose the `n_jobs` parameter, and default to it (`n_jobs=-1`) in their Python entry points. `n_jobs=-1` reserves one CPU core (`std::max(1, max_threads - 1)`) so the host OS stays responsive during intensive workloads — standardized across every parallel algorithm (LandTrendr, CCDC, TWDTW, SOM, Phenology, Mann-Kendall, BFAST Monitor, BFAST Lite). Pass an explicit positive integer to use a specific thread count instead (e.g. all cores with no reservation, or fewer to leave more headroom).
- **Memory Footprint**: Algorithms like TWDTW are strictly optimized via a 2-Row Dynamic Programming algorithm, restricting mathematical matrices to the CPU's L1 cache and avoiding heavy allocations.
- **Cross-Platform Compatibility**: Uses safe `#ifdef _OPENMP` boundaries to gracefully fallback to single-threaded operations on macOS environments using Apple Clang (which lacks native `libomp`), allowing `pip install` to succeed universally.

---

## License

Zeit is free software, licensed under the [GNU General Public License v2.0 or later](https://github.com/sacridini/zeit-cdts/blob/main/LICENSE) (`GPL-2.0-or-later`). Several algorithms are ports of existing open-source implementations (bfast, strucchangeRcpp, GLMnet, pymannkendall, GERSL/CCDC, ...); their origins, licenses and copyright notices are listed in [THIRD_PARTY_NOTICES.md](https://github.com/sacridini/zeit-cdts/blob/main/THIRD_PARTY_NOTICES.md). Because the CCDC lasso solver derives from GPL-2.0-only code, the compiled extension as a whole is distributed under GPL version 2.
