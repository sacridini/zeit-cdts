# Command Line Interface

<p class="lead">Run the core algorithms on GeoTIFF stacks straight from the terminal, with no Python code. Useful for shell scripts, cron jobs and HPC schedulers. Each command reads and writes files block by block, so inputs can be larger than memory.</p>

## Basic Usage

The CLI is structured around subcommands for each algorithm. You can invoke the CLI using the `zeit` command (if installed via pip) or by executing the python module directly:

```bash
zeit <algorithm> [OPTIONS] <input_file> <output_directory>
```
*(Alternatively: `python -m zeit.cli <algorithm> ...`)*

To get general help or see the list of available commands:
```bash
zeit --help
```

---

## 1. LandTrendr (`landtrendr`)

The `landtrendr` command processes a multi-band GeoTIFF representing a time series of a single spectral index (e.g., NBR, NDVI) and extracts spatial change events.

### Syntax
```bash
zeit landtrendr <input> <output_dir> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input multi-band GeoTIFF. Each band must represent a single year, in chronological order. |
| **`output_dir`** | `dirpath` | Directory where the resulting event maps (Year of Detection, Magnitude, Duration) will be saved. |

### Configuration Options

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-year` | `int` | read from the file | The calendar year of the first band. By default the years are read from the band descriptions (e.g. `yr1985`), as `zeit.load_raster` does; pass it for stacks whose bands are not named by year. |
| `--max-segments`| `int` | `6` | The maximum number of line segments the algorithm can fit per pixel. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use for parallel processing. `-1` uses all available cores. |
| `--chunk-size` | `int` | `512` | Size of the image chunks (in pixels) processed simultaneously to manage RAM. |
| `--save-vertices`| `flag` | `False` | If provided, saves the raw multi-band GeoTIFF containing all fitted vertices. |
| `--no-data-value` | `float` | the raster's NoData, or `0` | Value marking a missing year. By default the raster's NoData value, or `0` for integer stacks without one (how Earth Engine exports masked pixels). NaN is always missing. |

### Event Extraction Options

These options control how specific change events are extracted from the temporal trajectory:

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--event-type` | `str` | `loss` | Type of event to segment for and map. Choices: `loss` (value decreases) or `gain` (value increases). It sets both LandTrendr's `direction` and the events extracted. |
| `--sort-by` | `str` | `greatest` | How to select the event if multiple occur. Choices: `greatest`, `newest`, `fastest`, `longest`, `dsnr`. |
| `--min-mag` | `float`| `0.0` | Filter out events with a magnitude lower than this threshold. |
| `--min-dur` | `int` | `1` | Filter out events shorter than this duration in years. |
| `--pre-val-thresh`| `float`| `0.0` | Filter out events if the starting value was already below this threshold. |
| `--output-scale` | `float`| `1.0` | Scale factor applied to the output. Useful for converting scaled integers back to floats (e.g., `0.0001`). |
| `--prefix` | `str` | `lt_event` | Prefix added to all output file names. |

### End-to-End Example
Run LandTrendr on a 30-year NBR stack, extracting the greatest vegetation loss, using all CPU cores, and converting the output back to floating point (assuming NBR was scaled by 10000):
```bash
zeit landtrendr ./data/nbr_stack_1990_2020.tif ./results \
    --start-year 1990 \
    --max-segments 6 \
    --event-type loss \
    --sort-by greatest \
    --output-scale 0.0001 \
    --jobs -1
```

---

## 2. Continuous Change Detection (`ccdc`)

The `ccdc` command runs [`zeit.ccdc`](api/change-detection.md#ccdc) on a dense, multi-band, multi-date GeoTIFF, reading it block by block, and writes each pixel's segments as GeoTIFFs. Values must be surface reflectance × 10,000, bands ordered Blue, Green, Red, NIR, SWIR1, SWIR2 [, thermal].

### Syntax
```bash
zeit ccdc <input> <output_dir> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input GeoTIFF. Either its bands are named `date_band` (`2020-01-15_blue`, `2020-01-15_green`, …, as `zeit.save_raster` writes a `(time, band, y, x)` cube), and the dates are read from them; or it is interleaved by date (Date1-Band1, Date1-Band2, …, Date2-Band1, …) without such names, with `--dates-file` and `--num-bands`. |
| **`output_dir`** | `dirpath` | Directory where the segment rasters will be saved. |

### Configuration Options

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--dates-file` | `str` | *None* | Text file with one date per line, an ISO date (`2020-01-15`) or a Python ordinal day (`737439`), for a stack without dates in its band names. |
| `--num-bands` | `int` | `6` | With `--dates-file`: the number of bands per date in a stack interleaved by date, including the QA band. |
| `--qa-band` | `int` | `-1` | The zero-based index, among the bands of each date, of a band of Fmask codes (`0` clear, `1` water, `2` shadow, `3` snow, `4` cloud, `255` no observation). It is not used as a spectral band. `-1`: no QA band, every observation is clear. |
| `--max-segments`| `int` | `6` | The maximum number of segments to retain per pixel. |
| `--conse` | `int` | `6` | Consecutive anomalous observations that confirm a break (`conseq_anom`), as in the original CCDC. |
| `--chunk-size` | `int` | `512` | Size of the image blocks (in pixels) processed at once. |
| `--jobs` | `int` | `-1` | Number of CPU threads. `-1` uses all cores but one. |
| `--cold` | `flag` | `False` | Deprecated: the same as `--conse 6`, now the default. |
| `--prefix` | `str` | `ccdc` | Prefix added to all output GeoTIFF files. |

### Outputs

| File | Bands | Content |
| :--- | :--- | :--- |
| `<prefix>_t_start.tif`, `<prefix>_t_end.tif` | one per segment | First and last date of each segment, as decimal years (NaN past the last segment). |
| `<prefix>_t_break.tif` | one per segment | Date of the break that ended each segment, as a decimal year (NaN: no break). |
| `<prefix>_n_segments.tif` | 1 | Number of segments per pixel. |
| `<prefix>_rmse.tif` | one per segment and band (`1_blue`, …) | RMSE of each band's fit. |
| `<prefix>_coefs.tif` | one per segment, band and coefficient (`1_blue_a0`, `1_blue_c1`, …) | The harmonic coefficients `a0, c1, a1, b1, a2, b2, a3, b3` (see [`zeit.ccdc`](api/change-detection.md#ccdc)). |

In a stack interleaved by date, the bands are named `b1`, `b2`, … (`1_b1_a0`, …).

### End-to-End Example
Run CCDC on a stack written by `zeit.save_raster`, whose 7th band of each date (index 6) holds the Fmask codes:
```bash
zeit ccdc ./data/landsat_sr.tif ./results \
    --qa-band 6 \
    --max-segments 8 \
    --jobs -1
```

The same on a stack interleaved by date, with 6 spectral bands and 1 QA band per date, and the dates in a text file:
```bash
zeit ccdc ./data/dense_stack.tif ./results \
    --dates-file ./data/dates.txt \
    --num-bands 7 \
    --qa-band 6
```

---

## 3. BFAST Monitor (`bfast-monitor`)

The `bfast-monitor` command runs [`zeit.bfast_monitor`](api/change-detection.md#bfast_monitor) on a multi-band GeoTIFF where every band is one equally-spaced observation (e.g. a 16-day composite), reading it block by block. Time is regular, as in R's `ts`: observation `i` is at `start_time + i / frequency`. Both are read from the band dates (descriptions such as `2020-01-15`, as `zeit.save_raster` writes them); give `--start-time` and `--frequency` for a stack without dates. See the [BFAST Monitor tutorial](tutorials/bfast_monitor.md) for the full method background.

The four time-series commands (`bfast-monitor`, `bfast-lite`, `bfast`, `mann-kendall`) write one GeoTIFF, `<output_dir>/<prefix>.tif`, with one float32 band per metric, named after it.

### Syntax
```bash
zeit bfast-monitor <input> <output_dir> --monitor-start-time <year or date> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input multi-band GeoTIFF. Each band is one equally-spaced time step. |
| **`output_dir`** | `dirpath` | Directory where the single output GeoTIFF (7 bands, see below) will be saved. |

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--monitor-start-time` | `float` or date | *(required)* | When monitoring begins, as a decimal year (`2019.0`) or a date (`2019-01-01`): the boundary between the stable "history" and "monitoring" periods. |
| `--start-time` | `float` | from the band dates | The series' start time (e.g. `2015.0`). |
| `--frequency` | `int` | from the band dates | Observations per year (e.g. `23` for 16-day composites). |
| `--order` | `int` | `3` | Harmonic order for the seasonal regressors. |
| `--h` | `float` | `0.25` | MOSUM window size, as a fraction of history length. Must be one of `0.25`, `0.5`, `1.0`. |
| `--period` | `int` | `10` | Monitoring period parameter. Must be one of `2`, `4`, `6`, `8`, `10`. |
| `--alpha` | `float` | `0.05` | Significance level for the monitoring boundary. |
| `--min-valid` | `int` | `10` | Minimum valid (non-NaN) history observations per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available cores. |
| `--prefix` | `str` | `bfast_monitor` | Filename (without extension) for the output GeoTIFF. |

Output bands (in order): `breakpoint`, `breakpoint_idx`, `magnitude`, `sigma`, `n_history`, `has_break`, `valid` (see [`zeit.bfast_monitor`](api/change-detection.md#bfast_monitor)).

### End-to-End Example
```bash
zeit bfast-monitor ./data/ndvi_16day_stack.tif ./results \
    --monitor-start-time 2019-01-01 \
    --jobs -1

# A stack without dates in its band names: give the time axis
zeit bfast-monitor ./data/ndvi_16day_stack.tif ./results \
    --start-time 2015.0 --frequency 23 --monitor-start-time 2019.0
```

---

## 4. BFAST Lite (`bfast-lite`)

The `bfast-lite` command runs [`zeit.bfast_lite`](api/change-detection.md#bfast_lite): it retrospectively segments the *whole* series into an optimal number of pieces in a single pass - the modern, non-iterative alternative to classic `bfast`. Input and time axis as for `bfast-monitor`. See the [BFAST Lite tutorial](tutorials/bfast_lite.md).

### Syntax
```bash
zeit bfast-lite <input> <output_dir> [OPTIONS]
```

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-time` | `float` | from the band dates | The series' start time (e.g. `2010.0`). |
| `--frequency` | `int` | from the band dates | Observations per year. |
| `--order` | `int` | `3` | Harmonic order. |
| `--h` | `float` | `0.15` | Minimum segment size, as a fraction of the series length. |
| `--max-breaks-output` | `int` | `5` | Maximum number of breakpoints to report per pixel (`max_breaks` in Python). |
| `--min-valid` | `int` | `20` | Minimum valid observations per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. |
| `--prefix` | `str` | `bfast_lite` | Filename (without extension) for the output GeoTIFF. |

Output bands: `n_breaks`, `rss`, `lwz`, `n_valid`, `valid`, `breakpoint_idx_1..N` (see [`zeit.bfast_lite`](api/change-detection.md#bfast_lite)).

### End-to-End Example
```bash
zeit bfast-lite ./data/ndvi_16day_stack.tif ./results --max-breaks-output 5
```

---

## 5. Classic BFAST (`bfast`)

The `bfast` command runs [`zeit.bfast`](api/change-detection.md#bfast), the original iterative `bfast()` algorithm: an STL seasonal seed followed by alternating trend/season segmented regressions, distinguishing trend breaks from seasonal (phenological) breaks. Input and time axis as for `bfast-monitor`. See the [BFAST tutorial](tutorials/bfast.md) for the full method background, scope notes, and R cross-validation results.

### Syntax
```bash
zeit bfast <input> <output_dir> [OPTIONS]
```

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-time` | `float` | from the band dates | The series' start time (e.g. `2000.0`). |
| `--frequency` | `int` | from the band dates | Observations per year. Requires more than `2 * frequency` total observations. |
| `--order` | `int` | `3` | Harmonic order for the season sub-model. |
| `--h` | `float` | `0.15` | Minimum segment size (both trend and season), as a fraction of valid observations. |
| `--max-breaks-trend` | `int` | `5` | Maximum number of trend breakpoints to report per pixel. |
| `--max-breaks-season` | `int` | `5` | Maximum number of season breakpoints to report per pixel. |
| `--max-iter` | `int` | `10` | Maximum trend/season re-estimation iterations. |
| `--level` | `float` | `0.05` | Significance threshold for the preliminary structural-stability pre-check. |
| `--min-valid` | `int` | `20` | Minimum valid observations per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. |
| `--prefix` | `str` | `bfast` | Filename (without extension) for the output GeoTIFF. |

Output bands: `n_trend_breaks`, `n_season_breaks`, `magnitude`, `break_time`, `n_iter`, `n_valid`, `valid`, `trend_breakpoint_idx_1..N`, `season_breakpoint_idx_1..N` (see [`zeit.bfast`](api/change-detection.md#bfast)).

### End-to-End Example
```bash
zeit bfast ./data/ndvi_16day_stack.tif ./results
```

---

## 6. Mann-Kendall Trend Test (`mann-kendall`)

The `mann-kendall` command runs [`zeit.mann_kendall`](api/time-series.md#mann_kendall), the pixel-wise Mann-Kendall trend test and Theil-Sen slope estimator, across a multi-band GeoTIFF (one band per observation - typically one annual composite per band). No dates are needed. See the [Mann-Kendall tutorial](tutorials/mann_kendall.md).

### Syntax
```bash
zeit mann-kendall <input> <output_dir> [OPTIONS]
```

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--method` | `str` | `hamed_rao` | Trend test variant. Choices: `original`, `hamed_rao` (autocorrelation-corrected, recommended), `yue_wang`, `seasonal`. |
| `--alpha` | `float` | `0.05` | Significance level. |
| `--lag` | `int` | *None* | First significant lags for the autocorrelation correction (`hamed_rao`/`yue_wang` only). Defaults to the full series length. |
| `--period` | `int` | `1` | Season slots for `--method seasonal` (e.g. `23` for MODIS 16-day cycles), letting a raw sub-annual series be tested directly. |
| `--min-valid` | `int` | `4` | Minimum valid observations per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. |
| `--prefix` | `str` | `mann_kendall` | Filename (without extension) for the output GeoTIFF. |

Output bands: `trend`, `h`, `p`, `z`, `tau`, `s`, `var_s`, `slope`, `intercept`. `slope`/`intercept` are per time step (per band), so one observation per year gives a directly interpretable per-year trend.

### End-to-End Example
```bash
zeit mann-kendall ./data/annual_ndvi_stack.tif ./results --method hamed_rao --jobs -1
```

---

## 7. Minimum Mapping Unit Filter (`mmu-filter`)

The `mmu-filter` command applies a spatial Minimum Mapping Unit (MMU) filter to a single-band raster - typically a disturbance-year map from `landtrendr` - removing isolated pixel groups smaller than a given size to reduce "salt and pepper" noise.

### Syntax
```bash
zeit mmu-filter <input> <output> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input single-band GeoTIFF (e.g. a `landtrendr` year-of-detection map). |
| **`output`** | `filepath` | Path to save the filtered GeoTIFF. |

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--mmu-pixels` | `int` | `11` | Minimum connected-patch size, in pixels. Smaller patches are zeroed out (treated as the raster's `nodata`/background value). |

### End-to-End Example
```bash
zeit mmu-filter ./results/lt_event_yod.tif ./results/lt_event_yod_mmu.tif --mmu-pixels 9
```

---

## The other cube functions

`phenology`, `smooth`, `tmask`, `twdtw`, `snic`, `classify`, `som`, `coded` and `bandpass-adjust` run the Python function of the same name on any raster [`load_raster`](api/data.md#load_raster) reads, with the dates in the band names (as `save_raster` writes them, `yr1985`, `2020-01-15` or `2020-01-15_red` for a time × band stack). They share the same syntax and options, and write `<output_dir>/<prefix>.tif`, one band per map (or per date), computed block by block in one pass:

```bash
zeit <command> <input> <output_dir> [OPTIONS]
```

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process at once. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. |
| `--prefix` | `str` | the command | Name of the output file, without extension. |

### 8. Phenology (`phenology`)

[`zeit.phenology`](api/time-series.md#phenology): one band per metric (and year). See the [phenology tutorial](tutorials/phenology.md).

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--curve` | `str` | `beck` | `beck`, `elmore`, `gu`, `klos`, `zhang`, `ag` or `dl`. |
| `--method` | `str` | `threshold` | `threshold`, `derivative`, `gu` or `klosterman`. |
| `--weights` | `filepath` | *None* | A raster of observation weights in `[0, 1]` on the input's grid and dates (e.g. written from [`zeit.qc_sentinel2_scl`](api/data.md#qc_sentinel2_scl)). |
| `--not-annual` | flag | | Metrics per season instead of per calendar year. |
| `--max-seasons` | `int` | years | Seasons to report. |
| `--whittaker-lambda` | `float` | `10` | Whittaker smoothing before the fit. |

```bash
zeit phenology ./data/s2_ndvi_2019_2023.tif ./results --curve elmore --weights ./data/s2_weights.tif
```

### 9. Smoothing (`smooth`)

[`zeit.smooth`](api/preprocessing.md#smooth): the smoothed series, same dates and type.

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--method` | `str` | `whittaker` | `whittaker` (uneven dates, gaps filled) or `savgol`. |
| `--lmbda` | `float` | `10` | Whittaker: smoothness. |
| `--weights` | `filepath` | *None* | Whittaker: a raster of observation weights in `[0, 1]`. |
| `--window`, `--polyorder` | `int` | `5`, `2` | Savitzky-Golay: window length and polynomial order. |

```bash
zeit smooth ./data/ndvi_16d.tif ./results --lmbda 20
```

### 10. Cloud and shadow screening (`tmask`)

[`zeit.tmask`](api/preprocessing.md#tmask) on a `(time, band)` stack (bands named `date_band`): one band per date, `1` clear, `0` cloud, shadow or no data.

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--green`, `--swir` | `str` | `green`, `swir1` | Names of the green and SWIR-1 bands. |
| `--scale` | `float` | `10000` | Reflectance scale of the stack (`1` for 0-1 floats). |

```bash
zeit tmask ./data/landsat_sr_stack.tif ./results
```

### 11. TWDTW classification (`twdtw`)

[`zeit.twdtw`](api/time-series.md#twdtw): bands `label`, `distance` and one distance per pattern. The patterns come from a CSV with the columns `pattern`, `date` and one value column (or one column per band, named as the input's bands):

```text
pattern,date,ndvi
soy,2022-10-01,0.25
soy,2022-11-01,0.48
...
pasture,2022-10-01,0.55
```

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--patterns` | `filepath` | required | The CSV of patterns. |
| `--band` | `str` | *None* | A multi-band input with one-band patterns: the band to classify. |
| `--steepness`, `--midpoint` | `float` | `0.1`, `50` | The logistic time weight (midpoint in days). |
| `--max-elapsed` | `float` | *None* | Never match observations farther apart than this many days. |
| `--no-cycle` | flag | | Measure elapsed time between the dates, not between days of the year. |

```bash
zeit twdtw ./data/ndvi_2022.tif ./results --patterns ./data/patterns.csv
```

### 12. SNIC segmentation (`snic`)

[`zeit.snic`](api/time-series.md#snic): the segment labels (the input is read into memory). `--polygons` also writes `<prefix>.gpkg`, the segments as polygons with their means.

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--spacing` | `float` | `10` | Pixels between seeds. |
| `--compactness` | `float` | `0.5` | Larger: more regular segments (in data units). |
| `--grid` | `str` | `rectangular` | `rectangular`, `diamond`, `hexagonal` or `random`. |
| `--polygons` | flag | | Also write the polygons. |

```bash
zeit snic ./data/s2_ndvi_2022.tif ./results --spacing 8 --polygons
```

### 13. Classification (`classify`)

[`zeit.train_classifier`](api/post-processing.md#train_classifier) and [`zeit.classify`](api/post-processing.md#classify) in one call: a random forest trained on sample points (each band, or date and band, a feature) classifies every pixel. Bands `label` (and each class's probability with `--probability`).

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--samples` | `filepath` | *None* | Vector file of points with their class. |
| `--label` | `str` | `class` | Column of the samples holding the class. |
| `--model` | `filepath` | *None* | With `--samples`: where to save the trained model (joblib). Without: a saved model to classify with. |
| `--probability` | flag | | Also write each class's probability. |

```bash
zeit classify ./data/s2_2022.tif ./results --samples ./data/samples.gpkg --model ./results/rf.joblib
zeit classify ./data/s2_2023.tif ./results_2023 --model ./results/rf.joblib      # the same model, next year
```

### 14. SOM clustering (`som`)

[`zeit.som`](api/time-series.md#som): bands `label` and `distance`, and `<prefix>_prototypes.csv` with one row per neuron (`neuron`, its grid position `i`/`j`, `n_pixels`, then the prototype, one column per date or date and band).

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--x`, `--y` | `int` | `3`, `3` | Neurons of the grid. |
| `--sample` | `int` | `50000` | Pixels to train on, at random (`0` for all). |
| `--algorithm` | `str` | `online` | `online` or `batch`. |
| `--sigma`, `--learning-rate` | `float` | `1.0`, `0.5` | Initial neighbourhood radius and learning rate. |
| `--seed` | `int` | `42` | Random seed. |

```bash
zeit som ./data/LT_Stack_NDVI_Rondonia.tif ./results --x 2 --y 2 --sample 30000
```

### 15. Bandpass adjustment (`bandpass-adjust`)

[`zeit.bandpass_adjust`](api/preprocessing.md#bandpass_adjust): the reflectance series with its Sentinel-2 dates on Landsat 8's OLI bands (HLS's bandpass step only), as `<output_dir>/<prefix>.tif` with the input's bands and type. A GeoTIFF does not say which sensor took each date, so `--sensor` does.

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--sensor` | `str` ... | *required* | The sensor of every date (`sentinel-2a`, `S2B`, `landsat-8`...), or one per date. |
| `--etm` | `str` | *None* | `rma` or `ols`: also take Landsat 5/7 dates to OLI (Roy et al. 2016). |
| `--scale` | `float` | auto | Reflectance scale of the stack: 10000 for integers or values above 2, else 1. |

```bash
zeit bandpass-adjust ./data/s2_2023.tif ./results --sensor sentinel-2b --prefix s2_2023_oli
```

### 16. Embeddings (`embeddings`)

[`zeit.load_embeddings`](api/embeddings.md#load_embeddings): download the yearly embeddings of a foundation model over a region into one GeoTIFF, `<output_dir>/<prefix>.tif`, one band per year and dimension (`2024_A00`, `2024_A01`...), which `zeit.load_raster` reads back as a `(time, band, y, x)` cube that remembers which embeddings it holds. Unlike the other commands it takes no input raster: the region is given by `--bbox` or `--region`.

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--source` | `str` | *required* | `tessera` or `alphaearth`. |
| `--bbox` | 4 `float` | *None* | West, south, east, north in longitude and latitude. |
| `--region` | `filepath` | *None* | Vector file of the region; cells outside its polygons are NoData. |
| `--years` | `str` | all | Years, e.g. `2018 2020` or `2018-2024`. |
| `--crs`, `--res` | `str`, `float` | the region's UTM zone, 10 | Output CRS and cell size. |
| `--version`, `--variant`, `--depth` | | `v1.1` | TESSERA: dataset version and variant; the first `DEPTH` dimensions of a v2 store. |
| `--backend` | `str` | `source.coop` | AlphaEarth: the open COGs or `gee` (Earth Engine). |
| `--store` | `filepath` | *None* | A local copy of the product to read instead of the public one. |
| `--cache-dir` | `filepath` | `~/.cache/zeit/embeddings` | Where downloads are kept. |
| `--prefix` | `str` | the source | Name of the output file. |

```bash
zeit embeddings ./results --source tessera --bbox -63.0 -10.0 -62.9 -9.9 --years 2018-2024
zeit classify ./results/tessera.tif ./results --samples ./data/samples.gpkg   # every year and dimension a feature
```

---

## Note on AI Tools (Deep Learning)

Currently, the AI tools (`zeit.ai`) are **not** exposed via the CLI. 

**Why?** 
Deep learning architectures (like UTAE, TempCNN, or Siamese Networks) require highly specific initializations based on your dataset (e.g., number of input bands, number of target classes, path to pre-trained `.pth` weights, and GPU allocation strategies). These configurations are too complex and dynamic to be safely passed as simple terminal arguments.

To use the AI tools, use the [Python API](tutorials/ai.md): `zeit.ai.samples`, `train` and `predict` go from a cube and labelled samples to a map in three calls, and the models can be trained with a loop of your own.
