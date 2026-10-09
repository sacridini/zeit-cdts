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

The `ccdc` command runs the Continuous Change Detection and Classification algorithm (or its more conservative variant, COLD) on a highly dense, multi-band, and multi-date GeoTIFF stack.

### Syntax
```bash
zeit ccdc <input> <output_dir> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input stacked GeoTIFF. Bands must be interleaved by date (e.g., Date1-Band1, Date1-Band2, Date2-Band1, etc.). |
| **`output_dir`** | `dirpath` | Directory where the resulting harmonic coefficients and structural break dates will be saved. |

### Configuration Options

| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--num-bands` | `int` | `6` | The number of spectral bands provided per observation date. |
| `--qa-band` | `int` | `-1` | The zero-based index of the Quality Assessment (QA) mask band within the block of bands for a single date. `-1` disables QA masking. |
| `--dates-file` | `str` | *None* | Path to a plain text file containing the observation dates (one integer per line, usually in Julian days). If omitted, assumes a 16-day Landsat interval. |
| `--max-segments`| `int` | `6` | The maximum number of distinct change segments to retain per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available cores. |
| `--cold` | `flag` | `False` | Switches logic to the **COLD** algorithm variant, increasing the required consecutive anomalies for a break from 3 to 6. |
| `--prefix` | `str` | `ccdc` | Prefix added to all output GeoTIFF files. |

### End-to-End Example
Run CCDC on an image with 6 spectral bands and 1 QA band (7 total bands per date). The 7th band (index 6) is the QA mask. The dates are provided in a text file:
```bash
zeit ccdc ./data/dense_stack.tif ./results \
    --num-bands 7 \
    --qa-band 6 \
    --dates-file ./data/dates.txt \
    --max-segments 8 \
    --jobs -1
```

Run the COLD algorithm on the exact same dataset:
```bash
zeit ccdc ./data/dense_stack.tif ./results_cold \
    --num-bands 7 \
    --qa-band 6 \
    --dates-file ./data/dates.txt \
    --cold \
    --jobs -1
```

---

## 3. BFAST Monitor (`bfast-monitor`)

The `bfast-monitor` command runs near-real-time structural change monitoring on a multi-band GeoTIFF where every band is one equally-spaced observation (e.g. a 16-day composite), not a real calendar date - time is synthetic and regular, given by `--start-time` and `--frequency` (matching R's `ts`/`time()` semantics). See the [BFAST Monitor tutorial](tutorials/bfast_monitor.md) for the full method background.

### Syntax
```bash
zeit bfast-monitor <input> <output_dir> --start-time <float> --monitor-start-time <float> --frequency <int> [OPTIONS]
```

### Positional Arguments
| Argument | Type | Description |
| :--- | :---: | :--- |
| **`input`** | `filepath` | Path to the input multi-band GeoTIFF. Each band is one equally-spaced time step. |
| **`output_dir`** | `dirpath` | Directory where the single output GeoTIFF (7 bands, see below) will be saved. |

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-time` | `float` | *(required)* | The series' start time (e.g. `2015.0`). |
| `--monitor-start-time` | `float` | *(required)* | The time monitoring begins (e.g. `2019.0`) - the boundary between the stable "history" and "monitoring" periods. |
| `--frequency` | `int` | *(required)* | Observations per year (e.g. `23` for 16-day composites). |
| `--order` | `int` | `3` | Harmonic order for the seasonal regressors. |
| `--h` | `float` | `0.25` | MOSUM window size, as a fraction of history length. Must be one of `0.25`, `0.5`, `1.0`. |
| `--period` | `int` | `10` | Monitoring period parameter. Must be one of `2`, `4`, `6`, `8`, `10`. |
| `--alpha` | `float` | `0.05` | Significance level for the monitoring boundary. |
| `--min-valid` | `int` | `10` | Minimum valid (non-NaN) history observations per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. `-1` uses all available cores. |
| `--prefix` | `str` | `bfast_monitor` | Filename (without extension) for the output GeoTIFF. |

Output bands (in order): `breakpoint`, `breakpoint_idx`, `magnitude`, `sigma`, `n_history`, `has_break`, `valid` (see `zeit.bfast.BFM_METRIC_NAMES`).

### End-to-End Example
```bash
zeit bfast-monitor ./data/ndvi_16day_stack.tif ./results \
    --start-time 2015.0 \
    --monitor-start-time 2019.0 \
    --frequency 23 \
    --jobs -1
```

---

## 4. BFAST Lite (`bfast-lite`)

The `bfast-lite` command retrospectively segments the *whole* series into an optimal number of pieces in a single pass - the modern, non-iterative alternative to classic `bfast`. See the [BFAST Lite tutorial](tutorials/bfast_lite.md).

### Syntax
```bash
zeit bfast-lite <input> <output_dir> --start-time <float> --frequency <int> [OPTIONS]
```

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-time` | `float` | *(required)* | The series' start time (e.g. `2010.0`). |
| `--frequency` | `int` | *(required)* | Observations per year. |
| `--order` | `int` | `3` | Harmonic order. |
| `--h` | `float` | `0.15` | Minimum segment size, as a fraction of the series length. |
| `--max-breaks-output` | `int` | `5` | Maximum number of breakpoints to report (and search for) per pixel. |
| `--min-valid` | `int` | `20` | Minimum valid observations per pixel. |
| `--chunk-size` | `int` | `512` | Size of the image chunks to process simultaneously. |
| `--jobs` | `int` | `-1` | Number of CPU cores to use. |
| `--prefix` | `str` | `bfast_lite` | Filename (without extension) for the output GeoTIFF. |

Output bands: `n_breaks`, `rss`, `lwz`, `n_valid`, `valid`, `breakpoint_idx_1..N` (see `zeit.bfast.bfl_metric_names`).

### End-to-End Example
```bash
zeit bfast-lite ./data/ndvi_16day_stack.tif ./results --start-time 2010.0 --frequency 23 --max-breaks-output 5
```

---

## 5. Classic BFAST (`bfast`)

The `bfast` command runs the original iterative `bfast()` algorithm: an STL seasonal seed followed by alternating trend/season segmented regressions, distinguishing trend breaks from seasonal (phenological) breaks. See the [BFAST tutorial](tutorials/bfast.md) for the full method background, scope notes, and R cross-validation results.

### Syntax
```bash
zeit bfast <input> <output_dir> --start-time <float> --frequency <int> [OPTIONS]
```

### Configuration Options
| Option | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--start-time` | `float` | *(required)* | The series' start time (e.g. `2000.0`). |
| `--frequency` | `int` | *(required)* | Observations per year. Requires more than `2 * frequency` total observations. |
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

Output bands: `n_trend_breaks`, `n_season_breaks`, `magnitude`, `time`, `n_iter`, `n_valid`, `valid`, `trend_breakpoint_idx_1..N`, `season_breakpoint_idx_1..N` (see `zeit.bfast.bf_metric_names`).

### End-to-End Example
```bash
zeit bfast ./data/ndvi_16day_stack.tif ./results --start-time 2000.0 --frequency 23
```

---

## 6. Mann-Kendall Trend Test (`mann-kendall`)

The `mann-kendall` command runs the pixel-wise Mann-Kendall trend test and Theil-Sen slope estimator across a multi-band GeoTIFF (one band per observation - typically one annual composite per band). See the [Mann-Kendall tutorial](tutorials/mann_kendall.md).

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

Output bands: `trend`, `h`, `p`, `z`, `tau`, `s`, `var_s`, `slope`, `intercept` (see `zeit.trend.MK_METRIC_NAMES`). `slope`/`intercept` are per time step (per band), so one observation per year gives a directly interpretable per-year trend.

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

## Note on AI Tools (Deep Learning)

Currently, the AI tools (`zeit.ai`) are **not** exposed via the CLI. 

**Why?** 
Deep learning architectures (like UTAE, TempCNN, or Siamese Networks) require highly specific initializations based on your dataset (e.g., number of input bands, number of target classes, path to pre-trained `.pth` weights, and GPU allocation strategies). These configurations are too complex and dynamic to be safely passed as simple terminal arguments.

To use the AI tools, please utilize the [Python API](tutorials/ai.md) which allows full flexibility in defining PyTorch DataLoaders, Loss Functions, and Training Loops.
