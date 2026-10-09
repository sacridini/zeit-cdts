# Change Detection

<p class="lead">LandTrendr and CCDC, one function each for every kind of input, and the BFAST family at every level: one pixel, an in-memory array, a GeoTIFF on disk, or a Dask array. The xarray accessor forms are listed in <a href="../xarray/">Xarray Accessor</a>.</p>

## LandTrendr

Tutorial: [LandTrendr](../tutorials/landtrendr.md).

### `landtrendr` { .api #landtrendr }

<!-- sig: zeit.landtrendr -->
```python
zeit.landtrendr(
    data, years=None, direction="loss", band=None, max_segments=6,
    pval_threshold=0.05, recovery_threshold=0.25,
    prevent_fast_recovery=True, spike_threshold=0.9,
    best_model_proportion=0.75, vertex_count_overshoot=3,
    min_observations_needed=6, nodata="auto", fitted=False,
    chunks=None, n_jobs=-1,
)
```

Segments annual time series with LandTrendr (Kennedy et al. 2010), every pixel in parallel in C++ / OpenMP. One function for every input: it reads what `data` is and returns the same `xarray.Dataset` of vertices, georeferenced when the input is. All parameters but `data` are keyword-only. It replaces `run_landtrendr`, `run_landtrendr_array`, `run_landtrendr_image` and `DataArray.zeit.run_landtrendr`; the accessor form is now [`DataArray.zeit.landtrendr`](xarray.md#landtrendr).

`data` can be:

| Input | Example | Where the years come from |
| :--- | :--- | :--- |
| A raster file, folder, glob or Zarr/NetCDF store: anything [`load_raster`](data.md#load_raster) reads | `"LT_Stack_NDVI_Rondonia.tif"` | The dates `load_raster` finds (band descriptions such as `yr1985`, file names, a `time` coordinate) |
| A `(time, y, x)` cube, numpy- or dask-backed | the output of `load_raster` | Its `time` coordinate. A dask cube stays lazy. |
| A `(time, band, y, x)` cube or a `Dataset` | `cube` with `band="NBR"` | Its `time` coordinate; `band` names the index |
| A numpy array `(time, y, x)` | `stack` | `years=` |
| One pixel's series: a list, 1-D array or `pandas.Series` | `[5200, 5100, 2100, ...]` | `years=`, or the Series' index of dates or years |

LandTrendr needs one value per year: a series whose years repeat raises an error asking for annual composites first ([`build_annual_composites`](data.md#build_annual_composites), or `regularize_time_series(cube, freq="1YS")`).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | What to segment (see the table above). |
| `years` | 1-D array | `None` | Year of each time step. Only for numpy input and cubes without dates; otherwise taken from the `time` coordinate. |
| `direction` | `str` | `"loss"` | The change to look for. `"loss"`: drops of the index, such as vegetation loss on NDVI, NBR, EVI or wetness. `"gain"`: rises, for indices that rise with disturbance (SWIR, brightness) or to map regrowth. Output values are always in the original scale. |
| `band` | `str` or `int` | `None` | The index to segment in a `(time, band, y, x)` cube or a `Dataset` (band or variable name). |
| `max_segments` | `int` | `6` | Maximum segments per pixel. |
| `pval_threshold` | `float` | `0.05` | Maximum p-value of an accepted model. |
| `recovery_threshold` | `float` | `0.25` | Rejects recoveries faster than `1 / recovery_threshold` years. |
| `prevent_fast_recovery` | `bool` | `True` | Kept for compatibility; has no effect (the recovery check always applies, as in the original). |
| `spike_threshold` | `float` | `0.9` | Spike dampening ([`desawtooth`](preprocessing.md#desawtooth)). `1.0` disables it. |
| `best_model_proportion` | `float` | `0.75` | Prefer the model with more vertices whose p-value is at most `(2 - best_model_proportion)` times the best, as in the original. `0.75` accepts models within 1.25× of the best p-value. Values above `1` make the rule stricter than the best model itself, so most pixels end up as a flat line. |
| `vertex_count_overshoot` | `int` | `3` | Extra candidate vertices before pruning. |
| `min_observations_needed` | `int` | `6` | Pixels with fewer valid years are not segmented. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing year, which is left out of the fit (NaN always is). `"auto"`: the raster's NoData value; for integer data without one, `0` (how Earth Engine exports masked pixels). A number: that value. `None`: only NaN. |
| `fitted` | `bool` | `False` | Also return the fitted trajectory, `fitted (time, y, x)`: the series rebuilt from the vertices (LT-GEE's fitted values). |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads the raster into memory; `"auto"` or a dict of chunk sizes keeps it lazy, so the result is computed block by block, for rasters larger than memory. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Dims | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `vertex_year` | `(vertex, y, x)` | int16 | Year of each vertex; `0` past the last one. |
| `vertex_value` | `(vertex, y, x)` | float32 | Fitted value at each vertex; NaN past the last one. |
| `n_vertices` | `(y, x)` | uint8 | Number of vertices (`0`: not segmented). |
| `rmse` | `(y, x)` | float32 | RMSE of the fit: the noise estimate behind `dsnr` in `extract_events`. |
| `fitted` | `(time, y, x)` | float32 | Only with `fitted=True`: the fitted trajectory. |

The `x`/`y` coordinates and CRS of the input are kept, so the result goes straight to [`save_raster`](data.md#save_raster). `attrs` records the parameters, `direction` among them (the default `event_type` of `extract_events`). A single pixel's result has no `y`/`x` dims. With a dask cube or `chunks=`, the result is lazy.

```python
ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")    # (time, y, x), bands yr1985 ... yr2024
lt = zeit.landtrendr(ndvi)                                # looks for NDVI drops (direction="loss")
loss = zeit.extract_events(lt, min_magnitude=1500)        # greatest loss per pixel
zeit.save_raster(loss, "lt_rondonia")                     # one GeoTIFF per metric

# A raster larger than memory: lazy, computed block by block while it is written
lt = zeit.landtrendr("nbr_1985_2024.tif", chunks="auto")
zeit.save_raster(zeit.extract_events(lt, min_magnitude=2000), "lt_nbr")

# One pixel's series
px = zeit.landtrendr([8100, 8000, 8200, 8050, 3100, 4200, 5300, 6100, 6800, 7300, 7500, 7600],
                     years=range(2010, 2022))
px.vertex_year.values      # array([2010, 2013, 2014, 2018, 2021, 0, 0], dtype=int16)
```

### `extract_events` { .api }

<!-- sig: zeit.metrics.extract_events -->
```python
zeit.metrics.extract_events(
    lt, event_type=None, sort_by="greatest", min_magnitude=0.0,
    min_duration=1, pre_val_threshold=0.0, rmse_map=None,
)
```

Turns LandTrendr's segments into maps of one event per pixel, such as the greatest loss. Also exported as `zeit.extract_events`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `lt` | `xr.Dataset` | required | The result of [`landtrendr`](#landtrendr), in memory or dask. A numpy vertex stack `(2 × vertices, rows, cols)`, years in the first half and values in the second, is also accepted. |
| `event_type` | `str` | `None` | `"loss"` (index fell) or `"gain"` (index rose). Default: the `direction` LandTrendr ran with (`"loss"` for a numpy stack). |
| `sort_by` | `str` | `"greatest"` | Which event to keep: `"greatest"` (magnitude), `"newest"`, `"fastest"`, `"longest"` or `"dsnr"`. |
| `min_magnitude` | `float` | `0.0` | Ignore events smaller than this, in data units. A flat segment (magnitude `0`) is never an event. |
| `min_duration` | `int` | `1` | Ignore events shorter than this many years. |
| `pre_val_threshold` | `float` | `0.0` | For losses, ignore events starting below this value (for gains, above). `0` disables. |
| `rmse_map` | `np.ndarray` | `None` | Numpy stacks only: `(rows, cols)` RMSE of each pixel's fit, which adds the `dsnr` output. A LandTrendr Dataset brings its own (`rmse`). |

</div>

**Returns** an `xarray.Dataset` on the same grid and CRS as `lt` (lazy if `lt` is), ready for `save_raster`:

| Variable | Meaning |
| :--- | :--- |
| `yod` | Year of the vertex where the event starts, i.e. the last year before it (`0` = no event). The first year in which the change is visible is `yod + 1`. |
| `magnitude` | Size of the change, in data units. |
| `duration` | Years the change took. `1` is abrupt. |
| `pre_val`, `post_val` | Fitted value before and after. |
| `rate` | `magnitude / duration`. |
| `dsnr` | Magnitude divided by the fit's RMSE (LT-GEE's disturbance signal-to-noise ratio). Values above 2–3 are rarely noise. |

For a numpy vertex stack, a dict of `(rows, cols)` arrays with the same keys (`dsnr` only with `rmse_map`).

```python
lt = zeit.landtrendr(ndvi)
loss = zeit.extract_events(lt, min_magnitude=2000)
first_year_of_loss = (loss.yod + 1).where(loss.yod > 0)
regrowth = zeit.extract_events(lt, event_type="gain", sort_by="newest")   # same fit, rises
```

### `apply_vertices` { .api }

<!-- sig: zeit.apply_vertices -->
```python
zeit.apply_vertices(vertex_years, other_band_years, other_band_values)
```

"Fit to vertices": describes another band with the vertex years found on the primary index, by interpolating that band at those years.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `vertex_years` | 1-D array | required | Vertex years from the primary index. |
| `other_band_years` | 1-D array | required | Years of the other band's series. |
| `other_band_values` | 1-D array | required | The other band's values. |

</div>

**Returns** a list of vertices, `[{"year": 1985, "value": ...}, ...]`.

```python
px = zeit.landtrendr(nbr, years=years)                     # one pixel's NBR series
vertex_years = px.vertex_year.values[: int(px.n_vertices)]
ftv = zeit.apply_vertices(vertex_years, years, swir1)      # the same pixel's SWIR1
```

## CCDC

Tutorial: [CCDC](../tutorials/ccdc.md).

### `ccdc` { .api #ccdc }

<!-- sig: zeit.ccdc -->
```python
zeit.ccdc(
    data, qa=None, dates=None, bands=None, max_segments=6,
    conseq_anom=6, chi2_prob_threshold=0.99,
    tmax_cg_prob_threshold=0.999999, detection_bands=None, num_c=8,
    tmask_bands=None, thermal_band=None, valid_range=(0.0, 10000.0),
    thermal_range=(-9320.0, 7070.0), nodata="auto", chunks=None,
    n_jobs=-1,
)
```

Continuous Change Detection and Classification (Zhu & Woodcock 2014): fits a harmonic model to every stable period of each pixel's multi-band series and dates the breaks between them, every pixel in parallel in C++ / OpenMP. The engine is a port of the original MATLAB code, validated segment for segment against it. One function for every input: it reads what `data` is and returns the same `xarray.Dataset` of segments, georeferenced when the input is. All parameters but `data` are keyword-only. It replaces `zeit.ccdc.run_ccdc`, `run_ccdc_array`, `run_ccdc_image` and `DataArray.zeit.run_ccdc`; the accessor form is now [`DataArray.zeit.ccdc`](xarray.md#ccdc).

`data` can be:

| Input | Example | Where the dates and bands come from |
| :--- | :--- | :--- |
| A `(time, band, y, x)` cube, numpy- or dask-backed | the output of `load_raster` or `build_time_series` | Its `time` and `band` coordinates. A dask cube stays lazy. |
| A raster file whose bands are named `date_band`, or anything [`load_raster`](data.md#load_raster) reads | `"landsat_sr.tif"` (bands `2008-01-05_blue`, …, as `save_raster` writes a cube), a folder read with `pattern=` | The dates and band names `load_raster` finds |
| A raster interleaved by date without band names (all bands of date 1, then of date 2, …) | `"stack.tif"` | `dates=`, and `bands=`: the names of the bands of each date |
| A numpy array `(time, band, y, x)` | `stack` | `dates=` |
| One pixel: a `(time, band)` DataArray, or a `pandas.DataFrame` indexed by date with one column per band | `cube.isel(y=0, x=0)` | Its `time` coordinate or index |

Values are **surface reflectance × 10,000** (the original's thresholds, range test and cloud screen are defined on that scale), bands ordered Blue, Green, Red, NIR, SWIR1, SWIR2 [, brightness temperature], unless `bands=` orders them. Dates can be any dates; they are converted to ordinal days internally.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `ndarray` or `pd.DataFrame` | required | What to run on (see the table above). |
| `qa` | `str`, `DataArray` or `ndarray` | `None` | Fmask codes per date: `0` clear land, `1` water, `2` cloud shadow, `3` snow, `4` cloud, `255` no observation. The name of a band of the cube (it is then left out of the spectral bands), a `(time, y, x)` cube or array (`(time,)` for one pixel), or `None`: every date is clear, and dates with NaN are no observation. |
| `dates` | list | `None` | Date of each time step (strings, datetimes, `datetime64`). Only for numpy input and cubes or rasters without dates; otherwise taken from the `time` coordinate. |
| `bands` | list of `str` or `int` | `None` | Spectral bands to use, by name or 0-based position, in the order above. For a raster interleaved by date without band names: the names of the bands of each date (including the QA band, if any). |
| `max_segments` | `int` | `6` | Maximum segments returned per pixel. |
| `conseq_anom` | `int` | `6` | Consecutive anomalous observations that confirm a break (the original's `conse`). |
| `chi2_prob_threshold` | `float` | `0.99` | Change probability for the chi-squared threshold. Lower is more sensitive. |
| `tmax_cg_prob_threshold` | `float` | `0.999999` | Outlier probability: observations beyond it are dropped as noise (for example missed clouds). |
| `detection_bands` | list of `str` or `int` | `None` | Bands used for change detection, by name or 0-based position among the spectral bands. Default: Green to SWIR2. |
| `num_c` | `int` | `8` | Maximum number of harmonic coefficients (4, 6 or 8). |
| `tmask_bands` | list of `str` or `int` | `None` | Bands of the internal Tmask cloud screen. Default: Green and SWIR1. |
| `thermal_band` | `str` or `int` | `None` | The brightness-temperature band (°C × 100), if any. |
| `valid_range` | `tuple` | `(0.0, 10000.0)` | Valid range of the optical bands. |
| `thermal_range` | `tuple` | `(-9320.0, 7070.0)` | Valid range of the thermal band. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. `"auto"`: the raster's NoData value; for integer data without one, `0`. A number: that value. `None`: only NaN. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads the raster into memory; `"auto"` or a dict of chunk sizes keeps it lazy, so the result is computed block by block, for rasters larger than memory. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Dims | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `t_start`, `t_end` | `(segment, y, x)` | datetime64 | First and last date of each segment; `NaT` past the last one. |
| `t_break` | `(segment, y, x)` | datetime64 | Date of the break that ended the segment; `NaT` if it ended without one, and past the last segment. |
| `n_segments` | `(y, x)` | uint8 | Number of segments (`0`: no model). |
| `rmse` | `(segment, band, y, x)` | float32 | RMSE of each band's fit; NaN past the last segment. |
| `coefs` | `(segment, band, coef, y, x)` | float32 | The harmonic model of each band, `coef` = `a0`, `c1`, `a1`, `b1`, `a2`, `b2`, `a3`, `b3`; NaN past the last segment. |

The model is `a0 + c1·t + a1 cos(wt) + b1 sin(wt) + a2 cos(2wt) + b2 sin(2wt) + a3 cos(3wt) + b3 sin(3wt)` on the original's time axis: `t` = ordinal day + 366 (MATLAB datenum), `w` = 2π / 365.25. [`predict_synthetic_image`](#predict_synthetic_image) evaluates it on any date. The `x`/`y` coordinates and CRS of the input are kept, so the result goes straight to [`save_raster`](data.md#save_raster): one GeoTIFF per variable, dates as decimal years (NaN: none), and `coefs.tif` with one band per segment, band and coefficient (`1_blue_a0`, `1_blue_c1`, …). `attrs` records the parameters. A single pixel's result has no `y`/`x` dims. With a dask cube or `chunks=`, the result is lazy.

```python
cube = zeit.load_raster("landsat_sr.tif")          # (time, band, y, x): blue ... swir2 and fmask
segments = zeit.ccdc(cube, qa="fmask")             # QA band by name, left out of the spectral bands
first_break = segments.t_break.isel(segment=0)     # NaT where nothing changed
july = zeit.predict_synthetic_image(segments, "2020-07-01")   # (band, y, x)
zeit.save_raster(segments, "ccdc_out")             # t_start.tif, ..., coefs.tif

# A raster larger than memory: lazy, computed block by block while it is written
segments = zeit.ccdc("landsat_sr.tif", qa="fmask", chunks="auto")
zeit.save_raster(segments, "ccdc_out")

# A stack interleaved by date without band names
segments = zeit.ccdc("stack.tif", dates=dates, qa="fmask",
                     bands=["blue", "green", "red", "nir", "swir1", "swir2", "fmask"])

# One pixel: a DataFrame indexed by date, one column per band
px = zeit.ccdc(df, qa="fmask")
px[["t_start", "t_end", "t_break"]].to_dataframe().dropna(subset=["t_start"])
```

### `predict_synthetic_image` { .api }

<!-- sig: zeit.predict_synthetic_image -->
```python
zeit.predict_synthetic_image(
    segments, date=None, num_bands=6, target_julian_day=None,
)
```

Evaluates each pixel's CCDC model on a date, giving a cloud-free image for any day, including days without an acquisition. Each pixel uses the segment that covers the date; before the first segment, the first one; after the last, the last one.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `segments` | `xr.Dataset` | required | The result of [`ccdc`](#ccdc), in memory or dask (a lazy result gives a lazy image). |
| `date` | `str`, `datetime`, `datetime64` or `int` | required | The date to predict; an `int` is a Python ordinal day. |
| `num_bands` | `int` | `6` | Legacy numpy stacks only (below): number of bands. |
| `target_julian_day` | `int` | `None` | Former name of `date`. |

</div>

**Returns** a `(band, y, x)` `DataArray` on the grid and CRS of `segments` (`(band,)` for one pixel), NaN where a pixel has no model.

```python
july = zeit.predict_synthetic_image(segments, "2019-07-15")
zeit.save_raster(july, "synthetic_2019-07-15.tif")

# One pixel's curve, on the first day of every month
px = zeit.ccdc(df, qa="fmask")
days = pd.date_range("2010-01-01", "2020-12-01", freq="MS")
nir = [float(zeit.predict_synthetic_image(px, d).sel(band="nir")) for d in days]
```

The numpy layout of older versions, `(max_segments, 3 + 9 × bands, rows, cols)` (per segment `t_start`, `t_end` and `t_break` in ordinal days, then each band's RMSE and 8 coefficients), is still accepted: `predict_synthetic_image(stack, target_julian_day=day, num_bands=6)` returns a `(num_bands, rows, cols)` float32 array.

## BFAST family

All three take a regular series: observation `i` is at `start_time + i / frequency`. Tutorials: [BFAST](../tutorials/bfast.md), [BFAST Monitor](../tutorials/bfast_monitor.md), [BFAST Lite](../tutorials/bfast_lite.md).

### `run_bfast_monitor_dask` { .api }

<!-- sig: zeit.bfast.run_bfast_monitor_dask -->
```python
zeit.bfast.run_bfast_monitor_dask(
    arr, start_time, monitor_start_time, frequency, order=3, h=0.25,
    period=10, alpha=0.05, min_valid=10, n_jobs=-1,
)
```

Near-real-time monitoring (`bfastmonitor`, OLS-MOSUM, `history="all"`) for each pixel of a `(time, y, x)` Dask array.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `start_time` | `float` | required | Time of the first observation (a whole year, e.g. `2010.0`). |
| `monitor_start_time` | `float` | required | Start of the monitoring period. |
| `frequency` | `int` | required | Observations per year. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.25` | MOSUM window: `0.25`, `0.5` or `1.0`. |
| `period` | `int` | `10` | Boundary horizon: `2`, `4`, `6`, `8` or `10`. |
| `alpha` | `float` | `0.05` | Significance level. |
| `min_valid` | `int` | `10` | Minimum valid history observations. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `(7, y, x)`: `breakpoint`, `breakpoint_idx`, `magnitude`, `sigma`, `n_history`, `has_break`, `valid` (`zeit.bfast.BFM_METRIC_NAMES`).

### `run_bfast_lite_dask` { .api }

<!-- sig: zeit.bfast.run_bfast_lite_dask -->
```python
zeit.bfast.run_bfast_lite_dask(
    arr, start_time, frequency, order=3, h=0.15, max_breaks_output=5,
    min_valid=20, n_jobs=-1,
)
```

Optimal multiple breakpoints (`bfastlite`, LWZ criterion) for each pixel.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `start_time`, `frequency` | | required | Regular time axis. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.15` | Minimum segment size, fraction of the observations. |
| `max_breaks_output` | `int` | `5` | Break slots in the output. |
| `min_valid` | `int` | `20` | Minimum valid observations. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `(5 + max_breaks_output, y, x)`: `n_breaks`, `rss`, `lwz`, `n_valid`, `valid`, `breakpoint_idx_1…` (`zeit.bfast.bfl_metric_names(max_breaks_output)`).

### `run_bfast_dask` { .api }

<!-- sig: zeit.bfast.run_bfast_dask -->
```python
zeit.bfast.run_bfast_dask(
    arr, start_time, frequency, order=3, h=0.15, max_breaks_trend=5,
    max_breaks_season=5, max_iter=10, level=0.05, min_valid=20,
    n_jobs=-1,
)
```

Classic iterative BFAST (trend and seasonal breaks) for each pixel.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `arr` | `dask.array.Array` | required | `(time, y, x)`; NaN for gaps. |
| `start_time`, `frequency` | | required | Regular time axis. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.15` | Minimum segment size. |
| `max_breaks_trend`, `max_breaks_season` | `int` | `5`, `5` | Break slots in the output. |
| `max_iter` | `int` | `10` | Maximum trend/season iterations. |
| `level` | `float` | `0.05` | Significance of the stability pre-test (`1.0` always searches). |
| `min_valid` | `int` | `20` | Minimum valid observations. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** `n_trend_breaks`, `n_season_breaks`, `magnitude`, `time`, `n_iter`, `n_valid`, `valid`, then the trend and season break indices (`zeit.bfast.bf_metric_names(max_breaks_trend, max_breaks_season)`).

### GeoTIFF versions

`run_bfast_monitor_image`, `run_bfast_lite_image` and `run_bfast_image` take the same parameters as their Dask counterparts, with `input_path`, `output_dir`, `chunk_size` and `prefix` instead of `arr`. The input has one band per time step; the output is `<output_dir>/<prefix>.tif` with one band per metric (band descriptions set to the metric names). All three are exported at the top level and available in the [CLI](../cli.md).

```python
zeit.run_bfast_monitor_image("ndvi_16d.tif", "results/", start_time=2010.0,
                             monitor_start_time=2022.0, frequency=23)
```
