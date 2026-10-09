# Change Detection

<p class="lead">LandTrendr, CCDC and the BFAST family, at every level: one pixel, an in-memory array, a GeoTIFF on disk, or a Dask array. The xarray accessor forms are listed in <a href="../xarray/">Xarray Accessor</a>.</p>

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

Tutorial: [CCDC](../tutorials/ccdc.md). All CCDC functions expect **surface reflectance × 10,000**, **Python ordinal days**, and **Fmask QA codes** (`0` clear, `1` water, `2` shadow, `3` snow, `4` cloud, `255` fill).

### `run_ccdc` { .api }

<!-- sig: zeit.ccdc.run_ccdc -->
```python
zeit.ccdc.run_ccdc(
    dates, values, qa, min_obs=12, conseq_anom=6,
    chi2_prob_threshold=0.99, tmax_cg_prob_threshold=0.999999,
    detection_bands=None, num_c=8, tmask_bands=None,
    thermal_band=None, valid_range=(0.0, 10000.0),
    thermal_range=(-9320.0, 7070.0),
)
```

CCDC for a single pixel.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dates` | 1-D array | required | Ordinal days. |
| `values` | array | required | `(bands, dates)`: Blue, Green, Red, NIR, SWIR1, SWIR2 [, thermal]. |
| `qa` | 1-D array | required | Fmask codes per date. |
| `min_obs` | `int` | `12` | Kept for compatibility; the original fixes it at 12. |
| `conseq_anom` | `int` | `6` | Consecutive anomalies that confirm a break. |
| `chi2_prob_threshold` | `float` | `0.99` | Change probability for the chi-squared threshold. |
| `tmax_cg_prob_threshold` | `float` | `0.999999` | Outlier probability. |
| `detection_bands` | `list[int]` | `None` | Bands used for detection. Default: Green to SWIR2 (`[1, 2, 3, 4, 5]`). |
| `num_c` | `int` | `8` | Maximum coefficients (4, 6 or 8). |
| `tmask_bands` | `list[int]` | `None` | Bands for the internal Tmask screen. Default `[1, 4]`. |
| `thermal_band` | `int` | `None` | Index of a brightness-temperature band (°C × 100). |
| `valid_range` | `tuple` | `(0.0, 10000.0)` | Valid range of the optical bands. |
| `thermal_range` | `tuple` | `(-9320.0, 7070.0)` | Valid range of the thermal band. |

</div>

**Returns** a list of models, one dict each: `t_start`, `t_end`, `t_break` (`0` if none), `coefs` (bands × 8), `rmse`, `magnitude`, `change_prob`, `category`, `num_obs`.

### `run_ccdc_array` { .api }

<!-- sig: zeit.raster.run_ccdc_array -->
```python
zeit.raster.run_ccdc_array(
    dates, raster_stack, qa_stack, max_segments=6, n_jobs=-1,
    return_coefs=True, conseq_anom=6, **ccdc_kwargs,
)
```

CCDC for every pixel of an in-memory stack, in parallel. Also exported as `zeit.run_ccdc_array`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dates` | 1-D array | required | Ordinal days. |
| `raster_stack` | `np.ndarray` | required | `(bands, time, rows, cols)` reflectance × 10,000. |
| `qa_stack` | `np.ndarray` | required | `(time, rows, cols)` Fmask codes. |
| `max_segments` | `int` | `6` | Model slots in the output. |
| `n_jobs` | `int` | `-1` | Threads. |
| `return_coefs` | `bool` | `True` | Return full models; `False` returns only break dates. |
| `conseq_anom` | `int` | `6` | As in `run_ccdc`. |
| `chi2_prob_threshold`, `tmax_cg_prob_threshold`, `detection_bands`, `num_c`, `tmask_bands`, `thermal_band`, `valid_range`, `thermal_range` | | | Passed through (`**ccdc_kwargs`), as in `run_ccdc`. |

</div>

**Returns** `(max_segments, 3 + 9 × bands, rows, cols)`: per model `t_start`, `t_end`, `t_break`, then for each band its RMSE and 8 coefficients.

### `run_ccdc_image` { .api }

<!-- sig: zeit.raster.run_ccdc_image -->
```python
zeit.raster.run_ccdc_image(
    input_path, output_dir, dates, num_bands=6, qa_band_idx=-1,
    max_segments=6, chunk_size=512, n_jobs=-1, prefix="ccdc_break",
    return_coefs=True, conseq_anom=6,
)
```

CCDC on a date-interleaved GeoTIFF (all bands of date 1, then of date 2, …), block by block. Writes `<prefix>_coefs.tif` with `max_segments × (3 + 9 × num_bands)` bands. Also exported as `zeit.run_ccdc_image`; CLI: `zeit ccdc`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` | required | Date-interleaved GeoTIFF. |
| `output_dir` | `str` | required | Output folder. |
| `dates` | 1-D array | required | Ordinal days, one per date in the file. |
| `num_bands` | `int` | `6` | Bands per date in the file (including the QA band, if any). |
| `qa_band_idx` | `int` | `-1` | Position of the QA band (Fmask codes) within each date. `-1`: no QA band, everything clear. |
| `max_segments` | `int` | `6` | Model slots. |
| `chunk_size` | `int` | `512` | Block size in pixels. |
| `n_jobs` | `int` | `-1` | Threads. |
| `prefix` | `str` | `"ccdc_break"` | Output file prefix. |
| `return_coefs` | `bool` | `True` | As in `run_ccdc_array`. |
| `conseq_anom` | `int` | `6` | As in `run_ccdc`. |

</div>

### `predict_synthetic_image` { .api }

<!-- sig: zeit.ccdc.predict_synthetic_image -->
```python
zeit.ccdc.predict_synthetic_image(
    ccdc_coefs_stack, target_julian_day, num_bands=6,
)
```

Evaluates the CCDC model active on a given date for every pixel, giving a cloud-free image for any day. Also exported as `zeit.predict_synthetic_image`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ccdc_coefs_stack` | `np.ndarray` | required | Output of `run_ccdc_array` (or the `_coefs.tif` reshaped to `(segments, params, rows, cols)`). |
| `target_julian_day` | `int` | required | Ordinal day to predict. |
| `num_bands` | `int` | `6` | Bands to predict. |

</div>

**Returns** `(num_bands, rows, cols)` float32.

```python
from datetime import date
img = zeit.predict_synthetic_image(results, date(2019, 7, 15).toordinal())
```

### `predict` { .api }

<!-- sig: zeit.ccdc.predict -->
```python
zeit.ccdc.predict(coefs, dates)
```

Evaluates one band's 8 coefficients (from `run_ccdc`'s `coefs`) at any ordinal day or days.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `coefs` | array | required | 8 coefficients of one band. |
| `dates` | `int` or array | required | Ordinal day(s). |

</div>

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
