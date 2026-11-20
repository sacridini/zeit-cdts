# Change Detection

<p class="lead">LandTrendr and CCDC, one function each for every kind of input, the BFAST family and BEAST at every level: one pixel, an in-memory array, a GeoTIFF on disk, or a Dask array. The xarray accessor forms are listed in <a href="../xarray/">Xarray Accessor</a>.</p>

## LandTrendr

Tutorial: [LandTrendr](../tutorials/landtrendr.md).

### `landtrendr` { .api #landtrendr }

<!-- sig: zeit.landtrendr -->
```python
zeit.landtrendr(
    data, *, years=None, direction="loss", band=None, max_segments=6,
    pval_threshold=0.05, recovery_threshold=0.25,
    prevent_fast_recovery=True, spike_threshold=0.9,
    best_model_proportion=0.75, vertex_count_overshoot=3,
    min_observations_needed=6, nodata="auto", fitted=False, ftv=None,
    chunks=None, n_jobs=-1,
)
```

Segments annual time series with LandTrendr (Kennedy et al. 2010), every pixel in parallel in the Rust core. One function for every input: it reads what `data` is and returns the same `xarray.Dataset` of vertices, georeferenced when the input is. All parameters but `data` are keyword-only. It replaces `run_landtrendr`, `run_landtrendr_array`, `run_landtrendr_image` and `DataArray.zeit.run_landtrendr`; the accessor form is now [`DataArray.zeit.landtrendr`](xarray.md#landtrendr).

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
| `ftv` | list of `str` | `None` | Other bands to fit to the vertices of `band` (LT-GEE's fitted-to-vertex bands), by name: bands of the `(time, band, y, x)` cube or variables of the `Dataset`. Each is desawtoothed and fitted segment by segment through the vertex years found, in its own scale (not flipped by `direction`), as the original's `ftv_v1`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads the raster into memory; `"auto"` or a dict of chunk sizes keeps it lazy, so the result is computed block by block, for rasters larger than memory. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Dims | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `vertex_year` | `(vertex, y, x)` | int16 | Year of each vertex; `0` past the last one. |
| `vertex_value` | `(vertex, y, x)` | float32 | Fitted value at each vertex; NaN past the last one. |
| `n_vertices` | `(y, x)` | uint8 | Number of vertices (`0`: not segmented). |
| `rmse` | `(y, x)` | float32 | RMSE of the fit: the noise estimate behind `dsnr` in `extract_events`. |
| `fitted` | `(time, y, x)` | float32 | Only with `fitted=True`: the fitted trajectory. |
| `ftv_<band>` | `(time, y, x)` | float32 | For each band in `ftv`: its series fitted to the vertices. |
| `vertex_value_<band>` | `(vertex, y, x)` | float32 | For each band in `ftv`: its fitted value at each vertex year; NaN past the last vertex. |

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

#### Fitted to vertices (FTV)

`ftv=` describes the periods found on one index with other bands, as LT-GEE's `ftv` bands: segment on NBR, then fit NDVI, TCW or SWIR1 through the same vertex years.

```python
stack = zeit.load_raster("landsat_indices.tif")       # (time, band, y, x): nbr, ndvi, tcw, ...
lt = zeit.landtrendr(stack, band="nbr", ftv=["ndvi", "tcw"])
lt.ftv_ndvi                                           # (time, y, x): NDVI fitted to the NBR vertices
lt.vertex_value_tcw                                   # (vertex, y, x): TCW at each vertex year
```

It is a port of the original's `ftv_v1.pro` (with `apply_fitted_trajectory_v1.pro`), checked against it run under GDL. Each band is desawtoothed with `spike_threshold` and fitted segment by segment, early to late, by the better of a straight line between the vertices and a regression anchored at the segment's start (`find_best_trace`). A vertex year the band has no observation for (a cloud in that band only) moves to the band's nearest observation before it, or after it when that one is a vertex already. Missing first and last years get flat vertices, as in the segmentation. The bands must be on the same grid and years as the segmented one (see [`load_raster(..., like=)`](data.md#on-the-grid-of-another-raster)).

### `extract_events` { .api }

<!-- sig: zeit.metrics.extract_events -->
```python
zeit.metrics.extract_events(
    result, event_type=None, sort_by="greatest", min_magnitude=0.0,
    min_duration=1, pre_val_threshold=0.0, rmse_map=None, *,
    band=None,
)
```

Turns the result of any change detection algorithm into maps of one event per pixel, such as the greatest loss: the same variables whichever algorithm found the change, so maps of LandTrendr, CCDC and BFAST can be compared ([`agreement`](#agreement)), combined, or validated the same way. Also exported as `zeit.extract_events`.

| Result of | Candidate events of a pixel |
| :--- | :--- |
| [`landtrendr`](#landtrendr) | Each segment that goes in the direction of `event_type`. |
| [`ccdc`](#ccdc) | Each break followed by another segment; the change is the next segment's model minus the previous one's, both on the date of the break. A break at the end of the series, with no model after it yet, has no magnitude and is left out. |
| [`bfast_monitor`](#bfast_monitor) | The break, with its `magnitude` (the median residual of the monitoring period). |
| [`bfast_lite`](#bfast_lite) | Each break, with `magnitude_k`. |
| [`bfast`](#bfast) | Each trend break, with `trend_magnitude_k`. |
| [`embedding_change`](embeddings.md#embedding_change) | Each year: the distance between its embedding and the year before's, with no direction (`event_type="any"`); `dsnr` against the pixel's median distance. |

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `result` | `xr.Dataset` | required | The result of one of the algorithms above, in memory or dask. A numpy LandTrendr vertex stack `(2 × vertices, rows, cols)`, years in the first half and values in the second, is also accepted. |
| `event_type` | `str` | `None` | `"loss"` (the value fell), `"gain"` (it rose) or `"any"` (either; the magnitude is then the size of the change). Default: the `direction` LandTrendr ran with (`"loss"` for a numpy stack), `"any"` for the other algorithms. LandTrendr takes `"loss"` or `"gain"`. |
| `sort_by` | `str` | `"greatest"` | Which event to keep when a pixel has several: `"greatest"` (magnitude), `"newest"`, `"fastest"`, `"longest"` or `"dsnr"` (not for `bfast`, whose result keeps no noise estimate). |
| `min_magnitude` | `float` | `0.0` | Ignore events smaller than this, in data units. A flat segment (magnitude `0`) is never an event. |
| `min_duration` | `int` | `1` | Ignore events shorter than this many years. |
| `pre_val_threshold` | `float` | `0.0` | For losses, ignore events starting below this value (for gains, above). `0` disables. Needs the value before the change: LandTrendr, or CCDC with `band`. |
| `rmse_map` | `np.ndarray` | `None` | Numpy stacks only: `(rows, cols)` RMSE of each pixel's fit, which adds the `dsnr` output. A LandTrendr Dataset brings its own (`rmse`). |
| `band` | `str` | `None` | CCDC only: the band whose change is measured, e.g. `"nir"`. Without it (and with several bands), the magnitude is the length of the change vector over the run's detection bands, with no direction (`event_type="any"`). |

</div>

**Returns** an `xarray.Dataset` on the same grid and CRS as `result` (lazy if `result` is), ready for `save_raster`:

| Variable | Meaning |
| :--- | :--- |
| `yod` | Year of the last observation before the change (`0` = no event). For LandTrendr, the year of the vertex where the event starts; the first year in which the change is visible is `yod + 1`. For BFAST, the year of the time step before `date`. |
| `date` | Date of the first observation that shows the change (`NaT` = no event). For annual LandTrendr, January 1 of `yod + 1`; for CCDC, the break (`t_break`). |
| `magnitude` | Size of the change, in data units, positive in the direction of `event_type`. |
| `duration` | Years the change took. `1` is abrupt, and every break of CCDC and BFAST is. |
| `pre_val`, `post_val` | Fitted value before and after (NaN for BFAST, whose results keep no model, and for CCDC without `band`). |
| `rate` | `magnitude / duration`. |
| `dsnr` | Magnitude divided by the noise of the fit (LT-GEE's disturbance signal-to-noise ratio): LandTrendr's RMSE, the CCDC segment's RMSE (over several bands, the change vector standardized band by band), BFAST Monitor's `sigma`, BFAST Lite's residual standard deviation, the median yearly distance of an embedding. Values above 2–3 are rarely noise. |

For a numpy vertex stack, a dict of `(rows, cols)` arrays with the LandTrendr variables except `date` (`dsnr` only with `rmse_map`).

```python
lt = zeit.landtrendr(ndvi)
loss = zeit.extract_events(lt, min_magnitude=2000)
first_year_of_loss = (loss.yod + 1).where(loss.yod > 0)
regrowth = zeit.extract_events(lt, event_type="gain", sort_by="newest")   # same fit, rises

segments = zeit.ccdc(cube)
nir_loss = zeit.extract_events(segments, band="nir", event_type="loss")   # the greatest NIR drop
any_change = zeit.extract_events(segments, sort_by="newest")              # the latest break, any direction

bfm = zeit.extract_events(zeit.bfast_monitor(ndvi_16d, "2022-01-01"), event_type="loss")
```

### `recovery_metrics` { .api #recovery_metrics }

<!-- sig: zeit.recovery_metrics -->
```python
zeit.recovery_metrics(
    data, events=None, *, target=0.8, horizon=5, pre_years=2,
    basis="pre_value", event_type=None, sort_by="greatest",
    min_magnitude=0.0, min_duration=1, pre_val_threshold=0.0,
    years=None, band=None, nodata="auto", chunks=None, n_jobs=-1,
)
```

How each pixel recovered after its disturbance: the spectral recovery metrics of the forest literature (Kennedy et al. 2012; White et al. 2017; Frazier et al. 2018), on the Rust core. Detecting the year of a loss is half the work; the other half is how fast and how far the pixel came back after a fire, a harvest or an abandonment.

The disturbance runs from its start, the last year before it (an event's `yod`), to its trough, the year it bottoms out (`yod + duration`; the year after `yod` for the algorithms that find abrupt breaks). With `t0` the trough and `k` the horizon:

| Metric | Definition |
| :--- | :--- |
| `pre_val` | The mean of the `pre_years` years up to the start (NBR<sub>pre</sub>). |
| `dist_val`, `magnitude` | The value at `t0`, and `pre_val - dist_val` (ΔNBR). |
| `post_val` | The best of years `t0 + k - 1` and `t0 + k` (the maximum of years 4 and 5, Frazier et al. 2018): one noisy year does not decide. |
| `ari` | Absolute recovery (ΔNBR<sub>regrowth</sub>): `post_val - dist_val`. |
| `rri` | Relative recovery indicator (Kennedy et al. 2012): `ari / magnitude`. |
| `r80p` | `post_val / (target * pre_val)`: 1 or more when recovered (White et al. 2017). |
| `yryr` | Year-on-year recovery: `(value at t0 + k - dist_val) / k`. |
| `y2r` | Years to recovery (White et al. 2017): from `t0` until the value first reaches `target * pre_val`; 0 when it never fell below it, NaN when it has not reached it by the last year. |

On a LandTrendr result the metrics are measured on the fitted trajectory (the line through the vertices), after the event `extract_events` picks; on a series, on its values, after the `events` given. The metrics measured at the horizon are NaN when the series ends before it. Janus has it as the *Recovery (LandTrendr)* tool. RecoverMetricsR (R) computes RRI, R80P and YrYr alike (it takes the best of the 5 years after the trough, and counts the years to recovery from the year after it): on 1000 x 1000 pixels and 21 years, it takes 52 s; Zeit 0.1 s on one thread (benchmarks/vs_originals/recovery.py).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `Dataset`, path, `DataArray`, `ndarray`, list or `pd.Series` | required | A [`landtrendr`](#landtrendr) result, or an annual series (one observation per year) as for [`mann_kendall`](time-series.md#mann_kendall), with `events`. |
| `events` | `Dataset`, `(y, x)` layer or number | `None` | The disturbances: an [`extract_events`](#extract_events) result (`yod` and `duration`), or each pixel's start year (the trough is the year after); 0 or NaN for none. LandTrendr results pick their own when `None`. |
| `target` | `float` | `0.8` | The fraction of the pre-disturbance value that counts as recovered. |
| `horizon` | `int` | `5` | Years after the trough at which `post_val`, `ari`, `rri`, `r80p` and `yryr` are measured. |
| `pre_years` | `int` | `2` | Years averaged for `pre_val`. |
| `basis` | `str` | `"pre_value"` | `"pre_value"`: recovered at `target * pre_val`, for an index that is positive and drops with the disturbance (NBR, NDVI, TCW). `"magnitude"`: recovered once `target` of the loss came back, for any index; `r80p` is then `rri / target`. |
| `event_type` | `str` | `None` | `"loss"` or `"gain"` (the index rises with the disturbance: the signs turn, `post_val` is the lower of the two years; needs `basis="magnitude"`). Default: LandTrendr's direction, else `"loss"`. |
| `sort_by` | `str` | `"greatest"` | LandTrendr without `events`: which event, as in [`extract_events`](#extract_events). |
| `min_magnitude` | `float` | `0.0` | LandTrendr without `events`: as in `extract_events`. |
| `min_duration` | `int` | `1` | LandTrendr without `events`: as in `extract_events`. |
| `pre_val_threshold` | `float` | `0.0` | LandTrendr without `events`: as in `extract_events`. |
| `years` | list | `None` | One pixel's series without dates: its years. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; `"auto"` or a dict keeps them lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** an `xarray.Dataset` on the grid of the input: `yod` and `trough` (int32, the disturbance's years, 0 without one) and the nine metrics above (float32).

```python
lt = zeit.landtrendr(nbr, direction="loss")
rec = zeit.recovery_metrics(lt)                              # after each pixel's greatest loss
slow = rec.y2r > 10                                          # still under 80% of NBR_pre after 10 years

ev = zeit.extract_events(zeit.ccdc(bands), band="swir2", event_type="gain")
rec = zeit.recovery_metrics(annual_swir2, ev, event_type="gain", basis="magnitude")
```

### `agreement` { .api }

<!-- sig: zeit.agreement -->
```python
zeit.agreement(*events, tolerance=1)
```

Where and when several change maps agree, e.g. LandTrendr, CCDC and BFAST on the same area. Each map is the result of [`extract_events`](#extract_events) (or any `yod` map), so every algorithm is compared by the year of the last observation before the change.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `events` | `xr.Dataset`, `DataArray` or `dict` | required | Two or more maps of events on the same grid, or one `dict` of name → map. Without a `dict`, maps are named after their algorithm. Maps on different grids go through [`load_raster(m, like=...)`](data.md#on-the-grid-of-another-raster) first. |
| `tolerance` | `int` | `1` | Years two events may be apart and still agree (`0`: the same year). |

</div>

**Returns** an `xarray.Dataset` on the grid of the maps:

| Variable | Meaning |
| :--- | :--- |
| `yod` | The consensus year: the one the most maps agree with within `tolerance` (ties: the year more maps give exactly, then the earliest); `0` where no map has an event. |
| `n_detected` | How many maps have an event at the pixel. |
| `n_agree` | How many of them agree with the consensus year. |
| `spread` | Latest minus earliest year among the maps with an event; NaN without one. |
| `agrees` | `(map, y, x)`: `1` for the maps that agree with the consensus year. |

```python
lt = zeit.extract_events(zeit.landtrendr(nbr_annual))
cc = zeit.extract_events(zeit.ccdc(cube), band="nbr", event_type="loss")
bf = zeit.extract_events(zeit.bfast_lite(nbr_16d), event_type="loss")
agree = zeit.agreement({"landtrendr": lt, "ccdc": cc, "bfast": bf}, tolerance=1)
confident = agree.yod.where(agree.n_agree >= 2, 0)        # change two of the three algorithms see

# An ensemble in the LCMS way: the events of each algorithm as the features of a classifier
features = xr.merge([ev[["magnitude", "dsnr"]].fillna(0).rename({v: f"{name}_{v}" for v in ("magnitude", "dsnr")})
                     for name, ev in {"lt": lt, "ccdc": cc, "bfast": bf}.items()], compat="override")
model = zeit.train_classifier(features, "reference.gpkg", label="change")
change = zeit.classify(features, model)
```

### `apply_vertices` { .api }

<!-- sig: zeit.apply_vertices -->
```python
zeit.apply_vertices(vertex_years, other_band_years, other_band_values)
```

Describes another band with the vertex years found on the primary index, by interpolating that band's raw values at those years. For LT-GEE's fitted-to-vertex bands (the band fitted through the vertices, as the original does), use [`landtrendr(..., ftv=[...])`](#fitted-to-vertices-ftv).

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
    data, *, qa=None, dates=None, bands=None, max_segments=6,
    conseq_anom=6, chi2_prob_threshold=0.99,
    tmax_cg_prob_threshold=0.999999, detection_bands=None, num_c=8,
    tmask_bands=None, thermal_band=None, valid_range=(0.0, 10000.0),
    thermal_range=(-9320.0, 7070.0), nodata="auto", scale=None, chunks=None,
    n_jobs=-1,
)
```

Continuous Change Detection and Classification (Zhu & Woodcock 2014): fits a harmonic model to every stable period of each pixel's multi-band series and dates the breaks between them, every pixel in parallel in the Rust core. The engine follows the original MATLAB code and is validated segment for segment against it: the same segments, break dates, categories and observation counts, with coefficients, RMSE and magnitudes within small tolerances (the lasso is written from the paper, not ported from GLMnet; see [Algorithm Fidelity](../benchmarks/fidelity.md#2-ccdc-zhu-woodcock-2014)). One function for every input: it reads what `data` is and returns the same `xarray.Dataset` of segments, georeferenced when the input is. All parameters but `data` are keyword-only. It replaces `zeit.ccdc.run_ccdc`, `run_ccdc_array`, `run_ccdc_image` and `DataArray.zeit.run_ccdc`; the accessor form is now [`DataArray.zeit.ccdc`](xarray.md#ccdc).

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
| `qa` | `str`, `DataArray` or `ndarray` | `None` | Fmask codes per date: `0` clear land, `1` water, `2` cloud shadow, `3` snow, `4` cloud, `255` no observation. The name of a band of the cube (it is then left out of the spectral bands), a `(time, y, x)` cube or array (`(time,)` for one pixel), or `None`: every date is clear. [`qa_to_fmask`](data.md#qa_to_fmask) makes the codes from a quality band (QA_PIXEL, SCL, HLS). A date with any band NaN (or equal to `nodata`) is no observation, a cloud (`4`), whatever `qa` says: the cube can go in with its gaps. |
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
| `scale` | `float`, `"auto"` or `None` | `None` | The units of the spectral bands: the value that means reflectance 1. `None` or `10000`: reflectance × 10,000, as they are. `1`: reflectance 0-1, multiplied by 10,000. `"auto"`: 1 if the 99th percentile of the values is 2 or less, else 10,000, as [`unmix`](preprocessing.md#unmix). Landsat Collection 2 digital numbers cannot be told from reflectance × 10,000: read them with `load_raster(..., scale=True)`. The thermal band is not scaled. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads the raster into memory; `"auto"` or a dict of chunk sizes keeps it lazy, so the result is computed block by block, for rasters larger than memory. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

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

### `sccd` { .api #sccd }

<!-- sig: zeit.sccd -->
```python
zeit.sccd(
    data, *, qa=None, dates=None, bands=None, max_segments=6, conse=6,
    prob_threshold=0.99, outlier_prob=0.99999, max_angle=30.0,
    detection_bands=None, tmask_bands=None,
    valid_range=(0.0, 10000.0), nodata="auto", scale=None, chunks=None,
    n_jobs=-1,
)
```

Stochastic Continuous Change Detection (S-CCD; Ye, Rogan, Zhu & Eastman 2021, *Remote Sensing of Environment* 252:112167): CCDC whose harmonic model, once fitted on the first stable observations, is a state-space model that a Kalman filter updates at every new observation instead of refitting it. The trend and the seasons drift a little from one observation to the next, so the model follows slow changes, and a new image costs one update per pixel: on long series it runs about ten times faster than [`ccdc`](#ccdc). It also returns the near-real-time state of every pixel: how many of its latest observations already depart from the model, before `conse` of them confirm a change. Written from the paper; checked against pyxccd, the authors' implementation (see [fidelity](../benchmarks/fidelity.md)). It takes the inputs of [`ccdc`](#ccdc) (the same `data` table, surface reflectance × 10,000, Fmask `qa`); all parameters but `data` are keyword-only.

A change is `conse` consecutive observations whose departures from the model, standardized by each band's RMSE (no less than the median absolute difference of consecutive observations), have a squared norm above `chi2inv(prob_threshold, detection bands)` in every one, and point the same way: their mean angle to the median departure is below `max_angle`. A single observation beyond `chi2inv(outlier_prob, …)` is left out as noise.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `ndarray` or `pd.DataFrame` | required | What to run on, as [`ccdc`](#ccdc). |
| `qa` | `str`, `DataArray` or `ndarray` | `None` | Fmask codes per date, as [`ccdc`](#ccdc). |
| `dates` | list | `None` | Date of each time step, for numpy input and rasters without dates. |
| `bands` | list of `str` or `int` | `None` | Spectral bands to use, by name or 0-based position, ordered Blue, Green, Red, NIR, SWIR1, SWIR2. |
| `max_segments` | `int` | `6` | Maximum segments returned per pixel. |
| `conse` | `int` | `6` | Consecutive anomalous observations that confirm a change (the peek window). |
| `prob_threshold` | `float` | `0.99` | Change probability of the chi-squared threshold. Lower is more sensitive. |
| `outlier_prob` | `float` | `0.99999` | Outlier probability: a single observation beyond it is dropped as noise. |
| `max_angle` | `float` | `30.0` | Largest mean angle (degrees) of the peek window's departures to their median: departures that point every way are noise, not a change. |
| `detection_bands` | list of `str` or `int` | `None` | Bands of the change test. Default: Green to SWIR2 (with 6 bands or more; else all). |
| `tmask_bands` | list of `str` or `int` | `None` | Bands of the Tmask cloud screen. Default: Green and SWIR1. |
| `valid_range` | `tuple` | `(0.0, 10000.0)` | Valid range of the values. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN, as [`ccdc`](#ccdc). A date with any band NaN is no observation. |
| `scale` | `float`, `"auto"` or `None` | `None` | The units of the bands, as [`ccdc`](#ccdc): `None`/`10000` reflectance × 10,000; `1` reflectance 0-1; `"auto"` tells them. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; `"auto"` or a dict keeps them lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** an `xarray.Dataset`. The segments, as [`ccdc`](#ccdc)'s:

| Variable | Dims | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `t_start`, `t_end` | `(segment, y, x)` | datetime64 | First and last date of each segment's model; `NaT` past the last one. |
| `t_break` | `(segment, y, x)` | datetime64 | Date of the change that ended the segment (its first anomalous observation); `NaT` for the segment still monitored. |
| `change_prob` | `(segment, y, x)` | float32 | `1` for a confirmed change; for the last segment, the share of `conse` of its latest observations that already depart from it. |
| `num_obs` | `(segment, y, x)` | int32 | Observations in the model. |
| `rmse` | `(segment, band, y, x)` | float32 | RMSE of the model's one-step-ahead predictions. |
| `magnitude` | `(segment, band, y, x)` | float32 | The change: median departure of the peek window from the model, in the band's units; `0` without a change. |
| `coefs` | `(segment, band, coef, y, x)` | float32 | The filter's state at the end of the segment as harmonic coefficients, `coef` = `a0`, `c1`, `a1`, `b1`, `a2`, `b2` on [`ccdc`](#ccdc)'s time axis, so [`predict_synthetic_image`](#predict_synthetic_image) evaluates it. |
| `n_segments`, `n_breaks` | `(y, x)` | uint8 | Number of segments (`0`: no model) and of changes. |

The near-real-time state, from the latest observations:

| Variable | Dims | Type | Meaning |
| :--- | :--- | :--- | :--- |
| `nrt_mode` | `(y, x)` | uint8 | `0` no model (no clear observation, or less than a quarter clear), `1` monitoring, `2` waiting for enough observations to start a model (at the start, or after a change). |
| `n_anomalies` | `(y, x)` | uint8 | The latest consecutive observations that depart from the model (`conse` of them make a change). |
| `anomaly_start` | `(y, x)` | datetime64 | Date of the first of them; `NaT` if none. |
| `anomaly_statistic` | `(y, x)` | float32 | The smallest of their standardized squared norms, against `attrs["threshold"]`. |
| `anomaly_magnitude` | `(band, y, x)` | float32 | Their median departure from the model. |
| `last_statistic`, `last_observation` | `(y, x)` | float32, datetime64 | The latest clear observation's statistic and date. |
| `n_queue` | `(y, x)` | int32 | Observations waiting for a model (`nrt_mode` 2). |

One pixel also has, per date: `status (time)`, what became of each observation (`0` unused: not clear or missing, `1` in a model, `2` outlier, `3` the first observation of a change, `4` in the final peek window, `5` waiting for a model); `statistic (time)`, its standardized squared norm when tested; and `predicted`, `trend`, `annual`, `semiannual (time, band)`, the model's prediction and its components, filtered up to that date: the chart of a pixel. `attrs` records the parameters and the `threshold`. Coordinates and CRS are kept, so the result goes to [`save_raster`](data.md#save_raster); with a dask cube or `chunks=`, it is lazy.

```python
segments = zeit.sccd(cube, qa="fmask")
first_change = segments.t_break.isel(segment=0)     # NaT where nothing changed
alerts = segments.n_anomalies >= 3                  # changes in progress, not yet confirmed
july = zeit.predict_synthetic_image(segments, "2024-07-01")

# One pixel: its segments and, per date, what the filter did
px = zeit.sccd(df, qa="fmask")
chart = px[["status", "statistic"]].to_dataframe().join(px.predicted.sel(band="nir").to_series().rename("nir_model"))
```

### `predict_synthetic_image` { .api }

<!-- sig: zeit.predict_synthetic_image -->
```python
zeit.predict_synthetic_image(
    segments, date=None, num_bands=6, *, target_julian_day=None,
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

## CODED (forest degradation)

### `coded` { .api }

<!-- sig: zeit.coded -->
```python
zeit.coded(
    data, *, start=None, train_years=3.0, consec=3, thresh=3.0,
    min_years=3.0, min_obs=6, direction="loss", max_events=3,
    training=None, label="label", forest_label=None, forest_ndfi=0.5,
    endmembers="souza2005", bands=None, scale="auto",
    cloud_threshold=0.05, seed=42, nodata="auto", chunks=None,
    n_jobs=-1,
)
```

Continuous Degradation Detection (CODED; Bullock, Woodcock & Olofsson 2020): forest degradation (a disturbance after which the land is still forest: selective logging, understory fire) and deforestation (after which it is not) from the NDFI of every Landsat observation. Per pixel, in Rust: the fractions and NDFI of every observation (`unmix`, clouds masked by their fraction); a model of the NDFI (a constant and an annual harmonic) over a training period, with its RMSE; monitoring after it, where `consec` observations in a row more than `thresh` RMSEs below the model are a change; after each change, a new model of the next `min_years`, which is the land cover after it. Forest or not comes from the models: a random forest on their coefficients trained on land cover points, as in CODED, or else the model's mean NDFI against `forest_ndfi`. Also exported as `zeit.coded`; CLI: `zeit coded`. Tutorial: [Forest Degradation (CODED)](../tutorials/coded.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset` or path | required | Landsat-like reflectance `(time, band, y, x)` with blue, green, red, NIR, SWIR1 and SWIR2 (every observation, not composites), or the result of `unmix`. One pixel, a `(time, band)` series as `load_raster(pixel=)` gives, is read as a 1 × 1 cube and the result has no `y` and `x`. Mask the unusable dates first (`cube.where(zeit.qa_to_fmask(qa, "landsat_c2") <= 1)`): NaN is no observation. |
| `start` | year or date | `None` | Start of the monitoring; the training period is the `train_years` before it. Default: the training period starts with the series. |
| `train_years` | `float` | `3.0` | Years of the training period. |
| `consec` | `int` | `3` | Consecutive observations beyond the threshold that make a change. |
| `thresh` | `float` | `3.0` | The threshold, in RMSEs of the model. |
| `min_years` | `float` | `3.0` | Years after a change modelled as its new land cover; the least time between changes. |
| `min_obs` | `int` | `6` | Least observations to fit a model. |
| `direction` | `str` | `"loss"` | `"loss"` (drops of the NDFI, CODED's disturbances) or `"both"`. |
| `max_events` | `int` | `3` | Changes kept per pixel. |
| `training` | `GeoDataFrame` or path | `None` | Land cover points of the training period, for the random forest. |
| `label`, `forest_label` | | `"label"`, `None` | Column with the class, and the class that is forest (default: `"forest"` or `1`). |
| `forest_ndfi` | `float` | `0.5` | Without `training`: forest where the model's mean NDFI is at least this. |
| `endmembers`, `bands`, `scale`, `cloud_threshold` | | | The unmixing, as in `unmix`. |
| `seed`, `nodata`, `chunks`, `n_jobs` | | `42`, `"auto"`, `None`, `-1` | Random forest seed; as elsewhere. |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Meaning |
| :--- | :--- |
| `t_change` | `(event, y, x)`: date of each change, its first observation beyond the threshold. |
| `t_before` | The observation before it. |
| `ndfi_change` | Mean NDFI residual of the change's observations (negative: the NDFI fell). |
| `type` | 1 degradation (forest after the change), 2 deforestation, 3 disturbance (too few observations after it to tell), 0 none. |
| `post_ndfi` | Mean NDFI of the model after each change. |
| `n_events`, `forest` | Changes per pixel; forest in the training period. |
| `ndfi_mean`, `rmse`, `n_train` | The training model: mean NDFI, RMSE, observations. |
| `strata` | 1 forest, 2 non-forest (no change), 3 degradation, 4 deforestation, 5 disturbance, by the first change, with `flag_meanings`. |

The `strata` map is what CODED's authors recommend using the result for: the strata of a sample from which the area of degradation is estimated ([Validation](validation.md)). [`extract_events`](#extract_events) turns the changes into the same event maps as the other algorithms.

```python
result = zeit.coded(landsat, start=2000, training="land_cover_1997_1999.gpkg")
design = zeit.sampling_design(result.strata, expected_ua={"degradation": 0.6, "deforestation": 0.8})
points = zeit.stratified_sample(result.strata, design=design)
```

## EWMACD (monitoring with control charts)

### `ewmacd` { .api #ewmacd }

<!-- sig: zeit.ewmacd -->
```python
zeit.ewmacd(
    data, *, train_end, train_start=None, order=2, lmbda=0.3,
    sigmas=3.0, persistence=3, xbar_train=1.5, xbar_monitor=20.0,
    rounding=True, low=None, dates=None, band=None, nodata="auto",
    chunks=None, n_jobs=-1,
)
```

Exponentially Weighted Moving Average Change Detection (EWMACD; Brooks et al. 2014): disturbances, gains and recovery from every observation of a vegetation index, irregular and cloudy dates included, with statistical quality control charts. The method is light, and it catches small persistent shifts (thinning, drought, pests) as well as clearcuts. Unlike BFAST Monitor, it needs no regular series. Per pixel, in Rust:

1. A harmonic model (a constant and `order` sine and cosine pairs of the day of the year) is fitted on the training years. It is then refitted without the dates beyond `xbar_train` standard deviations, a Shewhart X-bar chart that screens out the clouds and shadows of those years.
2. The residuals of every date are screened again: training dates beyond `xbar_train`, and later dates beyond `xbar_monitor` (only gross anomalies).
3. The EWMA chart of the residuals kept is `z_j = (1 - lmbda) z_(j-1) + lmbda r_j`. Its control limits are `sigmas * sigma * sqrt(lmbda / (2 - lmbda) * (1 - (1 - lmbda)^(2j)))`, where `sigma` is the standard deviation of the training residuals.
4. The flags are `z / limit` truncated toward zero: 0 inside the limits, -1, -2... for a loss of 1, 2... limits, and positive values for a gain. A signal must last `persistence` dates.

The flags after the training years give the first disturbance, its severity, the date it recovered, the first gain and the state at the last date.

The flags are those of Brooks' EWMACD v1.3.0 script, the paper's own (Earth Engine has it as `ee.Algorithms.TemporalSegmentation.Ewmacd`). With `persistence=1` they were checked date by date on synthetic series (`tests/data/make_ewmacd_parity_fixtures.R`) and on 400 pixels of the paper's sample data. With more persistence, the first disturbance is the same but a few later dates differ: on those dates the script means to keep the previous sustained state but reads another date's flag. Values not above `low` are treated as missing from the start, whereas the script still fits them in its first regression. On 10,000 pixels and 71 dates, the script takes 2.8 s; Zeit 55 ms on one thread, 7 ms on 19 (benchmarks/vs_originals/ewmacd.py).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset`, path, `pd.Series`, `ndarray` or list | required | One index (NDVI, NBR, the tasseled cap angle of the paper): a `(time, y, x)` cube, a raster, or one pixel's series, as for [`mann_kendall`](time-series.md#mann_kendall). Every observation counts. |
| `train_end` | `int` | required | The first year after the training period, exclusive (as Brooks' `trainingEnd` and Earth Engine's `trainingEndYear`). The model is fitted on the years before it, and monitoring starts with it. The training years should be stable. |
| `train_start` | `int` | `None` | The first year looked at; earlier dates are left out. Default: the first year. |
| `order` | `int` | `2` | Sine and cosine pairs of the harmonic model. |
| `lmbda` | `float` | `0.3` | Weight of each new residual in the EWMA, in (0, 1]. Low values make a slow, smooth chart (subtle changes, few false signals); 1 gives a Shewhart chart of each date. |
| `sigmas` | `float` | `3.0` | The control limits, in standard deviations of the EWMA. |
| `persistence` | `int` | `3` | Dates a signal must last (Earth Engine's default). A shorter run takes the flag of the last run that lasted. |
| `xbar_train` | `float` | `1.5` | X-bar screen of the training dates, in standard deviations. |
| `xbar_monitor` | `float` | `20.0` | X-bar screen of the later dates. An abrupt change larger than this many standard deviations of a quiet pixel is screened as an anomaly until it comes back within it; raise it for smooth series. |
| `rounding` | `bool` | `True` | Flags truncated to integers. Otherwise the flag is the ratio `z / limit` itself beyond ±1, and 0 inside. |
| `low` | `float` | `None` | Values not above it are missing (Brooks' `lowthresh`: no vegetation, fill values). |
| `dates` | list | `None` | The dates of a series that has none. |
| `band` | `str` or `int` | `None` | The index in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; `"auto"` or a dict keeps them lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Meaning |
| :--- | :--- |
| `flags` | `(time, y, x)`: the chart's flags. NaN before `train_start` or where no model could be fitted. Dates left out of the chart carry the flag of the date before. |
| `break_time` | The first disturbance after the training years: the decimal year of the first negative flag. NaN when there is none. |
| `severity` | Its most negative flag until it recovers, in control limits. |
| `recovery_time` | The first date back inside the limits (or above) after the disturbance. |
| `gain_time` | The first positive flag after the training years. |
| `state` | At the last date: 0 stable, 1 disturbed, 2 recovered (disturbed, then back), 3 gain, with `flag_meanings`. |
| `sigma`, `rmse`, `r2` | The training residuals' standard deviation behind the limits; the harmonic model's RMSE and R². |
| `coefficients` | `(coefficient, y, x)`: the model's constant, sines and cosines. |
| `ewma`, `limit`, `fitted` | One pixel's series only, per date: the chart and its limits, NaN where the date was left out of the chart, and the model, at every date. |

```python
ndvi = zeit.load_raster("landsat_ndvi/*.tif")             # every scene, 2013-2023
res = zeit.ewmacd(ndvi, train_end=2017)                   # trained on 2013-2016
res.break_time.plot()
disturbed_now = res.state == 1

px = zeit.ewmacd(ndvi.isel(x=120, y=80), train_end=2017)  # one pixel: its chart
(px.ewma / px.limit).plot.line("o")                       # out of control beyond ±1
```

## BFAST family

Ports of R's `bfast` package: [`bfast_monitor`](#bfast_monitor) (near-real-time monitoring), [`bfast_lite`](#bfast_lite) (multiple breakpoints in one pass) and [`bfast`](#bfast) (classic iterative trend and season breaks). Tutorials: [BFAST Monitor](../tutorials/bfast_monitor.md), [BFAST Lite](../tutorials/bfast_lite.md), [BFAST](../tutorials/bfast.md).

Like `landtrendr` and `ccdc`, each is one function for every input: it reads what `data` is and returns an `xarray.Dataset` with one variable per metric, georeferenced when the input is. All parameters but `data` (and `monitor_start`) are keyword-only. They replace `run_bfast_monitor_image`, `run_bfast_lite_image`, `run_bfast_image`, `zeit.bfast.run_bfast_monitor_dask`, `run_bfast_lite_dask`, `run_bfast_dask` and the accessor methods `DataArray.zeit.run_bfast_monitor`, `run_bfast_lite` and `run_bfast`; the accessor forms are now [`DataArray.zeit.bfast_monitor`](xarray.md#bfast_monitor), [`bfast_lite`](xarray.md#bfast_lite) and [`bfast`](xarray.md#bfast). The engine module is internal (`zeit._bfast`).

### `bfast_monitor` { .api #bfast_monitor }

<!-- sig: zeit.bfast_monitor -->
```python
zeit.bfast_monitor(
    data, monitor_start, *, dates=None, start_time=None,
    frequency=None, order=3, h=0.25, period=10, alpha=0.05,
    min_valid=10, band=None, nodata="auto", chunks=None, n_jobs=-1,
)
```

Near-real-time disturbance monitoring (`bfastmonitor`, Verbesselt et al. 2012): fits a linear trend plus `order` harmonics on the history before `monitor_start`, then flags the first observation after it where the OLS-MOSUM process crosses its boundary (`type="OLS-MOSUM"`, `history="all"`). Every pixel runs in parallel in the Rust core.

`data` can be (the same for [`bfast_lite`](#bfast_lite), [`bfast`](#bfast), [`mann_kendall`](time-series.md#mann_kendall) and [`phenology`](time-series.md#phenology)):

| Input | Example | Where the dates come from |
| :--- | :--- | :--- |
| A raster file, folder, glob or Zarr/NetCDF store: anything [`load_raster`](data.md#load_raster) reads | `"ndvi_16d.tif"` | The dates `load_raster` finds (band descriptions such as `2020-01-15`, file names, a `time` coordinate) |
| A `(time, y, x)` cube, numpy- or dask-backed | the output of `load_raster` or `regularize_time_series` | Its `time` coordinate. A dask cube stays lazy. |
| A `(time, band, y, x)` cube or a `Dataset` | `cube` with `band="ndvi"` | Its `time` coordinate; `band` names the index |
| A numpy array `(time, y, x)` | `stack` | `dates=` |
| One pixel's series: a list, 1-D array or `pandas.Series` | `[0.71, 0.74, 0.69, ...]` | `dates=`, or the Series' index of dates |

**The time axis.** The BFAST family works on a regular series, like R's `ts`: observation `i` is at `start_time + i / frequency`, in decimal years. Both are read from the dates: `frequency` from their median spacing (23 for 16-day composites, 12 for monthly, 1 for annual data) and `start_time` from the first date. Composite irregular data to a fixed step first ([`regularize_time_series`](data.md#regularize_time_series)), with NaN for the gaps. A series without dates (numpy without `dates=`, a plain list) needs both `start_time` and `frequency`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | What to monitor (see the table above). |
| `monitor_start` | `str`, `datetime` or `float` | required | Start of the monitoring period: a date (`"2022-01-01"`) or a decimal year (`2022.0`). Observations before it are the history. |
| `dates` | list | `None` | Date of each time step. Only for numpy input, single series and cubes without dates; otherwise taken from the `time` coordinate. |
| `start_time` | `float` | `None` | Decimal year of the first observation. Default: from the first date. |
| `frequency` | `int` | `None` | Observations per year. Default: from the median spacing of the dates. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.25` | MOSUM window, as a fraction of the history: `0.25`, `0.5` or `1.0`. |
| `period` | `int` | `10` | Boundary horizon, in history lengths: `2`, `4`, `6`, `8` or `10`. |
| `alpha` | `float` | `0.05` | Significance level. |
| `min_valid` | `int` | `10` | Minimum valid history observations. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset` (band or variable name). |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. `"auto"`: the raster's NoData value; for integer data without one, `0` (how Earth Engine exports masked pixels). A number: that value. `None`: only NaN. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads the raster into memory; `"auto"` or a dict of chunk sizes keeps it lazy, so the result is computed block by block, for rasters larger than memory. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

`h` and `period` index a table of simulated critical values (from `strucchangeRcpp`), so other values raise an error.

**Returns** an `xarray.Dataset` of `(y, x)` float32 maps:

| Variable | Meaning |
| :--- | :--- |
| `breakpoint` | Decimal year of the first detected break on R's regular `ts` axis (`start_time + breakpoint_idx / frequency`); NaN if none. |
| `breakpoint_idx` | 0-based index of that observation in the series. |
| `magnitude` | Median residual over the monitoring period: size and direction of the shift (negative: lower than expected). |
| `sigma` | Residual standard error of the history fit. |
| `n_history` | Valid observations in the history. |
| `has_break` | `1` if a break was detected, else `0`. |
| `valid` | `0` if the history was too short to fit; the other variables are then NaN. |
| `break_date` | Date of the break's observation, from the data's dates (`NaT` if none). |

The `x`/`y` coordinates and CRS of the input are kept, so the result goes straight to [`save_raster`](data.md#save_raster): one GeoTIFF per variable for a folder, or one band per variable for a `.tif` path. `attrs` records the parameters, including the `start_time` and `frequency` used. A single pixel's result has no `y`/`x` dims. With a dask cube or `chunks=`, the result is lazy.

```python
ndvi = zeit.load_raster("ndvi_16d.tif")       # (time, y, x): 16-day composites from January 2010
bfm = zeit.bfast_monitor(ndvi, "2022-01-01")  # start_time 2010.0 and frequency 23, from the dates
alerts = (bfm.has_break == 1) & (bfm.magnitude < -0.1)
zeit.save_raster(bfm, "bfm_out")              # breakpoint.tif, ..., valid.tif

# A raster larger than memory: lazy, computed block by block while it is written
bfm = zeit.bfast_monitor("ndvi_16d.tif", 2022.0, chunks="auto")
zeit.save_raster(bfm, "bfm_out")

# One pixel's series without dates: give the time axis
px = zeit.bfast_monitor(values, 2022.0, start_time=2010.0, frequency=23)
float(px.breakpoint)                          # decimal year, NaN if no break
```

### `bfast_lite` { .api #bfast_lite }

<!-- sig: zeit.bfast_lite -->
```python
zeit.bfast_lite(
    data, *, dates=None, start_time=None, frequency=None, order=3,
    h=0.15, max_breaks=5, min_valid=20, band=None, nodata="auto",
    chunks=None, n_jobs=-1,
)
```

Multiple breakpoints in one pass (`bfastlite`, Masiliūnas et al. 2021): fits `response ~ trend + harmon` and chooses the number and position of breaks with the Bai & Perron dynamic program and the LWZ criterion. Input, time axis and output are as for [`bfast_monitor`](#bfast_monitor).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | What to segment (see [`bfast_monitor`](#bfast_monitor)). |
| `dates`, `start_time`, `frequency` | | `None`, `None`, `None` | The time axis, as in `bfast_monitor`. |
| `order` | `int` | `3` | Seasonal harmonics. |
| `h` | `float` | `0.15` | Minimum segment size, as a fraction of the observations. |
| `max_breaks` | `int` | `5` | Break slots in the output (formerly `max_breaks_output`). |
| `min_valid` | `int` | `20` | Minimum valid observations. |
| `band`, `nodata`, `chunks`, `n_jobs` | | `None`, `"auto"`, `None`, `-1` | As in `bfast_monitor`. |

</div>

**Returns** an `xarray.Dataset` of `(y, x)` float32 maps:

| Variable | Meaning |
| :--- | :--- |
| `n_breaks` | Number of breaks chosen by the criterion (`0` if none). |
| `rss` | Residual sum of squares of the selected model. |
| `lwz` | Value of the LWZ criterion. |
| `n_valid` | Valid observations used. |
| `valid` | `1` if the series had enough observations to fit. |
| `breakpoint_idx_1` … `breakpoint_idx_{max_breaks}` | 0-based index in the series (missing observations included) of the last observation before each break, in chronological order; NaN past `n_breaks`. |
| `magnitude_1` … `magnitude_{max_breaks}` | The model after each break minus the model before it, both on the first observation after the break (the seasonal terms cancel, the change of level, trend and season shape remains). |
| `break_date_1` … `break_date_{max_breaks}` | Date of the first observation after each break (`NaT` past `n_breaks`). |

```python
bfl = zeit.bfast_lite(ndvi, max_breaks=3)
first = bfl.break_date_1                                  # NaT where n_breaks == 0
loss = zeit.extract_events(bfl, event_type="loss")        # the greatest drop of each pixel
```

### `bfast` { .api #bfast }

<!-- sig: zeit.bfast -->
```python
zeit.bfast(
    data, *, dates=None, start_time=None, frequency=None, order=3,
    h=0.15, max_breaks_trend=5, max_breaks_season=5, max_iter=10,
    level=0.05, min_valid=20, band=None, nodata="auto", chunks=None,
    n_jobs=-1,
)
```

Classic iterative BFAST (Verbesselt et al. 2010): an STL seasonal seed, then alternating trend and season segmented regressions (breaks chosen with BIC) until neither changes, each preceded by an OLS-MOSUM stability test. Separates trend breaks from seasonal breaks. Input, time axis and output are as for [`bfast_monitor`](#bfast_monitor); the series must hold more than `2 × frequency` observations.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | What to decompose (see [`bfast_monitor`](#bfast_monitor)). |
| `dates`, `start_time`, `frequency` | | `None`, `None`, `None` | The time axis, as in `bfast_monitor`. |
| `order` | `int` | `3` | Harmonics of the season model. |
| `h` | `float` | `0.15` | Minimum segment size, as a fraction of the valid observations. |
| `max_breaks_trend`, `max_breaks_season` | `int` | `5`, `5` | Trend and season break slots in the output. |
| `max_iter` | `int` | `10` | Maximum trend/season iterations. |
| `level` | `float` | `0.05` | Significance of the stability pre-test (`1.0` always searches for breaks). |
| `min_valid` | `int` | `20` | Minimum valid observations. |
| `band`, `nodata`, `chunks`, `n_jobs` | | `None`, `"auto"`, `None`, `-1` | As in `bfast_monitor`. |

</div>

**Returns** an `xarray.Dataset` of `(y, x)` float32 maps:

| Variable | Meaning |
| :--- | :--- |
| `n_trend_breaks`, `n_season_breaks` | Number of trend and season breaks at convergence. |
| `magnitude` | Largest jump in the trend (R's `bf$Magnitude`); `0` without a trend break. |
| `break_time` | Decimal year of that jump (R's `bf$jump$x`); NaN if none. Formerly the `time` metric, renamed so it does not clash with the `time` coordinate. |
| `n_iter` | Iterations until convergence (or `max_iter`). |
| `n_valid` | Valid observations used. |
| `valid` | `1` if the series was long enough to fit. |
| `trend_breakpoint_idx_1` … `_{max_breaks_trend}` | 0-based index in the series (missing observations included) of the last observation before each trend break; NaN past `n_trend_breaks`. |
| `season_breakpoint_idx_1` … `_{max_breaks_season}` | The same for season breaks. |
| `trend_magnitude_1` … `_{max_breaks_trend}` | The jump of the trend at each trend break; `magnitude` is the largest of them. |
| `trend_break_date_1` … `_{max_breaks_trend}` | Date of the first observation after each trend break. |

```python
bf = zeit.bfast(ndvi)
abrupt = (bf.n_trend_breaks > 0) & (bf.magnitude < -0.1)
when = bf.break_time.where(abrupt)
zeit.save_raster(bf, "bfast.tif")                 # one band per variable, named
```

## BEAST (Bayesian changepoints, season and trend)

### `beast` { .api #beast }

<!-- sig: zeit.beast -->
```python
zeit.beast(
    data, *, season="harmonic", period=None, freq=None, deltat=None,
    start=None, times=None, scp_minmax=(0, 10), sorder_minmax=(0, 5),
    tcp_minmax=(0, 10), torder_minmax=(0, 1), sseg_minlength=None,
    tseg_minlength=None, samples=8000, burnin=200, chains=3, thin=5,
    seed=42, prec_value=1.5, prec_type="componentwise", ci=True,
    max_missing_rate=0.75, dates=None, band=None, nodata="auto",
    chunks=None, n_jobs=-1,
)
```

BEAST, the Bayesian estimator of abrupt change, seasonal change and trend (Zhao et al. 2019). Each series is decomposed into a piecewise harmonic season and a piecewise linear (or polynomial) trend. Instead of choosing one model, reversible-jump MCMC samples models of every number and position of changepoints and every harmonic order, each in proportion to how well it fits and to its prior, and the outputs average over them. The result therefore says how sure each answer is:

- the probability of a changepoint at each date (`trend_cp_occ`, `season_cp_occ`);
- the probability of each number of changepoints;
- the season and the trend, with their 95% credible intervals;
- the trend's slope and the probability that it rises;
- the season's amplitude.

From these, the most probable changepoints are listed with their probabilities, the jump of the component at each and their intervals. Every pixel runs in parallel in the Rust core.

It is a port of Rbeast 1.0.2's C code (Kaiguang Zhao, GPL ≥ 2): its `beast()` on one series, with the generic kernels (`cputype = "sse"`). The same seed gives the same chain and the same float32 outputs, to the last bit. This was checked on the Nile, Rbeast's simulated series, its Yellowstone NDVI with gaps, its irregular Ohio Landsat series (`beast.irreg()`), a short series and the uniform and constant priors (`tests/data/make_beast_fixtures.R`). Each pixel of a cube is fitted as `beast()` fits one series: from the seed, whatever the other pixels or the threads. Rbeast's `beast123()` instead carries its random stream from one pixel to the next. Zeit differs from Rbeast in two outputs:

- The slope's credible interval is the mean over the chains, in the slope's units; Rbeast sums it over the chains, per time step.
- `ncp_mode`, `ncp_median` and the percentiles are given with one chain too; Rbeast leaves them at 0.

By default Rbeast picks the vector kernels of the processor (AVX2), which sum in another order. Their chain then departs from the generic one, so the results agree in distribution but not bit for bit. On one series of 360 months, Rbeast's AVX2 kernels take 0.13 s and Zeit 0.19 s, as fast as Rbeast's generic kernels. On a cube of 10,000 pixels × 144 months, `beast123()` takes 328 s on one thread and 34.5 s on 19; Zeit takes 217 s and 18.1 s ([Against the Original Implementations](../benchmarks/vs-originals.md#beast-vs-rbeast)).

An irregular series (dates missing, or several in one step) is first put on a regular grid of step `deltat`, each step the mean of the values in it, as `beast.irreg()` does. Dates become decimal years, and the season's `period` defaults to 1.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset`, path, `pd.Series`, `ndarray` or list | required | One index: a `(time, y, x)` cube, a raster, a `Dataset` with `band=`, or one pixel's series, as for [`mann_kendall`](time-series.md#mann_kendall). Values alone are a regular series from `start` every `deltat`. |
| `season` | `str` | `"harmonic"` | `"harmonic"` (a season of sines and cosines) or `"none"` (a trend alone). |
| `period` | `float` | `None` | The season's period, in the units of the times. With dates it is in years, and the default is 1. Otherwise it is needed for a harmonic season (12 for a monthly series counted in months). |
| `freq` | `float` | `None` | The values per period instead (`period = freq * deltat`), Rbeast's older argument. |
| `deltat` | `float` | `None` | The grid's step. The default is the series' own step when it is regular. Otherwise, for dates it is the nearest of a day, half a month, a month and a week to the mean spacing, as in Rbeast (a 16-day series needs `deltat=1/23`). For `times` it is the period divided by the number of whole steps the mean spacing fits into it. Without times it is 1. |
| `start` | `float` | `None` | The time of the grid's first value: the first time, or 1 without times. |
| `times` | array | `None` | Numeric times of the values in any unit, with `period` and `deltat` in the same unit; instead of dates. |
| `scp_minmax`, `tcp_minmax` | `(int, int)` | `(0, 10)` | The least and most changepoints of the season and of the trend. |
| `sorder_minmax` | `(int, int)` | `(0, 5)` | The least and most harmonics of the season (from 1 to half the period minus one). |
| `torder_minmax` | `(int, int)` | `(0, 1)` | The least and most degree of the trend's polynomials (flat or linear segments). |
| `sseg_minlength`, `tseg_minlength` | `int` | `None` | The shortest segment of the season and of the trend, in steps. The default is half a period, or 3 without a season. |
| `samples`, `burnin`, `chains`, `thin` | `int` | `8000`, `200`, `3`, `5` | The MCMC: samples kept per chain, iterations discarded first, chains, and how many iterations apart the kept samples are. |
| `seed` | `int` | `42` | The random stream's seed (Rbeast's `mcmc.seed`): the same seed gives the same result. |
| `prec_value`, `prec_type` | `float`, `str` | `1.5`, `"componentwise"` | The coefficients' prior precision, and its model: `"componentwise"` (one for the season, one for the trend), `"uniform"` (one for all) or `"constant"` (fixed). |
| `ci` | `bool` | `True` | The 95% credible intervals of the season, the trend and the slope (Rbeast's `dump.ci`). They draw from the random stream, so the chain differs with and without them. |
| `max_missing_rate` | `float` | `0.75` | A series missing more than this fraction of its grid is not fitted (NaN). |
| `dates`, `band`, `nodata`, `chunks`, `n_jobs` | | `None`, `None`, `"auto"`, `None`, `-1` | As for [`mann_kendall`](time-series.md#mann_kendall). A dask cube stays lazy. |

</div>

**Returns** an `xarray.Dataset` on the regular `time` grid (decimal years for dates), `(y, x)` maps for a cube:

| Variable | Meaning |
| :--- | :--- |
| `trend`, `trend_sd`, `trend_ci` | The trend, its standard deviation and its 95% interval (`bound`: lower, upper), per date. |
| `trend_slope`, `trend_slope_sd`, `trend_slope_ci` | Its slope per unit of time, with the slope's standard deviation and interval. |
| `trend_slope_pos_prob`, `trend_slope_zero_prob` | The probability that the slope is positive, and that it is zero (a flat segment). |
| `trend_order` | The mean degree of the trend at each date. |
| `trend_cp_occ` | The probability of a trend changepoint at each date. |
| `trend_ncp`, `trend_ncp_mode`, `trend_ncp_median`, `trend_ncp_pct10`, `trend_ncp_pct90` | The mean, the mode, the median and the 10th and 90th percentiles of the number of trend changepoints. |
| `trend_ncp_prob` | `(tncp)`: the probability of 0, 1, ... trend changepoints. |
| `trend_cp`, `trend_cp_prob`, `trend_cp_abrupt`, `trend_cp_ci` | `(tcp)`: the trend changepoints found, most probable first, with NaN after the last. For each: its probability, the jump of the trend there, and its interval of dates (`bound`). |
| `season`, `season_sd`, `season_ci`, `season_order`, `season_cp_occ`, `season_ncp`... | The same for the season (dimensions `sncp`, `scp`); `season_order` is the mean number of harmonics. |
| `season_amp`, `season_amp_sd` | Rbeast's amplitude of the season at each date (the sum of the squared harmonic coefficients: a sine of amplitude 0.2 gives 0.04), and its standard deviation. |
| `data` | The series on the grid (the mean of each step for an irregular one). |
| `marg_lik`, `sig2`, `r2`, `rmse` | The fit: the marginal likelihood, the noise variance, R² and RMSE. |

A pixel not fitted (too many missing values) is NaN in every variable.

```python
ndvi = zeit.load_raster("modis_ndvi_16d/*.tif")
res = zeit.beast(ndvi, deltat=1 / 23)                    # 16-day composites, a yearly season
res.trend_ncp.plot()                                     # how many trend changes
when = res.trend_cp.isel(tcp=0).where(res.trend_cp_prob.isel(tcp=0) > 0.5)

px = zeit.beast(ndvi.isel(x=120, y=80), deltat=1 / 23)   # one pixel
px.trend.plot()
px.trend_ci.plot.line(x="time", hue="bound")             # the band around it
px.trend_cp_occ.plot()                                   # where it changes
```

Reference: Zhao, K., Wulder, M. A., Hu, T., Bright, R., Wu, Q., Qin, H., Li, Y., Toman, E., Mallick, B., Zhang, X., & Brown, M. (2019). Detecting change-point, trend, and seasonality in satellite time series data to track abrupt changes and nonlinear dynamics: A Bayesian ensemble algorithm. *Remote Sensing of Environment*, 232, 111181.

## Two dates and Sentinel-1 series

Tutorial: [Change between two dates and in Sentinel-1 series](../tutorials/change-bitemporal.md). Four functions that take the change between two dates, or in a radar series, from the statistics of the pixels themselves: [`cva`](#cva) (the magnitude and direction of the vector of band differences), [`imad`](#imad) (multivariate alteration detection, which needs no radiometric normalization), [`radcal`](#radcal) (the normalization that iMAD's unchanged pixels give) and [`omnibus`](#omnibus) (the omnibus test for Sentinel-1). They agree with the scripts of Mort Canty's *Image Analysis, Classification and Change Detection in Remote Sensing* (MIT licence), run as black boxes: see [Against the Original Implementations](../benchmarks/vs-originals.md#imad-radcal-and-the-omnibus-test-vs-canty).

### `cva` { .api #cva }

<!-- sig: zeit.cva -->
```python
zeit.cva(
    data, after=None, *, dates=(0, -1), bands=None, roles=None, direction="c2va",
    polar_bands=None, n_sectors=4, c2va_sectors=None, labels=None, scale="none",
    threshold="em", em_init=0.2, sample=1_000_000, nodata="auto", chunks=None, n_jobs=-1,
)
```

Change vector analysis of two dates. For each pixel the change vector holds the difference (after minus before) of every band; its length `rho` is the **magnitude** (Malila 1980) and a pixel has changed when `rho` exceeds the **threshold**. The **direction** is one of three descriptions:

| `direction` | What it is |
| :--- | :--- |
| `"c2va"` (default) | The angle, in degrees from 0 to 180, between the vector and the diagonal of uniform change: `acos(sum(d) / (sqrt(B) rho))`. 0 is every band up by the same amount, 90 is changes that sum to zero, 180 is every band down by the same amount. Any number of bands (the compressed CVA of Bovolo, Marchesi and Bruzzone 2012); a pixel with no change at all has no direction (NaN). |
| `"polar"` | The angle `atan2(d2, d1)` in degrees, 0 to 360, of the two bands `polar_bands`, cut into `n_sectors` equal sectors from 0 degrees (Malila 1980; Bovolo and Bruzzone 2007). With 4 sectors: 1 both up, 2 the second up and the first down, 3 both down, 4 the first up and the second down. |
| `"sign"` | The sector from 1 to 8 of the signs of the changes of red, NIR and SWIR1: `1 + [red up] + 2 [NIR up] + 4 [SWIR1 up]` (a change of exactly 0 is not up). 6 is vegetation loss (clearing to soil), 3 vegetation gain, 5 a burn scar, 1 darker in every band, 8 brighter in every band; the labels are a heuristic. |

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | A cube `(time, band, y, x)` (with `dates` choosing two of its dates), a date `(band, y, x)` with `after` the second, or anything [`load_raster`](data.md#load_raster) reads. Both dates need the same bands in the same order on the same grid, and must be radiometrically consistent ([`radcal`](#radcal) first if they are not). |
| `after` | `DataArray` or path | `None` | The second date when `data` is the first. |
| `dates` | pair | `(0, -1)` | The two dates of a cube: indices, labels of its `time` coordinate or date strings (the nearest date). |
| `bands` | list | `None` | The bands the vector is made of (names or indices); default all. A pixel is valid when all of them are valid at both dates. |
| `roles` | `dict` | `None` | For `"sign"`: the band of each of `red`, `nir`, `swir1` when their names are not the usual ones. |
| `direction` | `str` | `"c2va"` | `"c2va"`, `"polar"` or `"sign"`. |
| `polar_bands`, `n_sectors` | pair, `int` | `None`, `4` | For `"polar"`: the two bands, and the number of sectors. |
| `c2va_sectors` | list | `None` | For `"c2va"`: edges in degrees of sectors of the angle, e.g. `[0, 60, 120, 180]`. Without it `"c2va"` has no `sector` band. |
| `labels` | `dict` | `None` | `{sector: name}` replacing the default labels. |
| `scale` | `str` | `"none"` | `"std"` divides each band's difference by its standard deviation over the sample, so that every band weighs the same. |
| `threshold` | number or `str` | `"em"` | A number (units of the magnitude, after the scale); `"otsu"` (Otsu 1979: 1024 bins up to the 99.9th percentile of the sample); or `"em"` (Bruzzone and Prieto 2000): a two-class Gaussian mixture (no change, change) fitted to the magnitudes by EM, and the root between its means of the equation of the minimum error of Bayes' rule (Otsu's value if there is none). |
| `em_init` | `float` | `0.2` | The width of the EM seed sets around the middle of the magnitude range. |
| `sample` | `int` or `None` | `1_000_000` | The threshold and the `scale` are estimated on a lattice of this many valid pixels (every s-th row and column from the first: deterministic); `None`: every pixel. The output is for every pixel. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy, block by block with the threshold fixed. |
| `n_jobs` | `int` | `-1` | Threads for an in-memory cube. |

</div>

**Returns** `(band, y, x)` float32 with the bands `magnitude`, `direction` (degrees; with `"sign"` the sector), `change` (1 or 0) and, where the direction has sectors, `sector` (0 where nothing changed); missing pixels are NaN. `attrs["cva"]` is a JSON record: the threshold and how it was found (for EM the shares, means and standard deviations of the classes, the iterations and whether it converged; for Otsu the between-class variance and the bin width), the scale of each band and `sectors`: label, pixel count, area in hectares (grids in metres) and mean magnitude of each. `zeit.cva_table(result)` returns the sectors as a `DataFrame`.

```python
out = zeit.cva(cube, dates=("2021-07-16", "2022-07-27"), direction="sign", threshold="otsu")
zeit.cva_table(out)                         # pixels, hectares and mean magnitude of each sector
loss = out.sel(band="sector") == 6          # vegetation loss
```

### `imad` { .api #imad }

<!-- sig: zeit.imad -->
```python
zeit.imad(
    data, after=None, *, dates=(0, -1), bands=None, max_iter=50, tol=1e-3, lam=0.0,
    penalty="ridge", standardize="weighted", alpha=0.01, canonical=False,
    sample=1_000_000, nodata="auto", chunks=None, n_jobs=-1,
)
```

The iteratively re-weighted multivariate alteration detection (iMAD, IR-MAD) of two dates. The canonical correlation analysis of the bands of the two dates finds the linear combinations of each that agree most; their differences, the **MAD variates**, are uncorrelated change images ordered by variance (Nielsen, Conradsen and Simpson 1998). Each pixel's `chi2`, the sum of the squared standardized MAD variates, is approximately chi-square distributed with one degree of freedom per band where nothing changed, and `p_nochange` is its upper tail. Each iteration weights every pixel by the `p_nochange` of the last one in the means and covariances (Nielsen 2007; Canty and Nielsen 2008), until the canonical correlations stop changing. With `lam = 0` the result does not depend on the gain, the offset or any linear transform of either date, so the dates need no normalization first.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data`, `after`, `dates`, `bands`, `nodata`, `chunks`, `n_jobs` | | | As in [`cva`](#cva). |
| `max_iter` | `int` | `50` | At most this many iterations; `1` is the original, not iterated MAD. |
| `tol` | `float` | `1e-3` | Stops when no canonical correlation changed by more than this since the iteration before (`0`: always `max_iter`). 1e-3 is where Canty's `iMad.py` stops (same iteration counts and correlations on the benchmark data); the correlations still creep up by about 1e-5 per iteration afterwards. A warning says when `max_iter` came first. |
| `lam` | `float` | `0.0` | Nielsen's (2007) regularization for many correlated bands: `lam` times a penalty matrix is added to the covariances of the constraints. In the units of the band variances, and it breaks the invariance to linear transforms: scale the bands alike first. With `0`, collinear or constant bands are an error naming the date. |
| `penalty` | `str` | `"ridge"` | `"ridge"` (the length of the coefficient vectors) or `"curvature"` (their second differences along the bands, which must be in order of wavelength). |
| `standardize` | `str` | `"weighted"` | The standard deviation the MAD variates are divided by. `"weighted"` (Canty and Nielsen 2008, and Canty's scripts): `sqrt(2 (1 - rho))`, that under the current weights. `"all"` (Nielsen 2007): the unweighted standard deviation over all the pixels. See the note. |
| `alpha` | `float` | `0.01` | `change` is `p_nochange < alpha`. |
| `canonical` | `bool` | `False` | Also return the canonical variates `U1..UN` (date 1) and `V1..VN` (date 2). |
| `sample` | `int` or `None` | `1_000_000` | The statistics are computed on a lattice of this many valid pixels, then the transformation is applied to every pixel. `None`: all of them. |

</div>

**Returns** `(band, y, x)` float32 with the bands `MAD1..MADN` (MAD1 has the largest variance, the pair of the smallest canonical correlation; Canty's scripts number them the other way), `chi2`, `p_nochange`, `change` and, with `canonical`, `U1..UN`, `V1..VN` (descending correlation). `attrs["imad"]` is a JSON record: the canonical correlations of every iteration, `iterations`, `converged`, the coefficient matrices `a` and `b` (columns are canonical pairs), the means, the standard deviations and `loadings_x`, `loadings_y` (the correlation of each MAD variate with each band of each date). `zeit.imad_table(result)` returns the correlations of every iteration as a `DataFrame`.

!!! note "The weighted standardization over-flags where nothing changed"
    The weights shrink the variance that the weighted standardization estimates, so on data with no change at all the chi-square exceeds a chi-square of N degrees of freedom and `change` flags more than `alpha` of the pixels (15 to 30 % at `alpha = 0.01` on simulated pairs, and the same in Canty's script). It is what the published method and its reference implementation do, so it is the default; `standardize="all"` keeps the false alarms near `alpha` on such data but weights changed pixels down less. The ranking of the pixels by `chi2` is nearly the same either way.

```python
change = zeit.imad(cube, dates=("2021-07-16", "2022-07-27"))
zeit.imad_table(change).tail()              # the correlations settle
change.sel(band="p_nochange")               # 1 where nothing changed, 0 where something did
```

### `radcal` { .api #radcal }

<!-- sig: zeit.radcal -->
```python
zeit.radcal(
    data, after=None, *, reference=0, targets=None, threshold=0.95, holdout=1/3,
    bands=None, max_iter=50, tol=1e-3, lam=0.0, sample=1_000_000, nodata="auto",
    chunks=None, n_jobs=-1,
)
```

Relative radiometric normalization of dates to a reference date by the invariant pixels of iMAD (Canty, Nielsen and Schmidt 2004; Canty and Nielsen 2008). For each target date [`imad`](#imad) against the reference finds the pixels that did not change (`p_nochange` above `threshold`); an **orthogonal regression** of each band of the reference on the same band of the target, fitted on two thirds of them, gives its gain and offset (it allows for the error in both dates, unlike least squares); the other third tests the result with a paired t-test of the means and an F-test of the variances of the reference and the normalized target (approximate: the samples are paired). The normalized target is `a + b * target` for every pixel. The fit needs at least 30 invariant pixels.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | A cube `(time, band, y, x)` whose dates are normalized to `reference`, or the reference date `(band, y, x)` when `after` is the target (the result is then the normalized target). |
| `after` | `DataArray` or path | `None` | The target date, when `data` is the reference. |
| `reference` | date | `0` | The reference date of a cube (index, label or date string); copied unchanged. |
| `targets` | list | `None` | The dates to normalize (default every other); the others are copied unchanged. |
| `threshold` | `float` | `0.95` | The invariant pixels have `p_nochange` above this. About 5 % of the pixels that did not change pass 0.95, so give an image of a few hundred pixels per side at least. |
| `holdout` | `float` | `1/3` | The share of invariant pixels kept for the tests: every `round(1 / holdout)`-th in raster order (deterministic). `0`: all are used for the fit, no tests. |
| `bands` | list | `None` | The bands used and normalized; the others are not in the result. |
| `max_iter`, `tol`, `lam`, `sample`, `nodata`, `chunks`, `n_jobs` | | | As in [`imad`](#imad). |

</div>

**Returns** float32, the shape of the cube (or `(band, y, x)`), with the normalized dates; a missing pixel is NaN. `attrs["radcal"]` is a JSON record: `fits`, one row per target date and band (`a`, `se_a`, `b`, `se_b`, `r`, `rmse`, `n_train`, `n_test`, the means of target, reference and normalized target on the hold-out pixels, `t`, `p_t`, the three variances, `F`, `p_F`) and `imad`, the summary of each pair's iMAD with its number of invariant pixels. `zeit.radcal_table(result)` returns the fits as a `DataFrame`.

```python
norm = zeit.radcal(cube, reference="2021-07-16")        # every date to the first
zeit.radcal_table(norm)[["date", "band", "a", "b", "r", "p_t"]]
```

### `omnibus` { .api #omnibus }

<!-- sig: zeit.omnibus -->
```python
zeit.omnibus(
    data, *, bands=("vv", "vh"), enl=4.4, alpha=0.01, approximation="improved",
    per_date=False, median=0, nodata="auto", chunks=None, n_jobs=-1,
)
```

When the backscatter of each pixel of a Sentinel-1 series changed: the omnibus test of Conradsen, Nielsen and Skriver (2016) and Canty, Nielsen, Conradsen and Skriver (2020). The series of VV and VH (or one band) of every pixel is tested for the equality of all its dates by the likelihood ratio of independent multilook (complex Wishart) observations. If it rejects at `alpha`, the statistic factorizes into one test for each date against those before it since the last change: the first that rejects is a change, the series restarts at that date, and so on, so that every registered change has the false alarm probability `alpha`. The p-values use the improved approximations of Conradsen, Skriver and Nielsen (2024), accurate where Wilks' chi-square is not at few looks (on simulated speckle of 4.4 looks the test at `alpha = 0.01` rejects 0.98 % of the pixels with the improved p-values, 1.5 % with the chi-square). The direction of a change is the Loewner order of the difference between the date and the mean of those before it: brightening (up in every channel), darkening (down in every channel) or indefinite. The statistics are invariant to a per-pixel factor constant over the series, so sigma0 and gamma0 of one relative orbit give the same maps.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | A cube `(time, band, y, x)` of **linear** power (a negative median, which means dB, is refused) of **one relative orbit and pass**: the geometry changes the backscatter and the test reads that as change (a warning when `sat:relative_orbit` or `sat:orbit_state` varies). Values not positive (layover, shadow), NaN or nodata are skipped date by date: a pixel uses its own valid dates, a change is dated at the next valid date, a pixel with fewer than two valid dates is NaN. |
| `bands` | list | `("vv", "vh")` | The VV and VH bands by name or position; `("vv",)` for one polarization. |
| `enl` | `float` | `4.4` | The equivalent number of looks: 4.4 for Sentinel-1 IW GRDH. RTC products are resampled and differ: estimate it with [`estimate_enl`](#estimate_enl). |
| `alpha` | `float` | `0.01` | The significance level of each test. |
| `approximation` | `str` | `"improved"` | `"improved"` or `"chi2"`, Wilks' chi-square (as the Earth Engine tutorial of Canty et al.; more false alarms at few looks). |
| `per_date` | `bool` | `False` | Also return the direction at each date `(time, y, x)`: 1, 2 or 3 at the dates of changes, 0 elsewhere, NaN where the date is not valid. |
| `median` | `int` | `0` | The median filter of the p-values of Canty's scripts is not implemented (it is not rigorous): only `0`. |
| `nodata`, `chunks`, `n_jobs` | | | As in [`imad`](#imad). |

</div>

**Returns** `(band, y, x)` float32 with the bands, in this order, `first_change`, `last_change`, `n_changes`, `first_direction`, `last_direction`, `greatest_change`, `p_change`, `n_dates`. The three dates are decimal years (NaN where nothing changed); `greatest_change` is the change with the largest test statistic; directions are 0 (nothing), 1 (brightening), 2 (darkening), 3 (indefinite); `p_change` is the p-value of the omnibus test of the whole series. `attrs["omnibus"]` is a JSON record (ENL, `alpha`, approximation, bands, dates). With `per_date` the result is `(summary, per_date)`.

A test of one new date against the mean of those before it has limited power: a change is dated at its own date in a part of the pixels and a date or two later in the rest, and about `(j - 1) alpha` of the pixels report a change `j - 1` dates too early.

```python
s1 = zeit.load_raster("s1_rtc_orbit83/*.tif")            # (time, band: vv, vh, y, x), linear gamma0
enl = zeit.estimate_enl(s1, region=(120, 135, 40, 55))["enl"]["vv"]     # intact forest
change, direction = zeit.omnibus(s1, enl=enl, alpha=0.01, per_date=True)
change.sel(band="first_change")                          # when it first changed, as a decimal year
```

### `estimate_enl` { .api #estimate_enl }

<!-- sig: zeit.estimate_enl -->
```python
zeit.estimate_enl(data, *, bands=("vv", "vh"), region=None)
```

The equivalent number of looks of the speckle over a homogeneous area. On a uniform target the intensity of an `L`-look image has mean `m` and variance `m**2 / L`, so `L = mean**2 / variance`, made over the area for each date and band; the median over the dates is the estimate. Texture, a trend or an edge in the area lowers it: choose intact forest or calm water. The estimators of Anfinsen, Doulgeris and Eltoft (2009) are not implemented. `region` is `None` (the whole image), a pixel window `(row0, row1, col0, col1)` (end exclusive) or a boolean mask `(y, x)`. **Returns** `{"enl": {band: estimate}, "per_date": {band: [...]}, "n_pixels": count}`.

References: Malila (1980), *Proc. 6th Machine Processing of Remotely Sensed Data Symposium*, 326-335. Bovolo, Marchesi and Bruzzone (2012), *IEEE TGRS* 50(6), 2196-2212. Bovolo and Bruzzone (2007), *IEEE TGRS* 45(1), 218-236. Bruzzone and Prieto (2000), *IEEE TGRS* 38(3), 1171-1182. Otsu (1979), *IEEE SMC* 9(1), 62-66. Nielsen, Conradsen and Simpson (1998), *Remote Sensing of Environment* 64(1), 1-19. Nielsen (2007), *IEEE TIP* 16(2), 463-478. Canty and Nielsen (2008), *Remote Sensing of Environment* 112(3), 1025-1036. Canty, Nielsen and Schmidt (2004), *Remote Sensing of Environment* 91(3-4), 441-451. Conradsen, Nielsen and Skriver (2016), *IEEE TGRS* 54(5), 3007-3024. Canty, Nielsen, Conradsen and Skriver (2020), *Remote Sensing* 12(1), 46. Conradsen, Skriver and Nielsen (2024), *IEEE JSTARS* 17, 16359-16376. Nielsen, Skriver and Conradsen (2019), *IEEE GRSL* (the Loewner order).
