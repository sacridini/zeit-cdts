# Time-Series Analysis

<p class="lead">Trend tests, phenology, pattern matching, segmentation and clustering.</p>

## Trends

### `mann_kendall` { .api #mann_kendall }

<!-- sig: zeit.mann_kendall -->
```python
zeit.mann_kendall(
    data, method="hamed_rao", alpha=0.05, lag=None, period=1,
    min_valid=4, dates=None, band=None, nodata="auto", chunks=None,
    n_jobs=-1,
)
```

Mann-Kendall trend test and Theil-Sen slope for every pixel, a C++ port of `pymannkendall`. One function for every input, like [`bfast_monitor`](change-detection.md#bfast_monitor): it reads what `data` is and returns an `xarray.Dataset` with one variable per metric, georeferenced when the input is. All parameters but `data` are keyword-only. It replaces `zeit.trend.run_mann_kendall_dask`, `run_mann_kendall_image` and `DataArray.zeit.run_mann_kendall`; the accessor form is now [`DataArray.zeit.mann_kendall`](xarray.md#mann_kendall). Tutorial: [Trend Analysis](../tutorials/mann_kendall.md).

`data` is any input of the [BFAST family](change-detection.md#bfast_monitor): a raster `load_raster` reads, a `(time, y, x)` cube (a dask cube stays lazy), a `(time, band, y, x)` cube or `Dataset` with `band=`, a numpy array or one pixel's series. No dates are needed: the test runs on the order of the observations, so the slope is per **time step** (one value per year gives a slope per year).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | The series to test, one value per time step. |
| `method` | `str` | `"hamed_rao"` | `"original"`, `"hamed_rao"` (autocorrelation-corrected, for annual composites), `"yue_wang"` or `"seasonal"` (Hirsch & Slack, pools `period` season slots). |
| `alpha` | `float` | `0.05` | Significance level. |
| `lag` | `int` | `None` | Lags used by the autocorrelation corrections. `None`: all. |
| `period` | `int` | `1` | Observations per cycle, for `"seasonal"` (e.g. `23` for 16-day data). |
| `min_valid` | `int` | `4` | Pixels with fewer valid values are NaN. |
| `dates` | list | `None` | Date of each time step. Not needed: the test only uses the order of the observations. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN, as in [`bfast_monitor`](change-detection.md#bfast_monitor). |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; `"auto"` or a dict keeps them lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset` of `(y, x)` float32 maps (no `y`/`x` dims for one pixel; lazy for a dask cube or `chunks=`):

| Variable | Meaning |
| :--- | :--- |
| `trend` | `1` increasing, `-1` decreasing, `0` no significant trend. |
| `h` | `1` if significant at `alpha`, else `0`. |
| `p` | Two-sided p-value. |
| `z` | Standardised test statistic. |
| `tau` | Kendall's tau. |
| `s`, `var_s` | Mann-Kendall score and its (corrected) variance. |
| `slope` | Theil-Sen slope, per time step (per `period` for `"seasonal"`). |
| `intercept` | Intercept of the robust line. |

```python
annual = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")   # one NDVI value per year
mk = zeit.mann_kendall(annual)
browning = (mk.h == 1) & (mk.trend == -1)
zeit.save_raster(mk, "mk_rondonia")                       # slope.tif, p.tif, ...

# 16-day composites tested directly: 23 observations per year
mk16 = zeit.mann_kendall(ndvi_16d, method="seasonal", period=23)

# One series
zeit.mann_kendall([0.41, 0.44, 0.39, 0.47, 0.52, 0.49, 0.55, 0.58, 0.61, 0.60]).slope.item()
```

## Phenology

### `phenology` { .api #phenology }

<!-- sig: zeit.phenology -->
```python
zeit.phenology(
    data, curve="beck", method="threshold", weights=None, dates=None,
    annual=True, max_seasons=None, whittaker_lambda=10.0,
    apply_whittaker=True, apply_hants=False, hants_frequencies=3,
    hants_threshold=0.1, min_season_length=0, min_amplitude=0.0,
    min_pixel_amplitude=0.1, season_retry=True, band=None,
    nodata="auto", chunks=None, n_jobs=-1,
)
```

Land surface phenology, following the methodology of R `phenofit`: smooths each pixel's series (Whittaker or HANTS), splits it into seasons, fits a curve to each season and extracts 19 transition metrics plus the fit's R² and RMSE, every pixel in parallel in C++ / OpenMP. One function for every input, like [`bfast_monitor`](change-detection.md#bfast_monitor) (the same `data` table), with dates at any spacing. It replaces `zeit.phenology.run_phenology_dask` and `DataArray.zeit.run_phenology`; the accessor form is now [`DataArray.zeit.phenology`](xarray.md#phenology), and the engine module is internal (`zeit._phenology`). Tutorial: [Phenology](../tutorials/phenology.md).

The day numbering of the core (days since January 1st of the first year) and the first year are computed from the dates, so no day-of-year array is needed.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | A vegetation-index series with dates (see [`bfast_monitor`](change-detection.md#bfast_monitor)). |
| `curve` | `str` or `int` | `"beck"` | Fitted curve: `"beck"`, `"elmore"`, `"gu"`, `"klos"`, `"zhang"`, `"ag"` or `"dl"` (or its integer code). |
| `method` | `str` or `int` | `"threshold"` | `"threshold"`, `"derivative"`, `"gu"` or `"klosterman"` (or its integer code). Recorded in `attrs`; every metric family is computed whatever the method. |
| `weights` | `DataArray` or array | `None` | Per-observation reliability weights in `[0, 1]`, aligned with the data (e.g. from [`qc_sentinel2_scl`](data.md#qc_sentinel2_scl)). |
| `dates` | list | `None` | Date of each time step, for numpy input and series without dates. |
| `annual` | `bool` | `True` | `True`: one value per calendar year, `(year, y, x)`, dates as day of year. `False`: one value per detected season, `(season, y, x)`, dates as days since January 1st of the first year (`attrs["base_year"]`). |
| `max_seasons` | `int` | `None` | Years (with `annual`) or seasons per pixel. Default: the number of years in the series. |
| `whittaker_lambda` | `float` | `10.0` | Whittaker smoothness. |
| `apply_whittaker` | `bool` | `True` | Smooth with Whittaker before fitting. |
| `apply_hants` | `bool` | `False` | Smooth with HANTS (harmonics) instead. |
| `hants_frequencies` | `int` | `3` | HANTS harmonics. |
| `hants_threshold` | `float` | `0.1` | HANTS outlier threshold. |
| `min_season_length` | `int` | `0` | Drop seasons shorter than this many days. |
| `min_amplitude` | `float` | `0.0` | Drop seasons with a smaller amplitude. |
| `min_pixel_amplitude` | `float` | `0.1` | Skip pixels whose whole series varies less than this (water, urban). |
| `season_retry` | `bool` | `True` | Retry pixels with no season once with a relaxed trough threshold. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN, as in [`bfast_monitor`](change-detection.md#bfast_monitor). |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; `"auto"` or a dict keeps them lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset` with 21 variables, each `(year, y, x)` (`annual=True`, `year` coordinate from the first year) or `(season, y, x)` (`season` = 1, 2, …): `TRS2.sos`, `TRS2.eos`, `TRS5.sos`, `TRS5.eos`, `TRS6.sos`, `TRS6.eos`, `DER.sos`, `DER.pos`, `DER.eos`, `UD`, `SD`, `DD`, `RD`, `Greenup`, `Maturity`, `Senescence`, `Dormancy`, `LOS`, `POP`, `R2`, `RMSE` (see [the 19 metrics](../tutorials/phenology.md#the-19-metrics)). NaN where no season was found. The names contain dots, so select them with brackets: `pheno["TRS5.sos"]`. The result is georeferenced and lazy for a dask cube or `chunks=`; [`save_raster`](data.md#save_raster) writes one GeoTIFF per metric with one band per year (`2019`, `2020`, …) or season.

```python
ndvi_16d = zeit.regularize_time_series(ndvi, freq="16D", method="median")   # (time, y, x)
pheno = zeit.phenology(ndvi_16d, curve="beck", min_season_length=45, min_amplitude=0.15)
sos = pheno["TRS5.sos"]                         # (year, y, x), day of year
late = sos.sel(year=2021) - sos.sel(year=[2019, 2020]).mean("year")
zeit.save_raster(pheno, "pheno_out")            # TRS2.sos.tif, ..., one band per year

# Double cropping: up to 3 seasons per pixel instead of one value per year
seasons = zeit.phenology(ndvi_16d, annual=False, max_seasons=3)
```

## Pattern matching (TWDTW)

Tutorial: [Pattern Matching](../tutorials/twdtw.md).

### `twdtw` { .api }

<!-- sig: zeit.twdtw -->
```python
zeit.twdtw(
    data, patterns, band=None, steepness=0.1, midpoint=50.0,
    cycle="year", max_elapsed=None, nodata="auto", chunks=None,
    n_jobs=-1,
)
```

Classifies time series by Time-Weighted Dynamic Time Warping (Maus et al. 2016): each pattern, a typical series of a class, is matched against every pixel's series, and the class of a pixel is the pattern with the lowest distance. The distance is that of the R package [twdtw](https://cran.r-project.org/package=twdtw) (checked against it to 1e-12): the pattern may match any stretch of the series, each pair of matched observations costs their Euclidean distance plus a logistic time weight

$$
w(\Delta t) = \frac{1}{1 + e^{-\text{steepness}\,(\Delta t - \text{midpoint})}}
$$

of the days $\Delta t$ between them, and dates a pixel has no value for are left out of its series. One function for every input; it replaces `classify_twdtw`, `run_twdtw` and `run_twdtw_batch`.

| Input | Example | Where the dates come from |
| :--- | :--- | :--- |
| A cube `(time, y, x)`, or `(time, band, y, x)` for several bands; anything `load_raster` reads | `ndvi` | Its `time` coordinate. A dask cube stays lazy. |
| One pixel: a `pandas.Series` (or a `DataFrame`, one column per band) indexed by dates | `series` | Its index |

Patterns are `{name: series}`: a `pandas.Series` indexed by dates, a `DataFrame` with one column per band, or a `DataArray` with `time` (and `band`). With several bands, the cube's bands are taken by the patterns' column names.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path, `pd.Series` or `pd.DataFrame` | required | The series to classify (see the table above). |
| `patterns` | `dict` | required | `{name: pattern}`, one per class. Names cannot hold spaces. |
| `band` | `str` | `None` | A cube with a `band` dim and one-band patterns: the band to classify. |
| `steepness` | `float` | `0.1` | Steepness of the logistic time weight, per day. |
| `midpoint` | `float` | `50.0` | Days apart at which a pair costs half a unit more (R's `time_weight = c(0.1, 50)`). |
| `cycle` | `str` or `None` | `"year"` | `"year"`: the days between dates are counted between days of the year, around it (31 December and 1 January are a day apart), so a pattern of one year matches the same season of any year. `None`: between the dates themselves. |
| `max_elapsed` | `float` | `None` | Pairs farther apart than this many days are never matched. `None`: no limit. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation: the raster's NoData (`0` for integer data without one), a number, or `None`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset`:

| Variable | Dims | Meaning |
| :--- | :--- | :--- |
| `label` | `(y, x)` | The class: `1` for the first pattern, `2` for the second… (`0`: no observation). Its `flag_meanings` hold the names, which `zeit.plot` shows in the legend. |
| `distance` | `(y, x)` | TWDTW distance to that pattern. High values mean no pattern fits well. |
| `distances` | `(pattern, y, x)` | Distance to every pattern. |
| `pattern_value`, `pattern_time` | `(pattern, pattern_step[, band])` | The patterns, so that `zeit.plot(cube, fit=result)` draws the best one aligned with a clicked pixel's series. `save_raster` skips them. |

```python
patterns = {"soy": soy, "pasture": pasture, "forest": forest}   # pandas Series indexed by dates
classes = zeit.twdtw(ndvi, patterns)
classes.label.zeit.plot(basemap="satellite")                     # legend: soy, pasture, forest
unknown = classes.label.where(classes.distance < 3, 0)          # 0 where nothing fits
zeit.plot(ndvi, fit=classes)                                     # click a pixel: its best match
zeit.save_raster(classes, "twdtw")                               # label.tif, distance.tif, distances.tif
```

## Segmentation (SNIC)

Tutorial: [Segmentation](../tutorials/snic.md).

### `run_snic` { .api }

<!-- sig: zeit.segmentation.run_snic -->
```python
zeit.segmentation.run_snic(
    data, spacing=10, compactness=0.5, seeds=None, grid="rectangular",
    padding=None, tile_size=None, random_state=None, n_jobs=-1,
)
```

SNIC superpixels of an image or a whole cube; every leading axis becomes a feature. Given the same seeds, labels match the reference C implementation. Also exported as `zeit.run_snic`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `np.ndarray` | required | `(y, x)`, `(feature, y, x)` or `(time, band, y, x)`. Pixels with any NaN are left unlabelled (`-1`). |
| `spacing` | `float` or pair | `10` | Seed spacing in pixels. |
| `compactness` | `float` | `0.5` | Regularity of the segments, in data units. Higher is more compact. |
| `seeds` | `(n, 2)` array | `None` | Explicit `(row, col)` seeds; overrides the grid. |
| `grid` | `str` | `"rectangular"` | `"rectangular"`, `"diamond"`, `"hexagonal"` or `"random"`. |
| `padding` | `float` or pair | `None` | Seed-free margin. Default `spacing / 2`. |
| `tile_size` | `int` or pair | `None` | Segment tiles of this size independently, in parallel. |
| `random_state` | `int` | `None` | Seed for `grid="random"`. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** a `SnicResult` with `labels` `(y, x)`, `means` `(n_seeds, *features)`, `centroids`, `sizes` and `seeds`.

### `snic_to_polygons` { .api }

<!-- sig: zeit.segmentation.snic_to_polygons -->
```python
zeit.segmentation.snic_to_polygons(
    result, transform=None, crs=None, include_means=False,
)
```

Converts SNIC labels to a GeoDataFrame, one polygon per segment, like `sits_segment()`. Also exported as `zeit.snic_to_polygons`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `result` | `SnicResult` | required | Output of `run_snic`. |
| `transform` | `Affine` | `None` | Geotransform of the image (identity: pixel coordinates). |
| `crs` | | `None` | Coordinate reference system. |
| `include_means` | `bool` | `False` | Add each segment's mean features as columns `f0, f1, …`. |

</div>

### `snic_grid` { .api }

<!-- sig: zeit.segmentation.snic_grid -->
```python
zeit.segmentation.snic_grid(
    shape, spacing, padding=None, type="rectangular",
    random_state=None,
)
```

The seed grids of the R `snic` package, as `(n, 2)` `(row, col)` positions. Also exported as `zeit.snic_grid`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `shape` | `(int, int)` | required | Image `(rows, cols)`. |
| `spacing` | `float` or pair | required | Seed spacing. |
| `padding` | `float` or pair | `None` | Seed-free margin. |
| `type` | `str` | `"rectangular"` | Grid type, as in `run_snic`'s `grid`. |
| `random_state` | `int` | `None` | Seed for `"random"`. |

</div>

## Clustering (SOM)

### `SOM` { .api .cls }

<!-- sig: zeit.ai.SOM -->
```python
class zeit.ai.SOM(
    x, y, input_len, sigma=1.0, learning_rate=0.5,
    decay_function="asymptotic_decay",
    neighborhood_function="gaussian", topology="rectangular",
    random_seed=42, sigma_decay_function="asymptotic_decay",
)
```

Self-organizing map in C++ (online and batch, OpenMP). An operation-by-operation port of Python [`minisom`](https://github.com/JustGlowing/minisom) 2.3: with the same `random_seed` and arguments the trained codebook is bit-for-bit identical to `MiniSom.train` (`algorithm="online"`) or `MiniSom.train_batch_offline` (`algorithm="batch"`), 30–190 times faster. Tutorial: [Clustering](../tutorials/som.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `x`, `y` | `int` | required | Grid size. |
| `input_len` | `int` | required | Features per sample. |
| `sigma` | `float` | `1.0` | Initial spread of the neighbourhood function. |
| `learning_rate` | `float` | `0.5` | Initial learning rate. |
| `decay_function` | `str` | `"asymptotic_decay"` | `"asymptotic_decay"`, `"inverse_decay_to_zero"` or `"linear_decay_to_zero"`. |
| `neighborhood_function` | `str` | `"gaussian"` | `"gaussian"`, `"mexican_hat"`, `"bubble"` or `"triangle"`. |
| `topology` | `str` | `"rectangular"` | `"rectangular"` or `"hexagonal"`. |
| `random_seed` | `int` | `42` | Seed of the `numpy.random.RandomState` used for initialisation and sample order (the same draws as `minisom`). |
| `sigma_decay_function` | `str` | `"asymptotic_decay"` | `"asymptotic_decay"`, `"inverse_decay_to_one"` or `"linear_decay_to_one"`. |

</div>

| Method | Description |
| :--- | :--- |
| `random_weights_init(data)` / `pca_weights_init(data)` | Initialise the weights from random samples / from the first two principal components. |
| `train(data, num_iters, n_jobs=-1, algorithm="online", random_order=False, use_epochs=False)` | Train on `(samples, features)`, continuing from the current weights. `"online"`: `num_iters` single-sample updates (epochs with `use_epochs=True`). `"batch"`: `num_iters` passes over the data, parallel with `n_jobs`, same result for any `n_jobs`. |
| `predict(data, n_jobs=-1)` | Flat index `i * y + j` of each sample's best-matching unit. |
| `winner(x)` | Grid coordinates `(i, j)` of one sample's best-matching unit. |
| `quantization(data, n_jobs=-1)` / `quantization_error(data, n_jobs=-1)` | Best-matching codebook vector of each sample / mean distance to it. |
| `get_weights()` | Codebook, shape `(x, y, input_len)`. |
| `filter_noisy_samples(data, labels, n_jobs=-1)` | Boolean mask of samples to **keep**: `False` where a sample's label disagrees with the majority label of its neuron. |

```python
from zeit.ai import SOM

som = SOM(x=10, y=10, input_len=X.shape[1], sigma=1.5)
som.random_weights_init(X)
som.train(X, num_iters=20, algorithm="batch")    # 20 passes, OpenMP-parallel
clusters = som.predict(X)
```
