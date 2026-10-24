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

Mann-Kendall trend test and Theil-Sen slope for every pixel, a Rust port of `pymannkendall`. One function for every input, like [`bfast_monitor`](change-detection.md#bfast_monitor): it reads what `data` is and returns an `xarray.Dataset` with one variable per metric, georeferenced when the input is. All parameters but `data` are keyword-only. It replaces `zeit.trend.run_mann_kendall_dask`, `run_mann_kendall_image` and `DataArray.zeit.run_mann_kendall`; the accessor form is now [`DataArray.zeit.mann_kendall`](xarray.md#mann_kendall). Tutorial: [Trend Analysis](../tutorials/mann_kendall.md).

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

### `ar_trend` { .api #ar_trend }

<!-- sig: zeit.ar_trend -->
```python
zeit.ar_trend(
    data, time="step", harmonics=0, period=None, gaps="exact",
    min_valid=0, dates=None, band=None, nodata="auto", chunks=None,
    n_jobs=-1,
)
```

Linear trend with AR(1) errors, fitted by REML, for every pixel: remotePARTS's `fitAR` (Ives et al. 2021, *Remote Sensing of Environment* 266, 112678), on the Rust core. The series is `intercept + slope * t` (plus harmonics, for sub-annual data) and its errors follow an AR(1) process whose coefficient `rho` is estimated with the trend, so the t-test of the slope accounts for the autocorrelation of the series. A parametric alternative to [`mann_kendall`](#mann_kendall), and the per-pixel step of [`parts`](#parts). Same inputs and outputs as `mann_kendall`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset`, `ndarray`, list or `pd.Series` | required | The series, as for [`mann_kendall`](#mann_kendall). |
| `time` | `str` | `"step"` | `"step"`: `t = 1, 2, ...` as remotePARTS (slope per time step); `"year"`: decimal years of the dates (slope per year). |
| `harmonics` | `int` | `0` | Pairs of harmonics (cos, sin) of the seasonal cycle in the design. |
| `period` | `float` | `None` | Observations per cycle of the harmonics, for `time="step"`; from the dates when not given. |
| `gaps` | `str` | `"exact"` | `"exact"`: observations `k` dates apart are correlated by `rho**k`; `"ignore"`: the valid observations are taken as consecutive, as remotePARTS's `fitAR`. |
| `min_valid` | `int` | `0` | Pixels with fewer valid values are NaN (at least the design's columns + 2). |
| `dates` | list | `None` | Date of each time step (needed for `time="year"`). |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; `"auto"` or a dict keeps them lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset` of `(y, x)` float32 maps: for each coefficient (`intercept`, `slope`, `cos1`, `sin1`, ...) its value and `_se`, `_t`, `_p` (two-sided); `rho`, `mse` (the variance of the innovations) and `loglik`.

```python
ar = zeit.ar_trend(annual)                        # slope per year for annual composites
greening = (ar.slope > 0) & (ar.slope_p < 0.05)
ar16 = zeit.ar_trend(ndvi_16d, time="year", harmonics=2)   # 16-day data, slope per year
```

### `parts` { .api #parts }

<!-- sig: zeit.parts -->
```python
zeit.parts(
    data, trend=None, response="slope", classes=None, covariates=None,
    mask=None, partsize=1500, npart=None, ncross=6, nugget="ml",
    correlation="exponential", range=None, shape=None, fit_n=1000,
    scaled=False, seed=0, time="step", harmonics=0, period=None,
    gaps="exact", dates=None, band=None, nodata="auto", n_jobs=-1,
)
```

The regional test of the pixels' trends by PARTS (remotePARTS's `fitGLS_partition`, Ives et al. 2021). Counting the pixels with a significant trend says little about a region: neighbouring pixels are correlated, and thousands of tests are run at once. PARTS answers the regional questions with valid p-values: does the mean trend of the region differ from 0, do the trends differ between land-cover classes, do they depend on a covariate. It fits each pixel's trend ([`ar_trend`](#ar_trend)), fits the spatial correlation of the pixels to the correlations of their residual series (`fitCor`), and then a GLS of the trends on the design in random partitions of the pixels, combined into a t-test of each coefficient and a chi-square test of the design, both corrected for the correlation between partitions.

A step-by-step guide with figures, a worked example and the pitfalls: [Regional Trends (PARTS)](../tutorials/regional-trends/index.md).

Distances are in the cube's units: give a cube in projected coordinates. The results match remotePARTS's to 1e-10 for a given nugget; remotePARTS 1.0.4 crosses the partitions after the first without their nugget, which Zeit does not.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset` or `ndarray` | required | A `(time, y, x)` cube, in projected coordinates. |
| `trend` | `Dataset` | `None` | An [`ar_trend`](#ar_trend) result to test instead of fitting it. |
| `response` | `str` | `"slope"` | The variable of the trend tested. |
| `classes` | `(y, x)` layer | `None` | Classes (land cover, ...): a mean trend per class, and the chi-square test that they differ. |
| `covariates` | `dict` | `None` | `{name: (y, x) layer}`: their effects on the trend (with a mean), and the chi-square test of them. |
| `mask` | `(y, x)` layer | `None` | The pixels tested (a region of interest). |
| `partsize` | `int` | `1500` | Pixels per partition (remotePARTS uses 1500 to 3000). |
| `npart` | `int` | `None` | Partitions; all the pixels by default. A few hundred are often enough: the correlation between partitions bounds what more of them gain. |
| `ncross` | `int` | `6` | Partitions crossed to measure the correlation between partitions. |
| `nugget` | `"ml"` or `float` | `"ml"` | Each partition's maximum-likelihood nugget, or a given one. |
| `correlation` | `str` | `"exponential"` | `"exponential"`, `exp(-d / range)`, or `"exponential_power"`, `exp(-(d / range) ** shape)`. |
| `range` | `float` | `None` | The correlation's range when known; fitted to the residuals when `None`. |
| `shape` | `float` | `None` | The shape of `"exponential_power"`, when known. |
| `fit_n` | `int` | `1000` | Pixels sampled to fit the correlation. |
| `scaled` | `bool` | `False` | Distances divided by their largest in each matrix (remotePARTS's default `distm_scaled`), or in the cube's units. |
| `seed` | `int` | `0` | Seed of the samples and partitions. |
| `time` | `str` | `"step"` | As for [`ar_trend`](#ar_trend). |
| `harmonics` | `int` | `0` | As for [`ar_trend`](#ar_trend). |
| `period` | `float` | `None` | As for [`ar_trend`](#ar_trend). |
| `gaps` | `str` | `"exact"` | As for [`ar_trend`](#ar_trend). |
| `dates` | list | `None` | Date of each time step. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** a `PartsResult`: `table` (a `DataFrame` of the coefficients: `estimate`, `se`, `t`, `p`), `p_chisqr` (the chi-square test of the design against a single mean, `None` without classes or covariates; remotePARTS floors it at 1e-6), `f`, `df1`, `df2`, `correlation` (the function and its parameters), `nuggets`, `partitions` (each partition's coefficients and nugget), `npart`, `partsize`, `n_pixels`.

```python
res = zeit.parts(annual, classes=landcover, partsize=1500, npart=200)
print(res)                       # the mean trend of each class, and whether they differ
res.table.loc["class 3", "p"]
zeit.parts(annual, mask=roi)     # does the region's mean trend differ from 0?
```

### `emerging_hotspots` { .api #emerging_hotspots }

<!-- sig: zeit.emerging_hotspots -->
```python
zeit.emerging_hotspots(
    data, radius=1, shape="square", lag=1, reference="date",
    alpha=0.05, trend="hamed_rao", band=None, nodata="auto",
    n_jobs=-1,
)
```

Emerging hot spot analysis (ESRI's Space Time Pattern Mining; R sfdep's `emerging_hotspot_analysis`): where the values cluster high (hot spots) or low (cold spots), and how that changed over the dates. For each date, the Getis-Ord Gi* z-score of each pixel (Getis & Ord 1992): the sum of its neighbourhood (the pixel among it) against the mean, the neighbourhood the pixels within `radius` at that date and the `lag` dates before. A hot spot is a date whose Gi* is significantly positive, a cold spot significantly negative. Then the Mann-Kendall test of each pixel's Gi* series, and its pattern as ESRI's documentation defines them:

| Code | Pattern | Definition (hot spots; the cold ones, negative, alike) |
| :--- | :--- | :--- |
| 1 | new | hot at the last date only |
| 2 | consecutive | one run of hot dates, at the end |
| 3 | intensifying | hot on 90 % of the dates or more, the last one too, its Gi* rising significantly |
| 4 | persistent | the same, without a significant trend |
| 5 | diminishing | the same, its Gi* falling significantly |
| 6 | sporadic | hot at the last date, on and off before, never cold |
| 7 | oscillating | hot at the last date, cold at some date before |
| 8 | historical | hot on 90 % of the dates, not at the last |
| 0 | no pattern | |

Where two definitions hold, the first of the table. Each date's Gi* is computed from row prefix sums, a few operations a pixel whatever the radius's area, and the trends in parallel.

The Gi* equals sfdep 0.2.5's (`include_gi = TRUE`, queen neighbours, `k` the lag) to 1e-14, and the trend test Kendall's to its single precision (tests/data/make_hotspots_sfdep_fixtures.R). sfdep's patterns differ: it takes the significance of each Gi* from 199 permutations and has rules of its own (it calls sporadic a pixel not significant at the last date); Zeit's follow ESRI's definitions with the normal p-values of the Gi*. As sfdep's, the Gi* of a date measures the dates before it in its neighbourhood against that date's mean: with a lag, a rise everywhere lowers it (`reference="cube"` measures every date against the whole cube's). On a grid of 50 x 50 cells and 12 dates, sfdep takes 13 minutes (28 s with 9 permutations); Zeit 4 to 7 ms (benchmarks/vs_originals/hotspots.py).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | path, `DataArray`, `Dataset` or `ndarray` | required | One variable as a `(time, y, x)` cube (an index, a count, an anomaly). Missing values are left out of the sums and statistics. |
| `radius` | `int` | `1` | The neighbourhood in pixels: with `shape="square"` the (2 `radius` + 1)² pixels around (1: the 8 neighbours, queen contiguity, sfdep's default); with `"circle"` those whose centres are within it. |
| `shape` | `str` | `"square"` | `"square"` or `"circle"`. |
| `lag` | `int` | `1` | The dates before each date in its neighbourhood (sfdep's `k`, ESRI's neighbourhood time step). |
| `reference` | `str` | `"date"` | `"date"`: each date's Gi* against that date's mean and standard deviation (as sfdep: a place hot against the others at that date); `"cube"`: against the whole cube's (a date high everywhere is hot everywhere). |
| `alpha` | `float` | `0.05` | The significance of the Gi* (two-sided, normal) and of its trend. |
| `trend` | `str` | `"hamed_rao"` | The trend test of the Gi* series: `"hamed_rao"` (the Gi* of consecutive dates share neighbours, so their series are autocorrelated), `"original"` Mann-Kendall (ESRI's and sfdep's) or `"yue_wang"`. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. |
| `n_jobs` | `int` | `-1` | Threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.Dataset`: `category (y, x)` int8, the pattern's code (`-128` with fewer than 3 dates; `flag_values` and `flag_meanings`, the names in `zeit.emerging_hotspots.PATTERNS`); `gi (time, y, x)`, the Gi* z-scores, and `p`, their p-values; `trend_z`, `trend_p` and `tau`, the trend test of the Gi*; `n_hot` and `n_cold`, the dates a significant hot or cold spot.

```python
loss = -zeit.load_raster("ndvi_annual.tif")                  # (time, y, x): where the losses concentrate
ehs = zeit.emerging_hotspots(loss, radius=3, shape="circle")
ehs.category.zeit.plot()                                       # legend: the patterns
new_fronts = ehs.category.isin([1, 2, 3])                     # new, consecutive, intensifying hot spots
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

Land surface phenology, following the methodology of R `phenofit`: smooths each pixel's series (Whittaker or HANTS), splits it into seasons, fits a curve to each season and extracts 19 transition metrics plus the fit's R² and RMSE, every pixel in parallel in the Rust core. One function for every input, like [`bfast_monitor`](change-detection.md#bfast_monitor) (the same `data` table), with dates at any spacing. It replaces `zeit.phenology.run_phenology_dask` and `DataArray.zeit.run_phenology`; the accessor form is now [`DataArray.zeit.phenology`](xarray.md#phenology), and the engine module is internal (`zeit._phenology`). Tutorial: [Phenology](../tutorials/phenology.md).

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

### `snic` { .api }

<!-- sig: zeit.snic -->
```python
zeit.snic(
    data, spacing=10, compactness=0.5, seeds=None, grid="rectangular",
    padding=None, tile_size=None, random_state=None, nodata="auto",
    n_jobs=-1,
)
```

SNIC superpixels of an image or a whole cube: every value of a pixel off its `y`/`x` (each band, each date) is one feature, so two pixels are alike when their whole trajectories are, as `sits_segment(seg_fn = sits_snic())`. Given the same seeds, labels match the reference C implementation. It replaces `run_snic` (the engine, still in `zeit.segmentation`) and the accessor's `run_snic`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `ndarray` or path | required | `(y, x)`, `(band, y, x)`, `(time, y, x)` or `(time, band, y, x)`, read into memory. Pixels with any missing feature are left unlabelled (`-1`). |
| `spacing` | `float` or pair | `10` | Seed spacing in pixels. |
| `compactness` | `float` | `0.5` | Regularity of the segments, in data units. Higher is more compact. |
| `seeds` | `(n, 2)` array | `None` | Explicit `(row, col)` seeds; overrides the grid. |
| `grid` | `str` | `"rectangular"` | `"rectangular"`, `"diamond"`, `"hexagonal"` or `"random"`. |
| `padding` | `float` or pair | `None` | Seed-free margin. Default `spacing / 2`. |
| `tile_size` | `int` or pair | `None` | Segment tiles of this size independently, in parallel. |
| `random_state` | `int` | `None` | Seed for `grid="random"`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** an `xarray.Dataset`: `labels (y, x)` (georeferenced; `-1` unlabelled), `means (segment, ...)` with the cube's dims and coordinates (`(segment, time, band)` for a `(time, band, y, x)` cube), and `n_pixels`, `centroid_x`, `centroid_y (segment)` (map coordinates). `save_raster` writes the labels; [`snic_to_polygons`](#snic_to_polygons) makes polygons.

```python
seg = zeit.snic(ndvi, spacing=8, compactness=0.3)          # ndvi: (time, y, x)
zeit.save_raster(seg, "results/snic")                       # labels.tif
polygons = zeit.snic_to_polygons(seg, include_means=True)   # one column per date
```

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
| `result` | `Dataset` or `SnicResult` | required | Output of [`zeit.snic`](#snic) (its transform, CRS and feature names are used) or of the engine `run_snic`. |
| `transform` | `Affine` | `None` | Geotransform of a `SnicResult` (identity: pixel coordinates); taken from a Dataset. |
| `crs` | | `None` | Coordinate reference system of a `SnicResult`; taken from a Dataset. |
| `include_means` | `bool` | `False` | Add each segment's mean features as columns, named by the cube's coordinates (`2022-01-01_ndvi`, …; `f0, f1, …` for a `SnicResult`). |

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
| `type` | `str` | `"rectangular"` | Grid type, as in `zeit.snic`'s `grid`. |
| `random_state` | `int` | `None` | Seed for `"random"`. |

</div>

## Clustering (SOM)

### `som` { .api }

<!-- sig: zeit.som -->
```python
zeit.som(
    data, x=3, y=3, sample=50000, num_iters=None, algorithm="online",
    sigma=1.0, learning_rate=0.5, decay="linear_decay_to_zero",
    neighborhood="gaussian", topology="rectangular", init="random",
    seed=42, nodata="auto", chunks=None, n_jobs=-1,
)
```

Clusters the pixels of a map or a cube with a self-organizing map. Every value a pixel holds off its `y`/`x` is a feature (each date of an index, each band of each date, each map of a Dataset), so pixels share a neuron when their whole trajectories are alike. The SOM is trained on a random sample of the pixels by the [`SOM`](#som_1) engine (a bit-for-bit port of `minisom`), then every pixel goes to its best-matching neuron. Tutorial: [Clustering](../tutorials/som.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset`, `ndarray` or path | required | A map, a stack, a `(time, y, x)` or `(time, band, y, x)` cube (in memory or dask), a Dataset of maps (e.g. phenology metrics), or anything `load_raster` reads. The features should share a scale: the distance adds them up as they are. |
| `x`, `y` | `int` | `3`, `3` | Neurons of the grid: `x * y` clusters. Small grids cluster; large ones (10 × 10 and up) explore the data. |
| `sample` | `int` or `None` | `50000` | Pixels to train on, drawn at random among those with every feature; `None`: all of them. |
| `num_iters` | `int` | `None` | `"online"`: single-sample updates (default 20 passes over the sample). `"batch"`: passes (default 20). |
| `algorithm` | `str` | `"online"` | `"online"` or `"batch"` (parallel). On small grids the batch update can leave neurons empty; the online one does not, and is about as fast on 50,000 samples. |
| `sigma`, `learning_rate` | `float` | `1.0`, `0.5` | Initial neighbourhood radius (grid units) and learning rate. |
| `decay` | `str` | `"linear_decay_to_zero"` | Online: how the learning rate falls. Ending at 0 lets the prototypes settle; `minisom`'s default, `"asymptotic_decay"`, ends at a third of it, and the last samples seen still pull them (on 30,000 Rondônia trajectories: a mean distance 10% larger, prototypes up to twice as far from their pixels' mean). Or `"inverse_decay_to_zero"`. |
| `neighborhood` | `str` | `"gaussian"` | `"gaussian"`, `"mexican_hat"`, `"bubble"` or `"triangle"`. |
| `topology` | `str` | `"rectangular"` | `"rectangular"` or `"hexagonal"`. |
| `init` | `str` or `None` | `"random"` | Initial prototypes: `"random"` (sample pixels), `"pca"` (first two principal components; poor on a grid one neuron wide) or `None` (`minisom`'s random unit vectors). |
| `seed` | `int` | `42` | Seed of the sample, the initial weights and the sample order. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation. Pixels with a missing feature get no neuron. |
| `chunks` | `"auto"`, `dict` | `None` | A raster path: read lazily. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** an `xarray.Dataset`: `label (y, x)` (the neuron, `1` to `x * y`; `0` without one; the grid positions `i_j` in `flag_meanings`), `distance (y, x)` (to the neuron's prototype: large where the map represents the pixel poorly), `prototypes (neuron, ...)` with the cube's dims and coordinates (`prototypes.sel(neuron=3)` is a series over `time`) and `n_pixels (neuron)`; the grid position of each neuron in the `i`/`j` coordinates and the sample's mean distance in `attrs["quantization_error"]`. Georeferenced as the input; a lazy cube stays lazy. With `sample=None` and the same arguments, the prototypes are the engine's and `label` is `SOM.predict + 1`.

```python
ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")     # (time, y, x)
clusters = zeit.som(ndvi, x=2, y=2, sample=30_000)
clusters.prototypes.sel(neuron=1).plot()                  # one typical trajectory
clusters.label.zeit.plot()                                # the cluster map
zeit.plot(ndvi, fit=clusters)                             # click a pixel: its series and its prototype
```

### `clean_samples` { .api }

<!-- sig: zeit.clean_samples -->
```python
zeit.clean_samples(
    data, samples, label="class", x=None, y=None, num_iters=None,
    algorithm="online", sigma=1.0, learning_rate=0.5,
    decay="linear_decay_to_zero", neighborhood="gaussian",
    topology="rectangular", init="random", seed=42, date=None,
    nodata="auto", n_jobs=-1,
)
```

Checks labelled samples with a SOM before a classifier is trained on them, as `sits_som_clean_samples` of the R `sits` package: a SOM is trained on the features of the samples (read as [`train_classifier`](post-processing.md#train_classifier) reads them), and a sample whose class differs from the majority class of its neuron is suspicious.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | as in `train_classifier` | required | What the classifier will be trained on. |
| `samples` | `GeoDataFrame` or path | required | Points with their class, reprojected when in another CRS. |
| `label` | `str` | `"class"` | The column holding the class. |
| `x`, `y` | `int` | `None` | Neurons of the grid; by default a square grid of about `5 * sqrt(n)` neurons for `n` samples. |
| `date` | date | `None` | CCDC segments: the date whose models are the features. |
| `num_iters`, `algorithm`, `sigma`, `learning_rate`, `decay`, `neighborhood`, `topology`, `init`, `seed`, `nodata`, `n_jobs` | | | As in [`som`](#som). |

</div>

**Returns** the samples as given (`GeoDataFrame`, same CRS), with `neuron` (`0`: outside the data or a missing feature), `neuron_class` (the neuron's majority class), `purity` (the share of the neuron's samples in that class) and `keep` (the sample's class is its neuron's; `False` without a neuron).

```python
checked = zeit.clean_samples(stack, "samples.gpkg", label="class")
print(checked[~checked.keep])                              # the suspicious ones
rf = zeit.train_classifier(stack, checked[checked.keep], label="class")
```

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

The engine of [`zeit.som`](#som), for samples you arrange yourself: a self-organizing map in Rust (online and batch, the batch one multi-threaded), an operation-by-operation port of Python [`minisom`](https://github.com/JustGlowing/minisom) 2.3: with the same `random_seed` and arguments the trained codebook is bit-for-bit identical to `MiniSom.train` (`algorithm="online"`) or `MiniSom.train_batch_offline` (`algorithm="batch"`), 30–190 times faster. Tutorial: [Clustering](../tutorials/som.md).

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
som.train(X, num_iters=20, algorithm="batch")    # 20 passes, multi-threaded
clusters = som.predict(X)
```
