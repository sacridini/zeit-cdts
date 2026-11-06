# FORCE's Time Series Analysis Chain

<p class="lead">FORCE's higher-level time series analysis turns an irregular, cloudy index series into a regular one, summarizes it by year and season, and reads trends and changes from the summaries. Zeit has each step as a function, on any cube, in memory or dask: interpolate, spectral-temporal metrics, folds, trend, change-aftereffect-trend and clear-observation statistics.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">What does this pixel's year look like, is it trending, when did it lose most, and how well was it observed?</span></div>
<div><span class="k">Input</span><span class="v">One index with clouds as NaN: <code>(time, rows, cols)</code>, any spacing</span></div>
<div><span class="k">Output</span><span class="v">A regular series, metrics by period, a trend and a change map, and gap statistics</span></div>
<div><span class="k">Reference</span><span class="v">Ported from FORCE's TSA (Frantz 2019); RBF: Schwieder et al. (2016); STM: Müller et al. (2015); CAT: Hird et al. (2016); CSO: Ernst et al. (2018); polar metrics: Brooks et al. (2020)</span></div>
</div>

## How it works

FORCE's chain, and the Zeit function of each step:

| Step | FORCE | Zeit |
| :--- | :--- | :--- |
| Regular series every N days | TSI (RBF, linear, moving average, harmonic) | [`zeit.interpolate`](../api/preprocessing.md#interpolate) |
| Statistics of the series | STM | [`zeit.temporal_metrics`](../api/time-series.md#temporal_metrics) |
| Statistics by year, quarter, month, week or day | FBY, FBQ, FBM, FBW, FBD | `temporal_metrics(..., by=)` |
| Z-score or centering | STANDARDIZE | [`zeit.standardize`](../api/preprocessing.md#standardize) |
| Linear trend of a fold | TRY, TRQ, TRM, TRW, TRD | [`zeit.linear_trend`](../api/time-series.md#linear_trend) |
| Greatest change, and the trend around it | CAY, CAQ, CAM, CAW, CAD | [`zeit.cat`](../api/time-series.md#cat) |
| Clear observations and the gaps between them | CSO | [`zeit.clear_observations`](../api/time-series.md#clear_observations) |
| Land surface phenology: dates, values, integrals and rates of each season | POL (and TRO, CAO of each metric) | [`zeit.polar_metrics`](../api/time-series.md#polar_metrics) |

The folds are the link: the yearly mean of the interpolated series is a short series of its own, and `linear_trend` and `cat` take it along its `year` axis.

## Step by step

### 1. A cloudy NDVI series

The example builds its own data: five years of NDVI every 5 days on a 40 × 40 grid, with a season, a slow greening, noise and 35% of the observations lost to clouds. The 10 first rows lose their vegetation in July 2020.

```python
import numpy as np
import pandas as pd
import xarray as xr
import zeit

rng = np.random.default_rng(1)
dates = pd.date_range("2018-01-01", "2022-12-31", freq="5D")
ny = nx = 40
doy = dates.dayofyear.values[:, None, None]
year = (dates.year.values - 2018)[:, None, None]

ndvi = 0.55 + 0.25 * np.sin(2 * np.pi * (doy - 110) / 365.25) + 0.01 * year
ndvi = ndvi + rng.normal(0, 0.03, (len(dates), ny, nx))
cut = (dates >= "2020-07-01")[:, None, None] & (np.arange(ny)[None, :, None] < 10)
ndvi = np.where(cut, ndvi - 0.3, ndvi)
ndvi[rng.random(ndvi.shape) < 0.35] = np.nan                       # clouds

ndvi = xr.DataArray(ndvi.astype("float32"), dims=("time", "y", "x"), name="ndvi",
                    coords={"time": dates, "y": np.arange(ny), "x": np.arange(nx)})
```

With real data, `ndvi = zeit.load_raster("ndvi_stack.tif")` gives the same thing (and `chunks="auto"` keeps it lazy).

### 2. Interpolate every 16 days

```python
tsi = zeit.interpolate(ndvi, "rbf", step=16)        # (time, y, x): 114 dates, no gaps
```

The RBF ensemble (Gaussian kernels of 8, 16 and 32 days) fills the gaps from the observations around each date. The grid is FORCE's: from the first date, every 16 days, up to the last. The other methods are `"linear"`, `"moving"` and `"harmonic"`; the harmonic can be fit to part of the series and predict the rest (`fit=("2018-01-01", "2019-12-31")`), the near-real-time mode.

### 3. Spectral-temporal metrics and folds

```python
stm = zeit.temporal_metrics(tsi, ["p10", "median", "p90", "std"])  # (metric, y, x)
yearly = zeit.temporal_metrics(tsi, ["mean"], by="year")           # (metric, year, y, x)
```

`by` takes `"year"`, `"quarter"`, `"month"`, `"week"` or `"doy"`. `skew`, `kurtosis`, `iqr`, `range` and `count` are metrics too. To compare each year with the others, `zeit.standardize(yearly, dim="year")`.

### 4. Trend and change over the yearly folds

```python
trend = zeit.linear_trend(yearly)                   # Dataset: slope, sig, rsq, ...
greening = trend.sig == 1
browning = trend.sig == -1

change = zeit.cat(yearly, penalty=True)             # Dataset: change, change_time, after_slope, ...
change.change_time.isel(y=0, x=0).item()            # 2020.0: the year of the greatest loss
```

`slope` is NDVI per year, because the series has one value per year. `sig` is `1`, `-1` or `0` at 95% (`conf=`). `cat` finds the greatest loss between consecutive years, then fits the trend to the whole series (`total_*`), before the loss (`before_*`) and from it on (`after_*`): a recovery shows as a positive `after_slope`. With `penalty=True`, a dip the series recovers from next year counts for less.

### 5. How well was it observed?

```python
cso = zeit.clear_observations(ndvi, months=3, metrics=["count", "max"])    # (metric, window, y, x)
cso.sel(metric="count").mean(("y", "x"))            # clear observations per quarter
cso.sel(metric="max").mean(("y", "x"))              # the longest gap, in days
```

Each window is a quarter; the gaps run from the window's start to the first clear observation, between them, and from the last to the window's end. A cube of booleans (`ndvi.notnull()`) works too. Low counts and long gaps tell where the interpolation, and the metrics built on it, are least reliable.

### 6. Polar phenology metrics, and the trend of the start of the season

```python
pol = zeit.polar_metrics(tsi, ["DSS", "DES", "VPS", "IGS"])   # (metric, year, y, x)
pol.sel(metric="DSS").mean(("y", "x"))                       # day of the year each season starts
sos = zeit.linear_trend(pol.sel(metric="DSS"))               # days per year: slope, sig, ...
```

Each date of the interpolated series is a vector (the NDVI as the radius, the day of the year as the angle); the mean vector gives the time of least activity, which starts the phenological year, and the values accumulated along each season give its start (`DSS`, 20% of the total), middle and end (`DES`, 80%). `metrics` takes any of the [39 FORCE names](../api/time-series.md#polar_metrics): dates, lengths, values, amplitudes, integrals and rates. With 5 years of data there are 4 seasons, so one year at an end of the series is `NaN` (here 2022, which has no complete season). The result has a `year` axis, so `linear_trend` and `cat` take one metric along it directly: `sos.slope` is the change of the start of the season in days per year (a positive slope is a later start), and with so few seasons `sos.sig` should be read before the slope. The series must be regular, which is what `interpolate(..., step=16)` gives.

!!! tip "Save the maps"
    The results are `DataArray`s and `Dataset`s on the cube's grid: `zeit.save_raster(trend, "results/trend")` writes one GeoTIFF per variable, and `zeit.save_raster(stm, ...)` one per metric.

## Differences from FORCE

- **Floats, not int16.** FORCE scales its outputs by 10000 and rounds; Zeit returns floats (`rsq` from 0 to 1, `kurtosis` as is). Rounded, the values equal FORCE's to within 1 in all the comparisons below.
- **Student's t is exact.** FORCE iterates Hill's approximation for the critical value of the trend's test; Zeit tables the exact quantile.
- **The CAT penalty is corrected.** FORCE compares a candidate's penalized drop with the best one's unpenalized drop; Zeit compares penalized with penalized.
- **CSO over k intervals, and the last window.** FORCE mixes the number of observations with the number of intervals in its standard deviation, skewness, kurtosis and quantiles, and drops the last window when it starts in the month the range ends in; Zeit takes every statistic over the same intervals and keeps the window.
- **Constant series.** `skew` and `kurtosis` are NaN where FORCE leaves them undefined.
- **Polar metrics.** The dates count from 1 January of each season's year (FORCE counts continuous days from the first year's), so a trend along `year` is the change per year; what FORCE leaves undefined and writes as 0 (a season whose total is zero) is `NaN`; `DPV` is 0 without `adaptive`, where FORCE writes `-DPY`.
- **A harmonic pixel that does not converge.** The robust fit can cycle without settling; such a pixel is left out of the comparison with FORCE.

??? info "Validation"
    Against FORCE 3.11.00 on 10,000 pixels, the metrics, folds, standardization, linear and moving interpolation and the harmonic fit to part of the series are equal; the RBF and the full harmonic are within 1 of FORCE's int16, and the 39 polar metrics are equal. See [Against the Original Implementations](../benchmarks/vs-originals.md#force-tsa).

## References

- Frantz, D. (2019). FORCE—Landsat + Sentinel-2 Analysis Ready Data and Beyond. *Remote Sensing*, 11, 1124. [doi:10.3390/rs11091124](https://doi.org/10.3390/rs11091124)
- Schwieder, M., Leitão, P. J., da Cunha Bustamante, M. M., Ferreira, L. G., Rabe, A., & Hostert, P. (2016). Mapping Brazilian savanna vegetation gradients with Landsat time series. *International Journal of Applied Earth Observation and Geoinformation*, 52, 361-370.
- Zhu, Z., Woodcock, C. E., Holden, C., & Yang, Z. (2015). Generating synthetic Landsat images based on all available Landsat data: Predicting Landsat surface reflectance at any given time. *Remote Sensing of Environment*, 162, 67-83. [doi:10.1016/j.rse.2015.02.009](https://doi.org/10.1016/j.rse.2015.02.009)
- Müller, H., Rufin, P., Griffiths, P., Barros Siqueira, A. J., & Hostert, P. (2015). Mining dense Landsat time series for separating cropland and pasture in a heterogeneous Brazilian savanna landscape. *Remote Sensing of Environment*, 156, 490-499.
- Hird, J. N., Castilla, G., McDermid, G. J., & Bueno, I. T. (2016). A simple transformation for visualizing non-seasonal landscape change from dense time series of satellite data. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 9, 3372-3383.
- Ernst, S., Lymburner, L., & Sixsmith, J. (2018). Implications of pixel quality flags on the observation density of a continental Landsat archive. *Remote Sensing*, 10, 1570.
- Brooks, B.-G. J., Lee, D. C., Pomara, L. Y., & Hargrove, W. W. (2020). Monitoring broadscale vegetational diversity and change across North American landscapes using land surface phenology. *Forests*, 11(6), 606. [doi:10.3390/f11060606](https://doi.org/10.3390/f11060606) (the polar metrics, as FORCE's `polar-hl.c` implements them).
- Alken, P., et al. GNU Scientific Library, `multifit/multirobust.c` (the bisquare IRLS of the harmonic fit).
