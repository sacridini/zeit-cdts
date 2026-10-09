# Pre-processing

<p class="lead">Clean time series before analysing them: flag clouds and shadows that the QA band missed, smooth noise, and remove one-off spikes.</p>

## Cloud masking

### `apply_tmask_stack` { .api }

<!-- sig: zeit.tmask.apply_tmask_stack -->
```python
zeit.tmask.apply_tmask_stack(
    dates, green_stack, swir_stack, scale_factor=10000.0,
)
```

Runs Tmask on every pixel of a Green/SWIR1 stack. A robust (Huber) harmonic model is fitted to each band. Observations far above the Green model are flagged as cloud, and far below the SWIR1 model as shadow. Also exported as `zeit.apply_tmask_stack`. Tutorial: [Cloud Masking (Tmask)](../tutorials/tmask.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dates` | `np.ndarray` | required | Day numbers of the observations, e.g. Python ordinal days. |
| `green_stack` | `np.ndarray` | required | Green reflectance, `(time, rows, cols)`. |
| `swir_stack` | `np.ndarray` | required | SWIR1 reflectance, `(time, rows, cols)`. |
| `scale_factor` | `float` | `10000.0` | Divisor that brings the data to 0–1 reflectance. Use `1.0` if it already is. |

</div>

**Returns** a boolean array `(time, rows, cols)`: `True` = clear, `False` = cloud or shadow.

```python
from zeit.tmask import apply_tmask_stack

clear = apply_tmask_stack(dates, green, swir)
ccdc_qa = np.where(clear, 0, 4).astype("uint8")   # CCDC codes: 0 clear, 4 cloud
```

!!! warning
    For CCDC, mark flagged observations with `4` (cloud). Code `1` means water in CCDC and would be treated as clear.

### `run_tmask_pixel` { .api }

<!-- sig: zeit.tmask.run_tmask_pixel -->
```python
zeit.tmask.run_tmask_pixel(
    dates_julian, green_band, swir_band, scale_factor=10000.0,
)
```

Tmask for a single pixel's series. This is the building block of `apply_tmask_stack`. Series with fewer than 5 observations are returned as all clear. Also exported as `zeit.run_tmask_pixel`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dates_julian` | `np.ndarray` | required | Day numbers of the observations. |
| `green_band` | `np.ndarray` | required | Green reflectance series. |
| `swir_band` | `np.ndarray` | required | SWIR1 reflectance series. |
| `scale_factor` | `float` | `10000.0` | Divisor that brings the data to 0–1 reflectance. |

</div>

**Returns** a boolean array, `True` = clear.

```python
from zeit.tmask import run_tmask_pixel

dates = np.array([1, 17, 33, 49, 65, 81, 97])
green = np.array([900, 920, 4500, 910, 895, 905, 930])      # a cloud at index 2
swir = np.array([1200, 1180, 1190, 1210, 1195, 1205, 1188])

run_tmask_pixel(dates, green, swir)
# array([ True,  True, False,  True,  True,  True,  True])
```

## Smoothing

### `smooth` { .api }

<!-- sig: zeit.smooth -->
```python
zeit.smooth(
    data, method="whittaker", lmbda=10.0, weights=None, window=5,
    polyorder=2, nodata="auto", chunks=None, n_jobs=-1,
)
```

Smooths time series along `time`, keeping their dates, georeferencing and type. One function for every input: a cube `(time, y, x)` or `(time, band, y, x)` in memory or dask, anything [`load_raster`](data.md#load_raster) reads, or one pixel's `pandas.Series` indexed by dates. It replaces `apply_whittaker_filter` and `apply_savgol_filter`.

- **Whittaker** (default; Eilers 2003): a penalised least-squares curve that follows the data more or less closely (`lmbda`). The dates may be unevenly spaced: the penalty uses the time between them, measured in median steps of the series, so `lmbda` means the same on any spacing (on an even series it is the classic smoother). Missing observations (NaN, NoData) weigh 0 and are filled by the curve; `weights` down-weights hazy ones.
- **Savitzky-Golay** (`method="savgol"`): a local polynomial over `window` dates, as scipy's `savgol_filter`, for evenly spaced series (regularise first with [`regularize_time_series`](data.md#regularize_time_series)). Missing observations are first interpolated linearly in time.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path or `pd.Series` | required | The series to smooth (see above). |
| `method` | `str` | `"whittaker"` | `"whittaker"` or `"savgol"`. |
| `lmbda` | `float` | `10.0` | Whittaker: smoothness, larger is smoother (`10` light for a 16-day NDVI series, `1000` strong). |
| `weights` | array or `DataArray` | `None` | Whittaker: a weight per observation, same shape as `data` (`1` clear, `0.2` hazy, `0` cloudy; see [`qc_sentinel2_scl`](data.md#qc_sentinel2_scl)). |
| `window` | `int` | `5` | Savitzky-Golay: window length in dates (odd). |
| `polyorder` | `int` | `2` | Savitzky-Golay: order of the local polynomial. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation: the raster's NoData (`0` for integer data without one), a number, or `None`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Whittaker: threads. `-1` uses all cores but one. |

</div>

**Returns** an `xarray.DataArray` with the same dims and coordinates (a `pandas.Series` for a Series). Floats keep their type; integer data with a NoData value (NDVI × 10000 in Int16) is rounded back to its type, NoData where a pixel has no observation at all; integer data without one comes back as float32. `attrs["smoothing"]` names the method, and [`zeit.plot(cube, fit=smoothed)`](plot.md) draws the curve over a clicked pixel's series.

```python
ndvi = zeit.load_raster("S2_ndvi_2022.tif")                    # (time, y, x), clouds as NaN
smooth = zeit.smooth(ndvi, lmbda=100)                          # gaps filled by the curve
weighted = zeit.smooth(ndvi, weights=zeit.qc_sentinel2_scl(scl))
sg = zeit.smooth(ndvi_16d, method="savgol", window=7)
zeit.plot(ndvi, fit=smooth)                                    # click a pixel: raw and smoothed
```

### `desawtooth` { .api }

<!-- sig: zeit.desawtooth -->
```python
zeit.desawtooth(values, stopat=0.9)
```

LandTrendr's spike removal for a single series: one-year spikes (up or down) are dampened toward their neighbours. [`landtrendr`](change-detection.md#landtrendr) already applies it internally through its `spike_threshold` parameter, so call this only to inspect or pre-process series yourself.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `values` | 1-D array | required | One pixel's series. |
| `stopat` | `float` | `0.9` | Each point gets a spike score (1 = a perfect one-year spike). Spikes are corrected strongest first while the strongest score exceeds `stopat`, so lower values correct more points. As in the original IDL code, the first correction is always applied. |

</div>

```python
zeit.desawtooth(np.array([8000, 8100, 3000, 8050, 7900, 8000.0]))
# array([8000., 8034.7, 8025.2, 8050., 7900., 8000.])
```
