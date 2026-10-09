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

### `apply_whittaker_filter` { .api }

<!-- sig: zeit.smooth.apply_whittaker_filter -->
```python
zeit.smooth.apply_whittaker_filter(
    cube, lmbd=10.0, axis=0, weights=None,
)
```

Whittaker smoother along the time axis. It balances closeness to the data against roughness, controlled by `lmbd`, and accepts per-observation weights so that cloudy observations (weight 0) are ignored and gaps are filled. Usually the better choice for vegetation indices. Import from `zeit.smooth`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `np.ndarray` | required | Array with a time axis, e.g. `(time, rows, cols)`. |
| `lmbd` | `float` | `10.0` | Smoothness. Larger values give smoother curves. |
| `axis` | `int` | `0` | Time axis. |
| `weights` | `np.ndarray` | `None` | Same shape as `cube`: `0` ignores an observation, `1` trusts it. Values in between down-weight it (see `zeit.qc`). |

</div>

```python
from zeit.smooth import apply_whittaker_filter

smooth = apply_whittaker_filter(ndvi, lmbd=10, weights=clear.astype(float))
```

### `apply_savgol_filter` { .api }

<!-- sig: zeit.smooth.apply_savgol_filter -->
```python
zeit.smooth.apply_savgol_filter(
    cube, window_length=5, polyorder=2, axis=0,
)
```

Savitzky-Golay filter along the time axis (a moving local polynomial fit). Simple and fast, but every observation counts equally, so cloud drops pull the curve down. Zeros in the input are kept as zeros (no-data). Also exported as `zeit.apply_savgol_filter`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `np.ndarray` | required | `(time, rows, cols)` or `(time, bands, rows, cols)`. |
| `window_length` | `int` | `5` | Window length in observations. Must be odd and at most the series length. |
| `polyorder` | `int` | `2` | Order of the local polynomial. |
| `axis` | `int` | `0` | Time axis. |

</div>

```python
smooth = zeit.apply_savgol_filter(ndvi, window_length=7, polyorder=2)
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
