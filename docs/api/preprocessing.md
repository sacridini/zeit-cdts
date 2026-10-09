# Pre-processing

<p class="lead">Clean time series before analysing them: flag clouds and shadows that the QA band missed, smooth noise, and remove one-off spikes.</p>

## Cloud masking

### `tmask` { .api }

<!-- sig: zeit.tmask -->
```python
zeit.tmask(
    data, green="green", swir="swir1", scale=10000.0, nodata="auto",
    chunks=None,
)
```

Flags clouds and cloud shadows the QA band missed, from each pixel's time series (Tmask, Zhu & Woodcock 2014): a robust harmonic model (MATLAB's `robustfit` with Tukey's bisquare, as CCDC's own Tmask, in C++, every pixel in parallel) is fitted to the green and SWIR bands of every pixel, and observations more than 0.04 (reflectance) above it in green (clouds) or below it in SWIR (shadows) are flagged. One function for a `(time, band, y, x)` cube in memory or dask, or anything [`load_raster`](data.md#load_raster) reads; the dates come from its `time` coordinate. It replaces `apply_tmask_stack` and `run_tmask_pixel` (the per-pixel engine, still in `zeit._tmask`).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | A `(time, band, y, x)` cube with green and SWIR bands. |
| `green` | `str` or `int` | `"green"` | The green band, by name or position. |
| `swir` | `str` or `int` | `"swir1"` | The SWIR band (SWIR1, ~1.6 µm), by name or position. |
| `scale` | `float` | `10000.0` | Reflectance scale of the data (`1.0` for reflectance in 0-1); the thresholds are in reflectance. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation: the raster's NoData (`0` for integer data without one), a number, or `None`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy, computed block by block. |

</div>

**Returns** `clear (time, y, x)`, `bool`: `True` for a clear observation, `False` for a cloud, a shadow or no observation (NoData, NaN, or not above 0 in either band). Pixels with 5 or fewer valid dates are not screened. Georeferenced as the input.

```python
cube = zeit.load_raster("landsat/", pattern=r"_(?P<date>\d{8})_(?P<band>\w+)\.tif$")   # (time, band, y, x)
clear = zeit.tmask(cube, green="green", swir="swir1")
cube = cube.where(clear)                         # clouds and shadows become NaN
qa = xr.where(clear, 0, 4)                       # or as CCDC's Fmask codes (0 clear, 4 cloud)
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
