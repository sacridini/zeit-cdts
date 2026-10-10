# Pre-processing

<p class="lead">Clean time series before analysing them: put Landsat and Sentinel-2 in one series, flag clouds and shadows that the QA band missed, smooth noise, remove one-off spikes, and unmix pixels into the fractions of their materials.</p>

## Landsat and Sentinel-2 in one series

A dense series that mixes Landsat and Sentinel-2 has a step wherever the sensor changes, which CCDC or BFAST read as a break. There are two ways to remove it:

- **HLS** (Harmonized Landsat Sentinel-2, NASA): both sensors processed alike from level 1 (the same atmospheric correction and cloud mask, view and sun angles normalized, a common 30 m grid) and Sentinel-2 adjusted to Landsat 8's bands. [`build_time_series`](data.md#build_time_series) reads it from Planetary Computer, both products in one call: `collection=["hls2-l30", "hls2-s30"]` with the bands named by role. This is the way to go wherever HLS covers the dates (from 2013 for Landsat 8, 2015 for Sentinel-2).
- **[`bandpass_adjust`](#bandpass_adjust)** on cubes you already have (Landsat Collection 2 and Sentinel-2 L2A): only HLS's bandpass step, so only part of the step goes away.

Measured over the Libya-4 desert site (2022–2023, Planetary Computer, pairs of Landsat 8/9 and Sentinel-2 observations at most a day apart, median of the area):

| | Pairs | Sentinel-2 / Landsat, by band (blue ... SWIR2) |
| :--- | :---: | :--- |
| L2A and Collection 2, as they come | 21 | 1.14, 1.07, 1.09, 1.05, 1.09, 1.13 |
| L2A after `bandpass_adjust` | 21 | blue and red about a third closer (differences 0.030 → 0.021 and 0.042 → 0.031), the rest about the same |
| HLS (S30 and L30) | 27 | 1.01, 0.99, 0.99, 1.00, 1.01, 0.99 |

The rest of the L2A step comes from the steps `bandpass_adjust` does not do: Sen2Cor and LaSRC correct the atmosphere differently, and the view geometry differs.

```python
grid = dict(source="planetary_computer", bbox=bbox, start_date="2019-01-01", end_date="2024-12-31",
            apply_cloud_mask=True, resolution=30, epsg=32722)
hls = zeit.build_time_series(collection=["hls2-l30", "hls2-s30"],
                             bands=["blue", "green", "red", "nir", "swir1", "swir2"], **grid)
result = zeit.ccdc(hls.sel(band=["blue", "green", "red", "nir", "swir1", "swir2"]))
```

### `bandpass_adjust` { .api }

<!-- sig: zeit.bandpass_adjust -->
```python
zeit.bandpass_adjust(
    data, sensor=None, etm=None, bands=None, scale="auto", nodata="auto",
    chunks=None,
)
```

Takes Sentinel-2 reflectance to the bands of Landsat 8's OLI with the bandpass adjustment of HLS (Claverie et al. 2018): one line per band and Sentinel-2 unit, `OLI = slope × MSI + intercept`, from the HLS v2.0 User Guide (Table 5; S2A and S2B are also the v1.4 values, S2C has its own). OLI is the reference, as in HLS: Landsat 8 and 9 stay as they are (HLS applies no adjustment between them). Landsat 7 (ETM+) and 4-5 (TM) are left alone unless `etm` is given; then the lines of Roy et al. (2016, Table 2, surface reflectance) take them to OLI. Those were fitted on pre-collection data, and Landsat Collection 2 surface reflectance is generally used across sensors without them (see Earth Engine's FAQ on cross-sensor harmonization). It is one step of HLS, not HLS (see above); HLS products are refused, since their Sentinel-2 dates are already adjusted.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path or list | required | Surface reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask. Or a list of cubes on the same grid (one per collection): each is adjusted and they are joined into one series sorted by time, with the bands they share matched by role and named after it (`blue` ... `swir2`), whatever each catalog calls them, and the coordinates they share. |
| `sensor` | `str` or list | `None` | Each date's sensor: by default the cube's `platform` coordinate (which [`build_time_series`](data.md#build_time_series) keeps from STAC) or attribute; or one name for every date (`"sentinel-2a"`, `"S2B"`, `"landsat-8"`, `"LANDSAT_7"`...), or one per date. With a list of cubes, one of these per cube. A Sentinel-2 date of unknown unit is adjusted as S2A, with a warning. |
| `etm` | `"rma"`, `"ols"` or `None` | `None` | Landsat 7 and 4-5: left as they are, or taken to OLI with the reduced major axis or least squares lines of Roy et al. (2016). TM is treated as ETM+, as in Earth Engine's harmonization tutorial (no TM lines are published). |
| `bands` | `dict` | `None` | The cube's band of each role (`coastal`, `blue`, `green`, `red`, `nir`, `swir1`, `swir2`), when its names are not the usual ones (`red`, `B04`, `SR_B4`...). With a list of cubes, one dict for all or one per cube. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by in the data: 10000 for integers or values above 2, else 1 (a lazy cube is checked on a small window). The intercepts are in reflectance. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). They stay missing. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |

</div>

**Returns** the cube with the same dimensions, coordinates, type and georeferencing (a list: the joined series), lazy if the input is. Bands without a line (the red edge, QA bands) are copied. `attrs["bandpass_adjusted"]` says what was done, and an adjusted cube is refused a second time.

**Sentinel-2's NIR.** HLS adjusts the narrow NIR, B8A (`nir08` on Earth Search, `B8A` on Planetary Computer), the band that matches OLI's band 5. B08, the broad NIR (`nir` on Earth Search, `B08`), has no line: it is left as it is, with a warning. Load B8A, or pass `bands={"nir": ...}` when a band is B8A under another name.

```python
s2 = zeit.build_time_series(collection="sentinel-2-l2a", bands=["B02", "B03", "B04", "B8A", "B11", "B12"], **grid)
landsat = zeit.build_time_series(collection="landsat-c2-l2",
                                 bands=["blue", "green", "red", "nir08", "swir16", "swir22"], **grid)
mixed = zeit.bandpass_adjust([s2, landsat])      # one series: blue, green, red, nir, swir1, swir2
```

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

## Spectral mixture analysis

### `unmix` { .api }

<!-- sig: zeit.unmix -->
```python
zeit.unmix(
    data, endmembers="souza2005", bands=None, sum_to_one=True,
    nonneg=True, scale="auto", ndfi=True, cloud_threshold=None,
    nodata="auto", chunks=None, n_jobs=-1,
)
```

The fraction of each endmember (pure material) in every pixel and date: a 30 m Landsat pixel of forest after selective logging is part canopy (green vegetation), part dead wood and litter (non-photosynthetic vegetation), part soil and part shade, and the fractions show the damage that the reflectance hides. The unmixing is fully constrained least squares (non-negative fractions that add up to one; Heinz & Chang 2001), in C++, every pixel and date in parallel. Also exported as `zeit.unmix`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | Reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask. |
| `endmembers` | `str`, `DataFrame` or `dict` | `"souza2005"` | `"souza2005"`: green vegetation (`gv`), `shade`, non-photosynthetic vegetation (`npv`), `soil` and `cloud` for Landsat's blue, green, red, NIR, SWIR1 and SWIR2 (Souza et al. 2005, with the cloud endmember of CODED, as CODED uses them). Or a table: rows endmembers, columns bands of the cube, reflectance 0–1 (or × 10000). |
| `bands` | list of `str` | `None` | The cube's band for each column of the endmembers, when the names differ (by default found by name: `nir`, `B08`, `SR_B5`...). |
| `sum_to_one`, `nonneg` | `bool` | `True`, `True` | Fractions adding up to one; fractions non-negative. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by: 10000 for integers or values above 2, else 1. |
| `ndfi` | `bool` | `True` | Also the NDFI (below), when the endmembers include `gv`, `npv`, `soil` and `shade`. |
| `cloud_threshold` | `float` | `None` | Mask observations whose `cloud` fraction is above this (CODED: 0.05). |
| `nodata`, `chunks`, `n_jobs` | | `"auto"`, `None`, `-1` | As elsewhere. |

</div>

**Returns** an `xarray.Dataset` with one variable per endmember (its fraction), `rmse` (of the rebuilt spectrum, in reflectance) and `ndfi`, on the cube's dimensions but `band`, lazy if the cube is.

The **NDFI** (Normalized Difference Fraction Index; Souza et al. 2005) is `(GVs − (NPV + soil)) / (GVs + NPV + soil)`, with `GVs = GV / (1 − shade)`: near 1 in closed forest, lower where the canopy was damaged, below 0 on bare soil and pasture. It is also an index of [`compute_indices`](data.md) (`compute_indices(cube, ["NDFI"])`) and of the Earth Engine downloads.

```python
fractions = zeit.unmix(landsat, cloud_threshold=0.05)
fractions.ndfi.zeit.plot()                       # page through the years
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
