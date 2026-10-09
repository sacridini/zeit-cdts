# Xarray Accessor

<p class="lead"><code>import zeit</code> registers a <code>.zeit</code> accessor on every <code>xarray.DataArray</code>. Its methods run the C++ algorithms over Dask chunks, so they work on cubes larger than memory and on clusters. All of them are lazy: call <code>.compute()</code>, or write the result with <code>to_zarr_optimized</code>.</p>

!!! warning "Chunk in space, never in time"
    Each pixel needs its whole history, so keep `time` (and `band`, for CCDC) in a single chunk: `cube.chunk({"time": -1, "y": 512, "x": 512})`. See [Parallel & Cloud Processing](../tutorials/parallel-cloud-processing.md).

| Method | Input dims | Output | Parameters as in |
| :--- | :--- | :--- | :--- |
| `landtrendr` | `(time, y, x)` | `xr.Dataset` of vertices | [`zeit.landtrendr`](change-detection.md#landtrendr) |
| `run_ccdc` | `(band, time, y, x)` | `(segment, parameter, y, x)` | [`run_ccdc_array`](change-detection.md#run_ccdc_array) |
| `run_bfast_monitor` | `(time, y, x)` | `(metric, y, x)` | [`run_bfast_monitor_dask`](change-detection.md#run_bfast_monitor_dask) |
| `run_bfast_lite` | `(time, y, x)` | `(metric, y, x)` | [`run_bfast_lite_dask`](change-detection.md#run_bfast_lite_dask) |
| `run_bfast` | `(time, y, x)` | `(metric, y, x)` | [`run_bfast_dask`](change-detection.md#run_bfast_dask) |
| `run_mann_kendall` | `(time, y, x)` | `(metric, y, x)` | [`run_mann_kendall_dask`](time-series.md#run_mann_kendall_dask) |
| `run_phenology` | `(time, y, x)` | `(metric, year or season, y, x)` | [`run_phenology_dask`](time-series.md#run_phenology_dask) |
| `run_snic` | `(..., y, x)` | `xr.Dataset` of labels and means | [`run_snic`](time-series.md#run_snic) |
| `to_zarr_optimized` | any with `y`, `x` | writes a Zarr store | below |

Outputs with a `metric` dimension are labelled, so you can select by name: `result.sel(metric="slope")`.

## Change detection

### `landtrendr` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.landtrendr -->
```python
DataArray.zeit.landtrendr(**kwargs)
```

LandTrendr on this `(time, y, x)` cube: the same as [`zeit.landtrendr(cube, **kwargs)`](change-detection.md#landtrendr), with the same keyword arguments (`direction`, `max_segments`, `nodata`, `fitted`, …) and the same `xarray.Dataset` of vertices. The years come from the `time` coordinate. A dask-backed cube stays lazy and is computed block by block.

```python
ndvi = ndvi.chunk({"time": -1, "y": 512, "x": 512})
lt = ndvi.zeit.landtrendr(max_segments=6)                  # direction="loss": NDVI drops
loss = zeit.extract_events(lt, min_magnitude=2000)         # still lazy
zeit.save_raster(loss, "results/")                         # computed while written
```

### `run_ccdc` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_ccdc -->
```python
DataArray.zeit.run_ccdc(
    dates, qa_stack=None, max_segments=6, return_coefs=True,
    conseq_anom=6, n_jobs=-1, **ccdc_kwargs,
)
```

CCDC for every pixel of a `(band, time, y, x)` cube of reflectance × 10,000. `qa_stack` is a `(time, y, x)` array of Fmask codes (all clear if omitted). Extra keyword arguments go to `run_ccdc`.

### `run_bfast_monitor` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_bfast_monitor -->
```python
DataArray.zeit.run_bfast_monitor(
    start_time, monitor_start_time, frequency, order=3, h=0.25,
    period=10, alpha=0.05, min_valid=10, n_jobs=-1,
)
```

### `run_bfast_lite` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_bfast_lite -->
```python
DataArray.zeit.run_bfast_lite(
    start_time, frequency, order=3, h=0.15, max_breaks_output=5,
    min_valid=20, n_jobs=-1,
)
```

### `run_bfast` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_bfast -->
```python
DataArray.zeit.run_bfast(
    start_time, frequency, order=3, h=0.15, max_breaks_trend=5,
    max_breaks_season=5, max_iter=10, level=0.05, min_valid=20,
    n_jobs=-1,
)
```

## Time-series analysis

### `run_mann_kendall` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_mann_kendall -->
```python
DataArray.zeit.run_mann_kendall(
    method="hamed_rao", alpha=0.05, lag=None, period=1, min_valid=4,
    n_jobs=-1,
)
```

```python
trend = annual_ndvi.zeit.run_mann_kendall(method="hamed_rao").compute()
slope = trend.sel(metric="slope")
```

### `run_phenology` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_phenology -->
```python
DataArray.zeit.run_phenology(
    dates, curve_type, extraction_method=0, max_seasons=2,
    whittaker_lambda=10.0, apply_whittaker=True, apply_hants=False,
    hants_frequencies=3, hants_threshold=0.1, min_season_length=0,
    min_amplitude=0.0, min_pixel_amplitude=0.1, return_annual=True,
    base_year=2001, n_jobs=-1, weights=None, season_retry=True,
)
```

`weights` may be a DataArray or array aligned with the cube. The output is labelled with the 21 metric names and, with `return_annual=True`, a `year` coordinate starting at `base_year`.

### `run_snic` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_snic -->
```python
DataArray.zeit.run_snic(
    spacing=10, compactness=0.5, seeds=None, grid="rectangular",
    padding=None, tile_size=None, random_state=None, n_jobs=-1,
)
```

Returns an `xr.Dataset` with `labels` `(y, x)`, the mean trajectory of each segment `means` `(segment, …)` keeping the cube's other coordinates, and `centroid_row`, `centroid_col`, `n_pixels`.

## Storage

### `to_zarr_optimized` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.to_zarr_optimized -->
```python
DataArray.zeit.to_zarr_optimized(
    store_path, chunk_size={'y': 512, 'x': 512},
)
```

Rechunks the array spatially and writes it to a Zarr store with consolidated metadata, locally or to object storage (`s3://`, `gs://`). With a Dask-backed array, this is the step that triggers the computation, and every worker writes its own chunks.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `store_path` | `str` | required | Local path or fsspec URL. |
| `chunk_size` | `dict` | `{"y": 512, "x": 512}` | Chunk sizes of the stored array. |

</div>

```python
trend.zeit.to_zarr_optimized("s3://my-bucket/ndvi_trend.zarr")
```
