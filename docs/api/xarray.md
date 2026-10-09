# Xarray Accessor

<p class="lead"><code>import zeit</code> registers a <code>.zeit</code> accessor on every <code>xarray.DataArray</code> (and, to save or show results, on every <code>xarray.Dataset</code>). Its methods run the C++ algorithms over Dask chunks, so they work on cubes larger than memory and on clusters. On a dask-backed cube the algorithms are lazy: call <code>.compute()</code>, or write the result with <code>save_raster</code> or <code>to_zarr_optimized</code>. An in-memory cube gives an in-memory result.</p>

!!! warning "Chunk in space, never in time"
    Each pixel needs its whole history, so keep `time` (and `band`, for CCDC) in a single chunk: `cube.chunk({"time": -1, "y": 512, "x": 512})`. See [Parallel & Cloud Processing](../tutorials/parallel-cloud-processing.md).

| Method | Input dims | Output | Parameters as in |
| :--- | :--- | :--- | :--- |
| `landtrendr` | `(time, y, x)` | `xr.Dataset` of vertices | [`zeit.landtrendr`](change-detection.md#landtrendr) |
| `ccdc` | `(time, band, y, x)` | `xr.Dataset` of segments | [`zeit.ccdc`](change-detection.md#ccdc) |
| `bfast_monitor` | `(time, y, x)` | `xr.Dataset` of metrics | [`zeit.bfast_monitor`](change-detection.md#bfast_monitor) |
| `bfast_lite` | `(time, y, x)` | `xr.Dataset` of metrics | [`zeit.bfast_lite`](change-detection.md#bfast_lite) |
| `bfast` | `(time, y, x)` | `xr.Dataset` of metrics | [`zeit.bfast`](change-detection.md#bfast) |
| `mann_kendall` | `(time, y, x)` | `xr.Dataset` of metrics | [`zeit.mann_kendall`](time-series.md#mann_kendall) |
| `phenology` | `(time, y, x)` | `xr.Dataset` of metrics, by year or season | [`zeit.phenology`](time-series.md#phenology) |
| `run_snic` | `(..., y, x)` | `xr.Dataset` of labels and means | [`run_snic`](time-series.md#run_snic) |
| `save` | any with `y`, `x` (also on a `Dataset`) | writes a raster, returns its path | [`zeit.save_raster`](data.md#save_raster) |
| `plot` | any with `y`, `x` (also on a `Dataset`) | the viewer, a window or a figure | [`zeit.plot`](plot.md) |
| `to_zarr_optimized` | any with `y`, `x` | writes a Zarr store | below |

Each algorithm method returns an `xr.Dataset` with one variable per output, selected by name: `result.slope`, `result["TRS5.sos"]`.

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

### `ccdc` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.ccdc -->
```python
DataArray.zeit.ccdc(**kwargs)
```

CCDC on this `(time, band, y, x)` cube of reflectance × 10,000: the same as [`zeit.ccdc(cube, **kwargs)`](change-detection.md#ccdc), with the same keyword arguments (`qa`, `bands`, `max_segments`, `conseq_anom`, …) and the same `xarray.Dataset` of segments. The dates come from the `time` coordinate. A dask-backed cube stays lazy and is computed block by block, with `time` and `band` in one chunk.

```python
cube = cube.chunk({"time": -1, "band": -1, "y": 512, "x": 512})
segments = cube.zeit.ccdc(qa="fmask")       # QA band by name; still lazy
zeit.save_raster(segments, "ccdc_out/")      # computed while written
```

### `bfast_monitor` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.bfast_monitor -->
```python
DataArray.zeit.bfast_monitor(monitor_start, **kwargs)
```

BFAST Monitor on this `(time, y, x)` cube: the same as [`zeit.bfast_monitor(cube, monitor_start, **kwargs)`](change-detection.md#bfast_monitor), with the same keyword arguments (`order`, `h`, `period`, `alpha`, `nodata`, …) and the same `xarray.Dataset` of metrics. `start_time` and `frequency` come from the `time` coordinate unless given. A dask-backed cube stays lazy.

```python
ndvi_16d = ndvi_16d.chunk({"time": -1, "y": 512, "x": 512})
bfm = ndvi_16d.zeit.bfast_monitor("2022-01-01")    # still lazy
zeit.save_raster(bfm, "results/bfm")               # computed while written
```

### `bfast_lite` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.bfast_lite -->
```python
DataArray.zeit.bfast_lite(**kwargs)
```

BFAST Lite on this `(time, y, x)` cube: the same as [`zeit.bfast_lite(cube, **kwargs)`](change-detection.md#bfast_lite).

### `bfast` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.bfast -->
```python
DataArray.zeit.bfast(**kwargs)
```

Classic BFAST on this `(time, y, x)` cube: the same as [`zeit.bfast(cube, **kwargs)`](change-detection.md#bfast).

## Time-series analysis

### `mann_kendall` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.mann_kendall -->
```python
DataArray.zeit.mann_kendall(**kwargs)
```

Mann-Kendall test and Theil-Sen slope on this `(time, y, x)` cube: the same as [`zeit.mann_kendall(cube, **kwargs)`](time-series.md#mann_kendall), with the same keyword arguments (`method`, `alpha`, `lag`, `period`, …) and the same `xarray.Dataset` of metrics.

```python
mk = annual_ndvi.zeit.mann_kendall(method="hamed_rao")
slope = mk.slope.where(mk.h == 1)        # significant slopes only
```

### `phenology` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.phenology -->
```python
DataArray.zeit.phenology(**kwargs)
```

Phenology on this `(time, y, x)` cube: the same as [`zeit.phenology(cube, **kwargs)`](time-series.md#phenology), with the same keyword arguments (`curve`, `annual`, `max_seasons`, `weights`, …) and the same `xarray.Dataset` of 21 metrics. The day numbering and the first year come from the `time` coordinate.

```python
pheno = ndvi_16d.zeit.phenology(curve="beck", weights=weights)
sos = pheno["TRS5.sos"]                  # (year, y, x), day of year
```


### `run_snic` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.run_snic -->
```python
DataArray.zeit.run_snic(
    spacing=10, compactness=0.5, seeds=None, grid="rectangular",
    padding=None, tile_size=None, random_state=None, n_jobs=-1,
)
```

Returns an `xr.Dataset` with `labels` `(y, x)`, the mean trajectory of each segment `means` `(segment, …)` keeping the cube's other coordinates, and `centroid_row`, `centroid_col`, `n_pixels`.

## Saving and showing results

The results of the algorithms are `Dataset`s; `.zeit.save` and `.zeit.plot` write or show them (or any map or cube) without leaving a chain of calls:

```python
zeit.landtrendr(ndvi).zeit.save("results/lt")               # one GeoTIFF per variable
zeit.extract_events(lt).yod.zeit.plot(basemap="satellite")  # one map
lt.zeit.plot()                                              # a selector of the variables
```

### `save` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.save -->
```python
DataArray.zeit.save(path, **kwargs)
```

<!-- sig: zeit.xarray_api.ZeitDatasetAccessor.save -->
```python
Dataset.zeit.save(path, **kwargs)
```

Writes this map, cube or result: the same as [`zeit.save_raster(data, path, **kwargs)`](data.md#save_raster), with its keyword arguments (`nodata`, `dtype`, `compress`, `driver`, …). A `Dataset` with a path without extension becomes a folder with one GeoTIFF per variable; with `.tif`, one multiband raster. Returns the `Path` written. A lazy result is computed while it is written.

### `plot` { .api .meth }

<!-- sig: zeit.xarray_api.ZeitAccessor.plot -->
```python
DataArray.zeit.plot(**kwargs)
```

<!-- sig: zeit.xarray_api.ZeitDatasetAccessor.plot -->
```python
Dataset.zeit.plot(**kwargs)
```

Shows this map, cube or result: the same as [`zeit.plot(data, **kwargs)`](plot.md), with its keyword arguments (`var`, `fit`, `basemap`, `vector`, `static`, `save`, …): the interactive viewer in a notebook, a window in a script, a figure with `static=True` or `save=`. A `Dataset` opens with a selector of its variables.

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
mk.slope.zeit.to_zarr_optimized("s3://my-bucket/ndvi_slope.zarr")
```
