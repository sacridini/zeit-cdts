# Upgrading to the one-function API

Versions 0.27 to 0.32 reorganised Zeit around one idea: **load, run, export**. Data is read once into a georeferenced cube whose dates and georeferencing travel with it, every algorithm is a single function that understands what it is given, and results are `xarray.Dataset`s that `save_raster` writes back with their georeferencing.

```python
import zeit

ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")   # (time, y, x), years from the band names
lt = zeit.landtrendr(ndvi)                               # looks for NDVI drops
loss = zeit.extract_events(lt)                           # greatest loss per pixel
zeit.save_raster(loss, "lt_rondonia")                    # one GeoTIFF per metric
```

This page lists what changed, so that code written for 0.26 or earlier can be updated.

## Reading and writing

| Before | Now |
| :--- | :--- |
| `stack, profile = zeit.io.load_raster(path)` | `cube = zeit.load_raster(path)`: an `xarray.DataArray` `(time, y, x)` or `(time, band, y, x)` with a `time` coordinate and its CRS/transform in `.rio`. The numpy array is `cube.values`. |
| `load_raster(path, raster_check="landtrendr")` | `load_raster(path, validate="landtrendr")` |
| `years = np.arange(1985, 1985 + n)` | Read from the band names (`yr1985`, `2020-01-15`, …), or `load_raster(path, start_year=1985)` / `dates=[...]`. |
| `save_raster(arr, path, crs=profile["crs"], transform=profile["transform"])` | `save_raster(arr, path, like=cube)` (or a DataArray, which brings its own georeferencing). |
| `save_raster(..., reference_cube=cube)` | `save_raster(..., like=cube)` (`reference_cube=` still works, with a `DeprecationWarning`). |
| `save_raster` wrote `<name>_dates.csv` | The dates go into the band names and a `ZEIT_TIME` tag that `load_raster` reads back. Old `_dates.csv` files are still read. |
| `save_raster` returned `None` | It returns the `Path` written; a `Dataset` or `dict` becomes a folder with one GeoTIFF per variable. |

## Algorithms

Each algorithm is now one function. It takes a pixel's series, a numpy stack with `years`/`dates`, a cube in memory or dask, or a raster path, and returns a georeferenced `Dataset`. The xarray accessor methods are thin aliases (`cube.zeit.landtrendr()` is `zeit.landtrendr(cube)`).

| Before | Now |
| :--- | :--- |
| `zeit.run_landtrendr(years, values)` | `zeit.landtrendr(values, years=years)` |
| `zeit.run_landtrendr_array(years, stack, modifier=-1.0)` | `zeit.landtrendr(cube)` (`direction="loss"` is the default; `"gain"` replaces `modifier=+1`) |
| `zeit.run_landtrendr_image(path, out_dir, start_year=...)` | `zeit.save_raster(zeit.extract_events(zeit.landtrendr(path, chunks="auto")), out_dir)` |
| `cube.zeit.run_landtrendr(years)` | `cube.zeit.landtrendr()` |
| `no_data_value=0.0` | `nodata="auto"`: the raster's NoData, or 0 for integer stacks without one |
| `extract_events(vertices, rmse_map=rmse)["yod"]` | `zeit.extract_events(lt).yod` (a `Dataset`; `rmse` comes with `lt`) |
| `zeit.ccdc.run_ccdc(dates, values, qa)` | `zeit.ccdc(frame, qa="fmask")` for one pixel (a `DataFrame` indexed by date) |
| `zeit.run_ccdc_array(dates, stack, qa)` with `(band, time, y, x)` | `zeit.ccdc(cube, qa=...)` with `(time, band, y, x)`; dates from `time` |
| `zeit.run_ccdc_image(path, out_dir, dates, num_bands)` | `zeit.ccdc(path, dates=..., bands=[...])`, then `save_raster` |
| `predict_synthetic_image(stack, ordinal_day, num_bands)` | `zeit.predict_synthetic_image(segments, "2020-07-01")` (the numpy call still works) |
| `zeit.run_bfast_monitor_image(...)` / `cube.zeit.run_bfast_monitor(start_time, monitor_start_time, frequency)` | `zeit.bfast_monitor(cube, "2019-01-01")`: `start_time` and `frequency` come from the dates |
| `zeit.run_bfast_lite_image(...)` / `cube.zeit.run_bfast_lite(...)` | `zeit.bfast_lite(cube)` (`max_breaks_output` is now `max_breaks`) |
| `zeit.run_bfast_image(...)` / `cube.zeit.run_bfast(...)` | `zeit.bfast(cube)` (the `time` metric is now `break_time`) |
| `zeit.run_mann_kendall_image(...)` / `cube.zeit.run_mann_kendall()` | `zeit.mann_kendall(cube)` |
| `cube.zeit.run_phenology(dates=dates_doy, curve_type=0, base_year=...)` | `zeit.phenology(cube, curve="beck")`: day numbering and base year come from the dates |

Results that were a `DataArray` with a `metric` (or `vertex_info`, `parameter`) dimension are now a `Dataset` with one variable per metric: `result.sel(metric="trend")` becomes `result.trend`.

## Modules

The engines moved to private modules, so that each algorithm's name is its function: `zeit/landtrendr.py`, `ccdc.py`, `bfast.py` and `phenology.py` are now `_landtrendr.py`, `_ccdc.py`, `_bfast.py` and `_phenology.py`. Imports such as `from zeit.landtrendr import run_landtrendr` no longer work; use the functions above.

## Command line

The CLI runs on the same functions. `zeit landtrendr` reads the years from the band names (`--start-year` is only needed without them), and `--no-data-value` defaults to the raster's NoData. `zeit ccdc` reads the dates from `date_band` band names or from `--dates-file`. The BFAST and Mann-Kendall commands infer `--start-time` and `--frequency` from the band dates. See [Command line](../cli.md).

## Faster import

`import zeit` no longer loads PyTorch, Transformers or the STAC libraries: `zeit.ai`, the cube builders (`build_time_series`, …) and the CCDC classifier are loaded on first use.
