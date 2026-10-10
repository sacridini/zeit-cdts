# Visualisation

<p class="lead">One function to look at any Zeit data or result: a map, a time-series cube, a LandTrendr, CCDC, BFAST or Mann-Kendall result, or one pixel's series. In a notebook it is an interactive viewer that pages through dense time series at the display's frame rate; elsewhere it opens the same viewer in its own window, or draws a matplotlib figure for a report.</p>

!!! info "Install"
    `zeit.plot` needs the `plot` extra: `pip install zeit-cdts[plot]` (matplotlib and anywidget). Install `pywebview` too if you want a native window outside the notebook instead of a browser tab. `import zeit` does not load any of them; they are imported the first time you plot.

## Plotting

### `plot` { .api }

<!-- sig: zeit.plot -->
```python
zeit.plot(
    data, var=None, time=None, band=None, rgb=None, kind=None,
    cmap=None, vmin=None, vmax=None, classes=None, nodata="auto",
    title=None, static=None, ax=None, figsize=None, colorbar=True,
    ncols=4, max_size=None, save=None, height=480, fps=8,
    compress=True, fit=None, pixel=None, basemap=None, vector=None,
    opacity=None, vector_color="#ffd400", block=None,
)
```

Shows `data` in the most useful form for where you are. All parameters but `data` are keyword-only, and every colour choice is inferred from the data (see [Colours](#colours)) but can be overridden. Tutorial: [Plotting and Exploring Results](../tutorials/plotting.md).

| Where you are | What `zeit.plot` returns |
| :--- | :--- |
| A notebook: JupyterLab, Notebook 7, VS Code notebooks, Colab | An interactive viewer widget (anywidget), shown as the cell's output |
| A script or a terminal, on a machine with a screen | The same viewer in its own window: a native window with `pywebview`, otherwise a browser tab. In a script the call blocks until the window is closed, like `plt.show()` |
| `static=True`, `save=` given, or no screen (no display, CI, tests, or the environment variable `ZEIT_PLOT=static`) | A matplotlib `Figure` |
| One pixel's series, or `pixel=` | Always a matplotlib `Figure` |

`data` can be:

| Input | Example | Shown as |
| :--- | :--- | :--- |
| A `(time, y, x)` or `(time, band, y, x)` cube, numpy- or dask-backed | `zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")` | One frame per date. With `band=`, one band; a cube with red, green and blue bands is shown in colour |
| A map `(y, x)`: `DataArray` or numpy | `loss.yod` | One map |
| A raster path, read lazily | `"LT_Stack_NDVI_Rondonia.tif"` | As the cube [`load_raster`](data.md#load_raster) reads |
| A result `Dataset` | `zeit.landtrendr(ndvi)`, `zeit.extract_events(lt)`, `zeit.ccdc(cube)`, `zeit.mann_kendall(ndvi)` | One variable (`var=`, default the first map); the viewer has a selector for the others |
| One pixel's series: a 1-D `DataArray` or a `pandas.Series` | `ndvi.sel(x=-63.2, y=-10.21, method="nearest")` | A line plot (always static) |

Every dimension other than `y` and `x` becomes a frame axis, so the same viewer pages through dates, LandTrendr's vertices or CCDC's segments and coefficients. Frames are labelled the way [`save_raster`](data.md#save_raster) names bands: `1985` for annual dates, `2020-01-15` otherwise, `3` for the third vertex, `1_blue_a0` for one CCDC coefficient (segment, band, coefficient).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset`, `ndarray`, dask array, path or `pd.Series` | required | What to show (see the table above). |
| `var` | `str` | `None` | Variable of a `Dataset` to show. Default: the first one with `y` and `x` dims. |
| `time` | `int`, `str`, date or list | `None` | Static plots: the frames to draw. An index (`0`, `-1`), a label (`"2015"`), a year (`2015`) or date (`"2015-07-01"`, nearest), a list of these, or `"all"`. Default: up to 12 frames spread over the series. The viewer always shows every frame. |
| `band` | `str` or `int` | `None` | Band to show from a `(time, band, y, x)` cube (a label of its `band` coordinate). A cube of embeddings is shown through its principal components, or in the viewer's [embedding views](#embeddings); `band=` shows one dimension instead. |
| `rgb` | `bool` | `None` | Show the red, green and blue bands as a colour image. Default: when the `band` axis has them (`red`/`green`/`blue`, `r`/`g`/`b`, `B04`/`B03`/`B02`, `SR_B4`/`SR_B3`/`SR_B2`). `False` shows one band at a time. |
| `kind` | `str` | `None` | How to colour: `"continuous"`, `"diverging"`, `"categorical"`, `"years"` or `"rgb"`. Default: inferred (see [Colours](#colours)). |
| `cmap` | `str` | `None` | Any matplotlib colormap name. Default: depends on `kind`. |
| `vmin` | `float` | `None` | Lower colour limit. Default: the 2nd percentile of a sample. |
| `vmax` | `float` | `None` | Upper colour limit. Default: the 98th percentile of a sample. |
| `classes` | `list` or `dict` | `None` | Categories: a list of values, or `{value: label}` / `{value: (label, colour)}`. Values not listed are not drawn. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value drawn transparent, besides NaN. `"auto"`: the raster's NoData; for integer maps without one that are not categories, `0` (Earth Engine exports, LandTrendr's `yod`). A number: that value. `None`: only NaN. |
| `title` | `str` | `None` | Title of the figure or viewer. Default: the variable's name (in a static plot, only for a single map). |
| `static` | `bool` | `None` | `True`: a matplotlib figure. `False`: the interactive viewer. Default: the viewer, unless `save` is given or there is no screen (see above). |
| `ax` | matplotlib `Axes` | `None` | Static plots: draw one map into this axes (pick one frame with `time=`). |
| `figsize` | `tuple` | `None` | Static plots: figure size in inches. Default: from the number of maps and their aspect. |
| `colorbar` | `bool` | `True` | Static plots: draw the colour bar or the class legend. |
| `ncols` | `int` | `4` | Static plots: maps per row in a grid of frames. |
| `max_size` | `int` | `None` | Longest side, in cells, that maps are read at: 1024 for static plots, 800 for the viewer's preview (finer cells are fetched when you zoom in). The cube is never read at full resolution as a whole. |
| `save` | `str` | `None` | Write the figure to this file (PNG, PDF, SVG…, at 150 dpi) and return it. Makes the plot static unless `static=False`. |
| `height` | `int` | `480` | Viewer in a notebook: height of the map in pixels. The window version fills its window. |
| `fps` | `int` | `8` | Viewer: playback speed, in frames per second. Can be changed in the viewer. |
| `compress` | `bool` | `True` | Viewer: send frames deflate-compressed. Smaller transfers for remote notebooks, at a little work on each side. |
| `fit` | `Dataset` | `None` | A result on the same grid, from `landtrendr`, `ccdc`, `extract_events`, `bfast_monitor`, `bfast_lite`, `bfast` or `mann_kendall`. Clicking a pixel in the viewer (or `pixel=`) shows its series with what the algorithm made of it. Matched to the data by coordinates. |
| `pixel` | `(x, y)` | `None` | A point in the data's coordinates: plot that pixel's series (with `fit`'s overlays) as a figure, instead of maps. |
| `basemap` | `str` or xyzservices provider | `None` | A web map under the data: `"satellite"` (Esri World Imagery), `"osm"`, `"light"` / `"dark"` (Esri grey canvases), `"topo"` (OpenTopoMap), an xyzservices provider or its name (`"Esri.WorldImagery"`), or a `{z}/{x}/{y}` URL template. The data needs a CRS, any CRS. |
| `vector` | path, GeoDataFrame, GeoSeries or shapely geometry | `None` | Outlines drawn over the maps: a vector file, a GeoDataFrame or GeoSeries, a shapely geometry or a list of them. Reprojected to the data's CRS. |
| `opacity` | `float` | `None` | Opacity of the data, 0–1. Default: `1`, or `0.8` over a basemap. The viewer has a slider. |
| `vector_color` | `str` | `"#ffd400"` | Colour of the outlines. |
| `block` | `bool` | `None` | Window outside a notebook: wait until it is closed. Default: yes when running a script, no in an interactive interpreter (`python -i`, IPython), where the window stays open until `close()` or the interpreter exits. |

</div>

**Returns**

- a `matplotlib.figure.Figure` for a static plot (`save=` also writes it);
- the viewer widget in a notebook. It shows itself as the cell's output; assign it (`v = zeit.plot(...)`) and put `v` on a line of its own to show it later;
- a `Window` outside a notebook, with `url` (the page's address on 127.0.0.1, including its access token), `close()` and `wait()`. It is also a context manager.

#### The viewer

The viewer sends each frame once to the browser as one byte per cell (a colour index), and the browser's GPU paints it through a 256-colour lookup table (WebGL2). Frames are preloaded from the current one outwards, so once they have arrived, paging through time does not involve Python at all. It runs at the display's refresh rate, and keeps working while another cell runs.

| To | Do |
| :--- | :--- |
| Pan | Drag the map. |
| Zoom | Mouse wheel, around the cursor. When you stop (and playback is paused), the visible window is fetched again at a finer resolution, down to full resolution. |
| Reset the view | Double-click. |
| Play or pause | Space, or the ▶ button. |
| Step through frames | ← and →, Home and End for the first and last frame, or drag the slider. |
| Change the speed | The fps selector (1 to 60 frames per second). |
| Read a value | Hover: the frame, the coordinates and the value under the cursor appear below the map. The value is read back from the colour index, so it is exact for categories and within 1/254 of the colour range otherwise (`≈`); values beyond `vmin`/`vmax` read as the limit. |
| See a pixel's series | Click a pixel: its full series appears below the map, with `fit=`'s overlays. Click the chart to jump to that date. |
| Show another variable | The selector next to the title (`Dataset`s). |
| See the basemap through the data | The opacity slider (with `basemap=`). |

The keys work once the viewer has the focus: click it first.

What `fit=` draws over a pixel's series:

| `fit` from | Overlay |
| :--- | :--- |
| [`landtrendr`](change-detection.md#landtrendr) | The fitted segments, with a marker at each vertex |
| [`ccdc`](change-detection.md#ccdc) | Each segment's harmonic model (for the band shown, or `band=`) and its break |
| [`extract_events`](change-detection.md#extract_events) | The event as a shaded span, from `yod` to `yod + duration`; for events of CCDC and BFAST, a line on the `date` of the break |
| [`bfast_monitor`](change-detection.md#bfast_monitor) | The break, and where monitoring starts |
| [`bfast_lite`](change-detection.md#bfast_lite), [`bfast`](change-detection.md#bfast) | The breaks (trend and season breaks for `bfast`) |
| [`mann_kendall`](time-series.md#mann_kendall) | The Theil-Sen line, labelled increasing, decreasing or no trend |

The result is matched to the data by coordinates, so it may cover only part of the cube (a `fit` computed on a crop works on the whole map). A result without coordinates must have the data's shape.

#### Embeddings

A cube of embeddings ([`load_embeddings`](embeddings.md#load_embeddings)) has 64 or 128 dimensions per pixel, with no physical unit: what matters is how vectors compare. For it the viewer sends the vectors themselves to the browser, each year once, as one byte per dimension (a scale per dimension, from the 0.1 and 99.9 % percentiles of a sample), and the GPU computes each view from them. Changing view, reference or year does not involve Python. A panel beside the map picks the view:

| View | What it shows |
| :--- | :--- |
| Components (RGB) | Three of the first six principal components as red, green and blue (PC1–3 by default), stretched between their 2 and 98 % percentiles, with the share of the variance each explains. Fitted on *every year*, so a colour means the same embedding in each year and playing the years shows change; or on the *visible area*, refitted when you stop panning or zooming, which spreads the colours over the differences on screen (the states of a forest rather than forest versus city). |
| Similarity | The cosine similarity of every pixel to a reference vector: the pixel under the cursor, live as the mouse moves, or a pinned pixel (click a pixel; click it again or press Esc to unpin). *Keep the reference's year* fixes the year the reference is taken from while you page through the others ("did this pasture come to look like the forest?"). Colours from the 2nd percentile of the similarity to 1. |
| Change | 1 − the cosine similarity of each pixel's vectors in the year shown and the previous year (or a year you pick). Colours from 0 to the 98th percentile. |

Hovering reads the components, the similarity or the change under the cursor. The panel draws the *latent profile*, the values of the cursor's (and the pin's) vector along its dimensions, and the chart below the map follows the cursor: the similarity of its pixel (and the pin's) to the reference in every year.

To keep the browser's memory bounded, a year's vectors are at most 48 MB: a 500 × 500 cube of 128 dimensions (TESSERA) travels at full resolution, larger ones at coarser cells, and zooming in fetches the visible window at finer cells, as for any cube. Measured on a 500 × 500 × 128 cube (2026-10-10, Edge, a local window): the first year on screen in 1.2 s, four years in 3.7 s, and the similarity map redrawn about 30 ms after the mouse moves. A static plot (`static=True`) shows the components fitted on every year.

Outside a notebook, the viewer is served by a small HTTP server on 127.0.0.1 under a random token, so other users of the machine and other web pages cannot read the data. The server stops when the window is closed, when you call `close()`, or when Python exits.

#### Colours

Colours are chosen from a sample of the data (a few frames, decimated), never from the whole cube:

| The data | `kind` | Palette | Legend |
| :--- | :--- | :--- | :--- |
| A `band` axis with red, green and blue | `"rgb"` | 2–98 % stretch shared by the three bands | none |
| `bool`, `classes=` given, or integers with at most 32 distinct values (not years) | `"categorical"` | One colour per value (tab10, then tab20) | Class legend |
| Years: a name such as `yod`, `year` or `*_year`, dates (`datetime64`, e.g. CCDC's `t_break`), or integers all between 1800 and 2200 | `"years"` | `plasma` | Whole-year ticks |
| Values on both sides of 0 in comparable amounts (the smaller side at least 20 % of the larger) | `"diverging"` | `RdBu`, symmetric around 0 | Colour bar |
| Anything else | `"continuous"` | `viridis`, or `RdYlGn` (green high) when the name contains a vegetation index (`NDVI`, `EVI`, `SAVI`, `NBR`, `NDMI`, `NDWI`, …) | Colour bar, limits at the 2nd and 98th percentiles |

NaN and NoData (see `nodata`) are transparent, so a basemap or the page shows through. `load_raster` names a cube after its file, so `LT_Stack_NDVI_Rondonia.tif` gets the vegetation palette. A float map with a few values, such as Mann-Kendall's `trend` (−1, 0, 1), counts as continuous; give it `classes=` to get a legend.

#### Examples

```python
import zeit

ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")   # (time, y, x), 1985-2024

# In a notebook: the interactive viewer. Drag, wheel, space to play.
zeit.plot(ndvi)

# A static grid of years, written to a file
zeit.plot(ndvi, time=["1985", "2000", "2015", "2024"], save="ndvi_years.png")

# A result Dataset: the viewer has a selector for its variables
lt = zeit.landtrendr(ndvi)
loss = zeit.extract_events(lt, min_magnitude=2000)
zeit.plot(loss)                                   # yod first, with year ticks
zeit.plot(loss, var="magnitude", static=True)     # one variable as a figure

# Click a pixel to see its series and LandTrendr's fit
zeit.plot(ndvi, fit=lt)

# The same as a figure, for one pixel (x, y in the data's CRS)
zeit.plot(ndvi, fit=lt, pixel=(-63.2001, -10.2116), static=True)

# Over satellite imagery, with an area of interest
zeit.plot(loss.yod, basemap="satellite", vector="aoi.gpkg", opacity=0.7)

# Categories with your own labels and colours
mk = zeit.mann_kendall(ndvi)
zeit.plot(mk, var="trend", classes={-1: ("decreasing", "#d62728"), 0: ("no trend", "#dddddd"),
                                    1: ("increasing", "#2ca02c")})

# Into your own matplotlib layout
import matplotlib.pyplot as plt
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
zeit.plot(ndvi, time="1985", ax=a, static=True)
zeit.plot(ndvi, time="2024", ax=b, static=True)
```

In a script, the viewer opens in its own window and the script waits until you close it:

```python
# look.py: run with `python look.py`
import zeit

ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
lt = zeit.landtrendr(ndvi)
zeit.plot(ndvi, fit=lt)        # opens a window; returns when it is closed
```

In an interactive interpreter the call returns at once and the window stays open:

```python
>>> w = zeit.plot(ndvi)
>>> w.url
'http://127.0.0.1:52113/Qm3n.../'
>>> w.close()
```

Outside a notebook, set `ZEIT_PLOT=static` in the environment to get figures instead of windows, for example for batch jobs that render reports.
