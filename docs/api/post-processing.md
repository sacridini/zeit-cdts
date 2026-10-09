# Post-processing

<p class="lead">Clean up maps, classify CCDC results, derive masks, and review results against reference points.</p>

## Spatial filters

The filters take a map (a `DataArray`, a numpy array or a raster path) and give back the same kind, georeferencing, NoData and attributes kept.

### `apply_mmu_filter` { .api }

<!-- sig: zeit.spatial.apply_mmu_filter -->
```python
zeit.spatial.apply_mmu_filter(data, mmu_pixels=11, nodata="auto")
```

Minimum mapping unit filter: connected (4-neighbour) patches of pixels holding a value, whatever the value, smaller than `mmu_pixels` become NoData. Removes the salt-and-pepper noise of per-pixel change maps (a LandTrendr year of loss, a magnitude). Also exported as `zeit.apply_mmu_filter`; CLI: `zeit mmu-filter`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `ndarray` or path | required | A single map, e.g. a year-of-loss map. |
| `mmu_pixels` | `int` | `11` | Smallest patch to keep, in pixels (11 Landsat pixels ≈ 1 ha). |
| `nodata` | `"auto"` or `float` | `"auto"` | The value of "no feature": the map's NoData, else `0` (NaN for floats with NaN). |

</div>

```python
loss = zeit.extract_events(lt, min_magnitude=1500)
yod = zeit.apply_mmu_filter(loss.yod, mmu_pixels=11)
zeit.save_raster(yod, "results/loss_year_mmu.tif")
```

### `apply_majority_filter` { .api }

<!-- sig: zeit.spatial.apply_majority_filter -->
```python
zeit.spatial.apply_majority_filter(image, size=3)
```

Replaces each pixel of a class map with the most common class in its neighbourhood. Also exported as `zeit.apply_majority_filter`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `DataArray`, `ndarray` or path | required | 2-D class map. |
| `size` | `int` | `3` | Window size (3 = 3 × 3). |

</div>

### `apply_bayesian_filter` { .api }

<!-- sig: zeit.spatial.apply_bayesian_filter -->
```python
zeit.spatial.apply_bayesian_filter(probs, window_size=3)
```

Smooths per-class probabilities spatially, then takes the most likely class. Unlike the majority filter, confident pixels resist being overruled by their neighbours. Import from `zeit.spatial`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `probs` | `DataArray` or `ndarray` | required | `(class, y, x)` probabilities or scores, e.g. `zeit.classify(..., probability=True).probability`. |
| `window_size` | `int` | `3` | Averaging window. |

</div>

**Returns** the winning class per pixel `(y, x)`: its index for a numpy array, the class itself for a `DataArray` whose first dim has coordinates.

## Classification

### `train_classifier` { .api }

<!-- sig: zeit.train_classifier -->
```python
zeit.train_classifier(
    data, samples, label="class", model=None, date=None,
)
```

Trains a classifier on the features of `data` at sample points. The features of a pixel are what it holds off `y`/`x`: the bands of a map, every date (and band) of a cube, the maps of a Dataset (phenology or LandTrendr metrics…), or, for a [`zeit.ccdc`](change-detection.md#ccdc) result with `date`, the model of the segment covering that date: each band's coefficients (the intercept moved to the date, `a0 + c1 t`, as CCDC classifications use it) and RMSE. Loaded on first use (scikit-learn).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset` or path | required | What to classify (see above). |
| `samples` | `GeoDataFrame` or path | required | Points with their class. Points in another CRS are reprojected; points outside the data or on a pixel with a missing feature are left out. |
| `label` | `str` | `"class"` | The column holding the class (names or numbers). |
| `model` | classifier | `None` | Any scikit-learn classifier; default a random forest of 100 trees (`random_state=42`). |
| `date` | date | `None` | CCDC segments: the date whose models are the features. |

</div>

**Returns** the fitted model, with the feature names in `zeit_features_`.

### `classify` { .api }

<!-- sig: zeit.classify -->
```python
zeit.classify(data, model, date=None, probability=False)
```

Classifies every pixel with a trained model. The features are taken as in `train_classifier`, and reordered by name to the model's `zeit_features_`. A lazy cube stays lazy. Loaded on first use (scikit-learn).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset` or path | required | The features (see `train_classifier`). |
| `model` | classifier | required | A fitted classifier, e.g. from `train_classifier`. |
| `date` | date | `None` | CCDC segments: the date whose models are the features. |
| `probability` | `bool` | `False` | Also return each class's probability (`predict_proba`). |

</div>

**Returns** an `xarray.Dataset`: `label (y, x)`, `1` for the first class of `model.classes_`, `2` for the second… (`0`: a missing feature), with the names in its `flag_meanings` (the legend of `zeit.plot`); `probability (class, y, x)` with `probability=True`; the `class` coordinate holds the classes.

```python
segments = zeit.ccdc(cube, qa="fmask")
rf = zeit.train_classifier(segments, "samples.gpkg", label="class", date="2020-07-01")
land_cover = zeit.classify(segments, rf, date="2020-07-01", probability=True)
land_cover.label.zeit.plot()                                   # legend: the class names
zeit.save_raster(land_cover, "results/land_cover")             # label.tif, probability.tif

stack = zeit.load_raster("s2_2022.tif")                         # any cube: each date and band a feature
classes = zeit.classify(stack, zeit.train_classifier(stack, points))
```

## Masks

### `extract_water_mask` { .api }

<!-- sig: zeit.masks.extract_water_mask -->
```python
zeit.masks.extract_water_mask(
    segments, green="green", swir="swir1", threshold=500.0,
)
```

Persistent water from CCDC models: water reflects a little in green and almost nothing in SWIR, so a pixel is water when its green level is above its SWIR level and the SWIR level is below `threshold`. The levels are those of each pixel's first segment at its middle (`a0 + c1 t`). Also exported as `zeit.extract_water_mask`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `segments` | `Dataset` | required | A [`zeit.ccdc`](change-detection.md#ccdc) result. A numpy stack in the layout of older versions (`(segments, parameters, rows, cols)`) is still accepted, with integer band positions. |
| `green` | `str` or `int` | `"green"` | The green band. |
| `swir` | `str` or `int` | `"swir1"` | The SWIR band (SWIR1 or SWIR2). |
| `threshold` | `float` | `500.0` | Highest SWIR level of water (reflectance × 10000). |

</div>

**Returns** a `uint8` `(y, x)` map, `1` = water, georeferenced for a Dataset.

```python
water = zeit.extract_water_mask(segments)
```

## Validation

### `generate_landtrendr_accuracy_dashboard` { .api }

<!-- sig: zeit.validation.generate_landtrendr_accuracy_dashboard -->
```python
zeit.validation.generate_landtrendr_accuracy_dashboard(
    cube, points, lt_results=None,
    output_html="lt_accuracy_dashboard.html", window_size=25,
    year_dim="time",
)
```

Builds a self-contained HTML page for reviewing LandTrendr results at reference points. For each point it shows the index trajectory, the fitted segments, and true-colour image chips for every year, and lets you record the observed year of change. It computes agreement (including Kappa) live and exports the labels as CSV. Also exported as `zeit.generate_landtrendr_accuracy_dashboard`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `xr.DataArray` | required | The input cube, with time, band, y and x dimensions (true-colour bands are detected automatically). |
| `points` | `str`, GeoDataFrame or list | required | Vector file path, GeoDataFrame, or list of `(lon, lat)` tuples. An `id` column is used if present. |
| `lt_results` | `xr.DataArray` or `xr.Dataset` | `None` | LandTrendr / `extract_events` output, to show predicted years and fits. |
| `output_html` | `str` | `"lt_accuracy_dashboard.html"` | Output page. |
| `window_size` | `int` | `25` | Chip size in pixels (odd). |
| `year_dim` | `str` | `"time"` | Name of the time dimension of `cube`. |

</div>

```python
zeit.generate_landtrendr_accuracy_dashboard(
    cube=annual_cube,
    points="data/validation_points.shp",
    lt_results=events_ds,
    output_html="validation.html",
)
```
