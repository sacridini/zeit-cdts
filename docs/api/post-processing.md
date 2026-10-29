# Post-processing

<p class="lead">Clean up maps, classify CCDC results and derive masks.</p>

## Spatial filters

The filters take a map (a `DataArray`, a numpy array or a raster path) and give back the same kind, georeferencing, NoData and attributes kept.

### `apply_mmu_filter` { .api }

<!-- sig: zeit.spatial.apply_mmu_filter -->
```python
zeit.spatial.apply_mmu_filter(data, mmu_pixels=11, *, nodata="auto")
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
zeit.spatial.apply_majority_filter(
    image, size=3, *, iterations=1, nodata="auto", n_jobs=-1,
)
```

Majority (mode) filter: each pixel of a class map takes the most common class of its window (cut at the edges), its own when it is among the most common (else the smallest of them), `iterations` times or until nothing changes. NoData pixels stay and do not vote. Also exported as `zeit.apply_majority_filter`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `image` | `DataArray`, `ndarray` or path | required | 2-D class map. |
| `size` | `int` | `3` | Window size, odd (3 = 3 × 3). |
| `iterations` | `int` | `1` | How many times to apply it. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | The map's NoData (NaN for floats), a number, or none. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

### `mmu_sieve` { .api }

<!-- sig: zeit.spatial.mmu_sieve -->
```python
zeit.spatial.mmu_sieve(
    data, min_size=11, *, connectivity=8, nodata="auto",
)
```

Minimum mapping unit of a class map by a sieve: each patch (pixels of one class connected 4- or 8-wise) of fewer than `min_size` pixels is merged, the smallest first, into the neighbouring patch with which it shares the longest border (on a tie, the largest), and takes its class; a merge that joins patches of one class makes a single patch of them. Unlike [`apply_mmu_filter`](#apply_mmu_filter), the map stays complete (as with GDAL's sieve): the small patches take a class, they are not emptied. A patch surrounded only by NoData stays. Also exported as `zeit.mmu_sieve`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `ndarray` or path | required | 2-D class map. |
| `min_size` | `int` | `11` | Patches with fewer pixels are merged (11 Landsat pixels ≈ 1 ha). |
| `connectivity` | `int` | `8` | `4` or `8`: diagonal pixels in the same patch. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | The map's NoData (NaN for floats), a number, or none. |

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

### `smooth_probs` { .api }

<!-- sig: zeit.spatial.smooth_probs -->
```python
zeit.spatial.smooth_probs(
    probs, window_size=9, neigh_fraction=0.5, smoothness=20.0, *,
    n_jobs=-1,
)
```

Bayesian smoothing of class probabilities, as sits' `sits_smooth` (Câmara et al.). Each class's logit, `log(p / (sum - p))`, is pulled towards the mean `m` of the highest `neigh_fraction` of the logits of its window (mirrored at the edges), the more the less that neighbourhood varies (`s²`) against `smoothness`: `w = s² / (s² + smoothness)`, `x' = w x + (1 - w) m`. A pixel whose model is confident keeps its class against consistent neighbours, while the noise among uncertain pixels goes. The result equals sits 1.5.4's to 1e-15 (tests/test_postproc.py), except that a missing pixel stays missing where sits gives it its neighbours' mean. On 1000 x 1000 pixels and 4 classes, sits takes 10.7 s; Zeit 1.1 s on one thread, 0.15 s on 19 (benchmarks/vs_originals/smooth_probs.py). Also exported as `zeit.smooth_probs`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `probs` | `DataArray`, `ndarray` or path | required | `(class, y, x)` probabilities in [0, 1], e.g. `zeit.classify(..., probability=True).probability`. |
| `window_size` | `int` | `9` | The neighbourhood, odd. |
| `neigh_fraction` | `float` | `0.5` | The fraction of the neighbourhood's highest logits that make its mean. |
| `smoothness` | `float` or list | `20.0` | The prior's variance, one or one per class: higher smooths more. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** the smoothed probabilities, of the input's kind. As sits', they no longer sum to one: the class of a pixel is their `argmax` over `class`.

```python
land_cover = zeit.classify(stack, rf, probability=True)
probs = zeit.smooth_probs(land_cover.probability, window_size=9, smoothness=20)
label = (probs.fillna(-1).argmax("class") + 1).where(probs.notnull().any("class"))   # 1 = the first class; NaN where no data
label = zeit.mmu_sieve(label, min_size=11)            # then no patch under ~1 ha
```

### `desawtooth_classes` { .api }

<!-- sig: zeit.spatial.desawtooth_classes -->
```python
zeit.spatial.desawtooth_classes(
    data, max_length=1, *, nodata="auto", n_jobs=-1,
)
```

Temporal consistency of yearly (or any) classifications: in each pixel's series, a run of at most `max_length` dates of one class between two runs of another, the same before and after (forest, pasture, forest), takes that class (forest, forest, forest). The runs are those of the series given: forest, pasture, crop, pasture, forest with `max_length=1` only changes the crop. NoData dates stay and do not break a run. Also exported as `zeit.desawtooth_classes`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `ndarray` or path | required | Classes `(time, y, x)` (a DataArray's `time` anywhere; numpy: time first) or one series. |
| `max_length` | `int` | `1` | The longest excursion removed, in dates. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | The map's NoData (NaN for floats), a number, or none. |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

## Classification

### `train_classifier` { .api }

<!-- sig: zeit.train_classifier -->
```python
zeit.train_classifier(
    data, samples, *, label="class", model=None, date=None,
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

**Returns** the fitted model, with the feature names in `zeit_features_` and the training features in `zeit_train_` (for [`aoa`](#aoa)).

### `classify` { .api }

<!-- sig: zeit.classify -->
```python
zeit.classify(data, model, *, date=None, probability=False)
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

## Area of applicability

### `aoa` { .api }

<!-- sig: zeit.aoa -->
```python
zeit.aoa(
    data, train=None, *, model=None, weights=None, folds=None,
    distance="euclidean", lpd=False, date=None, n_jobs=-1,
)
```

Where a model can be applied (Meyer & Pebesma 2021): the dissimilarity index (DI) of each pixel to the training data, and the area of applicability (AOA), where it is no more dissimilar than the training data are to each other, as CAST's `aoa` (the same DI as CAST 1.0.4, to 1e-13). The predictors are scaled by the training data's means and standard deviations and weighted by their importance; a pixel's DI is its distance to the nearest training point over the mean distance between training points. The threshold is the upper whisker (75% quantile + 1.5 IQR, at most their largest) of the training points' own DI, each the distance to the nearest point of another cross-validation fold. A model's predictions outside the AOA are extrapolations: its validation says nothing about them. The nearest training points are found in a k-d tree up to 9 predictors (as CAST's FNN), from a matrix product beyond, the distances exact either way: with 2000 training points and 100,000 pixels, CAST takes 1.2 s with 8 predictors and 5.8 s with 64; Zeit 0.45 s and 1 s on one thread, 0.04 s and 0.3 s on 19 (benchmarks/vs_originals/aoa.py).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset`, `DataFrame` or array | required | What the model is applied to: anything `classify` takes, its features matched to the training's by name, or a table of points (predictors in columns). |
| `train` | `DataFrame`, array or `TrainDI` | `None` | The training data (points × predictors), or a `train_di` result to reuse; none with `model`. |
| `model` | classifier | `None` | A model from `train_classifier`: its training data (`zeit_train_`) and, unless `weights` says otherwise, its `feature_importances_` as the weights. |
| `weights` | `str`, `dict` or list | `None` | The predictors' weights: `None` (the model's importance, else all 1), `"importance"`, a dict by name or a list. Negative weights count as 0; a predictor of weight 0 is ignored, even missing. |
| `folds` | list or `SpatialFolds` | `None` | Each training point's cross-validation fold (any labels; `None`/NaN: in none), or a [`spatial_folds`](#spatial_folds) result (a buffer's neighbours are left out of each point's DI, as CAST's `CVtrain`). Spatially clustered samples should be given spatial folds, or the threshold is too strict. |
| `distance` | `str` | `"euclidean"` | `"euclidean"` or `"mahalanobis"` (CAST's `method = "MD"`): the Euclidean distance once the scaled predictors are whitened by their covariance in the training data, so correlated predictors (eight years of NDVI) do not count the same information several times. The weights then only say which predictors are used (0: left out): the distance does not change with their scale. With a `TrainDI`, its own. |
| `lpd` | `bool` | `False` | Also the local point density: how many training points are within the threshold's DI. |
| `date` | date | `None` | CCDC segments: the date whose models are the features. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns**, for a map, an `xarray.Dataset` georeferenced as the data: `DI (y, x)` float32; `AOA (y, x)`, `1` inside, `0` outside, `255` where a feature is missing; `LPD (y, x)` with `lpd=True`; the `threshold`, `mean_distance` and `distance` in its attributes. For points, a `DataFrame` of `DI`, `AOA` (NaN where a predictor is missing; CAST calls those inside) and `LPD`. With either distance the DI is CAST's to 1e-13; it is computed by blocks of pixels against every training point, the nearest ones exactly.

```python
rf = zeit.train_classifier(stack, "samples.gpkg", label="class")
land_cover = zeit.classify(stack, rf)
applicable = zeit.aoa(stack, model=rf)                          # weighted by the forest's importance
land_cover["label"] = land_cover.label.where(applicable.AOA == 1, 0)
zeit.save_raster(applicable, "results/aoa")                      # DI.tif, AOA.tif

# samples in clusters (sites, polygons): their spatial cross-validation folds
applicable = zeit.aoa(stack, model=rf, folds=samples["site"], lpd=True)
```

### `train_di` { .api }

<!-- sig: zeit.train_di -->
```python
zeit.train_di(
    train=None, *, model=None, weights=None, folds=None,
    distance="euclidean", n_jobs=-1,
)
```

The dissimilarity of the training data to itself and the AOA's threshold, as CAST's `trainDI`, to reuse with `aoa(data, train=td)` on several images.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `train` | `DataFrame` or array | `None` | The training data (points × predictors); points with a missing value are left out. None with `model`. |
| `model` | classifier | `None` | As in `aoa`. |
| `weights` | `str`, `dict` or list | `None` | As in `aoa`. |
| `folds` | list | `None` | As in `aoa`. |
| `distance` | `str` | `"euclidean"` | As in `aoa`. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** a `TrainDI`: `features`, `center`, `scale`, `weight`, `train` (scaled and weighted; whitened for the Mahalanobis distance), `mean_distance`, `train_di` (each training point's DI), `threshold`, `distance` and `whiten` (the matrix that whitens a scaled point, or `None`).

## Spatial cross-validation

Points near each other are alike: a random k-fold validation tests a model on points next to its training points and says it is better than it is where it will be applied. Spatial folds hold out whole regions, as blockCV's (Valavi et al. 2019): blocks of a grid, a point and its neighbours within a buffer, or clusters of the points. They serve to choose a model and to calibrate its area of applicability (`train_di`, `aoa`); the accuracy of the map is still that of a probability sample ([`accuracy`](validation.md#accuracy); Wadoux et al. 2021).

### `spatial_folds` { .api }

<!-- sig: zeit.spatial_folds -->
```python
zeit.spatial_folds(
    samples, *, method="blocks", k=5, size=None, shape="hexagon",
    selection="random", label=None, iterations=100,
    space="geographic", data=None, seed=0, n_jobs=-1,
)
```

Cross-validation folds of sample points that keep the test points away from the training points, as blockCV's `cv_spatial`, `cv_buffer` and `cv_cluster`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `samples` | `GeoDataFrame` or path | required | The points. Distances are in their CRS's units; a geographic CRS is replaced by its UTM zone (with `data`, the points are first taken to its CRS). |
| `method` | `str` | `"blocks"` | `"blocks"`: the points in blocks of a grid, the blocks in `k` folds. `"buffer"`: leave-one-out, each point tested without its neighbours within `size` in the training. `"clusters"`: `k` clusters by k-means, of the coordinates or of the predictors (`space="feature"`). |
| `k` | `int` | `5` | The number of folds (not for the buffer). |
| `size` | `float` | `None` | The blocks' size (a square's side; for hexagons, the distance between neighbouring centres) or the buffer's radius, in the CRS's units. For blocks, `None` takes the median range of `data`'s spatial autocorrelation (`autocorrelation_range`). |
| `shape` | `str` | `"hexagon"` | `"hexagon"` or `"square"`. |
| `selection` | `str` | `"random"` | How blocks go to folds: `"random"` (the most balanced of `iterations` draws), `"systematic"` (in rows) or `"checkerboard"` (squares, `k=2`). |
| `label` | `str` | `None` | A column of `samples` with the classes: the random draws then balance each class across the folds. |
| `iterations` | `int` | `100` | Random draws of the blocks' folds; the one closest to a k-th of the points (of each class) in every fold is kept. |
| `space` | `str` | `"geographic"` | For clusters: `"geographic"` (the coordinates) or `"feature"` (the predictors in `data` at the points, standardised). |
| `data` | `DataArray`, `Dataset` or path | `None` | The predictors (anything `classify` takes), for `size=None` and `space="feature"`. |
| `seed` | `int` | `0` | The random draws' seed. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** a `SpatialFolds`: `folds` (each point's fold, `-1` in none: never tested, always trained on), `points`, `size`, `blocks` (the blocks' polygons and folds, for `"blocks"`) and `exclude` (each point's neighbours, for `"buffer"`). It is a scikit-learn splitter (`split`, `get_n_splits`), for `cross_val_predict` or `GridSearchCV`, and `train_di` and `aoa` take it as `folds` (the buffer's neighbours left out of each point's DI, as CAST's `CVtrain`). `.plot()` draws the points by fold.

```python
folds = zeit.spatial_folds(samples, data=stack, k=5, label="class")    # hexagons as large as the autocorrelation
folds.plot()
from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(rf, x, y, cv=folds)
applicable = zeit.aoa(stack, model=rf, folds=folds)
```

### `autocorrelation_range` { .api }

<!-- sig: zeit.autocorrelation_range -->
```python
zeit.autocorrelation_range(
    data, *, sample=5000, bins=15, cutoff=None, seed=0, n_jobs=-1,
)
```

The range of each band's spatial autocorrelation, as blockCV's `cv_spatial_autocor`: the empirical semivariogram of a sample of pixels and an exponential model fitted to it, γ(h) = nugget + sill (1 − e<sup>−h/scale</sup>), by weighted least squares (weights N<sub>h</sub>/h², as gstat). The pairs are counted in the core, in parallel: 5000 pixels take 0.03 s.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset` or path | required | Anything `classify` takes; a geographic CRS's pixels are measured in their UTM zone. |
| `sample` | `int` | `5000` | Pixels with a value drawn per band. |
| `bins` | `int` | `15` | Bands of distance of the semivariogram. |
| `cutoff` | `float` | `None` | The largest distance of a pair; default a third of the map's diagonal (as gstat). |
| `seed` | `int` | `0` | The sample's seed. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** a `DataFrame` by band: `range` (the practical range, where the model reaches 95% of its sill above the nugget: 3 `scale`), `scale`, `nugget`, `sill` (above the nugget) and `pixels`; in `attrs`, the median `range` and each band's semivariogram (`variograms`, a `DataFrame` of `distance`, `gamma` and `pairs`).

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

Accuracy assessment and area estimation, and labelling reference points by eye, have a page of their own: [Validation](validation.md).
