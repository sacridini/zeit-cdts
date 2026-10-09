# Post-processing

<p class="lead">Clean up maps, classify CCDC results, derive masks, and review results against reference points.</p>

## Spatial filters

### `apply_mmu_filter` { .api }

<!-- sig: zeit.spatial.apply_mmu_filter -->
```python
zeit.spatial.apply_mmu_filter(input_path, output_path, mmu_pixels=11)
```

Minimum mapping unit filter for a single-band GeoTIFF on disk: connected patches of non-nodata pixels smaller than `mmu_pixels` are set to nodata. Removes the salt-and-pepper noise of per-pixel change maps. Also exported as `zeit.apply_mmu_filter`; CLI: `zeit mmu-filter`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_path` | `str` | required | Single-band GeoTIFF, e.g. a year-of-loss map. Background must be the file's nodata value (or `0`). |
| `output_path` | `str` | required | Filtered output. |
| `mmu_pixels` | `int` | `11` | Smallest patch to keep, in pixels (11 Landsat pixels ≈ 1 ha). |

</div>

```python
zeit.apply_mmu_filter("results/loss_year.tif", "results/loss_year_mmu.tif", mmu_pixels=11)
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
| `image` | `np.ndarray` | required | 2-D class map. |
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
| `probs` | `np.ndarray` | required | `(classes, rows, cols)` probabilities or scores. |
| `window_size` | `int` | `3` | Averaging window. |

</div>

**Returns** a `(rows, cols)` class-index map.

## CCDC classification

### `train_ccdc_classifier` { .api }

<!-- sig: zeit.classify.train_ccdc_classifier -->
```python
zeit.classify.train_ccdc_classifier(
    X_train, y_train, n_estimators=100, random_state=42,
)
```

Trains a scikit-learn random forest on CCDC features. Also exported as `zeit.train_ccdc_classifier`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `X_train` | `np.ndarray` | required | `(samples, features)`. Must have the same layout as the bands of the coefficient GeoTIFF you will classify, e.g. that file sampled at your training points. |
| `y_train` | `np.ndarray` | required | Class labels, `1…255` (`0` is used for no data in the output). |
| `n_estimators` | `int` | `100` | Number of trees. |
| `random_state` | `int` | `42` | Random seed. |

</div>

**Returns** a fitted `RandomForestClassifier`.

### `classify_ccdc_stack` { .api }

<!-- sig: zeit.classify.classify_ccdc_stack -->
```python
zeit.classify.classify_ccdc_stack(
    clf, coef_stack_path, output_path, chunk_size=512,
)
```

Applies a classifier to every pixel of a feature GeoTIFF, block by block, and writes a `uint8` class map. A natural feature stack is the coefficients of one [`zeit.ccdc`](change-detection.md#ccdc) segment written with `save_raster`, one band per band and coefficient (`blue_a0`, `blue_c1`, …, `swir2_b3`). Pixels whose first two bands are both zero are left as `0`. Also exported as `zeit.classify_ccdc_stack`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `clf` | classifier | required | Any fitted scikit-learn classifier. |
| `coef_stack_path` | `str` | required | CCDC coefficient GeoTIFF. |
| `output_path` | `str` | required | Output class map. |
| `chunk_size` | `int` | `512` | Block size in pixels. |

</div>

```python
from zeit.classify import train_ccdc_classifier, classify_ccdc_stack

segments = zeit.ccdc(cube, qa="fmask")
features = segments.coefs.isel(segment=0).fillna(0)     # first segment; 0 where a pixel has no model
zeit.save_raster(features, "results/coefs.tif")          # 48 bands: blue_a0 ... swir2_b3

clf = train_ccdc_classifier(X_train, y_train)            # X_train: coefs.tif sampled at labelled points
classify_ccdc_stack(clf, "results/coefs.tif", "results/land_cover.tif")
```

## Masks

### `extract_water_mask` { .api }

<!-- sig: zeit.masks.extract_water_mask -->
```python
zeit.masks.extract_water_mask(
    ccdc_coefs_stack, green_band_idx, swir_band_idx,
)
```

Persistent water mask from the first CCDC model of each pixel: water is brighter in Green than in SWIR and dark in SWIR. Also exported as `zeit.extract_water_mask`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ccdc_coefs_stack` | `np.ndarray` | required | `(segments, parameters, rows, cols)` in the numpy layout of older CCDC versions: per segment `t_start`, `t_end`, `t_break`, then each band's RMSE and 8 coefficients. |
| `green_band_idx` | `int` | required | 0-based index of the Green band. |
| `swir_band_idx` | `int` | required | 0-based index of the SWIR band. |

</div>

**Returns** a `uint8` `(rows, cols)` mask, `1` = water.

!!! bug "Known issue"
    The current implementation locates each band's intercept assuming 7 parameters per band, while CCDC outputs 9 (RMSE plus 8 coefficients). For any band other than the first, the wrong coefficient is read. Until this is fixed, compute the mask from the intercepts directly: in a [`zeit.ccdc`](change-detection.md#ccdc) result they are `segments.coefs.sel(segment=1, coef="a0")`; in the numpy layout, the intercept of band `b` is at parameter index `4 + 9 * b`.

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
