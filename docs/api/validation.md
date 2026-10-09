# Validation

<p class="lead">How accurate is a map, and how much area really changed? Draw a stratified random sample of the map, label it, and get the error matrix, the accuracies and the error-adjusted area of each class with their confidence intervals, following the good practices of Olofsson et al. (2014).</p>

The three functions work on any map: classes from [`classify`](post-processing.md#classify), [`ai.predict`](ai.md#predict), [`twdtw`](time-series.md#twdtw) or [`som`](time-series.md#som), a raster made elsewhere, or a map of events from [`extract_events`](change-detection.md#extract_events) or [`agreement`](change-detection.md#agreement) (stratified as "no change" and "change"). Areas come from each pixel's size, in hectares (in a geographic CRS, the area of each row on the sphere); dask maps are read lazily, a row strip at a time. Tutorial: [Accuracy & Area](../tutorials/accuracy.md).

```python
events = zeit.extract_events(zeit.landtrendr(ndvi))
design = zeit.sampling_design(events, expected_ua={"change": 0.7, "no change": 0.95})
points = zeit.stratified_sample(events, design=design)
points.to_file("to_label.gpkg")            # label a "ref" column: zeit.interpret, QGIS, the field...
acc = zeit.accuracy(events, "labelled.gpkg")
print(acc)                                 # overall, user's, producer's, mapped and estimated areas
```

## `sampling_design` { .api }

<!-- sig: zeit.sampling_design -->
```python
zeit.sampling_design(
    data, expected_ua=0.75, std_error=0.01, n=None,
    alloc="proportional", min_per_stratum=50, bins=None,
    nodata="auto",
)
```

How many points the map needs to reach a target standard error of the overall accuracy (Olofsson et al. 2014, eq. 13), and how many in each stratum. The strata are the map's classes. Also exported as `zeit.sampling_design`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset` or path | required | The map to assess: classes (integers; names from its `flag_meanings`), or a map of events. |
| `expected_ua` | `float` or `dict` | `0.75` | The user's accuracy you expect of each stratum: one value, or stratum name (or code) → value. Hard classes (0.6) need more points than easy ones (0.95). |
| `std_error` | `float` | `0.01` | Target standard error of the overall accuracy. |
| `n` | `int` | `None` | Total number of points; computed from `expected_ua` and `std_error` when `None`. |
| `alloc` | `str` or `dict` | `"proportional"` | `"proportional"` (to the area of each stratum), `"equal"`, `"neyman"` (to area × expected standard deviation), or stratum → number of points. |
| `min_per_stratum` | `int` | `50` | Strata that would get fewer points get this many, taken from the larger ones, so the total stays `n`. Olofsson et al. recommend 50–100 for rare classes such as change. `0` disables. |
| `bins` | sequence of `float` | `None` | Strata as intervals `[a, b)` of the map's values. For a map of events, periods of `yod` (`[2000, 2010, 2020]`), next to "no change". |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value outside every stratum: the map's NoData, a number, or none. A map of events keeps its `0` as "no change". |

</div>

**Returns** a `pandas.DataFrame` with one row per stratum (indexed by its code): `name`, `pixels`, `area`, `weight` (share of the area), `expected_ua` and `n`. `attrs` holds the total `n`, the `std_error`, how points were allocated and the `area_unit` (`"ha"`, or `"pixels"` without a CRS).

```python
design = zeit.sampling_design(classes, expected_ua={"forest": 0.9, "pasture": 0.85, "deforestation": 0.6})
design.attrs["n"]                                           # e.g. 641
zeit.sampling_design(events, n=300, bins=[2000, 2010, 2020])  # no change, 2000-2009, 2010-2019
```

## `stratified_sample` { .api }

<!-- sig: zeit.stratified_sample -->
```python
zeit.stratified_sample(
    data, n=None, design=None, bins=None, nodata="auto", seed=42,
    **design_kwargs,
)
```

Draws the points: a simple random sample of pixels within each stratum, without replacement. The same map and seed give the same points in memory or lazily, whatever the chunks. Also exported as `zeit.stratified_sample`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, `Dataset` or path | required | The map, as for `sampling_design`. |
| `n` | `int` | `None` | Total number of points, allocated as `sampling_design` does. |
| `design` | `DataFrame` | `None` | The table of `sampling_design`; its `n` column is used as is. |
| `bins`, `nodata` | | `None`, `"auto"` | As for `sampling_design`. |
| `seed` | `int` | `42` | Seed of the draw. |
| `design_kwargs` | | | Passed to `sampling_design` when there is no `design` (`expected_ua`, `alloc`, `min_per_stratum`...). |

</div>

**Returns** a `geopandas.GeoDataFrame` of points at the centres of the sampled pixels, in the map's CRS, with `stratum` (code), `stratum_name`, `row` and `col`. Keep the `stratum` column: [`accuracy`](#accuracy) weighs the points by it.

## `accuracy` { .api }

<!-- sig: zeit.accuracy -->
```python
zeit.accuracy(
    map, samples, reference="ref", strata=None, bins=None,
    date_tolerance=None, reference_date="ref_date", confidence=0.95,
    nodata="auto",
)
```

The map's error matrix in proportions of area, its overall, user's and producer's accuracies and the error-adjusted area of each class, each with a confidence interval. The estimators are Stehman's (2014) for stratified random sampling; with the map's classes as strata they are Olofsson et al. (2014), eqs. 4–11, checked against the paper's example and against R's `sits_accuracy`. Also exported as `zeit.accuracy`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `map` | `DataArray`, `Dataset` or path | required | The map to assess (classes, or events). |
| `samples` | `GeoDataFrame`, path or `zeit.interpret` session | required | The reference points, in any CRS. Points without a reference label are left out (and counted). |
| `reference` | `str` | `"ref"` | Column with the reference class: names of the map's classes, their codes, or for a map of events `"change"`/`"no change"` (or `True`/`False`, `1`/`0`). A class the map never gives is kept (its user's accuracy is NaN, its area estimated). |
| `strata` | `DataArray`, `Dataset` or path | `None` | The map the sample was stratified by, when it is not `map`: a sample of one map assesses another (Stehman 2014). Default: `map`'s classes. |
| `bins` | sequence of `float` | `None` | Classes of `map` as intervals of its values, as in `sampling_design`. |
| `date_tolerance` | `float` | `None` | Maps of events: also compare dates. Of the points both call change, the share whose years (of the map's `date` and `reference_date`) are at most this far apart. |
| `reference_date` | `str` | `"ref_date"` | Column with the reference date of the change. |
| `confidence` | `float` | `0.95` | Level of the confidence intervals. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | As for `sampling_design`. |

</div>

**Returns** an `Accuracy` object; `print(acc)` gives a summary.

| Attribute | Meaning |
| :--- | :--- |
| `overall` | `estimate`, `se`, `ci_low`, `ci_high` of the overall accuracy. |
| `users`, `producers` | The same, per class (rows). |
| `area` | Per class: `mapped` (area the map gives it), `estimate` (error-adjusted area), `se`, `ci` (half-width of the interval), `ci_low`, `ci_high`, in `area_unit`. |
| `confusion` | Error matrix in proportions of area: map classes in rows, reference classes in columns. |
| `counts` | The error matrix in points. |
| `date` | With `date_tolerance`: `within` (estimated share of the change points whose dates agree), `within_se`, `mean_difference` (years), `n`. |
| `n`, `n_unlabelled` | Points used, and points left out for having no label. |
| `to_dataframe()` | User's, producer's and area in one table. |

```python
acc = zeit.accuracy(events, points, date_tolerance=1)
acc.area.loc["change", ["estimate", "ci"]]         # hectares of change, ± the 95% interval
acc.users.loc["change"]                            # how often a mapped change is real
acc.producers.loc["change"]                        # how much of the real change the map finds

# The same reference points assess another algorithm's map, weighed by the first map's strata
acc_ccdc = zeit.accuracy(zeit.extract_events(segments, band="nir", event_type="loss"), points, strata=events)
```

!!! warning "Points chosen by hand are not a sample"
    The estimates assume the points were drawn at random within the strata (or across the whole map). Field points or points picked on the screen give an error matrix, but not unbiased accuracies or areas. `accuracy` warns when the points have no `stratum` column and weighs them as if they were stratified by the map.
