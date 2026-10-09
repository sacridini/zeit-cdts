# Forest Degradation (CODED)

<p class="lead">Deforestation is easy to see from space: the forest is gone. Degradation is not. Selective logging opens small gaps, an understory fire scorches the lower layers, and within two years the canopy closes again. Each Landsat pixel only darkens a little, for a short time. CODED finds these events by unmixing every observation into its materials and following the result through time, then tells degradation (the land is still forest afterwards) from deforestation (it is not).</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Where and when was the forest degraded or cleared?</span></div>
<div><span class="k">Input</span><span class="v">Every Landsat observation, six reflectance bands: <code>(time, band, y, x)</code></span></div>
<div><span class="k">Output</span><span class="v">Dates, magnitudes and types of change; a strata map for sample-based area estimates</span></div>
<div><span class="k">Reference</span><span class="v">Bullock, Woodcock &amp; Olofsson (2020), <em>Remote Sensing of Environment</em> 238; Souza et al. (2005), <em>Remote Sensing of Environment</em> 98</span></div>
</div>

## How it works

1. **Spectral mixture analysis.** A 30 m pixel is a mix of materials. `zeit.unmix` finds, for every observation, the fractions of green vegetation (GV), non-photosynthetic vegetation (NPV: dead wood, litter), soil, shade and cloud that best rebuild its spectrum, with the endmembers of Souza et al. (2005). Logging raises NPV and soil where the canopy was; the reflectance barely changes, the fractions do.
2. **NDFI.** The fractions are combined into the Normalized Difference Fraction Index: `(GVs − (NPV + soil)) / (GVs + NPV + soil)`, with `GVs = GV / (1 − shade)`. Closed forest is near 1, damaged forest lower, bare soil and pasture below 0.
3. **Monitoring.** For each pixel, a model of the NDFI (a constant plus an annual harmonic) is fitted to a training period, with its RMSE. After it, each observation is compared with the model: when `consec` observations in a row fall more than `thresh` RMSEs below it, there is a change, dated at the first of them.
4. **What came after.** A new model is fitted to the `min_years` after each change. If it is forest, the change was degradation; if not, deforestation. Forest or not is decided by a random forest trained on land cover points (as in CODED), or else by the model's mean NDFI.

## Step by step

The [example 21](https://github.com/sacridini/zeit-cdts/blob/main/examples/example_21_coded_degradation.py) builds a synthetic Landsat series by mixing known fractions: intact forest, forest selectively logged in mid-2006 (with cuts of varying intensity; the canopy closes within two years), forest cleared in mid-2008, and pasture, with a third of the observations lost to clouds.

### 1. Unmix

```python
import zeit

fractions = zeit.unmix(landsat, cloud_threshold=0.05)   # gv, shade, npv, soil, cloud, rmse, ndfi
```

Clouds that the QA mask missed have a large cloud fraction, and `cloud_threshold` (CODED uses 0.05) removes them. Use every observation, not composites: degradation lasts a year or two, and a composite blurs it away.

### 2. Run CODED

```python
result = zeit.coded(fractions, start=2005)                      # monitor from 2005, train on 2002-2004
result = zeit.coded(fractions, start=2005, training="lc.gpkg")   # forest or not from land cover points
```

`result.strata` is a map of forest, non-forest, degradation and deforestation (by the first change), and `result.t_change`, `result.type` and `result.ndfi_change` describe each change. `zeit.plot(fractions.ndfi, fit=result)` shows the NDFI of any pixel with the changes marked.

| Parameter | Default | Effect |
| :--- | :---: | :--- |
| `train_years` | 3 | Years that define the forest's normal state. |
| `consec` | 3 | Observations in a row beyond the threshold. More: fewer false alarms, later detections. |
| `thresh` | 3 | Threshold in RMSEs of the training model. Lower finds lighter degradation and more noise. |
| `min_years` | 3 | Years after a change that define the new land cover. |
| `forest_ndfi` | 0.5 | Without training points: forest where the model's mean NDFI is at least this. |

### 3. Estimate the area

CODED's authors recommend using the map as strata for a sample, not as the area itself: light degradation is easily missed. The strata go straight into [`sampling_design`](../api/validation.md#sampling_design), [`interpret`](../api/validation.md#interpret) and [`accuracy`](../api/validation.md#accuracy):

```python
points = zeit.stratified_sample(result.strata, n=200, min_per_stratum=30)
s = zeit.interpret(landsat, points, rgb=["swir1", "nir", "red"], series=fractions.ndfi,
                   map=result.strata, fit=result, save="reference.gpkg")
acc = s.accuracy()
```

In the example the reference comes from the known truth instead of an interpreter. The map finds 51 ha of degradation and misses the lightest cuts. The sample sees them, and the estimate is 56 ± 6 ha, close to the 54 ha that were logged:

```
Accuracy of the map, from 200 reference points:
  overall: 0.989 ± 0.015 (95% CI)

                      user's     producer's mapped (ha) estimated (ha)
class
forest         0.979 ± 0.029  1.000 ± 0.000         219        214 ± 6
non_forest     1.000 ± 0.000  1.000 ± 0.000         108        108 ± 0
degradation    1.000 ± 0.000  0.916 ± 0.106          51         56 ± 6
deforestation  1.000 ± 0.000  1.000 ± 0.000          54         54 ± 0
```

!!! note "About this implementation"
    `zeit.coded` follows the version of CODED described in the paper (its `cdd_simple.py`): a training model, monitoring by normalized residuals, and a new model after each change. The Earth Engine versions 1 and 2 use Earth Engine's CCDC for the monitoring and give similar but not identical results. The C++ monitoring is checked against a Python implementation of the algorithm, and the unmixing against `scipy.optimize.nnls`. The endmembers are the ones CODED uses (Souza et al. 2005, plus a cloud endmember), for Landsat 5, 7 and 8 surface reflectance.
