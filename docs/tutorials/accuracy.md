# Accuracy & Area

<p class="lead">A change map says that 18,000 ha were deforested. How sure is that number? Counting the pixels of a map gives a biased area: every map has omission and commission errors, and they rarely cancel. The good practice is to label a random sample of the map, estimate the error matrix in proportions of area, and correct the area with it. That gives an unbiased area with a confidence interval, which is what reports such as REDD+ or a paper's results need.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">How accurate is the map? How much area really changed, ± how much?</span></div>
<div><span class="k">Input</span><span class="v">Any map of classes or of events, and reference labels at sample points</span></div>
<div><span class="k">Output</span><span class="v">Overall, user's and producer's accuracies and error-adjusted areas, with confidence intervals</span></div>
<div><span class="k">Reference</span><span class="v">Olofsson et al. (2014), <em>Remote Sensing of Environment</em> 148; Stehman (2014), <em>IJRS</em> 35</div>
</div>

## How it works

1. **Stratify by the map.** Each class of the map is a stratum. Rare classes, change above all, would get few points in a simple random sample, so each stratum gets its own sample.
2. **Label the sample.** For each point, decide what really happened, from imagery better than the map's input or from the field. This is the reference.
3. **Estimate.** Each point stands for the area of its stratum divided by the stratum's sample size. Weighing the error matrix by those areas gives the proportion of the area in each (map, reference) pair. From that come the accuracies and the area of each reference class, with standard errors.

The same three steps work for any map, from zeit or not, and for maps of events: they are stratified as "no change" and "change".

## Step by step

### 1. Make the map of events

```python
import zeit

ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
loss = zeit.extract_events(zeit.landtrendr(ndvi), min_magnitude=1500)
```

Any change algorithm works the same way: `zeit.extract_events` gives the same maps for [CCDC](ccdc.md), [BFAST](bfast.md) and the others, and `zeit.agreement` combines them.

### 2. Size and draw the sample

```python
design = zeit.sampling_design(loss, expected_ua={"change": 0.7, "no change": 0.95}, std_error=0.01)
print(design)        # per stratum: name, pixels, area (ha), weight, expected_ua, n

points = zeit.stratified_sample(loss, design=design)
points.to_file("loss_points.gpkg")
```

`expected_ua` is your guess of each stratum's user's accuracy, and `std_error` the precision you want for the overall accuracy: together they set the sample size (Olofsson et al. 2014, eq. 13). Change is rare, so a proportional allocation would give it a handful of points; `min_per_stratum=50` (the default) moves points from the large strata to it.

### 3. Label the points

```python
s = zeit.interpret(landsat, points, rgb=["swir1", "nir", "red"], series="nbr",
                   map=loss, fit=zeit.landtrendr(ndvi), save="loss_reference.gpkg")
```

`zeit.interpret` goes through the points one by one: the map is centred on each, a strip of chips shows the place every year, and the chart its series. Press `1` or `2` for the class, click the chart (or press `d` on the image where the change appears) to date it, and `Enter` to save and move on. Each label is written to `loss_reference.gpkg` at once: close the window and run the same line tomorrow to go on.

By default it is **blind**: what the map and LandTrendr say at a point shows only once you have labelled it. A reference swayed by the map would only measure how much you agree with it. When every point has a label, the **Review** tab shows the accuracies, and a click on a cell of the error matrix (say, map "change", reference "no change") lists those points to look at again.

QGIS, Collect Earth or field visits work as well: what `zeit.accuracy` needs is a `ref` column (and `ref_date` for dates) in the points.

### 4. Estimate accuracy and area

```python
acc = s.accuracy(date_tolerance=1)          # or zeit.accuracy(loss, "loss_reference.gpkg", date_tolerance=1)
print(acc)
```

The summary looks like this one, of the numerical example in Olofsson et al. (2014): a map of 10 million 30 m pixels in four classes and 640 labelled points (`zeit` reproduces the paper's numbers):

```
Accuracy of the map, from 640 reference points:
  overall: 0.947 ± 0.018 (95% CI)

                         user's     producer's mapped (ha)    estimated (ha)
class
deforestation     0.880 ± 0.074  0.749 ± 0.213      18,000    21,158 ± 6,158
forest_gain       0.733 ± 0.101  0.847 ± 0.254      13,500    11,686 ± 3,756
stable_forest     0.927 ± 0.040  0.935 ± 0.034     288,000  285,770 ± 15,510
stable_nonforest  0.963 ± 0.021  0.962 ± 0.018     580,500  581,386 ± 16,281
```

The map shows 18,000 ha of deforestation; the sample says 21,158 ha, between 15,000 and 27,315 ha with 95% confidence. With `date_tolerance`, a last line gives the share of the change points whose dates agree.

- **User's accuracy** of change: of the pixels the map calls change, how many are. Low values mean false alarms (commission).
- **Producer's accuracy** of change: of the real change, how much the map finds. Low values mean missed change (omission).
- **Estimated area**: the area of change corrected for both, with its 95% interval. Report this, not the mapped area.

## Assessing other maps with the same points

The reference describes the ground, not the map, so one set of labelled points assesses every map of the same area: LandTrendr, CCDC, the agreement of both. Pass the map the sample was drawn from as `strata`, so the points keep their weights:

```python
ccdc_loss = zeit.extract_events(zeit.ccdc(cube), band="nir", event_type="loss")
acc_ccdc = zeit.accuracy(ccdc_loss, "loss_reference.gpkg", strata=loss)
```

!!! tip "Validation checked against the reference implementations"
    `zeit.accuracy` reproduces the numerical example of Olofsson et al. (2014) (areas, accuracies and their intervals) and R's `sits_accuracy` to 1e-9. With a sample stratified by one map assessing another, its 95% intervals cover the true accuracy in about 95% of repeated samples.
