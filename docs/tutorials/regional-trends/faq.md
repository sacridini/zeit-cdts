# Interpreting, FAQ and pitfalls

<p class="lead">How to read the output of <code>zeit.parts</code>, how far to trust it, how to report it, and the mistakes that come up most often.</p>

## Reading the output

```text
PARTS: 482234 pixels, 100 partitions of 1500                       ← (1)
spatial correlation: {'function': 'exponential', 'range': 23039.803728581348}  ← (2)
nugget (mean of the partitions): 0.08176                            ← (3)

              estimate         se        t        p
coefficient
mean        0.00151307 0.00170745 0.886154 0.375536                 ← (4)
```

1. **Pixels, partitions.** The pixels with a trend (and a class, a covariate, inside the mask) and how many of them were used (`npart × partsize`).
2. **Range**, in the CRS units. How far the residuals stay correlated: about 0.37 at one range, 0.05 at three. Compare it with the size of the region. A range close to or beyond the region's size means the region shares one signal, and a regional test then has little power.
3. **Nugget**, from 0 to 1. The share of a pixel's variation that is its own. Close to 0: neighbours move as one. Close to 1: the pixels are nearly independent.
4. **Each coefficient**: the GLS estimate, its standard error, `t = estimate / se`, and the two-sided p-value of "this coefficient is 0". For `mean`, it is the regional mean trend, in the units of the slope (per year with annual data or `time="year"`).

With classes or covariates, the last line is the **chi-square test**: whether the classes differ, or whether the covariates have an effect, with its mean F and degrees of freedom.

### What the result means, and what it does not

| Result | Means | Does not mean |
| :--- | :--- | :--- |
| `mean`: p < 0.05 | The region's mean trend differs from 0. The sign of the estimate says which way it goes. | That every pixel, or most pixels, trend that way. |
| `mean`: p ≥ 0.05 | No evidence of a regional mean trend at this level. | That there is no trend anywhere. Local trends can be real (a city, a reservoir) and average out, or the shared noise can be too large. |
| a class: p < 0.05 | That class's mean trend differs from 0. | That it differs from the other classes (that is the chi-square test). |
| chi-square p < 0.05 | The classes' mean trends are not all equal, or the covariates have an effect. | Which classes differ: look at the estimates, or test pairs ([Advanced](advanced.md#classes)). |
| per-pixel map: 26% significant | 26% of the pixels' own tests have p < 0.05. | That 26% of the area is changing, or that the region is. |

**Statistical significance is not ecological importance.** A trend of 0.0005 per year over 40 years is a change of 0.02 in the index. Report the estimate with its interval (estimate ± 1.96 SE), not the p-value alone.

## How much to trust a p-value

PARTS is far better calibrated than the alternatives, but not perfectly ([Is it calibrated?](how-it-works.md#is-it-calibrated)). In simulations with an exponential correlation, it gave p < 0.05 on 14% of the maps with no trend, because the fitted range tends to be too short. With the true range given, it gave 6.3%. In practice:

- **Look at the correlation plot** ([how](advanced.md#the-correlation-plot)). If the curve misses the dots badly, the p-value is approximate.
- **Treat p between 0.01 and 0.05 as weak evidence.** It is borderline under the method's own small bias. A p of 0.001 is not.
- **Run the sensitivity checks**: a range 1.5 to 2 times the fitted one, two or three partition sizes, two or three seeds ([Advanced](advanced.md#sensitivity-checks)). A conclusion that holds in all of them is solid.
- **Give `range=`** when an independent estimate exists, for example from a variogram of a longer record or a previous study of the region.

## Reporting

A template for the methods section:

> Per-pixel trends of [index] over [years] were estimated by REML assuming AR(1) temporal autocorrelation. The [mean trend of the region / differences in trends among the N classes / effect of covariate] were tested with PARTS (Ives et al. 2021) as implemented in Zeit [version], with an [exponential] spatial correlation function fitted to the residuals of [1000] randomly sampled pixels (range [r] km), [npart] random partitions of [partsize] pixels, a maximum-likelihood nugget per partition, and [seed]. [The sensitivity of the result to the range, the partition size and the seed was checked.]

And for the results, give the estimate, its SE (or 95% interval), t, df and p, and the share of significant pixels only as a description of the map:

> The mean trend was +0.00068 ± 0.00074 yr⁻¹ (t = 0.92, p = 0.36). Trends differed among zones (χ² test, F = 2.00, df = 2, 1498, p = 0.028). Individually, 12% of the pixels had a significant trend (p < 0.05), a figure that does not account for multiple testing or spatial correlation.

The worked example ends with [a filled-in version](worked-example.md#7-report-it).

## FAQ

**Why not just count the significant pixels and compare the share with 5%?**
Because the share varies a lot from map to map when pixels are correlated. In 300 simulated maps with no trend, it averaged 7.5% and exceeded 16% in one map in twenty. No fixed threshold separates "noise" from "trend" without modelling the correlation, and that is what PARTS does.

**Why not run a t-test or OLS on all the pixels' slopes?**
It treats the pixels as independent and gives p-values that are far too small. On the simulated map on the [overview page](index.md), it gave p = 10⁻²¹³ with no trend at all. In 300 similar, smaller maps, it gave p < 0.05 in 94% of them.

**Why not average the region into one series and test that?**
That is a valid test, and a good check: on the worked example's window it gave p = 0.094 against PARTS's 0.38. It can only answer "is the region's mean trending?", not compare classes or test covariates. It also throws away power when the correlation is short compared with the region, because it averages away the many independent parts of a large region.

**Mann-Kendall or AR(1) for the per-pixel step?**
PARTS uses the AR(1) trend: its residuals feed step 2, and its slope is the response. For maps of pixels, either is fine. Mann-Kendall with Hamed-Rao is robust to outliers, and the AR(1) trend gives a slope with a standard error.

**My region has 10,000 pixels. Is that enough?**
Yes: 6 partitions of 1500, or more partitions of 1000. With fewer than about 3000 pixels, lower `partsize` so that there are at least two partitions (`parts` raises an error otherwise). Very small regions have very little independent information when the range is long.

**What `partsize` should I use?**
Keep the default 1500 (remotePARTS's authors use 1500 to 3000). If the answer depends strongly on it, the correlation model is the problem, not the size ([Advanced](advanced.md#partition-size)).

**How many partitions?**
A few dozen to a hundred. More rarely changes the SE ([Advanced](advanced.md#how-many-partitions)).

**Can I use it on a geographic (lon/lat) raster?**
Reproject to a projected CRS first; `zeit.parts` refuses lon/lat ([Advanced](advanced.md#coordinate-systems)).

**Can I use 16-day or monthly data?**
Yes, with `time="year", harmonics=2` in both `ar_trend` and `parts` ([Advanced](advanced.md#sub-annual-data)).

**Why does another seed give another p?**
The partitions and the pixel sample are random. Differences of a few hundredths are normal ([Advanced](advanced.md#seeds-and-reproducibility)).

**Does PARTS find *where* the trends are?**
No. It tests statements about the whole region, its classes or its covariates. Maps of the per-pixel trends show where. Read them as a description, not as thousands of verdicts.

**Is the result the same as remotePARTS in R?**
The same method, to numerical precision for the same partitions. The partitions themselves differ (another random generator), so expect differences of the size of another seed ([Advanced](advanced.md#parity-with-remoteparts)).

## Pitfalls

- **Reporting the share of significant pixels as evidence of a regional trend.** The share is a description of the map. Use PARTS for the regional claim.
- **Fill values taken as data.** Zeros over the sea or outside the scene, with no nodata declared, are read as observations: a pixel that is 0 in some years and land in others gets a meaningless trend, and the sea enters the test. Mask them (`cube.where(cube != 0)`) or give `nodata=`.
- **Pixels with too few observations.** `parts` has no `min_valid`. Fit the trends with `zeit.ar_trend(cube, min_valid=...)` and pass them with `trend=`.
- **Different `time` or `harmonics` in `ar_trend` and `parts`.** `parts` recomputes the residuals for the correlation with its own arguments. Pass the same ones to both.
- **Mixed slope units.** With `time="step"` (the default), the slope is per time step: per year for annual data, per 16 days for 16-day data. Use `time="year"` for sub-annual series.
- **Water, urban and bare land in a vegetation trend.** They trend for other reasons (reservoirs filling, cities growing). Mask them, or make them classes of their own.
- **A smooth (Gaussian-like) correlation.** The exponential misfits it and PARTS can be badly anti-conservative. Check the correlation plot and try `correlation="exponential_power"`.
- **Tuning until significant.** Choosing the partition size, the seed or the range that gives p < 0.05 invalidates the test. Fix the settings first, then report the sensitivity.
- **Too many class comparisons.** Every pairwise test adds a chance of a false positive. Use the chi-square test first, then a few planned pairs.

## References

- Ives, A. R., Zhu, L., Wang, F., Zhu, J., Morrow, C. J., & Radeloff, V. C. (2021). Statistical inference for trends in spatiotemporal data. *Remote Sensing of Environment*, 266, 112678. [doi:10.1016/j.rse.2021.112678](https://doi.org/10.1016/j.rse.2021.112678)
- Morrow, C. J., & Ives, A. R. remotePARTS: Spatiotemporal Autoregression Analyses for Large Data Sets. R package, version 1.0.4. [CRAN](https://CRAN.R-project.org/package=remotePARTS)
- Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal variables. *Biometrika*, 48(3/4), 419–426. [doi:10.2307/2332763](https://doi.org/10.2307/2332763)
- Wilks, D. S. (2016). "The stippling shows statistically significant grid points": How research results are routinely overstated and overinterpreted, and what to do about it. *Bulletin of the American Meteorological Society*, 97(12), 2263–2273. [doi:10.1175/BAMS-D-15-00267.1](https://doi.org/10.1175/BAMS-D-15-00267.1)
