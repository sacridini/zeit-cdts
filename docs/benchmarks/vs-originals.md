# Against the Original Implementations

Twelve of Zeit's newer methods were checked against published implementations: emerging hot spots, recovery metrics, EWMACD, spectral unmixing, Bayesian smoothing of class probabilities, the area of applicability, BEAST, the AR(1) trends with their regional test (PARTS), the range of a map's autocorrelation, kNNDM's spatial folds and the error profile against their R originals, and S-CCD against pyxccd, its authors' Python package. This page times each port against that original on the same inputs, and records how the results were checked to agree. Every number comes from `benchmarks/results/vs-originals.json`, written by the scripts in `benchmarks/vs_originals/`, one per method.

---

## At a glance

<div class="bm-chart-card">
<p class="bm-chart-title">Time of the original and of Zeit, on the largest workload of each benchmark</p>
<p class="bm-chart-sub">Log scale. The originals are timed inside R (pyxccd inside Python), without loading R and its packages. Zeit is the best of three runs. Hover over (or tab to) a row for the exact numbers and the agreement.</p>
<div class="bm-chart-legend">
<span><i style="background:var(--bm-ref)"></i>Original</span>
<span><i style="background:var(--bm-gpu-lo)"></i>Zeit, 1 thread</span>
<span><i style="background:var(--bm-zeit)"></i>Zeit, 19 threads (8 for the AR(1) trends, PARTS and the spatial cross-validation)</span>
</div>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/vs_originals.svg"
</div>
</div>

On one thread, Zeit is 3× (the area of applicability with 8 predictors) to 30,000× (emerging hot spots) faster than the original, and the outputs compared agree to floating-point rounding (for the unmixing, to the tolerance at which the original's solver stops). Compiled loops against R's per-pixel ones account for most of the gain on one thread; the threads multiply it where there is enough work to share. BEAST is the exception: its original is already C, multithreaded and vectorized, and Zeit, a port of its generic kernels, is 1.5× to 1.9× faster on a cube and slower on one series.

---

## Results

Times of the call alone (see [Protocol](#protocol)); "1 thread" and "19 threads" (8 for the AR(1) trends, PARTS and the spatial cross-validation) are Zeit's, and the speedups are the original's time over Zeit's. The agreement is checked by the script on every run, which fails if it does not hold.

### Emerging hot spots vs sfdep

`zeit.emerging_hotspots` against the R package **sfdep 0.2.5**'s `emerging_hotspot_analysis` (queen neighbours, a time lag of 1), on a grid of cells with 12 dates of values. A square of cells gets a trend added over the dates.

| Workload | sfdep | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 20 × 20 cells, 199 permutations | 24.3 s | 1.2 ms | 1.8 ms | 20,883× / 13,308× | Gi* max diff 3.6e-15 |
| 50 × 50 cells, 199 permutations | 804 s (13.4 min) | 7.1 ms | 4.5 ms | 112,995× / 179,860× | Gi* max diff 1.8e-14 |
| 20 × 20 cells, 9 permutations | 1.39 s | 1.3 ms | 1.6 ms | 1,088× / 876× | Gi* max diff 3.6e-15 |
| 50 × 50 cells, 9 permutations | 28.4 s | 4.3 ms | 2.4 ms | 6,594× / 11,888× | Gi* max diff 1.8e-14 |
| 100 × 100 cells, 9 permutations | 520 s (8.7 min) | 17.1 ms | 6.0 ms | 30,481× / 86,056× | Gi* max diff 2.0e-14 |

**How they were checked.** The Getis-Ord Gi* of every cell and date, which sfdep returns with `include_gi = TRUE`, is compared with Zeit's (`zeit._core.hotspots.gi_star`, the same as `emerging_hotspots` computes); the script fails if they differ by more than 1e-10.

!!! warning "Not the same work"
    sfdep takes the significance of each Gi* from permutations (199 by default), and that is most of its time; Zeit takes it from the normal distribution of the Gi*, with no permutations. The two therefore classify some cells differently (sfdep also has rules of its own for the patterns; see [`emerging_hotspots`](../api/time-series.md#emerging_hotspots)), and the speedups compare two ways of answering the question, not one algorithm in two languages. With 9 permutations, sfdep's time is mostly its own. On the smallest grid, 19 threads are slower than one: there is too little work to share.

### Recovery metrics vs RecoverMetricsR

`zeit.recovery_metrics` against the R package **RecoverMetricsR 0.1.0** (GPL-3; run as a black box): `compute_recovery_metrics` (RRI, R80P, YrYr; pre-window 2, post-window 5) and `compute_recovery_time` (years to 80% of the pre-disturbance mean), on an annual series of 21 years (2000–2020), every pixel disturbed in 2010 and regrowing steadily.

| Workload | RecoverMetricsR | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 200 × 200 px × 21 years | 1.81 s | 7.5 ms | 6.1 ms | 243× / 298× | max rel. diff 5.9e-8 |
| 1000 × 1000 px × 21 years | 52.4 s | 98 ms | 61 ms | 535× / 861× | max rel. diff 6.0e-8 |

**How they were checked.** The four layers (RRI, R80P, YrYr, years to recovery) of every pixel, with the same pixels missing in both, to a relative difference of 1e-6. RecoverMetricsR takes the best of the 5 years after the trough where Zeit (Frazier et al. 2018) takes years 4 and 5, and counts the years to recovery from the year after the trough: on steady regrowths these give the same values, so the series are built that way. The remaining difference is float32's precision, Zeit's output type. The time of RecoverMetricsR is of both calls together.

### EWMACD vs Brooks' script

Zeit's EWMACD against **Brooks' EWMACD v1.3.0 script** (Brooks et al. 2014; `EWMACD v1.3.0.r`, published without a licence: the benchmark runs it from a path you give and never copies it), its per-pixel function `EWMACD.pixel.for.rasterEngine` applied to every pixel, as its `rasterEngine` does. Irregular dates over 2005–2012, seasonal pixels with noise, clouds and, in a third of them, a disturbance; 2 harmonics, lambda 0.3, persistence 1, training 2005–2008.

| Workload | Brooks' script | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 1,000 px × 75 dates | 0.30 s | 4.8 ms | 0.84 ms | 62× / 356× | the same flags |
| 10,000 px × 71 dates | 2.75 s | 55 ms | 6.6 ms | 50× / 417× | the same flags |

**How they were checked.** The flags of every pixel and date are identical (the script fails on any difference). The Zeit side is the Rust core that `zeit.ewmacd` runs (`_core.ewmacd.ewmacd_batch`), timed without the wrapper's handling of the cube. With a persistence above 1 the two differ on a few dates after the first disturbance (see [`ewmacd`](../api/change-detection.md#ewmacd)), so the benchmark uses 1. The two workloads have 75 and 71 dates because each draws its own irregular dates.

### S-CCD vs pyxccd

[`zeit.sccd`](../api/change-detection.md#sccd) against **pyxccd 1.1.0**'s `sccd_detect`, the implementation of S-CCD's authors (Ye et al. 2021; Apache-2.0; Zeit's S-CCD is written from the paper, and pyxccd is run as a black box in an environment of its own). pyxccd is called pixel by pixel, as its API does. Two workloads: 4096 pixels of a Landsat Collection 2 stack of Rondonia (2020–2026, the 61 dates of the path these pixels are on), and 1000 synthetic pixels of 35 years every 8 days, a third of the dates cloudy and a third of the pixels with a change: the long series S-CCD was made for. The same pixels also go through Zeit's own [`ccdc`](../api/change-detection.md#ccdc), to time S-CCD's claim, a Kalman update per observation instead of a lasso refit.

| Workload | pyxccd | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 4096 px × 61 dates (Rondonia) | 10.1 s | 1.10 s | 0.196 s | 9× / 52× | same number of breaks 85%; pyxccd's breaks within 1 obs. 76% |
| 1000 px × 1598 dates (synthetic) | 22.0 s | 1.16 s | 0.142 s | 19× / 155× | same number of breaks 100%; breaks within 1 obs. 100% |

| Workload | Zeit's CCDC, 1 thread | Zeit's S-CCD, 1 thread | CCDC, 19 threads | S-CCD, 19 threads | S-CCD's speedup (1 / 19) |
|:---|---:|---:|---:|---:|:---:|
| 4096 px × 61 dates (Rondonia) | 1.50 s | 1.10 s | 0.208 s | 0.196 s | 1.4× / 1.1× |
| 1000 px × 1598 dates (synthetic) | 12.1 s | 1.16 s | 1.43 s | 0.142 s | 10× / 10× |

**How they were checked.** The script compares the break dates of every pixel: the share of pixels with as many breaks in both, and the share of pyxccd's breaks with one of Zeit's within one clear observation. On the synthetic changes the two agree on every break; on Rondonia, where many statistics are near the threshold, they agree on most, for reasons given in [Fidelity](fidelity.md#9-newer-ports-against-their-originals). The Zeit side is the Rust core that `zeit.sccd` runs (`_core.sccd.sccd_batch`). On the short Rondonia series most of the work is the initialisation that S-CCD shares with CCDC (the lasso fits of the stability test), so S-CCD is only a little faster; on 35 years of observations the Kalman updates replace CCDC's refits and S-CCD is ten times faster, more than the paper's 3–5×. These timings were taken while other builds ran on the machine: over three runs, pyxccd took 5.5–10.1 s on Rondonia and 15.8–22.0 s on the synthetic series, and Zeit varied by up to 2× too, so the speedups are good to a factor of about 2. The Rondonia stack interleaves two Landsat paths; the dates of the other path, with no data in these pixels, are left out, because CCDC refuses a pixel with data on less than half of the dates (S-CCD and pyxccd do not).

### Spectral unmixing vs RStoolbox

Zeit's unmixing against the R package **RStoolbox 1.0.2.2**'s `mesma` with its NNLS solver (`method = "NNLS"`, `sum_to_one = FALSE`), 4 endmembers (vegetation, soil, water, non-photosynthetic vegetation) and 6 bands, on pixels mixed from them with Dirichlet fractions and noise.

| Workload | RStoolbox | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 200 × 200 px | 0.84 s | 47 ms | 5.4 ms | 18× / 157× | fractions max diff 3.7e-4; RMSE never larger |
| 1000 × 1000 px | 21.8 s | 1.25 s | 132 ms | 17× / 165× | fractions max diff 4.0e-4; RMSE never larger |

**How they were checked.** Both solve the same problem: non-negative fractions, free to sum to anything (mesma's `sum_to_one` only divides afterwards). mesma's NNLS stops at a tolerance, while Zeit's active set solves each pixel exactly, so the fractions agree to 1e-3 (a median of 7e-5) and Zeit's residual is never larger than mesma's (smaller by up to 5e-7); the script checks both. The Zeit side is the Rust core of [`zeit.unmix`](../api/preprocessing.md#unmix) (`_core.sma.unmix_batch`) with the same constraints, not `zeit.unmix`'s default fully constrained unmixing.

### Bayesian smoothing vs sits

`zeit.smooth_probs` is written from the paper (Camara et al. 2024, Remote Sensing 16(23):4572), no longer from the code of the R package **sits**. It agrees with bayesEO's `bayes_smooth` to about 2e-3, with the same class everywhere in the tests; the timings against sits and the agreement with its `sits_smooth` that were measured for the earlier implementation no longer apply, and are being measured again.

### Area of applicability vs CAST

`zeit.train_di` and `zeit.aoa` against the R package **CAST 1.0.4**'s `trainDI` and then `aoa` with it (with and without the local point density), on training points in 6 clusters (8 predictors, or 64, as the series of a deep learning model) and new points spread wider.

| Workload | CAST | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 500 training × 10,000 new | 0.09 s | 12.5 ms | 1.8 ms | 7× / 50× | DI max diff 2.4e-15 |
| 500 × 10,000, with LPD | 0.84 s | 15.4 ms | 3.1 ms | 54× / 269× | DI max diff 2.4e-15 |
| 500 × 100,000 | 0.31 s | 105 ms | 10.8 ms | 3× / 29× | DI max diff 4.0e-15 |
| 2000 × 100,000 | 1.21 s | 447 ms | 41 ms | 3× / 30× | DI max diff 1.6e-15 |
| 2000 × 100,000, 64 predictors | 5.77 s | 990 ms | 319 ms | 6× / 18× | DI max diff 1.8e-15 |

**How they were checked.** The DI of every new point, to 1e-9 (the largest difference was below 1e-14). Both sides time the training's dissimilarity and the new points' together. CAST finds the nearest training points with FNN's k-d tree; Zeit uses a k-d tree up to 9 predictors and a matrix product beyond, the distances exact either way. With 8 predictors both search a tree, and the gain on one thread is smallest (3×); most of Zeit's advantage there is its threads.

### BEAST vs Rbeast

`zeit.beast` against the R package **Rbeast 1.0.2** (GPL ≥ 2), whose C code it ports: `beast()` on one series, and `beast123()`, its batch with threads of its own, on a cube. The series are monthly, with a season of period 12, noise, 5% missing values, and changes of level and of slope. Both use `beast()`'s defaults, from seed 42: on one series, 8,000 samples and 3 chains; on the cube, 2,000 samples and 2 chains. Rbeast runs its AVX2 kernels, the fastest on this processor; Zeit runs the port of its generic kernels.

| Workload | Rbeast | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 1 series × 360 months | 0.13 s | 0.19 s | | 0.7× | the same chain as Rbeast's generic kernels |
| 10,000 px × 144 months | 328 s on 1 thread, 34.5 s on 19 | 217 s | 18.1 s | 1.5× / 1.9× | the same chain as Rbeast's generic kernels |

The cube's speedups compare each Zeit time with Rbeast on as many threads: Rbeast's `beast123()` is multithreaded too. On one series Zeit is slower than Rbeast's AVX2 kernels and as fast as its generic ones (0.19–0.21 s): it runs the same scalar arithmetic. On a cube it is faster, because `beast123()` costs Rbeast more per pixel than `beast()` does (33 ms against 21 ms here, on one thread).

**How they were checked.** With Rbeast's generic kernels (`cputype = "sse"`), Rbeast and Zeit run the same chain. The script checks that the trend and the number of changepoints of the series and of the cube's first pixel are identical to the last bit, and the tests compare every output on six series (see [Fidelity](fidelity.md)). The timed AVX2 runs sum in another order, so their chains drift apart, and the outputs agree only in distribution:

- the trends differ by a median of 2e-3 standard deviations of the series;
- Rbeast's mean number of trend changepoints is 0.31 higher on the cube.

Most of that gap is Rbeast's own. Zeit fits each pixel as `beast()` fits one series, from the seed, and its mean equals `beast()`'s exactly. `beast123()` carries its random stream from one pixel to the next and finds more changepoints: 0.06 more on average over 200 of these pixels, with the same generic kernels. Rbeast's 19-thread run also left one pixel unfitted (NaN).

!!! note "Windows' power throttling"
    Windows 11 slows down the processes it considers to be in the background (power throttling, or EcoQoS): after a few seconds of work, it moves them to the efficiency cores of a hybrid processor and lowers their clock. A benchmark run from a terminal is such a process. Zeit's unpinned threads ran 2.4 times slower for it. Rbeast's threads, which it pins to processors, were slowed less. The script takes both processes out of power throttling, so the times are of the computation. Zeit's one-thread runs are pinned to the first processor, as Rbeast pins its own.

### AR(1) trends and PARTS vs remotePARTS

`zeit.ar_trend` and `zeit.parts` against the R package **remotePARTS 1.0.4**: `fitAR_map`, the trend of every pixel's series with AR(1) errors, and `fitGLS_partition`, the test of the region's mean trend by generalized least squares on partitions of the pixels, with an exponential spatial correlation (range 8 pixels), its nugget estimated in each partition, and 6 pairs of partitions crossed. The series are annual, 30 years, with a slope that changes across the map and AR(1) errors whose innovations are spatially correlated. Zeit ran these on 1 thread and on 8 of the 20 logical processors, not on 19.

| Workload | fitAR_map | Zeit, 1 thread | Zeit, 8 threads | Speedup (1 / 8) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 50 × 50 px × 30 years | 4.09 s | 7.2 ms | 4.1 ms | 568× / 1,007× | slope max diff 5.3e-8 of the largest, ρ 1.0e-6 |
| 100 × 100 px × 30 years | 16.6 s | 20.2 ms | 8.5 ms | 821× / 1,960× | slope 6.9e-8, ρ 3.4e-6 |
| 200 × 200 px × 30 years | 66.8 s | 71.8 ms | 23.1 ms | 931× / 2,888× | slope 1.1e-7, ρ 7.7e-6 |

| Workload | fitGLS_partition | Zeit, 1 thread | Zeit, 8 threads | Speedup (1 / 8) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 2,500 px in 5 partitions of 500 | 2.47 s | 0.120 s | 0.030 s | 21× / 84× | partitions max rel. diff 2.7e-11; overall t 0.85% apart |
| 10,000 px in 6 partitions of 1500 | 72.5 s | 3.61 s | 1.15 s | 20× / 63× | partitions 1.6e-10; overall t 0.97% apart |
| 40,000 px in 26 partitions of 1500 | 262 s (4.4 min) | 12.8 s | 3.51 s | 20× / 75× | partitions 4.7e-15; overall t 0.028% apart |

Each partition is a GLS on a dense 1500 × 1500 correlation matrix, so the cost is the same linear algebra on both sides; Zeit's gain on one thread (20×) is smaller than on the per-pixel AR fits, and its threads share the partitions.

**How they were checked.** The AR(1) fits agree to the tolerance of the two optimizers (the script requires 1e-3 of the largest slope and 1e-2 on ρ; the largest differences were 1.1e-7 and 7.7e-6). The GLS is checked on remotePARTS' own slopes and partitions, read back from R (Zeit's timed runs draw their own partitions of the same size): every partition's estimate, standard error and nugget, and the overall estimate, must agree to 1e-6 (the largest difference was 1.6e-10). The overall t statistic is 0.03% to 1% apart because remotePARTS 1.0.4 computes the correlation between pairs of partitions after the first without their nugget, and Zeit with it (see [Fidelity](fidelity.md)).

### Spatial cross-validation vs blockCV and CAST

[`zeit.autocorrelation_range`](../api/post-processing.md#autocorrelation_range) against the R package **blockCV 4.0.1**'s `cv_spatial_autocor` (5000 pixels, which fits its variograms with automap); [`zeit.spatial_folds`](../api/post-processing.md#spatial_folds) with `method="knndm"` against **CAST 1.0.4**'s `knndm` (k = 5, its example's clustered points in a square of 100, 1000 regular prediction points); [`zeit.error_profile`](../api/post-processing.md#error_profile) against CAST's `errorProfiles` (with **scam 1.2.23**) of a linear model cross-validated in 5 folds by caret; and the Mahalanobis distance of `train_di` and `aoa` against CAST's `trainDI` and `aoa` with `method = "MD"`. The maps are white noise of 2 bands averaged over windows of 3, 7 and 15 pixels of 100 m, so correlated up to 300, 700 and 1500 m. Zeit ran on 1 thread and on 8.

| Workload | Original | Zeit, 1 thread | Zeit, 8 threads | Speedup (1 / 8) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| `autocorrelation_range`, 300 × 300 px, range 300 m | 2.74 s | 287 ms | 50 ms | 10× / 55× | range 0.89–0.99 × blockCV's |
| `autocorrelation_range`, 300 × 300 px, range 700 m | 1.87 s | 282 ms | 49 ms | 7× / 38× | range 1.00–1.02 × blockCV's |
| `autocorrelation_range`, 300 × 300 px, range 1500 m | 1.63 s | 278 ms | 50 ms | 6× / 33× | range 0.98–1.01 × blockCV's |
| kNNDM, 1000 clustered points | 1.81 s | 80 ms | 73 ms | 23× / 25× | W 0.572 (CAST's folds: 0.685) |
| kNNDM, 5000 clustered points | 40.2 s | 565 ms | 529 ms | 71× / 76× | W 0.267 (CAST's folds: 0.269) |
| `error_profile`, 2000 points, 5 folds | 0.45 s | 22.7 ms | 10.2 ms | 20× / 44× | windows max diff 5.3e-15; curve within 1.2% |
| Mahalanobis DI, 500 training × 10,000 new | 20.8 s | 15.3 ms | 3.0 ms | 1,359× / 6,867× | DI max diff 3.0e-14 |
| Mahalanobis DI, 2000 × 100,000 | 909 s (15.1 min) | 437 ms | 68 ms | 2,082× / 13,294× | DI max diff 3.3e-13 |

**How they were checked.**

- **The range.** blockCV and Zeit each draw their own 5000 pixels, so the ranges agree only to the sample: within 2% at 700 and 1500 m, and 1–11% shorter at 300 m, three pixels. The bands are automap's on both sides (2% to 100% of 0.35 of the map's diagonal), and the model is the one that fits best of the spherical, exponential and Gaussian (automap also tries Stein's Matérn). blockCV chose the spherical on every map, Zeit the Gaussian at 300 m. With an exponential or a Gaussian, blockCV reports gstat's range parameter, Zeit's `scale`, where Zeit's `range` is the practical range.
- **kNNDM.** The two draw different folds (each side clusters its own candidates), so the folds are not compared one by one. The script measures CAST's own folds with Zeit's W, which must equal CAST's to 1e-9, and records the W of each side's folds: smaller is better, closer to the distances from the area to predict. 8 threads gain little: they share only the core's distances, not the clustering of the candidates.
- **The error profile.** The windows of the DI and their RMSE, from CAST's predictions and variable weights, must equal CAST's to 1e-9 (the largest difference was 5.3e-15). The monotone curve is Zeit's own P-spline, not scam's: within 1.2% of the range of the metric, on 200 DI values.
- **The Mahalanobis distance.** The DI of every new point, to 1e-9. CAST takes 750 times as long with the Mahalanobis distance as with the Euclidean on the same 2000 × 100,000 points (909 s against 1.2 s); Zeit whitens the predictors once and searches the same k-d tree as for the Euclidean distance.

The kNNDM rows are the best of two runs, not three.

### Areas of transitions vs survey

[`zeit.transitions`](../api/validation.md#transitions) against R's **survey 4.5** (Lumley), the reference for design-based estimators, with **terra** for the mapped areas. The landscape is made up: 2048 × 2048 pixels of 30 m with Forest, Pasture, Crop and Water at two dates; patches are cleared, cropped and regrown between the dates. Two maps of the dates have errors: 6% of the classes are wrong, 80% of the changes are found and 1.5% of the pixels are false change. The strata are not the transitions: they come from an older alert map (its alerts, a buffer of 2 pixels and the rest), with 250 + 200 + 150 points whose reference is the truth.

The R script counts the transitions inside the strata with terra's `freq`. It then builds `svydesign(ids = ~1, strata = ~h, weights = ~A_h / n_h)` and estimates each transition's area with:

- `svytotal` (direct);
- the mapped area plus `svytotal` of y − x (difference);
- `svyratio(~y, ~x)` times the mapped area (ratio).

The user's and producer's accuracies come from `svyratio` and the overall accuracy from `svymean`. Its timing covers reading the three rasters, so both sides read them from files.

| Workload | survey + terra | Zeit | Speedup | Agreement |
|:---|---:|---:|:---:|:---|
| direct, 2048 × 2048 px, 600 points, 16 transitions | 0.48 s | 0.18 s | 2.7× | max diff 5.7e-15 |
| difference | 0.40 s | 0.21 s | 1.9× | max diff 7.6e-15 |
| ratio | 0.41 s | 0.19 s | 2.2× | max diff 5.7e-15 |

**How they were checked.** For every transition, the script compares the mapped area, the estimate and its standard error, and the user's and producer's accuracies with their standard errors. It also compares the overall accuracy. Each must agree with survey's to 1e-9 of the value (or of 1 for values below 1). A value survey leaves undefined (NA) must be NaN in Zeit, for example the ratio estimate of a transition no point is mapped as. The fixtures of `tests/test_accuracy_survey.py` (`tests/data/make_survey_fixtures.R`) hold the same estimators to 1e-10 on a smaller landscape.

Most of the time on both sides is reading the rasters and counting the pairs, not the estimates. A map of integer classes of up to 2^26 pixels is read once into memory, and Zeit counts its classes, the pairs and the pixels of each stratum per row with a few `bincount`s; larger maps stay lazy and are counted through dask. Before this, Zeit read each raster several times through dask and took 0.73–0.85 s, 0.6× survey's speed. The Janus benchmark of the same estimators (its `tools/bench_validation_vs_survey.py`) has the count in Rust.

### FORCE (TSA)

The time series analysis ported from **FORCE 3.11.00** (Frantz 2019; GPL-3.0-or-later): [`temporal_metrics`](../api/time-series.md#temporal_metrics), [`standardize`](../api/preprocessing.md#standardize), [`interpolate`](../api/preprocessing.md#interpolate), [`linear_trend`](../api/time-series.md#linear_trend), [`cat`](../api/time-series.md#cat), [`clear_observations`](../api/time-series.md#clear_observations) and [`polar_metrics`](../api/time-series.md#polar_metrics), against FORCE's own outputs on the same synthetic cube: 10,000 pixels, 325 dates over 2018-2022 interpolated to 114 dates (the script is `benchmarks/vs_originals/force_tsa.py`, which compares the outputs and reports every difference it cannot explain). Zeit's floats are truncated toward zero to int16 before comparing, as FORCE writes its outputs. "Equal" is the share of values identical after that, "within 1" the share that differ by at most 1.

| Product | Equal | Within 1 | Left out, and why |
|:---|---:|---:|:---|
| STM, 16 metrics (MIN, Q01-Q99, MAX, AVG, STD, RNG, IQR, SKW, KRT, NUM) | 100% (Q90: 98.97%) | 100% | nothing |
| Folds, 10 statistics × year, quarter, month, week and day (5, 4, 12, 52 and 365 bands) | 100% (KRT by week: 99.70%) | 100% | STD, SKW and KRT by day: 1,140,000 values each, a day with one or no observation (undefined in FORCE); SKW and KRT by week: 552 values of a constant series (NaN in Zeit) |
| Standardization: normalize of the interpolated series (114 bands), center of the yearly, quarterly and monthly folds | 100% | 100% | nothing |
| Interpolation, linear | 100% | 100% | nothing |
| Interpolation, moving average | 100% | 100% | nothing |
| Interpolation, RBF | 99.984% | 100% | nothing |
| Interpolation, harmonic fit to all the series | 99.999% | 100% | the pixels whose IRLS ran its 100 iterations without converging (98 of the 10,000, found by the core, none beyond 1) |
| Interpolation, harmonic fit to part of the series, and its near-real-time dates (120 bands) | 100% | 100% | the pixels whose IRLS does not converge (46 of the 10,000; one of them has values beyond 1, up to 6): it cycles and stops where the iteration cap, the rounding and the order of the sums happen to put it, which differs among FORCE, GSL and Zeit |
| TRY and TRM, the 12 trend variables of the yearly and monthly folds | 100% | 100% | nothing; `sig` matches although Zeit uses the exact t quantile |
| CAY, without the penalty (39 variables) | 99.995% | 100% | nothing; 21 values of `loss` differ by 1 (FORCE truncates the offset before dividing) |
| CAY, with the penalty | 99.995% | 100% | 58,149 values of the 1,491 pixels (15%) where the corrected comparison picks another step; `loss`: 18 values off by 1 |
| CAM, without the penalty | 99.980% | 100% | nothing; 78 values of `loss` off by 1 |
| CAM, with the penalty | 99.980% | 100% | 271,635 values of the 6,965 pixels (70%) where the corrected comparison picks another step; `loss`: 24 values off by 1 |
| POL, the 39 polar metrics × 5 years (RBF TSI, adaptive) | 100% | 100% | nothing; the year without a season is nodata in both |
| POL, not adaptive | 100% | 100% | `DPV`, 50,000 values (FORCE writes `-DPY`, Zeit 0) |
| POL, over a moving-average TSI with nodata (`MOVING_MAX` 8: 0.91% of the values, 71% of the pixels have a gap) | 100% | 100% | nothing |
| POL, thresholds 0.1, 0.4 and 0.9 | 100% | 100% | nothing |
| POL, a block of 100 pixels of zero index (the season's total is 0) | 100% | 100% | 15,600 values (400 per metric): `NaN` in Zeit, 0 in FORCE (an undefined float cast to int16) |
| POL, a series from 10 February (`DATE_RANGE` from the middle of the first year), where FORCE finds seasons | 100% | 100% | 26,016 values per metric (of 50,000) where FORCE finds no season (it writes 0): Zeit measures 19,777 of them, as the series covers them, and has `NaN` in the other 6,239; none where FORCE has a season and Zeit lacks it |
| TRO of each of the 39 metrics (12 variables) | 100% (the 9 date metrics: 90.6-93.1%) | 100% | R-squared of constant series (2,810-4,336 values by run); the date metrics are checked below |
| CAO of each of the 39 metrics (39 variables) | 100% of what is compared | 100% | the 16 variables of the 9 date metrics that FORCE's `in_ce` alters (1,440,000 values) and the R-squared of constant parts (6,247-9,680 values by run) |
| CSO m1 (59 monthly windows), NUM, MIN, MAX, RNG, AVG | 100% | 100% | the last window (Zeit makes 60) |
| CSO m3 (20 quarterly windows), NUM, MIN, MAX, RNG, AVG | 100% | 100% | none |
| CSO m1 and m3, IQR, STD, SKW, KRT, Q25, Q50, Q75 | see below | | by design: FORCE's statistics mix `n` observations with `k` intervals |

**How they were checked.** For the metrics, folds, interpolation and standardization the comparison is value by value against FORCE's rasters. The two products that differ by design are checked in a second way. With the penalty, the pixels where Zeit's corrected comparison picks another change step than FORCE are all reproduced by a literal Python port of FORCE's comparison (penalized against the best one's unpenalized drop, in float32), and every other value is equal or within 1 as above; the values of those pixels are the ones counted as left out in the table. For CSO, the statistics that depend on `n` (STD, SKW, KRT and the quantiles) differ from FORCE's by design: on 1-month windows Zeit's STD equals FORCE's in 68.9% of the values, SKW in 29.2%, KRT in 21.1%, Q25 in 57.6%, Q50 in 86.2% and Q75 in 75.8%. A literal Python port of FORCE's n-for-k code reproduces FORCE's STD, quantiles and IQR exactly, and its SKW and KRT within 1 (97.9% equal for KRT on 1-month windows; 99.7% and 98.5% for SKW and KRT on 3-month ones), which is the check that the difference is the definition, not an error.

**The polar metrics.** The five runs are over FORCE's own TSI (the RBF's, or the moving average's, which has nodata), with the 39 metrics written, so the input is exactly FORCE's. The dates differ by design (Zeit counts from 1 January of each season's year, FORCE in continuous days: the comparison adds 365 per year), and FORCE's TRO and CAO of the dates treat them in continuous days (`in_ce`: the slope is reduced by 365 and scaled by 1000 before the test and the offset): for those, Zeit's trend is within 1 of FORCE's after the same conversion, and a literal Python port of FORCE's code on FORCE's own POL outputs reproduces its TRO and CAO exactly (all 39 metrics, every run). Where FORCE leaves a value undefined and writes 0 (a season of total 0), Zeit has `NaN`; those pixels are left out of the comparison of their trends. The run without `adaptive` has `DPV` left out for the reason above. A series that does not start on 1 January is a difference by design: FORCE looks for the first start of the year in a window after the long-term start in year 0 and, if the first step is past it, never finds it nor any later one (the start is only advanced when one is found), so every metric of the pixel is `NaN` cast to 0; Zeit skips a start with no step in its window (one day wider, as the days are truncated), numbers each season by the year it starts in and measures the seasons whose start is in the series. Three more runs (`DATE_RANGE` from 10 February, 1 June and 1 December 2018, `force_tsa.sh` `ONLY="rbf_polar_feb rbf_polar_jun rbf_polar_dec"`) show it: FORCE has no season in 52% (26,016 of 50,000 pixel-years), 79% (39,632) and 72% (35,840) of the pixel-years; where it has one, all 39 metrics are equal in every value, and Zeit lacks none of them; where it has none Zeit measures the seasons the series covers. `benchmarks/vs_originals/force_tsa.py --part polar` (the zero block needs `force_tsa.sh` with `ZERO=10`, a cube of its own).

**Timings.** On a larger cube, 500 × 500 pixels (250,000) with the same 325 dates over 2018-2022, FORCE's compute stage against Zeit's core (`zeit._core.tsa.*_batch`, on FORCE's own series), with 1 and with 6 threads. Seconds; the gain is FORCE's time over Zeit's (below 1: FORCE is faster).

| Product | FORCE, 1 thread | FORCE, 6 threads | Zeit, 1 thread | Zeit, 6 threads | Gain, 1 thread | Gain, 6 threads |
|:---|---:|---:|---:|---:|---:|---:|
| Interpolation, linear (114 dates) | 0.11 | 0.01 | 0.089 | 0.027 | 1.2× | 0.5× |
| Interpolation, moving average | 0.06 | 0.05 | 0.140 | 0.037 | 0.4× | 1.3× |
| Interpolation, RBF | 1.75 | 0.50 | 0.89 | 0.18 | 2.0× | 2.7× |
| Interpolation, harmonic fit to all the series | 227.8 | 49.0 | 22.1 | 5.36 | 10× | 9.2× |
| STM, 16 metrics | 2.22 | 0.48 | 0.53 | 0.11 | 4.2× | 4.3× |
| Folds, AVG by year, quarter, month, week and day | 4.50 | 1.29 | 0.24 | 0.090 | 18× | 14× |
| TRY and CAY, no penalty (Zeit: yearly fold, trend and CAT) | 0.21 | 0.11 | 0.067 | 0.023 | 3.1× | 4.8× |
| CSO, 1-month windows (59) | 2.58 | 0.64 | 0.87 | 0.21 | 2.9× | 3.1× |
| Polar metrics, 6 (VSS VPS VES VSA RMR IGS), adaptive | 1.09 | 0.45 | 0.58 | 0.136 | 1.9× | 3.3× |
| Polar metrics, all 39, adaptive | 0.87 | 0.53 | 0.61 | 0.143 | 1.4× | 3.7× |

**How the FORCE time was isolated.** FORCE reads, computes and writes in separate threads and times the three stages itself (`progress-hl.c`, the "Virtual C-time" of its summary), but in whole seconds, too coarse for a stage of a second; the benchmark preloads a four-line library (`force_time_ms.c`) that makes `time()` count milliseconds, and FORCE's binary and code are untouched. What is timed is the compute stage alone (QAI screening, the NDVI and the analysis, no reading and no writing), the minimum of 10 runs with 6 threads, 5 with 1 (3 for the harmonic fit), one chunk, `NTHREAD_READ = NTHREAD_WRITE = 1`. Each configuration has the baseline's stage taken off, since every run also screens the QAI and computes the index (a run with no interpolation that writes the same TSS): the interpolations less the baseline's, the STM, the folds and TRY/CAY less the RBF run's (which has the baseline in it). The CSO has no baseline to take off: its input is the QAI alone and the screening is part of its stage (FORCE's own run with only NUM takes 1.61 s with 1 thread and 0.37 s with 6, the floor of its interval and moment loop). The Zeit side is the best of three after a warm-up, from the arrays already in memory (FORCE's int16 as float32, which the core reads as it is and works on in float64; the first measurements fed it a float64 copy, twice the memory traffic), on the same machine, one after the other, never together. FORCE computes the yearly fold inside the TRY/CAY run, so Zeit's time for that row includes it (the whole row: 0.067 s with 1 thread).

Where FORCE is still faster, the linear interpolation with 6 threads and the moving average with 1: FORCE's time there is the difference of two stages of 0.4-1.5 s whose runs vary by 0.1-0.2 s, so 0.01-0.2 s is its noise level and the gains of these rows are indicative only (FORCE's linear interpolation is at most about 0.1 s). What bounds Zeit there is memory, not the kernel: the output alone is 250,000 × 114 float64 values (228 MB, first touched, and so faulted in, by the threads), against FORCE's int16 (57 MB), and the input is 325 MB of float32 against FORCE's 162 MB of int16; with 6 threads Zeit's linear interpolation moves some 550 MB in 0.027 s, about 20 GB/s. What was done: the per-pixel pass is a single branch-free compaction that counts the valid steps as it goes (instead of a compaction and 114 bisections per pixel: from 2.4-3.2 µs per pixel with 1 thread to 0.36 for the linear interpolation), the moving average takes its sums from integer prefix sums when the series is integer-valued (as FORCE's are; any other series is added in order, as before), the output is not filled with NaN first, and the yearly fold runs four pixels at once, since the recurrence of a mean is a chain of a subtraction, a division and an addition (about 20 cycles a value) that independent series overlap. All results are bit for bit those of before (tested against the previous algorithms).

The polar rows are FORCE's POL stage over the RBF's (the run with the metrics written, less the RBF run's): 0.4-1.1 s from the difference of two stages of 1-4 s whose runs vary by 0.1-0.3 s, so FORCE's 1-thread time for the 39 metrics comes out below that of the 6 (noise, not a cost) and the gains of these two rows are indicative. Zeit's `polar_batch` takes FORCE's TSI as float32, 250,000 pixels × 114 dates; the 6 metrics cost nearly as much as the 39 (0.58 s against 0.61 s with 1 thread), so most of the time is the pass over the series that every metric shares.

The accuracy holds on this cube (`force_tsa.py --part stm,interp --ref <bench folder>`): STM 100% within 1 (Q90 98.98% equal, the rest equal), linear and moving interpolation 100% equal, RBF 99.983% equal and 100% within 1. The harmonic fit: 301 of 28.5 million values differ by more than 1 (at most 11), all in 13 of the 250,000 pixels (0.005%), and all 13 are pixels whose iteratively reweighted fit does not converge: the core reports the iterations (`zeit._core.tsa.harmonic_iterations`: 65535 for a fit that ran its 100 without settling), 2,219 pixels (0.9%) are like that, and the 13 are among them; the converged ones (99.1%) are all within 1 of FORCE, which is what `force_tsa.py` checks, counting the others instead of listing them. Such a series cycles between a few predictions and stops where the cap, the rounding and the order of the sums put it, so its value is not reproducible between FORCE, GSL and Zeit.

**Against the method's own code.** FORCE's polar metrics extend the **PolarMetrics** R package of Brooks, Lee, Hargrove and Pomara (0.1.0, GPL (>= 3), installed from GitHub). `benchmarks/vs_originals/polar.py` runs its `calc_metrics` and Zeit's core (`adaptive=False`) on 5,000 synthetic pixels, a sample every 5 days over 5 years (the package needs a whole number of samples per cycle; 30% of the pixels peak near the new year), and compares the eight metrics they share: `es`/DSS, `ls`/DES, `s_intv`/LGS, `ms`/DAV, `s_avg`/VGM, `s_sd`/VGV, `s_mag`/VAV and `a_avg`/IBT. A literal Python port of `calc_metrics` reproduces the package (the dates identical, the values to 1e-9) and isolates the three places where the definitions differ:

1. **The start of the year.** The package starts at the sample `ceiling(av_ang * spc / 2 pi)`; FORCE, so Zeit, at the first sample on or after the angle's day. They pick the same sample in 40% of the pixels and the next one in the rest.
2. **The steps it reads.** The package dates the season on the rotated year, but reads `v`, `VX` and `VY` at the same positions of the unrotated series (`v[es2ls]`, `calc_metrics.R` lines 165 and 290-295). Its `s_avg`, `s_sd`, `s_mag`, `a_avg` and the angle behind `ms` are therefore of other steps than the season it dates. This is the behavior of 0.1.0's code as run, not confirmed with the authors.
3. **The dates.** The package gives the position's day plus the angle's day, up to a step (4 days here) from the sample's own.

Where the year starts agree, `es` and `ls` are within 4 days of Zeit's (2 observed) and `s_intv` within 1; the values differ, because of (2). With the port reading the rotated steps from Zeit's start, all eight agree on all 20,000 pixel-years: `ms` equal, the values within 2e-7, and `es`, `ls` and `s_intv` within 1 day (95%, 98% and 93% equal; FORCE truncates a day from a float32 angle). On 5,000 pixels × 365 dates the package takes 5.0 s (one thread); Zeit's core 0.032 s with 1 thread (156×) and 0.0069 s with 8 (730×).

### FORCE (spectral adjustment)

[`spectral_adjust`](../api/preprocessing.md#spectral_adjust) takes Landsat and MODIS reflectance to the ten bands of Sentinel-2A (the red edge and the broad NIR predicted), as **FORCE 3.11.00**'s `SPECTRAL_ADJUST` does (Scheffler, Frantz and Segl 2020; FORCE's tables, GPL-3.0-or-later). It is compared with FORCE's own TSA on synthetic Landsat cubes of 200 x 200 pixels and 136 dates over 2018-2019 (`SENSORS = LND08`, `TARGET_SENSOR = SEN2A`, `SPECTRAL_ADJUST = TRUE`, each of the ten bands as an index, no interpolation, no noise filter; 4,937,286 of the 5,440,000 observations pass FORCE's QAI screening). FORCE writes a band used as an index as it is, in int16 (`index_band` in `index-compute-hl.c` copies `ard.dat`, without the 10,000 factor of the indices), so the comparison is of the integer outputs: Zeit's floats are truncated toward zero and wrapped to 16 bits, as FORCE's `(short)` does. The spectra are mixtures of green and dry vegetation, soil, water, snow and urban endmembers (Dirichlet abundances, a seasonal vegetation share, a brightness factor, noise), and the first row of every date holds edge cases: all zero, small negatives, high values, one band high, tiny values, flat grey, near-saturated (28,000-32,000), a negative band and pure endmembers. The three sensors have the same spectra and each its own table.

Two runs of the core against FORCE per sensor. In the first (`force_search=True`) the core makes FORCE's first search of the nearest cluster, which skips cluster 9 and may pick the global cluster (in FORCE's code `s != N_SIM-1` where the other searches test `s < N_CLS-1`); in the second, Zeit's default, the global cluster is never the first pick and cluster 9 competes. "Equal" is the share of observations whose ten bands are all identical.

| Sensor | Core's mode | Equal, all ten bands | Within 1, all ten bands | Values beyond 1 | Observations beyond 1 |
|:---|:---|---:|---:|---:|---:|
| Landsat 8 (LND08; LND09 shares its table) | FORCE's first search | 100% | 100% | 0 | 0 |
| Landsat 8 | Zeit's default | 97.711% | 97.711% | 725,874 of 49.4 million | 113,004 (2.289%) |
| Landsat 7 (LND07) | FORCE's first search | 100% | 100% | 0 | 0 |
| Landsat 7 | Zeit's default | 98.127% | 98.127% | 734,087 | 92,498 (1.873%) |
| Landsat 5 (LND05) | FORCE's first search | 100% | 100% | 0 | 0 |
| Landsat 5 | Zeit's default | 98.400% | 98.400% | 636,703 | 78,994 (1.600%) |

With FORCE's search the integer outputs are identical in all 49.4 million values of every sensor, the edge-case row included (the wrap in 16 bits of the saturated spectra too); not one value is off by 1, so the `acos` of the C library and Rust's agree on all of them. With Zeit's default, the observations beyond 1 of FORCE are exactly those where the two modes differ beyond 1 (113,004, 92,498 and 78,994 of them; none beyond 1 in one and not in the other), and the largest differences are in the predicted red edge bands (up to 521 in REDEDGE3 for Landsat 8, and 9,027 for Landsat 5, in its edge-case row), where a pixel whose first pick is another cluster gets another regression. By band, Zeit's default equals FORCE's in 97.7-99.0% of the values (Landsat 8); 234,098 values of the 49.4 million differ by exactly 1. The size of this difference is that of the synthetic spectra, which favor pure endmembers and so clusters near 9 and the global one; on 4,000 pixels of other spectra it was 1.3% (52). `benchmarks/vs_originals/force_tsa.py --part specadjust` (the cube: `force_tsa.sh` with `SENSOR=LND08 ONLY="spec_on spec_off"`, `YEARS="2018 2019"`, `SIZE=200`; also `LND07`, `LND05`). MODIS (MOD01, MOD02) is not compared: FORCE's TSA refuses it with `SPECTRAL_ADJUST` (`check_target_sensor` in `sensor-hl.c`: the union of MODIS's seven bands with Sentinel-2A's ten has 11, and the target sensor must have as many bands as the union), so there is no FORCE output to read; its tables are generated from FORCE's source, with the 50 `-1` of Aqua's SWIR1 and its implicit zero checked, and compared by checksum in the core's tests.

**Timings.** The same kind of cube, 500 x 500 pixels (250,000) with 136 dates over 2018-2019 (Landsat 8; 31.2 million observations pass the QAI screening), FORCE's compute stage with `SPECTRAL_ADJUST = TRUE` less the same run with `FALSE` (six bands as indices, the same QAI screening and index copy; the minimum of 3 runs each, with `force_time_ms.c` as above), against Zeit's core (`zeit._core.specadjust.specadjust_batch`, float32 input of the 31.2 million unmasked observations, the output of 31.2 million x 10 float64 values allocated in the timed call), 1 and 6 threads. Seconds.

| | FORCE | Zeit | Gain | Zeit, million observations/s |
|:---|---:|---:|---:|---:|
| 1 thread | 35.5 | 14.0 | 2.5× | 2.23 |
| 6 threads | 7.53 | 3.03 | 2.5× | 10.3 |

(FORCE's `spec_on` stage with 10 band indices copies four bands more than `spec_off` with 6: a few tenths of a second that stay in its time, which favors Zeit by under 3%.) FORCE's per-pixel code is a plain C loop over 51 clusters and 10 bands in OpenMP; Zeit's is the same arithmetic with the squared norm of each centre precomputed, the cosine compared with that of 15 degrees to skip `acos` on the far clusters, and a partial selection of the ten nearest, which is why the results are equal where FORCE's search is used. The gain is smaller than for the other FORCE products because the work is a regression on every observation (about 0.4 microseconds a pixel with one thread) and Zeit also pays for allocating 2.5 GB of float64 output; in the core's own profile (`examples/specadjust_profile.rs`, output reused) it is 2.9 million observations a second with 1 thread and 14.6 million with 8.

**Against the authors' code.** The method's authors' own code, **SpecHomo** (Scheffler, Apache-2.0), is not run: its trained classifiers are downloaded on first use as 218 MB (linear) and 504 MB (quantile) `dill` files, which execute arbitrary code when loaded, and their license is not declared anywhere in the package or its documentation. It would not reproduce FORCE's outputs anyway: SpecHomo uses seven Landsat 8 bands with the coastal band and eleven Sentinel-2A bands, a different weighting and threshold of the global cluster, and works in float32; FORCE's tables are a simplification of its own training. FORCE's code, written with the help of the paper's first author, is what the port follows.

### Tasseled Cap vs RStoolbox and FORCE

[`tasseled_cap`](../api/preprocessing.md#tasseled_cap) is compared with **FORCE 3.11.00**'s `TC-BRIGHT`, `TC-GREEN`, `TC-WET` and `TC-DI` (one matrix, Crist 1985, for every sensor) and with the R package **RStoolbox 1.0.2.2**'s `tasseledCap` (a matrix per sensor).

**FORCE.** FORCE's TSA on the synthetic cubes of the other FORCE comparisons (`INDEX = TC-BRIGHT TC-GREEN TC-WET TC-DI`, no interpolation, no noise filter, `TSS` written), a Sentinel-2A cube of 100 x 100 pixels and 325 dates and Landsat 8, 7 and 5 cubes of 200 x 200 pixels and 136 dates, with edge-case spectra (zeros, negatives, saturated values) in the Landsat ones. The input is FORCE's own `TSS` of the six bands (a second run with `INDEX = BLUE GREEN RED NIR SWIR1 SWIR2`, which FORCE writes as the raw int16; for Sentinel-2A its `NIR` is B8A, the eighth of its ten bands, and not `BROADNIR`), so that only the arithmetic is compared, and Zeit's float32 result is truncated as FORCE's `(short)` does.

| Cube | Observations computed | Values compared | Equal | Off by 1 | Beyond 1 |
|:---|---:|---:|---:|---:|---:|
| Sentinel-2A | 2,915,375 of 3,250,000 | 11,661,500 | 100% | 0 | 0 |
| Landsat 8 | 4,937,286 of 5,440,000 | 19,749,144 | 100% | 0 | 0 |
| Landsat 7 | 4,937,286 of 5,440,000 | 19,749,144 | 100% | 0 | 0 |
| Landsat 5 | 4,937,286 of 5,440,000 | 19,749,144 | 100% | 0 | 0 |

All four components are equal in every observation FORCE computes (the disturbance, `TC-DI`, is `TC-BRIGHT - TC-GREEN - TC-WET` accumulated in float32 in that order, as Zeit does), and the observations FORCE leaves as nodata are nodata in all four of its products. The equality is exact because Zeit adds the six float32 products in FORCE's band order; a matrix product by BLAS, which fuses and reorders them, would be off by 1 now and then.

**Timings.** A Landsat 8 cube of 500 x 500 pixels with 136 dates (34 million pixel-dates, 91.7% of them kept by FORCE's QAI screening), FORCE's compute stage with the four components less the same run with the six bands as indices (the same screening; the minimum of 3 runs, `force_time_ms.c` as above), against `zeit.tasseled_cap` of the whole cube (int16 in, float32 out, four components, the 8.3% that FORCE screens out included). Seconds.

| | FORCE | Zeit | Gain | Zeit, million pixel-dates/s |
|:---|---:|---:|---:|---:|
| 1 thread | 0.567 | 0.158 | 3.6× | 215 |
| 6 threads | 0.156 | 0.065 | 2.4× | 520 |

(FORCE's run with four indices keeps four bands in its time series and the other six, which favors FORCE's difference being too small; with the whole run FORCE takes 1.61 s with 1 thread and 0.39 s with 6.) The work is a few multiply-adds a value and memory-bound. Zeit's compiled core converts tiles of 4096 pixels of the six bands to float32 in a buffer that stays in the cache, adds the products in FORCE's order (no fused multiply-add) and writes each output value once; it reads 0.4 GB and writes 0.5 GB of float32, where FORCE writes 16 bits. A numpy version of the same arithmetic took 0.267 s and 0.189 s (0.8× FORCE on six threads).

**RStoolbox.** `tasseledCap(img, sat)` on a synthetic image of 1000 x 1000 pixels and six bands (mixtures of vegetation, soil and water with noise, reflectance 0-1), one thread.

| Sensor | Matrix | RStoolbox | Zeit, 1 thread | Speedup | Largest difference |
|:---|:---|---:|---:|---:|:---|
| Landsat 8 OLI | Baig et al. 2014 | 2.57 s | 24 ms | 107× | 8e-8 |
| Landsat 7 ETM+ | Huang et al. 2002 | 2.54 s | 51 ms | 50× | 7e-8 |
| Landsat 5 TM | Crist 1985 | 2.42 s | 28 ms | 88× | brightness 8e-8; greenness 1.5e-4; wetness 0.52 |

The OLI and ETM+ matrices are equal to 1e-7, the float32 rounding of Zeit against the double of R. For TM the brightness is equal and greenness and wetness are not, because RStoolbox's table has `+0.0002` and `+0.6806` for the SWIR1 weight of greenness and wetness, where Crist (1985) and FORCE have `-0.0002` and `-0.6806`: with those two numbers given to Zeit as `coefficients=`, all three components are equal to 8e-8. Zeit follows the paper. RStoolbox has no matrix for Sentinel-2, and Zeit gives Sentinel-2 Crist's, as FORCE does (the comparison with FORCE above).

### ImproPhe vs FORCE

[`improphe`](../api/preprocessing.md#improphe) is compared with **FORCE 3.11.00**'s CF-ImproPhe (`CFIMP`) and L2-ImproPhe (`L2IMP`), the code it is ported from (Frantz et al. 2016; GPL-3.0-or-later), on one synthetic tile (`force_improphe_cube.py`): a truth map of five classes in Voronoi patches (crop, forest, grass, water, urban, each a mix of two spectra with its own seasonal weight), a smooth gradient and a static brightness texture whose strength changes from patch to patch; **297 x 297 pixels of 10 m**. The fine data are 90 Sentinel-2A dates over 2018-2019 (10 bands; irregular revisit; clouds, shadows and swaths without data on 29 of them; the doys 1, 91, 181, 271 and 365 of 2018 among them, the limits of the seasonal windows; doys 91-250 of 2019 missing, so that a window is empty in that year; one column without any observation). L2's coarse scenes are 24 Landsat 8 dates at 30 m, block means (3 x 3) of the truth with noise, clouds and a hole of NoData; CF's is the year's peak NDVI of the truth at 30 m (int16, NoData -9999, a hole in each year). FORCE reads the 30 m files at 10 m by nearest neighbour, which for the integer factor is the replication of each pixel; Zeit's `reproject_match` onto the fine grid gives exactly that (checked). The fine cube Zeit gets is FORCE's own input, the BOA with the observations that FORCE's `SCREEN_QAI` drops set to NaN, so that only the method is compared. Parameters: `RESOLUTION = 10`, `KERNEL_SIZE` 100 and 200 (radii of 10 and 20 pixels, `mink` 1 and 6), `KERNEL_TEXT = 30`, `SEASONAL_WINDOW` `1 91 181 271 365` and `1 121 241 365`, both years predicted. `zeit.improphe(..., force=True)` truncated toward zero is compared with FORCE's int16 `_IMP` files over the pixels Zeit predicts.

**Two FORCE runs, and a trap in the conda package.** The principal components of FORCE take their eigenvectors from GSL's `gsl_eigen_symmv`, which calls a CBLAS, and Zeit's port follows GSL's own reference CBLAS. FORCE's reference build (conda-forge) is linked to OpenBLAS, so every configuration was run twice: **A**, as built (OpenBLAS with one thread), and **B**, with GSL's reference CBLAS preloaded (`LD_PRELOAD`). The file to preload is `libgslcblas.so.0.0.0`: conda-forge makes `libgslcblas.so.0` a symbolic link to `libcblas.so.3`, which is OpenBLAS again (a first "B" run with the link was A bit for bit; the check is that its `cblas_dnrm2` gives GSL's scaled sum of squares, not OpenBLAS's). FORCE's compute is deterministic in each run, and the two differ only in the last bits of the covariance and eigenvectors.

| FORCE run | Product | Radius | Seasons | Values compared | Equal | Within 1 | Largest difference |
|:---|:---|---:|:---|---:|---:|---:|---:|
| B (GSL's CBLAS) | CF 2018 and 2019 | 10, 20 | 4 and 3 windows | 8 x 88,065 | **100%** | 100% | 0 |
| B | L2, 24 scenes x 6 bands | 10, 20 | 4 and 3 windows | 4 x 12,516,660 | **100%** | 100% | 0 |
| A (OpenBLAS) | CF | 10 | 4 windows | 2 x 88,065 | 100% | 100% | 0 |
| A | CF | 10, 20 | 3 windows (and 20 with 4) | 6 x 88,065 | 99.998-99.999% | 100% | 1 |
| A | L2 | 10 | 4 and 3 windows | 2 x 12,516,660 | 99.9975-99.9977% | 100% | 1 |
| A | L2 | 20 | 4 windows | 12,516,660 | 99.9967% | 99.9997% | 11 |
| A | L2 | 20 | 3 windows | 12,516,660 | 99.9980% | 100% | 1 |

With FORCE built on GSL's own CBLAS the integer outputs are identical in every value, so are the intermediate arrays (the seasonal means, the principal components after the standardization and the texture of CF at radius 10 on the cube with band noise below, compared on a copy of FORCE's build that writes them: 100% of the pixels bit for bit, 3 and 2 components for the two years). With OpenBLAS the outputs differ in 0.002-0.003% of the values: the last bits of the eigenvectors move the standardized components by about one float32 unit, and a neighbour whose distance sits on one of the five thresholds goes in or out (perturbing the components of Zeit's own run by one unit in the last place moves the pixel with the largest difference between 3217.5 and 3221.5, FORCE's value being 3221). A second cube with a static noise of 10% of every band in every pixel (`--band-noise 0.1`, `NOISE=0.1` in the script; its seasonal means are nearly collinear in 2019, where one window is empty and takes the mean of all the scenes: the smallest eigenvalues are 0.12 against 2.9e7 for the largest, and the eigenvectors that FORCE's transposed projection uses depend on the rounding) gives the same B result, 100% equal (CF at radii 10 and 20, L2 at 10), and for A 99.98% equal in 2018 and 99.1-99.6% in 2019 (99.97% within 1, differences up to 644): in a rank-deficient case FORCE's own result depends on the BLAS it is linked to.

Where FORCE does not predict (screened by its QAI in the scene, or a NoData coarse value) it writes, in the L2 outputs, the coarse value as it is (the cloud reflectance of the scene) and the NoData value for the hole, and in the CF output a 0 (all 144 pixels of the 2018 hole); Zeit leaves them NaN. The pixels without any fine observation (a column) keep the coarse value in both.

**What the transposed projection costs.** `force=False` (the paper's PCA, float means, day 365 in the last window) is not meant to equal FORCE's output. Against FORCE's (B), on the first cube, it correlates 0.9996-0.99998, 30-91% of the values are within 1, 85-99% within 1% and 98-99.9% within 5% (on the cube with band noise the correlation is 0.9964-0.9995 and 2-44% are within 1). Against the truth of the cube (the RMSE, in the unit of the data, over the pixels Zeit predicts), with a third run that keeps FORCE's projection but uses float seasonal means (`pca="force"`, `force=False`): no projection is better throughout. FORCE's is much better in 2018 at radius 20 and somewhat worse elsewhere; the same projection on float means, whose eigenvectors come out with other signs, keeps part of that gain in 2018 and is 38% worse in 2019 at radius 10. The quality of FORCE's components depends on the signs GSL gives its eigenvectors, which the principal components do not; that is why they are Zeit's default. Nearest-neighbour upsampling of the coarse data is several times worse than any of them:

| Data, radius | FORCE (and Zeit `force=True`) | Zeit default (principal components) | FORCE's projection, float means | Nearest neighbour |
|:---|---:|---:|---:|---:|
| CF peak NDVI x 10000, 2018, 10 / 20 | 77.7 / 40.8 | 102.1 / 115.4 | 78.6 / 58.8 | 647.7 |
| CF, 2019, 10 / 20 | 67.5 / 43.7 | 64.2 / 47.7 | 93.2 / 42.3 | 632.7 |
| L2 reflectance x 10000, 10 / 20 | 60.2 / 59.2 | 61.6 / 61.5 | 58.4 / 55.5 | 133.5 |
| the cube with band noise: CF 2018 / 2019, 10 | 627 / 570 | 665 / 576 | 673 / 573 | 805 / 797 |
| the cube with band noise: CF 2018 / 2019, 20 | 496 / 533 | 494 / 527 | 495 / 549 | 805 / 797 |
| the cube with band noise: L2, 10 | 186 | 184 | 189 | 210 |

**FORCE 3.11.00's L2IMP does not start as released.** `param-hl.c` registers the target sensor, main product and quality product of `phl->sen` only; for the secondary sensors (`SENSORS_LOWRES`) they stay NULL, so `retrieve_sensor(&phl->sen2)` fails with "Item with key (null) not found in JSON" and, with that fixed, the reading of the Landsat scenes dereferences a NULL `main_product` and crashes. The L2 runs use a copy of FORCE's build tree in which `param-hl.c` fills in the target sensor (`LNDLG`, the six Landsat bands) and copies the two products (`force_improphe_l2fix.sh`; the original tree and the installed binaries are untouched, and nothing of the ImproPhe code is changed). CFIMP runs with the installed binary.

**How the first comparison found a flaw in Zeit.** The first run of Zeit's port of GSL's eigensolver against FORCE agreed to 99.1-99.99% and, with the two builds of FORCE then thought to be A and B (the symbolic link above), looked the same for both. A copy of FORCE's build that writes the covariance, the eigenvectors, the components and the texture (`force_improphe_dump.sh`) showed the seasonal means and the covariance equal and the eigenvectors different in the last bits; a small C program calling GSL's reduction and QR steps on that covariance (`gsl_stages.c`), against the same stages of the port (`examples/gsl_stages.rs`), put the first difference in the first Householder step: GSL calls the C library's `hypot`, which on glibc is correctly rounded, and Rust's `f64::hypot` on Windows is the C runtime's, which is not. The port now has a correctly rounded `hypot`, and its tridiagonal reduction, orthogonal factor, eigenvalues and eigenvectors equal GSL 2.7's with its reference CBLAS in every bit.

**Timings.** The 153 x 153 pixel tile of the same kind (84 Sentinel-2A dates; CF, two years; L2, four Landsat 8 scenes), `RESOLUTION = 10`, `KERNEL_TEXT = 30`, FORCE's compute stage (screening, seasonal means, PCA, standardization, textures, neighbourhood step; no reading and no writing, which FORCE does in other threads; the minimum of 3 runs, `force_time_ms.c` as above) against the whole `zeit.improphe(..., force=True)` call on the same files (the best of 3; it includes the regridding of the coarse data), with 1 and 6 threads. Seconds. A tile this small cuts the circle of radius 50 at its edges, which favors both.

| Product, radius | FORCE, 1 thread | Zeit | Gain | FORCE, 6 threads | Zeit | Gain |
|:---|---:|---:|---:|---:|---:|---:|
| CF, 10 | 0.479 | 0.329 | 1.5× | 0.115 | 0.093 | 1.2× |
| CF, 20 | 1.169 | 0.790 | 1.5× | 0.270 | 0.194 | 1.4× |
| CF, 50 | 3.773 | 2.224 | 1.7× | 1.296 | 0.534 | 2.4× |
| L2, 10 | 1.683 | 1.036 | 1.6× | 0.419 | 0.274 | 1.5× |
| L2, 20 | 4.371 | 2.689 | 1.6× | 1.131 | 0.646 | 1.8× |
| L2, 50 | 15.293 | 7.133 | 2.1× | 3.958 | 1.675 | 2.4× |

The cost is the neighbourhood step, as FORCE's (one year or scene at radius 10 on 1 thread: 34 ms for the means, components, standardization and texture, 120 ms for the neighbourhood step; radius 50: 34 and 1,124 ms). Zeit's gain comes from the two passes of the neighbourhood step (the first finds the threshold, with an early exit of the sum of differences once it exceeds the last threshold, and the weights are computed over the kept neighbours only), the rows of the circle precomputed, and the components laid out by band; the arithmetic and its order are FORCE's, which is why the results are equal. FORCE's step is O(`pi h^2`) per pixel with all its neighbours' distances computed in full.

### The resolution merge vs FORCE

[`sentinel2_resmerge`](../api/preprocessing.md#sentinel2_resmerge) is compared with **FORCE 3.11.00**'s resolution merge (`resmerge-ll.c`: `resolution_merge_1`, the regression, and `resolution_merge_2`, the ImproPhe-like weighted mean; Frantz; GPL-3.0-or-later), the code it is ported from. FORCE runs the merge only inside `force-l2ps`, between the coregistration and the radiometric correction, so there is no program to feed an ARD tile to. The comparison is therefore with a C harness (`benchmarks/vs_originals/force_resmerge/`, built by `make_harness.sh` in WSL) made of FORCE's own functions copied verbatim (`resolution_merge_1` and `_2`, `improphe_resmerge`, `rescale_weight_resmerge`, `distance_kernel`) behind shims that replace the brick and QAI accessors with plain arrays and masks, and linked statically to the real GSL 2.7 (`libgsl.a` and the reference `libgslcblas.a` of the conda env `force-ref`: conda-forge's `libgslcblas.so.0` is OpenBLAS, as for ImproPhe). It runs on one OpenMP thread, where FORCE's regression is deterministic. STARFM, the third option of the file (the USDA's code), is not in the harness nor in Zeit.

In `force-l2ps` the 20 m bands reach the 10 m grid when the digital numbers are read: `GDALRasterIO` into the 10 m buffer, whose default resampling is the nearest neighbour, so each 20 m pixel is four equal pixels. The scene made by `force_resmerge.py` has that structure: **500 x 500 pixels of 10 m**, int16 reflectance times 10000 of six land covers (blocks of 40 pixels, an illumination gradient, 5% noise in every band), the six 20 m bands as 2 x 2 blocks, a patch of constant spectra and a patch of zeros (rank deficient fits), a stripe and 1% scattered pixels of no data (4,466 in all), two clouds (2,946 pixels) and their shadows (1,000). Zeit gets it as float32 with NaN where FORCE has the "off" mask, and `clear` from the cloud and shadow masks. `force=True` is compared with FORCE's int16 output over the 245,534 pixels with data (241,660 of them predicted, the others left as they were).

| Method | Band | Identical | Max \|difference\| |
|:---|:---|---:|---:|
| regression | red edge 1, 2, 3, B8A, SWIR1, SWIR2 | 100% (all six) | 0 |
| ImproPhe | red edge 1, 2, 3, B8A, SWIR1, SWIR2 | 100% (all six) | 0 |

Every value of the six bands is the same, for both methods (the first comparison, on a 300 x 300 scene, was also identical). That needed GSL's `gsl_multifit_linear` ported to the operation (the column balancing, `gsl_linalg_SV_decomp_mod` and the Golub-Reinsch iteration, the reference CBLAS without fused multiply-adds, and the correctly rounded `hypot` of the eigensolver port, since Rust's `hypot` on Windows is not glibc's); the matrix is the same for the six bands, so Zeit decomposes it once per pixel, which gives the bits of GSL's six decompositions.

**What `force=False` changes.** The regression of FORCE writes each prediction into the band before the next pixel's window reads it. With `force=False` every window reads the original values: on this scene the two differ by 11-20 digital numbers on average (up to 490) and only 4-11% of the values are within 1, because the in-place sweep reads smoothed predictions where the other reads the blocks of the 20 m bands; `force=False` agrees with an independent least squares (numpy's `lstsq` on the same windows, 216 pixels) to 1e-4. ImproPhe has no feedback: the two modes differ by the truncation (all values within 1, 0.49 on average).

**Timings.** The same scene, the whole call (FORCE: the merge function inside the harness, gcc -O2, OpenMP; Zeit: `zeit.sentinel2_resmerge`, release build on Windows, with its conversions), the best of 3, milliseconds. **The machine was not quiet**: other builds ran throughout (load 93-96%), so these are indicative, and nothing was re-measured later. FORCE's regression with 4 threads is racy (its output differs from the one-thread run); only its time is of use.

| Method | FORCE, 1 thread | Zeit, 1 thread (`force=False`) | FORCE, 4 threads | Zeit, 4 threads (`force=False`) | Zeit `force=True`, 1 / 4 |
|:---|---:|---:|---:|---:|---:|
| regression | 3,691 | 627 | 1,292 | 185 | 630 / 617 (one thread by design) |
| ImproPhe | 86 | 140 | 39 | 50 | 139 / 50 |

Zeit's regression is 5.9 times as fast as FORCE's on one thread (the SVD is done once per pixel and not six times, in fixed-size arrays) and 7.0 times with four; its ImproPhe is slower than FORCE's C (0.6 times), where the time includes Python's conversions of the cube and the masks, and the kernel itself is small (13 neighbours).

### The parallax cloud test and the cloud buffer vs FORCE

[`parallax_clouds`](../api/preprocessing.md#parallax_clouds), [`cloud_displacement_index`](../api/preprocessing.md#cloud_displacement_index) and [`buffer_mask`](../api/preprocessing.md#buffer_mask) are compared with **FORCE 3.11.00**'s `cloud_parallax` (`cloud-ll.c`; Frantz's own parallax step, the Cloud Displacement Index of Frantz et al. 2018) and `buffer_` (`imagefuns-cl.c`), the code they are ported from (Frantz; GPL-3.0-or-later; `cloud-ll.c` as a whole derives from Zhu's Fmask, whose parts are not ported). `cloud_parallax` runs only inside `force-l2ps`, after the potential cloud pixels and the variability probability that Fmask's part computes, so the comparison is with a C harness (`benchmarks/vs_originals/force_parallax/`, built by `make_harness.sh` in WSL) made of FORCE's own functions copied verbatim (`cloud_parallax`, `gauss_kernel`, `covar_recurrence`, `variance`, the queue, `buffer_`) behind shims that replace the brick and QAI accessors with plain arrays; the one line added to `cloud_parallax` hands the CDI (which the function does not return) to the harness. The scene made by `force_parallax.py` is **512 x 512 pixels of 10 m**, int16 reflectance times 10000: a land surface whose red edge 3 and NIR vary smoothly and independently; four textured clouds seen at the same place in B07 and B8A and 6 pixels aside in B08; three bright built-up patches (all bands at the same place, ratios changing alike); a bright patch with a 1 pixel offset and a variability probability of 0.93 (the second way into the cloud class); a variability field; 53,252 potential cloud pixels; 4,510 off pixels (a stripe, scattered ones, a block, the top rows); 798 saturated pixels; and a bright-pixel mask that excludes 13% of the scene. Zeit gets the same arrays (float32 with NaN at the off pixels, the masks as boolean arrays, the variability as float).

| Comparison | Identical |
|:---|---:|
| Cloud mask (42,709 cloud pixels in FORCE), `parallax_clouds` vs `cloud_parallax` | 100% (0 of 262,144 pixels differ) |
| CDI on the 20 m grid (14,050 blocks with one; 10,541 at or below -0.5; 659 above 0), bit for bit | 100% (0 of 65,536 blocks) |
| Four more scenes (other seeds; variability times 0.5 and 1.15), the mask and the CDI | 100% |
| `buffer_mask(force=True)` vs `buffer_`: FORCE's cloud mask at 30 and 15 pixels (300 m at 10 and 20 m), random masks at radii 1, 2, 3, 7, 12 and 30, blocks at the edges | 100% (0 pixels in 9 cases) |

One thread and four threads of Zeit give the same mask (FORCE's threads do not race either). The scene exercises both ways into the cloud class (CDI of -0.5 or less, and CDI below 0 with a high variability), the erosion, both region-growing passes, the off blocks, the saturated and the non-bright gate; the comparison does not isolate each path.

**What `force=False` changes in the buffer.** FORCE's loop skips the pixels within `r` of the image's borders and then replaces the mask by what it drew, so a True pixel near an edge is lost (and its disc with it); `force=True` reproduces that. By default Zeit keeps every pixel and clips the discs at the edge: it differs from FORCE only where a source within `r` of an edge would have drawn (17,943 of 262,144 pixels at `r = 30` on the cloud mask; 40 pixels for blocks touching the edges at `r = 5`).

**Timings.** A 2048 x 2048 scene of the same kind, milliseconds, best of 3 (FORCE: `cloud_parallax` inside the harness, gcc -O2, OpenMP; Zeit: release build on Windows). **The machine was not quiet**: other builds ran throughout (load near 100%), so these are indicative, and nothing was re-measured later. Zeit's "whole call" includes the conversions of the cube and the masks to FORCE's int16 units and booleans in numpy.

| | FORCE, 1 thread | Zeit, 1 thread | FORCE, 4 threads | Zeit, 4 threads |
|:---|---:|---:|---:|---:|
| `cloud_parallax` / `parallax_clouds`, the whole call | 246 | 449 | 85 | 257 |
| the same, the compiled core alone | 246 | 253 | 85 | 80 |
| `buffer_` / `buffer_mask`, radius 30, the cloud mask | 4,156 | 661 | 1,344 | 196 |

The parallax test costs the same in Zeit's core as in FORCE's C; Python's conversions of the cube are the rest. The buffer is 6 times faster on one thread and 7 with four: FORCE draws a disc from every True pixel (a time that grows with the cloud's area and the radius squared), Zeit looks up per row the nearest source in the disc's runs.

### Topographic correction vs RStoolbox, GDAL and terra

[`topographic_correction`](../api/preprocessing.md#topographic_correction) is written from the published formulas (Horn 1981; Teillet et al. 1982; Soenen et al. 2005), so it is compared by outputs only: slope and aspect with `gdaldem slope` / `gdaldem aspect` (GDAL, `-compute_edges`) and `terra::terrain` (R, 8 neighbours = Horn), and the C-correction with **RStoolbox 1.0.2.2**'s `topCor(method = "C")` (the formula `rho (cos sz + c) / (cos i + c)`, `c = b / m` of the line `rho = b + m cos i`). `benchmarks/vs_originals/topocorr.py` (and `topocorr.R`) runs on synthetic terrain (hills of 1-6 km, slopes up to 62 degrees, median 22, 30 m pixels; no DEM ships with the repository) and a six-band scene made of a smooth vegetation/soil mix times `0.7 cos i / cos sz + 0.3` plus noise (sun zenith 38, azimuth 140).

**Slope and aspect**, 1000 x 1000 cells, maximum differences:

| | slope (degrees) | aspect (degrees) |
|:---|---:|---:|
| `terra::terrain`, inside (its edge is NA) | 8.1e-6 | 4.8e-5 |
| `gdaldem`, inside | 2.5e-4 (GDAL writes float32) | 0.089 (cells of almost no slope) |
| `gdaldem -compute_edges`, edge cells | 7.1 | 157 |

The edge cells are the one convention that differs: Zeit replicates the nearest cell and takes the difference over the single cell it spans (a plane's slope comes out exact on the edge); GDAL's `-compute_edges` does something else, and RStoolbox/terra leave the edge NA.

**C-correction.** RStoolbox is given Zeit's slope and aspect (radians), so that only the correction is compared. It fits the line on a sample of at most 10000 pixels of the image (random, with `spatSample`; all the pixels when the image has 10000), with no slope limit for `method = "C"`; Zeit on every pixel with a slope of at least `min_slope` (set to 0 here, to have the same pixels):

| Image | Zeit's `c` (six bands) | Max relative difference | Mean \|difference\| (reflectance 0-1) |
|:---|:---|---:|---:|
| 100 x 100 (RStoolbox fits on all the pixels) | 0.37, 0.36, 0.38, 0.32, 0.35, 0.37 | 4.3e-7 | 8.6e-9 |
| 1000 x 1000 (RStoolbox fits on 10000 of them) | 0.34 in every band | 3e-3 to 4.5e-2 (three runs) | 2e-5 to 1.7e-4 |

On the same pixels the two are the same computation (float32 against float64). With a sample the line, and so `c`, moves with the pixels RStoolbox happens to draw, and the correction by the same amount; the 1000 x 1000 figures change from run to run (RStoolbox draws a different sample each time). With RStoolbox's own `terrain` slope and aspect (NA on the edge) on the 100 x 100 image the maximum difference is 1.1e-4 in reflectance, from the slopes (float32 and edge cells).

**Timings.** Seconds, the best of 3, one thread, whole calls (Zeit's: slope and aspect, illumination, six regressions and the correction, with Python's conversions; RStoolbox's `topCor` of six bands, its sample of 10000): slope and aspect 1000 x 1000: `terra::terrain` 0.05, Zeit 0.037; the correction of 1000 x 1000 x 6: RStoolbox 0.41, Zeit 0.087 (4.7 times); 100 x 100 x 6: 0.06 and 0.003. RStoolbox fits on 10000 pixels where Zeit uses all of them (a million per band), so the correction itself is more work in Zeit. **One 2000 x 2000 x 6 date**, the whole call, Zeit only: 0.353 s with 1 thread and 0.117 s with 4 (SCS+C the same). The machine was not idle (other sessions' processes were running, load under 10% before the runs), so the times are indicative.

### The radiative transfer vs FORCE

The radiative transfer pieces of Zeit's core (`zeit_core::radtran`; the public atmospheric correction comes later) are a port of **FORCE 3.11.00** (David Frantz; GPL-3.0-or-later): all of `radtran-ll.c`, `wavelength` and `E0` of `table-ll.c` with the tables they read (the Thuillier irradiance, the HITRAN 2016 water vapour and the Bird and Riordan ozone absorption, the relative spectral response of Landsat 4, 5, 7, 8 and Sentinel-2A and B, 56 bands), `ozone_amount` of `gas-ll.c` and `localtime2gmt` of `sunview-ll.c`, with the overpass time of `standard_sunzenith`. The sun position (`sunpos`, and the zenith of `standard_sunzenith`) is not FORCE's `sun-cl.c` (adopted by FORCE from the NOAA calculator, whose license is not verified): it is `zeit_core::solpos`, a clean-room implementation from Meeus' *Astronomical Algorithms* (ch. 7, 25, 28) and NOAA's published documentation, so it is compared by the difference in degrees. The C harness (`force_radtran/`) compiles FORCE's `radtran-ll.c`, `table-ll.c` and `date-cl.c` as they are (and `sun-cl.c`, only as the reference of the sun position), with FORCE's flags (`gcc -O3 -fopenmp`, no `-ffast-math`), and copies the three functions of `gas-ll.c` and `sunview-ll.c` that cannot be compiled alone; both sides evaluate each function on the same float32 columns, and the bits of the outputs are compared.

| Function | Values | Bit for bit | 1 ulp | More |
| --- | --- | --- | --- | --- |
| The 23 functions of `radtran-ll.c` (grid of angles, AOD, MOD, elevation, the 56 bands with water vapour and ozone paths) | 5,477,447 | 5,477,447 | 0 | 0 |
| `wavelength`, `E0` (the 56 bands) | 112 | 112 | 0 | 0 |
| `ozone_amount` | 200,004 | 200,004 | 0 | 0 |
| `localtime2gmt` (year, month, day, day of year, hour, minute, second) | 1,400,000 | 1,400,000 | 0 | 0 |

The functions above are equal bit for bit (7,077,563 values of 27 functions). The sun position is not, and is reported in degrees against FORCE's `sunpos` (float32; 200,000 random places, dates and times, in 1984-2030, latitudes within 90 degrees; `standard_sunzenith` 200,000 with latitudes within 80):

| | Subset | Values | Mean | Maximum |
| --- | --- | --- | --- | --- |
| `sunpos` zenith | day (zenith < 90) | 101,593 | 0.0120 | 0.068 |
| `sunpos` zenith | night | 98,407 | 0.0124 | 0.088 |
| `sunpos` azimuth | day | 101,593 | 0.066 | 177 (sun within a fraction of a degree of the zenith) |
| `standard_sunzenith` | day | 170,711 | 0.0126 | 0.051 |
| `standard_sunzenith` | night | 29,289 | 0.0118 | 0.078 |

The difference is FORCE's: its Julian day is a float32, so it moves in steps of a quarter of a day (the error of the declination is a sawtooth of period 6 h, as large as 0.049 degrees at the equinoxes and 0.003 at the solstices, and is zero at 0, 6, 12 and 18 h UT); with the time at a multiple of 6 h the zeniths agree to 1e-4 degrees, and the signed difference has mean zero in every zenith band. Zeit's is the more accurate one (Meeus' method is 0.01 degrees against an ephemeris). Beyond about 82 degrees of latitude the overpass time of `standard_sunzenith` is NaN and the C turns it into `INT_MIN` hours: both sides then compute a sun position at an absurd instant and differ (402 cases), which has no meaning.

The other values are equal bit for bit: the arithmetic is the C's (`float` in `f32`, `double` where the C promotes, each float variable rounded where the C stores it; the products of two floats stay floats), and Windows' UCRT and glibc 2.39 give the same `sin`, `cos`, `tan`, `exp`, `pow`, `asin` and `acos` on these inputs. The first port of `wavelength` and `E0` summed `RSR*WVL` and `RSR*E0` in double where the C multiplies two floats, and was 1 ulp off in 5 bands: the harness found it. The relative spectral response is stored as the span between the first and last non-zero value of each band (the zeros add nothing to the sums). `python benchmarks/vs_originals/force_radtran.py`.


### Water vapour and gas transmittance vs FORCE

The rest of `gas-ll.c` of **FORCE 3.11.00** (David Frantz; GPL-3.0-or-later) is in Zeit's core (`zeit_core::gas`; the public atmospheric correction comes later): `wvp_transmitt_lut` (the transmittance of each band for 701 water vapour amounts and 101 air masses), `water_vapor` (the water vapour of each 60 m block of a Sentinel-2 image, found by a Nelder-Mead simplex so that the surface reflectances of the NIR band and of the water vapour band agree; GSL 2.7's `nmsimplex2` is ported in one dimension), `gas_transmittance` and `water_vapor_from_lut` (FORCE's daily and climatological `WVP_*.txt` files). The C harness (`force_gas/`) compiles FORCE's `gas-ll.c` as it is (but for its first include), with `radtran-ll.c` and `table-ll.c`, behind minimal stand-ins for FORCE's brick, `atc`, metadata and parameter structures, against the real GSL 2.7 linked statically with its own reference CBLAS (conda's `libgslcblas.so` is OpenBLAS, whose `dnrm2` rounds differently); the functions of `quality-cl.c`, `brick_base-cl.c`, `utils-cl.c`, `dir-cl.c` and `atc-ll.c` it uses are copied verbatim. FORCE's retrieval carries the last estimate of each OpenMP thread to its next pixel, so it depends on the number of threads; the harness runs with one thread, and Zeit's is that one-thread order (row-major blocks, one running sum).

| Scene (pixels, coarse cell) | LUT floats | Water vapour (`short`) | QAI after the fill | Gas transmittance, 13 bands | Blocks retrieved |
| --- | --- | --- | --- | --- | --- |
| 36 x 30, 120 m | 920,413 | 1,080 | 1,080 | 14,040 | 25 |
| 120 x 96, 120 m | 920,413 | 11,520 | 11,520 | 149,760 | 279 |
| 120 x 96, 60 m | 920,413 | 11,520 | 11,520 | 149,760 | 294 |
| 102 x 72, 300 m, low sun | 920,413 | 7,344 | 7,344 | 95,472 | 165 |
| 300 x 240, 300 m | 920,413 | 72,000 | 72,000 | 936,000 | 1,751 |
| 12 x 12, all water | 920,413 | 144 | 144 | 1,872 | 0 |

All 7,076,598 values are equal bit for bit (the LUT in float32, 0 ulp; the others whole numbers, 0 differences), and so are 300 of 300 `water_vapor_from_lut` scenarios on synthetic LUT files (random centres near and far from the sites, dates with and without a daily file, a month without a climatology, nodata 9999 and negative values, comments, doubled blanks and CRLF line ends, strict mode; 99 of them return -1). The scenes have a smooth water vapour field from 0.4 to 5 cm with a patch of 9 cm that cannot be retrieved, a sun and view geometry with cells outside the swath, an elevation for the 256 bins, TOA reflectances from the forward model plus noise, and a QAI with off pixels, shadow, water (some whole 60 m blocks) and other flags.

FORCE with 4 threads against FORCE with 1, on the large scene: 8,208 of 72,000 pixels differ, by 0.010 cm on average and 2.09 cm at most (the start value of the minimizer). Timings of the first run, one thread on both sides and not in a verified quiet window (so indicative only): FORCE 12.7 to 20.6 s per scene, almost all of it the LUT, Zeit 0.57 to 0.89 s for the whole call. `python benchmarks/vs_originals/force_gas.py`.

### Dark targets and the AOD vs FORCE

The first half of `aod-ll.c` of **FORCE 3.11.00** (David Frantz; GPL-3.0-or-later) is in Zeit's core (`zeit_core::aod`; the public atmospheric correction comes later). `extract_dark_target` finds the candidate dark targets of a type (water: SWIR2 under 2.5%, red under 20%, blue above NIR, a slope limit that depends on the elevation; vegetation: SWIR2 under 5%, red under 20%, blue below NIR, NDVI above 0.4, slope under 15 degrees), cuts them along the coarse grid so that none runs through the image, labels the 8-connected patches (Chang et al. 2004), measures their distance to the shore (Meijster et al. 2006), keeps the pixels whose spectrum decreases with the wavelength, and tabulates each target (bounding box, size, radius, mean illumination, sky view and elevation, the reflectance of the target and of its environment, three radii around it). `aod_lib_to_target` finds, for each spectrum of FORCE's water library (26 spectra, convolved with the spectral response of each band) or vegetation library (81), the AOD in each band at which the contaminated reference matches the target best (from -0.2 to 3 in steps of 0.001); `aod_from_target` fits a 2nd order polynomial (the Nelder-Mead simplex `nmsimplex2` of GSL 2.7, now ported in three dimensions) or an Angstrom power law to those AODs, and keeps the library spectrum with the highest R2; `aod_elev_dependency` finds the scale height of the aerosols with GSL 2.7's Brent minimizer. The shadow targets are disabled in FORCE 3.11.00 and are here. The C harness (`force_aod/`) compiles FORCE's `aod-ll.c` as it is (from its start to the end of `aod_polynomial_fit`, but for its first include), with `radtran-ll.c` and `table-ll.c`, behind minimal stand-ins for FORCE's brick, `atc`, metadata, parameter and topography structures, against the real GSL 2.7 linked statically with its own reference CBLAS; the functions of `quality-cl.c`, `brick_base-cl.c`, `utils-cl.c` and `imagefuns-cl.c` it uses are copied verbatim. The topography (`top_t`: binned DEM, slope, illumination, sky view; FORCE computes it in `topo-ll.c`, ported in [the next sections](#the-topography-vs-force)) is an input here, the same on both sides.

The scenes are Sentinel-2 at 10 m (13 bands of the spectral response table) on a land background with lakes (ellipses, one with an island), a river across the image that the grid cuts, ponds and streams too small or too narrow, salt-and-pepper water pixels, dark vegetation patches (some touching), cloud, cirrus, shadow and no-data regions, steep and high targets; the target spectra come from the forward model of the radiative transfer for a smooth AOD field and the libraries of FORCE, so that many targets pass the estimation.

| Scene (pixels, coarse cell) | Candidates (water / vegetation) | Valid after the fit | Values compared | Equal bit for bit |
| --- | --- | --- | --- | --- |
| 120 x 90, 150 m | 42 / 31 | 16 / 7 | 18,598 | 18,598 |
| 240 x 200, 300 m | 148 / 62 | 21 / 3 | 26,907 | 26,907 |
| 240 x 200, 200 m, noisy | 166 / 44 | 26 / 6 | 31,820 | 31,820 |
| 480 x 440, 300 m | 741 / 29 | 39 / 7 | 81,627 | 81,627 |
| 90 x 60, all cloud | 25 / 10 | 0 / 0 | 2,793 | 2,793 |
| 90 x 60, no candidates | 0 / 0 | 0 / 0 | 28 | 28 |

Per scene the values are the integers of the target lists (position, size, radius, grid cell), their floats (elevation, illumination, sky view, the reflectances of target and environment), the library AOD of every valid target, the fit (library spectrum, type of fit, R2, coefficients, fitted and estimated AOD) and the elevation factors (the scale height, the factor of the scene, of each coarse cell and of each target). In the four scenes with fits, 10 to 24 of the water targets have a successful 2nd order fit and the others, and all the vegetation targets, the Angstrom fit. Besides the scenes: `aod_elev_dependency` on 300 random lists of targets (9,703 values; in 89 of the cases Brent's method finds a minimum, in the others the AOD is taken as flat), `aodfileread` on 200 scenarios of synthetic `AOD_DDD.txt` files (nearest site, doubled blanks, CRLF, a file without a final newline, a day without a file: 200 of 200), and `dist_transform_` and `connectedcomponents_` on 7 random and smoothed images (104,133 values): all equal. In total 275,809 values, 0 differences.

FORCE runs its loops with OpenMP; the scene-average environment reflectance is a sum of doubles per thread, and the loop over the targets of `aod_from_target` gives each thread one `aodest`, `aodlog` and `aodfit` array that the next target it processes finds as the last one left it, so FORCE's output depends on the number of threads. The harness runs with one thread and Zeit's is that order (sums in pixel order, the arrays carried from target to target); with 4 threads FORCE gave the same targets and AOD on the large scene. Timings (one thread, not in a verified quiet window, so indicative only): extraction on the large scene FORCE 7 ms, Zeit 11 ms; estimation 27 ms and 24 ms. `python benchmarks/vs_originals/force_aod.py`.

### The AOD map vs FORCE and GDAL

The second half of `aod-ll.c` of **FORCE 3.11.00** is in Zeit's core too (`zeit_core::aod`, and `zeit_core::gdal_grid` for the interpolation; still no public function). `aod_map` adds the AOD of every valid target (water, shadow, vegetation), scaled from the target's elevation to the scene average, to its coarse grid cell with the squared R2 of its fit as weight, and divides; the cells without a target are marked (`xy_interp`). `interpolate_aod_map` interpolates the cells that have a green-band estimate with GDAL's `GDALGridCreate` and then smooths the result with a 3 x 3 mean in the cells with view data; `compile_aod` is the entry point: the AOD file or the global fallback, the bands of the estimation (visible and SWIR2), the extraction, the AOD per target, the elevation dependency, and the map, or the fallback if no target is valid.

What FORCE passes to GDAL: `GDALGridCreate(GGA_InverseDistanceToAPower, <"invdist" with its defaults: power 2, smoothing 0, no search radius, no limit of points, nodata 0>, k points, 0, nf-1, 0, ne-1, nf, ne, GDT_Float64, ...)`, once per band, over the whole grid (the nodes are at `(i + 0.5) * (nf - 1) / nf`, not at the cell centres; reproduced). With those options `GDALGridContextCreate` selects `GDALGridInverseDistanceToAPower2NoSmoothingNoSearchAVX` (`alg/gdalgridavx.cpp` of GDAL 3.9.3, the version of the conda env `force-ref`): the coordinates and values are rounded to `float`, 16 points at a time go into 8 `float` lanes (each lane adds the term of the first vector and then that of the second), the lanes are added from left to right, the remaining points are added one at a time (against a `double` threshold; the vector loop uses a `float` one), and `1 / r2` in the vector loop is the CPU's approximation `_mm256_rcp_ps`, not a division. Zeit's `gdal_grid.rs` does that operation by operation and calls `_mm_rcp_ss`, the scalar form of the same instruction. GDAL splits the rows among `GDAL_NUM_THREADS` threads and each cell is computed whole by one, so the thread count does not change the result.

Compared against the real `GDALGridCreate` (the harness links libgdal 3.9.3 of `force-ref`) and FORCE's `compile_aod` as it is (`OMP_NUM_THREADS=1`, `GDAL_NUM_THREADS=1`), on an Intel i5-13600K:

| | values | bit for bit | differ |
| :--- | ---: | ---: | ---: |
| `GDALGridCreate invdist`, 240 random grids (0 to 600 points, 1 x 1 to 60 x 50) | 143,903 | 143,903 | 0 |
| `compile_aod`, 16 runs on 11 scenes: AOD map, `xy_interp`, `atc->aod`, `Hp`, `Ha`, `xy_Ha`, `aod_bands`, return value | 51,690 | 51,690 | 0 |

The scenes: four ordinary ones (120 x 90 to 400 x 340 pixels of 10 m, up to 1,360 coarse cells of 100 m), one target (a lake) on an 8 x 6 grid and on a 2 x 2 grid, only water, only vegetation, targets that all fail the fit (no map, the fallback), no candidate target and all cloud (the fallback), the first scene again with an AOD file and with the AOD estimation switched off (with and without the file); the cells outside the swath keep no AOD. **The `rcpps` approximation depends on the CPU** (Intel's and AMD's tables differ: FORCE's own maps differ by about 1e-3 between the two), so "bit for bit" is on the same CPU; on a non-x86 target Zeit divides exactly. Only the AVX path of GDAL is ported; with `GDAL_USE_AVX=NO`, or on a CPU without AVX, GDAL uses its SSE path (four lanes, another order of the sums). Timings (one thread, not in a verified quiet window, so indicative only): `compile_aod` on the 400 x 340 scene, extraction through map, FORCE 78 ms, Zeit 73 ms; the 120 x 90 scene 13 ms and 12 ms. `python benchmarks/vs_originals/force_aod.py` (`--only-map` runs only this part).

### The topography vs FORCE

`topo-ll.c` of **FORCE 3.11.00** is in Zeit's core (`zeit_core::topo`; the public atmospheric correction comes later), and it is the `Top` that `compile_aod` reads. It is not `zeit.topographic_correction` (which corrects reflectance with the C and SCS+C methods from the papers, Horn's kernel in metres, and is unchanged): this is what FORCE's atmospheric correction derives from the DEM. `compile_topography` takes the DEM on the grid of the scene (FORCE warps the file with GDAL's `GDALWarpOperation`: resampling `DEM_RESAMPLING`, bilinear by default, nodata `DEM_NODATA`, -32767; no file means a flat DEM at 0 m; the warp is outside the core) and: grows the ocean (a no-data pixel next to a 0 m pixel becomes 0 m, scanning columns and rows in both directions), smooths the DEM with a box filter of `2k + 1` pixels, `k = 30 / res`, if the pixels are smaller than 30 m, computes slope and aspect (Horn 1981, in `float`, radians times 10000 as `ushort`; QAI flags: slope above 2 degrees, off where the 3 x 3 window has no data), the elevation statistics in km and the DEM binned in 255 classes (it fails outside -500 m to 9000 m), the illumination angle with the sun angles of the coarse cell under each pixel (QAI flag: deep shadow, poor, moderate) and the sky view factor. `cfactor_topography` derives the C-factor of the topographic correction on the SWIR2 band: the theoretical one (sky view, f-factor of the atmosphere) for flat pixels, the empirical one (the intercept over the slope of the regression of the SWIR2 reflectance on the illumination over a sparse neighbourhood of 3 km of sloped pixels of a similar SWIR index) where there are enough of them, then a 3 x 3 mean; `average_elevation_cell` averages the binned DEM over a coarse cell. The C harness (`force_topo/`) compiles FORCE's `topo-ll.c` as it is (but for its first include), with `radtran-ll.c`, behind minimal stand-ins for the brick, `atc` and parameter structures (no GSL, no GDAL); the functions of `quality-cl.c`, `brick_base-cl.c`, `utils-cl.c`, `stats-cl.c` and `atc-ll.c` it uses are copied verbatim; the warp of the DEM is a stub that hands over the DEM the harness holds.

The scenes are 14: a flat DEM (no file and zeros), a ramp, a cone and a valley with a low sun, mountains (sinusoids with noise) at 5, 10, 20, 30 and 250 m, a DEM with no-data holes and a frame, an ocean at 0 m with a no-data coast, a QAI with off pixels and flags already set, a QAI with nothing on, DEMs above 9 km and below -500 m (FORCE fails), and a large one (900 x 810 at 10 m with 3 km cells); the sun changes over the coarse grid, and the C-factor runs on a SWIR1 and SWIR2 made from the terrain and three land covers.

| What | Values | Equal | Differ | Largest difference |
|---|---:|---:|---:|---:|
| 14 scenes: return code, `atc->dem` (min, max, average, step; ulps), QAI afterwards, binned DEM, slope, aspect, illumination, sky view, C-factor, average elevation of the coarse cells | 11,620,508 | 11,620,508 | 0 | 0 |

The `atan`, `atan2`, `sqrt`, `cos` and `sin` of Windows' libm (Zeit) and of glibc 2.39 (FORCE, in WSL) gave the same bits in all of them. FORCE runs its loops with OpenMP and every one writes only its own pixel, except the sum of the elevation in `stats_topography` (`reduction(+)` of doubles): the last bits of the average elevation can depend on the number of threads (on the large scene, 4 threads against 1 gave every field equal, the average included); the harness runs with one thread and Zeit's sum is in pixel order. Timings (one thread, not in a verified quiet window, so indicative only): on the large scene `compile_topography` FORCE 71 ms, Zeit 84 ms; `cfactor_topography` 1,253 ms and 1,405 ms. `python benchmarks/vs_originals/force_topo.py`.

### The atmosphere of the coarse grid vs FORCE

`atc-ll.c`, `sunview-ll.c` and the functions of `atmo-ll.c` before the surface reflectance of **FORCE 3.11.00** are in Zeit's core (`zeit_core::atc`, `meta`, `sunview`, `atmo`; the public atmospheric correction comes later). `Atc::allocate` is `allocate_atc` (the whole `atc_t`: the Henyey-Greenstein terms, the environmental weighting, the wavelengths and solar irradiances, and the bricks of the coarse grid of 5 km cells, 256 elevation bins each for the families that depend on the elevation, with the band domains copied the way `copy_brick` does). `sun_target_view` computes the sun position and the view angles of every cell (Landsat: the distance to the nadir line fitted to the swath of the QAI by `viewgeo`, Sentinel-2: the grids of the metadata, with the pixels under a cell without data set off), the range of the cosines and the low sun flag. `atmo_angledep` gives, for the cells with a view angle, the phase functions, the gas transmittances (water vapour, ozone), the average elevation, the molecular optical depth at that elevation and the Fresnel reflection; `atmo_elevdep`, for each of the 256 elevation bins, the optical depths, the scattering transmittances, the path reflectance, the spherical albedo and the environmental weight, then the aerosol flag of the QAI; `interpolation_weights` and `interpolate_coarse` interpolate a coarse grid at a pixel without the cells that have no data; `apply_aoi` sets the pixels outside the area of interest off. The C harness (`force_atmo/`) compiles FORCE's `atc-ll.c` and `sunview-ll.c` as they are (but for their first include), copies the functions of `atmo-ll.c` verbatim (the rest of that file has the GDAL gridding and the level 2 compilation), with `radtran-ll.c`, `table-ll.c` and `date-cl.c`, behind minimal stand-ins for the brick, `meta` and parameter structures (no GSL, no GDAL).

Not FORCE's, and the same on both sides: the sun position (`sun-cl.c` is not opened; Zeit's `sunpos` is the clean-room `solpos.rs`, and the harness holds a table with the values Zeit gives for every coarse cell, looked up by the exact latitude and longitude, so the sun angles are not FORCE's bits), the warp of the map coordinates of a cell to longitude and latitude (GDAL in FORCE; an affine map here) and `brdf_factor` of `brdf-ll.c` (not ported: the factor is 1 for every band, which is what FORCE starts from; in Zeit the caller of `atmo_angledep` can pass a function). FORCE names the columns of the coarse grid `ne` and the rows `nf` in `atmo_angledep` and `atmo_elevdep` and takes the cell `e * nf + f`; on a grid with more columns than rows that takes the ozone's coordinates from a cell that is not the one written, and the port does the same (the scenes have both shapes).

The scenes are 13: Landsat at 100 m (800 x 700, a tilted swath), at 30 m (1200 x 1000, with an area of interest), 1000 x 400 and 400 x 1000 pixels (more columns than rows and the other way round), in December at 70 N (the sun below the horizon: 293,400 pixels with the low sun flag), in the south and west without an AOD map and the environmental weighting, with single scattering; Sentinel-2 at 20 m (1500 x 1200, three cells without view angles), at 10 m, at 60 m in two shapes, a whole tile (1830 x 1830 at 60 m: 22 x 22 cells, 13 bands) and a single coarse cell. Each has about 1,500 pixels for the interpolation weights on a coarse grid with cells without data (corners, edges, the centres of the cells, random pixels).

| What | Values | Equal | 1 ulp | More |
|---|---:|---:|---:|---:|
| 13 scenes: every field of `atc` after `allocate_atc`, `xy_sun`, `xy_view`, `cosszen`, `cosvzen`, `view`, the QAI after each stage, everything `atmo_angledep` and `atmo_elevdep` write, the interpolation weights and values, `apply_aoi` | 81,850,920 | 81,850,920 | 0 | 0 |

The `exp`, `log`, `sin`, `cos`, `tan`, `acos` and `atan2` of Windows' libm (Zeit) and of glibc 2.39 (FORCE, in WSL) gave the same bits in all of them. Every cell loop of FORCE writes only its own cell, so the result does not depend on the OpenMP threads (the harness runs with one); Zeit's are rayon loops over the cells and gave the same fields with 4 threads as with 1. Timings (one thread, not in a verified quiet window, so indicative only): on the whole tile `sun_target_view`, `atmo_angledep` and `atmo_elevdep` take FORCE 9 + 455 + 123 ms and Zeit 12 + 24 + 265 ms. `python benchmarks/vs_originals/force_atmo.py`.

### The surface reflectance and the Level 2 products vs FORCE

The rest of `atmo-ll.c` of **FORCE 3.11.00** is in Zeit's core (`zeit_core::l2`; the public atmospheric correction comes later): `surface_reflectance` (the BOA reflectance of a band from the coarse atmosphere interpolated to the pixel, with the environmental weighting of the adjacency effect, the topographic factor `A` and the BRDF factor), `background_reflectance` (the exponential moving averages of rings of 0.25, 1 and 4.5 km along the rows and the columns), `compile_l2_boa`, `compile_l2_qai`, `compile_l2_dst` (the distance to cloud, shadow and snow, `cloud_distance` of `cloud-ll.c`), `compile_l2_vzn` (the view zenith, the cells without data filled by GDAL's inverse distance weighting), `compile_l2_hot`, `compile_l2_aod`, `compile_l2_wvp`, `compile_level2` and `radiometric_correction`, which runs the stages in FORCE's order: the molecular optical depth of the scene, the water vapour (Landsat: the constant or the LUT; Sentinel-2: later), `atmo_angledep`, `compile_aod`, `atmo_elevdep`, `water_vapor` (Sentinel-2), `cfactor_topography`, `apply_aoi` and the products. The products are `short`s: the conversions are the ones gcc makes on x86-64 (`cvttsd2si` and the low 16 bits). Not ported: `compile_l2_ovv` (the quicklook), the warp of the products to the datacube (FORCE reprojects every product, and the coarse grid of the VZN, to the cube; here they stay on the grid of the scene) and `brdf_factor` of `brdf-ll.c` (the factor is a function the caller gives).

The C harness (`force_l2/`) is not a copy of functions: `main.c` is compiled against FORCE's own headers and linked with the object files that FORCE's `make` built for `force-l2ps` (`atmo-ll.o`, `aod-ll.o`, `gas-ll.o`, `topo-ll.o`, `atc-ll.o`, `sunview-ll.o`, `brick_base-cl.o`, `quality-cl.o`, ...), with the real GSL 2.7 (static, its reference CBLAS) and GDAL 3.9.3, so that `allocate_atc`, `sun_target_view`, `compile_topography` and `radiometric_correction` run as the L2PS runs them. Not FORCE's, and the same on both sides: the sun position (`sun-cl.o`'s `sunpos` is replaced by a table of the values Zeit's clean-room `sunpos` gives for every coarse cell), the map to longitude and latitude, the warp of the DEM file (the DEM the harness holds), `brdf_factor` (a factor of the view zenith and the band, in float) and the cloud flags (the QAI is given: FORCE's cloud detection is not ported). Like the parameter parser, a scene without `doatmo` has `doaod`, `dobrdf`, `dotopo` and `doenv` off (with `dotopo` and no atmosphere FORCE dereferences a C-factor it never computed and crashes).

The scenes are 13: Landsat at 30 m (800 x 700, all stages; AOD map), at 100 m (no environmental weighting, BRDF, `erase_cloud`, an area of interest), at 30 m without topographic correction and with single scattering, without the AOD estimation (the fallback), hazy (an AOD map with the high aerosol flag), without atmospheric correction (TOA), in December at 70 N with the sun below the horizon (flat DEM); Sentinel-2 at 20 m (1000 x 900, cells without view angles, an area of interest), at 10 m (BRDF, no environmental weighting), at 60 m in two shapes, at 20 m without atmospheric correction and a 1200 x 1200 tile (13 bands). The TOA reflectance comes from the radiative transfer for soil (three spectra by region), lakes (the spectra of FORCE's water library), dark dense vegetation (its vegetation library), opaque, unsure and cirrus cloud, cloud shadow and snow, the vapour band absorbed for a smooth field of 0.6 to 3.5 cm, with noise; the DEM has relief, flat lakes and an ocean patch. Sentinel-2 images are a multiple of 6 pixels in each direction (see below).

| What | Values | Equal | 1 ulp | More |
|---|---:|---:|---:|---:|
| atc after the correction: status, the water vapour, `Hr`, `Ha`, `Hp`, `aodmap`, the elevation statistics, `cosszen`, `cosvzen`, `mod`, `aod`, `od` per band | 696 | 696 | 0 | 0 |
| atc: the 18 coarse bricks (`xy_*`, all bands) | 76,752 | 76,752 | 0 | 0 |
| atc: 15 families of 256 elevation bricks (`xyz_*`) | 24,768,000 | 24,768,000 | 0 | 0 |
| QAI at the end (off, cloud, shadow, subzero, saturation, vapour fill, aerosol, slope, illumination) | 6,785,608 | 6,785,608 | 0 | 0 |
| BOA, every band (6 bands for Landsat, 10 for Sentinel-2) | 55,636,080 | 55,636,080 | 0 | 0 |
| DST, VZN, HOT, AOD, WVP (5,345,608 pixels each) | 26,728,040 | 26,728,040 | 0 | 0 |
| gaseous transmittance maps of the 13 bands (Sentinel-2) | 25,893,504 | 25,893,504 | 0 | 0 |
| product lists, domains and wavelengths | 363 | 363 | 0 | 0 |
| total | 139,889,050 | 139,889,050 | 0 | 0 |

The scenes carry 6.8 million pixels of QAI and 55.6 million BOA values, with 6,700 to 200,000 pixels off per scene, 1,000 to 100,000 of cloud, up to 18,000 with a reflectance below 0 and up to 13,000 above 1; the AOD map exists in 8 of the 13 scenes (the aerosol flag 1 where it was interpolated, 2 for the hazy scene, 3 for the fallback). The `exp`, `log`, `sin`, `cos`, `pow` and `atan2` of Windows' libm (Zeit) and of glibc 2.39 (FORCE, in WSL) gave the same bits in all of them; the IDW of the VZN and the AOD map executes `rcpps` on both sides, so the bits agree on a given CPU (a test with fixed numbers would depend on the CPU's table: none here). FORCE's loops write only their own pixel or row (the vertical pass of the background reflectance is sequential here); Zeit with 4 threads gave the same products as with 1.

Two things the comparison found. (1) With the sun below the horizon (the cosine of the zenith negative) `wvp_transmitt` of the C is NaN: it sums the whole table of 1,991 wavelengths and the terms outside the band's window (`pow` of a negative base) poison the sum; the port summed over the window only and gave a finite number; it now sums the whole table where the air mass is not positive and finite (the same NaN). (2) `gas_transmittance` and `water_vapor` of a Sentinel-2 image whose width is not a multiple of 6 write and read past the end of each row, into the first two columns of the next (and past the end of the array for the rows, which crashed the harness): the pixels of the first columns get the transmittance of the last block of the row above. A real Sentinel-2 image is a multiple of 6 pixels (1830, 5490, 10980), so the port still cuts the blocks at the border and the scenes are multiples of 6.

Timings (one thread, not in a verified quiet window, so indicative only): `radiometric_correction` takes FORCE 1.1 s and Zeit 0.9 s on the Landsat 30 m scene (800 x 700), 6.0 s and 2.2 s on the Sentinel-2 20 m one (1000 x 900), 7.9 s and 3.7 s on the 1200 x 1200 tile. `python benchmarks/vs_originals/force_l2.py`.

### The coregistration vs AROSICS

[`coregister`](../api/preprocessing.md#coregister) is a port of the global `COREG` of **AROSICS 1.14.0** (Scheffler et al. 2017; Apache-2.0), run here as it is in its own conda environment (`arosics-ref`, Python 3.14, GDAL 3.12, from conda-forge) by `benchmarks/vs_originals/arosics_driver.py`, on GeoTIFFs written by `benchmarks/vs_originals/arosics.py`, which then runs Zeit on the same files. AROSICS: `COREG(ref, tgt, ws=(256, 256), max_shift=5, max_iter=5, nodata=(nd, nd), align_grids=True, resamp_alg_deshift="cubic", ignore_errors=True)` and `calculate_spatial_shifts()`, then `correct_shifts()` for the aligned image. There is no satellite scene in the repository (the largest GeoTIFFs of `examples/data` are 25 x 25 pixels), so the pairs are synthetic, plus one of real image content: scikit-image's `camera` (512 x 512, 8 bit) with its copy moved by (2.3, -1.4) pixels (a cubic spline). The synthetic scenes are value noise (five octaves, 2-32 pixel features, 30 m cells) moved by known shifts.

| Pair | Size | Known shift (x, y) | Zeit's shift | AROSICS' shift | Difference |
|:---|:---|:---|:---|:---|:---|
| `synth_int` | 1024 x 1024 | (3, -2) | (3.000000, -2.000000) | (3.000000, -2.000000) | 1e-10 |
| `synth_sub` | 1024 x 1024 | (0.4, 1.7) | (0.285887, 1.783818) | the same | 9e-8 |
| `synth_neg` | 1024 x 1024 | (-4.25, 3.1) | (-4.157352, 3.070672) | the same | 1.1e-7 |
| `synth_zero` | 1024 x 1024 | (0, 0) | (0, 0) | (0, 0) | 6e-10 |
| `synth_margin_u16` (uint16, NoData 0; 300 columns and 200 rows of no data) | 1024 x 1024 | (2.5, -1.2) | (2.360820, -1.164250) | the same | 1.5e-7 |
| `synth_odd` | 1001 x 777 | (1.6, 0.7) | (1.710320, 0.768810) | the same | 6e-8 |
| `synth_small` (smaller than the window) | 200 x 200 | (1.3, -0.8) | (1.245660, -0.823640) | the same | 2.1e-7 |
| `synth_ws128` | 700 x 500 | (-2.2, 2.2) | (-2.121680, 2.124930) | the same | 6.7e-8 |
| `synth_diag_nodata` (a diagonal no-data edge) | 1024 x 1024 | (2, 1) | (2.000000, 1.000000) | the same | 8e-10 |
| `synth_4096` | 4096 x 4096 | (1.5, -2.5) | (1.585560, -2.393310) | the same | 1.2e-7 |
| `natural_camera` (uint8, NoData 0) | 512 x 512 | (2.3, -1.4) | (2.232950, -1.319590) | the same | 1.5e-8 |
| `synth_far` | 1024 x 1024 | (9, 0) | fails: above `max_shift` | fails: the same error | |
| `synth_flat` (constant) | 1024 x 1024 | (0, 0) | fails: window too small | fails: "matching window became too small" | |

(The subpixel part of a phase correlation is an estimate of a fraction, good to a tenth of a pixel on these scenes, not the known shift.) In all 13 pairs the two agree on whether the shift is found; on the 11 where it is, the integer part, the matching window (`[x0, y0, x1, y1]`, including the 200 x 200 image's, which AROSICS cuts to `[1, 1, 199, 199]`), the side of the final FFT window (128 where a whole-pixel shift was removed from the 256 window, 256 for the zero shift, 64 for the 128 window), the reliability (96.1-100%) and the structural similarity before and after the shift (0.9942-1.0000 after; the natural image 0.4858 to 0.9725) are equal, and the shifts differ by 2.1e-7 pixels at most: the single-precision noise of the two FFT libraries (scipy's pocketfft, rustfft), as both run the correlation in complex64. The aligned images (AROSICS' `correct_shifts()` resampled onto the same grid with GDAL's cubic, Zeit's `to_grid` with its port of the same kernel) differ by 0.00049 at most in float32 values of about 3000 (one unit in the last place), by 1 in the uint16 image (rounding), and are the same cells for whole-pixel shifts, over the pixels with data inside the 8 pixel border.

**Timings** (seconds, the best of 3; one pair at a time, with GDAL, OpenBLAS and OpenMP at 1 thread; **the machine was not idle**: other sessions' processes ran, 15-60% load before the runs, so these are indicative). AROSICS: `COREG(...)` plus `calculate_spatial_shifts()` (it reads the files, finds the footprints and overlap, takes the windows, correlates and measures the similarity). Zeit: the core's call on arrays that are already in memory, which is the same work minus reading the files, and the whole `zeit.coregister` of the date from `rioxarray`'s arrays, which also moves the whole image.

| Pair | AROSICS, `CPUs=1` | AROSICS, `CPUs=4` | Zeit, the core's estimate | Zeit, `coregister` (with the aligned image) |
|:---|---:|---:|---:|---:|
| `synth_sub` 1024 x 1024 | 0.038 | 0.040 | 0.011 | 0.039 |
| `synth_margin_u16` 1024 x 1024 | 0.080 | 0.080 | 0.011 | 0.035 |
| `synth_diag_nodata` 1024 x 1024 | 0.094 | 0.091 | 0.011 | 0.030 |
| `synth_4096` 4096 x 4096 | 0.174 | 0.178 | 0.035 | 0.323 |
| `natural_camera` 512 x 512 | 0.029 | 0.029 | 0.010 | 0.026 |

The estimate is 3 to 8 times faster in Zeit: AROSICS' cost is in the footprint polygon and the file reading, where Zeit's is the correlation of one window (about 10 ms of FFTs, similarities and the overlap of the two masks; it grows with the image only through that overlap, 35 ms at 4096 x 4096). `CPUs` makes no difference for AROSICS' estimate (it only parallelises the warps of `equalize_pixGrids` and `correct_shifts`, which a pair on one grid does not run), and a single date is a single thread in Zeit. The whole `zeit.coregister` of a 4096 x 4096 date is dominated by the cubic warp of its 16.8 million cells (0.29 s). **Eight dates of 2048 x 2048** (a cube, one reference): 0.51 s with `n_jobs=1` (64 ms per date) and 0.39 s with `n_jobs=4` (49 ms per date): the dates run at once in the correlation and the warp is the part that threads share the least.

### The local coregistration vs AROSICS

[`coregister_local`](../api/preprocessing.md#coregister_local) is a port of `COREG_LOCAL` of **AROSICS 1.14.0** (the tie point grid, the match of each point, `Tie_Point_Refiner`'s three filters and the warp by the control points), run as above in the `arosics-ref` environment by `arosics_driver.py local` (`COREG_LOCAL(ref, tgt, grid_res, window_size=(ws, ws), nodata=(nd, nd), align_grids=True, resamp_alg_deshift="cubic", tieP_filter_level=3)`, then `calculate_spatial_shifts()` and `correct_shifts()`; its `CoRegPoints_table` is written to JSON and the warped image to `.npy`), with `python benchmarks/vs_originals/arosics.py local run 1`, `... local run 4` and `... local compare`. The pairs are synthetic (the target is the reference's value-noise scene at the positions a smooth field of shifts moves it from: an affine part and a sinusoid, up to 3 pixels in `local_smooth`, `local_odd`, `local_2048` and `local_margin_u16`, up to 1.5 in `local_patches` and `local_ws128_fine`) plus scikit-image's `camera` warped by such a field (cubic spline, 8 bit). `local_patches` has three outliers planted in the target: a block of another texture, a featureless block and a block displaced 3 pixels more than the field (a good match that does not fit the model); `local_margin_u16` is uint16 with NoData 0, 300 columns and 150 rows of no data and a diagonal no-data edge.

Zeit runs `coregister_local(..., arosics_compat=True)` on the same files (the flag below) and its table is joined to AROSICS' on `POINT_ID`. Over the points both match:

| Pair | Size (cols x rows) | `grid_res` / `ws` | Points A/Z | Matched A/Z | Shifts (px) | Reliability | SSIM before / after | Windows equal | Control points A/Z |
|:---|:---|:---|:---|:---|---:|---:|---:|:---|:---|
| `local_smooth` | 1400 x 1200 | 256 / 256 | 30/30 | 30/30 | 1.4e-6 | 1.0e-5 | 5e-11 / 6e-6 | 30/30 | 27/27 |
| `local_patches` | 1400 x 1200 | 200 / 256 | 56/56 | 56/56 | 8.5e-7 | 4.3e-5 | 5e-11 / 4e-8 | 56/56 | 48/48 |
| `local_margin_u16` | 1100 x 1024 | 200 / 256 | 19/19 | 18/18 | 3.3e-7 | 5.1e-6 | 4e-9 / 5e-8 | 18/18 | 16/16 |
| `local_odd` | 777 x 1001 | 100 / 128 | 88/88 | 88/88 | 6.6e-7 | 2.8e-5 | 5e-11 / 2e-5 | 88/88 | 74/74 |
| `local_ws128_fine` | 700 x 900 | 64 / 128 | 165/165 | 164/164 | 5.4e-7 | 7.5e-5 | 5e-11 / 1e-5 | 164/164 | 147/147 |
| `local_2048` | 2048 x 2048 | 200 / 256 | 121/121 | 121/121 | 6.6e-7 | 1.1e-5 | 5e-11 / 8e-6 | 121/121 | 105/105 |
| `natural_camera_local` (uint8) | 512 x 512 | 64 / 128 | 81/81 | 74/74 | 2.0e-3 | 7.4e-6 | 4e-8 / 3e-4 | 74/74 | 67/67 |

(Largest absolute difference. The shifts of the natural image differ by 1.2e-7 at all points but one, at the top edge, where the 128 x 62 window's correlation peak moves by 2e-3: 8-bit data on a nearly degenerate window, the single-precision noise of the two FFT libraries. The similarity after the shift differs at the 1e-5 level in the float pairs and 3e-4 in the 8-bit image: GDAL's cubic of AROSICS' warp of the window against the port, at cells beside no data.)

**The points that are placed and the points that match are the same in all 7 pairs**, including the 62-pixel windows at the image edges and the window of the point (400, 200) of `local_margin_u16`, which AROSICS grows from the point to 162 x 164 pixels where the diagonal edge crosses its 256 window (x 319 to 481, y 118 to 282, the box of Zeit exactly; Zeit's first version, which only clipped the window, failed there: the branch of AROSICS that grows a box from the position, which the global comparison above never ran, is in the port now). Flags (the points that differ from AROSICS / flagged by AROSICS / flagged by Zeit, with `arosics_compat=True`):

| Pair | Reliability | SSIM | RANSAC | Any (`OUTLIER`) | Warped image, max / mean difference |
|:---|:---|:---|:---|:---|:---|
| `local_smooth` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 3 / 3 | 0 / 3 / 3 | 0.00098 / 2.7e-5 |
| `local_patches` | 0 / 1 / 1 | 0 / 1 / 1 | 0 / 6 / 6 | 0 / 8 / 8 | 0.00098 / 2.0e-5 |
| `local_margin_u16` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2 / 2 | 0 / 2 / 2 | 1 / 1.6e-5 |
| `local_odd` | 0 / 3 / 3 | 0 / 3 / 3 | 0 / 9 / 9 | 0 / 14 / 14 | 0.00049 / 1.1e-5 |
| `local_ws128_fine` | 0 / 0 / 0 | 0 / 1 / 1 | 0 / 16 / 16 | 0 / 17 / 17 | 0.00049 / 9.7e-6 |
| `local_2048` | 0 / 0 / 0 | 0 / 2 / 2 | 0 / 14 / 14 | 0 / 16 / 16 | 0.00049 / 1.5e-5 |
| `natural_camera_local` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 7 / 7 | 0 / 7 / 7 | 6 / 3.1e-4 |

The RANSAC flags are the same in every pair because the core reproduces scikit-image's `ransac` and numpy's `default_rng(0)`: the generator's first values and the samples of `Generator.choice(n, 6, replace=False)` are checked against numpy's output in the core's tests, and on the 54 points of `local_patches` the number of inliers at each of the 8 thresholds AROSICS tries (5, 7, ..., 19) is the same as AROSICS'. The warped images (GDAL's thin plate spline of the same control points, rasterio's GDAL 3.12 against the `arosics-ref` one) differ by 1e-3 at most in the float pairs (float32 rounding), by 1 in the uint16 pair, and by up to 6 in 0.01% of the cells of the 8-bit image, over the pixels with data inside the 8 pixel border; the valid cells are the same.

**AROSICS 1.14.0 puts RANSAC's flags on the wrong points.** Without `arosics_compat` (Zeit's default), the RANSAC flags of four pairs differ from AROSICS' (`local_patches` 8 points, `local_odd` 7, `local_ws128_fine` 26, `local_2048` 24; none in `local_smooth`, `local_margin_u16` and the natural image, where the reliability and similarity filters flagged nothing before it). Tracing AROSICS' `_RANSAC_outlier_detection` shows why: when the first two filters have removed points, the flags of RANSAC (one for each point it got, from 0) are assigned by `inGDF['outliers'] = outliers` to a table indexed by the points' positions in the whole table, so pandas aligns them by label and the k-th flag lands on the row labelled k, not on the k-th remaining point. The count flagged is right and the points are not. Zeit's default flags the points RANSAC found; `arosics_compat=True` reproduces AROSICS' assignment, to compare with it.

**Timings** (seconds; AROSICS the best of 3, Zeit a warm-up and the best of 2; **the machine was not idle**: another session's process was using about a core and the load was 15-20% before the runs, so these are indicative). AROSICS: `COREG_LOCAL(...)` plus `calculate_spatial_shifts()` (reads the files, the footprints, a `COREG` for each point in a `joblib` process, the filters), `CPUs=1` and `4`; its `correct_shifts()` (GDAL's warp) took 0.02-0.13 s. Zeit: the core's tie points on arrays in memory, `n_jobs=1` and `4`, and the whole `zeit.coregister_local` of the date, which warps as well (rasterio's GDAL, one thread):

| Pair | Points | AROSICS `CPUs=1` | AROSICS `CPUs=4` | Zeit tie points, 1 thread | Zeit tie points, 4 threads | `coregister_local`, 1 thread | `coregister_local`, 4 threads |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `local_smooth` | 30 | 2.10 | 1.97 | 0.250 | 0.095 | 0.366 | 0.216 |
| `local_patches` | 56 | 1.77 | 1.42 | 0.389 | 0.129 | 0.517 | 0.263 |
| `local_margin_u16` | 19 | 0.77 | 0.69 | 0.139 | 0.046 | 0.203 | 0.118 |
| `local_odd` | 88 | 1.80 | 1.64 | 0.185 | 0.068 | 0.259 | 0.151 |
| `local_ws128_fine` | 165 | 1.99 | 1.55 | 0.326 | 0.104 | 0.424 | 0.200 |
| `local_2048` | 121 | 5.25 | 4.24 | 1.023 | 0.351 | 1.419 | 0.757 |
| `natural_camera_local` | 81 | 1.82 | 1.64 | 0.151 | 0.063 | 0.184 | 0.094 |

The tie points are 4.5 to 12 times faster in Zeit on one thread against AROSICS on one CPU and 11 to 26 times on 4 threads against 4 CPUs (the window's FFTs and similarities cost the same in both; AROSICS also builds a `COREG` object with its footprints for each point and starts a pool of processes, and its 4 CPUs gain 6 to 22%, where the threads of the core gain a factor of 2.4 to 3.1 at 4). The warp of Zeit's `coregister_local` is the rest (0.05 to 0.4 s).

---

## Protocol

- **The same inputs.** Each script draws its inputs with a fixed seed, writes them to a temporary folder (CSV or raw float64), runs the original on the files from `Rscript`, and runs Zeit on the same arrays in memory.
- **The original's own timing.** Each R script (`benchmarks/vs_originals/<method>.R`) times only the call it benchmarks with `system.time` (elapsed), after reading the inputs and loading the packages, and prints it; starting R, loading the packages and reading and writing files are not counted. Each original is timed once, with its defaults otherwise, in one R process.
- **Zeit's timing.** A warm-up call, then the best of three (`best_of` in `benchmarks/vs_originals/common.py`, `time.perf_counter`), on 1 thread and on all the processors but one (`n_jobs=-1`: 19 here; the AR(1) trends, PARTS and the spatial cross-validation, timed later, on 8).
- **The check.** After timing, each script compares the outputs (the tolerances above) and stops with an error if they disagree, so no timing is recorded for a result that differs.
- **The record.** `record` in `common.py` writes one entry per method, original, workload and thread count to `benchmarks/results/vs-originals.json`, with the date and the machine, replacing an older entry of the same key.

### Machine

| | |
|:---|:---|
| Platform | Windows 11 (`Windows-11-10.0.26300-SP0`) |
| Processor | `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` (the Intel Core i5-13600K of the other benchmarks), 20 logical processors |
| R | 4.4.2, with sfdep 0.2.5, RecoverMetricsR 0.1.0, RStoolbox 1.0.2.2, sits 1.5.4, CAST 1.0.4, Rbeast 1.0.2, remotePARTS 1.0.4, blockCV 4.0.1 (with automap 1.1.20), scam 1.2.23 and survey 4.5 |
| Python | pyxccd 1.1.0, in a virtual environment of its own |
| Date | 2026-10-24 (the AR(1) trends and PARTS: 2026-10-28; the spatial cross-validation: 2026-10-29; the transitions: 2026-11-03; ImproPhe: 2026-11-06; the resolution merge: 2026-11-07; the coregistration: 2026-11-07) |

---

## Rerunning the benchmarks

From the repository root, with Zeit installed and R with the packages above:

```bash
python benchmarks/vs_originals/hotspots.py            # sides 20, 50, 100 (or give your own: ... hotspots.py 20 50)
python benchmarks/vs_originals/recovery.py
python benchmarks/vs_originals/unmix.py
python benchmarks/vs_originals/smooth_probs.py
python benchmarks/vs_originals/aoa.py
python benchmarks/vs_originals/beast.py
python benchmarks/vs_originals/parts.py               # sides 50, 100, 200; ZEIT_BENCH_THREADS=8 by default
python benchmarks/vs_originals/folds.py               # autocor knndm profile mahalanobis (CAST's MD takes 15 min)
python benchmarks/vs_originals/accuracy_survey.py     # zeit.transitions against survey (and terra)
python benchmarks/vs_originals/force_tsa.py --part stm,fold,standardize,interp,trend,cso,polar --jobs 4   # FORCE's TSA, against its saved outputs
wsl -e bash -lc 'bash benchmarks/vs_originals/force_tsa_timing.sh ~/force-ref-work/tsa-bench 6'   # FORCE's compute stage on the 500 x 500 cube (also with 1); then:
python benchmarks/vs_originals/force_tsa_timing.py    # Zeit's times against it (--profile: the cost by size)
python benchmarks/vs_originals/tasseled.py            # RStoolbox 1.0.2.2, against zeit.tasseled_cap
wsl -e bash -lc 'bash benchmarks/vs_originals/force_improphe_l2fix.sh'                      # once: the build of FORCE that L2IMP needs (one core)
wsl -e bash -lc 'bash benchmarks/vs_originals/force_improphe.sh ~/force-ref-work/imp-ref 6'  # CFIMP and L2IMP, each with OpenBLAS (A) and GSL's CBLAS (B); then:
python benchmarks/vs_originals/force_improphe.py --part cf,l2                              # zeit.improphe against both (--data: another cube)
wsl -e bash -lc 'bash benchmarks/vs_originals/force_improphe_timing.sh ~/force-ref-work/imp-bench 1'   # and with 6; then:
python benchmarks/vs_originals/force_improphe_timing.py                                    # Zeit's times against FORCE's compute stage
wsl -e bash -lc 'bash benchmarks/vs_originals/force_resmerge/make_harness.sh'                  # FORCE's resolution merge, verbatim, on GSL 2.7 (static); then:
python benchmarks/vs_originals/force_resmerge.py --size 500   # zeit.sentinel2_resmerge against it, and the times
wsl -e bash -lc 'bash benchmarks/vs_originals/force_parallax/make_harness.sh'                  # FORCE's cloud_parallax and buffer_, verbatim; then:
python benchmarks/vs_originals/force_parallax.py --size 512 --big 2048   # zeit.parallax_clouds and zeit.buffer_mask against it, and the times
wsl -e bash -lc 'bash benchmarks/vs_originals/force_radtran/make_harness.sh'                  # FORCE's radiative transfer (radtran-ll.c, table-ll.c, sun-cl.c as they are, for the sun position's reference); then:
python benchmarks/vs_originals/force_radtran.py   # zeit_core::radtran against it, bit for bit
wsl -e bash -lc 'bash benchmarks/vs_originals/force_gas/make_harness.sh'                  # FORCE's gas-ll.c as it is, on GSL 2.7 (static); then:
python benchmarks/vs_originals/force_gas.py   # zeit_core::gas against it, bit for bit, one OpenMP thread
wsl -e bash -lc 'bash benchmarks/vs_originals/force_aod/make_harness.sh'                  # FORCE's aod-ll.c as it is, on GSL 2.7 (static) and GDAL 3.9.3; then:
python benchmarks/vs_originals/force_aod.py   # zeit_core::aod against it, bit for bit, one OpenMP thread
wsl -e bash -lc 'bash benchmarks/vs_originals/force_topo/make_harness.sh'                 # FORCE's topo-ll.c as it is; then:
python benchmarks/vs_originals/force_topo.py  # zeit_core::topo against it, bit for bit, one OpenMP thread
wsl -e bash -lc 'bash benchmarks/vs_originals/force_atmo/make_harness.sh'                 # FORCE's atc-ll.c and sunview-ll.c as they are, atmo-ll.c's functions verbatim; then:
python benchmarks/vs_originals/force_atmo.py  # zeit_core::atc, sunview and atmo against it, bit for bit, one OpenMP thread
wsl -e bash -lc 'bash benchmarks/vs_originals/force_l2/make_harness.sh'                   # FORCE's radiometric_correction as it is (the objects of force-l2ps), on GSL 2.7 (static) and GDAL 3.9.3; then:
python benchmarks/vs_originals/force_l2.py    # zeit_core::l2 against it, bit for bit, one OpenMP thread
python benchmarks/vs_originals/topocorr.py            # gdaldem (GDAL_BIN) and terra/RStoolbox 1.0.2.2, against zeit.topographic_correction
python benchmarks/vs_originals/arosics.py            # AROSICS 1.14.0 (conda env arosics-ref; AROSICS_ENV=...), against zeit.coregister
python benchmarks/vs_originals/arosics.py local run 1   # COREG_LOCAL with CPUs=1 and with 4 (`local run 4`), then zeit.coregister_local against both:
python benchmarks/vs_originals/arosics.py local compare
python benchmarks/vs_originals/polar.py               # PolarMetrics 0.1.0 (R package from GitHub: R CMD INSTALL), against zeit.polar_metrics
python benchmarks/vs_originals/ewmacd.py "path/to/EWMACD v1.3.0.r"   # Brooks' script, not distributed
PYXCCD_PYTHON=path/to/pyxccd-env/python python benchmarks/vs_originals/sccd.py CCDC_Stack_Rondonia.tif CCDC_Stack_Rondonia_Dates.csv
```

The scripts look for `Rscript` at `C:\Program Files\R\R-4.4.2\bin\Rscript.exe` on Windows and on the `PATH` elsewhere; set the `RSCRIPT` environment variable to use another. Brooks' script is in `EWMACD v1.3.0.7z` at [VTechWorks](https://vtechworks.lib.vt.edu/items/version/7). `hotspots.py` runs sfdep with 199 permutations only up to 20 × 20 cells (the 50 × 50 run above took sfdep 13 minutes), and with 9 on every grid. Each script prints its rows as it records them; the chart above is redrawn from the JSON by `python docs/scripts/make_vs_originals_chart.py`.

!!! note "What is not here"
    These are the benchmarks of the Zeit library. Janus, the desktop application built on it, runs the same core; the comparison of its tools with the originals is kept with Janus, not on this page. The earlier ports (LandTrendr, CCDC, BFAST, TWDTW, SOM, the deep learning models...) are compared in [Performance](performance.md) and [Fidelity](fidelity.md).
