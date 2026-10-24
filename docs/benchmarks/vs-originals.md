# Against the Original Implementations

Eight of Zeit's newer methods were checked against published implementations: emerging hot spots, recovery metrics, EWMACD, spectral unmixing, Bayesian smoothing of class probabilities, the area of applicability and BEAST against their R originals, and S-CCD against pyxccd, its authors' Python package. This page times each port against that original on the same inputs, and records how the results were checked to agree. Every number comes from `benchmarks/results/vs-originals.json`, written by the scripts in `benchmarks/vs_originals/`, one per method.

---

## At a glance

<div class="bm-chart-card">
<p class="bm-chart-title">Time of the original and of Zeit, on the largest workload of each benchmark</p>
<p class="bm-chart-sub">Log scale. The originals are timed inside R (pyxccd inside Python), without loading R and its packages. Zeit is the best of three runs. Hover over (or tab to) a row for the exact numbers and the agreement.</p>
<div class="bm-chart-legend">
<span><i style="background:var(--bm-ref)"></i>Original</span>
<span><i style="background:var(--bm-gpu-lo)"></i>Zeit, 1 thread</span>
<span><i style="background:var(--bm-zeit)"></i>Zeit, 19 threads</span>
</div>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/vs_originals.svg"
</div>
</div>

On one thread, Zeit is 3× (the area of applicability with 8 predictors) to 30,000× (emerging hot spots) faster than the original, and the outputs compared agree to floating-point rounding (for the unmixing, to the tolerance at which the original's solver stops). Compiled loops against R's per-pixel ones account for most of the gain on one thread; the threads multiply it where there is enough work to share. BEAST is the exception: its original is already C, multithreaded and vectorized, and Zeit, a port of its generic kernels, is 1.5× to 1.9× faster on a cube and slower on one series.

---

## Results

Times of the call alone (see [Protocol](#protocol)); "1 thread" and "19 threads" are Zeit's, and the speedups are the original's time over Zeit's. The agreement is checked by the script on every run, which fails if it does not hold.

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

[`zeit.sccd`](../api/change-detection.md#sccd) against **pyxccd 1.1.0**'s `sccd_detect`, the implementation of S-CCD's authors (Ye et al. 2021; Apache-2.0, which GPL-2.0-only Zeit cannot take code from: Zeit's S-CCD is written from the paper, and pyxccd is run as a black box in an environment of its own). pyxccd is called pixel by pixel, as its API does. Two workloads: 4096 pixels of a Landsat Collection 2 stack of Rondonia (2020–2026, the 61 dates of the path these pixels are on), and 1000 synthetic pixels of 35 years every 8 days, a third of the dates cloudy and a third of the pixels with a change: the long series S-CCD was made for. The same pixels also go through Zeit's own [`ccdc`](../api/change-detection.md#ccdc), to time S-CCD's claim, a Kalman update per observation instead of a lasso refit.

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

`zeit.smooth_probs` against the R package **sits 1.5.4**: the Bayesian smoothing that `sits_smooth` applies to each block (`.smooth_fn_bayes`: the logits, `bayes_smoother_fraction`, back to probabilities), on probabilities of 4 classes, a 7 × 7 window, `neigh_fraction` 0.5 and `smoothness` 20.

| Workload | sits | Zeit, 1 thread | Zeit, 19 threads | Speedup (1 / 19) | Agreement |
|:---|---:|---:|---:|:---:|:---|
| 200 × 200 px × 4 classes | 0.40 s | 46 ms | 5.4 ms | 9× / 74× | max diff 6.7e-16 |
| 1000 × 1000 px × 4 classes | 10.7 s | 1.13 s | 147 ms | 9× / 73× | max diff 7.8e-16 |

**How they were checked.** Every smoothed probability, to 1e-12 (the largest difference was below 1e-15). sits is timed on its smoothing of the block only, without reading and writing its files, so the comparison is of the computation. sits gives a missing pixel its neighbours' mean where Zeit keeps it missing; the inputs have none.

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

---

## Protocol

- **The same inputs.** Each script draws its inputs with a fixed seed, writes them to a temporary folder (CSV or raw float64), runs the original on the files from `Rscript`, and runs Zeit on the same arrays in memory.
- **The original's own timing.** Each R script (`benchmarks/vs_originals/<method>.R`) times only the call it benchmarks with `system.time` (elapsed), after reading the inputs and loading the packages, and prints it; starting R, loading the packages and reading and writing files are not counted. Each original is timed once, with its defaults otherwise, in one R process.
- **Zeit's timing.** A warm-up call, then the best of three (`best_of` in `benchmarks/vs_originals/common.py`, `time.perf_counter`), on 1 thread and on all the processors but one (`n_jobs=-1`: 19 here).
- **The check.** After timing, each script compares the outputs (the tolerances above) and stops with an error if they disagree, so no timing is recorded for a result that differs.
- **The record.** `record` in `common.py` writes one entry per method, original, workload and thread count to `benchmarks/results/vs-originals.json`, with the date and the machine, replacing an older entry of the same key.

### Machine

| | |
|:---|:---|
| Platform | Windows 11 (`Windows-11-10.0.26300-SP0`) |
| Processor | `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel` (the Intel Core i5-13600K of the other benchmarks), 20 logical processors |
| R | 4.4.2, with sfdep 0.2.5, RecoverMetricsR 0.1.0, RStoolbox 1.0.2.2, sits 1.5.4, CAST 1.0.4 and Rbeast 1.0.2 |
| Python | pyxccd 1.1.0, in a virtual environment of its own |
| Date | 2026-10-24 |

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
python benchmarks/vs_originals/ewmacd.py "path/to/EWMACD v1.3.0.r"   # Brooks' script, not distributed
PYXCCD_PYTHON=path/to/pyxccd-env/python python benchmarks/vs_originals/sccd.py CCDC_Stack_Rondonia.tif CCDC_Stack_Rondonia_Dates.csv
```

The scripts look for `Rscript` at `C:\Program Files\R\R-4.4.2\bin\Rscript.exe` on Windows and on the `PATH` elsewhere; set the `RSCRIPT` environment variable to use another. Brooks' script is in `EWMACD v1.3.0.7z` at [VTechWorks](https://vtechworks.lib.vt.edu/items/version/7). `hotspots.py` runs sfdep with 199 permutations only up to 20 × 20 cells (the 50 × 50 run above took sfdep 13 minutes), and with 9 on every grid. Each script prints its rows as it records them; the chart above is redrawn from the JSON by `python docs/scripts/make_vs_originals_chart.py`.

!!! note "What is not here"
    These are the benchmarks of the Zeit library. Janus, the desktop application built on it, runs the same core; the comparison of its tools with the originals is kept with Janus, not on this page. The earlier ports (LandTrendr, CCDC, BFAST, TWDTW, SOM, the deep learning models...) are compared in [Performance](performance.md) and [Fidelity](fidelity.md), on the C++ core of Zeit 1.x.
