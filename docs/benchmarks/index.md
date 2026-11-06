# Benchmarks & Validation Suite

Welcome to the **Zeit Fidelity & Performance Validation Suite**. This section provides a comprehensive empirical evaluation of the algorithms implemented in Zeit, assessing both **scientific fidelity** (how faithfully Zeit reproduces the original tools and published papers it ports) and **computational performance** (single-core speedup and multi-core CPU scaling).

---

## Executive Summary & Mission

Earth Observation (EO) time-series analysis has historically suffered from fragmented software ecosystems. Canonical algorithms were originally authored across diverse, often proprietary or single-threaded environments:

- **LandTrendr** was authored in proprietary **IDL** (Kennedy *et al.* 2010).
- **CCDC** was authored in **MATLAB** with a compiled Fortran GLMnet lasso (Zhu & Woodcock 2014).
- **BFAST, Phenology, and TWDTW** gained widespread adoption through **R** packages (`bfast`, `phenofit`, `dtwSat`/`twdtw`).
- **Deep learning models** for satellite time-series (TempCNN, LightTAE, U-TAE) were scattered between R `sits` and standalone PyTorch repositories; Zeit's TempCNN and LightTAE are written from the papers and the authors' code.
- Researchers have frequently relied on cloud platforms like **Google Earth Engine (GEE)**, trading off local control, customization, and queue times for scalability.

**Zeit** was engineered to unify these algorithms into a modern, cloud-native architecture written in a **high-performance compiled core** (C++ with `Eigen3` and OpenMP up to 1.7, **Rust** from 2.0) and **PyTorch** (`zeit.ai`), exposed through Python, Dask, and Xarray. The compiled core's timings in this section were taken with the Rust core of zeit 2.3, which gives the same results as the C++ one (see [Performance](performance.md)).

### The Validation Mandate

To verify that Zeit can serve as a drop-in, scientifically reliable replacement for these reference implementations, we subjected Zeit to a rigorous battery of **13 benchmark comparisons** (plus one bonus architectural verification). Every algorithm was evaluated against the **authentic reference tool** across millions of observations, testing:

1. **Numerical and Statistical Fidelity:** Do Zeit outputs match the reference tools down to floating-point tolerance, identical breakpoint positions, and matching model categories?
2. **Execution Efficiency:** How much faster is Zeit on a single core, how effectively does it scale across threads, and how does local desktop processing compare against distributed cloud platforms like Google Earth Engine?

---

## Key Performance Indicators

<div class="bm-kpis">
  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Tested Algorithms</div>
    <div class="bm-kpi-value">13 + 1</div>
    <div class="bm-kpi-sub">13 canonical tools + 1 bonus segmentation port (U-TAE)</div>
  </div>

  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Algorithm Parity</div>
    <div class="bm-kpi-value">100%</div>
    <div class="bm-kpi-sub">Exact vertex/label/codebook match on LandTrendr, SNIC, SOM, TWDTW; CCDC with identical segment dates and categories</div>
  </div>

  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Phenology, Start of Season</div>
    <div class="bm-kpi-value">4.1 d</div>
    <div class="bm-kpi-sub">90th-percentile error of Beck's TRS5 on a synthetic cube with known truth (10.7 d in Zeit 1.x)</div>
  </div>

  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Max Single-Core Speedup</div>
    <div class="bm-kpi-value">30,000×</div>
    <div class="bm-kpi-sub">The fastest of the newer ports against its R original (see Against the Originals)</div>
  </div>

  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Vs. Google Earth Engine</div>
    <div class="bm-kpi-value">19.4×</div>
    <div class="bm-kpi-sub">Compute throughput on full 54.7M px Landsat tile, 8 threads (0 queue delay)</div>
  </div>
</div>

---

## Speed and Fidelity at a Glance

Every port is held to two standards at once: it must give the reference tool's answer, and it must give it faster. The chart lines both up for each algorithm. Hover over (or tab to) a row for the exact numbers; the full matrix follows below.

<div class="bm-chart-card">
<p class="bm-chart-title">Single-core speedup over the reference tool, and how closely Zeit reproduces it</p>
<p class="bm-chart-sub">Same task, one thread on each side. Log scale: each grid line is ten times the one before. A light extension shows the range over several workloads.</p>
<div class="bm-chart-legend">
<span><i style="background:var(--bm-zeit)"></i>Zeit faster</span>
<span><i style="background:var(--bm-zeit);opacity:.3"></i>Range across workloads</span>
<span><i style="background:var(--bm-ref)"></i>Zeit slower</span>
</div>
<div class="bm-chart-legend">
<span><i class="pill" style="background:var(--zeit-teal)"></i>Identical output</span>
<span><i class="pill" style="background:var(--zeit-teal-soft);box-shadow:inset 0 0 0 1px var(--zeit-teal)"></i>Equal within 1e-7</span>
<span><i class="pill" style="background:var(--bm-grid)"></i>Close: same method, small numerical differences</span>
</div>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/speedup_overview.svg"
</div>
</div>

Thirteen of the sixteen ports return the reference tool's own output, bit for bit or to floating-point rounding, and the fifteen measured on a single core are all faster before any parallelism (U-TAE by 8%, the Siamese detector by 7–16%, Presto by 17%). The three "close" ports fit curves or search for breaks with numerical optimisers, where tiny differences in the starting point or in the stopping rule move a date by a few days ([Fidelity](fidelity.md) has the details). The deep-learning ports run the same PyTorch operations as their references, so they gain little; Prithvi, measured on 14 threads and on the GPU, ties with its reference.

The ports added since (emerging hot spots, recovery metrics, EWMACD, unmixing, Bayesian smoothing, area of applicability) are timed against their R originals with the Rust core on a page of their own: [Against the Original Implementations](vs-originals.md).

---

## Comparison Matrix at a Glance

The table below summarizes the scope of our validation suite. Each comparison reflects reproducible benchmarks logged in the Zeit testing repository.

| Algorithm | Reference Implementation | Language / Engine | Test Scope | Primary Fidelity Metric | Single-Core Speedup | Status |
|:---|:---|:---|:---|:---|:---:|:---:|
| **LandTrendr** | Kennedy *et al.* (2010) `LandTrendr-2012` | Original IDL (via GDL) | 330 synthetic series + 54.7M px tile | **100%** identical vertex years (330/330) | **232×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **CCDC** | Zhu & Woodcock (2014) GERSL `CCDC` | Original MATLAB (GNU Octave) | 200 synthetic + 150 real Landsat px | **100%** identical dates, categories and observation counts; coefficients within test tolerances | **218×–260×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **SNIC (Superpixels)** | Achanta & Süsstrunk (2017) CVPR | C reference (`snic.c`) | Reference test fixtures (f32/f64) | **100%** identical segment labels (bitwise) | **1.8×–3.5×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **SOM** | Vettigli `minisom` (online & batch Kohonen SOM) | Python (`minisom`) | 1,500 + 20,000 samples, 6 scenarios | **max weight diff = 0.0** (exact codebook match) | **41×–67×** (training) | <span class="bm-pill bm-pill--compared">compared</span> |
| **TWDTW** | Maus *et al.* (2016) `dtwSat` / `twdtw` | R (`twdtw` C core) | 45 multi-class temporal series | **Distances identical** to R (1e-12) | **71.5×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Mann-Kendall** | `pymannkendall` (3 variants) | Python (pure Python) | 270 scenarios (90 series × 3 variants) | **100%** trend agreement (mean diff p-val = 0.0) | **65×–75×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **BFAST Monitor** | Verbesselt *et al.* (2012) `bfast` | R (`bfast::bfastmonitor`) | 40 scenarios (breaks, noise, NaN gaps) | **100%** break agreement (mag. corr = 1.0) | **16.9×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Phenology (Beck)** | Zeit 1.x (derived from `phenofit`); the truth of a synthetic generator | Python / Rust (Zeit) | Synthetic cube, 2,000 px, 5 years of 16 days, clouds, weights and gaps | **Start of season: median 1.50 d, 90th percentile 4.12 d** (Zeit 1.x: 2.43 / 10.7 d) | Same or faster than Zeit 1.x (Beck 0.25 s, 2,000 px, 4 jobs); not compared with `phenofit` | <span class="bm-pill bm-pill--compared">compared</span> |
| **TempCNN** | Pelletier *et al.* (2019), the authors' code | R `sits` (accuracy only) | `samples_modis_ndvi`, 12 dates, 4 classes, 5 seeds | **accuracy 0.861 ± 0.024** (`sits`: 0.879 ± 0.027) | Not compared | <span class="bm-pill bm-pill--compared">compared</span> |
| **LightTAE (LTAE)** | Sainte Fare Garnot & Landrieu (2020), the authors' code | R `sits` (accuracy only) | `samples_modis_ndvi`, 12 dates, 4 classes, 5 seeds | **accuracy 0.915 ± 0.011** (`sits`: 0.902 ± 0.016) | Not compared | <span class="bm-pill bm-pill--compared">compared</span> |
| **BFAST (Classic)** | Verbesselt *et al.* (2010) `bfast` | R (`bfast::bfast`) | 40 scenarios (trend/season breaks) | **87.5%** break count match (100% pos. match) | **13.2×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **BFAST Lite** | Jan Hackman *et al.* `bfast` | R (`bfast::bfastlite`) | 40 scenarios (single/multi breaks) | **100%** break count match (80% exact index) | **5.2×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Official U-TAE** | Garnot & Landrieu (2021) `utae-paps` | Python (official PyTorch repo) | Regular and padded sequence paths | **max abs diff = 0.0** (exact numerical match) | **1.08×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Siamese CNN (FC-Siam-diff)** | Daudt *et al.* (2018) `fully_convolutional_change_detection` | Python (official PyTorch code) | 40 same-weight cases (odd sizes too) + 5 training seeds | **max abs diff = 5.6e-16** (float64); trained F1 0.979 vs 0.971 | **1.07×–1.16×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Presto** | Tseng *et al.* (2023) `nasaharvest/presto` | Python (official PyTorch code) | 2,560 pixel series in 5 cases (clouds, missing sensors, 3–24 dates) | **max abs diff = 1.2e-6** (identical in 3 of 5 cases) | **1.17×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Prithvi-EO-2.0** | Szwarcman *et al.* (2024) `prithvi_mae.py` | Python (PyTorch + `timm`) | All six models (tiny-TL to 600M-TL), 5 cases each (dates and place, 1–4 dates, 112–224 px) | **max relative diff = 7.9e-7** (encoder tokens) | ≈1× (14 threads) | <span class="bm-pill bm-pill--compared">compared</span> |

!!! note "Status Definitions"
    - **`compared`**: Exact numerical or statistical benchmark against the official reference tool or canonical implementation.

---

## Hardware & Test Environment

All performance benchmarks were measured on a modern workstation environment:

<div class="bm-machine-box" style="grid-template-columns: 1fr;">
  <div class="bm-hw-spec">
    <div class="bm-hw-tag" style="color: var(--bm-zeit);">
      <span class="bm-hw-dot" style="background: var(--bm-zeit);"></span>
      Benchmark Workstation (Intel Core i5)
    </div>
    <ul>
      <li><strong>CPU:</strong> Intel Core i5-13600K (14 physical cores: 6 P-cores + 8 E-cores, 20 logical threads)</li>
      <li><strong>RAM:</strong> 64 GB DDR5</li>
      <li><strong>Operating System:</strong> Windows 11 Pro 64-bit</li>
      <li><strong>R Environment:</strong> R 4.4.2 (CRAN `bfast`, `phenofit`, `twdtw`, `torch` 0.17.0, `sits` 1.5.2)</li>
      <li><strong>Python Environment:</strong> Python 3.12.8, PyTorch 2.14.0, Zeit 0.18.0+</li>
      <li><strong>GDL / Octave:</strong> GDL 1.1.2 (native Windows), GNU Octave 11.3 (compiled Fortran GLMnet)</li>
    </ul>
    <p class="bm-kpi-sub">SOM (re-validated in Zeit 0.21.0 after the bit-exact <code>minisom</code> port) was measured on an AMD Ryzen 7 7730U laptop (8 cores / 16 threads, Windows 11, Python 3.10, NumPy 1.26, <code>minisom</code> 2.3.6). Its speedups are ratios measured on that same machine.</p>
  </div>
</div>

---

## Key Scientific & Engineering Insights

### 1. Parity Demands Meticulous Porting, Not High-Level Mimicry

Empirical validation demonstrates that reproducing published remote sensing algorithms requires rigorous line-by-line fidelity to author reference implementations rather than high-level approximations:

- **LandTrendr:** Replicating Kennedy *et al.* (2010) with 100% vertex year agreement (330/330 series) required reproducing subtle source behaviors: fitting on the despiked series, in-place `take_out_weakest2` mutation, whole-ladder Levenberg-Marquardt fallback, flat-line non-significance fallback, and single-precision F-test p-value tie-breaking.
- **CCDC:** Matching Zhu & Woodcock (2014) across 350 pixels (599 models, 249 breaks) required MATLAB `datenum` coordinate alignment and bisquare Tmask outlier detection. The lasso, once a port of the Fortran GLMnet, is now written from the paper (Friedman, Hastie & Tibshirani 2010): segment dates, categories and observation counts stay exact, and the coefficients agree within the tolerances of the parity test.
- **SNIC:** Achieving bit-for-bit label parity with Achanta & Süsstrunk (2017) required reproducing priority queue heap tie-breaking order, while resolving upstream heap underflow crashes on edge-case inputs.
- **SOM:** Reproducing Python `minisom` codebooks bit-for-bit required matching its `RandomState` draw sequence, NumPy's pairwise summation inside the distance norm, `argmin` tie-breaking on square-rooted distances, and sample-order accumulation in the batch update. An earlier Zeit variant that only *resembled* the batch algorithm (different decay, no learning rate, epoch-based `num_iters`) agreed with `minisom` at just ARI = 0.39 and ran 6–10× slower on the same call; the exact port is identical and 41–67× faster on one core.

Across all families, Zeit delivers **100% bitwise and statistical parity** with the authoritative reference implementations.

### 2. High-Performance Local Processing Beats Cloud Queues

The prevailing assumption in remote sensing has been that massive spatial workflows require distributed cloud infrastructure like Google Earth Engine. Our full Landsat tile benchmark demonstrates otherwise:

- **Zeit processing 54,731,482 pixels (41 years)** on a single desktop PC completed in **103.92 seconds** (526,693 pixels/sec) on 8 of its 20 threads.
- **Google Earth Engine's distributed cluster** executed active server compute on ~38M pixels in **1,396 seconds** (27,213 pixels/sec), with an additional **5.08 hours of cloud queue latency**.
- Zeit delivered **19.4× higher compute throughput** than GEE's server cluster, with zero queue overhead and zero network egress costs.

<div class="bm-chart-card">
<p class="bm-chart-title">LandTrendr on a full Landsat tile: a desktop PC against Google Earth Engine</p>
<p class="bm-chart-sub">54.7 million pixels × 41 years on an Intel Core i5-13600K; about 38 million active pixels on Earth Engine's batch workers. Top: throughput while computing. Bottom: the time from submitting the job to having the result.</p>
<div class="bm-chart-legend">
<span><i style="background:var(--bm-zeit)"></i>Zeit, 8 threads</span>
<span><i style="background:var(--bm-gpu-lo)"></i>Zeit, 1 thread</span>
<span><i style="background:var(--bm-ref)"></i>Google Earth Engine</span>
<span><i style="background:var(--bm-grid)"></i>Waiting in Earth Engine's queue</span>
</div>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/gee_tile.svg"
</div>
</div>

When both sides run in parallel, the gap stays wide for the statistical tools, whose reference implementations pay for each worker with an interpreter, and closes for the deep-learning models, which run on the same tensor libraries:

<div class="bm-chart-card">
<p class="bm-chart-title">Who finishes first in parallel</p>
<p class="bm-chart-sub">Wall-clock time of Zeit with 8 threads (20 for the neural networks, which run on PyTorch) and of the reference with its own parallel framework (19 R or Python workers, or intra-op threads for the neural networks). Each row is its own workload; log scale.</p>
<div class="bm-chart-legend">
<span><i class="dot" style="background:var(--bm-zeit)"></i>Zeit, 8 threads</span>
<span><i class="dot" style="background:var(--bm-ref)"></i>Reference, all workers</span>
</div>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/finish_first.svg"
</div>
</div>

### 3. Phenology: written from the papers

Zeit's phenology is written from the papers (Whittaker: Eilers 2003; HANTS: Roerink et al. 2000; the curves of Beck 2006, Elmore 2012, Gu 2009, Klosterman 2014, Zhang 2003, Jönsson & Eklundh, Fisher 2006), not from the code of R `phenofit`. R `phenofit` remains a useful reference to compare outputs with, but the earlier comparison on the real EVI raster no longer applies and was not redone, nor was the speed comparison with it.

What was measured is the error against a known truth: a synthetic cube of 2,000 pixels (5 years of 16 days, half with one season a year and half with two, with clouds, weights and gaps), the truth being the 50% crossings of the generator. The error of TRS5 at the start of season in the single-season pixels, median / 90th percentile in days:

| Curve | Zeit 1.x (derived from `phenofit`) | Current Zeit |
|:---|---:|---:|
| Beck | 2.43 / 10.7 | **1.50 / 4.12** |
| Elmore | 1.71 / 7.28 | **1.51 / 4.23** |
| Gu | 1.90 / 8.55 | **1.80 / 4.99** |
| Klosterman | 1.94 / 8.84 | **1.74 / 4.97** |
| Zhang | 1.71 / 7.51 | **1.50 / 4.12** |
| Asymmetric Gaussian | 1.95 / 10.5 | **1.70 / 5.15** |
| Double logistic | 1.73 / 8.39 | **1.49 / 4.15** |

The end of season stays at the same level (Beck: 1.43 / 4.63 d). With 4 jobs, on the cube, Beck takes 0.25 s, Gu 0.68 s and Klosterman 0.79 s (Zeit 1.x: 0.27, 1.58 and 3.56 s); on the real EVI raster (638 pixels, 575 dates), Beck takes 0.30 s and Gu 0.95 s.
---

## Explore the Deep Dives

To examine the detailed evidence, methodology, and visualizations, navigate to the dedicated benchmark pages:

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } **Algorithm Fidelity & Accuracy**

    ---

    Detailed test batteries, root-cause parity analyses, verifications against the authors' code, and per-metric accuracy breakdowns for LandTrendr, CCDC, BFAST, Phenology, TWDTW, SOM, and Deep Learning.

    [:octicons-arrow-right-24: Read Algorithm Fidelity](fidelity.md)

-   :material-speedometer:{ .lg .middle } **Performance & Parallel Scaling**

    ---

    Interactive log-scale speedup charts, multi-core CPU scaling tables, the Google Earth Engine Landsat tile benchmark, and absolute execution times.

    [:octicons-arrow-right-24: Read Performance & Scaling](performance.md)

-   :material-scale-balance:{ .lg .middle } **Against the Original Implementations**

    ---

    The newer ports timed against their R originals on the same inputs (sfdep, RecoverMetricsR, Brooks' EWMACD, RStoolbox, CAST): 3× to 30,000× faster on one thread, the outputs checked to agree on every run.

    [:octicons-arrow-right-24: Read Against the Originals](vs-originals.md)

</div>
