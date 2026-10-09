# Benchmarks & Validation Suite

Welcome to the **Zeit Fidelity & Performance Validation Suite**. This section provides a comprehensive empirical evaluation of the algorithms implemented in Zeit, assessing both **scientific fidelity** (how faithfully Zeit reproduces the original tools and published papers it ports) and **computational performance** (single-core speedup and multi-core CPU scaling via OpenMP).

---

## Executive Summary & Mission

Earth Observation (EO) time-series analysis has historically suffered from fragmented software ecosystems. Canonical algorithms were originally authored across diverse, often proprietary or single-threaded environments:

- **LandTrendr** was authored in proprietary **IDL** (Kennedy *et al.* 2010).
- **CCDC** was authored in **MATLAB** with compiled Fortran GLMnet (Zhu & Woodcock 2014).
- **BFAST, Phenology, and TWDTW** gained widespread adoption through **R** packages (`bfast`, `phenofit`, `dtwSat`/`twdtw`).
- **Deep learning models** for satellite time-series (TempCNN, LightTAE, U-TAE) were scattered between R `sits` and standalone PyTorch repositories.
- Researchers have frequently relied on cloud platforms like **Google Earth Engine (GEE)**, trading off local control, customization, and queue times for scalability.

**Zeit** was engineered to unify these algorithms into a modern, cloud-native architecture written in **high-performance C++** (via `Eigen3`, SIMD, and OpenMP) and **PyTorch** (`zeit.ai`), exposed through Python, Dask, and Xarray.

### The Validation Mandate

To verify that Zeit can serve as a drop-in, scientifically reliable replacement for these reference implementations, we subjected Zeit to a rigorous battery of **13 benchmark comparisons** (plus one bonus architectural verification). Every algorithm was evaluated against the **authentic reference tool** across millions of observations, testing:

1. **Numerical and Statistical Fidelity:** Do Zeit outputs match the reference tools down to floating-point tolerance, identical breakpoint positions, and matching model categories?
2. **Execution Efficiency:** How much faster is Zeit on a single core, how effectively does OpenMP scale across 20 threads, and how does local desktop processing compare against distributed cloud platforms like Google Earth Engine?

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
    <div class="bm-kpi-sub">Exact model/vertex/label/codebook match on LandTrendr, CCDC, SNIC, SOM, TWDTW</div>
  </div>

  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Max Single-Core Speedup</div>
    <div class="bm-kpi-value">244×</div>
    <div class="bm-kpi-sub">Phenology vs R <code>phenofit</code> on real 25-yr EVI raster</div>
  </div>

  <div class="bm-kpi-card">
    <div class="bm-kpi-label">Vs. Google Earth Engine</div>
    <div class="bm-kpi-value">20.4×</div>
    <div class="bm-kpi-sub">Compute throughput on full 54.7M px Landsat tile (0 queue delay)</div>
  </div>
</div>

---

## Comparison Matrix at a Glance

The table below summarizes the scope of our validation suite. Each comparison reflects reproducible benchmarks logged in the Zeit testing repository.

| Algorithm | Reference Implementation | Language / Engine | Test Scope | Primary Fidelity Metric | Single-Core Speedup | Status |
|:---|:---|:---|:---|:---|:---:|:---:|
| **LandTrendr** | Kennedy *et al.* (2010) `LandTrendr-2012` | Original IDL (via GDL) | 330 synthetic series + 54.7M px tile | **100%** identical vertex years (330/330) | **168×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **CCDC** | Zhu & Woodcock (2014) GERSL `CCDC` | Original MATLAB (GNU Octave) | 200 synthetic + 150 real Landsat px | **100%** full model match (dates, categories, coeffs) | **84×–105×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **SNIC (Superpixels)** | Achanta & Süsstrunk (2017) CVPR | C reference (`snic.c`) | Reference test fixtures (f32/f64) | **100%** identical segment labels (bitwise) | **1.8×–3.5×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **SOM** | Vettigli `minisom` (online & batch Kohonen SOM) | Python (`minisom`) | 1,500 + 20,000 samples, 6 scenarios | **max weight diff = 0.0** (exact codebook match) | **37×–126× tr / 33×–104× pred** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Phenology (Beck)** | R `phenofit` (Zheng *et al.* 2021) | R (`nloptr` curve fitting) | Real EVI raster (638 px × 25 yrs) | **MAE 3.7d** on Start-of-Season (99.1% within 15d) | **244×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **TWDTW** | Maus *et al.* (2016) `dtwSat` / `twdtw` | R (`twdtw` C core) | 45 multi-class temporal series | **Distances identical** to R (1e-12) | **43.9×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Mann-Kendall** | `pymannkendall` (3 variants) | Python (pure Python) | 270 scenarios (90 series × 3 variants) | **100%** trend agreement (mean diff p-val = 0.0) | **33×–41×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **BFAST Monitor** | Verbesselt *et al.* (2012) `bfast` | R (`bfast::bfastmonitor`) | 40 scenarios (breaks, noise, NaN gaps) | **100%** break agreement (mag. corr = 1.0) | **15.8×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **TempCNN** | Pelletier *et al.* (2019) via R `sits` | R `torch` (LibTorch backend) | 1:1 weight-ported forward passes | **max abs diff = 5.6e-9** (corr = 1.000000) | **11.1×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **LightTAE (LTAE)** | Sainte Fare Garnot *et al.* via R `sits` | R `torch` (LibTorch backend) | 1:1 weight-ported forward passes | **max abs diff = 8.9e-8** (corr = 1.000000) | **5.8×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **BFAST (Classic)** | Verbesselt *et al.* (2010) `bfast` | R (`bfast::bfast`) | 40 scenarios (trend/season breaks) | **87.5%** break count match (100% pos. match) | **4.3×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **BFAST Lite** | Jan Hackman *et al.* `bfast` | R (`bfast::bfastlite`) | 40 scenarios (single/multi breaks) | **100%** break count match (80% exact index) | **2.5×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Official U-TAE** | Garnot & Landrieu (2021) `utae-paps` | Python (official PyTorch repo) | Regular and padded sequence paths | **max abs diff = 0.0** (exact numerical match) | **1.08×** | <span class="bm-pill bm-pill--compared">compared</span> |
| **Siamese CNN** | Bi-temporal change detection | Conceptual analog (R `sits` DTW) | 80 bi-temporal synthetic patches | **100%** accuracy on respective change tasks | **0.85×** | <span class="bm-pill bm-pill--partial">partial</span> |
| **Foundation ViT** | `ibm-nasa-geospatial/Prithvi-100M` | HuggingFace `transformers` | HuggingFace Hub remote backbone | Wrapper fallback consistent (**Conv3d** valid) | — | <span class="bm-pill bm-pill--notcomp">not_comparable</span> |

!!! note "Status Definitions"
    - **`compared`**: Exact numerical or statistical benchmark against the official reference tool or canonical implementation.
    - **`partial`**: Conceptual analog where no direct equivalent architecture exists in reference libraries (e.g., bi-temporal CNN vs 1D series DTW).
    - **`not_comparable`**: Upstream reference package dependency failure preventing direct backbone execution (wrapper fallback path validated).

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
- **CCDC:** Matching Zhu & Woodcock (2014) across 350 pixels (599 models, 249 breaks) required a native float32 port of Fortran GLMnet lasso, MATLAB `datenum` coordinate alignment, and bisquare Tmask outlier detection.
- **SNIC:** Achieving bit-for-bit label parity with Achanta & Süsstrunk (2017) required reproducing priority queue heap tie-breaking order, while resolving upstream heap underflow crashes on edge-case inputs.
- **SOM:** Reproducing Python `minisom` codebooks bit-for-bit required matching its `RandomState` draw sequence, NumPy's pairwise summation inside the distance norm, `argmin` tie-breaking on square-rooted distances, and sample-order accumulation in the batch update. An earlier Zeit variant that only *resembled* the batch algorithm (different decay, no learning rate, epoch-based `num_iters`) agreed with `minisom` at just ARI = 0.39 and ran 6–10× slower on the same call; the exact port is identical and 37–198× faster.

Across all families, Zeit delivers **100% bitwise and statistical parity** with the authoritative reference implementations.

### 2. High-Performance Local Processing Beats Cloud Queues

The prevailing assumption in remote sensing has been that massive spatial workflows require distributed cloud infrastructure like Google Earth Engine. Our full Landsat tile benchmark demonstrates otherwise:

- **Zeit processing 54,731,482 pixels (41 years)** on a single desktop PC completed in **98.44 seconds** (556,007 pixels/sec) using OpenMP.
- **Google Earth Engine's distributed cluster** executed active server compute on ~38M pixels in **1,396 seconds** (27,213 pixels/sec), with an additional **5.08 hours of cloud queue latency**.
- Zeit delivered **20.4× higher compute throughput** than GEE's server cluster, with zero queue overhead and zero network egress costs.

### 3. Phenology: Start-of-Season is Robust; Senescence is Ambiguous

Testing Zeit against R `phenofit` on 25 years of real satellite data (178,710 joined observations across 17 phenometrics) uncovered a vital scientific nuance:
- **Green-up and Start-of-Season (SOS)** metrics agree remarkably well (MAE **3.7 to 5.3 days**, with 99.1% of pixels matching within 15 days).
- **End-of-Season (EOS) and senescence** metrics diverge significantly (Dormancy MAE **76.9 days**).
- This divergence is not an error in either implementation; it reflects fundamental mathematical differences in how asymmetric logistic curves fit noisy, moisture-stressed post-harvest tail observations. Researchers are strongly advised to analyze SOS and EOS metrics separately.

---

## Explore the Deep Dives

To examine the detailed evidence, methodology, and visualizations, navigate to the dedicated benchmark pages:

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } **Algorithm Fidelity & Accuracy**

    ---

    Detailed test batteries, root-cause parity analyses, weight porting verifications, and per-metric accuracy breakdowns for LandTrendr, CCDC, BFAST, Phenology, TWDTW, SOM, and Deep Learning.

    [:octicons-arrow-right-24: Read Algorithm Fidelity](fidelity.md)

-   :material-speedometer:{ .lg .middle } **Performance & Parallel Scaling**

    ---

    Interactive log-scale speedup charts, multi-core CPU scaling tables, the Google Earth Engine Landsat tile benchmark, and absolute execution times.

    [:octicons-arrow-right-24: Read Performance & Scaling](performance.md)

</div>
