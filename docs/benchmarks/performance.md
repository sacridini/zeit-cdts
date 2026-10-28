# Performance & Parallel CPU Scaling

This page documents the **computational performance** of `zeit` across single-core CPU execution, multi-core CPU scaling, and distributed cloud comparison (Google Earth Engine), measured on an **Intel Core i5-13600K** workstation (20 logical threads).

!!! note "Measured with the Rust core of zeit 2.3"
    Zeit's side of every table on this page was measured with the Rust core of zeit 2.3, in October 2026, on the same workloads as before. Two kinds of rows keep their earlier numbers: the deep-learning models (TempCNN, LightTAE, U-TAE), which run on PyTorch and not on the compiled core, and SNIC, whose published workload could not be rebuilt exactly. The reference tools (R, GDL, Octave, `snic.c`) were timed on this same machine in September 2026 and not again; `minisom`, first timed on another computer, was timed again here. The multi-thread runs use **8 of the 20 logical processors**, leaving the rest to the machine; the reference tools' parallel runs use 19 workers. The scripts are in `benchmarks/vs_references/` and the results in `benchmarks/results/vs-references-rust.json`.

---

## The Compiled Core, Module by Module

Every module of the Rust core timed alone, on deterministic synthetic data, the way Zeit's functions call it: on 1 thread (pinned to a performance core) and on 8 of the 20 logical processors. The methods added since 2.0 are marked ¹. Best of at least three runs after a warm-up, zeit 2.3, `python benchmarks/bench_core.py --threads 8` (the results in `benchmarks/results/win-i5-13600k-rust-2.3.json`).

| Module | Workload | 1 thread | 8 threads | Gain |
|:---|:---|---:|---:|:---:|
| LandTrendr | 150 × 150 px × 40 years | 238 ms | 34.0 ms | 7.0× |
| CCDC | 16 × 16 px × 6 bands × 457 dates | 110 ms | 15.5 ms | 7.1× |
| TWDTW | 100 × 100 px × 46 dates, 1 pattern | 15.4 ms | 2.25 ms | 6.9× |
| SOM, batch training | 20,000 × 8, 10 × 10, 20 iterations | 180 ms | 28.7 ms | 6.3× |
| SOM, winners | 200,000 × 8, 10 × 10 | 46.2 ms | 8.12 ms | 5.7× |
| SOM, online training | 20,000 steps (sequential) | 12.0 ms | – | – |
| Phenology | 300 series × 137 dates | 79.1 ms | 11.5 ms | 6.9× |
| Mann-Kendall | 50,000 series × 40 | 291 ms | 40.2 ms | 7.2× |
| BFAST Monitor | 20,000 series × 184 | 69.5 ms | 10.1 ms | 6.9× |
| BFAST Lite | 200 series × 138 | 717 ms | 112 ms | 6.4× |
| BFAST | 200 series × 138 | 175 ms | 25.4 ms | 6.9× |
| SNIC | 1024 × 1024 × 4 bands, 4096 seeds | 120 ms | 20.7 ms | 5.8× |
| Unmixing (SMA) | 200,000 px × 6 bands, 4 endmembers | 245 ms | 35.2 ms | 6.9× |
| CODED | 20,000 px × 260 dates | 40.8 ms | 6.09 ms | 6.7× |
| Tmask | 10,000 px × 274 dates | 634 ms | 91.8 ms | 6.9× |
| Whittaker | 100,000 series × 219 | 632 ms | 93.0 ms | 6.8× |
| Medoid | 256 × 256 px × 24 dates × 6 bands | – | 9.51 ms | – |
| AR(1) trend ¹ | 100,000 series × 30 | 151 ms | 22.5 ms | 6.7× |
| PARTS ¹ | 10,000 px, 20 partitions of 500 | 843 ms | 164 ms | 5.2× |
| Recovery metrics ¹ | 1000 × 1000 px × 21 years | 69.8 ms | 35.3 ms | 2.0× |
| EWMACD ¹ | 20,000 px × 120 dates | 140 ms | 22.5 ms | 6.2× |
| Emerging hot spots ¹ | 300 × 300 cells × 12 dates | 158 ms | 40.1 ms | 3.9× |
| BEAST ¹ | 10 series × 138, 3 chains of 8000 samples | 1,141 ms | 213 ms | 5.4× |
| S-CCD ¹ | 32 × 32 px × 6 bands × 457 dates | 157 ms | 23.2 ms | 6.8× |
| Area of applicability, training DI ¹ | 5000 points × 8 predictors, 10 folds | 87.7 ms | 12.5 ms | 7.0× |
| Area of applicability, new points ¹ | 100,000 × 8 against 2000, with LPD | 904 ms | 135 ms | 6.7× |
| Bayesian smoothing ¹ | 500 × 500 px × 4 classes, 9 × 9 window | 365 ms | 55.8 ms | 6.5× |
| Majority filter ¹ | 2000 × 2000, 3 × 3 window | 87.9 ms | 52.3 ms | 1.7× |
| Sieve ¹ | 2000 × 2000, 11 px (sequential) | 183 ms | – | – |
| Temporal consistency ¹ | 1,000,000 series × 20 | 74.5 ms | 18.6 ms | 4.0× |
| Warp, bilinear | 4 × 1024 × 1024 float32 | 53.3 ms | 7.79 ms | 6.8× |
| Warp, Lanczos | 3 × 1000 × 1000 uint8 | 97.6 ms | 14.2 ms | 6.9× |
| Warp, average | 3 × 700 × 700 from 2000 × 2000 int16 | 58.0 ms | 8.56 ms | 6.8× |

Most modules gain 6–7× on 8 threads. The recovery metrics, the majority filter, the temporal consistency and the emerging hot spots gain least: their work per pixel is a few comparisons or sums, so reading and writing memory, not the processors, sets their pace. The sieve and SOM's online training are sequential by nature.

!!! note "Windows' power throttling"
    Windows 11 moves a process it deems in the background, as a benchmark run from a terminal is, to the efficiency cores at a low clock after a few seconds of work (power throttling, or EcoQoS). One-thread timings came out up to 2.4 times slower for it. The benchmark scripts take their processes out of power throttling, so the times are of the computation.

---

## Single-Core Speedup

Single-core execution evaluates algorithmic efficiency without the confounding factor of multi-threading. Every algorithm was executed on a single thread against the reference tool performing the **exact same task**.

<div class="bm-chart-card">
<p class="bm-chart-title">Single-core speedup over the reference tool</p>
<p class="bm-chart-sub">Log scale; the dark line at 1× is break-even. LandTrendr (232×) and CCDC (218×) time the algorithm inside the interpreter (interpreter start-up and compilation excluded). The fidelity column is explained in the <a href="../#speed-and-fidelity-at-a-glance">overview</a>. Hover over a row for the exact numbers.</p>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/speedup_overview.svg"
</div>
</div>

### Single-Core Benchmark Data

| Algorithm | Reference Package | Input Scale | Zeit Time (ms) | Reference Time (ms) | Speedup | Architectural Rationale |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Phenology** | R `phenofit` | Real EVI raster (638 px) | **2.22 ms / px** | 823.50 ms / px | **370.3×** | Compiled Levenberg-Marquardt fits vs R's non-linear `nloptr` interpreter overhead |
| **LandTrendr** | Original IDL (GDL) | 30-year synthetic series | **0.0187 ms / call** | 4.33 ms / call | **231.7×** | Compiled segmented regression vs GDL interpreter execution (warm second pass) |
| **CCDC** | Original MATLAB (Octave) | Landsat C2 pixel stack | **0.650 ms / px** | 141.51 ms / px | **217.9×** | Compiled float32 GLMnet lasso vs Octave runtime dispatch (real 150-pixel stack: 259.9×) |
| **TWDTW** | R `twdtw` / `dtwSat` | 40-step temporal series | **0.0180 ms / call** | 1.29 ms / call | **71.5×** | Pre-allocated dynamic programming tables vs R data.frame boxing |
| **Mann-Kendall** | Python `pymannkendall` | 30-obs series (3 tests) | **0.0111 ms / call** | 0.77 ms / call | **68.9×** | Compiled accumulator loops vs pure Python nested loops (the variants: 64.6×–75.0×) |
| **SOM (Training)** | Python `minisom` | 1,500 samples, 5×5 grid (500 online iters) | **0.104 ms / fit** | 6.92 ms / fit | **66.6×** | Bit-exact port of `MiniSom.train`; per-sample Python/NumPy dispatch replaced by a native loop (batch SOM: 41.5× on 1 thread) |
| **BFAST Monitor** | R `bfast::bfastmonitor` | 100-obs satellite series | **0.0870 ms / call** | 1.47 ms / call | **16.9×** | Closed-form recursive OLS update vs R formula interface |
| **BFAST (Classic)**| R `bfast::bfast` | Multi-year decomposition | **7.62 ms / call** | 100.67 ms / call | **13.2×** | Compiled iterative STL decomposition vs R `ts` object overhead |
| **TempCNN** † | R `sits::sits_tempcnn` | 4 bands, 24 timesteps | **0.212 ms / pass** | 2.34 ms / pass | **11.1×** | PyTorch (LibTorch) vs R `torch` Lantern bridge |
| **LightTAE (LTAE)** †| R `sits::sits_lighttae` | 4 bands, 16-head attention | **0.736 ms / pass** | 4.28 ms / pass | **5.8×** | Efficient spatial-temporal pooling vs R memory copies |
| **BFAST Lite** | R `bfast::bfastlite` | Structural break series | **4.29 ms / call** | 22.22 ms / call | **5.2×** | Compiled segmented linear regression vs R vector dispatch |
| **SNIC (Superpixels)** † | Achanta & Süsstrunk (2017) C | Multi-spectral image (512×512×4) | **18.4 ms / pass** | 42.3 ms / pass | **2.3×** | Contiguous pixel-major distances vs the reference C's float64 copy |
| **Official U-TAE** † | Official `utae-paps` repo | Segmentation patch 32×32 | **9.26 ms / pass** | 9.96 ms / pass | **1.08×** | Both run on PyTorch CPU backend (parity validation) |

† Earlier measurements, kept: the deep-learning models run on PyTorch, not on the compiled core, and SNIC's workload could not be rebuilt exactly (see the note at the top).

---

## Multi-Core CPU Scaling

Zeit implements parallel execution in its compiled core (`n_jobs`, the Rust core's own threads), avoiding Python Global Interpreter Lock (GIL) contention and inter-process communication (IPC) overhead. Going from 1 thread to 8 on an Intel Core i5-13600K, most modules run 6–7× faster (the per-module gains are in [the table above](#the-compiled-core-module-by-module)).

### Relative Gain vs. Absolute Time

A common benchmarking pitfall is looking only at relative gain (T₁ / T_multi). Reference tools running in R or Python often show high relative scaling simply because their single-threaded baselines are slow.

The table below lines up **absolute wall-clock times** to show who finishes first when both sides run in parallel. Zeit uses 8 of the machine's 20 logical processors; the reference tools use 19 workers:

<div class="bm-chart-card">
<div class="bm-chart-legend">
<span><i class="dot" style="background:var(--bm-zeit)"></i>Zeit, 8 threads</span>
<span><i class="dot" style="background:var(--bm-ref)"></i>Reference, 19 workers</span>
</div>
<div class="bm-chart-scroll">
--8<-- "docs/assets/charts/finish_first.svg"
</div>
</div>

| Algorithm | Workload Scale | Reference Parallel Framework | Zeit (1 Thread) | Zeit (8 Threads) | Zeit Gain | Reference (1 Thread) | Reference (Multi-Worker) | Reference Gain | **Who Finishes First (Multi-Core)** |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LandTrendr** | **54.7M px** × 41 yrs | — (GDL has no native parallel) | 643.35 s | **103.92 s** | **6.19×** | — | — | — | **Zeit finishes tile in 1.7 min** |
| **CCDC** | **2.83M px** × 125 dates | — (Octave has no native parallel) | 147.09 s | **28.50 s** | **5.16×** | — | — | — | **Zeit finishes stack in 28.5 s** |
| **SNIC (Tiled)** † | 2048×2048 × 6 bands (16 tiles) | — (`snic.c` has no parallel mode) | 482 ms | **59 ms** (20 threads) | **8.17×** | — | — | — | **Zeit finishes scene in 59 ms** |
| **Phenology** | 638 px × 25 yrs | R `foreach` + `doParallel` (19w) | 1,158 ms | **209 ms** | **5.54×** | 525,395 ms | 93,504 ms | 5.62× | **Zeit 447× faster** (0.21s vs 93.5s) |
| **Mann-Kendall** | 5,000 series | Python `multiprocessing.Pool` (19w) | 17.65 ms | **3.62 ms** | **4.87×** | 4,058 ms | 20,109 ms | 0.20× | **Zeit 5,552× faster** (3.6ms vs 20.1s) |
| **BFAST Monitor**| 20,000 series | R `parallel` PSOCK cluster (19w) | 63.16 ms | **10.03 ms** | **6.30×** | 26,872 ms | 5,003 ms | 5.37× | **Zeit 499× faster** (10.0ms vs 5.0s) |
| **BFAST Lite** | 2,000 series | R `parallel` PSOCK cluster (19w) | 8,595 ms | **1,485 ms** | **5.79×** | 42,463 ms | 4,551 ms | 9.33× | **Zeit 3.06× faster** (1.5s vs 4.6s) |
| **BFAST (Classic)**| 400 series | R `parallel` PSOCK cluster (19w) | 1,413 ms | **239 ms** | **5.91×** | 25,119 ms | 3,335 ms | 7.53× | **Zeit 13.9× faster** (0.24s vs 3.3s) |
| **TWDTW** | 3,000 series × 1 pat | R `parallel` PSOCK cluster (19w) | 10.16 ms | **2.13 ms** | **4.78×** | 1,406 ms | 188.67 ms | 7.45× | **Zeit 88.8× faster** (2.1ms vs 188.7ms) |
| **SOM (Batch)** | 20,000 samples, 10×10, 200 iters | — (`minisom` has no parallel mode)| 1,530 ms | **256 ms** | **5.98×** | 63.5 s | — | — | **Zeit 248× faster** (0.26 s vs `minisom` `train_batch_offline` on 1 thread, 63.5 s) |
| **TempCNN** † | Batch = 512 | R `torch` intra-op threads | 15.68 ms | **3.94 ms** | **3.98×** | 15.06 ms | 7.78 ms | 1.94× | **Zeit 1.97× faster** (3.9ms vs 7.8ms) |
| **Official U-TAE** † | Batch = 32 | PyTorch CPU intra-op threads | 317.66 ms | **126.00 ms** | **2.52×** | 324.40 ms | 124.67 ms | 2.60× | **Tied** (both PyTorch backend) |
| **LightTAE** † | Batch = 512 | R `torch` intra-op threads | 32.77 ms | **14.61 ms** | **2.24×** | 42.08 ms | 26.98 ms | 1.56× | **Zeit 1.85× faster** (14.6ms vs 27.0ms) |

† Earlier measurements on 20 threads, kept (see the note at the top).

!!! warning "The Python IPC Penalty on Mann-Kendall"
    On a batch of 5,000 short time series, running Python's standard `multiprocessing.Pool(19)` caused execution time to explode from **4.06 seconds** (sequential) to **20.11 seconds** (parallel) — a **5× slowdown** (0.20× speedup). The operating system overhead of pickling objects, IPC socket transfers, and process synchronization dwarfed the actual statistical computation. In contrast, Zeit's threads share memory with sub-microsecond synchronization overhead, achieving a true **4.87× speedup** on 8 threads (down to **3.62 milliseconds**).

---

## Google Earth Engine vs. Local Desktop Benchmark

A central design goal of Zeit is eliminating cloud lock-in for regional and continental-scale analyses. We evaluated LandTrendr performance on an entire Landsat tile:

- **Workload:** Full Landsat Tile 215_067 (South America), covering **54,731,482 pixels × 41 annual observations** (1985–2025).
- **Environment:** Intel Core i5-13600K (desktop PC, 20 logical processors, 64 GB RAM); Zeit on 1 thread and on 8.
- **Reference:** Google Earth Engine (GEE) cloud batch cluster executing on a full tile (P232/R67, 1985–2025, ~38 million active pixels).

<div class="bm-chart-card">
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

### Full Landsat Tile Comparison

| Processing Platform | Execution Mode | Active Compute Time | Throughput | Cloud Queue Wait Time | Total Turnaround Time | Speedup vs. GEE |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Zeit (Local PC)** | **Multi-Thread (8 threads)** | **103.92 s** (1.73 min) | **526,693 px / s** | **0.0 s** | **1.73 min** | **19.4× compute throughput** |
| **Zeit (Local PC)** | Single-Thread (1 core) | 643.35 s (10.72 min) | 85,072 px / s | 0.0 s | 10.72 min | **3.13× compute throughput** |
| **Google Earth Engine** | Cloud Cluster (EECU batch workers) | 1,396.39 s (23.27 min) | 27,213 px / s | 18,308 s (5.08 hours) | 5.47 hours | 1.0× baseline |

!!! success "Takeaways from the GEE Benchmark"
    1. **Throughput Advantage:** Zeit on a single standard desktop workstation achieves **19.4× the active compute throughput** of GEE's cloud server cluster, on 8 of its 20 logical processors.
    2. **Even Single-Thread Beats GEE:** Running Zeit on a single core (85,072 px/s) is **3.13× faster** than GEE's distributed cluster active execution.
    3. **Zero Queue Latency:** Under standard GEE quotas, large batch export tasks regularly wait hours in the queue (5.08 hours in this benchmark). Zeit begins processing immediately and finishes in under two minutes.



---

## Earth Engine Download Throughput (`download_gee_image`)

`download_gee_image(method='direct'/'auto')` was rewritten on ideas from [geedim](https://github.com/leftfield-geospatial/geedim) (fixed pixel grid, byte-sized tiles, windowed writes), with four changes of its own: tiles are split in space before bands, tiles are sized on demand from the concurrency Earth Engine actually allows (guided scheduling), concurrency adapts to HTTP 429 responses, and requests go through `computePixels` in a single round trip. We compared it with the previous implementation (0.25° tiles, 4 threads, `getDownloadURL` + `rasterio.merge`) and with geedim 2.0.0.

- **Region:** WRS-2 path/row 217/076 (Rio de Janeiro). `sub`: 0.5° × 0.5° (1856 × 1857 px at 30 m); `small`: 0.2° × 0.2°.
- **Images:** 2020 annual medoid composite (6 bands, float32, 79 MB raw); NDVI of that composite (1 band, 13 MB raw); dense stack of Jan–Mar 2020 via `toBands()` (42 bands, 89 MB raw).
- **Protocol:** 3 rounds with method order rotated each round. Every run uses a distinct expression (end date shifted by seconds, same scenes) so Earth Engine's result cache cannot favor later runs. Timings are wall-clock medians.
- **Account:** noncommercial project in **Restricted Mode**, which allows only about **2 concurrent requests** (a standard tier allows about 40). Per-request latency varied up to 4× for identical requests.

| Image | Previous `zeit` | geedim 2.0 (`max_requests=2`) | geedim 2.0 (default, 32 requests) | **New `zeit`** |
|:---|:---:|:---:|:---:|:---:|
| Medoid composite (79 MB) | 68.7 s (57.7–73.8) | 265.9 s, **1/3 failed** | failed (HTTP 429) | **77.1 s** (70.6–94.8) |
| NDVI (13 MB) | 57.0 s (38.3–91.0) | 37.8 s | 47.9 s | **26.9 s** (21.1–30.7) |
| Dense stack (42 bands) | **failed 3/3** | 74.9 s | failed (HTTP 429) | **34.4 s** (33.9–37.0) |

All successful outputs are **pixel-identical** across the three implementations (same grid, same values, same `-inf` nodata).

!!! note "Reading these numbers"
    With only ~2 concurrent requests allowed, total time is bounded by Earth Engine's per-request compute speed (~0.7 MB/s for the medoid). No client can beat that. The new downloader's main speed lever, running up to `sub_tile_workers` requests at once (16 by default) with tiles small enough to keep them all busy, has almost no room to work here. The medoid result is a statistical tie with the previous implementation, whose four ~20 MB requests happen to suit a 2-request cap. Expect larger gains on a standard-tier account; they were not measurable with this project.

!!! success "What changed regardless of quota"
    1. **Dense stacks work.** Bands are split only when needed, so a `toBands()` stack stays under the 32 MB / 1024-band per-request limits. The previous implementation could not download it at all.
    2. **Composites aren't recomputed per band.** geedim splits bands first, so a 6-band medoid is computed several times over the same area, which explains its 3.4× longer run. `zeit` splits space first.
    3. **Adaptive concurrency.** Throttled requests reduce concurrency instead of failing, so no run of the new downloader failed. geedim at its default concurrency failed in 2 of 3 cases.
    4. **No mosaic step.** Tiles go straight into their window of the output file, so peak memory is one tile instead of the whole image.

---

## STAC Download Throughput (Landsat annual composites)

`build_time_series` reads cloud-optimized GeoTIFFs over HTTP, so a whole Landsat stack is bound by the network, not the CPU. We timed the input LandTrendr needs, one masked median per year of NBR bands, on a 150 km tile before and after the 0.25.0 changes.

- **Area:** GLanCE EU tile `EU_150-X032-Y024` (central Germany), 5664 × 5689 px at 30 m in EPSG:3035.
- **Data:** Landsat Collection 2 Level-2 from Planetary Computer, June–September scenes with < 30 % cloud cover, bands `nir08` + `swir22` + `qa_pixel`. The full 1984–2025 record has 1,396 such scenes.
- **Work:** search, read, mask, per-year median, written to GeoTIFF. Each run in a fresh process.
- **Machine:** 20-thread desktop with 68 GB RAM on a ~200 Mbit/s link (18–26 MB/s and 33 ms per request to Azure West Europe).

| Year (scenes) | 0.24.0: `build_time_series` + `median` | 0.25.0: `build_annual_composites` |
|:---|:---:|:---:|
| 1990 (17) | 104 s, 0.57 GB | **20 s**, 0.38 GB |
| 2016 (44) | 224 s, 2.10 GB | **64 s**, 1.41 GB |
| 2023 (100) | 419 s, 5.39 GB | **160 s**, 3.39 GB |
| **Full tile, 1984–2025** (fit on the three years) | ~2 h, ~63 GB | **~35 min**, ~41 GB |

With the new defaults alone (`chunksize=2048`, `dtype="float32"`), `build_time_series` + `median` does 2016 in 74 s. The composites agree with the 0.24.0 ones on 96 % of the pixels; the rest sit on 512 px chunk edges, where the old nearest-neighbour reads picked a neighbouring source pixel.

!!! success "What made the difference"
    1. **Large chunks.** 2048 px chunks instead of 512 px mean fewer requests and fewer COG blocks read twice at chunk edges: 35 % fewer bytes for the same pixels. 1024 px chunks took twice as long as 2048 px.
    2. **Half-size arrays.** Reads land as `float32` (or raw `uint16` for composites) instead of `float64`.
    3. **Per-chunk reduction.** `build_annual_composites` gathers all scenes of one spatial chunk and reduces it in NumPy (mask, per-scene scale/offset, NaN median). Dask's own `median` re-splits the data into small pieces, and on the 100-scene year it exhausted 68 GB of RAM.
    4. **The link is now the limit.** 20–22 MB/s of a measured 22–26 MB/s.

!!! note "What did not help"
    - **Downloading whole files** with 32 parallel streams saturated the link but moved 7.2 GB instead of 1.4 GB: 316 s for 2016.
    - **One process per 2048 px block** (6–9 processes): ~135 s, because every process re-reads the headers of every file.
    - **More threads.** 20 and 64 threads gave 84 s and 79 s. With 128 threads in one process (and with 64 threads on the 100-scene year before the per-chunk reduction) GDAL deadlocked inside its HTTP layer: no CPU, no traffic, no error.

**AWS vs. Planetary Computer.** The same Landsat files are on Earth Search, in the requester-pays `s3://usgs-landsat` bucket (AWS us-west-2). Reading it needs AWS credentials and costs about US$ 0.09/GB leaving AWS, so ~US$ 4 per tile above. From this machine, AWS us-west-2 measured 11–18 MB/s and 226 ms per request against 18–26 MB/s and 33 ms for Planetary Computer, so the same tile would take an estimated 40–90 min there (not measured: no credentials). Inside the data's region (an Azure West Europe or AWS us-west-2 VM) transfer is free and much faster, which is the way to scale to many tiles.
