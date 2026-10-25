# User Guide

<p class="lead">Step-by-step tutorials for every part of Zeit. Each one starts with what the method is for and a real output, explains how it works in plain language, then walks through the code and how to read the results.</p>

!!! tip "New here?"
    Do the [Quickstart](../getting-started/quickstart.md) first, and skim [Core Concepts](../getting-started/concepts.md) for the conventions (array shapes, dates, scale factors) that every tutorial uses. Unsure which method fits your question? See [Choosing an Algorithm](../getting-started/choosing-an-algorithm.md).

## Change detection

Find where and when the land surface changed.

<div class="gallery" markdown>

<a class="tile" href="landtrendr/">
  <img src="../assets/figures/thumbs/landtrendr.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Annual data</span><span class="tile-title">LandTrendr</span><span class="tile-text">Disturbance and recovery history from one image per year. Maps of year, magnitude and duration.</span></span>
</a>

<a class="tile" href="recovery/">
  <img src="../assets/figures/thumbs/recovery.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">After a disturbance</span><span class="tile-title">Recovery metrics</span><span class="tile-text">Years to recovery, R80P, RRI and the year-on-year recovery of every disturbed pixel.</span></span>
</a>

<a class="tile" href="ccdc/">
  <img src="../assets/figures/thumbs/ccdc.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Dense data</span><span class="tile-title">CCDC</span><span class="tile-text">Harmonic models of every clear observation. Dates changes and describes the land before and after.</span></span>
</a>

<a class="tile" href="sccd/">
  <img src="../assets/figures/thumbs/sccd.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Dense data · near real time</span><span class="tile-title">S-CCD</span><span class="tile-text">CCDC carried forward by a Kalman filter: the breaks, and the changes still in progress at the last image.</span></span>
</a>

<a class="tile" href="bfast/">
  <img src="../assets/figures/thumbs/bfast.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Regular series</span><span class="tile-title">BFAST</span><span class="tile-text">Separate breaks in the trend from breaks in the seasonal cycle.</span></span>
</a>

<a class="tile" href="bfast_monitor/">
  <img src="../assets/figures/thumbs/bfast_monitor.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Near real time</span><span class="tile-title">BFAST Monitor</span><span class="tile-text">Test new observations against a stable history. The basis of alert systems.</span></span>
</a>

<a class="tile" href="ewmacd/">
  <img src="../assets/figures/thumbs/ewmacd.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Irregular series</span><span class="tile-title">EWMACD</span><span class="tile-text">Control charts on every observation: disturbances, gains and recoveries, cloudy dates welcome.</span></span>
</a>

<a class="tile" href="bfast_lite/">
  <img src="../assets/figures/thumbs/bfast_lite.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Regular series</span><span class="tile-title">BFAST Lite</span><span class="tile-text">The optimal number of breaks in a whole series, in a single fast pass.</span></span>
</a>

<a class="tile" href="beast/">
  <img src="../assets/figures/thumbs/beast.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Annual or regular series</span><span class="tile-title">BEAST</span><span class="tile-text">Trend and season averaged over thousands of models: how probable each changepoint is, and how many there were.</span></span>
</a>

</div>

## Time-series analysis

Trends, seasons, patterns and objects.

<div class="gallery" markdown>

<a class="tile" href="mann_kendall/">
  <img src="../assets/figures/thumbs/mann_kendall.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Trends</span><span class="tile-title">Mann-Kendall</span><span class="tile-text">Significant greening and browning, with a robust slope per pixel.</span></span>
</a>

<a class="tile" href="regional-trends/">
  <img src="../assets/figures/thumbs/parts.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Regional trends</span><span class="tile-title">PARTS</span><span class="tile-text">Is the region, or each land-cover class, really trending? A test that accounts for correlated pixels.</span></span>
</a>

<a class="tile" href="hotspots/">
  <img src="../assets/figures/thumbs/hotspots.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Space-time patterns</span><span class="tile-title">Emerging hot spots</span><span class="tile-text">Getis-Ord Gi* at every date and its trend: new, intensifying, persistent, sporadic hot and cold spots.</span></span>
</a>

<a class="tile" href="phenology/">
  <img src="../assets/figures/thumbs/phenology.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Seasons</span><span class="tile-title">Phenology</span><span class="tile-text">Start, peak and end of every growing season, and 16 more metrics.</span></span>
</a>

<a class="tile" href="unmixing/">
  <img src="../assets/figures/thumbs/unmixing.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Spectral mixtures</span><span class="tile-title">Spectral unmixing</span><span class="tile-text">The fraction of each material in every pixel and date, with endmembers of your own.</span></span>
</a>

<a class="tile" href="twdtw/">
  <img src="../assets/figures/thumbs/twdtw.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Classification</span><span class="tile-title">TWDTW</span><span class="tile-text">Match each pixel to reference patterns, tolerating shifts in timing.</span></span>
</a>

<a class="tile" href="som/">
  <img src="../assets/figures/thumbs/som.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Clustering</span><span class="tile-title">SOM</span><span class="tile-text">Find the typical trajectories in a landscape without labels.</span></span>
</a>

<a class="tile" href="snic/">
  <img src="../assets/figures/thumbs/snic.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Segmentation</span><span class="tile-title">SNIC</span><span class="tile-text">Group pixels with similar histories into fields and patches.</span></span>
</a>

</div>

## Deep learning

PyTorch models for classification and change detection, weight-compatible with their reference implementations.

<div class="gallery" markdown>

<a class="tile" href="tempcnn/">
  <img src="../assets/figures/thumbs/tempcnn.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Pixel classification</span><span class="tile-title">TempCNN</span><span class="tile-text">A fast 1-D convolutional baseline for pixel time series.</span></span>
</a>

<a class="tile" href="postprocessing/">
  <img src="../assets/figures/thumbs/postprocessing.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">After classifying</span><span class="tile-title">Post-processing & AoA</span><span class="tile-text">Bayesian smoothing, a minimum mapping unit, and the area where the model can be applied.</span></span>
</a>

</div>

| Model | Task | Tutorial |
| :--- | :--- | :--- |
| Overview | Available models, data loading, loss functions | [Deep Learning](ai.md) |
| TempCNN | Pixel time-series classification (convolutional) | [TempCNN](tempcnn.md) |
| LTAE / LightTAE | Pixel time-series classification (attention) | [LTAE & LightTAE](ltae.md) |
| Presto | Pretrained pixel time-series model: few samples, gaps, embeddings of a cube | [Presto](presto.md) |
| U-TAE | Segmentation of whole image patches over time | [UTAE](utae.md) |
| Siamese | Change detection between two dates | [Siamese Change Detector](siamese.md) |
| Prithvi | Fine-tuning the Prithvi-EO-2.0 foundation model | [Prithvi](prithvi.md) |
| Any classifier | Smoothing the class map, a minimum mapping unit, the area of applicability | [Post-processing & AoA](postprocessing.md) |

## Getting and preparing data

<div class="gallery" markdown>

<a class="tile" href="stac-downloads/">
  <img src="../assets/figures/thumbs/smoothing.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Cloud catalogs</span><span class="tile-title">STAC Data Cubes</span><span class="tile-text">Lazy cubes from Earth Search, Planetary Computer or Brazil Data Cube; compositing and smoothing.</span></span>
</a>

<a class="tile" href="tmask/">
  <img src="../assets/figures/thumbs/tmask.webp" alt="" loading="lazy">
  <span class="tile-body"><span class="tile-kicker">Cloud masking</span><span class="tile-title">Tmask</span><span class="tile-text">Catch the clouds and shadows that single-image masks miss.</span></span>
</a>

</div>

- **[Plotting & Exploring Results](plotting.md)**: `zeit.plot` for maps, dense time series and pixel fits, in a notebook or a window.
- **[Google Earth Engine](gee-downloads.md)**: harmonised Landsat composites prepared on Google's servers.
- **[Embeddings (TESSERA, AlphaEarth)](embeddings.md)**: the yearly embeddings of foundation models as a cube: classify from few samples, find similar places, follow change year to year.
- **[Parallel & Cloud Processing](parallel-cloud-processing.md)**: from one core to a cluster, with Dask and Zarr.
- **[Zeit on a Cluster (HPC, Slurm)](hpc-slurm.md)**: installing on a cluster, a job array over the tiles with `zeit slurm`, time limits, nodes without internet access.
