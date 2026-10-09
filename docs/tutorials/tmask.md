# Cloud Masking with Tmask

<p class="lead">Single-image cloud masks miss thin clouds and small shadows, and every missed cloud looks like a land change to the algorithms downstream. Tmask uses the time dimension instead: it learns each pixel's normal seasonal behaviour and flags observations that are far too bright (cloud) or far too dark (shadow) for that time of year.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Which observations are clouds or shadows that the QA band missed?</span></div>
<div><span class="k">Input</span><span class="v">Green and SWIR1 reflectance stacks (× 10,000) and their dates</span></div>
<div><span class="k">Output</span><span class="v">A boolean clear mask, same shape as the input</span></div>
<div><span class="k">Reference</span><span class="v">Zhu & Woodcock (2014)</span></div>
</div>

<figure markdown>
  ![Tmask applied to a synthetic series: cloud-contaminated observations in the Green band and shadows in the SWIR1 band are flagged](../assets/figures/tmask_flags.png)
  <figcaption><strong>What Tmask does.</strong> Four years of a synthetic 16-day series with 7 clouds (too bright in Green) and 5 shadows (too dark in SWIR1) injected. Tmask fits a robust seasonal model to each band and flags the observations that depart from it too much (orange): all 12 contaminated ones, and nothing else.</figcaption>
</figure>

## How it works

For each pixel, Tmask fits a **robust harmonic regression** (a Huber fit, so the outliers barely affect it) to two bands:

- **Green**, where clouds are much brighter than the land below them;
- **SWIR1**, where shadows are much darker than expected.

An observation is flagged as cloud when its Green value is far **above** the seasonal model, and as shadow when its SWIR1 value is far **below** it. Everything else is kept as clear.

## Step by step

### 1. Load Green and SWIR1

```python
import numpy as np
from datetime import date
import zeit

green = zeit.load_raster("green_stack.tif", dates=acquisition_dates)   # (time, y, x), reflectance x 10000
swir = zeit.load_raster("swir1_stack.tif", dates=acquisition_dates)

dates = np.array([date.fromisoformat(d).toordinal() for d in acquisition_dates])
green, swir = green.values, swir.values
```

### 2. Run Tmask

```python
from zeit.tmask import apply_tmask_stack

clear = apply_tmask_stack(dates, green, swir, scale_factor=10000.0)
print(clear.dtype, clear.shape)   # bool (time, rows, cols); True = clear
print(f"{1 - clear.mean():.1%} of observations flagged")
```

If your reflectance is already in 0–1, pass `scale_factor=1.0`. Pixels with fewer than 5 observations are returned as all clear, since there is too little data to model the season.

### 3. Use the mask

**For CCDC**, convert to its QA codes (`0` clear, `4` cloud), and combine with the QA band you already have:

```python
tmask_qa = np.where(clear, 0, 4).astype(np.uint8)
qa = np.maximum(qa_from_qa_pixel, tmask_qa)     # keep the worst of the two
```

!!! warning "Do not use `(~clear).astype(np.uint8)` for CCDC"
    That produces `1` for clouds, and in CCDC's convention `1` means **water**, which counts as a valid observation. The clouds would be kept.

**For everything else**, set the flagged observations to `NaN` before compositing or smoothing:

```python
ndvi_clean = np.where(clear, ndvi, np.nan)
```

**To save it**:

```python
zeit.save_raster(clear.astype(np.uint8), "results/tmask_clear.tif", like="green_stack.tif")
```

### For a single pixel

```python
from zeit.tmask import run_tmask_pixel

clear_px = run_tmask_pixel(dates, green[:, 100, 200], swir[:, 100, 200])
```

## Good practice

- **Use Tmask as a second pass.** Apply the sensor's own QA mask first (Landsat `QA_PIXEL`, Sentinel-2 `SCL`, or `apply_cloud_mask=True` in `build_time_series`) to remove the obvious clouds, then Tmask to catch what it missed.
- **CCDC already includes it.** `zeit.ccdc` runs the original's Tmask screen internally. Running it beforehand is only needed for other algorithms.
- **It costs a robust fit per pixel.** For large areas, process in spatial blocks or with Dask.

## References

- Zhu, Z., & Woodcock, C. E. (2014). Automated cloud, cloud shadow, and snow detection in multitemporal Landsat data: An algorithm designed specifically for monitoring land cover change. *Remote Sensing of Environment*, 152, 217–234. [doi:10.1016/j.rse.2014.06.012](https://doi.org/10.1016/j.rse.2014.06.012)
