# Noise-Ordered Components (MNF)

<p class="lead">A hyperspectral image has hundreds of bands, but far fewer independent signals: neighbouring bands carry nearly the same information, each with its own noise. The Minimum Noise Fraction transform finds the combinations of bands that carry the most signal relative to noise, puts them first, and leaves the noise in the last components. Keep the first ones and you have a compact, cleaner image; invert them and you get the bands back with the noise taken out.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">How many independent signals does this image have, and what does it look like without the noise?</span></div>
<div><span class="k">Input</span><span class="v">A multiband image <code>(band, y, x)</code> or <code>(time, band, y, x)</code>: hyperspectral, or any stack with two bands or more</span></div>
<div><span class="k">Output</span><span class="v">Components <code>mnf1…mnfK</code> ordered by signal-to-noise ratio, a table of eigenvalues, and optionally the denoised bands</span></div>
<div><span class="k">Reference</span><span class="v">Green et al. (1988), <em>IEEE TGRS</em> 26:65-74; Lee et al. (1990), <em>IEEE TGRS</em> 28:295-304</span></div>
</div>

## How it works

1. **Signal and noise.** A pixel's spectrum is the signal plus noise, and the two are assumed uncorrelated: the covariance of the bands is the sum of the signal's and the noise's. The signal is smooth in space (a neighbouring pixel is nearly the same material); the noise is not (each pixel's noise is independent).
2. **The noise from the neighbours.** Subtract each pixel from the one beside it. The signal nearly cancels and the noise is left, twice over; half the covariance of those differences is the noise covariance. By default the right and lower neighbours are used together (`noise="diff_hv"`). Pairs with a missing pixel do not count.
3. **Whiten the noise, then a PCA.** The bands are rotated and scaled so that the noise has the same variance in every direction and no correlation between bands, and then a principal components analysis is run on the result. Because the noise is now the same everywhere, the variance of a component is its signal-to-noise ratio plus one: **the eigenvalue of component *i* is 1 + SNR<sub>i</sub>**. A component with an eigenvalue near 1 is noise.
4. **Order and units.** The components come highest SNR first. Unlike a PCA of the raw bands, the result does not depend on the units or the scale of each band, so a band in reflectance and another in digital numbers do not need rescaling.
5. **The inverse.** Setting the last components to zero (their mean) and transforming back gives the bands without the noise those components held.

## Step by step

### 1. Read the image

```python
import zeit

cube = zeit.load_raster("EnMAP_L2A_Groningen_20220810.tif", chunks="auto")
cube = cube.isel(y=slice(560, 910), x=slice(740, 1090)).load()   # (band, y, x): 224 x 350 x 350
refl = cube.where(cube != -32768) / 10000                          # reflectance 0-1, NaN = no data
```

This is the window of the [unmixing tutorial](unmixing.md): farmland and the Wadden Sea in an EnMAP scene, 30 m. The bands that the scene leaves out completely (no valid value) are left out of the MNF by themselves; so are constant ones.

### 2. Fit and look at the eigenvalues

```python
components, tr = zeit.mnf(refl, return_transform=True)
tr.table()[["component", "eigenvalue", "snr", "noise_fraction"]].head(12)
tr.n_with_snr(1.0)      # components whose signal is at least as strong as the noise
```

On this window (350 x 350 pixels, 218 bands once the 6 empty ones are left out; 1.7 s for the fit and the components) the first component has an eigenvalue of 82.8, an SNR of 82 (19 dB); the second 31.0 and the third 26.5; component 12 has 9.0; 60 components have an SNR of at least 1, and the last ones are at 0.92 to 1.14, which is what pure noise gives (a value a little under 1 means the estimate of the noise is a little high there).

| component | eigenvalue | snr | noise_fraction |
| ---: | ---: | ---: | ---: |
| 1 | 82.84 | 81.84 | 0.012 |
| 2 | 31.00 | 30.00 | 0.032 |
| 3 | 26.52 | 25.52 | 0.038 |
| 4 | 17.37 | 16.37 | 0.058 |
| 5 | 16.53 | 15.53 | 0.060 |
| 12 | 9.03 | 8.03 | 0.111 |


The table has one row per component. `snr_db`, the share of the signal each component holds and its cumulative share are in it too, and `flag` marks a component without variance (a band that is an exact combination of others). A sharp drop to about 1 tells how many independent signals there are; the components after it are noise, and the eigenvalue never drops below 1 by much unless the signal has fine spatial detail, which the difference estimator takes for noise.

### 3. Choose where the noise is measured

By default every pixel and its neighbours count. If the scene has a lot of fine texture, the differences hold signal and the noise is overestimated (eigenvalues come out lower, even below 1). Measure it in a homogeneous area instead:

```python
tr = zeit.mnf_fit(refl, noise_region=(x0, y0, x1, y1))   # a pixel window; or a boolean (y, x) array
```

Only the differences are taken inside the region (both pixels of each pair); the signal statistics are still those of the whole image. Open water, a flat field or a shadow are good places. `noise="diff_diag"` and the other directions are there to compare: if two directions give very different eigenvalues the noise is correlated between neighbours (stripes, resampling), and that part counts as signal.

### 4. Keep the first components, or denoise

```python
y12 = zeit.mnf(refl, 12)                          # only the first 12 components
clean = zeit.mnf_denoise(refl, snr_min=1.0)       # keep the components with SNR >= 1, rebuild the bands
clean, tr = zeit.mnf_denoise(refl, k=12, return_transform=True)
```

`snr_min` keeps the components whose eigenvalue is at least `1 + snr_min`. `k` and `snr_min` are exclusive, and one is required. The components after the first `k` are set to zero, which is their mean: the noise they hold goes, and so does any signal they hold.

On the same window, `snr_min=1.0` keeps 60 of the 218 components; the bands rebuilt from the first 5, 12 and 20 components differ from the original by a root mean square of 0.033, 0.026 and 0.020 in reflectance (0-1), which is the part of the image the dropped components hold, mostly noise and the fine detail of the scene. The reflectance spectrum of a pixel loses its band-to-band jitter: the standard deviation of its second difference along the wavelength goes from 0.030 to 0.027 with the 60 components kept (one pixel, 218 bands).

### 5. Apply it elsewhere, keep it

```python
other = zeit.mnf(refl_other_date, transform=tr)       # the same components for another date
tr.to_json("enmap_mnf.json")                          # keep it
tr = zeit.MNFTransform.from_json("enmap_mnf.json")
back = zeit.inverse_mnf(components, tr, k=12)         # from stored components to bands
```

With a time axis, `zeit.mnf(series)` pools the statistics of all dates, one transform for every date, so that component *i* means the same thing in each. `per_date=True` fits one per date.

## Choices and pitfalls

- **Components are not bands.** Their sign is arbitrary in principle; Zeit fixes it so that each component correlates positively with the band it is most correlated with. Their scale is "noise variance 1" (as ENVI and SPy); `scale="unit"` gives unit variance.
- **Few pixels.** The fit needs more valid pixels, and more pairs, than bands, and a warning appears below ten times as many.
- **Constant or duplicated bands** are handled (a constant band is left out; an exact duplicate gives one component with no variance and a warning).
- **Large images.** The statistics are gathered in blocks of rows, so a dask cube larger than memory works, and the result does not depend on the number of threads. `sample_fraction` uses a fraction of the pixels for the statistics.
- **It is not a classification.** The MNF components are a rotation of the bands. They are a good input for one (fewer, cleaner features), not a result.

## Checked against SPy

The transform equals SPy's (Spectral Python 0.26) from the same statistics to rounding error, and the statistics of Zeit's own estimators equal SPy's; see [Against the Original Implementations](../benchmarks/vs-originals.md#mnf-vs-spy).
