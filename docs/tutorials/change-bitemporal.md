# Change Between Two Dates and in Sentinel-1 Series

<p class="lead">Two images of the same place, a year apart, never agree in brightness: the sun, the atmosphere and the sensor changed too. Change vector analysis reads the change from the differences of the bands; iMAD finds it without being fooled by those differences, and uses the pixels that did not change to normalize one date to the other. For a radar series the omnibus test says, pixel by pixel, whether and when the backscatter changed.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">What changed between two dates, how much, and of what kind? Which pixels are unchanged, to normalize one image to the other? When did a Sentinel-1 pixel change, and did it brighten or darken?</span></div>
<div><span class="k">Input</span><span class="v">Two dates with the same bands on the same grid (<code>zeit.cva</code>, <code>zeit.imad</code>, <code>zeit.radcal</code>); a cube of linear VV and VH backscatter of one relative orbit (<code>zeit.omnibus</code>)</span></div>
<div><span class="k">Output</span><span class="v">Magnitude, direction and sectors of the change; the MAD variates, the chi-square and probability of no change; normalized images and the fits; dates, number and direction of changes</span></div>
<div><span class="k">Reference</span><span class="v">Malila (1980); Bovolo et al. (2012); Nielsen et al. (1998); Nielsen (2007); Canty &amp; Nielsen (2008); Conradsen et al. (2016, 2024); Canty et al. (2020)</span></div>
</div>

## How it works

### Change vector analysis

The change vector of a pixel has one component per band, the difference between the two dates. Its length is the **magnitude**; a threshold on it separates change from no change. Zeit offers a fixed value, Otsu's threshold of the histogram of the magnitudes, and the default, the minimum-error threshold of a mixture of two Gaussian classes (no change, change) fitted by EM (Bruzzone and Prieto 2000). The **direction** says what kind of change it was: the angle to the diagonal of uniform change (0 degrees every band up together, 180 every band down together), the angle in the plane of two bands (brightness and greenness, in four sectors), or the sector of the signs of the changes of red, NIR and SWIR1, where sector 6 (red and SWIR1 up, NIR down) is vegetation loss.

CVA reads the differences as they are, so a change of illumination is a change. Normalize the dates first when they differ in conditions.

### iMAD

The canonical correlation analysis of the bands of date 1 and the bands of date 2 finds the combination of each date that agrees most, then the next, and so on. The differences of the canonical variate pairs, the MAD variates, are uncorrelated, and because a canonical correlation does not change when either date is multiplied by a gain or shifted by an offset (or mixed by any invertible linear transform), neither do the MAD variates up to scale: no normalization is needed. Where nothing changed they are Gaussian, so their standardized sum of squares `chi2` follows a chi-square distribution and `p_nochange` is a probability. The iterations weight each pixel by that probability when computing the covariances, so that the changed pixels stop disturbing the analysis; they stop when the canonical correlations settle.

### The normalization

The pixels with `p_nochange` above 0.95 are the invariant pixels: a sample of the scene's stable surfaces across the whole range of brightness. For each band, an orthogonal regression of the reference on the target gives the gain and offset that map one onto the other, allowing for noise in both dates; a third of the invariant pixels is held back and tests the result (the means should agree, and so should the variances).

### The omnibus test

At each pixel the Sentinel-1 backscatter of all the dates is compared by a likelihood ratio that accounts for the speckle: the number of looks sets how much two single observations may differ by chance. If the hypothesis that all the dates are alike is rejected, the statistic splits into one test per date against those before it. The first date that rejects is a change; the series then restarts from it. Every registered change has the false alarm rate you ask for.

## Two Landsat dates

```python
import zeit

cube = zeit.load_raster("CCDC_Stack_Rondonia.tif")            # Landsat, (time, band, y, x)
bands = ["blue", "green", "red", "nir08", "swir16", "swir22"]
dates = ("2021-07-16", "2022-07-27")

change = zeit.imad(cube, dates=dates, bands=bands)
zeit.imad_table(change).tail(3)                                # the correlations of the last iterations
change.sel(band="p_nochange").plot()                           # 0: changed, 1: did not
```

`change` holds `MAD1..MAD6`, `chi2`, `p_nochange` and `change` (`p_nochange < 0.01`). On this pair iMAD flags most of the scene at 0.01: a year apart in the Amazon, with different sun angles and another sensor, it is not the same scene. Rank the pixels by `chi2` rather than cutting at 0.01, or look at the MAD variates themselves; `MAD1` is the largest change.

Normalize the second date to the first and read what it did:

```python
norm = zeit.radcal(cube, reference=dates[0], targets=[dates[1]], bands=bands)
zeit.radcal_table(norm)[["band", "b", "se_b", "a", "r", "p_t", "p_F"]]
```

`b` and `a` are the gain and the offset (`normalized = a + b * target`). A `p_t` close to 0 would say that the means of reference and normalized target still differ on the pixels held back.

## What kind of change

```python
out = zeit.cva(norm, dates=dates, direction="sign", threshold="otsu")
zeit.cva_table(out)                       # pixels, hectares and mean magnitude of each sector
loss = out.sel(band="sector") == 6        # vegetation loss
```

With another description of the direction: `zeit.cva(norm, dates=dates, direction="polar", polar_bands=["blue", "green"])` for the plane of two bands, or the default `"c2va"` for the angle to uniform change, with `c2va_sectors=[0, 60, 120, 180]` to cut it.

## A Sentinel-1 series

```python
s1 = zeit.load_raster("s1_rtc_orbit83/*.tif")             # (time, band: vv, vh, y, x): linear gamma0

enl = zeit.estimate_enl(s1, region=(120, 135, 40, 55))    # a block of intact forest
change, direction = zeit.omnibus(s1, enl=enl["enl"]["vv"], alpha=0.01, per_date=True)

change.sel(band="first_change").plot()                    # decimal year
change.sel(band="n_changes").plot()
direction.sel(time="2021-12-01", method="nearest").plot() # 1 brightening, 2 darkening, 3 indefinite
```

What to keep in mind:

- **Linear power.** Planetary Computer's `sentinel-1-rtc` gives linear gamma0; a series in dB is refused. Convert with `10 ** (dB / 10)`.
- **One orbit.** The viewing geometry changes the backscatter, and the test cannot tell that from change: use the dates of one relative orbit and pass.
- **The looks.** 4.4 is for Sentinel-1 IW GRDH; RTC products are resampled and differ. Estimate it on a homogeneous area and give the same value to everything that you compare.
- **Dates.** A change is found at its own date in a part of the pixels and a date or two later in the others: one new image against the mean of those before it has limited power. A pixel masked on a date uses its other dates; the change is dated at the next valid one.

## Results of this tutorial's methods against the original

The three methods reproduce Mort Canty's scripts, run as black boxes, on synthetic and real data: see [Against the Original Implementations](../benchmarks/vs-originals.md#imad-radcal-and-the-omnibus-test-vs-canty). The reference is *Image Analysis, Classification and Change Detection in Remote Sensing* (Canty, 4th and 5th editions) and the papers cited on the [API page](../api/change-detection.md#cva).
