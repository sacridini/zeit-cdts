# Preparing Images Before the Analysis

<p class="lead">Time series algorithms assume that a pixel means the same thing on every date. These functions take out what does not: images that sit a pixel or two off, view and sun geometry, bands that differ between sensors, and resolution. Most come from FORCE (David Frantz, GPL-3.0-or-later); each one says where it is from, when to use it and how to call it.</p>

| Problem | Function |
| :--- | :--- |
| The dates do not line up with a reference image | [`coregister`](#coregistration), [`coregister_local`](#coregistration) |
| The view angle changes the reflectance from date to date | [`nbar`](#nbar) (with [`sentinel2_angles`](#nbar)) |
| Landsat and MODIS next to Sentinel-2 | [`spectral_adjust`](#spectral-adjustment) |
| Brightness, greenness and wetness instead of bands | [`tasseled_cap`](#tasseled-cap) |
| The 20 m bands of Sentinel-2 at 10 m | [`sentinel2_resmerge`](#the-20-m-bands-at-10-m) |
| A coarse product, or Landsat, at the pixels of Sentinel-2 | [`improphe`](#improphe) |
| Multispectral bands at the pixels of a finer panchromatic band | [`pansharpen`](#pansharpening) |

For the atmospheric correction itself, from a Level 1C scene, see [Sentinel-2 Level 1C to Surface Reflectance](sentinel2-level2.md). For Landsat and Sentinel-2 in one series where HLS covers the dates, see [Pre-processing](../api/preprocessing.md#landsat-and-sentinel-2-in-one-series).

## Coregistration

**When.** Images of the same place that are a fraction of a pixel to a few pixels apart make a stable surface look like change. [`coregister`](../api/preprocessing.md#coregister) finds each date's shift to a reference image by phase correlation and moves the date; [`coregister_local`](../api/preprocessing.md#coregister_local) lets the shift vary across the scene, for a misregistration that grows from one side to the other. Both port the co-registration of AROSICS (Scheffler et al. 2017, *Remote Sensing* 9(7):676; AROSICS 1.14.0, Apache-2.0).

The cube and the reference must be on the same pixel grid (the same projected CRS and cell size, the cells a whole number apart): `zeit.load_raster(..., like=cube)` puts the reference there.

```python
reference = zeit.load_raster("reference.tif", like=cube)
aligned, shifts = zeit.coregister(cube, reference, band="nir", ref_band="nir")
shifts[["x_shift_px", "y_shift_px", "reliability", "ssim_before", "ssim_after", "success"]]

aligned, points = zeit.coregister_local(cube, reference, grid_res=100, window_size=128,
                                        band="nir", ref_band="nir")
```

A date that fails (no overlap of valid data, no clear match, a shift above `max_shift`, 5 pixels by default) stays as it was, with `success=False` and the `reason` in the table. Check `ssim_before` and `ssim_after` to see that the date agrees better afterwards, and `min_reliability` refuses the weak shifts.

For a single Level 1C scene, [`level2`](sentinel2-level2.md) takes `reference=` and coregisters the TOA reflectance with the global `coregister` before the atmospheric correction, as FORCE does with its own method.

## NBAR

**When.** The reflectance of a pixel depends on the angles of the sun and the sensor. [`nbar`](../api/preprocessing.md#nbar) normalises it to a nadir view and a chosen sun zenith with the c-factor method of Roy et al. (2016, 2017), the view-angle step of HLS. It needs the per-pixel `SZA`, `SAA`, `VZA` and `VAA`; for Sentinel-2, [`sentinel2_angles`](../api/preprocessing.md#sentinel2_angles) makes them from each tile's `MTD_TL.xml`. Written from the papers and the HLS User Guide, not from FORCE's code.

```python
angles = zeit.sentinel2_angles(["T1/MTD_TL.xml", "T2/MTD_TL.xml"], cube)   # one XML per date of cube.time
cube_nbar = zeit.nbar(cube, angles)
cube_nbar = zeit.bandpass_adjust(cube_nbar)     # HLS's order: view angles first, then the bandpass
```

The red edge parameters are an interpolation the paper found not particularly suited to that region, and the parameters are MODIS's snow-free ones: mask snow first and inspect `return_cfactor=True` where the geometry is extreme.

## Spectral adjustment

**When.** To put Landsat 4 to 9 and MODIS (Terra, Aqua) in the bands of Sentinel-2A, including the red edge bands that those sensors do not measure. [`spectral_adjust`](../api/preprocessing.md#spectral_adjust) is FORCE's spectral adjustment (Scheffler, Frantz and Segl 2020, *Remote Sensing of Environment* 241:111723). It is not [`bandpass_adjust`](../api/preprocessing.md#bandpass_adjust), which takes Sentinel-2 *towards* Landsat 8's OLI.

```python
s2like = zeit.spectral_adjust(landsat, sensor="landsat-8")      # blue ... swir22, plus rededge1-3 and nir08
ndvi = zeit.compute_indices(s2like, ["NDVI"])                   # nir is B08, as on Sentinel-2
```

With `force_search=True` the arithmetic is FORCE's to the letter; by default one search is done as in the paper, which changes 1.6-2.3 % of the observations of synthetic spectra beyond one unit of integer reflectance (see the [benchmark](../benchmarks/vs-originals.md#force-spectral-adjustment)).

## Tasseled Cap

**When.** Brightness, greenness and wetness are three weighted sums of six bands, a compact input for change detection. [`tasseled_cap`](../api/preprocessing.md#tasseled_cap) does by default what FORCE's TSA does, with Crist's (1985) matrix for every sensor.

```python
tc = zeit.tasseled_cap(landsat)                                       # as FORCE
tc = zeit.tasseled_cap(landsat, ["wetness"], coefficients="sensor")   # the matrix of each date's sensor
lt = zeit.landtrendr(tc.sel(band="wetness"))
```

Only Crist's matrix was derived for surface reflectance; the others were derived for top-of-atmosphere reflectance, so applying them to surface reflectance is common practice but approximate. `"disturbance"` is FORCE's `TC-DI` (`brightness - greenness - wetness`), not the standardized disturbance index of Healey et al. (2005).

## The 20 m bands at 10 m

**When.** The six 20 m bands of Sentinel-2 (red edge 1-3, B8A, SWIR1, SWIR2) on the 10 m grid, predicted from the 10 m green, red and broad NIR. [`sentinel2_resmerge`](../api/preprocessing.md#sentinel2_resmerge) is FORCE's resolution merge. The input is a cube on the 10 m grid where each 20 m pixel is four equal pixels. For a single Level 1C scene, [`level2`](sentinel2-level2.md) does it itself with `merge=`, on the TOA reflectance before the atmospheric correction as FORCE does; use `sentinel2_resmerge` for cubes that are already corrected or for time series.

```python
sharp = zeit.sentinel2_resmerge(s2, method="regression", clear=cloud_free)
fast = zeit.sentinel2_resmerge(s2, method="improphe", clear=cloud_free)    # faster
```

`clear` is True where a pixel is not cloud or shadow; the others are not predicted. FORCE's third option, STARFM, is not included.

## ImproPhe

**When.** To sharpen a coarse field (a 500 m phenology metric, say) or the scenes of a coarser sensor to the pixels of a finer one. [`improphe`](../api/preprocessing.md#improphe) gives each pixel the weighted mean of the coarse values at the neighbours that look like it in the fine data. It is FORCE's CF-ImproPhe and L2-ImproPhe (Frantz et al. 2016, *IEEE Transactions on Geoscience and Remote Sensing* 54(7):4153-4164).

```python
# a yearly 500 m field (year, y, x), with the Sentinel-2 cube of the same years
sharp = zeit.improphe(lsp_500m, sentinel2, by="year", kernel_size=500, texture_size=60)
# Landsat scenes (time, band, y, x) at 10 m from the same cube
l8_10m = zeit.improphe(landsat, sentinel2, by=None, kernel_size=300, texture_size=60)
```

The default radii (`kernel_size=2500`, `texture_size=330`, in map units) suit 10 m pixels. Everything is in memory, with a circle of `kernel_size` around each pixel: for a large area, work a tile at a time with a margin of that size.

## Pansharpening

**When.** To put multispectral bands on the grid of a finer panchromatic band, for any sensor, band count and resolution ratio above 1 (Landsat 8's band 8, Sentinel-2's 10 m with its 20 m or 60 m bands, the CBERS-4A WPM). [`pansharpen`](../api/preprocessing.md#pansharpen) has four methods, `"gsa"` (the default), `"brovey"`, `"hpf"` and `"pca"`.

```python
fused = zeit.pansharpen(wpm_ms, wpm_pan, method="gsa")                       # CBERS-4A WPM, ratio 4
fused = zeit.pansharpen(l8_ms, l8_b8, "gsa", intensity_bands=["B3", "B4"])   # Landsat 8: the pan sees green and red
```

The formulas were written from the literature without the papers at hand, and the defaults of `"hpf"` are Zeit's rather than the paper's; treat them as a starting point. Bands that the pan does not see (NIR, SWIR for a visible pan) get little detail from GSA, and `intensity_bands` chooses the bands that estimate the intensity. The registration of the pan to the bands is not checked: use [`coregister`](#coregistration) first.

## Credits

`spectral_adjust`, `tasseled_cap`, `improphe` and `sentinel2_resmerge` follow FORCE 3.11 (David Frantz, GPL-3.0-or-later; Frantz 2019, *Remote Sensing* 11:1124). The co-registration is AROSICS' (Scheffler et al. 2017, Apache-2.0). Each function's page in the [API reference](../api/preprocessing.md) says where it agrees with its original and where it differs on purpose, and the numbers are in [Against the Original Implementations](../benchmarks/vs-originals.md).
