# Spatiotemporal Fusion: Daily Landsat from MODIS (STARFM and ESTARFM)

Landsat 8 and 9 see a place every eight days at best, and under clouds not at all; MODIS sees it every day at 500 m. **Spatiotemporal fusion** predicts the Landsat image of a MODIS date by taking the fine detail from the Landsat images you have and the change from the MODIS images of the dates in between. Zeit has the two classic methods, written from their papers: [`fuse_starfm`](../api/preprocessing.md#fuse_starfm) (STARFM, Gao et al. 2006) and [`fuse_estarfm`](../api/preprocessing.md#fuse_estarfm) (ESTARFM, Zhu et al. 2010), and [`fusion_plan`](../api/preprocessing.md#fusion_plan) to see which dates and pairs a run would use.

Both start from **pairs**: a fine image and a coarse image of the same date. Everything else is predicted from them.

| | STARFM | ESTARFM |
|:---|:---|:---|
| Pairs | one or two (the nearest before and after a date) | two (the bracketing ones; extrapolates from the two nearest at the ends) |
| Bands | each on its own | all together (a similar pixel is similar in every band) |
| Neighbours | similar in the fine image of the pair, filtered by their spectral and temporal distance | similar on both pair dates, weighted by the correlation of their fine and coarse spectra |
| Coarse change | taken as it is | scaled by a coefficient regressed from the similar pixels |
| Strong side | simple, robust, one pair is enough | better where the land cover is heterogeneous and the change is not the same in every land cover |

## 1. The two series

The fine series is a reflectance cube `(time, band, y, x)` with the clouds already masked (NaN): any of Zeit's builders gives it (`zeit.build_time_series` of Landsat Collection 2 Level 2, HLS, Sentinel-2). The coarse series is MODIS in **its own grid and CRS**: Zeit places it on the fine grid per strip, bilinearly, so the sinusoidal projection of MODIS is fine. Which MODIS?

* **MCD43A4**, the nadir BRDF-adjusted reflectance (`modis-43A4-061` on the Planetary Computer; daily, Terra and Aqua together): the natural partner of a near-nadir sensor, because the view-angle noise of daily MODIS is gone. Mask it **per band** with its mandatory quality (0 full inversion, 1 magnitude inversion, which you may keep in the cloudy season).
* **MOD09A1** (`modis-09A1-061`): 8-day composites of real observations; better for abrupt events, noisier. Give the time axis the `start_datetime` and `end_datetime` coordinates of the items and a fine date pairs with the composite that contains it.

The bands are matched by role (blue, green, red, nir, swir1, swir2: `sur_refl_b01`... and `Nadir_Reflectance_Band1`... are known); MODIS band 5 has no Landsat counterpart and is not used. The script `benchmarks/vs_originals/fusion_real.py` reads both series of a small area from the Planetary Computer as an example (windowed reads of the COGs, the quality bands applied), and `zeit.load_raster` reads any COGs you have.

## 2. What would be done: `fusion_plan`

```python
import zeit

plan = zeit.fusion_plan(landsat, modis)                       # STARFM: the nearest pair on each side
plan = zeit.fusion_plan(landsat, modis, method="estarfm")     # ESTARFM: two pairs, extrapolating at the ends
print(plan)                                                   # date, kind, coarse_date, pairs, extrapolated
```

One row per output date: the coarse dates between the first and last Landsat date and the Landsat dates themselves (`kind="observed"`: the valid pixels are kept and the clouded ones predicted from the *other* pairs, which fills the gaps). Landsat dates that are not pairs (too many clouds, excluded, no MODIS of that date) are in `plan.attrs["unpaired"]` with the reason.

## 3. Predicting

```python
pred = zeit.fuse_starfm(landsat, modis, radius=900)           # (time, band, y, x), float32, on the Landsat grid
pred = zeit.fuse_estarfm(landsat, modis, radius=900)
pred.fusion_kind                                              # "observed" or "predicted", per date
pred, q = zeit.fuse_starfm(landsat, modis, quality=True)      # n_similar and source of every value
```

`radius` is the half window in map units: 750 m is 25 Landsat pixels, three MODIS pixels, the paper's choice; for Sentinel-2 at 10 m use 300-500 m, because the cost grows with its square. The work is done in strips of rows with their halo, threaded (`n_jobs`); the result does not depend on the strips or the threads.

## 4. Checking it: leave one date out

The honest test is to predict a Landsat date you have, without using it:

```python
d = "2023-06-16"
loo = zeit.fuse_estarfm(landsat, modis, dates=[d], include_fine_dates=False, exclude_dates=[d])
truth = landsat.sel(time=d)
```

`benchmarks/vs_originals/fusion_loo.py` does this for every clear date of a site and compares with the nearest Landsat date, the linear interpolation in time and the coarse image itself. The results on two 15 x 15 km sites are in [Against the Original Implementations](../benchmarks/vs-originals.md#spatiotemporal-fusion-starfm-and-estarfm):

| Site | Method | Red RMSE | NIR RMSE | NDVI RMSE |
|:---|:---|---:|---:|---:|
| Bahia | nearest Landsat date | 0.0196 | 0.0294 | 0.0548 |
| Bahia | linear interpolation in time | 0.0150 | 0.0229 | 0.0372 |
| Bahia | `fuse_starfm` (defaults) | 0.0179 | 0.0278 | 0.0522 |
| Bahia | `fuse_estarfm` (defaults) | 0.0179 | 0.0284 | 0.0515 |
| Goias | nearest Landsat date | 0.0121 | 0.0224 | 0.0488 |
| Goias | linear interpolation in time | 0.0095 | 0.0168 | 0.0343 |
| Goias | `fuse_starfm` (defaults) | 0.0108 | 0.0221 | 0.0442 |
| Goias | `fuse_estarfm` (defaults) | 0.0113 | 0.0215 | 0.0418 |
| Bahia, nearest pair 24+ days away | nearest Landsat date | 0.0320 | 0.0470 | 0.1188 |
| Bahia, nearest pair 24+ days away | linear interpolation in time | 0.0214 | 0.0341 | 0.0692 |
| Bahia, nearest pair 24+ days away | `fuse_starfm` (defaults) | 0.0228 | 0.0383 | 0.0780 |
| Bahia, nearest pair 24+ days away | `fuse_estarfm` (defaults) | 0.0219 | 0.0383 | 0.0770 |
| Goias, nearest pair 24+ days away | nearest Landsat date | 0.0191 | 0.0427 | 0.0962 |
| Goias, nearest pair 24+ days away | linear interpolation in time | 0.0134 | 0.0275 | 0.0566 |
| Goias, nearest pair 24+ days away | `fuse_starfm` (defaults) | 0.0137 | 0.0323 | 0.0628 |
| Goias, nearest pair 24+ days away | `fuse_estarfm` (defaults) | 0.0133 | 0.0315 | 0.0603 |

Reflectance 0-1, mean over the 25 (Bahia) and 22 (Goias) dates; the linear interpolation needs a clear date on each side, so it is averaged over fewer. Fusion is worth its cost where the Landsat dates around are weeks away, not days: at eight days a linear interpolation between two clear Landsat images is hard to beat.

## 5. Choices, and what they cost

* **`fuse_starfm` or `fuse_estarfm`.** ESTARFM wants two good pairs around the date and a coarse series with a clear change between them; where the change is small or the pairs are far from the date STARFM, which can use one pair, is safer. Look at `pred.fusion_extrapolated` and at the `source` of the quality layer.
* **Do not fuse an index.** Fuse the bands and compute the index afterwards: ESTARFM's weights come from the correlation of the spectra of a pixel, which an index does not have.
* **Clouds in the coarse series** stay missing per band (NaN); `fallback="coarse"` writes the coarse value where nothing can be predicted.
* **Georeferencing.** MODIS's geolocation is about 50 m: if you have measured a shift, `coarse_shift=(dx, dy)` in the units of the coarse CRS applies it.
* **Size.** A full scene with 46 dates writes about 17 GB as float32: select the area (or the dates) first.
* **Bandpasses.** Landsat and MODIS bands differ slightly; neither method needs an adjustment beforehand (STARFM's per-pair term absorbs an additive difference, ESTARFM's coefficient a gain).

## References

* Gao, F., Masek, J., Schwaller, M. and Hall, F. (2006). On the blending of the Landsat and MODIS surface reflectance: predicting daily Landsat surface reflectance. *IEEE TGRS* 44(8), 2207-2218. [doi:10.1109/TGRS.2006.872081](https://doi.org/10.1109/TGRS.2006.872081)
* Zhu, X., Chen, J., Gao, F., Chen, X. and Masek, J. G. (2010). An enhanced spatial and temporal adaptive reflectance fusion model for complex heterogeneous regions. *Remote Sensing of Environment* 114(11), 2610-2623. [doi:10.1016/j.rse.2010.05.032](https://doi.org/10.1016/j.rse.2010.05.032)
* Zhu, X., Cai, F., Tian, J. and Williams, T. K.-A. (2018). Spatiotemporal fusion of multisource remote sensing data: literature survey, taxonomy, principles, applications, and future directions. *Remote Sensing* 10(4), 527.
