# Pre-processing

<p class="lead">Clean time series before analysing them: put Landsat and Sentinel-2 in one series, flag clouds and shadows that the QA band missed, smooth noise, remove one-off spikes, and unmix pixels into the fractions of their materials.</p>

## Landsat and Sentinel-2 in one series

A dense series that mixes Landsat and Sentinel-2 has a step wherever the sensor changes, which CCDC or BFAST read as a break. There are two ways to remove it:

- **HLS** (Harmonized Landsat Sentinel-2, NASA): both sensors processed alike from level 1 (the same atmospheric correction and cloud mask, view and sun angles normalized, a common 30 m grid) and Sentinel-2 adjusted to Landsat 8's bands. [`build_time_series`](data.md#build_time_series) reads it from Planetary Computer, both products in one call: `collection=["hls2-l30", "hls2-s30"]` with the bands named by role. This is the way to go wherever HLS covers the dates (from 2013 for Landsat 8, 2015 for Sentinel-2).
- **[`nbar`](#nbar)** and **[`bandpass_adjust`](#bandpass_adjust)** on cubes you already have (Landsat Collection 2 and Sentinel-2 L2A): HLS's view-angle normalization and its bandpass step, in that order (`bandpass_adjust(nbar(cube, angles))`, as in HLS). Neither is the atmospheric correction, so only part of the step goes away. `nbar` needs the per-pixel sun and view angles; `bandpass_adjust` does not.
- **[`spectral_adjust`](#spectral_adjust)** goes the other way, Landsat and MODIS to Sentinel-2A's bands (FORCE's spectral adjustment), and predicts the red edge bands Landsat does not have.

Measured over the Libya-4 desert site (2022–2023, Planetary Computer, pairs of Landsat 8/9 and Sentinel-2 observations at most a day apart, median of the area):

| | Pairs | Sentinel-2 / Landsat, by band (blue ... SWIR2) |
| :--- | :---: | :--- |
| L2A and Collection 2, as they come | 21 | 1.14, 1.07, 1.09, 1.05, 1.09, 1.13 |
| L2A after `bandpass_adjust` | 21 | blue and red about a third closer (differences 0.030 → 0.021 and 0.042 → 0.031), the rest about the same |
| HLS (S30 and L30) | 27 | 1.01, 0.99, 0.99, 1.00, 1.01, 0.99 |

The rest of the L2A step comes from the steps `bandpass_adjust` does not do: Sen2Cor and LaSRC correct the atmosphere differently, and the view geometry differs.

```python
grid = dict(source="planetary_computer", bbox=bbox, start_date="2019-01-01", end_date="2024-12-31",
            apply_cloud_mask=True, resolution=30, epsg=32722)
hls = zeit.build_time_series(collection=["hls2-l30", "hls2-s30"],
                             bands=["blue", "green", "red", "nir", "swir1", "swir2"], **grid)
result = zeit.ccdc(hls.sel(band=["blue", "green", "red", "nir", "swir1", "swir2"]))
```

### `nbar` { .api }

<!-- sig: zeit.nbar -->
```python
zeit.nbar(
    data, angles=None, *, sensor=None, sza="scene", bands=None, params=None,
    nodata="auto", chunks=None, n_jobs=-1, return_cfactor=False,
)
```

Normalizes reflectance to what a sensor at nadir would see with the sun at a chosen zenith (NBAR, nadir BRDF-adjusted reflectance), by the c-factor method of Roy et al. (2016, *Remote Sensing of Environment* 176:255-271; 2017, 199:25-38). Each band is multiplied by `c = ρ(sza_norm, view 0) / ρ(sza, vza, raa)`, where ρ is the MODIS Ross-Thick / Li-Sparse-Reciprocal model `f_iso + f_vol·K_vol + f_geo·K_geo` with three fixed parameters per band (Roy et al. 2016, Table 5, the global MODIS values; the HLS User Guide v2.0, Table 4, applies the same to OLI and MSI), evaluated at the pixel's observed solar zenith, view zenith and relative azimuth (view minus sun azimuth). It is the view-angle step of HLS, and goes before [`bandpass_adjust`](#bandpass_adjust). HLS cubes are refused (already normalized).

**Written from the papers.** The implementation follows the equations of the papers and the HLS User Guide, not any other implementation (FORCE's, HLS's...): the checks are the published parameters, analytic values of the kernels and the c-factor, and the properties of the model (see `tests/test_nbar.py`). It is compared against the HLS product itself (the same method, the same parameters): for HLS L30 against Landsat Collection 2 of the same scene, the median ratio of HLS's NBAR to the USGS reflectance in every bin of view zenith and azimuth is within 0.0083 of Zeit's c in every band (0.01 and 0.02 tolerances), slope 0.98-1.07; for Sentinel-2 the angles from `sentinel2_angles` give c within 0.003 (std) of the c from HLS's own angles ([fidelity](../benchmarks/fidelity.md#10-nbar-against-hls-zeitnbar-zeitsentinel2_angles)). The kernels are those that Matsuoka et al. (2016, *ISPRS Annals* III-7:3-10) write after Wanner et al. (1995) and Lucht et al. (2000), with Li-Sparse's h/b = 2 and b/r = 1.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | Surface reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask. Bands are found by role: `coastal`, `blue`, `green`, `red`, `rededge1`-`rededge3`, `nir` (B08), `nir08` (B8A, OLI's band 5), `swir1`, `swir2`, named as Earth Search or Planetary Computer name them (`swir16`, `B04`...). Sentinel-2's B08 is normalized (it has MODIS's NIR parameters); bands without a role (B09, QA, the angles) are copied. |
| `angles` | `Dataset`, `DataArray` or `None` | `None` | Sun and view angles in degrees, `SZA`, `SAA`, `VZA`, `VAA`, on exactly the cube's grid and dates, `(time, y, x)` (`(y, x)` for a cube without time); nothing is resampled and a mismatch is an error. `None` takes bands of the cube with those names. Where to get them: HLS carries them (scaled by 0.01); Landsat Collection 2 has Level-1 angle bands (USGS credentials); Sentinel-2 has a 5 km grid in each tile's `MTD_TL.xml`, which [`sentinel2_angles`](#sentinel2_angles) turns into per-pixel angles on the cube's grid. `VZA` and `VAA` may have a `band` dimension (`sentinel2_angles(..., view_band="each")`): each band is then normalized with its own view geometry (the bands to normalize must all have one). A pixel with a missing angle, or a zenith outside 0-90°, is NaN. |
| `sensor` | `str` or list | `None` | As for `bandpass_adjust`. Only checks the sensors (Landsat 4-9, Sentinel-2) and keeps the Collection 2 names `SR_B1`... from being read as OLI's on Landsat 4, 5 and 7 (pass `bands=`). |
| `sza` | `float`, list, `"scene"` or `"hls"` | `"scene"` | The normalized solar zenith: a number of degrees (or one per date); `"scene"`, the mean of each date's observed solar zenith over the pixels whose angles are valid (Roy et al. 2016, 2017; with dask the angles are read once ahead of the computation); or `"hls"`, the `NBAR_SOLAR_ZENITH` tag as an attribute or coordinate of the cube (HLS computes it from the day and latitude with overpass times its guide does not give, so Zeit does not). The view is always nadir. Roy et al. advise against a normalized zenith far from the observed ones. |
| `bands` | `dict` | `None` | The cube's band of each role, when its names are not the usual ones. |
| `params` | `dict` | `None` | `{role: (f_iso, f_geo, f_vol)}` replacing the parameters of those roles (reflectance 0-1). |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). They stay missing. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default). |
| `return_cfactor` | `bool` | `False` | Also return the c-factors `(time, band, y, x)` of the normalized bands. They are not clipped; in a Landsat's field of view they stay within 0.95-1.05. |

</div>

**Returns** the cube with the same dimensions, coordinates, type (integers are rounded) and georeferencing, lazy if the input is, plus a `nbar_sza` coordinate (each date's normalized zenith). `attrs["nbar"]` says what was done, `attrs["nbar_bands"]` which bands, and a normalized cube is refused a second time. With `return_cfactor`, `(nbar, cfactor)`.

**Limits.** The red edge parameters are the linear interpolation between MODIS's red (645 nm) and NIR (858 nm) of Roy, Li and Zhang (2017, *Remote Sensing* 9:1325, Eq. 3); the paper found that interpolation not particularly suited to the red edge (a POLDER test), so treat the red edge result as less certain. The parameters are MODIS's snow-free ones: mask snow before. A pixel's c-factor is not clipped, so a geometry far from the model's range gives an extreme c: inspect `return_cfactor`. Speed: about 10 million pixels per second on one thread and 40-55 million on eight, for the ten bands at once or one.

```python
angles = xr.Dataset({"SZA": sza, "SAA": saa, "VZA": vza, "VAA": vaa})   # degrees, (time, y, x), the cube's grid
cube_nbar = zeit.nbar(cube, angles)
mixed = zeit.bandpass_adjust(cube_nbar)          # HLS's order: view angles first, then the bandpass
nbar, c = zeit.nbar(cube, angles, sza=35, return_cfactor=True)
```

### `sentinel2_angles` { .api }

<!-- sig: zeit.sentinel2_angles -->
```python
zeit.sentinel2_angles(
    source, like, *, view_band="B06", chunks=None, n_jobs=-1,
)
```

The sun and view angles of every pixel of a Sentinel-2 cube, from the `MTD_TL.xml` of its tiles, in the form [`nbar`](#nbar) takes. ESA gives each tile's angles as 23 × 23 grids with a 5 km step (`Geometric_Info/Tile_Angles`): the sun's zenith and azimuth, and the view zenith and azimuth of every band and detector. The grids start at the tile's upper left corner (the `ULX`, `ULY` of `Tile_Geocoding`) and run east (columns) and south (rows); the XML has no origin of its own, only the steps. A detector's grid is `NaN` where its swath does not reach, so a cell where two swaths overlap has two values.

How the grids become pixels (written from ESA's product description and Roy et al. 2017, *Remote Sensing of Environment* 199:25-38, Sec. 3.2, "bilinear interpolation from the 5 km grid values", the way HLS does it for its Sentinel-2 product):

- **Detectors are fused first, cell by cell**: the mean of the zeniths and the circular mean of the azimuths of the detectors that cover the cell. A cell is valid if any detector covers it.
- **Bilinear interpolation to the pixel centres**, over the valid corners of the cell only (their weights rescaled to sum to one). A pixel between the last valid node of a swath and the next, invalid one takes the valid corner's value (flat, within one 5 km cell); a pixel with no valid corner, or outside the 23 × 23 grid, is `NaN`. Azimuths are interpolated through their sine and cosine, so 0°/360° is no seam.
- **The CRS may differ**: if `like` is not in the tile's CRS (the neighbouring UTM zone, or HLS's grid, which uses the northern zone with negative northings for southern tiles), the pixel centres are projected to the tile's.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | STAC item, path, URL, XML text or list | required | The tile metadata of each date of `like`: an Earth Search item (asset `granule_metadata`) or a Planetary Computer item (`granule-metadata`, signed with `planetary_computer`), as a `pystac` Item or its dict; the path or URL of an `MTD_TL.xml`; or the XML. A list gives one per date, in the order of `like.time`; an entry may itself be a list of the tiles that make that date's mosaic (a pixel takes the first tile that has angles there). A source sensed more than a day from its date in `like` is an error. |
| `like` | `DataArray` | required | The cube the angles must match: its CRS, `y`, `x` (pixel centres) and `time`. With `view_band="each"` it also gives the bands. |
| `view_band` | `str` | `"B06"` | The band whose view angles every band takes: `"B06"` (the second red edge band) is HLS's choice, one view geometry for all bands. `"each"` gives each Sentinel-2 band of `like` its own `VZA` and `VAA` (a `band` dimension), which `nbar` uses band by band. Any other band name (`"B08"`, `"red"`...) takes that one for all. |
| `chunks` | `dict`, `"auto"` or `None` | `None` | `None` computes now unless `like` is dask (then lazy, with its chunks); otherwise lazy with those chunks. In memory, one 10980 × 10980 tile is four float32 images of 480 MB per date: use lazy for many dates. |
| `n_jobs` | `int` | `-1` | Threads of the in-memory computation (strips of rows). |

</div>

**Returns** an `xarray.Dataset` with `SZA`, `SAA`, `VZA`, `VAA` in degrees (float32, azimuths 0-360 clockwise from north, `(time, y, x)`) on exactly the coordinates of `like`.

**Against HLS.** HLS S30 uses the same grids: over a whole tile, Zeit's solar angles are 0.015° from HLS's (the Level-1C against the Level-2A metadata) and the view zenith has a standard deviation of 0.05° from it. At the borders between detectors HLS gives each pixel its detector's angles (a step of 20-60° in azimuth) where this blends the two over one 5 km cell; the c-factors differ by 0.0027 (std, 0.015 at most).

**Speed.** A whole 10980 × 10980 tile at 10 m, four layers, takes 4.3 s on one thread and 2.2 s on eight (Windows 11, memory bound beyond that); lazily it is computed with the cube's chunks.

```python
items = sorted(search.item_collection(), key=lambda i: i.datetime)   # Earth Search sentinel-2-l2a items, one per date
cube = zeit.build_time_series(source="earth_search", collection="sentinel-2-l2a", ...)
angles = zeit.sentinel2_angles(items, cube)            # same order as cube.time
cube_nbar = zeit.nbar(cube, angles)
each = zeit.sentinel2_angles(items, cube, view_band="each")   # a view geometry per band
```

### `bandpass_adjust` { .api }

<!-- sig: zeit.bandpass_adjust -->
```python
zeit.bandpass_adjust(
    data, *, sensor=None, etm=None, bands=None, scale="auto",
    nodata="auto", chunks=None,
)
```

Takes Sentinel-2 reflectance to the bands of Landsat 8's OLI with the bandpass adjustment of HLS (Claverie et al. 2018): one line per band and Sentinel-2 unit, `OLI = slope × MSI + intercept`, from the HLS v2.0 User Guide (Table 5; S2A and S2B are also the v1.4 values, S2C has its own). OLI is the reference, as in HLS: Landsat 8 and 9 stay as they are (HLS applies no adjustment between them). Landsat 7 (ETM+) and 4-5 (TM) are left alone unless `etm` is given; then the lines of Roy et al. (2016, Table 2, surface reflectance) take them to OLI. Those were fitted on pre-collection data, and Landsat Collection 2 surface reflectance is generally used across sensors without them (see Earth Engine's FAQ on cross-sensor harmonization). It is one step of HLS, not HLS (see above); HLS products are refused, since their Sentinel-2 dates are already adjusted.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path or list | required | Surface reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask. Or a list of cubes on the same grid (one per collection): each is adjusted and they are joined into one series sorted by time, with the bands they share matched by role and named after it (`blue` ... `swir2`), whatever each catalog calls them, and the coordinates they share. |
| `sensor` | `str` or list | `None` | Each date's sensor: by default the cube's `platform` coordinate (which [`build_time_series`](data.md#build_time_series) keeps from STAC) or attribute; or one name for every date (`"sentinel-2a"`, `"S2B"`, `"landsat-8"`, `"LANDSAT_7"`...), or one per date. With a list of cubes, one of these per cube. A Sentinel-2 date of unknown unit is adjusted as S2A, with a warning. |
| `etm` | `"rma"`, `"ols"` or `None` | `None` | Landsat 7 and 4-5: left as they are, or taken to OLI with the reduced major axis or least squares lines of Roy et al. (2016). TM is treated as ETM+, as in Earth Engine's harmonization tutorial (no TM lines are published). |
| `bands` | `dict` | `None` | The cube's band of each role (`coastal`, `blue`, `green`, `red`, `nir`, `swir1`, `swir2`), when its names are not the usual ones (`red`, `B04`, `SR_B4`...). With a list of cubes, one dict for all or one per cube. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by in the data: 10000 for integers or values above 2, else 1 (a lazy cube is checked on a small window). The intercepts are in reflectance. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). They stay missing. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |

</div>

**Returns** the cube with the same dimensions, coordinates, type and georeferencing (a list: the joined series), lazy if the input is. Bands without a line (the red edge, QA bands) are copied. `attrs["bandpass_adjusted"]` says what was done, and an adjusted cube is refused a second time.

**Sentinel-2's NIR.** HLS adjusts the narrow NIR, B8A (`nir08` on Earth Search, `B8A` on Planetary Computer), the band that matches OLI's band 5. B08, the broad NIR (`nir` on Earth Search, `B08`), has no line: it is left as it is, with a warning. Load B8A, or pass `bands={"nir": ...}` when a band is B8A under another name.

```python
s2 = zeit.build_time_series(collection="sentinel-2-l2a", bands=["B02", "B03", "B04", "B8A", "B11", "B12"], **grid)
landsat = zeit.build_time_series(collection="landsat-c2-l2",
                                 bands=["blue", "green", "red", "nir08", "swir16", "swir22"], **grid)
mixed = zeit.bandpass_adjust([s2, landsat])      # one series: blue, green, red, nir, swir1, swir2
```

### `spectral_adjust` { .api }

<!-- sig: zeit.spectral_adjust -->
```python
zeit.spectral_adjust(
    data, *, sensor=None, bands=None, scale="auto", nodata="auto",
    chunks=None, n_jobs=-1,
)
```

Takes Landsat 4, 5, 7, 8 and 9 and MODIS (Terra and Aqua) surface reflectance to the ten bands of Sentinel-2A (blue, green, red, the three red edge bands, the broad and the narrow NIR, SWIR1 and SWIR2), the red edge and the broad NIR being predicted, since those sensors do not have them. It is FORCE's spectral adjustment (FORCE 3.11, David Frantz, GPL-3.0-or-later; the method and the tables are those of Scheffler, Frantz and Segl 2020, *Remote Sensing of Environment* 241:111723). Each sensor has 50 clusters of spectra and a global one, each with a linear regression of the ten Sentinel-2A bands on the six of the sensor (blue, green, red, NIR, SWIR1, SWIR2) and an intercept. For each pixel the spectral angle to the centre of every cluster gives a weight, `1 - angle / 15°` (0.00001 beyond 15°); the closest clusters, at most ten, are kept while their weight is above that of 4°, or above that of 12° and the global cluster's, and the prediction is the average of their regressions weighted by those weights (the global cluster fills in where fewer than ten are close, and takes over where none is). All ten bands are rewritten, not only the new ones: Landsat 8's blue is not Sentinel-2A's. This is not [`bandpass_adjust`](#bandpass_adjust), which takes Sentinel-2 *towards* OLI with a line per band and predicts no band; Sentinel-2 dates pass through here unchanged.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | Surface reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask, with blue, green, red, NIR, SWIR1 and SWIR2 (found by role, as `bandpass_adjust` does). Sentinel-2 dates are passed through (their ten bands, named `blue green red rededge1 rededge2 rededge3 nir nir08 swir16 swir22` or `B02` ... `B12`). |
| `sensor` | `str` or list | `None` | Each date's sensor: by default the cube's `platform` coordinate (which [`build_time_series`](data.md#build_time_series) keeps from STAC) or attribute; or one name for every date (`"landsat-8"`, `"LC09"`, `"LND05"`, `"terra"`, `"MOD02"`...), or one per date. Landsat 4, 5, 7, 8 and 9 are told apart, and MODIS Terra from Aqua. Any other sensor is an error that names the date. |
| `bands` | `dict` | `None` | The cube's band of each role (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`; for MODIS `swir1` is band 6 and `swir2` band 7; and the ten Sentinel-2 names for Sentinel-2 dates), when its names are not the usual ones. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by in the data: 10000 for integers or values above 2, else 1. The method works on reflectance times 10000 (its intercepts are not scale-free): the data are converted to that and the result back to the data's scale. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). A pixel with a missing band is missing in all ten. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default). |
| `force_search` | `bool` | `False` | FORCE's first search as it is (cluster 9 skipped, the global cluster a possible first pick), so that the values truncated to integers are FORCE's in every pixel. |

</div>

**Returns** `(time, band, y, x)` (or `(band, y, x)`), float32, with the same coordinates and georeferencing and ten bands named as Earth Search names them, so that [`compute_indices`](data.md#compute_indices) finds the ones it uses: `blue`, `green`, `red`, `rededge1`, `rededge2`, `rededge3`, `nir` (B08), `nir08` (B8A), `swir16`, `swir22`. Lazy if the input is. `attrs["spectral_adjusted"]` says what was done for each sensor, and an adjusted cube is refused a second time.

**Where it differs from FORCE.** The arithmetic is FORCE's (its tables are generated from its source and checked, and with `force_search=True` the integer outputs equal FORCE's, 100% of 49 million values per sensor), with one difference on purpose: FORCE's first search for the closest cluster skips cluster 9 (it tests `s != 9` where the other searches leave out the global cluster, 50) and may pick the global cluster first. Here the global cluster is never the first pick and cluster 9 competes in every search, as in the second search and in the paper. On synthetic spectra of mixed land covers this changes 1.6-2.3% of the observations beyond one unit of the integer reflectance (see [Against the Original Implementations](../benchmarks/vs-originals.md#force-spectral-adjustment)). FORCE writes integers truncated toward zero and never reads the nodata value (its QA mask decides); here the output is float and not truncated.

```python
landsat = zeit.build_time_series(collection="landsat-c2-l2",
                                 bands=["blue", "green", "red", "nir08", "swir16", "swir22"], **grid)
s2like = zeit.spectral_adjust(landsat)           # blue ... swir22, plus rededge1-3 and nir08
ndvi = zeit.compute_indices(s2like, ["NDVI"])    # nir is B08, as on Sentinel-2
rededge = s2like.sel(band="rededge2") / s2like.sel(band="rededge1")   # the red edge, which Landsat does not measure
```

## Cloud masking

### `tmask` { .api }

<!-- sig: zeit.tmask -->
```python
zeit.tmask(
    data, *, green="green", swir="swir1", scale=10000.0,
    nodata="auto", chunks=None,
)
```

Flags clouds and cloud shadows the QA band missed, from each pixel's time series (Tmask, Zhu & Woodcock 2014): a robust harmonic model (MATLAB's `robustfit` with Tukey's bisquare, as CCDC's own Tmask, in Rust, every pixel in parallel) is fitted to the green and SWIR bands of every pixel, and observations more than 0.04 (reflectance) above it in green (clouds) or below it in SWIR (shadows) are flagged. One function for a `(time, band, y, x)` cube in memory or dask, or anything [`load_raster`](data.md#load_raster) reads; the dates come from its `time` coordinate. It replaces `apply_tmask_stack` and `run_tmask_pixel` (the per-pixel engine, still in `zeit._tmask`).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | A `(time, band, y, x)` cube with green and SWIR bands. |
| `green` | `str` or `int` | `"green"` | The green band, by name or position. |
| `swir` | `str` or `int` | `"swir1"` | The SWIR band (SWIR1, ~1.6 µm), by name or position. |
| `scale` | `float` | `10000.0` | Reflectance scale of the data (`1.0` for reflectance in 0-1); the thresholds are in reflectance. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation: the raster's NoData (`0` for integer data without one), a number, or `None`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy, computed block by block. |

</div>

**Returns** `clear (time, y, x)`, `bool`: `True` for a clear observation, `False` for a cloud, a shadow or no observation (NoData, NaN, or not above 0 in either band). Pixels with 5 or fewer valid dates are not screened. Georeferenced as the input.

```python
cube = zeit.load_raster("landsat/", pattern=r"_(?P<date>\d{8})_(?P<band>\w+)\.tif$")   # (time, band, y, x)
clear = zeit.tmask(cube, green="green", swir="swir1")
cube = cube.where(clear)                         # clouds and shadows become NaN
qa = xr.where(clear, 0, 4)                       # or as CCDC's Fmask codes (0 clear, 4 cloud)
```

## Spectral mixture analysis

### `unmix` { .api }

<!-- sig: zeit.unmix -->
```python
zeit.unmix(
    data, endmembers="souza2005", *, bands=None, sum_to_one=True,
    nonneg=True, scale="auto", ndfi=True, cloud_threshold=None,
    nodata="auto", chunks=None, n_jobs=-1,
)
```

The fraction of each endmember (pure material) in every pixel and date: a 30 m Landsat pixel of forest after selective logging is part canopy (green vegetation), part dead wood and litter (non-photosynthetic vegetation), part soil and part shade, and the fractions show the damage that the reflectance hides. The unmixing is fully constrained least squares (non-negative fractions that add up to one; Heinz & Chang 2001), in Rust, every pixel and date in parallel; the active set solves each pixel exactly. RStoolbox's `mesma` (R), whose NNLS iterates to a tolerance, takes 22 s on 1000 x 1000 pixels of 6 bands and 4 endmembers; Zeit 1.2 s on one thread, 0.13 s on 19, its residual never larger (benchmarks/vs_originals/unmix.py). Also exported as `zeit.unmix`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | Reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask. |
| `endmembers` | `str`, `DataFrame` or `dict` | `"souza2005"` | `"souza2005"`: green vegetation (`gv`), `shade`, non-photosynthetic vegetation (`npv`), `soil` and `cloud` for Landsat's blue, green, red, NIR, SWIR1 and SWIR2 (Souza et al. 2005, with the cloud endmember of CODED, as CODED uses them). Or a table: rows endmembers, columns bands of the cube, reflectance 0–1 (or × 10000). |
| `bands` | list of `str` | `None` | The cube's band for each column of the endmembers, when the names differ (by default found by name: `nir`, `B08`, `SR_B5`...). |
| `sum_to_one`, `nonneg` | `bool` | `True`, `True` | Fractions adding up to one; fractions non-negative. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by: 10000 for integers or values above 2, else 1. |
| `ndfi` | `bool` | `True` | Also the NDFI (below), when the endmembers include `gv`, `npv`, `soil` and `shade`. |
| `cloud_threshold` | `float` | `None` | Mask observations whose `cloud` fraction is above this (CODED: 0.05). |
| `nodata`, `chunks`, `n_jobs` | | `"auto"`, `None`, `-1` | As elsewhere. |

</div>

**Returns** an `xarray.Dataset` with one variable per endmember (its fraction), `rmse` (of the rebuilt spectrum, in reflectance) and `ndfi`, on the cube's dimensions but `band`, lazy if the cube is.

The **NDFI** (Normalized Difference Fraction Index; Souza et al. 2005) is `(GVs − (NPV + soil)) / (GVs + NPV + soil)`, with `GVs = GV / (1 − shade)`: near 1 in closed forest, lower where the canopy was damaged, below 0 on bare soil and pasture. It is also an index of [`compute_indices`](data.md) (`compute_indices(cube, ["NDFI"])`) and of the Earth Engine downloads.

```python
fractions = zeit.unmix(landsat, cloud_threshold=0.05)
fractions.ndfi.zeit.plot()                       # page through the years
```

## Smoothing

### `smooth` { .api }

<!-- sig: zeit.smooth -->
```python
zeit.smooth(
    data, *, method="whittaker", lmbda=10.0, weights=None, window=5,
    polyorder=2, nodata="auto", chunks=None, n_jobs=-1,
)
```

Smooths time series along `time`, keeping their dates, georeferencing and type. One function for every input: a cube `(time, y, x)` or `(time, band, y, x)` in memory or dask, anything [`load_raster`](data.md#load_raster) reads, or one pixel's `pandas.Series` indexed by dates. It replaces `apply_whittaker_filter` and `apply_savgol_filter`.

- **Whittaker** (default; Eilers 2003): a penalised least-squares curve that follows the data more or less closely (`lmbda`). The dates may be unevenly spaced: the penalty uses the time between them, measured in median steps of the series, so `lmbda` means the same on any spacing (on an even series it is the classic smoother). Missing observations (NaN, NoData) weigh 0 and are filled by the curve; `weights` down-weights hazy ones.
- **Savitzky-Golay** (`method="savgol"`): a local polynomial over `window` dates, as scipy's `savgol_filter`, for evenly spaced series (regularise first with [`regularize_time_series`](data.md#regularize_time_series)). Missing observations are first interpolated linearly in time.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path or `pd.Series` | required | The series to smooth (see above). |
| `method` | `str` | `"whittaker"` | `"whittaker"` or `"savgol"`. |
| `lmbda` | `float` | `10.0` | Whittaker: smoothness, larger is smoother (`10` light for a 16-day NDVI series, `1000` strong). |
| `weights` | array or `DataArray` | `None` | Whittaker: a weight per observation, same shape as `data` (`1` clear, `0.2` hazy, `0` cloudy; see [`qc_sentinel2_scl`](data.md#qc_sentinel2_scl)). |
| `window` | `int` | `5` | Savitzky-Golay: window length in dates (odd). |
| `polyorder` | `int` | `2` | Savitzky-Golay: order of the local polynomial. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation: the raster's NoData (`0` for integer data without one), a number, or `None`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Whittaker: threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** an `xarray.DataArray` with the same dims and coordinates (a `pandas.Series` for a Series). Floats keep their type; integer data with a NoData value (NDVI × 10000 in Int16) is rounded back to its type, NoData where a pixel has no observation at all; integer data without one comes back as float32. `attrs["smoothing"]` names the method, and [`zeit.plot(cube, fit=smoothed)`](plot.md) draws the curve over a clicked pixel's series.

```python
ndvi = zeit.load_raster("S2_ndvi_2022.tif")                    # (time, y, x), clouds as NaN
smooth = zeit.smooth(ndvi, lmbda=100)                          # gaps filled by the curve
weighted = zeit.smooth(ndvi, weights=zeit.qc_sentinel2_scl(scl))
sg = zeit.smooth(ndvi_16d, method="savgol", window=7)
zeit.plot(ndvi, fit=smooth)                                    # click a pixel: raw and smoothed
```

### `desawtooth` { .api }

<!-- sig: zeit.desawtooth -->
```python
zeit.desawtooth(values, stopat=0.9)
```

LandTrendr's spike removal for a single series: one-year spikes (up or down) are dampened toward their neighbours. [`landtrendr`](change-detection.md#landtrendr) already applies it internally through its `spike_threshold` parameter, so call this only to inspect or pre-process series yourself.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `values` | 1-D array | required | One pixel's series. |
| `stopat` | `float` | `0.9` | Each point gets a spike score (1 = a perfect one-year spike). Spikes are corrected strongest first while the strongest score exceeds `stopat`, so lower values correct more points. As in the original IDL code, the first correction is always applied. |

</div>

```python
zeit.desawtooth(np.array([8000, 8100, 3000, 8050, 7900, 8000.0]))
# array([8000., 8034.7, 8025.2, 8050., 7900., 8000.])
```

## Interpolation and standardization

Ported from the time series analysis of [FORCE](https://github.com/davidfrantz/force) 3.11.00 (Frantz 2019, GPL-3.0-or-later): `interpolate` is its TSI, `standardize` its STANDARDIZE. The statistics, trends and clear-observation counts of the same chain are in [FORCE time series analysis](time-series.md#force-time-series-analysis); the [tutorial](../tutorials/force_tsa.md) runs the whole chain, and [Against the original implementations](../benchmarks/vs-originals.md#force-tsa) has the agreement with FORCE.

### `interpolate` { .api #interpolate }

<!-- sig: zeit.interpolate -->
```python
zeit.interpolate(
    data, method="rbf", *, step=16, start=None, end=None, times=None,
    sigma=(8, 16, 32), cutoff=0.95, window=16, modes=3, trend=True,
    fit=None, period=365.25, dates=None, band=None, nodata="auto",
    chunks=None, n_jobs=-1,
)
```

Interpolates every pixel's series to a regular grid of dates, or to any dates you give. One function for every input, like [`smooth`](#smooth): a `(time, y, x)` cube (in memory or dask), anything [`load_raster`](data.md#load_raster) reads, a `(time, band, y, x)` cube or `Dataset` with `band=`, a numpy array with `dates=`, or one pixel's series. The dates are needed and must not decrease; NaN (and NoData) marks a missing observation. Ported from FORCE's TSA (Frantz 2019), with four methods:

- **RBF** (`"rbf"`, default; Schwieder et al. 2016): an ensemble of Gaussian kernels of standard deviations `sigma` (days), each cut where its mass reaches `cutoff`, weighted by how much data it finds around each date. The kernels are combined into one weight per integer day, the same average as FORCE's loop over kernels. A date with no valid observation within the largest kernel is NaN.
- **Linear** (`"linear"`): straight lines between the valid observations on either side, constant beyond the first and last.
- **Moving average** (`"moving"`): the mean of the valid observations within `window` days (closed).
- **Harmonic** (`"harmonic"`; Zhu et al. 2015): an offset, an optional slope and up to 3 harmonics of `period`, fit by iteratively reweighted least squares with bisquare weights (tuning 4.685, MAD scale; the algorithm of GSL's `gsl_multifit_robust`). `fit=(first, last)` fits on part of the series and predicts the rest, which is FORCE's near-real-time mode.

The grid is FORCE's: from `start` (default the first date) every `step` days, `floor((end - start + 1) / step)` dates, `end` being the last date by default. No date is made after `end`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path, `ndarray` or `pd.Series` | required | The series to interpolate (see above). |
| `method` | `str` | `"rbf"` | `"rbf"`, `"linear"`, `"moving"` or `"harmonic"`. |
| `step` | `int` | `16` | Days between the dates of the grid. |
| `start`, `end` | date | `None` | First and last date of the grid. Default: the first and last date of the series. |
| `times` | list of dates | `None` | Any dates for the result, instead of the grid (`start`, `end` and `step` are then not allowed). |
| `sigma` | sequence of `float` | `(8, 16, 32)` | RBF: standard deviations of the kernels in days, increasing. |
| `cutoff` | `float` | `0.95` | RBF: the share of each kernel's mass kept (0-1). |
| `window` | `float` | `16` | Moving average: the half-width in days. |
| `modes` | `int` | `3` | Harmonic: the number of harmonics (1-3). |
| `trend` | `bool` | `True` | Harmonic: add an offset-and-slope term. |
| `fit` | `(date, date)` | `None` | Harmonic: the window the model is fit on (inclusive). Default: the whole series. |
| `period` | `float` | `365.25` | Harmonic: the period in days. |
| `dates` | list | `None` | Date of each time step, for inputs that carry none. |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN, as in [`smooth`](#smooth). |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** an `xarray.DataArray` `(time, y, x)` with the new dates, the coordinates and georeferencing of the input, and `attrs` with the method and its parameters (a `pandas.Series` for a Series). Float32 for float32 input, else float64; the values are not rounded.

**Differences from FORCE**, all deliberate:

- The output is float, not FORCE's int16: no rounding to the data's scale and no clipping.
- The harmonic's period is 365.25 days, the real year; FORCE counts 365 on a calendar without leap days. The time is centered and scaled inside and the system solved by Cholesky on balanced columns, which does not change the fit.
- A pixel whose harmonic IRLS cycles without converging (about 1 in 100 in the benchmark, found by the core) is left out of the comparison with FORCE: neither implementation settles on a fit, and a handful of them differ from FORCE by more than 1. A pixel with fewer valid observations in `fit` than coefficients is NaN.

```python
ndvi = zeit.load_raster("S2_ndvi_2018_2022.tif")                   # (time, y, x), clouds as NaN
tsi = zeit.interpolate(ndvi, "rbf", step=16)                       # FORCE's default grid
daily = zeit.interpolate(ndvi, "linear", times=pd.date_range("2020", "2021", freq="D"))
nrt = zeit.interpolate(ndvi, "harmonic", fit=("2018-01-01", "2020-12-31"), end="2022-12-31")
```

### `standardize` { .api #standardize }

<!-- sig: zeit.standardize -->
```python
zeit.standardize(
    data, method="normalize", *, ddof=1, dim=None, dates=None,
    band=None, nodata="auto", chunks=None, n_jobs=-1,
)
```

Standardizes every pixel's series along an axis (FORCE's STANDARDIZE and STANDARDIZE_FOLD). The mean and the deviation are those of the valid observations; missing observations stay NaN. Takes the same inputs as [`interpolate`](#interpolate) (the dates are not needed), and any axis through `dim=`, such as the `year` of [`temporal_metrics(..., by="year")`](time-series.md#temporal_metrics).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path, `ndarray` or `pd.Series` | required | The series to standardize. |
| `method` | `str` | `"normalize"` | `"normalize"`: `(x - mean) / std`, the z-score; `"center"`: `x - mean`. |
| `ddof` | `int` | `1` | Degrees of freedom of the deviation (`n - ddof`; 1 is the sample deviation, as FORCE). |
| `dim` | `str` | `None` | The axis to standardize along: `"time"` (default when it exists) or another, such as `"year"`. A `metric` axis of length 1 is dropped; with more, choose one with `.sel(metric=...)`. |
| `dates` | list | `None` | Date of each time step, for inputs that carry none (not needed for the result). |
| `band` | `str` or `int` | `None` | The index to use in a `(time, band, y, x)` cube or a `Dataset`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation besides NaN. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Threads. `-1`: all the cores but one, or every core of a cluster job ([how they are counted](parallel.md#threads-n_jobs)). |

</div>

**Returns** the cube with the same dims, coordinates and georeferencing (a `pandas.Series` for a Series). A series with fewer than 2 valid observations is left as it is; a constant series gives NaN when normalized. Float32 for float32 input, else float64, not rounded as FORCE's int16 is.

```python
z = zeit.standardize(tsi)                                          # z-score of each pixel along time
anomaly = zeit.standardize(tsi, "center")                          # x - mean
yearly = zeit.temporal_metrics(ndvi, ["mean"], by="year")
zy = zeit.standardize(yearly, dim="year")                          # each year against the others
```
