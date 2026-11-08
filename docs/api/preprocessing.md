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

### `topographic_correction` { .api }

<!-- sig: zeit.topographic_correction -->
```python
zeit.topographic_correction(
    cube, dem, *, sun_zenith="scene", sun_azimuth="scene", method="c", sample=None,
    min_slope=5.0, min_sample=100, bands=None, nodata="auto", n_jobs=-1,
)
```

Takes the modulation of the relief's illumination out of surface reflectance, by the **C-correction** (Teillet, Guindon & Goodenough 1982, *Canadian Journal of Remote Sensing* 8:84-106) or **SCS+C** (Soenen, Peddle & Coburn 2005, *IEEE TGRS* 43:2148-2159). A pixel facing the sun is brighter than one facing away, whatever it is made of; the correction flattens that so that a classifier or an index sees the cover and not the slope. Written from the published formulas, not from any implementation, and computed in the Rust core (`zeit._core.topocorr`).

1. **Slope and aspect** of the DEM by Horn's (1981, *Proceedings of the IEEE* 69:14-47) 3 × 3 kernel, in the cube's projected units (metres; a foot-based CRS is converted). The slope is `atan(|∇z|)`, the aspect the direction the surface faces, clockwise from north. A geographic CRS is an error: reproject the cube and the DEM to the scene's UTM zone first.
2. **Illumination** `cos i = cos sz · cos s + sin sz · sin s · cos(saz − a)`, with the sun's zenith `sz` and azimuth `saz` and the slope `s` and aspect `a`.
3. **The line** `ρ = b + m · cos i`, fitted by least squares over the sample pixels of each band and date, and `c = b / m`.
4. **The correction** `ρ_H = ρ · (cos sz + c) / (cos i + c)` (`method="c"`) or `ρ_H = ρ · (cos s · cos sz + c) / (cos i + c)` (`method="scs+c"`; the sun-canopy-sensor geometry, for forests on steep slopes, which leaves a flat pixel's value times `cos s` on a slope).

It is multiplicative, so the units of the reflectance do not matter (scaled integers, 0-1, percent). It is an empirical flattening: Lambertian surface, the same cover on every slope, no diffuse or adjacent-terrain light, and a DEM about as fine as the image. FORCE's own C-factor, which is tied to its atmospheric correction's radiation terms, is a different step.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `DataArray` or path | required | Reflectance `(time, band, y, x)`, `(band, y, x)`, `(time, y, x)` or `(y, x)` in a projected CRS. A dask cube is read into memory (the line needs a whole date). NaN and the `nodata` values stay missing and stay out of the fit. |
| `dem` | `DataArray` or path | required | Elevation in metres `(y, x)` on **exactly** the cube's grid (same size and coordinates). It is never resampled: a mismatch is an error that suggests `dem.rio.reproject_match(cube)`. A NaN or NoData cell makes the slope of its eight neighbours NaN, and those pixels are left as they are. |
| `sun_zenith`, `sun_azimuth` | number, sequence, `DataArray`, `Dataset` or `"scene"` | `"scene"` | Degrees, azimuth clockwise from north. A number, one per date, or a per-pixel `DataArray` `(y, x)` / `(time, y, x)` on the cube's grid and dates. `"scene"` reads the cube's own bands `SZA` and `SAA` (per pixel; the angle bands are never corrected), else a coordinate or attribute `sun_zenith` (or `sun_elevation` or STAC's `view:sun_elevation`, zenith = 90 − elevation) and `sun_azimuth`. The output of [`sentinel2_angles`](#sentinel2_angles), or a `DataArray` with an `angle` dimension, may be given as either one and then serves both. A zenith outside 0-90° or missing leaves that pixel as it is. |
| `method` | `"c"` or `"scs+c"` | `"c"` | The correction above. |
| `sample` | boolean `DataArray` or array | `None` | `(y, x)` or `(time, y, x)`: the pixels the lines are fitted on (say the vegetated, clear ones), used as given. `None`: the pixels with a finite reflectance and `cos i` and a slope of at least `min_slope`. |
| `min_slope` | `float` | `5.0` | Degrees. Flat terrain has the same `cos i` (`cos sz`) everywhere and says nothing about the relief; in a mostly flat scene it would pull the line through one cluster. Below the limit a pixel is still corrected (its factor is 1). |
| `min_sample` | `int` | `100` | Fewest sample pixels for a line. |
| `bands` | list | `None` | The bands to correct; the default is all but `SZA`, `SAA`, `VZA`, `VAA`. The others are copied. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** the cube as float32 with the same dimensions, coordinates and georeferencing, missing values NaN, and a coordinate `topo_c` (`(time, band)` or `(band,)`): the `c` of every band and date, NaN where the band was copied. `attrs["topocorr"]` says the method and where the sun came from.

**Guards.** A band of a date is **copied unchanged**, with a warning that names it and why, when it has fewer than `min_sample` sample pixels, `cos i` has no variance, or the line's slope `m` is zero or negative (the reflectance does not rise with illumination: the topography does not drive it, or the sample is wrong). A pixel is **left as it is** when its `cos i + c` is below 0.01 (a deep shadow under a poorly fitted `c`), when `cos sz + c` (or `cos s · cos sz + c`) is not positive, when its sun angle is missing, or when its slope is NaN. The cells on the DEM's edge replicate the nearest cell and take the one-sided difference over one cell, so they have a slope; a flat cell has aspect 0.

**Against R.** RStoolbox's `topCor(method = "C")`, given Zeit's slope and aspect, gives the same corrected values to 4e-7 relative when the line is fitted on the same pixels (a 100 × 100 image, which RStoolbox samples whole); on a 1000 × 1000 image RStoolbox fits on a random sample of 10000 pixels (and so a slightly different `c`, a different one in each run) and the two differ by 0.3-4.5% at most, 1-2e-4 on average in reflectance (see the [benchmarks](../benchmarks/vs-originals.md#topographic-correction)). Slope and aspect agree with `gdaldem` (inside: 2.5e-4° in slope, the float32 of GDAL's output) and `terra::terrain` (8e-6°).

```python
dem = rioxarray.open_rasterio("dem.tif").squeeze(drop=True).rio.reproject_match(cube)
flat = zeit.topographic_correction(cube, dem, sun_zenith=38.5, sun_azimuth=142.0)
flat = zeit.topographic_correction(cube, dem, sun_zenith=angles, method="scs+c")   # angles from sentinel2_angles
flat.topo_c                     # the c of each band (and date)
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

### `tasseled_cap` { .api }

<!-- sig: zeit.tasseled_cap -->
```python
zeit.tasseled_cap(
    data, components=('brightness', 'greenness', 'wetness'), *,
    coefficients="crist1985", sensor=None, bands=None, nodata="auto",
    chunks=None, n_jobs=-1,
)
```

The Tasseled Cap of a reflectance cube: brightness (soil, built-up), greenness (green vegetation) and wetness (moisture, shade) as fixed weighted sums of the six reflective bands (blue, green, red, NIR, SWIR1, SWIR2), in the scale of the data. The default is what FORCE's TSA does (`TC-BRIGHT`, `TC-GREEN`, `TC-WET`, FORCE 3.11, David Frantz, GPL-3.0-or-later): the matrix of Crist (1985, *Remote Sensing of Environment* 17:301-306) for every sensor, in float32 with the six products added in the band order, so that FORCE's values truncated to integers are those of this function on the same integers. `"sensor"` gives each date the matrix published for its sensor instead. `"disturbance"` is FORCE's `TC-DI`, `brightness - greenness - wetness` of the same matrix, **not** the standardized disturbance index of Healey et al. (2005), which scales each component by the mean and standard deviation of the forest of the scene.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | Reflectance `(time, band, y, x)` or `(band, y, x)`, in memory or dask, with blue, green, red, NIR, SWIR1 and SWIR2 (found by role, as `spectral_adjust` does). Integers are accepted. For Sentinel-2 the NIR is B8A (`nir08`), FORCE's `NIR` for that sensor; a cube with B08 only is used with a warning. |
| `components` | `str` or list | `("brightness", "greenness", "wetness")` | What to compute, in this order: `"brightness"`, `"greenness"`, `"wetness"`, `"disturbance"`, or the names of a dict of `coefficients`. |
| `coefficients` | `str`, `dict` or array | `"crist1985"` | `"crist1985"` (FORCE's, every sensor); `"sensor"` (the matrix of each date's sensor, below); `"huang2002"`, `"baig2014"` (one matrix for every date); a dict `{component: six numbers (blue, green, red, NIR, SWIR1, SWIR2)}`; or an array `(components, 6)` whose rows are named by `components`. |
| `sensor` | `str` or list | `None` | Each date's sensor for `coefficients="sensor"`: by default the `platform` coordinate or attribute of the cube; or one name for every date (`"landsat-8"`, `"LC09"`, `"sentinel-2a"`...), or one per date. |
| `bands` | `dict` | `None` | The cube's band of each of the six roles (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`) when its names are not the usual ones. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN (the raster's NoData; 0 for integers without one). A pixel with a band missing is missing in every component. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` reads into memory; otherwise the result is lazy. |
| `n_jobs` | `int` | `-1` | Threads for an in-memory cube (a dask cube is parallel by its chunks). |

</div>

**Returns** `(time, band, y, x)` (or `(band, y, x)`), float32, whose `band` coordinate names the components, lazy if the input is. `attrs["tasseled_cap"]` says which matrix each sensor got, and a Tasseled Cap is refused a second time. Take one component for [`landtrendr`](change-detection.md#landtrendr) or the others with `.sel(band="wetness")`. [`compute_indices`](data.md#compute_indices) also has `TCB`, `TCG`, `TCW` and `TCD`, with Crist's matrix, for the reflectance cubes it takes.

**The matrices.** Coefficients for blue, green, red, NIR, SWIR1, SWIR2. Those of Crist (1985), Huang et al. (2002) and Baig et al. (2014) are equal to the table of RStoolbox's `tasseledCap` except for the two signs noted below; RStoolbox has no matrix for Sentinel-2.

**Sentinel-2.** `"sensor"` gives Sentinel-2 dates Crist's matrix, as FORCE does. The matrix published for MSI (Shi and Xu 2019, *IEEE JSTARS* 12:4038-4048) weighs all 13 bands of the L1C top-of-atmosphere product, B1, B9 and B10 included, which an L2A cube does not have, so it is not offered as a six-band matrix.

| Sensor | Paper | Derived for | Brightness | Greenness | Wetness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Landsat 4, 5 TM (and every sensor, by default) | Crist (1985) | reflectance factors (surface) | 0.2043 0.4158 0.5524 0.5741 0.3124 0.2303 | -0.1603 -0.2819 -0.4934 0.7940 -0.0002 -0.1446 | 0.0315 0.2021 0.3102 0.1594 -0.6806 -0.6109 |
| Landsat 7 ETM+ | Huang et al. (2002), *Int. J. Remote Sensing* 23:1741-1748 | at-satellite reflectance | 0.3561 0.3972 0.3904 0.6966 0.2286 0.1596 | -0.3344 -0.3544 -0.4556 0.6966 -0.0242 -0.2630 | 0.2626 0.2141 0.0926 0.0656 -0.7629 -0.5388 |
| Landsat 8 OLI (and 9) | Baig et al. (2014), *Remote Sensing Letters* 5:423-431 | top-of-atmosphere reflectance | 0.3029 0.2786 0.4733 0.5599 0.5080 0.1872 | -0.2941 -0.2430 -0.5424 0.7276 0.0713 -0.1608 | 0.1511 0.1973 0.3283 0.3407 -0.7117 -0.4559 |

Only Crist's matrix was derived for surface reflectance. The others were derived for top-of-atmosphere or at-sensor reflectance, and applying them to surface reflectance (as one does with Collection 2 or L2A cubes) is common practice but approximate; so is applying one matrix to every sensor, which is what FORCE does. RStoolbox's table for Landsat 5 TM has `+0.0002` and `+0.6806` for the SWIR1 weight of greenness and wetness, the opposite signs of Crist's table and FORCE's; Zeit follows the paper.

**Against FORCE and RStoolbox.** The integer outputs equal FORCE's in every observation computed (100% of 11.7 to 19.7 million values for each of a Sentinel-2A and Landsat 8, 7 and 5 synthetic cubes), and the matrices of ETM+ and OLI equal RStoolbox's to 1e-7 (see [Against the Original Implementations](../benchmarks/vs-originals.md#tasseled-cap-vs-rstoolbox-and-force)).

```python
landsat = zeit.build_time_series(collection="landsat-c2-l2",
                                 bands=["blue", "green", "red", "nir08", "swir16", "swir22"], **grid)
tc = zeit.tasseled_cap(landsat)                                      # as FORCE
tc = zeit.tasseled_cap(landsat, ["wetness"], coefficients="sensor")  # the matrix of each date's sensor
lt = zeit.landtrendr(tc.sel(band="wetness"))
```

## Sharpening coarse data (ImproPhe)

### `improphe` { .api }

<!-- sig: zeit.improphe -->
```python
zeit.improphe(
    coarse, fine, *, by="auto", seasons=(1, 91, 181, 271, 365),
    kernel_size=2500.0, texture_size=330.0, pc_variance=0.975,
    force=False, pca=None, nodata="auto", chunks=None, n_jobs=-1,
)
```

Sharpens a coarse field, or the scenes of a coarser sensor, to the pixels of a finer one: each pixel's value is the weighted mean of the coarse values at the pixels of its neighbourhood that look like it in the fine data. It is FORCE's CF-ImproPhe (a continuous field with a layer for each year, such as a 500 m phenology metric, sharpened with the Sentinel-2 or Landsat scenes of that year) and L2-ImproPhe (Landsat 30 m scenes sharpened with Sentinel-2 10 m), FORCE 3.11 (David Frantz, GPL-3.0-or-later), the method of Frantz, Stellmes, Roeder, Udelhoven, Mader and Hill (2016, *IEEE Transactions on Geoscience and Remote Sensing* 54(7):4153-4164).

The fine scenes (of the year, or all of them for L2) are averaged in the seasonal windows of the year, which gives one spectrum of `windows x bands` values per pixel; principal components keep 97.5% of their variance and are standardized, and the largest standard deviation of the components in a square of radius `texture_size` is the fine texture. Around each pixel, in a circle of radius `kernel_size`, a neighbour is kept if the mean absolute difference S of its components to the pixel's is under one of the five thresholds 0.025, 0.05, 0.1, 0.2 and 0.4: the smallest that keeps at least 0.5% of the circle's pixels (the pixel itself counts). The weight of a kept neighbour is 1 / (S T U), where T is the neighbour's fine texture and U the standard deviation of its coarse values in the texture square, each of the three factors rescaled over the kept neighbours to `1 + exp(25 x - 7.5)` with x from 0 to 1 (a value of 0 counts 1). A pixel with no neighbour (no fine data) keeps its coarse value; a pixel without a coarse value is NaN.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `coarse` | `DataArray` or path | required | What to sharpen, on any grid of the same CRS (nearest neighbour onto the fine grid, as FORCE's read does; for an integer factor that is the replication of each pixel). With `by="year"`: `(year, y, x)` or `(year, field, y, x)`; the layer dimension is `year`, or `time` with one date per year, or `band` named by the years. With `by=None`: scenes `(time, band, y, x)` or `(time, y, x)`. NaN and the NoData value are missing. |
| `fine` | `DataArray` or path | required | The fine reflectance `(time, band, y, x)` with a datetime `time` coordinate, NaN where missing (mask clouds first). Any scale; with `force=True` the seasonal means are cut to integers, which is FORCE's only for reflectance times 10000. Its grid is the output's. |
| `by` | `"year"`, `None` or `"auto"` | `"auto"` | `"year"`: CF-ImproPhe, each year with the fine scenes of that year. `None`: L2-ImproPhe, each coarse scene with the seasonal means of all the fine scenes. `"auto"`: a `year` dimension means `"year"`, several dates in a year mean `None`, else it asks. |
| `seasons` | sequence of `int` | `(1, 91, 181, 271, 365)` | The days of the year that limit the seasonal windows (FORCE's `SEASONAL_WINDOW`), `n + 1` for `n` windows, in FORCE's 365-day calendar. A limit larger than the next makes a window that crosses the new year (the days of the neighbouring years count). A window with no scene at a pixel takes the mean of all of them. |
| `kernel_size` | `float` | `2500.0` | Radius of the circle, in map units (FORCE's `KERNEL_SIZE`), cut to whole pixels of `fine`; a multiple of the pixel size is expected (FORCE refuses others; here a warning). The default suits 10 m pixels (250 pixels). |
| `texture_size` | `float` | `330.0` | Radius of the square of the texture, in map units (FORCE's `KERNEL_TEXT`), 33 pixels at 10 m. |
| `pc_variance` | `float` | `0.975` | Fraction of the variance the components keep. |
| `force` | `bool` | `False` | FORCE's arithmetic to the letter (below). |
| `pca` | `str` or `None` | `None` | The components: `"principal"` (the paper's) or `"force"` (FORCE's projection, below); `None` takes FORCE's with `force=True` and the principal ones otherwise. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Values that mean missing besides NaN: the raster's NoData of each input (0 for an integer `fine` without one, as `tasseled_cap`), a number for both, or `None`. |
| `chunks` | `"auto"`, `dict` | `None` | Inputs read from disk: `None` in memory; a dask input is loaded in any case. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default). |

</div>

**Returns** float32, the layers of `coarse` on the grid and coordinates of `fine`: `(year, [field], y, x)` or `(time, [band], y, x)`. `attrs["improphe"]` says the variant, the radius and texture radius in pixels, the minimum number of neighbours, the number of components (per year, or once) and `force`. A year (CF) or the whole series (L2) with no fine data gives the coarse values themselves.

```python
# CF-ImproPhe: a yearly 500 m field (year, y, x) with the Sentinel-2 cube of the same years, 10 m pixels
sharp = zeit.improphe(lsp_500m, sentinel2, by="year", kernel_size=500, texture_size=60)
# L2-ImproPhe: Landsat scenes (time, band, y, x) at 10 m from the same cube
l8_10m = zeit.improphe(landsat, sentinel2, by=None, kernel_size=300, texture_size=60)
# FORCE's arithmetic, to compare with its outputs after truncating to integers
same_as_force = zeit.improphe(lsp_500m, sentinel2, force=True, kernel_size=500, texture_size=60)
```

**Where it differs from FORCE.** The kernel (the neighbours, the thresholds, the weights and their order of summation) is FORCE's, in float32 and float64 as its own. With `force=False`, two differences are deliberate, and a third is a FORCE quirk that is left out:

- *The components.* FORCE projects the data on the transposed matrix of eigenvectors that GSL returns (`gsl_blas_dgemm(NoTrans, Trans, ...)` in `pca-cl.c`), that is on its rows and not its columns: its "components" are a rotation of the data that is not the principal axes, and the signs of GSL's eigenvectors change the distances. By default Zeit projects on the eigenvectors (the paper's PCA; with that projection the sign of an eigenvector does not matter) and takes them from a symmetric eigensolver. With `force=True` the eigenvectors are those of GSL 2.7's `gsl_eigen_symmv`, ported literally (Householder reduction, implicit QR, selection sort), and the projection is FORCE's. `pca="force"` takes FORCE's projection alone, with float means. Neither projection is better throughout on the synthetic data of the benchmarks: FORCE's was up to 2.8 times better in one year of a CF field and somewhat worse in others, and with float means (other eigenvector signs) it lost that gain and was 38% worse in the other year; the principal components do not depend on the signs, so they are the default.
- *The seasonal means.* FORCE keeps them as 16-bit integers, truncated toward zero, and excludes the upper limit of every window, so day 365 (and 366) is in no window of the default seasons. By default the means are floats and the last window includes its upper limit. `force=True` reproduces both.
- *An empty year.* If a year has no fine data FORCE copies the coarse field for that year and returns from the whole function, leaving the later years at 0. Zeit copies it for that year and goes on with the others.

FORCE writes int16, the result truncated toward zero, and where it does not predict it writes the coarse value as it is (an L2 scene pixel that its QAI screens, a pixel without fine data), 0 (a CF pixel whose coarse value is NoData) or the NoData value (an L2 pixel whose scene is NoData). Zeit returns float and gives NaN wherever the coarse value is missing, and the coarse value where there is no fine data.

**Agreement with FORCE.** On synthetic Sentinel-2A (10 m), Landsat 8 (30 m) and field (30 m) data, `zeit.improphe(..., force=True)` truncated to integers equals FORCE's built against GSL's reference CBLAS in every value (2 x 88,065 pixels for CF and 12.5 million values for L2, at radii of 10 and 20 pixels and two sets of seasons); FORCE as the reference build links it, to OpenBLAS, differs in the last bits of the eigenvectors and the results in 0.001-0.003% of the values by 1 or, rarely, more. The tables, and what the transposed projection costs in accuracy, are in [Against the Original Implementations](../benchmarks/vs-originals.md#improphe-vs-force). FORCE 3.11.00's own L2IMP does not start as released (it never fills in the settings of the secondary sensors), so that comparison used a copy of its build with those settings filled in.

**Cost and memory.** The work for every year or scene is the circle of radius `kernel_size` around each pixel (`pi h^2` neighbours of a few components each): on 6 threads of a desktop, 8e5 pixels/s of one year or scene at a radius of 10 pixels, 6e4 at 50, and 25 times the cost of 50 at the default 250 (minutes for a 1000 x 1000 tile). Zeit is 1.2 to 2.4 times as fast as FORCE's compute stage on the same data (see the benchmarks). Everything is in memory (the cube, the coarse field and the result), with no chunking: the components' statistics are those of the whole extent and the circle reaches `kernel_size` beyond any block. A dask input is computed first; for a large area the caller works a tile at a time, with a margin of `kernel_size`.

### `sentinel2_resmerge` { .api }

<!-- sig: zeit.sentinel2_resmerge -->
```python
zeit.sentinel2_resmerge(
    cube, *, method="regression", clear=None, force=False, bands=None,
    scale="auto", nodata="auto", n_jobs=-1,
)
```

The 20 m bands of Sentinel-2 (red edge 1-3, B8A, SWIR1, SWIR2) at 10 m, predicted from the 10 m green, red and broad NIR: FORCE's resolution merge of its Level 2 processing (`resmerge-ll.c`, FORCE 3.11, David Frantz, GPL-3.0-or-later). FORCE's third option, STARFM (the USDA's code), is not part of it. The input is a cube already on the 10 m grid, where each 20 m pixel is four equal pixels: that is how FORCE gets them (it reads the 20 m files into the 10 m buffer with GDAL's `RasterIO`, whose resampling is the nearest neighbour), and a nearest-neighbour `reproject_match` of the 20 m bands gives the same.

`method="regression"` fits, for each pixel and each of the six bands, `y = c0 + c1 green + c2 red + c3 NIR` by least squares over the 5 x 5 pixels around it (GSL's `gsl_multifit_linear`: the singular value decomposition of the column-balanced matrix, singular values at or below `epsilon` times the largest dropped) and evaluates it at the pixel. The pixel is in its own window; pixels with no data are left out, and a window with fewer than 25 pixels is filled with rows `(1, 0, 0, 0)` and `y = 0` (FORCE's: it pulls the fit towards zero at the borders and next to gaps); clouds and shadows are kept in the window. `method="improphe"` is ImproPhe's weighted mean with spectral instead of temporal similarity: the pixels of the disc of radius 2 that are not cloud, shadow or no data and whose mean absolute difference of green, red and NIR (reflectance) to the pixel is under 0.0025, 0.005, 0.01, 0.02 or 0.025 are the neighbours; the smallest of those cutoffs that has at least 3 of them is used, and each band is the mean of the neighbours' values weighted by `1 / (1 + exp(25 x - 7.5))`, `x` the difference rescaled over the neighbours of the cutoff. It is faster than the regression.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `DataArray` | required | Reflectance `(time, band, y, x)` or `(band, y, x)`, each date on its own, all bands on the 10 m grid, with `green red nir rededge1 rededge2 rededge3 nir08 swir16 swir22` (`nir` is B08, `nir08` B8A) or `B03 B04 B08 B05 B06 B07 B8A B11 B12`. Other bands pass through. In memory (a lazy cube is computed). |
| `method` | `"regression"` or `"improphe"` | `"regression"` | FORCE's `RES_MERGE = REGRESSION` or `IMPROPHE`. |
| `clear` | boolean array | `None` | Broadcastable to `(time, y, x)`: True where the pixel is not cloud or shadow. By default every pixel with data is. Pixels that are not clear are not predicted. |
| `force` | `bool` | `False` | FORCE's arithmetic (below). |
| `bands` | `dict` | `None` | The cube's band of each of the nine roles, when its names are not the usual ones. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by in the data (10000 for integers or values above 2, else 1). The thresholds of `"improphe"` are in reflectance and the regression's coefficients are not scale-free, so a cube in 0-1 and one in 0-10000 give the same. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN: the raster's NoData (0 for an integer cube without one), a number or `None`. A pixel with a missing value in any of the nine bands is FORCE's "off": left out of the windows and not predicted. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default). |

</div>

**Returns** float32, as `cube`, with the six bands predicted at the pixels that are neither off nor cloud or shadow, and every other value as it was. `attrs["sentinel2_resmerge"]` says the method.

```python
s2 = zeit.build_time_series(collection="sentinel-2-l2a", resolution=10, ...)   # the 20 m bands by nearest neighbour
sharp = zeit.sentinel2_resmerge(s2, method="regression", clear=cloud_free)
fast = zeit.sentinel2_resmerge(s2, method="improphe", clear=cloud_free)
```

**Where it differs from FORCE.** FORCE's regression writes each prediction into the band before the next pixel's window reads it (row-major with one thread; with OpenMP's several it is a race, so its output depends on the thread schedule). With `force=False` every window reads the original values, the pixels run in parallel and the output is float32, not truncated. `force=True` is FORCE's arithmetic: the values are taken as reflectance times 10000 (a cube in another `scale` is converted and rounded), predictions are truncated to integers (`(short)(v * 10000)`) held in float32, and for `"regression"` the sweep is FORCE's, in place, row-major, on one thread. The feedback is not a detail: on the synthetic scene of the benchmark the two differ by 11-20 digital numbers on average and up to 490 (the windows of `force=True` read smoothed predictions where those of `force=False` read the blocks of the 20 m bands), and `force=False` matches an independent least squares (numpy's `lstsq`, 216 pixels) to 1e-4 digital numbers. For `"improphe"` there is no feedback: the two modes differ by the truncation.

**Agreement with FORCE.** `force=True` equals the code of FORCE 3.11.00 in every value, for both methods, on a synthetic 500 x 500 scene with no-data pixels, clouds, shadows, constant spectra (a rank deficient fit) and zeros (6 bands x 245,534 pixels per method): the C harness is FORCE's own functions on plain arrays, linked to the real GSL 2.7 and its reference CBLAS, run on one thread (see [Against the Original Implementations](../benchmarks/vs-originals.md#the-resolution-merge-vs-force)). FORCE has no module that runs this on an ARD tile; it runs only inside `force-l2ps`.

## Co-registration


### `pansharpen` { .api }

<!-- sig: zeit.pansharpen -->
```python
zeit.pansharpen(
    ms, pan, method="gsa", *, ratio=None, weights=None, intensity_bands=None,
    kernel=None, gain="regression", nodata="auto", n_jobs=-1,
)
```

Multispectral bands on the grid of a finer panchromatic band, for any sensor: any number of bands with any names, any pair of resolutions with a ratio above 1 (whole or not: Landsat 8's 15 m band 8 and 30 m bands give 2, 2.5 m and 10 m give 4, 1.5 m and 6.5 m give 4.33), taken from the grids' georeferencing. The CBERS-4A WPM (a 2 m pan; blue, green, red and NIR at 8 m) is one example; Landsat 8/9 (band 8), Sentinel-2 (10 m with 20 m or 60 m bands) and any other pair of rasters are others. The bands are resampled to the pan's grid with the cubic kernel of the warp of `load_raster(like=)` and the pan's detail is injected by one of four methods, written from their papers (the equations, the references and the choices the papers leave open are in `rust/zeit-core/src/pansharpen.rs`). No unit is converted: the output has those of `ms` (digital numbers, radiance or reflectance).

`"gsa"` (default) is the Gram-Schmidt adaptive of Aiazzi, Baronti and Selva (2007, IEEE TGRS 45(10):3230-3239): `F_k = MS_k + g_k (P' - I)`, `g_k = cov(MS_k, I) / var(I)`, `P'` the pan matched to the mean and deviation of `I`, and `I` the intensity, the regression (with an intercept) of the pan reduced to the bands' resolution on the bands. `"brovey"` is the ratio transform, `F_k = MS_k P / I`: with `weights`, `I` is their mean (the pan in the units of the bands); without them the pan's response is unknown and `I` is fitted by the regression through the origin with weights not negative, which absorbs a gain between pan and bands and gives bands outside the pan's range a weight near 0. `"hpf"` is the high-pass filter of Schowengerdt (1980) with the kernel and modulation of Gangkofner, Pradhan and Holcomb (2008, PE&RS 74(9):1107-1118): `F_k = MS_k + gain sd(MS_k) / sd(LP) (P - LP)`, `LP` the box mean of the pan (side `2 floor(ratio / 2) + 1`, the box that matches the bands' cell: 3 for ratio 2, 5 for 4), computed in O(1) per pixel; by default (`gain="regression"`, zeit's choice) each band gets its least squares slope on `LP` instead of the paper's modulation `sd(MS_k) / sd(LP)` (`gain=1.0`), which over-sharpens bands that the pan does not explain and did not beat the cubic resampling in ERGAS on the benchmark scenes. `"pca"` replaces the first principal component by the pan matched to it (to compare with products made so, such as the INPE's `CB4A-WPM-PCA-FUSED-1`).

**Bands outside the pan's range** (the NIR or the SWIR for a visible pan; the pan's spectral response is not an input). GSA and the fitted Brovey estimate the intensity by regression, so such a band gets a weight near 0 in `I`, and in GSA a gain `g_k` that follows its covariance with `I`: little detail goes in. The Brovey ratio multiplies every band by `P / I` whatever the band (a known weakness of the ratio form: prefer GSA). HPF has no intensity and gives every band detail in proportion to its spread (the default `gain="regression"` damps the bands that the pan does not explain; the paper's modulation, `gain=1.0`, does not). PCA takes the first component of all the bands, which a band the pan does not see can dominate. `intensity_bands` chooses the bands that enter the estimate of the intensity (GSA, PCA, fitted Brovey), by label of the band coordinate or by position, with no names assumed; the others are still sharpened. All four methods work from 1 band except `"pca"`, which needs 2 (and at least 2 in `intensity_bands`).

**Sources.** No third-party code was used, and the formulas were written from the literature as recalled, without the papers at hand; they were checked by their properties, by Wald's protocol on synthetic scenes and on a CBERS-4A WPM and a Landsat 8 window and, for the Brovey, against GDAL's output. The defaults of `"hpf"` are zeit's and not Gangkofner et al.'s: the kernel and the regression gain were chosen on those two real windows, so treat them as a starting point.

**Grids.** The CRS must be the same (nothing is reprojected), the grids north-up, the pan's cells finer and the footprints overlapping; the ratio comes from the cell sizes (the geometric mean over x and y; `ratio=` overrides it). The corners need not coincide: the warp puts the bands on the pan's cells at any offset. What that does to the statistics: the pan reduced to the bands' resolution for the regression (a box filter of side `2 floor(ratio / 2) + 1` sampled every `round(ratio)` pixels) is then taken at the pan pixel nearest each cell's centre, off by up to half a pixel, which the filter's width hides; with whole blocks the reduction would be the cell's exact mean. A ratio that is not whole rounds the filter and the HPF kernel. Pan cells outside the bands' footprint are NaN. The registration of the pan to the bands (geolocation) is not checked or corrected: use [`coregister`](#coregister) first.

**Missing data.** A pixel that is NaN (or `nodata`) in the pan or in any band is NaN in the output and out of every statistic (the regression, means, variances and principal components; per date, over the whole scene); a missing cell of a band makes the pan cells it covers NaN.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ms` | `DataArray` | required | `(time, band, y, x)` or `(band, y, x)` with a CRS. In memory (a lazy cube is computed). |
| `pan` | `DataArray` | required | `(y, x)` (or one `band`) on the finer grid, or `(time, y, x)` matching `ms`'s dates. |
| `method` | `str` | `"gsa"` | `"gsa"`, `"brovey"`, `"hpf"` or `"pca"`. |
| `ratio` | `float` | `None` | Pan cells per band cell, when it must differ from the ratio of the cell sizes. |
| `weights` | sequence or `dict` | `None` | `"brovey"`: the weights of the intensity (in the bands' order, or by label; normalised to sum 1). Default: fitted. |
| `intensity_bands` | list | `None` | `"gsa"`, `"pca"`, fitted `"brovey"`: the bands (labels or positions) that enter the estimate of the intensity. Default: all. |
| `kernel` | `int` | `None` | `"hpf"`: the odd side of the box filter in pan pixels; `2 floor(ratio / 2) + 1` by default. |
| `gain` | `float` or `"regression"` | `"regression"` | `"hpf"`: each band's regression slope on the low-passed pan, or a number: the weight of the paper's modulation (1.0 is Gangkofner's). |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN, in both inputs: the raster's NoData (0 for integer data without one), a number or `None`. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default). The output does not depend on it. |

</div>

**Returns** float32, the bands of `ms` on the pan's grid, with the coordinates, CRS and attributes of `ms` (`attrs["pansharpen"]` says the method and the ratio). Everything is in memory: a 10,000 x 10,000 pan with four bands is 1.6 GB of output plus the resampled bands. A whole scene larger than that (a WPM scene is about 46,000 pixels across) needs tiles with a margin for the cubic resampling and the HPF's kernel, and the statistics of the whole scene computed beforehand and handed to the tiles.

```python
fused = zeit.pansharpen(wpm_ms, wpm_pan, method="gsa")                       # CBERS-4A WPM, ratio 4
fused = zeit.pansharpen(l8_ms, l8_b8, "gsa", intensity_bands=["B3", "B4"])   # Landsat 8: the pan sees green and red
```

### `coregister` { .api }

<!-- sig: zeit.coregister -->
```python
zeit.coregister(
    cube, reference, *, window_size=256, window_position=None, max_shift=5, max_iter=5,
    band=None, ref_band=None, resampling="cubic", nodata="auto", min_reliability=None, n_jobs=-1,
)
```

Moves each date of a cube to fit a reference image: the global co-registration of **AROSICS** (Scheffler et al. 2017, *Remote Sensing* 9(7):676; `COREG` of AROSICS 1.14.0, Apache-2.0, Daniel Scheffler, GFZ Potsdam), ported to the compiled core (`rust/zeit-core/src/coreg.rs`, which Janus links directly). For each date, one band is matched to the reference's by the **phase correlation** of a window at the centre of what the two images have in common: the cross power spectrum `F0 conj(F1) / (|F0| |F1|)` of the two windows (single-precision FFTs of the largest power of 2 inside the window), whose peak is their shift in whole pixels and whose side maxima give the part of a pixel. The integer shift is checked by moving the reference's window by it and correlating again (up to `max_iter` times); a shift above `max_shift` pixels, or with no clear match, is a failure. Every band of the date is then moved by the shift with the warp kernel of `load_raster(..., like=)` (GDAL's resampling, `resampling="cubic"` by default) onto the cube's own grid; a shift of a whole number of pixels is a plain copy of the cells.

The cube and the reference must be on the same pixel grid (the same projected CRS, the same cell size, the cells a whole number apart; their extents may differ and only what they share is matched). Where AROSICS resamples the window of the other image when the grids differ, `coregister` asks for the reference to be put on the cube's grid first (`zeit.load_raster(..., like=cube)`), and says so in the error.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `DataArray` | required | `(time, band, y, x)`, `(band, y, x)`, `(time, y, x)` or `(y, x)` in a projected CRS. A lazy cube is read a date at a time and the result is lazy. |
| `reference` | `DataArray` | required | `(band, y, x)` or `(y, x)` in the same CRS, on the same pixel grid. |
| `window_size` | `int` | `256` | The matching window's side in pixels (AROSICS' `ws`). It is cut down to the overlap, and the FFT takes the largest power of 2 inside it. |
| `window_position` | `(x, y)` | `None` | The window's centre in map coordinates (AROSICS' `wp`); by default the centroid of the overlap of the valid data. |
| `max_shift` | `float` | `5` | The largest shift accepted, in pixels. |
| `max_iter` | `int` | `5` | Tries to validate the integer shift. |
| `band`, `ref_band` | label or position | `None` | The band of the cube and of the reference to match (the first by default). |
| `resampling` | `str` | `"cubic"` | The resampling that moves the images (those of `load_raster`: `"cubic"`, `"bilinear"`, `"nearest"`, `"lanczos"`...). |
| `nodata` | `"auto"`, number, `None` or `(reference, cube)` | `"auto"` | Missing values besides NaN: each image's raster NoData, a number for both, `None`, or a pair. |
| `min_reliability` | `float` | `None` | Treats a shift whose reliability (AROSICS', 0-100) is below this as not found. `None` takes every shift AROSICS' checks accept, as its `COREG` does. |
| `n_jobs` | `int` | `-1` | Dates run at once (the processors zeit may use by default); each date's correlation is one thread. |

</div>

**Returns** `(aligned, shifts)`. `aligned` is the cube on its own grid with each date moved by its shift; the dates that failed are as they were. `shifts` is a `pandas.DataFrame`, one row per date (index `time`; one row for a cube without dates), with `x_shift_px` and `y_shift_px` (pixels: the amount the date was moved, `x` to the right and `y` down, AROSICS' convention), `x_shift_map` and `y_shift_map` (map units; `y` is negative for a shift down on a north-up grid), `reliability` (0-100), `ssim_before` and `ssim_after` (structural similarity of the window to the reference's before and after the shift), `success`, `reason` (why it failed), the matching window `window_x0, window_y0, window_x1, window_y1` (cells of the cube, `[x0, x1)`), `fft_size` (the side of the FFT window of the spectrum the shift is read from: after an integer shift the reference's window is moved by it and shrinks, so it is smaller than the matching window's largest power of 2) and `applied`.

```python
aligned, shifts = zeit.coregister(cube, reference, band="nir", ref_band="nir")
shifts[["x_shift_px", "y_shift_px", "reliability", "ssim_before", "ssim_after", "success"]]
good = aligned.sel(time=shifts.index[shifts.success])
```

**A date that fails stays as it was.** No valid data in common, an overlap of 16 x 16 pixels or less, a window that cannot be placed on valid data, no integer shift that survives the check in `max_iter` tries, or a shift above `max_shift` (AROSICS raises these as errors; here they are `success=False` with the `reason`). A featureless date (constant) fails that way. AROSICS itself applies any shift that passes those checks, however low its reliability or similarity (its local variant filters tie points by them); `min_reliability` is Zeit's option to refuse the weak ones, and `ssim_before` and `ssim_after` are there to check that the window agrees better afterwards.

**Where it differs from AROSICS.** The numbers are AROSICS' (see below); the plumbing is not:

- The overlap is that of the two images' *valid pixels* (not NaN, not NoData), and the window is placed on it as AROSICS places it on the intersection of the footprint polygons: the centroid, clipped to the overlap, moved to the grid by the floor of its edges, and cut down by one pixel a side when AROSICS' other window does not fit, as it does in an image smaller than the window. AROSICS fills the holes of a footprint, so a cloud masked as NaN is inside its window; here the window is grown from its centre until it meets the hole (as AROSICS grows it where its footprint polygon has an edge), or fails when nothing of 16 x 16 pixels is left.
- The aligned images are resampled onto the cube's grid; AROSICS with `align_grids=False` only moves the origin of the geotransform. Zeit's grid stays the cube's, which is what a time series needs.
- The window's values are float32 (AROSICS reads the file's type and converts to single precision for the FFT; the similarity of an integer image rounds and clamps the shifted window to the type as GDAL does).
- In AROSICS' retry loop, a shift found by a retry gets `(0, 0)` as its integer part (a quirk of `calculate_spatial_shifts`); it is kept, since the shifts are meant to be AROSICS'.
- This is the global variant; the local one, with a grid of tie points and a warp fitted to them, is [`coregister_local`](#coregister_local).

**Agreement with AROSICS 1.14.0.** On 13 pairs (12 synthetic and a natural image; the two where neither finds a shift fail alike), the shifts differ from AROSICS' by 2e-7 pixels at most (the single-precision noise of two FFT libraries); the integer part, the window, the FFT size and the reliability are equal; the similarity too on the synthetic pairs; and the two aligned images differ by one unit of float32 (see [Against the Original Implementations](../benchmarks/vs-originals.md#the-coregistration-vs-arosics)).

### `coregister_local` { .api }

<!-- sig: zeit.coregister_local -->
```python
zeit.coregister_local(
    cube, reference, *, grid_res=200, window_size=(256, 256), max_shift=5, max_iter=5,
    min_reliability=60, tie_point_filter_level=3, rs_max_outlier=10, rs_tolerance=2.5,
    rs_random_state=0, warp="auto", order=None, min_points=5, band=None, ref_band=None,
    resampling="cubic", nodata="auto", arosics_compat=False, n_jobs=-1,
)
```

Warps each date of a cube to fit a reference image by shifts that vary across the scene: the local co-registration of **AROSICS** (`COREG_LOCAL` of AROSICS 1.14.0, Apache-2.0, Daniel Scheffler, GFZ Potsdam; Scheffler et al. 2017, *Remote Sensing* 9(7):676), ported to the compiled core (`rust/zeit-core/src/coreg_local.rs`, with the matching of [`coregister`](#coregister) for each point). A grid of tie points is laid over the image, every `grid_res` pixels from the corner (AROSICS' own grid: `0, grid_res, 2 * grid_res, ...` up to one step past the image); each point on data of both images is matched with a window of `window_size` pixels centred on it, which gives the shift there, its reliability and the structural similarity of the window before and after. The points are then filtered in AROSICS' order: **level 1**, a reliability below `min_reliability`; **level 2**, a window that is less similar to the reference's after the shift; **level 3**, **RANSAC**: scikit-image's `ransac` of an affine model of the shifts, with 6 samples, up to 2000 trials and a residual threshold that AROSICS searches (in map units) until the outliers are `rs_max_outlier` percent plus or minus `rs_tolerance`. The points that remain are control points: each date is warped onto the cube's grid by GDAL (through rasterio) from them, with a **thin plate spline**, which AROSICS always uses (`warp="tps"`, what `"auto"` is), or a polynomial (`warp="polynomial"`, GDAL's choice of order from the number of points or an `order` of 1 to 3). It corrects what a single shift cannot: a misregistration that grows across the scene, a bad orthorectification, a viewing-geometry effect.

As in [`coregister`](#coregister), the cube and the reference must be on the same pixel grid and the overlap is that of the valid pixels (NaN, NoData); the points are on the cube's own pixel corners.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `DataArray` | required | `(time, band, y, x)`, `(band, y, x)`, `(time, y, x)` or `(y, x)` in a projected CRS. A lazy cube is read a date at a time; the result is in memory. |
| `reference` | `DataArray` | required | `(band, y, x)` or `(y, x)` in the same CRS, on the same pixel grid. |
| `grid_res` | `float` | `200` | The spacing of the tie points in pixels (AROSICS' `grid_res`). |
| `window_size` | `int` or `(x, y)` | `(256, 256)` | The matching window's side in pixels; a pair must be of equal values (the window is square). |
| `max_shift`, `max_iter` | `float`, `int` | `5`, `5` | As in `coregister`, for each point. |
| `min_reliability` | `float` | `60` | Tie points below this reliability (0-100) are flagged (level 1). |
| `tie_point_filter_level` | `int` | `3` | AROSICS' `tieP_filter_level`: 0 none, 1 reliability, 2 plus the similarity, 3 plus RANSAC. |
| `rs_max_outlier`, `rs_tolerance` | `float` | `10`, `2.5` | RANSAC's expected outliers and the tolerance of that, in percent. |
| `rs_random_state` | `int` or `None` | `0` | RANSAC's seed (AROSICS' `rs_random_state`); `None` takes one from the clock. With the same seed the flags are AROSICS'. |
| `warp` | `str` | `"auto"` | `"tps"` (GDAL's thin plate spline), `"polynomial"`, or `"auto"` (`"tps"`, as AROSICS). |
| `order` | `int` | `None` | The polynomial's order, 1 to 3 (`warp="polynomial"`); `None` lets GDAL choose from the number of points. |
| `min_points` | `int` | `5` | A date with fewer control points (AROSICS' `min_points_local_corr`) is left as it was and flagged. |
| `band`, `ref_band` | label or position | `None` | The band of the cube and of the reference to match (the first by default); every band of a date is warped with the same points. |
| `resampling` | `str` | `"cubic"` | The resampling of the warp (`"cubic"`, `"bilinear"`, `"nearest"`, `"lanczos"`...). |
| `nodata` | `"auto"`, number, `None` or `(reference, cube)` | `"auto"` | Missing values besides NaN: each image's raster NoData, a number for both, `None`, or a pair. |
| `arosics_compat` | `bool` | `False` | Reproduces a bug of AROSICS 1.14.0 in the RANSAC flags (below), to compare with it. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default): the points of each date, and the dates, run on them. |

</div>

**Returns** `(aligned, tie_points)`. `aligned` is the cube on its own grid with each date warped (the dates with fewer than `min_points` control points as they were). `tie_points` is a `pandas.DataFrame` with a row for every tie point of every date: `time`, `point_id` (the position in the whole grid, AROSICS' `POINT_ID`), `col` and `row` (the corner of the cube's pixels), `x` and `y` (map), `x_shift_px` and `y_shift_px`, `x_shift_map` and `y_shift_map` (the amount the date has to be moved there, `x` to the right and `y` down, as in `coregister`; NaN where the match failed), `abs_shift` (map units) and `angle` (degrees from north), `reliability`, `ssim_before`, `ssim_after` and `ssim_improved`, `success` and `reason`, `win_size_x` and `win_size_y`, `fft_size`, the flags of the filters `outlier_reliability`, `outlier_ssim` and `outlier_ransac` and their union `outlier`, `used` (the point is a control point) and `date_applied`. `tie_points.attrs["dates"]` is a table with one row per date: the points placed, matched and used, whether the date was warped, and why not.

```python
aligned, points = zeit.coregister_local(cube, reference, grid_res=100, window_size=128, band="nir", ref_band="nir")
points[points.used][["time", "x", "y", "x_shift_map", "y_shift_map"]]
points.attrs["dates"]
```

**A point or a date that fails is flagged, not raised.** A point whose window cannot be placed on data, with no clear match or a shift above `max_shift` has `success=False` and a `reason`; a date with too few control points is returned as it was (`date_applied=False`; AROSICS in that case corrects with the mean shift of the points, Zeit leaves the date alone, since a global shift is [`coregister`](#coregister)'s job). A date with more than 7000 control points is warped from 7000 evenly spaced ones (AROSICS draws them at random: a limit of GDAL's spline).

**Where it differs from AROSICS.**

- As in `coregister`, the overlap is the valid pixels, not the footprint polygons. A window that touches a diagonal edge or a hole is grown from the point until it meets the no-data pixels, as AROSICS grows it inside its polygon; the same windows on the benchmark's pairs, which have a diagonal edge.
- **RANSAC's flags are put on their points; AROSICS 1.14.0 puts them on others.** When the reliability or similarity filter has flagged points, AROSICS gives the flags of RANSAC (one for each point RANSAC got) to the table's rows by their labels instead of their positions (`inGDF['outliers'] = outliers` in `_RANSAC_outlier_detection`), so the flags land on the wrong points: real outliers stay and as many good points are dropped. Zeit's default flags the points RANSAC found. `arosics_compat=True` reproduces AROSICS' assignment, which is how the comparison below is exact.
- RANSAC is scikit-image's, with numpy's random generator (PCG64 seeded by a `SeedSequence`, `Generator.choice` without replacement) reproduced in the core, so the samples, the inliers at every threshold and the flags are AROSICS'. The time limit of the threshold search (`rs_timeout`, 20 s of clock) is not kept; with 5 or 6 points left for RANSAC (where AROSICS raises or returns nothing) none is flagged.
- The warp is GDAL's, as AROSICS' (rasterio's `reproject` with the control points, `SRC_METHOD=GCP_TPS`), onto the cube's own grid.
- Points run on threads of the core where AROSICS starts a process for each.

**Agreement with AROSICS 1.14.0.** On 7 pairs (6 synthetic and the `camera` image warped by a field; 19 to 165 points each), the points of the grid that are placed are the same, and so are the successes; the shifts differ from AROSICS' by 2e-6 pixels at most (2e-3 at one point of the natural image), the windows are equal, the reliability agrees to 8e-5 and the similarity before to 4e-8 and after to 2e-5 (3e-4 in the 8-bit image); the flags of the three filters are identical with `arosics_compat=True`; and the warped image, from the same control points, differs by 0.001 at most in float32 (by 1 in uint16), the 8-bit image by up to 6 in 0.01% of the cells (see [Against the Original Implementations](../benchmarks/vs-originals.md#the-local-coregistration-vs-arosics)).

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
| `data` | `DataArray` or path | required | A `(time, band, y, x)` cube with green and SWIR bands, or one pixel: a `(time, band)` series (the result is `(time)`). |
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

### `parallax_clouds` { .api }

<!-- sig: zeit.parallax_clouds -->
```python
zeit.parallax_clouds(
    cube, potential, *, variability=None, saturated=None, bright=None,
    bands=None, scale="auto", nodata="auto", return_cdi=False, n_jobs=-1,
)
```

Sentinel-2 clouds from the potential cloud pixels by the parallax test: FORCE's `cloud_parallax` (`cloud-ll.c`, FORCE 3.11, David Frantz, GPL-3.0-or-later; the method is Frantz, Haß, Uhl, Stoffels and Hill 2018, *Remote Sensing of Environment* 215:471-481). The detectors of the 10 m bands look at the ground one after the other, so a high object is seen at slightly different places in B07, B8A and B08, while a surface object (snow, a roof, bare soil) is at the same place in all of them. The ratio of red edge 3 to NIR (B07 / B8A) varies little at a cloud and the ratio of the broad NIR, smoothed by a 5 x 5 Gaussian kernel of sigma 0.5, to NIR (B08 / B8A) a lot; the Cloud Displacement Index is `(var_x - var_y) / (var_x + var_y)` of the two over 7 x 7 blocks of the 20 m grid (each block is 2 x 2 pixels of the 10 m one) and is close to -1 at a cloud. Blocks with a CDI of -0.5 or less, or below 0 where the minimum of the variability probability over the 15 x 15 blocks around is above 0.8, are cloud seeds if every block of their 3 x 3 neighbourhood qualifies (an erosion by one block); the clouds grow into the 8-connected blocks with a CDI in [-1, -0.25); each block goes back to four 10 m pixels; saturated pixels are added and every cloud pixel that is not bright is removed.

This is Frantz's own step. The rest of FORCE's cloud detection is derived from Zhu's Fmask and is not ported (Fmask's current repository, github.com/GERSL/Fmask, is MIT-licensed as of 2026-11-07, and none of its parts is used here): what `cloud_parallax` takes from it are inputs of this function, the potential cloud pixels (`potential`), the variability probability (`variability`), the bright pixels (`bright`) and the saturated ones (`saturated`). FORCE's clear-sky and clear-sky-land pixels and their counts, and the cloud cover, are passed to `cloud_parallax` but never read, and are not inputs.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `DataArray` | required | Reflectance `(time, band, y, x)` or `(band, y, x)` on the 10 m grid (20 m bands as four equal pixels, as for `sentinel2_resmerge`) with `rededge3` (B07), `nir` (B08) and `nir08` (B8A), or `B07 B08 B8A`; and `blue green red` unless `bright` is given, and `swir16` (B11) too for the computed variability. In memory. |
| `potential` | boolean array | required | Broadcastable to `(time, y, x)`: the potential cloud pixels (Fmask's, or from any cloud product). |
| `variability` | float array | `None` | Broadcastable to `(time, y, x)`: Fmask's variability probability in [0, 1] (FORCE keeps it times 10000 as an int16, and so does Zeit: it is rounded to that). By default `1 - max(abs(NDVI), abs(NDSI), whiteness)` of Zhu and Woodcock (2012, *Remote Sensing of Environment* 118:83-94), clipped to [0, 1]: `NDVI = (B8A - red) / (B8A + red)`, `NDSI = (green - SWIR1) / (green + SWIR1)`, the whiteness the sum over blue, green and red of the distance to their mean relative to the mean; a zero denominator gives an index of 0. Where `saturated`, a positive NDVI and a negative NDSI are set to 0, the paper's adjustment for pixels whose visible bands saturate (the paper gave it for Landsat 7; Sentinel-2's saturation is rare and flagged in its quality bands, which is where `saturated` comes from). |
| `saturated` | boolean array | `None` | Broadcastable to `(time, y, x)`: saturated pixels, which are cloud if bright. By default none. |
| `bright` | boolean array | `None` | Broadcastable to `(time, y, x)`. By default the mean of blue, green and red is above 0.15 (FORCE's 1500). |
| `bands` | `dict` | `None` | The cube's band of each role (`rededge3 nir nir08 blue green red swir16`), when its names are not the usual ones. |
| `scale` | `"auto"` or `float` | `"auto"` | What the reflectance is multiplied by in the data (10000 for integers or values above 2, else 1). The values are taken to FORCE's units (times 10000, as int16) for the computation. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Missing values besides NaN: the raster's NoData (0 for an integer cube without one), a number or `None`. A pixel with a missing value in B07, B08 or B8A is FORCE's "off": never cloud. |
| `return_cdi` | `bool` | `False` | Also return the CDI. |
| `n_jobs` | `int` | `-1` | Threads (the processors zeit may use by default). |

</div>

**Returns** a boolean `(time, y, x)` (or `(y, x)`) `DataArray`, True for cloud, with the cube's coordinates; with `return_cdi=True` a tuple of it and the CDI (below).

```python
pcp = potential_cloud_pixels(s2)        # Fmask's, or the cloud flag of a product
cloud = zeit.parallax_clouds(s2, pcp, saturated=saturated)
cloud = zeit.buffer_mask(cloud, 300)    # FORCE's CLOUD_BUFFER
```

**Where it differs from FORCE.** There is no `force=` switch: the values are FORCE's (`short` units, `float` and `double` where it has them, its order of summation) and its threads do not race, so there is nothing to choose. The one change is for odd sizes. FORCE steps through the image in twos and indexes the 20 m grid by `i / 2`, which writes outside the arrays when a side is odd (Sentinel-2 tiles are even); here the last row or column of an odd image is not part of any block and is never cloud (unless saturated and bright), while the smoothing still reads it. As in FORCE, a block whose first pixel is off is skipped even if the other three have data.

**Agreement with FORCE.** The cloud mask and the CDI (bit for bit) equal those of FORCE's `cloud_parallax` in every pixel and block on synthetic scenes with parallax-displaced clouds, built-up patches, off pixels, saturated and non-bright pixels (a C harness of FORCE's own functions; [Against the Original Implementations](../benchmarks/vs-originals.md#the-parallax-cloud-test-and-the-cloud-buffer-vs-force)).

### `cloud_displacement_index` { .api }

<!-- sig: zeit.cloud_displacement_index -->
```python
zeit.cloud_displacement_index(
    cube, *, potential=None, bands=None, scale="auto", nodata="auto", n_jobs=-1,
)
```

The Cloud Displacement Index of `parallax_clouds` on its own, `float32` `(time, y // 2, x // 2)` (or `(y // 2, x // 2)`): on the 20 m half-resolution grid where it is computed (FORCE does not bring it back to 10 m, the clouds are), the coordinates those of the centres of the 2 x 2 blocks. A block has a CDI only if it has a potential cloud pixel and at least three blocks of the 7 x 7 around it have one, since the ratios of the others are left out of the variances; else it is NaN (FORCE's -9999). Near -1 at a cloud, around 0 or above for a bright surface. `potential` (broadcastable to `(time, y, x)`) is every pixel by default, which gives a CDI everywhere; the other parameters are those of `parallax_clouds` (the bands are `rededge3`, `nir` and `nir08`).

### `buffer_mask` { .api }

<!-- sig: zeit.buffer_mask -->
```python
zeit.buffer_mask(
    mask, distance, *, units="m", resolution=None, force=False, n_jobs=-1,
)
```

Buffers the True pixels of a boolean mask: FORCE's cloud buffer (`buffer_` of `imagefuns-cl.c`, which its Level 2 processing applies to the cloud, cirrus and shadow masks and the snow layer, `CLOUD_BUFFER = 300` m by default). Every True pixel grows into the disc of the midpoint circle algorithm of integer radius `r`: the axis points at distance `r` are in, and so are `(r, 1)`, a little fatter than the true disc (89 pixels for `r = 5`). With `units="m"`, `r = int(distance / resolution)` as FORCE converts it (truncated; the resolution is the `x` spacing of a DataArray, or `resolution=`); with `units="px"` it is `distance`, a whole number. A radius of 0 changes nothing.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mask` | boolean `DataArray` or array | required | `(y, x)` or `(time, y, x)`; each date is buffered on its own. |
| `distance` | `float` | required | The buffer, in `units`. |
| `units` | `"m"` or `"px"` | `"m"` | |
| `resolution` | `float` | `None` | The pixel size in metres, for `units="m"` when the mask is not a DataArray with an `x` coordinate. |
| `force` | `bool` | `False` | FORCE's behaviour at the borders (below). |
| `n_jobs` | `int` | `-1` | Threads. |

</div>

**Returns** a boolean mask as `mask`.

**Where it differs from FORCE.** FORCE's loop only buffers the pixels at least `r` from the borders and then replaces the mask by what was drawn, so a True pixel within `r` of an edge is lost and a cloud at the edge gets no buffer. `force=True` does that (and equals FORCE in every pixel); the default keeps every True pixel and draws its disc clipped at the edge, the same everywhere else. Zeit does not draw from every True pixel as FORCE does: each output row looks up the nearest source in the rows of the disc's runs, so the time hardly depends on how much is True (FORCE's takes seconds with a large cloud and a radius of 30).

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
