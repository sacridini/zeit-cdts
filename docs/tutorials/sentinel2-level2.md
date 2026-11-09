# Sentinel-2 Level 1C to Surface Reflectance

<p class="lead">A Sentinel-2 Level 1C scene is reflectance at the top of the atmosphere. <code>zeit.level2</code> takes a product, a <code>.SAFE</code> directory or just its name, to surface reflectance: the water vapour and the gases, the aerosol estimated from the scene's dark targets, the adjacency effect, the topographic and BRDF corrections, and a cloud, shadow, snow and water mask from Zeit's own Fmask. It is what FORCE's <code>force-l2ps</code> does for one image, in the Zeit core.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">What was the surface reflectance of this scene, and which pixels can be trusted?</span></div>
<div><span class="k">Input</span><span class="v">A Level 1C <code>.SAFE</code> directory or the name of a product, and a DEM in metres (any grid and CRS) or <code>dem="glo30"</code></span></div>
<div><span class="k">Output</span><span class="v">A <code>Dataset</code> on the scene's 10 m grid: <code>boa</code>, <code>qai</code>, <code>dst</code>, <code>vzn</code>, <code>hot</code>, <code>aod</code>, <code>wvp</code></span></div>
<div><span class="k">Reference</span><span class="v">FORCE 3.11.00 (David Frantz, GPL-3.0-or-later; Frantz 2019, <em>Remote Sensing</em> 11:1124); Fmask 4.7 (Qiu, Zhu and Woodcock 2019, <em>Remote Sensing of Environment</em> 231:111205)</span></div>
</div>

!!! warning "What is, and what is not, equal to FORCE"
    With FORCE's cloud flags, sun, geography and BRDF factor as inputs, every product of `level2` equals that of `force-l2ps` in 100 % of the pixels. The public function does not take those from FORCE: it computes the cloud mask with Zeit's Fmask, the sun with Zeit's own implementation and the BRDF factor with [`nbar`](../api/preprocessing.md#nbar)'s c-factor. With them it differs a little from FORCE, as the [validation](#validation) shows. It is for Sentinel-2 only, keeps the products on the scene's 10 m grid, and does not merge the 20 m bands or coregister.

## What you need

- **The `.SAFE` directory** of a Level 1C product: `MTD_MSIL1C.xml` and one `GRANULE/<name>/` with its `IMG_DATA`. The 13 band files are read by GDAL (the product's JPEG 2000 files, or GeoTIFFs of the same names).
- **A DEM in metres**, on any grid and CRS: Zeit puts it on the scene's 10 m grid (bilinear by default, `dem_resampling`). `topography=True`, the default, needs it, and without it the call is a `ValueError` that says to pass the DEM or use `topography=False`. The atmosphere uses the elevation and the Fmask uses the DEM when it is given.
- Optionally, a **Global Surface Water Occurrence** raster (percent) to help the Fmask find water (`gswo`).

## Find and download a scene

The scene and the DEM can come from the network, with no account or credential:

```python
scenes = zeit.find_l1c(tile="23LKC", start="2023-07-01", end="2023-07-31", max_cloud=30)
safe = zeit.download_l1c(scenes.iloc[0])    # the path of the .SAFE; a DataFrame downloads them all
dem = zeit.download_dem(safe)               # one GeoTIFF of Copernicus GLO-30 over the scene
l2 = zeit.level2(safe, dem=dem)
```

Or in one call, with the name of the product and `dem="glo30"`:

```python
l2 = zeit.level2(scenes["product"][0], dem="glo30")
```

[`find_l1c`](../api/preprocessing.md#find_l1c) searches the `sentinel-2-l1c` collection of Earth Search. [`download_l1c`](../api/preprocessing.md#download_l1c) takes the files from Google Cloud's public bucket `gcp-public-data-sentinel-2`, because that one keeps the original layout of the product: only what `level2` reads (the two XML files and the 13 band files, 672 MB in the 2023 scene of the validation), each checked by size and MD5, and an interrupted download resumes. They go to the cache (`ZEIT_CACHE_DIR/l1c`) unless `dest=` says otherwise. A product that Earth Search lists and the bucket does not have (some reprocessed ones) is a `ValueError` that names it.

[`download_dem`](../api/preprocessing.md#download_dem) mosaics the GLO-30 tiles over the scene in their own geometry. Where there is no tile (the sea) the value is the no-data of `level2`, an area with no tile at all is a `ValueError` that suggests `topography=False`, and a scene across the antimeridian is not handled.

## Run it

```python
import zeit

l2 = zeit.level2(
    "S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149.SAFE",
    dem="cop_glo30_23LKC.tif",
)
print(l2)
```

The result is an `xarray.Dataset` on the grid of the scene (10 m, the CRS and transform of the bands). `attrs` tell how the correction went: `datetime` (UTC), `sensor`, `baseline`, `aod_green` (the optical depth of the scene in B03), `aod_map` (`False` when the fallback spectrum was used), `water_vapor_cm`, `n_targets` (the valid water, shadow and vegetation dark targets), the summary of the Fmask as `fmask_*` keys, `core_seconds` and `provenance`.

Without a DEM, switch the topographic correction off:

```python
l2 = zeit.level2(safe, brdf=False, topography=False)
```

## The products

| Variable | Dimensions | What it is |
| :--- | :--- | :--- |
| `boa` | `(band, y, x)`, `int16` | Surface reflectance times 10000, `_FillValue` -9999. `band` is `BLUE`, `GREEN`, `RED`, `REDEDGE1`, `REDEDGE2`, `REDEDGE3`, `BROADNIR`, `NIR`, `SWIR1`, `SWIR2` (`BROADNIR` is B08, `NIR` is B8A) |
| `qai` | `(y, x)`, `int16` | FORCE's quality bits: no data, cloud, shadow, snow, water, and what the correction flagged (below) |
| `dst` | `(y, x)`, `int16` | Distance to the nearest cloud, shadow or snow, in metres on the 10 m grid (0 on the targets). 32767 in the whole image when the scene has no cloud target |
| `vzn` | `(y, x)`, `int16` | View zenith angle, in degrees times 100 |
| `hot` | `(y, x)`, `int16` | Haze optimized transformation, in FORCE's units |
| `aod` | `(y, x)`, `int16` | Aerosol optical depth of the scene's green band, times 1000 |
| `wvp` | `(y, x)`, `int16` | Water vapour, in cm times 1000 |

Reflectance in 0-1 comes from `boa`, with the fill value left out:

```python
red = l2.boa.sel(band="RED")
red = red.where(red != -9999) / 10000
```

## Reading the QAI flags

`qai` packs several flags in each 16-bit value. The ones to know, by bit position (bit 0 is the least significant):

| Bit(s) | Flag | Values |
| :--- | :--- | :--- |
| 0 | No data (outside the scene) | 1 = off |
| 1-2 | Cloud state | 0 none, 1 less confident (the buffer), 2 confident and opaque, 3 cirrus |
| 3 | Cloud shadow | 1 = shadow |
| 4 | Snow | 1 = snow |
| 5 | Water | 1 = water |
| 6-7 | Aerosol | 0 good, 1 interpolated, 2 high AOD, 3 fill value of the scene |
| 8 | Sub-zero reflectance | the surface reflectance was below 0 in a band |
| 9 | Saturation | the surface reflectance was above 1 in a band |
| 10 | Low sun | the sun's zenith is above 75 degrees |
| 11-12 | Illumination | 0 good, 1 moderate, 2 poor, 3 deep shadow |
| 13 | Slope | the slope is above 2 degrees |
| 14 | Water vapour fill | the water vapour of the pixel is the scene's average |

The cloud buffer of the QAI is 300 m, as in FORCE, and Zeit's Fmask never marks cirrus. A clear-sky mask, then, is a few bit operations:

```python
q = l2.qai
off = (q & 1) != 0
cloud = ((q >> 1) & 3) != 0
shadow = ((q >> 3) & 1) != 0
snow = ((q >> 4) & 1) != 0
clear = ~(off | cloud | shadow | snow)

red = l2.boa.sel(band="RED").where(clear) / 10000
nir = l2.boa.sel(band="BROADNIR").where(clear) / 10000
ndvi = (nir - red) / (nir + red)
```

## Options

| Option | Default | What it does |
| :--- | :--- | :--- |
| `brdf` | `True` | Normalises the BRDF to nadir view and the sun at the zenith of the scene (FORCE's `DO_BRDF`). The factor is [`nbar`](../api/preprocessing.md#nbar)'s c-factor (Roy et al. 2016, 2017), not FORCE's `brdf_factor`; it enters the AOD retrieval even with `brdf=False` |
| `topography` | `True` | The topographic correction of the illumination (`DO_TOPO`). Needs `dem` |
| `aod` | `True` | Estimates the aerosol optical depth from the scene's dark targets; `False` uses the fallback spectrum |
| `adjacency` | `True` | The environmental weighting of the adjacency effect |
| `multiple_scattering` | `True` | `False` is single scattering |
| `erase_cloud` | `False` | Sets the cloud pixels to no data |
| `flags` | `None` | The cloud, shadow, snow and water bits of the QAI per pixel, `int16` `(nrows, ncols)` on the 10 m grid (only bits 1 to 5 are read). With them the Fmask does not run |
| `fmask` | `True` | Runs Zeit's Fmask when `flags` is `None`. `False` runs no cloud detection: there is no cloud flag, and `dst` is 32767 in the whole image |
| `gswo` | `None` | Global Surface Water Occurrence (percent) for the Fmask, put on the 20 m grid by nearest neighbour |
| `fmask_compat` | `False` | `True` reproduces three oddities of the original Fmask 4.7 code; the default corrects them |
| `dem_resampling` | `"bilinear"` | The resampling of the DEM onto the grid of the scene (FORCE's default) |
| `n_jobs` | `8` | Threads of the correction, the Fmask and the JPEG 2000 decoder |

### Your own cloud mask

Zeit's Fmask runs at 20 m. If you have a mask you trust more, give it as `flags`: the Fmask does not run, and the QAI carries your bits.

```python
import numpy as np

flags = np.zeros((10980, 10980), dtype=np.int16)    # the scene's 10 m grid: (rows, columns)
flags[my_cloud] |= 2 << 1                           # cloud state 2: confident and opaque
flags[my_shadow] |= 1 << 3                          # shadow
l2 = zeit.level2(safe, dem="dem.tif", flags=flags)
```

### Water and the Fmask

```python
l2 = zeit.level2(safe, dem="dem.tif", gswo="gswo_23LKC.tif")
fmask_summary = {k: v for k, v in l2.attrs.items() if k.startswith("fmask_")}
```

The `fmask_*` keys of `attrs` are the Fmask's summary, as plain numbers, booleans and strings, so the `Dataset` can be written to NetCDF.

## Save and look

```python
l2.zeit.save("l2_20230728")          # a folder: boa.tif, qai.tif, dst.tif, ...
l2.zeit.save("l2_20230728.tif")      # or one multi-band file
l2.zeit.plot()                       # a selector of the variables
```

See [`.zeit.save` and `.zeit.plot`](../api/xarray.md) for the options.

## Validation

`level2` was compared with `force-l2ps` of FORCE 3.11.00 (8 threads) on tile 23LKC (Brazil, UTM 23S, 10980 x 10980 pixels at 10 m). The scenes and the FORCE outputs are not in the repository. The tables are in [Against the Original Implementations](../benchmarks/vs-originals.md#sentinel-2-l1c-to-level-2-zeitlevel2).

**With FORCE's inputs.** Given FORCE's cloud flags, sun, geography and BRDF factor, `boa` (the 10 bands), `qai`, `aod`, `wvp`, `hot`, `dst` and `vzn` are equal to FORCE's in 100 % of the pixels in three runs: 2023-07-28 without BRDF, 2023-07-28 with BRDF (baseline 05.09) and 2021-07-28 without BRDF (baseline 03.01). In 2023 that is 120,560,400 pixels per band.

**With Zeit's own sun.** The sun of `level2` is a clean-room implementation, up to 0.0105 degrees from FORCE's. With it and FORCE's cloud flags (2023, without BRDF), the pixels equal to FORCE's go from 75.04 % (blue) to 92.45 % (SWIR2), and the mean difference from 0.56 to 0.08 in reflectance times 10000.

**With Zeit's Fmask.** Its cloud-or-shadow mask marks 19.93 % of the pixels against 19.70 % in FORCE's, agreeing in 98.12 % of them (IoU 0.910). Only the shadow diverges, by construction: FORCE marks shadow also inside the confident cloud and uses another version of the Fmask. Against the Sen2Cor scene classification of ESA (2023), the Fmask of Zeit and FORCE's QAI agree with it in 91.28 % of the pixels (IoU 0.58 in cloud or shadow), and mark about 20 % against its 13 %, with the 300 m ring included.

**Against ESA's L2A.** The L2A of the same tile and date is another chain, so no equality is expected, and FORCE against the L2A is the control. On the pixels valid in the three products, clear in both masks and classified by the SCL as vegetation or bare soil (23,405,935 pixels in 2023), the bias in reflectance times 10000 is:

| Band (2023-07-28, baseline 05.09) | Zeit minus L2A | FORCE minus L2A |
| :--- | ---: | ---: |
| BLUE | -24.58 | -25.13 |
| RED | 6.16 | 6.05 |
| NIR | 34.94 | 34.92 |
| SWIR1 | -174.71 | -174.70 |
| SWIR2 | -242.22 | -242.22 |

Zeit and FORCE are at nearly the same distance from the L2A in every band. The L2A is 165 to 242 above both in the two SWIR bands; that bias was not checked against the literature, so it is not known which chain is closer to the truth.

**Speed.** The whole chain with Zeit's Fmask took 141 to 147 s (113 to 119 s in the core) against 197 s for `force-l2ps`, both with 8 threads, on a machine that was not dedicated: these times are not a clean benchmark.

**Not compared.** A scene without any cloud target gives `dst` 32767 in every pixel; that case was not compared with FORCE.

## What it does not do

- It is for Sentinel-2 only.
- It keeps the products on the 10 m grid of the scene. FORCE's reprojection to its datacube is not done.
- It does not merge the 20 m bands into a 10 m grid (see [`sentinel2_resmerge`](../api/preprocessing.md#sentinel2_resmerge)) and does not coregister (see [`coregister`](../api/preprocessing.md#coregister)). Both are steps of their own: [Preparing images](preparing-images.md).

## Credits

`level2` is the atmospheric correction of FORCE 3.11.00 (David Frantz, GPL-3.0-or-later; Frantz 2019, *Remote Sensing* 11:1124) in the Zeit core. The cloud, shadow, snow and water mask is Zeit's port of Fmask 4.7 (Qiu, Zhu and Woodcock 2019, *Remote Sensing of Environment* 231:111205). The distance to cloud (`dst`) is rewritten in a clean room from FORCE's public documentation and the outputs of its DST product, and the sun is Zeit's own clean-room implementation. The full signature and every parameter are in the [API reference](../api/preprocessing.md#level2).
