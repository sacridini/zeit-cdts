# Presto

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">What is each pixel, from few labelled samples? An embedding of your own cube, for any period</span></div>
<div><span class="k">Input</span><span class="v">A cube of up to 24 dates of Sentinel-2 and/or Sentinel-1, ERA5, SRTM bands, gaps included</span></div>
<div><span class="k">Output</span><span class="v">A class map (fine-tuned), or a 128-dimension embedding cube</span></div>
<div><span class="k">Reference</span><span class="v">Tseng et al. (2023); same encodings as the authors' code with their weights</span></div>
</div>

Presto (*Pretrained Remote Sensing Transformer*) is a small transformer pretrained, without labels, on the time series of single pixels from all over the world: Sentinel-1 radar, Sentinel-2 optical bands, ERA5 temperature and precipitation, and SRTM elevation and slope, through up to two years of monthly composites. It was introduced in:

> Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., & Kerner, H. (2023). *Lightweight, Pre-trained Transformers for Remote Sensing Timeseries*. arXiv:2304.14065. [https://arxiv.org/abs/2304.14065](https://arxiv.org/abs/2304.14065)

`zeit.ai.Presto` is ported from the authors' code ([nasaharvest/presto](https://github.com/nasaharvest/presto), MIT) and loads their pretrained weights: on the same series it gives the same encodings as their encoder (identical, or within 1.2e-6, with clouds, missing sensors, 3 to 24 dates; see [Benchmarks](../benchmarks/index.md)). It has 0.4 million parameters: on a CPU it encodes about a thousand series of 12 dates per second and core (a million pixels take minutes), on a GPU tens of thousands.

## How It Works

1. **Tokens per group of bands and date.** The bands are grouped (Sentinel-1; Sentinel-2 RGB, red edge, 10 m NIR, 20 m NIR, SWIR; ERA5; NDVI; and SRTM, which is static) and each group at each date becomes one token, plus one token for the pixel's latitude and longitude. Each token carries the embedding of its month, of its group and of its date's position in the series.
2. **What is missing is left out.** A group missing at a date (a cloud, a band the cube does not have, a sensor not there yet) is not a token at all: Presto was pretrained by hiding most of its inputs and reconstructing them, so it is used to seeing only part of them. Gaps need no filling.
3. **Two transformer layers** over the tokens, and the mean of the tokens observed: one vector of 128 numbers per pixel, the encoding. `zeit.ai.Presto` adds a linear classifier on top; `presto_embeddings` returns the encoding itself.

The bands are read in the cube's own units and normalized as Presto was trained (Sentinel-2 reflectance divided by 10,000, Sentinel-1 shifted and scaled from dB, temperature from kelvin, NDVI computed from B8 and B4).

## When to Use It

- **Few labelled samples.** From the pretrained weights, a few hundred pixels are often enough: in the test suite, 150 pixels and 30 epochs separate three kinds of land perfectly, where the same network from random weights gets two thirds right.
- **Gappy series, several sensors.** Clouds, missing dates and missing bands are handled by the model itself.
- **An embedding of your own cube.** TESSERA and AlphaEarth are yearly embeddings of fixed years; `presto_embeddings` makes one from any cube you have, for any period of up to 24 dates, and it works with everything zeit does with embeddings.

For many labelled samples and long series of a single sensor, [LightTAE](ltae.md) or [TempCNN](tempcnn.md) trained from scratch are as good and simpler.

## The cube Presto reads

The bands are recognized by name (any case); `band_map={"B8": "my_nir"}` names others.

| Presto input | Band names recognized | Units read |
| :--- | :--- | :--- |
| Sentinel-2 B2, B3, B4 | `blue`, `green`, `red`, `B02`... | reflectance, 0-1 or 0-10000 (detected) |
| B5, B6, B7 | `rededge1`, `rededge2`, `rededge3`, `B05`... | |
| B8 | `nir`, `B08` | |
| B8A | `nir08`, `B8A` | |
| B11, B12 | `swir1`, `swir2`, `swir16`, `swir22`, `B11`, `B12` | |
| Sentinel-1 VV, VH | `VV`, `VH` | dB, or linear backscatter (detected) |
| ERA5 | `temperature_2m` / `t2m`, `total_precipitation` / `tp` | kelvin, metres |
| SRTM | `elevation` / `dem`, `slope` | metres, degrees (the mean of the dates is used) |

Any subset works: a cube with the six bands of HLS (`blue` to `swir2`) is read as Sentinel-2 B2, B3, B4, B8, B11 and B12, and everything else is missing. Presto was pretrained on **monthly** composites; `zeit.build_time_series` or `regularize_time_series` make them. It takes up to 24 dates, and their months are part of the input.

## From a cube to a map

```python
import zeit
from zeit import ai

cube = zeit.load_raster("s2_monthly_2023.tif")                     # (12, band, y, x), with dates
samples = ai.samples(cube, "fields.gpkg", label="class")            # pixels
model = ai.train(ai.Presto, samples, epochs=30)                     # from the pretrained weights
classes = ai.predict(model, cube)
```

`zeit.ai.train` builds the model from the samples: the months of the dates, the units of the bands (reflectance from 0 to 1 or to 10,000; dB or linear radar backscatter) and the latitude and longitude of the samples' centre, which every pixel gets when predicting. The whole network is fine-tuned. Presto takes the values as they are, missing ones as NaN, and not the 2-98% normalization of the other models (`model.zeit_meta_["raw"]` is `True`). See [Deep Learning](ai.md#from-a-cube-to-a-map).

## Embeddings of a cube

```python
emb = ai.presto_embeddings(cube)                  # (1, 128, y, x): bands A00-A127
near = zeit.similarity(emb, "reference_field.gpkg")
clusters = zeit.som(emb)
rf = zeit.train_classifier(emb, "samples.gpkg")
```

Each pixel gets its own latitude and longitude. The result is tagged as Presto embeddings (`embedding_source="presto"`), so a model trained on them refuses TESSERA's or AlphaEarth's, and so does `check_same`. `model=` takes a fine-tuned `Presto` to embed with its encoder; its embeddings are tagged as another space (`embedding_variant="fine-tuned <hash>"`).

## Instantiating the Model

```python
from zeit.ai import Presto

model = Presto(
    bands=["blue", "green", "red", "nir", "swir1", "swir2"],
    months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    num_classes=4,
    latlon=(-10.5, -55.2),
    reflectance_scale=1e4,     # reflectance from 0 to 1 -> Presto's 0-10000
)
logits = model(x)              # x: (batch, 12, 6), NaN where missing
encodings = model.embed(x)     # (batch, 128)
```

The weights (3 MB) are downloaded once from the authors' repository to `~/.cache/zeit/weights` and checked against their SHA-256. Without network access (the compute nodes of a cluster), run `zeit fetch weights presto` beforehand where there is access, with the cache in a folder both see, or download [`default_model.pt`](https://github.com/nasaharvest/presto/raw/11e207a668a34336ced1d8e492a1bd5849b96c4a/data/default_model.pt) elsewhere and set the environment variable `ZEIT_WEIGHTS` to the folder holding it. `pretrained=False` starts from random weights.

## Limits

- **One position for the whole map when classifying.** `zeit.ai.predict` gives every pixel the latitude and longitude of the training samples' centre; for a region far from where the model was trained, build the model with `latlon=` of that region. `presto_embeddings` gives each pixel its own.
- **Dynamic World** classes, an input of Presto, are not read: they are always missing, as in the authors' examples without them.
- **Monthly composites.** Other steps work (the months of the dates are what Presto sees), but they are not what it was pretrained on.

## References

- Tseng, G., Cartuyvels, R., Zvonkov, I., Purohit, M., Rolnick, D., & Kerner, H. (2023). Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. arXiv:2304.14065.
