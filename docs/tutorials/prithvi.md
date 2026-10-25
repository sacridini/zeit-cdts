# Prithvi

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">What is each pixel of a window, from a model pretrained on HLS around the world?</span></div>
<div><span class="k">Input</span><span class="v">Windows of the six HLS bands over a few dates, in their own units</span></div>
<div><span class="k">Output</span><span class="v">A class per pixel</span></div>
<div><span class="k">Reference</span><span class="v">Prithvi-EO-2.0, Szwarcman et al. (2024); same encoder outputs as the authors' code with their weights</span></div>
</div>

Prithvi-EO-2.0 is a vision transformer that NASA and IBM pretrained, as a masked autoencoder, on 4.2 million time series of HLS (Harmonized Landsat Sentinel-2) images from all over the world: it learned to rebuild windows of four dates from a quarter of them. It was introduced in:

> Szwarcman, D., Roy, S., Fraccaro, P., et al. (2024). *Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications*. arXiv:2412.02732. [https://arxiv.org/abs/2412.02732](https://arxiv.org/abs/2412.02732)

`zeit.ai.Prithvi` is ported from the authors' `prithvi_mae.py` (Apache-2.0) and loads their weights from Hugging Face, without `transformers`, `timm` or remote code: on the same windows its encoder gives the same tokens as theirs, to float rounding (8e-7 of the largest value at most, for all six models, with and without dates and place, 1 to 4 dates, other window sizes; see [Benchmarks](../benchmarks/index.md)). On top of the encoder, zeit adds a decoder from the tokens to a class per pixel.

## How It Works

1. **Tokens.** Each date of a window is cut into squares of 16 × 16 pixels (14 × 14 for the 600M models); each square of the six bands becomes a token, with the sines and cosines of its position in the window and in time. The TL models also add the date of each image (year and day of year) and the latitude and longitude of the place.
2. **The encoder**, a vision transformer of 12 to 32 layers, lets every token look at all the others, across space and dates.
3. **The decoder** (zeit's) puts the tokens of all dates side by side, as TerraTorch does, doubles their size back to the pixels, and joins the bands of the pixels themselves before the classifier: a token is 16 × 16 pixels (480 m at 30 m), and from the tokens alone an edge inside one cannot be placed to the pixel.

The bands are read in the cube's units (reflectance from 0 to 1 or times 10,000) and standardized with the means and deviations of Prithvi's pretraining; missing values are set at the mean.

## The models

| `variant` | Encoder | Weights | Dates and place |
| :--- | :--- | :--- | :--- |
| `"tiny-TL"` | 5 M parameters, 12 layers of 192 | 129 MB | yes |
| `"100M-TL"` (default) | 86 M, 12 layers of 768 | 455 MB | yes |
| `"300M"`, `"300M-TL"` | 304 M, 24 layers of 1024 | 1.3 GB | `-TL` only |
| `"600M"`, `"600M-TL"` | 632 M, 32 layers of 1280, squares of 14 | 2.6 GB | `-TL` only |

The weights are downloaded once from Hugging Face (`ibm-nasa-geospatial/Prithvi-EO-2.0-*`, at a fixed revision) to `~/.cache/zeit/weights`, and checked against their SHA-256. Without network access (the compute nodes of a cluster), run `zeit fetch weights prithvi-300m-tl` (or the variant used) beforehand where there is access, with the cache in a folder both see, or download the `.pt` file elsewhere and set `ZEIT_WEIGHTS` to the folder holding it. The larger models want a GPU: on a CPU of 14 threads, the 100M model takes about a quarter of a second per window of 4 × 224 × 224 (on a GPU, 30 ms).

## When to Use It

- **Few labelled windows** of a task Prithvi's pretraining knows: floods, burn scars, crops, land cover, with HLS-like data (Landsat and Sentinel-2 at 30 m).
- **Spatial patterns** matter: a field, a burn scar, a flooded area are shapes, and Prithvi sees whole windows. For the series of single pixels, [Presto](presto.md) (pretrained too) or [LightTAE](ltae.md) are lighter.

For many labelled windows, [UTAE](utae.md), trained from scratch, is smaller and often as good.

## From a cube to a map

```python
import zeit
from zeit import ai

cube = zeit.load_raster("hls_4dates.tif")                                 # (4, 6, y, x), with dates
samples = ai.samples(cube, "fields.gpkg", label="class", patch=224)
model = ai.train(ai.Prithvi, samples, epochs=20, lr=1e-4)                 # variant="100M-TL"
classes = ai.predict(model, cube)
```

The windows are a multiple of 16 pixels (14 for the 600M models); 224 is what Prithvi was pretrained on, and other sizes work (the positions are interpolated). `zeit.ai.train` builds the model from the samples: the six bands found by name (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`; `B02`... and `nir08`/`B8A`, the narrow NIR HLS uses, before the broad one), the dates, the units, and the latitude and longitude of the samples' centre. Prithvi takes the values as they are (`model.zeit_meta_["raw"]`), not the 2-98% normalization of the other models. See [Deep Learning](ai.md#from-a-cube-to-a-map).

To train the decoder only and keep the encoder as pretrained (faster, and safer with very few windows):

```python
model = ai.train(ai.Prithvi, samples, epochs=20, freeze_encoder=True)
```

## Instantiating the Model

```python
from zeit.ai import Prithvi

model = Prithvi(
    bands=["blue", "green", "red", "nir", "swir1", "swir2"],
    n_dates=4,
    num_classes=3,
    variant="300M-TL",
    dates=["2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01"],
    latlon=(-10.5, -55.2),
    reflectance_scale=1e4,          # reflectance from 0 to 1 -> times 10,000
)
logits = model(x)                   # x: (batch, 6, 4, 224, 224), NaN where missing -> (batch, 3, 224, 224)
tokens = model.encode(x)            # the encoder's last layer: (batch, 1 + 4 * 14 * 14, 1024)
```

## Before zeit 1.7

`zeit.ai.GeoFoundationViT` was a wrapper of the first Prithvi (Prithvi-100M) through `transformers` and the remote code of its repository, which no longer runs with current `transformers`; it then fell back to an untrained convolution without saying so. The name now builds a `Prithvi` (with a deprecation warning), and `model_id=` takes a Prithvi-EO-2.0 repository.

## References

- Szwarcman, D., et al. (2024). Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications. arXiv:2412.02732.
- Jakubik, J., et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv:2310.18660.
