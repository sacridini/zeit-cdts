# TempCNN

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Classify each pixel's time series into land-cover classes, fast.</span></div>
<div><span class="k">Input</span><span class="v"><code>(batch, bands, time)</code> tensors with a fixed length</span></div>
<div><span class="k">Output</span><span class="v">Class logits per pixel</span></div>
<div><span class="k">Reference</span><span class="v">Pelletier et al. (2019); written from the paper and the authors' code</span></div>
</div>

<figure markdown>
  ![TempCNN training on four synthetic land-cover classes: samples, accuracy per epoch and confusion matrix](../assets/figures/tempcnn_training.png)
  <figcaption><strong>A complete training run.</strong> 1,000 noisy NDVI series (23 dates, 12% cloud-contaminated observations) of four classes, split 70/30. After 30 epochs the held-out accuracy is 96%. The only confusion is between single and double cropping, whose second season is weak in many samples. It uses the same model and a training loop like the one below; the full script is in <code>docs/scripts/make_figures.py</code>.</figcaption>
</figure>

TempCNN is a 1D convolutional neural network designed for classifying **per-pixel satellite image time series**. It's a simpler, faster-to-train alternative to attention-based models like [LightTAE](ltae.md), and a strong baseline for most pixel time-series classification tasks. It was introduced in:

> Pelletier, C., Webb, G. I., & Petitjean, F. (2019). *Temporal convolutional neural network for the classification of satellite image time series*. **Remote Sensing**, 11(5), 523. [https://doi.org/10.3390/rs11050523](https://doi.org/10.3390/rs11050523)

`zeit.ai.TempCNN` is written from the paper and from the authors' code ([charlotte-pel/temporalCNN](https://github.com/charlotte-pel/temporalCNN)), with their defaults: kernel 5, 64 filters, dropout 0.5 and a dense layer of 256 units. It is no longer a port of R `sits`, and weights trained in `sits` do not load into it.

## How It Works

TempCNN treats the time axis of a pixel's spectral history like the spatial axis of a 1D signal, and applies a stack of 1D convolutions along it:

1. **Three convolutional blocks** (`Conv1d -> BatchNorm1d -> ReLU -> Dropout`, 64 filters and kernel size 5 by default), each convolving over the time axis while keeping every spectral band as a separate input channel.
2. **Flatten**: the full `(hidden_dim, n_times)` feature map is flattened into a single vector — **not** global-average-pooled. This is a deliberate architectural choice as in the authors' implementation: it means the dense layer's input size is tied to `n_times`, so **a given `TempCNN` instance is fixed to one sequence length** for its lifetime (unlike LightTAE, whose attention mechanism can be more flexible about padding, though `day_offsets` is still fixed per-instance too).
3. **Dense block** (`Linear -> BatchNorm1d -> ReLU -> Dropout`, default `256` nodes) followed by a **final linear classifier** producing `num_classes` logits (softmax applied externally, e.g. via `torch.nn.functional.cross_entropy` or manually at inference time).

## When to Use It

| | TempCNN | LightTAE |
|---|---|---|
| Best for | Fast baselines, shorter/noisier series, limited training data | Longer, well-sampled series where attention over specific timesteps helps |
| Compute cost | Lower (no attention, no positional encoding) | Moderate |
| Mean accuracy on `samples_modis_ndvi` (see below) | 0.861 | 0.915 |

See the [LTAE & LightTAE tutorial](ltae.md) for the attention-based alternative, and the [UTAE tutorial](utae.md) if you need whole-patch spatial segmentation rather than per-pixel classification.

## From a cube to a map

`TempCNN` classifies each pixel from its series of bands, `(time, band)`. From a cube and labelled points or polygons:

```python
import zeit
from zeit import ai

cube = zeit.load_raster("s2_ndvi_evi_2022.tif")            # (time, band, y, x)
samples = ai.samples(cube, "samples.gpkg", label="class")   # one sample per pixel
model = ai.train(ai.TempCNN, samples, epochs=50)
classes = ai.predict(model, cube)                           # label (y, x), georeferenced
```

The flatten before the dense layer ties a model to the number of dates it was built for (as in the authors' network): the cube to classify needs as many, e.g. the same composites of another year. See [Deep Learning](ai.md#from-a-cube-to-a-map) for the samples, the validation blocks and the normalization.

## Instantiating the Model

`zeit.ai.train` builds the model for you from the samples (bands, dates, classes); the arguments below pass through it, e.g. `ai.train(ai.TempCNN, samples, dropout_rate=0.3)`. To build it yourself:

```python
from zeit.ai import TempCNN

model = TempCNN(
    in_channels=n_bands,
    n_times=n_times,               # fixed sequence length this instance is built for
    num_classes=10,
    filters=64,                     # filters of each convolution
    kernel_size=5,
    n_conv=3,                       # convolution blocks
    dense_units=256,
    dropout_rate=0.5,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## A loop of your own

`zeit.ai.train` covers the usual case. For anything else (another optimizer, a scheduler, augmentation), the `SampleSet` is a PyTorch `Dataset` whose items hold the normalized `x`, the label `y` and the date `positions`; see [A loop of your own](ai.md#a-loop-of-your-own). A model trained this way can still be classified with `zeit.ai.predict` after `zeit.ai.train(model, samples, epochs=0)` records the samples' metadata in it.

## Accuracy against `sits`

Measured on `samples_modis_ndvi` of `sits` (12 dates, 4 classes), with the same 80/20 split on both sides and 5 seeds, the mean accuracy was:

| | `sits`, with its defaults | Zeit, the authors' defaults | Zeit, the pipeline's defaults (150 epochs) |
|---|---|---|---|
| TempCNN | 0.879 ± 0.027 | 0.839 ± 0.031 | 0.861 ± 0.024 |
| LightTAE | 0.902 ± 0.016 | 0.892 ± 0.020 | 0.915 ± 0.011 |

The authors' training settings for TempCNN (20 epochs, patience 0, 5% of the samples for validation) stop early on a set as small as this one.

---

## References

- Pelletier, C., Webb, G. I., & Petitjean, F. (2019). Temporal convolutional neural network for the classification of satellite image time series. **Remote Sensing**, 11(5), 523. [https://doi.org/10.3390/rs11050523](https://doi.org/10.3390/rs11050523)
- charlotte-pel/temporalCNN, the authors' code: [https://github.com/charlotte-pel/temporalCNN](https://github.com/charlotte-pel/temporalCNN)
- e-sensing/sits, used for the accuracy comparison above: [https://github.com/e-sensing/sits](https://github.com/e-sensing/sits)
