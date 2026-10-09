# Siamese Change Detector

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Where did this area change between two dates?</span></div>
<div><span class="k">Input</span><span class="v">Two co-registered images <code>(batch, bands, H, W)</code></span></div>
<div><span class="k">Output</span><span class="v">A per-pixel change map</span></div>
<div><span class="k">Reference</span><span class="v">Daudt et al. (2018) design; independent implementation</span></div>
</div>

The Siamese Change Detector is a **bi-temporal** architecture: given two co-registered images of the same area at two different dates, it outputs a per-pixel change probability map. It follows the general design of:

> Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully convolutional siamese networks for change detection*. **2018 25th IEEE International Conference on Image Processing (ICIP)** (pp. 4063–4067). [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)

Unlike [LightTAE](ltae.md), [TempCNN](tempcnn.md), and [UTAE](utae.md) — which were ported layer-for-layer and rigorously cross-validated against reference implementations (`sits` or the official UTAE repo) — `zeit.ai.SiameseChangeDetector` is a **compact, independent implementation** of the general Siamese/twin-encoder change-detection pattern, not a line-for-line port of a specific published codebase. Treat it as a solid, ready-to-train baseline architecture for two-date change detection rather than a bit-exact reproduction of any one paper's exact numbers.

## How It Works

The core idea of a Siamese network is **weight sharing**: the same encoder is applied independently to both input images, so that the two resulting feature maps live in a comparable representation space. `SiameseChangeDetector`'s forward pass:

1. **Twin encoding** (`forward_once`): each input image (`x_t0`, the "before" image, and `x_t1`, the "after" image) is passed *independently* through the **same** two-stage convolutional encoder (`ConvBlock` at 64 channels, max-pooled, then a second `ConvBlock` at 128 channels) — the weights are shared, so a difference in the two output feature maps reflects a genuine change in content, not a difference in how the two images were processed.
2. **Difference**: the absolute difference between the two encoded feature maps (`|feat_t0 - feat_t1|`) is computed — large values indicate the encoder detected substantially different content at that spatial location between the two dates.
3. **Decoder**: the difference map is passed through a decoding `ConvBlock`, upsampled back to the input resolution (bilinear upsampling), and a final `1x1` convolution (`classifier`) produces per-pixel class logits (by default `num_classes=2`: "no change" vs. "change").

## When to Use It

Use the Siamese Change Detector for classic **bi-temporal change detection**: you have exactly two dates (before/after an event — a wildfire, deforestation, a flood, construction) and want a change map between them. If you have a **longer time series** and want to classify or segment based on the whole trajectory rather than just two snapshots, use [LightTAE](ltae.md) (per-pixel) or [UTAE](utae.md) (whole-patch segmentation) instead.

## From a cube to a map

The data is a pair of images, `(before, after)`, each a map or a stack of bands on the same grid, and the classes say what changed (e.g. `change` and `same`; any number of classes):

```python
import zeit
from zeit import ai

before = zeit.load_raster("s2_2021_08.tif")                           # (band, y, x)
after = zeit.load_raster("s2_2023_08.tif", like="s2_2021_08.tif")    # on the same grid
samples = ai.samples((before, after), "change.gpkg", label="class", patch=64)
model = ai.train(ai.SiameseChangeDetector, samples, epochs=100)
change = ai.predict(model, (before, after), probability=True)
```

The encoder halves the window once, so `patch` is even. Label both what changed and what did not around it: polygons covering the unchanged land teach the model as much as the changed ones. See [Deep Learning](ai.md#from-a-cube-to-a-map).

## Instantiating the Model

`zeit.ai.train` builds the model for you from the samples (bands, dates, classes); the arguments below pass through it, e.g. `ai.train(ai.SiameseChangeDetector, samples, num_classes=2)`. To build it yourself:

```python
from zeit.ai import SiameseChangeDetector

model = SiameseChangeDetector(
    in_channels=6,     # spectral bands per image
    num_classes=2,     # 2 for binary change/no-change; more for multi-class change type
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## Loss Functions

Two natural options from `zeit.ai.losses`, depending on how you want to train:

- **`FocalLoss`** / **`TverskyLoss`**: apply directly to the classifier's `(B, num_classes, H, W)` output logits vs. the `(B, H, W)` label map — the standard approach if you're training the full pipeline (encoders + decoder + classifier) end-to-end as a segmentation problem. Both handle the severe class imbalance typical of change detection (changed pixels are usually a small minority).
- **`ContrastiveSiameseLoss`**: operates directly on the two *encoder* feature maps (`feat_t0`, `feat_t1` from `forward_once`) rather than the final classifier output — it pulls feature vectors together for unchanged pixels and pushes them apart (up to a margin) for changed pixels. Use this if you want to train the twin encoder as a metric-learning problem (e.g. for downstream thresholding or few-shot change detection), separately from or in addition to the classifier head.

```python
from zeit.ai.losses import FocalLoss, ContrastiveSiameseLoss

criterion = FocalLoss(alpha=0.25, gamma=2.0)
# or, to also supervise the encoder features directly:
contrastive_criterion = ContrastiveSiameseLoss(margin=2.0)
```

## A loop of your own

`zeit.ai.train` covers the usual case. For anything else (another optimizer, a scheduler, augmentation), the `SampleSet` is a PyTorch `Dataset` whose items hold the normalized `x`, the label `y` and the date `positions`; see [A loop of your own](ai.md#a-loop-of-your-own). A model trained this way can still be classified with `zeit.ai.predict` after `zeit.ai.train(model, samples, epochs=0)` records the samples' metadata in it.

## References

- Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully convolutional siamese networks for change detection. In **2018 25th IEEE International Conference on Image Processing (ICIP)** (pp. 4063–4067). [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)
