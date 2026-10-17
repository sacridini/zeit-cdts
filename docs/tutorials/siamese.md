# Siamese Change Detector

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Where did this area change between two dates?</span></div>
<div><span class="k">Input</span><span class="v">Two co-registered images <code>(batch, bands, H, W)</code></span></div>
<div><span class="k">Output</span><span class="v">A per-pixel change map</span></div>
<div><span class="k">Reference</span><span class="v">FC-Siam-diff, Daudt et al. (2018); same outputs as the authors' code</span></div>
</div>

The Siamese Change Detector is a **bi-temporal** architecture: given two co-registered images of the same area at two different dates, it outputs a per-pixel change probability map. It is the FC-Siam-diff network of:

> Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully convolutional siamese networks for change detection*. **2018 25th IEEE International Conference on Image Processing (ICIP)** (pp. 4063–4067). [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)

`zeit.ai.SiameseChangeDetector` is written from the paper, not copied from the authors' code (which publishes no license), and it is the same network: with the weights of the authors' `SiamUnet_diff` ([rcdaudt/fully_convolutional_change_detection](https://github.com/rcdaudt/fully_convolutional_change_detection)) it gives the same log-probabilities, to 1e-15 in double precision, for any image size. Trained the same way, the two reach the same accuracy, and zeit's predicts faster (it runs both dates through the encoder in one pass); see [Benchmarks](../benchmarks/index.md).

## How It Works

The core idea of a Siamese network is **weight sharing**: the same encoder is applied to both images, so that their features live in the same representation space and a difference between them reflects a difference in content, not in how the two images were processed. FC-Siam-diff is a U-Net built on that idea:

1. **Shared encoder** (`forward_once`): four stages of 3×3 convolutions, each followed by batch normalization, ReLU and dropout (16, 16 | 32, 32 | 64, 64, 64 | 128, 128, 128 channels), with a 2×2 max-pooling between stages. Both images, `x_t0` ("before") and `x_t1` ("after"), go through it with the same weights.
2. **Differences at every scale**: at the end of each stage, the absolute difference of the two dates' features, `|f₀ − f₁|`, is passed to the decoder (the skip connections of the U-Net). The fine scales keep the edges of a clearing or a one-pixel road; the coarse ones bring the context that tells a real change from a change of illumination.
3. **Decoder**: from the bottleneck (as in the authors' code, that of the second date), each stage upsamples by 2 (a transposed convolution), concatenates the difference of that scale, and applies the encoder's convolutions in reverse; the last convolution gives the per-pixel logits of `num_classes` classes (by default 2: "no change" and "change").

The image halves four times, so the windows are 16 pixels or more; any size works (an odd size is padded back in the decoder, as in the authors' code). Both dates go through the encoder in one pass (the authors' code runs it once per date). In prediction the result is the same; in training, batch normalization then takes the statistics of both dates together, which trains as well as the authors' code (F1 0.969 against 0.970 over six seeds) and faster.

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

The encoder halves the window four times, so `patch` is 16 or more (64 or more gives the deepest stage some context). Label both what changed and what did not around it: polygons covering the unchanged land teach the model as much as the changed ones. See [Deep Learning](ai.md#from-a-cube-to-a-map).

## Instantiating the Model

`zeit.ai.train` builds the model for you from the samples (bands, dates, classes); the arguments below pass through it, e.g. `ai.train(ai.SiameseChangeDetector, samples, num_classes=2)`. To build it yourself:

```python
from zeit.ai import SiameseChangeDetector

model = SiameseChangeDetector(
    in_channels=6,     # spectral bands per image
    num_classes=2,     # 2 for binary change/no-change; more for multi-class change type
    dropout=0.2,       # after every convolution, as in the paper
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## Loss Functions

Two natural options from `zeit.ai.losses`, depending on how you want to train:

- **`FocalLoss`** / **`TverskyLoss`**: apply directly to the classifier's `(B, num_classes, H, W)` output logits vs. the `(B, H, W)` label map — the standard approach if you're training the full pipeline (encoders + decoder + classifier) end-to-end as a segmentation problem. Both handle the severe class imbalance typical of change detection (changed pixels are usually a small minority).
- **`ContrastiveSiameseLoss`**: operates directly on the two *encoder* feature maps rather than the final classifier output (`forward_once(x)` gives the features of one image at each scale; the first, `forward_once(x)[0]`, is at the full resolution of the label map) — it pulls feature vectors together for unchanged pixels and pushes them apart (up to a margin) for changed pixels. Use this if you want to train the twin encoder as a metric-learning problem (e.g. for downstream thresholding or few-shot change detection), separately from or in addition to the classifier head.

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
