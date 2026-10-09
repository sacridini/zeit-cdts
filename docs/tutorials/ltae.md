# LTAE & LightTAE

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Classify pixel time series with temporal attention.</span></div>
<div><span class="k">Input</span><span class="v"><code>(batch, time, bands)</code> plus the day of each observation</span></div>
<div><span class="k">Output</span><span class="v">Class logits per pixel</span></div>
<div><span class="k">Reference</span><span class="v">Garnot & Landrieu (2020); weight-compatible with R <code>sits</code></span></div>
</div>

The Lightweight Temporal Attention Encoder (L-TAE) is a compact, fast attention mechanism designed specifically for classifying **per-pixel satellite image time series** — the kind of long, irregularly-sampled sequence of spectral observations you get from a single pixel's history in a data cube. It was introduced in:

> Garnot, V. S. F., & Landrieu, L. (2020). *Satellite image time series classification with pixel-set encoders and temporal self-attention.* CVPR 2020.

`zeit.ai` exposes two related classes:

- **`LTAE`**: the reusable temporal-fusion block itself — takes an encoded `(batch, seq_len, in_channels)` sequence and fuses it into a single `(batch, n_neurons[-1])` embedding via multi-head attention. Useful if you want to plug L-TAE fusion into your own custom architecture (e.g. as the temporal-fusion stage of a spatial-temporal model).
- **`LightTAE`**: the full, ready-to-train pixel time-series classifier — a small per-pixel MLP spatial encoder, followed by `LTAE` temporal fusion, followed by an MLP decoder to class logits. This is the model you want for standard "classify this pixel's time series into a land-cover class" tasks.

Both were **ported layer-for-layer from the R package [`sits`](https://github.com/e-sensing/sits)'s `sits_lighttae()`** (`.torch_light_temporal_attention_encoder` / `sits_lighttae()` in `sits`'s R/api_torch_psetae.R and R/sits_lighttae.R), so trained weights are directly portable between the two implementations via `state_dict()` — there is no name-translation table needed. This was checked by exporting a trained `sits_lighttae()` model's weights, loading them into `zeit.ai.LightTAE` via `load_state_dict`, and confirming the outputs match `sits`'s own predictions within float32 tolerance on the same input.

## How It Works

The "L" in L-TAE stands for *lightweight*, and the trick that makes it fast is a **learned "master query"**: unlike standard self-attention, where the query vector is computed from the input at every forward pass, L-TAE's query is a single learned parameter per attention head, shared across every input in the batch. This collapses what would normally be an `O(seq_len²)` self-attention computation down to `O(seq_len)` — each timestep only needs to attend *to* the master query, not to every other timestep.

`LightTAE`'s forward pass has three stages:

1. **Spatial encoder** (`_PixelSpatialEncoder`): a small per-timestep MLP (`Linear -> BatchNorm1d -> ReLU`, stacked) that independently encodes every `(pixel, time)` spectral observation from `n_bands` into a higher-dimensional embedding (default `(32, 64, 128)`).
2. **Temporal encoder** (`LTAE`): adds a sinusoidal positional encoding keyed on `day_offsets` (day counts from the first observation, *not* the raw calendar date — so it doesn't matter which year your series starts in), then fuses the sequence via the master-query multi-head attention described above, producing one embedding per pixel.
3. **Decoder**: a small MLP (`Linear -> BatchNorm1d -> ReLU`, stacked) mapping the fused embedding to `n_labels` class logits (softmax is applied externally, matching `sits`'s convention — use `torch.nn.functional.cross_entropy`, which expects raw logits, or apply `softmax`/`argmax` yourself at inference time).

Because the flatten/positional-encoding buffers are sized at construction time from `day_offsets`, **a given `LightTAE` (or `LTAE`) instance is tied to one fixed sequence length and temporal sampling pattern for its lifetime** — the same constraint `sits_lighttae()`'s `timeline` parameter imposes. If your pixels have varying numbers of valid observations, interpolate/gap-fill them onto a common `day_offsets` grid before feeding them in.

## When to Use It

| | LightTAE | TempCNN |
|---|---|---|
| Best for | Per-pixel time series with many, well-sampled observations where temporal ordering/attention genuinely helps | Shorter or noisier series, or when you want a simpler, faster-to-train baseline |
| Compute cost | Moderate (attention has more parameters than a plain conv stack) | Lower |
| Cross-validated against | `sits_lighttae()` (R) | `sits_tempcnn()` (R) |

See the [TempCNN tutorial](tempcnn.md) for the simpler 1D-CNN alternative, and the [UTAE tutorial](utae.md) if you need spatially-aware *segmentation* (a class per pixel over a whole image patch, not just a single pixel's own time series).

## From a cube to a map

`LightTAE` classifies each pixel from its series of bands, `(time, band)`. Its positional encoding takes the days since the first date, which `zeit.ai.samples` computes from the cube's dates (`samples.meta["positions"]`):

```python
import zeit
from zeit import ai

cube = zeit.load_raster("s2_16day_2022.tif")                # (time, band, y, x)
samples = ai.samples(cube, "samples.gpkg", label="class")
model = ai.train(ai.LightTAE, samples, epochs=50, loss="focal")
classes = ai.predict(model, cube)
```

Like `sits_lighttae()`, a model is built for one timeline (its layer norm and positional encoding are sized by it): the cube to classify needs as many dates. See [Deep Learning](ai.md#from-a-cube-to-a-map) for the samples, the validation blocks and the normalization.

## Instantiating the Model

`zeit.ai.train` builds the model for you from the samples (bands, dates, classes); the arguments below pass through it, e.g. `ai.train(ai.LightTAE, samples, n_heads=8)`. To build it yourself:

```python
from zeit.ai import LightTAE

model = LightTAE(
    n_bands=6,              # number of spectral bands per observation
    day_offsets=day_offsets,
    n_labels=10,             # number of land-cover classes
    layers_spatial_encoder=(32, 64, 128),  # spatial-encoder MLP widths
    n_heads=16,               # attention heads
    n_neurons=(256, 128),     # LTAE internal MLP widths (n_neurons[0] must equal d_model)
    dropout_rate=0.2,
    dim_input_decoder=128,    # must match n_neurons[-1]
    dim_layers_decoder=(64, 32),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Using the `LTAE` block on its own

If you want L-TAE's temporal fusion inside a custom architecture (e.g. after your own spatial feature extractor), instantiate it directly:

```python
from zeit.ai import LTAE

temporal_fusion = LTAE(
    in_channels=128,   # dimensionality of your incoming per-timestep features
    day_offsets=day_offsets,
    n_heads=16,
    n_neurons=(256, 128),
    dropout_rate=0.2,
).to(device)

# x: (batch, seq_len, in_channels) -> (batch, n_neurons[-1])
fused = temporal_fusion(x)
```

## A loop of your own

`zeit.ai.train` covers the usual case. For anything else (another optimizer, a scheduler, augmentation), the `SampleSet` is a PyTorch `Dataset` whose items hold the normalized `x`, the label `y` and the date `positions`; see [A loop of your own](ai.md#a-loop-of-your-own). A model trained this way can still be classified with `zeit.ai.predict` after `zeit.ai.train(model, samples, epochs=0)` records the samples' metadata in it.

## Validation Against `sits`

Both `LTAE` and `LightTAE` were validated end-to-end in-session against `sits_lighttae()`: a model trained in R was exported (`state_dict()`-compatible weight names, since the port is layer-for-layer), loaded into `zeit.ai.LightTAE` via `load_state_dict()`, and run on the same input data. Outputs matched `sits`'s predictions within float32 numerical tolerance, confirming a faithful architectural port rather than just a similar-looking reimplementation.

---

## References

- Garnot, V. S. F., Landrieu, L., Giordano, S., & Chehata, N. (2020). Satellite image time series classification with pixel-set encoders and temporal self-attention. In **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)** (pp. 12322–12331). [https://doi.org/10.1109/CVPR42600.2020.01234](https://doi.org/10.1109/CVPR42600.2020.01234)
- Garnot, V. S. F., & Landrieu, L. (2020). *Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series*. arXiv:2007.00586. [https://arxiv.org/abs/2007.00586](https://arxiv.org/abs/2007.00586)
- e-sensing/sits: [https://github.com/e-sensing/sits](https://github.com/e-sensing/sits)
