# UTAE (U-Net with Temporal Attention Encoder)

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Produce a class map for a whole image patch from its full history.</span></div>
<div><span class="k">Input</span><span class="v"><code>(batch, time, bands, H, W)</code> plus dates</span></div>
<div><span class="k">Output</span><span class="v">A class map per patch, and attention maps</span></div>
<div><span class="k">Reference</span><span class="v">Garnot & Landrieu (2021); bit-exact with the official code</span></div>
</div>

UTAE combines a U-Net with a temporal attention mechanism to perform **spatio-temporal segmentation** of satellite image time series — producing a full class map over a whole image patch (e.g. `256x256`) using its *entire* observation history, not just a per-pixel classification. It was introduced in:

> Garnot, V. S. F., & Landrieu, L. (2021). *Panoptic segmentation of satellite image time series with convolutional temporal attention networks*. ICCV 2021. [doi:10.1109/ICCV48922.2021.00483](https://doi.org/10.1109/ICCV48922.2021.00483)

`zeit.ai.UTAE` was **ported layer-for-layer from the official reference implementation** ([VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps), MIT License). This is the most rigorously validated model in `zeit.ai`: building both implementations with identical weights (loaded via `load_state_dict()`, with `state_dict()` key names matching directly, no translation table) and feeding them the same input reproduces the reference implementation's output **bit-for-bit exactly** (`max abs diff = 0.0`), including the padded-sequence (irregular temporal sampling) code path.

## How It Works

UTAE is a multi-scale U-Net where every stage is applied independently to each timestep (weights shared across time — see `_TemporallySharedBlock`), and temporal fusion happens once, at the bottleneck, via an image-aware L-TAE variant (`_LTAE2d`). The architecture has four parts:

1. **Encoder**: a standard convolutional U-Net encoder (`in_conv` + a stack of `_DownConvBlock`s), applied frame-by-frame with shared weights, producing a pyramid of per-timestep feature maps at decreasing spatial resolution.
2. **Temporal bottleneck (`_LTAE2d`)**: applies a per-pixel L-TAE (the same "learned master query" multi-head attention mechanism as [LTAE](ltae.md), but computed independently at every spatial position of the deepest feature map) to fuse the time dimension into a single feature map, **plus** per-head, per-timestep attention maps.
3. **Temporal aggregator (`_TemporalAggregator`)**: this is UTAE's key idea — the attention maps from step 2 are resampled to each decoder scale and used to weight the temporal aggregation of *every* skip connection, not just the bottleneck. This propagates the learned "when does this pixel matter" signal to every resolution of the decoder.
4. **Decoder**: a standard U-Net decoder (`_UpConvBlock` stack) that upsamples the fused bottleneck feature, concatenating each attention-weighted skip connection along the way, ending in a final `out_conv` producing per-pixel class logits over the whole patch.

UTAE supports **irregular temporal sampling**: pass a `pad_value` (default `0`) and pad shorter sequences in a batch up to a common length with that value — the model automatically builds a `pad_mask` and skips (or masks out) padded frames in both the shared-weight per-timestep convolutions and the attention mechanism, so you don't need every sample in a batch to have the exact same number of valid observations.

## When to Use It

Use UTAE when you need a **class map over a spatial patch** informed by its full time series — e.g. crop-type mapping, burned-area segmentation, or any task where spatial context (not just a single pixel's own spectral history) matters. If you only need a classification of individual pixels' own time series (no spatial context needed), [LightTAE](ltae.md) or [TempCNN](tempcnn.md) are lighter-weight and faster to train.

## From a cube to a map

`UTAE` segments windows of the cube, `(time, band, patch, patch)`, so it trains on patches; polygons give it dense labels:

```python
import zeit
from zeit import ai

cube = zeit.load_raster("s2_2022.tif")                                   # (time, band, y, x)
samples = ai.samples(cube, "fields.gpkg", label="crop", patch=32)        # windows of 32 x 32
model = ai.train(ai.UTAE, samples, epochs=100)
crops = ai.predict(model, cube, overlap=0.5)
```

The default U-Net halves the window three times, so `patch` is a multiple of 8. `predict` slides windows of the same size over the cube, overlapping by `overlap`, and averages each pixel's probabilities over them, weighted down towards the windows' edges, so the map has no seams; a lazy cube gives the same map, block by block. The dates' positions (days since the first date) are passed at every call, so a model classifies cubes with other dates too. See [Deep Learning](ai.md#from-a-cube-to-a-map).

## Instantiating the Model

`zeit.ai.train` builds the model for you from the samples (bands, dates, classes); the arguments below pass through it, e.g. `ai.train(ai.UTAE, samples, encoder_widths=[32, 32, 64, 64], decoder_widths=[16, 16, 32, 64])`. To build it yourself:

```python
from zeit.ai import UTAE

model = UTAE(
    input_dim=6,                    # number of spectral bands
    encoder_widths=(64, 64, 64, 128),
    decoder_widths=(32, 32, 64, 128),
    out_conv=[32, 10],              # final conv stack -> 10 output classes
    str_conv_k=4, str_conv_s=2, str_conv_p=1,  # strided-conv down/up-sampling geometry
    agg_mode="att_group",           # temporal aggregation mode for skip connections
    encoder_norm="group",
    n_head=16,
    d_model=256,
    d_k=4,
    pad_value=0,                    # value used for padded (missing) timesteps
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

`out_conv`'s last entry sets the number of output classes; `encoder_widths`/`decoder_widths` must have matching lengths and equal final entries (assertions enforce this at construction time).

## A loop of your own

`zeit.ai.train` covers the usual case. For anything else (another optimizer, a scheduler, augmentation), the `SampleSet` is a PyTorch `Dataset` whose items hold the normalized `x`, the label `y` and the date `positions`; see [A loop of your own](ai.md#a-loop-of-your-own). A model trained this way can still be classified with `zeit.ai.predict` after `zeit.ai.train(model, samples, epochs=0)` records the samples' metadata in it.

## Validation Against the Official Reference

`UTAE` (and its internal `LTAE2d`, `_TemporalAggregator`, etc.) is a line-for-line port of [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps). Validation methodology: the official repo was cloned locally, both implementations were instantiated with the same hyperparameters and the same random weights (copied via `load_state_dict()` — the `state_dict()` key names match with no translation needed), and run forward on identical random input. The outputs were bit-for-bit identical (`max abs diff = 0.0`), across both the regular (unpadded) code path and the padded-sequence (`pad_value`/`pad_mask`) code path used for irregular temporal sampling. This cross-check is not part of the pytest suite, since it requires the reference repo cloned locally rather than a pip dependency — see the source docstring in `zeit/ai/utae.py` for details.

---

## References

- Garnot, V. S. F., & Landrieu, L. (2021). Panoptic segmentation of satellite image time series with convolutional temporal attention networks. In **Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)** (pp. 4852–4861). [https://doi.org/10.1109/ICCV48922.2021.00483](https://doi.org/10.1109/ICCV48922.2021.00483)
- Garnot, V. S. F., & Landrieu, L. (2020). *Lightweight Temporal Self-Attention for Classifying Satellite Image Time Series*. arXiv:2007.00586. [https://arxiv.org/abs/2007.00586](https://arxiv.org/abs/2007.00586)
- Reference implementation: [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps) (MIT License)
