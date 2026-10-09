# GeoFoundationViT

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Reuse a large pretrained model when you have few labels.</span></div>
<div><span class="k">Input</span><span class="v">Backbone-dependent (Prithvi: 6 bands × time × H × W)</span></div>
<div><span class="k">Output</span><span class="v">A segmentation map from a fine-tuned head</span></div>
<div><span class="k">Reference</span><span class="v">Prithvi-100M (Jakubik et al., 2023) via HuggingFace</span></div>
</div>

`GeoFoundationViT` is a thin wrapper that lets you plug large, pretrained **geospatial foundation models** (Vision Transformers trained on massive satellite imagery corpora) into a `zeit` workflow, and fine-tune a lightweight classification/segmentation head on top for your own downstream task. Rather than training a model from scratch, you're doing **transfer learning** from a model that has already learned general-purpose visual representations of satellite imagery.

By default it loads NASA/IBM's **Prithvi-100M** via HuggingFace `transformers`:

> Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., Zadrozny, B., et al. (2023). *Foundation models for generalist geospatial artificial intelligence*. arXiv:2310.18660. [https://arxiv.org/abs/2310.18660](https://arxiv.org/abs/2310.18660)

but any compatible HuggingFace geospatial ViT (e.g. SatMAE-style models) can be loaded by passing a different `model_id`.

!!! note "A wrapper, not a port"
    **Unlike** [LightTAE](ltae.md), [TempCNN](tempcnn.md), and [UTAE](utae.md) — which are `zeit`-native architectures ported and validated against reference implementations — `GeoFoundationViT` is a **wrapper around an external pretrained model**. Its behavior and output quality depend entirely on the backbone you load; there is no `zeit`-side numerical validation to speak of here, since correctness is inherited from the upstream model.

## How It Works

```
input (B, Bands, Time, H, W) -> ViT backbone -> classifier (1x1 Conv2d) -> per-pixel class logits
```

1. **Backbone**: on construction, `GeoFoundationViT` attempts to load the requested `model_id` from the HuggingFace Hub via `transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)`.
2. **Graceful fallback**: if the download fails (no network access, model unavailable, missing `transformers` extras, etc.), the wrapper does **not** raise — it silently falls back to a randomly-initialized `Conv3d` projection (`self.fallback_conv`) standing in for the backbone. This keeps the class usable offline/in CI, but means predictions will be meaningless until you either restore network access or explicitly train the fallback conv from scratch. **Check `model.has_hf` after construction** to know which path you're on.
3. **Reshape + classify**: the backbone's patch-token output (`last_hidden_state`, shape `(B, Seq, Dim)`) is reshaped back into a 2D feature map (assuming a square patch grid, `H = W = sqrt(Seq)`), upsampled by 16x (matching the typical ViT patch size) to recover roughly the original spatial resolution, and passed through a final `1x1 Conv2d` classifier head to produce per-pixel class logits.

## When to Use It

Use `GeoFoundationViT` when you have **limited labeled data** for your specific task but want to benefit from representations learned on a much larger, general-purpose satellite imagery corpus — a classic transfer-learning scenario. If you have ample labeled training data and want an architecture purpose-built and validated for time-series classification/segmentation, prefer [LightTAE](ltae.md), [TempCNN](tempcnn.md), or [UTAE](utae.md) instead.

## From a cube to a map

`GeoFoundationViT` segments windows of the cube; with Prithvi-100M they are 224 × 224 pixels of 6 bands (blue, green, red, narrow NIR, SWIR 1 and 2) over 3 dates:

```python
import zeit
from zeit import ai

cube = zeit.load_raster("hls_3dates.tif")                               # (3, 6, y, x)
samples = ai.samples(cube, "fields.gpkg", label="class", patch=224)
model = ai.train(ai.GeoFoundationViT, samples, epochs=20, lr=1e-4)
classes = ai.predict(model, cube)
```

For the two-stage fine-tuning below, build the model yourself, freeze the backbone and pass the instance: `ai.train(model, samples, ...)`. See [Deep Learning](ai.md#from-a-cube-to-a-map).

## Instantiating the Model

`zeit.ai.train` builds the model for you from the samples (bands, dates, classes); the arguments below pass through it, e.g. `ai.train(ai.GeoFoundationViT, samples, model_id="ibm-nasa-geospatial/Prithvi-100M")`. To build it yourself:

```python
from zeit.ai import GeoFoundationViT

model = GeoFoundationViT(
    model_id="ibm-nasa-geospatial/Prithvi-100M",
    num_classes=2,
)

if not model.has_hf:
    print("Warning: backbone failed to load from HuggingFace Hub - using untrained fallback conv.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

## Fine-Tuning

Because the backbone carries pretrained weights worth preserving, it's common to **freeze it initially** and only train the lightweight classifier head, then optionally unfreeze the backbone for a lower-learning-rate fine-tuning pass once the head has converged:

```python
import torch.optim as optim
from zeit.ai.losses import FocalLoss

# Stage 1: freeze the backbone, train only the classifier head
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:  # your DataLoader
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Head-only] Epoch [{epoch + 1}/10], Loss: {epoch_loss / len(train_loader):.4f}")

# Stage 2 (optional): unfreeze the backbone for full fine-tuning at a lower LR
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Full fine-tune] Epoch [{epoch + 1}/5], Loss: {epoch_loss / len(train_loader):.4f}")
```

## A loop of your own

`zeit.ai.train` covers the usual case. For anything else (another optimizer, a scheduler, augmentation), the `SampleSet` is a PyTorch `Dataset` whose items hold the normalized `x`, the label `y` and the date `positions`; see [A loop of your own](ai.md#a-loop-of-your-own). A model trained this way can still be classified with `zeit.ai.predict` after `zeit.ai.train(model, samples, epochs=0)` records the samples' metadata in it.

## Caveats

- **Network access required for the pretrained path**: the first construction of `GeoFoundationViT` needs to reach the HuggingFace Hub (or a local cache) to download the backbone weights. In offline/air-gapped environments, pre-download the model or explicitly train the fallback path.
- **`trust_remote_code=True`**: loading Prithvi-style models runs custom model code shipped alongside the weights on the Hub. Only point `model_id` at sources you trust.
- **Fallback silently degrades quality**: because construction never raises on a failed download, always check `model.has_hf` in automated pipelines to avoid silently training/evaluating on the untrained fallback path.

---

## References

- Jakubik, J., Roy, S., Phillips, C. E., Fraccaro, P., Godwin, D., Zadrozny, B., et al. (2023). *Foundation models for generalist geospatial artificial intelligence*. arXiv:2310.18660. [https://arxiv.org/abs/2310.18660](https://arxiv.org/abs/2310.18660)
- Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D., & Ermon, S. (2022). SatMAE: Pre-training transformers for temporal and multi-spectral satellite imagery. In **Advances in Neural Information Processing Systems 35 (NeurIPS 2022)**.
