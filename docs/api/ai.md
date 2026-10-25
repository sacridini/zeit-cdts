# Deep Learning

<p class="lead">PyTorch models in <code>zeit.ai</code>, and the functions that take them from a cube to a map: <code>samples</code>, <code>train</code> and <code>predict</code>. The models are ordinary <code>nn.Module</code>s, which can also be trained with a loop of your own. Tutorials: <a href="../../tutorials/ai/">Deep Learning</a>.</p>

!!! note "Install"
    `zeit.ai` needs PyTorch, which comes with the `ai` extra: `pip install "zeit-cdts[ai]"` (since zeit 2.3; for a GPU, see [GPU Acceleration](../getting-started/installation.md#gpu-acceleration-for-zeitai)).

## From a cube to a map

The same path as [`train_classifier`](post-processing.md#train_classifier) and [`classify`](post-processing.md#classify), with the deep learning models: labelled points or polygons and a cube give the samples, `train` fits a model with a ready-made loop, and `predict` classifies every pixel into a georeferenced map. What prediction has to check (bands, dates, classes, normalization, patch size) travels with the model.

```python
from zeit import ai

samples = ai.samples(cube, "samples.gpkg", label="class")      # pixels; patch=64 for the patch models
model = ai.train(ai.TempCNN, samples, epochs=50)
classes = ai.predict(model, cube)                               # label (y, x), georeferenced
classes.zeit.save("classes")
ai.save(model, "tempcnn.pt")                                    # ai.load("tempcnn.pt") later
```

| Model | Samples | Dates when predicting |
| :--- | :--- | :--- |
| `TempCNN`, `LightTAE`, `Presto` | pixels (`patch=None`) | as many as in training (e.g. the same composites in another year) |
| `UTAE` | patches, a multiple of `2 ** (stages - 1)` pixels (8 by default) | any: the positions come from the cube's dates |
| `SiameseChangeDetector` | patches of a pair of images `(before, after)`, 16 or more | a pair |
| `Prithvi` | patches of the six HLS bands, a multiple of 16 pixels (224 as pretrained) | as in training |

### `samples` { .api }

<!-- sig: zeit.ai.samples -->
```python
zeit.ai.samples(
    data, samples, *, label="class", patch=None, split=0.2,
    split_by="block", block_size=64, seed=42, nodata="auto",
)
```

Labelled samples of a cube, for `train`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray`, path or pair | required | A `(time, y, x)` or `(time, band, y, x)` cube (in memory or dask), anything `load_raster` reads, or a pair of images `(before, after)` for the Siamese detector. |
| `samples` | `GeoDataFrame` or path | required | Points or polygons with their class, reprojected when in another CRS. |
| `label` | `str` | `"class"` | The column holding the class. |
| `patch` | `int` | `None` | `None`: one sample per pixel, `(time, band)`, for `TempCNN` and `LightTAE`; points on the same pixel count once, a polygon gives every pixel it covers. A size: a `(time, band, patch, patch)` window around each point or small polygon (larger polygons are tiled by windows), with every labelled pixel in it and `-1` elsewhere, for the patch models. |
| `split` | `float` | `0.2` | Share of the samples kept for validation (`0`: none). |
| `split_by` | `str` | `"block"` | `"block"`: whole blocks of `block_size` pixels go to validation, so that it measures how the model does away from what it was trained on (neighbouring pixels are alike: random pixels on both sides would measure that); `"random"`: samples at random. |
| `block_size` | `int` | `64` | Side of the validation blocks, in pixels. |
| `seed` | `int` | `42` | Seed of the split. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation. |

</div>

**Returns** a [`SampleSet`](#sampleset). The bands are normalized by their 2% and 98% quantiles in the training samples, as `sits` does, and missing values become `0` after it (gaps can be filled first with [`zeit.smooth`](preprocessing.md#smooth)). Patch models learn best from dense labels (polygons): a patch around a point has a single labelled pixel.

### `SampleSet` { .api .cls }

<!-- sig: zeit.ai.SampleSet -->
```python
class zeit.ai.SampleSet(X, y, is_val, meta, indices=None)
```

The samples, as a PyTorch `Dataset`, made by `samples`. Each item is a dict with `x` (normalized: `(time, band)` or `(time, band, patch, patch)`), `y` (the class index, or a `(patch, patch)` mask with `-1` where unlabelled) and `positions` (days since the first date). `train` and `val` are the two parts of the split, `classes` the class names, `meta` what `train` records in the model. For a loop of your own: `DataLoader(samples.train, batch_size=64, shuffle=True)`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `X`, `y` | `ndarray` | required | The raw values and the labels. |
| `is_val` | `ndarray` | required | Which samples are for validation. |
| `meta` | `dict` | required | Bands, dates, positions, classes, normalization, patch. |
| `indices` | `ndarray` | `None` | The samples this view holds (`train`, `val`). |

</div>

### `train` { .api }

<!-- sig: zeit.ai.train -->
```python
zeit.ai.train(
    model, samples, *, epochs=50, batch_size=None, lr=0.001,
    weight_decay=0.0, loss="ce", patience=10, device="auto", seed=42,
    verbose=False, **model_kwargs,
)
```

Trains a model on a `SampleSet` with Adam, keeping the weights of the epoch with the lowest validation loss.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | class or `nn.Module` | required | `TempCNN`, `LightTAE`, `Presto`, `UTAE`, `SiameseChangeDetector` or `Prithvi`: a class is built with the samples' bands, dates and classes (and `model_kwargs`); an instance is trained as it is. |
| `samples` | `SampleSet` | required | From `samples`: pixels or patches, as the model takes. |
| `epochs` | `int` | `50` | At most this many passes over the training samples. |
| `batch_size` | `int` | `None` | Samples per step: 64 pixels or 8 patches (a patch holds many labelled pixels, and smaller batches leave more steps per epoch). |
| `lr`, `weight_decay` | `float` | `0.001`, `0.0` | Adam's learning rate and weight decay. |
| `loss` | `str` or function | `"ce"` | `"ce"` (cross-entropy), `"focal"` (for imbalanced classes) or `(logits, target) -> loss`, with `target` `-1` where unlabelled. |
| `patience` | `int` | `10` | Stop after this many epochs without a lower validation loss, once the model has taken 50 steps (validation runs with batch normalization's running statistics, which need them to settle), and keep the best epoch (`None`: every epoch, the last one kept). |
| `device` | `str` | `"auto"` | `"auto"` (the GPU if there is one), `"cpu"`, `"cuda"`... |
| `seed` | `int` | `42` | Seed of the initial weights and the order of the samples. |
| `verbose` | `bool` | `False` | Print each epoch's losses and validation accuracy. |
| `**model_kwargs` | | | For a class: its own arguments (e.g. `dropout_rate`, UTAE's `encoder_widths`). |

</div>

**Returns** the trained model in evaluation mode, with `zeit_meta_` (bands, dates, classes, normalization, patch, the model's class and arguments) and `zeit_history_` (per epoch: `train_loss`, `val_loss`, `val_accuracy`).

### `predict` { .api }

<!-- sig: zeit.ai.predict -->
```python
zeit.ai.predict(
    model, data, *, batch_size=256, overlap=0.25, probability=False,
    device="auto", nodata="auto", chunks=None,
)
```

Classifies every pixel of a cube with a model trained by `train` (or loaded by `load`). [`zeit.classify(cube, model)`](post-processing.md#classify) does the same for such a model.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | required | A model with `zeit_meta_`. |
| `data` | as in `samples` | required | The cube; its bands are taken by name and normalized as the training samples were. A lazy cube stays lazy, computed block by block. |
| `batch_size` | `int` | `256` | Pixels or windows through the model at a time. |
| `overlap` | `float` | `0.25` | Patch models: the image is covered by windows of the training patch size, overlapping by this share; each pixel's probabilities are the average of the windows over it, weighted down towards their edges, so there are no seams. The windows are laid on the whole image, so a lazy cube gives the map in memory. |
| `probability` | `bool` | `False` | Also return each class's probability. |
| `device` | `str` | `"auto"` | As in `train`. |
| `nodata` | `"auto"`, `float` or `None` | `"auto"` | Value marking a missing observation. |
| `chunks` | `"auto"`, `dict` | `None` | A raster path: read lazily. |

</div>

**Returns** an `xarray.Dataset` like `zeit.classify`'s: `label (y, x)` (`1` for the first class, `0` where the pixel has no data at any date; the names in `flag_meanings`) and, with `probability=True`, `probability (class, y, x)`. Georeferenced as the input.

### `save` { .api }

<!-- sig: zeit.ai.save -->
```python
zeit.ai.save(model, path)
```

Saves a trained model: its weights, `zeit_meta_` and history.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | required | A model trained by `train`. |
| `path` | path | required | The file to write. |

</div>

### `load` { .api }

<!-- sig: zeit.ai.load -->
```python
zeit.ai.load(path, model=None, device="cpu")
```

Loads a model saved by `save`, ready for `predict`. A model `train` built from its class is built again; for one trained as an instance, pass one like it.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | path | required | The file `save` wrote. |
| `model` | `nn.Module` | `None` | An instance to put the weights in (a model trained from an instance). |
| `device` | `str` | `"cpu"` | Where to load the weights. |

</div>

## Data

### `STACCubeDataset` { .api .cls }

<!-- sig: zeit.ai.STACCubeDataset -->
```python
class zeit.ai.STACCubeDataset(cube, patch_size=256, stride=256)
```

A PyTorch `Dataset` of the windows of a `(time, band, y, x)` cube, lazy or not, for a loop of your own. The windows start every `stride` pixels and the last one of each row and column lies against the far edge, so they cover the whole cube (a cube smaller than a window gives one, padded with NaN). Only the window asked for is computed, so it works on cubes larger than memory.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cube` | `xr.DataArray` | required | `(time, band, y, x)` cube, e.g. from `build_time_series`. |
| `patch_size` | `int` | `256` | Window side in pixels. |
| `stride` | `int` | `256` | Step between windows. Smaller than `patch_size` gives overlapping windows. |

</div>

Each item is a dict: `x`, a float32 tensor `(time, band, patch_size, patch_size)` with missing values left as NaN; `valid`, `(time, patch_size, patch_size)`, True where every band has a value; `positions`, the days since the cube's first date (for UTAE's `batch_positions`; the day of the year would make dates a year apart collide); and `row`, `col`, the window's top-left pixel. `dataset.positions` holds the same positions.

```python
from zeit.ai import STACCubeDataset

dataset = STACCubeDataset(cube, patch_size=64, stride=64)
item = dataset[0]
x = torch.nan_to_num(item["x"])      # or fill the gaps as your model needs
```

## Pixel classifiers

### `TempCNN` { .api .cls }

<!-- sig: zeit.ai.TempCNN -->
```python
class zeit.ai.TempCNN(
    in_channels, n_times, num_classes=5, hidden_dims=(64, 64, 64),
    kernel_sizes=(3, 3, 3), dropout_rates=(0.2, 0.2, 0.2),
    dense_layer_nodes=256, dense_layer_dropout_rate=0.5,
)
```

Temporal convolutional network (Pelletier et al., 2019), ported layer for layer from R `sits`' `sits_tempcnn()`; weights load directly with `load_state_dict`. Input `(batch, bands, time)`, output class logits `(batch, num_classes)`. Tutorial: [TempCNN](../tutorials/tempcnn.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | required | Bands per observation. |
| `n_times` | `int` | required | Series length. Fixed for the life of the model (the flatten layer depends on it). |
| `num_classes` | `int` | `5` | Output classes. |
| `hidden_dims` | `tuple` | `(64, 64, 64)` | Filters of the three convolution blocks. |
| `kernel_sizes` | `tuple` | `(3, 3, 3)` | Kernel sizes. |
| `dropout_rates` | `tuple` | `(0.2, 0.2, 0.2)` | Dropout after each convolution block. |
| `dense_layer_nodes` | `int` | `256` | Width of the dense layer. |
| `dense_layer_dropout_rate` | `float` | `0.5` | Dropout of the dense layer. |

</div>

```python
from zeit.ai import TempCNN

model = TempCNN(in_channels=6, n_times=23, num_classes=4)
logits = model(torch.randn(32, 6, 23))   # (32, 4)
```

### `LightTAE` { .api .cls }

<!-- sig: zeit.ai.LightTAE -->
```python
class zeit.ai.LightTAE(
    n_bands, day_offsets, n_labels,
    layers_spatial_encoder=(32, 64, 128), n_heads=16,
    n_neurons=(256, 128), dropout_rate=0.2, dim_input_decoder=128,
    dim_layers_decoder=(64, 32),
)
```

Lightweight temporal attention classifier: per-observation MLP encoder, L-TAE temporal attention, MLP decoder. Ported from `sits_lighttae()`. Input `(batch, time, bands)`, output logits `(batch, n_labels)`. Tutorial: [LTAE & LightTAE](../tutorials/ltae.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_bands` | `int` | required | Bands per observation. |
| `day_offsets` | `list[float]` | required | Day of each observation, counted from the first. Fixes the series length. |
| `n_labels` | `int` | required | Output classes. |
| `layers_spatial_encoder` | `tuple` | `(32, 64, 128)` | Widths of the per-observation encoder. |
| `n_heads` | `int` | `16` | Attention heads. |
| `n_neurons` | `tuple` | `(256, 128)` | L-TAE widths; `n_neurons[0]` is the attention dimension. |
| `dropout_rate` | `float` | `0.2` | Dropout. |
| `dim_input_decoder` | `int` | `128` | Decoder input width; must equal `n_neurons[-1]`. |
| `dim_layers_decoder` | `tuple` | `(64, 32)` | Decoder hidden widths. |

</div>

```python
from zeit.ai import LightTAE

model = LightTAE(n_bands=6, day_offsets=list(range(0, 36 * 16, 16)), n_labels=5)
logits = model(torch.randn(8, 36, 6))    # (8, 5)
```

### `LTAE` { .api .cls }

<!-- sig: zeit.ai.LTAE -->
```python
class zeit.ai.LTAE(
    in_channels=128, day_offsets=None, n_heads=16,
    n_neurons=(256, 128), dropout_rate=0.2,
)
```

The L-TAE temporal attention block on its own, to build custom models. Input `(batch, time, in_channels)`, output `(batch, n_neurons[-1])`.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | `128` | Features per time step. |
| `day_offsets` | `list[float]` | `None` | Day of each observation, counted from the first. Required in practice. |
| `n_heads` | `int` | `16` | Attention heads. |
| `n_neurons` | `tuple` | `(256, 128)` | Internal widths. |
| `dropout_rate` | `float` | `0.2` | Dropout. |

</div>

### `Presto` { .api .cls }

<!-- sig: zeit.ai.Presto -->
```python
class zeit.ai.Presto(
    bands, months, num_classes=2, latlon=None, band_map=None,
    reflectance_scale=1.0, s1_linear=False, pretrained=True,
)
```

Presto (Tseng et al., 2023), a transformer pretrained on single pixels' series of Sentinel-1, Sentinel-2, ERA5 and SRTM, ported from the authors' code; with their weights it gives their encodings. Takes the series `(batch, dates, bands)` in the cube's units, NaN where missing, and normalizes them as Presto does; `forward` gives class logits `(batch, num_classes)`, `embed` the 128-dimension encodings. `zeit.ai.train` fills in the bands, months, units and position from the samples. Tutorial: [Presto](../tutorials/presto.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bands` | `list of str` | required | The cube's bands, in order, matched to Presto's inputs by name (`blue`, `nir`, `B8A`, `swir1`, `VV`, `t2m`, `dem`...). |
| `months` | `list of int` | required | Calendar month (1-12) of each date; up to 24 dates. |
| `num_classes` | `int` | `2` | Classes of the classifier. |
| `latlon` | `tuple` | `None` | Latitude and longitude (degrees) given to the pixels when `forward`/`embed` get none of their own. |
| `band_map` | `dict` | `None` | `{Presto input: band}` for bands whose names are not recognized. |
| `reflectance_scale` | `float` | `1.0` | Factor bringing Sentinel-2 to Presto's 0-10000 (`1e4` for reflectance from 0 to 1). |
| `s1_linear` | `bool` | `False` | The Sentinel-1 bands hold linear backscatter (converted to dB). |
| `pretrained` | `bool` | `True` | Start from the authors' weights, downloaded once (3 MB, checked by SHA-256; `ZEIT_WEIGHTS` names a folder with a local copy). |

</div>

```python
from zeit.ai import Presto

model = Presto(["blue", "green", "red", "nir", "swir1", "swir2"], months=range(1, 13), num_classes=4,
               latlon=(-10.5, -55.2), reflectance_scale=1e4)
logits = model(x)            # x: (batch, 12, 6), NaN where missing -> (batch, 4)
encodings = model.embed(x)   # (batch, 128)
```

### `presto_embeddings` { .api }

<!-- sig: zeit.ai.presto_embeddings -->
```python
zeit.ai.presto_embeddings(
    data, *, model=None, band_map=None, batch_size=8192,
    device="auto", nodata="auto", chunks=None,
)
```

Presto's encoding of every pixel of a cube, as an embedding cube `(1, 128, y, x)` (bands `A00`-`A127`, tagged `embedding_source="presto"`), for `similarity`, `som`, `train_classifier`, `zeit.plot` and the rest of zeit's embedding tools. Each pixel gets its own latitude and longitude. Tutorial: [Presto](../tutorials/presto.md#embeddings-of-a-cube).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data` | `DataArray` or path | required | A `(time, band, y, x)` cube of up to 24 dates with Sentinel-2 and/or Sentinel-1, ERA5, SRTM bands, with its dates and CRS. |
| `model` | `Presto` | `None` | A (fine-tuned) `Presto` whose encoder to use; by default the pretrained one, set for the cube's bands, dates and units. |
| `band_map` | `dict` | `None` | `{Presto input: band}` for bands whose names are not recognized. |
| `batch_size` | `int` | `8192` | Pixels per step. |
| `device` | `str` | `"auto"` | `"auto"` (the GPU if there is one), `"cpu"`, `"cuda"`... |
| `nodata` | any | `"auto"` | Value marking a missing observation. |
| `chunks` | any | `None` | A raster path: read lazily with these chunks. A lazy cube stays lazy. |

</div>

```python
emb = zeit.ai.presto_embeddings(cube)            # cube: 12 monthly composites
zeit.similarity(emb, "reference_fields.gpkg")
```

## Patch segmentation

### `UTAE` { .api .cls }

<!-- sig: zeit.ai.UTAE -->
```python
class zeit.ai.UTAE(
    input_dim, encoder_widths=(64, 64, 64, 128),
    decoder_widths=(32, 32, 64, 128), out_conv=(32, 20), str_conv_k=4,
    str_conv_s=2, str_conv_p=1, agg_mode="att_group",
    encoder_norm="group", n_head=16, d_model=256, d_k=4,
    encoder=False, return_maps=False, pad_value=0,
    padding_mode="reflect",
)
```

U-Net with temporal attention (Garnot & Landrieu, 2021), ported from the official `utae-paps` code and bit-exact with it. Input `(batch, time, bands, H, W)` plus `batch_positions` `(batch, time)` (day of each observation); output class scores `(batch, classes, H, W)`. Tutorial: [UTAE](../tutorials/utae.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `input_dim` | `int` | required | Bands. |
| `encoder_widths` | `list[int]` | `(64, 64, 64, 128)` | Encoder widths, finest level first. |
| `decoder_widths` | `list[int]` | `(32, 32, 64, 128)` | Decoder widths. Same length as the encoder; last value equal to the encoder's. |
| `out_conv` | `list[int]` | `(32, 20)` | Output convolution widths; the last is the number of classes. |
| `str_conv_k`, `str_conv_s`, `str_conv_p` | `int` | `4`, `2`, `1` | Kernel, stride and padding of the strided convolutions. |
| `agg_mode` | `str` | `"att_group"` | Temporal aggregation of skip connections: `"att_group"`, `"att_mean"` or `"mean"`. |
| `encoder_norm` | `str` | `"group"` | `"group"`, `"batch"` or `"instance"`. |
| `n_head` | `int` | `16` | Attention heads. |
| `d_model` | `int` | `256` | Attention width (divisible by `n_head`). |
| `d_k` | `int` | `4` | Key size per head. |
| `encoder` | `bool` | `False` | Return features instead of class scores. |
| `return_maps` | `bool` | `False` | Also return the decoder feature maps. |
| `pad_value` | `float` | `0` | Value marking padded time steps (for variable-length batches). |
| `padding_mode` | `str` | `"reflect"` | Spatial padding of the convolutions. |

</div>

```python
from zeit.ai import UTAE

model = UTAE(input_dim=6, out_conv=[32, 10]).eval()
x = torch.randn(2, 12, 6, 128, 128)
days = torch.arange(12.0).expand(2, -1) * 16
scores = model(x, batch_positions=days)   # (2, 10, 128, 128)
```

## Change detection

### `SiameseChangeDetector` { .api .cls }

<!-- sig: zeit.ai.SiameseChangeDetector -->
```python
class zeit.ai.SiameseChangeDetector(
    in_channels, num_classes=2, dropout=0.2,
)
```

Two-date change detection: FC-Siam-diff of Daudt et al. (2018), a U-Net whose encoder, shared by both images, passes the absolute difference of their features at every scale to the decoder. Written from the paper, it gives the outputs of the authors' `SiamUnet_diff` with the same weights. Images of 16 pixels or more (it halves them four times). Tutorial: [Siamese Change Detector](../tutorials/siamese.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_channels` | `int` | required | Bands per image. |
| `num_classes` | `int` | `2` | Output classes (2 for change / no change). |
| `dropout` | `float` | `0.2` | Rate of the dropout (`Dropout2d`) after every convolution, as in the paper. |

</div>

```python
from zeit.ai import SiameseChangeDetector

model = SiameseChangeDetector(in_channels=4)
logits = model(torch.randn(8, 4, 256, 256), torch.randn(8, 4, 256, 256))   # (8, 2, 256, 256)
```

## Foundation models

### `Prithvi` { .api .cls }

<!-- sig: zeit.ai.Prithvi -->
```python
class zeit.ai.Prithvi(
    bands, n_dates, num_classes=2, variant="100M-TL", dates=None,
    latlon=None, band_map=None, reflectance_scale=1.0,
    freeze_encoder=False, pretrained=True,
)
```

Prithvi-EO-2.0 (Szwarcman et al., 2024), the vision transformer NASA and IBM pretrained on HLS time series, ported from the authors' `prithvi_mae.py`: with their weights, its encoder gives their tokens. A decoder of zeit's brings the 16 × 16 tokens back to a class per pixel, with the pixels' own bands. Takes windows `(batch, bands, dates, H, W)` in the cube's units, NaN where missing, `H` and `W` multiples of 16 (14 for the 600M models); gives logits `(batch, num_classes, H, W)`; `encode` gives the encoder's tokens. `zeit.ai.train` fills in the bands, dates, units and place from the samples. Tutorial: [Prithvi](../tutorials/prithvi.md).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bands` | `list of str` | required | The cube's bands, in order; the six HLS bands (`blue`, `green`, `red`, narrow `nir`, `swir1`, `swir2`) are found by name. |
| `n_dates` | `int` | required | Dates of the windows (Prithvi was pretrained on 4; any number works). |
| `num_classes` | `int` | `2` | Classes of the decoder. |
| `variant` | `str` | `"100M-TL"` | `"tiny-TL"`, `"100M-TL"`, `"300M"`, `"300M-TL"`, `"600M"` or `"600M-TL"` (TL: the model also encodes the dates and the place). |
| `dates` | `list` | `None` | The dates, for the TL models. |
| `latlon` | `tuple` | `None` | Latitude and longitude (degrees) of the place, for the TL models. |
| `band_map` | `dict` | `None` | `{Prithvi band: cube band}` for bands whose names are not recognized. |
| `reflectance_scale` | `float` | `1.0` | Factor bringing the bands to reflectance times 10,000 (`1e4` for reflectance from 0 to 1). |
| `freeze_encoder` | `bool` | `False` | Train the decoder only. |
| `pretrained` | `bool` | `True` | Start from the authors' weights, downloaded once from Hugging Face (129 MB to 2.6 GB, checked by SHA-256; `ZEIT_WEIGHTS` names a folder with a local copy). |

</div>

```python
from zeit.ai import Prithvi

model = Prithvi(["blue", "green", "red", "nir", "swir1", "swir2"], n_dates=4, num_classes=3,
                dates=["2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01"], latlon=(-10.5, -55.2))
logits = model(x)    # x: (batch, 6, 4, 224, 224) -> (batch, 3, 224, 224)
```

`GeoFoundationViT`, its name before zeit 1.7 (a wrapper of Prithvi-100M through `transformers`), builds a `Prithvi` with a deprecation warning; its `model_id` names a Prithvi-EO-2.0 repository.

## Losses

### `FocalLoss` { .api .cls }

<!-- sig: zeit.ai.losses.FocalLoss -->
```python
class zeit.ai.losses.FocalLoss(
    alpha=0.25, gamma=2.0, reduction="mean",
)
```

Cross-entropy that down-weights easy examples, for heavily imbalanced problems such as change detection. Takes logits `(B, C, …)` and integer targets.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` | `float` | `0.25` | Overall weight. |
| `gamma` | `float` | `2.0` | Focusing strength. `0` gives plain cross-entropy. |
| `reduction` | `str` | `"mean"` | `"mean"` or `"sum"`. |

</div>

### `TverskyLoss` { .api .cls }

<!-- sig: zeit.ai.losses.TverskyLoss -->
```python
class zeit.ai.losses.TverskyLoss(alpha=0.3, beta=0.7, smooth=1.0)
```

Overlap-based loss for the positive (change) class, with separate weights for false positives and false negatives.

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `alpha` | `float` | `0.3` | Weight of false positives. |
| `beta` | `float` | `0.7` | Weight of false negatives. `beta > alpha` penalises missed changes more. |
| `smooth` | `float` | `1.0` | Smoothing term. |

</div>

### `ContrastiveSiameseLoss` { .api .cls }

<!-- sig: zeit.ai.losses.ContrastiveSiameseLoss -->
```python
class zeit.ai.losses.ContrastiveSiameseLoss(margin=2.0)
```

Contrastive loss on two feature maps: pulls features together where nothing changed (label 0) and pushes them at least `margin` apart where something changed (label 1).

<div class="params" markdown>

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `margin` | `float` | `2.0` | Minimum distance for changed pixels. |

</div>
