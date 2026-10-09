# Clustering (SOM)

<p class="lead">Find the main types of behaviour in a landscape without any labels. A self-organizing map groups millions of pixel trajectories into a small grid of prototypes, each one a typical time series, arranged so that similar prototypes sit next to each other.</p>

<div class="glance" markdown>
<div><span class="k">Answers</span><span class="v">Which typical trajectories exist here, and where?</span></div>
<div><span class="k">Input</span><span class="v">A 2-D array of samples × features (e.g. pixels × dates)</span></div>
<div><span class="k">Output</span><span class="v">Prototype vectors and the best-matching prototype of each pixel</span></div>
<div><span class="k">Reference</span><span class="v">Kohonen (1990, 2013); bit-exact port of <code>minisom</code></span></div>
</div>

<figure markdown>
  ![A 2x2 batch SOM of 40-year NDVI trajectories in Rondônia: a cluster map and the four prototype trajectories](../assets/figures/som_clusters.webp)
  <figcaption><strong>Result on real data.</strong> 40-year NDVI trajectories of 490,000 pixels in Rondônia, clustered by a 2 × 2 batch SOM trained on a sample of 30,000. The prototypes separate a gradient from intact forest (cluster 1) to the land cleared earliest (cluster 4). No labels were used. <em>Data: annual Landsat NDVI composites exported from <a href="https://github.com/eMapR/LT-GEE">LT-GEE</a> on Google Earth Engine.</em></figcaption>
</figure>

## How it works

A SOM is a grid of **neurons**, each holding a prototype vector with as many values as your features. Training repeatedly finds, for each sample, its closest neuron (its *best-matching unit*, BMU) and moves that neuron **and its grid neighbours** toward the sample. The neighbourhood and the learning rate shrink as training goes on. The result is a set of prototypes that covers the data and is topologically ordered: neighbours on the grid are similar.

Zeit implements both forms of the algorithm in C++:

- **Online SOM** (Kohonen, 1990): one sample at a time, sequential by definition.
- **Batch SOM** (Kohonen, 2013): every prototype is updated from all samples at once in each pass, parallelised with OpenMP. It is much faster on large satellite data sets.

`zeit.ai.SOM` is an **operation-by-operation port of Python [`minisom`](https://github.com/JustGlowing/minisom)**. Given the same seed and arguments it draws the same initial weights and sample order and applies the same arithmetic, so the trained codebook is **bit-for-bit identical** to `MiniSom`, 30–190 times faster. See [Algorithm Fidelity](../benchmarks/fidelity.md#6-self-organizing-maps-som).

## Step by step

### 1. Arrange the data as samples × features

For time-series clustering, each pixel is a sample and each date (or date × band) is a feature:

```python
import numpy as np
from zeit.ai import SOM

# stack: (time, rows, cols)
n_time, rows, cols = stack.shape
X = stack.reshape(n_time, -1).T              # (pixels, time)
valid = np.isfinite(X).all(axis=1)

rng = np.random.default_rng(0)
train = X[rng.choice(np.flatnonzero(valid), 30_000, replace=False)]
```

Training on a random sample is usually enough. Assigning all pixels afterwards is fast.

### 2. Initialise and train

```python
som = SOM(x=2, y=2, input_len=n_time, sigma=0.8, learning_rate=0.5, random_seed=0)
som.random_weights_init(train)           # or som.pca_weights_init(train)

# Batch SOM: 20 passes over the whole sample, parallelised with OpenMP
som.train(train, num_iters=20, algorithm="batch")

prototypes = som.get_weights().reshape(-1, n_time)   # (neurons, time): one typical trajectory each
print(som.quantization_error(train))                   # mean distance from samples to their BMU
```

`sigma` is the initial neighbourhood radius in grid units. Small grids (2 × 2, 3 × 3) work as a clustering method; larger grids (10 × 10 and up) are used to explore the data or as a first step before grouping neurons.

### 3. Assign every pixel

```python
bmu = som.predict(X[valid])              # flat neuron index i * y + j, from 0 to x*y - 1

cluster_map = np.full(rows * cols, -1)
cluster_map[valid] = bmu
cluster_map = cluster_map.reshape(rows, cols)
```

For a single sample, `som.winner(x)` returns the grid coordinates `(i, j)` of its BMU, as in `minisom`.

### 4. Clean labelled samples (optional)

SOMs are also used to check the training data of a supervised classifier: a sample whose label disagrees with the majority label of its neuron is suspicious. `filter_noisy_samples` returns a mask of the samples to **keep**:

```python
keep = som.filter_noisy_samples(X_samples, y_labels)
X_clean, y_clean = X_samples[keep], y_labels[keep]
```

This is the idea behind `sits_som_clean_samples()` in R `sits`.

## Online or batch?

| `algorithm` | Same as `minisom` | Meaning of `num_iters` | Parallel |
| :--- | :--- | :--- | :---: |
| `"online"` (default) | `MiniSom.train(data, num_iters, random_order, use_epochs)` | Number of single-sample updates (epochs with `use_epochs=True`) | No (sequential by definition) |
| `"batch"` | `MiniSom.train_batch_offline(data, num_iters)` | Number of full passes over the data | Yes (`n_jobs`) |

```python
# Online SOM: 50,000 single-sample updates, samples drawn in random order
som.train(X_train, num_iters=50_000, random_order=True)

# Online SOM for 5 epochs over the data
som.train(X_train, num_iters=5, use_epochs=True)
```

!!! warning "`num_iters` depends on the algorithm"
    With the default `algorithm="online"`, `num_iters=20` means 20 single-sample updates, far too few for real data. For large data sets use `algorithm="batch"` with tens of passes, or `"online"` with tens of thousands of updates (or `use_epochs=True`).

`train` continues from the current weights, like `minisom`. Batch results are identical for any `n_jobs`.

## Parameters

| Parameter | Default | Effect |
| :--- | :---: | :--- |
| `x`, `y` | required | Grid size. `x * y` is the number of prototypes. |
| `input_len` | required | Number of features per sample. |
| `sigma` | `1.0` | Initial spread of the neighbourhood function. |
| `learning_rate` | `0.5` | Initial learning rate (online training). |
| `decay_function` | `"asymptotic_decay"` | Learning-rate decay: `"asymptotic_decay"`, `"inverse_decay_to_zero"`, `"linear_decay_to_zero"`. |
| `neighborhood_function` | `"gaussian"` | `"gaussian"`, `"mexican_hat"`, `"bubble"` or `"triangle"`. |
| `topology` | `"rectangular"` | `"rectangular"` or `"hexagonal"` grid. |
| `sigma_decay_function` | `"asymptotic_decay"` | `"asymptotic_decay"`, `"inverse_decay_to_one"`, `"linear_decay_to_one"`. |
| `random_seed` | `42` | Seed for initialisation and sample order (the same draws as `minisom`). |

All the methods are listed in the [API reference](../api/time-series.md#som).

## Good practice

- **Scale features consistently.** The distance treats every feature equally. If you mix bands with different ranges, standardise them first.
- **Handle gaps before training.** Samples with `NaN` should be filled (for example with [`zeit.smooth`](../api/preprocessing.md#smooth)) or left out.
- **Interpret prototypes, not colours.** Plot `som.get_weights()` to understand what each cluster means, as in the figure above.
- **Reproducibility.** Results depend on `random_seed` and on the initialisation. Fix both, and you get the same codebook as `minisom` with the same settings.

## References

- Kohonen, T. (1990). The self-organizing map. *Proceedings of the IEEE*, 78(9), 1464–1480. [doi:10.1109/5.58325](https://doi.org/10.1109/5.58325)
- Kohonen, T. (2013). Essentials of the self-organizing map. *Neural Networks*, 37, 52–65. [doi:10.1016/j.neunet.2012.09.018](https://doi.org/10.1016/j.neunet.2012.09.018)
- Vettigli, G. (2018). MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map. [github.com/JustGlowing/minisom](https://github.com/JustGlowing/minisom)
- R package [`sits`](https://github.com/e-sensing/sits): `sits_som_map()` and `sits_som_clean_samples()`.
