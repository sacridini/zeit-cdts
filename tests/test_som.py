"""SOM: parity with Python MiniSom and invariants.

tests/data/som_minisom_parity.npz holds codebooks trained by MiniSom 2.3.6
(see make_som_parity_fixtures.py); zeit.ai.SOM is a port of the same
algorithms and must reproduce them. When minisom is installed, extra tests
compare against it live.
"""
import json
import os
import warnings

import numpy as np
import pytest

from zeit._som import SOM

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = np.load(os.path.join(HERE, "data", "som_minisom_parity.npz"))
PARITY_CASES = sorted({k.split("__")[0] for k in PARITY.files})

# Bitwise equal where NumPy and the C runtime share exp(). The fixture was made
# on Windows (MSVC exp) and a NumPy build with its own SIMD exp (AVX-512) or
# glibc may differ by an ulp; some cases grow the weights to ~1e12, hence a
# relative as well as an absolute tolerance.
RTOL = 1e-12
ATOL = 1e-10


def blobs(n_per_class=60, n_features=6, n_classes=4, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-5, 5, size=(n_classes, n_features))
    X = np.vstack([c + rng.normal(0, 0.5, size=(n_per_class, n_features)) for c in centers])
    y = np.repeat(np.arange(n_classes), n_per_class)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def train_from_config(cfg, X, init_weights, n_jobs=1):
    kw = dict(cfg["som"])
    som = SOM(**kw)
    if cfg["init"] == "random":
        som.random_weights_init(X)
    elif cfg["init"] == "pca":
        # eigh's eigenvector signs depend on the LAPACK build: start from
        # MiniSom's stored PCA weights (PCA itself: test_pca_init_matches_minisom).
        som.weights = init_weights.copy()
    np.testing.assert_array_equal(som.get_weights(), init_weights)
    tr = cfg["train"]
    if cfg["method"] == "train_batch_offline":
        som.train(X, tr["num_iteration"], algorithm="batch", n_jobs=n_jobs)
    else:
        som.train(X, tr["num_iteration"], random_order=tr.get("random_order", False),
                  use_epochs=tr.get("use_epochs", False))
    return som


# --- parity with the reference implementation ---

@pytest.mark.parametrize("name", PARITY_CASES)
def test_matches_minisom_fixture(name):
    X = PARITY[name + "__data"]
    want = PARITY[name + "__weights"]
    cfg = json.loads(str(PARITY[name + "__config"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        som = train_from_config(cfg, X, PARITY[name + "__init"])
    np.testing.assert_allclose(som.get_weights(), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name", [c for c in PARITY_CASES if c.startswith("batch")])
@pytest.mark.parametrize("n_jobs", [2, 4])
def test_batch_matches_fixture_with_threads(name, n_jobs):
    X = PARITY[name + "__data"]
    cfg = json.loads(str(PARITY[name + "__config"]))
    som = train_from_config(cfg, X, PARITY[name + "__init"], n_jobs=n_jobs)
    np.testing.assert_allclose(som.get_weights(), PARITY[name + "__weights"], rtol=RTOL, atol=ATOL)


def test_initial_weights_match_minisom_draws():
    # MiniSom.__init__: rand(x, y, input_len) * 2 - 1, unit-normalized.
    rng = np.random.RandomState(3)
    w = rng.rand(4, 5, 7) * 2 - 1
    w /= np.linalg.norm(w, axis=-1, keepdims=True)
    np.testing.assert_array_equal(SOM(4, 5, 7, random_seed=3).get_weights(), w)


def test_pca_init_matches_minisom():
    minisom = pytest.importorskip("minisom")
    X, _ = blobs(n_features=8)
    ref = minisom.MiniSom(5, 4, 8, random_seed=1)
    som = SOM(5, 4, 8, random_seed=1)
    ref.pca_weights_init(X)
    som.pca_weights_init(X)
    np.testing.assert_array_equal(som.get_weights(), ref.get_weights())


@pytest.mark.parametrize("algorithm", ["online", "batch"])
@pytest.mark.parametrize("neighborhood", ["gaussian", "mexican_hat", "bubble", "triangle"])
@pytest.mark.parametrize("topology", ["rectangular", "hexagonal"])
def test_matches_minisom_live(algorithm, neighborhood, topology):
    minisom = pytest.importorskip("minisom")
    X, _ = blobs(n_features=10)
    kw = dict(sigma=2.0, learning_rate=0.4, neighborhood_function=neighborhood,
              topology=topology, random_seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = minisom.MiniSom(6, 5, X.shape[1], **kw)
        som = SOM(6, 5, X.shape[1], **kw)
    ref.random_weights_init(X)
    som.random_weights_init(X)
    if algorithm == "batch":
        ref.train_batch_offline(X, 8)
        som.train(X, 8, algorithm="batch", n_jobs=2)
    else:
        ref.train(X, 600, random_order=True)
        som.train(X, 600, random_order=True)
    np.testing.assert_allclose(som.get_weights(), ref.get_weights(), rtol=RTOL, atol=ATOL)
    ref_bmus = np.array([np.ravel_multi_index(ref.winner(v), (6, 5)) for v in X])
    np.testing.assert_array_equal(som.predict(X), ref_bmus)
    assert som.quantization_error(X) == pytest.approx(ref.quantization_error(X), abs=1e-9)


# --- invariants ---

def brute_force_bmus(X, weights):
    flat = weights.reshape(-1, weights.shape[-1])
    return np.array([np.argmin(np.linalg.norm(x - flat, axis=-1)) for x in X])


@pytest.mark.parametrize("n_features", [1, 5, 8, 40, 200])
def test_predict_matches_brute_force(n_features):
    X, _ = blobs(n_features=n_features, seed=n_features)
    som = SOM(7, 3, n_features, random_seed=1)
    som.random_weights_init(X)
    som.train(X, 300)
    np.testing.assert_array_equal(som.predict(X), brute_force_bmus(X, som.get_weights()))
    np.testing.assert_array_equal(som.predict(X, n_jobs=1), som.predict(X, n_jobs=4))


def test_predict_ties_pick_first_neuron():
    som = SOM(2, 2, 2)
    som.weights[:] = 1.0  # every neuron equidistant
    som._is_trained = True
    np.testing.assert_array_equal(som.predict(np.array([[0.0, 0.0], [3.0, -2.0]])), [0, 0])


def test_batch_is_deterministic_across_thread_counts():
    X, _ = blobs(n_per_class=500, n_features=12)
    results = []
    for n_jobs in (1, 3, 8):
        som = SOM(8, 8, 12, sigma=2.0, random_seed=0)
        som.random_weights_init(X)
        som.train(X, 10, algorithm="batch", n_jobs=n_jobs)
        results.append(som.get_weights().copy())
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


@pytest.mark.parametrize("algorithm", ["online", "batch"])
def test_training_reduces_quantization_error(algorithm):
    X, y = blobs()
    som = SOM(5, 5, X.shape[1], sigma=1.5)  # default init: unit vectors, far from the data
    before = som.quantization_error(X)
    som.train(X, 2000 if algorithm == "online" else 20, algorithm=algorithm, random_order=True)
    assert som.quantization_error(X) < 0.5 * before
    # Well separated blobs never share a neuron.
    bmus = som.predict(X)
    for neuron in np.unique(bmus):
        assert len(np.unique(y[bmus == neuron])) == 1


def test_train_continues_from_current_weights():
    X, _ = blobs()
    a = SOM(4, 4, X.shape[1], random_seed=2)
    a.random_weights_init(X)
    start = a.get_weights().copy()
    a.train(X, 50)
    assert not np.array_equal(a.get_weights(), start)

    b = SOM(4, 4, X.shape[1], random_seed=2)
    b.random_weights_init(X)
    np.testing.assert_array_equal(b.get_weights(), start)
    b.train(X, 50)
    np.testing.assert_array_equal(a.get_weights(), b.get_weights())


def test_random_weights_init_picks_samples():
    X, _ = blobs()
    som = SOM(3, 3, X.shape[1])
    som.random_weights_init(X)
    for w in som.get_weights().reshape(-1, X.shape[1]):
        assert (X == w).all(axis=1).any()


def test_winner_and_quantization():
    X, _ = blobs()
    som = SOM(4, 3, X.shape[1])
    som.random_weights_init(X)
    som.train(X, 200)
    bmus = som.predict(X)
    assert som.winner(X[7]) == np.unravel_index(bmus[7], (4, 3))
    np.testing.assert_array_equal(som.quantization(X), som.get_weights().reshape(-1, X.shape[1])[bmus])


def reference_filter(winners, labels):
    clean = np.ones(len(labels), dtype=bool)
    for w in np.unique(winners):
        mask = winners == w
        vals, counts = np.unique(labels[mask], return_counts=True)
        clean[mask & (labels != vals[np.argmax(counts)])] = False
    return clean


@pytest.mark.parametrize("seed", range(5))
def test_filter_noisy_samples_matches_majority_rule(seed):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(300, 3))
    labels = rng.choice(np.array(["a", "b", "c"]), size=300)
    som = SOM(3, 3, 3, random_seed=seed)
    som.train(X, 300, random_order=True)
    np.testing.assert_array_equal(som.filter_noisy_samples(X, labels),
                                  reference_filter(som.predict(X), labels))


def test_filter_noisy_samples_flags_mislabeled():
    X, y = blobs(n_per_class=80)
    noisy = y.copy()
    flipped = [3, 50, 120]
    for i in flipped:
        noisy[i] = (y[i] + 1) % 4
    som = SOM(4, 4, X.shape[1], sigma=1.5)
    som.random_weights_init(X)
    som.train(X, 20, algorithm="batch")
    clean = som.filter_noisy_samples(X, noisy)
    assert not clean[flipped].any()
    assert clean.sum() >= len(X) - 10


# --- validation ---

def test_rejects_bad_input():
    som = SOM(3, 3, 4)
    with pytest.raises(ValueError):
        som.train(np.zeros((10, 3)), 10)             # wrong feature count
    with pytest.raises(ValueError):
        som.train(np.zeros(10), 10)                  # not 2D
    with pytest.raises(ValueError):
        som.train(np.full((10, 4), np.nan), 10)      # NaN
    with pytest.raises(ValueError):
        som.train(np.zeros((10, 4)), 0)              # no iterations
    with pytest.raises(ValueError):
        som.train(np.zeros((10, 4)), 10, algorithm="sgd")
    with pytest.raises(ValueError):
        som.predict(np.zeros((10, 4)))               # untrained
    with pytest.raises(ValueError):
        SOM(3, 3, 4, neighborhood_function="cone")
    with pytest.raises(ValueError):
        SOM(3, 3, 4, topology="torus")
    with pytest.raises(ValueError):
        SOM(3, 3, 4, decay_function="step")
    with pytest.raises(ValueError):
        SOM(3, 3, 1).pca_weights_init(np.zeros((10, 1)))
