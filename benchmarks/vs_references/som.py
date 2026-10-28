"""SOM with the Rust core against minisom 2.3.6, both run here (the published numbers, Zeit's and minisom's, were
taken on a Ryzen 7 7730U laptop, not on this machine).

The workloads of the published rows (the script that produced them is not kept; rebuilt from performance.md, the
fidelity page and cdts_vs_originals' compare_som.py / parallel_som.py, whose data they used):
- single core: 1,500 samples (5 Gaussian blobs, 6 features, seed 0), a 5 x 5 map, sigma 1.2, learning rate 0.5,
  seed 42, random_weights_init, then MiniSom.train(X, 500) (500 online updates) against zeit's SOM.train(X, 500)
  (algorithm 'online'): the time of one fit (train only, the map built and initialised outside the timing), the mean
  and the best of 20 fits after a warm-up, both pinned to one processor.
- multi-core: 20,000 samples (seed 0), a 10 x 10 map, 200 batch iterations (MiniSom.train_batch_offline against
  SOM.train(algorithm='batch')), zeit with n_jobs=1 and n_jobs=8 (one cold run each, then the warm best of 3),
  minisom one full run of 200 iterations on one thread (OpenBLAS kept to one thread) and, as published, the time of
  10 iterations times 20.
The codebooks must be identical (the port is bit-exact).

    python benchmarks/vs_references/som.py
"""
import common  # first: thread variables

import time

import numpy as np
from minisom import MiniSom

from zeit._som import SOM
from common import THREADS, once, pin, record, timed, warm_best

PUBLISHED = {"single": (0.42, 15.58), "batch": (3077.0, 664.0, 15, 126000.0)}


def blobs(n_per_class, seed=0, n_features=6, n_classes=5):
    """compare_som.py's make_blobs (n_per_class=300) / parallel_som.py's (n_per_class=4000, shuffled in place)."""
    rng = np.random.RandomState(seed)
    base = np.eye(n_classes, n_features) * 6.0
    base[:, n_classes:] = rng.uniform(-1, 1, size=(n_classes, n_features - n_classes))
    X = np.vstack([base[c] + rng.normal(0, 0.5, size=(n_per_class, n_features)) for c in range(n_classes)])
    if n_per_class == 300:  # compare_som.py: labels, then a permutation
        return X[rng.permutation(len(X))]
    rng.shuffle(X)
    return X


def zeit_som(X, side):
    s = SOM(side, side, X.shape[1], sigma=1.2, learning_rate=0.5, random_seed=42)
    s.random_weights_init(X)
    return s


def mini_som(X, side):
    m = MiniSom(side, side, X.shape[1], sigma=1.2, learning_rate=0.5, random_seed=42,
                neighborhood_function="gaussian")
    m.random_weights_init(X)
    return m


def fits(make, train, X, n=20):
    """Times train(model) for n fresh models (after one warm-up fit); returns (mean ms, best ms, last model)."""
    m = make(X)
    train(m)
    times = []
    for _ in range(n):
        m = make(X)
        t0 = time.perf_counter()
        train(m)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times)), float(np.min(times)), m


def single_core():
    X = blobs(300)
    pin(1)
    z_mean, z_best, zm = timed(lambda: fits(lambda d: zeit_som(d, 5), lambda s: s.train(X, 500, algorithm="online"), X),
                               "zeit online")
    m_mean, m_best, mm = timed(lambda: fits(lambda d: mini_som(d, 5), lambda s: s.train(X, 500, verbose=False), X),
                               "minisom online")
    pin(None)
    diff = float(np.max(np.abs(zm.get_weights() - mm.get_weights())))
    pub_z, pub_m = PUBLISHED["single"]
    record({"table": "single-core", "row": "SOM (training)", "order": 5,
            "workload": "1,500 samples x 6 features, 5x5 map, 500 online iterations (train only)", "unit": "ms/fit",
            "published_cpp_1t": pub_z, "rust_1t": z_mean, "rust_1t_best": z_best, "ratio_rust_cpp_1t": z_mean / pub_z,
            "reference_1t": m_mean, "reference_1t_best": m_best, "reference_published": pub_m,
            "reference_tool": "Python minisom 2.3.6 MiniSom.train (re-run here)",
            "speedup_vs_reference_1t": m_mean / z_mean, "published_speedup": 37.2,
            "agreement": {"max_abs_weight_diff_vs_minisom": diff},
            "notes": "both sides re-run on this machine (the published row was measured on a Ryzen 7 7730U laptop, "
                     "so ratio_rust_cpp_1t mixes machines; the speedup vs minisom is same-machine). Mean of 20 fits "
                     "after a warm-up, pinned to CPU 0; workload rebuilt (the producing script is not kept)."})


def multi_core():
    X = blobs(4000)
    s = zeit_som(X, 10)
    w0 = s.weights.copy()
    res = {}

    def go(n):
        s.weights = w0.copy()
        s.train(X, 200, n_jobs=n, algorithm="batch")
        res[n] = s.weights.copy()

    pin(1)
    t1 = timed(lambda: once(lambda: go(1)), "1T") * 1000
    pin(THREADS)
    t8 = timed(lambda: once(lambda: go(THREADS)), "8T") * 1000
    pin(1)
    w1 = timed(lambda: warm_best(lambda: go(1)), "1T warm") * 1000
    pin(THREADS)
    w10 = timed(lambda: warm_best(lambda: go(THREADS)), "8T warm") * 1000
    pin(1)
    holder = {}

    def mini_full():
        m = mini_som(X, 10)
        t0 = time.perf_counter()
        m.train_batch_offline(X, 200)
        holder["m"] = m
        return (time.perf_counter() - t0) * 1000.0

    def mini_10():
        m = mini_som(X, 10)
        t0 = time.perf_counter()
        m.train_batch_offline(X, 10)
        return (time.perf_counter() - t0) * 1000.0 * 20

    m200 = timed(mini_full, "minisom 200")
    m10x20 = timed(mini_10, "minisom 10 x 20")
    pin(None)
    diff = float(np.max(np.abs(res[1] - holder["m"].get_weights())))
    p1, pm, pthr, pref = PUBLISHED["batch"]
    record({"table": "multi-core", "row": "SOM (batch)", "order": 10,
            "workload": "20,000 samples x 6 features, 10x10 map, 200 batch iterations", "unit": "ms",
            "published_cpp_1t": p1, "published_cpp_multi": pm, "published_threads": pthr,
            "rust_1t": t1, "rust_8t": t8, "rust_1t_warm_best3": w1, "rust_8t_warm_best3": w10,
            "ratio_rust_cpp_1t": t1 / p1, "ratio_rust8_cpp19": t8 / pm,
            "reference_1t": m200, "reference_1t_extrapolated_10x20": m10x20, "reference_published": pref,
            "reference_tool": "Python minisom 2.3.6 train_batch_offline, 1 thread (re-run here)",
            "speedup_vs_reference_1t": m200 / t1, "speedup_vs_reference_multi": m200 / t8,
            "agreement": {"max_abs_weight_diff_vs_minisom": diff,
                          "outputs_1t_equal_8t": bool(np.array_equal(res[1], res[THREADS]))},
            "notes": "published: Ryzen 7 7730U laptop, 15 threads (ratio_rust_* mixes machines; the minisom speedup "
                     "is same-machine). minisom has no parallel mode: speedup_vs_reference_multi is minisom on 1 "
                     "thread over zeit on 8. rust_1t / rust_8t: one cold run each, then the warm best of 3."})


if __name__ == "__main__":
    common.start()
    single_core()
    multi_core()
