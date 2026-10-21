"""The batch functions of the compiled core release the GIL while they compute.

Each one runs in a thread while the main thread keeps a Python loop going: had
the call held the GIL, the loop would stop for the whole call.
"""
import threading
import time

import numpy as np
import pytest

from zeit import _core


def _series(n, t, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(0.5 + 0.2 * np.sin(np.arange(t) / 3.0) + rng.normal(0, 0.05, (n, t)))


def _seasonal(n, t, period, seed=0):
    rng = np.random.default_rng(seed)
    y = 0.5 + 0.2 * np.sin(2 * np.pi * np.arange(t) / period) + rng.normal(0, 0.03, (n, t))
    y[:, t // 2:] -= 0.25 * (rng.random((n, 1)) < 0.5)
    return np.ascontiguousarray(y)


def _bind(fn, *args, **kwargs):
    return lambda: fn(*args, **kwargs)


# name -> builder of the call; the inputs are made by the builder, outside the timed call
CALLS = {
    "mann_kendall": lambda: _bind(_core.mannkendall.fit_mann_kendall_batch, _series(20000, 40), 1, 0.05, -1, 1, 4, 1),
    "bfast_monitor": lambda: _bind(_core.bfastmonitor.fit_bfast_monitor_batch,
                                   _series(30000, 23 * 8), 2000.0, 2005.0, 23, 3, 0.25, 10, 0.05, 10, 1),
    "bfast_lite": lambda: _bind(_core.bfastlite.fit_bfast_lite_batch, _series(40, 23 * 6), 2000.0, 23, 3, 0.15, 5, 20, 1),
    "bfast": lambda: _bind(_core.bfast.fit_bfast_batch, _seasonal(60, 23 * 6, 23),
                           2000.0, 23, 3, 0.15, 5, 5, 10, 0.05, 20, 1),
    "twdtw": lambda: _bind(_core.twdtw.fit_twdtw_batch,
                           _series(160 * 160, 46).reshape(160, 160, 46), np.arange(0, 46 * 16, 16, dtype=np.int32),
                           np.sin(np.arange(23) / 3.0), np.arange(0, 23 * 16, 16, dtype=np.int32),
                           _core.twdtw.TWDTWParams(), float("inf"), 1),
    "phenology": lambda: _bind(_core.phenology.fit_phenology_batch,
                               _seasonal(400, 137, 46), np.arange(0, 137 * 8, 8, dtype=float), 0, 0, n_jobs=1),
    "landtrendr": lambda: _bind(_core.landtrendr.fit_trajectory_batch,
                                _series(80 * 80, 40).reshape(80, 80, 40), np.arange(1985, 2025, dtype=np.int32),
                                _core.landtrendr.LandTrendrParams(), -9999.0, 1),
    "medoid": lambda: _bind(_core.utils.compute_medoid, np.random.default_rng(0).random((512, 512, 24, 6)), -9999.0),
}


@pytest.mark.parametrize("name", list(CALLS))
def test_batch_releases_the_gil(name):
    call = CALLS[name]()
    t0 = time.perf_counter()
    call()
    duration = time.perf_counter() - t0
    if duration < 0.03:
        pytest.skip(f"{name} too fast here ({duration * 1e3:.1f} ms) to tell")

    done = threading.Event()
    worker = threading.Thread(target=lambda: (call(), done.set()))
    worker.start()
    longest, last = 0.0, time.perf_counter()
    while not done.is_set():
        now = time.perf_counter()
        longest, last = max(longest, now - last), now
    worker.join()
    # holding the GIL would stall this loop for about the whole call
    assert longest < 0.5 * duration, f"main thread stalled {longest:.3f} s of a {duration:.3f} s call"
