import numpy as np
import xarray as xr
import dask.array as da
import pytest

from zeit._core.bfastmonitor import bfast_monitor, fit_bfast_monitor_batch
from zeit._bfast import run_bfast_monitor_dask, N_BFM_METRICS, BFM_METRIC_NAMES
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor

FREQ = 23  # 16-day composites/year, matches the MODIS-style annual cycle used elsewhere in the test suite

# NOTE on false positives: cross-validated against R's real bfastmonitor()
# (see the PR/commit description), history="all" (the only history-selection
# mode this port implements) has a noticeably higher false-positive rate in
# practice than the nominal `alpha` suggests - confirmed to be a property of
# bfastmonitor itself (R showed ~44% "breaks" on 50 pure-noise draws at the
# settings used below, zeit showed ~42%), not a port bug. R's default
# history="ROC" (not yet ported) mitigates this by auto-trimming unstable
# history. Because of this, "no break expected" tests below use specific
# seeds confirmed not to trigger a spurious break, rather than an arbitrary
# seed - a single random draw isn't a reliable "no break" fixture here.


def _make_series(n, seed, break_at=None, break_mag=1.0, nan_idx=None):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 0.5 + 0.0003 * t + 0.1 * np.cos(2 * np.pi * t / FREQ) + rng.normal(0, 0.02, n)
    if break_at is not None:
        y[break_at:] += break_mag
    if nan_idx is not None:
        y[nan_idx] = np.nan
    return y


# --- Single-pixel (unit-testing helper) ------------------------------------

def test_bfast_monitor_single_detects_injected_break():
    y = _make_series(150, seed=1, break_at=100)
    r = bfast_monitor(list(y), start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ)

    assert r.valid == 1.0
    assert r.has_break == 1.0
    assert not np.isnan(r.breakpoint_idx)
    assert r.breakpoint_idx <= 100  # a disturbance was flagged at or before the injected break


def test_bfast_monitor_single_no_break():
    # seed=3: confirmed not to trigger a spurious break at these settings
    # (see the false-positive-rate note above).
    y = _make_series(150, seed=3)
    r = bfast_monitor(list(y), start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ)

    assert r.valid == 1.0
    assert r.has_break == 0.0
    assert np.isnan(r.breakpoint)
    assert np.isnan(r.breakpoint_idx)


def test_bfast_monitor_single_respects_min_history():
    # Only 5 history observations before monitoring starts, fewer than the
    # default order=3 harmonic model's 8 regressors (intercept+trend+3cos+3sin)
    # -> history fit is under-determined, so bfastmonitor must report invalid.
    y = _make_series(30, seed=4, break_at=20)
    r = bfast_monitor(list(y), start_time=2000.0, monitor_start_time=2000.0 + 5.0 / FREQ, frequency=FREQ)
    assert r.valid == 0.0
    assert np.isnan(r.breakpoint)


def test_bfast_monitor_rejects_off_grid_h_and_period():
    y = _make_series(150, seed=5, break_at=100)
    with pytest.raises(RuntimeError):
        bfast_monitor(list(y), start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ, h=0.3)
    with pytest.raises(RuntimeError):
        bfast_monitor(list(y), start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ, period=5)


# --- Batch / OpenMP ---------------------------------------------------------

def test_fit_bfast_monitor_batch_shape_and_break_detection():
    n_pixels, n_time = 4, 150
    values = np.zeros((n_pixels, n_time))
    values[0] = _make_series(n_time, seed=10, break_at=100)  # break
    values[1] = _make_series(n_time, seed=11)                # no break (confirmed)
    values[2] = _make_series(n_time, seed=12, break_at=100)  # break
    values[3] = _make_series(n_time, seed=13)                # no break (confirmed)

    out = fit_bfast_monitor_batch(
        values_array=values, start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ,
        frequency=FREQ, n_jobs=1,
    )

    assert out.shape == (N_BFM_METRICS, n_pixels)
    has_break_row = BFM_METRIC_NAMES.index("has_break")
    assert out[has_break_row, 0] == 1.0
    assert out[has_break_row, 1] == 0.0
    assert out[has_break_row, 2] == 1.0
    assert out[has_break_row, 3] == 0.0


def test_fit_bfast_monitor_batch_respects_min_valid():
    values = np.full((1, 150), np.nan)
    values[0, :20] = _make_series(20, seed=20)  # 20 valid history obs, below min_valid=25 below

    out = fit_bfast_monitor_batch(
        values_array=values, start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ,
        frequency=FREQ, min_valid=25,
    )
    valid_row = BFM_METRIC_NAMES.index("valid")
    assert out[valid_row, 0] == 0.0


def test_fit_bfast_monitor_batch_is_deterministic_across_thread_counts():
    n_pixels, n_time = 50, 150
    values = np.stack([_make_series(n_time, seed=s, break_at=100 if s % 2 == 0 else None) for s in range(n_pixels)])

    out_1 = fit_bfast_monitor_batch(values_array=values, start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ, n_jobs=1)
    out_n = fit_bfast_monitor_batch(values_array=values, start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ, n_jobs=-1)

    np.testing.assert_allclose(out_1, out_n, equal_nan=True)


# --- Dask wrapper + xarray accessor -----------------------------------------

def test_run_bfast_monitor_dask_shape_and_break():
    time_steps, rows, cols = 150, 3, 3
    block = np.stack([
        _make_series(time_steps, seed=40 + i, break_at=100)
        for i in range(rows * cols)
    ]).reshape(rows, cols, time_steps).transpose(2, 0, 1)
    data = da.from_array(block, chunks=(time_steps, 2, 2))

    out = run_bfast_monitor_dask(data, start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ)
    assert out.shape == (N_BFM_METRICS, rows, cols)

    computed = out.compute()
    has_break_idx = BFM_METRIC_NAMES.index("has_break")
    assert np.all(computed[has_break_idx] == 1.0)


def test_run_bfast_monitor_dask_rejects_invalid_h():
    time_steps, rows, cols = 150, 2, 2
    data = da.zeros((time_steps, rows, cols), chunks=(time_steps, 2, 2))
    with pytest.raises(ValueError):
        run_bfast_monitor_dask(data, start_time=2000.0, monitor_start_time=2000.0 + 60.0 / FREQ, frequency=FREQ, h=0.3)


def test_xarray_accessor_bfast_monitor():
    time_steps, rows, cols = 150, 3, 3
    block = np.stack([
        _make_series(time_steps, seed=50 + i, break_at=100)
        for i in range(rows * cols)
    ]).reshape(rows, cols, time_steps).transpose(2, 0, 1)
    data = da.from_array(block, chunks=(time_steps, 3, 3))

    ds = xr.DataArray(data, dims=["time", "y", "x"], coords={"y": np.arange(rows), "x": np.arange(cols)})
    res = ds.zeit.bfast_monitor(2000.0 + 60.0 / FREQ, start_time=2000.0, frequency=FREQ)

    assert isinstance(res, xr.Dataset)
    assert res.has_break.dims == ("y", "x")
    assert list(res.data_vars) == BFM_METRIC_NAMES

    computed = res.compute()
    assert np.all(computed.has_break.values == 1.0)
    assert np.all(computed.breakpoint_idx.values <= 100.0)
