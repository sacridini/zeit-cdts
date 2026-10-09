import numpy as np
import xarray as xr
import dask.array as da

from zeit._core.bfastlite import bfast_lite, fit_bfast_lite_batch
from zeit._bfast import run_bfast_lite_dask, bfl_metric_names
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor

FREQ = 23


def _make_series(n, seed, break_at=None, break_mag=0.4, nan_idx=None):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    y = 0.5 + 0.0003 * t + 0.1 * np.cos(2 * np.pi * t / FREQ) + 0.05 * np.sin(2 * np.pi * t / FREQ) + rng.normal(0, 0.02, n)
    if break_at is not None:
        y[break_at:] += break_mag
    if nan_idx is not None:
        y[nan_idx] = np.nan
    return y


# --- Single-pixel (unit-testing helper) ------------------------------------

def test_bfast_lite_single_detects_injected_break():
    y = _make_series(150, seed=1, break_at=80)
    r = bfast_lite(list(y), start_time=2000.0, frequency=FREQ)

    assert r.valid == 1.0
    assert int(r.n_breaks) >= 1
    assert 70 <= r.breakpoint_idx[0] <= 90  # close to the injected break


def test_bfast_lite_single_no_break():
    y = _make_series(150, seed=2)
    r = bfast_lite(list(y), start_time=2000.0, frequency=FREQ)

    assert r.valid == 1.0
    assert int(r.n_breaks) == 0
    assert len(r.breakpoint_idx) == 0


def test_bfast_lite_single_respects_min_valid():
    y = _make_series(20, seed=3)  # far fewer than any reasonable min_valid
    r = bfast_lite(list(y), start_time=2000.0, frequency=FREQ)
    assert r.valid == 0.0
    assert np.isnan(r.n_breaks)


def test_bfast_lite_single_handles_nan_gaps():
    y = _make_series(150, seed=4, break_at=80, nan_idx=np.arange(0, 150, 10))
    r = bfast_lite(list(y), start_time=2000.0, frequency=FREQ)
    assert r.valid == 1.0
    assert r.n_valid == 150 - 15  # every 10th of 150 dropped


# --- Batch / OpenMP ---------------------------------------------------------

def test_fit_bfast_lite_batch_shape_and_break_detection():
    n_pixels, n_time, max_breaks = 4, 150, 5
    values = np.zeros((n_pixels, n_time))
    values[0] = _make_series(n_time, seed=10, break_at=80)
    values[1] = _make_series(n_time, seed=11)
    values[2] = _make_series(n_time, seed=12, break_at=80)
    values[3] = _make_series(n_time, seed=13)

    out = fit_bfast_lite_batch(
        values_array=values, start_time=2000.0, frequency=FREQ, max_breaks_output=max_breaks,
    )

    assert out.shape == (5 + max_breaks, n_pixels)
    n_breaks_row = 0
    assert out[n_breaks_row, 0] >= 1
    assert out[n_breaks_row, 1] == 0
    assert out[n_breaks_row, 2] >= 1
    assert out[n_breaks_row, 3] == 0


def test_fit_bfast_lite_batch_is_deterministic_across_thread_counts():
    n_pixels, n_time = 30, 150
    values = np.stack([_make_series(n_time, seed=s, break_at=80 if s % 2 == 0 else None) for s in range(n_pixels)])

    out_1 = fit_bfast_lite_batch(values_array=values, start_time=2000.0, frequency=FREQ, n_jobs=1)
    out_n = fit_bfast_lite_batch(values_array=values, start_time=2000.0, frequency=FREQ, n_jobs=-1)

    np.testing.assert_allclose(out_1, out_n, equal_nan=True)


# --- Dask wrapper + xarray accessor -----------------------------------------

def test_run_bfast_lite_dask_shape_and_break():
    time_steps, rows, cols, max_breaks = 150, 3, 3, 5
    block = np.stack([
        _make_series(time_steps, seed=40 + i, break_at=80)
        for i in range(rows * cols)
    ]).reshape(rows, cols, time_steps).transpose(2, 0, 1)
    data = da.from_array(block, chunks=(time_steps, 2, 2))

    out = run_bfast_lite_dask(data, start_time=2000.0, frequency=FREQ, max_breaks_output=max_breaks)
    assert out.shape == (5 + max_breaks, rows, cols)

    computed = out.compute()
    assert np.all(computed[0] >= 1)  # n_breaks row


def test_xarray_accessor_bfast_lite():
    time_steps, rows, cols, max_breaks = 150, 3, 3, 5
    block = np.stack([
        _make_series(time_steps, seed=50 + i, break_at=80)
        for i in range(rows * cols)
    ]).reshape(rows, cols, time_steps).transpose(2, 0, 1)
    data = da.from_array(block, chunks=(time_steps, 3, 3))

    ds = xr.DataArray(data, dims=["time", "y", "x"], coords={"y": np.arange(rows), "x": np.arange(cols)})
    res = ds.zeit.bfast_lite(start_time=2000.0, frequency=FREQ, max_breaks=max_breaks)

    assert isinstance(res, xr.Dataset)
    assert res.n_breaks.dims == ("y", "x")
    assert list(res.data_vars) == bfl_metric_names(max_breaks)

    computed = res.compute()
    assert np.all(computed.n_breaks.values >= 1)
