import numpy as np
import xarray as xr
import dask.array as da

from zeit._core.bfast import bfast, fit_bfast_batch
from zeit._bfast import run_bfast_dask, bf_metric_names
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor

FREQ = 23
N = 200  # > 2*FREQ, the STL seasonal seed's own minimum-length requirement


def _make_series(n, seed, trend_break_at=None, trend_break_mag=0.4,
                  season_break_at=None, season_break_factor=2.5, nan_idx=None):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    season = 0.1 * np.cos(2 * np.pi * t / FREQ) + 0.05 * np.sin(2 * np.pi * t / FREQ)
    if season_break_at is not None:
        season = season.copy()
        season[season_break_at:] *= season_break_factor
    y = 0.5 + 0.0003 * t + season + rng.normal(0, 0.02, n)
    if trend_break_at is not None:
        y[trend_break_at:] += trend_break_mag
    if nan_idx is not None:
        y[nan_idx] = np.nan
    return y


# --- Single-pixel (unit-testing helper) ------------------------------------

def test_bfast_single_detects_injected_trend_break():
    y = _make_series(N, seed=1, trend_break_at=100)
    r = bfast(list(y), start_time=2000.0, frequency=FREQ)

    assert r.valid == 1.0
    assert int(r.n_trend_breaks) == 1
    assert 90 <= r.trend_breakpoint_idx[0] <= 110
    assert r.magnitude > 0.2
    assert not np.isnan(r.time)


def test_bfast_single_detects_injected_season_break_with_pretest_disabled():
    # The preliminary OLS-MOSUM significance pretest (matching R's
    # sctest(efp(...)), see bfast.h) has genuinely low power against a pure
    # seasonal-amplitude change: a zero-mean oscillation scaled up/down
    # averages to ~0 over the MOSUM window regardless of amplitude -
    # confirmed directly against real R output (see docs/tutorials/bfast.md's
    # Validation section, same seed/scenario as here). level=1.0 bypasses
    # the pretest to exercise the underlying season-breakpoint search engine
    # on its own, which does find the injected break (matching R's own
    # season_bp when its pretest is likewise bypassed).
    y = _make_series(300, seed=2, season_break_at=150, season_break_factor=2.5)
    r = bfast(list(y), start_time=2000.0, frequency=FREQ, level=1.0)

    assert r.valid == 1.0
    assert int(r.n_season_breaks) == 1
    assert 140 <= r.season_breakpoint_idx[0] <= 160


def test_bfast_single_suppresses_weak_season_break_at_default_level():
    # Same injected break, but at the default level=0.05: real R's bfast()
    # also reports zero breaks for this exact scenario (verified directly -
    # see docs/tutorials/bfast.md), since its preliminary significance test
    # rejects the change as not significant. This is intentional fidelity to
    # R's conservative default, not a detection failure.
    y = _make_series(300, seed=2, season_break_at=150, season_break_factor=2.5)
    r = bfast(list(y), start_time=2000.0, frequency=FREQ)

    assert r.valid == 1.0
    assert int(r.n_trend_breaks) == 0
    assert int(r.n_season_breaks) == 0


def test_bfast_single_no_break():
    y = _make_series(N, seed=3)
    r = bfast(list(y), start_time=2000.0, frequency=FREQ)

    assert r.valid == 1.0
    assert int(r.n_trend_breaks) == 0
    assert r.magnitude == 0.0
    assert np.isnan(r.time)


def test_bfast_single_respects_min_length():
    y = _make_series(2 * FREQ, seed=4)  # exactly at the STL minimum-length boundary
    r = bfast(list(y), start_time=2000.0, frequency=FREQ)
    assert r.valid == 0.0


def test_bfast_single_handles_nan_gaps():
    y = _make_series(N, seed=5, trend_break_at=100, nan_idx=np.arange(0, N, 10))
    r = bfast(list(y), start_time=2000.0, frequency=FREQ)
    assert r.valid == 1.0
    assert r.n_valid == N - len(np.arange(0, N, 10))
    assert int(r.n_trend_breaks) == 1


# --- Batch / OpenMP ---------------------------------------------------------

def test_fit_bfast_batch_shape_and_break_detection():
    n_pixels, max_bt, max_bs = 4, 5, 5
    values = np.stack([
        _make_series(N, seed=10, trend_break_at=100),
        _make_series(N, seed=11),
        _make_series(N, seed=12, trend_break_at=100),
        _make_series(N, seed=13),
    ])

    out = fit_bfast_batch(
        values_array=values, start_time=2000.0, frequency=FREQ,
        max_breaks_trend=max_bt, max_breaks_season=max_bs,
    )

    assert out.shape == (7 + max_bt + max_bs, n_pixels)
    n_trend_row = 0
    assert out[n_trend_row, 0] == 1
    assert out[n_trend_row, 1] == 0
    assert out[n_trend_row, 2] == 1
    assert out[n_trend_row, 3] == 0


def test_fit_bfast_batch_is_deterministic_across_thread_counts():
    n_pixels = 20
    values = np.stack([
        _make_series(N, seed=s, trend_break_at=100 if s % 2 == 0 else None)
        for s in range(n_pixels)
    ])

    out_1 = fit_bfast_batch(values_array=values, start_time=2000.0, frequency=FREQ, n_jobs=1)
    out_n = fit_bfast_batch(values_array=values, start_time=2000.0, frequency=FREQ, n_jobs=-1)

    np.testing.assert_allclose(out_1, out_n, equal_nan=True)


# --- Dask wrapper + xarray accessor -----------------------------------------

def test_run_bfast_dask_shape_and_break():
    rows, cols, max_bt, max_bs = 2, 2, 5, 5
    block = np.stack([
        _make_series(N, seed=40 + i, trend_break_at=100)
        for i in range(rows * cols)
    ]).reshape(rows, cols, N).transpose(2, 0, 1)
    data = da.from_array(block, chunks=(N, 2, 2))

    out = run_bfast_dask(data, start_time=2000.0, frequency=FREQ,
                          max_breaks_trend=max_bt, max_breaks_season=max_bs)
    assert out.shape == (7 + max_bt + max_bs, rows, cols)

    computed = out.compute()
    assert np.all(computed[0] == 1)  # n_trend_breaks row


def test_xarray_accessor_bfast():
    rows, cols, max_bt, max_bs = 2, 2, 5, 5
    block = np.stack([
        _make_series(N, seed=50 + i, trend_break_at=100)
        for i in range(rows * cols)
    ]).reshape(rows, cols, N).transpose(2, 0, 1)
    data = da.from_array(block, chunks=(N, 2, 2))

    ds = xr.DataArray(data, dims=["time", "y", "x"], coords={"y": np.arange(rows), "x": np.arange(cols)})
    res = ds.zeit.bfast(start_time=2000.0, frequency=FREQ,
                        max_breaks_trend=max_bt, max_breaks_season=max_bs)

    assert isinstance(res, xr.Dataset)
    assert res.n_trend_breaks.dims == ("y", "x")
    names = ["break_time" if n == "time" else n for n in bf_metric_names(max_bt, max_bs)]
    assert list(res.data_vars) == names

    computed = res.compute()
    assert np.all(computed.n_trend_breaks.values == 1)
