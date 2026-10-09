import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import pytest
from unittest.mock import patch

from zeit._phenology import run_phenology_dask
import zeit.xarray_api # Registers the accessor

def mock_fit_phenology_batch(values_array, dates_array, curve_type, extraction_method, max_seasons, whittaker_lambda, apply_whittaker, apply_hants, hants_frequencies, hants_threshold, min_season_length, min_amplitude, min_pixel_amplitude, n_jobs, **kwargs):
    n_pixels = values_array.shape[0]
    out_arr = np.zeros((21, n_pixels, max_seasons), dtype=np.float32)
    for s in range(max_seasons):
        out_arr[6, :, s] = 100.0 + s * 365.25 # DER.sos is index 6
        out_arr[8, :, s] = 200.0 + s * 365.25 # DER.eos is index 8
        out_arr[17, :, s] = 100.0             # LOS is index 17
        out_arr[18, :, s] = 150.0 + s * 365.25 # POP is index 18
        out_arr[19, :, s] = 0.9                # R2 is index 19
        out_arr[20, :, s] = 0.05               # RMSE is index 20
    return out_arr

@patch('zeit._phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_run_phenology_dask(mock_fit):
    time_steps = 20
    rows = 10
    cols = 10
    max_seasons = 2
    
    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    dates = np.arange(time_steps)
    
    result = run_phenology_dask(data, dates, curve_type=0, max_seasons=max_seasons)
    
    assert result.shape == (21, max_seasons, rows, cols)
    assert result.chunks == ((21,), (max_seasons,), (5, 5), (5, 5))

    res_computed = result.compute()

    assert res_computed.shape == (21, max_seasons, rows, cols)
    # DER.sos is at index 6
    np.testing.assert_allclose(res_computed[6, 0, 0, 0], 100.0)
    # DER.eos is at index 8
    np.testing.assert_allclose(res_computed[8, 0, 0, 0], 200.0)
    # LOS is at index 17: a duration, placed in the year of its season's POP (every year,
    # not only the first, which is where decoding it as a date put it)
    np.testing.assert_allclose(res_computed[17, 0, 0, 0], 100.0)
    np.testing.assert_allclose(res_computed[17, 1, 0, 0], 100.0)
    # POP is at index 18
    np.testing.assert_allclose(res_computed[18, 0, 0, 0], 150.0)
    # R2 is at index 19, RMSE at index 20; both ride along with POP's year
    np.testing.assert_allclose(res_computed[19, 0, 0, 0], 0.9)
    np.testing.assert_allclose(res_computed[20, 0, 0, 0], 0.05)

@patch('zeit._phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_xarray_accessor_phenology(mock_fit):
    time_steps = 20
    rows = 10
    cols = 10
    
    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    dates = pd.date_range("2001-01-01", periods=time_steps, freq="16D")
    ds = xr.DataArray(data, dims=["time", "y", "x"],
                      coords={"time": dates, "y": np.arange(rows), "x": np.arange(cols)})

    res = ds.zeit.phenology(curve="elmore", max_seasons=2)

    assert isinstance(res, xr.Dataset)
    assert res["DER.sos"].dims == ("year", "y", "x")
    assert res["DER.sos"].shape == (2, rows, cols) and len(res.data_vars) == 21
    assert res.year.values.tolist() == [2001, 2002]

    res_computed = res.compute()
    sos_mean = res_computed["DER.sos"].values.mean()
    assert 99.0 <= sos_mean <= 101.0
    eos_mean = res_computed["DER.eos"].values.mean()
    assert 199.0 <= eos_mean <= 201.0

def test_real_phenology_extraction_advanced_params():
    """
    Test the actual C++ backend (no mocks) with the new advanced parameters
    including HANTS, Derivative extraction, and season length constraints.
    """
    time_steps = 46 # 2 years of 16-day composites
    rows = 2
    cols = 2
    max_seasons = 2
    
    # Two years of 16-day composites
    dates = pd.DatetimeIndex(np.concatenate([pd.date_range(f"{2001 + k}-01-01", periods=23, freq="16D")
                                             for k in range(2)]))

    # Create a clean synthetic curve
    np.random.seed(42)
    data = np.zeros((time_steps, rows, cols))
    for t in range(time_steps):
        doy = dates[t].dayofyear - 1
        growth = np.sin((doy / 365.0) * np.pi - np.pi/2)
        growth = (growth + 1) / 2.0
        ndvi = 0.2 + growth * 0.6
        data[t, :, :] = ndvi + np.random.normal(0, 0.05, (rows, cols))
        
    ds = xr.DataArray(da.from_array(data, chunks=(time_steps, 2, 2)),
                      dims=["time", "y", "x"],
                      coords={"time": dates, "y": np.arange(rows), "x": np.arange(cols)})

    # Run with HANTS + Derivative Method
    res = ds.zeit.phenology(
        curve="beck",
        method="derivative",
        max_seasons=max_seasons,
        apply_whittaker=False,
        apply_hants=True,
        hants_frequencies=3,
        min_season_length=3,
        min_amplitude=0.1,
        n_jobs=1 # Test single thread as well
    )
    
    res_computed = res.compute()
    
    # Ensure it didn't crash and returned valid shapes
    assert len(res_computed.data_vars) == 21
    assert res_computed["LOS"].shape == (max_seasons, rows, cols)

    # Assert at least some seasons were detected
    assert np.nanmean(res_computed["LOS"].values) > 0

@patch('zeit._phenology.fit_phenology_batch', side_effect=mock_fit_phenology_batch)
def test_xarray_accessor_phenology_no_annual(mock_fit):
    time_steps = 20
    rows = 5
    cols = 5

    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, 5, 5))
    dates = pd.date_range("2001-01-01", periods=time_steps, freq="16D")
    ds = xr.DataArray(data, dims=["time", "y", "x"],
                      coords={"time": dates, "y": np.arange(rows), "x": np.arange(cols)})

    res = ds.zeit.phenology(curve="elmore", max_seasons=3, annual=False)

    assert isinstance(res, xr.Dataset)
    assert res["POP"].dims == ("season", "y", "x")
    assert res["POP"].shape == (3, rows, cols)


# --- New tests: gap-analysis follow-up (QC weights, season retry, GOF) -----
#
# These cover the three phenofit-parity gaps closed on top of the C++ core:
# 1. Per-observation QC/reliability weights feeding the Whittaker/HANTS
#    smoothers and the iterative curve-fit reweighting (mirrors phenofit's
#    check_input(..., w=...) + qcFUN.R).
# 2. A single relax-and-retry of the season-boundary trough threshold when
#    the first pass finds no season at all (mirrors phenofit's season_mov
#    r_max relaxation).
# 3. Per-season R2/RMSE goodness-of-fit metrics (mirrors phenofit's
#    get_GOF()), now metrics 19 and 20 of fit_phenology_batch's output.

from zeit._core.phenology import fit_phenology_batch as real_fit_phenology_batch
from zeit._core.phenology import debug_split_seasons, CurveType


def _synthetic_evi_curve(time_steps=46):
    dates = np.array([t * 16 + (t // 23) * 365 for t in range(time_steps)], dtype=np.float64)
    doy = dates % 365
    growth = (np.sin((doy / 365.0) * np.pi - np.pi / 2) + 1) / 2.0
    clean = 0.2 + growth * 0.6
    return dates, clean


def test_gof_metrics_present_and_reasonable_for_a_clean_fit():
    dates, clean = _synthetic_evi_curve()
    out = real_fit_phenology_batch(
        values_array=clean.reshape(1, -1), dates_array=dates,
        curve_type=int(CurveType.BECK), extraction_method=0,
        max_seasons=2, min_season_length=3, min_amplitude=0.05
    )
    assert out.shape[0] == 21
    r2, rmse = out[19, 0, 0], out[20, 0, 0]
    assert np.isfinite(r2) and np.isfinite(rmse)
    assert r2 > 0.8  # near-perfect double-logistic-shaped input
    assert rmse < 0.1


def test_gof_metrics_are_nan_when_no_season_converges():
    dates, _ = _synthetic_evi_curve()
    flat = np.full((1, dates.size), 0.3)
    out = real_fit_phenology_batch(
        values_array=flat, dates_array=dates,
        curve_type=int(CurveType.BECK), extraction_method=0,
        max_seasons=2, min_season_length=3, min_amplitude=0.05
    )
    assert np.all(np.isnan(out[19]))
    assert np.all(np.isnan(out[20]))


def test_weights_array_downweighting_outlier_improves_fit_quality():
    dates, clean = _synthetic_evi_curve()
    noisy = clean.copy()
    spike_idx = 5  # simulated cloud/snow contamination on the rising limb
    noisy[spike_idx] = 0.98
    vals = noisy.reshape(1, -1)

    kwargs = dict(
        dates_array=dates, curve_type=int(CurveType.BECK), extraction_method=0,
        max_seasons=2, min_season_length=3, min_amplitude=0.05
    )
    out_unweighted = real_fit_phenology_batch(values_array=vals, **kwargs)

    w = np.ones_like(vals)
    w[0, spike_idx] = 0.02  # QC flags this single observation as unreliable
    out_weighted = real_fit_phenology_batch(values_array=vals, weights_array=w, **kwargs)

    r2_unweighted, rmse_unweighted = out_unweighted[19, 0, 0], out_unweighted[20, 0, 0]
    r2_weighted, rmse_weighted = out_weighted[19, 0, 0], out_weighted[20, 0, 0]

    assert np.isfinite(r2_weighted) and np.isfinite(rmse_weighted)
    # Down-weighting the contaminated observation should give a fit that
    # tracks the (mostly clean) rest of the series more closely.
    assert r2_weighted > r2_unweighted
    assert rmse_weighted < rmse_unweighted


def test_weights_array_shape_mismatch_raises():
    dates, clean = _synthetic_evi_curve()
    bad_weights = np.ones((1, clean.size - 1))  # wrong time length
    with pytest.raises(RuntimeError):
        real_fit_phenology_batch(
            values_array=clean.reshape(1, -1), dates_array=dates,
            curve_type=int(CurveType.BECK), weights_array=bad_weights
        )


def test_season_retry_recovers_a_season_a_strict_pass_misses():
    # Shallow troughs (0.75) flanking a strong interior peak (1.0), with the
    # true series minimum (0.0) sitting where it never registers as a local
    # extremum. At the default rtrough_max=0.6 these troughs get rejected
    # (0.75 - 0.0 = 0.75 > 0.6 * amplitude), so a strict single pass finds no
    # season; season_retry relaxes the threshold once and recovers one.
    y = [1.0] * 3 + [0.0, 1.0, 0.75, 1.0, 0.75, 1.0, 0.0] + [1.0] * 3

    strict = debug_split_seasons(y, min_season_length=0, min_amplitude=0.0,
                                  rtrough_max=0.6, r_min_filter=0.02, retry_on_empty=False)
    retried = debug_split_seasons(y, min_season_length=0, min_amplitude=0.0,
                                   rtrough_max=0.6, r_min_filter=0.02, retry_on_empty=True)

    assert strict == []
    assert len(retried) >= 1


def test_season_retry_is_a_noop_when_seasons_are_already_found():
    dates, clean = _synthetic_evi_curve()
    # A well-behaved single-season curve should already be found on the
    # first, strict pass - retry must not change that result.
    strict = debug_split_seasons(list(clean), min_season_length=3, min_amplitude=0.05,
                                  rtrough_max=0.6, r_min_filter=0.02, retry_on_empty=False)
    retried = debug_split_seasons(list(clean), min_season_length=3, min_amplitude=0.05,
                                   rtrough_max=0.6, r_min_filter=0.02, retry_on_empty=True)
    assert strict == retried
    assert len(strict) >= 1


def test_run_phenology_dask_passes_weights_and_season_retry_through():
    time_steps = 20
    rows, cols = 4, 4

    captured = {}

    def capturing_mock(values_array, dates_array, curve_type, extraction_method, max_seasons,
                        whittaker_lambda, apply_whittaker, apply_hants, hants_frequencies,
                        hants_threshold, min_season_length, min_amplitude, min_pixel_amplitude,
                        n_jobs, weights_array=None, season_retry=True):
        captured['weights_array'] = weights_array
        captured['season_retry'] = season_retry
        n_pixels = values_array.shape[0]
        return np.zeros((21, n_pixels, max_seasons), dtype=np.float32)

    data = da.random.random((time_steps, rows, cols), chunks=(time_steps, rows, cols))
    weights = da.ones((time_steps, rows, cols), chunks=(time_steps, rows, cols))
    dates = np.arange(time_steps)

    with patch('zeit._phenology.fit_phenology_batch', side_effect=capturing_mock):
        result = run_phenology_dask(
            data, dates, curve_type=0, max_seasons=2,
            weights=weights, season_retry=False
        )
        result.compute()

    assert captured['season_retry'] is False
    assert captured['weights_array'] is not None
    assert captured['weights_array'].shape == (rows * cols, time_steps)


def test_min_season_length_is_calendar_days_not_observation_count():
    # Regression test: min_season_length is documented (README, tutorials) as
    # a calendar-day threshold, e.g. min_season_length=45 for MODIS 16-day
    # composites. It used to be compared against the *index* distance between
    # a season's start/end instead, so 45 meant ~45 observations = ~720 days
    # and rejected every normal annual growing season.
    n_years = 5
    dates = 1.0 + 16.0 * np.arange(23 * n_years)
    growth = 0.2 + 0.3 * np.maximum(0.0, np.sin(2 * np.pi * (dates - 80.0) / 365.0))
    values = growth[None, :]

    kwargs = dict(
        values_array=values, dates_array=dates,
        curve_type=int(CurveType.BECK), extraction_method=0,
        max_seasons=n_years, apply_whittaker=True, whittaker_lambda=5.0,
        min_amplitude=0.1, min_pixel_amplitude=0.1,
    )

    out_unfiltered = real_fit_phenology_batch(min_season_length=0, **kwargs)
    out_45_days = real_fit_phenology_batch(min_season_length=45, **kwargs)
    # A real ~180-day season must survive a genuinely-45-day threshold.
    assert np.isfinite(out_45_days).sum() == np.isfinite(out_unfiltered).sum()
    assert np.isfinite(out_45_days).sum() > 0

    # A threshold longer than any single season's real duration must still
    # reject everything, proving min_season_length is genuinely honored.
    out_too_long = real_fit_phenology_batch(min_season_length=720, **kwargs)
    assert np.all(np.isnan(out_too_long))
