import numpy as np
import pytest
from zeit import desawtooth
from zeit._landtrendr import run_landtrendr
from zeit.raster import run_landtrendr_array

def test_desawtooth_removes_spike():
    values = np.array([0.1, 0.12, 0.9, 0.15, 0.11, 0.13])
    filtered = desawtooth(values)
    assert filtered[2] < 0.9
    assert filtered[2] < 0.4
    assert np.isclose(filtered[0], 0.1)
    assert np.isclose(filtered[-1], 0.13)

def test_desawtooth_no_spike():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    filtered = desawtooth(values)
    np.testing.assert_allclose(values, filtered, atol=0.01)

def test_run_landtrendr_basic():
    # Only 10 observations: no candidate model is significant at p < 0.05 with
    # the original's 2*(vertices)-2 regression degrees of freedom, so -- exactly
    # like the original IDL (verified under GDL on this input) -- the result is
    # a flat line at the mean of the desawtoothed series over the whole range.
    years = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009])
    values = np.array([0.9, 0.9, 0.85, 0.2, 0.3, 0.4, 0.5, 0.85, 0.9, 0.9])
    vertices = run_landtrendr(years, values, max_segments=2, best_model_proportion=0.75,
                               recovery_threshold=1.0)
    assert [v['year'] for v in vertices] == [2000, 2009]
    assert np.isclose(vertices[0]['value'], 0.67577, atol=1e-4)
    assert vertices[0]['value'] == vertices[1]['value']

def test_run_landtrendr_array():
    # Piecewise-linear V (true kink at 2006), 15 observations (rather than a
    # handful) so degrees of freedom, find_vertices' candidate budget, and the
    # recovery-threshold check all have enough room; best_model_proportion=0.75
    # asks for the more detailed candidate instead of the simpler compromise
    # fit best_model_proportion's default would pick (see
    # test_run_landtrendr_basic). desawtooth always nudges the single sharpest
    # point in the series by at least a little (see desawtooth()), even here
    # where that point is a real kink rather than noise -- exactly what the
    # real LandTrendr algorithm does -- so the selected middle vertex lands one
    # year off the true kink, at 2007, not 2006.
    years = np.arange(2000, 2015)
    stack = np.zeros((15, 2, 2))
    stack[:, 0, 0] = list(np.linspace(1.0, 0.2, 7)) + list(np.linspace(0.2, 0.6, 9))[1:]
    stack[:, 1, 1] = 0

    output = run_landtrendr_array(years, stack, max_segments=2, n_jobs=2,
                                   min_observations_needed=15, best_model_proportion=0.75)

    assert output.shape == (6, 2, 2)
    assert output[0, 0, 0] == 2000
    assert output[1, 0, 0] == 2007
    assert output[2, 0, 0] == 2014
    assert np.all(output[:, 1, 1] == 0)

def test_run_landtrendr_batch():
    from zeit._landtrendr import run_landtrendr_batch

    # Same V as test_run_landtrendr_array, see its comment for why 15
    # observations, an explicit best_model_proportion, and a middle vertex at
    # 2007 (not the true kink year 2006) are used.
    years = np.arange(2000, 2015)

    # values shape [Y, X, Time]
    values = np.zeros((1, 2, 15))
    values[0, 0, :] = list(np.linspace(1.0, 0.2, 7)) + list(np.linspace(0.2, 0.6, 9))[1:]
    # Pixel 1: No data
    values[0, 1, :] = [-9999.0] * 15

    max_segments = 2
    verts, counts, rmse = run_landtrendr_batch(years, values, max_segments=max_segments,
                                                min_observations_needed=15, best_model_proportion=0.75)

    assert counts.shape == (2,)
    assert verts.shape == (2, max_segments + 1, 2)
    assert rmse.shape == (2,)

    # Pixel 0 should have 3 vertices (start, break, end). RMSE is small but
    # non-zero: desawtooth's mandatory single-point nudge (see desawtooth())
    # means the fit -- made to the desawtoothed series, as in the original --
    # no longer reproduces the raw observations exactly.
    assert counts[0] == 3
    assert verts[0, 0, 0] == 2000
    assert verts[0, 1, 0] == 2007
    assert verts[0, 2, 0] == 2014
    assert np.isclose(rmse[0], 0.0522, atol=1e-3)

    # Pixel 1 should have 0 vertices and no RMSE (fitting was skipped, no-data).
    assert counts[1] == 0
    assert rmse[1] == 0.0

def test_run_landtrendr_min_observations_needed():
    # Below min_observations_needed, fitting is skipped entirely and the raw
    # (unsegmented) trajectory is returned as-is.
    years = np.array([2000, 2001, 2002, 2003])
    values = np.array([0.8, 0.5, 0.2, 0.6])

    vertices = run_landtrendr(years, values, max_segments=2, min_observations_needed=6)
    assert len(vertices) == 4
    assert [v['year'] for v in vertices] == [2000, 2001, 2002, 2003]
    assert [v['value'] for v in vertices] == list(values)

    # With the gate lowered, normal segmentation proceeds instead.
    vertices = run_landtrendr(years, values, max_segments=2, min_observations_needed=2)
    assert len(vertices) <= 3

def test_run_landtrendr_sequential_fit_smooths_noisy_segment():
    # A noisy-but-stable segment (2000-2005, endpoints both 0.50) followed by a
    # drop to a new stable level. The per-segment regression-vs-point-to-point
    # choice (Kennedy et al. 2010 Sec. 2.5.3; find_best_trace in tbcd_v2.pro)
    # should pick the regression line for the first segment, since it has lower
    # SSE against the noisy interior points than the flat point-to-point line
    # through the two (coincidentally equal) endpoint values -- so the fitted
    # endpoint values are NOT the raw 0.50/0.50. The selected vertices match the
    # original IDL run under GDL on this input. modifier=-1.0 orients the
    # segmentation for a value DROP (see run_landtrendr's modifier docstring).
    years = np.arange(2000, 2014)
    values = np.array([0.50, 0.54, 0.46, 0.53, 0.47, 0.50, 0.30,
                       0.10, 0.11, 0.09, 0.10, 0.11, 0.09, 0.10])

    vertices = run_landtrendr(years, values, max_segments=3, min_observations_needed=6,
                               best_model_proportion=0.75, modifier=-1.0)
    by_year = {v['year']: v['value'] for v in vertices}

    assert list(by_year) == [2000, 2005, 2007, 2013]
    assert not np.isclose(by_year[2000], 0.50, atol=1e-6)
    assert not np.isclose(by_year[2005], 0.50, atol=1e-6)
    assert np.isclose(by_year[2000], 0.5256, atol=1e-3)
    assert np.isclose(by_year[2005], 0.4962, atol=1e-3)

def test_run_landtrendr_array_return_rmse():
    # Same V as test_run_landtrendr_array (see its comment) -> RMSE of the
    # selected fit is small but non-zero, since desawtooth's mandatory
    # single-point nudge means even this otherwise piecewise-linear series is
    # no longer fit exactly.
    years = np.arange(2000, 2015)
    stack = np.zeros((15, 2, 2))
    stack[:, 0, 0] = list(np.linspace(1.0, 0.2, 7)) + list(np.linspace(0.2, 0.6, 9))[1:]
    stack[:, 1, 1] = 0

    output, rmse_map = run_landtrendr_array(years, stack, max_segments=2, min_observations_needed=15,
                                             best_model_proportion=0.75, return_rmse=True)

    assert output.shape == (6, 2, 2)
    assert rmse_map.shape == (2, 2)
    assert np.isclose(rmse_map[0, 0], 0.0522, atol=1e-3)
    assert rmse_map[1, 1] == 0.0  # no-data pixel: fitting was skipped

def test_extract_events_dsnr():
    from zeit.metrics import extract_events

    years = np.array([2000, 2001, 2002, 2003, 2004, 2005, 2006])
    stack = np.zeros((7, 1, 1))
    # A disturbance (drop) at 2003 with some noise elsewhere, so the fit RMSE
    # is a real, non-zero value the DSNR calculation actually normalizes by.
    stack[:, 0, 0] = [0.82, 0.79, 0.81, 0.30, 0.32, 0.29, 0.31]

    vertices_stack, rmse_map = run_landtrendr_array(
        years, stack, max_segments=2, min_observations_needed=7, return_rmse=True
    )

    with pytest.raises(ValueError):
        extract_events(vertices_stack, sort_by="dsnr")  # no rmse_map given

    events = extract_events(vertices_stack, event_type="loss", sort_by="dsnr", rmse_map=rmse_map)
    assert "dsnr" in events
    assert rmse_map[0, 0] > 0.0
    expected_dsnr = events["magnitude"][0, 0] / rmse_map[0, 0]
    assert np.isclose(events["dsnr"][0, 0], expected_dsnr, atol=1e-4)

def test_run_landtrendr_vertex_count_overshoot():
    # A longer, exactly piecewise-linear 3-segment (4-vertex) trajectory.
    # With vertex_count_overshoot=0, the initial angle-culling has to pick the
    # true vertices directly out of every observation; overshoot just gives it
    # more candidates to choose from before pruning back down. Either way the
    # same true vertices should be recoverable, since they're the only ones
    # that make the fit exact (SSE=0).
    years = np.arange(2000, 2013)  # 13 points
    values = np.array([1.0, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 0.6, 0.2, 0.6, 1.0, 1.0, 1.0])

    for vco in (0, 3, 6):
        vertices = run_landtrendr(years, values, max_segments=6, vertex_count_overshoot=vco, min_observations_needed=6)
        years_out = [v['year'] for v in vertices]
        assert years_out[0] == 2000
        assert years_out[-1] == 2012

def test_run_landtrendr_batch_partial_no_data_is_skipped_not_fitted():
    # Years holding no_data_value are left out of the fit (the original's
    # `goods`), so a pixel with a few no-data years gets exactly the same
    # result as the single-pixel API given NaN for those years -- the -9999
    # itself must never be fitted as if it were a real observation.
    from zeit._landtrendr import run_landtrendr_batch

    years = np.arange(1990, 2020)
    rng = np.random.default_rng(0)
    series = np.full(30, 700.0) + rng.normal(0, 10, 30)
    series[12:] -= 300.0
    missing = [0, 7, 20, 29]

    with_nodata = series.copy()
    with_nodata[missing] = -9999.0
    with_nan = series.copy()
    with_nan[missing] = np.nan

    verts, counts, _ = run_landtrendr_batch(years, with_nodata[None, None, :],
                                            best_model_proportion=0.75, modifier=-1.0)
    expected = run_landtrendr(years, with_nan, best_model_proportion=0.75, modifier=-1.0)

    assert counts[0] == len(expected)
    assert [int(y) for y in verts[0, :counts[0], 0]] == [v["year"] for v in expected]
    np.testing.assert_allclose(verts[0, :counts[0], 1], [v["value"] for v in expected])
    assert expected[0]["year"] == 1990 and expected[-1]["year"] == 2019
    assert min(v["value"] for v in expected) > 0  # no -9999 leaking into the fit

def test_run_landtrendr_default_best_model_proportion_matches_original():
    # A real Landsat NDVI (x 10000) pixel from Rondonia: forest until 2005,
    # cleared in 2006, then a noisy pasture/regrowth cycle. With the original's
    # default bestmodelproportion = 0.75, pick_best_model6 accepts models within
    # (2 - 0.75) = 1.25x of the lowest p-value, and the original IDL (run under
    # GDL on exactly this input) returns the vertices below. A default above 1
    # makes that threshold stricter than the best model itself, so no model is
    # picked and the pixel collapses to a flat line (the 1.25 default of
    # releases 0.18.0-0.23.0 did that for most pixels of this scene).
    years = np.arange(1985, 2025)
    values = np.array([7821, 7792, 7286, 7261, 8148, 7768, 7222, 7246, 7398, 8217,
                       8064, 8090, 7630, 8260, 7615, 8377, 8530, 8648, 8374, 7862,
                       7802, 4038, 4138, 6627, 7140, 7497, 5716, 7846, 6370, 7075,
                       4525, 6459, 6953, 6931, 6497, 5450, 5855, 6092, 6260, 5590], dtype=float)
    vertices = run_landtrendr(years, values, modifier=-1.0)
    assert [v['year'] for v in vertices] == [1985, 2005, 2006, 2012, 2017, 2024]
    idl_values = [-7492.0, -8201.0, -4038.0, -8510.0, -5513.0, -6136.0]
    np.testing.assert_allclose([np.trunc(v['value'] * -1.0) for v in vertices], idl_values, atol=1.0)

    flat = run_landtrendr(years, values, modifier=-1.0, best_model_proportion=1.25)
    assert [v['year'] for v in flat] == [1985, 2024]
