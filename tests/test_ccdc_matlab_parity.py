"""Parity of zeit's CCDC against the original CCDC MATLAB code.

Every expected output in tests/data/ccdc_matlab_parity_expected.json was
produced by the original TrendSeasonalFit_v12_30Line.m (GERSL/CCDC, with its
bundled Fortran GLMnet) run under GNU Octave on exactly the inputs in
ccdc_matlab_parity_inputs.npz: Landsat-like pixels (6 optical bands as surface
reflectance x 10000, a constant brightness-temperature band in deg C x 100,
Fmask QA). The cases cover every branch of the original: mostly-cloudy (44) and
mostly-snow (54) pixels, 4/6/8-coefficient models, an early break found by
extending the first model backwards (14), the end-of-record fit (24), several
breaks, unflagged clouds removed by Tmask/Tmax_cg, and real Rondonia pixels.
"""
import json
import os

import numpy as np
import pytest

from zeit._ccdc import predict, run_ccdc

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = np.load(os.path.join(HERE, "data", "ccdc_matlab_parity_inputs.npz"))
with open(os.path.join(HERE, "data", "ccdc_matlab_parity_expected.json")) as f:
    EXPECTED = json.load(f)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_matches_original_matlab(name):
    dates = INPUTS[name + "__dates"]
    bands = INPUTS[name + "__bands"].astype(float)
    qa = INPUTS[name + "__qa"].astype(int)

    got = run_ccdc(dates, bands, qa, conseq_anom=6, chi2_prob_threshold=0.99,
                   detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4], thermal_band=6)
    want = EXPECTED[name]

    assert [(s["t_start"], s["t_end"], s["t_break"]) for s in got] == \
           [(s["t_start"], s["t_end"], s["t_break"]) for s in want]
    for g, w in zip(got, want):
        assert g["category"] == w["category"]
        assert g["num_obs"] == w["num_obs"]
        assert g["change_prob"] == pytest.approx(w["change_prob"], abs=1e-12)
        # the reference was exported with 10 significant digits
        np.testing.assert_allclose(g["coefs"], w["coefs"], rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(g["rmse"], w["rmse"], rtol=1e-8, atol=1e-6)
        np.testing.assert_allclose(g["magnitude"], w["magnitude"], rtol=1e-8, atol=1e-5)


def test_predict_matches_model_fit():
    # predict() evaluates the coefficients on the original's datenum time axis,
    # so the fitted model reproduces the stable observations it was fit to.
    name = "stable"
    dates = INPUTS[name + "__dates"]
    bands = INPUTS[name + "__bands"].astype(float)
    qa = INPUTS[name + "__qa"].astype(int)
    seg = run_ccdc(dates, bands, qa, detection_bands=[1, 2, 3, 4, 5], thermal_band=6)[0]
    clear = (qa < 2)
    resid = bands[3, clear] - predict(seg["coefs"][3], dates[clear])
    assert np.sqrt(np.mean(resid ** 2)) < 2 * seg["rmse"][3]


def test_batch_matches_single_pixel():
    # The OpenMP batch path must give exactly the single-pixel result; NaN in a
    # band marks a date without an observation (Fmask 255).
    from zeit._ccdc import run_ccdc_batch

    name_pairs = ["two_breaks", "unflagged_clouds", "mostly_snow_cat54"]  # same date grid
    dates = INPUTS[name_pairs[0] + "__dates"]
    values = np.stack([INPUTS[n + "__bands"].astype(float) for n in name_pairs])[None]  # (1, X, B, T)
    qa = np.stack([INPUTS[n + "__qa"].astype(int) for n in name_pairs])[None]
    values[0, 0, :, 5] = np.nan
    segs_arr, counts = run_ccdc_batch(dates, values, qa, max_segments=6, detection_bands=[1, 2, 3, 4, 5],
                                      tmask_bands=[1, 4], thermal_band=6, n_jobs=2)
    for x, n in enumerate(name_pairs):
        v = values[0, x].copy()
        q = qa[0, x].copy()
        missing = np.isnan(v).any(axis=0)
        v[:, missing] = 0.0
        q[missing] = 255
        single = run_ccdc(dates, v, q, detection_bands=[1, 2, 3, 4, 5], thermal_band=6)
        assert counts[x] == len(single)
        for i, s in enumerate(single):
            row = segs_arr[x, i]
            assert (row[0], row[1], row[2]) == (s["t_start"], s["t_end"], s["t_break"])
            assert np.allclose(row[4:12], s["coefs"][0])
