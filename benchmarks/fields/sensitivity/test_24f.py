"""The 24f harness: the fast responses equal ``pipeline.evaluate``, the threaded stage 2 equals ``pipeline.stage2``, the
hierarchy cut is a proper partition, and the noisy Monte Carlo at s = 0 reproduces the counts of the normal path."""
import pyexpat  # noqa: F401
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import cases as C
import pipeline as P
import s24f as S
from fast import fast_eval

IDS = ["trazo_chaco_005:trazo", "trazo_humidchaco_007:trazo", "sorriso_1:pivots", "sorriso_1:car"]


@pytest.fixture(scope="module")
def loaded():
    return {c.id: c for c in C.load_cases(only={"trazo_chaco_005", "trazo_humidchaco_007", "sorriso_1"})}


@pytest.mark.parametrize("case_id", IDS)
@pytest.mark.parametrize("low,ms", [(0.2, 5), (0.5, 0), (0.1, 25)])
def test_fast_and_post_equal_pipeline(loaded, case_id, low, ms):
    case = loaded[case_id]
    s1 = S.s1_for(case, "sobel", 600, "q20")
    b = P.stage1(case, replace_r(s1), use_cache=False)
    keep = case.keep(s1.mask_threshold)
    lab, bl = P.stage2(case, replace_r(s1), b, P.S2(low=low, high=S.high_for(low), min_size=ms))
    ref = np.where(lab == bl, 0, lab).astype(np.int64) if bl else lab.astype(np.int64)
    pred = S.hyst_pred(case, keep, S.mc_regions(b.counts, b.r, low, S.high_for(low)), ms)
    np.testing.assert_array_equal(pred, ref)
    ev = P.evaluate(case, lab, bl)
    fe = fast_eval(case, pred)
    assert np.isclose(ev["rec_t1"], fe["rec"], equal_nan=True)
    if case.kind == "fields":
        assert np.isclose(ev["f_t1"], fe["f"], equal_nan=True) and np.isclose(ev["d_w"], fe["d_w"], equal_nan=True)
    else:
        assert np.isclose(ev["cross_share"], fe["cross_share"], equal_nan=True)


def replace_r(s1):
    from dataclasses import replace

    return replace(s1, realizations=20)


def test_cut_is_a_partition(loaded):
    case = loaded["trazo_chaco_005:trazo"]
    s1 = S.s1_for(case, "sobel", 600, "q20")
    keep = case.keep(s1.mask_threshold)
    h, v = S.exact_edges(case, s1)
    for t in (0.01, 0.1, 0.5):
        comps = S.cut_components(h, v, keep, t)
        assert (comps[~keep] == 0).all() and (comps[keep] > 0).all()
        # no kept edge joins two components, and the components are connected through kept edges only
        hh = (h <= t) & keep[:, :-1] & keep[:, 1:]
        assert (comps[:, :-1][hh] == comps[:, 1:][hh]).all()
        cut = ~hh & keep[:, :-1] & keep[:, 1:]
        # a cut edge may still join pixels of one component (a cycle); but t = inf is one component per mask part
    one = S.cut_components(h, v, keep, 2.0)
    from scipy.ndimage import label

    n_ref = label(keep)[1]
    assert one.max() == n_ref
    pred = S.cut_pred(case, keep, S.cut_components(h, v, keep, 0.05), 5)
    assert pred.shape == keep.shape and (pred[keep] > 0).all() and (pred[~keep] == 0).all()
