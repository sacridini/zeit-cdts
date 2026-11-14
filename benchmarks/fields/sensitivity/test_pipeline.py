"""Stage 1 + stage 2 of the harness equals ``zeit.fields`` for the same parameters (run: pytest this file)."""
import pyexpat  # noqa: F401
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import zeit

import cases as C
import pipeline as P


@pytest.fixture(scope="module")
def loaded():
    return {c.id: c for c in C.load_cases(only={"trazo_chaco_005", "trazo_humidchaco_007", "sorriso_1"})}


CONFIGS = [
    (P.S1(realizations=30, seed=3), P.S2()),
    (P.S1(realizations=25, seed=4, markers=2000, min_distance=10, marker_radius=1, coloring="unique",
          mask_threshold=1500), P.S2(low=0.2, high=0.5, min_size=25, final="gradient")),
    (P.S1(gradient="sobel", realizations=20, seed=1, mask_threshold=None), P.S2(low=0.4, high=0.7, min_size=0)),
    (P.S1(gradient="multitemporal", realizations=20, seed=2), P.S2(final="gradient")),
    (P.S1(method="exact", markers=1000), P.S2(low=0.1, high=0.4)),
]


@pytest.mark.parametrize("case_id", ["trazo_chaco_005:trazo", "trazo_humidchaco_007:trazo", "sorriso_1:pivots"])
@pytest.mark.parametrize("s1,s2", CONFIGS, ids=[s.key() for s, _ in CONFIGS])
def test_equals_zeit_fields(loaded, case_id, s1, s2):
    case = loaded[case_id]
    b = P.stage1(case, s1, use_cache=False)
    lab, bl = P.stage2(case, s1, b, s2)
    data = case.stack if s1.gradient == "multitemporal" else case.img
    res = zeit.fields(
        data, gradient=s1.gradient, method=s1.method, realizations=s1.realizations, markers=s1.markers,
        min_distance=s1.min_distance, marker_radius=s1.marker_radius, coloring=s1.coloring, low=s2.low,
        high=s2.high, min_size=s2.min_size, final=s2.final, mask=case.keep(s1.mask_threshold),
        random_state=s1.seed, n_jobs=4)
    np.testing.assert_array_equal(res["labels"].values, lab)
    assert int(res.attrs["boundary_label"]) == bl


@pytest.mark.parametrize("case_id", ["trazo_chaco_005:trazo", "trazo_humidchaco_007:trazo", "sorriso_1:pivots"])
def test_matched_without_clip_equals_package(loaded, case_id):
    import matched as M

    case = loaded[case_id]
    s1 = P.S1(realizations=20, seed=0)
    b = P.stage1(case, s1, use_cache=False)
    lab, bl = P.stage2(case, s1, b, P.S2())
    ev = P.evaluate(case, lab, bl)
    pred = np.where(lab == bl, 0, lab).astype(np.int64) if bl else lab.astype(np.int64)
    ids, one, _ = M.match(pred, case.ref_labels, case.regions[1], clip=False)
    assert np.isclose((one >= 0.5).mean(), ev["matched"])
    assert ev["matched_clip"] >= 0 and "matched_in" in ev
