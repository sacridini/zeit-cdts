"""The error profile by the DI (zeit.error_profile) against CAST 1.0.4's errorProfiles
(tests/data/make_error_profile_cast_fixtures.R), and from a model of zeit.train_classifier to a map."""
import json
from pathlib import Path

import numpy as np
import pytest

import zeit
from zeit._aoa import train_di

DATA = Path(__file__).parent / "data"


def _cases():
    for case in json.loads((DATA / "error_profile_cast.json").read_text())["cases"]:
        observed, predicted = np.array(case["observed"]), np.array(case["predicted"])
        if not case["classes"]:
            observed, predicted = observed.astype(float), predicted.astype(float)
        yield case, observed, predicted


def _profile(case, observed, predicted, **kw):
    return zeit.error_profile(train=np.array(case["x"]), observed=observed, predicted=predicted,
                              folds=case["folds"], weights=case["weight"], **kw)


def test_the_windows_are_casts():
    for case, observed, predicted in _cases():
        ep = _profile(case, observed, predicted)
        assert ep.threshold == pytest.approx(case["threshold"], rel=1e-12)
        assert ep.metric == ("accuracy" if case["classes"] else "rmse")
        np.testing.assert_allclose(ep.points["DI"], case["window_di"], rtol=1e-12)
        ours, cast = ep.points["metric"].to_numpy(), np.array(case["window_metric"])
        if not case["classes"]:
            np.testing.assert_allclose(ours, cast, rtol=1e-12)
            continue
        # CAST gives 0 to a window whose five points are of one class and all predicted right; zeit 1
        td = train_di(np.array(case["x"]), weights=case["weight"], folds=case["folds"])
        known = td.train_di <= td.threshold
        order = np.flatnonzero(known)[np.argsort(td.train_di[known], kind="stable")]
        o, p = observed[order], predicted[order]
        single = np.array([len(set(o[i:i + 5])) == 1 and (o[i:i + 5] == p[i:i + 5]).all()
                           for i in range(len(ours))])
        assert single.sum() > 5
        assert (ours[single] == 1).all() and (cast[single] == 0).all()
        np.testing.assert_allclose(ours[~single], cast[~single], rtol=1e-12)


def test_the_curve_is_like_scams_and_monotone():
    case, observed, predicted = next(_cases())
    ep = _profile(case, observed, predicted)
    grid = np.array(case["grid"])
    # another basis and criterion: alike, within a few percent of the metric's range
    spread = np.ptp(case["window_metric"])
    assert np.abs(ep(grid) - np.array(case["curve"])).max() < 0.05 * spread
    wide = np.linspace(-0.1, 1.0, 500)
    assert (np.diff(ep(wide)) >= -1e-12).all()
    for _, obs, pred in list(_cases())[1:]:
        acc = zeit.error_profile(train=np.array(case["x"]), observed=obs, predicted=pred, folds=case["folds"])
        assert (np.diff(acc(wide)) <= 1e-12).all()
    lin = _profile(case, observed, predicted, calibration="linear")
    slope, intercept = np.polyfit(ep.points["DI"], ep.points["metric"], 1)
    assert lin(0.1) == pytest.approx(intercept + 0.1 * slope)
    assert "ErrorProfile(rmse by the DI: 140 windows of 5 points, monotone" in repr(ep)
    with pytest.raises(ValueError, match="metric"):
        _profile(case, observed, predicted, metric="r2")


def test_save_load_and_predict_points(tmp_path):
    case, observed, predicted = next(_cases())
    ep = _profile(case, observed, predicted)
    back = zeit.ErrorProfile.load(ep.save(tmp_path / "profile.json"))
    grid = np.linspace(0, 0.3, 30)
    np.testing.assert_array_equal(back(grid), ep(grid))
    assert back.points.equals(ep.points) and repr(back) == repr(ep)
    new = np.array(case["x"])[:40] + 0.3
    a = zeit.aoa(new, zeit.train_di(np.array(case["x"]), weights=case["weight"], folds=case["folds"]))
    expected = ep.predict(a)
    assert expected.name == "expected_rmse"
    inside = a["AOA"] == 1
    np.testing.assert_allclose(expected[inside], ep(a["DI"][inside].to_numpy()))
    assert expected[~inside].isna().all()
    assert ep.predict(a, outside=True).notna().all()


def test_from_train_classifier_to_a_map():
    pytest.importorskip("sklearn")
    import geopandas as gpd
    import xarray as xr
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(2)
    n = 60
    b1 = np.cumsum(rng.normal(size=(n, n)), axis=1) / 5
    b2 = np.cumsum(rng.normal(size=(n, n)), axis=0) / 5
    stack = xr.DataArray(np.stack([b1, b2]), dims=("band", "y", "x"),
                         coords={"band": ["b1", "b2"], "y": 10_000.0 - 30 * np.arange(n),
                                 "x": 500_000.0 + 30 * np.arange(n)}).rio.write_crs("EPSG:32722")
    rows, cols = rng.integers(0, n, 150), rng.integers(0, n // 2, 150)
    noisy = b1[rows, cols] + rng.normal(scale=0.5, size=150)
    samples = gpd.GeoDataFrame({"cls": np.where(noisy > np.median(noisy), "a", "b")},
                               geometry=gpd.points_from_xy(stack.x.values[cols], stack.y.values[rows]),
                               crs="EPSG:32722")
    rf = zeit.train_classifier(stack, samples, label="cls", folds="clusters",
                               model=RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=2))
    ep = zeit.error_profile(rf)
    assert ep.metric == "accuracy" and len(ep.points) > 50
    applicable = zeit.aoa(stack, model=rf)
    assert applicable.attrs["threshold"] == pytest.approx(ep.threshold)
    expected = ep.predict(applicable)
    assert expected.dims == ("y", "x") and expected.rio.crs.to_epsg() == 32722
    values = expected.values[applicable["AOA"].values == 1]
    assert np.isfinite(values).all() and (values <= 1 + 1e-9).all()
    with pytest.raises(ValueError, match="no cross-validation"):
        zeit.error_profile(zeit.train_classifier(stack, samples, label="cls",
                                                 model=RandomForestClassifier(n_estimators=5, n_jobs=2)))
