"""The area of applicability (Meyer & Pebesma 2021): zeit.train_di and zeit.aoa against CAST 1.0.4's trainDI and
aoa (tests/data/make_aoa_cast_fixtures.R), on tables and on maps."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

DATA = Path(__file__).parent / "data"


def _cases():
    ref = json.loads((DATA / "aoa_cast.json").read_text())
    for case in ref["cases"]:
        p = len(case["weight"])
        names = [f"v{j + 1}" for j in range(p)]
        train = pd.DataFrame(np.array(case["train"], dtype=float), columns=names)
        new = pd.DataFrame(np.array([[np.nan if v is None else v for v in row] for row in case["newdata"]]),
                           columns=names)
        yield case, train, new


def _nan(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def test_train_di_equals_cast():
    for case, train, _ in _cases():
        td = zeit.train_di(train, weights=case["weight"], folds=case["folds"], n_jobs=2)
        assert td.features == list(train.columns)
        assert td.threshold == pytest.approx(case["threshold"], rel=1e-13)
        assert td.mean_distance == pytest.approx(case["mean_distance"], rel=1e-13)
        np.testing.assert_allclose(td.train_di, _nan(case["train_di"]), rtol=1e-12, atol=0)


def test_aoa_of_points_equals_cast():
    for case, train, new in _cases():
        out = zeit.aoa(new, train, weights=case["weight"], folds=case["folds"], lpd=True, n_jobs=2)
        di = _nan(case["di"])
        np.testing.assert_allclose(out["DI"].to_numpy(), di, rtol=1e-12, atol=1e-14)
        known = ~np.isnan(di)
        # CAST calls a point with a missing predictor inside the AOA; zeit leaves it missing
        assert np.isnan(out["AOA"].to_numpy()[~known]).all()
        np.testing.assert_array_equal(out["AOA"].to_numpy()[known], np.array(case["aoa"], dtype=float)[known])
        np.testing.assert_array_equal(out["LPD"].to_numpy()[known], _nan(case["lpd"])[known])
        assert out.attrs["threshold"] == pytest.approx(case["threshold"], rel=1e-13)


def test_trained_di_reused_and_arrays():
    case, train, new = next(_cases())
    td = zeit.train_di(train.to_numpy(), n_jobs=1)
    assert td.features == ["v1", "v2", "v3", "v4"]
    a = zeit.aoa(new.to_numpy(), td)
    b = zeit.aoa(new, train)
    np.testing.assert_allclose(a["DI"].to_numpy(), b["DI"].to_numpy(), rtol=0, atol=0, equal_nan=True)
    assert "LPD" not in a
    assert "TrainDI(80 training points, 4 predictors" in repr(td)


def test_folds_labels_and_missing_points():
    case = json.loads((DATA / "aoa_cast.json").read_text())["cases"][1]
    train = np.array(case["train"], dtype=float)
    folds = np.array([f"site{g}" for g in case["folds"]], dtype=object)
    td = zeit.train_di(train, weights=case["weight"], folds=folds)
    assert td.threshold == pytest.approx(case["threshold"], rel=1e-13)
    # a point without a fold has no DI of its own but still counts as training data
    folds[0] = None
    td0 = zeit.train_di(train, weights=case["weight"], folds=folds)
    assert np.isnan(td0.train_di[0]) and td0.train.shape == train.shape
    # a training point with a missing predictor is left out
    holed = train.copy()
    holed[3, 1] = np.nan
    assert zeit.train_di(holed).train.shape == (len(train) - 1, train.shape[1])
    with pytest.raises(ValueError, match="folds"):
        zeit.train_di(train, folds=[1, 2])
    with pytest.raises(ValueError, match="weights"):
        zeit.train_di(train, weights=[1.0])


def test_weights_from_importance_and_zero_weight():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))

    class Model:
        feature_importances_ = np.array([0.5, 0.5, 0.0])
        zeit_train_ = x
        zeit_features_ = ["a", "b", "c"]

    td = zeit.train_di(model=Model())
    assert td.features == ["a", "b", "c"]
    np.testing.assert_allclose(td.weight, [0.5, 0.5, 0.0])
    # a predictor of weight 0 does not count, even missing
    new = np.array([[0.0, 0.0, 1e6], [0.0, 0.0, np.nan]])
    out = zeit.aoa(new, td)
    assert np.isfinite(out["DI"]).all() and out["DI"][0] == out["DI"][1]
    same = zeit.train_di(x[:, :2], weights=[0.5, 0.5])
    assert td.threshold == pytest.approx(same.threshold, rel=1e-13)
    with pytest.raises(ValueError, match="feature_importances_"):
        zeit.train_di(x, weights="importance")


def test_aoa_of_a_map_after_train_classifier():
    pytest.importorskip("sklearn")
    import geopandas as gpd
    import rioxarray  # noqa: F401
    from shapely.geometry import Point

    rng = np.random.default_rng(1)
    ny, nx = 40, 50
    y = np.arange(ny) * -10.0 + 5000
    x = np.arange(nx) * 10.0 + 300000
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    # the left half (where the samples are) is like the training, the right half far from it
    b1 = np.where(xx < 25, rng.normal(0.3, 0.05, (ny, nx)), 3.0)
    b2 = np.where(xx < 25, rng.normal(0.6, 0.05, (ny, nx)), -2.0)
    b1[0, 0] = np.nan
    stack = xr.DataArray(np.stack([b1, b2]), dims=("band", "y", "x"),
                         coords={"band": ["red", "nir"], "y": y, "x": x}).rio.write_crs("EPSG:32722")
    pts = [(int(r), int(c)) for r, c in zip(rng.integers(1, ny, 60), rng.integers(0, 25, 60))]
    samples = gpd.GeoDataFrame({"class": [int(b1[r, c] > 0.3) for r, c in pts]},
                               geometry=[Point(x[c], y[r]) for r, c in pts], crs="EPSG:32722")
    rf = zeit.train_classifier(stack, samples)
    assert rf.zeit_train_.shape == (60, 2)
    out = zeit.aoa(stack, model=rf, lpd=True)
    assert set(out.data_vars) == {"DI", "AOA", "LPD"}
    assert out["DI"].dtype == np.float32 and out["AOA"].dtype == np.uint8
    assert out.rio.crs.to_epsg() == 32722
    assert out["AOA"].values[0, 0] == 255 and np.isnan(out["DI"].values[0, 0])
    assert (out["AOA"].values[:, 25:] == 0).all()
    assert (out["AOA"].values[1:, :25] == 1).mean() > 0.95
    assert (out["LPD"].values[:, 25:] == 0).all()
    np.testing.assert_allclose(out.attrs["threshold"], zeit.train_di(model=rf).threshold)
