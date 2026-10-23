"""remotePARTS's methods (zeit.ar_trend, zeit.parts): the bindings against the R package's
fits (the fixtures of rust/zeit-core/tests, made by tests/data/make_remoteparts_fixtures.R),
and the API on a synthetic cube."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _core

FIXTURES = Path(__file__).resolve().parents[1] / "rust" / "zeit-core" / "tests" / "data"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text())


def _numbers(v):
    return np.atleast_1d(np.array([np.nan if e is None else e for e in np.atleast_1d(v)], dtype=float))


def _cm(v):
    """An R matrix or array written column-major."""
    return np.asarray(v["values"], dtype=float).reshape(v["dim"], order="F")


def test_fit_ar_batch_matches_fit_ar():
    for case in _fixture("remoteparts_ar.json")["ar"]:
        y = _numbers(case["y"])
        r = case["ignore"]
        if np.isnan(y).any() or abs(r["rho"]) > 0.999:
            continue
        design = np.column_stack([_numbers(c) for c in case["x"]])
        out = _core.parts.fit_ar_batch(y[None, :], design, gaps="ignore", n_jobs=1)[:, 0]
        q = design.shape[1]
        coef, se, p = out[:q], out[q:2 * q], out[3 * q:4 * q]
        assert abs(out[4 * q] - r["rho"]) < 1e-7, case["name"]
        np.testing.assert_allclose(np.abs(coef - _numbers(r["coefficients"])) / _numbers(r["se"]), 0, atol=1e-6)
        np.testing.assert_allclose(se, _numbers(r["se"]), rtol=1e-6)
        np.testing.assert_allclose(p, _numbers(r["p"]), rtol=3e-5)


def test_fit_correlation_and_parts_match_remoteparts():
    data = _fixture("remoteparts_parts.json")
    lng, lat, resp = (np.asarray(data[k], dtype=float) for k in ("lng", "lat", "y"))
    fc = data["fitcor"]
    resids = _cm(fc["resids"])
    n = resids.shape[0]
    fit = _core.parts.fit_correlation(resids, lng[:n], lat[:n], correlation="exponential", range=0.1, scaled=True)
    assert fit["range"] == pytest.approx(fc["exponential"], rel=1e-5)

    pm = np.asarray(data["partitions"], dtype=np.int64).reshape(-1, data["partsize"])
    case = next(c for c in data["parts"] if c["name"] == "classes_nugget0")
    design = np.column_stack([_numbers(c) for c in case["x"]])
    out = _core.parts.fit_parts(lng, lat, resp, design, np.ones((len(lng), 1)), pm, correlation="exponential",
                                range=0.1, scaled=True, nugget=0.0, ncross=6, n_jobs=-1)
    o = case["overall"]
    t = _cm(o["t"])
    np.testing.assert_allclose(out["coefficients"], t[:, 0], rtol=1e-8)
    np.testing.assert_allclose(out["se"], t[:, 1], rtol=1e-7)
    np.testing.assert_allclose(out["p"], t[:, 3], rtol=1e-6)
    assert out["rssr"] == pytest.approx(o["rssr"], rel=1e-7)
    assert out["p_chisqr"] == pytest.approx(o["p_chisqr"], abs=1e-6)
    np.testing.assert_allclose(out["part_coefficients"], _cm(case["coefficients"]), rtol=1e-8)


def _cube(seed=3, ny=30, nx=30, nt=20):
    """A projected (time, y, x) cube: three classes with their own trend, spatially smooth
    AR(1) noise, and the classes' layer."""
    rng = np.random.default_rng(seed)
    classes = np.zeros((ny, nx), dtype=int)
    classes[:, nx // 3:] = 1
    classes[:, 2 * nx // 3:] = 2
    slope = np.array([0.0, 0.004, -0.003])[classes]
    yy, xx = np.mgrid[0:ny, 0:nx]
    noise = np.zeros((nt, ny, nx))
    for t in range(nt):
        field = rng.standard_normal((ny + 4, nx + 4))
        smooth = sum(field[i:i + ny, j:j + nx] for i in range(5) for j in range(5)) / 5.0
        noise[t] = (0.5 * noise[t - 1] if t else 0) + 0.01 * smooth
    values = 0.6 + slope[None] * np.arange(nt)[:, None, None] + noise
    dates = pd.date_range("2001-01-01", periods=nt, freq="YS")
    cube = xr.DataArray(values, dims=("time", "y", "x"),
                        coords={"time": dates, "y": 9_000_000.0 - 30.0 * np.arange(ny), "x": 500_000.0 + 30.0 * np.arange(nx)})
    cube = cube.rio.write_crs("EPSG:32723")
    return cube, xr.DataArray(classes, dims=("y", "x"), coords={"y": cube.y, "x": cube.x})


def test_ar_trend_cube_and_pixel():
    cube, classes = _cube()
    ds = zeit.ar_trend(cube, n_jobs=1)
    for name in ("intercept", "slope", "slope_se", "slope_t", "slope_p", "rho", "mse", "loglik"):
        assert name in ds and ds[name].shape == (30, 30)
    assert ds.rio.crs is not None
    means = [float(ds.slope.where(classes == c).mean()) for c in range(3)]
    assert means == pytest.approx([0.0, 0.004, -0.003], abs=1e-3)
    # one pixel's series, per year, with a harmonic
    one = zeit.ar_trend(cube[:, 0, 0].to_series(), time="year", harmonics=1, period=1)
    assert {"slope", "cos1", "sin1"} <= set(one.data_vars)
    # the cube's accessor
    xr.testing.assert_equal(cube.zeit.ar_trend(n_jobs=1).slope, ds.slope)


def test_parts_classes_mean_and_mask():
    cube, classes = _cube()
    res = zeit.parts(cube, classes=classes, partsize=200, nugget=0.0, seed=1, n_jobs=2)
    assert list(res.table.index) == ["class 0", "class 1", "class 2"]
    assert res.table.loc["class 1", "estimate"] == pytest.approx(0.004, abs=1e-3)
    assert res.p_chisqr is not None and res.p_chisqr < 0.01
    assert res.npart == 4 and res.n_pixels == 900 and res.correlation["range"] > 0
    assert "chi-square" in repr(res)

    mean = zeit.parts(cube, mask=(classes == 0), partsize=100, seed=1, n_jobs=2)
    assert list(mean.table.index) == ["mean"] and mean.p_chisqr is None
    assert mean.n_pixels == 300 and mean.table.loc["mean", "p"] > 0.01

    cov = zeit.parts(cube, covariates={"east": cube.x.broadcast_like(classes)}, partsize=200, seed=1, n_jobs=2)
    assert list(cov.table.index) == ["intercept", "east"]


def test_parts_refuses_geographic_and_too_few_pixels():
    cube, _ = _cube()
    with pytest.raises(ValueError, match="geographic"):
        zeit.parts(cube.rio.write_crs("EPSG:4326"), partsize=200)
    with pytest.raises(ValueError, match="partition"):
        zeit.parts(cube, partsize=900)
