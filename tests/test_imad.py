"""zeit.imad and zeit.radcal: the iteratively re-weighted MAD and the normalization by its invariant pixels."""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _imad


def _pair(h=120, w=120, n=6, sigma=0.01, seed=0, changed=(slice(10, 40), slice(10, 40)), gain=True):
    """Two dates of a latent scene: date 2 = A date 1 + c + noise, with a block changed in two bands."""
    rng = np.random.default_rng(seed)
    base = 0.15 + 0.1 * rng.normal(size=(n, h, w)) + 0.05 * np.arange(n)[:, None, None]
    a = np.diag(rng.uniform(0.8, 1.2, n)) + 0.05 * (rng.uniform(size=(n, n)) - 0.5) if gain else np.eye(n)
    c = 0.02 * (rng.uniform(size=n) - 0.5) if gain else np.zeros(n)
    x = base + 0.0
    y = np.einsum("ij,jyx->iyx", a, base) + c[:, None, None] + sigma * rng.normal(size=(n, h, w))
    truth = np.zeros((h, w), bool)
    if changed is not None:
        r, cc = changed
        k = truth[r, cc].shape
        y[2][r, cc] += sigma * rng.uniform(5, 20, k) * rng.choice([-1, 1], k)
        y[4][r, cc] += sigma * rng.uniform(5, 20, k) * rng.choice([-1, 1], k)
        truth[r, cc] = True
    return x.astype(np.float64), y.astype(np.float64), truth


def _da(a, names=None, name=None):
    n = a.shape[0]
    return xr.DataArray(a, dims=("band", "y", "x"), name=name,
                        coords={"band": names or [f"b{i}" for i in range(n)], "y": np.arange(a.shape[1]) * 30.0,
                                "x": np.arange(a.shape[2]) * 30.0})


@pytest.fixture(scope="module")
def pair():
    x, y, truth = _pair()
    return _da(x), _da(y), truth


def test_outputs_and_the_reference_agree(pair):
    x, y, truth = pair
    out = zeit.imad(x, y, sample=None, max_iter=10, tol=0, n_jobs=2)
    assert out.dtype == np.float32 and out.dims == ("band", "y", "x")
    assert list(out.band.values) == [f"MAD{i}" for i in range(1, 7)] + ["chi2", "p_nochange", "change"]
    record = json.loads(out.attrs["imad"])
    assert record["iterations"] == 10 and len(record["rho"]) == 10
    fit = {k: np.asarray(record[k]) for k in ("mean", "a", "b", "sd")}
    ref = _imad._block_numpy(x.values.astype(np.float32), y.values.astype(np.float32), fit, [], False, 0.01)
    mad = out.values[:6]
    np.testing.assert_allclose(mad, ref[:6], atol=2e-5, rtol=1e-4)
    np.testing.assert_allclose(out.values[6], ref[6], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out.values[7], ref[7], atol=1e-5)
    # MAD 1 has the largest variance, and the MAD variates are uncorrelated under no weights only after iterating:
    # the first iteration's are
    first = zeit.imad(x, y, sample=None, max_iter=1, n_jobs=1)
    sd = [float(first.values[i].std()) for i in range(6)]
    assert sd == sorted(sd, reverse=True)
    corr = np.corrcoef(first.values[:6].reshape(6, -1))
    assert np.abs(corr - np.eye(6)).max() < 1e-4


def test_detection_and_calibration(pair):
    x, y, truth = pair
    out = zeit.imad(x, y, sample=None)
    record = json.loads(out.attrs["imad"])
    assert record["converged"] and record["iterations"] <= 50
    p = out.sel(band="p_nochange").values
    assert (p[truth] < 0.01).mean() > 0.95
    # the weighted standardization (Canty's) over-flags a little where nothing changed: the weights shrink the
    # variance it estimates
    assert 0.01 < (p[~truth] < 0.01).mean() < 0.3
    calibrated = zeit.imad(x, y, sample=None, standardize="all", tol=1e-6)
    pa = calibrated.sel(band="p_nochange").values
    assert (pa[~truth] < 0.01).mean() < 0.03 and (pa[truth] < 0.01).mean() > 0.9
    assert np.array_equal(out.sel(band="change").values, (p < 0.01).astype(np.float32))
    table = zeit.imad_table(out)
    assert list(table.columns)[:2] == ["iteration", "rho_1"] and len(table) == record["iterations"]
    assert np.isnan(table["max_delta"].iloc[0]) and table["max_delta"].iloc[-1] < 1e-3
    # loadings: the correlation of each MAD with each band, within [-1, 1]
    assert np.abs(np.asarray(record["loadings_x"])).max() <= 1.0 + 1e-9


def test_invariant_to_linear_transforms_of_either_date(pair):
    x, y, _ = pair
    rng = np.random.default_rng(5)
    base = zeit.imad(x, y, sample=None, max_iter=12, tol=0)
    rho0 = np.asarray(json.loads(base.attrs["imad"])["rho"])
    n = 6
    mats = [np.diag([2.0, -0.5, 3.0, 1.5, -1.0, 0.7]),
            np.eye(n) + 0.3 * rng.normal(size=(n, n))]
    assert np.linalg.cond(mats[1]) < 100
    for m in mats:
        for which in (0, 1):
            xs, ys = x.values.copy(), y.values.copy()
            if which == 0:
                xs = np.einsum("ij,jyx->iyx", m, xs) + rng.normal(size=(n, 1, 1))
            else:
                ys = np.einsum("ij,jyx->iyx", m, ys) + rng.normal(size=(n, 1, 1))
            out = zeit.imad(_da(xs), _da(ys), sample=None, max_iter=12, tol=0)
            rho = np.asarray(json.loads(out.attrs["imad"])["rho"])
            np.testing.assert_allclose(rho, rho0, atol=1e-8)
            np.testing.assert_allclose(out.sel(band="chi2").values, base.sel(band="chi2").values, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(out.sel(band="p_nochange").values, base.sel(band="p_nochange").values,
                                       atol=1e-4)
            # |MAD| is the same, the sign may flip
            diff = np.minimum(np.abs(out.values[:6] - base.values[:6]), np.abs(out.values[:6] + base.values[:6]))
            assert diff.max() < 1e-3 * np.abs(base.values[:6]).max()
    # a penalty breaks the invariance
    xs = x.values.copy()
    xs[0] *= 100.0
    a = zeit.imad(x, y, sample=None, max_iter=4, lam=0.01)
    b = zeit.imad(_da(xs), y, sample=None, max_iter=4, lam=0.01)
    assert abs(json.loads(a.attrs["imad"])["rho_final"][0] - json.loads(b.attrs["imad"])["rho_final"][0]) > 1e-6


def test_sample_against_all():
    x, y, _ = _pair(h=300, w=300, seed=3, changed=(slice(20, 80), slice(20, 80)))
    full = zeit.imad(_da(x), _da(y), sample=None)
    part = zeit.imad(_da(x), _da(y), sample=25_000)
    r1, r2 = (np.asarray(json.loads(o.attrs["imad"])["rho_final"]) for o in (full, part))
    assert json.loads(part.attrs["imad"])["stride"] == 2
    assert np.abs(r1 - r2).max() < 1e-3
    agree = (full.sel(band="change").values == part.sel(band="change").values).mean()
    assert agree > 0.98


def test_singular_bands_are_an_error_until_regularized(pair):
    x, y, _ = pair
    xs = x.values.copy()
    xs[1] = xs[0]
    with pytest.raises(ValueError, match="collinear"):
        zeit.imad(_da(xs), y, sample=None, max_iter=3)
    out = zeit.imad(_da(xs), y, sample=None, max_iter=3, lam=1e-6 * 0.01)
    assert np.isfinite(out.values[:6]).all()


def test_nodata_and_dates_and_lazy_equals_eager(pair):
    x, y, _ = pair
    xv = x.values.astype(np.float32)
    yv = y.values.astype(np.float32)
    xv[2, 5, 7] = np.nan
    yv[0, 9, 9] = -9999
    cube = xr.concat([_da(xv), _da(yv)], dim="time").assign_coords(
        time=pd.to_datetime(["2020-07-01", "2021-07-01"]))
    cube = cube.transpose("time", "band", "y", "x")
    cube.attrs["nodata"] = -9999
    out = zeit.imad(cube, dates=("2020-07-01", "2021-07-01"), nodata=-9999, max_iter=5, sample=None)
    bad = np.isnan(out.values[0])
    assert bad.sum() == 2 and bad[5, 7] and bad[9, 9]
    assert np.isnan(out.values).sum(axis=(1, 2)).tolist() == [2] * 9
    lazy = zeit.imad(cube.chunk({"y": 40, "x": 50, "band": -1}), dates=(0, 1), nodata=-9999, max_iter=5, sample=None)
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, out.values)
    with pytest.raises(ValueError, match="dates"):
        zeit.imad(cube, dates=(0, 5))


def test_threads_do_not_change_the_bits(pair):
    x, y, _ = pair
    a = zeit.imad(x, y, sample=None, max_iter=6, tol=0, n_jobs=1, canonical=True)
    b = zeit.imad(x, y, sample=None, max_iter=6, tol=0, n_jobs=4, canonical=True)
    assert a.shape[0] == 6 + 3 + 12
    np.testing.assert_array_equal(a.values, b.values)
    assert json.loads(a.attrs["imad"])["rho"] == json.loads(b.attrs["imad"])["rho"]


def _scene(seed=4, h=200, w=200, sigma=0.01):
    """reference = s + e1, target = A s + c + e2 (equal noise), with changed blocks."""
    rng = np.random.default_rng(seed)
    n = 4
    s = 0.2 + 0.1 * rng.normal(size=(n, h, w))
    gain = np.array([1.3, 0.8, 1.1, 0.9])
    off = np.array([0.02, -0.03, 0.01, 0.0])
    ref = s + sigma * rng.normal(size=s.shape)
    tgt = gain[:, None, None] * s + off[:, None, None] + sigma * rng.normal(size=s.shape)
    tgt[:, 20:70, 20:70] += 0.15 * rng.normal(size=(n, 50, 50))
    tgt[:, 120:170, 100:160] -= 0.1
    return ref, tgt, gain, off, sigma


def test_radcal_recovers_the_inverse_transform():
    ref, tgt, gain, off, sigma = _scene()
    out = zeit.radcal(_da(ref), after=_da(tgt), sample=None)
    table = zeit.radcal_table(out)
    assert len(table) == 4 and {"a", "b", "se_a", "se_b", "p_t", "p_F", "n_train", "n_test"} <= set(table.columns)
    for j, row in table.iterrows():
        assert abs(row["b"] - 1 / gain[j]) < 3 * row["se_b"], (j, row["b"], 1 / gain[j], row["se_b"])
        assert abs(row["a"] + off[j] / gain[j]) < 3 * row["se_a"]
        assert row["p_t"] > 0.01
        assert row["n_test"] > 0 and abs(row["n_test"] * 2 - row["n_train"]) <= 3
    invariant = np.ones(ref.shape[1:], bool)
    invariant[20:70, 20:70] = False
    invariant[120:170, 100:160] = False
    rmse = np.sqrt(((out.values - ref)[:, invariant] ** 2).mean(axis=1))
    assert (rmse < 3 * sigma * 1.5).all()
    assert out.dtype == np.float32


def test_radcal_series():
    ref, tgt, gain, off, sigma = _scene(seed=6)
    cube = xr.concat([_da(ref), _da(tgt), _da(ref * 1.0)], dim="time").assign_coords(
        time=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]))
    out = zeit.radcal(cube, reference="2020-01-01", sample=None, holdout=0)
    assert out.dims == cube.dims
    np.testing.assert_array_equal(out.values[0], ref.astype(np.float32))      # the reference is copied
    record = json.loads(out.attrs["radcal"])
    assert {r["date"] for r in record["fits"]} == {"2020-02-01", "2020-03-01"}
    assert all(np.isnan(r["p_t"]) for r in record["fits"])                   # no hold-out, no tests
    # a target that is the reference: the fit is the identity
    third = zeit.radcal_table(out).query("date == '2020-03-01'")
    np.testing.assert_allclose(third["b"], 1.0, atol=0.02)


def test_arguments_are_checked(pair):
    x, y, _ = pair
    with pytest.raises(ValueError, match="penalty"):
        zeit.imad(x, y, penalty="l1")
    with pytest.raises(ValueError, match="standardize"):
        zeit.imad(x, y, standardize="none")
    with pytest.raises(ValueError, match="bands"):
        zeit.imad(x, y, bands=["nope"])
    with pytest.raises(ValueError, match="same shape|different shapes"):
        zeit.imad(x, y.isel(x=slice(0, 50)))
    one = zeit.imad(x, y, bands=["b0", "b3"], sample=None, max_iter=3)
    assert list(one.band.values)[:2] == ["MAD1", "MAD2"] and one.sizes["band"] == 5
    with pytest.raises(ValueError, match="holdout"):
        zeit.radcal(x, after=y, holdout=1.5)
