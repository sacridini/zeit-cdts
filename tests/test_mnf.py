"""zeit.mnf: the Minimum Noise Fraction transform, the compiled core against its numpy reference and the API."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor
from zeit import _core
from zeit._mnf import _denoise_numpy, _fit_numpy, _forward_numpy, _stats_numpy

NB, H, W, R = 8, 96, 96, 3


def _fields(rng, h, w, r):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    out = []
    for _ in range(r):
        f = np.zeros((h, w))
        for _ in range(4):
            th, per, ph = rng.uniform(0, np.pi), rng.uniform(350, 700), rng.uniform(0, 2 * np.pi)
            f += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * (yy * np.cos(th) + xx * np.sin(th)) / per + ph)
        out.append((f - f.mean()) / f.std())
    return np.array(out)


def synth(nt=1, nb=NB, h=H, w=W, r=R, seed=0, hetero=False):
    """``(time, band, y, x)``: smooth latent fields mixed into the bands (4 noise standard deviations) plus white
    noise; returns the cube, the signal and the noise std of each band."""
    rng = np.random.default_rng(seed)
    sigma = 0.5 * 10 ** (np.arange(nb) / (nb - 1)) if hetero else np.ones(nb)
    cubes, sigs = [], []
    for _ in range(nt):
        f = _fields(rng, h, w, r)
        m = rng.normal(size=(nb, r))
        s = np.einsum("br,ryx->byx", m, f)
        s *= (4 * sigma / s.reshape(nb, -1).std(1))[:, None, None]
        cubes.append(s + sigma[:, None, None] * rng.normal(size=(nb, h, w)))
        sigs.append(s)
    return np.array(cubes), np.array(sigs), sigma


def da(x, names=None, dask=False):
    nt, nb, h, w = x.shape
    out = xr.DataArray(x, dims=("time", "band", "y", "x"),
                       coords={"time": pd.date_range("2020-01-01", periods=nt, freq="16D"),
                               "band": names or [f"B{j + 1}" for j in range(nb)], "y": np.arange(h) * 10.0,
                               "x": np.arange(w) * 10.0}, name="refl")
    out = out.rio.write_crs("EPSG:32633")
    return out.chunk({"y": 32, "x": 40}) if dask else out


@pytest.fixture(scope="module")
def data():
    return synth(seed=1)


def test_the_core_equals_the_numpy_definitions(data):
    x, _, _ = data
    rng = np.random.default_rng(3)
    x = x.copy()
    x[0, 2][rng.random((H, W)) < 0.04] = np.nan
    x[0, :, 30:40, 50:60] = -9999.0
    mask = np.ones((H, W), bool)
    mask[:, 70] = False
    region = np.zeros((H, W), bool)
    region[20:80, 10:60] = True
    for noise in ("diff_hv", "diff_h", "diff_v", "diff_diag", "diff_antidiag"):
        for kw in ({}, {"mask": mask, "region": region}, {"fraction": 0.3, "seed": 5}):
            st = _core.mnf.stats_batch(
                x, noise, None, None if "mask" not in kw else mask.astype(np.uint8),
                None if "region" not in kw else region.astype(np.uint8), [-9999.0], kw.get("fraction", 1.0),
                kw.get("seed", 0), 0, 2)
            n, mean, cov, counts, ncov = _stats_numpy(x, noise, kw.get("mask"), kw.get("region"), [-9999.0],
                                                      kw.get("fraction", 1.0), kw.get("seed", 0))
            core = st.fit().to_dict()
            assert st.n == n and list(st.pairs) == counts, (noise, kw)
            np.testing.assert_allclose(core["mean"], mean, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(core["cov"], cov, rtol=1e-11)
            np.testing.assert_allclose(core["noise_cov"], ncov, rtol=1e-10)
            lam, fwd, inv, kept = _fit_numpy(mean, cov, ncov)
            np.testing.assert_allclose(core["eigenvalues"], lam, rtol=1e-8)
            for i in range(R):      # the separated components, signs included
                np.testing.assert_allclose(core["forward"][i], fwd[i], rtol=1e-6, atol=1e-9 * np.abs(fwd[i]).max())


def test_forward_and_denoise_equal_the_numpy_reference(data):
    x, _, _ = data
    x = x.copy()
    x[0, 1, 5, 5] = np.nan
    mask = np.ones((H, W), bool)
    mask[:, 0] = False
    tr = zeit.mnf_fit(da(x), mask=mask)
    out = zeit.mnf(da(x), transform=tr, mask=mask)
    ref = _forward_numpy(x, tr.mean, tr.forward, tr.forward.shape[0], mask=mask)
    np.testing.assert_allclose(out.values, ref, rtol=1e-6, atol=1e-4)
    assert np.array_equal(np.isnan(out.values), np.isnan(ref))
    assert np.isnan(out.values[0, :, 5, 5]).all() and np.isnan(out.values[0, :, :, 0]).all()
    unit = zeit.mnf(da(x), 4, transform=tr, mask=mask, scale="unit")
    want = _forward_numpy(x, tr.mean, tr.forward, 4, mask=mask, scale="unit", lam=tr.eigenvalues)
    np.testing.assert_allclose(unit.values, want, rtol=1e-6, atol=1e-5)
    den = zeit.mnf_denoise(da(x), 3, transform=tr, mask=mask)
    np.testing.assert_allclose(den.values, _denoise_numpy(x, tr.mean, tr.forward, tr.inverse, 3, mask=mask),
                               rtol=1e-6, atol=1e-4)
    # k = all components is the input, k = 0 the mean
    full = zeit.mnf_denoise(da(x), tr.forward.shape[0], transform=tr, mask=mask)
    ok = ~np.isnan(full.values)
    np.testing.assert_allclose(full.values[ok], x[ok], atol=1e-4 * np.nanstd(x))
    zero = zeit.mnf_denoise(da(x), 0, transform=tr, mask=mask)
    np.testing.assert_allclose(zero.values[0, :, 10, 10], tr.mean, rtol=1e-6)


def test_eigenvalues_and_the_signal_components(data):
    x, s, sigma = data
    out, tr = zeit.mnf(da(x), return_transform=True)
    assert out.dims == ("time", "band", "y", "x") and out.dtype == np.float32
    assert list(out.band.values) == [f"mnf{i + 1}" for i in range(NB)]
    assert out.rio.crs == da(x).rio.crs and out.attrs["mnf"]["estimator"] == "diff_hv"
    assert (np.diff(tr.eigenvalues) <= 0).all()
    assert (tr.eigenvalues[:R] > 3).all() and np.abs(tr.eigenvalues[R:] - 1).max() < 0.3
    assert tr.n_with_snr(1.0) == R
    # the components have noise of unit variance: variance = eigenvalue over the pixels of the statistics
    v = out.values[0].reshape(NB, -1).var(1, ddof=1)
    np.testing.assert_allclose(v, tr.eigenvalues, rtol=1e-4)
    assert (np.abs(out.values[0].reshape(NB, -1).mean(1)) <= 2e-3 * np.sqrt(v)).all()
    # denoising with the signal components brings the data nearer to the truth by sqrt(r / B)
    den = zeit.mnf_denoise(da(x), snr_min=1.0)
    e = np.sqrt(np.mean((den.values - s) ** 2)) / np.sqrt(np.mean((x - s) ** 2))
    assert abs(e - np.sqrt(R / NB)) < 0.05, e
    assert den.attrs["mnf_denoise"] == {"k": R, "snr_min": 1.0}


def test_scale_invariance(data):
    x, _, _ = data
    t0 = zeit.mnf_fit(da(x))
    y = x.copy()
    y[0, 3] *= 1024
    y[0, 5] /= 1024
    y[0, 7] += 500
    t1 = zeit.mnf_fit(da(y))
    np.testing.assert_allclose(t1.eigenvalues, t0.eigenvalues, rtol=1e-9)
    a = zeit.mnf(da(x), transform=t0).values
    b = zeit.mnf(da(y), transform=t1).values
    sd = a.reshape(NB, -1).std(1)
    assert (np.abs(a - b).reshape(NB, -1).max(1) < 1e-5 * sd).all()


def test_integer_and_other_dtypes_give_the_same_components(data):
    x, _, _ = data
    xi = np.round(x * 50).astype(np.int32) + 20000
    ref = zeit.mnf(da(xi.astype(np.float64)), 3).values
    for dtype in (np.int16, np.int32, np.float32, np.uint16, np.uint32):
        out = zeit.mnf(da(xi.astype(dtype)), 3, nodata=None).values
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-3, err_msg=str(dtype))


def test_threads_do_not_change_a_bit(data):
    x, _, _ = data
    a, ta = zeit.mnf(da(x), return_transform=True, n_jobs=1)
    b, tb = zeit.mnf(da(x), return_transform=True, n_jobs=4)
    assert np.array_equal(a.values, b.values, equal_nan=True)
    assert np.array_equal(ta.forward, tb.forward) and np.array_equal(ta.noise_cov, tb.noise_cov)
    assert np.array_equal(ta.eigenvalues, tb.eigenvalues)


def test_sample_fraction_is_reproducible_and_seeded(data):
    x, _, _ = data
    a = zeit.mnf_fit(da(x), sample_fraction=0.25, seed=1)
    b = zeit.mnf_fit(da(x), sample_fraction=0.25, seed=1)
    c = zeit.mnf_fit(da(x), sample_fraction=0.25, seed=2)
    assert np.array_equal(a.forward, b.forward) and a.n_pixels == b.n_pixels
    assert not np.array_equal(a.mean, c.mean)
    assert abs(a.n_pixels / (H * W) - 0.25) < 0.03
    n, *_ = _stats_numpy(x, "diff_hv", fraction=0.25, seed=1)
    assert a.n_pixels == n
    with pytest.raises(ValueError, match="sample_fraction"):
        zeit.mnf_fit(da(x), sample_fraction=0.0)


def test_noise_region_window_and_mask(data):
    x, _, _ = data
    win = (10, 20, 60, 70)
    a = zeit.mnf_fit(da(x), noise_region=win)
    region = np.zeros((H, W), bool)
    region[20:70, 10:60] = True
    b = zeit.mnf_fit(da(x), noise_region=region)
    assert a.noise_region == "window" and b.noise_region == "mask"
    assert np.array_equal(a.noise_cov, b.noise_cov)
    assert a.n_pairs == {"h": 50 * 49, "v": 49 * 50} and a.n_pixels == H * W
    with pytest.raises(ValueError, match="too few|only 6 pairs|pairs"):
        zeit.mnf_fit(da(x), noise_region=(0, 0, 3, 3))
    with pytest.raises(ValueError, match="window"):
        zeit.mnf_fit(da(x), noise_region=(50, 0, 10, 3))


def test_given_noise_covariance(data):
    x, _, sigma = data
    tr = zeit.mnf_fit(da(x), noise="given", noise_cov=np.diag(sigma ** 2))
    assert tr.estimator == "given" and tr.n_pairs == {} and np.isnan(tr.table().maf_autocorrelation).all()
    near = zeit.mnf_fit(da(x))
    np.testing.assert_allclose(tr.eigenvalues[:R], near.eigenvalues[:R], rtol=0.06)
    with pytest.raises(ValueError, match="noise_cov"):
        zeit.mnf_fit(da(x), noise="given")
    with pytest.raises(ValueError, match="symmetric"):
        zeit.mnf_fit(da(x), noise="given", noise_cov=np.triu(np.ones((NB, NB))))
    with pytest.raises(ValueError, match="noise_cov"):
        zeit.mnf_fit(da(x), noise_cov=np.eye(NB))
    with pytest.raises(ValueError, match="noise="):
        zeit.mnf_fit(da(x), noise="nope")


def test_constant_and_empty_bands_are_left_out(data):
    x, _, _ = data
    c = x.copy()
    c[0, 4] = 42.0
    with pytest.warns(UserWarning, match="constant band"):
        tr = zeit.mnf_fit(da(c))
    assert tr.dropped == [{"band": 4, "reason": "constant"}] and len(tr.eigenvalues) == NB - 1
    assert (tr.forward[:, 4] == 0).all()
    den = zeit.mnf_denoise(da(c), 3, transform=tr)
    assert (den.values[0, 4] == 42.0).all() and list(den.band.values) == list(da(c).band.values)
    e = x.copy()
    e[0, 2] = np.nan
    with pytest.warns(UserWarning, match="band with no valid value"):
        tr = zeit.mnf_fit(da(e))
    assert tr.dropped == [{"band": 2, "reason": "empty"}] and tr.n_pixels == H * W
    out = zeit.mnf(da(e), transform=tr)
    assert np.isfinite(out.values).all()
    den = zeit.mnf_denoise(da(e), 3, transform=tr)
    assert np.isnan(den.values[0, 2]).all() and np.isfinite(den.values[0, 1]).all()


def test_the_table(data, tmp_path):
    x, _, _ = data
    tr = zeit.mnf_fit(da(x))
    t = tr.table()
    assert list(t.columns) == ["component", "eigenvalue", "snr", "snr_db", "noise_fraction", "signal_share",
                               "cumulative_signal_share", "maf_autocorrelation", "flag"]
    assert list(t.component) == list(range(1, NB + 1))
    np.testing.assert_allclose(t.snr, t.eigenvalue - 1)
    np.testing.assert_allclose(t.noise_fraction, 1 / t.eigenvalue)
    np.testing.assert_allclose(t.maf_autocorrelation, 1 - 1 / t.eigenvalue)
    assert abs(t.cumulative_signal_share.iloc[-1] - 1) < 1e-12 and (t.signal_share.iloc[:R] > 0.99 * 0 + 0).all()
    np.testing.assert_allclose(t.snr_db[t.snr > 0], 10 * np.log10(t.snr[t.snr > 0]))
    tr.to_csv(tmp_path / "t.csv")
    back = pd.read_csv(tmp_path / "t.csv")
    np.testing.assert_allclose(back.eigenvalue, t.eigenvalue)


def test_json_round_trip_and_from_covariances(data, tmp_path):
    x, _, _ = data
    c = x.copy()
    c[0, 4] = 7.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = zeit.mnf_fit(da(c))
    text = tr.to_json(tmp_path / "t.json")
    doc = json.loads(text)
    assert doc["format"] == "zeit-mnf" and doc["version"] == 1 and set(doc["n_pairs"]) == {"h", "v"}
    assert doc["dropped"] == [{"band": 4, "reason": "constant"}] and doc["estimator"] == "diff_hv"
    for src in (tmp_path / "t.json", text):
        back = zeit.MNFTransform.from_json(src)
        for name in ("mean", "cov", "noise_cov", "eigenvalues", "forward", "inverse", "correlation"):
            assert np.array_equal(getattr(back, name), getattr(tr, name)), name
        assert back.bands == tr.bands and back.dropped == tr.dropped and back.n_pairs == tr.n_pairs
        assert back.to_json() == text
    again = zeit.MNFTransform.from_covariances(tr.mean, tr.cov, tr.noise_cov, tr.bands)
    assert np.array_equal(again.forward, tr.forward) and np.array_equal(again.eigenvalues, tr.eigenvalues)
    with pytest.raises(ValueError, match="zeit-mnf"):
        zeit.MNFTransform.from_json('{"format": "other"}')


def test_a_transform_applies_to_another_date(data):
    a, _, _ = data
    b, _, _ = synth(seed=7)
    tr = zeit.mnf_fit(da(a))
    out = zeit.mnf(da(b), transform=tr)
    hand = np.einsum("ib,byx->iyx", tr.forward, b[0] - tr.mean[:, None, None])
    np.testing.assert_allclose(out.values[0], hand, rtol=1e-5, atol=1e-4)
    _, back = zeit.mnf(da(b), 2, transform=tr, return_transform=True)
    assert back is tr
    # bands matched by name in another order, by position (with a warning) when named differently
    names = list(da(b).band.values)
    shuffled = da(b).sel(band=names[::-1])
    np.testing.assert_allclose(zeit.mnf(shuffled, transform=tr).values, out.values, rtol=1e-6, atol=1e-5)
    other = da(b, names=[f"x{j}" for j in range(NB)])
    with pytest.warns(UserWarning, match="by position"):
        got = zeit.mnf(other, transform=tr)
    np.testing.assert_allclose(got.values, out.values, rtol=1e-6, atol=1e-5)
    with pytest.raises(ValueError, match="bands"):
        zeit.mnf(da(b[:, :5]), transform=tr)


def test_time_series_pooled_and_per_date():
    x, _, _ = synth(nt=3, seed=2)
    pooled, tr = zeit.mnf(da(x), return_transform=True)
    assert isinstance(tr, zeit.MNFTransform) and tr.n_pixels == 3 * H * W
    assert tr.n_pairs == {"h": 3 * H * (W - 1), "v": 3 * (H - 1) * W}      # no pair crosses two dates
    n, mean, cov, counts, ncov = _stats_numpy(x)
    np.testing.assert_allclose(tr.noise_cov, ncov, rtol=1e-10)
    assert pooled.shape == (3, NB, H, W)
    each, trs = zeit.mnf(da(x), per_date=True, return_transform=True)
    assert len(trs) == 3 and [t.date for t in trs] == [str(d) for d in da(x).time.values]
    for t in range(3):
        alone = zeit.mnf_fit(da(x[t:t + 1]))
        np.testing.assert_allclose(trs[t].eigenvalues, alone.eigenvalues, rtol=1e-12)
        np.testing.assert_allclose(each.values[t], zeit.mnf(da(x[t:t + 1]), transform=alone).values[0], atol=1e-4)
    with pytest.raises(ValueError, match="time axis"):
        zeit.mnf_fit(da(x[:1]).isel(time=0), per_date=True)
    den = zeit.mnf_denoise(da(x), snr_min=1.0, per_date=True)
    assert den.shape == (3, NB, H, W)


def test_a_cube_without_time_and_numpy_input(data):
    x, _, _ = data
    cube = da(x).isel(time=0, drop=True)
    out = zeit.mnf(cube, 4)
    assert out.dims == ("band", "y", "x") and out.shape == (4, H, W)
    np.testing.assert_allclose(out.values, zeit.mnf(da(x), 4).values[0], atol=1e-5)
    raw = zeit.mnf(x[0], 4)
    np.testing.assert_allclose(raw.values, out.values, atol=1e-5)
    # band order of dims does not matter
    t = zeit.mnf(da(x).transpose("time", "y", "x", "band"), 4)
    assert t.dims == ("time", "y", "x", "band")
    np.testing.assert_allclose(t.transpose("time", "band", "y", "x").values, zeit.mnf(da(x), 4).values, atol=1e-5)


def test_dask_gives_the_same_transform_and_components(data):
    x, _, _ = data
    x = x.copy()
    x[0, 3][10:20, 5:15] = np.nan
    dense = zeit.mnf_fit(da(x))
    lazy = zeit.mnf_fit(da(x, dask=True))
    assert lazy.n_pixels == dense.n_pixels and lazy.n_pairs == dense.n_pairs
    np.testing.assert_allclose(lazy.noise_cov, dense.noise_cov, rtol=1e-11)
    np.testing.assert_allclose(lazy.eigenvalues, dense.eigenvalues, rtol=1e-9)
    out = zeit.mnf(da(x, dask=True), 4, transform=dense)
    assert out.chunks is not None
    np.testing.assert_allclose(out.values, zeit.mnf(da(x), 4, transform=dense).values, atol=1e-5, equal_nan=True)
    den = zeit.mnf_denoise(da(x, dask=True), 3, transform=dense)
    np.testing.assert_allclose(den.values, zeit.mnf_denoise(da(x), 3, transform=dense).values, atol=1e-4,
                               equal_nan=True)
    both = zeit.inverse_mnf(out, dense, 3)
    np.testing.assert_allclose(both.values, zeit.inverse_mnf(zeit.mnf(da(x), 4, transform=dense), dense, 3).values,
                               atol=1e-4, equal_nan=True)


def test_inverse_mnf(data):
    x, _, _ = data
    y, tr = zeit.mnf(da(x), return_transform=True)
    full = zeit.inverse_mnf(y, tr)
    np.testing.assert_allclose(full.values, x, atol=1e-4 * x.std())
    assert list(full.band.values) == list(da(x).band.values) and full.attrs["mnf_denoise"]["k"] == NB
    three = zeit.inverse_mnf(y, tr, 3)
    np.testing.assert_allclose(three.values, zeit.mnf_denoise(da(x), 3, transform=tr).values, atol=1e-4)
    snr = zeit.inverse_mnf(y, tr, snr_min=1.0)
    np.testing.assert_allclose(snr.values, three.values, atol=1e-6)
    assert np.allclose(zeit.inverse_mnf(y, tr, 0).values[0, :, 3, 3], tr.mean)
    # the unit scale is undone
    yu = zeit.mnf(da(x), transform=tr, scale="unit")
    np.testing.assert_allclose(zeit.inverse_mnf(yu, tr, 3, scale="unit").values, three.values, atol=2e-4)
    # fewer components than asked
    with pytest.raises(ValueError, match="only 2 components"):
        zeit.inverse_mnf(y.isel(band=[0, 1]), tr, 3)
    with pytest.raises(ValueError, match="both"):
        zeit.inverse_mnf(y, tr, 3, snr_min=1.0)
    with pytest.raises(ValueError, match="k=9"):
        zeit.inverse_mnf(y, tr, 9)


def test_errors_and_arguments(data):
    x, _, _ = data
    cube = da(x)
    with pytest.raises(ValueError, match="give the components to keep"):
        zeit.mnf_denoise(cube)
    with pytest.raises(ValueError, match="not both"):
        zeit.mnf_denoise(cube, 2, snr_min=1.0)
    tr = zeit.mnf_fit(cube)
    with pytest.raises(ValueError, match="k=20"):
        zeit.mnf_denoise(cube, 20, transform=tr)
    with pytest.raises(ValueError, match="n_components"):
        zeit.mnf(cube, 20, transform=tr)
    with pytest.raises(ValueError, match="n_components"):
        zeit.mnf(cube, 0)
    with pytest.raises(ValueError, match="scale"):
        zeit.mnf(cube, scale="bad")
    with pytest.raises(ValueError, match="two bands"):
        zeit.mnf_fit(cube.isel(band=[0]))
    with pytest.raises(ValueError, match="already an MNF"):
        zeit.mnf(zeit.mnf(cube))
    with pytest.raises(ValueError, match="only 8 valid pixels"):
        zeit.mnf_fit(da(x[:, :, :2, :4]), noise="given", noise_cov=np.eye(NB))     # 8 pixels for 8 bands
    with pytest.raises(ValueError, match="pairs"):
        zeit.mnf_fit(da(x[:, :, :3, :3]))                     # 6 pairs in a direction for 8 bands
    with pytest.raises(ValueError, match="no pixel has every band valid"):
        zeit.mnf_fit(da(np.full((1, 3, 40, 40), np.nan)))
    with pytest.raises(ValueError, match="every band is constant"):
        zeit.mnf_fit(da(np.full((1, NB, 20, 20), 3.0)))
    with pytest.raises(ValueError, match="bands="):
        zeit.mnf_fit(cube, bands=["nope", "B1"])
    sub = zeit.mnf_fit(cube, bands=["B1", "B3", 5])
    assert sub.bands == ["B1", "B3", "B6"] and sub.eigenvalues.shape == (3,)
    with pytest.raises(ValueError, match="mask must be"):
        zeit.mnf_fit(cube, mask=np.ones((3, 3), bool))


def test_the_sign_rule_and_correlations(data):
    x, _, _ = data
    tr = zeit.mnf_fit(da(x))
    for i in range(NB):
        col = tr.correlation[:, i]
        j = int(np.argmax(np.abs(col)))
        assert col[j] > 0
    # the correlation with the bands is what the components have
    y = zeit.mnf(da(x), transform=tr).values[0].reshape(NB, -1)
    z = x[0].reshape(NB, -1)
    for i in range(NB):
        for j in (0, 3):
            got = np.corrcoef(y[i], z[j])[0, 1]
            assert abs(got - tr.correlation[j, i]) < 2e-3
    assert tr.snr.shape == (NB,) and tr.n_with_snr(-1) == NB
