"""zeit.omnibus and zeit.estimate_enl: the omnibus test for Sentinel-1 series."""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _core, _omnibus

K = 12


def _series(h=16, w=16, k=K, enl=4.4, seed=0, factors=None, bands=("vv", "vh")):
    """Gamma speckle of `enl` looks on a background; `factors(date) -> per-band factors`."""
    rng = np.random.default_rng(seed)
    mu = np.array([0.12, 0.03])[: len(bands), None, None] * np.exp(rng.normal(0, 0.3, (len(bands), h, w)))
    arr = np.empty((k, len(bands), h, w))
    for d in range(k):
        f = np.ones(len(bands)) if factors is None else np.asarray(factors(d), dtype=float)[: len(bands)]
        arr[d] = rng.gamma(enl, 1.0 / enl, (len(bands), h, w)) * mu * f[:, None, None]
    time = pd.date_range("2021-06-01", periods=k, freq="12D")
    return xr.DataArray(arr.astype(np.float32), dims=("time", "band", "y", "x"),
                        coords={"time": time, "band": list(bands), "y": np.arange(h) * 10.0,
                                "x": np.arange(w) * 10.0}).rio.write_crs("EPSG:32620")


def test_the_parameters_of_the_papers():
    f, rho, w2 = _core.omnibus.params("q", 2, 12.0, 15)
    assert f == 28 and abs(rho - 0.985185185185185) < 1e-14 and abs(w2 + 0.00158290462999) < 1e-12
    g, rho, w2 = _core.omnibus.params("r", 2, 12.0, 10)
    assert g == 2 and abs(rho - 0.985956790123457) < 1e-14 and abs(w2 + 0.000101434808286) < 1e-12
    f, rho, w2 = _core.omnibus.params("q", 1, 4.4, 10)
    assert f == 9 and abs(rho - 0.958333333333333) < 1e-14 and abs(w2 + 0.00425330812854) < 1e-12


def test_planes_dates_and_the_python_reference():
    factors = lambda d: [0.1, 0.1] if d >= 6 else [1, 1]
    cube = _series(h=12, w=12, seed=3, factors=factors)
    cube.values[4, 0, 0, 0] = 0.0          # a layover value
    cube.values[:, 1, 1, 1] = np.nan       # a pixel with no valid VH
    out = zeit.omnibus(cube, n_jobs=2)
    assert out.dtype == np.float32 and list(out.band.values) == list(_omnibus.PLANES)
    record = json.loads(out.attrs["omnibus"])
    assert record["enl"] == 4.4 and record["approximation"] == "improved" and len(record["dates"]) == K
    dates, _ = _omnibus._decimal_years(cube.time)
    found = out.sel(band="first_change").values
    assert np.isnan(found[1, 1]) and np.isnan(out.sel(band="n_dates").values[1, 1])
    valid = ~np.isnan(found)
    assert valid.mean() > 0.7
    allowed = dates[6:].astype(np.float32)
    assert np.isin(found[valid][np.isfinite(found[valid])], np.concatenate([allowed, dates[:6].astype(np.float32)])).all()
    assert out.sel(band="n_dates").values[0, 0] == K - 1
    ref = _omnibus._block_numpy(cube.values, [0, 1], 4.4, 0.01, "improved", False, [], dates)
    np.testing.assert_allclose(out.values, ref, atol=1e-5, equal_nan=True)
    chi = zeit.omnibus(cube, approximation="chi2")
    refc = _omnibus._block_numpy(cube.values, [0, 1], 4.4, 0.01, "chi2", False, [], dates)
    np.testing.assert_allclose(chi.values, refc, atol=1e-5, equal_nan=True)


def test_known_change_dates_and_directions():
    k = 12
    cube = _series(h=40, w=60, seed=5, k=k)
    arr = cube.values.copy()
    arr[6:, :, :, :20] *= 0.05                           # block A: darkening from date 6
    arr[3:8, :, :, 20:40] *= 10.0                        # block B: x 10 at date 3, back at date 8
    arr[5:, 0, :, 40:] *= 10.0                           # block C: VV up, VH down from date 5
    arr[5:, 1, :, 40:] *= 0.05
    out, per = zeit.omnibus(cube.copy(data=arr), per_date=True)
    dates, _ = _omnibus._decimal_years(cube.time)
    d32 = dates.astype(np.float32)
    first, last = out.sel(band="first_change").values, out.sel(band="last_change").values
    fdir, ldir = out.sel(band="first_direction").values, out.sel(band="last_direction").values
    nch = out.sel(band="n_changes").values
    a = (slice(None), slice(0, 20))
    near = lambda v, d: np.isclose(v, d32[d])
    assert (near(first[a], 6) & (fdir[a] == 2)).mean() > 0.85
    b = (slice(None), slice(20, 40))
    assert (near(first[b], 3) & (fdir[b] == 1) & near(last[b], 8) & (ldir[b] == 2) & (nch[b] == 2)).mean() > 0.75
    c = (slice(None), slice(40, None))
    assert ((fdir[c] == 3) & (nch[c] >= 1)).mean() > 0.85
    assert per.dims == ("time", "y", "x")
    assert (per.values[6][a] == 2).mean() > 0.8 and (per.values[:6][:, a[0], a[1]] == 0).mean() > 0.9
    # the last change is the first one when there is one
    one = nch == 1
    np.testing.assert_array_equal(first[one], last[one])


def test_no_change_false_alarms_and_scale_invariance():
    cube = _series(h=100, w=100, seed=9)
    out = zeit.omnibus(cube)
    p = out.sel(band="p_change").values
    assert abs((p < 0.01).mean() - 0.01) < 0.01
    assert (out.sel(band="n_changes").values > 0).mean() < 0.015
    rng = np.random.default_rng(1)
    scale = rng.uniform(0.01, 80, (1, 2, 100, 100)).astype(np.float32)
    scaled = zeit.omnibus(cube.copy(data=cube.values * scale))
    np.testing.assert_allclose(scaled.sel(band="p_change").values, p, atol=2e-5, rtol=1e-3)
    np.testing.assert_array_equal(scaled.sel(band="n_changes").values, out.sel(band="n_changes").values)


def test_single_pol_lazy_threads_and_errors():
    cube = _series(h=20, w=24, seed=2, factors=lambda d: [0.05, 1] if d >= 6 else [1, 1])
    single = zeit.omnibus(cube, bands=("vv",))
    assert json.loads(single.attrs["omnibus"])["bands"] == ["vv"]
    assert (single.sel(band="n_changes").values >= 1).mean() > 0.8
    a = zeit.omnibus(cube, n_jobs=1)
    b = zeit.omnibus(cube, n_jobs=4)
    np.testing.assert_array_equal(a.values, b.values)
    lazy = zeit.omnibus(cube.chunk({"y": 7, "x": 9}))
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, a.values)
    with pytest.raises(ValueError, match="linear power"):
        zeit.omnibus(cube.copy(data=10 * np.log10(cube.values)))
    with pytest.raises(ValueError, match="median"):
        zeit.omnibus(cube, median=3)
    with pytest.raises(ValueError, match="no band"):
        zeit.omnibus(cube, bands=("hh",))
    with pytest.raises(ValueError, match="two dates"):
        zeit.omnibus(cube.isel(time=[0]))
    mixed = cube.assign_coords(**{"sat:orbit_state": ("time", ["ascending", "descending"] * (K // 2))})
    with pytest.warns(UserWarning, match="orbit_state"):
        zeit.omnibus(mixed)


def test_estimate_enl():
    cube = _series(h=60, w=60, k=6, enl=7.5, seed=4)
    # homogeneous: the same mean everywhere
    rng = np.random.default_rng(0)
    arr = rng.gamma(7.5, 1 / 7.5, (6, 2, 60, 60)).astype(np.float32) * np.array([0.1, 0.03], dtype=np.float32)[:, None, None]
    flat = cube.copy(data=arr)
    est = zeit.estimate_enl(flat)
    assert abs(est["enl"]["vv"] - 7.5) < 0.5 and abs(est["enl"]["vh"] - 7.5) < 0.5
    assert len(est["per_date"]["vv"]) == 6 and est["n_pixels"] == 3600
    sub = zeit.estimate_enl(flat, region=(0, 30, 0, 30))
    assert sub["n_pixels"] == 900 and abs(sub["enl"]["vv"] - 7.5) < 1.0
    mask = np.zeros((60, 60), bool)
    mask[10:20, 10:20] = True
    assert zeit.estimate_enl(flat, bands=("vv",), region=mask)["n_pixels"] == 100
