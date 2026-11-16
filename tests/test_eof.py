"""Temporal EOF (zeit.eof): known modes and PCs made in the tests, the solvers, the determinism over the threads,
missing pixels and dates (the gappy projection), the varimax rotation, the weights and the errors."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

T, NY, NX = 120, 10, 12
S = np.array([10.0, 5.0, 2.0])


def _dates(n=T):
    return pd.DatetimeIndex((np.datetime64("2000-01", "M") + np.arange(n)).astype("datetime64[D]") + np.timedelta64(14, "D"))


def _modes():
    """Orthonormal U (T x 3, zero mean) and V (N x 3): a dipole, a monopole and a checkerboard."""
    t = np.arange(T)
    u = np.stack([np.sin(2 * np.pi * t / p + ph) for p, ph in ((29, 0.3), (47, 1.1), (61, 2.0))], axis=1)
    u -= u.mean(axis=0)
    u, _ = np.linalg.qr(u)
    ii, jj = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    v = np.stack([np.where(jj < NX // 2, 1.0, -1.0).ravel(), np.ones(NY * NX), (((ii + jj) % 2) * 2 - 1.0).ravel()],
                 axis=1)
    v, _ = np.linalg.qr(v)
    return u, v


def _cube(seed=0, noise=0.01, crs=None, lat0=None):
    u, v = _modes()
    rng = np.random.default_rng(seed)
    x = (u * S) @ v.T + noise * rng.normal(size=(T, NY * NX))
    coords = {"time": _dates()}
    if crs is None:
        coords.update(y=np.arange(NY, dtype=float), x=np.arange(NX, dtype=float))
    elif crs == "geographic":
        coords.update(y=lat0 - 0.5 * np.arange(NY), x=10.0 + 0.5 * np.arange(NX))
    else:
        coords.update(y=5_000_000.0 - 1000.0 * np.arange(NY), x=500_000.0 + 1000.0 * np.arange(NX))
    da = xr.DataArray(x.reshape(T, NY, NX), dims=("time", "y", "x"), coords=coords)
    if crs == "geographic":
        da = da.rio.write_crs("EPSG:4326")
    elif crs == "projected":
        da = da.rio.write_crs("EPSG:32633")
    return da, u, v


def _cos(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)


def test_known_modes_pcs_and_variance():
    da, u, v = _cube()
    ds = zeit.eof(da, n_modes=3, anomaly="none", weights="none", n_jobs=2)
    assert ds.eof.dims == ("mode", "y", "x") and ds.pc.dims == ("mode", "time")
    for k in range(3):
        assert _cos(ds.eof.values[k], v[:, k]) > 0.995
        assert _cos(ds.pc.values[k], u[:, k]) > 0.995
    # units scaling: unit-variance PCs, the EOF in data units per standard deviation of the PC
    np.testing.assert_allclose(ds.pc.values.std(axis=1, ddof=1), 1.0, rtol=1e-9)
    rebuilt = np.einsum("kt,kyx->tyx", ds.pc.values, ds.eof.values)
    np.testing.assert_allclose(rebuilt, da.values, atol=0.05)
    expected = S ** 2 / (S ** 2).sum()
    np.testing.assert_allclose(ds.variance.values, expected, atol=1e-2)      # the noise is 1 % of the energy
    np.testing.assert_allclose(ds.eigenvalue.values, ds.variance.values * ds.attrs["total_variance"], rtol=1e-9)
    assert ds.separable.values.tolist() == [True, True, True]
    assert (ds.north_error.values > 0).all()
    assert ds.attrs["solver"] == "exact" and ds.attrs["weights"] == "none" and ds.attrs["anomaly"] == "none"
    assert not ds.pc_filled.values.any() and ds.pc_filled.dims == ("time",)
    np.testing.assert_array_equal(ds["mode"].values, [1, 2, 3])
    np.testing.assert_array_equal(ds.time.values, da.time.values)


def test_scaling_none_gives_orthonormal_eofs_and_pcs_with_the_variance():
    da, _, _ = _cube()
    ds = zeit.eof(da, n_modes=3, anomaly="none", weights="none", scaling="none")
    flat = ds.eof.values.reshape(3, -1)
    np.testing.assert_allclose(flat @ flat.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(ds.pc.values.std(axis=1, ddof=1) ** 2, ds.eigenvalue.values, rtol=1e-6)


def test_sign_rule_monopole_pc_follows_the_mean():
    da, _, v = _cube()
    ds = zeit.eof(da, n_modes=3, anomaly="none", weights="none")
    mean_anomaly = (da - da.mean("time")).mean(("y", "x")).values
    mono = int(np.argmax([abs(v[:, 1] @ ds.eof.values[k].ravel()) for k in range(3)]))
    assert np.corrcoef(ds.pc.values[mono], mean_anomaly)[0, 1] > 0
    for k in range(3):                                   # the dipole: its largest PC value is positive
        pc = ds.pc.values[k]
        if k != mono and abs(np.corrcoef(pc, mean_anomaly)[0, 1]) < 1e-3:
            assert pc[np.argmax(np.abs(pc))] > 0


def test_randomized_equals_exact_and_threads_change_no_bit():
    da, _, _ = _cube(noise=0.05)
    ex = zeit.eof(da, n_modes=3, anomaly="none", weights="none", solver="exact")
    rd = zeit.eof(da, n_modes=3, anomaly="none", weights="none", solver="randomized", seed=1,
                  power_iterations=5)
    assert rd.attrs["solver"] == "randomized"
    np.testing.assert_allclose(rd.eigenvalue.values, ex.eigenvalue.values, rtol=1e-6)
    for k in range(3):
        assert _cos(rd.eof.values[k], ex.eof.values[k]) > 0.99999
    a = zeit.eof(da, n_modes=3, anomaly="none", weights="none", solver="randomized", n_jobs=1)
    b = zeit.eof(da, n_modes=3, anomaly="none", weights="none", solver="randomized", n_jobs=3)
    np.testing.assert_array_equal(a.eof.values, b.eof.values)
    np.testing.assert_array_equal(a.pc.values, b.pc.values)
    np.testing.assert_array_equal(a.eigenvalue.values, b.eigenvalue.values)


def test_missing_pixels_and_dates():
    da, u, v = _cube()
    vals = da.values.copy()
    vals[:, 0, 0] = np.nan                               # a pixel without data
    vals[:, 3, :4] = np.nan                              # a pixel with data on a few dates only
    vals[5:, 3, 0] = np.nan
    drop = 40                                            # a date seen by 30 % of the pixels
    gap = np.random.default_rng(3).random((NY, NX)) < 0.7
    gap[0, 0] = False
    vals[drop][gap] = np.nan
    ds = zeit.eof(da.copy(data=vals), n_modes=3, anomaly="none", weights="none", min_valid=0.8,
                  min_date_valid=0.5)
    assert np.isnan(ds.eof.values[:, 0, 0]).all() and np.isnan(ds.eof.values[:, 3, 0]).all() and np.isnan(ds.eof.values[:, 3, 2]).all()
    assert np.isfinite(ds.eof.values[:, 5, 5]).all()
    assert ds.attrs["n_pixels"] == NY * NX - 1 - 4 and ds.attrs["n_dates"] == T - 1
    assert ds.pc_filled.values.sum() == 1 and bool(ds.pc_filled.values[drop])
    full = zeit.eof(da, n_modes=3, anomaly="none", weights="none")
    for k in range(3):                                   # the dropped date's PC, by least squares over 30 % of the pixels
        sign = np.sign(ds.pc.values[k] @ full.pc.values[k])
        assert abs(ds.pc.values[k, drop] * sign - full.pc.values[k, drop]) < 0.15 * max(1.0, abs(full.pc.values[k, drop]))
    assert np.isfinite(ds.pc.values).all()
    assert (ds.pc.values[:, :drop] != 0).all()


def test_no_valid_pixel_is_an_error():
    da, _, _ = _cube()
    with pytest.raises(ValueError, match="no pixel"):
        zeit.eof(da * np.nan, n_modes=2, anomaly="none", weights="none")


def test_monthly_anomaly_removes_the_cycle_and_harmonic_too():
    da, u, v = _cube()
    cycle = 3.0 * np.sin(2 * np.pi * np.arange(T) / 12.0)
    cyc = da + xr.DataArray(cycle, dims="time", coords={"time": da.time})
    for kind in ("month", "harmonic"):
        ds = zeit.eof(cyc, n_modes=3, anomaly=kind, weights="none", harmonics=1)
        assert ds.attrs["anomaly"] == kind
        assert _cos(ds.eof.values[0], v[:, 0]) > 0.99
        assert 0.6 < ds.variance.values[0] < 0.8                   # the climatology takes some of the signal too
    with_cycle = zeit.eof(cyc, n_modes=3, anomaly="none", weights="none")
    assert with_cycle.variance.values[0] > 0.7 and _cos(with_cycle.eof.values[0], v[:, 0]) < 0.9   # the cycle leads


def test_base_detrend_standardize_run():
    da, _, v = _cube()
    trend = xr.DataArray(0.05 * np.arange(T), dims="time", coords={"time": da.time})
    ds = zeit.eof(da + trend, n_modes=3, anomaly="month", base=(2000, 2005), detrend=True, standardize=True,
                  weights="none")
    assert np.isfinite(ds.variance.values).all() and ds.variance.values.sum() <= 1.0 + 1e-9
    assert _cos(ds.eof.values[0], v[:, 0]) > 0.9


def test_correlation_maps():
    da, _, _ = _cube()
    ds = zeit.eof(da, n_modes=2, anomaly="none", weights="none", correlation_maps=True)
    assert ds.corrmap.dims == ("mode", "y", "x")
    for (j, i) in ((2, 3), (7, 9)):
        for k in range(2):
            expected = np.corrcoef(ds.pc.values[k], da.values[:, j, i])[0, 1]
            np.testing.assert_allclose(ds.corrmap.values[k, j, i], expected, atol=1e-5)
    assert "corrmap" not in zeit.eof(da, n_modes=2, anomaly="none", weights="none")


def test_north_with_autocorrelation():
    t = np.arange(T)
    rng = np.random.default_rng(5)
    a = np.sin(2 * np.pi * t / 29.0)
    b = np.cos(2 * np.pi * t / 29.0)                       # two modes of the same size: not separable
    x = np.outer(a, rng.normal(size=60)) + np.outer(b, rng.normal(size=60)) + 0.01 * rng.normal(size=(T, 60))
    da = xr.DataArray(x.reshape(T, 6, 10), dims=("time", "y", "x"),
                      coords={"time": _dates(), "y": np.arange(6.0), "x": np.arange(10.0)})
    ds = zeit.eof(da, n_modes=3, anomaly="none", weights="none", north_n="autocorrelation")
    assert ds.separable.dtype == bool and ds.separable.shape == (3,)
    plain = zeit.eof(da, n_modes=3, anomaly="none", weights="none", north_n="dates")
    assert (ds.north_error.values[:2] > plain.north_error.values[:2]).all()      # fewer independent dates -> larger error


def test_varimax_recovers_two_disjoint_patches():
    rng = np.random.default_rng(7)
    ny, nx = 8, 16
    pa = np.zeros((ny, nx))
    pa[:, :6] = 1.0
    pb = np.zeros((ny, nx))
    pb[:, 10:] = 1.0
    z = rng.normal(size=(T, 2))
    z -= z.mean(axis=0)
    q, _ = np.linalg.qr(z)
    q *= np.sqrt(T - 1)                                    # uncorrelated, unit variance, the same size
    x = np.einsum("t,yx->tyx", q[:, 0], pa) + np.einsum("t,yx->tyx", q[:, 1], pb)
    x += 0.001 * rng.normal(size=x.shape)
    da = xr.DataArray(x, dims=("time", "y", "x"),
                      coords={"time": _dates(), "y": np.arange(ny, dtype=float), "x": np.arange(nx, dtype=float)})
    ds = zeit.eof(da, n_modes=3, anomaly="none", weights="none", rotate=2)
    assert ds.eof_rotated.dims == ("mode_rotated", "y", "x") and ds.pc_rotated.dims == ("mode_rotated", "time")
    assert ds.variance_rotated.dims == ("mode_rotated",)
    shares = sorted(_cos(ds.eof_rotated.values[k], p) for k in range(2) for p in (pa, pb))[-2:]
    assert min(shares) > 0.99
    for k in range(2):
        assert max(_cos(ds.pc_rotated.values[k], q[:, 0]), _cos(ds.pc_rotated.values[k], q[:, 1])) > 0.999
    np.testing.assert_allclose(ds.variance_rotated.values.sum(), ds.variance.values[:2].sum(), rtol=1e-9)
    assert ds.variance_rotated.values[0] >= ds.variance_rotated.values[1]
    # without the rotation these variables are absent
    assert "eof_rotated" not in zeit.eof(da, n_modes=3, anomaly="none", weights="none")
    nok = zeit.eof(da, n_modes=3, anomaly="none", weights="none", rotate=3, kaiser=False)
    assert nok.eof_rotated.sizes["mode_rotated"] == 3


def test_weights_geographic_and_projected():
    geo, _, _ = _cube(crs="geographic", lat0=70.0)
    proj, _, _ = _cube(crs="projected")
    area = zeit.eof(geo, n_modes=3, anomaly="none", weights="area")
    cosl = zeit.eof(geo, n_modes=3, anomaly="none", weights="coslat")
    none = zeit.eof(geo, n_modes=3, anomaly="none", weights="none")
    assert area.attrs["weights"] == "area" and cosl.attrs["weights"] == "coslat"
    assert area.attrs["total_variance"] != none.attrs["total_variance"]
    np.testing.assert_allclose(area.variance.values, cosl.variance.values, atol=2e-3)   # area ~ cos(lat), up to the sampling
    assert area.rio.crs is not None
    with pytest.raises(ValueError, match="area"):
        zeit.eof(proj, n_modes=3, anomaly="none", weights="coslat")
    p = zeit.eof(proj, n_modes=3, anomaly="none", weights="area")          # equal for every pixel: = unweighted
    np.testing.assert_allclose(p.variance.values, zeit.eof(proj, n_modes=3, anomaly="none", weights="none")
                               .variance.values, rtol=1e-6)
    plain, _, _ = _cube()
    with pytest.warns(UserWarning, match="no CRS"):
        w = zeit.eof(plain, n_modes=3, anomaly="none", weights="area")
    assert w.attrs["weights"] == "none"
    custom = zeit.eof(plain, n_modes=3, anomaly="none", weights=np.ones((NY, NX)) * 5.0)
    assert custom.attrs["weights"] == "custom"
    np.testing.assert_allclose(custom.variance.values, zeit.eof(plain, n_modes=3, anomaly="none", weights="none")
                               .variance.values, rtol=1e-9)


def test_errors():
    da, _, _ = _cube()
    with pytest.raises(ValueError, match="anomaly"):
        zeit.eof(da, anomaly="bogus")
    with pytest.raises(ValueError, match="solver"):
        zeit.eof(da, solver="bogus", weights="none")
    with pytest.raises(ValueError, match="scaling"):
        zeit.eof(da, scaling="bogus", weights="none")
    with pytest.raises(ValueError, match="north_n"):
        zeit.eof(da, north_n="bogus", weights="none")
    with pytest.raises(ValueError, match="rotate"):
        zeit.eof(da, n_modes=3, rotate=1, weights="none")
    with pytest.raises(ValueError, match="rotate"):
        zeit.eof(da, n_modes=3, rotate=5, weights="none")
    with pytest.raises(ValueError, match="n_modes"):
        zeit.eof(da, n_modes=0, weights="none")
    with pytest.raises(ValueError, match="weights"):
        zeit.eof(da, weights="bogus")
    with pytest.raises(ValueError, match="shape"):
        zeit.eof(da, weights=np.ones((2, 2)))
    with pytest.raises(ValueError, match="one pixel"):
        zeit.eof(da.isel(y=0, x=0))
