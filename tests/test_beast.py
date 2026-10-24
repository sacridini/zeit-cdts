"""zeit.beast against Rbeast's beast() and beast.irreg() (tests/data/make_beast_fixtures.R): the same chain
from the same seed, so the same float32 outputs; and its behaviour on cubes, gaps and bad input."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

DATA = Path(__file__).parent / "data"
REF = json.loads((DATA / "beast_rbeast.json").read_text())
CASES = {c["name"]: c for c in REF["cases"]}


def _arr(v):
    return np.array([np.nan if x is None else x for x in np.ravel(np.array(v, dtype=object))], dtype=np.float64)


def _fit(case):
    a = dict(case["args"])
    kw = dict(seed=a["seed"], chains=a["chains"], samples=a.get("samples", 8000), ci=a.get("ci", True),
              prec_type=a.get("prec_type", "componentwise"), season=a.get("season", "harmonic"))
    for key in ("period", "deltat", "start"):
        if key in a and not (key == "start" and case["times"] is not None):
            kw[key] = a[key]
    for key in ("tcp_minmax", "torder_minmax", "scp_minmax"):
        if key in a:
            kw[key] = tuple(a[key])
    y = _arr(case["y"])
    if case["times"] is not None:
        kw["times"] = _arr(case["times"])
    return zeit.beast(y, **kw)


def _same(z, r, what):
    """Equal as float32 (NaN where NaN)."""
    z = np.asarray(z, dtype=np.float32).ravel()
    r = np.asarray(r, dtype=np.float64).astype(np.float32).ravel()
    assert z.shape == r.shape, what
    np.testing.assert_array_equal(z, r, err_msg=what)


@pytest.mark.parametrize("name", list(CASES))
def test_same_as_rbeast(name):
    case = CASES[name]
    ds = _fit(case)
    a = case["args"]
    chains = a["chains"]
    n = ds.sizes["time"]
    # Rbeast's time is accumulated in float32 (it drifts by up to 0.01 over 774 steps); zeit's is start + dt * i.
    np.testing.assert_allclose(ds.time.values, _arr(case["time"]), rtol=0, atol=1e-2)
    np.testing.assert_allclose(ds.data.values, _arr(case["data"]), rtol=2e-6, atol=1e-6)
    for key in ("marg_lik", "sig2", "r2", "rmse"):
        _same(ds[key].values, case[key], key)
    for comp, prefix in (("trend", "trend"), ("season", "season")):
        r = case[comp]
        if r is None:
            assert f"{prefix}_ncp" not in ds
            continue
        names = ["ncp", "ncp_prob", "cp_occ", "order", "sd", "cp", "cp_prob", "cp_abrupt"]
        if chains > 1:
            names += ["ncp_mode", "ncp_median", "ncp_pct90", "ncp_pct10"]
        for key in names:
            _same(ds[f"{prefix}_{key}"].values, r[key], f"{name} {prefix}_{key}")
        _same(ds[prefix].values, r["y"], f"{name} {prefix}")
        cp_ci = _arr(r["cp_ci"]).reshape(-1, 2)          # (cp, lower/upper)
        _same(ds[f"{prefix}_cp_ci"].values, cp_ci.T, f"{name} {prefix}_cp_ci")
        if a.get("ci", True):
            ci = _arr(r["ci"]).reshape(n, 2)              # (time, upper/lower)
            _same(ds[f"{prefix}_ci"].sel(bound="lower").values, ci[:, 1], f"{name} {prefix}_ci lower")
            _same(ds[f"{prefix}_ci"].sel(bound="upper").values, ci[:, 0], f"{name} {prefix}_ci upper")
        else:
            assert f"{prefix}_ci" not in ds
        if comp == "trend":
            for key in ("slope", "slope_sd", "slope_pos_prob", "slope_zero_prob"):
                _same(ds[f"trend_{key}"].values, r[key], f"{name} trend_{key}")
            if a.get("ci", True):
                # Rbeast sums the slope's interval over the chains, per time step; zeit's is the mean, per unit
                # of time as the slope.
                dt = float(ds.attrs["deltat"])
                sci = _arr(r["slope_ci"]).reshape(n, 2) / (chains * dt)
                np.testing.assert_allclose(ds.trend_slope_ci.sel(bound="lower").values, sci[:, 1], rtol=1e-5,
                                           atol=1e-6 * np.nanmax(np.abs(sci)))
                np.testing.assert_allclose(ds.trend_slope_ci.sel(bound="upper").values, sci[:, 0], rtol=1e-5,
                                           atol=1e-6 * np.nanmax(np.abs(sci)))
        else:
            _same(ds.season_amp.values, r["amp"], f"{name} season_amp")
            _same(ds.season_amp_sd.values, r["amp_sd"], f"{name} season_amp_sd")


def test_changepoints_found():
    """A level shift and a change of slope in a seasonal series: found where they are, and sure."""
    rng = np.random.default_rng(0)
    t = np.arange(240)
    y = 0.5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.1, t.size)
    y[120:] += 2.0
    y[180:] += 0.05 * (t[180:] - 180)
    ds = zeit.beast(y, period=12, samples=2000, chains=2)
    found = ds.trend_cp.values[:2]
    assert np.any(np.abs(found - 121) <= 2) and np.any(np.abs(found - 181) <= 6), found
    assert ds.trend_cp_prob.values[0] > 0.9
    assert ds.trend_cp_occ.sum() > 1.5
    assert float(ds.r2) > 0.9
    lower, upper = ds.trend_ci.sel(bound="lower"), ds.trend_ci.sel(bound="upper")
    # The fast interval (Rbeast's default) is of a subsample: the mean can be just out of it at a few dates.
    inside = ((lower <= ds.trend + 1e-6) & (ds.trend <= upper + 1e-6)).mean()
    assert inside > 0.95 and np.all(upper - lower >= 0)
    # Positive slope after the change of slope, flat before the shift.
    assert ds.trend_slope_pos_prob.values[200:230].mean() > 0.9
    assert abs(float(ds.trend_slope.values[40:100].mean())) < 0.01


def test_cube_matches_pixels_and_threads():
    """Each pixel of a cube is fitted as alone, from the seed, whatever the threads."""
    rng = np.random.default_rng(1)
    t = np.arange(96)
    base = np.sin(2 * np.pi * t / 12)
    values = np.stack([base + rng.normal(0, 0.2, t.size) + (t >= 40 + 5 * i) * (i + 1) for i in range(6)])
    values[2, 10:30] = np.nan
    values[4] = np.nan                                     # no data: NaN out
    dates = pd.date_range("2000-01-01", periods=t.size, freq="MS")
    cube = xr.DataArray(values.T.reshape(t.size, 2, 3), dims=("time", "y", "x"),
                        coords={"time": dates, "y": [1.0, 0.0], "x": [0.0, 1.0, 2.0]})
    kw = dict(samples=1000, chains=2, deltat=1 / 12)
    one = zeit.beast(cube, n_jobs=1, **kw)
    many = zeit.beast(cube, n_jobs=4, **kw)
    xr.testing.assert_identical(one, many)
    assert one.trend.dims == ("time", "y", "x") and one.trend_ci.dims == ("bound", "time", "y", "x")
    assert one.attrs["time_units"] == "decimal years"
    for i in range(6):
        yy, xx = divmod(i, 3)
        if i == 4:
            assert np.isnan(one.trend.values[:, yy, xx]).all() and np.isnan(one.marg_lik.values[yy, xx])
            continue
        alone = zeit.beast(pd.Series(values[i], index=dates), **kw)
        np.testing.assert_array_equal(one.trend.values[:, yy, xx], alone.trend.values)
        np.testing.assert_array_equal(one.trend_cp.values[:, yy, xx], alone.trend_cp.values)


def test_lazy_cube():
    pytest.importorskip("dask")
    rng = np.random.default_rng(2)
    values = rng.normal(0, 1, (60, 2, 2)) + np.arange(60)[:, None, None] * 0.05
    cube = xr.DataArray(values, dims=("time", "y", "x")).chunk({"x": 1})
    lazy = zeit.beast(cube, season="none", samples=1000, chains=1)
    eager = zeit.beast(cube.compute(), season="none", samples=1000, chains=1)
    xr.testing.assert_identical(lazy.compute(), eager)


def test_constant_and_short_series():
    # A constant series: the trend is the constant; with no noise the chain wanders over changepoints that do
    # not change it (Rbeast's ncp is also 1.998 here).
    flat = zeit.beast(np.full(50, 3.0), season="none", samples=1000, chains=1)
    np.testing.assert_allclose(flat.trend.values, 3.0, rtol=1e-6)
    assert float(flat.trend_ncp) == pytest.approx(1.998, abs=1e-6)
    tiny = zeit.beast([1.0, 2.0, 1.5, 3.0, 2.5], season="none", samples=1000, chains=1)
    assert tiny.sizes["time"] == 5 and np.isfinite(tiny.trend.values).all()


def test_seed_and_ci_change_the_chain():
    y = CASES["nile"]["y"]
    a = zeit.beast(y, season="none", seed=1, samples=1000, chains=1)
    b = zeit.beast(y, season="none", seed=2, samples=1000, chains=1)
    c = zeit.beast(y, season="none", seed=1, samples=1000, chains=1)
    xr.testing.assert_identical(a, c)
    assert not np.array_equal(a.trend.values, b.trend.values)
    assert "trend_ci" not in zeit.beast(y, season="none", seed=1, samples=1000, chains=1, ci=False)


def test_bad_input():
    y = np.arange(30.0)
    with pytest.raises(ValueError, match="period"):
        zeit.beast(y)
    with pytest.raises(ValueError, match="season"):
        zeit.beast(y, season="dummy")
    with pytest.raises(ValueError, match="prec_type"):
        zeit.beast(y, season="none", prec_type="orderwise")
    with pytest.raises(ValueError, match="tcp_minmax"):
        zeit.beast(y, season="none", tcp_minmax=(3, 1))
    with pytest.raises(ValueError, match="fewer than 2 values per period"):
        zeit.beast(y, period=1.5)
    with pytest.raises(ValueError, match="times"):
        zeit.beast(y, season="none", times=np.arange(10.0))
    # Too many gaps: not fitted.
    gappy = y.copy()
    gappy[:25] = np.nan
    assert np.isnan(zeit.beast(gappy, season="none", samples=1000, chains=1).trend.values).all()
