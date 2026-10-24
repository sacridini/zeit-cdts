"""Emerging hot spot analysis: the Gi* against sfdep 0.2.5 (tests/data/make_hotspots_sfdep_fixtures.R), the trend
of the Gi* against Kendall's Mann-Kendall, and ESRI's patterns on series made for each."""
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import zeit
from zeit import _core

DATA = Path(__file__).parent / "data"


def test_gi_star_and_trend_equal_sfdep():
    ref = json.loads((DATA / "hotspots_sfdep.json").read_text())
    for case in ref["cases"]:
        values = np.array(case["values"], dtype=float)
        nt = values.shape[0]
        da = xr.DataArray(values, dims=("time", "y", "x"), coords={"time": np.arange(2000, 2000 + nt)})
        ds = zeit.emerging_hotspots(da, radius=1, lag=case["k"], trend="original", n_jobs=2)
        np.testing.assert_allclose(ds.gi.values, np.array(case["gi"]), rtol=0, atol=1e-5)   # float32
        gi = _core.hotspots.gi_star(values, radius=1, lag=case["k"], n_jobs=2)
        np.testing.assert_allclose(gi, np.array(case["gi"]), rtol=0, atol=1e-13)
        # Kendall's tau and p are single precision (Fortran)
        np.testing.assert_allclose(ds.tau.values, np.array(case["tau"]), rtol=0, atol=1e-6)
        np.testing.assert_allclose(ds.trend_p.values, np.array(case["p"]), rtol=0, atol=1e-6)


def test_patterns_as_esri_defines_them():
    h, c, o = 3.0, -3.0, 0.1
    rows = [
        ([o] * 9 + [h], 0, False, 1),                  # new
        ([o] * 8 + [h, h], 1, True, 2),                # consecutive
        ([h] * 10, 1, True, 3),                        # intensifying
        ([h] * 4 + [o] + [h] * 5, 0, False, 4),        # persistent (9 of 10)
        ([h] * 10, -1, True, 5),                       # diminishing
        ([o, h, o, o, h, o, o, o, o, h], 0, False, 6),  # sporadic
        ([o, h, o, c, o, o, o, o, h, h], 0, False, 7),  # oscillating
        ([h] * 9 + [o], -1, True, 8),                  # historical
        ([o] * 9 + [c], 0, False, -1),
        ([c] * 10, -1, True, -3),
        ([o, h, o, o, h, o, c, o, o, o], 0, False, 0),
    ]
    gi = np.array([r[0] for r in rows])
    got = _core.hotspots.classify(gi, np.array([r[1] for r in rows], dtype=np.int32), np.array([r[2] for r in rows]))
    np.testing.assert_array_equal(got, [r[3] for r in rows])
    assert {k: v for k, v in zeit.emerging_hotspots.PATTERNS.items() if k in (1, -8)} == {
        1: "new hot spot", -8: "historical cold spot"}


def _cube(nt=12, ny=30, nx=40, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(nt, ny, nx))
    v[:, 5:12, 5:12] += 2.5                                        # always high: persistent
    v[:, 18:25, 22:30] += np.linspace(0, 4, nt)[:, None, None]      # rising: an emerging hot spot
    v[-3:, 3:9, 30:37] -= 3.0                                       # low at the last dates: a cold spot
    return xr.DataArray(v, dims=("time", "y", "x"),
                        coords={"time": [np.datetime64(f"{2000 + t}-07-01") for t in range(nt)],
                                "y": np.arange(ny) * -30.0, "x": np.arange(nx) * 30.0})


def test_patterns_found_on_a_cube():
    ds = zeit.emerging_hotspots(_cube(), radius=2, shape="circle")
    cat = ds.category.values
    assert cat.dtype == np.int8 and ds.category.attrs["flag_values"][0] == -8
    assert np.isin(cat[7:10, 7:10], [3, 4, 5]).all()                 # hot on every date
    assert np.isin(cat[20:23, 24:28], [1, 2, 3, 6]).mean() > 0.8     # emerging
    assert (ds.trend_z.values[20:23, 24:28] > 0).all()
    assert np.isin(cat[5:7, 32:35], [-1, -2]).all()                  # a new / consecutive cold spot
    assert ds.n_hot.values[8, 8] >= 11 and ds.n_cold.values[6, 33] >= 2
    # the whole cube's mean: a rise everywhere makes the last dates hot everywhere
    # (without a lag: the earlier dates in a neighbourhood are measured against the date's mean, as sfdep's)
    rise = _cube() + xr.DataArray(np.linspace(0, 3, 12), dims="time")
    flat = zeit.emerging_hotspots(_cube(), radius=2, shape="circle", lag=0)
    by_date = zeit.emerging_hotspots(rise, radius=2, shape="circle", lag=0)
    by_cube = zeit.emerging_hotspots(rise, radius=2, shape="circle", lag=0, reference="cube")
    np.testing.assert_allclose(by_date.gi, flat.gi, atol=1e-5)        # each date's own mean: unchanged
    assert (by_cube.gi.values[-1] > by_date.gi.values[-1]).mean() > 0.95


def test_missing_values_and_errors():
    da = _cube()
    v = da.values.copy()
    v[:, 0, 0] = np.nan
    v[3, 10, 10] = np.nan
    ds = zeit.emerging_hotspots(xr.DataArray(v, dims=da.dims, coords=da.coords))
    assert ds.category.values[0, 0] == -128 and np.isnan(ds.gi.values[:, 0, 0]).all()
    assert np.isnan(ds.gi.values[3, 10, 10]) and np.isfinite(ds.gi.values[3, 10, 11])
    assert ds.category.values[10, 10] != -128
    with pytest.raises(ValueError, match="shape"):
        zeit.emerging_hotspots(da, shape="hexagon")
    with pytest.raises(ValueError, match="trend"):
        zeit.emerging_hotspots(da, trend="sen")
    with pytest.raises(ValueError, match="one pixel"):
        zeit.emerging_hotspots(da.isel(y=0, x=0))
