"""zeit.ewmacd: the flags and the EWMA of Brooks' EWMACD v1.3.0 script (tests/data/
make_ewmacd_parity_fixtures.R), the events of known changes, persistence, a pixel and a cube."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _core

PARITY = Path(__file__).parent / "data" / "ewmacd_brooks_parity.json"


def _carried(z, charted):
    """The script's rounding=FALSE output: round(z) at the charted dates, carried over the others
    (0 for a first date left out)."""
    out = np.where(charted, np.round(z), np.nan)
    if np.isnan(out[0]):
        out[0] = 0.0
    for j in range(1, len(out)):
        if np.isnan(out[j]):
            out[j] = out[j - 1]
    return out


def test_flags_and_ewma_of_brooks_script():
    d = json.loads(PARITY.read_text())
    years = np.array(d["years"], dtype=np.int32)
    doys = np.array(d["doys"], dtype=np.float64)
    t = years + doys / 365.0
    values = np.array([[np.nan if v is None else v for v in p] for p in d["values"]], dtype=np.float64)
    for case in d["cases"]:
        kw = dict(harmonics=case["harmonics"], lmbda=case["lmbda"], persistence=1)
        flags, _ = _core.ewmacd.ewmacd_batch(values, years, doys, t, d["train_start"], d["train_end"], **kw)
        np.testing.assert_array_equal(flags, np.array(case["flags"], dtype=np.float64))
        for i, v in enumerate(values):
            e = _core.ewmacd.ewmacd_pixel(v, years, doys, t, d["train_start"], d["train_end"], **kw)
            np.testing.assert_array_equal(_carried(e["ewma"], e["charted"]), case["ewma"][i])


def _series(change=lambda t: 0.0 * t, noise=25.0, seed=3):
    rng = np.random.default_rng(seed)
    days = [np.datetime64(f"{y}-01-01") + np.timedelta64(int(d), "D")
            for y in range(2005, 2013) for d in rng.choice(365, 14, replace=False)]
    dates = pd.DatetimeIndex(sorted(days))
    t = zeit._time.fractional_years(dates)
    th = 2 * np.pi * dates.dayofyear.to_numpy() / 365
    v = 500 + 120 * np.sin(th) - 60 * np.cos(th) + rng.normal(0, noise, len(t)) + change(t)
    return pd.Series(v, index=dates)


def test_a_thinning_and_its_recovery():
    s = _series(lambda t: np.where((t >= 2009.5) & (t < 2011.0), -90.0, 0.0))
    s.iloc[5] -= 400  # a cloud in the training years
    res = zeit.ewmacd(s, train_end=2009)
    assert 2009.5 <= float(res.break_time) < 2009.8
    assert float(res.severity) <= -2  # 90 = 3.6 sd: 2.8 control limits of the EWMA
    assert 2011.0 <= float(res.recovery_time) < 2011.6
    assert int(res.state) == 2
    assert res.flags.sel(time=slice("2006", "2008")).max() == 0
    assert np.isnan(res.ewma.values[5])  # the cloud left out of the chart
    assert res.coefficients.sel(coefficient="constant") == pytest.approx(500, abs=10)
    assert float(res.r2) > 0.8


def test_gain_and_stable():
    assert int(zeit.ewmacd(_series(lambda t: np.where(t >= 2010, 100.0, 0.0)), train_end=2009).state) == 3
    stable = zeit.ewmacd(_series(seed=4), train_end=2009, persistence=4)
    assert int(stable.state) == 0 and np.isnan(float(stable.break_time))


def test_persistence_drops_a_short_signal():
    s = _series(seed=5)
    s[(s.index >= "2010-03-01") & (s.index < "2010-04-15")] -= 150  # two or three dates
    brief = zeit.ewmacd(s, train_end=2009, persistence=1)
    assert np.isfinite(float(brief.break_time))
    lasting = zeit.ewmacd(s, train_end=2009, persistence=6)
    assert np.isnan(float(lasting.break_time))


def test_cube_matches_the_pixels():
    series = [_series(lambda t: np.where(t >= 2010.3, -120.0, 0.0), seed=1), _series(seed=2),
              _series(lambda t: np.where(t >= 2011, 80.0, 0.0), seed=1), _series(seed=6)]
    # one date axis for all: the first's dates
    dates = series[0].index
    stack = np.stack([s.to_numpy() for s in series], axis=1).reshape(len(dates), 2, 2)
    stack[10, 1, 1] = np.nan
    cube = xr.DataArray(stack, dims=("time", "y", "x"),
                        coords={"time": dates, "y": [1.5, 0.5], "x": [0.5, 1.5]})
    res = zeit.ewmacd(cube, train_end=2009)
    assert res.flags.dims == ("time", "y", "x") and res.coefficients.dims == ("coefficient", "y", "x")
    for k in range(4):
        y, x = divmod(k, 2)
        one = zeit.ewmacd(pd.Series(stack[:, y, x], index=dates), train_end=2009)
        np.testing.assert_array_equal(res.flags.values[:, y, x], one.flags.values.astype(np.float32))
        for name in ("break_time", "state", "severity"):
            np.testing.assert_allclose(float(res[name].values[y, x]), float(one[name]), equal_nan=True)
    assert res.state.values[0, 0] == 1 and res.state.values[1, 0] == 3


def test_arguments():
    s = _series()
    with pytest.raises(ValueError, match="lmbda"):
        zeit.ewmacd(s, train_end=2009, lmbda=0)
    with pytest.raises(ValueError, match="train_end"):
        zeit.ewmacd(s, train_end=2005, train_start=2006)
    res = zeit.ewmacd(s, train_end=2009, train_start=2006)
    assert res.flags.sel(time="2005").isnull().all()
    few = zeit.ewmacd(s.iloc[:4], train_end=2009)
    assert few.flags.isnull().all() and np.isnan(float(few.state))
