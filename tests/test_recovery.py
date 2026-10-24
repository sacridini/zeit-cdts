"""zeit.recovery_metrics: the metrics of a known fire and regrowth, against the formulas written out
in numpy, on a pixel, a cube and a LandTrendr Dataset."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _core

YEARS = np.arange(2000, 2021)


def _fire(pre=0.6, low=0.1, fire=2005, rate=0.05):
    return np.where(YEARS < fire, pre, np.minimum(low + rate * (YEARS - fire), pre))


def _reference(values, start, trough, target=0.8, horizon=5, pre_years=2):
    """The metrics of a loss with the pre-value basis, written out."""
    at = dict(zip(YEARS.tolist(), values))
    pre = np.nanmean([at.get(y, np.nan) for y in range(start - pre_years + 1, start + 1)])
    dist = at[trough]
    post = np.nanmax([at.get(trough + horizon - 1, np.nan), at.get(trough + horizon, np.nan)])
    level = target * pre
    later = [y for y in YEARS if y >= trough and at[y] >= level]
    return dict(pre_val=pre, dist_val=dist, magnitude=pre - dist, post_val=post, ari=post - dist,
                rri=(post - dist) / (pre - dist), r80p=post / level,
                yryr=(at[trough + horizon] - dist) / horizon, y2r=later[0] - trough if later else np.nan)


def test_core_matches_the_formulas():
    values = _fire()
    out = _core.recovery.recovery_batch(values[None, :], YEARS, [2004], [2005])[:, 0]
    ref = _reference(values, 2004, 2005)
    for i, name in enumerate(_core.recovery.NAMES):
        assert out[i] == pytest.approx(ref[name], abs=1e-12), name
    assert out[-1] == 8  # 0.1 + 0.05 k >= 0.48


def test_pixel_series():
    rec = zeit.recovery_metrics(_fire(), 2004, years=YEARS)
    assert float(rec.y2r) == 8
    assert float(rec.rri) == pytest.approx(0.5, rel=1e-6)
    assert int(rec.yod) == 2004 and int(rec.trough) == 2005
    mag = zeit.recovery_metrics(_fire(), 2004, years=YEARS, basis="magnitude")
    assert float(mag.r80p) == pytest.approx(0.25 / 0.4, rel=1e-6)
    with pytest.raises(ValueError, match="magnitude"):
        zeit.recovery_metrics(_fire(), 2004, years=YEARS, event_type="gain")


def test_cube_with_a_map_of_events():
    dates = pd.DatetimeIndex([pd.Timestamp(y, 7, 1) for y in YEARS])
    stack = np.stack([_fire(fire=2005), _fire(fire=2010, rate=0.1), _fire(fire=2015), _fire()], axis=1)
    cube = xr.DataArray(stack.reshape(len(YEARS), 2, 2), dims=("time", "y", "x"),
                        coords={"time": dates, "y": [1.5, 0.5], "x": [0.5, 1.5]})
    starts = np.array([[2004, 2009], [2014, 0]])
    rec = zeit.recovery_metrics(cube, starts)
    assert rec.y2r.dims == ("y", "x")
    assert rec.y2r.values[0, 0] == 8 and rec.y2r.values[0, 1] == 4
    # the fire of 2015: its year 5 is the last year, measured; no event: NaN
    assert np.isfinite(rec.ari.values[1, 0])
    assert np.isnan(rec.ari.values[1, 1]) and rec.yod.values[1, 1] == 0
    lazy = zeit.recovery_metrics(cube.chunk({"y": 1}), xr.DataArray(starts, dims=("y", "x")))
    np.testing.assert_array_equal(lazy.y2r.compute().values, rec.y2r.values)


def test_landtrendr_dataset():
    rng = np.random.default_rng(1)
    series = _fire() + rng.normal(0, 0.01, len(YEARS))
    lt = zeit.landtrendr(series, years=YEARS, direction="loss")
    ev = zeit.extract_events(lt)
    rec = zeit.recovery_metrics(lt)
    assert int(rec.yod) == int(ev.yod)
    assert int(rec.trough) == int(ev.yod) + int(ev.duration)
    # on the line through the vertices
    n = int(lt.n_vertices)
    fitted = np.interp(YEARS, lt.vertex_year.values[:n], lt.vertex_value.values[:n])
    ref = _reference(fitted, int(ev.yod), int(ev.yod) + int(ev.duration))
    for name in ("pre_val", "dist_val", "post_val", "ari", "rri", "r80p", "y2r"):
        assert float(rec[name]) == pytest.approx(ref[name], rel=1e-5, abs=1e-6), name
    # the same with LandTrendr's own fitted series
    fit = zeit.recovery_metrics(zeit.landtrendr(series, years=YEARS, direction="loss", fitted=True))
    assert float(fit.y2r) == float(rec.y2r)
