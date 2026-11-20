"""zeit.cva: change vector analysis, its directions and thresholds."""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _cva

NAMES = ["blue", "green", "red", "nir08", "swir16", "swir22"]


def _scene(size=200, seed=7):
    """Background reflectance with noise between dates; block A cleared (red up, NIR down, SWIR1 up), block B flooded
    (every band down), block C one band shifted."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.02, 0.3, (6, size, size))
    after = base + rng.normal(0, 0.003, base.shape)
    a, b, c = (slice(10, 50), slice(10, 50)), (slice(10, 50), slice(100, 140)), (slice(100, 140), slice(10, 50))
    after[(2,) + a] += 0.05
    after[(3,) + a] -= 0.15
    after[(4,) + a] += 0.10
    after[(slice(None),) + b] -= 0.08
    after[(4,) + c] += 0.10
    truth = np.zeros((size, size), bool)
    for blk in (a, b, c):
        truth[blk] = True
    coords = {"band": NAMES, "y": np.arange(size) * 30.0, "x": np.arange(size) * 30.0}
    mk = lambda v: xr.DataArray(v, dims=("band", "y", "x"), coords=coords).rio.write_crs("EPSG:32620")
    return mk(base), mk(after), truth, (a, b, c)


@pytest.fixture(scope="module")
def scene():
    return _scene()


def test_c2va_formula_and_special_angles(scene):
    x, y, _, _ = scene
    out = zeit.cva(x, y, threshold=0.1, sample=None)
    assert out.dtype == np.float32 and list(out.band.values) == ["magnitude", "direction", "change"]
    d = y.values - x.values
    rho = np.sqrt((d ** 2).sum(0))
    alpha = np.degrees(np.arccos(np.clip(d.sum(0) / (np.sqrt(6) * rho), -1, 1)))
    np.testing.assert_allclose(out.sel(band="magnitude").values, rho, atol=1e-6)
    np.testing.assert_allclose(out.sel(band="direction").values, alpha, atol=1e-4)
    zero = xr.zeros_like(x[:, :6, :6])
    for delta, want in ((np.array([.2, .2, .2, .2, .2, .2]), 0.0), (-np.array([.2] * 6), 180.0),
                        (np.array([.3, 0, 0, 0, 0, 0]), np.degrees(np.arccos(1 / np.sqrt(6)))),
                        (np.array([.1, -.1, .1, -.1, .1, -.1]), 90.0)):
        got = zeit.cva(zero, zero + delta[:, None, None], threshold=0.0, sample=None)
        np.testing.assert_allclose(got.sel(band="direction").values, want, atol=1e-3)
    same = zeit.cva(zero, zero, threshold=0.5, sample=None)
    assert np.isnan(same.sel(band="direction").values).all() and (same.sel(band="change").values == 0).all()


def test_polar_and_the_numpy_reference(scene):
    x, y, _, _ = scene
    out = zeit.cva(x, y, direction="polar", polar_bands=["blue", "green"], n_sectors=4, threshold=0.05, sample=None)
    assert list(out.band.values) == ["magnitude", "direction", "change", "sector"]
    d = y.values - x.values
    theta = np.degrees(np.arctan2(d[1], d[0])) % 360
    np.testing.assert_allclose(out.sel(band="direction").values, theta, atol=1e-4)
    sector = out.sel(band="sector").values
    changed = out.sel(band="change").values == 1
    np.testing.assert_array_equal(sector[changed], np.floor(theta[changed] / 90) + 1)
    assert (sector[~changed] == 0).all()
    ref = _cva._block_numpy(x.values, y.values, list(range(6)), [1.0] * 6,
                            {"direction": "polar", "polar_bands": (0, 1), "n_sectors": 4}, 0.05, [])
    np.testing.assert_allclose(out.values, ref, atol=1e-4, equal_nan=True)
    table = zeit.cva_table(out)
    assert list(table["sector"]) == [0, 1, 2, 3, 4] and table["pixels"].sum() == 200 * 200
    assert table.loc[0, "label"] == "No change" and table["area_ha"].notna().all()


@pytest.mark.parametrize("threshold", ["otsu", "em"])
def test_known_changes_by_sign_sectors(scene, threshold):
    x, y, truth, (a, b, c) = scene
    out = zeit.cva(x, y, direction="sign", threshold=threshold, sample=None)
    change = out.sel(band="change").values == 1
    sector = out.sel(band="sector").values
    assert (sector[a] == 6)[change[a]].mean() > 0.99 and change[a].mean() > 0.99          # vegetation loss
    assert (sector[b] == 1)[change[b]].mean() > 0.99 and change[b].mean() > 0.99          # darker in all bands
    assert change[c].mean() > 0.99
    assert change[~truth].mean() < 0.005
    record = json.loads(out.attrs["cva"])
    assert record["threshold"]["method"] == threshold
    assert "Vegetation loss" in zeit.cva_table(out).set_index("sector").loc[6, "label"]
    ref = _cva._block_numpy(x.values, y.values, list(range(6)), [1.0] * 6,
                            {"direction": "sign", "roles": (2, 3, 4)}, record["threshold"]["threshold"], [])
    np.testing.assert_allclose(out.values, ref, atol=1e-4, equal_nan=True)


def test_c2va_sectors_scale_and_fixed_threshold(scene):
    x, y, _, _ = scene
    out = zeit.cva(x, y, c2va_sectors=[0, 60, 120, 180], threshold=0.05, sample=None)
    assert list(out.band.values)[-1] == "sector"
    ang = out.sel(band="direction").values
    sec = out.sel(band="sector").values
    ch = out.sel(band="change").values == 1
    np.testing.assert_array_equal(sec[ch], np.searchsorted([60, 120], ang[ch], side="right") + 1)
    scaled = zeit.cva(x, y, scale="std", threshold=1.0, sample=None)
    record = json.loads(scaled.attrs["cva"])
    sd = (y.values - x.values).reshape(6, -1).std(1)
    np.testing.assert_allclose(record["scale"], 1 / sd, rtol=1e-9)
    assert record["threshold"]["method"] == "fixed" and record["threshold"]["threshold"] == 1.0


def test_nodata_lazy_dates_and_errors(scene):
    x, y, _, _ = scene
    xv = x.values.astype(np.float32)
    yv = y.values.astype(np.float32)
    xv[3, 4, 5] = np.nan
    yv[1, 8, 9] = -9999
    cube = xr.concat([x.copy(data=xv), y.copy(data=yv)], dim="time").assign_coords(
        time=pd.to_datetime(["2020-07-01", "2021-07-01"]))
    cube = cube.transpose("time", "band", "y", "x")
    out = zeit.cva(cube, dates=("2020-07-01", "2021-07-01"), nodata=-9999, threshold=0.1, sample=None)
    assert np.isnan(out.values).sum(axis=(1, 2)).tolist() == [2, 2, 2]
    lazy = zeit.cva(cube.chunk({"y": 50, "x": 70}), dates=(0, 1), nodata=-9999, threshold=0.1, sample=None)
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, out.values)
    a = zeit.cva(x, y, threshold="otsu", n_jobs=1)
    b = zeit.cva(x, y, threshold="otsu", n_jobs=4)
    np.testing.assert_array_equal(a.values, b.values)
    with pytest.raises(ValueError, match="red, NIR and SWIR1"):
        zeit.cva(x.isel(band=[0, 1, 2]), y.isel(band=[0, 1, 2]), direction="sign")
    with pytest.raises(ValueError, match="direction"):
        zeit.cva(x, y, direction="rose")
    with pytest.raises(ValueError, match="threshold"):
        zeit.cva(x, y, threshold="max")
    with pytest.raises(ValueError, match="polar_bands"):
        zeit.cva(x, y, direction="polar")
