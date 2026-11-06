"""zeit.spectral_adjust: FORCE's spectral adjustment of Landsat and MODIS to Sentinel-2A."""
import math
import re
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _core
from zeit._specadjust import TARGET_BANDS, sensor_of

TABLES = Path(__file__).resolve().parents[1] / "rust" / "zeit-core" / "src" / "specadjust_tables.rs"
SENSORS = ["LND04", "LND05", "LND07", "LND08", "LND09", "MOD01", "MOD02"]
SOURCE = ["blue", "green", "red", "nir08", "swir16", "swir22"]
POOR, MEDI, GOOD = 0.2617995, 0.2094395, 0.0698132


@pytest.fixture(scope="module")
def tables():
    """FORCE's tables, parsed from the generated Rust file (the repository's, not the wheel's)."""
    if not TABLES.exists():
        pytest.skip("rust/zeit-core/src/specadjust_tables.rs is not in this tree")
    text = TABLES.read_text(encoding="utf-8")
    centre_text, coef_text = text.split("pub static COEFS")
    rows = re.findall(r"^\s+\[([-0-9., e]+)\],$", centre_text.split("pub static CENTER")[1], re.M)
    centre = np.array([[float(v) for v in r.split(",")] for r in rows]).reshape(7, 6, 51)
    rows = re.findall(r"^\s+\[([-0-9., e]+)\],$", coef_text, re.M)
    coefs = np.array([[float(v) for v in r.split(",")] for r in rows]).reshape(7, 10, 7, 51)
    return centre, coefs


def literal(tables, sensor, x, force):
    """FORCE's spectral_predict for one pixel, line by line (spec-adjust-hl.c:822-960), in Python floats."""
    centre, coefs = tables
    sid = SENSORS.index(sensor)
    weight = [0.0] * 51
    max_weight, max_cluster = -1.0, 0
    for s in range(51):
        xy = xx = yy = 0.0
        for b in range(6):
            c = centre[sid, b, s]
            xy += x[b] * c
            xx += x[b] * x[b]
            yy += c * c
        if xx > 0 and yy > 0:
            cosine = xy / math.sqrt(xx * yy)
            sam = math.acos(cosine) if abs(cosine) <= 1 else float("nan")      # C's acos: NaN beyond 1
        else:
            sam = 1.0
        weight[s] = 0.00001 if sam > POOR else 1.0 - sam / POOR
        if weight[s] > max_weight and (s != 9 if force else s != 50):
            max_weight, max_cluster = weight[s], s
    min_weight, med_weight = 1.0 - GOOD / POOR, 1.0 - MEDI / POOR
    cluster_select, weight_select = [0] * 10, [0.0] * 10
    c = 0
    while c < 10:
        cluster_select[c], weight_select[c] = 50, weight[50]
        if max_weight > min_weight or (max_weight > med_weight and max_weight > weight[50]):
            cluster_select[c], weight_select[c] = max_cluster, max_weight
            weight[max_cluster] = 0.0
        else:
            break
        max_weight = -1.0
        for s in range(50):
            if weight[s] > max_weight:
                max_weight, max_cluster = weight[s], s
        c += 1
    n_cluster = min(c + 1, 10)
    wpred, wsum = [0.0] * 10, 0.0
    for c in range(n_cluster):
        s = cluster_select[c]
        for b in range(10):
            pred = 0.0
            for b_ in range(6):
                pred += coefs[sid, b, b_, s] * x[b_]
            pred += coefs[sid, b, 6, s]
            wpred[b] += weight_select[c] * pred
        wsum += weight_select[c]
    return np.array([w / wsum for w in wpred])


def spectra(n, seed=0):
    """Realistic reflectance x 10000 (vegetation, soil, water, snow, urban) and the edge cases of FORCE's arithmetic."""
    rng = np.random.default_rng(seed)
    kinds = np.array([[400, 700, 500, 3800, 1800, 900], [1200, 1700, 2200, 2900, 3800, 3200],
                      [600, 700, 400, 200, 100, 50], [8000, 8200, 8000, 7000, 1500, 900],
                      [1500, 1700, 1900, 2300, 2600, 2200]], float)
    x = kinds[rng.integers(0, 5, n)] * rng.uniform(0.5, 1.5, (n, 1)) * rng.uniform(0.9, 1.1, (n, 6))
    x = np.floor(x)
    edge = np.array([[0, 0, 0, 0, 0, 0], [-30, -10, -20, -5, -1, -3], [0, 0, 0, 0, 0, 1], [12000, 11000, 15000, 20000, 18000, 16000],
                     [100, 100, 100, 100, 100, 100], [5000, 5000, 5000, 5000, 5000, 5000], [0, 500, 0, 3000, 0, 800]], float)
    return np.vstack([x, edge])


@pytest.mark.parametrize("sensor", ["LND08", "LND05", "MOD02", "MOD01", "LND04", "LND07"])
def test_the_core_is_forces_algorithm(tables, sensor):
    x = spectra(250, seed=SENSORS.index(sensor))
    for force in (False, True):
        got = _core.specadjust.specadjust_batch(x, sensor, 2, force)
        want = np.array([literal(tables, sensor, row, force) for row in x])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-9)
        # what FORCE writes: the truncation of the same numbers
        assert (np.trunc(got) == np.trunc(want)).mean() > 0.9999


def test_lnd09_is_lnd08_and_float32_is_float64():
    x = spectra(300, seed=11)
    a = _core.specadjust.specadjust_batch(x, "LND08", 1)
    np.testing.assert_array_equal(a, _core.specadjust.specadjust_batch(x, "LND09", 1))
    np.testing.assert_array_equal(a, _core.specadjust.specadjust_batch(x.astype(np.float32), "LND08", 1))
    assert not np.array_equal(a, _core.specadjust.specadjust_batch(x, "LND07", 1))


def test_the_s_not_9_fix(tables):
    """FORCE's first search skips cluster 9 (``s != 9``) and lets the global cluster be the first pick; zeit's does
    neither. Pixels that sit on a cluster-9 centre-like spectrum differ, and each mode is its own literal port."""
    rng = np.random.default_rng(5)
    centre = tables[0][3]                                           # LND08: [band, cluster]
    # spectra near the centre of cluster 9 and of the global cluster, where the two searches part
    near = np.vstack([np.floor(centre[:, c][None, :] * rng.uniform(0.7, 1.3, (80, 1)) + rng.normal(0, 40, (80, 6)))
                      for c in (9, 50)])
    fixed = _core.specadjust.specadjust_batch(near, "LND08", 1, False)
    force = _core.specadjust.specadjust_batch(near, "LND08", 1, True)
    differ = (fixed != force).any(axis=1)
    assert differ.sum() > 5
    for i in np.flatnonzero(differ)[:20]:
        np.testing.assert_allclose(force[i], literal(tables, "LND08", near[i], True), rtol=1e-12)
        np.testing.assert_allclose(fixed[i], literal(tables, "LND08", near[i], False), rtol=1e-12)
    # far from cluster 9 and the global one, the two agree (cluster 3's centre, whose weight is 1)
    exact = centre[:, 3][None, :]
    np.testing.assert_array_equal(_core.specadjust.specadjust_batch(exact, "LND08", 1, False),
                                  _core.specadjust.specadjust_batch(exact, "LND08", 1, True))


def test_nan_and_edges_in_the_core():
    x = spectra(20)
    x[3, 2] = np.nan
    out = _core.specadjust.specadjust_batch(x, "LND08", 1)
    assert np.isnan(out[3]).all() and np.isfinite(np.delete(out, 3, axis=0)).all()
    assert _core.specadjust.specadjust_batch(x[:0], "LND08", 1).shape == (0, 10)
    with pytest.raises(ValueError, match="6"):
        _core.specadjust.specadjust_batch(np.zeros((3, 5)), "LND08", 1)
    with pytest.raises(ValueError, match="unknown sensor"):
        _core.specadjust.specadjust_batch(np.zeros((3, 6)), "SEN2A", 1)


@pytest.mark.parametrize("name, sensor", [
    ("landsat-4", "LND04"), ("LANDSAT_5", "LND05"), ("landsat-7", "LND07"), ("Landsat 8", "LND08"), ("landsat-9", "LND09"),
    ("LC08", "LND08"), ("LC09", "LND09"), ("LE07", "LND07"), ("LT05", "LND05"), ("LT04", "LND04"), ("LND08", "LND08"),
    ("lnd04", "LND04"), ("terra", "MOD01"), ("MOD01", "MOD01"), ("MOD09GA", "MOD01"), ("aqua", "MOD02"), ("MOD02", "MOD02"),
    ("MYD09GA", "MOD02"), ("sentinel-2a", "SEN2"), ("Sentinel-2B", "SEN2"), ("S2C", "SEN2"),
    ("modis", None), ("TM", None), ("landsat", None), ("spot-7", None), ("LND06", None),
])
def test_sensor_names(name, sensor):
    assert sensor_of(name) == sensor


def _cube(platforms, bands=None, h=4, w=5, seed=0, scale=10000.0, dtype=np.float32):
    bands = bands or SOURCE + ["rededge1", "rededge2", "rededge3", "nir"]
    x = spectra(len(platforms) * h * w, seed=seed)[:len(platforms) * h * w].astype(float)
    data = x.reshape(len(platforms), h, w, 6)[..., [0, 1, 2, 3, 4, 5]].transpose(0, 3, 1, 2)
    rng = np.random.default_rng(seed + 1)
    extra = rng.uniform(100, 5000, (len(platforms), len(bands) - 6, h, w))
    data = np.concatenate([data, extra], axis=1) * (scale / 10000.0)
    da = xr.DataArray(data.astype(dtype), dims=("time", "band", "y", "x"), name="reflectance",
                      coords={"time": pd.date_range("2020-01-01", periods=len(platforms), freq="10D"), "band": bands,
                              "y": np.arange(h)[::-1] * 30.0 + 15, "x": np.arange(w) * 30.0 + 15,
                              "platform": ("time", list(platforms))})
    return da.rio.write_crs(32722)


def test_landsat_dates_become_sentinel2a_and_sentinel2_passes_through(tables):
    cube = _cube(["landsat-8", "sentinel-2a", "landsat-5", "terra", "landsat-9"])
    out = zeit.spectral_adjust(cube)
    assert list(out.band.values) == list(TARGET_BANDS) and out.dims == cube.dims
    assert out.dtype == np.float32 and out.rio.crs == cube.rio.crs
    assert (out.y == cube.y).all() and (out.x == cube.x).all() and (out.platform == cube.platform).all()
    for t, sensor in ((0, "LND08"), (2, "LND05"), (3, "MOD01"), (4, "LND09")):
        src = cube.isel(time=t).sel(band=SOURCE).transpose("y", "x", "band").values.reshape(-1, 6).astype(float)
        want = np.array([literal(tables, sensor, row, False) for row in src])
        got = out.isel(time=t).transpose("y", "x", "band").values.reshape(-1, 10)
        np.testing.assert_allclose(got, want, rtol=2e-6, atol=1e-2)
    # Sentinel-2: the ten bands taken from the cube by name, untouched
    s2 = cube.isel(time=1)
    for band in ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir"):
        np.testing.assert_array_equal(out.isel(time=1).sel(band=band), s2.sel(band=band))
    np.testing.assert_array_equal(out.isel(time=1).sel(band="nir08"), s2.sel(band="nir08"))
    np.testing.assert_array_equal(out.isel(time=1).sel(band="swir16"), s2.sel(band="swir16"))
    assert "LND08 -> Sentinel-2A (1 date)" in out.attrs["spectral_adjusted"]
    assert "Sentinel-2 passed through" in out.attrs["spectral_adjusted"]
    # the band names are those zeit.indices finds
    ndvi = zeit.compute_indices(out, ["NDVI"])
    assert "NDVI" in [str(b) for b in ndvi.band.values]


def test_scale_is_converted_and_back():
    a = _cube(["landsat-8", "landsat-7"], scale=10000.0, dtype=np.float64)
    b = _cube(["landsat-8", "landsat-7"], scale=1.0, dtype=np.float64)
    ya, yb = zeit.spectral_adjust(a), zeit.spectral_adjust(b)
    np.testing.assert_allclose(yb.values * 10000.0, ya.values, rtol=2e-6, atol=1e-2)
    c = _cube(["landsat-8", "landsat-7"], scale=10000.0, dtype=np.float64)
    np.testing.assert_allclose(zeit.spectral_adjust(c, scale=10000).values, ya.values)
    np.testing.assert_allclose(zeit.spectral_adjust(b, scale=1).values, yb.values)


def test_lazy_equals_eager_and_stays_lazy():
    cube = _cube(["landsat-8", "sentinel-2a", "landsat-7", "landsat-8"], h=6, w=7)
    eager = zeit.spectral_adjust(cube)
    lazy = zeit.spectral_adjust(cube.chunk({"time": 1, "band": 3, "y": 3, "x": 4}))
    assert lazy.chunks is not None
    with dask.config.set(scheduler="threads", num_workers=2):
        np.testing.assert_array_equal(lazy.compute().values, eager.values)
    assert lazy.dtype == np.float32 and lazy.dims == eager.dims


def test_nodata_and_nan_are_missing_in_all_ten_bands():
    cube = _cube(["landsat-8", "landsat-8"])
    cube[0, 0, 1, 2] = np.nan
    cube[1, 3, 0, 0] = 0.0          # no NoData stored, a float cube: only NaN is missing
    out = zeit.spectral_adjust(cube)
    assert out.isel(time=0, y=1, x=2).isnull().all()
    assert out.isel(time=1, y=0, x=0).notnull().all()
    out = zeit.spectral_adjust(cube, nodata=0.0)
    assert out.isel(time=1, y=0, x=0).isnull().all()


def test_no_time_dimension_and_sensor_argument(tables):
    cube = _cube(["landsat-8"]).isel(time=0, drop=True)
    out = zeit.spectral_adjust(cube, sensor="LC08")
    assert out.dims == ("band", "y", "x") and out.sizes["band"] == 10
    with pytest.raises(ValueError, match="needs each date's sensor"):
        zeit.spectral_adjust(cube)
    multi = _cube(["landsat-8", "landsat-5"]).drop_vars("platform")
    both = zeit.spectral_adjust(multi, sensor=["LND08", "LND05"])
    same = zeit.spectral_adjust(_cube(["landsat-8", "landsat-5"]))
    np.testing.assert_array_equal(both.values, same.values)


def test_errors_name_the_date_and_the_missing_bands():
    cube = _cube(["landsat-8", "modis", "landsat-8"])
    with pytest.raises(ValueError, match=r"unknown sensor 'modis'.*2020-01-11.*Terra"):
        zeit.spectral_adjust(cube)
    cube = _cube(["landsat-8"], bands=SOURCE).sel(band=SOURCE[:5])
    with pytest.raises(ValueError, match="swir2"):
        zeit.spectral_adjust(cube)
    with pytest.raises(ValueError, match="unknown role"):
        zeit.spectral_adjust(_cube(["landsat-8"]), bands={"coastal": "blue"})
    s2 = _cube(["sentinel-2a"], bands=["blue", "green", "red", "nir08", "swir16", "swir22"])
    with pytest.raises(ValueError, match="rededge1"):
        zeit.spectral_adjust(s2)
    done = zeit.spectral_adjust(_cube(["landsat-8"]))
    with pytest.raises(ValueError, match="already spectrally adjusted"):
        zeit.spectral_adjust(done)


def test_bands_by_role_and_the_accessor():
    cube = _cube(["landsat-8", "landsat-8"], bands=["b1", "b2", "b3", "b4", "b5", "b6"])
    named = zeit.spectral_adjust(cube, bands=dict(zip(["blue", "green", "red", "nir", "swir1", "swir2"],
                                                       ["b1", "b2", "b3", "b4", "b5", "b6"])))
    usual = zeit.spectral_adjust(_cube(["landsat-8", "landsat-8"], bands=SOURCE))
    np.testing.assert_array_equal(named.values, usual.values)
    np.testing.assert_array_equal(_cube(["landsat-8", "landsat-8"], bands=SOURCE).zeit.spectral_adjust().values,
                                  usual.values)


def test_force_search_is_the_cores_literal_mode():
    import numpy as np
    import xarray as xr
    import zeit
    from zeit import _core

    rng = np.random.default_rng(9)
    x = rng.uniform(0, 5000, size=(1, 6, 40, 50)).astype(np.float32)
    cube = xr.DataArray(x, dims=("time", "band", "y", "x"),
                        coords={"band": ["blue", "green", "red", "nir08", "swir16", "swir22"],
                                "time": [np.datetime64("2020-01-01")]})
    got = zeit.spectral_adjust(cube, sensor="landsat-8", scale=10000, force_search=True, n_jobs=2)
    flat = np.moveaxis(x[0], 0, -1).reshape(-1, 6)
    want = _core.specadjust.specadjust_batch(flat, "LND08", 2, True)
    np.testing.assert_allclose(np.moveaxis(got.values[0], 0, -1).reshape(-1, 10), want, rtol=1e-6)
