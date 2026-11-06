"""zeit.tasseled_cap: the Tasseled Cap of Crist (1985) as FORCE applies it, and the per-sensor matrices."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor
from zeit._tasseled import BAIG_2014, CRIST_1985, HUANG_2002

BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
FORCE = np.array([[0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303],
                  [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
                  [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109]], dtype=np.float32)
NAMES = ("brightness", "greenness", "wetness")


def _cube(platforms=("landsat-8",), h=6, w=7, seed=0, dtype=np.int16, bands=BANDS):
    rng = np.random.default_rng(seed)
    x = rng.integers(100, 6000, (len(platforms), len(bands), h, w))
    data = x.astype(dtype) if np.issubdtype(dtype, np.integer) else (x / 10000.0).astype(dtype)
    return xr.DataArray(data, dims=("time", "band", "y", "x"), name="reflectance",
                        coords={"time": pd.date_range("2020-01-01", periods=len(platforms), freq="10D"),
                                "band": bands, "y": np.arange(h) * 30.0, "x": np.arange(w) * 30.0,
                                "platform": ("time", list(platforms))})


def force_integers(x):
    """FORCE's index_tasseled (index-compute-hl.c), line by line, on int16 values (band, y, x): float32 products
    added in the band order, TC-DI as the accumulation of 1, -1, -1 times the three sums, then (short)."""
    f = [x[b].astype(np.float32) for b in range(6)]
    tmp = []
    for i in range(3):
        t = FORCE[i, 0] * f[0] + FORCE[i, 1] * f[1]
        for b in range(2, 6):
            t = t + FORCE[i, b] * f[b]
        tmp.append(t)
    di = np.float32(0) + np.float32(1) * tmp[0]
    di = di + np.float32(-1) * tmp[1]
    di = di + np.float32(-1) * tmp[2]
    return [t.astype(np.int16) for t in (*tmp, di)]


def test_crist_table_is_forces():
    got = np.array([CRIST_1985[k] for k in NAMES], dtype=np.float32)
    assert np.array_equal(got, FORCE)


def test_equals_force_arithmetic_on_integers():
    cube = _cube(["landsat-8", "sentinel-2a"], seed=3)
    out = zeit.tasseled_cap(cube, ["brightness", "greenness", "wetness", "disturbance"])
    assert out.dtype == np.float32 and out.dims == cube.dims
    assert list(out.band.values) == ["brightness", "greenness", "wetness", "disturbance"]
    for k in range(2):
        want = force_integers(cube.values[k])
        for i in range(4):
            assert np.array_equal(out.values[k, i].astype(np.int16), want[i]), (k, i)


def test_is_the_matrix_product_and_scale_free():
    cube = _cube(dtype=np.float32)
    out = zeit.tasseled_cap(cube)
    want = np.einsum("cb,tbyx->tcyx", FORCE.astype(np.float64), cube.values.astype(np.float64))
    np.testing.assert_allclose(out.values, want, rtol=1e-5, atol=1e-5)
    big = zeit.tasseled_cap(cube * 10000)
    np.testing.assert_allclose(big.values, out.values * 10000, rtol=1e-5, atol=1e-2)


def test_disturbance_is_b_minus_g_minus_w_in_any_position():
    cube = _cube(dtype=np.float32)
    all4 = zeit.tasseled_cap(cube, ["brightness", "greenness", "wetness", "disturbance"])
    one = zeit.tasseled_cap(cube, "disturbance")
    first = zeit.tasseled_cap(cube, ["disturbance", "wetness"])
    assert list(first.band.values) == ["disturbance", "wetness"]
    np.testing.assert_array_equal(one.values[:, 0], all4.values[:, 3])
    np.testing.assert_array_equal(first.values[:, 0], all4.values[:, 3])
    np.testing.assert_array_equal(first.values[:, 1], all4.values[:, 2])
    np.testing.assert_allclose(all4.values[:, 3], all4.values[:, 0] - all4.values[:, 1] - all4.values[:, 2],
                               rtol=1e-5, atol=1e-5)


def test_threads_and_tiles_do_not_change_the_values():
    cube = _cube(["landsat-8"] * 3, h=40, w=3000, dtype=np.float32)          # several tiles of rows per date
    one = zeit.tasseled_cap(cube, ["brightness", "disturbance"], n_jobs=1)
    many = zeit.tasseled_cap(cube, ["brightness", "disturbance"], n_jobs=3)
    np.testing.assert_array_equal(one.values, many.values)


def test_sensor_matrices_per_date():
    platforms = ["landsat-5", "landsat-7", "landsat-8", "landsat-9", "sentinel-2b"]
    cube = _cube(platforms, dtype=np.float32, seed=5)
    out = zeit.tasseled_cap(cube, coefficients="sensor")
    tables = [CRIST_1985, HUANG_2002, BAIG_2014, BAIG_2014, CRIST_1985]   # Sentinel-2: Crist, as FORCE
    for k, table in enumerate(tables):
        m = np.array([table[c] for c in NAMES])
        want = np.einsum("cb,byx->cyx", m, cube.values[k].astype(np.float64))
        np.testing.assert_allclose(out.values[k], want, rtol=2e-5, atol=1e-5, err_msg=platforms[k])
    assert "Huang" in out.attrs["tasseled_cap"] and "Baig" in out.attrs["tasseled_cap"]
    with pytest.raises(ValueError, match="no published Tasseled Cap"):
        zeit.tasseled_cap(cube, coefficients="sensor", sensor="terra")
    assert zeit.tasseled_cap(cube, coefficients="baig2014").attrs["tasseled_cap"].startswith("baig2014")


def test_sensor_names_and_errors():
    cube = _cube(["landsat-8", "landsat-8"]).drop_vars("platform")
    with pytest.raises(ValueError, match="tasseled_cap needs each date's sensor"):
        zeit.tasseled_cap(cube, coefficients="sensor")
    assert zeit.tasseled_cap(cube, coefficients="sensor", sensor="LC08").shape[1] == 3
    assert zeit.tasseled_cap(cube).shape[1] == 3          # the default needs no sensor


def test_custom_coefficients_and_missing_values():
    cube = _cube(dtype=np.float32)
    cube.values[0, 2, 1, 1] = np.nan
    mine = {"a": [1, 0, 0, 0, 0, 0], "b": [0, 0, 0, 0, 0, 2]}
    out = zeit.tasseled_cap(cube, ["a", "b"], coefficients=mine)
    np.testing.assert_allclose(out.values[0, 0, 0, 0], cube.values[0, 0, 0, 0])
    np.testing.assert_allclose(out.values[0, 1, 0, 0], 2 * cube.values[0, 5, 0, 0])
    assert np.isnan(out.values[0, :, 1, 1]).all() and not np.isnan(out.values[0, :, 1, 2]).any()
    arr = zeit.tasseled_cap(cube, ["a", "b"], coefficients=np.array(list(mine.values())))
    np.testing.assert_array_equal(arr.values, out.values)
    with pytest.raises(ValueError, match="no \\['wetness'\\]"):
        zeit.tasseled_cap(cube, "wetness", coefficients=mine)
    with pytest.raises(ValueError, match="rows"):
        zeit.tasseled_cap(cube, coefficients=np.ones((2, 6)))
    ints = _cube(dtype=np.int16)           # integers: 0 is nodata, as for the other functions
    ints.values[0, 0, 0, 0] = 0
    assert np.isnan(zeit.tasseled_cap(ints).values[0, :, 0, 0]).all()


def test_bands_dask_and_no_time():
    cube = _cube(["landsat-8", "landsat-5"], dtype=np.float32, bands=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])
    ref = zeit.tasseled_cap(cube)
    renamed = cube.assign_coords(band=["w", "x", "y", "z", "u", "v"])
    with pytest.raises(ValueError, match="pass bands="):
        zeit.tasseled_cap(renamed)
    mapped = zeit.tasseled_cap(renamed, bands={"blue": "w", "green": "x", "red": "y", "nir": "z", "swir1": "u",
                                               "swir2": "v"})
    np.testing.assert_array_equal(mapped.values, ref.values)
    lazy = zeit.tasseled_cap(cube.chunk({"time": 1, "y": 3, "x": 4}), ["disturbance", "brightness"])
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values[:, 0], (ref.values[:, 0] - ref.values[:, 1]) - ref.values[:, 2])
    single = zeit.tasseled_cap(cube.isel(time=0, drop=True))
    assert single.dims == ("band", "y", "x")
    np.testing.assert_array_equal(single.values, ref.values[0])
    np.testing.assert_array_equal(cube.zeit.tasseled_cap().values, ref.values)


def test_sentinel2_broad_nir_warns_and_b8a_is_preferred():
    cube = _cube(["sentinel-2a"], dtype=np.float32, bands=["blue", "green", "red", "nir", "swir16", "swir22"])
    with pytest.warns(UserWarning, match="broad NIR"):
        zeit.tasseled_cap(cube, coefficients="sensor")
    both = _cube(["sentinel-2a"], dtype=np.float32,
                 bands=["blue", "green", "red", "nir", "nir08", "swir16", "swir22"])
    out = zeit.tasseled_cap(both, "brightness")
    m = np.array(CRIST_1985["brightness"])
    v = both.values[0]
    want = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3] * v[4] + m[4] * v[5] + m[5] * v[6]
    np.testing.assert_allclose(out.values[0, 0], want, rtol=1e-5)


def test_refuses_a_second_time_and_bad_input():
    cube = _cube(dtype=np.float32)
    with pytest.raises(ValueError, match="already"):
        zeit.tasseled_cap(zeit.tasseled_cap(cube))
    with pytest.raises(ValueError, match="band dimension"):
        zeit.tasseled_cap(cube.isel(band=0, drop=True))
    with pytest.raises(ValueError, match="unknown role"):
        zeit.tasseled_cap(cube, bands={"nirr": "x"})
    with pytest.raises(ValueError, match="twice"):
        zeit.tasseled_cap(cube, ["brightness", "brightness"])
    with pytest.raises(ValueError, match="coefficients="):
        zeit.tasseled_cap(cube, coefficients="tm")


def test_compute_indices_has_the_crist_components():
    cube = _cube(dtype=np.float32)
    out = zeit.compute_indices(cube, ["TCB", "TCG", "TCW", "TCD"], band_map={"nir": "nir08"})
    tc = zeit.tasseled_cap(cube, ["brightness", "greenness", "wetness", "disturbance"])
    np.testing.assert_allclose(out.values, tc.values, rtol=1e-4, atol=1e-5)
