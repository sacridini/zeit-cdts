"""zeit.sentinel2_resmerge: FORCE's 20 m to 10 m resolution merge (regression and ImproPhe), ported from FORCE 3.11."""
import numpy as np
import pytest
import xarray as xr

import zeit
from zeit import _core

BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
PREDICTED = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
# each predicted band as c0 + c1 green + c2 red + c3 nir (reflectance times 10000)
COEF = {
    "rededge1": (200.0, 0.3, 0.2, 0.1),
    "rededge2": (100.0, 0.1, 0.1, 0.6),
    "rededge3": (-50.0, 0.2, -0.1, 0.9),
    "nir08": (10.0, 0.0, 0.0, 1.0),
    "swir16": (300.0, 0.5, 0.2, -0.3),
    "swir22": (0.0, 0.7, 0.3, 0.0),
}


def linear_scene(ny=16, nx=18, nt=None, seed=1, names=BANDS):
    """Green, red and NIR with independent noise, the 20 m bands exact linear functions of them."""
    rng = np.random.default_rng(seed)
    shape = (ny, nx) if nt is None else (nt, ny, nx)
    g = np.round(800 + 300 * rng.random(shape))
    r = np.round(600 + 250 * rng.random(shape))
    n = np.round(2500 + 700 * rng.random(shape))
    blue = np.round(400 + 100 * rng.random(shape))
    arrays = {"blue": blue, "green": g, "red": r, "nir": n}
    for name, (c0, c1, c2, c3) in COEF.items():
        arrays[name] = np.round(c0 + c1 * g + c2 * r + c3 * n)
    data = np.stack([arrays[b] for b in BANDS], axis=-3).astype(np.float32)
    dims = ("band", "y", "x") if nt is None else ("time", "band", "y", "x")
    coords = {"band": names, "y": np.arange(ny)[::-1] * 10.0 + 5.0, "x": np.arange(nx) * 10.0 + 5.0}
    if nt is not None:
        coords["time"] = np.datetime64("2024-01-01") + np.arange(nt).astype("timedelta64[D]")
    return xr.DataArray(data, dims=dims, coords=coords, name="reflectance")


def values(cube, band):
    return cube.sel(band=band).values


def test_shapes_coordinates_and_pass_through():
    cube = linear_scene(nt=3)
    out = zeit.sentinel2_resmerge(cube)
    assert out.dims == cube.dims and out.shape == cube.shape
    assert out.dtype == np.float32
    assert (out.band.values == cube.band.values).all()
    assert (out.time.values == cube.time.values).all()
    for name in ("blue", "green", "red", "nir"):
        np.testing.assert_array_equal(values(out, name), values(cube, name))
    assert "sentinel2_resmerge" in out.attrs
    one = zeit.sentinel2_resmerge(cube.isel(time=1))
    assert one.dims == ("band", "y", "x")
    np.testing.assert_array_equal(one.values, out.isel(time=1).values)


def test_band_order_does_not_matter():
    cube = linear_scene()
    shuffled = cube.isel(band=[9, 3, 0, 7, 1, 8, 2, 6, 5, 4])
    a = zeit.sentinel2_resmerge(cube)
    b = zeit.sentinel2_resmerge(shuffled)
    np.testing.assert_array_equal(a.sel(band=BANDS).values, b.sel(band=BANDS).values)
    assert list(b.band.values) == list(shuffled.band.values)


def test_regression_recovers_a_linear_band():
    cube = linear_scene()
    out = zeit.sentinel2_resmerge(cube, method="regression")
    for name in PREDICTED:
        err = np.abs(values(out, name) - values(cube, name))[2:-2, 2:-2]     # inside: the windows are full
        assert err.max() < 1.0, (name, err.max())     # the targets are rounded to integers
    # the border differs, as FORCE's: the padded rows (1, 0, 0, 0) with y = 0 pull the fit down
    full = np.abs(values(out, "rededge1") - values(cube, "rededge1"))
    inner = np.zeros(full.shape, bool)
    inner[2:-2, 2:-2] = True
    assert full[~inner].mean() > 2 * full[inner].mean()
    # noise in the 20 m values (their own, not the 10 m bands') is smoothed: the fit is closer to the line
    rng = np.random.default_rng(5)
    noisy = cube.copy()
    for name in PREDICTED:
        noisy.loc[{"band": name}] = values(cube, name) + np.round(rng.normal(0, 40, cube.shape[1:])).astype(np.float32)
    out = zeit.sentinel2_resmerge(noisy, method="regression")
    for name in PREDICTED:
        after = np.abs(values(out, name) - values(cube, name))[2:-2, 2:-2].mean()
        before = np.abs(values(noisy, name) - values(cube, name))[2:-2, 2:-2].mean()
        assert after < before / 2, (name, before, after)


def test_improphe_recovers_a_band_that_is_constant_per_class():
    ny, nx = 16, 20
    rng = np.random.default_rng(3)
    cube = linear_scene(ny, nx)
    # two land covers, left and right, with slightly different spectra inside each
    left = np.arange(nx) < nx // 2
    g = np.where(left, 1000.0, 3000.0) + np.round(rng.normal(0, 5, (ny, nx)))
    r = np.where(left, 800.0, 2800.0) + np.round(rng.normal(0, 5, (ny, nx)))
    n = np.where(left, 3000.0, 500.0) + np.round(rng.normal(0, 5, (ny, nx)))
    cube.loc[{"band": "green"}] = g.astype(np.float32)
    cube.loc[{"band": "red"}] = r.astype(np.float32)
    cube.loc[{"band": "nir"}] = n.astype(np.float32)
    truth = {name: np.where(left, 1000.0 + 100 * k, 9000.0 - 100 * k) for k, name in enumerate(PREDICTED)}
    noisy = cube.copy()
    for name in PREDICTED:
        noisy.loc[{"band": name}] = (truth[name] + rng.normal(0, 1, (ny, nx))).astype(np.float32)
    out = zeit.sentinel2_resmerge(noisy, method="improphe")
    for name in PREDICTED:
        assert np.abs(values(out, name) - truth[name]).max() < 5, name


def test_nan_is_off_and_clear_leaves_pixels_alone():
    cube = linear_scene()
    off_pixel = (6, 7)
    cloud_pixel = (9, 10)
    damaged = cube.copy()
    damaged[{"band": 1, "y": off_pixel[0], "x": off_pixel[1]}] = np.nan        # NaN in the green band
    clear = xr.DataArray(np.ones(cube.shape[1:], bool), dims=("y", "x"))
    clear[cloud_pixel] = False
    for method in ("regression", "improphe"):
        out = zeit.sentinel2_resmerge(damaged, method=method, clear=clear)
        # the NaN pixel stays NaN in the band that had it and keeps the other bands' input values
        assert np.isnan(out.values[1, off_pixel[0], off_pixel[1]])
        for name in PREDICTED:
            assert values(out, name)[off_pixel] == values(damaged, name)[off_pixel]
            assert values(out, name)[cloud_pixel] == values(damaged, name)[cloud_pixel]
        assert not np.isnan(out.values[:, :, :]).sum() > 1
    # a cloud is a neighbour of the regression but not of ImproPhe; the off pixel is neither's
    cloud_value = cube.copy()
    for name in PREDICTED:
        cloud_value.loc[{"band": name, "y": cloud_value.y.values[cloud_pixel[0]],
                         "x": cloud_value.x.values[cloud_pixel[1]]}] = 20000.0
    r_with = zeit.sentinel2_resmerge(cloud_value, method="regression", clear=clear)
    i_with = zeit.sentinel2_resmerge(cloud_value, method="improphe", clear=clear)
    near = (cloud_pixel[0] - 1, cloud_pixel[1] + 1)
    truth = values(cube, "nir08")[near]
    assert abs(values(r_with, "nir08")[near] - truth) > 100          # the cloud is in the regression's window
    assert abs(values(i_with, "nir08")[near] - truth) < 60          # and not in ImproPhe's mean
    # a clear given as a time-less array for a cube with time
    series = linear_scene(nt=2)
    out = zeit.sentinel2_resmerge(series, clear=clear)
    assert out.shape == series.shape


@pytest.mark.parametrize("method", ["regression", "improphe"])
def test_units_do_not_matter(method):
    cube = linear_scene()
    rng = np.random.default_rng(11)
    # fractions of a digital number: no distance sits exactly on one of ImproPhe's thresholds
    jittered = cube + rng.random(cube.shape).astype(np.float32)
    ref = zeit.sentinel2_resmerge(jittered, method=method)
    small = zeit.sentinel2_resmerge(jittered / 10000.0, method=method)
    np.testing.assert_allclose(small.values * 10000.0, ref.values, rtol=2e-6, atol=0.02)
    explicit = zeit.sentinel2_resmerge(jittered / 10000.0, method=method, scale=1.0)
    np.testing.assert_array_equal(explicit.values, small.values)
    # integers are the same as the floats that hold them
    as_int = zeit.sentinel2_resmerge(cube.astype(np.int16), method=method, nodata=None)
    np.testing.assert_array_equal(as_int.values, zeit.sentinel2_resmerge(cube, method=method).values)


def test_force_gives_truncated_integers_and_differs_from_the_free_regression():
    cube = linear_scene(ny=14, nx=14)
    rng = np.random.default_rng(9)
    noisy = cube.copy()
    for name in PREDICTED:
        noisy.loc[{"band": name}] = values(cube, name) + np.round(rng.normal(0, 30, cube.shape[1:])).astype(np.float32)
    free = zeit.sentinel2_resmerge(noisy, method="regression")
    forced = zeit.sentinel2_resmerge(noisy, method="regression", force=True)
    for name in PREDICTED:
        v = values(forced, name)
        np.testing.assert_array_equal(v, np.trunc(v))
    # the first pixel reads no earlier prediction: equal up to the truncation; the sweep feeds back after that
    for name in PREDICTED:
        assert abs(values(forced, name)[0, 0] - values(free, name)[0, 0]) < 1.0 + 1e-3
    diff = np.abs(forced.sel(band=PREDICTED).values - free.sel(band=PREDICTED).values)
    assert diff.max() > 1.0                    # the feedback
    assert np.median(diff) < 30                # but the same kind of map
    again = zeit.sentinel2_resmerge(noisy, method="regression", force=True, n_jobs=4)
    np.testing.assert_array_equal(again.values, forced.values)


def test_force_improphe_truncates():
    cube = linear_scene(ny=12, nx=12)
    free = zeit.sentinel2_resmerge(cube, method="improphe", n_jobs=1)
    forced = zeit.sentinel2_resmerge(cube, method="improphe", force=True, n_jobs=3)
    v = forced.sel(band=PREDICTED).values
    np.testing.assert_array_equal(v, np.trunc(v))
    f = free.sel(band=PREDICTED).values
    assert np.all(v <= f + 1e-3) or np.abs(v - f).max() < 1.0 + 1e-3
    np.testing.assert_array_equal(free.values, zeit.sentinel2_resmerge(cube, method="improphe", n_jobs=4).values)


def test_force_in_another_scale_is_rounded_to_integers_of_10000():
    cube = linear_scene(ny=10, nx=10)
    a = zeit.sentinel2_resmerge(cube, method="improphe", force=True)
    b = zeit.sentinel2_resmerge(cube / 10000.0, method="improphe", force=True)
    np.testing.assert_allclose(b.values * 10000.0, a.values, atol=1e-2)


def test_earth_search_and_sentinel_names():
    cube = linear_scene()
    renamed = cube.assign_coords(band=["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"])
    a = zeit.sentinel2_resmerge(cube)
    b = zeit.sentinel2_resmerge(renamed)
    np.testing.assert_array_equal(a.values, b.values)
    odd = cube.assign_coords(band=[f"b{k}" for k in range(10)])
    c = zeit.sentinel2_resmerge(odd, bands={"green": "b1", "red": "b2", "nir": "b6", "rededge1": "b3",
                                            "rededge2": "b4", "rededge3": "b5", "nir08": "b7", "swir16": "b8",
                                            "swir22": "b9"})
    np.testing.assert_array_equal(a.values, c.values)


def test_errors():
    cube = linear_scene()
    with pytest.raises(ValueError, match="swir22"):
        zeit.sentinel2_resmerge(cube.drop_sel(band="swir22"))
    with pytest.raises(ValueError, match="rededge1.*rededge2|rededge2"):
        zeit.sentinel2_resmerge(cube.drop_sel(band=["rededge1", "rededge2"]))
    with pytest.raises(ValueError, match="method"):
        zeit.sentinel2_resmerge(cube, method="starfm")
    with pytest.raises(ValueError, match="band"):
        zeit.sentinel2_resmerge(cube.isel(band=0, drop=True))
    with pytest.raises(TypeError):
        zeit.sentinel2_resmerge(cube.values)
    with pytest.raises(ValueError, match="clear"):
        zeit.sentinel2_resmerge(cube, clear=np.ones((3, 3), bool))
    with pytest.raises(ValueError, match="unknown role"):
        zeit.sentinel2_resmerge(cube, bands={"foo": "green"})


def test_core_checks_its_inputs():
    g = np.ones((4, 5), np.float32)
    t = np.ones((6, 4, 5), np.float32)
    m = np.zeros((4, 5), bool)
    ok = _core.resmerge.resmerge_batch(g, g, g, t, m, m, "improphe")
    assert ok.shape == (6, 4, 5)
    with pytest.raises(ValueError):
        _core.resmerge.resmerge_batch(g, g, g, t[:5], m, m, "improphe")
    with pytest.raises(ValueError):
        _core.resmerge.resmerge_batch(g, g[:3], g, t, m, m, "improphe")
    with pytest.raises(ValueError):
        _core.resmerge.resmerge_batch(g, g, g, t, m, m, "nope")
