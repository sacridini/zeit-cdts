"""zeit.pansharpen: Brovey, GSA, HPF and PCA fusion of multispectral bands with a finer pan, for any sensor."""
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

import zeit

NAMES = ["blue", "green", "red", "nir", "b5", "b6", "b7", "b8"]
X0, Y0 = 500000.0, 8000000.0
SUB = 6     # sub-pixels a side whose mean is a band cell


def field(x, y, nb, seed):
    """The bands as functions of the map coordinates: smooth, plus waves of 6 to 25 m shared by the bands."""
    rng = np.random.default_rng(seed)
    waves = [(rng.uniform(0.25, 1.0), rng.uniform(0, 3.14), rng.uniform(0, 6.28)) for _ in range(10)]
    tex = sum(np.cos(f * (np.cos(a) * x + np.sin(a) * y) + ph) for f, a, ph in waves) / 3
    base = np.sin(y * 0.011) + np.cos(x * 0.009) + 0.5 * np.sin((y + 2 * x) * 0.003)
    return np.stack([300 + 100 * (k + 1) + 60 * base * (1 + 0.3 * k) + 25 * (1 + 0.2 * k) * tex
                     + 20 * np.cos(y * (k + 3) * 0.0013) for k in range(nb)]).astype(np.float64)


def make(ny=16, nx=20, ratio=4.0, nb=4, pan_res=2.0, nt=None, seed=3, dx=0.0):
    """Bands of `ratio * pan_res` metre cells (the mean of the field over the cell) and a pan of `pan_res` metre
    cells (the mean of the bands' field plus noise). Returns (ms, pan, truth): the field at the pan's cells.
    `dx` shifts the pan's corner (metres)."""
    res = ratio * pan_res
    width, height = nx * res, ny * res
    sub = (np.arange(SUB) + 0.5) / SUB

    def averaged(n_y, n_x, cell, x0):
        ys = Y0 - (np.arange(n_y)[:, None] + sub[None, :]) * cell
        xs = x0 + (np.arange(n_x)[:, None] + sub[None, :]) * cell
        Y = np.repeat(ys.reshape(-1), n_x * SUB).reshape(n_y * SUB, n_x * SUB)
        X = np.tile(xs.reshape(-1), n_y * SUB).reshape(n_y * SUB, n_x * SUB)
        f = field(X, Y, nb, seed)
        return f.reshape(nb, n_y, SUB, n_x, SUB).mean(axis=(2, 4))

    coarse = averaged(ny, nx, res, X0)
    pny, pnx = int(round(height / pan_res)), int(round(width / pan_res))
    truth = averaged(pny, pnx, pan_res, X0 + dx)
    pan = truth.mean(axis=0) + 1.5 * np.random.default_rng(seed + 1).standard_normal(truth.shape[1:])
    cy = Y0 - (np.arange(ny) + 0.5) * res
    cx = X0 + (np.arange(nx) + 0.5) * res
    ms = xr.DataArray(coarse.astype(np.float32), dims=("band", "y", "x"),
                      coords={"band": NAMES[:nb], "y": cy, "x": cx}).rio.write_crs("EPSG:32723")
    py = Y0 - (np.arange(pny) + 0.5) * pan_res
    px = X0 + dx + (np.arange(pnx) + 0.5) * pan_res
    pn = xr.DataArray(pan.astype(np.float32), dims=("y", "x"), coords={"y": py, "x": px}).rio.write_crs("EPSG:32723")
    if nt:
        ms = xr.concat([ms * (1 + 0.1 * t) for t in range(nt)], dim="time").assign_coords(
            time=np.datetime64("2024-01-01") + np.arange(nt).astype("timedelta64[D]"))
        ms = ms.rio.write_crs("EPSG:32723")
    return ms, pn, truth.astype(np.float32)


def rmse(a, b):
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def cubic_only(ms, pan):
    return zeit.pansharpen(ms, pan, "hpf", gain=0.0)


@pytest.mark.parametrize("method", ["brovey", "gsa", "hpf", "pca"])
def test_shape_grid_and_attributes(method):
    ms, pan, _ = make()
    ms.attrs["long_name"] = "reflectance"
    out = zeit.pansharpen(ms, pan, method)
    assert out.dims == ms.dims and out.shape == (4, 64, 80) and out.dtype == np.float32
    assert (out.band.values == ms.band.values).all()
    np.testing.assert_array_equal(out.x.values, pan.x.values)
    np.testing.assert_array_equal(out.y.values, pan.y.values)
    assert out.rio.crs == ms.rio.crs and out.attrs["long_name"] == "reflectance"
    assert method in out.attrs["pansharpen"]
    assert np.isfinite(out.values).all()


def test_time_dimension_is_one_scene_per_date():
    ms, pan, _ = make(nt=3)
    out = zeit.pansharpen(ms, pan, "gsa")
    assert out.dims == ms.dims and out.shape[0] == 3
    for t in range(3):
        alone = zeit.pansharpen(ms.isel(time=t), pan, "gsa")
        np.testing.assert_array_equal(out.isel(time=t).values, alone.values)


@pytest.mark.parametrize("ratio,pan_res", [(2, 15.0), (3, 10.0), (4, 2.0), (2.5, 4.0), (4.333, 1.5), (8, 1.0)])
@pytest.mark.parametrize("nb", [1, 3, 4, 8])
def test_any_ratio_and_any_number_of_bands(ratio, pan_res, nb):
    ms, pan, truth = make(ny=14, nx=14, ratio=ratio, nb=nb, pan_res=pan_res)
    for method in ("brovey", "gsa", "hpf") + (("pca",) if nb > 1 else ()):
        out = zeit.pansharpen(ms, pan, method)
        assert out.shape[0] == nb and np.isfinite(out.values[:, 8:-8, 8:-8]).all(), method
        assert f"ratio {ratio:g}" in out.attrs["pansharpen"] or ratio == 4.333


@pytest.mark.parametrize("ratio,pan_res", [(2, 15.0), (3, 10.0), (4, 2.0), (2.5, 4.0), (4.333, 1.5)])
@pytest.mark.parametrize("nb", [3, 4, 8])
def test_the_injection_brings_the_bands_closer_to_the_field_than_the_cubic_resampling(ratio, pan_res, nb):
    ms, pan, truth = make(ny=24, nx=24, ratio=ratio, nb=nb, pan_res=pan_res)
    inner = (slice(None), slice(12, -12), slice(12, -12))
    base = rmse(cubic_only(ms, pan).values[inner], truth[inner])
    for method in ("gsa", "hpf", "pca", "brovey"):
        out = zeit.pansharpen(ms, pan, method)
        assert rmse(out.values[inner], truth[inner]) < base, (method, ratio, nb)


def test_pan_equal_to_the_intensity_changes_nothing():
    ms, pan, _ = make()
    cubic = cubic_only(ms, pan)
    w = xr.DataArray([0.1, 0.4, 0.3, 0.2], dims="band", coords={"band": NAMES[:4]})
    mix = ((cubic * w).sum("band") * 2 + 7).astype(np.float32).rio.write_crs(pan.rio.crs)
    out = zeit.pansharpen(ms, mix, "gsa")
    np.testing.assert_allclose(out.values, cubic.values, atol=0.1)
    mean = cubic.mean("band").astype(np.float32).rio.write_crs(pan.rio.crs)
    np.testing.assert_allclose(zeit.pansharpen(ms, mean, "brovey", weights=[1, 1, 1, 1]).values, cubic.values,
                               rtol=1e-4)
    np.testing.assert_allclose(zeit.pansharpen(ms, mean, "brovey").values, cubic.values, rtol=5e-3)


def test_bands_outside_the_pans_range_and_the_choice_of_intensity_bands():
    ms, pan, _ = make()
    cubic = cubic_only(ms, pan)
    # a pan that sees the blue, green and red (2 : 1 : 1) and not the NIR
    mix = ((cubic.sel(band="blue") * 2 + cubic.sel(band="green") + cubic.sel(band="red")) / 4 * 3 + 5)
    mix = mix.astype(np.float32).rio.write_crs(pan.rio.crs)
    seen = zeit.pansharpen(ms, mix, "gsa", intensity_bands=["blue", "green", "red"])
    np.testing.assert_allclose(seen.values, cubic.values, atol=0.1)
    same = zeit.pansharpen(ms, mix, "gsa", intensity_bands=[0, 1, 2])         # by position
    np.testing.assert_array_equal(seen.values, same.values)
    # with all the bands the regression still gives the NIR no weight, and the fit is as good
    allb = zeit.pansharpen(ms, mix, "gsa")
    np.testing.assert_allclose(allb.values, cubic.values, atol=0.1)
    # the fitted Brovey absorbs the pan's gain; the equal weights do not
    fit = zeit.pansharpen(ms, mix, "brovey", intensity_bands=["blue", "green", "red"])
    assert rmse(fit.values, cubic.values) < 0.03 * float(np.abs(cubic.values).mean())
    equal = zeit.pansharpen(ms, mix, "brovey", weights=[1, 1, 1, 0])
    assert rmse(equal.values, cubic.values) > 5 * rmse(fit.values, cubic.values)


def test_constants_change_nothing():
    ms, pan, _ = make()
    flat_ms = ms * 0 + xr.DataArray([100.0, 200.0, 300.0, 400.0], dims="band", coords={"band": NAMES[:4]})
    flat_ms = flat_ms.astype(np.float32).rio.write_crs(ms.rio.crs)
    flat_pan = (pan * 0 + 250).astype(np.float32).rio.write_crs(pan.rio.crs)
    want = np.broadcast_to(np.array([100.0, 200, 300, 400], dtype=np.float32)[:, None, None], (4, 64, 80))
    for method in ("gsa", "hpf", "pca"):
        np.testing.assert_allclose(zeit.pansharpen(flat_ms, flat_pan, method).values, want, atol=1e-3)
    out = zeit.pansharpen(flat_ms, flat_pan, "brovey", weights=[1, 1, 1, 1])      # 250 / 250
    np.testing.assert_allclose(out.values, want, rtol=1e-5)


def test_nan_in_the_pan_or_a_band():
    ms, pan, _ = make()
    clean = {m: zeit.pansharpen(ms, pan, m) for m in ("brovey", "gsa", "hpf", "pca")}
    ms = ms.copy()
    pan = pan.copy()
    pan[10, 11] = np.nan
    ms[1, 5, 6] = np.nan
    for method in clean:
        out = zeit.pansharpen(ms, pan, method)
        bad = np.isnan(out.values)
        assert bad[:, 10, 11].all()
        assert bad[:, 20:24, 24:28].all()          # the pan cells of the missing coarse cell
        assert bad.sum() == 4 * (1 + 16)           # nothing else
        assert np.isfinite(out.values[:, 40:50, 40:50]).all()
        d = np.abs(out.values[:, 40:50, 40:50] - clean[method].values[:, 40:50, 40:50]).max()
        assert d < 15, (method, d)


def test_nodata_value_is_missing():
    ms, pan, _ = make()
    ms = ms.copy()
    ms[:, 3, 3] = -9999
    out = zeit.pansharpen(ms, pan, "gsa", nodata=-9999)
    assert np.isnan(out.values[:, 12:16, 12:16]).all()
    assert np.isfinite(out.values[:, 20:, 20:]).all()


def test_integer_input_with_zero_nodata():
    ms, pan, _ = make()
    ms_i = (ms * 10).round().astype(np.uint16).rio.write_crs(ms.rio.crs)
    pan_i = (pan * 10).round().astype(np.uint16).rio.write_crs(pan.rio.crs)
    ms_i[:, 2, 2] = 0
    out = zeit.pansharpen(ms_i, pan_i, "hpf")
    assert out.dtype == np.float32 and np.isnan(out.values[:, 8:12, 8:12]).all()


def test_the_pan_may_cover_less_or_more_than_the_bands_and_the_corners_need_not_coincide():
    ms, pan, _ = make()
    small = pan.isel(y=slice(8, 48), x=slice(17, 71))      # not on a cell boundary of the bands
    out = zeit.pansharpen(ms, small, "gsa")
    assert out.shape == (4, 40, 54) and np.isfinite(out.values).all()
    np.testing.assert_array_equal(out.x.values, small.x.values)
    # the pan reaches 3 band cells beyond the right edge: NaN there
    wide = xr.concat([pan, pan.isel(x=slice(0, 12)).assign_coords(x=pan.x.values[-1] + 2.0 * (1 + np.arange(12)))],
                     dim="x").rio.write_crs(pan.rio.crs)
    out = zeit.pansharpen(ms, wide, "hpf")
    assert np.isnan(out.values[:, :, 80:]).all() and np.isfinite(out.values[:, 10:-10, 10:70]).all()
    # the shift of a third of a band cell changes the statistics and the result little
    ms2, pan2, truth2 = make(dx=2.7)
    shifted = zeit.pansharpen(ms2, pan2, "gsa")
    inner = (slice(None), slice(12, -12), slice(12, -12))
    assert rmse(shifted.values[inner], truth2[inner]) < rmse(cubic_only(ms2, pan2).values[inner], truth2[inner])


def test_grids_that_do_not_fit_say_what_is_wrong():
    ms, pan, _ = make()
    other = pan.rio.write_crs("EPSG:32722")
    with pytest.raises(ValueError, match="CRS"):
        zeit.pansharpen(ms, other, "gsa")
    nocrs = xr.DataArray(pan.values, dims=("y", "x"), coords={"y": pan.y, "x": pan.x})
    with pytest.raises(ValueError, match="no CRS"):
        zeit.pansharpen(ms, nocrs, "gsa")
    outside = pan.assign_coords(x=pan.x + 8 * 40)
    with pytest.raises(ValueError, match="do not overlap"):
        zeit.pansharpen(ms, outside, "gsa")
    coarse_pan = xr.DataArray(ms.values[0], dims=("y", "x"), coords={"y": ms.y, "x": ms.x}).rio.write_crs(ms.rio.crs)
    with pytest.raises(ValueError, match="not finer"):
        zeit.pansharpen(ms, coarse_pan, "gsa")
    with pytest.raises(ValueError, match="not finer"):
        zeit.pansharpen(pan.expand_dims(band=["pan"]), ms.isel(band=0, drop=True), "gsa")
    with pytest.raises(ValueError, match="method must be"):
        zeit.pansharpen(ms, pan, "wavelet")
    with pytest.raises(ValueError, match="one panchromatic band"):
        zeit.pansharpen(ms, xr.concat([pan, pan], dim="band"), "gsa")
    with pytest.raises(ValueError, match="kernel must be odd"):
        zeit.pansharpen(ms, pan, "hpf", kernel=8)
    with pytest.raises(ValueError, match="weights has"):
        zeit.pansharpen(ms, pan, "brovey", weights=[1, 2])
    with pytest.raises(ValueError, match="weights is for"):
        zeit.pansharpen(ms, pan, "gsa", weights=[1, 1, 1, 1])
    with pytest.raises(ValueError, match="intensity_bands"):
        zeit.pansharpen(ms, pan, "gsa", intensity_bands=["swir"])
    with pytest.raises(ValueError, match="not used by method='hpf'"):
        zeit.pansharpen(ms, pan, "hpf", intensity_bands=["red"])
    with pytest.raises(ValueError, match="at least 2 bands"):
        zeit.pansharpen(ms.isel(band=[0]), pan, "pca")
    with pytest.raises(ValueError, match="ratio must be"):
        zeit.pansharpen(ms, pan, "gsa", ratio=0.5)


def test_brovey_weights_by_name_and_by_position():
    ms, pan, _ = make()
    a = zeit.pansharpen(ms, pan, "brovey", weights=[0, 0, 1, 1])
    b = zeit.pansharpen(ms, pan, "brovey", weights={"red": 5, "nir": 5})
    np.testing.assert_allclose(a.values, b.values, rtol=1e-6)
    assert not np.allclose(a.values, zeit.pansharpen(ms, pan, "brovey", weights=[1, 1, 1, 1]).values)


@pytest.mark.parametrize("method", ["brovey", "gsa", "hpf", "pca"])
def test_deterministic_across_n_jobs(method):
    ms, pan, _ = make(ny=40, nx=44, ratio=2.5, pan_res=4.0)
    ref = zeit.pansharpen(ms, pan, method, n_jobs=1).values
    for n in (2, 5):
        np.testing.assert_array_equal(zeit.pansharpen(ms, pan, method, n_jobs=n).values, ref)
