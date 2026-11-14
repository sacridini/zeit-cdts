"""zeit.starfm: STARFM spatiotemporal fusion (FORCE 3.11's starfm() for any pairs, targets, bands and radius)."""

import numpy as np
import pytest
import xarray as xr

import zeit

NY, NX = 24, 30
DATES = np.array(["2020-01-01", "2020-01-17", "2020-02-02", "2020-02-18"], dtype="datetime64[ns]")


def _coords(times, bands=None):
    c = {"time": times, "y": 5000.0 - 30.0 * np.arange(NY), "x": 1000.0 + 30.0 * np.arange(NX)}
    if bands is not None:
        c["band"] = bands
    return c


def _field(rng, nb):
    """A two-class scene with a little texture, (band, y, x) reflectance."""
    cls = (np.arange(NX)[None, :] >= NX // 2).astype(float) * np.ones((NY, 1))
    base = np.stack([0.08 + 0.05 * b + (0.25 - 0.03 * b) * cls for b in range(nb)])
    return base + 0.004 * rng.standard_normal(base.shape)


def _coarse_of(fine):
    """The coarse image on the fine grid: block means of 3 x 3 pixels."""
    out = np.empty_like(fine)
    for i in range(0, NY, 3):
        for j in range(0, NX, 3):
            out[..., i:i + 3, j:j + 3] = fine[..., i:i + 3, j:j + 3].mean(axis=(-2, -1), keepdims=True)
    return out


def _scene(nb=2, seed=3, with_band=True):
    """Fine at DATES[0], DATES[1]; coarse at all four (the last two are the targets)."""
    rng = np.random.default_rng(seed)
    f = [_field(rng, nb) for _ in range(2)]
    f[1] = f[1] + 0.03 * (np.arange(NX)[None, None, :] / NX)
    c = [_coarse_of(x) for x in f]
    c.append(_coarse_of(f[0]) + 0.05)
    c.append(_coarse_of(f[1]) + 0.02 * rng.standard_normal((nb, NY, NX)) + 0.08)
    fa, ca = np.stack(f), np.stack(c)
    bands = [f"b{k}" for k in range(nb)]
    if with_band:
        fine = xr.DataArray(fa, dims=("time", "band", "y", "x"), coords=_coords(DATES[:2], bands))
        coarse = xr.DataArray(ca, dims=("time", "band", "y", "x"), coords=_coords(DATES, bands))
    else:
        fine = xr.DataArray(fa[:, 0], dims=("time", "y", "x"), coords=_coords(DATES[:2]))
        coarse = xr.DataArray(ca[:, 0], dims=("time", "y", "x"), coords=_coords(DATES))
    return fine, coarse


def test_shapes_dims_coords_and_default_targets():
    fine, coarse = _scene(nb=2)
    fine = fine.rio.write_crs("EPSG:32722")
    out = zeit.starfm(fine, coarse, radius=2, n_jobs=2)
    assert out.dims == ("time", "band", "y", "x")
    assert out.shape == (2, 2, NY, NX)
    assert out.dtype == np.float32
    assert (out.time.values == DATES[2:]).all()
    assert (out.y.values == fine.y.values).all() and (out.x.values == fine.x.values).all()
    assert list(out.band.values) == ["b0", "b1"]
    assert out.rio.crs == fine.rio.crs
    assert np.isfinite(out.values).all()
    assert "STARFM" in out.attrs["starfm"]


def test_no_band_dimension():
    fine, coarse = _scene(nb=1, with_band=False)
    out = zeit.starfm(fine, coarse, radius=1)
    assert out.dims == ("time", "y", "x") and out.shape == (2, NY, NX)
    banded, banded_c = _scene(nb=1)
    ref = zeit.starfm(banded, banded_c, radius=1)
    np.testing.assert_array_equal(out.values, ref.values[:, 0])


def test_pair_and_target_selection():
    fine, coarse = _scene(nb=1)
    both = zeit.starfm(fine, coarse, radius=2)
    only = zeit.starfm(fine, coarse, radius=2, target=[DATES[3]])
    assert (only.time.values == DATES[3:]).all()
    np.testing.assert_array_equal(only.values[0], both.values[1])
    # a date of fine that coarse does not have is not a pair
    extra = xr.concat([fine, fine.isel(time=[0]).assign_coords(time=[np.datetime64("2019-12-01", "ns")])],
                      dim="time")
    np.testing.assert_array_equal(zeit.starfm(extra, coarse, radius=2).values, both.values)
    # one pair only (the second pair left out of both) differs from two
    one = zeit.starfm(fine.isel(time=[0]), coarse.isel(time=[0, 2, 3]), radius=2)
    assert one.shape == both.shape
    assert not np.array_equal(one.values, both.values)
    # a target that is not in coarse
    with pytest.raises(ValueError, match="not in coarse.time"):
        zeit.starfm(fine, coarse, radius=2, target=["2020-03-01"])
    # no pair
    with pytest.raises(ValueError, match="at least one pair"):
        zeit.starfm(fine, coarse.assign_coords(time=DATES + np.timedelta64(1, "D")), radius=2)
    # nothing to predict
    with pytest.raises(ValueError, match="nothing to predict"):
        zeit.starfm(fine, coarse.isel(time=[0, 1]), radius=2)


def test_radius_is_required_and_validated():
    fine, coarse = _scene(nb=1)
    with pytest.raises(TypeError):
        zeit.starfm(fine, coarse)
    with pytest.raises(ValueError, match="radius"):
        zeit.starfm(fine, coarse, radius=-1)
    with pytest.raises(ValueError, match="radius"):
        zeit.starfm(fine, coarse, radius=1.5)


def test_grid_and_band_mismatch_errors():
    fine, coarse = _scene(nb=2)
    with pytest.raises(ValueError, match="load_raster"):
        zeit.starfm(fine, coarse.isel(y=slice(0, NY - 2)), radius=1)
    with pytest.raises(ValueError, match="fine grid"):
        zeit.starfm(fine, coarse.assign_coords(x=coarse.x + 15.0), radius=1)
    with pytest.raises(ValueError, match="same, in the same order"):
        zeit.starfm(fine, coarse.isel(band=[1, 0]).assign_coords(band=["b1", "b0"]), radius=1)
    with pytest.raises(ValueError, match="band dimension"):
        zeit.starfm(fine, coarse.isel(band=0, drop=True), radius=1)
    with pytest.raises(ValueError, match="dimensions"):
        zeit.starfm(fine.rename(y="lat"), coarse, radius=1)
    with pytest.raises(TypeError):
        zeit.starfm(fine.values, coarse, radius=1)


def test_bands_are_independent():
    fine, coarse = _scene(nb=3)
    out = zeit.starfm(fine, coarse, radius=2)
    for b in range(3):
        one = zeit.starfm(fine.isel(band=[b]), coarse.isel(band=[b]), radius=2)
        np.testing.assert_array_equal(one.values[:, 0], out.values[:, b])


def test_nan_and_nodata_pixels_are_off():
    fine, coarse = _scene(nb=2)
    ref = zeit.starfm(fine, coarse, radius=2)
    f2, c2 = fine.copy(), coarse.copy()
    f2[0, 0, 4, 5] = np.nan            # fine of a pair, band 0
    c2[3, 1, 10, 11] = np.nan          # coarse at the second target, band 1
    out = zeit.starfm(f2, c2, radius=2)
    # off in every target for the pair's pixel, band 0 only
    assert np.isnan(out.values[:, 0, 4, 5]).all()
    assert np.isfinite(out.values[:, 1, 4, 5]).all()
    # off in the second target only
    assert np.isnan(out.values[1, 1, 10, 11]) and np.isfinite(out.values[0, 1, 10, 11])
    assert np.isfinite(out.values).sum() == out.size - 3
    # an off pixel is not a candidate: the others stay the same far from it
    far = (slice(None), 0, slice(15, None), slice(15, None))
    np.testing.assert_array_equal(out.values[far], ref.values[far])
    # the same with a nodata value
    f3, c3 = (fine * 10000).round().astype("int16"), (coarse * 10000).round().astype("int16")
    f3[0, 0, 4, 5] = -9999
    out3 = zeit.starfm(f3.rio.write_nodata(-9999), c3.rio.write_nodata(-9999), radius=2)
    assert np.isnan(out3.values[:, 0, 4, 5]).all()
    assert out3.dtype == np.float32


def test_scale_auto_and_explicit():
    fine, coarse = _scene(nb=2)
    ref = zeit.starfm(fine, coarse, radius=2, scale=1)
    big = zeit.starfm(fine * 10000, coarse * 10000, radius=2)            # auto: values above 2
    np.testing.assert_allclose(big.values, ref.values * 10000, rtol=2e-5, atol=2e-2)
    explicit = zeit.starfm(fine * 10000, coarse * 10000, radius=2, scale=10000)
    np.testing.assert_array_equal(big.values, explicit.values)
    # explicit scale 1 on 0-10000 data changes the thresholds: not the same
    assert not np.allclose(zeit.starfm(fine * 10000, coarse * 10000, radius=2, scale=1).values, big.values,
                           rtol=1e-3, atol=1.0)
    with pytest.raises(ValueError, match="different scales"):
        zeit.starfm(fine, coarse * 10000, radius=2)


def test_uniform_change_gives_fine_plus_change():
    # fine identical at both pairs, a coarse image that only rises by 0.05: the prediction is fine + 0.05
    fine, coarse = _scene(nb=1, seed=11)
    fine = xr.concat([fine.isel(time=0), fine.isel(time=0)], dim="time").assign_coords(time=DATES[:2])
    c0 = coarse.isel(time=0)
    coarse = xr.concat([c0, c0, c0 + 0.05, c0 - 0.03], dim="time").assign_coords(time=DATES)
    out = zeit.starfm(fine, coarse, radius=3)
    f = fine.values[0]
    tol = 2 * f.std() / 40 + 1e-4       # the candidates differ from the pixel by at most the similarity threshold
    np.testing.assert_allclose(out.values[0], f + 0.05, atol=tol)
    np.testing.assert_allclose(out.values[1], f - 0.03, atol=tol)
    # and with no change at all the fine image stays (the coarse value where no candidate) within the same bound
    same = xr.concat([c0, c0, c0, c0], dim="time").assign_coords(time=DATES)
    np.testing.assert_allclose(zeit.starfm(fine, same, radius=3).values[0], f, atol=tol)


def test_parameters_have_an_effect_and_validation():
    fine, coarse = _scene(nb=1)
    a = zeit.starfm(fine, coarse, radius=2)
    b = zeit.starfm(fine, coarse, radius=2, nclass=4)
    c = zeit.starfm(fine, coarse, radius=2, unc_fine=0.05, unc_coarse=0.05)
    assert not np.array_equal(a.values, b.values) and not np.array_equal(a.values, c.values)
    with pytest.raises(ValueError, match="nclass"):
        zeit.starfm(fine, coarse, radius=2, nclass=0)
    with pytest.raises(ValueError, match="negative"):
        zeit.starfm(fine, coarse, radius=2, unc_fine=-1)


def test_determinism_across_n_jobs():
    fine, coarse = _scene(nb=2)
    one = zeit.starfm(fine, coarse, radius=3, n_jobs=1)
    many = zeit.starfm(fine, coarse, radius=3, n_jobs=4)
    np.testing.assert_array_equal(one.values, many.values)
