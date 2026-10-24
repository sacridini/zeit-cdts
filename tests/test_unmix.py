"""zeit.unmix with endmembers of one's own (pixels of the image, a hyperspectral scene): the fractions of known
mixtures, a pixel taken as an endmember unmixed into itself, the constraints."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit

NM = np.linspace(450, 2400, 40)
ENDMEMBERS = {
    "vegetation": np.clip(0.05 + 0.45 / (1 + np.exp(-(NM - 720) / 15)) - 0.2 * (NM > 1400), 0.01, None),
    "soil": 0.1 + 0.25 * (NM - 450) / 1950,
    "water": 0.08 * np.exp(-(NM - 450) / 300),
}


def _scene(fractions):
    """A (band, y, x) scene of the endmembers mixed by `fractions` (k, y, x)."""
    e = np.stack(list(ENDMEMBERS.values()))
    cube = np.einsum("kb,kyx->byx", e, fractions)
    return xr.DataArray(cube, dims=("band", "y", "x"), coords={"band": [f"{w:.0f} nm" for w in NM]})


def test_hyperspectral_scene_recovers_the_fractions():
    rng = np.random.default_rng(0)
    f = rng.dirichlet(np.ones(3), size=(5, 4)).transpose(2, 0, 1)
    scene = _scene(f)
    table = pd.DataFrame(ENDMEMBERS, index=list(scene.band.values)).T
    res = zeit.unmix(scene, table, ndfi=False)
    for j, name in enumerate(ENDMEMBERS):
        np.testing.assert_allclose(res[name].values, f[j], atol=1e-5)
    assert float(res.rmse.max()) < 1e-6


def test_a_pixel_as_endmember_is_itself():
    rng = np.random.default_rng(1)
    f = rng.dirichlet(np.ones(3), size=(6, 6)).transpose(2, 0, 1)
    scene = _scene(f)
    pins = {"a": (0, 0), "b": (2, 3), "c": (5, 5)}
    table = pd.DataFrame({k: scene.values[:, r, c] for k, (r, c) in pins.items()}, index=list(scene.band.values)).T
    for sum_to_one, nonneg in [(True, True), (False, True)]:
        res = zeit.unmix(scene, table, sum_to_one=sum_to_one, nonneg=nonneg, ndfi=False)
        for k, (r, c) in pins.items():
            for other in pins:
                want = 1.0 if other == k else 0.0
                assert float(res[other].values[r, c]) == pytest.approx(want, abs=1e-9), (k, other)


def test_series_with_band_names_and_constraints():
    # a (time, band, y, x) series of 3 bands in reflectance x 10000, endmembers as a dict of dicts
    e = {"bright": {"b1": 3000, "b2": 4000, "b3": 5000}, "dark": {"b1": 200, "b2": 150, "b3": 100}}
    f = np.array([0.25, 0.75])
    spectrum = f[0] * np.array([3000, 4000, 5000]) + f[1] * np.array([200, 150, 100])
    cube = xr.DataArray(np.tile(spectrum, (2, 1))[:, :, None, None], dims=("time", "band", "y", "x"),
                        coords={"band": ["b1", "b2", "b3"]})
    res = zeit.unmix(cube, e, ndfi=False)
    assert res.attrs["scale"] == 10000.0
    np.testing.assert_allclose(res.bright.values.ravel(), 0.25, atol=1e-9)
    np.testing.assert_allclose(res.dark.values.ravel(), 0.75, atol=1e-9)
    # a spectrum outside the endmembers: negative without the constraint, 0 with it
    outside = xr.DataArray(np.array([100.0, 50.0, 20.0])[:, None, None], dims=("band", "y", "x"),
                           coords={"band": ["b1", "b2", "b3"]})
    free = zeit.unmix(outside, e, sum_to_one=False, nonneg=False, scale=10000.0, ndfi=False)
    assert float(free.bright.squeeze()) < 0
    fcls = zeit.unmix(outside, e, scale=10000.0, ndfi=False)
    assert float(fcls.bright.squeeze()) == 0.0 and float(fcls.dark.squeeze()) == pytest.approx(1.0)


def test_hundreds_of_bands_named_by_wavelength():
    # EnMAP's 224 bands, labelled by their wavelengths (numbers): the solver's products over 128 bands or more go
    # by blocks of bands, and the endmember table's numeric columns are the cube's bands
    rng = np.random.default_rng(4)
    nm = np.linspace(420.0, 2450.0, 224)
    e = np.stack([0.05 + 0.4 / (1 + np.exp(-(nm - 720) / 20)), 0.1 + 0.25 * nm / 2450, 0.08 * np.exp(-nm / 600)])
    f = rng.dirichlet(np.ones(3), size=(4, 5)).transpose(2, 0, 1)
    scene = xr.DataArray(np.einsum("kb,kyx->byx", e, f), dims=("band", "y", "x"), coords={"band": nm})
    table = pd.DataFrame(e, index=["veg", "soil", "water"], columns=nm)
    for sum_to_one, nonneg in [(True, True), (False, True), (True, False), (False, False)]:
        res = zeit.unmix(scene, table, sum_to_one=sum_to_one, nonneg=nonneg, ndfi=False)
        for j, name in enumerate(table.index):
            np.testing.assert_allclose(res[name].values, f[j], atol=1e-6)


def test_repeated_bands_are_named():
    scene = xr.DataArray(np.ones((3, 2, 2)), dims=("band", "y", "x"), coords={"band": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="repeats the bands"):
        zeit.unmix(scene, pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["x", "y"], columns=["a", "b"]), ndfi=False)
