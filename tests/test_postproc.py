"""Post-processing of classifications: sits' Bayesian smoothing (parity with sits 1.5.4), the
majority filter, the sieve of small patches and the temporal consistency of class series."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import ndimage

import zeit

DATA = Path(__file__).parent / "data"


def test_smooth_probs_equals_sits():
    ref = json.loads((DATA / "smooth_probs_sits.json").read_text())
    for case in ref["cases"]:
        ny, nx, nc = case["nrows"], case["ncols"], case["nclasses"]
        probs = np.array(case["probs"], dtype=float).reshape(ny * nx, nc).T.reshape(nc, ny, nx)
        expected = np.array(case["smoothed"], dtype=float).reshape(ny * nx, nc).T.reshape(nc, ny, nx)
        got = zeit.smooth_probs(probs, window_size=case["window"], neigh_fraction=case["neigh_fraction"],
                                smoothness=case["smoothness"], n_jobs=2)
        missing = np.isnan(probs).any(axis=0)
        assert missing.sum() == len(case["missing"])
        # sits gives a missing pixel its neighbours' mean; zeit leaves it missing
        assert np.isnan(got[:, missing]).all()
        np.testing.assert_allclose(got[:, ~missing], expected[:, ~missing], rtol=0, atol=1e-13)


def test_smooth_probs_keeps_confident_pixels_and_removes_noise():
    rng = np.random.default_rng(4)
    ny, nx = 40, 40
    truth = (np.arange(nx)[None, :] >= 20).astype(int).repeat(ny, axis=0)
    p1 = np.where(truth == 1, 0.8, 0.2) + rng.normal(0, 0.25, (ny, nx))
    p1 = np.clip(p1, 0.01, 0.99)
    probs = np.stack([1 - p1, p1])
    before = (probs.argmax(0) != truth).mean()
    da = xr.DataArray(probs, dims=("class", "y", "x"), coords={"class": ["forest", "pasture"]}, attrs={"a": 1})
    smoothed = zeit.smooth_probs(da, window_size=7, smoothness=10)
    assert isinstance(smoothed, xr.DataArray) and smoothed.dims == da.dims and smoothed.attrs == {"a": 1}
    after = (smoothed.values.argmax(0) != truth).mean()
    assert after < before / 3
    # more smoothness, a smoother field
    stronger = zeit.smooth_probs(probs, window_size=7, smoothness=200)
    assert np.abs(np.diff(stronger[1], axis=0)).mean() < np.abs(np.diff(smoothed.values[1], axis=0)).mean()
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        zeit.smooth_probs(probs * 10000)
    with pytest.raises(ValueError, match="odd"):
        zeit.smooth_probs(probs, window_size=4)
    with pytest.raises(ValueError, match="one smoothness per class"):
        zeit.smooth_probs(probs, smoothness=[1, 2, 3])


def _mode_reference(m, size):
    """The majority filter by brute force: the window cut at the edges, the pixel's own class on
    a tie, else the smallest of the most frequent."""
    ny, nx = m.shape
    leg = size // 2
    out = m.copy()
    for i in range(ny):
        for j in range(nx):
            if np.isnan(m[i, j]):
                continue
            w = m[max(0, i - leg):i + leg + 1, max(0, j - leg):j + leg + 1]
            w = w[~np.isnan(w)]
            vals, counts = np.unique(w, return_counts=True)
            top = counts.max()
            own = counts[vals == m[i, j]][0]
            out[i, j] = m[i, j] if own == top else vals[counts == top].min()
    return out


def test_majority_filter_equals_brute_force():
    rng = np.random.default_rng(1)
    m = rng.integers(1, 5, (23, 31)).astype(float)
    m[rng.random(m.shape) < 0.05] = np.nan
    for size in (3, 5):
        np.testing.assert_array_equal(zeit.apply_majority_filter(m, size), _mode_reference(m, size))
    # many classes (more than a window's pixels)
    many = rng.integers(0, 60, (15, 15)).astype(float)
    np.testing.assert_array_equal(zeit.apply_majority_filter(many, 3), _mode_reference(many, 3))
    # iterations: twice is the filter applied to its result
    once = zeit.apply_majority_filter(m, 3)
    np.testing.assert_array_equal(zeit.apply_majority_filter(m, 3, iterations=2), _mode_reference(once, 3))


def test_majority_filter_nodata_and_dtype():
    m = np.ones((6, 6), dtype=np.uint8)
    m[2, 2] = 7
    m[0, :] = 255
    da = xr.DataArray(m, dims=("y", "x")).rio.write_nodata(255)
    out = zeit.apply_majority_filter(da, 3)
    assert out.dtype == np.uint8 and out.rio.nodata == 255
    assert out.values[2, 2] == 1 and (out.values[0] == 255).all()


def _patch_sizes(m, connectivity):
    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    sizes = []
    for c in np.unique(m[~np.isnan(m)]):
        lab, n = ndimage.label(m == c, structure=structure)
        sizes += list(np.bincount(lab.ravel())[1:])
    return np.array(sizes)


@pytest.mark.parametrize("connectivity", [4, 8])
def test_mmu_sieve_leaves_no_small_patch(connectivity):
    rng = np.random.default_rng(7)
    # blocky classes with salt-and-pepper
    m = np.kron(rng.integers(1, 4, (8, 8)), np.ones((6, 6))).astype(float)
    noise = rng.random(m.shape) < 0.08
    m[noise] = rng.integers(1, 6, noise.sum())
    m[:3, :3] = np.nan
    out = zeit.mmu_sieve(m, min_size=6, connectivity=connectivity)
    assert np.isnan(out[:3, :3]).all() and not np.isnan(out[~np.isnan(m)]).any()
    assert _patch_sizes(out, connectivity).min() >= 6
    # what changed was in small patches only, and big patches keep their class
    assert (out != m)[~np.isnan(m)].mean() < 0.15
    assert set(np.unique(out[~np.isnan(out)])) <= set(np.unique(m[~np.isnan(m)]))


def test_mmu_sieve_merges_into_the_longest_border():
    m = np.array([
        [1, 1, 1, 2, 2, 2],
        [1, 3, 1, 2, 2, 2],
        [1, 1, 1, 4, 2, 2],
        [1, 1, 1, 4, 2, 2],
        [1, 1, 1, 2, 2, 2],
    ])
    da = xr.DataArray(m, dims=("y", "x"), coords={"y": np.arange(5.0), "x": np.arange(6.0)})
    out = zeit.mmu_sieve(da, min_size=3, connectivity=4)
    assert isinstance(out, xr.DataArray) and out.dtype == m.dtype
    assert out.values[1, 1] == 1 and (out.values[2:4, 3] == 2).all()
    # an island alone in NoData stays
    iso = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=np.int16)
    np.testing.assert_array_equal(zeit.mmu_sieve(iso, 3, nodata=0), iso)
    with pytest.raises(ValueError, match="4 or 8"):
        zeit.mmu_sieve(m, 3, connectivity=6)


def test_desawtooth_classes():
    series = np.array([1, 2, 1, 1, 3, 3, 1, 5, 1])
    np.testing.assert_array_equal(zeit.desawtooth_classes(series), [1, 1, 1, 1, 3, 3, 1, 1, 1])
    np.testing.assert_array_equal(zeit.desawtooth_classes(series, max_length=2), np.ones(9))
    # a cube, time not first, NoData dates skipped
    rng = np.random.default_rng(3)
    cube = rng.integers(1, 4, (6, 5, 4)).astype(np.int16)
    cube[2, 0, 0] = -1
    da = xr.DataArray(np.moveaxis(cube, 0, -1), dims=("y", "x", "time"),
                      coords={"time": pd.date_range("2018-01-01", periods=6, freq="YS")}).rio.write_nodata(-1)
    out = zeit.desawtooth_classes(da)
    assert out.dims == ("y", "x", "time") and out.dtype == np.int16 and out.values[0, 0, 2] == -1
    for i in range(5):
        for j in range(4):
            s = cube[:, i, j]
            ok = s != -1
            np.testing.assert_array_equal(out.values[i, j][ok], zeit.desawtooth_classes(s[ok].astype(float)))
