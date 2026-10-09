"""SNIC superpixels: parity with the original C code and invariants.

tests/data/snic_reference_parity.npz holds, for each stored input, the seeds
the authors' reference implementation (github.com/achanta/SNIC, snic.c,
SNIC_main with doRGBtoLAB=0) placed and the labels it produced; see
make_snic_parity_fixtures.py. zeit is given the same seeds.
"""
import os

import numpy as np
import pytest
from scipy import ndimage

from zeit._core import snic as core
from zeit.segmentation import run_snic, snic_grid, snic_to_polygons

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = np.load(os.path.join(HERE, "data", "snic_reference_parity.npz"))
PARITY_CASES = sorted({k.split("__")[0] for k in PARITY.files})


# --- parity with the reference implementation ---

@pytest.mark.parametrize("name", PARITY_CASES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_reference_labels(name, dtype):
    img = PARITY[name + "__image"].astype(dtype)
    seeds = PARITY[name + "__seeds"]
    compactness = float(PARITY[name + "__compactness"])
    want = PARITY[name + "__labels"].astype(np.int32)

    res = run_snic(img, seeds=seeds, compactness=compactness)

    np.testing.assert_array_equal(res.labels, want)


# --- a direct, unoptimised SNIC as an oracle for inputs the reference cannot run ---

def _snic_oracle(img, seeds, compactness):
    """Textbook SNIC with the reference's distance and heap (heapq with an
    insertion counter breaks ties FIFO, unlike the reference's heap, so the
    oracle is only compared on inputs without distance ties)."""
    import heapq
    f, h, w = img.shape
    valid = ~np.isnan(img).any(axis=0)
    labels = np.full((h, w), -1)
    k_valid = sum(valid[r, c] for r, c in seeds)
    invwt = compactness ** 2 * k_valid / valid.sum()
    sums = np.zeros((len(seeds), f)); pos = np.zeros((len(seeds), 2)); n = np.zeros(len(seeds))
    heap, counter = [], 0
    for k, (r, c) in enumerate(seeds):
        if valid[r, c]:
            heap.append((0.0, counter, r, c, k)); counter += 1
    heapq.heapify(heap)
    while heap:
        _, _, r, c, k = heapq.heappop(heap)
        if labels[r, c] != -1:
            continue
        labels[r, c] = k
        sums[k] += img[:, r, c]; pos[k] += (r, c); n[k] += 1
        for rr, cc in ((r, c - 1), (r - 1, c), (r, c + 1), (r + 1, c)):
            if 0 <= rr < h and 0 <= cc < w and valid[rr, cc] and labels[rr, cc] == -1:
                d = (np.sum((sums[k] - img[:, rr, cc] * n[k]) ** 2)
                     + np.sum((pos[k] - np.array([rr, cc]) * n[k]) ** 2) * invwt) / n[k] ** 2
                heapq.heappush(heap, (d, counter, rr, cc, k)); counter += 1
    return labels


def test_matches_oracle_with_nan_and_custom_seeds():
    rng = np.random.default_rng(5)
    img = rng.normal(size=(4, 30, 34))
    img[:, 10:14, 5:25] = np.nan  # a masked band across the image
    img[2, 25, 30] = np.nan       # a single masked pixel, in one channel only
    seeds = np.array([[2, 3], [5, 30], [12, 8], [20, 20], [27, 5], [28, 31], [16, 16]], dtype=np.int32)
    res = run_snic(img, seeds=seeds, compactness=0.7)
    np.testing.assert_array_equal(res.labels, _snic_oracle(img, seeds, 0.7))
    # the seed on a masked pixel gives an empty segment
    assert res.sizes[2] == 0 and np.isnan(res.means[2]).all() and np.isnan(res.centroids[2]).all()
    assert np.all(res.labels[np.isnan(img).any(axis=0)] == -1)


# --- invariants ---

def _random_cube(seed=0, shape=(6, 2, 37, 41)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).cumsum(axis=-1).cumsum(axis=-2) * 0.1 + rng.normal(size=shape) * 0.05


def test_statistics_are_segment_means_centroids_and_sizes():
    cube = _random_cube()
    res = run_snic(cube, spacing=8, compactness=0.3)
    t, b = cube.shape[:2]
    assert res.means.shape == (len(res.seeds), t, b)
    lab = res.labels
    assert lab.min() >= 0
    rows, cols = np.indices(lab.shape)
    np.testing.assert_array_equal(res.sizes, np.bincount(lab.ravel(), minlength=len(res.seeds)))
    for k in range(len(res.seeds)):
        m = lab == k
        np.testing.assert_allclose(res.means[k], cube[:, :, m].mean(axis=-1), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(res.centroids[k], [rows[m].mean(), cols[m].mean()], rtol=1e-12)


def test_segments_are_4_connected_and_contain_their_seed():
    res = run_snic(_random_cube(1), spacing=7, compactness=0.2)
    for k, (r, c) in enumerate(res.seeds):
        m = res.labels == k
        _, n_components = ndimage.label(m)  # default structure = 4-connectivity
        assert n_components == 1
        assert res.labels[r, c] == k


def test_recovers_piecewise_constant_regions():
    # 4 x 4 blocks of distinct constant values, one seed per block, no spatial
    # term: every block becomes exactly one segment
    rng = np.random.default_rng(2)
    values = rng.permutation(16).reshape(4, 4).astype(float) * 10
    img = np.kron(values, np.ones((9, 9)))[None]
    seeds = np.array([[4 + 9 * i, 4 + 9 * j] for i in range(4) for j in range(4)], dtype=np.int32)
    res = run_snic(img, seeds=seeds, compactness=0.0)
    expected = np.kron(np.arange(16).reshape(4, 4), np.ones((9, 9), dtype=int))
    np.testing.assert_array_equal(res.labels, expected)


def test_time_series_separates_regions_a_single_date_cannot():
    # left half: vegetation peaking early in the year, right half: late. Their
    # temporal means are equal and at t=0 the two halves coincide: from that
    # date alone the split falls halfway between the seeds (column 15), from
    # the whole trajectory it falls on the true edge (column 20).
    t = np.arange(12)
    early = 0.5 + 0.3 * np.sin(2 * np.pi * t / 12)
    late = 0.5 - 0.3 * np.sin(2 * np.pi * t / 12)
    rng = np.random.default_rng(3)
    cube = np.empty((12, 30, 40))
    cube[:, :, :20] = early[:, None, None]
    cube[:, :, 20:] = late[:, None, None]
    cube += rng.normal(scale=0.01, size=cube.shape)
    seeds = np.array([[15, 5], [15, 25]], dtype=np.int32)

    res = run_snic(cube, seeds=seeds, compactness=1.0)
    assert np.all(res.labels[:, :20] == 0) and np.all(res.labels[:, 20:] == 1)
    np.testing.assert_allclose(res.means[0], early, atol=0.01)

    one_date = run_snic(cube[0], seeds=seeds, compactness=1.0)
    assert (one_date.labels[:, :20] == 1).sum() >= 60  # >= 2 of the 20 columns


def test_tiles_equal_independent_runs_and_are_deterministic():
    cube = _random_cube(4, shape=(3, 50, 70)).astype(np.float32)
    seeds = snic_grid(cube.shape[1:], spacing=6)
    tiled = run_snic(cube, seeds=seeds, compactness=0.4, tile_size=(20, 32), n_jobs=4)
    # every tile is a stand-alone SNIC run on its own seeds, labels = seed ids
    for r0 in range(0, 50, 20):
        for c0 in range(0, 70, 32):
            sub = cube[:, r0:r0 + 20, c0:c0 + 32]
            inside = ((seeds[:, 0] >= r0) & (seeds[:, 0] < r0 + 20)
                      & (seeds[:, 1] >= c0) & (seeds[:, 1] < c0 + 32))
            ids = np.flatnonzero(inside)
            alone = run_snic(sub, seeds=seeds[ids] - [r0, c0], compactness=0.4)
            np.testing.assert_array_equal(tiled.labels[r0:r0 + 20, c0:c0 + 32], ids[alone.labels])
    for n_jobs in (1, 2, 8):
        again = run_snic(cube, seeds=seeds, compactness=0.4, tile_size=(20, 32), n_jobs=n_jobs)
        np.testing.assert_array_equal(again.labels, tiled.labels)
    whole = run_snic(cube, seeds=seeds, compactness=0.4, tile_size=1000)
    np.testing.assert_array_equal(whole.labels, run_snic(cube, seeds=seeds, compactness=0.4).labels)


def test_degenerate_inputs():
    # one seed labels everything (the reference crashes here)
    img = np.random.default_rng(6).normal(size=(2, 9, 7))
    assert np.all(run_snic(img, seeds=[[4, 3]]).labels == 0)
    # the reference returns a non-existent label 1 for this 2 x 2 image
    assert np.all(run_snic(img[:, :2, :2], seeds=[[1, 1]]).labels == 0)
    # a 1-pixel-high image
    assert np.all(run_snic(img[:, :1, :], seeds=[[0, 1], [0, 5]]).labels >= 0)
    # a sits grid on a small image
    assert np.all(run_snic(img[:, :, :], spacing=3).labels >= 0)
    # a valid island cut off by NaNs with no seed stays unlabelled
    isl = np.ones((1, 9, 9))
    isl[0, :, 4] = np.nan
    res = run_snic(isl, seeds=[[4, 1]])
    assert np.all(res.labels[:, :4] == 0) and np.all(res.labels[:, 5:] == -1)
    # all NaN
    res = run_snic(np.full((1, 4, 4), np.nan), seeds=[[1, 1]])
    assert np.all(res.labels == -1) and res.sizes[0] == 0


def test_rejects_bad_arguments():
    img = np.zeros((1, 5, 5))
    with pytest.raises(ValueError):
        run_snic(img, seeds=[[5, 0]])
    with pytest.raises(ValueError):
        run_snic(img, seeds=[[0, 0]], compactness=-1)
    with pytest.raises(ValueError):
        run_snic(np.zeros(5))
    with pytest.raises(ValueError):
        snic_grid((5, 5), spacing=2, type="triangular")
    with pytest.raises(ValueError):
        snic_grid((10, 10), spacing=4, padding=5)


# --- seed grids (R snic::snic_grid, used by sits_snic) ---

def test_rectangular_grid_matches_r_snic():
    # R: n = floor((50 - 2*5 - 1) / 10) + 1 = 4; seq(6, 45, length.out = 4)
    #    = 6, 19, 32, 45 (1-based) -> 5, 18, 31, 44; expand.grid order (rows fastest)
    seeds = snic_grid((50, 50), spacing=10)
    axis = [5, 18, 31, 44]
    expected = np.array([[r, c] for c in axis for r in axis])
    np.testing.assert_array_equal(seeds, expected)
    # one position per axis falls back to the centre: mean(6, 15) = 10.5 -> round half even 10
    np.testing.assert_array_equal(snic_grid((20, 20), spacing=10), [[9, 9]])


@pytest.mark.parametrize("grid", ["rectangular", "diamond", "hexagonal", "random"])
def test_grids_stay_inside_padding(grid):
    h, w, pad = 60, 80, 4
    seeds = snic_grid((h, w), spacing=9, padding=pad, type=grid, random_state=0)
    assert seeds.dtype == np.int32 and len(seeds) > 0
    assert seeds[:, 0].min() >= pad - 1 and seeds[:, 0].max() <= h - pad
    assert seeds[:, 1].min() >= pad - 1 and seeds[:, 1].max() <= w - pad
    if grid == "random":
        assert len(seeds) == len(snic_grid((h, w), spacing=9, padding=pad))
        assert len(np.unique(seeds, axis=0)) == len(seeds)


# --- xarray and vector outputs ---

def test_xarray_accessor():
    xr = pytest.importorskip("xarray")
    import zeit.xarray_api  # noqa: F401

    cube = _random_cube(7, shape=(5, 3, 24, 30))
    da = xr.DataArray(cube, dims=("time", "band", "y", "x"),
                      coords={"time": np.arange(5), "band": ["red", "nir", "swir"],
                              "y": np.arange(24)[::-1] * 10.0, "x": np.arange(30) * 10.0})
    ds = da.zeit.snic(spacing=6, compactness=0.3)
    ref = run_snic(cube, spacing=6, compactness=0.3)
    np.testing.assert_array_equal(ds["labels"].values, ref.labels)
    assert ds["means"].dims == ("segment", "time", "band")
    assert list(ds["band"].values) == ["red", "nir", "swir"]
    np.testing.assert_allclose(ds["means"].values, ref.means)


def test_polygons_cover_segments():
    pytest.importorskip("geopandas")
    from rasterio.transform import from_origin

    res = run_snic(_random_cube(8, shape=(4, 30, 36)), spacing=8, compactness=0.3)
    gdf = snic_to_polygons(res, include_means=True)
    assert len(gdf) == int((res.sizes > 0).sum())
    np.testing.assert_allclose(gdf.geometry.area.values, res.sizes[gdf["supercells"].values])
    np.testing.assert_allclose(gdf["f0"].values, res.means[gdf["supercells"].values, 0])

    tr = from_origin(500000.0, 8000000.0, 10.0, 10.0)
    gdf = snic_to_polygons(res, transform=tr, crs="EPSG:32720")
    k = int(gdf["supercells"].iloc[0])
    r, c = res.centroids[k]
    assert gdf["x"].iloc[0] == pytest.approx(500000.0 + (c + 0.5) * 10.0)
    assert gdf["y"].iloc[0] == pytest.approx(8000000.0 - (r + 0.5) * 10.0)
    assert gdf.geometry.area.sum() == pytest.approx(res.sizes.sum() * 100.0)
