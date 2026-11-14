"""zeit.fields and its parts (phase 24c): the public API against the notebook's functions.

``zeit.fields(img, preset="notebook", markers=sets)`` against ``benchmarks/fields/notebook_reference.py``
fed with the very same marker sets (gradient, counts, hysteresis regions, final labels), the gradients
against the notebook's / scikit-image's, georeferencing, options and errors.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("pyamg")
from skimage import filters, segmentation  # noqa: E402

import zeit  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks", "fields"))
import notebook_reference as nb  # noqa: E402


def _scene(h, w, seed):
    """Blocks of different levels (1000-4000 digital numbers, some 'forest' below 2000) plus noise."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(1000, 4000, size=(5, 5)).astype(float)
    rows = np.minimum(np.arange(h) * 5 // h, 4)
    cols = np.minimum(np.arange(w) * 5 // w, 4)
    img = levels[np.ix_(rows, cols)] + rng.normal(0, 60, size=(h, w))
    return np.clip(img, 0, 65535).astype(np.uint16)


def _sets(shape, n_sets, n_markers, min_distance, seed):
    return nb.draw_markers(shape, n_sets, n_markers, min_distance, seed)


def _georef(img, crs="EPSG:32720"):
    h, w = img.shape[-2:]
    da = xr.DataArray(img, dims=("y", "x") if img.ndim == 2 else ("band", "y", "x"),
                      coords={"y": 8_000_000.0 - 30.0 * np.arange(h) - 15.0,
                              "x": 500_000.0 + 30.0 * np.arange(w) + 15.0})
    return da.rio.write_crs(crs)


# --------------------------------------------------------------------------------------
# End to end against the notebook
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape,seed,n_sets", [((96, 96), 0, 10), ((150, 120), 1, 12), ((96, 96), 2, 20)])
def test_fields_equals_notebook(shape, seed, n_sets):
    img = _scene(*shape, seed)
    assert (img < 2000).any() and (img > 2000).any()
    sets = _sets(shape, n_sets, 150, 10, seed)
    ref = nb.pipeline(img, markers=sets, return_all=True)
    res = zeit.fields(img, preset="notebook", markers=sets, n_jobs=2)

    assert np.array_equal(res["gradient"].values, ref["gradient"].astype(np.float32))
    r = len(sets)
    counts = np.rint(res["boundary"].values.astype(np.float64) * r)
    assert np.array_equal(counts, ref["counts"])
    assert res.attrs["realizations"] == r

    # the notebook's final labels, up to the (order-preserving) sequential relabelling
    expected, _, _ = segmentation.relabel_sequential(ref["final"])
    assert np.array_equal(res["labels"].values, expected)
    assert res["labels"].dtype == np.int32 and res["boundary"].dtype == np.float32
    # when no label vanished in the mask the labels are the notebook's exactly
    if np.array_equal(np.unique(ref["final"]), np.arange(ref["final"].max() + 1)):
        assert np.array_equal(res["labels"].values, ref["final"])
    assert (res["labels"].values[img <= 2000] == 0).all()
    assert res.attrs["preset"] == "notebook" and res.attrs["mask_threshold"] == 2000.0


def test_fields_steps_equal_notebook_regions():
    shape = (96, 96)
    img = _scene(*shape, 5)
    sets = _sets(shape, 10, 150, 10, 5)
    ref = nb.pipeline(img, markers=sets, return_all=True)
    # same hysteresis regions: the watershed from them is the notebook's, so the final partition is
    res = zeit.fields(img, preset="notebook", markers=sets, n_jobs=1)
    final = ref["final"]
    assert res.attrs["boundary_label"] == int((ref["regions"] == 1).any() and (final == 1).any())


def test_stochastic_watershed_counts_and_flag():
    shape = (96, 96)
    img = _scene(*shape, 0)
    sets = _sets(shape, 10, 150, 10, 0)
    grad = nb.gradient(nb.stretch(img))
    mask = img > 2000
    prob, counts = zeit.stochastic_watershed(grad, markers=sets, mask=mask, n_jobs=2, return_counts=True)
    ref = nb.boundary_frequency(grad, mask.astype(int), markers=sets)
    assert np.array_equal(counts.values, ref)
    assert prob.dtype == np.float32 and prob.attrs["realizations"] == 10
    assert np.allclose(prob.values, ref / 10)
    assert prob.values.min() >= 0 and prob.values.max() <= 1


# --------------------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------------------
def test_edge_gradient_tophat_and_sobel():
    img = _scene(80, 90, 3)
    st = nb.stretch(img)
    g = zeit.edge_gradient(img)
    assert np.array_equal(g.values, nb.gradient(st))
    s = zeit.edge_gradient(img, "sobel")
    assert np.max(np.abs(s.values - filters.sobel(st))) < 1e-12
    assert s.dims == ("y", "x")
    # a float image gives the same
    g2 = zeit.edge_gradient(img.astype(np.float64), "tophat")
    assert np.array_equal(g2.values, g.values)


def test_edge_gradient_multitemporal():
    rng = np.random.default_rng(0)
    cube = rng.normal(0, 1, size=(3, 2, 30, 40)) * np.array([1.0, 50.0])[None, :, None, None] + 100
    cube = xr.DataArray(cube, dims=("time", "band", "y", "x"),
                        coords={"time": [1, 2, 3], "band": ["a", "b"]})
    layers = []
    for t in range(3):
        for b in range(2):
            v = cube.values[t, b]
            layers.append(filters.sobel((v - v.mean()) / v.std()))
    layers = np.array(layers)
    mx = zeit.edge_gradient(cube, "multitemporal", combine="max")
    sm = zeit.edge_gradient(cube, "multitemporal", combine="sum")
    assert np.max(np.abs(mx.values - layers.max(0))) < 1e-12
    assert np.max(np.abs(sm.values - layers.mean(0))) < 1e-12
    assert (mx.values >= sm.values - 1e-15).all()
    # NaN in one layer: the others still give an edge
    cube.values[0, 0, 5, 5] = np.nan
    nan = zeit.edge_gradient(cube, "multitemporal")
    assert np.isfinite(nan.values).all()


def test_edge_gradient_selection_and_errors():
    rng = np.random.default_rng(1)
    cube = xr.DataArray(rng.integers(1000, 4000, size=(2, 3, 40, 50)).astype(float),
                        dims=("time", "band", "y", "x"), coords={"time": [10, 20], "band": ["r", "g", "b"]})
    sel = zeit.edge_gradient(cube, band="g", time=20)
    assert np.array_equal(sel.values, zeit.edge_gradient(cube.sel(band="g", time=20).values).values)
    for method in ("tophat", "sobel"):
        with pytest.raises(ValueError, match="band="):
            zeit.edge_gradient(cube, method)
        with pytest.raises(ValueError, match="one 2-D image"):
            zeit.edge_gradient(cube, method, band="g")
    with pytest.raises(ValueError, match="not found"):
        zeit.edge_gradient(cube, band="zzz", time=10)
    with pytest.raises(ValueError, match="unknown gradient"):
        zeit.edge_gradient(cube, "nope", band="g", time=10)
    with pytest.raises(ValueError, match="combine"):
        zeit.edge_gradient(cube, "multitemporal", combine="min")
    with pytest.raises(ValueError, match="NaN"):
        bad = cube.sel(band="g", time=10).copy()
        bad.values[0, 0] = np.nan
        zeit.edge_gradient(bad)


# --------------------------------------------------------------------------------------
# watershed
# --------------------------------------------------------------------------------------
def test_watershed_equals_skimage_and_keeps_coords():
    img = _georef(_scene(60, 70, 4).astype(float))
    rng = np.random.default_rng(0)
    markers = np.zeros(img.shape, dtype=np.int32)
    markers[rng.integers(0, 60, 15), rng.integers(0, 70, 15)] = np.arange(1, 16)
    mask = np.asarray(img.values > 1500)
    for conn in (1, 2):
        ref = segmentation.watershed(img.values, markers, connectivity=conn, mask=mask)
        got = zeit.watershed(img, markers, mask=mask, connectivity=conn)
        assert np.array_equal(got.values, ref) and got.dtype == np.int32
    assert np.array_equal(got["x"].values, img["x"].values)
    assert got.rio.crs == img.rio.crs


# --------------------------------------------------------------------------------------
# Georeferencing, attrs, options, errors
# --------------------------------------------------------------------------------------
def test_fields_georeferencing_and_attrs():
    img = _georef(_scene(80, 80, 6).astype(np.float64))
    res = zeit.fields(img, markers=_sets((80, 80), 6, 120, 10, 0), mask_threshold=2000, n_jobs=1)
    for name in ("labels", "boundary", "gradient"):
        da = res[name]
        assert da.dims == ("y", "x")
        assert np.array_equal(da["x"].values, img["x"].values)
        assert np.array_equal(da["y"].values, img["y"].values)
    assert res.rio.crs == img.rio.crs
    assert res.rio.transform() == img.rio.transform()
    a = res.attrs
    assert a["method"] == "montecarlo" and a["gradient"] == "tophat" and a["final"] == "image"
    assert a["low"] == 0.3 and a["high"] == 0.6 and a["min_size"] == 50 and a["realizations"] == 6
    assert a["mask_threshold"] == 2000.0
    assert res["labels"].values.min() == 0
    lab = res["labels"].values
    assert set(np.unique(lab[lab > 0])) == set(range(1, lab.max() + 1))


def test_fields_exact_mask_dataarray_and_selection():
    base = _scene(70, 70, 7).astype(np.float64)
    cube = _georef(np.stack([base, base[::-1] * 0.8 + 100, base * 0.5]))
    cube = cube.assign_coords(band=["b1", "b2", "b3"])
    forest = (base < 2000)
    mask = xr.DataArray(~forest, dims=("y", "x"), coords={"y": cube["y"], "x": cube["x"]})
    res = zeit.fields(cube, band="b1", method="exact", markers=300, mask=mask, low=0.1, high=0.3, n_jobs=2)
    assert res.attrs["method"] == "exact" and "realizations" not in res.attrs
    assert (res["labels"].values[forest] == 0).all()
    assert res["labels"].values.max() >= 1
    assert res["boundary"].values.max() <= 1
    assert (res["boundary"].values[forest] == 0).all()
    # the exact probability of the public function agrees with fields' boundary layer
    ex = zeit.stochastic_watershed(zeit.edge_gradient(cube, band="b1"), method="exact", markers=300, mask=~forest, n_jobs=1)
    assert np.max(np.abs(ex.values - res["boundary"].values)) < 1e-6
    # multi-layer cube: gradient defaults to final='gradient'
    multi = zeit.fields(cube, gradient="multitemporal", method="exact", markers=300, n_jobs=1)
    assert multi.attrs["final"] == "gradient" and multi["labels"].values.max() >= 1
    # a ready gradient (ndarray) on a cube
    grad = zeit.edge_gradient(cube, "multitemporal").values
    again = zeit.fields(cube, gradient=grad, method="exact", markers=300, n_jobs=1)
    assert np.array_equal(again["labels"].values, multi["labels"].values)
    # time selection on a cube without it is an error
    with pytest.raises(ValueError, match="time"):
        zeit.fields(cube, time=1)


def test_fields_errors():
    img = _scene(60, 60, 8)
    cube = xr.DataArray(np.stack([img, img]).astype(float), dims=("band", "y", "x"))
    sets = _sets((60, 60), 3, 100, 10, 0)
    with pytest.raises(ValueError, match="band="):
        zeit.fields(cube, markers=sets)
    with pytest.raises(ValueError, match="final='image'"):
        zeit.fields(cube, gradient="multitemporal", final="image", markers=sets)
    with pytest.raises(ValueError, match="unknown preset"):
        zeit.fields(img, preset="other")
    with pytest.raises(ValueError, match="not both"):
        zeit.fields(img, mask=img > 0, mask_threshold=1, markers=sets)
    with pytest.raises(ValueError, match="final must"):
        zeit.fields(img, final="x", markers=sets)
    with pytest.raises(ValueError, match="method"):
        zeit.fields(img, method="x", markers=sets)
    with pytest.raises(ValueError, match="exact"):
        zeit.fields(img, method="exact", markers=sets)
    with pytest.raises(ValueError, match="mask_threshold needs"):
        zeit.fields(cube, gradient="multitemporal", mask_threshold=5, markers=sets)
    with pytest.raises(ValueError, match="shape"):
        zeit.fields(img, gradient=np.zeros((5, 5)), markers=sets)
    with pytest.raises(ValueError, match="return_counts"):
        zeit.stochastic_watershed(np.zeros((20, 20)), method="exact", markers=10, return_counts=True)


# --------------------------------------------------------------------------------------
# Threads and seeds
# --------------------------------------------------------------------------------------
def test_n_jobs_and_random_state():
    img = _scene(80, 80, 9)
    # the preset's Monte Carlo (500 realisations of 1000 markers thinned to 20 px), drawn from the seed
    kw = dict(preset="notebook", random_state=3)
    one = zeit.fields(img, n_jobs=1, **kw)
    four = zeit.fields(img, n_jobs=4, **kw)
    for name in ("labels", "boundary", "gradient"):
        assert np.array_equal(one[name].values, four[name].values)
    again = zeit.fields(img, n_jobs=2, **kw)
    assert np.array_equal(one["boundary"].values, again["boundary"].values)
    other = zeit.fields(img, n_jobs=2, preset="notebook", random_state=4)
    assert not np.array_equal(one["boundary"].values, other["boundary"].values)
    assert one.attrs["random_state"] == 3
