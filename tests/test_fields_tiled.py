"""zeit.fields_tiled (phase 24g): overlapping tiles stitched, against zeit.fields on the whole image."""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pytest
import rasterio
import xarray as xr

import zeit
from zeit import cli

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyproj")

H, W = 600, 700


def _scene(h=H, w=W, seed=1):
    """Rectangular fields of 50 to 110 pixels, each its own level (1000-4000), plus noise."""
    rng = np.random.default_rng(seed)
    rb = np.cumsum(rng.integers(50, 110, 20))
    cb = np.cumsum(rng.integers(50, 110, 20))
    rows = np.searchsorted(rb, np.arange(h), side="right")
    cols = np.searchsorted(cb, np.arange(w), side="right")
    levels = rng.integers(1000, 4000, size=(rows.max() + 1, cols.max() + 1)).astype(float)
    img = levels[np.ix_(rows, cols)] + rng.normal(0, 60, (h, w))
    da = xr.DataArray(np.clip(img, 0, 65535).astype(np.uint16), dims=("y", "x"),
                      coords={"y": 8e6 - 30.0 * np.arange(h) - 15, "x": 5e5 + 30.0 * np.arange(w) + 15})
    return da.rio.write_crs("EPSG:32720")


def _read(path):
    with rasterio.open(path) as ds:
        return ds.read(1)


def _matched(a, b, thr):
    """The fraction of ``a``'s pixels lying in a field of ``a`` that has a field of ``b`` with IoU >= ``thr``
    (a pairing that is one to one for thr > 0.5)."""
    _, ai = np.unique(a.ravel(), return_inverse=True)
    _, bi = np.unique(b.ravel(), return_inverse=True)
    na_, nb_ = ai.max() + 1, bi.max() + 1
    co = np.bincount(ai * nb_ + bi, minlength=na_ * nb_).reshape(na_, nb_)
    sa, sb = co.sum(1), co.sum(0)
    ok = (co / (sa[:, None] + sb[None, :] - co)).max(1) >= thr
    return float((sa * ok).sum() / sa.sum())


EXACT = dict(method="exact", marker_density=2000.0, low=0.5, high=0.6)


@pytest.fixture(scope="module")
def exact_run(tmp_path_factory):
    da = _scene()
    out = tmp_path_factory.mktemp("exact")
    whole = zeit.fields(da, n_jobs=2, **EXACT)
    res = zeit.fields_tiled(da, out, tile=256, overlap=100, n_jobs=2, **EXACT)
    return da, whole, res, out


def test_exact_probability_of_the_cores_is_the_whole_images(exact_run):
    _, whole, res, out = exact_run
    assert res["tiles"] == (3, 3) and res["overlap"] == 100
    prob, ref = _read(out / "boundary.tif"), whole["boundary"].values
    assert prob.dtype == np.float32 and prob.shape == (H, W)
    d = np.abs(prob - ref)[50:-50, 50:-50]  # the image's own edge is reflected in the whole, not in a tile
    # the exact probability of a window is the whole's up to the basins cut by the window: 0.5 % of the
    # pixels differ by more than 0.02 (up to 0.07) at 2000 markers per million pixels, the mean is 1e-4
    assert d.mean() < 5e-4 and (d > 0.02).mean() < 0.01 and d.max() < 0.1


def test_exact_labels_are_the_whole_images_partition(exact_run):
    _, whole, res, out = exact_run
    lab, ref = _read(out / "labels.tif"), whole["labels"].values
    assert lab.dtype == np.int32 and res["boundary_label"] == whole.attrs["boundary_label"] == 1
    assert abs(res["n_fields"] - (ref.max() - 1)) <= 2
    assert _matched(ref, lab, 0.9) >= 0.99 and _matched(lab, ref, 0.9) >= 0.99
    assert set(np.unique(lab)) == set(range(1, lab.max() + 1)) and (lab == 1).sum() > 0  # sequential
    assert (lab == 1).sum() == pytest.approx((ref == 1).sum(), rel=0.02)


def test_monte_carlo_is_statistically_the_whole_images_and_independent_of_n_jobs(tmp_path):
    da = _scene()
    kw = dict(realizations=60, marker_density=1000.0, low=0.3)
    whole = zeit.fields(da, random_state=5, n_jobs=2, **kw)
    other = zeit.fields(da, random_state=6, n_jobs=2, **kw)["labels"].values  # the noise of the Monte Carlo itself
    a = zeit.fields_tiled(da, tmp_path / "a", tile=256, random_state=3, n_jobs=1, **kw)
    b = zeit.fields_tiled(da, tmp_path / "b", tile=256, random_state=3, n_jobs=4, **kw)
    assert a["overlap"] == 80  # 4 x min_distance
    la, lb, ref = (_read(tmp_path / "a" / "labels.tif"), _read(tmp_path / "b" / "labels.tif"),
                   whole["labels"].values)
    assert (la == lb).all()
    assert (_read(tmp_path / "a" / "boundary.tif") == _read(tmp_path / "b" / "boundary.tif")).all()
    # two whole-image runs with other seeds agree in about 0.96 of the pixels at IoU 0.5: the tiles do as well
    floor = min(_matched(ref, other, 0.5), _matched(other, ref, 0.5))
    assert floor > 0.9
    assert _matched(ref, la, 0.5) >= floor - 0.03 and _matched(la, ref, 0.5) >= floor - 0.03
    assert abs(la.max() - ref.max()) <= 0.1 * ref.max()
    c = zeit.fields_tiled(da, tmp_path / "c", tile=256, random_state=4, n_jobs=2, polygons=False, **kw)
    assert c["polygons"] is None and not (tmp_path / "c" / "fields.gpkg").exists()
    assert not (_read(tmp_path / "c" / "boundary.tif") == _read(tmp_path / "a" / "boundary.tif")).all()


def test_a_field_across_a_seam_is_one_polygon(tmp_path):
    from shapely.geometry import Point

    da = _scene(200, 300, seed=4)
    res = zeit.fields_tiled(da, tmp_path, tile=128, overlap=64, n_jobs=2, **EXACT)
    assert res["tiles"] == (2, 3)
    gdf = gpd.read_file(res["polygons"])
    assert list(gdf.columns) == ["field", "n_pixels", "area_ha", "perimeter_m", "compactness", "geometry"]
    assert gdf["field"].is_unique and len(gdf) == res["n_fields"]
    whole = zeit.fields_to_polygons(zeit.fields(da, n_jobs=2, **EXACT))
    assert abs(len(gdf) - len(whole)) <= 2
    lab = _read(res["labels"])
    assert (np.bincount(lab.ravel())[gdf["field"].to_numpy()] == gdf["n_pixels"].to_numpy()).all()

    def centre(row, col):
        return Point(5e5 + 30.0 * (col + 0.5), 8e6 - 30.0 * (row + 0.5))

    # every field whose pixels face each other across a seam (columns 127 | 128 and 255 | 256, rows 127 | 128)
    # is one polygon holding the pixels on both sides
    crossing = 0
    for axis, seams in ((1, (128, 256)), (0, (128,))):
        for s in seams:
            a, b = (lab[:, s - 1], lab[:, s]) if axis == 1 else (lab[s - 1], lab[s])
            for i in np.flatnonzero((a == b) & (a > 1)):
                (r0, c0), (r1, c1) = ((i, s - 1), (i, s)) if axis == 1 else ((s - 1, i), (s, i))
                geom = gdf.loc[gdf["field"] == a[i], "geometry"].iloc[0]
                assert geom.contains(centre(r0, c0)) and geom.contains(centre(r1, c1))
                crossing += 1
    assert crossing > 20
    assert (gdf.geom_type == "Polygon").mean() > 0.9  # the fields are, as in the whole, mostly single pieces


def test_multiband_tophat_mask_and_errors(tmp_path):
    da = _scene(160, 200, seed=2)
    cube = xr.concat([da, (da * 0.5).astype(np.uint16)], dim="band").assign_coords(band=["a", "b"])
    kw = dict(method="exact", marker_density=3000.0, low=0.5, high=0.6, n_jobs=2)
    res = zeit.fields_tiled(cube, tmp_path / "m", tile=100, overlap=60, polygons=False, **kw)  # multitemporal
    assert res["tiles"] == (2, 2) and res["n_fields"] > 5
    r2 = zeit.fields_tiled(cube, tmp_path / "s", tile=100, overlap=60, band="a", gradient="tophat", mask_threshold=3000,
                           polygons=False, **kw)
    assert (_read(r2["labels"]) == 0).any()
    with pytest.raises(ValueError, match="marker_density"):
        zeit.fields_tiled(da, tmp_path / "e", markers=100)
    with pytest.raises(ValueError, match="mask_quantile"):
        zeit.fields_tiled(da, tmp_path / "e", mask_quantile=10)
    with pytest.raises(TypeError, match="unexpected"):
        zeit.fields_tiled(da, tmp_path / "e", bogus=1)
    with pytest.raises(ValueError, match="overlap"):
        zeit.fields_tiled(da, tmp_path / "e", overlap=0)


def test_cli_tile(monkeypatch, tmp_path, capsys):
    da = _scene(260, 300, seed=3)
    zeit.save_raster(da, tmp_path / "img.tif")
    argv = ["zeit", "fields", str(tmp_path / "img.tif"), str(tmp_path / "out"), "--tile", "128", "--overlap", "50",
            "--method", "exact", "--marker-density", "3000", "--low", "0.5", "--polygons", "--jobs", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    assert "3 x 3 tiles" in capsys.readouterr().out
    expected = zeit.fields_tiled(da, tmp_path / "ref", tile=128, overlap=50, method="exact", marker_density=3000.0,
                                 low=0.5, n_jobs=2)
    assert (_read(tmp_path / "out" / "labels.tif") == _read(expected["labels"])).all()
    out = gpd.read_file(tmp_path / "out" / "fields.gpkg")
    assert len(out) == expected["n_fields"] and out.crs == "EPSG:32720"
    monkeypatch.setattr(sys, "argv", argv + ["--markers", "100"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "marker_density" in capsys.readouterr().out
