"""zeit.fuse_starfm, zeit.fuse_estarfm, zeit.fusion_plan: STARFM and ESTARFM spatiotemporal fusion."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _fusion

N = 64
RES = 30.0
X0, Y0 = 500000.0, 4000000.0
CLASS = np.array([[0.10, 0.18, 0.25], [0.03, 0.02, 0.01], [0.06, 0.05, 0.35], [0.08, 0.07, 0.28]])
FINE_BANDS = ["green", "red", "nir08"]
COARSE_BANDS = ["sur_refl_b04", "sur_refl_b01", "sur_refl_b02"]


def _classes(n=N):
    yy, xx = np.mgrid[0:n, 0:n] / (n / 96.0)
    c = np.zeros((n, n), dtype=int)
    c[(xx - 30) ** 2 + (yy - 30) ** 2 < 14 ** 2] = 1
    c[(xx >= 50) & (xx < 90) & (yy >= 10) & (yy < 60) & (c == 0)] = 2
    c[((xx - 30) ** 2 + (yy - 70) ** 2 < 18 ** 2) & (c == 0)] = 3
    return c


def _scene(shift=0.0):
    img = CLASS[_classes()].transpose(2, 0, 1) + shift
    return img.astype(np.float32)


def _block(img, k=8):
    b, h, w = img.shape
    return img.reshape(b, h // k, k, w // k, k).mean(axis=(2, 4)).astype(np.float32)


def _fine(images, days, crs="EPSG:32633"):
    n = images[0].shape[-1]
    cube = xr.DataArray(np.stack(images), dims=("time", "band", "y", "x"), name="reflectance",
                        coords={"time": pd.to_datetime(days), "band": FINE_BANDS,
                                "y": Y0 - RES * (np.arange(n) + 0.5), "x": X0 + RES * (np.arange(n) + 0.5)})
    return cube.rio.write_crs(crs)


def _coarse(images, days, k=8, crs="EPSG:32633", dx=0.0):
    n = images[0].shape[-1]
    cube = xr.DataArray(np.stack(images), dims=("time", "band", "y", "x"), name="modis",
                        coords={"time": pd.to_datetime(days), "band": COARSE_BANDS,
                                "y": Y0 - RES * k * (np.arange(n) + 0.5), "x": X0 + dx + RES * k * (np.arange(n) + 0.5)})
    return cube.rio.write_crs(crs)


def _homogeneous():
    """Fine on three dates, +0.04 / +0.08 over the first; daily coarse for 3 dates."""
    f = [_scene(0.0), _scene(0.08)]
    c = [_block(_scene(s)) for s in (0.0, 0.04, 0.08)]
    return (_fine(f, ["2020-01-01", "2020-01-03"]), _coarse(c, ["2020-01-01", "2020-01-02", "2020-01-03"]),
            _scene(0.04))


def test_homogeneous_change_is_exact_for_both_methods():
    fine, coarse, truth = _homogeneous()
    for fn, kw in ((zeit.fuse_starfm, dict(radius=240.0)), (zeit.fuse_estarfm, dict(radius=240.0)),
                   (zeit.fuse_estarfm, dict(radius=240.0, v_scope="cell"))):
        out = fn(fine, coarse, include_fine_dates=False, dates=["2020-01-02"], n_jobs=2, **kw)
        assert out.dims == ("time", "band", "y", "x") and out.dtype == np.float32
        assert list(out.band.values) == ["green", "red", "nir"]
        assert float(np.abs(out.values[0] - truth).max()) < 2e-5, fn.__name__


def test_coords_attrs_quality_and_the_plan():
    fine, coarse, truth = _homogeneous()
    pred, q = zeit.fuse_starfm(fine, coarse, radius=240.0, quality=True)
    assert list(pred.fusion_kind.values) == ["observed", "predicted", "observed"]
    assert list(pred.fusion_pairs.values)[1] == "2020-01-01,2020-01-03"
    assert pred.attrs["fusion"]["method"] == "starfm" and "Gao" in pred.attrs["fusion"]["citation"]
    assert not pred.fusion_extrapolated.values.any()
    assert list(q.band.values) == ["n_similar", "source"] and q.dtype == np.uint16
    assert (q.sel(band="source").values[[0, 2]] == 0).all()
    assert (q.sel(band="source").values[1] == 1).all()
    plan = zeit.fusion_plan(fine, coarse)
    assert list(plan.columns) == ["date", "kind", "coarse_date", "pairs", "extrapolated"]
    assert list(plan.kind) == ["observed", "predicted", "observed"]
    # the observed dates use the other date's pair only
    assert plan.pairs[0] == "2020-01-03" and plan.pairs[1] == "2020-01-01,2020-01-03"
    with pytest.raises(ValueError, match="no date can be predicted"):
        zeit.fusion_plan(fine, coarse, method="estarfm", exclude_dates=["2020-01-03"], include_fine_dates=False,
                         dates=["2020-01-02"])


def test_dask_and_strips_equal_in_memory(monkeypatch):
    pytest.importorskip("dask")
    fine, coarse, _ = _homogeneous()
    # a texture, so that the strips matter
    rng = np.random.default_rng(0)
    noisy = fine + xr.DataArray(rng.normal(0, 0.01, fine.shape).astype(np.float32), dims=fine.dims, coords=fine.coords)
    noisy = noisy.rio.write_crs("EPSG:32633")
    for fn in (zeit.fuse_starfm, zeit.fuse_estarfm):
        a = fn(noisy, coarse, radius=150.0, classes=6.0)
        monkeypatch.setattr(_fusion, "_STRIP_ROWS", 20)
        b = fn(noisy.chunk({"y": 16, "x": 32}), coarse.chunk({"time": 1}), radius=150.0, classes=6.0)
        monkeypatch.setattr(_fusion, "_STRIP_ROWS", None)
        assert np.array_equal(a.values, b.values, equal_nan=True), fn.__name__


def test_another_crs_with_the_same_sampling_equals_the_affine_case():
    fine, coarse, _ = _homogeneous()
    shifted = coarse.copy()
    shifted = shifted.assign_coords(x=shifted.x + 1000.0).rio.write_crs(
        "+proj=tmerc +lat_0=0 +lon_0=15 +k=0.9996 +x_0=501000 +y_0=0 +datum=WGS84 +units=m +no_defs")
    # a point (X, Y) of the fine CRS is (X + 1000, Y) in the other one: the grids are the same
    rng = np.random.default_rng(1)
    noisy = (fine + xr.DataArray(rng.normal(0, 0.01, fine.shape).astype(np.float32), dims=fine.dims,
                                 coords=fine.coords)).rio.write_crs("EPSG:32633")
    a = zeit.fuse_starfm(noisy, coarse, radius=150.0, classes=6.0)
    b = zeit.fuse_starfm(noisy, shifted, radius=150.0, classes=6.0)
    assert float(np.nanmax(np.abs(a.values - b.values))) < 1e-4


def test_refusals():
    fine, coarse, _ = _homogeneous()
    with pytest.raises(ValueError, match="no CRS"):
        zeit.fuse_starfm(xr.DataArray(fine.values, dims=fine.dims,
                                      coords={k: v for k, v in fine.coords.items() if k != "spatial_ref"}), coarse)
    ints = (fine * 10000).astype(np.int16)
    ints = ints.rio.write_crs("EPSG:32633")
    with pytest.raises(ValueError, match="digital numbers"):
        zeit.fuse_starfm(ints, coarse)
    no_roles = fine.assign_coords(band=["a", "b", "c"]).rio.write_crs("EPSG:32633")
    with pytest.raises(ValueError, match="counterpart|no band"):
        zeit.fuse_starfm(no_roles, coarse)
    one = fine.isel(time=[0]).rio.write_crs("EPSG:32633")
    with pytest.raises(ValueError, match="no date can be predicted"):
        zeit.fuse_estarfm(one, coarse)
    with pytest.raises(ValueError, match="weighting"):
        zeit.fuse_starfm(fine, coarse, weighting="x")


def test_cloud_hole_single_pair_and_fallback():
    f0, f1 = _scene(0.0), _scene(0.08)
    f0 = f0.copy()
    f0[:, 20:30, 20:30] = np.nan
    fine = _fine([f0, f1], ["2020-01-01", "2020-01-03"])
    c = [_block(_scene(s)) for s in (0.0, 0.04, 0.08)]
    coarse = _coarse(c, ["2020-01-01", "2020-01-02", "2020-01-03"])
    pred, q = zeit.fuse_estarfm(fine, coarse, radius=240.0, include_fine_dates=False, dates=["2020-01-02"],
                                quality=True)
    assert q.sel(band="source").values[0, 25, 25] == 2
    assert abs(pred.values[0, 0, 25, 25] - _scene(0.04)[0, 25, 25]) < 2e-5
    one = zeit.fuse_starfm(fine, coarse, radius=240.0, pairs="nearest", include_fine_dates=False,
                           dates=["2020-01-02"], exclude_dates=["2020-01-03"], fallback="coarse", quality=True)
    out, q = one
    assert q.sel(band="source").values[0, 25, 25] == 3 and np.isfinite(out.values[0, :, 25, 25]).all()
