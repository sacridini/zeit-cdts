"""Phase 9b: tmask, snic, train_classifier/classify, the spatial filters, the water mask and
regularize_time_series on georeferenced cubes and results."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit

TR = from_origin(500000.0, 9000000.0, 30.0, 30.0)


def grid(values, dims, **coords):
    h, w = values.shape[-2:]
    coords.update(y=TR.f - 30.0 * (np.arange(h) + 0.5), x=TR.c + 30.0 * (np.arange(w) + 0.5))
    da = xr.DataArray(values, dims=dims, coords=coords)
    return da.rio.write_crs("EPSG:32722").rio.write_transform(TR)


# --- tmask --------------------------------------------------------------------------


def landsat(seed=0, t=40, h=4, w=5):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=t, freq="16D")
    doy = dates.dayofyear.to_numpy()
    green = 800 + 150 * np.sin(2 * np.pi * doy / 365.25)
    swir = 1500 + 200 * np.cos(2 * np.pi * doy / 365.25)
    v = np.empty((t, 2, h, w))
    v[:, 0] = green[:, None, None] + rng.normal(0, 20, (t, h, w))
    v[:, 1] = swir[:, None, None] + rng.normal(0, 20, (t, h, w))
    v[5, 0] += 3000   # a cloud on every pixel
    v[12, 1] -= 900   # a shadow
    v[20, :, 0, 0] = np.nan   # a missing date on one pixel
    v[:, :, 3, 4] = 0         # a pixel with no data
    return grid(v.astype(np.float32), ("time", "band", "y", "x"), time=dates, band=["green", "swir1"])


def test_tmask_is_the_engine_on_the_cube():
    from zeit._tmask import run_tmask_pixel

    cube = landsat()
    clear = zeit.tmask(cube)
    assert clear.dims == ("time", "y", "x") and clear.dtype == bool and clear.rio.crs == cube.rio.crs
    assert not clear.values[5].any() and not clear.values[12, :3].any()
    ordinals = np.array([d.toordinal() for d in pd.DatetimeIndex(cube.time.values)], dtype=float)
    g, s = cube.values[:, 0, 1, 2].astype(float), cube.values[:, 1, 1, 2].astype(float)
    np.testing.assert_array_equal(clear.values[:, 1, 2], run_tmask_pixel(ordinals, g, s, 10000.0))
    assert not clear.values[20, 0, 0] and not clear.values[:, 3, 4].any()   # no observation: not clear
    lazy = zeit.tmask(cube.chunk({"y": 2, "x": 2}))
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, clear.values)
    with pytest.raises(ValueError, match="bands"):
        zeit.tmask(cube, swir="swir2")


# --- snic ---------------------------------------------------------------------------


def test_snic_is_georeferenced_and_polygonised(tmp_path):
    from zeit.segmentation import run_snic

    rng = np.random.default_rng(3)
    v = rng.random((3, 2, 24, 30)).astype(np.float32)
    v[:, :, :12] += 1.0
    cube = grid(v, ("time", "band", "y", "x"), time=pd.date_range("2022-01-01", periods=3, freq="MS"),
                band=["red", "nir"])
    seg = zeit.snic(cube, spacing=6, compactness=0.3)
    ref = run_snic(v, spacing=6, compactness=0.3)
    np.testing.assert_array_equal(seg.labels.values, ref.labels)
    np.testing.assert_allclose(seg.means.values, ref.means)
    assert seg.means.dims == ("segment", "time", "band") and seg.rio.crs == cube.rio.crs
    k = int(np.flatnonzero(ref.sizes > 0)[0])
    r, c = ref.centroids[k]
    assert seg.centroid_x.values[k] == pytest.approx(TR.c + 30.0 * (c + 0.5))
    assert seg.centroid_y.values[k] == pytest.approx(TR.f - 30.0 * (r + 0.5))
    gdf = zeit.snic_to_polygons(seg, include_means=True)
    assert gdf.crs == cube.rio.crs and len(gdf) == int((ref.sizes > 0).sum())
    assert "2022-01-01_red" in gdf.columns and "2022-03-01_nir" in gdf.columns
    np.testing.assert_allclose(gdf.geometry.area.values, 900.0 * ref.sizes[gdf.supercells.values])
    assert sorted(p.name for p in zeit.save_raster(seg, tmp_path / "snic").iterdir()) == ["labels.tif"]


# --- classification -----------------------------------------------------------------


def two_classes(seed=1):
    """(time, band, y, x): forest on the left (high NIR), crops on the right."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.1, 0.02, (4, 2, 10, 12))
    v[:, 1, :, :6] += 0.4
    v[:, 1, :, 6:] += np.array([0.1, 0.5, 0.6, 0.1])[:, None, None]
    v[:, :, 9, 11] = np.nan
    return grid(v, ("time", "band", "y", "x"), time=pd.date_range("2022-01-01", periods=4, freq="3MS"),
                band=["red", "nir"])


def samples(cube, crs="EPSG:4326"):
    import geopandas as gpd

    cells = [(1, 1, "forest"), (4, 3, "forest"), (8, 2, "forest"), (2, 8, "crop"), (6, 10, "crop"),
             (9, 7, "crop"), (9, 11, "crop")]   # the last one has no data
    x = [float(cube.x.values[c]) for _, c, _ in cells]
    y = [float(cube.y.values[r]) for r, _, _ in cells]
    gdf = gpd.GeoDataFrame({"class": [k for _, _, k in cells]}, geometry=gpd.points_from_xy(x, y), crs=cube.rio.crs)
    return gdf.to_crs(crs)


def test_train_and_classify_a_cube():
    cube = two_classes()
    model = zeit.train_classifier(cube, samples(cube))
    assert model.zeit_features_[:2] == ["2022-01-01_red", "2022-01-01_nir"] and len(model.zeit_features_) == 8
    result = zeit.classify(cube, model, probability=True)
    names = list(result["class"].values)
    expected = np.where(np.arange(12) < 6, names.index("forest") + 1, names.index("crop") + 1)
    expected = np.broadcast_to(expected, (10, 12)).copy()
    expected[9, 11] = 0
    np.testing.assert_array_equal(result.label.values, expected)
    assert result.label.attrs["flag_meanings"] == " ".join(names) and result.rio.crs == cube.rio.crs
    assert result.probability.dims == ("class", "y", "x")
    np.testing.assert_allclose(result.probability.sum("class").values[:9], 1.0, rtol=1e-5)
    lazy = zeit.classify(cube.chunk({"y": 5, "x": 6}), model)
    np.testing.assert_array_equal(lazy.label.values, result.label.values)
    with pytest.raises(ValueError, match="lacks features"):
        zeit.classify(cube.isel(time=[0, 1]), model)


def test_classify_a_dataset_of_maps():
    cube = two_classes()
    maps = xr.Dataset({"nir_mean": cube.sel(band="nir", drop=True).mean("time"),
                       "red_mean": cube.sel(band="red", drop=True).mean("time")})
    maps = maps.rio.write_crs(cube.rio.crs)
    model = zeit.train_classifier(maps, samples(cube, crs=cube.rio.crs))
    assert model.zeit_features_ == ["nir_mean", "red_mean"]
    assert (zeit.classify(maps, model).label.values[:, :6] == list(model.classes_).index("forest") + 1).all()


def ccdc_segments():
    """A CCDC result of 2 x 3 pixels and two bands: one segment each, a known model."""
    from zeit._ccdc_api import COEFS

    coefs = np.zeros((2, 2, len(COEFS), 2, 3), dtype=np.float32)
    coefs[0, :, COEFS.index("c1")] = 0.01                 # a slow rise, per day
    coefs[0, 0, COEFS.index("a0")] = 900.0 - 0.01 * 738000   # green ~900 around 2020
    coefs[0, 1, COEFS.index("a0")] = 300.0 - 0.01 * 738000   # swir ~300: water
    coefs[0, 1, COEFS.index("a0"), 1] = 1800.0 - 0.01 * 738000   # the second row: land
    t_start = np.full((2, 2, 3), np.datetime64("NaT", "ns"))
    t_end = t_start.copy()
    t_start[0] = np.datetime64("2019-01-01", "ns")
    t_end[0] = np.datetime64("2021-01-01", "ns")
    ds = xr.Dataset(
        {"coefs": (("segment", "band", "coef", "y", "x"), coefs),
         "rmse": (("segment", "band", "y", "x"), np.full((2, 2, 2, 3), 12.0, np.float32)),
         "t_start": (("segment", "y", "x"), t_start), "t_end": (("segment", "y", "x"), t_end),
         "n_segments": (("y", "x"), np.ones((2, 3), np.uint8))},
        coords={"segment": [1, 2], "band": ["green", "swir1"], "coef": COEFS,
                "y": TR.f - 30.0 * (np.arange(2) + 0.5), "x": TR.c + 30.0 * (np.arange(3) + 0.5)})
    return ds.rio.write_crs("EPSG:32722")


def test_ccdc_features_and_water():
    from zeit._ccdc_api import _DATENUM_OFFSET
    from zeit._classify_api import features

    seg = ccdc_segments()
    f = features(seg, date="2020-01-01")
    assert list(f.feature.values[:2]) == ["green_a0", "green_c1"] and f.feature.values[-1] == "swir1_rmse"
    t = pd.Timestamp("2020-01-01").toordinal() + _DATENUM_OFFSET
    expected = 900.0 - 0.01 * 738000 + 0.01 * t   # the intercept on the date
    np.testing.assert_allclose(f.sel(feature="green_a0").values, expected, rtol=1e-5)
    with pytest.raises(ValueError, match="date="):
        features(seg)
    water = zeit.extract_water_mask(seg)
    np.testing.assert_array_equal(water.values, [[1, 1, 1], [0, 0, 0]])
    assert water.dtype == np.uint8 and water.rio.crs == seg.rio.crs


# --- spatial filters ----------------------------------------------------------------


def test_filters_keep_the_map():
    from zeit.spatial import apply_bayesian_filter

    v = np.zeros((10, 10), dtype=np.int16)
    v[5, 5] = 2005
    v[6:10, 0:3] = 2010
    yod = grid(v, ("y", "x")).rio.write_nodata(0, encoded=False)
    out = zeit.apply_mmu_filter(yod, mmu_pixels=11)
    assert isinstance(out, xr.DataArray) and out.rio.nodata == 0 and out.rio.crs == yod.rio.crs
    assert out.values[5, 5] == 0 and (out.values[6:10, 0:3] == 2010).all()
    np.testing.assert_array_equal(zeit.apply_mmu_filter(v, mmu_pixels=11), out.values)   # numpy in, numpy out
    classes = grid(np.where(np.arange(100).reshape(10, 10) % 10 < 5, 1, 2).astype(np.uint8), ("y", "x"))
    classes[3, 3] = 2
    smooth = zeit.apply_majority_filter(classes)
    assert smooth.values[3, 3] == 1 and (smooth.x.values == classes.x.values).all()
    probs = grid(np.stack([np.full((10, 10), 0.6), np.full((10, 10), 0.4)]), ("class", "y", "x"),
                 **{"class": ["forest", "crop"]})
    winner = apply_bayesian_filter(probs)
    assert winner.dims == ("y", "x") and (winner.values == "forest").all()


# --- regularize ---------------------------------------------------------------------


def test_regularize_one_band_and_files(tmp_path):
    rng = np.random.default_rng(2)
    dates = pd.date_range("2022-01-01", periods=30, freq="5D")
    cube = grid(rng.random((30, 3, 4)).astype(np.float32), ("time", "y", "x"), time=dates)
    out = zeit.regularize_time_series(cube, freq="16D", method="medoid")
    assert out.dims == ("time", "y", "x") and out.rio.crs == cube.rio.crs
    first = cube.isel(time=slice(0, 4))   # the observations of the first 16 days
    exact = first.astype(np.float64)      # four dates: the median is between two, a tie the first date wins
    pick = np.abs(exact - exact.median("time")).argmin("time")
    np.testing.assert_array_equal(out.isel(time=0).values, first.isel(time=pick).values)
    path = zeit.save_raster(cube, tmp_path / "cube.tif")
    from_file = zeit.regularize_time_series(path, freq="16D")
    np.testing.assert_allclose(from_file.values, zeit.regularize_time_series(cube, freq="16D").values)
