"""The zeit command line: each subcommand of the cube functions writes what the Python
function returns (read from a GeoTIFF, lazily, and written in one pass)."""

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit import cli

# joblib (the saved classifier) sets array shapes, which NumPy 2.5 deprecates
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:joblib")

TR = from_origin(500000.0, 9000000.0, 30.0, 30.0)
H, W = 12, 15


def grid(values, dims, **coords):
    h, w = values.shape[-2:]
    coords.update(y=TR.f - 30.0 * (np.arange(h) + 0.5), x=TR.c + 30.0 * (np.arange(w) + 0.5))
    return xr.DataArray(values, dims=dims, coords=coords, name="ndvi").rio.write_crs("EPSG:32722")


def ndvi_series(seed=0):
    """Three years of monthly NDVI: a crop cycle, forest and bare soil in strips."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=36, freq="MS") + pd.DateOffset(days=14)
    steps = np.arange(36)
    kinds = [0.45 + 0.35 * np.sin(2 * np.pi * steps / 12), np.full(36, 0.85), np.full(36, 0.15)]
    truth = np.zeros((H, W), dtype=int)
    truth[:, 5:10] = 1
    truth[:, 10:] = 2
    v = np.stack([kinds[k] for k in truth.ravel()], axis=1).reshape(36, H, W) + rng.normal(0, 0.02, (36, H, W))
    return grid(v.astype(np.float32), ("time", "y", "x"), time=dates), truth


def run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["zeit", *map(str, argv)])
    cli.main()


def same_as(tmp_path, written, result):
    """The raster the command wrote holds what save_raster writes of ``result``."""
    expected = tmp_path / "expected.tif"
    if isinstance(result, xr.Dataset):
        result = result[[v for v in result.data_vars if {"y", "x"} <= set(result[v].dims)]]
    elif result.dtype == bool:
        result = result.astype("uint8")
    zeit.save_raster(result, expected)
    a = zeit.load_raster(written, masked=True)
    b = zeit.load_raster(expected, masked=True)
    assert a.shape == b.shape
    np.testing.assert_allclose(a.values, b.values, rtol=1e-6, equal_nan=True)
    return a


@pytest.fixture
def series_tif(tmp_path):
    da, truth = ndvi_series()
    path = tmp_path / "ndvi.tif"
    zeit.save_raster(da, path)
    return path, da, truth


def test_phenology(monkeypatch, tmp_path, series_tif):
    path, da, _ = series_tif
    run(monkeypatch, "phenology", path, tmp_path / "out", "--curve", "elmore")
    same_as(tmp_path, tmp_path / "out" / "phenology.tif", zeit.phenology(da, curve="elmore"))


def test_smooth_with_weights(monkeypatch, tmp_path, series_tif):
    path, da, _ = series_tif
    weights = xr.ones_like(da).where(da.time.dt.month != 6, 0.0)
    zeit.save_raster(weights, tmp_path / "weights.tif")
    run(monkeypatch, "smooth", path, tmp_path / "out", "--lmbda", 5, "--weights", tmp_path / "weights.tif",
        "--prefix", "wh")
    out = same_as(tmp_path, tmp_path / "out" / "wh.tif", zeit.smooth(da, lmbda=5, weights=weights))
    assert (out.time.values == da.time.values).all()


def test_tmask(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=40, freq="16D")
    doy = dates.dayofyear.to_numpy()
    v = np.empty((40, 2, 4, 5), dtype=np.float32)
    v[:, 0] = (800 + 150 * np.sin(2 * np.pi * doy / 365.25))[:, None, None] + rng.normal(0, 20, (40, 4, 5))
    v[:, 1] = (1500 + 200 * np.cos(2 * np.pi * doy / 365.25))[:, None, None] + rng.normal(0, 20, (40, 4, 5))
    v[5, 0] += 3000
    cube = grid(v, ("time", "band", "y", "x"), time=dates, band=["green", "swir1"])
    zeit.save_raster(cube, tmp_path / "landsat.tif")
    run(monkeypatch, "tmask", tmp_path / "landsat.tif", tmp_path / "out")
    out = same_as(tmp_path, tmp_path / "out" / "tmask.tif", zeit.tmask(cube))
    assert (out.values[5] == 0).all() and out.values.mean() > 0.9


def test_twdtw_from_a_csv_of_patterns(monkeypatch, tmp_path, series_tif):
    path, da, truth = series_tif
    first = da.isel(time=slice(0, 12))
    rows = []
    for name, col in (("crop", 2), ("forest", 7), ("bare", 12)):
        for when, value in zip(first.time.values, first.isel(y=0, x=col).values):
            rows.append({"pattern": name, "date": str(when)[:10], "ndvi": float(value)})
    pd.DataFrame(rows).to_csv(tmp_path / "patterns.csv", index=False)
    run(monkeypatch, "twdtw", path, tmp_path / "out", "--patterns", tmp_path / "patterns.csv")
    patterns = cli.read_patterns(tmp_path / "patterns.csv")
    assert list(patterns) == ["crop", "forest", "bare"] and len(patterns["crop"]) == 12
    out = same_as(tmp_path, tmp_path / "out" / "twdtw.tif", zeit.twdtw(da, patterns))
    assert (out.values[0] == truth + 1).all()   # the label band


def test_snic_with_polygons(monkeypatch, tmp_path, series_tif):
    gpd = pytest.importorskip("geopandas")
    path, da, _ = series_tif
    run(monkeypatch, "snic", path, tmp_path / "out", "--spacing", 4, "--polygons")
    same_as(tmp_path, tmp_path / "out" / "snic.tif", zeit.snic(da, spacing=4))
    polygons = gpd.read_file(tmp_path / "out" / "snic.gpkg")
    assert len(polygons) > 1 and polygons.crs == "EPSG:32722"


def test_classify_trains_saves_and_reuses_a_model(monkeypatch, tmp_path, series_tif):
    gpd = pytest.importorskip("geopandas")
    pytest.importorskip("joblib")
    path, da, truth = series_tif
    cells = [(r, c) for r in (1, 6, 10) for c in (1, 3, 6, 8, 11, 13)]
    names = np.array(["crop", "forest", "bare"])
    points = gpd.GeoDataFrame({"class": [names[truth[r, c]] for r, c in cells]},
                              geometry=gpd.points_from_xy([float(da.x[c]) for _, c in cells],
                                                          [float(da.y[r]) for r, _ in cells]), crs="EPSG:32722")
    points.to_file(tmp_path / "samples.gpkg")
    model = tmp_path / "rf.joblib"
    run(monkeypatch, "classify", path, tmp_path / "out", "--samples", tmp_path / "samples.gpkg", "--model", model)
    assert model.exists()
    rf = zeit.train_classifier(da, points)
    out = same_as(tmp_path, tmp_path / "out" / "classes.tif", zeit.classify(da, rf))
    run(monkeypatch, "classify", path, tmp_path / "again", "--model", model, "--probability")
    again = zeit.load_raster(tmp_path / "again" / "classes.tif")
    assert again.shape[0] == 4 and (again.values[0] == out.values[0]).all()   # label + 3 probabilities
    with pytest.raises(SystemExit):
        run(monkeypatch, "classify", path, tmp_path / "none")


def test_som_writes_the_map_and_the_prototypes(monkeypatch, tmp_path, series_tif):
    path, da, truth = series_tif
    run(monkeypatch, "som", path, tmp_path / "out", "--x", 3, "--y", 1, "--sample", 0)
    result = zeit.som(da, x=3, y=1, sample=None)
    same_as(tmp_path, tmp_path / "out" / "som.tif", result)
    table = pd.read_csv(tmp_path / "out" / "som_prototypes.csv")
    assert list(table.columns[:4]) == ["neuron", "i", "j", "n_pixels"] and len(table) == 3
    assert table.columns[4] == "2019-01-15" and table.shape[1] == 4 + 36
    np.testing.assert_allclose(table.iloc[:, 4:].values, result.prototypes.values)


def test_a_failure_exits_with_an_error(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, "smooth", tmp_path / "missing.tif", tmp_path / "out")
    assert "Error running smooth" in capsys.readouterr().out
