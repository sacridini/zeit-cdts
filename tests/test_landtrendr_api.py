"""zeit.landtrendr: one function for every input, and extract_events on its result."""
import os
import subprocess
import sys

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit.metrics import extract_events
from zeit.raster import run_landtrendr_array

YEARS = np.arange(2000, 2020)
TRANSFORM = from_origin(-63.0, -10.0, 0.001, 0.001)


def _series(drop_year=2008, start=8000.0, drop=3000.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    values = np.full(len(YEARS), start) + rng.normal(0, noise, len(YEARS))
    values[YEARS >= drop_year] -= drop
    return values


def _stack(rows=6, cols=7, seed=1):
    """(time, y, x) NDVI x 10000: half the pixels lose 3000 in a pixel-dependent year."""
    rng = np.random.default_rng(seed)
    stack = np.empty((len(YEARS), rows, cols), dtype=np.int16)
    for r in range(rows):
        for c in range(cols):
            drop = 2004 + (r * cols + c) % 12
            stack[:, r, c] = _series(drop_year=drop, drop=3000 if (r + c) % 2 else 0, noise=60,
                                     seed=int(rng.integers(1e6)))
    return stack


@pytest.fixture
def cube_path(tmp_path):
    stack = _stack()
    path = tmp_path / "ndvi.tif"
    with rasterio.open(path, "w", driver="GTiff", height=stack.shape[1], width=stack.shape[2],
                       count=stack.shape[0], dtype="int16", crs="EPSG:4326", transform=TRANSFORM) as dst:
        dst.write(stack)
        for i, y in enumerate(YEARS, start=1):
            dst.set_band_description(i, f"yr{y}")
    return str(path), stack


def test_exported_and_old_entry_points_gone():
    assert callable(zeit.landtrendr)
    for old in ("run_landtrendr", "run_landtrendr_array", "run_landtrendr_image"):
        assert not hasattr(zeit, old)
    assert "landtrendr" in zeit.__all__


def test_cube_result_structure(cube_path):
    path, stack = cube_path
    cube = zeit.load_raster(path)
    lt = zeit.landtrendr(cube, max_segments=4)
    assert isinstance(lt, xr.Dataset)
    assert lt.vertex_year.dims == ("vertex", "y", "x") and lt.sizes["vertex"] == 5
    assert lt.vertex_year.dtype == np.int16 and lt.n_vertices.dtype == np.uint8
    assert lt.rio.crs.to_epsg() == 4326
    np.testing.assert_allclose(lt.x.values, cube.x.values)
    assert lt.attrs["direction"] == "loss" and lt.attrs["max_segments"] == 4
    assert lt.attrs["first_year"] == 2000 and lt.attrs["last_year"] == 2019
    # every pixel starts at the first year and ends at the last one
    n = lt.n_vertices.values
    assert (lt.vertex_year.values[0] == 2000).all()
    last = np.take_along_axis(lt.vertex_year.values, (n - 1)[None].astype(int), axis=0)[0]
    assert (last == 2019).all()
    # past the last vertex: year 0, value NaN
    beyond = np.arange(5)[:, None, None] >= n[None]
    assert (lt.vertex_year.values[beyond] == 0).all() and np.isnan(lt.vertex_value.values[beyond]).all()


def test_same_result_as_the_numpy_engine(cube_path):
    path, stack = cube_path
    lt = zeit.landtrendr(path)
    vertices, rmse = run_landtrendr_array(YEARS, stack.astype(np.float32), modifier=-1.0, return_rmse=True)
    mv = lt.sizes["vertex"]
    np.testing.assert_array_equal(lt.vertex_year.values, vertices[:mv])
    np.testing.assert_allclose(np.nan_to_num(lt.vertex_value.values), vertices[mv:], rtol=1e-6)
    np.testing.assert_allclose(lt.rmse.values, rmse, rtol=1e-6)

    new = zeit.extract_events(lt)
    old = extract_events(vertices, event_type="loss", rmse_map=rmse)
    found = old["yod"] > 0
    for key in old:
        np.testing.assert_allclose(new[key].values[found], old[key][found], rtol=1e-6)
        if new[key].dtype.kind == "f":
            assert np.isnan(new[key].values[~found]).all()     # no event: NaN
        else:
            assert (new[key].values[~found] == 0).all() and new[key].rio.nodata == 0


def test_every_input_gives_the_same_vertices(cube_path):
    path, stack = cube_path
    reference = zeit.landtrendr(path).vertex_year.values
    cube = zeit.load_raster(path)
    assert np.array_equal(zeit.landtrendr(cube).vertex_year.values, reference)
    assert np.array_equal(zeit.landtrendr(stack, years=YEARS).vertex_year.values, reference)
    assert np.array_equal(cube.zeit.landtrendr().vertex_year.values, reference)
    lazy = zeit.landtrendr(path, chunks={"time": -1, "y": 3, "x": 4})
    assert isinstance(lazy.vertex_year.data, da.Array)
    assert np.array_equal(lazy.vertex_year.values, reference)
    # a time-chunked dask cube is rechunked to whole series
    split = cube.chunk({"time": 5, "y": 2, "x": 3})
    assert np.array_equal(zeit.landtrendr(split).vertex_year.values, reference)
    # the band axis named band (no dates) + years
    plain = zeit.load_raster(stack)
    assert plain.dims == ("band", "y", "x")
    assert np.array_equal(zeit.landtrendr(plain, years=YEARS).vertex_year.values, reference)


def test_band_selection_in_4d_cube_and_dataset(cube_path):
    path, stack = cube_path
    cube = zeit.load_raster(path)
    four = xr.concat([cube, cube * 0 + 5000], dim=pd.Index(["ndvi", "flat"], name="band")).transpose(
        "time", "band", "y", "x")
    with pytest.raises(ValueError, match="band="):
        zeit.landtrendr(four)
    reference = zeit.landtrendr(cube).vertex_year.values
    assert np.array_equal(zeit.landtrendr(four, band="ndvi").vertex_year.values, reference)
    ds = xr.Dataset({"ndvi": cube, "nbr": cube})
    with pytest.raises(ValueError, match="band="):
        zeit.landtrendr(ds)
    assert np.array_equal(zeit.landtrendr(ds, band="nbr").vertex_year.values, reference)
    assert np.array_equal(zeit.landtrendr(xr.Dataset({"ndvi": cube})).vertex_year.values, reference)


def test_single_pixel_inputs():
    values = _series(drop_year=2008)
    lt = zeit.landtrendr(values, years=YEARS, max_segments=3)
    assert lt.vertex_year.dims == ("vertex",)
    years = lt.vertex_year.values[: int(lt.n_vertices)]
    assert years[0] == 2000 and years[-1] == 2019
    assert 2007 in years
    series = pd.Series(values, index=pd.to_datetime([f"{y}-07-01" for y in YEARS]))
    assert np.array_equal(zeit.landtrendr(series, max_segments=3).vertex_year.values, lt.vertex_year.values)
    by_year = pd.Series(values, index=YEARS)
    assert np.array_equal(zeit.landtrendr(by_year, max_segments=3).vertex_year.values, lt.vertex_year.values)
    assert np.array_equal(zeit.landtrendr(list(values), years=YEARS, max_segments=3).vertex_year.values,
                          lt.vertex_year.values)
    ev = zeit.extract_events(lt)
    assert int(ev.yod) == 2007 and float(ev.magnitude) == pytest.approx(3000, rel=0.05)
    with pytest.raises(ValueError, match="years="):
        zeit.landtrendr(values)


def test_direction(cube_path):
    path, _ = cube_path
    loss = zeit.landtrendr(path)
    gain = zeit.landtrendr(path, direction="gain")
    assert gain.attrs["direction"] == "gain"
    assert zeit.extract_events(gain).attrs["event_type"] == "gain"
    assert zeit.extract_events(loss).attrs["event_type"] == "loss"
    with pytest.raises(ValueError, match="direction"):
        zeit.landtrendr(path, direction="down")


def test_missing_years(tmp_path):
    values = _series(drop_year=2010).astype(np.int16)
    gappy = values.copy()
    gappy[[3, 12]] = 0  # masked years, exported as 0 by Earth Engine
    clean = zeit.landtrendr(np.delete(values, [3, 12]).astype(float), years=np.delete(YEARS, [3, 12]))
    stack = gappy.reshape(-1, 1, 1)
    auto = zeit.landtrendr(stack, years=YEARS)                 # integer data: 0 is missing
    np.testing.assert_array_equal(auto.vertex_year.values[:, 0, 0], clean.vertex_year.values)
    kept = zeit.landtrendr(stack, years=YEARS, nodata=None)    # 0 taken as a value: a much worse fit
    assert float(kept.rmse.values[0, 0]) > 10 * float(auto.rmse.values[0, 0]) + 100

    floats = values.astype(np.float32).reshape(-1, 1, 1).copy()
    floats[[3, 12]] = np.nan
    np.testing.assert_array_equal(zeit.landtrendr(floats, years=YEARS).vertex_year.values[:, 0, 0],
                                  clean.vertex_year.values)
    floats[[3, 12]] = -1
    np.testing.assert_array_equal(zeit.landtrendr(floats, years=YEARS, nodata=-1).vertex_year.values[:, 0, 0],
                                  clean.vertex_year.values)

    # the raster's NoData value
    path = tmp_path / "nd.tif"
    with rasterio.open(path, "w", driver="GTiff", height=1, width=1, count=len(YEARS), dtype="int16",
                       crs="EPSG:4326", transform=TRANSFORM, nodata=-9999) as dst:
        data = values.copy()
        data[[3, 12]] = -9999
        dst.write(data.reshape(-1, 1, 1))
    lt = zeit.landtrendr(zeit.load_raster(str(path), start_year=2000))
    np.testing.assert_array_equal(lt.vertex_year.values[:, 0, 0], clean.vertex_year.values)

    empty = zeit.landtrendr(np.zeros((len(YEARS), 1, 1), dtype=np.int16), years=YEARS)
    assert int(empty.n_vertices.values[0, 0]) == 0 and np.isnan(empty.rmse.values[0, 0])
    assert int(zeit.extract_events(empty).yod.values[0, 0]) == 0


def test_fitted_trajectory():
    values = _series(drop_year=2008, noise=50, seed=3)
    lt = zeit.landtrendr(values, years=YEARS, fitted=True, max_segments=3)
    fit = lt.fitted.values
    assert fit.shape == (len(YEARS),) and lt.fitted.dims == ("time",)
    years = lt.vertex_year.values[: int(lt.n_vertices)]
    vals = lt.vertex_value.values[: int(lt.n_vertices)]
    np.testing.assert_allclose(fit, np.interp(YEARS, years, vals), rtol=1e-5)
    np.testing.assert_allclose(fit[np.isin(YEARS, years)], vals, rtol=1e-5)


def test_errors(cube_path):
    path, stack = cube_path
    with pytest.raises(ValueError, match="years="):
        zeit.landtrendr(stack)
    with pytest.raises(ValueError, match="time steps"):
        zeit.landtrendr(stack, years=YEARS[:5])
    cube = zeit.load_raster(path)
    monthly = cube.assign_coords(time=pd.date_range("2000-01-01", periods=len(YEARS), freq="MS"))
    with pytest.raises(ValueError, match="annual composites"):
        zeit.landtrendr(monthly)


def test_extract_events_dataset(cube_path):
    path, stack = cube_path
    lt = zeit.landtrendr(path)
    ev = zeit.extract_events(lt, min_magnitude=1500)
    assert list(ev.data_vars) == ["yod", "date", "magnitude", "duration", "pre_val", "post_val", "rate", "dsnr"]
    assert ev.yod.dtype == np.uint16 and ev.rio.crs.to_epsg() == 4326
    # date: the first year that shows the change
    has = ev.yod.values > 0
    assert (pd.DatetimeIndex(ev.date.values[has]).year == ev.yod.values[has] + 1).all()
    assert pd.isna(ev.date.values[~has]).all()
    # pixels with a 3000 drop: found in the right year; stable pixels: no event
    rows, cols = stack.shape[1:]
    for r in range(rows):
        for c in range(cols):
            drop = 2004 + (r * cols + c) % 12
            if (r + c) % 2:
                assert int(ev.yod.values[r, c]) in (drop - 1, drop)
            else:
                assert int(ev.yod.values[r, c]) == 0
    lazy = zeit.extract_events(zeit.landtrendr(path, chunks={"time": -1, "y": 3, "x": 3}), min_magnitude=1500)
    assert isinstance(lazy.yod.data, da.Array)
    xr.testing.assert_equal(lazy.compute(), ev)
    with pytest.raises(ValueError, match="sort_by"):
        zeit.extract_events(lt, sort_by="biggest")


def test_flat_segment_is_no_event():
    years = np.array([0, 2000, 2010, 0], dtype=np.float32)  # (2 * max_vertices, 1, 1) numpy stack
    stack = np.array([2000, 2010, 0, 0, 5000, 5000, 0, 0], dtype=np.float32).reshape(-1, 1, 1)
    out = zeit.extract_events(stack)
    assert out["yod"][0, 0] == 0 and "dsnr" not in out


def test_load_run_save_round_trip(cube_path, tmp_path):
    path, _ = cube_path
    events = zeit.extract_events(zeit.landtrendr(zeit.load_raster(path)))
    folder = zeit.save_raster(events, tmp_path / "out")
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"{k}.tif" for k in events.data_vars)
    with rasterio.open(folder / "yod.tif") as src:
        assert src.transform == TRANSFORM and src.dtypes[0] == "uint16"
        np.testing.assert_array_equal(src.read(1), events.yod.values)
    lt = zeit.landtrendr(path, fitted=True)
    folder = zeit.save_raster(lt, tmp_path / "lt")
    fitted = zeit.load_raster(str(folder / "fitted.tif"))
    assert fitted.time.dt.year.values.tolist() == YEARS.tolist()


def test_cli_landtrendr(cube_path, tmp_path):
    path, _ = cube_path
    out = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-m", "zeit.cli", "landtrendr", path, str(out), "--chunk-size", "4",
                             "--save-vertices", "--min-mag", "1500"], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    names = sorted(os.listdir(out))
    assert "lt_event_yod.tif" in names and "lt_vertices.tif" in names
    expected = zeit.extract_events(zeit.landtrendr(path), min_magnitude=1500).yod.values
    with rasterio.open(out / "lt_event_yod.tif") as src:
        np.testing.assert_array_equal(src.read(1), expected)
