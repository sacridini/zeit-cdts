"""zeit.bfast_monitor, bfast_lite, bfast, mann_kendall and phenology: every input, one result."""
import subprocess
import sys

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

import zeit
from zeit._bfast import BFM_METRIC_NAMES, run_bfast_dask, run_bfast_lite_dask, run_bfast_monitor_dask
from zeit._series_api import PHENOLOGY_METRICS, _regular_time
from zeit.trend import MK_METRIC_NAMES, run_mann_kendall_dask

DATES = pd.date_range("2010-01-01", periods=23 * 8, freq="16D")


@pytest.fixture(scope="module")
def cube():
    """16-day NDVI-like series, (time, y=3, x=4): row 0 drops by 0.3 in 2015."""
    rng = np.random.default_rng(0)
    t = np.arange(len(DATES))
    base = 0.5 + 0.2 * np.sin(2 * np.pi * t / 23)
    stack = np.stack([base + rng.normal(0, 0.02, len(t)) for _ in range(12)], -1).reshape(len(t), 3, 4)
    stack[23 * 5:, 0, :] -= 0.3
    return xr.DataArray(stack.astype(np.float32), dims=("time", "y", "x"),
                        coords={"time": DATES, "y": [3.5, 2.5, 1.5], "x": [0.5, 1.5, 2.5, 3.5]}).rio.write_crs(4326)


def _dask(cube):
    return da.from_array(cube.values, chunks=(-1, -1, -1))


def test_exported_and_old_entry_points_gone():
    for name in ("bfast_monitor", "bfast_lite", "bfast", "mann_kendall", "phenology"):
        assert callable(getattr(zeit, name)) and name in zeit.__all__
    for old in ("run_bfast_monitor_image", "run_bfast_lite_image", "run_bfast_image", "run_mann_kendall_image"):
        assert not hasattr(zeit, old)


def test_regular_time_from_dates(cube):
    assert _regular_time(cube, None, None) == (2010.0, 23)
    monthly = cube.isel(time=slice(0, 24)).assign_coords(time=pd.date_range("2001-03-01", periods=24, freq="MS"))
    assert _regular_time(monthly, None, None) == (2001 + 2 / 12, 12)
    annual = cube.isel(time=slice(0, 10)).assign_coords(time=pd.date_range("1990-01-01", periods=10, freq="YS"))
    assert _regular_time(annual, None, None) == (1990.0, 1)
    assert _regular_time(cube, 2000.5, 46) == (2000.5, 46)


def test_bfast_monitor_matches_the_engine(cube):
    result = zeit.bfast_monitor(cube, "2014-01-01")
    assert list(result.data_vars) == BFM_METRIC_NAMES
    assert result.rio.crs.to_epsg() == 4326 and result.breakpoint.dims == ("y", "x")
    assert result.attrs["frequency"] == 23 and result.attrs["monitor_start"] == 2014.0
    ref = run_bfast_monitor_dask(_dask(cube), start_time=2010.0, monitor_start_time=2014.0, frequency=23).compute()
    np.testing.assert_allclose(result.to_array().values, ref, equal_nan=True)
    assert (result.has_break.values[0] == 1).all() and (result.has_break.values[1:] == 0).all()
    assert np.nanmin(result.breakpoint.values[0]) >= 2014.0
    assert zeit.bfast_monitor(cube, 2014.0).equals(result)


def test_bfast_lite_and_bfast_match_the_engine(cube):
    lite = zeit.bfast_lite(cube, max_breaks=3)
    ref = run_bfast_lite_dask(_dask(cube), start_time=2010.0, frequency=23, max_breaks_output=3).compute()
    np.testing.assert_allclose(lite.to_array().values, ref, equal_nan=True)
    assert lite.n_breaks.values[0].min() >= 1 and (lite.n_breaks.values[1:] == 0).all()
    assert "breakpoint_idx_3" in lite

    classic = zeit.bfast(cube, max_breaks_trend=2, max_breaks_season=2)
    ref = run_bfast_dask(_dask(cube), start_time=2010.0, frequency=23, max_breaks_trend=2,
                         max_breaks_season=2).compute()
    np.testing.assert_allclose(classic.to_array().values, ref, equal_nan=True)
    assert "break_time" in classic and "time" not in classic.data_vars


def test_mann_kendall(cube):
    annual = cube.groupby("time.year").mean().rename(year="time")
    annual = annual.assign_coords(time=pd.to_datetime([f"{y}-01-01" for y in annual.time.values]))
    result = zeit.mann_kendall(annual)
    assert list(result.data_vars) == MK_METRIC_NAMES
    ref = run_mann_kendall_dask(da.from_array(annual.values, chunks=-1)).compute()
    np.testing.assert_allclose(result.to_array().values, ref, equal_nan=True)
    assert (result.trend.values[0] <= 0).all() and (result.trend.values[0] == -1).sum() >= 3   # 9 points: not all significant
    # no dates needed: a plain numpy stack works
    np.testing.assert_allclose(zeit.mann_kendall(annual.values).slope.values, result.slope.values, equal_nan=True)


def test_phenology(cube):
    result = zeit.phenology(cube)
    assert list(result.data_vars) == PHENOLOGY_METRICS
    assert result["TRS5.sos"].dims == ("year", "y", "x")
    assert result.year.values.tolist() == list(range(2010, 2019))
    assert result.rio.crs.to_epsg() == 4326
    sos = result["TRS5.sos"].values
    assert np.isfinite(sos).any() and np.nanmax(sos) <= 366
    raw = zeit.phenology(cube, annual=False, max_seasons=10)
    assert raw["POP"].dims == ("season", "y", "x") and raw.sizes["season"] == 10
    with pytest.raises(ValueError, match="curve"):
        zeit.phenology(cube, curve="sigmoid")
    weighted = zeit.phenology(cube, weights=np.ones(cube.shape, dtype=np.float32))
    np.testing.assert_allclose(weighted["POP"].values, result["POP"].values, equal_nan=True)


def test_inputs_pixel_numpy_dask_file(cube, tmp_path):
    reference = zeit.bfast_lite(cube)
    # dask: lazy result
    lazy = zeit.bfast_lite(cube.chunk({"y": 1, "x": 2}))
    assert isinstance(lazy.n_breaks.data, da.Array)
    xr.testing.assert_allclose(lazy.compute(), reference)
    # numpy + dates
    np.testing.assert_allclose(zeit.bfast_lite(cube.values, dates=DATES).n_breaks.values, reference.n_breaks.values)
    # one pixel: a Series indexed by date
    pixel = zeit.bfast_lite(pd.Series(cube.values[:, 0, 0], index=DATES))
    assert pixel.n_breaks.dims == () and float(pixel.n_breaks) == float(reference.n_breaks.values[0, 0])
    # a file written by save_raster keeps its dates
    path = zeit.save_raster(cube, tmp_path / "ndvi.tif")
    from_file = zeit.bfast_lite(str(path))
    np.testing.assert_allclose(from_file.n_breaks.values, reference.n_breaks.values)
    with pytest.raises(ValueError, match="dates="):
        zeit.bfast_lite(cube.values)


def test_save_results(cube, tmp_path):
    folder = zeit.save_raster(zeit.bfast_monitor(cube, "2014-01-01"), tmp_path / "bfm")
    assert sorted(p.name for p in folder.iterdir()) == sorted(f"{n}.tif" for n in BFM_METRIC_NAMES)
    folder = zeit.save_raster(zeit.phenology(cube), tmp_path / "pheno")
    with rasterio.open(folder / "TRS5.sos.tif") as src:
        assert src.count == 9 and src.descriptions[0] == "2010"


def test_cli_series_commands(cube, tmp_path):
    path = zeit.save_raster(cube, tmp_path / "ndvi.tif")
    for command, extra, expected in [
        ("bfast-monitor", ["--monitor-start-time", "2014.0"], "bfast_monitor.tif"),
        ("bfast-lite", [], "bfast_lite.tif"),
        ("mann-kendall", [], "mann_kendall.tif"),
    ]:
        out = tmp_path / command
        result = subprocess.run([sys.executable, "-m", "zeit.cli", command, str(path), str(out), *extra],
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        with rasterio.open(out / expected) as src:
            assert src.count > 1 and src.descriptions[0]
