"""zeit.twdtw: one function for every input, the distances of R's twdtw.

data/twdtw_r_parity.json holds series, patterns and the distances R's twdtw (1.0.1) gives
for them (time_weight = c(steepness, midpoint), cycle_length = "year", time_scale = "day";
see data/make_twdtw_parity_fixtures.*): one band, cloudy dates (left out, as R's
complete.cases), a pattern inside a four-year series and two bands.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit

CASES = json.loads((Path(__file__).parent / "data" / "twdtw_r_parity.json").read_text())


def _frame(item) -> pd.DataFrame:
    values = [[np.nan if e is None else e for e in (row if isinstance(row, list) else [row])]
              for row in item["values"]]
    frame = pd.DataFrame(values, index=pd.to_datetime(item["time"]))
    frame.columns = [f"b{i + 1}" for i in range(frame.shape[1])]
    return frame


def _as_input(frame: pd.DataFrame):
    return frame.iloc[:, 0] if frame.shape[1] == 1 else frame


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_distances_are_rs(case):
    series = _as_input(_frame(case["series"]))
    patterns = {name: _as_input(_frame(p)) for name, p in case["patterns"].items()}
    result = zeit.twdtw(series, patterns, steepness=case["steep"], midpoint=case["mid"])
    expected = [case["r_distance"][name] for name in patterns]
    np.testing.assert_allclose(result.distances.values, expected, rtol=1e-12)
    assert result.label.item() == int(np.argmin(expected)) + 1
    assert result.distance.item() == pytest.approx(min(expected), rel=1e-12)


def season(dates, peak, low=0.2, high=0.8, width=60.0):
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    d = np.minimum(np.abs(doy - peak), 366 - np.abs(doy - peak))
    return low + (high - low) * np.exp(-0.5 * (d / width) ** 2)


PATTERN_DATES = pd.date_range("2019-09-01", "2020-08-31", freq="16D")
PATTERNS = {"crop": pd.Series(season(PATTERN_DATES, 60), PATTERN_DATES),
            "forest": pd.Series(np.full(PATTERN_DATES.size, 0.75), PATTERN_DATES)}


def cube(nodata=None) -> xr.DataArray:
    """Crop on the left half, forest on the right; one pixel cloudy on a date, one never seen."""
    rng = np.random.default_rng(1)
    t = pd.date_range("2021-09-01", "2022-08-31", freq="16D")
    v = np.empty((t.size, 3, 4))
    v[:, :, :2] = season(t, 65)[:, None, None]
    v[:, :, 2:] = 0.75
    v += rng.normal(0, 0.03, v.shape)
    v[3, 0, 0] = np.nan
    v[:, 2, 3] = np.nan
    tr = from_origin(500000, 9000000, 30, 30)
    da = xr.DataArray(v, dims=("time", "y", "x"),
                      coords={"time": t, "y": tr.f - 30 * (np.arange(3) + 0.5), "x": tr.c + 30 * (np.arange(4) + 0.5)},
                      name="ndvi").rio.write_crs("EPSG:32722").rio.write_transform(tr)
    return da


def test_a_cube_is_classified_lazily_or_not():
    da = cube()
    result = zeit.twdtw(da, PATTERNS)
    expected = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(result.label.values, expected)
    assert result.label.attrs["flag_meanings"] == "crop forest" and result.label.rio.nodata == 0
    assert list(result.pattern.values) == ["crop", "forest"] and result.rio.crs == da.rio.crs
    assert np.isnan(result.distance.values[2, 3]) and np.isnan(result.distances.values[:, 2, 3]).all()
    lazy = zeit.twdtw(da.chunk({"y": 2, "x": 2}), PATTERNS)
    assert lazy.distances.chunks is not None
    np.testing.assert_array_equal(lazy.distances.values, result.distances.values)
    np.testing.assert_array_equal(lazy.label.values, result.label.values)
    # every pixel as R would see it alone
    one = zeit.twdtw(da.isel(y=0, x=0).to_series(), PATTERNS)
    np.testing.assert_allclose(one.distances.values, result.distances.values[:, 0, 0])


def test_nodata_marks_missing_dates():
    da = cube()
    scaled = (da * 10000).fillna(-9999).astype(np.int16).rio.write_nodata(-9999, encoded=False)
    patterns = {k: v * 10000 for k, v in PATTERNS.items()}
    result = zeit.twdtw(scaled, patterns)
    assert result.label.values[2, 3] == 0 and result.label.values[0, 0] == 1


def test_several_bands_are_taken_by_name():
    da = cube()
    two = xr.concat([da, 1 - da], dim="band").assign_coords(band=["ndvi", "swir"]).transpose("time", "band", "y", "x")
    two = two.rio.write_crs(da.rio.crs)
    patterns = {k: pd.DataFrame({"swir": 1 - v, "ndvi": v}) for k, v in PATTERNS.items()}
    result = zeit.twdtw(two, patterns)
    assert list(result.band.values) == ["swir", "ndvi"] and result.pattern_value.dims == ("pattern", "pattern_step", "band")
    np.testing.assert_array_equal(result.label.values[:, :2], 1)
    with pytest.raises(ValueError, match="band="):
        zeit.twdtw(two, PATTERNS)
    one = zeit.twdtw(two, PATTERNS, band="ndvi")
    np.testing.assert_array_equal(one.label.values, zeit.twdtw(da, PATTERNS).label.values)


def test_cycle_and_max_elapsed():
    da = cube()
    later = {k: pd.Series(v.values, v.index + pd.DateOffset(years=4)) for k, v in PATTERNS.items()}
    yearly = zeit.twdtw(da, PATTERNS)
    # with the cycle, the year of the pattern does not matter
    np.testing.assert_allclose(zeit.twdtw(da, later).distances.values, yearly.distances.values)
    plain = zeit.twdtw(da, later, cycle=None)   # the dates themselves: the pattern is 4 years off
    assert (plain.distances.values[np.isfinite(plain.distances.values)]
            > yearly.distances.values[np.isfinite(plain.distances.values)]).all()
    near = zeit.twdtw(da, PATTERNS, max_elapsed=3)
    assert (near.distances.values[np.isfinite(near.distances.values)]
            >= yearly.distances.values[np.isfinite(near.distances.values)] - 1e-12).all()


def test_results_save_and_plot(tmp_path):
    da = cube()
    result = zeit.twdtw(da, PATTERNS)
    folder = zeit.save_raster(result, tmp_path / "twdtw")
    assert sorted(p.name for p in folder.iterdir()) == ["distance.tif", "distances.tif", "label.tif"]
    from zeit._plot._data import Frames
    from zeit._plot._fit import overlays, pixel_series
    from zeit._plot._style import classes_from_attrs

    assert classes_from_attrs(result.label.attrs) == {1.0: "crop", 2.0: "forest"}
    lines = overlays(result, pixel_series(Frames(da), 0, 0), shape=(3, 4))
    assert len(lines) == 1 and lines[0]["label"].startswith("TWDTW: crop")
    assert all(y is not None for y in lines[0]["y"])
    result.label.zeit.plot(static=True, save=str(tmp_path / "label.png"))


def test_errors():
    da = cube()
    with pytest.raises(ValueError, match="cycle"):
        zeit.twdtw(da, PATTERNS, cycle="month")
    with pytest.raises(ValueError, match="spaces"):
        zeit.twdtw(da, {"soy bean": PATTERNS["crop"]})
    with pytest.raises(TypeError, match="pattern"):
        zeit.twdtw(da, {"crop": [0.1, 0.2]})
    with pytest.raises(ValueError, match="same bands"):
        zeit.twdtw(da, {"a": PATTERNS["crop"], "b": pd.DataFrame({"x": PATTERNS["crop"], "y": PATTERNS["crop"]})})
