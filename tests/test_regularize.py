import xarray as xr
import numpy as np
import pandas as pd
import pytest
from zeit.regularize import regularize_time_series

def test_regularize_median():
    time = pd.date_range("2020-01-01", periods=5, freq="5D")
    data = np.random.rand(5, 2, 3, 3)
    cube = xr.DataArray(data, dims=["time", "band", "y", "x"], coords={"time": time})
    
    result = regularize_time_series(cube, freq="16D", method="median")
    assert result.dims == ("time", "band", "y", "x")
    # 5 intervals of 5 days spans 20 days total. Start 01-01. Resampling '16D' from 01-01 gives bins: 01-01 and 01-17.
    assert len(result.time) == 2

def test_regularize_medoid():
    time = pd.date_range("2020-01-01", periods=5, freq="5D")
    data = np.random.rand(5, 2, 3, 3)
    cube = xr.DataArray(data, dims=["time", "band", "y", "x"], coords={"time": time})
    
    result = regularize_time_series(cube, freq="16D", method="medoid")
    assert result.dims == ("time", "band", "y", "x")
    assert len(result.time) == 2


def test_medoid_skips_a_missing_observation():
    """A date without data in a pixel is not its medoid (the distance of NaN is not 0); the
    annual medoid is the one cbers_to_landtrendr computed in C++."""
    times = pd.to_datetime(["2020-01-15", "2020-04-10", "2020-07-20"])
    v = np.array([[671.9, 167.0, 825.9], [np.nan, np.nan, np.nan], [375.6, 763.7, 392.4]])
    da = xr.DataArray(v.reshape(3, 3, 1, 1), dims=("time", "band", "y", "x"),
                      coords={"time": times, "band": ["b", "g", "r"], "y": [0.5], "x": [0.5]})
    out = regularize_time_series(da, freq="YS", method="medoid")
    assert np.isfinite(out.values).all()
    assert out.values.ravel().tolist() in (v[0].tolist(), v[2].tolist())


def _brute_medoid(series):
    """(time, band) -> the observation with every band closest to the per-band median of the
    valid values, the first one on a tie (float64)."""
    series = np.asarray(series, dtype=np.float64)
    median = np.nanmedian(series, axis=0)
    complete = ~np.isnan(series).any(axis=1)
    if not complete.any():
        return np.full(series.shape[1], np.nan)
    dist = np.where(complete, ((series - median) ** 2).sum(axis=1), np.inf)
    return series[int(np.argmin(dist))]


def test_medoid_is_the_brute_force_one_lazy_or_not():
    rng = np.random.default_rng(0)
    times = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.sort(rng.choice(120, 30, replace=False)), "D")
    v = rng.random((30, 3, 6, 7)).astype(np.float32)
    v[rng.random((30, 3, 6, 7)) < 0.1] = np.nan
    cube = xr.DataArray(v, dims=("time", "band", "y", "x"),
                        coords={"time": times, "band": ["b", "g", "r"], "y": np.arange(6.0), "x": np.arange(7.0)})
    out = regularize_time_series(cube, freq="MS", method="medoid")
    assert out.dims == ("time", "band", "y", "x") and out.dtype == np.float32
    for k, (_, group) in enumerate(cube.resample(time="MS")):
        for r in range(6):
            for c in range(7):
                expected = _brute_medoid(group.values[:, :, r, c])
                np.testing.assert_array_equal(out.values[k, :, r, c], expected.astype(np.float32))
    lazy = regularize_time_series(cube.chunk({"y": 3, "x": 4}), freq="MS", method="medoid")
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, out.values)
    single = regularize_time_series(cube.isel(band=0), freq="MS", method="medoid")
    assert single.dims == ("time", "y", "x")


def test_medoid_ties_take_the_first_date_and_needs_every_band():
    times = pd.to_datetime(["2020-01-05", "2020-01-10", "2020-01-20", "2020-01-25"])
    one = xr.DataArray(np.array([4.0, 1.0, 3.0, 2.0]).reshape(4, 1, 1), dims=("time", "y", "x"),
                       coords={"time": times, "y": [0.5], "x": [0.5]})
    # median 2.5: 2.0 (the 25th) and 3.0 (the 20th) are as close; the first date wins
    assert float(regularize_time_series(one, freq="MS", method="medoid").squeeze()) == 3.0
    two = xr.DataArray(np.array([[1.0, np.nan], [np.nan, 2.0]]).reshape(2, 2, 1, 1), dims=("time", "band", "y", "x"),
                       coords={"time": times[:2], "band": ["a", "b"], "y": [0.5], "x": [0.5]})
    assert np.isnan(regularize_time_series(two, freq="MS", method="medoid").values).all()   # no complete date


@pytest.mark.parametrize("method", ["median", "medoid"])
def test_an_integer_cube_keeps_its_nodata_out(method):
    times = pd.to_datetime(["2020-01-05", "2020-01-10", "2020-01-20", "2020-03-01"])
    v = np.array([1000, -9999, 3000, 5000], dtype=np.int16).reshape(4, 1, 1)
    cube = xr.DataArray(v, dims=("time", "y", "x"), coords={"time": times, "y": [0.5], "x": [0.5]})
    cube = cube.rio.write_crs("EPSG:32633").rio.write_nodata(-9999, encoded=False)
    out = regularize_time_series(cube, freq="MS", method=method)
    assert out.dtype == np.int16 and out.rio.nodata == -9999
    assert out.values.ravel().tolist() == [2000 if method == "median" else 1000, -9999, 5000]  # February: empty


def _gappy_cube():
    """Three bands-less dates in January, one in March, one in July (pixels 2x2)."""
    times = pd.to_datetime(["2020-01-02", "2020-01-10", "2020-01-20", "2020-03-05", "2020-07-04"])
    v = np.full((5, 2, 2), np.nan)
    v[0] = [[1.0, np.nan], [np.nan, np.nan]]       # 1 of 4 pixels
    v[1] = [[2.0, 3.0], [4.0, np.nan]]             # 3 of 4: the clearest of January
    v[2] = [[5.0, 6.0], [np.nan, 7.0]]             # 3 of 4: ties with the 10th, which is earlier
    v[3] = [[10.0, 10.0], [10.0, 10.0]]
    v[4] = [[20.0, 20.0], [20.0, 20.0]]
    return xr.DataArray(v, dims=("time", "y", "x"), coords={"time": times, "y": [1.5, 0.5], "x": [0.5, 1.5]})


def test_clearest_takes_the_clearest_date_pixel_by_pixel():
    out = regularize_time_series(_gappy_cube(), "MS", method="clearest")
    assert list(out.time.dt.strftime("%Y-%m").values) == [f"2020-{m:02d}" for m in range(1, 8)]
    # January: the 10th (3 of 4 pixels, earlier than the 20th), then the 20th, then the 2nd
    assert np.array_equal(out.values[0], [[2.0, 3.0], [4.0, 7.0]], equal_nan=True)
    assert np.isnan(out.values[1]).all() and (out.values[2] == 10).all() and (out.values[6] == 20).all()


def test_clearest_needs_every_band_and_stays_lazy():
    cube = _gappy_cube()
    two = xr.concat([cube, cube * 2], dim="band").assign_coords(band=["a", "b"]).transpose("time", "band", "y", "x")
    two = two.where(~((two.band == "b") & (two.time == two.time[1])))     # the 10th loses band b everywhere
    out = regularize_time_series(two, "MS", method="clearest")
    # the 10th has no complete observation now: January is the 20th (3 of 4 pixels), then the 2nd
    assert np.array_equal(out.values[0, 0], [[5.0, 6.0], [np.nan, 7.0]], equal_nan=True)
    assert np.array_equal(out.values[0, 1], 2 * out.values[0, 0], equal_nan=True)
    lazy = regularize_time_series(two.chunk({"x": 1}), "MS", method="clearest")
    assert np.array_equal(lazy.values, out.values, equal_nan=True)


def test_fill_linear_in_time_and_max_gap():
    cube = _gappy_cube()
    full = regularize_time_series(cube, "MS", fill="linear")
    # the value of the first pixel: 2 in January, 10 in March: February is the middle in time
    jan, feb, mar = full.values[0, 0, 0], full.values[1, 0, 0], full.values[2, 0, 0]
    assert (jan, mar) == (2.0, 10.0)
    assert np.isclose(feb, 2.0 + (10.0 - 2.0) * 31 / 60)          # 2020-01-01 -> 02-01 -> 03-01 (60 days)
    assert not np.isnan(full.values[:, 0, 0]).any()                  # no gap left (July is the last value)
    limited = regularize_time_series(cube, "MS", fill="linear", max_gap=30)
    assert np.isnan(limited.values[1]).all()                         # 60 days between its values
    assert np.isnan(limited.values[3:6]).all()                       # March -> July: 122 days
    assert np.array_equal(limited.values[0], full.values[0], equal_nan=True)


def test_fill_whittaker_fills_only_the_gaps_and_keeps_the_values():
    cube = _gappy_cube()
    plain = regularize_time_series(cube, "MS")
    filled = regularize_time_series(cube, "MS", fill="whittaker", lmbda=1.0)
    keep = plain.notnull()
    assert np.array_equal(filled.values[keep.values], plain.values[keep.values])
    assert filled.notnull().values[:, 0, 0].all()
    limited = regularize_time_series(cube, "MS", fill="whittaker", max_gap=30)
    assert np.isnan(limited.values[1]).all()


def test_integer_cube_with_nodata_is_filled_and_keeps_its_type():
    cube = _gappy_cube().fillna(-9999).astype("int16").rio.write_nodata(-9999)
    out = regularize_time_series(cube, "MS", fill="linear")
    assert out.dtype == np.int16 and out.values[1, 0, 0] == round(2.0 + 8.0 * 31 / 60)


def test_regularize_options_are_checked():
    with pytest.raises(ValueError):
        regularize_time_series(_gappy_cube(), "MS", fill="cubic")
    with pytest.raises(ValueError):
        regularize_time_series(_gappy_cube(), "MS", method="mode")
    with pytest.raises(ValueError):
        regularize_time_series(_gappy_cube(), "MS", fill="linear", max_gap=-1)
