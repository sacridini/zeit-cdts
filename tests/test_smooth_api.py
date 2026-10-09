"""zeit.smooth: Whittaker (the classic one on even series, divided differences on uneven
ones, gaps weighing 0) and Savitzky-Golay (scipy's), keeping the cube."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import xarray as xr
from rasterio.transform import from_origin
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

import zeit
from zeit._smooth import apply_whittaker_filter


def cube(dates, seed=0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    t = pd.DatetimeIndex(dates)
    doy = t.dayofyear.to_numpy()
    base = 0.5 + 0.3 * np.sin(2 * np.pi * doy / 365.25)
    v = (base[:, None, None] + rng.normal(0, 0.05, (t.size, 3, 4))).astype(np.float32)
    tr = from_origin(500000, 9000000, 30, 30)
    return xr.DataArray(v, dims=("time", "y", "x"),
                        coords={"time": t, "y": tr.f - 30 * (np.arange(3) + 0.5), "x": tr.c + 30 * (np.arange(4) + 0.5)},
                        name="ndvi").rio.write_crs("EPSG:32722").rio.write_transform(tr)


def whittaker_reference(x, y, w, lmbda):
    """The Whittaker smoother written out: divided second differences in median steps."""
    n = len(y)
    h = np.diff(x) / np.median(np.diff(x))
    rows, cols, vals = [], [], []
    for k in range(n - 2):
        h1, h2 = h[k], h[k + 1]
        for p, c in enumerate((2 / (h1 * (h1 + h2)), -2 / (h1 * h2), 2 / (h2 * (h1 + h2)))):
            rows.append(k), cols.append(k + p), vals.append(c)
    d = sp.csc_matrix((vals, (rows, cols)), shape=(n - 2, n))
    a = sp.diags(w) + lmbda * (d.T @ d)
    return spsolve(a.tocsc(), w * np.nan_to_num(y))


def test_whittaker_on_an_even_series_is_the_classic_one():
    da = cube(pd.date_range("2020-01-01", periods=46, freq="8D"))
    out = zeit.smooth(da, lmbda=20)
    expected = apply_whittaker_filter(da.values.astype(np.float64), lmbd=20.0)
    np.testing.assert_allclose(out.values, expected, rtol=1e-5, atol=1e-6)
    assert out.dtype == np.float32 and out.dims == da.dims and out.rio.crs == da.rio.crs
    assert (out.time.values == da.time.values).all() and out.attrs["smoothing"] == "Whittaker (lambda=20)"


def test_whittaker_on_uneven_dates_with_gaps():
    dates = pd.to_datetime(["2020-01-01", "2020-01-09", "2020-02-10", "2020-02-18", "2020-03-05", "2020-04-22",
                            "2020-04-30", "2020-05-08", "2020-06-25", "2020-07-03", "2020-08-04", "2020-08-12"])
    da = cube(dates).astype(np.float64)
    da[[2, 6], 0, 0] = np.nan
    out = zeit.smooth(da, lmbda=5)
    days = ((dates - dates[0]).days).to_numpy().astype(float)
    y = da.values[:, 0, 0]
    expected = whittaker_reference(days, y, np.isfinite(y).astype(float), 5.0)
    np.testing.assert_allclose(out.values[:, 0, 0], expected, rtol=1e-9)
    assert np.isfinite(out.values).all()  # the gaps are filled
    weights = xr.ones_like(da).where(np.isfinite(da), 0) * 0.5
    weighted = zeit.smooth(da, lmbda=5, weights=weights)
    np.testing.assert_allclose(weighted.values[:, 0, 0],
                               whittaker_reference(days, y, np.where(np.isfinite(y), 0.5, 0.0), 5.0), rtol=1e-9)


def test_savgol_is_scipys_and_fills_gaps():
    da = cube(pd.date_range("2020-01-01", periods=30, freq="16D")).astype(np.float64)
    out = zeit.smooth(da, method="savgol", window=7, polyorder=2)
    np.testing.assert_allclose(out.values, savgol_filter(da.values, 7, 2, axis=0), rtol=1e-12)
    da[5, 1, 1] = np.nan
    gap = zeit.smooth(da, method="savgol", window=7, polyorder=2)
    row = da.values[:, 1, 1].copy()
    row[5] = (row[4] + row[6]) / 2   # evenly spaced: the linear fill is the midpoint
    np.testing.assert_allclose(gap.values[:, 1, 1], savgol_filter(row, 7, 2), rtol=1e-12)


def test_integers_keep_their_type_and_nodata():
    da = cube(pd.date_range("2020-01-01", periods=23, freq="16D"))
    scaled = (da * 10000).astype(np.int16)
    scaled[:, 2, 3] = -9999
    scaled[4, 0, 0] = -9999
    scaled = scaled.rio.write_nodata(-9999, encoded=False)
    out = zeit.smooth(scaled, lmbda=10)
    assert out.dtype == np.int16 and out.rio.nodata == -9999
    assert (out.values[:, 2, 3] == -9999).all() and out.values[4, 0, 0] != -9999
    floats = zeit.smooth((da * 10000).astype(np.int16), lmbda=10)   # no NoData: floats
    assert floats.dtype == np.float32


def test_lazy_series_and_plot():
    da = cube(pd.date_range("2020-01-01", periods=23, freq="16D"))
    eager = zeit.smooth(da, lmbda=30)
    lazy = zeit.smooth(da.chunk({"time": 5, "x": 2}), lmbda=30)
    assert lazy.chunks is not None
    np.testing.assert_array_equal(lazy.values, eager.values)
    series = da.isel(y=0, x=0).to_series()
    smoothed = zeit.smooth(series, lmbda=30)
    assert isinstance(smoothed, pd.Series)
    np.testing.assert_allclose(smoothed.values, eager.values[:, 0, 0], rtol=1e-6)
    from zeit._plot._data import Frames
    from zeit._plot._fit import overlays, pixel_series

    lines = overlays(eager, pixel_series(Frames(da), 1, 2), shape=(3, 4))
    assert lines[0]["label"] == "Whittaker (lambda=30)"
    np.testing.assert_allclose(lines[0]["y"], eager.values[:, 2, 1], rtol=1e-6)


def test_errors():
    da = cube(pd.date_range("2020-01-01", periods=10, freq="16D"))
    with pytest.raises(ValueError, match="method"):
        zeit.smooth(da, method="loess")
    with pytest.raises(ValueError, match="odd"):
        zeit.smooth(da, method="savgol", window=4)
    with pytest.raises(ValueError, match="dates"):
        zeit.smooth(da.isel(time=[0, 1]))
