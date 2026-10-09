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
