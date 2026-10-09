import pytest
import numpy as np
import xarray as xr
import dask.array as da
import zeit  # This registers the xarray accessor automatically

def test_xarray_ccdc_accessor():
    """
    Test that the xarray accessor correctly maps the C++ CCDC algorithm over Dask blocks.
    """
    # Create dummy data: (bands, time, y, x)
    bands, time, y, x = 3, 20, 2, 2
    
    # Random scaled reflectance data
    data = np.random.randint(0, 10000, size=(bands, time, y, x), dtype=np.int16)
    
    # Create Dask-backed xarray DataArray chunked spatially (y:1, x:1)
    dask_data = da.from_array(data, chunks=(bands, time, 1, 1))
    
    da_arr = xr.DataArray(
        dask_data, 
        dims=["band", "time", "y", "x"],
        coords={"y": [10, 20], "x": [30, 40]}
    )
    
    # Dummy fractional years
    dates = np.linspace(2000.0, 2010.0, time)
    
    # Run CCDC lazily with Strategy A (n_jobs=-1): the accessor delegates to zeit.ccdc,
    # which puts the cube in (time, band, y, x) order and reads the dates
    max_segments = 4
    result = da_arr.zeit.ccdc(dates=dates, max_segments=max_segments, n_jobs=-1)

    # Check that it's still lazy (Dask array inside)
    assert isinstance(result.coefs.data, da.Array)

    # Check expected output dimensions
    assert result.coefs.dims == ("segment", "band", "coef", "y", "x")
    assert result.coefs.shape == (max_segments, bands, 8, y, x)
    assert result.t_start.dims == ("segment", "y", "x")

    # Compute the graph and verify execution completes without crashing
    computed_result = result.compute()
    assert computed_result.rmse.shape == (max_segments, bands, y, x)

def test_xarray_landtrendr_accessor():
    """
    Test that the xarray accessor correctly maps the C++ LandTrendr algorithm over Dask blocks.
    """
    # Create dummy data: (time, y, x) for LandTrendr (single index)
    time, y, x = 15, 2, 2
    
    # Generate an index with a sudden drop
    data = np.linspace(8000, 7000, time)
    data[7:] -= 3000  # Introduce a break
    data = np.broadcast_to(data[:, None, None], (time, y, x)).copy()
    
    # Chunk spatially
    dask_data = da.from_array(data, chunks=(time, 1, 1))
    
    da_arr = xr.DataArray(
        dask_data,
        dims=["time", "y", "x"],
        coords={"y": [10, 20], "x": [30, 40]}
    )
    
    years = np.arange(2000, 2000 + time)
    
    max_segments = 3
    max_vertices = max_segments + 1
    
    # Run LandTrendr lazily with Strategy A (n_jobs=-1): the accessor delegates to zeit.landtrendr
    result = da_arr.zeit.landtrendr(years=years, max_segments=max_segments, n_jobs=-1)

    # Check laziness
    assert isinstance(result.vertex_year.data, da.Array)
    assert result.vertex_year.dims == ("vertex", "y", "x")
    assert result.vertex_year.shape == (max_vertices, y, x)

    # Compute and verify: pixel (0,0) starts at the first year
    computed_result = result.compute()
    assert computed_result.vertex_year.values[0, 0, 0] == 2000
    assert computed_result.n_vertices.values.min() >= 2

def test_xarray_to_zarr(tmp_path):
    """
    Test that the xarray accessor can export a lazy computation directly to Zarr.
    """
    time, y, x = 15, 8, 8
    data = np.random.randn(time, y, x).astype(np.float32)
    
    # Create chunked xarray
    da_arr = xr.DataArray(
        da.from_array(data, chunks=(time, 4, 4)),
        dims=["time", "y", "x"]
    )
    
    # Run LT lazily
    years = np.arange(2000, 2000 + time)
    result = da_arr.zeit.landtrendr(years=years, max_segments=2, direction="gain")

    # Export to zarr using the accessor, optimizing chunks
    zarr_path = str(tmp_path / "test.zarr")
    result.vertex_value.zeit.to_zarr_optimized(zarr_path, chunk_size={"y": 4, "x": 4})

    # Read back and verify: 2 segments = 3 vertices
    ds_zarr = xr.open_zarr(zarr_path)
    assert "data" in ds_zarr.data_vars
    assert ds_zarr["data"].shape == (3, y, x)
    assert ds_zarr["data"].chunks == ((3,), (4, 4), (4, 4))

