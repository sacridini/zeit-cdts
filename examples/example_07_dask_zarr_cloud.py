import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import dask.distributed
import zeit
import os
import shutil

def main():
    print('=' * 60)
    print('Zeit Example 07: Dask, Zarr, and Cloud Strategy A')
    print('=' * 60)

    # 1. Start a local Dask cluster with 1 Worker (Strategy A: OpenMP handles threads)
    print('\n[1/4] Starting Dask Cluster (1 Worker, relying on OpenMP for threads)...')
    cluster = dask.distributed.LocalCluster(n_workers=1, threads_per_worker=1)
    client = dask.distributed.Client(cluster)
    print(f'Dashboard available at: {client.dashboard_link}')

    # 2. Create a massive synthetic time-series DataArray (Simulating Zarr from cloud)
    print('\n[2/4] Generating synthetic remote sensing data cube...')
    # Shape: (time, y, x)
    time_steps = 30
    y, x = 1000, 1000

    # We create a Dask array that is chunked spatially (e.g. 250x250 pixels per chunk)
    synthetic_data = da.random.randint(0, 10000, size=(time_steps, y, x), chunks=(time_steps, 250, 250))
    cube = xr.DataArray(
        synthetic_data,
        dims=['time', 'y', 'x'],
        coords={'time': pd.date_range('1990-01-01', periods=time_steps, freq='YS')}   # one date per year
    )

    # 3. Apply LandTrendr across the cluster using Strategy A
    print('\n[3/4] Mapping LandTrendr across the Dask cluster...')
    print('      Note: n_jobs=-1 tells Zeit to use all CPU cores via OpenMP inside each chunk!')
    # Lazy: the years come from the time coordinate, and each chunk is segmented when written.
    lt_results = zeit.landtrendr(cube, max_segments=4, n_jobs=-1)

    # 4. Save the results directly to Zarr format
    zarr_output = 'scratch_output_landtrendr.zarr'
    if os.path.exists(zarr_output):
        shutil.rmtree(zarr_output)

    print(f'\n[4/4] Executing computation and writing to {zarr_output}...')
    # The result is an xarray.Dataset (vertex_year, vertex_value, n_vertices, rmse):
    # Dask computes it chunk by chunk while it is written.
    lt_results.chunk({'y': 250, 'x': 250}).to_zarr(zarr_output, mode='w')

    # Verify the saved output
    saved_ds = xr.open_zarr(zarr_output)
    print('\nSuccess! Zarr store contents:')
    print(saved_ds)

    # Cleanup
    client.close()
    cluster.close()

if __name__ == '__main__':
    main()
