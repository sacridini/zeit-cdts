from typing import Any

import numpy as np
import xarray as xr


def regularize_time_series(cube: Any, freq: str = '16D', method: str = 'median') -> xr.DataArray:
    """
    Regularize a time series to a fixed step: one composite per period, by median or medoid.

    - cube: a cube with a ``time`` dim, ``(time, y, x)`` or ``(time, band, y, x)`` (in memory
      or dask), or anything ``load_raster`` reads.
    - freq: the step, a pandas frequency (``'16D'``, ``'MS'``, ``'1YS'``...).
    - method: ``'median'`` (per band, each band its own median) or ``'medoid'`` (the
      observation closest to the median across bands, so the bands of a composite come
      from one date; with a single band, the observation closest to the median).

    Periods without observations are NaN. Georeferencing and NoData are kept.
    """
    if method not in ("median", "medoid"):
        raise ValueError(f"Unknown method {method}. Use 'median' or 'medoid'.")
    if not isinstance(cube, xr.DataArray):
        from ._load import load_raster
        cube = load_raster(cube)
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("regularize_time_series needs a time dim with dates")
    crs = cube.rio.crs if "y" in cube.dims else None
    nodata = cube.rio.nodata if "y" in cube.dims else None
    if method == 'median':
        out = cube.resample(time=freq).median(dim='time')
    else:
        resampled = cube.resample(time=freq)

        def _compute_medoid(group):
            # group has dims (time, [band,] y, x); median across time: ([band,] y, x)
            if group.sizes['time'] == 0:
                return group.isel(time=0, drop=True)
            median_val = group.median(dim='time')

            # distance to median: dims (time, y, x)
            diff = (group - median_val) ** 2
            # skipna=False: a missing observation is NaN away, not 0 (it would be the medoid)
            dist = np.sqrt(diff.sum(dim='band', skipna=False)) if 'band' in group.dims else np.sqrt(diff)

            # find index of min distance along time
            idx = dist.fillna(np.inf).argmin(dim='time')

            # select the medoid values using advanced indexing
            return group.isel(time=idx)

        # Using map to apply the function to each temporal window
        out = resampled.map(_compute_medoid)
    if crs is not None and out.rio.crs is None:
        out = out.rio.write_crs(crs)
    if nodata is not None and out.rio.nodata is None:
        out = out.rio.write_nodata(nodata, encoded=False)
    return out
