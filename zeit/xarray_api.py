import xarray as xr
import dask.array as da
import numpy as np
from typing import Optional, Any
from pathlib import Path

@xr.register_dataarray_accessor("zeit")
class ZeitAccessor:
    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def ccdc(self, **kwargs: Any) -> xr.Dataset:
        """
        Runs CCDC on this (time, band, y, x) cube: the same as ``zeit.ccdc(cube, **kwargs)``.
        Dask-backed cubes stay lazy and are computed block by block (time and band in one chunk).
        """
        from zeit._ccdc_api import ccdc
        return ccdc(self._obj, **kwargs)

    def landtrendr(self, **kwargs: Any) -> xr.Dataset:
        """
        Runs LandTrendr on this (time, y, x) cube: the same as ``zeit.landtrendr(cube, **kwargs)``.
        Dask-backed cubes stay lazy and are computed block by block (time in one chunk).
        """
        from zeit._lt import landtrendr
        return landtrendr(self._obj, **kwargs)

    def phenology(self, **kwargs: Any) -> xr.Dataset:
        """
        Runs phenology on this (time, y, x) cube: the same as ``zeit.phenology(cube, ...)``.
        """
        from zeit._series_api import phenology
        return phenology(self._obj, **kwargs)

    def mann_kendall(self, **kwargs: Any) -> xr.Dataset:
        """
        Runs Mann-Kendall on this (time, y, x) cube: the same as ``zeit.mann_kendall(cube, ...)``.
        """
        from zeit._series_api import mann_kendall
        return mann_kendall(self._obj, **kwargs)

    def bfast_monitor(self, monitor_start: Any, **kwargs: Any) -> xr.Dataset:
        """
        Runs bfastmonitor on this (time, y, x) cube: the same as ``zeit.bfast_monitor(cube, ...)``.
        """
        from zeit._series_api import bfast_monitor
        return bfast_monitor(self._obj, monitor_start, **kwargs)

    def bfast_lite(self, **kwargs: Any) -> xr.Dataset:
        """
        Runs bfastlite on this (time, y, x) cube: the same as ``zeit.bfast_lite(cube, ...)``.
        """
        from zeit._series_api import bfast_lite
        return bfast_lite(self._obj, **kwargs)

    def bfast(self, **kwargs: Any) -> xr.Dataset:
        """
        Runs bfast on this (time, y, x) cube: the same as ``zeit.bfast(cube, ...)``.
        """
        from zeit._series_api import bfast
        return bfast(self._obj, **kwargs)

    def twdtw(self, patterns: Any, **kwargs: Any) -> xr.Dataset:
        """
        Classifies this (time, y, x) or (time, band, y, x) cube by TWDTW: the same as
        ``zeit.twdtw(cube, patterns, **kwargs)``.
        """
        from zeit._twdtw_api import twdtw
        return twdtw(self._obj, patterns, **kwargs)

    def smooth(self, **kwargs: Any) -> xr.DataArray:
        """
        Smooths this cube along time: the same as ``zeit.smooth(cube, **kwargs)``.
        """
        from zeit._smooth import smooth
        return smooth(self._obj, **kwargs)

    def snic(self, **kwargs: Any) -> xr.Dataset:
        """
        SNIC superpixels of this map or cube: the same as ``zeit.snic(da, **kwargs)``.
        """
        from zeit._snic_api import snic
        return snic(self._obj, **kwargs)

    def som(self, **kwargs: Any) -> xr.Dataset:
        """
        Clusters the pixels of this map or cube with a Self-Organizing Map: the same as
        ``zeit.som(da, **kwargs)``.
        """
        from zeit._som_api import som
        return som(self._obj, **kwargs)

    def tmask(self, **kwargs: Any) -> xr.DataArray:
        """
        Tmask cloud and shadow screening of this (time, band, y, x) cube: the same as
        ``zeit.tmask(cube, **kwargs)``.
        """
        from zeit._tmask_api import tmask
        return tmask(self._obj, **kwargs)

    def save(self, path: Any, **kwargs: Any) -> Path:
        """
        Writes this map, cube or result to a raster: the same as ``zeit.save_raster(da, path, **kwargs)``.
        Returns the path written.
        """
        from zeit._save import save_raster
        return save_raster(self._obj, path, **kwargs)

    def plot(self, **kwargs: Any) -> Any:
        """
        Shows this map or cube: the same as ``zeit.plot(da, **kwargs)`` (the viewer in a notebook,
        a window in a script, a figure with ``static=True`` or ``save=``).
        """
        from zeit._plot import plot
        return plot(self._obj, **kwargs)

    def to_zarr_optimized(self, store_path: str, chunk_size: dict = {"y": 512, "x": 512}) -> None:
        """
        Optimizes and saves the DataArray directly to a Zarr store, ideal for cloud storage (S3/GCS) 
        and rapid multidimensional time-series queries.
        """
        # Ensure optimal chunking before saving
        optimized_ds = self._obj.chunk(chunk_size)
        optimized_ds.to_dataset(name="data").to_zarr(store_path, mode="w", consolidated=True)


@xr.register_dataset_accessor("zeit")
class ZeitDatasetAccessor:
    """``.zeit`` on a Dataset, such as the result of ``zeit.landtrendr`` or ``zeit.ccdc``:
    write it or show it without leaving the chain of calls."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def save(self, path: Any, **kwargs: Any) -> Path:
        """
        Writes this result: the same as ``zeit.save_raster(ds, path, **kwargs)`` (a path without an
        extension becomes a folder with one GeoTIFF per variable, a ``.tif`` one multiband raster).
        Returns the path written.
        """
        from zeit._save import save_raster
        return save_raster(self._obj, path, **kwargs)

    def plot(self, **kwargs: Any) -> Any:
        """
        Shows this result: the same as ``zeit.plot(ds, **kwargs)``, with a selector of its variables
        (or one of them with ``var=``).
        """
        from zeit._plot import plot
        return plot(self._obj, **kwargs)
