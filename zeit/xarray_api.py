import xarray as xr
import dask.array as da
import numpy as np
from typing import Optional, Any

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

    def run_snic(self, spacing: Any = 10,
                 compactness: float = 0.5, seeds: Optional[np.ndarray] = None,
                 grid: str = "rectangular", padding: Optional[Any] = None,
                 tile_size: Optional[Any] = None, random_state: Optional[int] = None,
                 n_jobs: int = -1) -> xr.Dataset:
        """
        SNIC superpixel segmentation of the whole cube. The last two dims
        must be (y, x); every other dim (e.g. time, band) becomes a feature,
        so segments group pixels with similar trajectories.
        Returns a Dataset with "labels" (y, x; -1 = unlabelled), the segment
        mean trajectories "means" (segment, *other dims), "centroid_row",
        "centroid_col" and "n_pixels" (segment). Arguments as in
        zeit.segmentation.run_snic. The cube is loaded into memory.
        """
        from zeit.segmentation import run_snic

        obj = self._obj
        if obj.dims[-2:] != ("y", "x"):
            raise ValueError(f"the last two dims must be ('y', 'x'), got {obj.dims}")
        res = run_snic(np.asarray(obj.values), spacing=spacing,
                       compactness=compactness, seeds=seeds, grid=grid, padding=padding,
                       tile_size=tile_size, random_state=random_state, n_jobs=n_jobs)
        feature_dims = obj.dims[:-2]
        segment = np.arange(len(res.sizes))
        coords = {"segment": segment, "y": obj.coords.get("y"), "x": obj.coords.get("x")}
        coords.update({d: obj.coords[d] for d in feature_dims if d in obj.coords})
        return xr.Dataset(
            {
                "labels": (("y", "x"), res.labels),
                "means": (("segment",) + tuple(feature_dims), res.means),
                "centroid_row": ("segment", res.centroids[:, 0]),
                "centroid_col": ("segment", res.centroids[:, 1]),
                "n_pixels": ("segment", res.sizes),
            },
            coords={k: v for k, v in coords.items() if v is not None},
        )

    def to_zarr_optimized(self, store_path: str, chunk_size: dict = {"y": 512, "x": 512}) -> None:
        """
        Optimizes and saves the DataArray directly to a Zarr store, ideal for cloud storage (S3/GCS) 
        and rapid multidimensional time-series queries.
        """
        # Ensure optimal chunking before saving
        optimized_ds = self._obj.chunk(chunk_size)
        optimized_ds.to_dataset(name="data").to_zarr(store_path, mode="w", consolidated=True)


