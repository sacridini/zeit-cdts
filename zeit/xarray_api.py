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

    def run_phenology(self, dates: np.ndarray, curve_type: int, extraction_method: int = 0, max_seasons: int = 2, whittaker_lambda: float = 10.0,
apply_whittaker: bool = True, apply_hants: bool = False, hants_frequencies: int = 3, hants_threshold: float = 0.1, min_season_length: int = 0,
min_amplitude: float = 0.0, min_pixel_amplitude: float = 0.1, return_annual: bool = True, base_year: int = 2001, n_jobs: int = -1,
weights: Optional[Any] = None, season_retry: bool = True) -> xr.DataArray:
        """
        Runs Phenology extraction on an xarray DataArray using Dask.
        Assumes DataArray shape: (time, y, x).
        Returns an xarray DataArray with shape (metric, max_seasons, y, x)
        where metric is 0: SOS, 1: EOS, 2: LOS, 3: POP.

        weights: optional (time, y, x) array/DataArray of per-observation
        reliability weights in [0, 1], aligned with this DataArray (e.g. built
        with zeit.qc.qc_modis_summary/qc_modis_state/qc_sentinel2_scl from a
        QA band). Low-quality observations are down-weighted in the Whittaker/
        HANTS smoothing and in the iterative curve fit instead of being
        treated as equally trustworthy as clear observations.

        season_retry: when a pixel's first-pass season detection finds no
        growing season at all, retry once with a relaxed trough threshold
        (mirrors phenofit's season_mov r_max relaxation) before giving up on
        that pixel. Set False to disable for stricter/faster behaviour.
        """
        from zeit.phenology import run_phenology_dask

        arr = self._obj.data
        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        weights_arr = None
        if weights is not None:
            weights_arr = weights.data if isinstance(weights, xr.DataArray) else weights

        out = run_phenology_dask(
            arr=arr,
            dates=dates,
            curve_type=curve_type,
            extraction_method=extraction_method,
            max_seasons=max_seasons,
            whittaker_lambda=whittaker_lambda,
            apply_whittaker=apply_whittaker,
            apply_hants=apply_hants,
            hants_frequencies=hants_frequencies,
            hants_threshold=hants_threshold,
            min_season_length=min_season_length,
            min_amplitude=min_amplitude,
            min_pixel_amplitude=min_pixel_amplitude,
            return_annual=return_annual,
            base_year=base_year,
            n_jobs=n_jobs,
            weights=weights_arr,
            season_retry=season_retry
        )

        metrics = [
            "TRS2.sos", "TRS2.eos", "TRS5.sos", "TRS5.eos", "TRS6.sos", "TRS6.eos",
            "DER.sos", "DER.pos", "DER.eos",
            "UD", "SD", "DD", "RD",
            "Greenup", "Maturity", "Senescence", "Dormancy",
            "LOS", "POP",
            "R2", "RMSE"
        ]
        
        dim_name = "year" if return_annual else "season"
        years = np.arange(base_year, base_year + max_seasons)
        
        return xr.DataArray(
            out,
            dims=["metric", dim_name, "y", "x"],
            coords={
                "metric": metrics,
                dim_name: years if return_annual else np.arange(max_seasons),
                "y": self._obj.coords["y"],
                "x": self._obj.coords["x"],
            }
        )

    def run_mann_kendall(self, method: str = "hamed_rao", alpha: float = 0.05,
                          lag: Optional[int] = None, period: int = 1,
                          min_valid: int = 4, n_jobs: int = -1) -> xr.DataArray:
        """
        Pixel-wise Mann-Kendall trend test + Theil-Sen slope across the time
        dimension. Assumes DataArray shape: (time, y, x).
        Returns a DataArray with dim "metric": trend, h, p, z, tau, s,
        var_s, slope, intercept (see zeit.trend.MK_METRIC_NAMES).

        method: "original", "hamed_rao" (default - autocorrelation-corrected,
        recommended for annual composites), "yue_wang" (alternative
        autocorrelation correction), or "seasonal" (pools per-season MK
        scores over `period` slots, e.g. period=23 for MODIS 16-day annual
        cycles - lets you test a raw sub-annual series directly).

        slope/intercept are in units of this DataArray per TIME STEP, not
        calendar time (per full `period` cycle for method="seasonal") - use
        one observation per year, or `period=`, for a per-year trend.
        """
        from zeit.trend import run_mann_kendall_dask, MK_METRIC_NAMES

        arr = self._obj.data
        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        out = run_mann_kendall_dask(
            arr, method=method, alpha=alpha, lag=lag, period=period,
            min_valid=min_valid, n_jobs=n_jobs,
        )

        return xr.DataArray(
            out,
            dims=["metric", "y", "x"],
            coords={
                "metric": MK_METRIC_NAMES,
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x"),
            }
        )

    def run_bfast_monitor(self, start_time: float, monitor_start_time: float, frequency: int,
                           order: int = 3, h: float = 0.25, period: int = 10,
                           alpha: float = 0.05, min_valid: int = 10, n_jobs: int = -1) -> xr.DataArray:
        """
        Near-real-time structural change monitoring (bfastmonitor) across
        the time dimension. Assumes DataArray shape: (time, y, x).
        Returns a DataArray with dim "metric": breakpoint, breakpoint_idx,
        magnitude, sigma, n_history, has_break, valid (see
        zeit.bfast.BFM_METRIC_NAMES).

        See zeit.bfast.run_bfast_monitor_dask for the full parameter
        documentation and scope notes (only type="OLS-MOSUM" and
        history="all" are implemented).
        """
        from zeit.bfast import run_bfast_monitor_dask, BFM_METRIC_NAMES

        arr = self._obj.data
        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        out = run_bfast_monitor_dask(
            arr, start_time=start_time, monitor_start_time=monitor_start_time,
            frequency=frequency, order=order, h=h, period=period, alpha=alpha,
            min_valid=min_valid, n_jobs=n_jobs,
        )

        return xr.DataArray(
            out,
            dims=["metric", "y", "x"],
            coords={
                "metric": BFM_METRIC_NAMES,
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x"),
            }
        )

    def run_bfast_lite(self, start_time: float, frequency: int, order: int = 3, h: float = 0.15,
                        max_breaks_output: int = 5, min_valid: int = 20, n_jobs: int = -1) -> xr.DataArray:
        """
        Single-pass multiple-breakpoint detection (bfastlite) across the
        time dimension. Assumes DataArray shape: (time, y, x). Returns a
        DataArray with dim "metric": n_breaks, rss, lwz, n_valid, valid,
        breakpoint_idx_1..breakpoint_idx_{max_breaks_output} (see
        zeit.bfast.bfl_metric_names).

        See zeit.bfast.run_bfast_lite_dask for the full parameter
        documentation and scope notes (no STL decomposition; see
        run_bfast for the classic iterative bfast()).
        """
        from zeit.bfast import run_bfast_lite_dask, bfl_metric_names

        arr = self._obj.data
        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        out = run_bfast_lite_dask(
            arr, start_time=start_time, frequency=frequency, order=order, h=h,
            max_breaks_output=max_breaks_output, min_valid=min_valid, n_jobs=n_jobs,
        )

        return xr.DataArray(
            out,
            dims=["metric", "y", "x"],
            coords={
                "metric": bfl_metric_names(max_breaks_output),
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x"),
            }
        )

    def run_bfast(self, start_time: float, frequency: int, order: int = 3, h: float = 0.15,
                  max_breaks_trend: int = 5, max_breaks_season: int = 5, max_iter: int = 10,
                  level: float = 0.05, min_valid: int = 20, n_jobs: int = -1) -> xr.DataArray:
        """
        Classic iterative trend+season break detection (bfast) across the
        time dimension. Assumes DataArray shape: (time, y, x). Returns a
        DataArray with dim "metric": n_trend_breaks, n_season_breaks,
        magnitude, time, n_iter, n_valid, valid,
        trend_breakpoint_idx_1..max_breaks_trend,
        season_breakpoint_idx_1..max_breaks_season (see
        zeit.bfast.bf_metric_names).

        See zeit.bfast.run_bfast_dask for the full parameter documentation
        and scope notes (season="harmonic" and breaks="BIC" only,
        decomp="stl" only).
        """
        from zeit.bfast import run_bfast_dask, bf_metric_names

        arr = self._obj.data
        if not isinstance(arr, da.Array):
            arr = da.from_array(arr)

        out = run_bfast_dask(
            arr, start_time=start_time, frequency=frequency, order=order, h=h,
            max_breaks_trend=max_breaks_trend, max_breaks_season=max_breaks_season,
            max_iter=max_iter, level=level, min_valid=min_valid, n_jobs=n_jobs,
        )

        return xr.DataArray(
            out,
            dims=["metric", "y", "x"],
            coords={
                "metric": bf_metric_names(max_breaks_trend, max_breaks_season),
                "y": self._obj.coords.get("y"),
                "x": self._obj.coords.get("x"),
            }
        )

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


