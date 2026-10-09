import numpy as np
import xarray as xr
import dask.array as da
from typing import Optional
from zeit._core.phenology import fit_phenology_batch

# 19 phenology date/derived metrics + R2 + RMSE (goodness-of-fit of the
# fitted curve for that season, see zeit.qc and PhenologyMetrics in
# src/phenology.h).
N_METRICS = 21

def run_phenology_dask(
    arr: da.Array,
    dates: np.ndarray,
    curve_type: int,
    extraction_method: int = 0,
    max_seasons: int = 2,
    whittaker_lambda: float = 10.0,
    apply_whittaker: bool = True,
    apply_hants: bool = False,
    hants_frequencies: int = 3,
    hants_threshold: float = 0.1,
    min_season_length: int = 0,
    min_amplitude: float = 0.0,
    min_pixel_amplitude: float = 0.1,
    return_annual: bool = True,
    base_year: int = 2001,
    n_jobs: int = -1,
    weights: Optional[da.Array] = None,
    season_retry: bool = True
) -> da.Array:
    """
    Applies fit_phenology_batch across a Dask array.
    Input array shape: (time, y, x).
    Output array shape: (21, max_seasons, y, x) - 19 phenology metrics plus
    a per-season R2 and RMSE of the fitted curve (indices 19 and 20).

    weights: optional (time, y, x) Dask array of per-observation reliability
    weights in [0, 1] (e.g. from zeit.qc), aligned with `arr`. Feeds the
    Whittaker/HANTS smoothers and seeds the iterative curve-fit reweighting
    instead of treating every observation as equally trustworthy.
    """
    def _phenology_block(block, weights_block=None):
        if block.size == 0:
            return np.zeros((N_METRICS, max_seasons, block.shape[1], block.shape[2]), dtype=np.float32)

        # block is (time, y, x)
        time_steps, rows, cols = block.shape
        pixels = rows * cols

        # reshape to (pixels, time) and ensure C-contiguous
        values_2d = np.ascontiguousarray(block.reshape(time_steps, pixels).T)

        weights_2d = None
        if weights_block is not None:
            weights_2d = np.ascontiguousarray(
                weights_block.reshape(time_steps, pixels).T.astype(np.float64)
            )

        # run batch fitting
        out_3d = fit_phenology_batch(
            values_array=values_2d,
            dates_array=dates,
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
            n_jobs=n_jobs,
            weights_array=weights_2d,
            season_retry=season_retry
        )

        # out_3d is (N_METRICS, pixels, max_seasons)
        # We need (N_METRICS, max_seasons, rows, cols)
        # Transpose to (N_METRICS, max_seasons, pixels) and then reshape
        out_transposed = out_3d.transpose(0, 2, 1)
        out = out_transposed.reshape(N_METRICS, max_seasons, rows, cols)
        
        if return_annual:
            out_annual = np.full_like(out, np.nan)

            # `val` encodes a date as "days since `base_year`-01-01, 1-indexed"
            # (i.e. datetime(base_year, 1, 1) + timedelta(days=val - 1)), matching
            # the C++ core's day numbering. Vectorized with numpy datetime64 instead
            # of a pure-Python per-pixel loop, since this runs per Dask block and
            # scales with total pixel count across the whole (potentially global) cube.
            # Metrics 19 (R2) and 20 (RMSE) aren't dates - excluded here and
            # remapped separately below, keyed off each season's POP date.
            # LOS (17) is a duration and R2/RMSE (19, 20) are fit scores: not dates, so
            # they are placed in the year of their season's POP below instead.
            metric_ids = np.arange(N_METRICS)[:, None, None, None]
            is_date_metric = (metric_ids < 19) & (metric_ids != 17)
            valid = np.isfinite(out) & (out > 0) & is_date_metric
            if np.any(valid):
                m_idx, s_idx, r_idx, c_idx = np.nonzero(valid)
                vals = out[valid]

                epoch = np.datetime64(f"{base_year}-01-01", "D")
                # timedelta(days=val - 1) only ever carries a sub-day remainder,
                # so the calendar date depends solely on floor(val - 1) days.
                day_offset = np.floor(vals).astype(np.int64) - 1
                event_dates = epoch + day_offset.astype("timedelta64[D]")

                years = event_dates.astype("datetime64[Y]").astype(np.int64) + 1970
                year_idx = years - base_year

                year_start = event_dates.astype("datetime64[Y]")
                doy = (event_dates - year_start).astype(np.int64) + 1  # 1-based day-of-year

                in_range = (year_idx >= 0) & (year_idx < max_seasons)
                m_idx, r_idx, c_idx = m_idx[in_range], r_idx[in_range], c_idx[in_range]
                year_idx = year_idx[in_range]
                doy = doy[in_range]
                vals = vals[in_range]

                store_val = doy.astype(np.float64)

                # For a given (m, r, c), later seasons must win ties on the same
                # calendar year — matches the original loop's `for s in range(...)` order,
                # which `np.nonzero` preserves (C-order over (m, s, r, c)).
                out_annual[m_idx, year_idx, r_idx, c_idx] = store_val

            # LOS (17) and R2/RMSE (19, 20): not dates themselves, so they ride along with
            # the calendar year their season's POP (always populated for any
            # converged fit) falls into, instead of being decoded as dates.
            valid_gof = np.isfinite(out[18]) & (out[18] > 0) & (
                np.isfinite(out[17]) | np.isfinite(out[19]) | np.isfinite(out[20]))
            if np.any(valid_gof):
                s_idx, r_idx, c_idx = np.nonzero(valid_gof)
                epoch = np.datetime64(f"{base_year}-01-01", "D")
                day_offset = np.floor(out[18][valid_gof]).astype(np.int64) - 1
                event_dates = epoch + day_offset.astype("timedelta64[D]")
                years = event_dates.astype("datetime64[Y]").astype(np.int64) + 1970
                year_idx = years - base_year

                in_range = (year_idx >= 0) & (year_idx < max_seasons)
                year_idx, r_idx, c_idx = year_idx[in_range], r_idx[in_range], c_idx[in_range]
                for gof_idx in (17, 19, 20):
                    out_annual[gof_idx, year_idx, r_idx, c_idx] = out[gof_idx][valid_gof][in_range]

            out = out_annual

        return out.astype(np.float32)

    map_blocks_kwargs = dict(
        dtype=np.float32,
        drop_axis=[0], # remove time
        new_axis=[0, 1], # add metrics (N_METRICS) and max_seasons
        chunks=(N_METRICS, max_seasons, arr.chunks[1], arr.chunks[2])
    )

    if weights is not None:
        if not isinstance(weights, da.Array):
            weights = da.from_array(weights, chunks=arr.chunks)
        out = da.map_blocks(_phenology_block, arr, weights, **map_blocks_kwargs)
    else:
        out = da.map_blocks(_phenology_block, arr, **map_blocks_kwargs)
    return out
