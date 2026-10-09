import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import rasterio
from rasterio.windows import Window

from .metrics import extract_events
from typing import Tuple, List, Dict, Any, Union, Optional

def run_landtrendr_array(years: "np.ndarray", raster_stack: "np.ndarray", max_segments: int = 6, pval_threshold: float = 0.05, n_jobs: int = -1,
                          recovery_threshold: float = 0.25, prevent_fast_recovery: bool = True,
                          spike_threshold: float = 0.9, best_model_proportion: float = 0.75,
                          vertex_count_overshoot: int = 3, min_observations_needed: int = 6,
                          no_data_value: float = 0.0, return_rmse: bool = False, modifier: float = 1.0):
    """
    Apply LandTrendr across a 3D numpy array (time_steps, rows, cols) using C++ batch processing with OpenMP.

    If return_rmse is True, also returns a (rows, cols) array of each pixel's
    fit RMSE against every observation -- LT-GEE's per-pixel noise estimate,
    used to compute DSNR (disturbance magnitude / RMSE) in extract_events().

    modifier (float): +1.0 or -1.0, orienting the segmentation's asymmetric heuristics --
        see run_landtrendr's modifier docstring. Use -1.0 for loss (index-drop) detection,
        +1.0 (the default) for gain (index-rise) detection.
    """
    from ._landtrendr import run_landtrendr_batch
    import os as _os

    if n_jobs == -1:
        n_jobs = max(1, (_os.cpu_count() or 4) - 1)

    time_steps, rows, cols = raster_stack.shape
    max_vertices = max_segments + 1

    output = np.zeros((2 * max_vertices, rows, cols), dtype=np.float32)

    # Transpose from (time, row, col) to (row, col, time) for batch function
    values = np.transpose(raster_stack, (1, 2, 0))

    if n_jobs > 0:
        _os.environ['OMP_NUM_THREADS'] = str(n_jobs)

    # Run the C++ batch
    vertices_array, counts_array, rmse_array = run_landtrendr_batch(
        years, values, max_segments, pval_threshold, no_data_value=no_data_value,
        recovery_threshold=recovery_threshold, prevent_fast_recovery=prevent_fast_recovery,
        spike_threshold=spike_threshold, best_model_proportion=best_model_proportion,
        vertex_count_overshoot=vertex_count_overshoot, min_observations_needed=min_observations_needed,
        modifier=modifier, n_jobs=n_jobs
    )

    # vertices_array shape: (rows * cols, max_vertices, 2)
    # Reshape to (rows, cols, max_vertices, 2)
    vertices_array = vertices_array.reshape((rows, cols, max_vertices, 2))
    counts_array = counts_array.reshape((rows, cols))

    for i in range(max_vertices):
        mask = i < counts_array
        # years
        output[i, :, :] = np.where(mask, vertices_array[:, :, i, 0], 0)
        # values
        output[i + max_vertices, :, :] = np.where(mask, vertices_array[:, :, i, 1], 0)

    if return_rmse:
        return output, rmse_array.reshape((rows, cols)).astype(np.float32)
    return output


# ---------------------------------------------------------
# CCDC Raster Engine
# ---------------------------------------------------------
from .ccdc import run_ccdc

def _process_pixel_ccdc(args: Tuple[int, int, np.ndarray, np.ndarray], dates: Union[np.ndarray, List[int]], max_segments: int, conseq_anom: int = 6) -> Tuple[int, int, List[Dict[str, Any]]]:
    row, col, values, qa = args
    
    if np.all(values == 0) or np.all(np.isnan(values)):
        return row, col, []
        
    # Dates where the pixel has NaN in any band have no observation (Fmask 255)
    nan_mask = np.any(np.isnan(values), axis=0)
    qa = np.where(nan_mask, 255, qa)
    values = np.where(nan_mask[None, :], 0.0, values)
    
    try:
        segments = run_ccdc(dates, values, qa, conseq_anom=conseq_anom)
        return row, col, segments
    except Exception:
        return row, col, []

def run_ccdc_array(dates: "np.ndarray", raster_stack: "np.ndarray", qa_stack: "np.ndarray", max_segments: int = 6, n_jobs: int = -1, return_coefs: bool = True, conseq_anom: int = 6, **ccdc_kwargs) -> "np.ndarray":
    """
    Apply CCDC across a 4D numpy array (bands, time, rows, cols) using C++ OpenMP batch processing.

    dates are Python ordinal days, raster_stack surface reflectance x 10000 and
    qa_stack (time, rows, cols) Fmask codes -- see zeit.ccdc.run_ccdc. Extra
    keyword arguments are passed on to zeit.ccdc.run_ccdc_batch.
    Returns (max_segments, 3 + bands * 9, rows, cols): t_start, t_end, t_break,
    then per band rmse and the 8 harmonic coefficients.
    """
    from .ccdc import run_ccdc_batch
    import os as _os
    
    if n_jobs == -1:
        n_jobs = max(1, (_os.cpu_count() or 4) - 1)
        
    num_bands, time_steps, rows, cols = raster_stack.shape
    params_per_segment = 3 + num_bands * 9 if return_coefs else 1
    
    # Transpose raster_stack to [rows, cols, bands, time]
    values = np.transpose(raster_stack, (2, 3, 0, 1))
    
    # Transpose qa_stack from (time, rows, cols) to (rows, cols, time)
    qa = np.transpose(qa_stack, (1, 2, 0))
    
    segments_array, counts_array = run_ccdc_batch(dates, values, qa, max_segments=max_segments,
                                                  return_coefs=return_coefs, conseq_anom=conseq_anom,
                                                  n_jobs=n_jobs, **ccdc_kwargs)
    
    # segments_array shape: (rows * cols, max_segments, params_per_segment)
    # Reshape it to (rows, cols, max_segments, params_per_segment)
    segments_array = segments_array.reshape((rows, cols, max_segments, params_per_segment))
    counts_array = counts_array.reshape((rows, cols))
    
    # Final output shape: (max_segments, params_per_segment, rows, cols)
    output_stack = np.zeros((max_segments, params_per_segment, rows, cols), dtype=np.float32)
    
    for i in range(max_segments):
        mask = i < counts_array
        for p in range(params_per_segment):
            output_stack[i, p, :, :] = np.where(mask, segments_array[:, :, i, p], 0)
            
    return output_stack

def run_ccdc_image(input_path: str, output_dir: str, dates: "np.ndarray", num_bands: int = 6, qa_band_idx: int = -1,
                   max_segments: int = 6, chunk_size: int = 512, n_jobs: int = -1, prefix: str = "ccdc_break", return_coefs: bool = True, conseq_anom: int = 6) -> None:
    """
    High-level function to process a full GeoTIFF stack using CCDC with chunking.
    Assumes the stack is interleaved by date.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    os.makedirs(output_dir, exist_ok=True)
    
    with rasterio.open(input_path) as src:
        total_layers = src.count
        num_dates = total_layers // num_bands
        
        if len(dates) != num_dates:
            raise ValueError(f"Provided {len(dates)} dates, but image has {num_dates} dates.")
            
        profile = src.profile
        height, width = src.height, src.width
        
        print(f"CCDC Image dimensions: {width}x{height} pixels, {num_dates} dates, {num_bands} bands per date")
        
        out_path = os.path.join(output_dir, f"{prefix}_coefs.tif")
        p = profile.copy()
        
        params_per_seg = (3 + num_bands * 9) if return_coefs else 1
        total_bands = max_segments * params_per_seg
        
        p.update(count=total_bands, dtype='float32', nodata=0, driver='GTiff')
        
        dst_breaks = rasterio.open(out_path, 'w', **p)
        
        print(f"Processing CCDC in chunks of {chunk_size}x{chunk_size}...")
        for row in range(0, height, chunk_size):
            for col in range(0, width, chunk_size):
                window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                print(f"  Chunk: Row {row}-{row+window.height}, Col {col}-{col+window.width}")
                
                stack = src.read(window=window)
                stack = stack.reshape((num_dates, num_bands, window.height, window.width))
                stack = np.transpose(stack, (1, 0, 2, 3))
                
                if qa_band_idx >= 0 and qa_band_idx < num_bands:
                    qa_stack = stack[qa_band_idx, :, :, :].astype(int)
                else:
                    qa_stack = np.zeros((num_dates, window.height, window.width), dtype=int)
                    
                breaks_stack = run_ccdc_array(
                    dates=dates, 
                    raster_stack=stack, 
                    qa_stack=qa_stack,
                    max_segments=max_segments, 
                    n_jobs=n_jobs,
                    return_coefs=return_coefs,
                    conseq_anom=conseq_anom
                )
                
                # breaks_stack shape: (max_segments, params_per_seg, height, width)
                # flatten the first two dimensions for writing
                breaks_stack = breaks_stack.reshape((total_bands, window.height, window.width))
                
                dst_breaks.write(breaks_stack.astype('float32'), window=window)
                
        dst_breaks.close()
            
    print(f"Successfully processed CCDC and saved break dates to {output_dir}")

# ---------------------------------------------------------
# Shared chunked-image engine for per-pixel time-series tools (bfast family,
# Mann-Kendall) - these all take a multi-band GeoTIFF where each band is one
# equally-spaced time step, and produce a fixed, named set of per-pixel
# metrics, so a single generic windowed reader/writer covers all of them.
# ---------------------------------------------------------

def _run_timeseries_dask_image(input_path: str, output_path: str, dask_fn, dask_kwargs: Dict[str, Any],
                                metric_names: List[str], chunk_size: int = 512) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    import dask.array as da

    with rasterio.open(input_path) as src:
        n_time = src.count
        height, width = src.height, src.width
        src_nodata = src.nodata

        print(f"Image dimensions: {width}x{height} px, {n_time} time steps")

        profile = src.profile
        p = profile.copy()
        p.update(count=len(metric_names), dtype='float32', nodata=float('nan'), driver='GTiff')

        rows_range = list(range(0, height, chunk_size))
        cols_range = list(range(0, width, chunk_size))
        total_chunks = len(rows_range) * len(cols_range)

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_chunks, desc="Processing", unit="chunk")
        except ImportError:
            pbar = None

        with rasterio.open(output_path, 'w', **p) as dst:
            for row in rows_range:
                for col in cols_range:
                    window = Window(col, row, min(chunk_size, width - col), min(chunk_size, height - row))
                    if pbar is None:
                        print(f"  Chunk: Row {row}-{row + window.height}, Col {col}-{col + window.width}")

                    stack = src.read(window=window).astype(np.float64)
                    if src_nodata is not None:
                        stack[stack == src_nodata] = np.nan

                    arr = da.from_array(stack, chunks=stack.shape)
                    out = dask_fn(arr, **dask_kwargs).compute()

                    dst.write(out.astype('float32'), window=window)
                    if pbar is not None:
                        pbar.update(1)

            if pbar is not None:
                pbar.close()

            for i, name in enumerate(metric_names):
                dst.set_band_description(i + 1, name)

    print(f"Successfully processed and saved to {output_path}")


def run_bfast_monitor_image(input_path: str, output_dir: str, start_time: float, monitor_start_time: float,
                             frequency: int, order: int = 3, h: float = 0.25, period: int = 10,
                             alpha: float = 0.05, min_valid: int = 10, chunk_size: int = 512,
                             n_jobs: int = -1, prefix: str = "bfast_monitor") -> None:
    """
    High-level CLI/scripting entry point: runs bfastmonitor over a full
    multi-band time-series GeoTIFF (one band per equally-spaced observation)
    in chunks, writing a single multi-band output GeoTIFF - band order and
    names match `zeit.bfast.BFM_METRIC_NAMES`.
    """
    from .bfast import run_bfast_monitor_dask, BFM_METRIC_NAMES

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}.tif")
    _run_timeseries_dask_image(
        input_path, out_path, run_bfast_monitor_dask,
        dict(start_time=start_time, monitor_start_time=monitor_start_time, frequency=frequency,
             order=order, h=h, period=period, alpha=alpha, min_valid=min_valid, n_jobs=n_jobs),
        BFM_METRIC_NAMES, chunk_size=chunk_size,
    )


def run_bfast_lite_image(input_path: str, output_dir: str, start_time: float, frequency: int,
                          order: int = 3, h: float = 0.15, max_breaks_output: int = 5,
                          min_valid: int = 20, chunk_size: int = 512, n_jobs: int = -1,
                          prefix: str = "bfast_lite") -> None:
    """
    High-level CLI/scripting entry point: runs bfastlite over a full
    multi-band time-series GeoTIFF in chunks, writing a single multi-band
    output GeoTIFF - band order and names match
    `zeit.bfast.bfl_metric_names(max_breaks_output)`.
    """
    from .bfast import run_bfast_lite_dask, bfl_metric_names

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}.tif")
    _run_timeseries_dask_image(
        input_path, out_path, run_bfast_lite_dask,
        dict(start_time=start_time, frequency=frequency, order=order, h=h,
             max_breaks_output=max_breaks_output, min_valid=min_valid, n_jobs=n_jobs),
        bfl_metric_names(max_breaks_output), chunk_size=chunk_size,
    )


def run_bfast_image(input_path: str, output_dir: str, start_time: float, frequency: int,
                     order: int = 3, h: float = 0.15, max_breaks_trend: int = 5,
                     max_breaks_season: int = 5, max_iter: int = 10, level: float = 0.05,
                     min_valid: int = 20, chunk_size: int = 512, n_jobs: int = -1,
                     prefix: str = "bfast") -> None:
    """
    High-level CLI/scripting entry point: runs the classic iterative
    bfast() over a full multi-band time-series GeoTIFF in chunks, writing a
    single multi-band output GeoTIFF - band order and names match
    `zeit.bfast.bf_metric_names(max_breaks_trend, max_breaks_season)`.
    """
    from .bfast import run_bfast_dask, bf_metric_names

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}.tif")
    _run_timeseries_dask_image(
        input_path, out_path, run_bfast_dask,
        dict(start_time=start_time, frequency=frequency, order=order, h=h,
             max_breaks_trend=max_breaks_trend, max_breaks_season=max_breaks_season,
             max_iter=max_iter, level=level, min_valid=min_valid, n_jobs=n_jobs),
        bf_metric_names(max_breaks_trend, max_breaks_season), chunk_size=chunk_size,
    )


def run_mann_kendall_image(input_path: str, output_dir: str, method: str = "hamed_rao",
                            alpha: float = 0.05, lag: Optional[int] = None, period: int = 1,
                            min_valid: int = 4, chunk_size: int = 512, n_jobs: int = -1,
                            prefix: str = "mann_kendall") -> None:
    """
    High-level CLI/scripting entry point: runs the Mann-Kendall trend test +
    Theil-Sen slope estimator over a full multi-band time-series GeoTIFF (one
    band per observation) in chunks, writing a single multi-band output
    GeoTIFF - band order and names match `zeit.trend.MK_METRIC_NAMES`.
    """
    from .trend import run_mann_kendall_dask, MK_METRIC_NAMES

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{prefix}.tif")
    _run_timeseries_dask_image(
        input_path, out_path, run_mann_kendall_dask,
        dict(method=method, alpha=alpha, lag=lag, period=period, min_valid=min_valid, n_jobs=n_jobs),
        MK_METRIC_NAMES, chunk_size=chunk_size,
    )
