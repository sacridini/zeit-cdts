"""Low-level numpy engines behind zeit.landtrendr and zeit.ccdc."""
import numpy as np

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
def run_ccdc_array(dates: "np.ndarray", raster_stack: "np.ndarray", qa_stack: "np.ndarray", max_segments: int = 6, n_jobs: int = -1, return_coefs: bool = True, conseq_anom: int = 6, **ccdc_kwargs) -> "np.ndarray":
    """
    Apply CCDC across a 4D numpy array (bands, time, rows, cols) using C++ OpenMP batch processing.

    dates are Python ordinal days, raster_stack surface reflectance x 10000 and
    qa_stack (time, rows, cols) Fmask codes -- see zeit._ccdc.run_ccdc. Extra
    keyword arguments are passed on to zeit._ccdc.run_ccdc_batch.
    Returns (max_segments, 3 + bands * 9, rows, cols): t_start, t_end, t_break,
    then per band rmse and the 8 harmonic coefficients.
    """
    from ._ccdc import run_ccdc_batch
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
