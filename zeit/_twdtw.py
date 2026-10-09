import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional
from . import _core

def run_twdtw(
    ts_values: Union[np.ndarray, List[float]], 
    ts_dates: Union[np.ndarray, List[int]], 
    pattern_values: Union[np.ndarray, List[float]], 
    pattern_dates: Union[np.ndarray, List[int]], 
    alpha: float = 0.1, 
    beta: float = 0.05, 
    gamma: float = 50.0,
    max_time_warp: int = 365,
    subsequence_matching: bool = False,
    abort_threshold: float = float('inf'),
    return_path: bool = False
) -> Union[float, Tuple[float, List[Tuple[int, int]]]]:
    """
    Run Time-Weighted Dynamic Time Warping (TWDTW) on a single time series against a pattern.
    Supports multivariate data if ts_values is 2D (Time x Bands).
    """
    if isinstance(ts_values, list):
        ts_values = np.array(ts_values, dtype=np.float64)
    if isinstance(pattern_values, list):
        pattern_values = np.array(pattern_values, dtype=np.float64)
        
    num_bands = 1
    if ts_values.ndim == 2:
        num_bands = ts_values.shape[1]
    
    ts_values_flat = [float(v) for v in ts_values.flatten()]
    pat_values_flat = [float(v) for v in pattern_values.flatten()]

    ts_dates_list = [int(d) for d in (ts_dates.tolist() if isinstance(ts_dates, np.ndarray) else list(ts_dates))]
    pat_dates_list = [int(d) for d in (pattern_dates.tolist() if isinstance(pattern_dates, np.ndarray) else list(pattern_dates))]
    
    params = _core.twdtw.TWDTWParams()
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.max_time_warp = max_time_warp
    params.subsequence_matching = subsequence_matching
    
    res = _core.twdtw.fit_twdtw(
        ts_values_flat, 
        ts_dates_list, 
        pat_values_flat, 
        pat_dates_list, 
        num_bands,
        params, 
        float(abort_threshold),
        return_path
    )
    
    if return_path:
        return res.distance, res.path
    return res.distance


def run_twdtw_batch(
    values_array: np.ndarray, 
    dates_array: np.ndarray, 
    pattern_values: np.ndarray, 
    pattern_dates: np.ndarray, 
    alpha: float = 0.1, 
    beta: float = 0.05, 
    gamma: float = 50.0,
    max_time_warp: int = 365,
    subsequence_matching: bool = False,
    abort_threshold: float = float('inf'),
    n_jobs: int = -1
) -> np.ndarray:
    """
    Run highly optimized TWDTW on a batch of pixels (3D or 4D array) using OpenMP.
    """
    params = _core.twdtw.TWDTWParams()
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.max_time_warp = max_time_warp
    params.subsequence_matching = subsequence_matching
    
    values_array = np.ascontiguousarray(values_array, dtype=np.float64)
    dates_array = np.ascontiguousarray(dates_array, dtype=np.int32)
    pattern_values = np.ascontiguousarray(pattern_values, dtype=np.float64)
    pattern_dates = np.ascontiguousarray(pattern_dates, dtype=np.int32)
    
    result = _core.twdtw.fit_twdtw_batch(
        values_array, 
        dates_array, 
        pattern_values, 
        pattern_dates, 
        params,
        float(abort_threshold),
        n_jobs
    )
    
    return result

def classify_twdtw(
    values_array: np.ndarray, 
    dates_array: np.ndarray, 
    patterns: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    alpha: float = 0.1, 
    beta: float = 0.05, 
    gamma: float = 50.0,
    max_time_warp: int = 365,
    subsequence_matching: bool = False,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Classify a raster cube based on a dictionary of patterns using TWDTW.
    
    Args:
        values_array: 3D [Y, X, T] or 4D [Y, X, T, Bands] array.
        dates_array: 1D array of dates.
        patterns: dict mapping class_name to a tuple (pattern_values, pattern_dates).
                  pattern_values should be 1D or 2D (if multi-band).
        
    Returns:
        classification_map (np.ndarray): 2D integer array containing the index of the winning class.
        distance_map (np.ndarray): 2D float array containing the TWDTW distance of the winning class.
        class_names (List[str]): List of class names, where index matches classification_map values.
    """
    Y, X = values_array.shape[0], values_array.shape[1]
    
    best_dist = np.full((Y, X), np.inf, dtype=np.float64)
    best_class = np.full((Y, X), -1, dtype=np.int32)
    
    class_names = list(patterns.keys())
    
    for class_idx, class_name in enumerate(class_names):
        pat_vals, pat_dates = patterns[class_name]
        
        # Calculate TWDTW for this class, using the current best_dist as the abort_threshold
        # wait, abort_threshold per pixel needs to be passed, but the batch C++ takes a single float.
        # Since we can't pass a 2D array of abort_thresholds easily without modifying C++ again,
        # we pass inf for now, or just the maximum of the current best_dist if we want to be safe, 
        # but max(best_dist) might be inf initially.
        # To truly use per-pixel early abandonment in batch, we'd need to pass a 2D array of thresholds.
        # Since we didn't, we just run standard batch per class.
        
        dist = run_twdtw_batch(
            values_array, dates_array, pat_vals, pat_dates,
            alpha, beta, gamma, max_time_warp, subsequence_matching,
            abort_threshold=float('inf'), n_jobs=n_jobs
        )
        
        mask = dist < best_dist
        best_dist[mask] = dist[mask]
        best_class[mask] = class_idx
        
    return best_class, best_dist, class_names
