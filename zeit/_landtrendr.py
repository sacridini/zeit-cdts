import numpy as np
from typing import List, Dict, Union
from . import _core

def desawtooth(values: Union[np.ndarray, List[float]], stopat: float = 0.9) -> np.ndarray:
    """
    Remove spikes from a time series using LandTrendr's desawtooth algorithm.
    """
    values_list = values.tolist() if isinstance(values, np.ndarray) else list(values)
    filtered_list = _core.landtrendr.desawtooth(values_list, stopat)
    return np.array(filtered_list)

def run_landtrendr(years: Union[np.ndarray, List[int]], values: Union[np.ndarray, List[float]], max_segments: int = 6, pval_threshold: float = 0.05,
                    recovery_threshold: float = 0.25, prevent_fast_recovery: bool = True,
                    spike_threshold: float = 0.9, best_model_proportion: float = 0.75,
                    vertex_count_overshoot: int = 3, min_observations_needed: int = 6,
                    modifier: float = 1.0) -> List[Dict[str, Union[int, float]]]:
    """
    Run LandTrendr algorithm on a 1D time series of a single pixel.

    A port of the original LandTrendr-2012 IDL segmentation (fit_trajectory_v2 /
    tbcd_v2, Kennedy et al. 2010), validated vertex-for-vertex against that code.

    Args:
        years (np.ndarray): 1D array of years.
        values (np.ndarray): 1D array of spectral values. NaN marks a missing year: it is
            left out of the fit, and a missing first/last year gets a flat edge vertex,
            as in the original.
        max_segments (int): Maximum number of segments to fit.
        pval_threshold (float): P-value threshold for segment significance.
        recovery_threshold (float): Max allowed recovery rate per year.
        prevent_fast_recovery (bool): Kept for API compatibility; has no effect. As in the
            original, the recovery_threshold check is always applied (pass a large
            recovery_threshold to effectively disable it).
        spike_threshold (float): Desawtooth dampening factor (1.0 = no dampening). LT-GEE's spikeThreshold.
        best_model_proportion (float): Prefer the most-vertex candidate model whose p-value is at
            most (2 - best_model_proportion) times the lowest p-value found among candidates, as in the
            original's pick_best_model6. The default 0.75 accepts models within 1.25x of the best p;
            values above 1 make the threshold stricter than the best model itself, so selection falls
            through to a flat line unless several models tie at p = 0.
        vertex_count_overshoot (int): Extra vertices allowed in the initial candidate pool beyond
            max_segments + 1, pruned back down before model selection. LT-GEE's vertexCountOvershoot.
        min_observations_needed (int): Below this many (valid) observations, skip fitting entirely and
            return the raw trajectory unsegmented. LT-GEE's minObservationsNeeded.
        modifier (float): +1.0 or -1.0. The segmentation's asymmetric heuristics (trailing-edge
            recovery suppression, the recovery-rate eligibility check) are only meaningful once the
            series is oriented so an INCREASE always reads as the disturbance-like event being
            searched for. Use -1.0 to detect index drops (e.g. vegetation loss on an NDVI-like
            index) and +1.0 (the default) to detect index rises (e.g. vegetation gain). Output
            vertex values are always in the original, unflipped scale regardless of modifier.

    Returns:
        list of dicts containing the fitted vertices (year, value).
    """
    params = _core.landtrendr.LandTrendrParams()
    params.max_segments = max_segments
    params.pval_threshold = pval_threshold
    params.recovery_threshold = recovery_threshold
    params.prevent_fast_recovery = prevent_fast_recovery
    params.spike_threshold = spike_threshold
    params.best_model_proportion = best_model_proportion
    params.vertex_count_overshoot = vertex_count_overshoot
    params.min_observations_needed = min_observations_needed
    params.modifier = modifier

    # Ensure lists for C++ vector binding (or we could use pybind11::array in C++ directly for zero-copy)
    years_list = years.tolist() if isinstance(years, np.ndarray) else list(years)
    values_list = values.tolist() if isinstance(values, np.ndarray) else list(values)

    vertices = _core.landtrendr.fit_trajectory(years_list, values_list, params)

    return [{"year": v.year, "value": v.value} for v in vertices]

def run_landtrendr_batch(years: np.ndarray, values: np.ndarray, max_segments: int = 6, pval_threshold: float = 0.05, no_data_value: float = -9999.0,
                          recovery_threshold: float = 0.25, prevent_fast_recovery: bool = True,
                          spike_threshold: float = 0.9, best_model_proportion: float = 0.75,
                          vertex_count_overshoot: int = 3, min_observations_needed: int = 6,
                          modifier: float = 1.0, n_jobs: int = -1):
    """
    Run LandTrendr algorithm on a batch of pixels.

    Args:
        years (np.ndarray): 1D array of years [Time].
        values (np.ndarray): 3D array of spectral values [Y, X, Time].
        max_segments (int): Maximum number of segments to fit.
        pval_threshold (float): P-value threshold for segment significance.
        no_data_value (float): No data value in the array. Years holding it (or NaN) are left
            out of that pixel's fit, like NaN in run_landtrendr; all-no-data pixels are skipped.
        recovery_threshold (float): Max allowed recovery rate per year.
        prevent_fast_recovery (bool): Kept for API compatibility; has no effect (see run_landtrendr).
        spike_threshold (float): Desawtooth dampening factor (1.0 = no dampening). LT-GEE's spikeThreshold.
        best_model_proportion (float): Prefer the most-vertex candidate model whose p-value is at
            most (2 - best_model_proportion) times the lowest p-value found among candidates, as in the
            original's pick_best_model6. The default 0.75 accepts models within 1.25x of the best p;
            values above 1 make the threshold stricter than the best model itself, so selection falls
            through to a flat line unless several models tie at p = 0.
        vertex_count_overshoot (int): Extra vertices allowed in the initial candidate pool beyond
            max_segments + 1, pruned back down before model selection. LT-GEE's vertexCountOvershoot.
        min_observations_needed (int): Below this many (valid) observations, skip fitting entirely and
            return the raw trajectory unsegmented. LT-GEE's minObservationsNeeded.
        modifier (float): +1.0 or -1.0, orienting the segmentation's asymmetric heuristics --
            see run_landtrendr's modifier docstring. Use -1.0 for loss (index-drop) detection,
            +1.0 (the default) for gain (index-rise) detection.

    Returns:
        tuple of (vertices_array, counts_array, rmse_array)
        vertices_array: [Y*X, max_segments+1, 2] containing (year, value) for each vertex
        counts_array: [Y*X] containing the number of valid vertices found for each pixel
        rmse_array: [Y*X] RMSE of the selected fit against every observation for each pixel
            (0.0 where fitting was skipped/no-data). LT-GEE's DSNR = magnitude / this.
    """
    params = _core.landtrendr.LandTrendrParams()
    params.max_segments = max_segments
    params.pval_threshold = pval_threshold
    params.recovery_threshold = recovery_threshold
    params.prevent_fast_recovery = prevent_fast_recovery
    params.spike_threshold = spike_threshold
    params.best_model_proportion = best_model_proportion
    params.vertex_count_overshoot = vertex_count_overshoot
    params.min_observations_needed = min_observations_needed
    params.modifier = modifier

    import os as _os
    if n_jobs <= 0:
        n_jobs = max(1, _os.cpu_count() or 1)

    years = np.ascontiguousarray(years, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float64)

    return _core.landtrendr.fit_trajectory_batch(values, years, params, no_data_value, n_jobs)

def apply_vertices(vertex_years: Union[np.ndarray, List[int]], other_band_years: Union[np.ndarray, List[int]], other_band_values: Union[np.ndarray, List[float]]) -> List[Dict[str, Union[int, float]]]:
    """
    Applies LandTrendr structural vertices (FTV - Fitted to Vertices) to another spectral band.
    This effectively uses the segmentation derived from the primary index to smooth and fit the secondary index.
    """
    import numpy as np
    
    if len(vertex_years) == 0 or len(other_band_years) == 0:
        return []
        
    # Find the corresponding values in the other band for the vertex years
    # If a vertex year is missing in the other band, we interpolate it.
    fitted_values = np.interp(vertex_years, other_band_years, other_band_values)
    
    return [{"year": int(y), "value": float(v)} for y, v in zip(vertex_years, fitted_values)]
