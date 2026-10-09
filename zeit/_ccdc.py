import numpy as np
from typing import List, Dict, Any, Union, Optional
from . import _core

# The original CCDC parameterises time as MATLAB datenum, which is the Python
# ordinal day + 366; the harmonic coefficients refer to that time axis.
DATENUM_OFFSET = 366


def _params(conseq_anom, chi2_prob_threshold, tmax_cg_prob_threshold, detection_bands, num_c,
            tmask_bands, thermal_band, valid_range, thermal_range, min_obs=12):
    params = _core.ccdc.CCDCParams()
    params.min_obs = min_obs
    params.conseq_anom = conseq_anom
    params.chi2_prob_threshold = chi2_prob_threshold
    params.tmax_cg_prob_threshold = tmax_cg_prob_threshold
    params.num_c = num_c
    if detection_bands is not None:
        params.detection_bands = list(detection_bands)
    if tmask_bands is not None:
        params.tmask_bands = list(tmask_bands)
    params.thermal_band = -1 if thermal_band is None else int(thermal_band)
    params.valid_min, params.valid_max = valid_range
    params.thermal_min, params.thermal_max = thermal_range
    return params


def run_ccdc(
    dates: Union[np.ndarray, List[float]],
    values: Union[np.ndarray, List[List[float]]],
    qa: Union[np.ndarray, List[int]],
    min_obs: int = 12,
    conseq_anom: int = 6,
    chi2_prob_threshold: float = 0.99,
    tmax_cg_prob_threshold: float = 0.999999,
    detection_bands: Optional[List[int]] = None,
    num_c: int = 8,
    tmask_bands: Optional[List[int]] = None,
    thermal_band: Optional[int] = None,
    valid_range: tuple = (0.0, 10000.0),
    thermal_range: tuple = (-9320.0, 7070.0),
) -> List[Dict[str, Any]]:
    """
    Continuous Change Detection and Classification (CCDC, Zhu & Woodcock 2014).

    A port of the original MATLAB implementation (TrendSeasonalFit_v12_30Line.m,
    GERSL/CCDC), validated segment-for-segment against it. Like the original it
    expects Landsat-style inputs:

    Args:
        dates: Python ordinal days (``datetime.date.toordinal()``), one per observation.
        values: (n_bands, n_dates) surface reflectance x 10000, bands ordered
            Blue, Green, Red, NIR, SWIR1, SWIR2 [, brightness temperature].
            The original's lasso (lambda = 20), range test and cloud screen are
            defined on this scale.
        qa: Fmask codes per date: 0 clear land, 1 water, 2 cloud shadow, 3 snow,
            4 cloud, 255 no observation. 0 and 1 are both usable ("clear").
        min_obs: kept for API compatibility; the original fixes this at 12.
        conseq_anom: consecutive anomalous observations to flag a change (``conse``).
        chi2_prob_threshold: change probability; T_cg = chi2inv(p, len(detection_bands)).
        tmax_cg_prob_threshold: outlier probability; Tmax_cg = chi2inv(p, ...).
        detection_bands: 0-based bands used for change detection (default: Green..SWIR2,
            i.e. [1, 2, 3, 4, 5], when there are >= 6 bands; all bands otherwise).
        num_c: maximum number of harmonic coefficients (4, 6 or 8).
        tmask_bands: the two 0-based bands used by Tmask cloud screening
            (default [1, 4] = Green, SWIR1).
        thermal_band: index of a brightness-temperature band (deg C x 100), if any;
            it gets the thermal range test instead of valid_range.
        valid_range: exclusive valid range of the optical bands.
        thermal_range: exclusive valid range of the thermal band.

    Returns:
        One dict per time-series model: t_start, t_end, t_break (ordinal days,
        t_break 0 if no change), coefs (n_bands x 8: a0, c1, a1, b1, a2, b2, a3, b3
        on the datenum time axis, see predict), rmse, magnitude, change_prob,
        category and num_obs, with the original's meanings.
    """
    dates_list = [int(d) for d in (dates.tolist() if isinstance(dates, np.ndarray) else list(dates))]
    params = _params(conseq_anom, chi2_prob_threshold, tmax_cg_prob_threshold, detection_bands, num_c,
                     tmask_bands, thermal_band, valid_range, thermal_range, min_obs)

    # Ensure values is 2D: (num_bands, num_dates)
    if isinstance(values, np.ndarray):
        values_list = [values.tolist()] if values.ndim == 1 else values.tolist()
    elif len(values) > 0 and not isinstance(values[0], (list, tuple, np.ndarray)):
        values_list = [list(values)]
    else:
        values_list = [list(v) for v in values]
    qa_list = [int(q) for q in (qa.tolist() if isinstance(qa, np.ndarray) else list(qa))]

    segments = _core.ccdc.fit_ccdc(dates_list, values_list, qa_list, params)

    return [
        {
            "t_start": s.t_start,
            "t_end": s.t_end,
            "t_break": s.t_break,
            "coefs": s.coefs,
            "rmse": s.rmse,
            "magnitude": s.magnitude,
            "change_prob": s.change_prob,
            "category": s.category,
            "num_obs": s.num_obs,
        } for s in segments
    ]


def predict(coefs: Union[np.ndarray, List[float]], dates: Union[np.ndarray, List[int], int]) -> np.ndarray:
    """Evaluates one band's 8 CCDC coefficients at Python ordinal day(s) (the original's autoTSPred)."""
    c = np.zeros(8)
    cc = np.asarray(coefs, dtype=float)
    c[:len(cc)] = cc
    t = np.asarray(dates, dtype=float) + DATENUM_OFFSET
    w = 2.0 * np.pi / 365.25
    return (c[0] + c[1] * t + c[2] * np.cos(w * t) + c[3] * np.sin(w * t)
            + c[4] * np.cos(2 * w * t) + c[5] * np.sin(2 * w * t)
            + c[6] * np.cos(3 * w * t) + c[7] * np.sin(3 * w * t))


def predict_synthetic_image(segments: Any, date: Any = None, num_bands: int = 6, *,
                            target_julian_day: Optional[int] = None) -> Any:
    """
    Generates a cloud-free synthetic image for a date from CCDC harmonic coefficients.

    segments: the result of ``zeit.ccdc`` (an ``xarray.Dataset``); returns a georeferenced
        (band, y, x) DataArray. Each pixel uses the segment covering the date (before the first
        segment, the first; after the last, the last); pixels without a model are NaN.
        A numpy stack shaped (max_segments, 3 + num_bands * 9, rows, cols), as written by older
        versions, is also accepted and gives a (num_bands, rows, cols) array.
    date: a date (string, datetime, ``numpy.datetime64``) or a Python ordinal day.
        ``target_julian_day`` is its former name.
    num_bands: numpy stacks only.
    """
    if date is None:
        date = target_julian_day
    if date is None:
        raise ValueError("give the date to predict")
    import xarray as xr
    if isinstance(segments, xr.Dataset):
        from ._ccdc_api import predict_image
        return predict_image(segments, date)
    ccdc_coefs_stack = np.asarray(segments)
    target_julian_day = int(date)
    _, _, rows, cols = ccdc_coefs_stack.shape

    W = 2.0 * np.pi / 365.25
    t = float(target_julian_day) + DATENUM_OFFSET
    terms = np.array([
        1.0, t, np.cos(W * t), np.sin(W * t), np.cos(2.0 * W * t), np.sin(2.0 * W * t),
        np.cos(3.0 * W * t), np.sin(3.0 * W * t)
    ])

    synthetic_image = np.zeros((num_bands, rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            best_seg = 0
            for i in range(ccdc_coefs_stack.shape[0]):
                t_start = ccdc_coefs_stack[i, 0, r, c]
                t_end = ccdc_coefs_stack[i, 1, r, c]

                if t_start <= target_julian_day <= t_end:
                    best_seg = i
                    break

            idx = 3
            for b in range(num_bands):
                idx += 1  # skip RMSE
                coefs = ccdc_coefs_stack[best_seg, idx:idx + 8, r, c]
                idx += 8
                synthetic_image[b, r, c] = np.dot(coefs, terms)

    return synthetic_image


def run_ccdc_batch(dates: np.ndarray, values: np.ndarray, qa: np.ndarray, max_segments: int = 6,
                   return_coefs: bool = True, conseq_anom: int = 6, tmax_cg_prob_threshold: float = 0.999999,
                   detection_bands: List[int] = None, n_jobs: int = -1, chi2_prob_threshold: float = 0.99,
                   num_c: int = 8, tmask_bands: Optional[List[int]] = None, thermal_band: Optional[int] = None,
                   valid_range: tuple = (0.0, 10000.0), thermal_range: tuple = (-9320.0, 7070.0)):
    """
    Run CCDC algorithm on a batch of pixels (see run_ccdc for the input conventions).

    Args:
        dates (np.ndarray): 1D array of Python ordinal days [Time].
        values (np.ndarray): 4D array of surface reflectance x 10000 [Y, X, Bands, Time].
            NaN marks a date without an observation.
        qa (np.ndarray): 3D array of Fmask codes [Y, X, Time].
        max_segments (int): Maximum number of segments to return per pixel.
        return_coefs (bool): Whether to return harmonic coefficients.
        n_jobs (int): Number of threads for OpenMP to use. Default -1 (all but one).
        Other arguments: as in run_ccdc.

    Returns:
        tuple of (segments_array, counts_array). segments_array is
        [Y*X, max_segments, 3 + Bands * 9] = t_start, t_end, t_break, then per
        band rmse and the 8 coefficients (or [.., 1] = t_break if not return_coefs).
    """
    params = _params(conseq_anom, chi2_prob_threshold, tmax_cg_prob_threshold, detection_bands, num_c,
                     tmask_bands, thermal_band, valid_range, thermal_range)

    dates = np.ascontiguousarray(dates, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float64)
    qa = np.ascontiguousarray(qa, dtype=np.int32)

    return _core.ccdc.fit_ccdc_batch(values, qa, dates, params, max_segments, return_coefs, n_jobs)
