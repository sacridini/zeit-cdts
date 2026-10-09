"""The Tmask engine (one pixel, and a numpy stack), in C++ (``_core.tmask``);
``zeit.tmask`` is in ``_tmask_api``.

Tmask (Zhu & Woodcock 2014): a robust fit of a0 + c1 t + a1 cos(wt) + b1 sin(wt) to the
green and SWIR series of a pixel (MATLAB's robustfit, bisquare, as CCDC's autoTmask runs
it); an observation whose green rises above the fit by more than 0.04 (cloud) or whose SWIR
drops below it by more than 0.04 (shadow), in reflectance, is flagged. Before 0.43 the fit
was scikit-learn's HuberRegressor, about 190 times slower per core.
"""
import numpy as np

from . import _core


def run_tmask_pixel(dates_julian: np.ndarray, green_band: np.ndarray, swir_band: np.ndarray,
                    scale_factor: float = 10000.0) -> np.ndarray:
    """
    Tmask of a single pixel's observations.

    dates_julian: dates in days (Python ordinal days, or any day count).
    green_band, swir_band: the green and SWIR (SWIR1, ~1.6 um) reflectance.
    scale_factor: the reflectance scale (10000 for reflectance x 10000).

    Returns a boolean array, True = clear, False = cloud or shadow. With fewer than 5
    observations nothing is flagged.
    """
    clear = _core.tmask.tmask_pixel(np.asarray(dates_julian, dtype=np.float64).tolist(),
                                    np.asarray(green_band, dtype=np.float64).tolist(),
                                    np.asarray(swir_band, dtype=np.float64).tolist(), float(scale_factor))
    return np.asarray(clear, dtype=bool)


def apply_tmask_stack(dates: np.ndarray, green_stack: np.ndarray, swir_stack: np.ndarray,
                      scale_factor: float = 10000.0) -> np.ndarray:
    """
    Tmask of a (time, y, x) stack. Observations not above 0 are left out of the fit and
    reported clear, as before (``zeit.tmask`` reports them not clear); pixels with 5 or
    fewer valid observations are not screened.
    """
    t, h, w = green_stack.shape
    green = np.ascontiguousarray(np.moveaxis(green_stack, 0, -1).reshape(-1, t), dtype=np.float64)
    swir = np.ascontiguousarray(np.moveaxis(swir_stack, 0, -1).reshape(-1, t), dtype=np.float64)
    clear = _core.tmask.tmask_batch(green, swir, np.asarray(dates, dtype=np.float64), float(scale_factor))
    valid = (green > 0) & (swir > 0) & np.isfinite(green) & np.isfinite(swir)
    out = np.where(valid, clear.astype(bool), True)
    return np.moveaxis(out.reshape(h, w, t), -1, 0)
