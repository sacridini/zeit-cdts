"""The Tmask engine (one pixel, and a numpy stack); ``zeit.tmask`` is in ``_tmask_api``."""
import numpy as np
import warnings

def run_tmask_pixel(dates_julian: np.ndarray, green_band: np.ndarray, swir_band: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    """
    Implements the Tmask (Time-series based cloud masking) algorithm for a single pixel.
    Reference: Zhu and Woodcock (2014) - Automated cloud, cloud shadow, and snow detection.
    
    dates_julian: Array of Julian dates (Day of Year or continuous DOY).
    green_band: Reflectance array for Green band.
    swir_band: Reflectance array for SWIR (usually SWIR1, ~1.6um).
    scale_factor: The multiplier applied to reflectance (e.g., 10000 for standard Landsat/Sentinel).
    
    Returns: A boolean array where True means CLEAR, False means CLOUD/SHADOW.
    """
    # Number of observations
    n = len(dates_julian)
    mask = np.ones(n, dtype=bool)
    
    # If not enough data, assume all clear (or all masked)
    if n < 5:
        return mask
        
    # Scale to 0-1 for standard thresholds
    green = green_band / scale_factor
    swir = swir_band / scale_factor
    
    # Create harmonic design matrix: [1, t, cos(2pi t / 365.25), sin(2pi t / 365.25)]
    w = 2.0 * np.pi / 365.25
    X = np.column_stack((
        np.ones(n),
        dates_julian,
        np.cos(w * dates_julian),
        np.sin(w * dates_julian)
    ))
    
    from sklearn.linear_model import HuberRegressor

    # Tmask uses a robust estimator to fit the time-series model.
    # HuberRegressor is a great approximation of IRLS for this purpose.
    huber_green = HuberRegressor(epsilon=1.35, max_iter=100)
    huber_swir = HuberRegressor(epsilon=1.35, max_iter=100)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            huber_green.fit(X, green)
            huber_swir.fit(X, swir)
        except ValueError:
            return mask
            
    # Predict expected surface reflectance
    pred_green = huber_green.predict(X)
    pred_swir = huber_swir.predict(X)
    
    # Residuals
    res_green = green - pred_green
    res_swir = swir - pred_swir
    
    # Tmask Rules (Zhu & Woodcock 2014)
    # Clouds have unusually high Green reflectance compared to prediction.
    # Shadows have unusually low SWIR reflectance compared to prediction.
    CLOUD_THRESHOLD = 0.04
    SHADOW_THRESHOLD = -0.04
    
    is_cloud = res_green > CLOUD_THRESHOLD
    is_shadow = res_swir < SHADOW_THRESHOLD
    
    # Combine masks
    mask[is_cloud | is_shadow] = False
    
    return mask

def apply_tmask_stack(dates: np.ndarray, green_stack: np.ndarray, swir_stack: np.ndarray, scale_factor: float = 10000.0) -> np.ndarray:
    """
    Vectorized wrapper to apply Tmask to a 3D numpy stack.
    """
    t, h, w = green_stack.shape
    out_mask = np.ones((t, h, w), dtype=bool)
    
    for i in range(h):
        for j in range(w):
            g_pixel = green_stack[:, i, j]
            s_pixel = swir_stack[:, i, j]
            # Ignore NoData pixels
            valid = (g_pixel > 0) & (s_pixel > 0)
            if np.sum(valid) > 5:
                pixel_mask = run_tmask_pixel(dates[valid], g_pixel[valid], s_pixel[valid], scale_factor)
                
                # Reconstruct full length mask
                full_mask = np.ones(t, dtype=bool)
                full_mask[valid] = pixel_mask
                out_mask[:, i, j] = full_mask
                
    return out_mask
