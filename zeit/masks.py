"""Masks derived from CCDC models."""
from typing import Any, Union

import numpy as np


def extract_water_mask(segments: Any, green: Union[str, int] = "green", swir: Union[str, int] = "swir1",
                       threshold: float = 500.0) -> Any:
    """Persistent water from CCDC models: water reflects a little in green and almost nothing
    in SWIR, so a pixel is water when its green level is above its SWIR level and the SWIR
    level is below ``threshold`` (reflectance x 10000).

    segments: the result of ``zeit.ccdc`` (a Dataset). The levels are those of each pixel's
        first segment (the landscape at the start), at the middle of that segment
        (``a0 + c1 t``: the model's intercept at year 0 alone is not a reflectance).
        A numpy coefficient stack as older versions wrote it, ``(max_segments, params,
        rows, cols)``, is still accepted with integer band positions (it uses the raw
        intercepts; before 0.42 it read them assuming 7 parameters per band instead of 9).
    green, swir: the green and SWIR (SWIR1 or SWIR2) bands, by name (Dataset) or 0-based
        position.
    threshold: the highest SWIR level of water.

    Returns a ``(y, x)`` uint8 map, ``1`` = water (georeferenced for a Dataset).
    """
    import xarray as xr

    if isinstance(segments, xr.Dataset):
        return _from_segments(segments, green, swir, threshold)
    return _from_stack(np.asarray(segments), int(green), int(swir), threshold)


def _from_segments(segments, green, swir, threshold):
    import pandas as pd

    from ._ccdc_api import _DATENUM_OFFSET

    names = [str(b) for b in segments.band.values]

    def band(b):
        if isinstance(b, (int, np.integer)):
            return names[int(b)]
        if str(b) not in names:
            raise ValueError(f"band {b!r}: the CCDC result has the bands {names}")
        return str(b)

    first = segments.isel(segment=0)
    middle = first.t_start + (first.t_end - first.t_start) / 2
    days = (middle - np.datetime64("1970-01-01", "ns")) / np.timedelta64(1, "D")
    t = days + pd.Timestamp("1970-01-01").toordinal() + _DATENUM_OFFSET

    def level(b):
        c = first.coefs.sel(band=band(b))
        return c.sel(coef="a0") + c.sel(coef="c1") * t

    g, s = level(green), level(swir)
    water = ((g > s) & (s < threshold) & first.t_start.notnull()).astype(np.uint8)
    water = water.drop_vars([c for c in water.coords if c not in ("y", "x", "spatial_ref")]).rename("water")
    water.attrs = {"long_name": "persistent water (CCDC)"}
    if segments.rio.crs is not None:
        water = water.rio.write_crs(segments.rio.crs)
    return water


def _from_stack(ccdc_coefs_stack: np.ndarray, green_band_idx: int, swir_band_idx: int,
                threshold: float) -> np.ndarray:
    """The former numpy layout: (max_segments, params, rows, cols), per band an RMSE and the
    coefficients after the three dates (t_start, t_end, t_break)."""
    segments, params, rows, cols = ccdc_coefs_stack.shape

    # After t_start, t_end and t_break, each band holds its RMSE and 8 coefficients
    # (a0, c1, a1, b1, a2, b2, a3, b3): the intercept of band b is at 3 + 9 b + 1.
    green_intercept_idx = 4 + green_band_idx * 9
    swir_intercept_idx = 4 + swir_band_idx * 9

    water_mask = np.zeros((rows, cols), dtype=np.uint8)

    # We look at the very first segment (initial state of the landscape)
    green_intercept = ccdc_coefs_stack[0, green_intercept_idx, :, :]
    swir_intercept = ccdc_coefs_stack[0, swir_intercept_idx, :, :]

    # Avoid nodata (where intercept == 0)
    valid = (green_intercept != 0) | (swir_intercept != 0)

    is_water = valid & (green_intercept > swir_intercept) & (swir_intercept < threshold)
    water_mask[is_water] = 1

    return water_mask
