"""Spectral indices and per-pixel temporal metrics.

Shared by the STAC cubes (`zeit.cube`) and the Earth Engine downloads (`zeit.gee`), so
an index or a metric means the same thing whichever source it comes from. Formulas take
surface reflectance (0-1), not scaled digital numbers.
"""

import concurrent.futures
import re
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Band roles each index needs. For normalized differences the order is (a, b) in
# (a - b) / (a + b).
INDICES: Dict[str, tuple] = {
    "NDVI": ("nir", "red"),
    "EVI": ("nir", "red", "blue"),
    "SAVI": ("nir", "red"),
    "kNDVI": ("nir", "red"),
    "NBR": ("nir", "swir2"),
    "NDMI": ("nir", "swir1"),
    "NDWI": ("green", "nir"),
    "MNDWI": ("green", "swir1"),
}
NORMALIZED_DIFFERENCES = ("NDVI", "NBR", "NDMI", "NDWI", "MNDWI")

# Asset names holding each band role, in order of preference: common names (Landsat
# Collection 2, Sentinel-2 on Earth Search), Sentinel-2 on Planetary Computer, and the
# Landsat SR bands of the Earth Engine downloads.
BAND_CANDIDATES: Dict[str, tuple] = {
    "blue": ("blue", "B02", "SR_B2"),
    "green": ("green", "B03", "SR_B3"),
    "red": ("red", "B04", "SR_B4"),
    "nir": ("nir", "nir08", "B08", "SR_B5"),
    "swir1": ("swir16", "B11", "SR_B6"),
    "swir2": ("swir22", "B12", "SR_B7"),
}

DEFAULT_METRICS = ("median", "p10", "p25", "p75", "p90", "std")
_SIMPLE_METRICS = ("median", "mean", "std", "min", "max", "iqr", "count")


def canonical_index(name: str) -> Optional[str]:
    """The registered spelling of an index name (case-insensitive), or None."""
    for key in INDICES:
        if key.lower() == str(name).lower():
            return key
    return None


def index_formula(name: str, get: Callable):
    """Index ``name`` from reflectance bands, ``get(role)`` returning each band.

    Plain arithmetic, so it works on NumPy arrays and xarray objects alike. Use
    `index_array` for NumPy, which also turns divisions by zero into NaN.
    """
    if name in NORMALIZED_DIFFERENCES:
        a, b = (get(role) for role in INDICES[name])
        return (a - b) / (a + b)
    nir, red = get("nir"), get("red")
    if name == "EVI":
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * get("blue") + 1)
    if name == "SAVI":
        return 1.5 * (nir - red) / (nir + red + 0.5)
    if name == "kNDVI":
        return np.tanh(((nir - red) / (nir + red)) ** 2)
    raise ValueError(f"Unknown index {name!r}. Available: {', '.join(INDICES)}.")


def index_array(name: str, get: Callable) -> np.ndarray:
    """`index_formula` on float32 NumPy arrays, with NaN wherever the index is undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(index_formula(name, get), dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def parse_metrics(metrics: Sequence[str]) -> List[str]:
    """Validate temporal metric names, normalizing percentiles (``"P05"`` -> ``"p5"``).

    Accepted: ``median``, ``mean``, ``std``, ``min``, ``max``, ``iqr`` (p75 - p25),
    ``count`` (valid observations) and ``pNN`` for any integer percentile 0-100.
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    out = []
    for m in metrics:
        key = str(m).lower()
        match = re.fullmatch(r"p(\d{1,3})", key)
        if match and int(match.group(1)) <= 100:
            key = f"p{int(match.group(1))}"
        elif key not in _SIMPLE_METRICS:
            raise ValueError(
                f"Unknown metric {m!r}. Use {', '.join(_SIMPLE_METRICS)} or a percentile "
                "such as 'p10'."
            )
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("Provide at least one metric, e.g. ['median', 'p10', 'p90'].")
    return out


def _percentile(s, n, q):
    """Linear-interpolated percentile ``q`` of the first ``n`` values of sorted ``s``.

    Same interpolation as ``np.nanpercentile``'s default.
    """
    pos = q / 100.0 * np.maximum(n - 1, 0)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    v_lo = np.take_along_axis(s, lo[None], 0)[0]
    v_hi = np.take_along_axis(s, hi[None], 0)[0]
    return v_lo + (v_hi - v_lo) * (pos - lo)


def temporal_metrics(a: np.ndarray, metrics: Sequence[str], n_threads: int = 4) -> np.ndarray:
    """NaN-aware metrics along axis 0 of a ``(time, y, x)`` array.

    Returns ``(len(metrics), y, x)`` float32. Pixels without valid observations are NaN
    (``count`` is 0 there). Each strip of rows is sorted once (NaNs last) and every
    metric read from it, on a few threads (``np.sort`` releases the GIL). ``std`` is the
    population standard deviation (``ddof=0``).
    """
    metrics = parse_metrics(metrics)
    t, ny, nx = a.shape
    out = np.full((len(metrics), ny, nx), np.nan, dtype=np.float32)
    if "count" in metrics:
        out[metrics.index("count")] = 0
    if t == 0:
        return out

    def strip(r0):
        r1 = min(ny, r0 + 64)
        s = np.sort(a[:, r0:r1], axis=0)
        valid = np.isfinite(s)
        n = valid.sum(axis=0)
        empty = n == 0
        z = np.where(valid, s, 0).astype(np.float64)
        mean = z.sum(axis=0) / np.maximum(n, 1)
        for k, m in enumerate(metrics):
            if m == "count":
                out[k, r0:r1] = n
                continue
            if m == "mean":
                v = mean
            elif m == "std":
                d = np.where(valid, s - mean[None], 0)
                v = np.sqrt((d * d).sum(axis=0) / np.maximum(n, 1))
            elif m == "min":
                v = s[0]
            elif m == "max":
                v = np.take_along_axis(s, np.maximum(n - 1, 0)[None], 0)[0]
            elif m == "median":
                v = _percentile(s, n, 50)
            elif m == "iqr":
                v = _percentile(s, n, 75) - _percentile(s, n, 25)
            else:
                v = _percentile(s, n, int(m[1:]))
            v = np.asarray(v, dtype=np.float32)
            v[empty] = np.nan
            out[k, r0:r1] = v

    with concurrent.futures.ThreadPoolExecutor(n_threads) as ex:
        list(ex.map(strip, range(0, ny, 64)))
    return out


def resolve_band(role: str, available: Sequence[str], band_map: Optional[Dict[str, str]] = None) -> str:
    """The asset name that holds band ``role`` among ``available`` band names."""
    if band_map and role in band_map:
        return band_map[role]
    for name in BAND_CANDIDATES[role]:
        if name in available:
            return name
    raise ValueError(
        f"No '{role}' band among {list(available)} (looked for {', '.join(BAND_CANDIDATES[role])}). "
        f"Load it, or pass band_map={{'{role}': '<band name>'}}."
    )


def compute_indices(cube, indices: Sequence[str], band_map: Optional[Dict[str, str]] = None):
    """Spectral indices from a reflectance cube, lazily.

    cube: ``xarray.DataArray`` with a ``band`` dimension holding reflectance (0-1), e.g.
        from `build_time_series` (float dtype) or `build_annual_composites`.
    indices: Index names: NDVI, EVI, SAVI, kNDVI, NBR, NDMI, NDWI, MNDWI.
    band_map: Overrides for the band each role is read from, e.g. ``{"nir": "B8A"}``.
        By default each role is found by its usual asset names (``red``, ``B04``,
        ``SR_B4``...).

    Returns a DataArray with the same dimensions, whose ``band`` coordinate lists the
    indices. Divisions by zero become NaN. Stays lazy on a Dask-backed cube, so only
    the indices are ever written when you save it.
    """
    import pandas as pd
    import xarray as xr

    if np.issubdtype(cube.dtype, np.integer):
        raise ValueError(
            "compute_indices needs reflectance. Build the cube with a float dtype "
            "(the default), not raw integer digital numbers."
        )
    if isinstance(indices, str):
        indices = [indices]
    available = [str(b) for b in cube.band.values]
    names, layers = [], []
    for idx in indices:
        name = canonical_index(idx)
        if name is None:
            raise ValueError(f"Unknown index {idx!r}. Available: {', '.join(INDICES)}.")
        get = lambda role: cube.sel(band=resolve_band(role, available, band_map), drop=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            layer = index_formula(name, get)
        layers.append(layer.where(np.isfinite(layer)).astype(cube.dtype))
        names.append(name)
    out = xr.concat(layers, dim=pd.Index(names, name="band"), coords="minimal", compat="override")
    out = out.transpose(*cube.dims)
    out.attrs = cube.attrs
    out.name = "indices"
    return out
