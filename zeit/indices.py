"""Spectral indices and per-pixel temporal metrics.

Shared by the STAC cubes (`zeit.cube`) and the Earth Engine downloads (`zeit.gee`), so
an index or a metric means the same thing whichever source it comes from. Formulas take
surface reflectance (0-1), not scaled digital numbers.
"""

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
    # Souza et al. (2005): unmixed with their endmembers (zeit._sma.SOUZA_2005)
    "NDFI": ("blue", "green", "red", "nir", "swir1", "swir2"),
    # Tasseled Cap brightness, greenness, wetness and FORCE's disturbance (B - G - W), Crist (1985) for every sensor
    # as in FORCE (zeit._tasseled.CRIST_1985; zeit.tasseled_cap has the per-sensor matrices)
    "TCB": ("blue", "green", "red", "nir", "swir1", "swir2"),
    "TCG": ("blue", "green", "red", "nir", "swir1", "swir2"),
    "TCW": ("blue", "green", "red", "nir", "swir1", "swir2"),
    "TCD": ("blue", "green", "red", "nir", "swir1", "swir2"),
}
TASSELED_CAP = {"TCB": "brightness", "TCG": "greenness", "TCW": "wetness"}
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
# What Earth Engine's reducers do not give: only the local (compiled) computation has them.
_LOCAL_METRICS = ("range", "skew", "kurtosis")


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
    if name in TASSELED_CAP or name == "TCD":
        from ._tasseled import CRIST_1985

        def component(key):
            c = CRIST_1985[key]
            total = c[0] * get(INDICES[name][0])
            for k, role in enumerate(INDICES[name][1:], 1):
                total = total + c[k] * get(role)
            return total

        if name == "TCD":
            return (component("brightness") - component("greenness")) - component("wetness")
        return component(TASSELED_CAP[name])
    nir, red = get("nir"), get("red")
    if name == "EVI":
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * get("blue") + 1)
    if name == "SAVI":
        return 1.5 * (nir - red) / (nir + red + 0.5)
    if name == "kNDVI":
        return np.tanh(((nir - red) / (nir + red)) ** 2)
    if name == "NDFI":
        from ._sma import ndfi_from_bands
        return ndfi_from_bands([get(role) for role in INDICES["NDFI"]])
    raise ValueError(f"Unknown index {name!r}. Available: {', '.join(INDICES)}.")


def index_array(name: str, get: Callable) -> np.ndarray:
    """`index_formula` on float32 NumPy arrays, with NaN wherever the index is undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.asarray(index_formula(name, get), dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def parse_metrics(metrics: Sequence[str], extended: bool = True) -> List[str]:
    """Validate temporal metric names, normalizing percentiles (``"P05"`` -> ``"p5"``).

    Accepted: ``median``, ``mean``, ``std``, ``min``, ``max``, ``iqr`` (p75 - p25),
    ``count`` (valid observations) and ``pNN`` for any integer percentile 0-100; and,
    with ``extended`` (what Earth Engine cannot compute, so ``False`` there), ``range``
    (max - min), ``skew`` (M3 / sigma^3) and ``kurtosis`` (excess, M4 / sigma^4 - 3).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed = _SIMPLE_METRICS + (_LOCAL_METRICS if extended else ())
    out = []
    for m in metrics:
        key = str(m).lower()
        match = re.fullmatch(r"p(\d{1,3})", key)
        if match and int(match.group(1)) <= 100:
            key = f"p{int(match.group(1))}"
        elif key not in allowed:
            raise ValueError(
                f"Unknown metric {m!r}. Use {', '.join(allowed)} or a percentile "
                "such as 'p10'."
            )
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("Provide at least one metric, e.g. ['median', 'p10', 'p90'].")
    return out


def temporal_metrics(a: np.ndarray, metrics: Sequence[str], n_threads: int = 4) -> np.ndarray:
    """NaN-aware metrics along axis 0 of a ``(time, y, x)`` array.

    Returns ``(len(metrics), y, x)`` float32. Pixels without valid observations are NaN
    (``count`` is 0 there). Computed by the compiled core (``zeit._core.tsa``, the same
    kernel as the public ``zeit.temporal_metrics``) on ``n_threads`` threads; a value that
    is not finite counts as a missing observation. ``std`` is the population standard
    deviation (``ddof=0``).
    """
    from . import _core

    metrics = parse_metrics(metrics)
    t, ny, nx = a.shape
    if t == 0:
        out = np.full((len(metrics), ny, nx), np.nan, dtype=np.float32)
        if "count" in metrics:
            out[metrics.index("count")] = 0
        return out
    values = np.ascontiguousarray(a.reshape(t, ny * nx).T, dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    out = _core.tsa.stm_batch(values, metrics, 0.0, int(n_threads))  # (pixels, metrics)
    return np.ascontiguousarray(out.T, dtype=np.float32).reshape(len(metrics), ny, nx)


def resolve_band(role: str, available: Sequence[str], band_map: Optional[Dict[str, str]] = None) -> str:
    """The asset name that holds band ``role`` among ``available`` band names."""
    if band_map and role in band_map:
        return band_map[role]
    if role in available:   # named after the role itself ("swir1")
        return role
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
    indices: Index names: NDVI, EVI, SAVI, kNDVI, NBR, NDMI, NDWI, MNDWI, NDFI (the
        Normalized Difference Fraction Index of Souza et al. 2005, from the fractions of
        a spectral unmixing of the six Landsat bands, see `zeit.unmix`), and the Tasseled
        Cap TCB (brightness), TCG (greenness), TCW (wetness) and TCD (FORCE's disturbance,
        TCB - TCG - TCW), all with the matrix of Crist 1985 as FORCE applies it to every
        sensor, in reflectance units (see `zeit.tasseled_cap` for the per-sensor matrices
        and FORCE's integer arithmetic).
    band_map: Overrides for the band each role is read from, e.g. ``{"nir": "B8A"}``.
        By default each role is found by its usual asset names (``red``, ``B04``,
        ``SR_B4``...).

    Returns a DataArray with the same dimensions, whose ``band`` coordinate lists the
    indices. Divisions by zero become NaN. Stays lazy on a Dask-backed cube, so only
    the indices are ever written when you save it.
    """
    from ._embeddings import refuse

    refuse(cube, "zeit.compute_indices")
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


def nd(a, b):
    """The normalized difference ``(a - b) / (a + b)``, NaN where ``a + b`` is 0.

    The formula of NDVI (``nd(nir, red)``), NDWI, NBR and the other normalized
    differences of `compute_indices`. Works on numpy arrays, scalars and xarray objects
    (lazy on Dask); integer inputs are computed in floating point, so unsigned values
    cannot wrap around.

    Parameters
    ----------
    a, b
        The two bands, on the same grid (or broadcastable): numpy arrays, scalars or
        ``DataArray`` objects.

    Returns
    -------
    The index, a float32 or float64 array (or ``DataArray`` with ``a``'s coordinates).

    Examples
    --------
    >>> ndvi = zeit.nd(nir, red)
    """
    if hasattr(a, "dims") or hasattr(b, "dims"):
        a, b = _floats(a), _floats(b)
        total = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = ((a - b) / total).where(total != 0)
        out.attrs = {}
        return out
    a, b = _floats(a), _floats(b)
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total != 0, (a - b) / total, np.nan)


def _floats(x):
    """``x`` as a float array: integers of up to 16 bits as float32, larger ones as float64."""
    if hasattr(x, "dims"):
        if x.dtype.kind in "iub":
            return x.astype(np.float32 if x.dtype.itemsize <= 2 else np.float64)
        return x
    x = np.asarray(x)
    if x.dtype.kind in "iub":
        return x.astype(np.float32 if x.dtype.itemsize <= 2 else np.float64)
    return x


def normalized_difference(cube, a: str, b: str, band_map: Optional[Dict[str, str]] = None):
    """The normalized difference of two bands of a cube: ``(a - b) / (a + b)``, NaN where ``a + b`` is 0.

    ``normalized_difference(cube, "nir", "red")`` is NDVI, the same formula as
    `compute_indices` and `nd`. ``a`` and ``b`` are band names of the cube's ``band``
    dimension, or band roles (``"nir"``, ``"red"``, ``"swir1"``...) found by their usual
    asset names (``B08``, ``SR_B4``...) or ``band_map``, as `compute_indices` does.

    Parameters
    ----------
    cube
        A ``DataArray`` with a ``band`` dimension, e.g. ``load_raster(..., bands={...})``.
    a, b
        The two bands (band names or roles); ``a`` comes first in the difference.
    band_map
        Overrides for the band each role is read from, e.g. ``{"nir": "B8A"}``.

    Returns
    -------
    A DataArray without the ``band`` dimension: ``(time, y, x)`` for a ``(time, band, y, x)`` cube.
    """
    available = [str(x) for x in cube.band.values]

    def pick(name):
        if band_map and name in band_map:
            name = band_map[name]
        if str(name) in available:
            return cube.isel(band=available.index(str(name)), drop=True)
        return cube.sel(band=resolve_band(name, available, band_map), drop=True)

    out = nd(pick(a), pick(b))
    out.attrs = cube.attrs
    return out
