"""Quality bands to observation weights.

Each decoder turns the quality band of a sensor (MODIS MOD13 SummaryQA, MODIS
MOD09 500 m State QA, Sentinel-2 L2A Scene Classification Layer) into a weight per
observation, for the ``weights=`` of ``zeit.smooth`` and ``zeit.phenology``.

Four levels are used: ``wmax`` (good), ``wmid`` (usable, with a warning), ``wmin``
(measures the surface with a known bias, or may be contaminated: clouds, shadows,
snow) and ``0`` (no data: fill values, NoData, NaN and codes outside the valid
range). The classification is Zeit's, made from the product guides cited in each
function.

A numpy array gives a numpy array of the same shape. A ``DataArray`` or a raster
path gives a ``DataArray`` of weights with the input's dims, coordinates and
georeferencing (lazy if the input was). The raster's NoData and NaN get the weight
``0``, even when the NoData value would otherwise be a good code.
"""

from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np

__all__ = ["qc_modis_summary", "qc_modis_state", "qc_sentinel2_scl", "qa_mask", "QA_RULES"]

# Attributes that describe stored codes, not weights.
_NOT_FOR_WEIGHTS = ("scale_factor", "add_offset", "_FillValue", "nodata", "missing_value",
                    "units", "long_name", "valid_range")


def _check_weights(wmin: float, wmid: float, wmax: float) -> Tuple[float, float, float]:
    try:
        ok = 0 <= wmin <= wmid <= wmax
    except TypeError as exc:
        raise TypeError("wmin, wmid and wmax must be numbers") from exc
    if not ok:
        raise ValueError(f"weights must satisfy 0 <= wmin <= wmid <= wmax, got "
                         f"wmin={wmin}, wmid={wmid}, wmax={wmax}")
    return float(wmin), float(wmid), float(wmax)


def _integers(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Codes as int64 and a mask of the valid ones (finite, whole numbers)."""
    if a.dtype.kind == "f":
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(a) & (a == np.round(a)) & (np.abs(a) < 2.0 ** 62)
        return np.where(ok, a, 0).astype(np.int64), ok
    if a.dtype.kind in "iu":
        ok = np.ones(a.shape, dtype=bool)
        if a.dtype.kind == "u" and a.dtype.itemsize == 8:
            ok = a <= np.iinfo(np.int64).max
        return a.astype(np.int64), ok
    raise TypeError(f"quality codes must be integer or float numbers, got {a.dtype}")


def _nodata_of(da: Any) -> Any:
    for source in (da.attrs, da.encoding):
        for key in ("_FillValue", "nodata"):
            if source.get(key) is not None:
                return source[key]
    try:
        return da.rio.nodata
    except Exception:  # no rio accessor or no georeferencing
        return None


def _apply(source: Any, decode: Callable[..., np.ndarray], *args: float) -> Any:
    """Run ``decode(codes, *args)`` on a numpy array, a DataArray or a raster path."""
    if isinstance(source, (str, Path)):
        from ._load import load_raster

        source = load_raster(str(source), masked=False)

    if hasattr(source, "dims") and hasattr(source, "coords"):
        if source.dtype.kind not in "iuf":
            raise TypeError(f"quality codes must be integer or float numbers, got {source.dtype}")
        nodata = _nodata_of(source)
        if nodata is not None:
            nodata = float(nodata)
            if np.isnan(nodata):
                nodata = None

        def run(values: np.ndarray) -> np.ndarray:
            weights = decode(values, *args)
            if nodata is not None:
                weights = np.where(values.astype(np.float64) == nodata, 0.0, weights)
            return weights

        data = source.data
        if hasattr(data, "map_blocks"):  # dask: stay lazy
            data = data.map_blocks(run, dtype=np.float64)
        else:
            data = run(np.asarray(data))
        # copy keeps the coordinates with their attributes (the CRS lives in spatial_ref)
        out = source.copy(data=data)
        out.attrs = {k: v for k, v in source.attrs.items() if k not in _NOT_FOR_WEIGHTS}
        out.encoding = {k: v for k, v in source.encoding.items()
                        if k in ("grid_mapping", "preferred_chunks")}
        out.name = "weights"
        return out

    if isinstance(source, (bool, np.bool_, str, bytes)):
        raise TypeError(f"quality codes must be numbers, got {type(source).__name__}")
    values = np.asarray(source)
    if values.dtype.kind not in "iuf":
        raise TypeError(f"quality codes must be integer or float numbers, got {values.dtype}")
    weights = decode(values, *args)
    return weights[()] if weights.ndim == 0 else weights


def _summary(values: np.ndarray, wmin: float, wmid: float, wmax: float) -> np.ndarray:
    v, ok = _integers(values)
    table = np.array([wmax, wmid, wmin, wmin])
    weights = table[np.clip(v, 0, 3)]
    return np.where(ok & (v >= 0) & (v <= 3), weights, 0.0)


def _state(values: np.ndarray, wmin: float, wmid: float, wmax: float) -> np.ndarray:
    v, ok = _integers(values)
    # An int16 holding the bits of a uint16 reads as negative: -1 is the fill value.
    v = np.where((v >= -32768) & (v < 0), v & 0xFFFF, v)
    ok = ok & (v >= 0) & (v <= 65535)
    v = np.where(ok, v, 65535)

    def bits(i: int, j: int) -> np.ndarray:
        return (v >> i) & ((1 << (j - i + 1)) - 1)

    cloud, shadow, aerosol = bits(0, 1), bits(2, 2), bits(6, 7)
    cirrus, internal_cloud, fire = bits(8, 9), bits(10, 10), bits(11, 11)
    snow_mod35, adjacent, snow_internal = bits(12, 12), bits(13, 13), bits(15, 15)

    weights = np.full(v.shape, wmax, dtype=np.float64)

    def cap(condition: np.ndarray, value: float) -> None:
        np.copyto(weights, np.minimum(weights, value), where=condition)

    cap((cloud == 1) | (cloud == 2), wmin)
    cap(shadow == 1, wmin)
    cap(internal_cloud == 1, wmin)
    cap(cirrus == 3, wmin)
    cap(cirrus == 2, wmin)
    cap(cirrus == 1, wmid)
    cap(aerosol == 3, wmin)
    cap(aerosol == 0, wmid)
    cap(adjacent == 1, wmid)
    cap((snow_mod35 == 1) | (snow_internal == 1), wmin)
    cap(fire == 1, wmin)
    return np.where(ok & (v != 65535), weights, 0.0)


def _scl(values: np.ndarray, wmin: float, wmid: float, wmax: float) -> np.ndarray:
    v, ok = _integers(values)
    #               0  1     2     3     4     5     6     7     8     9     10    11
    table = np.array([0, 0, wmin, wmin, wmax, wmax, wmid, wmid, wmin, wmin, wmin, wmin])
    return np.where(ok & (v >= 0) & (v <= 11), table[np.clip(v, 0, 11)], 0.0)


def qc_modis_summary(qa: Any, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> Any:
    """Weights from the MOD13 / MYD13 SummaryQA ("pixel reliability") band.

    The layer is an 8-bit signed band with valid range 0 to 3 and fill value -1
    (255 when read as unsigned). Source: MODIS Vegetation Indices (MOD13) C5
    User's Guide, v1.00 (Solano, Didan, Jacobson and Huete), section 5.5.2,
    Table 4. The classification into weights is Zeit's, made from that guide.

    ======  ======================  ======================================
    Code    Meaning (guide)         Weight
    ======  ======================  ======================================
    0       Good data               ``wmax``
    1       Marginal data           ``wmid``
    2       Snow/ice                ``wmin``
    3       Cloudy                  ``wmin``
    -1/255  Fill / not processed    ``0``
    other   Outside the valid 0-3   ``0``
    ======  ======================  ======================================

    Parameters
    ----------
    qa
        SummaryQA values of any shape: a numpy array, a ``DataArray`` or the path
        of a raster. NaN, non-integer values and the raster's NoData get ``0``.
    wmin, wmid, wmax
        Weights for bad, doubtful and good observations, with
        ``0 <= wmin <= wmid <= wmax``.

    Returns
    -------
    A numpy array (numpy input) or a ``DataArray`` with the input's dims, coordinates
    and georeferencing, of float weights.
    """
    args = _check_weights(wmin, wmid, wmax)
    return _apply(qa, _summary, *args)


def qc_modis_state(qa: Any, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> Any:
    """Weights from the MOD09 / MYD09 500 m State QA band (``sur_refl_state_500m``).

    A 16-bit unsigned layer with fill value 65535. Source: MODIS Collection 6.1
    Land Surface Reflectance (MOD09) User Guide, section 3.3, Table 13 (State QA
    description); section 4.1 of the guide says cloud information must come from
    this layer. The classification into weights is Zeit's, made from that guide.
    Bit 0 is the least significant. The final weight is the minimum over all rows.

    ====================================  =============================
    Condition                             Weight
    ====================================  =============================
    Fill value 65535 or invalid code      ``0``
    Bits 0-1 = 01 (cloudy)                ``wmin``
    Bits 0-1 = 10 (mixed)                 ``wmin``
    Bits 0-1 = 11 (not set, clear)        no penalty
    Bit 2 (cloud shadow)                  ``wmin``
    Bit 10 (internal cloud algorithm)     ``wmin``
    Bits 8-9 = 11 (high cirrus)           ``wmin``
    Bits 8-9 = 10 (average cirrus)        ``wmin``
    Bits 8-9 = 01 (small cirrus)          ``wmid``
    Bits 6-7 = 11 (high aerosol)          ``wmin``
    Bits 6-7 = 00 (climatology aerosol)   ``wmid``
    Bits 6-7 = 01 or 10 (low, average)    no penalty
    Bit 13 (adjacent to cloud)            ``wmid``
    Bit 12 or bit 15 (snow/ice)           ``wmin``
    Bit 11 (fire)                         ``wmin``
    None of the above                     ``wmax``
    ====================================  =============================

    Bits 3-5 (land/water) and 14 (salt pan) are not used: they describe the target,
    not the quality of the observation, and masking water is a job for a land-cover
    mask. Signed input that holds the bit pattern of a uint16 (an int16 where -1 is
    the fill value) is read as unsigned.

    Parameters
    ----------
    qa
        State QA values of any shape: a numpy array, a ``DataArray`` or the path of
        a raster. NaN, non-integer values, values outside 0-65535 and the raster's
        NoData get ``0``.
    wmin, wmid, wmax
        Weights for bad, doubtful and good observations, with
        ``0 <= wmin <= wmid <= wmax``.

    Returns
    -------
    A numpy array (numpy input) or a ``DataArray`` with the input's dims, coordinates
    and georeferencing, of float weights.
    """
    args = _check_weights(wmin, wmid, wmax)
    return _apply(qa, _state, *args)


def qc_sentinel2_scl(scl: Any, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> Any:
    """Weights from the Sentinel-2 L2A Scene Classification Layer (SCL).

    Source: ESA/Copernicus SentiWiki, "S2 Processing", Table 3 (SCL classes), and the
    Sen2Cor Configuration and User Manual. The classification into weights is Zeit's,
    made from those guides.

    =====  ==========================  ==========
    Code   Class (ESA)                 Weight
    =====  ==========================  ==========
    0      NO_DATA                     ``0``
    1      SATURATED_OR_DEFECTIVE      ``0``
    2      CAST_SHADOWS                ``wmin``
    3      CLOUD_SHADOWS               ``wmin``
    4      VEGETATION                  ``wmax``
    5      NOT_VEGETATED               ``wmax``
    6      WATER                       ``wmid``
    7      UNCLASSIFIED                ``wmid``
    8      CLOUD_MEDIUM_PROBABILITY    ``wmin``
    9      CLOUD_HIGH_PROBABILITY      ``wmin``
    10     THIN_CIRRUS                 ``wmin``
    11     SNOW or ICE                 ``wmin``
    other  (>= 12, negative, NaN,      ``0``
           non-integer)
    =====  ==========================  ==========

    Parameters
    ----------
    scl
        SCL values of any shape: a numpy array, a ``DataArray`` or the path of a
        raster. NaN, fractional values (as in a bilinearly resampled SCL) and the
        raster's NoData get ``0``.
    wmin, wmid, wmax
        Weights for bad, doubtful and good observations, with
        ``0 <= wmin <= wmid <= wmax``.

    Returns
    -------
    A numpy array (numpy input) or a ``DataArray`` with the input's dims, coordinates
    and georeferencing, of float weights.
    """
    args = _check_weights(wmin, wmid, wmax)
    return _apply(scl, _scl, *args)


#: The rules of ``qa_mask``: the codes (or bits of the codes) of a quality band that keep an observation.
QA_RULES = {
    "fmask": "Fmask codes: 0 clear land and 1 water are usable; 2 shadow, 3 snow, 4 cloud and 255 fill are not",
    "landsat_c2": "Landsat Collection 2 QA_PIXEL bits: fill (0), dilated cloud (1), cloud (3), cloud shadow (4) "
                  "and snow (5) are not usable",
    "nonzero": "a generic mask: 0 is masked, anything else is usable",
    "scl": "Sentinel-2 scene classification (SCL): 4 vegetation, 5 bare soil and 6 water are usable",
    "hls_fmask": "HLS v2.0 Fmask bits: cloud (1), adjacent to cloud or shadow (2), cloud shadow (3) and snow or "
                 "ice (4) are not usable, nor 255 (fill)",
}


def _usable(c: Any, rule: str) -> Any:
    """The rule on integer codes ``c`` (numpy or xarray)."""
    if rule == "fmask":
        return (c == 0) | (c == 1)
    if rule == "landsat_c2":
        return (c & 0x3B) == 0
    if rule == "nonzero":
        return c != 0
    if rule == "scl":
        return (c >= 4) & (c <= 6)
    return (c != 255) & ((c & 0x1E) == 0)  # hls_fmask


def qa_mask(qa: Any, rule: str) -> Any:
    """The observations a quality band keeps, as a boolean "usable" mask.

    The rules are the ones Janus applies to a quality band. A code counts when it is a
    number of 0 or more (fractions are cut to the whole code); NaN, negative values and
    the raster's NoData are never usable.

    ======================  ================================================================
    ``rule``                Usable
    ======================  ================================================================
    ``"fmask"``             0 (clear land) and 1 (water); 2 shadow, 3 snow, 4 cloud and 255
                            (fill) are masked
    ``"landsat_c2"``        QA_PIXEL with the bits 0 (fill), 1 (dilated cloud), 3 (cloud),
                            4 (cloud shadow) and 5 (snow) clear (``code & 0x3B == 0``)
    ``"nonzero"``           any code but 0
    ``"scl"``               Sentinel-2 SCL 4 (vegetation), 5 (bare soil) and 6 (water)
    ``"hls_fmask"``         HLS v2.0 Fmask with the bits 1 to 4 (cloud, adjacent, shadow, snow
                            or ice) clear, and not 255 (fill)
    ======================  ================================================================

    Parameters
    ----------
    qa
        The quality band: a numpy array, a ``DataArray`` (lazy if it was) or the path of a
        raster (read with ``masked=True``).
    rule
        One of the rules above (case and ``-`` / ``_`` do not matter).

    Returns
    -------
    A boolean numpy array (numpy input) or a boolean ``DataArray`` with the input's dims,
    coordinates and georeferencing.

    Examples
    --------
    >>> usable = zeit.qa_mask(cube.sel(band="fmask"), "fmask")
    >>> clear = cube.sel(band="nir").where(usable)
    """
    key = str(rule).lower().replace("-", "_")
    if key not in QA_RULES:
        raise ValueError(f"unknown QA rule {rule!r}; use one of {', '.join(QA_RULES)}")
    if isinstance(qa, (str, Path)):
        from ._load import load_raster

        qa = load_raster(str(qa), masked=True)
    if hasattr(qa, "dims") and hasattr(qa, "coords"):
        if qa.dtype.kind not in "iufb":
            raise TypeError(f"quality codes must be numbers, got {qa.dtype}")
        nodata = _nodata_of(qa)
        if nodata is not None and not np.isnan(float(nodata)):
            qa = qa.where(qa != nodata)
        valid = qa.notnull() & (qa >= 0)
        out = valid & _usable(qa.fillna(0).astype("int64"), key)
        out.attrs = {}
        out.encoding = {}
        out.name = "usable"
        return out
    values = np.asarray(qa)
    if values.dtype.kind not in "iufb":
        raise TypeError(f"quality codes must be numbers, got {values.dtype}")
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(values) & (values >= 0) if values.dtype.kind == "f" else values >= 0
    codes = np.where(valid, values, 0).astype(np.int64)
    return valid & _usable(codes, key)
