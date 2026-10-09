"""
QC/QA-band decoders that turn a sensor's raw quality-assurance layer into
per-observation reliability weights in [0, 1], for the ``weights=`` of
``zeit.phenology`` and ``zeit.smooth``.

This mirrors phenofit's `qcFUN.R` (qc_summary, qc_StateQA, qc_sentinel2), so the
same "good/marginal/snow-or-cloud" weighting scheme used there is available in
zeit. A QA cube (a ``DataArray``, or a raster path) gives a weights cube with its
dims, dates and georeferencing, ready to pass along with the data it describes;
its NoData (and NaN) get the weight 0.
"""

import os
from typing import Any, Callable

import numpy as np

__all__ = ["qc_modis_summary", "qc_modis_state", "qc_sentinel2_scl"]


def _weights(qa: Any, decode: Callable[[np.ndarray], np.ndarray]) -> Any:
    """``decode`` (numpy QA values -> weights) applied to ``qa`` as given: a numpy array
    gives one, a ``DataArray`` (or a raster path, read with ``load_raster``) a
    ``DataArray`` with its coordinates and georeferencing, lazy if it was."""
    import xarray as xr

    if isinstance(qa, (str, os.PathLike)):
        from ._load import load_raster

        qa = load_raster(qa, masked=False)
    if not isinstance(qa, xr.DataArray):
        values = np.asarray(qa)
        return np.where(_missing(values, None), 0.0, decode(values))
    nodata = qa.rio.nodata if "y" in qa.dims and "x" in qa.dims else None
    crs = qa.rio.crs if "y" in qa.dims and "x" in qa.dims else None

    def _block(values):
        return np.where(_missing(values, nodata), 0.0, decode(values))

    out = xr.apply_ufunc(_block, qa, dask="parallelized", output_dtypes=[np.float64])
    out = out.rename("weights")
    out.attrs = {"long_name": "observation weight"}
    if crs is not None:
        out = out.rio.write_crs(crs)
    return out


def _missing(values: np.ndarray, nodata: Any) -> np.ndarray:
    missing = np.isnan(values) if np.issubdtype(values.dtype, np.floating) else np.zeros(values.shape, bool)
    if nodata is not None and not np.isnan(nodata):
        missing |= values == nodata
    return missing


def _get_bits(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract bits [start, end] (inclusive, 0-indexed from the LSB)."""
    n_bits = end - start + 1
    mask = (1 << n_bits) - 1
    return (np.asarray(x).astype(np.int64) >> start) & mask


def qc_modis_summary(qa: Any, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> Any:
    """
    Weights from the MOD13A1/A2/Q1 "SummaryQA" (pixel reliability) band.
    Port of phenofit's `qc_summary()` (R/qcFUN.R).

        0 good      -> wmax
        1 marginal  -> wmid
        2 snow/ice  -> wmin
        3 cloudy    -> wmin
        other/fill  -> 0.0 (excluded)

    ``qa``: a numpy array (weights of the same shape), or a ``DataArray`` or raster path
    (a ``DataArray`` of weights, NoData -> 0).
    """
    def decode(qa):
        w = np.zeros(qa.shape, dtype=np.float64)
        w[qa == 0] = wmax
        w[qa == 1] = wmid
        w[(qa >= 2) & (qa <= 3)] = wmin
        return w

    return _weights(qa, decode)


def qc_modis_state(qa: Any, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> Any:
    """
    Weights from the MOD09A1/MYD09A1 500m "State QA" 16-bit flag.
    Port of phenofit's `qc_StateQA()` (R/qcFUN.R): decodes cloud state
    (bits 0-1), cloud shadow (bit 2), aerosol quantity (bits 6-7) and snow/ice
    (bit 12).

        clear/climatology-aerosol, no shadow, no snow -> wmax (good)
        snow, or cloudy/mixed cloud, or high aerosol   -> wmin (bad)
        everything else                                -> wmid (marginal)

    ``qa``: a numpy array, a ``DataArray`` or a raster path, as in ``qc_modis_summary``.
    """
    def decode(qa):
        qa = np.where(np.isfinite(qa), qa, 0) if np.issubdtype(qa.dtype, np.floating) else qa
        qc_cloud = _get_bits(qa, 0, 1)
        qc_aerosol = _get_bits(qa, 6, 7)
        qc_snow = _get_bits(qa, 12, 12).astype(bool)

        w = np.full(qa.shape, wmid, dtype=np.float64)

        is_good = np.isin(qc_cloud, [0, 3]) & np.isin(qc_aerosol, [0, 1, 2]) & ~qc_snow
        is_bad = qc_snow | np.isin(qc_cloud, [1, 2]) | (qc_aerosol == 3)

        w[is_good] = wmax
        w[is_bad] = wmin
        return w

    return _weights(qa, decode)


def qc_sentinel2_scl(scl: Any, wmin: float = 0.2, wmid: float = 0.5, wmax: float = 1.0) -> Any:
    """
    Weights from the Sentinel-2 L2A Scene Classification Layer (SCL).
    Port of phenofit's `qc_sentinel2()` (R/qcFUN.R).

        4 vegetation, 5 bare soil, 6 water, 7 unclassified, 10 thin cirrus -> wmax
        8 cloud medium probability                                        -> wmid
        1 saturated/defective, 2 dark area, 3 cloud shadow, 9 cloud high  -> wmin
        11 snow (kept usable for phenology, per phenofit)                 -> wmin
        other (e.g. 0 no-data)                                            -> wmin

    ``scl``: a numpy array, a ``DataArray`` or a raster path, as in ``qc_modis_summary``
    (a NoData stored with the raster, often 0, gets 0).
    """
    qc_good = (4, 5, 6, 7, 10)
    qc_mid = (8,)

    def decode(scl):
        w = np.full(scl.shape, wmin, dtype=np.float64)
        w[np.isin(scl, qc_good)] = wmax
        w[np.isin(scl, qc_mid)] = wmid
        return w

    return _weights(scl, decode)
