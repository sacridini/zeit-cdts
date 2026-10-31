"""``zeit.texture``: the texture of an image (or of a series' temporal summary) in a moving window."""

from typing import Any, Optional, Sequence, Union

import numpy as np
import xarray as xr

#: The features ``zeit.texture`` computes, in this order by default.
FEATURES = (
    "contrast", "dissimilarity", "homogeneity", "asm", "energy", "correlation", "entropy",
    "glcm_mean", "glcm_variance", "mean", "std", "gradient",
)

#: Temporal summaries of a cube ``(time, y, x)`` before its texture.
SUMMARIES = ("median", "mean", "min", "max", "std", "amplitude")


def _summary(da: xr.DataArray, how: str) -> xr.DataArray:
    t = "time" if "time" in da.dims else da.dims[0]
    if how == "median":
        return da.median(t, skipna=True)
    if how == "mean":
        return da.mean(t, skipna=True)
    if how == "min":
        return da.min(t, skipna=True)
    if how == "max":
        return da.max(t, skipna=True)
    if how == "std":
        return da.std(t, skipna=True)
    if how == "amplitude":
        return da.quantile(0.95, t, skipna=True).drop_vars("quantile") - da.quantile(
            0.05, t, skipna=True
        ).drop_vars("quantile")
    raise ValueError(f"summary must be one of {', '.join(SUMMARIES)}, not {how!r}")


def texture(
    data: Any,
    *,
    features: Union[str, Sequence[str], None] = None,
    window: int = 7,
    levels: int = 32,
    distance: int = 1,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    summary: str = "median",
    n_jobs: int = -1,
) -> xr.DataArray:
    """Texture features of an image in a moving window: inputs for a classifier.

    The grey-level co-occurrence matrix's properties (Haralick et al. 1973) as
    scikit-image's ``graycomatrix(..., angles=[0, pi/4, pi/2, 3pi/4], symmetric=True,
    normed=True)`` and ``graycoprops`` give them for each pixel's window, averaged over
    the four angles, and three of spfeas's window statistics (Graesser et al. 2012).
    Fields, pastures, orchards and forests differ in texture where their spectra alike.

    Parameters
    ----------
    data
        A map ``(y, x)``, or a cube ``(time, y, x)`` summarized over time first
        (``summary``): a NumPy array or an xarray DataArray (its coordinates kept).
    features
        ``None`` (all twelve), or any of ``contrast``, ``dissimilarity``, ``homogeneity``, ``asm`` (angular second
        moment), ``energy``, ``correlation``, ``entropy`` (base 2), ``glcm_mean``,
        ``glcm_variance`` (the GLCM's), ``mean``, ``std`` (the window's values) and
        ``gradient`` (the window's mean Sobel magnitude).
    window
        The window's side in pixels (odd, at least 3).
    levels
        Grey levels the values are quantized to (2 to 254) over ``[lo, hi]``.
    distance
        The pairs' distance in pixels.
    lo, hi
        The range quantized (values outside it clipped). ``None``: the image's 2nd and
        98th percentiles.
    summary
        A cube's summary over time: ``"median"``, ``"mean"``, ``"min"``, ``"max"``,
        ``"std"`` or ``"amplitude"`` (95th minus 5th percentile).
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.DataArray
        ``(feature, y, x)`` float64, NaN where a window has no valid pair (or value).

    Examples
    --------
    >>> import numpy as np, zeit
    >>> img = np.random.default_rng(0).random((64, 64))
    >>> t = zeit.texture(img, features=["contrast", "entropy"], window=5)
    >>> t.sizes["feature"]
    2
    """
    from ._core import texture as _tex

    feats = list(FEATURES) if features is None else [features] if isinstance(features, str) else list(features)
    da = data if isinstance(data, xr.DataArray) else xr.DataArray(np.asarray(data, dtype=np.float64))
    if da.ndim == 3:
        da = _summary(da, summary)
    if da.ndim != 2:
        raise ValueError(f"data must be a map (y, x) or a cube (time, y, x), not {da.ndim}D")
    img = np.ascontiguousarray(da.values, dtype=np.float64)
    out = _tex.texture_features(
        img,
        feats,
        window=int(window),
        levels=int(levels),
        distance=int(distance),
        lo=float("nan") if lo is None else float(lo),
        hi=float("nan") if hi is None else float(hi),
        n_jobs=int(n_jobs),
    )
    ydim, xdim = da.dims
    coords = {k: v for k, v in da.coords.items() if set(v.dims) <= {ydim, xdim}}
    res = xr.DataArray(out, dims=("feature", ydim, xdim), coords=coords)
    res = res.assign_coords(feature=feats)
    res.attrs.update(
        {
            "window": int(window),
            "levels": int(levels),
            "distance": int(distance),
            "summary": summary if data is not None and np.ndim(data) == 3 else "none",
        }
    )
    if "crs" in da.attrs:
        res.attrs["crs"] = da.attrs["crs"]
    return res
