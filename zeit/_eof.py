"""``zeit.eof``: the temporal empirical orthogonal functions (EOF, principal components) of a cube: the few maps
and the time series that explain most of its variability, for cubes that do not fit in memory. The engine is
``_core.eof``, in Rust.
"""

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np
import xarray as xr

from ._load import sorted_by_time
from ._series_api import _series_cube
from ._threads import resolve

ANOMALIES = {"none": "mean", "month": "month", "harmonic": "harmonic"}


def days_of(time: Any) -> np.ndarray:
    """The dates as days since 1970-01-01 (int64)."""
    from ._time import to_datetime_index

    return np.asarray(to_datetime_index(time).values.astype("datetime64[D]").astype(np.int64), dtype=np.int64)


def _crs_of(cube: Any):
    if not hasattr(cube, "rio"):
        return None
    try:
        return cube.rio.crs
    except Exception:  # noqa: BLE001 - no grid information
        return None


def _weights(spec: Any, cube: xr.DataArray, valid_any: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    """The weights of the pixels, mean 1 over the pixels with data, and their name."""
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    if isinstance(spec, (xr.DataArray, np.ndarray)):
        w = np.asarray(spec, dtype=np.float64)
        if w.shape != (ny, nx):
            raise ValueError(f"weights has the shape {w.shape}, the grid is {(ny, nx)}")
        name = "custom"
    elif spec is None or spec == "none":
        return None, "none"
    elif spec == "area":
        crs = _crs_of(cube)
        if crs is None:
            warnings.warn("the data has no CRS: the weights are 1 (no area weighting)", stacklevel=3)
            return None, "none"
        from .vector import pixel_area

        w = pixel_area(cube).values.astype(np.float64)
        name = "area"
    elif spec == "coslat":
        crs = _crs_of(cube)
        if crs is None:
            warnings.warn("the data has no CRS: the weights are 1 (no area weighting)", stacklevel=3)
            return None, "none"
        if not crs.is_geographic:
            raise ValueError("weights='coslat' is for a geographic CRS (degrees); use weights='area' for a "
                             "projected one")
        w = np.broadcast_to(np.cos(np.deg2rad(cube.y.values))[:, None], (ny, nx)).astype(np.float64)
        name = "coslat"
    else:
        raise ValueError(f"weights must be 'area', 'coslat', 'none' or an array (y, x), got {spec!r}")
    w = w.reshape(-1)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("the weights must be finite and not negative")
    mean = w[valid_any].mean() if valid_any.any() and w[valid_any].mean() > 0 else w.mean()
    return w / mean, name


def eof(
    data: Any,
    n_modes: int = 10,
    *,
    anomaly: str = "month",
    harmonics: int = 2,
    base: Optional[Tuple[int, int]] = None,
    detrend: bool = False,
    standardize: bool = False,
    weights: Any = "area",
    min_valid: float = 0.8,
    min_date_valid: float = 0.5,
    scaling: str = "units",
    solver: str = "auto",
    oversampling: int = 10,
    power_iterations: int = 2,
    rotate: int = 0,
    kaiser: bool = True,
    north_n: str = "dates",
    correlation_maps: bool = False,
    seed: int = 42,
    band: Union[str, int, None] = None,
    nodata: Any = "auto",
    n_jobs: int = -1,
) -> xr.Dataset:
    """Temporal EOF analysis (principal components) of a cube: the spatial patterns (EOFs) and the time series
    (PCs) that explain most of its variance.

    The seasonal cycle is taken out of each pixel (anomalies), the pixels are weighted by their area, and the
    cube X (dates x pixels) is decomposed as ``X = U S V^T``. The dates are the short side: the singular
    vectors come from the T x T Gram matrix (exact) or from a randomized range finder with power iterations
    (Halko, Martinsson & Tropp 2011) in a few passes over the pixels, which are read in blocks, so cubes larger
    than the memory work. The maps come in one more pass. The result does not depend on ``n_jobs``.

    Parameters
    ----------
    data
        One variable (an index): a ``(time, y, x)`` cube, a raster, or a Dataset with ``band=``. Missing values
        are NaN.
    n_modes
        Modes computed (at most the number of dates - 1).
    anomaly
        What is removed from each pixel's series: ``"month"`` (default; the climatology of each calendar month,
        which needs two values of that month at least), ``"harmonic"`` (a constant and ``harmonics`` sine / cosine
        pairs of the year, by least squares) or ``"none"`` (only the mean).
    harmonics, base, detrend, standardize
        The order of the harmonic fit; ``base=(first_year, last_year)``, the years that define the climatology
        (default all); ``detrend`` removes a straight line too; ``standardize`` divides each pixel by the
        standard deviation of its anomalies (the correlation matrix instead of the covariance).
    weights
        ``"area"`` (default): the area of each pixel (``zeit.pixel_area``), normalized to mean 1, so a pixel of a
        high latitude counts for what it covers; ``"coslat"``: cos(latitude), only for a geographic CRS;
        ``"none"``; or an array (y, x). Without a CRS the weights are 1 (a warning).
    min_valid
        A pixel is kept when this fraction of the dates (after the anomalies) is valid; the others have NaN maps.
    min_date_valid
        A date is part of the fit when this fraction of the kept pixels is valid there; the other dates get their
        PC by least squares over their valid pixels (Everson & Sirovich 1995), flagged in ``pc_filled``.
    scaling
        ``"units"`` (default): each EOF in the data's units per standard deviation of its PC (the PCs have unit
        variance); ``"none"``: orthonormal EOFs, the PCs carry the variance.
    solver, oversampling, power_iterations, seed
        ``"auto"`` (exact with up to 256 dates, else randomized), ``"exact"`` or ``"randomized"``; the sketch
        has ``n_modes + oversampling`` columns and ``power_iterations`` power passes; the seed of the random
        test matrix.
    rotate
        Rotate the first ``rotate`` modes (at least 2) by Kaiser's (1958) varimax, with Kaiser normalization
        when ``kaiser``: the patterns become more local. The rotated EOFs are always in data units per standard
        deviation of the PC.
    north_n
        The number of independent dates of North et al.'s (1982) sampling error: ``"dates"`` (default, all
        of them) or ``"autocorrelation"`` (``T (1 - rho) / (1 + rho)`` with the lag-1 autocorrelation of each PC).
    correlation_maps
        Also give ``corrmap``: the correlation of each pixel with each PC.
    band, nodata, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        - ``eof (mode, y, x)``: the patterns; ``pc (mode, time)``: the time series;
        - ``variance (mode)``: the fraction of the total variance of each mode; ``eigenvalue (mode)``:
          ``s_k^2 / (T - 1)`` in the weighted data's units squared;
        - ``north_error (mode)`` and ``separable (mode)``: the sampling error of the eigenvalue and whether the
          mode is separated from its neighbours (North et al. 1982);
        - ``pc_filled (time)``: the PC of the date comes from the gappy projection, not from the fit;
        - with ``rotate``: ``eof_rotated (mode_rotated, y, x)``, ``pc_rotated (mode_rotated, time)`` (unit
          variance) and ``variance_rotated (mode_rotated)``, ordered by variance;
        - with ``correlation_maps``: ``corrmap (mode, y, x)``;
        - attributes ``total_variance`` (the sum of all the eigenvalues), ``solver`` (the one that ran),
          ``weights``, ``anomaly``, ``n_pixels`` and ``n_dates`` (kept in the fit).

    The sign of each mode is fixed so that its PC correlates positively with the area-weighted mean anomaly (a
    dipole: its largest value is positive).

    Examples
    --------
    >>> ndvi = zeit.load_raster("ndvi_16d.tif")                  # (time, y, x)
    >>> ds = zeit.eof(ndvi, n_modes=5, rotate=3)
    >>> ds.variance.values                                       # fraction of variance of each mode
    >>> ds.eof.sel(mode=1).zeit.plot()

    References
    ----------
    North, G. R., Bell, T. L., Cahalan, R. F., & Moeng, F. J. (1982). Sampling errors in the estimation of
    empirical orthogonal functions. Monthly Weather Review, 110, 699-706.
    Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness. SIAM Review, 53,
    217-288. Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis.
    Psychometrika, 23, 187-200. Everson, R., & Sirovich, L. (1995). Karhunen-Loeve procedure for gappy data.
    JOSA A, 12, 1657-1664.
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.eof")
    if anomaly not in ANOMALIES:
        raise ValueError(f"anomaly must be one of {sorted(ANOMALIES)}, got {anomaly!r}")
    if int(n_modes) < 1:
        raise ValueError(f"n_modes must be at least 1, got {n_modes!r}")
    cube, pixel = _series_cube(data, dates=None, band=band, chunks=None, need_dates=True, nodata=nodata)
    if pixel:
        raise ValueError("eof needs a map at each date, not one pixel's series")
    cube = sorted_by_time(cube).transpose("time", "y", "x")
    nt, ny, nx = cube.shape
    values = np.ascontiguousarray(np.asarray(cube.values, dtype=np.float64)).reshape(nt, ny * nx)
    days = days_of(cube.time.values)
    jobs = resolve(n_jobs)
    values = _core.anomaly.anomalies(values, days, method=ANOMALIES[anomaly], order=int(harmonics),
                                     base=None if base is None else (int(base[0]), int(base[1])),
                                     detrend=bool(detrend), standardize=bool(standardize), n_jobs=jobs)
    w, wname = _weights(weights, cube, np.isfinite(values).any(axis=0))
    r = _core.eof.decompose(values, w, modes=int(n_modes), solver=solver, oversampling=int(oversampling),
                            power_iterations=int(power_iterations), seed=int(seed), min_valid=float(min_valid),
                            min_date_valid=float(min_date_valid), scaling=scaling, rotate=int(rotate),
                            kaiser=bool(kaiser), north_n=north_n, correlation_maps=bool(correlation_maps),
                            n_jobs=jobs)
    m = len(r["singular"])
    coords = {c: cube.coords[c] for c in ("time", "y", "x") if c in cube.coords}
    coords["mode"] = np.arange(1, m + 1)
    ds = xr.Dataset(
        {
            "eof": (("mode", "y", "x"), r["eof"].reshape(m, ny, nx)),
            "pc": (("mode", "time"), r["pcs"]),
            "variance": (("mode",), r["explained"]),
            "eigenvalue": (("mode",), r["eigenvalues"]),
            "north_error": (("mode",), r["north_error"]),
            "separable": (("mode",), r["separable"]),
            "pc_filled": (("time",), r["pc_filled"]),
        },
        coords=coords,
    )
    ds["variance"].attrs.update(units="fraction", long_name="explained variance")
    ds["eof"].attrs.update(long_name="empirical orthogonal function",
                           units="data units per standard deviation of the PC" if scaling == "units"
                           else "orthonormal")
    if correlation_maps:
        ds["corrmap"] = (("mode", "y", "x"), r["corr"].reshape(m, ny, nx).astype(np.float32))
        ds["corrmap"].attrs.update(long_name="correlation of each pixel with each PC")
    if "rotated_eof" in r:
        mr = r["rotated_eof"].shape[0]
        ds = ds.assign_coords(mode_rotated=np.arange(1, mr + 1))
        ds["eof_rotated"] = (("mode_rotated", "y", "x"), r["rotated_eof"].reshape(mr, ny, nx))
        ds["pc_rotated"] = (("mode_rotated", "time"), r["rotated_pcs"])
        ds["variance_rotated"] = (("mode_rotated",), r["rotated_variance"])
        ds["variance_rotated"].attrs.update(units="fraction")
    ds.attrs = dict(algorithm="temporal EOF", total_variance=float(r["total_variance"]), solver=r["solver"],
                    weights=wname, anomaly=anomaly, scaling=scaling,
                    n_pixels=int(r["pixel_kept"].sum()), n_dates=int(r["date_kept"].sum()))
    crs = _crs_of(cube)
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds
