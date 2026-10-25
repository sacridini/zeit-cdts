"""Trends and their regional test with remotePARTS's methods (Ives et al. 2021, Remote Sensing of
Environment 266, 112678, doi:10.1016/j.rse.2021.112678), on the compiled core.

``ar_trend`` fits each pixel's series to a linear trend with AR(1) errors (REML): a t-test of
the trend that accounts for the autocorrelation of the series, an alternative to Mann-Kendall's.
``parts`` tests the trends of a region: whether their mean differs from 0, whether they differ
between classes, whether they depend on covariates, accounting for the spatial correlation of
the pixels (what counting significant pixels does not), by GLS on random partitions of the
pixels (PARTS).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._series_api import _as_dask, _metrics_dataset, _series_cube
from ._threads import resolve
from ._time import fractional_years


def _per_year(cube: xr.DataArray) -> Optional[int]:
    """Observations per year, from the median spacing of the dates."""
    if "time" not in cube.coords or cube.sizes["time"] < 2:
        return None
    years = fractional_years(pd.DatetimeIndex(cube.time.values))
    step = np.median(np.diff(years))
    return max(1, int(round(1.0 / step))) if step > 0 else None


def _design(cube: xr.DataArray, time: str, harmonics: int, period: Optional[float]):
    """The design shared by the pixels (time, q) and the names of its columns."""
    n = cube.sizes["time"]
    if time == "step":
        t = np.arange(1, n + 1, dtype=float)
    elif time == "year":
        if "time" not in cube.coords:
            raise ValueError('time="year" needs dates: pass dates=')
        t = np.asarray(fractional_years(pd.DatetimeIndex(cube.time.values)), dtype=float)
    else:
        raise ValueError(f'time must be "step" or "year", got {time!r}')
    columns, names = [np.ones(n), t], ["intercept", "slope"]
    if harmonics:
        if time == "year":
            phase = 2 * np.pi * t  # cycles per year
        else:
            period = period or _per_year(cube)
            if not period:
                raise ValueError("harmonics need period= (observations per cycle) or dates")
            phase = 2 * np.pi * np.arange(n) / period
        for k in range(1, harmonics + 1):
            columns += [np.cos(k * phase), np.sin(k * phase)]
            names += [f"cos{k}", f"sin{k}"]
    return np.column_stack(columns), names


def _metric_names(names: List[str]) -> List[str]:
    return names + [f"{c}_se" for c in names] + [f"{c}_t" for c in names] + [f"{c}_p" for c in names] + [
        "rho", "mse", "loglik"]


def ar_trend(data: Any, *, time: str = "step", harmonics: int = 0, period: Optional[float] = None,
             gaps: str = "exact", min_valid: int = 0, dates: Optional[Sequence[Any]] = None,
             band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto", chunks: Any = None,
             n_jobs: int = -1) -> xr.Dataset:
    """Linear trend with AR(1) errors, fitted by REML, per pixel (remotePARTS's ``fitAR``).

    The series is ``intercept + slope * t`` (plus harmonics) with errors that follow an AR(1)
    process, whose coefficient ``rho`` is estimated with the trend; the t-test of the slope
    accounts for that autocorrelation. It is the per-pixel step of ``zeit.parts``.

    Parameters
    ----------
    data
        As for ``zeit.mann_kendall``: a ``(time, y, x)`` cube, a raster, or one pixel's series.
    time
        ``"step"`` (default, remotePARTS's ``t = 1, 2, ...``: the slope is per time step) or
        ``"year"`` (decimal years of the dates: the slope is per year).
    harmonics, period
        Pairs of harmonics (cos, sin) of the seasonal cycle in the design, for sub-annual
        series; ``period`` is the observations per cycle for ``time="step"`` (from the dates
        when not given).
    gaps
        ``"exact"`` (default): a missing date counts, observations k dates apart are
        correlated by rho^k. ``"ignore"``: the observations are taken as consecutive, as
        remotePARTS's ``fitAR`` does.
    min_valid
        Minimum valid observations (at least the design's columns + 2).
    dates, band, nodata, chunks, n_jobs
        As for the BFAST functions.

    Returns
    -------
    xarray.Dataset
        For each coefficient (``intercept``, ``slope``, ``cos1``, ``sin1``, ...): its value,
        ``_se``, ``_t`` and ``_p`` (two-sided); ``rho``, ``mse`` (the innovations' variance)
        and ``loglik``.
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.ar_trend")
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, need_dates=(time == "year"),
                               nodata=nodata)
    design, names = _design(cube, time, harmonics, period)
    metrics = _metric_names(names)
    arr, lazy = _as_dask(cube)
    jobs = resolve(n_jobs)

    def _block(block):
        t, rows, cols = block.shape
        if block.size == 0:
            return np.full((len(metrics), rows, cols), np.nan, dtype=np.float32)
        values = np.ascontiguousarray(block.reshape(t, rows * cols).T, dtype=np.float64)
        out = _core.parts.fit_ar_batch(values, design, gaps=gaps, min_valid=min_valid, n_jobs=jobs)
        return out.reshape(len(metrics), rows, cols).astype(np.float32)

    import dask.array as da

    out = da.map_blocks(_block, arr, dtype=np.float32, drop_axis=[0], new_axis=[0],
                        chunks=(len(metrics), arr.chunks[1], arr.chunks[2]))
    return _metrics_dataset(out, metrics, cube, pixel, lazy, dict(
        algorithm="AR(1) REML trend (remotePARTS fitAR)", time=time, harmonics=harmonics, gaps=gaps))


@dataclass
class PartsResult:
    """The regional test of ``zeit.parts``.

    ``table``: one row per coefficient of the design (the mean trend, a class's mean trend, a
    covariate's effect), its estimate, standard error, t and p of the correlated t-test.
    ``p_chisqr``: the correlated chi-square test of the design against a single mean (that the
    classes differ, that the covariates have an effect); None without classes or covariates.
    """
    table: pd.DataFrame
    p_chisqr: Optional[float]
    f: float
    df1: int
    df2: int
    correlation: Dict[str, Any]
    nuggets: np.ndarray
    npart: int
    partsize: int
    n_pixels: int
    rank_deficient: bool
    partitions: pd.DataFrame
    details: Dict[str, Any] = field(repr=False, default_factory=dict)

    def __repr__(self) -> str:
        lines = [f"PARTS: {self.n_pixels} pixels, {self.npart} partitions of {self.partsize}",
                 f"spatial correlation: {self.correlation}",
                 f"nugget (mean of the partitions): {np.nanmean(self.nuggets):.4g}",
                 "", self.table.to_string(float_format=lambda v: f"{v:.6g}")]
        if self.p_chisqr is not None:
            note = " (rank-deficient covariance, replaced as remotePARTS does)" if self.rank_deficient else ""
            # remotePARTS floors the p-value at 1e-6
            p = "p <= 1e-06" if self.p_chisqr <= 1e-6 else f"p = {self.p_chisqr:.4g}"
            lines += ["", f"chi-square test against a single mean: F = {self.f:.4g}, df = ({self.df1}, "
                          f"{self.df2}), {p}{note}"]
        return "\n".join(lines)


def _coordinates(cube: xr.DataArray):
    crs = cube.rio.crs if hasattr(cube, "rio") else None
    if crs is not None and crs.is_geographic:
        raise ValueError("zeit.parts measures distances in the cube's units: reproject a cube in geographic "
                         "coordinates first (e.g. zeit.load_raster(..., crs='EPSG:...'))")
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    xs = np.asarray(cube.x.values, dtype=float) if "x" in cube.coords else np.arange(nx, dtype=float)
    ys = np.asarray(cube.y.values, dtype=float) if "y" in cube.coords else np.arange(ny, dtype=float)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def _map(value: Any, cube: xr.DataArray, what: str) -> np.ndarray:
    """A (y, x) layer as the cube's pixels, flattened."""
    values = np.asarray(value.values if isinstance(value, xr.DataArray) else value)
    if values.shape != (cube.sizes["y"], cube.sizes["x"]):
        raise ValueError(f"{what} must have the cube's (y, x) shape {(cube.sizes['y'], cube.sizes['x'])}, "
                         f"got {values.shape}")
    return values.ravel()


def parts(data: Any, *, trend: Optional[xr.Dataset] = None, response: str = "slope", classes: Any = None,
          covariates: Optional[Dict[str, Any]] = None, mask: Any = None, partsize: int = 1500,
          npart: Optional[int] = None, ncross: int = 6, nugget: Union[str, float] = "ml",
          correlation: str = "exponential", range: Optional[float] = None, shape: Optional[float] = None,
          fit_n: int = 1000, scaled: bool = False, seed: int = 0, time: str = "step", harmonics: int = 0,
          period: Optional[float] = None, gaps: str = "exact", dates: Optional[Sequence[Any]] = None,
          band: Union[str, int, None] = None, nodata: Union[float, str, None] = "auto",
          n_jobs: int = -1) -> PartsResult:
    """The regional test of the pixels' trends by PARTS (remotePARTS's ``fitGLS_partition``).

    1. Each pixel's trend by ``zeit.ar_trend`` (or ``trend=``, its result).
    2. The spatial correlation of the pixels: a function of distance (``correlation``) fitted
       to the correlations between the residual series of ``fit_n`` sampled pixels
       (remotePARTS's ``fitCor``), unless ``range=`` (and ``shape=``) give it.
    3. The pixels drawn into ``npart`` partitions of ``partsize`` (all of them by default),
       a GLS of the trends on the design in each, and the tests that combine them accounting
       for the correlation between partitions (measured on ``ncross`` of them).

    Parameters
    ----------
    data
        A ``(time, y, x)`` cube in projected coordinates (distances are in its units), or
        anything ``zeit.ar_trend`` takes.
    trend, response
        A ``zeit.ar_trend`` result to test instead of fitting it, and its variable tested
        (``"slope"``).
    classes
        A ``(y, x)`` layer of classes (land cover, ...): one mean trend per class, and the
        chi-square test that they differ.
    covariates
        ``{name: (y, x) layer}``: their effects on the trend (with a mean), and the
        chi-square test of them all.
    mask
        A ``(y, x)`` boolean layer of the pixels tested (an ROI).
    partsize, npart, ncross, seed
        The partitions (remotePARTS uses 1500 to 3000 pixels), how many (all the pixels by
        default; a few hundred are often enough: the correlation between partitions bounds
        what more of them gain), how many are crossed, and the seed of the draw.
    nugget
        ``"ml"`` (default): each partition's maximum-likelihood nugget; or a number.
    correlation, range, shape, fit_n, scaled
        ``"exponential"`` (exp(-d / range)) or ``"exponential_power"`` (exp(-(d / range)^shape));
        its parameters when known; the pixels sampled to fit them; distances divided by
        their largest in each matrix (remotePARTS's default ``distm_scaled``) or in the
        cube's units (default).
    time, harmonics, period, gaps
        As for ``zeit.ar_trend``.

    Returns
    -------
    PartsResult
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.parts")
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=None, need_dates=(time == "year"),
                               nodata=nodata)
    if pixel:
        raise ValueError("zeit.parts tests a region: give a (time, y, x) cube")
    cube = cube.compute() if cube.chunks is not None else cube
    jobs = resolve(n_jobs)
    design_t, names = _design(cube, time, harmonics, period)
    if trend is None:
        trend = ar_trend(cube, time=time, harmonics=harmonics, period=period, gaps=gaps, n_jobs=n_jobs)
    if response not in trend:
        raise ValueError(f"trend has no variable {response!r}")
    y = _map(trend[response], cube, "trend")
    px, py = _coordinates(cube)

    # the pixels tested: a trend, the layers defined, in the mask
    ok = np.isfinite(y)
    if mask is not None:
        ok &= _map(mask, cube, "mask").astype(bool)
    columns, colnames = [], []
    if classes is not None:
        cls = _map(classes, cube, "classes")
        cls_ok = pd.notna(cls)
        if np.issubdtype(np.asarray(cls).dtype, np.floating):
            cls_ok &= np.isfinite(cls.astype(float))
        ok &= cls_ok
        levels = sorted(pd.unique(cls[ok]))
        for level in levels:
            columns.append((cls == level).astype(float))
            colnames.append(f"class {level}")
    if covariates:
        if not columns:
            columns.append(np.ones_like(y))
            colnames.append("intercept")
        for name, layer in covariates.items():
            values = _map(layer, cube, f"covariate {name}").astype(float)
            ok &= np.isfinite(values)
            columns.append(values)
            colnames.append(name)
    if not columns:
        columns, colnames = [np.ones_like(y)], ["mean"]
    idx = np.flatnonzero(ok)
    n = idx.size
    design = np.column_stack(columns)[idx]
    design0 = np.ones((n, 1))
    yv, xv, yy = y[idx].astype(float), px[idx], py[idx]
    keep = design.any(axis=0) if classes is not None else np.ones(design.shape[1], bool)
    design, colnames = design[:, keep], [c for c, k in zip(colnames, keep) if k]

    # the spatial correlation, from the residuals of a sample
    if range is None:
        rng = np.random.default_rng(seed)
        sample = rng.choice(n, size=min(fit_n, n), replace=False)
        series = cube.values.reshape(cube.sizes["time"], -1)[:, idx[sample]].T
        resids = _core.parts.ar_residuals(np.ascontiguousarray(series, dtype=np.float64), design_t, gaps=gaps)
        good = np.isfinite(resids).sum(axis=1) >= 3
        span = np.hypot(np.ptp(xv[sample]), np.ptp(yv[sample])) or 1.0
        start = 0.1 if scaled else 0.1 * span
        fit = _core.parts.fit_correlation(resids[good], xv[sample][good], yy[sample][good],
                                          correlation=correlation, range=start, shape=1.0, scaled=scaled,
                                          n_jobs=jobs)
        range, shape = fit["range"], fit["shape"]
    cor = {"function": correlation, "range": float(range)}
    if correlation == "exponential_power":
        cor["shape"] = float(shape if shape is not None else 1.0)

    # the partitions
    partsize = min(partsize, n)
    most = n // partsize
    npart = most if npart is None else min(npart, most)
    if npart < 2:
        raise ValueError(f"{n} pixels make {npart} partition(s) of {partsize}: lower partsize= so that "
                         "there are at least two")
    pm = _core.parts.sample_partitions(n, partsize, npart, seed)
    out = _core.parts.fit_parts(xv, yy, yv, design, design0, pm, correlation=correlation, range=cor["range"],
                                shape=cor.get("shape", 1.0), scaled=scaled,
                                nugget=None if nugget == "ml" else float(nugget), ncross=ncross, n_jobs=jobs)
    table = pd.DataFrame({"estimate": out["coefficients"], "se": out["se"], "t": out["t"], "p": out["p"]},
                         index=pd.Index(colnames, name="coefficient"))
    per_part = pd.DataFrame(out["part_coefficients"], columns=colnames)
    per_part["nugget"] = out["part_nugget"]
    p_chisqr = out["p_chisqr"] if design.shape[1] > 1 else None
    return PartsResult(table=table, p_chisqr=p_chisqr, f=out["f"], df1=out["df1"], df2=out["df2"],
                       correlation=cor, nuggets=out["part_nugget"], npart=npart, partsize=partsize, n_pixels=n,
                       rank_deficient=bool(out["rank_deficient"]), partitions=per_part, details=out)
