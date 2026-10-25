"""``zeit.ewmacd``: Exponentially Weighted Moving Average Change Detection (EWMACD; Brooks et al.
2014), the continuous monitoring of an irregular series against a harmonic model of its training
years, with statistical quality control charts. The engine is ``_core.ewmacd``, in Rust.
"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._load import sorted_by_time
from ._series_api import _series_cube
from ._threads import resolve
from ._time import fractional_years

NAMES = ("break_time", "severity", "recovery_time", "gain_time", "state", "sigma", "rmse", "r2")
STATES = ("stable", "disturbed", "recovered", "gain")


def ewmacd(
    data: Any,
    *,
    train_end: int,
    train_start: Optional[int] = None,
    order: int = 2,
    lmbda: float = 0.3,
    sigmas: float = 3.0,
    persistence: int = 3,
    xbar_train: float = 1.5,
    xbar_monitor: float = 20.0,
    rounding: bool = True,
    low: Optional[float] = None,
    dates: Any = None,
    band: Union[str, int, None] = None,
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """EWMACD (Brooks et al. 2014): disturbances, gains and recovery from every observation of a
    vegetation index (Landsat, Sentinel-2; irregular, cloudy dates welcome), with EWMA control
    charts of the residuals from a harmonic model of the training years.

    Per pixel, from ``train_start`` on:

    1. a harmonic regression (a constant and ``order`` sine and cosine pairs of the day of the
       year) on the training years (before ``train_end``), refitted without the dates beyond
       ``xbar_train`` standard deviations of its residuals (a Shewhart X-bar chart: the clouds
       and shadows of the training years);
    2. the residuals of every date, screened once more: training dates beyond ``xbar_train``
       and later dates beyond ``xbar_monitor`` standard deviations are left out of the chart
       (only gross anomalies: the changes must stay in); the standard deviation of the training
       residuals kept, ``sigma``, scales the control limits;
    3. the EWMA chart ``z_j = (1 - lmbda) z_(j-1) + lmbda r_j`` (``z_1 = r_1``) and its control
       limits ``sigmas * sigma * sqrt(lmbda / (2 - lmbda) * (1 - (1 - lmbda)^(2j)))``;
    4. the flags, ``z / limit`` truncated toward zero: 0 inside the limits, -1, -2... a loss of
       the index (a disturbance) of 1, 2... times the limit, positive a gain;
    5. ``persistence``: a signal must last that many dates; a shorter run takes the flag of
       the last run that lasted;
    6. dates left out (missing, screened) carry the flag of the date before.

    The flags are those of Brooks' EWMACD v1.3.0 script (the paper's; Earth Engine's
    ``ee.Algorithms.TemporalSegmentation.Ewmacd``) with ``persistence=1``; with more, the first
    disturbance is the same and a few later dates differ, where the script, meaning to keep the
    previous sustained state, reads another date's flag. Unlike the script, values not above
    ``low`` are missing from the start (the script fits them in its first regression).

    Parameters
    ----------
    data
        One index (NDVI, NBR, the tasseled cap angle of the paper...): a ``(time, y, x)`` cube,
        a raster, a Dataset with ``band=``, or one pixel's series (``pd.Series`` with dates, a
        1-D DataArray with ``time``, or values with ``dates=``). Every observation counts: no
        compositing.
    train_end
        The first year after the training period (exclusive, as Brooks' ``trainingEnd`` and
        Earth Engine's ``trainingEndYear``): the model is fitted on the years before it and
        monitoring starts with it. The training years should be stable (a pixel disturbed in
        them signals at once after them).
    train_start
        The first year looked at (earlier dates are left out). Default: the first year.
    order
        Sine and cosine pairs of the harmonic model (2, as Brooks et al.).
    lmbda
        Weight of each new residual in the EWMA, in (0, 1]: low values are slow and smooth
        (subtle changes, few false signals), 1 a Shewhart chart of each date (0.3).
    sigmas
        The control limits, in standard deviations of the EWMA (3).
    persistence
        Dates a signal must last (3, as Earth Engine; Brooks' v1.9 script: half the dates of a
        year).
    xbar_train, xbar_monitor
        The X-bar screens, in standard deviations: 1.5 for the training dates, 20 after them.
        An abrupt change larger than ``xbar_monitor`` standard deviations of a quiet pixel (a
        clearcut on a smooth series) is screened as an anomaly until it comes back within it:
        raise it for such series.
    rounding
        Flags truncated to integers (multiples of the limit); else the ratio ``z / limit``
        itself beyond ±1 (0 inside).
    low
        Values not above it are missing (no vegetation, fill values; Brooks' ``lowthresh``).
    dates
        The dates of a series without them.
    band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        - ``flags (time, y, x)``: the chart's flags (NaN before ``train_start`` or where no model
          could be fitted);
        - ``break_time``: the first disturbance after the training years (decimal year of its
          first negative flag), ``severity`` its most negative flag until it recovers,
          ``recovery_time`` the first date back inside the limits (or above) after it,
          ``gain_time`` the first positive flag; NaN where none;
        - ``state``: at the last date, 0 stable, 1 disturbed, 2 recovered (disturbed, then back),
          3 gain (``flag_meanings``);
        - ``sigma`` (the residuals' standard deviation behind the limits), ``rmse`` and ``r2``
          of the harmonic model on its training dates, and its ``coefficients`` (constant, the
          sines, the cosines);
        - for one pixel's series, also ``ewma``, ``limit`` and ``fitted`` per date (the chart, its
          limits, NaN where the date was left out of the chart, and the model, at every date).

    Examples
    --------
    >>> ndvi = zeit.load_raster("landsat_ndvi/*.tif")            # every scene, 2013-2023
    >>> res = zeit.ewmacd(ndvi, train_end=2017)                  # trained on 2013-2016
    >>> res.break_time.plot()
    >>> res.state.where(res.state == 1).count()                  # pixels disturbed now

    References
    ----------
    Brooks, E. B., Wynne, R. H., Thomas, V. A., Blinn, C. E., & Coulston, J. W. (2014). On-the-fly
    massively multitemporal change detection using statistical quality control charts and Landsat
    data. IEEE Transactions on Geoscience and Remote Sensing, 52(6), 3316-3332.
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.ewmacd")
    if not 0 < lmbda <= 1:
        raise ValueError(f"lmbda must be in (0, 1], got {lmbda!r}")
    if int(order) < 1:
        raise ValueError(f"order must be at least 1, got {order!r}")
    cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, nodata=nodata)
    cube = sorted_by_time(cube)
    times = pd.DatetimeIndex(cube.time.values)
    years = times.year.to_numpy().astype(np.int32)
    doys = times.dayofyear.to_numpy().astype(np.float64)
    t = np.asarray(fractional_years(times), dtype=np.float64)
    start = int(years[0]) if train_start is None else int(train_start)
    end = int(train_end)
    if end <= start:
        raise ValueError(f"train_end ({end}) must come after train_start ({start})")
    if not np.any((years >= start) & (years < end)):
        raise ValueError(f"no date in the training years {start}-{end - 1}")
    kw = dict(harmonics=int(order), xbar_train=float(xbar_train), xbar_monitor=float(xbar_monitor),
              lmbda=float(lmbda), sigmas=float(sigmas), rounding=bool(rounding), persistence=max(int(persistence), 1),
              low=-np.inf if low is None else float(low))
    n_coef = 1 + 2 * int(order)
    coef_names = ["constant"] + [f"sin{k}" for k in range(1, int(order) + 1)] + \
        [f"cos{k}" for k in range(1, int(order) + 1)]
    attrs = dict(algorithm="EWMACD", train_start=start, train_end=end, order=int(order), lmbda=float(lmbda),
                 sigmas=float(sigmas), persistence=int(persistence), xbar_train=float(xbar_train),
                 xbar_monitor=float(xbar_monitor), rounding=int(bool(rounding)))

    if pixel:
        values = np.asarray(cube.values, dtype=np.float64).reshape(len(times), -1)[:, 0]
        e = _core.ewmacd.ewmacd_pixel(values, years, doys, t, start, end, **kw)
        ds = xr.Dataset({k: ("time", np.asarray(e[k], dtype=np.float64)) for k in ("flags", "ewma", "limit", "fitted")},
                        coords={"time": times})
        for k in NAMES:
            ds[k] = float(e[k])
        beta = np.asarray(e["beta"], dtype=np.float64)
        ds["coefficients"] = ("coefficient", beta if beta.size else np.full(n_coef, np.nan))
        ds = ds.assign_coords(coefficient=coef_names)
        ds["state"].attrs.update(flag_values=[0, 1, 2, 3], flag_meanings=" ".join(STATES))
        ds.attrs = attrs
        return ds

    series = cube.transpose("y", "x", "time")
    jobs = resolve(n_jobs)
    width = len(NAMES) + n_coef

    def _block(v):   # (..., time) -> (..., time + width)
        shape = v.shape[:-1]
        flat = np.ascontiguousarray(v.reshape(-1, v.shape[-1]), dtype=np.float64)
        flags, maps = _core.ewmacd.ewmacd_batch(flat, years, doys, t, start, end, n_jobs=jobs, **kw)
        return np.concatenate([flags, maps.T], axis=1).reshape(shape + (len(t) + width,))

    if series.chunks is not None:
        series = series.chunk({"time": -1})
    out = xr.apply_ufunc(_block, series, input_core_dims=[["time"]], output_core_dims=[["row"]],
                         dask="parallelized", output_dtypes=[np.float64],
                         dask_gufunc_kwargs={"output_sizes": {"row": len(t) + width}})
    nt = len(t)
    ds = xr.Dataset()
    ds["flags"] = out.isel(row=slice(0, nt)).rename(row="time").assign_coords(time=times) \
        .transpose("time", "y", "x").astype(np.float32)
    for k, name in enumerate(NAMES):
        ds[name] = out.isel(row=nt + k, drop=True).astype(np.float64 if name.endswith("_time") else np.float32)
    ds["coefficients"] = out.isel(row=slice(nt + len(NAMES), nt + width)).rename(row="coefficient") \
        .assign_coords(coefficient=coef_names).transpose("coefficient", "y", "x").astype(np.float32)
    ds["state"].attrs.update(flag_values=[0, 1, 2, 3], flag_meanings=" ".join(STATES))
    ds.attrs = attrs
    crs = getattr(cube.rio, "crs", None) if hasattr(cube, "rio") else None
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds
