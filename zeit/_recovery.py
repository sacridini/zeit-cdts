"""Recovery after a disturbance: the spectral recovery metrics of the forest literature (Kennedy et
al. 2012; White et al. 2017; Frazier et al. 2018), per pixel, on the compiled core.

``recovery_metrics`` measures, after each pixel's disturbance (LandTrendr's event, or any event of
``zeit.extract_events``), how much of the loss came back and how fast: the absolute and relative
recovery (ARI, RRI), the ratio of 80% of the pre-disturbance value (R80P), the year-on-year
recovery (YrYr) and the years to recovery (Y2R).
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._series_api import _series_cube
from ._threads import resolve
from ._time import years as _years_of

NAMES = ("pre_val", "dist_val", "magnitude", "post_val", "ari", "rri", "r80p", "yryr", "y2r")


def recovery_metrics(data: Any, events: Any = None, *, target: float = 0.8, horizon: int = 5, pre_years: int = 2,
                     basis: str = "pre_value", event_type: Optional[str] = None, sort_by: str = "greatest",
                     min_magnitude: float = 0.0, min_duration: int = 1, pre_val_threshold: float = 0.0,
                     years: Optional[Sequence[int]] = None, band: Union[str, int, None] = None,
                     nodata: Union[float, str, None] = "auto", chunks: Any = None, n_jobs: int = -1) -> xr.Dataset:
    """How each pixel recovered after its disturbance.

    The disturbance runs from its start, the last year before it (an event's ``yod``), to its
    trough, the year it bottoms out (``yod + duration``; for the algorithms that find abrupt
    breaks, the year after ``yod``). With ``t0`` the trough:

    - ``pre_val``: the mean of the ``pre_years`` years up to the start (NBR_pre);
    - ``dist_val``: the value at ``t0``; ``magnitude``: ``pre_val - dist_val`` (ΔNBR);
    - ``post_val``: the best of years ``t0 + horizon - 1`` and ``t0 + horizon`` (the maximum of
      years 4 and 5 after the trough, Frazier et al. 2018);
    - ``ari``, the absolute recovery (ΔNBR_regrowth): ``post_val - dist_val``;
    - ``rri``, the relative recovery indicator (Kennedy et al. 2012): ``ari / magnitude``;
    - ``r80p``: ``post_val / (target * pre_val)``, 1 or more when recovered (White et al. 2017);
    - ``yryr``, the year-on-year recovery: ``(value at t0 + horizon - dist_val) / horizon``;
    - ``y2r``, the years to recovery (White et al. 2017): from ``t0`` until the value first
      reaches ``target * pre_val``; 0 when it never fell below it, NaN when it has not reached
      it by the last year.

    The metrics measured at the horizon are NaN when the series ends before it.

    Parameters
    ----------
    data
        A Dataset of ``zeit.landtrendr``: the metrics are measured on its fitted trajectory (the
        line through the vertices; ``fitted`` when it has it), after the event that
        ``zeit.extract_events`` picks (``event_type``, ``sort_by`` and the thresholds below), or
        after ``events`` when given. Or an annual series (one observation per year): a
        ``(time, y, x)`` cube, a raster, or one pixel's series, as for ``zeit.mann_kendall``,
        with ``events``.
    events
        The disturbances: a Dataset of ``zeit.extract_events`` (``yod`` and ``duration``), or the
        start year of each pixel (a map, an array or a number; the trough is the year after).
        A start of 0 or NaN: no disturbance (NaN metrics).
    target
        The fraction of the pre-disturbance value that counts as recovered (0.8).
    horizon
        The years after the trough at which ``post_val``, ``ari``, ``rri``, ``r80p`` and
        ``yryr`` are measured (5).
    pre_years
        The years averaged for ``pre_val`` (2: the two years before the disturbance).
    basis
        ``"pre_value"`` (default): recovered at ``target * pre_val``, for an index that is
        positive and drops with the disturbance (NBR, NDVI, TCW...). ``"magnitude"``: recovered
        once ``target`` of the loss came back (``dist_val + target * magnitude``), for any index;
        ``r80p`` is then the share of that regained (``rri / target``).
    event_type
        ``"loss"`` (the index drops with the disturbance) or ``"gain"`` (it rises: the signs of
        ``magnitude``, ``ari`` and ``yryr`` turn, ``post_val`` is the lower of the two years;
        needs ``basis="magnitude"``). Default: the direction LandTrendr ran with, else "loss".
    sort_by, min_magnitude, min_duration, pre_val_threshold
        LandTrendr only, without ``events``: which event ``zeit.extract_events`` keeps.
    years
        One pixel's series without dates: its years.
    band, nodata, chunks, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        ``pre_val``, ``dist_val``, ``magnitude``, ``post_val``, ``ari``, ``rri``, ``r80p``,
        ``yryr`` and ``y2r`` (float32), plus ``yod`` and ``trough`` (the disturbance's years, 0
        without one), on the grid of the input.

    Examples
    --------
    >>> lt = zeit.landtrendr(nbr, direction="loss")
    >>> rec = zeit.recovery_metrics(lt)                          # after the greatest loss
    >>> rec.y2r.plot()
    >>> ev = zeit.extract_events(zeit.ccdc(bands), band="swir2")
    >>> rec = zeit.recovery_metrics(annual_nbr, ev, basis="magnitude")
    """
    from ._embeddings import refuse
    from . import _core

    if basis not in ("pre_value", "magnitude"):
        raise ValueError(f'basis must be "pre_value" or "magnitude", got {basis!r}')
    if not 0 < target:
        raise ValueError(f"target must be positive, got {target!r}")
    if int(horizon) < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon!r}")

    if isinstance(data, xr.Dataset) and "vertex_year" in data and "vertex_value" in data:
        if event_type is None:
            event_type = data.attrs.get("direction", "loss")
        series, year_values, pixel = _landtrendr_series(data, n_jobs)
        if events is None:
            from .metrics import extract_events

            events = extract_events(data, event_type=event_type, sort_by=sort_by, min_magnitude=min_magnitude,
                                    min_duration=min_duration, pre_val_threshold=pre_val_threshold)
        like = series
    else:
        if events is None:
            raise ValueError("events= is needed for a series (or pass the Dataset of zeit.landtrendr)")
        refuse(data, "zeit.recovery_metrics")
        dates = None
        if years is not None:
            dates = pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1) for y in years])
        cube, pixel = _series_cube(data, dates=dates, band=band, chunks=chunks, nodata=nodata)
        year_values = _years_of(cube.time.values)
        if len(np.unique(year_values)) != len(year_values):
            raise ValueError("the recovery metrics need one observation per year, but some years repeat; build "
                             "annual composites first (zeit.build_annual_composites)")
        if np.any(np.diff(year_values) <= 0):
            cube = cube.sortby("time")
            year_values = _years_of(cube.time.values)
        series = cube.transpose("y", "x", "time")
        like = series
    event_type = "loss" if event_type is None else event_type
    if event_type not in ("loss", "gain"):
        raise ValueError(f'event_type must be "loss" or "gain", got {event_type!r}')
    if event_type == "gain" and basis == "pre_value":
        raise ValueError('basis="pre_value" (a ratio of the pre-disturbance value) is for losses; '
                         'use basis="magnitude" for gains')

    start, trough = _event_years(events, like)
    year_values = np.asarray(year_values, dtype=np.int32)
    jobs = resolve(n_jobs)

    def _block(values, starts, troughs):
        shape = starts.shape
        v = np.ascontiguousarray(values.reshape(-1, values.shape[-1]), dtype=np.float64)
        out = _core.recovery.recovery_batch(v, year_values, starts.reshape(-1).astype(np.int32),
                                            troughs.reshape(-1).astype(np.int32), target=float(target),
                                            horizon=int(horizon), pre_years=int(pre_years), basis=basis,
                                            loss=event_type == "loss", n_jobs=jobs)
        return np.moveaxis(out.reshape((len(NAMES),) + shape), 0, -1).astype(np.float32)

    if series.chunks is not None:
        series = series.chunk({"time": -1})
    out = xr.apply_ufunc(_block, series, start, trough, input_core_dims=[["time"], [], []],
                         output_core_dims=[["metric"]], dask="parallelized", output_dtypes=[np.float32],
                         dask_gufunc_kwargs={"output_sizes": {"metric": len(NAMES)}}, join="override")
    ds = out.assign_coords(metric=list(NAMES)).to_dataset("metric")
    ds["yod"] = start.astype(np.int32)
    ds["trough"] = xr.where(start > 0, trough, 0).astype(np.int32)
    ds = ds[["yod", "trough"] + list(NAMES)]
    ds.attrs = dict(algorithm="Recovery metrics", target=float(target), horizon=int(horizon),
                    pre_years=int(pre_years), basis=basis, event_type=event_type)
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    crs = getattr(like.rio, "crs", None) if hasattr(like, "rio") else None
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds


def _landtrendr_series(lt: xr.Dataset, n_jobs: int):
    """LandTrendr's fitted trajectory (y, x, time) at each year, its years, and True for one pixel."""
    from . import _core

    pixel = "y" not in lt.vertex_year.dims
    if "fitted" in lt and "time" in lt.coords:
        series = lt.fitted
        year_values = _years_of(lt.time.values)
    else:
        if "first_year" not in lt.attrs or "last_year" not in lt.attrs:
            raise ValueError("the LandTrendr Dataset lost its first_year and last_year attributes")
        year_values = np.arange(int(lt.attrs["first_year"]), int(lt.attrs["last_year"]) + 1)
        vy, vv = lt.vertex_year, lt.vertex_value
        counts = lt.n_vertices if "n_vertices" in lt else (vy > 0).sum("vertex")
        years32 = year_values.astype(np.int32)

        def _traj(vertex_years, vertex_values, n):
            shape = n.shape
            k = vertex_years.shape[-1]
            out = _core.recovery.trajectory_batch(
                np.ascontiguousarray(vertex_years.reshape(-1, k), dtype=np.float64),
                np.nan_to_num(np.ascontiguousarray(vertex_values.reshape(-1, k), dtype=np.float64)),
                n.reshape(-1).astype(np.int32), years32, n_jobs=resolve(n_jobs))
            return out.reshape(shape + (len(years32),))

        if vy.chunks is not None:
            vy, vv = vy.chunk({"vertex": -1}), vv.chunk({"vertex": -1})
        series = xr.apply_ufunc(_traj, vy, vv, counts, input_core_dims=[["vertex"], ["vertex"], []],
                                output_core_dims=[["time"]], dask="parallelized", output_dtypes=[np.float64],
                                dask_gufunc_kwargs={"output_sizes": {"time": len(year_values)}})
    if pixel:
        series = series.expand_dims(("y", "x"))
    series = series.transpose("y", "x", "time")
    if lt.rio.crs is not None and not pixel:
        series = series.rio.write_crs(lt.rio.crs)
    return series, year_values, pixel


def _event_years(events: Any, like: xr.DataArray):
    """The start and trough years (int32, 0: none) on the (y, x) grid of ``like``."""
    if isinstance(events, xr.Dataset):
        if "yod" not in events:
            raise ValueError("events: a Dataset of zeit.extract_events (with yod) or a map of start years")
        yod = events["yod"]
        duration = events["duration"] if "duration" in events else 1
    else:
        yod, duration = events, 1

    def _grid(v):
        if isinstance(v, xr.DataArray):
            if v.ndim == 0:
                return v
            return v.transpose("y", "x") if set(v.dims) == {"y", "x"} else v
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            return xr.DataArray(arr)
        if arr.shape != (like.sizes["y"], like.sizes["x"]):
            raise ValueError(f"events of shape {arr.shape} for a grid of {(like.sizes['y'], like.sizes['x'])}")
        return xr.DataArray(arr, dims=("y", "x"), coords={k: like.coords[k] for k in ("y", "x") if k in like.coords})

    yod = _grid(yod).astype(np.float64)
    duration = _grid(duration).astype(np.float64) if not np.isscalar(duration) else duration
    start = yod.fillna(0)
    trough = start + (duration.fillna(1) if isinstance(duration, xr.DataArray) else duration)
    start = start.astype(np.int32)
    trough = xr.where(start > 0, trough, 0).astype(np.int32)
    if start.ndim == 0 and "y" in like.dims:
        start = start.expand_dims(y=like.sizes["y"], x=like.sizes["x"])
        trough = trough.expand_dims(y=like.sizes["y"], x=like.sizes["x"])
    for name in ("y", "x"):
        if name in like.coords and name in start.dims:
            start = start.assign_coords({name: like.coords[name]})
            trough = trough.assign_coords({name: like.coords[name]})
    return start, trough
