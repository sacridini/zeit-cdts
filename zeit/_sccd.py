"""``zeit.sccd``: Stochastic Continuous Change Detection (S-CCD; Ye, Rogan, Zhu & Eastman 2021),
CCDC whose harmonic model is a state-space model updated by a Kalman filter at each new
observation instead of refitted. The engine is ``_core.sccd``, in Rust, written from the paper.

It takes the inputs of ``zeit.ccdc`` and returns its segments (so ``predict_synthetic_image``
works on them), each with its change probability and magnitude, plus the near-real-time state of
every pixel: how many of its latest observations depart from the model, and since when.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._ccdc_api import _as_ccdc_cube, _band_indices, _missing_values, _where
from ._threads import resolve
from ._time import ordinal_days

COEFS = ["a0", "c1", "a1", "b1", "a2", "b2"]
#: ``status`` of each observation of a pixel's series.
STATUS = ("unused", "model", "outlier", "break", "peek", "queue")
#: ``nrt_mode``: what the latest observations of a pixel are being compared with.
MODES = ("no model", "monitoring", "waiting for observations")
_N_MONITOR = 7


def sccd(
    data: Any,
    *,
    qa: Any = None,
    dates: Optional[Sequence[Any]] = None,
    bands: Optional[Sequence[Union[str, int]]] = None,
    max_segments: int = 6,
    conse: int = 6,
    prob_threshold: float = 0.99,
    outlier_prob: float = 0.99999,
    max_angle: float = 30.0,
    detection_bands: Optional[Sequence[Union[str, int]]] = None,
    tmask_bands: Optional[Sequence[Union[str, int]]] = None,
    valid_range: tuple = (0.0, 10000.0),
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """S-CCD (Ye et al. 2021): continuous change detection with a Kalman filter.

    The model of each band is CCDC's harmonic (trend, annual and semiannual cycles), fitted on
    the first stable observations as CCDC's and then carried forward by a Kalman filter: each
    new observation updates the trend and the seasons a little instead of refitting them, so a
    pixel's model follows slow changes and a new image costs one update. A change is
    ``conse`` consecutive observations whose departure from the model, standardized by the
    model's RMSE, exceeds the chi-square quantile of ``prob_threshold`` in every one, all in
    the same direction (mean angle to their median below ``max_angle``).

    Parameters
    ----------
    data, qa, dates, bands, nodata, chunks
        As ``zeit.ccdc``: a ``(time, band, y, x)`` cube (in memory or dask), a raster or
        anything ``load_raster`` reads, a numpy array with ``dates``, or one pixel (a
        ``(time, band)`` DataArray or a ``pandas.DataFrame`` indexed by date). Surface
        reflectance x 10000, bands ordered Blue, Green, Red, NIR, SWIR1, SWIR2; ``qa`` the Fmask
        codes (0 clear, 1 water, 2 shadow, 3 snow, 4 cloud, 255 none), a band name, an array, or
        ``None`` (every date clear; NaN is no observation).
    max_segments
        Maximum number of segments returned per pixel.
    conse
        Consecutive anomalous observations that make a change (the peek window).
    prob_threshold
        Change probability: the threshold is ``chi2inv(prob_threshold, detection bands)``.
    outlier_prob
        A single observation beyond ``chi2inv(outlier_prob, detection bands)`` is left out.
    max_angle
        Largest mean angle (degrees) of the peek window's change vectors to their median: a
        window whose departures point every way is noise, not a change.
    detection_bands, tmask_bands
        Bands of the change test (default Green..SWIR2 with 6 bands or more) and of Tmask
        (default Green, SWIR1), by name or 0-based position.
    valid_range
        Values outside it (exclusive) are not observations.
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        Segments, as ``zeit.ccdc``:

        - ``t_start``, ``t_end``, ``t_break (segment, y, x)``: dates (``t_break`` NaT for the
          segment still being monitored);
        - ``change_prob (segment, y, x)``: 1 for a confirmed change; for the last segment, the
          share of ``conse`` of its latest observations that already depart from it;
        - ``num_obs (segment, y, x)``: observations in the model;
        - ``rmse``, ``magnitude (segment, band, y, x)``: the model's RMSE and the change's
          magnitude (median departure of the peek window, in the band's units);
        - ``coefs (segment, band, coef, y, x)``: ``a0 + c1 t + a1 cos wt + b1 sin wt + a2 cos 2wt
          + b2 sin 2wt`` on CCDC's time axis (``t`` = ordinal day + 366), the filter's state at
          the segment's end;
        - ``n_segments``, ``n_breaks (y, x)``.

        The near-real-time state, from the latest observations:

        - ``nrt_mode (y, x)``: 0 no model, 1 monitoring, 2 waiting for enough observations to
          start a model (``MODES``);
        - ``n_anomalies``: the latest consecutive observations that depart from the model
          (``conse`` of them make a change), ``anomaly_start`` the date of the first,
          ``anomaly_statistic`` the smallest of their standardized squared norms (the test
          statistic, against ``attrs["threshold"]``) and ``anomaly_magnitude (band, y, x)``
          their median departure;
        - ``last_statistic`` and ``last_observation``: the latest clear observation's statistic
          and date; ``n_queue``: observations waiting for a model (mode 2).

        One pixel also has, per date, ``status (time)`` (``STATUS``: unused, model, outlier,
        break, peek, queue), ``statistic (time)`` and the model's ``predicted``, ``trend``,
        ``annual`` and ``semiannual (time, band)`` components, filtered up to that date.

    Examples
    --------
    >>> segments = zeit.sccd(cube, qa="fmask")
    >>> first_break = segments.t_break.isel(segment=0).dt.year
    >>> alerts = segments.n_anomalies >= 3          # changes in progress, not yet confirmed
    >>> july = zeit.predict_synthetic_image(segments, "2024-07-01")
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.sccd")
    if int(conse) < 1:
        raise ValueError(f"conse must be at least 1, got {conse}")
    if int(max_segments) < 1:
        raise ValueError(f"max_segments must be at least 1, got {max_segments}")
    cube, qa_cube, pixel = _as_ccdc_cube(data, qa=qa, dates=dates, bands=bands, chunks=chunks)
    names = [str(b) for b in cube.band.values]
    params = dict(
        conse=int(conse), prob_threshold=float(prob_threshold), outlier_prob=float(outlier_prob),
        max_angle=float(max_angle),
        detection_bands=_band_indices(detection_bands, names, "detection_bands"),
        tmask_bands=_band_indices(tmask_bands, names, "tmask_bands"),
        valid_min=float(valid_range[0]), valid_max=float(valid_range[1]),
    )
    days = ordinal_days(cube.time.values).astype(np.int32)
    sentinels = _missing_values(cube, nodata)
    n_bands = len(names)
    max_segments = int(max_segments)
    per_segment = 5 + n_bands * (2 + len(COEFS))
    width = max_segments * per_segment + 1 + _N_MONITOR + n_bands

    def _clean(values: np.ndarray, qa_block: np.ndarray, shape: tuple) -> tuple:
        values = values.astype(np.float64, copy=True)
        for value in sentinels:
            values[values == value] = np.nan
        qa_values = np.asarray(qa_block, dtype=np.float64).reshape(shape)
        return values, np.where(np.isnan(qa_values), 255, qa_values).astype(np.int32)

    def _block(values: np.ndarray, qa_block: np.ndarray) -> np.ndarray:
        t, b, rows, cols = values.shape
        if rows == 0 or cols == 0:
            return np.zeros((width, rows, cols), dtype=np.float64)
        values, qa_values = _clean(values, qa_block, (t, rows, cols))
        out = _core.sccd.sccd_batch(
            np.ascontiguousarray(values.transpose(2, 3, 1, 0)), np.ascontiguousarray(qa_values.transpose(1, 2, 0)),
            days, max_segments=max_segments, n_jobs=resolve(n_jobs), **params)
        return out.reshape(rows, cols, width).transpose(2, 0, 1)

    if cube.chunks is not None or (qa_cube is not None and qa_cube.chunks is not None):
        import dask.array as da

        arr = cube.data if cube.chunks is not None else da.from_array(cube.values)
        arr = arr.rechunk({0: -1, 1: -1})
        qa_arr = (qa_cube.data if qa_cube is not None else da.zeros(
            (cube.sizes["time"],) + arr.shape[2:], dtype=np.int16, chunks=(-1,) + arr.chunks[2:]))
        if not hasattr(qa_arr, "dask"):
            qa_arr = da.from_array(np.asarray(qa_arr))
        qa_arr = qa_arr.rechunk((-1,) + arr.chunks[2:])
        out = da.map_blocks(lambda v, q: _block(v, q), arr, qa_arr[:, None], dtype=np.float64, drop_axis=[1],
                            new_axis=None, chunks=((width,),) + arr.chunks[2:])
    else:
        qa_values = np.zeros((cube.sizes["time"], cube.sizes["y"], cube.sizes["x"])) if qa_cube is None \
            else np.asarray(qa_cube.values)
        out = _block(np.asarray(cube.values), qa_values)

    threshold = _core.sccd.threshold(n_bands, params["prob_threshold"], params["detection_bands"])
    ds = _to_dataset(out, cube, names, max_segments, per_segment, pixel, params, threshold)
    if pixel:
        qa_values = np.zeros(cube.sizes["time"]) if qa_cube is None else np.asarray(qa_cube.values)
        values, qa_values = _clean(np.asarray(cube.values)[:, :, 0, 0], qa_values, (-1,))
        trace = _core.sccd.sccd_pixel(days, values.T, qa_values, **params)
        ds = ds.assign_coords(time=cube.time.values)
        ds["status"] = ("time", trace["status"].astype(np.uint8))
        ds["statistic"] = ("time", trace["statistic"])
        for name in ("predicted", "trend", "annual", "semiannual"):
            ds[name] = (("time", "band"), trace[name])
    return ds


def _to_dataset(out: Any, cube: xr.DataArray, names: list, max_segments: int, per_segment: int, pixel: bool,
                params: dict, threshold: float) -> xr.Dataset:
    n_bands = len(names)
    rows, cols = cube.sizes["y"], cube.sizes["x"]
    seg_width = max_segments * per_segment
    segs = out[:seg_width].reshape((max_segments, per_segment, rows, cols))
    counts = out[seg_width]
    mon = out[seg_width + 1:]
    present = np.arange(max_segments)[:, None, None] < counts[None]

    lazy = hasattr(segs, "dask")
    xp = __import__("dask.array", fromlist=["array"]) if lazy else np

    def as_dates(days, valid):
        valid = valid & (days > 0)
        stamp = (xp.where(valid, days, 719163) - 719163).astype("int64").astype("datetime64[D]").astype("datetime64[ns]")
        return xp.where(valid, stamp, np.datetime64("NaT", "ns"))

    per_band = segs[:, 5:].reshape((max_segments, n_bands, 2 + len(COEFS), rows, cols))
    mask_b = present[:, None]
    mode = mon[0]
    monitoring = mode == 1
    anomalous = monitoring & (mon[1] > 0)
    variables = {
        "t_start": (("segment", "y", "x"), as_dates(segs[:, 0], present)),
        "t_end": (("segment", "y", "x"), as_dates(segs[:, 1], present)),
        "t_break": (("segment", "y", "x"), as_dates(segs[:, 2], present)),
        "change_prob": (("segment", "y", "x"), _where(present, segs[:, 3]).astype(np.float32)),
        "num_obs": (("segment", "y", "x"), xp.where(present, segs[:, 4], 0).astype(np.int32)),
        "rmse": (("segment", "band", "y", "x"), _where(mask_b, per_band[:, :, 0]).astype(np.float32)),
        "magnitude": (("segment", "band", "y", "x"), _where(mask_b, per_band[:, :, 1]).astype(np.float32)),
        "coefs": (("segment", "band", "coef", "y", "x"),
                  _where(mask_b[:, :, None], per_band[:, :, 2:]).astype(np.float32)),
        "n_segments": (("y", "x"), counts.astype(np.uint8)),
        "n_breaks": (("y", "x"), (present & (segs[:, 2] > 0)).sum(axis=0).astype(np.uint8)),
        "nrt_mode": (("y", "x"), mode.astype(np.uint8)),
        "n_anomalies": (("y", "x"), mon[1].astype(np.uint8)),
        "anomaly_start": (("y", "x"), as_dates(mon[2], anomalous)),
        "anomaly_statistic": (("y", "x"), _where(anomalous, mon[3]).astype(np.float32)),
        "anomaly_magnitude": (("band", "y", "x"), _where(anomalous[None], mon[_N_MONITOR:]).astype(np.float32)),
        "last_statistic": (("y", "x"), _where(monitoring, mon[4]).astype(np.float32)),
        "last_observation": (("y", "x"), as_dates(mon[5], mon[5] > 0)),
        "n_queue": (("y", "x"), mon[6].astype(np.int32)),
    }
    coords = {"segment": np.arange(1, max_segments + 1), "band": names, "coef": COEFS}
    if not pixel:
        for name in ("y", "x", "spatial_ref"):
            if name in cube.coords:
                coords[name] = cube.coords[name]
    attrs = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in params.items() if v is not None}
    attrs.update(algorithm="S-CCD", threshold=threshold, time_axis="ordinal day + 366 (MATLAB datenum)",
                 status_codes=", ".join(f"{i} {s}" for i, s in enumerate(STATUS)),
                 nrt_modes=", ".join(f"{i} {s}" for i, s in enumerate(MODES)),
                 first_date=str(pd.Timestamp(cube.time.values[0]).date()),
                 last_date=str(pd.Timestamp(cube.time.values[-1]).date()))
    ds = xr.Dataset(variables, coords=coords, attrs=attrs)
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    if cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
    return ds
