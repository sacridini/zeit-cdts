import threading
from typing import Any, Dict, Optional, Union

import numpy as np
import xarray as xr

_SORT_IDS = {"greatest": 1, "newest": 2, "fastest": 3, "longest": 4, "dsnr": 5}
_EVENT_DTYPES = {"yod": np.uint16, "magnitude": np.float32, "duration": np.uint16, "pre_val": np.float32,
                 "post_val": np.float32, "rate": np.float32, "dsnr": np.float32}


def _extract_py(v_stack, rmse, max_v, rows, cols, is_loss, sort_id, min_mag, min_dur, pre_thresh):
    y_out = np.zeros((rows, cols), dtype=np.uint16)
    m_out = np.zeros((rows, cols), dtype=np.float32)
    d_out = np.zeros((rows, cols), dtype=np.uint16)
    pre_out = np.zeros((rows, cols), dtype=np.float32)
    post_out = np.zeros((rows, cols), dtype=np.float32)
    r_out = np.zeros((rows, cols), dtype=np.float32)
    dsnr_out = np.zeros((rows, cols), dtype=np.float32)

    for r in prange(rows):
        for c in range(cols):
            best_score = -999999.0
            pixel_rmse = rmse[r, c]

            # Iterate through segments (pairs of vertices)
            for i in range(max_v - 1):
                start_year = v_stack[i, r, c]
                end_year = v_stack[i+1, r, c]

                if start_year == 0 or end_year == 0:
                    break # No more valid vertices

                duration = end_year - start_year
                if duration <= 0:
                    continue

                pre_val = v_stack[i + max_v, r, c]
                post_val = v_stack[i+1 + max_v, r, c]

                if is_loss:
                    magnitude = pre_val - post_val
                    if pre_thresh > 0 and pre_val < pre_thresh: continue
                else:
                    magnitude = post_val - pre_val
                    if pre_thresh > 0 and pre_val > pre_thresh: continue

                if magnitude <= 0 or magnitude < min_mag: continue  # a flat segment is no event
                if duration < min_dur: continue

                rate = magnitude / duration
                dsnr = magnitude / pixel_rmse if pixel_rmse > 0 else 0.0

                score = 0.0
                if sort_id == 1: score = magnitude
                elif sort_id == 2: score = start_year
                elif sort_id == 3: score = rate
                elif sort_id == 4: score = duration
                elif sort_id == 5: score = dsnr

                if score > best_score:
                    best_score = score
                    y_out[r, c] = start_year
                    m_out[r, c] = magnitude
                    d_out[r, c] = duration
                    pre_out[r, c] = pre_val
                    post_out[r, c] = post_val
                    r_out[r, c] = rate
                    dsnr_out[r, c] = dsnr

    return y_out, m_out, d_out, pre_out, post_out, r_out, dsnr_out


try:
    # Compiled once per process (lazily, on the first call), not on every call.
    from numba import njit, prange
    _extract = njit(parallel=True, fastmath=True)(_extract_py)
except ImportError:  # pure Python fallback
    prange = range
    _extract = _extract_py

# One call of the parallel kernel at a time: dask runs blocks in threads, and numba's
# "workqueue" threading layer (the one on macOS, without TBB or OpenMP) aborts the process
# when a parallel kernel is entered from two threads at once. The kernel uses every core
# itself, so taking turns costs nothing.
_EXTRACT_LOCK = threading.Lock()


def extract_events(result: Any, event_type: Optional[str] = None, sort_by: str = "greatest",
                   min_magnitude: float = 0.0, min_duration: int = 1, pre_val_threshold: float = 0.0,
                   rmse_map: Optional[np.ndarray] = None, *, band: Any = None
                   ) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    One change event per pixel (e.g. the greatest loss), the same maps for every algorithm.

    Args:
        result: What a change detection algorithm of zeit returned (an `xarray.Dataset`,
            georeferenced, in memory or dask): `zeit.landtrendr` (one event per segment),
            `zeit.ccdc` (one per break between two segments), `zeit.bfast_monitor` (the
            break), `zeit.bfast_lite` (one per break) or `zeit.bfast` (one per trend break).
            A numpy LandTrendr vertex stack (max_vertices * 2, rows, cols), years in the
            first half and values in the second, is also accepted.
        event_type (str): "loss" (value decreases), "gain" (value increases) or "any"
            (either; the magnitude is then the size of the change). Default: the `direction`
            LandTrendr ran with ("loss" for a numpy stack), "any" for the other algorithms.
            LandTrendr takes "loss" or "gain".
        sort_by (str): which event a pixel keeps when it has several: "greatest"
            (magnitude), "newest" (date), "fastest" (rate), "longest" (duration), or "dsnr"
            (magnitude standardized by the fit's noise -- LT-GEE's disturbance
            signal-to-noise ratio; not available for `zeit.bfast`).
        min_magnitude (float): Minimum magnitude to consider.
        min_duration (int): Minimum duration (years) to consider.
        pre_val_threshold (float): Pre-change value threshold: a loss needs `pre_val` at or
            above it, a gain at or below it (0: no threshold).
        rmse_map (np.ndarray, optional): numpy stacks only: (rows, cols) RMSE of each pixel's
            fit. A LandTrendr Dataset brings its own (`rmse`).
        band: CCDC only: the band whose change is measured (e.g. "nir"). Without it, and
            with several bands, the magnitude is the length of the change vector over the
            CCDC run's detection bands (the event type is then "any" and `pre_val`/`post_val`
            are NaN).

    Returns:
        For a Dataset, an `xarray.Dataset` with the same grid and CRS holding:

        - 'yod': the year of the last observation before the change (for LandTrendr, the
          year of the vertex where the event starts); 0 (its NoData) without an event;
        - 'date': the date of the first observation that shows the change (for annual
          LandTrendr, January 1 of `yod + 1`); NaT without an event;
        - 'magnitude' (positive, in the direction of `event_type`), 'pre_val' and
          'post_val' (the fitted values before and after; NaN for BFAST, whose results keep
          no model), 'rate' (magnitude per year) and 'dsnr'; NaN without an event;
        - 'duration' in years (1 for the algorithms that find abrupt breaks); 0 without an
          event.

        For a numpy stack (0 where there is no event), a dict of 2D arrays with the
        LandTrendr keys ('dsnr' only when rmse_map is given, no 'date').
    """
    if sort_by.lower() not in _SORT_IDS:
        raise ValueError(f"sort_by must be one of {sorted(_SORT_IDS)}, got {sort_by!r}")
    if isinstance(result, xr.Dataset):
        if "vertex_year" in result and "vertex_value" in result:
            if band is not None:
                raise ValueError("band= is for CCDC results; LandTrendr's events are on the segmented index")
            return _extract_dataset(result, event_type, sort_by, min_magnitude, min_duration, pre_val_threshold)
        candidates = _break_candidates(result, band)
        if candidates is None:
            raise ValueError("extract_events takes the Dataset returned by zeit.landtrendr, zeit.ccdc, "
                             "zeit.bfast_monitor, zeit.bfast_lite or zeit.bfast")
        return _select_event(result, candidates, event_type, sort_by.lower(), float(min_magnitude),
                             float(min_duration), float(pre_val_threshold))
    if band is not None:
        raise ValueError("band= is for CCDC results")
    if sort_by.lower() == "dsnr" and rmse_map is None:
        raise ValueError("sort_by='dsnr' requires rmse_map (e.g. from run_landtrendr_array(..., return_rmse=True))")
    event_type = "loss" if event_type is None else event_type
    _check_event_type(event_type)
    return _extract_array(np.asarray(result), event_type, sort_by, min_magnitude, min_duration, pre_val_threshold,
                          rmse_map)


def _check_event_type(event_type: str, allowed: tuple = ("loss", "gain")) -> None:
    if not isinstance(event_type, str) or event_type.lower() not in allowed:
        raise ValueError(f"event_type must be one of {list(allowed)}, got {event_type!r}")


def _extract_array(vertices_stack: np.ndarray, event_type: str, sort_by: str, min_magnitude: float,
                   min_duration: int, pre_val_threshold: float, rmse_map: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    bands, rows, cols = vertices_stack.shape
    max_vertices = bands // 2

    has_rmse = rmse_map is not None
    if has_rmse:
        rmse_map = np.ascontiguousarray(rmse_map, dtype=np.float32)
    else:
        # Numba needs a concretely-typed array even when DSNR isn't scored/output.
        rmse_map = np.zeros((rows, cols), dtype=np.float32)

    with _EXTRACT_LOCK:
        out = _extract(np.ascontiguousarray(vertices_stack, dtype=np.float32), rmse_map, max_vertices, rows, cols,
                       event_type.lower() == "loss", _SORT_IDS[sort_by.lower()],
                       float(min_magnitude), float(min_duration), float(pre_val_threshold))
    result = dict(zip(("yod", "magnitude", "duration", "pre_val", "post_val", "rate", "dsnr"), out))
    if not has_rmse:
        del result["dsnr"]
    return result


def _extract_dataset(lt: xr.Dataset, event_type: Optional[str], sort_by: str, min_magnitude: float,
                     min_duration: int, pre_val_threshold: float) -> xr.Dataset:
    if "vertex_year" not in lt or "vertex_value" not in lt:
        raise ValueError("extract_events takes the Dataset returned by zeit.landtrendr "
                         "(with vertex_year and vertex_value)")
    if event_type is None:
        event_type = lt.attrs.get("direction", "loss")
    _check_event_type(event_type)

    pixel = "y" not in lt.vertex_year.dims
    years = lt.vertex_year
    values = lt.vertex_value
    rmse = lt["rmse"] if "rmse" in lt else xr.zeros_like(values.isel(vertex=0))
    if pixel:
        years, values, rmse = (v.expand_dims(("y", "x"), axis=(-2, -1)) for v in (years, values, rmse))
    years = years.transpose("vertex", "y", "x")
    values = values.transpose("vertex", "y", "x")
    rmse = rmse.transpose("y", "x")
    n_vertices = years.sizes["vertex"]
    names = list(_EVENT_DTYPES)

    def _block(year_block, value_block, rmse_block):
        stack = np.concatenate([np.asarray(year_block, dtype=np.float32),
                                np.nan_to_num(np.asarray(value_block, dtype=np.float32), nan=0.0)])
        rmse_values = np.asarray(rmse_block, dtype=np.float32)
        rmse_values = np.nan_to_num(rmse_values.reshape(rmse_values.shape[-2:]), nan=0.0)
        found = _extract_array(stack, event_type, sort_by, min_magnitude, min_duration, pre_val_threshold, rmse_values)
        return np.stack([found[k].astype(np.float32) for k in names])

    if years.chunks is not None:
        import dask.array as da

        y_d = years.data.rechunk({0: -1})
        v_d = values.data.rechunk({0: -1, 1: y_d.chunks[1], 2: y_d.chunks[2]})
        r_d = rmse.data.rechunk((y_d.chunks[1], y_d.chunks[2])) if hasattr(rmse.data, "rechunk") \
            else da.from_array(np.asarray(rmse.data), chunks=(y_d.chunks[1], y_d.chunks[2]))
        out = da.map_blocks(_block, y_d, v_d, r_d[None, ...],
                            dtype=np.float32, chunks=((len(names),),) + y_d.chunks[1:])
    else:
        out = _block(years.values, values.values, rmse.values)

    coords = {k: lt.coords[k] for k in ("y", "x", "spatial_ref") if k in lt.coords}
    variables = {name: (("y", "x"), out[i].astype(dtype)) for i, (name, dtype) in enumerate(_EVENT_DTYPES.items())}
    attrs = dict(event_type=event_type, sort_by=sort_by, min_magnitude=float(min_magnitude),
                 min_duration=int(min_duration), pre_val_threshold=float(pre_val_threshold))
    ds = xr.Dataset(variables, coords=coords if not pixel else {}, attrs=attrs)
    # Where there is no event: yod and duration 0 (their NoData), the values NaN.
    found = ds.yod > 0
    for name, dtype in _EVENT_DTYPES.items():
        if np.dtype(dtype).kind == "f":
            ds[name] = ds[name].where(found)
        else:
            ds[name] = ds[name].rio.write_nodata(0)
    ds["date"] = _days_to_dates(xr.where(found, _year_start_days(ds.yod.astype(np.float64) + 1), np.nan))
    ds = ds[["yod", "date"] + [n for n in _EVENT_DTYPES if n != "yod"]]
    ds.attrs = dict(algorithm="LandTrendr", **ds.attrs)
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    if lt.rio.crs is not None:
        ds = ds.rio.write_crs(lt.rio.crs)
    return ds


# ---------------------------------------------------------------------------
# Events of the algorithms that find breaks (CCDC, BFAST Monitor, BFAST Lite, BFAST)
# ---------------------------------------------------------------------------
# Each pixel has one or more candidate events along an "event" dimension, each with the
# signed change (after minus before), the values before and after (when the result keeps a
# model), the noise of the fit (for the DSNR), the year of the last observation before the
# change and the date (days since 1970) of the first observation that shows it. Then one
# function picks an event per pixel, the same way for every algorithm.

_EPOCH_ORDINAL = 719163   # date(1970, 1, 1).toordinal()


def _ufunc(func, values, dtype=np.float64):
    return xr.apply_ufunc(func, values, dask="parallelized", output_dtypes=[dtype])


def _year_start_days(year):
    """Days since 1970 of January 1 of each (float) year; NaN stays NaN."""
    def _f(y):
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(y)
        yi = np.where(ok, y, 1970).astype(np.int64)
        days = (yi - 1970).astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)
        return np.where(ok, days, np.nan)
    return _ufunc(_f, year)


def _decimal_year_days(value):
    """Days since 1970 of a decimal year (the inverse of zeit's fractional_years, to the day)."""
    def _f(v):
        v = np.asarray(v, dtype=np.float64)
        ok = np.isfinite(v)
        v = np.where(ok, v, 1970.0)
        year = np.floor(v).astype(np.int64)
        start = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)
        end = (year - 1969).astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)
        days = start + np.round((v - year) * (end - start))
        return np.where(ok, days, np.nan)
    return _ufunc(_f, value)


def _dates_to_days(dates):
    def _f(d):
        d = np.asarray(d, dtype="datetime64[ns]")
        days = d.astype("datetime64[D]").astype(np.int64).astype(np.float64)
        return np.where(np.isnat(d), np.nan, days)
    return _ufunc(_f, dates)


def _days_to_dates(days):
    def _f(d):
        d = np.asarray(d, dtype=np.float64)
        ok = np.isfinite(d)
        out = np.where(ok, d, 0).astype(np.int64).astype("datetime64[D]").astype("datetime64[ns]")
        return np.where(ok, out, np.datetime64("NaT", "ns"))
    return _ufunc(_f, days, "datetime64[ns]")


def _days_year(days):
    def _f(d):
        d = np.asarray(d, dtype=np.float64)
        ok = np.isfinite(d)
        year = np.where(ok, d, 0).astype(np.int64).astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)
        return np.where(ok, year + 1970, np.nan)
    return _ufunc(_f, days)


def _events(change, pre, post, noise, last_days, first_days) -> Dict[str, xr.DataArray]:
    return dict(change=change, pre=pre, post=post, noise=noise, yod=_days_year(last_days), date=first_days)


def _break_candidates(result: xr.Dataset, band: Any) -> Optional[Dict[str, Any]]:
    if "t_break" in result and "coefs" in result:
        return _ccdc_candidates(result, band)
    if band is not None:
        raise ValueError("band= is for CCDC results; BFAST runs on one index")
    if "breakpoint" in result and "has_break" in result:
        return _bfast_monitor_candidates(result)
    if "breakpoint_idx_1" in result and "n_breaks" in result:
        return _indexed_candidates(result, "breakpoint_idx_", "break_date_", "magnitude_", "bfast_lite")
    if "trend_breakpoint_idx_1" in result and "n_trend_breaks" in result:
        return _indexed_candidates(result, "trend_breakpoint_idx_", "trend_break_date_", "trend_magnitude_",
                                   "bfast")
    return None


def _ccdc_candidates(result: xr.Dataset, band: Any) -> Dict[str, Any]:
    from ._ccdc_api import _DATENUM_OFFSET, COEFS

    n = result.sizes["segment"]
    if n < 2:
        raise ValueError("this CCDC result keeps one segment per pixel (max_segments=1): no break has a "
                         "model after it")
    names = [str(b) for b in result.band.values]
    before = result.isel(segment=slice(0, n - 1))
    after = result.isel(segment=slice(1, n)).assign_coords(segment=before.segment.values)
    # A break with a model after it: the change is measured between the two models on the
    # date of the break. A break at the end of the series (no segment after it yet) has no
    # magnitude and is not an event.
    exists = before.t_break.notnull() & after.t_start.notnull()
    first = _dates_to_days(before.t_break)
    last = _dates_to_days(before.t_end)
    t = first + _EPOCH_ORDINAL + _DATENUM_OFFSET
    w = 2.0 * np.pi / 365.25
    terms = xr.concat([xr.ones_like(t), t, np.cos(w * t), np.sin(w * t), np.cos(2 * w * t), np.sin(2 * w * t),
                       np.cos(3 * w * t), np.sin(3 * w * t)], dim="coef").assign_coords(coef=COEFS)
    pre = (before.coefs.fillna(0) * terms).sum("coef")
    post = (after.coefs.fillna(0) * terms).sum("coef")
    rmse = before.rmse
    if band is not None:
        if str(band) not in names:
            raise ValueError(f"band {band!r} is not in the CCDC result's bands {names}")
        pre, post, noise = (v.sel(band=str(band), drop=True) for v in (pre, post, rmse))
        change = post - pre
        any_only = False
    elif len(names) == 1:
        pre, post, noise = (v.isel(band=0, drop=True) for v in (pre, post, rmse))
        change = post - pre
        band, any_only = names[0], False
    else:
        detection = result.attrs.get("detection_bands")
        if detection is not None:
            detection = [names[int(i)] for i in np.atleast_1d(detection)]
        elif len(names) >= 6:
            detection = names[1:6]   # Green..SWIR2, the original's default
        else:
            detection = names
        diff = (post - pre).sel(band=detection)
        change = np.sqrt((diff ** 2).sum("band"))
        # The change in units of each band's noise, combined: the DSNR of a change vector.
        standardized = np.sqrt(((diff / rmse.sel(band=detection)) ** 2).sum("band"))
        noise = xr.where(standardized > 0, change / standardized, np.nan)
        pre = post = xr.full_like(change, np.nan)
        any_only = True
    events = _events(change.where(exists), pre.where(exists), post.where(exists), noise.where(exists),
                     last.where(exists), first.where(exists))
    return dict(events={k: v.rename(segment="event").drop_vars("event", errors="ignore") for k, v in events.items()},
                algorithm="CCDC", any_only=any_only, band=None if band is None else str(band))


def _step_before(result: xr.Dataset, first):
    """Days since 1970 of the time step before ``first`` on a regular series (1 / frequency)."""
    return first - np.round(365.25 / float(result.attrs.get("frequency", 1)))


def _bfast_monitor_candidates(result: xr.Dataset) -> Dict[str, Any]:
    if "break_date" not in result:
        raise ValueError("this bfastmonitor result has no break_date; it was made by a zeit older than 0.48: "
                         "run it again")
    first = _dates_to_days(result.break_date)
    exists = (result.has_break == 1) & np.isfinite(first)
    nan = xr.full_like(first, np.nan)
    events = _events(result.magnitude.astype(np.float64).where(exists), nan, nan,
                     result.sigma.astype(np.float64).where(exists), _step_before(result, first).where(exists),
                     first.where(exists))
    return dict(events={k: v.expand_dims(event=1) for k, v in events.items()}, algorithm="bfastmonitor",
                any_only=False, band=None)


def _indexed_candidates(result: xr.Dataset, idx_prefix: str, date_prefix: str, mag_prefix: str,
                        algorithm: str) -> Dict[str, Any]:
    if f"{mag_prefix}1" not in result or f"{date_prefix}1" not in result:
        raise ValueError(f"this {algorithm} result has no magnitude or date per break; it was made by a zeit "
                         "older than 0.48: run it again")
    ks = sorted(int(str(v)[len(idx_prefix):]) for v in result.data_vars if str(v).startswith(idx_prefix))
    change = xr.concat([result[f"{mag_prefix}{k}"].astype(np.float64) for k in ks], dim="event")
    first = xr.concat([_dates_to_days(result[f"{date_prefix}{k}"]) for k in ks], dim="event")
    exists = np.isfinite(change) & np.isfinite(first)
    has_noise = "rss" in result and "n_valid" in result
    if has_noise:   # the residual standard deviation of the segmented fit
        noise = np.sqrt(result.rss.astype(np.float64) / result.n_valid).broadcast_like(change)
    else:
        noise = xr.full_like(change, np.nan)
    nan = xr.full_like(change, np.nan)
    events = _events(change.where(exists), nan, nan, noise.where(exists), _step_before(result, first).where(exists),
                     first.where(exists))
    return dict(events=events, algorithm=algorithm, any_only=False, band=None, no_dsnr=not has_noise)


def _select_event(result: xr.Dataset, candidates: Dict[str, Any], event_type: Optional[str], sort_by: str,
                  min_magnitude: float, min_duration: float, pre_val_threshold: float) -> xr.Dataset:
    ev = candidates["events"]
    algorithm = candidates["algorithm"]
    event_type = "any" if event_type is None else event_type
    _check_event_type(event_type, ("loss", "gain", "any"))
    event_type = event_type.lower()
    if candidates["any_only"] and event_type != "any":
        raise ValueError("without band=, a CCDC event is the change of several bands together and has no "
                         "direction: use event_type='any', or choose the band with band=")
    if sort_by == "dsnr" and candidates.get("no_dsnr"):
        raise ValueError(f"{algorithm} results keep no fit noise: sort_by='dsnr' is not available")

    change, pre = ev["change"], ev["pre"]
    magnitude = {"loss": -change, "gain": change, "any": abs(change)}[event_type]
    duration = xr.ones_like(magnitude)   # abrupt breaks: the change happens within a year
    rate = magnitude / duration
    dsnr = xr.where(ev["noise"] > 0, magnitude / ev["noise"], np.nan)
    ok = np.isfinite(magnitude) & (magnitude > 0) & (magnitude >= min_magnitude) & (duration >= min_duration)
    if pre_val_threshold > 0:
        if event_type == "any":
            raise ValueError("pre_val_threshold needs event_type 'loss' or 'gain'")
        if candidates["band"] is None:
            raise ValueError(f"pre_val_threshold needs the value before the change, which {algorithm} results "
                             "do not keep" + (" without band=" if algorithm == "CCDC" else ""))
        ok = ok & ((pre >= pre_val_threshold) if event_type == "loss" else (pre <= pre_val_threshold))

    score = {"greatest": magnitude, "newest": ev["date"], "fastest": rate, "longest": duration, "dsnr": dsnr}[sort_by]
    score = xr.where(ok & np.isfinite(score), score, -np.inf)
    # The first event with the highest score, as a one-hot weight rather than an index, so
    # that it works on dask too.
    n = magnitude.sizes["event"]
    index = xr.DataArray(np.arange(n), dims="event")
    pick = xr.where(score == score.max("event"), index, n).min("event")
    found = ok.any("event")
    chosen = (index == pick) & ok

    def take(values):
        return xr.where(chosen, values, 0).sum("event", skipna=False).where(found)

    variables = {"yod": take(ev["yod"]), "date": take(ev["date"]), "magnitude": take(magnitude),
                 "duration": take(duration), "pre_val": take(pre), "post_val": take(ev["post"]),
                 "rate": take(rate), "dsnr": take(dsnr)}
    spatial = "y" in variables["yod"].dims and "x" in variables["yod"].dims
    out = {}
    for name, values in variables.items():
        if name == "date":
            out[name] = _days_to_dates(values)
        elif name in ("yod", "duration"):
            values = values.fillna(0).astype(np.uint16)
            out[name] = values.rio.write_nodata(0) if spatial else values
        else:
            out[name] = values.astype(np.float32)
    coords = {k: result.coords[k] for k in ("y", "x", "spatial_ref") if k in result.coords}
    attrs = dict(algorithm=algorithm, event_type=event_type, sort_by=sort_by, min_magnitude=min_magnitude,
                 min_duration=int(min_duration), pre_val_threshold=pre_val_threshold)
    if candidates["band"] is not None:
        attrs["band"] = candidates["band"]
    ds = xr.Dataset(out, attrs=attrs)
    ds = ds.drop_vars([c for c in ds.coords if c not in ("y", "x")]).assign_coords(coords)
    if spatial and result.rio.crs is not None:
        ds = ds.rio.write_crs(result.rio.crs)
    return ds


# ---------------------------------------------------------------------------
# Agreement between maps of events
# ---------------------------------------------------------------------------

def agreement(*events: Any, tolerance: int = 1) -> xr.Dataset:
    """Where and when several change maps agree, e.g. LandTrendr, CCDC and BFAST.

    Parameters
    ----------
    *events
        Two or more maps of events on the same grid: what ``extract_events`` returns (its
        ``yod`` is used), a ``yod`` map (``DataArray``, 0 or NaN for no event), or one
        ``dict`` of name -> either. Maps on different grids go through
        ``zeit.load_raster(m, like=...)`` first.
    tolerance
        Years two events may be apart and still agree (``0``: the same year).

    Returns
    -------
    xarray.Dataset
        - ``yod``: the consensus year, the year (of the last observation before the change)
          that the most maps agree with within ``tolerance`` (ties: the year more maps give
          exactly, then the earliest); 0 where no map has an event;
        - ``n_detected``: how many maps have an event at the pixel;
        - ``n_agree``: how many of them agree with the consensus year;
        - ``spread``: the latest minus the earliest year among the maps with an event (NaN
          without one);
        - ``agrees (map, y, x)``: 1 for the maps that agree with the consensus year.

        ``attrs`` records the maps' names and the tolerance. ``n_agree / n_maps`` is the
        share of the algorithms behind each pixel's change.

    Examples
    --------
    >>> lt = zeit.extract_events(zeit.landtrendr(nbr_annual))
    >>> cc = zeit.extract_events(zeit.ccdc(cube), band="nbr", event_type="loss")
    >>> agree = zeit.agreement({"landtrendr": lt, "ccdc": cc}, tolerance=1)
    >>> confident = agree.yod.where(agree.n_agree == 2, 0)
    """
    if len(events) == 1 and isinstance(events[0], dict):
        named = {str(k): v for k, v in events[0].items()}
    else:
        named = {}
        for i, ev in enumerate(events):
            name = ev.attrs.get("algorithm", f"map{i + 1}") if isinstance(ev, (xr.Dataset, xr.DataArray)) \
                else f"map{i + 1}"
            while name in named:
                name = f"{name}_{i + 1}"
            named[name] = ev
    if len(named) < 2:
        raise ValueError("agreement compares two or more maps of events")
    if int(tolerance) < 0:
        raise ValueError(f"tolerance must be 0 or more years, got {tolerance!r}")

    years = []
    for name, ev in named.items():
        yod = ev["yod"] if isinstance(ev, xr.Dataset) else ev
        if not isinstance(yod, xr.DataArray):
            raise TypeError(f"{name}: expected the Dataset of extract_events or a yod DataArray, got {type(ev)}")
        yod = yod.astype(np.float64)
        years.append(yod.where(np.isfinite(yod) & (yod > 0)))
    first = years[0]
    for name, yod in zip(list(named)[1:], years[1:]):
        if yod.dims != first.dims or any(
                not np.array_equal(yod[d].values, first[d].values) for d in ("y", "x") if d in first.coords):
            raise ValueError(f"{name} is not on the grid of {list(named)[0]}: put it there with "
                             "zeit.load_raster(map, like=...)")
    years = [y.drop_vars([c for c in y.coords if c not in ("y", "x")]) for y in years]
    stack = xr.concat(years, dim="map").assign_coords(map=list(named))
    detected = stack.notnull()
    other = stack.rename(map="other")
    close = (abs(stack - other) <= int(tolerance)) & detected & other.notnull()
    support = close.sum("other")
    exact = ((stack == other) & detected).sum("other")
    # The year with the most support; ties go to the year more maps give exactly, then to
    # the earliest.
    score = xr.where(detected, support * 1e6 + exact * 1e4 - stack, -np.inf)
    n = stack.sizes["map"]
    index = xr.DataArray(np.arange(n), dims="map")
    pick = xr.where(score == score.max("map"), index, n).min("map")
    found = detected.any("map")
    consensus = xr.where(index == pick, stack.fillna(0), 0).sum("map").where(found)
    agrees = detected & (abs(stack - consensus) <= int(tolerance))

    spatial = "y" in consensus.dims and "x" in consensus.dims
    yod = consensus.fillna(0).astype(np.uint16)
    out = xr.Dataset({
        "yod": yod.rio.write_nodata(0) if spatial else yod,
        "n_detected": detected.sum("map").astype(np.uint8),
        "n_agree": agrees.sum("map").astype(np.uint8),
        "spread": (stack.max("map") - stack.min("map")).astype(np.float32),
        "agrees": agrees.astype(np.uint8).transpose("map", ...),
    }, attrs=dict(maps=list(named), n_maps=n, tolerance=int(tolerance)))
    reference = named[list(named)[0]]
    if spatial and reference.rio.crs is not None:
        out = out.rio.write_crs(reference.rio.crs)
    return out
