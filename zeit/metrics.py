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


def extract_events(lt: Any, event_type: Optional[str] = None, sort_by: str = "greatest",
                   min_magnitude: float = 0.0, min_duration: int = 1, pre_val_threshold: float = 0.0,
                   rmse_map: Optional[np.ndarray] = None) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    Extract one change event per pixel (e.g. the greatest loss) from LandTrendr's segments.

    Args:
        lt: The result of `zeit.landtrendr` (an `xarray.Dataset`, georeferenced, in memory
            or dask). A numpy vertex stack (max_vertices * 2, rows, cols), years in the first
            half and values in the second, is also accepted.
        event_type (str): "loss" (value decreases) or "gain" (value increases). Default: the
            `direction` LandTrendr ran with ("loss" for a numpy stack).
        sort_by (str): "greatest" (magnitude), "newest" (year), "fastest" (rate),
            "longest" (duration), or "dsnr" (magnitude standardized by the fit's RMSE --
            LT-GEE's disturbance signal-to-noise ratio).
        min_magnitude (float): Minimum magnitude to consider.
        min_duration (int): Minimum duration (years) to consider.
        pre_val_threshold (float): Pre-disturbance value threshold.
        rmse_map (np.ndarray, optional): numpy stacks only: (rows, cols) RMSE of each pixel's
            fit. A LandTrendr Dataset brings its own (`rmse`).

    Returns:
        For a LandTrendr Dataset, an `xarray.Dataset` with the same grid and CRS holding
        'yod' (year of detection) and 'duration' (0, their NoData, where there is no event),
        'magnitude', 'pre_val', 'post_val', 'rate' and 'dsnr' (NaN where there is no event);
        for a numpy stack (0 where there is no event), a dict of 2D arrays with the
        same keys ('dsnr' only when rmse_map is given).
    """
    if sort_by.lower() not in _SORT_IDS:
        raise ValueError(f"sort_by must be one of {sorted(_SORT_IDS)}, got {sort_by!r}")
    if isinstance(lt, xr.Dataset):
        return _extract_dataset(lt, event_type, sort_by, min_magnitude, min_duration, pre_val_threshold)
    if sort_by.lower() == "dsnr" and rmse_map is None:
        raise ValueError("sort_by='dsnr' requires rmse_map (e.g. from run_landtrendr_array(..., return_rmse=True))")
    event_type = "loss" if event_type is None else event_type
    _check_event_type(event_type)
    return _extract_array(np.asarray(lt), event_type, sort_by, min_magnitude, min_duration, pre_val_threshold,
                          rmse_map)


def _check_event_type(event_type: str) -> None:
    if event_type.lower() not in ("loss", "gain"):
        raise ValueError(f"event_type must be 'loss' or 'gain', got {event_type!r}")


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
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    if lt.rio.crs is not None:
        ds = ds.rio.write_crs(lt.rio.crs)
    return ds
