"""One pixel's series and what an algorithm made of it, ready to draw.

``pixel_series`` reads the full series of one cell; ``overlays`` turns a result of zeit at
that cell into things to draw over it: lines (LandTrendr's segments, CCDC's harmonic models,
the Mann-Kendall trend, a smoothed series, the TWDTW pattern aligned with the series, the
prototype of a SOM neuron),
vertical marks (breaks) and spans (an event's duration). The
viewer and the static plots draw the same overlays.

x values are milliseconds since 1970 when the series has dates (what JavaScript's Date and
matplotlib's date axis both take after conversion), else frame indices.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from ._data import Frames

FIT_COLOR = "#d62728"


def _ms(dates: Any) -> List[float]:
    idx = pd.DatetimeIndex(np.atleast_1d(np.asarray(dates)))
    return [float(v) / 1e6 for v in idx.asi8]


def _year_ms(years: Sequence[float]) -> List[float]:
    out = []
    for y in years:
        y = float(y)
        year = int(y // 1)
        start = pd.Timestamp(year, 1, 1)
        days = (pd.Timestamp(year + 1, 1, 1) - start).days
        seconds = int(round((y - year) * days * 86400))
        out.append(float(start.value + seconds * 10**9) / 1e6)
    return out


def clean(values: Any) -> List[Optional[float]]:
    """JSON-safe floats: NaN and inf become None."""
    return [None if v is None or not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float).ravel()]


def pixel_series(frames: Frames, col: int, row: int) -> Dict[str, Any]:
    """The series of cell (row, col) of the frames' grid (``row`` counted as stored)."""
    point = frames.da.isel(y=int(row), x=int(col))
    values = np.asarray(point.values, dtype=float)
    if frames.rgb:   # (frames..., band) -> one series per band
        values = values.reshape(-1, values.shape[-1])
        names = [str(b) for b in frames.da.band.values]
        series = [{"name": names[k], "y": clean(values[:, k])} for k in range(values.shape[1])]
    else:
        series = [{"name": frames.name or "value", "y": clean(values.reshape(-1))}]
    da = frames.da
    world = [float(da.x.values[col]) if "x" in da.coords else float(col),
             float(da.y.values[row]) if "y" in da.coords else float(row)]
    if frames.times is not None:
        x, is_time = _ms(frames.times), True
    else:
        x, is_time = list(range(frames.n)), False
    return {"x": x, "series": series, "labels": frames.labels, "is_time": is_time, "world": world,
            "cell": [int(col), int(row)]}


def _at(result: xr.Dataset, world: Sequence[float], cell: Sequence[int], shape: Sequence[int]) -> Optional[xr.Dataset]:
    """The result at a map location: by coordinates when it has them, else by cell."""
    if "y" not in result.dims or "x" not in result.dims:
        return result   # one pixel's result
    if "x" in result.coords and "y" in result.coords and result.sizes["x"] > 1 and result.sizes["y"] > 1:
        x, y = result.x.values, result.y.values
        dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
        if not (min(x) - dx / 2 <= world[0] <= max(x) + dx / 2 and min(y) - dy / 2 <= world[1] <= max(y) + dy / 2):
            return None
        return result.sel(x=world[0], y=world[1], method="nearest")
    if (result.sizes["y"], result.sizes["x"]) == tuple(shape):
        return result.isel(y=int(cell[1]), x=int(cell[0]))
    return None


def overlays(result: Any, series: Dict[str, Any], *, shape: Sequence[int], band: Any = None) -> List[Dict[str, Any]]:
    """What ``result`` (a zeit result Dataset, or a series such as ``zeit.smooth``'s) says
    about the pixel of ``series``."""
    if isinstance(result, xr.DataArray):
        return _series_fit(result, series, shape, band)
    if result is None or not isinstance(result, xr.Dataset):
        return []
    at = _at(result, series["world"], series["cell"], shape)
    if at is None:
        return []
    at = at.compute()
    is_time = series["is_time"]
    out: List[Dict[str, Any]] = []
    if "vertex_year" in at and "vertex_value" in at:                       # LandTrendr
        n = int(at.n_vertices) if "n_vertices" in at else int((at.vertex_year > 0).sum())
        years = at.vertex_year.values[:n].astype(float)
        values = at.vertex_value.values[:n].astype(float)
        if n and is_time:
            out.append({"kind": "line", "x": _year_ms(years), "y": clean(values), "label": "LandTrendr fit",
                        "color": FIT_COLOR, "markers": True})
    if "t_start" in at and "coefs" in at:                                  # CCDC
        out.extend(_ccdc(at, series, band))
    if "yod" in at and "date" in at and result.attrs.get("algorithm", "LandTrendr") != "LandTrendr":
        when = at.date.values                                              # extract_events of a break
        if not np.isnat(when) and is_time:
            out.append({"kind": "vline", "x": _ms([when])[0], "label": f"event {str(when)[:10]}",
                        "color": FIT_COLOR})
    elif "yod" in at and "duration" in at:                                 # extract_events
        yod, dur = float(at.yod), float(at.duration)
        if yod > 0 and is_time:
            x0, x1 = _year_ms([yod, yod + dur])
            out.append({"kind": "span", "x0": x0, "x1": x1, "label": f"event {int(yod)} ({int(dur)} y)",
                        "color": FIT_COLOR})
    if "breakpoint" in at and "has_break" in at:                           # bfast_monitor
        if float(at.has_break) == 1 and np.isfinite(float(at.breakpoint)) and is_time:
            out.append({"kind": "vline", "x": _year_ms([float(at.breakpoint)])[0], "label": "bfastmonitor break",
                        "color": FIT_COLOR})
        if "monitor_start" in result.attrs and is_time:
            out.append({"kind": "vline", "x": _year_ms([float(result.attrs["monitor_start"])])[0],
                        "label": "monitoring starts", "color": "#888888", "dashed": True})
    for prefix, label in (("breakpoint_idx_", "break"), ("trend_breakpoint_idx_", "trend break"),
                          ("season_breakpoint_idx_", "season break")):
        names = [v for v in at.data_vars if str(v).startswith(prefix)]
        if prefix == "breakpoint_idx_" and "n_breaks" not in at:
            continue
        for name in names:   # indices into the whole series, missing observations included
            idx = float(at[name])
            if np.isfinite(idx) and 0 <= int(idx) < len(series["x"]):
                out.append({"kind": "vline", "x": series["x"][int(idx)], "label": label, "color": FIT_COLOR})
    if "distances" in at and "pattern_value" in result:                   # TWDTW
        out.extend(_twdtw(result, at, series, band))
    if "prototypes" in result and "label" in at and "distance" in at:      # SOM
        out.extend(_som(result, at, series, band))
    if "slope" in at and "intercept" in at and "tau" in at:                # Mann-Kendall
        slope, intercept = float(at.slope), float(at.intercept)
        if np.isfinite(slope) and np.isfinite(intercept):
            steps = np.arange(len(series["x"]))
            trend = {1: "increasing", -1: "decreasing"}.get(int(at.trend), "no trend") if "trend" in at else ""
            out.append({"kind": "line", "x": series["x"], "y": clean(intercept + slope * steps),
                        "label": f"Theil-Sen ({trend})", "color": FIT_COLOR, "dashed": True})
    return out


def _ccdc(at: xr.Dataset, series: Dict[str, Any], band: Any) -> List[Dict[str, Any]]:
    from .._ccdc_api import _DATENUM_OFFSET

    if not series["is_time"]:
        return []
    bands = [str(b) for b in at.band.values]
    name = str(band) if band is not None and str(band) in bands else series["series"][0]["name"]
    b = bands.index(name) if name in bands else 0
    out = []
    w = 2.0 * np.pi / 365.25
    for k in range(at.sizes["segment"]):
        start, end, brk = at.t_start.values[k], at.t_end.values[k], at.t_break.values[k]
        if np.isnat(start):
            continue
        days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), periods=120)
        t = np.array([d.toordinal() for d in days], dtype=float) + _DATENUM_OFFSET
        c = at.coefs.values[k, b]
        terms = np.stack([np.ones_like(t), t, np.cos(w * t), np.sin(w * t), np.cos(2 * w * t), np.sin(2 * w * t),
                          np.cos(3 * w * t), np.sin(3 * w * t)])
        out.append({"kind": "line", "x": _ms(days.values), "y": clean(np.nan_to_num(c) @ terms),
                    "label": f"CCDC model ({bands[b]})" if k == 0 else None, "color": FIT_COLOR})
        if not np.isnat(brk):
            out.append({"kind": "vline", "x": _ms([brk])[0], "label": "CCDC break" if k == 0 else None,
                        "color": FIT_COLOR, "dashed": True})
    return out


def _series_fit(fit: xr.DataArray, series: Dict[str, Any], shape: Sequence[int], band: Any) -> List[Dict[str, Any]]:
    """A series over time at the pixel (e.g. ``zeit.smooth``'s), as a line."""
    if "time" not in fit.dims or not series["is_time"]:
        return []
    at = _at(fit, series["world"], series["cell"], shape)
    if at is None:
        return []
    if "band" in at.dims:
        names = [str(b) for b in np.atleast_1d(at.band.values)]
        name = str(band) if band is not None else series["series"][0]["name"]
        at = at.isel(band=names.index(name) if name in names else 0)
    if [d for d in at.dims if d != "time"]:
        return []
    at = at.compute()
    label = fit.attrs.get("smoothing") or str(fit.name or "fit")
    return [{"kind": "line", "x": _ms(at.time.values), "y": clean(at.values), "label": label, "color": FIT_COLOR}]


def _som(result: xr.Dataset, at: xr.Dataset, series: Dict[str, Any], band: Any) -> List[Dict[str, Any]]:
    """The prototype of the pixel's neuron in a ``zeit.som`` result, over its series."""
    k = int(at.label)
    proto = result.prototypes
    if not series["is_time"] or k < 1 or "time" not in proto.dims:
        return []
    proto = proto.sel(neuron=k)
    if "band" in proto.dims:
        names = [str(b) for b in proto.band.values]
        wanted = str(band) if band is not None else series["series"][0]["name"]
        proto = proto.isel(band=names.index(wanted) if wanted in names else 0)
    if [d for d in proto.dims if d != "time"]:
        return []
    proto = proto.compute()
    where = f"{int(proto.i)}_{int(proto.j)}" if "i" in proto.coords else str(k)
    return [{"kind": "line", "x": _ms(proto.time.values), "y": clean(proto.values),
             "label": f"SOM neuron {where} (distance {float(at.distance):.3g})", "color": FIT_COLOR, "dashed": True}]


def _twdtw(result: xr.Dataset, at: xr.Dataset, series: Dict[str, Any], band: Any) -> List[Dict[str, Any]]:
    """The best pattern of a ``zeit.twdtw`` result, aligned with the pixel's series."""
    from .._twdtw_api import match

    if not series["is_time"] or int(at.label) < 1:
        return []
    k = int(at.label) - 1
    name = str(result.pattern.values[k])
    values = np.array([np.nan if v is None else v for v in series["series"][0]["y"]], dtype=float)
    dates = pd.to_datetime(np.asarray(series["x"], dtype=float), unit="ms")
    pattern = result.pattern_value.isel(pattern=k)
    b = None
    if "band" in pattern.dims:
        bands = [str(v) for v in result.band.values]
        wanted = str(band) if band is not None else series["series"][0]["name"]
        b = bands.index(wanted) if wanted in bands else 0
        pattern = pattern.isel(band=b)
    steps = pattern.values[~pd.isna(result.pattern_time.isel(pattern=k).values)]
    path = match(result, values, dates, k, band=b)
    if not path:
        return []
    return [{"kind": "line", "x": [series["x"][i] for i, _ in path], "y": clean([steps[j] for _, j in path]),
             "label": f"TWDTW: {name} (distance {float(at.distance):.3g})", "color": FIT_COLOR, "markers": True}]
