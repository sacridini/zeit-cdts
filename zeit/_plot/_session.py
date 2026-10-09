"""The viewer's data side, independent of how it talks to the browser.

The browser asks, the session answers: ``handle(request) -> (content, buffers)``. The
notebook widget carries these requests over the widget's messages, the standalone window
over local HTTP; both use the same session, so they behave the same.

Requests:

- ``meta``: what to draw (sizes, labels, extent, colours, variables);
- ``frames`` (start, count): frames as colour indices at the preview resolution;
- ``detail`` (index, x0, y0, x1, y1, max_px): one frame's window at a finer resolution,
  for zooming in;
- ``select`` (var): show another variable of the Dataset;
- ``pixel`` (x, y): one cell's full series (added by the pixel inspector).
"""

import math
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ._data import Frames, prepare
from ._style import Style

Reply = Tuple[Dict[str, Any], List[bytes]]


class Session:
    def __init__(self, data: Any, *, var: Optional[str] = None, band: Any = None, rgb: Optional[bool] = None,
                 style_options: Optional[Dict[str, Any]] = None, max_size: int = 800, compress: bool = True,
                 title: Optional[str] = None):
        self.source = data
        self.band, self.rgb_option = band, rgb
        self.style_options = dict(style_options or {})
        self.max_size = max_size
        self.compress = compress
        self.title = title
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Reply]] = {
            "meta": self._meta, "frames": self._frames, "detail": self._detail, "select": self._select,
        }
        self._load(var)

    # ------------------------------------------------------------------ setup
    def _load(self, var: Optional[str]) -> None:
        from . import style_for

        frames, dataset = prepare(self.source, var=var, band=self.band, rgb=self.rgb_option)
        if not isinstance(frames, Frames):
            raise ValueError("the interactive viewer shows maps; plot a pixel's series with static=True")
        self.frames, self.dataset = frames, dataset
        first = not hasattr(self, "style")
        self.var = var if var is not None else (frames.name if dataset is not None else None)
        # The caller's colour choices are for the variable it asked for; others get their own.
        options = dict(self.style_options) if first else {}
        self.style, self.nodata = style_for(frames, **options)
        self.step = frames.step_for(self.max_size)
        self.flip = not frames.y_down()   # frames always go top row first

    def variables(self) -> List[str]:
        if self.dataset is None:
            return []
        return [v for v in self.dataset.data_vars if {"y", "x"} <= set(self.dataset[v].dims)]

    def handle(self, request: Dict[str, Any]) -> Reply:
        kind = request.get("type")
        if kind not in self.handlers:
            raise ValueError(f"unknown request {kind!r}")
        return self.handlers[kind](request)

    # ------------------------------------------------------------------ requests
    def _meta(self, request: Dict[str, Any]) -> Reply:
        f = self.frames
        h, w = math.ceil(f.height / self.step), math.ceil(f.width / self.step)
        left, right, bottom, top = (float(v) for v in f.extent())
        if left > right:
            left, right = right, left
        if bottom > top:
            bottom, top = top, bottom
        crs = None
        try:
            crs = f.da.rio.crs.to_string() if f.da.rio.crs is not None else None
        except Exception:  # noqa: BLE001 - not georeferenced
            crs = None
        meta = {
            "n": f.n, "labels": f.labels, "width": w, "height": h, "full_width": f.width, "full_height": f.height,
            "step": self.step, "rgb": f.rgb, "extent": [left, right, bottom, top], "crs": crs,
            "style": self.style.to_json(), "title": self.title if self.title is not None else f.name,
            "variables": self.variables(), "var": self.var, "compressed": self.compress,
            "dims": f.lead, "frame_bytes": w * h * (3 if f.rgb else 1),
        }
        return meta, [self.style.lut().tobytes()]

    def _encode(self, frame: np.ndarray) -> bytes:
        codes = self.style.encode(frame, self.nodata)
        if self.flip:
            codes = codes[::-1]
        raw = np.ascontiguousarray(codes).tobytes()
        return zlib.compress(raw, 1) if self.compress else raw

    def _frames(self, request: Dict[str, Any]) -> Reply:
        start = int(request["start"])
        count = max(1, min(int(request.get("count", 1)), self.frames.n - start))
        block = self.frames.block(start, count, self.step)
        return {"start": start, "count": count}, [self._encode(frame) for frame in block]

    def _detail(self, request: Dict[str, Any]) -> Reply:
        """A window of one frame, in full-resolution cell coordinates (top row first)."""
        f = self.frames
        index = int(request["index"])
        x0, x1 = sorted((int(math.floor(request["x0"])), int(math.ceil(request["x1"]))))
        y0, y1 = sorted((int(math.floor(request["y0"])), int(math.ceil(request["y1"]))))
        x0, x1 = max(0, x0), min(f.width, x1)
        y0, y1 = max(0, y0), min(f.height, y1)
        if x1 <= x0 or y1 <= y0:
            return {"empty": True}, []
        max_px = max(64, int(request.get("max_px", 1024)))
        step = max(1, math.ceil(max(x1 - x0, y1 - y0) / max_px))
        if step >= self.step:   # no finer than what the browser already has
            return {"empty": True}, []
        x0 -= x0 % step
        y0 -= y0 % step
        rows = slice(f.height - y1, f.height - y0) if self.flip else slice(y0, y1)
        sel = f.da.isel(f.index(index)).isel(y=rows, x=slice(x0, x1))
        if step > 1:
            sel = sel.isel(y=slice(None, None, step), x=slice(None, None, step))
        values = np.asarray(sel.values)
        if values.dtype.kind == "M":
            from ._data import decimal_years
            values = decimal_years(values)
        if f.rgb:
            values = np.moveaxis(values, 0, -1)
        h, w = values.shape[:2]
        content = {"index": index, "x0": x0, "y0": y0, "x1": x0 + w * step, "y1": y0 + h * step,
                   "width": w, "height": h, "step": step}
        return content, [self._encode(values)]

    def _select(self, request: Dict[str, Any]) -> Reply:
        self._load(request["var"])
        return self._meta(request)
