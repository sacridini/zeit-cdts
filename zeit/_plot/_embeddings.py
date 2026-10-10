"""The viewer's embedding mode: the vectors themselves travel to the browser.

A cube of embeddings (``zeit.load_embeddings``) has dozens of dimensions per pixel, and
what matters is how vectors compare. Instead of colours, the session sends each year's
vectors quantized to int8 (one scale per dimension), laid out as ``(ceil(D/4), h, w, 4)``
so they upload straight into a WebGL2 ``RGBA8I`` texture array. The browser then draws
the principal components, the cosine similarity of every pixel to the one under the
cursor, or the change from another year, on the GPU, without asking the kernel again.

A pixel without an embedding is all zeros with -128 in its first dimension.
"""

import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

#: Bytes of one year's vectors (preview or zoomed-in window): the cells are coarsened to fit.
YEAR_BYTES = 48e6
#: Bytes of quantized years kept for fitting the components of the visible area.
CACHE_BYTES = 256e6
MISSING = -128


def is_embedding_cube(da: Any, band: Any = None, rgb: Optional[bool] = None) -> bool:
    """Whether the viewer shows ``da`` in embedding mode."""
    return (isinstance(da, xr.DataArray) and band is None and rgb is not False and bool(da.attrs.get("embedding_source"))
            and da.sizes.get("band", 0) > 3 and {"y", "x"} <= set(da.dims))


def _jsonable_fit(fit: Dict[str, np.ndarray]) -> Dict[str, List]:
    return {k: np.asarray(v, dtype=float).tolist() for k, v in fit.items()}


class EmbeddingView:
    """The vectors of an embedding cube at the viewer's resolution, and their components."""

    def __init__(self, da: xr.DataArray, *, max_size: int = 800, flip: bool = False):
        from .._embedding_tools import _as_cube, _sample, pca_fit

        da = _as_cube(da, "zeit.plot")
        if "time" not in da.dims:
            da = da.expand_dims("time")
        self.da = da.transpose("time", "band", "y", "x")
        self.flip = flip
        self.dims = self.da.sizes["band"]
        self.layers = math.ceil(self.dims / 4)
        self.height, self.width = self.da.sizes["y"], self.da.sizes["x"]
        sample = _sample(self.da, 40_000)
        self.fit = pca_fit(sample)
        lo, hi = np.percentile(sample, 0.1, axis=0), np.percentile(sample, 99.9, axis=0)
        self.scale = (1.05 * np.maximum(np.abs(lo), np.abs(hi)) / 127).astype(np.float32)
        self.scale[~(self.scale > 0)] = 1e-6
        step = max(1, int(math.ceil(max(self.height, self.width) / max_size))) if max_size else 1
        while self.year_bytes(step) > YEAR_BYTES:
            step += 1
        self.step = step
        self.cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def year_bytes(self, step: int) -> int:
        return math.ceil(self.height / step) * math.ceil(self.width / step) * self.layers * 4

    def meta(self) -> Dict[str, Any]:
        from .._embeddings import describe, embedding_meta

        times = self.da.time.values if "time" in self.da.coords else None
        is_time = times is not None and np.issubdtype(np.asarray(times).dtype, np.datetime64)
        x = [float(v) / 1e6 for v in pd.DatetimeIndex(times).asi8] if is_time else list(range(self.da.sizes["time"]))
        identity = embedding_meta(self.da)
        return {"dims": self.dims, "layers": self.layers, "scale": self.scale.astype(float).tolist(),
                "pca": _jsonable_fit(self.fit), "source": describe(identity) if identity else "",
                "x": x, "is_time": bool(is_time)}

    # ------------------------------------------------------------------ vectors
    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Float vectors ``(D, h, w)`` (rows as stored) -> int8 ``(layers, h, w, 4)``, top row first."""
        missing = ~np.isfinite(values).all(axis=0)
        with np.errstate(invalid="ignore"):
            q = np.rint(values / self.scale[:, None, None])
        q = np.clip(np.nan_to_num(q), -127, 127).astype(np.int8)
        q[:, missing] = 0
        q[0, missing] = MISSING
        if self.flip:
            q = q[:, ::-1]
        d, h, w = q.shape
        if d < self.layers * 4:
            q = np.concatenate([q, np.zeros((self.layers * 4 - d, h, w), dtype=np.int8)])
        return np.ascontiguousarray(q.reshape(self.layers, 4, h, w).transpose(0, 2, 3, 1))

    def dequantize(self, q: np.ndarray) -> np.ndarray:
        """int8 ``(layers, h, w, 4)`` -> float ``(h * w, D)``, NaN where there is no embedding."""
        layers, h, w, _ = q.shape
        values = q.transpose(1, 2, 0, 3).reshape(h * w, layers * 4)[:, :self.dims].astype(np.float32) * self.scale
        values[q[0, :, :, 0].reshape(-1) == MISSING] = np.nan
        return values

    def years(self, start: int, count: int) -> List[np.ndarray]:
        """Quantized vectors of years ``start .. start + count - 1`` at the preview step (one read)."""
        sel = self.da.isel(time=slice(start, start + count), y=slice(None, None, self.step),
                           x=slice(None, None, self.step))
        values = np.asarray(sel.values, dtype=np.float32)
        out = [self.quantize(v) for v in values]
        for k, q in enumerate(out):
            self._remember(start + k, q)
        return out

    def _remember(self, index: int, q: np.ndarray) -> None:
        self.cache[index] = q
        self.cache.move_to_end(index)
        while len(self.cache) > 1 and sum(v.nbytes for v in self.cache.values()) > CACHE_BYTES:
            self.cache.popitem(last=False)

    def window(self, indices: Sequence[int], rows: slice, cols: slice, step: int) -> List[np.ndarray]:
        """Quantized vectors of a window (rows as stored) of some years."""
        sel = self.da.isel(time=list(indices), y=rows, x=cols)
        if step > 1:
            sel = sel.isel(y=slice(None, None, step), x=slice(None, None, step))
        return [self.quantize(v) for v in np.asarray(sel.values, dtype=np.float32)]

    def max_px(self) -> int:
        """Longest side of a zoomed-in window that stays within ``YEAR_BYTES``."""
        return max(64, int(math.sqrt(YEAR_BYTES / (self.layers * 4))))

    # ------------------------------------------------------------------ components
    def pca(self, window: Optional[Sequence[float]] = None) -> Dict[str, Any]:
        """The components of the visible area (full-resolution cells, top row first), from the
        years already sent to the browser (or read from the cube when none was)."""
        from .._embedding_tools import pca_fit

        if window is None:
            return _jsonable_fit(self.fit)
        x0, y0, x1, y1 = (float(v) for v in window)
        x0, x1 = (min(max(v, 0.0), float(self.width)) for v in sorted((x0, x1)))
        y0, y1 = (min(max(v, 0.0), float(self.height)) for v in sorted((y0, y1)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise ValueError("the visible area holds no pixel")
        s = self.step
        rows = []
        if self.cache:
            c0, c1 = int(x0 // s), max(int(x0 // s) + 1, int(math.ceil(x1 / s)))
            r0, r1 = int(y0 // s), max(int(y0 // s) + 1, int(math.ceil(y1 / s)))
            for q in self.cache.values():
                rows.append(self.dequantize(np.ascontiguousarray(q[:, r0:r1, c0:c1])))
        else:
            picks = np.unique(np.linspace(0, self.da.sizes["time"] - 1, min(self.da.sizes["time"], 6)).astype(int))
            step = max(1, int(math.ceil(math.sqrt((x1 - x0) * (y1 - y0) * len(picks) / 40_000))))
            top, bottom = int(y0), int(math.ceil(y1))
            stored = slice(self.height - bottom, self.height - top) if self.flip else slice(top, bottom)
            for q in self.window(picks, stored, slice(int(x0), int(math.ceil(x1))), step):
                rows.append(self.dequantize(q))
        values = np.concatenate(rows)
        return _jsonable_fit(pca_fit(values[np.isfinite(values).all(axis=1)]))
