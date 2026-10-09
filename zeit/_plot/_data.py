"""What to draw: any zeit input as a sequence of 2-D frames (or one pixel's series).

A ``Frames`` object hides where the data lives (numpy, dask, a file on disk) and how many
dimensions it has: every dimension other than ``y``/``x`` (and the ``band`` of an RGB cube)
becomes one frame axis, labelled like ``save_raster`` names its bands (``2015``,
``1_blue_a0``...). Frames are read at screen resolution, only when asked for.
"""

import itertools
import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

RGB_NAMES = [("red", "green", "blue"), ("r", "g", "b"), ("b04", "b03", "b02"), ("sr_b4", "sr_b3", "sr_b2")]


def _label(value: Any, dim: str) -> str:
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        stamp = pd.Timestamp(value)
        return str(stamp.year) if (stamp.month, stamp.day, stamp.hour) == (1, 1, 0) else stamp.strftime("%Y-%m-%d")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def decimal_years(values: np.ndarray) -> np.ndarray:
    days = values.astype("datetime64[D]")
    years = days.astype("datetime64[Y]")
    start = years.astype("datetime64[D]")
    length = (years + np.timedelta64(1, "Y")).astype("datetime64[D]") - start
    out = years.astype(np.int64).astype(np.float64) + 1970 + (days - start).astype(np.float64) / length.astype(np.float64)
    return np.where(np.isnat(values), np.nan, out)


def rgb_bands(da: xr.DataArray) -> Optional[List[Any]]:
    """The red, green and blue labels of a ``band`` axis, when it has them."""
    if "band" not in da.dims or "band" not in da.coords:
        return None
    labels = [str(b) for b in da.band.values]
    lower = [b.lower() for b in labels]
    for names in RGB_NAMES:
        if all(n in lower for n in names):
            return [da.band.values[lower.index(n)] for n in names]
    return None


class Frames:
    """A DataArray seen as ``n`` frames of ``(y, x)`` (or ``(y, x, 3)`` for RGB)."""

    def __init__(self, da: xr.DataArray, rgb: bool = False):
        if "y" not in da.dims or "x" not in da.dims:
            raise ValueError(f"a map needs y and x dimensions, got {da.dims}")
        self.rgb = rgb
        if rgb:
            da = da.sel(band=rgb_bands(da))
        lead = [d for d in da.dims if d not in ("y", "x") and not (rgb and d == "band")]
        order = lead + (["band"] if rgb else []) + ["y", "x"]
        self.da = da.transpose(*order)
        self.lead = lead
        self.sizes = [self.da.sizes[d] for d in lead]
        self.n = int(np.prod(self.sizes)) if lead else 1
        self.height, self.width = self.da.sizes["y"], self.da.sizes["x"]
        labels = [[_label(v, d) for v in (self.da[d].values if d in self.da.coords else range(1, s + 1))]
                  for d, s in zip(lead, self.sizes)]
        self.labels = ["_".join(p) for p in itertools.product(*labels)] if lead else [str(da.name or "")]
        self.times = None
        if lead == ["time"] and "time" in self.da.coords and np.issubdtype(self.da.time.dtype, np.datetime64):
            self.times = pd.DatetimeIndex(self.da.time.values)

    @property
    def name(self) -> str:
        return str(self.da.name) if self.da.name is not None else ""

    def index(self, i: int) -> dict:
        return dict(zip(self.lead, np.unravel_index(int(i), self.sizes))) if self.lead else {}

    def step_for(self, max_size: Optional[int]) -> int:
        """Decimation that brings the longest side to at most ``max_size`` cells."""
        if not max_size:
            return 1
        return max(1, int(math.ceil(max(self.height, self.width) / max_size)))

    def frame(self, i: int, step: int = 1) -> np.ndarray:
        """Frame ``i`` read every ``step`` cells: ``(h, w)``, or ``(h, w, 3)`` for RGB."""
        sel = self.da.isel(self.index(i))
        if step > 1:
            sel = sel.isel(y=slice(None, None, step), x=slice(None, None, step))
        values = np.asarray(sel.values)
        if values.dtype.kind == "M":  # dates (e.g. CCDC's t_break) as decimal years, NaT -> NaN
            values = decimal_years(values)
        return np.moveaxis(values, 0, -1) if self.rgb else values

    def sample(self, max_frames: int = 8, max_cells: int = 250_000) -> np.ndarray:
        """Values from a few frames, decimated: enough to choose colours, never the whole cube."""
        picks = np.unique(np.linspace(0, self.n - 1, min(self.n, max_frames)).astype(int))
        step = max(1, int(math.ceil(math.sqrt(self.height * self.width * len(picks) / max_cells))))
        return np.stack([self.frame(i, step) for i in picks])

    def extent(self) -> Tuple[float, float, float, float]:
        """(left, right, bottom, top) of the cells' outer edges, in the data's coordinates."""
        da = self.da
        if "x" in da.coords and "y" in da.coords and self.width > 1 and self.height > 1:
            x, y = da.x.values.astype(float), da.y.values.astype(float)
            dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
            return x.min() - dx / 2, x.max() + dx / 2, y.min() - dy / 2, y.max() + dy / 2
        return -0.5, self.width - 0.5, self.height - 0.5, -0.5

    def y_down(self) -> bool:
        """True when the first row is the top (the usual north-up raster)."""
        da = self.da
        return not ("y" in da.coords and self.height > 1 and float(da.y[1]) > float(da.y[0]))


def prepare(data: Any, *, var: Optional[str] = None, band: Any = None, rgb: Optional[bool] = None,
            chunks: Any = "auto") -> Tuple[Any, Optional[xr.Dataset]]:
    """The input as ``Frames`` (or a 1-D DataArray for one pixel's series), plus the Dataset
    it came from (so a viewer can offer its other variables)."""
    from .._load import load_raster

    dataset = None
    if isinstance(data, (str, bytes)) or hasattr(data, "__fspath__") or isinstance(data, (list, tuple)) and data \
            and isinstance(data[0], str):
        data = load_raster(data, chunks=chunks)
    if isinstance(data, xr.Dataset):
        dataset = data
        names = [v for v in data.data_vars if {"y", "x"} <= set(data[v].dims)]
        if var is None:
            if not names:
                one_d = [v for v in data.data_vars if data[v].ndim == 1]
                if not one_d:
                    raise ValueError("nothing to plot: the Dataset has no maps")
                var = one_d[0]
            else:
                var = names[0]
        if var not in data:
            raise ValueError(f"no variable {var!r}; the Dataset has {list(data.data_vars)}")
        da = data[var]
        if da.rio.crs is None and data.rio.crs is not None:
            da = da.rio.write_crs(data.rio.crs)
    elif isinstance(data, pd.Series):
        da = xr.DataArray(data.to_numpy(), dims=("time",), coords={"time": data.index}, name=data.name)
    elif isinstance(data, xr.DataArray):
        da = data
    else:
        array = np.asarray(data)
        dims = {1: ("time",), 2: ("y", "x"), 3: ("band", "y", "x")}.get(array.ndim)
        if dims is None:
            raise ValueError(f"cannot plot an array of shape {array.shape}")
        da = xr.DataArray(array, dims=dims)
    rename = {d: n for d, n in (("lat", "y"), ("latitude", "y"), ("lon", "x"), ("longitude", "x"))
              if d in da.dims and n not in da.dims}
    if rename:
        da = da.rename(rename)
    if band is not None and "band" in da.dims:
        da = da.sel(band=band)
    if da.ndim == 1 or not {"y", "x"} <= set(da.dims):
        return da, dataset
    use_rgb = rgb if rgb is not None else rgb_bands(da) is not None
    if use_rgb and rgb_bands(da) is None:
        raise ValueError("rgb=True needs a band axis with red, green and blue")
    return Frames(da, rgb=bool(use_rgb)), dataset
