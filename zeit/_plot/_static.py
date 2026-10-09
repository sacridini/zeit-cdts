"""Static plots with matplotlib: one map, a grid of dates, or one pixel's series."""

from typing import Any, List, Optional, Sequence

import numpy as np

from ._data import Frames
from ._style import Style

MAX_GRID = 12


def resolve_frames(frames: Frames, time: Any, default_all: bool) -> List[int]:
    """Frame indices from ``time``: an index, a label (``"2015"``), a date, a list, ``"all"``."""
    if time is None:
        if frames.n == 1 or not default_all:
            return [0]
        return list(np.unique(np.linspace(0, frames.n - 1, min(frames.n, MAX_GRID)).astype(int)))
    if isinstance(time, str) and time == "all":
        return list(range(frames.n))
    items = list(time) if isinstance(time, (list, tuple, np.ndarray)) else [time]
    return [_one(frames, t) for t in items]


def _one(frames: Frames, t: Any) -> int:
    if isinstance(t, (int, np.integer)) and not (frames.times is not None and t > 1000):
        if not -frames.n <= int(t) < frames.n:
            raise IndexError(f"frame {t} out of range for {frames.n} frames")
        return int(t) % frames.n
    text = str(t)
    if text in frames.labels:
        return frames.labels.index(text)
    if frames.times is not None:
        import pandas as pd
        stamp = pd.Timestamp(str(int(t)) if isinstance(t, (int, np.integer)) else t)
        return int(np.argmin(np.abs((frames.times - stamp).total_seconds())))
    raise ValueError(f"no frame {t!r}; frames are {frames.labels[:6]}{' ...' if frames.n > 6 else ''}")


def _image(style: Style, codes: np.ndarray) -> np.ndarray:
    """Colour indices -> RGBA floats, through the style's LUT (index 0 transparent)."""
    if style.kind == "rgb":
        rgba = np.zeros(codes.shape[:2] + (4,), dtype=np.float32)
        rgba[..., :3] = (codes.astype(np.float32) - 1).clip(0) / 254.0
        rgba[..., 3] = (codes > 0).all(axis=-1)
        return rgba
    return style.lut()[codes].astype(np.float32) / 255.0


def plot_maps(frames: Frames, style: Style, indices: Sequence[int], *, nodata: Optional[float] = None,
              ax: Any = None, figsize: Any = None, title: Optional[str] = None, colorbar: bool = True,
              max_size: Optional[int] = 1024, ncols: int = 4):
    import matplotlib.pyplot as plt

    n = len(indices)
    step = frames.step_for(max_size)
    extent = frames.extent()
    if not frames.y_down():
        extent = (extent[0], extent[1], extent[3], extent[2])
    if ax is not None:
        if n != 1:
            raise ValueError("ax= draws one map; pick one frame with time=")
        axes, fig = [ax], ax.figure
    else:
        cols = min(ncols, n)
        rows = int(np.ceil(n / cols))
        if figsize is None:
            aspect = frames.height / frames.width
            figsize = (min(4.2 * cols, 16) + (1.2 if colorbar else 0), min(4.2 * cols, 16) / cols * aspect * rows + 0.6)
        fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False, constrained_layout=True)
        axes = list(grid.ravel())
        for extra in axes[n:]:
            extra.set_visible(False)
    for a, i in zip(axes, indices):
        a.imshow(_image(style, style.encode(frames.frame(i, step), nodata)), extent=extent,
                 interpolation="nearest", origin="upper")
        if n > 1 or frames.n > 1:
            a.set_title(frames.labels[i], fontsize=10)
        if n > 1:
            a.set_xticks([]); a.set_yticks([])
    if title is not None or (n == 1 and frames.n == 1 and frames.name):
        fig.suptitle(title if title is not None else frames.name)
    if colorbar:
        _legend(fig, axes[:n], style)
    return fig


def _legend(fig, axes, style: Style) -> None:
    import matplotlib as mpl

    if style.kind == "rgb":
        return
    if style.kind == "categorical":
        handles = [mpl.patches.Patch(color=color, label=text) for _, text, color in style.classes]
        anchor = axes[-1] if len(axes) == 1 else axes[min(len(axes), 4) - 1]
        anchor.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False,
                      title=style.label or None, fontsize=9)
        return
    norm = mpl.colors.Normalize(style.vmin, style.vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=style.matplotlib_cmap())
    bar = fig.colorbar(mappable, ax=axes, shrink=0.8, label=style.label or None)
    ticks = style.ticks()
    bar.set_ticks([v for v, _ in ticks])
    bar.set_ticklabels([t for _, t in ticks])


def plot_series(da: Any, *, ax: Any = None, figsize: Any = None, title: Optional[str] = None):
    import matplotlib.pyplot as plt
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (9, 3.4), constrained_layout=True)
    else:
        fig = ax.figure
    dim = da.dims[0]
    x = da[dim].values if dim in da.coords else np.arange(da.size)
    if np.issubdtype(np.asarray(x).dtype, np.datetime64):
        x = pd.to_datetime(x)
    ax.plot(x, np.asarray(da.values, dtype=float), "o-", ms=3, lw=1.2, color="#1f77b4")
    ax.set_xlabel(dim)
    ax.set_ylabel(str(da.name) if da.name is not None else "")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return fig
