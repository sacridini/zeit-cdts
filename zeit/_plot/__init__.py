"""``zeit.plot``: look at any zeit data or result.

Static figures (matplotlib) for reports; in a notebook, an interactive viewer that pages
through dense time series at the browser's frame rate.
"""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from ._data import Frames, prepare
from ._style import Style, infer_style

__all__ = ["plot", "Style", "infer_style"]


def plot(
    data: Any,
    *,
    var: Optional[str] = None,
    time: Any = None,
    band: Any = None,
    rgb: Optional[bool] = None,
    kind: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    classes: Union[Dict[Any, Any], Sequence[Any], None] = None,
    nodata: Union[float, str, None] = "auto",
    title: Optional[str] = None,
    static: Optional[bool] = None,
    ax: Any = None,
    figsize: Any = None,
    colorbar: bool = True,
    ncols: int = 4,
    max_size: Optional[int] = 1024,
    save: Optional[str] = None,
):
    """Plot a map, a time series cube, a result of zeit or one pixel's series.

    Parameters
    ----------
    data
        A ``(time, y, x)`` / ``(time, band, y, x)`` cube or a map (DataArray, numpy, dask), a
        raster path (read lazily), a result ``Dataset`` (``var`` picks the variable), or one
        pixel's series (1-D DataArray or ``pandas.Series``).
    var
        Variable of a Dataset (default: the first map).
    time
        Frames to show in a static plot: an index, a label (``"2015"``), a date, a list or
        ``"all"`` (default: up to 12 frames spread over the series). Every dimension other
        than ``y``/``x`` is a frame axis, so it also pages LandTrendr vertices, CCDC
        segments...
    band
        Band to show from a ``(time, band, y, x)`` cube; a cube with red, green and blue bands
        is shown in colour unless ``rgb=False``.
    kind
        How to colour: ``"continuous"``, ``"diverging"``, ``"categorical"``, ``"years"``,
        ``"rgb"``. By default chosen from the data: integers with a few values (or ``bool``)
        are categories, year maps (``yod``...) get year ticks, values spanning 0 a diverging
        palette, the rest a continuous one with limits from the 2-98 % percentiles of a
        sample (never the whole cube).
    cmap, vmin, vmax
        Override the palette and its limits.
    classes
        Categories: a list of values, or ``{value: label}`` / ``{value: (label, colour)}``.
    nodata
        Value drawn transparent besides NaN: ``"auto"`` (the raster's NoData; for integer
        maps that are not categories, 0), a number, or ``None``.
    title, ax, figsize, colorbar, ncols
        Matplotlib layout of a static plot.
    static
        ``True`` for a matplotlib figure.
    max_size
        Longest side, in cells, the maps are read at (default 1024): enough for a screen,
        and the cube is never read at full resolution.
    save
        Static plots: also write the figure to this file (PNG, PDF, SVG...).

    Returns
    -------
    matplotlib.figure.Figure
        For a static plot.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
    >>> zeit.plot(ndvi, time=["1990", "2005", "2020"], static=True)
    >>> zeit.plot(zeit.extract_events(zeit.landtrendr(ndvi)), var="yod", static=True)
    """
    from . import _static

    prepared, dataset = prepare(data, var=var, band=band, rgb=rgb)
    if not isinstance(prepared, Frames):
        fig = _static.plot_series(prepared, ax=ax, figsize=figsize, title=title)
        return _finish(fig, save)

    frames = prepared
    style, nodata_value = style_for(frames, kind=kind, cmap=cmap, vmin=vmin, vmax=vmax, classes=classes,
                                    nodata=nodata)
    indices = _static.resolve_frames(frames, time, default_all=True)
    fig = _static.plot_maps(frames, style, indices, nodata=nodata_value, ax=ax, figsize=figsize, title=title,
                            colorbar=colorbar, max_size=max_size, ncols=ncols)
    return _finish(fig, save)


def style_for(frames: Frames, *, kind=None, cmap=None, vmin=None, vmax=None, classes=None, nodata="auto"):
    """The Style of some frames and the NoData value to draw transparent."""
    da = frames.da
    stored = None
    try:
        stored = da.rio.nodata
    except Exception:  # noqa: BLE001 - no rioxarray georeferencing on this array
        stored = None
    if isinstance(nodata, str):
        if nodata != "auto":
            raise ValueError(f"nodata must be 'auto', a number or None, got {nodata!r}")
        value = None if stored is None or (isinstance(stored, float) and np.isnan(stored)) else stored
    else:
        value = nodata
    sample = frames.sample()
    options = dict(name=frames.name, dtype=da.dtype, rgb=frames.rgb, kind=kind, cmap=cmap, vmin=vmin, vmax=vmax,
                   classes=classes, label=frames.name)
    style = infer_style(sample, nodata=value, **options)
    # Integer maps without NoData (Earth Engine exports, LandTrendr's yod): 0 means "none".
    if nodata == "auto" and value is None and np.issubdtype(da.dtype, np.integer) and style.kind != "categorical" \
            and (sample == 0).any():
        value = 0
        style = infer_style(sample, nodata=0, **options)
    return style, value


def _finish(fig, save: Optional[str]):
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
