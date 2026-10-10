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
    max_size: Optional[int] = None,
    save: Optional[str] = None,
    height: int = 480,
    fps: int = 8,
    compress: bool = True,
    fit: Any = None,
    pixel: Any = None,
    basemap: Any = None,
    vector: Any = None,
    opacity: Optional[float] = None,
    vector_color: str = "#ffd400",
    block: Optional[bool] = None,
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
        is shown in colour unless ``rgb=False``. A cube of embeddings
        (``zeit.load_embeddings``) is shown through its first three principal components as
        red, green and blue, fitted once on every year so that a colour means the same
        embedding in all of them (``band=`` shows one dimension instead). The viewer also
        shows, computed on the GPU, the cosine similarity of every pixel to the one under the
        cursor (or a pinned one) and the change from the previous year, and can fit the
        components on the visible area.
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
        ``True`` for a matplotlib figure; ``False`` for the interactive viewer. By default the
        viewer: in a notebook (JupyterLab, Notebook, VS Code, Colab) as a widget, elsewhere
        (a script, a terminal) in its own window. A figure instead when ``save`` is given or
        there is no screen to show a window on (no display, CI, tests, or the environment
        variable ``ZEIT_PLOT=static``).
    max_size
        Longest side, in cells, the maps are read at: 1024 for a static plot, 800 for the
        viewer's preview (which fetches finer cells when you zoom in). The cube is never read
        at full resolution as a whole.
    save
        Static plots: also write the figure to this file (PNG, PDF, SVG...).
    height, fps, compress
        Viewer: map height in pixels, playback speed, and whether frames travel compressed
        (smaller, for remote notebooks; slightly more work on each side).
    fit
        A result of zeit on the same grid (``landtrendr``, ``ccdc``, ``extract_events``,
        ``bfast_monitor``/``bfast_lite``/``bfast``, ``mann_kendall``): clicking a pixel in the
        viewer shows its series with what the algorithm made of it (segments, harmonic
        models, breaks, events, trend). It is matched to the data by coordinates.
    pixel
        ``(x, y)`` in the data's coordinates: a static plot of that pixel's series (with
        ``fit``), instead of maps.
    basemap
        A web map under the data: ``"satellite"`` (Esri World Imagery), ``"osm"``,
        ``"light"``, ``"dark"``, ``"topo"``, an xyzservices provider or name, or a
        ``{z}/{x}/{y}`` URL template. The data needs a CRS. The viewer loads the tiles in the
        browser; static plots download them (cached in ``~/.cache/zeit/tiles``).
    vector
        Outlines drawn over the maps: a vector file, a GeoDataFrame/GeoSeries, shapely
        geometries or a list of them (reprojected to the data's CRS).
    opacity
        Opacity of the data (default 1, or 0.8 over a basemap); the viewer has a slider.
    vector_color
        Colour of the outlines.
    block
        Window outside a notebook: wait until it is closed (like ``plt.show()``). By default
        yes when running a script, no in an interactive interpreter.

    Returns
    -------
    matplotlib.figure.Figure, Viewer or Window
        A figure for a static plot; the viewer widget in a notebook; a ``Window`` (with
        ``url`` and ``close()``) outside one. In the viewer: drag to pan,
        wheel to zoom (finer cells are fetched when needed), double-click to reset, space to
        play, arrow keys to step; the value under the cursor is shown below the map. Frames
        are preloaded from the current one outwards and, once in the browser, paging through
        time does not involve the kernel.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")
    >>> zeit.plot(ndvi, time=["1990", "2005", "2020"], static=True)
    >>> zeit.plot(zeit.extract_events(zeit.landtrendr(ndvi)), var="yod", static=True)
    """
    from . import _static

    prepared, dataset = prepare(data, var=var, band=band, rgb=rgb)
    if pixel is not None and isinstance(prepared, Frames):
        from ._fit import overlays, pixel_series
        col, row = _cell_of(prepared, pixel)
        series = pixel_series(prepared, col, row)
        series["overlays"] = overlays(fit, series, shape=(prepared.height, prepared.width), band=band)
        return _finish(_static.plot_pixel(series, ax=ax, figsize=figsize, title=title), save)
    if not isinstance(prepared, Frames):
        fig = _static.plot_series(prepared, ax=ax, figsize=figsize, title=title)
        return _finish(fig, save)

    from ._widget import in_notebook
    notebook = in_notebook()
    if static is None:
        static = save is not None or (not notebook and not can_open_window())
    if not static:
        from ._session import Session
        from ._widget import make_widget
        options = dict(kind=kind, cmap=cmap, vmin=vmin, vmax=vmax, classes=classes, nodata=nodata)
        session = Session(data, var=var, band=band, rgb=rgb, style_options=options, max_size=max_size or 800,
                          compress=compress, title=title, fit=fit, basemap=basemap, vector=vector,
                          opacity=opacity if opacity is not None else (0.8 if basemap else 1.0),
                          vector_color=vector_color)
        if notebook:
            return make_widget(session, height=height, fps=fps)
        from ._window import show_window
        return show_window(session, fps=fps, title=title, block=block)

    frames = prepared
    max_size = max_size or 1024
    style, nodata_value = style_for(frames, kind=kind, cmap=cmap, vmin=vmin, vmax=vmax, classes=classes,
                                    nodata=nodata)
    indices = _static.resolve_frames(frames, time, default_all=True)
    fig = _static.plot_maps(frames, style, indices, nodata=nodata_value, ax=ax, figsize=figsize, title=title,
                            colorbar=colorbar, max_size=max_size, ncols=ncols, basemap=basemap, vector=vector,
                            opacity=opacity if opacity is not None else (0.8 if basemap else 1.0),
                            vector_color=vector_color)
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
    if classes is None and kind is None:
        from ._style import classes_from_attrs
        classes = classes_from_attrs(da.attrs)
    options = dict(name=frames.name, dtype=da.dtype, rgb=frames.rgb, kind=kind, cmap=cmap, vmin=vmin, vmax=vmax,
                   classes=classes, label=frames.name)
    style = infer_style(sample, nodata=value, **options)
    # Integer maps without NoData (Earth Engine exports, LandTrendr's yod): 0 means "none".
    if nodata == "auto" and value is None and np.issubdtype(da.dtype, np.integer) and style.kind != "categorical" \
            and (sample == 0).any():
        value = 0
        style = infer_style(sample, nodata=0, **options)
    return style, value


def can_open_window() -> bool:
    """Whether a window can be shown here: not under tests or CI, not on a display-less
    Linux server, and not turned off with ``ZEIT_PLOT=static``."""
    import os
    import sys

    if os.environ.get("ZEIT_PLOT", "").lower() == "static":
        return False
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"):
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


def _cell_of(frames: Frames, pixel: Any):
    """(col, row) of the cell nearest to map coordinates (x, y)."""
    x, y = (float(v) for v in pixel)
    da = frames.da
    if "x" not in da.coords or "y" not in da.coords:
        return int(round(x)), int(round(y))
    col = int(np.abs(da.x.values - x).argmin())
    row = int(np.abs(da.y.values - y).argmin())
    return col, row


def _finish(fig, save: Optional[str]):
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
