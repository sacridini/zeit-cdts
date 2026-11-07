"""One pixel for the functions that want a ``(time, band, y, x)`` cube."""

import functools
from typing import Any, Callable

import xarray as xr


def is_pixel_series(data: Any) -> bool:
    """A ``(time, band)`` DataArray: what ``load_raster(pixel=)`` gives for several bands."""
    return isinstance(data, xr.DataArray) and data.ndim == 2 and set(data.dims) == {"time", "band"}


def accepts_pixel(func: Callable) -> Callable:
    """Let ``func(data, ...)`` take one pixel, a ``(time, band)`` series without ``y`` and ``x``: it runs on a
    1 x 1 cube (the pixel's ``y`` and ``x``, if it has them, are the coordinates) and its result comes back
    without ``y`` and ``x``."""

    @functools.wraps(func)
    def wrapper(data: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_pixel_series(data):
            return func(data, *args, **kwargs)
        cube = data.transpose("time", "band").expand_dims(("y", "x"), axis=(2, 3))
        result = func(cube, *args, **kwargs)
        return result.squeeze(("y", "x"), drop=True)

    return wrapper
