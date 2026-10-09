"""``zeit.tmask``: time-series cloud and shadow screening of a cube, one call.

Tmask (Zhu & Woodcock 2014) as zeit's C++ engine runs it on every pixel in parallel
(``_core.tmask``): a robust harmonic fit of the green and SWIR bands over every
clear-candidate date (MATLAB's robustfit, bisquare, as CCDC's autoTmask), and the dates
whose green rises (cloud) or whose SWIR drops (shadow) too far from it are flagged.
"""

from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

_MIN_OBSERVATIONS = 5  # pixels with this many valid dates or fewer are left as they are


def tmask(
    data: Any,
    *,
    green: Union[str, int] = "green",
    swir: Union[str, int] = "swir1",
    scale: float = 10000.0,
    nodata: Any = "auto",
    chunks: Any = None,
) -> xr.DataArray:
    """Flag clouds and cloud shadows from each pixel's time series (Tmask).

    Parameters
    ----------
    data
        A ``(time, band, y, x)`` cube with green and SWIR bands (numpy- or dask-backed,
        e.g. from ``load_raster`` or ``build_time_series``), or anything ``load_raster``
        reads. The dates come from its ``time`` coordinate.
    green, swir
        The green and SWIR (SWIR1, ~1.6 µm) bands, by name or position.
    scale
        The reflectance scale of the data (``10000`` for reflectance x 10000; ``1`` for
        reflectance in 0-1). The thresholds are in reflectance.
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``: ``"auto"`` (the
        raster's NoData; ``0`` for integer data without one), a number, or ``None``.
        Observations that are NoData, NaN, or not above 0 in either band are not
        screened (and not clear).
    chunks
        Inputs read from disk: ``None`` reads into memory; ``"auto"`` or a dict keeps the
        result lazy, computed block by block.

    Returns
    -------
    xarray.DataArray
        ``clear (time, y, x)``, ``bool``: ``True`` for a clear observation, ``False`` for
        a cloud, a shadow or no observation. Pixels with ``5`` or fewer valid dates are
        not screened (their valid dates are clear). Georeferenced as the input.

    Examples
    --------
    >>> cube = zeit.load_raster("landsat/", pattern=r"_(?P<date>\\d{8})_(?P<band>\\w+)\\.tif$")
    >>> clear = zeit.tmask(cube, green="green", swir="swir1")
    >>> cube = cube.where(clear)                  # clouds and shadows become NaN
    """
    from ._embeddings import refuse

    refuse(data, "zeit.tmask")
    from ._load import load_raster
    from ._lt import _missing_values, _with_nan

    kwargs = {}
    if not isinstance(data, (xr.DataArray, xr.Dataset, np.ndarray)) and not hasattr(data, "dask"):
        kwargs["chunks"] = chunks
    cube = load_raster(data, **kwargs)
    if chunks is not None and cube.chunks is None and isinstance(data, xr.DataArray):
        cube = cube.chunk(chunks)
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("tmask needs dates: load the data with zeit.load_raster(..., dates=...)")
    if "band" not in cube.dims:
        raise ValueError("tmask needs a (time, band, y, x) cube with green and SWIR bands")
    names = [str(b) for b in np.atleast_1d(cube.band.values)]

    def pick(band: Union[str, int], what: str) -> int:
        if isinstance(band, (int, np.integer)):
            return int(band)
        if str(band) not in names:
            raise ValueError(f"{what}={band!r}: the cube has the bands {names}")
        return names.index(str(band))

    pair = cube.isel(band=[pick(green, "green"), pick(swir, "swir")]).transpose("time", "band", "y", "x")
    sentinels = _missing_values(cube, nodata)
    ordinals = np.array([d.toordinal() for d in pd.DatetimeIndex(cube.time.values)], dtype=np.float64)

    def _block(block: np.ndarray) -> np.ndarray:
        return _screen(_with_nan(block, sentinels), ordinals, float(scale))

    if pair.chunks is not None:
        import dask.array as da

        arr = pair.data.rechunk({0: -1, 1: -1})
        out = da.map_blocks(_block, arr, dtype=bool, drop_axis=1, chunks=(arr.chunks[0],) + arr.chunks[2:])
    else:
        out = _block(np.asarray(pair.values))
    coords = {name: c for name, c in pair.coords.items() if "band" not in c.dims and name != "band"}
    clear = xr.DataArray(out, dims=("time", "y", "x"), coords=coords, name="clear",
                         attrs={"long_name": "clear observation (Tmask)"})
    if cube.rio.crs is not None:
        clear = clear.rio.write_crs(cube.rio.crs)
    return clear


def _screen(values: np.ndarray, ordinals: np.ndarray, scale: float) -> np.ndarray:
    """One (time, 2, y, x) block of green and SWIR -> (time, y, x) clear (C++, every pixel in
    parallel)."""
    from . import _core

    t, _, rows, cols = values.shape
    green = np.ascontiguousarray(np.moveaxis(values[:, 0], 0, -1).reshape(-1, t), dtype=np.float64)
    swir = np.ascontiguousarray(np.moveaxis(values[:, 1], 0, -1).reshape(-1, t), dtype=np.float64)
    clear = _core.tmask.tmask_batch(green, swir, ordinals, scale, _MIN_OBSERVATIONS)
    return np.moveaxis(clear.astype(bool).reshape(rows, cols, t), -1, 0)
