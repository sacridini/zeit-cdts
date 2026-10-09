"""``zeit.landtrendr``: one LandTrendr for every kind of input.

A single pixel's series, a numpy stack, a georeferenced cube (in memory or dask) or a
raster on disk all go through the same C++ batch segmentation and come back as the same
``xarray.Dataset`` of vertices.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._landtrendr import run_landtrendr_batch
from ._time import years as _years_of

_DIRECTIONS = {"loss": -1.0, "gain": 1.0}


def landtrendr(
    data: Any,
    *,
    years: Optional[Sequence[int]] = None,
    direction: str = "loss",
    band: Union[str, int, None] = None,
    max_segments: int = 6,
    pval_threshold: float = 0.05,
    recovery_threshold: float = 0.25,
    prevent_fast_recovery: bool = True,
    spike_threshold: float = 0.9,
    best_model_proportion: float = 0.75,
    vertex_count_overshoot: int = 3,
    min_observations_needed: int = 6,
    nodata: Union[float, str, None] = "auto",
    fitted: bool = False,
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """Segment annual time series with LandTrendr (Kennedy et al. 2010).

    One function for every input: it reads what ``data`` is and returns the same
    ``xarray.Dataset`` of vertices, georeferenced when the input is.

    Parameters
    ----------
    data
        - a raster file, folder, Zarr/NetCDF or anything else ``load_raster`` reads;
        - a cube ``(time, y, x)`` (numpy-backed or dask, e.g. from ``load_raster``);
        - a ``(time, band, y, x)`` cube or a ``Dataset``, with ``band`` naming the
          index to segment;
        - a numpy array ``(time, y, x)`` with ``years``;
        - one pixel's series: a list, 1-D array or ``pandas.Series`` (its index gives
          the years when it holds dates or years).
    years
        Year of each time step. Taken from the ``time`` coordinate when there is one;
        needed for numpy input and cubes without dates.
    direction
        ``"loss"`` (default) or ``"gain"``: the change the segmentation looks for, as a
        drop or a rise of the index. Use ``"loss"`` for vegetation loss on NDVI/NBR-like
        indices (which drop) and ``"gain"`` for indices that rise with disturbance
        (e.g. SWIR) or to map regrowth. LT-GEE flips such series by hand; here the
        output values are always in the original scale.
    band
        Index to segment in a ``(time, band, y, x)`` cube or a Dataset (band or
        variable name).
    max_segments, pval_threshold, recovery_threshold, prevent_fast_recovery,
    spike_threshold, best_model_proportion, vertex_count_overshoot,
    min_observations_needed
        LandTrendr parameters, as in LT-GEE (``maxSegments``, ``pvalThreshold``,
        ``recoveryThreshold``, ``preventOneYearRecovery``, ``spikeThreshold``,
        ``bestModelProportion``, ``vertexCountOvershoot``, ``minObservationsNeeded``).
    nodata
        Value marking missing years, which are left out of the fit (NaN always is).
        ``"auto"`` (default): the raster's NoData value; for integer data without one,
        ``0`` (how Earth Engine exports masked pixels, e.g. cloudy years of a composite).
        A number: that value. ``None``: only NaN.
    fitted
        Also return the fitted trajectory, ``fitted (time, y, x)``: the series
        rebuilt from the vertices (LT-GEE's fitted values).
    chunks
        Inputs read from disk: ``None`` reads the raster into memory; ``"auto"`` or a
        dict of chunk sizes keeps it lazy, so the result is computed block by block
        (e.g. while ``save_raster`` writes it), for rasters larger than memory.
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        - ``vertex_year (vertex, y, x)``: year of each vertex, 0 past the last one;
        - ``vertex_value (vertex, y, x)``: fitted value at each vertex, NaN past the
          last one;
        - ``n_vertices (y, x)``: number of vertices;
        - ``rmse (y, x)``: RMSE of the fit, the noise estimate for the DSNR of
          ``extract_events``;
        - ``fitted (time, y, x)`` with ``fitted=True``.

        The ``x``/``y`` coordinates and CRS of the input are kept, and ``attrs`` records
        the parameters (``direction`` among them, the default ``event_type`` of
        ``extract_events``). A single pixel's result has no ``y``/``x`` dims.

    Examples
    --------
    >>> ndvi = zeit.load_raster("LT_Stack_NDVI_Rondonia.tif")   # (time, y, x), 1985-2024
    >>> lt = zeit.landtrendr(ndvi)                              # looks for NDVI drops
    >>> loss = zeit.extract_events(lt)                          # greatest loss per pixel
    >>> zeit.save_raster(loss, "lt_rondonia")                   # one GeoTIFF per metric
    >>> zeit.landtrendr("nbr_stack.tif", chunks="auto")          # lazy, for large rasters
    >>> zeit.landtrendr([5200, 5100, 5300, 2100, 3500, 4600, 5000, 5100],
    ...                 years=range(2010, 2018)).vertex_year.values
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be 'loss' or 'gain', got {direction!r}")
    params = dict(max_segments=int(max_segments), pval_threshold=float(pval_threshold),
                  recovery_threshold=float(recovery_threshold), prevent_fast_recovery=bool(prevent_fast_recovery),
                  spike_threshold=float(spike_threshold), best_model_proportion=float(best_model_proportion),
                  vertex_count_overshoot=int(vertex_count_overshoot),
                  min_observations_needed=int(min_observations_needed), modifier=_DIRECTIONS[direction])

    cube, pixel = _as_series_cube(data, years=years, band=band, chunks=chunks)
    year_values = _years_of(cube.time.values)
    if not pd.Index(year_values).is_unique:
        raise ValueError("LandTrendr needs one observation per year, but some years repeat; build annual "
                         "composites first (zeit.build_annual_composites or "
                         "zeit.regularize_time_series(cube, freq='1YS'))")

    sentinels = _missing_values(cube, nodata)
    max_vertices = params["max_segments"] + 1
    n_out = 2 * max_vertices + 2 + (cube.sizes["time"] if fitted else 0)

    def _block(block: np.ndarray) -> np.ndarray:
        return _segment_block(block, year_values, sentinels, params, n_jobs, fitted)

    if cube.chunks is not None:
        import dask.array as da

        arr = cube.data.rechunk({0: -1})
        out = da.map_blocks(_block, arr, dtype=np.float32, chunks=((n_out,),) + arr.chunks[1:])
    else:
        out = _block(np.asarray(cube.values))

    return _to_dataset(out, cube, year_values, max_vertices, fitted, pixel, direction, params)


def _missing_values(cube: xr.DataArray, nodata: Any) -> list:
    """Values that mark a missing year (NaN aside)."""
    if nodata is None:
        return []
    if isinstance(nodata, str):
        if nodata != "auto":
            raise ValueError(f"nodata must be 'auto', a number or None, got {nodata!r}")
        stored = cube.rio.nodata
        if stored is not None and not np.isnan(stored):
            return [stored]
        return [0] if np.issubdtype(cube.dtype, np.integer) else []
    return [nodata]


def _as_series_cube(data: Any, *, years: Any, band: Any, chunks: Any):
    """The input as a (time, y, x) cube with a time coordinate; True for one pixel."""
    from ._load import load_raster

    if isinstance(data, pd.Series):
        values = data.to_numpy(dtype=float)
        index = data.index
        if years is None and isinstance(index, pd.DatetimeIndex):
            years = index.year
        elif years is None and pd.api.types.is_integer_dtype(index) and index.min() > 1000:
            years = index.to_numpy()
        return _pixel_cube(values, years), True
    if isinstance(data, (list, tuple)) or (isinstance(data, np.ndarray) and data.ndim == 1):
        return _pixel_cube(np.asarray(data, dtype=float), years), True
    if isinstance(data, xr.DataArray) and data.ndim == 1:
        if "time" in data.dims and years is None:
            years = _years_of(pd.to_datetime(data.time.values)) if np.issubdtype(data.time.dtype, np.datetime64) \
                else data.time.values
        return _pixel_cube(np.asarray(data.values, dtype=float), years), True

    kwargs = {}
    if not isinstance(data, (xr.DataArray, xr.Dataset, np.ndarray)) and not hasattr(data, "dask"):
        kwargs["chunks"] = chunks
    if isinstance(data, xr.Dataset):
        if band is None:
            names = [v for v in data.data_vars if data[v].ndim >= 3]
            if len(names) != 1:
                raise ValueError(f"the Dataset has the variables {names}; choose the index with band=")
            band = names[0]
        data = data[band]
        band = None
    cube = load_raster(data, start_year=None, dates=None, **kwargs)
    if chunks is not None and cube.chunks is None and isinstance(data, (xr.DataArray, np.ndarray)):
        cube = cube.chunk(chunks)

    if "band" in cube.dims and "time" in cube.dims:
        if band is None:
            if cube.sizes["band"] != 1:
                raise ValueError(f"the cube has {cube.sizes['band']} bands {list(np.atleast_1d(cube.band.values))}; "
                                 "choose the index to segment with band=")
            cube = cube.isel(band=0, drop=True)
        else:
            cube = cube.sel(band=band, drop=True)
    if years is not None:
        axis = "time" if "time" in cube.dims else "band"
        if axis not in cube.dims:
            raise ValueError("years given, but the data is a single map")
        if cube.sizes[axis] != len(years):
            raise ValueError(f"{len(years)} years for {cube.sizes[axis]} time steps")
        if axis == "band":
            cube = cube.drop_vars("band", errors="ignore").rename(band="time")
        cube = cube.assign_coords(time=pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1) for y in years]))
    if "time" not in cube.dims:
        raise ValueError("no dates in the data: pass years= (e.g. years=range(1985, 2025)), "
                         "or load it with zeit.load_raster(..., start_year=...)")
    if "time" not in cube.coords:
        raise ValueError("the time axis has no dates: pass years=")
    extra = [d for d in cube.dims if d not in ("time", "y", "x")]
    if extra:
        raise ValueError(f"LandTrendr segments one index (time, y, x); the cube also has {extra}")
    if cube.dims != ("time", "y", "x"):
        cube = cube.transpose("time", "y", "x")
    return cube, False


def _pixel_cube(values: np.ndarray, years: Any) -> xr.DataArray:
    if years is None:
        raise ValueError("one pixel's series needs years= (or a pandas Series indexed by dates or years)")
    years = np.asarray(list(years))
    if len(years) != len(values):
        raise ValueError(f"{len(years)} years for {len(values)} values")
    times = pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1) for y in years])
    return xr.DataArray(values.reshape(-1, 1, 1), dims=("time", "y", "x"), coords={"time": times})


def _segment_block(block: np.ndarray, years: np.ndarray, sentinels: list, params: dict, n_jobs: int,
                   fitted: bool) -> np.ndarray:
    """One (time, y, x) block -> (2 * max_vertices + 2 [+ time], y, x) float32: vertex
    years, vertex values, vertex count, RMSE [and the fitted series]."""
    t, rows, cols = block.shape
    max_vertices = params["max_segments"] + 1
    n_out = 2 * max_vertices + 2 + (t if fitted else 0)
    out = np.zeros((n_out, rows, cols), dtype=np.float32)
    if rows == 0 or cols == 0:
        return out
    values = block.astype(np.float64, copy=True)
    for value in sentinels:
        values[values == value] = np.nan
    if n_jobs == -1:
        import os
        n_jobs = max(1, (os.cpu_count() or 2) - 1)

    vertices, counts, rmse = run_landtrendr_batch(
        years, np.ascontiguousarray(np.transpose(values, (1, 2, 0))), no_data_value=np.nan, n_jobs=n_jobs, **params)
    vertices = vertices.reshape(rows, cols, max_vertices, 2)
    counts = counts.reshape(rows, cols)
    present = np.arange(max_vertices)[None, None, :] < counts[..., None]
    vy = np.where(present, vertices[..., 0], 0).transpose(2, 0, 1)
    vv = np.where(present, vertices[..., 1], np.nan).transpose(2, 0, 1)
    out[:max_vertices] = vy
    out[max_vertices:2 * max_vertices] = vv
    out[2 * max_vertices] = counts
    out[2 * max_vertices + 1] = np.where(counts > 0, rmse.reshape(rows, cols), np.nan)
    if fitted:
        out[2 * max_vertices + 2:] = _fit_from_vertices(vy, vv, counts, years)
    return out


def _fit_from_vertices(vy: np.ndarray, vv: np.ndarray, counts: np.ndarray, years: np.ndarray) -> np.ndarray:
    """The series rebuilt from the vertices: linear between consecutive vertices."""
    fit = np.full((len(years),) + counts.shape, np.nan, dtype=np.float32)
    for k in range(vy.shape[0] - 1):
        valid = counts > k + 1
        if not valid.any():
            break
        y0, y1 = vy[k], vy[k + 1]
        v0, v1 = vv[k], vv[k + 1]
        span = np.where(valid, y1 - y0, 1.0)
        for i, yr in enumerate(years):
            inside = valid & (yr >= y0) & (yr <= y1)
            if inside.any():
                fit[i] = np.where(inside, v0 + (v1 - v0) * (yr - y0) / span, fit[i])
    return fit


def _to_dataset(out: Any, cube: xr.DataArray, years: np.ndarray, max_vertices: int, fitted: bool, pixel: bool,
                direction: str, params: dict) -> xr.Dataset:
    spatial = ("y", "x")
    vertex = np.arange(1, max_vertices + 1)
    variables = {
        "vertex_year": (("vertex",) + spatial, out[:max_vertices].astype(np.int16)),
        "vertex_value": (("vertex",) + spatial, out[max_vertices:2 * max_vertices]),
        "n_vertices": (spatial, out[2 * max_vertices].astype(np.uint8)),
        "rmse": (spatial, out[2 * max_vertices + 1]),
    }
    coords = {"vertex": vertex}
    if fitted:
        variables["fitted"] = (("time",) + spatial, out[2 * max_vertices + 2:])
        coords["time"] = cube.time.values
    for name in ("y", "x", "spatial_ref"):
        if name in cube.coords and not pixel:
            coords[name] = cube.coords[name]
    attrs = {k: v for k, v in params.items() if k != "modifier"}
    attrs.update(algorithm="LandTrendr", direction=direction, prevent_fast_recovery=int(params["prevent_fast_recovery"]),
                 first_year=int(years[0]), last_year=int(years[-1]))
    ds = xr.Dataset(variables, coords=coords, attrs=attrs)
    if pixel:
        ds = ds.squeeze(("y", "x"), drop=True)
    elif cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
    return ds
