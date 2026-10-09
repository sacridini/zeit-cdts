"""``zeit.twdtw``: classify time series by their likeness to temporal patterns (TWDTW).

Time-Weighted Dynamic Time Warping (Maus et al. 2016, 2019) as the R package twdtw computes
it: each pattern is matched against any stretch of the series, the cost of a pair of
observations is their distance plus a logistic weight of the time elapsed between them,
``1 / (1 + exp(-steepness * (elapsed - midpoint)))``, and, with ``cycle="year"``, the
elapsed time is measured between days of the year, around the year. A pixel's series, a
cube in memory or dask or a raster on disk all go through the same C++ batch and come back
as the same ``xarray.Dataset``.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

_ANY_TIME = 2**31 - 1  # max_elapsed=None: no limit
_CYCLES = {"year": 366.0, None: 0.0}


def twdtw(
    data: Any,
    patterns: Mapping[str, Any],
    *,
    band: Optional[Any] = None,
    steepness: float = 0.1,
    midpoint: float = 50.0,
    cycle: Optional[str] = "year",
    max_elapsed: Optional[float] = None,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """Classify time series by Time-Weighted Dynamic Time Warping (Maus et al. 2016).

    Each pattern (a typical series of a class: a crop's season, a pasture's year) is
    matched against every pixel's series; the class of a pixel is the pattern with the
    lowest TWDTW distance. The distance is the R package twdtw's: the pattern may match any
    stretch of the series, each pair of observations costs their (Euclidean) distance plus
    a logistic time weight ``1 / (1 + exp(-steepness * (elapsed - midpoint)))``, and dates
    a pixel has no value for (NaN, NoData) are left out of its series.

    Parameters
    ----------
    data
        - a cube ``(time, y, x)``, or ``(time, band, y, x)`` for several bands (numpy- or
          dask-backed, e.g. from ``load_raster``), or anything ``load_raster`` reads;
        - one pixel's series: a ``pandas.Series`` (or a ``DataFrame``, one column per band)
          indexed by dates, or a ``DataArray`` with a ``time`` dim.
    patterns
        ``{name: pattern}``: each pattern a ``pandas.Series`` indexed by dates, a
        ``DataFrame`` indexed by dates with one column per band, or a ``DataArray`` with
        a ``time`` (and ``band``) dim. With several bands, the patterns' bands are taken
        from the cube by name.
    band
        A cube with a ``band`` dim and one-band patterns: the band to classify.
    steepness, midpoint
        The logistic time weight: its steepness and its midpoint in days (R's
        ``time_weight = c(0.1, 50)``). Pairs about ``midpoint`` days apart cost half a unit
        more than pairs on the same date.
    cycle
        ``"year"`` (default): the elapsed time is measured between days of the year,
        around it (31 December and 1 January are a day apart), so a pattern of one year
        matches the same season of any year. ``None``: between the dates themselves.
    max_elapsed
        Pairs of observations farther apart than this many days are never matched
        (R's ``max_elapsed``). ``None``: no limit.
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``: ``"auto"`` (the
        raster's NoData; ``0`` for integer data without one), a number, or ``None``.
    chunks
        Inputs read from disk: ``None`` reads into memory; ``"auto"`` or a dict keeps the
        result lazy, computed block by block.
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        - ``label (y, x)``: the class, ``1`` for the first pattern, ``2`` for the second...
          (``0``: no value), with the names in its ``flag_meanings`` (``zeit.plot`` shows
          them in its legend);
        - ``distance (y, x)``: the TWDTW distance to that pattern;
        - ``distances (pattern, y, x)``: the distance to every pattern;
        - ``pattern_value (pattern, pattern_step[, band])`` and ``pattern_time``: the
          patterns, so that ``zeit.plot(cube, fit=result)`` can draw the best one aligned
          with a pixel's series.

        A single pixel's result has no ``y``/``x`` dims.

    Examples
    --------
    >>> ndvi = zeit.load_raster("S2_ndvi_2022.tif")                 # (time, y, x)
    >>> patterns = {"soy": soy_series, "pasture": pasture_series}     # pandas Series by date
    >>> classes = zeit.twdtw(ndvi, patterns)
    >>> classes.label.zeit.plot()                                     # legend: soy, pasture
    """
    from ._embeddings import refuse

    refuse(data, "zeit.twdtw")
    if cycle not in _CYCLES:
        raise ValueError(f"cycle must be 'year' or None, got {cycle!r}")
    if not patterns:
        raise ValueError("patterns: give at least one {name: series}")
    names = [str(k) for k in patterns]
    if any(" " in n for n in names):
        raise ValueError(f"pattern names cannot hold spaces (they label the classes): {names}")
    pats = [_as_pattern(name, p) for name, p in patterns.items()]
    bands = pats[0][2]
    if any(p[2] != bands for p in pats):
        raise ValueError("every pattern needs the same bands")

    cube, pixel = _as_cube(data, bands=bands, band=band, chunks=chunks)
    from ._lt import _missing_values, _with_nan

    sentinels = _missing_values(cube, nodata)
    times = pd.DatetimeIndex(cube.time.values)
    cycle_length = _CYCLES[cycle]
    dates = _date_numbers(times, cycle)
    pattern_inputs = [(np.ascontiguousarray(v if v.shape[1] > 1 else v[:, 0]), _date_numbers(t, cycle))
                      for _, t, _, v in pats]
    params = dict(steepness=float(steepness), midpoint=float(midpoint), cycle_length=cycle_length,
                  max_elapsed=_ANY_TIME if max_elapsed is None else int(np.floor(max_elapsed)))
    multiband = bands is not None
    k = len(pats)

    def _block(block: np.ndarray) -> np.ndarray:
        values = _with_nan(block, sentinels)
        return _classify_block(values, dates, pattern_inputs, params, multiband, n_jobs)

    if cube.chunks is not None:
        import dask.array as da

        arr = cube.data.rechunk({0: -1, 1: -1} if multiband else {0: -1})
        out = da.map_blocks(_block, arr, dtype=np.float64, drop_axis=[0, 1] if multiband else [0],
                            new_axis=0, chunks=((k + 2,),) + arr.chunks[-2:])
    else:
        out = _block(np.asarray(cube.values))
    return _to_dataset(out, cube, pats, pixel, steepness, midpoint, cycle, max_elapsed)


def _as_pattern(name: str, pattern: Any) -> Tuple[str, pd.DatetimeIndex, Optional[List[str]], np.ndarray]:
    """(name, dates, band names or None, values (steps, bands))."""
    if isinstance(pattern, xr.DataArray):
        if "time" not in pattern.dims:
            raise ValueError(f"pattern {name!r}: a DataArray needs a time dim")
        if pattern.ndim == 2 and "band" in pattern.dims:
            frame = pattern.transpose("time", "band").to_pandas()
            frame.columns = [str(c) for c in frame.columns]
            pattern = frame
        elif pattern.ndim == 1:
            pattern = pattern.to_series()
        else:
            raise ValueError(f"pattern {name!r}: dims (time,) or (time, band), got {pattern.dims}")
    if isinstance(pattern, pd.DataFrame):
        if "time" in pattern.columns:
            pattern = pattern.set_index("time")
        bands = [str(c) for c in pattern.columns]
        values = pattern.to_numpy(dtype=np.float64)
        index = pattern.index
        if len(bands) == 1:
            bands, values = None, values[:, :1]
    elif isinstance(pattern, pd.Series):
        bands, values, index = None, pattern.to_numpy(dtype=np.float64)[:, None], pattern.index
    else:
        raise TypeError(f"pattern {name!r}: a pandas Series or DataFrame indexed by dates, or a DataArray "
                        f"with a time dim; got {type(pattern).__name__}")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(index))
    except (TypeError, ValueError):
        raise ValueError(f"pattern {name!r}: its index must hold dates") from None
    keep = np.isfinite(values).all(axis=1)
    order = np.argsort(dates[keep].asi8, kind="stable")
    if not keep.any():
        raise ValueError(f"pattern {name!r} has no values")
    return name, dates[keep][order], bands, values[keep][order]


def _as_cube(data: Any, *, bands: Optional[List[str]], band: Any, chunks: Any) -> Tuple[xr.DataArray, bool]:
    """The data as a (time, y, x) or (time, band, y, x) cube with dates; True for one pixel."""
    from ._load import load_raster

    pixel = False
    if isinstance(data, pd.DataFrame):
        data = xr.DataArray(data.to_numpy(dtype=np.float64), dims=("time", "band"),
                            coords={"time": pd.to_datetime(data.index), "band": [str(c) for c in data.columns]})
    elif isinstance(data, pd.Series):
        data = xr.DataArray(data.to_numpy(dtype=np.float64), dims=("time",),
                            coords={"time": pd.to_datetime(data.index)})
    if isinstance(data, xr.DataArray) and "y" not in data.dims and "x" not in data.dims:
        if "time" not in data.dims or "time" not in data.coords:
            raise ValueError("one pixel's series needs a time dim with dates")
        data = data.expand_dims({"y": 1, "x": 1}).transpose(..., "y", "x")
        pixel = True
    kwargs = {}
    if not isinstance(data, (xr.DataArray, xr.Dataset, np.ndarray)) and not hasattr(data, "dask"):
        kwargs["chunks"] = chunks
    cube = data if pixel else load_raster(data, **kwargs)
    if not pixel and chunks is not None and cube.chunks is None and isinstance(data, xr.DataArray):
        cube = cube.chunk(chunks)
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("TWDTW needs dates: load the data with zeit.load_raster(..., dates=...)")
    if "band" in cube.dims:
        names = [str(b) for b in np.atleast_1d(cube.band.values)]
        if bands is not None:
            missing = [b for b in bands if b not in names]
            if missing:
                raise ValueError(f"the patterns have the bands {bands}; the data has {names}")
            cube = cube.sel(band=bands)
        elif band is not None:
            cube = cube.sel(band=band, drop=True)
        elif len(names) == 1:
            cube = cube.isel(band=0, drop=True)
        else:
            raise ValueError(f"the data has the bands {names} and the patterns one: choose it with band=")
    elif bands is not None:
        raise ValueError(f"the patterns have the bands {bands}; the data has a single one")
    order = ("time", "band", "y", "x") if "band" in cube.dims else ("time", "y", "x")
    if set(cube.dims) != set(order):
        raise ValueError(f"TWDTW classifies (time, y, x) or (time, band, y, x) series, got {cube.dims}")
    return cube.transpose(*order), pixel


def _date_numbers(dates: pd.DatetimeIndex, cycle: Optional[str]) -> np.ndarray:
    """Days of the year (cycle="year"), else days since 1970."""
    if cycle == "year":
        return dates.dayofyear.to_numpy().astype(np.int32)
    return (dates.normalize().asi8 // (86400 * 10**9)).astype(np.int32)


def _engine_params(params: Dict[str, Any]):
    from . import _core

    p = _core.twdtw.TWDTWParams()
    p.alpha = 1.0
    p.beta = params["steepness"]
    p.gamma = params["midpoint"]
    p.cycle_length = params["cycle_length"]
    p.max_time_warp = params["max_elapsed"]
    p.subsequence_matching = True
    return p


def _classify_block(values: np.ndarray, dates: np.ndarray, patterns: Sequence[Tuple[np.ndarray, np.ndarray]],
                    params: Dict[str, Any], multiband: bool, n_jobs: int) -> np.ndarray:
    """One (time[, band], y, x) block -> (patterns + 2, y, x): every pattern's distance, the
    lowest one and its label (1-based; 0 for no value)."""
    from . import _core

    rows, cols = values.shape[-2:]
    k = len(patterns)
    out = np.full((k + 2, rows, cols), np.nan)
    if rows == 0 or cols == 0:
        return out
    if n_jobs == -1:
        import os
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    # the engine takes (y, x, time[, band])
    series = np.ascontiguousarray(np.moveaxis(values, (-2, -1), (0, 1)), dtype=np.float64)
    p = _engine_params(params)
    for i, (pat_values, pat_dates) in enumerate(patterns):
        dist = _core.twdtw.fit_twdtw_batch(series, dates, pat_values, pat_dates, p, float("inf"), n_jobs)
        out[i] = np.where(np.isfinite(dist), dist, np.nan)
    distances = out[:k]
    has = np.isfinite(distances).any(axis=0)
    best = np.argmin(np.where(np.isfinite(distances), distances, np.inf), axis=0)
    out[k] = np.where(has, np.take_along_axis(distances, best[None], axis=0)[0], np.nan)
    out[k + 1] = np.where(has, best + 1, 0)
    return out


def _to_dataset(out: Any, cube: xr.DataArray, pats: list, pixel: bool, steepness: float, midpoint: float,
                cycle: Optional[str], max_elapsed: Optional[float]) -> xr.Dataset:
    k = len(pats)
    names = [p[0] for p in pats]
    spatial = ("y", "x")
    label_dtype = np.uint8 if k < 255 else np.uint16
    label = xr.DataArray(out[k + 1].astype(label_dtype), dims=spatial,
                         attrs={"long_name": "TWDTW class", "flag_values": list(range(1, k + 1)),
                                "flag_meanings": " ".join(names)})
    steps = max(len(p[1]) for p in pats)
    bands = pats[0][2]
    width = 1 if bands is None else len(bands)
    pattern_value = np.full((k, steps, width), np.nan)
    pattern_time = np.full((k, steps), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    for i, (_, dates, _, values) in enumerate(pats):
        pattern_value[i, :len(dates)] = values
        pattern_time[i, :len(dates)] = dates.to_numpy(dtype="datetime64[ns]")
    variables = {
        "label": label,
        "distance": (spatial, out[k]),
        "distances": (("pattern",) + spatial, out[:k]),
        "pattern_time": (("pattern", "pattern_step"), pattern_time),
        "pattern_value": (("pattern", "pattern_step"), pattern_value[..., 0]) if bands is None
        else (("pattern", "pattern_step", "band"), pattern_value),
    }
    coords = {"pattern": names}
    if bands is not None:
        coords["band"] = bands
    for name in ("y", "x", "spatial_ref"):
        if name in cube.coords and not pixel:
            coords[name] = cube.coords[name]
    attrs = dict(algorithm="TWDTW", steepness=float(steepness), midpoint=float(midpoint),
                 cycle=cycle if cycle is not None else "none",
                 max_elapsed=float(max_elapsed) if max_elapsed is not None else float("inf"))
    ds = xr.Dataset(variables, coords=coords, attrs=attrs)
    if pixel:
        ds = ds.squeeze(("y", "x"), drop=True)
    elif cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
        ds["label"] = ds["label"].rio.write_nodata(0, encoded=False)
    return ds


def match(result: xr.Dataset, series_values: np.ndarray, series_dates: pd.DatetimeIndex, pattern: int,
          band: Optional[int] = None) -> List[Tuple[int, int]]:
    """The alignment of a pattern of a ``twdtw`` result with one series: (series index,
    pattern step) pairs, the series indices counting every date (missing ones included).
    ``band``: with multi-band patterns, align on that band only."""
    from . import _core

    values = result.pattern_value.isel(pattern=pattern).values
    times = pd.DatetimeIndex(result.pattern_time.isel(pattern=pattern).values)
    keep = ~times.isna()
    values, times = values[keep], times[keep]
    if values.ndim == 2:
        values = values[:, band if band is not None else 0]
    cycle = None if result.attrs.get("cycle") == "none" else "year"
    elapsed = result.attrs.get("max_elapsed", float("inf"))
    params = dict(steepness=float(result.attrs["steepness"]), midpoint=float(result.attrs["midpoint"]),
                  cycle_length=_CYCLES[cycle], max_elapsed=_ANY_TIME if not np.isfinite(elapsed) else int(elapsed))
    series_values = np.asarray(series_values, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(series_values))
    if valid.size == 0:
        return []
    dates = _date_numbers(pd.DatetimeIndex(series_dates)[valid], cycle)
    res = _core.twdtw.fit_twdtw(series_values[valid].tolist(), dates.tolist(), values.tolist(),
                                _date_numbers(times, cycle).tolist(), 1, _engine_params(params), float("inf"), True)
    return [(int(valid[i]), int(j)) for i, j in res.path]
