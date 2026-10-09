"""``zeit.ccdc``: one CCDC for every kind of input.

A pixel's multi-band series, a numpy stack, a georeferenced ``(time, band, y, x)`` cube
(in memory or dask) or a raster on disk all go through the same C++ batch and come back
as the same ``xarray.Dataset`` of segments.
"""

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._time import ordinal_days, to_datetime_index

COEFS = ["a0", "c1", "a1", "b1", "a2", "b2", "a3", "b3"]
_DATENUM_OFFSET = 366  # the original CCDC's time axis (MATLAB datenum) = Python ordinal + 366


def ccdc(
    data: Any,
    *,
    qa: Any = None,
    dates: Optional[Sequence[Any]] = None,
    bands: Optional[Sequence[Union[str, int]]] = None,
    max_segments: int = 6,
    conseq_anom: int = 6,
    chi2_prob_threshold: float = 0.99,
    tmax_cg_prob_threshold: float = 0.999999,
    detection_bands: Optional[Sequence[Union[str, int]]] = None,
    num_c: int = 8,
    tmask_bands: Optional[Sequence[Union[str, int]]] = None,
    thermal_band: Union[str, int, None] = None,
    valid_range: tuple = (0.0, 10000.0),
    thermal_range: tuple = (-9320.0, 7070.0),
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """Continuous Change Detection and Classification (CCDC/COLD, Zhu & Woodcock 2014).

    One function for every input: it reads what ``data`` is and returns the same
    ``xarray.Dataset`` of segments, georeferenced when the input is. The engine is a port
    of the original MATLAB code, validated segment for segment against it.

    Parameters
    ----------
    data
        - a ``(time, band, y, x)`` cube (e.g. from ``load_raster`` or
          ``build_time_series``), in memory or dask;
        - a raster file whose bands are ``date_band`` (written by ``save_raster``), a folder
          of single-date files read with ``pattern=``... anything ``load_raster`` reads;
        - a raster interleaved by date without descriptions, with ``dates`` and ``bands``;
        - a numpy array ``(time, band, y, x)`` with ``dates``;
        - one pixel: a ``(time, band)`` DataArray or a ``pandas.DataFrame`` indexed by
          date with one column per band.

        Values are surface reflectance x 10000 (the original's lasso, range test and cloud
        screen are defined on that scale), bands ordered Blue, Green, Red, NIR, SWIR1,
        SWIR2 [, brightness temperature] unless ``bands`` orders them.
    qa
        Fmask codes per date (0 clear land, 1 water, 2 cloud shadow, 3 snow, 4 cloud,
        255 no observation): the name of a band of the cube (it is then left out of the
        spectral bands), a ``(time, y, x)`` cube or array, or ``None`` (every date clear;
        dates with NaN are no observation).
    dates
        Date of each time step, for numpy input and cubes without dates.
    bands
        Spectral bands to use, by name or 0-based position, in the order above. For a raster
        interleaved by date without descriptions: the names of the bands of each date.
    max_segments
        Maximum number of segments returned per pixel.
    conseq_anom, chi2_prob_threshold, tmax_cg_prob_threshold, num_c, valid_range,
    thermal_range
        CCDC parameters, as in the original (``conse``, change and outlier probabilities,
        number of harmonic coefficients 4/6/8, valid ranges).
    detection_bands, tmask_bands, thermal_band
        Bands used for change detection (default Green..SWIR2 with 6+ bands), for Tmask
        cloud screening (default Green, SWIR1) and the brightness-temperature band, by
        name or 0-based position among the spectral bands.
    nodata
        Value marking a missing observation besides NaN: ``"auto"`` (default) the raster's
        NoData, or ``0`` for integer data without one; a number; or ``None``.
    chunks
        Inputs read from disk: ``None`` reads the raster into memory; ``"auto"`` or a dict
        keeps it lazy, computed block by block.
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        - ``t_start``, ``t_end``, ``t_break (segment, y, x)``: dates of each segment
          (``NaT`` past the last one; ``t_break`` is ``NaT`` when the segment ends without
          a break);
        - ``n_segments (y, x)``;
        - ``rmse (segment, band, y, x)``;
        - ``coefs (segment, band, coef, y, x)``: the harmonic model ``a0 + c1 t + a1 cos wt
          + b1 sin wt + ...`` (``coef`` = a0, c1, a1, b1, a2, b2, a3, b3) on the original's
          time axis (``t`` = ordinal day + 366, ``w`` = 2 pi / 365.25).

        Coordinates and CRS of the input are kept; ``attrs`` records the parameters.
        ``predict_synthetic_image`` turns it into an image for any date.

    Examples
    --------
    >>> cube = zeit.load_raster("landsat_sr/", pattern=r"_(?P<date>\\d{8})_(?P<band>\\w+)\\.tif$")
    >>> segments = zeit.ccdc(cube, qa="fmask")
    >>> first_break = segments.t_break.isel(segment=0).dt.year
    >>> july = zeit.predict_synthetic_image(segments, "2020-07-01")
    """
    from ._ccdc import run_ccdc_batch

    cube, qa_cube, pixel = _as_ccdc_cube(data, qa=qa, dates=dates, bands=bands, chunks=chunks)
    names = [str(b) for b in cube.band.values]
    params = dict(
        conseq_anom=int(conseq_anom), chi2_prob_threshold=float(chi2_prob_threshold),
        tmax_cg_prob_threshold=float(tmax_cg_prob_threshold), num_c=int(num_c),
        detection_bands=_band_indices(detection_bands, names, "detection_bands"),
        tmask_bands=_band_indices(tmask_bands, names, "tmask_bands"),
        thermal_band=None if thermal_band is None else _band_indices([thermal_band], names, "thermal_band")[0],
        valid_range=tuple(valid_range), thermal_range=tuple(thermal_range),
    )
    days = ordinal_days(cube.time.values)
    sentinels = _missing_values(cube, nodata)
    n_bands = len(names)
    per_segment = 3 + n_bands * 9
    max_segments = int(max_segments)

    def _block(values: np.ndarray, qa_block: np.ndarray) -> np.ndarray:
        t, b, rows, cols = values.shape
        out = np.zeros((max_segments * per_segment + 1, rows, cols), dtype=np.float64)
        if rows == 0 or cols == 0:
            return out
        values = values.astype(np.float64, copy=True)
        for value in sentinels:
            values[values == value] = np.nan
        qa_values = np.asarray(qa_block, dtype=np.float64).reshape(t, rows, cols)
        qa_values = np.where(np.isnan(qa_values), 255, qa_values).astype(np.int32)
        segs, counts = run_ccdc_batch(
            days, np.ascontiguousarray(values.transpose(2, 3, 1, 0)),
            np.ascontiguousarray(qa_values.transpose(1, 2, 0)), max_segments=max_segments,
            return_coefs=True, n_jobs=_threads(n_jobs), **params)
        segs = segs.reshape(rows, cols, max_segments, per_segment)
        out[:-1] = segs.transpose(2, 3, 0, 1).reshape(max_segments * per_segment, rows, cols)
        out[-1] = counts.reshape(rows, cols)
        return out

    if cube.chunks is not None or (qa_cube is not None and qa_cube.chunks is not None):
        import dask.array as da

        arr = cube.data if cube.chunks is not None else da.from_array(cube.values)
        arr = arr.rechunk({0: -1, 1: -1})
        qa_arr = (qa_cube.data if qa_cube is not None else da.zeros(
            (cube.sizes["time"],) + arr.shape[2:], dtype=np.int16, chunks=(-1,) + arr.chunks[2:]))
        if not hasattr(qa_arr, "dask"):
            qa_arr = da.from_array(np.asarray(qa_arr))
        qa_arr = qa_arr.rechunk((-1,) + arr.chunks[2:])
        out = da.map_blocks(lambda v, q: _block(v, q), arr, qa_arr[:, None], dtype=np.float64, drop_axis=[1],
                            new_axis=None, chunks=((max_segments * per_segment + 1,),) + arr.chunks[2:])
    else:
        qa_values = np.zeros((cube.sizes["time"], cube.sizes["y"], cube.sizes["x"])) if qa_cube is None \
            else np.asarray(qa_cube.values)
        out = _block(np.asarray(cube.values), qa_values)

    return _to_dataset(out, cube, names, max_segments, per_segment, pixel, params)


def _threads(n_jobs: int) -> int:
    if n_jobs == -1:
        import os
        return max(1, (os.cpu_count() or 2) - 1)
    return int(n_jobs)


def _missing_values(cube: xr.DataArray, nodata: Any) -> list:
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


def _band_indices(selection: Optional[Sequence[Union[str, int]]], names: list, what: str) -> Optional[list]:
    if selection is None:
        return None
    out = []
    for item in selection:
        if isinstance(item, (int, np.integer)):
            if not 0 <= int(item) < len(names):
                raise ValueError(f"{what}: band {item} out of range for {len(names)} bands")
            out.append(int(item))
        elif str(item) in names:
            out.append(names.index(str(item)))
        else:
            raise ValueError(f"{what}: no band {item!r} among {names}")
    return out


def _as_ccdc_cube(data: Any, *, qa: Any, dates: Any, bands: Any, chunks: Any):
    """The input as a (time, band, y, x) cube with dates, its QA cube (or None) and
    True for a single pixel."""
    from ._load import load_raster

    pixel = False
    if isinstance(data, pd.DataFrame):
        times = data.index if dates is None else to_datetime_index(dates)
        frame = data
        if isinstance(qa, str) and qa in frame.columns:
            qa_name = qa
            qa = frame[qa_name].to_numpy()
            frame = frame.drop(columns=[qa_name])
        cube = xr.DataArray(frame.to_numpy(dtype=float)[:, :, None, None], dims=("time", "band", "y", "x"),
                            coords={"time": pd.DatetimeIndex(times), "band": [str(c) for c in frame.columns]})
        pixel = True
    elif isinstance(data, xr.DataArray) and data.ndim == 2 and set(data.dims) == {"time", "band"}:
        cube = data.transpose("time", "band").expand_dims(("y", "x"), axis=(2, 3))
        pixel = True
    else:
        kwargs = {}
        if not isinstance(data, (xr.DataArray, xr.Dataset, np.ndarray)) and not hasattr(data, "dask"):
            kwargs["chunks"] = chunks
        cube = load_raster(data, **kwargs)
        if chunks is not None and cube.chunks is None and isinstance(data, (xr.DataArray, np.ndarray)):
            cube = cube.chunk(chunks)

    # A stack interleaved by date without descriptions: (layers, y, x) + dates + bands.
    if "time" not in cube.dims and "band" in cube.dims and cube.ndim == 3:
        if dates is None or bands is None:
            raise ValueError("no dates in the data: pass dates= (and bands=, the names of the bands of each date, "
                             "for a raster interleaved by date)")
        n_dates, n_bands = len(dates), len(bands)
        if cube.sizes["band"] != n_dates * n_bands:
            raise ValueError(f"{cube.sizes['band']} layers are not {n_dates} dates x {n_bands} bands")
        values = cube.data.reshape((n_dates, n_bands) + cube.shape[1:])
        cube = xr.DataArray(values, dims=("time", "band", "y", "x"),
                            coords={"time": to_datetime_index(dates), "band": [str(b) for b in bands],
                                    **{k: cube.coords[k] for k in ("y", "x", "spatial_ref") if k in cube.coords}},
                            attrs=cube.attrs)
        bands = None
    elif dates is not None:
        if "time" not in cube.dims:
            raise ValueError("dates given, but the data has no time axis")
        if len(dates) != cube.sizes["time"]:
            raise ValueError(f"{len(dates)} dates for {cube.sizes['time']} time steps")
        cube = cube.assign_coords(time=to_datetime_index(dates))
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("no dates in the data: pass dates=")
    if "band" not in cube.dims:
        cube = cube.expand_dims(band=["value"], axis=1)
    if "band" not in cube.coords:
        cube = cube.assign_coords(band=[str(i) for i in range(cube.sizes["band"])])
    cube = cube.transpose("time", "band", "y", "x")

    qa_cube = None
    if isinstance(qa, str):
        labels = [str(b) for b in cube.band.values]
        if qa not in labels:
            raise ValueError(f"qa={qa!r} is not a band of the cube {labels}")
        qa_cube = cube.sel(band=qa, drop=True)
        cube = cube.sel(band=[b for b in cube.band.values if str(b) != qa])
    elif qa is not None:
        qa_values = qa.data if isinstance(qa, xr.DataArray) else np.asarray(qa)
        qa_values = qa_values.reshape((cube.sizes["time"],) + (() if pixel else qa_values.shape[1:]))
        if pixel:
            qa_values = qa_values.reshape(-1, 1, 1)
        qa_cube = xr.DataArray(qa_values, dims=("time", "y", "x"))
        if qa_cube.shape != (cube.sizes["time"], cube.sizes["y"], cube.sizes["x"]):
            raise ValueError(f"qa has shape {qa_cube.shape}, expected (time, y, x) = "
                             f"{(cube.sizes['time'], cube.sizes['y'], cube.sizes['x'])}")
    if bands is not None:
        labels = [str(b) for b in cube.band.values]
        cube = cube.isel(band=_band_indices(bands, labels, "bands"))
    return cube, qa_cube, pixel


def _to_dataset(out: Any, cube: xr.DataArray, names: list, max_segments: int, per_segment: int, pixel: bool,
                params: dict) -> xr.Dataset:
    n_bands = len(names)
    rows, cols = cube.sizes["y"], cube.sizes["x"]
    segs = out[:-1].reshape((max_segments, per_segment, rows, cols))
    counts = out[-1]
    present = np.arange(max_segments)[:, None, None] < counts[None]

    lazy = hasattr(segs, "dask")
    xp = __import__("dask.array", fromlist=["array"]) if lazy else np

    def as_dates(days):
        valid = present & (days > 0)
        stamp = (xp.where(valid, days, 719163) - 719163).astype("int64").astype("datetime64[D]").astype("datetime64[ns]")
        return xp.where(valid, stamp, np.datetime64("NaT", "ns"))

    per_band = segs[:, 3:].reshape((max_segments, n_bands, 9, rows, cols))
    rmse = per_band[:, :, 0]
    coefs = per_band[:, :, 1:]
    mask_b = present[:, None]
    variables = {
        "t_start": (("segment", "y", "x"), as_dates(segs[:, 0])),
        "t_end": (("segment", "y", "x"), as_dates(segs[:, 1])),
        "t_break": (("segment", "y", "x"), as_dates(segs[:, 2])),
        "n_segments": (("y", "x"), counts.astype(np.uint8)),
        "rmse": (("segment", "band", "y", "x"), _where(mask_b, rmse).astype(np.float32)),
        "coefs": (("segment", "band", "coef", "y", "x"), _where(mask_b[:, :, None], coefs).astype(np.float32)),
    }
    coords = {"segment": np.arange(1, max_segments + 1), "band": names, "coef": COEFS}
    if not pixel:
        for name in ("y", "x", "spatial_ref"):
            if name in cube.coords:
                coords[name] = cube.coords[name]
    attrs = {k: (list(v) if isinstance(v, (tuple, list)) else v) for k, v in params.items() if v is not None}
    attrs.update(algorithm="CCDC", time_axis="ordinal day + 366 (MATLAB datenum)",
                 first_date=str(pd.Timestamp(cube.time.values[0]).date()),
                 last_date=str(pd.Timestamp(cube.time.values[-1]).date()))
    ds = xr.Dataset(variables, coords=coords, attrs=attrs)
    if pixel:
        return ds.squeeze(("y", "x"), drop=True)
    if cube.rio.crs is not None:
        ds = ds.rio.write_crs(cube.rio.crs)
    return ds


def _where(mask: Any, values: Any) -> Any:
    if hasattr(values, "dask"):
        import dask.array as da
        return da.where(mask, values, np.nan)
    return np.where(mask, values, np.nan)


def predict_image(segments: xr.Dataset, date: Any) -> xr.DataArray:
    """The modelled image of every band on ``date``, from ``zeit.ccdc`` segments.

    Each pixel uses the segment that covers the date; before the first segment, the first
    one; after the last, the last one; pixels without a model are NaN.
    """
    if isinstance(date, (int, np.integer)):
        stamp = pd.Timestamp.fromordinal(int(date)) if date > 9999 else pd.Timestamp(int(date), 1, 1)
    else:
        stamp = to_datetime_index([date])[0]
    t = float(stamp.toordinal() + _DATENUM_OFFSET)
    w = 2.0 * np.pi / 365.25
    terms = xr.DataArray([1.0, t, np.cos(w * t), np.sin(w * t), np.cos(2 * w * t), np.sin(2 * w * t),
                          np.cos(3 * w * t), np.sin(3 * w * t)], dims="coef", coords={"coef": COEFS})
    when = np.datetime64(stamp.to_datetime64(), "ns")
    exists = segments.t_start.notnull()
    covers = exists & (segments.t_start <= when) & (segments.t_end >= when)
    started = exists & (segments.t_start <= when)
    n = segments.sizes["segment"]
    index = xr.DataArray(np.arange(n), dims="segment")
    first_cover = xr.where(covers, index, n).min("segment")
    last_started = xr.where(started, index, -1).max("segment")
    pick = xr.where(first_cover < n, first_cover, xr.where(last_started >= 0, last_started, 0))
    # A one-hot weight over the segments instead of isel(segment=pick): works on dask too.
    chosen = (index == pick)
    model = (segments.coefs.fillna(0) * chosen).sum("segment")
    image = (model * terms).sum("coef")
    image = image.where(segments.n_segments > 0)
    image = image.transpose("band", *[d for d in ("y", "x") if d in image.dims])
    image = image.drop_vars([c for c in ("segment",) if c in image.coords])
    image = image.assign_coords(time=np.datetime64(stamp.to_datetime64(), "ns"))
    if segments.rio.crs is not None:
        image = image.rio.write_crs(segments.rio.crs)
    return image.rename("synthetic")
