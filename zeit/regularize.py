from typing import Any, Optional

import numpy as np
import xarray as xr


def regularize_time_series(
    cube: Any,
    freq: str = '16D',
    method: str = 'median',
    fill: Optional[str] = None,
    max_gap: Optional[float] = None,
    lmbda: float = 10.0,
) -> xr.DataArray:
    """
    Regularize a time series to a fixed step: one composite per period, optionally gap-filled.

    Parameters
    ----------
    cube
        A cube with a ``time`` dim, ``(time, y, x)`` or ``(time, band, y, x)`` (in memory
        or dask), or anything ``load_raster`` reads.
    freq
        The step, a pandas frequency (``'16D'``, ``'MS'``, ``'1YS'``...). The periods are
        labelled by their first day, as pandas does.
    method
        ``'median'`` (per band, each band its own median), ``'medoid'`` (the observation
        closest to the median across bands, so the bands of a composite come from one
        date; with a single band, the observation closest to the median; only
        observations with every band are candidates; it runs in the compiled core, every
        pixel in parallel) or ``'clearest'`` (the dates of a period are ranked by the
        share of the cube's pixels that have every band, the clearest first, ties by
        date; each pixel takes the observation of the first date that has one).
    fill
        ``None`` (default) leaves the periods without a value as NaN. ``'linear'`` fills
        them by linear interpolation in time between the values before and after (the
        series' ends stay NaN). ``'whittaker'`` fills the same gaps with a Whittaker
        curve through the values (``lmbda``), keeping the values.
    max_gap
        With ``fill``, the largest gap to fill, in days between the value before and the
        one after; larger gaps stay NaN. ``None`` (default) fills any.
    lmbda
        ``fill='whittaker'``: the smoothness of the curve, as in ``zeit.smooth``.

    Returns
    -------
    xarray.DataArray
        One composite per period, from the first period with a date to the last.
        Missing values (NaN, and the cube's NoData) are left out; periods without
        observations, and pixels without one in a period, are NaN (the NoData value in an
        integer cube that has one). Georeferencing and NoData are kept.

    Examples
    --------
    >>> reg = zeit.regularize_time_series(cube, "16D", method="clearest", fill="linear", max_gap=48)
    """
    from ._embeddings import refuse

    refuse(cube, "zeit.regularize_time_series")
    if method not in ("median", "medoid", "clearest"):
        raise ValueError(f"Unknown method {method}. Use 'median', 'medoid' or 'clearest'.")
    if fill not in (None, "linear", "whittaker"):
        raise ValueError(f"Unknown fill {fill}. Use None, 'linear' or 'whittaker'.")
    if max_gap is not None and max_gap < 0:
        raise ValueError("max_gap must be positive (days)")
    if not isinstance(cube, xr.DataArray):
        from ._load import load_raster
        cube = load_raster(cube)
    if "time" not in cube.dims or "time" not in cube.coords:
        raise ValueError("regularize_time_series needs a time dim with dates")
    spatial = "y" in cube.dims and "x" in cube.dims
    crs = cube.rio.crs if spatial else None
    nodata = cube.rio.nodata if spatial else None
    dtype = cube.dtype
    if nodata is not None and not np.isnan(nodata):
        work = cube.where(cube != nodata)          # NoData out of the medians (floats from here)
    else:
        nodata = None
        work = cube if np.issubdtype(dtype, np.floating) else cube.astype(np.float64)

    if method == 'median':
        out = work.resample(time=freq).median(dim='time')
    elif method == 'medoid':
        out = _medoid(work, freq)
    else:
        out = _clearest(work, freq)
    if fill is not None:
        out = _fill(out, fill, max_gap, lmbda)

    if np.issubdtype(dtype, np.integer) and nodata is not None:
        out = out.fillna(nodata).astype(dtype)     # back to the cube's type, gaps as NoData
    elif np.issubdtype(dtype, np.floating):
        out = out.astype(dtype)
    if crs is not None and out.rio.crs is None:
        out = out.rio.write_crs(crs)
    if nodata is not None and out.rio.nodata is None:
        out = out.rio.write_nodata(nodata, encoded=False)
    return out


def _clearest(cube: xr.DataArray, freq: str) -> xr.DataArray:
    """Per period, each pixel takes the observation (with every band) of the first date in the
    ranking of the dates by the share of pixels that have every band (stable: the earlier date wins a tie)."""
    spatial = [d for d in ("y", "x") if d in cube.dims]
    have_all = cube.notnull().all("band") if "band" in cube.dims else cube.notnull()
    share = have_all.mean(spatial).compute()

    def _pick(group: xr.DataArray) -> xr.DataArray:
        order = np.argsort(-share.sel(time=group.time).values, kind="stable")
        group = group.isel(time=order)
        have = have_all.sel(time=group.time)
        out = xr.full_like(group.isel(time=0, drop=True), np.nan)
        for k in range(group.sizes["time"] - 1, -1, -1):      # the clearest date last, so it wins
            out = xr.where(have.isel(time=k, drop=True), group.isel(time=k, drop=True), out)
        return out

    return cube.resample(time=freq).map(_pick).transpose(*cube.dims)


def _fill(reg: xr.DataArray, how: str, max_gap: Optional[float], lmbda: float) -> xr.DataArray:
    """The gaps of ``reg`` (NaN) filled linearly in time, those up to ``max_gap`` days wide, or
    by a Whittaker curve in the same gaps; the values are kept."""
    if reg.chunks is not None:
        reg = reg.chunk({"time": -1})
        if max_gap:
            reg = reg.load()
    linear = reg.interpolate_na("time", method="linear")
    if max_gap:
        valid = reg.notnull().values
        days = ((reg.time.values - reg.time.values[0]) / np.timedelta64(1, "D")).reshape((-1,) + (1,) * (reg.ndim - 1))
        before = np.maximum.accumulate(np.where(valid, days, -np.inf), axis=0)   # the last day with a value up to each period
        after = np.minimum.accumulate(np.where(valid, days, np.inf)[::-1], axis=0)[::-1]   # and the next one
        linear = linear.where(reg.copy(data=valid | (after - before <= max_gap)))
    if how == "linear":
        return linear
    from ._smooth import smooth

    return reg.where(reg.notnull() | linear.isnull(), smooth(reg, lmbda=lmbda, nodata=None))


def _medoid(cube: xr.DataArray, freq: str) -> xr.DataArray:
    """The medoid of each period by ``_core.utils.compute_medoid`` (in parallel over rows):
    per pixel, the median of each band over the period's valid values, then the observation
    with every band whose distance to it is smallest (the first one on a tie); NaN without one."""
    from . import _core

    single = "band" not in cube.dims
    work = cube.expand_dims(band=["value"], axis=1) if single else cube
    others = [d for d in work.dims if d not in ("time", "band")]
    if work.chunks is not None:
        work = work.chunk({"time": -1, "band": -1})

    def _block(values: np.ndarray) -> np.ndarray:
        # (..., time, band), the other dims first; the engine takes (rows, cols, time, band)
        lead = values.shape[:-2]
        flat = np.ascontiguousarray(values.reshape((-1, 1) + values.shape[-2:]), dtype=np.float64)
        out = _core.utils.compute_medoid(flat, np.nan)   # only NaN is missing (NoData is NaN by now)
        return np.asarray(out).reshape(lead + (values.shape[-1],))

    def _period(group: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(_block, group, input_core_dims=[["time", "band"]], output_core_dims=[["band"]],
                              dask="parallelized", output_dtypes=[np.float64])

    out = work.resample(time=freq).map(_period)
    out = out.transpose("time", "band", *others)
    return out.isel(band=0, drop=True) if single else out
