"""The omnibus change test for a Sentinel-1 series: ``zeit.omnibus``, ``zeit.estimate_enl``.

For each pixel a likelihood-ratio test of the equality of the backscatter (VV and VH, or one band) of all the dates of
a series is made, and, where it rejects, factorized into the tests of each date against those before it, which gives
the dates of the changes; after a change the series restarts at it (Conradsen, Nielsen and Skriver 2016, IEEE TGRS
54:3007-3024; Canty, Nielsen, Conradsen and Skriver 2020, Remote Sensing 12:46). The p-values use the improved
approximations of Conradsen, Skriver and Nielsen (2024, IEEE JSTARS 17:16359-16376). The direction of each change is
the Loewner order of the difference of the backscatter (Nielsen, Skriver and Conradsen 2019).
"""

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from . import _change as ch

PLANES = ("first_change", "last_change", "n_changes", "first_direction", "last_direction", "greatest_change",
          "p_change", "n_dates")
_DATE_PLANES = (0, 1, 5)
_ALIASES = {"vv": ("vv", "sigma0_vv", "gamma0_vv", "s1_vv", "vv_linear"),
            "vh": ("vh", "sigma0_vh", "gamma0_vh", "s1_vh", "vh_linear")}


def _decimal_years(time: xr.DataArray) -> Tuple[np.ndarray, str]:
    """The time values as decimal years (float64) and what they are."""
    values = np.asarray(time.values)
    if np.issubdtype(values.dtype, np.datetime64):
        import pandas as pd

        stamps = pd.DatetimeIndex(values)
        start = pd.DatetimeIndex([pd.Timestamp(year=y, month=1, day=1) for y in stamps.year])
        end = pd.DatetimeIndex([pd.Timestamp(year=y + 1, month=1, day=1) for y in stamps.year])
        return np.asarray(stamps.year.values + (stamps - start) / (end - start), dtype=np.float64), "decimal year"
    return values.astype(np.float64), "time coordinate"


def _positions(cube: xr.DataArray, bands: Sequence[Any]) -> list:
    names = [str(b) for b in cube.band.values]
    lower = [n.lower() for n in names]
    out = []
    for b in bands:
        if isinstance(b, (int, np.integer)) and not isinstance(b, bool):
            out.append(int(b))
            continue
        key = str(b).lower()
        for cand in _ALIASES.get(key, (key,)):
            if cand in lower:
                out.append(lower.index(cand))
                break
        else:
            raise ValueError(f"zeit.omnibus: the cube has no band {b!r} (bands: {names}); pass bands= naming the "
                             "VV (and VH) bands")
    if len(set(out)) != len(out):
        raise ValueError("bands= names the same band twice")
    return out


def _block(arr: np.ndarray, bands: Sequence[int], enl: float, alpha: float, approximation: str, per_date: bool,
           missing: Sequence[float], dates: np.ndarray, threads: int = 1) -> np.ndarray:
    """``(time, band, y, x)`` -> ``(8 [+ time], y, x)`` float32: the planes (dates as the values of `dates`) and with
    `per_date` the direction at each date, by the compiled core."""
    from . import _core

    arr = ch.kernel_array(arr)
    planes, per = _core.omnibus.apply(arr, [int(b) for b in bands], float(enl), float(alpha), approximation,
                                      bool(per_date), [float(v) for v in missing], int(threads))
    planes = np.asarray(planes)
    dates32 = np.asarray(dates, dtype=np.float32)
    for k in _DATE_PLANES:
        idx = planes[k]
        ok = np.isfinite(idx)
        planes[k] = np.where(ok, dates32[np.where(ok, idx, 0).astype(np.int64)], np.float32(np.nan))
    return np.concatenate([planes, np.asarray(per)], axis=0) if per is not None else planes


def _pvalue(z: float, f: float, w2: float) -> float:
    from scipy.special import gammaincc

    qf = gammaincc(f / 2.0, z / 2.0)
    return float(np.clip(qf - w2 * (qf - gammaincc((f + 4.0) / 2.0, z / 2.0)), 0.0, 1.0)) if w2 else float(qf)


def _block_numpy(arr: np.ndarray, bands: Sequence[int], enl: float, alpha: float, approximation: str,
                 per_date: bool, missing: Sequence[float], dates: np.ndarray, threads: int = 1) -> np.ndarray:
    """What ``_block`` computes, one pixel at a time in Python (and scipy's chi-square): the reference of the tests."""
    from math import log

    nt, _, ny, nx = arr.shape
    p = len(bands)
    out = np.full((8 + (nt if per_date else 0), ny, nx), np.nan, dtype=np.float32)
    n = float(enl)
    chi = approximation == "chi2"
    for y in range(ny):
        for x in range(nx):
            vals = arr[:, list(bands), y, x].astype(np.float64)
            ok = np.isfinite(vals).all(1) & (vals > 0).all(1)
            for m in missing:
                ok &= (vals != m).all(1)
            idx = np.flatnonzero(ok)
            s = vals[idx]
            k = len(idx)
            if k < 2:
                continue
            changes = []
            pch = np.nan
            l0 = 0
            while k - l0 >= 2:
                m = k - l0
                lnq = n * sum(m * log(m) + np.log(s[l0:, b]).sum() - m * log(s[l0:, b].sum()) for b in range(p))
                if chi:
                    f, rho, w2 = (m - 1.0) * p, 1.0, 0.0
                else:
                    f = (m - 1.0) * p
                    rho = 1 - (1 / (6 * (m - 1))) * (m / n - 1 / (n * m))
                    w2 = -(f / 4) * (1 - 1 / rho) ** 2
                pq = _pvalue(max(-2 * rho * lnq, 0.0), f, w2)
                if l0 == 0:
                    pch = pq
                if not pq < alpha:
                    break
                acc = s[l0].copy()
                found = False
                for j in range(2, m + 1):
                    t = l0 + j - 1
                    c = s[t]
                    lnr = n * sum(j * log(j) - (j - 1) * log(j - 1) + (j - 1) * log(acc[b]) + log(c[b])
                                  - j * log(acc[b] + c[b]) for b in range(p))
                    if chi:
                        g, rj, w2j = float(p), 1.0, 0.0
                    else:
                        g = float(p)
                        rj = 1 - (1 / (6 * n)) * (1 + 1 / (j * (j - 1)))
                        w2j = -(g / 4) * (1 - 1 / rj) ** 2
                    zj = max(-2 * rj * lnr, 0.0)
                    if _pvalue(zj, g, w2j) < alpha:
                        d = c - acc / (j - 1)
                        direction = 1 if (d > 0).all() else 2 if (d < 0).all() else 3
                        changes.append((t, direction, zj))
                        l0, found = t, True
                        break
                    acc += c
                if not found:
                    break
            out[7, y, x] = k
            out[6, y, x] = pch
            out[2, y, x] = len(changes)
            if changes:
                best = max(changes, key=lambda c: c[2])
                out[0, y, x] = dates[idx[changes[0][0]]]
                out[1, y, x] = dates[idx[changes[-1][0]]]
                out[3, y, x], out[4, y, x] = changes[0][1], changes[-1][1]
                out[5, y, x] = dates[idx[best[0]]]
            else:
                out[3, y, x] = out[4, y, x] = 0
            if per_date:
                out[8:, y, x] = np.nan
                out[8 + idx, y, x] = 0
                for t, d, _ in changes:
                    out[8 + idx[t], y, x] = d
    return out


def omnibus(
    data: Any,
    *,
    bands: Sequence[Any] = ("vv", "vh"),
    enl: float = 4.4,
    alpha: float = 0.01,
    approximation: str = "improved",
    per_date: bool = False,
    median: int = 0,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """When the radar backscatter of each pixel of a Sentinel-1 series changed: the omnibus change test.

    Each pixel's series of VV and VH backscatter (linear power, sigma0 or gamma0, not dB) is tested for the equality
    of all its dates by the likelihood ratio of independent complex Wishart (multilook) observations
    (Conradsen et al. 2016; Canty et al. 2020). If it rejects at the significance `alpha`, the statistic factorizes
    into one test for each date against the dates before it (since the last change): the first one that rejects is a
    change, the series then restarts at that date, and so on, so that every registered change has the false alarm
    probability `alpha` (Conradsen et al. 2016, Canty et al. 2020). The p-values of the diagonal case use the improved
    approximations of Conradsen, Skriver and Nielsen (2024), more accurate than Wilks' chi-square at few looks. The
    direction of a change is brightening (the backscatter went up in every channel), darkening (down in every
    channel) or indefinite (up in one, down in the other). The statistics are invariant to a per-pixel factor
    constant over the series, so sigma0 and the terrain-flattened gamma0 of one relative orbit give the same result.

    A test of one new date against the mean of those before it has limited power: a change is found at its own date
    in a part of the pixels and a date or two later in the rest, and about ``(j - 1) * alpha`` of the pixels report a
    change too early.

    Parameters
    ----------
    data
        A cube ``(time, band, y, x)`` of linear backscatter power (or anything ``zeit.load_raster`` reads), of one
        relative orbit and pass: the geometry changes the backscatter, and the test reads that as change (a warning
        if the cube's ``sat:relative_orbit`` or ``sat:orbit_state`` coordinates vary). Values that are not positive
        (layover and shadow in RTC products), NaN or the nodata value are skipped date by date: a pixel uses its
        valid dates, a change is dated at the next valid date, and a pixel with fewer than two valid dates is NaN.
    bands
        The VV band and, for dual polarization, the VH band, by name (``"vv"``, ``"vh"``, as Planetary Computer's
        ``sentinel-1-rtc`` has them) or position; ``("vv",)`` for a single polarization.
    enl
        The equivalent number of looks of the speckle (default 4.4, the usual value of Sentinel-1 IW GRDH). RTC
        products are resampled and have another one: estimate it on a homogeneous area with ``zeit.estimate_enl``.
    alpha
        The significance level of each test (default 0.01).
    approximation
        ``"improved"`` (default) or ``"chi2"``, Wilks' chi-square of ``-2 ln Q`` as the Earth Engine tutorial of
        Canty et al. uses it (more false alarms at few looks).
    per_date
        Also return the direction of the change at each date: ``(time, y, x)`` with 1, 2, 3 at the dates of the
        changes, 0 elsewhere, NaN where the date is not valid.
    median
        The median filter of the p-values of Canty's scripts is not implemented: it must be 0.
    nodata, chunks, n_jobs
        As in :func:`zeit.imad`.

    Returns
    -------
    xarray.DataArray
        ``(band, y, x)`` float32 with the bands ``first_change``, ``last_change``, ``greatest_change`` (the change
        with the largest test statistic) as decimal years (NaN where nothing changed), ``n_changes``,
        ``first_direction``, ``last_direction`` (0 nothing, 1 brightening, 2 darkening, 3 indefinite), ``p_change``
        (the p-value of the omnibus test of the whole series) and ``n_dates`` (the valid dates used), in that order:
        ``first_change``, ``last_change``, ``n_changes``, ``first_direction``, ``last_direction``,
        ``greatest_change``, ``p_change``, ``n_dates``. ``attrs["omnibus"]`` is a JSON record (the ENL, alpha, the
        approximation and the dates). With ``per_date`` a tuple ``(summary, per_date)``.

    Examples
    --------
    >>> change = zeit.omnibus(s1, enl=4.4, alpha=0.01)
    >>> change.sel(band="first_change")
    """
    from ._threads import resolve

    who = "zeit.omnibus"
    if approximation not in ("improved", "chi2"):
        raise ValueError(f"approximation= must be 'improved' or 'chi2', got {approximation!r}")
    if int(median) != 0:
        raise ValueError("median= is not implemented (the median filter of the p-values is not rigorous); use 0")
    if not float(enl) > 0:
        raise ValueError(f"enl= must be positive, got {enl!r}")
    if not 0 < float(alpha) < 1:
        raise ValueError(f"alpha= must be between 0 and 1, got {alpha!r}")
    cube = ch.to_cube(data, who, chunks)
    if not {"time", "band", "y", "x"} <= set(cube.dims):
        raise ValueError("zeit.omnibus needs a cube (time, band, y, x)")
    if cube.sizes["time"] < 2:
        raise ValueError("zeit.omnibus needs at least two dates")
    pos = _positions(cube, [bands] if isinstance(bands, (str, int)) else bands)
    if len(pos) not in (1, 2):
        raise ValueError("bands= must be one (VV) or two (VV, VH) bands")
    for key in ("sat:relative_orbit", "relative_orbit", "sat:orbit_state", "orbit_state"):
        if key in cube.coords and np.unique(np.asarray(cube.coords[key].values)).size > 1:
            warnings.warn(f"{who}: the series mixes {key} values; the backscatter depends on the viewing geometry "
                          "and the test reads that difference as change. Use one relative orbit and pass.",
                          stacklevel=2)
            break
    work = cube.transpose("time", "band", "y", "x")
    probe = work.isel(band=pos, y=slice(None, None, max(1, work.sizes["y"] // 64)),
                      x=slice(None, None, max(1, work.sizes["x"] // 64)),
                      time=slice(None, None, max(1, work.sizes["time"] // 8)))
    values = np.asarray(probe.values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size and np.median(values) < 0:
        raise ValueError(f"{who} needs linear power: the median of the bands is negative, which means dB. Convert "
                         "with 10 ** (dB / 10) first")
    missing = ch.missing_values(work, nodata)
    dates, units = _decimal_years(work.time)
    threads = resolve(n_jobs)
    nt = work.sizes["time"]
    if work.chunks is None:
        stacked = _block(work.values, pos, enl, alpha, approximation, per_date, missing, dates, threads)
    else:
        import dask.array as da

        arr = work.data.rechunk({0: -1, 1: -1})
        stacked = da.map_blocks(_block, arr, drop_axis=[1], bands=pos, enl=float(enl), alpha=float(alpha),
                                approximation=approximation, per_date=bool(per_date), missing=list(missing),
                                dates=dates, threads=1, dtype=np.float32,
                                chunks=((8 + (nt if per_date else 0),),) + arr.chunks[2:],
                                meta=np.empty((0, 0, 0), np.float32))
    template = work.isel(time=0, drop=True).isel(band=[0])
    names = [str(work.band.values[i]) for i in pos]
    record = {"enl": float(enl), "alpha": float(alpha), "approximation": approximation, "bands": names,
              "time_units": units, "dates": [float(d) for d in dates], "n_dates": int(nt)}
    summary = ch.result_array(stacked[:8], PLANES, template, "omnibus", record, "omnibus")
    summary.attrs["time_units"] = units
    if not per_date:
        return summary
    coords = {k: v for k, v in work.coords.items() if "band" not in v.dims}
    direction = xr.DataArray(stacked[8:], dims=("time", "y", "x"), coords=coords, name="change_direction")
    direction.attrs = {"omnibus": summary.attrs["omnibus"],
                       "long_name": "direction of the change at each date: 1 brightening, 2 darkening, "
                                    "3 indefinite, 0 none"}
    if work.rio.crs is not None:
        direction = direction.rio.write_crs(work.rio.crs)
    return summary, direction


def estimate_enl(
    data: Any,
    *,
    bands: Sequence[Any] = ("vv", "vh"),
    region: Any = None,
) -> Dict[str, Any]:
    """The equivalent number of looks of the speckle of a Sentinel-1 series over a homogeneous area.

    On a uniform target the intensity of a ``L``-look image has the mean ``m`` and the variance ``m**2 / L``, so
    ``L = mean**2 / variance`` (the moment estimate of Canty's monograph); it is made over the area for each date and
    band and the median over the dates is the estimate. It is biased down by any texture, trend or edge in the area
    (choose intact forest or calm water), and the better estimators for polarimetric data of Anfinsen, Doulgeris and
    Eltoft (2009, IEEE TGRS 47:3795-3809) are not implemented. Terrain-corrected (RTC) products are resampled and
    filtered, so their ENL differs from the 4.4 of GRDH and is spatially correlated.

    Parameters
    ----------
    data
        A cube ``(time, band, y, x)`` (or ``(band, y, x)``) of linear power.
    bands
        As in :func:`omnibus`.
    region
        The homogeneous area: ``None`` (the whole image), a pixel window ``(row0, row1, col0, col1)`` (end
        exclusive) or a boolean mask ``(y, x)``.

    Returns
    -------
    dict
        ``{"enl": {band: median over the dates}, "per_date": {band: [one per date]}, "n_pixels": count}``.
    """
    cube = ch.to_cube(data, "zeit.estimate_enl")
    if "time" not in cube.dims:
        cube = cube.expand_dims(time=[0])
    cube = cube.transpose("time", "band", "y", "x")
    pos = _positions(cube, [bands] if isinstance(bands, (str, int)) else bands)
    values = cube.isel(band=pos)
    if region is None:
        mask = np.ones((cube.sizes["y"], cube.sizes["x"]), bool)
    elif isinstance(region, (tuple, list)) and len(region) == 4 and not hasattr(region[0], "__len__"):
        r0, r1, c0, c1 = (int(v) for v in region)
        mask = np.zeros((cube.sizes["y"], cube.sizes["x"]), bool)
        mask[r0:r1, c0:c1] = True
    else:
        mask = np.asarray(region, dtype=bool)
        if mask.shape != (cube.sizes["y"], cube.sizes["x"]):
            raise ValueError(f"region= mask must be {(cube.sizes['y'], cube.sizes['x'])}, got {mask.shape}")
    sel = np.asarray(values.values, dtype=np.float64)[..., mask]
    names = [str(cube.band.values[i]) for i in pos]
    per: Dict[str, list] = {}
    final: Dict[str, float] = {}
    for b, name in enumerate(names):
        estimates = []
        for t in range(sel.shape[0]):
            v = sel[t, b]
            v = v[np.isfinite(v) & (v > 0)]
            estimates.append(float(v.mean() ** 2 / v.var()) if v.size > 2 and v.var() > 0 else float("nan"))
        per[name] = estimates
        final[name] = float(np.nanmedian(estimates)) if np.isfinite(estimates).any() else float("nan")
    return {"enl": final, "per_date": per, "n_pixels": int(mask.sum())}
