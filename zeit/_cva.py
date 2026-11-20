"""Change vector analysis: ``zeit.cva``.

The change vector of Malila (1980) is the vector of the differences of the bands of two dates; its length, the
magnitude, says how much a pixel changed and its direction what kind of change it was. The direction here is the
angle to the diagonal of uniform change of the compressed CVA of Bovolo, Marchesi and Bruzzone (2012, IEEE TGRS
50:2196-2212) for any number of bands, the polar angle of two bands (Bovolo and Bruzzone 2007, IEEE TGRS 45:218-236)
or the sector of the signs of the changes of red, NIR and SWIR1. The magnitude is thresholded by a fixed value, Otsu's
method (Otsu 1979) or the minimum-error threshold of a two-class Gaussian mixture fitted by EM (Bruzzone and Prieto
2000, IEEE TGRS 38:1171-1182).
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _change as ch

DIRECTIONS = ("c2va", "polar", "sign")
ROLES = ("red", "nir", "swir1")

#: The sectors of the signs of the changes of red, NIR and SWIR1: ``1 + [red up] + 2 [NIR up] + 4 [SWIR1 up]``
#: (a change of exactly 0 is not up).
SIGN_LABELS = {
    1: "Darker in all bands (water, flooding, wetting, shadow)",
    2: "Red up, NIR and SWIR down (turbid water, other)",
    3: "Vegetation gain (greening, regrowth)",
    4: "Visible and NIR up, SWIR down (other)",
    5: "NIR down, SWIR up (burn scar, drying vegetation)",
    6: "Vegetation loss (clearing to soil)",
    7: "NIR and SWIR up, red down (other)",
    8: "Brighter in all bands (bare soil, built-up, bright surfaces)",
}
#: The four sectors of the polar angle of (brightness, greenness) (or any two bands: right, up, left, down).
POLAR_LABELS_4 = {
    1: "Brighter and greener (both bands up)",
    2: "Darker and greener",
    3: "Darker and less green (both bands down)",
    4: "Brighter and less green (vegetation loss to soil)",
}


def _roles_of(names: List[str], roles: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """The position in `names` of each of red, NIR and SWIR1: the given ones, else the usual names; None when a
    role is not found."""
    from ._bandpass import _NAMES

    lower = [n.lower() for n in names]
    out: Dict[str, int] = {}
    for role in ROLES:
        if roles and role in roles:
            wanted = str(roles[role])
            if wanted not in names:
                raise ValueError(f"roles=: {wanted!r} is not one of the bands {names}")
            out[role] = names.index(wanted)
            continue
        for candidate in _NAMES[role]:
            if candidate.lower() in lower and lower.index(candidate.lower()) not in out.values():
                out[role] = lower.index(candidate.lower())
                break
    return out if len(out) == 3 else None


def cva(
    data: Any,
    after: Any = None,
    *,
    dates: Sequence[Any] = (0, -1),
    bands: Optional[Sequence[Any]] = None,
    roles: Optional[Dict[str, Any]] = None,
    direction: str = "c2va",
    polar_bands: Optional[Sequence[Any]] = None,
    n_sectors: int = 4,
    c2va_sectors: Optional[Sequence[float]] = None,
    labels: Optional[Dict[int, str]] = None,
    scale: str = "none",
    threshold: Any = "em",
    em_init: float = 0.2,
    sample: Optional[int] = 1_000_000,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.DataArray:
    """Change vector analysis of two dates: the magnitude and the direction of the change in every band at once.

    For each pixel the change vector ``d`` holds the difference (after minus before) of each band, divided by the
    standard deviation of that difference when ``scale="std"``. Its length ``rho = sqrt(sum d_b**2)`` is the
    magnitude (Malila 1980); a pixel is flagged as changed when ``rho`` exceeds the threshold. The direction is one
    of three descriptions:

    ``"c2va"`` (default)
        The angle between ``d`` and the diagonal of uniform change, ``acos(sum(d) / (sqrt(B) rho))``, from 0 degrees
        (every band up by the same amount, uniform brightening) through 90 (the changes sum to zero) to 180 (every
        band down by the same amount), for any number of bands (the compressed CVA of Bovolo et al. 2012). A pixel
        with no change at all has no direction (NaN).
    ``"polar"``
        The angle ``atan2(d_2, d_1)`` in degrees, 0 to 360, of two bands (``polar_bands``, for example Tasseled Cap
        brightness and greenness), and ``n_sectors`` equal sectors from 0 degrees (Malila 1980; Bovolo and Bruzzone
        2007). With 4 sectors: 1 both up, 2 the second up and the first down, 3 both down, 4 the first up and the
        second down.
    ``"sign"``
        The sector (1 to 8) of the signs of the changes of red, NIR and SWIR1: ``1 + [red up] + 2 [NIR up] + 4
        [SWIR1 up]``, where 6 is vegetation loss (clearing to soil), 3 vegetation gain, 5 a burn scar, 1 darker in
        every band and 8 brighter in every band. A change of exactly 0 is not up. The labels are a heuristic of
        what such changes usually are.

    The dates must be radiometrically consistent: normalize them first with ``zeit.radcal`` if they are not.

    Parameters
    ----------
    data, after, dates, bands, nodata, chunks, n_jobs
        As in :func:`zeit.imad`: a cube with `dates`, or two dates as `data` and `after`; `bands` the bands the vector
        is made of (default all).
    roles
        For ``direction="sign"``: the band of each of ``red``, ``nir``, ``swir1`` when its name is not a usual one
        (all three must be among the bands used).
    direction
        ``"c2va"``, ``"polar"`` or ``"sign"``.
    polar_bands
        For ``"polar"``: the two bands (names or indices among those used).
    n_sectors
        The sectors of the polar angle.
    c2va_sectors
        For ``"c2va"``: the edges in degrees of sectors of the angle, for example ``[0, 60, 120, 180]`` (3 sectors).
        Without it ``"c2va"`` has no sector band.
    labels
        ``{sector: name}`` overriding the default labels.
    scale
        ``"none"`` (default): the differences as they are; ``"std"``: each band's difference divided by its standard
        deviation over the valid pixels of the sample, so that every band weighs the same. A band whose difference
        is constant gets weight 0.
    threshold
        The threshold of the magnitude: a number (in the units of the magnitude, after the scale), ``"otsu"`` or
        ``"em"`` (default). The EM threshold fits two Gaussian classes (no change and change) to the magnitudes of the
        sample and takes the root of the equation of the minimum error of Bayes' rule between their means; if the
        mixture has none, Otsu's value is used.
    em_init
        The width of the EM seed sets around half the magnitude range, 0.2 as the paper.
    sample
        The threshold is estimated on a lattice of this many valid pixels (``None``: all); the output is for every
        pixel.

    Returns
    -------
    xarray.DataArray
        ``(band, y, x)`` float32 with the bands ``magnitude``, ``direction`` (degrees; for ``"sign"`` the sector),
        ``change`` (1 or 0) and, where the direction has sectors, ``sector`` (0 where nothing changed, 1 to the number
        of sectors where it did). Missing pixels are NaN. ``attrs["cva"]`` is a JSON record: the threshold and
        how it was found (for EM the classes' shares, means and standard deviations, the iterations and whether it
        converged; for Otsu the between-class variance), the scale of each band, and ``sectors``, one row per sector
        with its label, pixel count, area in hectares (when the grid is in metres) and mean magnitude.
        ``zeit.cva_table`` turns the sectors into a table.

    Examples
    --------
    >>> out = zeit.cva(cube, dates=("2021-07-15", "2022-07-20"), direction="sign", threshold="otsu")
    >>> out.sel(band="sector")
    """
    from . import _core
    from ._threads import resolve

    who = "zeit.cva"
    if direction not in DIRECTIONS:
        raise ValueError(f"direction= must be one of {DIRECTIONS}, got {direction!r}")
    if scale not in ("none", "std"):
        raise ValueError(f"scale= must be 'none' or 'std', got {scale!r}")
    x, y = ch.two_dates(data, after, dates, bands, who, chunks)
    names = [str(b) for b in x.band.values]
    nb = len(names)
    use = list(range(nb))
    missing = ch.missing_values(x, nodata)
    threads = resolve(n_jobs)

    kwargs: Dict[str, Any] = {"direction": direction}
    edges = None
    if direction == "polar":
        if polar_bands is None:
            if nb != 2:
                raise ValueError("direction='polar' needs polar_bands= (two bands) when more than two are used")
            polar_bands = names
        picks = []
        for b in polar_bands:
            if isinstance(b, (int, np.integer)) and str(b) not in names:
                picks.append(int(b))
            elif str(b) in names:
                picks.append(names.index(str(b)))
            else:
                raise ValueError(f"polar_bands=: {b!r} is not one of the bands {names}")
        if len(picks) != 2 or picks[0] == picks[1]:
            raise ValueError("polar_bands= must be two different bands")
        if int(n_sectors) < 1:
            raise ValueError(f"n_sectors= must be at least 1, got {n_sectors!r}")
        kwargs.update(polar_bands=(picks[0], picks[1]), n_sectors=int(n_sectors))
    elif direction == "sign":
        found = _roles_of(names, roles)
        if found is None:
            raise ValueError(f"direction='sign' needs the bands red, NIR and SWIR1: the bands {names} do not have "
                             "all three; pass roles={'red': ..., 'nir': ..., 'swir1': ...}")
        kwargs["roles"] = (found["red"], found["nir"], found["swir1"])
    else:
        if c2va_sectors is not None:
            edges = [float(e) for e in c2va_sectors]
            if len(edges) < 2 or edges != sorted(edges) or edges[0] != 0 or edges[-1] != 180:
                raise ValueError("c2va_sectors= must be ascending edges in degrees from 0 to 180, e.g. [0, 60, 120, 180]")
            kwargs["edges"] = edges
        if nb < 2:
            raise ValueError("direction='c2va' needs at least two bands (use direction='polar' or a single band's "
                             "magnitude)")

    # the threshold, on the lattice sample of the magnitudes
    stride = ch.lattice_stride(x.sizes["y"], x.sizes["x"], sample)
    xs, ys = ch.kernel_array(ch.lattice(x, stride)), ch.kernel_array(ch.lattice(y, stride))
    if xs.dtype != ys.dtype:
        xs, ys = xs.astype(np.float32), ys.astype(np.float32)
    rho, scales = _core.cva.sample(xs, ys, use, scale == "std", 1, [float(m) for m in missing])
    if rho.size < 20:
        raise ValueError(f"{who}: only {rho.size} valid pixels in the sample")
    if scale == "std" and (scales == 0).any():
        import warnings

        warnings.warn(f"{who}: the difference of band(s) {[names[i] for i in np.flatnonzero(scales == 0)]} is "
                      "constant; its weight is 0", stacklevel=2)
    info: Dict[str, Any] = {}
    if isinstance(threshold, str):
        if threshold == "otsu":
            t, sb, width = _core.cva.otsu(rho, 1024, None)
            info = {"method": "otsu", "between_class_variance": sb, "bin_width": width}
        elif threshold == "em":
            t, fit = _core.cva.em(rho, float(em_init), 500, 1e-9)
            info = {"method": "em", **fit}
            if not fit["converged"]:
                import warnings

                warnings.warn(f"{who}: the EM fit of the magnitudes did not converge in {fit['iterations']} "
                              "iterations", stacklevel=2)
        else:
            raise ValueError(f"threshold= must be a number, 'otsu' or 'em', got {threshold!r}")
    else:
        t = float(threshold)
        info = {"method": "fixed"}
    info.update(threshold=float(t), n_sample=int(rho.size), stride=int(stride))

    planes = _apply(x, y, use, scales, kwargs, float(t), missing, threads)
    has_sector = direction != "c2va" or edges is not None
    out_names = ["magnitude", "direction", "change"] + (["sector"] if has_sector else [])
    if has_sector:
        values = planes
    else:
        values = planes[:3] if not hasattr(planes, "compute") else planes[:3]
    n_sectors_out = {"sign": 8, "polar": int(n_sectors), "c2va": len(edges) - 1 if edges else 0}[direction]
    names_of = dict(_default_labels(direction, n_sectors_out, edges))
    names_of.update({int(k): str(v) for k, v in (labels or {}).items()})
    record: Dict[str, Any] = {"threshold": info, "direction": direction, "bands": names,
                              "scale": [float(s) for s in scales], "scale_mode": scale, "dates": None}
    result = ch.result_array(values, out_names, x, "cva", record, "cva")
    if has_sector:
        record["sectors"] = _sector_table(result, names_of, n_sectors_out)
        result.attrs["cva"] = ch.dumps(record)
    return result


def _default_labels(direction: str, n: int, edges: Optional[Sequence[float]]) -> Dict[int, str]:
    if direction == "sign":
        return dict(SIGN_LABELS)
    if direction == "polar":
        if n == 4:
            return dict(POLAR_LABELS_4)
        step = 360.0 / n
        return {k: f"angle {int((k - 1) * step)}-{int(k * step)} degrees" for k in range(1, n + 1)}
    return {k + 1: f"angle {edges[k]:g}-{edges[k + 1]:g} degrees" for k in range(n)}


def _apply(x: xr.DataArray, y: xr.DataArray, use: List[int], scales: np.ndarray, kwargs: Dict[str, Any],
           threshold: float, missing: Sequence[float], threads: int):
    """The four planes over the whole image (a dask array if the dates are lazy)."""
    scale_list = [float(s) for s in scales]
    if x.chunks is None and y.chunks is None:
        return _block(x.values, y.values, use, scale_list, kwargs, threshold, list(missing), threads)
    import dask.array as da

    xa = x.data.rechunk({0: -1})
    ya = y.data.rechunk({0: -1, 1: xa.chunks[1], 2: xa.chunks[2]})
    return da.map_blocks(_block, xa, ya, use=use, scale=scale_list, kwargs=kwargs, threshold=threshold,
                         missing=list(missing), threads=1, dtype=np.float32, chunks=((4,),) + xa.chunks[1:],
                         meta=np.empty((0, 0, 0), np.float32))


def _block(xb: np.ndarray, yb: np.ndarray, use: List[int], scale: List[float], kwargs: Dict[str, Any],
           threshold: float, missing: List[float], threads: int = 1) -> np.ndarray:
    """``(band, y, x)`` of two dates -> ``(magnitude, direction, change, sector)`` float32 by the compiled core."""
    from . import _core

    xb, yb = ch.kernel_array(xb), ch.kernel_array(yb)
    if xb.dtype != yb.dtype:
        xb, yb = xb.astype(np.float32), yb.astype(np.float32)
    return np.asarray(_core.cva.apply(xb, yb, use, scale, threshold=threshold, missing=missing, n_jobs=int(threads),
                                      **kwargs))


def _block_numpy(xb: np.ndarray, yb: np.ndarray, use: List[int], scale: List[float], kwargs: Dict[str, Any],
                 threshold: float, missing: List[float], threads: int = 1) -> np.ndarray:
    """What ``_block`` computes, in numpy (the reference of the tests)."""
    d = (yb[use].astype(np.float64) - xb[use].astype(np.float64)) * np.asarray(scale)[:, None, None]
    ok = np.isfinite(xb[use]).all(0) & np.isfinite(yb[use]).all(0)
    for m in missing:
        ok &= (xb[use] != m).all(0) & (yb[use] != m).all(0)
    rho = np.sqrt((d ** 2).sum(0))
    changed = rho > threshold
    direction = kwargs["direction"]
    sector = np.full(rho.shape, np.nan)
    if direction == "c2va":
        with np.errstate(invalid="ignore", divide="ignore"):
            ang = np.degrees(np.arccos(np.clip(d.sum(0) / (np.sqrt(len(use)) * rho), -1, 1)))
        edges = kwargs.get("edges")
        if edges is not None:
            sector = np.where(rho > 0, np.searchsorted(np.asarray(edges[1:-1]), ang, side="right") + 1.0, 1.0)
    elif direction == "polar":
        b1, b2 = kwargs["polar_bands"]
        ang = np.degrees(np.arctan2(d[b2], d[b1])) % 360.0
        ns = kwargs["n_sectors"]
        sector = np.minimum(np.floor(ang / (360.0 / ns)) + 1, ns)
    else:
        r, n, s = kwargs["roles"]
        code = (d[r] > 0) * 1 + (d[n] > 0) * 2 + (d[s] > 0) * 4
        ang = code + 1.0
        sector = ang
    sector = np.where(changed, sector, np.where(np.isnan(sector), np.nan, 0.0))
    out = np.stack([rho, ang, changed.astype(np.float64), sector])
    out[:, ~ok] = np.nan
    return out.astype(np.float32)


def _sector_table(result: xr.DataArray, labels: Dict[int, str], n: int) -> List[Dict[str, Any]]:
    """One row per sector: label, pixel count, area (ha, when the grid is in metres) and mean magnitude."""
    sector = result.sel(band="sector")
    mag = result.sel(band="magnitude")
    if sector.chunks is not None:
        import dask

        sector, mag = dask.compute(sector.data, mag.data)
    else:
        sector, mag = sector.values, mag.values
    pixel = None
    try:
        crs = result.rio.crs
        if crs is not None and crs.is_projected:
            rx, ry = result.rio.resolution()
            unit = crs.linear_units_factor[1] if hasattr(crs, "linear_units_factor") else 1.0
            pixel = abs(rx * ry) * unit ** 2 / 1e4
    except Exception:
        pixel = None
    rows = []
    for k in range(0, n + 1):
        m = sector == k
        count = int(m.sum())
        rows.append({"sector": k, "label": "No change" if k == 0 else labels.get(k, f"sector {k}"), "pixels": count,
                     "area_ha": None if pixel is None else count * pixel,
                     "mean_magnitude": float(np.nanmean(mag[m])) if count else None})
    return rows


def cva_table(result: xr.DataArray) -> pd.DataFrame:
    """The sectors of a ``zeit.cva`` result: label, pixel count, area (ha) and mean magnitude of each."""
    import json

    record = json.loads(result.attrs["cva"])
    if "sectors" not in record:
        raise ValueError("this CVA has no sectors: use direction='sign' or 'polar', or c2va_sectors=")
    return pd.DataFrame(record["sectors"])
