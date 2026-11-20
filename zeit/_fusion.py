"""Spatiotemporal fusion of a fine and a coarse series: ``zeit.fuse_starfm``, ``zeit.fuse_estarfm``, ``zeit.fusion_plan``.

Landsat, HLS or Sentinel-2 images are sharp but rare; MODIS images are daily but 500 m. Where a fine image and a
coarse image of the same date exist (a *pair*), the methods predict the fine image of any other coarse date: STARFM
(Gao, Masek, Schwaller and Hall 2006, IEEE TGRS 44:2207-2218) and ESTARFM (Zhu, Chen, Gao, Chen and Masek 2010,
Remote Sensing of Environment 114:2610-2623). Both are written from their papers (``rust/zeit-core/src/fusion``);
``zeit.starfm``, the FORCE variant used by ``sentinel2_resmerge``, is a different program: FORCE's streamlined
STARFM for the Sentinel-2 20 -> 10 m merge, on one grid, with its own arithmetic.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

#: The bands that can be fused, as in ``tasseled_cap``.
ROLES = ("blue", "green", "red", "nir", "swir1", "swir2")
#: How the MODIS products of the Planetary Computer name the bands of each role (MODIS band 5, 1230-1250 nm, has no
#: counterpart and is never used).
MODIS_NAMES = {
    "blue": ("sur_refl_b03", "Nadir_Reflectance_Band3"),
    "green": ("sur_refl_b04", "Nadir_Reflectance_Band4"),
    "red": ("sur_refl_b01", "Nadir_Reflectance_Band1"),
    "nir": ("sur_refl_b02", "Nadir_Reflectance_Band2"),
    "swir1": ("sur_refl_b06", "Nadir_Reflectance_Band6"),
    "swir2": ("sur_refl_b07", "Nadir_Reflectance_Band7"),
}
CITATION = ("Gao, Masek, Schwaller and Hall (2006) IEEE TGRS 44(8):2207-2218, doi:10.1109/TGRS.2006.872081 (STARFM); "
            "Zhu, Chen, Gao, Chen and Masek (2010) Remote Sensing of Environment 114(11):2610-2623, "
            "doi:10.1016/j.rse.2010.05.032 (ESTARFM)")
_EPOCH = np.datetime64("1970-01-01", "D")
#: Rows per strip (None: from a 256 MB budget); the tests set it to make several strips of a small image.
_STRIP_ROWS: Optional[int] = None


def _days(values: Any) -> np.ndarray:
    return np.asarray(values).astype("datetime64[D]").astype("int64")


def _to_date(day: int) -> np.datetime64:
    return _EPOCH + np.timedelta64(int(day), "D")


def _names(extra: Dict[str, Sequence[str]]) -> Dict[str, Sequence[str]]:
    from ._bandpass import _NAMES

    return {r: tuple(extra.get(r, ())) + tuple(_NAMES[r]) for r in ROLES}


def _series(data: Any, what: str, chunks: Any) -> xr.DataArray:
    from ._embeddings import refuse
    from ._load import load_raster

    refuse(data, f"zeit.{what}")
    cube = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if not {"time", "band", "y", "x"} <= set(cube.dims):
        raise ValueError(f"{what}: the series must be (time, band, y, x), got {cube.dims}")
    if cube.rio.crs is None:
        raise ValueError(f"{what}: the series has no CRS (rioxarray `write_crs`): the coarse grid must be placed "
                         "on the fine one")
    if np.issubdtype(cube.dtype, np.integer):
        scale = cube.attrs.get("scale_factor", cube.encoding.get("scale_factor") if hasattr(cube, "encoding") else None)
        if "scale" in cube.coords or scale is None:
            raise ValueError(f"{what}: the series holds integers (digital numbers); fusion needs reflectance. Build "
                             "it with a float dtype, or give it a scale_factor attribute")
    return cube.transpose("time", "band", "y", "x")


def _mask(arr: np.ndarray, nodata: Any, cube: xr.DataArray, valid_range: Optional[Tuple[float, float]]) -> np.ndarray:
    """float32 copy with the NoData and the values outside the valid range as NaN."""
    scale = None
    if np.issubdtype(cube.dtype, np.integer):
        scale = cube.attrs.get("scale_factor", cube.encoding.get("scale_factor"))
    out = np.array(arr, dtype=np.float32)
    values = []
    if nodata == "auto":
        stored = cube.rio.nodata
        if stored is not None and not np.isnan(stored):
            values = [stored]
    elif nodata is not None:
        values = [nodata]
    if scale is not None:
        raw = np.asarray(arr)
        for v in values:
            out[raw == v] = np.nan
        out *= np.float32(scale)
    else:
        for v in values:
            out[out == np.float32(v)] = np.nan
    if valid_range is not None:
        out[(out < valid_range[0]) | (out > valid_range[1])] = np.nan
    return out


def _roles(fine: xr.DataArray, coarse: xr.DataArray, bands: Any) -> Tuple[List[str], List[int], List[int]]:
    """The fused roles and the position of each in the fine and in the coarse series."""
    from ._specadjust import _find

    want = list(ROLES)
    maps: Dict[str, Dict[str, Any]] = {"fine": {}, "coarse": {}}
    if isinstance(bands, dict) and ({"fine", "coarse"} & set(bands)):
        for k in ("fine", "coarse"):
            maps[k] = dict(bands.get(k) or {})
        want = [r for r in want if r in maps["fine"] or r in maps["coarse"]] or want
    elif isinstance(bands, dict):
        maps["fine"] = maps["coarse"] = dict(bands)
        want = [r for r in want if r in bands] or want
    elif bands is not None:
        want = [str(b) for b in bands]
        bad = [b for b in want if b not in ROLES]
        if bad:
            raise ValueError(f"bands=: unknown role {bad}; roles are {list(ROLES)}")
    fa = [str(b) for b in fine.band.values]
    ca = [str(b) for b in coarse.band.values]
    roles, fi, ci = [], [], []
    for r in want:
        try:
            f = _find(fine, maps["fine"], _names({}), (r,), "a fused band")[r]
            c = _find(coarse, maps["coarse"], _names(MODIS_NAMES), (r,), "a fused band")[r]
        except ValueError:
            continue
        roles.append(r)
        fi.append(f)
        ci.append(c)
    if not roles:
        raise ValueError(f"no band of the fine series {fa} has a counterpart in the coarse series {ca}; "
                         "pass bands= ({'fine': {role: band}, 'coarse': {role: band}})")
    return roles, fi, ci


def _windows(coarse: xr.DataArray) -> Optional[List[Tuple[int, int]]]:
    for s, e in (("start_datetime", "end_datetime"), ("start", "end")):
        if s in coarse.coords and e in coarse.coords and coarse[s].dims == ("time",):
            try:
                return list(zip(_days(coarse[s].values).tolist(), _days(coarse[e].values).tolist()))
            except (ValueError, TypeError):
                return None
    return None


def _valid_fraction(fine: xr.DataArray, fi: Sequence[int], valid_range: Optional[Tuple[float, float]]) -> np.ndarray:
    out = []
    for k in range(fine.sizes["time"]):
        a = np.asarray(fine.isel(time=k, band=list(fi)).values, dtype=np.float32)
        ok = np.isfinite(a)
        if valid_range is not None:
            ok &= (a >= valid_range[0]) & (a <= valid_range[1])
        out.append(float(ok.all(axis=0).mean()))
    return np.asarray(out)


def _rule(method: str, dates: Any, start: Any, end: Any, include_fine_dates: bool, keep_observed: bool,
          exclude_dates: Any, pair_tolerance: Any, min_pair_valid: float, max_gap: Optional[int],
          pairs: str) -> Dict[str, Any]:
    day = lambda v: int(_days([np.datetime64(v)])[0]) if v is not None else None
    listed = None
    if dates is not None and not (isinstance(dates, str) and dates == "coarse"):
        if isinstance(dates, str):
            raise ValueError("dates must be 'coarse' or a list of dates")
        listed = [day(d) for d in dates]
    tol = None if (pair_tolerance is None or pair_tolerance == "auto") else int(pair_tolerance)
    return dict(method=method, dates=listed, start=day(start), end=day(end), include_fine_dates=include_fine_dates,
                keep_observed=keep_observed, exclude=[day(d) for d in (exclude_dates or [])], tolerance=tol,
                min_valid=float(min_pair_valid), max_gap=None if max_gap is None else int(max_gap), pairs=pairs)


def _plan(fine: xr.DataArray, coarse: xr.DataArray, fi: Sequence[int], valid_range: Any, rule: Dict[str, Any]):
    from . import _core

    fine_days = _days(fine.time.values)
    coarse_days = _days(coarse.time.values)
    if len(np.unique(coarse_days)) != len(coarse_days) or len(np.unique(fine_days)) != len(fine_days):
        raise ValueError("two images of the same series have the same date: composite them first")
    order_f = np.argsort(fine_days, kind="stable")
    order_c = np.argsort(coarse_days, kind="stable")
    if (order_f != np.arange(len(order_f))).any() or (order_c != np.arange(len(order_c))).any():
        raise ValueError("the series must be sorted by date")
    valid = _valid_fraction(fine, fi, valid_range)
    wins = _windows(coarse)
    targets, skipped, unpaired = _core.fusion.plan(
        fine_days.tolist(), valid.tolist(), coarse_days.tolist(), wins, **rule)
    return targets, skipped, unpaired, fine_days, coarse_days, valid


def fusion_plan(fine: Any, coarse: Any, *, method: str = "starfm", dates: Any = "coarse", start: Any = None,
                end: Any = None, include_fine_dates: bool = True, keep_observed: bool = True,
                exclude_dates: Any = None, pairs: str = "bracket", pair_tolerance: Any = "auto",
                min_pair_valid: float = 0.3, max_gap: Optional[int] = None,
                bands: Any = None, valid_range: Optional[Tuple[float, float]] = (-0.01, 1.6),
                chunks: Any = None) -> pd.DataFrame:
    """The dates a fusion would output and the pairs each one is predicted from, without running it.

    Parameters are those of :func:`fuse_starfm` that choose pairs and dates (``method="estarfm"`` takes the two
    bracketing pairs and flags the extrapolated dates).

    Returns
    -------
    pandas.DataFrame
        One row per output date: ``date``, ``kind`` (``"observed"`` for a fine date whose valid pixels are kept,
        ``"predicted"``), ``coarse_date``, ``pairs`` (the fine dates of the pairs, comma separated) and
        ``extrapolated``. Dates that cannot be output are in ``df.attrs["skipped"]``, fine dates that are not pairs
        (clouded, excluded, no coarse image) in ``df.attrs["unpaired"]``.

    Examples
    --------
    >>> zeit.fusion_plan(landsat, modis, method="estarfm", exclude_dates=["2023-06-19"])
    """
    f = _series(fine, "fusion_plan", chunks)
    c = _series(coarse, "fusion_plan", chunks)
    roles, fi, ci = _roles(f, c, bands)
    rule = _rule(method, dates, start, end, include_fine_dates, keep_observed, exclude_dates, pair_tolerance,
                 min_pair_valid, max_gap, pairs)
    targets, skipped, unpaired, fdays, cdays, _ = _plan(f, c, fi, valid_range, rule)
    return _table(targets, skipped, unpaired, fdays, cdays)


def _table(targets, skipped, unpaired, fdays, cdays) -> pd.DataFrame:
    rows = []
    for day, cj, prs, obs, extra in targets:
        rows.append({"date": pd.Timestamp(_to_date(day)), "kind": "observed" if obs is not None else "predicted",
                     "coarse_date": pd.Timestamp(_to_date(cdays[cj])),
                     "pairs": ",".join(str(_to_date(fdays[k])) for k, _ in prs), "extrapolated": bool(extra)})
    df = pd.DataFrame(rows, columns=["date", "kind", "coarse_date", "pairs", "extrapolated"])
    df.attrs["skipped"] = [(str(_to_date(d)), why) for d, why in skipped]
    df.attrs["unpaired"] = [(str(_to_date(fdays[i])), why) for i, why in unpaired]
    return df


def _pixel(transform: Any) -> float:
    return float((abs(transform.a) + abs(transform.e)) / 2.0)


def _geometry(ft: Any, fcrs: Any, ct: Any, ccrs: Any, y0: int, ny: int, nx: int, shift: Tuple[float, float]):
    """Coarse pixel coordinates of the fine pixel centres of rows ``y0:y0 + ny``: ``("affine", six)`` or
    ``("lookup", u, v)``."""
    from affine import Affine

    inv = ~ct
    sh = Affine.translation(shift[0], shift[1])
    if ccrs == fcrs:
        m = inv * sh * ft * Affine.translation(0, y0)
        return ("affine", (m.c, m.a, m.b, m.f, m.d, m.e))
    from pyproj import Transformer

    tr = Transformer.from_crs(fcrs, ccrs, always_xy=True)
    xs = np.arange(nx) + 0.5
    ys = np.arange(y0, y0 + ny) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    X, Y = ft * (gx, gy)
    cx, cy = tr.transform(np.asarray(X), np.asarray(Y))
    cx = np.asarray(cx) + shift[0]
    cy = np.asarray(cy) + shift[1]
    u, v = inv * (cx, cy)
    return ("lookup", np.ascontiguousarray(u, dtype=np.float32).ravel(), np.ascontiguousarray(v, dtype=np.float32).ravel())


def _cell_px(ft: Any, fcrs: Any, ct: Any, ccrs: Any, ny: int, nx: int, shift: Tuple[float, float]) -> float:
    """Size of a coarse cell in fine pixels near the middle of the image."""
    g = _geometry(ft, fcrs, ct, ccrs, ny // 2, 2, nx, shift)
    if g[0] == "affine":
        a = g[1]
        det = abs(a[1] * a[5] - a[2] * a[4])
    else:
        u = g[1].reshape(2, nx)
        v = g[2].reshape(2, nx)
        k = nx // 2
        det = abs((u[0, k + 1] - u[0, k]) * (v[1, k] - v[0, k]) - (u[1, k] - u[0, k]) * (v[0, k + 1] - v[0, k]))
    return 1.0 / np.sqrt(det) if det > 0 else 1.0


def _run(method: str, fine: Any, coarse: Any, *, dates, start, end, include_fine_dates, keep_observed, exclude_dates,
         pairs, pair_tolerance, min_pair_valid, max_gap, radius, sigma, resampling, coarse_shift, fallback, bands,
         nodata, quality, chunks, n_jobs, valid_range, params, strip_rows=None):
    from . import _core
    from ._threads import resolve

    f = _series(fine, method, chunks)
    c = _series(coarse, method, chunks)
    roles, fi, ci = _roles(f, c, bands)
    nb = len(roles)
    if method == "estarfm" and 2 * nb < 4:
        warnings.warn("ESTARFM with fewer than 2 bands: the correlation of the spectra of a pixel has too few "
                      "values to weigh the neighbours; fuse the bands and compute the index afterwards, or raise "
                      "classes", stacklevel=3)
    rule = _rule(method, dates, start, end, include_fine_dates, keep_observed, exclude_dates, pair_tolerance,
                 min_pair_valid, max_gap, pairs)
    targets, skipped, unpaired, fdays, cdays, _ = _plan(f, c, fi, valid_range, rule)
    used_f = sorted({p[0] for t in targets for p in t[2]} | {t[3] for t in targets if t[3] is not None})
    used_c = sorted({p[1] for t in targets for p in t[2]} | {t[1] for t in targets})
    fmap = {k: i for i, k in enumerate(used_f)}
    cmap = {k: i for i, k in enumerate(used_c)}
    tg = [(day, cmap[cj], [(fmap[a], cmap[b]) for a, b in prs], None if obs is None else fmap[obs], bool(extra))
          for day, cj, prs, obs, extra in targets]

    ft, ct = f.rio.transform(), c.rio.transform()
    fcrs, ccrs = f.rio.crs, c.rio.crs
    ny, nx = f.sizes["y"], f.sizes["x"]
    res = _pixel(ft)
    radius_px = max(1, int(round(radius / res)))
    shift = (float(coarse_shift[0]), float(coarse_shift[1]))
    same_grid = (fcrs == ccrs and f.sizes["y"] == c.sizes["y"] and f.sizes["x"] == c.sizes["x"]
                 and all(abs(a - b) < 1e-9 * max(1.0, abs(a)) for a, b in zip(tuple(ft), tuple(ct))))
    params = dict(params)
    if same_grid and not np.any(np.asarray(shift)):
        if params.get("v_scope") == "cell":
            warnings.warn("the coarse series is on the fine grid: v_scope='cell' has no meaning, using 'window'",
                          stacklevel=3)
            params["v_scope"] = "window"
        if params.get("pure") == "cells":
            warnings.warn("the coarse series is on the fine grid: pure='cells' has no meaning, using 'epsilon'",
                          stacklevel=3)
            params["pure"] = "epsilon"
    if params.get("spatial_scale") is not None:
        params["spatial_scale_px"] = float(params["spatial_scale"]) / res
    params.pop("spatial_scale", None)

    cc = np.stack([_mask(c.isel(time=k, band=ci).values, nodata, c, valid_range) for k in used_c])
    cc = np.ascontiguousarray(cc)
    cell = _cell_px(ft, fcrs, ct, ccrs, ny, nx, shift)
    halo = _core.fusion.halo(method, radius_px, cell, params.get("v_scope", "window"))
    if method == "estarfm" and params.get("v_scope") == "cell":
        halo += 1
    # sigma of the image, one fine date at a time
    sig = np.empty((len(used_f), nb))
    reads = {}
    for i, k in enumerate(used_f):
        a = _mask(f.isel(time=k, band=fi).values, nodata, f, valid_range)[None]
        sig[i] = _core.fusion.image_sigma(a, (0, 0, nx, ny), resolve(n_jobs))
        if ny * nx * nb * len(used_f) * 4 < 1 << 30:
            reads[k] = a
    sig = np.ascontiguousarray(sig.ravel())

    nt = len(tg)
    out = np.empty((nt, nb, ny, nx), dtype=np.float32)
    qual = np.empty((nt, 2, ny, nx), dtype=np.uint16) if quality else None
    strip_rows = strip_rows or _STRIP_ROWS
    if strip_rows is None:
        per_row = nx * nb * len(used_f) * 4 * 3
        strip_rows = int(max(2 * halo + 16, min(ny, (256 << 20) // max(per_row, 1))))
    y = 0
    while y < ny:
        h = min(strip_rows, ny - y)
        ey0, ey1 = max(0, y - halo), min(ny, y + h + halo)
        fa = np.stack([(reads[k][0][:, ey0:ey1] if k in reads else
                        _mask(f.isel(time=k, band=fi, y=slice(ey0, ey1)).values, nodata, f, valid_range))
                       for k in used_f])
        geom = _geometry(ft, fcrs, ct, ccrs, ey0, ey1 - ey0, nx, shift)
        kw = dict(sigma_values=sig, radius_px=radius_px, sigma=sigma, resampling=resampling, fallback=fallback,
                  want_quality=bool(quality), n_jobs=int(resolve(n_jobs)), **params)
        if geom[0] == "affine":
            kw["affine"] = geom[1]
        else:
            kw["u"], kw["v"] = geom[1], geom[2]
        fn = _core.fusion.starfm if method == "starfm" else _core.fusion.estarfm
        o, q = fn(np.ascontiguousarray(fa), cc, tg, (0, y - ey0, nx, h), **kw)
        out[:, :, y:y + h] = o
        if quality:
            qual[:, :, y:y + h] = q
        y += h

    days = [t[0] for t in tg]
    coords = {k: v for k, v in f.coords.items() if k in ("y", "x", "spatial_ref")}
    coords["time"] = np.asarray([_to_date(d) for d in days], dtype="datetime64[ns]")
    coords["band"] = roles
    obs = [targets[i][3] is not None for i in range(nt)]
    coords["fusion_kind"] = ("time", ["observed" if o_ else "predicted" for o_ in obs])
    coords["fusion_pairs"] = ("time", [",".join(str(_to_date(fdays[a])) for a, _ in t[2]) for t in targets])
    coords["fusion_coarse_date"] = ("time", np.asarray([_to_date(cdays[t[1]]) for t in targets], dtype="datetime64[ns]"))
    coords["fusion_extrapolated"] = ("time", [bool(t[4]) for t in targets])
    result = xr.DataArray(out, dims=("time", "band", "y", "x"), coords=coords, name=f.name)
    result.attrs = dict(f.attrs)
    result.attrs["fusion"] = {
        "method": method, "radius_px": radius_px, "radius": radius, "resampling": resampling, "sigma": sigma,
        "fallback": fallback, "coarse_shift": shift, "coarse": str(c.attrs.get("collection", c.name or "coarse")),
        "parameters": {k: v for k, v in params.items()}, "citation": CITATION,
        "skipped": [(str(_to_date(d)), why) for d, why in skipped],
    }
    try:
        from . import __version__ as _v

        result.attrs["fusion"]["zeit"] = _v
    except ImportError:
        pass
    result = result.rio.write_crs(f.rio.crs)
    if f.rio.crs is not None:
        result = result.rio.write_transform(ft)
    if quality:
        q_da = xr.DataArray(qual, dims=("time", "band", "y", "x"),
                            coords={"time": coords["time"], "band": ["n_similar", "source"], "y": result.y, "x": result.x})
        return result, q_da.rio.write_crs(f.rio.crs)
    return result


def fuse_starfm(fine: Any, coarse: Any, *, dates: Any = "coarse", start: Any = None, end: Any = None,
           include_fine_dates: bool = True, keep_observed: bool = True, exclude_dates: Any = None,
           pairs: str = "bracket", pair_tolerance: Any = "auto", min_pair_valid: float = 0.3,
           max_gap: Optional[int] = None, radius: float = 750.0, classes: float = 40.0, sigma: str = "image",
           fine_uncertainty: float = 0.002, coarse_uncertainty: float = 0.005, spatial_scale: Optional[float] = None,
           weighting: str = "direct", filter: str = "both", pure: str = "epsilon", resampling: str = "bilinear",
           coarse_shift: Tuple[float, float] = (0.0, 0.0), fallback: str = "missing", bands: Any = None,
           nodata: Any = "auto", valid_range: Optional[Tuple[float, float]] = (-0.01, 1.6), quality: bool = False,
           chunks: Any = None, n_jobs: int = -1) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """STARFM: predicts a fine series (Landsat, HLS, Sentinel-2) on the dates of a coarse one (MODIS).

    STARFM (Gao, Masek, Schwaller and Hall 2006, IEEE TGRS 44:2207-2218, doi:10.1109/TGRS.2006.872081) predicts each
    band of a fine pixel as a weighted mean, over the neighbours of a window that look like it in the fine image of a
    pair, of the neighbour's fine value of the pair plus the coarse change from the pair's date to the predicted one.
    The weights favour neighbours whose coarse pixel matches their fine value (pure coarse pixels), whose coarse value
    changed little between the dates, and that are near. A neighbour is similar when it differs from the pixel by at
    most ``2 sigma / classes``, ``sigma`` the standard deviation of the fine image.

    A *pair* is a fine date with a coarse image of the same date (or composite). Each predicted date uses the nearest
    pair before and after it (``pairs="bracket"``) or the nearest one. On a fine date the valid observed pixels are
    kept and the clouded ones are predicted from the other pairs (``keep_observed``): gap filling.

    Parameters
    ----------
    fine
        Reflectance ``(time, band, y, x)``, in memory or dask, or anything ``zeit.load_raster`` reads, with a CRS.
        Integers are refused unless they carry a ``scale_factor``. NaN is missing (clouds already masked).
    coarse
        The coarse series in its own CRS and grid (a MODIS sinusoidal cube, for instance): same kind of input. Its
        bands are matched to the fine ones by role (blue, green, red, nir, swir1, swir2; MODIS ``sur_refl_b0N`` and
        ``Nadir_Reflectance_BandN`` known); a band missing in either is not fused. Per-band NaN is honoured.
    dates
        ``"coarse"`` (default): every coarse date from ``start`` to ``end`` (default the first and last fine date);
        or a list of dates, each snapped to the nearest coarse date.
    include_fine_dates, keep_observed
        Also output the fine dates (default), with the valid observed pixels kept (default) and the others
        predicted without that date's own pair.
    exclude_dates
        Fine dates never used as pairs (leave-one-date-out experiments).
    pairs
        ``"bracket"`` (default): the nearest pair on each side of a date; ``"nearest"``: the nearest one.
    pair_tolerance, min_pair_valid, max_gap
        A fine date pairs with the composite whose window contains it if the coarse series has ``start_datetime``
        and ``end_datetime`` coordinates, else with the nearest coarse date within ``pair_tolerance`` days
        (``"auto"``: half the median coarse interval); a fine image needs ``min_pair_valid`` of its pixels valid in
        every band; pairs farther than ``max_gap`` days are not used.
    radius
        Half the window, map units of the fine series, rounded to pixels (default 750 = 25 Landsat pixels).
    classes
        ``m`` of the similarity threshold (default 40, FORCE's value; Zhu et al. 2010 use 4).
    sigma
        ``"image"`` (default): standard deviation of the whole fine image of the pair, per band; ``"window"``.
    fine_uncertainty, coarse_uncertainty
        Reflectance uncertainties of the sample filter (default 0.002 and 0.005).
    spatial_scale
        ``A`` of the spatial distance ``1 + d / A``, map units (default ``radius``).
    weighting
        ``"direct"`` (default) or ``"log"`` (logarithmic, for heterogeneous areas).
    filter
        ``"both"`` (default): a neighbour must pass the spectral and the temporal test; ``"either"``; ``"none"``.
    pure
        ``"epsilon"`` (default): through the weights only; ``"tiers"``: exact matches first; ``"cells"``: only the
        neighbours of homogeneous coarse pixels when there are some.
    resampling
        How the coarse image is read at a fine pixel: ``"bilinear"`` (default) or ``"nearest"``.
    coarse_shift
        ``(dx, dy)`` in the coarse CRS: a displacement of the coarse grid (MODIS geolocation is about 50 m).
    fallback
        ``"missing"`` (default) or ``"coarse"``: write the coarse value where nothing can be predicted.
    bands
        ``None`` (all roles in both), a list of roles, or ``{"fine": {role: band}, "coarse": {role: band}}``.
    nodata, valid_range
        Values that mean missing besides NaN (``"auto"``: the raster's NoData); reflectance outside ``valid_range``
        is missing.
    quality
        Also return ``n_similar`` (minimum over the bands) and ``source`` (0 observed, 1 all pairs, 2 fewer pairs
        than planned, 3 coarse, 255 missing) as a uint16 ``(time, ["n_similar", "source"], y, x)``.
    chunks, n_jobs
        Chunks of inputs read from disk; threads (``-1``: the processors zeit may use).

    Returns
    -------
    xarray.DataArray
        float32 ``(time, band, y, x)`` on the fine grid; coordinates along ``time``: ``fusion_kind`` (``"observed"``
        or ``"predicted"``), ``fusion_pairs``, ``fusion_coarse_date``, ``fusion_extrapolated``; ``attrs["fusion"]``
        has the method, the parameters and the citation. With ``quality=True`` a tuple ``(prediction, quality)``.
        The result is in memory, computed in strips of rows with their halo.

    Examples
    --------
    >>> pred = zeit.fuse_starfm(landsat, modis, radius=900)
    >>> pred, q = zeit.fuse_starfm(landsat, modis, exclude_dates=["2023-06-19"], quality=True)
    """
    params = dict(classes=float(classes), fine_unc=float(fine_uncertainty), coarse_unc=float(coarse_uncertainty),
                  spatial_scale=spatial_scale, weighting=weighting, filter=filter, pure=pure)
    return _run("starfm", fine, coarse, dates=dates, start=start, end=end, include_fine_dates=include_fine_dates,
                keep_observed=keep_observed, exclude_dates=exclude_dates, pairs=pairs,
                pair_tolerance=pair_tolerance, min_pair_valid=min_pair_valid, max_gap=max_gap, radius=radius,
                sigma=sigma, resampling=resampling, coarse_shift=coarse_shift, fallback=fallback, bands=bands,
                nodata=nodata, quality=quality, chunks=chunks, n_jobs=n_jobs, valid_range=valid_range, params=params)


def fuse_estarfm(fine: Any, coarse: Any, *, dates: Any = "coarse", start: Any = None, end: Any = None,
            include_fine_dates: bool = True, keep_observed: bool = True, exclude_dates: Any = None,
            pair_tolerance: Any = "auto", min_pair_valid: float = 0.3, max_gap: Optional[int] = None,
            radius: float = 750.0, classes: float = 4.0, sigma: str = "image", v_scope: str = "window",
            alpha: float = 0.05, single_pair: str = "fallback", v_min_change: float = 0.02,
            v_allow_negative: bool = False, resampling: str = "bilinear",
            coarse_shift: Tuple[float, float] = (0.0, 0.0), fallback: str = "missing", bands: Any = None,
            nodata: Any = "auto", valid_range: Optional[Tuple[float, float]] = (-0.01, 1.6), quality: bool = False,
            chunks: Any = None, n_jobs: int = -1) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """ESTARFM: predicts a fine series on the dates of a coarse one from two pairs.

    ESTARFM (Zhu, Chen, Gao, Chen and Masek 2010, Remote Sensing of Environment 114:2610-2623,
    doi:10.1016/j.rse.2010.05.032) uses two pairs, all bands together. The neighbours of a pixel are those within
    ``2 sigma / classes`` in every band on both pair dates; each is weighted by the correlation of its fine and
    coarse spectra and by its distance; the coarse change of a neighbour is scaled by a conversion coefficient (the
    slope of a regression of fine on coarse reflectance over the similar pixels, 1 when it is not significant); the
    predictions from the two pairs are weighted by how much the coarse window changed since each pair.

    A date outside the two pairs is extrapolated from the two nearest ones (flagged in ``fusion_extrapolated``). A
    pixel clear in only one pair is predicted from that pair alone (``single_pair="fallback"``, ``source`` 2).

    Parameters are those of :func:`fuse_starfm` except those of its weights (``pairs``, ``classes`` defaults to 4,
    ``fine_uncertainty``, ``coarse_uncertainty``, ``spatial_scale``, ``weighting``, ``filter``, ``pure``), and:

    v_scope
        ``"window"`` (default): one conversion coefficient per window, from all its similar pixels;
        ``"cell"``: one per coarse pixel, from its similar pixels also beyond the window (the paper's text).
    alpha
        Significance of the slope; a slope not significant at this level becomes 1 (default 0.05).
    v_min_change, v_allow_negative
        The coefficient is regressed only where the coarse reflectance of the similar pixels changed between the two
        pairs by more than ``v_min_change`` on average (reflectance, default 0.02: the same threshold as Zhu's own
        program, found by comparing outputs), and a negative slope is replaced by 1 (``v_allow_negative=False``).
        Elsewhere the coefficient is 1. ``v_min_change=0, v_allow_negative=True`` is the paper's rule alone.
    single_pair
        ``"fallback"`` (default) or ``"missing"``.

    Returns
    -------
    xarray.DataArray
        As :func:`fuse_starfm`; ``source`` is 1 (two pairs), 4 (two pairs, a coefficient set to 1), 2 (one pair), 3
        (coarse), 255 (missing), 0 (observed).

    Examples
    --------
    >>> pred = zeit.fuse_estarfm(landsat, modis, radius=900, v_scope="cell")
    """
    params = dict(classes=float(classes), v_scope=v_scope, alpha=float(alpha), single_pair=single_pair,
                  min_change=float(v_min_change), allow_negative=bool(v_allow_negative))
    return _run("estarfm", fine, coarse, dates=dates, start=start, end=end, include_fine_dates=include_fine_dates,
                keep_observed=keep_observed, exclude_dates=exclude_dates, pairs="bracket",
                pair_tolerance=pair_tolerance, min_pair_valid=min_pair_valid, max_gap=max_gap, radius=radius,
                sigma=sigma, resampling=resampling, coarse_shift=coarse_shift, fallback=fallback, bands=bands,
                nodata=nodata, quality=quality, chunks=chunks, n_jobs=n_jobs, valid_range=valid_range, params=params)
