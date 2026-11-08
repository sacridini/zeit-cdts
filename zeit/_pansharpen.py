"""Pansharpening: ``zeit.pansharpen``, multispectral bands fused with a finer panchromatic band, for any sensor.

The published methods, as recalled and without the papers at hand (Gillespie et al. 1987 for Brovey; Aiazzi, Baronti and Selva 2007 for GSA; Schowengerdt 1980
and Gangkofner et al. 2008 for the high-pass filter; Chavez et al. 1991 for the principal components), see
``rust/zeit-core/src/pansharpen.rs`` for the equations and for what was decided where the papers do not fix it. The
bands are resampled to the pan's grid with the warp of ``load_raster(like=)`` (cubic) and fused by the compiled core
(``zeit._core.pansharpen``). Nothing here depends on a sensor: the number of bands, their names and the ratio of the
resolutions (any real above 1) come from the data.
"""

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import xarray as xr

METHODS = ("brovey", "gsa", "hpf", "pca")
_CORNER_TOLERANCE = 1e-3     # in cells of the multispectral grid
_RATIO_TOLERANCE = 1e-6


def _float32_with_nan(da: xr.DataArray, nodata: Any) -> np.ndarray:
    """The values as float32 with NaN where missing: NaN itself and the raster's NoData (``"auto"``; 0 for integer
    data without one), a number or None."""
    if isinstance(nodata, str) and nodata != "auto":
        raise ValueError(f"nodata must be 'auto', a number or None, got {nodata!r}")
    raw = np.asarray(da.values)
    out = raw.astype(np.float32)
    values = []
    if nodata == "auto":
        stored = da.rio.nodata
        if stored is not None and not np.isnan(stored):
            values = [stored]
        elif np.issubdtype(raw.dtype, np.integer):
            values = [0]
    elif nodata is not None:
        values = [nodata]
    for value in values:
        out[raw == value] = np.nan
    return out


def _cells(da: xr.DataArray, what: str):
    """(transform, crs) of a raster's grid; ValueError if it is not a north-up (unrotated) georeferenced one."""
    from rasterio.crs import CRS

    from ._warp import transform_of

    if da.rio.crs is None:
        raise ValueError(f"{what} has no CRS: pansharpen needs both on georeferenced grids (rio.write_crs)")
    if da.sizes["x"] < 2 or da.sizes["y"] < 2:
        raise ValueError(f"{what} has {da.sizes['y']} x {da.sizes['x']} pixels: too small to know its cell size")
    t = transform_of(da)
    if t.b or t.d:
        raise ValueError(f"{what} is on a rotated grid, which pansharpen does not handle")
    return t, CRS.from_user_input(da.rio.crs)


def _check_grids(ms: xr.DataArray, pan: xr.DataArray, ratio: Optional[float]):
    """The ratio of the cell sizes and the pan's Grid, or a ValueError that says what is wrong."""
    from ._warp import _grid

    tm, crs_m = _cells(ms, "ms")
    tp, crs_p = _cells(pan, "pan")
    if crs_m != crs_p:
        raise ValueError(f"ms is in {crs_m.to_string()} and pan in {crs_p.to_string()}: pansharpen does not "
                         f"reproject, bring both to one CRS first (zeit.load_raster(..., like=pan))")
    if tm.a * tp.a <= 0 or tm.e * tp.e <= 0:
        raise ValueError("ms and pan are oriented differently (the sign of a cell size differs): flip one of them")
    rx, ry = abs(tm.a / tp.a), abs(tm.e / tp.e)
    found = float(np.sqrt(rx * ry))
    if found <= 1 + _RATIO_TOLERANCE:
        raise ValueError(f"the pan's cells ({abs(tp.a):g} x {abs(tp.e):g}) are not finer than the ms cells "
                         f"({abs(tm.a):g} x {abs(tm.e):g}): there is nothing to sharpen (swapped arguments?)")
    if ratio is not None:
        if not np.isfinite(ratio) or ratio < 1:
            raise ValueError(f"ratio must be a number of at least 1, got {ratio!r}")
        found = float(ratio)
    # the footprints must overlap
    mx0, mx1 = sorted((tm.c, tm.c + tm.a * ms.sizes["x"]))
    my0, my1 = sorted((tm.f, tm.f + tm.e * ms.sizes["y"]))
    px0, px1 = sorted((tp.c, tp.c + tp.a * pan.sizes["x"]))
    py0, py1 = sorted((tp.f, tp.f + tp.e * pan.sizes["y"]))
    if min(mx1, px1) <= max(mx0, px0) or min(my1, py1) <= max(my0, py0):
        raise ValueError(f"ms (x {mx0:g} to {mx1:g}, y {my0:g} to {my1:g}) and pan (x {px0:g} to {px1:g}, y "
                         f"{py0:g} to {py1:g}) do not overlap")
    return found, _grid(tp, (pan.sizes["y"], pan.sizes["x"]), crs_p)


def _weights(weights: Any, bands: Sequence) -> Optional[list]:
    if weights is None:
        return None
    if isinstance(weights, dict):
        names = [str(b) for b in bands]
        missing = [str(k) for k in weights if str(k) not in names]
        if missing:
            raise ValueError(f"weights names bands {missing} that ms does not have ({names})")
        given = {str(k): float(v) for k, v in weights.items()}
        return [given.get(n, 0.0) for n in names]
    w = [float(v) for v in np.asarray(weights, dtype=float).ravel()]
    if len(w) != len(bands):
        raise ValueError(f"weights has {len(w)} values and ms {len(bands)} bands")
    return w


def _selection(select: Any, bands: Sequence) -> Optional[list]:
    """Positions of the bands named by ``select``: labels of the band coordinate, or else positions."""
    if select is None:
        return None
    items = [select] if np.isscalar(select) or isinstance(select, str) else list(select)
    labels = list(bands)
    out = []
    for item in items:
        hit = [i for i, b in enumerate(labels) if b == item or str(b) == str(item)]
        if hit:
            out.append(hit[0])
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool) and -len(labels) <= item < len(labels):
            out.append(int(item) % len(labels))
        else:
            raise ValueError(f"intensity_bands: {item!r} is neither a band of ms ({[str(b) for b in labels]}) nor "
                             f"a position in it")
    if not out or len(set(out)) != len(out):
        raise ValueError("intensity_bands must name at least one band, each once")
    return out


def pansharpen(
    ms: xr.DataArray,
    pan: xr.DataArray,
    method: str = "gsa",
    *,
    ratio: Optional[float] = None,
    weights: Union[None, Sequence[float], Dict[Any, float]] = None,
    intensity_bands: Any = None,
    kernel: Optional[int] = None,
    gain: Union[float, str] = "regression",
    nodata: Union[float, str, None] = "auto",
    n_jobs: int = -1,
) -> xr.DataArray:
    """Multispectral bands on the grid of a finer panchromatic band, for any sensor.

    The bands are resampled to the pan's grid by the cubic kernel of the warp of ``load_raster(like=)`` and the
    pan's detail is injected by one of four methods, written from their papers (the equations, the references and
    the choices the papers leave open are in ``rust/zeit-core/src/pansharpen.rs``). Nothing depends on the sensor:
    any number of bands with any names, any resolutions with a ratio above 1 (not necessarily whole: 15 m and 30 m
    give 2, 2.5 m and 10 m give 4, 1.5 m and 6.5 m give 4.33), taken from the grids' georeferencing. The function
    converts no units: the output has those of ``ms`` (digital numbers, radiance or reflectance). Examples:
    Landsat 8/9 (band 8, 15 m, with the 30 m bands), Sentinel-2 (10 m bands with the 20 m or 60 m ones),
    CBERS-4A WPM (2 m pan, 8 m blue, green, red and NIR).

    ``"gsa"`` (default): Gram-Schmidt adaptive (Aiazzi, Baronti and Selva 2007, IEEE TGRS 45(10):3230-3239),
    ``F_k = MS_k + g_k (P' - I)``, ``g_k = cov(MS_k, I) / var(I)``, with the intensity ``I`` the regression (with an
    intercept) of the pan reduced to the bands' resolution on the bands; a band that the pan's spectral range does
    not cover (NIR or SWIR for a visible pan) gets a weight near 0 in ``I`` and a small gain. ``"brovey"``: the
    ratio transform ``F_k = MS_k P / I``. With ``weights`` the intensity is their mean (normalised to sum 1; the pan
    must be in the units of the bands); without them the pan's response is taken as unknown and ``I`` is fitted,
    the regression through the origin with weights not negative, so that ``I`` approximates the pan in its own units:
    unlike equal weights it absorbs a gain between pan and bands and gives the bands outside the pan's range a
    weight near 0 (all bands are still multiplied by ``P / I``: the ratio form cannot do better for them; prefer
    ``"gsa"``). ``"hpf"``: the high-pass filter (Schowengerdt 1980; Gangkofner, Pradhan and Holcomb 2008, PE&RS
    74(9):1107-1118), ``F_k = MS_k + gain sd(MS_k) / sd(LP) (P - LP)``, ``LP`` the box mean of the pan of side
    ``kernel``: it has no intensity, so every band gets detail in proportion to its spread, in or out of the
    pan's range. The default ``kernel`` is the odd side ``2 floor(ratio / 2) + 1`` (the box that matches the bands' cell; 3 for
    ratio 2, 5 for 4).
    ``"pca"``: the principal components with the first replaced by the pan matched to it (needs 2 bands at least;
    the other methods work from 1 band).

    Missing data. A pixel that is NaN (or ``nodata``) in the pan or in any band is NaN in the output, and does
    not enter any statistic (the regression, the means, variances and principal components, which are those of
    each date's whole scene). A missing multispectral cell makes the pan cells it covers NaN, and the pan cells
    outside the bands' footprint are NaN too.

    Grids. ``ms`` and ``pan`` must be in the same CRS, on north-up grids, the pan's cells finer, and overlap. The
    corners need not coincide: the warp's cubic kernel puts the bands on the pan's cells at any sub-pixel offset. What
    the offset costs: the cells of the bands are not whole blocks of the pan's, so the pan reduced to the bands'
    resolution for the regression is taken on a box filter at the pan pixel nearest each cell's centre, off by up to
    half a pixel, which the filter's width hides; the statistics change little. Whether the two images are
    registered to each other (the geolocation) is not checked or corrected: use :func:`zeit.coregister` first.

    Parameters
    ----------
    ms
        ``(time, band, y, x)`` or ``(band, y, x)``, with y and x coordinates and a CRS. Held in memory.
    pan
        ``(y, x)`` (also with a single ``band``) on the finer grid, or with a ``time`` dimension that matches
        ``ms``'s, one pan for each date.
    method
        ``"gsa"`` (default), ``"brovey"``, ``"hpf"`` or ``"pca"``.
    ratio
        The pan's cells per band cell on a side, when it must differ from the ratio of the cell sizes (default:
        their geometric mean over x and y). Used by the reduction of the pan and the default ``kernel``.
    weights
        ``"brovey"``: the weights of the intensity, a sequence in the order of ``ms``'s bands or a dict by band
        label (missing bands weigh 0); normalised to sum 1. Default: fitted (above).
    intensity_bands
        ``"gsa"``, ``"pca"`` and the fitted ``"brovey"``: the bands that enter the estimate of the intensity, by
        label of the band coordinate or else by position (a list, or one); the other bands are still sharpened.
        Default: all.
    kernel
        ``"hpf"``: the side of the box filter in pan pixels (odd).
    gain
        ``"hpf"``: ``"regression"`` (default): each band's least squares slope on the low-passed pan (zeit's choice,
        not the paper's): the same as Gangkofner's modulation for a band that is a linear function of the pan,
        smaller for one the pan does not explain, which the modulation over-sharpens (on the benchmark's scenes,
        synthetic and real, it does not beat the cubic resampling in ERGAS, the regression slope does). A number:
        the weight of Gangkofner's modulation ``sd(MS_k) / sd(LP)`` (1 is the paper's; 0 injects nothing).
    nodata
        Missing values besides NaN, in ``ms`` and ``pan``: ``"auto"`` (the raster's NoData; 0 for integer data
        without one), a number or ``None``.
    n_jobs
        Threads: the processors zeit may use by default (``-1``). The output does not depend on it.

    Returns
    -------
    xarray.DataArray
        float32, the bands of ``ms`` on the pan's grid (coordinates, CRS and the attributes of ``ms``).

    Examples
    --------
    >>> fused = zeit.pansharpen(ms, pan, method="gsa")
    >>> fused = zeit.pansharpen(ms, pan, method="gsa", intensity_bands=["red", "green", "blue"])
    """
    from . import _core
    from ._threads import resolve
    from ._warp import to_grid

    if not isinstance(ms, xr.DataArray) or not isinstance(pan, xr.DataArray):
        raise TypeError("pansharpen needs two xarray.DataArray (the bands and the pan)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    if "band" not in ms.dims or not {"y", "x"} <= set(ms.dims):
        raise ValueError("ms needs band, y and x dimensions (and optionally time)")
    extra = [d for d in ms.dims if d not in ("time", "band", "y", "x")]
    if extra:
        raise ValueError(f"ms has dimensions {extra}; pansharpen takes (time, band, y, x)")
    if not {"y", "x"} <= set(pan.dims):
        raise ValueError("pan needs y and x dimensions")
    if "band" in pan.dims:
        if pan.sizes["band"] != 1:
            raise ValueError(f"pan has {pan.sizes['band']} bands: it must be one panchromatic band")
        pan = pan.isel(band=0, drop=True)
    extra = [d for d in pan.dims if d not in ("time", "y", "x")]
    if extra:
        raise ValueError(f"pan has dimensions {extra}; pansharpen takes (y, x) or (time, y, x)")
    has_time = "time" in ms.dims
    if "time" in pan.dims:
        if not has_time or pan.sizes["time"] != ms.sizes["time"]:
            raise ValueError("pan has a time dimension that ms does not match")
    if isinstance(gain, str):
        if gain != "regression":
            raise ValueError(f"gain must be a number or 'regression', got {gain!r}")
    if kernel is not None:
        if method != "hpf":
            raise ValueError("kernel is for method='hpf'")
        if int(kernel) < 1 or int(kernel) % 2 == 0:
            raise ValueError(f"kernel must be odd, got {kernel}")
    if method == "pca" and ms.sizes["band"] < 2:
        raise ValueError("method='pca' needs at least 2 bands")

    ratio, grid = _check_grids(ms, pan, ratio)
    w = _weights(weights, ms["band"].values)
    if w is not None and method != "brovey":
        raise ValueError("weights is for method='brovey'")
    sel = _selection(intensity_bands, ms["band"].values)
    if sel is not None and method == "hpf":
        raise ValueError("intensity_bands is not used by method='hpf' (it has no intensity)")
    if sel is not None and w is not None:
        raise ValueError("give weights or intensity_bands, not both (weights fix the intensity)")
    if method == "pca" and sel is not None and len(sel) < 2:
        raise ValueError("method='pca' needs at least 2 bands in intensity_bands")
    threads = resolve(n_jobs)

    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = ms.transpose(*order)
    coarse = _float32_with_nan(work, nodata)
    if not has_time:
        coarse = coarse[None]
    pan_t = pan.transpose(*(("time",) if "time" in pan.dims else ()), "y", "x")
    pan_all = _float32_with_nan(pan_t, nodata)
    nt, nb = coarse.shape[:2]
    ph, pw = grid.shape
    out = np.empty((nt, nb, ph, pw), dtype=np.float32)
    coords_t = {"x": work["x"].values, "y": work["y"].values}
    t_ms, crs_ms = _cells(ms, "ms")[0], ms.rio.crs
    for k in range(nt):
        cube = xr.DataArray(coarse[k], dims=("band", "y", "x"), coords=coords_t)
        cube = cube.rio.write_crs(crs_ms).rio.write_transform(t_ms)
        fine = np.ascontiguousarray(to_grid(cube, grid, "cubic", n_threads=threads).values, dtype=np.float32)
        # a missing coarse cell is missing for all the pan's cells it covers, whatever the cubic made of it; so are
        # the pan's cells outside the bands (1 inside where there is data, 0 where missing, the fill outside)
        gone = (~np.isfinite(coarse[k])).any(axis=0)
        flag = xr.DataArray(np.where(gone, 0.0, 1.0).astype(np.float32), dims=("y", "x"), coords=coords_t)
        flag = flag.rio.write_crs(crs_ms).rio.write_transform(t_ms)
        inside = to_grid(flag, grid, "nearest", n_threads=threads).values > 0.5
        fine[:, ~inside] = np.nan
        p = pan_all[k] if "time" in pan.dims else pan_all
        res, _, _ = _core.pansharpen.pansharpen_batch(fine, np.ascontiguousarray(p), method, ratio, w, sel,
                                                      0 if kernel is None else int(kernel),
                                                      1.0 if isinstance(gain, str) else float(gain),
                                                      isinstance(gain, str), threads)
        out[k] = res
        del fine
    if not has_time:
        out = out[0]

    coords = {n: c for n, c in work.coords.items() if "y" not in c.dims and "x" not in c.dims and n != "spatial_ref"}
    coords["y"], coords["x"] = grid.y, grid.x
    result = xr.DataArray(out, dims=order, coords=coords, name=ms.name).transpose(*ms.dims)
    result.attrs = {k: v for k, v in ms.attrs.items() if k != "_FillValue"}
    detail = {"brovey": "weighted Brovey" if w is not None else "Brovey with fitted weights",
              "gsa": "Gram-Schmidt adaptive (Aiazzi, Baronti and Selva 2007)",
              "hpf": "high-pass filter (Schowengerdt 1980; Gangkofner et al. 2008)",
              "pca": "principal component substitution"}[method]
    result.attrs["pansharpen"] = f"{method}: {detail}, ratio {ratio:.6g}, bands resampled by the cubic kernel"
    result = result.rio.write_crs(grid.crs).rio.write_transform(grid.transform)
    result = result.rio.write_nodata(np.nan, encoded=False)
    return result
