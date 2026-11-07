"""Topographic correction of reflectance, ``zeit.topographic_correction``: the C-correction and SCS+C.

Written from the published formulas (Horn 1981 for slope and aspect; Teillet, Guindon & Goodenough 1982 for the
C-correction; Soenen, Peddle & Coburn 2005 for SCS+C), not from any implementation; computed by the compiled core
(``zeit._core.topocorr``).
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

#: The angle layers a cube may carry as bands; they are never corrected.
_ANGLE_BANDS = ("SZA", "SAA", "VZA", "VAA")

#: Metadata names (coordinate or attribute, any case) of the sun's position when the cube has no angle bands.
_ZENITH_KEYS = ("sun_zenith", "solar_zenith", "sza", "s2:mean_solar_zenith")
_ELEVATION_KEYS = ("sun_elevation", "solar_elevation", "view:sun_elevation")
_AZIMUTH_KEYS = ("sun_azimuth", "solar_azimuth", "saa", "view:sun_azimuth", "s2:mean_solar_azimuth")

_METHODS = {"c": "C-correction (Teillet et al. 1982)", "scs+c": "SCS+C (Soenen et al. 2005)"}


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    step = float(np.median(np.abs(np.diff(a)))) if a.size > 1 else 1.0
    return bool(np.allclose(a, b, rtol=0, atol=1e-6 * (step or 1.0)))


def _check_grid(layer: xr.DataArray, cube: xr.DataArray, what: str) -> None:
    """`layer` must be on exactly the grid of `cube`: same size and, if both have them, same coordinates."""
    for d in ("y", "x"):
        if d not in layer.dims:
            raise ValueError(f"{what} has no {d!r} dimension: it must be on the cube's grid")
        if layer.sizes[d] != cube.sizes[d]:
            raise ValueError(
                f"{what} has {layer.sizes[d]} {d} values, the cube {cube.sizes[d]}: it must be on exactly the cube's "
                "grid (zeit does not resample; bring it over with `rio.reproject_match(cube)`)")
        if d in layer.coords and d in cube.coords and not _same(np.asarray(layer[d].values, dtype=float),
                                                                np.asarray(cube[d].values, dtype=float)):
            raise ValueError(
                f"{what}: the {d} coordinates differ from the cube's: it must be on exactly the cube's grid (zeit "
                "does not resample; bring it over with `rio.reproject_match(cube)`)")


def _pixel_size(cube: xr.DataArray) -> Tuple[float, float]:
    """Metres per column (east) and per row (north, negative for a north-up raster) of the cube's grid."""
    crs = cube.rio.crs
    if crs is None:
        raise ValueError("the cube has no CRS: the slope needs the pixel size in metres, so give it a projected CRS "
                         "(`cube.rio.write_crs(...)`)")
    if crs.is_geographic:
        raise ValueError(
            "the cube is in a geographic CRS (degrees): the slope needs a projected CRS in metres. Reproject the "
            "cube and the DEM to the UTM zone of the scene first (`cube.rio.reproject(...)`)")
    factor = 1.0
    try:
        factor = float(crs.linear_units_factor[1])
    except Exception:  # an older rasterio: assume metres
        pass
    steps = []
    for d in ("x", "y"):
        v = np.asarray(cube[d].values, dtype=float)
        if v.size < 2:
            raise ValueError(f"the cube needs at least 2 {d} values to have a pixel size")
        step = (v[-1] - v[0]) / (v.size - 1)
        if step == 0 or not np.allclose(np.diff(v), step, rtol=1e-6, atol=0):
            raise ValueError(f"the {d} coordinates are not regularly spaced: the grid must be regular")
        steps.append(step * factor)
    return steps[0], steps[1]


def _dem_array(dem: Any, cube: xr.DataArray) -> np.ndarray:
    """The DEM as float64 ``(y, x)``, checked against the cube's grid, its NoData as NaN."""
    if not isinstance(dem, xr.DataArray):
        from ._load import load_raster

        dem = load_raster(dem)
    if "band" in dem.dims and dem.sizes["band"] == 1:
        dem = dem.isel(band=0, drop=True)
    if "time" in dem.dims and dem.sizes["time"] == 1:
        dem = dem.isel(time=0, drop=True)
    if set(dem.dims) != {"y", "x"}:
        raise ValueError(f"dem must be a DataArray of (y, x), got dimensions {list(dem.dims)}")
    _check_grid(dem, cube, "the DEM")
    out = np.ascontiguousarray(dem.transpose("y", "x").values, dtype=np.float64)
    nodata = dem.rio.nodata
    if nodata is not None and not np.isnan(nodata):
        out[out == nodata] = np.nan
    return out


def slope_aspect(dem: Any, *, n_jobs: int = -1) -> xr.Dataset:
    """Slope and aspect, in degrees, of a DEM by Horn's (1981) 3x3 kernel.

    The DEM is a DataArray ``(y, x)`` in a projected CRS in metres, with its elevations in metres. The slope is
    ``atan(|grad z|)`` (0 flat, 90 vertical); the aspect is the direction the surface faces (downslope), clockwise
    from north, in ``[0, 360)``, and 0 where it is flat. The cells on the edge replicate the nearest cell for
    their missing neighbours and take the difference over the one cell it spans (one-sided); a cell with a non-finite value among its eight neighbours is NaN.
    """
    from . import _core
    from ._threads import resolve

    if not isinstance(dem, xr.DataArray):
        from ._load import load_raster

        dem = load_raster(dem)
    if "band" in dem.dims and dem.sizes["band"] == 1:
        dem = dem.isel(band=0, drop=True)
    dx, dy = _pixel_size(dem)
    z = _dem_array(dem, dem)
    s, a = _core.topocorr.slope_aspect(z, dx, dy, resolve(n_jobs))
    coords = {k: v for k, v in dem.coords.items() if set(v.dims) <= {"y", "x"}}
    out = xr.Dataset({"slope": (("y", "x"), np.degrees(s)), "aspect": (("y", "x"), np.degrees(a))}, coords=coords)
    out["slope"].attrs = {"units": "degrees"}
    out["aspect"].attrs = {"units": "degrees", "long_name": "clockwise from north, direction the surface faces"}
    if dem.rio.crs is not None:
        out = out.rio.write_crs(dem.rio.crs)
    return out


def _meta(cube: xr.DataArray, keys: Sequence[str]) -> Optional[np.ndarray]:
    """The first of `keys` (any case) among the cube's coordinates, then attributes, as a float array."""
    wanted = [k.lower() for k in keys]
    for source in (cube.coords, cube.attrs):
        for k in wanted:
            for name in source:
                if str(name).lower() == k:
                    value = source[name]
                    value = value.values if hasattr(value, "values") else value
                    try:
                        return np.atleast_1d(np.asarray(value, dtype=float))
                    except (TypeError, ValueError):
                        continue
    return None


def _angle_band(cube: xr.DataArray, name: str) -> Optional[xr.DataArray]:
    if "band" not in cube.dims:
        return None
    for b in cube.band.values:
        if str(b).upper() == name:
            return cube.sel(band=b, drop=True)
    return None


def _layer(value: Any, name: str) -> Optional[xr.DataArray]:
    """`name` (``SZA``/``SAA``) out of a Dataset or a DataArray with an ``angle``/``band`` dimension."""
    if isinstance(value, xr.Dataset):
        for k in value.data_vars:
            if str(k).upper() == name:
                return value[k]
        raise ValueError(f"the Dataset has no {name} variable (it has {list(value.data_vars)})")
    if isinstance(value, xr.DataArray):
        for dim in ("angle", "band"):
            if dim in value.dims:
                for v in value[dim].values:
                    if str(v).upper() == name:
                        return value.sel({dim: v}, drop=True)
                raise ValueError(f"the DataArray has no {name} along {dim!r} (it has {list(value[dim].values)})")
    return None


def _sun(which: str, value: Any, other: Any, cube: xr.DataArray, n_dates: int) -> Tuple[np.ndarray, str]:
    """One sun angle (``zenith`` or ``azimuth``) as float32 ``(time, 1)`` or ``(time, y * x)``, in degrees, and a
    description of where it came from. `other` is the other parameter's value (a Dataset or a DataArray with both
    angles may be given as either)."""
    name = "SZA" if which == "zenith" else "SAA"
    param = f"sun_{which}="
    has_time = "time" in cube.dims
    layer = None
    how = ""
    if isinstance(value, str):
        if value != "scene":
            raise ValueError(f"{param} must be a number (degrees), a per-date sequence, a DataArray or 'scene', "
                             f"got {value!r}")
        layer = _layer(other, name) if other is not None else None
        if layer is not None:
            how = "the angles given"
        else:
            layer = _angle_band(cube, name)
            if layer is not None:
                how = f"the cube's {name} band"
            else:
                if which == "zenith":
                    found = _meta(cube, _ZENITH_KEYS)
                    if found is None:
                        elev = _meta(cube, _ELEVATION_KEYS)
                        found = None if elev is None else 90.0 - elev
                else:
                    found = _meta(cube, _AZIMUTH_KEYS)
                if found is None:
                    raise ValueError(
                        f"{param}'scene' found no {name} band or sun {which} metadata (a coordinate or attribute "
                        f"like {_ZENITH_KEYS[0] if which == 'zenith' else _AZIMUTH_KEYS[0]!r}) in the cube: pass "
                        f"{param} a number in degrees, one per date, or a per-pixel DataArray "
                        "(`zeit.sentinel2_angles` gives SZA and SAA for Sentinel-2)")
                value, how = found, "the cube's metadata"
    elif isinstance(value, (xr.Dataset, xr.DataArray)):
        layer = _layer(value, name)
        how = "the angles given"
        if layer is None:
            layer = value
            how = "a per-pixel DataArray"
    if layer is not None:
        if not isinstance(layer, xr.DataArray):
            raise ValueError(f"{param} must be a DataArray")
        extra = [d for d in layer.dims if d not in ("time", "y", "x")]
        if extra:
            raise ValueError(f"{param}: the angles have dimensions {extra} that the cube does not")
        _check_grid(layer, cube, f"the sun {which}")
        if "time" in layer.dims:
            if not has_time:
                raise ValueError(f"the sun {which} has a 'time' dimension, the cube has none")
            if layer.sizes["time"] != n_dates or not np.array_equal(layer.time.values, cube.time.values):
                raise ValueError(f"the sun {which}'s dates differ from the cube's (no reindexing): they must have "
                                 "the same times, in the same order")
            layer = layer.transpose("time", "y", "x")
            arr = np.asarray(layer.values, dtype=np.float32).reshape(n_dates, -1)
        else:
            arr = np.asarray(layer.transpose("y", "x").values, dtype=np.float32).reshape(1, -1)
            arr = np.broadcast_to(arr, (n_dates, arr.shape[1])).copy()
        return np.ascontiguousarray(arr), how
    arr = np.asarray(value, dtype=float)
    if arr.ndim > 1 or arr.size not in (1, n_dates):
        raise ValueError(f"{param} has {arr.size} values for {n_dates} dates")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{param} must be finite numbers, got {value!r}")
    if which == "zenith" and not np.all((arr >= 0) & (arr < 90)):
        raise ValueError(f"{param} must be a solar zenith of 0 to 90 degrees, got {value!r}")
    arr = np.broadcast_to(arr.reshape(-1), (n_dates,)).astype(np.float32).reshape(n_dates, 1)
    return np.ascontiguousarray(arr), how or (f"{value} degrees" if np.ndim(value) == 0 else f"one {which} per date")


def _sample(sample: Any, cube: xr.DataArray, n_dates: int) -> Optional[np.ndarray]:
    """The sample mask as uint8 ``(1 or time, y * x)``."""
    if sample is None:
        return None
    ny, nx = cube.sizes["y"], cube.sizes["x"]
    if isinstance(sample, xr.DataArray):
        _check_grid(sample, cube, "sample")
        extra = [d for d in sample.dims if d not in ("time", "y", "x")]
        if extra:
            raise ValueError(f"sample has dimensions {extra} that the cube does not")
        if "time" in sample.dims:
            if "time" not in cube.dims or sample.sizes["time"] != n_dates:
                raise ValueError("sample's time dimension does not match the cube's")
            m = sample.transpose("time", "y", "x").values
        else:
            m = sample.transpose("y", "x").values[None]
    else:
        m = np.asarray(sample)
        if m.shape not in ((ny, nx), (n_dates, ny, nx)) or (m.ndim == 3 and "time" not in cube.dims):
            raise ValueError(f"sample must be (y, x) or (time, y, x) = {(ny, nx)} / {(n_dates, ny, nx)}, "
                             f"got {m.shape}")
        m = m[None] if m.ndim == 2 else m
    if m.dtype != bool:
        if not np.issubdtype(m.dtype, np.number) or np.isnan(m.astype(float)).any():
            raise ValueError("sample must be a boolean mask")
        m = m != 0
    return np.ascontiguousarray(m.reshape(m.shape[0], -1).astype(np.uint8))


def topographic_correction(
    cube: Any,
    dem: Any,
    *,
    sun_zenith: Any = "scene",
    sun_azimuth: Any = "scene",
    method: str = "c",
    sample: Any = None,
    min_slope: float = 5.0,
    min_sample: int = 100,
    bands: Optional[Sequence[Any]] = None,
    nodata: Union[float, str, None] = "auto",
    n_jobs: int = -1,
) -> xr.DataArray:
    """Topographic correction of surface reflectance: the C-correction (Teillet, Guindon & Goodenough 1982) or
    SCS+C (Soenen, Peddle & Coburn 2005), which take out the modulation of the reflectance by the relief's
    illumination.

    Written from the published formulas, not from any implementation. The slope and aspect of the DEM come from
    Horn's (1981) 3x3 kernel; the cosine of the illumination angle is ``cos i = cos sz cos s + sin sz sin s
    cos(saz - a)`` (``sz``, ``saz`` the sun's zenith and azimuth, ``s``, ``a`` the slope and aspect). For each
    band of each date the reflectance is regressed on ``cos i`` over the sample pixels, ``rho = b + m cos i``,
    ``c = b / m``, and then ``rho_H = rho (cos sz + c) / (cos i + c)`` (``method="c"``) or
    ``rho_H = rho (cos s cos sz + c) / (cos i + c)`` (``method="scs+c"``). The correction is multiplicative, so
    it does not depend on the units of the reflectance (scaled integers, 0-1, percent). It is only an empirical
    flattening: it assumes a Lambertian surface and the same land cover on all slopes, does not model the
    diffuse or the adjacent-terrain light, and needs a DEM of about the resolution of the image. FORCE's own
    C-factor (in its atmospheric correction, tied to its radiation terms) is a different, later step.

    Parameters
    ----------
    cube
        Reflectance ``(time, band, y, x)``, ``(band, y, x)``, ``(time, y, x)`` or ``(y, x)``, in a projected CRS
        in metres (or another linear unit; a geographic CRS is an error: reproject first). In memory, or dask
        (read into memory here: the regression needs a whole date), or anything ``zeit.load_raster`` reads.
        Missing values (``nodata``) and NaN are kept NaN and stay out of the regression.
    dem
        Elevation in metres, a DataArray ``(y, x)`` (or a path) on exactly the cube's grid: same size and
        coordinates. It is never resampled; bring it over with ``dem.rio.reproject_match(cube)``. Its NoData
        and NaN make the slope of the cells next to them NaN, and those pixels are left uncorrected.
    sun_zenith, sun_azimuth
        The sun's zenith (0-90) and azimuth (clockwise from north), degrees: a number, a sequence with one per
        date, a per-pixel DataArray ``(y, x)`` or ``(time, y, x)`` on the cube's grid (and dates), or
        ``"scene"`` (default): the cube's own bands named ``SZA`` and ``SAA`` (per pixel; those bands are not
        corrected), else the metadata of the cube (a coordinate or attribute ``sun_zenith`` or ``sun_elevation``
        (zenith = 90 - elevation) and ``sun_azimuth``, as STAC's ``view:sun_elevation``/``view:sun_azimuth``,
        per date or one). A Dataset with ``SZA`` and ``SAA`` (``zeit.sentinel2_angles``'s output) or a DataArray
        with an ``angle`` dimension may be given as either one and then serves both when the other is
        ``"scene"``. A pixel whose sun zenith is outside 0-90 or missing is left uncorrected.
    method
        ``"c"`` (default) or ``"scs+c"``. The C-correction does not use the slope in its numerator; SCS+C
        keeps the sun-canopy-sensor geometry, which suits forests on steep slopes (``cos s`` in the numerator).
    sample
        A boolean mask ``(y, x)`` or ``(time, y, x)`` of the pixels the regression uses (for example the
        vegetated and clear ones), used as given. Default: every pixel with a finite reflectance and
        ``cos i`` and a slope of at least ``min_slope``.
    min_slope
        Slope, in degrees, below which a pixel does not enter the default sample (5): where the terrain is flat
        ``cos i`` is the same for all pixels (``cos sz``), carries no information about the relief, and in a
        mostly flat scene would pull the line through one cluster. Pixels below it are still corrected (a flat
        pixel's factor is 1).
    min_sample
        Fewest sample pixels for a regression (100). A band of a date with fewer, with no variance in
        ``cos i``, or with ``m <= 0`` (reflectance that does not increase with illumination: the topography is
        not what drives it, or the sample is wrong) is copied unchanged, with a warning, and its ``topo_c`` is
        NaN.
    bands
        The bands to correct (labels of ``band``); default all except the angle bands ``SZA``, ``SAA``,
        ``VZA``, ``VAA``. The others are copied.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData, 0 for integers without one), a number or
        ``None``.
    n_jobs
        Threads: the processors zeit may use by default (``-1``).

    Returns
    -------
    xarray.DataArray
        The cube as float32 with the same dimensions, coordinates and georeferencing (missing values NaN), the
        corrected bands replaced, and a coordinate ``topo_c`` (``(time, band)``, or ``(band,)``): the ``c`` of
        every band and date, NaN for the bands copied. ``attrs["topocorr"]`` says the method and how the sun
        was given. Pixels that cannot be corrected keep their values: ``cos i + c`` below 0.01 (a pixel
        in deep shadow of a poorly fitted ``c``), a non-positive numerator, a missing sun angle, a DEM cell next
        to a NaN.

    Notes
    -----
    The cells on the DEM's edge replicate the nearest cell for the neighbours they lack and take the difference
    over the one cell it spans (a one-sided Horn gradient), so they have a slope and are corrected; a flat cell
    has aspect 0.

    Examples
    --------
    >>> corrected = zeit.topographic_correction(cube, dem, sun_zenith=38.5, sun_azimuth=142.0)
    >>> corrected = zeit.topographic_correction(cube, dem, sun_zenith=angles, method="scs+c")   # sentinel2_angles
    """
    from . import _core
    from ._load import load_raster
    from ._series_api import _missing_as_nan
    from ._threads import resolve

    if method not in _METHODS:
        raise ValueError(f"method must be 'c' or 'scs+c', got {method!r}")
    if not (np.isfinite(min_slope) and 0 <= min_slope < 90):
        raise ValueError(f"min_slope must be a slope of 0 to 90 degrees, got {min_slope!r}")
    if int(min_sample) != min_sample or min_sample < 3:
        raise ValueError(f"min_sample must be an integer of at least 3, got {min_sample!r}")
    cube = cube if isinstance(cube, xr.DataArray) else load_raster(cube)
    unknown = [d for d in cube.dims if d not in ("time", "band", "y", "x")]
    if unknown or not {"y", "x"} <= set(cube.dims):
        raise ValueError("the cube must have dimensions (time, band, y, x), (band, y, x), (time, y, x) or (y, x), "
                         f"got {list(cube.dims)}")
    single = "band" not in cube.dims
    if single:
        cube = cube.expand_dims(band=[0]).transpose(*(("time",) if "time" in cube.dims else ()), "band", "y", "x")
    has_time = "time" in cube.dims
    n_dates = cube.sizes["time"] if has_time else 1
    ny, nx = cube.sizes["y"], cube.sizes["x"]

    dx, dy = _pixel_size(cube)
    z = _dem_array(dem, cube)
    slope, aspect = _core.topocorr.slope_aspect(z, dx, dy, resolve(n_jobs))

    zen, zen_how = _sun("zenith", sun_zenith, sun_azimuth, cube, n_dates)
    azi, azi_how = _sun("azimuth", sun_azimuth, sun_zenith, cube, n_dates)
    if zen.shape[1] != azi.shape[1]:
        # one of them per pixel, the other per date: bring both to per pixel
        zen = np.ascontiguousarray(np.broadcast_to(zen, (n_dates, ny * nx)))
        azi = np.ascontiguousarray(np.broadcast_to(azi, (n_dates, ny * nx)))
    mask = _sample(sample, cube, n_dates)

    labels = list(cube.band.values)
    if bands is None:
        chosen = [i for i, b in enumerate(labels) if str(b).upper() not in _ANGLE_BANDS]
    else:
        chosen = []
        for b in (bands if isinstance(bands, (list, tuple, np.ndarray)) else [bands]):
            hits = [i for i, v in enumerate(labels) if v == b or str(v) == str(b)]
            if not hits:
                raise ValueError(f"bands=: {b!r} is not a band of the cube {[str(v) for v in labels]}")
            chosen.append(hits[0])
    if not chosen:
        raise ValueError("there is no band to correct (the cube has only angle bands?)")

    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = _missing_as_nan(cube, nodata).transpose(*order)
    arr = np.asarray(work.values, dtype=np.float32)
    if not has_time:
        arr = arr[None]
    everything = chosen == list(range(len(cube.band)))
    sub = np.ascontiguousarray(arr if everything else arr[:, chosen])
    done, fits = _core.topocorr.correct(sub, slope, aspect, zen, azi, mask, method, float(np.radians(min_slope)),
                                        int(min_sample), resolve(n_jobs))
    if everything:
        arr = done
    else:
        arr = arr.copy()
        arr[:, chosen] = done
    c = np.full((n_dates, len(labels)), np.nan)
    fits = fits.reshape(n_dates, len(chosen), 5)
    c[:, chosen] = np.where(np.isfinite(fits[:, :, 0]), fits[:, :, 0], np.nan)

    failed = [(k, chosen[j]) for k in range(n_dates) for j in range(len(chosen)) if not np.isfinite(fits[k, j, 0])]
    if failed:
        def why(k: int, j: int) -> str:
            n, m = fits[k, j, 3], fits[k, j, 1]
            if n < min_sample:
                return f"{int(n)} sample pixels"
            return "no variance in cos i" if not np.isfinite(m) else f"slope of the regression {m:.3g} <= 0"
        names = [f"{cube.time.values[k] if has_time else 'scene'}/{labels[i]}: {why(k, chosen.index(i))}"
                 for k, i in failed[:4]]
        warnings.warn(f"topographic_correction: {len(failed)} band(s) of date(s) could not be fitted and are "
                      f"copied unchanged ({'; '.join(names)}{'; ...' if len(failed) > 4 else ''})", stacklevel=2)

    out = arr if has_time else arr[0]
    result = cube.copy(data=out.astype(np.float32, copy=False))
    result.attrs = dict(cube.attrs)
    result.attrs["topocorr"] = (f"{_METHODS[method]}; sun zenith from {zen_how}, azimuth from {azi_how}; "
                                f"slope and aspect by Horn (1981); default sample: slope >= {min_slope} degrees"
                                if mask is None else
                                f"{_METHODS[method]}; sun zenith from {zen_how}, azimuth from {azi_how}; "
                                "slope and aspect by Horn (1981); sample given")
    result = result.assign_coords(topo_c=((("time", "band") if has_time else ("band",)),
                                          c.astype(np.float32) if has_time else c[0].astype(np.float32)))
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    if cube.rio.nodata is not None:
        result = result.rio.write_nodata(np.nan, encoded=False)
    result = result.transpose(*cube.dims)
    if single:
        result = result.isel(band=0, drop=True)
    return result
