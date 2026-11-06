"""Nadir BRDF-adjusted reflectance, ``zeit.nbar``: Landsat and Sentinel-2 reflectance as seen from nadir.

The c-factor method of Roy et al. (2016, RSE 176:255-271; 2017, RSE 199:25-38): a pixel's reflectance is
multiplied by the ratio of the Ross-Thick / Li-Sparse-Reciprocal model (three fixed parameters per band, those of
MODIS) at nadir and the normalized solar zenith to the model at the pixel's own sun and view angles. Written from
the papers, not from any implementation; computed by the compiled core (``zeit._core.brdf``).
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

#: Roles that have BRDF parameters, in the order of the core. ``nir`` is Sentinel-2's broad NIR (B08), ``nir08``
#: the narrow one (B8A, OLI's band 5, TM/ETM+'s band 4); both have MODIS's NIR parameters.
ROLES = ("coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir1", "swir2")

#: Band names of each role (Earth Search, Planetary Computer).
_NAMES = {
    "coastal": ("coastal", "B01"),
    "blue": ("blue", "B02"),
    "green": ("green", "B03"),
    "red": ("red", "B04"),
    "rededge1": ("rededge1", "B05"),
    "rededge2": ("rededge2", "B06"),
    "rededge3": ("rededge3", "B07"),
    "nir": ("nir", "B08"),
    "nir08": ("nir08", "B8A"),
    "swir1": ("swir16", "B11"),
    "swir2": ("swir22", "B12"),
}
#: Landsat Collection 2 names, which mean different bands on OLI and on TM/ETM+.
_OLI_NAMES = {"coastal": "SR_B1", "blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir08": "SR_B5",
              "swir1": "SR_B6", "swir2": "SR_B7"}
#: The angle layers, in the order of the core.
ANGLES = ("SZA", "SAA", "VZA", "VAA")


def _parameters(params: Optional[Dict[str, Sequence[float]]]) -> Dict[str, Tuple[float, float, float]]:
    """``(f_iso, f_geo, f_vol)`` of every role: Roy et al. (2016) Table 5 (the core's), or those of ``params``."""
    from . import _core

    table = {r: _core.brdf.role(r) for r in ROLES}
    for role, values in (params or {}).items():
        if role not in ROLES:
            raise ValueError(f"params=: unknown role {role!r}; roles are {list(ROLES)}")
        values = tuple(float(v) for v in values)
        if len(values) != 3 or not all(np.isfinite(values)):
            raise ValueError(f"params=: {role!r} must be (f_iso, f_geo, f_vol), three finite numbers")
        table[role] = values
    return table


def _roles(cube: xr.DataArray, bands: Optional[Dict[str, Any]], sensors: Optional[List[str]],
           skip: Sequence[str]) -> Dict[str, Any]:
    """The cube's band of each role it has."""
    available = [str(b) for b in cube.band.values]
    lower = [b.lower() for b in available]
    out: Dict[str, Any] = {}
    for role, given in (bands or {}).items():
        if role not in ROLES:
            raise ValueError(f"bands=: unknown role {role!r}; roles are {list(ROLES)}")
        if str(given) not in available:
            raise ValueError(f"bands=: {given!r} is not a band of the cube {available}")
        out[role] = cube.band.values[available.index(str(given))]
    taken = {str(v) for v in out.values()} | {str(s) for s in skip}
    old = sensors is not None and any(s in ("ETM", "TM") for s in sensors)
    for role in ROLES:
        if role in out:
            continue
        names = list(_NAMES[role])
        if role in _OLI_NAMES and _OLI_NAMES[role].lower() in lower and _OLI_NAMES[role] not in taken:
            if old:
                raise ValueError(f"the cube has Landsat Collection 2 band names ({_OLI_NAMES[role]}), which are not "
                                 "the same bands on Landsat 4-5 and 7 as on 8-9: pass bands= (role -> band)")
            names.append(_OLI_NAMES[role])
        for name in names:
            if name.lower() in lower and available[lower.index(name.lower())] not in taken:
                out[role] = cube.band.values[lower.index(name.lower())]
                taken.add(available[lower.index(name.lower())])
                break
    return out


def _angle_layers(angles: Any, cube: xr.DataArray) -> Tuple[xr.DataArray, List[Any]]:
    """The four angles as a ``(angle, [time,] y, x)`` DataArray, and the bands of the cube they are (if any)."""
    def pick(names: Dict[str, Any], what: str) -> Dict[str, Any]:
        found = {}
        for a in ANGLES:
            hits = [k for k in names if str(k).upper() == a]
            if hits:
                found[a] = names[hits[0]]
        missing = [a for a in ANGLES if a not in found]
        if missing:
            raise ValueError(f"the angles of {what} lack {missing}: SZA, SAA, VZA and VAA, in degrees, are needed")
        return found

    own: List[Any] = []
    if angles is None:
        names = {str(b): b for b in cube.band.values}
        try:
            found = pick(names, "the cube's bands")
        except ValueError:
            raise ValueError(
                "nbar needs each pixel's sun and view angles: pass angles= (an xarray Dataset with SZA, SAA, VZA and "
                "VAA in degrees on the cube's grid) or have those four bands in the cube. HLS cubes carry them "
                "(request bands SZA, SAA, VZA, VAA) but are already NBAR; for Landsat Collection 2 they are the "
                "Level-1 angle bands (USGS credentials); for Sentinel-2 they come from the tile's MTD_TL.xml (the "
                "coming zeit.sentinel2_angles)") from None
        own = [found[a] for a in ANGLES]
        parts = [cube.sel(band=found[a]).drop_vars("band") for a in ANGLES]
        for p in parts:
            p.attrs = {}
        layers = xr.concat(parts, dim="angle").assign_coords(angle=list(ANGLES))
        return layers, own
    if isinstance(angles, xr.Dataset):
        found = pick({str(k): k for k in angles.data_vars}, "the Dataset")
        parts = [angles[found[a]] for a in ANGLES]
    elif isinstance(angles, xr.DataArray):
        dim = "angle" if "angle" in angles.dims else "band" if "band" in angles.dims else None
        if dim is None:
            raise ValueError("angles= as a DataArray needs an 'angle' (or 'band') dimension naming SZA, SAA, VZA, VAA")
        found = pick({str(v): v for v in angles[dim].values}, "the DataArray")
        parts = [angles.sel({dim: found[a]}).drop_vars(dim) for a in ANGLES]
    else:
        raise ValueError("angles= must be an xarray Dataset (variables SZA, SAA, VZA, VAA) or a DataArray with an "
                         "'angle' dimension, in degrees, on the cube's grid")
    parts = [p.drop_vars([c for c in p.coords if c not in p.dims]) for p in parts]
    for p in parts:
        p.attrs = {}
    layers = xr.concat(parts, dim="angle").assign_coords(angle=list(ANGLES))
    return layers, own


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        step = float(np.median(np.abs(np.diff(a)))) if a.size > 1 else 1.0
        return bool(np.allclose(a, b, rtol=0, atol=1e-6 * (step or 1.0)))
    return bool(np.array_equal(a, b))


def _align(layers: xr.DataArray, cube: xr.DataArray) -> xr.DataArray:
    """The angles ``(time, angle, y, x)`` (or ``(angle, y, x)``) checked against the cube, never resampled."""
    for d in ("y", "x"):
        if d not in layers.dims:
            raise ValueError(f"the angles have no {d!r} dimension: they must be on the cube's grid")
        if layers.sizes[d] != cube.sizes[d]:
            raise ValueError(f"the angles have {layers.sizes[d]} {d} values, the cube {cube.sizes[d]}: they must be on "
                             "the cube's grid (nbar does not resample)")
        if d in layers.coords and d in cube.coords and not _same(layers[d].values, cube[d].values):
            raise ValueError(f"the angles' {d} coordinates differ from the cube's: they must be on exactly the "
                             "cube's grid (nbar does not resample)")
    if "time" in cube.dims:
        if "time" not in layers.dims:
            raise ValueError("the angles have no 'time' dimension: they differ for each date, so nbar needs one "
                             "set of angles per date of the cube")
        if layers.sizes["time"] != cube.sizes["time"] or not np.array_equal(layers.time.values, cube.time.values):
            raise ValueError("the angles' dates differ from the cube's (nbar does not reindex): they must have the "
                             "same times, in the same order")
    elif "time" in layers.dims:
        raise ValueError("the angles have a 'time' dimension, the cube has none")
    extra = [d for d in layers.dims if d not in ("time", "angle", "y", "x")]
    if extra:
        raise ValueError(f"the angles have dimensions {extra} that the cube does not")
    order = (("time",) if "time" in cube.dims else ()) + ("angle", "y", "x")
    return layers.transpose(*order)


def _scene_sza(ang: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per date: the mean solar zenith over the pixels whose four angles are valid, the number of those pixels and
    of the pixels with a finite solar zenith. `ang` is ``(time, 4, y, x)``, numpy or dask."""
    chunked = hasattr(ang, "chunks") and not isinstance(ang, np.ndarray)
    if chunked:
        import dask.array as xp
    else:
        xp = np
    a = ang.astype(np.float64)
    sza, vza = a[:, 0], a[:, 2]
    finite = xp.isfinite(a).all(axis=1)
    valid = finite & (sza >= 0) & (sza < 90) & (vza >= 0) & (vza < 90)
    total = xp.where(valid, sza, 0.0).sum(axis=(1, 2))
    count = valid.sum(axis=(1, 2))
    seen = xp.isfinite(sza).sum(axis=(1, 2))
    if chunked:
        import dask

        total, count, seen = dask.compute(total, count, seen)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, np.asarray(count), np.asarray(seen)


def _norm_sza(sza: Any, cube: xr.DataArray, ang: Any, n_dates: int) -> Tuple[np.ndarray, str]:
    """The normalized solar zenith of every date and a description of how it was chosen."""
    if isinstance(sza, str):
        if sza == "scene":
            mean, count, seen = _scene_sza(ang)
            lost = np.flatnonzero((count == 0) & (seen > 0))
            if lost.size:
                warnings.warn(f"{lost.size} date(s) have solar zenith values but none inside the model's domain "
                              "(a zenith of 0 to 90 degrees, with every angle valid): are the angles in degrees? "
                              "(HLS stores them as integers in units of 0.01 degree.) Those dates are NaN.",
                              stacklevel=3)
            return mean, "the mean of the observed solar zenith of each date"
        if sza.lower() == "hls":
            value = None
            for key in cube.coords:
                if str(key).upper() == "NBAR_SOLAR_ZENITH":
                    value = np.asarray(cube.coords[key].values, dtype=float)
            if value is None:
                value = next((cube.attrs[k] for k in cube.attrs if str(k).upper() == "NBAR_SOLAR_ZENITH"), None)
            if value is None:
                raise ValueError("sza='hls' reads the NBAR_SOLAR_ZENITH tag of HLS files, and this cube has none "
                                 "(as an attribute or a coordinate). HLS computes it from the day and the tile's "
                                 "latitude with overpass times that its user guide does not give: pass sza= a number "
                                 "(degrees) or 'scene'")
            value = np.asarray(value, dtype=float)
            if value.ndim > 1 or value.size not in (1, n_dates):
                raise ValueError(f"the NBAR_SOLAR_ZENITH tag has {value.size} values for {n_dates} dates")
            return np.broadcast_to(value.reshape(-1), (n_dates,)).copy(), "the NBAR_SOLAR_ZENITH tag"
        raise ValueError(f"sza= must be a number (degrees), 'scene' or 'hls', got {sza!r}")
    if np.ndim(sza) > 1 or (np.ndim(sza) == 1 and np.size(sza) not in (1, n_dates)):
        raise ValueError(f"sza= has {np.size(sza)} values for {n_dates} dates")
    value = np.broadcast_to(np.asarray(sza, dtype=float).reshape(-1) if np.ndim(sza) else np.asarray(sza, dtype=float),
                            (n_dates,)).copy()
    if not np.all((value >= 0) & (value < 90)):
        raise ValueError(f"sza= must be a solar zenith of 0 to 90 degrees, got {sza!r}")
    return value, f"{sza} degrees" if np.ndim(sza) == 0 else "a solar zenith per date"


def _nbar_block(block: np.ndarray, ang: np.ndarray, sza_norm: Sequence[float], coefs: np.ndarray, pos: Sequence[int],
                with_c: bool, n_jobs: int, dtype: Any, block_info: Any = None) -> np.ndarray:
    """``(time, band, y, x)`` + ``(time, 4, y, x)`` -> ``(time, band [+ roles], y, x)``: the bands of `pos` are
    normalized, the others copied, and the c-factors follow when `with_c`."""
    from . import _core

    first = block_info[0]["array-location"][0][0] if block_info else 0
    nt, nb, ny, nx = block.shape
    nr = len(pos)
    out = np.empty((nt, nb + (nr if with_c else 0), ny, nx), dtype=dtype)
    out[:, :nb] = block
    for k in range(nt):
        norm = float(sza_norm[first + k])
        if not np.isfinite(norm):
            out[k, list(pos)] = np.nan
            if with_c:
                out[k, nb:] = np.nan
            continue
        refl = np.ascontiguousarray(block[k, list(pos)].reshape(nr, -1), dtype=dtype)
        a = np.ascontiguousarray(ang[k].reshape(4, -1), dtype=dtype)
        res = _core.brdf.nbar_bands(refl, a, coefs, norm, n_jobs, with_c)
        if with_c:
            res, c = res
            out[k, nb:] = c.reshape(nr, ny, nx)
        out[k, list(pos)] = res.reshape(nr, ny, nx)
    return out


def nbar(
    data: Any,
    angles: Any = None,
    *,
    sensor: Any = None,
    sza: Union[str, float, Sequence[float]] = "scene",
    bands: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Sequence[float]]] = None,
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
    return_cfactor: bool = False,
) -> Union[xr.DataArray, Tuple[xr.DataArray, xr.DataArray]]:
    """Reflectance as seen from nadir with the sun at a normalized zenith (NBAR), by the c-factor of Roy et al.
    (2016, 2017): each band is multiplied by ``c = rho(sza_norm, 0) / rho(sza, vza, raa)``, ``rho`` being the
    MODIS Ross-Thick / Li-Sparse-Reciprocal model ``f_iso + f_vol K_vol + f_geo K_geo`` with fixed parameters per
    band (Roy et al. 2016 Table 5; the HLS User Guide v2.0 Table 4 applies the same to OLI and MSI), taken at the
    pixel's observed solar zenith, view zenith and relative azimuth (``VAA - SAA``) in the denominator.

    Written from the papers (Roy et al. 2016 RSE 176:255-271, Eq. 1 and 7; Roy et al. 2017 RSE 199:25-38 and
    Remote Sensing 9:1325; the kernels as Matsuoka et al. 2016, ISPRS Annals III-7:3-10, write them after Wanner
    et al. 1995 and Lucht et al. 2000): not from FORCE's, HLS's or anyone else's code. In HLS the step comes
    before the bandpass adjustment: ``zeit.bandpass_adjust(zeit.nbar(cube, angles))``. The red edge parameters
    are the interpolation between MODIS's red and NIR of Roy, Li and Zhang (2017), which the paper found not
    particularly suited to the red edge. Not for snow (the parameters are MODIS's snow-free ones).

    Parameters
    ----------
    data
        Surface reflectance ``(time, band, y, x)`` or ``(band, y, x)``, in memory or dask, or anything
        ``zeit.load_raster`` reads. HLS cubes are refused (already NBAR), and so are cubes with ``attrs["nbar"]``.
    angles
        The sun and view angles in degrees, on exactly the cube's grid and dates (no resampling): an
        ``xarray.Dataset`` with variables ``SZA``, ``SAA``, ``VZA``, ``VAA`` (or a DataArray with an ``angle``
        dimension) of dimensions ``(time, y, x)`` (or ``(y, x)`` for a cube without time). ``None`` takes bands
        of the cube named like that. The angles of Sentinel-2 are in each tile's ``MTD_TL.xml`` (a 5 km grid); of
        Landsat Collection 2 in the Level-1 angle bands (USGS credentials); HLS carries them, scaled by 0.01.
        A pixel with a zenith outside 0-90 degrees or a missing angle is NaN.
    sensor
        As for ``bandpass_adjust``; it only checks that the sensors are Landsat 4-9 or Sentinel-2 (the parameters
        are the same for all, by band role) and, for Landsat 4, 5 and 7, that the Collection 2 band names
        (``SR_B1``...) are not guessed as OLI's. Not needed if the cube does not say.
    sza
        The normalized solar zenith: a number of degrees (or one per date), ``"scene"`` (default, the mean of
        the observed solar zenith of each date over the pixels whose angles are valid, as in Roy et al. 2016
        and 2017; with dask this reads the angles once ahead of the computation) or ``"hls"`` (the
        ``NBAR_SOLAR_ZENITH`` tag, as an attribute or coordinate of the cube, if it has one). The view is always
        nadir. Roy et al. (2016) advise against a normalized zenith far from the observed ones.
    bands
        The cube's band of each role (``coastal``, ``blue``, ``green``, ``red``, ``rededge1``, ``rededge2``,
        ``rededge3``, ``nir``, ``nir08``, ``swir1``, ``swir2``), when its names are not the usual ones (Earth
        Search ``red``, ``nir`` = B08, ``nir08`` = B8A, ``swir16``; Planetary Computer ``B04``...; Landsat 8-9
        ``SR_B4``...). Sentinel-2's B08 is adjusted (``nir``, MODIS's NIR parameters); bands without a role are
        copied.
    params
        ``{role: (f_iso, f_geo, f_vol)}`` replacing the parameters of those roles (reflectance 0-1).
    nodata
        Missing values besides NaN (``"auto"``: the raster's NoData, 0 for integers without one), a number or
        ``None``. They stay missing.
    chunks
        Inputs read from disk: ``None`` in memory, ``"auto"`` or a dict lazily.
    n_jobs
        Threads: the processors zeit may use by default (``-1``).
    return_cfactor
        Also return the c-factors, ``(time, band, y, x)`` float32 of the normalized bands, to inspect them (the
        c-factor is not clipped; it stays within 0.95-1.05 in a Landsat's field of view).

    Returns
    -------
    xarray.DataArray or tuple
        The cube with the same dimensions, coordinates, type and georeferencing (an integer cube is rounded), lazy
        if the cube is, and a ``nbar_sza`` coordinate (the normalized zenith of each date) when it has time.
        ``attrs["nbar"]`` says what was done, ``attrs["nbar_bands"]`` which bands (role=band). With
        ``return_cfactor``, ``(nbar, cfactor)``.

    Examples
    --------
    >>> angles = xr.Dataset({"SZA": sza, "SAA": saa, "VZA": vza, "VAA": vaa})   # degrees, (time, y, x)
    >>> cube_nbar = zeit.nbar(cube, angles)
    >>> adjusted = zeit.bandpass_adjust(cube_nbar)          # HLS order: NBAR, then the bandpass
    """
    from ._bandpass import _sensors
    from ._embeddings import refuse
    from ._load import load_raster
    from ._series_api import _missing_as_nan
    from ._threads import resolve

    refuse(data, "zeit.nbar")
    cube = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if "band" not in cube.dims:
        raise ValueError("nbar needs a cube with a band dimension, (time, band, y, x) or (band, y, x)")
    if not {"y", "x"} <= set(cube.dims):
        raise ValueError("nbar needs a cube with y and x dimensions")
    unknown = [d for d in cube.dims if d not in ("time", "band", "y", "x")]
    if unknown:
        raise ValueError(f"nbar needs a cube of (time, band, y, x) or (band, y, x), not dimensions {unknown}")
    if "id" in cube.coords and any(str(v).upper().startswith("HLS.") for v in np.atleast_1d(cube.id.values)):
        raise ValueError("the cube is HLS, whose reflectance is already normalized to nadir BRDF-adjusted "
                         "reflectance (NBAR)")
    if cube.attrs.get("nbar"):
        raise ValueError(f"the cube is already NBAR ({cube.attrs['nbar']})")
    if "scale" in cube.coords and np.issubdtype(cube.dtype, np.integer):
        raise ValueError("nbar needs reflectance: the cube holds digital numbers with per-scene scale and offset; "
                         "build it with a float dtype (build_time_series' default)")
    if not isinstance(return_cfactor, bool):
        raise ValueError("return_cfactor must be True or False")

    has_time = "time" in cube.dims
    n_dates = cube.sizes["time"] if has_time else 1
    sensors = None
    if sensor is not None or any(k in cube.coords for k in ("platform", "sensor", "spacecraft", "satellite")) \
            or any(cube.attrs.get(k) for k in ("platform", "sensor", "spacecraft", "satellite")):
        sensors = _sensors(cube, sensor)

    layers, own = _angle_layers(angles, cube)
    layers = _align(layers, cube)
    roles = _roles(cube, bands, sensors, own)
    if not roles:
        raise ValueError(f"no band of the cube {[str(b) for b in cube.band.values]} has a BRDF role; pass bands= "
                         f"(role -> band, roles {list(ROLES)})")
    table = _parameters(params)
    names = list(roles)
    coefs = np.array([table[r] for r in names], dtype=np.float64)
    position = {b: k for k, b in enumerate(cube.band.values)}
    pos = [position[roles[r]] for r in names]

    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = _missing_as_nan(cube, nodata).transpose(*order)
    arr = work.data if work.chunks is not None else work.values
    dtype = np.float64 if work.dtype == np.float64 else np.float32
    ang = layers.data if layers.chunks is not None else layers.values
    if not has_time:
        arr, ang = arr[None], ang[None]
    ang_dask = not isinstance(ang, np.ndarray)
    sza_norm, how = _norm_sza(sza, cube, ang, n_dates)
    threads = resolve(n_jobs)
    nb = cube.sizes["band"]
    nr = len(names)
    n_out = nb + (nr if return_cfactor else 0)

    if work.chunks is not None:
        import dask.array as da

        arr = arr.rechunk({1: -1})
        ang = (ang if ang_dask else da.from_array(ang, chunks=(arr.chunks[0], 4) + arr.chunks[2:])) \
            .rechunk((arr.chunks[0], (4,)) + arr.chunks[2:])
        chunks_out = (arr.chunks[0], (n_out,)) + arr.chunks[2:]
        out = da.map_blocks(_nbar_block, arr, ang, [float(v) for v in sza_norm], coefs, pos, return_cfactor, threads,
                            dtype, dtype=dtype, chunks=chunks_out, meta=np.empty((0, 0, 0, 0), dtype))
    else:
        if ang_dask:
            ang = np.asarray(ang)
        out = _nbar_block(np.asarray(arr), ang, [float(v) for v in sza_norm], coefs, pos, return_cfactor, threads,
                          dtype)
    if not has_time:
        out = out[0]
    axis = 1 if has_time else 0

    coords = {k: v for k, v in cube.coords.items() if "band" not in v.dims}
    if has_time:
        coords["nbar_sza"] = ("time", sza_norm)
    result = xr.DataArray(out[(slice(None),) * axis + (slice(0, nb),)], dims=order,
                          coords={**coords, "band": cube.band.values}, name=cube.name)
    if np.issubdtype(cube.dtype, np.integer):
        info = np.iinfo(cube.dtype)
        fill = cube.rio.nodata if cube.rio.nodata is not None else (0 if nodata == "auto" else nodata)
        result = result.round().clip(info.min, info.max).fillna(fill if fill is not None else 0).astype(cube.dtype)
    else:
        result = result.astype(cube.dtype)
    result = result.transpose(*cube.dims)
    result.attrs = dict(cube.attrs)
    used = ", ".join(f"{r}={roles[r]}" for r in names)
    result.attrs["nbar"] = f"c-factor, Roy et al. (2016, 2017); sza={how}; nadir view"
    result.attrs["nbar_bands"] = used
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    if cube.rio.nodata is not None:
        result = result.rio.write_nodata(cube.rio.nodata, encoded=False)
    if not return_cfactor:
        return result
    cf = xr.DataArray(out[(slice(None),) * axis + (slice(nb, None),)], dims=order,
                      coords={**coords, "band": [roles[r] for r in names]}, name="cfactor").transpose(*cube.dims)
    cf.attrs = {"long_name": "c-factor (NBAR / observed reflectance)", "nbar": result.attrs["nbar"],
                "nbar_bands": used}
    if cube.rio.crs is not None:
        cf = cf.rio.write_crs(cube.rio.crs)
    return result, cf
