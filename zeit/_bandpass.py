"""Landsat and Sentinel-2 on one radiometric scale: ``zeit.bandpass_adjust``.

A dense series that mixes sensors has a step wherever the sensor changes: the bands of
Sentinel-2's MSI are not those of Landsat 8's OLI, and CCDC or BFAST read the step as a
break. The bandpass adjustment of HLS (Harmonized Landsat Sentinel-2, Claverie et al. 2018)
removes it: a line per band and Sentinel-2 unit that takes MSI's reflectance to OLI's.
OLI is the reference, as in HLS: Landsat 8 and 9 stay as they are.

ETM+ (Landsat 7) and TM (Landsat 4-5) can also be taken to OLI with the lines of Roy et
al. (2016), but only when asked: they were fitted on pre-collection data, and for Landsat
Collection 2 surface reflectance the USGS products are considered consistent without them
(Earth Engine's FAQ on cross-sensor harmonization).
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

#: Roles of the adjusted bands, in this order in the tables.
ROLES = ("coastal", "blue", "green", "red", "nir", "swir1", "swir2")

#: HLS bandpass adjustment, OLI = slope * MSI + intercept (reflectance 0-1), per Sentinel-2
#: unit: HLS v2.0 User Guide (2026), Table 5; S2A and S2B are also the v1.4 values
#: (Claverie et al. 2018). NIR is MSI's narrow NIR, B8A; there is none for B08 or the red edge.
HLS = {
    "S2A": {"coastal": (0.9959, -0.0002), "blue": (0.9778, -0.0040), "green": (1.0053, -0.0009),
            "red": (0.9765, 0.0009), "nir": (0.9983, -0.0001), "swir1": (0.9987, -0.0011),
            "swir2": (1.0030, -0.0012)},
    "S2B": {"coastal": (0.9959, -0.0002), "blue": (0.9778, -0.0040), "green": (1.0075, -0.0008),
            "red": (0.9761, 0.0010), "nir": (0.9966, 0.0000), "swir1": (1.0000, -0.0003),
            "swir2": (0.9867, 0.0004)},
    "S2C": {"coastal": (1.0030, 0.0000), "blue": (0.9851, -0.0027), "green": (1.0038, -0.0009),
            "red": (0.9718, 0.0011), "nir": (0.9995, -0.0003), "swir1": (0.9994, -0.0007),
            "swir2": (0.9910, 0.0004)},
}

#: Roy et al. (2016), Table 2 (surface reflectance), ETM+ -> OLI: OLI = intercept + slope *
#: ETM+, by ordinary least squares or reduced major axis. TM is treated as ETM+, as Earth
#: Engine's harmonization tutorial does (no TM-specific lines are published).
ROY_2016 = {
    "ols": {"blue": (0.8474, 0.0003), "green": (0.8483, 0.0088), "red": (0.9047, 0.0061),
            "nir": (0.8462, 0.0412), "swir1": (0.8937, 0.0254), "swir2": (0.9071, 0.0172)},
    "rma": {"blue": (0.9785, -0.0095), "green": (0.9542, -0.0016), "red": (0.9825, -0.0022),
            "nir": (1.0073, -0.0021), "swir1": (1.0171, -0.0030), "swir2": (0.9949, 0.0029)},
}

#: Platform names (STAC's ``platform``, Earth Engine's ``SPACECRAFT_ID``, scene ids...) -> sensor.
_SENSORS = [
    (r"sentinel[-_ ]?2a|^s2a\b", "S2A"), (r"sentinel[-_ ]?2b|^s2b\b", "S2B"), (r"sentinel[-_ ]?2c|^s2c\b", "S2C"),
    (r"sentinel[-_ ]?2|^s2\b|\bmsi\b", "S2"),
    (r"landsat[-_ ]?0?9|^lc0?9", "OLI"), (r"landsat[-_ ]?0?8|^lc0?8|\boli", "OLI"),
    (r"landsat[-_ ]?0?7|^le0?7|etm", "ETM"), (r"landsat[-_ ]?0?[45]|^l[tm]0?[45]|\btm\b", "TM"),
]

#: Band names of each role. Sentinel-2's B08 (``nir`` on Earth Search, ``B08`` on Planetary
#: Computer) is the broad NIR, which HLS does not adjust; ``nir08``/``B8A`` is the narrow one.
_NAMES = {
    "coastal": ("coastal", "B01", "SR_B1"),
    "blue": ("blue", "B02", "SR_B2"),
    "green": ("green", "B03", "SR_B3"),
    "red": ("red", "B04", "SR_B4"),
    "nir": ("nir08", "B8A", "nir", "SR_B5"),
    "swir1": ("swir16", "B11", "SR_B6"),
    "swir2": ("swir22", "B12", "SR_B7"),
}
_BROAD_NIR = ("nir", "b08")


def sensor_of(name: Any) -> Optional[str]:
    """``"S2A"``, ``"S2B"``, ``"S2C"``, ``"S2"`` (unit unknown), ``"OLI"``, ``"ETM"``,
    ``"TM"`` or None, from a platform name such as ``"sentinel-2a"`` or ``"LANDSAT_8"``."""
    if name is None:
        return None
    text = str(name).strip().lower()
    for pattern, sensor in _SENSORS:
        if re.search(pattern, text):
            return sensor
    return None


def _sensors(cube: xr.DataArray, sensor: Any) -> List[str]:
    """The sensor of every date."""
    n = cube.sizes.get("time", 1)
    if sensor is None:
        for key in ("platform", "sensor", "spacecraft", "satellite"):
            if key in cube.coords:
                values = np.atleast_1d(np.asarray(cube.coords[key].values))
                break
        else:
            for key in ("platform", "sensor", "spacecraft", "satellite"):
                if cube.attrs.get(key):
                    values = np.array([cube.attrs[key]])
                    break
            else:
                raise ValueError("bandpass_adjust needs each date's sensor: the cube has no 'platform' coordinate "
                                 "(build_time_series keeps the one of STAC); pass sensor= (a name such as "
                                 "'sentinel-2a' or 'landsat-8' for every date, or one per date)")
        if values.size == 1:
            values = np.repeat(values, n)
    elif isinstance(sensor, str):
        values = np.repeat(np.array([sensor]), n)
    else:
        values = np.asarray(list(sensor), dtype=object)
        if values.size != n:
            raise ValueError(f"sensor= has {values.size} names for {n} dates")
    out = []
    for value in values:
        found = sensor_of(value)
        if found is None:
            raise ValueError(f"unknown sensor {value!r}: Sentinel-2 (A, B, C) or Landsat 4-9 are known")
        out.append(found)
    return out


def _roles(cube: xr.DataArray, bands: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The cube's band of each role it has."""
    available = [str(b) for b in cube.band.values]
    lower = [b.lower() for b in available]
    out = {}
    for role, given in (bands or {}).items():
        if role not in ROLES:
            raise ValueError(f"bands=: unknown role {role!r}; roles are {list(ROLES)}")
        if str(given) not in available:
            raise ValueError(f"bands=: {given!r} is not a band of the cube {available}")
        out[role] = cube.band.values[available.index(str(given))]
    for role in ROLES:
        if role in out:
            continue
        for name in _NAMES[role]:
            if name.lower() in lower and cube.band.values[lower.index(name.lower())] not in out.values():
                out[role] = cube.band.values[lower.index(name.lower())]
                break
    return out


def _scale_of(cube: xr.DataArray, scale: Any) -> float:
    if scale != "auto":
        return float(scale)
    if np.issubdtype(cube.dtype, np.integer):
        return 10000.0
    if cube.chunks is None:
        sample = cube.isel({d: slice(None, None, max(1, cube.sizes[d] // 16)) for d in cube.dims if d != "band"})
    else:   # lazy, maybe remote: a 64 x 64 window of three dates, not a read of the whole cube
        window = {d: slice(max(0, cube.sizes[d] // 2 - 32), cube.sizes[d] // 2 + 32) for d in ("y", "x")}
        if "time" in cube.dims:
            window["time"] = np.unique(np.linspace(0, cube.sizes["time"] - 1, 3).astype(int))
        sample = cube.isel(window)
    values = np.asarray(sample.values, dtype=float)
    high = np.nanpercentile(np.abs(values), 99) if np.isfinite(values).any() else 1.0
    return 10000.0 if high > 2.0 else 1.0


def coefficients(sensors: Sequence[str], roles: Sequence[str], etm: Optional[str] = None,
                 s2_skip: Sequence[str] = ()) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """``(slope, intercept)`` arrays ``(date, role)`` taking each date to OLI (reflectance
    0-1), and notes on what was left as it was. ``s2_skip``: roles left as they are on the
    Sentinel-2 dates (a broad NIR)."""
    if etm is not None and etm not in ROY_2016:
        raise ValueError(f"etm must be None, 'ols' or 'rma', got {etm!r}")
    slope = np.ones((len(sensors), len(roles)))
    intercept = np.zeros_like(slope)
    notes = []
    for i, sensor in enumerate(sensors):
        if sensor in ("S2", "S2A", "S2B", "S2C"):
            table = HLS["S2A" if sensor == "S2" else sensor]
        elif sensor in ("ETM", "TM") and etm is not None:
            table = ROY_2016[etm]
        else:
            continue
        for k, role in enumerate(roles):
            if role in table and not (sensor.startswith("S2") and role in s2_skip):
                slope[i, k], intercept[i, k] = table[role]
    if "S2" in sensors:
        notes.append("Sentinel-2 dates of unknown unit (A, B or C) were adjusted as Sentinel-2A")
    return slope, intercept, notes


def bandpass_adjust(
    data: Any,
    *,
    sensor: Any = None,
    etm: Optional[str] = None,
    bands: Optional[Dict[str, Any]] = None,
    scale: Union[str, float] = "auto",
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
) -> xr.DataArray:
    """Sentinel-2 and Landsat reflectance on the scale of Landsat 8's OLI, so that a series
    mixing them has no step where the sensor changes.

    Parameters
    ----------
    data
        Surface reflectance ``(time, band, y, x)`` or ``(band, y, x)``, in memory or dask, or
        anything ``zeit.load_raster`` reads. Or a list of cubes on the same grid (one per
        collection, say): each is adjusted, and they are joined into one series sorted by
        time, with the bands they share matched by role and named after it (``blue``...
        ``swir2``) and the coordinates they share.
    sensor
        Each date's sensor. By default the cube's ``platform`` coordinate (which
        ``build_time_series`` keeps from STAC) or attribute; or a name for every date
        (``"sentinel-2a"``, ``"S2B"``, ``"landsat-8"``, ``"LANDSAT_7"``...), or one per date.
        With a list of cubes, one of these per cube.
    etm
        Landsat 7 (ETM+) and 4-5 (TM): ``None`` (default) leaves them as they are; ``"rma"``
        or ``"ols"`` takes them to OLI with the reduced major axis or least squares lines of
        Roy et al. (2016). Those were fitted on pre-collection data: Collection 2 surface
        reflectance is generally used without them.
    bands
        The cube's band of each role (``coastal``, ``blue``, ``green``, ``red``, ``nir``,
        ``swir1``, ``swir2``), when its names are not the usual ones (``red``, ``B04``,
        ``SR_B4``...). For Sentinel-2 the NIR is B8A (``nir08``, ``B8A``): HLS has no line for
        B08 (``nir`` on Earth Search), which is left as it is, with a warning; pass
        ``bands={"nir": ...}`` to say a band is B8A. With a list of cubes, one dict for all or
        one per cube.
    scale
        What the reflectance is multiplied by in the data: ``"auto"`` (10000 for integers or
        values above 2, else 1), or a number. The intercepts are in reflectance.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without
        one), a number or ``None``. They stay missing.
    chunks
        Inputs read from disk: ``None`` in memory, ``"auto"`` or a dict lazily.

    Returns
    -------
    xarray.DataArray
        The cube with the same dimensions, coordinates, type and georeferencing; lazy if the
        cube is. Sentinel-2 dates go through the HLS bandpass adjustment of their unit (HLS
        v2.0, Claverie et al. 2018), Landsat 8 and 9 stay as they are, and bands without a
        line (red edge, B08, QA) are copied. ``attrs["bandpass_adjusted"]`` says what was done.
        This is the bandpass step of HLS only: not its common atmospheric correction, cloud
        mask or BRDF normalization (for those, load the HLS products themselves).

    Examples
    --------
    >>> s2 = zeit.build_time_series(collection="sentinel-2-l2a", ...)
    >>> landsat = zeit.build_time_series(collection="landsat-c2-l2", ...)   # the same grid
    >>> mixed = zeit.bandpass_adjust([s2, landsat])       # one series: blue, green, red, nir, swir1, swir2
    >>> zeit.ccdc(mixed)                             # no break where the sensor changes
    """
    if isinstance(data, (list, tuple)) and data and all(isinstance(c, xr.DataArray) for c in data):
        return _join(list(data), sensor=sensor, etm=etm, bands=bands, scale=scale, nodata=nodata)
    return _one(data, sensor=sensor, etm=etm, bands=bands, scale=scale, nodata=nodata, chunks=chunks)


def _join(cubes: List[xr.DataArray], *, sensor: Any, etm: Optional[str], bands: Any, scale: Any,
          nodata: Any) -> xr.DataArray:
    """Several cubes adjusted and joined in time, their shared bands named by role."""
    n = len(cubes)
    sensors = [None] * n if sensor is None else sensor
    if isinstance(sensors, str) or len(sensors) != n:
        raise ValueError(f"with {n} cubes, sensor= has one entry per cube (a name, or one per date)")
    maps = list(bands) if isinstance(bands, (list, tuple)) else [bands] * n
    if len(maps) != n:
        raise ValueError(f"with {n} cubes, bands= is one dict for all or one per cube")
    roles = [_roles(c, m) for c, m in zip(cubes, maps)]
    common = [r for r in ROLES if all(r in found for found in roles)]
    if not common:
        raise ValueError(f"the cubes share no band role ({list(ROLES)}); pass bands= for each")
    parts = []
    for cube, found, s, m in zip(cubes, roles, sensors, maps):
        part = _one(cube.sel(band=[found[r] for r in common]), sensor=s, etm=etm, bands=m, scale=scale, nodata=nodata)
        parts.append(part.assign_coords(band=common))
    from .cube import join_in_time

    out = join_in_time(parts)
    done = [x for x in dict.fromkeys(x for p in parts for x in p.attrs["bandpass_adjusted"].split("; "))
            if not x.startswith("nothing")]
    out.attrs["bandpass_adjusted"] = "; ".join(done) if done else parts[0].attrs["bandpass_adjusted"]
    out.attrs["bandpass_bands"] = ", ".join(common)
    return out


def _one(data: Any, *, sensor: Any, etm: Optional[str], bands: Any, scale: Any, nodata: Any,
         chunks: Any = None) -> xr.DataArray:
    from ._embeddings import refuse
    from ._load import load_raster

    refuse(data, "zeit.bandpass_adjust")
    cube = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if "band" not in cube.dims:
        raise ValueError("bandpass_adjust needs a cube with a band dimension, (time, band, y, x) or (band, y, x)")
    if "id" in cube.coords and any(str(v).upper().startswith("HLS.") for v in np.atleast_1d(cube.id.values)):
        raise ValueError("the cube is HLS, whose Sentinel-2 dates are already bandpass-adjusted to OLI")
    if cube.attrs.get("bandpass_adjusted"):
        raise ValueError(f"the cube is already bandpass-adjusted ({cube.attrs['bandpass_adjusted']})")
    if "scale" in cube.coords and np.issubdtype(cube.dtype, np.integer):
        raise ValueError("bandpass_adjust needs reflectance: the cube holds digital numbers with per-scene scale and "
                         "offset; build it with a float dtype (build_time_series' default)")
    sensors = _sensors(cube, sensor)
    roles = _roles(cube, bands)
    if not roles:
        raise ValueError(f"no band of the cube {[str(b) for b in cube.band.values]} is one HLS adjusts; pass "
                         f"bands= (role -> band, roles {list(ROLES)})")
    s2 = [s.startswith("S2") for s in sensors]
    nir = roles.get("nir")
    if nir is not None and any(s2) and str(nir).lower() in _BROAD_NIR and not (bands and "nir" in bands):
        warnings.warn(f"band {nir!r} of the Sentinel-2 dates is the broad NIR (B08), which HLS does not adjust: "
                      "it is left as it is. Load B8A (nir08) for an adjusted NIR, or pass bands={'nir': ...} if "
                      "this band is B8A.", stacklevel=2)
        s2_skip = ("nir",)
    else:
        s2_skip = ()
    names = list(roles)
    slope, intercept, notes = coefficients(sensors, names, etm, s2_skip)
    factor = _scale_of(cube, scale)

    nb = cube.sizes["band"]
    position = {b: k for k, b in enumerate(cube.band.values)}
    full_slope, full_intercept = np.ones((len(sensors), nb)), np.zeros((len(sensors), nb))
    for k, role in enumerate(names):
        full_slope[:, position[roles[role]]] = slope[:, k]
        full_intercept[:, position[roles[role]]] = intercept[:, k] * factor
    dims = ("time", "band") if "time" in cube.dims else ("band",)
    if "time" not in cube.dims:
        full_slope, full_intercept = full_slope[0], full_intercept[0]
    a = xr.DataArray(full_slope, dims=dims, coords={"band": cube.band.values})
    b = xr.DataArray(full_intercept, dims=dims, coords={"band": cube.band.values})

    from ._series_api import _missing_as_nan

    work = _missing_as_nan(cube, nodata)
    compute = np.float64 if work.dtype == np.float64 else np.float32
    out = work * a.astype(compute) + b.astype(compute)
    if np.issubdtype(cube.dtype, np.integer):
        info = np.iinfo(cube.dtype)
        fill = cube.rio.nodata if cube.rio.nodata is not None else (0 if nodata == "auto" else nodata)
        out = out.round().clip(info.min, info.max).fillna(fill if fill is not None else 0).astype(cube.dtype)
    else:
        out = out.astype(cube.dtype)
    out = out.transpose(*cube.dims)
    out.name = cube.name
    out.attrs = dict(cube.attrs)
    done = ["Sentinel-2 -> OLI: HLS v2.0 bandpass adjustment (Claverie et al. 2018)"] if any(s2) else []
    if etm is not None and any(s in ("ETM", "TM") for s in sensors):
        done.append(f"ETM+/TM -> OLI: Roy et al. (2016), {etm.upper()}")
    out.attrs["bandpass_adjusted"] = "; ".join(done) if done else "nothing to adjust (Landsat 8/9 only)"
    out.attrs["bandpass_bands"] = ", ".join(f"{r}={roles[r]}" for r in names)
    for note in notes:
        warnings.warn(note, stacklevel=2)
    if cube.rio.crs is not None:
        out = out.rio.write_crs(cube.rio.crs)
    if cube.rio.nodata is not None:
        out = out.rio.write_nodata(cube.rio.nodata, encoded=False)
    return out
