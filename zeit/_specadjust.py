"""Landsat and MODIS reflectance on the spectral response of Sentinel-2A: ``zeit.spectral_adjust``.

FORCE's spectral adjustment (Scheffler, Frantz and Segl 2020, RSE 241:111723, as implemented in
FORCE 3.11, David Frantz, GPL-3.0-or-later): the six bands of Landsat 4-9 or MODIS become the ten
bands of Sentinel-2A, the red edge (B5-B7) and the broad NIR (B8) predicted. Computed by the
compiled core (``zeit._core.specadjust``).
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

#: The sensors of FORCE's tables, by FORCE's name.
SENSORS = ("LND04", "LND05", "LND07", "LND08", "LND09", "MOD01", "MOD02")

#: The six source bands (the order of the core) and their names, as ``zeit.bandpass_adjust`` finds them.
SOURCE_ROLES = ("blue", "green", "red", "nir", "swir1", "swir2")

#: The ten Sentinel-2A bands, in the order of the core's output, named as Earth Search names them
#: (``nir`` is B08, the broad NIR; ``nir08`` is B8A), and the names a cube may have them under.
TARGET_BANDS = ("blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22")
_TARGET_NAMES = {
    "blue": ("blue", "B02", "B2"),
    "green": ("green", "B03", "B3"),
    "red": ("red", "B04", "B4"),
    "rededge1": ("rededge1", "B05", "B5"),
    "rededge2": ("rededge2", "B06", "B6"),
    "rededge3": ("rededge3", "B07", "B7"),
    "nir": ("nir", "B08", "B8"),
    "nir08": ("nir08", "B8A"),
    "swir16": ("swir16", "B11"),
    "swir22": ("swir22", "B12"),
}

#: Platform names (STAC's ``platform``, Earth Engine's ``SPACECRAFT_ID``, scene ids, FORCE's names) -> sensor.
#: ``SEN2`` is any Sentinel-2 unit: those dates are passed through. MODIS is Terra (MOD01 of FORCE;
#: MOD09...) or Aqua (MOD02; MYD09...): the bare word "modis" does not say which.
_SENSOR_PATTERNS = [
    (r"sentinel[-_ ]?2|^s2[abcd]?\b|\bmsi\b", "SEN2"),
    (r"^lnd0?([45789])$", None),     # FORCE's own names, handled below
    (r"landsat[-_ ]?0?9|^lc0?9|\boli[-_ ]?2", "LND09"),
    (r"landsat[-_ ]?0?8|^lc0?8|\boli", "LND08"),
    (r"landsat[-_ ]?0?7|^le0?7|etm", "LND07"),
    (r"landsat[-_ ]?0?5|^lt0?5", "LND05"),
    (r"landsat[-_ ]?0?4|^lt0?4", "LND04"),
    (r"^mod0?1$|terra|^mod\d\d[a-z]\d|^mod09|^mod13", "MOD01"),
    (r"^mod0?2$|aqua|^myd", "MOD02"),
]


def sensor_of(name: Any) -> Optional[str]:
    """FORCE's name of the sensor (``"LND04"``... ``"LND09"``, ``"MOD01"`` Terra, ``"MOD02"`` Aqua), ``"SEN2"``
    for any Sentinel-2 unit, or None, from a platform name such as ``"landsat-8"``, ``"LANDSAT_5"``, ``"LC09"`` or
    ``"Terra"``. Landsat 4 and 5 are told apart (``"TM"`` alone does not say which)."""
    if name is None:
        return None
    text = str(name).strip().lower()
    for pattern, sensor in _SENSOR_PATTERNS:
        found = re.search(pattern, text)
        if found:
            return sensor if sensor is not None else f"LND0{found.group(1)}".upper()
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
                raise ValueError("spectral_adjust needs each date's sensor: the cube has no 'platform' coordinate "
                                 "(build_time_series keeps the one of STAC); pass sensor= (a name such as "
                                 "'landsat-8' or 'LND08' for every date, or one per date)")
        if values.size == 1:
            values = np.repeat(values, n)
    elif isinstance(sensor, str):
        values = np.repeat(np.array([sensor]), n)
    else:
        values = np.asarray(list(sensor), dtype=object)
        if values.size != n:
            raise ValueError(f"sensor= has {values.size} names for {n} dates")
    if values.size != n:
        raise ValueError(f"the platform has {values.size} values for {n} dates")
    dates = cube.time.values if "time" in cube.dims else [None]
    out = []
    for date, value in zip(dates, values):
        found = sensor_of(value)
        if found is None:
            when = "" if date is None else f" (date {str(np.datetime_as_string(date, unit='D'))})"
            raise ValueError(f"unknown sensor {str(value)!r}{when}: the known sensors are Landsat 4, 5, 7, 8, 9, MODIS "
                             f"Terra (MOD01) and Aqua (MOD02), named in FORCE LND04, LND05, LND07, LND08, "
                             f"LND09, MOD01, MOD02, and Sentinel-2, which passes through")
        out.append(found)
    return out


def _find(cube: xr.DataArray, bands: Optional[Dict[str, Any]], names: Dict[str, Sequence[str]],
          roles: Sequence[str], what: str) -> Dict[str, int]:
    """The position along the band dimension of each of `roles`, by the given ``bands`` or by the usual names."""
    available = [str(b) for b in cube.band.values]
    lower = [b.lower() for b in available]
    given = {}
    for role, band in (bands or {}).items():
        if role not in roles:
            continue
        if str(band) not in available:
            raise ValueError(f"bands=: {band!r} is not a band of the cube {available}")
        given[role] = available.index(str(band))
    out = {}
    taken = set(given.values())
    for role in roles:
        if role in given:
            out[role] = given[role]
            continue
        for name in names[role]:
            if name.lower() in lower and lower.index(name.lower()) not in taken:
                out[role] = lower.index(name.lower())
                taken.add(out[role])
                break
    missing = [r for r in roles if r not in out]
    if missing:
        raise ValueError(f"the cube {available} has no band for {what}: {missing}; pass bands= (role -> band)")
    return out


def _source_names() -> Dict[str, Sequence[str]]:
    from ._bandpass import _NAMES

    return {r: _NAMES[r] for r in SOURCE_ROLES}


def _adjust_block(block: np.ndarray, sensors: Sequence[str], src: Sequence[int], dst: Sequence[int], factor: float,
                  n_jobs: int, dtype: Any, force_search: bool = False, block_info: Any = None) -> np.ndarray:
    """``(time, band, y, x)`` -> ``(time, 10, y, x)``: `sensors` is that of every date of the whole cube,
    `src` the positions of the six source bands, `dst` those of the ten Sentinel-2 bands."""
    from . import _core

    first = block_info[0]["array-location"][0][0] if block_info else 0
    nt, _, ny, nx = block.shape
    out = np.empty((nt, len(TARGET_BANDS), ny, nx), dtype=dtype)
    scale = 10000.0 / factor
    for k in range(nt):
        sensor = sensors[first + k]
        if sensor == "SEN2":
            out[k] = block[k, list(dst)]
            continue
        x = np.moveaxis(block[k, list(src)], 0, -1).reshape(-1, len(SOURCE_ROLES))
        if factor == 10000.0 and x.dtype in (np.float32, np.float64):
            x = np.ascontiguousarray(x)
        else:
            x = np.ascontiguousarray(x, dtype=np.float64) * scale
        y = _core.specadjust.specadjust_batch(x, sensor, n_jobs, force_search)          # (pixels, 10), x 10000
        if factor != 10000.0:
            y = y * (factor / 10000.0)
        out[k] = np.moveaxis(y.reshape(ny, nx, len(TARGET_BANDS)), -1, 0)
    return out


def spectral_adjust(
    data: Any,
    *,
    sensor: Any = None,
    bands: Optional[Dict[str, Any]] = None,
    scale: Union[str, float] = "auto",
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
    force_search: bool = False,
) -> xr.DataArray:
    """Landsat and MODIS surface reflectance on the spectral response of Sentinel-2A, with the red edge bands and
    the broad NIR that those sensors do not have predicted: FORCE's spectral adjustment.

    Each sensor (Landsat 4, 5, 7, 8, 9, MODIS Terra, MODIS Aqua) has 50 clusters of spectra and a global one (51),
    each with a linear regression of the ten Sentinel-2A bands on the six of the sensor (blue, green, red, NIR,
    SWIR1, SWIR2) and an intercept. For each pixel the spectral angle (SAM) to the centre of every cluster gives a
    weight, ``1 - SAM / 15 degrees`` (``0.00001`` beyond 15 degrees). The closest clusters, at most 10, while
    their weight is above that of 4 degrees, or above that of 12 degrees and the global cluster's, are used; the
    prediction is the average of their regressions weighted by those weights, and the global cluster fills in
    where fewer than 10 are close (it falls back on the global regression where none is). The tables are those of
    Scheffler, Frantz and Segl (2020), trained for FORCE, and the arithmetic is FORCE's 3.11
    (``spec-adjust-hl.c``, David Frantz, GPL-3.0-or-later), so that the values truncated to integers are FORCE's.
    All ten bands are rewritten, not only the new ones (the blue band of Landsat 8 is not Sentinel-2A's).

    This is not ``zeit.bandpass_adjust``, which takes Sentinel-2 *towards* OLI with a line per band (HLS) and
    predicts no new bands: this takes Landsat and MODIS towards Sentinel-2A, with a regression on all six bands.

    Differences from FORCE. FORCE's first search for the closest cluster skips cluster 9 (it tests ``s != 9``
    where it means to leave out the global cluster, 50, as its other searches do) and lets the global cluster be
    the first pick; here the global cluster never is, and cluster 9 competes in every search, as in the paper. The
    pixels this changes are those where cluster 9 or the global cluster is the closest: on synthetic mixtures of
    land covers, 1.6-2.3% of the observations differ from FORCE's integers by more than 1 (the core with FORCE's own
    search equals FORCE in every value; ``force_search=True`` gives it). FORCE never reads the
    nodata value (a mask decides which pixels are adjusted) and truncates to 16 bits; here a pixel with a missing
    band (NaN or ``nodata``) is missing in all ten, and the output is float and not truncated.

    Parameters
    ----------
    data
        Surface reflectance ``(time, band, y, x)`` or ``(band, y, x)``, in memory or dask, or anything
        ``zeit.load_raster`` reads, with the bands blue, green, red, NIR, SWIR1 and SWIR2 (``blue``/``B02``/
        ``SR_B2``... as for ``bandpass_adjust``). Sentinel-2 dates are passed through (their ten bands by name:
        ``blue green red rededge1 rededge2 rededge3 nir nir08 swir16 swir22``, or ``B02``... ``B12``).
    sensor
        Each date's sensor. By default the cube's ``platform`` coordinate (which ``build_time_series`` keeps
        from STAC) or attribute; or a name for every date (``"landsat-8"``, ``"LC09"``, ``"LND05"``, ``"terra"``,
        ``"MOD02"``...), or one per date. Landsat 4 and 5 are told apart, and so are MODIS Terra and Aqua. Other
        sensors are an error naming the date.
    bands
        The cube's band of each of the six roles (``blue``, ``green``, ``red``, ``nir``, ``swir1``, ``swir2``;
        for MODIS ``swir1`` is band 6 and ``swir2`` band 7), or of the ten Sentinel-2 names above for the
        Sentinel-2 dates, when its names are not the usual ones.
    scale
        What the reflectance is multiplied by in the data: ``"auto"`` (10000 for integers or values above 2, else
        1), or a number. The method works on reflectance times 10000 (its intercepts are not scale-free): the
        data are converted to that and the result back to the data's scale.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or
        ``None``.
    chunks
        Inputs read from disk: ``None`` in memory, ``"auto"`` or a dict lazily.
    n_jobs
        Threads: the processors zeit may use by default (``-1``).
    force_search
        Use FORCE's first search as it is (cluster 9 skipped, the global cluster a possible first pick), so that
        the values truncated to integers are FORCE's in every pixel. Default ``False``: the search as meant.

    Returns
    -------
    xarray.DataArray
        ``(time, band, y, x)`` (or ``(band, y, x)``), float32, with the same coordinates and georeferencing and ten
        bands named as Earth Search names them, so that ``zeit.compute_indices`` finds them: ``blue``, ``green``,
        ``red``, ``rededge1``, ``rededge2``, ``rededge3``, ``nir`` (B08), ``nir08`` (B8A), ``swir16``, ``swir22``.
        Lazy if the input is. ``attrs["spectral_adjusted"]`` says what was done for each sensor.

    Examples
    --------
    >>> landsat = zeit.build_time_series(collection="landsat-c2-l2", ...)
    >>> s2like = zeit.spectral_adjust(landsat)               # blue ... swir22, with rededge1-3 and nir08
    >>> s2like.sel(band="rededge1")
    """
    from ._embeddings import refuse
    from ._load import load_raster
    from ._series_api import _missing_as_nan
    from ._bandpass import _scale_of
    from ._threads import resolve

    refuse(data, "zeit.spectral_adjust")
    cube = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if "band" not in cube.dims:
        raise ValueError("spectral_adjust needs a cube with a band dimension, (time, band, y, x) or (band, y, x)")
    if cube.attrs.get("spectral_adjusted"):
        raise ValueError(f"the cube is already spectrally adjusted ({cube.attrs['spectral_adjusted']})")
    if "scale" in cube.coords and np.issubdtype(cube.dtype, np.integer):
        raise ValueError("spectral_adjust needs reflectance: the cube holds digital numbers with per-scene scale and "
                         "offset; build it with a float dtype (build_time_series' default)")
    if not {"y", "x"} <= set(cube.dims):
        raise ValueError("spectral_adjust needs a cube with y and x dimensions")
    unknown = [k for k in (bands or {}) if k not in SOURCE_ROLES and k not in TARGET_BANDS]
    if unknown:
        raise ValueError(f"bands=: unknown role {unknown}; roles are {list(SOURCE_ROLES)} and, for Sentinel-2 "
                         f"dates, {list(TARGET_BANDS)}")
    sensors = _sensors(cube, sensor)
    has_time = "time" in cube.dims
    needs_source = any(s != "SEN2" for s in sensors)
    needs_target = any(s == "SEN2" for s in sensors)
    src = list(_find(cube, bands, _source_names(), SOURCE_ROLES, "the six bands of Landsat or MODIS").values()) \
        if needs_source else [0] * len(SOURCE_ROLES)
    dst = list(_find(cube, bands, _TARGET_NAMES, TARGET_BANDS, "the ten bands of Sentinel-2").values()) \
        if needs_target else [0] * len(TARGET_BANDS)
    factor = _scale_of(cube, scale)

    work = _missing_as_nan(cube, nodata)
    dtype = np.float32
    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = work.transpose(*order)
    arr = work.data if work.chunks is not None else work.values
    if not has_time:
        arr = arr[None]
    threads = resolve(n_jobs)
    if work.chunks is not None:
        import dask.array as da

        arr = arr.rechunk({1: -1})
        chunks_out = (arr.chunks[0], (len(TARGET_BANDS),)) + arr.chunks[2:]
        out = da.map_blocks(_adjust_block, arr, sensors, src, dst, factor, threads, dtype, force_search, dtype=dtype,
                            chunks=chunks_out, meta=np.empty((0, 0, 0, 0), dtype))
    else:
        out = _adjust_block(np.asarray(arr), sensors, src, dst, factor, threads, dtype, force_search)
    if not has_time:
        out = out[0]

    coords = {k: v for k, v in cube.coords.items() if "band" not in v.dims}
    coords["band"] = list(TARGET_BANDS)
    result = xr.DataArray(out, dims=order, coords=coords, name=cube.name).transpose(*cube.dims)
    result.attrs = dict(cube.attrs)
    counts = {s: sensors.count(s) for s in dict.fromkeys(sensors)}
    done = [f"{s} -> Sentinel-2A ({n} date{'s' if n != 1 else ''})" if s != "SEN2"
            else f"Sentinel-2 passed through ({n} date{'s' if n != 1 else ''})" for s, n in counts.items()]
    result.attrs["spectral_adjusted"] = ("; ".join(done) + ": FORCE's spectral adjustment, Scheffler, Frantz and "
                                         "Segl (2020), RSE 241:111723")
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    if cube.rio.nodata is not None and not np.issubdtype(cube.dtype, np.integer):
        result = result.rio.write_nodata(cube.rio.nodata, encoded=False)
    return result
