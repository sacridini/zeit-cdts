"""Tasseled Cap brightness, greenness and wetness: ``zeit.tasseled_cap``.

The Tasseled Cap is a fixed linear map of the six reflective bands (blue, green, red, NIR, SWIR1, SWIR2) onto three
axes of the soil-vegetation-water cloud of a scene: brightness (soil, built-up), greenness (green vegetation) and
wetness (moisture, shade). FORCE's TSA (3.11, David Frantz, GPL-3.0-or-later) applies the matrix of Crist (1985) to
every sensor; the per-sensor matrices of Huang et al. (2002) and Baig et al. (2014) are here too.
The arithmetic is FORCE's (float32, the sum of the six products in the band order), so that the values truncated to
integers are FORCE's.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import xarray as xr

#: The six input bands, in the order of the coefficients, and their names as ``zeit.spectral_adjust`` finds them.
ROLES = ("blue", "green", "red", "nir", "swir1", "swir2")

#: The components of the published matrices, and the one derived from them (``disturbance``).
COMPONENTS = ("brightness", "greenness", "wetness")
DERIVED = ("disturbance",)

# Each table: component -> coefficients of blue, green, red, NIR, SWIR1, SWIR2.
#: Crist (1985, RSE 17:301-306), "A TM Tasseled Cap equivalent transformation for reflectance factor data", Landsat
#: 4-5 TM surface reflectance. It is the matrix FORCE applies to every sensor.
CRIST_1985 = {
    "brightness": (0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303),
    "greenness": (-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446),
    "wetness": (0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109),
}
#: Huang, Wylie, Yang, Homer and Zylstra (2002), IJRS 23(8):1741-1748, Landsat 7 ETM+ at-satellite reflectance.
HUANG_2002 = {
    "brightness": (0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596),
    "greenness": (-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630),
    "wetness": (0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388),
}
#: Baig, Zhang, Shuai and Tong (2014), Remote Sensing Letters 5(5):423-431, Landsat 8 OLI top-of-atmosphere
#: reflectance (used for Landsat 9's OLI-2 too).
BAIG_2014 = {
    "brightness": (0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872),
    "greenness": (-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608),
    "wetness": (0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559),
}
TABLES: Dict[str, Dict[str, Sequence[float]]] = {
    "crist1985": CRIST_1985, "huang2002": HUANG_2002, "baig2014": BAIG_2014,
}
#: ``coefficients="sensor"``: the table of each sensor (FORCE's names, ``"SEN2"`` any Sentinel-2 unit). Sentinel-2 takes
#: Crist (1985), as in FORCE: the matrix published for MSI (Shi and Xu 2019) is for all 13 bands of
#: the L1C top-of-atmosphere product, B1, B9 and B10 included, which an L2A cube does not have.
SENSOR_TABLE = {"LND04": "crist1985", "LND05": "crist1985", "LND07": "huang2002", "LND08": "baig2014",
                "LND09": "baig2014", "SEN2": "crist1985"}


def _matrices(coefficients: Any, sensors: Optional[List[str]], n: int, names: List[str],
              given: List[str]) -> np.ndarray:
    """``(date, row, 6)`` float32: the coefficients of each date for the rows ``names`` (the components asked for and
    those the disturbance is made of); the rows of an array are named by `given`, the components asked for."""
    if isinstance(coefficients, str):
        key = coefficients.lower()
        if key == "sensor":
            out = np.empty((n, len(names), len(ROLES)), dtype=np.float32)
            for i in range(n):
                table = SENSOR_TABLE.get(sensors[i])
                if table is None:
                    raise ValueError(f"no published Tasseled Cap for the sensor {sensors[i]!r}: those of Landsat 4, 5, "
                                     f"7, 8, 9 and Sentinel-2 are known; pass coefficients= a table or a dict")
                out[i] = [TABLES[table][c] for c in names]
            return out
        if key not in TABLES:
            raise ValueError(f"coefficients= {coefficients!r}: use {', '.join(map(repr, TABLES))}, 'sensor', a dict "
                             f"{{component: six coefficients}} or an array (components, 6)")
        table = TABLES[key]
    elif isinstance(coefficients, dict):
        table = {str(k): v for k, v in coefficients.items()}
    else:
        arr = np.asarray(coefficients, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(ROLES):
            raise ValueError(f"coefficients= an array must be (components, {len(ROLES)}), got {arr.shape}")
        if len(given) != arr.shape[0]:
            raise ValueError(f"coefficients= has {arr.shape[0]} rows for the components {given}: name them with "
                             f"components= (or pass a dict)")
        table = {name: arr[i] for i, name in enumerate(given)}
    missing = [c for c in names if c not in table]
    if missing:
        raise ValueError(f"coefficients= has no {missing}; it has {list(table)}")
    rows = np.empty((len(names), len(ROLES)), dtype=np.float32)
    for k, c in enumerate(names):
        values = np.asarray(table[c], dtype=float)
        if values.shape != (len(ROLES),):
            raise ValueError(f"coefficients= for {c!r} must be {len(ROLES)} numbers (blue, green, red, NIR, SWIR1, "
                             f"SWIR2), got {values.shape}")
        rows[k] = values
    return np.broadcast_to(rows, (n,) + rows.shape)


def _missing_values(cube: xr.DataArray, nodata: Any) -> List[float]:
    """The values that mean "missing" besides NaN: the raster's NoData (``"auto"``; for integers without one, 0, as
    Earth Engine exports masked pixels), a number, or none."""
    if isinstance(nodata, str) and nodata != "auto":
        raise ValueError(f"nodata must be 'auto', a number or None, got {nodata!r}")
    if nodata == "auto":
        stored = cube.rio.nodata
        if stored is not None and not np.isnan(stored):
            return [stored]
        return [0] if np.issubdtype(cube.dtype, np.integer) else []
    return [] if nodata is None else [nodata]


def _tile(block: np.ndarray, out: np.ndarray, k: int, r0: int, r1: int, src: Sequence[int], m: np.ndarray,
          rows: Sequence[int], disturbance: Optional[Sequence[int]], missing: Sequence[float]) -> None:
    """The components of date `k`, rows `r0`:`r1` of the image, into `out`."""
    make = np.array if missing else np.asarray      # a copy to mask, none otherwise
    x = [make(block[k, b, r0:r1], dtype=np.float32) for b in src]
    for value in missing:
        for band in x:
            band[band == np.float32(value)] = np.nan
    tmp = np.empty_like(x[0])
    done = {}
    for r in sorted(set(rows) | set(disturbance or ())):
        acc = np.multiply(m[r, 0], x[0], dtype=np.float32)           # FORCE's sum, in its order, in float32
        for j in range(1, len(ROLES)):
            np.multiply(m[r, j], x[j], out=tmp)
            acc += tmp
        done[r] = acc
    for i, r in enumerate(rows):
        out[k, i, r0:r1] = done[r]
    if disturbance is not None:
        b, g, w = disturbance
        np.subtract(done[b], done[g], out=tmp)
        out[k, len(rows), r0:r1] = tmp - done[w]


_KERNEL_DTYPES = tuple(np.dtype(t) for t in (np.int16, np.uint16, np.int32, np.float32, np.float64))


def _block(block: np.ndarray, src: Sequence[int], matrices: np.ndarray, rows: Sequence[int],
           disturbance: Optional[Sequence[int]], missing: Sequence[float] = (), threads: int = 1,
           block_info: Any = None) -> np.ndarray:
    """``(time, band, y, x)`` -> ``(time, component, y, x)`` float32. `matrices` is ``(date, row, 6)`` for all the
    dates of the cube; `rows` the rows to return, and `disturbance` (the rows of brightness, greenness and wetness)
    adds ``B - G - W`` last. Done by the compiled core on `threads` threads, with the arithmetic of ``_block_numpy``
    bit for bit."""
    from . import _core

    first = block_info[0]["array-location"][0][0] if block_info else 0
    block = np.asarray(block)
    if block.dtype not in _KERNEL_DTYPES:
        block = block.astype(np.float32)
    block = np.ascontiguousarray(block)
    m = np.ascontiguousarray(matrices[first:first + block.shape[0]], dtype=np.float32)
    return _core.tasseled.tasseled_batch(block, [int(b) for b in src], m, [int(r) for r in rows],
                                         None if disturbance is None else [int(r) for r in disturbance],
                                         [float(v) for v in missing], int(threads))


def _block_numpy(block: np.ndarray, src: Sequence[int], matrices: np.ndarray, rows: Sequence[int],
                 disturbance: Optional[Sequence[int]], missing: Sequence[float] = (), threads: int = 1,
                 block_info: Any = None) -> np.ndarray:
    """What ``_block`` computes, in numpy: the image is done in tiles of rows that stay in cache, on `threads`
    threads (numpy releases the GIL). The reference of the tests."""
    first = block_info[0]["array-location"][0][0] if block_info else 0
    nt, _, ny, nx = block.shape
    out = np.empty((nt, len(rows) + int(disturbance is not None), ny, nx), dtype=np.float32)
    step = max(1, 65536 // max(nx, 1))
    jobs = [(k, r0, min(r0 + step, ny)) for k in range(nt) for r0 in range(0, ny, step)]
    run = lambda j: _tile(block, out, j[0], j[1], j[2], src, matrices[first + j[0]], rows, disturbance, missing)
    if threads > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(min(threads, len(jobs))) as pool:
            list(pool.map(run, jobs))
    else:
        for j in jobs:
            run(j)
    return out


def tasseled_cap(
    data: Any,
    components: Union[str, Sequence[str]] = ("brightness", "greenness", "wetness"),
    *,
    coefficients: Any = "crist1985",
    sensor: Any = None,
    bands: Optional[Dict[str, Any]] = None,
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.DataArray:
    """Tasseled Cap brightness, greenness and wetness of a reflectance cube, in the scale of the data.

    Each component is a weighted sum of the six reflective bands (blue, green, red, NIR, SWIR1, SWIR2), computed in
    float32 as FORCE's ``TC-BRIGHT``, ``TC-GREEN``, ``TC-WET`` and ``TC-DI`` do (the six products added in the band
    order; ``index-compute-hl.c`` of FORCE 3.11, David Frantz, GPL-3.0-or-later): FORCE's values truncated to
    integers are those of this function applied to the same integers. The matrix is linear and has no intercept, so
    the result is in the units of the input (reflectance times 10000 gives components times 10000).

    ``"disturbance"`` is FORCE's ``TC-DI``: ``brightness - greenness - wetness`` of the same matrix, as it comes, not
    the standardized disturbance index of Healey et al. (2005), which scales each component by the mean and
    standard deviation of the forest of the scene.

    Parameters
    ----------
    data
        Reflectance ``(time, band, y, x)`` or ``(band, y, x)``, in memory or dask, or anything
        ``zeit.load_raster`` reads, with the bands blue, green, red, NIR, SWIR1 and SWIR2 (``blue``/``B02``/
        ``SR_B2``... as for ``spectral_adjust``). Integers are accepted (the result is float32 in their scale).
        For Sentinel-2 the NIR is B8A (``nir08``), which is the band of FORCE's ``NIR`` for Sentinel-2 (its broad NIR
        is ``BROADNIR``, not used here); a cube that has B08 only (``nir`` on Earth Search) is used with a warning,
        or say ``bands={"nir": ...}``.
    components
        What to compute, in this order: ``"brightness"``, ``"greenness"``, ``"wetness"`` (default, as in FORCE's
        ``TC-BRIGHT TC-GREEN TC-WET``), ``"disturbance"``, or a name of the dict of ``coefficients``.
    coefficients
        ``"crist1985"`` (default): the matrix of Crist (1985) for Landsat 4-5 TM reflectance factors, which FORCE
        applies to every sensor, Landsat 4-9 and Sentinel-2 alike. ``"sensor"``: the matrix published for each
        date's sensor: TM (Landsat 4, 5) Crist (1985); ETM+ (Landsat 7) Huang et al. (2002); OLI (Landsat 8, and
        OLI-2 of Landsat 9) Baig et al. (2014); MSI (Sentinel-2) Crist (1985), as FORCE, since the published MSI
        matrix (Shi and Xu 2019) takes the 13 bands of L1C top-of-atmosphere reflectance, B1, B9 and
        B10 included, which L2A does not have. The matrices are also named: ``"huang2002"``, ``"baig2014"`` (for
        every date). Or your own: a
        dict ``{component: six numbers (blue, green, red, NIR, SWIR1, SWIR2)}``, or an array ``(components, 6)``
        whose rows are named by ``components``. Those of Huang and Baig were derived for at-satellite
        (top-of-atmosphere) reflectance; applying them to surface reflectance is common practice but approximate,
        and only Crist's matrix was derived for reflectance factors of the surface.
    sensor
        Each date's sensor for ``coefficients="sensor"``: by default the cube's ``platform`` coordinate (which
        ``build_time_series`` keeps from STAC) or attribute; or a name for every date (``"landsat-8"``, ``"LC09"``,
        ``"LND05"``, ``"sentinel-2a"``...), or one per date. Other sensors (MODIS...) are an error naming the date.
    bands
        The cube's band of each of the six roles (``blue``, ``green``, ``red``, ``nir``, ``swir1``, ``swir2``) when
        its names are not the usual ones.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or
        ``None``. A pixel with a band missing is missing in every component (FORCE never reads a nodata value: a mask
        decides which pixels are computed, and its output is truncated to 16 bits).
    chunks
        Inputs read from disk: ``None`` in memory, ``"auto"`` or a dict lazily.
    n_jobs
        Threads for an in-memory cube: the processors zeit may use by default (``-1``). A dask cube is parallel by
        its chunks.

    Returns
    -------
    xarray.DataArray
        ``(time, band, y, x)`` (or ``(band, y, x)``), float32, whose ``band`` coordinate names the components, with the
        coordinates and georeferencing of the input; lazy if the input is. ``attrs["tasseled_cap"]`` says which
        matrix each sensor got. Pick one for ``zeit.landtrendr``'s ``ftv`` or ``zeit.ewmacd`` with
        ``.sel(band="wetness")``.

    Examples
    --------
    >>> tc = zeit.tasseled_cap(landsat)                          # brightness, greenness, wetness (Crist 1985)
    >>> tc = zeit.tasseled_cap(landsat, ["wetness", "disturbance"], coefficients="sensor")
    >>> zeit.landtrendr(tc.sel(band="wetness"))
    """
    from ._embeddings import refuse
    from ._load import load_raster
    from ._specadjust import _find, _sensors, _source_names
    from ._threads import resolve

    refuse(data, "zeit.tasseled_cap")
    cube = data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)
    if "band" not in cube.dims:
        raise ValueError("tasseled_cap needs a cube with a band dimension, (time, band, y, x) or (band, y, x)")
    if not {"y", "x"} <= set(cube.dims):
        raise ValueError("tasseled_cap needs a cube with y and x dimensions")
    if cube.attrs.get("tasseled_cap"):
        raise ValueError(f"the cube is already a Tasseled Cap ({cube.attrs['tasseled_cap']})")
    if "scale" in cube.coords and np.issubdtype(cube.dtype, np.integer):
        raise ValueError("tasseled_cap needs reflectance: the cube holds digital numbers with per-scene scale and "
                         "offset; build it with a float dtype (build_time_series' default)")
    unknown = [k for k in (bands or {}) if k not in ROLES]
    if unknown:
        raise ValueError(f"bands=: unknown role {unknown}; roles are {list(ROLES)}")
    names = [components] if isinstance(components, str) else [str(c) for c in components]
    if not names:
        raise ValueError("components= is empty")
    if len(set(names)) != len(names):
        raise ValueError(f"components= names a component twice: {names}")
    published = [c for c in names if c not in DERIVED]
    disturbance = "disturbance" in names
    # the matrix rows: the components asked for, then those of brightness, greenness, wetness that the disturbance needs
    rows_needed = published + ([c for c in COMPONENTS if c not in published] if disturbance else [])
    rows = list(range(len(published)))
    base = tuple(rows_needed.index(c) for c in COMPONENTS) if disturbance else None
    has_time = "time" in cube.dims
    n = cube.sizes["time"] if has_time else 1

    sensors = None
    if isinstance(coefficients, str) and coefficients.lower() == "sensor":
        try:
            sensors = _sensors(cube, sensor)
        except ValueError as error:
            raise ValueError(str(error).replace("spectral_adjust", "tasseled_cap")) from None
    else:
        try:
            sensors = _sensors(cube, sensor)
        except ValueError:
            sensors = None
    matrices = _matrices(coefficients, sensors, n, rows_needed, published)
    src = list(_find(cube, bands, _source_names(), ROLES, "the six bands of the Tasseled Cap").values())
    nir_name = str(cube.band.values[src[3]])
    if (sensors is not None and any(s == "SEN2" for s in sensors) and nir_name.lower() in ("nir", "b08")
            and not (bands and "nir" in bands)):
        warnings.warn(f"band {nir_name!r} of the Sentinel-2 dates is the broad NIR (B08); FORCE's Tasseled Cap uses "
                      "its NIR, B8A (nir08). Load B8A, or pass bands={'nir': ...} to say this band is it.",
                      stacklevel=2)

    missing = _missing_values(cube, nodata)
    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = cube.transpose(*order)
    arr = work.data if work.chunks is not None else work.values
    if not has_time:
        arr = arr[None]
    out_names = list(published) + (["disturbance"] if disturbance else [])
    if work.chunks is not None:
        import dask.array as da

        arr = arr.rechunk({1: -1})
        chunks_out = (arr.chunks[0], (len(out_names),)) + arr.chunks[2:]
        out = da.map_blocks(_block, arr, src=src, matrices=matrices, rows=rows, disturbance=base, missing=missing,
                            dtype=np.float32, chunks=chunks_out, meta=np.empty((0, 0, 0, 0), np.float32))
    else:
        out = _block(np.asarray(arr), src, matrices, rows, base, missing, resolve(n_jobs))
    if out_names != names:      # the disturbance comes last in the block; it may have been asked for earlier
        out = out[:, [out_names.index(c) for c in names]]
    if not has_time:
        out = out[0]
    coords = {k: v for k, v in cube.coords.items() if "band" not in v.dims}
    coords["band"] = names
    result = xr.DataArray(out, dims=order, coords=coords, name=cube.name).transpose(*cube.dims)
    result.attrs = dict(cube.attrs)
    result.attrs["tasseled_cap"] = _describe(coefficients, sensors)
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    return result


def _describe(coefficients: Any, sensors: Optional[List[str]]) -> str:
    if not isinstance(coefficients, str):
        return "custom coefficients"
    if coefficients.lower() != "sensor":
        return f"{coefficients.lower()}: {_CITES[coefficients.lower()]}"
    used = {}
    for s in sensors or []:
        used[s] = used.get(s, 0) + 1
    return "; ".join(f"{s} ({k} date{'s' if k != 1 else ''}): {SENSOR_TABLE[s]}, {_CITES[SENSOR_TABLE[s]]}"
                     for s, k in used.items())


_CITES = {
    "crist1985": "Crist (1985), RSE 17:301-306",
    "huang2002": "Huang et al. (2002), IJRS 23:1741-1748",
    "baig2014": "Baig et al. (2014), Remote Sensing Letters 5:423-431",
}
