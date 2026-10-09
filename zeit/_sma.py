"""Spectral mixture analysis and the NDFI: the fractions of green vegetation, non-photosynthetic
vegetation, soil, shade and cloud in each pixel, and the Normalized Difference Fraction Index of
Souza et al. (2005), the index of forest degradation in the Amazon.

The unmixing is fully constrained least squares (fractions non-negative and adding up to one,
Heinz & Chang 2001) in C++, every pixel and date in parallel, lazily on dask cubes.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

#: Endmembers of Souza et al. (2005) for Landsat surface reflectance (blue, green, red, NIR,
#: SWIR1, SWIR2), with the cloud endmember of CODED (Bullock et al. 2020), as CODED uses them
#: (its cdd_params.yaml; reflectance x 10000).
SOUZA_2005 = {
    "gv": [500, 900, 400, 6100, 3000, 1000],
    "shade": [0, 0, 0, 0, 0, 0],
    "npv": [1400, 1700, 2200, 3000, 5500, 3000],
    "soil": [2000, 3000, 3400, 5800, 6000, 5800],
    "cloud": [9000, 9600, 8000, 7800, 7200, 6500],
}
LANDSAT_ROLES = ["blue", "green", "red", "nir", "swir1", "swir2"]
ENDMEMBERS = {"souza2005": (SOUZA_2005, LANDSAT_ROLES, 10000.0)}


def endmember_table(endmembers: Any = "souza2005", bands: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """The endmembers as a DataFrame (rows: endmembers, columns: band roles or names), in
    reflectance (0-1)."""
    if isinstance(endmembers, str):
        if endmembers.lower() not in ENDMEMBERS:
            raise ValueError(f"unknown endmembers {endmembers!r}; known: {sorted(ENDMEMBERS)}, or give a DataFrame")
        table, roles, scale = ENDMEMBERS[endmembers.lower()]
        return pd.DataFrame(table, index=roles).T / scale
    if isinstance(endmembers, pd.DataFrame):
        df = endmembers.astype(float)
    elif isinstance(endmembers, dict):
        first = next(iter(endmembers.values()))
        if isinstance(first, dict):
            df = pd.DataFrame(endmembers).T.astype(float)
        else:
            if bands is None:
                raise ValueError("endmembers as lists need bands= (the band of each value)")
            df = pd.DataFrame(endmembers, index=list(bands)).T.astype(float)
    else:
        raise TypeError("endmembers: 'souza2005', a DataFrame (endmember x band) or a dict")
    if df.to_numpy().max() > 2:   # given x 10000, like the data often is
        df = df / 10000.0
    return df


def _resolve(cube: xr.DataArray, columns: Sequence[str], bands: Optional[Sequence[str]]) -> List[Any]:
    """The cube's band for each column of the endmember table."""
    from .indices import BAND_CANDIDATES, resolve_band

    available = [str(b) for b in cube.band.values]
    if bands is not None:
        if len(bands) != len(columns):
            raise ValueError(f"bands= needs one band per endmember column {list(columns)}")
        missing = [b for b in bands if str(b) not in available]
        if missing:
            raise ValueError(f"bands {missing} are not in the cube {available}")
        return [cube.band.values[available.index(str(b))] for b in bands]
    out = []
    for column in columns:
        if column in available:
            out.append(cube.band.values[available.index(column)])
        else:
            try:
                if column not in BAND_CANDIDATES:
                    raise ValueError(column)
                out.append(cube.band.values[available.index(resolve_band(column, available))])
            except ValueError:
                raise ValueError(f"no band {column!r} in the cube {available}; pass bands= (the cube's band "
                                 f"for each of {list(columns)})") from None
    return out


def _as_cube(data: Any, chunks: Any) -> xr.DataArray:
    from ._load import load_raster

    if not isinstance(data, xr.DataArray):
        data = load_raster(data, chunks=chunks)
    if "band" not in data.dims:
        raise ValueError("unmixing needs a cube with a band dimension (time, band, y, x) or (band, y, x)")
    return data


def _scale_of(cube: xr.DataArray, scale: Any) -> float:
    if scale != "auto":
        return float(scale)
    if np.issubdtype(cube.dtype, np.integer):
        return 10000.0
    sample = cube.isel({d: slice(None, None, max(1, cube.sizes[d] // 16)) for d in cube.dims if d != "band"})
    values = np.asarray(sample.values, dtype=float)
    high = np.nanpercentile(values, 99) if np.isfinite(values).any() else 1.0
    return 10000.0 if high > 2.0 else 1.0


def _missing(cube: xr.DataArray, nodata: Any) -> xr.DataArray:
    from ._series_api import _missing_as_nan
    return _missing_as_nan(cube, nodata)


def unmix(
    data: Any,
    endmembers: Any = "souza2005",
    *,
    bands: Optional[Sequence[str]] = None,
    sum_to_one: bool = True,
    nonneg: bool = True,
    scale: Union[str, float] = "auto",
    ndfi: bool = True,
    cloud_threshold: Optional[float] = None,
    nodata: Union[float, str, None] = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.Dataset:
    """Spectral mixture analysis: the fraction of each endmember in every pixel and date.

    Parameters
    ----------
    data
        A reflectance cube ``(time, band, y, x)`` or image ``(band, y, x)``, in memory or
        dask, or anything ``zeit.load_raster`` reads.
    endmembers
        ``"souza2005"`` (green vegetation, shade, non-photosynthetic vegetation, soil and
        cloud for Landsat's blue, green, red, NIR, SWIR1 and SWIR2: Souza et al. 2005, with
        CODED's cloud), or a DataFrame (rows: endmembers, columns: bands of the cube) or a
        dict of endmember -> {band: value}, in reflectance (0-1, or x 10000).
    bands
        The cube's band for each endmember column, when the names differ (by default found
        by name: ``nir``, ``B08``, ``SR_B5``...).
    sum_to_one, nonneg
        Fractions adding up to one, and non-negative (both by default: fully constrained).
    scale
        What the data's reflectance is multiplied by: ``"auto"`` (10000 for integers or
        values above 2, else 1), or a number.
    ndfi
        Also compute the NDFI, when the endmembers include ``gv``, ``npv``, ``soil`` and
        ``shade``: ``(GVs - (NPV + soil)) / (GVs + NPV + soil)``, with
        ``GVs = GV / (1 - shade)`` (Souza et al. 2005), from -1 to 1.
    cloud_threshold
        Mask (NaN) observations whose ``cloud`` fraction is above this (CODED uses 0.05).
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without
        one), a number or ``None``.
    chunks
        Inputs read from disk: ``None`` in memory, ``"auto"`` or a dict lazily.
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        One variable per endmember (its fraction), ``rmse`` (of the rebuilt spectrum, in
        reflectance) and ``ndfi``, with the cube's dimensions but ``band``, its coordinates
        and CRS; lazy if the cube is.

    Examples
    --------
    >>> fractions = zeit.unmix(landsat)            # gv, shade, npv, soil, cloud, rmse, ndfi
    >>> fractions.ndfi.zeit.plot()
    """
    from ._embeddings import refuse

    refuse(data, "zeit.unmix")
    from ._core.sma import unmix_batch

    cube = _as_cube(data, chunks)
    table = endmember_table(endmembers, bands)
    selected = _resolve(cube, list(table.columns), bands if not isinstance(endmembers, str) else bands)
    factor = _scale_of(cube, scale)
    cube = _missing(cube.sel(band=selected), nodata)
    names = [str(n) for n in table.index]
    E = np.ascontiguousarray(table.to_numpy(dtype=float))
    k = len(names)

    def _unmix(block):
        shape = block.shape[:-1]
        flat = np.ascontiguousarray(block.reshape(-1, block.shape[-1]), dtype=np.float64) / factor
        out = unmix_batch(flat, E, sum_to_one=sum_to_one, nonneg=nonneg, n_jobs=n_jobs)
        return out.reshape(shape + (k + 1,)).astype(np.float32)

    if cube.chunks is not None:
        cube = cube.chunk({"band": -1})
    fractions = xr.apply_ufunc(_unmix, cube, input_core_dims=[["band"]], output_core_dims=[["fraction"]],
                               dask="parallelized", output_dtypes=[np.float32],
                               dask_gufunc_kwargs={"output_sizes": {"fraction": k + 1}})
    fractions = fractions.assign_coords(fraction=names + ["rmse"])
    dims = [d for d in cube.dims if d != "band"]
    ds = xr.Dataset({name: fractions.sel(fraction=name, drop=True).transpose(*dims) for name in names + ["rmse"]})
    if ndfi and {"gv", "npv", "soil", "shade"} <= set(names):
        ds["ndfi"] = ndfi_of(ds)
    if cloud_threshold is not None:
        if "cloud" not in ds:
            raise ValueError("cloud_threshold needs a 'cloud' endmember")
        clear = ds["cloud"] <= cloud_threshold
        ds = ds.map(lambda v: v.where(clear))
    for coord in ("spatial_ref",):
        if coord in cube.coords and coord not in ds.coords:
            ds = ds.assign_coords({coord: cube.coords[coord]})
    if cube.rio.crs is not None and "x" in ds.dims:
        ds = ds.rio.write_crs(cube.rio.crs)
    ds.attrs.update(endmembers=endmembers if isinstance(endmembers, str) else "custom", scale=factor,
                    sum_to_one=int(sum_to_one), nonneg=int(nonneg))
    return ds


def ndfi_of(fractions: Any) -> Any:
    """The NDFI of fractions (``gv``, ``npv``, ``soil``, ``shade``: a Dataset or a dict of
    arrays): ``(GVs - (NPV + soil)) / (GVs + NPV + soil)``, ``GVs = GV / (1 - shade)``."""
    gv, npv, soil, shade = (fractions[k] for k in ("gv", "npv", "soil", "shade"))
    with np.errstate(divide="ignore", invalid="ignore"):
        gvs = gv / (1 - shade)
        out = (gvs - (npv + soil)) / (gvs + npv + soil)
    if isinstance(out, xr.DataArray):
        return out.where(np.isfinite(out)).astype(np.float32).rename("ndfi")
    out = np.asarray(out, dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def ndfi_from_bands(arrays: Sequence[Any]) -> Any:
    """NDFI from Landsat reflectance (0-1) in blue, green, red, NIR, SWIR1, SWIR2 order, with
    the Souza et al. (2005) endmembers: NumPy arrays or DataArrays of the same shape."""
    from ._core.sma import unmix_batch

    table = endmember_table("souza2005")
    E = np.ascontiguousarray(table.to_numpy(dtype=float))
    names = list(table.index)

    def _ndfi(*bands):
        stack = np.stack([np.asarray(b, dtype=np.float64) for b in bands], axis=-1)
        out = unmix_batch(stack.reshape(-1, stack.shape[-1]), E)
        fr = {n: out[:, i].reshape(stack.shape[:-1]) for i, n in enumerate(names)}
        return ndfi_of(fr)

    if isinstance(arrays[0], xr.DataArray):
        return xr.apply_ufunc(_ndfi, *arrays, dask="parallelized", output_dtypes=[np.float32])
    return _ndfi(*arrays)
