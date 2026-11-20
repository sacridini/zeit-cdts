"""What the bi-temporal change tools share (``zeit.cva``, ``zeit.imad``, ``zeit.radcal``): the two dates as arrays
of bands, the missing values, and the deterministic lattice sample the global statistics are computed from."""

import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

#: The dtypes the compiled kernels read as they are; anything else is converted to float32.
KERNEL_DTYPES = tuple(np.dtype(t) for t in (np.int16, np.uint16, np.int32, np.float32, np.float64))


def to_cube(data: Any, who: str, chunks: Any = None) -> xr.DataArray:
    """`data` as a DataArray: itself, or what ``zeit.load_raster`` reads."""
    from ._embeddings import refuse
    from ._load import load_raster

    refuse(data, who)
    return data if isinstance(data, xr.DataArray) else load_raster(data, chunks=chunks)


def _pick_date(cube: xr.DataArray, which: Any, who: str) -> xr.DataArray:
    """One date of a cube with a time dimension: an index (negative from the end), a label of the time
    coordinate or a date string (the nearest date)."""
    n = cube.sizes["time"]
    if isinstance(which, (int, np.integer)) and not isinstance(which, bool):
        if not -n <= which < n:
            raise ValueError(f"{who}: date {which} is out of the {n} dates of the cube")
        return cube.isel(time=int(which))
    try:
        return cube.sel(time=which, method="nearest")
    except Exception as error:
        raise ValueError(f"{who}: cannot find the date {which!r} in the cube's time coordinate: {error}") from None


def _as_bands(cube: xr.DataArray, who: str) -> xr.DataArray:
    if "time" in cube.dims:
        if cube.sizes["time"] != 1:
            raise ValueError(f"{who}: a cube with {cube.sizes['time']} dates needs dates= to choose two (or give the "
                             "two dates as separate arrays)")
        cube = cube.isel(time=0)
    if "band" not in cube.dims:
        cube = cube.expand_dims(band=[0])
    if not {"y", "x"} <= set(cube.dims):
        raise ValueError(f"{who} needs cubes with y and x dimensions")
    return cube.transpose("band", "y", "x")


def two_dates(data: Any, after: Any, dates: Sequence[Any], bands: Optional[Sequence[Any]], who: str,
              chunks: Any = None) -> Tuple[xr.DataArray, xr.DataArray]:
    """The dates before and after as ``(band, y, x)`` DataArrays with the same bands, from a cube
    ``(time, band, y, x)`` and `dates` (two indices, labels or date strings), or from two arrays."""
    cube = to_cube(data, who, chunks)
    if after is not None:
        second = to_cube(after, who, chunks)
        first, second = _as_bands(cube, who), _as_bands(second, who)
    else:
        if "time" not in cube.dims:
            raise ValueError(f"{who}: give the second date as after=, or a cube with a time dimension")
        if len(dates) != 2:
            raise ValueError(f"{who}: dates= must be two dates (before, after), got {dates!r}")
        first = _as_bands(_pick_date(cube, dates[0], who), who)
        second = _as_bands(_pick_date(cube, dates[1], who), who)
    if first.sizes != second.sizes:
        raise ValueError(f"{who}: the two dates have different shapes {dict(first.sizes)} and {dict(second.sizes)}: "
                         "resample one onto the other's grid first")
    if bands is not None:
        picks = list(bands) if not isinstance(bands, (str, int)) else [bands]
        names = [str(b) for b in first.band.values]
        index = []
        for b in picks:
            if isinstance(b, (int, np.integer)) and not isinstance(b, bool) and str(b) not in names:
                index.append(int(b))
            elif str(b) in names:
                index.append(names.index(str(b)))
            else:
                raise ValueError(f"{who}: bands= {b!r} is not a band of the dates {names}")
        first, second = first.isel(band=index), second.isel(band=index)
        if "band" in second.coords and "band" in first.coords:
            pass
    if first.sizes["band"] != second.sizes["band"]:
        raise ValueError(f"{who}: the two dates have different numbers of bands")
    second = second.assign_coords(band=first.band.values)
    if second.chunks is None and first.chunks is not None:
        second = second.chunk(first.chunks)
    return first, second


def missing_values(cube: xr.DataArray, nodata: Any) -> List[float]:
    """The values that mean "missing" besides NaN: the raster's NoData (``"auto"``; for integers without one, 0, as
    Earth Engine exports masked pixels), a number, or none."""
    from ._tasseled import _missing_values

    return [float(v) for v in _missing_values(cube, nodata)]


def kernel_array(a: Any) -> np.ndarray:
    """`a` as a C-contiguous array the kernels read (float32 for an unsupported dtype)."""
    a = np.asarray(a)
    if a.dtype not in KERNEL_DTYPES:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


def lattice_stride(ny: int, nx: int, sample: Optional[int]) -> int:
    """The stride of the lattice sample: every ``s``-th row and column, ``s = ceil(sqrt(pixels / sample))``."""
    if sample is None:
        return 1
    if int(sample) < 1:
        raise ValueError(f"sample= must be a positive number of pixels or None, got {sample!r}")
    return max(1, math.ceil(math.sqrt(ny * nx / int(sample))))


def lattice(cube: xr.DataArray, stride: int) -> np.ndarray:
    """The lattice sample of ``(band, y, x)`` as an array (computed once if lazy)."""
    sub = cube.data[:, ::stride, ::stride] if stride > 1 else cube.data
    if hasattr(sub, "compute"):
        sub = sub.compute()
    return np.asarray(sub)


def valid_rows(x: np.ndarray, y: np.ndarray, missing: Sequence[float]) -> np.ndarray:
    """The valid pixels of two samples ``(band, ny, nx)`` as float64 rows ``(pixels, 2 bands)``: every band of both
    dates finite and none of the missing values."""
    n = x.shape[0]
    z = np.concatenate([x.reshape(n, -1), y.reshape(n, -1)]).astype(np.float64).T
    ok = np.isfinite(z).all(axis=1)
    for m in missing:
        ok &= (z != m).all(axis=1)
    return np.ascontiguousarray(z[ok])


def jsonable(value: Any) -> Any:
    """`value` with numpy types and arrays turned to plain Python, for ``json.dumps``."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    return value


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value))


def result_array(values: Any, names: Sequence[str], template: xr.DataArray, name: str, attrs: Dict[str, Any],
                 key: str) -> xr.DataArray:
    """A ``(band, y, x)`` DataArray of the planes `values` with the coordinates and georeferencing of `template`
    (``(band, y, x)``), `attrs[key]` the JSON record."""
    coords = {k: v for k, v in template.coords.items() if "band" not in v.dims}
    coords["band"] = list(names)
    out = xr.DataArray(values, dims=("band", "y", "x"), coords=coords, name=name)
    out.attrs = {k: v for k, v in template.attrs.items() if k not in ("long_name", "scale_factor", "add_offset")}
    out.attrs[key] = dumps(attrs)
    if template.rio.crs is not None:
        out = out.rio.write_crs(template.rio.crs)
    return out
