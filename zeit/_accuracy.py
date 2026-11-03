"""Accuracy assessment and area estimation, after the good practices of Olofsson et al. (2014).

``sampling_design`` sizes and allocates a stratified random sample of a map,
``stratified_sample`` draws it, and ``accuracy`` turns the map and the reference labels of the
sample into an error matrix in proportions of area, overall, user's and producer's
accuracies and error-adjusted areas, each with its confidence interval.

The estimators are Stehman's (2014) for stratified random sampling, so the strata need not be
the classes of the map being assessed: a sample stratified by one map assesses another one
(``accuracy(..., strata=the_first_map)``). When the strata are the map's classes they reduce to
Olofsson et al. (2014), eqs. 4-11, and give the numbers of its example and of R's
``sits_accuracy``. Like them, without a finite population correction (negligible on maps of
thousands of pixels and more).

The areas can also come from model-assisted estimators, which add what the map says everywhere
(its mapped areas, known) to what the sample says: the difference estimator (the mapped area
corrected by the estimated area of the map's errors; Stehman 2013, McRoberts et al. 2018) and
the ratio estimator (the mapped area times the estimated ratio of the reference's area to the
map's; Stehman 2014). They matter when the strata are not the map's classes (with the map's
classes as strata the three agree) and for rare classes and changes, whose intervals they
narrow when the map is good. ``transitions`` assesses the transitions between two dates of a
map of classes (each pair of classes a class: "Forest -> Pasture"), with the same estimators.

Strata and areas come from the map: the area of each pixel from its transform and CRS
(hectares; in a geographic CRS, the area of each row on the sphere). A map of integer classes of up
to 2^26 pixels is read once and counted in memory; larger ones, and maps of other values, lazily
for dask maps.
"""

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import xarray as xr

NO_CHANGE, CHANGE = "no change", "change"
_EARTH_RADIUS = 6371007.181   # authalic radius, metres
ESTIMATORS = ("direct", "difference", "ratio")


# ---------------------------------------------------------------------------
# Strata of a map
# ---------------------------------------------------------------------------

@dataclass
class _Strata:
    codes: xr.DataArray          # (y, x) int32 stratum per pixel, -1 outside every stratum
    names: Dict[int, str]        # code -> name, in the order of the codes
    events: Optional[xr.Dataset]  # the map of events, when the map is one
    source: Any                  # what the strata were made of


def _is_events(data: Any) -> bool:
    return isinstance(data, xr.Dataset) and "yod" in data


def _raw_map(data: Any, nodata: Any) -> Tuple[xr.DataArray, Any]:
    """(the map as it is stored, its nodata value or None when there is none or it is NaN)."""
    from ._load import load_raster

    if isinstance(data, xr.Dataset):
        names = [v for v in data.data_vars if {"y", "x"} <= set(data[v].dims) and data[v].ndim == 2]
        if len(names) != 1:
            raise ValueError(f"the Dataset has the maps {names}; pass one of them (e.g. result.label)")
        data = data[names[0]]
    da = data if isinstance(data, xr.DataArray) else load_raster(data, chunks="auto")
    da = da.squeeze(drop=True) if da.ndim > 2 else da
    if set(da.dims) != {"y", "x"}:
        raise ValueError(f"expected a map with dims (y, x), got {da.dims}")
    da = da.transpose("y", "x")
    if nodata == "auto":
        nodata = da.rio.nodata
    if nodata is not None and isinstance(nodata, float) and np.isnan(nodata):
        nodata = None
    return da, nodata


def _as_map(data: Any, nodata: Any) -> xr.DataArray:
    da, nodata = _raw_map(data, nodata)
    return da if nodata is None else da.where(da != nodata)


# A map of classes of this many pixels or fewer is read once into memory (its codes as int32: 256 MB at most) and
# counted there; a larger one stays lazy.
_IN_MEMORY = 1 << 26
_SLAB = 1 << 22   # pixels counted at a time, to bound the temporary arrays


def _distinct(v: np.ndarray) -> np.ndarray:
    """The distinct values of an integer array, sorted (a table of the values seen when their range is small)."""
    flat = v.reshape(-1)
    if flat.size == 0:
        return np.zeros(0, dtype=np.int64)
    lo, hi = int(flat.min()), int(flat.max())
    if hi - lo > _SLAB:
        return np.unique(flat).astype(np.int64)
    seen = np.zeros(hi - lo + 1, dtype=bool)
    for i in range(0, flat.size, _SLAB):
        seen[np.subtract(flat[i:i + _SLAB], lo, dtype=np.int64)] = True
    return np.flatnonzero(seen) + lo


def _code_block(v: np.ndarray, nodata: Any) -> np.ndarray:
    codes = v.astype(np.int32)
    if nodata is not None:
        codes[v == nodata] = -1
    return codes


def _integer_codes(raw: xr.DataArray, nodata: Any) -> Tuple[xr.DataArray, np.ndarray]:
    """(the codes of a map of integer classes, -1 at nodata; its classes): read once, into memory when it is not
    too large."""
    values = raw.data
    if hasattr(values, "dask") and raw.size <= _IN_MEMORY:
        values = values.compute()
    if hasattr(values, "dask"):
        import dask.array as dask_array
        found = np.asarray(dask_array.unique(values).compute(), dtype=np.int64)
        codes = dask_array.map_blocks(_code_block, values, nodata, dtype=np.int32)
    else:
        values = np.asarray(values)
        found = _distinct(values)
        codes = _code_block(values, nodata)
    if nodata is not None:
        found = found[found != nodata]
    return xr.DataArray(codes, coords=raw.coords, dims=raw.dims), found


def _strata(data: Any, bins: Optional[Sequence[float]] = None, nodata: Any = "auto") -> _Strata:
    """The strata of a map: its classes, intervals of its values (``bins``) or, for a map of
    events, "no change" and "change" (or a stratum per period of ``yod`` with ``bins``)."""
    strata = _strata_of(data, bins, nodata)
    crs = strata.source.rio.crs
    if crs is not None:
        strata.codes = strata.codes.rio.write_crs(crs)
    return strata


def _strata_of(data: Any, bins: Optional[Sequence[float]], nodata: Any) -> _Strata:
    if _is_events(data):
        yod = data["yod"].astype(np.float64)
        yod = yod.where(np.isfinite(yod), 0)
        if bins is None:
            codes = xr.where(yod > 0, 1, 0)
            names = {0: NO_CHANGE, 1: CHANGE}
        else:
            edges = np.asarray(bins, dtype=np.float64)
            codes = xr.where(yod > 0, _digitize(yod, edges), 0)
            names = {0: NO_CHANGE}
            names.update({i + 1: _interval(edges[i], edges[i + 1], years=True) for i in range(len(edges) - 1)})
        return _Strata(codes.astype(np.int32), names, data, data)

    raw, nd = _raw_map(data, nodata)
    da = raw if nd is None else raw.where(raw != nd)
    valid = da.notnull()
    if bins is not None:
        edges = np.asarray(bins, dtype=np.float64)
        codes = xr.where(valid, _digitize(da.astype(np.float64), edges), -1)
        names = {i + 1: _interval(edges[i], edges[i + 1]) for i in range(len(edges) - 1)}
        return _Strata(codes.astype(np.int32), names, None, da)
    from ._plot._style import classes_from_attrs

    if np.issubdtype(raw.dtype, np.integer) and np.can_cast(raw.dtype, np.int32):
        codes, found = _integer_codes(raw, nd)
        if len(found) > 1000:
            raise ValueError(f"the map has {len(found)} distinct values: give the strata as intervals with bins=")
        labels = classes_from_attrs(da.attrs) or {}
        names = {int(v): labels.get(float(v), str(int(v))) for v in found}
        return _Strata(codes, names, None, da)
    values = da.data
    if hasattr(values, "dask"):
        import dask.array as dask_array
        found = dask_array.unique(values[~dask_array.isnan(values)] if np.issubdtype(da.dtype, np.floating)
                                  else values).compute()
    else:
        found = np.unique(values[~np.isnan(values)] if np.issubdtype(da.dtype, np.floating) else values)
    found = np.asarray(found, dtype=np.float64)
    found = found[np.isfinite(found)]
    if len(found) and not np.allclose(found, np.round(found)):
        raise ValueError("the map has non-integer values: give the strata as intervals with bins=")
    if len(found) > 1000:
        raise ValueError(f"the map has {len(found)} distinct values: give the strata as intervals with bins=")
    labels = classes_from_attrs(da.attrs) or {}
    names = {int(v): labels.get(float(v), str(int(v))) for v in found}
    codes = xr.where(valid, da.fillna(-1), -1).astype(np.int32)
    return _Strata(codes, names, None, da)


def _digitize(values: xr.DataArray, edges: np.ndarray) -> xr.DataArray:
    """1-based interval of each value in [edges[i], edges[i + 1]); -1 outside."""
    def _f(v):
        idx = np.digitize(v, edges)
        return np.where((idx >= 1) & (idx < len(edges)) & np.isfinite(v), idx, -1)
    return xr.apply_ufunc(_f, values, dask="parallelized", output_dtypes=[np.int64])


def _interval(lo: float, hi: float, years: bool = False) -> str:
    if years:
        return f"{int(lo)}-{int(hi) - 1}" if hi - lo > 1 else str(int(lo))
    fmt = (lambda v: str(int(v)) if float(v).is_integer() else f"{v:g}")
    return f"{fmt(lo)}-{fmt(hi)}"


# ---------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------

def _row_areas(da: xr.DataArray) -> Tuple[np.ndarray, str]:
    """(the area of a pixel on each row, its unit): hectares from the transform and CRS; in
    a geographic CRS the area of each row on the sphere; without a CRS, one pixel each."""
    from ._warp import transform_of

    ny = da.sizes["y"]
    crs = da.rio.crs
    if crs is None:
        return np.ones(ny), "pixels"
    t = transform_of(da)
    if crs.is_geographic:
        lat_top = np.radians(t.f + t.e * np.arange(ny))
        lat_bottom = np.radians(t.f + t.e * (np.arange(ny) + 1))
        area = _EARTH_RADIUS ** 2 * np.radians(abs(t.a)) * np.abs(np.sin(lat_top) - np.sin(lat_bottom))
        return area / 1e4, "ha"
    try:
        factor = crs.linear_units_factor[1]
    except Exception:  # noqa: BLE001 - CRS without linear units
        factor = 1.0
    return np.full(ny, abs(t.a * t.e - t.b * t.d) * factor ** 2 / 1e4), "ha"


def _counts_per_row(strata: _Strata) -> np.ndarray:
    """(stratum, row) pixel counts, in the order of ``strata.names``; one pass, lazily."""
    codes = strata.codes
    if not hasattr(codes.data, "dask"):
        return _count_rows(np.asarray(codes.data), np.fromiter(strata.names, dtype=np.int64, count=len(strata.names)))
    stack = xr.concat([(codes == c).sum("x") for c in strata.names], dim="stratum")
    return np.asarray(stack.values, dtype=np.int64).reshape(len(strata.names), codes.sizes["y"])


def _count_rows(v: np.ndarray, names: np.ndarray) -> np.ndarray:
    """(code, row) pixel counts of a (y, x) array in memory, the codes in the order of ``names``: one bincount of each
    slab of rows, of each pixel's value plus the range of the values times its row when the range is small, else of
    its position in ``names`` (k for none) plus k + 1 times its row."""
    k, (ny, nx) = len(names), v.shape
    counts = np.zeros((ny, k + 1), dtype=np.int64)
    if k == 0 or v.size == 0:
        return counts[:, :k].T.copy()
    lo, hi = int(names.min()), int(names.max())
    lut = None
    if hi - lo <= _SLAB:
        lut = np.full(hi - lo + 2, k, dtype=np.int64)   # the last entry: outside the range
        lut[names - lo] = np.arange(k)
    else:
        order = np.argsort(names)
        sorted_names = names[order]
    step = max(1, _SLAB // max(nx, 1))
    rows = np.arange(min(step, ny), dtype=np.int64)[:, None]
    for r0 in range(0, ny, step):
        block = v[r0:r0 + step]
        n = block.shape[0]
        base = min(lo, int(block.min()))
        width = max(hi, int(block.max())) - base + 1
        if width * n <= 4 * _SLAB:   # a few codes: count (value, row) directly, then take the columns of the names
            key = np.subtract(block, base, dtype=np.int64)
            key += width * rows[:n]
            seen = np.bincount(key.reshape(-1), minlength=width * n).reshape(n, width)
            counts[r0:r0 + n, :k] = seen[:, names - base]
            continue
        block = block.astype(np.int64)
        if lut is not None:
            rel = block - lo
            rel[(rel < 0) | (rel > hi - lo)] = hi - lo + 1
            pos = lut[rel]
        else:
            at = np.clip(np.searchsorted(sorted_names, block), 0, k - 1)
            pos = np.where(sorted_names[at] == block, order[at], k)
        key = pos + (k + 1) * np.arange(n, dtype=np.int64)[:, None]
        counts[r0:r0 + n] = np.bincount(key.reshape(-1), minlength=(k + 1) * n).reshape(n, k + 1)
    return counts[:, :k].T.copy()


@dataclass
class _Design:
    table: pd.DataFrame          # per stratum: name, pixels, area, weight
    counts: np.ndarray           # (stratum, row)
    unit: str


def _stratum_table(strata: _Strata) -> _Design:
    counts = _counts_per_row(strata)
    row_area, unit = _row_areas(strata.codes)
    area = counts @ row_area
    total = area.sum()
    table = pd.DataFrame({"name": list(strata.names.values()), "pixels": counts.sum(axis=1),
                          "area": area, "weight": area / total if total > 0 else np.zeros_like(area)},
                         index=pd.Index(list(strata.names), name="stratum"))
    return _Design(table, counts, unit)


# ---------------------------------------------------------------------------
# Sampling design
# ---------------------------------------------------------------------------

def sampling_design(
    data: Any,
    *,
    expected_ua: Union[float, Dict[Any, float]] = 0.75,
    std_error: float = 0.01,
    n: Optional[int] = None,
    alloc: Union[str, Dict[Any, int]] = "proportional",
    min_per_stratum: int = 50,
    bins: Optional[Sequence[float]] = None,
    nodata: Any = "auto",
) -> pd.DataFrame:
    """How many sample points a map needs, and how many in each stratum (Olofsson et al. 2014).

    Parameters
    ----------
    data
        The map to assess, whose classes are the strata: a map of classes (``DataArray`` or
        a raster path, e.g. the ``label`` of ``zeit.classify``), or a map of events (what
        ``zeit.extract_events`` or ``zeit.agreement`` return), stratified as "no change"
        and "change".
    expected_ua
        The user's accuracy expected of each stratum: one value, or a dict of stratum
        (name or code) -> value, others 0.75. Classes expected to be mapped well (0.9) need
        fewer points than hard ones (0.6).
    std_error
        Target standard error of the overall accuracy.
    n
        Total sample size; computed from ``expected_ua`` and ``std_error`` when ``None``
        (Olofsson et al. 2014, eq. 13).
    alloc
        ``"proportional"`` (to the area of each stratum), ``"equal"``, ``"neyman"`` (to area x
        expected standard deviation), or a dict of stratum -> number of points.
    min_per_stratum
        Strata smaller than this get this many points, taken from the larger strata
        (Olofsson et al. recommend 50-100 for rare classes such as change); 0 disables.
    bins
        Strata as intervals of the map's values (edges, ``[a, b)``), e.g. periods of the
        year of change of a map of events: ``bins=[2000, 2010, 2020]``.
    nodata
        Value of the map outside every stratum: ``"auto"`` (the map's NoData), a number or
        ``None``. A map of events keeps its 0 as "no change".

    Returns
    -------
    pandas.DataFrame
        One row per stratum (the index is the stratum's code): ``name``, ``pixels``,
        ``area`` (hectares, see ``attrs["area_unit"]``), ``weight`` (share of the area),
        ``expected_ua`` and ``n``. ``attrs`` keeps the total ``n``, ``std_error`` and how the
        sample was allocated.

    Examples
    --------
    >>> design = zeit.sampling_design(events, expected_ua={"change": 0.6, "no change": 0.95})
    >>> points = zeit.stratified_sample(events, design=design)
    """
    strata = _strata(data, bins, nodata)
    design = _stratum_table(strata)
    table = design.table
    table = table[table.pixels > 0].copy()
    if table.empty:
        raise ValueError("the map has no pixel in any stratum")
    table["expected_ua"] = [_lookup(expected_ua, code, name, 0.75) for code, name in zip(table.index, table.name)]
    if not ((table.expected_ua > 0) & (table.expected_ua < 1)).all():
        raise ValueError("expected_ua must be between 0 and 1")
    s = np.sqrt(table.expected_ua * (1 - table.expected_ua))
    if n is None:
        if not 0 < std_error < 1:
            raise ValueError("std_error must be between 0 and 1")
        w = table.weight
        total_pixels = float(table.pixels.sum())
        n = (w * s).sum() ** 2 / (std_error ** 2 + (w * s ** 2).sum() / total_pixels)
        n = int(np.ceil(n))
    n = int(n)
    if isinstance(alloc, dict):
        counts = np.array([_lookup(alloc, code, name, None) for code, name in zip(table.index, table.name)])
        if any(c is None for c in counts):
            raise ValueError(f"alloc must give a number for every stratum: {list(table.name)}")
        counts = counts.astype(np.int64)
        how = "given"
    else:
        if alloc == "proportional":
            share = table.weight.to_numpy()
        elif alloc == "equal":
            share = np.ones(len(table))
        elif alloc == "neyman":
            share = (table.weight * s).to_numpy()
        else:
            raise ValueError(f"alloc must be 'proportional', 'equal', 'neyman' or a dict, got {alloc!r}")
        counts = _allocate(n, share / share.sum(), int(min_per_stratum))
        how = alloc + (f", at least {int(min_per_stratum)} per stratum" if min_per_stratum else "")
    table["n"] = np.minimum(counts, table.pixels.to_numpy())
    table.attrs.update(n=int(table.n.sum()), std_error=float(std_error), alloc=how, area_unit=design.unit)
    return table


def _lookup(values: Any, code: int, name: str, default: Any) -> Any:
    if not isinstance(values, dict):
        return values
    for key in (name, code, str(code)):
        if key in values:
            return values[key]
    return default


def _allocate(n: int, share: np.ndarray, minimum: int) -> np.ndarray:
    """``n`` points by ``share`` (largest remainders), at least ``minimum`` per stratum, the
    extra taken from the strata above the minimum in proportion."""
    counts = _round(n * share, n)
    if minimum <= 0:
        return counts
    low = counts < minimum
    for _ in range(len(counts)):
        if not low.any():
            break
        need = int(minimum * low.sum())
        rest = n - need
        if rest <= 0:
            return np.where(low, minimum, counts if not low.all() else minimum).astype(np.int64)
        free = share * ~low
        counts = np.where(low, minimum, _round(rest * free / free.sum(), rest) if free.sum() > 0 else 0)
        low = low | (counts < minimum)
    return counts.astype(np.int64)


def _round(values: np.ndarray, total: int) -> np.ndarray:
    floor = np.floor(values).astype(np.int64)
    extra = int(total - floor.sum())
    if extra > 0:
        order = np.argsort(-(values - floor), kind="stable")
        floor[order[:extra]] += 1
    return floor


# ---------------------------------------------------------------------------
# Drawing the sample
# ---------------------------------------------------------------------------

def stratified_sample(
    data: Any,
    n: Optional[int] = None,
    *,
    design: Optional[pd.DataFrame] = None,
    bins: Optional[Sequence[float]] = None,
    nodata: Any = "auto",
    seed: int = 42,
    **design_kwargs: Any,
):
    """A stratified random sample of a map's pixels: the points to label for ``zeit.accuracy``.

    Parameters
    ----------
    data
        The map, as for ``sampling_design``.
    n
        Total number of points (allocated as ``sampling_design`` does); or
    design
        The table of ``sampling_design``, whose ``n`` column is used as is. Without ``n``
        and ``design``, ``sampling_design(data, **design_kwargs)`` decides.
    bins, nodata
        As for ``sampling_design``.
    seed
        Seed of the random draw: the same map and seed give the same points, in memory or
        lazy, whatever the chunks.

    Returns
    -------
    geopandas.GeoDataFrame
        One point per sampled pixel, at its centre, in the CRS of the map: ``stratum``
        (code), ``stratum_name``, ``row`` and ``col``. Add the reference label of each point
        (e.g. with ``zeit.interpret``, QGIS or field visits) in a column such as ``ref``.
    """
    import geopandas as gpd
    from ._warp import transform_of

    strata = _strata(data, bins, nodata)
    if design is None:
        design = sampling_design(data, n=n, bins=bins, nodata=nodata, **design_kwargs)
    counts = _counts_per_row(strata)
    rng = np.random.default_rng(seed)
    codes = list(strata.names)
    picks: List[Tuple[int, int, int]] = []   # (row, rank within the row, code)
    for code in design.index:
        k = int(design.loc[code, "n"])
        if k <= 0 or code not in codes:
            continue
        per_row = counts[codes.index(code)]
        total = int(per_row.sum())
        ranks = np.sort(rng.choice(total, size=min(k, total), replace=False))
        ends = np.cumsum(per_row)
        rows = np.searchsorted(ends, ranks, side="right")
        within = ranks - (ends[rows] - per_row[rows])
        picks.extend(zip(rows.tolist(), within.tolist(), [int(code)] * len(rows)))
    rows_needed = sorted({r for r, _, _ in picks})
    cols_of = {}
    if rows_needed:
        block = np.asarray(strata.codes.isel(y=rows_needed).values)
        for i, r in enumerate(rows_needed):
            cols_of[r] = block[i]
    out_rows, out_cols, out_codes = [], [], []
    for r, w, code in picks:
        cols = np.flatnonzero(cols_of[r] == code)
        out_rows.append(r)
        out_cols.append(int(cols[w]))
        out_codes.append(code)
    out_rows, out_cols = np.asarray(out_rows, dtype=np.int64), np.asarray(out_cols, dtype=np.int64)
    da = strata.codes
    if "x" in da.coords and "y" in da.coords and da.sizes["x"] > 1 and da.sizes["y"] > 1:
        t = transform_of(da)
        xs, ys = t * (out_cols + 0.5, out_rows + 0.5)
    else:
        xs, ys = out_cols + 0.5, out_rows + 0.5
    order = np.lexsort((out_cols, out_rows, np.asarray(out_codes)))
    gdf = gpd.GeoDataFrame(
        {"stratum": np.asarray(out_codes, dtype=np.int64)[order],
         "stratum_name": [strata.names[c] for c in np.asarray(out_codes)[order]],
         "row": out_rows[order], "col": out_cols[order]},
        geometry=gpd.points_from_xy(np.asarray(xs)[order], np.asarray(ys)[order]), crs=da.rio.crs)
    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@dataclass
class Accuracy:
    """The accuracy of a map and the areas of its classes, estimated from a reference sample.

    Attributes
    ----------
    confusion
        Error matrix in proportions of area: rows the map's classes, columns the reference's.
    counts
        The error matrix in sample points.
    overall
        Overall accuracy, with its ``se``, ``ci_low`` and ``ci_high`` (a ``pandas.Series``).
    users, producers
        User's and producer's accuracy of each class (``estimate``, ``se``, ``ci_low``,
        ``ci_high``); NaN for a class the map or the reference never gives.
    area
        Per class: ``mapped`` (the area the map gives it; inside the strata when they are
        given on the map's grid), ``estimate`` (the error-adjusted area), ``se``, ``ci``
        (half-width of the confidence interval), ``ci_low``, ``ci_high``; in ``area_unit``.
    estimator
        How ``area`` was estimated: ``"direct"``, ``"difference"`` or ``"ratio"``.
    date
        For maps of events with ``date_tolerance``: of the points that both the map and the
        reference call change, the estimated share whose dates are within the tolerance and
        the mean difference in years.
    """
    confusion: pd.DataFrame
    counts: pd.DataFrame
    overall: pd.Series
    users: pd.DataFrame
    producers: pd.DataFrame
    area: pd.DataFrame
    area_unit: str
    confidence: float
    n: int
    n_unlabelled: int = 0
    date: Optional[pd.Series] = None
    strata: Optional[pd.DataFrame] = field(default=None, repr=False)
    estimator: str = "direct"

    def to_dataframe(self) -> pd.DataFrame:
        """One row per class: user's and producer's accuracy and area, with their intervals."""
        parts = {"users": self.users, "producers": self.producers, "area": self.area}
        return pd.concat(parts, axis=1)

    def __repr__(self) -> str:
        pct = int(round(self.confidence * 100))
        lines = [f"Accuracy of the map, from {self.n} reference points"
                 + (f" ({self.n_unlabelled} without a label left out)" if self.n_unlabelled else "") + ":",
                 f"  overall: {self.overall.estimate:.3f} ± {self.overall.ci_high - self.overall.estimate:.3f} "
                 f"({pct}% CI)", ""]
        table = pd.DataFrame({
            "user's": [_pm(self.users.loc[c]) for c in self.users.index],
            "producer's": [_pm(self.producers.loc[c]) for c in self.producers.index],
            f"mapped ({self.area_unit})": [f"{v:,.0f}" for v in self.area.mapped],
            f"estimated ({self.area_unit}{'' if self.estimator == 'direct' else ', ' + self.estimator})":
                [f"{e:,.0f} ± {h:,.0f}" if np.isfinite(e) else "-" for e, h in zip(self.area.estimate, self.area.ci)],
        }, index=self.users.index)
        lines.append(table.to_string())
        if self.date is not None:
            lines += ["", f"  dates of change within {self.date.tolerance:g} year(s): {self.date.within:.3f} "
                          f"(mean difference {self.date.mean_difference:.2f} years, {int(self.date.n)} points)"]
        return "\n".join(lines)


def _pm(row: pd.Series) -> str:
    if not np.isfinite(row.estimate):
        return "-"
    return f"{row.estimate:.3f} ± {row.ci_high - row.estimate:.3f}"


def accuracy(
    map: Any,   # noqa: A002 - the name says what it is
    samples: Any,
    *,
    reference: str = "ref",
    strata: Any = None,
    bins: Optional[Sequence[float]] = None,
    date_tolerance: Optional[float] = None,
    reference_date: str = "ref_date",
    confidence: float = 0.95,
    estimator: str = "direct",
    nodata: Any = "auto",
) -> Accuracy:
    """Accuracy and error-adjusted areas of a map, from a reference sample (Olofsson et al. 2014).

    Parameters
    ----------
    map
        The map to assess: a map of classes (``DataArray`` or raster path) or a map of
        events (what ``zeit.extract_events`` or ``zeit.agreement`` return: "change" where it
        has a ``yod``, "no change" elsewhere; with ``bins``, a class per period).
    samples
        The reference points (``GeoDataFrame``, vector file, or what ``zeit.interpret``
        returns), with the reference class in the ``reference`` column, in any CRS. Points
        without a reference label are left out. Reference classes are names of the map's
        classes (its ``flag_meanings``), their codes, or for a map of events ``"change"``/
        ``"no change"`` (``True``/``False``, ``1``/``0`` too).
    reference
        Column with the reference class.
    strata
        The strata the sample was drawn from, when they are not the classes of ``map``:
        the map that was stratified (a map of classes or of events). Its areas weigh the
        points. Default: the classes of ``map`` (a sample from ``stratified_sample(map)``,
        or a simple random sample).
    bins
        Classes of ``map`` as intervals of its values, as in ``sampling_design``.
    date_tolerance
        Maps of events: also compare the dates. Of the points that the map and the
        reference both call change, the share whose years (of the map's ``date`` and the
        ``reference_date`` column) are at most this many years apart.
    reference_date
        Column with the date of the change in the reference.
    confidence
        Level of the confidence intervals.
    estimator
        How the areas are estimated: ``"direct"`` (Stehman 2014: the sample alone, the
        reference's share of each stratum weighed by its area), ``"difference"`` (the mapped
        area plus the estimated area of the reference minus the map's: Stehman 2013, McRoberts
        et al. 2018) or ``"ratio"`` (the mapped area times the estimated ratio of the
        reference's area to the map's: Stehman 2014). The last two use the map's areas, known
        everywhere: with strata that are not the map's classes, or a rare class, their
        intervals are narrower when the map is good. With the map's classes as strata the three
        agree. The ratio is NaN for a class no sample point is mapped as. The accuracies are
        the same whatever the estimator.
    nodata
        As for ``sampling_design``.

    Returns
    -------
    Accuracy
        ``overall``, ``users``, ``producers``, ``area`` (error-adjusted, with intervals),
        ``confusion`` (in proportions of area) and ``counts``; ``print`` it for a summary,
        ``to_dataframe()`` for a table.

    Examples
    --------
    >>> points = zeit.stratified_sample(events, n=400)
    >>> # ... label each point in a "ref" column (zeit.interpret, QGIS, field visits) ...
    >>> acc = zeit.accuracy(events, points)
    >>> acc.area.loc["change", ["estimate", "ci"]]
    """
    _check_estimator(estimator)
    samples, n_unlabelled = _labelled(samples, reference)
    map_strata = _strata(map, bins, nodata)
    return _assess(samples, n_unlabelled, map_strata, strata, nodata, confidence, estimator,
                   lambda s: _reference_names(s[reference], map_strata),
                   date_tolerance=date_tolerance, reference_date=reference_date)


def transitions(
    map_from: Any,
    map_to: Any,
    samples: Any,
    *,
    reference_from: str = "ref_from",
    reference_to: str = "ref_to",
    strata: Any = None,
    confidence: float = 0.95,
    estimator: str = "direct",
    nodata: Any = "auto",
) -> Accuracy:
    """The area of each transition between two dates of a map of classes, from a reference sample.

    Each pair of a class at the first date and a class at the second is a class of a map of
    transitions ("Forest -> Pasture", and "Forest -> Forest" where nothing changed), assessed as
    ``accuracy`` assesses a map: the error matrix of the transitions, their user's and
    producer's accuracies and the error-adjusted area of each transition with its interval
    (Olofsson et al. 2014; Stehman 2014: the strata need not be the transitions), by the
    direct, difference or ratio estimator.

    Parameters
    ----------
    map_from, map_to
        The map of classes at the first and at the second date (``DataArray`` or raster path),
        on the same grid.
    samples
        The reference points (``GeoDataFrame``, vector file, or what ``zeit.interpret``
        returns), in any CRS, with the reference class at each date in the
        ``reference_from`` and ``reference_to`` columns (names or codes of the maps' classes).
        Points without both are left out.
    reference_from, reference_to
        Columns with the reference class at the first and at the second date.
    strata
        The strata the sample was drawn from (a map of classes or of events), when they are
        not the transitions: e.g. the classes of one of the dates, or a map of change. Its
        areas weigh the points. Default: the transitions themselves.
    confidence
        Level of the confidence intervals.
    estimator
        ``"direct"``, ``"difference"`` or ``"ratio"``, as for ``accuracy``: the model-assisted
        ones use the mapped area of each transition and narrow the intervals of rare
        transitions when the maps are good.
    nodata
        As for ``sampling_design``.

    Returns
    -------
    Accuracy
        As ``accuracy``'s, its classes the transitions ``"<from> -> <to>"``: ``area`` has the
        mapped and the estimated area of each.

    Examples
    --------
    >>> # points stratified by a map of change, labelled with the class at each date
    >>> tr = zeit.transitions(lc_2015, lc_2022, points, strata=change, estimator="difference")
    >>> tr.area.loc["Forest -> Pasture", ["estimate", "ci"]]
    """
    _check_estimator(estimator)
    import geopandas as gpd

    if hasattr(samples, "samples") and isinstance(getattr(samples, "samples"), gpd.GeoDataFrame):
        samples = samples.samples
    if not isinstance(samples, gpd.GeoDataFrame):
        samples = gpd.read_file(samples)
    for column in (reference_from, reference_to):
        if column not in samples.columns:
            raise ValueError(f"the samples have no {column!r} column (the reference class at a date); columns: "
                             f"{list(samples.columns)}")
    given = np.ones(len(samples), dtype=bool)
    for column in (reference_from, reference_to):
        given &= (samples[column].notna() & (samples[column].astype(str).str.strip() != "")).to_numpy()
    n_unlabelled = int((~given).sum())
    samples = samples[given].reset_index(drop=True)
    a, b = _strata(map_from, None, nodata), _strata(map_to, None, nodata)
    if a.events is not None or b.events is not None:
        raise ValueError("transitions are between two maps of classes, not of events")
    if a.codes.shape != b.codes.shape or not np.allclose(_grid_of(a.codes), _grid_of(b.codes)):
        raise ValueError("the two maps should be on the same grid")
    pairs = _transition_strata(a, b)

    def reference(s):
        f = _reference_names(s[reference_from], a)
        t = _reference_names(s[reference_to], b)
        return [f"{x}{SEPARATOR}{y}" for x, y in zip(f, t)]

    return _assess(samples, n_unlabelled, pairs, strata, nodata, confidence, estimator, reference)


SEPARATOR = " -> "


def _check_estimator(estimator: str) -> None:
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator should be one of {ESTIMATORS}, not {estimator!r}")


def _grid_of(codes: xr.DataArray) -> np.ndarray:
    from ._warp import transform_of

    try:
        return np.asarray(tuple(transform_of(codes))[:6], dtype=np.float64)
    except Exception:  # noqa: BLE001 - no coordinates: the shape alone
        return np.zeros(6)


def _transition_strata(a: _Strata, b: _Strata) -> _Strata:
    """A map of transitions: a code for each (class at a, class at b) found, named "<a> -> <b>"."""
    base = int(max(b.names)) + 1 if b.names else 1
    lo = int(min(min(a.names, default=0), min(b.names, default=0), 0))
    shift = -lo
    width = base + shift
    va, vb = a.codes.data, b.codes.data
    if hasattr(va, "dask") or hasattr(vb, "dask"):
        import dask.array as dask_array
        valid = (a.codes >= 0) & (b.codes >= 0)
        codes = xr.where(valid, (a.codes.astype(np.int64) + shift) * width + (b.codes.astype(np.int64) + shift), -1)
        found = np.asarray(dask_array.unique(codes.data).compute())
    else:   # both in memory: one pass, the transitions found from a table of the codes seen
        va, vb = np.asarray(va), np.asarray(vb)
        pair = (va.astype(np.int64) + shift) * width + (vb.astype(np.int64) + shift)
        pair[(va < 0) | (vb < 0)] = -1
        codes = xr.DataArray(pair, coords=a.codes.coords, dims=a.codes.dims)
        found = _distinct(pair)
    found = found[found >= 0]
    names = {}
    for code in found.tolist():
        i, j = divmod(int(code), width)
        names[int(code)] = f"{a.names[i - shift]}{SEPARATOR}{b.names[j - shift]}"
    codes = codes.astype(np.int64)
    if a.codes.rio.crs is not None:
        codes = codes.rio.write_crs(a.codes.rio.crs)
    return _Strata(codes, names, None, codes)


def _inside(map_strata: _Strata, strata: _Strata) -> _Strata:
    """The map's classes inside the strata (the population the sample was drawn from), when the
    strata are on the map's grid; else the whole map."""
    a, b = map_strata.codes, strata.codes
    if a.shape != b.shape or not np.allclose(_grid_of(a), _grid_of(b)):
        return map_strata
    codes = a.where(b.data >= 0, -1)
    return _Strata(codes, map_strata.names, map_strata.events, map_strata.source)


def _assess(samples, n_unlabelled: int, map_strata: _Strata, strata: Any, nodata: Any, confidence: float,
            estimator: str, reference_of, date_tolerance: Optional[float] = None,
            reference_date: str = "ref_date") -> Accuracy:
    """The estimates of ``map_strata`` from the labelled ``samples`` (``reference_of(samples)``: the
    reference class of each), weighed by the areas of ``strata`` (default: the map's classes)."""
    strata_strata = map_strata if strata is None else _strata(strata, None, nodata)
    if "stratum" not in samples.columns and strata is None:
        warnings.warn("the samples have no 'stratum' column: they are weighed as a sample stratified by the "
                      "map's classes (right for a stratified or simple random sample, not for points chosen by "
                      "hand)", stacklevel=3)

    points = _to_crs(samples, map_strata.codes)
    m_codes, inside = _values_at(map_strata.codes, points)
    points_s = _to_crs(samples, strata_strata.codes)
    h_codes, inside_s = _values_at(strata_strata.codes, points_s)
    keep = inside & inside_s & (m_codes >= 0) & (h_codes >= 0)
    if (~keep).any():
        warnings.warn(f"{int((~keep).sum())} reference points fall outside the map or its strata and are left "
                      "out", stacklevel=3)
    if not keep.any():
        raise ValueError("no reference point falls on the map")
    samples, m_codes, h_codes = samples[keep].reset_index(drop=True), m_codes[keep], h_codes[keep]
    points = points[keep].reset_index(drop=True)

    map_names = [map_strata.names[int(c)] for c in m_codes]
    ref_names = reference_of(samples)
    design = _stratum_table(strata_strata).table
    map_design = design if strata is None else _stratum_table(_inside(map_strata, strata_strata)).table
    mapped ={n_: float(a) for n_, a in zip(map_design.name, map_design.area)}
    classes = list(dict.fromkeys(list(map_strata.names.values()) + sorted(set(ref_names) - set(map_strata.names.values()))))

    est = _Estimator(np.asarray(h_codes), design, confidence)
    m = np.asarray(map_names, dtype=object)
    r = np.asarray(ref_names, dtype=object)

    # the error matrix: each point adds its weight (its stratum's over its points) to its cell
    position = {c: k for k, c in enumerate(classes)}
    cell = (np.array([position[c] for c in m], dtype=np.int64), np.array([position[c] for c in r], dtype=np.int64))
    n_cells = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(n_cells, cell, 1)
    p_cells = np.zeros((len(classes), len(classes)))
    np.add.at(p_cells, cell, est.point_weights(len(m)))
    confusion = pd.DataFrame(p_cells, index=pd.Index(classes, name="map"), columns=pd.Index(classes, name="reference"))
    counts = pd.DataFrame(n_cells, index=confusion.index, columns=confusion.columns)
    overall = pd.Series(est.mean(m == r, interval=True), index=["estimate", "se", "ci_low", "ci_high"])
    users = pd.DataFrame([est.ratio((m == c) & (r == c), m == c) for c in classes], index=pd.Index(classes, name="class"),
                         columns=["estimate", "se", "ci_low", "ci_high"])
    producers = pd.DataFrame([est.ratio((m == c) & (r == c), r == c) for c in classes],
                             index=users.index, columns=users.columns)
    total_area = float(design.area.sum())
    rows = []
    for c in classes:
        p, se = est.share(estimator, r == c, m == c, mapped.get(c, 0.0) / total_area if total_area > 0 else 0.0)
        half = est.z * se * total_area
        rows.append([mapped.get(c, 0.0), p * total_area, se * total_area, half, p * total_area - half,
                     p * total_area + half])
    area = pd.DataFrame(rows, index=users.index, columns=["mapped", "estimate", "se", "ci", "ci_low", "ci_high"])

    date = None
    if date_tolerance is not None:
        date = _date_agreement(map_strata, samples, m, r, est, points, reference_date, float(date_tolerance))
    return Accuracy(confusion=confusion, counts=counts, overall=overall, users=users, producers=producers,
                    area=area, area_unit=_row_areas(strata_strata.codes)[1], confidence=confidence,
                    n=len(samples), n_unlabelled=n_unlabelled, date=date, strata=design, estimator=estimator)


def _labelled(samples: Any, reference: str):
    import geopandas as gpd

    if hasattr(samples, "samples") and isinstance(getattr(samples, "samples"), gpd.GeoDataFrame):
        samples = samples.samples   # a zeit.interpret session
    if not isinstance(samples, gpd.GeoDataFrame):
        samples = gpd.read_file(samples)
    if reference not in samples.columns:
        raise ValueError(f"the samples have no {reference!r} column (the reference class); columns: "
                         f"{list(samples.columns)}")
    labelled = samples[reference].notna() & (samples[reference].astype(str).str.strip() != "")
    return samples[labelled].reset_index(drop=True), int((~labelled).sum())


def _to_crs(samples, da: xr.DataArray):
    crs = da.rio.crs
    if crs is not None and samples.crs is not None and samples.crs != crs:
        return samples.to_crs(crs)
    return samples


def _values_at(codes: xr.DataArray, points) -> Tuple[np.ndarray, np.ndarray]:
    """(the value of ``codes`` at each point, which points fall on the map)."""
    from ._warp import transform_of

    if "x" in codes.coords and codes.sizes["x"] > 1 and codes.sizes["y"] > 1:
        t = transform_of(codes)
        cols, rows = ~t * (points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
    else:
        cols, rows = points.geometry.x.to_numpy(), points.geometry.y.to_numpy()
    rows, cols = np.floor(rows).astype(np.int64), np.floor(cols).astype(np.int64)
    inside = (rows >= 0) & (rows < codes.sizes["y"]) & (cols >= 0) & (cols < codes.sizes["x"])
    out = np.full(len(rows), -1, dtype=np.int64)
    idx = np.flatnonzero(inside)
    if len(idx):
        values = codes.isel(y=xr.DataArray(rows[idx], dims="p"), x=xr.DataArray(cols[idx], dims="p")).values
        out[idx] = np.asarray(values, dtype=np.int64)
    return out, inside


_TRUE = {"true", "1", "yes", CHANGE}
_FALSE = {"false", "0", "no", NO_CHANGE, "stable", "nochange", "no_change"}


def _reference_names(values: pd.Series, strata: _Strata) -> List[str]:
    """The reference classes as names of the map's classes when they are codes of them."""
    by_code = {str(c): n for c, n in strata.names.items()}
    names = set(strata.names.values())
    out = []
    for v in values:
        if strata.events is not None and len(strata.names) == 2:
            text = str(v).strip().lower()
            if isinstance(v, (bool, np.bool_)) or text in _TRUE | _FALSE:
                out.append(CHANGE if (bool(v) if isinstance(v, (bool, np.bool_)) else text in _TRUE) else NO_CHANGE)
                continue
        if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
            key = str(int(v))
            out.append(by_code.get(key, key))
        else:
            text = str(v).strip()
            out.append(text if text in names else by_code.get(text, text))
    return out


class _Estimator:
    """Stehman's (2014) estimators of a stratified random sample: means of indicators and ratios
    of them, weighted by the area of each stratum, with their variances."""

    def __init__(self, strata: np.ndarray, design: pd.DataFrame, confidence: float):
        self.z = NormalDist().inv_cdf(0.5 + confidence / 2)
        self.groups = []
        for code, row in design.iterrows():
            members = np.flatnonzero(strata == code)
            if len(members) == 0 or row.weight == 0:
                continue
            self.groups.append((members, float(row.weight)))
        sampled = sum(w for _, w in self.groups)
        if sampled < 0.999999:
            missing = [str(n) for (c, n), w in zip(design.name.items(), design.weight) if w > 0
                       and not (strata == c).any()]
            warnings.warn(f"no reference point in the strata {missing}: they are left out, and the estimates cover "
                          f"{sampled:.1%} of the area", stacklevel=3)
            self.groups = [(m, w / sampled) for m, w in self.groups]

    def _interval(self, est: float, se: float) -> List[float]:
        return [est, se, est - self.z * se, est + self.z * se]

    def mean(self, y: np.ndarray, interval: bool = False) -> List[float]:
        y = np.asarray(y, dtype=np.float64)
        est = sum(w * y[m].mean() for m, w in self.groups)
        var = sum(w ** 2 * _var(y[m]) / len(m) for m, w in self.groups)
        return self._interval(est, np.sqrt(var)) if interval else [est]

    def ratio(self, y: np.ndarray, x: np.ndarray) -> List[float]:
        y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
        ybar = sum(w * y[m].mean() for m, w in self.groups)
        xbar = sum(w * x[m].mean() for m, w in self.groups)
        if xbar == 0:
            return [np.nan] * 4
        r = ybar / xbar
        # the variance of the residuals y - r x (as var(y) + r^2 var(x) - 2 r cov(y, x), without its
        # cancellation: exactly 0 where every point agrees)
        var = sum(w ** 2 * _var(y[m] - r * x[m]) / len(m) for m, w in self.groups) / xbar ** 2
        return self._interval(r, np.sqrt(max(var, 0.0)))

    def point_weights(self, n: int) -> np.ndarray:
        """Each point's weight in ``mean``: its stratum's weight over its points (0 if left out)."""
        out = np.zeros(n)
        for m, w in self.groups:
            out[m] = w / len(m)
        return out

    def share(self, estimator: str, y: np.ndarray, x: np.ndarray, x_share: float) -> Tuple[float, float]:
        """(estimate, standard error) of the share of the area where ``y`` (the reference's
        indicator of a class), given the map's indicator ``x`` of it and its share of the area
        ``x_share`` (known: the mapped area over the total).

        direct: the stratified mean of y (Stehman 2014). difference: x_share plus the stratified
        mean of y - x, its variance that of y - x (Sarndal et al. 1992, sec. 6.3; Stehman 2013;
        McRoberts et al. 2018). ratio: x_share times the ratio of the stratified means of y and
        x, its variance that of the ratio times x_share squared (the combined ratio estimator,
        Cochran 1977, sec. 6.11; Stehman 2013)."""
        y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
        if estimator == "difference":
            est, se, _, _ = self.mean(y - x, interval=True)
            return x_share + est, se
        if estimator == "ratio":
            est, se, _, _ = self.ratio(y, x)
            return x_share * est, x_share * se
        est, se, _, _ = self.mean(y, interval=True)
        return est, se


def _var(v: np.ndarray) -> float:
    return float(np.var(v, ddof=1)) if len(v) > 1 else 0.0


def _date_agreement(strata: _Strata, samples, m, r, est: _Estimator, points, column: str,
                    tolerance: float) -> pd.Series:
    if strata.events is None or "date" not in strata.events:
        raise ValueError("date_tolerance needs a map of events (with a date), such as extract_events returns")
    if column not in samples.columns:
        raise ValueError(f"date_tolerance needs the reference date of the change in a {column!r} column")
    dates = strata.events["date"]
    codes = xr.where(dates.notnull(), dates.dt.year, -1).astype(np.int64)
    map_year, _ = _values_at(codes, points)
    ref = pd.to_datetime(samples[column], errors="coerce")
    ref_year = np.where(ref.notna(), ref.dt.year.fillna(-1).astype(int), -1)
    both = (m == CHANGE) & (r == CHANGE) & (map_year > 0) & (ref_year > 0)
    diff = np.abs(map_year - ref_year).astype(np.float64)
    within = both & (diff <= tolerance)
    share = est.ratio(within, both)
    mean_diff = est.ratio(np.where(both, diff, 0.0), both)
    return pd.Series({"tolerance": tolerance, "within": share[0], "within_se": share[1],
                      "mean_difference": mean_diff[0], "n": int(both.sum())})
