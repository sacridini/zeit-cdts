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

Strata and areas come from the map: the area of each pixel from its transform and CRS
(hectares; in a geographic CRS, the area of each row on the sphere), lazily for dask maps.
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


def _as_map(data: Any, nodata: Any) -> xr.DataArray:
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
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        da = da.where(da != nodata)
    return da


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

    da = _as_map(data, nodata)
    valid = da.notnull()
    if bins is not None:
        edges = np.asarray(bins, dtype=np.float64)
        codes = xr.where(valid, _digitize(da.astype(np.float64), edges), -1)
        names = {i + 1: _interval(edges[i], edges[i + 1]) for i in range(len(edges) - 1)}
        return _Strata(codes.astype(np.int32), names, None, da)
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
    from ._plot._style import classes_from_attrs

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
    stack = xr.concat([(codes == c).sum("x") for c in strata.names], dim="stratum")
    return np.asarray(stack.values, dtype=np.int64).reshape(len(strata.names), codes.sizes["y"])


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
        Per class: ``mapped`` (the area the map gives it), ``estimate`` (the error-adjusted
        area), ``se``, ``ci`` (half-width of the confidence interval), ``ci_low``,
        ``ci_high``; in ``area_unit``.
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
            f"estimated ({self.area_unit})": [f"{e:,.0f} ± {h:,.0f}" for e, h in zip(self.area.estimate, self.area.ci)],
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
    samples, n_unlabelled = _labelled(samples, reference)
    map_strata = _strata(map, bins, nodata)
    strata_strata = map_strata if strata is None else _strata(strata, None, nodata)
    if "stratum" not in samples.columns and strata is None:
        warnings.warn("the samples have no 'stratum' column: they are weighed as a sample stratified by the "
                      "map's classes (right for a stratified or simple random sample, not for points chosen by "
                      "hand)", stacklevel=2)

    points = _to_crs(samples, map_strata.codes)
    m_codes, inside = _values_at(map_strata.codes, points)
    points_s = _to_crs(samples, strata_strata.codes)
    h_codes, inside_s = _values_at(strata_strata.codes, points_s)
    keep = inside & inside_s & (m_codes >= 0) & (h_codes >= 0)
    if (~keep).any():
        warnings.warn(f"{int((~keep).sum())} reference points fall outside the map or its strata and are left "
                      "out", stacklevel=2)
    if not keep.any():
        raise ValueError("no reference point falls on the map")
    samples, m_codes, h_codes = samples[keep].reset_index(drop=True), m_codes[keep], h_codes[keep]
    points = points[keep].reset_index(drop=True)

    map_names = [map_strata.names[int(c)] for c in m_codes]
    ref_names = _reference_names(samples[reference], map_strata)
    classes = list(dict.fromkeys(list(map_strata.names.values()) + sorted(set(ref_names) - set(map_strata.names.values()))))
    design = _stratum_table(strata_strata).table
    map_design = design if strata is None else _stratum_table(map_strata).table

    est = _Estimator(np.asarray(h_codes), design, confidence)
    m = np.asarray(map_names, dtype=object)
    r = np.asarray(ref_names, dtype=object)

    confusion = pd.DataFrame(0.0, index=pd.Index(classes, name="map"), columns=pd.Index(classes, name="reference"))
    counts = pd.DataFrame(0, index=confusion.index, columns=confusion.columns)
    for i in classes:
        for j in classes:
            y = (m == i) & (r == j)
            confusion.loc[i, j] = est.mean(y)[0]
            counts.loc[i, j] = int(y.sum())
    overall = pd.Series(est.mean(m == r, interval=True), index=["estimate", "se", "ci_low", "ci_high"])
    users = pd.DataFrame([est.ratio((m == c) & (r == c), m == c) for c in classes], index=pd.Index(classes, name="class"),
                         columns=["estimate", "se", "ci_low", "ci_high"])
    producers = pd.DataFrame([est.ratio((m == c) & (r == c), r == c) for c in classes],
                             index=users.index, columns=users.columns)
    total_area = float(design.area.sum())
    mapped = {n_: float(a) for n_, a in zip(map_design.name, map_design.area)}
    rows = []
    for c in classes:
        p, se, _, _ = est.mean(r == c, interval=True)
        half = est.z * se * total_area
        rows.append([mapped.get(c, 0.0), p * total_area, se * total_area, half, p * total_area - half,
                     p * total_area + half])
    area = pd.DataFrame(rows, index=users.index, columns=["mapped", "estimate", "se", "ci", "ci_low", "ci_high"])

    date = None
    if date_tolerance is not None:
        date = _date_agreement(map_strata, samples, m, r, est, points, reference_date, float(date_tolerance))
    return Accuracy(confusion=confusion, counts=counts, overall=overall, users=users, producers=producers,
                    area=area, area_unit=_row_areas(strata_strata.codes)[1], confidence=confidence,
                    n=len(samples), n_unlabelled=n_unlabelled, date=date, strata=design)


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
        var = sum(w ** 2 * (_var(y[m]) + r ** 2 * _var(x[m]) - 2 * r * _cov(y[m], x[m])) / len(m)
                  for m, w in self.groups) / xbar ** 2
        return self._interval(r, np.sqrt(max(var, 0.0)))


def _var(v: np.ndarray) -> float:
    return float(np.var(v, ddof=1)) if len(v) > 1 else 0.0


def _cov(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.cov(a, b, ddof=1)[0, 1]) if len(a) > 1 else 0.0


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
