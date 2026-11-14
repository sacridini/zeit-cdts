"""Accuracy of a field segmentation against a reference: boundaries, objects and sizes.

``segmentation_accuracy`` compares a label image (the result of ``zeit.fields``, or any
segmentation) with a reference one, or with polygons rasterised onto its grid. Everything is
computed inside a *region* (the pixels the reference speaks about), in three groups:

* boundaries: precision, recall and F of the boundary pixels, with a distance tolerance;
* objects (``kind="fields"``): over- and under-segmentation (Clinton et al. 2010) of each
  reference object against the predicted object that overlaps it most, and the share of the
  reference objects found with an IoU above a threshold;
* sizes: the areas of the reference and predicted objects, compared by quantiles, by the
  Wasserstein distance and the Kolmogorov-Smirnov test of their logarithms, and counted by size
  class.

With ``kind="property"`` (a cadastre such as the CAR, which delimits properties, not fields) only
the recall of the boundaries is meaningful, and the share of predicted fields that cross a
property boundary.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

_QUANTILES = (5, 25, 50, 75, 95)
_DEFAULT_CLASSES_HA = (0.0, 5.0, 20.0, 100.0, np.inf)


@dataclass
class SegmentationAccuracy:
    """The accuracy of a segmentation against a reference (see ``zeit.segmentation_accuracy``).

    Attributes
    ----------
    kind
        ``"fields"`` or ``"property"``.
    boundary
        ``precision``, ``recall`` and ``f`` of the boundary pixels at ``tolerance`` pixels, and
        the numbers of predicted (``n_pred``) and reference (``n_ref``) boundary pixels.
        ``precision`` and ``f`` are NaN for ``kind="property"``.
    objects
        ``kind="fields"``: ``n_ref``, ``n_pred`` (predicted objects overlapping the reference
        area by more than half their own area), ``n_unmatched`` (reference objects no predicted
        object touches), the means (``over_seg``, ``under_seg``, ``d``, per object) and the
        area-weighted means (``over_seg_w``, ``under_seg_w``, ``d_w``), ``iou_fraction`` (share
        of reference objects whose best match has IoU >= ``iou_threshold``) and
        ``matched_fraction`` (the same, matching one to one by descending IoU). With
        ``complete_only``, ``n_ref`` is the number of complete reference objects kept and
        ``n_ref_all`` the number before dropping.
    clip, complete_only
        The options the result was computed with.
    object_table
        One row per reference object: ``pixels``, ``area``, the ``best`` predicted object, its
        ``pred_pixels``, the ``overlap`` (pixels), ``over_seg``, ``under_seg``, ``d``, ``iou``
        and ``one_to_one`` (the predicted object it is paired with, 0 for none).
    sizes
        ``kind="fields"``: per set (``reference``, ``predicted``) the number, mean and quantiles
        (5, 25, 50, 75, 95) of the object areas, in ``area_unit``.
    size_tests
        ``wasserstein`` (of log10 areas), ``ks`` and ``ks_p`` (two-sample test on log10 areas).
    size_classes
        Number of reference and predicted objects in each size class.
    crossing
        ``kind="property"``: ``n_pred`` (predicted objects that overlap a property), ``n_crossing``,
        ``share`` of them and ``area_share`` (of their area).
    area_unit
        ``"ha"`` (with a CRS) or ``"pixels"``.
    """
    kind: str
    tolerance: float
    boundary: pd.Series
    objects: Optional[pd.Series] = None
    object_table: Optional[pd.DataFrame] = field(default=None, repr=False)
    sizes: Optional[pd.DataFrame] = None
    size_tests: Optional[pd.Series] = None
    size_classes: Optional[pd.DataFrame] = None
    crossing: Optional[pd.Series] = None
    area_unit: str = "pixels"
    region_pixels: int = 0
    iou_threshold: float = 0.5
    min_share: float = 0.1
    clip: bool = False
    complete_only: bool = False

    def summary(self) -> pd.Series:
        """The scalar results in one Series, with names like ``boundary.f`` and ``objects.d``."""
        parts = {"boundary": self.boundary, "objects": self.objects, "size_tests": self.size_tests,
                 "crossing": self.crossing}
        out = {}
        for prefix, s in parts.items():
            if s is not None:
                out.update({f"{prefix}.{k}": v for k, v in s.items()})
        return pd.Series(out, dtype=float)

    def __repr__(self) -> str:
        b = self.boundary
        opts = ("" + (", predictions clipped to the region" if self.clip else "")
                + (", complete reference fields only" if self.complete_only else ""))
        lines = [f"Segmentation accuracy ({self.kind}, tolerance {self.tolerance:g} px, "
                 f"{self.region_pixels} pixels in the region{opts}):",
                 f"  boundaries: precision {b.precision:.3f}, recall {b.recall:.3f}, F {b.f:.3f} "
                 f"({int(b.n_pred)} predicted, {int(b.n_ref)} reference pixels)"]
        if self.objects is not None:
            o = self.objects
            lines += [f"  objects: {int(o.n_ref)} reference, {int(o.n_pred)} predicted ({int(o.n_unmatched)} reference "
                      f"objects unmatched)",
                      f"    over-segmentation {o.over_seg:.3f}, under-segmentation {o.under_seg:.3f}, D {o.d:.3f} "
                      f"(area-weighted {o.over_seg_w:.3f}, {o.under_seg_w:.3f}, {o.d_w:.3f})",
                      f"    IoU >= {self.iou_threshold:g}: {o.iou_fraction:.3f} (one to one: {o.matched_fraction:.3f})"]
        if self.size_tests is not None:
            t = self.size_tests
            lines += [f"  sizes ({self.area_unit}): Wasserstein of log10 {t.wasserstein:.3f}, "
                      f"KS {t.ks:.3f} (p = {t.ks_p:.3g})", self.sizes.round(3).to_string()]
        if self.size_classes is not None:
            lines.append(self.size_classes.to_string())
        if self.crossing is not None:
            c = self.crossing
            lines.append(f"  predicted fields crossing a property boundary (second overlap >= {self.min_share:g}): "
                         f"{int(c.n_crossing)} of {int(c.n_pred)} ({c.share:.3f}; {c.area_share:.3f} of the area)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _pred_labels(pred: Any):
    """(labels as an int array, the DataArray they came from or None, the boundary label)."""
    boundary = 0
    if isinstance(pred, xr.Dataset):
        if "labels" not in pred:
            raise ValueError("the Dataset has no 'labels' variable: pass the result of zeit.fields")
        boundary = int(pred.attrs.get("boundary_label", 0))
        pred = pred["labels"]
    elif isinstance(pred, xr.DataArray):
        boundary = int(pred.attrs.get("boundary_label", 0))
    return pred, boundary


def _values2d(a: Any, what: str) -> np.ndarray:
    if isinstance(a, xr.DataArray):
        a = a.compute()
        if "y" in a.dims and "x" in a.dims:
            a = a.transpose(..., "y", "x")
        a = a.values
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"{what} must be 2-D, got shape {a.shape}")
    return a


def _integers(a: np.ndarray, what: str) -> np.ndarray:
    if a.dtype.kind == "b":
        return a.astype(np.int32)
    if a.dtype.kind in "iu":
        return a.astype(np.int64)
    if a.dtype.kind == "f":
        finite = np.where(np.isfinite(a), a, 0)
        if np.any(finite != np.round(finite)):
            raise TypeError(f"{what} must hold integer labels")
        return finite.astype(np.int64)
    raise TypeError(f"{what} must hold integer labels, got {a.dtype}")


def _georeference(da: Any, transform: Any, crs: Any):
    from ._fields import _georef

    if isinstance(da, xr.DataArray) and "x" in da.coords and "y" in da.coords:
        return _georef(da, transform, crs)
    return transform, crs


def _same_transform(a: Any, b: Any) -> bool:
    return all(abs(p - q) <= 1e-6 * max(abs(a.a), abs(a.e), 1e-12) for p, q in zip(tuple(a)[:6], tuple(b)[:6]))


def _rasterise(gdf: Any, shape, transform, crs) -> np.ndarray:
    from rasterio import features

    if transform is None:
        raise ValueError("rasterising polygons needs the grid's transform: pass a georeferenced `pred` or `transform=`")
    if crs is not None and gdf.crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = [(g, i + 1) for i, g in enumerate(gdf.geometry) if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, dtype=np.int64)
    out = features.rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=False,
                             dtype="int32")
    return out.astype(np.int64)


def _row_area(shape, transform, crs):
    """Area of a pixel on each row in hectares, planar in a projected CRS and geodesic in a geographic one;
    None without a CRS or transform."""
    if crs is None or transform is None:
        return None
    from shapely.geometry import box

    from ._fields import _measure

    ny = shape[0]
    w, h = abs(transform.a), abs(transform.e)
    from pyproj import CRS

    if CRS.from_user_input(crs).is_geographic:
        tops = transform.f + transform.e * np.arange(ny)
        # the longitude only sets the width; the latitude of each row decides its area
        boxes = [box(transform.c, min(t, t + transform.e), transform.c + w, max(t, t + transform.e)) for t in tops]
        return _measure(boxes, crs)[0]
    area = _measure([box(0.0, 0.0, w, h)], crs)[0][0]
    return np.full(ny, area)


def _boundary_pixels(lab: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Pixels in ``region`` with a 4-neighbour (anywhere in the grid) of a different label."""
    b = np.zeros(lab.shape, dtype=bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b & region


def _hit_share(a: np.ndarray, b: np.ndarray, tolerance: float) -> float:
    """Share of the pixels of ``a`` within ``tolerance`` of a pixel of ``b`` (NaN when ``a`` is empty)."""
    n = int(a.sum())
    if n == 0:
        return float("nan")
    if not b.any():
        return 0.0
    if tolerance == 0:
        return float((a & b).sum()) / n
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(~b)
    return float((dist[a] <= tolerance).sum()) / n


# ---------------------------------------------------------------------------
# The function
# ---------------------------------------------------------------------------

def segmentation_accuracy(
    pred: Any,
    reference: Any,
    *,
    kind: str = "fields",
    tolerance: float = 1,
    region: Any = None,
    iou_threshold: float = 0.5,
    min_share: float = 0.1,
    size_classes: Optional[Sequence[float]] = None,
    transform: Any = None,
    crs: Any = None,
    clip: bool = False,
    complete_only: bool = False,
) -> SegmentationAccuracy:
    """The accuracy of a field segmentation against a reference: boundaries, objects and sizes.

    **Region.** Everything is computed inside the region, by default the pixels where the
    reference has an object (``reference > 0``: the digitised or declared area), for
    ``kind="fields"`` grown by ``tolerance`` pixels (Euclidean) so that boundary pixels just
    outside the reference still count. ``region=`` (a boolean raster) replaces it.

    **Boundaries.** A boundary pixel of a label image is a pixel of the region with a 4-neighbour
    of a different label, wherever that neighbour is (so a field's edge against unlabelled land,
    label 0, is a boundary, and both pixels of a pair of neighbouring labels are). The label of the
    boundary network of ``zeit.fields`` (``attrs["boundary_label"]``) is treated as 0. With
    ``tolerance`` t, the precision is the share of predicted boundary pixels within a Euclidean
    distance t (pixels) of a reference boundary pixel, the recall the share of reference boundary
    pixels within t of a predicted one, and F their harmonic mean; ``tolerance=0`` asks for
    coincident pixels.

    **Objects** (``kind="fields"``; Clinton et al. 2010, Photogramm. Eng. Remote Sens.
    76(3):289). For each reference object r (its pixels in the region) the predicted object p
    with the largest overlap (ties: the lowest label): ``over_seg = 1 - |r & p| / |r|``,
    ``under_seg = 1 - |r & p| / |p|`` and ``d = sqrt((over_seg^2 + under_seg^2) / 2)``. |p| is
    counted in full (on the whole grid, not only in the region): a field that leaks out of the
    reference area is under-segmentation. This assumes the reference is complete around r (a
    reference with unmapped neighbours makes predicted fields look too large). A reference object
    that no predicted object touches has ``over_seg = under_seg = d = 1``, ``iou = 0``. The
    means are per object and weighted by the reference area. ``iou_fraction`` is the share of
    reference objects whose best match has IoU >= ``iou_threshold``; ``matched_fraction`` pairs
    reference and predicted objects one to one, by descending IoU, and counts the reference
    objects whose pair reaches it. Over- and under-segmentation and IoU are in pixels.

    **Clipping and complete fields** (phase 24e). The reference is only known inside the region, but
    by default a predicted object is counted in full, so when the region is smaller than the grid (for
    example the footprints of annotated chips) a predicted field that runs out of the region has a
    large |p| and a low IoU with a reference field that is complete inside it, and over- and
    under-segmentation are biased. ``clip=True`` clips the predicted objects to the region before every
    object metric (over- and under-segmentation, D and their weighted means, the matching,
    ``iou_fraction``, ``matched_fraction``, the predicted sizes and ``n_pred``; ``kind="property"``: the
    crossing share too). The boundary precision, recall and F already use the region (the pixels of the
    region, with neighbours anywhere) and are not changed. ``complete_only=True`` restricts the
    per-reference-object metrics to the reference objects that lie entirely inside the region and do not
    touch the grid's edge (an object cut by either is not a field of the reference): the matching is
    done among all the reference objects, then ``object_table``, the means, the fractions and the
    reference sizes use only the kept ones. ``objects.n_ref`` is the number kept and ``objects.n_ref_all``
    the number before dropping; with none left the means are NaN. ``complete_only`` needs
    ``kind="fields"`` (an error with ``"property"``). Both default to False: the results are then
    unchanged.

    **Sizes** (``kind="fields"``). The areas of the reference objects and of the predicted objects
    that overlap the reference area by more than half of their own: ha with a CRS (planar in a
    projected one, geodesic in a geographic one, as ``fields_to_polygons``), else pixels. Their
    quantiles, the Wasserstein distance and the two-sample Kolmogorov-Smirnov test of log10 areas,
    and the number of each in the size classes: ``size_classes``, edges in ha (pixels without a
    CRS), by default ``[0, 5, 20, 100, inf]`` (only with a CRS), as ``[lo, hi)`` intervals.

    **Properties** (``kind="property"``, a cadastre: properties, not fields). Only the recall of
    the boundaries is computed (precision is meaningless, fields subdivide properties; it and F are
    NaN) and the share of predicted fields that cross a property boundary: a predicted object
    overlapping a property crosses when its second-largest overlap with a property is at least
    ``min_share`` of its area (in full). ``crossing`` also gives the share of their area.

    Parameters
    ----------
    pred
        The Dataset of ``zeit.fields`` (its ``labels``), a labels DataArray or an integer array.
    reference
        A label raster on the same grid (DataArray or array, 0 = no reference), or a GeoDataFrame
        of polygons (fields, or properties), rasterised onto ``pred``'s grid (reprojected to its
        CRS; each polygon its own id, the later one wins where they overlap; a pixel belongs to a
        polygon when its centre does).
    kind
        ``"fields"`` or ``"property"``.
    tolerance
        Boundary distance tolerance in pixels (>= 0).
    region
        Boolean raster of the pixels evaluated, instead of the default.
    iou_threshold
        IoU from which a reference object counts as found.
    min_share
        ``kind="property"``: the share of a predicted object's area that a second property must
        cover for the object to cross.
    size_classes
        Edges of the size classes, in ha (pixels without a CRS).
    transform, crs
        Affine transform and CRS of the grid, instead of ``pred``'s (needed to rasterise polygons
        or measure areas when ``pred`` is an array).
    clip
        Clip the predicted objects to the region before the object metrics (see above).
    complete_only
        Only the reference objects entirely inside the region and away from the grid's edge (see above).

    Returns
    -------
    SegmentationAccuracy
        ``print(result)`` gives a summary; ``result.summary()`` the scalars as a Series and
        ``result.object_table`` the reference objects with their best match.

    Examples
    --------
    >>> res = zeit.fields(swir, preset="notebook", random_state=0)
    >>> acc = zeit.segmentation_accuracy(res, digitised_polygons, tolerance=2)
    >>> acc.boundary.f, acc.objects.iou_fraction
    """
    if kind not in ("fields", "property"):
        raise ValueError(f"kind must be 'fields' or 'property', got {kind!r}")
    if not tolerance >= 0:
        raise ValueError("tolerance must be >= 0")
    if complete_only and kind != "fields":
        raise ValueError("complete_only needs kind='fields' (a property reference has no per-field metrics)")

    pred_in, boundary_label = _pred_labels(pred)
    pred_arr = _integers(_values2d(pred_in, "pred"), "pred")
    shape = pred_arr.shape
    transform, crs = _georeference(pred_in, transform, crs)
    pred_arr = np.where(pred_arr > 0, pred_arr, 0)
    if boundary_label == 1:
        pred_arr[pred_arr == 1] = 0
    elif boundary_label > 1:
        pred_arr[pred_arr == boundary_label] = 0

    if hasattr(reference, "geometry") and hasattr(reference, "crs"):
        ref_arr = _rasterise(reference, shape, transform, crs)
    else:
        ref_raw = _values2d(reference, "reference")
        if ref_raw.shape != shape:
            raise ValueError(f"the reference's shape {ref_raw.shape} differs from pred's {shape}")
        r_transform, _ = _georeference(reference, None, None)
        if (isinstance(reference, xr.DataArray) and isinstance(pred_in, xr.DataArray) and r_transform is not None
                and transform is not None and "x" in pred_in.coords and "x" in reference.coords
                and not _same_transform(r_transform, transform)):
            raise ValueError("pred and reference are not on the same grid (their transforms differ)")
        ref_arr = _integers(ref_raw, "reference")
        ref_arr = np.where(ref_arr > 0, ref_arr, 0)

    has_ref = ref_arr > 0
    if region is None:
        if kind == "fields" and tolerance > 0 and has_ref.any():
            from scipy.ndimage import distance_transform_edt

            region_arr = distance_transform_edt(~has_ref) <= tolerance
        else:
            region_arr = has_ref.copy()
    else:
        region_arr = _values2d(region, "region").astype(bool)
        if region_arr.shape != shape:
            raise ValueError(f"the region's shape {region_arr.shape} differs from pred's {shape}")
    ref_eff = np.where(region_arr, ref_arr, 0)   # the reference inside the region
    has_eff = ref_eff > 0

    # Boundaries
    bp = _boundary_pixels(pred_arr, region_arr)
    br = _boundary_pixels(ref_arr, region_arr)
    recall = _hit_share(br, bp, tolerance)
    if kind == "fields":
        precision = _hit_share(bp, br, tolerance)
        if np.isnan(precision) or np.isnan(recall):
            f = float("nan")
        else:
            f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    else:
        precision = f = float("nan")
    boundary = pd.Series({"precision": precision, "recall": recall, "f": f,
                          "n_pred": int(bp.sum()), "n_ref": int(br.sum())})

    result = SegmentationAccuracy(kind=kind, tolerance=float(tolerance), boundary=boundary,
                                  region_pixels=int(region_arr.sum()), iou_threshold=float(iou_threshold),
                                  min_share=float(min_share), clip=bool(clip), complete_only=bool(complete_only))
    weights_row = _row_area(shape, transform, crs)
    result.area_unit = "pixels" if weights_row is None else "ha"

    # The boundaries above use the full labels; from here on the predicted objects may be clipped to the region
    if clip:
        pred_arr = np.where(region_arr, pred_arr, 0)

    # Overlaps between reference and predicted objects (pixels of the reference inside the region)
    pairs = pd.DataFrame({"r": ref_eff[has_eff], "p": pred_arr[has_eff]}).groupby(["r", "p"]).size()
    pairs = pairs.rename("inter").reset_index()
    real = pairs[pairs.p > 0]
    pred_rows = np.broadcast_to(np.arange(shape[0])[:, None], shape)

    def sums(labels, mask):
        vals = labels[mask]
        df = pd.DataFrame({"label": vals, "pixels": 1})
        df["area"] = 1.0 if weights_row is None else weights_row[pred_rows[mask]]
        out = df.groupby("label").sum()
        return out

    pred_sizes = sums(pred_arr, pred_arr > 0)
    pred_in_ref = sums(pred_arr, (pred_arr > 0) & has_eff)

    if kind == "property":
        result.crossing = _crossing(real, pred_sizes, min_share)
        return result

    ref_sizes = sums(ref_eff, has_eff)
    # best match of each reference object: largest overlap, lowest label on a tie
    best = real.sort_values(["inter", "p"], ascending=[False, True]).drop_duplicates("r").set_index("r")
    table = ref_sizes.rename(columns={"pixels": "pixels", "area": "area"}).copy()
    table["best"] = best["p"].reindex(table.index).fillna(0).astype(np.int64)
    table["overlap"] = best["inter"].reindex(table.index).fillna(0).astype(np.int64)
    table["pred_pixels"] = table["best"].map(pred_sizes["pixels"]).fillna(0).astype(np.int64)
    table["over_seg"] = 1 - table["overlap"] / table["pixels"]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["under_seg"] = np.where(table["best"] > 0, 1 - table["overlap"] / table["pred_pixels"], 1.0)
    table["over_seg"] = np.where(table["best"] > 0, table["over_seg"], 1.0)
    table["d"] = np.sqrt((table["over_seg"] ** 2 + table["under_seg"] ** 2) / 2)
    union = table["pixels"] + table["pred_pixels"] - table["overlap"]
    table["iou"] = np.where(table["best"] > 0, table["overlap"] / union, 0.0)

    # one to one, by descending IoU
    cand = real.copy()
    cand["iou"] = cand["inter"] / (cand["r"].map(ref_sizes["pixels"]) + cand["p"].map(pred_sizes["pixels"])
                                   - cand["inter"])
    cand = cand.sort_values(["iou", "r", "p"], ascending=[False, True, True])
    used_r, used_p, pair_of = set(), set(), {}
    for r, p in zip(cand["r"].to_numpy(), cand["p"].to_numpy()):
        if r in used_r or p in used_p:
            continue
        used_r.add(r)
        used_p.add(p)
        pair_of[r] = p
    table["one_to_one"] = [pair_of.get(r, 0) for r in table.index]
    iou_pair = {(r, p): i for r, p, i in zip(cand["r"], cand["p"], cand["iou"])}
    pair_iou = np.array([iou_pair.get((r, p), 0.0) for r, p in zip(table.index, table["one_to_one"])])
    table.index.name = "reference"
    table = table[["pixels", "area", "best", "pred_pixels", "overlap", "over_seg", "under_seg", "d", "iou",
                   "one_to_one"]]
    n_all = len(table)
    if complete_only:
        # reference objects with a pixel outside the region, or on the grid's edge, are cut
        cut = np.unique(np.concatenate([ref_arr[has_ref & ~region_arr], ref_arr[0], ref_arr[-1], ref_arr[:, 0],
                                        ref_arr[:, -1]]))
        whole = ~table.index.isin(cut[cut > 0]) if len(table) else np.zeros(0, dtype=bool)
        table, pair_iou = table[whole], pair_iou[whole]
    result.object_table = table

    n = len(table)
    w = table["area"].to_numpy()

    def mean(x, weights=None):
        return float(np.average(x, weights=weights)) if n else float("nan")

    keep = pred_in_ref["pixels"] > 0.5 * pred_sizes["pixels"].reindex(pred_in_ref.index)
    pred_objs = pred_sizes.loc[pred_in_ref.index[keep.to_numpy()]]
    result.objects = pd.Series({
        "n_ref": n, "n_pred": len(pred_objs), "n_unmatched": int((table["best"] == 0).sum()),
        "over_seg": mean(table["over_seg"]), "under_seg": mean(table["under_seg"]), "d": mean(table["d"]),
        "over_seg_w": mean(table["over_seg"], w), "under_seg_w": mean(table["under_seg"], w),
        "d_w": mean(table["d"], w),
        "iou_fraction": float((table["iou"] >= iou_threshold).mean()) if n else float("nan"),
        "matched_fraction": float((pair_iou >= iou_threshold).mean()) if n else float("nan"),
    })
    if complete_only:
        result.objects["n_ref_all"] = n_all

    # Sizes
    ra, pa = table["area"].to_numpy(float), pred_objs["area"].to_numpy(float)
    rows = {}
    for name, a in (("reference", ra), ("predicted", pa)):
        q = np.percentile(a, _QUANTILES) if len(a) else np.full(len(_QUANTILES), np.nan)
        rows[name] = {"n": len(a), "mean": float(a.mean()) if len(a) else float("nan"),
                      **{f"q{k:02d}": float(v) for k, v in zip(_QUANTILES, q)}}
    result.sizes = pd.DataFrame(rows).T
    if len(ra) and len(pa):
        from scipy.stats import ks_2samp, wasserstein_distance

        la, lp = np.log10(ra), np.log10(pa)
        ks = ks_2samp(la, lp)
        result.size_tests = pd.Series({"wasserstein": float(wasserstein_distance(la, lp)),
                                       "ks": float(ks.statistic), "ks_p": float(ks.pvalue)})
    else:
        result.size_tests = pd.Series({"wasserstein": np.nan, "ks": np.nan, "ks_p": np.nan})
    if size_classes is None and weights_row is not None:
        size_classes = _DEFAULT_CLASSES_HA
    if size_classes is not None:
        edges = np.asarray(size_classes, dtype=float)
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("size_classes must be increasing edges (at least two)")
        names = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-1], edges[1:])]
        result.size_classes = pd.DataFrame({
            "reference": np.histogram(ra, bins=_closed(edges, ra))[0],
            "predicted": np.histogram(pa, bins=_closed(edges, pa))[0]}, index=names)
    return result


def _closed(edges: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Histogram bins [lo, hi) for every class, the last one included on the right when finite."""
    if np.isinf(edges[-1]):
        edges = edges.copy()
        edges[-1] = max(np.nanmax(a) if len(a) else 0.0, edges[-2]) * 2 + 1
    else:
        edges = edges.copy()
        edges[-1] = np.nextafter(edges[-1], np.inf)   # the last class takes its upper edge
    return edges


def _crossing(real: pd.DataFrame, pred_sizes: pd.DataFrame, min_share: float) -> pd.Series:
    """Predicted objects overlapping a property whose second-largest overlap is >= min_share of their area."""
    overlapping = real.groupby("p")["inter"].apply(lambda s: np.sort(s.to_numpy())[::-1])
    n = len(overlapping)
    crossing = [p for p, s in overlapping.items() if len(s) > 1 and s[1] >= min_share * pred_sizes.loc[p, "pixels"]]
    area = pred_sizes["area"]
    total = float(area.loc[overlapping.index].sum())
    return pd.Series({
        "n_pred": n, "n_crossing": len(crossing),
        "share": len(crossing) / n if n else float("nan"),
        "area_share": float(area.loc[crossing].sum()) / total if total else float("nan"),
    })
