"""zeit.segmentation_accuracy: boundaries, objects and sizes, on small cases worked out by hand.

The grids are 6 rows x 8 columns (6 x 10 for the properties); a boundary between two labels is two pixels
wide (the last column of one and the first of the other), so a vertical boundary is 2 * 6 = 12 pixels.
"""

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit

TR = from_origin(500000.0, 9000000.0, 30.0, 30.0)
CRS = "EPSG:32722"


def _grid(cols, labels, rows=6):
    """An (rows, sum of cols) label image from column bands: cols=[4, 4], labels=[1, 2]."""
    return np.concatenate([np.full((rows, c), v, dtype=np.int32) for c, v in zip(cols, labels)], axis=1)


def _da(lab, tr=TR, crs=CRS):
    h, w = lab.shape
    da = xr.DataArray(lab, dims=("y", "x"),
                      coords={"y": tr.f + tr.e * (np.arange(h) + 0.5), "x": tr.c + tr.a * (np.arange(w) + 0.5)})
    return da.rio.write_crs(crs) if crs else da


def test_perfect_match():
    # Both fields reproduced with other labels: the boundary is columns 3 and 4 (12 pixels) in both.
    ref = _grid([4, 4], [1, 2])
    res = zeit.segmentation_accuracy(_grid([4, 4], [7, 9]), ref, tolerance=0)
    assert (res.boundary.precision, res.boundary.recall, res.boundary.f) == (1, 1, 1)
    assert (res.boundary.n_pred, res.boundary.n_ref) == (12, 12)
    o = res.objects
    assert (o.over_seg, o.under_seg, o.d, o.over_seg_w, o.under_seg_w) == (0, 0, 0, 0, 0)
    assert (o.iou_fraction, o.matched_fraction, o.n_ref, o.n_pred, o.n_unmatched) == (1, 1, 2, 2, 0)
    assert res.size_tests.wasserstein == 0 and res.size_tests.ks == 0
    assert res.area_unit == "pixels" and res.size_classes is None      # no CRS: no size classes by default
    assert list(res.object_table["best"]) == [7, 9] and list(res.object_table["one_to_one"]) == [7, 9]
    assert "F 1.000" in repr(res) and res.summary()["boundary.f"] == 1


def test_a_field_split_in_two():
    # Reference: field 1 = columns 0-5 (36 px), field 2 = columns 6-7 (12 px); boundary columns 5, 6 = 12 px.
    # Prediction splits field 1 at column 4: label 5 = columns 0-3 (24 px), label 6 = columns 4-5 (12 px).
    # Predicted boundaries: columns 3, 4 and 5, 6 = 24 px, of which 12 coincide with the reference's.
    ref = _grid([6, 2], [1, 2])
    pred = _grid([4, 2, 2], [5, 6, 7])
    res = zeit.segmentation_accuracy(pred, ref, tolerance=0, iou_threshold=0.7)
    b = res.boundary
    assert (b.n_pred, b.n_ref) == (24, 12)
    assert b.precision == 12 / 24 and b.recall == 1
    assert b.f == pytest.approx(2 * 0.5 * 1 / 1.5)
    t = res.object_table
    # field 1: best match is label 5 (overlap 24 of 36 and of 24): OS = 1 - 24/36 = 1/3, US = 1 - 24/24 = 0,
    # D = sqrt((1/9 + 0)/2); IoU = 24 / (36 + 24 - 24) = 2/3. Field 2 is exact.
    assert t.loc[1, "best"] == 5 and t.loc[1, "overlap"] == 24
    assert t.loc[1, "over_seg"] == pytest.approx(1 / 3) and t.loc[1, "under_seg"] == 0
    assert t.loc[1, "d"] == pytest.approx(np.sqrt(1 / 18)) and t.loc[1, "iou"] == pytest.approx(2 / 3)
    assert t.loc[2, "iou"] == 1 and t.loc[2, "d"] == 0
    o = res.objects
    assert o.over_seg == pytest.approx(1 / 6)                   # (1/3 + 0) / 2
    assert o.over_seg_w == pytest.approx(36 / 3 / 48)           # (36 * 1/3 + 12 * 0) / 48 = 0.25
    assert o.under_seg == 0
    assert o.iou_fraction == 0.5 and o.matched_fraction == 0.5  # only field 2 reaches 0.7
    assert zeit.segmentation_accuracy(pred, ref, tolerance=0).objects.iou_fraction == 1   # 2/3 >= 0.5
    assert o.n_pred == 3
    # Sizes in pixels: reference {12, 36}, predicted {12, 12, 24}. Wasserstein of the log10s: the quantile
    # functions differ on (1/2, 2/3) by log10(36/12) and on (2/3, 1) by log10(36/24); KS: the CDFs differ most
    # at 24 (reference 1/2, predicted 1).
    assert res.size_tests.wasserstein == pytest.approx(np.log10(3) / 6 + np.log10(1.5) / 3)
    assert res.size_tests.ks == pytest.approx(0.5) and 0 <= res.size_tests.ks_p <= 1
    assert res.sizes.loc["reference", "q50"] == 24 and res.sizes.loc["predicted", "q50"] == 12
    assert res.sizes.loc["reference", "n"] == 2 and res.sizes.loc["predicted", "n"] == 3


def test_two_fields_merged():
    # Reference: A = columns 0-1 (12 px), B = columns 2-4 (18 px), C = columns 5-7 (18 px); boundaries at
    # columns 1, 2 and 4, 5 = 24 px. Prediction merges A and B (label 4, 30 px) and keeps C (label 5):
    # its boundary is columns 4, 5 = 12 px.
    ref = _grid([2, 3, 3], [1, 2, 3])
    pred = _grid([5, 3], [4, 5])
    res = zeit.segmentation_accuracy(pred, ref, tolerance=0)
    assert res.boundary.precision == 1 and res.boundary.recall == 12 / 24
    t = res.object_table
    assert list(t["over_seg"]) == [0, 0, 0]
    # A: US = 1 - 12/30 = 0.6, IoU 12/30; B: US = 1 - 18/30 = 0.4, IoU 18/30; C exact.
    assert t["under_seg"].to_list() == pytest.approx([0.6, 0.4, 0.0])
    assert t["iou"].to_list() == pytest.approx([0.4, 0.6, 1.0])
    assert t.loc[1, "d"] == pytest.approx(np.sqrt(0.36 / 2))
    o = res.objects
    assert o.under_seg == pytest.approx(1 / 3)                              # (0.6 + 0.4 + 0) / 3
    assert o.under_seg_w == pytest.approx((12 * 0.6 + 18 * 0.4) / 48)       # 0.3
    assert o.iou_fraction == pytest.approx(2 / 3)                           # B and C reach 0.5
    # one to one by descending IoU: C-5 (1), B-4 (0.6); A has only label 4, taken
    assert list(t["one_to_one"]) == [0, 4, 5] and o.matched_fraction == pytest.approx(2 / 3)


def test_boundary_shifted_one_pixel():
    # Reference boundary: columns 3, 4. Prediction: columns 4, 5. At tolerance 0 only column 4 coincides
    # (6 of 12 pixels both ways); at tolerance 1 column 5 is next to column 4 and column 3 next to column 4.
    ref = _grid([4, 4], [1, 2])
    pred = _grid([5, 3], [1, 2])
    r0 = zeit.segmentation_accuracy(pred, ref, tolerance=0)
    assert (r0.boundary.precision, r0.boundary.recall, r0.boundary.f) == (0.5, 0.5, 0.5)
    r1 = zeit.segmentation_accuracy(pred, ref, tolerance=1)
    assert (r1.boundary.precision, r1.boundary.recall, r1.boundary.f) == (1, 1, 1)
    # a shift of 2 pixels (columns 5, 6 vs 3, 4) is half within tolerance 1: column 5 is next to column 4,
    # column 6 is 2 away; and column 4 is next to column 5, column 3 is 2 away
    r2 = zeit.segmentation_accuracy(_grid([6, 2], [1, 2]), ref, tolerance=1)
    assert r2.boundary.precision == 0.5 and r2.boundary.recall == 0.5
    assert r2.boundary.f == 0.5


def test_property_crossing_and_recall():
    # Properties: 1 = columns 0-4, 2 = columns 5-9 of a 6 x 10 grid (boundary columns 4, 5 = 12 px).
    # Predicted fields: P1 = columns 0-5 (36 px: 30 in property 1, 6 in property 2), P2 = columns 6-9 (24 px).
    # P1's second overlap is 6 = 1/6 of its area: it crosses at min_share 0.1 and not at 0.2. The predicted
    # boundary is columns 5, 6: 6 of the 12 property-line pixels coincide at tolerance 0, all within 1 pixel.
    prop = _grid([5, 5], [1, 2])
    pred = _grid([6, 4], [10, 11])
    r = zeit.segmentation_accuracy(pred, prop, kind="property", tolerance=0, min_share=0.1)
    assert r.boundary.recall == 0.5 and np.isnan(r.boundary.precision) and np.isnan(r.boundary.f)
    assert r.objects is None and r.size_tests is None
    c = r.crossing
    assert (c.n_pred, c.n_crossing) == (2, 1) and c.share == 0.5
    assert c.area_share == pytest.approx(36 / 60)
    r2 = zeit.segmentation_accuracy(pred, prop, kind="property", tolerance=0, min_share=0.2)
    assert r2.crossing.n_crossing == 0 and r2.crossing.area_share == 0
    r3 = zeit.segmentation_accuracy(pred, prop, kind="property", tolerance=1, min_share=0.16)
    assert r3.boundary.recall == 1 and r3.crossing.n_crossing == 1       # 1/6 >= 0.16: counts
    assert "crossing" in repr(r)


def test_region_grows_by_the_tolerance():
    # One reference field of 2 x 4 pixels (rows 2-3, columns 2-5) in a 6 x 8 grid, predicted exactly.
    # Tolerance 0: the region is the field, 8 boundary pixels (each has a neighbour outside the field).
    # Tolerance 1: the ring of 4-neighbours is added: 4 above + 4 below + 2 left + 2 right = 12 more.
    ref = np.zeros((6, 8), dtype=np.int32)
    ref[2:4, 2:6] = 1
    pred = np.where(ref > 0, 4, 0)
    r0 = zeit.segmentation_accuracy(pred, ref, tolerance=0)
    assert (r0.boundary.n_ref, r0.boundary.n_pred, r0.region_pixels) == (8, 8, 8)
    r1 = zeit.segmentation_accuracy(pred, ref, tolerance=1)
    assert (r1.boundary.n_ref, r1.boundary.n_pred, r1.region_pixels) == (20, 20, 20)
    assert r1.boundary.f == 1 and r1.objects.d == 0
    # an explicit region replaces the default: only the left half of the field's columns
    reg = np.zeros((6, 8), dtype=bool)
    reg[:, :4] = True
    r2 = zeit.segmentation_accuracy(pred, ref, tolerance=0, region=reg)
    assert r2.region_pixels == 24
    # boundary pixels in the region: the field's 4 pixels (columns 2-3 of rows 2-3; each has a 0 neighbour above
    # or below) and the 0 pixels next to them: row 1 and row 4 in columns 2-3 (4) and column 1 of rows 2-3 (2);
    # (2, 4) is a 0 neighbour too, but outside the region
    assert r2.boundary.n_ref == 4 + 4 + 2 and r2.boundary.n_pred == 10
    assert r2.object_table.loc[1, "pixels"] == 4          # |r| inside the region
    assert r2.object_table.loc[1, "pred_pixels"] == 8     # |p| in full
    # OS = 1 - 4/4 = 0; US = 1 - 4/8 = 0.5
    assert r2.objects.over_seg == 0 and r2.objects.under_seg == 0.5


def test_geodataframe_reference_equals_raster():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box

    ref = _grid([4, 4], [1, 2])
    pred = _da(_grid([5, 3], [7, 9]))
    x0, y0 = TR.c, TR.f
    gdf = gpd.GeoDataFrame({"id": [10, 20]}, crs=CRS, geometry=[
        box(x0, y0 - 6 * 30, x0 + 4 * 30, y0), box(x0 + 4 * 30, y0 - 6 * 30, x0 + 8 * 30, y0)])
    from_raster = zeit.segmentation_accuracy(pred, _da(ref), tolerance=1)
    from_polygons = zeit.segmentation_accuracy(pred, gdf, tolerance=1)
    assert from_polygons.boundary.equals(from_raster.boundary)
    assert from_polygons.objects.equals(from_raster.objects)
    np.testing.assert_array_equal(from_polygons.object_table.to_numpy(), from_raster.object_table.to_numpy())
    # The other CRS: the polygons are reprojected to the grid's
    other = zeit.segmentation_accuracy(pred, gdf.to_crs("EPSG:4326"), tolerance=1)
    assert other.boundary.equals(from_raster.boundary)
    # hectares and size classes with a CRS: 24 px * 900 m2 = 2.16 ha reference fields; predicted 30 px = 2.7 ha
    # and 18 px = 1.62 ha
    assert from_raster.area_unit == "ha"
    np.testing.assert_allclose(from_raster.object_table["area"], [2.16, 2.16])
    assert list(from_raster.size_classes.index) == ["[0, 5)", "[5, 20)", "[20, 100)", "[100, inf)"]
    assert list(from_raster.size_classes["reference"]) == [2, 0, 0, 0]
    assert list(from_raster.size_classes["predicted"]) == [2, 0, 0, 0]
    custom = zeit.segmentation_accuracy(pred, gdf, size_classes=[0, 2, 3, np.inf])
    assert list(custom.size_classes["reference"]) == [0, 2, 0]      # 2.16 ha is in [2, 3)
    assert list(custom.size_classes["predicted"]) == [1, 1, 0]      # 1.62 in [0, 2), 2.7 in [2, 3)


def test_fields_dataset_boundary_label_is_ignored():
    # Prediction: label 7 = columns 0-2, label 1 = column 3 (the boundary network), label 9 = columns 4-7.
    # With boundary_label = 1 label 1 is unlabelled land: 2 predicted objects; boundaries 7|0 and 0|9 are
    # columns 2, 3, 4 = 18 px, 12 of them (columns 3, 4) in the reference's boundary: precision 12/18,
    # recall 1. Object 1 (24 px): best 7, overlap 18, OS = 1 - 18/24 = 0.25, US = 1 - 18/18 = 0.
    lab = _grid([3, 1, 4], [7, 1, 9])
    ds = xr.Dataset({"labels": _da(lab)}, attrs={"boundary_label": 1})
    ref = _grid([4, 4], [1, 2])
    res = zeit.segmentation_accuracy(ds, ref, tolerance=0)
    assert res.boundary.precision == pytest.approx(12 / 18) and res.boundary.recall == 1
    assert res.objects.n_pred == 2
    assert res.object_table.loc[1, "over_seg"] == 0.25 and res.object_table.loc[1, "under_seg"] == 0
    ds0 = xr.Dataset({"labels": _da(lab)}, attrs={"boundary_label": 0})   # label 1 is a field then
    assert zeit.segmentation_accuracy(ds0, ref, tolerance=0).objects.n_pred == 3


def _clip_case():
    """A 6 x 8 grid, region = columns 0-4. Reference: 3 = column 0 rows 0-4 (5 px, touches the grid's top),
    1 = rows 1-4 columns 1-2 (8 px, complete), 2 = rows 1-4 columns 3-5 (12 px; column 5 is outside the region,
    so 8 px inside). Prediction: 9 and 7 equal to 3 and 1, 8 = rows 1-4 columns 3-7 (20 px: runs out)."""
    ref = np.zeros((6, 8), dtype=np.int32)
    ref[0:5, 0] = 3
    ref[1:5, 1:3] = 1
    ref[1:5, 3:6] = 2
    pred = np.zeros((6, 8), dtype=np.int32)
    pred[0:5, 0] = 9
    pred[1:5, 1:3] = 7
    pred[1:5, 3:8] = 8
    region = np.zeros((6, 8), dtype=bool)
    region[:, :5] = True
    return pred, ref, region


def test_clip_the_prediction_to_the_region():
    pred, ref, region = _clip_case()
    plain = zeit.segmentation_accuracy(pred, ref, tolerance=0, region=region)
    t = plain.object_table
    # reference 2 inside the region: 8 px; its best match 8 is counted in full (20 px): overlap 8, IoU = 8 / 20,
    # OS = 0, US = 1 - 8/20 = 0.6; 8 covers 8 of its 20 px in the region, not more than half: 2 predicted objects
    assert t.loc[2, "pixels"] == 8 and t.loc[2, "pred_pixels"] == 20
    assert t.loc[2, "iou"] == pytest.approx(0.4) and t.loc[2, "under_seg"] == pytest.approx(0.6)
    assert plain.objects.iou_fraction == pytest.approx(2 / 3) and plain.objects.matched_fraction == pytest.approx(2 / 3)
    assert plain.objects.n_pred == 2 and (plain.clip, plain.complete_only) == (False, False)
    # defaults unchanged
    explicit = zeit.segmentation_accuracy(pred, ref, tolerance=0, region=region, clip=False, complete_only=False)
    assert explicit.objects.equals(plain.objects) and explicit.object_table.equals(plain.object_table)

    clipped = zeit.segmentation_accuracy(pred, ref, tolerance=0, region=region, clip=True)
    t = clipped.object_table
    # 8 clipped to columns 3-4: 8 px, IoU 1, US 0: every reference object is found one to one
    assert t.loc[2, "pred_pixels"] == 8 and t.loc[2, "iou"] == 1 and t.loc[2, "under_seg"] == 0
    assert clipped.objects.iou_fraction == 1 and clipped.objects.matched_fraction == 1
    assert clipped.objects.under_seg == 0 and clipped.objects.d == 0 and clipped.objects.n_pred == 3
    assert list(t["one_to_one"]) == [7, 8, 9] and clipped.clip
    # the predicted sizes are the clipped ones (8, 5, 8 px)
    assert clipped.sizes.loc["predicted", "n"] == 3 and clipped.sizes.loc["predicted", "mean"] == pytest.approx(7)
    # the boundaries do not change
    assert clipped.boundary.equals(plain.boundary)
    assert "clipped" in repr(clipped) and "clipped" not in repr(plain)


def test_complete_only_drops_the_cut_reference_fields():
    pred, ref, region = _clip_case()
    res = zeit.segmentation_accuracy(pred, ref, tolerance=0, region=region, complete_only=True)
    # 3 touches the grid's edge and 2 runs out of the region: only 1 is kept (8 px, IoU 1)
    assert list(res.object_table.index) == [1]
    assert (res.objects.n_ref, res.objects.n_ref_all, res.objects.n_unmatched) == (1, 3, 0)
    assert res.objects.iou_fraction == 1 and res.objects.matched_fraction == 1 and res.objects.d == 0
    assert res.sizes.loc["reference", "n"] == 1 and res.complete_only
    assert "complete" in repr(res)
    # the default: nothing dropped, no n_ref_all
    assert "n_ref_all" not in zeit.segmentation_accuracy(pred, ref, tolerance=0, region=region).objects
    # together: the same on the clipped prediction
    both = zeit.segmentation_accuracy(pred, ref, tolerance=0, region=region, clip=True, complete_only=True)
    assert both.objects.n_ref == 1 and both.objects.matched_fraction == 1
    # the default region (the reference grown by the tolerance) holds every reference field: only the edge cuts
    grown = zeit.segmentation_accuracy(pred, ref, tolerance=1, complete_only=True)
    assert list(grown.object_table.index) == [1, 2]
    # none left: NaN, not an error (the one reference field touches the edge)
    only_edge = np.where(ref == 3, 3, 0)
    none = zeit.segmentation_accuracy(pred, only_edge, tolerance=0, region=region, complete_only=True)
    assert none.objects.n_ref == 0 and none.objects.n_ref_all == 1
    assert np.isnan(none.objects.iou_fraction) and np.isnan(none.objects.matched_fraction) and np.isnan(none.objects.d)
    assert len(none.object_table) == 0
    repr(none)


def test_clip_and_complete_only_with_properties():
    pred, ref, region = _clip_case()
    with pytest.raises(ValueError, match="complete_only"):
        zeit.segmentation_accuracy(pred, ref, kind="property", complete_only=True)
    # clip also applies to the crossing: each clipped object overlaps one property only
    r = zeit.segmentation_accuracy(pred, ref, kind="property", tolerance=0, region=region, clip=True)
    assert r.crossing.n_pred == 3 and r.crossing.n_crossing == 0


def test_unmatched_reference_object_and_errors():
    ref = _grid([4, 4], [1, 2])
    pred = np.where(ref == 2, 5, 0)            # nothing predicted over field 1
    res = zeit.segmentation_accuracy(pred, ref, tolerance=0)
    t = res.object_table
    assert t.loc[1, "best"] == 0 and (t.loc[1, "over_seg"], t.loc[1, "under_seg"], t.loc[1, "iou"]) == (1, 1, 0)
    assert res.objects.n_unmatched == 1 and res.objects.iou_fraction == 0.5
    with pytest.raises(ValueError, match="shape"):
        zeit.segmentation_accuracy(pred, ref[:, :6])
    with pytest.raises(ValueError, match="kind"):
        zeit.segmentation_accuracy(pred, ref, kind="x")
    other = TR * TR.translation(30, 0)
    with pytest.raises(ValueError, match="same grid"):
        zeit.segmentation_accuracy(_da(pred), _da(ref, tr=other))
    with pytest.raises(ValueError, match="transform"):
        zeit.segmentation_accuracy(pred, __import__("geopandas").GeoDataFrame(geometry=[]))
