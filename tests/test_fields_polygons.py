"""zeit.fields_to_polygons: the polygons of the field labels, with their area and perimeter."""

import warnings

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit

gpd = pytest.importorskip("geopandas")
pytest.importorskip("pyproj")

TR = from_origin(500000.0, 9000000.0, 30.0, 30.0)


def _labels(crs="EPSG:32722", tr=TR, boundary_label=1):
    """Label 1 a one-pixel network line, 2 a 4 x 3 block, 3 two separate 2 x 2 blocks, 0 elsewhere."""
    lab = np.zeros((10, 12), dtype=np.int32)
    lab[:, 5] = 1
    lab[0:4, 0:3] = 2
    lab[6:8, 0:2] = 3
    lab[6:8, 8:10] = 3
    h, w = lab.shape
    da = xr.DataArray(lab, dims=("y", "x"),
                      coords={"y": tr.f + tr.e * (np.arange(h) + 0.5), "x": tr.c + tr.a * (np.arange(w) + 0.5)})
    if crs is not None:
        da = da.rio.write_crs(crs)
    return xr.Dataset({"labels": da}, attrs={"boundary_label": boundary_label}), lab


def test_projected_areas_are_exact():
    ds, lab = _labels()
    gdf = zeit.fields_to_polygons(ds)
    assert list(gdf.columns) == ["field", "n_pixels", "area_ha", "perimeter_m", "compactness", "geometry"]
    assert list(gdf["field"]) == [2, 3] and gdf.crs == "EPSG:32722"
    assert list(gdf["n_pixels"]) == [12, 8]
    np.testing.assert_allclose(gdf["area_ha"], gdf["n_pixels"] * 900 / 1e4, rtol=1e-12)
    np.testing.assert_allclose(gdf["perimeter_m"], [(4 + 3) * 2 * 30.0, 16 * 30.0], rtol=1e-12)
    a, p = gdf["area_ha"].to_numpy() * 1e4, gdf["perimeter_m"].to_numpy()
    np.testing.assert_allclose(gdf["compactness"], 4 * np.pi * a / p ** 2)


def test_disconnected_label_is_a_multipolygon():
    ds, _ = _labels()
    gdf = zeit.fields_to_polygons(ds).set_index("field")
    assert gdf.loc[3, "geometry"].geom_type == "MultiPolygon" and len(gdf.loc[3, "geometry"].geoms) == 2
    assert gdf.loc[2, "geometry"].geom_type == "Polygon"


def test_boundary_label_dropped_or_kept():
    ds, _ = _labels()
    assert 1 not in set(zeit.fields_to_polygons(ds)["field"])
    kept = zeit.fields_to_polygons(ds, include_boundary=True)
    assert list(kept["field"]) == [1, 2, 3] and kept.set_index("field").loc[1, "n_pixels"] == 10
    ds0, _ = _labels(boundary_label=0)       # no network: label 1 is a field
    assert list(zeit.fields_to_polygons(ds0)["field"]) == [1, 2, 3]


def test_a_hole_is_subtracted():
    ds, lab = _labels()
    lab[1, 1] = 0
    ds["labels"].values[:] = lab
    row = zeit.fields_to_polygons(ds).set_index("field").loc[2]
    assert row["n_pixels"] == 11
    np.testing.assert_allclose(row["area_ha"], 11 * 900 / 1e4)
    np.testing.assert_allclose(row["perimeter_m"], (14 + 4) * 30.0)


def test_feet_are_converted_to_metres():
    ds, _ = _labels(crs="EPSG:2263", tr=from_origin(1000000.0, 200000.0, 100.0, 100.0))   # US feet
    gdf = zeit.fields_to_polygons(ds)
    np.testing.assert_allclose(gdf["area_ha"], gdf["n_pixels"] * (100 * 0.3048006096012192) ** 2 / 1e4, rtol=1e-9)


def test_geographic_area_is_geodesic():
    from pyproj import Geod

    step = 0.001
    ds, _ = _labels(crs="EPSG:4326", tr=from_origin(-54.0, -10.0, step, step))
    gdf = zeit.fields_to_polygons(ds).set_index("field")
    # a 4 x 3 pixel block with its corner at (-54, -10): its area on the WGS84 ellipsoid
    lon = [-54.0, -54.0 + 3 * step, -54.0 + 3 * step, -54.0]
    lat = [-10.0, -10.0, -10.0 - 4 * step, -10.0 - 4 * step]
    area, perim = Geod(ellps="WGS84").polygon_area_perimeter(lon, lat)
    np.testing.assert_allclose(gdf.loc[2, "area_ha"], abs(area) / 1e4, rtol=1e-9)
    np.testing.assert_allclose(gdf.loc[2, "perimeter_m"], perim, rtol=1e-9)
    assert 0 < gdf.loc[2, "compactness"] < 1
    # about 111.3 m per 0.001 degree of latitude, 109.6 m of longitude at 10 degrees south
    assert 12 * 111.0 * 109.0 / 1e4 < gdf.loc[2, "area_ha"] < 12 * 111.5 * 110.0 / 1e4


def test_no_crs_gives_nan_with_a_warning():
    ds, _ = _labels(crs=None)
    with pytest.warns(UserWarning, match="no CRS"):
        gdf = zeit.fields_to_polygons(ds)
    assert gdf.crs is None and gdf["area_ha"].isna().all() and gdf["perimeter_m"].isna().all()
    assert list(gdf["n_pixels"]) == [12, 8] and gdf["compactness"].notna().all()
    # given explicitly, it is measured
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gdf = zeit.fields_to_polygons(ds, crs="EPSG:32722")
    np.testing.assert_allclose(gdf["area_ha"], [12 * 900 / 1e4, 8 * 900 / 1e4])


def test_array_input_with_transform_and_crs():
    _, lab = _labels()
    gdf = zeit.fields_to_polygons(lab, transform=TR, crs="EPSG:32722", include_boundary=True)
    assert list(gdf["field"]) == [1, 2, 3]       # a bare array knows no boundary label
    assert gdf.geometry.total_bounds[1] == pytest.approx(9000000.0 - 10 * 30.0)


def test_exported():
    assert "fields_to_polygons" in zeit.__all__ and callable(zeit.fields_to_polygons)
