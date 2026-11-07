"""zeit.zonal_stats and zeit.zonal_areas: a layer's zones measured on maps."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

import zeit


def grid(a, crs="EPSG:32722", res=10.0, x0=500000.0, ytop=8000100.0):
    a = np.asarray(a, dtype=float)
    ny, nx = a.shape
    da = xr.DataArray(a, dims=("y", "x"), coords={"y": ytop - res * (np.arange(ny) + 0.5),
                                                   "x": x0 + res * (np.arange(nx) + 0.5)})
    return da.rio.write_crs(crs) if crs else da


def zones(crs="EPSG:32722"):
    # rows 0-3 x columns 0-4; rows 4-9 x columns 5-9; outside; no geometry
    geoms = [box(500000, 8000060, 500050, 8000100), box(500050, 8000000, 500100, 8000060),
             box(0, 0, 10, 10), None]
    return gpd.GeoDataFrame({"code": ["a", "b", "c", "d"]}, geometry=geoms, crs=crs)


RNG = np.random.default_rng(3)
VALUES = RNG.normal(10, 3, (10, 10))
VALUES[0, :2] = np.nan
CLASSES = RNG.integers(1, 4, (10, 10)).astype(float)
CLASSES[9, :] = np.nan


def test_stats_equal_numpy_in_each_zone():
    t = zeit.zonal_stats(zones(), [grid(VALUES)], labels=["v"])
    assert list(t.long.columns) == ["id", "fid", "date", "n", "mean", "std", "min", "p10", "p25", "median",
                                    "p75", "p90", "max"]
    assert t.transitions is None
    assert len(t.long) == 4 and list(t.long.id) == ["0", "1", "2", "3"]
    for k, (r, c) in enumerate([(slice(0, 4), slice(0, 5)), (slice(4, 10), slice(5, 10))]):
        v = VALUES[r, c].astype("float32").astype("float64")
        v = v[np.isfinite(v)]
        row = t.long.iloc[k]
        assert row.n == len(v) and row.date == "v"
        assert row["mean"] == pytest.approx(v.mean()) and row["std"] == pytest.approx(v.std())
        assert row["min"] == v.min() and row["max"] == v.max() and row["median"] == pytest.approx(np.median(v))
        for q, col in [(10, "p10"), (25, "p25"), (75, "p75"), (90, "p90")]:
            assert row[col] == pytest.approx(np.percentile(v, q))
    assert list(t.long.n[2:]) == [0, 0] and t.long["mean"][2:].isna().all()  # outside; no geometry
    assert list(t.wide.columns) == ["id", "fid", "pixels", "mean_v"]
    assert list(t.wide.pixels) == [20, 30, 0, 0]  # the pixels, valid or not
    assert t.wide.mean_v[0] == pytest.approx(t.long["mean"][0])


def test_stat_choice_labels_id_field_and_window():
    a, b = grid(VALUES), grid(VALUES * 2)
    t = zeit.zonal_stats(zones(), [a, b], labels=["First map", "2nd"], stat="p90", id_field="code")
    assert list(t.wide.columns) == ["id", "fid", "pixels", "p90_first_map", "p90_2nd"]
    assert list(t.wide.id) == ["a", "b", "c", "d"]
    assert t.wide.p90_2nd[0] == pytest.approx(2 * t.wide.p90_first_map[0], rel=1e-6)
    w = zeit.zonal_stats(zones(), a, window=(0, 0, 5, 10))
    assert list(w.wide.pixels) == [20, 0, 0, 0]  # only the first 5 columns left
    with pytest.raises(ValueError):
        zeit.zonal_stats(zones(), a, stat="mode")
    with pytest.raises(ValueError):
        zeit.zonal_stats(zones(), [a, grid(VALUES[:5])])


def test_all_touched_and_zones_in_another_crs_and_from_a_file(tmp_path):
    z = zones().iloc[:2].copy()
    z["geometry"] = [box(500006, 8000066, 500044, 8000094), box(500056, 8000006, 500094, 8000054)]
    inside = zeit.zonal_stats(z, grid(VALUES))
    touched = zeit.zonal_stats(z, grid(VALUES), all_touched=True)
    assert list(inside.wide.pixels) == [6, 12]
    assert list(touched.wide.pixels) == [20, 30]
    path = tmp_path / "z.gpkg"
    z.to_file(path, layer="z", driver="GPKG")
    for source in (z.to_crs(4326), path):
        again = zeit.zonal_stats(source, grid(VALUES), layer="z" if source is path else None)
        assert again.wide.pixels.tolist() == inside.wide.pixels.tolist()
        assert again.long["mean"].tolist() == pytest.approx(inside.long["mean"].tolist())


def test_files_tuples_and_a_stack_as_maps(tmp_path):
    zeit.save_raster(xr.concat([grid(VALUES), grid(VALUES + 1)], dim="band").assign_coords(band=[1, 2]),
                     tmp_path / "m.tif")
    one = zeit.zonal_stats(zones(), [(str(tmp_path / "m.tif"), 2, "second")])
    assert one.long.date[0] == "second" and one.long["mean"][0] == pytest.approx(
        zeit.zonal_stats(zones(), grid(VALUES + 1)).long["mean"][0])
    by_band = zeit.zonal_stats(zones(), tmp_path / "m.tif", bands=1)
    assert by_band.wide.columns[-1] == "mean_m"
    stack = xr.concat([grid(VALUES), grid(VALUES + 1)], dim="time").assign_coords(time=[2000, 2001])
    st = zeit.zonal_stats(zones(), stack)
    assert list(st.long.date[:2]) == ["2000", "2000"] and "mean_2001" in st.wide.columns


def test_areas_by_class_with_names_percent_and_wide():
    names = {1: "forest", 2: "Pasture land", 3: "crop"}
    t = zeit.zonal_areas(zones(), [grid(CLASSES)], names=names, labels=["lc"])
    assert list(t.long.columns) == ["id", "fid", "date", "value", "class", "pixels", "area_m2", "area_ha", "percent"]
    z0 = t.long[t.long.id == "0"]
    sub = CLASSES[0:4, 0:5]
    for v in (1, 2, 3):
        row = z0[z0.value == v].iloc[0]
        assert row.pixels == (sub == v).sum() and row["class"] == names[v]
        assert row.area_m2 == pytest.approx((sub == v).sum() * 100, rel=2e-3)  # UTM scale at the centre
        assert row.area_ha == pytest.approx(row.area_m2 / 1e4, abs=1e-6)
    assert z0.percent.sum() == pytest.approx(100.0)
    # the last row of the zone 1 has no data: left out of the counts
    assert t.long[t.long.id == "1"].pixels.sum() == 25
    assert list(t.wide.columns) == ["id", "fid", "pixels", "ha_forest", "ha_pasture_land", "ha_crop"]
    assert t.wide.ha_forest[0] == pytest.approx(z0[z0.value == 1].area_ha.iloc[0])
    assert t.wide.iloc[2:, 3:].eq(0).all().all()
    assert t.transitions is None
    unnamed = zeit.zonal_areas(zones(), grid(CLASSES), labels=["lc"])
    assert list(unnamed.wide.columns)[3:] == ["ha_1", "ha_2", "ha_3"]


def test_values_are_rounded_halves_away_from_zero_and_years_floor():
    a = np.array([[0.5, 1.5, 2.4, -0.5, 2.5]] * 2)
    z = gpd.GeoDataFrame(geometry=[box(500000, 8000080, 500050, 8000100)], crs="EPSG:32722")
    t = zeit.zonal_areas(z, grid(a, res=10.0, ytop=8000100.0), labels=["m"])
    counts = dict(zip(t.long.value, t.long.pixels))
    assert counts == {-1: 2, 1: 2, 2: 4, 3: 2}
    y = zeit.zonal_areas(z, grid(a), kind="years", labels=["m"])
    assert dict(zip(y.long.value, y.long.pixels)) == {-1: 2, 0: 2, 1: 2, 2: 4}
    assert list(y.long.columns)[4] == "year" and y.long.year.tolist() == ["-1", "0", "1", "2"]
    assert "ha_2" in y.wide.columns
    with pytest.raises(ValueError):
        zeit.zonal_areas(z, grid(a), kind="other")


def test_transitions_and_several_maps():
    before = grid(CLASSES)
    after = grid(np.where(CLASSES == 1, 2, CLASSES))
    names = {1: "forest", 2: "pasture", 3: "crop"}
    t = zeit.zonal_areas(zones(), [before, after], names=names, labels=["2000", "2001"], transitions=(0, 1))
    assert list(t.long.date.unique()) == ["2000", "2001"]
    assert list(t.wide.columns)[3:] == ["ha_forest_2000", "ha_pasture_2000", "ha_crop_2000",
                                         "ha_forest_2001", "ha_pasture_2001", "ha_crop_2001"]
    m = t.transitions
    assert list(m.columns) == ["id", "fid", "from", "from_class", "to", "to_class", "pixels", "area_m2", "area_ha"]
    row = m[(m.id == "0") & (m["from"] == 1)]
    assert list(row.to) == [2] and row.from_class.iloc[0] == "forest" and row.to_class.iloc[0] == "pasture"
    assert row.pixels.iloc[0] == (CLASSES[0:4, 0:5] == 1).sum()
    # what stays is listed too
    assert ((m["from"] == m["to"]) & (m["from"] == 3)).any()
    assert m.pixels.sum() == np.isfinite(CLASSES[0:4, 0:5]).sum() + np.isfinite(CLASSES[4:10, 5:10]).sum()


def test_geodesic_areas_in_degrees_follow_the_latitude():
    a = np.ones((20, 4))
    like = grid(a, crs="EPSG:4326", res=0.05, x0=-50.0, ytop=-10.0)
    z = gpd.GeoDataFrame(geometry=[box(-50.0, -11.0, -49.8, -10.0)], crs=4326)
    t = zeit.zonal_areas(z, like, labels=["m"])
    row = t.long.iloc[0]
    assert row.pixels == 80
    per = 0.05 * 111_195.0
    expect = 80 * per * per * np.cos(np.radians(10.5))
    assert row.area_m2 == pytest.approx(expect, rel=1e-2)
    # the same area as pixel_area gives for those pixels
    pa = zeit.pixel_area(like).values[0:20, 0:4].sum()
    assert row.area_m2 == pytest.approx(pa, rel=1e-4)
    p = zeit.pixel_area(like)
    assert p.values[0, 0] != p.values[-1, 0]


def test_equal_area_and_no_crs_use_the_transform():
    a = np.ones((10, 10))
    z = gpd.GeoDataFrame(geometry=[box(0, 0, 50, 50)], crs=None)
    flat = zeit.zonal_areas(z, grid(a, crs=None, res=5.0, x0=0.0, ytop=50.0))
    assert flat.long.area_m2[0] == flat.long.pixels[0] * 25
    ea = zeit.zonal_areas(gpd.GeoDataFrame(geometry=[box(0, 0, 300, 300)], crs="EPSG:6933"),
                          grid(a, crs="EPSG:6933", res=30.0, x0=0.0, ytop=300.0))
    assert ea.long.area_m2[0] == ea.long.pixels[0] * 900
