"""accuracy(estimator=...) and transitions against R's survey package (tests/data/make_survey_fixtures.R):
the direct, difference and ratio estimators of the area of each class and of each transition between two dates,
from a sample stratified by another map, to 1e-10."""
import json
import os

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "data", "accuracy_survey.json"), encoding="utf-8"))
X0, Y0, RES = 500000.0, 9000000.0, 30.0


def _raster(flat, names=None):
    ny, nx = R["shape"]
    values = np.asarray(flat, dtype=np.uint8).reshape(ny, nx)
    attrs = {}
    if names:
        attrs = {"flag_values": [1, 2, 3], "flag_meanings": " ".join(names)}
    da = xr.DataArray(values, dims=("y", "x"), attrs=attrs,
                      coords={"y": Y0 - RES / 2 - RES * np.arange(ny), "x": X0 + RES / 2 + RES * np.arange(nx)})
    return da.rio.write_crs(32722).rio.write_transform(from_origin(X0, Y0, RES, RES))


def _points(table):
    rows, cols = np.asarray(table["row"]), np.asarray(table["col"])
    x, y = X0 + (cols + 0.5) * RES, Y0 - (rows + 0.5) * RES
    pts = gpd.GeoDataFrame({k: np.asarray(v) for k, v in table.items()}, geometry=gpd.points_from_xy(x, y),
                           crs=32722)
    pts["ref_from"] = [R["names"][c - 1] for c in pts["ref_from"]]
    pts["ref"] = pts["ref_to"] = [R["names"][c - 1] for c in pts["ref_to"]]
    return pts


@pytest.fixture(scope="module")
def maps():
    return (_raster(R["map_from"], R["names"]), _raster(R["map_to"], R["names"]), _raster(R["strata"]))


def _close(ours, theirs, what):
    if theirs is None:
        assert np.isnan(ours), what
    else:
        assert ours == pytest.approx(theirs, rel=1e-10, abs=1e-9), what


def _check(acc, ref, estimator):
    assert set(acc.area.index) == set(ref["classes"]), "the classes"
    for name, r in ref["classes"].items():
        area = acc.area.loc[name]
        assert area.mapped == pytest.approx(r["mapped"], rel=1e-12)
        _close(area.estimate, r[estimator][0], f"{name}: area")
        _close(area.se, r[estimator][1], f"{name}: se")
        for kind, frame in (("users", acc.users), ("producers", acc.producers)):
            _close(frame.loc[name, "estimate"], r[kind][0], f"{name}: {kind}")
            _close(frame.loc[name, "se"], r[kind][1], f"{name}: {kind} se")
    _close(acc.overall.estimate, ref["overall"][0], "overall")
    _close(acc.overall.se, ref["overall"][1], "overall se")


@pytest.mark.parametrize("estimator", ["direct", "difference", "ratio"])
def test_transitions_as_survey(maps, estimator):
    a, b, strata = maps
    acc = zeit.transitions(a, b, _points(R["points"]), strata=strata, estimator=estimator)
    assert acc.estimator == estimator and acc.n == len(R["points"]["row"])
    assert "Forest -> Pasture" in acc.area.index
    _check(acc, R["transitions"], estimator)
    # the confusion of the transitions sums to 1, its counts to the sample
    assert acc.confusion.to_numpy().sum() == pytest.approx(1.0)
    assert int(acc.counts.to_numpy().sum()) == acc.n


@pytest.mark.parametrize("estimator", ["direct", "difference", "ratio"])
def test_classes_strata_not_the_map(maps, estimator):
    _, b, strata = maps
    acc = zeit.accuracy(b, _points(R["points"]), strata=strata, estimator=estimator)
    _check(acc, R["classes"], estimator)


@pytest.mark.parametrize("estimator", ["direct", "difference", "ratio"])
def test_strata_the_map_estimators_agree(maps, estimator):
    _, b, _ = maps
    pts = _points(R["points_by_map"])
    acc = zeit.accuracy(b, pts, estimator=estimator)
    _check(acc, R["by_map"], estimator)
    direct = zeit.accuracy(b, pts)
    np.testing.assert_allclose(acc.area.estimate, direct.area.estimate, rtol=1e-12)


def test_rare_transitions_narrower(maps):
    a, b, strata = maps
    pts = _points(R["points"])
    direct = zeit.transitions(a, b, pts, strata=strata)
    diff = zeit.transitions(a, b, pts, strata=strata, estimator="difference")
    # the change mapped well: its interval several times narrower
    assert diff.area.loc["Forest -> Pasture", "ci"] < 0.3 * direct.area.loc["Forest -> Pasture", "ci"]
    assert "difference" in repr(diff)


def test_codes_and_errors(maps):
    a, b, strata = maps
    pts = _points(R["points"])
    codes = pts.copy()
    codes["ref_from"] = [R["names"].index(n) + 1 for n in pts["ref_from"]]
    codes["ref_to"] = [R["names"].index(n) + 1 for n in pts["ref_to"]]
    by_code = zeit.transitions(a, b, codes, strata=strata)
    by_name = zeit.transitions(a, b, pts, strata=strata)
    np.testing.assert_allclose(by_code.area.estimate, by_name.area.estimate)
    # unnamed maps: the transitions of codes
    plain = zeit.transitions(_raster(R["map_from"]), _raster(R["map_to"]), codes, strata=strata)
    assert "1 -> 2" in plain.area.index
    np.testing.assert_allclose(plain.area.loc["1 -> 2", "estimate"], by_name.area.loc["Forest -> Pasture", "estimate"])
    # a point without a class at one date is left out
    missing = pts.copy()
    missing.loc[0, "ref_to"] = None
    assert zeit.transitions(a, b, missing, strata=strata).n == len(pts) - 1
    with pytest.raises(ValueError, match="estimator"):
        zeit.transitions(a, b, pts, estimator="regression")
    with pytest.raises(ValueError, match="same grid"):
        zeit.transitions(a, b.isel(x=slice(0, 50)), pts)


def test_mapped_inside_the_strata(maps):
    _, b, strata = maps
    # strata over all but the first 10 rows: the map's areas there are the population's
    part = strata.copy(data=np.where(np.arange(strata.sizes["y"])[:, None] < 10, 0, strata.values).astype(np.uint8))
    part = part.rio.write_nodata(0)
    pts = _points(R["points"])
    pts = pts[pts["row"] >= 10]
    acc = zeit.accuracy(b, pts, strata=part, estimator="difference")
    values = np.asarray(R["map_to"]).reshape(R["shape"])[10:]
    for code, name in enumerate(R["names"], start=1):
        assert acc.area.loc[name, "mapped"] == pytest.approx((values == code).sum() * R["pixel_area"])
    assert acc.area.estimate.sum() == pytest.approx(float((part.values > 0).sum()) * R["pixel_area"])


@pytest.mark.parametrize("estimator", ["direct", "difference"])
def test_lazy_and_in_memory_agree(maps, monkeypatch, estimator):
    """Maps of integer classes are read once and counted in memory; lazily (larger than _IN_MEMORY) and as floats
    they give the same numbers."""
    import zeit._accuracy as accuracy_module

    a, b, strata = maps
    pts = _points(R["points"])
    in_memory = zeit.transitions(a.chunk(16), b.chunk(16), pts, strata=strata.chunk(16), estimator=estimator)
    monkeypatch.setattr(accuracy_module, "_IN_MEMORY", 0)
    lazy = zeit.transitions(a.chunk(16), b.chunk(16), pts, strata=strata.chunk(16), estimator=estimator)
    as_floats = zeit.transitions(a.astype(np.float64), b.astype(np.float64), pts, strata=strata.astype(np.float32),
                                 estimator=estimator)
    for other in (lazy, as_floats):
        assert list(other.area.index) == list(in_memory.area.index)
        np.testing.assert_allclose(other.area.to_numpy(), in_memory.area.to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(other.strata.pixels.to_numpy(), in_memory.strata.pixels.to_numpy())
    _check(in_memory, R["transitions"], estimator)


@pytest.mark.parametrize("slab", [1 << 22, 64])
def test_counts_per_row(monkeypatch, slab):
    """The (class, row) counts by each of _count_rows' ways: the values directly, a table of the names, a search."""
    import zeit._accuracy as accuracy_module

    monkeypatch.setattr(accuracy_module, "_SLAB", slab)
    rng = np.random.default_rng(3)
    for names in (np.array([4, 1, 9, 60]), np.array([7, -3, 70_000_000, 2])):
        v = rng.choice(np.append(names, [-1, 1234]), size=(37, 8)).astype(np.int64)
        counts = accuracy_module._count_rows(v, names)
        assert counts.shape == (len(names), v.shape[0])
        np.testing.assert_array_equal(counts, [(v == c).sum(axis=1) for c in names])
