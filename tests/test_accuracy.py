"""sampling_design, stratified_sample and accuracy: Olofsson et al. (2014) and R's sits."""
import json
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit

HERE = os.path.dirname(os.path.abspath(__file__))
SITS = json.load(open(os.path.join(HERE, "data", "accuracy_sits.json"), encoding="utf-8"))


def _class_map(pixels, names, shape, crs=32722, res=30.0):
    """A map whose classes 1..k cover ``pixels`` cells each, row after row."""
    ny, nx = shape
    values = np.repeat(np.arange(1, len(pixels) + 1, dtype=np.uint8), pixels).reshape(ny, nx)
    xs = 500000 + res / 2 + res * np.arange(nx)
    ys = 9000000 - res / 2 - res * np.arange(ny)
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": ys, "x": xs},
                      attrs={"flag_values": list(range(1, len(pixels) + 1)), "flag_meanings": " ".join(names)})
    return da.rio.write_crs(crs).rio.write_transform(from_origin(500000, 9000000, res, res))


def _labelled_sample(case, shape, seed=1):
    """The map of a fixture and a stratified sample whose error matrix is the fixture's."""
    names, matrix = case["labels"], np.array(case["matrix"])
    m = _class_map(case["area"], names, shape)
    design = zeit.sampling_design(m, alloc={n: int(matrix[i].sum()) for i, n in enumerate(names)})
    pts = zeit.stratified_sample(m, design=design, seed=seed)
    pts = pts.sort_values(["stratum", "row", "col"]).reset_index(drop=True)
    pts["ref"] = np.concatenate([np.repeat(names, matrix[i]) for i in range(len(names))])
    return m, pts


@pytest.fixture(scope="module")
def olofsson():
    return _labelled_sample(SITS["olofsson2014"], (2500, 4000))


def test_the_example_of_olofsson_et_al_2014(olofsson):
    m, pts = olofsson
    acc = zeit.accuracy(m, pts)
    assert acc.area_unit == "ha" and acc.n == 640
    # the paper's numbers (section 5, 30 m pixels: 0.09 ha), with its z = 1.96
    published = {"a_deforestation": (21158, 6158), "b_forest_gain": (11686, 3756),
                 "c_stable_forest": (285770, 15510), "d_stable_nonforest": (581386, 16282)}
    for name, (area, ci) in published.items():
        assert acc.area.loc[name, "estimate"] == pytest.approx(area, abs=1)
        assert acc.area.loc[name, "se"] * 1.96 == pytest.approx(ci, abs=1)
    assert acc.overall.estimate == pytest.approx(0.947, abs=5e-4)
    assert acc.overall.se * 1.96 == pytest.approx(0.018, abs=5e-4)
    np.testing.assert_allclose(acc.users.estimate, [0.88, 0.73, 0.93, 0.96], atol=5e-3)
    np.testing.assert_allclose(acc.users.se * 1.96, [0.07, 0.10, 0.04, 0.02], atol=5e-3)
    np.testing.assert_allclose(acc.producers.estimate, [0.75, 0.85, 0.93, 0.96], atol=5e-3)
    assert acc.counts.to_numpy().tolist() == SITS["olofsson2014"]["matrix"]
    assert acc.confusion.to_numpy().sum() == pytest.approx(1.0)
    assert acc.area.mapped.tolist() == [18000, 13500, 288000, 580500]
    text = repr(acc)
    assert "overall: 0.947" in text and "21,158" in text


def _producers_eq7(matrix, area):
    """Olofsson et al. (2014) eq. 7, written out: the variance of the producer's accuracy."""
    n_i = matrix.sum(axis=1)
    w = area / area.sum()
    p = w[:, None] * matrix / n_i[:, None]
    users = np.diag(matrix) / n_i
    prod = np.diag(p) / p.sum(axis=0)
    out = []
    for j in range(len(area)):
        n_dot_j = (w * matrix[:, j] / n_i).sum()
        term1 = w[j] ** 2 * (1 - prod[j]) ** 2 * users[j] * (1 - users[j]) / (n_i[j] - 1)
        others = [w[i] ** 2 * (matrix[i, j] / n_i[i]) * (1 - matrix[i, j] / n_i[i]) / (n_i[i] - 1)
                  for i in range(len(area)) if i != j]
        out.append(np.sqrt((term1 + prod[j] ** 2 * sum(others)) / n_dot_j ** 2))
    return np.array(out)


@pytest.mark.parametrize("case,shape", [("olofsson2014", (2500, 4000)), ("three_classes", (400, 500))])
def test_matches_sits_accuracy(case, shape, olofsson):
    fixture = SITS[case]
    m, pts = olofsson if case == "olofsson2014" else _labelled_sample(fixture, shape)
    acc = zeit.accuracy(m, pts)
    cell = 0.09   # ha per 30 m pixel
    np.testing.assert_allclose(acc.area.estimate / cell, fixture["adjusted_area"], rtol=1e-9)
    np.testing.assert_allclose(acc.area.se / cell, fixture["stderr_area"], rtol=1e-9)
    np.testing.assert_allclose(acc.users.estimate, fixture["user"], rtol=1e-9)
    np.testing.assert_allclose(acc.producers.estimate, fixture["producer"], rtol=1e-9)
    assert acc.overall.estimate == pytest.approx(fixture["overall"], rel=1e-9)
    # sits gives no standard errors of the accuracies: the producer's against eq. 7
    np.testing.assert_allclose(acc.producers.se,
                               _producers_eq7(np.array(fixture["matrix"]), np.array(fixture["area"], float)),
                               rtol=1e-9)


def test_sample_size_and_allocation(olofsson):
    m, _ = olofsson
    # Olofsson et al. (2014), section 5.1: expected user's accuracies 0.7, 0.6, 0.9, 0.95 and a
    # standard error of 0.01 for the overall accuracy -> 641 points
    ua = {"a_deforestation": 0.7, "b_forest_gain": 0.6, "c_stable_forest": 0.9, "d_stable_nonforest": 0.95}
    design = zeit.sampling_design(m, expected_ua=ua, std_error=0.01, min_per_stratum=0)
    assert design.attrs["n"] == 641 and design.n.sum() == 641
    np.testing.assert_allclose(design.n, 641 * design.weight, atol=1)   # proportional
    # at least 50 in the rare strata, the rest proportional, the total kept
    design = zeit.sampling_design(m, expected_ua=ua, std_error=0.01)
    assert design.n.tolist()[:2] == [50, 50] and design.n.sum() == 641
    rest = design.n.to_numpy()[2:]
    assert rest[1] / rest[0] == pytest.approx(0.645 / 0.32, rel=0.01)
    assert zeit.sampling_design(m, n=400, alloc="equal").n.tolist() == [100] * 4
    neyman = zeit.sampling_design(m, n=400, alloc="neyman", expected_ua=ua, min_per_stratum=0)
    s = np.sqrt(np.array(list(ua.values())) * (1 - np.array(list(ua.values()))))
    np.testing.assert_allclose(neyman.n, 400 * neyman.weight * s / (neyman.weight * s).sum(), atol=1)
    assert design.attrs["area_unit"] == "ha" and design.area.sum() == pytest.approx(900000)
    with pytest.raises(ValueError, match="every stratum"):
        zeit.sampling_design(m, alloc={"a_deforestation": 10})
    with pytest.raises(ValueError, match="alloc"):
        zeit.sampling_design(m, alloc="random")


def test_stratified_sample():
    m = _class_map([3000, 17000, 20000], ["a", "b", "c"], (200, 200))
    pts = zeit.stratified_sample(m, n=150, min_per_stratum=30)
    assert len(pts) == 150 and pts.crs.to_epsg() == 32722
    assert pts.groupby("stratum").size().tolist() == [30, 55, 65]   # 120 split 0.425 : 0.5
    # each point is the centre of a pixel of its stratum
    values = m.sel(x=xr.DataArray(pts.geometry.x), y=xr.DataArray(pts.geometry.y)).values
    assert (values == pts.stratum.to_numpy()).all() and pts.stratum_name.tolist()[0] == "a"
    assert not pts.duplicated(["row", "col"]).any()
    # the same points lazily, whatever the chunks; another seed, other points
    lazy = zeit.stratified_sample(m.chunk({"y": 37, "x": 51}), n=150, min_per_stratum=30)
    pd.testing.assert_frame_equal(pd.DataFrame(lazy.drop(columns="geometry")), pd.DataFrame(pts.drop(columns="geometry")))
    other = zeit.stratified_sample(m, n=150, min_per_stratum=30, seed=7)
    assert not (other.row.to_numpy() == pts.row.to_numpy()).all()
    # roughly uniform within a stratum: the points of class c spread over its rows
    rows_c = pts[pts.stratum == 3].row
    assert rows_c.min() < 110 and rows_c.max() > 190


def test_maps_of_events_and_dates():
    rng = np.random.default_rng(0)
    yod = np.zeros((100, 100), dtype=np.uint16)
    yod[:20] = rng.integers(2001, 2020, size=(20, 100))
    date = np.where(yod > 0, (yod.astype("int64") + 1 - 1970).astype("datetime64[Y]").astype("datetime64[ns]"),
                    np.datetime64("NaT", "ns"))
    coords = {"y": 1000 - 15 - 30 * np.arange(100.0), "x": 15 + 30 * np.arange(100.0)}
    events = xr.Dataset({"yod": (("y", "x"), yod), "date": (("y", "x"), date)}, coords=coords).rio.write_crs(32722)
    design = zeit.sampling_design(events, n=100)
    assert design.name.tolist() == ["no change", "change"] and design.n.tolist() == [50, 50]
    pts = zeit.stratified_sample(events, design=design)
    truth = pts.stratum == 1
    # the reference agrees, except 5 change points that are no change; booleans work too
    ref = truth.copy()
    ref.iloc[np.flatnonzero(truth)[:5]] = False
    pts["ref"] = ref
    years = events.yod.values[pts.row, pts.col].astype(int)
    pts["ref_date"] = pd.to_datetime([f"{y + 1 + (i % 3 == 0) * 2}-03-01" if y else None
                                      for i, y in enumerate(years)])
    acc = zeit.accuracy(events, pts, date_tolerance=1)
    assert acc.users.loc["change", "estimate"] == pytest.approx(0.9)
    assert acc.producers.loc["change", "estimate"] == pytest.approx(1.0)
    assert acc.area.loc["change", "mapped"] == pytest.approx(2000 * 0.09)
    # dates: one in three change points of the reference is two years later
    assert acc.date.n == 45 and acc.date.within == pytest.approx(30 / 45)
    assert acc.date.mean_difference == pytest.approx(30 / 45 * 0 + 15 / 45 * 2)
    # strings as well
    pts["ref"] = np.where(ref, "change", "no change")
    assert zeit.accuracy(events, pts).users.loc["change", "estimate"] == pytest.approx(0.9)
    # bins: a stratum per period of yod
    periods = zeit.sampling_design(events, n=90, bins=[2001, 2010, 2020], min_per_stratum=0)
    assert periods.name.tolist() == ["no change", "2001-2009", "2010-2019"]


def test_a_sample_stratified_by_one_map_assesses_another():
    """Stehman (2014): the sample's strata need not be the classes of the map assessed. The
    estimates hit the truth, and the 95% intervals cover it about 95% of the time."""
    rng = np.random.default_rng(5)
    shape = (120, 120)
    truth = np.where(rng.random(shape) < 0.1, 2, 1).astype(np.uint8)             # 10% change
    map_a = np.where(rng.random(shape) < 0.85, truth, 3 - truth).astype(np.uint8)  # the stratified map
    map_b = np.where(rng.random(shape) < 0.75, truth, 3 - truth).astype(np.uint8)  # the one assessed
    coords = {"y": 3000 - 15 - 30 * np.arange(shape[0]), "x": 15 + 30 * np.arange(shape[1])}
    attrs = {"flag_values": [1, 2], "flag_meanings": "stable change"}

    def as_map(v):
        return xr.DataArray(v, dims=("y", "x"), coords=coords, attrs=attrs).rio.write_crs(32722)

    a, b = as_map(map_a), as_map(map_b)
    true_oa = (map_b == truth).mean()
    true_change = (truth == 2).mean() * truth.size * 0.09
    covered, hits = 0, []
    runs = 120
    for seed in range(runs):
        pts = zeit.stratified_sample(a, n=200, min_per_stratum=60, seed=seed)
        pts["ref"] = np.where(truth[pts.row, pts.col] == 2, "change", "stable")
        acc = zeit.accuracy(b, pts, strata=a)
        covered += acc.overall.ci_low <= true_oa <= acc.overall.ci_high
        hits.append(acc.area.loc["change", "estimate"])
    assert 0.88 <= covered / runs <= 1.0
    assert np.mean(hits) == pytest.approx(true_change, rel=0.05)
    # weighing by map B's classes instead would be wrong: the sample was not drawn from them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        biased = zeit.accuracy(b, pts.drop(columns="stratum"))
    assert biased.area.loc["change", "estimate"] != pytest.approx(acc.area.loc["change", "estimate"])


def test_areas_in_a_geographic_crs():
    lats = np.arange(-0.5, -10.5, -1.0)
    da = xr.DataArray(np.ones((10, 4), dtype=np.uint8), dims=("y", "x"),
                      coords={"y": lats, "x": [-60.5, -59.5, -58.5, -57.5]}).rio.write_crs(4326)
    design = zeit.sampling_design(da, n=10)
    r = 6371007.181
    expected = r ** 2 * np.radians(4) * (np.sin(np.radians(0)) - np.sin(np.radians(-10))) / 1e4
    assert design.area.sum() == pytest.approx(expected, rel=1e-9)
    assert design.attrs["area_unit"] == "ha"


def test_samples_and_errors(olofsson, tmp_path):
    m, pts = olofsson
    partial = pts.copy()
    partial.loc[:9, "ref"] = None
    acc = zeit.accuracy(m, partial)
    assert acc.n == 630 and acc.n_unlabelled == 10
    # another CRS, and from a file
    path = tmp_path / "ref.gpkg"
    pts.to_crs(4326).to_file(path)
    assert zeit.accuracy(m, str(path)).overall.estimate == pytest.approx(zeit.accuracy(m, pts).overall.estimate)
    # codes of the map's classes as the reference
    codes = pts.copy()
    codes["ref"] = codes.ref.map({n: i + 1 for i, n in enumerate(SITS["olofsson2014"]["labels"])})
    assert zeit.accuracy(m, codes).overall.estimate == pytest.approx(0.946511888, rel=1e-8)
    with pytest.raises(ValueError, match="'label' column"):
        zeit.accuracy(m, pts, reference="label")
    outside = pts.copy()
    outside.loc[0, "geometry"] = outside.geometry[0].buffer(1).centroid.__class__(0, 0)
    with pytest.warns(UserWarning, match="outside"):
        zeit.accuracy(m, outside)
    # a class the map never gives shows up as a reference class
    extra = pts.copy()
    extra.loc[0, "ref"] = "water"
    acc = zeit.accuracy(m, extra)
    assert "water" in acc.users.index and np.isnan(acc.users.loc["water", "estimate"])
    assert acc.area.loc["water", "mapped"] == 0 and acc.area.loc["water", "estimate"] > 0
    table = acc.to_dataframe()
    assert ("area", "estimate") in table.columns
