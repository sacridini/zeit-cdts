"""Spatial cross-validation folds (zeit.spatial_folds, as blockCV's cv_spatial, cv_buffer and cv_cluster) and the
range of a map's spatial autocorrelation (zeit.autocorrelation_range, as cv_spatial_autocor)."""
import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

import zeit
from zeit._folds import _hexagon_polygon, _hexagons


def _points(xy, crs="EPSG:32722", **columns):
    return gpd.GeoDataFrame(columns, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs)


def _scatter(n=400, seed=0, extent=10_000.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, extent, (n, 2))


def _smooth_map(n=120, pixel=100.0, window=7, seed=1, bands=2):
    """White noise averaged over window x window pixels: correlated up to window pixels apart, and not beyond."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(bands, n + window - 1, n + window - 1))
    c = np.cumsum(np.cumsum(np.pad(noise, ((0, 0), (1, 0), (1, 0))), axis=1), axis=2)
    w = window
    smooth = (c[:, w:, w:] - c[:, :-w, w:] - c[:, w:, :-w] + c[:, :-w, :-w]) / w ** 2
    da = xr.DataArray(smooth, dims=("band", "y", "x"),
                      coords={"band": [f"b{i + 1}" for i in range(bands)],
                              "y": 500_000.0 - pixel * (np.arange(n) + 0.5), "x": 300_000.0 + pixel * (np.arange(n) + 0.5)})
    return da.rio.write_crs("EPSG:32722")


def test_square_blocks_by_hand():
    # a grid of points 100 apart, blocks of 500: 25 points a block
    g = np.arange(0, 2000, 100.0) + 50
    xy = np.array([(x, y) for x in g for y in g])
    f = zeit.spatial_folds(_points(xy), size=500, shape="square", k=4, seed=3)
    assert f.method == "blocks" and len(f) == len(xy) and f.k == 4
    assert len(f.blocks) == 16 and (f.blocks["points"] == 25).all()
    # every point of a block in its fold, and the blocks' polygons hold their points (the grid starts at the
    # points' corner: many are on an edge, touching two blocks)
    joined = gpd.sjoin(f.points.assign(fold=f.folds), f.blocks, predicate="intersects")
    own = joined[joined["fold_left"] == joined["fold_right"]]
    assert set(own.index) == set(range(len(xy)))
    # 16 blocks in 4 folds, at random but balanced: 4 blocks each
    assert (np.bincount(f.folds) == 100).all()
    assert "SpatialFolds(blocks: 400 points, 4 folds of 100-100 points, size 500" in repr(f)


def test_hexagons_are_the_nearest_centres():
    rng = np.random.default_rng(5)
    xy = rng.uniform(-3000, 3000, (3000, 2))
    origin = np.zeros(2)
    cells = _hexagons(xy, 700.0, origin)
    # a hexagonal grid: each point lies in the cell of the nearest centre
    radius = 700.0 / np.sqrt(3)
    centres = {}
    for q in range(-12, 13):
        for r in range(-12, 13):
            centres[(q, r)] = (radius * np.sqrt(3) * (q + r / 2), radius * 1.5 * r)
    keys = list(centres)
    c = np.array([centres[k_] for k_ in keys])
    nearest = np.argmin(((xy[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
    assert [tuple(cells[i]) for i in range(len(xy))] == [keys[j] for j in nearest]
    # and inside its polygon, of the size asked (neighbouring centres 700 apart)
    poly = _hexagon_polygon(1, -2, 700.0, origin)
    cx, cy = centres[(1, -2)]
    assert poly.contains(gpd.points_from_xy([cx], [cy])[0])
    assert poly.area == pytest.approx(np.sqrt(3) / 2 * 700.0 ** 2)


def test_systematic_checkerboard_and_balance():
    g = np.arange(0, 2000, 100.0) + 50
    xy = np.array([(x, y) for x in g for y in g])
    pts = _points(xy)
    f = zeit.spatial_folds(pts, size=500, shape="square", selection="systematic", k=3)
    assert f.folds.tolist() == zeit.spatial_folds(pts, size=500, shape="square", selection="systematic",
                                                   k=3, seed=99).folds.tolist()
    board = zeit.spatial_folds(pts, size=500, shape="square", selection="checkerboard", k=2)
    b = board.blocks
    rows = np.round((b.geometry.centroid.y - 50 - 250) / 500).astype(int)
    cols = np.round((b.geometry.centroid.x - 50 - 250) / 500).astype(int)
    assert ((rows + cols) % 2 == b["fold"]).all()
    with pytest.raises(ValueError, match="k=2"):
        zeit.spatial_folds(pts, size=500, shape="square", selection="checkerboard", k=3)
    with pytest.raises(ValueError, match="fewer than the 5 folds"):
        zeit.spatial_folds(pts, size=5000, k=5)
    # classes in a corner: many draws spread them across the folds better than one
    xy = _scatter(600, seed=2)
    label = np.where((xy[:, 0] < 3000) & (xy[:, 1] < 3000), "rare", "common")
    pts = _points(xy, cls=label)

    def spread(f):
        rare = np.bincount(f.folds[label == "rare"], minlength=5)
        return rare.max() - rare.min()

    one = [spread(zeit.spatial_folds(pts, size=1000, k=5, label="cls", iterations=1, seed=s)) for s in range(10)]
    many = [spread(zeit.spatial_folds(pts, size=1000, k=5, label="cls", iterations=200, seed=s)) for s in range(10)]
    assert np.mean(many) < np.mean(one)


def test_buffer_leaves_out_the_neighbours():
    xy = _scatter(300, seed=4)
    f = zeit.spatial_folds(_points(xy), method="buffer", size=800)
    assert f.method == "buffer" and f.get_n_splits() == 300
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    for i in (0, 17, 299):
        assert set(f.exclude[i].tolist()) == set(np.flatnonzero((d[i] <= 800) & (np.arange(300) != i)).tolist())
    for n_split, (train, test) in enumerate(f.split()):
        assert test.tolist() == [n_split]
        assert (d[n_split, train] > 800).all() and len(train) == 300 - 1 - len(f.exclude[n_split])
        if n_split == 5:
            break
    # the training's DI without the neighbours is larger: the buffer is in train_di
    rng = np.random.default_rng(0)
    pred = np.column_stack([xy / 1000 + rng.normal(scale=0.1, size=xy.shape), rng.normal(size=300)])
    plain = zeit.train_di(pred, folds=np.arange(300))
    buffered = zeit.train_di(pred, folds=f)
    assert (buffered.train_di >= plain.train_di - 1e-12).all() and buffered.threshold > plain.threshold


def test_clusters_and_scikit_learn():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(6)
    centres = np.array([[1000, 1000], [9000, 1000], [5000, 9000]])
    xy = np.concatenate([c + rng.normal(scale=300, size=(50, 2)) for c in centres])
    f = zeit.spatial_folds(_points(xy), method="clusters", k=3, n_jobs=2)
    truth = np.repeat(np.arange(3), 50)
    # the three clusters found, whatever their numbers
    assert all(len(np.unique(f.folds[truth == c])) == 1 for c in range(3))
    assert len(np.unique(f.folds)) == 3
    y = truth % 2
    x = np.column_stack([xy, rng.normal(size=150)])
    pred = cross_val_predict(RandomForestClassifier(n_estimators=10, random_state=0, n_jobs=2), x, y, cv=f)
    assert pred.shape == (150,)
    # in the space of the predictors: two groups of values, wherever the points are
    da = _smooth_map(n=60, bands=1)
    da = da.copy(data=np.where(da.x.values[None, None, :] < da.x.values.mean(), 0.0, 10.0) + 0 * da.values)
    pts = _points(np.column_stack([rng.uniform(da.x.min(), da.x.max(), 80),
                                   rng.uniform(da.y.min(), da.y.max(), 80)]))
    f = zeit.spatial_folds(pts, method="clusters", space="feature", data=da, k=2, n_jobs=2)
    west = pts.geometry.x.to_numpy() < float(da.x.values.mean())
    assert len(np.unique(f.folds[west])) == 1 and len(np.unique(f.folds[~west])) == 1
    assert f.folds[west][0] != f.folds[~west][0]


def test_autocorrelation_range_and_blocks_by_it():
    da = _smooth_map()
    r = zeit.autocorrelation_range(da, sample=2000, n_jobs=2)
    assert list(r.index) == ["b1", "b2"] and (r["pixels"] == 2000).all()
    # correlated up to 7 pixels (700 m): the exponential's practical range is of that order
    assert ((r["range"] > 300) & (r["range"] < 2500)).all(), r
    assert r.attrs["range"] == pytest.approx(r["range"].median())
    v = r.attrs["variograms"]["b1"]
    assert (np.diff(v["distance"]) > 0).all() and v["gamma"].iloc[0] < v["gamma"].iloc[-1]
    # the same with 1 thread, and from the same seed
    again = zeit.autocorrelation_range(da, sample=2000, n_jobs=1)
    assert again["range"].tolist() == r["range"].tolist()
    # the blocks' size from the data
    pts = _points(np.column_stack([np.random.default_rng(0).uniform(300_000, 312_000, 200),
                                   np.random.default_rng(1).uniform(488_000, 500_000, 200)]))
    f = zeit.spatial_folds(pts, data=da, k=3)
    assert f.size == pytest.approx(zeit.autocorrelation_range(da).attrs["range"])
    with pytest.raises(ValueError, match="give size="):
        zeit.spatial_folds(pts, k=3)


def test_geographic_points_are_measured_in_metres():
    lonlat = np.column_stack([np.linspace(-47.9, -47.8, 50), np.linspace(-15.8, -15.7, 50)])
    f = zeit.spatial_folds(_points(lonlat, crs="EPSG:4326"), method="buffer", size=1000)
    assert f.points.crs.is_projected and f.points.crs.to_epsg() == 32723
    # points about 300 m apart along the line: the next ones on each side within 1 km
    assert 4 <= len(f.exclude[25]) <= 8
