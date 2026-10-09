"""zeit.som and zeit.clean_samples: the SOM engine on any cube, georeferenced.

Clusters a cube of three kinds of trajectories; with every pixel as the sample, the codebook
and labels are the engine's (itself MiniSom's, bit for bit); lazy gives what in memory does;
dims, coordinates, NoData and georeferencing; the plot overlay; clean_samples flags
mislabelled points.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

import zeit
from zeit._som import SOM

T, H, W = 24, 30, 40
TRANSFORM = from_origin(500000.0, 4500000.0, 30.0, 30.0)


def scene(seed=0, dtype=np.float32):
    """Three kinds of trajectories in vertical strips: a crop cycle, forest, bare soil."""
    rng = np.random.default_rng(seed)
    steps = np.arange(T)
    kinds = [0.3 + 0.4 * np.sin(2 * np.pi * steps / 12), np.full(T, 0.8), np.full(T, 0.2)]
    truth = np.zeros((H, W), dtype=int)
    truth[:, 15:28] = 1
    truth[:, 28:] = 2
    v = np.stack([kinds[k] for k in truth.ravel()], axis=1).reshape(T, H, W)
    v = v + rng.normal(0, 0.03, v.shape)
    return v.astype(dtype), truth


def cube(values, bands=None, nodata=None):
    coords = {"time": pd.date_range("2020-01-01", periods=values.shape[0], freq="MS"),
              "y": TRANSFORM.f + TRANSFORM.e * (np.arange(values.shape[-2]) + 0.5),
              "x": TRANSFORM.c + TRANSFORM.a * (np.arange(values.shape[-1]) + 0.5)}
    dims = ("time", "y", "x")
    if bands is not None:
        dims = ("time", "band", "y", "x")
        coords["band"] = bands
    da = xr.DataArray(values, dims=dims, coords=coords, name="ndvi").rio.write_crs("EPSG:32633")
    return da.rio.write_nodata(nodata, encoded=False) if nodata is not None else da


def purity(labels, truth):
    return sum(np.bincount(truth[labels == k]).max() for k in np.unique(labels)) / labels.size


def test_clusters_the_three_trajectories():
    v, truth = scene()
    out = zeit.som(cube(v), x=3, y=1)
    assert set(out.data_vars) == {"label", "distance", "prototypes", "n_pixels"}
    assert out.label.dims == ("y", "x") and out.label.dtype == np.uint8
    assert purity(out.label.values, truth) == 1.0 and (out.n_pixels > 0).all()
    assert out.prototypes.dims == ("neuron", "time") and (out.time.values == cube(v).time.values).all()
    assert list(out.neuron.values) == [1, 2, 3] and list(out.i.values) == [0, 1, 2] and list(out.j.values) == [0, 0, 0]
    assert out.label.attrs["flag_meanings"] == "0_0 1_0 2_0"
    assert int(out.n_pixels.sum()) == H * W and out.attrs["n_samples"] == H * W
    assert out.attrs["quantization_error"] < 0.2
    # each prototype is the mean trajectory of its pixels, near enough
    for k in range(1, 4):
        mean = v[:, out.label.values == k].mean(axis=1)
        assert np.abs(out.prototypes.sel(neuron=k).values - mean).max() < 0.05


@pytest.mark.parametrize("init", ["random", "pca", None])
@pytest.mark.parametrize("algorithm,decay", [("batch", "linear_decay_to_zero"), ("online", "linear_decay_to_zero"),
                                             ("online", "asymptotic_decay")])
def test_every_pixel_as_the_sample_is_the_engine(init, algorithm, decay):
    v, _ = scene(seed=1)
    v[3, 2:5, 2:5] = np.nan
    out = zeit.som(cube(v), x=2, y=3, sample=None, init=init, algorithm=algorithm, decay=decay, sigma=0.8, seed=7)
    X = v.reshape(T, -1).T.astype(np.float64)
    ok = np.isfinite(X).all(axis=1)
    engine = SOM(2, 3, T, sigma=0.8, decay_function=decay, random_seed=7)
    if init == "random":
        engine.random_weights_init(X[ok])
    elif init == "pca":
        engine.pca_weights_init(X[ok])
    engine.train(X[ok], 20 if algorithm == "batch" else 20 * int(ok.sum()), algorithm=algorithm)
    assert np.array_equal(out.prototypes.values.reshape(6, T), engine.weights.reshape(6, T))
    labels = np.zeros(H * W)
    labels[ok] = engine.predict(X[ok]) + 1
    assert (out.label.values.ravel() == labels).all()
    assert out.label.values[3, 3] == 0 and np.isnan(out.distance.values[3, 3])
    w = engine.weights.reshape(6, T)
    near = np.linalg.norm(X[ok] - w[engine.predict(X[ok])], axis=1)
    assert np.allclose(out.distance.values.ravel()[ok], near, rtol=1e-6)


def test_the_sample_is_reproducible_and_bounded():
    v, truth = scene()
    a = zeit.som(cube(v), x=3, y=1, sample=200, seed=5)
    b = zeit.som(cube(v), x=3, y=1, sample=200, seed=5)
    assert a.attrs["n_samples"] == 200 and (a.label.values == b.label.values).all()
    assert purity(a.label.values, truth) == 1.0


def test_lazy_and_from_a_file_give_the_result_in_memory(tmp_path):
    v, _ = scene()
    da = cube(v)
    eager = zeit.som(da, x=3, y=2, sample=300)
    lazy = zeit.som(da.chunk({"y": 10, "x": 16}), x=3, y=2, sample=300)
    assert lazy.label.chunks is not None
    assert (lazy.label.values == eager.label.values).all()
    assert np.allclose(lazy.distance.values, eager.distance.values)
    assert (lazy.n_pixels.values == eager.n_pixels.values).all()
    path = tmp_path / "ndvi.tif"
    zeit.save_raster(da, path)
    from_file = zeit.som(path, x=3, y=2, sample=300, chunks="auto")
    assert (from_file.label.values == eager.label.values).all()


def test_a_cube_of_bands_keeps_its_dims_and_names():
    v, truth = scene()
    two = np.stack([v, 1 - v], axis=1)
    out = zeit.som(cube(two, bands=["ndvi", "swir"]), x=3, y=1)
    assert out.prototypes.dims == ("neuron", "time", "band") and list(out.band.values) == ["ndvi", "swir"]
    assert purity(out.label.values, truth) == 1.0


def test_nodata_and_georeferencing(tmp_path):
    v, _ = scene()
    scaled = np.round(v * 10000).astype(np.int16)
    scaled[:, 0, 0] = -9999
    scaled[5, 1, 1] = -9999
    out = zeit.som(cube(scaled, nodata=-9999), x=3, y=1)
    assert out.label.values[0, 0] == 0 and out.label.values[1, 1] == 0 and out.label.values[2, 2] > 0
    assert out.rio.crs == "EPSG:32633" and out.label.rio.nodata == 0
    assert (out.x.values == cube(v).x.values).all()
    written = zeit.save_raster(out, tmp_path / "som")
    assert sorted(p.name for p in written.iterdir()) == ["distance.tif", "label.tif"]  # prototypes have no map
    back = zeit.load_raster(tmp_path / "som" / "label.tif")
    assert (back.values == out.label.values).all()


def test_a_dataset_of_maps():
    v, truth = scene()
    ds = xr.Dataset({"amplitude": cube(v).max("time") - cube(v).min("time"), "mean": cube(v).mean("time")})
    out = zeit.som(ds, x=3, y=1)
    assert out.prototypes.dims == ("neuron", "feature") and list(out.feature.values) == ["amplitude", "mean"]
    assert purity(out.label.values, truth) == 1.0


def test_the_plot_draws_the_prototype_of_the_pixel():
    from zeit._plot import _fit
    from zeit._plot._data import Frames

    v, _ = scene()
    da = cube(v)
    out = zeit.som(da, x=3, y=1)
    series = _fit.pixel_series(Frames(da), col=30, row=4)
    lines = _fit.overlays(out, series, shape=(H, W))
    assert len(lines) == 1 and lines[0]["label"].startswith("SOM neuron")
    k = int(out.label.values[4, 30])
    assert np.allclose(lines[0]["y"], out.prototypes.sel(neuron=k).values)


def test_accessor():
    v, _ = scene()
    da = cube(v)
    assert (da.zeit.som(x=3, y=1).label.values == zeit.som(da, x=3, y=1).label.values).all()


def test_errors():
    v, _ = scene()
    da = cube(v)
    with pytest.raises(ValueError, match="neighborhood"):
        zeit.som(da, neighborhood="square")
    with pytest.raises(ValueError, match="sample"):
        zeit.som(da, sample=0)
    with pytest.raises(ValueError, match="algorithm"):
        zeit.som(da, algorithm="minibatch")
    with pytest.raises(ValueError, match="decay"):
        zeit.som(da, decay="exponential")
    with pytest.raises(ValueError, match="init"):
        zeit.som(da, init="kmeans")
    with pytest.raises(ValueError, match="every feature"):
        zeit.som(da.where(False))


# --- clean_samples ------------------------------------------------------------------


def _samples(truth, flipped, crs="EPSG:32633"):
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    rng = np.random.default_rng(3)
    rows, cols = rng.integers(0, H, 120), rng.integers(0, W, 120)
    names = np.array(["crop", "forest", "bare"])
    labels = names[truth[rows, cols]].astype(object)
    for i in flipped:
        labels[i] = names[(truth[rows[i], cols[i]] + 1) % 3]
    xs = TRANSFORM.c + TRANSFORM.a * (cols + 0.5)
    ys = TRANSFORM.f + TRANSFORM.e * (rows + 0.5)
    points = gpd.GeoDataFrame({"class": labels}, geometry=[Point(a, b) for a, b in zip(xs, ys)], crs="EPSG:32633")
    return points.to_crs(crs)


def test_clean_samples_flags_the_mislabelled_points():
    v, truth = scene()
    flipped = [4, 40, 90]
    points = _samples(truth, flipped)
    out = zeit.clean_samples(cube(v), points, label="class")
    assert list(out.columns[:2]) == ["class", "geometry"]
    assert {"neuron", "neuron_class", "purity", "keep"} <= set(out.columns)
    assert not out.keep.iloc[flipped].any()
    assert out.keep.sum() == len(points) - len(flipped)
    assert (out.neuron > 0).all() and out.purity.between(0, 1).all()


def test_clean_samples_keeps_the_crs_and_marks_points_off_the_data():
    from shapely.geometry import Point

    v, truth = scene()
    points = _samples(truth, [], crs="EPSG:4326")
    far = points.iloc[:1].copy()
    far["geometry"] = [Point(0.0, 0.0)]
    points = pd.concat([points, far], ignore_index=True)
    out = zeit.clean_samples(cube(v), points, label="class", x=3, y=3)
    assert out.crs == points.crs and out.geometry.equals(points.geometry)
    assert out.neuron.iloc[-1] == 0 and not out.keep.iloc[-1] and out.neuron_class.iloc[-1] is None
    assert out.keep.iloc[:-1].all()
    with pytest.raises(ValueError, match="column"):
        zeit.clean_samples(cube(v), points, label="kind")
