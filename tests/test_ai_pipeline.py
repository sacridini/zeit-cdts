"""zeit.ai.samples, train, predict, save and load: from a cube and labelled points or polygons
to a georeferenced map, with the models of zeit.ai. Small models on a synthetic cube of three
kinds of seasons, on the CPU."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

torch = pytest.importorskip("torch")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import box  # noqa: E402

import zeit  # noqa: E402
from zeit import ai  # noqa: E402

T, H, W = 24, 48, 60
TR = from_origin(500000.0, 4500000.0, 30.0, 30.0)
NAMES = np.array(["early", "late", "forest"])
UTAE_SMALL = dict(encoder_widths=[8, 16, 16], decoder_widths=[8, 16, 16], d_model=16, n_head=4, d_k=4)


def scene(seed=0):
    """An early and a late crop season and forest, in vertical strips; ndvi and red."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    kinds = [0.3 + 0.4 * np.sin(2 * np.pi * t / 24), 0.3 + 0.4 * np.sin(2 * np.pi * (t - 8) / 24), np.full(T, 0.8)]
    truth = np.zeros((H, W), dtype=int)
    truth[:, 20:40] = 1
    truth[:, 40:] = 2
    ndvi = np.stack([kinds[k] for k in truth.ravel()], axis=1).reshape(T, H, W) + rng.normal(0, 0.05, (T, H, W))
    red = 0.2 - 0.1 * ndvi + rng.normal(0, 0.01, (T, H, W))
    v = np.stack([ndvi, red], axis=1).astype(np.float32)
    v[3, :, :10, :10] = np.nan          # a cloud
    v[:, :, 0, W - 1] = np.nan          # a pixel without data
    cube = xr.DataArray(v, dims=("time", "band", "y", "x"),
                        coords={"time": pd.date_range("2021-01-01", periods=T, freq="SMS"), "band": ["ndvi", "red"],
                                "y": TR.f + TR.e * (np.arange(H) + 0.5), "x": TR.c + TR.a * (np.arange(W) + 0.5)})
    return cube.rio.write_crs("EPSG:32633"), truth


def points(cube, truth, n=240, seed=0, crs="EPSG:32633"):
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(0, H, n), rng.integers(0, W - 1, n)
    gdf = gpd.GeoDataFrame({"class": NAMES[truth[rows, cols]]},
                           geometry=gpd.points_from_xy(cube.x.values[cols], cube.y.values[rows]), crs="EPSG:32633")
    return gdf.to_crs(crs)


def accuracy(result, model, truth):
    labels = result.label.values
    names = np.array(model.zeit_meta_["classes"])
    has = labels > 0
    return float((names[labels[has] - 1] == NAMES[truth[has]]).mean())


@pytest.fixture(scope="module")
def data():
    cube, truth = scene()
    return cube, truth, points(cube, truth)


@pytest.fixture(scope="module")
def tempcnn(data):
    cube, truth, pts = data
    s = ai.samples(cube, pts, block_size=12)
    return ai.train(ai.TempCNN, s, epochs=25, device="cpu"), s


# --- samples ------------------------------------------------------------------------


def test_pixel_samples(data):
    cube, truth, pts = data
    s = ai.samples(cube, pts.to_crs("EPSG:4326"), label="class", block_size=12)   # reprojected
    assert s.classes == ["early", "forest", "late"] and s.patch is None
    assert s.meta["bands"] == ["ndvi", "red"] and len(s.meta["times"]) == T
    assert s.meta["positions"][:3] == [0.0, 14.0, 31.0]                          # days since the first date
    assert len(s.train) + len(s.val) == len(s) and 0.1 < len(s.val) / len(s) < 0.35
    item = s[0]
    assert item["x"].shape == (T, 2) and item["y"].dtype == torch.long and item["positions"].shape == (T,)
    assert torch.isfinite(item["x"]).all()                                       # missing values: 0
    # the normalization comes from the training samples only
    train_ndvi = s.X[~s.is_val][:, :, 0]
    assert np.isclose(s.meta["norm_low"][0], np.nanquantile(train_ndvi, 0.02))


def test_block_split_keeps_blocks_whole(data):
    cube, truth, pts = data
    s = ai.samples(cube, pts, block_size=12)
    from zeit.ai.pipeline import _rasterize
    rows, cols = np.nonzero(_rasterize(pts.geometry, np.zeros(len(pts), int), TR, (H, W)) >= 0)
    blocks_val = {(r // 12, c // 12) for r, c, v in zip(rows, cols, s.is_val) if v}
    blocks_train = {(r // 12, c // 12) for r, c, v in zip(rows, cols, s.is_val) if not v}
    assert blocks_val and blocks_train and not blocks_val & blocks_train
    random = ai.samples(cube, pts, split_by="random", split=0.25)
    assert abs(random.is_val.mean() - 0.25) < 0.02
    assert not ai.samples(cube, pts, split=0).val.indices.size


def test_patch_samples_and_polygons(data):
    cube, truth, _ = data
    polys = gpd.GeoDataFrame({"class": ["early", "forest"]},
                             geometry=[box(TR.c + 30 * 2, TR.f - 30 * 40, TR.c + 30 * 18, TR.f - 30 * 4),   # 16 x 36
                                       box(TR.c + 30 * 45, TR.f - 30 * 10, TR.c + 30 * 50, TR.f - 30 * 5)],  # 5 x 5
                             crs="EPSG:32633")
    s = ai.samples(cube, polys, patch=16, split=0)
    assert len(s) == 1 * 3 + 1                    # the large polygon tiled, the small one centred
    item = s[0]
    assert item["x"].shape == (T, 2, 16, 16) and item["y"].shape == (16, 16)
    assert set(np.unique(s.y)) == {-1, 0, 1}
    pixel = ai.samples(cube, polys, split=0)        # pixel mode: every pixel the polygons cover
    assert len(pixel) == 16 * 36 + 5 * 5


# --- train and predict --------------------------------------------------------------


def test_tempcnn_from_cube_to_map(data, tempcnn):
    cube, truth, _ = data
    model, s = tempcnn
    assert set(model.zeit_history_[0]) == {"epoch", "train_loss", "val_loss", "val_accuracy"}
    assert model.zeit_meta_["model"] == "TempCNN" and model.zeit_meta_["model_kwargs"]["n_times"] == T
    out = ai.predict(model, cube, probability=True, device="cpu")
    assert out.label.dims == ("y", "x") and out.rio.crs == "EPSG:32633" and (out.x.values == cube.x.values).all()
    assert out.label.attrs["flag_meanings"] == "early forest late"
    assert accuracy(out, model, truth) > 0.95
    assert out.label.values[0, W - 1] == 0 and np.isnan(out.probability.values[:, 0, W - 1]).all()
    has = out.label.values > 0
    assert np.allclose(out.probability.values[:, has].sum(axis=0), 1, atol=1e-5)
    lazy = ai.predict(model, cube.chunk({"y": 16, "x": 25}), device="cpu")
    assert lazy.label.chunks is not None and (lazy.label.values == out.label.values).all()
    assert (zeit.classify(cube, model).label.values == out.label.values).all()   # one classify


def test_lighttae(data):
    cube, truth, pts = data
    s = ai.samples(cube, pts, block_size=12)
    model = ai.train(ai.LightTAE, s, epochs=25, device="cpu")
    assert accuracy(ai.predict(model, cube, device="cpu"), model, truth) > 0.95


def strips():
    """The three strips as polygons: dense labels, as segmentation models are trained on."""
    edges = [(0, 20), (20, 40), (40, W)]
    return gpd.GeoDataFrame({"class": list(NAMES)}, crs="EPSG:32633",
                            geometry=[box(TR.c + 30 * a, TR.f - 30 * H, TR.c + 30 * b, TR.f) for a, b in edges])


def test_utae_windows_cover_the_image_and_lazy_is_the_same(data):
    cube, truth, _ = data
    s = ai.samples(cube, strips(), patch=12, block_size=12)
    model = ai.train(ai.UTAE, s, epochs=30, device="cpu", **UTAE_SMALL)
    out = ai.predict(model, cube, device="cpu", overlap=0.5)
    assert (out.label.values > 0).sum() == H * W - 1
    # a sanity check (chance is 1/3): a short training of a small U-TAE lands between 0.78
    # and 1.0 with the seed, and arithmetic differs by platform (0.81 on macOS arm64)
    assert accuracy(out, model, truth) > 0.7
    lazy = ai.predict(model, cube.chunk({"y": 20, "x": 25}), device="cpu", overlap=0.5)
    assert (lazy.label.values == out.label.values).all()
    # UTAE takes other dates: positions come from the cube's own
    later = cube.isel(time=slice(0, 20)).assign_coords(time=cube.time.values[:20] + np.timedelta64(365, "D"))
    assert ai.predict(model, later, device="cpu").label.shape == (H, W)


def test_siamese_change_detection():
    rng = np.random.default_rng(1)
    n = 32
    t = from_origin(500000.0, 4500000.0, 10.0, 10.0)
    coords = {"band": ["b", "g", "r", "n"], "y": t.f - 10 * (np.arange(n) + 0.5), "x": t.c + 10 * (np.arange(n) + 0.5)}
    before = rng.normal(0.3, 0.02, (4, n, n)).astype(np.float32)
    after = before.copy()
    after[:, 8:20, 6:22] += 0.3
    pair = tuple(xr.DataArray(v, dims=("band", "y", "x"), coords=coords).rio.write_crs("EPSG:32633")
                 for v in (before, after))
    changed = box(t.c + 10 * 6, t.f - 10 * 20, t.c + 10 * 22, t.f - 10 * 8)
    labels = gpd.GeoDataFrame({"class": ["change", "same"]},
                              geometry=[changed, box(t.c, t.f - 10 * n, t.c + 10 * n, t.f).difference(changed)],
                              crs="EPSG:32633")
    s = ai.samples(pair, labels, patch=16, split=0)
    assert s.meta["positions"] == [0.0, 1.0] and s.meta["times"] is None
    model = ai.train(ai.SiameseChangeDetector, s, epochs=30, batch_size=4, device="cpu")
    out = ai.predict(model, pair, device="cpu")
    truth = np.zeros((n, n), dtype=bool)
    truth[8:20, 6:22] = True
    found = np.array(model.zeit_meta_["classes"])[out.label.values - 1] == "change"
    assert (found == truth).mean() > 0.95


def test_vit_offline(monkeypatch, data):
    from zeit.ai import foundation

    def offline(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(foundation.AutoModel, "from_pretrained", offline)
    cube, truth, pts = data
    six = xr.concat([cube] * 3, dim="band").assign_coords(band=list("abcdef")).isel(time=slice(0, 3))
    s = ai.samples(six, pts.iloc[:20], patch=16, split=0)
    model = ai.train(ai.GeoFoundationViT, s, epochs=1, device="cpu")
    assert ai.predict(model, six, device="cpu").label.shape == (H, W)


def test_save_and_load(tmp_path, data, tempcnn):
    cube, truth, _ = data
    model, s = tempcnn
    ai.save(model, tmp_path / "tempcnn.pt")
    again = ai.load(tmp_path / "tempcnn.pt")
    assert again.zeit_meta_ == model.zeit_meta_ and again.zeit_history_ == model.zeit_history_
    assert (ai.predict(again, cube, device="cpu").label.values == ai.predict(model, cube, device="cpu").label.values).all()
    built = ai.TempCNN(in_channels=2, n_times=T, num_classes=3)
    trained = ai.train(built, s, epochs=1, device="cpu")
    ai.save(trained, tmp_path / "instance.pt")
    with pytest.raises(ValueError, match="model="):
        ai.load(tmp_path / "instance.pt")
    assert ai.load(tmp_path / "instance.pt", model=ai.TempCNN(in_channels=2, n_times=T, num_classes=3)).zeit_meta_


def test_errors(data, tempcnn):
    cube, truth, pts = data
    model, s = tempcnn
    with pytest.raises(ValueError, match="patches"):
        ai.train(ai.UTAE, s, epochs=1, device="cpu")
    with pytest.raises(ValueError, match="pixels"):
        ai.train(ai.TempCNN, ai.samples(cube, pts, patch=8, split=0), epochs=1, device="cpu")
    with pytest.raises(ValueError, match="multiple of 4"):
        ai.train(ai.UTAE, ai.samples(cube, pts, patch=10, split=0), epochs=1, device="cpu", **UTAE_SMALL)
    with pytest.raises(ValueError, match="24 dates"):
        ai.predict(model, cube.isel(time=slice(0, 20)), device="cpu")
    with pytest.raises(ValueError, match="lacks bands"):
        ai.predict(model, cube.sel(band=["ndvi"]).assign_coords(band=["evi"]).pipe(
            lambda c: xr.concat([c, c.assign_coords(band=["nir"])], dim="band")), device="cpu")
    with pytest.raises(ValueError, match="two classes"):
        ai.samples(cube, pts[pts["class"] == "forest"])
    with pytest.raises(ValueError, match="time"):
        ai.samples(cube.isel(time=0), pts)
    with pytest.raises(ValueError, match="zeit_meta_"):
        ai.predict(ai.TempCNN(in_channels=2, n_times=T, num_classes=3), cube)
    with pytest.raises(ValueError, match="work with"):
        ai.train(torch.nn.Linear, s)


def test_stac_cube_dataset_covers_the_edges():
    cube, _ = scene()
    ds = ai.STACCubeDataset(cube, patch_size=20, stride=20)
    assert ds.y_starts == [0, 20, 28] and ds.x_starts == [0, 20, 40]
    item = ds[len(ds) - 1]
    assert item["x"].shape == (T, 2, 20, 20) and (item["row"], item["col"]) == (28, 40)
    assert item["positions"][:3].tolist() == [0.0, 14.0, 31.0]
    corner = ds[2]                                                   # row 0, the last column
    assert (corner["row"], corner["col"]) == (0, 40)
    assert corner["valid"].shape == (T, 20, 20) and not corner["valid"][:, 0, 19].any()   # the pixel without data
    small = ai.STACCubeDataset(cube.isel(y=slice(0, 10), x=slice(0, 10)), patch_size=16)
    assert len(small) == 1 and torch.isnan(small[0]["x"][:, :, 12, 12]).all()      # padded with NaN
