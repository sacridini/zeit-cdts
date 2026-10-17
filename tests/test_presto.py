"""zeit.ai.Presto: its inputs from a cube's bands, training and prediction, embeddings.

The weights are not downloaded here (pretrained=False); the agreement with the authors'
encoder, with their weights, is checked against their code outside the test suite."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

torch = pytest.importorskip("torch")
gpd = pytest.importorskip("geopandas")

from zeit import ai  # noqa: E402
from zeit.ai.presto import INPUTS, Presto, match_bands, settings  # noqa: E402

T, H, W = 12, 30, 36
TR = from_origin(500000.0, 8800000.0, 10.0, 10.0)
HLS = ["blue", "green", "red", "nir", "swir1", "swir2"]
NAMES = np.array(["crop", "forest", "pasture"])


def scene(seed=0):
    """Three kinds of land in vertical strips, told apart by their NDVI through the year;
    six bands of reflectance (0-1), monthly."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    ndvi = [0.25 + 0.5 * np.clip(np.sin(np.pi * (t - 2) / 6), 0, None), np.full(T, 0.8), 0.45 + 0.15 * np.cos(2 * np.pi * t / 12)]
    truth = np.zeros((H, W), dtype=int)
    truth[:, 12:24] = 1
    truth[:, 24:] = 2
    nd = np.stack([ndvi[k] for k in truth.ravel()], 1).reshape(T, H, W) + rng.normal(0, 0.03, (T, H, W))
    red = np.clip(0.08 - 0.05 * nd + rng.normal(0, 0.005, (T, H, W)), 0.01, None)
    nir = red * (1 + nd) / (1 - nd)
    v = np.stack([red * 0.6, red * 1.1, red, nir, nir * 0.5, nir * 0.3], 1).astype(np.float32)
    v[4, :, :8, :8] = np.nan   # a cloud
    cube = xr.DataArray(v, dims=("time", "band", "y", "x"),
                        coords={"time": pd.date_range("2022-01-01", periods=T, freq="MS"), "band": HLS,
                                "y": TR.f + TR.e * (np.arange(H) + 0.5), "x": TR.c + TR.a * (np.arange(W) + 0.5)})
    return cube.rio.write_crs("EPSG:32721"), truth


def test_bands_by_name_and_settings_from_the_data():
    assert match_bands(HLS) == {"B2": 0, "B3": 1, "B4": 2, "B8": 3, "B11": 4, "B12": 5}
    assert match_bands(["B02", "B8A", "VV", "dem"]) == {"B2": 0, "B8A": 1, "VV": 2, "elevation": 3}
    assert match_bands(["x", "red"], band_map={"B8": "x"}) == {"B8": 0, "B4": 1}
    with pytest.raises(ValueError, match="none of the bands"):
        match_bands(["ndvi", "evi"])
    times = pd.date_range("2022-01-01", periods=3, freq="MS")
    s = settings(["red", "nir", "VV"], times, low=[0.01, 0.1, 0.0], high=[0.2, 0.5, 0.3])
    assert s["months"] == [1, 2, 3] and s["reflectance_scale"] == 1e4 and s["s1_linear"] is True
    s = settings(["red", "VV"], times, low=[100.0, -20.0], high=[3000.0, -3.0])
    assert s["reflectance_scale"] == 1.0 and s["s1_linear"] is False
    with pytest.raises(ValueError, match="dates"):
        settings(["red"], None, [0.0], [1.0])


def test_inputs_as_presto_normalizes_them():
    model = Presto(["red", "nir", "VV", "elevation"], months=[6, 7], latlon=(-10, -50), reflectance_scale=1e4,
                   s1_linear=True, pretrained=False)
    x = torch.tensor([[[0.05, 0.30, 0.01, 800.0], [np.nan, 0.25, 0.02, np.nan]]])
    inputs, missing = model._prepare(x)
    i = INPUTS.index
    assert torch.isclose(inputs[0, 0, i("B4")], torch.tensor(0.05))                 # 0-1 -> 0-10000 -> / 10000
    assert torch.isclose(inputs[0, 0, i("VV")], torch.tensor(float((10 * np.log10(0.01) + 25) / 25)))   # linear -> dB
    assert torch.isclose(inputs[0, 1, i("elevation")], torch.tensor(800 / 2000))   # static: from the other date
    assert torch.isclose(inputs[0, 0, i("NDVI")], torch.tensor((0.30 - 0.05) / 0.35))
    assert missing[0, 1, i("B4")] and missing[0, 1, i("NDVI")] and not missing[0, 1, i("B8")]
    assert missing[0, :, i("B2")].all() and missing[0, :, i("temperature_2m")].all()    # not in the cube
    assert model.embed(x).shape == (1, 128) and model(x).shape == (1, 2)
    with pytest.raises(ValueError, match="24 dates"):
        Presto(["red"], months=list(range(1, 13)) * 3, pretrained=False)


def test_a_missing_date_is_as_if_it_were_not_there():
    """Presto drops the tokens of what is missing: a date of NaN changes nothing but the
    attention to it (here: the encoding of three dates, one of them all missing, is that of
    the series where the date is masked)."""
    torch.manual_seed(0)
    model = Presto(HLS, months=[1, 2, 3], latlon=(0, 0), pretrained=False).eval()
    x = torch.rand(4, 3, 6) * 0.3
    full = model.embed(x)
    x2 = x.clone()
    x2[:, 1] = float("nan")
    gap = model.embed(x2)
    assert torch.isfinite(gap).all() and not torch.allclose(full, gap)
    x3 = x2.clone()
    x3[:, 1] = 123.0 * float("nan")
    assert torch.equal(model.embed(x3), gap)


def _points(cube, truth, n=150, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = rng.integers(0, H, n), rng.integers(0, W, n)
    return gpd.GeoDataFrame({"class": NAMES[truth[rows, cols]]}, crs="EPSG:32721",
                            geometry=gpd.points_from_xy(cube.x.values[cols], cube.y.values[rows]))


def test_train_predict_save_load(tmp_path):
    cube, truth = scene()
    s = ai.samples(cube, _points(cube, truth), block_size=12)
    lat, lon = s.meta["latlon"]
    assert -80 < lat < 0 and -60 < lon < -50                         # UTM 21S, in the south
    model = ai.train(ai.Presto, s, epochs=3, device="cpu", pretrained=False)
    assert model.zeit_meta_["raw"] and model.zeit_meta_["model_kwargs"]["reflectance_scale"] == 1e4
    assert model.zeit_meta_["model_kwargs"]["months"] == list(range(1, 13))
    out = ai.predict(model, cube, device="cpu", probability=True)
    assert out.label.shape == (H, W) and (out.label.values > 0).all()   # the cloud is one date of twelve
    ai.save(model, tmp_path / "presto.pt")
    again = ai.load(tmp_path / "presto.pt")   # no download: the weights are in the file
    np.testing.assert_allclose(ai.predict(again, cube, device="cpu", probability=True).probability.values,
                               out.probability.values, atol=1e-6)


def _weights_here():
    import os
    from pathlib import Path

    given = os.environ.get("ZEIT_WEIGHTS")
    return ((Path(given) / "default_model.pt").exists() if given else False) or         (Path.home() / ".cache" / "zeit" / "weights" / "default_model.pt").exists()


@pytest.mark.skipif(not _weights_here(), reason="Presto's weights are not on this computer (no download in tests)")
def test_pretrained_learns_from_few_samples():
    """From the authors' weights, 30 epochs on 150 pixels tell the three kinds of land apart
    (from random weights, the same training gets about two thirds right)."""
    cube, truth = scene()
    s = ai.samples(cube, _points(cube, truth), split=0)
    model = ai.train(ai.Presto, s, epochs=30, device="cpu")
    out = ai.predict(model, cube, device="cpu")
    names = np.array(model.zeit_meta_["classes"])
    assert (names[out.label.values - 1] == NAMES[truth]).mean() > 0.95


def test_embeddings_cube():
    cube, truth = scene()
    model = Presto(HLS, months=list(range(1, 13)), reflectance_scale=1e4, pretrained=False)
    emb = ai.presto_embeddings(cube, model=model, device="cpu")
    assert emb.dims == ("time", "band", "y", "x") and emb.shape == (1, 128, H, W)
    assert emb.attrs["embedding_source"] == "presto" and emb.attrs["embedding_variant"].startswith("fine-tuned")
    assert np.isfinite(emb.values).all() and emb.rio.crs == cube.rio.crs
    lazy = ai.presto_embeddings(cube.chunk({"y": 11, "x": 13}), model=model, device="cpu")
    np.testing.assert_allclose(lazy.values, emb.values, rtol=1e-5, atol=1e-5)
    # each pixel has its own latitude and longitude: two pixels of the same series differ
    same = cube.copy()
    same.values[:] = cube.values[:, :, :1, :1]
    e = ai.presto_embeddings(same, model=model, device="cpu").values[0]
    assert not np.allclose(e[:, 0, 0], e[:, -1, -1])
