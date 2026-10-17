"""zeit.ai.Prithvi: its bands from a cube, windows to classes, the pipeline.

The weights are not downloaded here (pretrained=False); the agreement of the encoder with
the authors', with their weights, is checked against their code outside the test suite."""
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.transform import from_origin

torch = pytest.importorskip("torch")
gpd = pytest.importorskip("geopandas")
from shapely.geometry import box  # noqa: E402

from zeit import ai  # noqa: E402
from zeit.ai.prithvi import MEAN, STD, Prithvi, match_bands  # noqa: E402

HLS = ["blue", "green", "red", "nir", "swir1", "swir2"]


def test_bands_by_name():
    assert match_bands(HLS) == dict(zip(["blue", "green", "red", "nir", "swir1", "swir2"], range(6)))
    # Sentinel-2 on Earth Search: the narrow NIR (nir08, B8A) over the broad one, as HLS
    es = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22"]
    assert match_bands(es)["nir"] == 4 and match_bands(es)["swir1"] == 5
    assert match_bands(["B02", "B03", "B04", "B8A", "B11", "B12"])["nir"] == 3
    with pytest.raises(ValueError, match="six HLS bands"):
        match_bands(["blue", "green", "red"])
    assert match_bands(["b", "g", "r", "n", "s1", "s2"],
                       band_map=dict(zip(["blue", "green", "red", "nir", "swir1", "swir2"],
                                         ["b", "g", "r", "n", "s1", "s2"])))["swir2"] == 5


def test_standardized_as_prithvi_was_trained():
    model = Prithvi(["swir2"] + HLS[:5], n_dates=1, variant="tiny-TL", reflectance_scale=1e4, pretrained=False)
    x = torch.full((1, 6, 1, 16, 16), 0.2)
    x[0, 0] = float("nan")            # swir2 missing: at Prithvi's mean
    out = model.standardize(x)
    assert torch.allclose(out[0, 0], torch.tensor((2000 - MEAN[0]) / STD[0]))   # blue, from the cube's 2nd band
    assert (out[0, 5] == 0).all()
    with pytest.raises(ValueError, match="multiple of 16"):
        model(torch.rand(1, 6, 1, 20, 16))


def test_dates_and_place_for_the_tl_models():
    model = Prithvi(HLS, n_dates=2, variant="tiny-TL", dates=["2021-01-01", "2021-12-31"], latlon=(-10, -55),
                    pretrained=False)
    assert model._temporal.tolist() == [[2021, 1], [2021, 365]] and model._location.tolist() == [-10, -55]
    torch.manual_seed(0)
    x = torch.rand(1, 6, 2, 32, 32) * 0.3
    other = Prithvi(HLS, n_dates=2, variant="tiny-TL", dates=["2021-06-01", "2021-06-15"], latlon=(-10, -55),
                    pretrained=False)
    other.load_state_dict(model.state_dict())
    assert not torch.allclose(model.encode(x), other.encode(x))   # the dates are part of the input
    assert model.encode(x).shape == (1, 1 + 2 * 2 * 2, 192)


def scene(n=64, seed=0):
    rng = np.random.default_rng(seed)
    t = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    water = np.zeros((n, n), dtype=bool)
    water[:, : n // 3] = True
    land = np.array([0.04, 0.07, 0.06, 0.35, 0.20, 0.10])
    lake = np.array([0.06, 0.05, 0.03, 0.02, 0.01, 0.01])
    v = np.where(water[None, None], lake[None, :, None, None], land[None, :, None, None]) + rng.normal(0, 0.01, (2, 6, n, n))
    cube = xr.DataArray(v.astype(np.float32), dims=("time", "band", "y", "x"), coords={
        "time": pd.to_datetime(["2022-05-01", "2022-08-01"]), "band": HLS,
        "y": t.f - 30.0 * (np.arange(n) + 0.5), "x": t.c + 30.0 * (np.arange(n) + 0.5)}).rio.write_crs("EPSG:32721")
    edge = t.c + 30.0 * (n // 3)
    labels = gpd.GeoDataFrame({"class": ["water", "land"]}, crs="EPSG:32721", geometry=[
        box(t.c, t.f - 30.0 * n, edge, t.f), box(edge, t.f - 30.0 * n, t.c + 30.0 * n, t.f)])
    return cube, labels, water


def test_pipeline_train_predict_save_load(tmp_path):
    cube, labels, water = scene()
    s = ai.samples(cube, labels, patch=32, split=0)
    model = ai.train(ai.Prithvi, s, epochs=15, device="cpu", variant="tiny-TL", pretrained=False)
    kw = model.zeit_meta_["model_kwargs"]
    assert model.zeit_meta_["raw"] and kw["reflectance_scale"] == 1e4 and kw["n_dates"] == 2
    assert len(kw["dates"]) == 2 and kw["latlon"] is not None
    out = ai.predict(model, cube, device="cpu", probability=True)
    found = np.array(model.zeit_meta_["classes"])[out.label.values - 1] == "water"
    assert (found == water).mean() > 0.9
    ai.save(model, tmp_path / "prithvi.pt")
    again = ai.load(tmp_path / "prithvi.pt")   # no download: the weights are in the file
    np.testing.assert_allclose(ai.predict(again, cube, device="cpu", probability=True).probability.values,
                               out.probability.values, atol=1e-5)
    with pytest.raises(ValueError, match="multiple of 16"):
        ai.train(ai.Prithvi, ai.samples(cube, labels, patch=24, split=0), epochs=1, device="cpu",
                 variant="tiny-TL", pretrained=False)


def test_geofoundationvit_is_the_old_name():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ai.GeoFoundationViT(HLS, n_dates=1, model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
                                    pretrained=False)
    assert model.variant == "tiny-TL" and any(issubclass(w.category, DeprecationWarning) for w in caught)
    with pytest.raises(ValueError, match="not a Prithvi-EO-2.0 model"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ai.GeoFoundationViT(HLS, n_dates=1, model_id="ibm-nasa-geospatial/Prithvi-100M", pretrained=False)
