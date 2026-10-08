import numpy as np
import pytest
import xarray as xr

from zeit.indices import (
    INDICES, canonical_index, compute_indices, index_array, parse_metrics, resolve_band,
    temporal_metrics,
)


def _bands(**values):
    return lambda role: np.asarray(values[role], dtype=np.float32)


def test_index_values():
    get = _bands(blue=0.04, green=0.08, red=0.05, nir=0.30, swir1=0.15, swir2=0.08)
    expected = {
        "NDVI": (0.30 - 0.05) / (0.30 + 0.05),
        "EVI": 2.5 * (0.30 - 0.05) / (0.30 + 6 * 0.05 - 7.5 * 0.04 + 1),
        "SAVI": 1.5 * (0.30 - 0.05) / (0.30 + 0.05 + 0.5),
        "kNDVI": np.tanh(((0.30 - 0.05) / (0.30 + 0.05)) ** 2),
        "NBR": (0.30 - 0.08) / (0.30 + 0.08),
        "NDMI": (0.30 - 0.15) / (0.30 + 0.15),
        "NDWI": (0.08 - 0.30) / (0.08 + 0.30),
        "MNDWI": (0.08 - 0.15) / (0.08 + 0.15),
    }
    assert set(expected) == set(INDICES)
    for name, value in expected.items():
        np.testing.assert_allclose(index_array(name, get), value, rtol=1e-5, err_msg=name)


def test_index_array_undefined_is_nan():
    get = _bands(red=[0.0, np.nan, 0.1], nir=[0.0, 0.3, 0.3])
    out = index_array("NDVI", get)
    assert out.dtype == np.float32
    assert np.isnan(out[:2]).all() and np.isfinite(out[2])


def test_canonical_index():
    assert canonical_index("ndvi") == "NDVI"
    assert canonical_index("KNDVI") == "kNDVI"
    assert canonical_index("nir") is None


def test_parse_metrics():
    assert parse_metrics(["Median", "P05", "p90", "std", "p5"]) == ["median", "p5", "p90", "std"]
    assert parse_metrics("iqr") == ["iqr"]
    for bad in (["p101"], ["variance"], []):
        with pytest.raises(ValueError):
            parse_metrics(bad)


def test_temporal_metrics_match_numpy():
    rng = np.random.default_rng(0)
    metrics = ["median", "mean", "std", "min", "max", "p10", "p90", "iqr", "count"]
    for t in (1, 2, 7, 30):
        a = rng.random((t, 130, 9)).astype("float32")
        a[a < 0.3] = np.nan
        a[:, 0, 0] = np.nan  # no valid observation
        got = temporal_metrics(a, metrics)
        assert got.shape == (len(metrics), 130, 9) and got.dtype == np.float32
        with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
            ref = [
                np.nanmedian(a, 0), np.nanmean(a, 0), np.nanstd(a, 0), np.nanmin(a, 0),
                np.nanmax(a, 0), np.nanpercentile(a, 10, 0), np.nanpercentile(a, 90, 0),
                np.nanpercentile(a, 75, 0) - np.nanpercentile(a, 25, 0),
            ]
        for k, r in enumerate(ref):
            np.testing.assert_allclose(got[k], r, rtol=1e-5, atol=1e-6, err_msg=metrics[k])
        np.testing.assert_array_equal(got[-1], np.isfinite(a).sum(0))
        assert np.isnan(got[:-1, 0, 0]).all() and got[-1, 0, 0] == 0


def test_temporal_metrics_empty_time_axis():
    out = temporal_metrics(np.empty((0, 3, 4), dtype="float32"), ["median", "count"])
    assert np.isnan(out[0]).all() and (out[1] == 0).all()


def test_resolve_band():
    assert resolve_band("nir", ["B04", "B08", "SCL"]) == "B08"
    assert resolve_band("nir", ["red", "nir08"]) == "nir08"
    assert resolve_band("nir", ["red", "nir", "nir08"]) == "nir"       # Earth Search S2: B08
    assert resolve_band("nir", ["B8A"], {"nir": "B8A"}) == "B8A"
    with pytest.raises(ValueError, match="swir2"):
        resolve_band("swir2", ["red", "nir08"])


def _cube(dtype="float32"):
    rng = np.random.default_rng(1)
    data = rng.uniform(0.01, 0.5, size=(4, 3, 5, 6))
    data[0, 0, 0, 0] = np.nan
    data = np.nan_to_num(data).astype(dtype) if np.dtype(dtype).kind in "iu" else data.astype(dtype)
    return xr.DataArray(
        data, dims=("time", "band", "y", "x"),
        coords={"band": ["red", "nir08", "swir22"],
                "common_name": ("band", ["red", "nir", "swir22"]),
                "time": np.arange(4), "y": np.arange(5), "x": np.arange(6)},
        attrs={"crs": "epsg:32722"},
    )


def test_compute_indices_on_cube():
    cube = _cube()
    out = compute_indices(cube.chunk({"time": 1}), ["ndvi", "NBR"])
    assert out.dims == cube.dims and out.band.values.tolist() == ["NDVI", "NBR"]
    assert out.attrs == cube.attrs and out.dtype == np.float32
    out = out.compute()
    red, nir, swir2 = (cube.sel(band=b).values for b in ("red", "nir08", "swir22"))
    np.testing.assert_allclose(out.sel(band="NDVI").values, (nir - red) / (nir + red), rtol=1e-6)
    np.testing.assert_allclose(out.sel(band="NBR").values, (nir - swir2) / (nir + swir2), rtol=1e-6)
    assert np.isnan(out.values[0, 0, 0, 0])


def test_compute_indices_errors():
    with pytest.raises(ValueError, match="reflectance"):
        compute_indices(_cube("uint16"), ["NDVI"])
    with pytest.raises(ValueError, match="Unknown index"):
        compute_indices(_cube(), ["NDXI"])
    with pytest.raises(ValueError, match="blue"):
        compute_indices(_cube(), ["EVI"])
