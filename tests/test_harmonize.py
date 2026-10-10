"""zeit.harmonize: Sentinel-2 (and optionally ETM+/TM) on Landsat 8's OLI scale."""
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._harmonize import HLS, ROY_2016, sensor_of

BANDS = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22", "rededge1"]
ROLES = ["coastal", "blue", "green", "red", "nir", "swir1", "swir2"]


def _cube(platforms, h=4, w=5, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.02, 0.45, (len(platforms), len(BANDS), h, w)).astype(dtype)
    da = xr.DataArray(data, dims=("time", "band", "y", "x"), name="reflectance",
                      coords={"time": pd.date_range("2020-01-01", periods=len(platforms), freq="10D"),
                              "band": BANDS, "y": np.arange(h)[::-1] * 30.0 + 15, "x": np.arange(w) * 30.0 + 15,
                              "platform": ("time", list(platforms))})
    return da.rio.write_crs(32722)


def test_tables_are_the_published_ones():
    # HLS v2.0 User Guide, Table 5 (S2A/S2B also v1.4); Roy et al. 2016, Table 2.
    assert HLS["S2A"]["nir"] == (0.9983, -0.0001) and HLS["S2B"]["swir2"] == (0.9867, 0.0004)
    assert HLS["S2C"]["blue"] == (0.9851, -0.0027) and HLS["S2A"]["blue"] == HLS["S2B"]["blue"]
    assert ROY_2016["ols"]["nir"] == (0.8462, 0.0412) and ROY_2016["rma"]["swir1"] == (1.0171, -0.0030)
    assert all(set(t) == set(ROLES) for t in HLS.values())


@pytest.mark.parametrize("name, sensor", [
    ("sentinel-2a", "S2A"), ("Sentinel-2B", "S2B"), ("sentinel-2c", "S2C"), ("S2A", "S2A"),
    ("sentinel-2", "S2"), ("landsat-8", "OLI"), ("LANDSAT_9", "OLI"), ("landsat-7", "ETM"),
    ("LANDSAT_5", "TM"), ("LC08", "OLI"), ("LE07", "ETM"), ("LT05", "TM"), ("modis", None),
])
def test_sensor_names(name, sensor):
    assert sensor_of(name) == sensor


def test_sentinel2_goes_to_oli_and_landsat_stays():
    cube = _cube(["sentinel-2a", "sentinel-2b", "landsat-8", "landsat-9", "sentinel-2c"])
    out = zeit.harmonize(cube)
    for t, unit in ((0, "S2A"), (1, "S2B"), (4, "S2C")):
        for role, band in zip(ROLES, BANDS):
            a, b = HLS[unit][role]
            np.testing.assert_allclose(out.isel(time=t).sel(band=band), a * cube.isel(time=t).sel(band=band) + b,
                                       rtol=1e-6, atol=1e-7)
    xr.testing.assert_equal(out.isel(time=[2, 3]), cube.isel(time=[2, 3]).assign_attrs(out.attrs))
    xr.testing.assert_equal(out.sel(band="rededge1"), cube.sel(band="rededge1").assign_attrs(out.attrs))   # no line
    assert out.dtype == np.float32 and out.dims == cube.dims and out.rio.crs == cube.rio.crs
    assert "HLS" in out.attrs["harmonized"] and "nir=nir08" in out.attrs["harmonized_bands"]


def test_the_step_between_sensors_goes_away():
    """OLI reflectance seen by MSI (the inverse of the HLS line) comes back to OLI."""
    truth = _cube(["landsat-8"] * 6, seed=3)
    platforms = ["landsat-8", "sentinel-2a", "landsat-8", "sentinel-2b", "sentinel-2a", "landsat-9"]
    msi = truth.copy().assign_coords(platform=("time", platforms))
    for t, p in enumerate(platforms):
        unit = sensor_of(p)
        if unit.startswith("S2"):
            for role, band in zip(ROLES, BANDS):
                a, b = HLS[unit][role]
                msi.loc[{"band": band}][t] = (truth.sel(band=band)[t] - b) / a
    assert float(abs(msi - truth).max()) > 0.004                      # the step
    np.testing.assert_allclose(zeit.harmonize(msi).values, truth.values, atol=1e-6)


def test_etm_and_tm_only_when_asked():
    cube = _cube(["landsat-7", "landsat-5", "landsat-8"])
    assert float(abs(zeit.harmonize(cube) - cube).max()) == 0
    for method in ("rma", "ols"):
        out = zeit.harmonize(cube, etm=method)
        for t in (0, 1):
            a, b = ROY_2016[method]["red"]
            np.testing.assert_allclose(out.isel(time=t).sel(band="red"), a * cube.isel(time=t).sel(band="red") + b,
                                       rtol=1e-6)
        np.testing.assert_allclose(out.sel(band="coastal"), cube.sel(band="coastal"))   # no ETM+ coastal band
        assert "Roy" in out.attrs["harmonized"]
    with pytest.raises(ValueError, match="etm"):
        zeit.harmonize(cube, etm="nope")


def test_integers_scaled_by_10000_keep_type_and_nodata():
    cube = (_cube(["sentinel-2a", "landsat-8"]) * 10000).round().astype(np.uint16)
    cube[0, 1, 0, 0] = 0   # NoData
    out = zeit.harmonize(cube)
    assert out.dtype == np.uint16 and int(out[0, 1, 0, 0]) == 0
    a, b = HLS["S2A"]["blue"]
    expected = np.round(a * cube[0, 1].values.astype(float) + b * 10000)
    np.testing.assert_array_equal(out[0, 1].values.ravel()[1:], expected.ravel()[1:])
    np.testing.assert_array_equal(out[1].values, cube[1].values)


def test_lazy_stays_lazy_and_matches():
    cube = _cube(["sentinel-2a", "sentinel-2b", "landsat-8"], h=16, w=16)
    lazy = zeit.harmonize(cube.chunk({"time": 1, "y": 8, "x": 8}))
    assert lazy.chunks is not None
    np.testing.assert_allclose(lazy.compute().values, zeit.harmonize(cube).values)


def test_sensor_argument_and_single_image():
    cube = _cube(["x"] * 2).drop_vars("platform")
    with pytest.raises(ValueError, match="sensor"):
        zeit.harmonize(cube)
    one = zeit.harmonize(cube, sensor="sentinel-2b")
    per_date = zeit.harmonize(cube, sensor=["sentinel-2b", "landsat-8"])
    np.testing.assert_allclose(one[0], per_date[0])
    np.testing.assert_allclose(per_date[1], cube[1])
    image = zeit.harmonize(cube.isel(time=0, drop=True), sensor="S2B")
    np.testing.assert_allclose(image, one[0])
    with pytest.raises(ValueError, match="dates"):
        zeit.harmonize(cube, sensor=["S2A"])
    with pytest.raises(ValueError, match="unknown sensor"):
        zeit.harmonize(cube, sensor="modis")


def test_broad_nir_of_sentinel2_is_left_with_a_warning():
    cube = _cube(["sentinel-2a", "landsat-7"]).assign_coords(band=["coastal", "blue", "green", "red", "nir",
                                                                   "swir16", "swir22", "rededge1"])
    with pytest.warns(UserWarning, match="B08"):
        out = zeit.harmonize(cube, etm="rma")
    np.testing.assert_allclose(out[0].sel(band="nir"), cube[0].sel(band="nir"))       # S2's B08: as it was
    a, b = ROY_2016["rma"]["nir"]
    np.testing.assert_allclose(out[1].sel(band="nir"), a * cube[1].sel(band="nir") + b, rtol=1e-6)   # ETM+'s NIR
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = zeit.harmonize(cube, bands={"nir": "nir"})                               # "this one is B8A"
    a, b = HLS["S2A"]["nir"]
    np.testing.assert_allclose(out[0].sel(band="nir"), a * cube[0].sel(band="nir") + b, rtol=1e-6)


def test_refusals():
    cube = _cube(["sentinel-2a"])
    with pytest.raises(ValueError, match="already harmonized"):
        zeit.harmonize(zeit.harmonize(cube))
    raw = (cube * 10000).astype(np.uint16).assign_coords(scale=(("time", "band"), np.full((1, 8), 1e-4)))
    with pytest.raises(ValueError, match="digital numbers"):
        zeit.harmonize(raw)
    with pytest.raises(ValueError, match="band dimension"):
        zeit.harmonize(cube.isel(band=0))
    with pytest.raises(ValueError, match="role"):
        zeit.harmonize(cube, bands={"vre": "rededge1"})
    with pytest.warns(UserWarning, match="unknown unit"):
        zeit.harmonize(cube.assign_coords(platform=("time", ["sentinel-2"])))


def test_cli(tmp_path):
    import subprocess
    import sys

    cube = (_cube(["sentinel-2b", "landsat-8", "sentinel-2b"]) * 10000).round().astype(np.uint16).drop_vars("platform")
    path = zeit.save_raster(cube, tmp_path / "mixed.tif")
    out = tmp_path / "out"
    run = subprocess.run([sys.executable, "-m", "zeit.cli", "harmonize", str(path), str(out),
                          "--sensor", "sentinel-2b", "landsat-8", "sentinel-2b"], capture_output=True, text=True,
                         timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    written = zeit.load_raster(out / "harmonized.tif")
    expected = zeit.harmonize(cube, sensor=["sentinel-2b", "landsat-8", "sentinel-2b"])
    np.testing.assert_array_equal(written.values, expected.values)


def test_a_list_of_cubes_becomes_one_series():
    """As two STAC collections come: other band names, other per-date coordinates."""
    s2 = _cube(["sentinel-2a", "sentinel-2b"], seed=1).isel(band=slice(1, 7))
    s2 = s2.assign_coords(band=["B02", "B03", "B04", "B8A", "B11", "B12"], granule=("time", ["a", "b"]))
    s2 = s2.assign_coords(time=pd.to_datetime(["2020-01-03", "2020-01-20"]))
    landsat = _cube(["landsat-8", "landsat-7", "landsat-9"], seed=2).sel(
        band=["blue", "green", "red", "nir08", "swir16", "swir22"]).assign_coords(wrs=("time", [1, 2, 3]))
    out = zeit.harmonize([s2, landsat], etm="rma")
    assert list(out.band.values) == ["blue", "green", "red", "nir", "swir1", "swir2"]
    assert out.sizes["time"] == 5 and out.indexes["time"].is_monotonic_increasing
    assert "granule" not in out.coords and "wrs" not in out.coords and list(out.platform.values)[:2] == [
        "landsat-8", "sentinel-2a"]
    a, b = HLS["S2B"]["nir"]
    np.testing.assert_allclose(out.sel(time="2020-01-20", band="nir"), a * s2[1].sel(band="B8A") + b, rtol=1e-6)
    a, b = ROY_2016["rma"]["red"]
    np.testing.assert_allclose(out.sel(time="2020-01-11", band="red"), a * landsat[1].sel(band="red") + b, rtol=1e-6)
    assert "HLS" in out.attrs["harmonized"] and "Roy" in out.attrs["harmonized"] and out.rio.crs == s2.rio.crs
    with pytest.raises(ValueError, match="different grids"):
        zeit.harmonize([s2, landsat.isel(x=slice(1, None))])
    with pytest.raises(ValueError, match="one entry per cube"):
        zeit.harmonize([s2, landsat], sensor="S2A")
    named = zeit.harmonize([s2.drop_vars("platform"), landsat], sensor=["S2A", None])
    np.testing.assert_allclose(named.sel(time="2020-01-03"), out.sel(time="2020-01-03"))   # S2A, as its platform
