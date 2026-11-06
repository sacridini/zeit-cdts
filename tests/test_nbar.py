"""zeit.nbar: NBAR by the c-factor (Roy et al. 2016, 2017), the analytic values of the specification."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit import _core
from zeit._nbar import ROLES

RED = (0.1690, 0.0227, 0.0574)       # f_iso, f_geo, f_vol
NIR = (0.3093, 0.0330, 0.1535)
BLUE = (0.0774, 0.0079, 0.0372)

# (sza, vza, raa) -> (k_vol, k_geo), from the specification, 6 decimals
KERNELS = [
    ((0, 0, 0), (0.000000, 0.000000)), ((30, 0, 0), (-0.031443, -0.698222)), ((30, 0, 180), (-0.031443, -0.698222)),
    ((30, 7.5, 0), (0.006595, -0.512314)), ((30, 7.5, 90), (-0.032118, -0.718905)),
    ((30, 7.5, 180), (-0.066297, -0.870441)), ((45, 10, 0), (0.018369, -0.870292)),
    ((45, 10, 180), (-0.094152, -1.291603)), ((60, 10, 180), (-0.072121, -1.652704)),
    ((30, 7.5, 100), (-0.038351, -0.748224)), ((30, 7.5, 260), (-0.038351, -0.748224)),
]
# band, (sza, vza, raa), sza_norm, c
C_VALUES = [
    ("red", (30, 7.5, 0), 30, 0.959407), ("red", (30, 7.5, 180), 30, 1.040636), ("red", (30, 7.5, 90), 30, 1.003370),
    ("red", (45, 10, 180), 45, 1.051881), ("red", (40, 10, 45), 35, 0.983404), ("red", (30, 0, 123), 30, 1.0),
    ("nir", (30, 7.5, 0), 30, 0.959190), ("nir", (30, 7.5, 180), 30, 1.040804), ("nir", (30, 7.5, 90), 30, 1.002801),
    ("nir", (45, 10, 180), 45, 1.053565), ("nir", (40, 10, 45), 35, 0.977951),
    ("blue", (30, 7.5, 0), 30, 0.960818), ("blue", (30, 7.5, 180), 30, 1.039042),
    ("blue", (45, 10, 180), 45, 1.051122),
]


def _cube(n=2, h=6, w=7, bands=("blue", "green", "red", "nir08", "swir16", "swir22"), seed=0, dtype=np.float32,
          platforms=None):
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.02, 0.45, (n, len(bands), h, w)).astype(dtype)
    coords = {"time": pd.date_range("2020-01-01", periods=n, freq="10D"), "band": list(bands),
              "y": np.arange(h)[::-1] * 30.0 + 15, "x": np.arange(w) * 30.0 + 15}
    if platforms is not None:
        coords["platform"] = ("time", list(platforms))
    return xr.DataArray(data, dims=("time", "band", "y", "x"), name="reflectance", coords=coords).rio.write_crs(32722)


def _angles(cube, seed=1, dtype=np.float32):
    rng = np.random.default_rng(seed)
    shape = (cube.sizes["time"], cube.sizes["y"], cube.sizes["x"])
    dims = ("time", "y", "x")
    coords = {k: cube[k] for k in dims}
    sza = rng.uniform(25, 45, shape)
    saa = rng.uniform(100, 140, shape)
    vza = rng.uniform(0, 10, shape)
    vaa = rng.uniform(0, 360, shape)
    return xr.Dataset({k: xr.DataArray(v.astype(dtype), dims=dims, coords=coords)
                       for k, v in dict(SZA=sza, SAA=saa, VZA=vza, VAA=vaa).items()})


def ref_kernels(ts, tv, phi):
    """The kernels of the specification, in numpy, degrees in."""
    ts, tv, phi = (np.radians(np.asarray(v, dtype=float)) for v in (ts, tv, phi))
    cos_xi = np.clip(np.cos(ts) * np.cos(tv) + np.sin(ts) * np.sin(tv) * np.cos(phi), -1, 1)
    xi = np.arccos(cos_xi)
    kvol = ((np.pi / 2 - xi) * cos_xi + np.sin(xi)) / (np.cos(ts) + np.cos(tv)) - np.pi / 4
    d = np.sqrt(np.tan(ts) ** 2 + np.tan(tv) ** 2 - 2 * np.tan(ts) * np.tan(tv) * np.cos(phi))
    sec = 1 / np.cos(ts) + 1 / np.cos(tv)
    cos_t = 2.0 * np.sqrt(d ** 2 + (np.tan(ts) * np.tan(tv) * np.sin(phi)) ** 2) / sec
    t = np.arccos(np.clip(cos_t, -1, 1))
    overlap = (t - np.sin(t) * np.cos(t)) * sec / np.pi
    kgeo = overlap - sec + 0.5 * (1 + cos_xi) / (np.cos(ts) * np.cos(tv))
    return kvol, kgeo


def ref_c(coef, ts, tv, phi, norm):
    f_iso, f_geo, f_vol = coef
    kv, kg = ref_kernels(ts, tv, phi)
    nv, ng = ref_kernels(norm, 0.0, 0.0)
    return (f_iso + f_vol * nv + f_geo * ng) / (f_iso + f_vol * kv + f_geo * kg)


def _one_pixel(role, ts, tv, phi, norm, bands=("blue", "green", "red", "nir08")):
    """The c-factor of a 1 x 1 cube through zeit.nbar: reflectance 1, so that nbar is c."""
    cube = xr.DataArray(np.ones((1, len(bands), 1, 1), np.float64), dims=("time", "band", "y", "x"),
                        coords={"time": pd.date_range("2020-01-01", periods=1), "band": list(bands),
                                "y": [0.0], "x": [0.0]})
    ang = xr.Dataset({k: xr.DataArray(np.full((1, 1, 1), v), dims=("time", "y", "x"),
                                      coords={"time": cube.time, "y": cube.y, "x": cube.x})
                      for k, v in dict(SZA=ts, SAA=0.0, VZA=tv, VAA=phi).items()})
    out = zeit.nbar(cube, ang, sza=norm)
    return float(out.sel(band=role).values.squeeze())


@pytest.mark.parametrize("angles, want", KERNELS)
def test_kernels_are_the_specifications(angles, want):
    kv, kg = _core.brdf.kernels(*(np.array([float(a)]) for a in angles))
    assert abs(kv[0] - want[0]) < 1e-6 and abs(kg[0] - want[1]) < 1e-6
    rv, rg = ref_kernels(*angles)
    assert abs(kv[0] - rv) < 1e-12 and abs(kg[0] - rg) < 1e-12


@pytest.mark.parametrize("band, angles, norm, want", C_VALUES)
def test_c_factors_through_the_api(band, angles, norm, want):
    ts, tv, phi = angles
    role = "nir08" if band == "nir" else band
    # the sun azimuth at 0 and the view azimuth at the relative azimuth
    assert abs(_one_pixel(role, ts, tv, phi, norm) - want) < 1e-6


def test_properties_of_the_model():
    kv, kg = _core.brdf.kernels(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 77.0]))
    assert np.abs(kv).max() < 1e-15 and np.abs(kg).max() < 1e-15
    rng = np.random.default_rng(3)
    ts, tv, phi = rng.uniform(0, 80, 200), rng.uniform(0, 40, 200), rng.uniform(0, 360, 200)
    a = _core.brdf.kernels(ts, tv, phi)
    for other in (360 - phi, -phi):
        b = _core.brdf.kernels(ts, tv, other)
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], atol=1e-12)
    np.testing.assert_allclose(a[0], ref_kernels(ts, tv, phi)[0], atol=1e-12)
    np.testing.assert_allclose(a[1], ref_kernels(ts, tv, phi)[1], atol=1e-12)
    # c = 1 at nadir with the sun at the normalized zenith, whatever the azimuth
    for phi0 in (0, 17, 123, 300):
        assert abs(_one_pixel("red", 30, 0, phi0, 30) - 1) < 1e-12
    # no kernel weights: c = 1 for any geometry
    cube = _cube(1, 3, 3, bands=("red",)).astype(np.float64)
    out = zeit.nbar(cube, _angles(cube).astype(np.float64), params={"red": (0.2, 0.0, 0.0)}, sza=40)
    np.testing.assert_allclose(out.values, cube.values, rtol=1e-12)


def test_the_published_parameters_and_the_red_edge():
    assert _core.brdf.role("red") == RED and _core.brdf.role("nir") == NIR and _core.brdf.role("nir08") == NIR
    assert _core.brdf.role("blue") == BLUE == _core.brdf.role("coastal")
    assert _core.brdf.role("green") == (0.1306, 0.0178, 0.0580)
    assert _core.brdf.role("swir1") == (0.3430, 0.0453, 0.1154) and _core.brdf.role("swir2") == (0.2658, 0.0387, 0.0639)
    for nm, want in ((705, (0.2085, 0.0256, 0.0845)), (740, (0.2316, 0.0273, 0.1003)),
                     (783, (0.2599, 0.0294, 0.1197))):
        np.testing.assert_allclose(_core.brdf.rededge(nm), want, atol=5e-5)
    for i, role in enumerate(("rededge1", "rededge2", "rededge3")):
        assert _core.brdf.role(role) == _core.brdf.rededge((705, 740, 783)[i])
    assert set(_core.brdf.ROLES) == set(ROLES)


def test_matches_the_numpy_reference_and_keeps_type():
    bands = ("blue", "green", "red", "rededge1", "nir", "nir08", "swir16", "swir22", "B09")
    cube = _cube(3, 5, 6, bands=bands, platforms=["sentinel-2a", "sentinel-2b", "sentinel-2a"])
    ang = _angles(cube)
    out = zeit.nbar(cube, ang, sza=35.0)
    assert out.dtype == np.float32 and out.dims == cube.dims
    roles = {"blue": "blue", "green": "green", "red": "red", "rededge1": "rededge1", "nir": "nir", "nir08": "nir08",
             "swir1": "swir16", "swir2": "swir22"}
    assert out.attrs["nbar"].startswith("c-factor, Roy et al.") and "red=red" in out.attrs["nbar_bands"]
    for role, band in roles.items():
        c = ref_c(_core.brdf.role(role), ang.SZA.values, ang.VZA.values, ang.VAA.values - ang.SAA.values, 35.0)
        np.testing.assert_allclose(out.sel(band=band).values, cube.sel(band=band).values * c, rtol=2e-6)
    np.testing.assert_array_equal(out.sel(band="B09").values, cube.sel(band="B09").values)   # no role: copied
    assert "nbar_bands" in out.attrs and "B09" not in out.attrs["nbar_bands"]
    assert out.rio.crs == cube.rio.crs
    np.testing.assert_allclose(out.nbar_sza.values, 35.0)


def test_scene_zenith_is_the_mean_of_the_valid_pixels_of_each_date():
    cube = _cube(2, 4, 4, bands=("red",))
    ang = _angles(cube)
    ang["SZA"].values[0, 0, 0] = np.nan            # not valid
    ang["VZA"].values[1, 1, 1] = 95.0             # not valid
    out, c = zeit.nbar(cube, ang, return_cfactor=True)
    for t in range(2):
        valid = np.isfinite(ang.SZA.values[t]) & (ang.VZA.values[t] < 90)
        want = ang.SZA.values[t][valid].astype(float).mean()
        assert abs(float(out.nbar_sza[t]) - want) < 1e-9
    assert np.isnan(out.values[0, 0, 0, 0]) and np.isnan(out.values[1, 0, 1, 1])
    assert np.isnan(c.values[0, 0, 0, 0]) and np.isnan(c.values[1, 0, 1, 1])
    assert np.isfinite(out.values).sum() == out.size - 2
    assert "mean of the observed" in out.attrs["nbar"]


def test_return_cfactor_and_linearity_in_scale():
    cube = _cube(2, 5, 5)
    ang = _angles(cube)
    out, c = zeit.nbar(cube, ang, sza=33, return_cfactor=True)
    assert c.dtype == np.float32 and c.dims == cube.dims and list(c.band.values) == list(cube.band.values)
    np.testing.assert_allclose(out.values, cube.values * c.values, rtol=1e-6)
    assert 0.9 < float(c.min()) and float(c.max()) < 1.1
    # reflectance 0-1 and 0-10000 give the same c, and integers keep their type and nodata
    big = (cube * 10000).round().astype(np.uint16)
    big[0, 1, 0, 0] = 0
    out10k = zeit.nbar(big, ang, sza=33)
    assert out10k.dtype == np.uint16 and int(out10k[0, 1, 0, 0]) == 0
    expected = np.round(big.values.astype(float) * c.values)
    ok = big.values != 0
    np.testing.assert_array_equal(out10k.values[ok], expected[ok])
    float10k = zeit.nbar(cube * 10000, ang, sza=33)
    np.testing.assert_allclose(float10k.values, out.values * 10000, rtol=2e-6)


def test_dask_equals_numpy_and_stays_lazy():
    cube = _cube(3, 16, 18)
    ang = _angles(cube)
    eager, c0 = zeit.nbar(cube, ang, return_cfactor=True)
    lazy, c1 = zeit.nbar(cube.chunk({"time": 1, "y": 8, "x": 9}), ang.chunk({"time": 2, "y": 16, "x": 6}),
                         return_cfactor=True)
    assert lazy.chunks is not None and c1.chunks is not None
    np.testing.assert_array_equal(lazy.compute().values, eager.values)
    np.testing.assert_array_equal(c1.compute().values, c0.values)
    np.testing.assert_allclose(lazy.nbar_sza.values, eager.nbar_sza.values)
    # lazy angles with an eager cube, and a fixed zenith (nothing is read ahead)
    both = zeit.nbar(cube.chunk({"y": 8}), ang.chunk({"time": 1}), sza=30)
    np.testing.assert_allclose(both.compute().values, zeit.nbar(cube, ang, sza=30).values)


def test_angles_inside_the_cube_and_other_forms():
    cube = _cube(2, 4, 5, bands=("blue", "red", "nir08"))
    ang = _angles(cube)
    stacked = xr.concat([ang[k].astype(np.float32) for k in ("SZA", "SAA", "VZA", "VAA")], dim="band")
    stacked = stacked.assign_coords(band=["SZA", "SAA", "VZA", "VAA"])
    full = xr.concat([cube, stacked.transpose("time", "band", "y", "x")], dim="band")
    inside = zeit.nbar(full)
    outside = zeit.nbar(cube, ang)
    np.testing.assert_allclose(inside.sel(band=["blue", "red", "nir08"]).values, outside.values)
    np.testing.assert_array_equal(inside.sel(band=["SZA", "SAA", "VZA", "VAA"]).values,
                                  stacked.transpose("time", "band", "y", "x").values)    # copied
    array_form = ang.to_array("angle")
    np.testing.assert_allclose(zeit.nbar(cube, array_form).values, outside.values)
    # a single image, and the accessor
    one = zeit.nbar(cube.isel(time=0, drop=True), ang.isel(time=0, drop=True))
    np.testing.assert_allclose(one.values, zeit.nbar(cube.isel(time=[0]), ang.isel(time=[0])).values[0])
    np.testing.assert_allclose(cube.zeit.nbar(ang).values, outside.values)


def test_alignment_errors_and_missing_angles():
    cube = _cube(2, 4, 5)
    ang = _angles(cube)
    with pytest.raises(ValueError, match="sun and view angles"):
        zeit.nbar(cube)
    with pytest.raises(ValueError, match="lack"):
        zeit.nbar(cube, ang.drop_vars("VAA"))
    with pytest.raises(ValueError, match="x values"):
        zeit.nbar(cube, ang.isel(x=slice(1, None)))
    shifted = ang.assign_coords(y=ang.y + 30.0)
    with pytest.raises(ValueError, match="y coordinates differ"):
        zeit.nbar(cube, shifted)
    with pytest.raises(ValueError, match="time"):
        zeit.nbar(cube, ang.isel(time=0, drop=True))
    with pytest.raises(ValueError, match="dates differ"):
        zeit.nbar(cube, ang.assign_coords(time=ang.time.values + np.timedelta64(1, "D")))
    with pytest.raises(ValueError, match="dates differ"):
        zeit.nbar(cube, ang.isel(time=[1, 0]))
    with pytest.raises(ValueError, match="no 'time'|time"):
        zeit.nbar(cube.isel(time=0, drop=True), ang)
    with pytest.raises(ValueError, match="Dataset"):
        zeit.nbar(cube, np.zeros((2, 4, 4, 5)))


def test_refusals():
    cube = _cube(2, 4, 5)
    ang = _angles(cube)
    hls = cube.assign_coords(id=("time", ["HLS.L30.T23LKC.2023206T131429.v2.0", "HLS.S30.T23LKC.2023207T131429.v2.0"]))
    with pytest.raises(ValueError, match="HLS"):
        zeit.nbar(hls, ang)
    with pytest.raises(ValueError, match="already NBAR"):
        zeit.nbar(zeit.nbar(cube, ang), ang)
    raw = (cube * 10000).astype(np.uint16).assign_coords(scale=(("time", "band"), np.full((2, 6), 1e-4)))
    with pytest.raises(ValueError, match="digital numbers"):
        zeit.nbar(raw, ang)
    with pytest.raises(ValueError, match="no band"):
        zeit.nbar(cube.assign_coords(band=list("abcdef")), ang)
    with pytest.raises(ValueError, match="unknown role"):
        zeit.nbar(cube, ang, bands={"purple": "red"})
    with pytest.raises(ValueError, match="not a band"):
        zeit.nbar(cube, ang, bands={"red": "nope"})
    with pytest.raises(ValueError, match="params"):
        zeit.nbar(cube, ang, params={"red": (1.0, 2.0)})
    with pytest.raises(ValueError, match="0 to 90"):
        zeit.nbar(cube, ang, sza=95)
    with pytest.raises(ValueError, match="NBAR_SOLAR_ZENITH"):
        zeit.nbar(cube, ang, sza="hls")
    with pytest.raises(ValueError, match="number"):
        zeit.nbar(cube, ang, sza="noon")
    with pytest.raises(ValueError, match="values for 2 dates"):
        zeit.nbar(cube, ang, sza=[30, 31, 32])


def test_hls_zenith_tag_is_read_when_the_cube_has_it():
    cube = _cube(2, 4, 5, bands=("red",))
    ang = _angles(cube)
    tagged = cube.assign_coords(NBAR_SOLAR_ZENITH=("time", [30.0, 40.0]))
    out = zeit.nbar(tagged, ang, sza="hls")
    np.testing.assert_allclose(out.nbar_sza.values, [30.0, 40.0])
    np.testing.assert_allclose(out.values, zeit.nbar(cube, ang, sza=[30.0, 40.0]).values)
    attr = cube.copy()
    attr.attrs["NBAR_SOLAR_ZENITH"] = 31.5
    np.testing.assert_allclose(zeit.nbar(attr, ang, sza="hls").nbar_sza.values, 31.5)


def test_nan_negative_and_invalid_geometry():
    cube = _cube(1, 4, 4, bands=("red", "nir08"))
    ang = _angles(cube)
    cube[0, 0, 0, 0] = np.nan
    cube[0, 1, 1, 1] = -0.05                      # a negative reflectance is multiplied, not clipped
    ang["VZA"].values[0, 2, 2] = np.nan
    ang["SZA"].values[0, 3, 3] = 90.0
    ang["VAA"].values[0, 0, 3] = np.nan
    out = zeit.nbar(cube, ang, sza=35)
    assert np.isnan(out.values[0, 0, 0, 0]) and np.isfinite(out.values[0, 1, 0, 0])
    c = ref_c(_core.brdf.role("nir08"), ang.SZA.values[0, 1, 1], ang.VZA.values[0, 1, 1],
              ang.VAA.values[0, 1, 1] - ang.SAA.values[0, 1, 1], 35)
    assert out.values[0, 1, 1, 1] < 0 and abs(out.values[0, 1, 1, 1] - (-0.05 * c)) < 1e-6
    assert np.isnan(out.values[0, :, 2, 2]).all() and np.isnan(out.values[0, :, 3, 3]).all()
    assert np.isnan(out.values[0, :, 0, 3]).all()
    assert np.isfinite(out.values).sum() == out.size - 1 - 6
    # the raster's nodata and an all-invalid date
    cube2 = _cube(2, 3, 3, bands=("red",))
    ang2 = _angles(cube2)
    ang2["SZA"].values[1] = 120.0
    with pytest.warns(UserWarning, match="degrees"):
        out2 = zeit.nbar(cube2, ang2)
    assert np.isnan(out2.values[1]).all() and np.isfinite(out2.values[0]).all() and np.isnan(out2.nbar_sza.values[1])


def test_sensor_checks():
    cube = _cube(2, 3, 3, bands=("SR_B2", "SR_B4", "SR_B5"), platforms=["landsat-8", "landsat-8"])
    ang = _angles(cube)
    out = zeit.nbar(cube, ang, sza=30)
    assert "red=SR_B4" in out.attrs["nbar_bands"] and "nir08=SR_B5" in out.attrs["nbar_bands"]
    old = cube.assign_coords(platform=("time", ["landsat-7", "landsat-5"]))
    with pytest.raises(ValueError, match="Collection 2 band names"):
        zeit.nbar(old, ang, sza=30)
    out = zeit.nbar(old, ang, sza=30, bands={"blue": "SR_B2", "red": "SR_B4", "nir08": "SR_B5"})
    assert out.shape == cube.shape
    with pytest.raises(ValueError, match="unknown sensor"):
        zeit.nbar(cube, ang, sensor="modis", sza=30)
    # order with bandpass_adjust: both can be chained and each records itself
    s2 = _cube(2, 3, 3, bands=("blue", "green", "red", "nir08", "swir16", "swir22"),
               platforms=["sentinel-2a", "sentinel-2b"])
    both = zeit.bandpass_adjust(zeit.nbar(s2, _angles(s2), sza=30))
    assert "nbar" in both.attrs and "bandpass_adjusted" in both.attrs
