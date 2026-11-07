"""zeit.parallax_clouds, zeit.cloud_displacement_index and zeit.buffer_mask: FORCE's Sentinel-2 parallax cloud test
and cloud buffer, ported from FORCE 3.11."""
import numpy as np
import pytest
import xarray as xr

import zeit
from zeit import _core

BANDS = ["blue", "green", "red", "rededge3", "nir", "nir08", "swir16"]
NY = NX = 60


def _cube(ny=NY, nx=NX, nt=None, dtype=np.float32):
    """A flat bright surface (reflectance 0.3 in every band), (band, y, x) or (time, band, y, x)."""
    shape = (len(BANDS), ny, nx)
    arr = np.full(shape, 0.3, dtype=np.float64)
    dims = ("band", "y", "x")
    coords = {"band": BANDS, "y": 5000.0 - 10.0 * np.arange(ny), "x": 3000.0 + 10.0 * np.arange(nx)}
    if nt:
        arr = np.stack([arr] * nt)
        dims = ("time",) + dims
        coords["time"] = np.arange(nt)
    return xr.DataArray(arr.astype(dtype), dims=dims, coords=coords)


def _set(cube, band, sl, value, shift=0):
    i = BANDS.index(band)
    ys, xs = sl
    cube[..., i, ys, slice(xs.start + shift, xs.stop + shift)] = value


def _displaced_cloud(cube):
    """B07 and B8A see a bright object at the same place, B08 6 pixels (60 m) to the right of it."""
    sl = (slice(20, 40), slice(20, 40))
    for b in ("rededge3", "nir08"):
        _set(cube, b, sl, 0.5)
    _set(cube, "nir", sl, 0.5, shift=6)
    for b in ("blue", "green", "red"):
        _set(cube, b, sl, 0.5)
    return cube


def _static_surface(cube):
    """A bright built-up patch seen at the same place in every band, the red edge changing more than B08."""
    sl = (slice(20, 40), slice(20, 40))
    _set(cube, "rededge3", sl, 0.4)
    _set(cube, "nir08", sl, 0.5)
    _set(cube, "nir", sl, 0.55)
    for b in ("blue", "green", "red"):
        _set(cube, b, sl, 0.5)
    return cube


def test_displaced_cloud_has_negative_cdi_and_is_cloud():
    cube = _displaced_cloud(_cube())
    pcp = np.zeros((NY, NX), bool)
    pcp[15:45, 15:50] = True
    cloud, cdi = zeit.parallax_clouds(cube, pcp, variability=np.full((NY, NX), 0.5), return_cdi=True)
    assert cloud.dtype == bool and cloud.dims == ("y", "x") and cloud.shape == (NY, NX)
    assert cdi.dims == ("y", "x") and cdi.shape == (NY // 2, NX // 2) and cdi.dtype == np.float32
    # at the left edge of the object only the broad NIR ratio varies
    assert cdi.values[10, 10] == pytest.approx(-1.0)
    assert cloud.values[28:32, 22:26].all()
    assert not cloud.values[:10].any() and not cloud.values[:, 55:].any()
    # where nothing is a potential cloud, there is no CDI
    assert np.isnan(cdi.values[:5]).all()


def test_static_bright_surface_is_not_cloud():
    cube = _static_surface(_cube())
    pcp = np.zeros((NY, NX), bool)
    pcp[15:45, 15:45] = True
    cloud, cdi = zeit.parallax_clouds(cube, pcp, variability=np.full((NY, NX), 0.5), return_cdi=True)
    assert not cloud.values.any()
    assert cdi.values[10, 10] > 0.3


def test_cloud_displacement_index_function():
    cube = _displaced_cloud(_cube())
    pcp = np.zeros((NY, NX), bool)
    pcp[15:45, 15:50] = True
    cdi = zeit.cloud_displacement_index(cube, potential=pcp)
    _, cdi2 = zeit.parallax_clouds(cube, pcp, variability=np.full((NY, NX), 0.5), return_cdi=True)
    np.testing.assert_array_equal(cdi.values, cdi2.values)
    # the 20 m grid's coordinates: the centre of each block
    assert cdi.y.values[0] == pytest.approx(4995.0) and cdi.x.values[0] == pytest.approx(3005.0)
    # without potential every pixel is one
    all_cdi = zeit.cloud_displacement_index(cube)
    assert np.isfinite(all_cdi.values[10, 10])


def test_potential_gates_everything():
    cube = _displaced_cloud(_cube())
    cloud = zeit.parallax_clouds(cube, np.zeros((NY, NX), bool))
    assert not cloud.values.any()


def test_bright_and_saturated_gate():
    cube = _displaced_cloud(_cube())
    pcp = np.ones((NY, NX), bool)
    var = np.full((NY, NX), 0.5)
    base = zeit.parallax_clouds(cube, pcp, variability=var)
    assert base.values.any()
    # a dark cube: nothing is bright (mean of blue, green and red below 0.15)
    dark = cube.copy()
    for b in ("blue", "green", "red"):
        dark.loc[dict(band=b)] = 0.1
    assert not zeit.parallax_clouds(dark, pcp, variability=var).values.any()
    # unless the user says what is bright; saturated pixels are cloud if bright
    sat = np.zeros((NY, NX), bool)
    sat[2, 3] = True
    out = zeit.parallax_clouds(dark, pcp, variability=var, bright=np.ones((NY, NX), bool), saturated=sat)
    assert out.values[2, 3]
    assert not zeit.parallax_clouds(dark, pcp, variability=var, saturated=sat).values[2, 3]


def test_nodata_is_off():
    cube = _displaced_cloud(_cube())
    cube[BANDS.index("nir"), 28:32, 22:26] = np.nan
    cloud = zeit.parallax_clouds(cube, np.ones((NY, NX), bool), variability=np.full((NY, NX), 0.5))
    assert not cloud.values[28:32, 22:26].any()
    # integer data: 0 is nodata, scale 10000
    ints = (_displaced_cloud(_cube()) * 10000).round().astype(np.uint16)
    ints[BANDS.index("nir"), 28:32, 22:26] = 0
    cloud = zeit.parallax_clouds(ints, np.ones((NY, NX), bool), variability=np.full((NY, NX), 0.5))
    assert not cloud.values[28:32, 22:26].any() and cloud.values.any()


def test_scale_does_not_matter():
    a = zeit.parallax_clouds(_displaced_cloud(_cube()), np.ones((NY, NX), bool), variability=np.full((NY, NX), 0.5))
    b = zeit.parallax_clouds(_displaced_cloud(_cube()) * 10000, np.ones((NY, NX), bool),
                             variability=np.full((NY, NX), 0.5))
    np.testing.assert_array_equal(a.values, b.values)


def test_time_dimension_and_threads():
    cube = _displaced_cloud(_cube(nt=3))
    cube[1] = 0.3          # the second date has no cloud
    pcp = np.ones((NY, NX), bool)
    var = np.full((NY, NX), 0.5)
    c1 = zeit.parallax_clouds(cube, pcp, variability=var, n_jobs=1)
    c4 = zeit.parallax_clouds(cube, pcp, variability=var, n_jobs=4)
    assert c1.dims == ("time", "y", "x") and c1.shape == (3, NY, NX)
    np.testing.assert_array_equal(c1.values, c4.values)
    assert c1.values[0].any() and not c1.values[1].any() and c1.values[2].any()
    cdi = zeit.cloud_displacement_index(cube, potential=pcp)
    assert cdi.dims == ("time", "y", "x") and cdi.shape == (3, NY // 2, NX // 2)


def test_variability_computed_from_zhu_and_woodcock():
    cube = _displaced_cloud(_cube())
    pcp = np.ones((NY, NX), bool)
    # without variability it is computed from the bands: a flat grey surface has NDVI = NDSI = 0 and whiteness 0
    # (variability 1 > 0.8) and the displaced cloud is found
    cloud = zeit.parallax_clouds(cube, pcp)
    assert cloud.values.any()
    from zeit._parallax import _variability

    vis = np.stack([np.full((2, 2), v, np.float32) for v in (0.2, 0.2, 0.2)])
    nir = np.full((2, 2), 0.2, np.float32)
    swir = np.full((2, 2), 0.2, np.float32)
    np.testing.assert_allclose(_variability(vis, nir, swir, np.zeros((2, 2), bool)), 1.0)
    # vegetation: NDVI = 0.14 / 0.26 -> 0.46; snow: NDSI = 0.8 -> 0.2
    veg = _variability(np.stack([np.full((1, 1), 0.06, np.float32)] * 3), np.full((1, 1), 0.2, np.float32),
                       np.full((1, 1), 0.06, np.float32), np.zeros((1, 1), bool))
    assert veg[0, 0] == pytest.approx(1 - 0.14 / 0.26, abs=1e-6)
    snow = _variability(np.stack([np.full((1, 1), v, np.float32) for v in (0.9, 0.9, 0.9)]),
                        np.full((1, 1), 0.9, np.float32), np.full((1, 1), 0.1, np.float32), np.zeros((1, 1), bool))
    assert snow[0, 0] == pytest.approx(0.2, abs=1e-6)
    # saturation: a positive NDVI and a negative NDSI are zeroed
    vis = np.stack([np.full((1, 1), 0.06, np.float32)] * 3)
    nir, swir = np.full((1, 1), 0.2, np.float32), np.full((1, 1), 0.3, np.float32)
    assert _variability(vis, nir, swir, np.zeros((1, 1), bool))[0, 0] == pytest.approx(1 - 0.6667, abs=1e-3)
    assert _variability(vis, nir, swir, np.ones((1, 1), bool))[0, 0] == pytest.approx(1.0)


def test_odd_size_is_accepted():
    cube = _displaced_cloud(_cube(ny=61, nx=63))
    cloud = zeit.parallax_clouds(cube, np.ones((61, 63), bool), variability=np.full((61, 63), 0.5))
    assert cloud.shape == (61, 63)
    assert not cloud.values[-1].any() and not cloud.values[:, -1].any()
    assert zeit.cloud_displacement_index(cube).shape == (30, 31)


def test_parallax_errors():
    cube = _displaced_cloud(_cube())
    pcp = np.ones((NY, NX), bool)
    with pytest.raises(ValueError, match="no band"):
        zeit.parallax_clouds(cube.sel(band=["red", "nir"]), pcp)
    with pytest.raises(ValueError, match="bright="):
        zeit.parallax_clouds(cube.sel(band=["rededge3", "nir", "nir08"]), pcp, variability=np.zeros((NY, NX)))
    # no visible bands is fine when bright and variability are given
    zeit.parallax_clouds(cube.sel(band=["rededge3", "nir", "nir08"]), pcp, variability=np.zeros((NY, NX)),
                         bright=pcp)
    with pytest.raises(ValueError, match="does not broadcast"):
        zeit.parallax_clouds(cube, np.ones((5, 5), bool))
    with pytest.raises(ValueError, match="unknown role"):
        zeit.parallax_clouds(cube, pcp, bands={"purple": "red"})
    with pytest.raises(ValueError, match="not a band"):
        zeit.parallax_clouds(cube, pcp, bands={"red": "nope"}, variability=np.zeros((NY, NX)))
    with pytest.raises(TypeError):
        zeit.parallax_clouds(cube.values, pcp)
    with pytest.raises(ValueError, match="band, y and x"):
        zeit.parallax_clouds(cube.isel(band=0, drop=True), pcp)


def test_bands_mapping():
    cube = _displaced_cloud(_cube())
    renamed = cube.assign_coords(band=["a", "b", "c", "d", "e", "f", "g"])
    mapping = {"blue": "a", "green": "b", "red": "c", "rededge3": "d", "nir": "e", "nir08": "f", "swir16": "g"}
    pcp = np.ones((NY, NX), bool)
    var = np.full((NY, NX), 0.5)
    np.testing.assert_array_equal(zeit.parallax_clouds(renamed, pcp, variability=var, bands=mapping).values,
                                  zeit.parallax_clouds(cube, pcp, variability=var).values)


# ---------------------------------------------------------------------------------------------------------------
# the buffer

def test_buffer_radius_and_shape():
    m = np.zeros((41, 41), bool)
    m[20, 20] = True
    out = zeit.buffer_mask(m, 5, units="px")
    assert out.dtype == bool and out.shape == m.shape
    # FORCE's midpoint circle: the axis points at distance r are in, (r + 1, 0) and (4, 4) are not, (3, 3) is
    # (a little fatter than the disc: (5, 1) is in)
    assert out[20, 25] and out[20, 15] and out[25, 20] and out[15, 20]
    assert not out[20, 26] and not out[24, 24] and out[23, 23]
    assert out[21, 25] and out.sum() == 89
    np.testing.assert_array_equal(zeit.buffer_mask(m, 0, units="px"), m)


def test_buffer_meters_use_the_resolution():
    m = np.zeros((41, 41), bool)
    m[20, 20] = True
    a = zeit.buffer_mask(m, 50, units="m", resolution=10)
    np.testing.assert_array_equal(a, zeit.buffer_mask(m, 5, units="px"))
    # truncation as FORCE's int conversion: 59 m at 10 m is 5 px
    np.testing.assert_array_equal(zeit.buffer_mask(m, 59, resolution=10), a)
    da = xr.DataArray(m, dims=("y", "x"), coords={"y": -20.0 * np.arange(41), "x": 20.0 * np.arange(41)})
    b = zeit.buffer_mask(da, 100)     # 20 m pixels: 5 px
    assert isinstance(b, xr.DataArray) and b.dims == ("y", "x")
    np.testing.assert_array_equal(b.values, a)
    with pytest.raises(ValueError, match="pixel size"):
        zeit.buffer_mask(m, 100)


def test_buffer_edges_and_force():
    m = np.zeros((30, 30), bool)
    m[0, 0] = True
    m[15, 15] = True
    assert not zeit.buffer_mask(m, 3, units="px", force=True)[0, 0]       # FORCE loses it
    out = zeit.buffer_mask(m, 3, units="px")
    assert out[0, 0] and out[0, 3] and out[3, 0] and not out[0, 4]
    inner = zeit.buffer_mask(m, 3, units="px", force=True)
    np.testing.assert_array_equal(inner[8:], out[8:])


def test_buffer_time_and_errors():
    m = np.zeros((3, 20, 20), bool)
    m[1, 10, 10] = True
    da = xr.DataArray(m, dims=("time", "y", "x"), coords={"time": np.arange(3), "x": np.arange(20) * 10.0})
    out = zeit.buffer_mask(da, 2, units="px", n_jobs=2)
    assert out.dims == ("time", "y", "x") and not out.values[0].any() and out.values[1].sum() == 13
    with pytest.raises(ValueError, match="units"):
        zeit.buffer_mask(m, 2, units="km")
    with pytest.raises(ValueError, match="whole number"):
        zeit.buffer_mask(m, 2.5, units="px")
    with pytest.raises(ValueError, match="at least 0"):
        zeit.buffer_mask(m, -1, units="px")
    with pytest.raises(ValueError, match="boolean"):
        zeit.buffer_mask(np.full((4, 4), 7), 1, units="px")
    with pytest.raises(ValueError, match="dimensions"):
        zeit.buffer_mask(np.zeros(4, bool), 1, units="px")


def test_core_module_present():
    assert hasattr(_core.parallax, "parallax_batch") and hasattr(_core.parallax, "buffer_batch")
