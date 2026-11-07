"""zeit.topographic_correction: Horn's slope and aspect, the C-correction and SCS+C, on synthetic scenes."""
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import zeit
from zeit._topocorr import slope_aspect

H, W, RES = 90, 110, 30.0


def _grid(h=H, w=W):
    return {"y": np.arange(h)[::-1] * RES + 15.0, "x": np.arange(w) * RES + 15.0}


def _dem(h=H, w=W, seed=0, amp=110.0):
    """Smooth hills, slopes up to about 30 degrees, facing every way."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w] * RES
    z = np.zeros((h, w))
    for _ in range(4):
        ky, kx = rng.uniform(0.002, 0.005, 2)
        z += amp / 2 * np.sin(ky * yy + rng.uniform(0, 6)) * np.sin(kx * xx + rng.uniform(0, 6))
    return xr.DataArray(z + 500, dims=("y", "x"), coords=_grid(h, w)).rio.write_crs(32722)


def _cos_i(dem, sz, saz):
    sa = slope_aspect(dem)
    s, a = np.radians(sa.slope.values), np.radians(sa.aspect.values)
    sz, saz = np.radians(sz), np.radians(saz)
    return np.cos(sz) * np.cos(s) + np.sin(sz) * np.sin(s) * np.cos(saz - a), s, sa.slope.values, sa.aspect.values


def _cube(values, dem, times=None, bands=None):
    """`values` is (band, y, x) or (time, band, y, x)."""
    coords = dict(_grid(*dem.shape))
    dims = ("band", "y", "x")
    coords["band"] = bands or [f"b{i}" for i in range(values.shape[-3])]
    if values.ndim == 4:
        dims = ("time",) + dims
        coords["time"] = times if times is not None else pd.date_range("2020-01-01", periods=values.shape[0],
                                                                         freq="10D")
    return xr.DataArray(values.astype(np.float32), dims=dims, coords=coords, name="reflectance") \
        .rio.write_crs(32722)


def _lambertian(dem, sz=40.0, saz=150.0, rho0=(0.10, 0.25, 0.4), noise=0.002, seed=3):
    ci, _, _, _ = _cos_i(dem, sz, saz)
    rng = np.random.default_rng(seed)
    out = np.stack([r * ci / np.cos(np.radians(sz)) + rng.normal(0, noise, ci.shape) for r in rho0])
    return _cube(out, dem), np.array(rho0)


# --- slope and aspect -------------------------------------------------------------------------------------

def _plane(dem_like, gx, gy):
    """z = gx * easting + gy * northing."""
    yy, xx = np.meshgrid(dem_like.y.values, dem_like.x.values, indexing="ij")
    return xr.DataArray(gx * xx + gy * yy, dims=("y", "x"), coords=dem_like.coords).rio.write_crs(32722)


@pytest.mark.parametrize("gx,gy,aspect", [
    (0.5, 0.0, 270.0),      # rises to the east: faces west
    (-0.5, 0.0, 90.0),
    (0.0, 0.5, 180.0),      # rises to the north: faces south
    (0.0, -0.5, 0.0),
    (0.3, 0.3, 225.0),      # rises to the north-east: faces south-west
    (-0.3, 0.3, 135.0),
])
def test_horn_on_planes(gx, gy, aspect):
    out = slope_aspect(_plane(_dem(20, 25), gx, gy))
    inner = out.isel(y=slice(1, -1), x=slice(1, -1))
    assert np.allclose(inner.slope, np.degrees(np.arctan(np.hypot(gx, gy))), atol=1e-4)
    assert np.allclose(inner.aspect, aspect, atol=1e-3)
    assert out.slope.attrs["units"] == "degrees"


def test_horn_on_a_cone():
    """Slope atan(k) everywhere off the apex and the aspect pointing away from it (downslope)."""
    dem = _dem(61, 61)
    yy, xx = np.meshgrid(dem.y.values, dem.x.values, indexing="ij")
    cy, cx = dem.y.values[30], dem.x.values[30]
    k = 0.4
    cone = dem.copy(data=3000 - k * np.hypot(xx - cx, yy - cy))
    out = slope_aspect(cone)
    r = np.hypot(xx - cx, yy - cy) / RES
    far = (r > 8) & (r < 28)
    assert np.allclose(out.slope.values[far], np.degrees(np.arctan(k)), atol=0.25)
    want = np.degrees(np.arctan2(xx - cx, yy - cy)) % 360        # azimuth from the apex, clockwise from north
    d = (out.aspect.values - want + 180) % 360 - 180
    assert np.abs(d[far]).max() < 1.5


def test_slope_and_aspect_edges_flat_and_nan():
    dem = _plane(_dem(12, 12), 0.2, 0.0)
    out = slope_aspect(dem)
    assert np.isfinite(out.slope.values).all()                      # edge replication: edges have a slope
    flat = slope_aspect(dem.copy(data=np.full(dem.shape, 100.0)))
    assert (flat.slope.values == 0).all() and (flat.aspect.values == 0).all()
    z = dem.values.copy()
    z[5, 5] = np.nan
    out = slope_aspect(dem.copy(data=z)).slope.values
    assert np.isnan(out[4:7, 4:7]).sum() == 8 and np.isfinite(out[5, 5])    # Horn's kernel skips the centre
    assert np.isfinite(out).sum() == out.size - 8


def test_slope_needs_a_projected_crs():
    dem = _dem(10, 10)
    with pytest.raises(ValueError, match="geographic"):
        slope_aspect(dem.rio.write_crs(4326))
    with pytest.raises(ValueError, match="no CRS"):
        slope_aspect(xr.DataArray(dem.values, dims=("y", "x"), coords=dem.coords).drop_vars("spatial_ref"))


def test_pixel_size_in_feet_is_converted():
    # a plane of 0.5 m per m, in US survey feet: the same slope
    dem = _plane(_dem(10, 10), 0.5, 0.0)
    feet = dem.assign_coords(x=dem.x * 3.280839895, y=dem.y * 3.280839895).rio.write_crs(2231)
    a = slope_aspect(feet).slope.values[1:-1, 1:-1]
    assert np.allclose(a, np.degrees(np.arctan(0.5)), atol=0.01)


# --- the correction ---------------------------------------------------------------------------------------

def test_c_correction_flattens_a_lambertian_scene():
    dem = _dem()
    cube, rho0 = _lambertian(dem)
    _, _, slope, aspect = _cos_i(dem, 40, 150)
    out = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150)
    assert out.dtype == np.float32 and out.dims == cube.dims and out.shape == cube.shape
    assert out.rio.crs == cube.rio.crs and (out.band == cube.band).all()
    for b, r in enumerate(rho0):
        before, after = cube.values[b], out.values[b]
        assert np.abs(after - r).mean() < 0.02 * r and abs(after.mean() - r) < 0.01 * r
        assert after.std() < 0.2 * before.std()
    # north-facing (dark) against south-facing (bright) slopes of the same steepness
    steep = (slope > 12) & (slope < 30)
    north = steep & ((aspect < 60) | (aspect > 300))
    south = steep & (np.abs(aspect - 180) < 60)
    assert north.sum() > 100 and south.sum() > 100
    for b in range(3):
        raw = abs(cube.values[b][north].mean() - cube.values[b][south].mean())
        fixed = abs(out.values[b][north].mean() - out.values[b][south].mean())
        assert raw > 0.3 * rho0[b] and fixed < 0.02 * rho0[b]
    # c came out (about 0: the data have no offset); it is a coordinate
    assert out.topo_c.dims == ("band",) and np.abs(out.topo_c.values).max() < 0.05
    assert "C-correction" in out.attrs["topocorr"]


def test_c_is_recovered_and_both_methods_follow_the_formulas():
    dem = _dem()
    ci, s, slope, aspect = _cos_i(dem, 35, 120)
    c_true, m = 0.5, 0.3
    rng = np.random.default_rng(5)
    values = np.stack([m * (c_true + ci) + rng.normal(0, 0.001, ci.shape), 2 * m * (c_true + ci)])
    cube = _cube(values, dem)
    for method in ("c", "scs+c"):
        out = zeit.topographic_correction(cube, dem, sun_zenith=35, sun_azimuth=120, method=method)
        assert np.allclose(out.topo_c.values, c_true, atol=0.02)
        c = out.topo_c.values[:, None, None].astype(np.float64)
        num = np.cos(np.radians(35)) if method == "c" else np.cos(s) * np.cos(np.radians(35))
        want = cube.values * (num + c) / (ci + c)
        assert np.allclose(out.values, want, rtol=2e-5, atol=1e-6)
        if method == "c":      # constant: m (cos sz + c)
            assert np.abs(out.values[1] - 2 * m * (np.cos(np.radians(35)) + c_true)).max() < 0.01


def test_flat_pixels_are_unchanged():
    dem = _dem()
    z = dem.values.copy()
    z[:30, :40] = 500.0
    dem = dem.copy(data=z)
    cube, _ = _lambertian(dem)
    out = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150)
    assert np.allclose(out.values[:, 1:29, 1:39], cube.values[:, 1:29, 1:39])


def test_units_do_not_matter_and_threads_agree():
    dem = _dem()
    cube, _ = _lambertian(dem)
    a = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150, n_jobs=1)
    b = zeit.topographic_correction(cube * 10000, dem, sun_zenith=40, sun_azimuth=150, n_jobs=3)
    assert np.allclose(a.values * 10000, b.values, rtol=1e-5)
    assert np.allclose(a.topo_c.values, b.topo_c.values, rtol=1e-4)


def test_time_dimension_and_the_shapes():
    dem = _dem()
    c1, _ = _lambertian(dem, sz=30, saz=100, seed=1)
    c2, _ = _lambertian(dem, sz=45, saz=160, seed=2)
    cube = _cube(np.stack([c1.values, c2.values]), dem)
    out = zeit.topographic_correction(cube, dem, sun_zenith=[30, 45], sun_azimuth=[100, 160])
    assert out.dims == ("time", "band", "y", "x") and out.topo_c.dims == ("time", "band")
    one = zeit.topographic_correction(c2, dem, sun_zenith=45, sun_azimuth=160)
    assert np.allclose(out.values[1], one.values, atol=1e-6)
    assert (out.values[0].std(axis=(1, 2)) < 0.3 * cube.values[0].std(axis=(1, 2))).all()
    # a cube with no band dimension, with and without time
    flat = zeit.topographic_correction(c1.isel(band=0, drop=True), dem, sun_zenith=30, sun_azimuth=100)
    assert flat.dims == ("y", "x") and np.allclose(flat.values, zeit.topographic_correction(
        c1, dem, sun_zenith=30, sun_azimuth=100).values[0], atol=1e-6)
    tflat = zeit.topographic_correction(cube.isel(band=0, drop=True), dem, sun_zenith=[30, 45],
                                        sun_azimuth=[100, 160])
    assert tflat.dims == ("time", "y", "x") and tflat.topo_c.dims == ("time",)
    assert np.allclose(tflat.values, out.values[:, 0], atol=1e-6)


def test_integer_cube_and_nodata():
    dem = _dem()
    cube, _ = _lambertian(dem)
    scaled = (cube * 10000).round().astype(np.int16).rio.write_nodata(0)
    scaled.values[0, :5, :5] = 0
    out = zeit.topographic_correction(scaled, dem, sun_zenith=40, sun_azimuth=150)
    assert out.dtype == np.float32 and np.isnan(out.values[0, :5, :5]).all() and np.isfinite(out.values[1]).all()
    assert out.values[1].mean() == pytest.approx(2500, rel=0.05)


# --- guards -----------------------------------------------------------------------------------------------

def test_a_band_that_gets_darker_with_illumination_is_copied_with_a_warning():
    dem = _dem()
    ci, _, _, _ = _cos_i(dem, 40, 150)
    cube = _cube(np.stack([0.5 - 0.3 * ci, 0.1 + 0.3 * ci]), dem)
    with pytest.warns(UserWarning, match="could not be fitted.*<= 0"):
        out = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150)
    assert np.isnan(out.topo_c.values[0]) and np.isfinite(out.topo_c.values[1])
    assert np.array_equal(out.values[0], cube.values[0])
    assert not np.array_equal(out.values[1], cube.values[1])


def test_too_few_sample_pixels_and_the_sample_mask():
    dem = _dem()
    cube, _ = _lambertian(dem)
    mask = np.zeros(dem.shape, bool)
    mask[:5, :5] = True
    with pytest.warns(UserWarning, match="25 sample pixels"):
        out = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150, sample=mask)
    assert np.array_equal(out.values, cube.values) and np.isnan(out.topo_c.values).all()
    out = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150, sample=mask, min_sample=10)
    assert np.isfinite(out.topo_c.values).all()
    # a sample of the steep pixels fits as the default does, a DataArray mask too, (time, y, x) too
    sl = slope_aspect(dem).slope.values
    ref = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150)
    same = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150, sample=sl >= 5)
    assert np.allclose(ref.values, same.values, atol=1e-4)      # (a pixel at exactly 5 degrees may differ)
    xmask = xr.DataArray(sl >= 5, dims=("y", "x"), coords=dem.coords)
    assert np.allclose(ref.values, zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150,
                                                               sample=xmask).values, atol=1e-4)
    # min_slope changes the default sample
    other = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150, min_slope=10)
    assert not np.allclose(ref.topo_c.values, other.topo_c.values, atol=1e-9)
    tmask = np.stack([sl >= 5, sl >= 5])
    t = zeit.topographic_correction(_cube(np.stack([cube.values] * 2), dem), dem, sun_zenith=40, sun_azimuth=150,
                                    sample=tmask)
    assert np.allclose(t.values[0], ref.values, atol=1e-4)


def test_nan_pixels_and_nan_dem_cells_stay_as_they_are():
    dem = _dem()
    cube, _ = _lambertian(dem)
    v = cube.values.copy()
    v[:, 10, 10] = np.nan
    z = dem.values.copy()
    z[50, 60] = np.nan
    out = zeit.topographic_correction(cube.copy(data=v), dem.copy(data=z), sun_zenith=40, sun_azimuth=150)
    assert np.isnan(out.values[:, 10, 10]).all()
    # the eight neighbours of the NaN cell have no slope; the cell itself has (Horn's kernel skips the centre)
    bad = np.array([[r, c] for r in (49, 50, 51) for c in (59, 60, 61) if (r, c) != (50, 60)])
    assert np.array_equal(out.values[:, bad[:, 0], bad[:, 1]], v[:, bad[:, 0], bad[:, 1]])
    assert np.isfinite(out.values[:, 50, 60]).all()


def test_pixels_with_a_missing_sun_or_a_horizon_sun_are_unchanged():
    dem = _dem()
    cube, _ = _lambertian(dem)
    zen = np.full(dem.shape, 40.0)
    zen[:5, :5] = np.nan
    zen[5:10, :5] = 95
    out = zeit.topographic_correction(cube, dem, sun_zenith=xr.DataArray(zen, dims=("y", "x"), coords=dem.coords),
                                      sun_azimuth=150)
    assert np.array_equal(out.values[:, :10, :5], cube.values[:, :10, :5])
    assert not np.array_equal(out.values[:, 20:, 20:], cube.values[:, 20:, 20:])


# --- the sun ----------------------------------------------------------------------------------------------

def test_sun_angle_forms_agree():
    dem = _dem()
    cube, _ = _lambertian(dem)
    ref = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150)
    like = xr.DataArray(np.full(dem.shape, 1.0), dims=("y", "x"), coords=dem.coords)
    zen, azi = like * 40, like * 150
    for kw in (
        dict(sun_zenith=zen, sun_azimuth=azi),
        dict(sun_zenith=zen, sun_azimuth=150),
        dict(sun_zenith=[40], sun_azimuth=np.array([150.0])),
        dict(sun_zenith=xr.Dataset({"SZA": zen, "SAA": azi})),                     # sentinel2_angles style
        dict(sun_zenith=xr.concat([zen, azi], dim="angle").assign_coords(angle=["SZA", "SAA"]),
             sun_azimuth="scene"),
        dict(sun_azimuth=xr.Dataset({"SAA": azi, "SZA": zen}), sun_zenith=zen),
    ):
        out = zeit.topographic_correction(cube, dem, **kw)
        assert np.allclose(out.values, ref.values, atol=1e-6), kw.keys()
    # 'scene': the cube's own bands, which are not corrected
    withbands = xr.concat([cube, cube.isel(band=[0]).copy(data=zen.values[None]).assign_coords(band=["SZA"]),
                           cube.isel(band=[0]).copy(data=azi.values[None]).assign_coords(band=["SAA"])], dim="band")
    out = zeit.topographic_correction(withbands, dem)
    assert list(out.band.values) == ["b0", "b1", "b2", "SZA", "SAA"]
    assert np.allclose(out.values[:3], ref.values, atol=1e-6)
    assert np.array_equal(out.values[3], zen.values) and np.isnan(out.topo_c.values[3:]).all()
    # 'scene': metadata, as an attribute (elevation) or a per-date coordinate
    meta = cube.copy()
    meta.attrs["view:sun_elevation"] = 50.0
    meta.attrs["view:sun_azimuth"] = 150.0
    out = zeit.topographic_correction(meta, dem)
    assert np.allclose(out.values, ref.values, atol=1e-6) and "metadata" in out.attrs["topocorr"]
    tcube = _cube(np.stack([cube.values, cube.values]), dem)
    tcube = tcube.assign_coords(sun_zenith=("time", [40.0, 40.0]), sun_azimuth=("time", [150.0, 150.0]))
    assert np.allclose(zeit.topographic_correction(tcube, dem).values[1], ref.values, atol=1e-6)
    # a per-date, per-pixel DataArray
    tz = xr.concat([zen, zen], dim="time").assign_coords(time=tcube.time)
    out = zeit.topographic_correction(tcube, dem, sun_zenith=tz, sun_azimuth=150)
    assert np.allclose(out.values[0], ref.values, atol=1e-6)


def test_errors():
    dem = _dem()
    cube, _ = _lambertian(dem)
    ok = dict(sun_zenith=40, sun_azimuth=150)
    f = zeit.topographic_correction
    with pytest.raises(ValueError, match="reproject_match"):
        f(cube, dem.isel(y=slice(1, None)), **ok)
    with pytest.raises(ValueError, match="reproject_match"):
        f(cube, dem.assign_coords(x=dem.x + 15), **ok)
    with pytest.raises(ValueError, match="dimensions"):
        f(cube, dem.expand_dims(band=2), **ok)
    with pytest.raises(ValueError, match="geographic"):
        f(cube.rio.write_crs(4326), dem, **ok)
    with pytest.raises(ValueError, match="method"):
        f(cube, dem, method="minnaert", **ok)
    with pytest.raises(ValueError, match="0 to 90"):
        f(cube, dem, sun_zenith=95, sun_azimuth=150)
    with pytest.raises(ValueError, match="3 values for 1 dates"):
        f(cube, dem, sun_zenith=[40, 41, 42], sun_azimuth=150)
    with pytest.raises(ValueError, match="found no SZA band"):
        f(cube, dem)
    with pytest.raises(ValueError, match="'scene'"):
        f(cube, dem, sun_zenith="noon", sun_azimuth=150)
    with pytest.raises(ValueError, match="sample must be"):
        f(cube, dem, sample=np.ones((3, 3), bool), **ok)
    with pytest.raises(ValueError, match="not a band"):
        f(cube, dem, bands=["nir"], **ok)
    with pytest.raises(ValueError, match="min_slope"):
        f(cube, dem, min_slope=95, **ok)
    zen = xr.DataArray(np.full((5, 5), 40.0), dims=("y", "x"))
    with pytest.raises(ValueError, match="grid"):
        f(cube, dem, sun_zenith=zen, sun_azimuth=150)
    with pytest.raises(ValueError, match="time"):
        f(cube, dem, sun_zenith=xr.DataArray(np.full((2,) + dem.shape, 40.0), dims=("time", "y", "x"),
                                             coords={**dem.coords, "time": [1, 2]}), sun_azimuth=150)


def test_bands_argument_copies_the_others():
    dem = _dem()
    cube, _ = _lambertian(dem)
    out = zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150, bands=["b1"])
    assert np.array_equal(out.values[0], cube.values[0]) and not np.array_equal(out.values[1], cube.values[1])
    assert np.isnan(out.topo_c.values[[0, 2]]).all() and np.isfinite(out.topo_c.values[1])


def test_no_warning_on_a_clean_scene():
    dem = _dem()
    cube, _ = _lambertian(dem)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zeit.topographic_correction(cube, dem, sun_zenith=40, sun_azimuth=150)
