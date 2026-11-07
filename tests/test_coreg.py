"""zeit.coregister: the global co-registration of AROSICS (phase correlation of one window), ported to the core."""
import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr

import zeit

RES = 30.0
X0, Y0 = 500000.0, 4600000.0
CRS = "EPSG:32633"


class Scene:
    """Value noise of 5 octaves (lattices of 2 to 32 pixels, smoothstep between nodes): continuous, so a scene at
    fractional coordinates is exact."""

    def __init__(self, seed=0, size=2048):
        rng = np.random.default_rng(seed)
        self.lat = [rng.uniform(-1, 1, (size // (2 << o) + 4, size // (2 << o) + 4)) for o in range(5)]

    def __call__(self, x, y):
        v = 0.0
        for o, lat in enumerate(self.lat):
            period = float(2 << o)
            u, w = x / period, y / period
            iu, iw = np.floor(u).astype(int), np.floor(w).astype(int)
            fu, fw = u - iu, w - iw
            su, sw = fu * fu * (3 - 2 * fu), fw * fw * (3 - 2 * fw)
            top = lat[iw, iu] * (1 - su) + lat[iw, iu + 1] * su
            bot = lat[iw + 1, iu] * (1 - su) + lat[iw + 1, iu + 1] * su
            v = v + (top * (1 - sw) + bot * sw) * np.sqrt(period)
        return (3000.0 + 300.0 * v).astype(np.float32)

    def image(self, ny, nx, move=(0.0, 0.0), origin=(0, 0)):
        """The scene on a grid of ny x nx cells whose first cell is `origin` (row, col) of the scene's, the content
        moved by `move` = (dx, dy) cells (to the right and down for positive)."""
        r, c = np.meshgrid(np.arange(ny) + origin[0], np.arange(nx) + origin[1], indexing="ij")
        return self(c + 0.5 - move[0], r + 0.5 - move[1])


def _da(arr, col0=0, row0=0, crs=CRS, res=RES, dims=None, **coords):
    ny, nx = arr.shape[-2:]
    dims = dims or {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}[arr.ndim]
    c = {"y": Y0 - res * (np.arange(ny) + row0 + 0.5), "x": X0 + res * (np.arange(nx) + col0 + 0.5)}
    c.update(coords)
    da = xr.DataArray(arr, dims=dims, coords=c)
    return da.rio.write_crs(crs) if crs else da


NY, NX = 360, 420


@pytest.fixture(scope="module")
def scene():
    return Scene(7)


@pytest.fixture(scope="module")
def reference(scene):
    return _da(scene.image(NY, NX))


def _target(scene, shift, **kw):
    """A date that has to be moved by `shift` = (dx, dy) cells to fit the reference."""
    return scene.image(NY, NX, move=(-shift[0], -shift[1]), **kw)


@pytest.mark.parametrize("shift,tol", [((3, -2), 0.15), ((0.4, 1.7), 0.2), ((-4, 4), 0.15), ((0, 0), 0.1)])
def test_known_shifts_are_recovered(scene, reference, shift, tol):
    cube = _da(_target(scene, shift))
    aligned, shifts = zeit.coregister(cube, reference, n_jobs=1)
    row = shifts.iloc[0]
    assert row.success and row.applied and row.reason is None
    assert abs(row.x_shift_px - shift[0]) < tol and abs(row.y_shift_px - shift[1]) < tol
    # map units: x to the right, y negative for a shift down on a north-up grid
    assert row.x_shift_map == pytest.approx(row.x_shift_px * RES)
    assert row.y_shift_map == pytest.approx(-row.y_shift_px * RES)
    assert row.reliability > 20 and row.ssim_after >= row.ssim_before - 1e-9
    assert aligned.shape == cube.shape and aligned.dims == cube.dims
    np.testing.assert_array_equal(aligned["x"], cube["x"])
    inner = (slice(15, -15), slice(15, -15))
    err_after = np.abs(aligned.values[inner] - reference.values[inner]).mean()
    err_before = np.abs(cube.values[inner] - reference.values[inner]).mean()
    assert err_after <= err_before + 1e-6
    if shift == (3, -2):
        # a whole number of cells: the cells are copied
        np.testing.assert_array_equal(aligned.values[inner], reference.values[inner])


def test_integer_part_and_window(scene, reference):
    cube = _da(_target(scene, (3, -2)))
    _, shifts = zeit.coregister(cube, reference, window_size=128, n_jobs=1)
    row = shifts.iloc[0]
    assert (row.window_x1 - row.window_x0, row.window_y1 - row.window_y0) == (128, 128)
    # centred on the overlap
    assert row.window_x0 == NX // 2 - 64 and row.window_y0 == NY // 2 - 64
    assert row.fft_size == 64  # the spectrum of the window moved by the integer shift: 126 pixels give 64


def test_time_and_band_dimensions(scene, reference):
    shifts_in = [(2, 1), (-1.5, 3), (0, -3)]
    bands = ["red", "nir"]
    arr = np.stack([np.stack([_target(scene, s), _target(scene, s) * 0.5]) for s in shifts_in]).astype(np.float32)
    cube = _da(arr, dims=("time", "band", "y", "x"), time=np.array(["2020-01-01", "2020-02-01", "2020-03-01"],
                                                                 dtype="datetime64[ns]"), band=bands)
    ref = xr.concat([reference, reference * 0.5], dim="band").assign_coords(band=bands)
    aligned, shifts = zeit.coregister(cube, ref, band="red", ref_band="red", n_jobs=2)
    assert aligned.dims == cube.dims and aligned.shape == cube.shape
    assert list(shifts.index) == list(cube.time.values) and shifts.index.name == "time"
    assert shifts.success.all()
    for (dx, dy), (_, row) in zip(shifts_in, shifts.iterrows()):
        assert abs(row.x_shift_px - dx) < 0.2 and abs(row.y_shift_px - dy) < 0.2
    # both bands of a date move with it: the nir band is the red one halved
    np.testing.assert_allclose(aligned.sel(band="nir").values, aligned.sel(band="red").values * 0.5, rtol=1e-5)
    # selecting the match band by position gives the same shifts
    _, again = zeit.coregister(cube, ref, band=0, ref_band=0, n_jobs=1)
    np.testing.assert_allclose(again.x_shift_px, shifts.x_shift_px)
    # without bands and without dates
    one, row = zeit.coregister(cube.sel(band="red").isel(time=1), reference)
    assert one.dims == ("y", "x") and len(row) == 1 and row.success.iloc[0]


def test_nodata_margins_are_left_out(scene, reference):
    arr = _target(scene, (2, 1)).astype(np.float32)
    arr[:, :140] = np.nan  # no data on the left 140 columns
    arr[260:, :] = np.nan  # and the bottom 100 rows
    cube = _da(arr)
    aligned, shifts = zeit.coregister(cube, reference)
    row = shifts.iloc[0]
    assert row.success and abs(row.x_shift_px - 2) < 0.2 and abs(row.y_shift_px - 1) < 0.2
    assert row.window_x0 >= 140 and row.window_y1 <= 260
    # the no-data stays no-data (moved with the image)
    assert np.isnan(aligned.values[:, :100]).all() and np.isnan(aligned.values[-60:, :]).all()


def test_nodata_value_and_integer_cubes(scene, reference):
    arr = (_target(scene, (-3, 2)) + 6000).astype(np.uint16)
    arr[:, :80] = 0
    cube = _da(arr).rio.write_nodata(0)
    ref = _da((scene.image(NY, NX) + 6000).astype(np.uint16)).rio.write_nodata(0)
    aligned, shifts = zeit.coregister(cube, ref)
    row = shifts.iloc[0]
    assert row.success and abs(row.x_shift_px + 3) < 0.2 and abs(row.y_shift_px - 2) < 0.2
    assert row.window_x0 >= 80
    assert aligned.dtype == np.uint16


def test_featureless_date_is_flagged_and_left_alone(scene, reference):
    good = _target(scene, (2, 2))
    flat = np.full((NY, NX), 100.0, np.float32)
    cube = _da(np.stack([good, flat]), dims=("time", "y", "x"), time=[1, 2])
    aligned, shifts = zeit.coregister(cube, reference)
    assert list(shifts.success) == [True, False]
    assert shifts.reason.iloc[1] is not None and not shifts.applied.iloc[1]
    np.testing.assert_array_equal(aligned.values[1], flat)


def test_shift_above_max_shift_fails(scene, reference):
    cube = _da(_target(scene, (8, 0)))
    aligned, shifts = zeit.coregister(cube, reference, window_size=64)
    assert not shifts.success.iloc[0] and "max_shift" in shifts.reason.iloc[0]
    np.testing.assert_array_equal(aligned.values, cube.values)
    _, ok = zeit.coregister(cube, reference, window_size=64, max_shift=10)
    assert ok.success.iloc[0] and abs(ok.x_shift_px.iloc[0] - 8) < 0.2


def test_min_reliability_flags_a_low_peak(scene, reference):
    noisy = _target(scene, (2, 1)) + np.random.default_rng(0).normal(0, 4000, (NY, NX)).astype(np.float32)
    cube = _da(noisy)
    _, shifts = zeit.coregister(cube, reference)
    rel = shifts.reliability.iloc[0]
    _, flagged = zeit.coregister(cube, reference, min_reliability=min(rel + 1, 100.0))
    if rel < 100:
        assert not flagged.success.iloc[0] and "reliability" in flagged.reason.iloc[0]


def test_extents_may_differ_and_window_position(scene):
    # a reference larger than the cube, whose first cell is 25 rows and 40 columns into it
    ref = _da(scene.image(500, 600))
    cube = _da(scene.image(300, 380, move=(-3, 2), origin=(25, 40)), col0=40, row0=25)
    aligned, shifts = zeit.coregister(cube, ref, window_size=128)
    row = shifts.iloc[0]
    assert row.success and abs(row.x_shift_px - 3) < 0.2 and abs(row.y_shift_px + 2) < 0.2
    assert aligned.shape == cube.shape
    # a window position (map x, y) of the cube's centre
    x, y = float(cube.x[190]), float(cube.y[150])
    _, at = zeit.coregister(cube, ref, window_size=128, window_position=(x, y))
    assert abs(at.window_x0.iloc[0] + 64 - 190) <= 1 and abs(at.window_y0.iloc[0] + 64 - 150) <= 1
    with pytest.raises(ValueError, match="outside the overlap"):
        zeit.coregister(cube, ref, window_position=(X0 - 1e5, y))


def test_errors(scene, reference):
    cube = _da(_target(scene, (1, 1)))
    with pytest.raises(ValueError, match="different CRS"):
        zeit.coregister(cube, _da(scene.image(NY, NX), crs="EPSG:32634"))
    with pytest.raises(ValueError, match="need a CRS"):
        zeit.coregister(_da(scene.image(NY, NX), crs=None), reference)
    with pytest.raises(ValueError, match="projected"):
        zeit.coregister(_da(scene.image(NY, NX), crs="EPSG:4326", res=0.001), _da(scene.image(NY, NX), crs="EPSG:4326",
                                                                                   res=0.001))
    far = _da(scene.image(NY, NX), col0=5000)
    with pytest.raises(ValueError, match="do not overlap"):
        zeit.coregister(cube, far)
    with pytest.raises(ValueError, match="cell size"):
        zeit.coregister(cube, _da(scene.image(NY, NX), res=10.0))
    half = reference.assign_coords(x=reference.x + RES / 2, y=reference.y)
    with pytest.raises(ValueError, match="same pixel grid"):
        zeit.coregister(cube, half)
    with pytest.raises(ValueError, match="no band"):
        zeit.coregister(_da(np.stack([_target(scene, (1, 1))] * 2), band=["a", "b"]), reference, band="c")
    with pytest.raises(ValueError, match="unknown resampling"):
        zeit.coregister(cube, reference, resampling="sinc")
    with pytest.raises(TypeError):
        zeit.coregister(cube.values, reference)
    # no valid data in the cube: flagged, not an error
    empty = _da(np.full((NY, NX), np.nan, np.float32))
    _, shifts = zeit.coregister(empty, reference)
    assert not shifts.success.iloc[0] and "no valid data" in shifts.reason.iloc[0]


# --- zeit.coregister_local: AROSICS' COREG_LOCAL ---------------------------------------------------------

LY, LX = 640, 760


def _field(x, y):
    """The shift (dx, dy) in cells, to move the target by to fit the reference at (x, y): an affine part and a gentle
    sinusoid, up to about 3 cells."""
    return 1.0 + 0.0015 * x + 1.0 * np.sin(y / 200.0), -0.8 + 0.001 * y + 0.7 * np.sin(x / 230.0)


def _warped(scene, ny=LY, nx=LX, field=_field):
    """The target of a reference `scene.image(ny, nx)`: at each cell the scene at the position `field` moves it from."""
    r, c = np.meshgrid(np.arange(ny) + 0.5, np.arange(nx) + 0.5, indexing="ij")
    dx, dy = field(c, r)
    return scene(c + dx, r + dy)


LOCAL = dict(grid_res=100, window_size=128, n_jobs=2)


@pytest.fixture(scope="module")
def local_ref(scene):
    return _da(scene.image(LY, LX))


@pytest.fixture(scope="module")
def local_result(scene, local_ref):
    cube = _da(_warped(scene))
    aligned, points = zeit.coregister_local(cube, local_ref, **LOCAL)
    return cube, aligned, points


def test_local_recovers_a_smooth_field(scene, local_ref, local_result):
    cube, aligned, points = local_result
    # AROSICS' grid: 0, 100, ..., 700 columns and 0, ..., 600 rows, the points on data
    # (the id is the position in the whole grid of 9 x 8 corners, as AROSICS' POINT_ID: the ones past the image are out)
    assert len(points) == 8 * 7 and points.point_id.is_monotonic_increasing and points.point_id.max() < 9 * 8
    assert list(points.point_id[:3]) == [0, 1, 2] and points.point_id.iloc[8] == 9
    assert sorted(set(points.col)) == list(np.arange(0, LX, 100.0)) and sorted(set(points.row)) == list(np.arange(0, LY, 100.0))
    used = points[points.used]
    assert len(used) >= 30 and points.date_applied.all()
    want_x, want_y = _field(used.col.values, used.row.values)
    # a window sees the field averaged over its 128 cells, which differs from its value at the centre by what the
    # field's slope there adds up to
    assert np.abs(used.x_shift_px - want_x).max() < 0.6 and np.abs(used.y_shift_px - want_y).max() < 0.6
    assert np.median(np.abs(used.x_shift_px - want_x)) < 0.2 and np.median(np.abs(used.y_shift_px - want_y)) < 0.2
    # map units: x right, y negative for a shift down on a north-up grid
    np.testing.assert_allclose(used.x_shift_map, used.x_shift_px * RES, atol=1e-6)
    np.testing.assert_allclose(used.y_shift_map, -used.y_shift_px * RES, atol=1e-6)
    np.testing.assert_allclose(used.abs_shift, np.hypot(used.x_shift_map, used.y_shift_map))
    np.testing.assert_allclose(used.x, X0 + used.col * RES)
    np.testing.assert_allclose(used.y, Y0 - used.row * RES)
    assert (used.reliability >= 60).all() and used.ssim_improved.all() and not used.outlier.any()
    # the date is warped onto the cube's grid and fits the reference better than before
    assert aligned.shape == cube.shape and aligned.dims == cube.dims
    np.testing.assert_array_equal(aligned["x"], cube["x"])
    inner = (slice(40, -40), slice(40, -40))
    before = np.abs(cube.values[inner] - local_ref.values[inner]).mean()
    after = np.abs(aligned.values[inner] - local_ref.values[inner]).mean()
    assert after < 0.4 * before, (before, after)
    dates = points.attrs["dates"]
    assert bool(dates.applied.iloc[0]) and dates.reason.iloc[0] is None and dates.used.iloc[0] == len(used)
    assert "COREG_LOCAL" in aligned.attrs["coregister_local"]


def test_local_polynomial_warp_and_the_same_points(scene, local_ref, local_result):
    cube, aligned, points = local_result
    # an affine field: a polynomial of order 1 is exact, and the points are those of the thin plate spline
    affine = lambda x, y: (1.0 + 0.002 * x - 0.001 * y, -1.0 + 0.0015 * y + 0.001 * x)  # noqa: E731
    cube_a = _da(_warped(scene, field=affine))
    out, pts = zeit.coregister_local(cube_a, local_ref, warp="polynomial", order=1, **LOCAL)
    inner = (slice(40, -40), slice(40, -40))
    before = np.abs(cube_a.values[inner] - local_ref.values[inner]).mean()
    assert np.abs(out.values[inner] - local_ref.values[inner]).mean() < 0.3 * before
    assert "polynomial" in out.attrs["coregister_local"]
    # the tie points do not depend on how the date is warped
    _, again = zeit.coregister_local(cube, local_ref, warp="polynomial", **LOCAL)
    for col in ("x_shift_px", "y_shift_px", "outlier", "used"):
        np.testing.assert_array_equal(again[col].values, points[col].values)
    # n_jobs does not change them either
    _, one = zeit.coregister_local(cube, local_ref, **{**LOCAL, "n_jobs": 1})
    np.testing.assert_array_equal(one.x_shift_px.values, points.x_shift_px.values)
    np.testing.assert_array_equal(one.outlier.values, points.outlier.values)


def test_local_filters_catch_bad_points(scene, local_ref):
    arr = _warped(scene)
    # a patch of another scene (changed texture) over the window of the point (300, 300), and a featureless patch
    # over the one of (500, 200)
    other = Scene(99)
    arr[230:370, 230:370] = other.image(LY, LX)[230:370, 230:370]
    arr[130:270, 430:570] = 1234.0
    cube = _da(arr)
    aligned, points = zeit.coregister_local(cube, local_ref, **LOCAL)
    at = lambda c, r: points[(points.col == c) & (points.row == r)].iloc[0]  # noqa: E731
    changed, flat = at(300, 300), at(500, 200)
    assert not changed.used and not flat.used
    # wrong where it matched: a flag of one of the filters
    assert (not changed.success) or changed.outlier
    assert (not flat.success) and flat.reason is not None and not flat.outlier
    # the flags are the union of the three filters
    np.testing.assert_array_equal(points.outlier, points.outlier_reliability | points.outlier_ssim | points.outlier_ransac)
    assert (points.used == (points.success & ~points.outlier)).all()
    # the rest still warps the date
    assert points.date_applied.all() and points.used.sum() >= 25
    inner = (slice(40, -40), slice(40, -40))
    ok = np.ones_like(aligned.values[inner], bool)
    ok[90:340, 90:340] = False  # away from the patches, with a margin for the warp's reach
    ok[90:240, 390:540] = False
    d_after = np.abs(aligned.values[inner] - local_ref.values[inner])[ok].mean()
    d_before = np.abs(arr[inner] - local_ref.values[inner])[ok].mean()
    assert d_after < 0.6 * d_before


def test_local_filter_levels(scene, local_ref):
    arr = _warped(scene)
    arr[230:370, 230:370] = Scene(99).image(LY, LX)[230:370, 230:370]
    cube = _da(arr)
    flags = {}
    for level in (0, 1, 2, 3):
        _, p = zeit.coregister_local(cube, local_ref, tie_point_filter_level=level, **LOCAL)
        flags[level] = p
    assert not flags[0].outlier.any()
    assert not flags[1].outlier_ssim.any() and not flags[1].outlier_ransac.any()
    assert not flags[2].outlier_ransac.any()
    # a higher level adds flags, and keeps those of the lower ones
    assert flags[3].outlier.sum() >= flags[2].outlier.sum() >= flags[1].outlier.sum() >= 0
    assert (flags[3].outlier_reliability == flags[1].outlier_reliability).all()
    # an unreasonably high minimum reliability flags everything it can
    _, strict = zeit.coregister_local(cube, local_ref, min_reliability=101, tie_point_filter_level=1, **LOCAL)
    assert strict.outlier[strict.success].all() and not strict.used.any() and not strict.date_applied.any()


def test_local_ransac_seed(scene, local_ref):
    arr = _warped(scene)
    cube = _da(arr)
    _, a = zeit.coregister_local(cube, local_ref, rs_random_state=3, **LOCAL)
    _, b = zeit.coregister_local(cube, local_ref, rs_random_state=3, **LOCAL)
    np.testing.assert_array_equal(a.outlier_ransac.values, b.outlier_ransac.values)
    _, c = zeit.coregister_local(cube, local_ref, rs_random_state=None, **LOCAL)
    assert len(c) == len(a)


def test_local_arosics_compat_only_changes_where_ransac_flags_land(scene, local_ref, local_result):
    cube, _, points = local_result
    _, quirk = zeit.coregister_local(cube, local_ref, arosics_compat=True, **LOCAL)
    for col in ("x_shift_px", "y_shift_px", "outlier_reliability", "outlier_ssim"):
        np.testing.assert_array_equal(quirk[col].values, points[col].values)
    # the flags of RANSAC are the only ones that move; the default puts them on the points RANSAC found
    arr = _warped(scene)
    arr[230:370, 230:370] = Scene(99).image(LY, LX)[230:370, 230:370]
    arr[130:270, 430:570] = 1234.0
    cube2 = _da(arr)
    _, fixed = zeit.coregister_local(cube2, local_ref, **LOCAL)
    _, quirk2 = zeit.coregister_local(cube2, local_ref, arosics_compat=True, **LOCAL)
    assert (fixed.outlier_reliability | fixed.outlier_ssim).any()
    assert fixed.outlier_ransac.sum() > 0 and quirk2.outlier_ransac.sum() <= fixed.outlier_ransac.sum()


def test_local_nodata_and_integer_cubes(scene, local_ref):
    arr = _warped(scene).astype(np.float32)
    arr[:, :200] = np.nan  # no data on the left 200 columns
    arr[520:, :] = np.nan  # and the bottom 120 rows
    aligned, points = zeit.coregister_local(_da(arr), local_ref, **LOCAL)
    # no point on the no-data margins
    assert (points.col >= 200).all() and (points.row <= 520).all()
    assert points.used.sum() >= 12 and points.date_applied.all()
    assert np.isnan(aligned.values[:, :100]).all()
    # an integer cube with a no-data value
    iarr = (_warped(scene) + 6000).astype(np.uint16)
    iarr[:, :200] = 0
    cube = _da(iarr).rio.write_nodata(0)
    ref = _da((scene.image(LY, LX) + 6000).astype(np.uint16)).rio.write_nodata(0)
    out, ipoints = zeit.coregister_local(cube, ref, **LOCAL)
    assert out.dtype == np.uint16 and (ipoints.col >= 200).all() and ipoints.date_applied.all()
    assert (out.values[:, :100] == 0).all()


def test_local_time_band_dimensions_and_a_date_with_too_few_points(scene, local_ref):
    good = _warped(scene)
    flat = np.full((LY, LX), 100.0, np.float32)
    arr = np.stack([np.stack([good, good * 0.5]), np.stack([flat, flat])]).astype(np.float32)
    bands = ["red", "nir"]
    cube = _da(arr, dims=("time", "band", "y", "x"), time=np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]"),
               band=bands)
    ref = xr.concat([local_ref, local_ref * 0.5], dim="band").assign_coords(band=bands)
    aligned, points = zeit.coregister_local(cube, ref, band="red", ref_band="red", **LOCAL)
    assert aligned.dims == cube.dims and aligned.shape == cube.shape
    dates = points.attrs["dates"]
    assert list(dates.index) == list(cube.time.values) and dates.index.name == "time"
    assert list(dates.applied) == [True, False]
    assert "no tie point matched" in dates.reason.iloc[1] and dates.reason.iloc[0] is None
    d0, d1 = (points[points.time == t] for t in cube.time.values)
    assert d0.date_applied.all() and not d1.date_applied.any() and not d1.success.any() and not d1.used.any()
    # the bad date is left as it was; both bands of the good one move with it
    np.testing.assert_array_equal(aligned.values[1], cube.values[1])
    np.testing.assert_allclose(aligned.sel(band="nir").values[0], aligned.sel(band="red").values[0] * 0.5, rtol=1e-4)
    assert not np.array_equal(aligned.values[0], cube.values[0])
    # fewer valid points than min_points: left as it was
    out, few = zeit.coregister_local(cube.isel(time=0), ref, band="red", ref_band="red", min_points=10_000, **LOCAL)
    assert "fewer than min_points" in few.attrs["dates"].reason.iloc[0] and not few.date_applied.any()
    np.testing.assert_array_equal(out.values, cube.values[0])
    # without bands and without dates
    one, pts = zeit.coregister_local(cube.sel(band="red").isel(time=0), local_ref, **LOCAL)
    assert one.dims == ("y", "x") and pts.date_applied.all()


def test_local_extents_may_differ(scene):
    ref = _da(scene.image(900, 1000))
    r, c = np.meshgrid(np.arange(500) + 0.5 + 60, np.arange(600) + 0.5 + 80, indexing="ij")
    dx, dy = _field(c, r)
    cube = _da(scene(c + dx, r + dy), col0=80, row0=60)
    aligned, points = zeit.coregister_local(cube, ref, grid_res=100, window_size=128, n_jobs=2)
    assert aligned.shape == cube.shape and points.date_applied.all()
    used = points[points.used]
    # the grid is the cube's own: columns and rows counted from its first cell
    assert (used.col % 100 == 0).all() and used.col.max() <= 600 and used.row.max() <= 500
    want_x, want_y = _field(used.col.values + 80, used.row.values + 60)
    assert np.abs(used.x_shift_px - want_x).max() < 0.6 and np.abs(used.y_shift_px - want_y).max() < 0.6


def test_local_errors(scene, local_ref):
    cube = _da(_warped(scene))
    with pytest.raises(ValueError, match="different CRS"):
        zeit.coregister_local(cube, _da(scene.image(LY, LX), crs="EPSG:32634"))
    with pytest.raises(ValueError, match="warp must be"):
        zeit.coregister_local(cube, local_ref, warp="spline")
    with pytest.raises(ValueError, match="order"):
        zeit.coregister_local(cube, local_ref, warp="tps", order=2)
    with pytest.raises(ValueError, match="square"):
        zeit.coregister_local(cube, local_ref, window_size=(128, 256))
    with pytest.raises(ValueError, match="grid_res"):
        zeit.coregister_local(cube, local_ref, grid_res=0)
    with pytest.raises(ValueError, match="tie_point_filter_level"):
        zeit.coregister_local(cube, local_ref, tie_point_filter_level=4)
    with pytest.raises(ValueError, match="rs_max_outlier"):
        zeit.coregister_local(cube, local_ref, rs_max_outlier=100)
    with pytest.raises(ValueError, match="unknown resampling"):
        zeit.coregister_local(cube, local_ref, resampling="sinc")
    with pytest.raises(ValueError, match="do not overlap"):
        zeit.coregister_local(cube, _da(scene.image(LY, LX), col0=5000))
    with pytest.raises(TypeError):
        zeit.coregister_local(cube.values, local_ref)
    # nothing on data: every date flagged, the cube returned as it is
    empty = _da(np.full((LY, LX), np.nan, np.float32))
    out, pts = zeit.coregister_local(empty, local_ref, **LOCAL)
    assert len(pts) == 0 and not pts.attrs["dates"].applied.any()
    assert "no tie point on data" in pts.attrs["dates"].reason.iloc[0]
