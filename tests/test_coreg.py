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
