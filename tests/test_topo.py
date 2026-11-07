"""The topography of FORCE's atmospheric correction (zeit._core.topo), ported from FORCE 3.11's topo-ll.c: slope, aspect,
illumination angle, sky view factor and binned elevation, the C-factor, and the average elevation of the coarse cells. There
is no public function yet; the comparison with FORCE's C code is benchmarks/vs_originals/force_topo.py, whose harness printed
the reference values below for this scene (dyadic rationals and integers, the same bits in the harness and here)."""
import numpy as np
import pytest

from zeit import _core

F32 = np.float32
NX, NY, CS = 40, 32, 16
GNX, GNY = 3, 2
GN = GNX * GNY
ND = -32767


def tiny_scene(flat=False, do_c=True, dem_scale=1.0, qai_off=True):
    """40 x 32 pixels of 10 m, a coarse grid of 3 x 2 cells of 160 m; a hill (a paraboloid), a ridge, a hole of no-data, a
    sea of 0 m next to no-data, a frame of off pixels, a sun that changes over the grid."""
    ii, jj = np.mgrid[0:NY, 0:NX]
    dem = (600 - 6 * (np.abs(ii - 14) + np.abs(jj - 20)) + ((jj * 3 + ii) % 5) / 4).astype(F32) * F32(dem_scale)
    dem[:, :6] = 0.0  # sea
    dem[10:13, 6:8] = ND  # its coast without data
    dem[24:27, 30:33] = ND  # a hole
    qai = np.zeros((NY, NX), np.int16)
    if qai_off:
        qai[:1, :] = 1
        qai[:, -2:] = 1
        qai[20, 20] = 1
        qai[5, 10] = 0b1010  # cloud, already set
    g = np.arange(GN)
    gi, gj = g // GNX, g % GNX
    czen = (0.75 + 0.03125 * gi - 0.015625 * gj).astype(F32)
    szen = (0.625 - 0.03125 * gi + 0.015625 * gj).astype(F32)
    zen = (0.75 - 0.0625 * gi + 0.03125 * gj).astype(F32)
    azi = (2.5 + 0.125 * gj - 0.0625 * gi).astype(F32)
    sc = dict(nx=NX, ny=NY, res=10.0, gnx=GNX, gny=GNY, gres=10.0 * CS, dem_nodata=ND, flat=flat, do_c=do_c, sun_zen=zen,
              sun_azi=azi, sun_czen=czen, sun_szen=szen, dem=dem.ravel(), qai=qai.ravel())
    if do_c:
        i16 = np.int16
        toa = np.zeros((4, NY, NX), i16)
        toa[0] = 500 + (ii * jj) % 17
        toa[1] = 800 + (np.abs(ii - 14) * 7 + jj * 3) % 400 + (ii > 16) * 150
        toa[2] = 400 + (ii + 2 * jj) % 13
        toa[3] = 2000 + (np.abs(ii - 14) * 5 + jj) % 300 + (ii > 16) * 350
        z = np.arange(256)[:, None]
        sc.update(domain=["GREEN", "SWIR2", "RED", "SWIR1"], toa=toa.ravel(),
                  rho_p=(0.0078125 + (255 - z) / 65536.0 + g[None] / 4096.0).astype(F32).ravel(),
                  tss=(0.625 + (255 - z) / 1024.0 + g[None] / 256.0).astype(F32).ravel(),
                  tsd=(0.875 + (255 - z) / 4096.0 + 0 * g[None]).astype(F32).ravel())
    return sc


def parse(raw, sc):
    npix, gn = sc["nx"] * sc["ny"], sc["gnx"] * sc["gny"]
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    o = {"ret": int(take("<i4", 1)[0]), "stats": take("<f4", 4), "cnum": int(take("<i4", 1)[0]), "qai": take("<i2", npix)}
    if o["ret"] == 0:
        for k, dt in (("dem", "u1"), ("slope", "<u2"), ("aspect", "<u2"), ("ill", "<i2"), ("sky", "<u2")):
            o[k] = take(dt, npix)
        if sc["do_c"]:
            o["cnull"] = int(take("<i4", 1)[0])
            o["cf"] = take("<u2", npix)
        o["xy_dem"] = take("<i4", gn)
    return o


def run(sc):
    raw, _, _ = _core.topo.scene(sc)
    return parse(bytes(raw), sc)


def test_reference_values_of_force():
    o = run(tiny_scene())
    assert o["ret"] == 0 and o["cnum"] == 255 and o["cnull"] == 0
    assert o["stats"].tolist() == REF["stats"]
    for k in ("dem", "slope", "aspect", "ill", "sky", "cf", "qai"):
        assert int(o[k].astype(np.int64).sum()) == REF[k], k
    assert list(o["xy_dem"]) == REF["xy_dem"]
    assert int(o["slope"].max()) == REF["slope_max"] and int(o["ill"].min()) == -10000


def test_flat_dem():
    # FORCE with no DEM file: 0 m everywhere, no slope, the illumination of the sun, the sky fully visible
    sc = tiny_scene(flat=True, do_c=False)
    o = run(sc)
    assert o["ret"] == 0 and not o["slope"].any() and not o["aspect"].any()
    assert (o["sky"][sc["qai"] == 0] == 10000).all()
    assert tuple(o["stats"][1:3]) == (0.0, F32(0.001))
    on = sc["qai"] & 1 == 0
    assert (o["ill"][~on] == -10000).all() and (o["ill"][on] > 0).all()
    assert (o["xy_dem"] == 0).all()


def test_dem_out_of_bounds_fails_like_force():
    o = run(tiny_scene(do_c=False, dem_scale=30.0))  # up to 18 km
    assert o["ret"] == 1 and o["stats"][2] > 9
    o = run(tiny_scene(do_c=False, dem_scale=-1.0))
    assert o["ret"] == 1 and o["stats"][1] < -0.5


def test_all_pixels_off():
    sc = tiny_scene(do_c=False)
    sc["qai"] = np.ones_like(sc["qai"])
    o = run(sc)
    assert o["ret"] == 0 and np.isnan(o["stats"][0]) and (o["ill"] == -10000).all()


def test_bad_inputs_are_errors():
    sc = tiny_scene(do_c=False)
    sc["ny"], sc["nx"] = 2, 2
    with pytest.raises(ValueError):
        _core.topo.scene(sc)
    sc = tiny_scene(do_c=False)
    del sc["sun_szen"]
    with pytest.raises(ValueError):
        _core.topo.scene(sc)


# printed by the harness of FORCE (force_topo.py's run_force on tiny_scene()): the sums of the fields, dem.avg/min/max/step
# (float32 values as doubles), the average elevation bins of the coarse cells
REF = {"dem": 212531, "slope": 8692051, "aspect": 35063226, "ill": 3964431, "sky": 8752735, "cf": 30048531, "qai": 11004042,
       "stats": [0.4213554859161377, 0.0, 0.5809183716773987, 0.0022781111765652895], "xy_dem": [133, 233, 209, 134, 226, 200],
       "slope_max": 14458}
