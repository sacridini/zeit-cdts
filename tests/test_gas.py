"""The water vapour and gas transmittance of FORCE's atmospheric correction (zeit._core.gas), ported from FORCE 3.11's
gas-ll.c. There is no public function yet; the comparison with FORCE's C code is benchmarks/vs_originals/force_gas.py,
whose harness printed the reference values below for this scene (dyadic rationals and integers, the same bits in the
harness and here)."""
import numpy as np

from zeit import _core

F32 = np.float32
NB, NX, NY = 13, 24, 24
B_NIR, B_VAP, RSR0 = 8, 9, 30


def tiny_scene():
    gn = 4
    sun = (0.5 + 0.0625 * np.arange(gn)).astype(F32)
    view = (0.96875 - 0.015625 * np.arange(gn)).astype(F32)
    view[3] = -9999
    b = np.arange(NB)[:, None]
    g = np.arange(gn)[None, :]
    tso = (0.96875 + 0.00390625 * (b % 4) + 0 * g).astype(F32)
    tvo = (0.9765625 + 0.00390625 * (g % 2) + 0 * b).astype(F32)
    z = np.arange(256)[:, None, None]
    rho_p = ((20 + b[None] + 2 * g[None] + z % 5) / 512.0).astype(F32)
    t = ((470 - z + b[None] + g[None]) / 512.0).astype(F32)
    s = ((40 + b[None] + g[None] + 0 * z) / 512.0).astype(F32)
    ii, jj = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    dem = (10 + 2 * ii + jj).astype(np.uint8).ravel()
    nir = np.where((ii + jj) % 11 == 0, 1200, 2500 + 7 * ii + 3 * jj)
    vap = 550 + ((nir - 500) * (45 + (ii * jj) % 13)) // 100
    vap[0:6, 6:12] = 6000
    vap[12:18, 0:6] = nir[12:18, 0:6] + 900
    toa = np.stack([1000 + 37 * k + ii + jj for k in range(NB)]).astype(np.int16)
    toa[B_NIR], toa[B_VAP] = nir, vap
    q = np.zeros((NY, NX), np.int32)
    q[:, -3:] |= 1
    q[6:12, 12:18] |= 1 << 5
    q[((ii * 5 + jj * 3) % 17) == 0] |= 1 << 3
    q[18:24, 6:12] |= 1 << 5
    q[((ii * 7 + jj) % 19) == 0] |= 1 << 1
    return (NX, NY, 10.0, 2, 2, 120.0, -9999, B_NIR, B_VAP, (RSR0 + np.arange(NB)).astype(np.int32),
            np.array([0.5, 0.6875], F32), np.array([0.9375, 0.96875], F32), sun, view, tso.ravel(), tvo.ravel(),
            rho_p.ravel(), t.ravel(), s.ravel(), toa.ravel(), q.astype(np.int16).ravel(), dem)


def test_scene_equals_the_c_harness():
    lut, wvp, qai, tg, w_avg, n = _core.gas.water_vapor_scene(*tiny_scene())
    assert lut.shape == (NB * 701 * 101,) and int((lut != 0).sum()) == 437424
    assert n == 8 and np.array([w_avg], F32).view(np.uint32)[0] == 0x3F298000
    blocks = wvp.reshape(NY, NX)[::6, ::6]
    assert blocks.tolist() == [[2125, 662, 531, 515], [437, 281, 662, 406], [662, 406, 662, 662], [593, 662, 662, 662]]
    assert int(wvp.astype(np.int64).sum()) == 340830
    assert int(((qai >> 14) & 1).sum()) == 252  # the blocks filled with the scene average
    tgr = tg.reshape(NB, NY * NX)
    assert [int(r.astype(np.int64).sum()) for r in tgr] == [3748662, 3761820, 3773286, 3766806, 3624120, 3680100,
                                                            3747348, 3587940, 3744090, 1470042, 8352, 3777696,
                                                            3639618]


def test_lut_files(tmp_path):
    (tmp_path / "WVP_0000-03-00.txt").write_text("10.0 50.0 1.5 12\n10.5 50.5 2.5 12\n20.0 60.0 4.0 12\n")
    (tmp_path / "WVP_2020-03-14.txt").write_text("10.0 50.0 9999 TBD\r\n10.9 50.9 3.25 MOD # a comment\n")
    d = str(tmp_path)
    got = _core.gas.water_vapor_from_lut([
        (d, 10.0, 50.0, 2020, 3, 14, 2.0, False),  # the daily nearest site is nodata: the climatology's
        (d, 10.9, 50.9, 2020, 3, 14, 2.0, False),  # the daily value
        (d, 10.4, 50.4, 2020, 3, 20, 2.0, False),  # no daily file
        (d, 30.0, 30.0, 2020, 3, 14, 2.0, False),  # farther than 1.5 degrees: nothing
        (d, 30.0, 30.0, 2020, 3, 14, 2.0, True),  # strict: a failure
        ("NULL", 0.0, 0.0, 2020, 3, 14, 2.0, True),  # the constant of the parameter file
    ])
    assert got == [1.5, 3.25, 2.5, -1.0, -1.0, 2.0]
