"""The dark targets and the AOD over each of them of FORCE's atmospheric correction (zeit._core.aod), ported from the first
half of FORCE 3.11's aod-ll.c. There is no public function yet; the comparison with FORCE's C code is
benchmarks/vs_originals/force_aod.py, whose harness printed the reference values below for this scene (dyadic rationals
and integers, the same bits in the harness and here)."""
import numpy as np

from zeit import _core

F32 = np.float32
NB, NX, NY, CS, GNX, GNY = 13, 60, 45, 15, 4, 3
GN = GNX * GNY
DOMAIN = ["ULTRABLUE", "BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "VAPOR", "CIRRUS",
          "SWIR1", "SWIR2"]
WVL = np.array([0.4375, 0.5, 0.5625, 0.65625, 0.703125, 0.75, 0.78125, 0.84375, 0.875, 0.9375, 1.375, 1.609375, 2.1875], F32)
E0 = np.array([1920, 1984, 1856, 1536, 1408, 1280, 1216, 1088, 960, 928, 360, 240, 90], F32)
MODB = np.array([0.25, 0.1875, 0.125, 0.0625, 0.0390625, 0.03125, 0.0234375, 0.015625, 0.01171875, 0.0078125, 0.001953125,
                 0.0009765625, 0.00048828125], F32)
T_W = np.array([1459, 1332, 1079, 637, 470, 376, 324, 262, 232, 105, 102, 83, 54])
T_V = np.array([1369, 1105, 1079, 620, 1051, 1993, 2860, 3135, 3245, 1691, 182, 1424, 323])
T_L = np.array([1733, 1603, 1564, 1624, 2174, 2505, 2772, 2867, 2974, 1545, 981, 3323, 2607])


def tiny_scene():
    """60 x 45 pixels of 10 m, a coarse grid of 4 x 3 cells of 150 m (the last column outside the swath); lakes A, B (with
    an island and a shadow pixel), C (steep, high), a pond and a stream, vegetation V1, V2, V3 (steep), cloud over land."""
    lwvl = np.log(WVL.astype(np.float64)).astype(F32)
    g = np.arange(GN)
    gi, gj = g // GNX, g % GNX
    sun = (0.625 + 0.015625 * gi + 0.0078125 * gj).astype(F32)
    view = (0.96875 - 0.0078125 * (gi + gj)).astype(F32)
    view[gj == GNX - 1] = -9999
    view_azi = (90 + 4 * g).astype(F32)
    view_azi[gj == GNX - 1] = -9999
    b = np.arange(NB)[:, None]
    tso = (0.96875 + 0.00390625 * (b % 4) + 0 * g[None]).astype(F32)
    tvo = (0.9765625 + 0.00390625 * (g[None] % 2) + 0 * b).astype(F32)
    tg = (0.90625 + 0.0078125 * (b % 5) + 0 * g[None]).astype(F32)
    tg[9] = 0.5
    hr = (0.96875 + g / 256).astype(F32)
    mod = (MODB[:, None] * hr[None]).astype(F32)
    brdf = (1 + (b % 3) / 32 + 0 * g[None]).astype(F32)
    ii, jj = np.mgrid[0:NY, 0:NX]
    cls = np.zeros((NY, NX), int)
    for r, c, v in [((2, 13), (18, 29), 1), ((18, 29), (2, 14), 1), ((22, 25), (6, 9), 0), ((33, 44), (33, 44), 1),
                    ((3, 6), (3, 6), 1), ((31, 35), (15, 30), 1), ((33, 44), (3, 14), 2), ((3, 14), (33, 44), 2),
                    ((20, 29), (33, 44), 2)]:
        cls[r[0]:r[1], c[0]:c[1]] = v
    wig = (ii + jj) % 3
    toa = np.zeros((NB, NY, NX), int)
    for k in range(NB):
        land = T_L[k] + 7 * ((ii * 3 + jj * 5) % 11)
        toa[k] = np.where(cls == 1, T_W[k], np.where(cls == 2, T_V[k], land)) + wig
    qai = np.zeros((NY, NX), np.int16)
    qai[:, 45:] |= 1
    qai[2:4, 18:21] |= 1 << 1
    qai[19, 3] |= 1 << 3
    qai[7, 23] |= 1
    qai[35, 5] |= 2 << 1
    qai[5, 35] |= 1 << 3
    qai[36:41, 20:31] |= 3 << 1
    toa[:, 36:41, 20:31] = 4500
    toa[:, qai & 1 != 0] = -9999
    slope = (40 + (ii * 7 + jj * 3) % 90).astype(np.uint16)
    slope[33:44, 33:44] = 300
    slope[20:29, 33:44] = 3000
    return dict(
        nb=NB, nx=NX, ny=NY, res=10.0, gnx=GNX, gny=GNY, gres=150.0, nodata_view=-9999, multi=True, domain=DOMAIN,
        rsr_band=(30 + np.arange(NB)).astype(np.int32), aod_bands=((WVL < 0.7) | (np.arange(NB) == 12)).astype(np.uint8),
        wvl=WVL, lwvl=lwvl, e0=E0, dem=np.array([0.75, 0.0, 2.0, 0.0078125], F32), sob=np.array([0.84375], F32),
        sun_czen=sun, sun_azi=(150 - g).astype(F32), view_czen=view, view_azi=view_azi, tso=tso.ravel(), tvo=tvo.ravel(),
        tg=tg.ravel(), mod=mod.ravel(), brdf=brdf.ravel(), pr=np.full(GN, 1.0625, F32),
        pa=(0.5 + 0.0625 * (g % 4)).astype(F32), hr=hr, fresnel=(0.03125 + 0.00390625 * (g % 3)).astype(F32),
        dem_bin=(32 + 40 * gi + 8 * gj).astype(F32), toa=toa.astype(np.int16).reshape(NB, -1).ravel(), qai=qai.ravel(),
        top_dem=(32 + 40 * (ii // CS) + 8 * np.minimum(jj // CS, GNX - 1) + (ii + jj) % 3).astype(np.uint8).ravel(),
        slope=slope.ravel(), ill=(8000 + (ii * 13 + jj * 7) % 1500).astype(np.int16).ravel(),
        sky=(9500 + (ii * 5 + jj * 11) % 400).astype(np.uint16).ravel())


def parse(raw, n_wat=4, n_veg=2):
    """The dumps of the stages, in the layout of force_aod/main.inc."""
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    obj = np.dtype([("i", "<i4", 12), ("f", "<f4", 7), ("ttoa", "<f4", NB), ("etoa", "<f4", NB)])
    out = {}
    for kind in ("wat", "veg"):
        out["ext_" + kind] = take(obj, int(take("<i4", 1)[0]))
    for kind, ln in (("wat", 26), ("veg", 81)):
        out["lib_" + kind] = take(np.dtype(("<f4", (ln, NB))), int(take("<i4", 1)[0]))
    fit = np.dtype([("i", "<i4", 3), ("rsq", "<f4"), ("coef", "<f4", 3), ("aod", "<f4", NB), ("est", "<f4", NB)])
    for kind, n in (("wat", n_wat), ("veg", n_veg)):
        out["k_" + kind] = int(take("<i4", 1)[0])
        out["fit_" + kind] = take(fit, n)
    out["hp"], out["ha"] = take("<f4", 2)
    out["xy_ha"] = take("<f4", GN)
    out["ha_obj"] = take("<f4", n_wat + n_veg)
    assert pos == len(raw)
    return out


def bits(a):
    return [int(x) for x in np.asarray(a, F32).ravel().view(np.uint32)]


def test_scene_equals_the_c_harness():
    raw, t_ext, t_fit = _core.aod.scene(tiny_scene())
    out = parse(raw)
    wat, veg = out["ext_wat"], out["ext_veg"]
    assert wat["i"].tolist() == [[114, 1, 2, 12, 18, 28, 7, 23, 0, 1, 1, 3], [9, 0, 3, 5, 3, 5, 0, 0, 0, 0, 0, 2],
                                 [122, 1, 18, 28, 2, 13, 23, 7, 1, 0, 4, 3], [56, 0, 31, 34, 16, 29, 0, 0, 0, 0, 0, 2]]
    assert veg["i"].tolist() == [[80, 1, 4, 12, 34, 42, 8, 38, 0, 2, 2, 6], [80, 1, 34, 42, 4, 12, 38, 8, 2, 0, 8, 6]]
    assert bits([wat["f"][0][5], wat["f"][0][3], wat["ttoa"][0][1], wat["etoa"][0][1]]) == [0x3EA60000, 0x3F535E25,
                                                                                           0x3E087FCC, 0x3E20C79F]
    assert bits([veg["f"][1][5], veg["f"][1][3], veg["ttoa"][1][1], veg["etoa"][1][1]]) == [0x3F630000, 0x3F5AE632,
                                                                                           0x3DE28241, 0x3E1FF1B2]
    # the band-wise AOD that matches best, for each spectrum of the library
    sums = lambda a: [int(np.asarray(x, F32).view(np.uint32).astype(np.uint64).sum()) for x in a]  # noqa: E731
    assert sums(out["lib_wat"]) == [163181446271, 159002343490]
    assert sums(out["lib_veg"]) == [442601072498, 405873465065]
    # the fits
    assert (out["k_wat"], out["k_veg"]) == (2, 2)
    f = out["fit_wat"]
    assert f["i"].tolist() == [[1, 5, 1], [0, 0, 0], [1, 5, 1], [0, 0, 0]]
    assert bits(f["rsq"][[0, 2]]) == [0x3F2F1172, 0x3F37693A]
    assert bits(f["coef"][0]) == [0x3D3129CF, 0xBFB33EEA, 0]
    f = out["fit_veg"]
    assert f["i"].tolist() == [[1, 3, 1], [1, 3, 1]]
    assert bits(f["rsq"]) == [0x3F7FB391, 0x3F7F16B0]
    assert bits(f["aod"][:, 1]) == [0x3E0482E6, 0x3E2E3A91]
    # the elevation dependency: no minimum in the interval, the AOD is flat
    assert bits([out["hp"], out["ha"]]) == [0x447A0000, 0x3F7FCEDE]
    assert bits(out["ha_obj"]) == [0x3F7FEAC1, 0, 0x3F7FDA61, 0, 0x3F7FE6A9, 0x3F7FC5EA]
    assert t_ext >= 0 and t_fit >= 0


def test_a_scene_without_candidates_and_a_missing_band():
    sc = tiny_scene()
    sc["toa"] = (np.tile(T_L[:, None], (1, NX * NY)) + 11).astype(np.int16).ravel()  # land only
    raw, _, _ = _core.aod.scene(sc)
    out = parse(raw, 0, 0)
    assert len(out["ext_wat"]) == 0 and len(out["ext_veg"]) == 0 and out["k_wat"] == 0
    assert float(out["hp"]) == 1000.0
    sc = tiny_scene()
    sc["domain"] = [d if d != "SWIR2" else "" for d in DOMAIN]
    try:
        _core.aod.scene(sc)
    except ValueError as e:
        assert "SWIR2" in str(e)
    else:
        raise AssertionError("a missing domain must be an error")


def test_elevation_dependency_finds_the_scale_height():
    zs = [F32(z) for z in (0.0625, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    aod = [F32(0.25 * np.exp(-z / 1.5)) for z in zs]
    view, dembin = [F32(0.96875)] * 3 + [F32(-9999)], [F32(v) for v in (16, 64, 128, 200)]
    (res,) = _core.aod.elev_cases([(F32(0.75), F32(0.0), F32(2.0), F32(0.0078125), view, dembin, list(zip(zs, aod)))])
    assert abs(float(np.array([res[0]], np.uint32).view(F32)[0]) - 1.5) < 0.01  # the scale height of the decay
    # a flat DEM: the flat AOD, Hp = 1000
    (res,) = _core.aod.elev_cases([(F32(0.75), F32(0.5), F32(0.5), F32(0.0078125), view, dembin, list(zip(zs, aod)))])
    assert res[0] == 0x447A0000


def test_aod_files(tmp_path):
    wvl = [0.5, 0.65, 2.2]
    lwvl = [float(np.log(w)) for w in wvl]
    (tmp_path / "AOD_045.txt").write_text("10.0 50.0 -2.0 -1.0 -0.1\r\n12.0  52.0 -3.0 -1.5 0.0\n")
    d = str(tmp_path)
    got = _core.aod.aodfile(wvl, lwvl, [("NULL", 0.0, 0.0, 45), (d, 10.5, 50.5, 45), (d, 11.9, 52.1, 45),
                                       (d, 10.0, 50.0, 46)])
    fallback, first, second, missing = got
    assert missing is None
    spec = lambda g: np.array(g, np.uint32).view(F32)  # noqa: E731
    assert spec(fallback)[0] > spec(fallback)[1] > spec(fallback)[2] > 0
    # the first site: exp(-2) * wvl^(-1 - 0.1 ln wvl)
    want = np.exp(-2.0) * np.array(wvl) ** (-1.0 - 0.1 * np.array(lwvl))
    assert np.allclose(spec(first), want, rtol=1e-6)
    want = np.exp(-3.0) * np.array(wvl) ** -1.5
    assert np.allclose(spec(second), want, rtol=1e-6)


def test_the_image_functions():
    img = np.zeros((9, 12), np.uint8)
    img[1:4, 1:5] = 1
    img[4, 5] = 1  # touches the block by its corner: one segment (8-connected)
    img[6:9, 8:12] = 1
    dist, lab, n = _core.aod.image_functions(img.ravel(), 12, 9)
    assert n == 2
    lab = lab.reshape(9, 12)
    assert lab[1, 1] == lab[4, 5] == 1 and lab[7, 9] == 2 and lab[0, 0] == 0
    # the distance of each cell to the next TRUE cell, truncated
    ti, tj = np.nonzero(img)
    ii, jj = np.mgrid[0:9, 0:12]
    want = np.floor(np.sqrt(((ii[..., None] - ti) ** 2 + (jj[..., None] - tj) ** 2).min(-1))).astype(int)
    assert (dist.reshape(9, 12) == want).all()


def test_the_libraries():
    water, veg = _core.aod.libraries([30 + k for k in range(NB)], 1, 2, 3)
    water, veg = water.reshape(-1, NB), veg.reshape(-1, NB)
    assert water.shape == (26, NB) and veg.shape == (81, NB)
    assert (water[:, 12] == 0).all() and (water[:, 1] > 0).all()  # the library ends at 900 nm
    assert veg[0, 1] == F32(0.01) and veg[0, 2] == F32(0.02) and (veg[:, 1] == veg[:, 3]).all()
