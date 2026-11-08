"""The atmosphere of the coarse grid of FORCE's atmospheric correction (zeit._core.atmo), ported from FORCE 3.11's atc-ll.c
(allocate_atc), sunview-ll.c (sun_target_view, viewgeo, view_angle) and, from atmo-ll.c, interpolation_weights,
interpolate_coarse, atmo_angledep, atmo_elevdep and apply_aoi. There is no public function yet; the comparison with
FORCE's C code is benchmarks/vs_originals/force_atmo.py (the sun position is Zeit's on both sides, see zeit_core::sunview).
These tests check the structure after allocate_atc against the values of init_atc (the harness printed the same bits for
FORCE), the geometry against the formulas, and the stages against the radiative transfer functions of zeit._core.radtran."""
import math

import numpy as np

from zeit import _core

F32 = np.float32
D2R = 0.01745329251994329547
NX, NY, RES = 300, 250, 100.0  # 6 x 5 coarse cells of 5000 m
GNX, GNY = 6, 5
GN = GNX * GNY
NB = 4
DOMAIN = ["BLUE", "GREEN", "RED", "NIR"]
RSR = np.array([31, 32, 33, 38], np.int32)
XY = 18
XYZ = 16


def rt(name, *cols):
    return _core.radtran.radtran_eval(name, *[np.ascontiguousarray(c, F32) for c in cols])


def scene(mission=1, aodmap=True, doenv=True, domulti=True, nodata_cell=(1, 1), npts=0, aoi=False):
    """A scene of 300 x 250 pixels of 100 m (6 x 5 coarse cells), 4 bands, a DEM binned in 40 steps, an AOD per cell."""
    rng = np.random.default_rng(5)
    wvl = rt("wavelength", RSR)[0]
    lon0, x0, kgeo = 10.0, 500000.0, 9.0e-6
    ulx, uly = x0, 48.0 / kgeo
    qai = np.zeros((NY, NX), np.int16)
    ii, jj = np.mgrid[0:NY, 0:NX]
    qai[:2, :] = 1
    qai[ii == 100] |= 0b1010  # a cloud, already set
    top_dem = ((jj // 8 + ii // 8) % 40).astype(np.uint8)
    xy_aod = np.stack([(0.2 + 0.02 * np.arange(GN)) * (wvl[b] / 0.55) ** -1.3 for b in range(NB)]).astype(F32)
    sc = dict(mission=mission, nx=NX, ny=NY, res=RES, ulx=ulx, uly=uly, domain=DOMAIN, wavelength=wvl, rsr_band=RSR,
              year=2021, doy=172, hh=10, mm=30, ss=0, tz=0, lon0=lon0, x0=x0, kgeo=kgeo, domulti=int(domulti),
              doenv=int(doenv), aodmap=int(aodmap), dem4=np.array([0.5, 0.25, 4.0, 0.0625], F32), cnum=255,
              wvp_hp_ha=np.array([2.5, 2.0, 0.875], F32), mod=rt("molecular_optical_depth", wvl)[0],
              aod=np.array([0.25, 0.1875, 0.125, 0.0625], F32), gn=GN, xy_aod=xy_aod.ravel(),
              xy_interp=(np.arange(GN) % 3 == 0).astype(F32), qai=qai.ravel(), top_dem=top_dem.ravel(), s2nx=0, s2ny=0,
              s2nodata=-9999.0, vzen=np.zeros(0, F32), vazi=np.zeros(0, F32), w_ji=np.zeros(0, np.int32),
              coarse=np.zeros(0, F32), w_res=5000.0, w_full_res=100.0, w_nf=0, w_ne=0, w_nodata=-9999.0,
              aoi=np.zeros(0, np.uint8))
    if mission == 1:
        vzen = (2.0 + 0.5 * np.arange(GN)).astype(F32)
        vazi = (100.0 + 2.5 * (np.arange(GN) % 7)).astype(F32)
        if nodata_cell:
            vzen[nodata_cell[0] * GNX + nodata_cell[1]] = -9999.0
        sc.update(s2nx=GNX, s2ny=GNY, vzen=vzen, vazi=vazi)
    if npts:
        coarse = (np.arange(12, dtype=F32).reshape(3, 4) * 10).copy()
        coarse[1, 2] = -9999.0
        sc.update(w_ji=np.array([[50, 50], [0, 0], [199, 149], [75, 75], [100, 50]], np.int32).ravel(), coarse=coarse.ravel(),
                  w_nf=4, w_ne=3)
    if aoi:
        sc["aoi"] = (jj.ravel() % 3 != 0).astype(np.uint8)
    return sc


def parse(raw, sc):
    npix, gn, nb = NX * NY, GN, NB
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    def meta():
        n, nx, ny = (int(v) for v in take("<i4", 3))
        res = float(take("<f8", 1)[0])
        nod = take("<i4", n)
        dom = bytes(take("u1", 16 * n))
        return dict(nb=n, nx=nx, ny=ny, res=res, nodata=tuple(nod), domain=[dom[16 * k:16 * k + 16].rstrip(b"\0").decode() for k in range(n)])

    o = {}
    o["nx_ny_nc_res"] = tuple(int(v) for v in take("<i4", 4))
    o["nodata"] = float(take("<f4", 1)[0])
    o["cos"] = take("<f4", 4)
    o["scal"] = take("<f4", 8)  # wvp, cc, Hr, Ha, Hp, km, Fr, Fa
    o["dem"] = take("<f4", 4)
    o["cnum"] = int(take("<i4", 1)[0])
    o["view0"] = take("<f4", 7)
    o["tthg"] = take("<f4", 10)
    o["aodmap"] = int(take("<i4", 1)[0])
    o["band"] = np.array([np.concatenate([take("<f4", 7).astype(np.float64), take("<i4", 1)]) for _ in range(nb)])
    o["xy_meta"] = [meta() for _ in range(XY)]
    o["xyz_meta"] = [meta() for _ in range(2 * XYZ)]
    assert int(take("<i4", 1)[0]) == 0
    o["sun"] = take("<f4", 8 * gn).reshape(8, gn)
    o["view"] = take("<f4", 8 * gn).reshape(8, gn)
    o["cos_sv"] = take("<f4", 4)
    o["view_geo"] = take("<f4", 7)
    o["qai_sun"] = take("<i2", npix)
    assert int(take("<i4", 1)[0]) == 0
    for k, n in (("brdf", nb), ("psi", 1), ("pr", 1), ("pa", 1), ("tsw", nb), ("tvw", nb), ("tso", nb), ("tvo", nb), ("tg", nb),
                 ("dem_bin", 1), ("hr", 1), ("mod", nb), ("fresnel", 1)):
        o["a_" + k] = take("<f4", n * gn).reshape(n, gn)
    assert int(take("<i4", 1)[0]) == 0
    for k, n in (("od", nb), ("mod", nb), ("aod", nb), ("rho_p", nb), ("ts", nb), ("tsd", nb), ("tss", nb), ("tv", nb),
                 ("tvd", nb), ("tvs", nb), ("t", nb), ("s", nb), ("f", nb), ("hr", 1), ("ha", 1)):
        o["e_" + k] = take("<f4", 256 * n * gn).reshape(256, n, gn)
    o["qai_elev"] = take("<i2", npix)
    wd = np.dtype([("g", "<i4", 4), ("w", "<f4", 6), ("v", "<f4")])
    o["weights"] = take(wd, len(sc["w_ji"]) // 2)
    o["aoi_ret"] = int(take("<i4", 1)[0])
    o["qai_aoi"] = take("<i2", npix)
    take("<f8", 3)
    assert pos == len(raw)
    return o


def run(sc, n_jobs=1):
    raw, *_ = _core.atmo.scene({**sc, "n_jobs": n_jobs})
    return parse(bytes(raw), sc)


def test_allocation_has_the_initial_values_of_init_atc():
    o = run(scene())
    assert o["nx_ny_nc_res"] == (6, 5, 30, 5000) and o["nodata"] == -9999.0
    # the Henyey-Greenstein terms with the C's arithmetic (the bits the FORCE harness printed)
    g1, g2, alpha = F32(0.801), F32(0.58), F32(0.948)
    hg = [(1.0 - float(g1 * g1)) * float(alpha), (1.0 - float(g2 * g2)) * (1.0 - float(alpha)), 1.0 + float(g1 * g1),
          1.0 + float(g2 * g2), 2.0 * float(g1), 2.0 * float(g2)]
    sob = 3.0 * float(F32(F32(alpha * F32(g1 + g2)) - g2))
    np.testing.assert_array_equal(o["tthg"], np.array([g1, g2, alpha, *hg, sob], F32))
    # the environmental weighting for 100 m pixels: km = 0.1 + 0.05
    km = F32(0.1 + 0.05)
    assert o["scal"][5] == km
    np.testing.assert_array_equal(o["scal"][6:], [F32(1.0 - (0.930 * math.exp(-0.8 * float(km)) + 0.070 * math.exp(-1.10 * float(km)))),
                                                  F32(1.0 - (0.375 * math.exp(-0.2 * float(km)) + 0.625 * math.exp(-1.83 * float(km))))])
    # what later stages set starts as no-data: wvp, cc, Hr, Ha, Hp, the DEM statistics, the cosines, the view line
    np.testing.assert_array_equal(o["scal"][:5], [-9999.0] * 5)
    assert o["cnum"] == -9999 and (o["cos"] == -9999.0).all() and (o["view0"] == -9999.0).all() and o["aodmap"] == 0
    wvl = rt("wavelength", RSR)[0]
    np.testing.assert_array_equal(o["band"][:, 0], wvl)
    np.testing.assert_array_equal(o["band"][:, 1], np.array([math.log(float(w)) for w in wvl], F32))
    np.testing.assert_array_equal(o["band"][:, 2], (o["band"][:, 1].astype(F32) * o["band"][:, 1].astype(F32)))
    np.testing.assert_array_equal(o["band"][:, 3], rt("e0", RSR)[0])
    assert (o["band"][:, 4:7] == -9999.0).all() and (o["band"][:, 7] == 0).all()
    # the bricks: per band the domain of the image, of band 0 for the bricks with fewer bands (copy_brick), 256 elevation bins
    for m in o["xy_meta"] + o["xyz_meta"]:
        assert (m["nx"], m["ny"], m["res"]) == (6, 5, 5000.0) and set(m["nodata"]) == {-9999}
    mods = o["xy_meta"][0]
    assert mods["nb"] == NB and mods["domain"] == DOMAIN
    assert o["xy_meta"][12]["nb"] == 1 and o["xy_meta"][12]["domain"] == ["BLUE"]  # fresnel
    assert o["xy_meta"][15]["nb"] == 8 and o["xy_meta"][15]["domain"] == ["BLUE"] * 8  # view
    assert o["xyz_meta"][6]["domain"] == ["BLUE"] and o["xyz_meta"][10]["nb"] == NB  # Hr[0], rho_p[0]


def test_sentinel2_view_angles_and_the_cell_without_data():
    sc = scene()
    o = run(sc)
    g_nodata = 1 * GNX + 1
    view, vzen, vazi = o["view"], sc["vzen"], sc["vazi"]
    for g in range(GN):
        if g == g_nodata:
            assert (view[:, g] == -9999.0).all()
            continue
        zen, azi = F32(float(vzen[g]) * D2R), F32(float(vazi[g]) * D2R)
        assert view[0, g] == zen and view[1, g] == azi
        assert view[2, g] == F32(math.cos(float(zen))) and view[5, g] == F32(math.sin(float(azi)))
    # the pixels of the cell (100 m, 5000 m a cell: 50 pixels) are off, and only they (besides the 2 first rows)
    off = (o["qai_sun"].reshape(NY, NX) & 1).astype(bool)
    expect = np.zeros((NY, NX), bool)
    expect[:2, :] = True
    expect[50:100, 50:100] = True
    assert (off == expect).all()
    # the range of the cosines does not include the cell without data
    c = view[2][np.arange(GN) != g_nodata]
    assert o["cos_sv"][2] == c.min() and o["cos_sv"][3] == c.max()
    assert (o["view_geo"][5], o["view_geo"][6]) == (F32(786000.0), F32(F32(786000.0) * F32(786000.0)))
    # the sun of June 21 at 48 N, 10:30 UTC at 10 E: high, no low sun flag
    assert 0.3 < o["sun"][0].min() and o["sun"][0].max() < 0.6 and not ((o["qai_sun"] >> 10) & 1).any()


def test_landsat_view_geometry_from_the_swath():
    sc = scene(mission=0)
    nx, ny = NX, NY
    qai = np.ones((ny, nx), np.int16)
    ii, jj = np.mgrid[0:ny, 0:nx]
    shift = (ii * 0.2126).astype(int)
    qai[(jj >= shift) & (jj < shift + 200)] = 0  # a parallelogram of on pixels
    sc["qai"] = qai.ravel()
    o = run(sc)
    a, b, c, ab, angle, h, h2 = o["view_geo"]
    assert (h, h2) == (F32(705000.0), F32(F32(705000.0) * F32(705000.0)))
    # the corners FORCE looks for: the upper left (0, 0), the right-most pixel the first time (column 251, row 245), the
    # left-most the last time (column 0, row 4) and the lower right (column 251, row 249): the middle of the top is
    # (125, 122), the middle of the bottom (125, 126)
    assert (a, b, c, ab) == (-4.0, 0.0, 500.0, 4.0)
    zen = o["view"][0]
    assert 0 <= zen.min() and zen.max() < 0.1
    # right and left looking azimuths differ by pi
    azi = o["view"][1]
    assert abs(float(azi.max() - azi.min()) - np.pi) < 1e-5
    # views of Landsat have no cell without data
    assert not (o["view"] == -9999.0).any()


def test_low_sun_flag_before_dawn():
    sc = scene(mission=0)
    sc.update(hh=3, doy=355)  # 3 UTC in December at 10 E: the sun is below the horizon
    o = run(sc)
    assert (o["sun"][0] > 1.5).all() and o["cos_sv"][0] < 0
    low = ((o["qai_sun"] >> 10) & 1).astype(bool)
    off = (o["qai_sun"] & 1).astype(bool)
    assert (low == ~off).all()


def test_angle_dependent_parameters_follow_the_radiative_transfer_functions():
    sc = scene()
    o = run(sc)
    g = 8
    ms, mv = o["sun"][2, g], o["view"][2, g]
    psi = rt("backscatter", [ms], [mv], [o["sun"][1, g]], [o["view"][1, g]])[0][0]
    assert o["a_psi"][0, g] == psi
    assert o["a_pr"][0, g] == rt("phase_molecular", [psi])[0][0]
    hg = o["tthg"][3:9]
    assert o["a_pa"][0, g] == rt("phase_aerosol", [psi], *[[v] for v in hg])[0][0]
    assert o["a_fresnel"][0, g] == rt("fresnel_reflection", [o["sun"][0, g]])[0][0]
    # the elevation bin of the cell (0, 0): the rounded mean of the binned DEM of its 50 x 50 pixels, off pixels left out
    qai = sc["qai"].reshape(NY, NX)
    top = sc["top_dem"].reshape(NY, NX)[:50, :50].astype(np.float64)
    on = (qai[:50, :50] & 1) == 0
    assert o["a_dem_bin"][0, 0] == np.floor(top[on].mean() + 0.5)
    # Hr and MOD at that elevation
    dem0 = int(o["a_dem_bin"][0, 0])
    hr = rt("mod_elev_factor", [F32(dem0 * F32(0.0625) + F32(0.25) + F32(0.0625) / F32(2.0))])[0][0]
    assert o["a_hr"][0, 0] == hr
    for b in range(NB):
        assert o["a_mod"][b, 0] == F32(sc["mod"][b] / F32(1.0) * hr)
        assert o["a_brdf"][b, 0] == 1.0  # brdf-ll.c is not ported: no correction
        tsw = rt("wvp_transmitt", [2.5], [o["sun"][2, 0]], [RSR[b]])[0][0]
        assert o["a_tsw"][b, 0] == tsw
    # a cell without view angles is not touched
    assert (o["a_tg"][:, 1 * GNX + 1] == 0).all() and o["a_psi"][0, 1 * GNX + 1] == 0


def test_elevation_dependent_parameters_for_every_bin():
    sc = scene()
    o = run(sc)
    g = 8
    ms, mv = o["sun"][2, g], o["view"][2, g]
    pr, pa = o["a_pr"][0, g], o["a_pa"][0, g]
    zmed = F32(0.25 + 0.0625 / 2.0)
    for z in (0, 1, 100, 255):
        zm = zmed
        for _ in range(z):
            zm = F32(zm + F32(0.0625))
        hr, ha = rt("mod_elev_factor", [zm])[0][0], rt("aod_elev_factor", [zm], [2.0])[0][0]
        assert o["e_hr"][z, 0, g] == hr and o["e_ha"][z, 0, g] == ha
        for b in range(NB):
            mod = rt("mod_elev_scale", [sc["mod"][b]], [1.0], [hr])[0][0]
            aod = rt("aod_elev_scale", [sc["xy_aod"][b * GN + g]], [0.875], [ha])[0][0]
            od = rt("optical_depth", [aod], [mod])[0][0]
            t, ts, tv, tsd, tss, tvd, tvs = (v[0] for v in rt("scatt_transmitt", [aod], [mod], [od], [ms], [mv]))
            assert (o["e_mod"][z, b, g], o["e_aod"][z, b, g], o["e_od"][z, b, g]) == (mod, aod, od)
            assert (o["e_t"][z, b, g], o["e_ts"][z, b, g], o["e_tv"][z, b, g]) == (t, ts, tv)
            assert (o["e_tsd"][z, b, g], o["e_tss"][z, b, g], o["e_tvd"][z, b, g], o["e_tvs"][z, b, g]) == (tsd, tss, tvd, tvs)
            rho = rt("path_ref", [1], [o["tthg"][9]], [aod], [mod], [od], [pa], [pr], [tsd], [tvd], [ms], [mv])[0][0]
            assert o["e_rho_p"][z, b, g] == rho
            assert o["e_s"][z, b, g] == rt("sphere_albedo", [aod], [mod], [od])[0][0]
            fa, fr = o["scal"][7], o["scal"][6]
            assert o["e_f"][z, b, g] == rt("env_weight", [aod], [mod], [fa], [fr])[0][0]
    # the cell without view angles keeps zeros
    assert (o["e_t"][:, :, 1 * GNX + 1] == 0).all()
    # without the environmental weighting F is 1, and the threads do not matter
    sc2 = scene(doenv=False)
    assert (run(sc2)["e_f"][:, :, g] == 1.0).all()
    o4 = run(sc, n_jobs=4)
    for k in ("e_t", "e_rho_p", "a_tg", "a_dem_bin", "qai_elev"):
        np.testing.assert_array_equal(o[k], o4[k])


def test_aerosol_flag_of_the_qai():
    for aodmap in (True, False):
        o = run(scene(aodmap=aodmap))
        qai = o["qai_elev"]
        off = (qai & 1).astype(bool)
        aer = (qai >> 6) & 3
        assert (aer[off] == 0).all()
        if not aodmap:
            assert (aer[~off] == 3).all()
        else:
            assert 3 not in set(aer.tolist())
            assert (aer == 2).any() and (aer == 1).any() and (aer == 0).any()
            # high AOD pixels: the AOD of the green band at their elevation bin and cell is above 0.6
            top = scene()["top_dem"].reshape(NY, NX)
            cell = (np.arange(NY)[:, None] // 50) * GNX + (np.arange(NX)[None, :] // 50)
            green = o["e_aod"][:, 1, :].astype(np.float64)
            high = green[top, cell] > 0.6
            assert ((aer.reshape(NY, NX) == 2) == (high & ~off.reshape(NY, NX))).all()


def test_interpolation_weights_and_the_area_of_interest():
    sc = scene(npts=1, aoi=True)
    o = run(sc)
    w = o["weights"]
    assert len(w) == 5
    # the corner of the cells 0, 1, 4, 5 (pixel (50, 50) of 100 m cells of 5000 m): the weights are 1/2 each
    assert tuple(w["g"][0]) == (0, 1, 4, 5) and tuple(w["w"][0]) == (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    assert w["v"][0] == 0.5 * (0.5 * 0 + 0.5 * 10) + 0.5 * (0.5 * 40 + 0.5 * 50)
    # the first pixel is at the edge of the grid: the cell itself, twice
    assert tuple(w["g"][1]) == (0, 0, 0, 0) and w["v"][1] == 0
    # the last pixel: the cell 11 (row 2, column 3)
    assert tuple(w["g"][2]) == (11, 11, 11, 11) and w["v"][2] == 110
    # (75, 75): the cells 5, 6, 9, 10, whose weights favour the first; the cell 6 has no data (coarse[1, 2])
    assert tuple(w["g"][3]) == (5, 6, 9, 10) and tuple(w["w"][3][:2]) == (1.0, 0.0)
    # the area of interest: apply_aoi sets the pixels off where the AOI is 0
    assert o["aoi_ret"] == 0
    off = (o["qai_aoi"] & 1).astype(bool)
    aoi = sc["aoi"].astype(bool)
    assert (off[~aoi]).all() and (off[aoi] == (sc["qai"][aoi] & 1).astype(bool)).all()
    assert run(scene())["aoi_ret"] == 10  # no AOI: CANCEL, the QAI is as it came


# ---------------------------------------------------------------------------------------------------------------
# the surface reflectance and the Level 2 products (zeit_core::l2, FORCE's radiometric_correction); the comparison with
# FORCE is benchmarks/vs_originals/force_l2.py
# ---------------------------------------------------------------------------------------------------------------

L_DOMAIN = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
L_RSR = np.array([22, 23, 24, 25, 27, 28], np.int32)
L_FLAGS = ("domulti", "doenv", "doatmo", "dotopo", "dobrdf", "erase_cloud", "doaod", "odst", "ovzn", "ohot", "oaod", "owvp")
OFF, SUBZERO, SATURATION = 1, 1 << 8, 1 << 9


def l2_scene(**flags):
    """A Landsat-like scene of 300 x 250 pixels of 100 m (a tilted swath), 6 bands, a TOA reflectance of soil with a
    cloud, a cloud shadow and a few pixels that are negative or above 1."""
    rng = np.random.default_rng(11)
    wvl = rt("wavelength", L_RSR)[0]
    ii, jj = np.mgrid[0:NY, 0:NX]
    qai = np.zeros((NY, NX), np.int16)
    shift = (ii * 0.2126).astype(int)
    qai[~((jj >= shift) & (jj < shift + int(NX * 0.7)))] |= OFF
    qai[100:110, 120:130] |= 1 << 1  # opaque cloud
    qai[120:126, 150:156] |= 2 << 1  # unsure cloud
    qai[100:110, 135:140] |= 1 << 3  # shadow
    spectrum = np.array([0.06, 0.09, 0.11, 0.3, 0.25, 0.12])
    toa = np.round(10000 * spectrum[:, None, None] * (1 + 0.1 * rng.random((1, NY, NX)))).astype(np.int16)
    toa[:, 100:110, 120:130] = 5000
    toa[2, 20, 100] = -300  # a little below 0
    toa[3, 30, 100] = -12000  # reflectance below -1: the pixel is set off
    toa[1, 40, 100] = 15000  # above 1: saturated
    toa[:, qai & OFF != 0] = -9999
    sc = dict(mission=0, nx=NX, ny=NY, res=RES, ulx=500000.0, uly=48.0 / 9.0e-6, domain=L_DOMAIN, wavelength=wvl,
              rsr_band=L_RSR, year=2021, doy=172, hh=10, mm=30, ss=0, tz=0, lon0=10.0, x0=500000.0, kgeo=9.0e-6, wvp=F32(2.0),
              dem_nodata=-32767, dem=np.zeros(0, F32), gn=GN, qai=qai.ravel(), toa=toa.ravel(), s2nx=0, s2ny=0,
              s2nodata=-9999.0, vzen=np.zeros(0, F32), vazi=np.zeros(0, F32), aoi=np.zeros(0, np.uint8))
    sc.update({k: 0 for k in L_FLAGS})
    sc.update(flags)
    return sc


def l2_run(sc, n_jobs=1):
    """The dump of zeit._core.atmo.level2: (status, products as [(domains, data)])."""
    raw, _ = _core.atmo.level2(dict(sc, n_jobs=n_jobs))
    raw = bytes(raw)
    if np.frombuffer(raw, "<i4", 1)[0] != 0:
        return 1, []
    nb, npix = len(sc["domain"]), sc["nx"] * sc["ny"]
    # the status and the time, the scalars of the atc, the per band values
    pos = 4 + 8 + 4 * 4 + 4 + 4 * 4 + 4 + 4 * 4 + nb * 16

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    for n_bands in [nb, nb, 1, 1, 1, 1, nb, nb, nb, nb, nb, nb, 1, 1, 1, 8, 8, 1]:  # the 18 coarse bricks
        assert int(take("<i4", 1)[0]) == n_bands
        take("<f4", n_bands * sc["gn"])
    for n_bands in [nb] * 13 + [1, 1]:  # the 15 families of 256 bricks
        assert int(take("<i4", 1)[0]) == n_bands
        take("<f4", 256 * n_bands * sc["gn"])
    prods = []
    for _ in range(int(take("<i4", 1)[0])):
        n = int(take("<i4", 1)[0])
        dom = []
        for _ in range(n):
            dom.append(bytes(take("u1", 16)).rstrip(b"\0").decode())
            take("<f4", 1)
        prods.append((dom, take("<i2", n * npix).reshape(n, npix)))
    take("<i4", 1)  # the gaseous transmittance maps (Sentinel-2)
    assert pos == len(raw)
    return 0, prods


def test_level2_without_atmosphere_gives_the_toa_and_the_flags():
    sc = l2_scene(odst=1, ovzn=1, ohot=1, oaod=1, owvp=1)
    status, prods = l2_run(sc)
    assert status == 0 and len(prods) == 7
    (_, qai), (dom, boa), (_, dst), (_, vzn), (_, hot), (_, aod), (_, wvp) = prods
    assert dom == L_DOMAIN
    qai = qai[0]
    toa = sc["toa"].reshape(6, -1)
    q0 = sc["qai"]
    off0 = (q0 & OFF) != 0
    p_neg, p_big, p_off = 20 * NX + 100, 40 * NX + 100, 30 * NX + 100
    # a reflectance below -1 sets the pixel off (and the QAI of the off pixels is 1, the BOA the no-data value)
    off = qai == 1
    assert (qai[off0] == 1).all() and off[p_off] and not off0[p_off] and off.sum() == off0.sum() + 1
    assert (boa[:, off] == -9999).all()
    # the TOA reflectance is kept (to the truncation of the reflectance times 10000), the flags come from the value
    ok = ~off & (toa[1] < 32000)
    assert (np.abs(boa[:, ok].astype(int) - toa[:, ok]) <= 1).all()
    assert abs(int(boa[2, p_neg]) + 300) <= 1 and (qai[p_neg] & SUBZERO) and not (qai[p_neg] & SATURATION)
    assert boa[1, p_big] == 15000 and (qai[p_big] & SATURATION)
    # without the atmospheric correction there is no AOD and no water vapour; HOT is blue - red / 2 - 800
    assert (aod == -9999).all() and (wvp == -9999).all()
    h = toa[0] - 0.5 * toa[2] - 800
    live = ~off0 & (np.abs(h) < 30000)
    assert (hot[0, live] == np.trunc(h[live]).astype(np.int16)).all() and (hot[0, off0] == -9999).all()
    # the distance to the next cloud, shadow or snow pixel in metres: 0 on them, 100 next to them, no data where off
    cloud = ((q0 >> 1) & 3 > 0) | ((q0 >> 3) & 1 == 1)
    assert (dst[0, cloud & ~off] == 0).all() and (dst[0, off] == -9999).all()
    assert dst[0, 99 * NX + 125] == 100 and dst[0, 98 * NX + 125] == 200
    # the view zenith: no no-data pixel is left on the pixels that are not off
    assert (vzn[0, ~off] >= 0).all() and (vzn[0, ~off] < 1000).all() and (vzn[0, off] == -9999).all()


def test_level2_erase_cloud_and_the_missing_bands():
    sc = l2_scene(erase_cloud=1)
    status, prods = l2_run(sc)
    qai, boa = prods[0][1][0], prods[1][1]
    p_unsure, p_cloud = 122 * NX + 152, 105 * NX + 125
    assert (boa[:, p_unsure] == -9999).all() and qai[p_unsure] & OFF == 0  # erased, but not off
    assert (boa[:, p_cloud] != -9999).all()  # only the unsure clouds (the flag 2) are erased
    # a sensor without the NIR band cannot give the BOA (FORCE returns NULL)
    sc = l2_scene()
    keep = [0, 1, 2, 4, 5]
    sc.update(domain=[L_DOMAIN[b] for b in keep], wavelength=sc["wavelength"][keep], rsr_band=L_RSR[keep],
              toa=sc["toa"].reshape(6, -1)[keep].ravel())
    assert l2_run(sc)[0] == 1


def test_level2_atmospheric_correction_and_its_products():
    sc = l2_scene(doatmo=1, doenv=1, domulti=1, doaod=1, odst=1, ovzn=1, ohot=1, oaod=1, owvp=1)
    status, prods = l2_run(sc)
    assert status == 0 and len(prods) == 7
    (_, qai), (dom, boa), (_, dst), (_, vzn), (_, hot), (_, aod), (_, wvp) = prods
    qai = qai[0]
    off = qai == 1
    live = ~off
    assert dom == L_DOMAIN and live.sum() > 20000
    # the correction removes the path reflectance: the BOA of soil is below its TOA in the blue
    toa = sc["toa"].reshape(6, -1)
    assert np.median(boa[0, live] - toa[0, live]) < 0
    # (the AOD is made before the BOA, which sets one more pixel off: it is the no-data value where the input is off; the
    # Landsat WVP comes after the BOA)
    off_in = (sc["qai"] & OFF) != 0
    assert (boa[:, off] == -9999).all() and (aod[0, off_in] == -9999).all() and (wvp[0, off] == -9999).all()
    assert (aod[0, ~off_in] > 0).all() and (wvp[0, live] == 2000).all()  # the constant of the parameter file: 2 cm
    # the threads do not matter
    status3, prods3 = l2_run(sc, n_jobs=3)
    assert status3 == 0 and all(np.array_equal(a[1], b[1]) for a, b in zip(prods, prods3))
    # the topographic correction with a flat DEM, and without the environmental weighting
    status, prods_t = l2_run(dict(sc, dem=np.full(NX * NY, 300.0, F32), dotopo=1))
    assert status == 0 and np.median(np.abs(prods_t[1][1][3, live].astype(int) - boa[3, live])) < 800
    boa_e = l2_run(dict(sc, doenv=0))[1][1][1]
    assert 0 < np.abs(boa_e[:, live].astype(int) - boa[:, live]).max() < 12000
