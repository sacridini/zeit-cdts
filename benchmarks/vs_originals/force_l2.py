"""The surface reflectance and the Level 2 products of zeit-core (``zeit_core::l2``, a port of FORCE 3.11.00's
``atmo-ll.c`` from ``surface_reflectance`` on: ``background_reflectance``, ``compile_l2_boa``, ``compile_l2_qai``,
``compile_l2_dst`` (with ``cloud_distance`` of ``cloud-ll.c``), ``compile_l2_vzn``, ``compile_l2_hot``,
``compile_l2_aod``, ``compile_l2_wvp``, ``compile_level2`` and ``radiometric_correction``) against FORCE's own
``radiometric_correction`` on synthetic scenes.

The C side is ``force_l2/`` (``make_harness.sh``, WSL): a ``main.c`` compiled against FORCE's headers and linked with the
object files FORCE's ``make`` built for ``force-l2ps`` (atmo-ll, aod-ll, gas-ll, topo-ll, atc-ll, sunview-ll, radtran-ll,
brick_base-cl, quality-cl, ...), so that ``radiometric_correction`` runs as the L2PS runs it: ``allocate_atc``,
``sun_target_view``, ``compile_topography``, then ``atmo_angledep``, ``compile_aod``, ``atmo_elevdep``, ``water_vapor``,
``cfactor_topography``, ``apply_aoi`` and ``compile_level2``, against the real GSL 2.7 (static, with its reference
CBLAS) and the real GDAL 3.9.3 of the env ``force-ref`` (``GDAL_NUM_THREADS=1``), run with ``OMP_NUM_THREADS=1`` (the
scene-average environment reflectance of the AOD estimation is a sum per thread, see ``zeit_core::aod``). Not FORCE's, and
the same on both sides: the sun position (``sun-cl.c`` is not opened: the harness holds a table of the values Zeit's clean
room ``sunpos`` gives for every coarse cell), the map to longitude and latitude (an affine map), the warp of the DEM (the
DEM the harness holds) and ``brdf_factor`` (``brdf-ll.c`` is not ported: a factor of the view zenith and the band, in
float). The cloud flags of the QAI are given (FORCE's cloud detection is not ported). Zeit's side is
``zeit._core.atmo.level2`` (Windows, UCRT libm; the harness runs on glibc 2.39)::

    python benchmarks/vs_originals/force_l2.py [--scale 1.0] [--skip-build]

Scenes: Landsat and Sentinel-2 TOA reflectance from a forward model of the radiative transfer for a land background (three
soil spectra by region), lakes (the spectra of FORCE's water library), dark dense vegetation (its vegetation library),
cloud (opaque, unsure, cirrus), cloud shadow, snow and no-data regions, a DEM with relief (with a no-data patch, the
oceans of ``ocean_topography``); with ``doatmo``, ``dotopo``, ``doenv``, ``domulti``, ``dobrdf``, ``erase_cloud``,
``doaod`` on and off, with and without an area of interest, and all the optional products. Prints, per scene and field,
the number of values compared, how many are equal bit for bit, 1 ulp apart (float fields) and more apart: the atc after
the correction (the scalars, the water vapour, the AOD, the 18 coarse bricks, 15 families of 256 elevation bricks), the
QAI (final), the BOA of each band, and the DST, VZN, HOT, AOD and WVP. Exits with 1 if anything differs by more than 1
float32 ulp (the integers and the products exactly).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import no_power_throttling, workdir
from force_atmo import F32, LANDSAT, SENTINEL2, Table, rt, smooth_field, _wsl

HERE = Path(__file__).resolve().parent
DATA = workdir("force_l2")
EXE = "$HOME/force-ref-work/l2/l2_harness"

XY = ["mod", "aod", "Pr", "Pa", "Hr", "Ha", "Tvw", "Tsw", "Tvo", "Tso", "Tg", "brdf", "fresnel", "dem", "interp", "view",
      "sun", "psi"]
XYZ = ["od", "mod", "aod", "rho_p", "Ts", "tsd", "tss", "Tv", "tvd", "tvs", "T", "s", "F", "Hr", "Ha"]
FLAGS = ["domulti", "doenv", "doatmo", "dotopo", "dobrdf", "erase_cloud", "doaod", "odst", "ovzn", "ohot", "oaod", "owvp"]
DEFAULT_FLAGS = dict(domulti=1, doenv=1, doatmo=1, dotopo=1, dobrdf=0, erase_cloud=0, doaod=1, odst=1, ovzn=1, ohot=1,
                     oaod=1, owvp=1)

# reflectance spectra: wavelength (um) -> reflectance, linearly interpolated
SOIL = [(0.44, 0.07), (0.5, 0.09), (0.56, 0.11), (0.66, 0.15), (0.75, 0.2), (0.86, 0.26), (0.94, 0.27), (1.38, 0.3),
        (1.61, 0.34), (2.2, 0.3)]
VEG = [(0.44, 0.02), (0.49, 0.03), (0.56, 0.07), (0.66, 0.025), (0.705, 0.1), (0.74, 0.3), (0.78, 0.4), (0.86, 0.42),
       (0.94, 0.4), (1.38, 0.15), (1.61, 0.2), (2.2, 0.08)]
SNOW = [(0.44, 0.9), (0.56, 0.92), (0.66, 0.88), (0.86, 0.7), (1.61, 0.1), (2.2, 0.05)]
CLOUD = [(0.44, 0.55), (0.66, 0.55), (0.86, 0.55), (1.38, 0.45), (1.61, 0.4), (2.2, 0.3)]


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_l2' / 'make_harness.sh')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness build failed: {r.stdout}{r.stderr}")


def spec(table, wvl):
    t = np.array(table)
    return np.interp(wvl, t[:, 0], t[:, 1])


def ellipse(ny, nx, ci, cj, ri, rj, rot=0.0):
    ii, jj = np.mgrid[0:ny, 0:nx]
    di, dj = ii - ci, jj - cj
    c, s = np.cos(rot), np.sin(rot)
    u, v = c * di + s * dj, -s * di + c * dj
    return (u / ri) ** 2 + (v / rj) ** 2 <= 1.0


def make_scene(rng, mission, nx, ny, res, lat0=48.0, lon0=10.0, doy=172, hh=10, mm=30, aoi=False, nodata_cells=(),
               qai_kind=None, relief=1200.0, dem=True, aod550=0.12, wvp=2.0, lakes=True, **flags):
    """The inputs of one scene as the binding and the harness take them (``flags`` override DEFAULT_FLAGS)."""
    from zeit import _core

    fl = dict(DEFAULT_FLAGS)
    fl.update(flags)
    if not fl["doatmo"]:  # as parse_param_lower: without the atmospheric correction these are off
        fl.update(doaod=0, dobrdf=0, dotopo=0, doenv=0)
    if not dem:
        fl["dotopo"] = 0  # (FORCE needs a DEM file for DO_TOPO)
    domain, rsr = LANDSAT if mission == 0 else SENTINEL2
    rsr = np.array(rsr, np.int32)
    nb = len(domain)
    wvl = rt("wavelength", rsr)[0]
    npix = nx * ny
    kgeo = 9.0e-6
    x0, ulx = 500000.0, 500000.0 + float(rng.integers(-20, 20)) * 1000.0
    uly = lat0 / kgeo
    gnx, gny = int(np.ceil(nx * res / 5000.0)), int(np.ceil(ny * res / 5000.0))
    gn = gnx * gny
    year, tz, ss = 2021, 0, 0
    qai_kind = qai_kind or ("swath" if mission == 0 else "frame")
    ii, jj = np.mgrid[0:ny, 0:nx]
    m2p = 1.0 / res  # metres to pixels

    # the sun of the coarse cells, as zeit's sunpos gives it, keyed by the latitude and longitude of the cell
    e, f = np.mgrid[0:gny, 0:gnx]
    x = ulx + f.ravel() * 5000.0
    y = uly + e.ravel() * -5000.0
    lon = lon0 + (x - x0) * kgeo
    lat = y * kgeo
    n = lon.size
    zen, azi = rt("sunpos", lat, lon, np.full(n, year), np.full(n, doy), np.full(n, hh), np.full(n, mm), np.full(n, ss),
                  np.full(n, tz))
    # the radiative transfer of a uniform atmosphere: the TOA reflectance of each spectrum
    ms = np.full(1, np.cos(np.radians(zen.mean())), F32)
    mv = np.full(1, 0.9995, F32)
    sob, pa, pr, fres = np.full(1, 0.84, F32), np.full(1, 0.7, F32), np.full(1, 1.0, F32), np.full(1, 0.03, F32)
    water_lib, veg_lib = _core.aod.libraries(list(map(int, rsr)), 1, 2, 3)
    water_lib, veg_lib = water_lib.reshape(-1, nb), veg_lib.reshape(-1, nb)

    def forward(rho, water=False):
        """rho: (n, nb) surface reflectance -> (n, nb) TOA reflectance."""
        out = np.zeros_like(rho, dtype=np.float64)
        for b in range(nb):
            aod = np.full(1, aod550 * (wvl[b] / 0.55) ** -1.3, F32)
            mod = rt("molecular_optical_depth", wvl[b:b + 1])[0]
            od = rt("optical_depth", aod, mod)[0]
            T, ts, tv, tsd, tss, tvd, tvs = rt("scatt_transmitt", aod, mod, od, ms, mv)
            rhop = rt("path_ref", np.ones(1, F32), sob, aod, mod, od, pa, pr, tsd, tvd, ms, mv)[0]
            s = rt("sphere_albedo", aod, mod, od)[0]
            tg = 0.93
            r = (rho[:, b] + fres * tss) / ts if water else rho[:, b]
            out[:, b] = tg * (rhop + ts * (tvd * r + tvs * r) / (1 - r * s))
        return out

    soil = np.stack([spec(SOIL, wvl) * k for k in (0.8, 1.0, 1.25)])
    t_soil = forward(soil)
    t_water = forward(water_lib, True)
    t_veg = forward(veg_lib)
    # the targets the AOD estimation accepts: TOA reflectance that falls from band to band over the visible bands and the
    # SWIR2 (the water ones), by more than the noise
    aodb = [b for b in range(nb) if wvl[b] < 0.7 or domain[b] == "SWIR2"]
    sw2 = domain.index("SWIR2")

    def falling(t, skip_sw2):
        ok = np.ones(len(t), bool)
        use = [b for b in aodb if not (skip_sw2 and b == sw2)]
        for k in range(1, len(use)):
            ok &= (t[:, use[k]] < t[:, use[k - 1]] - 0.004)
        return ok

    good_w = falling(t_water, False) & (t_water[:, sw2] < 0.02)
    good_v = falling(t_veg, True) & (t_veg[:, sw2] < 0.045)
    if good_w.any():
        water_lib, t_water = water_lib[good_w], t_water[good_w]
    if good_v.any():
        veg_lib, t_veg = veg_lib[good_v], t_veg[good_v]
    t_snow = forward(spec(SNOW, wvl)[None])[0]
    t_cloud = forward(spec(CLOUD, wvl)[None])[0]

    # the landscape
    cls = np.zeros((ny, nx), np.int32)  # 0 soil, 1.. water spectrum + 1, 1000.. vegetation spectrum + 1000
    region = np.digitize(smooth_field(rng, ny, nx, 300.0 * m2p), [-0.25, 0.25])
    if lakes:
        for k in range(9):
            ci, cj = rng.uniform(0.1 * ny, 0.9 * ny), rng.uniform(0.1 * nx, 0.9 * nx)
            ri, rj = rng.uniform(500, 1500) * m2p, rng.uniform(500, 1500) * m2p
            m = ellipse(ny, nx, ci, cj, ri, rj, rng.uniform(0, np.pi))
            cls[m] = 1 + int(rng.integers(0, len(water_lib)))
            cv = ellipse(ny, nx, ci + rng.uniform(-2000, 2000) * m2p, cj + rng.uniform(-2000, 2000) * m2p,
                         rng.uniform(500, 1500) * m2p, rng.uniform(500, 1500) * m2p, rng.uniform(0, np.pi))
            cv &= cls == 0
            cls[cv] = 1000 + int(rng.integers(0, len(veg_lib)))
    wmask, vmask = (cls >= 1) & (cls < 1000), cls >= 1000
    toa = np.zeros((nb, ny, nx))
    for b in range(nb):
        toa[b] = t_soil[region, b]
        toa[b][wmask] = t_water[cls[wmask] - 1, b]
        toa[b][vmask] = t_veg[cls[vmask] - 1000, b]

    # QAI: no data, water, cloud (opaque, unsure, cirrus), cloud shadow, snow
    qai = np.zeros((ny, nx), np.int16)
    if qai_kind == "swath":  # a parallelogram that leans, as a Landsat scene does
        shift = (ii * 0.2126).astype(int)
        qai[~((jj >= shift) & (jj < shift + int(nx * 0.7)))] |= 1
    elif qai_kind == "frame":
        qai[:3, :] |= 1
        qai[:, -4:] |= 1
        qai[ny // 3:ny // 3 + 5, nx // 4:nx // 4 + 9] |= 1
    qai[wmask] |= 1 << 5
    for k in range(5):
        c = ellipse(ny, nx, rng.uniform(0.15 * ny, 0.85 * ny), rng.uniform(0.15 * nx, 0.85 * nx), rng.uniform(300, 900) * m2p,
                    rng.uniform(300, 900) * m2p, rng.uniform(0, np.pi))
        val = [1, 1, 2, 3, 1][k]
        halo = ellipse(ny, nx, *np.argwhere(c).mean(0), 1.3 * rng.uniform(300, 900) * m2p, 1.3 * rng.uniform(300, 900) * m2p) \
            if c.any() else c
        qai[halo & (qai & 6 == 0)] |= 2 << 1
        qai[c] = (qai[c] & ~6) | (val << 1)
        shd = np.roll(np.roll(c, int(400 * m2p), 0), int(300 * m2p), 1) & (qai & 6 == 0)
        qai[shd] |= 1 << 3
        if val == 1:
            for b in range(nb):
                toa[b][c] = t_cloud[b] * rng.uniform(0.9, 1.1)
        if val == 3:
            toa[:, c] *= 1.1
        toa[:, shd] *= 0.3
    sn = ellipse(ny, nx, ny * 0.8, nx * 0.2, 700 * m2p, 900 * m2p) & (qai & 0b1110 == 0)
    qai[sn] |= 1 << 4
    for b in range(nb):
        toa[b][sn] = t_snow[b]
    if "VAPOR" in domain:  # the absorption of the water vapour band, for a smooth field of 0.6 to 3.5 cm
        wf = (0.6 + 2.9 * (0.5 + 0.5 * smooth_field(rng, ny, nx, 4000.0 * m2p))).astype(F32)
        grid_w = np.linspace(0.5, 3.6, 64).astype(F32)
        vb = domain.index("VAPOR")
        tw = rt("wvp_transmitt", grid_w, np.full(64, ms[0], F32), np.full(64, rsr[vb], F32))[0] *             rt("wvp_transmitt", grid_w, np.full(64, mv[0], F32), np.full(64, rsr[vb], F32))[0]
        toa[vb] *= np.interp(wf, grid_w, tw)
    toa *= (1 + 0.01 * rng.standard_normal((1, ny, nx)))
    toa += 0.0003 * rng.standard_normal(toa.shape)
    toa = np.clip(np.round(10000 * toa), -9998, 32000).astype(np.int16)
    toa[:, qai & 1 != 0] = -9999

    # the DEM
    if dem:
        z = relief * (0.5 + 0.5 * smooth_field(rng, ny, nx, 9000.0 * m2p)) + 0.06 * relief * smooth_field(rng, ny, nx, 1800.0 * m2p)
        z = z.astype(F32)
        for lk in range(1, int(cls.max()) + 1):
            m = cls == lk
            if m.any() and lk < 1000:
                z[m] = z[m].mean()  # a lake is flat
        z[: max(ny // 10, 2), : max(nx // 10, 2)] = -32767  # an ocean (no data in the DEM)
        dem_arr = z
    else:
        dem_arr = np.zeros(0, F32)

    sc = dict(mission=mission, nx=nx, ny=ny, res=float(res), ulx=ulx, uly=uly, domain=domain, wavelength=wvl,
              rsr_band=rsr, year=year, doy=doy, hh=hh, mm=mm, ss=ss, tz=tz, lon0=lon0, x0=x0, kgeo=kgeo, wvp=F32(wvp),
              dem_nodata=-32767, dem=dem_arr.ravel(), gn=gn, qai=qai.ravel(), toa=toa.ravel(), s2nx=0, s2ny=0,
              s2nodata=-9999.0, vzen=np.zeros(0, F32), vazi=np.zeros(0, F32), gnx=gnx, gny=gny, sun=(lat, lon, zen, azi),
              aoi=(rng.random(npix) < 0.85).astype(np.uint8) if aoi else np.zeros(0, np.uint8), **fl)
    if mission == 1:
        gj, gi = np.meshgrid(np.arange(gnx), np.arange(gny))
        vzen = (2.0 + 9.0 * (gj / max(gnx - 1, 1)) + rng.uniform(0, 1.5, gj.shape)).astype(F32)
        vazi = (95.0 + 40.0 * (gi / max(gny - 1, 1)) + rng.uniform(0, 3.0, gj.shape) + 100 * (gj % 2)).astype(F32)
        for (e_, f_) in nodata_cells:
            if e_ < gny and f_ < gnx:
                vzen[e_, f_] = -9999.0
        sc.update(s2nx=gnx, s2ny=gny, vzen=vzen.ravel(), vazi=vazi.ravel())
    return sc


def write_scene(path, sc):
    with open(path, "wb") as f:
        a = lambda x, dt: f.write(np.ascontiguousarray(x, dt).tobytes())  # noqa: E731
        a([sc["mission"], sc["nx"], sc["ny"]], np.int32)
        a([sc["res"], sc["ulx"], sc["uly"]], np.float64)
        a([len(sc["domain"])], np.int32)
        for b, d in enumerate(sc["domain"]):
            f.write(d.encode().ljust(16, b"\0"))
            a([sc["wavelength"][b]], np.float32)
            a([sc["rsr_band"][b]], np.int32)
        a([sc[k] for k in ("year", "doy", "hh", "mm", "ss", "tz")], np.int32)
        a([sc["lon0"], sc["x0"], sc["kgeo"]], np.float64)
        a([sc[k] for k in FLAGS], np.int32)
        a([sc["wvp"]], F32)
        a([sc["dem_nodata"], int(len(sc["dem"]) > 0)], np.int32)
        if len(sc["dem"]):
            a(sc["dem"], F32)
        a([sc["s2nx"], sc["s2ny"]], np.int32)
        a([sc["s2nodata"]], F32)
        if sc["mission"] == 1:
            a(sc["vzen"], F32)
            a(sc["vazi"], F32)
        a([sc["gn"]], np.int32)
        a(sc["qai"], np.int16)
        a(sc["toa"], np.int16)
        lat, lon, zen, azi = sc["sun"]
        a([len(lat)], np.int32)
        rec = np.zeros(len(lat), np.dtype([("lat", "<f8"), ("lon", "<f8"), ("zen", "<f4"), ("azi", "<f4")]))
        rec["lat"], rec["lon"], rec["zen"], rec["azi"] = lat, lon, zen, azi
        f.write(rec.tobytes())
        a([len(sc["aoi"]) > 0], np.int32)
        if len(sc["aoi"]):
            a(sc["aoi"], np.uint8)


def parse(raw, sc):
    """The dump of the C harness or of the binding (see force_l2/main.c)."""
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    nb, gn, npix = len(sc["domain"]), sc["gn"], sc["nx"] * sc["ny"]
    o = {"status": int(take("<i4", 1)[0]), "ms": float(take("<f8", 1)[0])}
    if o["status"] != 0:
        return o
    o["wvp_h"] = take("<f4", 4)
    o["aodmap"] = int(take("<i4", 1)[0])
    o["dem"] = take("<f4", 4)
    o["cnum"] = int(take("<i4", 1)[0])
    o["cos"] = take("<f4", 4)
    per = []
    for _ in range(nb):
        per.append(np.concatenate([take("<f4", 3).astype(np.float64), take("<i4", 1).astype(np.float64)]))
    o["band"] = np.array(per)
    for k in XY:
        n = int(take("<i4", 1)[0])
        o["xy_" + k] = take("<f4", n * gn).reshape(n, gn)
    for k in XYZ:
        n = int(take("<i4", 1)[0])
        o["xyz_" + k] = take("<f4", 256 * n * gn).reshape(256, n, gn)
    nprod = int(take("<i4", 1)[0])
    prods = []
    for _ in range(nprod):
        n = int(take("<i4", 1)[0])
        dom, wl = [], []
        for _ in range(n):
            dom.append(bytes(take("u1", 16)).rstrip(b"\0").decode())
            wl.append(float(take("<f4", 1)[0]))
        prods.append((dom, np.array(wl, F32), take("<i2", n * npix).reshape(n, npix)))
    o["products"] = prods
    ntg = int(take("<i4", 1)[0])
    o["tg"] = take("<i2", ntg * npix).reshape(ntg, npix)
    assert pos == len(raw), (pos, len(raw))
    return o


def run_force(name, sc):
    fin, fout = DATA / f"{name}.in.bin", DATA / f"{name}.out.bin"
    write_scene(fin, sc)
    cmd = f"OMP_NUM_THREADS=1 GDAL_NUM_THREADS=1 {EXE} scene {_wsl(fin)} {_wsl(fout)}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed: {r.stdout}{r.stderr}")
    return parse(fout.read_bytes(), sc)


def run_zeit(sc, n_jobs=1):
    from zeit import _core

    d = {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in sc.items() if k != "sun"}
    d["n_jobs"] = n_jobs
    raw, _ = _core.atmo.level2(d)
    return parse(bytes(raw), sc)


def compare(table, title, sc, c, z):
    table.first = True
    T = lambda field, a, b, note="": table.floats(title, field, a, b, note)  # noqa: E731
    E = lambda field, a, b, note="": table.exact(title, field, a, b, note)  # noqa: E731
    E("status", c["status"], z["status"])
    if c["status"] != 0 or z["status"] != 0:
        return
    E("aodmap, cnum", [c["aodmap"], c["cnum"]], [z["aodmap"], z["cnum"]], f" (AOD map: {bool(c['aodmap'])})")
    T("wvp, Hr, Ha, Hp, dem, cos", np.concatenate([c["wvp_h"], c["dem"], c["cos"]]), np.concatenate([z["wvp_h"], z["dem"], z["cos"]]))
    T("per band mod, aod, od", c["band"], z["band"])
    for k in XY:
        T("atc xy_" + k, c["xy_" + k], z["xy_" + k])
    for k in XYZ:
        T("atc xyz_" + k, c["xyz_" + k], z["xyz_" + k])
    pc, pz = c["products"], z["products"]
    E("products", len(pc), len(pz), f" ({len(pc)})")
    if len(c["tg"]) or len(z["tg"]):
        E("gas transmittance maps", c["tg"], z["tg"], f" ({len(c['tg'])} bands)")
    names = ["QAI", "BOA", "DST", "VZN", "HOT", "AOD", "WVP"]
    names = names[:2] + [n for n, k in zip(names[2:], ["odst", "ovzn", "ohot", "oaod", "owvp"]) if sc[k]]
    for name, (dc, wc, ac), (dz, wz, az) in zip(names, pc, pz):
        qa = pc[0][2][0]
        note = ""
        if name == "QAI":
            q = ac[0]
            note = (f" (off {int((q & 1).sum())}, cloud {int(((q >> 1) & 3 > 0).sum())}, subzero {int(((q >> 8) & 1).sum())}, "
                    f"saturated {int(((q >> 9) & 1).sum())}, aerosol 1/2/3: "
                    f"{[int((((q >> 6) & 3) == v).sum()) for v in (1, 2, 3)]})")
        if name == "BOA":
            valid = int((ac != -9999).sum())
            note = f" ({len(dc)} bands: {','.join(dc)}; {valid} values not no-data)"
        E(name + " domains", dc, dz)
        E(name + " wavelengths", wc, wz)
        E(name + " values", ac, az, note)


def scene_list(scale):
    """The scenes: a title and the arguments of make_scene."""
    k = np.sqrt(scale)

    def m6(v):  # Sentinel-2 images: a multiple of 6 pixels (FORCE's 60 m blocks run past the edge of any other size)
        return int(v) // 6 * 6

    return [
        ("Landsat 30 m 800x700, all on", dict(mission=0, nx=int(800 * k), ny=int(700 * k), res=30.0)),
        ("Landsat 100 m 700x600, no env, brdf, erase cloud, AOI",
         dict(mission=0, nx=int(700 * k), ny=int(600 * k), res=100.0, doenv=0, dobrdf=1, erase_cloud=1, aoi=True)),
        ("Landsat 30 m, no topo, single scattering", dict(mission=0, nx=int(700 * k), ny=int(650 * k), res=30.0, dotopo=0,
                                                         domulti=0, lat0=35.0, doy=250)),
        ("Landsat 30 m, no AOD estimation (fallback)", dict(mission=0, nx=int(700 * k), ny=int(600 * k), res=30.0, doaod=0,
                                                           dobrdf=1, wvp=3.5, lon0=-60.0, lat0=-20.0, doy=30, hh=14)),
        ("Landsat 30 m, hazy (AOD 0.5 at 550 nm)", dict(mission=0, nx=int(700 * k), ny=int(600 * k), res=30.0, aod550=0.5,
                                                       lat0=60.0, lon0=25.0, doy=200)),
        ("Landsat 100 m, no atmospheric correction (TOA)", dict(mission=0, nx=int(600 * k), ny=int(600 * k), res=100.0,
                                                               doatmo=0)),
        ("Landsat 30 m, Dec 70 N: low sun, flat DEM", dict(mission=0, nx=int(700 * k), ny=int(600 * k), res=30.0, lat0=70.0,
                                                          doy=355, hh=11, mm=0, dem=False)),
        ("Sentinel-2 20 m 1000x900, all on, no-data cells",
         dict(mission=1, nx=m6(1000 * k), ny=m6(900 * k), res=20.0, nodata_cells=[(1, 1), (3, 4), (0, 5)], aoi=True)),
        ("Sentinel-2 10 m 800x700, no env, brdf", dict(mission=1, nx=m6(800 * k), ny=m6(700 * k), res=10.0, doenv=0,
                                                       dobrdf=1, lat0=-33.0, lon0=150.0, doy=20, hh=0, mm=15)),
        ("Sentinel-2 60 m 900x300 (cols > rows), no topo", dict(mission=1, nx=m6(900 * k), ny=m6(300 * k), res=60.0,
                                                                dotopo=0, nodata_cells=[(0, 3), (2, 7)])),
        ("Sentinel-2 60 m 300x900 (rows > cols), single scattering, erase cloud",
         dict(mission=1, nx=m6(300 * k), ny=m6(900 * k), res=60.0, domulti=0, erase_cloud=1, nodata_cells=[(7, 1)])),
        ("Sentinel-2 20 m, no atmospheric correction (TOA)", dict(mission=1, nx=m6(600 * k), ny=m6(500 * k), res=20.0,
                                                                 doatmo=0, doaod=0)),
        ("Sentinel-2 60 m 1200x1200 tile, no optional products",
         dict(mission=1, nx=m6(1200 * k), ny=m6(1200 * k), res=60.0, odst=0, ovzn=0, ohot=0, oaod=0, owvp=0,
              nodata_cells=[(5, 5), (0, 0)])),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the area of the large scenes")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", help="run only these scenes (all are generated, as the draws are in sequence)")
    args = ap.parse_args()
    no_power_throttling()
    rng = np.random.default_rng(2310)
    if not args.skip_build:
        build_harness()
    scenes = scene_list(args.scale)
    print(f"{'scene':44s} {'field':16s} {'values':>9s} {'equal':>9s} {'1 ulp':>6s} {'more':>6s} {'max':>5s}")
    table = Table()
    for n_, (title, kw) in enumerate(scenes):
        sc = make_scene(rng, **kw)
        if args.only and n_ not in args.only:
            continue
        c = run_force(f"scene{n_}", sc)
        z = run_zeit(sc)
        compare(table, title, sc, c, z)
        print(f"{'':44s} FORCE {c['ms']:.0f} ms, Zeit {z['ms']:.0f} ms (radiometric_correction, 1 thread)")
        if n_ == 0:
            z4 = run_zeit(sc, n_jobs=4)
            same = all(np.array_equal(a[2], b[2]) for a, b in zip(z["products"], z4["products"]))
            print(f"{'':44s} Zeit with 4 threads vs 1: every product equal: {same}")
            table.bad |= not same
    print(f"{'total':44s} {'':16s} {table.tv:9d} {table.te:9d} {table.t1:6d} {table.tv - table.te - table.t1:6d}")
    sys.exit(1 if table.bad else 0)


if __name__ == "__main__":
    main()
