"""The atmosphere of the coarse grid of zeit-core (``zeit_core::atc``, ``sunview``, ``atmo``: ports of FORCE 3.11.00's
``atc-ll.c`` (``allocate_atc`` with ``init_atc``, ``init_atc_xy``, ``init_atc_xyz``), ``sunview-ll.c`` (``sun_target_view``,
``viewgeo``, ``view_angle``) and, from ``atmo-ll.c``, ``interpolation_weights``, ``interpolate_coarse``, ``atmo_angledep``,
``atmo_elevdep`` and ``apply_aoi``) against FORCE's own functions on synthetic scenes.

The C side is ``force_atmo/`` (``make_harness.sh``, WSL): FORCE's atc-ll.c and sunview-ll.c compiled as they are (whole,
but for their first include, behind shims for the brick, meta and parameter structures), the functions above of atmo-ll.c
copied verbatim, radtran-ll.c, table-ll.c and date-cl.c as they are; run with ``OMP_NUM_THREADS=1`` (the loops write only
their own cell, so the number of threads does not change the result; Zeit's loops are rayon loops that do not either).
Not FORCE's, and the same on both sides: the sun position (``sun-cl.c`` is not opened: the harness holds a table of the
values Zeit's ``sunpos`` gives for every coarse cell, looked up by the latitude and longitude), the warp of the map
coordinates to longitude and latitude (an affine map) and ``brdf_factor`` (``brdf-ll.c`` is not ported: 1 for every band,
which is what it starts from). Zeit's side is ``zeit._core.atmo`` (Windows, UCRT libm; the harness runs on glibc 2.39:
``exp``, ``log``, ``sin``, ``cos``, ``tan``, ``acos``, ``atan2``, ``pow``)::

    python benchmarks/vs_originals/force_atmo.py [--scale 1.0] [--skip-build]

Scenes: Landsat (the view geometry from the nadir line fitted to the swath of the QAI: a tilted parallelogram) at 30 and
100 m, in the north and south, east and west (the ozone model), in summer and with the sun below the horizon (the low sun
flag); Sentinel-2 (the view angle grids of the metadata with cells without data, which set the pixels off) at 10, 20 and 60
m, up to a whole tile (22 x 22 cells, 13 bands); grids with more columns than rows and more rows than columns (FORCE's loops
name them the other way round, see ``zeit_core::atmo``); with and without an AOD map, with and without the environmental
weighting and the multiple scattering. The interpolation weights on coarse grids with cells without data, at the pixels of
every position relative to the cell centres, the edges and the corners; ``apply_aoi`` with and without an area of interest.
Prints, per scene and field, the number of values compared, how many are equal bit for bit, 1 ulp apart (the libm of the
two sides) and more apart. Exits with 1 if anything differs by more than 1 float32 ulp (the integers and the flags
exactly).
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_atmo")
EXE = "$HOME/force-ref-work/atmo/atmo_harness"
F32 = np.float32
ND = -9999.0

XY = ["mod", "aod", "Pr", "Pa", "Hr", "Ha", "Tvw", "Tsw", "Tvo", "Tso", "Tg", "brdf", "fresnel", "dem", "interp", "view",
      "sun", "psi"]
XYZ = ["od", "mod", "aod", "Hr", "Ha", "rho_p", "Ts", "tsd", "tss", "Tv", "tvd", "tvs", "T", "s", "F", "Tg"]
ANGLE = ["brdf", "psi", "Pr", "Pa", "Tsw", "Tvw", "Tso", "Tvo", "Tg", "dem", "Hr", "mod", "fresnel"]
ELEV = ["od", "mod", "aod", "rho_p", "Ts", "tsd", "tss", "Tv", "tvd", "tvs", "T", "s", "F", "Hr", "Ha"]
LANDSAT = (["ULTRABLUE", "BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"], [21, 22, 23, 24, 25, 27, 28])
SENTINEL2 = (["ULTRABLUE", "BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "VAPOR", "CIRRUS",
              "SWIR1", "SWIR2"], list(range(30, 43)))


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_atmo' / 'make_harness.sh')}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness build failed: {r.stdout}{r.stderr}")


def ordered(a):
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp_compare(a, b):
    """(values, bit-equal, 1 ulp, more, max ulp) of float32 arrays, NaN == NaN."""
    a, b = np.asarray(a, F32).ravel(), np.asarray(b, F32).ravel()
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    d = np.abs(ordered(a) - ordered(b))
    d = np.where(nan_a & nan_b, 0, d)
    d = np.where(nan_a ^ nan_b, 1 << 40, d)
    return a.size, int((d == 0).sum()), int((d == 1).sum()), int((d > 1).sum()), int(d.max()) if a.size else 0


def rt(name, *cols):
    from zeit import _core

    return _core.radtran.radtran_eval(name, *[np.ascontiguousarray(c, F32) for c in cols])


def smooth_field(rng, ny, nx, scale, n=6):
    ii, jj = np.mgrid[0:ny, 0:nx].astype(np.float64)
    f = np.zeros((ny, nx))
    for _ in range(n):
        a = rng.uniform(0, 2 * np.pi)
        k = 2 * np.pi / (scale * rng.uniform(0.5, 2.0))
        f += rng.uniform(0.3, 1.0) * np.sin(k * (np.cos(a) * jj + np.sin(a) * ii) + rng.uniform(0, 2 * np.pi))
    return f / np.abs(f).max()


def make_scene(rng, mission, nx, ny, res, lat0=48.0, lon0=10.0, doy=172, hh=10, mm=30, domulti=True, doenv=True,
               aodmap=True, qai_kind="swath", aoi=False, nodata_cells=(), weights=True):
    """The inputs of one scene as the binding and the harness take them. ``mission`` 0 is Landsat, 1 Sentinel-2;
    ``lat0`` and ``lon0`` place the scene (the map coordinates are metres from the false origin x0 = 500000)."""
    domain, rsr = LANDSAT if mission == 0 else SENTINEL2
    rsr = np.array(rsr, np.int32)
    nb = len(domain)
    wvl = rt("wavelength", rsr)[0]
    npix = nx * ny
    kgeo = 9.0e-6
    x0, ulx = 500000.0, 500000.0 + float(rng.integers(-20, 20)) * 1000.0
    uly = lat0 / kgeo
    gnx = int(np.ceil(nx * res / 5000.0))
    gny = int(np.ceil(ny * res / 5000.0))
    gn = gnx * gny
    year, tz, ss = 2021, 0, 0
    # QAI
    qai = np.zeros((ny, nx), np.int16)
    ii, jj = np.mgrid[0:ny, 0:nx]
    if qai_kind == "swath":  # a parallelogram that leans, as a Landsat scene does
        shift = (ii * 0.2126).astype(int)
        qai[~((jj >= shift) & (jj < shift + int(nx * 0.7)))] |= 1
    elif qai_kind == "frame":
        qai[:3, :] |= 1
        qai[:, -4:] |= 1
        qai[ny // 3:ny // 3 + 5, nx // 4:nx // 4 + 9] |= 1
    if qai_kind in ("swath", "frame", "dirty"):  # flags of other stages: cloud, shadow, water, slope
        qai |= (rng.integers(0, 4, qai.shape) * (rng.random(qai.shape) < 0.2) << 1).astype(np.int16)
        qai |= ((rng.random(qai.shape) < 0.05).astype(np.int16) << 3)
        qai |= ((rng.random(qai.shape) < 0.05).astype(np.int16) << 5)
    # binned DEM and the elevation statistics
    dem_min, dem_max = 0.05, 2.9
    step = F32((dem_max - dem_min) / 255.0)
    z = 0.5 + 0.4 * smooth_field(rng, ny, nx, 60.0 * 30.0 / res) + 0.1 * rng.random((ny, nx))
    top_dem = np.clip((z * 200.0).astype(int), 0, 255).astype(np.uint8)
    top_dem = np.where(rng.random((ny, nx)) < 0.01, rng.integers(0, 256, (ny, nx)), top_dem).astype(np.uint8)
    dem4 = np.array([1.0, dem_min, dem_max, step], F32)
    mod = rt("molecular_optical_depth", wvl)[0]
    aod = (0.12 * (wvl / 0.55) ** -1.3).astype(F32)
    # the AOD per band and coarse cell, a smooth field (some cells above 0.6 at low elevation), and the cells that were
    # interpolated
    base = (0.45 + 0.5 * smooth_field(rng, gny, gnx, 3.0)).ravel() if gn > 1 else np.array([0.65])
    xy_aod = np.stack([base * (wvl[b] / 0.55) ** -1.3 for b in range(nb)]).astype(F32)
    xy_interp = (rng.random(gn) < 0.3).astype(F32)
    sc = dict(mission=mission, nx=nx, ny=ny, res=float(res), ulx=ulx, uly=uly, domain=domain, wavelength=wvl,
              rsr_band=rsr, year=year, doy=doy, hh=hh, mm=mm, ss=ss, tz=tz, lon0=lon0, x0=x0, kgeo=kgeo,
              domulti=int(domulti), doenv=int(doenv), aodmap=int(aodmap), dem4=dem4, cnum=255,
              wvp_hp_ha=np.array([rng.uniform(0.5, 4.0), 2.0, 0.84], F32), mod=mod, aod=aod, gn=gn, xy_aod=xy_aod.ravel(),
              xy_interp=xy_interp, qai=qai.ravel(), top_dem=top_dem.ravel(), s2nx=0, s2ny=0, s2nodata=-9999.0,
              vzen=np.zeros(0, F32), vazi=np.zeros(0, F32), gnx=gnx, gny=gny)
    if mission == 1:
        gj, gi = np.meshgrid(np.arange(gnx), np.arange(gny))
        vzen = (2.0 + 9.0 * (gj / max(gnx - 1, 1)) + rng.uniform(0, 1.5, gj.shape)).astype(F32)
        vazi = (95.0 + 40.0 * (gi / max(gny - 1, 1)) + rng.uniform(0, 3.0, gj.shape) + 100 * (gj % 2)).astype(F32)
        for (e, f) in nodata_cells:
            if e < gny and f < gnx:
                vzen[e, f] = -9999.0
        sc.update(s2nx=gnx, s2ny=gny, vzen=vzen.ravel(), vazi=vazi.ravel())
    # the sun of the coarse cells, as zeit's sunpos gives it, keyed by the latitude and longitude of the cell as the C
    # computes them (upper left corner: ulx + f * 5000, uly - e * 5000)
    e, f = np.mgrid[0:gny, 0:gnx]
    x = ulx + f.ravel() * 5000.0
    y = uly + e.ravel() * -5000.0
    lon = lon0 + (x - x0) * kgeo
    lat = y * kgeo
    n = lon.size
    zen, azi = rt("sunpos", lat, lon, np.full(n, year), np.full(n, doy), np.full(n, hh), np.full(n, mm), np.full(n, ss),
                  np.full(n, tz))
    sc["sun"] = (lat, lon, zen, azi)
    # the interpolation weights
    sc.update(w_ji=np.zeros(0, np.int32), coarse=np.zeros(0, F32), w_res=5000.0, w_full_res=100.0, w_nf=0, w_ne=0,
              w_nodata=-9999.0)
    if weights:
        wres, wfull = [(5000.0, 100.0), (5000.0, 30.0), (3000.0, 20.0), (5000.0, 10.0)][int(rng.integers(0, 4))]
        wnf, wne = int(rng.integers(1, 9)), int(rng.integers(1, 8))
        coarse = rng.uniform(0, 1, (wne, wnf)).astype(F32)
        coarse[rng.random(coarse.shape) < 0.25] = -9999.0
        coarse[0, 0] = -9999.0
        span_j, span_i = int(wnf * wres / wfull), int(wne * wres / wfull)
        pts = [(0, 0), (span_j - 1, span_i - 1), (span_j - 1, 0), (0, span_i - 1)]
        half = int(wres / wfull / 2)
        pts += [(half + k * half * 2, half + m * half * 2) for k in range(min(wnf, 4)) for m in range(min(wne, 4))]  # centres
        pts += [(int(rng.integers(0, span_j)), int(rng.integers(0, span_i))) for _ in range(1500)]
        sc.update(w_ji=np.array([p for p in pts if p[0] < span_j and p[1] < span_i], np.int32).ravel(),
                  coarse=coarse.ravel(), w_res=wres, w_full_res=wfull, w_nf=wnf, w_ne=wne)
    sc["aoi"] = (rng.random(npix) < 0.8).astype(np.uint8) if aoi else np.zeros(0, np.uint8)
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
        a([sc["domulti"], sc["doenv"], sc["aodmap"]], np.int32)
        a(sc["dem4"], F32)
        a([sc["cnum"]], np.int32)
        a(sc["wvp_hp_ha"], F32)
        a(sc["mod"], F32)
        a(sc["aod"], F32)
        a([sc["s2nx"], sc["s2ny"]], np.int32)
        a([sc["s2nodata"]], F32)
        if sc["mission"] == 1:
            a(sc["vzen"], F32)
            a(sc["vazi"], F32)
        a([sc["gn"]], np.int32)
        a(sc["xy_aod"], F32)
        a(sc["xy_interp"], F32)
        a(sc["qai"], np.int16)
        a(sc["top_dem"], np.uint8)
        lat, lon, zen, azi = sc["sun"]
        a([len(lat)], np.int32)
        rec = np.zeros(len(lat), np.dtype([("lat", "<f8"), ("lon", "<f8"), ("zen", "<f4"), ("azi", "<f4")]))
        rec["lat"], rec["lon"], rec["zen"], rec["azi"] = lat, lon, zen, azi
        f.write(rec.tobytes())
        npts = len(sc["w_ji"]) // 2
        a([npts], np.int32)
        if npts:
            a([sc["w_res"], sc["w_full_res"]], np.float64)
            a([sc["w_nf"], sc["w_ne"]], np.int32)
            a([sc["w_nodata"]], F32)
            a(sc["coarse"], F32)
            a(sc["w_ji"], np.int32)
        a([len(sc["aoi"]) > 0], np.int32)
        if len(sc["aoi"]):
            a(sc["aoi"], np.uint8)


class Reader:
    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def take(self, dtype, n):
        a = np.frombuffer(self.raw, dtype, n, self.pos)
        self.pos += a.nbytes
        return a

    def brick_meta(self):
        nb, nx, ny = (int(v) for v in self.take("<i4", 3))
        res = float(self.take("<f8", 1)[0])
        nodata = tuple(int(v) for v in self.take("<i4", nb))
        raw = bytes(self.take("u1", 16 * nb))
        dom = tuple(raw[16 * k:16 * k + 16].rstrip(b"\0").decode() for k in range(nb))
        return (nb, nx, ny, res, nodata, dom)


def parse(raw, sc, nb, gn, npix, npts):
    """The dumps of the stages (see force_atmo/main.inc), from the bytes of the C harness or of the binding."""
    r = Reader(raw)
    o = {}
    o["a_int"] = r.take("<i4", 4)
    o["a_nodata"] = r.take("<f4", 1)
    o["a_scal"] = np.concatenate([r.take("<f4", 4), r.take("<f4", 8), r.take("<f4", 4)])
    o["a_cnum"] = r.take("<i4", 1)
    o["a_view"] = r.take("<f4", 7)
    o["a_tthg"] = r.take("<f4", 10)
    o["a_aodmap"] = r.take("<i4", 1)
    per = []
    for _ in range(nb):
        per.append(np.concatenate([r.take("<f4", 7).astype(np.float64), r.take("<i4", 1).astype(np.float64)]))
    o["a_band"] = np.array(per, np.float64)
    o["a_xymeta"] = [r.brick_meta() for _ in range(len(XY))]
    o["a_xyzmeta"] = [r.brick_meta() for _ in range(2 * len(XYZ))]
    # sun_target_view
    o["sun_ret"] = int(r.take("<i4", 1)[0])
    o["xy_sun"] = r.take("<f4", 8 * gn).reshape(8, gn)
    o["xy_view"] = r.take("<f4", 8 * gn).reshape(8, gn)
    o["sun_scal"] = r.take("<f4", 4)
    o["sun_view"] = r.take("<f4", 7)
    o["qai_sun"] = r.take("<i2", npix)
    # atmo_angledep
    o["angle_ret"] = int(r.take("<i4", 1)[0])
    for k in ANGLE:
        n = nb if k in ("brdf", "Tsw", "Tvw", "Tso", "Tvo", "Tg", "mod") else 1
        o["angle_" + k] = r.take("<f4", n * gn).reshape(n, gn)
    # atmo_elevdep
    o["elev_ret"] = int(r.take("<i4", 1)[0])
    for k in ELEV:
        n = 1 if k in ("Hr", "Ha") else nb
        o["elev_" + k] = r.take("<f4", 256 * n * gn).reshape(256, n, gn)
    o["qai_elev"] = r.take("<i2", npix)
    # the weights
    wd = np.dtype([("g", "<i4", 4), ("w", "<f4", 6), ("v", "<f4")])
    o["weights"] = r.take(wd, npts)
    o["aoi_ret"] = int(r.take("<i4", 1)[0])
    o["qai_aoi"] = r.take("<i2", npix)
    o["ms"] = r.take("<f8", 3)
    assert r.pos == len(raw), (r.pos, len(raw))
    return o


def run_force(name, sc):
    fin, fout = DATA / f"{name}.in.bin", DATA / f"{name}.out.bin"
    write_scene(fin, sc)
    cmd = f"OMP_NUM_THREADS=1 {EXE} scene {_wsl(fin)} {_wsl(fout)}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed: {r.stdout}{r.stderr}")
    return parse(fout.read_bytes(), sc, len(sc["domain"]), sc["gn"], sc["nx"] * sc["ny"], len(sc["w_ji"]) // 2)


def run_zeit(sc, n_jobs=1):
    from zeit import _core

    d = {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in sc.items() if k != "sun"}
    d["n_jobs"] = n_jobs
    raw, *_ = _core.atmo.scene(d)
    return parse(bytes(raw), sc, len(sc["domain"]), sc["gn"], sc["nx"] * sc["ny"], len(sc["w_ji"]) // 2)


class Table:
    def __init__(self):
        self.bad, self.tv, self.te, self.t1 = False, 0, 0, 0
        self.first = True

    def line(self, title, field, n, same, one, more, mx, note=""):
        self.tv += n
        self.te += same
        self.t1 += one
        flag = ""
        if more:
            self.bad = True
            flag = "  <-- MORE THAN 1 ULP"
        print(f"{title if self.first else '':44s} {field:16s} {n:9d} {same:9d} {one:6d} {more:6d} {mx:5d}{note}{flag}")
        self.first = False

    def floats(self, title, field, a, b, note=""):
        n, same, one, more, mx = ulp_compare(a, b)
        self.line(title, field, n, same, one, more, mx, note)

    def exact(self, title, field, a, b, note=""):
        a, b = np.asarray(a), np.asarray(b)
        same = int((a == b).sum()) if a.shape == b.shape else 0
        self.line(title, field, a.size, same, 0, a.size - same, 0 if same == a.size else 9, note)


def compare(table, title, sc, c, z):
    table.first = True
    T = lambda field, a, b, note="": table.floats(title, field, a, b, note)  # noqa: E731
    E = lambda field, a, b, note="": table.exact(title, field, a, b, note)  # noqa: E731
    E("alloc ints", c["a_int"], z["a_int"], f" (nc {c['a_int'][2]}, res {c['a_int'][3]})")
    T("alloc nodata, scalars", np.concatenate([c["a_nodata"], c["a_scal"], c["a_view"], c["a_tthg"]]),
      np.concatenate([z["a_nodata"], z["a_scal"], z["a_view"], z["a_tthg"]]))
    E("alloc cnum, aodmap", np.concatenate([c["a_cnum"], c["a_aodmap"]]), np.concatenate([z["a_cnum"], z["a_aodmap"]]))
    T("alloc per band", c["a_band"], z["a_band"], " (wvl, lwvl, lwvl2, E0, od, mod, aod)")
    E("alloc bricks", [str(m) for m in c["a_xymeta"]] + [str(m) for m in c["a_xyzmeta"]],
      [str(m) for m in z["a_xymeta"]] + [str(m) for m in z["a_xyzmeta"]], f" ({len(c['a_xymeta']) + len(c['a_xyzmeta'])} bricks)")
    E("sun return", c["sun_ret"], z["sun_ret"])
    for k, nm in ((0, "zen"), (1, "azi"), (2, "cos zen"), (3, "cos azi"), (4, "sin zen"), (5, "sin azi"), (6, "tan zen"),
                  (7, "tan azi")):
        T(f"sun {nm}", c["xy_sun"][k], z["xy_sun"][k])
    nodata_cells = int((c["xy_view"][0] == -9999).sum())
    for k, nm in ((0, "zen"), (1, "azi"), (2, "cos zen"), (3, "cos azi"), (4, "sin zen"), (5, "sin azi"), (6, "tan zen"),
                  (7, "tan azi")):
        T(f"view {nm}", c["xy_view"][k], z["xy_view"][k], f" ({nodata_cells} cells without data)" if k == 0 else "")
    T("cosszen, cosvzen", c["sun_scal"], z["sun_scal"], f" (sun {c['sun_scal'][0]:.4f}..{c['sun_scal'][1]:.4f})")
    T("view geometry", c["sun_view"], z["sun_view"], " (a, b, c, ab, angle, H, H2)" if sc["mission"] == 0 else " (H, H2)")
    E("QAI after sun", c["qai_sun"], z["qai_sun"],
      f" ({int(((c['qai_sun'] >> 10) & 1).sum())} low sun, {int((c['qai_sun'] & 1).sum())} off)")
    E("angledep return", c["angle_ret"], z["angle_ret"])
    for k in ANGLE:
        T("angle " + k, c["angle_" + k], z["angle_" + k])
    E("elevdep return", c["elev_ret"], z["elev_ret"])
    for k in ELEV:
        T("elev " + k, c["elev_" + k], z["elev_" + k])
    aer = [int((((c["qai_elev"] >> 6) & 3) == v).sum()) for v in range(4)]
    E("QAI after elevdep", c["qai_elev"], z["qai_elev"], f" (aerosol 0/1/2/3: {aer})")
    if len(c["weights"]):
        E("weights cells", c["weights"]["g"], z["weights"]["g"], f" ({len(c['weights'])} pixels)")
        T("weights", c["weights"]["w"], z["weights"]["w"])
        T("interpolated value", c["weights"]["v"], z["weights"]["v"])
    E("apply_aoi", np.concatenate([[c["aoi_ret"]], c["qai_aoi"]]), np.concatenate([[z["aoi_ret"]], z["qai_aoi"]]),
      f" (return {c['aoi_ret']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the area of the large scenes")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    no_power_throttling()
    rng = np.random.default_rng(23)
    if not args.skip_build:
        build_harness()
    big = int(round(1830 * np.sqrt(args.scale)))
    scenes = [
        ("Landsat 100 m, 800x700, tilted swath", dict(mission=0, nx=800, ny=700, res=100.0)),
        ("Landsat 30 m, 1200x1000, tilted, AOI", dict(mission=0, nx=1200, ny=1000, res=30.0, aoi=True)),
        ("Landsat 100 m, 1000x400 (cols > rows)", dict(mission=0, nx=1000, ny=400, res=100.0, lat0=52.0)),
        ("Landsat 100 m, 400x1000 (rows > cols)", dict(mission=0, nx=400, ny=1000, res=100.0, lat0=35.0, doy=300)),
        ("Landsat, Dec, 70 N: low sun", dict(mission=0, nx=700, ny=600, res=100.0, lat0=70.0, doy=355, hh=11, mm=0)),
        ("Landsat, south and west, no AOD map", dict(mission=0, nx=600, ny=600, res=100.0, lat0=-20.0, lon0=-60.0, doy=30,
                                                     hh=14, aodmap=False, doenv=False)),
        ("Landsat, single scattering", dict(mission=0, nx=500, ny=500, res=100.0, lat0=5.0, lon0=-10.0, domulti=False)),
        ("Sentinel-2 20 m, 1500x1200, no-data cells", dict(mission=1, nx=1500, ny=1200, res=20.0, qai_kind="frame",
                                                           nodata_cells=[(1, 1), (3, 4), (0, 5)], aoi=True)),
        ("Sentinel-2 10 m, 1000x900", dict(mission=1, nx=1000, ny=900, res=10.0, qai_kind="dirty", nodata_cells=[(1, 0)],
                                           lat0=-33.0, lon0=150.0, doy=20, hh=0, mm=15)),
        ("Sentinel-2 60 m, 900x300 (cols > rows)", dict(mission=1, nx=900, ny=300, res=60.0, qai_kind="frame",
                                                        nodata_cells=[(0, 3), (2, 7)], aodmap=False)),
        ("Sentinel-2 60 m, 300x900 (rows > cols)", dict(mission=1, nx=300, ny=900, res=60.0, qai_kind="dirty",
                                                        nodata_cells=[(7, 1)], doenv=False)),
        (f"Sentinel-2 60 m tile, {big}x{big}", dict(mission=1, nx=big, ny=big, res=60.0, qai_kind="frame",
                                                    nodata_cells=[(5, 5), (20, 3), (0, 0)])),
        ("one coarse cell (Sentinel-2 10 m, 400x400)", dict(mission=1, nx=400, ny=400, res=10.0, qai_kind="clean")),
    ]
    print(f"{'scene':44s} {'field':16s} {'values':>9s} {'equal':>9s} {'1 ulp':>6s} {'more':>6s} {'max':>5s}")
    table = Table()
    for k, (title, kw) in enumerate(scenes):
        sc = make_scene(rng, **kw)
        c = run_force(f"scene{k}", sc)
        z = run_zeit(sc)
        compare(table, title, sc, c, z)
        print(f"{'':44s} FORCE {c['ms'][0]:.1f} + {c['ms'][1]:.1f} + {c['ms'][2]:.1f} ms (sun_target_view, angledep, "
              f"elevdep), Zeit {z['ms'][0]:.1f} + {z['ms'][1]:.1f} + {z['ms'][2]:.1f} ms (1 thread)")
        if k == 0:
            z4 = run_zeit(sc, n_jobs=4)
            same = all(np.array_equal(z[f], z4[f]) for f in z if f.startswith(("angle_", "elev_", "qai_")))
            print(f"{'':44s} Zeit with 4 threads vs 1: every field equal: {same}")
            table.bad |= not same
    print(f"{'total':44s} {'':16s} {table.tv:9d} {table.te:9d} {table.t1:6d} {table.tv - table.te - table.t1:6d}")
    sys.exit(1 if table.bad else 0)


if __name__ == "__main__":
    main()
