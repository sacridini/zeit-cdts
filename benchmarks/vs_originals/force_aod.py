"""The dark targets and the AOD over each of them of zeit-core (``zeit_core::aod``, a port of the first half of FORCE
3.11.00's ``aod-ll.c``: ``aodfileread``, ``extract_dark_target`` for the water and the vegetation type, the spectral
libraries ``water_lib`` and ``veg_lib``, ``aod_lib_to_target``, ``aod_from_target`` with ``aod_linear_fit`` and
``aod_polynomial_fit`` (GSL's ``nmsimplex2`` in three dimensions), ``aod_elev_dependency`` (GSL's Brent minimizer), and
the ``dist_transform_`` and ``connectedcomponents_`` of ``imagefuns-cl.c`` they use) against FORCE's own functions on
synthetic scenes.

The C side is ``force_aod/`` (``make_harness.sh``, WSL): FORCE's aod-ll.c compiled as it is (up to ``aod_map``, and but
for its first include, behind shims for the brick, atc, meta, parameter and topography structures), with radtran-ll.c and
table-ll.c, against the real GSL 2.7 and its reference CBLAS, linked statically; run with ``OMP_NUM_THREADS=1`` (the
scene-average environment reflectance of FORCE is a sum per thread, and its loop over the targets hands each thread one
set of work arrays, so the output depends on the number of threads; Zeit's is FORCE with one thread, see the notes of
``zeit_core::aod``). Zeit's side is ``zeit._core.aod`` (Windows, UCRT libm; the harness runs on glibc 2.39)::

    python benchmarks/vs_originals/force_aod.py [--scale 1.0] [--skip-build]

Scenes: Sentinel-2 (13 bands of the RSR table, domains ULTRABLUE to SWIR2) at 10 m with a coarse grid of 150 to 300 m
cells; a land background with a smooth field of reflectances; lakes (ellipses, one with an island), a river across
the image that the grid cuts, ponds and streams too small or narrow, salt-and-pepper water pixels, dark vegetation
patches (some touching), cloud, cirrus, shadow and no-data regions (a frame, a block, the cells outside the swath),
steep targets, high ones (above 1 km) and flat ones; the target spectra come from the forward model of the radiative
transfer for a smooth AOD field and the libraries of FORCE (so that many targets pass the AOD estimation), with noise.
Prints, per scene and stage, the number of values compared, how many are equal bit for bit and how many differ (the
integers exactly, the floats in float32 ulps): the target lists (position, size, radius, elevation, reflectance of
target and environment), the band-wise AOD that best matches each library spectrum, the fitted AOD and its
coefficients, the elevation factors. Exits with 1 if anything differs by more than 1 float32 ulp. Then the elevation
dependency on small random lists of targets (Brent), the files of ``aodfileread`` and the segmentation functions.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import no_power_throttling, workdir

HERE = Path(__file__).resolve().parent
DATA = workdir("force_aod")
EXE = "$HOME/force-ref-work/aod/aod_harness"
F32 = np.float32
RSR0 = 30  # RSR_START_SEN2A
NB = 13
DOMAIN = ["ULTRABLUE", "BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "VAPOR", "CIRRUS",
          "SWIR1", "SWIR2"]
CIRRUS_QAI = 3 << 1


def _wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def build_harness():
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_aod' / 'make_harness.sh')}"],
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


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.size, int((a == b).sum()), int((a != b).sum())


def rt(name, *cols):
    from zeit import _core

    return _core.radtran.radtran_eval(name, *[np.ascontiguousarray(c, F32) for c in cols])


def ellipse(ny, nx, ci, cj, ri, rj, rot=0.0):
    ii, jj = np.mgrid[0:ny, 0:nx]
    di, dj = ii - ci, jj - cj
    c, s = np.cos(rot), np.sin(rot)
    u, v = c * di + s * dj, -s * di + c * dj
    return (u / ri) ** 2 + (v / rj) ** 2 <= 1.0


def make_scene(rng, nx, ny, cs, kind="normal"):
    """The inputs of one scene (arrays as the binding and the harness take them); ``cs`` is the coarse cell in pixels."""
    from zeit import _core

    res = 10.0
    gres = res * cs
    gnx, gny = int(np.ceil(nx / cs)), int(np.ceil(ny / cs))
    gn, npix = gnx * gny, nx * ny
    nodata = -9999
    rsr = (RSR0 + np.arange(NB)).astype(np.int32)
    wvl = rt("wavelength", rsr)[0]
    lwvl = np.log(wvl.astype(np.float64)).astype(F32)
    e0 = rt("e0", rsr)[0]
    aod_bands = ((wvl < 0.7) | (np.arange(NB) == 12)).astype(np.uint8)
    gj, gi = np.meshgrid(np.arange(gnx), np.arange(gny))
    gi, gj = gi.ravel(), gj.ravel()

    def U(lo, hi, *shape):
        return rng.uniform(lo, hi, shape).astype(F32)

    # geometry and atmosphere per coarse cell
    sun = (0.62 + 0.15 * gi / max(gny - 1, 1) + 0.02 * rng.random(gn)).astype(F32)
    view = (0.95 + 0.04 * gj / max(gnx - 1, 1) + 0.004 * rng.random(gn)).astype(F32)
    sun_azi, view_azi = U(130, 170, gn), U(80, 280, gn)
    swath_out = gj == gnx - 1  # the cells of the last column are outside the swath
    view[swath_out] = nodata
    view_azi[swath_out] = nodata
    dem_bin = np.clip(30 + 7 * gi + 5 * gj + rng.integers(-4, 5, gn), 0, 255).astype(F32)
    dem_step, dem_min = F32(0.0078), F32(0.0)
    z_cell = (dem_bin * dem_step + dem_min + dem_step / 2).astype(F32)
    hr = rt("mod_elev_factor", z_cell)[0]
    mod = np.stack([rt("molecular_optical_depth", np.full(gn, wvl[b], F32))[0] * hr for b in range(NB)]).astype(F32)
    tso, tvo = U(0.95, 1.0, NB, gn), U(0.95, 1.0, NB, gn)
    tg = U(0.88, 0.99, NB, gn)
    tg[9] = U(0.4, 0.7, gn)
    brdf = U(0.9, 1.1, NB, gn)
    pr, pa, fresnel = U(0.9, 1.4, gn), U(0.3, 1.5, gn), U(0.02, 0.05, gn)
    sob = F32(0.84)
    aod550 = (0.08 + 0.22 * (0.5 + 0.5 * np.sin(gi / 2.3) * np.cos(gj / 1.7))).astype(F32)

    # the forward model: TOA reflectance of a uniform surface (per band and cell) for the AOD of each cell
    water_lib, veg_lib = _core.aod.libraries(list(map(int, rsr)), 1, 2, 3)
    water_lib = water_lib.reshape(-1, NB)
    vegs = np.array([[0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.34, 0.35, 0.35, 0.34, 0.01, 0.15, 0.04]], F32)
    vegs = np.repeat(vegs, 6, 0)
    vegs[:, 1] = np.linspace(0.012, 0.035, 6)  # blue
    vegs[:, 2] = vegs[:, 1] + np.linspace(0.015, 0.04, 6)  # green
    vegs[:, 3] = vegs[:, 1] * 1.1  # red
    vegs[:, 0] = vegs[:, 1] * 0.8
    landspec = np.array([0.08, 0.09, 0.11, 0.14, 0.21, 0.26, 0.29, 0.3, 0.31, 0.3, 0.1, 0.36, 0.28], F32)

    def forward(surface, water):
        """surface: (n, NB) reflectance, returns (n, NB, gn) TOA reflectance."""
        out = np.zeros((surface.shape[0], NB, gn))
        for b in range(NB):
            aod = (aod550 * (wvl[b] / 0.55) ** -1.3).astype(F32)
            od = rt("optical_depth", aod, mod[b])[0]
            T, ts, tv, tsd, tss, tvd, tvs = rt("scatt_transmitt", aod, mod[b], od, sun, np.where(swath_out, F32(0.97), view))
            rhop = rt("path_ref", np.ones(gn), np.full(gn, sob), aod, mod[b], od, pa, pr, tsd, tvd, sun, view)[0]
            s = rt("sphere_albedo", aod, mod[b], od)[0]
            for k in range(surface.shape[0]):
                rho = surface[k, b]
                rhow = (rho + fresnel * tss) / ts if water else np.full(gn, rho)
                out[k, b] = tg[b] * (rhop + ts * (tvd * rhow + tvs * rhow) / (1 - rhow * s))
        return out

    view_ok = np.where(swath_out, F32(0.97), view)
    view_save = view.copy()
    view = view_ok  # the forward model needs a usable view cosine in the cells outside the swath
    t_water = forward(water_lib, True)  # (26, NB, gn)
    t_veg = forward(vegs, False)
    t_land = forward(landspec[None], False)[0]  # (NB, gn)
    view = view_save

    # the landscape
    ii, jj = np.mgrid[0:ny, 0:nx]
    cls = np.zeros((ny, nx), np.int32)  # 0 land, 1..26 water spectrum + 1, 100..105 vegetation spectrum + 100
    slope = rng.integers(20, 120, (ny, nx)).astype(np.uint16)
    ndisk = 10
    for k in range(ndisk):
        ci, cj = rng.uniform(0.1 * ny, 0.9 * ny), rng.uniform(0.1 * nx, 0.9 * nx)
        ri, rj, rot = rng.uniform(5, 14), rng.uniform(5, 14), rng.uniform(0, np.pi)
        m = ellipse(ny, nx, ci, cj, ri, rj, rot)
        if k == 1:  # a lake with an island
            m &= ~ellipse(ny, nx, ci, cj, ri * 0.35, rj * 0.35, rot)
        cls[m] = 1 + int(rng.integers(0, 26))
        if k == 2:
            slope[m] = 300  # steep: fails above 1 km
        # vegetation patches next to some lakes, touching them (8-connected, different classes)
        cv = ellipse(ny, nx, ci + rng.uniform(-18, 18), cj + rng.uniform(-18, 18), rng.uniform(6, 16), rng.uniform(6, 16),
                     rng.uniform(0, np.pi))
        cv &= cls == 0
        cls[cv] = 100 + int(rng.integers(0, 6))
        if k == 3:
            slope[cv] = 3000  # too steep for vegetation
    # a river across the image, which the coarse grid cuts into pieces, and a stream too narrow for a target
    ri_ = int(0.62 * ny)
    for j in range(nx):
        i0 = ri_ + int(4 * np.sin(j / 9.0))
        cls[max(i0 - 4, 0):i0 + 5, j] = np.where(cls[max(i0 - 4, 0):i0 + 5, j] == 0, 3, cls[max(i0 - 4, 0):i0 + 5, j])
    cls[int(0.2 * ny):int(0.2 * ny) + 4, 5:nx - 5] = 7  # 4 pixels wide: radius under 3
    cls[3:6, 3:6] = 9  # a pond of 9 pixels: under 10
    salt = rng.random((ny, nx)) < 0.004
    cls[salt & (cls == 0)] = 12
    pix_cell = (np.minimum(ii // cs, gny - 1) * gnx + np.minimum(jj // cs, gnx - 1)).astype(int)

    toa = np.zeros((NB, ny, nx))
    base = t_land[:, pix_cell] * (0.7 + 0.6 * (0.5 + 0.5 * np.sin(ii / 17.0) * np.cos(jj / 23.0)))[None]
    toa[:] = base
    wmask = (cls >= 1) & (cls < 100)
    vmask = cls >= 100
    for b in range(NB):
        toa[b][wmask] = t_water[cls[wmask] - 1, b, pix_cell[wmask]]
        toa[b][vmask] = t_veg[cls[vmask] - 100, b, pix_cell[vmask]]
    toa = np.round(10000 * toa + rng.normal(0, 10 if kind != "noisy" else 40, toa.shape)).astype(np.int16)

    # QAI: cloud (some with cloudy TOA, some with the flag only), cirrus, shadow, no data
    qai = np.zeros((ny, nx), np.int16)
    for k in range(4):
        m = ellipse(ny, nx, rng.uniform(0, ny), rng.uniform(0, nx), rng.uniform(6, 20), rng.uniform(6, 20))
        c = int(rng.integers(1, 4))
        qai[m] |= c << 1
        if k % 2 == 0:
            toa[:, m] = np.round(10000 * rng.uniform(0.3, 0.6, (NB, m.sum()))).astype(np.int16)
    for k in range(3):
        m = ellipse(ny, nx, rng.uniform(0, ny), rng.uniform(0, nx), rng.uniform(5, 15), rng.uniform(5, 15))
        qai[m] |= 1 << 3
    qai[(rng.random((ny, nx)) < 0.01)] |= 1 << 1  # scattered cloud
    qai[(rng.random((ny, nx)) < 0.01)] |= 1 << 3  # scattered shadow
    qai |= (rng.integers(0, 2, (ny, nx)) << 5).astype(np.int16)  # water flag: unread
    qai[:, (gnx - 1) * cs:] |= 1  # off: the cells outside the swath
    qai[:2, :] |= 1
    qai[ny // 2 - 6:ny // 2 + 6, nx // 3:nx // 3 + 12] |= 1  # a block without data
    toa[:, qai & 1 != 0] = -9999
    if kind == "allcloud":
        qai |= 1 << 1
    if kind == "empty":  # no candidates at all
        toa = np.round(10000 * np.broadcast_to(t_land[:, pix_cell], toa.shape)).astype(np.int16)
        toa[:, qai & 1 != 0] = -9999
    ill = np.clip(rng.normal(8200, 700, (ny, nx)), 3000, 10000).astype(np.int16)
    ill[(ii > 2 * ny // 3) & (jj < nx // 4)] = 4000
    sky = np.clip(rng.normal(9600, 300, (ny, nx)), 5000, 10000).astype(np.uint16)
    top_dem = np.clip(dem_bin[pix_cell] + rng.integers(-3, 4, (ny, nx)), 0, 255).astype(np.uint8)
    dem = np.array([float(z_cell.mean()), 0.0, 2.0, float(dem_step)], F32)
    return dict(nb=NB, nx=nx, ny=ny, res=res, gnx=gnx, gny=gny, gres=float(gres), nodata_view=nodata, multi=True,
                domain=DOMAIN, rsr_band=rsr, aod_bands=aod_bands, wvl=wvl, lwvl=lwvl, e0=e0, dem=dem,
                sob=np.array([sob], F32), sun_czen=sun, sun_azi=sun_azi, view_czen=view, view_azi=view_azi,
                tso=tso.ravel(), tvo=tvo.ravel(), tg=tg.ravel(), mod=mod.ravel(), brdf=brdf.ravel(), pr=pr, pa=pa, hr=hr,
                fresnel=fresnel, dem_bin=dem_bin, toa=toa.reshape(NB, -1).ravel(), qai=qai.ravel(),
                top_dem=top_dem.ravel(), slope=slope.ravel(), ill=ill.ravel(), sky=sky.ravel())


def write_scene(path, sc):
    with open(path, "wb") as f:
        a = lambda x, dt: f.write(np.ascontiguousarray(x, dt).tobytes())  # noqa: E731
        a([sc["nb"], sc["nx"], sc["ny"]], np.int32)
        a([sc["res"]], np.float64)
        a([sc["gnx"], sc["gny"]], np.int32)
        a([sc["gres"]], np.float64)
        a([sc["nodata_view"], int(sc["multi"])], np.int32)
        for d in sc["domain"]:
            f.write(d.encode().ljust(16, b"\0"))
        a(sc["rsr_band"], np.int32)
        a(sc["aod_bands"], np.uint8)
        for k in ("wvl", "lwvl", "e0", "dem", "sob", "sun_czen", "sun_azi", "view_czen", "view_azi", "tso", "tvo", "tg",
                  "mod", "brdf", "pr", "pa", "hr", "fresnel", "dem_bin"):
            a(sc[k], F32)
        a(sc["toa"], np.int16)
        a(sc["qai"], np.int16)
        a(sc["top_dem"], np.uint8)
        a(sc["slope"], np.uint16)
        a(sc["ill"], np.int16)
        a(sc["sky"], np.uint16)


def veg_lib_size():
    """The number of spectra of FORCE's vegetation library (its float loops: 0.01 to 0.05 in steps of 0.005)."""
    n, rb = 0, F32(0.01)
    while float(rb) <= 0.05:
        g = F32(0.01)
        while float(g) <= 0.05:
            n += 1
            g = F32(float(g) + 0.005)
        rb = F32(float(rb) + 0.005)
    return n


def parse(raw, nb, gn):
    """The dumps of the stages (see force_aod/main.inc), from the bytes of the C harness or of the binding."""
    pos = 0

    def take(dtype, n):
        nonlocal pos
        a = np.frombuffer(raw, dtype, n, pos)
        pos += a.nbytes
        return a

    obj = np.dtype([("i", "<i4", 12), ("f", "<f4", 7), ("ttoa", "<f4", nb), ("etoa", "<f4", nb)])
    out = {}
    for kind in ("wat", "veg"):
        n = int(take("<i4", 1)[0])
        out["ext_" + kind] = take(obj, n)
    wat_n, veg_n = 26, veg_lib_size()
    for kind, ln in (("wat", wat_n), ("veg", veg_n)):
        n = int(take("<i4", 1)[0])
        out["lib_" + kind] = take(np.dtype(("<f4", (ln, nb))), n)
    fit = np.dtype([("i", "<i4", 3), ("rsq", "<f4"), ("coef", "<f4", 3), ("aod", "<f4", nb), ("est", "<f4", nb)])
    for kind in ("wat", "veg"):
        k = int(take("<i4", 1)[0])
        out["k_" + kind] = k
        out["fit_" + kind] = take(fit, len(out["ext_" + kind]))
    out["hp"], out["ha"] = take("<f4", 2)
    out["xy_ha"] = take("<f4", gn)
    out["ha_obj"] = take("<f4", len(out["ext_wat"]) + len(out["ext_veg"]))
    assert pos == len(raw), (pos, len(raw))
    return out


def run_force(name, sc, threads=1):
    fin, fout = DATA / f"{name}.in", DATA / f"{name}.out"
    write_scene(fin, sc)
    cmd = f"OMP_NUM_THREADS={threads} {EXE} scene {_wsl(fin)} {_wsl(fout)}"
    r = subprocess.run(["wsl", "-e", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"harness failed on {name}: {r.stdout}{r.stderr}")
    ms = [float(x) for x in r.stdout.split()[-2:]]
    return parse(fout.read_bytes(), NB, sc["gnx"] * sc["gny"]), ms


def run_zeit(sc):
    from zeit import _core

    raw, t_ext, t_fit = _core.aod.scene(sc)
    return parse(raw, NB, sc["gnx"] * sc["gny"]), [t_ext, t_fit]


def compare(title, c, z):
    """Print the rows of one scene; returns (bad, values, equal)."""
    bad, tv, te = False, 0, 0
    rows = []
    for kind in ("wat", "veg"):
        e_c, e_z = c["ext_" + kind], z["ext_" + kind]
        if len(e_c) != len(e_z):
            print(f"{title}: {kind}: {len(e_c)} candidate targets in FORCE, {len(e_z)} in Zeit")
            return True, 0, 0
        rows += [(f"{kind} targets: integers", exact(e_c["i"], e_z["i"])),
                 (f"{kind} targets: floats", ulp_compare(np.concatenate([e_c["f"].ravel(), e_c["ttoa"].ravel(), e_c["etoa"].ravel()]),
                                                          np.concatenate([e_z["f"].ravel(), e_z["ttoa"].ravel(), e_z["etoa"].ravel()]))),
                 (f"{kind} library AOD", ulp_compare(c["lib_" + kind], z["lib_" + kind])),
                 (f"{kind} fit: ids", exact(np.append(c["fit_" + kind]["i"].ravel(), c["k_" + kind]),
                                           np.append(z["fit_" + kind]["i"].ravel(), z["k_" + kind]))),
                 (f"{kind} fit: AOD, R2, coefficients",
                  ulp_compare(np.concatenate([c["fit_" + kind]["aod"].ravel(), c["fit_" + kind]["est"].ravel(),
                                              c["fit_" + kind]["rsq"], c["fit_" + kind]["coef"].ravel()]),
                              np.concatenate([z["fit_" + kind]["aod"].ravel(), z["fit_" + kind]["est"].ravel(),
                                              z["fit_" + kind]["rsq"], z["fit_" + kind]["coef"].ravel()])))]
    rows.append(("elevation: Hp, Ha, cells, targets",
                 ulp_compare(np.concatenate([[c["hp"], c["ha"]], c["xy_ha"], c["ha_obj"]]),
                             np.concatenate([[z["hp"], z["ha"]], z["xy_ha"], z["ha_obj"]]))))
    for name, r in rows:
        if len(r) == 3:  # exact: (n, equal, differ)
            n, e, d = r
            mx = 0
            diff = d
        else:
            n, e, one, more, mx = r
            diff = one + more
            bad |= more > 0
        bad |= len(r) == 3 and r[2] > 0
        tv += n
        te += e
        print(f"{title + ': ' + name:58s} {n:9d} {e:12d} {diff:8d} {mx:6d}")
    return bad, tv, te


def check_elev(rng, scale):
    from zeit import _core

    cases = []
    for k in range(int(300 * scale)):
        nt = int(rng.choice([0, 3, 4, 5, 8, 20, 60, 60, 100]))
        mn = float(rng.choice([0.0, 0.0, 0.2, -0.3]))
        mx = mn + float(rng.choice([0.0, 0.05, 1.0, 1.0, 2.4, 3.8]))
        step = (mx - mn) / 256
        avg = (mn + mx) / 2
        z = rng.uniform(mn, mx if mx > mn else mn + 1, nt) if k % 7 else np.full(nt, mn + 0.5)
        hs = float(rng.choice([0.3, 0.7, 1.5, 3.0, 8.0, 30.0]))
        a = 0.2 * np.exp(-z / hs) + rng.normal(0, 0.01, nt) if k % 5 else rng.uniform(0.02, 0.5, nt)
        if k % 11 == 0 and nt:
            a[0] = 1e-3
        view = [-9999.0 if rng.random() < 0.25 else 0.97, 0.97, 0.97, -9999.0]
        dembin = rng.integers(0, 256, 4).astype(float)
        cases.append((F32(avg), F32(mn), F32(mx), F32(step), [F32(v) for v in view], [F32(v) for v in dembin],
                      [(F32(zz), F32(aa)) for zz, aa in zip(z, a)]))
    lines = [str(len(cases))]
    for avg, mn, mx, step, view, dembin, tg in cases:
        lines.append(f"{float(avg).hex()} {float(mn).hex()} {float(mx).hex()} {float(step).hex()} {len(tg)}")
        lines.append(" ".join(f"{float(v).hex()} {float(d).hex()}" for v, d in zip(view, dembin)))
        lines.append(" ".join(f"{float(zz).hex()} {float(aa).hex()}" for zz, aa in tg))
    fin = DATA / "elev.txt"
    fin.write_text("\n".join(lines) + "\n")
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"{EXE} elev < {_wsl(fin)}"], capture_output=True, text=True)
    ref = [[int(t, 16) for t in ln.split()] for ln in r.stdout.splitlines()]
    out = _core.aod.elev_cases(cases)
    n = sum(len(a) for a in ref)
    same = sum(int(x == y) for a, b in zip(ref, out) for x, y in zip(a, b))
    hp_flat = sum(1 for a in ref if np.uint32(a[0]).view(F32) == 1000)
    print(f"{'aod_elev_dependency (Brent)':34s} {'values':>9s} {n:9d} {same:12d} {n - same:8d}   "
          f"({len(cases)} cases, {hp_flat} with the flat AOD, Hp 1000)")
    return len(ref) == len(out) and same == n


def check_aodfile(rng, scale):
    from zeit import _core

    rsr = np.arange(RSR0, RSR0 + NB, dtype=np.int32)
    wvl = rt("wavelength", rsr)[0]
    lwvl = np.log(wvl.astype(np.float64)).astype(F32)
    root = DATA / "aod"
    root.mkdir(exist_ok=True)
    for f in root.glob("AOD_*.txt"):
        f.unlink()
    sites = [(float(x), float(y)) for x, y in zip(rng.uniform(-10, 10, 40), rng.uniform(40, 55, 40))]
    for doy in (45, 46, 200, 201, 365):
        with open(root / f"AOD_{doy:03d}.txt", "w") as f:
            for x, y in sites:
                co = (rng.uniform(-3.5, -1.5), rng.uniform(-2, -0.8), rng.uniform(-0.2, 0.05))
                sep = "  " if rng.random() < 0.2 else " "
                tail = "\r\n" if rng.random() < 0.3 else "\n"
                f.write(f"{x:.2f}{sep}{y:.2f} {co[0]:.4f} {co[1]:.4f} {co[2]:.4f}{tail}")
    with open(root / "AOD_100.txt", "w") as f:
        f.write("1.0 45.0 -2.5 -1.3 -0.05 extra tokens\n2.0 46.0 -2.0 -1.0 -0.1")  # a file without trailing newline
    scen = []
    for k in range(int(200 * scale)):
        doy = [45, 46, 200, 201, 365, 100, 77][k % 7]  # 77: no file (NULL)
        lon, lat = rng.uniform(-12, 12), rng.uniform(38, 57)
        scen.append((str(root) if k % 13 else "NULL", float(lon), float(lat), doy))
    lines = [str(NB), " ".join(float(v).hex() for v in wvl), " ".join(float(v).hex() for v in lwvl)]
    lines += [f"{_wsl(root) if s[0] != 'NULL' else 'NULL'} {s[1]!r} {s[2]!r} {s[3]}" for s in scen]
    fin = DATA / "aodfile.txt"
    fin.write_text("\n".join(lines) + "\n")
    r = subprocess.run(["wsl", "-e", "bash", "-lc", f"{EXE} aodfile < {_wsl(fin)}"], capture_output=True, text=True)
    # (FORCE prints its messages on stdout too: only "NULL" and the lines of hexadecimal bits are results)
    lines_ = [ln for ln in r.stdout.splitlines() if ln == "NULL" or all(len(t) == 8 for t in ln.split())]
    ref = [None if ln == "NULL" else [int(t, 16) for t in ln.split()] for ln in lines_]
    out = _core.aod.aodfile(list(wvl), list(lwvl), scen)
    nnull = sum(1 for a in ref if a is None)
    same = sum(int(a == b) for a, b in zip(ref, out))
    print(f"{'aodfileread':34s} {'scenarios':>9s} {len(scen):9d} {same:12d} {len(scen) - same:8d}   (of which {nnull} NULL: "
          "no file)")
    return len(ref) == len(out) and same == len(scen)


def check_image(rng):
    from zeit import _core

    tot = bad = 0
    for k, (nx, ny, p) in enumerate([(40, 30, 0.5), (97, 61, 0.7), (64, 64, 0.3), (1, 20, 0.5), (30, 1, 0.5),
                                     (200, 150, 0.6), (120, 90, 0.2)]):
        img = (rng.random((ny, nx)) < p).astype(np.uint8)
        if k == 5:  # blobs, with holes
            from scipy.ndimage import gaussian_filter

            img = (gaussian_filter(rng.random((ny, nx)), 3) > 0.5).astype(np.uint8)
        fin, fout = DATA / f"img{k}.in", DATA / f"img{k}.out"
        with open(fin, "wb") as f:
            f.write(np.array([nx, ny], np.int32).tobytes() + img.tobytes())
        r = subprocess.run(["wsl", "-e", "bash", "-lc", f"{EXE} image {_wsl(fin)} {_wsl(fout)}"], capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)
        raw = fout.read_bytes()
        n = int(np.frombuffer(raw, np.int32, 1)[0])
        dist = np.frombuffer(raw, np.uint16, nx * ny, 4)
        lab = np.frombuffer(raw, np.int32, nx * ny, 4 + 2 * nx * ny)
        dz, lz, nz = _core.aod.image_functions(img.ravel(), nx, ny)
        tot += 2 * nx * ny + 1
        bad += int((dist != dz).sum() + (lab != lz).sum() + (n != nz))
    print(f"{'dist_transform_, connectedcomponents_':34s} {'values':>9s} {tot:9d} {tot - bad:12d} {bad:8d}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of random cases and the big scene")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    no_power_throttling()
    rng = np.random.default_rng(23)
    if not args.skip_build:
        build_harness()
    big = int(round(240 * np.sqrt(args.scale)))
    scenes = [("small 120x90, 150 m cells", (120, 90, 15, "normal")),
              ("medium 240x200, 300 m cells", (240, 200, 30, "normal")),
              ("medium 240x200, 200 m cells, noisy", (240, 200, 20, "noisy")),
              (f"large {big * 2}x{big * 2 - 40}, 300 m cells", (big * 2, big * 2 - 40, 30, "normal")),
              ("all cloud 90x60 (no environment)", (90, 60, 15, "allcloud")),
              ("no candidates 90x60", (90, 60, 15, "empty"))]
    print(f"{'scene / stage':58s} {'values':>9s} {'bit for bit':>12s} {'differ':>8s} {'max':>6s}")
    bad = False
    tot = np.zeros(2, np.int64)
    for k, (title, (nx, ny, cs, kind)) in enumerate(scenes):
        sc = make_scene(rng, nx, ny, cs, kind)
        c, ms_c = run_force(f"scene{k}", sc)
        z, ms_z = run_zeit(sc)
        b, tv, te = compare(title, c, z)
        bad |= b
        tot += (tv, te)
        nvw = int((c["fit_wat"]["i"][:, 0] > 0).sum()) if len(c["fit_wat"]) else 0
        nvv = int((c["fit_veg"]["i"][:, 0] > 0).sum()) if len(c["fit_veg"]) else 0
        poly = int(((c["fit_wat"]["i"][:, 0] > 0) & (c["fit_wat"]["i"][:, 2] == 0)).sum()) if len(c["fit_wat"]) else 0
        print(f"{'':58s} candidates {len(c['ext_wat'])} water / {len(c['ext_veg'])} vegetation; valid after the fit "
              f"{nvw} / {nvv} ({poly} water with the polynomial fit); Hp {c['hp']:.2f}; FORCE {ms_c[0]:.1f} ms "
              f"extraction + {ms_c[1]:.1f} ms estimation, Zeit {ms_z[0]:.1f} + {ms_z[1]:.1f} ms")
        if k == 3:  # FORCE with 4 threads: the effect of the sums per thread and the work arrays per thread
            c4, _ = run_force(f"scene{k}_t4", sc, threads=4)
            d = 0
            for kind_ in ("wat", "veg"):
                a, bb = c["fit_" + kind_], c4["fit_" + kind_]
                d += int(((a["aod"] != bb["aod"]).any(axis=1)).sum())
                e1, e4 = c["ext_" + kind_], c4["ext_" + kind_]
                d += int(((e1["etoa"] != e4["etoa"]).any(axis=1)).sum())
            print(f"{'':58s} FORCE with 4 threads vs its 1 thread: {d} target records (environment or AOD) differ")
    print(f"{'total (scenes, bit for bit)':58s} {tot[0]:9d} {tot[1]:12d} {tot[0] - tot[1]:8d}")
    bad |= not check_elev(rng, args.scale)
    bad |= not check_aodfile(rng, args.scale)
    bad |= not check_image(rng)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
