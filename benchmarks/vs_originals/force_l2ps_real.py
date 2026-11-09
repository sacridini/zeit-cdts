"""The atmospheric correction of zeit-core (``zeit._core.l2scene``, ``zeit._level2``) against FORCE 3.11.00's ``force-l2ps``
on a real Sentinel-2 L1C scene.

The scenes (not in the repository; ``ZEIT_L2_REAL_DATA`` or ``C:\\Users\\eduar\\Documents\\dev\\data_force_l2_real``): tile 23LKC
of 2023-07-28 (processing baseline 05.09) and of 2021-07-28 (03.01), from the public Google Cloud bucket
``gcp-public-data-sentinel-2``; the DEM is Copernicus GLO-30 mosaicked over the tile (``cop_glo30_23LKC.tif``). The same DEM
file and the same digital numbers go to both sides. FORCE's GDAL (the conda env ``force-ref``) has no JPEG 2000 driver, so
both read the GeoTIFF copies ``safe_to_tiff.py`` makes of the .SAFE (same decoded values, the names ``*_Bxx.jp2`` kept).

FORCE side (``force_l2ps_real/run_force.sh``, in WSL): ``DO_REPROJ = DO_TILE = FALSE``, ``RES_MERGE = NONE``, ``DO_ATMO``, ``DO_AOD``,
``DO_TOPO`` with the DEM (bilinear, ``DEM_NODATA`` -32767), ``DO_BRDF`` FALSE or TRUE, 8 threads. Zeit gets the cloud, shadow,
snow and water bits of FORCE's own output QAI (the cloud detection is not Zeit's here) and computes the rest: the nodata, the
saturation, the sun, slope and illumination, the aerosol and vapour flags.

Three runs separate the sources of difference: ``zeit`` (Zeit's sun and the projection of rasterio), ``geo`` (FORCE's
longitude and latitude of the coarse cells, from ``sunproj``) and ``force`` (also FORCE's sun angles, from its own ``sunpos``).

    python benchmarks/vs_originals/force_l2ps_real.py --scene 2023 [--brdf] [--modes zeit,geo,force] [--n-jobs 8]

Prints, per product and band: the fraction of pixels equal, the largest and the mean absolute difference (reflectance x 10000
for the BOA), and writes maps of where they differ next to the FORCE output.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree  # noqa: F401  (before rasterio: DLL clash on Windows)
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))  # the clone's zeit, not an installed one

from force_atmo import _wsl  # noqa: E402

DATA = Path(os.environ.get("ZEIT_L2_REAL_DATA", r"C:\Users\eduar\Documents\dev\data_force_l2_real"))
SCENES = {
    "2023": "S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149",
    "2021": "S2B_MSIL1C_20210728T132239_N0301_R038_T23LKC_20210728T151918",
}
DEM = DATA / "cop_glo30_23LKC.tif"
BOA_NAMES = ["BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2"]
QAI_FIELDS = {  # name: (shift, mask)
    "off": (0, 1), "cloud": (1, 3), "shadow": (3, 1), "snow": (4, 1), "water": (5, 1), "aerosol": (6, 3),
    "subzero": (8, 1), "saturation": (9, 1), "lowsun": (10, 1), "illumination": (11, 3), "slope": (13, 1),
    "vaporfill": (14, 1),
}
FLAG_MASK = 0b111110


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}: {r.stdout}{r.stderr}")
    return r.stdout


def read(path):
    with rasterio.open(path) as d:
        return d.read()


def force_files(out_dir):
    d = next(p for p in Path(out_dir).iterdir() if p.is_dir() and p.name.startswith("T"))
    f = {}
    for p in d.glob("*.tif"):
        f[p.stem.split("_")[-1]] = p
    return f


def sunproj(scene_tif, desc, out):
    """FORCE's longitude, latitude, sun zenith and azimuth of the corners of the coarse cells (``sunproj.c``)."""
    ulx, uly, res, nx, ny = desc["geo_grid"]
    y, mo, d, hh, mm, ss = desc["date"]
    band = next((scene_tif / "GRANULE").iterdir()) / "IMG_DATA"
    band = sorted(band.glob("*_B02.jp2"))[0]
    exe = "$HOME/force-ref-work/real/sunproj"
    if not (out.exists() and out.stat().st_size == nx * ny * 24):
        sh(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_l2ps_real' / 'make_sunproj.sh')}"])
        sh(["wsl", "-e", "bash", "-lc",
            f"source ~/force-ref-work/env.sh; {exe} {_wsl(band)} {y} {mo} {d} {hh} {mm} {ss} {nx} {ulx!r} {uly!r} {res!r} {_wsl(out)}"])
    raw = out.read_bytes()
    n = nx * ny
    lon = np.frombuffer(raw, np.float64, n, 0)
    lat = np.frombuffer(raw, np.float64, n, 8 * n)
    zen = np.frombuffer(raw, np.float32, n, 16 * n)
    azi = np.frombuffer(raw, np.float32, n, 20 * n)
    return lon, lat, zen, azi


C_EXE = "$HOME/force-ref-work/real/l2_harness"
DOMAINS = ["ULTRABLUE", "BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "VAPOR", "CIRRUS",
           "SWIR1", "SWIR2"]


def scene_dict(desc, qai, toa, dem, geo, params):
    """The scene as ``force_l2.write_scene`` and ``zeit._core.atmo.level2`` take it (the layout of ``force_l2/main.c``)."""
    import datetime

    from force_l2 import FLAGS

    ulx, uly, res, gnx, gny = desc["geo_grid"]
    y, mo, d, hh, mm, ss = desc["date"]
    nr, nc = desc["nrows"], desc["ncols"]
    lon, lat, zen, azi = geo
    s2nx, s2ny, s2nodata = desc["s2"]
    sc = dict(mission=1, nx=nc, ny=nr, res=desc["res"], ulx=desc["ulx"], uly=desc["uly"], domain=DOMAINS,
              wavelength=np.asarray(desc["wavelength"], np.float32), rsr_band=np.asarray(desc["rsr_band"], np.int32), year=y, doy=datetime.date(y, mo, d).timetuple().tm_yday,
              hh=hh, mm=mm, ss=ss, tz=0, lon0=0.0, x0=0.0, kgeo=1.0, wvp=0.0, dem_nodata=-32767, dem=dem.ravel(),
              s2nx=s2nx, s2ny=s2ny, s2nodata=s2nodata, vzen=desc["vzen"], vazi=desc["vazi"],
              gn=int(np.ceil(nc * desc["res"] / 5000) * np.ceil(nr * desc["res"] / 5000)), qai=qai.ravel(), toa=toa.ravel(),
              sun=(lat, lon, zen, azi), aoi=np.zeros(0, np.uint8))
    for k in FLAGS:
        sc[k] = int(params.get(k, 0))
    j, i = np.meshgrid(np.arange(gnx), np.arange(gny))
    rec = np.zeros(gnx * gny, np.dtype([("x", "<f8"), ("y", "<f8"), ("lon", "<f8"), ("lat", "<f8")]))
    rec["x"], rec["y"] = ulx + j.ravel() * res, uly + i.ravel() * -res
    rec["lon"], rec["lat"] = lon, lat
    sc["geo_table"] = rec
    return sc


def run_c(desc, qai, toa, dem, geo, params, out_dir, threads=8):
    """FORCE's own ``radiometric_correction`` (``force_l2/main.c``: ``allocate_atc``, ``sun_target_view``,
    ``compile_topography`` and ``radiometric_correction`` of the FORCE objects, 8 OpenMP threads) on Zeit's inputs: the TOA
    and the QAI of the core, the DEM of Zeit's warp, FORCE's longitude and latitude and sun angles of the coarse cells."""
    from force_l2 import parse, write_scene

    nr, nc = desc["nrows"], desc["ncols"]
    sc = scene_dict(desc, qai, toa, dem, geo, params)
    fin, fout = out_dir / "l2_real.in.bin", out_dir / "l2_real.out.bin"
    write_scene(fin, sc)
    rec = sc["geo_table"]
    with open(fin, "ab") as f:
        f.write(np.int32(len(rec)).tobytes() + rec.tobytes())
    log = sh(["wsl", "-e", "bash", "-lc",
              f"source ~/force-ref-work/env.sh; OMP_NUM_THREADS={threads} GDAL_NUM_THREADS=1 {C_EXE} scene {_wsl(fin)} {_wsl(fout)}"])
    print("FORCE's compile_aod says:", log.strip())
    o = parse(fout.read_bytes(), sc)
    if o["status"] != 0:
        raise RuntimeError("FORCE's radiometric_correction failed on the scene")
    out = {"ms": o["ms"], "aod_scene": o["band"][:, 1], "aodmap": bool(o["aodmap"])}
    for dom, _, arr in o["products"]:
        if dom == ["QAI"]:
            out["qai"] = arr[0].reshape(nr, nc)
        elif len(dom) == 10:
            out["boa"] = arr.reshape(10, nr, nc)
        else:
            out[dom[0].lower()] = arr[0].reshape(nr, nc)
    return out


def read_dump(path):
    """The intermediate results of ``l2ps_dump.c`` (FORCE's L2PS up to the cloud detection), as memory maps."""
    hdr = np.fromfile(path, "<i4", 5)
    nx, ny, nb, gnx, gny = [int(v) for v in hdr]
    nc, gn = nx * ny, gnx * gny
    dem_stats = np.fromfile(path, "<f4", 4, offset=20)
    off = 36
    out = {"nx": nx, "ny": ny, "nb": nb, "gnx": gnx, "gny": gny, "dem_stats": dem_stats}

    def take(name, dtype, shape):
        nonlocal off
        a = np.memmap(path, dtype, "r", off, shape)
        off += a.nbytes
        out[name] = a

    take("qai_bounds", "<i2", (ny, nx))
    take("qai_sun", "<i2", (ny, nx))
    take("toa", "<i2", (nb, ny, nx))
    take("qai_topo", "<i2", (ny, nx))
    take("dem", "u1", (ny, nx))
    take("slope", "<u2", (ny, nx))
    take("aspect", "<u2", (ny, nx))
    take("ill", "<i2", (ny, nx))
    take("sky", "<u2", (ny, nx))
    take("qai_cloud", "<i2", (ny, nx))
    for k in ("sun_zen", "sun_azi", "view_zen", "view_azi"):
        take(k, "<f4", (gn,))
    take("cc", "<f4", (1,))
    take("brdf", "<f4", (nb, gn))
    return out


def diff_line(name, a, b, note=""):
    a, b = np.asarray(a), np.asarray(b)
    ne = int((a != b).sum())
    mx = int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max()) if ne else 0
    print(f"  {name:34s} {a.size:12d} values, {ne:10d} differ, max |diff| {mx}{note}")
    return ne


def stats(name, a, b, nodata=-9999):
    """Fraction equal, the largest and the mean absolute difference over the pixels valid on both sides, and the
    pixels valid on one side only."""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    va, vb = a != nodata, b != nodata
    both = va & vb
    d = np.abs(a - b)[both]
    return {
        "name": name, "n": int(both.sum()), "equal": float((d == 0).mean()) if d.size else float("nan"),
        "max": int(d.max()) if d.size else 0, "mean": float(d.mean()) if d.size else 0.0,
        "gt1": int((d > 1).sum()), "only_force": int((va & ~vb).sum()), "only_zeit": int((vb & ~va).sum()),
    }


def show(rows, title):
    print(f"\n{title}")
    print(f"{'':14}{'valid both':>12}{'equal':>10}{'max':>7}{'mean':>9}{'>1':>10}{'only FORCE':>12}{'only Zeit':>11}")
    for r in rows:
        print(f"{r['name']:14}{r['n']:12d}{r['equal'] * 100:9.4f}%{r['max']:7d}{r['mean']:9.4f}{r['gt1']:10d}{r['only_force']:12d}{r['only_zeit']:11d}")


def load_ref(force):
    """FORCE's products as arrays."""
    ref = {"boa": read(force["BOA"]), "qai": read(force["QAI"])[0]}
    for key in ("DST", "HOT", "AOD", "VZN", "WVP"):
        if key in force:
            ref[key.lower()] = read(force[key])[0]
    return ref


def compare(ref, zeit, label, maps_dir):
    res = {"label": label}
    boa_f = ref["boa"]
    rows = [stats(BOA_NAMES[b], boa_f[b], zeit["boa"][b]) for b in range(boa_f.shape[0])]
    res["boa"] = rows
    show(rows, f"BOA [{label}] (reflectance x 10000)")
    prods = []
    for key, name, nodata in [("DST", "dst", -9999), ("HOT", "hot", -9999), ("AOD", "aod", -9999), ("VZN", "vzn", -9999),
                              ("WVP", "wvp", -9999)]:
        if name in ref and name in zeit:
            prods.append(stats(key, ref[name], zeit[name], nodata))
    res["products"] = prods
    show(prods, f"DST, HOT, AOD, VZN, WVP [{label}]")
    q_f = ref["qai"].astype(np.int64)
    q_z = zeit["qai"].astype(np.int64)
    qrows = []
    for name, (s, m) in QAI_FIELDS.items():
        a, b = (q_f >> s) & m, (q_z >> s) & m
        ne = a != b
        qrows.append({"name": name, "force": int((a > 0).sum()), "zeit": int((b > 0).sum()), "differ": int(ne.sum())})
    res["qai"] = qrows
    qall = {"name": "QAI whole", "force": 0, "zeit": 0, "differ": int((q_f != q_z).sum())}
    print(f"\nQAI by field [{label}]  (a pixel that is off is 1 and nothing else in the final QAI)")
    print(f"{'':14}{'FORCE set':>12}{'Zeit set':>12}{'differ':>10}")
    for r in qrows + [qall]:
        print(f"{r['name']:14}{r['force']:12d}{r['zeit']:12d}{r['differ']:10d}")
    res["qai"].append(qall)
    maps(zeit, boa_f, q_f, q_z, label, maps_dir)
    return res


def maps(zeit, boa_f, q_f, q_z, label, maps_dir):
    """Maps of where the BOA of four bands and the QAI differ (the largest difference in blocks of 10 x 10 pixels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps_dir.mkdir(exist_ok=True, parents=True)

    def blocks(d, k=10):
        h, w = (d.shape[0] // k) * k, (d.shape[1] // k) * k
        return d[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3))

    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    for a, b in zip(ax.ravel(), [0, 2, 7, 9]):
        d = np.abs(boa_f[b].astype(np.int32) - zeit["boa"][b].astype(np.int32))
        d[(boa_f[b] == -9999) | (zeit["boa"][b] == -9999)] = 0
        im = a.imshow(blocks(d), cmap="magma", vmin=0, vmax=max(2, np.percentile(d, 99.9)))
        a.set_title(f"|BOA {BOA_NAMES[b]} FORCE - Zeit| max per 10x10 [{label}]")
        fig.colorbar(im, ax=a, shrink=0.7)
    d = (q_f != q_z).astype(np.uint8)
    ax.ravel()[4].imshow(blocks(d), cmap="gray_r")
    ax.ravel()[4].set_title("QAI differs")
    ax.ravel()[5].imshow(blocks(((q_f & FLAG_MASK) > 0).astype(np.uint8)), cmap="gray_r")
    ax.ravel()[5].set_title("FORCE flags (cloud, shadow, snow, water)")
    fig.tight_layout()
    fig.savefig(maps_dir / f"diff_{label}.png", dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=sorted(SCENES), default="2023")
    ap.add_argument("--brdf", action="store_true")
    ap.add_argument("--modes", default="zeit,geo,force")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--save", action="store_true", help="keep Zeit's products (the QAI, AOD, WVP, HOT, DST, VZN and four BOA bands) in the FORCE output folder")
    a = ap.parse_args()

    from common import no_power_throttling

    no_power_throttling()
    from zeit import _core, _level2

    print("zeit._core:", _core.__file__)
    name = SCENES[a.scene]
    tag = "brdf" if a.brdf else "nobrdf"
    force_dir = DATA / f"force_out_{a.scene}_{tag}"
    force = force_files(force_dir)
    ref = load_ref(force)
    safe = _level2.Safe(DATA / "tif" / f"{name}.SAFE")
    t0 = time.perf_counter()
    desc = safe.describe()
    dn = _level2.read_bands(safe, desc)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    dem = _level2.dem_on_grid(DEM, safe)
    t_dem = time.perf_counter() - t0
    print(f"read the 13 bands {t_read:.1f} s, DEM warp {t_dem:.1f} s; scene {desc['ncols']} x {desc['nrows']}, baseline {desc['baseline']}")
    ref_dem = DATA / "dem_gdalwarp_23LKC.tif"  # gdalwarp of GDAL 3.9.3 with FORCE's options (force_l2ps_real/gdalwarp_dem.sh)
    if ref_dem.exists():
        g = read(ref_dem)[0]
        d = np.abs(g.astype(np.float64) - dem)
        print(f"DEM, Zeit's warp against gdalwarp: equal in {(g == dem).mean() * 100:.4f}% of the pixels, max |diff| {d.max():.6g} m")
    qf = read(force["QAI"])[0]
    flags = (qf & FLAG_MASK).astype(np.int16)
    flags[(qf & 1) > 0] = 0
    lon_z, lat_z = _level2.geo_tables(safe, desc)
    lon_f, lat_f, zen_f, azi_f = sunproj(DATA / "tif" / f"{name}.SAFE", desc, DATA / f"sunproj_{a.scene}.bin")
    print("max |lon, lat| FORCE - rasterio at the coarse corners:", np.abs(lon_f - lon_z).max(), np.abs(lat_f - lat_z).max())
    results = []
    for mode in a.modes.split(","):
        if mode == "c":
            # FORCE's C functions on Zeit's TOA, QAI and DEM, with FORCE's geography and sun
            sh(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_l2' / 'make_harness.sh')} $HOME/force-src $HOME/force-ref-work/real"])
            qai0, toa = _level2.prepare(safe, desc, dn, a.n_jobs)
            qai0 = qai0 | flags
            params = dict(domulti=1, doenv=1, doatmo=1, dotopo=1, dobrdf=int(a.brdf), doaod=1, odst=1, ovzn=1, ohot=1, oaod=1, owvp=1)
            t0 = time.perf_counter()
            out = run_c(desc, qai0, toa, dem, (lon_f, lat_f, zen_f, azi_f), params, DATA)
            print(f"\n===== mode c (FORCE's functions on Zeit's inputs): {out['ms'] / 1e3:.1f} s in radiometric_correction, "
                  f"call {time.perf_counter() - t0:.1f} s, AOD green {out['aod_scene'][2]:.4f}, aodmap {out['aodmap']} =====")
            np.savez(force_dir / f"c_{tag}.npz", **{k: out[k] for k in ("qai", "aod", "wvp", "hot", "dst", "vzn") if k in out})
            r = compare(ref, out, f"{a.scene}_{tag}_c", force_dir / "maps")
            r.update(mode=mode, core_s=out["ms"] / 1e3)
            results.append(r)
            continue
        if mode == "dump":
            # FORCE's L2PS up to the cloud detection (l2ps_dump) against the core, stage by stage; then the core with the
            # flags FORCE had at the time of the correction (the dump's), geography and sun of FORCE
            dmp = read_dump(DATA / f"l2ps_dump_{a.scene}_{tag}.bin")
            qai0, toa = _level2.prepare(safe, desc, dn, a.n_jobs)
            print("\nFORCE's L2PS up to the cloud detection against the core")
            diff_line("QAI after bounds_level1", dmp["qai_bounds"], qai0)
            for b_ in range(dmp["nb"]):
                diff_line(f"TOA band {b_} ({DOMAINS[b_]})", dmp["toa"][b_], toa[b_])
            fl = (np.asarray(dmp["qai_cloud"]) & FLAG_MASK).astype(np.int16)
            print("  cloud flags of the dump against the final QAI of FORCE:", int((fl != (qf & FLAG_MASK)).sum()), "pixels differ")
            sun_zen_c = np.degrees(dmp["sun_zen"]).astype(np.float32)
            out = _level2._level2_arrays(safe, flags=fl, dobrdf=a.brdf, n_jobs=a.n_jobs, bands=dn, dem_array=dem,
                                 geography=(lon_f, lat_f), sun=(zen_f, azi_f), brdf_table=np.asarray(dmp["brdf"]))
            print(f"  valid targets (water, shadow, vegetation): {list(out['n_targets'])}  (the dump run's log: see l2ps log)")
            diff_line("QAI after the topography and the flags", dmp["qai_cloud"], out["qai_in"].reshape(dmp["qai_cloud"].shape),
                      " (the dump's QAI is after the cloud detection; the core's has the dump's flags)")
            qt = out["qai_in"].reshape(dmp["qai_cloud"].shape)
            for name, (sh_, m_) in QAI_FIELDS.items():
                if name in ("aerosol", "subzero", "vaporfill"):
                    continue
                diff_line(f"  QAI field {name}", (np.asarray(dmp["qai_cloud"]) >> sh_) & m_, (qt >> sh_) & m_)
            diff_line("binned DEM", dmp["dem"], out["top_dem"].reshape(dmp["dem"].shape))
            diff_line("slope", dmp["slope"], out["top_slope"].reshape(dmp["slope"].shape))
            diff_line("aspect", dmp["aspect"], out["top_aspect"].reshape(dmp["aspect"].shape))
            diff_line("illumination", dmp["ill"], out["top_ill"].reshape(dmp["ill"].shape))
            diff_line("sky view", dmp["sky"], out["top_sky"].reshape(dmp["sky"].shape))
            print("  DEM statistics FORCE:", np.asarray(dmp["dem_stats"]), " core:", np.array(out["dem_stats"]))
            print("  sun zenith (rad) max |FORCE - core|:", float(np.abs(dmp["sun_zen"] - np.radians(out["sun_zen"])).max()))
            print("  view zenith (rad) max |FORCE - core|:", float(np.abs(np.asarray(dmp["view_zen"]) - out["view_zen"]).max()))
            r = compare(ref, out, f"{a.scene}_{tag}_dump", force_dir / "maps")
            r.update(mode=mode, n_targets=list(out["n_targets"]))
            results.append(r)
            continue
        if mode == "port":
            # FORCE's functions against the port, on the same inputs, field by field (the comparison of force_l2.py)
            from force_atmo import Table
            from force_l2 import compare as compare_fields
            from force_l2 import parse, write_scene

            sh(["wsl", "-e", "bash", "-lc", f"bash {_wsl(HERE / 'force_l2' / 'make_harness.sh')} $HOME/force-src $HOME/force-ref-work/real"])
            qai0, toa = _level2.prepare(safe, desc, dn, a.n_jobs)
            qai0 = qai0 | flags
            params = dict(domulti=1, doenv=1, doatmo=1, dotopo=1, dobrdf=0, doaod=1, odst=1, ovzn=1, ohot=1, oaod=1, owvp=1)
            sc = scene_dict(desc, qai0, toa, dem, (lon_f, lat_f, zen_f, azi_f), params)
            fin, fout = DATA / "port.in.bin", DATA / "port.out.bin"
            write_scene(fin, sc)
            with open(fin, "ab") as f:
                f.write(np.int32(len(sc["geo_table"])).tobytes() + sc["geo_table"].tobytes())
            log = sh(["wsl", "-e", "bash", "-lc", f"source ~/force-ref-work/env.sh; OMP_NUM_THREADS={a.n_jobs} GDAL_NUM_THREADS=1 {C_EXE} scene {_wsl(fin)} {_wsl(fout)}"])
            print("FORCE's compile_aod says:", log.strip())
            c = parse(fout.read_bytes(), sc)
            d = {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in sc.items() if k not in ("sun", "geo_table")}
            ulx, uly, res, gnx, gny = desc["geo_grid"]
            d.update(n_jobs=a.n_jobs, geo_n=gnx, geo_res=float(res), geo_lon=np.ascontiguousarray(lon_f), geo_lat=np.ascontiguousarray(lat_f),
                     sun_zen=np.ascontiguousarray(zen_f), sun_azi=np.ascontiguousarray(azi_f))
            raw, _ = _core.atmo.level2(d)
            z = parse(bytes(raw), sc)
            print(f"{'scene':44s} {'field':16s} {'values':>9s} {'equal':>9s} {'1 ulp':>6s} {'more':>6s} {'max':>5s}")
            table = Table()
            compare_fields(table, f"{a.scene} {tag}, C vs port", sc, c, z)
            print(f"FORCE {c['ms']:.0f} ms, Zeit {z['ms']:.0f} ms ({a.n_jobs} threads each)")
            continue
        kw = {}
        if mode in ("geo", "force"):
            kw["geography"] = (lon_f, lat_f)
        if mode == "force":
            kw["sun"] = (zen_f, azi_f)
        t0 = time.perf_counter()
        out = _level2._level2_arrays(safe, flags=flags, dobrdf=a.brdf, n_jobs=a.n_jobs, bands=dn, dem_array=dem, **kw)
        wall = time.perf_counter() - t0
        print(f"\n===== mode {mode}: core {out['ms'] / 1e3:.1f} s (call {wall:.1f} s), "
              f"AOD green {out['aod_scene'][2]:.4f}, aodmap {out['aodmap']} =====")
        if mode in ("zeit", "geo"):
            dz = np.abs(out["sun_zen"] - zen_f[: out["sun_zen"].size]).max()
            print(f"sun zenith Zeit - FORCE at the coarse cells: max {dz:.5f} deg")
        print("valid targets (water, shadow, vegetation):", out["n_targets"], "- FORCE prints water/vegetation in its log")
        dump_path = DATA / f"l2ps_dump_{a.scene}_{tag}.bin"
        if dump_path.exists() and "xy_brdf" in out:
            nb_ = 13
            fb = np.fromfile(dump_path, "<f4", nb_ * len(out["sun_zen"]), offset=dump_path.stat().st_size - nb_ * len(out["sun_zen"]) * 4)
            zb = np.asarray(out["xy_brdf"]).reshape(nb_, -1)
            fb = fb.reshape(nb_, -1)
            print("BRDF factor, Zeit's NBAR against FORCE's brdf_factor, max |diff| per band:",
                  np.round(np.abs(fb - zb).max(axis=1), 4).tolist(), " FORCE's range", [round(float(fb[1:9].min()), 4), round(float(fb[1:9].max()), 4)])
        if a.save:
            keep = {k: out[k] for k in ("qai", "aod", "wvp", "hot", "dst", "vzn", "xy_aod", "xy_interp", "xy_dem", "xy_brdf") if k in out}
            keep.update({f"boa{b}": out["boa"][b] for b in (0, 2, 7, 9)})
            np.savez(force_dir / f"zeit_{mode}.npz", **keep)
        r = compare(ref, out, f"{a.scene}_{tag}_{mode}", force_dir / "maps")
        r.update(mode=mode, core_s=out["ms"] / 1e3, aod_green=float(out["aod_scene"][2]),
                 n_targets=list(out["n_targets"]))
        results.append(r)
    (force_dir / f"result_{a.scene}_{tag}.json").write_text(json.dumps(results, indent=1))
    print(f"\nread {t_read:.1f} s, DEM {t_dem:.1f} s")


if __name__ == "__main__":
    main()
