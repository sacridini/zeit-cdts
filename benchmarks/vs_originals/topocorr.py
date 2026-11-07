"""zeit.topographic_correction against the usual tools, on synthetic terrain (no DEM ships with the repository):

* Horn's slope and aspect (``zeit._topocorr.slope_aspect``) against ``gdaldem slope`` / ``gdaldem aspect`` (GDAL, with
  ``-compute_edges``, which replicates the edge cells as zeit does) and against ``terra::terrain`` (R);
* the C-correction against RStoolbox's ``topCor(method = "C")`` (RStoolbox 1.0.2.2, GPL (>= 3)), given zeit's slope
  and aspect so that only the correction is compared. RStoolbox fits the line on a regular sample of at most 10000
  pixels of the whole image (it samples ``min(ncell, 10000)`` cells and, for ``method = "C"``, keeps flat ones and
  those facing away from the sun), zeit on every pixel with a slope of at least ``min_slope``; the formula
  ``rho (cos sz + c) / (cos i + c)``, ``c = b / m``, is the same. With a 100 x 100 image (the whole of it is the
  sample) and ``min_slope=0`` the two are the same computation. With 1000 x 1000 they differ by the sampling.

    python benchmarks/vs_originals/topocorr.py            # GDAL_BIN=...\\Library\\bin if gdaldem is not on the PATH
"""
import os
import subprocess

import numpy as np
import xarray as xr

import zeit  # first: it sets up the DLL search path that rasterio needs on Windows
import rioxarray  # noqa: E402,F401
from common import best_of, no_power_throttling, record, run_r, workdir
from zeit._topocorr import slope_aspect

GDAL_BIN = os.environ.get("GDAL_BIN", r"C:\Users\eduar\miniforge3\envs\geo\Library\bin")
SZ, SAZ = 38.0, 140.0       # the sun, degrees
RES = 30.0


def terrain(side, seed=11):
    """Hills of 1 to 6 km with some finer relief, slopes mostly 0-35 degrees, as float32 (a DEM's precision)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side] * RES
    z = np.zeros((side, side))
    for _ in range(14):
        k = rng.uniform(2 * np.pi / 6000, 2 * np.pi / 1000)
        th = rng.uniform(0, 2 * np.pi)
        z += rng.uniform(15, 60) * np.sin(k * (np.cos(th) * xx + np.sin(th) * yy) + rng.uniform(0, 6.3))
    fine = rng.normal(0, 1, (side, side))
    for _ in range(3):
        fine = (fine + np.roll(fine, 1, 0) + np.roll(fine, 1, 1) + np.roll(fine, (1, 1), (0, 1))) / 4
    z += 40 * fine / fine.std() * 0.3 + 800
    grid = {"y": (np.arange(side)[::-1] * RES + 15.0), "x": np.arange(side) * RES + 15.0}
    return xr.DataArray(z.astype(np.float32), dims=("y", "x"), coords=grid).rio.write_crs(32722)


def smooth_field(side, rng, passes=12):
    f = rng.normal(0, 1, (side, side))
    for _ in range(passes):
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5
    return (f - f.min()) / (f.max() - f.min())


def scene(dem, sa, seed=5):
    """Six bands: land cover (a smooth field mixing vegetation and soil) times the light of the relief, with a
    diffuse part (the C-correction's offset), and noise."""
    side = dem.shape[0]
    rng = np.random.default_rng(seed)
    veg = smooth_field(side, rng)
    soil = np.array([0.08, 0.12, 0.16, 0.22, 0.30, 0.26])
    green = np.array([0.03, 0.06, 0.04, 0.45, 0.22, 0.10])
    s, a = np.radians(sa.slope.values), np.radians(sa.aspect.values)
    sz, saz = np.radians(SZ), np.radians(SAZ)
    cosi = np.cos(sz) * np.cos(s) + np.sin(sz) * np.sin(s) * np.cos(saz - a)
    light = 0.7 * cosi / np.cos(sz) + 0.3
    bands = [(veg * green[b] + (1 - veg) * soil[b]) * light + rng.normal(0, 0.004, (side, side)) for b in range(6)]
    return np.stack(bands), cosi


def gdal(d, what, extra=()):
    env = {**os.environ, "PATH": GDAL_BIN + os.pathsep + os.environ["PATH"], "GDAL_NUM_THREADS": "1"}
    out = d / f"gdal_{what}.tif"
    subprocess.run([os.path.join(GDAL_BIN, "gdaldem.exe"), what, str(d / "dem.tif"), str(out), "-compute_edges", *extra],
                   check=True, env=env, stdout=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    return rioxarray.open_rasterio(out, masked=True).squeeze(drop=True).values.astype(np.float64)


def circular(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def main():
    no_power_throttling()
    # -- slope and aspect -------------------------------------------------------------------------------------
    side = 1000
    d = workdir("topocorr")
    dem = terrain(side)
    dem.rio.to_raster(d / "dem.tif")
    dem = rioxarray.open_rasterio(d / "dem.tif").squeeze(drop=True)
    sa = slope_aspect(dem)
    zs, za = sa.slope.values.astype(np.float64), sa.aspect.values.astype(np.float64)
    print(f"terrain {side}x{side}, {RES:.0f} m; slope up to {zs.max():.1f} degrees, median {np.median(zs):.1f}")
    gs, ga = gdal(d, "slope"), gdal(d, "aspect")
    ok = np.isfinite(ga)
    edge = np.ones((side, side), bool)
    edge[1:-1, 1:-1] = False
    print(f"  vs gdaldem (-compute_edges; GDAL writes float32): slope max |diff| {np.nanmax(np.abs(zs - gs)[~edge]):.2e} degrees "
          f"inside, {np.nanmax(np.abs(zs - gs)[edge]):.2e} on the edge cells; aspect max circular |diff| "
          f"{np.nanmax(circular(za, ga)[ok & ~edge]):.2e} inside, {np.nanmax(circular(za, ga)[ok & edge]):.2e} on the edge "
          f"({int((~ok).sum())} flat cells NoData in GDAL)")
    t_r = run_r("topocorr.R", d, "terrain")
    ts = np.fromfile(d / "terra_slope.bin").reshape(side, side)
    ta = np.fromfile(d / "terra_aspect.bin").reshape(side, side)
    inner = (slice(1, -1), slice(1, -1))
    okt = np.isfinite(ta[inner])
    print(f"  vs terra::terrain (interior, its edges are NA): slope max |diff| {np.nanmax(np.abs(zs - ts)[inner]):.2e} degrees;"
          f" aspect max circular |diff| {np.nanmax(circular(za, ta)[inner][okt]):.2e} degrees")
    t_z = best_of(lambda: slope_aspect(dem, n_jobs=1))
    record("slope and aspect (Horn)", "terra 1.x terrain (R)", f"{side}x{side} px", t_r, t_z, 1,
           f"max diff to gdaldem {np.nanmax(np.abs(zs - gs)[~edge]):.1e} deg (slope, inside)")

    # -- the C-correction -------------------------------------------------------------------------------------
    for side, min_slope, label in ((100, 0.0, "whole image is the sample"), (1000, 0.0, "sampled by RStoolbox")):
        d = workdir(f"topocorr{side}")
        dem = terrain(side, seed=3)
        dem.rio.to_raster(d / "dem.tif")
        dem = rioxarray.open_rasterio(d / "dem.tif").squeeze(drop=True)
        sa = slope_aspect(dem)
        bands, _ = scene(dem, sa)
        cube = xr.DataArray(bands.astype(np.float32), dims=("band", "y", "x"),
                            coords={"band": [f"b{i}" for i in range(6)], "y": dem.y.values, "x": dem.x.values}) \
            .rio.write_crs(32722)
        bands.reshape(6, -1).T.astype(np.float64).tofile(d / "img.bin")
        np.stack([np.radians(sa.slope.values).ravel(), np.radians(sa.aspect.values).ravel()], 1).astype(np.float64).tofile(d / "sa.bin")
        t_r = run_r("topocorr.R", d, "topcor", np.radians(SAZ), np.radians(SZ))
        theirs = np.fromfile(d / "topcor_given.bin").reshape(side, side, 6).transpose(2, 0, 1)
        out = {}
        t_z = best_of(lambda: out.update(z=zeit.topographic_correction(
            cube, dem, sun_zenith=SZ, sun_azimuth=SAZ, min_slope=min_slope, n_jobs=1)))
        mine = out["z"]
        diff = np.abs(mine.values.astype(np.float64) - theirs)
        rel = diff / np.abs(theirs)
        before = np.abs(cube.values - theirs).max()
        print(f"C-correction {side}x{side} ({label}): zeit c = {np.array2string(mine.topo_c.values, precision=4)}")
        print(f"  vs RStoolbox topCor, slope/aspect given: max |diff| {diff.max():.2e}, max relative {rel.max():.2e}, "
              f"mean |diff| {diff.mean():.2e} (the correction changes the image by {before:.2e} at most)")
        if side == 100:
            assert rel.max() < 2e-5, rel.max()
        record("topographic correction (C)", "RStoolbox 1.0.2.2 topCor (R)", f"{side}x{side} px x 6 bands", t_r, t_z, 1,
               f"max rel diff {rel.max():.1e}; {label}; slope/aspect given")
        if side == 100:
            run_r("topocorr.R", d, "topcor", np.radians(SAZ), np.radians(SZ), "own")
            own = np.fromfile(d / "topcor_own.bin").reshape(side, side, 6).transpose(2, 0, 1)
            fin = np.isfinite(own)
            dd = np.abs(mine.values - own)[fin]
            print(f"  against its own terra slope and aspect (NA on the edge, {int((~fin).sum() / 6)} pixels): "
                  f"max |diff| {dd.max():.2e}, mean {dd.mean():.2e}")

    # -- timing: 2000 x 2000 x 6 ------------------------------------------------------------------------------
    side = 2000
    dem = terrain(side, seed=9)
    sa = slope_aspect(dem)
    bands, _ = scene(dem, sa)
    cube = xr.DataArray(bands.astype(np.float32), dims=("band", "y", "x"),
                        coords={"band": [f"b{i}" for i in range(6)], "y": dem.y.values, "x": dem.x.values}).rio.write_crs(32722)
    print(f"{side}x{side} x 6 bands, one date (the whole call: slope and aspect, illumination, six regressions, correction)")
    for threads in (1, 4):
        t = best_of(lambda: zeit.topographic_correction(cube, dem, sun_zenith=SZ, sun_azimuth=SAZ, n_jobs=threads))
        print(f"  C, {threads} thread(s): {t:.3f} s")
    t = best_of(lambda: zeit.topographic_correction(cube, dem, sun_zenith=SZ, sun_azimuth=SAZ, method="scs+c", n_jobs=4))
    print(f"  SCS+C, 4 threads: {t:.3f} s")


if __name__ == "__main__":
    main()
