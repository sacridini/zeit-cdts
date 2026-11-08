"""The real-data parts of pansharpen_wald.py: a CBERS-4A WPM window, GDAL as an oracle, Landsat 8 band 8.

A WPM window (written outside the repository by a script that reads the BDC's public GeoTIFFs by range requests, the
BDC's files are strips, not COGs, so a window costs its rows) is a folder with ``pan.tif`` (2 m), ``ms_band1.tif`` to
``ms_band4.tif`` (8 m: blue, green, red, NIR) and ``inpe_pca.tif`` (the INPE's CB4A-WPM-PCA-FUSED-1 product of the
same scene and window, RGB, 2 m, 1-255).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr

import zeit
from pansharpen_wald import METHODS, block_mean, grid_da, metrics, row, wald

NAMES = ["blue", "green", "red", "nir"]


def read(path):
    da = rioxarray.open_rasterio(path, masked=False).astype("float32")
    return da


def load(folder):
    folder = Path(folder)
    pan = read(folder / "pan.tif").isel(band=0, drop=True)
    ms = xr.concat([read(folder / f"ms_band{b}.tif").isel(band=0, drop=True) for b in (1, 2, 3, 4)], dim="band")
    ms = ms.assign_coords(band=NAMES).rio.write_crs(pan.rio.crs)
    return ms, pan


def run_real(folder):
    ms, pan = load(folder)
    print(f"ms {ms.shape} {ms.rio.resolution()}, pan {pan.shape} {pan.rio.resolution()}")
    print("band means", ms.mean(dim=("y", "x")).values.round(1), "pan", float(pan.mean()))

    # --- the pan's offset to the bands: the pan reduced to 8 m against each band, then at 2 m
    pan8 = pan.coarsen(y=4, x=4).mean().assign_coords(y=ms.y, x=ms.x).rio.write_crs(ms.rio.crs)
    print("\noffset of the bands to the pan (zeit.coregister, phase correlation; + is right / down, pixels)")
    for b in NAMES:
        try:
            _, sh = zeit.coregister(ms.sel(band=[b]), pan8, band=b, window_size=512, max_shift=3)
            r = sh.iloc[0]
            print(f"  {b:6s} at 8 m: x {r.x_shift_px:+.3f} y {r.y_shift_px:+.3f}  (reliability {r.reliability:.0f}, ok {r.success})")
        except Exception as e:                                      # noqa: BLE001
            print(f"  {b:6s} at 8 m: {type(e).__name__}: {e}")
    nir2 = zeit.pansharpen(ms, pan, "hpf", gain=0.0).sel(band=["nir"])
    for b in ("nir", "green"):
        cube = zeit.pansharpen(ms, pan, "hpf", gain=0.0).sel(band=[b])
        try:
            _, sh = zeit.coregister(cube, pan, band=b, window_size=1024, max_shift=12)
            r = sh.iloc[0]
            print(f"  {b:6s} at 2 m: x {r.x_shift_px:+.3f} y {r.y_shift_px:+.3f}  (reliability {r.reliability:.0f}, ok {r.success})")
        except Exception as e:                                      # noqa: BLE001
            print(f"  {b:6s} at 2 m: {type(e).__name__}: {e}")
    del nir2

    # --- Wald's protocol at ratio 4
    print("\nWald's protocol (ms 32 m + pan 8 m fused to 8 m, against the 8 m bands)")
    ms_arr = ms.values.astype(np.float64)
    pan_arr = pan.values.astype(np.float64)
    res = wald(ms_arr, pan_arr, 4, extras=True, names=NAMES)
    for k, v in res.items():
        print(row(k, v))
    # the Brovey of equal weights (a gain between pan and bands is not absorbed) and the GSA on the visible bands only
    res2 = wald(ms_arr, pan_arr, 4, methods=("brovey",), names=NAMES, weights=[0.25] * 4)
    print(row("brovey=", res2["brovey"]))
    res3 = wald(ms_arr, pan_arr, 4, methods=("gsa", "brovey", "pca"), names=NAMES, intensity_bands=["green", "red", "nir"])
    for k in ("gsa", "brovey", "pca"):
        print(row(k + "/grn-red-nir", res3[k]))
    hpf = {g: wald(ms_arr, pan_arr, 4, methods=("hpf",), names=NAMES, gain=g)["hpf"] for g in (0.5, 0.75)}
    for g, m in hpf.items():
        print(row(f"hpf g{g}", m))

    # --- the INPE's PCA product
    with rasterio.open(Path(folder) / "inpe_pca.tif") as s:
        inpe = s.read().astype(np.float64)
    print("\ncorrelation with the INPE's CB4A-WPM-PCA-FUSED-1 (R, G, B; 2 m), where the product has data")
    ok = (inpe > 0).all(axis=0)
    print(f"  data in {ok.mean() * 100:.1f}% of the window")
    rgb = ["red", "green", "blue"]
    for name, kw in [("cubic", {"method": "hpf", "gain": 0.0}), ("pca (4 bands)", {"method": "pca"}),
                     ("pca (r,g,b)", {"method": "pca", "intensity_bands": rgb}), ("gsa", {"method": "gsa"}),
                     ("brovey", {"method": "brovey"}), ("hpf", {"method": "hpf"}), ("pan", None)]:
        if kw is None:
            out = np.broadcast_to(pan_arr, (3,) + pan_arr.shape)
        else:
            out = zeit.pansharpen(ms, pan, **kw).sel(band=rgb).values
        good = ok & np.isfinite(out).all(axis=0)
        cc = [np.corrcoef(out[i][good], inpe[i][good])[0, 1] for i in range(3)]
        print(f"  {name:14s} CC(R,G,B) " + " ".join(f"{c:.4f}" for c in cc))
    # a pca of the 3 visible bands, ms alone (no pan) as to show what the product's colour is
    vis = ms.sel(band=rgb)
    up = zeit.pansharpen(vis, pan, "hpf", gain=0.0).values
    print("  (the bands' cubic resampling is the 'cubic' line)")
    # the best integer shift of the INPE product against zeit's pca, to see whether it is registered to the pan
    ref = zeit.pansharpen(ms, pan, "pca", intensity_bands=rgb).sel(band="red").values
    inner = (slice(20, -20), slice(20, -20))
    fin = np.isfinite(ref[inner])
    best = max(((dy, dx, np.corrcoef(ref[inner][fin], np.roll(inpe[0], (dy, dx), (0, 1))[inner][fin])[0, 1])
                for dy in range(-3, 4) for dx in range(-3, 4)), key=lambda t: t[2])
    print(f"  INPE red against zeit pca red, best roll (dy, dx) = ({best[0]}, {best[1]}), CC {best[2]:.4f}")
    del up


def run_gdal(folder):
    """GDAL's weighted Brovey (`gdal raster pansharpen`, GDAL 3.12.1, the program form of gdal_pansharpen) against zeit's "brovey" with the
    same weights and input, in the pixels away from the borders; and what the resampling is."""
    folder = Path(folder)
    ms, pan = load(folder)
    ny, nx = pan.shape
    weights = [0.25, 0.25, 0.25, 0.25]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Float32 copies, so that GDAL does not round its output to the integers of the inputs
        files = {}
        for name, src in [("pan", folder / "pan.tif")] + [(f"b{b}", folder / f"ms_band{b}.tif") for b in (1, 2, 3, 4)]:
            files[name] = tmp / f"{name}_f32.tif"
            subprocess.run(["gdal_translate", "-q", "-ot", "Float32", str(src), str(files[name])], check=True)
        out = tmp / "gdal_out.tif"
        # `gdal raster pansharpen` is the program form of gdal_pansharpen (weighted Brovey, cubic by default)
        subprocess.run(["gdal", "raster", "pansharpen", "-q", "--overwrite", "-j", "4", "-r", "cubic",
                        *[f"--weights={w}" for w in weights], str(files["pan"]),
                        *[str(files[f"b{b}"]) for b in (1, 2, 3, 4)], str(out)], check=True)
        with rasterio.open(out) as v:
            g = v.read().astype(np.float64)
        # GDAL's own cubic resampling of the bands to the pan's grid (gdal_translate -outsize -r cubic)
        up = []
        for b in (1, 2, 3, 4):
            dst = tmp / f"up{b}.tif"
            subprocess.run(["gdal_translate", "-q", "-outsize", str(nx), str(ny), "-r", "cubic", "-ot", "Float32",
                            str(folder / f"ms_band{b}.tif"), str(dst)], check=True)
            with rasterio.open(dst) as s_:
                up.append(s_.read(1).astype(np.float64))
        up = np.stack(up)
    z_up = zeit.pansharpen(ms, pan, "hpf", gain=0.0).values.astype(np.float64)
    z = zeit.pansharpen(ms, pan, "brovey", weights=weights).values.astype(np.float64)
    pan_arr = pan.values.astype(np.float64)
    inner = (slice(None), slice(16, -16), slice(16, -16))

    def diff(name, a, b):
        ok = np.isfinite(a[inner]) & np.isfinite(b[inner])
        d = np.abs(a[inner] - b[inner])[ok]
        print(f"  {name:55s} mean |d| {d.mean():9.4f}  max {d.max():10.3f}  p99.9 {np.quantile(d, 0.999):9.3f}  "
              f"CC {np.corrcoef(a[inner][ok], b[inner][ok])[0, 1]:.6f}  ({ok.mean() * 100:.2f}% of pixels)")

    print("GDAL's weighted Brovey (`gdal raster pansharpen`, cubic) against zeit.pansharpen(..., 'brovey', weights)")
    print("  GDAL 3.12.1; interior pixels (16 px off the borders); values in the units of the data (DN)")
    print(f"  zeit's NaN pixels: {int(np.isnan(z).any(axis=0).sum())} of {z.shape[1] * z.shape[2]}; "
          f"the bands' mean: {np.nanmean(up, axis=(1, 2)).round(1).tolist()}")
    diff("cubic resampling: zeit's warp vs gdal_translate", z_up, up)
    diff("Brovey: zeit vs GDAL (each its own resampling)", z, g)
    # zeit's Brovey on GDAL's resampled bands: what is left is the formula
    from zeit import _core
    fed, _, _ = _core.pansharpen.pansharpen_batch(up.astype(np.float32), pan_arr.astype(np.float32), "brovey", 4.0,
                                                  weights, None, n_jobs=4)
    diff("Brovey: zeit core on GDAL's resampling vs GDAL", fed.astype(np.float64), g)
    ref = up * (pan_arr / (up * np.array(weights)[:, None, None]).sum(axis=0))
    diff("Brovey: numpy's formula on GDAL's resampling vs GDAL", ref, g)
    diff("Brovey: numpy's formula on GDAL's resampling vs zeit core", ref, fed.astype(np.float64))


def main(mode, args):
    if mode == "real":
        run_real(args[0])
    elif mode == "gdal":
        run_gdal(args[0])
    elif mode == "landsat":
        run_landsat(args[0])
    else:
        raise SystemExit(f"unknown mode {mode!r}")


def run_landsat(folder):
    """Landsat 8 L1 (gcp-public-data-landsat, public): band 8 (15 m) with bands 2-6 (30 m), a 400 x 400 window of
    the bands. Landsat's pan grid is shifted by half a pan pixel from the 2 x 2 blocks of the bands (the centres of
    the two grids coincide), so this is also the case of corners that do not coincide."""
    folder = Path(folder)
    names = ["B2", "B3", "B4", "B5", "B6"]
    ms = xr.concat([read(folder / f"ms_{n}.tif").isel(band=0, drop=True) for n in names], dim="band")
    pan = read(folder / "pan_B8.tif").isel(band=0, drop=True)
    ms = ms.assign_coords(band=names).rio.write_crs(pan.rio.crs)
    tm, tp = ms.rio.transform(), pan.rio.transform()
    print(f"ms {ms.shape} cells {tm.a:g}, corner ({tm.c:g}, {tm.f:g}); pan {pan.shape} cells {tp.a:g}, corner "
          f"({tp.c:g}, {tp.f:g}): the pan's corner is {(tp.c - tm.c) / tm.a:+.3f} band cells east of the bands'")
    full = zeit.pansharpen(ms, pan, "gsa")
    print(f"gsa output {full.shape}, NaN {int(np.isnan(full.values).sum())} values (the pan's edge outside the bands)")
    ms_arr, pan_arr = ms.values.astype(np.float64), pan.values.astype(np.float64)
    print("Wald's protocol at ratio 2 (the pan cropped to 2 x the bands; the half-pixel offset of the real grids is "
          "not kept by the degradation)")
    res = wald(ms_arr, pan_arr, 2, extras=True, names=names)
    for k, v in res.items():
        print(row(k, v))
    # the pan (0.50-0.68 um) sees the green and the red, not the NIR and the SWIR
    res = wald(ms_arr, pan_arr, 2, methods=("gsa", "brovey", "pca"), names=names, intensity_bands=["B3", "B4"])
    for k in ("gsa", "brovey", "pca"):
        print(row(k + "/B3,B4", res[k]))
    from zeit import _core
    print("weights of the intensity (gsa on the whole window, all bands | B3,B4 only), brovey fitted")
    fine = zeit.pansharpen(ms, pan, "hpf", gain=0.0).values
    for kw in ({}, {"select": [1, 2]}):
        for m in ("gsa", "brovey"):
            _, w, g = _core.pansharpen.pansharpen_batch(np.nan_to_num(fine), np.nan_to_num(pan_arr[:fine.shape[1], :fine.shape[2]]).astype(np.float32),
                                                        m, 2.0, None, kw.get("select"), n_jobs=4)
            print(f"  {m:7s} {kw or 'all'}: w {np.round(w, 3).tolist()}  g {np.round(g, 3).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
