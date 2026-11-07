"""Runs AROSICS' global COREG on pairs of GeoTIFFs; the half of ``arosics.py`` that lives in the ``arosics-ref`` conda
environment (AROSICS 1.14.0 and its GDAL, Python 3.14), which has none of Zeit's packages.

    python arosics_driver.py run <spec.json> <result.json>
    python arosics_driver.py natural <folder>          # writes a pair of a natural image (scikit-image's camera)

``spec.json``: ``{"pairs": [{"name", "ref", "tgt", "nodata", "ws", "cpus", "aligned": "<npy to write>"}], "repeat"}``.
Each pair is timed as ``COREG(...)`` plus ``calculate_spatial_shifts()`` (best of ``repeat``; the footprints, the
overlap and the windows are in it), and ``correct_shifts()`` with ``align_grids=True`` (GDAL's cubic warp of the
whole target) on its own.
"""
import json
import sys
import time
from pathlib import Path

# This folder holds arosics.py (the benchmark), which would shadow the package of the same name.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import numpy as np  # noqa: E402


def run(spec_path, out_path):
    from arosics import COREG

    spec = json.load(open(spec_path))
    out = []
    for p in spec["pairs"]:
        nd = p.get("nodata")
        kw = dict(ws=(p["ws"], p["ws"]), max_shift=p.get("max_shift", 5), max_iter=p.get("max_iter", 5),
                  nodata=(nd, nd), CPUs=p.get("cpus", 1), q=True, progress=False, ignore_errors=True,
                  align_grids=True, resamp_alg_deshift="cubic")
        times = []
        for _ in range(spec.get("repeat", 3)):
            t0 = time.perf_counter()
            try:
                crl = COREG(p["ref"], p["tgt"], **kw)
                crl.calculate_spatial_shifts()
                err = None
            except Exception as e:  # raised before ignore_errors applies (no overlap, ...)
                crl, err = None, repr(e)
            times.append(time.perf_counter() - t0)
        r = {"name": p["name"], "t_estimate": min(times), "cpus": kw["CPUs"]}
        if crl is None:
            r.update(success=False, error=err)
            out.append(r)
            continue
        ok = bool(crl.success)
        r.update(success=ok, error=None if ok else (repr(crl.tracked_errors[-1]) if crl.tracked_errors else None))
        if crl.matchBox is not None and crl.matchBox.boxImYX:
            ys, xs = zip(*crl.matchBox.boxImYX)
            r["window"] = [min(xs), min(ys), max(xs), max(ys)]
        r["fft_win_size_YX"] = list(crl.fft_win_size_YX) if crl.fft_win_size_YX else None
        if ok:
            r.update(x_shift_px=float(crl.x_shift_px), y_shift_px=float(crl.y_shift_px),
                     x_shift_map=float(crl.x_shift_map), y_shift_map=float(crl.y_shift_map),
                     reliability=None if crl.shift_reliability is None else float(crl.shift_reliability),
                     ssim_before=None if crl.ssim_orig is None else float(crl.ssim_orig),
                     ssim_after=None if crl.ssim_deshifted is None else float(crl.ssim_deshifted),
                     ssim_improved=None if crl.ssim_improved is None else bool(crl.ssim_improved))
            t0 = time.perf_counter()
            res = crl.correct_shifts()
            r["t_correct"] = time.perf_counter() - t0
            if p.get("aligned"):
                np.save(p["aligned"], np.asarray(res["arr_shifted"]))
        out.append(r)
    json.dump(out, open(out_path, "w"), indent=1)


def natural(folder):
    """A natural image as a pair: scikit-image's ``camera`` (512 x 512, 8 bit), the target moved by (2.3, -1.4)
    pixels (cubic spline), on a 30 m UTM grid. Not a satellite scene (there is none in the repository), but real
    image content: edges, texture and flat areas."""
    from pathlib import Path

    from osgeo import gdal, osr
    from scipy import ndimage as ndi
    from skimage import data

    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    cam = data.camera().astype(np.float64)
    dx, dy = 2.3, -1.4
    tgt = ndi.shift(cam, (-dy, -dx), order=3, mode="nearest")  # content moved by (-dx, -dy): to align, move by (dx, dy)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    for name, arr in (("camera_ref", cam), ("camera_tgt", tgt)):
        a = np.clip(np.rint(arr), 1, 255).astype(np.uint8)
        ds = gdal.GetDriverByName("GTiff").Create(str(folder / f"{name}.tif"), a.shape[1], a.shape[0], 1, gdal.GDT_Byte)
        ds.SetGeoTransform((500000.0, 30.0, 0.0, 4600000.0, 0.0, -30.0))
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).SetNoDataValue(0)
        ds.GetRasterBand(1).WriteArray(a)
        ds = None
    json.dump({"dx": dx, "dy": dy}, open(folder / "camera_shift.json", "w"))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "natural":
        natural(sys.argv[2])
    else:
        raise SystemExit(__doc__)
