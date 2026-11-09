"""The Level 2 products of zeit (``zeit._level2._level2_arrays``, the engine of ``zeit.level2``) and of FORCE 3.11.00 against ESA's Sen2Cor L2A of the same tile and date.

An independent check: the L2A is another chain (Sen2Cor: another atmosphere model and look-up tables, another aerosol and vapour
retrieval, its own cloud mask), so no equality is expected; the question is how close zeit gets to it and whether it gets as close
as FORCE does. The Zeit-FORCE difference is the control.

The scenes (not in the repository; ``ZEIT_L2_REAL_DATA`` or ``C:\\Users\\eduar\\Documents\\dev\\data_force_l2_real``): tile 23LKC of
2023-07-28 (L1C baseline 05.09) and of 2021-07-28 (03.01). The L2A comes from the public bucket ``gcp-public-data-sentinel-2``
(``L2/tiles/23/L/KC/``), the product with the baseline of the L1C used, in ``l2a/<product>.SAFE`` (only B02-B08, B8A, B11, B12,
SCL at 20 m and the metadata). Sides:

- zeit: ``_level2_arrays(safe_tif, dem=DEM, dobrdf=False)`` with its own Fmask (Sen2Cor does not normalise the BRDF);
- FORCE: ``force_out_<scene>_nobrdf`` (BOA and QAI; no BRDF either), already computed, not run here;
- L2A: DN -> reflectance x 10000 as ``DN + BOA_ADD_OFFSET`` (baseline >= 04.00, the sign as in ``MTD_MSIL2A.xml``) over the
  ``BOA_QUANTIFICATION_VALUE``; no offset before; DN 0 is no data.

Everything is compared on the 20 m grid: the 10 m bands (BLUE, GREEN, RED, BROADNIR) are averaged in 2x2 blocks (ignoring
-9999 / DN 0), the 20 m ones are taken as ``[::2, ::2]`` (zeit and FORCE write them replicated to 10 m). Common pixels: valid in
the three, clear in both QAIs (off, cloud and shadow bits 0) and SCL in {4, 5} (vegetation, bare soil), then SCL == 6 (water).
Per band: mean bias (A - B), MAE, RMSE, p95 of |d| and the correlation, for Zeit-L2A, FORCE-L2A and Zeit-FORCE. The masks: the
agreement and the IoU of "cloud or shadow" of zeit's Fmask and of FORCE's QAI against the SCL classes {3, 8, 9, 10} on the valid pixels (SCL 0
and 1 out).

    python benchmarks/vs_originals/level2_esa.py --scene 2023 [--n-jobs 8]

Prints the tables and writes ``l2a/result_<scene>.json`` in the data folder.
"""
import argparse
import json
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("GDAL_NUM_THREADS", "8")

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))  # the clone's zeit, not an installed one

DATA = Path(os.environ.get("ZEIT_L2_REAL_DATA", r"C:\Users\eduar\Documents\dev\data_force_l2_real"))
SCENES = {
    "2023": ("S2B_MSIL1C_20230728T132239_N0509_R038_T23LKC_20230728T150149",
             "S2B_MSIL2A_20230728T132239_N0509_R038_T23LKC_20230728T154244"),
    "2021": ("S2B_MSIL1C_20210728T132239_N0301_R038_T23LKC_20210728T151918",
             "S2B_MSIL2A_20210728T132239_N0301_R038_T23LKC_20210728T155746"),
}
DEM = DATA / "cop_glo30_23LKC.tif"
BOA_NAMES = ["BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2"]
# FORCE band -> (Sentinel-2 band, resolution of the L2A file, index in the 13-band list of MTD_MSIL2A.xml)
L2A_BANDS = {"BLUE": ("B02", 10, 1), "GREEN": ("B03", 10, 2), "RED": ("B04", 10, 3), "REDEDGE1": ("B05", 20, 4),
             "REDEDGE2": ("B06", 20, 5), "REDEDGE3": ("B07", 20, 6), "BROADNIR": ("B08", 10, 7), "NIR": ("B8A", 20, 8),
             "SWIR1": ("B11", 20, 11), "SWIR2": ("B12", 20, 12)}
NATIVE_10 = {"BLUE", "GREEN", "RED", "BROADNIR"}
SCL_CLOUD = (3, 8, 9, 10)
NODATA = -9999


def read(path):
    with rasterio.open(path) as d:
        return d.read()


def force_files(out_dir):
    d = next(p for p in Path(out_dir).iterdir() if p.is_dir() and p.name.startswith("T"))
    return {p.stem.split("_")[-1]: p for p in d.glob("*.tif")}


def block_mean(a, nodata):
    """2x2 block mean of a 10 m array, ignoring ``nodata``; NaN where the four pixels are no data."""
    h, w = a.shape
    b = a.reshape(h // 2, 2, w // 2, 2).astype(np.float32)
    b[b == nodata] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(b, axis=(1, 3))


def l2a_offsets(product):
    """``BOA_QUANTIFICATION_VALUE``, ``{band index: BOA_ADD_OFFSET}`` and the baseline of ``MTD_MSIL2A.xml``."""
    root = ET.parse(product / "MTD_MSIL2A.xml").getroot()
    quant = float(next(e.text for e in root.iter() if e.tag.endswith("BOA_QUANTIFICATION_VALUE")))
    base = next(e.text for e in root.iter() if e.tag.endswith("PROCESSING_BASELINE"))
    offs = {int(e.get("band_id")): float(e.text) for e in root.iter() if e.tag.endswith("BOA_ADD_OFFSET")}
    return quant, offs, base


def l2a_file(product, res, token):
    f = sorted(product.glob(f"GRANULE/*/IMG_DATA/R{res}m/*_{token}_{res}m.jp2"))
    if len(f) != 1:
        raise FileNotFoundError(f"{product}: {token} {res} m: {f}")
    return f[0]


def load_l2a(product, grid_shape):
    """The L2A reflectance x 10000 of the ten bands on the 20 m grid (NaN where DN 0) and its SCL."""
    quant, offs, base = l2a_offsets(product)
    out = []
    for name in BOA_NAMES:
        tok, res, idx = L2A_BANDS[name]
        dn = read(l2a_file(product, res, tok))[0]
        off = offs.get(idx, 0.0)
        if res == 10:
            a = block_mean(dn, 0)  # DN 0 -> NaN
            a = (a + off) * (10000.0 / quant)  # the mean of DN + offset: the offset is the same for the four
        else:
            a = dn.astype(np.float32)
            a[dn == 0] = np.nan
            a = (a + off) * (10000.0 / quant)
        if a.shape != grid_shape:
            raise ValueError(f"{name}: {a.shape} != {grid_shape}")
        out.append(a.astype(np.float32))
    scl = read(l2a_file(product, 20, "SCL"))[0]
    return np.stack(out), scl, dict(baseline=base, quantification=quant, offsets=sorted(set(offs.values())) or None)


def to20(boa, name_idx, name):
    """A zeit or FORCE band on the 20 m grid."""
    a = boa[name_idx]
    if name in NATIVE_10:
        return block_mean(a, NODATA)
    r = a[::2, ::2].astype(np.float32)
    r[r == NODATA] = np.nan
    return r


def metrics(a, b):
    d = a.astype(np.float64) - b
    ad = np.abs(d)
    return dict(n=int(d.size), bias=float(d.mean()), mae=float(ad.mean()), rmse=float(np.sqrt((d * d).mean())),
                p95=float(np.percentile(ad, 95)),
                r=float(np.corrcoef(a, b)[0, 1]) if d.size > 2 else float("nan"))


def table(title, sets, mask):
    rows = {}
    print(f"\n=== {title}: {int(mask.sum()):,} common pixels (20 m) ===")
    print(f"{'band':9s} | {'Zeit-L2A':^39s} | {'FORCE-L2A':^39s} | {'Zeit-FORCE (control)':^39s}")
    print(f"{'':9s} | " + " | ".join([f"{'bias':>7s} {'MAE':>6s} {'RMSE':>6s} {'p95':>6s} {'r':>6s}"] * 3))
    for i, name in enumerate(BOA_NAMES):
        z, f, e = (sets[k][i][mask] for k in ("zeit", "force", "l2a"))
        m = {"zeit_l2a": metrics(z, e), "force_l2a": metrics(f, e), "zeit_force": metrics(z, f)}
        rows[name] = m
        print(f"{name:9s} | " + " | ".join(
            f"{m[k]['bias']:7.1f} {m[k]['mae']:6.1f} {m[k]['rmse']:6.1f} {m[k]['p95']:6.1f} {m[k]['r']:6.3f}"
            for k in ("zeit_l2a", "force_l2a", "zeit_force")))
    return dict(n=int(mask.sum()), bands=rows)


def mask_scores(pred, ref, valid):
    p, r = pred[valid], ref[valid]
    tp = int((p & r).sum())
    fp = int((p & ~r).sum())
    fn = int((~p & r).sum())
    tn = int((~p & ~r).sum())
    n = tp + fp + fn + tn
    return dict(n=n, agreement=(tp + tn) / n, iou=tp / max(tp + fp + fn, 1), tp=tp, fp=fp, fn=fn, tn=tn,
                pred_fraction=(tp + fp) / n, ref_fraction=(tp + fn) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=sorted(SCENES), default="2023")
    ap.add_argument("--n-jobs", type=int, default=8)
    a = ap.parse_args()

    from common import no_power_throttling

    no_power_throttling()
    from zeit import _core, _level2

    print("zeit._core:", _core.__file__)
    l1c, l2a_name = SCENES[a.scene]
    product = DATA / "l2a" / f"{l2a_name}.SAFE"
    force = force_files(DATA / f"force_out_{a.scene}_nobrdf")

    t0 = time.perf_counter()
    safe = _level2.Safe(DATA / "tif" / f"{l1c}.SAFE")
    out = _level2._level2_arrays(safe, dem=DEM, dobrdf=False, n_jobs=a.n_jobs)
    t_zeit = time.perf_counter() - t0
    print(f"zeit level2 (read, DEM, Fmask, correction): {t_zeit:.1f} s; core {out.get('ms', float('nan')) / 1e3:.1f} s; fmask {out.get('fmask')}")
    qz = out["qai"][::2, ::2]
    zeit20 = np.stack([to20(out["boa"], i, n) for i, n in enumerate(BOA_NAMES)])
    del out
    boa_f = read(force["BOA"])
    qf = read(force["QAI"])[0][::2, ::2]
    force20 = np.stack([to20(boa_f, i, n) for i, n in enumerate(BOA_NAMES)])
    del boa_f
    l2a20, scl, info = load_l2a(product, zeit20.shape[1:])
    print(f"L2A {l2a_name}: baseline {info['baseline']}, quantification {info['quantification']}, BOA_ADD_OFFSET {info['offsets']}")

    def clear(q):
        return ((q & 1) == 0) & (((q >> 1) & 3) == 0) & (((q >> 3) & 1) == 0)

    valid_all = ~np.isnan(zeit20).any(0) & ~np.isnan(force20).any(0) & ~np.isnan(l2a20).any(0)
    base = valid_all & clear(qz) & clear(qf)
    sets = {"zeit": zeit20, "force": force20, "l2a": l2a20}
    res = dict(scene=a.scene, l2a_product=l2a_name, l1c=l1c, l2a=info, grid=list(zeit20.shape[1:]), time_zeit_s=t_zeit,
               valid_all=int(valid_all.sum()), clear_valid=int(base.sum()))
    res["veg_soil"] = table("SCL in {4, 5} (vegetation, bare soil)", sets, base & np.isin(scl, (4, 5)))
    res["water"] = table("SCL == 6 (water)", sets, base & (scl == 6))
    # the SCL composition of the common clear pixels
    res["scl_hist_clear"] = {int(k): int((base & (scl == k)).sum()) for k in range(12)}
    print("SCL histogram of the clear common pixels:", res["scl_hist_clear"])

    ref = np.isin(scl, SCL_CLOUD)
    ok = ~np.isin(scl, (0, 1)) & ((qz & 1) == 0) & ((qf & 1) == 0)
    masks = {}
    for lab, q in (("zeit_fmask", qz), ("force_qai", qf)):
        pred = (((q >> 1) & 3) > 0) | (((q >> 3) & 1) > 0)
        masks[lab] = mask_scores(pred, ref, ok)
    masks["zeit_vs_force"] = mask_scores((((qz >> 1) & 3) > 0) | (((qz >> 3) & 1) > 0),
                                         (((qf >> 1) & 3) > 0) | (((qf >> 3) & 1) > 0), ok)
    res["masks"] = masks
    print("\n=== cloud or shadow vs SCL {3, 8, 9, 10} (valid pixels, SCL 0 and 1 out) ===")
    for k, m in masks.items():
        print(f"{k:14s} n={m['n']:,} agreement {m['agreement'] * 100:.2f}% IoU {m['iou']:.3f} "
              f"(predicted {m['pred_fraction'] * 100:.2f}%, reference {m['ref_fraction'] * 100:.2f}%)")
    dst = DATA / "l2a" / f"result_{a.scene}.json"
    dst.write_text(json.dumps(res, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
