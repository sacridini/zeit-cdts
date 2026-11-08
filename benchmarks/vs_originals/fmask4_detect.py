"""The scenes of the comparison of Zeit's potential pixels of Fmask 4.7 (``zeit_core::fmask::detect``) with the original .m
files run by GNU Octave (see ``fmask4.py``, which calls this module): synthetic Sentinel-2 20 m scenes of fixed seeds
(land with texture, water, thick and thin cloud, cirrus, snow, bright urban surface, saturation, no-data in frames and
blocks), the Octave chain of ``fmask4/fm_scene.m`` (autoFmask.m's calls) and the comparison, stage by stage, of each Rust
function fed with the Octave output of the stages before it.
"""
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from zeit import _core

fm = _core.fmask
F32 = np.float32

# name: (rows, cols, seed, fractions of the classes, dem, gswo)
SCENES = {
    "A": dict(rows=320, cols=300, seed=1, cloud=0.10, thin=0.10, cirr=0.08, snow=0.02, urban=0.05, water=0.12, dem=True, gs="good"),
    "B": dict(rows=300, cols=340, seed=2, cloud=0.12, thin=0.08, cirr=0.06, snow=0.03, urban=0.04, water=0.10, dem=False, gs=None),
    "C": dict(rows=360, cols=320, seed=3, cloud=0.75, thin=0.05, cirr=0.04, snow=0.0, urban=0.02, water=0.05, dem=True, gs=None),
    "F": dict(rows=300, cols=300, seed=4, cloud=0.08, thin=0.05, cirr=0.05, snow=0.0, urban=0.03, water=0.15, dem=True, gs="zero"),
    "G": dict(rows=300, cols=300, seed=5, cloud=0.08, thin=0.05, cirr=0.05, snow=0.01, urban=0.03, water=0.15, dem=False, gs="low"),
    "E": dict(rows=450, cols=500, seed=6, cloud=0.08, thin=0.04, cirr=0.04, snow=0.60, urban=0.02, water=0.05, dem=False, gs="good", nan_green=3),
    "H": dict(rows=450, cols=500, seed=8, cloud=0.0, thin=0.0, cirr=0.0, snow=1.0, urban=0.0, water=0.0, dem=False, gs=None, rough_size=0.15, nodata=False, nan_green=2, sat=0.00003),
    "T": dict(rows=1300, cols=100, seed=9, cloud=0.0, thin=0.0, cirr=0.0, snow=1.0, urban=0.0, water=0.0, dem=False, gs=None, rough_size=0.15, nodata=False, nan_green=1, sat=0.00003),
    "D": dict(rows=1000, cols=1000, seed=7, cloud=0.12, thin=0.08, cirr=0.06, snow=0.25, urban=0.04, water=0.08, dem=True, gs="good", tall_dem=True),
}


def _sm(rng, rows, cols, sigma):
    f = ndimage.gaussian_filter(rng.standard_normal((rows, cols)).astype(F32), sigma)
    return (f - f.mean()) / f.std()


def _top(f, frac):
    """The pixels of the highest ``frac`` of a field."""
    if frac <= 0:
        return np.zeros(f.shape, bool)
    return f > np.quantile(f, 1 - frac)


def make_scene(rows, cols, seed, cloud=0.1, thin=0.1, cirr=0.1, snow=0.0, urban=0.05, water=0.1, dem=True, gs="good",
               nan_green=0, tall_dem=False, rough_size=0.22, nodata=True, sat=0.004):
    """One scene: a dict of the bands (float32, x10000), the saturation flags (uint8), the DEM and the GSWO."""
    rng = np.random.default_rng(seed)
    shp = (rows, cols)
    veg = 0.5 + 0.5 * np.tanh(_sm(rng, rows, cols, 30))
    t1 = 0.5 + 0.5 * np.tanh(_sm(rng, rows, cols, 8))
    noise = lambda: (1 + 0.03 * rng.standard_normal(shp)).astype(F32)
    b = {
        "blue": (400 + 300 * t1) * noise(),
        "green": (650 + 350 * t1) * noise(),
        "red": (500 + 800 * (1 - veg)) * noise(),
        "nir": (1500 + 2500 * veg) * noise(),
        "swir1": (1200 + 1500 * (1 - veg)) * noise(),
        "swir2": (700 + 900 * (1 - veg)) * noise(),
        "cirrus": (15 + 15 * t1) * noise(),
    }
    b = {k: v.astype(F32) for k, v in b.items()}
    u = rng.random(shp).astype(F32)

    def put(m, **vals):
        for k, v in vals.items():
            b[k][m] = np.broadcast_to(np.asarray(v, F32), shp)[m]

    # cirrus over land (and the thin cloud's): more cirrus signal only
    m = _top(_sm(rng, rows, cols, 20), cirr)
    put(m, cirrus=150 + 350 * u)
    # thin cloud: the land plus a veil
    m = _top(_sm(rng, rows, cols, 15), thin)
    a = 600 + 1200 * u
    put(m, blue=b["blue"] + a, green=b["green"] + a, red=b["red"] + a, nir=b["nir"] + 0.8 * a, swir1=b["swir1"] + 0.4 * a,
        swir2=b["swir2"] + 0.2 * a, cirrus=120 + 300 * u)
    # thick cloud (bright, white, high cirrus)
    m = _top(_sm(rng, rows, cols, 18), cloud)
    mcloud = m.copy()
    c = 3500 + 4500 * u
    put(m, blue=c * 0.95, green=c, red=c * 0.98, nir=c * 0.95, swir1=c * 0.7, swir2=c * 0.55, cirrus=400 + 500 * u)
    # water: dark, NDVI < 0, with some pixels of a higher NIR (0 < NDVI < 0.1 branch)
    mw = _top(_sm(rng, rows, cols, 12), water)
    put(mw, blue=700 + 100 * u, green=550 + 60 * u, red=350 + 40 * u, nir=150 + 250 * u, swir1=50 + 100 * u, swir2=30 + 60 * u, cirrus=12)
    shore = mw & (rng.random(shp) < 0.1)
    put(shore, red=300 + 0 * u, nir=300 + 100 * u)  # NDVI about 0.0..0.14: the 0 < NDVI < 0.1 and nir < 500 test
    # cloud over water (bright, white, SWIR1 and cirrus well above the clear water's)
    rng2 = np.random.default_rng(seed + 1000)
    mwc = mw & _top(_sm(rng2, rows, cols, 6), 0.3)
    put(mwc, blue=2500 + 100 * u, green=2300 + 100 * u, red=2000 + 100 * u, nir=300 + 100 * u, swir1=600 + 100 * u, swir2=500 + 100 * u, cirrus=600 + 100 * u)
    # bright urban surface (NDBI > 0)
    m = _top(_sm(rng, rows, cols, 6), urban)
    murb = m.copy()
    put(m, blue=1800 + 300 * u, green=2000 + 300 * u, red=2300 + 300 * u, nir=2600 + 300 * u, swir1=3200 + 300 * u, swir2=2800 + 300 * u)
    # snow (potential cloud pixels, NDSI > 0.15, bright in the visible, dark in SWIR1)
    # (smooth, so that the spectral-contextual index of DetectAbsSnow finds absolute snow; a rough block, the same
    # pixel noise as the other classes, keeps part of it from being absolute)
    m = _top(_sm(rng, rows, cols, 25), snow)
    us = (0.5 + 0.02 * _sm(rng, rows, cols, 10) + 0.002 * rng.standard_normal(shp)).astype(F32)
    rough = np.zeros(shp, bool)
    rough[rows // 5: rows // 5 + int(rows * rough_size), cols // 3: cols // 3 + int(cols * rough_size)] = True
    us = np.where(rough, u, us).astype(F32)
    put(m, blue=5500 + 500 * us, green=6000 + 500 * us, red=5800 + 500 * us, nir=5000 + 500 * us, swir1=900 + 400 * us, swir2=400 + 100 * us, cirrus=20)
    # saturation: spots of one or more visible bands
    sat = _top(rng.random(shp), sat) | _top(_sm(rng, rows, cols, 2), sat / 2)
    s = {k: (sat & (rng.random(shp) < 0.6)).astype(np.uint8) for k in ("sb", "sg", "sr")}
    s["sg"] |= sat.astype(np.uint8) & (b["green"] > 5000).astype(np.uint8)
    for k, band in (("sb", "blue"), ("sg", "green"), ("sr", "red")):
        b[band][s[k] == 1] = 9000 + 1000 * u[s[k] == 1]
    # no data: a frame, a block, a thin column and a diagonal-ish block
    nd = np.zeros(shp, bool)
    nd[:4, :] = nd[-4:, :] = nd[:, :4] = nd[:, -4:] = True
    nd[rows // 3: rows // 3 + rows // 8, cols // 2: cols // 2 + cols // 6] = True
    nd[:, cols // 5] = True
    nd[rows - rows // 6:, : cols // 4] = True
    for k in (b if nodata else ()):
        b[k][nd] = -9999
    for k in (s if nodata else ()):
        s[k][nd] = 0
    if nan_green:
        ys, xs = rng.integers(10, rows - 10, nan_green), rng.integers(10, cols - 10, nan_green)
        b["green"][ys, xs] = np.nan
    # bands 7 and 8 for the CDI (band 8A is the NIR above): the ratios to it vary more in half of the cloud, over
    # water and in half of the urban surface (the displaced high objects), and little elsewhere
    conf = (mcloud | mwc) & (_sm(rng2, rows, cols, 30) > 0)
    urbconf = murb & (_sm(rng2, rows, cols, 10) > 0)
    sd8 = (0.02 + 0.2 * conf + 0.15 * urbconf).astype(F32)
    b8 = (b["nir"] * (1 + sd8 * rng2.standard_normal(shp))).astype(F32)
    b7 = (0.9 * b["nir"] * (1 + 0.02 * rng2.standard_normal(shp))).astype(F32)
    if nodata:
        b7[nd] = -9999
        b8[nd] = -9999
    out = dict(b, **s)
    out["b7"], out["b8"] = b7, b8
    out["mask"] = (b["blue"] > -9999)
    if dem:
        top = 2600 if tall_dem else 1000
        d = 600 + top * (0.5 + 0.5 * np.tanh(_sm(rng, rows, cols, 50))) + 60 * _sm(rng, rows, cols, 3)
        out["dem"] = d.astype(F32)
    else:
        out["dem"] = None
    if gs is None:
        out["gs"] = None
    elif gs == "zero":
        out["gs"] = np.zeros(shp, np.uint8)
    elif gs == "low":
        out["gs"] = np.where(mw, rng.integers(0, 5, shp), rng.integers(0, 30, shp)).astype(np.uint8)
    else:
        g = np.where(mw, rng.integers(60, 101, shp), rng.integers(0, 10, shp))
        g[rows // 10: rows // 10 + rows // 8, cols // 10: cols // 10 + cols // 8] = 85  # an old lake, now dry land
        out["gs"] = g.astype(np.uint8)
    if out["dem"] is not None:
        gy, gx = np.gradient(out["dem"].astype(np.float64), 20.0)
        out["slope"] = np.degrees(np.arctan(np.hypot(gx, gy))).astype(F32)
    else:
        out["slope"] = None
    return out


def scene_of(name):
    p = dict(SCENES[name])
    rows, cols, seed = p.pop("rows"), p.pop("cols"), p.pop("seed")
    return make_scene(rows, cols, seed, **p)


def write_scene(c, name):
    sc = scene_of(name)
    rows, cols = sc["blue"].shape
    for k in ("blue", "green", "red", "nir", "swir1", "swir2", "cirrus", "sb", "sg", "sr"):
        sc[k].tofile(c.d / f"{name}.{k}")
    sc["b7"].tofile(c.d / f"{name}.b7")
    sc["b8"].tofile(c.d / f"{name}.b8")
    if sc["slope"] is not None:
        sc["slope"].tofile(c.d / f"{name}.slope")
    if sc["dem"] is not None:
        sc["dem"].tofile(c.d / f"{name}.dem")
    if sc["gs"] is not None:
        sc["gs"].tofile(c.d / f"{name}.gs")
    c.lines.append(f"fm_scene {name} {rows} {cols} {int(sc['dem'] is not None)} {int(sc['gs'] is not None)}")


# ------------------------------------------------------------------------------------------------------------------
def _ordered32(a):
    i = np.ascontiguousarray(a, F32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def _ordered64(a):
    i = np.ascontiguousarray(a, np.float64).view(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFFFFFFFFFF), i)


def ulps_any(a, b):
    """(max ulps, number differing, NaN mismatches) of two float arrays of the same dtype (f32 or f64)."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.isnan(a), np.isnan(b)
    ok = ~(na | nb)
    f = _ordered32 if a.dtype == np.float32 else _ordered64
    d = np.abs(f(a)[ok] - f(b)[ok])
    return (int(d.max()) if d.size else 0), int((d > 0).sum()), int((na != nb).sum())


def _rd(d, name, ext, dtype):
    return np.fromfile(Path(d) / f"{name}.{ext}", dtype)


def _flag(d, name, ext):
    return bool(_rd(d, name, ext, np.uint8)[0])


def compare_scene(d, name, rep):
    """Each stage of Rust on the Octave input of that stage."""
    d = Path(d)
    if (d / f"{name}.err").exists():
        rep.note(f"scene {name}", 0, "OCTAVE ERROR: " + (d / f"{name}.err").read_text())
        rep.bad += 1
        return
    sc = scene_of(name)
    rows, cols = sc["blue"].shape
    n = rows * cols
    g = lambda k, dt: _rd(d, name, k, dt)
    flat = lambda k: np.ascontiguousarray(sc[k]).ravel()
    dem = flat("dem") if sc["dem"] is not None else None
    gs = flat("gs") if sc["gs"] is not None else None
    tag = f"[{name} {rows}x{cols}]"

    def cont(stage, got, want):
        mx, nd, nan = ulps_any(got, want)
        rep.note(f"{tag} {stage}", n, f"max {mx} ulp, {nd} differ, {nan} NaN mismatch ({np.asarray(got).dtype})")
        rep.bad += nan > 0 or mx > 0

    def msk(stage, got, want):
        got, want = np.asarray(got).ravel(), np.asarray(want).ravel()
        nd = int((got != want).sum())
        rep.note(f"{tag} {stage}", got.size, f"{nd} differ" if nd else "equal")
        rep.bad += nd > 0

    # FmaskParameters
    prm = g("prm", np.float64)
    msk("FmaskParameters (9 values)", np.array(fm.parameters()), prm)
    mask = g("mask", np.uint8)
    msk("ObservMask == blue > -9999", mask, flat("mask").astype(np.uint8))
    ndvi_o, ndsi_o, ndbi_o = g("ndvi", F32), g("ndsi", F32), g("ndbi", F32)
    cont("NDVI", fm.ndvi(flat("red"), flat("nir")), ndvi_o)
    cont("NDSI", fm.ndsi(flat("green"), flat("swir1")), ndsi_o)
    cont("NDBI", fm.ndbi(flat("nir"), flat("swir1")), ndbi_o)
    satu_o = g("satubv", np.uint8)
    msk("Saturate", fm.saturate(flat("sb"), flat("sg"), flat("sr")), satu_o)
    psnow_o = g("psnow", np.uint8)
    msk("DetectSnow", fm.detect_snow(flat("green"), flat("nir"), ndsi_o), psnow_o)
    water_o, wall_o = g("water", np.uint8), g("waterall", np.uint8)
    w, wa = fm.detect_water(mask, flat("nir"), ndvi_o, psnow_o, gs)
    msk(f"DetectWater: water ({int(water_o.sum())} px, spectral+GSWO, psnow {int(psnow_o.sum())})", w, water_o)
    msk(f"DetectWater: waterAll (GSWO added {int((wall_o > 0).sum() - (water_o > 0).sum())} px)", wa, wall_o)
    idp_o, cn_o, wh_o, hot_o = g("idplcd", np.uint8), g("cn", np.float64), g("whiteness", F32), g("hot", F32)
    idp, cn, wh, hot = fm.detect_potential_pixels(mask, flat("blue"), flat("green"), flat("red"), flat("nir"), flat("swir1"), flat("swir2"),
                                                  flat("cirrus"), ndvi_o, ndsi_o, satu_o, dem)
    msk("DetectPotentialPixels: idplcd", idp, idp_o)
    cont("DetectPotentialPixels: BandCirrusNormal", cn, cn_o)
    cont("DetectPotentialPixels: whiteness", wh, wh_o)
    cont("DetectPotentialPixels: HOT", hot, hot_o)
    cont("NormalizaCirrusDEM (on the Octave idplcd)", fm.normaliza_cirrus_dem(mask, idp_o, flat("cirrus"), dem), g("cn2", np.float64))
    abs_ = fm.detect_abs_snow(flat("green"), flat("sg"), ndsi_o, psnow_o, rows, cols, 20.0)
    empty = _flag(d, name, "absempty")
    npsnow = int(g("psnow_n", np.float64)[0])
    if empty or abs_ is None:
        rep.note(f"{tag} DetectAbsSnow (psnow {npsnow})", 1, "equal ([] on both sides)" if (empty and abs_ is None) else "MISMATCH in [] vs array")
        rep.bad += not (empty and abs_ is None)
    else:
        msk(f"DetectAbsSnow (psnow {npsnow}, {int(abs_.sum())} absolute)", abs_, g("abssnow", np.uint8))
    idp2_o = g("idplcd2", np.uint8)
    cn_in = cn_o
    r = fm.detect_potential_cloud(mask, water_o, flat("swir1"), cn_in, flat("sg"), flat("sr"), ndvi_o, ndsi_o, ndbi_o, idp2_o, wh_o, hot_o, prm[3], prm[4])
    sum_clr, cloud, used, tl, th = r
    rep.note(f"{tag} DetectPotentialCloud: sum_clr", 1, f"Rust {sum_clr}, Octave {int(g('sumclr', np.float64)[0])}" + ("" if sum_clr == int(g('sumclr', np.float64)[0]) else " MISMATCH"))
    rep.bad += sum_clr != int(g("sumclr", np.float64)[0])
    msk(f"DetectPotentialCloud: cloud ({int(cloud.sum())} px, sum_clr {sum_clr}{' <= 40000' if sum_clr <= 40000 else ''})", cloud, g("cloud", np.uint8))
    uo = _flag(d, name, "idusedempty")
    if uo or used is None:
        rep.note(f"{tag} DetectPotentialCloud: idused", 1, "equal ([] on both sides)" if (uo and used is None) else "MISMATCH in [] vs array")
        rep.bad += not (uo and used is None)
    else:
        msk("DetectPotentialCloud: idused", used, g("idused", np.uint8))
    msk("DetectPotentialCloud: t_templ, t_temph", np.array([tl, th]), g("tl", np.float64))
    cont("probThin", fm.prob_thin(cn_o), g("pthin", np.float64))
    cont("probwBrightness", fm.probw_brightness(flat("swir1")), g("pwb", F32))
    idlnd_o = g("idlnd", np.uint8)
    if (d / f"{name}.plb").exists():
        cont("problBrightness", fm.probl_brightness(hot_o, idlnd_o, 0.175, 1 - 0.175), g("plb", F32))
    cont("problSpectralVaribility", fm.probl_spectral_variability(ndvi_o, ndsi_o, ndbi_o, wh_o, flat("sg"), flat("sr")), g("psv", F32))

    # (Part A) the values kept inside the .m, against the literal replicas
    if (d / f"{name}.lprob").exists():
        rr = fm.detect_potential_cloud_probs(mask, water_o, flat("swir1"), cn_in, flat("sg"), flat("sr"), ndvi_o, ndsi_o, ndbi_o, idp2_o, wh_o, hot_o, prm[3], prm[4])
        lp, wp, cmx, wmx = rr[3]
        cont("DetectPotentialCloud: lprob_final (replica of lines 112-126)", lp, g("lprob", F32))
        cont("DetectPotentialCloud: wprob_final (replica)", wp, g("wprob", F32))
        thr = g("thr", F32)
        msk("DetectPotentialCloud: clr_max, wclr_max (single, NaN equal)", np.nan_to_num(np.array([cmx, wmx], F32), nan=-1.0), np.nan_to_num(thr, nan=-1.0))
        pc = g("probclass", np.uint8).copy()
        if np.isnan(thr[1]):
            pc[3] = 1  # no clear water: prctile([]) is a double NaN in Octave, and nothing is above NaN either way
        msk("  classes in the .m are single (lprob, wprob, clr_max, wclr_max)", pc, np.ones(4, np.uint8))
    if (d / f"{name}.scsi").exists():
        ab, sc_r = fm.detect_abs_snow_scsi(flat("green"), flat("sg"), ndsi_o, psnow_o, rows, cols, 20.0)
        so = g("scsi", np.float64)
        mx, nd, nan = ulps_any(sc_r.astype(F32), so.astype(F32))
        rep.note(f"{tag} DetectAbsSnow: scsi (replica of lines 41-88)", n,
                 f"max {mx} ulp(f32), {nd} differ, {nan} NaN mismatch; product class single: {_flag(d, name, 'scsiclass')} "
                 "(the few pixels are where single(std) rounds the other way: std in double differs from Octave's by ~1e-12 and the value is at a rounding tie of single)")
        rep.bad += nan > 0 or mx > 2 or not _flag(d, name, "scsiclass")
        msk("  (scsi < 9 mask of the replica)", (sc_r < 9), (so < 9))

    # (Part B)
    if (d / f"{name}.cdi").exists():
        compare_commission(d, name, rep, sc, flat, g, tag, rows, cols, n, mask, ndbi_o, ndvi_o, psnow_o, water_o, g("cloud", np.uint8), cont, msk)


def build_direct(d, base="A"):
    """DetectPotentialCloud alone with exactly 39999, 40000 and 40001 clear pixels (and corners), from the arrays of
    scene ``base`` (Octave's outputs). Returns the case lines."""
    d = Path(d)
    sc = scene_of(base)
    rows, cols = sc["blue"].shape
    n = rows * cols
    g = lambda k, dt: np.fromfile(d / f"{base}.{k}", dt)
    mask = g("mask", np.uint8)
    lines = []
    cases = {}
    for k in (39999, 40000, 40001, 60000):
        idp = np.ones(n, np.uint8)
        pos = np.flatnonzero(mask)[:k]
        idp[pos] = 0  # exactly k clear (observed, not potential) pixels
        cases[f"dir_{k}"] = (idp, g("water", np.uint8))
    idp = np.ones(n, np.uint8)
    idp[np.flatnonzero(mask)[:50000]] = 0
    cases["dir_allwater"] = (idp, np.ones(n, np.uint8))  # no clear land: idused = idclr
    cases["dir_noland_small"] = (idp, (np.arange(n) % 2000 != 0).astype(np.uint8))  # land < 0.1 % of the observed
    cases["dir_allland"] = (idp, np.zeros(n, np.uint8))  # no clear water: wclr_h is NaN
    for cid, (idp, water) in cases.items():
        idp.tofile(d / f"{cid}.idplcd")
        water.tofile(d / f"{cid}.water")
        for k, dt in (("mask", np.uint8), ("swir1", F32), ("ndvi", F32), ("ndsi", F32), ("ndbi", F32), ("whiteness", F32), ("hot", F32), ("cn", np.float64)):
            g(k, dt).tofile(d / f"{cid}.{k}")
        sc["sg"].tofile(d / f"{cid}.sg")
        sc["sr"].tofile(d / f"{cid}.sr")
        lines.append(f"fm_pc_direct {cid} {rows} {cols}")
    return lines, list(cases)


def compare_direct(d, ids, base, rep):
    d = Path(d)
    sc = scene_of(base)
    rows, cols = sc["blue"].shape
    for cid in ids:
        if (d / f"{cid}.err").exists():
            rep.note(f"direct {cid}", 0, "OCTAVE ERROR: " + (d / f"{cid}.err").read_text())
            rep.bad += 1
            continue
        g = lambda k, dt: np.fromfile(d / f"{cid}.{k}", dt)
        r = fm.detect_potential_cloud(g("mask", np.uint8), g("water", np.uint8), g("swir1", F32), g("cn", np.float64), np.ascontiguousarray(sc["sg"]).ravel(),
                                      np.ascontiguousarray(sc["sr"]).ravel(), g("ndvi", F32), g("ndsi", F32), g("ndbi", F32), g("idplcd", np.uint8),
                                      g("whiteness", F32), g("hot", F32), 0.5, 20.0)
        sc_o = int(g("sumclr", np.float64)[0])
        nd = int((r[1] != g("cloud", np.uint8)).sum())
        ue = _flag(d, cid, "idusedempty")
        ud = 0 if (ue and r[2] is None) else (-1 if (ue or r[2] is None) else int((r[2] != g("idused", np.uint8)).sum()))
        rep.note(f"[direct {cid}] DetectPotentialCloud", rows * cols,
                 f"sum_clr {r[0]} vs {sc_o}; cloud {int(r[1].sum())} px, {nd} differ; idused {'[]' if r[2] is None else 'array'}, {ud} differ")
        rep.bad += (r[0] != sc_o) or nd > 0 or ud != 0


def timing_scene(size=5490, tile_rows=549):
    """The chain of Rust on a scene of ``size`` x ``size`` (a scene of tile_rows tiled), stage by stage."""
    reps = size // tile_rows
    p = dict(SCENES["A"])
    p.pop("rows"), p.pop("cols"), p.pop("seed")
    p["snow"] = 0.3  # so that DetectAbsSnow runs
    sc = make_scene(tile_rows, tile_rows, 11, **p)
    t = lambda a: np.ascontiguousarray(np.tile(a, (reps, reps))).ravel()
    s = {k: t(v) for k, v in sc.items() if v is not None}
    n = s["blue"].size
    print(f"\nChain of the Rust potential pixels on {size} x {size} ({n} pixels), RAYON_NUM_THREADS as set:")
    tt = {}

    def tm(name, fn):
        t0 = time.perf_counter()
        r = fn()
        tt[name] = time.perf_counter() - t0
        print(f"  {name:40s} {tt[name]:7.2f} s")
        return r

    mask = s["mask"].astype(np.uint8)
    ndvi = tm("NDVI", lambda: fm.ndvi(s["red"], s["nir"]))
    ndsi = tm("NDSI", lambda: fm.ndsi(s["green"], s["swir1"]))
    ndbi = tm("NDBI", lambda: fm.ndbi(s["nir"], s["swir1"]))
    sat = tm("Saturate", lambda: fm.saturate(s["sb"], s["sg"], s["sr"]))
    psnow = tm("DetectSnow", lambda: fm.detect_snow(s["green"], s["nir"], ndsi))
    water, _ = tm("DetectWater (with GSWO)", lambda: fm.detect_water(mask, s["nir"], ndvi, psnow, s["gs"]))
    idp, cn, wh, hot = tm("DetectPotentialPixels (with DEM)", lambda: fm.detect_potential_pixels(
        mask, s["blue"], s["green"], s["red"], s["nir"], s["swir1"], s["swir2"], s["cirrus"], ndvi, ndsi, sat, s["dem"]))
    tm("NormalizaCirrusDEM (no DEM)", lambda: fm.normaliza_cirrus_dem(mask, idp, s["cirrus"], None))
    abs_ = tm(f"DetectAbsSnow (psnow {int(psnow.sum())})", lambda: fm.detect_abs_snow(s["green"], s["sg"], ndsi, psnow, size, size, 20.0))
    if abs_ is not None:
        idp = idp.copy()
        idp[abs_ == 1] = 0
    r = tm("DetectPotentialCloud", lambda: fm.detect_potential_cloud(mask, water, s["swir1"], cn, s["sg"], s["sr"], ndvi, ndsi, ndbi, idp, wh, hot, 0.5, 20.0))
    cdi = tm("CDI", lambda: fm.cdi(s["b7"], s["b8"], s["nir"], size, size))
    pfpl = tm("DetectPotentialFalsePositivePixels (with slope, cdi)", lambda: fm.detect_potential_false_positive_pixels(
        mask, psnow, ndbi, ndvi, water, size, size, 20.0, s["slope"], cdi, False))
    fin = tm("ErodeCommissons (with cdi)", lambda: fm.erode_commissons(r[1], pfpl, water, 5, size, size, cdi))
    print(f"  total {sum(tt.values()):.2f} s; sum_clr {r[0]}, cloud {int(r[1].sum())} px, pfpl {int(pfpl.sum())} px, final cloud {int(fin.sum())} px")


def compare_commission(d, name, rep, sc, flat, g, tag, rows, cols, n, mask, ndbi_o, ndvi_o, psnow_o, water_o, cloud_o, cont, msk):
    nir = flat("nir")
    cdi_r = fm.cdi(flat("b7"), flat("b8"), nir, rows, cols)
    cdi_o = g("cdi", np.float64)
    mx, nd, nan = ulps_any(cdi_r, cdi_o)
    rel = np.abs(cdi_r - cdi_o)[~np.isnan(cdi_o)]
    f08 = int(((cdi_r < -0.8) != (cdi_o < -0.8)).sum())
    f05 = int(((cdi_r < -0.5) != (cdi_o < -0.5)).sum())
    rep.note(f"{tag} CDI", n, f"max {mx} ulp, {nd} differ (max abs {np.nanmax(rel):.1e}), {nan} NaN mismatch; flips of cdi<-0.8: {f08}, of cdi<-0.5: {f05}; "
                              f"Octave class double: {_flag(d, name, 'cdiclass')} (stdfilt: cancellation in var7-var8)")
    rep.bad += nan > 0 or f08 > 0 or f05 > 0 or np.nanmax(rel) > 1e-9
    # EnhanceLine: the imfilter differs from Octave's by the order of the sums
    eo = g("ndbienh", F32)
    er = fm.enhance_line(ndbi_o, rows, cols)
    mx, nd, nan = ulps_any(er, eo)
    from scipy import ndimage as ndi
    scale = ndi.correlate(np.abs(ndbi_o.reshape(rows, cols)).astype(np.float64), np.ones((3, 3)) * 2 / 6 * 1.0, mode="constant").ravel()
    err = np.abs(er.astype(np.float64) - eo.astype(np.float64)) / (np.finfo(F32).eps * np.maximum(scale, 1e-300))
    ok = ~np.isnan(eo)
    t_o = (eo > 0) & (eo > ndvi_o)
    t_r = (er > 0) & (er > ndvi_o)
    flips = np.flatnonzero(t_o != t_r)
    thr = np.where(ndvi_o > 0, ndvi_o, 0.0)  # the tighter of the two thresholds
    dist = np.abs(np.maximum(eo[flips], 0).astype(np.float64) - thr[flips]) if flips.size else np.array([0.0])
    rep.note(f"{tag} EnhanceLine (ndbi)", n, f"{nd} differ, max {err[ok].max():.2f} eps of the sums, {nan} NaN mismatch; flips of (e>0 & e>ndvi): {flips.size}"
                                           + (f" (their distance to the threshold: max {(dist / (np.finfo(F32).eps * np.maximum(scale[flips], 1e-300))).max():.2f} eps of the sums)" if flips.size else ""))
    rep.bad += nan > 0 or err[ok].max() > 8
    # DetectPotentialFalsePositivePixels, fed with Octave's enhanced ndbi (exact) and with the raw ndbi (flips counted)
    slope = flat("slope") if sc.get("slope") is not None else None
    for tagv, pname, sl, cd in (("a (cdi, slope)", "pfpl_a", slope, cdi_o), ("b (cdi)", "pfpl_b", None, cdi_o), ("c (no cdi)", "pfpl_c", None, None)):
        if pname == "pfpl_a" and slope is None:
            continue
        want = g(pname, np.uint8)
        got_e = fm.detect_potential_false_positive_pixels(mask, psnow_o, eo, ndvi_o, water_o, rows, cols, 20.0, sl, cd, True)
        msk(f"DetectPotentialFalsePositivePixels {tagv} on Octave's enhanced ndbi ({int(want.sum())} px)", got_e, want)
        got = fm.detect_potential_false_positive_pixels(mask, psnow_o, ndbi_o, ndvi_o, water_o, rows, cols, 20.0, sl, cd, False)
        nd = int((got != want).sum())
        rep.note(f"{tag}   ... with Rust's own EnhanceLine", n, f"{nd} differ" if nd else "equal")
        rep.bad += nd > 0 and flips.size == 0  # differences are only explained by flips of the tests
    # ErodeCommissons on Octave's inputs
    for ename, pname, cd in (("ero_a", "pfpl_a", cdi_o), ("ero_b", "pfpl_b", cdi_o), ("ero_c", "pfpl_c", None), ("ero_d", "pfpl_a", None)):
        if pname == "pfpl_a" and slope is None:
            continue
        want = g(ename, np.uint8)
        got = fm.erode_commissons(cloud_o, g(pname, np.uint8), water_o, 5, rows, cols, cd)
        msk(f"ErodeCommissons {ename} (pcloud {int(cloud_o.sum())} px -> {int(want.sum())}, {'with' if cd is not None else 'no'} cdi)", got, want)
