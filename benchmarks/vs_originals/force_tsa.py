"""Zeit's time series analysis against FORCE 3.11.00's higher-level TSA, on the same synthetic cube.

The FORCE side (``force_tsa.sh``) wrote its outputs in ``workdir("force_tsa")/ref``; this script feeds Zeit's core
with FORCE's own series (the int16 TSI, -9999 as NaN) and compares what comes back with FORCE's int16 outputs, which
are quantized: Zeit's float result is converted the way FORCE writes it (truncation toward zero; SKW and KRT times
1000 and clamped to +-30000; the STM's IQR as trunc(q75) - trunc(q25); the standardized values times 10000).

    python benchmarks/vs_originals/force_tsa.py --part stm,fold,standardize [--jobs 4]

Parts: ``stm`` (the 16 statistics of the whole series), ``fold`` (the ten statistics by year, quarter, month, week and
day of the year), ``standardize``, ``interp`` (the linear, moving, RBF and harmonic interpolations of FORCE's own TSS
onto its 16-day grid, and the harmonic fit's NRT product; one pixel whose IRLS cycles is left out, see
``CYCLING_PIXELS``), ``trend`` (``zeit.linear_trend`` and ``zeit.cat`` of the average folds by year and by month,
against TRY, TRM, CAY and CAM with and without the change penalty), ``cso`` (``zeit.clear_observations``' core on
FORCE's TSS and windows against cso_m1 and cso_m3: NUM, MIN, MAX, RNG and AVG equal; STD, SKW, KRT, the quantiles and IQR
differ by design, shown with a port of FORCE's own code).

For each product and band it prints the share of pixels that are equal, within +-1, and the largest |difference|.
Where FORCE's result is undefined (a standard deviation, skewness or kurtosis of one value, a skewness or kurtosis
of a constant series: 0/0, which FORCE casts to int16 as it happens) the pixels are left out, and the count is printed.
The target is 100% equal or within +-1; the exit code is 1 when something is outside it.

``--ref <folder>`` reads the outputs from another folder of the same layout: the 500 x 500 measurement cube of the timing
(``force_tsa_timing.sh`` and ``force_tsa_timing.py``), where the parts ``stm`` and ``interp`` check that the agreement
holds on a larger cube (``int_harmonic_part``, with its cycling pixel, is not run there).
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd

import force_ref as ref
from common import no_power_throttling

# FORCE writes STM/FOLD nodata as this.
NODATA = -9999
STM_BANDS = ["MIN", "Q01", "Q10", "Q25", "Q50", "Q75", "Q90", "Q99", "MAX", "AVG", "STD", "RNG", "IQR", "SKW", "KRT", "NUM"]
FOLD_PRODUCTS = {"FBY": "year", "FBQ": "quarter", "FBM": "month", "FBW": "week", "FBD": "doy"}
FOLD_TYPES = ["AVG", "Q50", "STD", "MIN", "MAX", "SKW", "KRT", "IQR", "Q25", "NUM"]

_failed = []
VERBOSE = False


def tsi(config="int_rbf"):
    """FORCE's TSI as ``(pixels, time)`` float64 (NaN where nodata), its dates, and the cube's shape."""
    a, dates = ref.load(config, "TSI")
    nb, ny, nx = a.shape
    v = a.reshape(nb, ny * nx).T.astype(np.float64)
    v[v == NODATA] = np.nan
    return v, pd.to_datetime(dates, format="%Y%m%d"), (ny, nx)


def trunc16(x):
    """``(short)x`` of a float array as FORCE writes it: toward zero; NaN is nodata."""
    x = np.asarray(x, dtype=np.float64)
    out = np.trunc(np.nan_to_num(x, nan=NODATA))
    return np.clip(out, -32768, 32767).astype(np.int32)


def _scaled(x, scale=1000.0, limit=30000.0):
    return np.clip(x * scale, -limit, limit)


def report(title, names, got, want, skip=None, show=16):
    """Prints the agreement of ``got`` with FORCE's ``want`` (both ``(bands, ...)`` ints), band by band when there
    are at most ``show`` bands, else the worst one; ``skip`` marks the pixels left out."""
    got = np.asarray(got).reshape(len(names), -1).astype(np.int64)
    want = np.asarray(want).reshape(len(names), -1).astype(np.int64)
    keep = np.ones(got.shape, bool) if skip is None else ~np.asarray(skip).reshape(got.shape)
    diff = np.abs(got - want)
    n = keep.sum(1)
    equal = ((diff == 0) & keep).sum(1)
    near = ((diff <= 1) & keep).sum(1)
    worst = np.where(keep, diff, 0).max(1)
    pct = lambda k: np.where(n > 0, 100.0 * k / np.maximum(n, 1), 100.0)  # a band left out entirely counts as agreeing
    total_left = int((~keep).sum())
    if len(names) <= show:
        print()
    print(f"{title}   ({len(names)} bands x {got.shape[1]} pixels; {total_left} values left out: undefined in FORCE or a known, explained difference)")
    if len(names) <= show:
        print(f"  {'band':<10}{'equal %':>10}{'within 1 %':>12}{'max |dif|':>11}{'left out':>10}")
        for i, name in enumerate(names):
            print(f"  {name:<10}{pct(equal)[i]:>10.3f}{pct(near)[i]:>12.3f}{worst[i]:>11d}{int((~keep[i]).sum()):>10d}")
    else:
        i = int(np.argmin(pct(near)))
        print(f"  all bands: equal {100.0 * equal.sum() / max(n.sum(), 1):.3f} %, within 1 "
              f"{100.0 * near.sum() / max(n.sum(), 1):.3f} %, max |dif| {worst.max()}; "
              f"worst band {names[i]}: within 1 {pct(near)[i]:.3f} %, max |dif| {worst[i]}")
    if near.sum() < n.sum():
        _failed.append(title)


# ---- STM -----------------------------------------------------------------------------------------------------------

def part_stm(jobs):
    from zeit import _core

    v, _, _ = tsi()
    names = ["min", "p1", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean", "std", "skew", "kurtosis", "count"]
    t0 = time.perf_counter()
    m = dict(zip(names, np.asarray(_core.tsa.stm_batch(v, names, 1.0, jobs)).T))
    seconds = time.perf_counter() - t0
    # IQR as FORCE writes it: both quantiles truncated first
    q25, q75 = trunc16(m["p25"]), trunc16(m["p75"])
    got = np.stack([
        trunc16(m["min"]), trunc16(m["p1"]), trunc16(m["p10"]), q25, trunc16(m["p50"]), q75, trunc16(m["p90"]),
        trunc16(m["p99"]), trunc16(m["max"]), trunc16(m["mean"]), trunc16(m["std"]), trunc16(m["max"] - m["min"]),
        q75 - q25, trunc16(_scaled(m["skew"])), trunc16(_scaled(m["kurtosis"])), trunc16(m["count"]),
    ])
    want, bands = ref.load("rbf_stm", "STM")
    assert bands == STM_BANDS, bands
    n = m["count"]
    undefined_moment = (n == 1)
    constant = np.isnan(m["skew"]) & (n > 1)  # M2 = 0
    skip = np.zeros(got.shape, bool)
    for name in ("STD", "SKW", "KRT"):
        skip[STM_BANDS.index(name)] |= undefined_moment
    for name in ("SKW", "KRT"):
        skip[STM_BANDS.index(name)] |= constant
    print(f"STM: Zeit {seconds * 1000:.1f} ms for {v.shape[0]} pixels x {v.shape[1]} dates ({jobs} threads); "
          f"{int(undefined_moment.sum())} pixels with one value, {int(constant.sum())} constant")
    report("STM (rbf_stm)", STM_BANDS, got, want, skip)


# ---- fold ----------------------------------------------------------------------------------------------------------

def _fold_conversion(kind, m):
    """The int16 FORCE writes for ``FOLD_TYPE=kind`` from Zeit's fold metrics (name -> ``(groups, pixels)``)."""
    n = m["count"]
    got = {
        "AVG": lambda: trunc16(m["mean"]), "Q50": lambda: trunc16(m["p50"]), "STD": lambda: trunc16(m["std"]),
        "MIN": lambda: trunc16(m["min"]), "MAX": lambda: trunc16(m["max"]),
        "SKW": lambda: trunc16(_scaled(m["skew"])), "KRT": lambda: trunc16(_scaled(m["kurtosis"])),
        "IQR": lambda: trunc16(m["p75"] - m["p25"]), "Q25": lambda: trunc16(m["p25"]), "NUM": lambda: n.astype(np.int32),
    }[kind]()
    got = np.where(n == 0, NODATA, got)
    skip = np.zeros(got.shape, bool)
    if kind in ("STD", "SKW", "KRT"):
        skip |= n == 1
    if kind in ("SKW", "KRT"):
        skip |= np.isnan(m["skew"]) & (n > 1)
    return got, skip


def part_fold(jobs):
    from zeit import _core
    from zeit._tsa import _periods

    v, dates, _ = tsi()
    names = ["mean", "p25", "p50", "p75", "std", "min", "max", "skew", "kurtosis", "count"]
    folds = {}
    seconds = 0.0
    for product, by in FOLD_PRODUCTS.items():
        group, coord = _periods(pd.DatetimeIndex(dates), by)
        t0 = time.perf_counter()
        out = np.asarray(_core.tsa.fold_batch(v, group, len(coord), names, 1.0, jobs))  # (pixels, metrics, groups)
        seconds += time.perf_counter() - t0
        folds[product] = {name: out[:, k, :].T for k, name in enumerate(names)}  # (groups, pixels)
    print(f"fold: Zeit {seconds * 1000:.1f} ms for the five folds of {v.shape[0]} pixels x {v.shape[1]} dates "
          f"({jobs} threads)")
    for kind in FOLD_TYPES:
        for product in FOLD_PRODUCTS:
            want, bands = ref.load(f"fold_{kind}", product)
            got, skip = _fold_conversion(kind, folds[product])
            assert got.shape[0] == len(bands), (product, got.shape, len(bands))
            report(f"fold {kind:<3} {product} (fold_{kind})", bands, got, want, skip, show=12 if VERBOSE else 0)


# ---- standardize ---------------------------------------------------------------------------------------------------

def part_standardize(jobs):
    from zeit import _core

    # NORMALIZE on the TSI: FORCE's std is in units of 1e-4, the output the z-score times 10000
    v, dates, _ = tsi()
    z = np.asarray(_core.tsa.standardize_batch(v, "normalize", 1.0, jobs)).T  # (time, pixels)
    got = trunc16(np.clip(z * 10000.0, -32768, 32767))
    want, bands = ref.load("standardize", "TSI")
    skip = np.broadcast_to(np.isnan(z).all(0), z.shape)
    report("NORMALIZE of the TSI (standardize)", bands, got, want, skip, show=0)
    # CENTER on the folds of the average, from the unstandardized ones of rbf_fold
    for product in ("FBY", "FBQ", "FBM"):
        folded, bands = ref.load("rbf_fold", product)
        x = folded.reshape(len(bands), -1).T.astype(np.float64)
        x[x == NODATA] = np.nan
        c = np.asarray(_core.tsa.standardize_batch(x, "center", 1.0, jobs)).T
        want, bands = ref.load("standardize", product)
        report(f"CENTER of {product} (standardize)", bands, trunc16(c), want, None, show=12)


# ---- interpolation -------------------------------------------------------------------------------------------------

def _agreement(title, got, want, skip=None):
    """The ``report`` plus the share within +-2 and the number of pixels x dates beyond +-1 (for the harmonic fit,
    where two IRLS can stop at slightly different points)."""
    got, want = np.asarray(got).astype(np.int64), np.asarray(want).astype(np.int64)
    keep = np.ones(got.shape, bool) if skip is None else ~np.asarray(skip)
    diff = np.abs(got - want)[keep]
    n = max(int(keep.sum()), 1)
    print(f"  {title}: within 2 {100.0 * (diff <= 2).sum() / n:.4f} %, beyond 1: {int((diff > 1).sum())} of {n}, "
          f"beyond 2: {int((diff > 2).sum())}, max |dif| {int(diff.max()) if diff.size else 0}")


# The one pixel of the synthetic cube (row-major index) whose bisquare IRLS does not converge when fit on 2018-2020: it
# cycles between four predictions (1365-1409 at the last date) and stops after the 100 iterations at the point GSL's
# iteration cap, rounding and the order of the sums happen to give, which differs among FORCE, GSL in numpy and Zeit.
# The others (9999 of 10000) agree within +-1.
CYCLING_PIXELS = (3250,)


def part_interp(jobs):
    from zeit import _core

    # FORCE's TSS (int16, -9999 missing) at its own dates and calendar (365 days a year, so period 365)
    # (the measurement cube's folder has the TSS in "baseline" and only some of the configurations)
    a, dates = ref.load("int_none" if (ref.REF / "out" / "int_none").exists() else "baseline", "TSS")
    nb, ny, nx = a.shape
    v = a.reshape(nb, ny * nx).T.astype(np.float64)
    v[v == NODATA] = np.nan
    t_in = np.array([ref.ce_of(str(d)[:8]) for d in dates], dtype=np.float64)  # "20180101_SEN2A"
    first, last = ref.ce_of(20180101), ref.ce_of(20221231)
    grid = ref.tsi_grid(first, last, 16).astype(np.float64)
    fit_all = (float(first), float(last))
    fit_part = (float(ref.ce_of(20180101)), float(ref.ce_of(20201231)))
    configs = [
        ("int_linear", dict(method="linear")),
        ("int_moving", dict(method="moving", window=16.0)),
        ("int_rbf", dict(method="rbf", sigma=[8.0, 16.0, 32.0], cutoff=0.95)),
        ("int_harmonic", dict(method="harmonic", modes=3, trend=True, fit_lo=fit_all[0], fit_hi=fit_all[1],
                              period=365.0)),
        ("int_harmonic_part", dict(method="harmonic", modes=3, trend=True, fit_lo=fit_part[0], fit_hi=fit_part[1],
                                   period=365.0)),
    ]
    configs = [c for c in configs if (ref.REF / "out" / c[0]).exists()]
    print(f"interpolation: {v.shape[0]} pixels x {v.shape[1]} dates -> {len(grid)} dates ({jobs} threads)")
    for config, kw in configs:
        t0 = time.perf_counter()
        out = np.asarray(_core.tsa.interp_batch(v, t_in, grid, n_jobs=jobs, **kw))  # (pixels, grid)
        seconds = time.perf_counter() - t0
        want, bands = ref.load(config, "TSI")
        assert len(bands) == len(grid), (config, len(bands), len(grid))
        print(f"{config}: Zeit {seconds * 1000:.1f} ms")
        got = trunc16(out.T)
        skip = None
        if config == "int_harmonic_part":
            skip = np.zeros(got.shape, bool)
            skip[:, list(CYCLING_PIXELS)] = True
        report(f"TSI ({config})", bands, got, want, skip, show=0)
        if kw["method"] == "harmonic":
            _agreement("TSI", got, want.reshape(len(bands), -1), skip)
        if config == "int_harmonic_part":
            # the NRT product: TSS - prediction for the dates after the fit, only where TSS is valid
            late = t_in > fit_part[1]
            pred = np.asarray(_core.tsa.interp_batch(v, t_in, t_in[late], n_jobs=jobs, **kw))
            nrt = trunc16((v[:, late] - pred).T)
            want_nrt, nbands = ref.load(config, "NRT")
            assert len(nbands) == int(late.sum()), (len(nbands), int(late.sum()))
            skip = np.zeros(nrt.shape, bool)
            skip[:, list(CYCLING_PIXELS)] = True
            report("NRT (int_harmonic_part)", nbands, nrt, want_nrt, skip, show=0)
            _agreement("NRT", nrt, want_nrt.reshape(len(nbands), -1), skip)


# ---- trend and change (TRY, TRM, CAY, CAM) --------------------------------------------------------------------------

# FORCE's trend and CAT, in plain Python, as the C is written (H/trend-hl.c, C/stats-cl.c, GPL-3, David Frantz): the
# same recurrences and int16 casts, Hill's approximate critical t (iterated to 1e-4 in p) and, with CHANGE_PENALTY, the
# penalized change of a candidate compared with the best one's *unpenalized* change, in float32. It is the check that the
# few pixels where Zeit's result differs (the exact critical t near the limit; the corrected penalty) differ for that and
# nothing else: with it the 100x100 pixels of FORCE's outputs come out equal.
TREND_BANDS = ["MEAN", "OFFSET", "SLOPE", "PCT_GAIN", "ABS_GAIN", "RSQ", "SIG", "RMSE", "MAE", "MAXRES", "NUM", "LENGTH"]
CAT_BANDS = (["CHANGE", "LOSS", "YEAR-OF-CHANGE"]
             + [f"{part}-{name.replace('_', '-')}" for part in ("TOTAL", "BEFORE", "AFTER") for name in TREND_BANDS])
# (fold of the fold_AVG run, its trend product, its change product, the period of each fold step)
TREND_FOLDS = (("FBY", "TRY", "CAY", lambda i: 2018 + i), ("FBM", "TRM", "CAM", lambda i: 1 + i))
CONF = np.float32(0.95)


def _short(x):
    """C's ``(short)x`` of a double on x86: truncation, wrapping to 16 bits; NaN and the out of range give 0."""
    if x != x or abs(x) >= 2.0 ** 31:
        return 0
    return ((int(x) + 32768) & 0xFFFF) - 32768


def _hill_inv_t(p, df):
    import math

    a = 1.0 / (df - 0.5)
    b = 48.0 / (a * a)
    c = ((20700.0 * a / b - 98.0) * a - 16.0) * a + 96.36
    d = ((94.5 / (b + c) - 3.0) / b + 1.0) * math.sqrt(a * math.pi * 0.5) * df
    if df == 1:
        return math.cos(p * math.pi / 2) / math.sin(p * math.pi / 2)
    if df == 2:
        return math.sqrt(2.0 / (p * (2.0 - p)) - 2.0)
    y = (d * p) ** (2.0 / df)
    if y > 0.05 + a:
        pz = 0.5 * (1.0 - p)
        if pz > 0.42:
            r = math.sqrt(-math.log(0.5 - pz))
            x = (((2.3212128 * r + 4.8501413) * r - 2.2979648) * r - 2.7871893) / ((1.6370678 * r + 3.5438892) * r + 1.0)
        else:
            r = pz * pz
            x = pz * (((-25.4410605 * r + 41.3911977) * r - 18.6150006) * r + 2.5066282) / (
                (((3.1308291 * r - 21.0622410) * r + 23.0833674) * r - 8.4735109) * r + 1.0)
        y = x * x
        if df < 5:
            c = c + 0.3 * (df - 4.5) * (x + 0.6)
        c = (((0.05 * d * x - 5.0) * x - 7.0) * x - 2.0) * x + b + c
        y = (((((0.4 * y + 6.3) * y + 36.0) * y + 94.5) / c - y - 3.0) / b + 1.0) * x
        y = a * y * y
        y = math.exp(y) - 1.0 if y > 0.002 else 0.5 * y * y + y
        return math.sqrt(df * y)
    y = ((1.0 / (((df + 6.0) / (df * y) - 0.089 * d - 0.822) * (df + 2.0) * 3.0) + 0.5 / (df + 4.0)) * y - 1.0) * (
        df + 1.0) / (df + 2.0) + 1.0 / y
    return math.sqrt(df * y)


def _hill_t_p(t, df):
    import math

    abst, tsq = abs(t), t * t
    if df == 1:
        return 1.0 - 2.0 * math.atan(abst) / math.pi
    if df == 2:
        return 1.0 - abst / math.sqrt(tsq + 2.0)
    if df == 3:
        return 1.0 - 2.0 * (math.atan(abst / math.sqrt(3.0)) + abst * math.sqrt(3.0) / (tsq + 3.0)) / math.pi
    if df == 4:
        return 1.0 - abst * (1.0 + 2.0 / (tsq + 4.0)) / math.sqrt(tsq + 4.0)
    a9 = df - 0.5
    b9 = 48.0 * a9 * a9
    t9 = t * t / df
    z8 = a9 * math.log(1.0 + t9) if t9 >= 0.04 else a9 * (((1.0 - t9 * 0.75) * t9 / 3.0 - 0.5) * t9 + 1.0) * t9
    p7 = ((0.4 * z8 + 3.3) * z8 + 24.0) * z8 + 85.5
    b7 = 0.8 * z8 ** 2 + 100.0 + b9
    z = (1.0 + (-p7 / b7 + z8 + 3.0) / b9) * math.sqrt(z8)
    az = abs(z)
    q = (((((0.0000053830 * az + 0.0000488906) * az + 0.0000380036) * az + 0.0032776263) * az + 0.0211410061) * az
         + 0.0498673470) * az + 1.0
    return q ** -16


def hill_critical_t(df, p=None):
    """FORCE's critical t of the two-tailed test (``tscore``): Hill's approximation, adjusted until its p is within
    1e-4 of the level."""
    p0 = float(np.float32(1.0) - CONF) if p is None else p
    p1, diff, t = p0, 1.0, 0.0
    while abs(diff) > 0.0001:
        t = _hill_inv_t(p1, df)
        diff = _hill_t_p(t, df) - p0
        p1 -= diff
    return t


def _fit(y, lo, hi, plen, crit, whole, nf):
    """FORCE's trend of the valid values of ``y[lo:hi]`` (x = the index): the 12 int16 of a trend or of a part of the
    change, NODATA where FORCE leaves them. ``whole``: trend(), not a part of cat()."""
    out = [NODATA] * 12
    k = mx = my = vx = vy = cv = 0.0
    for f in range(lo, hi):
        if y[f] != y[f]:
            continue
        k += 1.0
        if k == 1.0:
            mx, my = float(f), y[f]
        else:
            omx, omy = mx, my
            mx = omx + (f - omx) / k
            my = omy + (y[f] - omy) / k
            vx += (f - omx) * (f - mx)
            vy += (y[f] - omy) * (y[f] - my)
            cv += (k - 1.0) / k * (f - omx) * (y[f] - omy)
    if k < 1 or (whole and k < 3):
        return out
    out[0], out[10], out[11] = _short(my), _short(k), _short(plen if not whole else nf)
    if k < 3:
        return out
    cv, vx, vy = cv / (k - 1.0), vx / (k - 1.0), vy / (k - 1.0)
    slp = cv / vx
    off = my - slp * mx
    with np.errstate(all="ignore"):
        rsq = (cv * cv / (vx * vy) if vx * vy != 0 else float("nan")) * 10000.0
    ssqe = sae = sxsq = 0.0
    maxe = -32768.0
    for f in range(lo, hi):
        if y[f] != y[f]:
            continue
        e = y[f] - (off + slp * f)
        ssqe += e * e
        sae += abs(e)
        sxsq += (f - mx) * (f - mx)
        if abs(e) > maxe:
            maxe = abs(e)
    seb = (1.0 / (k - 2.0) * ssqe) ** 0.5 / sxsq ** 0.5
    with np.errstate(all="ignore"):
        t = np.float32(slp) / np.float32(seb)
    tc = np.float32(crit[int(k) - 2])
    sig = -1 if t <= -tc else 1 if t >= tc else 0
    plen = nf if whole else plen
    yhat = off + slp * (0 if whole else lo)
    if whole:
        gain = _short(slp * nf / off * 100.0) if off > 0 else 0
    else:
        gain = _short(slp * plen / yhat * 100.0) if yhat > 0 else 0
    out[1], out[2] = _short(yhat if not whole else off), _short(slp)
    out[3], out[4], out[5], out[6] = gain, _short(slp * plen), _short(rsq), sig
    out[7], out[8], out[9] = _short((ssqe / k) ** 0.5), _short(sae / k), _short(maxe)
    return out


def _change_step(y, penalty, bug=True, f32=True):
    """The step of the greatest change and its change: FORCE's loop as written (``bug``: the penalized change against
    the best one's real change) or with the penalized ones compared with each other; in float32 or double."""
    ft = np.float32 if f32 else float
    best = ft(-32768.0)
    best_real, f_change = None, 0
    nf = len(y)
    for f in range(1, nf):
        if y[f] != y[f]:
            continue
        pre = next((g for g in range(f - 1, -1, -1) if y[g] == y[g]), None)
        if pre is None:
            continue
        now = real = ft(y[pre] - y[f])
        if now > 0 and penalty:
            post = next((g for g in range(f + 1, nf) if y[g] == y[g]), None)
            if post is None:
                continue
            now = ft(float(now) * (float(y[pre] - y[post]) / 1e4))
        if now > best:
            best = real if bug else now
            best_real, f_change = real, f
    return (None, 0) if best_real is None else (f_change, int(best_real))


def literal_trend(v, crit):
    """FORCE's trend (12 int16 per row) of the rows of ``v`` (pixels, folds), NaN missing."""
    nf = v.shape[1]
    return np.array([_fit(row, 0, nf, nf, crit, True, nf) for row in v.tolist()], dtype=np.int32)


def literal_cat(v, crit, penalty, period):
    """FORCE's CAT (39 int16 per row) of the rows of ``v``; ``period(i)``: the YEAR_OF_CHANGE of fold step ``i``."""
    nf = v.shape[1]
    out = np.full((v.shape[0], 39), NODATA, dtype=np.int32)
    for p, y in enumerate(v.tolist()):
        f, change = _change_step(y, penalty)
        if f is None:
            continue
        out[p, 0], out[p, 2] = change, period(f)
        total = _fit(y, 0, nf, nf, crit, False, nf)
        before = _fit(y, 0, f - 1, f - 1, crit, False, nf)
        after = _fit(y, f, nf, nf - f, crit, False, nf)
        out[p, 3:15], out[p, 15:27], out[p, 27:39] = total, before, after
        out[p, 1] = _short(np.float32(change) / np.float32(total[1]) * np.float32(100)) if total[1] > 0 else 0
    return out


def _to_force(a, names, scale=None):
    """Zeit's float result ``(pixels, bands)`` as the int16 FORCE writes: truncation, NaN as NODATA; R-squared times
    10000."""
    a = np.array(a, dtype=np.float64)
    for band, factor in (scale or {}).items():
        a[:, names.index(band)] *= factor
    return trunc16(a)


def part_trend(jobs):
    """Zeit's trend and CAT of FORCE's average folds (FBY, FBM of fold_AVG: the RBF TSI, AVG) against TRY/TRM and
    CAY/CAM of rbf_trend_nopen and rbf_trend_pen. A difference is "known" when it is the exact critical t against Hill's
    near the limit (SIG), or the corrected penalty (another f_change); the literal FORCE in Python must then give
    FORCE's value."""
    from zeit import _core

    for fold, tr_product, ca_product, period in TREND_FOLDS:
        a, _ = ref.load("fold_AVG", fold)
        nf = a.shape[0]
        v = a.reshape(nf, -1).T.astype(np.float64)
        v[v == NODATA] = np.nan
        crit = {df: hill_critical_t(df) for df in range(1, nf)}
        crit_list = [float("nan")] + [crit[df] for df in range(1, nf - 1)]
        from scipy import stats

        exact = {df: stats.t.ppf(1 - 0.05 / 2, df) for df in range(1, nf - 1)}
        print(f"\n== {fold}: {v.shape[0]} pixels x {nf} folds; critical t of df 1..{nf - 2}, Hill's (FORCE) | exact (Zeit): "
              + ", ".join(f"{crit[d]:.4f} | {exact[d]:.4f}" for d in range(1, nf - 1)))

        # ---- trend
        t0 = time.perf_counter()
        z = np.asarray(_core.tsa.trend_batch(v, 0.95, "two", jobs))
        seconds = time.perf_counter() - t0
        got = _to_force(z, TREND_BANDS, {"RSQ": 10000.0}).T
        want, bands = ref.load("rbf_trend_nopen", tr_product)
        assert bands == TREND_BANDS, bands
        want = want.reshape(12, -1)
        lit = literal_trend(v, crit_list).T
        print(f"{tr_product}: Zeit {seconds * 1000:.2f} ms for {v.shape[0]} pixels x {nf} folds ({jobs} threads)")
        report(f"{tr_product} literal FORCE in Python (check of the port)", bands, lit, want)
        _compare_with_known_causes(f"{tr_product} (rbf_trend_nopen)", bands, got, want, lit,
                                   pixel_cause=None, sig_rows=[bands.index("SIG")], v=v, rsq_rows=[bands.index("RSQ")])

        # ---- change, without and with the penalty
        for config, penalty in (("rbf_trend_nopen", False), ("rbf_trend_pen", True)):
            t0 = time.perf_counter()
            z = np.asarray(_core.tsa.cat_batch(v, penalty, 0.95, "two", jobs))
            seconds = time.perf_counter() - t0
            z = z.copy()
            idx = z[:, 2]
            z[:, 2] = np.where(np.isnan(idx), np.nan, [period(int(i)) if i == i else 0 for i in idx])
            got = _to_force(z, CAT_BANDS, {f"{p}-RSQ": 10000.0 for p in ("TOTAL", "BEFORE", "AFTER")}).T
            want, bands = ref.load(config, ca_product)
            assert bands == CAT_BANDS, bands
            want = want.reshape(39, -1)
            lit = literal_cat(v, crit_list, penalty, period).T
            fixed = [_change_step(row, penalty, bug=False)[0] for row in v.tolist()]
            fixed = np.array([NODATA if f is None else period(f) for f in fixed])
            print(f"{ca_product}: Zeit {seconds * 1000:.2f} ms ({config}, {jobs} threads)")
            report(f"{ca_product} literal FORCE in Python (check of the port)", bands, lit, want, show=0)
            _compare_with_known_causes(f"{ca_product} ({config})", bands, got, want, lit,
                                       pixel_cause=(penalty, fixed, period),
                                       sig_rows=[i for i, b in enumerate(bands) if b.endswith("-SIG")], v=v,
                                       rsq_rows=[i for i, b in enumerate(bands) if b.endswith("-RSQ")])


def _compare_with_known_causes(title, names, got, want, lit, pixel_cause, sig_rows, v, rsq_rows):
    """Reports ``got`` (Zeit) against ``want`` (FORCE) band by band and counts what differs beyond +-1, separating the
    values explained by a known cause: the significance where the literal FORCE (Hill) gives FORCE's value and Zeit's
    differs; whole pixels of the change where Zeit's f_change is not FORCE's because of the corrected penalty."""
    got, want, lit = (np.asarray(x, dtype=np.int64) for x in (got, want, lit))
    nb, npx = got.shape
    undefined = np.zeros(got.shape, bool)
    for r in rsq_rows:  # R-squared of a constant series: 0/0, which FORCE casts as it happens
        undefined[r] = (got[r] == NODATA) & (want[r] != NODATA)
    outside = (np.abs(got - want) > 1) & ~undefined
    # (2) the corrected penalty: the pixels whose change is another step than FORCE's (the whole pixel differs)
    pen_pixels = np.zeros(npx, bool)
    if pixel_cause is not None:
        penalty, fixed, period = pixel_cause
        year_row = names.index("YEAR-OF-CHANGE")
        differs = got[year_row] != want[year_row]
        pen_pixels = differs & (lit[year_row] == want[year_row])
        n_unexplained_step = int((differs & ~pen_pixels).sum())
        not_fixed = int((fixed != got[year_row]).sum())
        print(f"  change step differs from FORCE's in {int(differs.sum())} pixels: {int(pen_pixels.sum())} where the literal "
              f"FORCE (penalized against the best one's real change, float32) gives FORCE's step, "
              f"{n_unexplained_step} unexplained; Zeit's step is not that of the corrected comparison in float32 in "
              f"{not_fixed} pixels")
    # (1) the significance: Zeit's exact critical t against Hill's, in the pixels with the same change step
    sig_known = np.zeros(got.shape, bool)
    for r in sig_rows:
        sig_known[r] = (got[r] != want[r]) & (lit[r] == want[r]) & ~pen_pixels
    known = sig_known | pen_pixels[None, :]
    # a pixel left out entirely must be one the literal FORCE reproduces in every band
    if pen_pixels.any():
        bad = int(((lit[:, pen_pixels] != want[:, pen_pixels]).any(0)).sum())
        print(f"  of those {int(pen_pixels.sum())} pixels, {bad} differ from FORCE in some band even for the literal Python")
    n_sig = int(sig_known.sum())
    if sig_rows:
        total_sig_diff = int(sum(((got[r] != want[r]) & ~pen_pixels).sum() for r in sig_rows))
        print(f"  SIG differs from FORCE (outside the pixels of another step) in {total_sig_diff} values: {n_sig} where Hill's critical t (the literal FORCE) "
              f"gives FORCE's value and the exact one does not, {total_sig_diff - n_sig} unexplained")
    print(f"  beyond +-1 in {int(outside.sum())} values; {int((outside & known).sum())} of them from those causes, "
          f"{int((outside & ~known).sum())} left")
    report(title + ", the known causes left out", names, got, want, skip=known | undefined, show=16 if len(names) <= 16 else 0)
    off = (got != want) & ~known & ~undefined
    if off.any():
        print("  values off by 1 (the rest of the bands equal): "
              + ", ".join(f"{names[r]} {int(off[r].sum())}" for r in range(nb) if off[r].any()))


# ---- clear-sky observations (CSO) ----------------------------------------------------------------------------------

# FORCE's CSO in plain Python, as the C is written (H/cso-hl.c, C/stats-cl.c, GPL-3, David Frantz): the intervals from the
# window's start through the valid observations to its end, Pébay's recurrences over the k intervals, and then the
# observations' count n where the intervals' count k belongs: STD divides by n - 1, SKW and KRT by n, the quantiles are
# those of the first n intervals (in time order). With more observations than intervals (the same day twice) FORCE reads
# memory left over from before; those windows are marked and left out. A window without a valid observation: STD, SKW,
# KRT and the quantiles come out 0 (n = 0 leaves the skewness and kurtosis at 0, and GSL's quantile of no values is 0).
CSO_PRODUCTS = ["NUM", "MIN", "MAX", "RNG", "IQR", "AVG", "STD", "SKW", "KRT", "Q25", "Q50", "Q75"]
CSO_EXACT = ["NUM", "MIN", "MAX", "RNG", "AVG"]          # the same on both sides, except for quantization
CSO_BY_DESIGN = ["STD", "SKW", "KRT", "Q25", "Q50", "Q75", "IQR"]  # Zeit's k against FORCE's n
# Zeit's metric of each product
CSO_ZEIT = {"NUM": "count", "MIN": "min", "MAX": "max", "RNG": "range", "AVG": "mean", "STD": "std", "SKW": "skew",
            "KRT": "kurtosis", "Q25": "p25", "Q50": "p50", "Q75": "p75", "IQR": "iqr"}


def _div(a, b):
    """C's double division: x / 0 is inf or NaN, not an exception."""
    import math

    if b != 0:
        return a / b
    return math.nan if a == 0 or a != a else math.copysign(math.inf, a)


def _clamp(x):
    """FORCE's ``if (x < -30000) x = -30000; if (x > 30000) x = 30000``: a NaN stays NaN (Python's min and max would not)."""
    return -30000.0 if x < -30000 else 30000.0 if x > 30000 else x


def force_windows(first, last, step):
    """FORCE's windows ``[start, end)`` as continuous days (``ce``) for DATE_RANGE ``first``..``last`` (YYYYMMDD) and
    MONTH_STEP: counted while ``year < last year or month < last month``, so a window that starts in the last month is
    lost (December, with a range that ends then)."""
    y0, m0, y1, m1 = first // 10000, first // 100 % 100, last // 10000, last // 100 % 100
    count, year, month = 0, y0, m0
    while year < y1 or (year <= y1 and month < m1):
        count += 1
        month += step
        if month > 12:
            year, month = year + 1, month - 12
    out, year, month = [], y0, m0
    for _ in range(count):
        end_year, end_month = (year, month + step) if month + step <= 12 else (year + 1, month + step - 12)
        out.append((ref.ce_of(year * 10000 + month * 100 + 1), ref.ce_of(end_year * 10000 + end_month * 100 + 1)))
        year, month = end_year, end_month
    return out


def _force_window(obs, start, end):
    """One window of FORCE's CSO: the 12 int16 of ``CSO_PRODUCTS`` and whether it read stale memory (n > k). ``obs``:
    the ``ce`` of the valid observations in the window, in time order."""
    import math

    n = k = 0
    mean = var = skew = kurt = 0.0
    gaps = []
    left = start

    def push(d):
        nonlocal k, mean, var, skew, kurt
        k += 1
        delta = d - mean
        delta_n = delta / k
        delta_n2 = delta_n * delta_n
        tmp = delta * delta_n * (k - 1)
        mean += delta_n
        kurt += tmp * delta_n2 * (k * k - 3 * k + 3) + 6 * delta_n2 * var - 4 * delta_n * skew
        skew += tmp * delta_n * (k - 2) - 3 * delta_n * var
        var += tmp
        gaps.append(float(np.float32(d)))

    for ce in obs:
        n += 1
        if ce > left:
            push(ce - left)
            left = ce
    if end - left > 0:
        push(end - left)
    low, high = min(gaps), max(gaps)
    sk = kt = 0.0
    if n > 0:
        sd = math.sqrt(var / n)
        sk = _clamp(_div(skew, n * sd ** 3) * 1000)
        kt = _clamp((_div(kurt, (n * (var / n)) * (var / n)) - 3) * 1000)
    stale = n > k
    q = [0, 0, 0]
    if n > 0 and not stale:
        first_n = np.sort(np.array(gaps[:n], dtype=np.float64))
        for i, p in enumerate((0.25, 0.5, 0.75)):
            index = p * (n - 1)
            lhs = int(index)
            delta = index - lhs
            value = first_n[lhs] if lhs == n - 1 else (1 - delta) * first_n[lhs] + delta * first_n[lhs + 1]
            q[i] = _short(float(np.float32(value)))
    std = _short(math.sqrt(v) if (v := _div(var, n - 1)) >= 0 else math.nan)
    return [n, _short(low), _short(high), _short(high - low), q[2] - q[0], _short(mean), std, _short(sk), _short(kt),
            q[0], q[1], q[2]], stale


def literal_cso(valid, ce, windows):
    """FORCE's CSO (12 int16 per window) of the rows of ``valid`` (pixels, time; boolean) at the days ``ce``:
    ``(products, windows, pixels)``, and the ``(windows, pixels)`` mask of the windows that read stale memory."""
    spans = [(int(np.searchsorted(ce, a, "left")), int(np.searchsorted(ce, b, "left"))) for a, b in windows]
    out = np.zeros((len(CSO_PRODUCTS), len(windows), valid.shape[0]), np.int32)
    stale = np.zeros((len(windows), valid.shape[0]), bool)
    for p in range(valid.shape[0]):
        row = valid[p]
        for w, ((a, b), (lo, hi)) in enumerate(zip(windows, spans)):
            values, stale[w, p] = _force_window(ce[lo:hi][row[lo:hi]].tolist(), a, b)
            out[:, w, p] = values
    return out, stale


def _cso_prm_check():
    """The CSO's SCREEN_QAI and DATE_RANGE against those of the TSS that feeds the core (int_none)."""
    def lines(name):
        text = (ref.REF / "prm" / f"{name}.prm").read_text()
        return {k.strip(): v.strip() for k, _, v in (l.partition("=") for l in text.splitlines()
                                                      if "=" in l and not l.lstrip().startswith("#"))}

    tss = lines("int_none")
    for config in ("cso_m1", "cso_m3"):
        cso = lines(config)
        same = all(cso[key] == tss[key] for key in ("SCREEN_QAI", "DATE_RANGE", "DOY_RANGE"))
        print(f"{config}: SCREEN_QAI, DATE_RANGE and DOY_RANGE the same as int_none's (the TSS): {same}")
        assert same, (config, {key: (cso[key], tss[key]) for key in ("SCREEN_QAI", "DATE_RANGE", "DOY_RANGE")})


def part_cso(jobs):
    """Zeit's CSO of FORCE's TSS (valid where it is not -9999) on FORCE's own windows and ``ce`` against cso_m1 and
    cso_m3. NUM, MIN, MAX, RNG and AVG must agree (+-1, quantized); STD, SKW, KRT, the quantiles and the IQR differ by
    design (Zeit uses the k intervals where FORCE mixes in the n observations), shown against FORCE's own code in Python."""
    from zeit import _core

    _cso_prm_check()
    a, dates = ref.load("int_none", "TSS")
    nb, ny, nx = a.shape
    tss = a.reshape(nb, ny * nx).T
    valid = tss != NODATA
    ce = np.array([ref.ce_of(str(d)[:8]) for d in dates], dtype=np.int64)
    assert (np.diff(ce) >= 0).all()
    same_day = int((np.diff(ce) == 0).sum())
    clouds = ~ref.truth()["masked"].reshape(nb, -1).T
    print(f"CSO: {valid.shape[0]} pixels x {nb} dates; valid observations: {int(valid.sum())} "
          f"({int((valid != clouds).sum())} differ from the truth's mask: the TSS is also nodata where the index is "
          f"undefined, so the TSS is the mask used); dates on the same day: {same_day}")
    names = [CSO_ZEIT[p] for p in CSO_PRODUCTS]
    values = np.where(valid, 1.0, np.nan)
    for config, step in (("cso_m1", 1), ("cso_m3", 3)):
        windows = force_windows(20180101, 20221231, step)
        want = {p: ref.load(config, p) for p in CSO_PRODUCTS}
        nw = len(windows)
        for p, (arr, bands) in want.items():
            assert arr.shape[0] == nw, (config, p, arr.shape[0], nw)
        print(f"\n== {config}: {nw} windows of {step} month(s), {bands[0]} to {bands[-1]} (FORCE's bands); Zeit's own "
              f"calendar would make {nw + (1 if step == 1 else 0)} (the last window is not compared)")
        t0 = time.perf_counter()
        out = np.asarray(_core.tsa.cso_batch(values, ce.astype(np.float64), np.array(windows, dtype=np.float64), names,
                                             jobs))  # (pixels, metrics, windows)
        seconds = time.perf_counter() - t0
        print(f"{config}: Zeit {seconds * 1000:.1f} ms ({jobs} threads)")
        m = {p: out[:, i, :].T for i, p in enumerate(CSO_PRODUCTS)}  # (windows, pixels)
        n = m["NUM"]
        got = {
            "NUM": n.astype(np.int32), "MIN": trunc16(m["MIN"]), "MAX": trunc16(m["MAX"]), "RNG": trunc16(m["RNG"]),
            "AVG": trunc16(m["AVG"]), "STD": trunc16(m["STD"]), "SKW": trunc16(_scaled(m["SKW"])),
            "KRT": trunc16(_scaled(m["KRT"])), "Q25": trunc16(m["Q25"]), "Q50": trunc16(m["Q50"]),
            "Q75": trunc16(m["Q75"]), "IQR": trunc16(m["Q75"]) - trunc16(m["Q25"]),
        }
        report(f"CSO {config}: NUM, MIN, MAX, RNG, AVG", CSO_EXACT, np.stack([got[p] for p in CSO_EXACT]),
               np.stack([want[p][0].reshape(nw, -1) for p in CSO_EXACT]))
        t0 = time.perf_counter()
        lit, stale = literal_cso(valid, ce, windows)
        print(f"{config}: FORCE's code in Python {time.perf_counter() - t0:.1f} s; {int(stale.sum())} windows with more "
              f"observations than intervals (left out where they read stale memory), "
              f"{int((n == 0).sum())} windows without a valid observation")
        skip = np.stack([stale if p in ("Q25", "Q50", "Q75", "IQR") else np.zeros(stale.shape, bool)
                         for p in CSO_PRODUCTS])
        report(f"CSO {config}: FORCE's code in Python, the 12 products", CSO_PRODUCTS, lit,
               np.stack([want[p][0].reshape(nw, -1) for p in CSO_PRODUCTS]), skip)
        print(f"\n  by design: Zeit (k intervals) against FORCE (n), values of {nw} windows x {valid.shape[0]} pixels")
        print(f"  {'product':<9}{'Zeit = FORCE %':>16}{'values changed':>16}{'pixels changed':>16}"
              f"{'of them n=0':>13}{'FORCE = python %':>18}")
        for product in CSO_BY_DESIGN:
            w = want[product][0].reshape(nw, -1).astype(np.int64)
            lit_p = lit[CSO_PRODUCTS.index(product)].astype(np.int64)
            ok = ~stale if product in ("Q25", "Q50", "Q75", "IQR") else np.ones(w.shape, bool)
            changed = (got[product] != w) & ok
            print(f"  {product:<9}{100.0 * (1 - changed.sum() / ok.sum()):>16.3f}{int(changed.sum()):>16}"
                  f"{int(changed.any(0).sum()):>16}{int((changed & (n == 0)).sum()):>13}"
                  f"{100.0 * ((lit_p == w) & ok).sum() / ok.sum():>18.3f}")
        # the port of FORCE's code must give FORCE's own values, whatever differs from Zeit's
        for product in CSO_BY_DESIGN:
            w = want[product][0].reshape(nw, -1).astype(np.int64)
            ok = ~stale if product in ("Q25", "Q50", "Q75", "IQR") else np.ones(w.shape, bool)
            # SKW and KRT times 1000 land on exact integers for regular series (e.g. -500.0): FORCE's compiled C and
            # Python's evaluation order of the same doubles can fall either side of it, so those are allowed +-1.
            tolerance = 1 if product in ("SKW", "KRT") else 0
            if (np.abs(lit[CSO_PRODUCTS.index(product)] - w) > tolerance)[ok].any():
                _failed.append(f"CSO {config}: FORCE's code in Python does not give FORCE's {product}")


# The parts.
PARTS = {"stm": part_stm, "fold": part_fold, "standardize": part_standardize, "interp": part_interp,
         "trend": part_trend, "cso": part_cso}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--part", default="stm,fold,standardize", help=f"comma-separated parts of {list(PARTS)}")
    p.add_argument("--jobs", type=int, default=4, help="Zeit's threads (the machine's cap is 8 in all)")
    p.add_argument("--verbose", action="store_true", help="every band of the folds with at most 12 bands")
    p.add_argument("--ref", help="read FORCE's outputs from this folder instead of workdir('force_tsa')/ref: the "
                                 "measurement cube's bench folder (parts stm and interp: the configurations it has)")
    args = p.parse_args(argv)
    if args.ref:
        ref.use(args.ref)
    global VERBOSE
    VERBOSE = args.verbose
    parts = [s.strip() for s in args.part.split(",") if s.strip()]
    unknown = [s for s in parts if s not in PARTS]
    if unknown:
        p.error(f"unknown part {unknown}; choose from {list(PARTS)}")
    no_power_throttling()
    for name in parts:
        PARTS[name](args.jobs)
    if _failed:
        print(f"\nOUTSIDE +-1: {len(_failed)} product(s):", *_failed, sep="\n  ")
        return 1
    print("\nEvery compared value equal or within +-1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
