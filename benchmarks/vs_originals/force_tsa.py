"""Zeit's time series analysis against FORCE 3.11.00's higher-level TSA, on the same synthetic cube.

The FORCE side (``force_tsa.sh``) wrote its outputs in ``workdir("force_tsa")/ref``; this script feeds Zeit's core
with FORCE's own series (the int16 TSI, -9999 as NaN) and compares what comes back with FORCE's int16 outputs, which
are quantized: Zeit's float result is converted the way FORCE writes it (truncation toward zero; SKW and KRT times
1000 and clamped to +-30000; the STM's IQR as trunc(q75) - trunc(q25); the standardized values times 10000).

    python benchmarks/vs_originals/force_tsa.py --part stm,fold,standardize [--jobs 4]

Parts: ``stm`` (the 16 statistics of the whole series), ``fold`` (the ten statistics by year, quarter, month, week and
day of the year), ``standardize``. The interpolation, trend and CSO parts come with their own functions in ``PARTS``.

For each product and band it prints the share of pixels that are equal, within +-1, and the largest |difference|.
Where FORCE's result is undefined (a standard deviation, skewness or kurtosis of one value, a skewness or kurtosis
of a constant series: 0/0, which FORCE casts to int16 as it happens) the pixels are left out, and the count is printed.
The target is 100% equal or within +-1; the exit code is 1 when something is outside it.
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
    print(f"{title}   ({len(names)} bands x {got.shape[1]} pixels; {total_left} values left out as undefined in FORCE)")
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


# The parts; the interpolation, trend and CSO comparisons are added here.
PARTS = {"stm": part_stm, "fold": part_fold, "standardize": part_standardize}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--part", default="stm,fold,standardize", help=f"comma-separated parts of {list(PARTS)}")
    p.add_argument("--jobs", type=int, default=4, help="Zeit's threads (the machine's cap is 8 in all)")
    p.add_argument("--verbose", action="store_true", help="every band of the folds with at most 12 bands")
    args = p.parse_args(argv)
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
