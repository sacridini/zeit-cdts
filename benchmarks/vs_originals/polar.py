"""zeit.polar_metrics against the authors' R package, PolarMetrics 0.1.0 (Brooks, Lee, Hargrove and Pomara; the
method's own code, GPL (>= 3)), per pixel: ``calc_metrics`` on a regular series (a sample every 5 days, 73 a year,
5 years: the package takes a whole number of samples per cycle) of seasonal pixels with noise and a late and an
early peak (so the phenological year wraps over the new year), the metrics they share.

    python benchmarks/vs_originals/polar.py [--pixels 5000] [--jobs 8]

The package is installed from its GitHub tarball with ``R CMD INSTALL`` (pure R: depends on ``zoo``).  The metrics in
common, and what each means in the two (FORCE's names; Zeit is run with ``adaptive=False``, the package's fixed year):

====== ========= ====================================================================================
R      Zeit      definition
====== ========= ====================================================================================
es     DSS       first step with the accumulated fraction above ``lcut`` (R) / at least ``start`` (Zeit)
ls     DES       first step with the fraction at least ``hcut`` / ``end``
s_intv LGS       ``ls - es``
s_avg  VGM       mean of the series over the steps ``es .. ls - 1``
s_sd   VGV       its sample standard deviation
s_mag  VAV       length of the mean vector of those steps
ms     DAV       day of the year of that mean vector's angle
a_avg  IBT       mean over one year of the series
====== ========= ====================================================================================

What differs (each shown below with a port of the package's code in Python, ``port_r``):

1. The year.  R starts the phenological year at the sample ``av_idx = ceiling(av_ang * spc / 2 pi)`` and takes
   ``spc`` samples from it; FORCE (so Zeit) at the first sample on or after the day of the year ``av_ang`` points
   to (within one step).  With the days of the year 1, 6, ..., 361 (not 0, 5, ..., 360) the two are one sample
   apart for the pixels whose angle falls in the last three days of a step.
2. The windows.  R (0.1.0) takes the cycle's accumulated fractions with the *rotated* positions, but reads the
   series and its vectors at those positions *unrotated* (``v[es2ls]`` with ``es2ls`` a position of the
   rotated accumulation), so the vectors, their mean and ``a_avg`` are of other steps than the dates it finds
   (``port_r(..., rotate=False)`` is that; ``rotate=True`` reads the steps the accumulation is of).
3. The dates.  R adds ``av_doy`` (the day of the angle) to the position's day; Zeit/FORCE use the sample's own day.
4. ``ems`` and ``lms`` split the window at the sample where the angle passes the mean vector's (``ms``); FORCE's
   early and late parts split it where the fraction reaches ``mid`` (``DEV``, ``DLV``, ``VEV``, ``VLV``): not common.
"""
import argparse
import sys
import time

import numpy as np

from zeit import _core
from common import best_of, no_power_throttling, record, run_r, threads_label, workdir

SPC, YEARS, STEP = 73, 5, 5
LCUT, HCUT = 0.2, 0.8
R_COLUMNS = ["es", "ems", "ms", "lms", "ls", "s_intv", "s_avg", "s_sd", "s_mag", "s_mag_std", "ems_mag", "lms_mag",
             "a_avg"]
_failed = []


def series(n, rng):
    """``(n, YEARS * SPC)`` float32-exact: a bell-shaped season of a random timing, width and height, a floor, a
    year-to-year jitter of the timing, noise; the days of the year of the samples (1, 6, ..., 361)."""
    doy = 1 + STEP * np.arange(SPC)
    t = np.tile(doy, YEARS)
    year = np.repeat(np.arange(YEARS), SPC)
    peak = np.where(rng.random(n) < 0.7, rng.uniform(130, 270, n), rng.uniform(-60, 60, n) % 365)  # a few early/late
    width = rng.uniform(35, 90, n)
    amp = rng.uniform(0.3, 0.8, n)
    floor = rng.uniform(0.05, 0.25, n)
    jitter = rng.normal(0, 8, (n, YEARS))
    d = np.abs(t[None, :] - (peak[:, None] + jitter[:, year]))
    d = np.minimum(d, 365 - d)
    v = floor[:, None] + amp[:, None] * np.exp(-0.5 * (d / width[:, None]) ** 2) + rng.normal(0, 0.015, (n, len(t)))
    return np.maximum(v, 0.0).astype(np.float32), t


def rseq(a, b):
    """R's ``a:b`` (1-based, inclusive, descending when ``b < a``) as 0-based positions."""
    return (np.arange(a, b + 1) if b >= a else np.arange(a, b - 1, -1)) - 1


def port_r(v, t, spc=SPC, lcut=LCUT, hcut=HCUT, start_idx=None, rotate=False):
    """A literal port of PolarMetrics 0.1.0's ``calc_metrics`` (timing ``from_vecs_integer``, ``cal_yr``) for one
    pixel: ``(years, 13)`` in ``R_COLUMNS`` order.  ``start_idx`` replaces ``av_idx`` (the first step of the year, 1 to
    ``spc``); ``rotate`` reads the series and the vectors at the steps of the rotated year (the package reads them at the
    rotated positions, unrotated)."""
    v = np.asarray(v, dtype=np.float64)
    r = 2 * np.pi * (t % 365) / 365
    vx_, vy_ = v * np.cos(r), v * np.sin(r)
    rv_ang = np.arctan2(vy_.mean(), vx_.mean()) % (2 * np.pi)
    av_ang = (rv_ang + np.pi) % (2 * np.pi)
    av_doy = int(365 * av_ang / (2 * np.pi))
    av_idx = int(np.ceil(av_ang * (spc / (2 * np.pi)))) if start_idx is None else start_idx
    nyr = len(v) // spc
    if start_idx is None and av_idx % spc == 0:  # R: a year starting at the last step is the calendar year
        npy, begin = nyr, 1
    else:
        npy, begin = nyr - 1, av_idx
    pos = (begin - 1) + np.arange(npy * spc)  # the input position of each accumulated step (sum_cycle's vidx)
    cs = np.concatenate([np.cumsum(v[pos[i * spc:(i + 1) * spc]]) for i in range(npy)])
    if rotate:
        vv, rr, vx, vy = v[pos], r[pos], vx_[pos], vy_[pos]
        tt = t[pos]
    else:
        vv, rr, vx, vy, tt = v, r, vx_, vy_, t
    out = np.full((npy, 13), np.nan)

    def ang(x, y):
        return np.arctan2(y, x) % (2 * np.pi)

    for J in range(1, npy + 1):
        s0 = spc * (J - 1)
        x = cs[s0:s0 + spc]
        pr = x / x.max()
        es = s0 + 1 + int(np.argmax(pr > lcut))
        ls = s0 + 1 + int(np.argmax(pr >= hcut))
        w = rseq(es, ls - 1)
        ms_ang = ang(vx[w].mean(), vy[w].mean())
        k = int(np.argmax(rr[w] > ms_ang)) + 1  # which.max (1 when none is)
        ms_idx = k + es
        w1, w2 = rseq(es, ms_idx - 1), rseq(ms_idx, ls - 1)
        ems_ang = ang(vx[w1].mean(), vy[w1].mean())
        lms_ang = ang(vx[w2].mean(), vy[w2].mean())

        def day(a):
            return int(365 * a / (2 * np.pi))

        def cal(d):  # as.integer(ceiling((d + av_doy) %% 365)); the rotated steps carry their own day
            return int(np.ceil((d + (0 if rotate else av_doy)) % 365))

        sm = np.hypot(vx[w].mean(), vy[w].mean())
        mu = vv[w].mean()
        be = np.arange(spc * (J - 1), spc * J)  # the calendar cycle J (the rotated year's, rotated)
        out[J - 1] = [cal(tt[es - 1]), cal(day(ems_ang)), cal(day(ms_ang)), cal(day(lms_ang)), cal(tt[ls - 1]),
                      (tt[ls - 1] - tt[es - 1]) % 365 if rotate else tt[ls - 1] - tt[es - 1], mu, vv[w].std(ddof=1), sm, sm / mu,
                      np.hypot(vx[w1].mean(), vy[w1].mean()), np.hypot(vx[w2].mean(), vy[w2].mean()), vv[be].mean()]
    return np.vstack([out, np.full((YEARS - npy, 13), np.nan)])  # a pixel with av_idx = spc has YEARS cycles


def zeit_polar(v, t, jobs):
    """Zeit's core with a fixed year: ``(pixels, metrics, years)`` and the metrics' names (the dates are days from
    1 January of the band's year)."""
    from zeit._tsa import POLAR_NAMES

    ce = (np.arange(v.shape[1]) * STEP + 1).astype(np.int64)
    out = np.asarray(_core.tsa.polar_batch(v, ce, STEP, list(POLAR_NAMES), LCUT, 0.5, HCUT, False, jobs))
    return out, list(POLAR_NAMES)


def zeit_season_start(v, t):
    """FORCE's first step of each pixel's phenological year (1-based; ``SPC + 1`` is the next year's): the first sample on
    or after ``theta.doy`` (and within a step of it), by the day FORCE gives each step (truncated from a float32 angle)."""
    r = 2 * np.pi * (t % 365) / 365
    ang = np.arctan2((v * np.sin(r)).mean(1), (v * np.cos(r)).mean(1))
    ang = np.where(ang <= 0, ang + 2 * np.pi, ang)
    theta = np.where(ang < np.pi, ang + np.pi, ang - np.pi)
    doy = (theta * 365 / (2 * np.pi)).astype(int)  # theta.doy, truncated
    # FORCE's day of each step is truncated from its float32 angle: a day that falls just short counts one day less
    day = (((t[:SPC] / 365.0 * 2.0 * np.pi).astype(np.float32).astype(np.float64)) * 365.0 / (2.0 * np.pi)
           ).astype(np.float32).astype(int)
    first = day[None, :] >= doy[:, None]
    idx = np.where(first.any(1), first.argmax(1) + 1, SPC + 1)  # past the last step: the next year's first
    return idx, doy


def compare(title, got, want, mod=None, keep=None, tol=0.0):
    """Share equal, share within ``tol`` and the largest |difference| of two ``(pixels, years)`` arrays; ``mod`` makes it
    circular."""
    d = np.abs(np.asarray(got, float) - np.asarray(want, float))
    if mod:
        d = np.minimum(d, mod - d)
    if keep is None:
        keep = np.ones(d.shape, bool)
    keep = keep & np.isfinite(d)
    n = int(keep.sum())
    print(f"  {title:<18}{100.0 * ((d == 0) & keep).sum() / max(n, 1):>9.3f}{100.0 * ((d <= tol) & keep).sum() / max(n, 1):>9.3f}"
          f"{float(d[keep].max()) if n else 0.0:>14.6g}{n:>9d}")
    return 100.0 * ((d <= tol) & keep).sum() / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    no_power_throttling()
    n = args.pixels
    v, t = series(n, np.random.default_rng(23))
    d = workdir("polar")
    vp, tp = d / f"values_{n}.csv", d / f"doys_{n}.csv"
    np.savetxt(vp, v.astype(np.float64), delimiter=",", fmt="%.9g")
    np.savetxt(tp, t, delimiter=",", fmt="%d")
    t_r = run_r("polar.R", vp, tp, SPC, LCUT, HCUT)
    rm = np.loadtxt(d / f"values_{n}_metrics.csv", delimiter=",", skiprows=1)
    years = YEARS - 1
    rm = rm[np.argsort(rm[:, 0], kind="stable")]
    pix = rm[:, 0].astype(int) - 1
    ref = np.full((n, YEARS, 13), np.nan)
    ref[pix, rm[:, 1].astype(int) - 1] = rm[:, 3:] if rm.shape[1] == 16 else rm[:, 2:]
    npy = years
    ref = ref[:, :npy]

    # 1. the port against the package: the same numbers
    lit = np.array([port_r(v[i], t) for i in range(n)])
    print(f"\nthe port of calc_metrics in Python against the package ({n} pixels x {npy} years)")
    print(f"  {'metric':<18}{'equal %':>9}{'in tol %':>9}{'max |dif|':>14}{'values':>9}")
    for k, name in enumerate(R_COLUMNS):
        compare(name, lit[:, :npy, k], ref[:, :, k], tol=1e-9 if name in ("s_avg", "s_sd", "s_mag", "s_mag_std", "ems_mag",
                                                                           "lms_mag", "a_avg") else 0.0)
    ok_port = all(np.nanmax(np.abs(lit[:, :npy, k] - ref[:, :, k])) < 1e-6 for k in range(13))
    if not ok_port:
        _failed.append("the port of the package differs from the package")

    # 2. Zeit (fixed year) against the package, over the metrics they share
    threads = threads_label(args.jobs)
    out, names = zeit_polar(v, t, args.jobs)
    zy = out.shape[2]
    start, theta_doy = zeit_season_start(v, t)
    r_idx = np.array([int(np.ceil(((np.arctan2(((v[i] * np.sin(2 * np.pi * (t % 365) / 365)).mean()),
                                                 ((v[i] * np.cos(2 * np.pi * (t % 365) / 365)).mean())) + np.pi) %
                                    (2 * np.pi)) * SPC / (2 * np.pi))) for i in range(n)])
    r_idx = np.where(r_idx == 0, SPC, r_idx)
    print(f"\nthe first step of the phenological year: Zeit/FORCE's and R's av_idx agree in "
          f"{100.0 * np.mean(start == r_idx):.1f}% of the pixels (the others are one step apart: "
          f"{100.0 * np.mean(np.abs(start - r_idx) % SPC == 1):.1f}%)")
    # the year band of a season: s if the start's day is before 182, else s + 1; NaN where there is none
    shift = (theta_doy >= 182).astype(int)
    def zeit_metric(name, scale=None):
        a = out[:, names.index(name), :]
        res = np.full((n, npy), np.nan)
        for s in range(npy):
            res[:, s] = a[np.arange(n), s + shift]
        return res
    agree = (start == r_idx) & (r_idx != SPC)  # R's year at the last step is the calendar year: a different year
    print(f"  (R's av_idx = {SPC}, a calendar year: {100.0 * np.mean(r_idx == SPC):.1f}% of the pixels)")
    pairs = [("es", "DSS", 365), ("ls", "DES", 365), ("s_intv", "LGS", None), ("ms", "DAV", 365), ("s_avg", "VGM", None),
             ("s_sd", "VGV", None), ("s_mag", "VAV", None), ("a_avg", "IBT", None)]

    def zeit_pair(zname, mod):
        z = zeit_metric(zname)
        return z % 365 if mod else z

    # a day is truncated from a float32 angle in Zeit/FORCE (one day short now and then) and R's day is its position's
    # plus the angle's day (up to a step, 4 days, from the sample's); the values are float32 sums in Zeit
    print("\nZeit (adaptive=False) against the package (dates within 4 days; values within 1e-4)")
    print(f"  {'metric':<18}{'equal %':>9}{'in tol %':>9}{'max |dif|':>14}{'values':>9}")
    for rname, zname, mod in pairs:
        k = R_COLUMNS.index(rname)
        z = zeit_pair(zname, mod)
        tol = 4.0 if mod else (1.0 if rname == "s_intv" else 1e-4)
        print(f"{rname} / {zname}")
        compare("  all pixels", z, ref[:, :, k], mod=mod, tol=tol)
        compare("  same year start", z, ref[:, :, k], mod=mod, keep=np.broadcast_to(agree[:, None], z.shape), tol=tol)

    # 3. the port reading the steps the accumulation is of (rotate=True): the package's year, then FORCE's
    print("\nthe port reading the rotated year's steps (rotate=True), against Zeit: with R's own start of the year "
          "(pixels where it is Zeit's) and with Zeit's (all pixels)")
    print(f"  {'metric':<18}{'equal %':>9}{'in tol %':>9}{'max |dif|':>14}{'values':>9}")
    own = np.array([port_r(v[i], t, rotate=True) for i in range(n)])
    fixed = np.array([port_r(v[i], t, start_idx=int(start[i]), rotate=True) for i in range(n)])
    for rname, zname, mod in pairs:
        k = R_COLUMNS.index(rname)
        z = zeit_pair(zname, mod)
        tol = 1.0 if (mod or rname == "s_intv") else 1e-4
        print(f"{rname} / {zname}")
        for label, got, keep in (("  R's start", own, np.broadcast_to(agree[:, None], z.shape)),
                                 ("  Zeit's start", fixed, None)):
            if compare(label, got[:, :npy, k], z, mod=mod, keep=keep, tol=tol) < 100.0:
                _failed.append(f"the port with the rotated steps differs from Zeit in {rname} ({label.strip()})")

    # 4. times
    t_z = best_of(lambda: zeit_polar(v, t, args.jobs))
    record("polar", "PolarMetrics 0.1.0 (R)", f"{n} pixels x {v.shape[1]} dates", t_r, t_z, threads,
           "metrics in common; the year and the steps it reads differ, see polar.py")
    if args.jobs != 1:
        t_z1 = best_of(lambda: zeit_polar(v, t, 1))
        record("polar", "PolarMetrics 0.1.0 (R)", f"{n} pixels x {v.shape[1]} dates", t_r, t_z1, 1,
               "metrics in common; the year and the steps it reads differ, see polar.py")
    if _failed:
        print("\nFAILED:", *_failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
