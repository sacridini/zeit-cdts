"""Timing of Zeit's TSA functions against FORCE 3.11.00's higher-level TSA, on the 500 x 500 measurement cube.

The FORCE side is ``force_tsa_timing.sh`` (WSL): it runs each configuration three times on the cube with 1 and with 6
compute threads and keeps FORCE's own time of the compute stage (screen_qai, the index, the TSA; no reading, no writing),
which it prints in whole seconds and ``force_time_ms.c`` makes milliseconds. It copies the series Zeit is fed with, and
``compute.tsv``, to ``workdir("force_tsa")/bench``. This script then times Zeit's core on the same arrays::

    python benchmarks/vs_originals/force_tsa_timing.py [--jobs 1 6] [--no-record]

What FORCE's time is made of (the best, minimum, of the repetitions of each configuration, in compute-stage seconds):

* interpolation (linear, moving, rbf, harmonic): the configuration's compute stage less the baseline's (no interpolation,
  just the index and the QAI screening, the same TSS written);
* STM, folds and trend+CAT: the configuration's compute stage less the one of the RBF interpolation they are built on
  (which already has the baseline in it);
* CSO: the compute stage as it is. The CSO has no baseline to take off: its input is the QAI alone, and the QAI screening
  is part of the stage (the TSA baseline's stage, with the index on top of it, is an upper bound of it).

Zeit's side: the ``zeit._core.tsa.*_batch`` functions on FORCE's series (int16, -9999 as NaN, float32): the TSS for the
interpolations and the CSO, its RBF TSI for the STM and the folds, its yearly average fold (FBY) for the trend and the CAT.
FORCE computes the yearly fold inside the TRY/CAY run, so the Zeit time of "trend+cat" includes the yearly fold of the TSI.
``best_of``: a warm-up and the best of three.
"""
import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import force_ref as ref
from common import best_of, no_power_throttling, record, workdir

SIZE_LABEL = "250,000 px x 325 dates"
ORIGINAL = "FORCE 3.11.00"
NODATA = -9999
# FORCE's int16 values are exact in float32, which the core reads as it is (the work is in float64 either way):
# half the memory traffic of the float64 copy of the cube the first measurements fed it.
DTYPE = np.float32
STM_NAMES = ["min", "p1", "p10", "p25", "p50", "p75", "p90", "p99", "max", "mean", "std", "skew", "kurtosis", "count"]
CSO_NAMES = ["count", "min", "max", "range", "iqr", "mean", "std", "skew", "kurtosis", "p25", "p50", "p75"]
BENCH = workdir("force_tsa") / "bench"


def wait_quiet(limit_minutes=30):
    """Waits (one check a minute, at most ``limit_minutes``) until no cargo, rustc or FORCE is running and the load is
    low; returns False when it does not happen."""
    import psutil

    for attempt in range(limit_minutes + 1):
        busy = [p.name() for p in psutil.process_iter(["name"])
                if (p.info["name"] or "").lower() in ("cargo.exe", "rustc.exe")]
        force = subprocess.run(["wsl", "-e", "bash", "-c", "pgrep -a 'force|gdal|cargo|rustc' || true"],
                               capture_output=True, text=True).stdout.strip()
        load = psutil.cpu_percent(interval=10)
        if not busy and not force and load < 15:
            print(f"quiet: no cargo/rustc/FORCE, load {load:.0f}%")
            return True
        print(f"busy ({busy or ''} {force} load {load:.0f}%), waiting a minute")
        time.sleep(50)
    return False


def force_times():
    """``{(config, ncomp): [compute seconds of each repetition]}`` from compute.tsv."""
    rows = [r.split("\t") for r in (BENCH / "compute.tsv").read_text().splitlines()[1:]]
    out = {}
    for config, ncomp, rep, wall, i_ms, c_ms, o_ms in rows:
        out.setdefault((config, int(ncomp)), []).append(float(c_ms) / 1000.0)
    return out


def force_isolated(ms, ncomp):
    """FORCE's isolated seconds per product (minimum of the repetitions), and the notes on how."""
    best = lambda c: min(ms[(c, ncomp)])
    base, rbf = best("baseline"), best("int_rbf")
    out = {
        "interp-linear": best("int_linear") - base, "interp-moving": best("int_moving") - base,
        "interp-rbf": rbf - base, "interp-harmonic": best("int_harmonic") - base,
        "stm": best("rbf_stm") - rbf, "fold": best("rbf_fold") - rbf, "trend+cat": best("rbf_trend_nopen") - rbf,
        "cso": best("cso_m1"),
    }
    return out, {"baseline": base, "rbf": rbf, "cso_num": best("cso_num") if ("cso_num", ncomp) in ms else None}


def load_inputs():
    """The three series of the FORCE outputs, as Zeit takes them."""
    a, dates = ref.load("baseline", "TSS")
    nb, ny, nx = a.shape
    tss = np.ascontiguousarray(a.reshape(nb, ny * nx).T, dtype=DTYPE)
    tss[tss == NODATA] = np.nan
    t_in = np.array([ref.ce_of(str(d)[:8]) for d in dates], dtype=np.float64)
    a, tsi_dates = ref.load("int_rbf", "TSI")
    tsi = np.ascontiguousarray(a.reshape(a.shape[0], -1).T, dtype=DTYPE)
    tsi[tsi == NODATA] = np.nan
    tsi_dates = pd.to_datetime(tsi_dates, format="%Y%m%d")
    a, _ = ref.load("rbf_fold", "FBY")
    fby = np.ascontiguousarray(a.reshape(a.shape[0], -1).T, dtype=np.float64)  # 5 values a pixel: not worth the dtype
    fby[fby == NODATA] = np.nan
    return tss, t_in, tsi, tsi_dates, fby


def zeit_functions(tss, t_in, tsi, tsi_dates, fby):
    """``{product: f(n_jobs)}`` the calls to time."""
    from zeit import _core
    from zeit._tsa import _periods

    first, last = ref.ce_of(20180101), ref.ce_of(20221231)
    grid = ref.tsi_grid(first, last, 16).astype(np.float64)
    fit = (float(first), float(last))
    interp = {
        "interp-linear": dict(method="linear"),
        "interp-moving": dict(method="moving", window=16.0),
        "interp-rbf": dict(method="rbf", sigma=[8.0, 16.0, 32.0], cutoff=0.95),
        "interp-harmonic": dict(method="harmonic", modes=3, trend=True, fit_lo=fit[0], fit_hi=fit[1], period=365.0),
    }
    fns = {p: (lambda jobs, kw=kw: _core.tsa.interp_batch(tss, t_in, grid, n_jobs=jobs, **kw)) for p, kw in interp.items()}
    fns["stm"] = lambda jobs: _core.tsa.stm_batch(tsi, STM_NAMES, 1.0, jobs)
    groups = {by: _periods(pd.DatetimeIndex(tsi_dates), by) for by in ("year", "quarter", "month", "week", "doy")}

    def fold(jobs):
        for group, coord in groups.values():
            _core.tsa.fold_batch(tsi, group, len(coord), ["mean"], 1.0, jobs)

    fns["fold"] = fold
    year_group, year_coord = groups["year"]

    def trend_cat(jobs):
        _core.tsa.fold_batch(tsi, year_group, len(year_coord), ["mean"], 1.0, jobs)  # FORCE folds inside the run
        _core.tsa.trend_batch(fby, 0.95, "two", jobs)
        _core.tsa.cat_batch(fby, False, 0.95, "two", jobs)

    fns["trend+cat"] = trend_cat
    ce = t_in
    windows = np.array(_force_windows(), dtype=np.float64)
    values = np.where(np.isnan(tss), np.nan, 1.0).astype(DTYPE)
    fns["cso"] = lambda jobs: _core.tsa.cso_batch(values, ce, windows, CSO_NAMES, jobs)
    return fns


def _force_windows():
    from force_tsa import force_windows

    return force_windows(20180101, 20221231, 1)


def profile(inputs, jobs_list):
    """Where Zeit's time goes in the products it does not win: the time of the pieces at 1/8, 1/4, 1/2 and all of the
    pixels (a cost that does not scale with the pixels is fixed, one that does is per pixel), and what filling and
    allocating the output alone costs (``np.full`` of the output's size, which the core does serially before the threads)."""
    from zeit import _core
    from zeit._tsa import _periods

    tss, t_in, tsi, tsi_dates, fby = inputs
    first, last = ref.ce_of(20180101), ref.ce_of(20221231)
    grid = ref.tsi_grid(first, last, 16).astype(np.float64)
    group, coord = _periods(pd.DatetimeIndex(tsi_dates), "year")
    n = tss.shape[0]
    print("\nprofile: seconds (best of 3) by number of pixels; per 1000 pixels in ms in brackets")
    for jobs in jobs_list:
        print(f"\n{jobs} thread(s)")
        header = "pixels".ljust(10) + "".join(f"{k:>20}" for k in
                                             ("np.full out", "linear", "moving", "year fold", "trend", "cat"))
        print(header)
        for frac in (8, 4, 2, 1):
            m = n // frac
            s, ti, f = tss[:m], tsi[:m], fby[:m]
            times = [
                best_of(lambda: np.full((m, len(grid)), np.nan)),
                best_of(lambda: _core.tsa.interp_batch(s, t_in, grid, method="linear", n_jobs=jobs)),
                best_of(lambda: _core.tsa.interp_batch(s, t_in, grid, method="moving", window=16.0, n_jobs=jobs)),
                best_of(lambda: _core.tsa.fold_batch(ti, group, len(coord), ["mean"], 1.0, jobs)),
                best_of(lambda: _core.tsa.trend_batch(f, 0.95, "two", jobs)),
                best_of(lambda: _core.tsa.cat_batch(f, False, 0.95, "two", jobs)),
            ]
            print(f"{m:<10}" + "".join(f"{t:>12.4f} ({1000 * t / m * 1000:5.2f})" for t in times))


NOTES = {
    "interp-linear": "FORCE: compute stage less the baseline's",
    "interp-moving": "FORCE: compute stage less the baseline's",
    "interp-rbf": "FORCE: compute stage less the baseline's",
    "interp-harmonic": "FORCE: compute stage less the baseline's",
    "stm": "FORCE: compute stage less the RBF run's (16 metrics)",
    "fold": "FORCE: compute stage less the RBF run's (5 folds, AVG)",
    "trend+cat": "FORCE: compute stage less the RBF run's (TRY+CAY, no penalty); Zeit: yearly fold + trend + CAT",
    "cso": "FORCE: whole compute stage, QAI screening included (no baseline to take off); 1-month windows",
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--jobs", type=int, nargs="+", default=[1, 6], help="Zeit's n_jobs (and FORCE's NTHREAD_COMPUTE)")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--no-wait", action="store_true", help="do not wait for a quiet machine")
    p.add_argument("--only", nargs="+", help="products to time")
    p.add_argument("--profile", action="store_true", help="only the profile of the slower products by size")
    args = p.parse_args(argv)
    ref.use(BENCH)
    no_power_throttling()
    ms = force_times()
    if not args.no_wait and not wait_quiet():
        print("the machine stayed busy for 30 minutes; not measuring")
        return 2
    print("loading FORCE's outputs of the measurement cube ...")
    inputs = load_inputs()
    print(f"TSS {inputs[0].shape}, TSI {inputs[2].shape}, FBY {inputs[4].shape}")
    if args.profile:
        profile(inputs, args.jobs)
        return 0
    fns = zeit_functions(*inputs)
    rows = []
    for jobs in args.jobs:
        isolated, parts = force_isolated(ms, jobs)
        spread = {c: (min(v), float(np.median(v)), max(v)) for (c, n), v in ms.items() if n == jobs}
        print(f"\nFORCE, {jobs} compute threads: compute stage min/median/max over repetitions (s):")
        for c, (lo, mid, hi) in spread.items():
            print(f"  {c:<18}{lo:8.3f}{mid:8.3f}{hi:8.3f}")
        for product, fn in fns.items():
            if args.only and product not in args.only:
                continue
            t_zeit = best_of(lambda: fn(jobs))
            t_force = isolated[product]
            rows.append((product, jobs, t_force, t_zeit))
            note = NOTES[product]
            if product == "cso" and parts["cso_num"] is not None:
                note += f"; the NUM-only CSO run's stage is {parts['cso_num']:.2f} s, the TSA baseline's {parts['baseline']:.2f} s"
            if t_force <= 0:
                print(f"  {product}: FORCE's isolated time {t_force:.3f} s is not positive: not recorded")
                continue
            if args.no_record:
                print(f"{product:<18} {jobs} threads: FORCE {t_force:8.3f} s   zeit {t_zeit:8.4f} s   x{t_force / t_zeit:,.1f}")
            else:
                record("force-tsa " + product, ORIGINAL, SIZE_LABEL, t_force, t_zeit, jobs, note)
    print("\n| product | threads | FORCE (s) | Zeit (s) | x |")
    for product, jobs, tf, tz in rows:
        print(f"| {product} | {jobs} | {tf:.3f} | {tz:.4f} | {tf / tz:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
