"""Timing of ``zeit.tasseled_cap`` against FORCE 3.11.00's TC-BRIGHT, TC-GREEN, TC-WET and TC-DI, on a Landsat 8 cube of 500 x 500 pixels.

The FORCE side is ``force_tsa_timing.sh`` with ``SENSOR=LND08 SIZE=500 YEARS="2018 2019"`` and the configurations ``tc`` (the four
components as indices) and ``tc_src`` (the six input bands as indices: the same screening and copying of the TSS, no
arithmetic), three times with 1 and with 6 compute threads, FORCE's own time of the compute stage (screen_qai, the index, no
reading and no writing; ``force_time_ms.c`` makes its seconds milliseconds); it copies ``compute.tsv`` and the cube's values
(``spec_src.npz``) to ``workdir("force_specadjust")/bench``. This script times Zeit on the same values::

    python benchmarks/vs_originals/force_tc_timing.py [--jobs 1 6] [--no-record] [--no-wait]

FORCE's time is the minimum of the repetitions of ``tc`` less that of ``tc_src`` (the screening and the TSS bricks are common;
``tc`` keeps four bands and ``tc_src`` six, a difference that stays in FORCE's time and favours FORCE's number being too high).
Zeit's is the best of three of ``zeit.tasseled_cap`` (four components, int16 in, float32 out, ``nodata=None``) on the whole cube,
the observations FORCE screens out included (9% of them), on that many threads (the compiled core, over tiles of 4096 pixels).
"""
import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

from common import best_of, no_power_throttling, record, workdir

ORIGINAL = "FORCE 3.11.00"
BENCH = workdir("force_specadjust") / "bench"
COMPONENTS = ["brightness", "greenness", "wetness", "disturbance"]


def force_seconds(ncomp):
    rows = [r.split("\t") for r in (BENCH / "compute.tsv").read_text().splitlines()[1:]]
    pick = lambda cfg: [float(r[5]) / 1000.0 for r in rows if r[0] == cfg and int(r[1]) == ncomp]
    return pick("tc"), pick("tc_src")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--jobs", type=int, nargs="+", default=[1, 6])
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args(argv)
    no_power_throttling()
    if not args.no_wait:
        from force_tsa_timing import wait_quiet

        if not wait_quiet():
            print("the machine stayed busy for 30 minutes; not measuring")
            return 2
    import zeit

    values = np.load(BENCH / "spec_src.npz")["values"]            # (dates, 6, y, x) int16
    nt, _, ny, nx = values.shape
    cube = xr.DataArray(values, dims=("time", "band", "y", "x"),
                        coords={"time": pd.date_range("2018-01-01", periods=nt, freq="5D"),
                                "band": ["blue", "green", "red", "nir08", "swir16", "swir22"]})
    keep = ~np.load(BENCH / "truth.npz")["masked"]
    print(f"{ny * nx} pixels x {nt} dates = {ny * nx * nt:,} pixel-dates ({100 * keep.mean():.1f}% of them kept by FORCE's screening)")
    for jobs in args.jobs:
        tc, src = force_seconds(jobs)
        t_force = min(tc) - min(src)
        print(f"FORCE, {jobs} compute threads: tc {sorted(tc)}, tc_src {sorted(src)} -> {t_force:.3f} s (tc alone {min(tc):.3f} s)")
        t_zeit = best_of(lambda: zeit.tasseled_cap(cube, COMPONENTS, nodata=None, n_jobs=jobs))
        rate = ny * nx * nt / t_zeit / 1e6
        print(f"  Zeit {t_zeit:.3f} s ({rate:.1f} M pixel-dates/s), gain x{t_force / t_zeit:.1f}")
        size = f"250,000 px x {nt} dates"
        note = ("FORCE: compute stage of TC-BRIGHT TC-GREEN TC-WET TC-DI less the six bands as indices (same screening); Zeit: "
                "tasseled_cap, four components, int16 to float32, the whole cube (FORCE screens 9% out)")
        if not args.no_record and t_force > 0:
            record("force-tsa tasseled cap", ORIGINAL, size, t_force, t_zeit, jobs, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
