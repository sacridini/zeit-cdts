"""Timing of Zeit's spectral adjustment against FORCE 3.11.00's SPECTRAL_ADJUST, on a Landsat 8 cube of 500 x 500 pixels.

The FORCE side is ``force_tsa_timing.sh`` with ``SENSOR=LND08 SIZE=500 YEARS="2018 2019"`` and the configurations
``spec_off spec_on`` (WSL, one workdir of its own): the same TSA run with ``SPECTRAL_ADJUST = FALSE`` (the six source bands as
indices) and ``TRUE`` (the ten adjusted bands), three times with 1 and with 6 compute threads, FORCE's own time of the
compute stage (screen_qai, the adjustment, the index, no reading and no writing; ``force_time_ms.c`` makes its seconds
milliseconds). It copies ``compute.tsv`` and the cube's values (``spec_src.npz``) to ``workdir("force_specadjust")/bench``.
This script then times Zeit's core (``zeit._core.specadjust.specadjust_batch``) on the same values::

    python benchmarks/vs_originals/force_specadjust_timing.py [--jobs 1 6] [--no-record] [--no-wait]

FORCE's time is the minimum of the repetitions of ``spec_on`` less that of ``spec_off`` (the same screening and index
copy, the adjustment off; spec_on writes ten bands and spec_off six, a difference of a few tenths of a second that stays in
FORCE's time). Zeit's is the best of three after a warm-up on the observations FORCE processes (those the QAI lets through,
91% of the pixel-dates), as float32, which the core reads as it is.
"""
import argparse
import sys

import numpy as np

from common import best_of, no_power_throttling, record, workdir

ORIGINAL = "FORCE 3.11.00"
NODATA = -9999
BENCH = workdir("force_specadjust") / "bench"


def force_seconds(ncomp):
    rows = [r.split("\t") for r in (BENCH / "compute.tsv").read_text().splitlines()[1:]]
    pick = lambda cfg: [float(r[5]) / 1000.0 for r in rows if r[0] == cfg and int(r[1]) == ncomp]
    return pick("spec_on"), pick("spec_off")


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
    from zeit import _core

    src = np.load(BENCH / "spec_src.npz")
    values = src["values"]                                      # (dates, 6, y, x)
    nt, _, ny, nx = values.shape
    # The observations FORCE adjusts are those its QAI screening lets through; the cube's generator keeps what it masks
    # (truth.npz, nodata swaths, clouds, their buffers and shadows): 90.8% of the pixel-dates kept, FORCE's own count on
    # the 200 x 200 cube of the accuracy run (force_tsa.py --part specadjust: 4,937,286 of 5,440,000, which this mask
    # reproduces to 0.04%).
    keep = ~np.load(BENCH / "truth.npz")["masked"]              # (dates, y, x)
    flat = values.transpose(0, 2, 3, 1)[keep].astype(np.float32)
    del values
    print(f"{ny * nx} pixels x {nt} dates; {flat.shape[0]} pixel-dates to adjust ({100 * keep.mean():.1f}% of the cube)")
    for jobs in args.jobs:
        on, off = force_seconds(jobs)
        t_force = min(on) - min(off)
        print(f"FORCE, {jobs} compute threads: spec_on {sorted(on)}, spec_off {sorted(off)} -> {t_force:.3f} s")
        t_zeit = best_of(lambda: _core.specadjust.specadjust_batch(flat, "LND08", jobs))
        size = f"250,000 px x {nt} dates"
        note = (f"FORCE: compute stage of SPECTRAL_ADJUST TRUE (10 bands as indices) less FALSE (6 bands); Zeit: specadjust_batch "
                f"on the {flat.shape[0]:,} unmasked pixel-dates as float32")
        rate = flat.shape[0] / t_zeit / 1e6
        print(f"  Zeit {t_zeit:.3f} s ({rate:.2f} M pixel-dates/s), gain x{t_force / t_zeit:.1f}")
        if not args.no_record and t_force > 0:
            record("force-tsa specadjust", ORIGINAL, size, t_force, t_zeit, jobs, note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
