"""Timing of ``zeit.improphe`` against FORCE 3.11.00's CFIMP and L2IMP on the 153 x 153 measurement cube (10 m pixels, 2 years of
Sentinel-2A dates; L2: 4 Landsat 8 scenes of 30 m).

The FORCE side is ``force_improphe_timing.sh`` (WSL): each configuration (CF or L2, radius 10, 20 and 50 pixels) three times
with 1 and with 6 compute threads, FORCE's own compute stage kept (screening of the QAI, seasonal means, PCA, standardization,
textures and the kernel; no reading, no writing), in milliseconds through ``force_time_ms.c``. This script times Zeit's whole call
(``zeit.improphe(..., force=True)`` on the screened cube read from the same files: the float32 conversion, the regridding of the
coarse data, the seasonal means, the PCA, the textures and the kernel; the best of three after a warm-up) with 1 and 6 threads::

    python benchmarks/vs_originals/force_improphe_timing.py [--jobs 1 6] [--no-record]
"""
import argparse
import sys

import numpy as np

import force_improphe as fi
from common import best_of, no_power_throttling, record, workdir
from force_tsa_timing import wait_quiet

ORIGINAL = "FORCE 3.11.00"
BENCH = workdir("force_improphe") / "bench"
RADII = (10, 20, 50)


def force_times():
    rows = [r.split("\t") for r in (BENCH / "compute.tsv").read_text().splitlines()[1:]]
    out = {}
    for config, ncomp, rep, wall, i_ms, c_ms, o_ms in rows:
        out.setdefault((config, int(ncomp)), []).append(float(c_ms) / 1000.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, nargs="+", default=[1, 6])
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    no_power_throttling()
    if not a.no_wait and not wait_quiet():
        print("no quiet window; not measuring")
        return 1
    import zeit

    fi.DATA = BENCH
    ms = force_times()
    fine = fi.load_fine()
    fld = fi.load_field()
    l8, _ = fi.load_l8()
    n = fine.sizes["y"]
    size_cf = f"{n}x{n} px, 2 years, {fine.sizes['time']} dates"
    size_l2 = f"{n}x{n} px, {l8.sizes['time']} scenes, {fine.sizes['time']} dates"
    for h in RADII:
        for jobs in a.jobs:
            kw = dict(seasons=fi.WINDOWS["w4"], kernel_size=h * 10.0, texture_size=fi.KSD_M, force=True, nodata=None, n_jobs=jobs)
            for mod, coarse, by, size in (("cf", fld, "year", size_cf), ("l2", l8, None, size_l2)):
                key = (f"{mod}_h{h}", jobs)
                if key not in ms:
                    print(f"no FORCE time for {key}")
                    continue
                t_f = min(ms[key])
                t_z = best_of(lambda: zeit.improphe(coarse, fine, by=by, **kw))
                print(f"{mod} h={h} threads={jobs}: FORCE {t_f:.3f} s (min of {len(ms[key])}), Zeit {t_z:.3f} s, x{t_f / t_z:.2f}")
                if not a.no_record:
                    record(f"force improphe {mod}", ORIGINAL, f"{size}, radius {h} px", t_f, t_z, jobs,
                           "FORCE: compute stage, min of 3; Zeit: the whole call, best of 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
