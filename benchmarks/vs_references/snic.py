"""SNIC with the Rust core. The script behind the published SNIC rows (512 x 512 x 4 against snic.c; 2048 x 2048 x 6
in 16 tiles) is not kept anywhere (repository, history, cdts_vs_originals), so the workloads are rebuilt from the
table's description:
- single core: a 512 x 512 image of 4 bands (smooth float64 patterns with noise, as benchmarks/bench_core.py's
  SNIC workload), seeds on a rectangular grid of spacing 10 (2,601), compactness 10 (snic.c's default), the whole
  image at once (zeit.segmentation.run_snic with n_jobs=1): the mean and the best of 5 runs after a warm-up.
- multi-core: a 2048 x 2048 image of 6 bands, the same kind of data and seeds, tile_size=512 (16 tiles), n_jobs=1
  then n_jobs=8, one cold run each, then the warm best of 3.
With --reference, the authors' snic.c (pinned commit, downloaded and built as tests/data/make_snic_parity_fixtures.py
does) is timed on the same single-core image with the same number of seeds, only to tell whether the rebuilt
workload is comparable with the published one (the stored 42.3 ms stays the reference).

    python benchmarks/vs_references/snic.py [--reference]
"""
import common  # first: thread variables

import argparse
import ctypes
import importlib.util
import tempfile
import time

import numpy as np

from zeit.segmentation import run_snic, snic_grid
from common import REPO, THREADS, once, pin, record, timed, warm_best

PUBLISHED = {"single": (18.4, 42.3), "tiled": (482.0, 59.0)}


def image(side, bands, seed=42):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:side, 0:side] / side
    data = np.stack([np.sin(6 * x + k) * np.cos(4 * y - k) for k in range(bands)])
    return np.ascontiguousarray(data + rng.normal(0, 0.05, data.shape))


def runs(fn, n=5):
    fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(t)), float(np.min(t))


def build_reference():
    spec = importlib.util.spec_from_file_location("fixtures", REPO / "tests" / "data" / "make_snic_parity_fixtures.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    work = tempfile.mkdtemp(prefix="snic-ref-")  # the downloaded code in a folder of its own
    return fx.build_reference(work), work


def reference_time(ref, img, n_seeds):
    f, h, w = img.shape
    labels = np.empty(h * w, dtype=np.int32)
    num = ctypes.c_int(0)
    buf = np.ascontiguousarray(img, dtype=np.float64).copy()

    def go():
        b = buf.copy()
        ref.SNIC_main(b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), w, h, f, n_seeds, 10.0, 0,
                      labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), ctypes.byref(num))
    copy_ms = runs(lambda: buf.copy())[0]
    mean, best = runs(go)
    return mean - copy_ms, best - copy_ms, num.value


def main(reference):
    img = image(512, 4)
    seeds = snic_grid((512, 512), 10)
    pin(1)
    z_mean, z_best = timed(lambda: runs(lambda: run_snic(img, seeds=seeds, compactness=10.0, n_jobs=1)), "SNIC 512")
    res = run_snic(img, seeds=seeds, compactness=10.0, n_jobs=1)
    ref_cal = None
    if reference:
        lib, work = build_reference()  # outside the timing: the compiler is another process
        r_mean, r_best, numlabels = timed(lambda: reference_time(lib, img, len(seeds)), "snic.c")
        ref_cal = {"snic_c_mean_ms": r_mean, "snic_c_best_ms": r_best, "snic_c_labels": numlabels,
                   "zeit_over_snic_c": z_mean / r_mean, "build_dir": work}
    pin(None)
    pz, pr = PUBLISHED["single"]
    record({"table": "single-core", "row": "SNIC", "order": 10,
            "workload": f"512x512x4 float64 (rebuilt), {len(seeds)} grid seeds, compactness 10, one pass, run_snic",
            "unit": "ms/pass", "published_cpp_1t": pz, "rust_1t": z_mean, "rust_1t_best": z_best,
            "ratio_rust_cpp_1t": z_mean / pz, "reference_1t": pr, "reference_tool": "Achanta & Susstrunk snic.c",
            "speedup_vs_reference_1t": pr / z_mean, "published_speedup": 2.3, "reference_calibration": ref_cal,
            "agreement": {"labelled_px": int((res.labels >= 0).sum()), "segments": int((res.sizes > 0).sum())},
            "notes": "the producing script of the published row is not kept: workload rebuilt from the table's "
                     "description, so the ratio to the published C++ value is indicative only; "
                     "reference_calibration (with --reference) times snic.c on this same image: if snic.c takes "
                     "much longer than the recorded 42.3 ms, the rebuilt image is heavier than the published one and "
                     "zeit_over_snic_c (against the published 18.4/42.3 = 0.435) is the comparable figure. Label "
                     "parity with snic.c is covered by tests/test_snic.py."})

    big = image(2048, 6)
    gseeds = snic_grid((2048, 2048), 10)
    out = {}

    def go(n):
        out[n] = run_snic(big, seeds=gseeds, compactness=10.0, tile_size=512, n_jobs=n).labels

    pin(1)
    t1 = timed(lambda: once(lambda: go(1)), "tiled 1T") * 1000
    pin(THREADS)
    t8 = timed(lambda: once(lambda: go(THREADS)), "tiled 8T") * 1000
    pin(1)
    w1 = timed(lambda: warm_best(lambda: go(1)), "tiled 1T warm") * 1000
    pin(THREADS)
    w10 = timed(lambda: warm_best(lambda: go(THREADS)), "tiled 8T warm") * 1000
    pin(None)
    p1, pm = PUBLISHED["tiled"]
    record({"table": "multi-core", "row": "SNIC (tiled)", "order": 3,
            "workload": f"2048x2048x6 float64 (rebuilt), {len(gseeds)} grid seeds, tile_size 512 (16 tiles)",
            "unit": "ms", "published_cpp_1t": p1, "published_cpp_multi": pm, "published_threads": 20,
            "rust_1t": t1, "rust_8t": t8, "rust_1t_warm_best3": w1, "rust_8t_warm_best3": w10,
            "ratio_rust_cpp_1t": t1 / p1, "ratio_rust8_cpp19": t8 / pm,
            "agreement": {"labels_1t_equal_8t": bool(np.array_equal(out[1], out[THREADS]))},
            "notes": "workload rebuilt (see the single-core row); the published thread count is given as 20 in the "
                     "table. rust_1t / rust_8t: one cold run each, then the warm best of 3. No reference tool "
                     "(snic.c has no parallel mode)."})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true", help="also time snic.c on the single-core image")
    a = ap.parse_args()
    common.start()
    main(a.reference)
