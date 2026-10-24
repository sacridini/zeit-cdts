"""zeit's S-CCD (what zeit.sccd runs: _core.sccd.sccd_batch) against pyxccd's sccd_detect, the authors'
implementation (Apache-2.0: run as a black box in its own environment, never read), and against zeit's own
CCDC on the same pixels (S-CCD's claim: a Kalman update per observation instead of a lasso refit).

The pixels are 4096 of a Landsat Collection 2 stack of Rondonia (125 dates 2020-2026, four 32 x 32 windows;
tests/data/make_sccd_pyxccd_fixtures.py reads the same stack), on the dates with data, and 1000 synthetic
pixels of 35 years every 8 days (1597 dates, a third cloudy: the long series S-CCD was made for). pyxccd runs
pixel by pixel, as its API does; zeit on 1 thread and on all the processors but one.

    python benchmarks/vs_originals/sccd.py CCDC_Stack_Rondonia.tif CCDC_Stack_Rondonia_Dates.csv

with ``PYXCCD_PYTHON`` the python of an environment with pyxccd (``pip install pyxccd``).
"""
import os
import subprocess
import sys

import numpy as np

from zeit import _core
from zeit._ccdc import run_ccdc_batch
from common import HERE, best_of, record, threads_label, workdir

sys.path.insert(0, str(HERE.parent.parent / "tests" / "data"))

CORNERS = [(300, 300), (800, 800), (1300, 400), (500, 1300)]
SIZE = 32

PYX = '''
import sys, time
import numpy as np
from pyxccd import sccd_detect
d = np.load(sys.argv[1])
dates, values, qa = d["dates"].astype(np.int64), d["values"].astype(np.int64), d["qa"].astype(np.int64)
def run():
    out = []
    for p in range(len(values)):
        v = values[p]
        out.append(sccd_detect(dates, v[0], v[1], v[2], v[3], v[4], v[5], qa[p]))
    return out
run()
times = []
for _ in range(3):
    t0 = time.perf_counter(); out = run(); times.append(time.perf_counter() - t0)
breaks = np.zeros((len(out), 8), np.int64)
for p, r in enumerate(out):
    b = sorted(int(s["t_break"]) for s in np.atleast_1d(r.rec_cg))[:8]
    breaks[p, :len(b)] = b
np.save(sys.argv[2], breaks)
print("SECONDS", min(times))
'''


def pixels(tif, csv):
    import pandas as pd
    import rasterio
    from rasterio.windows import Window
    from make_sccd_pyxccd_fixtures import fmask

    dates = pd.read_csv(csv)["Ordinal_Day"].to_numpy().astype(np.int32)
    nt = len(dates)
    vals, qas = [], []
    with rasterio.open(tif) as src:
        for r, c in CORNERS:
            a = src.read(window=Window(c, r, SIZE, SIZE)).reshape(nt, 7, SIZE * SIZE)
            vals.append(a[:, :6])
            qas.append(a[:, 6])
    dn, qp = np.concatenate(vals, axis=2), np.concatenate(qas, axis=1)
    sr = dn * 0.275 - 2000.0
    missing = np.isnan(qp) | np.isnan(sr).any(axis=1) | ((np.nan_to_num(qp).astype(np.int64) & 1) == 1)
    qa = np.where(missing, 255, fmask(np.nan_to_num(qp, nan=1.0)))
    values = np.where(missing[:, None], 0, np.round(np.nan_to_num(sr)))
    return dates, values.transpose(2, 1, 0).astype(np.float64), qa.T.astype(np.int32)


def synthetic(n, rng):
    """35 years of two Landsats (8 days), a third of the dates cloudy, a third of the pixels with a change."""
    import datetime

    first = datetime.date(1990, 1, 1).toordinal()
    dates = np.arange(first, datetime.date(2024, 12, 31).toordinal(), 8, dtype=np.int32)
    t = dates.astype(float)
    w = 2 * np.pi / 365.25
    values = np.empty((n, 6, len(dates)))
    for p in range(n):
        level = rng.uniform(400, 3000, 6)
        level[3] = rng.uniform(1500, 3500)  # NIR, which falls at a change
        amp = level * rng.uniform(0.05, 0.3, 6)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.uniform(15, 60)
        v = level[:, None] + amp[:, None] * np.sin(w * t + phase) + 0.3 * amp[:, None] * np.cos(2 * w * t)
        if p % 3 == 1:
            at = rng.integers(len(dates) // 5, 4 * len(dates) // 5)
            v[:, at:] += rng.uniform(300, 1200) * np.array([1, 1, 1, -0.3, 1, 1])[:, None]
        values[p] = np.clip(np.round(v + rng.normal(0, noise, v.shape)), 1, 9999)
    qa = np.where(rng.random((n, len(dates))) < 0.33, 4, 0).astype(np.int32)
    return dates, values, qa


def compare(d, name, dates, values, qa):
    n, nb, nt = values.shape
    np.savez(d / f"{name}.npz", dates=dates, values=values, qa=qa)
    (d / "pyx.py").write_text(PYX)
    python = os.environ.get("PYXCCD_PYTHON", sys.executable)
    run = subprocess.run([python, "-I", str(d / "pyx.py"), str(d / f"{name}.npz"), str(d / f"{name}_pyx.npy")],
                         capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(run.stderr[-3000:])
    t_pyx = float(next(line.split()[1] for line in run.stdout.splitlines() if line.startswith("SECONDS ")))
    pyx = np.load(d / f"{name}_pyx.npy")

    v4, q3 = values[:, None], qa[:, None]
    ms = 8
    width = 5 + nb * 8
    for threads in (1, -1):
        out = {}
        t_z = best_of(lambda: out.update(r=_core.sccd.sccd_batch(v4, q3, dates, max_segments=ms, n_jobs=threads)))
        t_c = best_of(lambda: run_ccdc_batch(dates, v4, q3, max_segments=ms, n_jobs=threads))
        r = out["r"]
        segs = r[:, :ms * width].reshape(n, ms, width)
        counts = r[:, ms * width].astype(int)
        same = within = total = 0
        for p in range(n):
            ours = sorted(int(s[2]) for s in segs[p, :counts[p]] if s[2] > 0)
            theirs = [int(t) for t in pyx[p] if t > 0]
            same += len(ours) == len(theirs)
            clear = np.sort(dates[qa[p] < 2])
            for t in theirs:
                total += 1
                within += any(abs(np.searchsorted(clear, u) - np.searchsorted(clear, t)) <= 1 for u in ours)
        note = f"same number of breaks {same / n:.0%}, pyxccd's breaks within 1 obs {within / max(total, 1):.0%}"
        size = f"{n} pixels x {nt} dates" + (" (synthetic)" if name == "synthetic" else "")
        record("sccd", "pyxccd 1.1.0 sccd_detect", size, t_pyx, t_z, threads_label(threads), note)
        record("sccd", "zeit.ccdc (CCDC, same pixels)", size, t_c, t_z, threads_label(threads),
               "S-CCD's time against CCDC's in zeit")


def main(tif, csv):
    d = workdir("sccd")
    dates, values, qa = pixels(tif, csv)
    # The stack interleaves two Landsat paths and these windows are in one: the other path's dates have no
    # data in any pixel. CCDC refuses a pixel with data on less than half of the dates (S-CCD and pyxccd do
    # not), so those dates are left out for all three.
    seen = (qa != 255).any(axis=0)
    compare(d, "rondonia", dates[seen], values[:, :, seen], qa[:, seen])
    compare(d, "synthetic", *synthetic(1000, np.random.default_rng(0)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
