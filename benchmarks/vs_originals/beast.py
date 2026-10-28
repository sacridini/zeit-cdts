"""zeit.beast against Rbeast 1.0.2 (R; Zhao et al. 2019, GPL >= 2): beast() on one series and beast123() (its batch,
multithreaded) on a cube, monthly with a season of 12, changes of level and slope and noise, beast()'s defaults.

Rbeast picks kernels for the processor (AVX2 here); zeit is a port of its generic C kernels ('sse'), whose float32
sums are in another order: the chains part after a few hundred iterations, so the timed runs agree in distribution, not
bit for bit. With cputype 'sse' they are the same chain: checked on the series and on the cube's first pixel (beast123
carries its random stream from one pixel to the next where zeit starts each from the seed, as beast() does).

Windows 11 slows the processes it deems in the background (power throttling, "EcoQoS": on a hybrid processor, the
efficiency cores at a low clock) after a few seconds of work, and a benchmark run from a terminal is one: zeit's threads
ran 2.4 times slower for it, Rbeast's (pinned to processors) less. Both processes are taken out of it here, so that the
times are of the computation. On one thread Rbeast pins itself to the first processor; zeit's one-thread runs are
pinned there too.

    python benchmarks/vs_originals/beast.py
"""
import contextlib

import numpy as np

from zeit import _core
from common import best_of, no_power_throttling, record, run_r, threads_label, workdir

KW = dict(period=12.0, seed=42)


def series(n, n_dates, rng):
    t = np.arange(n_dates)
    v = np.empty((n, n_dates))
    for k in range(n):
        amp, noise = rng.uniform(0.1, 0.5), rng.uniform(0.02, 0.1)
        y = 0.4 + amp * np.sin(2 * np.pi * t / 12 + rng.uniform(0, 6)) + rng.normal(0, noise, n_dates)
        if k % 2 == 0:
            t0 = rng.integers(12, n_dates - 12)
            y[t0:] += rng.uniform(-0.4, 0.4)
        if k % 3 == 0:
            t1 = rng.integers(12, n_dates - 12)
            y[t1:] += rng.uniform(-0.005, 0.005) * (t[t1:] - t1)
        y[rng.random(n_dates) < 0.05] = np.nan
        v[k] = y
    return v


@contextlib.contextmanager
def pinned(threads):
    """On one thread, the process on the first processor, as Rbeast pins its thread."""
    try:
        import psutil
    except ImportError:
        yield
        return
    proc = psutil.Process()
    before = proc.cpu_affinity()
    if threads == 1:
        proc.cpu_affinity([0])
    try:
        yield
    finally:
        proc.cpu_affinity(before)


def trend_ncp(out, layout_n):
    fields, (n, _, _) = layout_n
    off = {name: (o, shape) for name, o, shape in fields}
    o, _ = off["trend"]
    return out[:, o:o + n], out[:, off["trend_ncp"][0]]


def main():
    no_power_throttling()
    d = workdir("beast")
    for n_px, n_dates, samples, chains, label in ((1, 360, 8000, 3, "1 series x 360 months"),
                                                   (10000, 144, 2000, 2, "10000 px x 144 months")):
        v = series(n_px, n_dates, np.random.default_rng(n_px))
        path = d / f"values_{n_px}.bin"
        v.astype(np.float64).tofile(path)
        layout = _core.beast.beast_layout(n_dates, samples=samples, chains=chains, **KW)
        # The same chain with Rbeast's generic kernels, on the first series (beast123's first pixel is beast()'s).
        first = d / f"values_{n_px}_first.bin"
        v[:1].astype(np.float64).tofile(first)
        run_r("beast.R", first, 1, n_dates, samples, chains, 1, "sse")
        ref = np.fromfile(d / f"values_{n_px}_first_sse_out.bin").reshape(1, n_dates + 1)
        got = _core.beast.beast_batch(v[:1], samples=samples, chains=chains, n_jobs=1, **KW)
        tr, ncp = trend_ncp(np.asarray(got), layout)
        assert np.array_equal(tr[0], ref[0, :n_dates].astype(np.float32)) and ncp[0] == np.float32(ref[0, -1])
        for threads in ((1,) if n_px == 1 else (1, -1)):
            t_r = run_r("beast.R", path, n_px, n_dates, samples, chains, threads_label(threads), "default")
            ref = np.fromfile(d / f"values_{n_px}_default_out.bin").reshape(n_px, n_dates + 1)
            out = {}
            with pinned(threads):
                t_z = best_of(lambda: out.update(r=_core.beast.beast_batch(v, samples=samples, chains=chains,
                                                                            n_jobs=threads, **KW)),
                              repeat=3 if n_px == 1 else 1)
            tr, ncp = trend_ncp(np.asarray(out["r"]), layout)
            unfit_r, unfit_z = int(np.isnan(ref[:, -1]).sum()), int(np.isnan(ncp).sum())
            sd = np.nanstd(v, axis=1, keepdims=True)
            diff = float(np.nanmedian(np.abs(tr - ref[:, :n_dates]) / sd))
            dncp = float(np.nanmean(ncp) - np.nanmean(ref[:, -1]))
            note = (f"the same chain with Rbeast's generic kernels; with its AVX2 kernels the trends differ by a "
                    f"median {diff:.1e} sd, the mean number of changepoints by {dncp:+.2f}")
            if unfit_r or unfit_z:
                note += f"; not fitted: {unfit_r} pixels by Rbeast, {unfit_z} by zeit"
            # Rbeast's beast123 has threads too: each time is against Rbeast on as many.
            record("beast", "Rbeast 1.0.2 (R)", label, t_r, t_z, threads_label(threads), note)


if __name__ == "__main__":
    main()
