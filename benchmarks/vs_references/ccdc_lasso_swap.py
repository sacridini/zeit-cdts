"""Runs the CCDC on fixed inputs and saves everything it returns, to compare two builds of the Rust core.

Made for the swap of the CCDC's lasso (the GLMnet-derived solver, then the clean-room zeit-core/src/lasso.rs):
run it with the old extension (``--out ccdc_before.npz``), rebuild, run it again (``--out ccdc_after.npz``) and
compare with ``--compare before.npz after.npz [--report report.md]``.

Inputs (all generated here, no random state outside the seeds):
  parity   the pixels of tests/data/ccdc_matlab_parity_inputs.npz (the MATLAB parity cases, also the pixels of
           tests/test_ccdc_api.py), with the parity test's options
  battery  the 200 synthetic Landsat-like pixels of benchmarks/vs_references/ccdc.py (synthetic_battery)
  metrics  the pixel of tests/test_metrics_ccdc.py::test_ccdc_basic (default options, 6 bands)
  cube     ~2000 pixels, 20 years of irregular Landsat (two sensors, gaps, flagged and unflagged clouds, planted
           breaks), 6 optical bands and a constant thermal band; the first 300 are also run with num_c 4 and 6
At most 4 threads (one: the pixels are run one by one).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ZEIT_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from zeit._ccdc import run_ccdc  # noqa: E402

MAX_SEG = 12
NB = 7
THERMAL = 2932 * 10.0 - 27315.0


def _pack(segs_per_pixel):
    """Per-pixel lists of segments -> fixed arrays (padded with NaN)."""
    n = len(segs_per_pixel)
    out = {
        "n_seg": np.zeros(n, int),
        "t_start": np.full((n, MAX_SEG), np.nan), "t_end": np.full((n, MAX_SEG), np.nan),
        "t_break": np.full((n, MAX_SEG), np.nan), "category": np.full((n, MAX_SEG), np.nan),
        "num_obs": np.full((n, MAX_SEG), np.nan), "change_prob": np.full((n, MAX_SEG), np.nan),
        "rmse": np.full((n, MAX_SEG, NB), np.nan), "magnitude": np.full((n, MAX_SEG, NB), np.nan),
        "coefs": np.full((n, MAX_SEG, NB, 8), np.nan),
    }
    for p, segs in enumerate(segs_per_pixel):
        out["n_seg"][p] = len(segs)
        for i, s in enumerate(segs[:MAX_SEG]):
            for k in ("t_start", "t_end", "t_break", "category", "num_obs", "change_prob"):
                out[k][p, i] = s[k]
            nb = len(s["rmse"])
            out["rmse"][p, i, :nb] = s["rmse"]
            out["magnitude"][p, i, :nb] = s["magnitude"]
            out["coefs"][p, i, :nb, :] = np.asarray(s["coefs"], float)[:, :8]
    return out


PARITY_OPTS = dict(conseq_anom=6, chi2_prob_threshold=0.99, detection_bands=[1, 2, 3, 4, 5],
                   tmask_bands=[1, 4], thermal_band=6)


def run_pixels(pixels, **opts):
    return _pack([run_ccdc(d, b, q, **opts) for d, b, q in pixels])


def parity_pixels():
    z = np.load(REPO / "tests" / "data" / "ccdc_matlab_parity_inputs.npz")
    names = sorted({k.split("__")[0] for k in z.files})
    pix = [(z[n + "__dates"], z[n + "__bands"].astype(float), z[n + "__qa"].astype(int)) for n in names]
    return names, pix


def battery_pixels():
    sys.path.insert(0, str(REPO / "benchmarks" / "vs_references"))
    from ccdc import synthetic_battery  # imports common (thread variables only)
    ordinals, optical, fmask, kinds = synthetic_battery()
    pix = []
    for p in range(optical.shape[1]):
        bands = np.concatenate([optical[:, p, :].astype(float), np.full((optical.shape[0], 1), THERMAL)], axis=1).T
        pix.append((ordinals, bands, fmask[:, p]))
    return pix


def metrics_pixel():
    rng = np.random.default_rng(0)
    dates = np.arange(date(2000, 1, 1).toordinal(), date(2008, 1, 1).toordinal(), 16)
    w = 2 * np.pi / 365.25
    base = np.array([450.0, 750.0, 550.0, 3000.0, 1800.0, 900.0])
    values = base[:, None] * (1 + 0.15 * np.cos(w * dates[None, :]))
    values += rng.normal(0, 25, values.shape)
    values[:, len(dates) // 2:] += np.array([150.0, 250.0, 450.0, -1300.0, 900.0, 700.0])[:, None]
    return [(dates, values, np.zeros(len(dates), int))]


def cube_pixels(n=2000, seed=7):
    """Irregular Landsat: two sensors 16 days apart and 8 days offset, 25% of the passes lost, clouds, planted breaks."""
    rng = np.random.default_rng(seed)
    d0 = date(2000, 1, 1).toordinal()
    a = np.arange(d0, date(2020, 1, 1).toordinal(), 16)
    b = np.arange(d0 + 8, date(2020, 1, 1).toordinal(), 16)
    all_dates = np.sort(np.concatenate([a, b]))
    keep = rng.random(all_dates.size) > 0.25
    dates = all_dates[keep]
    T = dates.size
    t = dates.astype(float)
    w = 2 * np.pi / 365.25
    base = np.array([450, 750, 550, 3000, 1800, 900], float)
    delta = np.array([150, 250, 450, -1300, 900, 700], float)
    pix = []
    for p in range(n):
        lvl = base * rng.uniform(0.7, 1.3, 6)
        amp = lvl * rng.uniform(0.03, 0.3, 6)
        ph = rng.uniform(0, 2 * np.pi)
        y = lvl[None, :] + amp[None, :] * np.cos(w * t[:, None] + ph) \
            + 0.4 * amp[None, :] * np.cos(2 * w * t[:, None] + 2 * ph)
        nbk = rng.choice([0, 1, 1, 2, 3])
        for tb in sorted(rng.integers(T // 8, 7 * T // 8, nbk)):
            y[tb:] *= np.exp(rng.uniform(-0.6, 0.6, 6))  # multiplicative, so the reflectance stays positive
        y += (t[:, None] - t[0]) / 365.25 * delta[None, :] * rng.uniform(-0.01, 0.01)
        y += rng.normal(0, 1, y.shape) * lvl * rng.uniform(0.02, 0.06)
        q = np.zeros(T, int)
        q[rng.random(T) < 0.20] = 4
        q[rng.random(T) < 0.05] = 2
        q[rng.random(T) < 0.03] = 3
        hidden = (q == 0) & (rng.random(T) < 0.05)  # unflagged clouds: Tmask's job
        y[hidden] += np.array([1500, 1700, 1800, 1400, 1000, 600]) * rng.uniform(0.6, 1.2)
        y[q == 4] = np.array([3000, 3200, 3400, 4000, 3800, 3000])
        y = np.clip(np.round(y), 50, 9999)
        bands = np.concatenate([y, np.full((T, 1), THERMAL)], axis=1).T
        pix.append((dates, bands, q))
    return pix


def collect():
    out = {}
    t0 = time.time()

    def add(prefix, packed):
        for k, v in packed.items():
            out[f"{prefix}__{k}"] = v

    names, pix = parity_pixels()
    out["parity__names"] = np.array(names)
    add("parity", run_pixels(pix, **PARITY_OPTS))
    add("battery", run_pixels(battery_pixels(), **PARITY_OPTS))
    add("metrics", run_pixels(metrics_pixel()))
    cube = cube_pixels()
    add("cube", run_pixels(cube, **PARITY_OPTS))
    for nc in (4, 6):
        add(f"cube_c{nc}", run_pixels(cube[:300], num_c=nc, **PARITY_OPTS))
    print(f"ran in {time.time() - t0:.1f} s")
    return out


def _rel_abs(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    d = np.abs(a[m] - b[m])
    r = d / np.maximum(np.abs(b[m]), 1e-12)
    return d, r


def _stats(x):
    if x.size == 0:
        return "n/a"
    return f"median {np.median(x):.3g}, p99 {np.percentile(x, 99):.3g}, max {x.max():.3g}"


def compare(before, after, report):
    B, A = np.load(before), np.load(after)
    lines = ["# CCDC lasso swap: before (GLMnet-derived) vs after (clean-room lasso.rs)", ""]
    sets = sorted({k.split("__")[0] for k in B.files if k.endswith("__n_seg")})
    for s in sets:
        nb, na = B[f"{s}__n_seg"], A[f"{s}__n_seg"]
        n = nb.size
        same_n = nb == na
        lines += [f"## {s} ({n} pixels)", "",
                  f"- same number of segments: {same_n.sum()} / {n} ({100 * same_n.mean():.2f}%)"]
        tb_b, tb_a = B[f"{s}__t_break"], A[f"{s}__t_break"]
        ts_b, ts_a = B[f"{s}__t_start"], A[f"{s}__t_start"]
        te_b, te_a = B[f"{s}__t_end"], A[f"{s}__t_end"]
        exact = same_n & np.all((tb_b == tb_a) | (np.isnan(tb_b) & np.isnan(tb_a)), axis=1) \
            & np.all((ts_b == ts_a) | (np.isnan(ts_b) & np.isnan(ts_a)), axis=1) \
            & np.all((te_b == te_a) | (np.isnan(te_b) & np.isnan(te_a)), axis=1)
        lines.append(f"- same break/start/end dates, exactly: {exact.sum()} / {n} ({100 * exact.mean():.2f}%)")
        # within one observation: dates differ by at most the largest gap to a neighbour; approximated by 16 days
        near = same_n & np.all(np.nan_to_num(np.abs(tb_b - tb_a)) <= 16, axis=1) \
            & np.all(np.nan_to_num(np.abs(ts_b - ts_a)) <= 16, axis=1) \
            & np.all(np.nan_to_num(np.abs(te_b - te_a)) <= 16, axis=1)
        lines.append(f"- same dates within +-16 days (about one observation): {near.sum()} / {n} "
                     f"({100 * near.mean():.2f}%)")
        for k in ("category", "num_obs"):
            eq = same_n & np.all((B[f"{s}__{k}"] == A[f"{s}__{k}"]) | np.isnan(B[f"{s}__{k}"]), axis=1)
            lines.append(f"- same {k}: {eq.sum()} / {n}")
        # numbers compared where both builds have the same segment structure
        sel = exact
        # The six optical bands apart from the constant thermal band (the old fit of a constant column left
        # ~1e-3 of single-precision noise in its rmse, the new one gives exactly 0). a0 and c1 are collinear on the
        # datenum axis (t ~ 7e5), so their differences are shown through the model's values at the segment's
        # start, middle and end (`predicted`), and against the segment's rmse.
        def part(k, bands):
            x = A[f"{s}__{k}"][sel], B[f"{s}__{k}"][sel]
            return [v[:, :, bands] if v.ndim >= 3 else v for v in x]

        for k in ("rmse", "magnitude"):
            a, b = part(k, slice(0, 6))
            d, r = _rel_abs(a, b)
            lines.append(f"- {k}, optical bands (pixels with identical dates, {int(sel.sum())}): abs diff {_stats(d)}; "
                         f"rel diff {_stats(r)}")
        a, b = part("coefs", slice(0, 6))
        d, r = _rel_abs(a[..., 2:], b[..., 2:])
        lines.append(f"- harmonic coefficients (a1..b3), optical: abs diff {_stats(d)}; rel diff {_stats(r)}")
        d, r = _rel_abs(a[..., 1], b[..., 1])
        lines.append(f"- trend coefficient c1 (per day), optical: abs diff {_stats(d)}; rel diff {_stats(r)}")
        ts = np.stack([A[f"{s}__t_start"][sel], 0.5 * (A[f"{s}__t_start"][sel] + A[f"{s}__t_end"][sel]),
                       A[f"{s}__t_end"][sel]]) + 366.0
        w = 2 * np.pi / 365.25
        terms = np.stack([np.ones_like(ts), ts] + [f(m * w * ts) for m in (1, 2, 3) for f in (np.cos, np.sin)], axis=-1)
        pa = np.einsum("tpsk,psbk->tpsb", terms, np.nan_to_num(a))
        pb = np.einsum("tpsk,psbk->tpsb", terms, np.nan_to_num(b))
        valid = ~np.isnan(A[f"{s}__t_start"][sel])
        dp = np.abs(pa - pb)[:, valid, :].ravel() if valid.any() else np.zeros(0)
        rm = np.broadcast_to(np.maximum(B[f"{s}__rmse"][sel][:, :, :6], 1e-9)[None], pa.shape)[:, valid, :].ravel()
        lines.append(f"- predicted value at start/middle/end of each segment, optical: abs diff {_stats(dp)}; "
                     f"as a fraction of the rmse {_stats(dp / rm)}")
        a, b = part("rmse", slice(6, 7))
        d, r = _rel_abs(a, b)
        lines.append(f"- rmse of the constant thermal band: abs diff {_stats(d)} (before: median "
                     f"{np.nanmedian(b):.2g}, after: {np.nanmedian(a):.2g})")
        d, r = _rel_abs(A[f"{s}__change_prob"][sel], B[f"{s}__change_prob"][sel])
        lines.append(f"- change_prob: abs diff {_stats(d)}")
        bad = np.where(~exact)[0]
        lines.append(f"- pixels with other dates or number of segments: {bad.size} (first: {bad[:12].tolist()})")
        lines.append("")
    Path(report).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare[0], a.compare[1], a.report or "ccdc_lasso_swap_report.md")
    else:
        np.savez_compressed(a.out, **collect())
        print("saved", a.out)


if __name__ == "__main__":
    main()
