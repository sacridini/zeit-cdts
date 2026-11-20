"""zeit.mnf against SPy (the `spectral` package 0.26, MIT), by outputs only.

Three images: a synthetic one (512 x 512 x 50 bands, 6 latent fields; written as a GeoTIFF too), SPy's documented
example 92AV3C (AVIRIS Indian Pines, 145 x 145 x 220, with its documented noise subregion), and a real EnMAP L2A scene
(224 bands) when one is given (``--enmap path``).

Step 1 measures SPy's undocumented conventions (the divisor of ``calc_stats``, what ``noise_from_diffs`` returns, what
``reduce`` and ``denoise`` do about the mean). Step 2 feeds both the same statistics and compares eigenvalues,
components and denoised bands. Step 3 compares the statistics and eigenvalues of Zeit's own estimators with SPy's
``noise_from_diffs`` directions. Then the times (best of 3) and the peak memory.

    python benchmarks/vs_originals/mnf.py --threads 4 [--no-real] [--no-timing] [--enmap path] [--size 512x512x50]

Run in an environment that has `spectral`; the number of threads is applied to numpy's BLAS (before numpy is
imported) and to Zeit.
"""
import argparse
import os
import sys
import warnings

_args = argparse.ArgumentParser()
_args.add_argument("--threads", type=int, default=4)
_args.add_argument("--no-real", action="store_true")
_args.add_argument("--no-timing", action="store_true")
_args.add_argument("--enmap", default=r"C:\Users\eduar\Downloads\EnMAP_L2A_Groningen_20220810.tif")
_args.add_argument("--size", default="512x512x50")
_args.add_argument("--rss", default=None, help="child mode: peak memory of '<spy|zeit>:<dataset>'")
_args.add_argument("--rss-only", action="store_true", help="only the peak memory of the whole denoise")
ARGS = _args.parse_args()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_v] = str(ARGS.threads)

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import spectral as spy  # noqa: E402

import zeit  # noqa: E402
from zeit import _core  # noqa: E402
from common import best_of, no_power_throttling, record, workdir  # noqa: E402

THREADS = ARGS.threads
# 92AV3C's documented noise subregion has 740 pixels for 220 bands: Zeit's warning about it is expected here
warnings.filterwarnings("ignore", message="only .* pairs")
D = workdir("mnf")
AV3C = "https://github.com/spectralpython/sample-data/raw/master/92AV3C.lan"
#: SPy's direction names and Zeit's estimators, with the pairs (dy, dx).
DIRECTIONS = {"right": ("diff_h", (0, 1)), "lower": ("diff_v", (1, 0)), "lowerright": ("diff_diag", (1, 1)),
              "lowerleft": ("diff_antidiag", (1, -1))}


def rel(a, b):
    """Largest absolute difference over the largest value of `b`."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max())


# ---------------------------------------------------------------------------------------------------------------
# data

def synthetic(h, w, nb, r=6, seed=0):
    """Smooth latent fields (sums of plane waves, periods 350 to 700 pixels) mixed into `nb` bands: signal 4 noise
    standard deviations, white noise of a standard deviation that grows from 0.5 to 5 across the bands."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    f = np.empty((r, h, w))
    for l in range(r):
        g = np.zeros((h, w))
        for _ in range(4):
            th, per, ph = rng.uniform(0, np.pi), rng.uniform(350, 700), rng.uniform(0, 2 * np.pi)
            g += rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * (yy * np.cos(th) + xx * np.sin(th)) / per + ph)
        f[l] = (g - g.mean()) / g.std()
    sigma = 0.5 * 10 ** (np.arange(nb) / (nb - 1))
    s = np.einsum("br,ryx->byx", rng.normal(size=(nb, r)), f)
    s *= (4 * sigma / s.reshape(nb, -1).std(1))[:, None, None]
    x = s + sigma[:, None, None] * rng.normal(size=(nb, h, w))
    return (x * 100).astype(np.float32)


def cube_of(x, names=None):
    """A DataArray ``(time=1, band, y, x)`` from ``(band, y, x)``."""
    nb, h, w = x.shape
    out = xr.DataArray(x[None], dims=("time", "band", "y", "x"),
                       coords={"band": names or [f"b{j + 1}" for j in range(nb)], "y": np.arange(h) * 10.0,
                               "x": np.arange(w) * 10.0}, name="reflectance")
    return out.rio.write_crs("EPSG:32632")


def datasets(skip_real):
    """Yields ``(name, cube, noise_region window or None, note)``."""
    h, w, nb = (int(v) for v in ARGS.size.split("x"))
    x = synthetic(h, w, nb)
    cube = cube_of(x)
    cube.isel(time=0, drop=True).rio.to_raster(D / "synthetic.tif")        # the file the others can use
    yield f"synthetic {w}x{h}x{nb}", cube, None, "6 latent fields, float32"
    lan = D / "92AV3C.lan"
    if not lan.exists():
        import urllib.request

        try:
            urllib.request.urlretrieve(AV3C, lan)
        except Exception as e:  # pragma: no cover - needs the network
            print(f"92AV3C.lan could not be downloaded ({e}); skipped")
    if lan.exists():
        img = spy.open_image(str(lan)).load()                       # SPy's own reader: (rows, cols, bands)
        xs = np.asarray(img, dtype=np.int16).transpose(2, 0, 1)
        yield "92AV3C 145x145x220", cube_of(xs), (85, 117, 122, 137), "AVIRIS Indian Pines, int16, noise subregion"
    if not skip_real and os.path.exists(ARGS.enmap):
        import rasterio

        with rasterio.open(ARGS.enmap) as src:
            nodata = src.nodata
            first = src.read(1) != nodata
            # the window of the unmixing tutorial (350 x 350 pixels of farmland and sea) when it is complete
            win = (740, 560, 1090, 910)
            if not first[win[1]:win[3], win[0]:win[2]].all():
                win = complete_window(first, 350)
            arr = src.read(window=((win[1], win[3]), (win[0], win[2])))
            empty = [b for b in range(src.count) if not (arr[b] != nodata).any()]
            # a window the NoData does not touch (SPy has no mask); the bands the scene leaves out are dropped
            keep = [b for b in range(src.count) if b not in empty]
            names = [src.descriptions[b] or f"b{b + 1}" for b in keep]
        yield (f"EnMAP Groningen {arr.shape[2]}x{arr.shape[1]}x{len(keep)}", cube_of(arr[keep], names), None,
               f"EnMAP L2A, int16, {len(empty)} empty bands dropped")


def complete_window(valid, size):
    """The first ``size x size`` window (x0, y0, x1, y1) of the valid pixels, by an integral image."""
    s = np.pad(valid.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ny, nx = valid.shape
    for y0 in range(0, ny - size, 8):
        for x0 in range(0, nx - size, 8):
            if s[y0 + size, x0 + size] - s[y0, x0 + size] - s[y0 + size, x0] + s[y0, x0] == size * size:
                return x0, y0, x0 + size, y0 + size
    raise RuntimeError("no complete window")


# ---------------------------------------------------------------------------------------------------------------
# step 1: SPy's conventions

def shifted(X, d):
    """The differences ``z(p) - z(p + d)`` over the pairs inside `X` ``(rows, cols, bands)``."""
    dy, dx = d
    ys, yq = slice(0, X.shape[0] - dy), slice(dy, X.shape[0])
    xs, xq = (slice(0, X.shape[1] - dx), slice(dx, X.shape[1])) if dx >= 0 else (slice(-dx, X.shape[1]),
                                                                                  slice(0, X.shape[1] + dx))
    return (X[ys, xs] - X[yq, xq]).reshape(-1, X.shape[2])


def noise_convention(X, direction):
    """What ``noise_from_diffs(X, direction)`` returns, among the covariance of the differences or half of it, centred
    or not, with divisor m - 1 or m: exactly one must match."""
    d = shifted(X, DIRECTIONS[direction][1])
    m = len(d)
    got = spy.noise_from_diffs(X, direction)
    out = {}
    for centred in (True, False):
        c = d - d.mean(0) if centred else d
        s = c.T @ c
        for div_name, div in (("m-1", m - 1), ("m", m)):
            for half in (1.0, 0.5):
                out[(centred, div_name, half)] = rel(got.cov, s / div * half)
    best = min(out, key=out.get)
    others = sorted(v for k, v in out.items() if k != best)
    assert out[best] < 1e-10 and others[0] > 1e-6, (best, out)
    mean_dbar, mean_zero = rel(got.mean, d.mean(0)) if np.abs(d.mean(0)).max() > 0 else 0, float(np.abs(got.mean).max())
    return {"centred": best[0], "divisor": best[1], "factor": best[2], "err": out[best], "nsamples": got.nsamples,
            "m": m, "mean_is_dbar": mean_dbar < 1e-10, "mean_abs": mean_zero}, got


def signal_convention(X):
    pts = X.reshape(-1, X.shape[2])
    n = len(pts)
    got = spy.calc_stats(X)
    c1 = np.cov(pts, rowvar=False)
    return {"n-1": rel(got.cov, c1), "n": rel(got.cov, c1 * (n - 1) / n), "mean": rel(got.mean, pts.mean(0)),
            "nsamples": (got.nsamples, n)}, got


def reduce_mapping(ours, theirs):
    """Per component: sign, scale and offset taking Zeit's component to SPy's, and the error left after them."""
    k = ours.shape[-1]
    slope, offset, err = np.empty(k), np.empty(k), np.empty(k)
    for i in range(k):
        a, b = ours[..., i].ravel().astype(np.float64), theirs[..., i].ravel()
        sign = 1.0 if np.dot(a - a.mean(), b - b.mean()) >= 0 else -1.0
        slope[i] = (np.cov(a, b)[0, 1]) / a.var(ddof=1)
        offset[i] = np.median(b - sign * a)
        err[i] = np.abs(b - sign * a - offset[i]).max() / b.std()
    return slope, offset, err


# ---------------------------------------------------------------------------------------------------------------
# clusters of near-equal eigenvalues, compared as subspaces

def clusters(lam, tol=1e-4):
    """Groups of consecutive eigenvalues that are not separated from a neighbour by more than `tol` relative."""
    out, cur = [], [0]
    for i in range(1, len(lam)):
        if abs(lam[i] - lam[i - 1]) <= tol * abs(lam[i]):
            cur.append(i)
        else:
            out.append(cur)
            cur = [i]
    out.append(cur)
    return out


def max_angle(a, b):
    """Largest principal angle (radians) between the column spaces of `a` and `b` ``(pixels, k)``."""
    qa, _ = np.linalg.qr(a - a.mean(0))
    qb, _ = np.linalg.qr(b - b.mean(0))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.arccos(np.clip(s.min(), 0, 1)))


# ---------------------------------------------------------------------------------------------------------------

def zeit_stats(x, noise, region, threads):
    reg = None
    if region is not None:
        reg = np.zeros(x.shape[2:], np.uint8)
        reg[region[1]:region[3], region[0]:region[2]] = 1
    return _core.mnf.stats_batch(np.ascontiguousarray(x), noise, None, None, reg, [], 1.0, 0, 0, threads)


def study(name, cube, region, note, timing):
    print(f"\n=== {name}: {note}, {THREADS} threads ===")
    x = cube.values                                                  # (1, band, y, x), the file's dtype
    nb = x.shape[1]
    X = x[0].astype(np.float64).transpose(1, 2, 0)                   # (rows, cols, bands), SPy's layout
    Xn = X if region is None else X[region[1]:region[3], region[0]:region[2]]

    # step 1: conventions
    sc, signal = signal_convention(X)
    print(f"calc_stats: cov at divisor n-1 {sc['n-1']:.1e}, at n {sc['n']:.1e}; mean {sc['mean']:.1e}; "
          f"nsamples {sc['nsamples'][0]} (n = {sc['nsamples'][1]})")
    conv = {}
    for direction in DIRECTIONS:
        conv[direction], _ = noise_convention(Xn, direction)
    c = conv["right"]
    print(f"noise_from_diffs: the covariance of the differences, {'centred' if c['centred'] else 'not centred'}, "
          f"divisor {c['divisor']}, times {c['factor']} (error {c['err']:.1e}); mean is the mean difference: "
          f"{c['mean_is_dbar']} (|mean| {c['mean_abs']:.2e}); same for all four directions: "
          f"{all((v['centred'], v['divisor'], v['factor']) == (c['centred'], c['divisor'], c['factor']) for v in conv.values())}")

    # step 2: identical statistics
    noise = spy.noise_from_diffs(Xn, "lowerright")
    mnfr = spy.mnf(signal, noise)
    tr = zeit.MNFTransform.from_covariances(signal.mean, signal.cov, noise.cov,
                                            [str(b) for b in cube.band.values])
    lam_c = np.asarray(mnfr.napc.eigenvalues)               # SPy returns complex numbers (numpy's eig), imaginary 0
    print(f"SPy's eigenvalues are {lam_c.dtype}, the largest imaginary part {np.abs(lam_c.imag).max():.1e}")
    lam_spy = lam_c.real
    err_lam = float(np.abs(tr.eigenvalues / lam_spy - 1).max())
    print(f"eigenvalues from the same statistics: largest relative difference {err_lam:.1e} "
          f"(first {tr.eigenvalues[0]:.4g} vs {lam_spy[0]:.4g}, SNR > 1: {tr.n_with_snr(1.0)} "
          f"vs {mnfr.num_with_snr(1.0)})")
    assert err_lam < 1e-6, err_lam
    m = len(lam_spy)
    ours = zeit.mnf(cube, transform=tr, nodata=None).values[0].transpose(1, 2, 0)          # (rows, cols, m), noise scale
    theirs = np.real(np.asarray(mnfr.reduce(X, num=m)))
    slope, offset, err = reduce_mapping(ours, theirs)
    sep = np.ones(m, bool)
    groups = clusters(lam_spy)
    for g in groups:
        if len(g) > 1:
            sep[g] = False
    print(f"reduce(num={m}) against zeit.mnf: scale of every component {np.abs(slope).min():.9f} to "
          f"{np.abs(slope).max():.9f}; offset (SPy minus Zeit) {np.abs(offset).max():.2e} at most "
          f"(components have std {np.sqrt(lam_spy[0]):.3g} .. {np.sqrt(lam_spy[-1]):.3g})")
    print(f"  separated components ({sep.sum()} of {m}): largest error {err[sep].max():.1e} of the std; "
          f"worst of any component {err.max():.1e}")
    assert err[sep].max() < 1e-5, err[sep].max()
    angles = []
    for g in groups:
        if len(g) > 1:
            step = max(1, X.shape[0] * X.shape[1] // 40000)
            a = ours.reshape(-1, m)[::step][:, g]
            b = theirs.reshape(-1, m)[::step][:, g]
            angles.append(max_angle(a, b))
    if angles:
        print(f"  {len(angles)} clusters of near-equal eigenvalues ({sum(len(g) for g in groups if len(g) > 1)} "
              f"components): largest principal angle {max(angles):.1e} rad")
        assert max(angles) < 1e-3, max(angles)
    k = max(1, mnfr.num_with_snr(1.0))
    den_spy = np.real(np.asarray(mnfr.denoise(X, num=k)))
    den_zeit = zeit.mnf_denoise(cube, k, transform=tr, nodata=None).values[0].transpose(1, 2, 0).astype(np.float64)
    sd = X.reshape(-1, nb).std(0)
    d_restored = float((np.abs(den_spy - den_zeit).reshape(-1, nb).max(0) / sd).max())
    d_nomean = float((np.abs(den_spy - (den_zeit - tr.mean)).reshape(-1, nb).max(0) / sd).max())
    print(f"denoise(num={k}) against zeit.mnf_denoise: largest difference {d_restored:.1e} of a band's std "
          f"(if SPy did not restore the mean: {d_nomean:.1e})")
    assert d_restored < 1e-5, d_restored

    # step 3: end to end
    print("end to end (statistics from the data): ")
    spy_cov = {}
    for direction, (est, d) in DIRECTIONS.items():
        stz = zeit_stats(x, est, region, THREADS)
        dz = stz.fit().to_dict()
        got = spy.noise_from_diffs(Xn, direction)
        r = rel(got.cov, dz["noise_cov"])
        lam_z = zeit.mnf_fit(cube, noise=est, noise_region=region, nodata=None, n_jobs=THREADS).eigenvalues
        lam_s = np.real(np.asarray(spy.mnf(signal, got).napc.eigenvalues))
        e = float(np.abs(lam_z / lam_s - 1).max())
        print(f"  {est:<14} noise covariance {r:.1e}, eigenvalues {e:.1e}")
        assert r < 1e-10 and e < 1e-6, (est, r, e)
        spy_cov[direction] = (got.cov, int(dz["n_pairs"][0]), conv[direction])
    # diff_hv = the pooled estimate of the two directions, from SPy's own covariances
    (ch, mh, vh), (cv, mv, vv) = spy_cov["right"], spy_cov["lower"]

    def m2(cov, m_, conv_):
        div = m_ - 1 if conv_["divisor"] == "m-1" else m_
        return cov * div / conv_["factor"]

    pooled = (m2(ch, mh, vh) + m2(cv, mv, vv)) / (2.0 * (mh + mv - 2))
    lam_hv = zeit.mnf_fit(cube, noise="diff_hv", noise_region=region, nodata=None, n_jobs=THREADS).eigenvalues
    lam_s = np.real(np.asarray(spy.mnf(signal, spy.GaussianStats(mean=signal.mean, cov=pooled, nsamples=mh + mv))
                               .napc.eigenvalues))
    e = float(np.abs(lam_hv / lam_s - 1).max())
    print(f"  {'diff_hv':<14} eigenvalues {e:.1e} (SPy statistics pooled by hand)")
    assert e < 1e-6, e
    if timing:
        times(name, cube, region, X, Xn, k)


def times(name, cube, region, X, Xn, k):
    """Best of 3, SPy and Zeit on the same threads: statistics, fit, forward, denoise."""
    x = cube.values
    size = f"{name}, {THREADS} thread{'s' if THREADS != 1 else ''}"
    box = {}

    def spy_stats():
        box["s"] = spy.calc_stats(X)
        box["n"] = spy.noise_from_diffs(Xn, "lowerright")

    t_s = best_of(spy_stats)
    t_z = best_of(lambda: box.update(z=zeit_stats(x, "diff_diag", region, THREADS)))
    record("mnf: statistics", "SPy 0.26", size, t_s, t_z, THREADS, "calc_stats + noise_from_diffs (one direction)")
    t_hv = best_of(lambda: zeit_stats(x, "diff_hv", region, THREADS))
    print(f"  Zeit with the default diff_hv (two directions): {t_hv:.3f} s")
    t_s = best_of(lambda: box.update(m=spy.mnf(box["s"], box["n"])))
    t_z = best_of(lambda: box.update(t=box["z"].fit()))
    record("mnf: fit", "SPy 0.26", size, t_s, t_z, THREADS, "mnf(signal, noise) / the eigenproblem")
    tr = zeit.mnf_fit(cube, noise="diff_diag", noise_region=region, nodata=None, n_jobs=THREADS)
    t_s = best_of(lambda: box["m"].reduce(X, num=k))
    t_z = best_of(lambda: zeit.mnf(cube, k, transform=tr, nodata=None, n_jobs=THREADS))
    record("mnf: forward", "SPy 0.26", size, t_s, t_z, THREADS, f"reduce(num={k}) / mnf(n_components={k})")
    t_s = best_of(lambda: box["m"].denoise(X, num=k))
    t_z = best_of(lambda: zeit.mnf_denoise(cube, k, transform=tr, nodata=None, n_jobs=THREADS))
    record("mnf: denoise", "SPy 0.26", size, t_s, t_z, THREADS, f"denoise(num={k}) / mnf_denoise(k={k})")
    t_s = best_of(lambda: (spy_stats(), box.update(m=spy.mnf(box["s"], box["n"])), box["m"].denoise(X, num=k)), 2)
    t_z = best_of(lambda: zeit.mnf_denoise(cube, k=k, noise="diff_diag", noise_region=region, nodata=None, n_jobs=THREADS), 2)
    record("mnf: whole denoise", "SPy 0.26", size, t_s, t_z, THREADS, "statistics + fit + denoise")


def working_set_mb():
    """Current working set of this process (Windows), MB."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return c.WorkingSetSize / 2 ** 20


def rss_child(which, key):
    """The extra memory at the peak of the whole denoise (statistics, fit, denoising) of one library on one image,
    above what the process holds with the image loaded: the working set sampled every 2 ms during the run."""
    import gc
    import threading
    import time

    for name, cube, region, note in datasets(False):
        if not name.startswith(key):
            continue
        x = cube.values
        k = 8
        X = Xn = None
        if which == "spy":
            X = x[0].astype(np.float64).transpose(1, 2, 0)          # the input SPy needs: float64, (rows, cols, bands)
            Xn = X if region is None else X[region[1]:region[3], region[0]:region[2]]
        gc.collect()
        base, top, stop = working_set_mb(), [0.0], threading.Event()

        def watch():
            while not stop.is_set():
                top[0] = max(top[0], working_set_mb())
                time.sleep(0.002)

        thread = threading.Thread(target=watch)
        thread.start()
        if which == "spy":
            mnfr = spy.mnf(spy.calc_stats(X), spy.noise_from_diffs(Xn, "lowerright"))
            out = mnfr.denoise(X, num=k)
        else:
            out = zeit.mnf_denoise(cube, k=k, noise="diff_diag", noise_region=region, nodata=None, n_jobs=THREADS)
        stop.set()
        thread.join()
        print(f"PEAK {which} {name}: image {x.nbytes / 2 ** 20:.0f} MB as stored, the process holds {base:.0f} MB with it "
              f"(SPy's float64 copy included), extra at the peak {top[0] - base:.0f} MB")
        return out


def main():
    no_power_throttling()
    if ARGS.rss:
        which, key = ARGS.rss.split(":")
        return rss_child(which, key)
    print(f"spectral {spy.__version__}, zeit from {os.path.dirname(zeit.__file__)}, {THREADS} threads")
    if not ARGS.rss_only:
        for name, cube, region, note in datasets(ARGS.no_real):
            study(name, cube, region, note, not ARGS.no_timing)
    if not ARGS.no_timing:
        import subprocess

        print("\npeak memory of the whole denoise (statistics, fit, k = 8), each library in its own process:")
        for key in ("synthetic", "92AV3C", "EnMAP"):
            for which in ("spy", "zeit"):
                cmd = [sys.executable, __file__, "--threads", str(THREADS), "--rss", f"{which}:{key}",
                       "--enmap", ARGS.enmap, "--size", ARGS.size]
                out = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
                for line in out.splitlines():
                    if line.startswith("PEAK"):
                        print("  " + line[5:])


if __name__ == "__main__":
    main()
