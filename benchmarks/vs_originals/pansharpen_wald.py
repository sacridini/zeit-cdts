"""zeit.pansharpen against Wald's protocol, the INPE's product, GDAL's gdal_pansharpen and the clock.

    python benchmarks/vs_originals/pansharpen_wald.py synthetic          # Wald (and the true field) on synthetic scenes
    python benchmarks/vs_originals/pansharpen_wald.py timing             # 10000 x 10000, four bands
    python benchmarks/vs_originals/pansharpen_wald.py real <folder>      # a CBERS-4A WPM window (BDC, public): Wald,
                                                                         # the pan's offset, the INPE's PCA product
    python benchmarks/vs_originals/pansharpen_wald.py gdal <folder>      # gdal_pansharpen's weighted Brovey, as oracle
    python benchmarks/vs_originals/pansharpen_wald.py landsat <folder>   # Landsat 8 band 8 (15 m) with the 30 m bands

The metrics are the harness's own (the standard definitions of Wald, Ranchin and Mangolini 1997, and of ERGAS,
Wald 2002, and SAM, Kruse et al. 1993), not zeit's. Wald's protocol: the bands and the pan are degraded by the
ratio (the mean of each ratio x ratio block), fused, and the result is compared with the original bands. A fusion
is judged against the cubic resampling of the degraded bands (the "naive" reference: no pan detail injected).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import rioxarray  # noqa: F401
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import no_power_throttling  # noqa: E402

import zeit  # noqa: E402

METHODS = ("brovey", "gsa", "hpf", "pca")


# --- metrics (harness code, independent of zeit) -------------------------------------------------------------


def metrics(ref, fused, ratio):
    """RMSE and correlation per band, ERGAS, SAM (degrees) between `(nb, h, w)` arrays, over the pixels that both
    have; ERGAS = 100 / ratio * sqrt(mean_k (RMSE_k / mean_k)^2)."""
    ref = np.asarray(ref, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    ok = np.isfinite(ref).all(axis=0) & np.isfinite(fused).all(axis=0)
    r, f = ref[:, ok], fused[:, ok]
    rmse = np.sqrt(((r - f) ** 2).mean(axis=1))
    cc = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(r, f)])
    ergas = 100.0 / ratio * np.sqrt(np.mean((rmse / r.mean(axis=1)) ** 2))
    dot = (r * f).sum(axis=0)
    den = np.sqrt((r ** 2).sum(axis=0) * (f ** 2).sum(axis=0))
    sam = np.degrees(np.arccos(np.clip(dot / np.where(den > 0, den, 1), -1, 1))).mean()
    return {"rmse": rmse, "cc": cc, "ergas": float(ergas), "sam": float(sam)}


def row(name, m):
    return (f"  {name:8s} ERGAS {m['ergas']:7.3f}  SAM {m['sam']:6.3f}  RMSE "
            + " ".join(f"{v:8.2f}" for v in m["rmse"]) + "  CC " + " ".join(f"{v:.4f}" for v in m["cc"]))


def block_mean(a, r):
    h, w = a.shape[-2] // r * r, a.shape[-1] // r * r
    a = a[..., :h, :w]
    return a.reshape(*a.shape[:-2], h // r, r, w // r, r).mean(axis=(-3, -1))


def grid_da(values, res, names=None, x0=500000.0, y0=8000000.0, crs="EPSG:32723"):
    n_y, n_x = values.shape[-2:]
    coords = {"y": y0 - (np.arange(n_y) + 0.5) * res, "x": x0 + (np.arange(n_x) + 0.5) * res}
    if values.ndim == 3:
        da = xr.DataArray(values.astype(np.float32), dims=("band", "y", "x"),
                          coords={"band": names or [f"b{i}" for i in range(values.shape[0])], **coords})
    else:
        da = xr.DataArray(values.astype(np.float32), dims=("y", "x"), coords=coords)
    return da.rio.write_crs(crs)


def wald(ms, pan, ratio, methods=METHODS, extras=False, sees=0, names=None, **kw):
    """Wald's protocol on arrays `ms` `(nb, h, w)` and `pan` `(h*ratio, w*ratio)` (same footprint, whole ratio)."""
    ms_lo = block_mean(ms, ratio)
    pan_lo = block_mean(pan, ratio)
    ms_da = grid_da(ms_lo, ratio * 2.0, names)    # the cell size is arbitrary: only the ratio matters
    pan_da = grid_da(pan_lo, 2.0)
    out = {}
    base = zeit.pansharpen(ms_da, pan_da, "hpf", gain=0.0)
    out["cubic"] = metrics(ms, base.values, ratio)
    for m in methods:
        out[m] = metrics(ms, zeit.pansharpen(ms_da, pan_da, m, **kw).values, ratio)
    if extras:
        out["hpf-g1"] = metrics(ms, zeit.pansharpen(ms_da, pan_da, "hpf", gain=1.0).values, ratio)
        if sees < ms.shape[0] and sees > 1:
            vis = [f"b{i}" for i in range(sees)]
            out["pca-sel"] = metrics(ms, zeit.pansharpen(ms_da, pan_da, "pca", intensity_bands=vis).values, ratio)
            out["gsa-sel"] = metrics(ms, zeit.pansharpen(ms_da, pan_da, "gsa", intensity_bands=vis).values, ratio)
    return out


# --- synthetic scenes -----------------------------------------------------------------------------------------


def power_law_field(shape, rng, beta=2.2):
    """A random field with a power spectrum of slope -beta (a natural image's texture), zero mean, unit variance."""
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1
    spec = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / f ** (beta / 2)
    spec[0, 0] = 0
    out = np.fft.ifft2(spec).real
    return (out - out.mean()) / out.std()


def synthetic(ny, nx, ratio, nb, seed=1, sees=3):
    """The fine scene (nb, ny*ratio, nx*ratio): bands mix 4 spectra by smooth abundance fields; the pan is the mean
    of the first `sees` bands (a pan that does not see the others) plus 1% noise."""
    rng = np.random.default_rng(seed)
    shape = (ny * ratio, nx * ratio)
    spectra = rng.uniform(80, 900, (4, nb))
    ab = np.stack([power_law_field(shape, rng) for _ in range(4)])
    ab = np.exp(1.2 * ab)
    ab /= ab.sum(axis=0)
    fine = np.einsum("sk,syx->kyx", spectra, ab) + 5 * np.stack([power_law_field(shape, rng, 1.6) for _ in range(nb)])
    pan = fine[: min(sees, nb)].mean(axis=0)
    pan = pan + 0.01 * pan.std() * rng.standard_normal(shape)
    return fine.astype(np.float32), pan.astype(np.float32)


def run_synthetic():
    print("Wald's protocol on synthetic scenes (ms degraded by the ratio, fused, compared with the original ms)")
    results = {}
    for ratio, nb, sees in [(2, 4, 3), (3, 4, 3), (4, 3, 3), (4, 4, 3), (4, 4, 4), (4, 8, 3), (8, 4, 3)]:
        fine, pan = synthetic(1024 // ratio, 1024 // ratio, ratio, nb, sees=sees)
        # the original ms is the fine scene reduced by the ratio; the pan the one at the fine scene
        # the original bands are the fine scene at the ratio coarser grid, the original pan the fine one: Wald
        # degrades both by the ratio, fuses and compares with the original bands
        res = wald(block_mean(fine, ratio), pan, ratio, extras=True, sees=sees)
        print(f"ratio {ratio}, {nb} bands, the pan sees {sees}, {fine.shape[1]} x {fine.shape[2]}")
        for k, v in res.items():
            print(row(k, v))
        results[f"r{ratio}_b{nb}_s{sees}"] = {k: {a: (b.tolist() if hasattr(b, "tolist") else b) for a, b in v.items()}
                                      for k, v in res.items()}
    # a ratio that is not whole, against the true field (no Wald: the degradation is not by blocks): the bands are
    # the mean of the fine scene over cells of 2.5 x 2.5 pan pixels (exact, by splitting each pixel in 2 x 2)
    print("ratio 2.5, 4 bands: against the true field, pan cells of 10 m, band cells of 25 m")
    fine, pan = synthetic(400, 400, 5, 4, seed=7)[0][:, :800, :800], synthetic(400, 400, 5, 4, seed=7)[1][:800, :800]
    ms25 = area_mean(fine, 2.5)
    pan_da = grid_da(pan, 10.0)
    ms_da = grid_da(ms25, 25.0)
    cub = zeit.pansharpen(ms_da, pan_da, "hpf", gain=0.0)
    inner = (slice(None), slice(20, -20), slice(20, -20))
    truth = fine[:, : cub.shape[1], : cub.shape[2]]
    print(row("cubic", metrics(truth[inner], cub.values[inner], 2.5)))
    for m in METHODS:
        out = zeit.pansharpen(ms_da, pan_da, m)
        print(row(m, metrics(truth[inner], out.values[inner], 2.5)))
    out = zeit.pansharpen(ms_da, pan_da, "hpf", gain=1.0)
    print(row("hpf-g1", metrics(truth[inner], out.values[inner], 2.5)))
    return results


def area_mean(a, r):
    """The mean of `(nb, h, w)` over cells of r x r pixels, r a multiple of 0.5 (each pixel split in 2 x 2)."""
    up = np.repeat(np.repeat(a, 2, axis=-2), 2, axis=-1)
    step = int(round(r * 2))
    n_y, n_x = up.shape[-2] // step, up.shape[-1] // step
    up = up[..., : n_y * step, : n_x * step]
    return up.reshape(a.shape[0], n_y, step, n_x, step).mean(axis=(2, 4))


# --- timing --------------------------------------------------------------------------------------------------


def run_timing(size=10000, nb=4, ratio=4, threads=8):
    no_power_throttling()
    rng = np.random.default_rng(0)
    ny = size // ratio
    coarse = rng.uniform(100, 3000, (nb, ny, ny)).astype(np.float32)
    pan = rng.uniform(100, 3000, (size, size)).astype(np.float32)
    ms = grid_da(coarse, ratio * 2.0)
    pn = grid_da(pan, 2.0)
    res = {}
    # warm-up on a small scene
    zeit.pansharpen(ms.isel(y=slice(0, 100), x=slice(0, 100)), pn.isel(y=slice(0, 400), x=slice(0, 400)), "gsa",
                    n_jobs=threads)
    for m in METHODS:
        t = time.perf_counter()
        out = zeit.pansharpen(ms, pn, m, n_jobs=threads)
        res[m] = time.perf_counter() - t
        print(f"{m:7s} {size} x {size}, {nb} bands, {threads} threads: {res[m]:.1f} s")
        del out
    # the cubic resampling alone (the share of the warp)
    t = time.perf_counter()
    from zeit._warp import Grid, _grid, to_grid, transform_of
    g = _grid(transform_of(pn), pn.shape, pn.rio.crs)
    to_grid(ms, g, "cubic", n_threads=threads)
    res["resample"] = time.perf_counter() - t
    print(f"cubic resampling alone: {res['resample']:.1f} s")
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    if mode == "synthetic":
        run_synthetic()
    elif mode == "timing":
        run_timing()
    else:
        import pansharpen_real
        pansharpen_real.main(mode, sys.argv[2:])
