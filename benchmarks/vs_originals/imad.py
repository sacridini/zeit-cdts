"""zeit.imad and zeit.radcal against Mort Canty's iMad.py, iMadmap.py and radcal.py (CRC5Docker, MIT licence), run as
black boxes (``canty_run.py``; their code is not read), on a synthetic six-band pair of 1000 x 1000 pixels with 10 % and
40 % of the pixels changed, and on two Landsat dates over Rondonia (1690 x 1675 pixels, six bands: the dry seasons of
2021 and 2022 of the CCDC stack).

    set CANTY_PYTHON=<python of an env with gdal, scipy, matplotlib>
    set CANTY_SRC=<CRC5Docker>/src
    set CANTY_LIB=<folder with libprov_means.so>        (Windows only)
    python benchmarks/vs_originals/imad.py

What must agree, and what is found (see docs/benchmarks/vs-originals.md): the canonical correlations of the last
iteration, the MAD variates (Canty numbers them the other way: his band 1 is our MAD6), the chi-square, the change
map at 0.01 and the fits of the normalization.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import rasterio

import change_data as cd
import zeit
from common import best_of, no_power_throttling, record, workdir

LANDSAT = Path(os.environ.get("ZEIT_BENCH_CCDC", r"C:\Users\eduar\Downloads\CCDC_Stack_Rondonia.tif"))
DATES = ("2021-07-16", "2022-07-27")
THREADS = 4


def landsat_pair(d):
    """The six reflective bands of two dates of the CCDC stack of Rondonia as float32 GeoTIFFs (cached)."""
    paths = [d / f"landsat_{date.replace('-', '')}.tif" for date in DATES]     # no "-": radcal.py splits on it
    if all(p.exists() for p in paths):
        return paths
    import csv

    with open(LANDSAT.with_name(LANDSAT.stem + "_Dates.csv")) as f:
        dates = [row["Date"] for row in csv.DictReader(f)]
    with rasterio.open(LANDSAT) as src:
        for date, path in zip(DATES, paths):
            k = dates.index(date)
            a = src.read(indexes=[k * 7 + b + 1 for b in range(6)], out_dtype="float32")
            with rasterio.open(path, "w", driver="GTiff", count=6, dtype="float32", width=src.width,
                               height=src.height, crs=src.crs, transform=src.transform) as dst:
                dst.write(a)
    return paths


def read(path):
    with rasterio.open(path) as ds:
        return ds.read().astype(np.float64)


def canty_rho(stdout):
    m = re.search(r"rho:\s*\[([^\]]*)\]", stdout)
    return np.array(m.group(1).split(), dtype=float)


def canty_fits(stdout):
    """slope, intercept, correlation, P(t), P(F) of each band from radcal.py's table, and the counts."""
    rows = [list(map(float, l.split())) for l in stdout.splitlines() if re.match(r"^\s*\d+\s+[-\d.]+", l)]
    counts = re.search(r"training: (\d+), for testing: (\d+)", stdout)
    return np.array(rows), (int(counts.group(1)), int(counts.group(2))) if counts else None


def compare(name, f1, f2, label):
    """One pair of files through Canty's three scripts and Zeit's, with the agreement and the times."""
    d = f1.parent
    x, y = read(f1), read(f2)
    n = x.shape[0]
    size = f"{x.shape[2]}x{x.shape[1]} px x {n} bands, {label}"
    print(f"\n{size}")
    t_mad, out = cd.canty("iMad.py", "-n", "-i", 50, f1, f2, threads=THREADS)
    madfile = d / f"MAD({f1.stem}-{f2.stem}).tif"
    t_map, _ = cd.canty("iMadmap.py", madfile, 0.01, threads=THREADS)
    # radcal.py splits the invariant pixels at random: three runs show how much its own fits move
    runs = [cd.canty("radcal.py", "-t", 0.95, madfile, threads=THREADS) for _ in range(3)]
    t_rad, rout = min(r[0] for r in runs), runs[-1][1]
    fits_runs = np.array([canty_fits(r[1])[0] for r in runs])
    norm_c = read(d / f"{f2.stem}_norm")
    theirs = read(madfile)
    rho_c = canty_rho(out)
    cmap = read(str(madfile).replace(".tif", "_cmap.tif"))
    fits_c, counts_c = canty_fits(rout)
    t_canty = t_mad + t_map + t_rad

    kw = dict(sample=None, n_jobs=THREADS)
    out_z = {}

    def zeit_imad():
        out_z["imad"] = zeit.imad(zeit.load_raster(f1), zeit.load_raster(f2), **kw).compute()

    t_z = best_of(zeit_imad, repeat=2)
    res = out_z["imad"]
    rec = json.loads(res.attrs["imad"])
    rho_z = np.array(rec["rho_final"])
    print(f"  iterations: Canty stops at its own tolerance; Zeit {rec['iterations']} (tol 1e-3, converged {rec['converged']})")
    print(f"  canonical correlations: max |difference| {np.abs(rho_z - rho_c).max():.2e}  (Canty {np.round(rho_c, 5)})")
    mad_z = res.values[:n]
    ok_px = np.isfinite(mad_z).all(0)
    corr = [abs(np.corrcoef(mad_z[i][ok_px], theirs[n - 1 - i][ok_px])[0, 1]) for i in range(n)]
    ratio = [mad_z[i][ok_px].std() / theirs[n - 1 - i][ok_px].std() for i in range(n)]
    chi_ratio = res.sel(band="chi2").values[ok_px] / theirs[n][ok_px]
    print(f"  MAD variates (his band {n} + 1 - i against our MAD i): min |r| {min(corr):.7f}, std ratio {min(ratio):.5f}-{max(ratio):.5f}")
    print(f"  chi-square: ratio {np.nanmin(chi_ratio):.5f} to {np.nanmax(chi_ratio):.5f}, r "
          f"{np.corrcoef(res.sel(band='chi2').values[ok_px], theirs[n][ok_px])[0, 1]:.9f}")
    flagged_c = (cmap != 0).any(0)
    flagged_z = res.sel(band="change").values == 1
    pz = res.sel(band="p_nochange").values
    differ = flagged_c != flagged_z
    print(f"  change map at 0.01: {100 * (1 - differ.mean()):.4f} % of the pixels agree; the {differ.sum()} that differ have "
          f"|p - 0.01| <= {np.abs(pz[differ] - 0.01).max() if differ.any() else 0:.1e}; {100 * flagged_z.mean():.1f} % flagged")

    # the normalization
    out_r = {}

    def zeit_radcal():
        out_r["r"] = zeit.radcal(zeit.load_raster(f1), after=zeit.load_raster(f2), **kw).compute()

    t_zr = best_of(zeit_radcal, repeat=2)
    r = out_r["r"]
    table = zeit.radcal_table(r)
    rec_r = json.loads(r.attrs["radcal"])
    print(f"  radcal: invariant pixels Zeit {rec_r['imad'][0]['n_invariant']} (train {table['n_train'][0]}, test "
          f"{table['n_test'][0]}), Canty train {counts_c[0]}, test {counts_c[1]}")
    worst = 0.0
    for j in range(n):
        db = abs(table["b"][j] - fits_c[j, 1])
        spread = np.ptp(fits_runs[:, j, 1])
        worst = max(worst, db / max(table["se_b"][j], spread))
        print(f"    band {j + 1}: slope Zeit {table['b'][j]:.5f} +- {table['se_b'][j]:.5f}, Canty {fits_c[j, 1]:.5f} (his three runs "
              f"span {spread:.1e}), intercept {table['a'][j]:.5g} against {fits_c[j, 2]:.5g}; |db| {db:.1e}")
    rng = (norm_c.reshape(n, -1).max(1) - norm_c.reshape(n, -1).min(1))
    mad_diff = np.abs(r.values - norm_c).reshape(n, -1).mean(1) / rng
    pear = [np.corrcoef(r.values[j].ravel(), norm_c[j].ravel())[0, 1] for j in range(n)]
    print(f"  normalized images: min r {min(pear):.6f}, mean |difference| / range <= {100 * mad_diff.max():.4f} %")
    print(f"  times (s, {THREADS} threads): Canty iMad {t_mad:.1f} + iMadmap {t_map:.1f} + radcal {t_rad:.1f} = {t_canty:.1f}; "
          f"Zeit imad {t_z:.2f}, radcal {t_zr:.2f}")
    assert np.abs(rho_z - rho_c).max() < 1e-3 and min(corr) > 0.9999 and 1 - differ.mean() > 0.999, name
    # Canty's hold-out is random and ours is every third invariant pixel: the fits differ by their sampling error
    print(f"  largest |slope difference|: {worst:.2f} times the larger of Zeit's standard error and the span of Canty's runs")
    assert worst < 3.5 and min(pear) > 0.9999 and mad_diff.max() < 5e-3, name
    note = (f"canonical correlations within {np.abs(rho_z - rho_c).max():.1e}, |r| of the MAD variates > {min(corr):.6f}, change "
            f"map at 0.01 {100 * (1 - differ.mean()):.3f} % equal, normalized images r > {min(pear):.5f}; Canty's time is "
            f"iMad + iMadmap + radcal end to end")
    record("imad + radcal", "Canty iMad.py, iMadmap.py, radcal.py (MIT)", size, t_canty, t_z + t_zr, THREADS, note)
    # default sample of 1e6 pixels
    t_s = best_of(lambda: zeit.imad(zeit.load_raster(f1), zeit.load_raster(f2), n_jobs=THREADS).compute(), repeat=2)
    print(f"  Zeit imad with the default sample of 1e6 pixels: {t_s:.2f} s")


def main():
    no_power_throttling()
    d = workdir("imad")
    only = sys.argv[1:]
    for label, share, seed in (("10 % changed", 0.10, 1), ("40 % changed", 0.40, 2)):
        if only and "synthetic" not in only:
            continue
        f1, f2 = d / f"syn{int(share * 100)}_1.tif", d / f"syn{int(share * 100)}_2.tif"
        if not f1.exists():
            cd.optical_pair(1000, change=share, seed=seed, path=str(d / f"syn{int(share * 100)}"))
        compare("synthetic", f1, f2, f"synthetic, {label}")
    if only and "landsat" not in only:
        return
    f1, f2 = landsat_pair(d)
    compare("landsat", f1, f2, f"Landsat {DATES[0]} and {DATES[1]}, Rondonia")


if __name__ == "__main__":
    main()
