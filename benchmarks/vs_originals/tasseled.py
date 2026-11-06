"""zeit.tasseled_cap against RStoolbox's tasseledCap (RStoolbox 1.0.2.2, GPL (>= 3)) on a synthetic six-band image
of 1000 x 1000 pixels, for the sensors both have: Landsat 7 ETM+ (Huang et al. 2002) and Landsat 8 OLI (Baig et al.
2014) equal to 1e-6; Landsat 5 TM (Crist 1985) differs in two coefficients, where RStoolbox's table has the sign of the
SWIR1 weight of greenness (+0.0002) and of wetness (+0.6806) the opposite of the paper (Crist 1985, Table 1: -0.0002 and
-0.6806; FORCE has them negative too). Zeit is checked against RStoolbox's own table too (``coefficients=`` its numbers):
equal to 1e-6.

    python benchmarks/vs_originals/tasseled.py
"""
import numpy as np
import xarray as xr

import zeit
from common import best_of, no_power_throttling, record, run_r, workdir

SENSORS = [("landsat5tm", "landsat-5", "TM"), ("landsat7etm", "landsat-7", "ETM+"), ("landsat8oli", "landsat-8", "OLI")]
BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]
NAMES = ["brightness", "greenness", "wetness"]
SIDE = 1000


def spectra(n, rng):
    """Mixtures of vegetation, soil, water and shade, as reflectance: ``(n, 6)``."""
    e = np.array([[0.02, 0.05, 0.03, 0.45, 0.22, 0.10], [0.10, 0.14, 0.20, 0.26, 0.38, 0.33],
                  [0.08, 0.06, 0.04, 0.02, 0.01, 0.01], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return rng.dirichlet(np.ones(4), n) @ e + rng.normal(0, 0.004, (n, 6))


def main():
    no_power_throttling()
    d = workdir("tasseled")
    pixels = spectra(SIDE * SIDE, np.random.default_rng(7))
    path = d / "img.bin"
    pixels.astype(np.float64).tofile(path)
    cube = xr.DataArray(pixels.T.reshape(6, SIDE, SIDE).astype(np.float32), dims=("band", "y", "x"),
                        coords={"band": BANDS})
    print(f"{SIDE}x{SIDE} pixels x 6 bands; the maximum |difference| of each component (reflectance 0-1)")
    print(f"  {'sensor':<14}{'brightness':>12}{'greenness':>12}{'wetness':>12}")
    for sat, platform, label in SENSORS:
        t_r = run_r("tasseled.R", path, SIDE, sat, sat)
        theirs = np.fromfile(d / f"img_{sat}.bin").reshape(-1, 3).T.reshape(3, SIDE, SIDE)
        c = cube.expand_dims(time=[np.datetime64("2020-01-01")]).assign_coords(platform=("time", [platform]))
        out = {}
        t_z = best_of(lambda: out.update(z=zeit.tasseled_cap(c, coefficients="sensor").isel(time=0).values))
        diff = np.abs(out["z"].astype(np.float64) - theirs).reshape(3, -1).max(1)
        print(f"  {label:<14}{diff[0]:>12.2e}{diff[1]:>12.2e}{diff[2]:>12.2e}")
        rows = {}
        if label == "TM":
            # RStoolbox's table, as it is (the SWIR1 weights positive in greenness and wetness): the paper's, with
            # those two signs turned, which is what the difference above is
            from zeit._tasseled import CRIST_1985

            rows = {k: list(v) for k, v in CRIST_1985.items()}
            rows["greenness"][4], rows["wetness"][4] = 0.0002, 0.6806
            same = zeit.tasseled_cap(c, coefficients=rows).isel(time=0).values
            err = float(np.abs(same.astype(np.float64) - theirs).max())
            print(f"  {'TM, with RStoolbox table':<26}{err:>12.2e} (all three components)")
            assert err < 1e-6, err
            assert diff[0] < 1e-6 and diff[1] > 1e-5 and diff[2] > 1e-2
            note = "TM: RStoolbox has +0.0002 and +0.6806 for SWIR1 in greenness and wetness; Crist 1985 and FORCE -0.0002, -0.6806"
        else:
            assert diff.max() < 1e-6, diff
            note = f"max diff {diff.max():.1e} (float32 against float64)"
        record("tasseled cap", "RStoolbox 1.0.2.2 (R)", f"{SIDE}x{SIDE} px x 6 bands, {label}", t_r, t_z, 1, note)


if __name__ == "__main__":
    main()
