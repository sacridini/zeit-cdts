"""The best available pixel composite (zeit-core's bap) against the same score written with numpy and scipy
(scipy.ndimage.distance_transform_edt for the distance to clouds, a weighted sum, argmax over the dates), the way a
Python implementation of Griffiths et al. (2013) and White et al. (2014) computes it; FORCE's and geebap's are not
runnable here (a Linux C program, Earth Engine).

    python benchmarks/vs_originals/bap.py            # synthetic: 60 dates of 1024 x 1024
    python benchmarks/vs_originals/bap.py --network  # also a year of Landsat over Rondonia (Planetary Computer)

The network run times zeit.build_annual_composites with method="bap" and "medoid" on the same scenes, the reading
included: the score costs little beside the download.

Measured 2026-11-01 (Windows 11, 20 logical processors; dask on 4 workers): 60 dates of 1024 x 1024, zeit-core 2.67 s
on one thread and 0.85 s on four, numpy + scipy 5.92 s (2.2x and 7.0x), the same date at every pixel. Landsat C2 L2,
June to September 2023 over 370 x 367 pixels (33 scenes): bap 9.5 s, medoid 9.2 s, both read and composited (the
reading is the time); the bap's pixels come from 2 of the 33 scenes (days 211 and 212, the target 212.5).
"""

import sys
import time

import numpy as np
from scipy import ndimage

from zeit._core import bap

T, SIDE = 60, 1024
OPTS = dict(target_doy=200.0, doy_sigma=30.0, cloud_min=1.0, cloud_max=50.0, w_doy=0.3, w_cloud=0.5, w_sensor=0.1,
            w_cover=0.1)


def states_of(rng):
    """Dates of clear land with round clouds (some dates half cloudy), a corner without data."""
    s = np.full((T, SIDE, SIDE), 2, dtype=np.uint8)
    yy, xx = np.ogrid[:SIDE, :SIDE]
    for t in range(T):
        for _ in range(rng.integers(0, 40)):
            y, x, r = rng.integers(0, SIDE, 2).tolist() + [int(rng.integers(5, 60))]
            s[t][(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 1
        if t % 7 == 0:
            s[t, : SIDE // 3, : SIDE // 3] = 0
    return s


def reference(states, doy, sensor, cover, o):
    best_s = np.full(states.shape[1:], -np.inf, dtype=np.float64)
    best_t = np.full(states.shape[1:], -1, dtype=np.int32)
    for t in range(states.shape[0]):
        clouds = states[t] == 1
        d = ndimage.distance_transform_edt(~clouds) if clouds.any() else np.full(clouds.shape, o["cloud_max"])
        cs = np.clip((np.minimum(d, o["cloud_max"]) - o["cloud_min"]) / (o["cloud_max"] - o["cloud_min"]), 0, 1)
        dd = abs(doy[t] - o["target_doy"]) % 365.25
        ds = np.exp(-0.5 * (min(dd, 365.25 - dd) / o["doy_sigma"]) ** 2)
        s = o["w_doy"] * ds + o["w_sensor"] * sensor[t] + o["w_cover"] * cover[t] + o["w_cloud"] * cs
        s = np.where(states[t] == 2, s, -np.inf)
        better = s > best_s
        best_s[better], best_t[better] = s[better], t
    return best_t


def synthetic():
    rng = np.random.default_rng(0)
    states = states_of(rng)
    doy = np.sort(rng.uniform(100, 300, T))
    sensor = np.where(rng.random(T) < 0.3, 0.5, 1.0)
    cover = rng.uniform(0.3, 1.0, T)
    t0 = time.perf_counter()
    ref = reference(states, doy, sensor, cover, OPTS)
    t_ref = time.perf_counter() - t0
    for n_jobs in (1, 4):
        bap.best_dates(states[:2], doy[:2], sensor[:2], cover[:2], **OPTS, n_jobs=n_jobs)
        t0 = time.perf_counter()
        date, score, clear = bap.best_dates(states, doy, sensor, cover, **OPTS, n_jobs=n_jobs)
        t = time.perf_counter() - t0
        same = (date == ref).mean()
        print(f"zeit-core bap, {n_jobs} thread(s): {t:.2f} s; numpy + scipy {t_ref:.2f} s ({t_ref / t:.1f}x); "
              f"the same date at {same:.4%} of the pixels")


def network():
    import zeit

    bbox = [-63.05, -9.95, -62.95, -9.85]   # Rondonia: the STAC guide's square
    for method in ("bap", "medoid"):
        t0 = time.perf_counter()
        cube = zeit.build_annual_composites(bbox=bbox, start_year=2023, end_year=2023, season=("06-01", "09-30"),
                                            bands=["red", "nir08", "swir16"], method=method, epsg=32720,
                                            resolution=30, cloud_cover_max=70, bap_info=method == "bap")
        v = cube.compute()
        n = np.isfinite(v.values[0, 0]).mean()
        print(f"{method}: {time.perf_counter() - t0:.1f} s (read and composited), {n:.1%} of the pixels with a value")


if __name__ == "__main__":
    synthetic()
    if "--network" in sys.argv:
        network()
