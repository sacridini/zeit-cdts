"""A synthetic ARD cube in FORCE's format (one tile of Sentinel-2A BOA + QAI), for the TSA/CSO comparison.

Runs in the Python of the ``force-ref`` conda env inside WSL (numpy + GDAL), no Zeit involved::

    python -I force_tsa_cube.py <ard_dir> [--size 100] [--years 2018 2022] [--seed 1]

The series are what the TSA exercises: an irregular acquisition calendar (a ~5-day revisit with ~10% of the dates
missing, never 29 February: FORCE's 365-day calendar puts it on 1 March), a double-logistic seasonal NDVI whose
amplitude and phase change per pixel, noise, a linear trend in a third of the pixels and an abrupt loss in 2020
(followed by a recovery) in another third, and clouds, shadows and nodata in patches on ~40% of the observations.
``truth.npz`` (next to the ``ard`` folder) keeps the dates and the parameters.
"""
import argparse
import datetime
import os

import numpy as np
from osgeo import gdal, osr

NAMES = ["BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2"]
NODATA = -9999
ORIGIN = (500000.0, 5000000.0)
RES = 30.0


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_dates(y0, y1, rng):
    """Every 5 days from 1 January of y0, ~10% dropped at random, 29 February removed."""
    d = datetime.date(y0, 1, 1)
    end = datetime.date(y1, 12, 31)
    out = []
    while d <= end:
        if not (d.month == 2 and d.day == 29):
            out.append(d)
        d += datetime.timedelta(days=5)
    keep = rng.random(len(out)) >= 0.10
    return [x for x, k in zip(out, keep) if k]


def blobs(n, rng, count, rmin, rmax):
    """Boolean mask of ``count`` discs of random centre and radius (pixels)."""
    yy, xx = np.mgrid[0:n, 0:n]
    m = np.zeros((n, n), bool)
    for _ in range(count):
        cy, cx = rng.uniform(0, n, 2)
        r = rng.uniform(rmin, rmax)
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return m


def dilate(m, k):
    """Square dilation by ``k`` pixels (the cloud buffer)."""
    out = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def write_tif(path, arr, wkt, res):
    drv = gdal.GetDriverByName("GTiff")
    n = arr.shape[1]
    ds = drv.Create(path, arr.shape[2], n, arr.shape[0], gdal.GDT_Int16,
                    ["COMPRESS=DEFLATE", "TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=64"])
    ds.SetGeoTransform((ORIGIN[0], res, 0, ORIGIN[1], 0, -res))
    ds.SetProjection(wkt)
    for b in range(arr.shape[0]):
        bd = ds.GetRasterBand(b + 1)
        bd.WriteArray(arr[b])
        bd.SetNoDataValue(NODATA if arr.shape[0] == 10 else 1)
    ds = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="output folder; the cube goes to <root>/ard, the truth to <root>/truth.npz")
    ap.add_argument("--size", type=int, default=100, help="pixels per side (one tile, 30 m)")
    ap.add_argument("--years", type=int, nargs=2, default=[2018, 2022])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    n = a.size
    rng = np.random.default_rng(a.seed)

    ard = os.path.join(a.root, "ard")
    tile = os.path.join(ard, "X0000_Y0000")
    os.makedirs(tile, exist_ok=True)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wkt = srs.ExportToWkt()
    with open(os.path.join(ard, "datacube-definition.prj"), "w") as f:
        f.write("PROJECTION = %s\n" % wkt)
        f.write("ORIGIN_GEO_X = 15.000000\nORIGIN_GEO_Y = 45.150000\n")
        f.write("ORIGIN_MAP_X = %f\nORIGIN_MAP_Y = %f\n" % ORIGIN)
        f.write("TILE_SIZE_X = %f\nTILE_SIZE_Y = %f\n" % (n * RES, n * RES))

    dates = make_dates(a.years[0], a.years[1], rng)
    nt = len(dates)
    t_years = np.array([(d - datetime.date(a.years[0], 1, 1)).days / 365.0 for d in dates])
    doys = np.array([d.timetuple().tm_yday for d in dates], float)

    # Per-pixel parameters.
    base = rng.uniform(0.15, 0.30, (n, n))
    amp = rng.uniform(0.35, 0.60, (n, n))
    start = rng.uniform(90, 150, (n, n))       # green-up (doy)
    end = rng.uniform(250, 310, (n, n))        # senescence (doy)
    w1 = rng.uniform(8, 16, (n, n))
    w2 = rng.uniform(10, 20, (n, n))
    nir_level = rng.uniform(3000, 4500, (n, n))
    group = rng.permutation(n * n).reshape(n, n) % 3  # 0: trend, 1: disturbance, 2: none
    slope = np.where(group == 0, rng.choice([-1, 1], (n, n)) * rng.uniform(0.01, 0.04, (n, n)), 0.0)  # NDVI/year
    dist_doy = rng.uniform(100, 260, (n, n))   # disturbance date in 2020
    dist_date = np.array([datetime.date(2020, 1, 1) + datetime.timedelta(days=int(x) - 1) for x in dist_doy.ravel()])
    t_dist = np.array([(d - datetime.date(a.years[0], 1, 1)).days / 365.0 for d in dist_date]).reshape(n, n)
    dist_depth = np.where(group == 1, rng.uniform(0.3, 0.6, (n, n)), 0.0)
    recov_tau = rng.uniform(0.6, 1.5, (n, n))  # years

    masked = np.zeros((nt, n, n), bool)
    cloudy_obs = rng.random(nt) < 0.40
    for i, d in enumerate(dates):
        doy = doys[i]
        ndvi = base + amp * (logistic((doy - start) / w1) - logistic((doy - end) / w2))
        ndvi = ndvi + slope * (t_years[i] - 2.5)
        dt = t_years[i] - t_dist
        ndvi = ndvi - np.where(dt >= 0, dist_depth * np.exp(-np.maximum(dt, 0) / recov_tau), 0.0)
        ndvi = np.clip(ndvi + rng.normal(0, 0.025, (n, n)), 0.02, 0.95)
        nir = nir_level + rng.normal(0, 60, (n, n))
        red = nir * (1 - ndvi) / (1 + ndvi)
        refl = {"BLUE": red * 0.5, "GREEN": red * 1.0 + 200, "RED": red, "REDEDGE1": red + 0.3 * (nir - red),
                "REDEDGE2": red + 0.7 * (nir - red), "REDEDGE3": red + 0.8 * (nir - red),
                "BROADNIR": nir * 1.02, "NIR": nir, "SWIR1": 2200 - 900 * ndvi, "SWIR2": 1500 - 700 * ndvi}
        arr = np.zeros((10, n, n), np.float64)
        for b, nm in enumerate(NAMES):
            arr[b] = refl[nm] + rng.normal(0, 30, (n, n))

        qai = np.zeros((n, n), np.int16)
        if cloudy_obs[i]:
            cloud = blobs(n, rng, rng.integers(1, 4), n * 0.08, n * 0.25)
            buf = dilate(cloud, max(1, n // 40)) & ~cloud
            shadow = np.roll(np.roll(cloud, max(2, n // 20), 0), max(2, n // 20), 1) & ~cloud & ~buf
            qai[buf] |= 1 << 1            # cloud state 1: buffer (less confident cloud)
            qai[cloud] |= 2 << 1          # cloud state 2: opaque
            qai[shadow] |= 1 << 3
            arr[:, cloud] = (arr[:, cloud] * 0.2 + np.array([3200, 3100, 3000, 3000, 3000, 3000, 3000, 3000, 2500, 2000])[:, None] * 0.8)
            arr[:, shadow] *= 0.5
            masked[i] = cloud | buf | shadow
        if rng.random() < 0.05:           # a swath without data
            x0 = int(rng.integers(0, n - n // 5))
            nd = np.zeros((n, n), bool)
            nd[:, x0:x0 + n // 5] = True
            qai[nd] = 1
            arr[:, nd] = NODATA
            masked[i] |= nd
        out = np.clip(np.round(arr), -32000, 32000).astype(np.int16)
        out[:, (qai & 1) == 1] = NODATA
        stamp = d.strftime("%Y%m%d")
        write_tif(os.path.join(tile, "%s_LEVEL2_SEN2A_BOA.tif" % stamp), out, wkt, RES)
        write_tif(os.path.join(tile, "%s_LEVEL2_SEN2A_QAI.tif" % stamp), qai[None], wkt, RES)

    np.savez_compressed(
        os.path.join(a.root, "truth.npz"),
        dates=np.array([int(d.strftime("%Y%m%d")) for d in dates], np.int32),
        masked=masked, group=group, slope=slope, dist_depth=dist_depth, dist_doy=dist_doy,
        base=base, amp=amp, start=start, end=end, w1=w1, w2=w2, nir_level=nir_level, recov_tau=recov_tau,
        cloudy_obs=cloudy_obs, size=n, years=np.array(a.years))
    print(nt, "dates,", int(cloudy_obs.sum()), "cloudy; written to", tile)


if __name__ == "__main__":
    main()
