"""A synthetic ARD cube for FORCE's CF-ImproPhe and L2-ImproPhe: Sentinel-2A at 10 m (the fine data), Landsat 8 at 30 m
(L2-ImproPhe's coarse scenes) and a continuous field at 30 m or 90 m (CF-ImproPhe's), all made from one truth map.

Runs in the Python of the ``force-ref`` conda env inside WSL (numpy + GDAL), no Zeit involved::

    python -I force_improphe_cube.py <root> [--size 297] [--cf-factor 3] [--seed 1]

``<root>/ard`` is a FORCE datacube (``datacube-definition.prj``, one tile ``X0000_Y0000`` of ``size`` x ``size`` pixels of
10 m with the Sentinel-2A BOA and QAI and the Landsat 8 BOA and QAI at 30 m) and ``<root>/coarse/X0000_Y0000/FIELD.tif`` is
the field (one int16 band per year, NoData -9999). ``<root>/truth.npz`` has the truth (classes, the field and the Landsat
bands at 10 m) and the dates.

The truth (spec 4.3 of the FORCE specification): five classes in patches (Voronoi cells of random seeds, four per class),
each a mix of two spectra whose weight follows its own season (crop, forest, grass, water, urban), a smooth gradient, a static
per-pixel brightness texture whose strength changes from patch to patch, and a little noise per date. The field is the year's
peak NDVI of the truth, averaged over blocks of ``cf-factor`` pixels (the coarse grid), times 10000 (2018 and 2019 differ:
the vegetation's amplitude is 1.0 and 0.8 of its own).

The edge cases: every seasonal window has data except, in 2019, doy 91-250 (so a window is empty in that year); doy 1, 91,
181, 271 and 365 are observed in 2018 (the windows' limits); clouds, shadows and swaths without data on about a third of the
Sentinel-2 dates; a column (``size // 2``) without any Sentinel-2 data; a hole (QAI NoData) in the Landsat scenes and one in
each year of the field.
"""
import argparse
import datetime
import os

import numpy as np
from osgeo import gdal, osr

NODATA = -9999
ORIGIN = (500000.0, 5000000.0)
RES = 10.0
LND_BANDS = [0, 1, 2, 7, 8, 9]   # BLUE GREEN RED NIR SWIR1 SWIR2 among the ten Sentinel-2 bands

# (vegetated/wet spectrum, dry/bare spectrum, amplitude, phase) in BLUE GREEN RED NIR SWIR1 SWIR2 (reflectance x 10000)
CLASSES = [
    ([300, 600, 350, 4200, 1800, 800], [1200, 1700, 2200, 2800, 3800, 3200], 1.0, 100),   # crop
    ([250, 500, 300, 3800, 1600, 700], [350, 650, 450, 3200, 1900, 900], 0.3, 130),       # forest
    ([300, 600, 350, 4200, 1800, 800], [600, 900, 1300, 2600, 3200, 2300], 0.7, 80),      # grass
    ([600, 700, 450, 250, 100, 60], [700, 800, 550, 300, 120, 70], 0.2, 0),               # water
    ([1500, 1700, 1900, 2200, 2600, 2300], [1400, 1600, 1800, 2100, 2500, 2200], 0.1, 0),  # urban
]
YEAR_AMP = {2018: 1.0, 2019: 0.8}


def ten_bands(b6):
    """``(6, ...)`` BLUE GREEN RED NIR SWIR1 SWIR2 to the ten Sentinel-2 bands (red edge between red and NIR)."""
    blue, green, red, nir, sw1, sw2 = b6
    return np.stack([blue, green, red, red + 0.3 * (nir - red), red + 0.7 * (nir - red), red + 0.8 * (nir - red),
                     nir * 1.02, nir, sw1, sw2])


def blobs(n, rng, count, rmin, rmax):
    yy, xx = np.mgrid[0:n, 0:n]
    m = np.zeros((n, n), bool)
    for _ in range(count):
        cy, cx = rng.uniform(0, n, 2)
        r = rng.uniform(rmin, rmax)
        m |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return m


def dilate(m, k):
    out = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def write_tif(path, arr, wkt, res, nodata=NODATA):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, arr.shape[2], arr.shape[1], arr.shape[0], gdal.GDT_Int16,
                    ["COMPRESS=DEFLATE", "TILED=YES", "BLOCKXSIZE=64", "BLOCKYSIZE=64"])
    ds.SetGeoTransform((ORIGIN[0], res, 0, ORIGIN[1], 0, -res))
    ds.SetProjection(wkt)
    for b in range(arr.shape[0]):
        bd = ds.GetRasterBand(b + 1)
        bd.WriteArray(arr[b])
        bd.SetNoDataValue(nodata)
    ds = None


def voronoi(n, rng, cells):
    seeds = rng.uniform(0, n, (cells, 2))
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), np.inf)
    idx = np.zeros((n, n), np.int32)
    for i, (sy, sx) in enumerate(seeds):
        d = (yy - sy) ** 2 + (xx - sx) ** 2
        better = d < best
        best[better] = d[better]
        idx[better] = i
    return idx


class Truth:
    """The 10 m truth: ``refl(doy, year)`` is the ``(6, n, n)`` float reflectance."""

    def __init__(self, n, rng, band_noise=0.0):
        self.n = n
        cell = voronoi(n, rng, 20)
        assign = np.tile(np.arange(5), 4)
        rng.shuffle(assign)
        self.cls = assign[cell]
        tex_cell = voronoi(n, rng, 12)
        sigma = rng.uniform(0.004, 0.05, 12)[tex_cell]
        yy, xx = np.mgrid[0:n, 0:n]
        grad = (xx + yy) / (2.0 * n) - 0.5
        self.bright = (1.0 + 0.3 * grad) * (1.0 + sigma * rng.normal(0, 1, (n, n)))
        # a static factor of its own for each band and pixel: more independent directions in the data, so that more
        # principal components are kept (0 draws nothing, the random stream is not changed)
        self.band_factor = (1.0 + band_noise * rng.normal(0, 1, (6, n, n))) if band_noise > 0 else None
        self.vegs = np.array([c[0] for c in CLASSES], float)       # (5, 6)
        self.drys = np.array([c[1] for c in CLASSES], float)
        self.amp = np.array([c[2] for c in CLASSES])
        self.phase = np.array([c[3] for c in CLASSES], float)

    def refl(self, doy, year):
        cls = self.cls
        w = 0.5 + 0.5 * self.amp[cls] * YEAR_AMP[year] * np.sin(2 * np.pi * (doy - self.phase[cls]) / 365.0)
        w = np.clip(w, 0, 1)
        spec = w[None] * self.vegs[cls].T + (1 - w)[None] * self.drys[cls].T       # (6, n, n)
        spec = spec * self.bright[None]
        return spec if self.band_factor is None else spec * self.band_factor

    def peak_ndvi(self, year):
        best = np.full((self.n, self.n), -1.0)
        for doy in range(1, 366, 5):
            r = self.refl(doy, year)
            ndvi = (r[3] - r[2]) / (r[3] + r[2])
            best = np.maximum(best, ndvi)
        return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--size", type=int, default=297, help="pixels per side of the 10 m tile (a multiple of 9)")
    ap.add_argument("--cf-factor", type=int, default=3, help="pixels of the fine grid per pixel of the field")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--band-noise", type=float, default=0.0,
                    help="a static relative noise of this size on every band of every pixel (more components)")
    ap.add_argument("--s2-per-year", type=int, default=55, help="Sentinel-2 dates a year (random doys)")
    ap.add_argument("--l8-per-year", type=int, default=12)
    a = ap.parse_args()
    n, f = a.size, a.cf_factor
    if n % 9 or n % f:
        raise SystemExit("size must be a multiple of 9 and of cf-factor")
    rng = np.random.default_rng(a.seed)
    tr = Truth(n, rng, a.band_noise)

    ard = os.path.join(a.root, "ard")
    tile = os.path.join(ard, "X0000_Y0000")
    coarse_dir = os.path.join(a.root, "coarse", "X0000_Y0000")
    os.makedirs(tile, exist_ok=True)
    os.makedirs(coarse_dir, exist_ok=True)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32633)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    wkt = srs.ExportToWkt()
    with open(os.path.join(ard, "datacube-definition.prj"), "w") as fh:
        fh.write("PROJECTION = %s\n" % wkt)
        fh.write("ORIGIN_GEO_X = 15.000000\nORIGIN_GEO_Y = 45.150000\n")
        fh.write("ORIGIN_MAP_X = %f\nORIGIN_MAP_Y = %f\n" % ORIGIN)
        fh.write("TILE_SIZE_X = %f\nTILE_SIZE_Y = %f\n" % (n * RES, n * RES))

    # ---- Sentinel-2A: the fine data -------------------------------------------------------------------------------
    s2 = []
    for year in (2018, 2019):
        doys = set(rng.choice(np.arange(1, 366), a.s2_per_year, replace=False).tolist())
        if year == 2018:
            doys |= {1, 91, 181, 271, 365}
        else:
            doys = {d for d in doys if not 91 <= d <= 250}
        s2 += [(year, d) for d in sorted(doys)]
    col = n // 2
    cloudy = rng.random(len(s2)) < 0.35
    nodata_scenes = 0
    s2_dates = []
    for i, (year, doy) in enumerate(s2):
        date = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
        s2_dates.append(int(date.strftime("%Y%m%d")))
        r = ten_bands(tr.refl(doy, year))
        r = r * (1 + rng.normal(0, 0.01, r.shape)) + rng.normal(0, 15, r.shape)
        qai = np.zeros((n, n), np.int16)
        if cloudy[i]:
            cloud = blobs(n, rng, int(rng.integers(1, 4)), n * 0.06, n * 0.2)
            buf = dilate(cloud, max(1, n // 60)) & ~cloud
            sh = np.roll(np.roll(cloud, max(2, n // 30), 0), max(2, n // 30), 1) & ~cloud & ~buf
            qai[buf] |= 1 << 1
            qai[cloud] |= 2 << 1
            qai[sh] |= 1 << 3
            cloud_refl = np.array([3200, 3100, 3000, 3000, 3000, 3000, 3000, 3000, 2500, 2000], float)
            r[:, cloud] = r[:, cloud] * 0.2 + cloud_refl[:, None] * 0.8
            r[:, sh] *= 0.5
        if i % 19 == 7 or (nodata_scenes == 0 and i == 3):   # a swath without data
            x0 = int(rng.integers(0, n - n // 6))
            qai[:, x0:x0 + n // 6] = 1
            nodata_scenes += 1
        qai[:, col] = 1                                       # the column without any fine data
        out = np.clip(np.round(r), -32000, 32000).astype(np.int16)
        out[:, (qai & 1) == 1] = NODATA
        stamp = date.strftime("%Y%m%d")
        write_tif(os.path.join(tile, "%s_LEVEL2_SEN2A_BOA.tif" % stamp), out, wkt, RES)
        write_tif(os.path.join(tile, "%s_LEVEL2_SEN2A_QAI.tif" % stamp), qai[None], wkt, RES, 1)

    # ---- Landsat 8 at 30 m: L2-ImproPhe's coarse scenes ----------------------------------------------------------
    m = n // 3
    l8 = []
    for year in (2018, 2019):
        l8 += [(year, d) for d in sorted(rng.choice(np.arange(1, 366), a.l8_per_year, replace=False).tolist())]
    l8_dates, l8_truth = [], []
    for i, (year, doy) in enumerate(l8):
        date = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
        l8_dates.append(int(date.strftime("%Y%m%d")))
        t10 = tr.refl(doy, year)[:, :n, :n]                                           # (6, n, n)
        l8_truth.append(np.round(t10).astype(np.int16))
        r = t10.reshape(6, m, 3, m, 3).mean((2, 4))
        r = r * (1 + rng.normal(0, 0.01, r.shape)) + rng.normal(0, 15, r.shape)
        qai = np.zeros((m, m), np.int16)
        if rng.random() < 0.3:
            cloud = blobs(m, rng, int(rng.integers(1, 3)), m * 0.05, m * 0.15)
            qai[cloud] |= 2 << 1
            r[:, cloud] = r[:, cloud] * 0.2 + 3000 * 0.8
        qai[m // 6:m // 6 + 5, m // 3:m // 3 + 5] = 1                                 # the hole
        out = np.clip(np.round(r), -32000, 32000).astype(np.int16)
        out[:, (qai & 1) == 1] = NODATA
        stamp = date.strftime("%Y%m%d")
        write_tif(os.path.join(tile, "%s_LEVEL2_LND08_BOA.tif" % stamp), out, wkt, 3 * RES)
        write_tif(os.path.join(tile, "%s_LEVEL2_LND08_QAI.tif" % stamp), qai[None], wkt, 3 * RES, 1)

    # ---- the field: the year's peak NDVI, block means of cf-factor pixels ----------------------------------------
    mc = n // f
    peak = np.stack([tr.peak_ndvi(y) for y in (2018, 2019)])                           # (2, n, n) at 10 m
    field = peak.reshape(2, mc, f, mc, f).mean((2, 4))
    cf = np.round(field * 10000).astype(np.int16)
    cf[0, mc // 5:mc // 5 + 4, mc // 2:mc // 2 + 4] = NODATA                           # a hole in 2018
    cf[1, mc // 2:mc // 2 + 4, mc // 6:mc // 6 + 4] = NODATA                           # another in 2019
    write_tif(os.path.join(coarse_dir, "FIELD.tif"), cf, wkt, f * RES)

    np.savez_compressed(
        os.path.join(a.root, "truth.npz"), cls=tr.cls, peak_ndvi=peak.astype(np.float32), size=n, cf_factor=f,
        s2_dates=np.array(s2_dates, np.int32), l8_dates=np.array(l8_dates, np.int32), l8_truth=np.stack(l8_truth),
        column=col, cloudy=cloudy)
    print(len(s2), "Sentinel-2A dates,", int(cloudy.sum()), "cloudy,", len(l8), "Landsat 8 dates; tile", tile)


if __name__ == "__main__":
    main()
