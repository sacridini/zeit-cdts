"""zeit.nbar and zeit.sentinel2_angles against NASA's HLS v2.0 products (a validation against a product, not against
another implementation: the NBAR of HLS is the same method, the c-factor of Roy et al. 2016, 2017, with the same
fixed parameters).

    python benchmarks/vs_originals/nbar_hls.py                  # all parts (needs the network)
    python benchmarks/vs_originals/nbar_hls.py --part l30       # HLS L30 against Landsat Collection 2 Level-2
    python benchmarks/vs_originals/nbar_hls.py --part s30       # HLS S30 against Sentinel-2 L2A (Earth Search)
    python benchmarks/vs_originals/nbar_hls.py --part angles    # sentinel2_angles against HLS S30's angle bands

HLS publishes the NBAR reflectance but not the reflectance before it, so the test compares HLS's NBAR with the
surface reflectance of the same scene from another processor and checks that the ratio R = NBAR_HLS / SR varies with
the view and sun geometry as Zeit's c-factor does:

* L30: HLS L30 (NBAR, built from the Landsat Level-1 with LaSRC) against USGS Collection 2 Level-2 (LaSRC too) of the
  same Landsat 9 scene and day, tile 23LKC, 2023-08-02 (both read from Planetary Computer onto HLS's 30 m grid, nearest
  neighbour; scale 2.75e-05, offset -0.2). c from Zeit with HLS's own angle bands (0.01 degree, fill 40000) and the
  ``NBAR_SOLAR_ZENITH`` tag of the file as the normalized zenith.
* S30: HLS S30 against Sentinel-2 L2A (Sen2Cor, Earth Search) of the tile 23LKC, 2023-07-23. HLS S30 is also
  bandpass-adjusted after the NBAR (the slope and offset are in the file's tags), which is undone before the ratio.
  c from Zeit with its own angles, ``zeit.sentinel2_angles`` from the tile's MTD_TL.xml.
* angles: the angles of ``zeit.sentinel2_angles`` against the SZA, SAA, VZA and VAA bands of HLS S30 (HLS reads the same
  ESA 5 km grids), over the whole tile (a decimated read, HLS's overviews are nearest neighbour) and in the windows.

For each band the pixels of the windows (clear in both products) are binned by view zenith (1 degree) and relative
azimuth (six sectors of 60 degrees), and the median of R is compared with the median of c in every bin with enough
pixels: the criterion of the specification is |median(R) - median(c)| below 0.01 for the green, red and NIR bands and
0.02 for the rest (0.03 against Sen2Cor), and the slope of (R - 1) on (c - 1) between 0.8 and 1.2.

The thread rule applies (8 at most): dask and GDAL are capped here.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GDAL_NUM_THREADS", "4")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")

import dask
import numpy as np
import pystac_client
import rasterio
import stackstac
import xarray as xr
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zeit  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "nbar-hls.json"
WORKERS = 4                               # dask threads of the reads
PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
ES = "https://earth-search.aws.element84.com/v1"

L30_ID, LANDSAT_ID = "HLS.L30.T23LKC.2023214T131426.v2.0", "LC09_L2SP_221071_20230802_02_T1"
S30_ID, S2_ID = "HLS.S30.T23LKC.2023204T132241.v2.0", "S2A_23LKC_20230723_0_L2A"
HLS_ULX, HLS_ULY, PIXEL, NPIX = 199980.0, -1699980.0, 30.0, 3660     # HLS grid (EPSG:32623, negative northings)

#: role -> (HLS L30 asset, Landsat C2 L2 asset), the bands HLS L30 normalizes
L30_BANDS = {"coastal": ("B01", "coastal"), "blue": ("B02", "blue"), "green": ("B03", "green"), "red": ("B04", "red"),
             "nir08": ("B05", "nir08"), "swir16": ("B06", "swir16"), "swir22": ("B07", "swir22")}
#: role (= the Earth Search asset) -> (HLS S30 asset, number in the bandpass tag); B05-B08 have no bandpass adjustment
S30_BANDS = {"coastal": ("B01", "01"), "blue": ("B02", "02"), "green": ("B03", "03"), "red": ("B04", "04"),
             "rededge1": ("B05", None), "rededge2": ("B06", None), "rededge3": ("B07", None), "nir": ("B08", None),
             "nir08": ("B8A", "8a"), "swir16": ("B11", "11"), "swir22": ("B12", "12")}
#: the tolerance of the bin medians by role (specification 5.2)
STRICT = ("green", "red", "nir08", "nir")


def catalog(url, signed):
    import planetary_computer

    return pystac_client.Client.open(url, modifier=planetary_computer.sign_inplace if signed else None)


def read(items, assets, bounds, epsg):
    """The assets of the items on the grid of `bounds` (30 m, the CRS `epsg`, nearest neighbour), raw values."""
    cube = stackstac.stack(items, assets=assets, bounds=bounds, epsg=epsg, resolution=PIXEL, dtype=np.float32,
                           rescale=False, fill_value=np.float32(np.nan), resampling=Resampling.nearest, chunksize=512,
                           snap_bounds=False)
    with dask.config.set(scheduler="threads", num_workers=WORKERS):
        out = cube.compute()
    # stackstac's coordinates may be the pixel's upper left corner: zeit wants the centres
    if abs(float(out.x[0]) - bounds[0]) < PIXEL / 4:
        out = out.assign_coords(x=out.x + PIXEL / 2, y=out.y - PIXEL / 2)
    return out


def read_hls(item, assets, r0, c0, size, shift=0.0):
    """A window of HLS layers in the tile's own grid (no resampling: it is the grid the other products are read
    onto), ``(band, y, x)`` raw values with the file's nodata as NaN, pixel-centre coordinates (`shift` metres added
    to the northings: 10,000,000 for the southern UTM zone's grid, HLS uses the northern zone's negative ones)."""
    from rasterio.windows import Window

    layers = []
    for a in assets:
        with rasterio.Env(GDAL_NUM_THREADS=WORKERS):
            with rasterio.open(item.assets[a].href) as r:
                v = r.read(1, window=Window(c0, r0, size, size)).astype(np.float32)
                if r.nodata is not None:
                    v[v == r.nodata] = np.nan
        layers.append(v)
    x = HLS_ULX + (c0 + np.arange(size) + 0.5) * PIXEL
    y = HLS_ULY + shift - (r0 + np.arange(size) + 0.5) * PIXEL
    return xr.DataArray(np.stack(layers), dims=("band", "y", "x"), coords={"band": list(assets), "y": y, "x": x})


def overview(href, factor):
    """A whole HLS layer decimated by `factor` (nearest neighbour: the pixel at ``factor * i + factor // 2``)."""
    with rasterio.Env(GDAL_NUM_THREADS=WORKERS):
        with rasterio.open(href) as r:
            n = r.width // factor
            return r.read(1, out_shape=(n, n), resampling=Resampling.nearest)


def hls_angle(raw):
    """HLS angle layers (uint16, 0.01 degree, fill 40000) in degrees, NaN at the fill."""
    raw = np.asarray(raw, dtype=np.float64)
    return np.where((raw == 40000) | ~np.isfinite(raw), np.nan, raw * 0.01).astype(np.float32)


def tags(href):
    with rasterio.open(href) as r:
        return r.tags()


def choose_windows(item, size, count):
    """Windows of `size` pixels in HLS's grid, clear and with view angles, spread over the view zenith: the lowest,
    the highest and the middle median (a decimated read of the tile, one pixel in ten)."""
    fac = 10
    vza = hls_angle(overview(item.assets["VZA"].href, fac))
    fmask = overview(item.assets["Fmask"].href, fac).astype(int)
    clear = (fmask & 0b00111110) == 0
    step, side = 500 // fac, size // fac
    found = []
    for r0 in range(0, vza.shape[0] - side + 1, step):
        for c0 in range(0, vza.shape[1] - side + 1, step):
            v, k = vza[r0:r0 + side, c0:c0 + side], clear[r0:r0 + side, c0:c0 + side]
            if np.isfinite(v).mean() >= 0.99 and k.mean() >= 0.85:
                found.append((float(np.nanmedian(v)), r0 * fac, c0 * fac, float(np.nanmax(v) - np.nanmin(v))))
    found.sort()
    if not found:
        raise SystemExit("no clear window with view angles")
    pick = [found[0], found[-1]] + [found[len(found) // 2]]
    out = []
    for f in pick[:count]:
        if f not in out:
            out.append(f)
    return [(r0, c0) for _, r0, c0, _ in out]


def bounds_of(r0, c0, size):
    return (HLS_ULX + c0 * PIXEL, HLS_ULY - (r0 + size) * PIXEL, HLS_ULX + (c0 + size) * PIXEL, HLS_ULY - r0 * PIXEL)


def fmask_clear(raw):
    """HLS Fmask: no cloud (bit 1), adjacent cloud (2), shadow (3), snow/ice (4) or water (5)."""
    return (np.nan_to_num(raw, nan=255).astype(int) & 0b00111110) == 0


def landsat_clear(raw):
    """Collection 2 QA_PIXEL: no fill (bit 0), dilated cloud (1), cirrus (2), cloud (3), shadow (4), snow (5), water (7)."""
    return (np.nan_to_num(raw, nan=1).astype(int) & 0b10111111) == 0


def cube_of(shape_like, bands, crs):
    """A cube of ones with the bands named for zeit.nbar's roles, on the grid of `shape_like` (a DataArray y, x)."""
    data = np.ones((1, len(bands), shape_like.sizes["y"], shape_like.sizes["x"]), dtype=np.float32)
    return xr.DataArray(data, dims=("time", "band", "y", "x"), name="ones",
                        coords={"time": np.array(["2023-01-01"], dtype="datetime64[ns]"), "band": list(bands),
                                "y": shape_like.y.values, "x": shape_like.x.values}).rio.write_crs(crs)


def slope(x, y, w=None):
    """The least-squares slope of y on x (weights `w`), NaN if x does not vary."""
    w = np.ones(x.size) if w is None else w
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    var = np.average((x - mx) ** 2, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / var) if var > 1e-10 else float("nan")


def binned(ratio, c, vza, raa, ok, role, tol, min_n):
    """Per band: the bins of 1 degree of view zenith x six sectors of relative azimuth with at least `min_n` pixels,
    the medians of R and c in them, and the regression of (R - 1) on (c - 1)."""
    r, cc, v, a = ratio[ok], c[ok], vza[ok], raa[ok]
    keep = np.isfinite(r) & np.isfinite(cc) & np.isfinite(v) & np.isfinite(a) & (np.abs(r - 1) < 0.5)
    r, cc, v, a = r[keep], cc[keep], v[keep], a[keep]
    key = np.floor(v).astype(int) * 6 + ((a % 360) // 60).astype(int)
    rows = []
    for k in np.unique(key):
        s = key == k
        if s.sum() >= min_n:
            rows.append((int(k // 6), int(k % 6), int(s.sum()), float(np.median(r[s])), float(np.median(cc[s]))))
    if len(rows) < 2:
        return {"pixels": int(keep.sum()), "bins": len(rows)}
    n = np.array([x[2] for x in rows], dtype=float)
    mr, mc = np.array([x[3] for x in rows]), np.array([x[4] for x in rows])
    diff = mr - mc
    bias = float(np.average(diff, weights=n))
    sl_bins = slope(mc - 1, mr - 1, n)
    sl_pix = slope(cc - 1, r - 1)
    return {
        "pixels": int(keep.sum()), "bins": len(rows), "c_min": float(cc.min()), "c_max": float(cc.max()),
        "c_std": float(cc.std()), "median_R": float(np.median(r)), "median_c": float(np.median(cc)),
        "max_abs_diff": float(np.abs(diff).max()), "mean_diff": bias,
        "max_abs_diff_without_bias": float(np.abs(diff - bias).max()),
        "tolerance": tol, "within_tolerance": bool(np.abs(diff).max() < tol),
        "slope_bins": sl_bins, "slope_pixels": sl_pix,
        "slope_in_0.8_1.2": bool(0.8 <= sl_bins <= 1.2) if np.isfinite(sl_bins) else None,
    }


def within_windows(d, tol, min_n):
    """The same bins after each window's own offset (median R - median c) is taken out of its R: the part of the
    disagreement that is the two processors' (aerosol, calibration, resampling) and not the geometry's."""
    adj = np.concatenate([r - (np.median(r) - np.median(c)) for r, c in zip(d["R"], d["c"]) if r.size])
    cat = {k: np.concatenate(v) for k, v in d.items()}
    out = binned(adj, cat["c"], cat["vza"], cat["raa"], np.ones(adj.size, bool), "", tol, min_n)
    return {k: out[k] for k in ("max_abs_diff", "slope_bins", "slope_pixels", "within_tolerance") if k in out}


def report(title, per_band, tol_strict, tol_other):
    print(f"\n{title}")
    print(f"{'band':9s} {'pixels':>9s} {'bins':>4s} {'c range':>13s} {'med R':>7s} {'med c':>7s} {'max|dmed|':>9s} "
          f"{'bias':>7s} {'w/o bias':>8s} {'tol':>5s} {'slope(bins)':>11s} {'slope(pix)':>10s}")
    for role, s in per_band.items():
        if "c_min" not in s:
            print(f"{role:9s} {s['pixels']:9d} {s['bins']:4d}  (too few bins)")
            continue
        print(f"{role:9s} {s['pixels']:9d} {s['bins']:4d} {s['c_min']:6.3f}-{s['c_max']:6.3f} {s['median_R']:7.4f} "
              f"{s['median_c']:7.4f} {s['max_abs_diff']:9.4f} {s['mean_diff']:7.4f} {s['max_abs_diff_without_bias']:8.4f} "
              f"{s['tolerance']:5.2f} {s['slope_bins']:11.3f} {s['slope_pixels']:10.3f}")
    print("after each window's own offset is removed from R (what is left is the geometry's part):")
    print(f"{'band':9s} {'max|dmed|':>9s} {'slope(bins)':>11s} {'slope(pix)':>10s}")
    for role, s in per_band.items():
        w = s.get("within_windows")
        if w:
            print(f"{role:9s} {w['max_abs_diff']:9.4f} {w['slope_bins']:11.3f} {w['slope_pixels']:10.3f}")


def l30(size, count, min_n):
    hls_cat = catalog(PC, True)
    item = hls_cat.get_collection("hls2-l30").get_item(L30_ID)
    landsat = hls_cat.get_collection("landsat-c2-l2").get_item(LANDSAT_ID)
    nbar_sza = float(tags(item.assets["B04"].href)["NBAR_SOLAR_ZENITH"])
    print(f"HLS L30 {L30_ID}: NBAR_SOLAR_ZENITH {nbar_sza:.4f}; Landsat {LANDSAT_ID}")
    wins = choose_windows(item, size, count)
    print("windows (row, column of the tile's pixel grid):", wins)
    acc = {r: {k: [] for k in ("R", "c", "vza", "raa")} for r in L30_BANDS}
    n_ok = 0
    for r0, c0 in wins:
        b = bounds_of(r0, c0, size)
        t = time.time()
        hls = read_hls(item, [v[0] for v in L30_BANDS.values()] + ["SZA", "SAA", "VZA", "VAA", "Fmask"], r0, c0, size)
        usgs = read([landsat], [v[1] for v in L30_BANDS.values()] + ["qa_pixel"], b, 32623).isel(time=0)
        print(f"  window {r0},{c0}: read in {time.time() - t:.0f} s, {hls.sizes['y']} x {hls.sizes['x']} pixels")
        names = [v[0] for v in L30_BANDS.values()]
        ang = xr.Dataset({k: (("time", "y", "x"), hls_angle(hls.sel(band=k).values)[None])
                          for k in ("SZA", "SAA", "VZA", "VAA")},
                         coords={"time": np.array(["2023-01-01"], dtype="datetime64[ns]"), "y": hls.y.values,
                                 "x": hls.x.values})
        ones = cube_of(hls, list(L30_BANDS), 32623)
        _, c = zeit.nbar(ones, ang, sza=nbar_sza, return_cfactor=True)
        c = c.values[0]
        clear = fmask_clear(hls.sel(band="Fmask").values) & landsat_clear(usgs.sel(band="qa_pixel").values)
        vza, saa, vaa = (ang[k].values[0] for k in ("VZA", "SAA", "VAA"))
        raa = (vaa - saa) % 360
        for i, role in enumerate(L30_BANDS):
            n = hls.sel(band=names[i]).values * 1e-4                               # HLS: scale 0.0001, nodata -9999
            n = np.where(hls.sel(band=names[i]).values <= -9000, np.nan, n)
            sr = usgs.sel(band=L30_BANDS[role][1]).values
            sr = np.where(sr == 0, np.nan, sr * 2.75e-05 - 0.2)
            good = clear & np.isfinite(n) & np.isfinite(sr) & (sr > 0.02) & (sr < 1.0) & (n > 0)
            acc[role]["R"].append((n / sr)[good])
            acc[role]["c"].append(c[i][good])
            acc[role]["vza"].append(vza[good])
            acc[role]["raa"].append(raa[good])
        n_ok += int(clear.sum())
    per_band = {}
    for role, d in acc.items():
        cat = {k: np.concatenate(v) for k, v in d.items()}
        tol = 0.01 if role in STRICT else 0.02
        per_band[role] = binned(cat["R"], cat["c"], cat["vza"], cat["raa"], np.ones(cat["R"].size, bool), role, tol, min_n)
        per_band[role]["within_windows"] = within_windows(d, tol, min_n)
    report(f"HLS L30 NBAR / USGS Collection 2 Level-2 against zeit's c-factor ({len(wins)} windows of {size} x {size})",
           per_band, 0.01, 0.02)
    return {"windows": wins, "size": size, "nbar_solar_zenith": nbar_sza, "bands": per_band}


def s30(size, count, min_n):
    hls_cat, es = catalog(PC, True), catalog(ES, False)
    item = hls_cat.get_collection("hls2-s30").get_item(S30_ID)
    s2 = es.get_collection("sentinel-2-l2a").get_item(S2_ID)
    t = tags(item.assets["B04"].href)
    nbar_sza = float(t["NBAR_SOLAR_ZENITH"])
    slopes = {}
    for role, (_, num) in S30_BANDS.items():
        if num is not None:
            a, b = (float(v) for v in t[f"MSI band {num} bandpass adjustment slope and offset"].split(","))
            slopes[role] = (a, b)
    print(f"HLS S30 {S30_ID}: NBAR_SOLAR_ZENITH {nbar_sza:.4f}; bandpass (slope, offset): {slopes}")
    wins = choose_windows(item, size, count)
    print("windows (row, column of the tile's pixel grid):", wins)
    # Earth Search lists scale 0.0001 and offset -0.1 for these assets, but an item with
    # earthsearch:boa_offset_applied has the offset already taken out of the stored values (checked: the 0.1th
    # percentile of B12 is DN 50, and a median red of 758 would be negative reflectance with the offset): none is added
    boa_offset = 0.0 if s2.properties.get("earthsearch:boa_offset_applied") else -0.1
    print("Sentinel-2 L2A reflectance = DN * 1e-4 %+g" % boa_offset)
    # the tile's pixel grid in EPSG:32723: the same grid as HLS's (the northings differ by 10,000,000 m)
    acc = {r: {k: [] for k in ("R", "c", "vza", "raa")} for r in S30_BANDS}
    angle_stats = []
    mtd = s2.assets["granule_metadata"].href
    for r0, c0 in wins:
        bx = bounds_of(r0, c0, size)
        b = (bx[0], bx[1] + 10_000_000.0, bx[2], bx[3] + 10_000_000.0)
        t0 = time.time()
        hls = read_hls(item, [v[0] for v in S30_BANDS.values()] + ["SZA", "SAA", "VZA", "VAA", "Fmask"], r0, c0, size,
                       shift=10_000_000.0)
        sr = read([s2], list(S30_BANDS) + ["scl"], b, 32723).isel(time=0)
        print(f"  window {r0},{c0}: read in {time.time() - t0:.0f} s, {hls.sizes['y']} x {hls.sizes['x']} pixels")
        ones = cube_of(hls, list(S30_BANDS), 32723)
        ones = ones.assign_coords(time=np.array(["2023-07-23"], dtype="datetime64[ns]"))
        t1 = time.time()
        ang = zeit.sentinel2_angles(mtd, ones)
        dt_ang = time.time() - t1
        _, c = zeit.nbar(ones, ang, sza=nbar_sza, return_cfactor=True)
        c = c.values[0]
        theirs = {k: hls_angle(hls.sel(band=k).values) for k in ("SZA", "SAA", "VZA", "VAA")}
        angle_stats.append(angle_differences(ang, theirs, f"window {r0},{c0}", dt_ang))
        clear = fmask_clear(hls.sel(band="Fmask").values) & np.isin(sr.sel(band="scl").values, (4, 5))
        vza, saa, vaa = (ang[k].values[0] for k in ("VZA", "SAA", "VAA"))
        raa = (vaa - saa) % 360
        for i, role in enumerate(S30_BANDS):
            raw = hls.sel(band=S30_BANDS[role][0]).values
            n = np.where(raw <= -9000, np.nan, raw * 1e-4)
            if role in slopes:                                    # undo HLS's bandpass adjustment: out = a * nbar + b
                n = (n - slopes[role][1]) / slopes[role][0]
            s = sr.sel(band=role).values
            s = np.where(s == 0, np.nan, s * 1e-4 + boa_offset)
            good = clear & np.isfinite(n) & np.isfinite(s) & (s > 0.02) & (s < 1.0) & (n > 0)
            acc[role]["R"].append((n / s)[good])
            acc[role]["c"].append(c[i][good])
            acc[role]["vza"].append(vza[good])
            acc[role]["raa"].append(raa[good])
    per_band = {}
    for role, d in acc.items():
        cat = {k: np.concatenate(v) for k, v in d.items()}
        tol = 0.03
        per_band[role] = binned(cat["R"], cat["c"], cat["vza"], cat["raa"], np.ones(cat["R"].size, bool), role, tol, min_n)
        per_band[role]["within_windows"] = within_windows(d, tol, min_n)
    report(f"HLS S30 NBAR (bandpass undone) / Sentinel-2 L2A against zeit's c-factor with sentinel2_angles "
           f"({len(wins)} windows of {size} x {size})", per_band, 0.03, 0.03)
    return {"windows": wins, "size": size, "nbar_solar_zenith": nbar_sza, "bandpass": slopes, "bands": per_band,
            "angles_in_windows": angle_stats}


def angle_differences(ours, theirs, label, seconds):
    """Zeit's angles (a Dataset (time, y, x)) minus HLS's (arrays y, x): statistics in degrees; azimuths circular."""
    out = {"where": label, "seconds": seconds}
    for k in ("SZA", "SAA", "VZA", "VAA"):
        d = ours[k].values[0].astype(np.float64) - theirs[k].astype(np.float64)
        if k.endswith("A") and k[1:] == "AA":
            d = (d + 180) % 360 - 180
        d = d[np.isfinite(d)]
        out[k] = {"n": int(d.size), "mean": float(d.mean()), "std": float(d.std()), "max_abs": float(np.abs(d).max()),
                  "p99_abs": float(np.percentile(np.abs(d), 99))}
    return out


def angles_whole_tile():
    """The angles of zeit.sentinel2_angles at the pixels of a decimated read of HLS S30's angle bands."""
    hls_cat, es = catalog(PC, True), catalog(ES, False)
    item = hls_cat.get_collection("hls2-s30").get_item(S30_ID)
    s2 = es.get_collection("sentinel-2-l2a").get_item(S2_ID)
    fac = 10
    theirs = {k: hls_angle(overview(item.assets[k].href, fac)) for k in ("SZA", "SAA", "VZA", "VAA")}
    n = theirs["SZA"].shape[0]
    # the pixel i of the decimated read is the pixel fac * i + fac // 2 of the tile (UTM 23S: add 10,000,000 m)
    cols = fac * np.arange(n) + fac // 2
    x = HLS_ULX + (cols + 0.5) * PIXEL
    y = HLS_ULY + 10_000_000.0 - (cols + 0.5) * PIXEL
    like = xr.DataArray(np.zeros((1, 1, n, n), np.float32), dims=("time", "band", "y", "x"),
                        coords={"time": np.array(["2023-07-23"], dtype="datetime64[ns]"), "band": ["B06"], "y": y,
                                "x": x}).rio.write_crs(32723)
    t = time.time()
    ours = zeit.sentinel2_angles(s2, like)
    secs = time.time() - t
    stats = angle_differences(ours, theirs, f"the whole tile, {n} x {n} pixels one in {fac}", secs)
    print("\nzeit.sentinel2_angles - HLS S30 angle bands (degrees), the whole tile:")
    print_angle_stats(stats)
    # what the differences do to the c-factor: c from both sets of angles, normalized zenith as in the file
    nbar_sza = float(tags(item.assets["B04"].href)["NBAR_SOLAR_ZENITH"])
    ones = xr.concat([like.copy(data=np.ones((1, 1, n, n), np.float32)).assign_coords(band=[b])
                      for b in ("blue", "red", "rededge1", "nir08", "swir16")], dim="band")
    theirs_ds = xr.Dataset({k: (("time", "y", "x"), v[None]) for k, v in theirs.items()},
                           coords={"time": like.time, "y": like.y, "x": like.x})
    _, c_ours = zeit.nbar(ones, ours, sza=nbar_sza, return_cfactor=True)
    _, c_hls = zeit.nbar(ones, theirs_ds, sza=nbar_sza, return_cfactor=True)
    stats["c_difference"] = {}
    print("c from zeit's angles minus c from HLS's angles (the normalized zenith of the file):")
    for b in c_ours.band.values:
        d = (c_ours.sel(band=b).values - c_hls.sel(band=b).values).ravel()
        d = d[np.isfinite(d)]
        stats["c_difference"][str(b)] = {"mean": float(d.mean()), "std": float(d.std()), "max_abs": float(np.abs(d).max()),
                                         "p99_abs": float(np.percentile(np.abs(d), 99))}
        v = stats["c_difference"][str(b)]
        print(f"  {str(b):9s} mean {v['mean']:+.5f}  std {v['std']:.5f}  max |d| {v['max_abs']:.5f}  99% |d| {v['p99_abs']:.5f}")
    return stats


def print_angle_stats(s):
    for k in ("SZA", "SAA", "VZA", "VAA"):
        v = s[k]
        print(f"  {k}: n {v['n']:8d}  mean {v['mean']:+.5f}  std {v['std']:.5f}  max |d| {v['max_abs']:.5f}  "
              f"99% |d| {v['p99_abs']:.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=["l30", "s30", "angles", "all"], default="all")
    ap.add_argument("--size", type=int, default=1000, help="window side in 30 m pixels")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--min-pixels", type=int, default=300, help="smallest bin that counts")
    args = ap.parse_args()
    if os.name == "nt":
        try:
            from common import no_power_throttling

            no_power_throttling()
        except Exception as e:                  # a benchmark nicety, not needed here
            print("power throttling not changed:", e)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.part in ("l30", "all"):
        results["l30"] = l30(args.size, args.windows, args.min_pixels)
    if args.part in ("s30", "all"):
        results["s30"] = s30(args.size, args.windows, args.min_pixels)
        for a in results["s30"]["angles_in_windows"]:
            print(f"\nangles, {a['where']} ({a['seconds']:.2f} s for sentinel2_angles):")
            print_angle_stats(a)
    if args.part in ("angles", "all"):
        results["angles_whole_tile"] = angles_whole_tile()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print("\nwritten", OUT)


if __name__ == "__main__":
    main()
