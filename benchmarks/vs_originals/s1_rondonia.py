"""A real Sentinel-1 series for the omnibus benchmark: the RTC gamma0 (VV, VH, linear power) of Planetary Computer's
``sentinel-1-rtc`` over the BR-364 deforestation frontier in Rondonia, one relative orbit and pass (the one with the
most scenes), June 2021 to June 2022, as one two-band GeoTIFF per date on a common 10 m grid.

Needs pystac-client, planetary-computer, odc-stac and rasterio (a separate environment, not Zeit's): the path of its
Python goes in ``S1_PYTHON`` when imad.py's neighbour ``omnibus.py`` calls ``fetch`` through it, or run this file in it:

    python s1_rondonia.py <folder> [size_in_pixels]
"""
import json
import sys
from pathlib import Path

BBOX = (-63.30, -9.90, -63.10, -9.70)     # lon, lat: a 22 km square of forest, pasture and clearings
DATETIME = "2021-06-01/2022-06-30"


def fetch(folder, size=400, bbox=BBOX):
    """Writes ``s1_<date>.tif`` (VV, VH) of the central ``size`` x ``size`` pixels of the box and ``info.json``; returns
    the folder's files. Cached: nothing is read again if ``info.json`` is there."""
    import collections

    import dask
    import numpy as np
    import odc.stac
    import planetary_computer as pc
    import pystac_client
    import rasterio
    from rasterio.transform import from_origin

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    info = folder / "info.json"
    if info.exists():
        return sorted(folder.glob("s1_*.tif"))
    dask.config.set(scheduler="threads", num_workers=4)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=list(bbox), datetime=DATETIME).items())
    groups = collections.Counter((i.properties.get("sat:relative_orbit"), i.properties.get("sat:orbit_state"))
                                 for i in items)
    (orbit, state), count = groups.most_common(1)[0]
    items = [i for i in items if (i.properties.get("sat:relative_orbit"), i.properties.get("sat:orbit_state")) == (orbit, state)]
    ds = odc.stac.load(items, bands=["vv", "vh"], bbox=bbox, resolution=10, groupby="solar_day", dtype="float32",
                       chunks={"x": 1024, "y": 1024}, crs="EPSG:32720")
    ny, nx = ds.sizes["y"], ds.sizes["x"]
    y0, x0 = (ny - size) // 2, (nx - size) // 2
    ds = ds.isel(y=slice(y0, y0 + size), x=slice(x0, x0 + size)).compute()
    xs, ys = ds.x.values, ds.y.values
    transform = from_origin(float(xs[0]) - 5.0, float(ys[0]) + 5.0, 10.0, 10.0)
    dates = []
    for k in range(ds.sizes["time"]):
        arr = np.stack([ds["vv"].values[k], ds["vh"].values[k]]).astype("float32")
        if not np.isfinite(arr).any():
            continue
        date = str(ds.time.values[k])[:10]
        dates.append(date)
        with rasterio.open(folder / f"s1_{date.replace('-', '')}.tif", "w", driver="GTiff", count=2, dtype="float32",
                           width=size, height=size, crs="EPSG:32720", transform=transform, nodata=float("nan")) as dst:
            dst.write(arr)
    info.write_text(json.dumps({"bbox": list(bbox), "orbit": orbit, "state": state, "scenes_in_orbit": count,
                                "dates": dates, "size": size, "collection": "sentinel-1-rtc"}, indent=1))
    return sorted(folder.glob("s1_*.tif"))


if __name__ == "__main__":
    files = fetch(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400)
    print(len(files), "dates in", sys.argv[1])
