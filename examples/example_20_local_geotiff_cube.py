"""
Example 20: Local GeoTIFF Ingestion End-to-End

Writes a small archive of individual per-date, per-band GeoTIFF files to
disk (no network needed - this is what a local Landsat/Sentinel-2 ARD
archive typically looks like), lazily reassembles them into a single
Dask-backed datacube with `zeit.load_raster` (Zeit's "load images from
disk" entry point, the local counterpart to `zeit.build_time_series`'s STAC
fetch), computes NDVI from the reassembled
Red/NIR bands, and saves the result with `zeit.save_raster`.
"""
import os
import shutil
import numpy as np
from rasterio.transform import from_origin
import zeit

ARCHIVE_DIR = os.path.join("data", "local_ard_archive")


def write_synthetic_archive(rows=25, cols=25, dates=("20220115", "20220131", "20220216"), seed=15):
    """
    Writes one single-band GeoTIFF per (date, band) combination, named
    `<date>_<band>.tif` - a common local-archive naming convention. Includes
    a mild greening trend across dates so NDVI computed after re-ingestion
    has something meaningful to show.
    """
    rng = np.random.RandomState(seed)
    if os.path.exists(ARCHIVE_DIR):
        shutil.rmtree(ARCHIVE_DIR)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    transform = from_origin(500000.0, 8800000.0, 30.0, 30.0)
    written = []
    for i, date in enumerate(dates):
        red = 0.15 - 0.01 * i + rng.normal(0, 0.01, (rows, cols))
        nir = 0.35 + 0.03 * i + rng.normal(0, 0.02, (rows, cols))
        for band_name, arr in (("B04", red), ("B08", nir)):
            path = os.path.join(ARCHIVE_DIR, f"{date}_{band_name}.tif")
            zeit.save_raster(arr.astype("float32"), path, crs="EPSG:32721", transform=transform, nodata=np.nan)
            written.append(path)
    return written


def main():
    print("Zeit Example 20: Local GeoTIFF Ingestion (zeit.load_raster)")

    print("\n[1/3] Writing a synthetic local ARD archive (per-date, per-band GeoTIFFs) to disk...")
    written = write_synthetic_archive()
    print(f"    Wrote {len(written)} files to {ARCHIVE_DIR}/, e.g. {os.path.basename(written[0])}")

    print("\n[2/3] Lazily reassembling the archive into a Dask-backed datacube with zeit.load_raster()...")
    cube = zeit.load_raster(
        ARCHIVE_DIR, pattern=r"(?P<date>\d{8})_(?P<band>[A-Z0-9]+)\.tif", date_format="%Y%m%d", chunks="auto",
    )
    print(f"    Reassembled cube: dims={cube.dims}, shape={cube.shape}, "
          f"bands={cube.band.values.tolist()}, dates={[str(d)[:10] for d in cube.time.values]}")

    print("\n[3/3] Computing NDVI from the reassembled Red/NIR bands and saving with zeit.save_raster()...")
    red = cube.sel(band="B04")
    nir = cube.sel(band="B08")
    ndvi = ((nir - red) / (nir + red)).compute()
    print(f"    Mean NDVI per date: {[f'{v:.3f}' for v in ndvi.mean(dim=['y', 'x']).values]}")

    out_tif = os.path.join("data", "local_cube_ndvi.tif")
    zeit.save_raster(ndvi.astype("float32"), out_tif)   # the dates and georeferencing travel with it

    print(f"\nDone! NDVI time series ({ndvi.sizes['time']} bands) saved to {out_tif}")


if __name__ == "__main__":
    main()
