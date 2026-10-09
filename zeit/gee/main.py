import ee
import os
import warnings
import concurrent.futures
from typing import Optional, Sequence, Tuple, Union
from ..indices import DEFAULT_METRICS, INDICES, NORMALIZED_DIFFERENCES, canonical_index, parse_metrics
from .auth import initialize_gee
from .harmonization import get_harmonized_collection
from .composites import create_annual_medoid
from .downloader import download_gee_image
from .roi import resolve_roi
from .drive_sync import submit_drive_export, wait_and_download_task

# Harmonized SR band holding each band role (see zeit.indices).
_GEE_BANDS = dict(blue="SR_B2", green="SR_B3", red="SR_B4", nir="SR_B5", swir1="SR_B6", swir2="SR_B7")
_SR_BANDS = list(_GEE_BANDS.values())

# Earth Engine reducer output suffix of each temporal metric (zeit.indices.parse_metrics).
_EE_SUFFIX = {"median": "p50", "mean": "mean", "std": "stdDev", "min": "min", "max": "max",
              "count": "count"}


def _compute_indices(img: "ee.Image", indices: list) -> "ee.Image":
    """Select SR bands and indices from a harmonized image.

    Indices are computed on surface reflectance: the Collection 2 Level-2 digital
    numbers are scaled (x 2.75e-05, - 0.2) first. Requested SR bands stay as they come.
    The image properties (system:time_start...) are kept.
    """
    refl = img.select(_SR_BANDS).multiply(2.75e-05).add(-0.2)
    band = lambda role: refl.select(_GEE_BANDS[role])
    out = img.select([])
    for b in indices:
        name = canonical_index(b)
        if name is None:
            out = out.addBands(img.select([b]))
            continue
        if name in NORMALIZED_DIFFERENCES:
            index = refl.normalizedDifference([_GEE_BANDS[r] for r in INDICES[name]])
        elif name == "kNDVI":
            index = refl.normalizedDifference([_GEE_BANDS["nir"], _GEE_BANDS["red"]]).pow(2).tanh()
        elif name == "NDFI":
            from .._sma import SOUZA_2005
            fr = refl.select([_GEE_BANDS[r] for r in INDICES["NDFI"]]).unmix(
                [[v / 10000 for v in SOUZA_2005[k]] for k in ("gv", "shade", "npv", "soil", "cloud")], True, True)
            index = fr.expression("((GV / (1 - SHADE)) - (NPV + SOIL)) / ((GV / (1 - SHADE)) + NPV + SOIL)",
                                  {"GV": fr.select(0), "SHADE": fr.select(1), "NPV": fr.select(2),
                                   "SOIL": fr.select(3)})
        elif name == "EVI":
            index = refl.expression(
                "2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)",
                {"NIR": band("nir"), "RED": band("red"), "BLUE": band("blue")})
        else:  # SAVI
            index = refl.expression(
                "1.5 * (NIR - RED) / (NIR + RED + 0.5)", {"NIR": band("nir"), "RED": band("red")})
        out = out.addBands(index.rename(name).toFloat())
    return out


def _band_names(indices: Optional[list]) -> list:
    """Output names of `_compute_indices`: indices in their registered spelling."""
    return [canonical_index(b) or b for b in (indices or _SR_BANDS)]


def _resolve_indices(indices: Optional[list], bands: Optional[list]) -> Optional[list]:
    """`indices`, or the value of its deprecated alias `bands`."""
    if bands is None:
        return indices
    if indices is not None:
        raise TypeError("Pass indices only: bands is its deprecated name.")
    warnings.warn("`bands` is deprecated, use `indices` (it takes the same SR bands and "
                  "index names).", DeprecationWarning, stacklevel=3)
    return bands


def _stm_reducer_plan(metrics: list) -> Tuple[list, list]:
    """Percentiles and plain reducers (by metric name) one combined reducer needs."""
    pct = {int(m[1:]) for m in metrics if m.startswith("p")}
    if "median" in metrics:
        pct.add(50)
    if "iqr" in metrics:
        pct.update((25, 75))
    simple = [m for m in ("mean", "std", "min", "max", "count") if m in metrics]
    return sorted(pct), simple


def _stm_output_sources(band: str, metric: str) -> list:
    """Reducer output band(s) a metric is read from: one, or (p75, p25) for the IQR."""
    if metric == "iqr":
        return [f"{band}_p75", f"{band}_p25"]
    return [f"{band}_{_EE_SUFFIX.get(metric, metric)}"]


def compute_stm(col: "ee.ImageCollection", indices: Optional[list] = None,
                metrics: Sequence[str] = DEFAULT_METRICS) -> "ee.Image":
    """Spectral temporal metrics of a harmonized collection, computed by Earth Engine.

    indices: Indices and SR bands, as in `download_gee_timeseries` (default: the six SR
        bands).
    metrics: as in `zeit.indices.parse_metrics`.

    Indices are computed on every image first, then each metric is taken over time with
    one combined reducer. Bands are named "<band>_<metric>" (e.g. "NDVI_p10"), ordered
    band by band, as in `zeit.cube.build_spectral_temporal_metrics`.
    """
    metrics = parse_metrics(metrics)
    names = _band_names(indices)
    pct, simple = _stm_reducer_plan(metrics)
    reducers = ([ee.Reducer.percentile(pct)] if pct else []) + [
        {"mean": ee.Reducer.mean, "std": ee.Reducer.stdDev, "min": ee.Reducer.min,
         "max": ee.Reducer.max, "count": ee.Reducer.count}[m]() for m in simple]
    reducer = reducers[0]
    for r in reducers[1:]:
        reducer = reducer.combine(r, sharedInputs=True)

    stats = col.map(lambda img: _compute_indices(img, names)).reduce(reducer)
    out = []
    for b in names:
        for m in metrics:
            src = _stm_output_sources(b, m)
            img = stats.select(src[0])
            if len(src) == 2:
                img = img.subtract(stats.select(src[1]))
            out.append(img.toFloat().rename(f"{b}_{m}"))
    return ee.Image.cat(out)


def download_gee_timeseries_drive(
    roi: Union[str, tuple, list, "ee.Geometry"],
    start_date: str,
    end_date: str,
    out_dir: str,
    tile_label: str,
    indices: Optional[list] = None,
    project: Optional[str] = None,
    drive_folder: str = "zeit_exports",
    max_concurrent_tasks: int = 10,
    delete_after: bool = True,
    poll_interval: int = 15,
    bands: Optional[list] = None,
) -> list:
    """
    Downloads an annual-medoid time series via GEE batch Export.image.toDrive
    instead of the synchronous getDownloadURL tiling in download_gee_timeseries.
    Submits one export task per year up front (so GEE computes them in
    parallel server-side), then polls/downloads finished files with bounded
    local concurrency -- avoiding the per-request compute overhead that makes
    method='direct' slow for a full multi-decade time series.

    Args:
        roi: Region of interest: a Landsat WRS-2 tile id ('217/076'), a
            (min_lon, min_lat, max_lon, max_lat) bbox, or an ee.Geometry.
        start_date, end_date (str): YYYY-MM-DD.
        out_dir (str): Local directory to save the downloaded GeoTIFFs.
        tile_label (str): Identifier (e.g. "214_064") used to prefix filenames
            and the Drive export description, so concurrent tiles don't collide.
        indices (list, optional): Indices (e.g. ["NDVI"]) and SR bands to export, as
            in download_gee_timeseries. If None, keeps the raw harmonized SR_B2-SR_B7
            bands.
        project (str, optional): Google Cloud Project ID.
        drive_folder (str): Google Drive folder name used for every export
            in this batch (created automatically by GEE on first export).
        max_concurrent_tasks (int): How many years to have in flight
            (submitted-but-not-yet-downloaded) at once.
        delete_after (bool): Delete each file from Drive once downloaded
            locally, so a multi-tile batch doesn't fill up Drive quota.
        poll_interval (int): Seconds between task-status checks.
        bands (list, optional): Deprecated name of `indices`.

    Returns:
        list[str]: Local file paths successfully downloaded.
    """
    indices = _resolve_indices(indices, bands)
    initialize_gee(project=project)

    geom = resolve_roi(roi)
    os.makedirs(out_dir, exist_ok=True)

    # Medoid selection runs on the raw harmonized spectral bands (matching
    # LT-GEE's own medoidMosaic: squared distance to the annual per-band
    # median summed across all SR bands); indices are derived only from the
    # single per-year composite pixel that selection picks. Computing an
    # index first would collapse medoid selection to 1D distance-to-median
    # in index space -- a different, non-standard criterion for which
    # candidate scene wins a given year.
    col = get_harmonized_collection(geom, start_date, end_date)

    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])
    years = list(range(start_year, end_year + 1))

    downloaded = []

    def _submit(year):
        medoid_raw = create_annual_medoid(col, year)
        img_medoid = _compute_indices(medoid_raw, indices) if indices else medoid_raw
        prefix = f"{tile_label}_{year}"
        task = submit_drive_export(img_medoid, prefix, drive_folder, geom.bounds())
        return year, prefix, task

    def _wait(year_prefix_task):
        year, prefix, task = year_prefix_task
        out_path = os.path.join(out_dir, f"{prefix}.tif")
        print(f"[{tile_label}] Waiting on year {year} (task {task.id})...")
        result = wait_and_download_task(
            task, drive_folder, prefix, out_path,
            poll_interval=poll_interval, delete_after=delete_after,
        )
        if result:
            print(f"[{tile_label}] Downloaded year {year} -> {result}")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        submitted = [_submit(y) for y in years]
        for result in executor.map(_wait, submitted):
            if result:
                downloaded.append(result)

    print(f"[{tile_label}] Finished: {len(downloaded)}/{len(years)} years downloaded.")
    return downloaded


def download_gee_timeseries(
    roi: Union[str, tuple, list, "ee.Geometry"],
    start_date: str, 
    end_date: str, 
    out_dir: str, 
    method: str = 'auto',
    composite_type: str = 'annual',
    indices: Optional[list] = None,
    project: Optional[str] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    bands: Optional[list] = None,
) -> None:
    """
    Downloads time series data from Google Earth Engine.
    
    Args:
        roi (str, tuple, list, or ee.Geometry): Region of interest: a Landsat
            WRS-2 tile id such as '217/076' (downloads that tile's footprint
            bounding box), a (min_lon, min_lat, max_lon, max_lat) bbox, or an
            ee.Geometry.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        out_dir (str): Output directory to save the files.
        method (str): Download method passed to download_gee_image: 'auto' (default;
            tiled direct download, falling back to Drive export for very large or
            compute-heavy images), 'direct' or 'drive'.
        composite_type (str): 'annual' (one medoid composite per year, for LandTrendr),
            'dense' (every observation in one stack, for CCDC) or 'stm' (spectral
            temporal metrics: per-year statistics of each band or index over every clear
            observation, one file per year).
        indices (list, optional): Indices (NDVI, EVI, SAVI, kNDVI, NBR, NDMI, NDWI,
            MNDWI) and SR bands (SR_B2-SR_B7) to export. Default: the six SR bands.
        project (str, optional): Google Cloud Project ID for authentication.
        metrics (list): With composite_type='stm', the statistics to export: median,
            mean, std, min, max, iqr, count, or a percentile such as 'p10'.
        bands (list, optional): Deprecated name of `indices`.
    """
    if composite_type not in ('annual', 'dense', 'stm'):
        raise NotImplementedError(f"Composite type '{composite_type}' is not currently supported.")
    indices = _resolve_indices(indices, bands)
    if composite_type == 'stm':
        metrics = parse_metrics(metrics)
    initialize_gee(project=project)
    
    geom = resolve_roi(roi)


    os.makedirs(out_dir, exist_ok=True)
    
    print("Preparing harmonized collection...")
    col = get_harmonized_collection(geom, start_date, end_date)

    start_year = int(start_date.split('-')[0])
    end_year = int(end_date.split('-')[0])

    if composite_type == 'annual':
        # Medoid selection runs on the raw harmonized spectral bands (matching
        # LT-GEE's own medoidMosaic: squared distance to the annual per-band
        # median summed across all SR bands); indices are derived only from
        # the single per-year composite pixel that selection picks -- computing
        # an index first would collapse medoid selection to 1D distance in
        # index space, a different (non-standard) criterion for which
        # candidate scene wins a given year.
        print(f"Extracting annual composites from {start_year} to {end_year}...")
        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            medoid_raw = create_annual_medoid(col, year)
            img_medoid = _compute_indices(medoid_raw, indices) if indices else medoid_raw
            filename = os.path.join(out_dir, f"landsat_medoid_{year}.tif")
            download_gee_image(img_medoid, geom, filename, method=method)
    elif composite_type == 'dense':
        # Dense per-observation stack, not a medoid composite -- indices are
        # computed per image before flattening, same as any other band.
        if indices:
            col = col.map(lambda img: _compute_indices(img, indices))
        print("Extracting dense time series dates...")

        # Get dates in milliseconds from GEE
        dates_ms = col.aggregate_array('system:time_start').getInfo()
        
        if not dates_ms:
            print("No images found in the given date range.")
            return
            
        import pandas as pd
        from datetime import datetime
        
        dates_list = []
        ordinal_dates = []
        
        for ms in dates_ms:
            dt = datetime.utcfromtimestamp(ms / 1000.0)
            dates_list.append(dt.strftime('%Y-%m-%d'))
            ordinal_dates.append(dt.toordinal())
            
        # Save to CSV
        csv_path = os.path.join(out_dir, "ccdc_dates.csv")
        df = pd.DataFrame({
            'Date': dates_list,
            'Ordinal_Day': ordinal_dates
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved dates for CCDC to: {csv_path}")
        
        print("Flattening collection for dense stack download...")
        # Convert the ImageCollection to a single multi-band Image
        # The bands will be ordered chronologically, matching the dates array
        dense_image = col.toBands()
        
        filename = os.path.join(out_dir, "landsat_dense_stack.tif")
        download_gee_image(dense_image, geom, filename, method=method)
    else:
        # Spectral temporal metrics: indices on every observation, then statistics
        # over the year, reduced server-side so only the metrics are downloaded.
        print(f"Computing spectral temporal metrics from {start_year} to {end_year}...")
        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            yearly = col.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            img_stm = compute_stm(yearly, indices, metrics)
            filename = os.path.join(out_dir, f"landsat_stm_{year}.tif")
            download_gee_image(img_stm, geom, filename, method=method)
        
    print("Process finished.")
