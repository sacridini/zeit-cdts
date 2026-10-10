import xarray as xr
from pystac_client import Client
import stackstac
import geopandas as gpd

# Known public STAC catalogs
STAC_CATALOGS = {
    "earth_search": "https://earth-search.aws.element84.com/v1",
    "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    "brazil_data_cube": "https://data.inpe.br/bdc/stac/v1/"
}

from typing import List, Optional, Sequence, Tuple, Union
import xarray as xr

import concurrent.futures
import re
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError

from .indices import INDICES, canonical_index, index_array, parse_metrics, temporal_metrics, DEFAULT_METRICS

# GDAL settings for many concurrent COG range reads: read the header in one request,
# and time out and retry stalled or failed requests instead of waiting on them forever.
GDAL_ENV = stackstac.rio_reader.LayeredEnv(
    always=dict(
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_INGESTED_BYTES_AT_OPEN="65536",
        GDAL_HTTP_TIMEOUT="60",
        GDAL_HTTP_CONNECTTIMEOUT="15",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="1",
    ),
    open=dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE=True),
    read=dict(VSI_CACHE=False),
)

# Read errors that turn an asset into nodata instead of failing the whole cube: missing
# files and corrupted blocks. stackstac matches exception *instances* whose message is a
# regex; access errors (missing credentials, requester-pays buckets) still raise.
ERRORS_AS_NODATA = (
    RasterioIOError("HTTP response code: 404"),
    RasterioIOError(".*IReadBlock failed"),
    RasterioIOError(".*not recognized as( being in)? a supported file format"),
)

# Sentinel-2 SCL classes kept as clear: vegetation, bare soil, water, unclassified, snow.
SCL_CLEAR = [4, 5, 6, 7, 11]

# Sentinel-2 L2A reflectance assets (not SCL, AOT, WVP...).
_S2_SPECTRAL = re.compile(r"B(0[1-9]|1[0-2]|8A)")


def _is_planetary_computer(source: str) -> bool:
    return "planetarycomputer" in STAC_CATALOGS.get(source, source)


def _qa_band_for(collection_name: str, source: str = "") -> Optional[str]:
    if "sentinel" in collection_name or "s2" in collection_name:
        # Planetary Computer keeps the Sentinel-2 asset names in capitals (B04, SCL).
        return "SCL" if _is_planetary_computer(source) else "scl"
    if "landsat" in collection_name or "l8" in collection_name:
        return "qa_pixel"
    return None


def _stac_band_map(collection_name: str, source: str = "") -> dict:
    """Asset name of each band role (see `zeit.indices`) for the known collections."""
    if "landsat" in collection_name:
        return dict(blue="blue", green="green", red="red", nir="nir08", swir1="swir16", swir2="swir22")
    if "sentinel-2" in collection_name or "s2" in collection_name:
        if _is_planetary_computer(source):
            return dict(blue="B02", green="B03", red="B04", nir="B08", swir1="B11", swir2="B12")
        return dict(blue="blue", green="green", red="red", nir="nir", swir1="swir16", swir2="swir22")
    return {}


def _clear_mask(qa: np.ndarray, qa_band: str) -> np.ndarray:
    """Boolean clear-sky mask from an integer QA array."""
    if qa_band.lower() == "scl":
        return np.isin(qa, SCL_CLEAR)
    # Landsat QA_PIXEL: bit 6 is set on clear pixels.
    return (qa & (1 << 6)) > 0


def _validate_stac_item(item, band):
    """Helper function to test if a STAC item URL is physically readable."""
    try:
        url = item.assets[band].href
        with rasterio.open(url) as src:
            pass # Just opening is enough to test if header exists and is valid
        return item
    except Exception:
        return None

def build_time_series(
    source: str = "earth_search", 
    collection: Union[str, List[str]] = "sentinel-2-l2a", 
    bbox: Optional[List[float]] = None, 
    vector_path: Optional[str] = None,
    tiles: Optional[List[str]] = None,
    start_date: str = "2020-01-01", 
    end_date: str = "2020-12-31", 
    cloud_cover_max: int = 30,
    bands: Optional[List[str]] = None,
    apply_cloud_mask: bool = False,
    resolution: Optional[float] = None,
    epsg: int = 4326,
    validate_items: bool = False,
    access_token: Optional[str] = None,
    chunksize: int = 2048,
    dtype: str = "float32",
) -> xr.DataArray:
    """
    Builds a lazy Dask-backed xarray DataCube from a STAC catalog.
    
    source: A string from STAC_CATALOGS or a custom STAC API URL.
    collection: The dataset collection ID (e.g., "sentinel-2-l2a", "CBERS4A_WFI_L4_SR").
    bbox: [minx, miny, maxx, maxy] in WGS84 (EPSG:4326).
    vector_path: Path to a shapefile or geojson to derive the bounding box.
    tiles: List of specific MGRS/WRS tiles to fetch (e.g., ["20LKP"]).
    start_date, end_date: YYYY-MM-DD strings.
    cloud_cover_max: Maximum cloud cover percentage for image filtering.
    bands: List of band names to load (e.g., ["red", "green", "blue", "nir"]).
    apply_cloud_mask: Automatically identify platform and mask out clouds (requires QA band).
    resolution: Target spatial resolution in meters (if reprojection is needed).
    validate_items: If True, tests each STAC item's URL before stacking to drop corrupted files.
    access_token: API token for restricted catalogs like Brazil Data Cube (BDC).
    chunksize: Spatial chunk size in pixels. Large chunks (a multiple of the 256 px COG
        blocks) mean fewer requests and fewer blocks re-read at chunk edges.
    dtype: Output data type. Float types (default "float32") hold reflectance (scale and
        offset applied) with NaN as nodata. Integer types (e.g. "uint16") hold the raw
        digital numbers with 0 as nodata, at half the memory, plus per-scene `scale` and
        `offset` coordinates (time, band) to recover reflectance; the cloud mask can't be
        applied in place then, so `apply_cloud_mask` must stay False (load the QA band
        through `bands` instead).
    """
    integer_output = np.issubdtype(np.dtype(dtype), np.integer)
    if integer_output and apply_cloud_mask:
        raise ValueError(
            "apply_cloud_mask needs a float dtype (masked pixels become NaN). "
            "Use dtype='float32', or load the QA band through `bands` and mask it yourself."
        )
    
    # 1. Resolve Spatial Boundary
    if vector_path is not None:
        gdf = gpd.read_file(vector_path).to_crs("EPSG:4326")
        bounds = gdf.total_bounds
        bbox = [bounds[0], bounds[1], bounds[2], bounds[3]]
    
    if bbox is None and tiles is None:
        raise ValueError("Must provide either bbox, vector_path, or tiles")
        
    # 2. Connect to STAC API
    stac_url = STAC_CATALOGS.get(source, source)
    catalog = Client.open(stac_url)
    
    # 3. Search for items
    query_params = {}
    if cloud_cover_max < 100:
        query_params["eo:cloud_cover"] = {"lt": cloud_cover_max}
        
    if tiles:
        # Determine the correct STAC property based on collection name heuristically
        col_name = collection[0].lower() if isinstance(collection, list) else collection.lower()
        if "sentinel" in col_name or "s2" in col_name:
            query_params["s2:mgrs_tile"] = {"in": tiles}
        elif "landsat" in col_name:
            # For Landsat, if it's a single tile (e.g., "215065"), inject into query
            if len(tiles) == 1 and len(tiles[0]) == 6 and tiles[0].isdigit():
                query_params["landsat:wrs_path"] = {"eq": tiles[0][:3]}
                query_params["landsat:wrs_row"]  = {"eq": tiles[0][3:]}
            
    search_kwargs = {
        "collections": [collection] if isinstance(collection, str) else collection,
        "datetime": f"{start_date}/{end_date}",
        "query": query_params
    }
    
    if bbox is not None:
        search_kwargs["bbox"] = bbox
        
    search = catalog.search(**search_kwargs)
    items = search.item_collection()
    
    # 3.5 Microsoft Planetary Computer SAS Token Signing
    if "planetarycomputer" in stac_url:
        try:
            import planetary_computer
            print("Signing items with Planetary Computer SAS tokens...")
            items = [planetary_computer.sign(item) for item in items]
        except ImportError:
            raise ImportError("Please install 'planetary-computer' via pip to use Microsoft Planetary Computer.")
            
    print(f"Found {len(items)} scenes in {source} for {collection}")
    
    if len(items) == 0:
        raise ValueError("No images found for the given criteria.")
        
    items_list = list(items)

    # Manual post-filtering for Landsat tiles if multiple were provided
    # since STAC query doesn't easily support OR conditions across multiple path/row pairs
    if tiles and "landsat" in col_name:
        filtered_items = []
        for item in items_list:
            path = item.properties.get("landsat:wrs_path", "")
            row = item.properties.get("landsat:wrs_row", "")
            # Some catalogs store them as integers or unpadded strings
            pr = f"{int(path):03d}{int(row):03d}" if path and row else ""
            if pr in tiles:
                filtered_items.append(item)
        
        if filtered_items or len(tiles) > 1: # if we found matches or we were explicitly filtering
            items_list = filtered_items
            print(f"Filtered to {len(items_list)} items matching Landsat tiles: {tiles}")
    
    # 3.7 BDC Token injection
    if access_token:
        print("Injecting access token into asset URLs...")
        for item in items_list:
            for asset_key in item.assets:
                asset = item.assets[asset_key]
                if "?" in asset.href:
                    asset.href = f"{asset.href}&access_token={access_token}"
                else:
                    asset.href = f"{asset.href}?access_token={access_token}"
    
    # 3.8 Add QA band if apply_cloud_mask is requested but not in bands
    col_name = collection[0].lower() if isinstance(collection, list) else collection.lower()
    qa_band = None
    if apply_cloud_mask:
        qa_band = _qa_band_for(col_name, source)
        if qa_band and bands and qa_band not in bands:
            bands = list(bands) + [qa_band]   # not the caller's list
            
    # 3.6 Pre-flight Validation
    if validate_items and bands:
        print(f"Iniciando validação de {len(items_list)} cenas para bloquear arquivos corrompidos...")
        valid_items = []
        band_to_check = bands[0]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(_validate_stac_item, item, band_to_check): item for item in items_list}
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res is not None:
                    valid_items.append(res)
                    
        removed = len(items_list) - len(valid_items)
        print(f"Cenas íntegras aprovadas: {len(valid_items)} (Removidas/Corrompidas: {removed})")
        items_list = valid_items
        
        if len(items_list) == 0:
            raise ValueError("All scenes were invalid or corrupted after validation.")
        
    # 4. Build DataCube via stackstac
    stack_kwargs = {
        "assets": bands,
        "resolution": resolution,
        "epsg": epsg,
        "chunksize": chunksize,
        "dtype": np.dtype(dtype),
        "gdal_env": GDAL_ENV,
        "errors_as_nodata": ERRORS_AS_NODATA,
    }
    if integer_output:
        stack_kwargs.update(fill_value=np.dtype(dtype).type(0), rescale=False)
    else:
        # stackstac refuses to rescale into float32 (a scale like 2.75e-05 isn't "safely"
        # castable), so the scale and offset are applied below instead. The NaN fill must
        # also be of the output type to pass its casting check.
        stack_kwargs.update(rescale=False, fill_value=np.dtype(dtype).type(np.nan))
    if bbox is not None:
        stack_kwargs["bounds_latlon"] = bbox
        
    cube = stackstac.stack(items_list, **stack_kwargs)
    scale, offset = _item_scale_offset(cube, items_list)
    if integer_output:
        # Raw digital numbers: carry the per-scene factors so reflectance can be recovered.
        cube = cube.assign_coords(scale=(("time", "band"), scale), offset=(("time", "band"), offset))
    else:
        cube = _rescale(cube, scale, offset)

    # 5. Apply Cloud Mask semantics natively
    if apply_cloud_mask and qa_band and qa_band in cube.band.values:
        qa = cube.sel(band=qa_band)
        
        # Nodata QA pixels are NaN in a float cube: map them to 0 (not clear) before the
        # integer cast, which is undefined for NaN.
        # The band name as a keyword: positional, dask would pass it as a 0-d array.
        valid_mask = xr.apply_ufunc(_clear_mask, qa.fillna(0).astype("uint16"), kwargs={"qa_band": qa_band},
                                    dask="parallelized", output_dtypes=[bool])
        cube = cube.where(valid_mask)
            
        print(f"Applied semantic cloud mask using QA band '{qa_band}'")

    return cube


def _nanmedian_time(a: np.ndarray, n_threads: int = 4) -> np.ndarray:
    """NaN-aware median along axis 0 of a ``(time, y, x)`` float32 array.

    ``np.sort`` puts NaNs last, so the median is the middle of the ``n`` valid values.
    Rows are sorted in strips on a few threads (``np.sort`` releases the GIL). Matches
    ``np.nanmedian(a, axis=0)``; all-NaN pixels stay NaN.
    """
    t, ny, nx = a.shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    if t == 0:
        return out

    def strip(r0):
        r1 = min(ny, r0 + 64)
        s = np.sort(a[:, r0:r1], axis=0)
        n = np.isfinite(s).sum(axis=0)
        lo = np.clip((n - 1) // 2, 0, t - 1)[None]
        hi = np.clip(n // 2, 0, t - 1)[None]
        m = 0.5 * (np.take_along_axis(s, lo, 0)[0] + np.take_along_axis(s, hi, 0)[0])
        m[n == 0] = np.nan
        out[r0:r1] = m

    with concurrent.futures.ThreadPoolExecutor(n_threads) as ex:
        list(ex.map(strip, range(0, ny, 64)))
    return out


def _masked_reflectance(raw, i, clear, scale, offset):
    """Band ``i`` of a raw ``(time, band, y, x)`` block as float32 reflectance.

    NaN where the QA mask says cloud or the band is nodata (0). ``scale`` and ``offset``
    are this band's ``(time,)`` per-scene factors.
    """
    v = raw[:, i]
    out = np.where(clear & (v != 0), v.astype(np.float32), np.float32(np.nan))
    out *= scale[:, None, None]
    out += offset[:, None, None]
    return out


def _composite_block(raw, band_idx, qa_idx, qa_band, scale, offset, method):
    """Reduce one spatial chunk of raw digital numbers to a composite.

    raw: ``(time, band, y, x)`` integer array, 0 = nodata.
    scale, offset: ``(time, len(band_idx))`` per-scene factors (they differ between
        scenes when a provider changes processing baseline).
    Returns ``(len(band_idx), y, x)`` float32 reflectance.
    """
    clear = _clear_mask(raw[:, qa_idx], qa_band) if qa_idx is not None else True
    vals = np.empty((raw.shape[0], len(band_idx)) + raw.shape[2:], dtype=np.float32)
    for k, i in enumerate(band_idx):
        vals[:, k] = _masked_reflectance(raw, i, clear, scale[:, k], offset[:, k])

    med = np.stack([_nanmedian_time(vals[:, k]) for k in range(len(band_idx))])
    if method == "median":
        return med
    # Medoid: the real observation closest (squared distance over bands) to the median.
    dist = np.nansum((vals - med[None]) ** 2, axis=1)
    dist[~np.isfinite(vals).any(axis=1)] = np.inf
    best = np.argmin(dist, axis=0)[None, None]
    medoid = np.take_along_axis(vals, np.broadcast_to(best, (1,) + vals.shape[1:]), 0)[0]
    medoid[:, ~np.isfinite(med).any(axis=0)] = np.nan
    return medoid


def _item_scale_offset(cube: xr.DataArray, items) -> Tuple[np.ndarray, np.ndarray]:
    """Per-scene, per-band scale and offset from each item's ``raster:bands`` metadata.

    Read from the items rather than the cube's coordinates: stackstac drops metadata that
    doesn't line up across bands (Earth Search Sentinel-2, for one). Sentinel-2 items
    without ``raster:bands`` (Planetary Computer) get the L2A factors from their
    processing baseline: reflectance = DN / 10000, minus 0.1 from baseline 04.00 on.
    """
    bands = list(cube.band.values)
    scale = np.ones((cube.sizes["time"], len(bands)), dtype=np.float32)
    offset = np.zeros_like(scale)
    by_id = {item.id: item for item in items}
    for i, item_id in enumerate(cube.coords["id"].values):
        item = by_id.get(item_id)
        if item is None:
            continue
        baseline = (getattr(item, "properties", None) or {}).get("s2:processing_baseline")
        for k, band in enumerate(bands):
            asset = item.assets.get(band)
            meta = asset.extra_fields.get("raster:bands") if asset is not None else None
            if isinstance(meta, list) and meta and isinstance(meta[0], dict):
                scale[i, k] = meta[0].get("scale", 1.0)
                offset[i, k] = meta[0].get("offset", 0.0)
            elif baseline is not None and _S2_SPECTRAL.fullmatch(str(band)):
                scale[i, k] = 1e-4
                offset[i, k] = -0.1 if float(baseline) >= 4.0 else 0.0
    return scale, offset



def _rescale(cube: xr.DataArray, scale: np.ndarray, offset: np.ndarray) -> xr.DataArray:
    """Apply per-scene ``(time, band)`` scale and offset to a float cube, keeping its dtype."""
    if np.all(scale == 1) and np.all(offset == 0):
        return cube
    dt = cube.dtype
    # Positional (no coords): stackstac time coordinates can hold duplicate timestamps.
    out = cube * xr.DataArray(scale.astype(dt), dims=("time", "band"))         + xr.DataArray(offset.astype(dt), dims=("time", "band"))
    out.attrs = cube.attrs
    return out


def _cube_scale_offset(cube: xr.DataArray, bands: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """The ``scale``/``offset`` coordinates of a raw cube from `build_time_series`."""
    if "scale" not in cube.coords:
        n = (cube.sizes["time"], len(bands))
        return np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    pick = lambda c: cube.coords[c].sel(band=bands).transpose("time", "band").values.astype(np.float32)
    return pick("scale"), pick("offset")


def _composite_cube(cube, bands, qa_band, method):
    """Lazy composite of a raw integer cube: one dask task per spatial chunk."""
    import dask.array as da

    names = list(cube.band.values)
    band_idx = [names.index(b) for b in bands]
    qa_idx = names.index(qa_band) if qa_band else None
    scale, offset = _cube_scale_offset(cube, bands)
    # All scenes and bands of a spatial chunk in one block, so every byte is read once.
    raw = cube.data.rechunk({0: -1, 1: -1})
    return da.map_blocks(
        _composite_block, raw, band_idx, qa_idx, qa_band, scale, offset, method,
        drop_axis=0, dtype=np.float32, chunks=((len(bands),),) + raw.chunks[2:],
    )


def _stm_block(raw, layers, band_pos, qa_idx, qa_band, scale, offset, metrics):
    """Spectral temporal metrics of one spatial chunk of raw digital numbers.

    raw: ``(time, band, y, x)`` integer array, 0 = nodata.
    layers: ``(name, source)`` pairs: an index with the asset of each band role it needs
        (``source`` a dict), or a plain band (``source`` its asset name).
    band_pos: asset name -> ``(position in raw, column in scale/offset)``.
    Returns ``(len(layers) * len(metrics), y, x)`` float32, layer-major.
    """
    clear = _clear_mask(raw[:, qa_idx], qa_band) if qa_idx is not None else True
    cache = {}

    def band(asset):
        if asset not in cache:
            i, k = band_pos[asset]
            cache[asset] = _masked_reflectance(raw, i, clear, scale[:, k], offset[:, k])
        return cache[asset]

    out = []
    for name, source in layers:
        if isinstance(source, dict):
            values = index_array(name, lambda role: band(source[role]))
        else:
            values = band(source)
        out.append(temporal_metrics(values, metrics))
    return np.concatenate(out)


def _stm_cube(cube, layers, assets, qa_band, metrics):
    """Lazy spectral temporal metrics of a raw integer cube: one dask task per chunk."""
    import dask.array as da

    names = list(cube.band.values)
    band_pos = {a: (names.index(a), k) for k, a in enumerate(assets)}
    qa_idx = names.index(qa_band) if qa_band else None
    scale, offset = _cube_scale_offset(cube, assets)
    # All scenes and bands of a spatial chunk in one block, so every byte is read once.
    raw = cube.data.rechunk({0: -1, 1: -1})
    n_out = len(layers) * len(metrics)
    return da.map_blocks(
        _stm_block, raw, layers, band_pos, qa_idx, qa_band, scale, offset, metrics,
        drop_axis=0, dtype=np.float32, chunks=((n_out,),) + raw.chunks[2:],
    )


def _annual_stack(reduce, out_bands, name, *, source, collection, bbox, vector_path,
                  start_year, end_year, season, load, cloud_cover_max, resolution, epsg,
                  access_token, chunksize):
    """Stack ``reduce(raw_cube) -> (len(out_bands), y, x)`` over each year's season window.

    Each year is read as raw integers (`build_time_series` with ``dtype="uint16"``) on the
    grid every year shares, and the results stacked to ``(time, band, y, x)``. Years
    without scenes are NaN, so the time axis has no gaps.
    """
    import dask.array as da

    if bbox is None and vector_path is None:
        raise ValueError("Provide bbox or vector_path so every year shares the same grid.")
    if vector_path is not None:
        bbox = list(gpd.read_file(vector_path).to_crs("EPSG:4326").total_bounds)
    wraps = season[1] < season[0]

    years = list(range(start_year, end_year + 1))
    results = {}
    template = None
    for year in years:
        try:
            cube = build_time_series(
                source=source, collection=collection, bbox=bbox,
                start_date=f"{year}-{season[0]}",
                end_date=f"{year + 1 if wraps else year}-{season[1]}",
                cloud_cover_max=cloud_cover_max, bands=list(load),
                apply_cloud_mask=False, resolution=resolution, epsg=epsg,
                access_token=access_token, chunksize=chunksize, dtype="uint16",
            )
        except ValueError as e:
            if "No images found" not in str(e):
                raise
            print(f"{year}: no scenes, filled with NaN")
            continue
        results[year] = reduce(cube)
        if template is None:
            template = cube

    if template is None:
        raise ValueError("No images found for any year in the given criteria.")

    first = next(iter(results.values()))
    empty = da.full(first.shape, np.nan, dtype=np.float32, chunks=first.chunks)
    data = da.stack([results.get(y, empty) for y in years])

    out = xr.DataArray(
        data,
        dims=("time", "band", "y", "x"),
        coords={
            "time": pd.to_datetime([f"{y}-01-01" for y in years]),
            "year": ("time", years),
            "band": list(out_bands),
            "x": template.x.values,
            "y": template.y.values,
        },
        attrs={k: v for k, v in template.attrs.items() if k in ("crs", "transform", "resolution")},
        name=name,
    )
    if "epsg" in template.coords:
        out = out.assign_coords(epsg=template.coords["epsg"])
    return out


def build_annual_composites(
    source: str = "planetary_computer",
    collection: str = "landsat-c2-l2",
    bbox: Optional[List[float]] = None,
    vector_path: Optional[str] = None,
    start_year: int = 1985,
    end_year: int = 2024,
    season: Tuple[str, str] = ("06-01", "09-30"),
    bands: Optional[List[str]] = None,
    cloud_cover_max: int = 30,
    method: str = "median",
    apply_cloud_mask: bool = True,
    resolution: Optional[float] = 30,
    epsg: int = 4326,
    access_token: Optional[str] = None,
    chunksize: int = 2048,
) -> xr.DataArray:
    """
    Builds one cloud-masked composite per year from a STAC catalog: the annual stack
    LandTrendr (and other annual trajectory methods) expects.

    Scenes are read as raw integers and reduced chunk by chunk (mask, per-scene scale and
    offset, then median or medoid), so memory stays bounded by a few chunks however many
    scenes a year has, and every pixel is downloaded once.

    source, collection, bbox, vector_path, cloud_cover_max, resolution, epsg,
    access_token, chunksize: as in `build_time_series`. An area (bbox or vector_path) is
        required so that every year shares the same grid.
    start_year, end_year: Inclusive range of years.
    season: ("MM-DD", "MM-DD") window inside each year. A window that wraps the new year
        (e.g. ("12-01", "02-28")) is labelled with the year it starts in.
    bands: Bands to composite, e.g. ["nir08", "swir22"] for NBR.
    method: "median" (per band) or "medoid" (the real observation closest to the
        multi-band median, keeping bands consistent).
    apply_cloud_mask: Mask clouds and shadows with the collection's QA band.

    Returns a lazy float32 DataArray (time, band, y, x) of reflectance, with ``time`` on
    January 1st of each year and a ``year`` coordinate. Years without scenes are NaN.
    """
    if method not in ("median", "medoid"):
        raise ValueError(f"Unknown method {method!r}. Use 'median' or 'medoid'.")
    if bbox is None and vector_path is None:
        raise ValueError("Provide bbox or vector_path so every year shares the same grid.")
    if not bands:
        raise ValueError("Provide the bands to composite, e.g. ['nir08', 'swir22'].")

    qa_band = _qa_band_for(collection.lower(), source) if apply_cloud_mask else None
    load = list(bands) + ([qa_band] if qa_band and qa_band not in bands else [])
    return _annual_stack(
        lambda cube: _composite_cube(cube, bands, qa_band, method),
        bands, f"{method}_composite",
        source=source, collection=collection, bbox=bbox, vector_path=vector_path,
        start_year=start_year, end_year=end_year, season=season, load=load,
        cloud_cover_max=cloud_cover_max, resolution=resolution, epsg=epsg,
        access_token=access_token, chunksize=chunksize,
    )


def build_spectral_temporal_metrics(
    source: str = "planetary_computer",
    collection: str = "landsat-c2-l2",
    bbox: Optional[List[float]] = None,
    vector_path: Optional[str] = None,
    start_year: int = 1985,
    end_year: int = 2024,
    season: Tuple[str, str] = ("01-01", "12-31"),
    indices: Optional[List[str]] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    band_map: Optional[dict] = None,
    cloud_cover_max: int = 30,
    apply_cloud_mask: bool = True,
    resolution: Optional[float] = 30,
    epsg: int = 4326,
    access_token: Optional[str] = None,
    chunksize: int = 1024,
) -> xr.DataArray:
    """
    Builds per-year spectral temporal metrics (STMs) from a STAC catalog: statistics such
    as the median and percentiles of each index over every clear observation of a year.

    Each index is computed on every observation first and the statistics taken over time
    afterwards (the median NDVI, not the NDVI of the median bands). As in
    `build_annual_composites`, scenes are read as raw integers and reduced chunk by chunk:
    the raw bands pass through memory once and only the metrics are kept.

    source, collection, bbox, vector_path, start_year, end_year, cloud_cover_max,
    resolution, epsg, access_token: as in `build_annual_composites`.
    season: ("MM-DD", "MM-DD") window inside each year (default: the whole year).
    indices: What to summarize: index names (NDVI, EVI, SAVI, kNDVI, NBR, NDMI, NDWI,
        MNDWI), band roles (blue, green, red, nir, swir1, swir2) or asset names.
    metrics: median, mean, std, min, max, iqr, count, or a percentile such as "p10".
    band_map: Asset name of each band role, e.g. {"nir": "B8A"}. Known for Landsat
        Collection 2 and Sentinel-2 L2A on Planetary Computer and Earth Search.
    apply_cloud_mask: Mask clouds and shadows with the collection's QA band.
    chunksize: Spatial chunk size in pixels. Memory per chunk in flight is about
        scenes x chunksize^2 x (2 bytes per band loaded + 4 per band or index used).

    Returns a lazy float32 DataArray (time, band, y, x), one band per index and metric
    named "<index>_<metric>" (e.g. "NDVI_p10"), with ``time`` on January 1st of each year
    and a ``year`` coordinate. Years without scenes are NaN.
    """
    if not indices:
        raise ValueError("Provide the indices to summarize, e.g. ['NDVI', 'NBR'].")
    metrics = parse_metrics(metrics)
    col = collection.lower()
    roles = {**_stac_band_map(col, source), **(band_map or {})}

    layers, assets = [], []
    for item in indices:
        name = canonical_index(item)
        if name is not None:
            missing = [r for r in INDICES[name] if r not in roles]
            if missing:
                raise ValueError(
                    f"{name} needs the {missing[0]} band, which has no default asset name "
                    f"for {collection!r}; pass band_map={{'{missing[0]}': '<asset name>'}}."
                )
            used = {r: roles[r] for r in INDICES[name]}
            layers.append((name, used))
            assets += list(used.values())
        else:
            asset = roles.get(item, item)
            layers.append((item, asset))
            assets.append(asset)
    assets = list(dict.fromkeys(assets))

    qa_band = _qa_band_for(col, source) if apply_cloud_mask else None
    load = assets + ([qa_band] if qa_band and qa_band not in assets else [])
    out_bands = [f"{name}_{m}" for name, _ in layers for m in metrics]
    return _annual_stack(
        lambda cube: _stm_cube(cube, layers, assets, qa_band, metrics),
        out_bands, "spectral_temporal_metrics",
        source=source, collection=collection, bbox=bbox, vector_path=vector_path,
        start_year=start_year, end_year=end_year, season=season, load=load,
        cloud_cover_max=cloud_cover_max, resolution=resolution, epsg=epsg,
        access_token=access_token, chunksize=chunksize,
    )
