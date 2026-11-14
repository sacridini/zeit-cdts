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
import json
import math
import re
import threading
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
try:  # stackstac internals, for the read-error warning only: without them the cube is read as before
    from stackstac.nodata_reader import NodataReader, exception_matches
    from stackstac.rio_reader import AutoParallelRioReader
except ImportError:  # pragma: no cover - another stackstac layout
    AutoParallelRioReader = None

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


class ReadErrorWarning(UserWarning):
    """An asset (or a block of it) could not be read and became nodata (see `ERRORS_AS_NODATA`)."""


#: Files that had a read error turned into nodata in this process: ``{url without its token: message}``.
READ_ERRORS: dict = {}
_READ_ERRORS_LOCK = threading.Lock()


def _report_read_error(url: str, what: str) -> None:
    """Warn, once per file, that ``url`` became nodata, naming the file (its name carries the scene and the band)."""
    key = str(url).split("?")[0]
    with _READ_ERRORS_LOCK:
        first = key not in READ_ERRORS
        READ_ERRORS[key] = what
    if first:
        name = key.rsplit("/", 1)[-1]
        warnings.warn(ReadErrorWarning(
            f"{name}: {what}; read as nodata (scene and band are in the file name; the cube has gaps there). "
            f"URL: {key}"), stacklevel=2)


class _WatchedDataset:
    """A stackstac dataset whose tolerated read errors (`ERRORS_AS_NODATA`) are reported before they become a
    fully masked block."""

    def __init__(self, ds, url, errors):
        self._ds, self._url, self._errors = ds, url, errors

    def read(self, window, **kwargs):
        try:
            return self._ds.read(window=window, **kwargs)
        except Exception as e:
            if not exception_matches(e, self._errors):
                raise
            _report_read_error(self._url, f"a block could not be read ({str(e).strip()[:200]})")
            return np.ma.masked_all((1, int(window.height), int(window.width)), dtype=kwargs["out_dtype"])

    def close(self):
        self._ds.close()


if AutoParallelRioReader is not None:
    class _ZeitReader(AutoParallelRioReader):
        """stackstac's reader, reporting (`ReadErrorWarning`, once per file) what `ERRORS_AS_NODATA` turns into
        nodata.

        stackstac has no callback for it, and its own plain warning does not say which scene it was; the error is
        caught one level down, in the dataset, and answered with the same fully masked block stackstac would
        give."""

        def _open(self):
            ds = super()._open()
            if isinstance(ds, NodataReader):   # the open failed and stackstac already turned it into nodata
                _report_read_error(self.url, "could not be opened")
                return ds
            return _WatchedDataset(ds, self.url, self.errors_as_nodata)
else:  # pragma: no cover
    _ZeitReader = None

# Landsat QA_PIXEL bits that must be 0 on a clear pixel: fill, dilated cloud, cirrus, cloud, cloud shadow.
LANDSAT_BAD_BITS = 0b11111

# Sentinel-2 SCL classes kept as clear: vegetation, bare soil, water, unclassified, snow.
SCL_CLEAR = [4, 5, 6, 7, 11]

# Sentinel-2 L2A reflectance assets (not SCL, AOT, WVP...).
_S2_SPECTRAL = re.compile(r"B(0[1-9]|1[0-2]|8A)")


def _is_planetary_computer(source: str) -> bool:
    return "planetarycomputer" in STAC_CATALOGS.get(source, source)


# HLS v2.0 (Harmonized Landsat Sentinel-2; User Guide, Tables 3, 6-9): the asset of each band
# role in its Landsat (L30) and Sentinel-2 (S30) products. The S30 NIR is the narrow B8A, the
# one adjusted to OLI's band 5; L30's B10/B11 are thermal, S30's B11/B12 are SWIR.
HLS_ROLES = {
    "l30": dict(coastal="B01", blue="B02", green="B03", red="B04", nir="B05", swir1="B06", swir2="B07"),
    "s30": dict(coastal="B01", blue="B02", green="B03", red="B04", nir="B8A", swir1="B11", swir2="B12"),
}
HLS_ANGLES = ("SZA", "SAA", "VZA", "VAA")   # degrees x 100, fill 40000


def _hls_kind(collection_name: str) -> Optional[str]:
    """``"l30"`` or ``"s30"`` for an HLS collection (``hls2-l30``, ``HLSS30.v2.0``...), else None."""
    name = collection_name.lower()
    if "hls" not in name:
        return None
    return "l30" if "l30" in name else "s30" if "s30" in name else None


def _qa_band_for(collection_name: str, source: str = "") -> Optional[str]:
    if _hls_kind(collection_name):   # before the Sentinel-2 test: "hls2-s30" contains "s2"
        return "Fmask"
    if "sentinel" in collection_name or "s2" in collection_name:
        # Planetary Computer keeps the Sentinel-2 asset names in capitals (B04, SCL).
        return "SCL" if _is_planetary_computer(source) else "scl"
    if "landsat" in collection_name or "l8" in collection_name:
        return "qa_pixel"
    return None


def _stac_band_map(collection_name: str, source: str = "") -> dict:
    """Asset name of each band role (see `zeit.indices`) for the known collections."""
    kind = _hls_kind(collection_name)
    if kind:
        return {k: v for k, v in HLS_ROLES[kind].items() if k != "coastal"}
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
    if qa_band.lower() == "fmask":
        # HLS Fmask: bits 1 cloud, 2 adjacent to cloud or shadow, 3 cloud shadow; 255 is fill.
        return ((qa & 0b1110) == 0) & (qa != 255)
    # Landsat QA_PIXEL (Collection 2, Landsat 4-9): bit 0 fill, 1 dilated cloud, 2 cirrus (8-9 only), 3 cloud,
    # 4 cloud shadow, 5 snow, 6 clear, 7 water. The clear bit only says that the cloud and dilated cloud bits are
    # 0, so a cloud shadow, a cirrus and a fill pixel can have it set: they are excluded one by one. Snow and
    # water are surfaces, not defects (as in Sentinel-2's SCL and HLS's Fmask above), and stay clear.
    return ((qa & LANDSAT_BAD_BITS) == 0) & ((qa & (1 << 6)) > 0)


def _validate_stac_item(item, band):
    """Helper function to test if a STAC item URL is physically readable."""
    try:
        url = item.assets[band].href
        with rasterio.open(url) as src:
            pass # Just opening is enough to test if header exists and is valid
        return item
    except Exception:
        return None

#: The day Landsat 7's scan line corrector failed: its scenes from then on have gaps (stripes) in every band.
SLC_FAILURE = pd.Timestamp("2003-05-31")


def _norm_platform(p) -> str:
    """A platform name as one string: ``"landsat-8"`` for ``"LANDSAT_8"``, ``"Landsat-8"``..."""
    return str(p).strip().lower().replace("_", "-")


def _item_props(item) -> dict:
    return getattr(item, "properties", None) or {}


def _item_time(item) -> Optional[pd.Timestamp]:
    """The acquisition time of a STAC item, UTC and without time zone (None if it has none)."""
    t = getattr(item, "datetime", None) or _item_props(item).get("datetime")
    if t is None:
        return None
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def _filter_platforms(items, platforms: Optional[Sequence[str]], slc_off: bool):
    """The items of the given platforms (``["landsat-8", "landsat-9"]``; None keeps every platform) and, unless
    ``slc_off``, without Landsat 7 scenes acquired from 2003-05-31 on (its scan line corrector failed).

    The platform is the item's ``platform`` property, as Planetary Computer and Earth Search spell it
    (``landsat-5``, ``landsat-7``, ``landsat-8``, ``landsat-9``); the case and ``_`` / ``-`` do not matter.
    An item without a platform is dropped when ``platforms`` is given."""
    wanted = None if platforms is None else {_norm_platform(p) for p in ([platforms] if isinstance(platforms, str)
                                                                         else platforms)}
    out = []
    for item in items:
        p = _norm_platform(_item_props(item).get("platform", ""))
        if wanted is not None and p not in wanted:
            continue
        if not slc_off and p == "landsat-7":
            t = _item_time(item)
            if t is not None and t >= SLC_FAILURE:
                continue
        out.append(item)
    return out


def _item_native(item) -> Optional[Tuple[int, Tuple[float, ...]]]:
    """``(epsg, proj:transform)`` of a STAC item, from its properties (or its first asset that has them)."""
    props = _item_props(item)
    sources = [props] + [getattr(a, "extra_fields", None) or {} for a in (getattr(item, "assets", None) or {}).values()]
    for s in sources:
        transform = s.get("proj:transform")
        code = s.get("proj:epsg") or s.get("proj:code")
        if transform is None or code is None:
            continue
        try:
            epsg = int(str(code).split(":")[-1])
        except ValueError:
            continue
        return epsg, tuple(float(v) for v in transform)
    return None


def _native_lattice(items, epsg: int, resolution: float) -> Optional[Tuple[float, float]]:
    """The offsets ``(x, y)``, in ``[0, resolution)``, of the pixel edges of the items' own grid expressed in
    ``epsg``, or None when the items are not on a grid of ``resolution`` in (a CRS equivalent to) ``epsg``.

    Equivalent CRSs: the same one, or the two hemispheres of one UTM zone (326zz and 327zz differ by a false
    northing of 10,000,000 m, so a scene in one is a pure shift of the other: Planetary Computer keeps the
    Southern Hemisphere's Landsat scenes in the northern zone, with negative northings). Landsat's pixel edges
    are at 15 mod 30 m in x and y of its own zone. When the items disagree the most common lattice wins."""
    offsets = Counter()
    for item in items:
        native = _item_native(item)
        if native is None:
            continue
        code, t = native
        if abs(t[0]) != resolution or abs(t[4]) != resolution or t[1] != 0 or t[3] != 0:
            continue
        if code == epsg:
            shift = 0.0
        elif 32601 <= code <= 32660 and epsg == code + 100:     # north -> south of the same zone
            shift = 10_000_000.0
        elif 32701 <= code <= 32760 and epsg == code - 100:     # south -> north
            shift = -10_000_000.0
        else:
            continue
        offsets[(round(t[2] % resolution, 6), round((t[5] + shift) % resolution, 6))] += 1
    if not offsets:
        return None
    return offsets.most_common(1)[0][0]


def _snap_to_lattice(bounds, lattice: Tuple[float, float], resolution: float) -> Tuple[float, float, float, float]:
    """``(minx, miny, maxx, maxy)`` widened to the pixel edges of a lattice (offsets ``(x, y)`` mod ``resolution``)."""
    lx, ly = lattice
    minx, miny, maxx, maxy = bounds
    # a tolerance of a millionth of a pixel: a bound exactly on an edge must not move a whole pixel
    eps = 1e-6
    return (math.floor((minx - lx) / resolution + eps) * resolution + lx,
            math.floor((miny - ly) / resolution + eps) * resolution + ly,
            math.ceil((maxx - lx) / resolution - eps) * resolution + lx,
            math.ceil((maxy - ly) / resolution - eps) * resolution + ly)


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
    platforms: Optional[Sequence[str]] = None,
    slc_off: bool = False,
    align: str = "native",
) -> xr.DataArray:
    """
    Builds a lazy Dask-backed xarray DataCube from a STAC catalog.
    
    source: A string from STAC_CATALOGS or a custom STAC API URL.
    collection: The dataset collection ID (e.g., "sentinel-2-l2a", "CBERS4A_WFI_L4_SR"). For
        HLS (Harmonized Landsat Sentinel-2: "hls2-l30", "hls2-s30" on Planetary Computer),
        a list of both builds each and joins them into one series sorted by time.
    bbox: [minx, miny, maxx, maxy] in WGS84 (EPSG:4326).
    vector_path: Path to a shapefile or geojson to derive the bounding box.
    tiles: List of specific MGRS/WRS tiles to fetch (e.g., ["20LKP"]).
    start_date, end_date: YYYY-MM-DD strings.
    cloud_cover_max: Maximum cloud cover percentage for image filtering.
    bands: List of band names to load (e.g., ["red", "green", "blue", "nir"]). For HLS, band
        roles (coastal, blue, green, red, nir, swir1, swir2) are read from each product's
        own asset (nir: L30's B05, S30's B8A) and keep the role's name.
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
    platforms: Keep only the items of these platforms, e.g. ``["landsat-8", "landsat-9"]``
        (the items' ``platform`` property, spelled ``landsat-5`` ... ``landsat-9`` by Planetary
        Computer and Earth Search). Applied to the items before anything is read.
    slc_off: Landsat 7 scenes acquired after its scan line corrector failed (2003-05-31) have
        stripes of no data in every band. They are dropped (default) unless ``slc_off=True``.
    align: ``"native"`` (default) puts the output grid on the pixel edges of the scenes' own grid
        when the scenes are in the requested ``epsg`` (or in the other hemisphere's twin of that
        UTM zone: Planetary Computer keeps the Southern Hemisphere's Landsat scenes in the northern
        zone, e.g. 32621 for 32721) at the requested ``resolution``, so no pixel is resampled:
        the bounds are widened to that lattice (Landsat's edges are at 15 mod 30 m, not at the
        multiples of 30 m stackstac snaps to) and passed to stackstac. Otherwise (another CRS or
        resolution, no ``bbox``) the grid is the one of ``"resolution"``: the bounds snapped to
        multiples of ``resolution``, as before.

    Read errors that `ERRORS_AS_NODATA` tolerates (a missing file, a corrupted block) still become
    nodata, but each file that had one raises a `ReadErrorWarning` naming it (see `READ_ERRORS`).
    """
    if align not in ("native", "resolution"):
        raise ValueError(f"Unknown align {align!r}. Use 'native' or 'resolution'.")
    hls = [_hls_kind(c) for c in ([collection] if isinstance(collection, str) else collection)]
    if all(hls) and len(hls) > 1:
        if bands is None or any(b not in HLS_ROLES["l30"] for b in bands if b != "Fmask"):
            raise ValueError("HLS L30 and S30 together need bands= as roles (coastal, blue, green, red, nir, swir1, "
                             "swir2): the same asset name is another band in each (B05, B11...)")
        parts = [build_time_series(source=source, collection=c, bbox=bbox, vector_path=vector_path, tiles=tiles,
                                   start_date=start_date, end_date=end_date, cloud_cover_max=cloud_cover_max,
                                   bands=list(bands), apply_cloud_mask=apply_cloud_mask, resolution=resolution,
                                   epsg=epsg, validate_items=validate_items, access_token=access_token,
                                   chunksize=chunksize, dtype=dtype, platforms=platforms, slc_off=slc_off,
                                   align=align) for c in collection]
        return join_in_time(parts)
    renames = {}
    if hls[0] and bands:   # roles -> this product's assets, named back after the roles below
        roles = HLS_ROLES[hls[0]]
        renames = {roles[b]: b for b in bands if b in roles}
        bands = [roles.get(b, b) for b in bands]
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
    try:
        catalog = Client.open(stac_url)
    except Exception as e:
        from ._network import connection_failed, offline_message

        if not connection_failed(e):
            raise
        raise RuntimeError(offline_message(
            f"the STAC catalog {stac_url}", "zeit fetch tiles for the tiles of stac_tiles, or by building the cube "
            "there and saving it with zeit.save_raster")) from e
    
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

    # 3.4 Platform and Landsat 7 SLC-off filter, on the items, before anything is read
    if platforms is not None or not slc_off:
        found = len(items_list)
        items_list = _filter_platforms(items_list, platforms, slc_off)
        if len(items_list) != found:
            print(f"Kept {len(items_list)} of {found} scenes (platforms={platforms}, slc_off={slc_off})")
        if not items_list:
            raise ValueError("No images found for the given criteria (every scene was dropped by platforms / slc_off).")

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
        if qa_band is None:
            warnings.warn(f"apply_cloud_mask: no known QA band for {col_name!r} (Sentinel-2's SCL, Landsat's "
                          "QA_PIXEL and HLS's Fmask are); the cube is not masked", stacklevel=2)
            
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
    if _ZeitReader is not None:
        stack_kwargs["reader"] = _ZeitReader
    if integer_output:
        stack_kwargs.update(fill_value=np.dtype(dtype).type(0), rescale=False)
    else:
        # stackstac refuses to rescale into float32 (a scale like 2.75e-05 isn't "safely"
        # castable), so the scale and offset are applied below instead. The NaN fill must
        # also be of the output type to pass its casting check.
        stack_kwargs.update(rescale=False, fill_value=np.dtype(dtype).type(np.nan))
    if bbox is not None:
        stack_kwargs["bounds_latlon"] = bbox
        lattice = _native_lattice(items_list, epsg, resolution) if align == "native" and resolution else None
        if lattice is not None:
            # The scenes' own pixel edges, not multiples of the resolution: stackstac takes the bounds as given.
            from stackstac.geom_utils import reproject_bounds

            stack_kwargs.pop("bounds_latlon")
            stack_kwargs["bounds"] = _snap_to_lattice(reproject_bounds(tuple(bbox), 4326, epsg), lattice, resolution)
            stack_kwargs["snap_bounds"] = False

    cube = stackstac.stack(items_list, **stack_kwargs)
    if hls[0] and not integer_output:   # HLS's fill values, in case the reader kept them
        # Fmask's fill (255) stays: _clear_mask reads it as not clear.
        fills = [40000 if str(b) in HLS_ANGLES else -1 if str(b) == "Fmask" else -9999 for b in cube.band.values]
        cube = cube.where(cube != xr.DataArray(np.array(fills, dtype=cube.dtype), dims="band",
                                               coords={"band": cube.band.values}))
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

    if renames:
        cube = cube.assign_coords(band=[renames.get(str(b), str(b)) for b in cube.band.values])
    return cube


def join_in_time(cubes: Sequence[xr.DataArray]) -> xr.DataArray:
    """Cubes on one grid with the same bands (one per collection, say) as one series sorted by
    time, with the coordinates they all have."""
    first = cubes[0]
    for cube in cubes[1:]:
        for d in ("y", "x"):
            if cube.sizes[d] != first.sizes[d] or (d in cube.coords and not np.allclose(cube[d], first[d])):
                raise ValueError("the cubes are on different grids: put them on one first, e.g. with the same "
                                 "bbox, resolution and epsg in build_time_series, or zeit.load_raster(..., like=)")
        if "band" in first.dims and list(map(str, cube.band.values)) != list(map(str, first.band.values)):
            raise ValueError(f"the cubes have other bands: {list(first.band.values)} and {list(cube.band.values)}")
    shared = set.intersection(*(set(c.coords) for c in cubes))
    cubes = [c.drop_vars([k for k in c.coords if k not in shared]) for c in cubes]
    out = xr.concat(cubes, "time", coords="minimal", compat="override", combine_attrs="drop_conflicts")
    return out.sortby("time")


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


def _clear_count(raw, band_idx, clear) -> np.ndarray:
    """Clear observations of each pixel of a raw block: the QA's clear (``clear``) and a value in every band."""
    ok = (raw[:, band_idx] != 0).all(axis=1)
    return (ok & clear).sum(axis=0)


def _composite_block(raw, band_idx, qa_idx, qa_band, scale, offset, method, count=False):
    """Reduce one spatial chunk of raw digital numbers to a composite.

    raw: ``(time, band, y, x)`` integer array, 0 = nodata.
    scale, offset: ``(time, len(band_idx))`` per-scene factors (they differ between
        scenes when a provider changes processing baseline).
    Returns ``(len(band_idx), y, x)`` float32 reflectance; with ``count``, one more layer: the
    pixel's clear observations (`_clear_count`).
    """
    clear = _clear_mask(raw[:, qa_idx], qa_band) if qa_idx is not None else True
    out = _composite_values(raw, band_idx, clear, scale, offset, method)
    if count:
        n = _clear_count(raw, band_idx, clear).astype(np.float32)
        out = np.concatenate([out, n[None]])
    return out


def _composite_values(raw, band_idx, clear, scale, offset, method):
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
    processing baseline: reflectance = DN / 10000, minus 0.1 from baseline 04.00 on. Earth Search's
    items list that offset too but have it already applied (``earthsearch:boa_offset_applied``): none. HLS
    items have none either: reflectance is DN / 10000, L30's thermal bands and the angles
    DN / 100 (HLS v2.0 User Guide, Tables 6-8).
    """
    bands = list(cube.band.values)
    scale = np.ones((cube.sizes["time"], len(bands)), dtype=np.float32)
    offset = np.zeros_like(scale)
    by_id = {item.id: item for item in items}
    for i, item_id in enumerate(cube.coords["id"].values):
        item = by_id.get(item_id)
        if item is None:
            continue
        props = getattr(item, "properties", None) or {}
        baseline = props.get("s2:processing_baseline")
        # Earth Search has already taken the BOA offset off its COGs (baseline 04.00 on) but still lists
        # it in raster:bands: applying it again would take 0.1 off every reflectance.
        applied = props.get("earthsearch:boa_offset_applied") is True
        hls = str(item_id).upper().startswith("HLS.")
        for k, band in enumerate(bands):
            asset = item.assets.get(band)
            meta = asset.extra_fields.get("raster:bands") if asset is not None else None
            if isinstance(meta, list) and meta and isinstance(meta[0], dict):
                scale[i, k] = meta[0].get("scale", 1.0)
                offset[i, k] = 0.0 if applied else meta[0].get("offset", 0.0)
            elif hls:   # HLS items carry no raster:bands: reflectance x 10000, thermal and angles x 100
                thermal = ".L30." in str(item_id).upper() and str(band) in ("B10", "B11")
                scale[i, k] = 1.0 if str(band) == "Fmask" else 0.01 if thermal or str(band) in HLS_ANGLES else 1e-4
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


def _composite_cube(cube, bands, qa_band, method, count=False):
    """Lazy composite of a raw integer cube: one dask task per spatial chunk (with ``count``, one more layer:
    the clear observations of each pixel, from the same read)."""
    import dask.array as da

    names = list(cube.band.values)
    band_idx = [names.index(b) for b in bands]
    qa_idx = names.index(qa_band) if qa_band else None
    scale, offset = _cube_scale_offset(cube, bands)
    # All scenes and bands of a spatial chunk in one block, so every byte is read once.
    raw = cube.data.rechunk({0: -1, 1: -1})
    return da.map_blocks(
        _composite_block, raw, band_idx, qa_idx, qa_band, scale, offset, method, count,
        drop_axis=0, dtype=np.float32, chunks=((len(bands) + int(count),),) + raw.chunks[2:],
    )


#: The quality states of the best available pixel composite (zeit-core's bap): no observation (outside the scene, a
#: fill, a defective pixel), a cloud or its shadow (the clear pixels near it score less), a clear observation.
BAP_NO_DATA, BAP_CLOUD, BAP_CLEAR = 0, 1, 2

#: The default weights of the best available pixel's score: its day of the year near the target day, its distance
#: to the nearest cloud or shadow, its sensor, its scene's cloud cover.
BAP_WEIGHTS = dict(doy=0.3, cloud=0.5, sensor=0.1, cover=0.1)

#: The bands a best available pixel composite adds with ``bap_info=True``: the day of the year of each pixel's
#: date, its score and how many clear observations the pixel had.
BAP_INFO_BANDS = ("bap_doy", "bap_score", "bap_clear")


def _qa_states(qa: np.ndarray, qa_band: Optional[str]) -> np.ndarray:
    """No data, cloud or clear (BAP_*) of each pixel of an integer QA array.

    Clouds are what scores the pixels near them lower: Sentinel-2's SCL cloud shadow, medium and high cloud
    probability and thin cirrus (3, 8, 9, 10); Landsat QA_PIXEL's dilated cloud, cirrus, cloud and shadow bits (1-4);
    HLS Fmask's cloud, adjacent and shadow bits (1-3). Clear is what `_clear_mask` keeps; the rest (no data, a
    defective pixel, a dark area...) is neither.
    """
    if qa_band is None:
        return np.full(qa.shape, BAP_CLEAR, dtype="uint8")
    clear = _clear_mask(qa, qa_band)
    name = qa_band.lower()
    if name == "scl":
        cloud = np.isin(qa, [3, 8, 9, 10])
    elif name == "fmask":
        cloud = ((qa & 0b1110) != 0) & (qa != 255)
    else:
        cloud = ((qa & 0b11110) != 0) & ((qa & 1) == 0)
    states = np.where(clear, BAP_CLEAR, np.where(cloud, BAP_CLOUD, BAP_NO_DATA)).astype("uint8")
    return states


def _sensor_scores(platforms, times) -> np.ndarray:
    """Each scene's sensor score: 0.5 for Landsat 7 after its scan line corrector failed (2003-05-31), else 1."""
    out = np.ones(len(times), dtype="float64")
    slc_off = np.datetime64("2003-05-31")
    for i, (p, t) in enumerate(zip(platforms, times)):
        p = str(p).lower().replace("_", "-")
        if ("landsat-7" in p or p == "le07" or "etm" in p) and np.datetime64(t, "D") >= slc_off:
            out[i] = 0.5
    return out


def _scene_dates(cube: xr.DataArray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Day of the year, sensor score and cover score (1 - eo:cloud_cover / 100) of each scene of a cube."""
    times = pd.DatetimeIndex(cube.time.values)
    doy = times.dayofyear.values.astype("float64")
    t = cube.sizes["time"]
    platforms = cube.coords["platform"].values if "platform" in cube.coords else [""] * t
    platforms = np.broadcast_to(np.asarray(platforms, dtype=object), (t,))
    sensor = _sensor_scores(platforms, cube.time.values)
    if "eo:cloud_cover" in cube.coords:
        cc = np.broadcast_to(np.asarray(cube.coords["eo:cloud_cover"].values, dtype="float64"), (t,))
        cover = np.where(np.isfinite(cc), 1.0 - np.clip(cc, 0, 100) / 100.0, 1.0)
    else:
        cover = np.ones(t)
    return doy, sensor, cover


def _bap_block(raw, band_idx, qa_idx, qa_band, scale, offset, doy, sensor, cover, opts, info, pads, count=False):
    """The best available pixel composite of one spatial chunk of raw digital numbers (with `pads` = (top, bottom,
    left, right) rows and columns of its neighbours around it, the margin of the cloud distance).

    raw: ``(time, band, y, x)`` integer array, 0 = nodata. Each pixel takes, of its clear observations (the QA's
    and every band's), the one of the highest score (zeit-core's bap: day of the year, distance to clouds, sensor,
    cover). Returns ``(len(band_idx) [+ 3], y, x)`` float32 reflectance (and the date's day of the year, its score,
    the pixel's clear observations), the margins cut off; with ``count``, one more last layer, the same clear
    observations (the ones that count for the score), for ``clear_count``.
    """
    from ._core import bap

    t = raw.shape[0]
    if qa_idx is not None:
        states = _qa_states(raw[:, qa_idx], qa_band)
    else:
        states = np.full((t,) + raw.shape[2:], BAP_CLEAR, dtype="uint8")
    # a band without a value: no observation
    empty = (raw[:, band_idx] == 0).any(axis=1)
    states[empty & (states == BAP_CLEAR)] = BAP_NO_DATA
    date, score, clear = bap.best_dates(np.ascontiguousarray(states), doy, sensor, cover, **opts, n_jobs=1)
    top, bottom, left, right = pads
    ys = slice(top, raw.shape[2] - bottom)
    xs = slice(left, raw.shape[3] - right)
    date, score, clear = date[ys, xs], score[ys, xs], clear[ys, xs]
    sub = raw[:, :, ys, xs]
    has = date >= 0
    d = np.where(has, date, 0)
    out = np.full((len(band_idx) + (3 if info else 0) + int(count),) + date.shape, np.nan, dtype=np.float32)
    for k, i in enumerate(band_idx):
        v = np.take_along_axis(sub[:, i], d[None], 0)[0].astype(np.float32)
        v = v * scale[d, k] + offset[d, k]
        out[k] = np.where(has, v, np.nan)
    if info:
        out[len(band_idx)] = np.where(has, doy[d], np.nan)
        out[len(band_idx) + 1] = score
        out[len(band_idx) + 2] = clear
    if count:
        out[-1] = clear
    return out


def _bap_opts(target_doy, doy_sigma, cloud_distance, weights) -> dict:
    """zeit._core.bap.best_dates's settings, the cloud distances still in metres (`_bap_cube` makes them pixels)."""
    w = {**BAP_WEIGHTS, **(weights or {})}
    unknown = set(w) - set(BAP_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown weights {sorted(unknown)}: use {sorted(BAP_WEIGHTS)}.")
    lo, hi = cloud_distance
    if not (0 <= lo < hi):
        raise ValueError(f"cloud_distance {cloud_distance}: (near, far) metres with 0 <= near < far.")
    return dict(target_doy=float(target_doy), doy_sigma=float(doy_sigma), cloud_min=float(lo),
                cloud_max=float(hi), w_doy=w["doy"], w_cloud=w["cloud"], w_sensor=w["sensor"],
                w_cover=w["cover"])


def _at_least(chunks: Tuple[int, ...], size: int) -> Tuple[int, ...]:
    """Chunks of an axis merged so that each is at least `size` long (or the axis is one chunk)."""
    out = []
    for c in chunks:
        if out and (out[-1] < size or c < size):
            out[-1] += c
        else:
            out.append(c)
    if len(out) > 1 and out[-1] < size:
        out[-2] += out.pop()
    return tuple(out)


def _pixel_metres(cube: xr.DataArray) -> float:
    """A cube's pixel size in metres: its x spacing, in degrees of longitude at its middle latitude when its CRS is
    geographic."""
    dx = abs(float(cube.x.values[1] - cube.x.values[0])) if cube.sizes["x"] > 1 else float(
        cube.attrs.get("resolution", 1.0) if np.isscalar(cube.attrs.get("resolution", 1.0)) else 1.0)
    epsg = int(cube.coords["epsg"]) if "epsg" in cube.coords else None
    if epsg == 4326:
        lat = float(np.mean(cube.y.values))
        return dx * 111_320.0 * np.cos(np.radians(lat))
    return dx


def _bap_cube(cube, bands, qa_band, opts, info, count=False):
    """Lazy best available pixel composite of a raw integer cube: one dask task per spatial chunk, each with the
    margin of the cloud distance read from its neighbours. `opts`' cloud distances in metres."""
    import dask.array as da

    px = _pixel_metres(cube)
    opts = dict(opts, cloud_min=opts["cloud_min"] / px, cloud_max=opts["cloud_max"] / px)
    names = list(cube.band.values)
    band_idx = [names.index(b) for b in bands]
    qa_idx = names.index(qa_band) if qa_band else None
    scale, offset = _cube_scale_offset(cube, bands)
    doy, sensor, cover = _scene_dates(cube)
    margin = int(np.ceil(opts["cloud_max"])) if opts["w_cloud"] > 0 and qa_idx is not None else 0
    raw = cube.data.rechunk({0: -1, 1: -1})
    if margin:
        raw = raw.rechunk({2: _at_least(raw.chunks[2], margin), 3: _at_least(raw.chunks[3], margin)})
    out_chunks = raw.chunks
    ny, nx = len(out_chunks[2]), len(out_chunks[3])
    # an axis of one chunk has no neighbours to take a margin from
    my, mx = (margin if ny > 1 else 0), (margin if nx > 1 else 0)
    padded = da.overlap.overlap(raw, depth={0: 0, 1: 0, 2: my, 3: mx}, boundary="none") if my or mx else raw

    def block(b, block_info=None):
        _, _, iy, ix = block_info[0]["chunk-location"]
        pads = (my if iy > 0 else 0, my if iy < ny - 1 else 0, mx if ix > 0 else 0, mx if ix < nx - 1 else 0)
        return _bap_block(b, band_idx, qa_idx, qa_band, scale, offset, doy, sensor, cover, opts, info, pads, count)

    n_out = len(bands) + (3 if info else 0) + int(count)
    return da.map_blocks(block, padded, drop_axis=0, dtype=np.float32,
                         chunks=((n_out,),) + out_chunks[2:])


def _season_middle(season: Tuple[str, str]) -> float:
    """The day of the year in the middle of a ("MM-DD", "MM-DD") window (one that wraps the new year too)."""
    a = pd.Timestamp(f"2001-{season[0]}").dayofyear
    b = pd.Timestamp(f"2001-{season[1]}").dayofyear
    if b < a:
        b += 365
    m = (a + b) / 2.0
    return m - 365 if m > 365 else m


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
                  access_token, chunksize, platforms=None, slc_off=False, align="native", counted=False):
    """Stack ``reduce(raw_cube) -> (len(out_bands), y, x)`` over each year's season window.

    Each year is read as raw integers (`build_time_series` with ``dtype="uint16"``) on the
    grid every year shares, and the results stacked to ``(time, band, y, x)``. Years
    without scenes are NaN, so the time axis has no gaps. With ``counted``, ``reduce`` returns one
    more layer, the clear observations, which becomes the ``n_clear`` coordinate (time, y, x) of
    int16 (0 in the years without scenes). ``attrs["scenes"]`` is a JSON object, year to the ids of
    the scenes read for it.
    """
    import dask.array as da

    if bbox is None and vector_path is None:
        raise ValueError("Provide bbox or vector_path so every year shares the same grid.")
    if vector_path is not None:
        bbox = list(gpd.read_file(vector_path).to_crs("EPSG:4326").total_bounds)
    wraps = season[1] < season[0]

    years = list(range(start_year, end_year + 1))
    results = {}
    scenes = {}
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
                platforms=platforms, slc_off=slc_off, align=align,
            )
        except ValueError as e:
            if "No images found" not in str(e):
                raise
            print(f"{year}: no scenes, filled with NaN")
            continue
        results[year] = reduce(cube)
        scenes[str(year)] = [str(v) for v in cube.coords["id"].values] if "id" in cube.coords else []
        if template is None:
            template = cube

    if template is None:
        raise ValueError("No images found for any year in the given criteria.")

    first = next(iter(results.values()))
    empty = da.full(first.shape, np.nan, dtype=np.float32, chunks=first.chunks)
    data = da.stack([results.get(y, empty) for y in years])
    n_clear = None
    if counted:   # the last layer of each year: slices of the same graph, read once if computed together
        n_clear = da.where(da.isnan(data[:, -1]), 0.0, data[:, -1]).astype(np.int16)
        data = data[:, :-1]

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
        attrs={**{k: v for k, v in template.attrs.items() if k in ("crs", "transform", "resolution")},
               "scenes": json.dumps({str(y): scenes.get(str(y), []) for y in years})},
        name=name,
    )
    if n_clear is not None:
        out = out.assign_coords(n_clear=(("time", "y", "x"), n_clear))
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
    target_doy: Optional[float] = None,
    doy_sigma: float = 30.0,
    cloud_distance: Tuple[float, float] = (30.0, 1500.0),
    weights: Optional[dict] = None,
    bap_info: bool = False,
    platforms: Optional[Sequence[str]] = None,
    slc_off: Optional[bool] = None,
    align: str = "native",
    clear_count: bool = False,
) -> xr.DataArray:
    """
    Builds one cloud-masked composite per year from a STAC catalog: the annual stack
    LandTrendr (and other annual trajectory methods) expects.

    Scenes are read as raw integers and reduced chunk by chunk (mask, per-scene scale and
    offset, then median, medoid or best available pixel), so memory stays bounded by a few
    chunks however many scenes a year has, and every pixel is downloaded once.

    source, collection, bbox, vector_path, cloud_cover_max, resolution, epsg,
    access_token, chunksize: as in `build_time_series`. An area (bbox or vector_path) is
        required so that every year shares the same grid.
    start_year, end_year: Inclusive range of years.
    season: ("MM-DD", "MM-DD") window inside each year. A window that wraps the new year
        (e.g. ("12-01", "02-28")) is labelled with the year it starts in.
    bands: Bands to composite, e.g. ["nir08", "swir22"] for NBR.
    method: "median" (per band), "medoid" (the real observation closest to the
        multi-band median, keeping bands consistent) or "bap" (the best available pixel:
        each pixel's clear observation of the highest score, all its bands from that one
        date; see below).
    apply_cloud_mask: Mask clouds and shadows with the collection's QA band (Landsat's
        QA_PIXEL: fill, dilated cloud, cirrus, cloud and cloud shadow must be 0 and clear 1;
        snow and water stay clear).
    platforms: Keep only the scenes of these platforms, e.g. ``["landsat-8", "landsat-9"]``
        (as in `build_time_series`).
    slc_off: Keep the Landsat 7 scenes acquired after its scan line corrector failed
        (2003-05-31), which have stripes of no data. Default None: dropped for "median" and
        "medoid" (False), kept for "bap" (True), whose sensor score already weighs them 0.5, so
        that they fill what the others do not.
    align: ``"native"`` (default) or ``"resolution"``: as in `build_time_series`. The grid is
        on Landsat's own pixel edges only where ``epsg`` is the scenes' UTM zone (or its
        other-hemisphere twin) and ``resolution`` is 30: the default ``epsg=4326`` is not, and
        keeps the old grid.
    clear_count: Add the ``n_clear`` coordinate (below). It comes from the same read as the
        composite and costs a few array operations, but a consumer that computes the composite
        and ``n_clear`` in two separate calls reads the scenes twice; so it is opt-in.
    target_doy, doy_sigma, cloud_distance, weights, bap_info: the best available pixel's
        score (method="bap"), a weighted sum, after Griffiths et al. (2013) and White et
        al. (2014), of what makes an observation good:

        - ``doy``: its day of the year near ``target_doy`` (default: the middle of the
          season), ``exp(-0.5 (days / doy_sigma)^2)``;
        - ``cloud``: its distance to the nearest cloud or shadow of its scene, 0 at the
          first of ``cloud_distance`` (metres) or nearer, 1 at the second or farther,
          linear between (a scene without clouds scores 1);
        - ``sensor``: 0.5 for Landsat 7 after its scan line corrector failed, else 1;
        - ``cover``: 1 - the scene's cloud cover (``eo:cloud_cover``).

        ``weights`` changes theirs (default ``BAP_WEIGHTS``: doy 0.3, cloud 0.5, sensor
        0.1, cover 0.1), e.g. ``{"sensor": 0}``. Each chunk is read with the margin of
        the cloud distance around it, so the composite is the same whatever the chunks.
        ``bap_info=True`` adds three bands: ``bap_doy`` (each pixel's date as its day of
        the year), ``bap_score`` and ``bap_clear`` (its clear observations).

    Returns a lazy float32 DataArray (time, band, y, x) of reflectance, with ``time`` on
    January 1st of each year and a ``year`` coordinate. Years without scenes are NaN.
    ``attrs["scenes"]`` is a JSON string, ``{"2019": ["LC08_L2SP_...", ...], ...}``: the ids of the
    scenes read for each year (a JSON string because netCDF attributes cannot hold a dict; use
    ``json.loads``). With ``clear_count=True`` the ``n_clear`` coordinate, ``(time, y, x)`` int16,
    holds each pixel's clear observations per year: the QA band's clear (all of them if there is
    no QA band) with a value in every composited band; 0 in years without scenes.
    """
    if method not in ("median", "medoid", "bap"):
        raise ValueError(f"Unknown method {method!r}. Use 'median', 'medoid' or 'bap'.")
    if bbox is None and vector_path is None:
        raise ValueError("Provide bbox or vector_path so every year shares the same grid.")
    if not bands:
        raise ValueError("Provide the bands to composite, e.g. ['nir08', 'swir22'].")

    qa_band = _qa_band_for(collection.lower(), source) if apply_cloud_mask else None
    load = list(bands) + ([qa_band] if qa_band and qa_band not in bands else [])
    out_bands = list(bands)
    if method == "bap":
        opts = _bap_opts(_season_middle(season) if target_doy is None else target_doy, doy_sigma,
                         cloud_distance, weights)
        if bap_info:
            out_bands += list(BAP_INFO_BANDS)
        reduce = lambda cube: _bap_cube(cube, bands, qa_band, opts, bap_info, clear_count)
    else:
        reduce = lambda cube: _composite_cube(cube, bands, qa_band, method, clear_count)
    return _annual_stack(
        reduce,
        out_bands, f"{method}_composite",
        source=source, collection=collection, bbox=bbox, vector_path=vector_path,
        start_year=start_year, end_year=end_year, season=season, load=load,
        cloud_cover_max=cloud_cover_max, resolution=resolution, epsg=epsg,
        access_token=access_token, chunksize=chunksize,
        platforms=platforms, slc_off=(method == "bap") if slc_off is None else slc_off, align=align,
        counted=clear_count,
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
    platforms: Optional[Sequence[str]] = None,
    slc_off: bool = False,
    align: str = "native",
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
    platforms, slc_off, align: as in `build_time_series` (Landsat 7 scenes after 2003-05-31 are
        dropped unless ``slc_off=True``). ``attrs["scenes"]`` lists each year's scenes, as in
        `build_annual_composites`.

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
        platforms=platforms, slc_off=slc_off, align=align,
    )
