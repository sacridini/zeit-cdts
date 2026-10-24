# Windows: on import, rasterio adds every PATH folder holding gdal*.dll to the DLL search
# path (e.g. a standalone C:\Program Files\GDAL). That folder's own libexpat, OpenSSL or
# SQLite then shadow Python's when those standard-library modules are first imported
# afterwards ("DLL load failed while importing pyexpat"). Importing them first keeps Python's.
import sys as _sys

if _sys.platform == "win32":
    for _module in ("pyexpat", "_ssl", "_hashlib", "_sqlite3", "_lzma", "_bz2", "_ctypes"):
        try:
            __import__(_module)
        except ImportError:
            pass
    del _module
del _sys

from ._landtrendr import desawtooth, apply_vertices
from ._lt import landtrendr
from ._ccdc_api import ccdc
from ._sccd import sccd
from ._series_api import bfast_monitor, bfast_lite, bfast, mann_kendall, phenology
from ._parts import ar_trend, parts
from ._recovery import recovery_metrics
from ._ewmacd import ewmacd
from ._beast import beast
from .metrics import extract_events, agreement
from ._accuracy import sampling_design, stratified_sample, accuracy
from ._ccdc import predict_synthetic_image
from ._twdtw_api import twdtw

from .spatial import apply_mmu_filter, apply_majority_filter, desawtooth_classes, mmu_sieve, smooth_probs
from ._aoa import TrainDI, aoa, train_di
from ._hotspots import emerging_hotspots
from .segmentation import snic_grid, snic_to_polygons
from ._snic_api import snic
from ._som_api import som, clean_samples
from ._smooth import smooth
from .masks import extract_water_mask
from .qc import qc_modis_summary, qc_modis_state, qc_sentinel2_scl

from ._tmask_api import tmask
from ._sma import unmix
from ._coded import coded
from ._bandpass import bandpass_adjust

from .regularize import regularize_time_series

from .indices import compute_indices
from .io import load_raster, save_raster, get_georef

try:
    import zeit.xarray_api # This registers the xarray accessor automatically
except ImportError:
    pass

# Loaded on first use, so that `import zeit` stays fast: zeit.ai pulls in torch, the STAC cube builders pystac-client and stackstac, the classifiers sklearn.
_LAZY = {
    "build_time_series": ".cube",
    "build_annual_composites": ".cube",
    "build_spectral_temporal_metrics": ".cube",
    "train_classifier": "._classify_api",
    "classify": "._classify_api",
    "plot": "._plot",
    "interpret": "._plot._interpret",
    "load_embeddings": "._embeddings",
    "similarity": "._embedding_tools",
    "embedding_change": "._embedding_tools",
    "run_tiles": "._tiles",
    "run_worker": "._tiles",
    "invite_helpers": "._tiles",
    "tile_urls": "._tiles",
    "stac_tiles": "._tiles",
    "gee_tiles": "._tiles",
}
_LAZY_MODULES = {"ai", "cube"}


def __getattr__(name):
    import importlib

    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY:
        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'zeit' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__) | _LAZY_MODULES)

__all__ = [
    "landtrendr", "desawtooth", "apply_vertices",
    "ccdc", "sccd",
    "bfast_monitor", "bfast_lite", "bfast", "mann_kendall", "phenology",
    "ar_trend", "parts", "recovery_metrics", "ewmacd", "beast",
    "twdtw", "smooth", "tmask", "snic", "som", "clean_samples", "unmix", "coded",
    "extract_events", "agreement", "predict_synthetic_image",
    "sampling_design", "stratified_sample", "accuracy",
    "train_classifier", "classify",
    "apply_mmu_filter", "apply_majority_filter", "mmu_sieve", "smooth_probs", "desawtooth_classes",
    "aoa", "train_di", "TrainDI",
    "emerging_hotspots",
    "snic_grid", "snic_to_polygons",
    "extract_water_mask",
    "qc_modis_summary", "qc_modis_state", "qc_sentinel2_scl",
    "build_time_series",
    "bandpass_adjust",
    "build_annual_composites",
    "build_spectral_temporal_metrics",
    "compute_indices",
    "regularize_time_series",
    "load_raster", "save_raster", "plot", "interpret",
    "load_embeddings", "similarity", "embedding_change",
    "run_tiles", "run_worker", "invite_helpers", "tile_urls", "stac_tiles", "gee_tiles",
    "ai"
]
