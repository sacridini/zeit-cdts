from ._landtrendr import desawtooth, apply_vertices
from ._lt import landtrendr
from ._ccdc_api import ccdc
from ._series_api import bfast_monitor, bfast_lite, bfast, mann_kendall, phenology
from .metrics import extract_events
from ._ccdc import predict_synthetic_image
from .twdtw import run_twdtw, run_twdtw_batch

from .spatial import apply_mmu_filter, apply_majority_filter
from .segmentation import run_snic, snic_grid, snic_to_polygons
from .smooth import apply_savgol_filter
from .masks import extract_water_mask
from .qc import qc_modis_summary, qc_modis_state, qc_sentinel2_scl

try:
    from .tmask import run_tmask_pixel, apply_tmask_stack
except ImportError:
    pass

from .local import build_local_cube
from .regularize import regularize_time_series

from .indices import compute_indices
from .io import load_raster, save_raster, get_georef
from .validation import generate_landtrendr_accuracy_dashboard

try:
    import zeit.xarray_api # This registers the xarray accessor automatically
except ImportError:
    pass

# Loaded on first use, so that `import zeit` stays fast: zeit.ai pulls in torch and
# transformers, the STAC cube builders pystac-client and stackstac, the classifier sklearn.
_LAZY = {
    "build_time_series": ".cube",
    "build_annual_composites": ".cube",
    "build_spectral_temporal_metrics": ".cube",
    "train_ccdc_classifier": ".classify",
    "classify_ccdc_stack": ".classify",
}
_LAZY_MODULES = {"ai", "cube", "classify"}


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
    "ccdc",
    "bfast_monitor", "bfast_lite", "bfast", "mann_kendall", "phenology",
    "extract_events", "predict_synthetic_image",
    "train_ccdc_classifier", "classify_ccdc_stack",
    "apply_mmu_filter", "apply_majority_filter", "apply_savgol_filter",
    "run_snic", "snic_grid", "snic_to_polygons",
    "extract_water_mask",
    "qc_modis_summary", "qc_modis_state", "qc_sentinel2_scl",
    "run_tmask_pixel", "apply_tmask_stack",
    "build_time_series",
    "build_annual_composites",
    "build_spectral_temporal_metrics",
    "compute_indices",
    "build_local_cube",
    "regularize_time_series",
    "load_raster", "save_raster",
    "generate_landtrendr_accuracy_dashboard",
    "ai"
]
