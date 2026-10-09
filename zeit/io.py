import os
import numpy as np
import rasterio
from rasterio.transform import from_origin


from typing import Dict, Any, Optional, Tuple

from ._load import load_raster  # noqa: F401  (zeit.io.load_raster)
from ._save import save_raster  # noqa: F401  (zeit.io.save_raster)

def get_georef(reference_cube: Any) -> Dict[str, Any]:
    """
    Extracts the geographical reference (CRS and Transform) from a STAC/Xarray DataArray.
    
    Returns:
        dict: A dictionary containing 'crs' (str) and 'transform' (rasterio.Affine).
    """
    crs = "EPSG:4326"
    transform = None
    
    try:
        if hasattr(reference_cube, 'rio') and reference_cube.rio.crs is not None:
            crs = reference_cube.rio.crs
            transform = reference_cube.rio.transform()
        elif hasattr(reference_cube, 'transform'):
            transform = reference_cube.transform
            if hasattr(reference_cube, 'crs'):
                crs = reference_cube.crs
                
        if transform is None and 'x' in reference_cube.coords and 'y' in reference_cube.coords:
            from rasterio.transform import from_origin
            x_res = float(abs(reference_cube.x[1] - reference_cube.x[0]))
            y_res = float(abs(reference_cube.y[1] - reference_cube.y[0]))
            # x/y hold cell centres: the origin is the outer corner of the first cell.
            x_min = float(reference_cube.x.min()) - x_res / 2
            y_max = float(reference_cube.y.max()) + y_res / 2
            transform = from_origin(x_min, y_max, x_res, y_res)
            
    except Exception as e:
        pass
        
    return {
        'crs': crs,
        'transform': transform
    }
