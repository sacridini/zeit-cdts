import os
import numpy as np
import rasterio
from rasterio.transform import from_origin


from typing import Dict, Any, Optional, Tuple

from ._load import load_raster  # noqa: F401  (zeit.io.load_raster)

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
            x_min = float(reference_cube.x.min())
            y_max = float(reference_cube.y.max())
            x_res = float(abs(reference_cube.x[1] - reference_cube.x[0]))
            y_res = float(abs(reference_cube.y[1] - reference_cube.y[0]))
            transform = from_origin(x_min, y_max, x_res, y_res)
            
    except Exception as e:
        pass
        
    return {
        'crs': crs,
        'transform': transform
    }

def save_raster(array: "np.ndarray", output_path: str, reference_cube: Any = None, crs: str = "EPSG:4326", transform: Any = None, nodata: Optional[float] = None) -> None:
    """
    Effortlessly saves a 2D, 3D, or 4D array (or xarray cube) to a GeoTIFF file.
    If 4D (Time, Bands, Y, X), it flattens the first two dimensions into layers.
    """
    import os
    import numpy as np
    import rasterio
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if reference_cube is None and hasattr(array, 'coords') and 'x' in array.coords:
        reference_cube = array

    if hasattr(array, 'compute'):
        raw_array = array.compute()
        if hasattr(raw_array, 'values'):
            raw_array = raw_array.values
    elif hasattr(array, 'values'):
        raw_array = array.values
    else:
        raw_array = np.asarray(array)

    if raw_array.ndim == 2:
        count = 1
        height, width = raw_array.shape
        array_to_write = raw_array[np.newaxis, ...]
    elif raw_array.ndim == 3:
        count, height, width = raw_array.shape
        array_to_write = raw_array
    elif raw_array.ndim == 4:
        t, b, height, width = raw_array.shape
        count = t * b
        array_to_write = raw_array.reshape(count, height, width)
    else:
        raise ValueError(f"Array must be 2D, 3D or 4D, got {raw_array.ndim}D")

    if reference_cube is not None:
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
                x_min = float(reference_cube.x.min())
                y_max = float(reference_cube.y.max())
                x_res = float(abs(reference_cube.x[1] - reference_cube.x[0]))
                y_res = float(abs(reference_cube.y[1] - reference_cube.y[0]))
                transform = from_origin(x_min, y_max, x_res, y_res)
        except Exception as e:
            pass

    if transform is None:
        print("Warning: No geotransform provided. Saving with a dummy identity matrix.")
        transform = rasterio.Affine.identity()

    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': count,
        'dtype': array_to_write.dtype.name,
        'crs': crs,
        'transform': transform,
        'compress': 'deflate',
        'tiled': True
    }
    
    if nodata is not None:
        profile['nodata'] = nodata

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array_to_write)

    # Auto-extract dates to CSV if a time coordinate is present
    time_source = None
    if hasattr(array, 'coords') and 'time' in array.coords:
        time_source = array
    elif reference_cube is not None and hasattr(reference_cube, 'coords') and 'time' in reference_cube.coords:
        time_source = reference_cube

    if time_source is not None:
        try:
            import pandas as pd
            datetimes = pd.to_datetime(time_source.time.values)
            fractional_years = []
            for dt in datetimes:
                year = dt.year
                days_in_year = 366 if dt.is_leap_year else 365
                frac = year + (dt.dayofyear - 1) / days_in_year
                fractional_years.append(frac)
            
            df = pd.DataFrame({
                'Date': datetimes,
                'Fractional_Year': fractional_years,
                'Ordinal_Day': [dt.toordinal() for dt in datetimes]
            })
            
            base_path, _ = os.path.splitext(output_path)
            csv_path = f"{base_path}_dates.csv"
            df.to_csv(csv_path, index=False)
            print(f"Time coordinate detected. Extracted dates saved to {csv_path}")
        except ImportError:
            print("Warning: 'pandas' is required to auto-extract dates to CSV, but it is not installed.")
        except Exception as e:
            print(f"Warning: Failed to extract dates to CSV: {e}")
