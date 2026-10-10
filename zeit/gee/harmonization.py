import warnings

import ee

def apply_cloud_mask(image: ee.Image) -> ee.Image:
    """
    Applies cloud and cloud shadow mask using the QA_PIXEL band.
    
    Args:
        image (ee.Image): Landsat Collection 2 Level 2 image.
        
    Returns:
        ee.Image: Cloud-masked image.
    """
    qa = image.select('QA_PIXEL')
    # Bits 1 (dilated cloud), 3 (cloud), 4 (cloud shadow)
    cloud_shadow_bitmask = (1 << 4)
    clouds_bitmask = (1 << 3)
    
    mask = (qa.bitwiseAnd(cloud_shadow_bitmask).eq(0)
            .And(qa.bitwiseAnd(clouds_bitmask).eq(0)))
    
    return image.updateMask(mask)

def harmonize_oli_to_etm(image: ee.Image) -> ee.Image:
    """
    Deprecated (removed in 2.0): rescales Landsat 8/9 (OLI) bands with coefficients that are
    not those of Roy et al. (2016) and have no known source, applied to the digital numbers
    rather than to reflectance. ``get_harmonized_collection`` no longer uses it.

    Args:
        image (ee.Image): Landsat 8 or 9 image.

    Returns:
        ee.Image: The rescaled image.
    """
    warnings.warn("harmonize_oli_to_etm is deprecated and will be removed in zeit 2.0: its coefficients are not "
                  "Roy et al.'s (2016) and Landsat Collection 2 surface reflectance is used across sensors "
                  "without an adjustment", DeprecationWarning, stacklevel=2)
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    # Simplified coefficients (slope, intercept) from Roy et al. 2016
    slopes = ee.Image.constant([0.9747, 0.9783, 0.9806, 1.0004, 0.9859, 0.9888])
    intercepts = ee.Image.constant([-0.0002, -0.0039, -0.0028, 0.0007, 0.0016, 0.0003])
    
    harmonized = image.select(bands).multiply(slopes).add(intercepts)
    # Return harmonized bands while preserving other properties and non-harmonized bands (like QA)
    return image.addBands(harmonized, overwrite=True)

def get_harmonized_collection(roi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    """
    Landsat 5, 7, 8 and 9 Collection 2 surface reflectance as one cloud-masked collection, with
    the bands of Landsat 5/7 renamed as Landsat 8/9's (SR_B2 blue ... SR_B7 SWIR2). The
    reflectance is not adjusted between sensors: Collection 2 is used across them as it comes
    (Earth Engine's FAQ on cross-sensor harmonization).
    
    Args:
        roi (ee.Geometry): Region of interest.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        
    Returns:
        ee.ImageCollection: The cloud-masked Landsat collection, SR_B2 ... SR_B7.
    """
    common_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    
    oli = l8.merge(l9).map(apply_cloud_mask).select(common_bands).map(lambda img: img.toFloat())
    
    l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
    
    def rename_etm(image: ee.Image) -> ee.Image:
        etm_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
        oli_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        return image.select(etm_bands, oli_bands).toFloat()
    
    etm = l5.merge(l7).map(apply_cloud_mask).map(rename_etm)
    
    # Merge and sort by time
    return oli.merge(etm).sort('system:time_start')
