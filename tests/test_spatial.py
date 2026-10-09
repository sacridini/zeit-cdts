import numpy as np
import rasterio
import os
from zeit.spatial import apply_mmu_filter

def test_apply_mmu_filter(tmp_path):
    # Create a dummy raster with salt and pepper noise
    input_path = str(tmp_path / "noisy_map.tif")
    output_path = str(tmp_path / "filtered_map.tif")
    
    # 10x10 map
    # A single pixel of noise
    data = np.zeros((10, 10), dtype=np.uint8)
    data[5, 5] = 1
    
    # A larger patch (3x3 = 9 pixels, still below default MMU of 11)
    data[0:3, 0:3] = 1
    
    # A patch that should survive (4x3 = 12 pixels, > 11)
    data[6:10, 0:3] = 1
    
    profile = {
        "driver": "GTiff",
        "height": 10,
        "width": 10,
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "transform": rasterio.transform.from_origin(0, 0, 10, 10)
    }
    
    with rasterio.open(input_path, "w", **profile) as dst:
        dst.write(data, 1)
        
    import zeit

    zeit.save_raster(apply_mmu_filter(input_path, mmu_pixels=11), output_path)

    assert os.path.exists(output_path)

    with rasterio.open(output_path) as src:
        filtered = src.read(1)
        assert src.nodata == 0
        
    # The single pixel should be removed
    assert filtered[5, 5] == 0
    
    # The 3x3 patch (9 pixels) should be removed
    assert np.all(filtered[0:3, 0:3] == 0)
    
    # The 4x3 patch (12 pixels) should survive
    assert np.all(filtered[6:10, 0:3] == 1)


from zeit.spatial import apply_majority_filter

def test_majority_filter():
    # Create a 5x5 image with salt and pepper noise
    image = np.zeros((5, 5))
    image[2, 2] = 99 # Noise
    
    filtered = apply_majority_filter(image, size=3)
    # The noise pixel should be replaced by the surrounding 0s
    assert filtered[2, 2] == 0
