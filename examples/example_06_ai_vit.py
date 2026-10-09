"""
Example 06: Geospatial Foundation Model (ViT)
"""
import os
import torch
import numpy as np
from zeit import build_time_series, save_raster
from zeit.ai import GeoFoundationViT
from zeit.spatial import apply_majority_filter

bbox = [-55.01, -11.01, -55.00, -11.00]
cube = build_time_series(bbox=bbox, start_date="2021-01-01", end_date="2021-06-30", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=20)

print("Downloading Cube...")
cube_data = np.nan_to_num(cube.compute().values)

tensor_data = torch.tensor(cube_data, dtype=torch.float32).permute(1, 0, 2, 3).unsqueeze(0)

print("Initializing Foundation ViT Wrapper...")
model = GeoFoundationViT(num_classes=2)

print("Running Forward Pass...")
logits = model(tensor_data)
raw_predictions = torch.argmax(logits, dim=1).squeeze().numpy()

print("Applying Spatial Majority Filter...")
clean_predictions = apply_majority_filter(raw_predictions, size=3).astype(np.uint8)

out_tif = os.path.join("data", "vit_prediction.tif")
save_raster(clean_predictions, out_tif, like=cube, nodata=255)

print(f"Done! Output saved to {out_tif}")
