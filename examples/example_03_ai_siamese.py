"""
Example 03: Siamese Change Detector
"""
import os
import torch
import numpy as np
from zeit import build_time_series, save_raster
from zeit.ai import SiameseChangeDetector

bbox = [-55.01, -11.01, -55.00, -11.00]
cube = build_time_series(bbox=bbox, start_date="2021-01-01", end_date="2021-12-31", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=5)

print("Extracting T0 and T1...")
img_t0 = np.nan_to_num(cube.isel(time=0).compute().values)
img_t1 = np.nan_to_num(cube.isel(time=-1).compute().values)

tensor_t0 = torch.tensor(img_t0, dtype=torch.float32).unsqueeze(0)
tensor_t1 = torch.tensor(img_t1, dtype=torch.float32).unsqueeze(0)

print("Initializing Siamese Network...")
model = SiameseChangeDetector(in_channels=6, num_classes=2)

print("Running Forward Pass...")
change_logits = model(tensor_t0, tensor_t1)
predicted_change = torch.argmax(change_logits, dim=1).squeeze().numpy().astype(np.uint8)

out_tif = os.path.join("data", "siamese_prediction.tif")
save_raster(predicted_change, out_tif, like=cube, nodata=255)

print(f"Done! Output saved to {out_tif}")
