"""
Example 05: TempCNN
"""
import os
import torch
import numpy as np
from zeit import build_time_series, save_raster
from zeit.ai import TempCNN

bbox = [-55.01, -11.01, -55.00, -11.00]
cube = build_time_series(bbox=bbox, start_date="2021-01-01", end_date="2021-06-30", source="earth_search", bands=['blue', 'green', 'red', 'nir', 'swir16', 'swir22'], resolution=30, epsg=3857, cloud_cover_max=20)

print("Downloading Cube...")
cube_data = np.nan_to_num(cube.compute().values)
T, C, H, W = cube_data.shape

reshaped_data = cube_data.transpose(2, 3, 1, 0).reshape(H * W, C, T)
tensor_data = torch.tensor(reshaped_data, dtype=torch.float32)

print("Initializing TempCNN...")
model = TempCNN(in_channels=C, n_times=T, num_classes=3)

print("Running Forward Pass...")
logits = model(tensor_data)
predictions = torch.argmax(logits, dim=1).view(H, W).numpy().astype(np.uint8)

out_tif = os.path.join("data", "tempcnn_prediction.tif")
save_raster(predictions, out_tif, like=cube, nodata=255)

print(f"Done! Output saved to {out_tif}")
