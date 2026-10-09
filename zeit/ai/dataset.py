import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset


class STACCubeDataset(Dataset):
    """
    PyTorch Dataset of the windows of a (time, band, y, x) cube, lazy (dask) or not.

    The windows cover the whole cube: they start every ``stride`` pixels, and the last
    one of each row and column lies against the far edge (a cube smaller than a window
    gives one window, padded with NaN). Only the window asked for is computed, so cubes
    larger than memory can be iterated.

    Each item is a dict:

    - ``x``: ``(time, band, patch_size, patch_size)``, float32, missing values left as NaN
      (fill them as your model needs; ``zeit.ai.samples``/``train`` do it for you);
    - ``valid``: ``(time, patch_size, patch_size)`` bool, True where every band has a value;
    - ``positions``: ``(time,)`` days since the cube's first date (for the positional
      encodings of UTAE and LightTAE; the day of the year would make dates a year apart
      collide);
    - ``row``, ``col``: the window's top-left pixel in the cube.
    """

    def __init__(self, cube: xr.DataArray, patch_size: int = 256, stride: int = 256):
        if cube.dims != ("time", "band", "y", "x"):
            if set(cube.dims) != {"time", "band", "y", "x"}:
                raise ValueError(f"STACCubeDataset takes a (time, band, y, x) cube, got {cube.dims}")
            cube = cube.transpose("time", "band", "y", "x")
        self.cube = cube
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.time_len, self.bands, self.h, self.w = cube.shape

        self.y_starts = self._starts(self.h)
        self.x_starts = self._starts(self.w)
        self.num_patches_y = len(self.y_starts)
        self.num_patches_x = len(self.x_starts)

        if "time" in cube.coords and np.issubdtype(cube["time"].dtype, np.datetime64):
            times = pd.DatetimeIndex(cube["time"].values)
            days = np.asarray((times - times[0]).total_seconds() / 86400.0)
        else:
            days = np.arange(self.time_len)
        self.positions = torch.tensor(days, dtype=torch.float32)

    def _starts(self, n: int) -> list:
        p, s = self.patch_size, self.stride
        if n <= p:
            return [0]
        starts = list(range(0, n - p + 1, s))
        if starts[-1] != n - p:
            starts.append(n - p)
        return starts

    def __len__(self) -> int:
        return self.num_patches_y * self.num_patches_x

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        y0 = self.y_starts[idx // self.num_patches_x]
        x0 = self.x_starts[idx % self.num_patches_x]
        p = self.patch_size
        values = np.asarray(self.cube.isel(y=slice(y0, y0 + p), x=slice(x0, x0 + p)).values, dtype=np.float32)
        patch = np.full((self.time_len, self.bands, p, p), np.nan, dtype=np.float32)
        patch[:, :, :values.shape[2], :values.shape[3]] = values
        x = torch.from_numpy(patch)
        return {"x": x, "valid": torch.isfinite(x).all(dim=1), "positions": self.positions,
                "row": y0, "col": x0}
