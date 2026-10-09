"""``zeit.snic``: SNIC superpixels of an image or a cube, georeferenced, one call."""

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr


def snic(
    data: Any,
    *,
    spacing: Union[float, Tuple[float, float]] = 10,
    compactness: float = 0.5,
    seeds: Optional[np.ndarray] = None,
    grid: str = "rectangular",
    padding: Optional[Union[float, Tuple[float, float]]] = None,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    random_state: Optional[int] = None,
    nodata: Any = "auto",
    n_jobs: int = -1,
) -> xr.Dataset:
    """Segment an image or an image time series into SNIC superpixels (Achanta & Süsstrunk 2017).

    Every value of a pixel off its ``y``/``x`` (each band, each date, each band of each
    date) is one feature, so two pixels are alike when their whole trajectories are, as
    ``sits_segment`` does with the R ``snic`` package. Seeds are placed on R snic's grids
    (``snic_grid``); from the same seeds the labels are the reference implementation's.

    Parameters
    ----------
    data
        A map ``(y, x)``, a stack ``(band, y, x)``, a cube ``(time, y, x)`` or
        ``(time, band, y, x)`` (read into memory), or anything ``load_raster`` reads.
    spacing, grid, padding, random_state
        The seed grid: seeds every ``spacing`` pixels (one value or ``(row, col)``),
        ``grid`` ``"rectangular"``, ``"diamond"``, ``"hexagonal"`` or ``"random"``, a margin
        ``padding`` (default ``spacing / 2``) free of seeds.
    seeds
        Explicit ``(n, 2)`` (row, col) seeds instead of a grid.
    compactness
        Larger values give more regular segments; smaller ones follow the data more
        closely. The feature term is in data units: ``0.5`` (sits' default) for
        reflectance in 0-1, larger for data scaled by 10000.
    tile_size
        Segment tiles of this many pixels independently and in parallel (segments never
        cross tile edges). ``None``: the whole image at once.
    nodata
        Value marking a missing observation, as in ``zeit.landtrendr``: ``"auto"``, a
        number or ``None``. Pixels with any missing feature are left unlabelled.
    n_jobs
        CPU threads (``-1``: all but one).

    Returns
    -------
    xarray.Dataset
        - ``labels (y, x)``: the segment of every pixel (``-1``: unlabelled), georeferenced;
        - ``means (segment, ...)``: each segment's mean of every feature, with the cube's
          dims and coordinates (``(segment, time, band)`` for a ``(time, band, y, x)`` cube);
        - ``n_pixels``, ``centroid_x``, ``centroid_y`` ``(segment)``: size and centre (map
          coordinates; NaN for an empty segment).

        ``zeit.snic_to_polygons(result)`` makes polygons of it; ``save_raster`` writes the
        labels.

    Examples
    --------
    >>> ndvi = zeit.load_raster("S2_ndvi_2022.tif")             # (time, y, x)
    >>> seg = zeit.snic(ndvi, spacing=8, compactness=0.3)
    >>> polygons = zeit.snic_to_polygons(seg, include_means=True)
    """
    from ._load import load_raster
    from ._lt import _missing_values
    from .segmentation import run_snic

    if isinstance(data, np.ndarray):
        dims = {2: ("y", "x"), 3: ("band", "y", "x"), 4: ("time", "band", "y", "x")}.get(data.ndim)
        if dims is None:
            raise ValueError(f"snic takes 2-D to 4-D arrays, got shape {data.shape}")
        cube = xr.DataArray(data, dims=dims)
    elif isinstance(data, xr.DataArray):
        cube = data.compute()  # SNIC segments the whole image at once; no dates needed
    else:
        cube = load_raster(data)
    if "y" not in cube.dims or "x" not in cube.dims:
        raise ValueError(f"snic needs y and x dims, got {cube.dims}")
    features = [d for d in cube.dims if d not in ("y", "x")]
    cube = cube.transpose(*features, "y", "x")
    values = np.asarray(cube.values)
    sentinels = _missing_values(cube, nodata)
    values = values.astype(np.float64 if values.dtype != np.float32 else np.float32, copy=True)
    for value in sentinels:
        values[values == value] = np.nan
    res = run_snic(values, spacing=spacing, compactness=compactness, seeds=seeds, grid=grid, padding=padding,
                   tile_size=tile_size, random_state=random_state, n_jobs=n_jobs)

    n = len(res.sizes)
    rows, cols = res.centroids[:, 0], res.centroids[:, 1]
    if "x" in cube.coords and "y" in cube.coords and cube.sizes["x"] > 0 and cube.sizes["y"] > 0:
        from ._warp import transform_of

        t = transform_of(cube) if cube.sizes["x"] > 1 and cube.sizes["y"] > 1 else None
    else:
        t = None
    if t is not None:
        cx, cy = t * (cols + 0.5, rows + 0.5)
    else:
        cx, cy = cols + 0.5, rows + 0.5
    coords = {"segment": np.arange(n)}
    for name in ("y", "x"):
        if name in cube.coords:
            coords[name] = cube.coords[name]
    for d in features:
        if d in cube.coords:
            coords[d] = cube.coords[d]
    labels = xr.DataArray(res.labels, dims=("y", "x"), attrs={"long_name": "SNIC segment"})
    ds = xr.Dataset(
        {
            "labels": labels,
            "means": (("segment", *features), res.means),
            "n_pixels": ("segment", res.sizes),
            "centroid_x": ("segment", np.asarray(cx, dtype=np.float64)),
            "centroid_y": ("segment", np.asarray(cy, dtype=np.float64)),
        },
        coords=coords,
        attrs={"algorithm": "SNIC", "compactness": float(compactness)},
    )
    crs = cube.rio.crs if "y" in cube.dims else None
    if crs is not None:
        ds = ds.rio.write_crs(crs)
        if t is not None:
            ds = ds.rio.write_transform(t)
        ds["labels"] = ds["labels"].rio.write_nodata(-1, encoded=False)
    return ds


def as_result(ds: xr.Dataset):
    """A ``zeit.snic`` Dataset as the engine's ``SnicResult`` (for snic_to_polygons), with
    its transform, CRS and the names of its features."""
    from .segmentation import SnicResult

    labels = np.asarray(ds["labels"].values, dtype=np.int32)
    means = np.asarray(ds["means"].values)
    feature_dims = [d for d in ds["means"].dims if d != "segment"]
    names = _feature_names(ds["means"], feature_dims)
    transform = None
    if "x" in ds.coords and "y" in ds.coords and ds.sizes.get("x", 0) > 1 and ds.sizes.get("y", 0) > 1:
        from ._warp import transform_of

        transform = transform_of(ds["labels"])
    # centroids back to (row, col) pixel positions
    if transform is not None:
        cols, rows = ~transform * (ds["centroid_x"].values, ds["centroid_y"].values)
    else:
        cols, rows = ds["centroid_x"].values, ds["centroid_y"].values
    centroids = np.column_stack([np.asarray(rows) - 0.5, np.asarray(cols) - 0.5])
    result = SnicResult(labels=labels, means=means, centroids=centroids,
                        sizes=np.asarray(ds["n_pixels"].values, dtype=np.int64),
                        seeds=np.zeros((len(centroids), 2), dtype=np.int32))
    crs = ds.rio.crs if ds.rio.crs is not None else None
    return result, transform, crs, names


def _feature_names(means: xr.DataArray, dims: list) -> list:
    """Column names of the flattened features: their coordinates joined by '_'
    (dates as YYYY-MM-DD), or f0, f1... without coordinates."""
    if not dims:
        return [str(means.name or "value")]
    labels = []
    for d in dims:
        if d not in means.coords:
            labels.append([str(i) for i in range(means.sizes[d])])
            continue
        values = means.coords[d].values
        if np.issubdtype(values.dtype, np.datetime64):
            labels.append([str(v)[:10] for v in pd.DatetimeIndex(values).strftime("%Y-%m-%d")])
        else:
            labels.append([str(v) for v in values])
    names = labels[0]
    for more in labels[1:]:
        names = [f"{a}_{b}" for a in names for b in more]
    return names
