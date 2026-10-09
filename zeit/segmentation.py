"""SNIC superpixel segmentation of images and satellite image time series.

SNIC (Simple Non-Iterative Clustering; Achanta & Susstrunk, CVPR 2017) grows
every superpixel from a seed with one priority queue, joining at each step the
unlabelled pixel closest (in feature + image space) to the running mean of the
cluster that reached it. From the same seeds, the C++ core gives the same
labels as the authors' reference implementation, pixel for pixel.

For a time series every (time, band) pair is one feature, so two pixels are
close when their whole trajectories are - the approach sits_snic() takes with
the R `snic` package. The per-segment mean time series comes back with the
labels, ready to be classified as one sample per segment.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from . import _core

_GRIDS = ("rectangular", "diamond", "hexagonal", "random")


@dataclass
class SnicResult:
    """Output of :func:`run_snic`.

    labels: (y, x) int32, the segment (= seed index) of every pixel, -1 where
        the pixel had a NaN feature or could not be reached from any seed.
    means: (n_seeds, *feature_shape) float64, the segment mean of every
        feature, e.g. (n_seeds, time, band) for a (time, band, y, x) cube.
    centroids: (n_seeds, 2) float64, (row, col) centre of mass.
    sizes: (n_seeds,) int64 pixel counts. Segments seeded on a masked pixel
        are empty (size 0, NaN mean and centroid).
    seeds: (n_seeds, 2) int32, the (row, col) seeds used.
    """
    labels: np.ndarray
    means: np.ndarray
    centroids: np.ndarray
    sizes: np.ndarray
    seeds: np.ndarray


def snic_grid(
    shape: Tuple[int, int],
    spacing: Union[float, Tuple[float, float]],
    padding: Optional[Union[float, Tuple[float, float]]] = None,
    type: str = "rectangular",
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Seed grid of the R ``snic`` package (``snic_grid``), used by sits_snic().

    shape: (height, width) of the image.
    spacing: distance between seeds in pixels, one value or (row, col).
    padding: margin kept free of seeds, one value or (row, col); defaults to
        ``spacing / 2``.
    type: "rectangular", "diamond" (a second rectangular grid shifted by half
        a step, with the step scaled by sqrt(2)), "hexagonal" (the column step
        scaled by sqrt(3)) or "random" (as many seeds as the rectangular grid,
        sampled without replacement).

    Returns an (n, 2) int32 array of 0-based (row, col).
    """
    if type not in _GRIDS:
        raise ValueError(f"Unknown grid type {type!r}; choose from {_GRIDS}")
    h, w = int(shape[0]), int(shape[1])
    spacing = np.broadcast_to(np.asarray(spacing, dtype=float), (2,)).copy()
    padding = spacing / 2 if padding is None else np.broadcast_to(np.asarray(padding, dtype=float), (2,)).copy()
    if h < 1 or w < 1:
        raise ValueError("shape must be at least 1 x 1")
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 1):
        raise ValueError("spacing must be finite and > 1")
    if np.any(~np.isfinite(padding)) or np.any(padding < 0):
        raise ValueError("padding must be finite and >= 0")
    if padding[0] >= h / 2 or padding[1] >= w / 2:
        raise ValueError("padding leaves no room for seeds")

    def count(sp):
        return (int(np.floor((h - 2 * padding[0] - 1) / sp[0])) + 1,
                int(np.floor((w - 2 * padding[1] - 1) / sp[1])) + 1)

    def rect(sp):
        # R: seq(padding + 1, size - padding, length.out = n), 1-based.
        n_r, n_c = count(sp)
        if n_r < 1 or n_c < 1:
            raise ValueError("no valid seed positions for this spacing/padding")
        r0, r1 = padding[0] + 1, h - padding[0]
        c0, c1 = padding[1] + 1, w - padding[1]
        rows = np.array([(r0 + r1) / 2]) if n_r == 1 else np.linspace(r0, r1, n_r)
        cols = np.array([(c0 + c1) / 2]) if n_c == 1 else np.linspace(c0, c1, n_c)
        rr, cc = np.meshgrid(rows, cols, indexing="xy")  # expand.grid: rows vary fastest
        return np.column_stack([rr.ravel(), cc.ravel()])

    def staggered(sp):
        base = rect(sp)
        shifted = base + sp / 2
        keep = (shifted[:, 0] <= h - padding[0]) & (shifted[:, 1] <= w - padding[1])
        return np.vstack([base, shifted[keep]])

    if type == "rectangular":
        rc = rect(spacing)
    elif type == "diamond":
        rc = staggered(spacing * np.sqrt(2))
    elif type == "hexagonal":
        rc = staggered(spacing * np.array([1.0, np.sqrt(3)]))
    else:
        n_r, n_c = count(spacing)
        inner = (int(round(h - 2 * padding[0])), int(round(w - 2 * padding[1])))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(inner[0] * inner[1], size=min(n_r * n_c, inner[0] * inner[1]), replace=False)
        rc = np.column_stack(np.unravel_index(idx, inner, order="F")).astype(float) + 1 + padding

    # R snic rounds (half to even, like numpy) and passes 1-based indices.
    rc = np.round(rc).astype(np.int64) - 1
    rc[:, 0] = np.clip(rc[:, 0], 0, h - 1)
    rc[:, 1] = np.clip(rc[:, 1], 0, w - 1)
    return rc.astype(np.int32)


def run_snic(
    data: np.ndarray,
    spacing: Union[float, Tuple[float, float]] = 10,
    compactness: float = 0.5,
    seeds: Optional[np.ndarray] = None,
    grid: str = "rectangular",
    padding: Optional[Union[float, Tuple[float, float]]] = None,
    tile_size: Optional[Union[int, Tuple[int, int]]] = None,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> SnicResult:
    """Segment an image or an image time series into SNIC superpixels.

    data: (y, x), (feature, y, x) or (time, band, y, x). All leading axes are
        flattened into one feature vector per pixel. float32 input is used
        as-is (no float64 copy). Pixels with any NaN are left unlabelled (-1).
    seeds: explicit (n, 2) (row, col) pixel positions. When None (default),
        seeds are placed with the R snic / sits_snic() grids (snic_grid) from
        spacing (default 10, as sits_snic()), grid and padding.
    compactness: M in d = ||c_i - c_k||^2 + (M / S)^2 ||p_i - p_k||^2, with
        S = sqrt(N / K) the mean seed spacing. Larger values give more regular,
        compact segments; smaller values follow the data more closely. The
        feature term is in data units, so scale M with the data (0.5 is the
        sits default for reflectance-scaled cubes, ~10 is the reference
        default for 0-255 CIELAB images).
    tile_size: segment tiles of this many pixels (int or (rows, cols))
        independently and in parallel, like sits_segment() blocks. Segments
        never cross tile edges. None segments the whole image at once (SNIC
        itself is a sequential algorithm; OpenMP is then only used to reorder
        the data).
    n_jobs: OpenMP threads (-1 = all but one).
    """
    arr = np.asarray(data)
    if arr.ndim < 2:
        raise ValueError("data must have at least 2 dimensions (y, x)")
    feature_shape = arr.shape[:-2]
    height, width = arr.shape[-2:]
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    planar = np.ascontiguousarray(arr.reshape((-1, height, width)))

    if seeds is not None:
        seeds_rc = np.asarray(seeds, dtype=np.int32).reshape(-1, 2)
    else:
        seeds_rc = snic_grid((height, width), spacing, padding=padding, type=grid,
                             random_state=random_state)

    if tile_size is None:
        tile_h = tile_w = 0
    else:
        tile_h, tile_w = np.broadcast_to(np.asarray(tile_size, dtype=int), (2,))

    labels, means, centroids, sizes = _core.snic.snic_segment(
        planar, seeds_rc, compactness=float(compactness),
        tile_height=int(tile_h), tile_width=int(tile_w), n_jobs=int(n_jobs),
    )
    return SnicResult(
        labels=labels,
        means=means.reshape((len(seeds_rc),) + feature_shape),
        centroids=centroids,
        sizes=sizes,
        seeds=seeds_rc,
    )


def snic_to_polygons(result, transform=None, crs=None, include_means: bool = False):
    """Polygonise SNIC labels into a GeoDataFrame, one row per non-empty
    segment, like sits_segment(): columns ``supercells`` (label), ``x``/``y``
    (centroid in map coordinates), ``n_pixels`` and ``geometry``. With
    include_means, the segment means are added, one column per feature.

    result: the Dataset of ``zeit.snic`` (its transform, CRS and feature names
        are used; the means columns are named by its coordinates, e.g.
        ``2022-01-01_ndvi``), or the ``SnicResult`` of ``run_snic`` (means as
        ``f0, f1, ...``).
    transform: rasterio Affine of the image for a SnicResult (identity = pixel
        coordinates); taken from a Dataset.
    crs: CRS of the polygons for a SnicResult; taken from a Dataset.
    """
    import xarray as xr

    names = None
    if isinstance(result, xr.Dataset):
        from ._snic_api import as_result

        result, found_transform, found_crs, names = as_result(result)
        transform = found_transform if transform is None else transform
        crs = found_crs if crs is None else crs
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import shape
    from shapely.ops import unary_union

    transform = Affine.identity() if transform is None else transform
    labels = result.labels
    geoms = {}
    for geom, value in features.shapes(labels, mask=labels >= 0, transform=transform, connectivity=4):
        geoms.setdefault(int(value), []).append(shape(geom))

    ids = np.array(sorted(geoms), dtype=np.int64)
    rows, cols = result.centroids[ids, 0], result.centroids[ids, 1]
    # pixel centre of the centroid in map coordinates
    xs, ys = transform * (cols + 0.5, rows + 0.5)
    data = {
        "supercells": ids,
        "x": np.asarray(xs),
        "y": np.asarray(ys),
        "n_pixels": result.sizes[ids],
    }
    if include_means:
        flat = result.means.reshape(len(result.means), -1)[ids]
        for j in range(flat.shape[1]):
            data[names[j] if names is not None else f"f{j}"] = flat[:, j]
    geometry = [unary_union(geoms[i]) if len(geoms[i]) > 1 else geoms[i][0] for i in ids]
    return gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
