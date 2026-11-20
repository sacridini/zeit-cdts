"""The synthetic scenes of the change detection benchmarks (imad.py, omnibus.py): a six-band pair for iMAD and
radiometric normalization, and a dual-polarization Sentinel-1 series with known change dates for the omnibus test.
Both are saved as GeoTIFFs (the originals read files) and returned as arrays.
"""
import os

import numpy as np
import rasterio
from rasterio.transform import from_origin

# six spectral signatures (blue, green, red, NIR, SWIR1, SWIR2): water, forest, crop, bare soil, built-up
CLASSES = np.array([[0.04, 0.05, 0.04, 0.02, 0.01, 0.01],
                    [0.02, 0.05, 0.03, 0.45, 0.22, 0.10],
                    [0.04, 0.09, 0.06, 0.30, 0.25, 0.14],
                    [0.10, 0.14, 0.20, 0.26, 0.38, 0.33],
                    [0.14, 0.16, 0.18, 0.22, 0.26, 0.24]], dtype=np.float64)


def _write(path, array, nodata=None):
    array = np.asarray(array, dtype=np.float32)
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2], count=array.shape[0],
                       dtype="float32", crs="EPSG:32620", transform=from_origin(500000, 9000000, 30, 30),
                       nodata=nodata) as dst:
        dst.write(array)


def optical_pair(size, change=0.0, seed=1, sigma=0.004, path=None):
    """Two dates of ``size`` x ``size`` pixels, six bands: date 1 a map of five land covers with spatial grain and
    noise, date 2 the same scene seen by another sensor state (a gain per band, an offset, noise of the same
    ``sigma``), where a share ``change`` of the pixels (square blocks and a band) changed cover.

    Returns ``(x, y, truth)``: float32 ``(6, size, size)`` each and the boolean map of the changed pixels.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 5, (size // 16 + 1, size // 16 + 1))
    cover = np.kron(coarse, np.ones((16, 16), dtype=int))[:size, :size]
    truth = np.zeros((size, size), bool)
    block = max(8, size // 12)
    while truth.mean() < change:
        r, c = rng.integers(0, size - block, 2)
        truth[r:r + block, c:c + block] = True
    cover2 = np.where(truth, (cover + 1 + rng.integers(0, 3, (size, size))) % 5, cover)
    texture = rng.normal(0, 0.02, (size, size))        # a within-class brightness that both dates share
    base = CLASSES[cover] * (1 + texture[..., None])
    base2 = CLASSES[cover2] * (1 + texture[..., None])
    x = base + rng.normal(0, sigma, base.shape)
    gain = np.array([1.12, 0.95, 1.05, 0.88, 1.18, 0.92])
    offset = np.array([0.012, -0.008, 0.005, 0.02, -0.015, 0.004])
    y = base2 * gain + offset + rng.normal(0, sigma, base.shape)
    x = np.moveaxis(x, -1, 0).astype(np.float32)
    y = np.moveaxis(y, -1, 0).astype(np.float32)
    if path is not None:
        _write(f"{path}_1.tif", x)
        _write(f"{path}_2.tif", y)
    return x, y, truth


def sar_series(size, k=12, enl=4.4, seed=2, path=None):
    """A VV and VH series of ``k`` dates, ``size`` x ``size`` pixels, linear power with gamma speckle of ``enl``
    looks: a background of random mean backscatter, and three blocks with known changes (as the tests):
    A, both bands x 0.3 from date 6 on (clear-cut); B, x 3 at date 3 and back at date 8 (a flood);
    C, VV x 3 and VH x 0.3 from date 5 on (indefinite).

    Returns ``(series, truth)``: float32 ``(k, 2, size, size)`` and the int16 ``(3, size, size)`` truth (the index
    of the date of the first change, -1 if none; the second and the number of changes).
    """
    rng = np.random.default_rng(seed)
    mean = np.stack([np.exp(rng.normal(np.log(0.12), 0.45, (size, size))),
                     np.exp(rng.normal(np.log(0.03), 0.45, (size, size)))])
    factor = np.ones((k, 2, size, size))
    a = (slice(size // 10, size // 10 + size // 5),) * 2
    b = (slice(size // 2, size // 2 + size // 5), slice(size // 10, size // 10 + size // 5))
    c = (slice(size // 10, size // 10 + size // 5), slice(size // 2, size // 2 + size // 5))
    factor[(slice(6, None), slice(None)) + a] = 0.3
    factor[(slice(3, 8), slice(None)) + b] = 3.0
    factor[(slice(5, None), 0) + c] = 3.0
    factor[(slice(5, None), 1) + c] = 0.3
    series = rng.gamma(enl, 1.0 / enl, (k, 2, size, size)) * mean[None] * factor
    truth = np.full((3, size, size), -1, dtype=np.int16)
    truth[(0,) + a], truth[(2,) + a] = 6, 1
    truth[(0,) + b], truth[(1,) + b], truth[(2,) + b] = 3, 8, 2
    truth[(0,) + c], truth[(2,) + c] = 5, 1
    series = series.astype(np.float32)
    if path is not None:
        for i in range(k):
            _write(f"{path}_{i:02d}.tif", series[i])
    return series, truth


# ---------------------------------------------------------------------------
# Mort Canty's scripts (CRC5Docker, MIT), run as a black box by canty_run.py

def canty(*args, threads=4, timeout=3600):
    """Runs one of Canty's scripts (``canty("iMad.py", "-n", "-i", "50", file1, file2)``) in its own Python
    (``CANTY_PYTHON``, with gdal and scipy; ``CANTY_SRC`` the ``src`` folder of CRC5Docker, ``CANTY_LIB`` the folder
    with the compiled ``libprov_means`` on Windows) and returns ``(seconds, stdout)``: the wall time of the whole
    run, its own input and output included."""
    import subprocess
    import time
    from pathlib import Path

    py = os.environ["CANTY_PYTHON"]
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    root = Path(py).parent
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               MPLBACKEND="Agg", GDAL_NUM_THREADS="1",
               GDAL_DATA=str(root / "Library" / "share" / "gdal"), PROJ_DATA=str(root / "Library" / "share" / "proj"))
    env["PATH"] = os.pathsep.join([str(root), str(root / "Library" / "bin"), str(root / "Scripts")]
                                  + [p for p in env.get("PATH", "").split(os.pathsep)
                                     if not any(s in p for s in ("GDAL", "Strawberry"))])
    t0 = time.perf_counter()
    p = subprocess.run([py, "-I", str(Path(__file__).with_name("canty_run.py")), *map(str, args)], env=env,
                       capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    seconds = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{args[0]} failed:\n{p.stdout[-1500:]}\n{p.stderr[-2500:]}")
    return seconds, p.stdout
