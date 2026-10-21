import numpy as np
import dask.array as da
from typing import Optional

from zeit import _core

fit_mann_kendall_batch = _core.mannkendall.fit_mann_kendall_batch
MKMethod = _core.mannkendall.MKMethod

# trend, h, p, z, tau, s, var_s, slope, intercept
N_MK_METRICS = 9
MK_METRIC_NAMES = ["trend", "h", "p", "z", "tau", "s", "var_s", "slope", "intercept"]

_METHOD_MAP = {
    "original": int(MKMethod.ORIGINAL),
    "hamed_rao": int(MKMethod.HAMED_RAO),
    "yue_wang": int(MKMethod.YUE_WANG),
    "seasonal": int(MKMethod.SEASONAL),
}


def run_mann_kendall_dask(
    arr: da.Array,
    method: str = "hamed_rao",
    alpha: float = 0.05,
    lag: Optional[int] = None,
    period: int = 1,
    min_valid: int = 4,
    n_jobs: int = -1,
) -> da.Array:
    """
    Pixel-wise Mann-Kendall trend test + Theil-Sen slope across a Dask
    array's time axis. Ported from pymannkendall (Hussain & Mahmud, 2019,
    JOSS, doi:10.21105/joss.01556) to the compiled core for per-pixel throughput,
    with the same Dask map_blocks strategy as run_phenology_dask.

    Input array shape: (time, y, x).
    Output array shape: (9, y, x) - see MK_METRIC_NAMES for row order.
    Pixels with fewer than `min_valid` non-NaN observations are all-NaN.

    method:
      "original"  - classic Mann-Kendall (Mann 1945, Kendall 1975).
      "hamed_rao" - Hamed & Rao (1998) autocorrelation-corrected variance
                    (default; the standard choice for annual EO composites,
                    which are typically serially correlated).
      "yue_wang"  - Yue & Wang (2004) alternative autocorrelation correction.
      "seasonal"  - Hirsch & Slack (1984): reshapes the series into `period`
                    season slots (e.g. period=23 for MODIS 16-day annual
                    cycles, 12 for monthly data) and pools the per-season MK
                    scores, so a raw sub-annual time series can be tested for
                    trend directly without pre-aggregating to one
                    observation/year.

    IMPORTANT - units of slope/intercept: for "original"/"hamed_rao"/
    "yue_wang", slope is per TIME STEP (index spacing in `arr`, not calendar
    time) - pass one observation per year for a directly interpretable
    per-year trend. For "seasonal", slope is per full `period` cycle (e.g.
    per year, if `period` covers one year), so raw sub-annual composites
    already give a per-year slope without needing pre-aggregation.

    lag: number of first significant lags used by the autocorrelation
    correction (hamed_rao/yue_wang only). None (default) uses the full
    series length, matching pymannkendall's default.
    """
    if method not in _METHOD_MAP:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(_METHOD_MAP)}")
    method_int = _METHOD_MAP[method]
    lag_arg = -1 if lag is None else lag

    def _block(block):
        if block.size == 0:
            return np.full((N_MK_METRICS, block.shape[1], block.shape[2]), np.nan, dtype=np.float32)

        time_steps, rows, cols = block.shape
        pixels = rows * cols

        values_2d = np.ascontiguousarray(block.reshape(time_steps, pixels).T)

        out = fit_mann_kendall_batch(
            values_array=values_2d,
            method=method_int,
            alpha=alpha,
            lag=lag_arg,
            period=period,
            min_valid=min_valid,
            n_jobs=n_jobs,
        )  # (9, pixels)

        return out.reshape(N_MK_METRICS, rows, cols).astype(np.float32)

    return da.map_blocks(
        _block,
        arr,
        dtype=np.float32,
        drop_axis=[0],
        new_axis=[0],
        chunks=(N_MK_METRICS, arr.chunks[1], arr.chunks[2]),
    )
