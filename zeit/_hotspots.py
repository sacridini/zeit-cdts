"""``zeit.emerging_hotspots``: emerging hot spot analysis (ESRI's Space Time Pattern Mining, sfdep's
``emerging_hotspot_analysis``), the Getis-Ord Gi* of every pixel at every date and the Mann-Kendall trend of each
pixel's Gi*, each pixel classified into ESRI's seventeen patterns. The engine is ``_core.hotspots``, in Rust.
"""

from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._load import sorted_by_time
from ._series_api import _series_cube
from ._threads import resolve

#: The patterns by code (ESRI's): hot spots positive, cold spots negative, 0 no pattern.
PATTERNS = {
    1: "new hot spot", 2: "consecutive hot spot", 3: "intensifying hot spot", 4: "persistent hot spot",
    5: "diminishing hot spot", 6: "sporadic hot spot", 7: "oscillating hot spot", 8: "historical hot spot",
    0: "no pattern",
    -1: "new cold spot", -2: "consecutive cold spot", -3: "intensifying cold spot", -4: "persistent cold spot",
    -5: "diminishing cold spot", -6: "sporadic cold spot", -7: "oscillating cold spot", -8: "historical cold spot",
}
TRENDS = {"original": 0, "mann_kendall": 0, "hamed_rao": 1, "yue_wang": 2}


def emerging_hotspots(
    data: Any,
    radius: int = 1,
    *,
    shape: str = "square",
    lag: int = 1,
    reference: str = "date",
    alpha: float = 0.05,
    trend: str = "hamed_rao",
    band: Union[str, int, None] = None,
    nodata: Any = "auto",
    n_jobs: int = -1,
) -> xr.Dataset:
    """Emerging hot spot analysis: where values cluster high (hot spots) or low (cold spots), and how that
    changed over the dates.

    For each date, the Getis-Ord Gi* z-score of each pixel (Getis & Ord 1992): its neighbourhood's sum against the
    mean, the neighbourhood the pixels within ``radius`` (the pixel itself among them) at that date and the ``lag``
    dates before; a hot spot where it is significantly positive, a cold spot where significantly negative. Then the
    Mann-Kendall test of each pixel's series of Gi*, and its pattern as ESRI defines them: a *new* hot spot (hot at
    the last date only), *consecutive* (one run of hot dates at the end), *intensifying* / *persistent* /
    *diminishing* (hot on 90 % of the dates or more, the last one too, its Gi* rising significantly / without a
    significant trend / falling), *sporadic* (hot at the last date, on and off before, never cold), *oscillating*
    (hot at the last date, cold at some date before), *historical* (hot on 90 % of the dates but not at the last);
    the same for cold spots. Where two definitions hold, the first of that list.

    Parameters
    ----------
    data
        One variable (an index, a count, an anomaly): a ``(time, y, x)`` cube, a raster, or a Dataset with
        ``band=``. Missing values are left out of the sums and statistics.
    radius
        The neighbourhood in pixels: with ``shape="square"`` the (2 radius + 1)^2 pixels around (1: the 8
        neighbours, queen contiguity, sfdep's default); with ``"circle"`` those whose centres are within it.
    lag
        The dates before each date in its neighbourhood (sfdep's ``k``, ESRI's neighbourhood time step).
    reference
        ``"date"``: each date's Gi* against that date's mean and standard deviation (as sfdep: a location hot
        against the others at that date; with a lag, the dates before in its neighbourhood are measured against
        that date's mean too, so a rise everywhere lowers it); ``"cube"``: against the whole cube's (a date that is
        high everywhere is hot everywhere).
    alpha
        The significance of the Gi* (two-sided, normal) and of its trend.
    trend
        The trend test on the Gi* series: ``"hamed_rao"`` (default; the Gi* of consecutive dates share
        neighbours, so their series are autocorrelated), ``"original"`` Mann-Kendall (ESRI's and sfdep's) or
        ``"yue_wang"``.
    band, nodata, n_jobs
        As for ``zeit.mann_kendall``.

    Returns
    -------
    xarray.Dataset
        - ``category (y, x)`` int8: the pattern, ``1`` new ... ``8`` historical hot spot, ``-1`` ... ``-8`` the
          cold spots, ``0`` no pattern, ``-128`` fewer than 3 dates (``flag_values`` / ``flag_meanings``; the names
          in ``zeit.emerging_hotspots.PATTERNS``);
        - ``gi (time, y, x)`` the Gi* z-scores and ``p`` their p-values;
        - ``trend_z``, ``trend_p`` and ``tau``: the trend test of the Gi*;
        - ``n_hot`` and ``n_cold``: the dates a significant hot / cold spot.

    The Gi* equals sfdep 0.2.5's (``include_gi = TRUE``, queen neighbours, ``k`` = ``lag``) to 1e-14. sfdep's
    patterns use permutation p-values and rules of its own (it calls sporadic a pixel not significant at the last
    date); these follow ESRI's documentation with the normal p-values of the Gi*.

    Examples
    --------
    >>> ndvi = zeit.load_raster("ndvi_annual.tif")                       # (time, y, x), one map a year
    >>> ehs = zeit.emerging_hotspots(-ndvi, radius=3, shape="circle")   # where the losses concentrate
    >>> ehs.category.zeit.plot()

    References
    ----------
    Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics.
    Geographical Analysis, 24(3), 189-206. ESRI, "How Emerging Hot Spot Analysis works" (ArcGIS Pro).
    """
    from ._embeddings import refuse
    from . import _core

    refuse(data, "zeit.emerging_hotspots")
    if shape not in ("square", "circle"):
        raise ValueError(f"shape must be 'square' or 'circle', got {shape!r}")
    if reference not in ("date", "cube"):
        raise ValueError(f"reference must be 'date' or 'cube', got {reference!r}")
    if trend not in TRENDS:
        raise ValueError(f"trend must be one of {sorted(TRENDS)}, got {trend!r}")
    if int(radius) < 1 or int(lag) < 0:
        raise ValueError(f"radius must be at least 1 and lag at least 0, got {radius!r}, {lag!r}")
    cube, pixel = _series_cube(data, dates=None, band=band, chunks=None, need_dates=False, nodata=nodata)
    if pixel:
        raise ValueError("emerging_hotspots needs a map at each date (its neighbourhoods), not one pixel's series")
    cube = sorted_by_time(cube).transpose("time", "y", "x")
    values = np.ascontiguousarray(np.asarray(cube.values, dtype=np.float64))
    r = _core.hotspots.emerging(values, radius=int(radius), circle=shape == "circle", lag=int(lag),
                                cube_reference=reference == "cube", alpha=float(alpha), method=TRENDS[trend],
                                n_jobs=resolve(n_jobs))
    coords = {c: cube.coords[c] for c in ("time", "y", "x") if c in cube.coords}
    codes = sorted(PATTERNS)
    ds = xr.Dataset(
        {
            "category": (("y", "x"), r["category"]),
            "gi": (("time", "y", "x"), r["gi"].astype(np.float32)),
            "p": (("time", "y", "x"), r["p"].astype(np.float32)),
            "trend_z": (("y", "x"), r["trend_z"].astype(np.float32)),
            "trend_p": (("y", "x"), r["trend_p"].astype(np.float32)),
            "tau": (("y", "x"), r["tau"].astype(np.float32)),
            "n_hot": (("y", "x"), r["n_hot"]),
            "n_cold": (("y", "x"), r["n_cold"]),
        },
        coords=coords,
    )
    ds["category"].attrs.update(flag_values=codes, flag_meanings=" ".join(PATTERNS[c].replace(" ", "_")
                                                                         for c in codes), _FillValue=-128)
    ds.attrs = dict(algorithm="emerging hot spot analysis", radius=int(radius), shape=shape, lag=int(lag),
                    reference=reference, alpha=float(alpha), trend=trend)
    crs = getattr(cube.rio, "crs", None) if hasattr(cube, "rio") else None
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds


emerging_hotspots.PATTERNS = PATTERNS
