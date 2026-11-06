"""Reads the FORCE reference outputs made by force_tsa.sh (read-only: rasterio, no Zeit).

The outputs are in ``workdir("force_tsa")/ref/out/<config>/X0000_Y0000/*_<PRODUCT>.tif`` (TSS, TSI, STM, FBY, FBQ, FBM,
FBW, FBD, TRY, TRM, CAY, CAM, NRT, and for the CSO configurations NUM, MIN, ...); the comparison scripts take the
series, the folds and the metrics from here, and FORCE's calendar (``ce_of``) and TSI grid (``tsi_grid``).
"""
import datetime
from pathlib import Path

import numpy as np
import rasterio

from common import workdir

REF = workdir("force_tsa") / "ref"

# FORCE's calendar has 365 days a year: 29 February is not there (it shares the doy of 1 March), so the doy comes from
# the table of a non-leap year (C/date-cl.c).
_CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def doy_of(yyyymmdd):
    y, m, d = int(yyyymmdd) // 10000, int(yyyymmdd) // 100 % 100, int(yyyymmdd) % 100
    if m == 2 and d == 29:
        d = 28 + 1  # shares the doy of 1 March (60)
    return int(_CUM[m - 1]) + d


def ce_of(yyyymmdd):
    """FORCE's continuous day: ``year * 365 + doy`` (doy from the non-leap table)."""
    return int(yyyymmdd) // 10000 * 365 + doy_of(yyyymmdd)


def tsi_grid(ce_min, ce_max, int_day):
    """The TSI's time steps: ``ce_min + i * int_day`` for ``i < floor((ce_max - ce_min + 1) / int_day)``."""
    return ce_min + np.arange((ce_max - ce_min + 1) // int_day) * int_day


def path(config, product):
    hits = sorted((REF / "out" / config / "X0000_Y0000").glob(f"*_{product}.tif"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{config}/{product}: {len(hits)} files in {REF / 'out' / config}")
    return hits[0]


def load(config, product):
    """One output of one configuration: ``(array int16 [bands, y, x], band names)``.

    The band names are FORCE's descriptions (a YYYYMMDD for TSS/TSI/FBY.. series, a statistic or a trend/change
    metric for the others).
    """
    with rasterio.open(path(config, product)) as ds:
        return ds.read().astype(np.int16, copy=False), list(ds.descriptions)


def times():
    """``{config: seconds}`` from times.tsv (wall time of each FORCE run, the reading and writing included)."""
    rows = [r.split("\t") for r in (REF / "times.tsv").read_text().splitlines()[1:]]
    return {r[0]: float(r[1]) for r in rows}


def truth():
    """The cube generator's truth.npz (dates, the clouds' masks per observation, the pixels' parameters)."""
    return np.load(REF / "truth.npz")
