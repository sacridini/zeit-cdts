"""Regenerate tests/data/sccd_pyxccd.npz: pixels of a Landsat Collection 2 stack of Rondonia and
what pyxccd's ``sccd_detect`` (the authors' S-CCD, Apache-2.0, run as a black box: its source
was not read for zeit's) finds in them.

Run in an environment with pyxccd and rasterio (``pip install pyxccd rasterio pandas``), not
zeit's: ::

    python make_sccd_pyxccd_fixtures.py CCDC_Stack_Rondonia.tif CCDC_Stack_Rondonia_Dates.csv

The stack has 125 dates of 7 bands each (Blue, Green, Red, NIR, SWIR1, SWIR2 as Collection 2
digital numbers, then QA_PIXEL), 2020-02-04 to 2026-08-23; the CSV has an ``Ordinal_Day``
column. Four 32 x 32 windows are read and 384 of their pixels drawn at random.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

CORNERS = [(300, 300), (800, 800), (1300, 400), (500, 1300)]
SIZE = 32
N_PIXELS = 384
MAX_SEGS = 4


def fmask(qa_pixel: np.ndarray) -> np.ndarray:
    """Collection 2 QA_PIXEL bits as Fmask codes: 0 clear, 1 water, 2 shadow, 3 snow, 4 cloud."""
    qi = qa_pixel.astype(np.int64)
    cloud = (((qi >> 1) | (qi >> 2) | (qi >> 3)) & 1).astype(bool)
    code = np.full(qi.shape, 4, dtype=np.int64)
    code[((qi >> 6) & 1).astype(bool)] = 0
    code[((qi >> 7) & 1).astype(bool) & ~cloud] = 1
    code[((qi >> 5) & 1).astype(bool)] = 3
    code[((qi >> 4) & 1).astype(bool)] = 2
    code[cloud] = 4
    return code


def main(tif: str, csv: str) -> None:
    from pyxccd import sccd_detect

    dates = pd.read_csv(csv)["Ordinal_Day"].to_numpy().astype(np.int64)
    nt = len(dates)
    vals, qas = [], []
    with rasterio.open(tif) as src:
        for r, c in CORNERS:
            a = src.read(window=Window(c, r, SIZE, SIZE)).reshape(nt, 7, SIZE * SIZE)
            vals.append(a[:, :6])
            qas.append(a[:, 6])
    dn = np.concatenate(vals, axis=2)
    qp = np.concatenate(qas, axis=1)
    pick = np.sort(np.random.default_rng(0).choice(dn.shape[2], N_PIXELS, replace=False))
    dn, qp = dn[:, :, pick], qp[:, pick]

    sr = dn * 0.275 - 2000.0  # Collection 2 surface reflectance x 10000
    missing = np.isnan(qp) | np.isnan(sr).any(axis=1) | ((np.nan_to_num(qp).astype(np.int64) & 1) == 1)
    qa = np.where(missing, 255, fmask(np.nan_to_num(qp, nan=1.0)))
    values = np.where(missing[:, None], 0, np.round(np.nan_to_num(sr))).astype(np.int16)
    values, qa = values.transpose(2, 1, 0), qa.T.astype(np.uint8)  # (pixel, band, time), (pixel, time)

    n = values.shape[0]
    out = dict(
        n_breaks=np.zeros(n, np.int32), t_start=np.zeros((n, MAX_SEGS), np.int32),
        t_break=np.zeros((n, MAX_SEGS), np.int32), num_obs=np.zeros((n, MAX_SEGS), np.int32),
        rmse=np.zeros((n, MAX_SEGS, 6)), magnitude=np.zeros((n, MAX_SEGS, 6)),
        nrt_mode=np.zeros(n, np.int32), anomaly_conse=np.full(n, -1, np.int32), min_rmse=np.zeros((n, 6)),
    )
    for p in range(n):
        v = values[p].astype(np.int64)
        r = sccd_detect(dates, v[0], v[1], v[2], v[3], v[4], v[5], qa[p].astype(np.int64))
        segs = sorted(np.atleast_1d(r.rec_cg), key=lambda s: int(s["t_break"]))[:MAX_SEGS]
        out["n_breaks"][p] = len(segs)
        for k, s in enumerate(segs):
            out["t_start"][p, k] = int(s["t_start"])
            out["t_break"][p, k] = int(s["t_break"])
            out["num_obs"][p, k] = int(s["num_obs"])
            out["rmse"][p, k] = np.asarray(s["rmse"], float)[:6]
            out["magnitude"][p, k] = np.asarray(s["magnitude"], float)[:6]
        out["nrt_mode"][p] = int(r.nrt_mode)
        if int(r.nrt_mode) % 10 in (1, 3, 5):
            out["anomaly_conse"][p] = int(np.atleast_1d(r.nrt_model)[0]["anomaly_conse"])
        out["min_rmse"][p] = np.asarray(r.min_rmse, float)[:6]

    path = Path(__file__).parent / "sccd_pyxccd.npz"
    np.savez_compressed(path, dates=dates.astype(np.int32), values=values, qa=qa, **out)
    print(f"{n} pixels, {int(out['n_breaks'].sum())} breaks -> {path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
