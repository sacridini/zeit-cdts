"""iMAD change detection and relative radiometric normalization: ``zeit.imad``, ``zeit.radcal``.

The multivariate alteration detection (MAD) of Nielsen, Conradsen and Simpson (1998, RSE 64:1-19) takes the
canonical correlation analysis of the bands of two dates and gives the differences of the canonical variate pairs:
uncorrelated change images, ordered by their variance, that do not depend on the gain or offset (indeed on any
linear transform) of either date. Iterating with each pixel weighted by its probability of no change (Nielsen 2007,
IEEE TIP 16:463-478; Canty and Nielsen 2008, RSE 112:1025-1036) sharpens the separation of the changed pixels, and
the pixels that kept a high probability of no change are the invariant pixels the relative radiometric
normalization is fitted on (Canty, Nielsen and Schmidt 2004, RSE 91:441-451).

Written from the papers; the results agree with Canty's ``iMad.py`` and ``radcal.py`` (see the benchmark page).
"""

import json
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _change as ch

PENALTIES = ("ridge", "curvature")
STANDARDIZE = ("weighted", "all")


def _fit(x: xr.DataArray, y: xr.DataArray, missing: Sequence[float], *, max_iter: int, tol: float, lam: float,
         penalty: str, standardize: str, sample: Optional[int], n_jobs: int, who: str) -> Dict[str, Any]:
    """The iterations on the lattice sample of the two dates (all the valid pixels if `sample` is None)."""
    from . import _core

    if penalty not in PENALTIES:
        raise ValueError(f"penalty= must be one of {PENALTIES}, got {penalty!r}")
    if standardize not in STANDARDIZE:
        raise ValueError(f"standardize= must be one of {STANDARDIZE}, got {standardize!r}")
    if int(max_iter) < 1:
        raise ValueError(f"max_iter= must be at least 1, got {max_iter!r}")
    if float(lam) < 0:
        raise ValueError(f"lam= must be >= 0, got {lam!r}")
    n = x.sizes["band"]
    stride = ch.lattice_stride(x.sizes["y"], x.sizes["x"], sample)
    z = ch.valid_rows(ch.lattice(x, stride), ch.lattice(y, stride), missing)
    if z.shape[0] < 2 * (2 * n) + 2:
        raise ValueError(f"{who}: only {z.shape[0]} valid pixels in the sample for {n} bands")
    try:
        fit = _core.imad.fit(z, n, int(max_iter), float(tol), float(lam), penalty, standardize == "weighted", n_jobs)
    except ValueError as error:
        raise ValueError(f"{who}: {error}") from None
    for text in fit["warnings"]:
        warnings.warn(f"{who}: {text}", stacklevel=3)
    fit["stride"] = stride
    fit["z"] = z
    return fit


def _summary(fit: Dict[str, Any], *, tol: float, lam: float, penalty: str, standardize: str, bands: List[str],
             alpha: float) -> Dict[str, Any]:
    """The JSON record of a fit (``attrs["imad"]``)."""
    n = len(bands)
    rho = np.asarray(fit["rho_history"])
    return {
        "bands": bands,
        "rho": rho.tolist(),
        "max_delta": [None if not np.isfinite(v) else float(v) for v in fit["max_delta"]],
        "rho_final": np.asarray(fit["rho"]).tolist(),
        "iterations": int(fit["iterations"]),
        "converged": bool(fit["converged"]),
        "tol": tol, "lam": lam, "penalty": penalty, "standardize": standardize, "alpha": alpha,
        "n_stats_pixels": int(fit["n_pixels"]),
        "stride": int(fit["stride"]),
        "mean": np.asarray(fit["mean"]).tolist(),
        "a": np.asarray(fit["a"]).tolist(), "b": np.asarray(fit["b"]).tolist(),
        "sd": np.asarray(fit["sd"]).tolist(),
        "loadings_x": np.asarray(fit["loadings_x"]).tolist(),
        "loadings_y": np.asarray(fit["loadings_y"]).tolist(),
        "mad_order": "MAD1 is the pair of the smallest canonical correlation (the largest variance)",
        "n_bands": n,
    }


def _block(xb: np.ndarray, yb: np.ndarray, fit: Dict[str, Any], missing: Sequence[float], canonical: bool,
           alpha: float, threads: int = 1) -> np.ndarray:
    """``(band, y, x)`` of the two dates -> ``(MAD1..N, chi2, p_nochange, change[, U1..UN, V1..VN])`` float32, by the
    compiled core."""
    from . import _core

    n = xb.shape[0]
    xb, yb = ch.kernel_array(xb), ch.kernel_array(yb)
    if xb.dtype != yb.dtype:
        xb, yb = xb.astype(np.float32), yb.astype(np.float32)
    mad, chi2, pnc, cv = _core.imad.apply(xb, yb, fit["mean"], fit["a"], fit["b"], fit["sd"],
                                          [float(v) for v in missing], bool(canonical), int(threads))
    change = np.where(np.isnan(pnc), np.float32(np.nan), (pnc < alpha).astype(np.float32))
    parts = [mad, chi2, pnc, change] + ([cv] if cv is not None else [])
    return np.concatenate(parts, axis=0)


def _block_numpy(xb: np.ndarray, yb: np.ndarray, fit: Dict[str, Any], missing: Sequence[float], canonical: bool,
                 alpha: float, threads: int = 1) -> np.ndarray:
    """What ``_block`` computes, in numpy (and scipy's chi-square): the reference of the tests."""
    from scipy.special import gammaincc

    n = xb.shape[0]
    xf = xb.reshape(n, -1).astype(np.float64)
    yf = yb.reshape(n, -1).astype(np.float64)
    ok = np.isfinite(xf).all(0) & np.isfinite(yf).all(0)
    for m in missing:
        ok &= (xf != m).all(0) & (yf != m).all(0)
    mean, a, b, sd = (np.asarray(fit[k], dtype=np.float64) for k in ("mean", "a", "b", "sd"))
    u = a.T @ (xf - mean[:n, None])          # canonical variates, descending correlation
    v = b.T @ (yf - mean[n:, None])
    mad = (u - v)[::-1]
    chi2 = ((mad / sd[:, None]) ** 2).sum(0)
    pnc = gammaincc(n / 2.0, chi2 / 2.0)
    change = (pnc < alpha).astype(np.float64)
    planes = [mad, chi2[None], pnc[None], change[None]] + ([u, v] if canonical else [])
    out = np.concatenate(planes, axis=0)
    out[:, ~ok] = np.nan
    return out.astype(np.float32).reshape((out.shape[0],) + xb.shape[1:])


def _apply(x: xr.DataArray, y: xr.DataArray, fit: Dict[str, Any], missing: Sequence[float], canonical: bool,
           alpha: float, n_jobs: int):
    """The planes of the final transformation over the whole image: an array, or a dask array if the dates are
    lazy (computed block by block with the fixed transformation)."""
    nout = x.sizes["band"] + 3 + (2 * x.sizes["band"] if canonical else 0)
    if x.chunks is None and y.chunks is None:
        return _block(x.values, y.values, fit, missing, canonical, alpha, n_jobs)
    import dask.array as da

    xa = x.data.rechunk({0: -1})
    ya = y.data.rechunk({0: -1, 1: xa.chunks[1], 2: xa.chunks[2]})
    return da.map_blocks(_block, xa, ya, fit={k: fit[k] for k in ("mean", "a", "b", "sd")}, missing=list(missing),
                         canonical=canonical, alpha=alpha, threads=1, dtype=np.float32,
                         chunks=((nout,),) + xa.chunks[1:], meta=np.empty((0, 0, 0), np.float32))


def imad(
    data: Any,
    after: Any = None,
    *,
    dates: Sequence[Any] = (0, -1),
    bands: Optional[Sequence[Any]] = None,
    max_iter: int = 50,
    tol: float = 1e-3,
    lam: float = 0.0,
    penalty: str = "ridge",
    standardize: str = "weighted",
    alpha: float = 0.01,
    canonical: bool = False,
    sample: Optional[int] = 1_000_000,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.DataArray:
    """The iteratively re-weighted multivariate alteration detection (iMAD, IR-MAD) of two dates.

    The canonical correlation analysis of the bands of the two dates finds the linear combinations of each that
    agree most; their differences, the MAD variates, are uncorrelated change images ordered by variance (MAD1 the
    largest, the pair of the smallest correlation; Canty's scripts number them the other way). A pixel's sum of the
    squared standardized MAD variates, ``chi2``, is approximately chi-square distributed with one degree of freedom
    per band where nothing changed, and ``p_nochange`` is its upper tail. Each iteration weights every pixel by the
    ``p_nochange`` of the last one in the means and covariances, until the canonical correlations stop changing
    (Nielsen 2007; Canty and Nielsen 2008). The result does not depend on gains, offsets or any linear transform of
    either date (``lam = 0``), so the dates need no radiometric normalization first.

    Parameters
    ----------
    data
        A cube ``(time, band, y, x)`` with `dates` picking the two dates, a single date ``(band, y, x)`` (with
        `after` the second), or anything ``zeit.load_raster`` reads. Both dates must have the same bands, in the
        same order, on the same grid.
    after
        The second date when `data` is the first (``(band, y, x)``, or anything ``load_raster`` reads).
    dates
        The two dates of a cube with several: indices (negative from the end), labels of its time coordinate or
        date strings (the nearest date). Default the first and the last.
    bands
        The bands to use (names or indices); default all. The statistics are those of the pixels where every band
        of both dates is valid.
    max_iter
        At most this many iterations; 1 is the original, not iterated MAD of Nielsen et al. (1998).
    tol
        Stops when no canonical correlation changed by more than this since the previous iteration (``0``: always
        ``max_iter`` iterations). The default, 1e-3, is where Canty's ``iMad.py`` stops (found by running it on the
        same data: the iteration counts and the correlations agree); the correlations still creep up by about 1e-5 per
        iteration after that.
    lam
        The regularization of Nielsen (2007) for many correlated bands (hyperspectral): ``lam`` times a penalty
        matrix is added to the covariance matrices of the constraints. ``0`` (default) is none. It is in the units of
        the variances of the bands and breaks the invariance to linear transforms, so scale the bands alike first.
        With ``lam = 0`` collinear or constant bands are an error naming the date.
    penalty
        ``"ridge"`` (the length of the coefficient vectors) or ``"curvature"`` (the second differences of the
        coefficients along the bands, which must be in order of wavelength).
    standardize
        ``"weighted"`` (default, Canty and Nielsen 2008, and what Canty's scripts do): the MAD variates are divided
        by ``sqrt(2 (1 - rho))``, their standard deviation under the current weights. ``"all"`` (Nielsen 2007):
        by their unweighted standard deviation over all the pixels. The weighted one is a little anti-conservative
        for the p-values on data with no change at all, because the weighting shrinks the estimated variance;
        ``"all"`` is calibrated there but reweights less when much has changed.
    alpha
        The significance of the ``change`` band: ``p_nochange < alpha``.
    canonical
        Also return the canonical variates ``U1..UN`` (date 1) and ``V1..VN`` (date 2), in order of descending
        correlation.
    sample
        The statistics are computed on a lattice of this many valid pixels (every s-th row and column from the
        first, deterministic); ``None``: every pixel. The transformation is then applied to the whole image.
    nodata
        Missing values besides NaN: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or
        ``None``. A pixel with a band missing at either date is missing in every output.
    chunks
        For inputs read from disk: ``None`` in memory, ``"auto"`` or a dict lazily.
    n_jobs
        Threads of the compiled core (``-1``: the processors zeit may use). A dask cube is parallel by its chunks.

    Returns
    -------
    xarray.DataArray
        ``(band, y, x)`` float32 with the bands ``MAD1..MADN``, ``chi2``, ``p_nochange``, ``change`` (1 or 0) and,
        with ``canonical``, ``U1..UN``, ``V1..VN``; the coordinates and georeferencing of the input. ``attrs["imad"]``
        is a JSON record: the correlations of each iteration (``rho``), ``iterations``, ``converged``, the
        coefficient matrices ``a`` and ``b`` (columns are canonical pairs, descending correlation), the means, the
        standard deviations and ``loadings_x``, ``loadings_y`` (the correlation of each MAD variate with each band
        of each date). ``zeit.imad_table`` turns it into a table.

    Examples
    --------
    >>> change = zeit.imad(cube, dates=("2021-07-15", "2022-07-20"), bands=["blue", "green", "red", "nir08"])
    >>> change.sel(band="change")
    """
    from ._threads import resolve

    x, y = ch.two_dates(data, after, dates, bands, "zeit.imad", chunks)
    missing = ch.missing_values(x, nodata)
    names = [str(b) for b in x.band.values]
    fit = _fit(x, y, missing, max_iter=max_iter, tol=tol, lam=lam, penalty=penalty, standardize=standardize,
               sample=sample, n_jobs=resolve(n_jobs), who="zeit.imad")
    n = len(names)
    out_names = [f"MAD{i + 1}" for i in range(n)] + ["chi2", "p_nochange", "change"]
    if canonical:
        out_names += [f"U{i + 1}" for i in range(n)] + [f"V{i + 1}" for i in range(n)]
    values = _apply(x, y, fit, missing, canonical, alpha, resolve(n_jobs))
    record = _summary(fit, tol=tol, lam=lam, penalty=penalty, standardize=standardize, bands=names, alpha=alpha)
    return ch.result_array(values, out_names, x, "imad", record, "imad")


def imad_table(result: xr.DataArray) -> pd.DataFrame:
    """The canonical correlations of every iteration of a ``zeit.imad`` result: one row per iteration, the columns
    ``rho_1..rho_N`` (descending) and ``max_delta`` (the largest change since the iteration before)."""
    record = json.loads(result.attrs["imad"])
    rho = np.asarray(record["rho"])
    table = pd.DataFrame(rho, columns=[f"rho_{i + 1}" for i in range(rho.shape[1])])
    table.insert(0, "iteration", np.arange(1, len(rho) + 1))
    table["max_delta"] = [np.nan if v is None else v for v in record["max_delta"]]
    return table


def radcal(
    data: Any,
    after: Any = None,
    *,
    reference: Any = 0,
    targets: Optional[Sequence[Any]] = None,
    threshold: float = 0.95,
    holdout: float = 1 / 3,
    bands: Optional[Sequence[Any]] = None,
    max_iter: int = 50,
    tol: float = 1e-3,
    lam: float = 0.0,
    sample: Optional[int] = 1_000_000,
    nodata: Any = "auto",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.DataArray:
    """Relative radiometric normalization of dates to a reference date by the invariant pixels of iMAD.

    For each target date, ``zeit.imad`` against the reference finds the pixels that did not change (those whose
    probability of no change, ``p_nochange``, exceeds `threshold`); an orthogonal regression of each band of the
    reference on the same band of the target, fitted on two thirds of those pixels, gives its gain and offset (the
    errors of both dates enter, unlike least squares), and the other third tests the result: a paired t-test of the
    means and an F-test of the variances of the reference and the normalized target (Canty, Nielsen and Schmidt
    2004; Canty and Nielsen 2008). The normalized target is ``a + b * target`` for every pixel.

    Parameters
    ----------
    data
        A cube ``(time, band, y, x)`` whose dates are normalized to `reference`, or the reference date
        ``(band, y, x)`` when `after` is the target (the result is then the normalized target ``(band, y, x)``).
    after
        The target date, when `data` is the reference.
    reference
        The reference date of a cube: an index, a label or a date string (the nearest). It is copied unchanged.
    targets
        The dates to normalize (default every other date); the dates not named are copied unchanged.
    threshold
        The invariant pixels have ``p_nochange`` above this (default 0.95; the pixels expected to be unchanged
        are then about 5 % of those that did not change, so give a large `sample` or a large image).
    holdout
        The share of the invariant pixels kept for the tests: every ``round(1 / holdout)``-th in raster order
        (default 1/3). ``0``: all are used for the fit and there are no tests. At least 30 pixels are needed for the
        fit.
    bands
        The bands to use for the iMAD and normalize (names or indices); the others are not in the result.
    max_iter, tol, lam, sample, nodata, chunks, n_jobs
        As in :func:`imad`.

    Returns
    -------
    xarray.DataArray
        float32, the shape of the cube (or ``(band, y, x)``), with the normalized dates; a missing pixel is NaN.
        ``attrs["radcal"]`` is a JSON record: ``fits``, one row per target date and band (``a``, ``se_a``, ``b``,
        ``se_b``, ``r``, ``rmse``, ``n_train``, ``n_test``, the means of target, reference and normalized target on
        the hold-out pixels, ``t``, ``p_t``, the variances, ``F`` and ``p_F``), and ``imad``, the summary of each
        pair's iMAD. ``zeit.radcal_table`` turns the fits into a table.

    Examples
    --------
    >>> normalized = zeit.radcal(cube, reference="2021-07-15")
    >>> zeit.radcal_table(normalized)
    """
    from . import _core
    from ._threads import resolve

    who = "zeit.radcal"
    threads = resolve(n_jobs)
    if not 0 <= float(holdout) < 1:
        raise ValueError(f"holdout= must be in [0, 1), got {holdout!r}")
    if not 0.5 <= float(threshold) < 1:
        raise ValueError(f"threshold= must be in [0.5, 1), got {threshold!r}")
    every = 0 if float(holdout) == 0 else max(2, int(round(1.0 / float(holdout))))
    pair = after is not None
    cube = ch.to_cube(data, who, chunks)
    if pair:
        ref, tgt = ch.two_dates(cube, after, (0, 0), bands, who, chunks)
        series = [tgt]
        times: List[Any] = [None]
        index_of_ref = None
    else:
        if "time" not in cube.dims:
            raise ValueError("zeit.radcal needs a cube with a time dimension, or the reference as data and the "
                             "target as after=")
        n_t = cube.sizes["time"]
        ref_index = ch._pick_date(cube.assign_coords(_i=("time", np.arange(n_t))), reference, who)["_i"].item() \
            if not isinstance(reference, (int, np.integer)) else int(reference) % n_t
        picks = list(range(n_t)) if targets is None else []
        if targets is not None:
            for t in targets:
                picks.append(int(ch._pick_date(cube.assign_coords(_i=("time", np.arange(n_t))), t, who)["_i"].item())
                             if not isinstance(t, (int, np.integer)) else int(t) % n_t)
        picks = [k for k in picks if k != ref_index]
        ref, _ = ch.two_dates(cube, None, (ref_index, ref_index), bands, who, chunks)
        series = {k: ch.two_dates(cube, None, (k, k), bands, who, chunks)[0] for k in picks}
        index_of_ref = ref_index
    missing = ch.missing_values(ref, nodata)
    names = [str(b) for b in ref.band.values]
    n = len(names)
    fits: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    outputs: Dict[int, Any] = {}
    items = [(0, series[0])] if pair else list(series.items())
    for k, target in items:
        fit = _fit(ref, target, missing, max_iter=max_iter, tol=tol, lam=lam, penalty="ridge", standardize="weighted",
                   sample=sample, n_jobs=threads, who=who)
        try:
            rows, n_invariant = _core.imad.radcal_fit(fit["z"], fit["mean"], fit["a"], fit["b"], fit["sd"],
                                                      float(threshold), every, threads)
        except ValueError as error:
            raise ValueError(f"{who}: {error}") from None
        label = None if pair else _label(cube, k)
        for j, row in enumerate(rows):
            fits.append({"date": label, "band": names[j], **row})
        summaries.append({"date": label, "n_invariant": int(n_invariant), **_summary(
            fit, tol=tol, lam=lam, penalty="ridge", standardize="weighted", bands=names, alpha=0.01)})
        gain = np.array([r["b"] for r in rows])[:, None, None]
        offset = np.array([r["a"] for r in rows])[:, None, None]
        values = target.astype(np.float64)
        for m in missing:
            values = values.where(values != m)
        outputs[k] = (values * gain + offset).astype(np.float32)
    record = {"reference": None if pair else _label(cube, index_of_ref), "threshold": float(threshold),
              "holdout_every": every, "fits": fits, "imad": summaries}
    if pair:
        result = outputs[0].transpose("band", "y", "x")
        result.attrs = {k: v for k, v in ref.attrs.items() if k not in ("scale_factor", "add_offset")}
        result.attrs["radcal"] = ch.dumps(record)
        if ref.rio.crs is not None:
            result = result.rio.write_crs(ref.rio.crs)
        return result
    sub = cube if bands is None else cube.isel(band=[list(cube.band.values.astype(str)).index(b) for b in names])
    planes = []
    for k in range(cube.sizes["time"]):
        if k in outputs:
            planes.append(outputs[k])
        else:
            keep = sub.isel(time=k).astype(np.float32)
            for m in missing:
                keep = keep.where(keep != m)
            planes.append(keep)
    result = xr.concat([p.drop_vars("time", errors="ignore") for p in planes], dim="time")
    result = result.assign_coords(time=cube.time.values).transpose(*sub.dims)
    result.attrs = {k: v for k, v in cube.attrs.items() if k not in ("scale_factor", "add_offset")}
    result.attrs["radcal"] = ch.dumps(record)
    result.name = cube.name
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    return result


def _label(cube: xr.DataArray, k: int) -> Any:
    value = cube.time.values[k]
    return str(value)[:10] if np.issubdtype(np.asarray(value).dtype, np.datetime64) else str(value)


def radcal_table(result: xr.DataArray) -> pd.DataFrame:
    """The fits of a ``zeit.radcal`` result: one row per target date and band, with the gain ``b`` and offset ``a``
    (``normalized = a + b * target``) and their standard errors, the correlation ``r`` and ``rmse`` of the fit on the
    training pixels, and the tests on the hold-out pixels (``t``, ``p_t``, ``F``, ``p_F``) with the means and
    variances they compare."""
    record = json.loads(result.attrs["radcal"])
    return pd.DataFrame(record["fits"])
