"""The Minimum Noise Fraction (MNF) transform: ``zeit.mnf``, ``zeit.mnf_fit``, ``zeit.inverse_mnf``,
``zeit.mnf_denoise`` and ``zeit.MNFTransform``.

The MNF (Green, Berman, Switzer and Craig 1988, IEEE TGRS 26:65-74; the same components as the noise-adjusted
principal components of Lee, Woodyatt and Berman 1990, IEEE TGRS 28:295-304) orders the components of a multiband
image by signal-to-noise ratio instead of variance. The noise covariance is estimated from the differences between
neighbouring pixels (half the covariance of ``z(p) - z(q)``: the signal is smooth, so the difference is noise
twice), the noise is whitened and a principal components rotation follows. The eigenvalue of a component is
``1 + SNR``. Unlike a principal components analysis the result does not depend on the units of the bands.
"""

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

#: The noise estimators (``noise=``): the pair of pixels that is differenced.
ESTIMATORS = {
    "diff_h": ((0, 1), "h"),
    "diff_v": ((1, 0), "v"),
    "diff_diag": ((1, 1), "diag"),
    "diff_antidiag": ((1, -1), "antidiag"),
}
#: Names of the directions of an estimator, in the order of the counts of pairs.
_DIRECTIONS = {"diff_h": ["h"], "diff_v": ["v"], "diff_hv": ["h", "v"], "diff_diag": ["diag"],
               "diff_antidiag": ["antidiag"], "given": []}
_KERNEL_DTYPES = tuple(np.dtype(t) for t in (np.int16, np.uint16, np.int32, np.float32, np.float64))
#: Relative floor of the noise eigenvalues and the eigenvalue under which a component has no variance.
EPS_NOISE = 1e-9
DEGENERATE = 1e-6


def _version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("zeit-cdts")
    except Exception:  # pragma: no cover - not installed
        return "unknown"


@dataclass
class MNFTransform:
    """A fitted MNF: the statistics, the matrices and the table of eigenvalues, usable on another image or date.

    ``forward`` ``(components, bands)`` has the loadings ``a_i`` in its rows (zero at the bands left out), so that
    the component ``i`` of a pixel is ``a_i . (z - mean)``; ``inverse`` ``(bands, components)`` has the patterns
    ``p_i`` in its columns, ``z - mean = sum p_i y_i``. Both are in the ``"noise"`` scale: noise of unit variance
    in every component, a variance equal to the eigenvalue. ``correlation[j, i]`` is the correlation of band ``j``
    with component ``i``. Bands that are constant, or have no valid value, are left out (``dropped``).
    """

    bands: List[str]
    kept: np.ndarray
    dropped: List[Dict[str, Any]]
    mean: np.ndarray
    cov: np.ndarray
    noise_cov: np.ndarray
    eigenvalues: np.ndarray
    forward: np.ndarray
    inverse: np.ndarray
    correlation: np.ndarray
    estimator: str = "given"
    noise_region: str = "whole"
    n_pixels: int = 0
    n_pairs: Dict[str, int] = field(default_factory=dict)
    sample_fraction: float = 1.0
    seed: int = 0
    noise_floored: int = 0
    floored: Optional[np.ndarray] = None
    per_date: bool = False
    date: Optional[str] = None

    # -- derived -----------------------------------------------------------------------------------------------

    @property
    def snr(self) -> np.ndarray:
        """The signal-to-noise ratio of each component: ``eigenvalue - 1``."""
        return self.eigenvalues - 1.0

    def n_with_snr(self, snr: float) -> int:
        """The number of components whose SNR is at least `snr`."""
        return int(np.sum(self.snr >= snr))

    def table(self):
        """One row a component: the eigenvalue, the SNR (and in dB), the noise fraction, the share of the signal
        (the SNR above zero), the lag-1 autocorrelation ``1 - 1/eigenvalue`` of the direction of the estimator, and a
        ``flag`` (``degenerate``: no variance, a band that combines others exactly; ``floored_noise``: mostly in a
        direction where the noise covariance is singular)."""
        import pandas as pd

        lam = np.asarray(self.eigenvalues, dtype=float)
        snr = lam - 1.0
        positive = np.maximum(snr, 0.0)
        total = positive.sum()
        share = positive / total if total > 0 else np.zeros_like(positive)
        with np.errstate(divide="ignore", invalid="ignore"):
            db = np.where(snr > 0, 10 * np.log10(np.where(snr > 0, snr, 1.0)), np.nan)
            frac = 1.0 / lam
            maf = 1.0 - frac
        if self.estimator == "given":
            maf = np.full_like(lam, np.nan)
        floored = np.zeros(len(lam), bool) if self.floored is None else np.asarray(self.floored, bool)
        flag = ["degenerate" if lam[i] < DEGENERATE else "floored_noise" if floored[i] else ""
                for i in range(len(lam))]
        return pd.DataFrame({
            "component": np.arange(1, len(lam) + 1), "eigenvalue": lam, "snr": snr, "snr_db": db,
            "noise_fraction": frac, "signal_share": share, "cumulative_signal_share": np.cumsum(share),
            "maf_autocorrelation": maf, "flag": flag})

    def to_csv(self, path) -> None:
        """Writes the table of eigenvalues as CSV."""
        self.table().to_csv(path, index=False)

    # -- JSON --------------------------------------------------------------------------------------------------

    def to_json(self, path=None) -> str:
        """The transform as JSON (``"format": "zeit-mnf"``); written to `path` when given. Floats are written in
        their shortest round-trip form, so ``from_json`` gives the same bits."""
        names = _DIRECTIONS.get(self.estimator, [])
        doc = {
            "format": "zeit-mnf", "version": 1, "zeit": _version(),
            "bands": list(self.bands), "kept": [int(k) for k in self.kept], "dropped": self.dropped,
            "estimator": self.estimator, "noise_region": self.noise_region,
            "sample_fraction": float(self.sample_fraction), "seed": int(self.seed),
            "per_date": bool(self.per_date), "date": self.date,
            "n_pixels": int(self.n_pixels), "n_pairs": {k: int(v) for k, v in self.n_pairs.items()} if names else {},
            "mean": self.mean.tolist(), "cov": self.cov.tolist(), "noise_cov": self.noise_cov.tolist(),
            "eigenvalues": self.eigenvalues.tolist(), "forward": self.forward.tolist(),
            "inverse": self.inverse.tolist(), "correlation": self.correlation.tolist(),
            "eps_noise": EPS_NOISE, "noise_floored": int(self.noise_floored),
            "floored": [bool(f) for f in (self.floored if self.floored is not None else [])],
        }
        text = json.dumps(doc)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path_or_text) -> "MNFTransform":
        """A transform written by ``to_json``: a path, or the JSON text itself."""
        text = str(path_or_text)
        if not text.lstrip().startswith("{"):
            text = Path(path_or_text).read_text(encoding="utf-8")
        d = json.loads(text)
        if d.get("format") != "zeit-mnf":
            raise ValueError("not a zeit MNF transform (format != 'zeit-mnf')")
        if d.get("version") != 1:
            raise ValueError(f"MNF transform version {d.get('version')!r} is not known to this zeit")
        nb = len(d["bands"])
        m = len(d["eigenvalues"])
        return cls(
            bands=list(d["bands"]), kept=np.asarray(d["kept"], dtype=np.int64), dropped=list(d["dropped"]),
            mean=np.asarray(d["mean"], dtype=float), cov=np.asarray(d["cov"], dtype=float).reshape(nb, nb),
            noise_cov=np.asarray(d["noise_cov"], dtype=float).reshape(nb, nb),
            eigenvalues=np.asarray(d["eigenvalues"], dtype=float),
            forward=np.asarray(d["forward"], dtype=float).reshape(m, nb),
            inverse=np.asarray(d["inverse"], dtype=float).reshape(nb, m),
            correlation=np.asarray(d["correlation"], dtype=float).reshape(nb, m),
            estimator=d["estimator"], noise_region=d["noise_region"], n_pixels=d["n_pixels"],
            n_pairs=dict(d["n_pairs"]), sample_fraction=d["sample_fraction"], seed=d["seed"],
            noise_floored=d["noise_floored"], floored=np.asarray(d.get("floored", [False] * m), dtype=bool),
            per_date=d["per_date"], date=d["date"])

    @classmethod
    def from_covariances(cls, mean, cov, noise_cov, bands: Optional[Sequence[str]] = None) -> "MNFTransform":
        """The transform of given statistics: the mean, the total covariance and the noise covariance ``(bands,
        bands)`` (from a dark frame, another image, or another program's statistics)."""
        from . import _core

        mean = np.ascontiguousarray(mean, dtype=float)
        core = _core.mnf.fit_cov(mean, np.ascontiguousarray(cov, dtype=float),
                                 np.ascontiguousarray(noise_cov, dtype=float))
        names = [f"B{j + 1}" for j in range(len(mean))] if bands is None else [str(b) for b in bands]
        if len(names) != len(mean):
            raise ValueError(f"bands= has {len(names)} names for {len(mean)} bands")
        return _from_core(core, names, "given", "given", 1.0, 0)

    # -- use ---------------------------------------------------------------------------------------------------

    def _core(self):
        from . import _core

        empty = [d["band"] for d in self.dropped if d["reason"] == "empty"]
        return _core.mnf.Transform.from_parts(
            len(self.bands), [int(k) for k in self.kept], empty, np.ascontiguousarray(self.mean, dtype=float),
            np.ascontiguousarray(self.forward, dtype=float), np.ascontiguousarray(self.inverse, dtype=float),
            np.ascontiguousarray(self.eigenvalues, dtype=float))

    def _choose(self, k: Optional[int], snr_min: Optional[float], default: Optional[int]) -> int:
        m = len(self.eigenvalues)
        if k is not None and snr_min is not None:
            raise ValueError("give k or snr_min, not both")
        if k is None and snr_min is None:
            if default is None:
                raise ValueError("give the components to keep: k=<number> or snr_min=<SNR>")
            return default
        if snr_min is not None:
            if snr_min < -1:
                raise ValueError("snr_min must be at least -1 (the SNR of a component cannot be lower)")
            return self.n_with_snr(float(snr_min))
        if int(k) != k or not 0 <= int(k) <= m:
            raise ValueError(f"k={k}: the transform has {m} components, k must be an integer from 0 to {m}")
        return int(k)

    def apply(self, data: Any, n_components: Optional[int] = None, *, scale: str = "noise", mask: Any = None,
              nodata: Union[float, str, None] = "auto", bands: Optional[Sequence[Any]] = None,
              chunks: Any = None, n_jobs: int = -1) -> xr.DataArray:
        """The components of `data` ``(band, y, x)`` or ``(time, band, y, x)`` by this transform (the statistics are
        this transform's, not the data's): float32 ``mnf1 ... mnfK``, NaN where a band is missing. The bands are
        matched to the transform's by name, else by count (with a warning)."""
        return _apply(self, data, n_components, scale, mask, nodata, bands, chunks, n_jobs, None)

    def inverse_transform(self, components: Any, k: Optional[int] = None, *, snr_min: Optional[float] = None,
                          scale: str = "noise", chunks: Any = None, n_jobs: int = -1) -> xr.DataArray:
        """The bands rebuilt from the first `k` of the `components` (all of them by default; or those with an SNR of
        at least `snr_min`): the others are set to zero, their mean, so the noise they hold goes."""
        return inverse_mnf(components, self, k, snr_min=snr_min, scale=scale, chunks=chunks, n_jobs=n_jobs)

    def denoise(self, data: Any, k: Optional[int] = None, *, snr_min: Optional[float] = None,
                mask: Any = None, nodata: Union[float, str, None] = "auto", bands: Optional[Sequence[Any]] = None,
                chunks: Any = None, n_jobs: int = -1) -> xr.DataArray:
        """`data` rebuilt from the first `k` components (or those with an SNR of at least `snr_min`) of this
        transform, without keeping the components."""
        return _denoise(self, data, k, snr_min, mask, nodata, bands, chunks, n_jobs, None)


def _from_core(core, names: List[str], estimator: str, region: str, fraction: float, seed: int,
               per_date: bool = False, date: Optional[str] = None) -> MNFTransform:
    d = core.to_dict()
    directions = _DIRECTIONS[estimator]
    pairs = dict(zip(directions, (int(v) for v in d["n_pairs"])))
    return MNFTransform(
        bands=list(names), kept=np.asarray(d["kept"], dtype=np.int64),
        dropped=[{"band": int(j), "reason": r} for j, r in d["dropped"]], mean=d["mean"], cov=d["cov"],
        noise_cov=d["noise_cov"], eigenvalues=d["eigenvalues"], forward=d["forward"], inverse=d["inverse"],
        correlation=d["correlation"], estimator=estimator, noise_region=region, n_pixels=int(d["n_pixels"]),
        n_pairs=pairs, sample_fraction=float(fraction), seed=int(seed), noise_floored=int(d["noise_floored"]),
        floored=np.asarray(d["floored"], dtype=bool), per_date=per_date, date=date)


# ---------------------------------------------------------------------------------------------------------------
# Reading the input

def _cube(data: Any, chunks: Any, name: str) -> xr.DataArray:
    from ._embeddings import refuse
    from ._load import load_raster

    refuse(data, f"zeit.{name}")
    if isinstance(data, xr.DataArray):
        cube = data
    elif isinstance(data, np.ndarray):
        if data.ndim not in (3, 4):
            raise ValueError(f"{name} needs (band, y, x) or (time, band, y, x), got {data.ndim} dimensions")
        dims = ("band", "y", "x") if data.ndim == 3 else ("time", "band", "y", "x")
        cube = xr.DataArray(data, dims=dims, coords={"band": [f"B{j + 1}" for j in range(data.shape[-3])]})
    else:
        cube = load_raster(data, chunks=chunks)
    if "band" not in cube.dims or not {"y", "x"} <= set(cube.dims):
        raise ValueError(f"{name} needs a cube with band, y and x dimensions, (band, y, x) or (time, band, y, x)")
    if set(cube.dims) - {"time", "band", "y", "x"}:
        raise ValueError(f"{name} takes (band, y, x) or (time, band, y, x), got {cube.dims}")
    if "band" not in cube.coords:
        cube = cube.assign_coords(band=[f"B{j + 1}" for j in range(cube.sizes["band"])])
    if cube.attrs.get("mnf"):
        raise ValueError(f"the cube is already an MNF ({cube.attrs['mnf']}): give the bands, not the components")
    return cube


def _pick_bands(cube: xr.DataArray, bands: Optional[Sequence[Any]]) -> xr.DataArray:
    if bands is None:
        return cube
    names = [str(b) for b in cube.band.values]
    pick = []
    for b in bands:
        if isinstance(b, (int, np.integer)) and not isinstance(b, bool):
            if not 0 <= int(b) < len(names):
                raise ValueError(f"bands=: index {b} is out of the {len(names)} bands")
            pick.append(int(b))
        else:
            if str(b) not in names:
                raise ValueError(f"bands=: {b!r} is not one of the bands {names}")
            pick.append(names.index(str(b)))
    if len(set(pick)) != len(pick):
        raise ValueError("bands= names a band twice")
    return cube.isel(band=pick)


def _missing(cube: xr.DataArray, nodata: Any) -> List[float]:
    from ._tasseled import _missing_values

    return [float(v) for v in _missing_values(cube, nodata)]


def _mask_array(mask: Any, cube: xr.DataArray, what: str) -> Optional[np.ndarray]:
    """A boolean ``(y, x)`` mask as uint8 (nonzero = use), or None."""
    if mask is None:
        return None
    arr = mask.values if isinstance(mask, xr.DataArray) else np.asarray(mask)
    shape = (cube.sizes["y"], cube.sizes["x"])
    if arr.shape != shape:
        raise ValueError(f"{what} must be (y, x) = {shape}, got {arr.shape}")
    use = arr != 0
    if arr.dtype.kind == "f":
        use &= ~np.isnan(arr)
    return np.ascontiguousarray(use, dtype=np.uint8)


def _region_array(region: Any, cube: xr.DataArray) -> Tuple[Optional[np.ndarray], str]:
    """The noise region as ``(mask, kind)``: a boolean raster or a pixel window ``(x0, y0, x1, y1)``, half-open."""
    if region is None:
        return None, "whole"
    if isinstance(region, (tuple, list)) and len(region) == 4 and all(np.isscalar(v) for v in region):
        x0, y0, x1, y1 = (int(v) for v in region)
        ny, nx = cube.sizes["y"], cube.sizes["x"]
        if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
            raise ValueError(f"noise_region={tuple(region)}: a window (x0, y0, x1, y1) inside the {nx} x {ny} pixels, "
                             "x0 < x1 and y0 < y1")
        m = np.zeros((ny, nx), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        return m, "window"
    return _mask_array(region, cube, "noise_region"), "mask"


def _noise_args(noise: str, noise_cov: Any, nb: int) -> Tuple[str, Optional[np.ndarray]]:
    if noise not in _DIRECTIONS:
        raise ValueError(f"noise= {noise!r}: use {', '.join(repr(k) for k in _DIRECTIONS)}")
    if noise == "given":
        if noise_cov is None:
            raise ValueError("noise='given' needs noise_cov= (bands x bands)")
        nc = np.ascontiguousarray(noise_cov, dtype=float)
        if nc.shape != (nb, nb):
            raise ValueError(f"noise_cov must be ({nb}, {nb}), got {nc.shape}")
        return noise, nc
    if noise_cov is not None:
        raise ValueError("noise_cov is for noise='given'")
    return noise, None


def _in_memory(arr: np.ndarray) -> np.ndarray:
    if arr.dtype not in _KERNEL_DTYPES:
        arr = arr.astype(np.float64)
    return np.ascontiguousarray(arr)


def _rows_per_block(nx: int, nb: int) -> int:
    return max(2, (8 << 20) // max(nx * nb * 8, 1))


def _stats_dask(arr, nt: int, noise: str, noise_cov, mask, region, missing, fraction, seed, date0: int):
    """The statistics of a dask cube read in blocks of rows (each with its halo row) in order: the same arithmetic
    as the in-memory path, with one block in memory at a time."""
    from . import _core

    _, nb, ny, nx = arr.shape

    def run(active):
        st = _core.mnf.Stats(nb, noise, noise_cov)
        step = _rows_per_block(nx, nb)
        for t in range(nt):
            for r0 in range(0, ny, step):
                rows = min(step, ny - r0)
                last = min(r0 + rows + 1, ny)
                blk = np.asarray(arr[t, :, r0:last].compute(), dtype=np.float64)
                halo = np.ascontiguousarray(blk[:, rows]) if last > r0 + rows else None
                st.add_block(np.ascontiguousarray(blk[:, :rows]), halo, t, r0, ny, mask, region, missing, fraction,
                             seed, date0, active)
        return st

    st = run(None)
    valid = np.asarray(st.band_valid)
    if st.n == 0 and (valid == 0).any() and (valid > 0).any():
        st2 = run((valid > 0).astype(np.uint8).tolist())
        return st2
    return st


def _fit_one(arr, nt, names, noise, noise_cov, mask, region, region_kind, missing, fraction, seed, date0, n_jobs,
             dask: bool, per_date: bool = False, date: Optional[str] = None) -> MNFTransform:
    from . import _core

    nb = arr.shape[1]
    if dask:
        st = _stats_dask(arr, nt, noise, noise_cov, mask, region, missing, fraction, seed, date0)
    else:
        st = _core.mnf.stats_batch(_in_memory(arr), noise, noise_cov, mask, region, missing, fraction, seed, date0,
                                   n_jobs)
    core = st.fit()
    tr = _from_core(core, names, noise, region_kind, fraction, seed, per_date, date)
    used = nb - sum(1 for d in tr.dropped if d["reason"] == "empty")
    if tr.n_pixels < 10 * used:
        warnings.warn(f"only {tr.n_pixels} valid pixels for {used} bands: the statistics are poor (10 times the "
                      "bands or more is wanted)", stacklevel=4)
    for name_, m in tr.n_pairs.items():
        if m < 10 * used:
            warnings.warn(f"only {m} pairs of neighbouring pixels (direction {name_}) for {used} bands: the noise "
                          "estimate is poor (10 times the bands or more is wanted)", stacklevel=4)
    for reason, what in (("constant", "constant band"), ("empty", "band with no valid value")):
        gone = [str(names[d["band"]]) for d in tr.dropped if d["reason"] == reason]
        if gone:
            shown = ", ".join(repr(g) for g in gone[:5]) + (f" and {len(gone) - 5} more" if len(gone) > 5 else "")
            warnings.warn(f"{len(gone)} {what}{'s' if len(gone) != 1 else ''} left out of the MNF: {shown}",
                          stacklevel=4)
    if tr.noise_floored:
        warnings.warn(f"{tr.noise_floored} directions of the noise covariance are singular (bands with identical or "
                      "no noise) and were given the floor of 1e-9 of the largest; the components in them are not "
                      "reliable", stacklevel=4)
    return tr


def _work(cube: xr.DataArray) -> Tuple[xr.DataArray, bool, Any]:
    has_time = "time" in cube.dims
    order = (("time",) if has_time else ()) + ("band", "y", "x")
    work = cube.transpose(*order)
    arr = work.data if work.chunks is not None else work.values
    if not has_time:
        arr = arr[None]
    return work, has_time, arr


def _date_label(cube: xr.DataArray, t: int) -> Optional[str]:
    if "time" not in cube.dims or "time" not in cube.coords:
        return None
    return str(cube.time.values[t])


def mnf_fit(
    data: Any,
    *,
    noise: str = "diff_hv",
    noise_region: Any = None,
    noise_cov: Any = None,
    mask: Any = None,
    bands: Optional[Sequence[Any]] = None,
    nodata: Union[float, str, None] = "auto",
    sample_fraction: float = 1.0,
    seed: int = 0,
    per_date: bool = False,
    chunks: Any = None,
    n_jobs: int = -1,
) -> Union[MNFTransform, List[MNFTransform]]:
    """Fits the MNF transform of a multiband image.

    The mean and the covariance of the pixels that have every band valid give the total (signal plus noise)
    covariance; the noise covariance is half the covariance of the differences between each pixel and its
    neighbour (pairs with a missing pixel do not count, nor do pairs that cross dates or the edge of the image). The
    components solve ``total a = lambda noise a`` with ``a' noise a = 1``, ordered by decreasing ``lambda = 1 + SNR``;
    the eigenvalues are in ``MNFTransform.table()``. The statistics are gathered in blocks of rows (the cube may be
    larger than the memory when it is a dask array) and merged in a fixed order, so the result does not depend on the
    number of threads. All is in float64.

    Parameters
    ----------
    data
        ``(band, y, x)`` or ``(time, band, y, x)``: a DataArray (in memory or dask), an array, or anything
        ``zeit.load_raster`` reads. A pixel is used when every band is finite and different from the NoData value.
    noise
        ``"diff_hv"`` (default): the right and lower neighbours, pooled (robust to row or column structure);
        ``"diff_h"``, ``"diff_v"``, ``"diff_diag"`` (lower right), ``"diff_antidiag"`` (lower left), or ``"given"``
        with `noise_cov`. Noise that is correlated between neighbours (stripes) counts as signal.
    noise_region
        Where the differences are taken: a boolean ``(y, x)`` array or a pixel window ``(x0, y0, x1, y1)`` (half
        open), both pixels of a pair inside; the signal statistics stay those of the whole image. Pick a homogeneous
        area (water, a uniform field) when the scene has much fine detail.
    noise_cov
        ``(bands, bands)`` noise covariance, for ``noise="given"``.
    mask
        Boolean ``(y, x)``: the pixels used.
    bands
        The bands to use (names or indices); by default all. A band with no valid value is left out
        automatically, and a constant band too (both in ``dropped``).
    nodata
        As ``zeit.tasseled_cap``: ``"auto"`` (the raster's NoData; 0 for integers without one), a number or None.
    sample_fraction, seed
        Use only this fraction of the pixels (and pair anchors) for the statistics, chosen by a hash of the pixel and
        the seed (independent of blocks and threads).
    per_date
        With a time axis: one transform per date (a list) instead of statistics pooled over the dates.
    chunks
        For a path: ``None`` in memory, ``"auto"`` or a dict lazily.
    n_jobs
        Threads (``-1``: as zeit's default).

    Returns
    -------
    MNFTransform or list of MNFTransform
    """
    from ._threads import resolve

    cube = _pick_bands(_cube(data, chunks, "mnf_fit"), bands)
    if cube.sizes["band"] < 2:
        raise ValueError("the MNF needs two bands or more")
    names = [str(b) for b in cube.band.values]
    noise_, nc = _noise_args(noise, noise_cov, len(names))
    mask_ = _mask_array(mask, cube, "mask")
    region, kind = _region_array(noise_region, cube)
    if noise_ == "given" and region is not None:
        raise ValueError("noise_region is for the difference estimators, not noise='given'")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    missing = _missing(cube, nodata)
    work, has_time, arr = _work(cube)
    dask = work.chunks is not None
    nt = arr.shape[0]
    kind = "given" if noise_ == "given" else kind
    jobs = resolve(n_jobs)
    if per_date:
        if not has_time:
            raise ValueError("per_date=True needs a time axis")
        return [_fit_one(arr[t:t + 1], 1, names, noise_, nc, mask_, region, kind, missing, sample_fraction, seed, t,
                         jobs, dask, True, _date_label(cube, t)) for t in range(nt)]
    return _fit_one(arr, nt, names, noise_, nc, mask_, region, kind, missing, sample_fraction, seed, 0, jobs, dask)


def _match_bands(tr: MNFTransform, cube: xr.DataArray) -> xr.DataArray:
    """The cube's bands in the order of the transform's: by name when they are the same names, else by count."""
    have = [str(b) for b in cube.band.values]
    if sorted(have) == sorted(tr.bands) and len(set(have)) == len(have):
        return cube.sel(band=list(tr.bands)) if have != list(tr.bands) else cube
    if len(have) != len(tr.bands):
        raise ValueError(f"the data have {len(have)} bands {have}, the transform {len(tr.bands)} {tr.bands}")
    warnings.warn(f"the bands of the data {have} are not named as those of the transform {tr.bands}: they are "
                  "matched by position", stacklevel=4)
    return cube


def _forward_block(block: np.ndarray, tr: MNFTransform, k_out: int, scale: str, missing: Sequence[float],
                   mask: Optional[np.ndarray], n_jobs: int = 1, block_info: Any = None) -> np.ndarray:
    from . import _core

    sel = mask
    if mask is not None and block_info:
        (_, _), (_, _), (y0, y1), (x0, x1) = block_info[0]["array-location"]
        sel = np.ascontiguousarray(mask[y0:y1, x0:x1])
    return _core.mnf.forward(_in_memory(np.asarray(block)), tr._core(), int(k_out), scale, [float(v) for v in missing], sel,
                             int(n_jobs))


def _denoise_block(block: np.ndarray, tr: MNFTransform, k: int, missing: Sequence[float],
                   mask: Optional[np.ndarray], n_jobs: int = 1, block_info: Any = None) -> np.ndarray:
    from . import _core

    sel = mask
    if mask is not None and block_info:
        (_, _), (_, _), (y0, y1), (x0, x1) = block_info[0]["array-location"]
        sel = np.ascontiguousarray(mask[y0:y1, x0:x1])
    return _core.mnf.denoise(_in_memory(np.asarray(block)), tr._core(), int(k), [float(v) for v in missing], sel,
                             int(n_jobs))


def _inverse_block(block: np.ndarray, tr: MNFTransform, k: int, scale: str, n_jobs: int = 1) -> np.ndarray:
    from . import _core

    return _core.mnf.inverse(np.ascontiguousarray(block, dtype=np.float32), tr._core(), int(k), scale, int(n_jobs))


def _wrap(out, cube: xr.DataArray, work: xr.DataArray, has_time: bool, names: List[str], attrs: Dict[str, Any]
          ) -> xr.DataArray:
    order = work.dims
    if not has_time:
        out = out[0]
    coords = {k: v for k, v in cube.coords.items() if "band" not in v.dims}
    coords["band"] = names
    result = xr.DataArray(out, dims=order, coords=coords, name=cube.name).transpose(*cube.dims)
    result.attrs = {k: v for k, v in cube.attrs.items() if k not in ("mnf", "mnf_denoise")}
    result.attrs.update(attrs)
    if cube.rio.crs is not None:
        result = result.rio.write_crs(cube.rio.crs)
    return result


def _apply_dates(tr: Union[MNFTransform, Sequence[MNFTransform]], cube: xr.DataArray, has_time: bool, nt: int):
    """The transform of each date: one for all, or one per date."""
    if isinstance(tr, MNFTransform):
        return [tr] * nt
    trs = list(tr)
    if not has_time or len(trs) != nt:
        raise ValueError(f"{len(trs)} transforms for {nt} dates: per_date transforms go with a cube of as many dates")
    return trs


def _apply(tr: Union[MNFTransform, Sequence[MNFTransform]], data: Any, n_components: Optional[int], scale: str,
           mask: Any, nodata: Any, bands: Optional[Sequence[Any]], chunks: Any, n_jobs: int,
           _stats: Optional[Dict[str, Any]]) -> xr.DataArray:
    from ._threads import resolve

    if scale not in ("noise", "unit"):
        raise ValueError(f"scale= {scale!r}: use 'noise' or 'unit'")
    first = tr if isinstance(tr, MNFTransform) else list(tr)[0]
    cube = _pick_bands(_cube(data, chunks, "mnf"), bands)
    cube = _match_bands(first, cube)
    m = len(first.eigenvalues)
    k_out = m if n_components is None else int(n_components)
    if not 1 <= k_out <= m:
        raise ValueError(f"n_components={n_components}: the transform has {m} components, ask for 1 to {m}")
    mask_ = _mask_array(mask, cube, "mask")
    missing = _missing(cube, nodata)
    work, has_time, arr = _work(cube)
    nt = arr.shape[0]
    trs = _apply_dates(tr, cube, has_time, nt)
    jobs = resolve(n_jobs)
    if work.chunks is not None:
        import dask.array as da

        arr = arr.rechunk({1: -1})
        parts = []
        for t in range(nt):
            parts.append(da.map_blocks(
                _forward_block, arr[t:t + 1], trs[t], k_out, scale, missing, mask_, dtype=np.float32,
                chunks=((1,), (k_out,)) + arr.chunks[2:], meta=np.empty((0, 0, 0, 0), np.float32)))
        out = parts[0] if nt == 1 else da.concatenate(parts, axis=0)
    else:
        a = _in_memory(np.asarray(arr))
        if all(t is trs[0] for t in trs):
            out = _forward_block(a, trs[0], k_out, scale, missing, mask_, jobs)
        else:
            out = np.concatenate([_forward_block(a[t:t + 1], trs[t], k_out, scale, missing, mask_, jobs)
                                  for t in range(nt)])
    names = [f"mnf{i + 1}" for i in range(k_out)]
    info = {"estimator": first.estimator, "eigenvalues": [float(v) for v in first.eigenvalues],
            "n_pixels": int(first.n_pixels), "scale": scale}
    return _wrap(out, cube, work, has_time, names, {"mnf": info})


def mnf(
    data: Any,
    n_components: Optional[int] = None,
    *,
    transform: Union[MNFTransform, Sequence[MNFTransform], None] = None,
    noise: str = "diff_hv",
    noise_region: Any = None,
    noise_cov: Any = None,
    mask: Any = None,
    bands: Optional[Sequence[Any]] = None,
    nodata: Union[float, str, None] = "auto",
    sample_fraction: float = 1.0,
    seed: int = 0,
    per_date: bool = False,
    scale: str = "noise",
    return_transform: bool = False,
    chunks: Any = None,
    n_jobs: int = -1,
) -> Union[xr.DataArray, Tuple[xr.DataArray, Union[MNFTransform, List[MNFTransform]]]]:
    """The Minimum Noise Fraction components of a multiband image, the highest signal-to-noise ratio first.

    Fits the transform (see ``mnf_fit`` for how, and for the parameters `noise` ... `n_jobs`) and applies it, or
    applies `transform` (fitted on another date or image). Component ``i`` has noise of unit variance and a variance
    ``1 + SNR_i`` (``scale="noise"``, as ENVI and SPy), or unit variance (``scale="unit"``); eigenvalues near 1
    are noise. Invalid pixels (a band missing, or outside `mask`) are NaN in every component.

    Parameters
    ----------
    data
        ``(band, y, x)`` or ``(time, band, y, x)``.
    n_components
        How many components to return (all by default).
    transform
        An ``MNFTransform`` (or, with a time axis, a list of one per date) to apply instead of fitting. Its bands
        must be named as the data's, else they are matched by position with a warning.
    per_date
        With a time axis: a transform per date, each applied to its date, instead of one for all the dates from
        statistics pooled over them (components then mean the same in every date).
    scale
        ``"noise"`` (default) or ``"unit"``.
    return_transform
        Also return the ``MNFTransform`` (or the list).

    Returns
    -------
    xarray.DataArray
        float32 ``(..., band, y, x)`` with ``band = ["mnf1", ...]`` and the coordinates and georeferencing of the
        input (lazy if the input is dask); ``attrs["mnf"]`` has the estimator, the eigenvalues, the number of pixels
        and the scale. With ``return_transform=True`` also the transform, whose ``table()`` has the eigenvalues,
        SNR and shares of the signal.

    Examples
    --------
    >>> y, tr = zeit.mnf(s2, return_transform=True)        # components, noise from the neighbours
    >>> tr.table()[["component", "eigenvalue", "snr"]].head()
    >>> y = zeit.mnf(s2, 5, noise_region=(85, 117, 122, 137))
    """
    if transform is not None:
        out = _apply(transform, data, n_components, scale, mask, nodata, bands, chunks, n_jobs, None)
        return (out, transform) if return_transform else out
    if n_components is not None and n_components < 1:
        raise ValueError("n_components must be 1 or more")
    tr = mnf_fit(data, noise=noise, noise_region=noise_region, noise_cov=noise_cov, mask=mask, bands=bands,
                 nodata=nodata, sample_fraction=sample_fraction, seed=seed, per_date=per_date, chunks=chunks,
                 n_jobs=n_jobs)
    out = _apply(tr, data, n_components, scale, mask, nodata, bands, chunks, n_jobs, None)
    return (out, tr) if return_transform else out


def inverse_mnf(
    components: Any,
    transform: MNFTransform,
    k: Optional[int] = None,
    *,
    snr_min: Optional[float] = None,
    scale: str = "noise",
    chunks: Any = None,
    n_jobs: int = -1,
) -> xr.DataArray:
    """The bands rebuilt from MNF components: ``mean + sum_{i <= k} p_i y_i``.

    The components after the first `k` are set to zero, which is their mean over the pixels the transform was
    fitted on: the noise directions are replaced by the mean, not shrunk towards the pixel's own value. With
    `k` equal to the components of the transform the input comes back (to float32 rounding); with ``k=0`` the
    mean. Constant bands come back as their constant.

    Parameters
    ----------
    components
        ``(band, y, x)`` or ``(time, band, y, x)`` as ``zeit.mnf`` returns, ``mnf1 ... mnfK``, K at least `k`.
    transform
        The ``MNFTransform`` they came from.
    k
        Components to keep (all of those given by default).
    snr_min
        Instead of `k`: keep the components whose SNR (eigenvalue minus 1) is at least this. `k` and `snr_min` are
        mutually exclusive.
    scale
        The scale of `components`: ``"noise"`` or ``"unit"`` (as given to ``mnf``).

    Returns
    -------
    xarray.DataArray
        float32, the bands of the transform with their names; ``attrs["mnf_denoise"]`` has `k` and `snr_min`.
    """
    from ._threads import resolve

    if scale not in ("noise", "unit"):
        raise ValueError(f"scale= {scale!r}: use 'noise' or 'unit'")
    cube = _cube(components, chunks, "inverse_mnf") if not (
        isinstance(components, xr.DataArray) and components.attrs.get("mnf")) else components
    if "band" not in cube.dims or not {"y", "x"} <= set(cube.dims):
        raise ValueError("inverse_mnf needs components with band, y and x dimensions")
    k_in = cube.sizes["band"]
    use = transform._choose(k, snr_min, min(k_in, len(transform.eigenvalues)))
    if use > k_in:
        raise ValueError(f"k={use}: only {k_in} components were given")
    work, has_time, arr = _work(cube)
    if work.chunks is not None:
        import dask.array as da

        arr = arr.rechunk({1: -1})
        out = da.map_blocks(_inverse_block, arr, transform, use, scale, dtype=np.float32,
                            chunks=(arr.chunks[0], (len(transform.bands),)) + arr.chunks[2:],
                            meta=np.empty((0, 0, 0, 0), np.float32))
    else:
        out = _inverse_block(np.asarray(arr), transform, use, scale, resolve(n_jobs))
    return _wrap(out, cube, work, has_time, list(transform.bands),
                 {"mnf_denoise": {"k": int(use), "snr_min": None if snr_min is None else float(snr_min)}})


def _denoise(tr: Union[MNFTransform, Sequence[MNFTransform]], data: Any, k: Optional[int], snr_min: Optional[float],
             mask: Any, nodata: Any, bands: Optional[Sequence[Any]], chunks: Any, n_jobs: int,
             _unused: Any) -> xr.DataArray:
    from ._threads import resolve

    first = tr if isinstance(tr, MNFTransform) else list(tr)[0]
    cube = _pick_bands(_cube(data, chunks, "mnf_denoise"), bands)
    cube = _match_bands(first, cube)
    ks = [t._choose(k, snr_min, None) for t in ([tr] if isinstance(tr, MNFTransform) else list(tr))]
    mask_ = _mask_array(mask, cube, "mask")
    missing = _missing(cube, nodata)
    work, has_time, arr = _work(cube)
    nt = arr.shape[0]
    trs = _apply_dates(tr, cube, has_time, nt)
    kk = [ks[0]] * nt if isinstance(tr, MNFTransform) else ks
    jobs = resolve(n_jobs)
    if work.chunks is not None:
        import dask.array as da

        arr = arr.rechunk({1: -1})
        parts = [da.map_blocks(_denoise_block, arr[t:t + 1], trs[t], kk[t], missing, mask_, dtype=np.float32,
                               chunks=arr[t:t + 1].chunks, meta=np.empty((0, 0, 0, 0), np.float32))
                 for t in range(nt)]
        out = parts[0] if nt == 1 else da.concatenate(parts, axis=0)
    else:
        a = _in_memory(np.asarray(arr))
        if nt == 1 or (isinstance(tr, MNFTransform)):
            out = _denoise_block(a, trs[0], kk[0], missing, mask_, jobs)
        else:
            out = np.concatenate([_denoise_block(a[t:t + 1], trs[t], kk[t], missing, mask_, jobs)
                                  for t in range(nt)])
    attrs = {"mnf_denoise": {"k": int(kk[0]) if len(set(kk)) == 1 else [int(v) for v in kk],
                             "snr_min": None if snr_min is None else float(snr_min)}}
    return _wrap(out, cube, work, has_time, [str(b) for b in cube.band.values], attrs)


def mnf_denoise(
    data: Any,
    k: Optional[int] = None,
    *,
    snr_min: Optional[float] = None,
    transform: Union[MNFTransform, Sequence[MNFTransform], None] = None,
    return_transform: bool = False,
    **fit_kwargs: Any,
) -> Union[xr.DataArray, Tuple[xr.DataArray, Union[MNFTransform, List[MNFTransform]]]]:
    """Removes the noise from a multiband image: the bands rebuilt from their first `k` MNF components.

    The forward and inverse transforms in one pass, without keeping the components (``2 B k`` operations a pixel). The
    components after the first `k` are set to their mean (zero): the noise they hold goes, and so does the signal
    they hold if `k` is too small. Give exactly one of `k` and `snr_min` (the components whose SNR, eigenvalue minus
    1, is at least this; 1 means signal at least as strong as the noise). The output keeps the bands' names, their
    units and the NaN of the invalid pixels.

    Parameters
    ----------
    data
        ``(band, y, x)`` or ``(time, band, y, x)``.
    k, snr_min
        Components to keep, or the minimum SNR of those kept; mutually exclusive, one is required.
    transform
        Apply this transform instead of fitting one.
    return_transform
        Also return the ``MNFTransform``.
    **fit_kwargs
        The parameters of ``zeit.mnf_fit`` (``noise``, ``noise_region``, ``mask``, ``bands``, ``nodata``,
        ``sample_fraction``, ``seed``, ``per_date``, ``chunks``, ``n_jobs``).

    Returns
    -------
    xarray.DataArray
        float32, the bands (all of them, denoised); ``attrs["mnf_denoise"]`` has `k` and `snr_min`.

    Examples
    --------
    >>> clean = zeit.mnf_denoise(enmap, snr_min=1.0)
    >>> clean, tr = zeit.mnf_denoise(enmap, k=12, noise_region=(85, 117, 122, 137), return_transform=True)
    """
    if k is not None and snr_min is not None:
        raise ValueError("give k or snr_min, not both")
    if k is None and snr_min is None:
        raise ValueError("give the components to keep: k=<number> or snr_min=<SNR>")
    run = {key: fit_kwargs[key] for key in ("mask", "nodata", "bands", "chunks", "n_jobs") if key in fit_kwargs}
    if transform is None:
        transform = mnf_fit(data, **fit_kwargs)
    out = _denoise(transform, data, k, snr_min, run.get("mask"), run.get("nodata", "auto"), run.get("bands"),
                   run.get("chunks"), run.get("n_jobs", -1), None)
    return (out, transform) if return_transform else out


# ---------------------------------------------------------------------------------------------------------------
# The numpy reference: what the compiled core computes, in plain numpy (the oracle of the tests)

def _splitmix64(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64)
        return z ^ (z >> np.uint64(31))


def _sampled(nt: int, ny: int, nx: int, date0: int, fraction: float, seed: int) -> np.ndarray:
    """``(time, y, x)`` bool: the pixels of the sample (the rule of the core)."""
    if fraction >= 1.0:
        return np.ones((nt, ny, nx), dtype=bool)
    key = ((np.arange(date0, date0 + nt, dtype=np.uint64)[:, None, None] * np.uint64(ny)
            + np.arange(ny, dtype=np.uint64)[None, :, None]) * np.uint64(nx)
           + np.arange(nx, dtype=np.uint64)[None, None, :])
    h = _splitmix64(key ^ _splitmix64(np.array([seed], dtype=np.uint64))[0])
    return (h >> np.uint64(11)).astype(np.float64) * 2.0 ** -53 < fraction


def _stats_numpy(x: np.ndarray, noise: str = "diff_hv", mask=None, region=None, missing: Sequence[float] = (),
                 fraction: float = 1.0, seed: int = 0, date0: int = 0):
    """``(n, mean, cov, pair counts, noise cov)`` of a cube ``(time, band, y, x)`` from the definitions."""
    x = np.asarray(x, dtype=np.float64)
    nt, nb, ny, nx = x.shape
    ok = np.isfinite(x)
    for v in missing:
        ok &= x != v
    valid = ok.all(axis=1)
    if mask is not None:
        valid &= np.asarray(mask, bool)[None]
    samp = _sampled(nt, ny, nx, date0, fraction, seed)
    pts = x.transpose(0, 2, 3, 1)[valid & samp]
    n = len(pts)
    mean = pts.mean(0)
    cov = np.cov(pts, rowvar=False)
    dirs = [d for d, _ in ([ESTIMATORS["diff_h"], ESTIMATORS["diff_v"]] if noise == "diff_hv"
                           else [ESTIMATORS[noise]] if noise in ESTIMATORS else [])]
    counts, total, acc = [], 0, np.zeros((nb, nb))
    for dy, dx in dirs:
        ys = slice(0, ny - dy) if dy >= 0 else slice(-dy, ny)
        yq = slice(dy, ny) if dy >= 0 else slice(0, ny + dy)
        xs = slice(0, nx - dx) if dx >= 0 else slice(-dx, nx)
        xq = slice(dx, nx) if dx >= 0 else slice(0, nx + dx)
        both = valid[:, ys, xs] & valid[:, yq, xq] & samp[:, ys, xs]
        if region is not None:
            r = np.asarray(region, bool)
            both &= r[None, ys, xs] & r[None, yq, xq]
        d = (x[:, :, ys, xs] - x[:, :, yq, xq]).transpose(0, 2, 3, 1)[both]
        d = d - d.mean(0)
        acc += d.T @ d
        counts.append(len(d))
        total += len(d)
    ncov = acc / (2.0 * (total - len(dirs))) if dirs else None
    return n, mean, cov, counts, ncov


def _fit_numpy(mean: np.ndarray, cov: np.ndarray, noise_cov: np.ndarray):
    """``(eigenvalues, forward (m, nb), inverse (nb, m), kept)``: the algorithm of the core with numpy's ``eigh``;
    the signs by the rule of the correlation with the band most correlated."""
    nb = len(mean)
    kept = [j for j in range(nb) if cov[j, j] > 0 and np.isfinite(cov[j, j])]
    c = cov[np.ix_(kept, kept)]
    nz = noise_cov[np.ix_(kept, kept)]
    s = 1.0 / np.sqrt(np.diag(c))
    sig = s[:, None] * c * s[None, :]
    sn = s[:, None] * nz * s[None, :]
    d, e = np.linalg.eigh((sn + sn.T) / 2)
    d, e = d[::-1], e[:, ::-1]
    d = np.maximum(d, EPS_NOISE * d[0])
    w = e / np.sqrt(d)
    cc = w.T @ sig @ w
    lam, u = np.linalg.eigh((cc + cc.T) / 2)
    lam, u = lam[::-1], u[:, ::-1]
    a = s[:, None] * (w @ u)
    p = (e * np.sqrt(d)) @ u / s[:, None]
    r = np.sqrt(np.maximum(lam, 0))[None, :] * s[:, None] * p
    for i in range(len(lam)):
        mx = np.abs(r[:, i]).max()
        if mx > 0:
            j = int(np.nonzero(np.abs(r[:, i]) >= mx * (1 - 1e-12))[0][0])
            if r[j, i] < 0:
                a[:, i] *= -1
                p[:, i] *= -1
    forward = np.zeros((len(lam), nb))
    inverse = np.zeros((nb, len(lam)))
    forward[:, kept] = a.T
    inverse[kept] = p
    return lam, forward, inverse, kept


def _forward_numpy(x: np.ndarray, mean, forward, k_out: int, missing: Sequence[float] = (), mask=None,
                   scale: str = "noise", lam=None) -> np.ndarray:
    """``(time, band, y, x)`` -> ``(time, k_out, y, x)`` float32, NaN where a band is invalid."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    for v in missing:
        ok &= x != v
    valid = ok.all(axis=1)
    if mask is not None:
        valid &= np.asarray(mask, bool)[None]
    c = np.where(valid[:, None], x - np.asarray(mean)[None, :, None, None], 0.0)
    y = np.einsum("ib,tbyx->tiyx", forward[:k_out], c)
    if scale == "unit":
        y = y / np.sqrt(np.where(lam[:k_out] >= DEGENERATE, lam[:k_out], np.inf))[None, :, None, None]
    y[~np.broadcast_to(valid[:, None], y.shape)] = np.nan
    return y.astype(np.float32)


def _denoise_numpy(x: np.ndarray, mean, forward, inverse, k: int, missing: Sequence[float] = (), mask=None
                   ) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    for v in missing:
        ok &= x != v
    valid = ok.all(axis=1)
    if mask is not None:
        valid &= np.asarray(mask, bool)[None]
    c = np.where(valid[:, None], x - np.asarray(mean)[None, :, None, None], 0.0)
    y = np.einsum("ib,tbyx->tiyx", forward[:k], c)
    z = np.asarray(mean)[None, :, None, None] + np.einsum("bi,tiyx->tbyx", inverse[:, :k], y)
    z[~np.broadcast_to(valid[:, None], z.shape)] = np.nan
    return z.astype(np.float32)
