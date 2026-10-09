import numpy as np
import xarray as xr
import dask.array as da
import pytest

from zeit._core.mannkendall import fit_mann_kendall_batch, mk_test_single, MKMethod
from zeit.trend import run_mann_kendall_dask, N_MK_METRICS, MK_METRIC_NAMES
import zeit.xarray_api  # noqa: F401 - registers the .zeit accessor

pymannkendall = pytest.importorskip(
    "pymannkendall",
    reason="pymannkendall is only needed to cross-validate the C++ port against the reference implementation",
)

TREND_CODE = {"increasing": 1, "decreasing": -1, "no trend": 0}
FIELDS = ["trend_code", "h", "p", "z", "tau", "s", "var_s", "slope", "intercept"]


def _assert_matches_reference(y, ref, cpp_vals, rtol=1e-6):
    ref_vals = [
        TREND_CODE[ref.trend], float(ref.h), ref.p, ref.z, ref.Tau,
        ref.s, ref.var_s, ref.slope, ref.intercept,
    ]
    for fname, rv, cv in zip(FIELDS, ref_vals, cpp_vals):
        both_nan = np.isnan(rv) if isinstance(rv, float) else False
        both_nan = both_nan and (np.isnan(cv) if isinstance(cv, float) else False)
        if both_nan:
            continue
        if fname == "h":
            assert rv == cv, f"{fname}: ref={rv} cpp={cv}"
        else:
            assert abs(rv - cv) < rtol, f"{fname}: ref={rv} cpp={cv}"


# --- Cross-validation against the reference pymannkendall implementation ---

@pytest.mark.parametrize("kind", ["trend", "no_trend", "ties", "autocorrelated", "with_nans", "decreasing"])
def test_mk_test_single_matches_pymannkendall_original(kind):
    rng = np.random.RandomState(0)
    y = _make_series(kind, rng)
    ref = pymannkendall.original_test(y)
    cpp = mk_test_single(list(y), method=int(MKMethod.ORIGINAL))
    _assert_matches_reference(y, ref, cpp)


@pytest.mark.parametrize("kind", ["trend", "no_trend", "ties", "autocorrelated", "with_nans", "decreasing"])
def test_mk_test_single_matches_pymannkendall_hamed_rao(kind):
    rng = np.random.RandomState(1)
    y = _make_series(kind, rng)
    ref = pymannkendall.hamed_rao_modification_test(y)
    cpp = mk_test_single(list(y), method=int(MKMethod.HAMED_RAO))
    _assert_matches_reference(y, ref, cpp)


@pytest.mark.parametrize("kind", ["trend", "no_trend", "ties", "autocorrelated", "with_nans", "decreasing"])
def test_mk_test_single_matches_pymannkendall_yue_wang(kind):
    rng = np.random.RandomState(2)
    y = _make_series(kind, rng)
    ref = pymannkendall.yue_wang_modification_test(y)
    cpp = mk_test_single(list(y), method=int(MKMethod.YUE_WANG))
    _assert_matches_reference(y, ref, cpp)


def test_mk_test_single_matches_pymannkendall_seasonal():
    rng = np.random.RandomState(3)
    period = 12
    y = 0.02 * np.arange(96) + rng.normal(0, 1.0, 96)
    ref = pymannkendall.seasonal_test(y, period=period)
    cpp = mk_test_single(list(y), method=int(MKMethod.SEASONAL), period=period)
    _assert_matches_reference(y, ref, cpp)


def _make_series(kind, rng):
    if kind == "trend":
        return np.arange(30) * 0.4 + rng.normal(0, 0.3, 30)
    if kind == "decreasing":
        return -np.arange(25) * 0.3 + rng.normal(0, 0.2, 25)
    if kind == "no_trend":
        return rng.normal(0, 1.0, 40)
    if kind == "ties":
        return np.repeat(np.arange(10), 3).astype(float) + rng.normal(0, 0.01, 30)
    if kind == "autocorrelated":
        return np.cumsum(rng.normal(0, 1, 50)) * 0.1
    if kind == "with_nans":
        y = np.arange(30) * 0.4 + rng.normal(0, 0.3, 30)
        y[5] = np.nan
        y[17] = np.nan
        return y
    raise ValueError(kind)


# --- Batch / OpenMP -------------------------------------------------------

def test_fit_mann_kendall_batch_shape_and_trend_sign():
    rng = np.random.RandomState(4)
    n_pixels, n_time = 6, 25
    values = np.zeros((n_pixels, n_time))
    values[0] = np.arange(n_time) * 0.5 + rng.normal(0, 0.1, n_time)   # increasing
    values[1] = -np.arange(n_time) * 0.5 + rng.normal(0, 0.1, n_time)  # decreasing
    values[2] = rng.normal(0, 1.0, n_time)                             # flat/no trend
    values[3:] = rng.normal(0, 1.0, (3, n_time))

    out = fit_mann_kendall_batch(values_array=values, method=int(MKMethod.HAMED_RAO), n_jobs=1)

    assert out.shape == (9, n_pixels)
    assert out[0, 0] == 1.0   # increasing
    assert out[0, 1] == -1.0  # decreasing


def test_fit_mann_kendall_batch_respects_min_valid():
    values = np.full((1, 10), np.nan)
    values[0, :3] = [1.0, 2.0, 3.0]  # only 3 valid points, below default min_valid=4

    out = fit_mann_kendall_batch(values_array=values, method=int(MKMethod.ORIGINAL), min_valid=4)
    assert np.all(np.isnan(out))

    out2 = fit_mann_kendall_batch(values_array=values, method=int(MKMethod.ORIGINAL), min_valid=3)
    assert not np.all(np.isnan(out2))


def test_fit_mann_kendall_batch_is_deterministic_across_thread_counts():
    rng = np.random.RandomState(5)
    values = rng.normal(0, 1, (200, 20))
    values += (np.arange(20) * 0.1)[None, :]

    out_1 = fit_mann_kendall_batch(values_array=values, method=int(MKMethod.HAMED_RAO), n_jobs=1)
    out_n = fit_mann_kendall_batch(values_array=values, method=int(MKMethod.HAMED_RAO), n_jobs=-1)

    np.testing.assert_allclose(out_1, out_n, equal_nan=True)


# --- Dask wrapper + xarray accessor ---------------------------------------

def test_run_mann_kendall_dask_shape_and_trend():
    time_steps, rows, cols = 25, 4, 4
    rng = np.random.RandomState(6)

    block = rng.normal(0, 0.2, (time_steps, rows, cols))
    block += (np.arange(time_steps) * 0.5)[:, None, None]  # every pixel trends up
    data = da.from_array(block, chunks=(time_steps, 2, 2))

    out = run_mann_kendall_dask(data, method="hamed_rao")
    assert out.shape == (N_MK_METRICS, rows, cols)

    computed = out.compute()
    assert np.all(computed[0] == 1.0)  # trend row: all increasing


def test_xarray_accessor_mann_kendall():
    time_steps, rows, cols = 25, 5, 5
    rng = np.random.RandomState(7)

    block = rng.normal(0, 0.2, (time_steps, rows, cols))
    block += (np.arange(time_steps) * 0.5)[:, None, None]
    data = da.from_array(block, chunks=(time_steps, 5, 5))

    ds = xr.DataArray(data, dims=["time", "y", "x"], coords={"y": np.arange(rows), "x": np.arange(cols)})
    res = ds.zeit.mann_kendall(method="hamed_rao")

    assert isinstance(res, xr.Dataset)
    assert res.trend.dims == ("y", "x")
    assert list(res.data_vars) == MK_METRIC_NAMES

    computed = res.compute()
    trend_mean = computed.trend.values.mean()
    assert trend_mean == 1.0
