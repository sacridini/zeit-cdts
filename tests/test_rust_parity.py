"""The Rust core against the C++ one, on the modules already converted.

The Rust modules are ports of the C++ ones operation by operation, so here they
must give the same bits (NaN where the C++ gives NaN), on any thread count, for
the edge cases as well as the ordinary ones. The parity of each module with its
authors' code is checked by its own tests, which run on either core
(ZEIT_BACKEND=rust); see RUST_MIGRATION.md.
"""
import numpy as np
import pytest

from zeit import _core

_rust = pytest.importorskip("zeit._rust", reason="zeit built without its Rust core")


def assert_same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.dtype.kind == "f":
        both_nan = np.isnan(a) & np.isnan(b)
        differ = (a.view(np.uint64 if a.itemsize == 8 else np.uint32) != b.view(np.uint64 if b.itemsize == 8 else np.uint32))
        differ &= ~both_nan
    else:
        differ = a != b
    assert not differ.any(), (
        f"{differ.sum()} of {differ.size} values differ, e.g. at {np.argwhere(differ)[0]}: "
        f"cpp {a[tuple(np.argwhere(differ)[0])]!r} rust {b[tuple(np.argwhere(differ)[0])]!r}")


# ---------------------------------------------------------------- Mann-Kendall
def _mk_series(seed, n_pixels=3000, n_time=30):
    rng = np.random.default_rng(seed)
    v = 0.5 + 0.01 * np.arange(n_time) + rng.normal(0, 0.1, (n_pixels, n_time))
    v[rng.random(v.shape) < 0.15] = np.nan          # gaps
    v[::5] = np.round(v[::5], 1)                     # ties
    v[1::17] = 0.3                                   # constant (variance 0)
    v[2::19, : n_time - 3] = np.nan                  # 3 valid: below min_valid
    v[3::23] = np.nan                                # empty
    v[4::29] = -v[4::29]                             # decreasing
    return v


@pytest.mark.parametrize("method", [0, 1, 2, 3])
@pytest.mark.parametrize("n_jobs", [1, -1])
def test_mann_kendall_batch(method, n_jobs):
    v = _mk_series(method)
    for lag, period, alpha in [(-1, 1, 0.05), (3, 4, 0.1), (0, 12, 0.01)]:
        a = _core.mannkendall.fit_mann_kendall_batch(v, method, alpha, lag, period, 4, n_jobs)
        b = _rust.mannkendall.fit_mann_kendall_batch(v, method, alpha, lag, period, 4, n_jobs)
        assert_same_bits(a, b)


@pytest.mark.parametrize("n_time", [1, 2, 4, 5, 8, 60])
def test_mann_kendall_lengths(n_time):
    v = _mk_series(7, 400, n_time)
    for method in range(4):
        a = _core.mannkendall.fit_mann_kendall_batch(v, method, 0.05, -1, 3, 1, 1)
        b = _rust.mannkendall.fit_mann_kendall_batch(v, method, 0.05, -1, 3, 1, 1)
        assert_same_bits(a, b)


def test_mann_kendall_layouts_and_dtypes():
    v = _mk_series(8, 500, 24)
    for arr in (np.asfortranarray(v), v[:, ::2], (v * 100).astype(np.float32), np.nan_to_num(v * 100).astype(np.int64)):
        a = _core.mannkendall.fit_mann_kendall_batch(np.ascontiguousarray(arr, dtype=np.float64), 1)
        b = _rust.mannkendall.fit_mann_kendall_batch(arr, 1)
        assert_same_bits(a, b)


def test_mann_kendall_infinities_do_not_crash():
    v = _mk_series(9, 50, 20)
    v[::3, 5] = np.inf
    v[1::3, 7] = -np.inf
    out = _rust.mannkendall.fit_mann_kendall_batch(v, 1)
    assert out.shape == (9, 50)


def test_mann_kendall_single():
    rng = np.random.default_rng(10)
    for method in range(4):
        for _ in range(50):
            y = list(rng.normal(0, 1, rng.integers(3, 40)))
            a = _core.mannkendall.mk_test_single(y, method, 0.05, -1, 4)
            b = _rust.mannkendall.mk_test_single(y, method, 0.05, -1, 4)
            assert a[:2] == b[:2]
            assert_same_bits(np.array(a[2:]), np.array(b[2:]))


def test_mann_kendall_method_enum():
    for name in ("ORIGINAL", "HAMED_RAO", "YUE_WANG", "SEASONAL"):
        assert int(getattr(_core.mannkendall.MKMethod, name)) == int(getattr(_rust.mannkendall.MKMethod, name))
        assert int(getattr(_rust.mannkendall, name)) == int(getattr(_core.mannkendall, name))


# ---------------------------------------------------------------------- TWDTW
def _tw_params(mod, **kw):
    p = mod.twdtw.TWDTWParams()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


TW_SETTINGS = [
    dict(),
    dict(subsequence_matching=True),
    dict(cycle_length=366.0, max_time_warp=60, alpha=1.0, beta=0.1, gamma=30.0),
    dict(max_time_warp=10),
]


@pytest.mark.parametrize("bands", [1, 2, 3, 4, 5, 7, 9])
@pytest.mark.parametrize("n_jobs", [1, -1])
def test_twdtw_batch(bands, n_jobs):
    rng = np.random.default_rng(bands)
    extra = (bands,) if bands > 1 else ()
    v = rng.normal(0.5, 0.2, (30, 30, 46) + extra)
    v[rng.random(v.shape) < 0.05] = np.nan   # dates dropped from their series
    v[0, 0] = np.nan                          # no date left
    dates = np.arange(0, 46 * 16, 16, dtype=np.int32)
    pat = rng.normal(0.5, 0.2, (23,) + extra)
    pat_dates = np.arange(0, 23 * 16, 16, dtype=np.int32)
    for kw in TW_SETTINGS:
        for abort in (np.inf, 12.0):   # a finite one prunes with LB_Keogh and abandons early
            a = _core.twdtw.fit_twdtw_batch(v, dates, pat, pat_dates, _tw_params(_core, **kw), abort, n_jobs)
            b = _rust.twdtw.fit_twdtw_batch(v, dates, pat, pat_dates, _tw_params(_rust, **kw), abort, n_jobs)
            assert_same_bits(a, b)


@pytest.mark.parametrize("bands", [1, 3])
def test_twdtw_single_with_path(bands):
    rng = np.random.default_rng(20 + bands)
    for kw in TW_SETTINGS:
        for _ in range(10):
            n, m = rng.integers(1, 40), rng.integers(1, 20)
            ts, pat = rng.normal(size=n * bands).tolist(), rng.normal(size=m * bands).tolist()
            td = np.sort(rng.choice(730, n, replace=False)).tolist()
            pd_ = np.sort(rng.choice(365, m, replace=False)).tolist()
            for return_path in (False, True):
                a = _core.twdtw.fit_twdtw(ts, td, pat, pd_, bands, _tw_params(_core, **kw), float("inf"), return_path)
                b = _rust.twdtw.fit_twdtw(ts, td, pat, pd_, bands, _tw_params(_rust, **kw), float("inf"), return_path)
                assert_same_bits(np.array(a.distance), np.array(b.distance))
                assert a.path == b.path


def test_twdtw_int64_dates_and_params_defaults():
    v = np.random.default_rng(0).normal(size=(4, 5, 20))
    dates, pat_dates = np.arange(0, 320, 16), np.arange(0, 160, 16)
    pat = np.sin(np.arange(10.0))
    a = _core.twdtw.fit_twdtw_batch(v, dates, pat, pat_dates, _core.twdtw.TWDTWParams())
    b = _rust.twdtw.fit_twdtw_batch(v, dates, pat, pat_dates, _rust.twdtw.TWDTWParams())
    assert_same_bits(a, b)
    pc, pr = _core.twdtw.TWDTWParams(), _rust.twdtw.TWDTWParams()
    for name in ("alpha", "beta", "gamma", "max_time_warp", "subsequence_matching", "cycle_length"):
        assert getattr(pc, name) == getattr(pr, name)


# ----------------------------------------------------------------------- SNIC
@pytest.mark.parametrize("features", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_snic(features, dtype):
    rng = np.random.default_rng(features)
    y, x = np.mgrid[0:120, 0:97] / 40.0
    data = np.stack([np.sin(3 * x + k) * np.cos(2 * y - k) for k in range(features)])
    data = (data + rng.normal(0, 0.1, data.shape)).astype(dtype)
    data[:, rng.random((120, 97)) < 0.03] = np.nan       # masked pixels
    seeds = np.array([(r, c) for r in range(3, 120, 9) for c in range(3, 97, 9)], dtype=np.int32)
    seeds[0] = np.argwhere(np.isnan(data[0]))[0]         # a seed on a masked pixel: empty segment
    for compactness, th, tw in [(10.0, 0, 0), (0.5, 40, 50), (30.0, 120, 20)]:
        for n_jobs in (1, -1):
            a = _core.snic.snic_segment(data, seeds, compactness, th, tw, n_jobs)
            b = _rust.snic.snic_segment(data, seeds, compactness, th, tw, n_jobs)
            for u, v in zip(a, b):
                assert_same_bits(u, v)


def test_snic_errors_match():
    data = np.zeros((1, 4, 4))
    cases = [
        (np.zeros((4, 4)), np.array([[0, 0]])),          # not 3D
        (data, np.array([[0, 0, 0]])),                   # seeds not (n, 2)
        (data, np.array([[4, 0]])),                      # seed outside
        (data, np.zeros((0, 2))),                        # no seed
    ]
    for d, s in cases:
        with pytest.raises(ValueError) as ec:
            _core.snic.snic_segment(d, s)
        with pytest.raises(ValueError) as er:
            _rust.snic.snic_segment(d, s)
        assert str(ec.value) == str(er.value)
    with pytest.raises(ValueError, match="compactness"):
        _rust.snic.snic_segment(data, np.array([[0, 0]]), float("nan"))
