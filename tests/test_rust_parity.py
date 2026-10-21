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


# ------------------------------------------------------------------ Whittaker
@pytest.mark.parametrize("lmbda", [0.0, 1.0, 10.0, 1e4])
@pytest.mark.parametrize("weighted", [False, True])
def test_whittaker_batch(lmbda, weighted):
    rng = np.random.default_rng(int(lmbda) + weighted)
    x = np.sort(rng.choice(2000, 90, replace=False)).astype(float)
    v = rng.normal(0.5, 0.2, (2000, 90))
    v[rng.random(v.shape) < 0.3] = np.nan
    v[::40] = np.nan                        # nothing to smooth
    v[1::40, 2:] = np.nan                   # two observations
    w = rng.random(v.shape) if weighted else np.empty((0, 0))
    for n_jobs in (1, -1):
        assert_same_bits(_core.smooth.whittaker_batch(v, x, w, lmbda, n_jobs),
                         _rust.smooth.whittaker_batch(v, x, w, lmbda, n_jobs))


def test_whittaker_single_and_errors():
    rng = np.random.default_rng(3)
    x, y, w = np.arange(30.0) * 5, rng.normal(size=30), rng.random(30)
    assert_same_bits(np.array(_core.smooth.whittaker(list(x), list(y), list(w), 7.0)),
                     np.array(_rust.smooth.whittaker(list(x), list(y), list(w), 7.0)))
    bad = [0.0, 2.0, 1.0, 3.0]
    with pytest.raises(ValueError, match="x must increase"):
        _rust.smooth.whittaker(bad, [1.0] * 4, [1.0] * 4, 10.0)
    # the C++ batch would terminate the process here (an exception out of OpenMP)
    with pytest.raises(ValueError, match="x must increase"):
        _rust.smooth.whittaker_batch(np.ones((3, 4)), np.array(bad), np.empty((0, 0)), 10.0)


# --------------------------------------------------------------------- medoid
@pytest.mark.parametrize("bands", [1, 2, 3, 4, 6, 7])
def test_medoid(bands):
    rng = np.random.default_rng(bands)
    m = rng.random((40, 30, 24, bands))
    m[rng.random(m.shape) < 0.1] = np.nan
    m[0, 0] = np.nan                         # no observation
    m[1, 1, :, 0] = -9999.0                  # a band with none
    for no_data in (-9999.0, np.nan):
        assert_same_bits(_core.utils.compute_medoid(m, no_data), _rust.utils.compute_medoid(m, no_data))


# ------------------------------------------------------------------------ SMA
@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("bands", [1, 3, 6])
def test_unmix(k, bands):
    if k > bands + 1:
        pytest.skip("more endmembers than equations")
    rng = np.random.default_rng(10 * k + bands)
    em = np.abs(rng.normal(0.3, 0.15, (k, bands)))
    v = rng.dirichlet(np.ones(k), 2000) @ em + rng.normal(0, 0.01, (2000, bands))
    v[::97, 0] = np.nan
    for sum_to_one in (True, False):
        for nonneg in (True, False):
            for n_jobs in (1, -1):
                assert_same_bits(_core.sma.unmix_batch(v, em, sum_to_one, nonneg, 1000.0, n_jobs),
                                 _rust.sma.unmix_batch(v, em, sum_to_one, nonneg, 1000.0, n_jobs))
    y, e = list(v[1]), list(em.ravel())
    assert_same_bits(np.array(_core.sma.unmix_pixel(y, e, k)), np.array(_rust.sma.unmix_pixel(y, e, k)))


def test_nnls():
    rng = np.random.default_rng(11)
    for _ in range(300):   # square, over- and underdetermined, rank deficient
        m, n = int(rng.integers(1, 9)), int(rng.integers(1, 7))
        a, b = rng.normal(size=m * n), rng.normal(size=m)
        if rng.random() < 0.2 and n > 1:
            a = a.reshape(m, n)
            a[:, -1] = a[:, 0]
            a = a.ravel()
        assert_same_bits(np.array(_core.sma.nnls(list(a), list(b), m, n)), np.array(_rust.sma.nnls(list(a), list(b), m, n)))


# ---------------------------------------------------------------------- CODED
def _coded_params(mod, **kw):
    p = mod.coded.Params()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


@pytest.mark.parametrize("n_features", [0, 1, 4])
def test_coded(n_features):
    rng = np.random.default_rng(n_features)
    t = np.sort(2000 + rng.random(260) * 12)
    ndfi = 0.85 + 0.03 * np.sin(2 * np.pi * t)[None] + rng.normal(0, 0.02, (1500, 260))
    for k in range(1500):
        for _ in range(rng.integers(0, 3)):
            at = 2003.5 + rng.random() * 7
            ndfi[k] -= rng.uniform(0.1, 0.8) * ((t >= at) & (t < at + rng.uniform(0.3, 5)))
    ndfi[rng.random(ndfi.shape) < 0.25] = np.nan
    feats = rng.random((1500, n_features, 260))
    for kw in (dict(train_start=2000.0), dict(train_start=2000.0, loss_only=False, consec=2, max_events=5),
               dict(train_start=2001.0, min_obs=10, thresh=2.0)):
        for n_jobs in (1, -1):
            assert_same_bits(_core.coded.coded_batch(t, ndfi, feats, _coded_params(_core, **kw), n_jobs),
                             _rust.coded.coded_batch(t, ndfi, feats, _coded_params(_rust, **kw), n_jobs))


def test_coded_errors():
    p = _coded_params(_rust, train_start=2000.0)
    with pytest.raises(RuntimeError, match="sorted"):
        _rust.coded.coded_batch(np.array([2001.0, 2000.0]), np.zeros((1, 2)), np.zeros((1, 0, 2)), p)
    assert _rust.coded.coded_size(4, 3) == _core.coded.coded_size(4, 3)


# ---------------------------------------------------------------------- Tmask
@pytest.mark.parametrize("min_observations", [5, 12])
def test_tmask_batch(min_observations):
    rng = np.random.default_rng(min_observations)
    days = np.arange(730120, 730120 + 6 * 365, 8, dtype=float)
    g = 800 + 200 * np.sin(2 * np.pi * days / 365.25)[None] + rng.normal(0, 30, (1500, days.size))
    s = 2000 + 300 * np.cos(2 * np.pi * days / 365.25)[None] + rng.normal(0, 30, (1500, days.size))
    g[rng.random(g.shape) < 0.1] += 3000
    s[rng.random(s.shape) < 0.05] -= 1500
    g[rng.random(g.shape) < 0.2] = np.nan
    g[::50, 10:] = np.nan                    # too few observations
    for n_jobs in (1, -1):
        assert_same_bits(_core.tmask.tmask_batch(g, s, days, 10000.0, min_observations, 0.04, 0.04, n_jobs),
                         _rust.tmask.tmask_batch(g, s, days, 10000.0, min_observations, 0.04, 0.04, n_jobs))


def test_robustfit():
    rng = np.random.default_rng(12)
    for _ in range(300):   # up to 7 columns with the intercept: both of Eigen's row-norm paths
        n, p = int(rng.integers(5, 80)), int(rng.integers(1, 7))
        x = rng.normal(size=(n, p))
        y = rng.normal(size=n) + x @ rng.normal(size=p)
        y[rng.random(n) < 0.1] += 10
        assert_same_bits(np.array(_core.tmask.robustfit(list(x.ravel()), p, list(y))),
                         np.array(_rust.tmask.robustfit(list(x.ravel()), p, list(y))))


# ------------------------------------------------- BFAST Monitor and BFAST Lite
def _bfast_series(seed, freq, years, n_pixels=600):
    rng = np.random.default_rng(seed)
    n = freq * years
    t = 2000 + np.arange(n) / freq
    v = 0.6 + 0.2 * np.sin(2 * np.pi * t)[None] + rng.normal(0, 0.03, (n_pixels, n))
    for k in range(n_pixels):
        v[k, rng.integers(n // 5, 4 * n // 5):] -= rng.uniform(-0.3, 0.3)
    v[rng.random(v.shape) < 0.12] = np.nan
    v[::97] = np.nan                         # nothing to fit
    return v


# frequency, years, harmonic order: 3 to 10 regressors, so Eigen's coefficient-based
# products and both kinds of row blocks of its GEMM kernel
BFAST_CASES = [(23, 8, 3), (12, 15, 2), (4, 6, 1), (23, 3, 3), (46, 10, 3), (12, 8, 4)]


@pytest.mark.parametrize("freq,years,order", BFAST_CASES)
def test_bfast_monitor(freq, years, order):
    v = _bfast_series(freq + order, freq, years)
    for h, period in ((0.25, 10), (0.5, 4), (1.0, 2)):
        for n_jobs in (1, -1):
            args = (v, 2000.0, 2000.0 + years * 0.6, freq, order, h, period, 0.05, 10, n_jobs)
            assert_same_bits(_core.bfastmonitor.fit_bfast_monitor_batch(*args),
                             _rust.bfastmonitor.fit_bfast_monitor_batch(*args))


def test_bfast_monitor_single_and_errors():
    v = _bfast_series(1, 23, 8, 5)[2]
    a = _core.bfastmonitor.bfast_monitor(list(v), 2000.0, 2005.0, 23)
    b = _rust.bfastmonitor.bfast_monitor(list(v), 2000.0, 2005.0, 23)
    for name in ("breakpoint", "breakpoint_idx", "magnitude", "sigma", "n_history", "has_break", "valid"):
        assert_same_bits(np.array(getattr(a, name)), np.array(getattr(b, name)))
    # the C++ batch terminates the process on these (an exception out of OpenMP)
    for h, period in ((0.3, 10), (0.25, 3)):
        with pytest.raises(RuntimeError, match="must be one of"):
            _rust.bfastmonitor.fit_bfast_monitor_batch(np.zeros((2, 50)), 2000.0, 2001.0, 23, 3, h, period)


@pytest.mark.parametrize("freq,years,order", [(23, 6, 3), (12, 10, 2), (4, 12, 1), (23, 4, 3), (12, 8, 4)])
def test_bfast_lite(freq, years, order):
    v = _bfast_series(100 + freq + order, freq, years, 200)
    for h, max_breaks in ((0.15, 5), (0.25, 2)):
        for n_jobs in (1, -1):
            args = (v, 2000.0, freq, order, h, max_breaks, 20, n_jobs)
            assert_same_bits(_core.bfastlite.fit_bfast_lite_batch(*args), _rust.bfastlite.fit_bfast_lite_batch(*args))
    y = list(v[3])
    a, b = _core.bfastlite.bfast_lite(y, 2000.0, freq, order), _rust.bfastlite.bfast_lite(y, 2000.0, freq, order)
    for name in ("n_breaks", "rss", "lwz", "n_valid", "valid", "breakpoint_idx", "magnitude", "first_after_idx"):
        assert_same_bits(np.array(getattr(a, name), dtype=float), np.array(getattr(b, name), dtype=float))
