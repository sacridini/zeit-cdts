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


@pytest.mark.parametrize("freq,years,order", [(23, 6, 3), (12, 10, 2), (4, 12, 1), (12, 8, 3)])
def test_bfast(freq, years, order):
    v = _bfast_series(200 + freq + order, freq, years, 120)
    for h in (0.15, 0.25):
        for n_jobs in (1, -1):
            args = (v, 2000.0, freq, order, h, 5, 5, 10, 0.05, 20, n_jobs)
            assert_same_bits(_core.bfast.fit_bfast_batch(*args), _rust.bfast.fit_bfast_batch(*args))
    y = list(v[5])
    a, b = _core.bfast.bfast(y, 2000.0, freq, order), _rust.bfast.bfast(y, 2000.0, freq, order)
    for name in ("n_trend_breaks", "n_season_breaks", "magnitude", "time", "n_iter", "n_valid", "valid",
                 "trend_breakpoint_idx", "season_breakpoint_idx", "trend_magnitude", "trend_first_after_idx"):
        assert_same_bits(np.array(getattr(a, name), dtype=float), np.array(getattr(b, name), dtype=float))


# ------------------------------------------------------------------------ SOM
@pytest.mark.parametrize("dim", [2, 5, 8, 13, 130])   # NumPy's pairwise sums: < 8, 8..128, > 128
def test_som(dim):
    rng = np.random.default_rng(dim)
    data = rng.normal(size=(400, dim))
    for x, y in ((5, 5), (3, 7)):
        xx, yy = (np.ascontiguousarray(a) for a in np.meshgrid(np.arange(x, dtype=float), np.arange(y, dtype=float),
                                                              indexing="ij"))
        w0 = rng.normal(size=(x, y, dim))
        order = rng.integers(0, 400, 200).astype(np.int64)
        for neighborhood in range(4):
            for decay in range(3):
                schedule = (0.5, 1.2, decay, decay, neighborhood, xx, yy)
                wa, wb = w0.copy(), w0.copy()
                _core.som.train_batch(wa, data, 10, *schedule, n_jobs=-1)
                _rust.som.train_batch(wb, data, 10, *schedule, n_jobs=-1)
                assert_same_bits(wa, wb)
                wa, wb = w0.copy(), w0.copy()
                _core.som.train_online(wa, data, order, 200, False, *schedule)
                _rust.som.train_online(wb, data, order, 200, False, *schedule)
                assert_same_bits(wa, wb)
                assert_same_bits(_core.som.predict_bmus(data, wa, -1), _rust.som.predict_bmus(data, wb, -1))


# ----------------------------------------------------------------- LandTrendr
LT_YEARS = np.arange(1985, 2025, dtype=np.int32)


def _lt_series(seed, n=600):
    """Stable, disturbed, recovering, trending, spiky and integer-scaled series,
    with NaN gaps, missing first and last years, short series and no-data."""
    rng = np.random.default_rng(seed)
    t = LT_YEARS.size
    out = np.empty((n, t))
    for k in range(n):
        kind = k % 6
        base = rng.uniform(0.3, 0.9)
        y = np.full(t, base) + rng.normal(0, rng.uniform(0.005, 0.05), t)
        if kind in (1, 2, 4):
            at, depth = rng.integers(3, t - 3), rng.uniform(0.1, 0.6)
            y[at:] -= depth
            if kind == 2:                                  # recovery, slow or fast
                y[at:] += np.minimum(rng.uniform(0.005, 0.08) * np.arange(t - at), depth)
        if kind == 3:
            y += rng.uniform(-0.01, 0.01) * np.arange(t)
        if kind == 4:
            for _ in range(rng.integers(1, 4)):
                y[rng.integers(0, t)] += rng.choice([-1, 1]) * rng.uniform(0.2, 0.6)
        if kind == 5:
            y = np.round(y * 1000)                         # NBR x 1000
        out[k] = y
    out[rng.random(out.shape) < 0.08] = np.nan
    out[::17, :3] = np.nan                                 # missing first years
    out[::19, -2:] = np.nan                                # missing last years
    out[::23, 6:] = np.nan                                 # below min_observations_needed
    out[::29] = -9999.0                                    # all no-data
    out[1::31, 10] = -9999.0
    return out


def _lt_params(mod, **kw):
    p = mod.landtrendr.LandTrendrParams()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


LT_SETTINGS = [
    dict(),
    dict(modifier=-1.0),
    dict(max_segments=3, pval_threshold=0.1),
    dict(max_segments=8, recovery_threshold=1.0, spike_threshold=1.0),
    dict(best_model_proportion=1.25, vertex_count_overshoot=0, min_observations_needed=10),
    dict(prevent_fast_recovery=False, recovery_threshold=0.1, modifier=-1.0, spike_threshold=0.75),
]


@pytest.mark.parametrize("setting", range(len(LT_SETTINGS)))
def test_landtrendr_batch(setting):
    kw = LT_SETTINGS[setting]
    v = _lt_series(setting).reshape(20, 30, LT_YEARS.size)
    for n_jobs in (1, -1):
        a = _core.landtrendr.fit_trajectory_batch(v, LT_YEARS, _lt_params(_core, **kw), -9999.0, n_jobs)
        b = _rust.landtrendr.fit_trajectory_batch(v, LT_YEARS, _lt_params(_rust, **kw), -9999.0, n_jobs)
        for x, y in zip(a, b):
            assert_same_bits(x, y)


@pytest.mark.parametrize("setting", range(3))
def test_landtrendr_single(setting):
    kw = LT_SETTINGS[setting]
    v = _lt_series(10 + setting, 150)
    v[v == -9999.0] = np.nan
    for y in v:
        a = _core.landtrendr.fit_trajectory(LT_YEARS.tolist(), y.tolist(), _lt_params(_core, **kw))
        b = _rust.landtrendr.fit_trajectory(LT_YEARS.tolist(), y.tolist(), _lt_params(_rust, **kw))
        assert_same_bits(np.array([(x.year, x.value) for x in a], dtype=float).reshape(-1, 2),
                         np.array([(x.year, x.value) for x in b], dtype=float).reshape(-1, 2))
    pc, pr = _lt_params(_core), _lt_params(_rust)
    for name in ("max_segments", "pval_threshold", "prevent_fast_recovery", "recovery_threshold", "spike_threshold",
                 "best_model_proportion", "vertex_count_overshoot", "min_observations_needed", "modifier"):
        assert getattr(pc, name) == getattr(pr, name)
    assert (_rust.landtrendr.Vertex(2000, 0.5).year, _rust.landtrendr.Vertex(2000, 0.5).value) == (2000, 0.5)


def test_landtrendr_desawtooth():
    for y in _lt_series(20, 200):
        for stopat in (0.9, 0.75, 1.0):
            assert_same_bits(np.array(_core.landtrendr.desawtooth(y.tolist(), stopat)),
                             np.array(_rust.landtrendr.desawtooth(y.tolist(), stopat)))


def test_landtrendr_fit_to_vertices():
    rng = np.random.default_rng(21)
    v = _lt_series(21)
    seg = _core.landtrendr.fit_trajectory_batch(v.reshape(20, 30, -1), LT_YEARS, _lt_params(_core), -9999.0, -1)
    vertex_years, counts = seg[0][:, :, 0].astype(np.int32), seg[1].astype(np.int32)
    other = v + rng.normal(0, 0.02, v.shape)
    other[other < -9000] = np.nan
    for spike in (0.9, 1.0):
        for n_jobs in (1, -1):
            assert_same_bits(_core.landtrendr.fit_to_vertices_batch(other, LT_YEARS, vertex_years, counts, spike, n_jobs),
                             _rust.landtrendr.fit_to_vertices_batch(other, LT_YEARS, vertex_years, counts, spike, n_jobs))
    for y in other[:200]:   # vertex years anywhere, observed or not
        vys = rng.choice(LT_YEARS, rng.integers(0, 7), replace=False).tolist()
        assert_same_bits(np.array(_core.landtrendr.fit_to_vertices(LT_YEARS.tolist(), y.tolist(), vys, 0.9)),
                         np.array(_rust.landtrendr.fit_to_vertices(LT_YEARS.tolist(), y.tolist(), vys, 0.9)))


# ----------------------------------------------------------------------- CCDC
def _ccdc_stack(seed, n_pixels, n_bands=6, years=14, step=16):
    """Landsat-like stacks ([pixels, bands, time] reflectance x 10000, Fmask QA):
    stable, abrupt breaks (one or two), gradual change, outliers that Tmask must
    catch, clouds, shadows, water, snow-covered and cloud-covered pixels (the
    original's two fallback branches), short and sparse series, out-of-range
    and saturated values, and dates without an observation."""
    rng = np.random.default_rng(seed)
    dates = np.arange(730120, 730120 + int(years * 365.25), step, dtype=np.int32)
    dates = np.sort(dates + rng.integers(-3, 4, dates.size)).astype(np.int32)
    t = dates.astype(float) + 366
    w = 2 * np.pi / 365.25
    T = dates.size
    v = np.empty((n_pixels, n_bands, T))
    qa = np.zeros((n_pixels, T), dtype=np.int32)
    base = np.array([400, 700, 600, 2500, 1800, 1000, 2800][:n_bands], dtype=float)
    for k in range(n_pixels):
        kind = k % 12
        amp = rng.uniform(50, 600, n_bands)
        ph = rng.uniform(0, 2 * np.pi, n_bands)
        y = (base[:, None] * rng.uniform(0.6, 1.4) + amp[:, None] * np.sin(w * t[None] + ph[:, None])
             + 0.3 * amp[:, None] * np.cos(2 * w * t[None] + ph[:, None])
             + rng.normal(0, rng.uniform(10, 80), (n_bands, T)))
        if kind in (1, 2, 9):                                     # abrupt breaks
            for _ in range(1 if kind != 2 else 2):
                at = rng.integers(T // 5, 4 * T // 5)
                y[:, at:] += rng.uniform(-1, 1, (n_bands, 1)) * rng.uniform(300, 2500)
        if kind == 3:                                             # gradual change
            y += rng.uniform(-0.15, 0.15, (n_bands, 1)) * (t - t[0])[None]
        q = np.where(rng.random(T) < rng.uniform(0.05, 0.4), 4, 0)
        q[rng.random(T) < 0.05] = 2
        q[rng.random(T) < 0.03] = 1
        q[rng.random(T) < 0.03] = 3
        y[:4, q == 4] += rng.uniform(1500, 5000)                  # flagged clouds
        y[3:, q == 2] *= 0.4                                      # flagged shadows
        miss = rng.random(T) < 0.05                               # undetected clouds and shadows
        y[:3, miss] += rng.uniform(800, 3000)
        miss = rng.random(T) < 0.03
        y[4:6, miss] *= 0.3
        if kind == 4:                                             # mostly snow
            q = np.where(rng.random(T) < 0.85, 3, q)
            y[:3, q == 3] = rng.uniform(4000, 9000)
        if kind == 5:                                             # mostly cloud
            q = np.where(rng.random(T) < 0.85, 4, q)
        if kind == 6:                                             # short series
            q[rng.integers(14, 40):] = 255
        if kind == 7:                                             # few clear observations
            q = np.where(rng.random(T) < 0.7, 4, q)
        if kind == 8:                                             # out of range and saturated
            y[:, rng.random(T) < 0.05] = -50
            y[1, rng.random(T) < 0.05] = 16000
        if kind == 10:                                            # half of the dates missing
            q[rng.random(T) < 0.55] = 255
        if kind == 11:                                            # integer values
            y = np.round(y)
        if n_bands == 7:                                          # brightness temperature, deg C x 100
            y[6] = 1500 + 1200 * np.sin(w * t + rng.uniform(0, 6)) + rng.normal(0, 100, T)
            y[6, q == 4] -= 2500
            if kind == 8:
                y[6, rng.random(T) < 0.05] = -9500
        v[k], qa[k] = y, q
    gone = rng.random((n_pixels, T)) < 0.04                       # no observation (batch)
    v[np.repeat(gone[:, None], n_bands, axis=1)] = np.nan
    v[::37] = np.nan                                              # empty pixels
    v[1::41] = 0.0
    return dates, v, qa


def _ccdc_params(mod, **kw):
    p = mod.ccdc.CCDCParams()
    for k, val in kw.items():
        setattr(p, k, val)
    return p


CCDC_SETTINGS = [
    (6, dict()),
    (6, dict(num_c=4)),
    (6, dict(num_c=6, conseq_anom=4)),
    (6, dict(conseq_anom=8, chi2_prob_threshold=0.95, tmax_cg_prob_threshold=0.9999)),
    (6, dict(detection_bands=[2, 3, 4], tmask_bands=[0, 5])),
    (6, dict(detection_bands=[3], num_c=5, valid_min=-100.0, valid_max=12000.0)),
    (7, dict(thermal_band=6, detection_bands=[1, 2, 3, 4, 5], tmask_bands=[1, 4])),
    (7, dict(thermal_band=6, detection_bands=[0, 1, 2, 3, 4, 5, 6], num_c=6, thermal_min=-5000.0)),
    (7, dict()),
    (5, dict(conseq_anom=3)),
]


def _segments_array(segs):
    rows = [[s.t_start, s.t_end, s.t_break, s.change_prob, s.category, s.num_obs]
            + list(s.rmse) + list(s.magnitude) + [c for band in s.coefs for c in band] for s in segs]
    return np.array(rows, dtype=float) if rows else np.zeros((0, 0))


@pytest.mark.parametrize("setting", range(len(CCDC_SETTINGS)))
def test_ccdc_batch(setting):
    n_bands, kw = CCDC_SETTINGS[setting]
    dates, v, qa = _ccdc_stack(setting, 240, n_bands, years=[14, 9, 20, 6][setting % 4], step=[16, 8][setting % 2])
    v = v.reshape(12, 20, n_bands, -1)
    qa = qa.reshape(12, 20, -1)
    for return_coefs in (True, False):
        for n_jobs in (1, -1):
            a = _core.ccdc.fit_ccdc_batch(v, qa, dates, _ccdc_params(_core, **kw), 6, return_coefs, n_jobs)
            b = _rust.ccdc.fit_ccdc_batch(v, qa, dates, _ccdc_params(_rust, **kw), 6, return_coefs, n_jobs)
            for x, y in zip(a, b):
                assert_same_bits(x, y)
    a = _core.ccdc.fit_ccdc_batch(v, qa, dates, _ccdc_params(_core, **kw), 2, True, -1)
    b = _rust.ccdc.fit_ccdc_batch(v, qa, dates, _ccdc_params(_rust, **kw), 2, True, -1)
    for x, y in zip(a, b):
        assert_same_bits(x, y)


@pytest.mark.parametrize("setting", [0, 2, 3, 6, 9])
def test_ccdc_single(setting):
    n_bands, kw = CCDC_SETTINGS[setting]
    dates, v, qa = _ccdc_stack(100 + setting, 72, n_bands)
    rng = np.random.default_rng(setting)
    for k in range(v.shape[0]):
        order = rng.permutation(dates.size) if k % 3 == 0 else np.arange(dates.size)   # unsorted dates
        d, x, q = dates[order].copy(), v[k][:, order], qa[k][order]
        if k % 5 == 0:
            d[5] = d[6]                                           # a repeated date
        x = np.nan_to_num(x, nan=-1.0)
        a = _core.ccdc.fit_ccdc(d.tolist(), x.tolist(), q.tolist(), _ccdc_params(_core, **kw))
        b = _rust.ccdc.fit_ccdc(d.tolist(), x.tolist(), q.tolist(), _ccdc_params(_rust, **kw))
        assert_same_bits(_segments_array(a), _segments_array(b))


def test_ccdc_api():
    pc, pr = _core.ccdc.CCDCParams(), _rust.ccdc.CCDCParams()
    for name in ("min_obs", "conseq_anom", "chi2_prob_threshold", "tmax_cg_prob_threshold", "detection_bands",
                 "num_c", "tmask_bands", "thermal_band", "valid_min", "valid_max", "thermal_min", "thermal_max"):
        assert getattr(pc, name) == getattr(pr, name)
    pr.detection_bands = [1, 2]
    assert pr.detection_bands == [1, 2]
    s = _rust.ccdc.CCDCSegment()
    s.coefs, s.t_break = [[1.0] * 8], 5
    assert (s.coefs, s.t_break, s.rmse, s.change_prob) == ([[1.0] * 8], 5, [], 0.0)
    dates, v, qa = _ccdc_stack(3, 2)
    x = np.nan_to_num(v[0], nan=1.0).tolist()
    assert _rust.ccdc.fit_ccdc([], x, []) == _core.ccdc.fit_ccdc([], x, []) == []
    for mod in (_core, _rust):
        for p, msg in [(dict(tmask_bands=[1]), "exactly 2"), (dict(detection_bands=[6]), "detection band"),
                       (dict(tmask_bands=[1, -1]), "tmask band")]:
            with pytest.raises(ValueError, match=msg):
                mod.ccdc.fit_ccdc(dates.tolist(), x, qa[0].tolist(), _ccdc_params(mod, **p))
            with pytest.raises(ValueError, match=msg):
                mod.ccdc.fit_ccdc_batch(v.reshape(1, 2, 6, -1), qa.reshape(1, 2, -1), dates, _ccdc_params(mod, **p))
        with pytest.raises(ValueError, match="same length"):
            mod.ccdc.fit_ccdc(dates.tolist(), x, qa[0, 1:].tolist())
        with pytest.raises(ValueError, match="one value per date"):
            mod.ccdc.fit_ccdc(dates.tolist(), [x[0][1:]] + x[1:], qa[0].tolist())


# ---------------------------------------------------------------- warp
WARP_METHODS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14)
WARP_DTYPES = (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float32,
               np.float64)
# destination pixel -> source pixel (x = c + a u + b v, y = d + e u + f v) and GDAL's scales
WARP_GEOMETRIES = {
    "up": ((1.3, 0.37, 0.0, 2.1, 0.0, 0.41), (1 / 0.37, 1 / 0.41)),
    "down": ((-3.0, 2.3, 0.0, -2.5, 0.0, 1.7), (1 / 2.3, 1 / 1.7)),
    "rotated": ((4.0, 0.9, 0.25, -3.0, -0.2, 1.05), (1.0, 1.0)),
    "same": ((0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0)),
}
_IDENTITY = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_NO_LATTICE = (np.empty(0, np.int8), np.empty((0, 2)), np.empty(0, np.int64), np.empty((0, 2)), (1, 0, 0))
_NO_DENSE = (np.empty((0, 0, 2)), 0, 0)
_NO_TRANSFORMER = (0, np.empty(0, dtype=np.uintp))


def _warp_values(dtype, shape, seed):
    """Values of a raster type: its whole range at a few cells, repeats (for the modes), a NoData value (7) and
    the destination NoData (3) here and there."""
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        v = rng.normal(100, 40, shape).astype(dtype)
        v[rng.random(shape) < 0.3] = 42.0
        v.flat[::97] = np.finfo(dtype).max
        v.flat[5::89] = -np.finfo(dtype).max / 2
    else:
        info = np.iinfo(dtype)
        lo, hi = max(info.min, -(2 ** 62)), min(info.max, 2 ** 62)
        v = rng.integers(lo, hi, shape, dtype=dtype, endpoint=True)
        small = rng.integers(max(info.min, -3), min(info.max, 12), shape, dtype=dtype, endpoint=True)
        v = np.where(rng.random(shape) < 0.5, small, v)
        v.flat[::97] = info.max
        v.flat[5::89] = info.min
    v.flat[3::13] = 7
    v.flat[1::11] = 3
    return np.ascontiguousarray(v)


def _warp_pair(*args, **kw):
    a = _core.warp.warp(*args, **kw)
    b = _rust.warp.warp(*args, **kw)
    assert_same_bits(a, b)
    return a


def _affine_args(geometry, method):
    gt, scales = WARP_GEOMETRIES[geometry]
    offset = 0.0 if method >= 5 else 0.5
    return scales, (0, offset, gt, _IDENTITY) + _NO_LATTICE + _NO_DENSE + _NO_TRANSFORMER


@pytest.mark.parametrize("geometry", sorted(WARP_GEOMETRIES))
@pytest.mark.parametrize("method", WARP_METHODS)
def test_warp_methods_and_types(method, geometry):
    scales, coords = _affine_args(geometry, method)
    for k, dtype in enumerate(WARP_DTYPES):
        nan = np.dtype(dtype).kind == "f"
        for bands in (1, 3):
            src = _warp_values(dtype, (bands, 40, 36), 100 * method + k)
            for has_nodata, src_nodata in ((False, 0.0), (True, 7.0), (True, float("nan") if nan else -1.0)):
                for per_band in (True, False):
                    for dst_nodata in (3.0, 0.0):
                        for n_threads in (1, 3):
                            _warp_pair(src, method, (40, 36), (0, 0), scales, has_nodata, src_nodata, dst_nodata,
                                       (0, 0, 30, 28), *coords, per_band=per_band, n_threads=n_threads)


@pytest.mark.parametrize("method", WARP_METHODS)
def test_warp_window_and_region(method):
    """A window of the source and a region of the destination away from both origins."""
    gt = (12.0, 0.8, 0.05, 9.0, -0.04, 0.7)
    offset = 0.0 if method >= 5 else 0.5
    coords = (0, offset, gt, _IDENTITY) + _NO_LATTICE + _NO_DENSE + _NO_TRANSFORMER
    for dtype in (np.uint8, np.int16, np.float32, np.float64):
        whole = _warp_values(dtype, (2, 70, 64), method)
        src = np.ascontiguousarray(whole[:, 4:66, 3:60])
        for has_nodata in (False, True):
            for per_band in (True, False):
                a = _warp_pair(src, method, (70, 64), (4, 3), (1.25, 1.4), has_nodata, 7.0, 3.0, (5, 6, 33, 31),
                               *coords, per_band=per_band, n_threads=2)
                b = _rust.warp.warp(whole, method, (70, 64), (0, 0), (1.25, 1.4), has_nodata, 7.0, 3.0,
                                    (5, 6, 33, 31), *coords, per_band=per_band, n_threads=2)
                assert_same_bits(a, b)  # any window holding the cells needed gives the same values


def _lattice(cells_y, cells_x, seed, cell_x0=1, cell_y0=2):
    """A lattice with cells at every level, its nodes on a smooth map with points that fail (NaN) here and there."""
    rng = np.random.default_rng(seed)
    level = rng.integers(0, 5, cells_y * cells_x).astype(np.int8)

    def at(u, v):
        return 3.0 + 0.45 * u + 0.03 * v + 0.0004 * u * v, -2.0 + 0.02 * u + 0.5 * v - 0.0003 * u * u

    cu = (cell_x0 + np.arange(cells_x + 1)) * 16 + 0.5
    cv = (cell_y0 + np.arange(cells_y + 1)) * 16 + 0.5
    grid0 = np.stack(at(cu[None, :], cv[:, None]), axis=-1)
    node_off = np.zeros(level.size, np.int64)
    blocks, used = [], 0
    for c, lev in enumerate(level):
        if lev == 0:
            continue
        cy, cx = divmod(c, cells_x)
        steps = np.arange(0, 17, 16 >> lev) + 0.5
        block = np.stack(at((cell_x0 + cx) * 16 + steps[None, :], (cell_y0 + cy) * 16 + steps[:, None]),
                         axis=-1).reshape(-1, 2)
        if c % 7 == 3:
            block[len(block) // 2] = np.nan
        node_off[c] = used
        used += len(block)
        blocks.append(block)
    nodes = np.concatenate(blocks) if blocks else np.empty((0, 2))
    return level, np.ascontiguousarray(grid0), node_off, np.ascontiguousarray(nodes), (cells_x, cell_x0, cell_y0)


@pytest.mark.parametrize("method", WARP_METHODS)
def test_warp_lattice_and_dense(method):
    offset = 0.0 if method >= 5 else 0.5
    lat = _lattice(4, 5, method)
    r0, c0, h, w = 2 * 16 + 3, 16 + 5, 30, 50
    rng = np.random.default_rng(method)
    dense = np.stack(np.meshgrid(np.arange(c0, c0 + w + 2) * 0.8 - 12.3, np.arange(r0, r0 + h + 2) * 0.9 - 20.1),
                     axis=-1) + rng.normal(0, 0.2, (h + 2, w + 2, 2))
    dense[rng.random(dense.shape[:2]) < 0.05] = np.nan
    dense[5, 7] = np.inf
    for dtype in (np.uint8, np.uint16, np.int32, np.float32, np.float64):
        src = _warp_values(dtype, (2, 45, 50), 7 * method)
        lattice = (1, offset, _IDENTITY, _IDENTITY) + lat + _NO_DENSE + _NO_TRANSFORMER
        dense_coords = (2, offset, _IDENTITY, _IDENTITY) + _NO_LATTICE + (np.ascontiguousarray(dense), r0, c0) \
            + _NO_TRANSFORMER
        for has_nodata in (False, True):
            _warp_pair(src, method, (45, 50), (0, 0), (2.2, 2.0), has_nodata, 7.0, 3.0, (r0, c0, h, w), *lattice,
                       per_band=True, n_threads=3)
            _warp_pair(src, method, (45, 50), (0, 0), (1.2, 1.1), has_nodata, 7.0, 3.0, (r0, c0, h, w),
                       *dense_coords, per_band=False, n_threads=2)


def _transformer(width):
    """A ctypes GDALTransformerFunc: destination to source wrapping around the source's columns (an antimeridian),
    failing in a corner; source to destination failing past column width - 3 (the edge of a projection's domain)."""
    import ctypes

    p_double = ctypes.POINTER(ctypes.c_double)
    func = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, p_double, p_double, p_double,
                            ctypes.POINTER(ctypes.c_int))

    def transform(arg, dst_to_src, n, x, y, z, ok):
        xs = np.ctypeslib.as_array(x, (n,))
        ys = np.ctypeslib.as_array(y, (n,))
        oks = np.ctypeslib.as_array(ok, (n,))
        if dst_to_src:
            u, v = xs.copy(), ys.copy()
            xs[:] = np.mod(0.93 * u + 0.05 * v - 4.0, width)
            ys[:] = 0.88 * v + 0.03 * u - 1.5
            oks[:] = ~((u > 40) & (v > 30))
        else:
            oks[:] = xs < width - 3
            xs[:] = (xs + 4.0) / 0.93
            ys[:] = (ys + 1.5) / 0.88
        return 1

    cb = func(transform)
    return cb, ctypes.cast(cb, ctypes.c_void_p).value


@pytest.mark.parametrize("method", WARP_METHODS)
def test_warp_exact_with_a_transformer(method):
    offset = 0.0 if method >= 5 else 0.5
    cb, address = _transformer(36)
    args = np.arange(1, 5, dtype=np.uintp)  # one per thread: GDAL's own, here mere tokens
    for kind in (3, 0):  # exact; affine with the transformer for the edge checks
        coords = (kind, offset, (-2.0, 0.93, 0.0, -1.5, 0.0, 0.88), _IDENTITY) + _NO_LATTICE + _NO_DENSE \
            + (address, args)
        for dtype in (np.uint8, np.int16, np.float64):
            src = _warp_values(dtype, (3, 40, 36), method)
            for has_nodata, per_band in ((True, True), (True, False), (False, False)):
                for n_threads in (1, 4):
                    _warp_pair(src, method, (40, 36), (0, 0), (1.05, 1.1), has_nodata, 7.0, 3.0, (0, 0, 44, 41),
                               *coords, per_band=per_band, n_threads=n_threads)
        assert_same_bits(_core.warp.coords(*coords, (0, 0, 45, 42)), _rust.warp.coords(*coords, (0, 0, 45, 42)))
    del cb


def test_warp_coords():
    cases = [
        (0, 0.5, WARP_GEOMETRIES["rotated"][0], _IDENTITY) + _NO_LATTICE + _NO_DENSE + _NO_TRANSFORMER,
        (1, 0.0, _IDENTITY, _IDENTITY) + _lattice(3, 4, 5) + _NO_DENSE + _NO_TRANSFORMER,
        (2, 0.5, _IDENTITY, _IDENTITY) + _NO_LATTICE + (np.random.default_rng(3).normal(size=(60, 70, 2)), 30, 14)
        + _NO_TRANSFORMER,
    ]
    for case in cases:
        for region in ((2 * 16, 16, 3 * 16, 4 * 16), (40, 20, 7, 33), (32, 16, 0, 5)):
            assert_same_bits(_core.warp.coords(*case, region), _rust.warp.coords(*case, region))


def test_warp_lattice_checks():
    rng = np.random.default_rng(8)
    u, v = np.meshgrid(np.arange(9) * 8.0, np.arange(7) * 8.0)
    g = np.stack([3 + 0.4 * u + 0.002 * u * v, -5 + 0.6 * v - 0.003 * u * u], axis=-1)
    g = np.repeat(g[None], 5, axis=0) + rng.normal(0, 0.3, (5, 7, 9, 2))
    g[1, 3, 3, 0] = np.nan
    g[2, 0, :, 1] = np.inf
    g[3] += 500  # outside the source
    g[4] = np.nan
    g[0, 2, 2] = [-0.0, 0.0]
    for n_threads in (1, 3):
        a = _core.warp.lattice_checks(g, 40, 30, n_threads)
        b = _rust.warp.lattice_checks(g, 40, 30, n_threads=n_threads)
        for x, y in zip(a, b):
            assert_same_bits(x, y)


def test_warp_errors():
    scales, coords = _affine_args("same", 0)
    names = ("kind", "offset", "gt", "inv", "level", "grid0", "node_off", "nodes", "cells", "dense", "dense_row0",
             "dense_col0", "transformer", "transformer_args")
    good = dict(zip(names, coords), src=np.zeros((1, 10, 10), np.uint8), method=0, src_shape=(10, 10),
                window=(0, 0), scales=scales, has_src_nodata=False, src_nodata=0.0, dst_nodata=0.0,
                region=(0, 0, 5, 5))
    lat = dict(zip(names[4:9], _lattice(2, 2, 1)), kind=1)
    bad = [dict(src=np.zeros((10, 10))), dict(src=np.zeros((1, 10, 10))[:, ::2]), dict(method=7), dict(method=15),
           dict(region=(0, 0, -1, 5)), dict(window=(1, 0)), dict(scales=(0.0, 1.0)), dict(kind=5), dict(kind=3),
           dict(kind=2), dict(src=np.zeros((1, 10, 10), bool)), dict(src=np.zeros((1, 10, 10), np.float16)),
           dict(transformer=12345), dict(lat, cells=(0, 0, 0)), dict(lat, region=(0, 0, 40, 5)),
           dict(lat, node_off=np.zeros(3, np.int64)), dict(lat, grid0=np.zeros((4, 2))),
           dict(lat, level=np.full(4, 5, np.int8)), dict(lat, node_off=np.full(4, 10 ** 6, np.int64)),
           dict(kind=2, dense=np.zeros((10, 10, 3))), dict(kind=2, dense=np.zeros((3, 3, 2))),
           dict(transformer=12345, transformer_args=np.arange(1, 3, dtype=np.uintp), n_threads=3)]
    for change in bad:
        kw = dict(good, **change)
        with pytest.raises((ValueError, TypeError)) as a:
            _core.warp.warp(**kw)
        with pytest.raises((ValueError, TypeError)) as b:
            _rust.warp.warp(**kw)
        assert (a.type, str(a.value)) == (b.type, str(b.value)), change
    for mod in (_core, _rust):
        with pytest.raises(ValueError, match="bad region"):
            mod.warp.coords(*coords, (0, -1, 3, 3))
        with pytest.raises(ValueError, match="g must be"):
            mod.warp.lattice_checks(np.zeros((1, 4, 3, 2)), 10, 10)



# ----------------------------------------------------------------- Phenology
def _ph_series(seed, n_pixels=150, years=3, step=16):
    """Double-logistic seasons with noise and cloud dips, and the edge cases:
    gaps, nodata, empty, flat, low-amplitude and infinite rows."""
    rng = np.random.default_rng(seed)
    n = years * (368 // step)
    t = 1.0 + step * np.arange(n, dtype=float)
    doy = (t - 1) % 365.25
    shape = (n_pixels, 1)
    base, amp = rng.uniform(0.1, 0.3, shape), rng.uniform(0.2, 0.6, shape)
    sos, eos = rng.uniform(80, 160, shape), rng.uniform(230, 320, shape)
    rsp, rau = rng.uniform(0.04, 0.15, shape), rng.uniform(0.03, 0.12, shape)
    v = base + amp * (1 / (1 + np.exp(-rsp * (doy - sos))) - 1 / (1 + np.exp(-rau * (doy - eos))))
    v += rng.normal(0, 0.03, v.shape)
    cloud = rng.random(v.shape) < 0.1
    v[cloud] -= rng.uniform(0.05, 0.3, cloud.sum())
    v[1::11, rng.integers(0, n, 3)] = np.nan       # gaps
    v[2::13, rng.integers(0, n, 2)] = -9999.0      # nodata, fitted as a value
    v[3::17] = np.nan                              # empty
    v[4::19] = 0.4                                 # flat
    v[5::23] *= 0.1                                # below min_pixel_amplitude
    v[6::29, n // 2] = np.inf
    return v, t


def _ph_weights(seed, shape):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0, 1, shape)
    w[rng.random(shape) < 0.1] = 0.0
    w[rng.random(shape) < 0.05] = np.nan           # taken as 1
    w[rng.random(shape) < 0.05] = 1.7              # clamped to 1
    return w


def _ph_both(values, dates, curve, **kw):
    a = _core.phenology.fit_phenology_batch(values, dates, curve, **kw)
    b = _rust.phenology.fit_phenology_batch(values, dates, curve, **kw)
    assert_same_bits(a, b)
    return a


@pytest.mark.parametrize("curve", range(7))
def test_phenology_curves(curve):
    v, t = _ph_series(curve)
    a = _ph_both(v, t, curve, max_seasons=3, n_jobs=-1)
    assert np.isfinite(a).sum() > 0.5 * a.size
    w = _ph_weights(curve, v.shape)
    _ph_both(v, t, curve, max_seasons=3, n_jobs=1, weights_array=w)


PH_SETTINGS = [
    dict(apply_whittaker=False, apply_hants=True),
    dict(apply_whittaker=False, apply_hants=True, hants_frequencies=0, hants_threshold=0.05),
    dict(apply_whittaker=False, apply_hants=True, hants_frequencies=2, hants_threshold=0.2),
    dict(apply_whittaker=False),                   # the raw series: NaN within seasons
    dict(whittaker_lambda=0.0),
    dict(whittaker_lambda=1000.0),
    dict(min_season_length=150, min_amplitude=0.15),
    dict(season_retry=False, rtrough_max=0.3, r_min_filter=0.1),
    dict(rtrough_max=0.9, min_pixel_amplitude=0.0),
    dict(max_seasons=0),
    dict(max_seasons=1),
    dict(max_seasons=6),
]


@pytest.mark.parametrize("setting", range(len(PH_SETTINGS)))
@pytest.mark.parametrize("curve", [0, 2, 5])
def test_phenology_settings(setting, curve):
    v, t = _ph_series(100 + setting, n_pixels=80)
    kw = dict(max_seasons=3, n_jobs=-1) | PH_SETTINGS[setting]
    _ph_both(v, t, curve, **kw)
    _ph_both(v, t, curve, weights_array=_ph_weights(setting, v.shape), **kw)


@pytest.mark.parametrize("step,years", [(8, 2), (1, 1), (1, 3), (30, 6)])
def test_phenology_cadences(step, years):
    v, t = _ph_series(7, n_pixels=40, years=years, step=step)
    _ph_both(v, t, 1, max_seasons=4)
    if len(t) <= 400:   # HANTS's X^T X: deeper, Eigen's blocks follow the machine's L1 cache
        _ph_both(v, t, 6, max_seasons=4, apply_whittaker=False, apply_hants=True)


def test_phenology_short_series_and_errors():
    for n in (0, 1, 2, 3):
        v = np.linspace(0.2, 0.6, 5 * n).reshape(5, n) if n else np.zeros((5, 0))
        _ph_both(v, np.arange(n, dtype=float), 0, max_seasons=2)
    v, t = _ph_series(3, n_pixels=20)
    _ph_both(v, t, 9, max_seasons=2)   # no such curve: no season fitted
    assert_same_bits(_core.phenology.fit_phenology_batch(v, t, _core.phenology.CurveType.ZHANG),   # the enums as ints
                     _rust.phenology.fit_phenology_batch(v, t, _rust.phenology.CurveType.ZHANG))
    for f in (_core.phenology.fit_phenology_batch, _rust.phenology.fit_phenology_batch):
        for args in ((v[0], t, 0), (v, t[:, None], 0), (v, t[:-1], 0)):
            with pytest.raises(RuntimeError):
                f(*args)
        with pytest.raises(RuntimeError):
            f(v, t, 0, weights_array=np.ones((2, 2)))
    assert ({k: int(x) for k, x in _core.phenology.CurveType.__members__.items()}
            == {k: int(x) for k, x in _rust.phenology.CurveType.__members__.items()})
    assert ({k: int(x) for k, x in _core.phenology.ExtractionMethod.__members__.items()}
            == {k: int(x) for k, x in _rust.phenology.ExtractionMethod.__members__.items()})
    assert int(_core.phenology.GU) == int(_rust.phenology.GU) == 2


def test_phenology_split_seasons():
    v, t = _ph_series(11, n_pixels=300)
    kw_sets = [dict(), dict(min_season_length=120, min_amplitude=0.1), dict(rtrough_max=0.3, retry_on_empty=False),
               dict(r_min_filter=0.2, rtrough_max=0.8)]
    for y in v[:, :]:
        if not np.isfinite(y).any():
            continue
        y = np.where(np.isfinite(y), y, 0.0)
        for kw in kw_sets:
            for dates in (None, t.tolist()):
                a = _core.phenology.debug_split_seasons(y.tolist(), dates, **kw)
                b = _rust.phenology.debug_split_seasons(y.tolist(), dates, **kw)
                assert a == b
    for f in (_core.phenology.debug_split_seasons, _rust.phenology.debug_split_seasons):
        with pytest.raises(RuntimeError):
            f([0.1, 0.5, 0.2], [1.0, 2.0])
