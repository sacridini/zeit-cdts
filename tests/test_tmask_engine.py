"""The C++ Tmask engine: MATLAB's robustfit (as CCDC runs it) against a Python port of the
same algorithm, and the screening of injected clouds and shadows."""
import numpy as np
import pytest
import scipy.linalg

from zeit import _core
from zeit._tmask import apply_tmask_stack, run_tmask_pixel

W = 2 * np.pi / 365.25


def robustfit_reference(x, y):
    """MATLAB's robustfit (bisquare, tune 4.685, leverage-adjusted residuals, MAD scale of
    the residuals past the rank) stopped after 5 iterations, as CCDC's robustfit_cor."""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    p = X.shape[1]
    eps = np.finfo(float).eps
    q, r, _ = scipy.linalg.qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > max(n, p) * eps * diag.max()).sum())
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    h = (q[:, :rank] ** 2).sum(axis=1)
    adj = 1 / np.sqrt(1 - np.minimum(0.9999, h))
    tiny = 1e-6 * np.std(y, ddof=1) or 1.0
    b0, it = np.zeros(p), 1
    while np.any(np.abs(b - b0) > np.sqrt(eps) * np.maximum(np.abs(b), np.abs(b0))):
        it += 1
        if it > 5:
            break
        radj = (y - X @ b) * adj
        tail = np.sort(np.abs(radj))[max(1, rank) - 1:]
        scale = max(np.median(tail) / 0.6745, tiny) * 4.685
        u = radj / scale
        sw = np.sqrt(np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0))
        b0 = b
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return b


@pytest.mark.parametrize("seed", range(5))
def test_robustfit_is_matlabs(seed):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(0, 1500, 60))
    x = np.column_stack([days, np.cos(W * days), np.sin(W * days)])
    y = 0.1 + 1e-5 * days + 0.03 * np.cos(W * days) + rng.normal(0, 0.005, days.size)
    y[rng.random(days.size) < 0.15] += 0.3   # outliers
    got = np.asarray(_core.tmask.robustfit(x.ravel().tolist(), 3, y.tolist()))
    np.testing.assert_allclose(got, robustfit_reference(x, y), rtol=1e-8, atol=1e-12)


def series(rng, t=140):
    days = 737000.0 + 8.0 * np.arange(t)
    green = 900 + 200 * np.cos(W * days) + rng.normal(0, 40, t)
    swir = 1800 + 300 * np.sin(W * days) + rng.normal(0, 60, t)
    return days, green, swir


def test_injected_clouds_and_shadows_are_flagged():
    rng = np.random.default_rng(7)
    for _ in range(20):
        days, green, swir = series(rng)
        cloud = rng.random(days.size) < 0.15
        shadow = ~cloud & (rng.random(days.size) < 0.10)
        green[cloud] += rng.uniform(800, 3000, cloud.sum())
        swir[shadow] -= rng.uniform(800, 1200, shadow.sum())
        clear = run_tmask_pixel(days, green, swir)
        np.testing.assert_array_equal(clear, ~(cloud | shadow))


def test_clean_series_and_short_series_are_clear():
    rng = np.random.default_rng(8)
    days, green, swir = series(rng)
    assert run_tmask_pixel(days, green, swir).all()
    assert run_tmask_pixel(days[:4], green[:4] + [0, 5000, 0, 0], swir[:4]).all()   # too few to model


def test_the_stack_keeps_its_former_conventions():
    rng = np.random.default_rng(9)
    t, h, w = 40, 3, 4
    days = 737000.0 + 16.0 * np.arange(t)
    green = 900 + 200 * np.cos(W * days)[:, None, None] + rng.normal(0, 30, (t, h, w))
    swir = 1800 + 300 * np.sin(W * days)[:, None, None] + rng.normal(0, 30, (t, h, w))
    green[10] += 3000
    green[:, 0, 0] = 0             # no data: reported clear by the stack function, as before
    clear = apply_tmask_stack(days, green, swir)
    assert clear.shape == (t, h, w) and clear[:, 0, 0].all()
    assert not clear[10, 1:].any() and clear[11].all()
    one = run_tmask_pixel(days, green[:, 2, 3], swir[:, 2, 3])
    np.testing.assert_array_equal(clear[:, 2, 3], one)
    n_threads = _core.tmask.tmask_batch(np.ascontiguousarray(green.reshape(t, -1).T),
                                        np.ascontiguousarray(swir.reshape(t, -1).T), days, n_jobs=1)
    np.testing.assert_array_equal(n_threads, _core.tmask.tmask_batch(np.ascontiguousarray(green.reshape(t, -1).T),
                                                                      np.ascontiguousarray(swir.reshape(t, -1).T),
                                                                      days, n_jobs=4))
