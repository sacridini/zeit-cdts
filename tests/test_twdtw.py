import pytest
import numpy as np
from zeit._twdtw import run_twdtw, run_twdtw_batch, classify_twdtw

def test_run_twdtw_single():
    ts_values = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1])
    ts_dates = np.array([1, 45, 90, 135, 180, 225])
    pat_values = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1])
    pat_dates = np.array([1, 45, 90, 135, 180, 225])

    dist = run_twdtw(ts_values, ts_dates, pat_values, pat_dates)
    assert isinstance(dist, float)
    assert dist > 0.0 and dist < 0.1

def test_run_twdtw_multivariate():
    # 2 bands
    ts_values = np.array([[0.1, 0.5], [0.8, 0.9], [0.1, 0.2]])
    ts_dates = np.array([10, 100, 200])
    
    pat_values = np.array([[0.1, 0.5], [0.8, 0.9], [0.1, 0.2]])
    pat_dates = np.array([10, 100, 200])
    
    dist = run_twdtw(ts_values, ts_dates, pat_values, pat_dates)
    assert dist > 0.0 and dist < 0.1

def test_run_twdtw_path():
    ts_values = [0.1, 0.8, 0.1]
    ts_dates = [10, 100, 200]
    pat_values = [0.1, 0.8, 0.1]
    pat_dates = [10, 100, 200]
    
    dist, path = run_twdtw(ts_values, ts_dates, pat_values, pat_dates, return_path=True)
    assert isinstance(path, list)
    assert len(path) >= 3
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_run_twdtw_subsequence():
    # Long time series
    ts_values = [0.1, 0.1, 0.1, 0.8, 0.9, 0.1, 0.1, 0.1]
    ts_dates = [10, 40, 70, 100, 130, 160, 190, 220]
    
    # Short pattern
    pat_values = [0.8, 0.9]
    pat_dates = [100, 130]
    
    dist_global = run_twdtw(ts_values, ts_dates, pat_values, pat_dates, subsequence_matching=False)
    dist_subseq = run_twdtw(ts_values, ts_dates, pat_values, pat_dates, subsequence_matching=True)
    
    assert dist_subseq < dist_global

def test_classify_twdtw():
    # 2x2 pixels, 3 time steps
    Y, X, T = 2, 2, 3
    values = np.zeros((Y, X, T), dtype=np.float64)
    dates = np.array([10, 50, 100], dtype=np.int32)
    
    # Pixel (0,0) looks like Soy
    values[0, 0, :] = [0.1, 0.9, 0.1]
    # Pixel (1,1) looks like Corn
    values[1, 1, :] = [0.2, 0.4, 0.8]
    
    patterns = {
        "Soy": (np.array([0.1, 0.9, 0.1]), dates),
        "Corn": (np.array([0.2, 0.4, 0.8]), dates)
    }
    
    classes_map, dist_map, names = classify_twdtw(values, dates, patterns)
    
    assert classes_map.shape == (2, 2)
    assert names[classes_map[0, 0]] == "Soy"
    assert names[classes_map[1, 1]] == "Corn"

def test_run_twdtw_batch_multivariate():
    Y, X, T, B = 2, 2, 3, 2
    values = np.random.rand(Y, X, T, B).astype(np.float64)
    dates = np.array([10, 50, 100], dtype=np.int32)
    pat = np.random.rand(T, B).astype(np.float64)
    
    res = run_twdtw_batch(values, dates, pat, dates)
    assert res.shape == (Y, X)
