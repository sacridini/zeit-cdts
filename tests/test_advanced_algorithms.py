import pytest
import numpy as np

from zeit.ai.som import SOM
from zeit._smooth import apply_whittaker_filter
from zeit.spatial import apply_bayesian_filter

def test_som_filter():
    # Create simple 1D dataset (3 classes)
    data = np.vstack([
        np.random.normal(0, 0.1, (50, 1)), # Class 0
        np.random.normal(5, 0.1, (50, 1)), # Class 1
        np.random.normal(10, 0.1, (50, 1)) # Class 2
    ])
    labels = np.array([0]*50 + [1]*50 + [2]*50)
    
    # Inject some noise
    labels[0] = 1 # Should be 0
    labels[55] = 2 # Should be 1
    
    som = SOM(x=5, y=5, input_len=1, random_seed=42)
    som.train(data, num_iters=100, algorithm="batch")
    
    clean_mask = som.filter_noisy_samples(data, labels)
    
    assert len(clean_mask) == 150
    # The noisy samples should be flagged as False
    assert clean_mask[0] == False
    assert clean_mask[55] == False
    
def test_whittaker_filter():
    cube = np.random.rand(10, 5, 5)
    
    # Should reduce variance/roughness
    smoothed = apply_whittaker_filter(cube, lmbd=50.0)
    
    assert smoothed.shape == cube.shape
    
    # A smoothed series should have a smaller sum of absolute differences
    diff_orig = np.sum(np.abs(np.diff(cube, axis=0)))
    diff_smooth = np.sum(np.abs(np.diff(smoothed, axis=0)))
    
    assert diff_smooth < diff_orig
    
def test_bayesian_filter():
    # 2 classes, 5x5 image
    probs = np.zeros((2, 5, 5))
    
    # Class 0 is dominant everywhere
    probs[0, :, :] = 0.8
    probs[1, :, :] = 0.2
    
    # Except for a noisy pixel at (2,2) where Class 1 spiked
    probs[0, 2, 2] = 0.4
    probs[1, 2, 2] = 0.6
    
    # A standard argmax would classify (2,2) as Class 1
    naive_class = np.argmax(probs, axis=0)
    assert naive_class[2, 2] == 1
    
    # Bayesian filter uses the neighborhood (which is heavily Class 0)
    # So it should override the slight spike of Class 1
    bayesian_class = apply_bayesian_filter(probs, window_size=3)
    
    # The noisy pixel should be smoothed out to Class 0
    assert bayesian_class[2, 2] == 0
