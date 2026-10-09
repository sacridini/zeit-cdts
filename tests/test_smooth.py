import numpy as np
from zeit._smooth import apply_savgol_filter

def test_savgol_filter():
    # Create a dummy time series (10 timesteps, 1 band, 2x2 pixels)
    # Add noise to a straight line
    time_series = np.array([1, 5, 2, 8, 3, 7, 4, 9, 5, 10], dtype=float)
    cube = np.zeros((10, 2, 2))
    for i in range(10):
        cube[i, :, :] = time_series[i]
        
    smoothed = apply_savgol_filter(cube, window_length=5, polyorder=2, axis=0)
    
    assert smoothed.shape == (10, 2, 2)
    # Check if variance is reduced (smoothed)
    assert np.var(smoothed[:, 0, 0]) < np.var(cube[:, 0, 0])
