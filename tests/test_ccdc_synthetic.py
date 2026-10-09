import numpy as np
from zeit._ccdc import predict_synthetic_image

def test_predict_synthetic_image():
    # Shape: (max_segments=2, params_per_seg=3 + bands*9, rows=2, cols=2); with
    # 1 band params_per_seg = 3 + 9 = 12: t_start, t_end, t_break, rmse, 8 coefs
    ccdc_coefs_stack = np.zeros((2, 12, 2, 2), dtype=np.float32)

    # Segment 1: active from day 1 to 100
    ccdc_coefs_stack[0, 0, :, :] = 1   # t_start
    ccdc_coefs_stack[0, 1, :, :] = 100 # t_end

    # band 1 coefficients start at index 4 (0=start, 1=end, 2=break, 3=rmse)
    ccdc_coefs_stack[0, 4, :, :] = 500.0 # intercept

    # Synthetic image for day 50 (should fall in segment 1)
    synthetic = predict_synthetic_image(ccdc_coefs_stack, target_julian_day=50, num_bands=1)

    assert synthetic.shape == (1, 2, 2)

    # Because only intercept is set and terms[0] = 1, output should be exactly 500
    assert np.allclose(synthetic, 500.0)
