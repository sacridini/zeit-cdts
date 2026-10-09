import numpy as np
from zeit.metrics import extract_events
from zeit._ccdc import run_ccdc

def test_extract_events():
    # Shape: (max_vertices * 2, rows, cols)
    # Let's say max_segments = 2, max_vertices = 3
    stack = np.zeros((6, 2, 2), dtype=np.float32)
    
    # Pixel 0,0: Year=[2000, 2005, 2010], Value=[0.8, 0.2, 0.9]
    # Loss event: 2000 to 2005. Mag = 0.6, dur = 5, rate = 0.12
    # Gain event: 2005 to 2010. Mag = 0.7, dur = 5, rate = 0.14
    stack[0, 0, 0] = 2000; stack[3, 0, 0] = 0.8
    stack[1, 0, 0] = 2005; stack[4, 0, 0] = 0.2
    stack[2, 0, 0] = 2010; stack[5, 0, 0] = 0.9
    
    # Test loss extraction
    loss_events = extract_events(stack, event_type="loss", sort_by="greatest")
    assert loss_events["yod"][0, 0] == 2000
    assert np.isclose(loss_events["magnitude"][0, 0], 0.6)
    assert loss_events["duration"][0, 0] == 5
    
    # Test gain extraction
    gain_events = extract_events(stack, event_type="gain", sort_by="greatest")
    assert gain_events["yod"][0, 0] == 2005
    assert np.isclose(gain_events["magnitude"][0, 0], 0.7)
    
def test_ccdc_basic():
    # Landsat-like pixel (surface reflectance x 10000, Blue..SWIR2, the scale
    # the original CCDC's lasso and range tests are defined on), observed every
    # 16 days for 8 years, with a forest-to-bare change halfway through.
    from datetime import date
    rng = np.random.default_rng(0)
    dates = np.arange(date(2000, 1, 1).toordinal(), date(2008, 1, 1).toordinal(), 16)
    w = 2 * np.pi / 365.25
    base = np.array([450.0, 750.0, 550.0, 3000.0, 1800.0, 900.0])
    values = base[:, None] * (1 + 0.15 * np.cos(w * dates[None, :]))
    values += rng.normal(0, 25, values.shape)
    t_change = len(dates) // 2
    values[:, t_change:] += np.array([150.0, 250.0, 450.0, -1300.0, 900.0, 700.0])[:, None]

    qa = np.zeros(len(dates), dtype=int)  # Fmask 0 = clear land

    segments = run_ccdc(dates, values, qa)

    assert len(segments) == 2
    assert segments[0]["t_break"] == dates[t_change]
    assert segments[0]["change_prob"] == 1
    assert segments[1]["t_start"] == dates[t_change]
    # coefs is a list of lists: coefs[band][8 harmonic coefficients]
    assert len(segments[0]["coefs"]) == 6
    assert len(segments[0]["coefs"][0]) == 8
    # NIR dropped by ~1300
    assert segments[0]["magnitude"][3] < -1000
