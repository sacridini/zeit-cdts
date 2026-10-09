"""Cases for the TWDTW parity with R's twdtw: series and patterns (written to cases.json)."""
import json
import pathlib

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).parent  # cases.json here; then run make_twdtw_parity_fixtures.R
rng = np.random.default_rng(5)


def season(dates, peak_doy, low=0.2, high=0.8, width=60.0):
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    d = np.minimum(np.abs(doy - peak_doy), 366 - np.abs(doy - peak_doy))
    return low + (high - low) * np.exp(-0.5 * (d / width) ** 2)


cases = []


def add(name, series_dates, series_values, patterns, steep=0.1, mid=50.0):
    cases.append(dict(name=name, steep=steep, mid=mid,
                      series=dict(time=[str(d.date()) for d in series_dates],
                                  values=np.asarray(series_values).round(5).tolist()),
                      patterns={k: dict(time=[str(d.date()) for d in t], values=np.asarray(v).round(5).tolist())
                                for k, (t, v) in patterns.items()}))


p_dates = pd.date_range("2019-09-01", "2020-08-31", freq="16D")
pats = {"crop": (p_dates, season(p_dates, 60)), "forest": (p_dates, 0.75 + 0.03 * np.sin(np.arange(p_dates.size))),
        "pasture": (p_dates, season(p_dates, 330, 0.3, 0.6, 90))}
s_dates = pd.date_range("2021-09-03", "2022-08-31", freq="16D")
s = season(s_dates, 70) + rng.normal(0, 0.03, s_dates.size)
add("one_band_one_season", s_dates, s, pats)
s2 = s.copy()
s2[[3, 4, 10]] = np.nan
add("cloudy_dates_dropped", s_dates, s2, pats)
long_dates = pd.date_range("2018-01-05", "2021-12-28", freq="8D")
add("pattern_inside_a_long_series", long_dates, season(long_dates, 300) + rng.normal(0, 0.05, long_dates.size),
    pats, steep=0.05, mid=30.0)
two = {k: (t, np.stack([v, 1 - v + 0.1 * np.cos(np.arange(t.size))], 1)) for k, (t, v) in pats.items()}
add("two_bands", s_dates, np.stack([s, 1 - s + rng.normal(0, 0.02, s.size)], 1), two)
(here / "cases.json").write_text(json.dumps(cases).replace("NaN", "null"))
print(len(cases), "cases")
