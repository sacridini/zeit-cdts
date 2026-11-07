"""The radiative transfer pieces of FORCE's atmospheric correction (zeit._core.radtran.radtran_eval), ported from
FORCE 3.11. There is no public function yet; the comparison with FORCE's C code is benchmarks/vs_originals/force_radtran.py."""
import numpy as np
import pytest

from zeit import _core

ev = _core.radtran.radtran_eval
F32 = np.float32


def a(*v):
    return np.array(v, F32)


def test_molecular_optical_depth_decreases_with_wavelength():
    (mod,) = ev("molecular_optical_depth", a(0.45, 0.55, 0.65, 0.865))
    assert mod.dtype == F32
    assert np.all(np.diff(mod) < 0)
    assert mod[1] == pytest.approx(0.0985, abs=1e-4)


def test_scatt_transmitt_outputs():
    aod, mod = a(0.2, 0.0), a(0.1, 0.0)
    out = ev("scatt_transmitt", aod, mod, aod + mod, a(0.8, 0.7), a(0.95, 0.9))
    assert len(out) == 7
    total, ts, tv, tsd, tss, tvd, tvs = out
    np.testing.assert_array_equal(total, ts * tv)
    np.testing.assert_array_equal(tss, ts - tsd)
    np.testing.assert_array_equal(tvs, tv - tvd)
    # without atmosphere everything is direct
    assert (ts[1], tv[1], tsd[1], tvd[1], tss[1], tvs[1]) == (1, 1, 1, 1, 0, 0)


def test_band_tables():
    b = np.arange(56, dtype=F32)
    (wvl,) = ev("wavelength", b)
    (e0,) = ev("e0", b)
    # thermal bands have no response in 410-2400 nm
    assert wvl[6] == 0 and e0[6] == 0
    # Sentinel-2A: red (33) is near 0.665 um, nir (38) near 0.842, the water vapour band (39) near 0.945
    assert wvl[33] == pytest.approx(0.665, abs=0.01)
    assert wvl[39] == pytest.approx(0.945, abs=0.01)
    assert e0[33] > e0[38] > e0[39] > 0


def test_gas_transmittance_per_band():
    n = 3
    band = np.full(n, 39, F32)  # Sentinel-2A water vapour band
    (t,) = ev("wvp_transmitt", a(0.0, 2.0, 4.0), np.full(n, 1.0, F32), band)
    assert t[0] == pytest.approx(1.0, abs=1e-6) and t[0] > t[1] > t[2] > 0
    (o,) = ev("ozone_transmitt", a(0.0, 0.3), a(1.0, 1.0), np.full(2, 33, F32))
    assert o[0] == pytest.approx(1.0, abs=1e-6) and o[1] < 1


def test_sun_position_and_standard_zenith():
    # noon GMT on the summer solstice at 45 N, 0 E: zenith about 22 degrees, azimuth south
    zen, azi = ev("sunpos", a(45.0), a(0.0), a(2020), a(172), a(12), a(0), a(0), a(0))
    assert zen[0] == pytest.approx(21.6, abs=1.5) and azi[0] == pytest.approx(180.0, abs=10)
    (z,) = ev("standard_sunzenith", a(2021, 2021), a(172, 355), a(0, 0), a(45, 45), a(10, 10))
    assert 15 < z[0] < 40 and z[1] > z[0] + 30


def test_localtime2gmt():
    out = ev("localtime2gmt", a(2021, 2021), a(100, 100), a(10.5, 23.0), a(15.0, -30.0))
    year, month, day, doy, hh, mm, ss = (o.tolist() for o in out)
    assert (doy, hh, mm, ss) == ([100, 101], [9, 1], [30, 0], [0, 0])
    assert month == [4, 4] and day == [10, 11] and year == [2021, 2021]


def test_ozone_amount_equator_minimum():
    (o,) = ev("ozone_amount", a(0.0, 10.0), a(0.0, 50.0), a(100, 80))
    assert o[0] == F32(0.235) and 0.2 < o[1] < 0.5


def test_errors():
    with pytest.raises(ValueError, match="unknown"):
        ev("nope", a(1.0))
    with pytest.raises(ValueError, match="takes 2 arrays"):
        ev("optical_depth", a(1.0))
    with pytest.raises(ValueError, match="same length"):
        ev("optical_depth", a(1.0, 2.0), a(1.0))
    with pytest.raises(ValueError, match="band index"):
        ev("wavelength", a(56))
