//! Sun position (zenith and azimuth) from a place and a date with its time of day and time zone.
//!
//! Clean-room implementation from Meeus, Astronomical Algorithms (1998), ch. 7, 25, 28, and NOAA's published solar
//! calculation documentation; GPL-3.0-or-later (Zeit). It does not derive from FORCE's `sun-cl.c`.
//!
//! The solar coordinates are Meeus' low-accuracy ones (about 0.01 degree for 1900-2100); the time is UT with the
//! difference to dynamical time ignored; the refraction is the approximate one of NOAA's documentation.

use std::f64::consts::PI;

#[inline]
fn rad(x: f64) -> f64 {
    x * PI / 180.0
}

#[inline]
fn deg(x: f64) -> f64 {
    x * 180.0 / PI
}

#[inline]
fn mod360(x: f64) -> f64 {
    x - 360.0 * (x / 360.0).floor()
}

/// Julian day at 0h of a Gregorian calendar day (Meeus ch. 7); the Gregorian rule is always applied.
pub fn julian_day(year: i32, month: i32, day: i32) -> f64 {
    let (mut y, mut m) = (year as f64, month as f64);
    if month <= 2 {
        y -= 1.0;
        m += 12.0;
    }
    let a = (y / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y + 4716.0)).floor() + (30.6001 * (m + 1.0)).floor() + day as f64 + b - 1524.5
}

/// The apparent longitude of the sun and the declination (both degrees) at a Julian day, and the quantities the
/// equation of time needs: `(lambda_app, declination, eps, l0, m, e)` with `eps` the corrected obliquity (degrees),
/// `l0` the mean longitude and `m` the mean anomaly (degrees) and `e` the eccentricity.
pub fn solar_coordinates(jd: f64) -> (f64, f64, f64, f64, f64, f64) {
    let t = (jd - 2451545.0) / 36525.0;
    let l0 = mod360(280.46646 + t * (36000.76983 + t * 0.0003032));
    let m = mod360(357.52911 + t * (35999.05029 - 0.0001537 * t));
    let e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
    let mr = rad(m);
    let c = mr.sin() * (1.914602 - t * (0.004817 + 0.000014 * t))
        + (2.0 * mr).sin() * (0.019993 - 0.000101 * t)
        + (3.0 * mr).sin() * 0.000289;
    let true_lon = l0 + c;
    let omega = 125.04 - 1934.136 * t;
    let lambda_app = true_lon - 0.00569 - 0.00478 * rad(omega).sin();
    let eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
    let eps = eps0 + 0.00256 * rad(omega).cos();
    let s = (rad(eps).sin() * rad(lambda_app).sin()).clamp(-1.0, 1.0);
    (lambda_app, deg(s.asin()), eps, l0, m, e)
}

/// The equation of time in minutes (Meeus ch. 28; positive when the sundial is ahead of the mean clock).
pub fn equation_of_time(eps: f64, l0: f64, m: f64, e: f64) -> f64 {
    let y = (rad(eps) / 2.0).tan().powi(2);
    let (l0, m) = (rad(l0), rad(m));
    let er = y * (2.0 * l0).sin() - 2.0 * e * m.sin() + 4.0 * e * y * m.sin() * (2.0 * l0).cos()
        - 0.5 * y * y * (4.0 * l0).sin()
        - 1.25 * e * e * (2.0 * m).sin();
    4.0 * deg(er)
}

/// The atmospheric refraction in degrees (NOAA's approximate formula) for a geometric elevation in degrees.
pub fn refraction(elev: f64) -> f64 {
    let te = rad(elev).tan();
    let arcsec = if elev > 85.0 {
        0.0
    } else if elev > 5.0 {
        58.1 / te - 0.07 / te.powi(3) + 0.000086 / te.powi(5)
    } else if elev > -0.575 {
        1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    } else {
        -20.772 / te
    };
    arcsec / 3600.0
}

/// Sun zenith (refracted, degrees; above 90 at night) and azimuth (degrees clockwise from north, in [0, 360)) at a
/// latitude and longitude (degrees, east positive) on a Gregorian date and a time of day in a time zone
/// (`tz_hours` east of UT). The time of day is not reduced: hours outside 0-23 move the instant to another day.
#[allow(clippy::too_many_arguments)]
pub fn sun_position(
    lat_deg: f64,
    lon_deg: f64,
    year: i32,
    month: i32,
    day: i32,
    hh: i32,
    mm: i32,
    ss: i32,
    tz_hours: f64,
) -> (f64, f64) {
    let secs = hh as f64 * 3600.0 + mm as f64 * 60.0 + ss as f64;
    let jd = julian_day(year, month, day) + secs / 86400.0 - tz_hours / 24.0;
    let (_, delta_deg, eps, l0, m, e) = solar_coordinates(jd);
    let eot = equation_of_time(eps, l0, m, e);
    let clock_min = hh as f64 * 60.0 + mm as f64 + ss as f64 / 60.0;
    let x = clock_min + eot + 4.0 * lon_deg - 60.0 * tz_hours;
    let tst = x - 1440.0 * (x / 1440.0).floor();
    let mut h = tst / 4.0 - 180.0;
    if h < -180.0 {
        h += 360.0;
    }
    let phi = rad(lat_deg);
    let delta = rad(delta_deg);
    let cos_z = (phi.sin() * delta.sin() + phi.cos() * delta.cos() * rad(h).cos()).clamp(-1.0, 1.0);
    let zen_geom = deg(cos_z.acos());
    let sin_z = rad(zen_geom).sin();
    let pole = lat_deg.abs() >= 90.0 - 1e-10;
    let azi = if pole || phi.cos() * sin_z < 1e-12 {
        if lat_deg >= 0.0 {
            180.0
        } else {
            0.0
        }
    } else {
        let c = ((phi.sin() * rad(zen_geom).cos() - delta.sin()) / (phi.cos() * sin_z)).clamp(-1.0, 1.0);
        let theta = deg(c.acos());
        if h > 0.0 {
            mod360(theta + 180.0)
        } else {
            mod360(540.0 - theta)
        }
    };
    let elev = 90.0 - zen_geom;
    (zen_geom - refraction(elev), azi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_day_meeus() {
        assert!((julian_day(2000, 1, 1) + 0.5 - 2451545.0).abs() < 1e-9);
        assert!((julian_day(1987, 1, 27) - 2446822.5).abs() < 1e-9);
        assert!((julian_day(1988, 1, 27) - 2447187.5).abs() < 1e-9);
        assert!((julian_day(1987, 4, 10) - 2446895.5).abs() < 1e-9);
        assert!((julian_day(2000, 3, 1) - julian_day(2000, 2, 28) - 2.0).abs() < 1e-9);
        assert!((julian_day(1900, 3, 1) - julian_day(1900, 2, 28) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn day_rollover_by_time_zone() {
        // 2000-01-01 0h at UT+3 is 1999-12-31 21h UT: JD 2451544.5 - 0.125 = 2451544.375 (the spec's 2451543.5 base is
        // a slip: JD0 of 2000-01-01 is 2451544.5)
        let jd = julian_day(2000, 1, 1) + 0.0 - 3.0 / 24.0;
        assert!((jd - 2451544.375).abs() < 1e-9);
    }

    #[test]
    fn meeus_example_25a() {
        let (lam, dec, eps, l0, _m, _e) = solar_coordinates(2448908.5);
        assert!((l0 - 201.80720).abs() < 3e-4, "L0 {l0}");
        assert!((lam - 199.90895).abs() < 3e-4, "lambda {lam}");
        assert!((dec - -7.78507).abs() < 3e-4, "delta {dec}");
        assert!((eps - 23.43999).abs() < 3e-4, "eps {eps}");
    }

    #[test]
    fn meeus_example_28a_equation_of_time() {
        let (_, _, eps, l0, m, e) = solar_coordinates(2448908.5);
        let eot = equation_of_time(eps, l0, m, e);
        assert!((eot - 13.710).abs() < 0.02, "EoT {eot}");
    }

    #[test]
    fn equinox_and_solstice_noon() {
        // equator on the equinox: the minimum zenith over the day is small
        let mut zmin = 180.0f64;
        for k in 0..1440 {
            let (z, _) = sun_position(0.0, 0.0, 2021, 3, 20, 0, k, 0, 0.0);
            zmin = zmin.min(z);
        }
        assert!(zmin < 0.5, "{zmin}");
        // Tropic of Cancer at the June solstice
        let mut zmin = 180.0f64;
        for k in 0..1440 {
            let (z, _) = sun_position(23.44, 0.0, 2021, 6, 21, 0, k, 0, 0.0);
            zmin = zmin.min(z);
        }
        assert!(zmin < 1.0, "{zmin}");
        // midnight at the equator on the equinox
        let (z, a) = sun_position(0.0, 0.0, 2021, 3, 20, 0, 0, 0, 0.0);
        assert!(z > 170.0 && (0.0..360.0).contains(&a), "{z} {a}");
    }

    #[test]
    fn morning_afternoon_symmetry() {
        // find solar noon at the equator on the equinox, then compare +-k minutes
        let (_, _, eps, l0, m, e) = solar_coordinates(julian_day(2021, 3, 20) + 0.5);
        let noon = 720.0 - equation_of_time(eps, l0, m, e);
        let at = |min: f64| {
            let s = (min * 60.0).round() as i32;
            sun_position(0.0, 0.0, 2021, 3, 20, 0, 0, s, 0.0)
        };
        let (zm, am) = at(noon - 120.0);
        let (za, aa) = at(noon + 120.0);
        assert!((am + aa - 360.0).abs() < 0.5, "{am} {aa}");
        assert!((zm - za).abs() < 0.05, "{zm} {za}");
    }

    #[test]
    fn poles() {
        let (z, a) = sun_position(90.0, 0.0, 2021, 6, 21, 12, 0, 0, 0.0);
        assert_eq!(a, 180.0);
        assert!((z - 66.6).abs() < 0.1, "{z}");
        let (z2, _) = sun_position(90.0, 0.0, 2021, 6, 21, 3, 0, 0, 0.0);
        assert!((z - z2).abs() < 0.05);
        let (z, a) = sun_position(-90.0, 0.0, 2021, 6, 21, 12, 0, 0, 0.0);
        assert_eq!(a, 0.0);
        assert!((z - 113.4).abs() < 0.1, "{z}");
    }

    #[test]
    fn day_offsets() {
        let eq = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).abs() < 1e-9 && (a.1 - b.1).abs() < 1e-9;
        let f = |hh, tz| sun_position(40.0, -75.0, 2020, 6, 21, hh, 0, 0, tz);
        assert!(eq(f(22, 5.0), f(17, 0.0)));
        assert!(eq(f(2, -5.0), f(7, 0.0)));
        assert!(eq(f(-3, 0.0), sun_position(40.0, -75.0, 2020, 6, 20, 21, 0, 0, 0.0)));
        assert!(eq(f(27, 0.0), sun_position(40.0, -75.0, 2020, 6, 22, 3, 0, 0, 0.0)));
    }

    #[test]
    fn azimuth_in_range_everywhere() {
        for lat in [-90.0, -45.0, 0.0, 45.0, 90.0] {
            for lon in [-180.0, 0.0, 180.0] {
                for hh in 0..24 {
                    let (z, a) = sun_position(lat, lon, 2021, 9, 22, hh, 0, 0, 0.0);
                    assert!(z.is_finite() && a.is_finite() && (0.0..360.0).contains(&a), "{lat} {lon} {hh}: {z} {a}");
                }
            }
        }
    }

    #[test]
    fn refraction_pieces_join() {
        assert_eq!(refraction(86.0), 0.0);
        for edge in [5.0, -0.575] {
            let d = (refraction(edge + 1e-9) - refraction(edge - 1e-9)).abs() * 60.0;
            assert!(d < 1.0, "{edge}: {d} arcmin");
        }
    }
}
