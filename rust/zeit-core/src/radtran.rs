//! The radiative transfer of FORCE's atmospheric correction: a port of FORCE 3.11.00 (Framework for Operational
//! Radiometric Correction for Environmental monitoring), Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later
//! (https://github.com/davidfrantz/force).
//!
//! From `src/modules/lower-level/`: `radtran-ll.c` (all of it), `table-ll.c` (`wavelength`, `E0` and the tables
//! they read, in [`crate::radtran_tables`]), `gas-ll.c` (`ozone_amount`) and `sunview-ll.c` (`localtime2gmt`,
//! `standard_sunzenith`); from `src/modules/cross-level/`: the day-of-year part of `date-cl.c`. `brdf-ll.c` and
//! `sun-cl.c` are not ported: the sun position is [`crate::solpos`], a clean-room implementation (so `sunpos` and
//! the zenith of `standard_sunzenith` are not bit for bit FORCE's).
//!
//! The arithmetic is the C's: C `float` is `f32`, and where C promotes to `double` (`sin`, `exp`, `pow`, `log`
//! of a float, a double literal in an expression) the value is computed in `f64` and converted where C stores it
//! in a float. Every `float` variable of the C is rounded to `f32` where it is assigned, in the same order. The
//! values equal FORCE's bit for bit where the platform's libm agrees with glibc's (`sin`, `cos`, `tan`, `exp`,
//! `pow`, `asin`, `acos`); where it differs in the last bit of a double, a float can differ by one ulp.
//!
//! Names follow FORCE's; `mod` (a keyword) is `mod_`.

use crate::radtran_tables as tab;
pub use crate::radtran_tables::{
    RSR_DIM, RSR_START_LND04, RSR_START_LND05, RSR_START_LND07, RSR_START_LND08, RSR_START_LND09, RSR_START_SEN2A,
    RSR_START_SEN2B, RSR_START_SEN2C, RSR_START_SEN2D, WVL_DIM,
};

/// `_R2D_CONV_` and `_D2R_CONV_` of const-cl.h.
const R2D: f64 = 57.295_779_513_082_322_864_65;
const D2R: f64 = 0.017_453_292_519_943_295_47;

/// C's `(int)x` of a double on x86-64 (`cvttsd2si`): NaN and out of range give `INT_MIN`.
#[inline]
fn c_int(x: f64) -> i32 {
    if x.is_nan() || x >= 2_147_483_648.0 || x <= -2_147_483_649.0 {
        i32::MIN
    } else {
        x as i32
    }
}

// ---------------------------------------------------------------------------------------------------------------
// radtran-ll.c
// ---------------------------------------------------------------------------------------------------------------

/// Molecular optical depth at sea level, `wvl` in um.
pub fn molecular_optical_depth(wvl: f32) -> f32 {
    let w = wvl as f64;
    (0.0088 * w.powf(-4.15 + 0.2 * w)) as f32
}

/// Elevation scaling factor of the AOD; `z` the elevation in km, `hp` the scale height.
pub fn aod_elev_factor(z: f32, hp: f32) -> f32 {
    ((-z / hp) as f64).exp() as f32
}

/// Elevation scaling factor of the MOD; `z` the elevation in km.
pub fn mod_elev_factor(z: f32) -> f32 {
    let z = z as f64;
    (-0.1188 * z - 0.0016 * z * z).exp() as f32
}

/// MOD scaled from the elevation of `hr_is` to that of `hr` (first to sea level, then to the target).
pub fn mod_elev_scale(mod_: f32, hr_is: f32, hr: f32) -> f32 {
    mod_ / hr_is * hr
}

/// AOD scaled from the elevation of `ha_is` to that of `ha`.
pub fn aod_elev_scale(aod: f32, ha_is: f32, ha: f32) -> f32 {
    aod / ha_is * ha
}

/// Optical depth.
pub fn optical_depth(aod: f32, mod_: f32) -> f32 {
    aod + mod_
}

/// The transmittances of `scatt_transmitt`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScattTransmitt {
    /// Total scattering transmittance, `Ts * Tv` (the return value of the C).
    pub total: f32,
    /// Down, total.
    pub ts: f32,
    /// Up, total.
    pub tv: f32,
    /// Down, direct.
    pub tsd: f32,
    /// Down, diffuse.
    pub tss: f32,
    /// Up, direct.
    pub tvd: f32,
    /// Up, diffuse.
    pub tvs: f32,
}

/// The total, direct and diffuse scattering transmittance, down (`ms` the air mass down) and up (`mv`).
pub fn scatt_transmitt(aod: f32, mod_: f32, od: f32, ms: f32, mv: f32) -> ScattTransmitt {
    let tmp = (-1.0 * (0.52 * mod_ as f64 + 0.167 * aod as f64)) as f32;
    let ts = ((tmp / ms) as f64).exp() as f32;
    let tv = ((tmp / mv) as f64).exp() as f32;
    let tvd = (-1.0 * od as f64 / mv as f64).exp() as f32;
    let tvs = tv - tvd;
    let tsd = (-1.0 * od as f64 / ms as f64).exp() as f32;
    let tss = ts - tsd;
    ScattTransmitt {
        total: ts * tv,
        ts,
        tv,
        tsd,
        tss,
        tvd,
        tvs,
    }
}

/// Path reflectance, with the multiple (Sobolev, `multi`) or the single (Gordon) scattering approximation.
#[allow(clippy::too_many_arguments)]
pub fn path_ref(
    multi: bool,
    sob: f32,
    aod: f32,
    mod_: f32,
    od: f32,
    pa: f32,
    pr: f32,
    tsd: f32,
    tvd: f32,
    ms: f32,
    mv: f32,
) -> f32 {
    if multi {
        let xav = (sob * aod) / od;
        let phase = (aod / od) * pa + (mod_ / od) * pr;
        let (mvd, msd) = (mv as f64, ms as f64);
        let rv = (1.0 + 1.5 * mvd + (1.0 - 1.5 * mvd) * tvd as f64) as f32;
        let rs = (1.0 + 1.5 * msd + (1.0 - 1.5 * msd) * tsd as f64) as f32;
        let (xavd, odd) = (xav as f64, od as f64);
        let a = 1.0 - (rv * rs) as f64 / (4.0 + (3.0 - xavd) * odd);
        let b = (3.0 + xavd) * mvd * msd - 2.0 * (mv + ms) as f64 + phase as f64;
        let e = 1.0 - (-od as f64 * (1.0 / mvd + 1.0 / msd)).exp();
        (a + b * e / (mv + ms) as f64 / 4.0) as f32
    } else {
        let (mvd, msd) = (mv as f64, ms as f64);
        ((aod * pa) as f64 / (4.0 * mvd * msd) + (mod_ * pr) as f64 / (4.0 * mvd * msd)) as f32
    }
}

/// Spherical albedo.
pub fn sphere_albedo(aod: f32, mod_: f32, od: f32) -> f32 {
    ((-1.0 * od as f64).exp() * (0.92 * mod_ as f64 + 0.333 * aod as f64)) as f32
}

/// Aerosol environmental weighting function; `r` the target radius in km.
pub fn env_weight_aerosol(r: f32) -> f32 {
    let r = r as f64;
    (1.0 - (0.375 * (-0.2 * r).exp() + 0.625 * (-1.83 * r).exp())) as f32
}

/// Molecular environmental weighting function; `r` the target radius in km.
pub fn env_weight_molecular(r: f32) -> f32 {
    let r = r as f64;
    (1.0 - (0.930 * (-0.8 * r).exp() + 0.070 * (-1.10 * r).exp())) as f32
}

/// Environmental weighting function from the aerosol and molecular ones.
pub fn env_weight(aod: f32, mod_: f32, fa: f32, fr: f32) -> f32 {
    let (aod, mod_, fa, fr) = (aod as f64, mod_ as f64, fa as f64, fr as f64);
    ((0.8333 * aod * fa + 0.5 * mod_ * fr) / (0.8333 * aod + 0.5 * mod_)) as f32
}

/// Cosine of the backscattering angle; `ms`, `mv` the cosines of the zenith angles (air masses), `sazi`, `vazi` the
/// azimuths.
pub fn backscatter(ms: f32, mv: f32, sazi: f32, vazi: f32) -> f32 {
    let (msd, mvd) = (ms as f64, mv as f64);
    let root = (((1.0 - mv * mv) * (1.0 - ms * ms)) as f64).sqrt();
    (-1.0 * mvd * msd - root * ((vazi - sazi) as f64).cos()) as f32
}

/// Molecular phase function.
pub fn phase_molecular(cospsi: f32) -> f32 {
    (0.75 * (1.0 + (cospsi * cospsi) as f64)) as f32
}

/// Aerosol phase function from the six parameters `hg` (a double Henyey-Greenstein).
pub fn phase_aerosol(cospsi: f32, hg: &[f32; 6]) -> f32 {
    let a = ((hg[2] - hg[4] * cospsi) as f64).powf(1.5);
    let b = ((hg[3] + hg[5] * cospsi) as f64).powf(1.5);
    (hg[0] as f64 / a + hg[1] as f64 / b) as f32
}

/// The span of one band's spectral response: its first wavelength index and its values.
#[inline]
fn rsr_row(b_rsr: usize) -> (usize, &'static [f32]) {
    tab::RSR[b_rsr]
}

/// One-way water vapour transmittance of band `b_rsr` (an index into the RSR rows); `w` the path water vapour in
/// cm, `m` the air mass of the path.
///
/// The C sums over every wavelength of the table, with the response 0 outside the band; this sums over the band's
/// window, which is the same sum (the terms outside add `0 * finite`), except where an air mass that is not positive (a
/// sun below the horizon) or not finite makes the term outside the window NaN or infinite (`pow` of a negative base,
/// `exp` of a huge value): there the C's result is NaN, and this runs the whole table to give the same.
pub fn wvp_transmitt(w: f32, m: f32, b_rsr: usize) -> f32 {
    let tmp = w / m;
    let (start, vals) = rsr_row(b_rsr);
    let term = |k: usize| -> f64 {
        let tmp2 = (tab::AW[k] * tmp) as f64;
        ((-1.2110662 * tmp2) / (1.0 + 24.1229127 * tmp2).powf(0.3669996)).exp()
    };
    let (mut a, mut s) = (0.0f64, 0.0f64);
    if tmp >= 0.0 && tmp.is_finite() {
        for (k, &r) in vals.iter().enumerate() {
            let r = r as f64;
            a += r * term(start + k);
            s += r;
        }
    } else {
        for k in 0..WVL_DIM {
            let r = if k >= start && k < start + vals.len() { vals[k - start] as f64 } else { 0.0 };
            a += r * term(k);
            s += r;
        }
    }
    (a / s) as f32
}

/// One-way ozone transmittance of band `b_rsr`; `o` the total ozone, `m` the air mass of the path. As
/// [`wvp_transmitt`]: the sum is over the band's window unless the C's terms outside it are not finite.
pub fn ozone_transmitt(o: f32, m: f32, b_rsr: usize) -> f32 {
    let tmp = -o / m;
    let (start, vals) = rsr_row(b_rsr);
    let term = |k: usize| -> f64 { ((tab::AO[k] * tmp) as f64).exp() };
    let (mut a, mut s) = (0.0f64, 0.0f64);
    if tmp <= 0.0 && tmp.is_finite() {
        for (k, &r) in vals.iter().enumerate() {
            let r = r as f64;
            a += r * term(start + k);
            s += r;
        }
    } else {
        for k in 0..WVL_DIM {
            let r = if k >= start && k < start + vals.len() { vals[k - start] as f64 } else { 0.0 };
            a += r * term(k);
            s += r;
        }
    }
    (a / s) as f32
}

/// Total gas transmittance (water vapour and ozone, down and up).
pub fn gas_transmitt(tsw: f32, tvw: f32, tso: f32, tvo: f32) -> f32 {
    tsw * tvw * tso * tvo
}

/// Fresnel reflection of a flat surface (Kay et al. 2009); `i` the incidence angle in radians.
pub fn fresnel_reflection(i: f32) -> f32 {
    let id = i as f64;
    let iphase = (id.sin() / 1.3333).asin() as f32;
    let (d, s) = ((i - iphase) as f64, (i + iphase) as f64);
    let r1 = d.sin() / s.sin();
    let r2 = d.tan() / s.tan();
    (0.5 * (r1 * r1 + r2 * r2)) as f32
}

/// Cosine of the illumination angle from the cosine and sine of the sun zenith (`csz`, `ssz`) and of the terrain
/// slope (`ctz`, `stz`), the sun azimuth `sa` and the terrain aspect `ta`.
pub fn illumin(csz: f32, ssz: f32, ctz: f32, stz: f32, sa: f32, ta: f32) -> f32 {
    ((csz * ctz) as f64 + (ssz * stz) as f64 * ((sa - ta) as f64).cos()) as f32
}

/// C-factor, empirical: intercept `b` over slope `m` of the regression, 0 if `b < 0` or `m <= 0`.
pub fn c_factor_emp(b: f64, m: f64) -> f32 {
    if b < 0.0 || m <= 0.0 {
        0.0
    } else {
        (b / m) as f32
    }
}

/// C-factor, theoretical (Kobayashi & Sanga-Ngoie 2008): sky view factor `h0`, f-factor `f`, air mass down `ms`.
pub fn c_factor_com(h0: f32, f: f32, ms: f32) -> f32 {
    h0 * f * ms
}

/// f-factor: the diffuse over the direct scattering transmittance down.
pub fn f_factor(tss: f32, tsd: f32) -> f32 {
    tss / tsd
}

// ---------------------------------------------------------------------------------------------------------------
// table-ll.c
// ---------------------------------------------------------------------------------------------------------------

/// The main wavelength (um) of band `b_rsr`: the mean of the wavelengths weighted by the response; 0 if it has none
/// in 410-2400 nm.
pub fn wavelength(b_rsr: usize) -> f32 {
    let (start, vals) = rsr_row(b_rsr);
    let (mut v, mut s) = (0.0f64, 0.0f64);
    for (k, &r) in vals.iter().enumerate() {
        // RSR*WVL is a product of two floats in the C: rounded to float before it is added to the double
        v += (r * tab::WVL[start + k]) as f64;
        s += r as f64;
    }
    if s > 0.0 {
        (v / s) as f32
    } else {
        0.0
    }
}

/// The exoatmospheric solar irradiance (W/m2/um) of band `b_rsr`: the Thuillier spectrum weighted by the response.
pub fn e0(b_rsr: usize) -> f32 {
    let (start, vals) = rsr_row(b_rsr);
    let (mut e, mut s) = (0.0f64, 0.0f64);
    for (k, &r) in vals.iter().enumerate() {
        e += (r * tab::E0[start + k]) as f64;
        s += r as f64;
    }
    if s > 0.0 {
        (e / s) as f32
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------------------------------------------
// gas-ll.c
// ---------------------------------------------------------------------------------------------------------------

/// Total column ozone (cm atm) from a climatological model of longitude, latitude and day of year.
pub fn ozone_amount(lon: f32, lat: f32, doy: i32) -> f32 {
    let d: f32 = 0.9856;
    let g: f32 = 20.0;
    let omin: f32 = 235.0;
    let doy_f = doy as f32;
    let north = lat > 0.0;
    let a: f32 = if north { 150.0 } else { 100.0 };
    let h: f32 = if north { 3.0 } else { 2.0 };
    let beta: f32 = if north { 1.28 } else { 1.5 };
    let c: f32 = if north { 40.0 } else { 30.0 };
    let f: f32 = if north { -30.0 } else { 152.625 };
    let i: f32 = if north && lon > 0.0 {
        20.0
    } else if north && lon < 0.0 {
        0.0
    } else {
        -75.0
    };
    // D*(DOY+F), H*(lon+I) and beta*lat are products of floats: rounded to float before the double multiplication
    let s_lat = ((beta * lat) as f64 * D2R).sin();
    let o = omin as f64
        + (a as f64
            + c as f64 * ((d * (doy_f + f)) as f64 * D2R).sin()
            + g as f64 * ((h * (lon + i)) as f64 * D2R).sin())
            * s_lat
            * s_lat;
    (o as f32) / 1000.0
}

// ---------------------------------------------------------------------------------------------------------------
// date-cl.c (the day-of-year routines), sun-cl.c, sunview-ll.c
// ---------------------------------------------------------------------------------------------------------------

/// FORCE's `date_t`, the fields the solar position reads. A year has 365 days (leap years are ignored).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Date {
    pub year: i32,
    pub month: i32,
    pub day: i32,
    /// Day of year, 1-365.
    pub doy: i32,
    pub hh: i32,
    pub mm: i32,
    pub ss: i32,
    /// Time zone, hours.
    pub tz: i32,
}

/// Month and day of a day of year in a 365 day year.
pub fn doy2md(doy: i32) -> (i32, i32) {
    const START: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    for m in (1..12).rev() {
        if doy > START[m] {
            return (m as i32 + 1, doy - START[m]);
        }
    }
    (1, doy)
}

/// Day of year of a month and day (0 for a month outside 1-12).
pub fn md2doy(m: i32, d: i32) -> i32 {
    const START: [i32; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if (1..=12).contains(&m) {
        START[(m - 1) as usize] + d
    } else {
        0
    }
}

impl Date {
    /// The date of a day of year (`set_date_year`, `set_date_doy`), the time zone and the time of day 0.
    pub fn from_year_doy(year: i32, doy: i32) -> Self {
        let mut d = Date {
            year,
            ..Date::default()
        };
        d.set_doy(doy);
        d
    }

    /// `set_date_doy`: month, day and day of year of a day of year.
    pub fn set_doy(&mut self, doy: i32) {
        let (m, d) = doy2md(doy);
        self.month = m;
        self.day = d;
        self.doy = md2doy(m, d);
    }
}

/// Sun zenith and azimuth in degrees (FORCE's `sunpos`) at a place (`latitude`, `longitude`) and date (with its
/// time of day and time zone). FORCE's `sun-cl.c` is not ported (FORCE adopted it from the NOAA ESRL Solar Position
/// Calculator, whose license is not verified): the position is [`crate::solpos::sun_position`], written in a clean
/// room from the published formulas.
pub fn sunpos(latitude: f32, longitude: f32, date: &Date) -> (f32, f32) {
    let (zen, azi) = crate::solpos::sun_position(
        latitude as f64,
        longitude as f64,
        date.year,
        date.month,
        date.day,
        date.hh,
        date.mm,
        date.ss,
        date.tz as f64,
    );
    (zen as f32, azi as f32)
}

/// The GMT date of a local overpass time in decimal hours (`localtime2gmt`): the day moves by one at most, within
/// 1-365, and the time of day is cut to whole hours, minutes and seconds.
pub fn localtime2gmt(dmeta: &Date, localtime: f64, lon: f64) -> Date {
    let mut gmttime = localtime - lon / 15.0;
    let mut doy = dmeta.doy;
    if gmttime > 24.0 {
        doy += 1;
        gmttime -= 24.0;
        if doy > 365 {
            doy = 365;
        }
    }
    if gmttime < 0.0 {
        doy -= 1;
        gmttime += 24.0;
        if doy < 1 {
            doy = 1;
        }
    }
    let mut dgmt = *dmeta;
    dgmt.set_doy(doy);
    let hh = c_int(gmttime);
    let left_mm = (gmttime - hh as f64) * 60.0;
    let mm = c_int(left_mm);
    let left_ss = (left_mm - mm as f64) * 60.0;
    let ss = c_int(left_ss);
    dgmt.hh = hh;
    dgmt.mm = mm;
    dgmt.ss = ss;
    dgmt
}

/// The standardized sun zenith (degrees) of a latitude and longitude on a date: the mean of the Landsat and
/// Sentinel-2 sun zeniths at their overpass times (`standard_sunzenith`, for the BRDF correction's harmonization).
/// NaN overpass times (beyond about 82 degrees of latitude) turn into `INT_MIN` hours as in the C.
pub fn standard_sunzenith(dmeta: &Date, lat: f64, lon: f64) -> f64 {
    let rlat = lat * D2R;
    let incl_landsat = 98.2 * D2R;
    let incl_sentinel2 = 98.62 * D2R;
    let mut localtime = 0.0f64;
    localtime += 10.18333333333 - (rlat.tan() / incl_landsat.tan()).asin() * R2D / 15.0;
    localtime += 10.5 - (rlat.tan() / incl_sentinel2.tan()).asin() * R2D / 15.0;
    localtime /= 2.0;
    let dgmt = localtime2gmt(dmeta, localtime, lon);
    let (szen, _) = sunpos(lat as f32, lon as f32, &dgmt);
    szen as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-30)
    }

    // The reference values below were printed by the C harness of benchmarks/vs_originals/force_radtran (FORCE's own
    // functions, gcc -O3, glibc 2.39). Bit for bit where the C library agrees; the tests allow 1 ulp of f32.
    fn ulp1(a: f32, b: f32) -> bool {
        a == b || (a.to_bits() as i64 - b.to_bits() as i64).abs() <= 1
    }

    /// The values the C harness printed (bits of the float32), for the same arguments.
    #[test]
    fn values_of_the_c_harness() {
        let hg = [1.0f32, 0.5, 1.2, 1.3, 0.4, 0.45];
        let s2 = RSR_START_SEN2A;
        let t = scatt_transmitt(0.2, 0.1, 0.3, 0.8, 0.95);
        let mut d = Date::from_year_doy(2021, 172);
        (d.hh, d.mm, d.ss) = (10, 30, 0);
        let (zen, azi) = sunpos(45.0, 10.0, &d);
        let mut d2 = Date::from_year_doy(2019, 300);
        (d2.hh, d2.mm, d2.ss, d2.tz) = (13, 15, 40, -3);
        let (zen2, azi2) = sunpos(-23.5, -46.6, &d2);
        let cases: Vec<(&str, f32, u32)> = vec![
            ("mod 0.55", molecular_optical_depth(0.55), 0x3dc9b820),
            ("mod 0.865", molecular_optical_depth(0.865), 0x3c8056da),
            ("aod_elev_factor", aod_elev_factor(1.5, 2.0), 0x3ef1da07),
            ("mod_elev_factor", mod_elev_factor(1.5), 0x3f5571d3),
            ("total", t.total, 0x3f524cad),
            ("ts", t.ts, 0x3f661481),
            ("tv", t.tv, 0x3f69fdb3),
            ("tsd", t.tsd, 0x3f2ff231),
            ("tss", t.tss, 0x3e588940),
            ("tvd", t.tvd, 0x3f3aadb3),
            ("tvs", t.tvs, 0x3e3d4000),
            (
                "path multi",
                path_ref(true, 0.9, 0.2, 0.1, 0.3, 1.2, 1.1, 0.7, 0.8, 0.8, 0.95),
                0x3df320fc,
            ),
            (
                "path single",
                path_ref(false, 0.9, 0.2, 0.1, 0.3, 1.2, 1.1, 0.7, 0.8, 0.8, 0.95),
                0x3debca1c,
            ),
            ("sphere_albedo", sphere_albedo(0.2, 0.1, 0.3), 0x3df0a093),
            ("env_weight_aerosol", env_weight_aerosol(0.5), 0x3ed21b1c),
            ("env_weight_molecular", env_weight_molecular(0.5), 0x3eac247d),
            ("env_weight", env_weight(0.2, 0.1, 0.4, 0.8), 0x3efc1021),
            ("backscatter", backscatter(0.8, 0.9, 1.0, 2.5), 0xbf3d0e59),
            ("phase_molecular", phase_molecular(-0.3), 0x3f5147ae),
            ("phase_aerosol", phase_aerosol(-0.3, &hg), 0x3f874c54),
            ("wvp S2A wvp band", wvp_transmitt(3.0, 0.9, s2 + 9), 0x3ead27d6),
            ("wvp L8 nir", wvp_transmitt(1.2, 0.7, RSR_START_LND08 + 4), 0x3f7f6468),
            ("ozone S2A red", ozone_transmitt(0.3, 0.9, s2 + 3), 0x3f7b6f00),
            (
                "ozone L8 red",
                ozone_transmitt(0.28, 0.6, RSR_START_LND08 + 3),
                0x3f785e3f,
            ),
            ("fresnel", fresnel_reflection(0.7), 0x3cc9768a),
            ("illumin", illumin(0.8, 0.6, 0.9, 0.43588989, 1.0, 2.0), 0x3f5c7ea4),
            ("wavelength L8 nir", wavelength(25), 0x3f5d370b),
            ("wavelength S2A red", wavelength(33), 0x3f2a24a7),
            ("wavelength S2A wvp", wavelength(39), 0x3f71ef17),
            ("e0 L8 nir", e0(25), 0x446defb1),
            ("e0 S2A red", e0(33), 0x44bcc854),
            ("e0 S2A wvp", e0(39), 0x444adada),
            ("ozone north", ozone_amount(10.0, 50.0, 80), 0x3ecb2c34),
            ("ozone south", ozone_amount(-60.0, -40.0, 250), 0x3ead4e1d),
        ];
        // The sun position is no longer FORCE's code (a clean-room one from Meeus and NOAA's documentation): the
        // values of the C harness are matched to 0.02 degrees (0.1 for the azimuth), not bit for bit.
        let near = [
            ("sun zen", zen, 0x41c0055f),
            ("sun azi", azi, 0x43159c2d),
            ("sun zen tz", zen2, 0x41b6e522),
            ("sun azi tz", azi2, 0x4392fe12),
            (
                "standard 45N",
                standard_sunzenith(&Date::from_year_doy(2021, 172), 45.0, 10.0) as f32,
                0x41cc5aba,
            ),
            (
                "standard 33S",
                standard_sunzenith(&Date::from_year_doy(2021, 30), -33.0, 151.0) as f32,
                0x4207898a,
            ),
        ];
        for (what, got, bits) in near {
            let want = f32::from_bits(bits);
            let tol = if what.contains("azi") { 0.1 } else { 0.02 };
            assert!((got - want).abs() < tol, "{what}: {got:?} against {want:?}");
        }
        for (what, got, bits) in cases {
            let want = f32::from_bits(bits);
            assert!(
                ulp1(got, want),
                "{what}: {got:?} ({:#x}) against {want:?} ({bits:#x})",
                got.to_bits()
            );
        }
        let g = localtime2gmt(&Date::from_year_doy(2021, 100), 10.25, 150.0);
        assert_eq!((g.month, g.day, g.doy, g.hh, g.mm, g.ss), (4, 10, 100, 0, 15, 0));
        let g = localtime2gmt(&Date::from_year_doy(2021, 100), 23.0, -30.0);
        assert_eq!((g.month, g.day, g.doy, g.hh, g.mm, g.ss), (4, 11, 101, 1, 0, 0));
    }

    #[test]
    fn tables_have_the_c_shape() {
        assert_eq!(tab::RSR.len(), 56);
        assert_eq!(tab::E0.len(), 1991);
        assert_eq!(tab::WVL[0], 0.41);
        assert_eq!(tab::WVL[1990], 2.4);
        assert_eq!(tab::E0[0], 1613.99);
        assert_eq!(tab::AW[1], 5.3094e-07);
        assert_eq!(RSR_START_LND09, RSR_START_LND08);
        assert_eq!(RSR_START_SEN2C, RSR_START_SEN2A);
        // every span ends inside the 1991 wavelengths
        for (start, v) in tab::RSR.iter() {
            assert!(start + v.len() <= WVL_DIM);
        }
    }

    #[test]
    fn band_wavelengths_and_irradiance() {
        // Landsat 8 OLI: blue 0.48, green 0.56, red 0.655, nir 0.865 um (to a few nm); thermal has none
        let l8 = RSR_START_LND08;
        for (k, w) in [(1usize, 0.48f32), (2, 0.56), (3, 0.655), (4, 0.865)] {
            assert!(
                (wavelength(l8 + k) - w).abs() < 0.01,
                "band {k}: {}",
                wavelength(l8 + k)
            );
        }
        assert_eq!(wavelength(6), 0.0);
        assert_eq!(e0(6), 0.0);
        // Sentinel-2A B04 (red) 0.665, B8A 0.865
        let s2 = RSR_START_SEN2A;
        assert!((wavelength(s2 + 3) - 0.665).abs() < 0.01);
        assert!((wavelength(s2 + 8) - 0.865).abs() < 0.01);
        // E0 decreases from the blue to the nir
        assert!(e0(s2 + 1) > e0(s2 + 3) && e0(s2 + 3) > e0(s2 + 8));
        assert!(e0(s2 + 8) > 800.0 && e0(s2 + 8) < 1100.0);
    }

    #[test]
    fn molecular_depth_and_elevation() {
        // Rayleigh optical depth at 0.55 um is about 0.097
        assert!((molecular_optical_depth(0.55) - 0.0973).abs() < 0.002);
        assert!(molecular_optical_depth(0.45) > molecular_optical_depth(0.65));
        assert_eq!(mod_elev_factor(0.0), 1.0);
        assert!(mod_elev_factor(2.0) < 1.0);
        assert_eq!(aod_elev_factor(0.0, 2.0), 1.0);
        assert!(ulp1(aod_elev_factor(1.0, 2.0), (-0.5f64).exp() as f32));
        assert_eq!(mod_elev_scale(0.1, 1.0, 1.0), 0.1);
        assert_eq!(optical_depth(0.25, 0.5), 0.75);
        // scaling to a given elevation and back
        let h = mod_elev_factor(1.5);
        assert!(close(mod_elev_scale(mod_elev_scale(0.1, 1.0, h), h, 1.0), 0.1, 1e-6));
    }

    #[test]
    fn scattering_transmittances_are_consistent() {
        let (aod, mod_) = (0.2f32, 0.1f32);
        let od = optical_depth(aod, mod_);
        let t = scatt_transmitt(aod, mod_, od, 0.8, 0.95);
        assert_eq!(t.total, t.ts * t.tv);
        assert_eq!(t.tvs, t.tv - t.tvd);
        assert_eq!(t.tss, t.ts - t.tsd);
        assert!(t.tsd < t.ts && t.ts < 1.0 && t.tvd < t.tv && t.tv < 1.0);
        // no atmosphere: everything is transmitted directly
        let t0 = scatt_transmitt(0.0, 0.0, 0.0, 0.7, 0.9);
        assert_eq!(
            (t0.ts, t0.tv, t0.tsd, t0.tvd, t0.tss, t0.tvs),
            (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
        );
        assert!(close(f_factor(t.tss, t.tsd), t.tss / t.tsd, 0.0));
        assert!(close(c_factor_com(0.5, 0.2, 0.8), 0.5 * 0.2 * 0.8, 0.0));
    }

    #[test]
    fn path_reflectance_single_and_multiple() {
        let (aod, mod_) = (0.2f32, 0.1f32);
        let od = optical_depth(aod, mod_);
        let (ms, mv) = (0.8f32, 0.95f32);
        let t = scatt_transmitt(aod, mod_, od, ms, mv);
        let cospsi = backscatter(ms, mv, 2.0, 0.5);
        let pr = phase_molecular(cospsi);
        let pa = phase_aerosol(cospsi, &[1.0, 0.5, 1.2, 1.3, 0.4, 0.45]);
        let single = path_ref(false, 0.0, aod, mod_, od, pa, pr, t.tsd, t.tvd, ms, mv);
        let multi = path_ref(true, 0.9, aod, mod_, od, pa, pr, t.tsd, t.tvd, ms, mv);
        assert!(single > 0.0 && multi > 0.0 && single.is_finite() && multi.is_finite());
        assert!((single - multi).abs() < 0.05);
        // no aerosol and no air: no path reflectance in the single scattering approximation
        assert_eq!(path_ref(false, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, ms, mv), 0.0);
        // the molecular phase function is 0.75 at right angles and 1.5 forwards
        assert_eq!(phase_molecular(0.0), 0.75);
        assert_eq!(phase_molecular(1.0), 1.5);
        assert!((0.0..=1.0).contains(&sphere_albedo(aod, mod_, od)));
    }

    #[test]
    fn environment_weighting() {
        assert_eq!(env_weight_aerosol(0.0), 0.0);
        assert!(env_weight_aerosol(10.0) > 0.85 && env_weight_aerosol(10.0) < 1.0);
        assert!(env_weight_molecular(10.0) > 0.99);
        assert!(env_weight_aerosol(1.0) < env_weight_aerosol(2.0));
        let f = env_weight(0.2, 0.1, 0.5, 0.9);
        assert!(f > 0.5 && f < 0.9);
        // the weights alone
        assert!(close(env_weight(0.2, 0.0, 0.5, 0.9), 0.5, 1e-6));
    }

    #[test]
    fn backscatter_geometry() {
        // sun and view at nadir: cos(psi) = -1
        assert!(close(backscatter(1.0, 1.0, 0.0, 0.0), -1.0, 1e-6));
        // same azimuth: cos(psi) = -mv*ms - sin*sin = -1 for equal angles; opposite azimuths: -0.64 + 0.36
        assert!(close(backscatter(0.8, 0.8, 0.0, 0.0), -1.0, 1e-6));
        assert!(close(backscatter(0.8, 0.8, 0.0, std::f32::consts::PI), -0.28, 1e-5));
    }

    #[test]
    fn gas_transmittance_of_a_band() {
        let s2 = RSR_START_SEN2A;
        let (nir, wvp) = (s2 + 8, s2 + 9);
        // no gas, full transmittance (sum of exp(0) * rsr / sum of rsr)
        assert!(close(wvp_transmitt(0.0, 1.0, nir), 1.0, 1e-6));
        assert!(close(ozone_transmitt(0.0, 1.0, nir), 1.0, 1e-6));
        // more water vapour absorbs more in the water vapour band than in the nir band
        let (a, b) = (wvp_transmitt(3.0, 1.0, wvp), wvp_transmitt(3.0, 1.0, nir));
        assert!(a < b && a > 0.0 && b < 1.0);
        // w / m is the path: a smaller m (a cosine) is a longer path
        assert!(wvp_transmitt(3.0, 0.5, wvp) < a);
        // ozone absorbs in the green/red (Chappuis band), not in the nir
        let (green, red) = (s2 + 2, s2 + 3);
        assert!(ozone_transmitt(0.3, 1.0, green) < 1.0);
        assert!(ozone_transmitt(0.3, 1.0, red) < ozone_transmitt(0.3, 1.0, nir));
        assert!(wvp_transmitt(1.0, 1.0, 6).is_nan()); // a band without response: 0 / 0 as in the C
        assert_eq!(gas_transmitt(0.9, 0.9, 0.99, 0.99), 0.9f32 * 0.9 * 0.99 * 0.99);
    }

    #[test]
    fn fresnel_and_illumination() {
        // reflection grows to 1 at grazing incidence, 2% at nadir
        assert!((fresnel_reflection(0.001) - 0.0200).abs() < 0.0005);
        assert!(fresnel_reflection(1.4) > fresnel_reflection(0.7));
        assert!(fresnel_reflection(1.5707) > 0.95);
        // flat terrain: cos i = cos(sun zenith)
        let (z, a) = (0.6f32, 1.0f32);
        let ci = illumin(z.cos(), z.sin(), 1.0, 0.0, a, 0.3);
        assert!(close(ci, z.cos(), 1e-6));
        assert_eq!(c_factor_emp(-1.0, 1.0), 0.0);
        assert_eq!(c_factor_emp(1.0, 0.0), 0.0);
        assert_eq!(c_factor_emp(1.0, 4.0), 0.25);
    }

    #[test]
    fn ozone_climatology() {
        let o = ozone_amount(10.0, 50.0, 80);
        assert!(o > 0.2 && o < 0.5, "{o}");
        // doy as in the C: the equator has the minimum
        assert_eq!(ozone_amount(0.0, 0.0, 100), 0.235);
        // southern: a different branch
        assert!(ozone_amount(-60.0, -40.0, 250) != ozone_amount(-60.0, 40.0, 250));
    }

    #[test]
    fn day_of_year_routines() {
        assert_eq!(doy2md(1), (1, 1));
        assert_eq!(doy2md(31), (1, 31));
        assert_eq!(doy2md(32), (2, 1));
        assert_eq!(doy2md(60), (3, 1));
        assert_eq!(doy2md(365), (12, 31));
        for doy in 1..=365 {
            let (m, d) = doy2md(doy);
            assert_eq!(md2doy(m, d), doy);
        }
        assert_eq!(md2doy(13, 1), 0);
    }

    #[test]
    fn sun_position_known_cases() {
        // equinox noon at the equator on the Greenwich meridian: the sun is near the zenith
        let mut d = Date::from_year_doy(2020, 80);
        d.hh = 12;
        let (zen, _) = sunpos(0.0, 0.0, &d);
        assert!(zen < 3.0, "{zen}");
        // 12:00 GMT at 45 N on the summer solstice (doy 172): zenith about 45 - 23.4
        let mut d = Date::from_year_doy(2020, 172);
        d.hh = 12;
        let (zen, azi) = sunpos(45.0, 0.0, &d);
        assert!((zen - 21.6).abs() < 1.5, "{zen}");
        assert!((azi - 180.0).abs() < 10.0, "{azi}");
        // the sun is below the horizon at midnight
        d.hh = 0;
        assert!(sunpos(45.0, 0.0, &d).0 > 90.0);
    }

    #[test]
    fn local_time_to_gmt() {
        let d = Date::from_year_doy(2021, 100);
        // 10.5 h local at 15 E is 9:30 GMT
        let g = localtime2gmt(&d, 10.5, 15.0);
        assert_eq!((g.doy, g.hh, g.mm, g.ss), (100, 9, 30, 0));
        // 23 h local at 30 W is 25 h: the next day, 1:00
        let g = localtime2gmt(&d, 23.0, -30.0);
        assert_eq!((g.doy, g.hh, g.mm, g.ss), (101, 1, 0, 0));
        // 1 h local at 30 E is -1 h: the day before, 23:00
        let g = localtime2gmt(&d, 1.0, 30.0);
        assert_eq!((g.doy, g.hh), (99, 23));
        // the day stays within 1-365
        let g = localtime2gmt(&Date::from_year_doy(2021, 1), 1.0, 30.0);
        assert_eq!(g.doy, 1);
        let g = localtime2gmt(&Date::from_year_doy(2021, 365), 23.0, -30.0);
        assert_eq!(g.doy, 365);
    }

    #[test]
    fn standard_zenith_is_a_sun_zenith() {
        let d = Date::from_year_doy(2021, 172);
        let z = standard_sunzenith(&d, 45.0, 10.0);
        assert!(z > 15.0 && z < 40.0, "{z}");
        // winter has a higher zenith at the same place
        let w = standard_sunzenith(&Date::from_year_doy(2021, 355), 45.0, 10.0);
        assert!(w > z + 30.0, "{w}");
        // the southern hemisphere has the opposite season
        assert!(standard_sunzenith(&d, -45.0, 10.0) > z + 30.0);
    }

    #[test]
    fn a_sun_below_the_horizon_gives_the_c_nan_for_the_water_vapour() {
        // the C sums the whole table: for a negative air mass the terms outside the band's window are NaN
        // (pow of a negative base) and 0 * NaN poisons the sum; the band's window alone would give a finite value
        for band in [21, 22, 30, 42] {
            assert!(wvp_transmitt(2.0, -0.0592, band).is_nan(), "band {band}");
            assert!(wvp_transmitt(2.0, 0.6, band).is_finite());
            assert!(ozone_transmitt(0.3, -0.0592, band).is_finite());
        }
    }
}
