//! Nadir BRDF-adjusted reflectance (NBAR) by the c-factor: the reflectance of a pixel as it would
//! be seen from nadir with the sun at a chosen zenith, from its observed sun and view geometry.
//!
//! Written from the papers (clean room), not from any implementation:
//!
//! * Roy, D. P., Zhang, H. K., Ju, J., Gomez-Dans, J. L., Lewis, P. E., Schaaf, C. B., Sun, Q., Li,
//!   J., Huang, H., Kovalskyy, V. (2016), "A general method to normalize Landsat reflectance data
//!   to nadir BRDF adjusted reflectance", Remote Sensing of Environment 176:255-271: Eq. (1), the
//!   c-factor of Eq. (7), and the fixed global MODIS BRDF parameters of Table 5.
//! * Roy, D. P., Li, J., Zhang, H. K., Yan, L., Huang, H., Li, Z. (2017), "Examination of Sentinel-2A
//!   multi-spectral instrument (MSI) reflectance anisotropy and the suitability of a general
//!   method to normalize MSI reflectance to nadir BRDF adjusted reflectance", Remote Sensing of
//!   Environment 199:25-38 (Eqs. 5 and 6).
//! * Roy, D. P., Li, Z., Zhang, H. K. (2017), "Adjustment of Sentinel-2 multi-spectral instrument
//!   (MSI) red-edge band reflectance to nadir BRDF adjusted reflectance (NBAR) and
//!   quantification of red-edge band BRDF effects", Remote Sensing 9(12):1325: Eq. (3), the
//!   red edge parameters interpolated between the red and the NIR of MODIS.
//! * The kernels, RossThick and LiSparse-Reciprocal with h/b = 2 and b/r = 1, as Matsuoka et al.
//!   (2016, ISPRS Annals III-7:3-10, Eqs. 2 and 4) write them after Wanner, Li and Strahler (1995,
//!   JGR 100:21077-21089) and Lucht, Schaaf and Strahler (2000, IEEE TGRS 38:977-998). The HLS
//!   User Guide v2.0 (Tables 4 and 8, section 4.4) gives the coefficients for OLI and MSI.
//!
//! The model (Roy 2016 Eq. 1): `rho(sza, vza, raa) = f_iso + f_vol K_vol + f_geo K_geo`, three
//! parameters per band. The c-factor is `rho(sza_norm, 0, 0) / rho(sza, vza, raa)`, and the NBAR
//! the reflectance observed times it. The numerator is constant for a date and a band (nadir, the
//! normalized solar zenith: at nadir the azimuth is of no consequence), the denominator is the
//! pixel's.
//!
//! Decisions where the papers are silent.
//!
//! * A pixel whose solar or view zenith is not finite, is negative or is 90 degrees or more, or
//!   whose azimuths are not finite, has no c-factor: NaN. The c-factor is not clipped.
//! * The relative azimuth is `vaa - saa` in degrees; the kernels use only its cosine and the
//!   square of its sine, so the sign and the turns of 360 do not matter.
//! * `cos(xi)` and the cosine of the Li-Sparse `t` are clamped to [-1, 1] (rounding, or geometry
//!   outside the domain). A model value that is not positive (not possible with positive
//!   parameters) gives NaN.
//! * The Li-Sparse kernel is written with the primed angles (`tan theta' = b/r tan theta`) in
//!   its distance term, as Wanner et al. define it; with the fixed b/r = 1 they are the observed
//!   angles, and the choice does not matter.
//! * The red edge parameters are not tabulated: they are Eq. (3) of Roy, Li and Zhang (2017) at
//!   705, 740 and 783 nm, linear in the wavelength between the red (645 nm) and NIR (858 nm)
//!   parameters of MODIS. The interpolation is simple and the paper found it not particularly
//!   suited to the red edge (a POLDER test): a limitation, not a bug.
//!
//! Not for snow (the parameters are MODIS's snow-free ones), and Roy 2016 does not recommend it
//! for a solar zenith far from that of the overpass.
//!
//! Computed in f64 whatever the type of the data (f32 or f64, through `Real`); the kernels of a
//! pixel are computed once for all the bands.

use crate::parallel::{for_each_row, uninit_f64};
use crate::tsa::Sample;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// The parameters of the model for one band, reflectance 0-1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RossLi {
    pub f_iso: f64,
    pub f_geo: f64,
    pub f_vol: f64,
}

impl RossLi {
    pub const fn new(f_iso: f64, f_geo: f64, f_vol: f64) -> RossLi {
        RossLi { f_iso, f_geo, f_vol }
    }

    /// The model's reflectance for the kernel values `k_vol` and `k_geo`.
    #[inline]
    pub fn reflectance(self, k_vol: f64, k_geo: f64) -> f64 {
        self.f_iso + self.f_vol * k_vol + self.f_geo * k_geo
    }
}

// Roy et al. (2016), Table 5 (12 global months of MODIS MCD43A1, 2010; the columns are f_iso,
// f_geo, f_vol), as the HLS User Guide v2.0 Table 4 uses them for OLI and MSI.
/// Blue (MODIS band 3), also used for coastal/aerosol (HLS Table 4).
pub const BLUE: RossLi = RossLi::new(0.0774, 0.0079, 0.0372);
/// Green (MODIS band 4).
pub const GREEN: RossLi = RossLi::new(0.1306, 0.0178, 0.0580);
/// Red (MODIS band 1, 645 nm).
pub const RED: RossLi = RossLi::new(0.1690, 0.0227, 0.0574);
/// NIR (MODIS band 2, 858 nm), for the broad (B08) and the narrow (B8A, OLI B5) NIR alike.
pub const NIR: RossLi = RossLi::new(0.3093, 0.0330, 0.1535);
/// SWIR 1.6 um (MODIS band 6).
pub const SWIR1: RossLi = RossLi::new(0.3430, 0.0453, 0.1154);
/// SWIR 2.1 um (MODIS band 7).
pub const SWIR2: RossLi = RossLi::new(0.2658, 0.0387, 0.0639);

/// Wavelengths (nm) of MODIS's red and NIR bands, where the red edge interpolation starts and ends.
pub const RED_NM: f64 = 645.0;
pub const NIR_NM: f64 = 858.0;
/// Centre wavelengths (nm) of Sentinel-2's red edge bands B05, B06, B07.
pub const REDEDGE_NM: [f64; 3] = [705.0, 740.0, 783.0];

/// The parameters at `lambda_nm`, linear in the wavelength between the red (645 nm) and the NIR
/// (858 nm) of MODIS: Roy, Li and Zhang (2017) Eq. (3).
pub fn rededge(lambda_nm: f64) -> RossLi {
    let t = (lambda_nm - RED_NM) / (NIR_NM - RED_NM);
    let lerp = |a: f64, b: f64| a + t * (b - a);
    RossLi::new(lerp(RED.f_iso, NIR.f_iso), lerp(RED.f_geo, NIR.f_geo), lerp(RED.f_vol, NIR.f_vol))
}

/// The roles that have parameters, in this order.
pub const ROLES: [&str; 11] = [
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir1", "swir2",
];

/// The parameters of a role (`ROLES`; `nir` is the broad NIR, B08, and `nir08` the narrow one,
/// B8A or OLI's B5), or None.
pub fn role(name: &str) -> Option<RossLi> {
    Some(match name {
        "coastal" | "blue" => BLUE,
        "green" => GREEN,
        "red" => RED,
        "rededge1" => rededge(REDEDGE_NM[0]),
        "rededge2" => rededge(REDEDGE_NM[1]),
        "rededge3" => rededge(REDEDGE_NM[2]),
        "nir" | "nir08" => NIR,
        "swir1" => SWIR1,
        "swir2" => SWIR2,
        _ => return None,
    })
}

/// Li-Sparse's crown height over its vertical radius, and the vertical over the horizontal radius.
const H_OVER_B: f64 = 2.0;
const B_OVER_R: f64 = 1.0;

/// The zeniths in radians that the model is defined for: finite and in [0, 90 degrees).
#[inline]
fn valid_zenith(z: f64) -> bool {
    (0.0..FRAC_PI_2).contains(&z)
}

/// RossThick and LiSparse-Reciprocal (h/b = 2, b/r = 1) for a solar zenith, a view zenith and a
/// relative azimuth, in radians: `(k_vol, k_geo)`. NaN for a zenith outside [0, 90 degrees) or a
/// non-finite angle. `k_vol(0, 0, _) = k_geo(0, 0, _) = 0`.
pub fn kernels(sza_rad: f64, vza_rad: f64, raa_rad: f64) -> (f64, f64) {
    if !(valid_zenith(sza_rad) && valid_zenith(vza_rad) && raa_rad.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let (cs, ss) = (sza_rad.cos(), sza_rad.sin());
    let (cv, sv) = (vza_rad.cos(), vza_rad.sin());
    let (cp, sp) = (raa_rad.cos(), raa_rad.sin());

    // RossThick
    let cos_xi = (cs * cv + ss * sv * cp).clamp(-1.0, 1.0);
    let xi = cos_xi.acos();
    let k_vol = ((FRAC_PI_2 - xi) * cos_xi + xi.sin()) / (cs + cv) - FRAC_PI_4;

    // LiSparse-Reciprocal, with the primed angles tan(theta') = b/r tan(theta)
    let tan_s = B_OVER_R * sza_rad.tan();
    let tan_v = B_OVER_R * vza_rad.tan();
    let (ts, tv) = (tan_s.atan(), tan_v.atan());
    let (cs2, cv2) = (ts.cos(), tv.cos());
    let cos_xi2 = cs2 * cv2 + ts.sin() * tv.sin() * cp;
    let d2 = (tan_s * tan_s + tan_v * tan_v - 2.0 * tan_s * tan_v * cp).max(0.0);
    let sec = 1.0 / cs2 + 1.0 / cv2;
    let cos_t = H_OVER_B * (d2 + (tan_s * tan_v * sp).powi(2)).sqrt() / sec;
    let t = cos_t.clamp(-1.0, 1.0).acos();
    let overlap = (t - t.sin() * t.cos()) * sec / PI;
    let k_geo = overlap - sec + 0.5 * (1.0 + cos_xi2) / (cs2 * cv2);
    (k_vol, k_geo)
}

/// `kernels` of a geometry in degrees (the relative azimuth is `vaa - saa`).
#[inline]
fn kernels_deg(sza: f64, saa: f64, vza: f64, vaa: f64) -> (f64, f64) {
    kernels(sza.to_radians(), vza.to_radians(), (vaa - saa).to_radians())
}

/// The model at nadir for the normalized solar zenith (the c-factor's numerator), NaN when
/// `sza_norm_deg` is outside [0, 90).
#[inline]
fn numerator(coef: RossLi, sza_norm_deg: f64) -> f64 {
    let (kv, kg) = kernels(sza_norm_deg.to_radians(), 0.0, 0.0);
    coef.reflectance(kv, kg)
}

/// The c-factor from the numerator and the pixel's kernels: NaN when the model's reflectance of
/// the pixel is not positive or either is NaN.
#[inline]
fn c_of(coef: RossLi, num: f64, k_vol: f64, k_geo: f64) -> f64 {
    let den = coef.reflectance(k_vol, k_geo);
    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

/// The c-factor of one band and pixel (angles in degrees): `rho(sza_norm, 0, 0) / rho(sza, vza,
/// vaa - saa)`. NaN if any angle is not finite or outside the model's domain.
pub fn cfactor(coef: RossLi, sza_deg: f64, saa_deg: f64, vza_deg: f64, vaa_deg: f64, sza_norm_deg: f64) -> f64 {
    let (kv, kg) = kernels_deg(sza_deg, saa_deg, vza_deg, vaa_deg);
    c_of(coef, numerator(coef, sza_norm_deg), kv, kg)
}

/// A data type of the batch: read as f64 (`Sample`), written back from f64.
pub trait Real: Sample {
    fn from_f64(v: f64) -> Self;
}

impl Real for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

impl Real for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

/// The sun and view angles of the pixels, in degrees, one slice each (all the same length).
#[derive(Clone, Copy)]
pub struct Geometry<'a, T> {
    pub sza: &'a [T],
    pub saa: &'a [T],
    pub vza: &'a [T],
    pub vaa: &'a [T],
}

impl<T> Geometry<'_, T> {
    fn len(&self) -> usize {
        let n = self.sza.len();
        assert!(
            self.saa.len() == n && self.vza.len() == n && self.vaa.len() == n,
            "the four angles have as many pixels"
        );
        n
    }
}

/// Pixels per task.
const BLOCK: usize = 1024;

/// The kernels of every pixel, `(k_vol, k_geo)` interleaved.
fn kernels_batch<T: Real>(geo: &Geometry<T>, n_jobs: i32) -> Vec<f64> {
    let n = geo.len();
    let mut k = uninit_f64(2 * n);
    for_each_row(
        &mut k,
        2 * BLOCK,
        n_jobs,
        || (),
        |_, block, rows| {
            let first = block * BLOCK;
            for (i, pair) in rows.chunks_mut(2).enumerate() {
                let p = first + i;
                let (kv, kg) = kernels_deg(
                    geo.sza[p].to_f64(),
                    geo.saa[p].to_f64(),
                    geo.vza[p].to_f64(),
                    geo.vaa[p].to_f64(),
                );
                pair[0] = kv;
                pair[1] = kg;
            }
        },
    );
    k
}

/// `out[p] = value(p, c)` for the band of `coef`, `c` its c-factor at pixel `p`.
fn band<T: Real>(
    kernels: &[f64],
    coef: RossLi,
    sza_norm_deg: f64,
    out: &mut [T],
    n_jobs: i32,
    value: impl Fn(usize, f64) -> f64 + Sync,
) {
    let num = numerator(coef, sza_norm_deg);
    for_each_row(
        out,
        BLOCK,
        n_jobs,
        || (),
        |_, block, rows| {
            let first = block * BLOCK;
            for (i, o) in rows.iter_mut().enumerate() {
                let p = first + i;
                *o = T::from_f64(value(p, c_of(coef, num, kernels[2 * p], kernels[2 * p + 1])));
            }
        },
    );
}

/// The c-factor of every pixel for each band of `coefs` (bands x pixels in `out`), for the
/// normalized solar zenith `sza_norm_deg`. NaN where the geometry is outside the model's domain.
pub fn cfactor_bands<T: Real>(geo: &Geometry<T>, coefs: &[RossLi], sza_norm_deg: f64, out: &mut [T], n_jobs: i32) {
    let n = geo.len();
    assert_eq!(out.len(), coefs.len() * n, "out is bands x pixels");
    if n == 0 {
        return;
    }
    let k = kernels_batch(geo, n_jobs);
    for (coef, o) in coefs.iter().zip(out.chunks_mut(n)) {
        band(&k, *coef, sza_norm_deg, o, n_jobs, |_, c| c);
    }
}

/// The NBAR of the bands of `refl` (bands x pixels, one `coefs` entry per band), into `out` (the
/// same shape): the reflectance times the c-factor of its band, NaN where the reflectance is
/// NaN or the geometry is outside the model's domain. With `c_out` (the same shape) the
/// c-factors are written too. The pixels' kernels are computed once for all the bands.
pub fn nbar_bands<T: Real>(
    refl: &[T],
    geo: &Geometry<T>,
    coefs: &[RossLi],
    sza_norm_deg: f64,
    out: &mut [T],
    c_out: Option<&mut [T]>,
    n_jobs: i32,
) {
    let n = geo.len();
    assert_eq!(refl.len(), coefs.len() * n, "refl is bands x pixels");
    assert_eq!(out.len(), refl.len(), "out is bands x pixels");
    if n == 0 {
        return;
    }
    let k = kernels_batch(geo, n_jobs);
    for ((coef, r), o) in coefs.iter().zip(refl.chunks(n)).zip(out.chunks_mut(n)) {
        band(&k, *coef, sza_norm_deg, o, n_jobs, |p, c| c * r[p].to_f64());
    }
    if let Some(c_out) = c_out {
        assert_eq!(c_out.len(), refl.len(), "c_out is bands x pixels");
        for (coef, o) in coefs.iter().zip(c_out.chunks_mut(n)) {
            band(&k, *coef, sza_norm_deg, o, n_jobs, |_, c| c);
        }
    }
}

/// `nbar_bands` for one band.
pub fn nbar_batch<T: Real>(
    refl: &[T],
    geo: &Geometry<T>,
    coef: RossLi,
    sza_norm_deg: f64,
    out: &mut [T],
    n_jobs: i32,
) {
    nbar_bands(refl, geo, &[coef], sza_norm_deg, out, None, n_jobs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: f64 = PI / 180.0;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn k(ts: f64, tv: f64, phi: f64) -> (f64, f64) {
        kernels(ts * D, tv * D, phi * D)
    }

    // Specification tables: kernels (sza, vza, raa) -> (k_vol, k_geo), 6 decimals.
    const KERNEL_TABLE: [([f64; 3], [f64; 2]); 10] = [
        ([0.0, 0.0, 0.0], [0.000000, 0.000000]),
        ([30.0, 0.0, 0.0], [-0.031443, -0.698222]),
        ([30.0, 0.0, 180.0], [-0.031443, -0.698222]),
        ([30.0, 7.5, 0.0], [0.006595, -0.512314]),
        ([30.0, 7.5, 90.0], [-0.032118, -0.718905]),
        ([30.0, 7.5, 180.0], [-0.066297, -0.870441]),
        ([45.0, 10.0, 0.0], [0.018369, -0.870292]),
        ([45.0, 10.0, 180.0], [-0.094152, -1.291603]),
        ([60.0, 10.0, 180.0], [-0.072121, -1.652704]),
        ([30.0, 7.5, 100.0], [-0.038351, -0.748224]),
    ];

    // c-factor table: band, (sza, vza, raa), sza_norm, rho_obs, rho_norm, c.
    type Row = (&'static str, [f64; 3], f64, f64, f64, f64);
    const C_TABLE: [Row; 14] = [
        ("red", [30.0, 7.5, 0.0], 30.0, 0.157749, 0.151346, 0.959407),
        ("red", [30.0, 7.5, 180.0], 30.0, 0.145436, 0.151346, 1.040636),
        ("red", [30.0, 7.5, 90.0], 30.0, 0.150837, 0.151346, 1.003370),
        ("red", [45.0, 10.0, 180.0], 45.0, 0.134276, 0.141243, 1.051881),
        ("red", [40.0, 10.0, 45.0], 35.0, 0.150523, 0.148025, 0.983404),
        ("red", [30.0, 0.0, 123.0], 30.0, 0.151346, 0.151346, 1.000000),
        ("nir", [30.0, 7.5, 0.0], 30.0, 0.293406, 0.281432, 0.959190),
        ("nir", [30.0, 7.5, 180.0], 30.0, 0.270399, 0.281432, 1.040804),
        ("nir", [30.0, 7.5, 90.0], 30.0, 0.280646, 0.281432, 1.002801),
        ("nir", [45.0, 10.0, 180.0], 45.0, 0.252225, 0.265735, 1.053565),
        ("nir", [40.0, 10.0, 45.0], 35.0, 0.282383, 0.276157, 0.977951),
        ("blue", [30.0, 7.5, 0.0], 30.0, 0.073598, 0.070714, 0.960818),
        ("blue", [30.0, 7.5, 180.0], 30.0, 0.068057, 0.070714, 1.039042),
        ("blue", [45.0, 10.0, 180.0], 45.0, 0.063694, 0.066950, 1.051122),
    ];

    #[test]
    fn kernels_match_the_specification_table() {
        let mut worst = 0.0f64;
        for ([ts, tv, phi], [kv, kg]) in KERNEL_TABLE {
            let (a, b) = k(ts, tv, phi);
            worst = worst.max((a - kv).abs()).max((b - kg).abs());
            assert!((a - kv).abs() < 1e-6, "kvol({ts},{tv},{phi}) = {a}, want {kv}");
            assert!((b - kg).abs() < 1e-6, "kgeo({ts},{tv},{phi}) = {b}, want {kg}");
        }
        // (30, 7.5, 260) is (30, 7.5, 100)
        let (a, b) = k(30.0, 7.5, 260.0);
        assert!((a - -0.038351).abs() < 1e-6 && (b - -0.748224).abs() < 1e-6);
        eprintln!("kernel table: worst error {worst:.2e}");
    }

    fn named(name: &str) -> RossLi {
        role(name).unwrap()
    }

    #[test]
    fn c_factors_match_the_specification_table() {
        let mut worst = 0.0f64;
        for (name, [ts, tv, phi], norm, _, _, c) in C_TABLE {
            // the table has the saa at 0 and the vaa at the relative azimuth
            let got = cfactor(named(name), ts, 0.0, tv, phi, norm);
            worst = worst.max((got - c).abs());
            assert!((got - c).abs() < 1e-6, "{name} ({ts},{tv},{phi}) norm {norm}: c = {got}, want {c}");
        }
        eprintln!("c-factor table: worst error {worst:.2e}");
    }

    #[test]
    fn the_model_reflectances_match_the_specification_table() {
        let mut worst = 0.0f64;
        for (name, [ts, tv, phi], norm, obs, normed, _) in C_TABLE {
            let coef = named(name);
            let (kv, kg) = k(ts, tv, phi);
            let (nv, ng) = k(norm, 0.0, 0.0);
            let (o, n) = (coef.reflectance(kv, kg), coef.reflectance(nv, ng));
            worst = worst.max((o - obs).abs()).max((n - normed).abs());
            assert!((o - obs).abs() < 1e-6, "{name} rho_obs {o} vs {obs}");
            assert!((n - normed).abs() < 1e-6, "{name} rho_norm {n} vs {normed}");
        }
        eprintln!("rho table: worst error {worst:.2e}");
    }

    // property 1
    #[test]
    fn c_is_one_at_nadir_and_the_normalized_zenith_for_any_azimuth() {
        for name in ROLES {
            let coef = role(name).unwrap();
            for phi in [0.0, 17.0, 123.0, 180.0, 270.0, 359.0] {
                let c = cfactor(coef, 30.0, 10.0, 0.0, 10.0 + phi, 30.0);
                assert!((c - 1.0).abs() < 1e-12, "{name} {phi}: {c}");
            }
        }
    }

    // property 2
    #[test]
    fn the_kernels_vanish_at_nadir_with_the_sun_at_nadir() {
        for phi in [0.0, 33.0, 90.0, 180.0, -70.0] {
            let (v, g) = k(0.0, 0.0, phi);
            assert!(v.abs() < 1e-15 && g.abs() < 1e-15, "{phi}: {v} {g}");
        }
    }

    // property 3
    #[test]
    fn the_kernels_are_symmetric_in_the_azimuth() {
        let mut r = Rng(5);
        for _ in 0..500 {
            let (ts, tv, phi) = (r.next() * 80.0, r.next() * 40.0, r.next() * 360.0);
            let (a, b) = k(ts, tv, phi);
            for other in [360.0 - phi, -phi, phi + 360.0] {
                let (c, d) = k(ts, tv, other);
                assert!((a - c).abs() < 1e-12 && (b - d).abs() < 1e-12, "{ts} {tv} {phi} vs {other}");
            }
        }
    }

    // property 4
    #[test]
    fn the_numerator_does_not_depend_on_the_azimuth() {
        for ts in [0.0, 20.0, 45.0, 70.0] {
            let base = kernels(ts * D, 0.0, 0.0);
            for phi in [10.0, 90.0, 200.0] {
                let other = kernels(ts * D, 0.0, phi * D);
                assert!((base.0 - other.0).abs() < 1e-12 && (base.1 - other.1).abs() < 1e-12, "{ts} {phi}");
            }
        }
    }

    // property 5
    #[test]
    fn without_kernel_weights_c_is_one_for_any_geometry() {
        let iso = RossLi::new(0.2, 0.0, 0.0);
        let mut r = Rng(9);
        for _ in 0..300 {
            let c = cfactor(iso, r.next() * 80.0, r.next() * 360.0, r.next() * 30.0, r.next() * 360.0, 20.0 + r.next() * 40.0);
            assert!((c - 1.0).abs() < 1e-12, "{c}");
        }
    }

    // property 6
    #[test]
    fn the_red_edge_is_the_published_interpolation() {
        let table = [
            (705.0, [0.2085, 0.0256, 0.0845]),
            (740.0, [0.2316, 0.0273, 0.1003]),
            (783.0, [0.2599, 0.0294, 0.1197]),
        ];
        for (nm, [i, g, v]) in table {
            let p = rededge(nm);
            let near = |a: f64, b: f64| (a - b).abs() < 5e-5 + 1e-12;
            assert!(near(p.f_iso, i) && near(p.f_geo, g) && near(p.f_vol, v), "{nm}: {p:?}");
        }
        assert_eq!(rededge(RED_NM), RED);
        assert!((rededge(NIR_NM).f_iso - NIR.f_iso).abs() < 1e-15);
        assert_eq!(role("rededge2").unwrap(), rededge(740.0));
    }

    // property 8
    #[test]
    fn the_coefficients_are_roy_2016_table_5_and_hls_table_4() {
        let t = |p: RossLi| (p.f_iso, p.f_geo, p.f_vol);
        assert_eq!(t(role("blue").unwrap()), (0.0774, 0.0079, 0.0372));
        assert_eq!(t(role("coastal").unwrap()), (0.0774, 0.0079, 0.0372));
        assert_eq!(t(role("green").unwrap()), (0.1306, 0.0178, 0.0580));
        assert_eq!(t(role("red").unwrap()), (0.1690, 0.0227, 0.0574));
        assert_eq!(t(role("nir").unwrap()), (0.3093, 0.0330, 0.1535));
        assert_eq!(t(role("nir08").unwrap()), (0.3093, 0.0330, 0.1535));
        assert_eq!(t(role("swir1").unwrap()), (0.3430, 0.0453, 0.1154));
        assert_eq!(t(role("swir2").unwrap()), (0.2658, 0.0387, 0.0639));
        assert!(role("nir09").is_none());
        assert!(ROLES.iter().all(|r| role(r).is_some()));
    }

    fn scene(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut r = Rng(seed);
        let sza: Vec<f32> = (0..n).map(|_| (20.0 + r.next() * 40.0) as f32).collect();
        let saa: Vec<f32> = (0..n).map(|_| (100.0 + r.next() * 40.0) as f32).collect();
        let vza: Vec<f32> = (0..n).map(|_| (r.next() * 12.0) as f32).collect();
        let vaa: Vec<f32> = (0..n).map(|_| (r.next() * 360.0) as f32).collect();
        let refl: Vec<f32> = (0..n).map(|_| (0.02 + r.next() * 0.4) as f32).collect();
        (sza, saa, vza, vaa, refl)
    }

    // properties 7 (NaN, linearity) and the batch against the scalar function
    #[test]
    fn the_batch_is_the_scalar_function_and_nan_in_is_nan_out() {
        let n = 5000;
        let (sza, saa, mut vza, vaa, mut refl) = scene(n, 3);
        vza[10] = f32::NAN;
        refl[20] = f32::NAN;
        vza[30] = 95.0;
        let mut sza = sza;
        sza[40] = 90.0;
        sza[50] = -1.0;
        let geo = Geometry { sza: &sza, saa: &saa, vza: &vza, vaa: &vaa };
        let coefs = [RED, NIR, SWIR2];
        let all: Vec<f32> = coefs.iter().flat_map(|_| refl.iter().copied()).collect();
        let (mut out, mut c) = (vec![0.0f32; 3 * n], vec![0.0f32; 3 * n]);
        nbar_bands(&all, &geo, &coefs, 35.0, &mut out, Some(&mut c), 1);
        for b in 0..3 {
            for p in 0..n {
                let want = cfactor(
                    coefs[b],
                    f64::from(sza[p]),
                    f64::from(saa[p]),
                    f64::from(vza[p]),
                    f64::from(vaa[p]),
                    35.0,
                );
                let got = f64::from(c[b * n + p]);
                if [10, 30, 40, 50].contains(&p) {
                    assert!(got.is_nan() && out[b * n + p].is_nan(), "pixel {p}");
                } else {
                    assert!((got - want).abs() < 1e-6, "pixel {p}: {got} vs {want}");
                    if p != 20 {
                        assert!((f64::from(out[b * n + p]) - want * f64::from(refl[p])).abs() < 1e-6);
                    }
                }
            }
            assert!(out[b * n + 20].is_nan());
            assert!(!c[b * n + 20].is_nan(), "a NaN reflectance does not blank the c-factor");
        }
        // linear in the reflectance
        let doubled: Vec<f32> = refl.iter().map(|v| v * 2.0).collect();
        let (mut a, mut b) = (vec![0.0f32; n], vec![0.0f32; n]);
        nbar_batch(&refl, &geo, RED, 35.0, &mut a, 1);
        nbar_batch(&doubled, &geo, RED, 35.0, &mut b, 1);
        for p in 0..n {
            assert!(
                (b[p] - 2.0 * a[p]).abs() <= 1e-6 * a[p].abs() || (a[p].is_nan() && b[p].is_nan()),
                "{p}: {} {}",
                a[p],
                b[p]
            );
        }
        // an invalid normalized zenith blanks everything
        nbar_batch(&refl, &geo, RED, 90.0, &mut a, 1);
        assert!(a.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn threads_and_types_agree() {
        let n = 20_000;
        let (sza, saa, vza, vaa, refl) = scene(n, 8);
        let geo = Geometry { sza: &sza, saa: &saa, vza: &vza, vaa: &vaa };
        let coefs = [BLUE, RED];
        let two: Vec<f32> = refl.iter().chain(refl.iter()).copied().collect();
        let (mut one, mut many) = (vec![0.0f32; 2 * n], vec![0.0f32; 2 * n]);
        nbar_bands(&two, &geo, &coefs, 30.0, &mut one, None, 1);
        nbar_bands(&two, &geo, &coefs, 30.0, &mut many, None, 3);
        assert_eq!(one, many);
        let d = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        let (s64, a64, z64, v64) = (d(&sza), d(&saa), d(&vza), d(&vaa));
        let geo64 = Geometry { sza: &s64, saa: &a64, vza: &z64, vaa: &v64 };
        let r64: Vec<f64> = two.iter().map(|&x| f64::from(x)).collect();
        let mut o64 = vec![0.0f64; 2 * n];
        nbar_bands(&r64, &geo64, &coefs, 30.0, &mut o64, None, 2);
        for i in 0..2 * n {
            assert!((o64[i] - f64::from(one[i])).abs() < 1e-6);
        }
        let mut c = vec![0.0f64; 2 * n];
        cfactor_bands(&geo64, &coefs, 30.0, &mut c, 2);
        for i in 0..2 * n {
            assert!((c[i] * r64[i] - o64[i]).abs() < 1e-12);
        }
        // empty input
        let e: [f32; 0] = [];
        nbar_bands(&e, &Geometry { sza: &e, saa: &e, vza: &e, vaa: &e }, &coefs[..0], 30.0, &mut [], None, 1);
    }

    #[test]
    fn c_stays_near_one_in_a_landsat_view_field() {
        // Roy 2016 Fig. 6: about 0.95-1.05 within +-7.5 degrees
        let mut r = Rng(11);
        for _ in 0..2000 {
            let c = cfactor(RED, 25.0 + r.next() * 20.0, 100.0, r.next() * 7.5, r.next() * 360.0, 35.0);
            assert!(c > 0.9 && c < 1.1, "{c}");
        }
    }
}
