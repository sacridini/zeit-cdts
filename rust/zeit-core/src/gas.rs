//! Atmospheric gases of FORCE's atmospheric correction: the water vapour retrieval of Sentinel-2, the gas
//! transmittance map, and the water vapour look-up tables for the sensors without a vapour band. A port of FORCE
//! 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring, Copyright (C) 2013-2022
//! David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force), `src/modules/lower-level/gas-ll.c`:
//! `wvp_transmitt_lut`, `fun_wvp`, `water_vapor`, `gas_transmittance` and `water_vapor_from_lut` (`ozone_amount` is
//! in [`crate::radtran`]). The minimizer of `water_vapor` is GSL 2.7's `nmsimplex2` in one dimension
//! ([`crate::gsl_nmsimplex`]).
//!
//! The arithmetic is the C's, as in [`crate::radtran`]: C `float` is `f32`, with the promotion to `double` where C
//! promotes (a `/ 10000.0`, `floor`, `sqrt`), and the order of the operations. Where this differs from the C:
//!
//! * **Order of the pixels.** FORCE runs the 60 m pixels in an OpenMP `for schedule(guided)`; each thread keeps the
//!   water vapour of its last pixel as the start of the next pixel's minimization (`w`), and sums `w_sum` and `num`
//!   per thread. Both make FORCE's output depend on the number of threads and on how the chunks fall (the estimate
//!   moves by up to the minimizer's tolerance, about 0.01 cm). This port is FORCE with one thread: the 60 m blocks in
//!   row-major order, one warm start carried over, one sum, in `f64` in that order. It is deterministic and equals
//!   FORCE run with `OMP_NUM_THREADS=1` bit for bit.
//! * **The LUT** is built by [`WvLut::build`], not inside `water_vapor` (FORCE keeps it in a global, `_WVLUT_`, and
//!   rebuilds it on each `water_vapor` call); the caller builds it once and passes it to [`water_vapor`] and
//!   [`gas_transmittance`], and `free_wvlut` is the drop of the [`WvLut`].
//! * **Blocks at the edge.** FORCE reads and writes six pixels past the border when the image is not a multiple of 6:
//!   for a width that is not, the pixels past the end of a row are the first columns of the next one (`gas_transmittance`
//!   gives those the transmittance of the last block of the row above, `water_vapor` averages them into that block), and
//!   past the end of the array for a height that is not (undefined behaviour). Here the blocks are cut at the border. A
//!   Sentinel-2 image is a multiple of 6 pixels (1830, 5490, 10980), where the two are the same; the comparison with
//!   FORCE uses sizes that are multiples of 6 (`force_l2.py` found the wrap-around).
//! * **Errors.** Where the C returns `NULL` or `-1` after a message, or GSL aborts, this returns a [`GasError`] (or
//!   skips the pixel, for a non-finite value of the minimized function, which only an inconsistent `atc` produces).
//! * **Scene centre.** `water_vapor_from_lut` takes the scene centre's longitude and latitude and the date as
//!   arguments (FORCE gets them from the projection and the date of the `xy_sun` brick). A line of a LUT with fewer
//!   than three fields (a blank line, a comment line) crashes FORCE (`atof(NULL)`); here it is skipped.

use crate::atc::{Atc, C_ZEN};
use crate::gsl_nmsimplex::minimize;
use crate::parallel::{pool, threads};
use crate::qai;
use crate::radtran::{gas_transmitt, wvp_transmitt};
use rayon::prelude::*;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Number of columns of the LUT's water vapour axis: 0 to 7 cm in steps of 0.01 (`nw`).
pub const LUT_NW: usize = 701;
/// Number of entries of its air mass axis: cosines 0 to 1 in steps of 0.01 (`nm`).
pub const LUT_NM: usize = 101;
/// `NPOW_10` of FORCE: the size of its string buffers (a LUT file name, a line of a LUT).
const NPOW_10: usize = 1024;

/// What can go wrong in these functions (where FORCE prints a message and returns `NULL` or `-1`).
#[derive(Debug, Clone, PartialEq)]
pub enum GasError {
    /// The cosine range of the sun and view zenith does not fit the LUT's air mass axis (0 to 1).
    AirMassRange,
    /// The TOA brick has no band of this domain (`find_domain`).
    NoDomain(&'static str),
    /// The inputs do not have the sizes they claim.
    Shape(String),
    /// `Buffer Overflow in assembling filename`.
    FileName,
    /// `Unable to open LUT`.
    OpenLut(String),
    /// `no daily wvp value available` (strict checking).
    NoDailyValue,
}

impl fmt::Display for GasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GasError::AirMassRange => write!(f, "the air mass range does not fit the water vapour LUT"),
            GasError::NoDomain(d) => write!(f, "the TOA brick has no band of domain {d}"),
            GasError::Shape(s) => write!(f, "{s}"),
            GasError::FileName => write!(f, "buffer overflow in assembling filename"),
            GasError::OpenLut(p) => write!(f, "unable to open LUT {p}"),
            GasError::NoDailyValue => write!(f, "no daily wvp value available"),
        }
    }
}

impl std::error::Error for GasError {}

/// The grid of a full-resolution brick (FORCE's TOA, QAI, DEM and WVP bricks): `ny` rows of `nx` pixels, `res` metres.
#[derive(Clone, Copy, Debug)]
pub struct SceneGrid {
    pub nx: usize,
    pub ny: usize,
    pub res: f64,
}

/// The TOA brick: reflectance times 10000 per band (`short`), and the domain of each band (`"NIR"`, `"VAPOR"`, ...).
#[derive(Clone, Debug)]
pub struct Toa {
    pub grid: SceneGrid,
    pub bands: Vec<Vec<i16>>,
    pub domain: Vec<String>,
}

impl Toa {
    /// `find_domain`: the first band of this domain.
    pub fn find_domain(&self, domain: &str) -> Option<usize> {
        self.domain.iter().position(|d| d == domain)
    }
}

/// `convert_brick_ji2p`: the cell of the coarse grid (`to_nx` cells a row, `to_res` metres) under the pixel `(i, j)` of
/// a grid of `from_res` metres.
#[inline]
pub(crate) fn ji2p(from_res: f64, to_res: f64, to_nx: usize, i: usize, j: usize) -> usize {
    let i_to = (i as f64 * from_res / to_res).floor() as i64;
    let j_to = (j as f64 * from_res / to_res).floor() as i64;
    (i_to * to_nx as i64 + j_to) as usize
}

/// `fequal` of `utils-cl.c`: equal within `FLT_EPSILON` of the larger magnitude.
#[inline]
pub(crate) fn fequal(a: f32, b: f32) -> bool {
    let diff = (a - b).abs();
    let (aa, bb) = (a.abs(), b.abs());
    let max = if bb > aa { bb } else { aa };
    diff <= max * f32::EPSILON
}

// ---------------------------------------------------------------------------------------------------------------
// wvp_transmitt_lut
// ---------------------------------------------------------------------------------------------------------------

/// The water vapour transmittance look-up table (`_WVLUT_`): the one-way transmittance of each band for the water
/// vapour `w = kw * 0.01` cm (`kw` in 0..701) along an air mass `m = km * 0.01` (`km` in 0..101), `val[b][kw][km]`.
/// Only the `km` between the smallest and largest cosine of the image's zenith angles are computed; the others are 0.
#[derive(Clone, Debug)]
pub struct WvLut {
    pub nb: usize,
    val: Vec<f32>,
}

impl WvLut {
    /// `wvp_transmitt_lut`: the table for the bands `rsr_band[b]` (`meta->cal[b].rsr_band`; its first `nb` of the
    /// `atc.xy_tg` are used), for the zenith cosines `atc.cosszen` and `atc.cosvzen`. `n_jobs` as everywhere in zeit
    /// (every entry is independent: the result does not depend on it).
    pub fn build(rsr_band: &[usize], atc: &Atc, n_jobs: i32) -> Result<Self, GasError> {
        let nb = atc.xy_tg.nb();
        if rsr_band.len() < nb {
            return Err(GasError::Shape(format!("{} RSR bands for {nb} bands", rsr_band.len())));
        }
        let m_min = if atc.cosszen[0] < atc.cosvzen[0] { atc.cosszen[0] } else { atc.cosvzen[0] };
        let km_min = (m_min as f64 * 100.0).floor();
        let m_max = if atc.cosszen[1] > atc.cosvzen[1] { atc.cosszen[1] } else { atc.cosvzen[1] };
        let km_max = (m_max as f64 * 100.0).ceil();
        // (NaN fails both tests)
        if !(km_min >= 0.0 && km_max < LUT_NM as f64) {
            return Err(GasError::AirMassRange);
        }
        let (km_min, km_max) = (km_min as usize, km_max as usize);
        let mut val = vec![0f32; nb * LUT_NW * LUT_NM];
        let row = |r: usize, out: &mut [f32]| {
            let (b, kw) = (r / LUT_NW, r % LUT_NW);
            let w = (kw as f64 * 0.01) as f32;
            for km in km_min..=km_max {
                let m = (km as f64 * 0.01) as f32;
                out[km] = wvp_transmitt(w, m, rsr_band[b]);
            }
        };
        let t = threads(n_jobs);
        if t <= 1 {
            val.chunks_mut(LUT_NM).enumerate().for_each(|(r, o)| row(r, o));
        } else {
            pool(t).install(|| val.par_chunks_mut(LUT_NM).enumerate().for_each(|(r, o)| row(r, o)));
        }
        Ok(WvLut { nb, val })
    }

    /// `_WVLUT_.val[b][kw][km]`; NaN where the C would read out of the table (an index outside 0..701, 0..101).
    #[inline]
    pub fn get(&self, b: usize, kw: i32, km: i32) -> f32 {
        if b < self.nb && (0..LUT_NW as i32).contains(&kw) && (0..LUT_NM as i32).contains(&km) {
            self.val[(b * LUT_NW + kw as usize) * LUT_NM + km as usize]
        } else {
            f32::NAN
        }
    }

    /// The table as `val[b][kw][km]` flattened (for tests and the comparison with FORCE).
    pub fn values(&self) -> &[f32] {
        &self.val
    }
}

// ---------------------------------------------------------------------------------------------------------------
// fun_wvp, water_vapor
// ---------------------------------------------------------------------------------------------------------------

/// The `param` array of `fun_wvp` (the 16 floats of the C, with the bands as integers).
struct WvpParams {
    nir: f32,
    wvp: f32,
    b_nir: usize,
    b_wvp: usize,
    b_nir_rsr: usize,
    b_wvp_rsr: usize,
    ms: f32,
    mv: f32,
    to_nir: f32,
    to_wvp: f32,
    rho_p_nir: f32,
    rho_p_wvp: f32,
    t_nir: f32,
    t_wvp: f32,
    s_nir: f32,
    s_wvp: f32,
}

/// `fun_wvp`: the absolute difference of the BOA reflectance of the reference (NIR) and the measurement (vapour)
/// channel for the water vapour `v`.
fn fun_wvp(lut: &WvLut, v: f64, p: &WvpParams) -> f64 {
    let mut w = v as f32;
    // negative water vapour values are not allowed
    if w < 0.0 {
        w = 0.0;
    }
    let (tsw_nir, tvw_nir, tsw_wvp, tvw_wvp);
    if w > 7.0 {
        // beyond the tabulated values: compute
        tsw_nir = wvp_transmitt(w, p.ms, p.b_nir_rsr);
        tvw_nir = wvp_transmitt(w, p.mv, p.b_nir_rsr);
        tsw_wvp = wvp_transmitt(w, p.ms, p.b_wvp_rsr);
        tvw_wvp = wvp_transmitt(w, p.mv, p.b_wvp_rsr);
    } else {
        let kw = (w as f64 / 0.01).floor() as i32;
        let kms = (p.ms as f64 / 0.01).floor() as i32;
        let kmv = (p.mv as f64 / 0.01).floor() as i32;
        tsw_nir = lut.get(p.b_nir, kw, kms);
        tvw_nir = lut.get(p.b_nir, kw, kmv);
        tsw_wvp = lut.get(p.b_wvp, kw, kms);
        tvw_wvp = lut.get(p.b_wvp, kw, kmv);
    }
    let tg_nir = tsw_nir * tvw_nir * p.to_nir;
    let tg_wvp = tsw_wvp * tvw_wvp * p.to_wvp;

    let tmp = (p.nir - p.rho_p_nir) / tg_nir;
    let sr_reference = tmp / (p.t_nir + p.s_nir * tmp);
    let tmp = (p.wvp - p.rho_p_wvp) / tg_wvp;
    let sr_measure = tmp / (p.t_wvp + p.s_wvp * tmp);

    ((sr_reference - sr_measure) as f64).abs()
}

/// The result of [`water_vapor`].
#[derive(Clone, Debug)]
pub struct WaterVapor {
    /// Water vapour in cm times 1000 (`short`), 0 where there is none (pixels that are off, or no estimate and no
    /// scene average).
    pub wvp: Vec<i16>,
    /// The scene average of the retrieved 60 m values (`w_avg`) that filled the water, shadow and failed blocks;
    /// `None` if no block was retrieved (nothing is filled then).
    pub w_avg: Option<f32>,
    /// Number of 60 m blocks retrieved (`num`).
    pub n: usize,
}

/// `water_vapor`: the water vapour of each 60 m block (6 x 6 pixels) of a Sentinel-2 image, by the complete radiative
/// transfer, assuming that the BOA reflectance of the NIR reference band (domain `"NIR"`, 0.865 um) and the NIR
/// vapour band (domain `"VAPOR"`, 0.945 um) are equal, by a Nelder-Mead simplex ([`crate::gsl_nmsimplex`], start 3
/// cm or the last estimate, step 0.25, tolerance 0.01, at most 100 iterations). The reference reflectance is the mean
/// of the pixels of the block that are not off, shadow or water and have a TOA reflectance of at least 0.15, the
/// elevation (to select the elevation bin of the path reflectance, transmittance and albedo) their mean `dem`. A
/// block without such a pixel, in a cell of `atc.xy_view` with no data, or whose estimate is not in (0, 7] cm, gets no
/// value; the water vapour is replicated to the 36 pixels that are not off. Pixels that got none (and are not off) are
/// then set to the mean of the retrieved values and flagged in `qai` (`set_vaporfill`).
///
/// `rsr_band[b]` is `meta->cal[b].rsr_band`, `qai` the QAI of the image (`grid`), `dem` the DEM in metres/10 or
/// whatever unit the elevation bins of the `atc` are in (`small`, 0..256). See the module's notes on the order of the
/// pixels.
pub fn water_vapor(
    rsr_band: &[usize],
    atc: &Atc,
    lut: &WvLut,
    toa: &Toa,
    qai: &mut [i16],
    dem: &[u8],
) -> Result<WaterVapor, GasError> {
    let grid = toa.grid;
    let (nx, ny) = (grid.nx, grid.ny);
    let npix = nx * ny;
    if qai.len() != npix || dem.len() != npix || toa.bands.iter().any(|b| b.len() != npix) {
        return Err(GasError::Shape("the TOA, QAI and DEM bricks differ in size".into()));
    }
    let b_reference = toa.find_domain("NIR").ok_or(GasError::NoDomain("NIR"))?;
    let b_measure = toa.find_domain("VAPOR").ok_or(GasError::NoDomain("VAPOR"))?;
    let nb = atc.xy_tso.nb();
    if b_reference >= nb || b_measure >= nb || rsr_band.len() <= b_reference.max(b_measure) {
        return Err(GasError::Shape("the bands of the TOA brick are not in the atc".into()));
    }
    if atc.xyz_rho_p.len() < crate::atc::N_ELEV || atc.xyz_t.len() < crate::atc::N_ELEV || atc.xyz_s.len() < crate::atc::N_ELEV {
        return Err(GasError::Shape("the atc needs 256 elevation bins".into()));
    }

    let xy_ms = atc.xy_sun.band(C_ZEN);
    let xy_mv = atc.xy_view.band(C_ZEN);
    let xy_tvo_r = atc.xy_tvo.band(b_reference);
    let xy_tvo_m = atc.xy_tvo.band(b_measure);
    let xy_tso_r = atc.xy_tso.band(b_reference);
    let xy_tso_m = atc.xy_tso.band(b_measure);
    let toa_r = &toa.bands[b_reference];
    let toa_m = &toa.bands[b_measure];
    let nodata_view = atc.xy_view.nodata as f32;

    let mut wvp = vec![0i16; npix];
    let (mut w_sum, mut num) = (0f64, 0f64);
    // start value, set to the last value in the loop
    let mut w = 3.0f32;

    let mut i = 0;
    while i < ny {
        let mut j = 0;
        while j < nx {
            let (bi, bj) = ((ny - i).min(6), (nx - j).min(6));

            // aggregate the reference channel to 60 m
            let (mut reference, mut dem_sum, mut k) = (0f32, 0f32, 0i32);
            for ii in 0..bi {
                for jj in 0..bj {
                    let p = (i + ii) * nx + j + jj;
                    if qai::off(qai[p]) || qai::shadow(qai[p]) || qai::water(qai[p]) || toa_r[p] < 1500 {
                        continue;
                    }
                    reference = (reference as f64 + toa_r[p] as f64 / 10000.0) as f32;
                    dem_sum += dem[p] as f32;
                    k += 1;
                }
            }
            if k == 0 {
                j += 6;
                continue;
            }
            reference /= k as f32;
            let z = (dem_sum / k as f32).round() as i32;

            let p = i * nx + j;
            let measure = (toa_m[p] as f64 / 10000.0) as f32;
            let g = ji2p(grid.res, atc.xy_view.res, atc.xy_view.nx, i, j);
            if fequal(xy_mv[g], nodata_view) {
                j += 6;
                continue;
            }
            let z = z as usize; // 0..=255 (a mean of bytes)
            let params = WvpParams {
                nir: reference,
                wvp: measure,
                b_nir: b_reference,
                b_wvp: b_measure,
                b_nir_rsr: rsr_band[b_reference],
                b_wvp_rsr: rsr_band[b_measure],
                ms: xy_ms[g],
                mv: xy_mv[g],
                to_nir: xy_tso_r[g] * xy_tvo_r[g],
                to_wvp: xy_tso_m[g] * xy_tvo_m[g],
                rho_p_nir: atc.xyz_rho_p[z].band(b_reference)[g],
                rho_p_wvp: atc.xyz_rho_p[z].band(b_measure)[g],
                t_nir: atc.xyz_t[z].band(b_reference)[g],
                t_wvp: atc.xyz_t[z].band(b_measure)[g],
                s_nir: atc.xyz_s[z].band(b_reference)[g],
                s_wvp: atc.xyz_s[z].band(b_measure)[g],
            };

            // optimize the radiative transfer
            let mut f = |v: f64| fun_wvp(lut, v, &params);
            let Some(x) = minimize(&mut f, w as f64, 0.25, 1e-2, 100) else {
                j += 6;
                continue;
            };
            w = x as f32;
            if w > 7.0 || w <= 0.0 {
                w = 3.0;
                j += 6;
                continue;
            }
            w_sum += w as f64;
            num += 1.0;

            // replicate the value at the original resolution
            for ii in 0..bi {
                for jj in 0..bj {
                    let p = (i + ii) * nx + j + jj;
                    if qai::off(qai[p]) {
                        continue;
                    }
                    wvp[p] = (w * 1000.0) as i16;
                }
            }
            j += 6;
        }
        i += 6;
    }

    // fill water and shadow with the mean water vapour, set the QAI flag
    let w_avg = if num > 0.0 {
        let w_avg = (w_sum / num) as f32;
        for p in 0..npix {
            if qai::off(qai[p]) || wvp[p] > 0 {
                continue;
            }
            qai::set_vaporfill(&mut qai[p], true);
            wvp[p] = (w_avg * 1000.0) as i16;
        }
        Some(w_avg)
    } else {
        None
    };
    Ok(WaterVapor { wvp, w_avg, n: num as usize })
}

// ---------------------------------------------------------------------------------------------------------------
// gas_transmittance
// ---------------------------------------------------------------------------------------------------------------

/// `gas_transmittance`: the gaseous transmittance of band `b` times 10000 (`short`) for each pixel that is not off,
/// from the water vapour `wvp` (cm times 1000, as [`water_vapor`] returns it; the value of the top left pixel of each
/// 60 m block is used) through the LUT, and the ozone transmittance of the `atc`. Pixels off, and blocks in a cell of
/// `atc.xy_view` with no data, are 0.
pub fn gas_transmittance(
    atc: &Atc,
    b: usize,
    wvp: &[i16],
    grid: &SceneGrid,
    qai: &[i16],
    lut: &WvLut,
) -> Result<Vec<i16>, GasError> {
    let (nx, ny) = (grid.nx, grid.ny);
    let npix = nx * ny;
    if wvp.len() != npix || qai.len() != npix || b >= atc.xy_tso.nb() || b >= atc.xy_tvo.nb() {
        return Err(GasError::Shape("the WVP and QAI bricks differ in size, or the band is not in the atc".into()));
    }
    let xy_ms = atc.xy_sun.band(C_ZEN);
    let xy_mv = atc.xy_view.band(C_ZEN);
    let xy_tso = atc.xy_tso.band(b);
    let xy_tvo = atc.xy_tvo.band(b);
    let nodata_view = atc.xy_view.nodata as f32;
    let mut tg_ = vec![0i16; npix];

    let mut i = 0;
    while i < ny {
        let mut j = 0;
        while j < nx {
            let (bi, bj) = ((ny - i).min(6), (nx - j).min(6));
            let p = i * nx + j;
            let g = ji2p(grid.res, atc.xy_view.res, atc.xy_view.nx, i, j);
            if fequal(xy_mv[g], nodata_view) {
                j += 6;
                continue;
            }

            // gaseous transmittance
            let w = (wvp[p] as f64 / 1000.0) as f32;
            let kw = (w as f64 / 0.01).floor() as i32;
            let (ms, mv) = (xy_ms[g], xy_mv[g]);
            let kms = (ms as f64 / 0.01).floor() as i32;
            let kmv = (mv as f64 / 0.01).floor() as i32;
            let tsw = lut.get(b, kw, kms);
            let tvw = lut.get(b, kw, kmv);
            let tg = gas_transmitt(tsw, tvw, xy_tso[g], xy_tvo[g]);

            // replicate the value at the original resolution
            for ii in 0..bi {
                for jj in 0..bj {
                    let p = (i + ii) * nx + j + jj;
                    if qai::off(qai[p]) {
                        continue;
                    }
                    tg_[p] = (tg * 10000.0) as i16;
                }
            }
            j += 6;
        }
        i += 6;
    }
    Ok(tg_)
}

// ---------------------------------------------------------------------------------------------------------------
// water_vapor_from_lut
// ---------------------------------------------------------------------------------------------------------------

/// The part of FORCE's `par_ll_t` that `water_vapor_from_lut` reads.
#[derive(Clone, Debug)]
pub struct WvpSource {
    /// `d_wvp`: the directory of the LUT files; `None` is FORCE's `"NULL"`, to use the constant `wvp`.
    pub dir: Option<String>,
    /// `wvp`: the constant water vapour (cm) used when `dir` is `None`.
    pub wvp: f32,
    /// `wvp_strict`: fail when the daily LUT has no value.
    pub strict: bool,
}

/// C's `atof` on a token: the longest decimal (or `inf`/`nan`) prefix as a double, 0 if there is none.
pub(crate) fn atof(s: &[u8]) -> f64 {
    let mut k = 0;
    while k < s.len() && matches!(s[k], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        k += 1;
    }
    let start = k;
    if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
        k += 1;
    }
    let lower: Vec<u8> = s[k..].iter().take(8).map(|c| c.to_ascii_lowercase()).collect();
    if lower.starts_with(b"infinity") || lower.starts_with(b"inf") || lower.starts_with(b"nan") {
        let n = if lower.starts_with(b"infinity") { 8 } else { 3 };
        return String::from_utf8_lossy(&s[start..k + n]).parse::<f64>().unwrap_or(0.0);
    }
    let d0 = k;
    while k < s.len() && s[k].is_ascii_digit() {
        k += 1;
    }
    let mut nd = k - d0;
    if k < s.len() && s[k] == b'.' {
        k += 1;
        let f0 = k;
        while k < s.len() && s[k].is_ascii_digit() {
            k += 1;
        }
        nd += k - f0;
    }
    if nd == 0 {
        return 0.0;
    }
    if k < s.len() && (s[k] == b'e' || s[k] == b'E') {
        let mut e = k + 1;
        if e < s.len() && (s[e] == b'+' || s[e] == b'-') {
            e += 1;
        }
        let e0 = e;
        while e < s.len() && s[e].is_ascii_digit() {
            e += 1;
        }
        if e > e0 {
            k = e;
        }
    }
    String::from_utf8_lossy(&s[start..k]).parse::<f64>().unwrap_or(0.0)
}

/// The lines of a file as FORCE's `fgets(buffer, NPOW_10, fp)` reads them (at most 1023 bytes, ending after a newline),
/// cut at the first `\r`, `\n` or `#` (and at a NUL, where the C string ends).
fn fgets_lines(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let end = (pos + NPOW_10 - 1).min(data.len());
        let mut e = pos;
        while e < end {
            e += 1;
            if data[e - 1] == b'\n' {
                break;
            }
        }
        let chunk = &data[pos..e];
        pos = e;
        let cut = chunk.iter().position(|&c| matches!(c, b'\r' | b'\n' | b'#' | 0)).unwrap_or(chunk.len());
        out.push(&chunk[..cut]);
    }
    out
}

/// `water_vapor_from_lut`: the water vapour (cm) of an image from FORCE's LUT files in `src.dir`, or `src.wvp` if
/// `src.dir` is `None`. For the scene centre `(center_lon, center_lat)` (geographic, decimal degrees) and the date
/// of the image, the climatology `WVP_0000-MM-00.txt` and the daily `WVP_YYYY-MM-DD.txt` are read (each, if it
/// exists): plain text, one line per site, blank-separated `lon lat water_vapour source`, the value of the line
/// closest to the centre within 1.5 degrees is taken (the climatology's also as the fallback, `avg`, and the daily's
/// replaces it); a value over 10 (the nodata 9999) or below 0 falls back to `avg`. The result is -1 if nothing was
/// found (the caller must treat a negative value as a failure, as FORCE does); `Err` where FORCE prints and
/// returns -1 (file name too long, unreadable LUT, strict and no daily value).
pub fn water_vapor_from_lut(
    src: &WvpSource,
    center_lon: f64,
    center_lat: f64,
    year: i32,
    month: i32,
    day: i32,
) -> Result<f32, GasError> {
    let Some(dir) = &src.dir else {
        return Ok(src.wvp);
    };
    let mut wvp = -1f32;
    let mut avg = -1f32;
    let names = [format!("{dir}/WVP_{:04}-{:02}-{:02}.txt", 0, month, 0), format!("{dir}/WVP_{:04}-{:02}-{:02}.txt", year, month, day)];
    if names.iter().any(|n| n.len() >= NPOW_10) {
        return Err(GasError::FileName);
    }
    for (k, name) in names.iter().enumerate() {
        if !Path::new(name).exists() {
            continue;
        }
        let mut data = Vec::new();
        let mut fp = File::open(name).map_err(|_| GasError::OpenLut(name.clone()))?;
        // (a directory opens and reads as empty, as with fgets)
        let _ = fp.read_to_end(&mut data);

        let mut min_dist = i64::MAX as f32;
        // read line by line and find the closest coordinate
        for line in fgets_lines(&data) {
            let mut tok = line.split(|&c| c == b' ').filter(|t| !t.is_empty());
            let (Some(tx), Some(ty), Some(tw)) = (tok.next(), tok.next(), tok.next()) else {
                continue;
            };
            let (site_x, site_y) = (atof(tx), atof(ty));
            let (diff_x, diff_y) = (center_lon - site_x, center_lat - site_y);
            let dist = (diff_x * diff_x + diff_y * diff_y).sqrt() as f32;
            // only use fairly near estimates (less than 1.5 degrees)
            if dist < min_dist && (dist as f64) < 1.5 {
                min_dist = dist;
                wvp = atof(tw) as f32;
                if k == 0 {
                    avg = wvp;
                }
            }
        }
    }
    // if the daily value has nodata, and strict checking is enabled
    if wvp < 0.0 && src.strict {
        return Err(GasError::NoDailyValue);
    }
    // if the daily value has nodata, use the average value
    if wvp < 0.0 || wvp > 10.0 {
        wvp = avg;
    }
    Ok(wvp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::radtran::RSR_START_SEN2A;

    const NB: usize = 13;
    const B_NIR: usize = 8;
    const B_VAP: usize = 9;

    #[test]
    fn atof_prefix() {
        assert_eq!(atof(b"3.25"), 3.25);
        assert_eq!(atof(b"  -1.5e2xyz"), -150.0);
        assert_eq!(atof(b"abc"), 0.0);
        assert_eq!(atof(b"9999"), 9999.0);
        assert_eq!(atof(b"1e"), 1.0);
        assert_eq!(atof(b".5"), 0.5);
        assert!(atof(b"inf").is_infinite());
    }

    #[test]
    fn fgets_cuts() {
        let l = fgets_lines(b"1 2 3 MOD\r\n4 5 6 # comment\n\n7 8 9");
        assert_eq!(l, vec![&b"1 2 3 MOD"[..], &b"4 5 6 "[..], &b""[..], &b"7 8 9"[..]]);
        // a line longer than the buffer of FORCE (1023 characters) is read in pieces
        let long = vec![b'7'; 2500];
        assert_eq!(fgets_lines(&long).iter().map(|l| l.len()).collect::<Vec<_>>(), vec![1023, 1023, 454]);
    }

    /// The scene of `benchmarks/vs_originals/force_gas` for the reference values below: 24 x 24 pixels of 10 m on a
    /// 2 x 2 grid of 120 m (the cell at the bottom right outside the swath), 13 Sentinel-2A bands; every input a dyadic
    /// rational or an integer, so that it is the same bits as the one the harness got.
    struct Tiny {
        atc: Atc,
        toa: Toa,
        qai: Vec<i16>,
        dem: Vec<u8>,
        rsr: Vec<usize>,
    }

    fn tiny(vap_of: impl Fn(i32, i32, i32) -> i32) -> Tiny {
        let (nx, ny, gn) = (24usize, 24usize, 4usize);
        let mut atc = Atc::zeros(2, 2, 120.0, NB, -9999);
        atc.cosszen = [0.5, 0.6875];
        atc.cosvzen = [0.9375, 0.96875];
        for g in 0..gn {
            atc.xy_sun.bands[C_ZEN][g] = 0.5 + 0.0625 * g as f32;
            atc.xy_view.bands[C_ZEN][g] = if g == 3 { -9999.0 } else { 0.96875 - 0.015625 * g as f32 };
        }
        for b in 0..NB {
            for g in 0..gn {
                atc.xy_tso.bands[b][g] = 0.96875 + 0.00390625 * (b % 4) as f32;
                atc.xy_tvo.bands[b][g] = 0.9765625 + 0.00390625 * (g % 2) as f32;
            }
        }
        for z in 0..crate::atc::N_ELEV {
            for b in 0..NB {
                for g in 0..gn {
                    atc.xyz_rho_p[z].bands[b][g] = (20 + b + 2 * g + z % 5) as f32 / 512.0;
                    atc.xyz_t[z].bands[b][g] = (470 - z as i32 + b as i32 + g as i32) as f32 / 512.0;
                    atc.xyz_s[z].bands[b][g] = (40 + b + g) as f32 / 512.0;
                }
            }
        }
        let npix = nx * ny;
        let mut bands = vec![vec![0i16; npix]; NB];
        let mut qai = vec![0i16; npix];
        let mut dem = vec![0u8; npix];
        for i in 0..ny as i32 {
            for j in 0..nx as i32 {
                let p = (i * nx as i32 + j) as usize;
                dem[p] = (10 + 2 * i + j) as u8;
                let nir = if (i + j) % 11 == 0 { 1200 } else { 2500 + 7 * i + 3 * j };
                for (k, band) in bands.iter_mut().enumerate() {
                    band[p] = (1000 + 37 * k as i32 + i + j) as i16;
                }
                bands[B_NIR][p] = nir as i16;
                bands[B_VAP][p] = vap_of(i, j, nir) as i16;
                let mut q = 0i16;
                if j >= nx as i32 - 3 {
                    q |= 1;
                }
                if (6..12).contains(&i) && (12..18).contains(&j) {
                    q |= 1 << 5;
                }
                if (i * 5 + j * 3) % 17 == 0 {
                    q |= 1 << 3;
                }
                if (18..24).contains(&i) && (6..12).contains(&j) {
                    q |= 1 << 5;
                }
                if (i * 7 + j) % 19 == 0 {
                    q |= 1 << 1; // cloud bits do not matter
                }
                qai[p] = q;
            }
        }
        let mut domain = vec![String::new(); NB];
        domain[B_NIR] = "NIR".into();
        domain[B_VAP] = "VAPOR".into();
        Tiny {
            atc,
            toa: Toa { grid: SceneGrid { nx, ny, res: 10.0 }, bands, domain },
            qai,
            dem,
            rsr: (0..NB).map(|b| RSR_START_SEN2A + b).collect(),
        }
    }

    fn reference_vapor(i: i32, j: i32, nir: i32) -> i32 {
        if i < 6 && (6..12).contains(&j) {
            6000
        } else if (12..18).contains(&i) && j < 6 {
            nir + 900
        } else {
            550 + ((nir - 500) * (45 + (i * j) % 13)) / 100
        }
    }

    /// The values the C harness printed (FORCE's gas-ll.c, GSL 2.7, one thread) for the scene above.
    #[test]
    fn values_of_the_c_harness() {
        let mut t = tiny(reference_vapor);
        let lut = WvLut::build(&t.rsr, &t.atc, 1).unwrap();
        for (b, kw, km, v) in [
            (8, 0, 50, 0x3f800000u32),
            (9, 300, 94, 0x3eb1c6f9),
            (9, 150, 100, 0),
            (3, 700, 96, 0x3f7a8342),
            (9, 0, 49, 0),
            (12, 400, 60, 0x3f6f73d3),
        ] {
            assert_eq!(lut.get(b, kw, km).to_bits(), v, "LUT {b} {kw} {km}");
        }
        assert_eq!(lut.values().iter().filter(|&&v| v != 0.0).count(), 437424);

        let wv = water_vapor(&t.rsr, &t.atc, &lut, &t.toa, &mut t.qai, &t.dem).unwrap();
        assert_eq!(wv.n, 8);
        assert_eq!(wv.w_avg.unwrap().to_bits(), 0x3f298000);
        let nx = 24;
        let blocks: Vec<Vec<i16>> = (0..4).map(|r| (0..4).map(|c| wv.wvp[r * 6 * nx + c * 6]).collect()).collect();
        assert_eq!(
            blocks,
            vec![vec![2125, 662, 531, 515], vec![437, 281, 662, 406], vec![662, 406, 662, 662], vec![593, 662, 662, 662]]
        );
        assert_eq!(wv.wvp.iter().map(|&v| v as i64).sum::<i64>(), 340830);
        assert_eq!(wv.wvp.iter().filter(|&&v| v > 0).count(), 504);
        assert_eq!(t.qai.iter().filter(|&&q| qai::vaporfill(q)).count(), 252);
        assert_eq!(t.qai.iter().map(|&q| q as i64).sum::<i64>(), 4131476);

        let grid = t.toa.grid;
        let sums = [
            3748662i64, 3761820, 3773286, 3766806, 3624120, 3680100, 3747348, 3587940, 3744090, 1470042, 8352, 3777696,
            3639618,
        ];
        for (b, &sum) in sums.iter().enumerate() {
            let tg = gas_transmittance(&t.atc, b, &wv.wvp, &grid, &t.qai, &lut).unwrap();
            assert_eq!(tg.iter().map(|&v| v as i64).sum::<i64>(), sum, "band {b}");
            if b == 8 {
                let blocks: Vec<Vec<i16>> = (0..4).map(|r| (0..4).map(|c| tg[r * 6 * nx + c * 6]).collect()).collect();
                assert_eq!(
                    blocks,
                    vec![
                        vec![9410, 9444, 9486, 9486],
                        vec![9449, 9453, 9483, 9489],
                        vec![9446, 9451, 0, 0],
                        vec![9447, 9446, 0, 0]
                    ]
                );
            }
        }
    }

    #[test]
    fn more_absorption_means_more_vapor() {
        // the vapour channel of one retrievable block, darker and darker (more absorbed): the water vapour grows
        let mut last = 0;
        for k in 0..6 {
            let mut t = tiny(move |i, j, nir| {
                if i < 6 && j < 6 {
                    1400 + 60 * (5 - k)
                } else {
                    reference_vapor(i, j, nir)
                }
            });
            let lut = WvLut::build(&t.rsr, &t.atc, 1).unwrap();
            let wv = water_vapor(&t.rsr, &t.atc, &lut, &t.toa, &mut t.qai, &t.dem).unwrap();
            let v = wv.wvp[0];
            assert!(v > last, "{k}: {v} after {last}");
            last = v;
        }
    }

    #[test]
    fn no_data_no_estimate_no_fill() {
        // every pixel off, or every pixel dark in the reference band, or all water: nothing retrieved, nothing filled
        for case in 0..3 {
            let mut t = tiny(reference_vapor);
            let n = t.qai.len();
            for p in 0..n {
                match case {
                    0 => t.qai[p] |= 1,
                    1 => t.toa.bands[B_NIR][p] = 1499,
                    _ => t.qai[p] |= 1 << 5,
                }
            }
            let before = t.qai.clone();
            let lut = WvLut::build(&t.rsr, &t.atc, 1).unwrap();
            let wv = water_vapor(&t.rsr, &t.atc, &lut, &t.toa, &mut t.qai, &t.dem).unwrap();
            assert_eq!((wv.n, wv.w_avg), (0, None), "case {case}");
            assert!(wv.wvp.iter().all(|&v| v == 0));
            assert_eq!(t.qai, before);
            // the transmittance map of it: water vapour 0 everywhere, zero where the pixel is off
            let tg = gas_transmittance(&t.atc, B_NIR, &wv.wvp, &t.toa.grid, &t.qai, &lut).unwrap();
            assert_eq!(tg.len(), n);
            if case == 0 {
                assert!(tg.iter().all(|&v| v == 0));
            }
        }
    }

    #[test]
    fn a_missing_vapour_band_is_an_error() {
        let mut t = tiny(reference_vapor);
        t.toa.domain[B_VAP].clear();
        let lut = WvLut::build(&t.rsr, &t.atc, 1).unwrap();
        assert_eq!(
            water_vapor(&t.rsr, &t.atc, &lut, &t.toa, &mut t.qai, &t.dem).unwrap_err(),
            GasError::NoDomain("VAPOR")
        );
    }

    #[test]
    fn lut_threads_do_not_matter() {
        let t = tiny(reference_vapor);
        let a = WvLut::build(&t.rsr, &t.atc, 1).unwrap();
        let b = WvLut::build(&t.rsr, &t.atc, 3).unwrap();
        assert!(a.values().iter().zip(b.values()).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert!(a.get(0, 701, 50).is_nan() && a.get(0, 0, -1).is_nan() && a.get(13, 0, 50).is_nan());
    }

    #[test]
    fn vapour_from_the_lut_files() {
        let d = std::env::temp_dir().join(format!("zeit-gas-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        // the climatology of March, and the daily LUT of 2020-03-14: nodata at one site, a value at another
        std::fs::write(d.join("WVP_0000-03-00.txt"), "10.0 50.0 1.5 12\n10.5 50.5 2.5 12\n20.0 60.0 4.0 12\n").unwrap();
        std::fs::write(d.join("WVP_2020-03-14.txt"), "10.0 50.0 9999 TBD\r\n10.9 50.9 3.25 MOD # a comment\n").unwrap();
        let dir = Some(d.to_str().unwrap().to_string());
        let src = |strict| WvpSource { dir: dir.clone(), wvp: 2.0, strict };
        let get = |s: &WvpSource, lon, lat, y, m, dd| water_vapor_from_lut(s, lon, lat, y, m, dd);
        // the daily site nearest to the centre is nodata: the climatology's value (avg) of the same site
        assert_eq!(get(&src(false), 10.0, 50.0, 2020, 3, 14), Ok(1.5));
        // a centre nearest to the second daily site: its value
        assert_eq!(get(&src(false), 10.9, 50.9, 2020, 3, 14), Ok(3.25));
        // another day of March: the climatology's closest
        assert_eq!(get(&src(false), 10.4, 50.4, 2020, 3, 20), Ok(2.5));
        // farther than 1.5 degrees from every site: nothing, -1 (strict: an error as well)
        assert_eq!(get(&src(false), 30.0, 30.0, 2020, 3, 14), Ok(-1.0));
        assert_eq!(get(&src(true), 30.0, 30.0, 2020, 3, 14), Err(GasError::NoDailyValue));
        // a month with no files
        assert_eq!(get(&src(false), 10.0, 50.0, 2020, 4, 14), Ok(-1.0));
        // the constant of the parameter file
        assert_eq!(get(&WvpSource { dir: None, wvp: 2.0, strict: true }, 0.0, 0.0, 2020, 4, 14), Ok(2.0));
        std::fs::remove_dir_all(&d).ok();
    }
}
