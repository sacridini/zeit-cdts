//! The topography of FORCE's atmospheric correction: slope, aspect, illumination angle, sky view factor and the binned
//! elevation, the C-factor of the topographic correction, and the average elevation of a coarse grid cell. A port of
//! FORCE 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring, Copyright (C)
//! 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force),
//! `src/modules/lower-level/topo-ll.c` lines 40-1048: `ocean_topography`, `smooth_topography`,
//! `exposition_topography`, `stats_topography`, `illumination_topography`, `compile_topography`,
//! `cfactor_topography` and `average_elevation_cell`, with the helpers of `cross-level/stats-cl.c` they call
//! (`covar_recurrence`, `variance`, `covariance`, `linreg_coefs`: Welford/Pebay recurrences) and the setters of
//! `quality-cl.c` ([`crate::qai`]). [`crate::radtran`] has `illumin`, `f_factor`, `c_factor_com` and `c_factor_emp`.
//!
//! This is NOT [`crate::topocorr`] (the C and SCS+C corrections of the Zeit API, from the papers, with Horn's kernel
//! on the DEM in metres): the FORCE slope and aspect are stored as angles in radians times 10000 (`ushort`), computed
//! in `float`, from a DEM that FORCE has smoothed first.
//!
//! The arithmetic is the C's, as in [`crate::radtran`]: C `float` is `f32`, rounded where it is assigned; where C
//! promotes to `double` (`atan`, `sqrt`, `cos`, `/ 10000.0`, `/ 1000.0`, `M_PI`, a `double` literal or `res`) the value
//! is computed in `f64`. Where this differs from the C or from FORCE's own behaviour:
//!
//! * **The DEM.** FORCE warps the DEM file onto the grid of the scene (`warp_from_disc_to_known_brick` of
//!   `brick_io-cl.cpp`) before anything else; that is GDAL's `GDALWarpOperation` into a Float32 in-memory dataset with the
//!   resampling of `DEM_RESAMPLING` (`NN`, `BL` (the default) or `CC`, i.e. `GRA_NearestNeighbour`, `GRA_Bilinear`,
//!   `GRA_Cubic`), an accurate transformer (`GDALCreateGenImgProjTransformer`, error threshold 0), the source and the
//!   destination no-data value both `DEM_NODATA` (default -32767), `INIT_DEST` the same, `NUM_THREADS` the threads of
//!   the run. If `FILE_DEM` is `NULL`, the DEM is flat (0 m, with no no-data). That is outside the core: the functions
//!   here take the DEM already on the grid of the scene, in metres, with the no-data value. A public API that reads a
//!   DEM must warp it that way (a bilinear warp of a file with the no-data value set, not a nearest-neighbour
//!   crop) to reproduce FORCE.
//! * **Threads.** FORCE runs the pixel loops with OpenMP. Every loop writes only its own pixel and reads data that
//!   no thread writes in the same loop, so the result does not depend on the threads, but for the elevation
//!   statistics of `stats_topography`: `sum` and `num` are `reduction(+)` sums of doubles, so the last bits of the
//!   average elevation (`atc.dem.avg`) depend on the number of threads (the minimum and the maximum do not). This port
//!   sums in pixel order: FORCE with one thread (`OMP_NUM_THREADS=1`).
//! * **Errors.** Where the C prints a message and returns `FAILURE`/`NULL`, this returns a [`TopoError`]. Grids with
//!   fewer than 3 rows or columns (the 3x3 window of the slope does not fit; the C reads out of the array) and a pixel
//!   size that is not positive are errors too.
//! * **Omitted.** The `FORCE_DEBUG` and `FORCE_CLOCK` blocks (they only print or write files), `cite_me`.
//!
//! The brick machinery of FORCE is plain arrays here (the QAI as `&[i16]`, the TOA as [`crate::gas::Toa`], the coarse
//! grids as [`crate::atc::CoarseBrick`]).

#![allow(clippy::needless_range_loop)] // the loops are the C's, index by index

use crate::atc::{Atc, CoarseBrick, AZI, C_ZEN, N_ELEV, S_ZEN, ZEN};
use crate::gas::{fequal, ji2p, Toa};
use crate::qai;
use crate::radtran::{c_factor_com, c_factor_emp, f_factor, illumin};
use std::f64::consts::PI;
use std::fmt;

/// `_R2D_CONV_` and `_D2R_CONV_` of `const-cl.h`.
const R2D: f64 = 57.29577951308232286465;
const D2R: f64 = 0.01745329251994329547;

/// `_BYTE_LEN_` of `enum-cl.h`.
const BYTE_LEN: i32 = 256;

/// What can go wrong in these functions (where FORCE prints a message and returns `FAILURE`/`NULL`).
#[derive(Debug, Clone, PartialEq)]
pub enum TopoError {
    /// The inputs do not have the sizes they claim, or the grid is too small or has no pixel size.
    Shape(String),
    /// `DEM out of bounds: min %f max %f.` (below -500 m or above 9000 m).
    DemBounds { min: f32, max: f32 },
    /// The TOA brick has no band of this domain (`find_domain`).
    NoDomain(&'static str),
}

impl fmt::Display for TopoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopoError::Shape(s) => write!(f, "{s}"),
            TopoError::DemBounds { min, max } => write!(f, "DEM out of bounds: min {min:.6} max {max:.6}"),
            TopoError::NoDomain(d) => write!(f, "the brick has no band of domain {d}"),
        }
    }
}

impl std::error::Error for TopoError {}

/// The topographic derivatives (`top_t` of `topo-ll.h`), on the grid of the image, one value per pixel.
#[derive(Clone, Debug, Default)]
pub struct Top {
    /// The binned elevation (`dem`, band 0): the elevation of a pixel is `dem * step + min + step / 2` with the `dem`
    /// statistics of the [`Atc`].
    pub dem: Vec<u8>,
    /// The slope (`exp`, band `ZEN`), in radians times 10000 (`slope < 870` is 5 degrees).
    pub slope: Vec<u16>,
    /// The aspect (`exp`, band `AZI`), in radians times 10000, from the north, clockwise.
    pub aspect: Vec<u16>,
    /// The illumination angle (`ill`): the cosine times 10000; -10000 where the pixel is off.
    pub ill: Vec<i16>,
    /// The sky view factor (`sky`), times 10000.
    pub sky: Vec<u16>,
    /// The C-factor of the SWIR2 (`c`), times 10000; [`cfactor_topography`] gives it, `compile_topography` leaves it empty.
    pub c: Vec<u16>,
}

fn shape(nx: usize, ny: usize, res: f64) -> Result<(), TopoError> {
    if nx < 3 || ny < 3 {
        return Err(TopoError::Shape(format!("the grid is {nx} x {ny}: at least 3 x 3 pixels are needed")));
    }
    if !(res > 0.0 && res.is_finite()) {
        return Err(TopoError::Shape(format!("the pixel size is {res}: it must be positive")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// stats-cl.c
// ---------------------------------------------------------------------------------------------------------------

/// `covar_recurrence`: the one-pass update of the means, the (sums of squares of the) variances and the covariance of
/// `x` and `y` with the `n`-th observation (Pebay 2008).
fn covar_recurrence(x: f64, y: f64, mx: &mut f64, my: &mut f64, vx: &mut f64, vy: &mut f64, cv: &mut f64, n: f64) {
    let (oldmx, oldmy) = (*mx, *my);
    let (oldvx, oldvy) = (*vx, *vy);
    let oldcv = *cv;
    *mx = oldmx + (x - oldmx) / n;
    *my = oldmy + (y - oldmy) / n;
    *vx = oldvx + (x - oldmx) * (x - *mx);
    *vy = oldvy + (y - oldmy) * (y - *my);
    *cv = oldcv + (n - 1.0) / n * (x - oldmx) * (y - oldmy);
}

/// `linreg_coefs`: the slope `cov / varx` and the intercept `my - slope * mx` of the regression.
fn linreg_coefs(mx: f64, my: f64, cov: f64, varx: f64) -> (f64, f64) {
    let slope = cov / varx;
    (slope, my - slope * mx)
}

// ---------------------------------------------------------------------------------------------------------------
// the DEM
// ---------------------------------------------------------------------------------------------------------------

/// `ocean_topography`: oceans are often masked out in DEMs (but are valid data). Every no-data pixel that follows
/// a pixel of 0 m, scanning each column down, then up, then each row to the right, then to the left, is set to 0 m
/// (the ocean grows into the no-data of the coast and propagates along the scan).
pub fn ocean_topography(dem: &mut [f32], nx: usize, ny: usize, nodata: f32) {
    for j in 0..nx {
        for i in 0..ny.saturating_sub(1) {
            let (p, np) = (i * nx + j, (i + 1) * nx + j);
            if fequal(dem[p], 0.0) && fequal(dem[np], nodata) {
                dem[np] = 0.0;
            }
        }
    }
    for j in 0..nx {
        for i in (1..ny).rev() {
            let (p, np) = (i * nx + j, (i - 1) * nx + j);
            if fequal(dem[p], 0.0) && fequal(dem[np], nodata) {
                dem[np] = 0.0;
            }
        }
    }
    for i in 0..ny {
        for j in 0..nx.saturating_sub(1) {
            let (p, np) = (i * nx + j, i * nx + j + 1);
            if fequal(dem[p], 0.0) && fequal(dem[np], nodata) {
                dem[np] = 0.0;
            }
        }
    }
    for i in 0..ny {
        for j in (1..nx).rev() {
            let (p, np) = (i * nx + j, i * nx + j - 1);
            if fequal(dem[p], 0.0) && fequal(dem[np], nodata) {
                dem[np] = 0.0;
            }
        }
    }
}

/// `smooth_topography`: the DEM smoothed with a box low-pass filter of `2k + 1` pixels a side, `k = (int)(30 / res)`,
/// without the no-data pixels, as high resolution DEMs have undesired effects; nothing is done if the pixel size is
/// 30 m or more. A pixel that is no-data (exactly) stays so.
pub fn smooth_topography(dem: &mut [f32], nx: usize, ny: usize, res: f64, nodata: f32) {
    if res >= 30.0 {
        return;
    }
    let k = (30.0 / res) as i32;
    let mut buf = vec![0.0f32; nx * ny];
    for i in 0..ny {
        for j in 0..nx {
            let p = i * nx + j;
            buf[p] = nodata;
            if dem[p] == nodata {
                continue;
            }
            let (mut sum, mut num) = (0.0f32, 0.0f32);
            for ii in -k..=k {
                for jj in -k..=k {
                    let (ni, nj) = (i as i64 + ii as i64, j as i64 + jj as i64);
                    if ni > ny as i64 - 1 || ni < 0 || nj > nx as i64 - 1 || nj < 0 {
                        continue;
                    }
                    let np = ni as usize * nx + nj as usize;
                    if fequal(dem[np], nodata) {
                        continue;
                    }
                    sum += dem[np];
                    num += 1.0;
                }
            }
            if num > 0.0 {
                buf[p] = sum / num;
            }
        }
    }
    dem.copy_from_slice(&buf);
}

/// `exposition_topography`: slope and aspect with the method of Horn (1981), in radians times 10000, and the QAI:
/// the slope flag where the slope is above 2 degrees, the off flag where the 3x3 window has no-data. The border
/// pixels (which the window does not fit) take the values of their neighbour inwards, or are off if it is.
/// Returns `(slope, aspect)`.
pub fn exposition_topography(dem: &[f32], nx: usize, ny: usize, res: f64, nodata: f32, qai: &mut [i16]) -> (Vec<u16>, Vec<u16>) {
    let nc = nx * ny;
    let mut slp = vec![0u16; nc];
    let mut asp = vec![0u16; nc];
    // `nodata = (short)get_brick_nodata(DEM, 0)`
    let nodata = nodata as i16 as f32;
    let mut d = [[0.0f32; 3]; 3];
    for i in 1..ny - 1 {
        for j in 1..nx - 1 {
            let p = i * nx + j;
            if qai::off(qai[p]) {
                continue;
            }
            let mut valid = true;
            for ii in 0..3 {
                for jj in 0..3 {
                    let tmp = dem[(i + ii - 1) * nx + (j + jj - 1)];
                    if fequal(tmp, nodata) {
                        valid = false;
                    } else {
                        d[ii][jj] = tmp;
                    }
                }
            }
            if valid {
                // the numerator is a float, the divisor a double
                let devx =
                    (((d[0][2] + 2.0 * d[1][2] + d[2][2]) - (d[0][0] + 2.0 * d[1][0] + d[2][0])) as f64 / (8.0 * res)) as f32;
                let devy =
                    (((d[2][0] + 2.0 * d[2][1] + d[2][2]) - (d[0][0] + 2.0 * d[0][1] + d[0][2])) as f64 / (8.0 * res)) as f32;
                let mut slope = ((((devx * devx + devy * devy) as f64).sqrt()).atan() * R2D) as f32;
                if slope > 2.0 {
                    qai::set_slope(&mut qai[p], true);
                }
                let mut aspect;
                if slope > 0.0 {
                    aspect = ((devy as f64).atan2((-devx) as f64) * R2D) as f32;
                    if aspect < 0.0 {
                        aspect = 90.0 - aspect;
                    } else if aspect > 90.0 {
                        aspect = 360.0 - aspect + 90.0;
                    } else {
                        aspect = 90.0 - aspect;
                    }
                } else {
                    slope = 0.0;
                    aspect = 0.0;
                }
                slp[p] = (slope as f64 * D2R * 10000.0) as u16;
                asp[p] = (aspect as f64 * D2R * 10000.0) as u16;
            } else {
                qai::set_off(&mut qai[p], true);
            }
        }
    }
    // the left and right columns, then the top and bottom rows, one after the other
    for p in (0..nc).step_by(nx) {
        if qai::off(qai[p]) {
            continue;
        }
        if qai::off(qai[p + 1]) {
            qai::set_off(&mut qai[p], true);
        } else {
            slp[p] = slp[p + 1];
            asp[p] = asp[p + 1];
        }
    }
    for p in (nx - 1..nc).step_by(nx) {
        if qai::off(qai[p]) {
            continue;
        }
        if qai::off(qai[p - 1]) {
            qai::set_off(&mut qai[p], true);
        } else {
            slp[p] = slp[p - 1];
            asp[p] = asp[p - 1];
        }
    }
    for p in 0..nx {
        if qai::off(qai[p]) {
            continue;
        }
        if qai::off(qai[p + nx]) {
            qai::set_off(&mut qai[p], true);
        } else {
            slp[p] = slp[p + nx];
            asp[p] = asp[p + nx];
        }
    }
    for p in (ny - 1) * nx..nc {
        if qai::off(qai[p]) {
            continue;
        }
        if qai::off(qai[p - nx]) {
            qai::set_off(&mut qai[p], true);
        } else {
            slp[p] = slp[p - nx];
            asp[p] = asp[p - nx];
        }
    }
    (slp, asp)
}

/// `stats_topography`: the elevation statistics in km (`atc.dem`: minimum, maximum + 1 m, average, 255 bins) over the
/// pixels that are on, and the binned DEM (`step` km classes; 0 where the pixel is off). Fails if the elevation is
/// below -500 m or above 9000 m (`atc.dem` is set by then).
pub fn stats_topography(atc: &mut Atc, dem: &[f32], qai: &[i16]) -> Result<Vec<u8>, TopoError> {
    let nc = dem.len();
    let mut mn = i16::MAX as f32;
    let mut mx = i16::MIN as f32;
    let (mut sum, mut num) = (0.0f64, 0.0f64);
    for p in 0..nc {
        if qai::off(qai[p]) {
            continue;
        }
        let d = (dem[p] as f64 / 1000.0) as f32; // -> kilometer
        if d < mn {
            mn = d;
        }
        if d > mx {
            mx = d;
        }
        sum += d as f64;
        num += 1.0;
    }
    atc.dem.min = mn;
    atc.dem.max = mx;
    atc.dem.avg = (sum / num) as f32;
    atc.dem.max = (atc.dem.max as f64 + 0.001) as f32; // add 1m
    atc.dem.cnum = BYTE_LEN - 1;
    atc.dem.step = (atc.dem.max - atc.dem.min) / atc.dem.cnum as f32;

    let mut cdem = vec![0u8; nc];
    for p in 0..nc {
        if qai::off(qai[p]) {
            continue;
        }
        let d = (dem[p] as f64 / 1000.0) as f32;
        cdem[p] = (((d - atc.dem.min) / atc.dem.step) as f64).floor() as u8;
    }
    if atc.dem.min < -0.5 || atc.dem.max > 9.0 {
        return Err(TopoError::DemBounds { min: atc.dem.min, max: atc.dem.max });
    }
    Ok(cdem)
}

/// `illumination_topography`: the illumination angle (cosine times 10000; -10000 where off), its QAI flag (3: deep
/// shadow, 2: poor, 1: moderate; the flag is left as it is for good illumination) and a simple sky view factor (the
/// portion of the sky dome that diffuses onto a tilted surface, times 10000). The sun angles are those of the coarse
/// cell under the pixel. Returns `(ill, sky)`.
pub fn illumination_topography(
    atc: &Atc,
    nx: usize,
    res: f64,
    slope: &[u16],
    aspect: &[u16],
    qai: &mut [i16],
) -> (Vec<i16>, Vec<u16>) {
    let nc = slope.len();
    let mut ill = vec![0i16; nc];
    let mut sky = vec![0u16; nc];
    let sun = &atc.xy_sun;
    for p in 0..nc {
        if qai::off(qai[p]) {
            ill[p] = -10000;
            continue;
        }
        // convert_brick_p2p
        let i = p / nx;
        let g = ji2p(res, sun.res, sun.nx, i, p - i * nx);
        let slp = (slope[p] as f64 / 10000.0) as f32;
        let asp = (aspect[p] as f64 / 10000.0) as f32;
        // illumination angle
        if slp == 0.0 {
            ill[p] = (sun.bands[C_ZEN][g] * 10000.0) as i16;
        } else {
            ill[p] = (illumin(
                sun.bands[C_ZEN][g],
                sun.bands[S_ZEN][g],
                (slp as f64).cos() as f32,
                (slp as f64).sin() as f32,
                sun.bands[AZI][g],
                asp,
            ) * 10000.0) as i16;
        }
        // set illumination QAI
        if ill[p] < 0 {
            qai::set_illumination(&mut qai[p], 3); // deep shadow
        } else if (ill[p] as f64) < 1736.482 {
            qai::set_illumination(&mut qai[p], 2); // poor
        } else if (ill[p] as f64) < 5735.764 {
            qai::set_illumination(&mut qai[p], 1); // moderate
        }
        // sky view factor (portion of the sky dome diffusing on to a tilted surface)
        sky[p] = (10000.0 - (slp as f64 / PI) * 10000.0) as u16;
    }
    (ill, sky)
}

/// `compile_topography`: the topographic derivatives of the image: the DEM on the grid of the image (`dem`, metres,
/// `None` for the flat DEM of FORCE with no `FILE_DEM`; see the notes above about how FORCE warps it) is cleaned of
/// the oceans, smoothed, and gives slope and aspect, the elevation statistics and the binned elevation
/// (`atc.dem`), the illumination angle and the sky view factor. `qai` is the QAI of the image (`nx * ny`), which this
/// updates (off, slope and illumination flags); `atc` needs its `xy_sun` brick.
pub fn compile_topography(
    dem: Option<&[f32]>,
    dem_nodata: i32,
    nx: usize,
    ny: usize,
    res: f64,
    atc: &mut Atc,
    qai: &mut [i16],
) -> Result<Top, TopoError> {
    shape(nx, ny, res)?;
    let nc = nx * ny;
    if qai.len() != nc || dem.is_some_and(|d| d.len() != nc) {
        return Err(TopoError::Shape("the DEM and the QAI must have nx * ny pixels".into()));
    }
    let nodata = dem_nodata as f32;
    // flat DEM (z = 0 m) if there is no file
    let mut dem: Vec<f32> = dem.map_or_else(|| vec![0.0; nc], |d| d.to_vec());
    ocean_topography(&mut dem, nx, ny, nodata);
    smooth_topography(&mut dem, nx, ny, res, nodata);
    let (slope, aspect) = exposition_topography(&dem, nx, ny, res, nodata, qai);
    let bem = stats_topography(atc, &dem, qai)?;
    let (ill, sky) = illumination_topography(atc, nx, res, &slope, &aspect, qai);
    Ok(Top { dem: bem, slope, aspect, ill, sky, c: Vec::new() })
}

/// `cfactor_topography`: the C-factor of the topographic correction (Gao & Zhang 2009; Zhang et al. 2015's
/// stratification by land cover is the SWIR index), derived on the SWIR2 band and propagated through the spectrum in the
/// radiometric correction, for every pixel with a sufficient illumination (the others are 0). It is the theoretical C
/// (from the sky view of the sun zenith and the f-factor of the atmosphere) where the pixel is flat or has too few
/// similar sloped neighbours; else the empirical C (the intercept over the slope of the regression of the SWIR2
/// reflectance on the illumination over a sparse 3 km neighbourhood of sloped pixels (> 2 degrees) of a similar
/// SWIR index), never below the theoretical one and at most 6.5535, finally smoothed with a 3x3 mean. The result is
/// times 10000 (`ushort`). `toa` needs the domains `SWIR1` and `SWIR2`, whose band index selects the band of the
/// `xyz_*` bricks of the `atc`; `dem` is the binned elevation, `slope` and `ill` as [`Top`] has them.
pub fn cfactor_topography(atc: &Atc, toa: &Toa, qai: &[i16], dem: &[u8], slope: &[u16], ill: &[i16]) -> Result<Vec<u16>, TopoError> {
    let (nx, ny, res) = (toa.grid.nx, toa.grid.ny, toa.grid.res);
    let nc = nx * ny;
    if !(res > 0.0 && res.is_finite()) {
        return Err(TopoError::Shape(format!("the pixel size is {res}: it must be positive")));
    }
    if qai.len() != nc || dem.len() != nc || slope.len() != nc || ill.len() != nc {
        return Err(TopoError::Shape("the QAI, the DEM, the slope and the illumination must have nx * ny pixels".into()));
    }
    let b_sw1 = toa.find_domain("SWIR1").ok_or(TopoError::NoDomain("SWIR1"))?;
    let b_sw2 = toa.find_domain("SWIR2").ok_or(TopoError::NoDomain("SWIR2"))?;
    let (sw1, sw2) = (&toa.bands[b_sw1], &toa.bands[b_sw2]);
    if sw1.len() != nc || sw2.len() != nc {
        return Err(TopoError::Shape("the TOA bands must have nx * ny pixels".into()));
    }
    if atc.xyz_rho_p.len() < N_ELEV || atc.xyz_tss.len() < N_ELEV || atc.xyz_tsd.len() < N_ELEV {
        return Err(TopoError::Shape("the atc needs one brick per elevation bin".into()));
    }
    let s_min: u16 = 350; // 2° slope
    let sun = &atc.xy_sun;
    let (xy_szen, xy_ms) = (&sun.bands[ZEN], &sun.bands[C_ZEN]);

    // kernel for sampling neighborhood: 3 km, shrinking by sqrt(2) to 100 m, 0, and the same on the other side.
    // (`K[nk] = 3000/res`, `K[nk] = K[nk-1]/sqrt(2)` are int conversions; after the loop `nk++` takes in the zero
    // that `alloc` left after the last value, if that is not zero.)
    let mut kk = vec![0i32; (3000.0 / res * 2.0 + 1.0) as usize + 64];
    let mut nk = 0usize;
    kk[nk] = (3000.0 / res) as i32;
    nk += 1;
    while kk[nk - 1] as f64 > 100.0 / res {
        kk[nk] = (kk[nk - 1] as f64 / 2.0f64.sqrt()) as i32;
        nk += 1;
    }
    if kk[nk - 1] != 0 {
        nk += 1;
    }
    for k in (0..nk.saturating_sub(1)).rev() {
        kk[nk] = -kk[k];
        nk += 1;
    }

    // compute SWIR index
    let mut swir = vec![0.0f32; nc];
    for p in 0..nc {
        if qai::off(qai[p]) {
            continue;
        }
        let tmp = (sw1[p] as f64 / 10000.0 + sw2[p] as f64 / 10000.0) as f32;
        swir[p] = if tmp == 0.0 { 0.0 } else { ((sw1[p] as f64 / 10000.0 - sw2[p] as f64 / 10000.0) / tmp as f64) as f32 };
    }

    // estimate C for every pixel
    let mut cor = vec![0.0f32; nc];
    for i in 0..ny {
        for j in 0..nx {
            let p = i * nx + j;
            // only do for illuminated pixels
            if qai::off(qai[p]) || ill[p] < 0 {
                continue;
            }
            let g = ji2p(res, sun.res, sun.nx, i, j);
            let z = dem[p] as usize;

            // f-factor, h0-factor, theoretical C-factor
            let rho_p = atc.xyz_rho_p[z].bands[b_sw2][g];
            let tss = atc.xyz_tss[z].bands[b_sw2][g];
            let tsd = atc.xyz_tsd[z].bands[b_sw2][g];
            let szen = xy_szen[g];
            let ms = xy_ms[g];
            let f = f_factor(tss, tsd);
            let h0 = ((PI + (2.0 * szen) as f64) / (2.0 * PI)) as f32;
            let c_ = c_factor_com(h0, f, ms);

            // only do for sloped pixels > 2°
            if slope[p] > s_min {
                let (mut num, mut mx, mut my, mut varx, mut vary, mut cov) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                // sample neighborhood
                for ii in 0..nk {
                    for jj in 0..nk {
                        let (ip, jp) = (i as i64 + kk[ii] as i64, j as i64 + kk[jj] as i64);
                        if ip < 0 || jp < 0 || ip > ny as i64 - 1 || jp > nx as i64 - 1 {
                            continue;
                        }
                        let np = ip as usize * nx + jp as usize;
                        // only use illuminated and sloped pixels > 2°
                        if qai::off(qai[np]) || ill[np] < 0 || slope[np] < s_min {
                            continue;
                        }
                        // only do for same land cover
                        if ((swir[p] - swir[np]) as f64).abs() > 0.025 {
                            continue;
                        }
                        num += 1.0;
                        // linear regression
                        if fequal(num as f32, 1.0) {
                            mx = ill[np] as f64 / 10000.0;
                            my = sw2[np] as f64 / 10000.0;
                        } else {
                            covar_recurrence(
                                ill[np] as f64 / 10000.0,
                                sw2[np] as f64 / 10000.0,
                                &mut mx,
                                &mut my,
                                &mut varx,
                                &mut vary,
                                &mut cov,
                                num,
                            );
                        }
                    }
                }
                if num > 2.0 {
                    // regression parameters + c-factor
                    let (mut gain, mut offset) = linreg_coefs(mx, my, cov / (num - 1.0), varx / (num - 1.0));
                    // if offset < path reflectance, use path reflectance as offset and recompute gain
                    if offset < rho_p as f64 {
                        offset = rho_p as f64;
                        gain = (my - offset) / mx;
                    }
                    // shield against extreme C-values
                    cor[p] = if offset < 10.0 * gain { c_factor_emp(offset, gain) } else { 10.0 };
                }
            }

            // use computed C if estimated C < 0
            if cor[p] < 0.0 || cor[p] < c_ {
                cor[p] = c_;
            }
            if cor[p] as f64 > u16::MAX as f64 / 10000.0 {
                cor[p] = (u16::MAX as f64 / 10000.0) as f32;
            }
        }
    }

    // smooth C-factor with lowpass
    let mut cf = vec![0u16; nc];
    for i in 0..ny {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) || ill[p] < 0 {
                continue;
            }
            let (mut num, mut mx) = (0.0f64, 0.0f64);
            for ip in (i as i64 - 1)..=(i as i64 + 1) {
                for jp in (j as i64 - 1)..=(j as i64 + 1) {
                    if ip < 0 || jp < 0 || ip > ny as i64 - 1 || jp > nx as i64 - 1 {
                        continue;
                    }
                    let np = ip as usize * nx + jp as usize;
                    if qai::off(qai[np]) || ill[np] < 0 {
                        continue;
                    }
                    mx += cor[np] as f64;
                    num += 1.0;
                }
            }
            let mut c = if num > 0.0 { (mx / num * 10000.0) as f32 } else { cor[p] * 10000.0 };
            if c > u16::MAX as f32 {
                c = u16::MAX as f32;
            }
            if num > 0.0 {
                cf[p] = c as u16;
            }
        }
    }
    Ok(cf)
}

/// `average_elevation_cell`: the average binned elevation of the fine pixels (`fdem`, `fnx` x `fny` pixels of `fres`
/// metres, the binned DEM) that are on in the cell `g` of the coarse grid `cdem` (the elevation bin of the cell is
/// stored in its band 0, as an `int`: the bricks of the C hold it as one); 0 if none is.
pub fn average_elevation_cell(g: usize, cdem: &mut CoarseBrick, fdem: &[u8], fnx: usize, fny: usize, fres: f64, qai: &[i16]) {
    cdem.bands[0][g] = average_elevation_value(g, cdem.nx, cdem.res, fdem, fnx, fny, fres, qai);
}

/// The value `average_elevation_cell` stores for the cell `g` of a coarse grid of `cnx` cells a row and `cres` metres:
/// the C's `set_brick` on the `int` brick (`(int)round(average)`) as a float. It reads the DEM and the QAI only, so the
/// cells can be computed in any order, or in parallel.
#[allow(clippy::too_many_arguments)]
pub fn average_elevation_value(g: usize, cnx: usize, cres: f64, fdem: &[u8], fnx: usize, fny: usize, fres: f64, qai: &[i16]) -> f32 {
    // convert_brick_p2ji(CDEM, FDEM, g, &i, &j): coarse cell to fine position
    let i_from = g / cnx;
    let i = (i_from as f64 * cres / fres).floor() as i64;
    let j = ((g - i_from * cnx) as f64 * cres / fres).floor() as i64;
    // cellsize in fine pixels
    let cell_size = (cres / fres).floor() as i64;
    let (mut sum, mut num) = (0.0f64, 0.0f64);
    for ii in i..(i + cell_size) {
        for jj in j..(j + cell_size) {
            if ii > fny as i64 - 1 || jj > fnx as i64 - 1 {
                continue;
            }
            let p = ii as usize * fnx + jj as usize;
            if qai::off(qai[p]) {
                continue;
            }
            sum += fdem[p] as f64;
            num += 1.0;
        }
    }
    // set_brick on an int brick: (int)val
    if num > 0.0 {
        ((sum / num).round() as f32) as i32 as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gas::SceneGrid;

    /// A scene of `n x n` pixels of `res` metres on a coarse grid of `cs` pixels a cell, with the sun at `szen` degrees
    /// zenith, `sazi` degrees azimuth, and a DEM (a cone).
    fn scene(n: usize, res: f64, cs: usize, szen: f64, sazi: f64) -> (Atc, Vec<f32>) {
        let gn = n.div_ceil(cs);
        let mut atc = Atc::zeros(gn, gn, res * cs as f64, 13, -9999);
        for g in 0..gn * gn {
            let z = (szen + (g % 3) as f64).to_radians();
            atc.xy_sun.bands[ZEN][g] = z as f32;
            atc.xy_sun.bands[AZI][g] = sazi.to_radians() as f32;
            atc.xy_sun.bands[C_ZEN][g] = z.cos() as f32;
            atc.xy_sun.bands[S_ZEN][g] = z.sin() as f32;
        }
        let c = n as f32 / 2.0;
        let dem = (0..n * n)
            .map(|p| {
                let (i, j) = ((p / n) as f32, (p % n) as f32);
                (1500.0 - 12.0 * ((i - c).powi(2) + (j - c).powi(2)).sqrt()).max(0.0)
            })
            .collect();
        (atc, dem)
    }

    #[test]
    fn flat_dem_gives_zero_slope_and_sun_illumination() {
        let n = 20;
        let (mut atc, _) = scene(n, 30.0, 5, 30.0, 150.0);
        let mut qai = vec![0i16; n * n];
        let top = compile_topography(None, -32767, n, n, 30.0, &mut atc, &mut qai).unwrap();
        assert!(top.slope.iter().all(|&s| s == 0) && top.aspect.iter().all(|&s| s == 0));
        assert!(top.sky.iter().all(|&s| s == 10000));
        assert!(qai.iter().all(|&q| q == 0));
        for p in 0..n * n {
            let g = ji2p(30.0, atc.xy_sun.res, atc.xy_sun.nx, p / n, p % n);
            assert_eq!(top.ill[p], (atc.xy_sun.bands[C_ZEN][g] * 10000.0) as i16);
        }
        // the elevation statistics of a flat DEM at 0 m: min 0, max 1 m
        assert_eq!((atc.dem.min, atc.dem.cnum), (0.0, 255));
        assert_eq!(atc.dem.max, 0.001);
        assert!(top.dem.iter().all(|&d| d == 0));
    }

    #[test]
    fn ramp_has_the_slope_and_aspect_of_horn() {
        // a plane rising 1 m per 2 m to the east: slope atan(0.5), facing west (aspect 270 degrees)
        let n = 12;
        let (mut atc, _) = scene(n, 10.0, 4, 40.0, 100.0);
        let dem: Vec<f32> = (0..n * n).map(|p| 100.0 + (p % n) as f32 * 5.0).collect();
        let mut qai = vec![0i16; n * n];
        let top = compile_topography(Some(&dem), -32767, n, n, 10.0, &mut atc, &mut qai).unwrap();
        // res 10 m < 30 m: the smoothing (k = 3) moves the borders of the plane only
        let p = 6 * n + 6;
        let slope = top.slope[p] as f64 / 10000.0;
        assert!((slope - 0.5f64.atan()).abs() < 1e-4, "{slope}");
        let aspect = top.aspect[p] as f64 / 10000.0;
        assert!((aspect - 270f64.to_radians()).abs() < 1e-3, "{aspect}");
        assert!((qai[p] >> qai::BIT_SLP) & 1 == 1 && qai.iter().all(|&q| !qai::off(q)));
    }

    #[test]
    fn ocean_takes_the_nodata_next_to_it_and_the_borders_copy_inwards() {
        let (nx, ny) = (6usize, 5usize);
        let nd = -32767.0f32;
        let mut dem = vec![10.0f32; nx * ny];
        dem[0] = 0.0;
        dem[1] = nd;
        dem[nx] = nd;
        dem[2 * nx + 2] = nd;
        ocean_topography(&mut dem, nx, ny, nd);
        assert_eq!((dem[1], dem[nx], dem[2 * nx + 2]), (0.0, 0.0, nd));
        let mut qai = vec![0i16; nx * ny];
        let mut dem2 = vec![10.0f32; nx * ny];
        dem2[2 * nx + 2] = nd;
        let (s, _) = exposition_topography(&dem2, nx, ny, 30.0, nd, &mut qai);
        // the pixels whose 3x3 window holds the no-data pixel are off, and so are the border pixels next to them
        assert!(qai::off(qai[2 * nx + 2]) && qai::off(qai[nx + 1]) && !qai::off(qai[nx + 4]));
        assert_eq!(s[nx + 4], 0);
    }

    #[test]
    fn the_bounds_are_checked() {
        let n = 8;
        let (mut atc, _) = scene(n, 30.0, 4, 30.0, 150.0);
        let mut qai = vec![0i16; n * n];
        let mut dem = vec![100.0f32; n * n];
        dem[10] = 9500.0;
        let e = compile_topography(Some(&dem), -32767, n, n, 30.0, &mut atc, &mut qai).unwrap_err();
        assert!(matches!(e, TopoError::DemBounds { .. }));
        assert!(compile_topography(None, -32767, 2, n, 30.0, &mut atc, &mut qai[..16].to_vec()).is_err());
    }

    #[test]
    fn the_cone_is_lit_on_one_side_and_the_c_factor_is_bounded() {
        let n = 40;
        let (mut atc, dem) = scene(n, 30.0, 8, 60.0, 130.0);
        let mut qai = vec![0i16; n * n];
        let mut top = compile_topography(Some(&dem), -32767, n, n, 30.0, &mut atc, &mut qai).unwrap();
        assert!(top.ill.iter().any(|&v| v > 7000) && top.ill.iter().any(|&v| v < 5000));
        // C-factor: path reflectance and transmittances for every bin
        for z in 0..N_ELEV {
            for b in 0..13 {
                for g in 0..atc.xy_sun.ncell() {
                    atc.xyz_rho_p[z].bands[b][g] = 0.01 + 0.0001 * z as f32;
                    atc.xyz_tss[z].bands[b][g] = 0.8 - 0.0005 * z as f32;
                    atc.xyz_tsd[z].bands[b][g] = 0.9;
                }
            }
        }
        let mut domain: Vec<String> = (0..13).map(|b| format!("B{b}")).collect();
        domain[11] = "SWIR1".into();
        domain[12] = "SWIR2".into();
        let mut bands = vec![vec![0i16; n * n]; 13];
        for p in 0..n * n {
            bands[11][p] = (1500 + (top.ill[p] as i32 / 8)) as i16;
            bands[12][p] = (600 + (top.ill[p] as i32 / 5) + (p % 7) as i32) as i16;
        }
        let toa = Toa { grid: SceneGrid { nx: n, ny: n, res: 30.0 }, bands, domain };
        let c = cfactor_topography(&atc, &toa, &qai, &top.dem, &top.slope, &top.ill).unwrap();
        for p in 0..n * n {
            if top.ill[p] < 0 {
                assert_eq!(c[p], 0);
            }
        }
        assert!(c.iter().any(|&v| v > 0));
        top.c = c;
        let missing = Toa { domain: vec![String::new(); 13], ..toa };
        assert_eq!(cfactor_topography(&atc, &missing, &qai, &top.dem, &top.slope, &top.ill), Err(TopoError::NoDomain("SWIR1")));
    }

    #[test]
    fn average_elevation_of_a_coarse_cell() {
        let (fnx, fny) = (10usize, 10usize);
        let fdem: Vec<u8> = (0..fnx * fny).map(|p| (p % 7) as u8).collect();
        let mut qai = vec![0i16; fnx * fny];
        qai[3] = 1; // off
        let mut c = CoarseBrick::new(3, 3, 40.0, 1, -9999);
        // cell 1 is rows 0-3, columns 4-7 of the 10 m grid; cell 2 reaches over the right border; cell 8 is mostly off
        average_elevation_cell(1, &mut c, &fdem, fnx, fny, 10.0, &qai);
        let mut sum = 0u32;
        for i in 0..4 {
            for j in 4..8 {
                sum += fdem[i * fnx + j] as u32;
            }
        }
        assert_eq!(c.bands[0][1], (sum as f64 / 16.0).round() as f32);
        average_elevation_cell(2, &mut c, &fdem, fnx, fny, 10.0, &qai);
        assert!(c.bands[0][2] > 0.0);
        average_elevation_cell(8, &mut c, &fdem, fnx, fny, 10.0, &qai); // rows 8-11, columns 8-11: partly outside
        assert!(c.bands[0][8] >= 0.0);
        let all_off = vec![1i16; fnx * fny];
        average_elevation_cell(0, &mut c, &fdem, fnx, fny, 10.0, &all_off);
        assert_eq!(c.bands[0][0], 0.0);
    }
}
