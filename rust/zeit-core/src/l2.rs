//! The surface reflectance and the Level 2 products of FORCE's atmospheric correction, and the order in which the stages
//! run. A port of FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force), `src/modules/lower-level/atmo-ll.c` from `surface_reflectance` on:
//! `surface_reflectance` (the BOA reflectance of one band, with the topographic, BRDF and environmental terms),
//! `background_reflectance` (the three rings of exponential moving averages of the adjacency correction),
//! `compile_l2_boa`, `compile_l2_qai`, `compile_l2_dst` (with `cloud_distance` of `cloud-ll.c`), `compile_l2_vzn`,
//! `compile_l2_hot`, `compile_l2_aod`, `compile_l2_wvp`, `compile_level2` and `radiometric_correction`. The pieces
//! before them are in [`crate::atmo`] (the atmosphere of every coarse cell, the interpolation of the coarse grid), the
//! stages that `radiometric_correction` calls in [`crate::aod`], [`crate::gas`] and [`crate::topo`].
//!
//! The order of `radiometric_correction` (and with it of this port): with `doatmo`, the molecular optical depth of the
//! scene (`atc.mod_` from the wavelengths, `atc.hr` from the average elevation), the water vapour (Landsat: from the
//! LUT or the constant of the parameter file; Sentinel-2: 0 until `water_vapor` has run), `atmo_angledep`,
//! `compile_aod`, `atmo_elevdep`, `water_vapor` (Sentinel-2 only: it builds the transmittance LUT, fills the QAI's
//! vapour flag), `cfactor_topography` (with `dotopo`); then `apply_aoi` and `compile_level2`: the HOT and the AOD
//! product (they read the TOA and the QAI as they are), the BOA (band by band, in the order of the domains BLUE, GREEN,
//! RED, REDEDGE1-3, BROADNIR, NIR, SWIR1, SWIR2: the gas transmittance (Sentinel-2), the background reflectance
//! (`doenv`), the surface reflectance; the QAI gets the subzero, saturation and off flags as it goes, so a pixel that
//! the green band sets off is off for the bands after it), the WVP, the QAI (the off pixels become 1) and with it the
//! DST and the VZN. FORCE has run `allocate_atc`, `sun_target_view`, the conversion to TOA and `compile_topography`
//! before (and the cloud detection, whose flags the QAI brings here).
//!
//! The arithmetic is the C's, as in [`crate::radtran`]: C `float` is `f32`, rounded where it is assigned; where C
//! promotes to `double` (`/ 10000.0`, `cos`, `M_PI`, a `double` literal, `res`) the value is computed in `f64`. The
//! products are `short`: a `double` or `float` goes to `short` as x86-64 does it (truncated to a 32 bit integer, whose
//! low 16 bits are kept: a value out of range or NaN gives what the C gives, see [`c_short`]). Where this differs from
//! the C or from FORCE's own behaviour:
//!
//! * **The datacube.** FORCE's products are warped to the datacube (`cube_t`: the grid of the tile system, the
//!   resampling of `RESAMPLING`, nearest neighbour for the QAI; `warp_from_brick_to_unknown_brick`), and the VZN warps
//!   its coarse grid of view zenith angles to the cube (the cube's resolution set to the coarse grid's for the call)
//!   before interpolating it with the pixels of the QAI that has been warped already. None of that is here: the products
//!   stay on the grid of the scene (`doreproj` off), and the VZN is interpolated on that grid. What the public API
//!   decides about the output grid is not decided here.
//! * **The overview.** `compile_l2_ovv` (the quicklook JPEG) is not ported.
//! * **The metadata.** What `compile_l2_*` set on the bricks (names, file names, the format of the output file, the
//!   product name) is not carried; the scale of each product is [`SCALE_BOA`], [`SCALE_VZN`], [`SCALE_AOD`] and
//!   [`SCALE_WVP`], the no-data value [`NODATA`] (the QAI's is 1). The BOA's domain and wavelength per band are in
//!   [`Boa`].
//! * **The BRDF.** `brdf_factor` of `brdf-ll.c` is not ported: the factor is the [`BrdfFn`] that `atmo_angledep`
//!   takes, and with `dobrdf` the surface reflectance multiplies by the interpolated `xy_brdf`.
//! * **`tss_sw2`.** The surface reflectance interpolates `tss_sw2` and `tsd_sw2` from `xyz_tss[z]` and `xyz_tsd[z]` of
//!   the band `b` (the arrays it prepares for the SWIR2 are not used), so `f0 = f`; this port does the same.
//! * **Threads.** FORCE runs the pixel loops with OpenMP; every loop writes only its own pixel (or row) and reads
//!   what no thread writes in the same loop, so the result does not depend on the number of threads, and the loops here
//!   are rayon loops over rows (`n_jobs`). The vertical pass of the background reflectance runs over the rows (the
//!   columns are independent: the same arithmetic per column, in the same order) on one thread.
//! * **Errors.** Where the C prints a message and returns `FAILURE`/`NULL`, this returns an [`L2Error`].

use crate::aod::{compile_aod, dist_transform, AodError, AodParams};
use crate::atc::{Atc, AtmoError, FORCE_NO_DATA, ZEN};
use crate::atmo::{apply_aoi, atmo_angledep, atmo_elevdep, interpolate_coarse, interpolation_weights, BrdfFn, L2Params};
use crate::gas::{self, ji2p, GasError, SceneGrid, Toa, WvLut, WvpSource};
use crate::meta::Meta;
use crate::parallel::{pool, threads};
use crate::qai;
use crate::radtran::{env_weight, env_weight_aerosol, env_weight_molecular, mod_elev_factor, mod_elev_scale, molecular_optical_depth};
use crate::sunview::{Mission, SceneGeo};
use crate::topo::{cfactor_topography, Top, TopoError};
use rayon::prelude::*;
use std::fmt;

/// `_FORCE_NO_DATA_` as the `short` of the products.
pub const NODATA: i16 = FORCE_NO_DATA as i16;
/// The scale of the BOA reflectance (`short`, reflectance times 10000).
pub const SCALE_BOA: f64 = 10000.0;
/// The scale of the VZN product (the view zenith in degrees times 100).
pub const SCALE_VZN: f64 = 100.0;
/// The scale of the AOD product (the AOD of the green band at 550 nm times 1000).
pub const SCALE_AOD: f64 = 1000.0;
/// The scale of the WVP product (the water vapour in cm times 1000).
pub const SCALE_WVP: f64 = 1000.0;

/// `_R2D_CONV_` of `const-cl.h`.
const R2D: f64 = 57.29577951308232286465;

/// What `radiometric_correction` can report instead of printing a message and returning `NULL`.
#[derive(Clone, Debug, PartialEq)]
pub enum L2Error {
    Atmo(AtmoError),
    Aod(AodError),
    Gas(GasError),
    Topo(TopoError),
    /// Anything else: the sizes of the inputs, a failed lookup of the water vapour.
    Other(String),
}

impl fmt::Display for L2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            L2Error::Atmo(e) => write!(f, "{e}"),
            L2Error::Aod(e) => write!(f, "{e}"),
            L2Error::Gas(e) => write!(f, "{e}"),
            L2Error::Topo(e) => write!(f, "{e}"),
            L2Error::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for L2Error {}

impl From<AtmoError> for L2Error {
    fn from(e: AtmoError) -> Self {
        L2Error::Atmo(e)
    }
}
impl From<AodError> for L2Error {
    fn from(e: AodError) -> Self {
        L2Error::Aod(e)
    }
}
impl From<GasError> for L2Error {
    fn from(e: GasError) -> Self {
        L2Error::Gas(e)
    }
}
impl From<TopoError> for L2Error {
    fn from(e: TopoError) -> Self {
        L2Error::Topo(e)
    }
}

/// A `double` to a `short` as gcc compiles `(short)x` on x86-64: `cvttsd2si` (truncated to a 32 bit integer, the
/// "integer indefinite" `INT_MIN` for NaN and for a value that does not fit), then the low 16 bits. For a value in range
/// that is the truncation; this keeps the C's result out of range too (where C says it is undefined).
#[inline]
pub fn c_short(x: f64) -> i16 {
    let i = if x.is_nan() || x >= 2147483648.0 || x < -2147483648.0 { i32::MIN } else { x as i32 };
    i as i16
}

/// Runs `f` on the pool of `n_jobs` threads (or here if that is one).
fn run<R: Send>(n_jobs: i32, f: impl FnOnce(bool) -> R + Send) -> R {
    let t = threads(n_jobs);
    if t <= 1 {
        f(false)
    } else {
        pool(t).install(|| f(true))
    }
}

/// `f(row, a_row, b_row)` for every row of `nx` values of `a` and `b`.
fn rows2<A: Send, B: Send>(n_jobs: i32, nx: usize, a: &mut [A], b: &mut [B], f: impl Fn(usize, &mut [A], &mut [B]) + Sync + Send) {
    run(n_jobs, |par| {
        if par {
            a.par_chunks_mut(nx).zip(b.par_chunks_mut(nx)).enumerate().for_each(|(i, (x, y))| f(i, x, y));
        } else {
            a.chunks_mut(nx).zip(b.chunks_mut(nx)).enumerate().for_each(|(i, (x, y))| f(i, x, y));
        }
    })
}

/// `f(row, a_row)` for every row of `nx` values of `a`.
fn rows1<A: Send>(n_jobs: i32, nx: usize, a: &mut [A], f: impl Fn(usize, &mut [A]) + Sync + Send) {
    run(n_jobs, |par| {
        if par {
            a.par_chunks_mut(nx).enumerate().for_each(|(i, x)| f(i, x));
        } else {
            a.chunks_mut(nx).enumerate().for_each(|(i, x)| f(i, x));
        }
    })
}

fn shape(grid: &SceneGrid, lens: &[(&str, usize)]) -> Result<(), AtmoError> {
    let nc = grid.nx * grid.ny;
    for (name, n) in lens {
        if *n != nc {
            return Err(AtmoError::Shape(format!("{name} has {n} values for a grid of {nc} pixels")));
        }
    }
    Ok(())
}

/// What `surface_reflectance` reads besides the atmosphere: the TOA of the band (reflectance times 10000), the
/// background reflectance (`bck_`, with `doenv`), the gaseous transmittance (`Tg_`, times 10000; `None` interpolates the
/// coarse `xy_Tg`) and, with `dotopo`, the illumination (`ill_`), sky view (`sky_`) and C-factor (`cf_`) of the topography;
/// `dem` is the binned elevation (`dem_`).
pub struct SurfaceInput<'a> {
    pub toa: &'a [i16],
    pub bck: Option<&'a [i16]>,
    pub tg: Option<&'a [i16]>,
    pub dem: &'a [u8],
    pub ill: &'a [i16],
    pub sky: &'a [u16],
    pub cf: &'a [u16],
}

/// `surface_reflectance`: the BOA reflectance (times 10000, `short`) of the band `b` of the image `grid`. With
/// `par.doatmo` the atmosphere of the coarse grid (`atc` after `atmo_angledep` and `atmo_elevdep`) is interpolated to
/// each pixel; with `par.doenv` the target reflectance corrects the adjacency effect with `inp.bck`, else the
/// homogeneous target is assumed; with `par.dotopo` the illumination factor `A` is applied; with `par.dobrdf` the
/// interpolated BRDF factor. Without `doatmo` the result is the TOA reflectance. Pixels that are off give the no-data
/// value; a reflectance below 0 sets the subzero flag of `qai`, one above 1 the saturation flag, one below -1 the off
/// flag (and the no-data value); with `par.erase_cloud` the pixels with the cloud flag 2 (confident, opaque) give the no-data value.
pub fn surface_reflectance(
    par: &L2Params,
    atc: &Atc,
    b: usize,
    inp: &SurfaceInput,
    grid: &SceneGrid,
    qai: &mut [i16],
    n_jobs: i32,
) -> Result<Vec<i16>, AtmoError> {
    let (nx, ny, fres) = (grid.nx, grid.ny, grid.res);
    let nc = nx * ny;
    shape(grid, &[("the QAI", qai.len()), ("the TOA", inp.toa.len()), ("the DEM", inp.dem.len())])?;
    if par.doenv && par.doatmo {
        match inp.bck {
            Some(bck) => shape(grid, &[("the background reflectance", bck.len())])?,
            None => return Err(AtmoError::Shape("doenv needs the background reflectance".into())),
        }
    }
    if let Some(tg) = inp.tg {
        shape(grid, &[("the gaseous transmittance", tg.len())])?;
    }
    if par.dotopo && par.doatmo {
        shape(grid, &[("the illumination", inp.ill.len()), ("the sky view factor", inp.sky.len()), ("the C-factor", inp.cf.len())])?;
    }
    let (nf, ne) = (atc.xy_view.nx as i32, atc.xy_view.ny as i32);
    let gres = atc.xy_view.res;
    let vnodata = atc.xy_view.nodata as f32;
    atc.xy_mod.find_domain("SWIR2").ok_or(AtmoError::Domain("SWIR2"))?;
    if atc.xyz_t.len() < crate::atc::N_ELEV || atc.xy_view.ncell() != (nf * ne) as usize {
        return Err(AtmoError::Shape("the atc needs one brick per elevation bin".into()));
    }
    fn band(v: &[crate::atc::CoarseBrick], b: usize) -> Vec<&[f32]> {
        v.iter().map(|c| c.bands[b].as_slice()).collect()
    }
    let (xyz_t, xyz_ts, xyz_tss, xyz_tsd) = (band(&atc.xyz_t, b), band(&atc.xyz_ts, b), band(&atc.xyz_tss, b), band(&atc.xyz_tsd, b));
    let (xyz_tvs, xyz_tvd, xyz_s, xyz_rho_p) =
        (band(&atc.xyz_tvs, b), band(&atc.xyz_tvd, b), band(&atc.xyz_s, b), band(&atc.xyz_rho_p, b));
    let xy_vz = atc.xy_view.bands[ZEN].as_slice();
    let xy_sz = atc.xy_sun.bands[ZEN].as_slice();
    let (xy_tg, xy_tvo, xy_tso, xy_brdf) = (atc.xy_tg.bands[b].as_slice(), atc.xy_tvo.bands[b].as_slice(), atc.xy_tso.bands[b].as_slice(), atc.xy_brdf.bands[b].as_slice());
    let e0 = atc.e0[b];

    let mut boa = vec![0i16; nc];
    rows2(n_jobs, nx, &mut boa, qai, |i, brow, qrow| {
        let p0 = i * nx;
        for j in 0..nx {
            let p = p0 + j;
            if qai::off(qrow[j]) {
                brow[j] = NODATA;
                continue;
            }
            let toa = (inp.toa[p] as f64 / 10000.0) as f32;
            let ref_: f32;
            if par.doatmo {
                let z = inp.dem[p] as usize;

                // smooth atc variables
                let w = interpolation_weights(j as i32, i as i32, nf, ne, gres, fres, xy_vz, vnodata);
                let t = interpolate_coarse(&w, xyz_t[z]);
                let ts = interpolate_coarse(&w, xyz_ts[z]);
                let tss = interpolate_coarse(&w, xyz_tss[z]);
                let tsd = interpolate_coarse(&w, xyz_tsd[z]);
                let tvs = interpolate_coarse(&w, xyz_tvs[z]);
                let tvd = interpolate_coarse(&w, xyz_tvd[z]);
                let s = interpolate_coarse(&w, xyz_s[z]);
                let rho_p = interpolate_coarse(&w, xyz_rho_p[z]);
                // (the C interpolates the SWIR2 transmittances from the arrays of the band itself)
                let tss_sw2 = interpolate_coarse(&w, xyz_tss[z]);
                let tsd_sw2 = interpolate_coarse(&w, xyz_tsd[z]);
                let brdf = if par.dobrdf { interpolate_coarse(&w, xy_brdf) } else { 1.0f32 };

                let tg = match inp.tg {
                    None => interpolate_coarse(&w, xy_tg),
                    Some(tg_) => (tg_[p] as f64 / 10000.0) as f32,
                };

                // topographic correction factor
                let mut a = 1.0f32;
                if par.dotopo {
                    let sky = (inp.sky[p] as f64 / 10000.0) as f32;
                    let ill = (inp.ill[p] as f64 / 10000.0) as f32;
                    let cf = (inp.cf[p] as f64 / 10000.0) as f32;

                    if ill > 0.0 {
                        let szen = interpolate_coarse(&w, xy_sz);
                        let ms = (szen as f64).cos() as f32;
                        let tso = interpolate_coarse(&w, xy_tso);
                        let tvo = interpolate_coarse(&w, xy_tvo);
                        let e0_ = e0 * tvo * tso;
                        let f = e0_ * tss / (e0_ * tsd);
                        let f0 = e0_ * tss_sw2 / (e0_ * tsd_sw2);
                        let h0 = ((std::f64::consts::PI + (2.0f32 * szen) as f64) / (2.0 * std::f64::consts::PI)) as f32;

                        a = (ms + cf / f0 * f / h0) / (ill + sky * cf / f0 * f / h0);
                        if a < 0.0 {
                            a = -10000.0;
                        }
                    } else {
                        a = 1.0;
                    }
                }

                if par.doenv {
                    // target reflectance
                    let bck = (inp.bck.unwrap()[p] as f64 / 10000.0) as f32;
                    let tmp = 1.0f32 - bck * s;
                    ref_ = a * brdf * (toa / tg * tmp - rho_p * tmp - ts * tvs * bck) / (ts * tvd);
                } else {
                    // homogeneous target reflectance
                    let tmp = (toa - rho_p) / tg;
                    ref_ = a * brdf * tmp / (t + s * tmp);
                }
            } else {
                // top-of-atmosphere
                ref_ = toa;
            }

            if (ref_ as f64) < 0.0 {
                qai::set_subzero(&mut qrow[j], true);
            }
            if (ref_ as f64) > 1.0 {
                qai::set_saturation(&mut qrow[j], true);
            }

            if par.erase_cloud && qai::cloud(qrow[j]) == 2 {
                brow[j] = NODATA;
            } else if (ref_ as f64) < -1.0 {
                brow[j] = NODATA;
                qai::set_off(&mut qrow[j], true);
            } else if ref_ as f64 * 10000.0 > i16::MAX as f64 {
                brow[j] = i16::MAX;
            } else {
                brow[j] = c_short(ref_ as f64 * 10000.0);
            }
        }
    });
    Ok(boa)
}

/// `background_reflectance`: the background reflectance (times 10000, `short`) of the band `b` that corrects the
/// adjacency effect: the BOA reflectance of every pixel without the environmental correction (homogeneous target), the
/// mean of every row and column of the pixels that are neither off nor cloud, three rings (radius 0.25, 1 and 4.5 km)
/// of exponential moving averages (alpha `2 / (r + 1)`, r in pixels) along each row (left to right, then right to
/// left) and then each column (top to bottom, then bottom to top; the pixels that are off are skipped, the cloud
/// pixels take the running value), their mean weighted by the environmental weights of the rings, and the pixel's own
/// value weighted by the weight `F` of its elevation bin. `tg` is the gaseous transmittance of the band as in
/// [`SurfaceInput`], `dem` the binned elevation.
pub fn background_reflectance(
    atc: &Atc,
    b: usize,
    toa: &[i16],
    tg: Option<&[i16]>,
    dem: &[u8],
    grid: &SceneGrid,
    qai: &[i16],
    n_jobs: i32,
) -> Result<Vec<i16>, AtmoError> {
    let (nx, ny, res) = (grid.nx, grid.ny, grid.res);
    let nc = nx * ny;
    shape(grid, &[("the QAI", qai.len()), ("the TOA", toa.len()), ("the DEM", dem.len())])?;
    if let Some(tg) = tg {
        shape(grid, &[("the gaseous transmittance", tg.len())])?;
    }
    if atc.xyz_rho_p.len() < crate::atc::N_ELEV || b >= atc.xy_tg.nb() || atc.mod_.len() <= b || atc.aod.len() <= b {
        return Err(AtmoError::Shape("the atc needs one brick per elevation bin, and the band".into()));
    }
    let xyz_rho_p: Vec<&[f32]> = atc.xyz_rho_p.iter().map(|c| c.bands[b].as_slice()).collect();
    let xyz_t: Vec<&[f32]> = atc.xyz_t.iter().map(|c| c.bands[b].as_slice()).collect();
    let xyz_s: Vec<&[f32]> = atc.xyz_s.iter().map(|c| c.bands[b].as_slice()).collect();
    let xyz_f: Vec<&[f32]> = atc.xyz_f.iter().map(|c| c.bands[b].as_slice()).collect();
    let xy_tg = atc.xy_tg.bands[b].as_slice();
    let (gnx, gres) = (atc.xy_sun.nx, atc.xy_sun.res);

    // radius of rings
    let km: [f32; 3] = [0.25, 1.00, 4.50];
    let mut r = [0.0f32; 3];
    for k in 0..3 {
        r[k] = ((km[k] * 1000.0f32) as f64 / res) as f32;
    }
    // alpha for exponential moving average
    let mut alpha = [0.0f32; 3];
    for k in 0..3 {
        alpha[k] = (2.0 / (r[k] as f64 + 1.0)) as f32;
    }

    // BOA reflectance without env. correction
    let mut rf = vec![0i16; nc];
    rows1(n_jobs, nx, &mut rf, |i, row| {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) {
                continue;
            }
            let g = ji2p(res, gres, gnx, i, j);
            let z = dem[p] as usize;
            let tgv = match tg {
                None => xy_tg[g],
                Some(t) => (t[p] as f64 / 10000.0) as f32,
            };
            let rho_p = xyz_rho_p[z][g];
            let t = xyz_t[z][g];
            let s = xyz_s[z][g];
            let tmp = ((toa[p] as f64 / 10000.0 - rho_p as f64) / tgv as f64) as f32;
            row[j] = c_short((tmp / (t + s * tmp) * 10000.0f32) as f64);
        }
    });

    // row means
    let mut imean = vec![0.0f32; ny];
    rows1(n_jobs, 1, &mut imean, |i, m| {
        let (mut sum, mut num) = (0.0f64, 0.0f64);
        for p in i * nx..(i + 1) * nx {
            if qai::off(qai[p]) || qai::cloud(qai[p]) > 0 {
                continue;
            }
            sum += rf[p] as f64 / 10000.0;
            num += 1.0;
        }
        if num > 0.0 {
            m[0] = (sum / num) as f32;
        }
    });

    // column means (every column sums its rows in order, as the C's loop over a column does)
    let mut jsum = vec![0.0f64; nx];
    let mut jnum = vec![0.0f64; nx];
    for i in 0..ny {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) || qai::cloud(qai[p]) > 0 {
                continue;
            }
            jsum[j] += rf[p] as f64 / 10000.0;
            jnum[j] += 1.0;
        }
    }
    let jmean: Vec<f32> = (0..nx).map(|j| if jnum[j] > 0.0 { (jsum[j] / jnum[j]) as f32 } else { 0.0 }).collect();

    // exponential moving average, left to right, then right to left
    let mut avg = vec![[0.0f32; 3]; nc];
    rows1(n_jobs, nx, &mut avg, |i, row| {
        let mut old = [imean[i]; 3];
        let p0 = i * nx;
        for j in 0..nx {
            let p = p0 + j;
            if qai::off(qai[p]) {
                continue;
            }
            for k in 0..3 {
                if qai::cloud(qai[p]) > 0 {
                    row[j][k] = old[k];
                } else {
                    row[j][k] = (((alpha[k] * rf[p] as f32) as f64) / 10000.0 + (1.0 - alpha[k] as f64) * old[k] as f64) as f32;
                    old[k] = row[j][k];
                }
            }
        }
        for j in (0..nx).rev() {
            let p = p0 + j;
            if qai::off(qai[p]) {
                continue;
            }
            for k in 0..3 {
                if qai::cloud(qai[p]) > 0 {
                    row[j][k] = old[k];
                } else {
                    row[j][k] = ((alpha[k] * row[j][k]) as f64 + (1.0 - alpha[k] as f64) * old[k] as f64) as f32;
                    old[k] = row[j][k];
                }
            }
        }
    });

    // exponential moving average, top to bottom, then bottom to top (every column keeps its own running value)
    let mut old: Vec<[f32; 3]> = jmean.iter().map(|&m| [m; 3]).collect();
    let mut pass = |i: usize, old: &mut Vec<[f32; 3]>| {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) {
                continue;
            }
            for k in 0..3 {
                if qai::cloud(qai[p]) > 0 {
                    avg[p][k] = old[j][k];
                } else {
                    avg[p][k] = ((alpha[k] * avg[p][k]) as f64 + (1.0 - alpha[k] as f64) * old[j][k] as f64) as f32;
                    old[j][k] = avg[p][k];
                }
            }
        }
    };
    for i in 0..ny {
        pass(i, &mut old);
    }
    for i in (0..ny).rev() {
        pass(i, &mut old);
    }

    // relative influence of rings
    let mod_ = mod_elev_scale(atc.mod_[b], 1.0, atc.hr);
    let aod = atc.aod[b];
    let mut f = [0.0f32; 3];
    for k in 0..3 {
        let fa = env_weight_aerosol(km[k]);
        let fr = env_weight_molecular(km[k]);
        f[k] = env_weight(aod, mod_, fa, fr);
    }
    let df = [f[0], f[1] - f[0], f[2] - f[1]];
    let sf = df[0] + df[1] + df[2];

    // background reflectance
    let mut bck = vec![0i16; nc];
    rows1(n_jobs, nx, &mut bck, |i, row| {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) {
                continue;
            }
            let g = ji2p(res, gres, gnx, i, j);
            let z = dem[p] as usize;
            let f_ = xyz_f[z][g];
            let a = avg[p];
            let w_avg = (df[0] * a[0] + df[1] * a[1] + df[2] * a[2]) / sf;
            row[j] = c_short((rf[p] as f64 / 10000.0 * f_ as f64 + ((1.0f32 - f_) * w_avg) as f64) * 10000.0);
        }
    });
    Ok(bck)
}

/// The BOA product: the bands in the order BLUE, GREEN, RED, REDEDGE1, REDEDGE2, REDEDGE3, BROADNIR, NIR, SWIR1,
/// SWIR2 (those the sensor has), each with its domain and wavelength (um); `product` is "BOA", or "TOA" without `doatmo`.
#[derive(Clone, Debug)]
pub struct Boa {
    pub product: &'static str,
    pub bands: Vec<Vec<i16>>,
    pub domain: Vec<String>,
    pub wavelength: Vec<f32>,
}

/// The domains of the BOA bands, in the order of the output.
const BOA_DOMAINS: [&str; 10] = ["BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2", "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2"];

/// `compile_l2_boa`: the BOA (or TOA, without `par.doatmo`) bands: for each band of the domains above that the TOA has,
/// the gaseous transmittance (`par.doatmo`, Sentinel-2: from `wvp`, the water vapour times 1000, through `lut`), the
/// background reflectance (`par.doatmo` and `par.doenv`) and the surface reflectance. The pixels that are off after
/// all the bands (the surface reflectance sets some off) are the no-data value in every band. `top` has the binned
/// elevation and, with `par.dotopo`, the illumination, sky view factor and C-factor.
#[allow(clippy::too_many_arguments)]
pub fn compile_l2_boa(
    par: &L2Params,
    mission: Mission,
    atc: &Atc,
    toa: &Toa,
    qai: &mut [i16],
    wvp: Option<&[i16]>,
    lut: Option<&WvLut>,
    top: &Top,
    n_jobs: i32,
) -> Result<Boa, L2Error> {
    let grid = toa.grid;
    let nc = grid.nx * grid.ny;
    if top.dem.len() != nc {
        return Err(L2Error::Other(format!("the binned DEM has {} values for a grid of {nc} pixels", top.dem.len())));
    }
    if par.dotopo && (top.ill.len() != nc || top.sky.len() != nc || top.c.len() != nc) {
        return Err(L2Error::Other("the topographic correction needs the illumination, sky view factor and C-factor".into()));
    }
    let bands: Vec<usize> = BOA_DOMAINS.iter().filter_map(|d| toa.find_domain(d)).collect();
    let mut out = Boa {
        product: if par.doatmo { "BOA" } else { "TOA" },
        bands: Vec::with_capacity(bands.len()),
        domain: Vec::new(),
        wavelength: Vec::new(),
    };
    for &b in &bands {
        if toa.bands[b].len() != nc {
            return Err(L2Error::Other("the TOA bands must have nx * ny pixels".into()));
        }
        let tg = if par.doatmo && mission == Mission::Sentinel2 {
            let (Some(wvp), Some(lut)) = (wvp, lut) else {
                return Err(L2Error::Other("the gas transmittance needs the water vapour and its LUT".into()));
            };
            Some(gas::gas_transmittance(atc, b, wvp, &grid, qai, lut)?)
        } else {
            None
        };
        let bck = if par.doatmo && par.doenv {
            Some(background_reflectance(atc, b, &toa.bands[b], tg.as_deref(), &top.dem, &grid, qai, n_jobs)?)
        } else {
            None
        };
        let inp = SurfaceInput {
            toa: &toa.bands[b],
            bck: bck.as_deref(),
            tg: tg.as_deref(),
            dem: &top.dem,
            ill: &top.ill,
            sky: &top.sky,
            cf: &top.c,
        };
        let boa = surface_reflectance(par, atc, b, &inp, &grid, qai, n_jobs)?;
        out.bands.push(boa);
        out.domain.push(toa.domain[b].clone());
        out.wavelength.push(atc.wvl.get(b).copied().unwrap_or(0.0));
    }
    // force nodata in all bands
    for p in 0..nc {
        if qai::off(qai[p]) {
            for band in out.bands.iter_mut() {
                band[p] = NODATA;
            }
        }
    }
    for d in ["NIR", "SWIR1", "RED"] {
        if !out.domain.iter().any(|x| x == d) {
            return Err(L2Error::Atmo(AtmoError::Domain(d)));
        }
    }
    Ok(out)
}

/// `compile_l2_qai`: makes sure that the off bit is set exclusively: a pixel that is off becomes 1.
pub fn compile_l2_qai(qai: &mut [i16]) {
    for q in qai.iter_mut() {
        if qai::off(*q) {
            *q = 1;
        }
    }
}

/// `cloud_distance` of `cloud-ll.c`: the distance in metres to the next pixel that is cloud (any cloud flag), cloud
/// shadow or snow, capped at 32767 (`SHRT_MAX`); the pixels that are off give `nodata`; with no such pixel in the image
/// the distance is `nx + ny` pixels.
pub fn cloud_distance(qai: &[i16], grid: &SceneGrid, nodata: i16) -> Vec<i16> {
    let (nx, ny, res) = (grid.nx, grid.ny, grid.res);
    let nc = nx * ny;
    let mut to_dist = vec![0u8; nc];
    let mut k = 0;
    for p in 0..nc {
        if qai::cloud(qai[p]) > 0 || qai::shadow(qai[p]) || qai::snow(qai[p]) {
            to_dist[p] = 1;
            k += 1;
        }
    }
    let dist_pix: Vec<u16> = if k > 0 { dist_transform(&to_dist, nx, ny) } else { vec![(ny + nx) as u16; nc] };
    (0..nc)
        .map(|p| {
            if qai::off(qai[p]) {
                nodata
            } else {
                let dist = dist_pix[p] as f64 * res;
                if dist < i16::MAX as f64 {
                    c_short(dist)
                } else {
                    i16::MAX
                }
            }
        })
        .collect()
}

/// `compile_l2_dst`: the distance to the next cloud, cloud shadow or snow pixel in metres ([`cloud_distance`]), from the
/// final QAI.
pub fn compile_l2_dst(qai: &[i16], grid: &SceneGrid) -> Vec<i16> {
    cloud_distance(qai, grid, NODATA)
}

/// `compile_l2_vzn`: the view zenith angle in degrees times 100 for every pixel that is not off (`qai` is the final QAI
/// of the image `grid`): the angle of each cell of the coarse grid (`atc.xy_view`, `short`), the cells without data
/// filled by the inverse distance weighting of GDAL ([`crate::gdal_grid`], over the cell indices), interpolated to
/// the pixels. With the IDW's reciprocal (`rcpps`), the bits depend on the CPU as in [`crate::aod`].
pub fn compile_l2_vzn(atc: &Atc, grid: &SceneGrid, qai: &[i16], n_jobs: i32) -> Result<Vec<i16>, L2Error> {
    shape(grid, &[("the QAI", qai.len())])?;
    let (nf, ne) = (atc.xy_view.nx, atc.xy_view.ny);
    let ng = nf * ne;
    let (gres, fres) = (atc.xy_view.res, grid.res);
    let nodata = NODATA;
    if atc.xy_view.ncell() != ng || ng == 0 {
        return Err(L2Error::Other("the view zenith grid is empty".into()));
    }

    // copy coarse view zenith
    let coarse: Vec<i16> = (0..ng)
        .map(|g| {
            if atc.xy_view.is_nodata(ZEN, g) {
                nodata
            } else {
                c_short(atc.xy_view.bands[ZEN][g] as f64 * R2D * 100.0)
            }
        })
        .collect();

    // copy to arrays
    let (mut gx, mut gy, mut gz) = (Vec::new(), Vec::new(), Vec::new());
    for e in 0..ne {
        for f in 0..nf {
            let g = e * nf + f;
            if coarse[g] == nodata {
                continue;
            }
            gx.push(f as f64);
            gy.push(e as f64);
            gz.push(coarse[g] as f64);
        }
    }
    // compute interpolation
    let grid_i = crate::gdal_grid::invdist_grid(&gx, &gy, &gz, 0.0, (nf - 1) as f64, 0.0, (ne - 1) as f64, nf, ne);

    // copy to float array
    let fcoarse: Vec<f32> = (0..ng).map(|g| if coarse[g] == nodata { grid_i[g] as f32 } else { coarse[g] as f32 }).collect();

    // interpolate at full res
    let (nx, nc) = (grid.nx, grid.nx * grid.ny);
    let mut vzn = vec![0i16; nc];
    rows1(n_jobs, nx, &mut vzn, |i, row| {
        for j in 0..nx {
            let p = i * nx + j;
            if qai::off(qai[p]) {
                row[j] = nodata;
            } else {
                let w = interpolation_weights(j as i32, i as i32, nf as i32, ne as i32, gres, fres, &fcoarse, nodata as f32);
                row[j] = c_short(interpolate_coarse(&w, &fcoarse) as f64);
            }
        }
    });
    Ok(vzn)
}

/// `compile_l2_hot`: the haze optimized transform `blue - 0.5 * red - 800` of the TOA reflectance (times 10000), for
/// the pixels that are not off.
pub fn compile_l2_hot(toa: &Toa, qai: &[i16]) -> Result<Vec<i16>, L2Error> {
    shape(&toa.grid, &[("the QAI", qai.len())])?;
    let blue = toa.find_domain("BLUE").ok_or(AtmoError::Domain("BLUE"))?;
    let red = toa.find_domain("RED").ok_or(AtmoError::Domain("RED"))?;
    let (blue_, red_) = (&toa.bands[blue], &toa.bands[red]);
    shape(&toa.grid, &[("the BLUE band", blue_.len()), ("the RED band", red_.len())])?;
    Ok((0..qai.len())
        .map(|p| if qai::off(qai[p]) { NODATA } else { c_short(blue_[p] as f64 - 0.5 * red_[p] as f64 - 800.0) })
        .collect())
}

/// `compile_l2_aod`: the AOD of the green band at the elevation of every pixel (`dem`, the binned elevation) times 1000,
/// interpolated from the coarse grid (`atc.xyz_aod` after `atmo_elevdep`); the no-data value where the pixel is off or
/// without `par.doatmo`. The product's wavelength is that of the green band (`atc.wvl`).
pub fn compile_l2_aod(par: &L2Params, atc: &Atc, grid: &SceneGrid, qai: &[i16], dem: &[u8], n_jobs: i32) -> Result<Vec<i16>, L2Error> {
    shape(grid, &[("the QAI", qai.len()), ("the DEM", dem.len())])?;
    let nx = grid.nx;
    let b_green = atc.xy_aod.find_domain("GREEN").ok_or(AtmoError::Domain("GREEN"))?;
    let (nf, ne) = (atc.xy_aod.nx as i32, atc.xy_aod.ny as i32);
    let gres = atc.xy_aod.res;
    let vnodata = atc.xy_aod.nodata as f32;
    let xy_aod: Vec<&[f32]> = atc.xyz_aod.iter().map(|c| c.bands[b_green].as_slice()).collect();
    if xy_aod.len() < crate::atc::N_ELEV {
        return Err(L2Error::Atmo(AtmoError::Shape("the atc needs one brick per elevation bin".into())));
    }
    let mut aod = vec![0i16; nx * grid.ny];
    rows1(n_jobs, nx, &mut aod, |i, row| {
        for j in 0..nx {
            let p = i * nx + j;
            let z = dem[p] as usize;
            if qai::off(qai[p]) || !par.doatmo {
                row[j] = NODATA;
            } else {
                let w = interpolation_weights(j as i32, i as i32, nf, ne, gres, grid.res, xy_aod[z], vnodata);
                row[j] = c_short((interpolate_coarse(&w, xy_aod[z]) * 1000.0f32) as f64);
            }
        }
    });
    Ok(aod)
}

/// `compile_l2_wvp`: the water vapour times 1000. With `wvp` (Sentinel-2, from `water_vapor`) that is the brick as it
/// is; without it (Landsat) the scene's `atc.wvp` for every pixel that is not off, the no-data value without
/// `par.doatmo`.
pub fn compile_l2_wvp(par: &L2Params, atc: &Atc, qai: &[i16], wvp: Option<Vec<i16>>) -> Vec<i16> {
    match wvp {
        Some(w) => w,
        None => qai
            .iter()
            .map(|&q| if qai::off(q) || !par.doatmo { NODATA } else { c_short((atc.wvp * 1000.0f32) as f64) })
            .collect(),
    }
}

/// The Level 2 products of one scene (`compile_level2`), on the grid of the scene: the BOA, the QAI (the off pixels
/// are 1) and, if asked for in the parameters, the DST, VZN, HOT, AOD and WVP.
#[derive(Clone, Debug)]
pub struct Level2 {
    pub qai: Vec<i16>,
    pub boa: Boa,
    pub dst: Option<Vec<i16>>,
    pub vzn: Option<Vec<i16>>,
    pub hot: Option<Vec<i16>>,
    pub aod: Option<Vec<i16>>,
    pub wvp: Option<Vec<i16>>,
}

/// `compile_level2`: the products, in FORCE's order (see the module notes). `wvp` is the water vapour of
/// [`gas::water_vapor`] (Sentinel-2) and `lut` its LUT (the one the gas transmittance reads); `qai` is changed as in
/// the C (flags of the surface reflectance; the off pixels become 1).
#[allow(clippy::too_many_arguments)]
pub fn compile_level2(
    par: &L2Params,
    mission: Mission,
    atc: &Atc,
    toa: &Toa,
    qai: &mut [i16],
    wvp: Option<Vec<i16>>,
    lut: Option<&WvLut>,
    top: &Top,
    n_jobs: i32,
) -> Result<Level2, L2Error> {
    let grid = toa.grid;
    let hot = if par.ohot { Some(compile_l2_hot(toa, qai)?) } else { None };
    let aod = if par.oaod { Some(compile_l2_aod(par, atc, &grid, qai, &top.dem, n_jobs)?) } else { None };

    // do BOA near the end (TOA is altered within)
    let boa = compile_l2_boa(par, mission, atc, toa, qai, wvp.as_deref(), lut, top, n_jobs)?;

    // do WVP after BOA
    let wvp = if par.owvp { Some(compile_l2_wvp(par, atc, qai, wvp)) } else { None };

    // do QAI at the very end (QAI is altered within)
    compile_l2_qai(qai);

    let dst = if par.odst { Some(compile_l2_dst(qai, &grid)) } else { None };
    let vzn = if par.ovzn { Some(compile_l2_vzn(atc, &grid, qai, n_jobs)?) } else { None };
    Ok(Level2 { qai: qai.to_vec(), boa, dst, vzn, hot, aod, wvp })
}

/// What `radiometric_correction` reads from the parameter file besides [`L2Params`]: the directory of the AOD files
/// (`d_aod`, `None` is `"NULL"`) and where the water vapour of a Landsat image comes from (`d_wvp`, `wvp`,
/// `wvp_strict`).
#[derive(Clone, Debug)]
pub struct L2Options {
    pub aod_dir: Option<String>,
    pub wvp: WvpSource,
}

/// `radiometric_correction`: the radiometric core of FORCE's Level 2 processing, from the TOA reflectance (`toa`, with
/// the metadata `meta`), the QAI (`qai`, after `sun_target_view`, `compile_topography` and the cloud detection) and the
/// topography (`top`, from [`crate::topo::compile_topography`]; this sets `top.c`), on the coarse grid `atc` that
/// [`crate::atc::Atc::allocate`] and [`crate::sunview::sun_target_view`] have filled and `compile_topography` has given
/// its elevation statistics. `aoi` is the area of interest (one byte per pixel) or `None`. `brdf` is the BRDF factor
/// of [`atmo_angledep`]. See the module notes for the order.
#[allow(clippy::too_many_arguments)]
pub fn radiometric_correction(
    par: &L2Params,
    opts: &L2Options,
    meta: &Meta,
    mission: Mission,
    geo: &SceneGeo,
    atc: &mut Atc,
    toa: &Toa,
    qai: &mut [i16],
    aoi: Option<&[u8]>,
    top: &mut Top,
    brdf: Option<BrdfFn>,
    n_jobs: i32,
) -> Result<Level2, L2Error> {
    let grid = toa.grid;
    let nb = toa.bands.len();
    let nc = grid.nx * grid.ny;
    if qai.len() != nc || top.dem.len() != nc {
        return Err(L2Error::Other("the QAI and the binned DEM must have nx * ny pixels".into()));
    }
    let mut wvp = None;
    let mut lut = None;

    if par.doatmo {
        // elevation stats (mean, max/min, # of 100m classes), rayleigh scattering @ sea level, elevation correction
        // factors for optical depths
        if atc.wvl.len() < nb || atc.mod_.len() < nb {
            return Err(L2Error::Other("the atc must have a wavelength and a MOD for every band of the TOA".into()));
        }
        for b in 0..nb {
            atc.mod_[b] = molecular_optical_depth(atc.wvl[b]);
        }
        atc.hr = mod_elev_factor(atc.dem.avg);

        // precipitable water vapor
        if mission == Mission::Landsat {
            let (lon, lat) = geo.lonlat((atc.xy_sun.nx / 2) as i32, (atc.xy_sun.ny / 2) as i32, atc.xy_sun.res);
            atc.wvp = gas::water_vapor_from_lut(&opts.wvp, lon, lat, geo.date.year, geo.date.month, geo.date.day)?;
            if atc.wvp < 0.0 {
                return Err(L2Error::Other("cannot read wvp from LUT".into()));
            }
        } else {
            atc.wvp = 0.0;
        }

        // angle-dependent coarse-grid atmospheric modelling
        atmo_angledep(atc, meta, geo, &grid, qai, &top.dem, brdf, n_jobs)?;

        // compile AOD, use image-based water/shadow targets, refine by DODB, use external values
        let (lon, lat) = geo.lonlat((atc.xy_aod.nx / 2) as i32, (atc.xy_aod.ny / 2) as i32, atc.xy_aod.res);
        let rsr_band: Vec<usize> = meta.cal.iter().map(|c| c.rsr_band).collect();
        let ap = AodParams {
            doaod: par.doaod,
            dir: opts.aod_dir.as_deref(),
            center_lon: lon,
            center_lat: lat,
            doy: geo.date.doy,
            rsr_band: &rsr_band,
            multi: par.domulti,
        };
        let _dark = compile_aod(atc, &ap, toa, qai, top)?;

        // elevation-dependent coarse-grid atmospheric modelling
        atmo_elevdep(atc, par, &grid, qai, &top.dem, n_jobs)?;

        // water vapor and gaseous transmittance estimation
        if mission == Mission::Sentinel2 {
            let l = WvLut::build(&rsr_band, atc, n_jobs)?;
            let w = gas::water_vapor(&rsr_band, atc, &l, toa, qai, &top.dem)?;
            wvp = Some(w.wvp);
            lut = Some(l);
        }

        // estimate topographic correction factor
        if par.dotopo {
            top.c = cfactor_topography(atc, toa, qai, &top.dem, &top.slope, &top.ill)?;
        }
    }

    // apply AOI mask
    apply_aoi(qai, aoi)?;

    // Level 2 datasets
    compile_level2(par, mission, atc, toa, qai, wvp, lut.as_ref(), top, n_jobs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: usize, ny: usize, res: f64) -> SceneGrid {
        SceneGrid { nx, ny, res }
    }

    #[test]
    fn c_short_is_what_the_x86_conversion_gives() {
        assert_eq!((c_short(0.0), c_short(1234.9), c_short(-1234.9), c_short(32767.0), c_short(-32768.0)), (0, 1234, -1234, 32767, -32768));
        // out of range: the low 16 bits of the 32 bit integer; NaN and the range of the integer: INT_MIN, whose low bits are 0
        assert_eq!(c_short(40000.7), 40000u16 as i16);
        assert_eq!((c_short(f64::NAN), c_short(f64::INFINITY), c_short(-3.0e10)), (0, 0, 0));
    }

    #[test]
    fn the_cloud_distance_is_the_truncated_euclidean_distance_in_metres() {
        let (nx, ny) = (12, 9);
        let mut qai = vec![0i16; nx * ny];
        qai[2 * nx + 3] = 1 << 1; // cloud
        qai[6 * nx + 9] = 1 << 3; // shadow
        qai[8 * nx + 0] = 1 << 4; // snow
        qai[4 * nx + 4] = 1; // off
        let flagged: Vec<(i64, i64)> = vec![(2, 3), (6, 9), (8, 0)];
        let d = cloud_distance(&qai, &grid(nx, ny, 30.0), -9999);
        for i in 0..ny {
            for j in 0..nx {
                let p = i * nx + j;
                if p == 4 * nx + 4 {
                    assert_eq!(d[p], -9999);
                    continue;
                }
                let best = flagged.iter().map(|&(a, b)| ((a - i as i64).pow(2) + (b - j as i64).pow(2)) as f64).fold(f64::MAX, f64::min);
                assert_eq!(d[p], (best.sqrt().floor() * 30.0) as i16, "pixel ({i}, {j})");
            }
        }
        // nothing flagged: nx + ny pixels; the distance is capped at 32767
        let d = cloud_distance(&vec![0i16; nx * ny], &grid(nx, ny, 30.0), -9999);
        assert!(d.iter().all(|&v| v == ((nx + ny) * 30) as i16));
        let d = cloud_distance(&vec![0i16; nx * ny], &grid(nx, ny, 3000.0), -9999);
        assert!(d.iter().all(|&v| v == i16::MAX));
    }

    #[test]
    fn hot_wvp_and_qai_products() {
        let toa = Toa {
            grid: grid(2, 2, 10.0),
            bands: vec![vec![1000, 2000, -30000, 5], vec![400, 800, 30000, 7]],
            domain: vec!["BLUE".into(), "RED".into()],
        };
        let qai = [0i16, 1, 0, 0];
        // blue - red / 2 - 800, truncated; the pixel that is off is the no-data value; the last wraps as the C's cast does
        let hot = compile_l2_hot(&toa, &qai).unwrap();
        assert_eq!(hot, vec![0, NODATA, c_short(-30000.0 - 15000.0 - 800.0), 5 - 3 - 800]);
        assert_eq!(compile_l2_hot(&Toa { domain: vec!["BLUE".into(), "NIR".into()], ..toa.clone() }, &qai), Err(L2Error::Atmo(AtmoError::Domain("RED"))));
        // the QAI: the off pixels become 1, the other flags stay
        let mut q = [0b101i16, 0b1000, 0, 0b11];
        compile_l2_qai(&mut q);
        assert_eq!(q, [1, 0b1000, 0, 1]);
        // the water vapour of Landsat: the scene's, times 1000, where the pixel is not off and the atmosphere is corrected
        let mut atc = Atc::zeros(1, 1, 5000.0, 1, -9999);
        atc.wvp = 2.4;
        let par = L2Params { doatmo: true, ..Default::default() };
        assert_eq!(compile_l2_wvp(&par, &atc, &[0, 1, 0], None), vec![2400, NODATA, 2400]);
        assert_eq!(compile_l2_wvp(&L2Params::default(), &atc, &[0, 1, 0], None), vec![NODATA; 3]);
        assert_eq!(compile_l2_wvp(&par, &atc, &[0, 1], Some(vec![7, 8])), vec![7, 8]);
    }

    #[test]
    fn the_view_zenith_product_fills_the_cells_without_data() {
        // a 3 x 2 coarse grid of 5000 m whose view zenith is 0.1 rad everywhere but in one cell without data
        let mut atc = Atc::zeros(3, 2, 5000.0, 1, -9999);
        atc.xy_view = crate::atc::CoarseBrick::new(3, 2, 5000.0, 8, -9999);
        atc.xy_view.bands[ZEN] = vec![0.1, 0.1, 0.1, 0.1, -9999.0, 0.1];
        let g = grid(300, 200, 50.0);
        let qai = vec![0i16; 300 * 200];
        let v = compile_l2_vzn(&atc, &g, &qai, 1).unwrap();
        // 0.1 rad is 572.9 hundredths of a degree: the cells keep (short)572 and the filled one is the IDW of those (572)
        assert!(v.iter().all(|&x| (571..=572).contains(&x)), "{:?}", v.iter().min());
        // the pixels that are off are the no-data value
        let mut q2 = qai.clone();
        q2[10] = 1;
        let v2 = compile_l2_vzn(&atc, &g, &q2, 2).unwrap();
        assert_eq!(v2[10], NODATA);
        assert_eq!(v2[11], v[11]);
    }

    #[test]
    fn the_threads_do_not_change_the_surface_reflectance() {
        // a scene without the atmospheric correction: the TOA reflectance, flags and erasing of the clouds
        let (nx, ny) = (7, 5);
        let atc = {
            let mut a = Atc::zeros(1, 1, 5000.0, 1, -9999);
            a.xy_mod.domain = vec!["SWIR2".into()];
            a
        };
        let toa: Vec<i16> = (0..nx * ny).map(|p| (p as i16 - 12) * 900).collect();
        let dem = vec![0u8; nx * ny];
        let mut qai: Vec<i16> = (0..nx * ny).map(|p| if p == 3 { 2 << 1 } else { 0 }).collect();
        let inp = SurfaceInput { toa: &toa, bck: None, tg: None, dem: &dem, ill: &[], sky: &[], cf: &[] };
        let par = L2Params { erase_cloud: true, ..Default::default() };
        let mut q3 = qai.clone();
        let a = surface_reflectance(&par, &atc, 0, &inp, &grid(nx, ny, 5000.0), &mut qai, 1).unwrap();
        let b = surface_reflectance(&par, &atc, 0, &inp, &grid(nx, ny, 5000.0), &mut q3, 3).unwrap();
        assert_eq!((&a, &qai), (&b, &q3));
        assert_eq!(a[3], NODATA); // the unsure cloud is erased (and is not off)
        assert_eq!(qai[3] & 1, 0);
        // p = 0: reflectance -1.08 < -1: off; p = 1: -0.99: subzero; p = 34: 1.98: saturated (the float of 1.98 is a little less)
        assert_eq!((a[0], qai[0] & 1), (NODATA, 1));
        assert!(qai::subzero(qai[1]) && !qai::saturation(qai[1]) && a[1] == -9900);
        assert!(qai::saturation(qai[34]) && (19799..=19800).contains(&a[34]));
    }
}
