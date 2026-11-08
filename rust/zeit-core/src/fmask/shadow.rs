//! The cloud shadow step of Fmask 4.7 (https://github.com/GERSL/Fmask, commit e32f5bd, MIT License, Copyright (c) 2018
//! Global Environmental Remote Sensing Lab), Sentinel-2 route, one function per .m file:
//!
//! * [`detect_potential_cloud_shadow`]: `DetectPotentialCloudShadow.m` (the SCS+C topographic correction of NIR and SWIR
//!   included, `getDataTopoCorrected`, with `polyfit`),
//! * [`stratied_sample_handler`]: `stratiedSampleHanlder.m` (with `distance = 0`, the only call of the shadow route;
//!   the `rangesearch` branch of `distance > 0` is not ported),
//! * [`project_dem_to_plane`]: `ProjectDEM2Plane.m`,
//! * [`get_real_cloud_position_s2`]: `getRealCloudPositionS2.m` (of one pixel),
//! * [`match_cloud_shadow`]: `MatchCloudShadow.m`,
//! * [`buffer_masks`]: `BufferMasks.m`.
//!
//! What is NOT ported, because the route is Sentinel-2: `getRealCloudPosition.m` and `getSensorViewGeo.m` (the view
//! geometry of Landsat), the cloud height by temperature (`data_bt_c`, `t_templ`, `t_temph`: Sentinel-2 has no
//! thermal band, so `data_bt_c` is empty in `autoFmask.m` and `MatchCloudShadow.m` never enters those branches) and
//! the Landsat sensors.
//!
//! The classes of MATLAB are followed down to the internal values. `single + double` and `single .* double` are
//! single, the double converted to single before the operation. A comparison of a single with a double constant is
//! done in single in GNU Octave (which the harness runs), and here too (where the constant is representable, as 200
//! of `shadow_prob>200`, it makes no difference). So the heights of a scene with a DEM are single (`200.00 +
//! base_heigh_cloud` is single, and so are the range `Min_cl_height:step_interval:Max_cl_height`, `h`, `i_xy`,
//! `record_base_h` and `height_clds_recorder`), and those of a scene without one are double; the view-angle shift
//! (`getRealCloudPositionS2`) is single in both, since `tan(VZAxy)` is single; `tmp_xys` is a double array holding
//! single values. The images are row-major slices, but the order of the COLUMNS (MATLAB's linear order) is kept
//! where it changes the result (the vector of the clear land of the sampling, the last-write-wins of the plane, the
//! pixel lists of the clouds, the labels).
//!
//! Deviations from MATLAB, all in the inputs of the comparison and none in the logic:
//!
//! * `rng(1); randperm(n)` ([`stratied_sample_handler`], `stratiedSampleHanlder.m:75`; `MatchCloudShadow.m:237-238`)
//!   cannot be reproduced (MATLAB's generator is not Octave's). Both sides use the same deterministic draw,
//!   [`fmask_randperm`]: a 64-bit LCG restarted at every `rng(1)` and a partial Fisher-Yates shuffle (the Octave
//!   shims `randperm.m` and `rng.m` of `benchmarks/vs_originals/fmask4/shims_shadow` are the same algorithm), so the
//!   subset of the samples and of the big clouds is NOT the one MATLAB's `randperm` gives; like the sun of the
//!   atmospheric correction in phase 23g, this is a deviation of the harness, not of the method.
//! * the view angles `VZA` and `VAA` are per-pixel arrays on the 20 m grid in hundredths of a degree, as
//!   `single(VZA(...))/100` of `MatchCloudShadow.m:277-278` implies; in the original they come from another grid
//!   (`angles_view`), which is a decision of the caller.
//! * `slope`, `aspect` and `dem` are ready inputs (single). `mean` of a single vector is Octave's (the sum in double
//!   rounded to single, which `mean(x)` of GNU Octave 11 gives bit for bit; MATLAB does not specify the order).
//! * the colon operator (`a:s:b`, `for i_dem=dem_b:ele_strata:dem_t`, `for base_h=Min:step:Max`) is Octave's
//!   (`Range.cc`: the number of elements with the tolerance of 3 eps and the last element clipped to the limit).
//! * `knnsearch(..., 'distance', 'cityblock')` has no tie rule in MATLAB's documentation; Octave's with up to 50 points
//!   is a stable sort of the distances, the smaller index first, which is what [`knn_cityblock`] does (for more points
//!   Octave's kd-tree may order the ties otherwise).

use super::mlab;
use rayon::prelude::*;

/// `pi / 180`, as `deg2rad` of MATLAB and Octave multiplies (`deg * (pi / 180)`).
pub const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// The single-precision `cos`, `sin` and `tan` of the platform's C library (`f32::cos` calls it): what GNU Octave's
/// `cos (single)` is on Windows bit for bit (2,000,000 values tested), and not the double result rounded to single
/// (which differs from it in 0.1 % of the arguments). MATLAB's own differ in the last bit in some arguments.
#[inline]
fn cosf(x: f32) -> f32 {
    x.cos()
}
#[inline]
fn sinf(x: f32) -> f32 {
    x.sin()
}
#[inline]
fn tanf(x: f32) -> f32 {
    x.tan()
}

// ---------------------------------------------------------------------------------------------------------------
// the random draw that replaces rng(1); randperm
// ---------------------------------------------------------------------------------------------------------------

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// The first `k` elements (0-based) of a permutation of `0..n`, after `rng(1)`: a 64-bit LCG (`s = s * 6364136223846793005
/// + 1442695040888963407 mod 2^64`, starting at `s = 1`) and a partial Fisher-Yates shuffle in which step `i` swaps
/// position `i` with `i + (s >> 32) mod (n - i)`, `s` being the state after `i + 1` steps. The first `k` elements
/// of the full shuffle are the same as these of the partial one, so `fmask_randperm(n, k)` is the first `k` of
/// `fmask_randperm(n, n)`. It is a deviation from MATLAB (see the module documentation).
pub fn fmask_randperm(n: usize, k: usize) -> Vec<usize> {
    assert!(n < (1usize << 32), "fmask_randperm: n < 2^32");
    let k = k.min(n);
    let mut p: Vec<usize> = (0..n).collect();
    let mut s: u64 = 1;
    for i in 0..k {
        s = s.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        let r = (s >> 32) as usize;
        let j = i + r % (n - i);
        p.swap(i, j);
    }
    p.truncate(k);
    p
}

// ---------------------------------------------------------------------------------------------------------------
// the colon operator
// ---------------------------------------------------------------------------------------------------------------

macro_rules! colon_impl {
    ($name:ident, $t:ty, $doc:expr) => {
        #[doc = $doc]
        pub fn $name(base: $t, inc: $t, limit: $t) -> Vec<$t> {
            let eps = <$t>::EPSILON;
            let ct: $t = 3.0 * eps;
            let xtfloor = |x: $t| -> $t {
                let q: $t = 1.0;
                let rmax = q / (2.0 - ct);
                let mut t1 = 1.0 + x.floor();
                t1 = (ct / q) * (if t1 < 0.0 { -t1 } else { t1 });
                t1 = if rmax < t1 { rmax } else { t1 };
                t1 = if ct > t1 { ct } else { t1 };
                t1 = (x + t1).floor();
                if x <= 0.0 || (t1 - x) < rmax {
                    t1
                } else {
                    t1 - 1.0
                }
            };
            let xteq = |u: $t, v: $t| -> bool {
                let (tu, tv) = (u.abs(), v.abs());
                (u - v).abs() < (if tu > tv { tu } else { tv }) * ct
            };
            if !base.is_finite() || !inc.is_finite() || limit.is_nan() || limit.is_infinite() {
                return Vec::new();
            }
            let n: i64 = if inc == 0.0 || (limit > base && inc < 0.0) || (limit < base && inc > 0.0) {
                0
            } else {
                let tmp = xtfloor((limit - base + inc) / inc);
                let mut n = if tmp > 0.0 { tmp as i64 } else { 0 };
                if !xteq(base + ((n - 1) as $t) * inc, limit) {
                    if xteq(base + ((n - 2) as $t) * inc, limit) {
                        n -= 1;
                    } else if xteq(base + (n as $t) * inc, limit) {
                        n += 1;
                    }
                }
                n
            };
            if n <= 0 {
                return Vec::new();
            }
            let mut fin = base;
            if n > 1 {
                fin = base + ((n - 1) as $t) * inc;
                if (inc > 0.0 && fin >= limit) || (inc < 0.0 && fin <= limit) {
                    fin = limit;
                }
                if base == base.round() && inc == inc.round() && fin == fin.round() {
                    fin = fin.round();
                }
            }
            (0..n)
                .map(|i| if i == 0 { base } else if i < n - 1 { base + (i as $t) * inc } else { fin })
                .collect()
        }
    };
}

colon_impl!(colon_f64, f64, "`base:inc:limit` of doubles, the elements as GNU Octave's range makes them: `base + i * inc`, the last one clipped to `limit`, the count with the 3 eps tolerance of `Range.cc`.");
colon_impl!(colon_f32, f32, "`base:inc:limit` of singles (a single base, step or limit makes the whole range single), as [`colon_f64`].");

// ---------------------------------------------------------------------------------------------------------------
// polyfit (degree 1)
// ---------------------------------------------------------------------------------------------------------------

/// `polyfit(x, y, 1)`: `[p1, p2]` of `y = p1 x + p2` by the QR decomposition of the Vandermonde matrix `[x, 1]`
/// (Householder reflections, then `R \ (Q' y)`), as MATLAB and Octave do. The rounding of the LAPACK routine (the order
/// of its sums) is not reproduced: the result is the same to the last bits of a double. With fewer than two points
/// the answer is NaN.
pub fn polyfit1(x: &[f64], y: &[f64]) -> (f64, f64) {
    let m = x.len();
    assert_eq!(y.len(), m);
    if m < 2 {
        return (f64::NAN, f64::NAN);
    }
    let mut a1 = x.to_vec();
    let mut a2 = vec![1.0f64; m];
    let mut b = y.to_vec();
    // reflection of column `c` below row `r`: returns (beta, tau) and leaves v (v[r] = 1) in the column
    fn reflect(col: &mut [f64], r: usize) -> (f64, f64) {
        let alpha = col[r];
        let xnorm = col[r + 1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if xnorm == 0.0 {
            return (alpha, 0.0);
        }
        let beta = -alpha.signum() * alpha.hypot(xnorm);
        let tau = (beta - alpha) / beta;
        let sc = 1.0 / (alpha - beta);
        for v in col[r + 1..].iter_mut() {
            *v *= sc;
        }
        (beta, tau)
    }
    fn apply(v: &[f64], r: usize, tau: f64, target: &mut [f64]) {
        if tau == 0.0 {
            return;
        }
        let mut w = target[r];
        for i in r + 1..v.len() {
            w += v[i] * target[i];
        }
        w *= tau;
        target[r] -= w;
        for i in r + 1..v.len() {
            target[i] -= w * v[i];
        }
    }
    let (beta1, tau1) = reflect(&mut a1, 0);
    apply(&a1, 0, tau1, &mut a2);
    apply(&a1, 0, tau1, &mut b);
    let (beta2, tau2) = reflect(&mut a2, 1);
    apply(&a2, 1, tau2, &mut b);
    let (r11, r12, r22) = (beta1, a2[0], beta2);
    let p2 = b[1] / r22;
    let p1 = (b[0] - r12 * p2) / r11;
    (p1, p2)
}

// ---------------------------------------------------------------------------------------------------------------
// stratiedSampleHanlder.m
// ---------------------------------------------------------------------------------------------------------------

/// `stratiedSampleHanlder(data, dem_b, dem_t, dim, total_sample, ele_strata, 0, resolution)` (`distance = 0`: no
/// minimum distance among the samples, the `rangesearch` branch is not ported): the 0-based positions in `data` of the
/// samples, stratum by stratum. The strata are `dem_b:ele_strata:dem_t` (a single range, `data` being single), the
/// stratum of `i` is `data >= i & data < i + ele_strata` (single); every stratum with a pixel gives
/// `round(total_sample / number of strata with pixels)` pixels, drawn by [`fmask_randperm`] (`rng(1)` before each).
pub fn stratied_sample_handler(data: &[f32], dem_b: f32, dem_t: f32, total_sample: f64, ele_strata: f64) -> Vec<usize> {
    let strata = colon_f32(dem_b, ele_strata as f32, dem_t);
    let ele32 = ele_strata as f32;
    // the pixels of every stratum (`find`), in the order of the vector
    let ids: Vec<Vec<u32>> = strata
        .par_iter()
        .map(|&i_dem| {
            let hi = i_dem + ele32;
            data.iter().enumerate().filter(|(_, &v)| v >= i_dem && v < hi).map(|(p, _)| p as u32).collect()
        })
        .collect();
    let n_avail = ids.iter().filter(|v| !v.is_empty()).count();
    let num_each = mlab::round_half_away(total_sample / n_avail as f64);
    let mut out = Vec::new();
    for v in &ids {
        if v.is_empty() {
            continue;
        }
        let mut num = v.len();
        if (num as f64) > num_each {
            num = num_each as usize;
        }
        for q in fmask_randperm(v.len(), num) {
            out.push(v[q] as usize);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// DetectPotentialCloudShadow.m
// ---------------------------------------------------------------------------------------------------------------

/// What `getDataTopoCorrected` computed (the .m does not return it): the `c` of the SCS+C of the NIR and of the SWIR
/// (`c_fitted(1,2)/c_fitted(1,1)`), and whether the correction was applied (`~isnan(c)`).
#[derive(Clone, Debug)]
pub struct TopoInfo {
    pub c_nir: f64,
    pub c_swir: f64,
    pub applied_nir: bool,
    pub applied_swir: bool,
    pub n_samples: usize,
}

/// `getDataTopoCorrected` (SCS+C, stratified by `cos i` in 0.1 steps, 40,000 samples). `slope` and `aspect` are single
/// degrees. The `cos i` and the corrected bands are single as in MATLAB (the double scalars are converted to single
/// before they meet a single array).
fn topo_corrected(
    rows: usize,
    cols: usize,
    nir: &[f32],
    swir: &[f32],
    clear_land: &[u8],
    zen_deg: f64,
    azi_deg: f64,
    slope: &[f32],
    aspect: &[f32],
) -> (Vec<f32>, Vec<f32>, TopoInfo) {
    let n = rows * cols;
    let zen_rad = zen_deg * DEG2RAD;
    let zc = zen_rad.cos();
    let zs = zen_rad.sin();
    let (zc32, zs32) = (zc as f32, zs as f32);
    let k32 = DEG2RAD as f32;
    let azi32 = azi_deg as f32;
    let mut cs = vec![0f32; n]; // cos(deg2rad(slope))
    let mut cos_sita = vec![0f32; n];
    cs.par_iter_mut().zip(cos_sita.par_iter_mut()).enumerate().for_each(|(p, (c, s))| {
        let sl = slope[p] * k32;
        *c = cosf(sl);
        let sn = sinf(sl);
        let ca = cosf((azi32 - aspect[p]) * k32);
        *s = zc32 * *c + (zs32 * sn) * ca;
    });
    // cos_sita(index_exclude_cloud_water): the clear land in the order of the columns
    let mut lin: Vec<u32> = Vec::new();
    for x in 0..cols {
        for y in 0..rows {
            if clear_land[y * cols + x] != 0 {
                lin.push((y * cols + x) as u32);
            }
        }
    }
    let vec_cos: Vec<f32> = lin.iter().map(|&p| cos_sita[p as usize]).collect();
    let mut mn = f32::NAN;
    let mut mx = f32::NAN;
    for &v in &vec_cos {
        if v.is_nan() {
            continue;
        }
        if mn.is_nan() || v < mn {
            mn = v;
        }
        if mx.is_nan() || v > mx {
            mx = v;
        }
    }
    let ids = stratied_sample_handler(&vec_cos, mn, mx, 40000.0, 0.1);
    let xs: Vec<f64> = ids.iter().map(|&q| vec_cos[q] as f64).collect();
    let zc_only = zc32;
    let correct = |band: &[f32]| -> (Vec<f32>, f64, bool) {
        let ys: Vec<f64> = ids.iter().map(|&q| band[lin[q] as usize] as f64).collect();
        let (p1, p2) = polyfit1(&xs, &ys);
        let c = p2 / p1;
        if c.is_nan() {
            return (band.to_vec(), c, false);
        }
        let c32 = c as f32;
        let out: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|p| {
                let f1 = cs[p] * zc_only + c32;
                let f2 = cos_sita[p] + c32;
                (band[p] * f1) / f2
            })
            .collect();
        (out, c, true)
    };
    let (nir_c, c_nir, a_nir) = correct(nir);
    let (swir_c, c_swir, a_swir) = correct(swir);
    (nir_c, swir_c, TopoInfo { c_nir, c_swir, applied_nir: a_nir, applied_swir: a_swir, n_samples: ids.len() })
}

/// `min(a, b)` of MATLAB: a NaN is ignored.
#[inline]
fn matlab_min(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() || a < b {
        a
    } else {
        b
    }
}

/// `DetectPotentialCloudShadow(data_meta, nir, swir, clear_land, observation, slope, aspect)`: the potential cloud
/// shadow, 1 where the fill-in of the regional minima of both the NIR and the SWIR is above 200 (reflectance times
/// 10000), 255 where there is no observation. `nir` and `swir` are single (TOA times 10000), `clear_land` the clear
/// land pixels (0/1), `observation` the mask of the observed pixels. With `topo = Some((slope, aspect))` (single
/// degrees; MATLAB's `~isempty(slope_data)&&~isempty(aspect_data)`) the bands are first corrected with the SCS+C
/// method. Returns the mask and what the correction fitted.
pub fn detect_potential_cloud_shadow(
    rows: usize,
    cols: usize,
    nir: &[f32],
    swir: &[f32],
    clear_land: &[u8],
    observation: &[u8],
    topo: Option<(&[f32], &[f32])>,
    sun_zenith_deg: f64,
    sun_azimuth_deg: f64,
) -> (Vec<u8>, Option<TopoInfo>) {
    let n = rows * cols;
    assert!(nir.len() == n && swir.len() == n && clear_land.len() == n && observation.len() == n);
    let percent_low = 0.175f64;
    let (mut nir, mut swir, info) = match topo {
        Some((slope, aspect)) => {
            let (a, b, i) = topo_corrected(rows, cols, nir, swir, clear_land, sun_zenith_deg, sun_azimuth_deg, slope, aspect);
            (a, b, Some(i))
        }
        None => (nir.to_vec(), swir.to_vec(), None),
    };
    let pick = |band: &[f32]| -> f32 {
        let v: Vec<f32> = band.iter().zip(clear_land).filter(|(_, &c)| c != 0).map(|(&b, _)| b).collect();
        mlab::prctile(&v, 100.0 * percent_low)
    };
    let backg_b4 = pick(&nir);
    let backg_b5 = pick(&swir);
    let fill = |band: &mut Vec<f32>, backg: f32| {
        band.par_iter_mut().zip(observation.par_iter()).for_each(|(b, &o)| {
            if o == 0 || b.is_nan() {
                *b = backg;
            }
        });
        let filled = mlab::imfill_gray(band, rows, cols);
        filled.par_iter().zip(band.par_iter()).map(|(f, b)| f - b).collect::<Vec<f32>>()
    };
    let nir_dif = fill(&mut nir, backg_b4);
    let swir_dif = fill(&mut swir, backg_b5);
    let out: Vec<u8> = (0..n)
        .into_par_iter()
        .map(|p| {
            if observation[p] == 0 {
                255
            } else {
                (matlab_min(nir_dif[p], swir_dif[p]) > 200.0f32) as u8
            }
        })
        .collect();
    (out, info)
}

// ---------------------------------------------------------------------------------------------------------------
// ProjectDEM2Plane.m
// ---------------------------------------------------------------------------------------------------------------

/// The result of `ProjectDEM2Plane`: `dem_plane_r` and `dem_plane_c` (row and column, 1-based, of the pixel that
/// lands on each cell of the expanded plane, 0 where none does), row-major on `rows x cols` of the plane
/// (`dim + 2 * dim_expd`).
pub struct Plane {
    pub rows: usize,
    pub cols: usize,
    pub row: Vec<u16>,
    pub col: Vec<u16>,
}

/// `ProjectDEM2Plane(dim, pshadow, dem, dem_base, sun_elevation_rad, sun_tazi_rad, sun_azimuth_deg, dim_expd, resolu)`:
/// every pixel of the image is projected to the plane along the sun light by its height above `dem_base` (no DEM:
/// identity); the pixels are visited by columns and a cell hit twice keeps the last. A pixel whose projection is NaN
/// is dropped (the .m removes them from both lists; `sub2ind` of MATLAB errors on a NaN before that). The
/// `pshadow` argument is not used by the .m either. The sizes must fit in 16 bits.
pub fn project_dem_to_plane(
    rows: usize,
    cols: usize,
    dem: Option<&[f32]>,
    dem_base: f64,
    sun_elevation_rad: f64,
    sun_tazi_rad: f64,
    sun_azimuth_deg: f64,
    dim_expd: usize,
    resolution: f64,
) -> Plane {
    assert!(rows + 2 * dim_expd < 65536 && cols + 2 * dim_expd < 65536, "the plane must fit in 16 bits");
    let (pr, pc) = (rows + 2 * dim_expd, cols + 2 * dim_expd);
    let mut row = vec![0u16; pr * pc];
    let mut col = vec![0u16; pr * pc];
    let base32 = dem_base as f32;
    let den32 = (resolution * sun_elevation_rad.tan()) as f32;
    let (s32, c32) = (sun_tazi_rad.sin() as f32, sun_tazi_rad.cos() as f32);
    let west = sun_azimuth_deg < 180.0;
    let (drow, dcol) = (dim_expd as f64, dim_expd as f64);
    for x in 0..cols {
        for y in 0..rows {
            let (li, lj) = ((y + 1) as f64, (x + 1) as f64);
            let (mut pi, mut pj) = (li, lj);
            if let Some(d) = dem {
                let d_ij = (d[y * cols + x] - base32) / den32;
                let (di, dj) = if west { (0.0f32 - d_ij * s32, 0.0f32 - d_ij * c32) } else { (d_ij * s32, d_ij * c32) };
                pi = (li + di as f64).round();
                pj = (lj + dj as f64).round();
            }
            let mut ii = pi + drow;
            let mut jj = pj + dcol;
            if ii.is_nan() || jj.is_nan() {
                continue;
            }
            if ii < 1.0 {
                ii = 1.0;
            }
            if jj < 1.0 {
                jj = 1.0;
            }
            if ii > pr as f64 {
                ii = pr as f64;
            }
            if jj > pc as f64 {
                jj = pc as f64;
            }
            let p = (ii as usize - 1) * pc + (jj as usize - 1);
            row[p] = (y + 1) as u16;
            col[p] = (x + 1) as u16;
        }
    }
    Plane { rows: pr, cols: pc, row, col }
}

// ---------------------------------------------------------------------------------------------------------------
// getRealCloudPositionS2.m
// ---------------------------------------------------------------------------------------------------------------

/// `getRealCloudPositionS2(x, y, h, VZAxy, VAAxy, resolu)` for one pixel: `x` is the column and `y` the row (doubles
/// in MATLAB), `h` the height (a double is converted to single because `tan(VZAxy)` is single), `vzaxy` and `vaaxy`
/// single radians. Everything is single: `dist_move = h .* tan(VZAxy) ./ resolu`, `delt_x = dist_move(2) .*
/// cos(pi/2 - VAAxy)`, `delt_y = dist_move(1) .* -sin(pi/2 - VAAxy)`, `x + delt_x`, `y + delt_y`. Returns `(x_new, y_new)`.
pub fn get_real_cloud_position_s2(x: f64, y: f64, h: f32, vzaxy: f32, vaaxy: f32, resolution: f64) -> (f32, f32) {
    let dist_move = (h * tanf(vzaxy)) / resolution as f32;
    let a = (std::f64::consts::FRAC_PI_2 as f32) - vaaxy;
    let delt_x = dist_move * cosf(a);
    let delt_y = dist_move * (-sinf(a));
    ((x as f32) + delt_x, (y as f32) + delt_y)
}

// ---------------------------------------------------------------------------------------------------------------
// MatchCloudShadow.m
// ---------------------------------------------------------------------------------------------------------------

/// The inputs of `MatchCloudShadow` (Sentinel-2 route). The arrays are row-major `rows x cols` on the 20 m grid.
pub struct MatchInput<'a> {
    pub rows: usize,
    pub cols: usize,
    /// `mask`: 1 where the pixel is observed.
    pub mask: &'a [u8],
    /// `plcim`: the potential cloud (`> 0`).
    pub plcim: &'a [u8],
    /// `plsim`: the potential cloud shadow (`== 1`; 255 outside the observations does not count).
    pub plsim: &'a [u8],
    /// `isShadowater`.
    pub is_shadow_water: bool,
    /// `waterAll`.
    pub water_all: &'a [u8],
    /// `data_dem`, single (empty in MATLAB for no DEM).
    pub dem: Option<&'a [f32]>,
    pub sun_zenith_deg: f64,
    pub sun_azimuth_deg: f64,
    /// `data_meta.Resolution(1)`.
    pub resolution: f64,
    /// `ptm`, the number of clear-sky pixels (`sum_clr`).
    pub ptm: f64,
    /// `num_near`, 14 in `autoFmask.m`.
    pub num_near: usize,
    /// `angles_view.VZA` and `.VAA` on the 20 m grid, in hundredths of a degree (`single(VZA(...))/100`).
    pub vza: &'a [f32],
    pub vaa: &'a [f32],
}

/// One cloud that was given a shadow: the values the .m keeps inside (`record_base_h`, `record_thresh`,
/// `record_base_h_near`) at the moment of the match.
#[derive(Clone, Debug)]
pub struct MatchEvent {
    pub cloud_type_cur: usize,
    pub cloud_type: usize,
    pub record_base_h: f64,
    pub record_thresh: f64,
    pub record_base_h_near: f64,
}

/// The outputs of `MatchCloudShadow` (`similar_num`, `data_cloud_matched`, `data_shadow_matched`) and the events.
pub struct MatchOutput {
    /// The matching was skipped (`ptm <= 40000 || revised_ptm >= 0.90`): `similar_num` is `[-1]`.
    pub skipped: bool,
    pub num_clouds: usize,
    pub similar_num: Vec<f64>,
    pub cloud_matched: Vec<u8>,
    pub shadow_matched: Vec<u8>,
    pub events: Vec<MatchEvent>,
}

/// The condition of `autoFmask.m:218` that skips the shadow detection altogether (`sum_clr <= 40000`).
pub fn autofmask_skips_shadow(sum_clr: f64) -> bool {
    sum_clr <= 40000.0
}

/// The `k` nearest of `cents` to `q` by the city-block distance of `knnsearch(..., 'distance', 'cityblock')`: the stable
/// sort of the distances (the smaller index first among equals). Returns `(index, distance)`.
pub fn knn_cityblock(cents: &[(f64, f64)], q: (f64, f64), k: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(f64, usize)> = cents.iter().enumerate().map(|(i, c)| ((c.0 - q.0).abs() + (c.1 - q.1).abs(), i)).collect();
    v.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    v.truncate(k);
    v.into_iter().map(|(d, i)| (i, d)).collect()
}

/// The geometry and the classes of the loop of heights of one cloud (see the module documentation).
#[derive(Clone, Copy)]
struct Geo {
    dem: bool,
    tan_v: f32,
    cos_va: f32,
    nsin_va: f32,
    res32: f32,
    cos_t64: f64,
    sin_t64: f64,
    cos_t32: f32,
    sin_t32: f32,
    den64: f64,
    den32: f32,
}

/// The per-height constants of the projection of a cloud.
struct Coef {
    dx: f32,
    dy: f32,
    a32: f32,
    b32: f32,
    a64: f64,
    b64: f64,
}

impl Geo {
    /// `h` (the height, a single value in `f64` with a DEM) and `h_bias = h - base_heigh_cloud`.
    fn coef(&self, h: f64, base_heigh_cloud: f64) -> Coef {
        let (hb32, i32_, i64_) = if self.dem {
            let h32 = h as f32;
            (h32 - base_heigh_cloud as f32, h32 / self.den32, 0.0)
        } else {
            ((h - base_heigh_cloud) as f32, 0.0, h / self.den64)
        };
        let dm = (hb32 * self.tan_v) / self.res32;
        Coef {
            dx: dm * self.cos_va,
            dy: dm * self.nsin_va,
            a32: i32_ * self.cos_t32,
            b32: i32_ * self.sin_t32,
            a64: i64_ * self.cos_t64,
            b64: i64_ * self.sin_t64,
        }
    }

    /// `(X, Y)` = (column, row) on the plane (before the margin is added) of the pixel at column `ox`, row `oy`.
    #[inline]
    fn xy(&self, c: &Coef, ox: f32, oy: f32) -> (f64, f64) {
        let tx = ox + c.dx;
        let ty = oy + c.dy;
        if self.dem {
            ((tx - c.a32).round() as f64, (ty - c.b32).round() as f64)
        } else {
            ((tx as f64 - c.a64).round(), (ty as f64 - c.b64).round())
        }
    }
}

/// `MatchCloudShadow(mask, plcim, plsim, isShadowater, waterAll, data_dem, [], t_templ, t_temph, data_meta, ptm,
/// num_near, angles_view)` for Sentinel-2: matches every potential cloud with its potential shadow by moving it
/// along the sun light through the heights from 200 m to 12,000 m (above the cloud's surface) and keeping the
/// height of the best similarity; the shadow found is projected from all the pixels of the cloud. The labels of the
/// clouds are those of `bwlabeln(, 8)` (the order of the first pixel by columns).
pub fn match_cloud_shadow(inp: &MatchInput) -> MatchOutput {
    let (rows, cols) = (inp.rows, inp.cols);
    let n = rows * cols;
    assert!(inp.mask.len() == n && inp.plcim.len() == n && inp.plsim.len() == n && inp.water_all.len() == n);
    assert!(inp.vza.len() == n && inp.vaa.len() == n);
    let cloud_pot: Vec<u8> = (0..n).into_par_iter().map(|p| ((inp.plcim[p] > 0) && inp.mask[p] == 1) as u8).collect();
    let shadow_pot: Vec<u8> = inp.plsim.par_iter().map(|&v| (v == 1) as u8).collect();
    let n_cloud: usize = cloud_pot.iter().map(|&v| v as usize).sum();
    let n_mask: usize = inp.mask.iter().map(|&v| v as usize).sum();
    let revised_ptm = n_cloud as f64 / n_mask as f64;
    // MatchCloudShadow.m:48 (and :50-52, which sets the shadow where there is NO potential shadow)
    if inp.ptm <= 40000.0 || revised_ptm >= 0.90 {
        return MatchOutput {
            skipped: true,
            num_clouds: 0,
            similar_num: vec![-1.0],
            cloud_matched: cloud_pot,
            shadow_matched: shadow_pot.iter().map(|&v| (v == 0) as u8).collect(),
            events: Vec::new(),
        };
    }
    let max_similar = 0.95f64;
    let num_pix = 3.0f64;
    let max_match_num = 1_000_000usize;
    let t_similar = 0.425f64;
    let t_buffer = 0.90f64;
    let l_pt = 0.175f64;
    let h_pt = 1.0 - l_pt;
    let pct_h = 100.0 * h_pt;
    let dim_expd = 2000usize;
    let num_near = inp.num_near;
    let has_dem = inp.dem.is_some();

    let sun_elevation_rad = (90.0 - inp.sun_zenith_deg) * DEG2RAD;
    let sun_tazi_rad = (inp.sun_azimuth_deg - 90.0) * DEG2RAD;
    let sub_size = inp.resolution;
    let dem_base_heigh: f64 = match inp.dem {
        Some(d) => {
            let v: Vec<f32> = d.iter().zip(inp.mask).filter(|(_, &m)| m != 0).map(|(&d, _)| d).collect();
            mlab::prctile(&v, 0.001) as f64
        }
        None => 0.0,
    };
    let step_interval = 2.0 * sub_size * sun_elevation_rad.tan();
    let plane = project_dem_to_plane(rows, cols, inp.dem, dem_base_heigh, sun_elevation_rad, sun_tazi_rad, inp.sun_azimuth_deg, dim_expd, inp.resolution);

    let (seg, num) = mlab::bwlabel(&cloud_pot, rows, cols);
    let regs = mlab::regionprops(&seg, rows, cols, num);
    let mut matched_cents: Vec<(f64, f64)> = Vec::new();
    let mut heights_rec: Vec<f64> = Vec::new();
    let mut similar_num = vec![0f64; num];
    let mut record_base_h_num = 0usize;
    let num_revisited = if num > num_near { num_near } else { 0 };
    let num_all = num + num_revisited;
    let i_xy_min = 200.0 / (sub_size * sun_elevation_rad.tan());
    let mut shadow_lab = vec![0u32; n];
    let mut events = Vec::new();
    let geo_base = Geo {
        dem: has_dem,
        tan_v: 0.0,
        cos_va: 0.0,
        nsin_va: 0.0,
        res32: sub_size as f32,
        cos_t64: sun_tazi_rad.cos(),
        sin_t64: sun_tazi_rad.sin(),
        cos_t32: sun_tazi_rad.cos() as f32,
        sin_t32: sun_tazi_rad.sin() as f32,
        den64: sub_size * sun_elevation_rad.tan(),
        den32: (sub_size * sun_elevation_rad.tan()) as f32,
    };
    let (dimexp_r, dimexp_c) = (plane.rows, plane.cols);
    let mut obs: Vec<(u16, u16)> = Vec::new();

    for cloud_type_cur in 1..=num_all {
        let mut cloud_type = cloud_type_cur;
        if cloud_type > num && num_revisited < num {
            cloud_type = cloud_type_cur - num;
        }
        let reg = &regs[cloud_type - 1];
        let area = reg.area;
        let ox_all: Vec<f32> = reg.pixel_list.iter().map(|&(x, _)| x as f32).collect();
        let oy_all: Vec<f32> = reg.pixel_list.iter().map(|&(_, y)| y as f32).collect();
        let cid_all: Vec<usize> = reg.pixel_list.iter().map(|&(x, y)| (y as usize - 1) * cols + (x as usize - 1)).collect();

        // MatchCloudShadow.m:193: sqrt(area/2*pi), as it is
        let r_obj = (area as f64 / 2.0 * std::f64::consts::PI).sqrt();

        let (base_heigh_cloud, min_cl, max_cl);
        match inp.dem {
            None => {
                base_heigh_cloud = 0.0f64;
                min_cl = 200.00f64;
                max_cl = 12000.00f64;
            }
            Some(d) => {
                let v: Vec<f32> = cid_all.iter().map(|&p| d[p]).collect();
                let b = mlab::prctile(&v, pct_h) - dem_base_heigh as f32;
                base_heigh_cloud = b as f64;
                min_cl = (200.00f32 + b) as f64;
                max_cl = (12000.00f32 + b) as f64;
            }
        }

        // the pixels that are moved: all, or a sample of 1,000,000 of the big clouds (rng(1); randperm)
        let (sox, soy): (Vec<f32>, Vec<f32>) = if area > max_match_num {
            let sm = fmask_randperm(area, max_match_num);
            (sm.iter().map(|&q| ox_all[q]).collect(), sm.iter().map(|&q| oy_all[q]).collect())
        } else {
            (ox_all.clone(), oy_all.clone())
        };

        let center_cur = reg.centroid;
        // MatchCloudShadow.m:277-278: pi/180*mean(single(VZA(orin_cid_all))/100)
        let mean32 = |a: &[f32]| -> f32 {
            let s: f64 = cid_all.iter().map(|&p| (a[p] / 100.0f32) as f64).sum();
            (s / area as f64) as f32
        };
        let vzaxy = (std::f64::consts::PI / 180.0) as f32 * mean32(inp.vza);
        let vaaxy = (std::f64::consts::PI / 180.0) as f32 * mean32(inp.vaa);
        let a = (std::f64::consts::FRAC_PI_2 as f32) - vaaxy;
        let geo = Geo { tan_v: tanf(vzaxy), cos_va: cosf(a), nsin_va: -sinf(a), ..geo_base };

        // the height predicted by the neighbouring clouds (MatchCloudShadow.m:282-302)
        let mut record_base_h_near = 0.0f64;
        if record_base_h_num >= num_near {
            let nn = knn_cityblock(&matched_cents, center_cur, num_near);
            let sel: Vec<usize> = nn.iter().filter(|(_, d)| !(cloud_type_cur > num && *d == 0.0)).map(|(i, _)| *i).collect();
            let tmp: Vec<f64> = sel.iter().map(|&i| heights_rec[i]).collect();
            let std_of = |v: &[f64]| -> f64 {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
            };
            let (near, sd) = if has_dem {
                let t32: Vec<f32> = tmp.iter().map(|&v| v as f32).collect();
                (mlab::prctile(&t32, pct_h) as f64, std_of(&tmp) as f32 as f64)
            } else {
                (mlab::prctile_f64(&tmp, pct_h), std_of(&tmp))
            };
            record_base_h_near = near;
            if sd >= 1000.0 || record_base_h_near <= min_cl || record_base_h_near >= max_cl {
                record_base_h_near = 0.0;
            }
        }

        let mut record_thresh = 0.0f64;
        let mut record_base_h = 0.0f64;
        let mut dist_pre = 0.0f64;
        let mut dist_passed = false;
        let mut dist_first = true;
        let (cpc_i, cpc_j) = (center_cur.1, center_cur.0);
        let mut num_matched = 0usize;

        let heights: Vec<f64> = if has_dem {
            colon_f32(min_cl as f32, step_interval as f32, max_cl as f32).into_iter().map(|v| v as f64).collect()
        } else {
            colon_f64(min_cl, step_interval, max_cl)
        };
        let max_minus_step = if has_dem { (max_cl as f32 - step_interval as f32) as f64 } else { max_cl - step_interval };

        for &base_h in &heights {
            let c = geo.coef(base_h, base_heigh_cloud);
            // the cells of the plane hit by the moved cloud, and what the plane says is there
            obs.clear();
            for k in 0..sox.len() {
                let (x, y) = geo.xy(&c, sox[k], soy[k]);
                let ii = y + dim_expd as f64;
                let jj = x + dim_expd as f64;
                if ii > 0.0 && jj > 0.0 && ii <= dimexp_r as f64 && jj <= dimexp_c as f64 {
                    let p = (ii as usize - 1) * dimexp_c + (jj as usize - 1);
                    obs.push((plane.row[p], plane.col[p]));
                }
            }
            if !dist_passed {
                let (mut si, mut sj) = (0f64, 0f64);
                for &(ri, rj) in &obs {
                    si += ri as f64;
                    sj += rj as f64;
                }
                let cnt = obs.len() as f64;
                let (ctmp_i, ctmp_j) = (si / cnt, sj / cnt);
                let (dj, di) = (ctmp_j - cpc_j, ctmp_i - cpc_i);
                let dist_cur = (dj * dj + di * di).sqrt();
                if dist_first {
                    dist_pre = dist_cur;
                    dist_first = false;
                } else if dist_pre >= dist_cur || dist_cur < i_xy_min {
                    dist_pre = dist_cur;
                    record_thresh = 0.0;
                    continue;
                } else {
                    dist_passed = true;
                }
            }
            // the similarity of the matched shadow
            let (mut out_all, mut inside, mut ex_self, mut unsure, mut sure, mut water) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
            for &(ri, rj) in &obs {
                if ri < 1 || ri as usize > rows || rj < 1 || rj as usize > cols {
                    out_all += 1;
                    continue;
                }
                let id = (ri as usize - 1) * cols + (rj as usize - 1);
                inside += 1;
                let out_obs = inp.mask[id] == 0;
                let ex = seg[id] != cloud_type as i32;
                ex_self += ex as usize;
                unsure += (out_obs || (ex && cloud_pot[id] == 1)) as usize;
                sure += (ex && shadow_pot[id] == 1) as usize;
                water += (inp.water_all[id] != 0) as usize;
            }
            if !inp.is_shadow_water && water == inside {
                break;
            }
            let matched_all = sure as f64 + 0.5 * unsure as f64 + out_all as f64;
            let total_all = ex_self as f64 + out_all as f64;
            let thresh_match = matched_all / total_all;

            let mut iter_con = true;
            if num_matched > 0 && (record_base_h_near > 0.0 && base_h >= record_base_h_near) {
                iter_con = false;
            }
            if iter_con {
                if thresh_match >= t_buffer * record_thresh && base_h < max_minus_step && record_thresh < max_similar {
                    if thresh_match > record_thresh {
                        record_thresh = thresh_match;
                        record_base_h = base_h;
                    }
                    continue;
                } else if record_thresh >= t_similar {
                    num_matched += 1;
                    if base_h < record_base_h_near {
                        if thresh_match >= record_thresh || thresh_match >= max_similar {
                            record_thresh = thresh_match;
                            record_base_h = base_h;
                        }
                        continue;
                    }
                } else {
                    record_thresh = 0.0;
                    continue;
                }
            }
            if num_matched < 1 {
                break;
            }
            if r_obj > num_pix && cloud_type_cur <= num {
                matched_cents.push(center_cur);
                heights_rec.push(record_base_h);
                record_base_h_num += 1;
            }
            similar_num[cloud_type - 1] = record_thresh;
            events.push(MatchEvent { cloud_type_cur, cloud_type, record_base_h, record_thresh, record_base_h_near });

            // the shadow of all the pixels of the cloud at the height found
            let c = geo.coef(record_base_h, base_heigh_cloud);
            let mut sids: Vec<u32> = Vec::with_capacity(ox_all.len());
            for k in 0..ox_all.len() {
                let (x, y) = geo.xy(&c, ox_all[k], oy_all[k]);
                let ii = y + dim_expd as f64;
                let jj = x + dim_expd as f64;
                // MatchCloudShadow.m:548-549: strictly below the size of the plane (the search above used <=)
                if ii > 0.0 && jj > 0.0 && ii < dimexp_r as f64 && jj < dimexp_c as f64 {
                    let p = (ii as usize - 1) * dimexp_c + (jj as usize - 1);
                    let (ri, rj) = (plane.row[p], plane.col[p]);
                    if ri < 1 || ri as usize > rows || rj < 1 || rj as usize > cols {
                        continue;
                    }
                    sids.push(((ri as usize - 1) * cols + (rj as usize - 1)) as u32);
                }
            }
            if cloud_type_cur > num {
                for v in shadow_lab.iter_mut() {
                    if *v == cloud_type as u32 {
                        *v = 0;
                    }
                }
            }
            for s in sids {
                shadow_lab[s as usize] = cloud_type as u32;
            }
            break;
        }
    }
    let shadow_matched: Vec<u8> = (0..n).into_par_iter().map(|p| ((shadow_lab[p] > 0) && cloud_pot[p] != 1) as u8).collect();
    MatchOutput { skipped: false, num_clouds: num, similar_num, cloud_matched: cloud_pot, shadow_matched, events }
}

// ---------------------------------------------------------------------------------------------------------------
// BufferMasks.m
// ---------------------------------------------------------------------------------------------------------------

/// `BufferMasks(pcloud, cdpix, pshadow, csdpix, psnow, sdpix)`: each mask dilated by `strel('square', 2 * pix + 1)`
/// when its `pix > 0`, as it is otherwise. Returns `(cloud, shadow, snow)`.
pub fn buffer_masks(
    rows: usize,
    cols: usize,
    pcloud: &[u8],
    cdpix: usize,
    pshadow: &[u8],
    csdpix: usize,
    psnow: &[u8],
    sdpix: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let d = |m: &[u8], pix: usize| -> Vec<u8> {
        if pix > 0 {
            mlab::dilate_bin_square(m, rows, cols, 2 * pix + 1)
        } else {
            m.to_vec()
        }
    };
    (d(pcloud, cdpix), d(pshadow, csdpix), d(psnow, sdpix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colon_matches_octave() {
        // 1:0.1:2 has 11 elements, the last clipped to 2
        let v = colon_f64(1.0, 0.1, 2.0);
        assert_eq!(v.len(), 11);
        assert_eq!(v[10], 2.0);
        assert_eq!(v[3], 1.0 + 3.0 * 0.1);
        let s = colon_f32(0.3, 0.1, 1.7);
        assert_eq!(s.len(), 15);
        assert_eq!(s[14], 1.7f32);
        assert!(colon_f64(5.0, 1.0, 1.0).is_empty());
        assert_eq!(colon_f64(5.0, -2.0, 0.0), vec![5.0, 3.0, 1.0]);
        assert_eq!(colon_f64(200.0, 34.0, 12000.0).len(), 348);
    }

    #[test]
    fn randperm_is_a_deterministic_permutation() {
        let p = fmask_randperm(1000, 1000);
        let mut s = p.clone();
        s.sort();
        assert_eq!(s, (0..1000).collect::<Vec<_>>());
        assert_eq!(p, fmask_randperm(1000, 1000));
        assert_eq!(&p[..40], &fmask_randperm(1000, 40)[..]);
        assert_ne!(p, (0..1000).collect::<Vec<_>>());
        // the first draw: s = 1 * A + C, (s >> 32) % 1000
        let s1 = 1u64.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        assert_eq!(p[0], ((s1 >> 32) as usize) % 1000);
    }

    #[test]
    fn polyfit_recovers_a_line() {
        let x: Vec<f64> = (0..200).map(|i| i as f64 * 0.01 - 0.3).collect();
        let y: Vec<f64> = x.iter().map(|v| 3.5 * v - 0.25).collect();
        let (p1, p2) = polyfit1(&x, &y);
        assert!((p1 - 3.5).abs() < 1e-12 && (p2 + 0.25).abs() < 1e-12);
        let (q1, q2) = polyfit1(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]);
        assert!((q2 / q1).is_nan() || !(q2 / q1).is_finite());
    }

    #[test]
    fn stratified_sample_takes_equal_parts() {
        let data: Vec<f32> = (0..30000).map(|i| (i % 1000) as f32 / 1000.0).collect(); // 0..0.999
        let ids = stratied_sample_handler(&data, 0.0, 0.999, 4000.0, 0.1);
        // 10 strata of 3000 pixels, 400 each
        assert_eq!(ids.len(), 4000);
        let mut cnt = [0usize; 10];
        for &q in &ids {
            cnt[(data[q] * 10.0) as usize] += 1;
        }
        // (the strata edges are single sums of 0.1, so a value on an edge may fall in the neighbour)
        assert!(cnt.iter().all(|&c| (c as i64 - 400).abs() <= 8), "{cnt:?}");
        let mut u = ids.clone();
        u.sort();
        u.dedup();
        assert_eq!(u.len(), ids.len());
    }

    #[test]
    fn potential_shadow_finds_a_dark_pit() {
        let (rows, cols) = (40, 50);
        let mut nir = vec![3000f32; rows * cols];
        let mut swir = vec![2500f32; rows * cols];
        for y in 15..22 {
            for x in 20..30 {
                nir[y * cols + x] = 2000.0;
                swir[y * cols + x] = 1500.0;
            }
        }
        // a pit in the NIR only does not count
        for y in 5..9 {
            for x in 5..9 {
                nir[y * cols + x] = 1000.0;
            }
        }
        let clear = vec![1u8; rows * cols];
        let mut obs = vec![1u8; rows * cols];
        obs[0] = 0;
        let (m, info) = detect_potential_cloud_shadow(rows, cols, &nir, &swir, &clear, &obs, None, 40.0, 150.0);
        assert!(info.is_none());
        assert_eq!(m[0], 255);
        assert_eq!(m[18 * cols + 25], 1);
        assert_eq!(m[6 * cols + 6], 0);
        assert_eq!(m[2 * cols + 2], 0);
    }

    #[test]
    fn buffer_masks_dilates_by_square() {
        let (rows, cols) = (9, 9);
        let mut a = vec![0u8; 81];
        a[4 * 9 + 4] = 1;
        let z = vec![0u8; 81];
        let (c, s, n) = buffer_masks(rows, cols, &a, 1, &a, 0, &z, 2);
        assert_eq!(c.iter().map(|&v| v as usize).sum::<usize>(), 9);
        assert_eq!(s, a);
        assert_eq!(n, z);
    }

    #[test]
    fn plane_without_dem_is_the_identity_shifted() {
        let p = project_dem_to_plane(5, 6, None, 0.0, 0.7, 0.3, 120.0, 10, 20.0);
        assert_eq!((p.rows, p.cols), (25, 26));
        assert_eq!(p.row[(3 + 10) * 26 + (4 + 10)], 4);
        assert_eq!(p.col[(3 + 10) * 26 + (4 + 10)], 5);
        assert_eq!(p.row[0], 0);
    }

    #[test]
    fn matches_a_cloud_with_its_shadow() {
        let (rows, cols) = (160usize, 200usize);
        let n = rows * cols;
        let (zen, azi) = (40.0f64, 140.0f64);
        let elev = (90.0 - zen) * DEG2RAD;
        let tazi = (azi - 90.0) * DEG2RAD;
        let h = 3000.0f64;
        let i_xy = h / (20.0 * elev.tan());
        // the cloud is a disc; its shadow is the disc moved by (-i_xy cos, -i_xy sin)
        let (cy, cx, r) = (110.0f64, 120.0f64, 8.0f64);
        let mut cloud = vec![0u8; n];
        let mut shadow = vec![0u8; n];
        let (sy, sx) = ((cy - i_xy * tazi.sin()).round(), (cx - i_xy * tazi.cos()).round());
        for y in 0..rows {
            for x in 0..cols {
                let (fy, fx) = (y as f64 + 1.0, x as f64 + 1.0);
                if (fy - cy).powi(2) + (fx - cx).powi(2) <= r * r {
                    cloud[y * cols + x] = 1;
                }
                if (fy - sy).powi(2) + (fx - sx).powi(2) <= r * r {
                    shadow[y * cols + x] = 1;
                }
            }
        }
        assert!(shadow.iter().any(|&v| v == 1), "the shadow is inside the scene");
        let mask = vec![1u8; n];
        let water = vec![0u8; n];
        let vza = vec![0f32; n];
        let vaa = vec![9000f32; n];
        let out = match_cloud_shadow(&MatchInput {
            rows,
            cols,
            mask: &mask,
            plcim: &cloud,
            plsim: &shadow,
            is_shadow_water: false,
            water_all: &water,
            dem: None,
            sun_zenith_deg: zen,
            sun_azimuth_deg: azi,
            resolution: 20.0,
            ptm: 1.0e5,
            num_near: 14,
            vza: &vza,
            vaa: &vaa,
        });
        assert!(!out.skipped);
        assert_eq!(out.num_clouds, 1);
        assert_eq!(out.events.len(), 1);
        let tp = (0..n).filter(|&p| shadow[p] == 1 && out.shadow_matched[p] == 1).count();
        let area = shadow.iter().filter(|&&v| v == 1).count();
        assert!(tp as f64 > 0.6 * area as f64, "{tp} of {area}");
        assert!(out.similar_num[0] > 0.425);
        // a scene with almost all clouds skips the matching and marks the non-shadow as shadow (MatchCloudShadow.m:51)
        let out2 = match_cloud_shadow(&MatchInput { ptm: 100.0, rows, cols, mask: &mask, plcim: &cloud, plsim: &shadow, is_shadow_water: false, water_all: &water, dem: None, sun_zenith_deg: zen, sun_azimuth_deg: azi, resolution: 20.0, num_near: 14, vza: &vza, vaa: &vaa });
        assert!(out2.skipped && out2.similar_num == vec![-1.0]);
        assert_eq!(out2.shadow_matched[0], 1);
    }
}
