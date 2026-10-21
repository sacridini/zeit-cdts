//! GDAL's resampling of a source point (gdalwarpkernel.cpp): the source cells
//! given, the filters and the bilinear, cubic, general and Lanczos kernels.

use super::raster::{cvt_i32, ifloor, NoDataTest, Raster};
use super::{BILINEAR, CUBIC, CUBICSPLINE, LANCZOS};
use std::f64::consts::PI;

/// The source cells given: a window of the whole source, and GDAL's unified
/// validity mask (a cell is valid when any band is not NoData): tested on the
/// value itself for one band, a mask computed beforehand for several.
#[derive(Clone, Copy)]
pub struct Src<'a, T> {
    pub data: &'a [T],
    pub plane: usize,
    pub h: i32,
    pub w: i32,
    pub r0: i32,
    pub c0: i32,
    pub wh: i32,
    pub ww: i32,
    /// GDAL has a validity mask.
    pub masked: bool,
    /// Several bands.
    pub valid: Option<&'a [u8]>,
    pub nodata: NoDataTest,
}

impl<T: Raster> Src<'_, T> {
    #[inline(always)]
    pub fn at(&self, r: i32, c: i32) -> usize {
        ((r - self.r0) as isize * self.ww as isize + (c - self.c0) as isize) as usize
    }
    #[inline(always)]
    pub fn held(&self, r: i32, c: i32) -> bool {
        r >= self.r0 && c >= self.c0 && r < self.r0 + self.wh && c < self.c0 + self.ww
    }
    #[inline(always)]
    pub fn valid_at(&self, k: usize) -> bool {
        !self.masked
            || match self.valid {
                Some(v) => v[k] != 0,
                None => !self.nodata.test(self.data[k]),
            }
    }
    /// valid_at(k), given v, the value of band `band` at k: without a mask of
    /// several bands, the band is the one tested (the first, or a band alone).
    #[inline(always)]
    pub fn valid_cell(&self, band: usize, k: usize, v: T) -> bool {
        !self.masked
            || match self.valid {
                Some(m) => m[k] != 0,
                None => {
                    debug_assert_eq!(band, 0);
                    !self.nodata.test(v)
                }
            }
    }
    /// valid_cell(band, k, v), k unchecked.
    ///
    /// # Safety
    /// k < plane (a cell `held` by the window).
    #[inline(always)]
    pub unsafe fn valid_cell_unchecked(&self, band: usize, k: usize, v: T) -> bool {
        !self.masked
            || match self.valid {
                Some(m) => {
                    debug_assert!(k < m.len());
                    // SAFETY: the mask has a value for every cell of the plane.
                    unsafe { *m.get_unchecked(k) != 0 }
                }
                None => {
                    debug_assert_eq!(band, 0);
                    !self.nodata.test(v)
                }
            }
    }
    /// GDAL's density of a cell: 1 when valid, else 0.
    #[inline(always)]
    pub fn ok(&self, r: i32, c: i32) -> bool {
        self.held(r, c) && self.valid_at(self.at(r, c))
    }
    #[inline(always)]
    pub fn val(&self, b: usize, r: i32, c: i32) -> f64 {
        self.data[b * self.plane + self.at(r, c)].to_f64()
    }
    #[inline(always)]
    pub fn raw(&self, b: usize, r: i32, c: i32) -> T {
        self.data[b * self.plane + self.at(r, c)]
    }
}

fn filter_radius(method: i32) -> i32 {
    match method {
        BILINEAR => 1,
        CUBIC => 2,
        CUBICSPLINE => 2,
        LANCZOS => 3,
        _ => 0,
    }
}

// GWKBilinear, GWKCubic (CubicKernel) and GWKBSpline.
#[inline(always)]
fn filter_bilinear(x: f64) -> f64 {
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        return 1.0 - abs_x;
    }
    0.0
}

#[inline(always)]
fn filter_cubic(x: f64) -> f64 {
    let abs_x = x.abs();
    if abs_x <= 1.0 {
        let x2 = x * x;
        return x2 * (1.5 * abs_x - 2.5) + 1.0;
    } else if abs_x <= 2.0 {
        let x2 = x * x;
        return x2 * (-0.5 * abs_x + 2.5) - 4.0 * abs_x + 2.0;
    }
    0.0
}

#[inline(always)]
fn filter_bspline(x: f64) -> f64 {
    let xp2 = x + 2.0;
    let xp1 = x + 1.0;
    let xm1 = x - 1.0;
    let xp2c = xp2 * xp2 * xp2;
    // (GDAL's nested conditional, unrolled; the same sums in the same order)
    if !(xp2 > 0.0) {
        return 0.0;
    }
    if !(xp1 > 0.0) {
        return 0.0 + xp2c;
    }
    let inner = if x > 0.0 {
        (if xm1 > 0.0 { -4.0 * xm1 * xm1 * xm1 } else { 0.0 }) + 6.0 * x * x * x
    } else {
        0.0
    };
    (inner + -4.0 * xp1 * xp1 * xp1) + xp2c
}

/// The filter of method M.
#[inline(always)]
fn filter<const M: i32>(x: f64) -> f64 {
    match M {
        BILINEAR => filter_bilinear(x),
        CUBIC => filter_cubic(x),
        _ => filter_bspline(x),
    }
}

pub struct Kernel {
    pub xscale: f64,
    pub yscale: f64,
    pub use4: bool,
    pub x_radius: i32,
    pub y_radius: i32,
    pub filt_init_x: i32,
    pub filt_init_y: i32,
    // GWKResampleCreateWrkStruct's Lanczos constants.
    cos_pi_xscale_over3: f64,
    sin_pi_xscale_over3: f64,
    cos_pi_xscale: f64,
    sin_pi_xscale: f64,
    cos_pi_yscale_over3: f64,
    sin_pi_yscale_over3: f64,
    cos_pi_yscale: f64,
    sin_pi_yscale: f64,
}

impl Kernel {
    pub fn new(method: i32, xscale: f64, yscale: f64) -> Kernel {
        let radius = filter_radius(method);
        let (x_filter, y_filter) = (radius as f64, radius as f64);
        let x_radius = if xscale < 1.0 {
            cvt_i32((x_filter / xscale).ceil())
        } else {
            cvt_i32(x_filter)
        };
        let y_radius = if yscale < 1.0 {
            cvt_i32((y_filter / yscale).ceil())
        } else {
            cvt_i32(y_filter)
        };
        let mut k = Kernel {
            xscale,
            yscale,
            use4: xscale >= 0.95 && yscale >= 0.95,
            x_radius,
            y_radius,
            filt_init_x: ((radius + 1) % 2) - x_radius,
            filt_init_y: ((radius + 1) % 2) - y_radius,
            cos_pi_xscale_over3: 0.0,
            sin_pi_xscale_over3: 0.0,
            cos_pi_xscale: 0.0,
            sin_pi_xscale: 0.0,
            cos_pi_yscale_over3: 0.0,
            sin_pi_yscale_over3: 0.0,
            cos_pi_yscale: 0.0,
            sin_pi_yscale: 0.0,
        };
        if method == LANCZOS {
            if xscale < 1.0 {
                k.cos_pi_xscale_over3 = (PI / 3.0 * xscale).cos();
                k.sin_pi_xscale_over3 = (1.0 - k.cos_pi_xscale_over3 * k.cos_pi_xscale_over3).sqrt();
                k.cos_pi_xscale = (4.0 * k.cos_pi_xscale_over3 * k.cos_pi_xscale_over3 - 3.0) * k.cos_pi_xscale_over3;
                k.sin_pi_xscale = (1.0 - k.cos_pi_xscale * k.cos_pi_xscale).sqrt();
            }
            if yscale < 1.0 {
                k.cos_pi_yscale_over3 = (PI / 3.0 * yscale).cos();
                k.sin_pi_yscale_over3 = (1.0 - k.cos_pi_yscale_over3 * k.cos_pi_yscale_over3).sqrt();
                k.cos_pi_yscale = (4.0 * k.cos_pi_yscale_over3 * k.cos_pi_yscale_over3 - 3.0) * k.cos_pi_yscale_over3;
                k.sin_pi_yscale = (1.0 - k.cos_pi_yscale * k.cos_pi_yscale).sqrt();
            }
        }
        k
    }
}

/// A thread's work space (GWKResampleWrkStruct).
pub struct Work {
    weights_x: Vec<f64>,
    weights_y: Vec<f64>,
    last_src_x: i32,
    last_src_y: i32,
    last_delta_x: f64,
    last_delta_y: f64,
    /// The sums of the Lanczos weights last computed: those of the next
    /// sample with the same source cell and offset.
    sum_x: f64,
    sum_y: f64,
}

impl Work {
    pub fn new(k: &Kernel) -> Work {
        Work {
            weights_x: vec![0.0; ((k.x_radius + 1) * 2) as usize],
            weights_y: vec![0.0; ((k.y_radius + 1) * 2) as usize],
            last_src_x: -10,
            last_src_y: -10,
            last_delta_x: -10.0,
            last_delta_y: -10.0,
            sum_x: 0.0,
            sum_y: 0.0,
        }
    }
}

/// GWKBilinearResample4Sample.
#[inline(always)]
pub fn bilinear4<T: Raster>(
    s: &Src<T>,
    band: usize,
    src_x: f64,
    src_y: f64,
    density: &mut f64,
    real: &mut f64,
) -> bool {
    let mut isx = ifloor(src_x - 0.5);
    let mut isy = ifloor(src_y - 0.5);
    let mut ratio_x = 1.5 - (src_x - isx as f64);
    let mut ratio_y = 1.5 - (src_y - isy as f64);
    if isx == -1 {
        isx = 0;
        ratio_x = 1.0;
    }
    if isy == -1 {
        isy = 0;
        ratio_y = 1.0;
    }
    let (mut acc_real, mut acc_density, mut acc_divisor) = (0.0, 0.0, 0.0);
    // The four cells in GDAL's order (upper left, upper right, lower left,
    // lower right; isx, isy >= 0 here), those in the source and valid.
    let m = [
        ratio_x * ratio_y,
        (1.0 - ratio_x) * ratio_y,
        ratio_x * (1.0 - ratio_y),
        (1.0 - ratio_x) * (1.0 - ratio_y),
    ];
    if isy + 1 < s.h && isx + 1 < s.w && s.held(isy, isx) && s.held(isy + 1, isx + 1) {
        let k00 = s.at(isy, isx);
        let k10 = k00 + s.ww as usize;
        let d = &s.data[band * s.plane..(band + 1) * s.plane];
        // (isy, isx) and (isy + 1, isx + 1) are in the window: the four cells
        // are in its plane, d.
        debug_assert!(k10 + 1 < d.len());
        // SAFETY: in the plane (above).
        let cells = unsafe {
            [
                (k00, *d.get_unchecked(k00)),
                (k00 + 1, *d.get_unchecked(k00 + 1)),
                (k10, *d.get_unchecked(k10)),
                (k10 + 1, *d.get_unchecked(k10 + 1)),
            ]
        };
        // Without branches on the cells' validity (unpredictable with scattered
        // NoData): an invalid cell adds +0, its value left out. The sums start at
        // +0 and the weights are >= +0, so they are never -0, and adding +0
        // leaves them as GDAL's, which skips the cell.
        for (q, &(k, v)) in cells.iter().enumerate() {
            // SAFETY: in the plane (above).
            let ok = unsafe { s.valid_cell_unchecked(band, k, v) };
            if T::SMALL {
                if !ok {
                    continue;
                }
                acc_divisor += m[q];
                acc_real += v.to_f64() * m[q];
                acc_density += 1.0 * m[q];
                continue;
            }
            let mq = if ok { m[q] } else { 0.0 };
            let vq = if ok { v.to_f64() } else { 0.0 };
            acc_divisor += mq;
            acc_real += vq * mq;
            acc_density += 1.0 * mq;
        }
    } else {
        for (q, &mq) in m.iter().enumerate() {
            let (r, c) = (isy + q as i32 / 2, isx + q as i32 % 2);
            if r >= s.h || c >= s.w || !s.ok(r, c) {
                continue;
            }
            acc_divisor += mq;
            acc_real += s.val(band, r, c) * mq;
            acc_density += 1.0 * mq;
        }
    }

    if acc_divisor == 1.0 {
        *real = acc_real;
        *density = acc_density;
        return false;
    } else if acc_divisor < 0.00001 {
        *real = 0.0;
        *density = 0.0;
        return false;
    }
    *real = acc_real / acc_divisor;
    *density = acc_density / acc_divisor;
    true
}

/// GWKCubicComputeWeights.
#[inline(always)]
fn cubic_weights(x: f64) -> [f64; 4] {
    let half_x = 0.5 * x;
    let three_x = 3.0 * x;
    let half_x2 = half_x * x;
    [
        half_x * (-1.0 + x * (2.0 - x)),
        1.0 + half_x2 * (-5.0 + three_x),
        half_x * (1.0 + x * (4.0 - three_x)),
        half_x2 * (-1.0 + x),
    ]
}

/// CONVOL4.
#[inline(always)]
fn convol4(v1: &[f64; 4], v2: &[f64; 4]) -> f64 {
    v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3]
}

/// GWKCubicResample4Sample.
pub fn cubic4<T: Raster>(s: &Src<T>, band: usize, src_x: f64, src_y: f64, density: &mut f64, real: &mut f64) -> bool {
    let isx = cvt_i32(src_x - 0.5);
    let isy = cvt_i32(src_y - 0.5);
    let delta_x = src_x - 0.5 - isx as f64;
    let delta_y = src_y - 0.5 - isy as f64;
    if isx - 1 < 0 || isx + 2 >= s.w || isy - 1 < 0 || isy + 2 >= s.h {
        return bilinear4(s, band, src_x, src_y, density, real);
    }
    let mut value_dens = [0.0; 4];
    let mut value_real = [0.0; 4];
    let coeffs_x = cubic_weights(delta_x);
    // Any of the 4 x 4 cells invalid: bilinear (GDAL checks them row by row,
    // the same outcome). Every row's density is the same sum of the weights.
    let row_density = convol4(&coeffs_x, &[1.0; 4]);
    if s.held(isy - 1, isx - 1) && s.held(isy + 2, isx + 2) {
        let mut k0 = [0usize; 4];
        for (i, k) in k0.iter_mut().enumerate() {
            *k = s.at(isy - 1 + i as i32, isx - 1);
        }
        if s.masked {
            for &k in &k0 {
                for q in 0..4 {
                    if !s.valid_at(k + q) {
                        return bilinear4(s, band, src_x, src_y, density, real);
                    }
                }
            }
        }
        let d = &s.data[band * s.plane..];
        for i in 0..4 {
            let p = &d[k0[i]..k0[i] + 4];
            let vals = [p[0].to_f64(), p[1].to_f64(), p[2].to_f64(), p[3].to_f64()];
            value_dens[i] = row_density;
            value_real[i] = convol4(&coeffs_x, &vals);
        }
    } else {
        for i in -1..3 {
            let mut vals = [0.0; 4];
            for (k, v) in vals.iter_mut().enumerate() {
                let c = isx - 1 + k as i32;
                if !s.ok(isy + i, c) {
                    return bilinear4(s, band, src_x, src_y, density, real);
                }
                *v = s.val(band, isy + i, c);
            }
            value_dens[(i + 1) as usize] = row_density;
            value_real[(i + 1) as usize] = convol4(&coeffs_x, &vals);
        }
    }
    let coeffs_y = cubic_weights(delta_y);
    *density = convol4(&coeffs_y, &value_dens);
    *real = convol4(&coeffs_y, &value_real);
    true
}

/// GWKResample (bilinear and cubic when downsampling, cubic spline).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn resample<T: Raster, const M: i32>(
    s: &Src<T>,
    k: &Kernel,
    wk: &mut Work,
    band: usize,
    src_x: f64,
    src_y: f64,
    density: &mut f64,
    real: &mut f64,
) -> bool {
    // A copy of its own: the stores to the weights would otherwise have the
    // loops read its fields again.
    let s = &{ *s };
    let masked = s.masked;
    let (mut acc_real, mut acc_density, mut acc_weight) = (0.0, 0.0, 0.0);
    let isx = ifloor(src_x - 0.5);
    let isy = ifloor(src_y - 0.5);
    let delta_x = src_x - 0.5 - isx as f64;
    let delta_y = src_y - 0.5 - isy as f64;
    let (xscale, yscale) = (k.xscale, k.yscale);

    let mut j = k.filt_init_y;
    let mut j_max = k.y_radius;
    if isy + j < 0 {
        j = -isy;
    }
    if isy + j_max >= s.h {
        j_max = s.h - isy - 1;
    }
    let mut i_min = k.filt_init_x;
    let mut i_max = k.x_radius;
    if isx + i_min < 0 {
        i_min = -isx;
    }
    if isx + i_max >= s.w {
        i_max = s.w - isx - 1;
    }

    let x_below1 = xscale < 1.0;
    let y_below1 = yscale < 1.0;
    let nx = (i_max - i_min + 1).max(0) as usize;
    let wx = &mut wk.weights_x[..nx];
    for (w, i) in wx.iter_mut().zip(i_min..) {
        *w = if x_below1 {
            filter::<M>((i as f64 - delta_x) * xscale)
        } else {
            filter::<M>(i as f64 - delta_x)
        };
    }
    let wx = &*wx;
    while j <= j_max {
        let r = isy + j;
        let weight1 = if y_below1 {
            filter::<M>((j as f64 - delta_y) * yscale)
        } else {
            filter::<M>(j as f64 - delta_y)
        };
        let (mut real_local, mut density_local, mut weight_local) = (0.0, 0.0, 0.0);
        for (&weight2, i) in wx.iter().zip(i_min..) {
            let c = isx + i;
            if masked && !s.ok(r, c) {
                continue;
            }
            real_local += s.val(band, r, c) * weight2;
            if masked {
                density_local += 1.0 * weight2;
            }
            weight_local += weight2;
        }
        acc_real += real_local * weight1;
        acc_density += density_local * weight1;
        acc_weight += weight_local * weight1;
        j += 1;
    }

    if acc_weight < 0.000001 || (masked && acc_density < 0.000001) {
        *density = 0.0;
        return false;
    }
    if acc_weight < 0.99999 || acc_weight > 1.00001 {
        *real = acc_real / acc_weight;
        *density = if masked { acc_density / acc_weight } else { 1.0 };
    } else {
        *real = acc_real;
        *density = if masked { acc_density } else { 1.0 };
    }
    true
}

const THREE_PI_PI: f64 = 3.0 * PI * PI;

/// The Lanczos weights of one axis from `lo` to `hi`, at `w[i - init]`
/// (GWKResampleOptimizedLanczos, the same for x and y).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn lanczos_weights(
    w: &mut [f64],
    init: i32,
    lo: i32,
    hi: i32,
    delta: f64,
    scale: f64,
    cos_pi_scale_over3: f64,
    sin_pi_scale_over3: f64,
    cos_pi_scale: f64,
    sin_pi_scale: f64,
) {
    let at = |i: i32| (i - init) as usize;
    if scale < 1.0 {
        let mut x = (lo as f64 - delta) * scale;
        let pi_x_over3 = PI / 3.0 * x;
        let mut cos_over3 = pi_x_over3.cos();
        let mut sin_over3 = pi_x_over3.sin();
        let mut sin = (3.0 - 4.0 * sin_over3 * sin_over3) * sin_over3;
        let mut cos = (4.0 * cos_over3 * cos_over3 - 3.0) * cos_over3;
        let w = &mut w[at(lo)..];
        w[0] = if x == 0.0 {
            1.0
        } else {
            THREE_PI_PI * sin * sin_over3 / (x * x)
        };
        let n = (hi - lo).max(0) as usize;
        for wi in &mut w[1..n + 1] {
            x += scale;
            let new_sin = sin * cos_pi_scale + cos * sin_pi_scale;
            let new_sin_over3 = sin_over3 * cos_pi_scale_over3 + cos_over3 * sin_pi_scale_over3;
            *wi = if x == 0.0 {
                1.0
            } else {
                THREE_PI_PI * new_sin * new_sin_over3 / (x * x)
            };
            let new_cos = cos * cos_pi_scale - sin * sin_pi_scale;
            let new_cos_over3 = cos_over3 * cos_pi_scale_over3 - sin_over3 * sin_pi_scale_over3;
            sin = new_sin;
            cos = new_cos;
            sin_over3 = new_sin_over3;
            cos_over3 = new_cos_over3;
        }
    } else {
        let sin_pi_delta_over3 = ((-PI / 3.0) * delta).sin();
        let sin2_pi_delta_over3 = sin_pi_delta_over3 * sin_pi_delta_over3;
        let cos_pi_delta_over3 = (1.0 - sin2_pi_delta_over3).sqrt();
        let sin_pi_delta = (3.0 - 4.0 * sin2_pi_delta_over3) * sin_pi_delta_over3;
        let inv_pi2_over3 = 3.0 / (PI * PI);
        let inv_pi2_over3_x_sin_pi_delta = inv_pi2_over3 * sin_pi_delta;
        let a = -0.5 * inv_pi2_over3_x_sin_pi_delta * sin_pi_delta_over3;
        let sin_pi_over3 = 0.8660254037844386;
        let b = sin_pi_over3 * inv_pi2_over3_x_sin_pi_delta * cos_pi_delta_over3;
        let cst = [inv_pi2_over3_x_sin_pi_delta * sin_pi_delta_over3, a - b, a + b];
        if hi >= lo {
            let mut k = ((lo + 3) % 3) as usize; // the constant of i: (i + 3) % 3
            for (wi, i) in w[at(lo)..at(hi) + 1].iter_mut().zip(lo..) {
                let x = i as f64 - delta;
                *wi = if x == 0.0 { 1.0 } else { cst[k] / (x * x) };
                k = if k == 2 { 0 } else { k + 1 };
            }
        }
    }
}

/// The sum of w[lo..=hi], in order (none when the range is empty).
#[inline(always)]
fn weight_sum(w: &[f64], lo: i32, hi: i32) -> f64 {
    let mut sum = 0.0;
    if hi >= lo {
        for &v in &w[lo as usize..hi as usize + 1] {
            sum += v;
        }
    }
    sum
}

/// GWKResampleOptimizedLanczos.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn lanczos<T: Raster>(
    s: &Src<T>,
    k: &Kernel,
    wk: &mut Work,
    band: usize,
    src_x: f64,
    src_y: f64,
    density: &mut f64,
    real: &mut f64,
) -> bool {
    // A copy of its own, as in resample.
    let s = &{ *s };
    let masked = s.masked;
    let (mut acc_real, mut acc_density, mut acc_weight) = (0.0, 0.0, 0.0);
    let isx = ifloor(src_x - 0.5);
    let isy = ifloor(src_y - 0.5);
    let delta_x = src_x - 0.5 - isx as f64;
    let delta_y = src_y - 0.5 - isy as f64;
    let (xscale, yscale) = (k.xscale, k.yscale);

    let mut j_min = k.filt_init_y;
    let mut j_max = k.y_radius;
    if isy + j_min < 0 {
        j_min = -isy;
    }
    if isy + j_max >= s.h {
        j_max = s.h - isy - 1;
    }
    let mut i_min = k.filt_init_x;
    let mut i_max = k.x_radius;
    if isx + i_min < 0 {
        i_min = -isx;
    }
    if isx + i_max >= s.w {
        i_max = s.w - isx - 1;
    }

    if xscale < 1.0 {
        while (i_min as f64 - delta_x) * xscale < -3.0 {
            i_min += 1;
        }
        while (i_max as f64 - delta_x) * xscale > 3.0 {
            i_max -= 1;
        }
    } else {
        while i_min as f64 - delta_x < -3.0 {
            i_min += 1;
        }
        while i_max as f64 - delta_x > 3.0 {
            i_max -= 1;
        }
    }
    if isx != wk.last_src_x || delta_x != wk.last_delta_x {
        lanczos_weights(
            &mut wk.weights_x,
            k.filt_init_x,
            i_min,
            i_max,
            delta_x,
            xscale,
            k.cos_pi_xscale_over3,
            k.sin_pi_xscale_over3,
            k.cos_pi_xscale,
            k.sin_pi_xscale,
        );
        wk.last_src_x = isx;
        wk.last_delta_x = delta_x;
        wk.sum_x = weight_sum(&wk.weights_x, i_min - k.filt_init_x, i_max - k.filt_init_x);
    }

    if yscale < 1.0 {
        while (j_min as f64 - delta_y) * yscale < -3.0 {
            j_min += 1;
        }
        while (j_max as f64 - delta_y) * yscale > 3.0 {
            j_max -= 1;
        }
    } else {
        while j_min as f64 - delta_y < -3.0 {
            j_min += 1;
        }
        while j_max as f64 - delta_y > 3.0 {
            j_max -= 1;
        }
    }
    if isy != wk.last_src_y || delta_y != wk.last_delta_y {
        lanczos_weights(
            &mut wk.weights_y,
            k.filt_init_y,
            j_min,
            j_max,
            delta_y,
            yscale,
            k.cos_pi_yscale_over3,
            k.sin_pi_yscale_over3,
            k.cos_pi_yscale,
            k.sin_pi_yscale,
        );
        wk.last_src_y = isy;
        wk.last_delta_y = delta_y;
        wk.sum_y = weight_sum(&wk.weights_y, j_min - k.filt_init_y, j_max - k.filt_init_y);
    }

    let (fx, fy) = (k.filt_init_x, k.filt_init_y);
    // (empty when a range is: nothing to sum)
    let wx = &wk.weights_x[(i_min - fx) as usize..(i_max - fx + 1).max(i_min - fx) as usize];
    let wy = &wk.weights_y[(j_min - fy) as usize..(j_max - fy + 1).max(j_min - fy) as usize];
    lanczos_sum(
        s,
        band,
        isx,
        isy,
        i_min,
        j_min,
        wx,
        wy,
        wk.sum_x * wk.sum_y,
        masked,
        &mut acc_real,
        &mut acc_density,
        &mut acc_weight,
        density,
        real,
    )
}

/// The sums of GWKResampleOptimizedLanczos over the weights wx (columns
/// isx + i_min..) and wy (rows isy + j_min..).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn lanczos_sum<T: Raster>(
    s: &Src<T>,
    band: usize,
    isx: i32,
    isy: i32,
    i_min: i32,
    j_min: i32,
    wx: &[f64],
    wy: &[f64],
    sums: f64,
    masked: bool,
    acc_real: &mut f64,
    acc_density: &mut f64,
    acc_weight: &mut f64,
    density: &mut f64,
    real: &mut f64,
) -> bool {
    let (nx, ny) = (wx.len() as i32, wy.len() as i32);
    if !masked {
        *acc_weight = sums; // the row's weights summed times the column's
    }

    if T::BYTE && !masked {
        // The Byte case without masks (with GDAL's SSE2 sums of 6 columns).
        if *acc_weight < 0.000001 {
            *density = 0.0;
            return false;
        }
        let c = isx + i_min;
        if nx == 6 {
            let (w0, w1, w2, w3, w4, w5) = (wx[0], wx[1], wx[2], wx[3], wx[4], wx[5]);
            let row_acc = |r: i32| {
                let p = &s.data[band * s.plane + s.at(r, c)..][..6];
                let p0 = p[0].to_f64() * w0;
                let p1 = p[1].to_f64() * w1;
                let p2 = p[2].to_f64() * w2;
                let p3 = p[3].to_f64() * w3;
                let row = (p0 + p2) + (p1 + p3);
                let row_end = p[4].to_f64() * w4 + p[5].to_f64() * w5;
                row + row_end
            };
            let mut j = 0;
            while j < ny - 1 {
                *acc_real += row_acc(isy + j_min + j) * wy[j as usize];
                *acc_real += row_acc(isy + j_min + j + 1) * wy[(j + 1) as usize];
                j += 2;
            }
            if j == ny - 1 {
                *acc_real += row_acc(isy + j_min + j) * wy[j as usize];
            }
        } else {
            for (j, &w1) in wy.iter().enumerate() {
                let r = isy + j_min + j as i32;
                let p = if nx > 0 {
                    &s.data[band * s.plane + s.at(r, c)..][..nx as usize]
                } else {
                    &s.data[..0]
                };
                let mut i = 0;
                let (mut row_acc1, mut row_acc2) = (0.0, 0.0);
                while i < nx - 1 {
                    row_acc1 += p[i as usize].to_f64() * wx[i as usize];
                    row_acc2 += p[i as usize + 1].to_f64() * wx[i as usize + 1];
                    i += 2;
                }
                if i == nx - 1 {
                    row_acc1 += p[i as usize].to_f64() * wx[i as usize];
                }
                *acc_real += (row_acc1 + row_acc2) * w1;
            }
        }
        if *acc_weight < 0.99999 || *acc_weight > 1.00001 {
            let inv_acc = 1.0 / *acc_weight;
            *real = *acc_real * inv_acc;
        } else {
            *real = *acc_real;
        }
        *density = 1.0;
        return true;
    }

    let mut count_valid = 0;
    for (j, &w1) in wy.iter().enumerate() {
        let r = isy + j_min + j as i32;
        if masked {
            let c = isx + i_min;
            if nx > 0 && s.held(r, c) && s.held(r, c + nx - 1) {
                // the row in the window: its cells in order
                let k0 = s.at(r, c);
                let p = &s.data[band * s.plane + k0..][..nx as usize];
                for (i, (&v, &wxi)) in p.iter().zip(wx).enumerate() {
                    if !s.valid_cell(band, k0 + i, v) {
                        continue;
                    }
                    count_valid += 1;
                    let w2 = w1 * wxi;
                    *acc_real += v.to_f64() * w2;
                    *acc_density += 1.0 * w2;
                    *acc_weight += w2;
                }
                continue;
            }
            for (i, &wxi) in wx.iter().enumerate() {
                let c = c + i as i32;
                if !s.ok(r, c) {
                    continue;
                }
                count_valid += 1;
                let w2 = w1 * wxi;
                *acc_real += s.val(band, r, c) * w2;
                *acc_density += 1.0 * w2;
                *acc_weight += w2;
            }
        } else {
            let c = isx + i_min;
            let p = if nx > 0 {
                &s.data[band * s.plane + s.at(r, c)..][..nx as usize]
            } else {
                &s.data[..0]
            };
            let mut row_acc_real = 0.0;
            for (v, &wxi) in p.iter().zip(wx) {
                row_acc_real += v.to_f64() * wxi;
            }
            *acc_real += row_acc_real * w1;
        }
    }

    if *acc_weight < 0.000001 || (masked && (*acc_density < 0.000001 || count_valid < ny * nx / 2)) {
        *density = 0.0;
        return false;
    }
    if *acc_weight < 0.99999 || *acc_weight > 1.00001 {
        let inv_acc = 1.0 / *acc_weight;
        *real = *acc_real * inv_acc;
        *density = if masked { *acc_density * inv_acc } else { 1.0 };
    } else {
        *real = *acc_real;
        *density = if masked { *acc_density } else { 1.0 };
    }
    true
}
