//! Sentinel-2 resolution merge: the six 20 m bands (red edge 1-3, narrow NIR B8A, SWIR1, SWIR2) predicted at 10 m
//! from the 10 m green, red and broad NIR (B08).
//!
//! Ported from FORCE 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring,
//! Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force),
//! src/modules/lower-level/resmerge-ll.c: `resolution_merge_1` (`regression`), and `resolution_merge_2`,
//! `improphe_resmerge` and `rescale_weight_resmerge` (`improphe`); the second is the ImproPhe weighted mean in a
//! spectral instead of a temporal setup. FORCE's third option (STARFM, the USDA's code) is not here.
//!
//! The 20 m bands are on the 10 m grid before either runs: FORCE reads them with GDAL's `RasterIO` into the 10 m
//! buffer, whose default resampling is the nearest neighbour, so each 20 m pixel is four equal 10 m pixels.
//!
//! `regression`: per pixel, a least squares fit `y = c0 + c1 green + c2 red + c3 bnir` over the 5 x 5 pixels
//! around it (the pixel in its own kernel; those that are off left out, clouds and shadows kept), evaluated at the
//! pixel. A kernel with fewer than 25 pixels is filled with rows `(1, 0, 0, 0)` and `y = 0`. The fit is GSL 2.7's
//! `gsl_multifit_linear` (the SVD of the column-balanced matrix, modified Golub-Reinsch, the singular values at or
//! below `epsilon * s_max` discarded), ported function by function to the operation: GSL's reference CBLAS plain
//! IEEE without fused multiply-adds, `hypot` correctly rounded as glibc's (`gsl_symmv::hypot`). The matrix is the
//! same for the six bands, so it is decomposed once per pixel, which gives the bits of six decompositions.
//!
//! `improphe`: the pixels of the disc of radius 2 (13 pixels) that are neither off, cloud nor shadow are the
//! neighbours when their mean absolute difference of green, red and bnir (reflectance) to the pixel is under one of
//! 0.0025, 0.005, 0.01, 0.02, 0.025; the cutoff used is the smallest with at least 3 neighbours (the last when
//! none); the prediction of each band is the mean of its values at the neighbours of that cutoff, weighted by
//! `1 / rescale_weight(S)` with `S` rescaled over the neighbours of the cutoff.
//!
//! Two modes (`force`). `force = true` is FORCE's arithmetic: the values are `short` (here floats that hold
//! integers), the predictions are truncated to integers, and `regression` runs the pixels in row-major order on
//! one thread with each prediction written back into the bands before the next pixel reads them, as FORCE does
//! with one OpenMP thread (with more it is racy). With `force` false the values are not truncated, `regression`
//! reads the original values at every neighbour and runs the rows in parallel.

// GSL's loops are ported index by index, in its order of operations.
#![allow(clippy::needless_range_loop)]

use crate::gsl_symmv::hypot;
use crate::improphe::rescale_weight;
use crate::parallel::{for_each_row, transpose};

/// The six predicted bands: red edge 1-3, NIR (B8A), SWIR1, SWIR2.
pub const NT: usize = 6;
const R: isize = 2;
const W: usize = 5;
const NW: usize = W * W;
const NV: usize = 4;

/// The scene on the 10 m grid, row-major.
pub struct Scene<'a> {
    pub nx: usize,
    pub ny: usize,
    pub green: &'a [f32],
    pub red: &'a [f32],
    pub bnir: &'a [f32],
    /// The six bands to predict, `(NT, ny * nx)`.
    pub targets: &'a [f32],
    /// No data (FORCE's `get_off`).
    pub off: &'a [bool],
    /// Cloud or shadow.
    pub bad: &'a [bool],
}

// ---------------------------------------------------------------------------------------------------------------
// GSL 2.7: the pieces of gsl_multifit_linear, for a matrix of 25 x 4

const GSL_DBL_MIN: f64 = 2.2250738585072014e-308;
const GSL_SQRT_DBL_MAX: f64 = 1.3407807929942596e+154;
const GSL_SQRT_DBL_MIN: f64 = 1.4916681462400413e-154;
const EPS: f64 = f64::EPSILON;

/// cblas_dnrm2: the scaled sum of squares of GSL's reference CBLAS.
fn nrm2(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return x[0].abs();
    }
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &v in x {
        if v != 0.0 {
            let ax = v.abs();
            if scale < ax {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// gsl_linalg_householder_transform: `v` becomes the Householder vector (`v[0]` the `beta`), returns `tau`.
fn householder_transform(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 1 {
        return 0.0;
    }
    let xnorm = nrm2(&v[1..]);
    if xnorm == 0.0 {
        return 0.0;
    }
    let alpha = v[0];
    let sign = if alpha >= 0.0 { 1.0 } else { -1.0 };
    let beta = -sign * hypot(alpha, xnorm);
    let tau = (beta - alpha) / beta;
    let s = alpha - beta;
    if s.abs() > GSL_DBL_MIN {
        let a = 1.0 / s;
        for x in &mut v[1..] {
            *x *= a;
        }
        v[0] = beta;
    } else {
        let a = EPS / s;
        for x in &mut v[1..] {
            *x *= a;
        }
        let b = 1.0 / EPS;
        for x in &mut v[1..] {
            *x *= b;
        }
        v[0] = beta;
    }
    tau
}

/// A view of a row-major matrix: `d[(r0 + i) * tda + c0 + j]`.
struct View {
    tda: usize,
    r0: usize,
    c0: usize,
    rows: usize,
    cols: usize,
}

impl View {
    #[inline]
    fn at(&self, i: usize, j: usize) -> usize {
        (self.r0 + i) * self.tda + self.c0 + j
    }
}

/// gsl_linalg_householder_hm (the non-BLAS branch, GSL is not built with USE_BLAS): `A := H A`, `v[0]` taken as 1.
fn householder_hm(tau: f64, v: &[f64], d: &mut [f64], m: &View) {
    if tau == 0.0 {
        return;
    }
    for j in 0..m.cols {
        let mut wj = d[m.at(0, j)];
        for i in 1..m.rows {
            wj += d[m.at(i, j)] * v[i];
        }
        let a0j = d[m.at(0, j)];
        d[m.at(0, j)] = a0j - tau * wj;
        for i in 1..m.rows {
            let aij = d[m.at(i, j)];
            d[m.at(i, j)] = aij - tau * v[i] * wj;
        }
    }
}

/// gsl_linalg_householder_hm1: `A` is built up from the identity with its first column as the vector.
fn householder_hm1(tau: f64, d: &mut [f64], m: &View) {
    if tau == 0.0 {
        d[m.at(0, 0)] = 1.0;
        for j in 1..m.cols {
            d[m.at(0, j)] = 0.0;
        }
        for i in 1..m.rows {
            d[m.at(i, 0)] = 0.0;
        }
        return;
    }
    for j in 1..m.cols {
        let mut wj = 0.0;
        for i in 1..m.rows {
            let vi = d[m.at(i, 0)];
            wj += d[m.at(i, j)] * vi;
        }
        d[m.at(0, j)] = -tau * wj;
        for i in 1..m.rows {
            let vi = d[m.at(i, 0)];
            let aij = d[m.at(i, j)];
            d[m.at(i, j)] = aij - tau * vi * wj;
        }
    }
    for i in 1..m.rows {
        let vi = d[m.at(i, 0)];
        d[m.at(i, 0)] = -tau * vi;
    }
    d[m.at(0, 0)] = 1.0 - tau;
}

/// gsl_linalg_householder_left: `A := A - tau v (A^T v)^T`, with GSL's `dgemv(Trans)` and `dger`.
fn householder_left(tau: f64, v: &[f64], d: &mut [f64], m: &View) {
    if tau == 0.0 {
        return;
    }
    let mut work = [0.0f64; NV];
    for j in 0..m.rows {
        let temp = 1.0 * v[j];
        if temp != 0.0 {
            for i in 0..m.cols {
                work[i] += temp * d[m.at(j, i)];
            }
        }
    }
    for i in 0..m.rows {
        let tmp = -tau * v[i];
        for j in 0..m.cols {
            d[m.at(i, j)] += work[j] * tmp;
        }
    }
}

/// gsl_linalg_householder_right: `A := A - tau (A v) v^T`, with `v[0]` taken as 1.
fn householder_right(tau: f64, v: &[f64], d: &mut [f64], m: &View) {
    if tau == 0.0 {
        return;
    }
    let mut work = [0.0f64; NV];
    for i in 0..m.rows {
        let mut temp = 0.0;
        for j in 0..m.cols {
            temp += v[j] * d[m.at(i, j)];
        }
        work[i] += 1.0 * temp;
    }
    for i in 0..m.rows {
        let tmp = -tau * work[i];
        for j in 0..m.cols {
            d[m.at(i, j)] += v[j] * tmp;
        }
    }
}

/// gsl_linalg_givens
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s1 = 1.0 / (1.0 + t * t).sqrt();
        (s1 * t, s1)
    } else {
        let t = -b / a;
        let c1 = 1.0 / (1.0 + t * t).sqrt();
        (c1, c1 * t)
    }
}

/// The square matrices of the SVD (4 x 4).
type Sq = [f64; NV * NV];

/// `M <= M G` on columns `p` and `q`: `(c Mip - s Miq, s Mip + c Miq)`.
fn rotate(m: &mut Sq, p: usize, q: usize, c: f64, s: f64) {
    for i in 0..NV {
        let a = m[i * NV + p];
        let b = m[i * NV + q];
        m[i * NV + p] = c * a - s * b;
        m[i * NV + q] = s * a + c * b;
    }
}

fn swap_columns(m: &mut Sq, p: usize, q: usize) {
    for i in 0..NV {
        m.swap(i * NV + p, i * NV + q);
    }
}

fn chop_small_elements(d: &[f64], f: &mut [f64]) {
    let n = d.len();
    let mut d_i = d[0];
    for i in 0..n - 1 {
        let f_i = f[i];
        let d_ip1 = d[i + 1];
        if f_i.abs() < EPS * (d_i.abs() + d_ip1.abs()) {
            f[i] = 0.0;
        }
        d_i = d_ip1;
    }
}

fn trailing_eigenvalue(d: &[f64], f: &[f64]) -> f64 {
    let n = d.len();
    let da = d[n - 2];
    let db = d[n - 1];
    let fa = if n > 2 { f[n - 3] } else { 0.0 };
    let fb = f[n - 2];
    let ta = da * da + fa * fa;
    let tb = db * db + fb * fb;
    let tab = da * fb;
    let dt = (ta - tb) / 2.0;
    let s = ta + tb;
    let da2 = da * da;
    let db2 = db * db;
    let fa2 = fa * fa;
    let fb2 = fb * fb;
    let p = (da2 * db2) + (fa2 * db2) + (fa2 * fb2);
    let dd = hypot(dt, tab);
    let r1 = s / 2.0 + dd;
    if dt >= 0.0 {
        if r1 > 0.0 {
            p / r1
        } else {
            0.0
        }
    } else {
        r1
    }
}

/// C's `frexp` exponent: `x = m 2^e`, `m` in [0.5, 1).
fn frexp_exp(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0 {
        // subnormal
        let y = x * 18446744073709551616.0; // 2^64
        return frexp_exp(y) - 64;
    }
    e - 1022
}

/// `ldexp(1.0, k)`
fn pow2(k: i32) -> f64 {
    if k > 1023 {
        f64::INFINITY
    } else if k >= -1022 {
        f64::from_bits(((1023 + k) as u64) << 52)
    } else if k >= -1074 {
        f64::from_bits(1u64 << (k + 1074))
    } else {
        0.0
    }
}

fn create_schur(d0: f64, f0: f64, d1: f64) -> (f64, f64) {
    let (mut d0, mut f0, mut d1) = (d0, f0, d1);
    let mut apq = 2.0 * d0 * f0;
    if d0 == 0.0 || f0 == 0.0 {
        return (1.0, 0.0);
    }
    if d0.abs() < GSL_SQRT_DBL_MIN
        || d0.abs() > GSL_SQRT_DBL_MAX
        || f0.abs() < GSL_SQRT_DBL_MIN
        || f0.abs() > GSL_SQRT_DBL_MAX
        || d1.abs() < GSL_SQRT_DBL_MIN
        || d1.abs() > GSL_SQRT_DBL_MAX
    {
        let (d0_exp, f0_exp) = (frexp_exp(d0), frexp_exp(f0));
        let scale = pow2(-(d0_exp + f0_exp) / 4);
        d0 *= scale;
        f0 *= scale;
        d1 *= scale;
        apq = 2.0 * d0 * f0;
    }
    if apq != 0.0 {
        let tau = (f0 * f0 + (d1 + d0) * (d1 - d0)) / apq;
        let t = if tau >= 0.0 {
            1.0 / (tau + hypot(1.0, tau))
        } else {
            -1.0 / (-tau + hypot(1.0, tau))
        };
        let c = 1.0 / hypot(1.0, t);
        (c, t * c)
    } else {
        (1.0, 0.0)
    }
}

/// The 2 x 2 block of columns `a`, `a + 1`.
fn svd2(d: &mut [f64], f: &mut [f64], u: &mut Sq, v: &mut Sq, a: usize) {
    let d0 = d[0];
    let f0 = f[0];
    let d1 = d[1];
    if d0 == 0.0 {
        let (c, s) = givens(f0, d1);
        d[0] = c * f0 - s * d1;
        f[0] = s * f0 + c * d1;
        d[1] = 0.0;
        rotate(u, a, a + 1, c, s);
        swap_columns(v, a, a + 1);
    } else if d1 == 0.0 {
        let (c, s) = givens(d0, f0);
        d[0] = d0 * c - f0 * s;
        f[0] = 0.0;
        rotate(v, a, a + 1, c, s);
    } else {
        let (c, s) = create_schur(d0, f0, d1);
        let mut a11 = c * d0 - s * f0;
        let mut a21 = -s * d1;
        let mut a12 = s * d0 + c * f0;
        let mut a22 = c * d1;
        rotate(v, a, a + 1, c, s);
        if hypot(a11, a21) < hypot(a12, a22) {
            std::mem::swap(&mut a11, &mut a12);
            std::mem::swap(&mut a21, &mut a22);
            swap_columns(v, a, a + 1);
        }
        let (c, s) = givens(a11, a21);
        d[0] = c * a11 - s * a21;
        f[0] = c * a12 - s * a22;
        d[1] = s * a12 + c * a22;
        rotate(u, a, a + 1, c, s);
    }
}

fn chase_out_intermediate_zero(d: &mut [f64], f: &mut [f64], u: &mut Sq, a: usize, k0: usize) {
    let n = d.len();
    let mut x = f[k0];
    let mut y = d[k0 + 1];
    for k in k0..n - 1 {
        let (c, s) = givens(y, -x);
        rotate(u, a + k0, a + k + 1, c, s);
        d[k + 1] = s * x + c * y;
        if k == k0 {
            f[k] = c * x - s * y;
        }
        if k < n - 2 {
            let z = f[k + 1];
            f[k + 1] = c * z;
            x = -s * z;
            y = d[k + 2];
        }
    }
}

fn chase_out_trailing_zero(d: &mut [f64], f: &mut [f64], v: &mut Sq, a: usize) {
    let n = d.len();
    let mut x = d[n - 2];
    let mut y = f[n - 2];
    for k in (0..n - 1).rev() {
        let (c, s) = givens(x, y);
        rotate(v, a + k, a + n - 1, c, s);
        d[k] = c * x - s * y;
        if k == n - 2 {
            f[k] = s * x + c * y;
        }
        if k > 0 {
            let z = f[k - 1];
            f[k - 1] = c * z;
            x = d[k - 1];
            y = s * z;
        }
    }
}

fn qrstep(d: &mut [f64], f: &mut [f64], u: &mut Sq, v: &mut Sq, a: usize) {
    let n = d.len();
    if n == 1 {
        return;
    }
    if n == 2 {
        svd2(d, f, u, v, a);
        return;
    }
    for i in 0..n - 1 {
        if d[i] == 0.0 {
            chase_out_intermediate_zero(d, f, u, a, i);
            return;
        }
    }
    if d[n - 1] == 0.0 {
        chase_out_trailing_zero(d, f, v, a);
        return;
    }
    let d0 = d[0];
    let f0 = f[0];
    let d1 = d[1];
    let mut y;
    let mut z;
    {
        let mu = trailing_eigenvalue(d, f);
        y = d0 * d0 - mu;
        z = d0 * f0;
    }
    let mut ak;
    let mut bk = 0.0f64;
    let mut zk;
    let mut ap = d0;
    let mut bp = f0;
    let mut aq = d1;
    for k in 0..n - 1 {
        let (c, s) = givens(y, z);
        rotate(v, a + k, a + k + 1, c, s);
        {
            let bk1 = c * bk - s * z;
            let ap1 = c * ap - s * bp;
            let bp1 = s * ap + c * bp;
            let zp1 = -s * aq;
            let aq1 = c * aq;
            if k > 0 {
                f[k - 1] = bk1;
            }
            ak = ap1;
            bk = bp1;
            zk = zp1;
            ap = aq1;
            bp = if k < n - 2 { f[k + 1] } else { 0.0 };
            y = ak;
            z = zk;
        }
        let (c, s) = givens(y, z);
        rotate(u, a + k, a + k + 1, c, s);
        {
            let ak1 = c * ak - s * zk;
            let bk1 = c * bk - s * ap;
            let zk1 = -s * bp;
            let ap1 = s * bk + c * ap;
            let bp1 = c * bp;
            d[k] = ak1;
            bk = bk1;
            zk = zk1;
            ap = ap1;
            bp = bp1;
            aq = if k < n - 2 { d[k + 2] } else { 0.0 };
            y = bk;
            z = zk;
        }
    }
    f[n - 2] = bk;
    d[n - 1] = ap;
}

/// gsl_linalg_bidiag_decomp of the 4 x 4 matrix `a` (`tau_u` 4, `tau_v` 3).
fn bidiag_decomp(a: &mut Sq, tau_u: &mut [f64; NV], tau_v: &mut [f64; NV - 1]) {
    let n = NV;
    for j in 0..n {
        // the column j, rows j..
        let len = n - j;
        let mut v = [0.0f64; NV];
        for i in 0..len {
            v[i] = a[(j + i) * n + j];
        }
        let tau_j = householder_transform(&mut v[..len]);
        for i in 0..len {
            a[(j + i) * n + j] = v[i];
        }
        if j + 1 < n {
            // GSL sets the first element to 1 for the product and restores it (it stays in `a`)
            v[0] = 1.0;
            let m = View {
                tda: n,
                r0: j,
                c0: j + 1,
                rows: n - j,
                cols: n - j - 1,
            };
            householder_left(tau_j, &v[..len], a, &m);
        }
        tau_u[j] = tau_j;
        if j + 1 < n {
            let len = n - j - 1;
            let mut v = [0.0f64; NV];
            for i in 0..len {
                v[i] = a[j * n + j + 1 + i];
            }
            let tau_j = householder_transform(&mut v[..len]);
            for i in 0..len {
                a[j * n + j + 1 + i] = v[i];
            }
            if j + 1 < n {
                v[0] = 1.0;
                let m = View {
                    tda: n,
                    r0: j + 1,
                    c0: j + 1,
                    rows: n - j - 1,
                    cols: n - j - 1,
                };
                householder_right(tau_j, &v[..len], a, &m);
            }
            tau_v[j] = tau_j;
        }
    }
}

/// gsl_linalg_bidiag_unpack2: `a` becomes U, `tau_u` the diagonal, `tau_v` the superdiagonal, `v` V.
fn bidiag_unpack2(a: &mut Sq, tau_u: &mut [f64; NV], tau_v: &mut [f64; NV - 1], v: &mut Sq) {
    let n = NV;
    *v = [0.0; NV * NV];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for i in (0..n - 1).rev() {
        let len = n - (i + 1);
        let mut h = [0.0f64; NV];
        for k in 0..len {
            h[k] = a[i * n + i + 1 + k];
        }
        let ti = tau_v[i];
        let m = View {
            tda: n,
            r0: i + 1,
            c0: i + 1,
            rows: len,
            cols: len,
        };
        householder_hm(ti, &h[..len], v, &m);
    }
    for i in 0..n - 1 {
        tau_v[i] = a[i * n + i + 1];
    }
    for j in (0..n).rev() {
        let tj = tau_u[j];
        let ajj = a[j * n + j];
        let m = View {
            tda: n,
            r0: j,
            c0: j,
            rows: n - j,
            cols: n - j,
        };
        tau_u[j] = ajj;
        householder_hm1(tj, a, &m);
    }
}

/// gsl_linalg_SV_decomp of the square matrix `a` (4 x 4): `a` becomes U; returns the singular values.
fn sv_decomp(a: &mut Sq, v: &mut Sq) -> [f64; NV] {
    let n = NV;
    let mut s = [0.0f64; NV];
    let mut f = [0.0f64; NV - 1];
    bidiag_decomp(a, &mut s, &mut f);
    bidiag_unpack2(a, &mut s, &mut f, v);
    chop_small_elements(&s, &mut f);
    let mut b = n - 1;
    let mut iter = 0usize;
    while b > 0 {
        let fbm1 = f[b - 1];
        if fbm1 == 0.0 || fbm1.is_nan() {
            b -= 1;
            continue;
        }
        let mut aa = b - 1;
        while aa > 0 {
            let fam1 = f[aa - 1];
            if fam1 == 0.0 || fam1.is_nan() {
                break;
            }
            aa -= 1;
        }
        iter += 1;
        if iter > 100 * n {
            // GSL: "SVD decomposition failed to converge" (its error handler aborts)
            break;
        }
        let n_block = b - aa + 1;
        let mut rescale = false;
        let mut scale = 1.0f64;
        let mut norm = 0.0f64;
        for i in 0..n_block {
            let x = s[aa + i].abs();
            if x > norm {
                norm = x;
            }
        }
        for i in 0..n_block - 1 {
            let x = f[aa + i].abs();
            if x > norm {
                norm = x;
            }
        }
        if norm > GSL_SQRT_DBL_MAX {
            scale = norm / GSL_SQRT_DBL_MAX;
            rescale = true;
        } else if norm < GSL_SQRT_DBL_MIN && norm > 0.0 {
            scale = norm / GSL_SQRT_DBL_MIN;
            rescale = true;
        }
        let sb = &mut s[aa..aa + n_block];
        let fb = &mut f[aa..aa + n_block - 1];
        if rescale {
            let r = 1.0 / scale;
            sb.iter_mut().for_each(|x| *x *= r);
            fb.iter_mut().for_each(|x| *x *= r);
        }
        qrstep(sb, fb, a, v, aa);
        chop_small_elements(sb, fb);
        if rescale {
            sb.iter_mut().for_each(|x| *x *= scale);
            fb.iter_mut().for_each(|x| *x *= scale);
        }
    }
    for j in 0..n {
        let sj = s[j];
        if sj < 0.0 {
            for i in 0..n {
                v[i * n + j] = -v[i * n + j];
            }
            s[j] = -sj;
        }
    }
    for i in 0..n {
        let mut s_max = s[i];
        let mut i_max = i;
        for j in i + 1..n {
            if s[j] > s_max {
                s_max = s[j];
                i_max = j;
            }
        }
        if i_max != i {
            s.swap(i, i_max);
            swap_columns(a, i, i_max);
            swap_columns(v, i, i_max);
        }
    }
    s
}

/// The SVD of the balanced 25 x 4 matrix, `gsl_multifit_linear`'s: `U` (25 x 4, in `a`), the singular values and `V`.
struct Fit {
    u: [f64; NW * NV],
    s: [f64; NV],
    q: Sq,
    d: [f64; NV],
}

/// gsl_multifit_linear's decomposition of the 25 x 4 matrix `x`: the columns balanced (gsl_linalg_balance_columns),
/// then gsl_linalg_SV_decomp_mod.
fn decompose(x: &[f64; NW * NV]) -> Fit {
    let mut a = *x;
    // gsl_linalg_balance_columns
    let mut dvec = [1.0f64; NV];
    for j in 0..NV {
        let mut s = 0.0f64;
        for i in 0..NW {
            s += a[i * NV + j].abs();
        }
        let mut f = 1.0f64;
        if s == 0.0 || !s.is_finite() {
            dvec[j] = f;
            continue;
        }
        while s > 1.0 {
            s /= 2.0;
            f *= 2.0;
        }
        while s < 0.5 {
            s *= 2.0;
            f /= 2.0;
        }
        dvec[j] = f;
        if f != 1.0 {
            let r = 1.0 / f;
            for i in 0..NW {
                a[i * NV + j] *= r;
            }
        }
    }

    // gsl_linalg_SV_decomp_mod (N = 4 is more than 1): A = L R, R upper triangular
    let mut tau = [0.0f64; NV];
    for i in 0..NV {
        let len = NW - i;
        let mut v = [0.0f64; NW];
        for k in 0..len {
            v[k] = a[(i + k) * NV + i];
        }
        let tau_i = householder_transform(&mut v[..len]);
        for k in 0..len {
            a[(i + k) * NV + i] = v[k];
        }
        if i + 1 < NV {
            let m = View {
                tda: NV,
                r0: i,
                c0: i + 1,
                rows: NW - i,
                cols: NV - (i + 1),
            };
            householder_hm(tau_i, &v[..len], &mut a, &m);
        }
        tau[i] = tau_i;
    }
    let mut xm: Sq = [0.0; NV * NV];
    for i in 0..NV {
        for j in i..NV {
            xm[i * NV + j] = a[i * NV + j];
        }
    }
    for j in (0..NV).rev() {
        let m = View {
            tda: NV,
            r0: j,
            c0: j,
            rows: NW - j,
            cols: NV - j,
        };
        householder_hm1(tau[j], &mut a, &m);
    }
    let mut q: Sq = [0.0; NV * NV];
    let s = sv_decomp(&mut xm, &mut q);
    // U = L X
    let mut u = [0.0f64; NW * NV];
    for i in 0..NW {
        let mut sum = [0.0f64; NV];
        for j in 0..NV {
            let lij = a[i * NV + j];
            if lij == 0.0 {
                continue;
            }
            for k in 0..NV {
                sum[k] += lij * xm[j * NV + k];
            }
        }
        u[i * NV..i * NV + NV].copy_from_slice(&sum);
    }
    Fit { u, s, q, d: dvec }
}

impl Fit {
    /// gsl_multifit_linear_tsvd's coefficients for `y` (tolerance `epsilon`) and gsl_multifit_linear_est at `x`.
    fn estimate(&self, y: &[f64; NW], x: &[f64; NV]) -> f64 {
        // xt = U^T y (dgemv, Trans)
        let mut xt = [0.0f64; NV];
        for j in 0..NW {
            let temp = 1.0 * y[j];
            if temp != 0.0 {
                for i in 0..NV {
                    xt[i] += temp * self.u[j * NV + i];
                }
            }
        }
        // QSI = Q S^-1 with the singular values at or below tol * s0 dropped
        let mut qsi = self.q;
        let s0 = self.s[0];
        for j in 0..NV {
            let sj = self.s[j];
            let alpha = if sj <= EPS * s0 { 0.0 } else { 1.0 / sj };
            for i in 0..NV {
                qsi[i * NV + j] *= alpha;
            }
        }
        // c = QSI xt (dgemv, NoTrans), then the balancing undone
        let mut c = [0.0f64; NV];
        for i in 0..NV {
            let mut temp = 0.0;
            for j in 0..NV {
                temp += xt[j] * qsi[i * NV + j];
            }
            c[i] += 1.0 * temp;
        }
        for i in 0..NV {
            c[i] /= self.d[i];
        }
        // est = x . c
        let mut r = 0.0;
        for i in 0..NV {
            r += x[i] * c[i];
        }
        r
    }
}

// ---------------------------------------------------------------------------------------------------------------
// method 1: regression

/// `(short)(v)` of C on x86: the int conversion truncated to 16 bits.
#[inline]
fn to_short(v: f64) -> f32 {
    (v as i32 as i16) as f32
}

/// The estimates (reflectance) of the six bands at pixel `(i, j)`, or None where the pixel is not predicted. `tgt`
/// is read at the neighbours.
fn regression_pixel(sc: &Scene, tgt: &[f32], denom: f64, i: usize, j: usize) -> Option<[f64; NT]> {
    let (nx, ny) = (sc.nx, sc.ny);
    let npix = nx * ny;
    let p = i * nx + j;
    if sc.off[p] || sc.bad[p] {
        return None;
    }
    let x = [
        1.0,
        f64::from(sc.green[p]) / denom,
        f64::from(sc.red[p]) / denom,
        f64::from(sc.bnir[p]) / denom,
    ];
    let mut a = [0.0f64; NW * NV];
    let mut y = [[0.0f64; NW]; NT];
    for k in 0..NW {
        a[k * NV] = 1.0;
    }
    let mut k = 0;
    for ii in -R..=R {
        for jj in -R..=R {
            let ni = i as isize + ii;
            let nj = j as isize + jj;
            if ni < 0 || ni >= ny as isize || nj < 0 || nj >= nx as isize {
                continue;
            }
            let np = ni as usize * nx + nj as usize;
            if sc.off[np] {
                continue;
            }
            a[k * NV + 1] = f64::from(sc.green[np]) / denom;
            a[k * NV + 2] = f64::from(sc.red[np]) / denom;
            a[k * NV + 3] = f64::from(sc.bnir[np]) / denom;
            for b in 0..NT {
                y[b][k] = f64::from(tgt[b * npix + np]) / denom;
            }
            k += 1;
        }
    }
    // the rest of the kernel: (1, 0, 0, 0) and y = 0 (the matrix was zeroed, the intercept column set)
    let fit = decompose(&a);
    let mut est = [0.0f64; NT];
    for b in 0..NT {
        est[b] = fit.estimate(&y[b], &x);
    }
    Some(est)
}

/// FORCE's `resolution_merge_1`: the six bands at 10 m by the regression on green, red and bnir over 5 x 5 pixels.
/// `denom` is what the values are divided by to be reflectance (10000 for FORCE's). Returns `(NT, ny * nx)`: the
/// predictions where a pixel is neither off, cloud nor shadow, the input elsewhere.
pub fn regression(sc: &Scene, denom: f64, force: bool, n_jobs: i32) -> Vec<f32> {
    let (nx, ny) = (sc.nx, sc.ny);
    let npix = nx * ny;
    let mut out = sc.targets.to_vec();
    if npix == 0 {
        return out;
    }
    let value = |est: f64| {
        if force {
            to_short(est * 10000.0)
        } else {
            (est * denom) as f32
        }
    };
    if force {
        // row-major, in place: the later pixels read the earlier predictions
        for i in 0..ny {
            for j in 0..nx {
                if let Some(est) = regression_pixel(sc, &out, denom, i, j) {
                    let p = i * nx + j;
                    for b in 0..NT {
                        out[b * npix + p] = value(est[b]);
                    }
                }
            }
        }
        return out;
    }
    let width = nx * NT;
    let mut px = vec![0.0f32; npix * NT];
    for_each_row(
        &mut px,
        width,
        n_jobs,
        || (),
        |_, i, row| {
            for j in 0..nx {
                let p = i * nx + j;
                match regression_pixel(sc, sc.targets, denom, i, j) {
                    Some(est) => {
                        for b in 0..NT {
                            row[j * NT + b] = value(est[b]);
                        }
                    }
                    None => {
                        for b in 0..NT {
                            row[j * NT + b] = sc.targets[b * npix + p];
                        }
                    }
                }
            }
        },
    );
    transpose(&px, npix, NT)
}

// ---------------------------------------------------------------------------------------------------------------
// method 2: ImproPhe

const NS: usize = 5;
/// FORCE's `float Sthr[5]`, compared with the double `S`.
const STHR: [f32; NS] = [0.0025, 0.005, 0.01, 0.02, 0.025];
const H: isize = 2;
const MINK: usize = 3;

/// The prediction of the six bands at pixel `(i, j)` (reflectance times `scale`, truncated as `short` with `force`),
/// or None where the pixel is not predicted.
fn improphe_pixel(sc: &Scene, denom: f64, force: bool, i: usize, j: usize) -> Option<[f32; NT]> {
    let (nx, ny) = (sc.nx, sc.ny);
    let npix = nx * ny;
    let p = i * nx + j;
    if sc.off[p] || sc.bad[p] {
        return None;
    }
    let mut sclass = [0usize; NW];
    let mut srecord = [0.0f64; NW];
    let mut crecord = [[0.0f64; NW]; NT];
    let mut sn = [0usize; NS];
    let mut srange = [[i32::MAX as f64; NS], [i32::MIN as f64; NS]];
    let mut wn = 0usize;
    let m = [sc.green, sc.red, sc.bnir];
    for ii in -H..=H {
        for jj in -H..=H {
            let ni = i as isize + ii;
            let nj = j as isize + jj;
            if ni < 0 || nj < 0 || ni >= ny as isize || nj >= nx as isize {
                continue;
            }
            let np = nx * ni as usize + nj as usize;
            // not in the circular kernel
            if ii * ii + jj * jj > H * H {
                continue;
            }
            if sc.off[np] || sc.bad[np] {
                continue;
            }
            let mut sum = 0.0f64;
            for band in &m {
                sum += ((f64::from(band[p]) - f64::from(band[np])) / denom).abs();
            }
            let s = sum / 3.0;
            let mut sc_idx = None;
            for (k, t) in STHR.iter().enumerate() {
                if s < f64::from(*t) {
                    sc_idx = Some(k);
                    break;
                }
            }
            let Some(sc_idx) = sc_idx else { continue };
            sclass[wn] = sc_idx;
            srecord[wn] = s;
            if s > 0.0 {
                for k in sc_idx..NS {
                    if s > srange[1][k] {
                        srange[1][k] = s;
                    }
                    if s < srange[0][k] {
                        srange[0][k] = s;
                    }
                }
            }
            for f in 0..NT {
                crecord[f][wn] = f64::from(sc.targets[f * npix + np]) / denom;
            }
            wn += 1;
            for k in sc_idx..NS {
                sn[k] += 1;
            }
        }
    }
    if wn == 0 {
        let mut o = [0.0f32; NT];
        for f in 0..NT {
            o[f] = sc.targets[f * npix + p];
        }
        return Some(o);
    }
    let mut cut = NS - 1;
    for (k, &n) in sn.iter().enumerate() {
        if n >= MINK {
            cut = k;
            break;
        }
    }
    let mut weightxdata = [0.0f64; NT];
    let mut weight = 0.0f64;
    for k in 0..wn {
        if sclass[k] > cut {
            continue;
        }
        let ss = rescale_weight(srecord[k], srange[0][cut], srange[1][cut]);
        let w = 1.0 / ss;
        for f in 0..NT {
            weightxdata[f] += w * crecord[f][k];
        }
        weight += w;
    }
    let mut o = [0.0f32; NT];
    for f in 0..NT {
        let v = weightxdata[f] / weight;
        o[f] = if force {
            to_short(v * 10000.0)
        } else {
            (v * denom) as f32
        };
    }
    Some(o)
}

/// FORCE's `resolution_merge_2`: the weighted mean of the neighbours that look like the pixel in green, red and
/// bnir. Returns `(NT, ny * nx)`, as `regression`.
pub fn improphe(sc: &Scene, denom: f64, force: bool, n_jobs: i32) -> Vec<f32> {
    let (nx, ny) = (sc.nx, sc.ny);
    let npix = nx * ny;
    if npix == 0 {
        return sc.targets.to_vec();
    }
    let mut px = vec![0.0f32; npix * NT];
    for_each_row(
        &mut px,
        nx * NT,
        n_jobs,
        || (),
        |_, i, row| {
            for j in 0..nx {
                let p = i * nx + j;
                match improphe_pixel(sc, denom, force, i, j) {
                    Some(o) => row[j * NT..j * NT + NT].copy_from_slice(&o),
                    None => {
                        for b in 0..NT {
                            row[j * NT + b] = sc.targets[b * npix + p];
                        }
                    }
                }
            }
        },
    );
    transpose(&px, npix, NT)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene whose target bands are exact linear functions of green, red and bnir.
    fn linear_scene(ny: usize, nx: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let npix = ny * nx;
        let mut g = vec![0.0f32; npix];
        let mut r = vec![0.0f32; npix];
        let mut n = vec![0.0f32; npix];
        let mut t = vec![0.0f32; NT * npix];
        let coef: [[f64; 4]; NT] = [
            [200.0, 0.3, 0.2, 0.1],
            [100.0, 0.1, 0.1, 0.6],
            [-50.0, 0.2, -0.1, 0.9],
            [10.0, 0.0, 0.0, 1.0],
            [300.0, 0.5, 0.2, -0.3],
            [0.0, 0.7, 0.3, 0.0],
        ];
        for i in 0..ny {
            for j in 0..nx {
                let p = i * nx + j;
                // smooth, non-collinear fields
                let gg = 800.0 + 300.0 * ((i as f64 * 0.37).sin() + (j as f64 * 0.23).cos());
                let rr = 600.0 + 250.0 * ((i as f64 * 0.11 + j as f64 * 0.29).sin());
                let nn = 2500.0 + 700.0 * ((i as f64 * 0.19).cos() * (j as f64 * 0.31).sin());
                g[p] = gg.round() as f32;
                r[p] = rr.round() as f32;
                n[p] = nn.round() as f32;
                for b in 0..NT {
                    let c = coef[b];
                    t[b * npix + p] =
                        (c[0] + c[1] * f64::from(g[p]) + c[2] * f64::from(r[p]) + c[3] * f64::from(n[p])) as f32;
                }
            }
        }
        (g, r, n, t)
    }

    #[test]
    fn svd_reconstructs_the_matrix() {
        let mut x = [0.0f64; NW * NV];
        for k in 0..NW {
            x[k * NV] = 1.0;
            x[k * NV + 1] = 0.1 + 0.03 * ((k * 7 % 11) as f64);
            x[k * NV + 2] = 0.05 + 0.02 * ((k * 5 % 13) as f64);
            x[k * NV + 3] = 0.3 + 0.01 * ((k * 3 % 17) as f64);
        }
        let fit = decompose(&x);
        // U S Q^T D == X (the columns were divided by D)
        for i in 0..NW {
            for j in 0..NV {
                let mut v = 0.0;
                for k in 0..NV {
                    v += fit.u[i * NV + k] * fit.s[k] * fit.q[j * NV + k];
                }
                v *= fit.d[j];
                assert!((v - x[i * NV + j]).abs() < 1e-12, "{i} {j}: {v} vs {}", x[i * NV + j]);
            }
        }
        assert!(fit.s.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn fit_recovers_a_linear_function() {
        let mut x = [0.0f64; NW * NV];
        let mut y = [0.0f64; NW];
        for k in 0..NW {
            x[k * NV] = 1.0;
            x[k * NV + 1] = 0.1 + 0.03 * ((k * 7 % 11) as f64);
            x[k * NV + 2] = 0.05 + 0.02 * ((k * 5 % 13) as f64);
            x[k * NV + 3] = 0.3 + 0.01 * ((k * 3 % 17) as f64);
            y[k] = 0.2 + 0.5 * x[k * NV + 1] - 0.3 * x[k * NV + 2] + 0.8 * x[k * NV + 3];
        }
        let fit = decompose(&x);
        let at = [1.0, 0.4, 0.2, 0.5];
        let est = fit.estimate(&y, &at);
        assert!((est - (0.2 + 0.5 * 0.4 - 0.3 * 0.2 + 0.8 * 0.5)).abs() < 1e-12, "{est}");
    }

    #[test]
    fn rank_deficient_fit_drops_the_null_direction() {
        // green = red in every row: the fit is the minimum norm one, still exact on the data's span
        let mut x = [0.0f64; NW * NV];
        let mut y = [0.0f64; NW];
        for k in 0..NW {
            let g = 0.1 + 0.02 * ((k * 7 % 11) as f64);
            x[k * NV] = 1.0;
            x[k * NV + 1] = g;
            x[k * NV + 2] = g;
            x[k * NV + 3] = 0.3 + 0.01 * ((k * 3 % 17) as f64);
            y[k] = 0.1 + 0.4 * g + 0.5 * x[k * NV + 3];
        }
        let fit = decompose(&x);
        let est = fit.estimate(&y, &[1.0, 0.3, 0.3, 0.4]);
        assert!((est - (0.1 + 0.4 * 0.3 + 0.5 * 0.4)).abs() < 1e-10, "{est}");
    }

    fn masks(n: usize) -> (Vec<bool>, Vec<bool>) {
        (vec![false; n], vec![false; n])
    }

    #[test]
    fn regression_recovers_a_linear_band() {
        let (ny, nx) = (14, 17);
        let (g, r, n, t) = linear_scene(ny, nx);
        let (off, bad) = masks(ny * nx);
        let sc = Scene {
            nx,
            ny,
            green: &g,
            red: &r,
            bnir: &n,
            targets: &t,
            off: &off,
            bad: &bad,
        };
        let out = regression(&sc, 10000.0, false, 1);
        assert_eq!(out.len(), t.len());
        // inside, where the kernel is full: at the border the rows (1, 0, 0, 0) with y = 0 pull the fit down
        for b in 0..NT {
            for i in 2..ny - 2 {
                for j in 2..nx - 2 {
                    let k = b * ny * nx + i * nx + j;
                    assert!((out[k] - t[k]).abs() < 0.02, "{} vs {}", out[k], t[k]);
                }
            }
        }
        // and the border differs from the line, as FORCE's does
        assert!((out[0] - t[0]).abs() > 0.01);
    }

    #[test]
    fn regression_force_truncates_and_feeds_back() {
        let (ny, nx) = (12, 12);
        let (g, r, n, mut t) = linear_scene(ny, nx);
        // integer targets with noise, so that the fit is not exact
        for (k, v) in t.iter_mut().enumerate() {
            *v = v.round() + ((k * 31 % 17) as f32) - 8.0;
        }
        let (off, bad) = masks(ny * nx);
        let sc = Scene {
            nx,
            ny,
            green: &g,
            red: &r,
            bnir: &n,
            targets: &t,
            off: &off,
            bad: &bad,
        };
        let forced = regression(&sc, 10000.0, true, 4);
        assert!(forced.iter().all(|v| *v == v.trunc()));
        let free = regression(&sc, 10000.0, false, 4);
        // the first pixel has no earlier prediction to read: its values agree up to the truncation
        for b in 0..NT {
            let (a, c) = (forced[b * ny * nx], free[b * ny * nx]);
            assert!((a - c).abs() < 1.0 + 1e-3, "{a} {c}");
        }
        // later pixels read the predictions: the two differ somewhere
        assert!(forced.iter().zip(&free).any(|(a, b)| (a - b).abs() > 1.0));
        // and the free mode does not depend on the threads
        assert_eq!(free, regression(&sc, 10000.0, false, 1));
    }

    #[test]
    fn regression_skips_off_cloud_and_shadow_centres() {
        let (ny, nx) = (12, 12);
        let (g, r, n, t) = linear_scene(ny, nx);
        let (mut off, mut bad) = masks(ny * nx);
        off[4 * nx + 4] = true;
        bad[2 * nx + 6] = true;
        let mut t2 = t.clone();
        t2[4 * nx + 4] = 12345.0;
        t2[2 * nx + 6] += 0.3;
        let sc = Scene {
            nx,
            ny,
            green: &g,
            red: &r,
            bnir: &n,
            targets: &t2,
            off: &off,
            bad: &bad,
        };
        let out = regression(&sc, 10000.0, false, 1);
        assert_eq!(out[4 * nx + 4], 12345.0);
        assert_eq!(out[2 * nx + 6], t2[2 * nx + 6]);
        // the cloudy pixel is a neighbour (its line is as good as the others'), the fit is still the line
        assert!(
            (out[4 * nx + 8] - t[4 * nx + 8]).abs() < 0.5,
            "{} {}",
            out[4 * nx + 8],
            t[4 * nx + 8]
        );
    }

    #[test]
    fn improphe_constant_scene_is_the_mean_of_the_disc() {
        let (ny, nx) = (7, 7);
        let npix = ny * nx;
        let g = vec![1000.0f32; npix];
        let r = vec![800.0f32; npix];
        let n = vec![3000.0f32; npix];
        let mut t = vec![0.0f32; NT * npix];
        for b in 0..NT {
            for p in 0..npix {
                t[b * npix + p] = 500.0 * (b + 1) as f32;
            }
        }
        let (off, bad) = masks(npix);
        let sc = Scene {
            nx,
            ny,
            green: &g,
            red: &r,
            bnir: &n,
            targets: &t,
            off: &off,
            bad: &bad,
        };
        let out = improphe(&sc, 10000.0, false, 2);
        for (a, b) in out.iter().zip(&t) {
            assert!((a - b).abs() < 1e-3, "{a} {b}");
        }
        let forced = improphe(&sc, 10000.0, true, 2);
        // truncation of 0.05 * 10000 = 500 in floating point may give 499
        for (a, b) in forced.iter().zip(&t) {
            assert!((a - b).abs() <= 1.0, "{a} {b}");
        }
    }

    #[test]
    fn improphe_ignores_dissimilar_and_cloudy_neighbours() {
        let (ny, nx) = (5, 5);
        let npix = ny * nx;
        let mut g = vec![1000.0f32; npix];
        let mut r = vec![800.0f32; npix];
        let mut n = vec![3000.0f32; npix];
        let mut t = vec![0.0f32; NT * npix];
        for b in 0..NT {
            for p in 0..npix {
                t[b * npix + p] = 1000.0;
            }
        }
        // a different land cover on the right of the centre, a cloud above it
        for i in 0..ny {
            for j in 3..nx {
                let p = i * nx + j;
                g[p] = 3000.0;
                r[p] = 2800.0;
                n[p] = 500.0;
                for b in 0..NT {
                    t[b * npix + p] = 9000.0;
                }
            }
        }
        let (off, mut bad) = masks(npix);
        bad[nx + 2] = true;
        t[nx + 2] = 7777.0;
        let sc = Scene {
            nx,
            ny,
            green: &g,
            red: &r,
            bnir: &n,
            targets: &t,
            off: &off,
            bad: &bad,
        };
        let out = improphe(&sc, 10000.0, false, 1);
        // the centre: only the 1000s count
        assert!((out[2 * nx + 2] - 1000.0).abs() < 1e-3, "{}", out[2 * nx + 2]);
        // the cloudy pixel keeps its value
        assert_eq!(out[nx + 2], 7777.0);
    }

    #[test]
    fn scale_does_not_matter() {
        let (ny, nx) = (10, 10);
        let (g, r, n, t) = linear_scene(ny, nx);
        let (off, bad) = masks(ny * nx);
        let sc = Scene {
            nx,
            ny,
            green: &g,
            red: &r,
            bnir: &n,
            targets: &t,
            off: &off,
            bad: &bad,
        };
        let a = regression(&sc, 10000.0, false, 1);
        let k = 1.0 / 10000.0;
        let g1: Vec<f32> = g.iter().map(|v| v * k as f32).collect();
        let r1: Vec<f32> = r.iter().map(|v| v * k as f32).collect();
        let n1: Vec<f32> = n.iter().map(|v| v * k as f32).collect();
        let t1: Vec<f32> = t.iter().map(|v| v * k as f32).collect();
        let sc1 = Scene {
            nx,
            ny,
            green: &g1,
            red: &r1,
            bnir: &n1,
            targets: &t1,
            off: &off,
            bad: &bad,
        };
        let b = regression(&sc1, 1.0, false, 1);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y * 10000.0).abs() < 0.05, "{x} {y}");
        }
    }
}
