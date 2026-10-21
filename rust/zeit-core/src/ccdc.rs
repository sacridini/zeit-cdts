//! CCDC (Zhu & Woodcock 2014): a port of src/ccdc.cpp, itself a port of the
//! original MATLAB (GERSL/CCDC TrendSeasonalFit_v12_30Line.m with autoTSFit,
//! autoTSPred, autoTmask, robustfit_cor, update_cft and the bundled Fortran
//! GLMnet). Same operations in the same order, so the same bits as the C++:
//! every model fit is a lasso (lambda = 20) in single precision, like the
//! Fortran's `real` arrays, on MATLAB datenum time; Tmask's robust fit is the
//! one of robustfit.rs. Being derived from GLMnet (GPL-2.0-only), this file is
//! distributed under GPL version 2; see THIRD_PARTY_NOTICES.md.
//!
//! What the C++ computes again and again is computed once, to the same values:
//! the harmonic terms of every date (once per batch: the dates are the same for
//! every pixel), and the lasso's standardized design with its inner products
//! (once per fit of all the bands, which share the dates).

// The loops index observations and bands as the C++ does; the observation
// indices are 1-based like the original's (i, i_start, i_break, ...).
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::cmath::lgamma;
use crate::eigen::qr::{Mat, Order};
use crate::parallel;
use crate::robustfit::{median_of, robust_fit};

const W: f64 = 2.0 * std::f64::consts::PI / 365.25; // annual cycle, as in the original
const DATENUM_OFFSET: i32 = 366; // MATLAB datenum = Python ordinal + 366

// Fixed constants of TrendSeasonalFit_v12_30Line.m
const MIN_NUM_C: i32 = 4;
const MID_NUM_C: i32 = 6;
const MAX_NUM_C: i32 = 8;
const N_TIMES: i32 = 3;
const NUM_YRS: f64 = 365.25;
const T_CONST: f64 = 3.89;
const MINI_YRS: f64 = 1.0;
const T_SN: f64 = 0.75;
const T_CLR: f64 = 0.25;
const CLOUD_SCREEN_OFFSET: f64 = 400.0; // "median(green) + 400" in the insufficient-clear branch
const GLMNET_LAMBDA: f32 = 20.0;
const GLMNET_THRESH: f32 = 1e-4;

/// The parameters of CCDCParams.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Kept for API compatibility; the original's 3 x 4 is fixed.
    pub min_obs: i32,
    /// The original's `conse`.
    pub conseq_anom: i32,
    /// Change probability -> T_cg.
    pub chi2_prob_threshold: f64,
    /// Tmax_cg = chi2inv(1 - 1e-6).
    pub tmax_cg_prob_threshold: f64,
    /// Max number of coefficients: 4, 6 or 8.
    pub num_c: i32,
    /// B_detect (0-based); empty -> {1..5} with >= 6 bands, else all.
    pub detection_bands: Vec<i32>,
    /// Tmask/cloud-screen bands (0-based); empty -> {1, 4} (green, SWIR1).
    pub tmask_bands: Vec<i32>,
    /// Index of a brightness-temperature band (deg C x 100), -1 if none.
    pub thermal_band: i32,
    /// Optical range test (exclusive), surface reflectance x 10000.
    pub valid_min: f64,
    pub valid_max: f64,
    /// Thermal range test (exclusive).
    pub thermal_min: f64,
    pub thermal_max: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            min_obs: 12,
            conseq_anom: 6,
            chi2_prob_threshold: 0.99,
            tmax_cg_prob_threshold: 0.999999,
            num_c: 8,
            detection_bands: Vec::new(),
            tmask_bands: Vec::new(),
            thermal_band: -1,
            valid_min: 0.0,
            valid_max: 10000.0,
            thermal_min: -9320.0,
            thermal_max: 7070.0,
        }
    }
}

/// One time-series model of a pixel: one element of the original's rec_cg.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Segment {
    /// First observation of the model (input date convention).
    pub t_start: i32,
    /// Last observation of the model.
    pub t_end: i32,
    /// First observation of the change, 0 if none.
    pub t_break: i32,
    /// Per band.
    pub rmse: Vec<f64>,
    /// [band][8]: a0 c1 a1 b1 a2 b2 a3 b3 (time axis = date + 366).
    pub coefs: Vec<Vec<f64>>,
    /// Per band, 0 if no change.
    pub magnitude: Vec<f64>,
    /// 0..1
    pub change_prob: f64,
    /// The original's QA code + number of coefficients.
    pub category: i32,
    /// Clear observations used by the model.
    pub num_obs: i32,
}

/// An error of the C++'s: the invalid_argument of make_ctx and fit_ccdc, or
/// what a fit throws (std::length_error) or does not define.
type Fit<T> = Result<T, &'static str>;

/// std::max(a, b): `a < b ? b : a` (NaN in `a` is kept, unlike f64::max).
#[inline]
fn cmax<T: PartialOrd>(a: T, b: T) -> T {
    if a < b {
        b
    } else {
        a
    }
}

/// *std::min_element: the first of the smallest values.
fn min_element(v: &[f64]) -> f64 {
    let mut m = v[0];
    for &x in &v[1..] {
        if x < m {
            m = x;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Chi-square quantile (MATLAB chi2inv), accurate to ~1e-14.

fn gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = lgamma(a);
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..10000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        return sum * (-x + a * x.ln() - gln).exp();
    }
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..10000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-17 {
            break;
        }
    }
    1.0 - (-x + a * x.ln() - gln).exp() * h
}

fn chi2inv(p: f64, df: i32) -> f64 {
    let a = df as f64 / 2.0;
    let (mut lo, mut hi) = (0.0, 1.0);
    while gamma_p(a, hi / 2.0) < p {
        hi *= 2.0;
    }
    let mut i = 0;
    while i < 200 && hi - lo > 1e-14 * hi {
        let mid = 0.5 * (lo + hi);
        if gamma_p(a, mid / 2.0) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        i += 1;
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// autoTSPred: the model's terms at datenum t, and its value there

/// 1, t, cos(wt), sin(wt), cos(2wt), sin(2wt), cos(3wt), sin(3wt).
fn terms(t: f64) -> [f64; 8] {
    [
        1.0,
        t,
        (W * t).cos(),
        (W * t).sin(),
        (2.0 * W * t).cos(),
        (2.0 * W * t).sin(),
        (3.0 * W * t).cos(),
        (3.0 * W * t).sin(),
    ]
}

#[inline]
fn ts_pred(terms: &[f64; 8], c: &[f64; 8]) -> f64 {
    let mut s = 0.0;
    for k in 0..8 {
        s += terms[k] * c[k];
    }
    s
}

// ---------------------------------------------------------------------------
// GLMnet (5/17/08) elnetu -> standard -> elnet1, gaussian lasso (alpha = 1),
// one user lambda, standardize = true, unit weights and penalty factors.
// Single precision throughout, exactly like the original's `real` Fortran.
// The design part of `standard` (chkvars, the centred and scaled columns) and
// the inner products of the columns do not depend on y: they are computed once
// for all the bands fitted on the same observations.

/// At most 7 predictors: t and three harmonics.
const MAX_NI: usize = 7;

struct Lasso {
    no: usize,
    ni: usize,
    ju: [bool; MAX_NI],
    any_ju: bool,
    /// The standardized design, column-major no x ni.
    x: Vec<f32>,
    xm: [f32; MAX_NI],
    xs: [f32; MAX_NI],
    /// The weights w[i] = 1 / sw and their square roots v[i], equal for every i.
    w: f32,
    v: f32,
    /// The inner products of the columns, symmetric, computed when first needed.
    gram: [f32; MAX_NI * MAX_NI],
    known: [bool; MAX_NI * MAX_NI],
}

impl Lasso {
    /// The design of autoTSFit with `df` coefficients at the dates `rows` of
    /// `terms`: t, cos(wt), sin(wt), ... in single precision.
    fn new(terms: &[[f64; 8]], rows: &[usize], df: i32) -> Fit<Lasso> {
        if df < 1 {
            // std::vector of a negative size in the C++: std::length_error
            return Err("num_c must be at least 1");
        }
        let (no, ni) = (rows.len(), (df - 1) as usize);
        let mut x = vec![0f32; no * ni];
        for j in 0..ni {
            for (i, &r) in rows.iter().enumerate() {
                x[j * no + i] = terms[r][j + 1] as f32;
            }
        }

        // chkvars
        let mut ju = [false; MAX_NI];
        for j in 0..ni {
            let col = &x[j * no..(j + 1) * no];
            ju[j] = no > 0 && col[1..].iter().any(|&e| e != col[0]);
        }
        let any_ju = ju.iter().any(|&u| u);
        let (mut xm, mut xs) = ([0f32; MAX_NI], [1f32; MAX_NI]);
        let (mut w, mut v) = (1f32, 1f32);
        if any_ju {
            // standard
            let mut sw = 0f32;
            for _ in 0..no {
                sw += 1.0;
            }
            w = 1.0 / sw;
            v = w.sqrt();
            let mut xv = [0f32; MAX_NI];
            for j in 0..ni {
                if !ju[j] {
                    continue;
                }
                let col = &mut x[j * no..(j + 1) * no];
                let mut s = 0f32;
                for &e in col.iter() {
                    s += w * e;
                }
                xm[j] = s;
                for e in col.iter_mut() {
                    *e = v * (*e - xm[j]);
                }
                let mut s = 0f32;
                for &e in col.iter() {
                    s += e * e;
                }
                xv[j] = s;
            }
            for j in 0..ni {
                if !ju[j] {
                    continue;
                }
                xs[j] = xv[j].sqrt();
                for e in x[j * no..(j + 1) * no].iter_mut() {
                    *e /= xs[j];
                }
            }
        }
        Ok(Lasso {
            no,
            ni,
            ju,
            any_ju,
            x,
            xm,
            xs,
            w,
            v,
            gram: [0.0; MAX_NI * MAX_NI],
            known: [false; MAX_NI * MAX_NI],
        })
    }

    /// The sum over the observations of X(i, j) * X(i, k).
    fn gram(&mut self, j: usize, k: usize) -> f32 {
        let ni = self.ni;
        if !self.known[j * ni + k] {
            let no = self.no;
            let (xj, xk) = (&self.x[j * no..(j + 1) * no], &self.x[k * no..(k + 1) * no]);
            let mut s = 0f32;
            for i in 0..no {
                s += xj[i] * xk[i];
            }
            // the products, and so the sum, are the same either way round
            self.gram[j * ni + k] = s;
            self.gram[k * ni + j] = s;
            self.known[j * ni + k] = true;
            self.known[k * ni + j] = true;
        }
        self.gram[j * ni + k]
    }

    /// a0 and beta of y (the standardized in place).
    fn solve(&mut self, y: &mut [f32]) -> (f32, [f32; MAX_NI]) {
        let (no, ni) = (self.no, self.ni);
        let mut beta = [0f32; MAX_NI];
        if !self.any_ju {
            // jerr = 7777: the MATLAB wrapper then gets no fit at all
            let mut ym = 0f32;
            for &e in y.iter() {
                ym += (1.0 / no as f32) * e;
            }
            return (ym, beta);
        }

        // the y part of standard
        let ju = self.ju;
        let mut ym = 0f32;
        for &e in y.iter() {
            ym += self.w * e;
        }
        for e in y.iter_mut() {
            *e = self.v * (*e - ym);
        }
        let mut ys = 0f32;
        for &e in y.iter() {
            ys += e * e;
        }
        let ys = ys.sqrt();
        for e in y.iter_mut() {
            *e /= ys;
        }
        let mut g = [0f32; MAX_NI];
        for j in 0..ni {
            if !ju[j] {
                continue;
            }
            let col = &self.x[j * no..(j + 1) * no];
            let mut s = 0f32;
            for i in 0..no {
                s += y[i] * col[i];
            }
            g[j] = s;
        }
        let xv = [1f32; MAX_NI];
        let alm = GLMNET_LAMBDA / ys;
        let thr = GLMNET_THRESH;

        // elnet1 (flmin >= 1: the single given lambda; beta = alpha = 1). The
        // Fortran's rsq is left out: it changes nothing returned.
        let nx = ni;
        let (ab, dem) = (alm * 1.0, alm * 0.0);
        let mut a = [0f32; MAX_NI];
        let mut da = [0f32; MAX_NI];
        let mut c = [0f32; MAX_NI * MAX_NI]; // C(j, l) = c[l * ni + j]
        let mut mm = [0usize; MAX_NI];
        let mut ia = [0usize; MAX_NI];
        let (mut nin, mut iz, mut jz) = (0usize, 0, 1);
        loop {
            if !(iz * jz != 0) {
                let mut dlx = 0f32;
                let mut overflow = false;
                for k in 0..ni {
                    if !ju[k] {
                        continue;
                    }
                    let ak = a[k];
                    let u = g[k] + ak * xv[k];
                    let vv = u.abs() - 1.0 * ab;
                    a[k] = 0.0;
                    if vv > 0.0 {
                        a[k] = vv.copysign(u) / (xv[k] + 1.0 * dem);
                    }
                    if a[k] == ak {
                        continue;
                    }
                    if mm[k] == 0 {
                        nin += 1;
                        if nin > nx {
                            overflow = true;
                            break;
                        }
                        for j in 0..ni {
                            if !ju[j] {
                                continue;
                            }
                            if mm[j] != 0 {
                                c[(nin - 1) * ni + j] = c[(mm[j] - 1) * ni + k];
                                continue;
                            }
                            if j == k {
                                c[(nin - 1) * ni + j] = xv[j];
                                continue;
                            }
                            c[(nin - 1) * ni + j] = self.gram(j, k);
                        }
                        mm[k] = nin;
                        ia[nin - 1] = k;
                    }
                    let del = a[k] - ak;
                    dlx = cmax(del.abs() / xv[k].sqrt(), dlx);
                    let ck = (mm[k] - 1) * ni;
                    for j in 0..ni {
                        if ju[j] {
                            g[j] -= c[ck + j] * del;
                        }
                    }
                }
                if overflow || dlx < thr {
                    break;
                }
                if nin > nx {
                    break;
                }
            }
            iz = 1;
            for l in 0..nin {
                da[l] = a[ia[l]];
            }
            loop {
                let mut dlx = 0f32;
                for l in 0..nin {
                    let k = ia[l];
                    let ak = a[k];
                    let u = g[k] + ak * xv[k];
                    let vv = u.abs() - 1.0 * ab;
                    a[k] = 0.0;
                    if vv > 0.0 {
                        a[k] = vv.copysign(u) / (xv[k] + 1.0 * dem);
                    }
                    if a[k] == ak {
                        continue;
                    }
                    let del = a[k] - ak;
                    dlx = cmax(del.abs() / xv[k].sqrt(), dlx);
                    let ck = (mm[k] - 1) * ni;
                    for j in 0..nin {
                        g[ia[j]] -= c[ck + ia[j]] * del;
                    }
                }
                if dlx < thr {
                    break;
                }
            }
            for l in 0..nin {
                da[l] = a[ia[l]] - da[l];
            }
            for j in 0..ni {
                if mm[j] != 0 {
                    continue;
                }
                if !ju[j] {
                    continue;
                }
                let mut s = 0f32;
                for l in 0..nin {
                    s += da[l] * c[l * ni + j];
                }
                g[j] -= s;
            }
            jz = 0;
        }
        let nk = nin.min(nx);

        // back in elnetu: unstandardize, intercept (dot_product in active order)
        let mut ca = [0f32; MAX_NI];
        for l in 0..nk {
            ca[l] = ys * a[ia[l]] / self.xs[ia[l]];
        }
        let mut dp = 0f32;
        for l in 0..nk {
            dp += ca[l] * self.xm[ia[l]];
        }
        for l in 0..nk {
            beta[ia[l]] = ca[l];
        }
        (ym - dp, beta)
    }

    /// autoTSFit of the values `y` at the design's dates `rows`: the 8
    /// coefficients and the RMSE, the residuals in `v_dif`.
    fn ts_fit(
        &mut self,
        terms: &[[f64; 8]],
        rows: &[usize],
        y: &[f64],
        ybuf: &mut Vec<f32>,
        v_dif: &mut Vec<f64>,
    ) -> ([f64; 8], f64) {
        ybuf.clear();
        ybuf.extend(y.iter().map(|&v| v as f32));
        let (a0, beta) = self.solve(ybuf);
        let mut cft = [0.0; 8];
        cft[0] = a0 as f64;
        for j in 0..self.ni {
            cft[j + 1] = beta[j] as f64;
        }
        let df = self.ni as i32 + 1;
        v_dif.clear();
        let mut ss = 0.0;
        for (i, &r) in rows.iter().enumerate() {
            let vd = y[i] - ts_pred(&terms[r], &cft);
            v_dif.push(vd);
            ss += vd * vd;
        }
        (cft, ss.sqrt() / ((rows.len() as i32 - df) as f64).sqrt())
    }
}

// ---------------------------------------------------------------------------
// One pixel of TrendSeasonalFit_v12_30Line.m

/// The make_ctx of the C++: the bands and thresholds of a call.
struct Ctx<'p> {
    nb: usize,
    b_detect: Vec<usize>,
    is_detect: Vec<bool>,
    b1: usize,
    b2: usize,
    conse: i32,
    num_c: i32,
    t_cg: f64,
    tmax_cg: f64,
    p: &'p Params,
}

impl<'p> Ctx<'p> {
    fn new(params: &'p Params, nb: usize) -> Fit<Ctx<'p>> {
        let nbi = nb as i32;
        let mut b_detect = params.detection_bands.clone();
        if b_detect.is_empty() {
            b_detect = if nb >= 6 {
                vec![1, 2, 3, 4, 5]
            } else {
                (0..nbi).collect()
            };
        }
        let mut tb = params.tmask_bands.clone();
        if tb.is_empty() {
            tb = if nb >= 5 {
                vec![1, 4]
            } else {
                vec![0, if nb > 1 { 1 } else { 0 }]
            };
        }
        if tb.len() != 2 {
            return Err("tmask_bands must hold exactly 2 band indices");
        }
        if b_detect.iter().any(|&b| b < 0 || b >= nbi) {
            return Err("detection band index out of range");
        }
        if tb.iter().any(|&b| b < 0 || b >= nbi) {
            return Err("tmask band index out of range");
        }
        let df = b_detect.len() as i32;
        let b_detect: Vec<usize> = b_detect.iter().map(|&b| b as usize).collect();
        let mut is_detect = vec![false; nb];
        for &b in &b_detect {
            is_detect[b] = true;
        }
        Ok(Ctx {
            nb,
            b_detect,
            is_detect,
            b1: tb[0] as usize,
            b2: tb[1] as usize,
            conse: params.conseq_anom,
            num_c: params.num_c,
            t_cg: chi2inv(params.chi2_prob_threshold, df),
            tmax_cg: chi2inv(params.tmax_cg_prob_threshold, df),
            p: params,
        })
    }

    /// norm(row(B_detect))^2, rounded like MATLAB's.
    fn vec_row(&self, row: &[f64]) -> f64 {
        let mut s = 0.0;
        for &b in &self.b_detect {
            s += row[b] * row[b];
        }
        let n = s.sqrt();
        n * n
    }
}

/// A pixel's observations in date order.
struct Series<'a> {
    /// MATLAB datenum dates.
    sdate: &'a [f64],
    /// The model terms of each date.
    terms: &'a [[f64; 8]],
    /// [date][band]
    line: &'a [f64],
    qa: &'a [i32],
}

/// The models of all the bands fitted on the same observations (fit_all).
struct Models {
    nb: usize,
    fit_cft: Vec<[f64; 8]>,
    rmse: Vec<f64>,
    /// [obs][band]: the residuals of the last fit that kept them.
    rec_v_dif: Vec<f64>,
    yv: Vec<f64>,
    ybuf: Vec<f32>,
    v_dif: Vec<f64>,
}

impl Models {
    fn new(nb: usize) -> Models {
        Models {
            nb,
            fit_cft: vec![[0.0; 8]; nb],
            rmse: vec![0.0; nb],
            rec_v_dif: Vec::new(),
            yv: Vec::new(),
            ybuf: Vec::new(),
            v_dif: Vec::new(),
        }
    }

    fn fit_all(&mut self, s: &Series, rows: &[usize], df: i32, keep_resid: bool) -> Fit<()> {
        let (nb, n) = (self.nb, rows.len());
        if keep_resid {
            self.rec_v_dif.clear();
            self.rec_v_dif.resize(n * nb, 0.0);
        }
        let mut lasso = Lasso::new(s.terms, rows, df)?;
        for bb in 0..nb {
            self.yv.clear();
            self.yv.extend(rows.iter().map(|&r| s.line[r * nb + bb]));
            let (cft, rmse) = lasso.ts_fit(s.terms, rows, &self.yv, &mut self.ybuf, &mut self.v_dif);
            self.fit_cft[bb] = cft;
            self.rmse[bb] = rmse;
            if keep_resid {
                for k in 0..n {
                    self.rec_v_dif[k * nb + bb] = self.v_dif[k];
                }
            }
        }
        Ok(())
    }

    fn coefs(&self) -> Vec<Vec<f64>> {
        self.fit_cft.iter().map(|c| c.to_vec()).collect()
    }
}

/// autoTmask: flags the observations `rows` whose green or SWIR1 residual
/// from a robust 2-frequency harmonic fit exceeds T_const x adj_rmse.
fn auto_tmask(s: &Series, c: &Ctx, rows: &[usize], yr: f64, var1: f64, var2: f64) -> Vec<bool> {
    let yr = yr.ceil();
    let (n, nb) = (rows.len(), c.nb);
    let x = Mat::from_fn(n, 4, Order::Col, |i, j| {
        let r = rows[i];
        match j {
            0 => s.terms[r][2],
            1 => s.terms[r][3],
            2 => ((W / yr) * s.sdate[r]).cos(),
            _ => ((W / yr) * s.sdate[r]).sin(),
        }
    });
    let y1: Vec<f64> = rows.iter().map(|&r| s.line[r * nb + c.b1]).collect();
    let y2: Vec<f64> = rows.iter().map(|&r| s.line[r * nb + c.b2]).collect();
    let (c1, c2) = (robust_fit(&x, &y1), robust_fit(&x, &y2));
    (0..n)
        .map(|i| {
            let r = rows[i];
            // cos(t * W) and sin(t * W) are the terms' cos(W * t) and sin(W * t)
            let (cw, sw) = (s.terms[r][2], s.terms[r][3]);
            let ty = s.sdate[r] * W / yr;
            let (cy, sy) = (ty.cos(), ty.sin());
            let p1 = c1[0] + c1[1] * cw + c1[2] * sw + c1[3] * cy + c1[4] * sy;
            let p2 = c2[0] + c2[1] * cw + c2[2] * sw + c2[3] * cy + c2[4] * sy;
            (y1[i] - p1).abs() > T_CONST * var1 || (y2[i] - p2).abs() > T_CONST * var2
        })
        .collect()
}

fn update_cft(i_span: i32, num_c: i32) -> i32 {
    if i_span < MID_NUM_C * N_TIMES {
        return MIN_NUM_C.min(num_c);
    }
    if i_span < MAX_NUM_C * N_TIMES {
        return MID_NUM_C.min(num_c);
    }
    MAX_NUM_C.min(num_c)
}

/// A new model of the record: zero RMSE, coefficients and magnitudes.
fn new_seg(s: &mut Segment, nb: usize) {
    s.rmse = vec![0.0; nb];
    s.magnitude = vec![0.0; nb];
    s.coefs = vec![vec![0.0; 8]; nb];
}

/// The record's model `k` (1-based), made if it is not there yet.
fn rec_at(rec: &mut Vec<Option<Segment>>, k: i32) -> &mut Segment {
    let k = k as usize;
    if rec.len() < k {
        rec.resize(k, None);
    }
    rec[k - 1].get_or_insert_with(Segment::default)
}

/// Observations a..=b (1-based) of `d`.
fn rows(d: &[usize], a: i32, b: i32) -> &[usize] {
    &d[(a - 1) as usize..b as usize]
}

/// The per-band medians of the rows r0..r1 of `m`.
fn median_rows(m: &[Vec<f64>], r0: usize, r1: usize, nb: usize) -> Vec<f64> {
    let mut col = Vec::with_capacity(r1.saturating_sub(r0));
    (0..nb)
        .map(|b| {
            col.clear();
            col.extend(m[r0..r1].iter().map(|row| row[b]));
            median_of(&col)
        })
        .collect()
}

fn fit_pixel(s: &Series, c: &Ctx) -> Fit<Vec<Segment>> {
    let num_t = s.sdate.len();
    let nb = c.nb;
    let p = c.p;
    let thermal = |b: usize| b as i32 == p.thermal_band;
    let mut rec: Vec<Option<Segment>> = Vec::new();
    let mut m = Models::new(nb);

    let n_exist = s.qa.iter().filter(|&&q| q < 255).count();
    if (n_exist as f64) / (num_t as f64) < 0.5 {
        return Ok(Vec::new());
    }

    let mut idrange = vec![false; num_t];
    let mut idclr = vec![false; num_t];
    let mut idsn = vec![false; num_t];
    let (mut n_clr, mut n_all, mut n_snow) = (0i32, 0i32, 0i32);
    for t in 0..num_t {
        let mut ok = true;
        for b in 0..nb {
            let v = s.line[t * nb + b];
            if thermal(b) {
                ok = ok && v > p.thermal_min && v < p.thermal_max;
            } else {
                ok = ok && v > p.valid_min && v < p.valid_max;
            }
        }
        idrange[t] = ok;
        idclr[t] = s.qa[t] < 2;
        idsn[t] = s.qa[t] == 3;
        n_clr += idclr[t] as i32;
        n_all += (s.qa[t] < 255) as i32;
        n_snow += idsn[t] as i32;
    }
    let clr_pct = n_clr as f64 / n_all as f64;
    let sn_pct = n_snow as f64 / ((n_clr + n_snow) as f64 + 0.01);
    const N_MIN: i32 = N_TIMES * MIN_NUM_C;

    if clr_pct < T_CLR {
        if sn_pct > T_SN {
            // mostly snow: one 4-coefficient model per band over clear + snow
            let d: Vec<usize> = (0..num_t).filter(|&t| idsn[t] || idclr[t]).collect();
            let n_sn = d.len();
            if (n_sn as i32) < N_MIN {
                return Ok(Vec::new());
            }
            let mut seg = Segment::default();
            new_seg(&mut seg, nb);
            for b in 0..nb {
                let rows: Vec<usize> = d
                    .iter()
                    .copied()
                    .filter(|&r| {
                        let v = s.line[r * nb + b];
                        if thermal(b) {
                            v > p.thermal_min && v < p.thermal_max
                        } else {
                            v < p.valid_max
                        }
                    })
                    .collect();
                if !thermal(b) && (rows.len() as i32) < N_MIN {
                    seg.coefs[b][0] = 10000.0;
                } else {
                    m.yv.clear();
                    m.yv.extend(rows.iter().map(|&r| s.line[r * nb + b]));
                    let mut lasso = Lasso::new(s.terms, &rows, MIN_NUM_C)?;
                    let (cft, rmse) = lasso.ts_fit(s.terms, &rows, &m.yv, &mut m.ybuf, &mut m.v_dif);
                    seg.coefs[b] = cft.to_vec();
                    seg.rmse[b] = rmse;
                }
            }
            seg.t_start = s.sdate[d[0]] as i32;
            seg.t_end = s.sdate[d[n_sn - 1]] as i32;
            seg.t_break = 0;
            seg.change_prob = 0.0;
            seg.num_obs = n_sn as i32;
            seg.category = 50 + MIN_NUM_C;
            return Ok(vec![seg]);
        }
        // not enough clear observations: screen clouds with green < median + 400
        let d: Vec<usize> = (0..num_t).filter(|&t| idrange[t]).collect();
        let g1: Vec<f64> = d.iter().map(|&r| s.line[r * nb + c.b1]).collect();
        let thr = median_of(&g1) + CLOUD_SCREEN_OFFSET;
        let e: Vec<usize> = d.iter().copied().filter(|&r| s.line[r * nb + c.b1] < thr).collect();
        if (e.len() as i32) < N_MIN {
            return Ok(Vec::new());
        }
        let mut seg = Segment::default();
        new_seg(&mut seg, nb);
        m.fit_all(s, &e, MIN_NUM_C, false)?;
        seg.coefs = m.coefs();
        seg.rmse = m.rmse.clone();
        seg.t_start = s.sdate[e[0]] as i32;
        seg.t_end = s.sdate[e[e.len() - 1]] as i32;
        seg.t_break = 0;
        seg.change_prob = 0.0;
        seg.num_obs = e.len() as i32;
        seg.category = 40 + MIN_NUM_C;
        return Ok(vec![seg]);
    }

    // ---- normal procedure
    let conse = c.conse;
    if conse < 1 {
        // the C++ reads outside its vectors (or throws std::length_error)
        return Err("conseq_anom must be at least 1");
    }
    let mut d: Vec<usize> = (0..num_t).filter(|&t| idclr[t] && idrange[t]).collect();
    // datenum, value and model of observation i (1-based) of d
    let x = |d: &[usize], i: i32| s.sdate[d[(i - 1) as usize]];
    let y = |d: &[usize], i: i32, b: usize| s.line[d[(i - 1) as usize] * nb + b];
    let pred = |d: &[usize], i: i32, cft: &[f64; 8]| ts_pred(&s.terms[d[(i - 1) as usize]], cft);
    let first_at_or_after = |d: &[usize], t: f64| {
        for k in 1..=d.len() as i32 {
            if x(d, k) >= t {
                return k;
            }
        }
        d.len() as i32 + 1
    };

    let mut adj_rmse = vec![f64::NAN; nb];
    let mut diffs = Vec::with_capacity(d.len());
    for b in 0..nb {
        diffs.clear();
        for i in 2..=d.len() as i32 {
            diffs.push((y(&d, i, b) - y(&d, i - 1, b)).abs());
        }
        adj_rmse[b] = median_of(&diffs);
    }

    let mut i = N_MIN;
    let mut i_start = 1;
    let mut bl_train = false;
    let mut num_fc = 1;
    let rec_fc = num_fc;
    let mut i_count = 0.0;
    let mut tmpcg_rmse = vec![0.0; nb];
    let mut v_dif: Vec<Vec<f64>> = Vec::new(); // [row][band]
    let mut v_dif_mag: Vec<Vec<f64>> = Vec::new();
    let mut vec_mag: Vec<f64> = Vec::new();
    let mut ids_old: Vec<i32> = Vec::new();

    while i <= d.len() as i32 - conse {
        let mut i_span = i - i_start + 1;
        let mut time_span = (x(&d, i) - x(&d, i_start)) / NUM_YRS;
        if i_span >= N_MIN && time_span >= MINI_YRS {
            if !bl_train {
                // Tmask over i_start .. i + conse
                let bl = auto_tmask(
                    s,
                    c,
                    &d[(i_start - 1) as usize..(i + conse) as usize],
                    (x(&d, i + conse) - x(&d, i_start)) / NUM_YRS,
                    adj_rmse[c.b1],
                    adj_rmse[c.b2],
                );
                let n_ids = bl.len() as i32;
                let mut rm_ids = Vec::new();
                i_span = 0;
                for k in 0..n_ids - conse {
                    if bl[k as usize] {
                        rm_ids.push(i_start + k);
                    } else {
                        i_span += 1;
                    }
                }
                if i_span < N_MIN {
                    i += 1;
                    continue;
                }
                let mut cp = d.clone();
                for &id in rm_ids.iter().rev() {
                    cp.remove((id - 1) as usize);
                }
                let i_rec = i;
                i = i_start + i_span - 1;
                time_span = (x(&cp, i) - x(&cp, i_start)) / NUM_YRS;
                if time_span < MINI_YRS {
                    i = i_rec;
                    i += 1;
                    continue;
                }
                d = cp;
                m.fit_all(s, rows(&d, i_start, i), MIN_NUM_C, true)?;
                let last = (i - i_start) as usize * nb;
                let mut vd = vec![0.0; nb];
                for &b in &c.b_detect {
                    let mini_rmse = cmax(adj_rmse[b], m.rmse[b]);
                    let v_start = m.rec_v_dif[b] / mini_rmse;
                    let v_end = m.rec_v_dif[last + b] / mini_rmse;
                    let v_slope = m.fit_cft[b][1] * (x(&d, i) - x(&d, i_start)) / mini_rmse;
                    vd[b] = v_slope.abs() + v_start.abs() + v_end.abs();
                }
                if c.vec_row(&vd) > c.t_cg {
                    i_start += 1;
                    i += 1;
                    continue;
                }

                bl_train = true;
                i_count = 0.0;
                let i_break = if num_fc == rec_fc {
                    1
                } else {
                    first_at_or_after(&d, rec_at(&mut rec, num_fc - 1).t_break as f64)
                };
                if i_start > i_break {
                    let mut i_ini = i_start - 1;
                    while i_ini >= i_break {
                        let ini_conse = if i_start - i_break < conse {
                            i_start - i_break
                        } else {
                            conse
                        };
                        v_dif = vec![vec![0.0; nb]; ini_conse as usize];
                        v_dif_mag = v_dif.clone();
                        vec_mag = vec![0.0; ini_conse as usize];
                        for ic in 1..=ini_conse {
                            let o = i_ini - ic + 1;
                            let r = (ic - 1) as usize;
                            for b in 0..nb {
                                v_dif_mag[r][b] = y(&d, o, b) - pred(&d, o, &m.fit_cft[b]);
                                if c.is_detect[b] {
                                    v_dif[r][b] = v_dif_mag[r][b] / cmax(adj_rmse[b], m.rmse[b]);
                                }
                            }
                            vec_mag[r] = c.vec_row(&v_dif[r]);
                        }
                        if min_element(&vec_mag) > c.t_cg {
                            break;
                        } else if vec_mag[0] > c.tmax_cg {
                            d.remove((i_ini - 1) as usize);
                            i -= 1;
                        }
                        i_start = i_ini;
                        i_ini -= 1;
                    }
                }
                if num_fc == rec_fc && i_start - i_break >= conse {
                    m.fit_all(s, rows(&d, i_break, i_start - 1), MIN_NUM_C, false)?;
                    let seg = rec_at(&mut rec, num_fc);
                    new_seg(seg, nb);
                    seg.t_end = x(&d, i_start - 1) as i32;
                    seg.coefs = m.coefs();
                    seg.rmse = m.rmse.clone();
                    seg.t_break = x(&d, i_start) as i32;
                    seg.change_prob = 1.0;
                    seg.t_start = x(&d, 1) as i32;
                    seg.category = 10 + MIN_NUM_C;
                    seg.num_obs = i_start - i_break;
                    let med = median_rows(&v_dif_mag, 0, v_dif_mag.len(), nb);
                    for b in 0..nb {
                        seg.magnitude[b] = -med[b];
                    }
                    num_fc += 1;
                }
            }

            if bl_train {
                let (ids_a, ids_b) = (i_start, i);
                i_span = i - i_start + 1;
                let update_num_c = update_cft(i_span, c.num_c);
                if i_count == 0.0 || i_span <= MAX_NUM_C * N_TIMES {
                    i_count = x(&d, i) - x(&d, i_start);
                    m.fit_all(s, rows(&d, ids_a, ids_b), update_num_c, true)?;
                    let seg = rec_at(&mut rec, num_fc);
                    new_seg(seg, nb);
                    seg.t_start = x(&d, i_start) as i32;
                    seg.t_end = x(&d, i) as i32;
                    seg.t_break = 0;
                    seg.coefs = m.coefs();
                    seg.rmse = m.rmse.clone();
                    seg.change_prob = 0.0;
                    seg.num_obs = i - i_start + 1;
                    seg.category = update_num_c;
                    v_dif = vec![vec![0.0; nb]; conse as usize];
                    v_dif_mag = v_dif.clone();
                    vec_mag = vec![0.0; conse as usize];
                    for ic in 1..=conse {
                        let r = (ic - 1) as usize;
                        for b in 0..nb {
                            v_dif_mag[r][b] = y(&d, i + ic, b) - pred(&d, i + ic, &m.fit_cft[b]);
                            if c.is_detect[b] {
                                v_dif[r][b] = v_dif_mag[r][b] / cmax(adj_rmse[b], m.rmse[b]);
                            }
                        }
                        vec_mag[r] = c.vec_row(&v_dif[r]);
                    }
                    ids_old.clear();
                    ids_old.extend(ids_a..=ids_b);
                } else {
                    if x(&d, i) - x(&d, i_start) >= 1.33 * i_count {
                        i_count = x(&d, i) - x(&d, i_start);
                        m.fit_all(s, rows(&d, ids_a, ids_b), update_num_c, true)?;
                        let seg = rec_at(&mut rec, num_fc);
                        seg.coefs = m.coefs();
                        seg.rmse = m.rmse.clone();
                        seg.num_obs = i - i_start + 1;
                        seg.category = update_num_c;
                        ids_old.clear();
                        ids_old.extend(ids_a..=ids_b);
                    }
                    let seg = rec_at(&mut rec, num_fc);
                    seg.t_end = x(&d, i) as i32;
                    let category = seg.category;
                    let n_rmse = N_TIMES * category;
                    // RMSE from the n_rmse observations closest in season to i + conse
                    let x_conse = x(&d, i + conse);
                    let d_yr: Vec<f64> = ids_old
                        .iter()
                        .map(|&k| {
                            let d_rt = x(&d, k) - x_conse;
                            ((d_rt / NUM_YRS).round() * NUM_YRS - d_rt).abs()
                        })
                        .collect();
                    let mut order: Vec<usize> = (0..ids_old.len()).collect();
                    order.sort_by(|&a, &b| {
                        if d_yr[a] < d_yr[b] {
                            std::cmp::Ordering::Less
                        } else if d_yr[b] < d_yr[a] {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    });
                    order.truncate(n_rmse.max(0) as usize);
                    for &b in &c.b_detect {
                        let mut ss = 0.0;
                        for &k in &order {
                            let v = m.rec_v_dif[(ids_old[k] - ids_old[0]) as usize * nb + b];
                            ss += v * v;
                        }
                        tmpcg_rmse[b] = ss.sqrt() / ((n_rmse - category) as f64).sqrt();
                    }
                    let last = (conse - 1) as usize;
                    v_dif[..=last].rotate_left(1);
                    v_dif_mag[..=last].rotate_left(1);
                    vec_mag[..=last].rotate_left(1);
                    v_dif[last].clear();
                    v_dif[last].resize(nb, 0.0);
                    v_dif_mag[last].clear();
                    v_dif_mag[last].resize(nb, 0.0);
                    vec_mag[last] = 0.0;
                    for b in 0..nb {
                        v_dif_mag[last][b] = y(&d, i + conse, b) - pred(&d, i + conse, &m.fit_cft[b]);
                        if c.is_detect[b] {
                            v_dif[last][b] = v_dif_mag[last][b] / cmax(adj_rmse[b], tmpcg_rmse[b]);
                        }
                    }
                    vec_mag[last] = c.vec_row(&v_dif[last]);
                }
                if min_element(&vec_mag) > c.t_cg {
                    let seg = rec_at(&mut rec, num_fc);
                    seg.t_break = x(&d, i + 1) as i32;
                    seg.change_prob = 1.0;
                    seg.magnitude = median_rows(&v_dif_mag, 0, conse as usize, nb);
                    num_fc += 1;
                    i_start = i + 1;
                    bl_train = false;
                } else if vec_mag[0] > c.tmax_cg {
                    d.remove(i as usize); // erase(i + 1)
                    i -= 1;
                }
            }
        }
        i += 1;
    }

    if bl_train {
        let mut id_last = conse;
        for ic in (1..=conse).rev() {
            if vec_mag[(ic - 1) as usize] <= c.t_cg {
                id_last = ic;
                break;
            }
        }
        let len = d.len() as i32;
        let seg = rec_at(&mut rec, num_fc);
        seg.change_prob = (conse - id_last) as f64 / conse as f64;
        seg.t_end = x(&d, len - conse + id_last) as i32;
        if conse > id_last {
            seg.t_break = x(&d, len - conse + id_last + 1) as i32;
            seg.magnitude = median_rows(&v_dif_mag, id_last as usize, conse as usize, nb);
        }
    } else {
        let i_st = if num_fc == rec_fc {
            1
        } else {
            first_at_or_after(&d, rec_at(&mut rec, num_fc - 1).t_break as f64)
        };
        let len = d.len() as i32;
        if len - i_st + 1 > conse {
            let bl = auto_tmask(
                s,
                c,
                &d[(i_st - 1) as usize..],
                (x(&d, len) - x(&d, i_st)) / NUM_YRS,
                adj_rmse[c.b1],
                adj_rmse[c.b2],
            );
            let n_ids = bl.len() as i32;
            for k in (0..n_ids - conse).rev() {
                if bl[k as usize] {
                    d.remove((i_st + k - 1) as usize);
                }
            }
        }
        let len = d.len() as i32;
        if len - i_st + 1 >= conse {
            m.fit_all(s, rows(&d, i_st, len), MIN_NUM_C, false)?;
            let seg = rec_at(&mut rec, num_fc);
            new_seg(seg, nb);
            seg.t_start = x(&d, i_st) as i32;
            seg.t_end = x(&d, len) as i32;
            seg.t_break = 0;
            seg.coefs = m.coefs();
            seg.rmse = m.rmse.clone();
            seg.change_prob = 0.0;
            seg.num_obs = len - i_st + 1;
            seg.category = 20 + MIN_NUM_C;
        }
    }

    Ok(rec.into_iter().flatten().collect())
}

/// The dates in the order the original reads the images (stable: the first
/// of equal dates first), their datenums and the model terms there.
fn date_order(dates: &[i32]) -> (Vec<usize>, Vec<f64>, Vec<[f64; 8]>) {
    let mut order: Vec<usize> = (0..dates.len()).collect();
    order.sort_by_key(|&t| dates[t]);
    let sdate: Vec<f64> = order.iter().map(|&t| dates[t] as f64 + DATENUM_OFFSET as f64).collect();
    let terms = sdate.iter().map(|&t| terms(t)).collect();
    (order, sdate, terms)
}

/// fit_pixel on the series in date order, its dates back in the input convention.
fn fit_sorted(s: &Series, c: &Ctx) -> Fit<Vec<Segment>> {
    let mut segs = fit_pixel(s, c)?;
    for seg in &mut segs {
        seg.t_start -= DATENUM_OFFSET;
        seg.t_end -= DATENUM_OFFSET;
        if seg.t_break > 0 {
            seg.t_break -= DATENUM_OFFSET;
        }
    }
    Ok(segs)
}

/// CCDC for one pixel (TrendSeasonalFit_v12_30Line.m). dates: integer days
/// (Python ordinal days), bands: [band][time], qa: Fmask codes (0 clear land,
/// 1 water, 2 shadow, 3 snow, 4 cloud, 255 no data).
pub fn fit_ccdc(dates: &[i32], bands: &[Vec<f64>], qa: &[i32], params: &Params) -> Fit<Vec<Segment>> {
    if bands.is_empty() || dates.is_empty() {
        return Ok(Vec::new());
    }
    if qa.len() != dates.len() {
        return Err("qa and dates must have the same length");
    }
    if bands.iter().any(|b| b.len() != dates.len()) {
        return Err("every band must have one value per date");
    }
    let nb = bands.len();
    let c = Ctx::new(params, nb)?;
    let (order, sdate, terms) = date_order(dates);
    let mut line = vec![0.0; order.len() * nb];
    for (k, &t) in order.iter().enumerate() {
        for b in 0..nb {
            line[k * nb + b] = bands[b][t];
        }
    }
    let q: Vec<i32> = order.iter().map(|&t| qa[t]).collect();
    let s = Series {
        sdate: &sdate,
        terms: &terms,
        line: &line,
        qa: &q,
    };
    fit_sorted(&s, &c)
}

/// CCDC of every pixel of `values` ([pixels][bands][times]) with `qa`
/// ([pixels][times]): (segments [pixels][max_segments][per segment], counts),
/// per segment t_start, t_end, t_break, then per band rmse and the 8
/// coefficients, or t_break alone without `return_coefs`. A date with NaN in
/// any band has no observation; a pixel without data, or whose fit fails, has
/// no segment.
pub fn fit_ccdc_batch(
    values: &[f64],
    qa: &[i32],
    dates: &[i32],
    pixels: usize,
    nb: usize,
    params: &Params,
    max_segments: usize,
    return_coefs: bool,
    n_jobs: i32,
) -> Fit<(Vec<f64>, Vec<i32>)> {
    let times = dates.len();
    assert!(values.len() == pixels * nb * times && qa.len() == pixels * times);
    let c = Ctx::new(params, nb)?;
    let (order, sdate, terms) = date_order(dates);
    let pps = if return_coefs { 3 + nb * 9 } else { 1 };
    let seg_width = max_segments * pps;
    let width = seg_width + 1; // the segments, then their count
    let mut per_pixel = vec![0.0; pixels * width];
    parallel::for_each_row(
        &mut per_pixel,
        width,
        n_jobs,
        || (vec![0.0; times * nb], vec![0i32; times]),
        |(line, q): &mut (Vec<f64>, Vec<i32>), px, row| {
            let v = &values[px * nb * times..(px + 1) * nb * times];
            let mut any_data = false;
            for (k, &t) in order.iter().enumerate() {
                q[k] = qa[px * times + t];
                let mut nan_here = false;
                for b in 0..nb {
                    let x = v[b * times + t];
                    line[k * nb + b] = x;
                    if x.is_nan() {
                        nan_here = true;
                    } else if x != 0.0 {
                        any_data = true;
                    }
                }
                if nan_here {
                    // no observation on this date
                    q[k] = 255;
                    line[k * nb..(k + 1) * nb].fill(0.0);
                }
            }
            if !any_data {
                return;
            }
            let s = Series {
                sdate: &sdate,
                terms: &terms,
                line,
                qa: q,
            };
            let segs = fit_sorted(&s, &c).unwrap_or_default();
            let n_segs = segs.len().min(max_segments);
            row[seg_width] = n_segs as f64;
            for (i, seg) in segs.iter().take(n_segs).enumerate() {
                let out = &mut row[i * pps..(i + 1) * pps];
                if return_coefs {
                    out[0] = seg.t_start as f64;
                    out[1] = seg.t_end as f64;
                    out[2] = seg.t_break as f64;
                    let mut idx = 3;
                    for b in 0..nb {
                        out[idx] = seg.rmse[b];
                        idx += 1;
                        for k in 0..8 {
                            out[idx] = seg.coefs[b][k];
                            idx += 1;
                        }
                    }
                } else {
                    out[0] = seg.t_break as f64;
                }
            }
        },
    );
    let mut segments = Vec::with_capacity(pixels * seg_width);
    let mut counts = Vec::with_capacity(pixels);
    for row in per_pixel.chunks_exact(width) {
        segments.extend_from_slice(&row[..seg_width]);
        counts.push(row[seg_width] as i32);
    }
    Ok((segments, counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi2inv_values() {
        // MATLAB: chi2inv(0.99, 5) = 15.0863, chi2inv(0.999999, 5) = 35.8882
        assert!((chi2inv(0.99, 5) - 15.086_272_469_388_99).abs() < 1e-9);
        assert!((chi2inv(0.999_999, 5) - 35.888_186_879_610_4).abs() < 1e-6);
        assert!((chi2inv(0.5, 2) - 2.0 * 2f64.ln()).abs() < 1e-12);
    }

    fn pixel(break_at: Option<usize>) -> (Vec<i32>, Vec<Vec<f64>>, Vec<i32>) {
        let dates: Vec<i32> = (0..230).map(|k| 730_120 + 16 * k).collect();
        let bands = (0..6)
            .map(|b| {
                dates
                    .iter()
                    .enumerate()
                    .map(|(k, &d)| {
                        let base = 500.0 + 300.0 * b as f64 + 200.0 * (W * (d as f64 + 366.0)).sin();
                        let shift = if break_at.is_some_and(|at| k >= at) {
                            1500.0
                        } else {
                            0.0
                        };
                        base + shift + ((k * 7919 + b * 104_729) % 97) as f64 - 48.0
                    })
                    .collect()
            })
            .collect();
        (dates, bands, vec![0; 230])
    }

    #[test]
    fn finds_a_break() {
        let (dates, bands, qa) = pixel(Some(120));
        let segs = fit_ccdc(&dates, &bands, &qa, &Params::default()).unwrap();
        assert!(segs.len() >= 2, "{segs:?}");
        assert_eq!(segs[0].t_break, dates[120]);
        assert_eq!(segs[0].change_prob, 1.0);
        let (dates, bands, qa) = pixel(None);
        let segs = fit_ccdc(&dates, &bands, &qa, &Params::default()).unwrap();
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].t_break, 0);
    }

    #[test]
    fn rejects_bad_bands() {
        let (dates, bands, qa) = pixel(None);
        let p = Params {
            tmask_bands: vec![1],
            ..Params::default()
        };
        assert_eq!(
            fit_ccdc(&dates, &bands, &qa, &p),
            Err("tmask_bands must hold exactly 2 band indices")
        );
        let p = Params {
            detection_bands: vec![6],
            ..Params::default()
        };
        assert_eq!(
            fit_ccdc(&dates, &bands, &qa, &p),
            Err("detection band index out of range")
        );
        assert_eq!(
            fit_ccdc(&dates, &bands, &qa[1..], &Params::default()),
            Err("qa and dates must have the same length")
        );
    }
}
