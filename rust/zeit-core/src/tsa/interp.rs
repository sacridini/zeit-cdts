//! Interpolation of a series to new dates: linear, moving average, ensembles
//! of radial basis functions and robust harmonic regression.
//!
//! Ported from FORCE (Framework for Operational Radiometric Correction for
//! Environmental monitoring), Copyright (C) 2013-2025 David Frantz,
//! GPL-3.0-or-later, https://github.com/davidfrantz/force, src/modules:
//! higher-level/interpolate-hl.c. The RBF ensemble is FORCE's adaptation of
//! Andreas Rabe's `RBF_fitting_large_scale.py`. The robust fit follows the
//! iteratively reweighted least squares of GSL's `gsl_multifit_robust` with
//! the bisquare weights (multifit/multirobust.c, Copyright (C) 2013 Patrick
//! Alken, GPL-3.0-or-later): OLS start, residuals adjusted by the leverage,
//! MAD scale from all but the smallest p residuals, tuning 4.685, the lower
//! bound of the scale, the relative convergence test of sqrt(eps) and 100
//! iterations at most.
//!
//! Unlike FORCE, which keeps int16 and truncates, the work is in f64 and the
//! results stay f64; NaN marks a missing observation and an output without
//! enough data. The times are in days (any origin, real or FORCE's calendar
//! `ce`) and increase along the series; the new times may be any.
//!
//! Two differences of form, not of result: the RBF's kernels are folded into
//! one weight per distance (the ensemble of FORCE is a weighted mean of
//! weighted means, which is one weighted mean), and the harmonic fit centers
//! and scales the time (the fit does not change: the cosines and sines absorb
//! the phase and the intercept the offset).

use super::Sample;
use crate::parallel;

/// How a series is interpolated.
#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    /// Straight lines between the valid observations on either side; constant
    /// beyond the first and last.
    Linear,
    /// The mean of the valid observations within `window` days (closed).
    Moving { window: f64 },
    /// An ensemble of Gaussian kernels of standard deviations `sigma` (days),
    /// each cut where the mass of the kernel reaches `cutoff`.
    Rbf { sigma: Vec<f64>, cutoff: f64 },
    /// y = a0 [+ a1 t] + sum over m <= `modes` of b_m cos(2 pi m t / period) + c_m sin(...),
    /// fit robustly to the observations within `fit` (inclusive), predicted everywhere.
    Harmonic {
        modes: usize,
        trend: bool,
        fit: (f64, f64),
        period: f64,
    },
}

/// The buffers of one thread.
#[derive(Default)]
struct Work {
    /// The valid observations of the series: times and values.
    t: Vec<f64>,
    ti: Vec<i64>,
    y: Vec<f64>,
    /// The harmonic design rows of the valid observations.
    rows: Vec<f64>,
    idx: Vec<usize>,
    wts: Vec<f64>,
    res: Vec<f64>,
    fac: Vec<f64>,
    abs: Vec<f64>,
    /// For the linear interpolation, the number of valid observations before each step.
    prev: Vec<u32>,
    /// The prefix sums (integers) and counts of the valid observations.
    pre_sum: Vec<i64>,
    pre_num: Vec<u32>,
}

/// The values of every series of `values` (pixels, n_in), at the times `t_out`:
/// (pixels, n_out), flat. `t_in` increases (ties allowed). `values` may be f64
/// or f32 (the work is in f64 either way).
pub fn interp_batch<T: Sample>(
    values: &[T],
    n_pixels: usize,
    n_in: usize,
    t_in: &[f64],
    t_out: &[f64],
    method: &Method,
    n_jobs: i32,
) -> Result<Vec<f64>, String> {
    if values.len() != n_pixels * n_in {
        return Err("values must be n_pixels x n_in".into());
    }
    if t_in.len() != n_in {
        return Err("one time for each step of the series".into());
    }
    if t_in.iter().chain(t_out).any(|t| !t.is_finite()) {
        return Err("the times must be finite".into());
    }
    if t_in.windows(2).any(|w| w[1] < w[0]) {
        return Err("the times of the series must not decrease".into());
    }
    let n_out = t_out.len();
    if n_pixels == 0 || n_out == 0 || n_in == 0 {
        return Ok(vec![f64::NAN; n_pixels * n_out]);
    }
    // Every kernel writes all of its row: the output is not filled here, its
    // pages are first written (and so faulted in) by the threads.
    let mut out = parallel::uninit_f64(n_pixels * n_out);
    match method {
        Method::Linear => {
            // Where each new time falls among all the times of the series, once
            // for every pixel: the first step with t >= x.
            let at: Vec<u32> = t_out.iter().map(|&x| t_in.partition_point(|&v| v < x) as u32).collect();
            parallel::for_each_row(&mut out, n_out, n_jobs, Work::default, |w, p, row| {
                linear(&values[p * n_in..(p + 1) * n_in], t_in, t_out, &at, w, row);
            });
        }
        Method::Moving { window } => {
            if !(*window >= 0.0) || !window.is_finite() {
                return Err("window must be a non-negative number of days".into());
            }
            let window = *window;
            // The steps of each window, once for every pixel.
            let span: Vec<(u32, u32)> = t_out
                .iter()
                .map(|&x| {
                    let lo = t_in.partition_point(|&v| v < x - window);
                    let hi = t_in.partition_point(|&v| v <= x + window).max(lo);
                    (lo as u32, hi as u32)
                })
                .collect();
            parallel::for_each_row(&mut out, n_out, n_jobs, Work::default, |w, p, row| {
                moving(&values[p * n_in..(p + 1) * n_in], &span, w, row);
            });
        }
        Method::Rbf { sigma, cutoff } => {
            let kernel = Kernel::new(sigma, *cutoff)?;
            let integer = t_in.iter().chain(t_out).all(|t| t.fract() == 0.0 && t.abs() < 4.0e15);
            if integer {
                let table = kernel.table();
                let h = kernel.reach as i64;
                parallel::for_each_row(&mut out, n_out, n_jobs, Work::default, |w, p, row| {
                    compact(&values[p * n_in..(p + 1) * n_in], t_in, w);
                    w.ti.clear();
                    w.ti.extend(w.t.iter().map(|&t| t as i64));
                    for (o, &x) in row.iter_mut().zip(t_out) {
                        *o = rbf_table(&w.ti, &w.y, x as i64, h, &table);
                    }
                });
            } else {
                parallel::for_each_row(&mut out, n_out, n_jobs, Work::default, |w, p, row| {
                    compact(&values[p * n_in..(p + 1) * n_in], t_in, w);
                    for (o, &x) in row.iter_mut().zip(t_out) {
                        *o = rbf_direct(&w.t, &w.y, x, &kernel);
                    }
                });
            }
        }
        Method::Harmonic {
            modes,
            trend,
            fit,
            period,
        } => {
            if !(1..=3).contains(modes) {
                return Err("modes must be 1, 2 or 3".into());
            }
            if !(*period > 0.0) || !period.is_finite() {
                return Err("period must be a positive number of days".into());
            }
            let design = Design::new(t_in, t_out, *modes, *trend, *period);
            let p = design.p;
            let (lo, hi) = *fit;
            let in_fit: Vec<bool> = t_in.iter().map(|&t| t >= lo && t <= hi).collect();
            parallel::for_each_row(&mut out, n_out, n_jobs, Work::default, |w, px, row| {
                let series = &values[px * n_in..(px + 1) * n_in];
                w.idx.clear();
                w.idx
                    .extend((0..n_in).filter(|&i| in_fit[i] && !series[i].to_f64().is_nan()));
                let mut c = [0.0; MAXP];
                if w.idx.len() < p || robust_fit(series, &design, w, &mut c).is_none() {
                    row.fill(f64::NAN);
                    return;
                }
                for (o, x) in row.iter_mut().zip(design.x_out.chunks_exact(p)) {
                    *o = x.iter().zip(&c).map(|(a, b)| a * b).sum();
                }
            });
        }
    }
    Ok(out)
}

/// The valid observations of `series` into `w.t` and `w.y`.
fn compact<T: Sample>(series: &[T], t_in: &[f64], w: &mut Work) {
    // written without a branch on the (unpredictable) gaps: every step is
    // stored, the position advances only on a valid one
    let n = series.len();
    w.t.resize(n, 0.0);
    w.y.resize(n, 0.0);
    let mut k = 0;
    let (ts, ys) = (&mut w.t[..], &mut w.y[..]);
    for i in 0..n {
        let v = series[i].to_f64();
        ts[k] = t_in[i];
        ys[k] = v;
        k += usize::from(!v.is_nan());
    }
    w.t.truncate(k);
    w.y.truncate(k);
}

// ---- linear ----------------------------------------------------------------

/// `interpolate_linear`: the last valid observation before `x` and the first
/// after; one at `x` is the value; constant beyond the ends; none: NaN.
///
/// `at[k]` is the first step of the whole series with t >= t_out[k]. One pass
/// compacts the valid observations (without a branch on the gaps, which are as
/// unpredictable as the clouds) and counts them as it goes: the count at `at[k]`
/// is where t_out[k] falls among the valid observations, which replaces a
/// bisection of them for each new time.
fn linear<T: Sample>(series: &[T], t_in: &[f64], t_out: &[f64], at: &[u32], w: &mut Work, out: &mut [f64]) {
    let n = series.len();
    let Work { t, y, prev: rank, .. } = w;
    ensure(t, n + 1, 0.0);
    ensure(y, n + 1, 0.0);
    ensure(rank, n + 1, 0);
    let (t, y, rank) = (&mut t[..n + 1], &mut y[..n + 1], &mut rank[..n + 1]);
    let mut k = 0usize;
    for (s, (&ti, r)) in series.iter().zip(t_in.iter().zip(rank.iter_mut())) {
        let v = s.to_f64();
        *r = k as u32;
        t[k] = ti;
        y[k] = v;
        k += usize::from(v == v);
    }
    rank[n] = k as u32;
    let nv = k;
    if nv == 0 {
        out.fill(f64::NAN);
        return;
    }
    for ((o, &x), &a) in out.iter_mut().zip(t_out).zip(at) {
        let r = rank[a as usize] as usize; // the valid observations before x
        // the neighbours clamped into the valid observations, so that the
        // interpolation is always computed and the cases only select
        let (l, h) = (r.saturating_sub(1), r.min(nv - 1));
        let (xl, xr, yl, yr) = (t[l], t[h], y[l], y[h]);
        let line = (yl * (xr - x) + yr * (x - xl)) / (xr - xl);
        let exact = r < nv && xr == x;
        let inside = r > 0 && r < nv;
        let end = if r == 0 { yl } else { yr }; // the first or the last observation
        *o = if exact {
            yr
        } else if inside {
            line
        } else {
            end
        };
    }
}

/// `v` grown to at least `n` elements.
#[inline]
fn ensure<V: Copy>(v: &mut Vec<V>, n: usize, fill: V) {
    if v.len() < n {
        v.resize(n, fill);
    }
}

// ---- moving average ----------------------------------------------------------

/// `interpolate_moving`: the mean of the valid observations with |t - x| <= window.
///
/// `span[k]` are the steps of the whole series within the window of the new
/// time k. When every value of the series is an integer (as FORCE's int16 are),
/// the sums are exact in any order and come from prefix sums; otherwise the
/// window is added up in order, as always.
fn moving<T: Sample>(series: &[T], span: &[(u32, u32)], w: &mut Work, out: &mut [f64]) {
    let n = series.len();
    let Work { pre_sum, pre_num, .. } = w;
    ensure(pre_sum, n + 1, 0);
    ensure(pre_num, n + 1, 0);
    let (mut s, mut c, mut exact) = (0i64, 0u32, true);
    pre_sum[0] = 0;
    pre_num[0] = 0;
    for ((v, ps), pc) in series.iter().zip(pre_sum[1..].iter_mut()).zip(pre_num[1..].iter_mut()) {
        let v = v.to_f64();
        let ok = v == v;
        let iv = (v as i64) & i64::from(ok).wrapping_neg(); // 0 for a gap (`as` gives 0 for NaN anyway)
        exact &= !ok | ((iv as f64 == v) & (v.abs() < 1.0e12));
        s += iv;
        c += u32::from(ok);
        *ps = s;
        *pc = c;
    }
    if exact {
        for (o, &(lo, hi)) in out.iter_mut().zip(span) {
            let (lo, hi) = (lo as usize, hi as usize);
            let num = pre_num[hi] - pre_num[lo];
            *o = if num > 0 {
                (pre_sum[hi] - pre_sum[lo]) as f64 / f64::from(num)
            } else {
                f64::NAN
            };
        }
    } else {
        for (o, &(lo, hi)) in out.iter_mut().zip(span) {
            let (mut sum, mut num) = (0.0, 0.0);
            for v in &series[lo as usize..hi as usize] {
                let v = v.to_f64();
                if !v.is_nan() {
                    sum += v;
                    num += 1.0;
                }
            }
            *o = if num > 0.0 { sum / num } else { f64::NAN };
        }
    }
}

// ---- RBF ensemble ------------------------------------------------------------

/// The kernels of FORCE's `rbf_kernel`.
struct Kernel {
    sigma: Vec<f64>,
    /// The largest distance (days) of each kernel's mass up to `cutoff`.
    max_ce: Vec<i64>,
    /// The weight of each kernel: the sum of its kernel up to `max_ce`, the centre counted twice.
    max_w: Vec<f64>,
    /// The largest `max_ce`.
    reach: usize,
}

impl Kernel {
    fn new(sigma: &[f64], cutoff: f64) -> Result<Kernel, String> {
        if sigma.is_empty() || sigma.iter().any(|&s| !(s > 0.0) || !s.is_finite()) {
            return Err("sigma must be a non-empty list of positive numbers of days".into());
        }
        if !(cutoff > 0.0 && cutoff <= 1.0) {
            return Err("cutoff must be in (0, 1]".into());
        }
        let last = sigma[sigma.len() - 1];
        let hbin = (4.0 * last).floor() as i64; // half of FORCE's 8 sigma bins
        let mut max_ce = Vec::new();
        let mut max_w = Vec::new();
        for &s in sigma {
            let (mut sum, mut mc, mut mw) = (0.0f32, 0i64, 0.0f64);
            for d in 0..=hbin {
                let k = Kernel::at(s, d as f64) as f32;
                sum += k; // FORCE's float accumulation decides where the kernel ends
                if f64::from(sum) <= cutoff / 2.0 {
                    mc = d;
                    mw += f64::from(k);
                }
            }
            if mw == 0.0 {
                return Err("cutoff is too small for sigma: the centre of the kernel alone exceeds it".into());
            }
            max_ce.push(mc);
            max_w.push(mw * 2.0);
        }
        let reach = *max_ce.iter().max().unwrap() as usize;
        Ok(Kernel {
            sigma: sigma.to_vec(),
            max_ce,
            max_w,
            reach,
        })
    }

    #[inline]
    fn at(sigma: f64, d: f64) -> f64 {
        1.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma) * (-0.5 * d * d / (sigma * sigma)).exp()
    }

    /// The weight of an observation at distance `d`: the kernels that reach it,
    /// each over its weight.
    #[inline]
    fn combined(&self, d: f64) -> f64 {
        let a = d.abs();
        let mut s = 0.0;
        for k in 0..self.sigma.len() {
            if a <= self.max_ce[k] as f64 {
                s += Kernel::at(self.sigma[k], d) / self.max_w[k];
            }
        }
        s
    }

    /// The combined weight at each integer distance -reach..=reach.
    fn table(&self) -> Vec<f64> {
        let h = self.reach as i64;
        (-h..=h).map(|d| self.combined(d as f64)).collect()
    }
}

/// The ensemble at `x` with integer times: the weighted mean with the weight table.
fn rbf_table(t: &[i64], y: &[f64], x: i64, h: i64, table: &[f64]) -> f64 {
    let mut j = t.partition_point(|&v| v < x - h);
    let (mut num, mut den) = (0.0, 0.0);
    while j < t.len() && t[j] <= x + h {
        let k = table[(t[j] - x + h) as usize];
        num += y[j] * k;
        den += k;
        j += 1;
    }
    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

/// The ensemble at `x` with any times.
fn rbf_direct(t: &[f64], y: &[f64], x: f64, kernel: &Kernel) -> f64 {
    let h = kernel.reach as f64;
    let mut j = t.partition_point(|&v| v < x - h);
    let (mut num, mut den) = (0.0, 0.0);
    while j < t.len() && t[j] <= x + h {
        let k = kernel.combined(t[j] - x);
        num += y[j] * k;
        den += k;
        j += 1;
    }
    if den > 0.0 {
        num / den
    } else {
        f64::NAN
    }
}

// ---- harmonic ----------------------------------------------------------------

const MAXP: usize = 8;
/// GSL's default tuning constant of the bisquare weights and the iteration cap.
const TUNING: f64 = 4.685;
const MAX_ITER: usize = 100;

/// The design matrix of the harmonic model at the input and output times, in
/// centred and scaled time.
struct Design {
    p: usize,
    x_in: Vec<f64>,
    x_out: Vec<f64>,
}

impl Design {
    fn new(t_in: &[f64], t_out: &[f64], modes: usize, trend: bool, period: f64) -> Design {
        let p = 1 + usize::from(trend) + 2 * modes;
        let (a, b) = (t_in[0], t_in[t_in.len() - 1]);
        let centre = 0.5 * (a + b);
        let scale = (0.5 * (b - a)).max(1.0);
        let omega = 2.0 * std::f64::consts::PI / period;
        let rows = |ts: &[f64]| {
            let mut v = Vec::with_capacity(ts.len() * p);
            for &t in ts {
                let u = t - centre;
                v.push(1.0);
                if trend {
                    v.push(u / scale);
                }
                for m in 1..=modes {
                    let (s, c) = (omega * m as f64 * u).sin_cos();
                    v.push(c);
                    v.push(s);
                }
            }
            v
        };
        Design {
            p,
            x_in: rows(t_in),
            x_out: rows(t_out),
        }
    }
}


type Mat = [[f64; MAXP]; MAXP];

/// Weighted least squares of `y` on `rows` (n x p) into `c`, by the Cholesky
/// factor of the normal equations of the columns scaled to unit norm; leaves
/// the factor in `l` and the column scales in `d`. False when singular.
fn wls(
    rows: &[f64],
    y: &[f64],
    w: Option<&[f64]>,
    p: usize,
    c: &mut [f64; MAXP],
    l: &mut Mat,
    d: &mut [f64; MAXP],
) -> bool {
    let mut a: Mat = [[0.0; MAXP]; MAXP];
    let mut b = [0.0; MAXP];
    for (i, x) in rows.chunks_exact(p).enumerate() {
        let wi = w.map_or(1.0, |w| w[i]);
        if wi == 0.0 {
            continue;
        }
        let wy = wi * y[i];
        for r in 0..p {
            let wx = wi * x[r];
            b[r] += x[r] * wy;
            for s in 0..=r {
                a[r][s] += wx * x[s];
            }
        }
    }
    for r in 0..p {
        if !(a[r][r] > 0.0) {
            return false;
        }
        d[r] = 1.0 / a[r][r].sqrt();
    }
    for r in 0..p {
        b[r] *= d[r];
        for s in 0..=r {
            a[r][s] *= d[r] * d[s];
        }
    }
    for r in 0..p {
        for s in 0..=r {
            let mut v = a[r][s];
            for k in 0..s {
                v -= l[r][k] * l[s][k];
            }
            if r == s {
                if !(v > 1e-13) {
                    return false;
                }
                l[r][r] = v.sqrt();
            } else {
                l[r][s] = v / l[s][s];
            }
        }
    }
    // L z = b, then L^T s = z
    for r in 0..p {
        let mut v = b[r];
        for k in 0..r {
            v -= l[r][k] * b[k];
        }
        b[r] = v / l[r][r];
    }
    for r in (0..p).rev() {
        let mut v = b[r];
        for k in r + 1..p {
            v -= l[k][r] * b[k];
        }
        b[r] = v / l[r][r];
    }
    for r in 0..p {
        c[r] = b[r] * d[r];
    }
    true
}

/// `robust_madsigma`: the median of the absolute residuals without the
/// smallest p - 1, over 0.6745 (`buf` is scratch of the length of `r`).
fn mad_sigma(r: &[f64], p: usize, buf: &mut Vec<f64>) -> f64 {
    buf.clear();
    buf.extend(r.iter().map(|v| v.abs()));
    let m = r.len() - p + 1;
    let (lhs, rhs) = ((m - 1) / 2, m / 2);
    let kr = p - 1 + rhs;
    let (left, vr, _) = buf.select_nth_unstable_by(kr, f64::total_cmp);
    let vr = *vr;
    let median = if lhs == rhs {
        vr
    } else {
        0.5 * (left.iter().copied().fold(f64::NEG_INFINITY, f64::max) + vr)
    };
    median / 0.6745
}

/// `gsl_multifit_robust` with the bisquare weights on the observations
/// `w.idx` of `series`: the number of iterations (`MAX_ITER` when it did not
/// converge) and whether it converged, or None when the first, unweighted fit is singular. A later
/// singular fit (fewer than p observations with weight) stops the iteration at
/// the last good coefficients.
fn robust_fit<T: Sample>(
    series: &[T],
    design: &Design,
    w: &mut Work,
    c: &mut [f64; MAXP],
) -> Option<(usize, bool)> {
    let p = design.p;
    let Work {
        idx,
        rows,
        y,
        wts,
        res,
        fac,
        abs,
        ..
    } = w;
    let n = idx.len();
    rows.clear();
    y.clear();
    for &i in idx.iter() {
        rows.extend_from_slice(&design.x_in[i * p..(i + 1) * p]);
        y.push(series[i].to_f64());
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let sigy = (y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1) as f64).sqrt();
    // a lower bound of the scale: a very good first fit leaves no outliers to find
    let sig_lower = if sigy == 0.0 { 1.0 } else { 1.0e-6 * sigy };

    let mut l: Mat = [[0.0; MAXP]; MAXP];
    let mut d = [0.0; MAXP];
    if !wls(rows, y, None, p, c, &mut l, &mut d) {
        return None;
    }
    // the leverage of each observation, h = |L^-1 D x|^2, and 1 / sqrt(1 - h)
    fac.clear();
    for x in rows.chunks_exact(p) {
        let (mut z, mut h) = ([0.0; MAXP], 0.0);
        for r in 0..p {
            let mut v = x[r] * d[r];
            for k in 0..r {
                v -= l[r][k] * z[k];
            }
            z[r] = v / l[r][r];
            h += z[r] * z[r];
        }
        fac.push(1.0 / (1.0 - h.min(0.9999)).sqrt());
    }
    let residuals = |c: &[f64; MAXP], res: &mut Vec<f64>| {
        res.clear();
        res.extend(
            rows.chunks_exact(p)
                .zip(y.iter())
                .map(|(x, &yi)| yi - x.iter().zip(c).map(|(a, b)| a * b).sum::<f64>()),
        );
    };
    residuals(c, res);

    wts.clear();
    wts.resize(n, 0.0);
    let tol = f64::EPSILON.sqrt();
    let mut numit = 0;
    let mut converged = false;
    while !converged && numit < MAX_ITER {
        numit += 1;
        for (r, f) in res.iter_mut().zip(fac.iter()) {
            *r *= f;
        }
        let sig = mad_sigma(res, p, abs);
        let scale = 1.0 / (sig.max(sig_lower) * TUNING);
        for (wi, r) in wts.iter_mut().zip(res.iter()) {
            let u = r * scale;
            *wi = if u.abs() < 1.0 { (1.0 - u * u) * (1.0 - u * u) } else { 0.0 };
        }
        let prev = *c;
        if !wls(rows, y, Some(wts), p, c, &mut l, &mut d) {
            *c = prev;
            break;
        }
        residuals(c, res);
        converged = (0..p).all(|i| (c[i] - prev[i]).abs() <= tol * c[i].abs().max(prev[i].abs()));
    }
    Some((numit, converged))
}

/// How the robust fit of each series of `values` (pixels, n_in) ended, as
/// `interp_batch` runs it with the harmonic method: the iterations of the
/// reweighted least squares (1..=100), `u16::MAX` when it ran all 100 without
/// converging (a series that cycles: its result depends on where the cycle is
/// cut, so it is not a reproducible value), and 0 when it was not fit (fewer
/// observations than coefficients, or singular).
#[allow(clippy::too_many_arguments)]
pub fn harmonic_iterations<T: Sample>(
    values: &[T],
    n_pixels: usize,
    n_in: usize,
    t_in: &[f64],
    modes: usize,
    trend: bool,
    fit: (f64, f64),
    period: f64,
    n_jobs: i32,
) -> Result<Vec<u16>, String> {
    if values.len() != n_pixels * n_in || t_in.len() != n_in {
        return Err("values must be n_pixels x n_in, with one time for each step".into());
    }
    if !(1..=3).contains(&modes) || !(period > 0.0) || !period.is_finite() {
        return Err("modes must be 1, 2 or 3 and period a positive number of days".into());
    }
    if n_pixels == 0 || n_in == 0 {
        return Ok(vec![0; n_pixels]);
    }
    let design = Design::new(t_in, &[], modes, trend, period);
    let p = design.p;
    let in_fit: Vec<bool> = t_in.iter().map(|&t| t >= fit.0 && t <= fit.1).collect();
    let mut out = vec![0u16; n_pixels];
    parallel::for_each_row(&mut out, 1, n_jobs, Work::default, |w, px, row| {
        let series = &values[px * n_in..(px + 1) * n_in];
        w.idx.clear();
        w.idx
            .extend((0..n_in).filter(|&i| in_fit[i] && !series[i].to_f64().is_nan()));
        let mut c = [0.0; MAXP];
        if w.idx.len() >= p {
            if let Some((numit, converged)) = robust_fit(series, &design, w, &mut c) {
                row[0] = if converged { numit as u16 } else { u16::MAX };
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// 5 years every 5 days with gaps: times, one series with NaN.
    fn series(seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut s = seed;
        let t: Vec<f64> = (0..365).map(|i| 738000.0 + 5.0 * i as f64).collect();
        let y = t
            .iter()
            .map(|&x| {
                if lcg(&mut s) < 0.3 {
                    f64::NAN
                } else {
                    3000.0 + 2000.0 * (2.0 * std::f64::consts::PI * x / 365.0).sin() + 200.0 * lcg(&mut s)
                }
            })
            .collect();
        (t, y)
    }

    #[test]
    fn linear_is_np_interp() {
        let (t, y) = series(1);
        let valid: Vec<(f64, f64)> = t.iter().copied().zip(y.iter().copied()).filter(|p| !p.1.is_nan()).collect();
        let out_t: Vec<f64> = (-3..120).map(|i| 737990.0 + 16.0 * i as f64).collect();
        let got = interp_batch(&y, 1, t.len(), &t, &out_t, &Method::Linear, 1).unwrap();
        for (g, &x) in got.iter().zip(&out_t) {
            let r = valid.partition_point(|p| p.0 < x);
            let want = if r == 0 {
                valid[0].1
            } else if r == valid.len() {
                valid[r - 1].1
            } else if valid[r].0 == x {
                valid[r].1
            } else {
                let ((xl, yl), (xr, yr)) = (valid[r - 1], valid[r]);
                yl + (yr - yl) * (x - xl) / (xr - xl)
            };
            assert!((g - want).abs() < 1e-9, "{x}: {g} vs {want}");
        }
    }

    #[test]
    fn moving_window_is_closed() {
        let t = [0.0, 10.0, 20.0, 30.0];
        let y = [1.0, 2.0, f64::NAN, 4.0];
        let got = interp_batch(&y, 1, 4, &t, &[10.0, 20.0, 100.0], &Method::Moving { window: 10.0 }, 1).unwrap();
        assert_eq!(got[0], 1.5); // the observations at 0 and 10 (20 is missing)
        assert_eq!(got[1], 3.0); // 10 and 30 are both 10 days away
        assert!(got[2].is_nan());
    }

    /// FORCE's loop for each kernel, written out.
    fn rbf_by_kernel(t: &[f64], y: &[f64], x: f64, sigma: &[f64], cutoff: f64) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        let hbin = (4.0 * sigma[sigma.len() - 1]) as i64;
        for &s in sigma {
            let k = |d: f64| 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * s) * (-0.5 * d * d / (s * s)).exp();
            let (mut sum, mut max_ce, mut w) = (0.0f32, 0.0, 0.0);
            for d in 0..=hbin {
                sum += k(d as f64) as f32;
                if f64::from(sum) <= cutoff / 2.0 {
                    max_ce = d as f64;
                    w += f64::from(k(d as f64) as f32);
                }
            }
            w *= 2.0;
            let (mut syw, mut sw) = (0.0, 0.0);
            for (&ti, &yi) in t.iter().zip(y) {
                if yi.is_nan() || (ti - x).abs() > max_ce {
                    continue;
                }
                syw += yi * k(ti - x);
                sw += k(ti - x);
            }
            if sw > 0.0 {
                num += syw / w;
                den += sw / w;
            }
        }
        if den > 0.0 {
            num / den
        } else {
            f64::NAN
        }
    }

    #[test]
    fn rbf_is_the_ensemble_of_kernels() {
        let (t, y) = series(2);
        let sigma = vec![8.0, 16.0, 32.0];
        let out_t: Vec<f64> = (-4..125).map(|i| 737990.0 + 16.0 * i as f64).collect();
        let m = Method::Rbf {
            sigma: sigma.clone(),
            cutoff: 0.95,
        };
        let got = interp_batch(&y, 1, t.len(), &t, &out_t, &m, 1).unwrap();
        for (g, &x) in got.iter().zip(&out_t) {
            let want = rbf_by_kernel(&t, &y, x, &sigma, 0.95);
            assert!(
                (g.is_nan() && want.is_nan()) || (g - want).abs() < 1e-7 * want.abs().max(1.0),
                "{x}: {g} vs {want}"
            );
        }
        // fractional times take the direct path
        let half: Vec<f64> = out_t.iter().map(|x| x + 0.5).collect();
        let a = interp_batch(&y, 1, t.len(), &t, &half, &m, 1).unwrap();
        for (g, &x) in a.iter().zip(&half) {
            let want = rbf_by_kernel(&t, &y, x, &sigma, 0.95);
            assert!(
                (g.is_nan() && want.is_nan()) || (g - want).abs() < 1e-7 * want.abs().max(1.0),
                "{x}: {g} vs {want}"
            );
        }
    }

    /// The bisection of the valid observations, as the linear and moving kernels were first written.
    fn naive_linear(t: &[f64], y: &[f64], x: f64) -> f64 {
        let v: Vec<(f64, f64)> = t.iter().copied().zip(y.iter().copied()).filter(|p| !p.1.is_nan()).collect();
        let n = v.len();
        if n == 0 {
            return f64::NAN;
        }
        let r = v.partition_point(|p| p.0 < x);
        if r < n && v[r].0 == x {
            v[r].1
        } else if r == 0 {
            v[0].1
        } else if r == n {
            v[n - 1].1
        } else {
            let ((xl, yl), (xr, yr)) = (v[r - 1], v[r]);
            (yl * (xr - x) + yr * (x - xl)) / (xr - xl)
        }
    }

    fn naive_moving(t: &[f64], y: &[f64], x: f64, window: f64) -> f64 {
        let (mut sum, mut num) = (0.0, 0.0);
        for (&ti, &yi) in t.iter().zip(y) {
            if !yi.is_nan() && ti >= x - window && ti <= x + window {
                sum += yi;
                num += 1.0;
            }
        }
        if num > 0.0 {
            sum / num
        } else {
            f64::NAN
        }
    }

    #[test]
    fn linear_and_moving_are_the_bisection_and_the_window_sum_bit_for_bit() {
        // ties in t, a series without observations, one with a single one, new times in any order
        let mut s = 11;
        let n = 120;
        let mut t: Vec<f64> = (0..n).map(|i| 1000.0 + 5.0 * i as f64 + (lcg(&mut s) * 3.0).floor()).collect();
        t[40] = t[39];
        let mut out_t: Vec<f64> = (-5..60).map(|i| 990.0 + 11.5 * i as f64).collect();
        out_t.extend([t[3], t[39], 1e9, -1e9]);
        out_t.swap(2, 30);
        let mut cube: Vec<f64> = Vec::new();
        let mut shape: Vec<Vec<f64>> = Vec::new();
        for k in 0..6 {
            let y: Vec<f64> = (0..n)
                .map(|_| {
                    let gap = lcg(&mut s) < 0.4 + 0.1 * k as f64;
                    let v = 3000.0 * lcg(&mut s) - 500.0;
                    match (gap, k % 2) {
                        (true, _) => f64::NAN,
                        (false, 0) => v.round(), // integers: the prefix sums
                        _ => v,                  // fractions: the window summed in order
                    }
                })
                .collect();
            shape.push(y.clone());
            cube.extend(y);
        }
        shape.push(vec![f64::NAN; n]);
        cube.extend(vec![f64::NAN; n]);
        let mut one = vec![f64::NAN; n];
        one[17] = 42.0;
        shape.push(one.clone());
        cube.extend(one);
        let px = shape.len();
        let cube32: Vec<f32> = cube.iter().map(|&v| v as f32).collect();
        let back: Vec<f64> = cube32.iter().map(|&v| f64::from(v)).collect();
        let eq = |a: f64, b: f64| (a.is_nan() && b.is_nan()) || a == b;
        for jobs in [1, 3] {
            let lin = interp_batch(&cube, px, n, &t, &out_t, &Method::Linear, jobs).unwrap();
            let mov = interp_batch(&cube, px, n, &t, &out_t, &Method::Moving { window: 16.0 }, jobs).unwrap();
            let lin32 = interp_batch(&cube32, px, n, &t, &out_t, &Method::Linear, jobs).unwrap();
            for (p, y) in shape.iter().enumerate() {
                for (k, &x) in out_t.iter().enumerate() {
                    let i = p * out_t.len() + k;
                    assert!(eq(lin[i], naive_linear(&t, y, x)), "linear px {p} x {x}: {}", lin[i]);
                    assert!(eq(mov[i], naive_moving(&t, y, x, 16.0)), "moving px {p} x {x}: {}", mov[i]);
                    let yb = &back[p * n..(p + 1) * n];
                    assert!(eq(lin32[i], naive_linear(&t, yb, x)), "linear f32 px {p} x {x}");
                }
            }
        }
        let mov32 = interp_batch(&cube32, px, n, &t, &out_t, &Method::Moving { window: 16.0 }, 1).unwrap();
        for p in 0..px {
            for (k, &x) in out_t.iter().enumerate() {
                let want = naive_moving(&t, &back[p * n..(p + 1) * n], x, 16.0);
                assert!(eq(mov32[p * out_t.len() + k], want), "moving f32 px {p} x {x}");
            }
        }
    }

    #[test]
    fn harmonic_iterations_flag_a_fit_that_does_not_converge() {
        let t: Vec<f64> = (0..300).map(|i| 738000.0 + 6.0 * i as f64).collect();
        let y: Vec<f64> = t.iter().map(|&x| truth(x)).collect();
        let mut few = vec![f64::NAN; 300];
        few[..7].fill(1.0);
        let cube: Vec<f64> = y.iter().chain(&few).copied().collect();
        let it = harmonic_iterations(&cube, 2, 300, &t, 3, true, (0.0, 1e9), 365.0, 1).unwrap();
        assert!(it[0] >= 1 && it[0] < 100, "{it:?}");
        assert_eq!(it[1], 0);
    }

    fn truth(x: f64) -> f64 {
        let w = 2.0 * std::f64::consts::PI / 365.0;
        3000.0 + 0.5 * (x - 738500.0) + 2000.0 * (w * x).sin() + 700.0 * (2.0 * w * x).cos() - 300.0 * (3.0 * w * x).sin()
    }

    fn harmonic() -> Method {
        Method::Harmonic {
            modes: 3,
            trend: true,
            fit: (0.0, 1e9),
            period: 365.0,
        }
    }

    #[test]
    fn harmonic_recovers_a_noiseless_curve_and_extrapolates() {
        let t: Vec<f64> = (0..300).map(|i| 738000.0 + 6.0 * i as f64).collect();
        let y: Vec<f64> = t.iter().map(|&x| truth(x)).collect();
        let out_t = [738000.0, 738800.0, 740500.0];
        let got = interp_batch(&y, 1, 300, &t, &out_t, &harmonic(), 1).unwrap();
        for (g, &x) in got.iter().zip(&out_t) {
            assert!((g - truth(x)).abs() < 1e-6, "{x}: {g} vs {}", truth(x));
        }
    }

    #[test]
    fn harmonic_resists_outliers_and_needs_enough_data() {
        let mut s = 7;
        let t: Vec<f64> = (0..300).map(|i| 738000.0 + 6.0 * i as f64).collect();
        let mut y: Vec<f64> = t.iter().map(|&x| truth(x) + 20.0 * (lcg(&mut s) - 0.5)).collect();
        for i in (0..300).step_by(17) {
            y[i] -= 4000.0; // clouds
        }
        let got = interp_batch(&y, 1, 300, &t, &t, &harmonic(), 1).unwrap();
        let worst = got.iter().zip(&t).map(|(g, &x)| (g - truth(x)).abs()).fold(0.0, f64::max);
        assert!(worst < 60.0, "{worst}");
        // fewer observations than coefficients (8)
        let mut few = vec![f64::NAN; 300];
        for v in few.iter_mut().take(7) {
            *v = 1.0;
        }
        let got = interp_batch(&few, 1, 300, &t, &[738100.0], &harmonic(), 1).unwrap();
        assert!(got[0].is_nan());
    }
}
