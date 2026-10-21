//! LandTrendr (Kennedy et al. 2010): a port of src/landtrendr.cpp, itself a
//! port of fit_trajectory_v2.pro + tbcd_v2.pro (LandTrendr-2012) validated
//! vertex for vertex against the original IDL under GDL, and of ftv_v1.pro.
//! Same operations in the same order, so the same bits as the C++: the IDL
//! quirks it keeps (the single-precision p-value, the integer vertex values of
//! extend_to_edge, take_out_weakest2 editing the series) are kept here too.

// The loops index observations and vertices as the C++ does.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::cmath::lgamma;
use crate::parallel;

/// std::max(a, b): `a < b ? b : a` (NaN in `a` is kept, unlike f64::max).
#[inline]
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// std::min(a, b): `b < a ? b : a`.
#[inline]
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// The index of the first largest value (std::max_element).
fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[best] < v[i] {
            best = i;
        }
    }
    best
}

/// The index of the first smallest value (std::min_element).
fn argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] < v[best] {
            best = i;
        }
    }
    best
}

/// The parameters of LandTrendr (the C++'s LandTrendrParams).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub max_segments: i32,
    pub pval_threshold: f64,
    pub prevent_fast_recovery: bool,
    pub recovery_threshold: f64,
    pub spike_threshold: f64,
    pub best_model_proportion: f64,
    pub vertex_count_overshoot: i32,
    pub min_observations_needed: i32,
    pub modifier: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_segments: 6,
            pval_threshold: 0.05,
            prevent_fast_recovery: true,
            recovery_threshold: 0.25,
            spike_threshold: 0.9,
            best_model_proportion: 0.75,
            vertex_count_overshoot: 3,
            min_observations_needed: 6,
            modifier: 1.0,
        }
    }
}

/// A vertex of the fitted trajectory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub year: i32,
    pub value: f64,
}

/// The selected model's vertices and the RMSE of the fit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrajectoryResult {
    pub vertices: Vec<Vertex>,
    pub rmse: f64,
}

// ---- p-value of the F statistic ------------------------------------------

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1.0e-30 {
        d = 1.0e-30;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=1000 {
        let m2 = 2 * m;
        let (mf, m2f) = (m as f64, m2 as f64);
        let mut aa = mf * (b - mf) * x / ((qam + m2f) * (a + m2f));
        d = 1.0 + aa * d;
        if d.abs() < 1.0e-30 {
            d = 1.0e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1.0e-30 {
            c = 1.0e-30;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + mf) * (qab + mf) * x / ((a + m2f) * (qap + m2f));
        d = 1.0 + aa * d;
        if d.abs() < 1.0e-30 {
            d = 1.0e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1.0e-30 {
            c = 1.0e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1.0e-15 {
            break;
        }
    }
    h
}

fn betai(a: f64, b: f64, x: f64) -> f64 {
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn f_pval(f_stat: f64, df1: f64, df2: f64) -> f64 {
    if f_stat < 0.0 {
        return 1.0;
    }
    if df1 <= 0.0 || df2 <= 0.0 {
        return 1.0;
    }
    let x = df2 / (df2 + df1 * f_stat);
    betai(df2 / 2.0, df1 / 2.0, x)
}

// ---- desawtooth -------------------------------------------------------------

/// find_correction.pro's formula at one point: (correction, proportion).
#[inline]
fn correction_at(v: &[f64], i: usize) -> (f64, f64) {
    let diff_2 = (v[i - 1] - v[i + 1]).abs();
    let diff_minus1 = (v[i] - v[i + 1]).abs();
    let diff_plus1 = (v[i] - v[i - 1]).abs();
    let mut md = cmax(diff_minus1, diff_plus1);
    if md == 0.0 {
        md = diff_2; // diff_2 is then 0 too, so the proportion is 0
    }
    let prop = if md > 0.0 { 1.0 - diff_2 / md } else { 0.0 };
    (prop * (((v[i - 1] + v[i + 1]) / 2.0) - v[i]), prop)
}

/// desawtooth.pro: dampens spikes while the strongest proportional
/// correction is above `stopat` (the first correction is always applied).
pub fn desawtooth(vals: &[f64], stopat: f64) -> Vec<f64> {
    let mut v = vals.to_vec();
    let n = v.len();
    if n < 3 {
        return v;
    }
    let mut correction = vec![0.0; n];
    let mut prop_correction = vec![0.0; n];
    for i in 1..n - 1 {
        (correction[i], prop_correction[i]) = correction_at(&v, i);
    }
    let mut prop = 1.0;
    while prop > stopat {
        let mut wh_max = 0;
        let mut max_prop = prop_correction[0];
        for i in 1..n {
            if prop_correction[i] > max_prop {
                max_prop = prop_correction[i];
                wh_max = i;
            }
        }
        v[wh_max] += correction[wh_max];
        prop = max_prop;
        // only the entries reading v[wh_max] change
        let lo = wh_max.saturating_sub(2).max(1);
        let hi = (wh_max + 2).min(n - 2);
        for i in lo..=hi {
            (correction[i], prop_correction[i]) = correction_at(&v, i);
        }
    }
    v
}

// ---- vertex search ---------------------------------------------------------

fn angle_diff(x0: f64, x1: f64, x2: f64, y0: f64, y1: f64, y2: f64, yrange: f64, distweightfactor: f64) -> f64 {
    let ydiff2 = y2 - y1;
    let ydiff1 = y1 - y0;
    let angle1 = (ydiff1 / (x1 - x0)).atan();
    let angle2 = (ydiff2 / (x2 - x1)).atan();
    let scaler = cmax(0.0, (ydiff2 * distweightfactor) / yrange) + 1.0;
    cmax(angle1.abs(), angle2.abs()) * scaler
}

/// vet_verts: drops the vertices of least bend down to `desired_count`.
fn vet_verts(x: &[i32], y: &[f64], vertices: &[usize], desired_count: i64, distweightfactor: f64) -> Vec<usize> {
    let n_verts = vertices.len();
    let n_to_remove = n_verts as i64 - desired_count;
    if n_to_remove <= 0 || n_verts <= 3 {
        return vertices.to_vec();
    }
    let min_y = y[argmin(y)];
    let max_y = y[argmax(y)];
    let mut yr = max_y - min_y;
    if yr == 0.0 {
        yr = 1.0;
    }
    let mut range_x = (x[x.len() - 1] - x[0]) as f64;
    if range_x == 0.0 {
        range_x = 1.0;
    }
    let yscale: Vec<f64> = y.iter().map(|&v| ((v - min_y) / yr) * range_x).collect();
    let mut sc_yr = yscale[argmax(&yscale)] - yscale[argmin(&yscale)];
    if sc_yr == 0.0 {
        sc_yr = 1.0;
    }
    let mut v = vertices.to_vec();
    let ratio = |v: &[usize], i: usize| {
        angle_diff(
            x[v[i - 1]] as f64,
            x[v[i]] as f64,
            x[v[i + 1]] as f64,
            yscale[v[i - 1]],
            yscale[v[i]],
            yscale[v[i + 1]],
            sc_yr,
            distweightfactor,
        )
    };
    let mut slope_ratios: Vec<f64> = (1..n_verts - 1).map(|i| ratio(&v, i)).collect();
    let mut count = n_verts;
    for _ in 0..n_to_remove {
        let mut min_val = f64::MAX;
        let mut worst = None;
        for i in 0..count - 2 {
            if slope_ratios[i] < min_val {
                min_val = slope_ratios[i];
                worst = Some(i);
            }
        }
        let Some(worst) = worst else { break };
        let worst_idx = worst + 1; // in the vertex array
        v.remove(worst_idx);
        slope_ratios.remove(worst);
        count -= 1;
        if worst_idx != 1 {
            let left = worst_idx - 1;
            slope_ratios[left - 1] = ratio(&v, left);
        }
        if worst_idx != count - 1 {
            let right = worst_idx;
            slope_ratios[right - 1] = ratio(&v, right);
        }
    }
    v
}

/// OLS of y over x on [lo, hi]: the fitted values into `fitted` (if given) and the SSE.
fn regress_range(x: &[i32], y: &[f64], lo: usize, hi: usize, mut fitted: Option<&mut Vec<f64>>) -> f64 {
    let m = (hi - lo + 1) as f64;
    let x0 = x[lo] as f64;
    let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) = (0.0, 0.0, 0.0, 0.0);
    for i in lo..=hi {
        let dx = x[i] as f64 - x0;
        sum_x += dx;
        sum_y += y[i];
        sum_xx += dx * dx;
        sum_xy += dx * y[i];
    }
    let denom = m * sum_xx - sum_x * sum_x;
    let (mut slope, mut intercept) = (0.0, sum_y / m);
    if denom.abs() > 1e-9 {
        slope = (m * sum_xy - sum_x * sum_y) / denom;
        intercept = (sum_y - slope * sum_x) / m;
    }
    if let Some(f) = fitted.as_mut() {
        f.clear();
    }
    let mut sse = 0.0;
    for i in lo..=hi {
        let f = intercept + slope * (x[i] as f64 - x0);
        if let Some(fv) = fitted.as_mut() {
            fv.push(f);
        }
        let err = y[i] - f;
        sse += err * err;
    }
    sse
}

/// split_series: the point of largest residual inside [lo, hi], or None.
fn split_series(x: &[i32], y: &[f64], lo: usize, hi: usize, is_end_segment: bool, disttest: bool) -> Option<usize> {
    let mut fitted = Vec::with_capacity(hi - lo + 1);
    regress_range(x, y, lo, hi, Some(&mut fitted));
    let m = hi - lo + 1;
    let mut diff: Vec<f64> = (0..m).map(|i| (y[lo + i] - fitted[i]).abs()).collect();
    diff[0] = 0.0;
    diff[m - 1] = 0.0;
    if disttest && is_end_segment && m >= 3 && !(y[hi] > y[hi - 1]) {
        diff[m - 2] = 0.0;
    }
    let mut best_idx = 0;
    let mut best_val = diff[0];
    for i in 1..m {
        if diff[i] > best_val {
            best_val = diff[i];
            best_idx = i;
        }
    }
    if best_idx == 0 {
        None
    } else {
        Some(lo + best_idx)
    }
}

/// find_vertices: recursive splitting of the segment of largest SSE / span.
fn find_vertices(x: &[i32], y: &[f64], target_count: i64, distweightfactor: f64) -> Vec<usize> {
    let n = x.len();
    let m = target_count.min(n as i64 - 2);
    let mut verts = vec![0, n - 1];
    let disttest = distweightfactor != 0.0;
    let mut count = 0;
    while (verts.len() as i64) < m {
        let nseg = verts.len() - 1;
        let mut mses = vec![0.0; nseg];
        for s in 0..nseg {
            let (lo, hi) = (verts[s], verts[s + 1]);
            let span = (hi - lo + 1) as f64;
            if span > 2.0 {
                mses[s] = regress_range(x, y, lo, hi, None) / span;
            }
        }
        let split_at = loop {
            let s = argmax(&mses);
            if mses[s] <= 0.0 {
                break None;
            }
            let is_end_segment = s == nseg - 1;
            if let Some(c) = split_series(x, y, verts[s], verts[s + 1], is_end_segment, disttest) {
                break Some(c);
            }
            mses[s] = 0.0; // rejected: the next-worst segment
        };
        let Some(split_at) = split_at else { break };
        verts.push(split_at);
        verts.sort_unstable();
        count += 1;
        if count > 20 {
            break;
        }
    }
    verts
}

// ---- fitting ---------------------------------------------------------------

/// Gauss-Jordan inversion of the n x n row-major `a` in place; false if singular.
fn invert_matrix(a: &mut Vec<f64>, n: usize) -> bool {
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for i in 0..n {
        let mut max_el = a[i * n + i].abs();
        let mut pivot = i;
        for k in i + 1..n {
            if a[k * n + i].abs() > max_el {
                max_el = a[k * n + i].abs();
                pivot = k;
            }
        }
        if max_el == 0.0 {
            return false;
        }
        if pivot != i {
            for j in 0..n {
                a.swap(i * n + j, pivot * n + j);
                inv.swap(i * n + j, pivot * n + j);
            }
        }
        let diag = a[i * n + i];
        for j in 0..n {
            a[i * n + j] /= diag;
            inv[i * n + j] /= diag;
        }
        for k in 0..n {
            if k != i {
                let factor = a[k * n + i];
                for j in 0..n {
                    a[k * n + j] -= factor * a[i * n + j];
                    inv[k * n + j] -= factor * inv[i * n + j];
                }
            }
        }
    }
    *a = inv;
    true
}

/// The vertex values of the piecewise-linear OLS fit with fixed breakpoints.
fn fit_piecewise_ols(x: &[i32], y: &[f64], verts: &[usize]) -> Vec<f64> {
    let (n, k) = (x.len(), verts.len());
    let mut xm = vec![0.0; n * k];
    for i in 0..n {
        let xi = x[i];
        for j in 0..k {
            let vj = x[verts[j]];
            if j > 0 && xi >= x[verts[j - 1]] && xi <= vj {
                let v_prev = x[verts[j - 1]];
                if vj > v_prev {
                    xm[i * k + j] = (xi - v_prev) as f64 / (vj - v_prev) as f64;
                }
            } else if j + 1 < k && xi >= vj && xi <= x[verts[j + 1]] {
                let v_next = x[verts[j + 1]];
                if v_next > vj {
                    xm[i * k + j] = (v_next - xi) as f64 / (v_next - vj) as f64;
                }
            } else if xi == vj {
                xm[i * k + j] = 1.0;
            }
        }
    }
    let mut xtx = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for r in 0..n {
                s += xm[r * k + i] * xm[r * k + j];
            }
            xtx[i * k + j] = s;
        }
    }
    if !invert_matrix(&mut xtx, k) {
        // the values of the observations at the vertices
        return verts.iter().map(|&v| y[v]).collect();
    }
    let mut xty = vec![0.0; k];
    for i in 0..k {
        let mut s = 0.0;
        for r in 0..n {
            s += xm[r * k + i] * y[r];
        }
        xty[i] = s;
    }
    (0..k)
        .map(|i| {
            let mut s = 0.0;
            for j in 0..k {
                s += xtx[i * k + j] * xty[j];
            }
            s
        })
        .collect()
}

/// find_best_trace: per segment, early to late, the better of point-to-point
/// and (anchored) regression; the slope of every segment into `slopes`.
fn fit_piecewise_sequential(x: &[i32], y: &[f64], verts: &[usize], mut slopes: Option<&mut Vec<f64>>) -> Vec<f64> {
    let k = verts.len();
    let mut fitted = vec![0.0; k];
    if let Some(s) = slopes.as_mut() {
        s.clear();
        s.resize(k.saturating_sub(1), 0.0);
    }
    if k < 2 {
        if k == 1 {
            fitted[0] = y[verts[0]];
        }
        return fitted;
    }
    for j in 0..k - 1 {
        let (i0, i1) = (verts[j], verts[j + 1]);
        let (x0, x1) = (x[i0], x[i1]);
        let span = (x1 - x0) as f64;

        let p2p_y0 = if j == 0 { y[verts[j]] } else { fitted[j] };
        let p2p_y1 = y[verts[j + 1]];
        let p2p_slope = if span > 0.0 { (p2p_y1 - p2p_y0) / span } else { 0.0 };
        let mut p2p_sse = 0.0;
        for i in i0..=i1 {
            let err = y[i] - (p2p_y0 + p2p_slope * (x[i] - x0) as f64);
            p2p_sse += err * err;
        }

        let (reg_y0, reg_slope);
        if j == 0 {
            let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) = (0.0, 0.0, 0.0, 0.0);
            let m = (i1 - i0 + 1) as f64;
            for i in i0..=i1 {
                let dx = (x[i] - x0) as f64;
                sum_x += dx;
                sum_y += y[i];
                sum_xx += dx * dx;
                sum_xy += dx * y[i];
            }
            let denom = m * sum_xx - sum_x * sum_x;
            if denom.abs() > 1e-9 {
                reg_slope = (m * sum_xy - sum_x * sum_y) / denom;
                reg_y0 = (sum_y - reg_slope * sum_x) / m;
            } else {
                reg_slope = p2p_slope;
                reg_y0 = p2p_y0;
            }
        } else {
            reg_y0 = fitted[j];
            let (mut num, mut den) = (0.0, 0.0);
            for i in i0..=i1 {
                let dx = (x[i] - x0) as f64;
                num += dx * (y[i] - reg_y0);
                den += dx * dx;
            }
            reg_slope = if den > 1e-9 { num / den } else { p2p_slope };
        }
        let reg_y1 = reg_y0 + reg_slope * span;
        let mut reg_sse = 0.0;
        for i in i0..=i1 {
            let err = y[i] - (reg_y0 + reg_slope * (x[i] - x0) as f64);
            reg_sse += err * err;
        }

        if reg_sse < p2p_sse {
            fitted[j] = reg_y0;
            fitted[j + 1] = reg_y1;
            if let Some(s) = slopes.as_mut() {
                s[j] = reg_slope;
            }
        } else {
            fitted[j] = p2p_y0;
            fitted[j + 1] = p2p_y0 + p2p_slope * span;
            if let Some(s) = slopes.as_mut() {
                s[j] = p2p_slope;
            }
        }
    }
    fitted
}

/// take_out_weakest: removes the interior vertex whose local straight line
/// fits best (SSE / span; the first minimum).
fn take_out_weakest(x: &[i32], y: &[f64], verts: &[usize], vertvals: &[f64]) -> Vec<usize> {
    let k = verts.len();
    if k <= 2 {
        return verts.to_vec();
    }
    let mut best_mse = f64::MAX;
    let mut best_remove = None;
    for i in 1..k - 1 {
        let (i0, i1) = (verts[i - 1], verts[i + 1]);
        let span = (x[i1] - x[i0]) as f64;
        if span <= 0.0 {
            continue;
        }
        let slope = (vertvals[i + 1] - vertvals[i - 1]) / span;
        let mut mse = 0.0;
        for j in i0..=i1 {
            let fitted_y = vertvals[i - 1] + slope * (x[j] - x[i0]) as f64;
            let err = y[j] - fitted_y;
            mse += err * err;
        }
        mse /= span;
        if mse < best_mse {
            best_mse = mse;
            best_remove = Some(i);
        }
    }
    let mut result = verts.to_vec();
    if let Some(i) = best_remove {
        result.remove(i);
    }
    result
}

/// One rung of tbcd_v2.pro's model ladder, in modifier-space.
struct Candidate {
    verts: Vec<usize>,
    vertvals: Vec<f64>,
    slopes: Vec<f64>,
    f_stat: f64,
    pval: f64,
}

/// fill_from_vertices.pro: the piecewise-linear fitted value at every observation.
fn interpolate_fit(x: &[i32], verts: &[usize], vertvals: &[f64]) -> Vec<f64> {
    let mut yfit = vec![0.0; x.len()];
    for j in 0..verts.len().saturating_sub(1) {
        let (i0, i1) = (verts[j], verts[j + 1]);
        let span = (x[i1] - x[i0]) as f64;
        let slope = if span > 0.0 {
            (vertvals[j + 1] - vertvals[j]) / span
        } else {
            0.0
        };
        for i in i0..=i1 {
            yfit[i] = vertvals[j] + slope * (x[i] - x[i0]) as f64;
        }
    }
    yfit
}

/// calc_fitting_stats3.pro: (F, p), the p-value rounded through float32 as
/// the original's `1 - f_test1(...)`.
fn calc_fitting_stats(y: &[f64], yfit: &[f64], n_predictors: i64) -> (f64, f64) {
    let n = y.len();
    let mut mean_y = 0.0;
    for &v in y {
        mean_y += v;
    }
    mean_y /= n as f64;
    let (mut ss, mut ss_resid) = (0.0, 0.0);
    for i in 0..n {
        ss += (y[i] - mean_y) * (y[i] - mean_y);
        ss_resid += (y[i] - yfit[i]) * (y[i] - yfit[i]);
    }
    if ss_resid > ss {
        ss_resid = ss; // rounding guard when there is no trend
    }
    let df_regr = n_predictors;
    let df_resid = n as i64 - n_predictors - 1;
    if df_regr <= 0 || df_resid <= 0 {
        return (0.0, 1.0);
    }
    let ms_regr = (ss - ss_resid) / df_regr as f64;
    let ms_resid = ss_resid / df_resid as f64;
    // "because of glitch in f_test1, a zero mistakenly gets f score of 1"
    let f_stat = if ms_regr < 0.00001 { 0.00001 } else { ms_regr / ms_resid };
    let p_upper = f_pval(f_stat, df_regr as f64, df_resid as f64);
    let cdf = (1.0 - p_upper) as f32;
    (f_stat, (1.0f32 - cdf) as f64)
}

fn value_range(v: &[f64]) -> f64 {
    v[argmax(v)] - v[argmin(v)]
}

/// check_slopes.pro: no recovery segment steeper than `threshold` of the range.
fn check_slopes(m: &Candidate, threshold: f64) -> bool {
    let range_of_vals = value_range(&m.vertvals);
    !m.slopes.iter().any(|&s| s < 0.0 && s.abs() / range_of_vals > threshold)
}

/// take_out_weakest2.pro: drops (or flattens) the vertex ending the fastest
/// recovery when it exceeds `threshold`, editing the observation under it in
/// `y` (by reference, as the original); else take_out_weakest.
fn take_out_weakest2(info: &Candidate, threshold: f64, x: &[i32], y: &mut [f64]) -> Vec<usize> {
    let (verts, mut vertvals) = (&info.verts, info.vertvals.clone());
    let k = verts.len();
    let range_of_vals = value_range(&info.vertvals);
    let mut violator = None;
    let mut worst = -1.0;
    for (s, &slope) in info.slopes.iter().enumerate() {
        if slope < 0.0 && slope != -1.0 {
            let scaled = slope.abs() / range_of_vals;
            if scaled > worst {
                worst = scaled;
                violator = Some(s);
            }
        }
    }
    if let Some(v) = violator {
        if worst > threshold {
            let vi = v + 1;
            let idx = verts[vi];
            if vi == k - 1 {
                y[idx] = y[idx - 1];
                vertvals[k - 1] = y[idx];
            } else {
                let slope = (y[idx + 1] - y[idx - 1]) / (x[idx + 1] - x[idx - 1]) as f64;
                y[idx] = (x[idx] - x[idx - 1]) as f64 * slope + y[idx - 1];
                let mut result = verts.clone();
                result.remove(vi);
                return result;
            }
        }
    }
    take_out_weakest(x, y, verts, &vertvals)
}

/// The selection loop around pick_best_model6 + check_slopes.
fn select_model(
    ladder: &mut [Candidate],
    recovery_threshold: f64,
    best_model_proportion: f64,
    n_vertices_orig: usize,
    is_f7: bool,
) -> usize {
    let fstats: Vec<f64> = ladder.iter().map(|m| m.f_stat).collect();
    let mut best = 0;
    let mut increment = 0;
    let mut notdone = true;
    while notdone {
        increment += 1;
        let mut mn = ladder[0].pval;
        for m in ladder.iter() {
            mn = cmin(mn, m.pval);
        }
        let thr = (2.0 - best_model_proportion) * mn;
        let picked = ladder.iter().position(|m| m.pval <= thr);
        if let Some(p) = picked {
            best = p;
            let ok = check_slopes(&ladder[best], recovery_threshold);
            if !ok {
                ladder[best].pval = 1.0;
            }
            notdone = !ok && increment <= n_vertices_orig;
        } else {
            if is_f7 {
                best = ladder.len() - 1;
                ladder[best].pval = 1.0;
            } else {
                best = argmin(&fstats);
            }
            notdone = false;
        }
    }
    best
}

/// tbcd_v2.pro's flat vertex at the start (end) of the year range, then the
/// interior vertex of least bend dropped past max_count vertices, on the
/// vertex values truncated to integers as the original's intarr (and with an
/// integer division in the "other end" block).
fn extend_to_edge(idx: &mut Vec<usize>, vals: &mut Vec<f64>, front: bool, n_all: usize, max_count: i64) {
    if front {
        idx.insert(0, 0);
        vals.insert(0, vals[0]);
    } else {
        idx.push(n_all - 1);
        vals.push(vals[vals.len() - 1]);
    }
    let nv = idx.len();
    if nv as i64 <= max_count {
        return;
    }
    let tv: Vec<i64> = vals.iter().map(|v| v.trunc() as i64).collect();
    let mut sc_yr = tv.iter().max().expect("vertices") - tv.iter().min().expect("vertices");
    if sc_yr == 0 {
        sc_yr = 1;
    }
    let mut minv = None;
    let mut min_ratio = f64::MAX;
    for i in 1..nv - 1 {
        let ydiff1 = tv[i] - tv[i - 1];
        let ydiff2 = tv[i + 1] - tv[i];
        let angle1 = (ydiff1 as f64 / (idx[i] as i64 - idx[i - 1] as i64) as f64).atan();
        let angle2 = (ydiff2 as f64 / (idx[i + 1] as i64 - idx[i] as i64) as f64).atan();
        let weight = if front {
            cmax(0.0, (ydiff2 as f64 * 2.0) / sc_yr as f64)
        } else {
            ((ydiff2 * 2) / sc_yr).max(0) as f64
        };
        let r = cmax(angle1.abs(), angle2.abs()) * (weight + 1.0);
        if r < min_ratio {
            min_ratio = r;
            minv = Some(i);
        }
    }
    if let Some(m) = minv {
        idx.remove(m);
        vals.remove(m);
    }
}

/// LandTrendr on one pixel's series (NaN observations left out).
pub fn fit_trajectory(years: &[i32], values: &[f64], p: &Params) -> TrajectoryResult {
    let mut out = TrajectoryResult::default();
    let n_all = years.len();
    if n_all == 0 || values.len() != n_all {
        return out;
    }
    let goods: Vec<usize> = (0..n_all).filter(|&i| !values[i].is_nan()).collect();
    let n = goods.len();
    if n == 0 {
        return out;
    }
    // too few observations: the raw trajectory, unsegmented (RMSE left at 0)
    if (n as i64) < (p.min_observations_needed as i64).max(2) {
        out.vertices = goods
            .iter()
            .map(|&g| Vertex {
                year: years[g],
                value: values[g],
            })
            .collect();
        return out;
    }
    let x: Vec<i32> = goods.iter().map(|&g| years[g]).collect();
    let raw: Vec<f64> = goods.iter().map(|&g| values[g]).collect();

    // 1. desawtooth, then into modifier-space
    let mut y = if p.spike_threshold < 1.0 {
        desawtooth(&raw, p.spike_threshold)
    } else {
        raw.clone()
    };
    for v in y.iter_mut() {
        *v *= p.modifier;
    }

    // 2-3. candidate vertices
    let max_count = p.max_segments as i64 + 1;
    let overshoot_count = max_count + (p.vertex_count_overshoot as i64).max(0);
    let orig_v = vet_verts(&x, &y, &find_vertices(&x, &y, overshoot_count, 2.0), max_count, 2.0);
    let n_vertices = orig_v.len();

    let make_f6 = |v: Vec<usize>, y: &[f64]| {
        let mut slopes = Vec::new();
        let vertvals = fit_piecewise_sequential(&x, y, &v, Some(&mut slopes));
        let (f_stat, pval) = calc_fitting_stats(y, &interpolate_fit(&x, &v, &vertvals), 2 * v.len() as i64 - 2);
        Candidate {
            verts: v,
            vertvals,
            slopes,
            f_stat,
            pval,
        }
    };
    let make_f7 = |v: Vec<usize>, y: &[f64]| {
        let vertvals = fit_piecewise_ols(&x, y, &v);
        let slopes = (0..v.len() - 1)
            .map(|s| (vertvals[s + 1] - vertvals[s]) / (v[s + 1] - v[s]) as f64)
            .collect();
        let (f_stat, pval) = calc_fitting_stats(y, &interpolate_fit(&x, &v, &vertvals), 2 * v.len() as i64 - 2);
        Candidate {
            verts: v,
            vertvals,
            slopes,
            f_stat,
            pval,
        }
    };

    // 4. F6 ladder
    let mut ladder = vec![make_f6(orig_v.clone(), &y)];
    for _ in 1..=n_vertices.saturating_sub(2) {
        let v = take_out_weakest2(ladder.last().expect("a rung"), p.recovery_threshold, &x, &mut y);
        ladder.push(make_f6(v, &y));
    }
    let mut best = select_model(
        &mut ladder,
        p.recovery_threshold,
        p.best_model_proportion,
        n_vertices,
        false,
    );

    // 5. not significant: the F7 ladder (joint fit, plain take_out_weakest)
    if ladder[best].pval > p.pval_threshold {
        ladder.clear();
        ladder.push(make_f7(orig_v.clone(), &y));
        for _ in 1..=n_vertices.saturating_sub(2) {
            let prev = ladder.last().expect("a rung");
            let v = take_out_weakest(&x, &y, &prev.verts, &prev.vertvals);
            ladder.push(make_f7(v, &y));
        }
        best = select_model(
            &mut ladder,
            p.recovery_threshold,
            p.best_model_proportion,
            n_vertices,
            true,
        );
    }
    let chosen = &ladder[best];

    // 6. in all-year terms
    let (out_idx, out_vals) = if chosen.pval > p.pval_threshold {
        let mut mean_y = 0.0;
        for &v in &y {
            mean_y += v;
        }
        mean_y /= n as f64;
        (vec![0, n_all - 1], vec![mean_y, mean_y])
    } else {
        let mut idx: Vec<usize> = chosen.verts.iter().map(|&v| goods[v]).collect();
        let mut vals = chosen.vertvals.clone();
        if goods[0] != 0 {
            extend_to_edge(&mut idx, &mut vals, true, n_all, max_count);
        }
        if goods[n - 1] != n_all - 1 {
            extend_to_edge(&mut idx, &mut vals, false, n_all, max_count);
        }
        (idx, vals)
    };
    out.vertices = out_idx
        .iter()
        .zip(&out_vals)
        .map(|(&i, &v)| Vertex {
            year: years[i],
            value: v * p.modifier,
        })
        .collect();

    // RMSE of the output trajectory against every valid raw observation
    let all_fit = interpolate_fit(years, &out_idx, &out_vals);
    let mut sse = 0.0;
    for i in 0..n {
        let err = raw[i] * p.modifier - all_fit[goods[i]];
        sse += err * err;
    }
    let n_params = out_idx.len() as i64;
    out.rmse = (sse / (n as i64 - n_params).max(1) as f64).sqrt();
    out
}

/// ftv_v1.pro's flat edge vertex (float vertex values here).
fn extend_ftv_to_edge(idx: &mut Vec<usize>, vals: &mut Vec<f64>, front: bool, n_all: usize, max_count: i64) {
    if front {
        idx.insert(0, 0);
        vals.insert(0, vals[0]);
    } else {
        idx.push(n_all - 1);
        vals.push(vals[vals.len() - 1]);
    }
    let nv = idx.len();
    if nv as i64 <= max_count {
        return;
    }
    let mut sc_yr = value_range(vals);
    if sc_yr == 0.0 {
        sc_yr = 1.0;
    }
    let mut minv = None;
    let mut min_ratio = f64::MAX;
    for i in 1..nv - 1 {
        let r = angle_diff(
            idx[i - 1] as f64,
            idx[i] as f64,
            idx[i + 1] as f64,
            vals[i - 1],
            vals[i],
            vals[i + 1],
            sc_yr,
            2.0,
        );
        if r < min_ratio {
            min_ratio = r;
            minv = Some(i);
        }
    }
    if let Some(m) = minv {
        idx.remove(m);
        vals.remove(m);
    }
}

/// Fitted-to-vertices (apply_fitted_trajectory_v1.pro + ftv_v1.pro): the
/// fitted value of every year of another band on the segmentation's vertex
/// years; all NaN without observations or vertices.
pub fn fit_to_vertices(years: &[i32], values: &[f64], vertex_years: &[i32], spike_threshold: f64) -> Vec<f64> {
    let n_all = years.len();
    let mut yfit = vec![f64::NAN; n_all];
    let goods: Vec<usize> = (0..n_all).filter(|&i| !values[i].is_nan()).collect();
    let (n, k) = (goods.len(), vertex_years.len());
    if n == 0 || k == 0 {
        return yfit;
    }
    let x: Vec<i32> = goods.iter().map(|&g| years[g]).collect();
    let raw: Vec<f64> = goods.iter().map(|&g| values[g]).collect();
    let y = if spike_threshold < 1.0 {
        desawtooth(&raw, spike_threshold)
    } else {
        raw
    };

    let mut v: Vec<usize> = Vec::new();
    for &vy in vertex_years {
        let (mut at, mut before, mut after) = (None, None, None);
        for j in 0..n {
            if x[j] == vy {
                at = Some(j);
            }
            if x[j] < vy {
                before = Some(j);
            }
            if x[j] > vy && after.is_none() {
                after = Some(j);
            }
        }
        if at.is_none() {
            // the last observation before it, or the first after when there is none
            // before or that one is taken already
            at = before;
            if before.is_none() || (!v.is_empty() && at == v.last().copied() && at != Some(n - 1)) {
                at = after;
            }
        }
        if let Some(a) = at {
            v.push(a);
        }
    }
    v.sort_unstable();
    v.dedup();

    let (out_idx, out_vals) = if v.len() < 2 {
        let mut mean_y = 0.0;
        for &val in &y {
            mean_y += val;
        }
        mean_y /= n as f64;
        (vec![0, n_all - 1], vec![mean_y, mean_y])
    } else {
        let mut vals = fit_piecewise_sequential(&x, &y, &v, None);
        let mut idx: Vec<usize> = v.iter().map(|&vi| goods[vi]).collect();
        if goods[0] != 0 {
            extend_ftv_to_edge(&mut idx, &mut vals, true, n_all, k as i64);
        }
        if goods[n - 1] != n_all - 1 {
            extend_ftv_to_edge(&mut idx, &mut vals, false, n_all, k as i64);
        }
        (idx, vals)
    };
    if out_idx[0] == out_idx[out_idx.len() - 1] {
        yfit.fill(out_vals[0]); // a single year
        return yfit;
    }
    let mut fit = interpolate_fit(years, &out_idx, &out_vals);
    let (first, last) = (out_idx[0], out_idx[out_idx.len() - 1]);
    for (i, f) in fit.iter_mut().enumerate() {
        if i < first || i > last {
            *f = f64::NAN;
        }
    }
    fit
}

/// fit_to_vertices on every pixel: `values` (pixels x T), `vertex_years`
/// (pixels x max_vertices, `counts` of them valid): pixels x T, NaN where nothing.
pub fn fit_to_vertices_batch(
    values: &[f64],
    years: &[i32],
    vertex_years: &[i32],
    counts: &[i32],
    max_vertices: usize,
    spike_threshold: f64,
    n_jobs: i32,
) -> Vec<f64> {
    let (times, pixels) = (years.len(), counts.len());
    let mut out = vec![f64::NAN; pixels * times];
    parallel::for_each_row(
        &mut out,
        times,
        n_jobs,
        || (),
        |_, p, row| {
            let nv = (counts[p] as i64).min(max_vertices as i64);
            if nv <= 0 {
                return;
            }
            let verts = &vertex_years[p * max_vertices..p * max_vertices + nv as usize];
            let fit = fit_to_vertices(years, &values[p * times..(p + 1) * times], verts, spike_threshold);
            row.copy_from_slice(&fit);
        },
    );
    out
}

/// LandTrendr on every pixel of `values` (pixels x T): (vertices pixels x
/// max_vertices x 2 of (year, value), `no_data` past each count; counts, the
/// full number of vertices of each pixel; rmse). `no_data` and NaN values are
/// left out of the fits.
pub fn fit_trajectory_batch(
    values: &[f64],
    pixels: usize,
    years: &[i32],
    p: &Params,
    no_data: f64,
    n_jobs: i32,
) -> (Vec<f64>, Vec<i32>, Vec<f64>) {
    let times = years.len();
    let max_vertices = (p.max_segments as i64 + 1).max(0) as usize;
    let width = max_vertices * 2 + 2; // vertices, then count and rmse
    let mut per_pixel = vec![0.0; pixels * width];
    parallel::for_each_row(
        &mut per_pixel,
        width,
        n_jobs,
        || vec![0.0; times],
        |pixel: &mut Vec<f64>, px, row| {
            row[..2 * max_vertices].fill(no_data);
            let mut has_valid = false;
            for t in 0..times {
                let v = values[px * times + t];
                if v == no_data || v.is_nan() {
                    pixel[t] = f64::NAN;
                } else {
                    pixel[t] = v;
                    has_valid = true;
                }
            }
            if !has_valid {
                return;
            }
            let r = fit_trajectory(years, pixel, p);
            for (i, vert) in r.vertices.iter().take(max_vertices).enumerate() {
                row[2 * i] = vert.year as f64;
                row[2 * i + 1] = vert.value;
            }
            row[2 * max_vertices] = r.vertices.len() as f64;
            row[2 * max_vertices + 1] = r.rmse;
        },
    );
    let mut vertices = Vec::with_capacity(pixels * 2 * max_vertices);
    let mut counts = Vec::with_capacity(pixels);
    let mut rmse = Vec::with_capacity(pixels);
    for row in per_pixel.chunks_exact(width) {
        vertices.extend_from_slice(&row[..2 * max_vertices]);
        counts.push(row[2 * max_vertices] as i32);
        rmse.push(row[2 * max_vertices + 1]);
    }
    (vertices, counts, rmse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_disturbance() {
        let years: Vec<i32> = (1985..2025).collect();
        let values: Vec<f64> = years
            .iter()
            .map(|&y| {
                if y < 2000 {
                    0.8
                } else if y < 2002 {
                    0.3
                } else {
                    0.3 + 0.03 * (y - 2002) as f64
                }
            })
            .map(|v| v + 0.002 * ((v * 1000.0) as i64 % 7) as f64)
            .collect();
        let p = Params {
            modifier: -1.0,
            ..Params::default()
        };
        let r = fit_trajectory(&years, &values, &p);
        assert!(r.vertices.len() >= 3, "{r:?}");
        assert!(r.vertices.iter().any(|v| (1999..=2000).contains(&v.year)), "{r:?}");
    }

    #[test]
    fn desawtooth_removes_a_spike() {
        let d = desawtooth(&[1.0, 1.0, 5.0, 1.0, 1.0], 0.9);
        assert!(d[2] < 5.0);
    }

    #[test]
    fn short_series_pass_through() {
        let r = fit_trajectory(&[2000, 2001, 2002], &[1.0, f64::NAN, 3.0], &Params::default());
        assert_eq!(r.vertices.len(), 2);
        assert_eq!(r.rmse, 0.0);
    }
}
