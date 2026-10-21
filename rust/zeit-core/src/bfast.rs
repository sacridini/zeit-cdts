//! bfast (Verbesselt et al. 2010), the classic iterative trend and season
//! break detection: a port of src/bfast.cpp, same operations in the same order
//! (Eigen's products and LDLT, the STL seed, strucchangeRcpp's OLS-MOSUM test
//! and BIC-selected breakpoints), so the same bits.

// The dynamic program and the tables are indexed as the C++ does.
#![allow(clippy::needless_range_loop)]

use crate::bfast_lite::recursive_cum_rss;
use crate::eigen::ldlt::{xtx, Ldlt};
use crate::eigen::qr::{mat_t_vec, mat_vec, residual, sum_seq, Mat, Order};
use crate::eigen::sum_by;
use crate::{parallel, stl};
use std::f64::consts::PI;

/// The result of one series (the C++'s BFResult).
#[derive(Clone, Debug, PartialEq)]
pub struct BfResult {
    pub n_trend_breaks: f64,
    pub n_season_breaks: f64,
    /// The largest trend jump, 0 without a trend break.
    pub magnitude: f64,
    /// Fractional-year time of the largest trend jump.
    pub time: f64,
    pub n_iter: f64,
    pub n_valid: f64,
    pub valid: f64,
    pub trend_breakpoint_idx: Vec<f64>,
    pub season_breakpoint_idx: Vec<f64>,
    pub trend_magnitude: Vec<f64>,
    pub trend_first_after_idx: Vec<f64>,
}

impl Default for BfResult {
    fn default() -> Self {
        BfResult {
            n_trend_breaks: f64::NAN,
            n_season_breaks: f64::NAN,
            magnitude: f64::NAN,
            time: f64::NAN,
            n_iter: f64::NAN,
            n_valid: f64::NAN,
            valid: 0.0,
            trend_breakpoint_idx: Vec::new(),
            season_breakpoint_idx: Vec::new(),
            trend_magnitude: Vec::new(),
            trend_first_after_idx: Vec::new(),
        }
    }
}

/// The `~ trend` design matrix: intercept and 1, 2, ...
fn trend_design(n_time: usize) -> Mat {
    Mat::from_fn(n_time, 2, Order::Col, |i, j| if j == 0 { 1.0 } else { (i + 1) as f64 })
}

/// The `~ harmonic` design matrix (no intercept): `order` cosines then sines,
/// the Nyquist sine dropped when 2 order == frequency.
fn harmonic_design(n_time: usize, start_time: f64, frequency: i32, order: i32) -> Mat {
    let order = order.min(frequency);
    let drop_nyquist = 2 * order == frequency;
    let n_cols = (2 * order - drop_nyquist as i32).max(0) as usize;
    let mut x = Mat::zeros(n_time, n_cols, Order::Col);
    for i in 0..n_time {
        let t = start_time + i as f64 / frequency as f64;
        let mut col = 0;
        for k in 1..=order {
            x.set(i, col, (2.0 * PI * t * k as f64).cos());
            col += 1;
        }
        for k in 1..=order {
            if drop_nyquist && k == order {
                continue;
            }
            x.set(i, col, (2.0 * PI * t * k as f64).sin());
            col += 1;
        }
    }
    x
}

/// BIC of m breaks (strucchangeRcpp's BIC.breakpointsfull).
fn bic_score(rss: f64, n: usize, n_regressors: usize, m_breaks: usize) -> f64 {
    let n_d = n as f64;
    let logl = n_d * ((rss / n_d).ln() + 1.0 + (2.0 * PI).ln());
    let df = (n_regressors + 1) as f64 * (m_breaks + 1) as f64;
    logl + n_d.ln() * df
}

/// strucchangeRcpp's `sc.me` (Brownian bridge increments, functional "max"):
/// [k - 1][h row][significance column], as src/bfast.cpp holds it.
#[rustfmt::skip]
const SC_ME: [[[f64; 4]; 10]; 6] = [
    [[0.7552, 0.8017, 0.8444, 0.8977], [0.9809, 1.0483, 1.1119, 1.1888], [1.1211, 1.2059, 1.2845, 1.3767],
     [1.2170, 1.3158, 1.4053, 1.5131], [1.2811, 1.3920, 1.4917, 1.6118], [1.3258, 1.4448, 1.5548, 1.6863],
     [1.3514, 1.4789, 1.5946, 1.7339], [1.3628, 1.4956, 1.6152, 1.7572], [1.3610, 1.4976, 1.6210, 1.7676],
     [1.3751, 1.5115, 1.6341, 1.7808]],
    [[0.7997, 0.8431, 0.8838, 0.9351], [1.0448, 1.1067, 1.1654, 1.2388], [1.2030, 1.2805, 1.3509, 1.4362],
     [1.3112, 1.4042, 1.4881, 1.5876], [1.3870, 1.4865, 1.5779, 1.6930], [1.4422, 1.5538, 1.6530, 1.7724],
     [1.4707, 1.5900, 1.6953, 1.8223], [1.4892, 1.6105, 1.7206, 1.8559], [1.4902, 1.6156, 1.7297, 1.8668],
     [1.5067, 1.6319, 1.7455, 1.8827]],
    [[0.8250, 0.8668, 0.9040, 0.9519], [1.0802, 1.1419, 1.1986, 1.2700], [1.2491, 1.3259, 1.3951, 1.4820],
     [1.3647, 1.4516, 1.5326, 1.6302], [1.4449, 1.5421, 1.6322, 1.7470], [1.5045, 1.6089, 1.7008, 1.8143],
     [1.5353, 1.6560, 1.7510, 1.8756], [1.5588, 1.6751, 1.7809, 1.9105], [1.5630, 1.6828, 1.7901, 1.9190],
     [1.5785, 1.6981, 1.8071, 1.9395]],
    [[0.8414, 0.8828, 0.9205, 0.9681], [1.1066, 1.1663, 1.2217, 1.2918], [1.2792, 1.3533, 1.4212, 1.5013],
     [1.3973, 1.4506, 1.5593, 1.6536], [1.4852, 1.5791, 1.6690, 1.7741], [1.5429, 1.6465, 1.7420, 1.8573],
     [1.5852, 1.6927, 1.7941, 1.9140], [1.6057, 1.7195, 1.8212, 1.9450], [1.6089, 1.7245, 1.8269, 1.9592],
     [1.6275, 1.7435, 1.8495, 1.9787]],
    [[0.8541, 0.8948, 0.9321, 0.9799], [1.1247, 1.1846, 1.2395, 1.3088], [1.3040, 1.3765, 1.4440, 1.5252],
     [1.4250, 1.5069, 1.5855, 1.6791], [1.5154, 1.6077, 1.6921, 1.7967], [1.5738, 1.6770, 1.7687, 1.8837],
     [1.6182, 1.7217, 1.8176, 1.9377], [1.6460, 1.7540, 1.8553, 1.9788], [1.6462, 1.7574, 1.8615, 1.9897],
     [1.6644, 1.7777, 1.8816, 2.0085]],
    [[0.8653, 0.9048, 0.9414, 0.9880], [1.1415, 1.1997, 1.2530, 1.6220], [1.3223, 1.3938, 1.4596, 1.5392],
     [1.4483, 1.5305, 1.6100, 1.7014], [1.5392, 1.6317, 1.7139, 1.8154], [1.6025, 1.7018, 1.7930, 1.9061],
     [1.6462, 1.7499, 1.8439, 1.9605], [1.6697, 1.7769, 1.8763, 1.9986], [1.6802, 1.7889, 1.8932, 2.0163],
     [1.6939, 1.8052, 1.9074, 2.0326]],
];

/// R's approx(x, y, xout, rule = 2): linear, clamped to the ends.
fn approx_rule2(xs: &[f64], ys: &[f64], xout: f64) -> f64 {
    let n = xs.len();
    if xout <= xs[0] {
        return ys[0];
    }
    if xout >= xs[n - 1] {
        return ys[n - 1];
    }
    for i in 0..n - 1 {
        if xout >= xs[i] && xout <= xs[i + 1] {
            let frac = (xout - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + frac * (ys[i + 1] - ys[i]);
        }
    }
    ys[n - 1]
}

/// pvalue.efp(stat, "Brownian bridge increments", functional "max", h, k).
fn pvalue_bbi_max(stat: f64, k: usize, h: f64) -> f64 {
    let k = k.clamp(1, 6);
    const TABLE_H: [f64; 10] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50];
    const TABLE_P: [f64; 4] = [0.1, 0.05, 0.025, 0.01];
    let mut crit = [0.0; 4];
    for col in 0..4 {
        let vals: Vec<f64> = (0..10).map(|row| SC_ME[k - 1][row][col]).collect();
        crit[col] = approx_rule2(&TABLE_H, &vals, h);
    }
    let xs = [0.0, crit[0], crit[1], crit[2], crit[3]];
    let ys = [1.0, TABLE_P[0], TABLE_P[1], TABLE_P[2], TABLE_P[3]];
    approx_rule2(&xs, &ys, stat)
}

/// sctest(efp(y ~ X, h, type = "OLS-MOSUM"))$p.value of the whole fit.
fn ols_mosum_pvalue(x: &Mat, y: &[f64], h: f64) -> f64 {
    let (n, k) = (x.rows, x.cols);
    let nh = (h * n as f64).floor() as i64;
    if n <= k || nh < 1 || nh > n as i64 {
        return 1.0;
    }
    let nh = nh as usize;
    let df = n - k;
    let beta = Ldlt::new(&xtx(x, n)).solve(&mat_t_vec(x, y));
    let e = residual(x, &beta, y);
    let mean_e = sum_by(n, |i| e[i]) / n as f64;
    let sigma = (sum_by(n, |i| {
        let d = e[i] - mean_e;
        d * d
    }) / df as f64)
        .sqrt();
    if !(sigma > 0.0) || !sigma.is_finite() {
        return 1.0;
    }
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + e[i];
    }
    let denom = sigma * (n as f64).sqrt();
    let mut max_abs = 0.0f64;
    for j in 0..=n - nh {
        let w = ((cum[nh + j] - cum[j]) / denom).abs();
        if max_abs < w {
            max_abs = w;
        }
    }
    pvalue_bbi_max(max_abs, k, h)
}

/// One segmented-regression fit: the BIC-optimal breaks, the fitted values
/// and the coefficients of every segment.
#[derive(Default)]
struct SegFit {
    breakpoints: Vec<usize>,
    fitted: Vec<f64>,
    seg_coeffs: Vec<Vec<f64>>,
}

/// The OLS fit of rows [a, b] of x / y: its coefficients, its fitted values
/// into `fitted`, and its RSS.
fn fit_range(x: &Mat, y: &[f64], a: usize, b: usize, fitted: &mut [f64]) -> (Vec<f64>, f64) {
    let len = b - a + 1;
    let xs = Mat::from_data(len, x.cols, Order::Col, {
        let mut d = Vec::with_capacity(len * x.cols);
        for j in 0..x.cols {
            d.extend_from_slice(&x.data[j * x.rows + a..j * x.rows + a + len]);
        }
        d
    });
    let ys = &y[a..=b];
    let beta = Ldlt::new(&xtx(&xs, len)).solve(&mat_t_vec(&xs, ys));
    let fit = mat_vec(&xs, &beta);
    fitted[a..=b].copy_from_slice(&fit);
    let rss = sum_by(len, |r| {
        let d = ys[r] - fit[r];
        d * d
    });
    (beta, rss)
}

fn fit_segmented(x: &Mat, y: &[f64], h: f64, max_breaks_output: i32, skip_search: bool) -> SegFit {
    let (n, k) = (x.rows, x.cols);
    let mut out = SegFit {
        fitted: vec![0.0; n],
        ..SegFit::default()
    };
    let single = |out: &mut SegFit| {
        let (beta, _) = fit_range(x, y, 0, n - 1, &mut out.fitted);
        out.seg_coeffs.push(beta);
    };
    if skip_search {
        // R's `nobp` path when the OLS-MOSUM test does not reject stability
        single(&mut out);
        return out;
    }
    let hh = (h * n as f64).floor() as i64;
    if !(hh > k as i64 && hh <= (n / 2) as i64) {
        single(&mut out);
        return out;
    }
    let hh = hh as usize;
    let rss_triang: Vec<Vec<f64>> = (0..=n - hh).map(|s| recursive_cum_rss(x, y, s, n, k)).collect();
    let rss = |i: usize, j: usize| rss_triang[i][j - i - k];

    let max_m = (max_breaks_output as i64)
        .min((n as f64 / hh as f64).ceil() as i64 - 2)
        .max(0) as usize;
    let mut f = vec![vec![f64::NAN; n]; max_m + 1];
    let mut ptr = vec![vec![usize::MAX; n]; max_m + 1];
    for i in hh - 1..n {
        f[0][i] = rss(0, i);
    }
    let mut max_m_reached = 0;
    for m in 1..=max_m {
        let (i_lo, i_hi) = ((m + 1) * hh - 1, n - 1);
        if i_lo > i_hi {
            break;
        }
        let mut any = false;
        for i in i_lo..=i_hi {
            let (j_lo, j_hi) = (m * hh - 1, i - hh);
            let mut best = f64::INFINITY;
            let mut best_j = None;
            for j in j_lo..=j_hi {
                if f[m - 1][j].is_nan() {
                    continue;
                }
                let val = f[m - 1][j] + rss(j + 1, i);
                if val < best {
                    best = val;
                    best_j = Some(j);
                }
            }
            if let Some(j) = best_j {
                f[m][i] = best;
                ptr[m][i] = j;
                any = true;
            }
        }
        if !any {
            break;
        }
        max_m_reached = m;
    }

    let mut best_bic = bic_score(rss(0, n - 1), n, k, 0);
    let mut best_breaks: Vec<usize> = Vec::new();
    for m in 1..=max_m_reached {
        if f[m][n - 1].is_nan() {
            continue;
        }
        let bic_m = bic_score(f[m][n - 1], n, k, m);
        if bic_m < best_bic {
            best_bic = bic_m;
            let mut breaks = vec![0; m];
            let mut cur = n - 1;
            for level in (1..=m).rev() {
                cur = ptr[level][cur];
                breaks[level - 1] = cur;
            }
            best_breaks = breaks;
        }
    }
    if best_breaks.is_empty() {
        single(&mut out);
        return out;
    }
    let mut seg_start = 0;
    for s in 0..=best_breaks.len() {
        let seg_end = if s < best_breaks.len() { best_breaks[s] } else { n - 1 };
        let (beta, _) = fit_range(x, y, seg_start, seg_end, &mut out.fitted);
        out.seg_coeffs.push(beta);
        seg_start = seg_end + 1;
    }
    out.breakpoints = best_breaks;
    out
}

/// Linear interpolation of the inner NaN runs, the nearest value at the ends
/// (the STL seed's complete grid). At least one value must be finite.
fn linear_interpolate_nan(v: &mut [f64]) {
    let n = v.len();
    let Some(first) = (0..n).find(|&i| !v[i].is_nan()) else {
        return;
    };
    let last = (0..n).rev().find(|&i| !v[i].is_nan()).expect("a value");
    for i in 0..first {
        v[i] = v[first];
    }
    for i in last + 1..n {
        v[i] = v[last];
    }
    let mut prev = first;
    let mut i = first + 1;
    while i <= last {
        if !v[i].is_nan() {
            prev = i;
            i += 1;
            continue;
        }
        let mut next = i;
        while next <= last && v[next].is_nan() {
            next += 1;
        }
        let (v0, v1) = (v[prev], v[next]);
        for j in prev + 1..next {
            v[j] = v0 + (v1 - v0) * (j - prev) as f64 / (next - prev) as f64;
        }
        prev = next;
        i = next + 1;
    }
}

/// The parameters of a bfast run.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub start_time: f64,
    pub frequency: i32,
    pub h: f64,
    pub max_breaks_trend: i32,
    pub max_breaks_season: i32,
    pub max_iter: i32,
    pub level: f64,
    pub min_valid: i32,
}

fn bfast_impl(y_raw: &[f64], p: &Params, xtrend_full: &Mat, xharm_full: &Mat) -> BfResult {
    let mut res = BfResult::default();
    let n_raw = y_raw.len();
    if p.frequency < 2 || n_raw as i64 <= 2 * p.frequency as i64 {
        return res;
    }
    let valid_rows: Vec<usize> = (0..n_raw).filter(|&i| !y_raw[i].is_nan()).collect();
    let n = valid_rows.len();
    res.n_valid = n as f64;
    if (n as i64) < p.min_valid as i64 || n as i64 <= 2 * p.frequency as i64 {
        return res;
    }
    res.valid = 1.0;

    // the STL seed of the season, on the interpolated full grid
    let mut y_interp = y_raw.to_vec();
    linear_interpolate_nan(&mut y_interp);
    let st_full = stl::periodic_seasonal(&y_interp, p.frequency).expect("checked above");

    let xt = Mat::from_fn(n, 2, Order::Col, |r, j| xtrend_full.get(valid_rows[r], j));
    let xh = Mat::from_fn(n, xharm_full.cols, Order::Col, |r, j| xharm_full.get(valid_rows[r], j));
    let yv: Vec<f64> = valid_rows.iter().map(|&i| y_raw[i]).collect();
    let mut st: Vec<f64> = valid_rows.iter().map(|&i| st_full[i]).collect();

    let (mut prev_trend, mut prev_season): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
    let (mut trend_fit, mut season_fit) = (SegFit::default(), SegFit::default());
    let mut n_iter = 0;
    for iter in 0..p.max_iter.max(0) {
        let vt: Vec<f64> = (0..n).map(|r| yv[r] - st[r]).collect();
        let trend_stable = ols_mosum_pvalue(&xt, &vt, p.h) > p.level;
        trend_fit = fit_segmented(&xt, &vt, p.h, p.max_breaks_trend, trend_stable);
        let wt: Vec<f64> = (0..n).map(|r| yv[r] - trend_fit.fitted[r]).collect();
        let season_stable = ols_mosum_pvalue(&xh, &wt, p.h) > p.level;
        season_fit = fit_segmented(&xh, &wt, p.h, p.max_breaks_season, season_stable);
        st.copy_from_slice(&season_fit.fitted);
        n_iter = iter + 1;
        let converged = trend_fit.breakpoints == prev_trend && season_fit.breakpoints == prev_season;
        prev_trend = trend_fit.breakpoints.clone();
        prev_season = season_fit.breakpoints.clone();
        if converged {
            break;
        }
    }

    res.n_trend_breaks = trend_fit.breakpoints.len() as f64;
    res.n_season_breaks = season_fit.breakpoints.len() as f64;
    res.n_iter = n_iter as f64;
    res.trend_breakpoint_idx = trend_fit.breakpoints.iter().map(|&b| valid_rows[b] as f64).collect();
    res.season_breakpoint_idx = season_fit.breakpoints.iter().map(|&b| valid_rows[b] as f64).collect();

    // the largest trend jump: the right segment's fit at its first point minus
    // the left one's at its last (R bfast's Mag / Time)
    if !trend_fit.breakpoints.is_empty() {
        let (mut best_jump, mut best_row) = (0.0f64, 0usize);
        for (i, &end_left) in trend_fit.breakpoints.iter().enumerate() {
            let start_right = end_left + 1;
            let (bl, br) = (&trend_fit.seg_coeffs[i], &trend_fit.seg_coeffs[i + 1]);
            let y1 = sum_seq(2, |j| xt.get(end_left, j) * bl[j]);
            let y2 = sum_seq(2, |j| xt.get(start_right, j) * br[j]);
            let jump = y2 - y1;
            res.trend_magnitude.push(jump);
            res.trend_first_after_idx.push(valid_rows[start_right] as f64);
            if i == 0 || jump.abs() > best_jump.abs() {
                best_jump = jump;
                best_row = valid_rows[end_left];
            }
        }
        res.magnitude = best_jump;
        res.time = p.start_time + best_row as f64 / p.frequency as f64;
    } else {
        res.magnitude = 0.0;
        res.time = f64::NAN;
    }
    res
}

/// bfast of one series (unit-testing helper; min_valid 20).
pub fn bfast(y: &[f64], order: i32, p: &Params) -> BfResult {
    let xt = trend_design(y.len());
    let xh = harmonic_design(y.len(), p.start_time, p.frequency, order);
    bfast_impl(y, &Params { min_valid: 20, ..*p }, &xt, &xh)
}

/// Rows of the batch's output.
pub fn n_metrics(max_breaks_trend: usize, max_breaks_season: usize) -> usize {
    7 + 3 * max_breaks_trend + max_breaks_season
}

/// bfast on every row of `values` (`n_pixels` x T): n_metrics x n_pixels
/// (n_trend_breaks, n_season_breaks, magnitude, time, n_iter, n_valid, valid,
/// then the trend break indices, the season break indices, the trend jumps
/// and the first indices after the trend breaks), NaN where nothing.
pub fn batch(values: &[f64], n_pixels: usize, n_time: usize, order: i32, p: &Params, n_jobs: i32) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time);
    let xt = trend_design(n_time);
    let xh = harmonic_design(n_time, p.start_time, p.frequency, order);
    let (mbt, mbs) = (p.max_breaks_trend.max(0) as usize, p.max_breaks_season.max(0) as usize);
    let width = n_metrics(mbt, mbs);
    let mut per_pixel = vec![f64::NAN; n_pixels * width];
    parallel::for_each_row(
        &mut per_pixel,
        width,
        n_jobs,
        || (),
        |_, px, out| {
            let r = bfast_impl(&values[px * n_time..(px + 1) * n_time], p, &xt, &xh);
            out[..7].copy_from_slice(&[
                r.n_trend_breaks,
                r.n_season_breaks,
                r.magnitude,
                r.time,
                r.n_iter,
                r.n_valid,
                r.valid,
            ]);
            for b in 0..r.trend_breakpoint_idx.len().min(mbt) {
                out[7 + b] = r.trend_breakpoint_idx[b];
                out[7 + mbt + mbs + b] = r.trend_magnitude[b];
                out[7 + 2 * mbt + mbs + b] = r.trend_first_after_idx[b];
            }
            for b in 0..r.season_breakpoint_idx.len().min(mbs) {
                out[7 + mbt + b] = r.season_breakpoint_idx[b];
            }
        },
    );
    parallel::transpose(&per_pixel, n_pixels, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_trend_break() {
        let y: Vec<f64> = (0..23 * 6)
            .map(|i| {
                let t = 2000.0 + i as f64 / 23.0;
                0.6 + 0.2 * (2.0 * PI * t).sin() + 0.005 * ((i * 37 % 11) as f64 - 5.0)
                    - if i >= 70 { 0.3 } else { 0.0 }
            })
            .collect();
        let p = Params {
            start_time: 2000.0,
            frequency: 23,
            h: 0.15,
            max_breaks_trend: 5,
            max_breaks_season: 5,
            max_iter: 10,
            level: 0.05,
            min_valid: 20,
        };
        let r = bfast(&y, 3, &p);
        assert_eq!(r.valid, 1.0);
        assert!(r.n_trend_breaks >= 1.0, "{r:?}");
        assert!((r.magnitude + 0.3).abs() < 0.1, "{r:?}");
    }
}
