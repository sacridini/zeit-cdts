//! bfastmonitor (Verbesselt et al.; strucchangeRcpp's OLS-MOSUM monitoring,
//! Chu, Stinchcombe & White 1996), `history = "all"`: a port of
//! src/bfast_monitor.cpp, same operations in the same order (Eigen's X^T X,
//! LDLT and products included), so the same bits.

use crate::eigen::ldlt::{xtx, Ldlt};
use crate::eigen::qr::{mat_t_vec, mat_vec, sum_seq, Mat, Order};
use crate::eigen::sum_by;
use crate::parallel;
use std::f64::consts::{E, PI};

mod table;

const WINSIZE_GRID: [f64; 3] = [0.25, 0.5, 1.0];
const PERIOD_GRID: [i32; 5] = [2, 4, 6, 8, 10];
const ALPHA_GRID_N: usize = 50;

/// The result of one series (the C++'s BFMResult).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BfmResult {
    /// Fractional-year time of the first break, or NaN.
    pub breakpoint: f64,
    /// 0-based index into the series, or NaN.
    pub breakpoint_idx: f64,
    /// Median residual over the monitoring period.
    pub magnitude: f64,
    /// Residual standard error of the history fit.
    pub sigma: f64,
    /// Valid observations in the history fit.
    pub n_history: f64,
    pub has_break: f64,
    pub valid: f64,
}

impl Default for BfmResult {
    fn default() -> Self {
        BfmResult {
            breakpoint: f64::NAN,
            breakpoint_idx: f64::NAN,
            magnitude: f64::NAN,
            sigma: f64::NAN,
            n_history: f64::NAN,
            has_break: 0.0,
            valid: 0.0,
        }
    }
}

/// Values per pixel of the batch.
pub const N_METRICS: usize = 7;

/// The critical value of h, period and alpha: h and period on the table's
/// grid (as strucchangeRcpp requires), 1 - alpha linearly interpolated and
/// clamped (R's approx()).
pub fn critval(h: f64, period: i32, alpha: f64) -> Result<f64, &'static str> {
    let wi = (0..3)
        .rev()
        .find(|&i| (h - WINSIZE_GRID[i]).abs() < 1e-9)
        .ok_or("h must be one of 0.25, 0.5, 1.0 (bfastmonitor OLS-MOSUM critical-value table grid)")?;
    let pi = (0..5)
        .rev()
        .find(|&i| period == PERIOD_GRID[i])
        .ok_or("period must be one of 2, 4, 6, 8, 10 (bfastmonitor OLS-MOSUM critical-value table grid)")?;
    let t = &table::CRITVAL[wi][pi];
    let target = 1.0 - alpha;
    let (lo, step) = (0.95, 0.001);
    let pos = (target - lo) / step;
    if pos <= 0.0 {
        return Ok(t[0]);
    }
    if pos >= (ALPHA_GRID_N - 1) as f64 {
        return Ok(t[ALPHA_GRID_N - 1]);
    }
    let k0 = pos.floor() as usize;
    let frac = pos - k0 as f64;
    Ok(t[k0] * (1.0 - frac) + t[k0 + 1] * frac)
}

fn log_plus(x: f64) -> f64 {
    if x <= E {
        1.0
    } else {
        x.ln()
    }
}

/// The OLS-MOSUM boundary: critval * sqrt(2 logPlus(k / histsize)).
fn mosum_boundary(k_over_histsize: f64, critval: f64) -> f64 {
    critval * (2.0 * log_plus(k_over_histsize)).sqrt()
}

/// The design matrix of bfastpp's `response ~ trend + harmon`: intercept,
/// trend (1, 2, ...), `order` cosines then sines at start_time + i / frequency,
/// the Nyquist sine dropped when 2 order == frequency. Column-major.
pub fn design_matrix(n_time: usize, start_time: f64, frequency: i32, order: i32) -> Mat {
    let order = order.min(frequency);
    let drop_nyquist = 2 * order == frequency;
    let n_cols = (2 + 2 * order - drop_nyquist as i32) as usize;
    let mut x = Mat::zeros(n_time, n_cols, Order::Col);
    for i in 0..n_time {
        let t = start_time + i as f64 / frequency as f64;
        x.set(i, 0, 1.0);
        x.set(i, 1, (i + 1) as f64);
        let mut col = 2;
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

/// The design matrix in both storage orders: columns for the products, rows
/// for `X.row(i).dot(beta)`.
pub(crate) struct Design {
    pub x: Mat,
    rows: Vec<f64>,
}

impl Design {
    pub fn new(x: Mat) -> Design {
        let rows = (0..x.rows * x.cols).map(|e| x.get(e / x.cols, e % x.cols)).collect();
        Design { x, rows }
    }

    /// `X.row(i).dot(beta)`: a strided row and a dynamic vector, one term at a time.
    #[inline]
    fn row_dot(&self, i: usize, beta: &[f64]) -> f64 {
        let p = self.x.cols;
        let r = &self.rows[i * p..(i + 1) * p];
        sum_seq(p, |j| r[j] * beta[j])
    }

    /// The rows `idx` of X, column-major.
    fn gather(&self, idx: &[usize]) -> Mat {
        let (n, p) = (idx.len(), self.x.cols);
        let mut data = Vec::with_capacity(n * p);
        for j in 0..p {
            let col = &self.x.data[j * self.x.rows..(j + 1) * self.x.rows];
            data.extend(idx.iter().map(|&i| col[i]));
        }
        Mat::from_data(n, p, Order::Col, data)
    }
}

/// bfastmonitor on one series `y` (NaN skipped) with the design matrix `x`
/// and the critical value `crit`.
#[allow(clippy::too_many_arguments)]
fn monitor(
    y: &[f64],
    design: &Design,
    start_time: f64,
    monitor_start_time: f64,
    frequency: i32,
    h: f64,
    crit: f64,
    min_valid: i32,
) -> BfmResult {
    let mut res = BfmResult::default();
    let p = design.x.cols;
    let n_time = y.len();
    let (mut hist, mut mon) = (Vec::with_capacity(n_time), Vec::with_capacity(n_time));
    for (i, &v) in y.iter().enumerate() {
        let t = start_time + i as f64 / frequency as f64;
        if v.is_nan() {
            continue;
        }
        if t < monitor_start_time {
            hist.push(i);
        } else {
            mon.push(i);
        }
    }
    let n_hist = hist.len();
    res.n_history = n_hist as f64;
    let k = (h * n_hist as f64).floor() as i64;
    if !(k > 1 && n_hist > p && n_hist as i64 >= min_valid as i64) {
        return res;
    }
    let k = k as usize;
    res.valid = 1.0;

    // OLS on the history: LDLT of X^T X
    let xh = design.gather(&hist);
    let yh: Vec<f64> = hist.iter().map(|&i| y[i]).collect();
    let beta = Ldlt::new(&xtx(&xh, n_hist)).solve(&mat_t_vec(&xh, &yh));
    let fitted = mat_vec(&xh, &beta);
    let df = n_hist - p;
    let sigma = (sum_by(n_hist, |r| {
        let d = yh[r] - fitted[r];
        d * d
    }) / df as f64)
        .sqrt();
    res.sigma = sigma;
    if !(sigma > 0.0) || !sigma.is_finite() {
        return res;
    }

    let combined: Vec<usize> = hist.iter().chain(mon.iter()).copied().collect();
    let n_total = combined.len();
    if k > n_total {
        return res;
    }
    let e: Vec<f64> = combined.iter().map(|&i| y[i] - design.row_dot(i, &beta)).collect();

    let full_len = n_total - k + 1;
    let hist_len = n_hist - k + 1;
    let denom = sigma * (n_hist as f64).sqrt();
    let mut running = 0.0;
    for &v in &e[..k] {
        running += v;
    }
    let mut found = false;
    for j in 0..full_len {
        if j > 0 {
            running += e[j + k - 1] - e[j - 1];
        }
        if j >= hist_len && !found {
            let process = running / denom;
            let k1 = n_hist + 1 + (j - hist_len);
            let boundary = mosum_boundary(k1 as f64 / n_hist as f64, crit);
            if process.is_finite() && process.abs() > boundary {
                found = true;
                res.has_break = 1.0;
                let row = combined[k1 - 1];
                res.breakpoint_idx = row as f64;
                res.breakpoint = start_time + row as f64 / frequency as f64;
            }
        }
    }

    // magnitude: the median residual over the monitoring period
    let mut tmp: Vec<f64> = mon.iter().map(|&i| y[i] - design.row_dot(i, &beta)).collect();
    if !tmp.is_empty() {
        let mid = tmp.len() / 2;
        let odd = tmp.len() % 2 == 1;
        let (lower, &mut hi, _) = tmp.select_nth_unstable_by(mid, f64::total_cmp);
        res.magnitude = if odd {
            hi
        } else {
            let lo = lower
                .iter()
                .copied()
                .fold(lower[0], |m, v| if v.total_cmp(&m).is_gt() { v } else { m });
            0.5 * (lo + hi)
        };
    }
    res
}

/// bfastmonitor on one series (unit-testing helper; min_valid 10).
#[allow(clippy::too_many_arguments)]
pub fn bfast_monitor(
    y: &[f64],
    start_time: f64,
    monitor_start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    period: i32,
    alpha: f64,
) -> Result<BfmResult, &'static str> {
    let design = Design::new(design_matrix(y.len(), start_time, frequency, order));
    let crit = critval(h, period, alpha)?;
    Ok(monitor(
        y,
        &design,
        start_time,
        monitor_start_time,
        frequency,
        h,
        crit,
        10,
    ))
}

/// bfastmonitor on every row of `values` (`n_pixels` x T, row-major):
/// N_METRICS x n_pixels (breakpoint, breakpoint_idx, magnitude, sigma,
/// n_history, has_break, valid).
#[allow(clippy::too_many_arguments)]
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    start_time: f64,
    monitor_start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    period: i32,
    alpha: f64,
    min_valid: i32,
    n_jobs: i32,
) -> Result<Vec<f64>, &'static str> {
    assert_eq!(values.len(), n_pixels * n_time);
    let design = Design::new(design_matrix(n_time, start_time, frequency, order));
    let crit = critval(h, period, alpha)?;
    let mut per_pixel = vec![f64::NAN; n_pixels * N_METRICS];
    parallel::for_each_row(
        &mut per_pixel,
        N_METRICS,
        n_jobs,
        || (),
        |_, p, out| {
            let y = &values[p * n_time..(p + 1) * n_time];
            let r = monitor(
                y,
                &design,
                start_time,
                monitor_start_time,
                frequency,
                h,
                crit,
                min_valid,
            );
            out.copy_from_slice(&[
                r.breakpoint,
                r.breakpoint_idx,
                r.magnitude,
                r.sigma,
                r.n_history,
                r.has_break,
                r.valid,
            ]);
        },
    );
    Ok(parallel::transpose(&per_pixel, n_pixels, N_METRICS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_a_drop_in_the_monitoring_period() {
        let freq = 23;
        let y: Vec<f64> = (0..23 * 8)
            .map(|i| {
                let t = 2000.0 + i as f64 / 23.0;
                0.6 + 0.2 * (2.0 * PI * t).sin() + 0.01 * ((i * 37 % 11) as f64 - 5.0)
                    - if t >= 2006.0 { 0.3 } else { 0.0 }
            })
            .collect();
        let r = bfast_monitor(&y, 2000.0, 2005.0, freq, 3, 0.25, 10, 0.05).unwrap();
        assert_eq!(r.valid, 1.0);
        assert_eq!(r.has_break, 1.0);
        assert!(r.breakpoint >= 2006.0 && r.breakpoint < 2006.5, "{r:?}");
        assert!(critval(0.3, 10, 0.05).is_err() && critval(0.25, 3, 0.05).is_err());
    }
}
