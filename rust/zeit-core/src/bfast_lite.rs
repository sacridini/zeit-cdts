//! bfastlite (Masiliūnas et al. 2021; strucchangeRcpp's breakpoints with the
//! LWZ criterion): a port of src/bfast_lite.cpp, same operations in the same
//! order (Eigen's products, LDLT and QR included), so the same bits.

// The dynamic program indexes its tables as the C++ does.
#![allow(clippy::needless_range_loop)]

use crate::bfast_monitor::design_matrix;
use crate::eigen::ldlt::{xtx_rows, Ldlt};
use crate::eigen::qr::{mat_t_vec, sum_seq, ColPivQr, Mat, Order};
use crate::eigen::sum_by;
use crate::parallel;
use std::f64::consts::PI;

/// The result of one series (the C++'s BFLResult).
#[derive(Clone, Debug, PartialEq)]
pub struct BflResult {
    pub n_breaks: f64,
    pub rss: f64,
    pub lwz: f64,
    pub n_valid: f64,
    pub valid: f64,
    pub breakpoint_idx: Vec<f64>,
    pub magnitude: Vec<f64>,
    pub first_after_idx: Vec<f64>,
}

impl Default for BflResult {
    fn default() -> Self {
        BflResult {
            n_breaks: f64::NAN,
            rss: f64::NAN,
            lwz: f64::NAN,
            n_valid: f64::NAN,
            valid: 0.0,
            breakpoint_idx: Vec::new(),
            magnitude: Vec::new(),
            first_after_idx: Vec::new(),
        }
    }
}

/// Recursive residuals (Brown-Durbin-Evans) of the segment [seg_start, n) of
/// x / y: cum_rss[t] is the RSS of the OLS fit on [seg_start, seg_start + k + t].
/// The Gram matrix is refactorized (LDLT) at every step, as the C++ does.
pub(crate) fn recursive_cum_rss(x: &Mat, y: &[f64], seg_start: usize, n: usize, k: usize) -> Vec<f64> {
    let len = n - seg_start;
    if len <= k {
        return Vec::new();
    }
    let mut cum = vec![0.0; len - k];
    let mut gram = xtx_rows(x, seg_start, k);
    let block = Mat::from_fn(k, k, Order::Col, |r, j| x.get(seg_start + r, j));
    let mut xty = mat_t_vec(&block, &y[seg_start..seg_start + k]);
    for t in 0..len - k {
        let ldlt = Ldlt::new(&gram);
        let row = seg_start + k + t;
        let xr: Vec<f64> = (0..k).map(|j| x.get(row, j)).collect();
        let yr = y[row];
        let x1xr = ldlt.solve(&xr);
        let fr = 1.0 + sum_by(k, |j| xr[j] * x1xr[j]);
        let beta = ldlt.solve(&xty);
        let betar_dot_xr = sum_by(k, |j| xr[j] * beta[j]);
        let w = (yr - betar_dot_xr) / fr.sqrt();
        cum[t] = if t == 0 { 0.0 } else { cum[t - 1] } + w * w;
        // XtX += xr xr^T (an outer product), Xty += xr yr
        for j in 0..k {
            for i in 0..k {
                let v = gram.get(i, j) + xr[j] * xr[i];
                gram.set(i, j, v);
            }
        }
        for j in 0..k {
            xty[j] += xr[j] * yr;
        }
    }
    cum
}

/// LWZ (Liu, Wu & Zidek 1997) of m breaks.
fn lwz_score(rss: f64, n: usize, n_regressors: usize, m_breaks: usize) -> f64 {
    let n_d = n as f64;
    let logl = n_d * ((rss / n_d).ln() + 1.0 + (2.0 * PI).ln());
    let df = (n_regressors + 1) as f64 * (m_breaks + 1) as f64;
    let lwz_k = 0.299 * n_d.ln().powf(2.1);
    logl + lwz_k * df
}

/// bfastlite of one series `y_raw` (NaN dropped) with the design matrix `x_full`.
fn lite(y_raw: &[f64], x_full: &Mat, h: f64, max_breaks_output: i32, min_valid: i32) -> BflResult {
    let mut res = BflResult::default();
    let k = x_full.cols;
    let valid_rows: Vec<usize> = (0..y_raw.len()).filter(|&i| !y_raw[i].is_nan()).collect();
    let n = valid_rows.len();
    res.n_valid = n as f64;
    let hh = (h * n as f64).floor() as i64;
    if !(hh > k as i64 && hh <= (n / 2) as i64 && n as i64 >= min_valid as i64) {
        return res;
    }
    let hh = hh as usize;
    res.valid = 1.0;

    let x = Mat::from_fn(n, k, Order::Col, |r, j| x_full.get(valid_rows[r], j));
    let y: Vec<f64> = valid_rows.iter().map(|&i| y_raw[i]).collect();

    // rss_triang[s][t]: the RSS of the OLS fit on [s, s + k + t]
    let rss_triang: Vec<Vec<f64>> = (0..=n - hh).map(|s| recursive_cum_rss(&x, &y, s, n, k)).collect();
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

    let rss0 = rss(0, n - 1);
    let (mut best_lwz, mut best_m, mut best_rss) = (lwz_score(rss0, n, k, 0), 0, rss0);
    let mut best_breaks: Vec<usize> = Vec::new();
    for m in 1..=max_m_reached {
        if f[m][n - 1].is_nan() {
            continue;
        }
        let total = f[m][n - 1];
        let lwz_m = lwz_score(total, n, k, m);
        if lwz_m < best_lwz {
            best_lwz = lwz_m;
            best_m = m;
            best_rss = total;
            let mut breaks = vec![0; m];
            let mut cur = n - 1;
            for level in (1..=m).rev() {
                cur = ptr[level][cur];
                breaks[level - 1] = cur;
            }
            best_breaks = breaks;
        }
    }
    res.n_breaks = best_m as f64;
    res.rss = best_rss;
    res.lwz = best_lwz;
    res.breakpoint_idx = best_breaks.iter().map(|&b| valid_rows[b] as f64).collect();

    // the magnitude of each break: the right segment's model minus the left
    // one's at the first observation after the break
    if best_m > 0 {
        let mut coefs = Vec::with_capacity(best_m + 1);
        let mut start = 0;
        for s in 0..=best_m {
            let end = if s < best_m { best_breaks[s] } else { n - 1 };
            let len = end - start + 1;
            let block = Mat::from_fn(len, k, Order::Col, |r, j| x.get(start + r, j));
            coefs.push(ColPivQr::new(block).solve(&y[start..start + len]));
            start = end + 1;
        }
        for i in 0..best_m {
            let row = best_breaks[i] + 1;
            res.magnitude
                .push(sum_seq(k, |j| x.get(row, j) * (coefs[i + 1][j] - coefs[i][j])));
            res.first_after_idx.push(valid_rows[row] as f64);
        }
    }
    res
}

/// bfastlite of one series (unit-testing helper; min_valid 20).
pub fn bfast_lite(y: &[f64], start_time: f64, frequency: i32, order: i32, h: f64, max_breaks_output: i32) -> BflResult {
    let x = design_matrix(y.len(), start_time, frequency, order);
    lite(y, &x, h, max_breaks_output, 20)
}

/// Rows of the batch's output for `max_breaks_output`.
pub fn n_metrics(max_breaks_output: usize) -> usize {
    5 + 3 * max_breaks_output
}

/// bfastlite on every row of `values` (`n_pixels` x T): n_metrics x n_pixels
/// (n_breaks, rss, lwz, n_valid, valid, then max_breaks_output breakpoint
/// indices, magnitudes and first-after indices), NaN where nothing.
#[allow(clippy::too_many_arguments)]
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    start_time: f64,
    frequency: i32,
    order: i32,
    h: f64,
    max_breaks_output: i32,
    min_valid: i32,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time);
    let x = design_matrix(n_time, start_time, frequency, order);
    let mb = max_breaks_output.max(0) as usize;
    let width = n_metrics(mb);
    let mut per_pixel = vec![f64::NAN; n_pixels * width];
    parallel::for_each_row(
        &mut per_pixel,
        width,
        n_jobs,
        || (),
        |_, p, out| {
            let r = lite(
                &values[p * n_time..(p + 1) * n_time],
                &x,
                h,
                max_breaks_output,
                min_valid,
            );
            out[..5].copy_from_slice(&[r.n_breaks, r.rss, r.lwz, r.n_valid, r.valid]);
            for b in 0..r.breakpoint_idx.len().min(mb) {
                out[5 + b] = r.breakpoint_idx[b];
                out[5 + mb + b] = r.magnitude[b];
                out[5 + 2 * mb + b] = r.first_after_idx[b];
            }
        },
    );
    parallel::transpose(&per_pixel, n_pixels, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_one_break() {
        let y: Vec<f64> = (0..23 * 6)
            .map(|i| {
                let t = 2000.0 + i as f64 / 23.0;
                0.6 + 0.2 * (2.0 * PI * t).sin() + 0.005 * ((i * 37 % 11) as f64 - 5.0)
                    - if i >= 70 { 0.3 } else { 0.0 }
            })
            .collect();
        let r = bfast_lite(&y, 2000.0, 23, 3, 0.15, 5);
        assert_eq!(r.valid, 1.0);
        assert_eq!(r.n_breaks, 1.0, "{r:?}");
        assert!((r.breakpoint_idx[0] - 69.0).abs() <= 1.0, "{r:?}");
        assert!((r.magnitude[0] + 0.3).abs() < 0.05, "{r:?}");
    }
}
