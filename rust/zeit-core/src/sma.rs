//! Linear spectral mixture analysis: a port of src/sma.cpp, same operations in
//! the same order (Eigen's row-major matrices, products and QR included), so
//! the same bits. The fractions f of the endmembers E (k x bands) that best
//! rebuild a spectrum y (bands), y ~ E^T f:
//!
//! - nonneg and sum_to_one: fully constrained least squares (Heinz & Chang
//!   2001), the NNLS (Lawson & Hanson 1974) of the system with one more row,
//!   delta * 1^T f = delta;
//! - nonneg only: NNLS; sum_to_one only: least squares with the extra row;
//! - neither: ordinary least squares.

use crate::eigen::qr::{mat_t_vec, mat_vec, mat_vec_rows, ColPivQr, Mat, Order};
use crate::parallel;

/// Least squares on the columns in `passive` (the others stay 0).
fn solve_passive(a: &Mat, b: &[f64], passive: &[bool]) -> Vec<f64> {
    let n = a.cols;
    let cols: Vec<usize> = (0..n).filter(|&j| passive[j]).collect();
    let mut z = vec![0.0; n];
    if cols.is_empty() {
        return z;
    }
    let sub = Mat::from_fn(a.rows, cols.len(), Order::Row, |i, c| a.get(i, cols[c]));
    let s = ColPivQr::new(sub).solve(b);
    for (c, &j) in cols.iter().enumerate() {
        z[j] = s[c];
    }
    z
}

/// Lawson & Hanson (1974), chapter 23, as in scipy.optimize.nnls: min ||A x - b||
/// subject to x >= 0, with at most `max_iter` iterations (3 n if negative).
fn nnls_impl(a: &Mat, b: &[f64], max_iter: i64) -> Vec<f64> {
    let n = a.cols;
    let max_iter = if max_iter < 0 { 3 * n as i64 } else { max_iter };
    let mut x = vec![0.0; n];
    let mut passive = vec![false; n];
    let tol = 10.0 * f64::EPSILON * a.max_abs() * a.rows.max(a.cols) as f64;
    let gradient = |x: &[f64]| {
        let ax = mat_vec(a, x);
        let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
        mat_t_vec(a, &r)
    };
    let mut w = gradient(&x);
    let mut iter = 0;
    loop {
        let mut best = None;
        let mut best_w = tol;
        for j in 0..n {
            if !passive[j] && w[j] > best_w {
                best_w = w[j];
                best = Some(j);
            }
        }
        let Some(best) = best else { break };
        passive[best] = true;
        loop {
            iter += 1;
            if iter > max_iter {
                return x;
            }
            let z = solve_passive(a, b, &passive);
            if !(0..n).any(|j| passive[j] && z[j] <= 0.0) {
                x = z;
                break;
            }
            let mut alpha = f64::INFINITY;
            for j in 0..n {
                if passive[j] && z[j] <= 0.0 {
                    let v = x[j] / (x[j] - z[j]);
                    if v < alpha {
                        alpha = v;
                    }
                }
            }
            for j in 0..n {
                x[j] += alpha * (z[j] - x[j]);
            }
            for j in 0..n {
                if passive[j] && x[j].abs() <= tol {
                    passive[j] = false;
                    x[j] = 0.0;
                }
            }
        }
        w = gradient(&x);
    }
    x
}

/// NNLS of A (m x n, row-major) and b.
pub fn nnls(a: &[f64], b: &[f64], m: usize, n: usize, max_iter: i64) -> Result<Vec<f64>, &'static str> {
    if a.len() != m * n || b.len() != m {
        return Err("nnls: A must be m x n and b of length m");
    }
    let a = Mat::from_data(m, n, Order::Row, a.to_vec());
    Ok(nnls_impl(&a, b, max_iter))
}

/// The system of the spectra: E^T (bands x k), plus the sum-to-one row.
fn make_system(endmembers: &[f64], k: usize, bands: usize, sum_to_one: bool, delta: f64) -> Mat {
    let rows = bands + sum_to_one as usize;
    Mat::from_fn(rows, k, Order::Row, |i, j| {
        if i < bands {
            endmembers[j * bands + i]
        } else {
            delta
        }
    })
}

/// The fractions of one spectrum then the RMSE of the rebuilt one, into `out`
/// (k + 1 values); NaN for a spectrum with a non-finite band.
fn solve_one(y: &[f64], system: &Mat, k: usize, sum_to_one: bool, nonneg: bool, delta: f64, out: &mut [f64]) {
    let bands = y.len();
    if y.iter().any(|v| !v.is_finite()) {
        out.fill(f64::NAN);
        return;
    }
    let mut b = y.to_vec();
    if sum_to_one {
        b.push(delta);
    }
    let f = if nonneg {
        nnls_impl(system, &b, -1)
    } else {
        ColPivQr::new(system.clone()).solve(&b)
    };
    let rebuilt = mat_vec_rows(system, bands, &f);
    let mut sse = 0.0;
    for i in 0..bands {
        sse += (rebuilt[i] - y[i]) * (rebuilt[i] - y[i]);
    }
    out[..k].copy_from_slice(&f[..k]);
    out[k] = (sse / bands as f64).sqrt();
}

/// The fractions of one spectrum `y`, then the RMSE.
pub fn unmix_pixel(
    y: &[f64],
    endmembers: &[f64],
    k: usize,
    sum_to_one: bool,
    nonneg: bool,
    delta: f64,
) -> Result<Vec<f64>, &'static str> {
    let bands = y.len();
    if k == 0 || endmembers.len() != k * bands {
        return Err("unmix: endmembers must be k x bands");
    }
    let system = make_system(endmembers, k, bands, sum_to_one, delta);
    let mut out = vec![0.0; k + 1];
    solve_one(y, &system, k, sum_to_one, nonneg, delta, &mut out);
    Ok(out)
}

/// `unmix_pixel` on every spectrum of `values` (`n` x `bands`, row-major):
/// n x (k + 1), the fractions then the RMSE.
#[allow(clippy::too_many_arguments)]
pub fn unmix_batch(
    values: &[f64],
    n: usize,
    bands: usize,
    endmembers: &[f64],
    k: usize,
    sum_to_one: bool,
    nonneg: bool,
    delta: f64,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n * bands);
    assert_eq!(endmembers.len(), k * bands);
    let system = make_system(endmembers, k, bands, sum_to_one, delta);
    let mut out = vec![0.0; n * (k + 1)];
    parallel::for_each_row(
        &mut out,
        k + 1,
        n_jobs,
        || (),
        |_, p, row| {
            solve_one(
                &values[p * bands..(p + 1) * bands],
                &system,
                k,
                sum_to_one,
                nonneg,
                delta,
                row,
            )
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_fractions() {
        let em = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.2, 0.2, 0.6, 0.6, 0.2, 0.2,
        ];
        let f = [0.2, 0.5, 0.3];
        let y: Vec<f64> = (0..6).map(|i| (0..3).map(|j| f[j] * em[j * 6 + i]).sum()).collect();
        for (sum_to_one, nonneg) in [(true, true), (false, true), (true, false), (false, false)] {
            let out = unmix_pixel(&y, &em, 3, sum_to_one, nonneg, 1000.0).unwrap();
            for j in 0..3 {
                assert!((out[j] - f[j]).abs() < 1e-6, "{sum_to_one} {nonneg} {out:?}");
            }
            assert!(out[3] < 1e-6);
        }
    }

    #[test]
    fn nnls_clips_negative() {
        // min ||x - (1, -1)||, x >= 0 -> (1, 0)
        assert_eq!(
            nnls(&[1.0, 0.0, 0.0, 1.0], &[1.0, -1.0], 2, 2, -1).unwrap(),
            vec![1.0, 0.0]
        );
    }
}
