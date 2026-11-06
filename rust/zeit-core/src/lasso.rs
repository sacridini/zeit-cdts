//! The Gaussian lasso by cyclic coordinate descent: an implementation of
//! Friedman, Hastie & Tibshirani (2010, J. Stat. Softw. 33(1)) written from
//! the paper and from glmnet's user documentation (no third-party code).
//!
//! It minimises `1/(2n) sum (y - b0 - Z t)^2 + lambda sum |t|` where `Z` are
//! the columns standardised (mean 0, population standard deviation 1) and the
//! intercept is not penalised; the coefficients are returned on the original
//! scale of X and y. Covariance updates (the Gram matrix is built once, each
//! coordinate update costs O(p)), columns visited in increasing order, one full
//! pass then passes over the active set until the largest squared step is below
//! `thresh` times the population variance of y, then a full pass again.
//! The preprocessing (centring, scales, Gram matrix) is always done in f64; the
//! descent runs in f32 (`lasso_gaussian`) or in f64 (`lasso_gaussian_f64`).

use std::ops::{Add, Mul, Sub};

/// The minimal arithmetic the descent needs, for f32 and f64.
pub trait LassoFloat: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl LassoFloat for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl LassoFloat for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// A fitted lasso, on the original scale of X and y.
#[derive(Debug, Clone, PartialEq)]
pub struct LassoFit {
    pub intercept: f64,
    /// One coefficient per column; 0.0 for the excluded (constant) ones.
    pub coefs: Vec<f64>,
    /// Sweeps (full or active) used.
    pub iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LassoError {
    /// No observations.
    Empty,
    /// A column as long as y is required.
    LengthMismatch,
    /// NaN or infinity in X or y.
    NonFinite,
    /// lambda < 0 or NaN, thresh <= 0 or NaN, max_iter == 0.
    BadParameter,
}

impl std::fmt::Display for LassoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            LassoError::Empty => "lasso: no observations",
            LassoError::LengthMismatch => "lasso: a column and y differ in length",
            LassoError::NonFinite => "lasso: NaN or infinite value",
            LassoError::BadParameter => "lasso: bad lambda, thresh or max_iter",
        })
    }
}

impl std::error::Error for LassoError {}

/// The lasso with the descent in f32 (preprocessing in f64).
pub fn lasso_gaussian(
    x_cols: &[&[f64]],
    y: &[f64],
    lambda: f64,
    thresh: f64,
    max_iter: usize,
) -> Result<LassoFit, LassoError> {
    lasso_generic::<f32>(x_cols, y, lambda, thresh, max_iter)
}

/// The lasso with f64 everywhere.
pub fn lasso_gaussian_f64(
    x_cols: &[&[f64]],
    y: &[f64],
    lambda: f64,
    thresh: f64,
    max_iter: usize,
) -> Result<LassoFit, LassoError> {
    lasso_generic::<f64>(x_cols, y, lambda, thresh, max_iter)
}

fn soft<T: LassoFloat>(u: T, a: T) -> T {
    let zero = T::from_f64(0.0);
    if u > a {
        u - a
    } else if zero - u > a {
        u + a
    } else {
        zero
    }
}

fn lasso_generic<T: LassoFloat>(
    x_cols: &[&[f64]],
    y: &[f64],
    lambda: f64,
    thresh: f64,
    max_iter: usize,
) -> Result<LassoFit, LassoError> {
    let n = y.len();
    if n == 0 {
        return Err(LassoError::Empty);
    }
    if x_cols.iter().any(|c| c.len() != n) {
        return Err(LassoError::LengthMismatch);
    }
    if lambda.is_nan() || lambda < 0.0 || thresh.is_nan() || thresh <= 0.0 || max_iter == 0 {
        return Err(LassoError::BadParameter);
    }
    if y.iter().any(|v| !v.is_finite()) || x_cols.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
        return Err(LassoError::NonFinite);
    }
    let p = x_cols.len();
    let nf = n as f64;
    let ybar = y.iter().sum::<f64>() / nf;
    let yc: Vec<f64> = y.iter().map(|v| v - ybar).collect();
    let nulldev = yc.iter().map(|v| v * v).sum::<f64>() / nf;
    let null_fit = |iterations, coefs: Vec<f64>| LassoFit { intercept: ybar, coefs, iterations, converged: true };

    // Standardised columns (centred in f64 before any product); constant ones are left out.
    let mut kept: Vec<usize> = Vec::new();
    let mut means = vec![0.0; p];
    let mut scales = vec![1.0; p];
    let mut z: Vec<Vec<f64>> = Vec::new();
    for (j, col) in x_cols.iter().enumerate() {
        let m = col.iter().sum::<f64>() / nf;
        let s = (col.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / nf).sqrt();
        if s <= 1e-12 * m.abs().max(1.0) {
            continue;
        }
        means[j] = m;
        scales[j] = s;
        kept.push(j);
        z.push(col.iter().map(|v| (v - m) / s).collect());
    }
    let q = kept.len();
    if q == 0 || nulldev == 0.0 {
        return Ok(null_fit(0, vec![0.0; p]));
    }

    // c_j = z_j' yc / n and the Gram matrix, in f64 then converted.
    let c: Vec<T> = z.iter().map(|zj| T::from_f64(zj.iter().zip(&yc).map(|(a, b)| a * b).sum::<f64>() / nf)).collect();
    let mut gram = vec![T::from_f64(0.0); q * q];
    for a in 0..q {
        gram[a * q + a] = T::from_f64(1.0);
        for b in a + 1..q {
            let g = T::from_f64(z[a].iter().zip(&z[b]).map(|(u, v)| u * v).sum::<f64>() / nf);
            gram[a * q + b] = g;
            gram[b * q + a] = g;
        }
    }

    let zero = T::from_f64(0.0);
    let lam = T::from_f64(lambda);
    let stop = T::from_f64(thresh * nulldev);
    let mut t = vec![zero; q];
    let mut g = c;
    let mut iter = 0usize;
    let mut converged = false;

    // One sweep over `idx`; the largest squared step.
    let sweep = |idx: &[usize], t: &mut [T], g: &mut [T]| -> T {
        let mut dmax = zero;
        for &j in idx {
            let new = soft(g[j] + t[j], lam);
            let d = new - t[j];
            if d != zero {
                for k in 0..q {
                    g[k] = g[k] - gram[k * q + j] * d;
                }
                t[j] = new;
            }
            let d2 = d * d;
            if d2 > dmax {
                dmax = d2;
            }
        }
        dmax
    };

    let all: Vec<usize> = (0..q).collect();
    'outer: while iter < max_iter {
        let dmax = sweep(&all, &mut t, &mut g);
        iter += 1;
        if dmax < stop {
            converged = true;
            break;
        }
        let active: Vec<usize> = (0..q).filter(|&j| t[j] != zero).collect();
        while iter < max_iter {
            let dmax = sweep(&active, &mut t, &mut g);
            iter += 1;
            if dmax < stop {
                continue 'outer;
            }
        }
    }

    let mut coefs = vec![0.0; p];
    let mut intercept = ybar;
    for (i, &j) in kept.iter().enumerate() {
        let beta = t[i].to_f64() / scales[j];
        coefs[j] = beta;
        intercept -= beta * means[j];
    }
    Ok(LassoFit { intercept, coefs, iterations: iter, converged })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic uniform generator in [-1, 1).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
        fn vec(&mut self, n: usize) -> Vec<f64> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    fn refs(cols: &[Vec<f64>]) -> Vec<&[f64]> {
        cols.iter().map(|c| c.as_slice()).collect()
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    /// Least squares by the normal equations (Gaussian elimination), [b0, b...].
    fn ols(cols: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let k = cols.len() + 1;
        let col = |j: usize, i: usize| if j == 0 { 1.0 } else { cols[j - 1][i] };
        let mut a = vec![vec![0.0; k + 1]; k];
        for r in 0..k {
            for c in 0..k {
                a[r][c] = (0..n).map(|i| col(r, i) * col(c, i)).sum();
            }
            a[r][k] = (0..n).map(|i| col(r, i) * y[i]).sum();
        }
        for i in 0..k {
            let piv = (i..k).max_by(|&u, &v| a[u][i].abs().total_cmp(&a[v][i].abs())).unwrap();
            a.swap(i, piv);
            for r in 0..k {
                if r != i {
                    let f = a[r][i] / a[i][i];
                    for c in i..=k {
                        a[r][c] -= f * a[i][c];
                    }
                }
            }
        }
        (0..k).map(|i| a[i][k] / a[i][i]).collect()
    }

    fn two_cols(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut r = Lcg(7);
        let x1 = r.vec(n);
        let x2 = r.vec(n);
        let y = (0..n).map(|i| 2.0 + 3.0 * x1[i] - 1.5 * x2[i]).collect();
        (vec![x1, x2], y)
    }

    /// Series like the CCDC's: time in days, annual harmonics, reflectance-like y.
    fn ccdc_like(n: usize, harmonics: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut r = Lcg(seed);
        let days: Vec<f64> = (0..n).map(|i| 730000.0 + 16.0 * i as f64 + 3.0 * r.next()).collect();
        let w = 2.0 * std::f64::consts::PI / 365.25;
        let mut cols = vec![days.clone()];
        for h in 1..=harmonics {
            cols.push(days.iter().map(|d| (w * h as f64 * d).cos()).collect());
            cols.push(days.iter().map(|d| (w * h as f64 * d).sin()).collect());
        }
        let y = (0..n)
            .map(|i| 3000.0 + 0.2 * (days[i] - 730000.0) + 1500.0 * cols[1][i] + 800.0 * cols[2][i] + 200.0 * r.next())
            .collect();
        (cols, y)
    }

    #[test]
    fn t1_least_squares_when_lambda_is_zero() {
        let (cols, y) = two_cols(60);
        let want = ols(&cols, &y);
        let fit = lasso_gaussian_f64(&refs(&cols), &y, 0.0, 1e-20, 10_000).unwrap();
        assert!(close(fit.intercept, 2.0, 1e-8) && close(fit.coefs[0], 3.0, 1e-8) && close(fit.coefs[1], -1.5, 1e-8));
        assert!(close(fit.intercept, want[0], 1e-8) && close(fit.coefs[0], want[1], 1e-8));
        let f32fit = lasso_gaussian(&refs(&cols), &y, 0.0, 1e-10, 10_000).unwrap();
        assert!(close(f32fit.intercept, 2.0, 1e-3) && close(f32fit.coefs[0], 3.0, 1e-3) && close(f32fit.coefs[1], -1.5, 1e-3));
    }

    #[test]
    fn t2_huge_lambda_gives_the_mean() {
        let (cols, y) = two_cols(40);
        let ybar = y.iter().sum::<f64>() / 40.0;
        for fit in [
            lasso_gaussian(&refs(&cols), &y, 1e9, 1e-4, 100).unwrap(),
            lasso_gaussian_f64(&refs(&cols), &y, 1e9, 1e-4, 100).unwrap(),
        ] {
            assert_eq!(fit.coefs, vec![0.0, 0.0]);
            assert!(close(fit.intercept, ybar, 1e-12));
            assert!(fit.converged && fit.iterations <= 2);
        }
    }

    #[test]
    fn t3_one_predictor_closed_form() {
        let n = 50;
        let mut r = Lcg(3);
        let x: Vec<f64> = r.vec(n).iter().map(|v| 10.0 + 4.0 * v).collect();
        let y: Vec<f64> = x.iter().map(|v| 5.0 + 2.0 * v + 3.0 * r.next()).collect();
        let nf = n as f64;
        let m = x.iter().sum::<f64>() / nf;
        let ybar = y.iter().sum::<f64>() / nf;
        let s = (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / nf).sqrt();
        let cov = x.iter().zip(&y).map(|(a, b)| (a - m) * (b - ybar)).sum::<f64>() / nf;
        let edge = cov.abs() / s;
        for lambda in [edge * 0.5, edge * 0.99, edge * 1.01, edge * 2.0] {
            let beta = cov.signum() * (cov.abs() / s - lambda).max(0.0) / s;
            let fit = lasso_gaussian_f64(&[&x], &y, lambda, 1e-14, 10_000).unwrap();
            assert!(close(fit.coefs[0], beta, 1e-9), "{} vs {}", fit.coefs[0], beta);
            assert!(close(fit.intercept, ybar - beta * m, 1e-9));
        }
        // Negative covariance.
        let yn: Vec<f64> = y.iter().map(|v| -v).collect();
        let fit = lasso_gaussian_f64(&[&x], &yn, edge * 0.5, 1e-14, 10_000).unwrap();
        assert!(fit.coefs[0] < 0.0);
    }

    #[test]
    fn t4_orthonormal_columns() {
        // Exactly orthogonal columns with variance 1 once standardised.
        let x1 = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let x2 = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let x3 = vec![1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let y: Vec<f64> = (0..8).map(|i| 4.0 * x1[i] - 2.0 * x2[i] + 0.5 * x3[i] + 1.0).collect();
        let lambda = 1.0;
        let fit = lasso_gaussian_f64(&[&x1, &x2, &x3], &y, lambda, 1e-12, 100).unwrap();
        let want = [3.0, -1.0, 0.0];
        for j in 0..3 {
            assert!(close(fit.coefs[j], want[j], 1e-12), "{:?}", fit.coefs);
        }
        // The full pass, one active pass that changes nothing, the full check.
        assert!(fit.converged && fit.iterations <= 3, "{}", fit.iterations);
    }

    #[test]
    fn t5_constant_columns_are_left_out() {
        let (cols, y) = two_cols(40);
        let base = lasso_gaussian_f64(&refs(&cols), &y, 0.3, 1e-12, 10_000).unwrap();
        let zeros = vec![0.0; 40];
        let fives = vec![5.0; 40];
        let with = vec![cols[0].clone(), zeros.clone(), cols[1].clone(), fives.clone()];
        let fit = lasso_gaussian_f64(&refs(&with), &y, 0.3, 1e-12, 10_000).unwrap();
        assert_eq!(fit.coefs[1], 0.0);
        assert_eq!(fit.coefs[3], 0.0);
        assert!(close(fit.coefs[0], base.coefs[0], 1e-12) && close(fit.coefs[2], base.coefs[1], 1e-12));
        assert!(close(fit.intercept, base.intercept, 1e-12));
        let ybar = y.iter().sum::<f64>() / 40.0;
        let only = lasso_gaussian(&[&zeros, &fives], &y, 0.3, 1e-4, 100).unwrap();
        assert_eq!(only.coefs, vec![0.0, 0.0]);
        assert!(close(only.intercept, ybar, 1e-12) && only.converged && only.iterations == 0);
    }

    #[test]
    fn t6_large_time_column_f32_matches_f64() {
        for (n, h) in [(40, 1), (120, 2), (200, 3)] {
            let (cols, y) = ccdc_like(n, h, 11 + n as u64);
            let a = lasso_gaussian(&refs(&cols), &y, 20.0, 1e-10, 10_000).unwrap();
            let b = lasso_gaussian_f64(&refs(&cols), &y, 20.0, 1e-10, 10_000).unwrap();
            // Coefficients times the column's spread are on the scale of y.
            for (j, col) in cols.iter().enumerate() {
                let m = col.iter().sum::<f64>() / n as f64;
                let s = (col.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64).sqrt();
                assert!(((a.coefs[j] - b.coefs[j]) * s).abs() <= 1e-3 * 1000.0, "col {j}: {} vs {}", a.coefs[j], b.coefs[j]);
            }
            assert!((a.intercept - b.intercept).abs() <= 1e-3 * b.intercept.abs().max(1.0));
        }
    }

    #[test]
    fn t7_invariances() {
        let (cols, y) = two_cols(50);
        let base = lasso_gaussian_f64(&refs(&cols), &y, 0.2, 1e-14, 10_000).unwrap();
        // A constant added to a column and to y: only the intercept moves.
        let shifted = vec![cols[0].iter().map(|v| v + 7.0).collect::<Vec<_>>(), cols[1].clone()];
        let ys: Vec<f64> = y.iter().map(|v| v + 11.0).collect();
        let f = lasso_gaussian_f64(&refs(&shifted), &ys, 0.2, 1e-14, 10_000).unwrap();
        assert!(close(f.coefs[0], base.coefs[0], 1e-9) && close(f.coefs[1], base.coefs[1], 1e-9));
        assert!(close(f.intercept, base.intercept + 11.0 - 7.0 * base.coefs[0], 1e-9));
        // y and lambda scaled together.
        let y3: Vec<f64> = y.iter().map(|v| v * 3.0).collect();
        let f = lasso_gaussian_f64(&refs(&cols), &y3, 0.6, 1e-14, 10_000).unwrap();
        assert!(close(f.coefs[0], 3.0 * base.coefs[0], 1e-9) && close(f.coefs[1], 3.0 * base.coefs[1], 1e-9));
        assert!(close(f.intercept, 3.0 * base.intercept, 1e-9));
        // A column scaled by a.
        let scaled = vec![cols[0].iter().map(|v| v * 4.0).collect::<Vec<_>>(), cols[1].clone()];
        let f = lasso_gaussian_f64(&refs(&scaled), &y, 0.2, 1e-14, 10_000).unwrap();
        assert!(close(f.coefs[0], base.coefs[0] / 4.0, 1e-9) && close(f.coefs[1], base.coefs[1], 1e-9));
    }

    #[test]
    fn t8_errors() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 4.0];
        let short = vec![1.0, 2.0];
        assert_eq!(lasso_gaussian(&[&short], &y, 1.0, 1e-4, 10).unwrap_err(), LassoError::LengthMismatch);
        let nan = vec![1.0, f64::NAN, 3.0];
        assert_eq!(lasso_gaussian(&[&nan], &y, 1.0, 1e-4, 10).unwrap_err(), LassoError::NonFinite);
        assert_eq!(lasso_gaussian(&[&x], &nan, 1.0, 1e-4, 10).unwrap_err(), LassoError::NonFinite);
        assert_eq!(lasso_gaussian(&[&x], &y, -1.0, 1e-4, 10).unwrap_err(), LassoError::BadParameter);
        assert_eq!(lasso_gaussian(&[&x], &y, f64::NAN, 1e-4, 10).unwrap_err(), LassoError::BadParameter);
        assert_eq!(lasso_gaussian(&[&x], &y, 1.0, 0.0, 10).unwrap_err(), LassoError::BadParameter);
        assert_eq!(lasso_gaussian(&[&x], &y, 1.0, 1e-4, 0).unwrap_err(), LassoError::BadParameter);
        assert_eq!(lasso_gaussian(&[], &[], 1.0, 1e-4, 10).unwrap_err(), LassoError::Empty);
        // No columns: the mean. One observation: its value.
        let f = lasso_gaussian(&[], &y, 1.0, 1e-4, 10).unwrap();
        assert!(f.coefs.is_empty() && f.converged && close(f.intercept, 7.0 / 3.0, 1e-12));
        let f = lasso_gaussian(&[&[5.0]], &[9.0], 1.0, 1e-4, 10).unwrap();
        assert_eq!((f.intercept, f.coefs), (9.0, vec![0.0]));
        // Constant y.
        let f = lasso_gaussian(&[&x], &[2.0; 3], 0.1, 1e-4, 10).unwrap();
        assert_eq!((f.intercept, f.coefs), (2.0, vec![0.0]));
    }

    #[test]
    fn t9_kkt_conditions() {
        for (n, h) in [(30, 1), (100, 2), (200, 3)] {
            let (cols, y) = ccdc_like(n, h, 5 + n as u64);
            let lambda = 20.0;
            let fit = lasso_gaussian_f64(&refs(&cols), &y, lambda, 1e-20, 100_000).unwrap();
            assert!(fit.converged);
            let nf = n as f64;
            let resid: Vec<f64> =
                (0..n).map(|i| y[i] - fit.intercept - cols.iter().zip(&fit.coefs).map(|(c, b)| c[i] * b).sum::<f64>()).collect();
            for (j, col) in cols.iter().enumerate() {
                let m = col.iter().sum::<f64>() / nf;
                let s = (col.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / nf).sqrt();
                let gj = (0..n).map(|i| (col[i] - m) / s * resid[i]).sum::<f64>() / nf;
                if fit.coefs[j] != 0.0 {
                    assert!((gj - lambda * fit.coefs[j].signum()).abs() < 1e-8 * lambda.max(gj.abs()), "col {j}: {gj}");
                } else {
                    assert!(gj.abs() <= lambda * (1.0 + 1e-8), "col {j}: {gj}");
                }
            }
        }
    }

    #[test]
    fn max_iter_stops_without_converging() {
        let (cols, y) = ccdc_like(100, 3, 99);
        let fit = lasso_gaussian_f64(&refs(&cols), &y, 0.0, 1e-30, 2).unwrap();
        assert_eq!(fit.iterations, 2);
        assert!(!fit.converged);
    }
}
