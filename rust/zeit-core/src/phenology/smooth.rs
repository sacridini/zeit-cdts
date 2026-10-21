//! The smoothers of the phenology: a port of the Whittaker (phenofit's wWHIT,
//! a sparse Cholesky), HANTS and wTSM of src/phenology_math.cpp.

// The loops index the series as the C++'s do.
#![allow(clippy::needless_range_loop)]

use crate::eigen::ldlt::{xtx, Ldlt};
use crate::eigen::qr::{mat_t_vec, mat_vec, Mat, Order};
use crate::eigen::sparse::{Csc, SimplicialLlt};
use crate::eigen::{max_coeff, min_coeff, pmax, pmin, std_max, std_min, sum_by};

/// The pentadiagonal W + lambda D^T D in full symmetric storage, its values
/// as setFromTriplets sums the triplets the C++ lists: on the diagonal w
/// first, then lambda, 4 lambda, lambda of the three second differences
/// that cover it, in that order.
fn whittaker_matrix(w: &[f64], lambda: f64, a: &mut Csc) {
    let n = w.len();
    let blocks = n.saturating_sub(2);
    for j in 0..n {
        for k in a.outer[j]..a.outer[j + 1] {
            let i = a.inner[k];
            a.values[k] = if i == j {
                let mut d = w[j];
                if j >= 2 && j - 2 < blocks {
                    d += lambda * 1.0;
                }
                if j >= 1 && j - 1 < blocks {
                    d += lambda * 4.0;
                }
                if j < blocks {
                    d += lambda * 1.0;
                }
                d
            } else if i.abs_diff(j) == 1 {
                let (lo, hi) = (i.min(j), i.max(j));
                // (hi, lo) from the blocks lo - 1 (if any) and lo (if any)
                let mut v = 0.0;
                let mut first = true;
                for b in [lo.wrapping_sub(1), lo] {
                    if b < blocks && hi <= b + 2 && lo >= b {
                        v = if first { lambda * -2.0 } else { v + lambda * -2.0 };
                        first = false;
                    }
                }
                v
            } else {
                lambda * 1.0
            };
        }
    }
}

/// The pattern of `whittaker_matrix`: rows j - 2 ... j + 2 of every column j.
/// Without second differences (n < 3) the off-diagonal entries are not there.
fn whittaker_pattern(n: usize) -> Csc {
    let reach = if n >= 3 { 2 } else { 0 };
    let mut outer = vec![0];
    let mut inner = vec![];
    for j in 0..n {
        for i in j.saturating_sub(reach)..(j + reach + 1).min(n) {
            inner.push(i);
        }
        outer.push(inner.len());
    }
    let nnz = inner.len();
    Csc {
        n,
        outer,
        inner,
        values: vec![0.0; nnz],
    }
}

/// The C++'s thread_local SimplicialLLT and its analysis, kept per worker
/// for the length of the series.
#[derive(Default)]
pub struct WhittakerCache {
    solver: Option<(Csc, SimplicialLlt)>,
}

/// `z.cwiseMax(lo).cwiseMin(hi)` in place: packets for the even part.
fn clamp_packets(z: &mut [f64], lo: f64, hi: f64) {
    let even = z.len() / 2 * 2;
    for v in z[..even].iter_mut() {
        *v = pmin(pmax(*v, lo), hi);
    }
    for v in z[even..].iter_mut() {
        *v = std_min(std_max(*v, lo), hi);
    }
}

/// eigen_whittaker: phenofit's wWHIT (2 iterations, the weights of the points
/// below the curve lowered, the series lifted to the curve where above its
/// middle); None when the matrix is not positive definite.
pub fn whittaker(y: &[f64], weights: Option<&[f64]>, lambda: f64, cache: &mut WhittakerCache) -> Option<Vec<f64>> {
    let n = y.len();
    if n == 0 {
        return Some(vec![]);
    }
    let mut w: Vec<f64> = match weights {
        Some(w) => w.to_vec(),
        None => vec![1.0; n],
    };
    let iters = 2;
    let delta_t = 3usize;
    let wfact = 0.5;
    let mut yiter = y.to_vec();
    let mut z = vec![0.0; n];
    let ylu_min = min_coeff(y);
    let ylu_max = max_coeff(y);
    let zc = ylu_min + (ylu_max - ylu_min) * 0.5;

    if cache.solver.as_ref().is_none_or(|(a, _)| a.n != n) {
        let a = whittaker_pattern(n);
        let llt = SimplicialLlt::analyze(&a);
        cache.solver = Some((a, llt));
    }
    let (a, llt) = cache.solver.as_mut().expect("analyzed above");

    let mut b = vec![0.0; n];
    for iter in 0..iters {
        whittaker_matrix(&w, lambda, a);
        for i in 0..n {
            b[i] = w[i] * yiter[i];
        }
        if !llt.factorize(a) {
            return None;
        }
        z = llt.solve(&b);

        let mut m = w.iter().filter(|&&v| v > 0.5).count() as i64;
        if m < 2 {
            m = 2;
        }
        let w_ceil: Vec<f64> = w.iter().map(|v| v.ceil()).collect();
        let yfitmean = sum_by(n, |i| z[i] * w_ceil[i]) / m as f64;
        let variance = sum_by(n, |i| {
            let d = (z[i] - yfitmean) * w_ceil[i];
            d * d
        }) / (m - 1) as f64;
        let yfitstd = std_max(0.0, variance).sqrt();

        for i in 0..n {
            let m1 = i.saturating_sub(delta_t);
            let m2 = (n - 1).min(i + delta_t);
            let (mut yi_min, mut yi_max) = (1e9, -1e9);
            for &zj in &z[m1..=m2] {
                if zj < yi_min {
                    yi_min = zj;
                }
                if zj > yi_max {
                    yi_max = zj;
                }
            }
            if y[i] < z[i] - 1e-8 && (yi_min > yfitmean || iter < 1) {
                let mut ydiff = 0.0;
                if yi_max - yi_min < 0.8 * yfitstd && yfitstd > 1e-8 {
                    ydiff = 2.0 * (z[i] - y[i]) / yfitstd;
                }
                w[i] = wfact * w[i] * (-ydiff * ydiff).exp();
            }
        }

        clamp_packets(&mut z, ylu_min, ylu_max);
        for i in 0..n {
            if z[i] > yiter[i] && z[i] > zc {
                yiter[i] = z[i];
            }
        }
    }
    Some(z)
}

/// `std::clamp(v, lo, hi)`.
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if hi < v {
        hi
    } else {
        v
    }
}

/// `W * M` for the dense diagonal W = diag(w) of the C++ (a matrix product):
/// w_i m_i, the products by the zeros of W adding nothing but turning a
/// non-finite value of the column into NaN for every other row.
fn diag_times(w: &[f64], col: &[f64], out: &mut [f64]) {
    let n = col.len();
    let bad: Vec<usize> = (0..n).filter(|&k| !col[k].is_finite()).collect();
    for i in 0..n {
        out[i] = if bad.iter().any(|&k| k != i) {
            f64::NAN
        } else {
            let v = w[i] * col[i];
            if v == 0.0 {
                0.0
            } else {
                v
            }
        };
    }
}

/// eigen_hants: harmonic regression (mean and `num_frequencies` sines and
/// cosines of the year) reweighted up to 10 times, the points more than
/// `threshold` below the fit rejected.
pub fn hants(y: &[f64], t: &[f64], num_frequencies: i32, threshold: f64, initial_weights: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![];
    }
    let mut w: Vec<f64> = match initial_weights {
        Some(iw) => iw.iter().map(|&v| clamp(v, 0.0, 1.0)).collect(),
        None => vec![1.0; n],
    };
    let nf = num_frequencies.max(0) as usize;
    let p = 1 + 2 * nf;
    let pi = std::f64::consts::PI;
    let mut x = Mat::zeros(n, p, Order::Col);
    for i in 0..n {
        x.set(i, 0, 1.0);
        for j in 1..=nf {
            let angle = 2.0 * pi * j as f64 * t[i] / 365.25;
            x.set(i, 2 * j - 1, angle.sin());
            x.set(i, 2 * j, angle.cos());
        }
    }

    let mut y_fit = vec![];
    let mut xw = Mat::zeros(n, p, Order::Col);
    let mut yw = vec![0.0; n];
    for _ in 0..10 {
        for j in 0..p {
            diag_times(&w, &x.data[j * n..(j + 1) * n], &mut xw.data[j * n..(j + 1) * n]);
        }
        diag_times(&w, y, &mut yw);
        let xtx_m = xtx(&xw, n);
        let xty = mat_t_vec(&xw, &yw);
        let beta = Ldlt::new(&xtx_m).solve(&xty);
        y_fit = mat_vec(&x, &beta);

        let mut changed = false;
        for i in 0..n {
            let residual = y[i] - y_fit[i];
            if residual < -threshold && w[i] > 0.0 {
                w[i] = 0.0;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    y_fit
}

/// eigen_wTSM: TIMESAT's weight update after a fit `yfit` of `y`.
pub fn wtsm(y: &[f64], yfit: &[f64], w: &[f64], iter: i32, nptperyear: i32, wfact: f64) -> Vec<f64> {
    let n = y.len();
    let mut wnew = w.to_vec();
    let mut m_count = 0.0;
    for &v in w {
        if v > 0.5 {
            m_count += 1.0;
        }
    }
    if m_count < 2.0 {
        return wnew;
    }
    let mut yfitmean = 0.0;
    for i in 0..n {
        yfitmean += yfit[i] * w[i].ceil();
    }
    yfitmean /= m_count;
    let mut variance = 0.0;
    for i in 0..n {
        let d = yfit[i] - yfitmean;
        variance += d * d * w[i].ceil();
    }
    let yfitstd = (variance / (m_count - 1.0)).sqrt();
    let delta_t = (nptperyear as f64 / 7.0).floor() as usize;

    for i in 0..n {
        let m1 = i.saturating_sub(delta_t);
        let m2 = (n - 1).min(i + delta_t);
        let (mut yi_min, mut yi_max) = (yfit[m1], yfit[m1]);
        for &v in &yfit[m1..=m2] {
            if v < yi_min {
                yi_min = v;
            }
            if v > yi_max {
                yi_max = v;
            }
        }
        if y[i] < yfit[i] - 1e-8 && (yi_min > yfitmean || iter < 2) {
            let mut ydiff = 0.0;
            if yi_max - yi_min < 0.8 * yfitstd {
                ydiff = 2.0 * (yfit[i] - y[i]) / yfitstd;
            }
            wnew[i] = wfact * w[i] * (-ydiff * ydiff).exp();
        }
    }
    wnew
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whittaker_follows_the_upper_envelope() {
        let y: Vec<f64> = (0..46)
            .map(|i| {
                0.3 + 0.4 * (i as f64 * std::f64::consts::PI / 23.0).sin().abs() - if i % 7 == 3 { 0.2 } else { 0.0 }
            })
            .collect();
        let mut cache = WhittakerCache::default();
        let z = whittaker(&y, None, 10.0, &mut cache).unwrap();
        assert_eq!(z.len(), y.len());
        assert!(z.iter().all(|v| v.is_finite()));
        // the dips are cloud-like: the curve stays above them
        assert!(z[3] > y[3]);
    }

    #[test]
    fn hants_fits_a_harmonic() {
        let t: Vec<f64> = (0..46).map(|i| i as f64 * 8.0).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&t| 0.5 + 0.2 * (2.0 * std::f64::consts::PI * t / 365.25).sin())
            .collect();
        let fit = hants(&y, &t, 3, 0.1, None);
        for (a, b) in fit.iter().zip(&y) {
            assert!((a - b).abs() < 1e-9);
        }
    }
}
