//! MATLAB's robustfit (bisquare, tune 4.685, at most 5 iterations) as CCDC's
//! autoTmask runs it: a port of robust_fit in src/ccdc.cpp, same operations in
//! the same order (Eigen's column-major QR and products included), so the same
//! bits. Used by Tmask, and by CCDC when it moves over.

use crate::eigen::qr::{mat_vec, sum_seq, ColPivQr, Mat, Order};
use crate::eigen::sum_by;

/// The median of `v` (MATLAB's): the middle value, or the mean of the two.
pub fn median_of(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}

/// `Q.rowwise().squaredNorm()` assigned to a vector: rows in pairs through
/// Eigen's packet path (the first column, then blocks of four added in pairs,
/// then one at a time), an odd last row through the scalar path (one at a time).
fn rowwise_squared_norm(q: &Mat) -> Vec<f64> {
    let (n, cols) = (q.rows, q.cols);
    let sq = |i: usize, j: usize| {
        let v = q.get(i, j);
        v * v
    };
    let packetwise = |i: usize| {
        if cols == 0 {
            return 0.0;
        }
        let size4 = (cols - 1) & !3;
        let mut p = sq(i, 0);
        let mut j = 1;
        while j < size4 {
            p += (sq(i, j) + sq(i, j + 1)) + (sq(i, j + 2) + sq(i, j + 3));
            j += 4;
        }
        while j < cols {
            p += sq(i, j);
            j += 1;
        }
        p
    };
    let paired = n - n % 2;
    (0..n)
        .map(|i| {
            if i < paired {
                packetwise(i)
            } else {
                sum_seq(cols, |j| sq(i, j))
            }
        })
        .collect()
}

/// robust_fit: the coefficients [intercept, columns of `xin`...] of y on `xin`
/// (n x p, column-major), an intercept added.
pub fn robust_fit(xin: &Mat, y: &[f64]) -> Vec<f64> {
    let n = xin.rows;
    let p = xin.cols + 1;
    let x = Mat::from_fn(n, p, Order::Col, |i, j| if j == 0 { 1.0 } else { xin.get(i, j - 1) });
    let eps = f64::EPSILON;

    let mut qr = ColPivQr::new(x.clone());
    qr.set_threshold(n.max(p) as f64 * eps);
    let xrank = qr.rank();
    let mut b = qr.solve(y);

    // leverage from the full-rank part of the QR: E = X(:, perm) / R
    let q = qr.q_thin(xrank);
    let h = rowwise_squared_norm(&q);
    let adjfactor: Vec<f64> = h
        .iter()
        .map(|&hi| 1.0 / (1.0 - if hi < 0.9999 { hi } else { 0.9999 }).sqrt())
        .collect();

    let mean_y = sum_by(n, |i| y[i]) / n as f64;
    let sd_y = if n > 1 {
        (sum_by(n, |i| {
            let d = y[i] - mean_y;
            d * d
        }) / (n - 1) as f64)
            .sqrt()
    } else {
        0.0
    };
    let mut tiny_s = 1e-6 * sd_y;
    if tiny_s == 0.0 {
        tiny_s = 1.0;
    }
    let d = eps.sqrt();
    const TUNE: f64 = 4.685;

    let mut b0 = vec![0.0; p];
    let mut wxrank = xrank;
    let mut iter = 1;
    let not_converged = |b: &[f64], b0: &[f64]| {
        (0..p).any(|k| {
            let (u, v) = (b[k].abs(), b0[k].abs());
            (b[k] - b0[k]).abs() > d * if u < v { v } else { u }
        })
    };
    while not_converged(&b, &b0) {
        iter += 1;
        if iter > 5 {
            break;
        }
        let xb = mat_vec(&x, &b);
        let radj: Vec<f64> = (0..n).map(|i| (y[i] - xb[i]) * adjfactor[i]).collect();
        let mut rs: Vec<f64> = radj.iter().map(|r| r.abs()).collect();
        rs.sort_by(f64::total_cmp);
        let mad_s = median_of(&rs[wxrank.max(1) - 1..]) / 0.6745;
        let scale = if mad_s < tiny_s { tiny_s } else { mad_s } * TUNE;
        let sw: Vec<f64> = radj
            .iter()
            .map(|&ra| {
                let r = ra / scale;
                let wi = if r.abs() < 1.0 {
                    (1.0 - r * r) * (1.0 - r * r)
                } else {
                    0.0
                };
                wi.sqrt()
            })
            .collect();
        b0.copy_from_slice(&b);
        let xw = Mat::from_fn(n, p, Order::Col, |i, j| x.get(i, j) * sw[i]);
        let yw: Vec<f64> = (0..n).map(|i| y[i] * sw[i]).collect();
        let wqr = ColPivQr::new(xw);
        wxrank = wqr.rank();
        b = wqr.solve(&yw);
    }
    b
}

/// robustfit_bisquare: `x` n x `n_columns` row-major, `y` n values.
pub fn robustfit_bisquare(x: &[f64], n_columns: usize, y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let xin = Mat::from_fn(n, n_columns, Order::Col, |i, j| x[i * n_columns + j]);
    robust_fit(&xin, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ignores_an_outlier() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|v| 2.0 + 0.5 * v).collect();
        y[10] += 100.0;
        let b = robustfit_bisquare(&x, 1, &y);
        assert!((b[0] - 2.0).abs() < 1e-6 && (b[1] - 0.5).abs() < 1e-6, "{b:?}");
    }
}
