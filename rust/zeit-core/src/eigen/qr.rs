//! Eigen 3.4's dense matrices, matrix-vector products and ColPivHouseholderQR,
//! with the same operations in the same order as the C++ build runs them, so
//! that the modules whose C++ used them give the same bits.
//!
//! The order of the additions follows from the storage order, as in Eigen:
//! a sum over a run of values contiguous by type (a column of a column-major
//! matrix, a row of a row-major one, a vector) runs in packets of 2 doubles;
//! a strided one runs one value at a time. In a matrix-vector product the
//! row-major kernel accumulates each row in 2 lanes (even and odd columns),
//! adds the lanes, then the last odd column; the column-major kernel
//! accumulates each row one column after the other.
//! (Eigen/src/Core/Redux.h, products/GeneralMatrixVector.h,
//! products/TriangularSolverVector.h, Householder/*, QR/ColPivHouseholderQR.h.)

// The loops index rows and columns as Eigen's do.
#![allow(clippy::needless_range_loop)]

use super::{pmadd, sum_by, PACKET};

/// The storage order of a matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Col,
    Row,
}

/// A dense matrix of doubles in either storage order.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub order: Order,
    pub data: Vec<f64>,
    /// The steps between consecutive rows and columns in `data`.
    row_step: usize,
    col_step: usize,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize, order: Order) -> Mat {
        Mat::from_data(rows, cols, order, vec![0.0; rows * cols])
    }

    /// The matrix of `data`, laid out in `order`.
    pub fn from_data(rows: usize, cols: usize, order: Order, data: Vec<f64>) -> Mat {
        assert_eq!(data.len(), rows * cols);
        let (row_step, col_step) = match order {
            Order::Col => (1, rows),
            Order::Row => (cols, 1),
        };
        Mat {
            rows,
            cols,
            order,
            data,
            row_step,
            col_step,
        }
    }

    /// The matrix whose (i, j) coefficient is `f(i, j)`.
    pub fn from_fn(rows: usize, cols: usize, order: Order, f: impl Fn(usize, usize) -> f64) -> Mat {
        let mut m = Mat::zeros(rows, cols, order);
        for i in 0..rows {
            for j in 0..cols {
                m.set(i, j, f(i, j));
            }
        }
        m
    }

    #[inline(always)]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.row_step + j * self.col_step
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[self.idx(i, j)]
    }

    #[inline(always)]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        let k = self.idx(i, j);
        self.data[k] = v;
    }

    fn swap_cols(&mut self, a: usize, b: usize) {
        for i in 0..self.rows {
            let (ia, ib) = (self.idx(i, a), self.idx(i, b));
            self.data.swap(ia, ib);
        }
    }

    /// Whether a column is contiguous by type (Eigen vectorizes sums over it).
    fn packed_cols(&self) -> bool {
        self.order == Order::Col
    }

    /// Whether a row is contiguous by type.
    fn packed_rows(&self) -> bool {
        self.order == Order::Row
    }

    /// The largest absolute coefficient: `cwiseAbs().maxCoeff()`.
    pub fn max_abs(&self) -> f64 {
        self.data
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| if m < v.abs() { v.abs() } else { m })
    }
}

/// A sum one value at a time (Eigen's DefaultTraversal): `c(0) + c(1) + ...`.
#[inline]
pub fn sum_seq(n: usize, c: impl Fn(usize) -> f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut res = c(0);
    for i in 1..n {
        res += c(i);
    }
    res
}

/// A sum in packets when `packed`, else one value at a time.
#[inline]
fn sum(packed: bool, n: usize, c: impl Fn(usize) -> f64) -> f64 {
    if packed {
        sum_by(n, c)
    } else {
        sum_seq(n, c)
    }
}

/// One row of the row-major matrix-vector kernel: `sum_j a(j) * b(j)` in two
/// lanes (packet multiply-adds), the lanes added, then the last odd term.
#[inline]
pub(super) fn gemv_row(n: usize, a: impl Fn(usize) -> f64, b: impl Fn(usize) -> f64) -> f64 {
    let (mut c0, mut c1) = (0.0, 0.0);
    let mut j = 0;
    while j + PACKET <= n {
        c0 = pmadd(a(j), b(j), c0);
        c1 = pmadd(a(j + 1), b(j + 1), c1);
        j += PACKET;
    }
    let mut cc = c0 + c1;
    while j < n {
        cc += a(j) * b(j);
        j += 1;
    }
    cc
}

/// Row `i` of `rows` of the column-major kernel: `sum_j a(j) * b(j)`, one
/// column after the other from 0 (one block of columns: fewer than 128). The
/// rows go by packets of 2 (multiply-adds) but for an odd last one (scalar).
#[inline]
pub(super) fn gemv_col(i: usize, rows: usize, n: usize, a: impl Fn(usize) -> f64, b: impl Fn(usize) -> f64) -> f64 {
    assert!(n < 128, "the column-major kernel splits 128 columns or more in blocks");
    let packed = i < rows / PACKET * PACKET;
    let mut c = 0.0;
    for j in 0..n {
        c = if packed { pmadd(a(j), b(j), c) } else { c + a(j) * b(j) };
    }
    c
}

/// `A * x`, assigned to a new vector (Eigen: zero, then add 1 * the product).
pub fn mat_vec(a: &Mat, x: &[f64]) -> Vec<f64> {
    mat_vec_rows(a, a.rows, x)
}

/// `A.topRows(rows) * x`.
pub fn mat_vec_rows(a: &Mat, rows: usize, x: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), a.cols);
    if rows == 1 {
        // one row: an inner product (dot), in packets when the row is contiguous
        return vec![0.0 + 1.0 * sum(a.packed_rows(), a.cols, |j| a.get(0, j) * x[j])];
    }
    match a.order {
        Order::Row => (0..rows)
            .map(|i| 0.0 + 1.0 * gemv_row(a.cols, |j| a.get(i, j), |j| x[j]))
            .collect(),
        Order::Col => {
            // the column-major kernel, a column at a time for every row together
            assert!(
                a.cols < 128,
                "the column-major kernel splits 128 columns or more in blocks"
            );
            let packed = rows / PACKET * PACKET;
            let mut c = vec![0.0; rows];
            for (j, &xj) in x.iter().enumerate() {
                let col = &a.data[j * a.rows..j * a.rows + rows];
                for i in 0..packed {
                    c[i] = pmadd(col[i], xj, c[i]);
                }
                for i in packed..rows {
                    c[i] += col[i] * xj;
                }
            }
            c.iter().map(|&v| v * 1.0 + 0.0).collect()
        }
    }
}

/// `A.transpose() * r`.
pub fn mat_t_vec(a: &Mat, r: &[f64]) -> Vec<f64> {
    assert_eq!(r.len(), a.rows);
    if a.cols == 1 {
        return vec![0.0 + 1.0 * sum(a.packed_cols(), a.rows, |i| a.get(i, 0) * r[i])];
    }
    // A^T is row-major when A is column-major, and the other way round
    match a.order {
        Order::Col => {
            // the row-major kernel's two lanes of every output, advanced together
            let (n, p) = (a.rows, a.cols);
            let even = n / PACKET * PACKET;
            let (mut c0, mut c1) = (vec![0.0; p], vec![0.0; p]);
            let mut i = 0;
            while i < even {
                let (r0, r1) = (r[i], r[i + 1]);
                for j in 0..p {
                    c0[j] = pmadd(a.data[j * n + i], r0, c0[j]);
                    c1[j] = pmadd(a.data[j * n + i + 1], r1, c1[j]);
                }
                i += PACKET;
            }
            (0..p)
                .map(|j| {
                    let mut cc = c0[j] + c1[j];
                    for i in even..n {
                        cc += a.data[j * n + i] * r[i];
                    }
                    0.0 + 1.0 * cc
                })
                .collect()
        }
        Order::Row => (0..a.cols)
            .map(|j| gemv_col(j, a.cols, a.rows, |i| a.get(i, j), |i| r[i]) * 1.0 + 0.0)
            .collect(),
    }
}

/// Eigen's ColPivHouseholderQR of a matrix (the decomposition as
/// `computeInPlace` runs it).
pub struct ColPivQr {
    qr: Mat,
    h_coeffs: Vec<f64>,
    /// `m_colsPermutation.indices()`.
    perm: Vec<usize>,
    nonzero_pivots: usize,
    max_pivot: f64,
    threshold: Option<f64>,
}

impl ColPivQr {
    pub fn new(mut qr: Mat) -> ColPivQr {
        let (rows, cols) = (qr.rows, qr.cols);
        let size = rows.min(cols);
        let packed = qr.packed_cols();
        let col_norm = |m: &Mat, j: usize, from: usize| {
            sum(packed, rows - from, |i| {
                let v = m.get(from + i, j);
                v * v
            })
            .sqrt()
        };

        let mut h_coeffs = vec![0.0; size];
        let mut temp = vec![0.0; cols];
        let mut transpositions = vec![0usize; cols];
        let mut norms_direct: Vec<f64> = (0..cols).map(|k| col_norm(&qr, k, 0)).collect();
        let mut norms_updated = norms_direct.clone();

        let eps = f64::EPSILON;
        let max_norm = norms_updated
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| if m < v { v } else { m });
        let t = max_norm * eps;
        let threshold_helper = t * t / rows as f64;
        let norm_downdate_threshold = eps.sqrt();

        let mut nonzero_pivots = size;
        let mut max_pivot = 0.0f64;

        for k in 0..size {
            // the column of the biggest norm (the first one on ties)
            let mut biggest = k;
            for j in k + 1..cols {
                if norms_updated[j] > norms_updated[biggest] {
                    biggest = j;
                }
            }
            let biggest_sq = norms_updated[biggest] * norms_updated[biggest];
            if nonzero_pivots == size && biggest_sq < threshold_helper * (rows - k) as f64 {
                nonzero_pivots = k;
            }
            transpositions[k] = biggest;
            if k != biggest {
                qr.swap_cols(k, biggest);
                norms_updated.swap(k, biggest);
                norms_direct.swap(k, biggest);
            }

            // the Householder vector of column k, below the diagonal
            let (tau, beta) = make_householder_in_place(&mut qr, k, k);
            h_coeffs[k] = tau;
            qr.set(k, k, beta);
            if beta.abs() > max_pivot {
                max_pivot = beta.abs();
            }

            apply_householder_left(&mut qr, k, k + 1, k, tau, &mut temp[(k + 1).min(cols)..]);

            // the norms of the remaining columns (LAPACK's stable downdate)
            for j in k + 1..cols {
                if norms_updated[j] != 0.0 {
                    let mut temp = qr.get(k, j).abs() / norms_updated[j];
                    temp = (1.0 + temp) * (1.0 - temp);
                    temp = if temp < 0.0 { 0.0 } else { temp };
                    let ratio = norms_updated[j] / norms_direct[j];
                    let temp2 = temp * (ratio * ratio);
                    if temp2 <= norm_downdate_threshold {
                        norms_direct[j] = col_norm(&qr, j, k + 1);
                        norms_updated[j] = norms_direct[j];
                    } else {
                        norms_updated[j] *= temp.sqrt();
                    }
                }
            }
        }

        let mut perm: Vec<usize> = (0..cols).collect();
        for (k, &t) in transpositions.iter().enumerate().take(size) {
            perm.swap(k, t);
        }
        ColPivQr {
            qr,
            h_coeffs,
            perm,
            nonzero_pivots,
            max_pivot,
            threshold: None,
        }
    }

    /// `setThreshold`: the relative threshold of `rank`.
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = Some(threshold);
    }

    fn threshold(&self) -> f64 {
        self.threshold
            .unwrap_or(f64::EPSILON * self.qr.rows.min(self.qr.cols) as f64)
    }

    /// The rank: the pivots above threshold times the biggest.
    pub fn rank(&self) -> usize {
        let t = self.max_pivot.abs() * self.threshold();
        (0..self.nonzero_pivots)
            .filter(|&i| self.qr.get(i, i).abs() > t)
            .count()
    }

    pub fn nonzero_pivots(&self) -> usize {
        self.nonzero_pivots
    }

    /// The least-squares solution of A x = b (`solve`).
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let (rows, cols) = (self.qr.rows, self.qr.cols);
        assert_eq!(b.len(), rows);
        let nzp = self.nonzero_pivots;
        let mut dst = vec![0.0; cols];
        if nzp == 0 {
            return dst;
        }
        let mut c = b.to_vec();
        // Q^T c: the reflectors in order (the adjoint sequence)
        for k in 0..nzp {
            apply_householder_left_vec(&self.qr, k, self.h_coeffs[k], &mut c[k..]);
        }
        solve_upper(&self.qr, nzp, &mut c[..nzp]);
        for i in 0..nzp {
            dst[self.perm[i]] = c[i];
        }
        dst
    }

    /// `householderQ() * Identity(rows, n)`: the first `n` columns of Q,
    /// column-major as the C++'s MatrixXd.
    pub fn q_thin(&self, n: usize) -> Mat {
        let rows = self.qr.rows;
        let mut dst = Mat::from_fn(rows, n, Order::Col, |i, j| if i == j { 1.0 } else { 0.0 });
        let len = self.h_coeffs.len();
        assert!(len < 48 || n == 1, "Eigen applies 48 reflectors or more by blocks");
        // a square identity is known to stay one in the columns before k
        let identity = n == rows;
        let mut work = vec![0.0; n];
        // Q = H_0 H_1 ... applied to the identity: the last reflector first
        for k in (0..len).rev() {
            let c0 = if identity { k } else { 0 };
            apply_reflector_to(&self.qr, k, self.h_coeffs[k], &mut dst, k, c0, &mut work);
        }
        dst
    }
}

/// makeHouseholderInPlace on column `j` of `m` from row `r`: the essential
/// part stored below, returns (tau, beta).
fn make_householder_in_place(m: &mut Mat, r: usize, j: usize) -> (f64, f64) {
    let size = m.rows - r;
    let packed = m.packed_cols();
    let tail_sq_norm = if size == 1 {
        0.0
    } else {
        sum(packed, size - 1, |i| {
            let v = m.get(r + 1 + i, j);
            v * v
        })
    };
    let c0 = m.get(r, j);
    let tol = f64::MIN_POSITIVE;
    if tail_sq_norm <= tol {
        for i in r + 1..m.rows {
            m.set(i, j, 0.0);
        }
        (0.0, c0)
    } else {
        let mut beta = (c0 * c0 + tail_sq_norm).sqrt();
        if c0 >= 0.0 {
            beta = -beta;
        }
        let d = c0 - beta;
        for i in r + 1..m.rows {
            let v = m.get(i, j) / d;
            m.set(i, j, v);
        }
        (((beta - c0) / beta), beta)
    }
}

/// applyHouseholderOnTheLeft on the block of `m` from row `r` and column `c0`
/// on, with the essential vector stored in column `ej` of `m` below row `r`.
fn apply_householder_left(m: &mut Mat, r: usize, c0: usize, ej: usize, tau: f64, work: &mut [f64]) {
    let (rows, cols) = (m.rows - r, m.cols - c0);
    if rows == 1 {
        for j in c0..m.cols {
            let v = m.get(r, j) * (1.0 - tau);
            m.set(r, j, v);
        }
        return;
    }
    if tau == 0.0 || cols == 0 {
        return;
    }
    let packed = m.packed_cols();
    let n = rows - 1;
    // tmp = essential^T * bottom
    for jj in 0..cols {
        let j = c0 + jj;
        let e = |i: usize| m.get(r + 1 + i, ej);
        let b = |i: usize| m.get(r + 1 + i, j);
        work[jj] = if cols == 1 {
            0.0 + 1.0 * sum(packed, n, |i| e(i) * b(i))
        } else if packed {
            0.0 + 1.0 * gemv_row(n, b, e)
        } else {
            gemv_col(jj, cols, n, b, e) * 1.0 + 0.0
        };
    }
    for jj in 0..cols {
        let j = c0 + jj;
        work[jj] += m.get(r, j);
        let v = m.get(r, j) - tau * work[jj];
        m.set(r, j, v);
    }
    // bottom -= (tau * essential) * tmp
    for i in 0..n {
        let te = tau * m.get(r + 1 + i, ej);
        for jj in 0..cols {
            let j = c0 + jj;
            let v = m.get(r + 1 + i, j) - work[jj] * te;
            m.set(r + 1 + i, j, v);
        }
    }
}

/// applyHouseholderOnTheLeft of reflector `k` of `qr` on the vector `c`
/// (the rows from k on of the right-hand side).
fn apply_householder_left_vec(qr: &Mat, k: usize, tau: f64, c: &mut [f64]) {
    if c.len() == 1 {
        c[0] *= 1.0 - tau;
        return;
    }
    if tau == 0.0 {
        return;
    }
    let n = c.len() - 1;
    let e = |i: usize| qr.get(k + 1 + i, k);
    let mut tmp = 0.0 + 1.0 * sum(qr.packed_cols(), n, |i| e(i) * c[1 + i]);
    tmp += c[0];
    c[0] -= tau * tmp;
    for i in 0..n {
        c[1 + i] -= tmp * (tau * e(i));
    }
}

/// applyHouseholderOnTheLeft of reflector `k` of `qr` on the block of the
/// column-major `dst` from row `r` and column `c0` on.
fn apply_reflector_to(qr: &Mat, k: usize, tau: f64, dst: &mut Mat, r: usize, c0: usize, work: &mut [f64]) {
    let (rows, cols) = (dst.rows - r, dst.cols - c0);
    if rows == 1 {
        for j in c0..dst.cols {
            let v = dst.get(r, j) * (1.0 - tau);
            dst.set(r, j, v);
        }
        return;
    }
    if tau == 0.0 || cols == 0 {
        return;
    }
    let n = rows - 1;
    let packed = qr.packed_cols(); // the essential vector
    let e = |i: usize| qr.get(k + 1 + i, k);
    for jj in 0..cols {
        let b = |i: usize| dst.get(r + 1 + i, c0 + jj);
        // dst is column-major: one column, an inner product in packets when the
        // essential vector is contiguous too; several, the row-major kernel
        // over dst^T
        work[jj] = if cols == 1 {
            0.0 + 1.0 * sum(packed, n, |i| e(i) * b(i))
        } else {
            0.0 + 1.0 * gemv_row(n, b, e)
        };
    }
    for jj in 0..cols {
        work[jj] += dst.get(r, c0 + jj);
        let v = dst.get(r, c0 + jj) - tau * work[jj];
        dst.set(r, c0 + jj, v);
    }
    for i in 0..n {
        let te = tau * e(i);
        for jj in 0..cols {
            let v = dst.get(r + 1 + i, c0 + jj) - work[jj] * te;
            dst.set(r + 1 + i, c0 + jj, v);
        }
    }
}

const PANEL: usize = 8;

/// The upper-triangular solve of the top-left `n` x `n` block of `m` in place
/// on `rhs` (`triangularView<Upper>().solveInPlace`), by panels of 8 rows.
fn solve_upper(m: &Mat, n: usize, rhs: &mut [f64]) {
    match m.order {
        Order::Col => {
            let mut pi = n;
            while pi > 0 {
                let width = pi.min(PANEL);
                let start = pi - width;
                for k in 0..width {
                    let i = pi - k - 1;
                    if rhs[i] != 0.0 {
                        rhs[i] /= m.get(i, i);
                        let r = width - k - 1;
                        let s = i - r;
                        for t in 0..r {
                            rhs[s + t] -= rhs[i] * m.get(s + t, i);
                        }
                    }
                }
                // the rows above the panel: rhs[0..start] -= M[0..start, panel] * rhs[panel]
                for row in 0..start {
                    let c = gemv_col(row, start, width, |j| m.get(row, start + j), |j| rhs[start + j]);
                    rhs[row] += -c; // c * alpha, alpha = -1
                }
                pi -= width;
            }
        }
        Order::Row => {
            let mut pi = n;
            while pi > 0 {
                let width = pi.min(PANEL);
                let r = n - pi;
                let start = pi - width;
                if r > 0 {
                    for row in start..pi {
                        let cc = gemv_row(r, |j| m.get(row, pi + j), |j| rhs[pi + j]);
                        rhs[row] += -cc; // alpha * cc, alpha = -1
                    }
                }
                for k in 0..width {
                    let i = pi - k - 1;
                    let s = i + 1;
                    if k > 0 {
                        // a row segment of a row-major matrix and a vector: in packets
                        rhs[i] -= sum_by(k, |t| m.get(i, s + t) * rhs[s + t]);
                    }
                    if rhs[i] != 0.0 {
                        rhs[i] /= m.get(i, i);
                    }
                }
                pi -= width;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual(a: &Mat, x: &[f64], b: &[f64]) -> f64 {
        mat_vec(a, x)
            .iter()
            .zip(b)
            .map(|(u, v)| (u - v).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn solves_square_and_least_squares_in_both_orders() {
        for order in [Order::Col, Order::Row] {
            for &(rows, cols) in &[(3, 3), (7, 4), (30, 8), (12, 10)] {
                let a = Mat::from_fn(rows, cols, order, |i, j| {
                    ((i * 7 + j * 13) % 11) as f64 + 0.1 * (i * j) as f64
                });
                let x: Vec<f64> = (0..cols).map(|j| j as f64 - 1.5).collect();
                let b = mat_vec(&a, &x);
                let qr = ColPivQr::new(a.clone());
                assert_eq!(qr.rank(), cols);
                let got = qr.solve(&b);
                assert!(residual(&a, &got, &b) < 1e-9, "{order:?} {rows}x{cols}");
            }
        }
    }

    #[test]
    fn rank_deficient_and_thin_q() {
        // the third column is the sum of the first two
        let a = Mat::from_fn(6, 3, Order::Col, |i, j| match j {
            0 => i as f64,
            1 => 1.0,
            _ => i as f64 + 1.0,
        });
        let qr = ColPivQr::new(a);
        assert_eq!(qr.rank(), 2);
        let q = qr.q_thin(2);
        for j in 0..2 {
            let n: f64 = (0..6).map(|i| q.get(i, j) * q.get(i, j)).sum();
            assert!((n - 1.0).abs() < 1e-12);
        }
    }
}
