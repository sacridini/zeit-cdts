//! Eigen 3.4's matrix-matrix product for `X^T X` and its LDLT (lower, with
//! diagonal pivoting), with the same operations in the same order as the C++
//! build, so the same bits.
//!
//! The product: below 20 (depth + rows + columns) Eigen takes the
//! coefficient-based path, each coefficient a sum in packets; above, its GEBP
//! kernel, whose order per coefficient depends on the block of rows it falls
//! in (Eigen/src/Core/products/GeneralBlockPanelKernel.h, packets of 2
//! doubles, blocks of MR rows): rows by 6 (NEON only) and by 4 and the last
//! odd row sum over the depth one term after the other; rows by 2, in the
//! columns taken by 4, alternate two accumulators over the depth unrolled by
//! 8, then add them, then the rest. The packet rows multiply-add (fused on
//! NEON), the last odd row does not.
//! The depth is taken in one block: Eigen splits it past ~400 (from the size
//! of the processor's L1 cache, so the C++'s own bits differ from a machine to
//! another there).

// The loops index rows and columns as Eigen's do.
#![allow(clippy::needless_range_loop)]

use super::qr::{sum_seq, Mat, Order};
use super::{pmadd, sum_by, MR};

/// `X.topRows(n).transpose() * X.topRows(n)` for a column-major `x`: p x p,
/// column-major, its lower triangle only (what the LDLT reads; Eigen's product
/// is not symmetric to the bit, the two triangles falling in different blocks
/// of rows). The upper triangle is left 0.
pub fn xtx(x: &Mat, n: usize) -> Mat {
    xtx_rows(x, 0, n)
}

/// `X.middleRows(start, n).transpose() * X.middleRows(start, n)`, lower triangle.
pub fn xtx_rows(x: &Mat, start: usize, n: usize) -> Mat {
    assert_eq!(x.order, Order::Col);
    assert!(start + n <= x.rows);
    let p = x.cols;
    let col = |j: usize| &x.data[j * x.rows + start..j * x.rows + start + n];
    let mut out = Mat::zeros(p, p, Order::Col);
    if p == 1 {
        // a one-coefficient product: Eigen's inner product (dot), in packets
        let a = col(0);
        out.set(0, 0, 0.0 + 1.0 * sum_by(n, |k| a[k] * a[k]));
        return out;
    }
    if n + p + p < 20 && n > 0 {
        // coefficient-based: a row of X^T (a column of X) and a column of X, in packets
        for j in 0..p {
            for i in j..p {
                let (a, b) = (col(i), col(j));
                out.set(i, j, sum_by(n, |k| a[k] * b[k]));
            }
        }
        return out;
    }

    // The accumulation of coefficient (i, j) follows from its block of rows:
    // packet rows multiply-add one term after the other, but rows by 2 in the
    // columns by 4 alternate two accumulators over the unrolled depth (CD);
    // the last odd row multiplies then adds. Computed by blocks of 2 rows x 4
    // columns, eight chains advancing together as in Eigen's kernel (each its
    // own operations, in order), over the lower triangle.
    let mc3 = if MR >= 6 { p / 6 * 6 } else { 0 };
    let mc2 = mc3 + (p - mc3) / 4 * 4;
    let mc1 = mc2 + (p - mc2) / 2 * 2;
    let cols4 = p / 4 * 4;
    let peeled = n & !7;
    let mut c = vec![0.0; p * p];
    let mut i = 0;
    while i < mc1 {
        let mut j = 0;
        while j <= i + 1 && j < p {
            let w = (p - j).min(4);
            let cd = i >= mc2 && j + 4 <= cols4;
            let block = match (w, cd) {
                (4, true) => block_cd::<4>(col(i), col(i + 1), j, &col, n, peeled),
                (4, false) => block_seq::<4>(col(i), col(i + 1), j, &col, n),
                (3, _) => block_seq::<3>(col(i), col(i + 1), j, &col, n),
                (2, _) => block_seq::<2>(col(i), col(i + 1), j, &col, n),
                _ => block_seq::<1>(col(i), col(i + 1), j, &col, n),
            };
            for (r, row) in block.iter().enumerate() {
                for (t, &v) in row.iter().enumerate().take(w) {
                    c[(j + t) * p + i + r] = v;
                }
            }
            j += 4;
        }
        i += 2;
    }
    if mc1 < p {
        // the last odd row: scalar
        let a = col(mc1);
        for j in 0..p {
            let b = col(j);
            let mut v = 0.0;
            for k in 0..n {
                v += a[k] * b[k];
            }
            c[j * p + mc1] = v;
        }
    }
    for j in 0..p {
        for i in j..p {
            out.set(i, j, c[j * p + i] * 1.0 + 0.0);
        }
    }
    out
}

/// Rows `a0` and `a1` times the W columns from `j` on, multiply-adds one term
/// after the other: [row][column].
#[inline(always)]
fn block_seq<'a, const W: usize>(
    a0: &[f64],
    a1: &[f64],
    j: usize,
    col: &impl Fn(usize) -> &'a [f64],
    n: usize,
) -> [[f64; 4]; 2] {
    let b: [&[f64]; W] = std::array::from_fn(|t| &col(j + t)[..n]);
    let (a0, a1) = (&a0[..n], &a1[..n]);
    let (mut c0, mut c1) = ([0.0; W], [0.0; W]);
    for k in 0..n {
        for t in 0..W {
            c0[t] = pmadd(a0[k], b[t][k], c0[t]);
            c1[t] = pmadd(a1[k], b[t][k], c1[t]);
        }
    }
    let mut out = [[0.0; 4]; 2];
    out[0][..W].copy_from_slice(&c0);
    out[1][..W].copy_from_slice(&c1);
    out
}

/// As `block_seq`, with two accumulators alternating over the depth unrolled
/// by 8 (even terms in c, odd ones in d), added, then the rest in c.
#[inline(always)]
fn block_cd<'a, const W: usize>(
    a0: &[f64],
    a1: &[f64],
    j: usize,
    col: &impl Fn(usize) -> &'a [f64],
    n: usize,
    peeled: usize,
) -> [[f64; 4]; 2] {
    let b: [&[f64]; W] = std::array::from_fn(|t| &col(j + t)[..n]);
    let (a0, a1) = (&a0[..n], &a1[..n]);
    let (mut c0, mut c1, mut d0, mut d1) = ([0.0; W], [0.0; W], [0.0; W], [0.0; W]);
    let mut k = 0;
    while k < peeled {
        for t in 0..W {
            c0[t] = pmadd(a0[k], b[t][k], c0[t]);
            c1[t] = pmadd(a1[k], b[t][k], c1[t]);
        }
        for t in 0..W {
            d0[t] = pmadd(a0[k + 1], b[t][k + 1], d0[t]);
            d1[t] = pmadd(a1[k + 1], b[t][k + 1], d1[t]);
        }
        k += 2;
    }
    for t in 0..W {
        c0[t] += d0[t];
        c1[t] += d1[t];
    }
    for k in peeled..n {
        for t in 0..W {
            c0[t] = pmadd(a0[k], b[t][k], c0[t]);
            c1[t] = pmadd(a1[k], b[t][k], c1[t]);
        }
    }
    let mut out = [[0.0; 4]; 2];
    out[0][..W].copy_from_slice(&c0);
    out[1][..W].copy_from_slice(&c1);
    out
}

/// Eigen's LDLT<MatrixXd, Lower> of a symmetric matrix (its lower triangle).
pub struct Ldlt {
    m: Mat,
    transpositions: Vec<usize>,
}

const PANEL: usize = 8;

impl Ldlt {
    pub fn new(a: &Mat) -> Ldlt {
        assert_eq!(a.rows, a.cols);
        let size = a.rows;
        let mut m = Mat::from_fn(size, size, Order::Col, |i, j| a.get(i, j));
        let mut tr: Vec<usize> = (0..size).collect();
        if size <= 1 {
            return Ldlt { m, transpositions: tr };
        }
        let mut temp = vec![0.0; size];
        for k in 0..size {
            // the largest diagonal coefficient (the first on ties)
            let mut big = k;
            for i in k + 1..size {
                if m.get(i, i).abs() > m.get(big, big).abs() {
                    big = i;
                }
            }
            tr[k] = big;
            if k != big {
                // the transposition, on the lower triangle only
                let s = size - big - 1;
                for j in 0..k {
                    let (u, v) = (m.get(k, j), m.get(big, j));
                    m.set(k, j, v);
                    m.set(big, j, u);
                }
                for t in 0..s {
                    let (u, v) = (m.get(big + 1 + t, k), m.get(big + 1 + t, big));
                    m.set(big + 1 + t, k, v);
                    m.set(big + 1 + t, big, u);
                }
                let (u, v) = (m.get(k, k), m.get(big, big));
                m.set(k, k, v);
                m.set(big, big, u);
                for i in k + 1..big {
                    let tmp = m.get(i, k);
                    let v = m.get(big, i);
                    m.set(i, k, v);
                    m.set(big, i, tmp);
                }
            }
            let rs = size - k - 1;
            if k > 0 {
                for j in 0..k {
                    temp[j] = m.get(j, j) * m.get(k, j);
                }
                // A10 * temp: a strided row, one term at a time
                let s = sum_seq(k, |j| m.get(k, j) * temp[j]);
                let v = m.get(k, k) - s;
                m.set(k, k, v);
                if rs > 0 {
                    // A21 -= A20 * temp: the column-major kernel (one row: an inner product)
                    for r in 0..rs {
                        let row = k + 1 + r;
                        let c = if rs == 1 {
                            -(sum_seq(k, |j| m.get(row, j) * temp[j]))
                        } else {
                            -super::qr::gemv_col(r, rs, k, |j| m.get(row, j), |j| temp[j])
                        };
                        let v = m.get(row, k) + c;
                        m.set(row, k, v);
                    }
                }
            }
            let akk = m.get(k, k);
            let valid = akk.abs() > 0.0;
            if k == 0 && !valid {
                // an all-zero diagonal: nothing more to do
                tr = (0..size).collect();
                return Ldlt { m, transpositions: tr };
            }
            if rs > 0 && valid {
                for r in k + 1..size {
                    let v = m.get(r, k) / akk;
                    m.set(r, k, v);
                }
            }
        }
        Ldlt { m, transpositions: tr }
    }

    /// The solution of A x = b (`solve`).
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.m.rows;
        assert_eq!(b.len(), n);
        let mut x = b.to_vec();
        for k in 0..n {
            x.swap(k, self.transpositions[k]);
        }
        // L, unit lower, column-major: panels of 8 from the top
        let mut pi = 0;
        while pi < n {
            let width = (n - pi).min(PANEL);
            for k in 0..width {
                let i = pi + k;
                if x[i] != 0.0 {
                    let r = width - k - 1;
                    for t in 0..r {
                        x[i + 1 + t] -= x[i] * self.m.get(i + 1 + t, i);
                    }
                }
            }
            let end = pi + width;
            for row in end..n {
                let c = super::qr::gemv_col(row - end, n - end, width, |j| self.m.get(row, pi + j), |j| x[pi + j]);
                x[row] += -c;
            }
            pi = end;
        }
        for i in 0..n {
            let d = self.m.get(i, i);
            if d.abs() > f64::MIN_POSITIVE {
                x[i] /= d;
            } else {
                x[i] = 0.0;
            }
        }
        // L^T, unit upper, row-major (a transposed view): panels of 8 from the bottom
        let u = |i: usize, j: usize| self.m.get(j, i);
        let mut pi = n;
        while pi > 0 {
            let width = pi.min(PANEL);
            let r = n - pi;
            let start = pi - width;
            if r > 0 {
                for row in start..pi {
                    let cc = super::qr::gemv_row(r, |j| u(row, pi + j), |j| x[pi + j]);
                    x[row] += -cc;
                }
            }
            for k in 0..width {
                let i = pi - k - 1;
                if k > 0 {
                    // a row of the transposed view (a column of m, contiguous): in packets
                    x[i] -= sum_by(k, |t| u(i, i + 1 + t) * x[i + 1 + t]);
                }
            }
            pi = start;
        }
        for k in (0..n).rev() {
            x.swap(k, self.transpositions[k]);
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_normal_equations() {
        for &(n, p) in &[(8, 3), (30, 5), (60, 8), (200, 10)] {
            let x = Mat::from_fn(n, p, Order::Col, |i, j| {
                ((i * 7 + j * 3) % 13) as f64 * 0.1 + (i as f64).sin() * j as f64
            });
            let beta: Vec<f64> = (0..p).map(|j| j as f64 * 0.5 - 1.0).collect();
            let y = super::super::qr::mat_vec(&x, &beta);
            let xty = super::super::qr::mat_t_vec(&x, &y);
            let got = Ldlt::new(&xtx(&x, n)).solve(&xty);
            for j in 0..p {
                assert!((got[j] - beta[j]).abs() < 1e-8, "{n}x{p}: {got:?}");
            }
        }
    }
}
