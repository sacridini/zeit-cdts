//! Generalized least squares under a spatial covariance: remotePARTS's
//! `fitGLS` (Ives et al. 2021, doi:10.1016/j.rse.2021.112678), the regression
//! of PARTS on one partition of pixels: y = X b + e with Cov(e) = s2 ((1 -
//! nugget) V + nugget I), and the F test of X against a null design X0.
//!
//! remotePARTS finds the maximum-likelihood nugget with a Cholesky factor of
//! the n x n covariance at each step of its optimizer. Here V is reduced once
//! to a tridiagonal T = Q'VQ (Householder), in which (1 - nugget) V + nugget I
//! is (1 - nugget) T + nugget I: each step is then O(n). With a given nugget,
//! one Cholesky factor. Both give the cross-products [X X0 y]' V^-1 [X X0 y],
//! from which every statistic follows.

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch};
use faer::linalg::householder::{
    apply_block_householder_sequence_transpose_on_the_left_in_place_scratch,
    apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj,
};
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::{Conj, Mat, Par};

use crate::dist::{f_upper, t_two_sided};
use crate::optimize::fmin_netlib;
use crate::parallel;

/// The nugget of the covariance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nugget {
    Fixed(f64),
    /// The maximum-likelihood nugget, as remotePARTS's `optimize_nugget` (on
    /// [0.001, 0.999] to within eps^(1/4)).
    Ml,
}

/// The fit of the null design (when it has fewer columns than X) and the F
/// test of X against it.
#[derive(Clone, Debug, PartialEq)]
pub struct NullFit {
    pub coefficients: Vec<f64>,
    pub se: Vec<f64>,
    pub sse: f64,
    /// SSE0 / (n - p), as remotePARTS (the full model's degrees of freedom).
    pub mse: f64,
    pub loglik: f64,
    /// (SSE0 - SSE) / (p - p0)
    pub msr: f64,
    pub f: f64,
    pub df1: usize,
    pub df2: usize,
    pub p: f64,
}

/// A fit.
#[derive(Clone, Debug, PartialEq)]
pub struct GlsFit {
    pub coefficients: Vec<f64>,
    pub se: Vec<f64>,
    /// Signed.
    pub t: Vec<f64>,
    /// Two-sided, on `df` degrees of freedom.
    pub p: Vec<f64>,
    /// The covariance of the coefficients (p x p, by column).
    pub covar: Vec<f64>,
    pub sse: f64,
    pub mse: f64,
    /// n - p
    pub df: usize,
    /// log det((1 - nugget) V + nugget I)
    pub logdet_v: f64,
    pub loglik: f64,
    pub nugget: f64,
    pub null: Option<NullFit>,
}

/// Fits `y` on `x` (n x `p`, by column) under the covariance `v` (n x n, by
/// column; its lower triangle is read), and tests it against `x0` (n x `p0`,
/// by column; usually a column of ones). None when the covariance or the
/// design is singular.
#[allow(clippy::too_many_arguments)]
pub fn fit(
    v: &[f64],
    n: usize,
    x: &[f64],
    p: usize,
    x0: &[f64],
    p0: usize,
    y: &[f64],
    nugget: Nugget,
    n_jobs: i32,
) -> Option<GlsFit> {
    assert_eq!(v.len(), n * n, "v must be n x n");
    assert_eq!(x.len(), n * p, "x must be n x p");
    assert_eq!(x0.len(), n * p0, "x0 must be n x p0");
    assert_eq!(y.len(), n, "y must have n values");
    if p == 0 || n <= p {
        return None;
    }
    let threads = parallel::threads(n_jobs);
    let run = |par: Par| {
        let k = p + p0 + 1;
        let z = Mat::from_fn(n, k, |i, j| {
            if j < p {
                x[j * n + i]
            } else if j < p + p0 {
                x0[(j - p) * n + i]
            } else {
                y[i]
            }
        });
        let (gram, logdet, nugget) = match nugget {
            Nugget::Fixed(nugget) => {
                let (g, logdet) = cholesky_gram(v, n, nugget, z, par)?;
                (g, logdet, nugget)
            }
            Nugget::Ml => {
                let tri = Tridiagonal::new(v, n, &z, par);
                let nugget = tri.ml_nugget(p, k);
                let all: Vec<usize> = (0..k).collect();
                let mut g = vec![0.0; k * k];
                let logdet = tri.gram(nugget, &all, &mut g);
                if logdet.is_nan() {
                    return None;
                }
                (g, logdet, nugget)
            }
        };
        statistics(&gram, k, logdet, n, p, p0, nugget)
    };
    if threads == 1 {
        run(Par::Seq)
    } else {
        parallel::pool(threads).install(|| run(Par::rayon(threads)))
    }
}

/// The lower triangle of (1 - nugget) V + nugget I.
fn covariance(v: &[f64], n: usize, nugget: f64) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| {
        if i < j {
            0.0
        } else if i == j {
            (1.0 - nugget) * v[j * n + i] + nugget
        } else {
            (1.0 - nugget) * v[j * n + i]
        }
    })
}

/// The cross-products of the columns of `z` under the covariance's Cholesky
/// factor L (L^-1 z)'(L^-1 z) (k x k, by column), and its log determinant.
fn cholesky_gram(v: &[f64], n: usize, nugget: f64, mut z: Mat<f64>, par: Par) -> Option<(Vec<f64>, f64)> {
    let mut a = covariance(v, n, nugget);
    let mut mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(n, par, Default::default()));
    cholesky_in_place(
        a.as_mut(),
        Default::default(),
        par,
        MemStack::new(&mut mem),
        Default::default(),
    )
    .ok()?;
    let logdet = 2.0 * (0..n).map(|i| a[(i, i)].ln()).sum::<f64>();
    solve_lower_triangular_in_place(a.as_ref(), z.as_mut(), par);
    let k = z.ncols();
    let g = z.transpose() * &z;
    Some(((0..k * k).map(|e| g[(e % k, e / k)]).collect(), logdet))
}

/// V = Q T Q' with T tridiagonal, and Q'z by row.
struct Tridiagonal {
    n: usize,
    k: usize,
    diag: Vec<f64>,
    off: Vec<f64>,
    /// Q'z, n x k by row
    qz: Vec<f64>,
}

impl Tridiagonal {
    fn new(v: &[f64], n: usize, z: &Mat<f64>, par: Par) -> Tridiagonal {
        let k = z.ncols();
        let mut a = covariance(v, n, 0.0);
        let mut z = z.clone();
        if n > 1 {
            let bs = faer::linalg::qr::no_pivoting::factor::recommended_block_size::<f64>(n, n);
            let mut h = Mat::<f64>::zeros(bs, n - 1);
            let mut mem = MemBuffer::new(StackReq::any_of(&[
                tridiag_in_place_scratch::<f64>(n, par, Default::default()),
                apply_block_householder_sequence_transpose_on_the_left_in_place_scratch::<f64>(n - 1, bs, k),
            ]));
            tridiag_in_place(a.as_mut(), h.as_mut(), par, MemStack::new(&mut mem), Default::default());
            apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj(
                a.as_ref().submatrix(1, 0, n - 1, n - 1),
                h.as_ref(),
                Conj::No,
                z.as_mut().subrows_mut(1, n - 1),
                par,
                MemStack::new(&mut mem),
            );
        }
        Tridiagonal {
            n,
            k,
            diag: (0..n).map(|i| a[(i, i)]).collect(),
            off: (1..n).map(|i| a[(i, i - 1)]).collect(),
            qz: (0..n * k).map(|e| z[(e / k, e % k)]).collect(),
        }
    }

    /// The cross-products of the columns `cols` of z under (1 - nugget) V +
    /// nugget I into `g` (|cols|^2, by column), from the LDL' of (1 - nugget)
    /// T + nugget I: its log determinant, or NaN when it is not positive
    /// definite.
    fn gram(&self, nugget: f64, cols: &[usize], g: &mut [f64]) -> f64 {
        let m = cols.len();
        g[..m * m].fill(0.0);
        let mut w = vec![0.0; 2 * m];
        let (mut d, mut logdet) = (0.0, 0.0);
        for i in 0..self.n {
            let a = (1.0 - nugget) * self.diag[i] + nugget;
            let row = &self.qz[i * self.k..(i + 1) * self.k];
            let (prev, cur) = w.split_at_mut(m);
            if i == 0 {
                d = a;
                for (c, &col) in cur.iter_mut().zip(cols) {
                    *c = row[col];
                }
            } else {
                let e = (1.0 - nugget) * self.off[i - 1];
                let l = e / d;
                d = a - l * e;
                prev.copy_from_slice(cur);
                for ((c, &col), pr) in cur.iter_mut().zip(cols).zip(prev.iter()) {
                    *c = row[col] - l * pr;
                }
            }
            if !(d > 0.0) {
                return f64::NAN;
            }
            logdet += d.ln();
            for b in 0..m {
                let wb = cur[b] / d;
                for a in b..m {
                    g[b * m + a] += cur[a] * wb;
                }
            }
        }
        for b in 0..m {
            for a in b + 1..m {
                g[a * m + b] = g[b * m + a];
            }
        }
        logdet
    }

    /// remotePARTS's optimize_nugget: the nugget that maximizes the
    /// likelihood of the regression on the first `p` columns of z (y the last
    /// of `k`), its lower bound where the optimum is within the tolerance of it.
    fn ml_nugget(&self, p: usize, k: usize) -> f64 {
        let (lower, upper) = (0.001, 0.999);
        let tol = f64::EPSILON.sqrt().sqrt();
        let cols: Vec<usize> = (0..p).chain([k - 1]).collect();
        let n = self.n as f64;
        let mut g = vec![0.0; cols.len() * cols.len()];
        let minus_loglik = |nugget: f64| {
            let logdet = self.gram(nugget, &cols, &mut g);
            match regress(&g, cols.len(), p) {
                Some((_, _, sse)) if !logdet.is_nan() => {
                    0.5 * (n * (2.0 * std::f64::consts::PI).ln() + n * (sse / n).ln() + logdet + n)
                }
                _ => f64::MAX,
            }
        };
        let x = fmin_netlib(lower, upper, minus_loglik, tol);
        if lower + tol >= x {
            lower
        } else {
            x
        }
    }
}

/// The least squares of the last column of the Gram matrix `g` (m x m, by
/// column) on its first `p`: the coefficients, the inverse of their
/// cross-products (p x p, by column), and the residual sum of squares.
fn regress(g: &[f64], m: usize, p: usize) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    // Cholesky of g (lower, by column)
    let mut l = vec![0.0; m * m];
    for j in 0..m {
        let mut s = g[j * m + j];
        for k in 0..j {
            s -= l[k * m + j] * l[k * m + j];
        }
        if !(s > 0.0) {
            return None;
        }
        let ljj = s.sqrt();
        l[j * m + j] = ljj;
        for i in j + 1..m {
            let mut s = g[j * m + i];
            for k in 0..j {
                s -= l[k * m + i] * l[k * m + j];
            }
            l[j * m + i] = s / ljj;
        }
    }
    let sse = l[p * m + p] * l[p * m + p];
    // L_xx' b = the last row of L
    let mut beta = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = l[i * m + p];
        for k in i + 1..p {
            s -= l[i * m + k] * beta[k];
        }
        beta[i] = s / l[i * m + i];
    }
    // (X'X)^-1 = L_xx^-T L_xx^-1: the columns of L_xx^-1 first
    let mut li = vec![0.0; p * p];
    for c in 0..p {
        for i in c..p {
            let mut s = if i == c { 1.0 } else { 0.0 };
            for k in c..i {
                s -= l[k * m + i] * li[c * p + k];
            }
            li[c * p + i] = s / l[i * m + i];
        }
    }
    let mut inv = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..=a {
            let s: f64 = (a..p).map(|k| li[a * p + k] * li[b * p + k]).sum();
            inv[a * p + b] = s;
            inv[b * p + a] = s;
        }
    }
    Some((beta, inv, sse))
}

/// The Gram matrix of the columns `cols` of `g` (k x k, by column).
fn sub_gram(g: &[f64], k: usize, cols: &[usize]) -> Vec<f64> {
    let m = cols.len();
    (0..m * m).map(|e| g[cols[e / m] * k + cols[e % m]]).collect()
}

/// fitGLS_cpp's statistics from the cross-products of [X X0 y].
fn statistics(g: &[f64], k: usize, logdet: f64, n: usize, p: usize, p0: usize, nugget: f64) -> Option<GlsFit> {
    let nf = n as f64;
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let cols: Vec<usize> = (0..p).chain([k - 1]).collect();
    let (beta, inv, sse) = regress(&sub_gram(g, k, &cols), p + 1, p)?;
    let df = n - p;
    let mse = sse / df as f64;
    let covar: Vec<f64> = inv.iter().map(|v| v * mse).collect();
    let se: Vec<f64> = (0..p).map(|i| covar[i * p + i].sqrt()).collect();
    let t: Vec<f64> = beta.iter().zip(&se).map(|(b, s)| b / s).collect();
    let pv = t.iter().map(|&t| t_two_sided(t, df as f64)).collect();
    let loglik = -0.5 * (nf * ln_2pi + nf * (sse / nf).ln() + logdet + nf);
    let null = if p0 > 0 && p0 < p {
        let cols0: Vec<usize> = (p..p + p0).chain([k - 1]).collect();
        let (beta0, inv0, sse0) = regress(&sub_gram(g, k, &cols0), p0 + 1, p0)?;
        let mse0 = sse0 / df as f64;
        let ssr = sse0 - sse;
        let (df1, df2) = (p - p0, df);
        let f = df2 as f64 / df1 as f64 * ssr / sse;
        Some(NullFit {
            coefficients: beta0,
            se: (0..p0).map(|i| (mse0 * inv0[i * p0 + i]).sqrt()).collect(),
            sse: sse0,
            mse: mse0,
            loglik: -0.5 * (nf * ln_2pi + nf * ((n - p0) as f64 * mse0 / nf).ln() + logdet + nf),
            msr: ssr / df1 as f64,
            f,
            df1,
            df2,
            p: f_upper(f, df1 as f64, df2 as f64),
        })
    } else {
        None
    };
    Some(GlsFit {
        coefficients: beta,
        se,
        t,
        p: pv,
        covar,
        sse,
        mse,
        df,
        logdet_v: logdet,
        loglik,
        nugget,
        null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An exponential covariance on a line of points, a design of a mean and
    /// a slope, and a series.
    fn problem(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let pos: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37) % 1.0).collect();
        let v = (0..n * n)
            .map(|e| (-(pos[e / n] - pos[e % n]).abs() / 0.2).exp())
            .collect();
        let x = (0..2 * n).map(|e| if e < n { 1.0 } else { pos[e - n] }).collect();
        let y = (0..n)
            .map(|i| 0.3 + 0.5 * pos[i] + ((i * 7919) % 13) as f64 / 40.0)
            .collect();
        (v, x, vec![1.0; n], y)
    }

    #[test]
    fn tridiagonal_and_cholesky_give_the_same_fit() {
        let n = 120;
        let (v, x, x0, y) = problem(n);
        let a = fit(&v, n, &x, 2, &x0, 1, &y, Nugget::Fixed(0.3), 1).unwrap();
        let z = Mat::from_fn(n, 4, |i, j| match j {
            0 | 1 => x[j * n + i],
            2 => 1.0,
            _ => y[i],
        });
        let tri = Tridiagonal::new(&v, n, &z, Par::Seq);
        let mut g = vec![0.0; 16];
        let logdet = tri.gram(0.3, &[0, 1, 2, 3], &mut g);
        let b = statistics(&g, 4, logdet, n, 2, 1, 0.3).unwrap();
        assert!((a.logdet_v - b.logdet_v).abs() < 1e-9 * a.logdet_v.abs());
        for j in 0..2 {
            assert!((a.coefficients[j] - b.coefficients[j]).abs() < 1e-10 * a.se[j]);
            assert!((a.se[j] - b.se[j]).abs() < 1e-10 * a.se[j]);
        }
        let (na, nb) = (a.null.unwrap(), b.null.unwrap());
        assert!((na.f - nb.f).abs() < 1e-9 * na.f);
    }

    #[test]
    fn same_fit_on_any_threads() {
        let n = 300;
        let (v, x, x0, y) = problem(n);
        let a = fit(&v, n, &x, 2, &x0, 1, &y, Nugget::Ml, 1).unwrap();
        let b = fit(&v, n, &x, 2, &x0, 1, &y, Nugget::Ml, 4).unwrap();
        assert!((a.nugget - b.nugget).abs() < 1e-9);
        assert!((a.coefficients[1] - b.coefficients[1]).abs() < 1e-9 * a.se[1]);
    }
}
