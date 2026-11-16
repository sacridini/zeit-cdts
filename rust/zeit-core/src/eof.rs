//! Temporal empirical orthogonal functions (EOF, principal components) of a
//! cube X(t, j) of T dates by N pixels, for cubes that do not fit in memory.
//!
//! The cube is read in blocks of pixels through the [`Blocks`] trait, already
//! anomalized and weighted: `Xw = X diag(sqrt w)`, one block being a T x cols
//! column-major matrix (`out[j * T + t]`), gaps 0. The decomposition
//! `Xw = U S V^T` is found from the T x T side, so only U and S need all the
//! pixels in one pass (the Gram matrix, [`Solver::Exact`]) or in a few passes
//! (the randomized range finder with power iterations of Halko, Martinsson &
//! Tropp 2011, Alg. 4.4, [`Solver::Randomized`]); the maps V come in one more
//! pass, block by block ([`loadings`]).
//!
//! References: North, Bell, Cahalan & Moeng 1982 (sampling errors of the
//! eigenvalues, [`north`]); Kaiser 1958 (varimax, [`varimax`]); Everson &
//! Sirovich 1995 (gappy projection, [`project_gappy`]).
//!
//! Determinism: the blocks have a fixed size (they are the caller's: 4096
//! pixels for [`Dense`]); each block's partial sums go in their own buffer,
//! made with sequential matrix products, and are added in block order, so the
//! parallelism (across blocks only) changes no bit of the result. The random
//! test matrix of the randomized solver is never stored: its entries are a
//! counter-based hash of (seed, pixel, column), independent of blocks and
//! threads. The small dense factorizations (T x T, T x l) run on one thread.

use crate::parallel;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use rayon::prelude::*;
use std::ops::Range;

/// The pixels of a block of a cube read at a time.
pub const BLOCK_PIXELS: usize = 4096;

/// How many blocks are held at once between two reductions: fixed, so the
/// sums do not depend on the threads (it only bounds the memory).
const WAVE: usize = 64;

/// The cube, read in blocks of pixels: weighted anomalies, with the gaps at 0.
pub trait Blocks: Sync {
    /// The dates T of the fit (the rows of every block).
    fn dates(&self) -> usize;
    /// The number of blocks.
    fn block_count(&self) -> usize;
    /// Fills `out` (replacing its contents) with block `b`, `out[j * T + t]`,
    /// and returns its pixels: a range of the global pixel index, which the
    /// random test matrix and the maps use.
    fn block(&self, b: usize, out: &mut Vec<f64>) -> Range<usize>;
    /// The square roots of the weights of the pixels of `range`, the factor
    /// `block` multiplied them by (needed for the weighted mean anomaly that
    /// fixes the signs). Without weights, 1.
    fn sqrt_weights(&self, range: Range<usize>, out: &mut Vec<f64>) {
        out.clear();
        out.resize(range.len(), 1.0);
    }
}

/// How the singular vectors are found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Solver {
    /// `Exact` when T <= 256 (one pass over the data, its Gram cheap next to
    /// the randomized passes) or the sketch (modes + 10) is at least T / 2,
    /// else `Randomized` with 10 and 2.
    Auto,
    /// The eigendecomposition of the T x T Gram matrix.
    Exact,
    /// Halko, Martinsson & Tropp's Alg. 4.4 (`oversampling` p, `power_iterations` q).
    Randomized {
        oversampling: usize,
        power_iterations: usize,
    },
}

/// What `decompose` is asked.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// Modes to compute (at most T - 1).
    pub modes: usize,
    pub solver: Solver,
    /// The seed of the random test matrix.
    pub seed: u64,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            modes: 10,
            solver: Solver::Auto,
            seed: 0x5EED,
        }
    }
}

/// The left side of the decomposition `Xw = U S V^T`.
#[derive(Clone, Debug)]
pub struct Decomposition {
    /// T.
    pub dates: usize,
    /// The singular values s_k, descending.
    pub singular: Vec<f64>,
    /// U, mode by mode (`u[k * T + t]`), unit vectors.
    pub u: Vec<f64>,
    /// `||Xw||_F^2`, accumulated exactly in the first pass.
    pub total: f64,
    /// m_t = sum_j sqrt(w_j) Xw_jt, the area-weighted mean anomaly of each date.
    pub weighted_sum: Vec<f64>,
    /// The next singular value, s_(k+1) (for North's test of the last mode).
    pub next_singular: f64,
    /// The solver that ran (never `Auto`).
    pub solver_used: Solver,
}

impl Decomposition {
    /// The modes found.
    pub fn modes(&self) -> usize {
        self.singular.len()
    }

    /// The eigenvalues lambda_k = s_k^2 / (T - 1).
    pub fn eigenvalues(&self) -> Vec<f64> {
        let d = (self.dates.max(2) - 1) as f64;
        self.singular.iter().map(|s| s * s / d).collect()
    }

    /// The explained fractions s_k^2 / ||Xw||_F^2.
    pub fn explained(&self) -> Vec<f64> {
        self.singular.iter().map(|s| s * s / self.total).collect()
    }

    /// The principal components, mode by mode (`modes x T`): with `units`,
    /// unit variance (`sqrt(T - 1) U_k`); else U_k S_k, which carry the variance.
    pub fn pcs(&self, units: bool) -> Vec<f64> {
        let t = self.dates;
        let f = ((t.max(2) - 1) as f64).sqrt();
        let mut out = self.u.clone();
        for (k, col) in out.chunks_mut(t).enumerate() {
            let c = if units { f } else { self.singular[k] };
            col.iter_mut().for_each(|v| *v *= c);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// The random test matrix

fn splitmix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The entry (pixel, column) of the test matrix: a standard normal value from a
/// hash of the seed and the position (Box-Muller on two uniforms).
fn gauss(seed: u64, pixel: u64, col: u64) -> f64 {
    const U53: f64 = 1.0 / 9_007_199_254_740_992.0;
    let h1 = splitmix64(splitmix64(splitmix64(seed) ^ pixel) ^ col);
    let h2 = splitmix64(h1 ^ 0xD1B5_4A32_D192_ED03);
    let u1 = ((h1 >> 11) as f64 + 0.5) * U53; // in (0, 1)
    let u2 = (h2 >> 11) as f64 * U53;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Passes over the blocks

/// The sum over the blocks of `f(b)` (every result `len` long), added in block
/// order, `WAVE` blocks at a time, the blocks of a wave on the threads of `n_jobs`.
fn sum_blocks<F>(nb: usize, len: usize, n_jobs: i32, report: &dyn Fn(f64), f: F) -> Vec<f64>
where
    F: Fn(usize) -> Vec<f64> + Sync,
{
    let threads = parallel::threads(n_jobs).max(1);
    let mut acc = vec![0.0; len];
    let mut start = 0;
    while start < nb {
        let end = (start + WAVE).min(nb);
        let parts: Vec<Vec<f64>> = if threads == 1 {
            (start..end).map(&f).collect()
        } else {
            parallel::pool(threads).install(|| (start..end).into_par_iter().map(&f).collect())
        };
        for part in parts {
            for (a, v) in acc.iter_mut().zip(part) {
                *a += v;
            }
        }
        start = end;
        report(start as f64 / nb as f64);
    }
    acc
}

fn read(x: &dyn Blocks, b: usize, t: usize, buf: &mut Vec<f64>) -> Range<usize> {
    buf.clear();
    let r = x.block(b, buf);
    assert_eq!(buf.len(), t * r.len(), "block {b} is not T x cols");
    r
}

/// Block `b`'s Gram matrix `X_b X_b^T` (T x T), its squared norm and its
/// weighted sum over the pixels: `[gram, norm, m]`.
fn pass_gram(x: &dyn Blocks, t: usize, b: usize) -> Vec<f64> {
    let mut buf = Vec::new();
    let range = read(x, b, t, &mut buf);
    let cols = range.len();
    let mut out = vec![0.0; t * t + 1 + t];
    let xb = MatRef::from_column_major_slice(&buf, t, cols);
    if cols > 0 {
        let g = MatMut::from_column_major_slice_mut(&mut out[..t * t], t, t);
        matmul(g, Accum::Replace, xb, xb.transpose(), 1.0, Par::Seq);
    }
    out[t * t] = norm_and_sum(x, t, range, &buf, &mut out[t * t + 1..]);
    out
}

/// The squared norm of the block, and its weighted sum over the pixels in `m`.
fn norm_and_sum(x: &dyn Blocks, t: usize, range: Range<usize>, buf: &[f64], m: &mut [f64]) -> f64 {
    let mut sw = Vec::new();
    x.sqrt_weights(range, &mut sw);
    let mut norm = 0.0;
    for (col, &w) in buf.chunks(t).zip(&sw) {
        for (mt, &v) in m.iter_mut().zip(col) {
            norm += v * v;
            *mt += w * v;
        }
    }
    norm
}

/// Block `b`'s sketch `X_b Omega_b` (T x l), its squared norm and weighted sum:
/// `[y, norm, m]`.
fn pass_sketch(x: &dyn Blocks, t: usize, l: usize, seed: u64, b: usize) -> Vec<f64> {
    let mut buf = Vec::new();
    let range = read(x, b, t, &mut buf);
    let cols = range.len();
    let mut out = vec![0.0; t * l + 1 + t];
    if cols > 0 {
        let mut om = vec![0.0; cols * l];
        for c in 0..l {
            for (j, o) in om[c * cols..(c + 1) * cols].iter_mut().enumerate() {
                *o = gauss(seed, (range.start + j) as u64, c as u64);
            }
        }
        let xb = MatRef::from_column_major_slice(&buf, t, cols);
        let y = MatMut::from_column_major_slice_mut(&mut out[..t * l], t, l);
        matmul(
            y,
            Accum::Replace,
            xb,
            MatRef::from_column_major_slice(&om, cols, l),
            1.0,
            Par::Seq,
        );
    }
    out[t * l] = norm_and_sum(x, t, range, &buf, &mut out[t * l + 1..]);
    out
}

/// One pass of the power iteration with `q` (T x l): `Z = X_b^T Q`, then
/// `[X_b Z (when want_y), Z^T Z]`.
fn pass_power(x: &dyn Blocks, t: usize, l: usize, q: &[f64], want_y: bool, b: usize) -> Vec<f64> {
    let mut buf = Vec::new();
    let range = read(x, b, t, &mut buf);
    let cols = range.len();
    let ylen = if want_y { t * l } else { 0 };
    let mut out = vec![0.0; ylen + l * l];
    if cols == 0 {
        return out;
    }
    let xb = MatRef::from_column_major_slice(&buf, t, cols);
    let mut z = vec![0.0; cols * l];
    let zm = MatMut::from_column_major_slice_mut(&mut z, cols, l);
    matmul(
        zm,
        Accum::Replace,
        xb.transpose(),
        MatRef::from_column_major_slice(q, t, l),
        1.0,
        Par::Seq,
    );
    let zr = MatRef::from_column_major_slice(&z, cols, l);
    if want_y {
        let y = MatMut::from_column_major_slice_mut(&mut out[..t * l], t, l);
        matmul(y, Accum::Replace, xb, zr, 1.0, Par::Seq);
    }
    let g = MatMut::from_column_major_slice_mut(&mut out[ylen..], l, l);
    matmul(g, Accum::Replace, zr.transpose(), zr, 1.0, Par::Seq);
    out
}

// ---------------------------------------------------------------------------
// Small dense algebra (one thread, so the bits do not depend on the pool)

/// The eigenvalues (descending) and eigenvectors (columns, column-major) of the
/// symmetric `n x n` matrix `g`, from its lower triangle.
fn eig_desc(g: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    parallel::pool(1).install(|| {
        let e = MatRef::from_column_major_slice(g, n, n)
            .self_adjoint_eigen(Side::Lower)
            .map_err(|e| format!("the eigendecomposition did not converge: {e:?}"))?;
        let s = e.S().column_vector();
        let u = e.U();
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        for c in 0..n {
            let src = n - 1 - c; // faer's are ascending
            vals[c] = s[src];
            for r in 0..n {
                vecs[c * n + r] = u[(r, src)];
            }
        }
        Ok((vals, vecs))
    })
}

/// The orthonormal basis (thin QR) of the columns of `y` (T x l).
fn orthonormal(y: &[f64], t: usize, l: usize) -> Vec<f64> {
    parallel::pool(1).install(|| {
        let q = Mat::from_fn(t, l, |i, j| y[j * t + i]).qr().compute_thin_Q();
        let mut out = vec![0.0; t * l];
        for j in 0..l {
            for i in 0..t {
                out[j * t + i] = q[(i, j)];
            }
        }
        out
    })
}

// ---------------------------------------------------------------------------
// The decomposition

/// The singular values and left singular vectors of the cube `x`, its squared
/// norm and weighted sum, in a few passes over its blocks. The signs are not
/// fixed: see [`fix_signs`]. `progress` gets the fraction done (0 to 1).
pub fn decompose(
    x: &dyn Blocks,
    s: &Settings,
    n_jobs: i32,
    progress: &(dyn Fn(f64) + Sync),
) -> Result<Decomposition, String> {
    let t = x.dates();
    let nb = x.block_count();
    if t < 3 {
        return Err("an EOF needs at least 3 dates".into());
    }
    if nb == 0 {
        return Err("an EOF needs at least one pixel".into());
    }
    if s.modes == 0 {
        return Err("modes must be at least 1".into());
    }
    let k = s.modes.min(t - 1);
    let solver = match s.solver {
        Solver::Auto => {
            if t <= 256 || (k + 10) * 2 >= t {
                Solver::Exact
            } else {
                Solver::Randomized {
                    oversampling: 10,
                    power_iterations: 2,
                }
            }
        }
        other => other,
    };
    match solver {
        Solver::Exact | Solver::Auto => {
            let acc = sum_blocks(nb, t * t + 1 + t, n_jobs, progress, |b| pass_gram(x, t, b));
            let (vals, vecs) = eig_desc(&acc[..t * t], t)?;
            let singular: Vec<f64> = vals[..k].iter().map(|v| v.max(0.0).sqrt()).collect();
            Ok(Decomposition {
                dates: t,
                singular,
                u: vecs[..k * t].to_vec(),
                total: acc[t * t],
                weighted_sum: acc[t * t + 1..].to_vec(),
                next_singular: vals[k].max(0.0).sqrt(),
                solver_used: Solver::Exact,
            })
        }
        Solver::Randomized {
            oversampling,
            power_iterations,
        } => {
            let l = (k + oversampling).clamp(k + 1, t);
            let passes = (power_iterations + 2) as f64;
            let part = |i: usize| move |f: f64| progress((i as f64 + f) / passes);
            let acc = sum_blocks(nb, t * l + 1 + t, n_jobs, &part(0), |b| pass_sketch(x, t, l, s.seed, b));
            let total = acc[t * l];
            let weighted_sum = acc[t * l + 1..].to_vec();
            let mut q = orthonormal(&acc[..t * l], t, l);
            for i in 0..power_iterations {
                let y = sum_blocks(nb, t * l + l * l, n_jobs, &part(1 + i), |b| {
                    pass_power(x, t, l, &q, true, b)
                });
                q = orthonormal(&y[..t * l], t, l);
            }
            // the last pass: G = Q^T X X^T Q, and the Ritz values and vectors
            let g = sum_blocks(nb, l * l, n_jobs, &part(1 + power_iterations), |b| {
                pass_power(x, t, l, &q, false, b)
            });
            let (vals, vecs) = eig_desc(&g, l)?;
            let mut u = vec![0.0; k * t];
            matmul(
                MatMut::from_column_major_slice_mut(&mut u, t, k),
                Accum::Replace,
                MatRef::from_column_major_slice(&q, t, l),
                MatRef::from_column_major_slice(&vecs, l, l).get(.., ..k),
                1.0,
                Par::Seq,
            );
            progress(1.0);
            Ok(Decomposition {
                dates: t,
                singular: vals[..k].iter().map(|v| v.max(0.0).sqrt()).collect(),
                u,
                total,
                weighted_sum,
                next_singular: vals[k].max(0.0).sqrt(),
                solver_used: Solver::Randomized {
                    oversampling: l - k,
                    power_iterations,
                },
            })
        }
    }
}

/// Fixes the sign of each mode so that its PC correlates positively with the
/// area-weighted mean anomaly: `U_k . m > 0`. When that is (nearly) zero, a
/// dipole (`|U_k . m| < 1e-6 ||m||`, or a mean anomaly that is only rounding
/// noise), the largest value of the PC is made positive instead.
pub fn fix_signs(d: &mut Decomposition) {
    let t = d.dates;
    let norm_m = d.weighted_sum.iter().map(|v| v * v).sum::<f64>().sqrt();
    let noise = norm_m <= 1e-10 * d.total.max(0.0).sqrt();
    for col in d.u.chunks_mut(t) {
        let dot: f64 = col.iter().zip(&d.weighted_sum).map(|(a, b)| a * b).sum();
        let flip = if noise || dot.abs() < 1e-6 * norm_m {
            let big = col.iter().fold(0.0f64, |m, &v| if v.abs() > m.abs() { v } else { m });
            big < 0.0
        } else {
            dot < 0.0
        };
        if flip {
            col.iter_mut().for_each(|v| *v = -*v);
        }
    }
}

/// The right singular vectors of a block, `V_b = X_b^T U S^-1` (orthonormal
/// over all the pixels): `out[k * cols + j]`, `cols x modes`, mode by mode. A
/// mode with a null singular value gives zeros.
pub fn loadings(d: &Decomposition, block: &[f64], cols: usize, out: &mut [f64]) {
    let t = d.dates;
    let k = d.modes();
    assert_eq!(block.len(), t * cols);
    assert_eq!(out.len(), cols * k);
    let mut us = d.u.clone();
    for (col, &s) in us.chunks_mut(t).zip(&d.singular) {
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        col.iter_mut().for_each(|v| *v *= inv);
    }
    if cols == 0 || k == 0 {
        return;
    }
    matmul(
        MatMut::from_column_major_slice_mut(out, cols, k),
        Accum::Replace,
        MatRef::from_column_major_slice(block, t, cols).transpose(),
        MatRef::from_column_major_slice(&us, t, k),
        1.0,
        Par::Seq,
    );
}

/// The orthonormal maps of `loadings` in data units per standard deviation of
/// the PC: `L_jk = s_k V_jk / (sqrt(w_j) sqrt(T - 1))`, in place; `sqrt_w` are
/// the square roots of the weights of the block's pixels.
pub fn scale_units(d: &Decomposition, out: &mut [f64], cols: usize, sqrt_w: &[f64]) {
    let f = ((d.dates.max(2) - 1) as f64).sqrt();
    for (col, &s) in out.chunks_mut(cols).zip(&d.singular) {
        for (v, &w) in col.iter_mut().zip(sqrt_w) {
            *v *= s / (w * f);
        }
    }
}

/// The lag-1 autocorrelation of `v` (consecutive pairs both finite, the mean of
/// the finite values); NaN with fewer than 3 finite values or a constant series.
pub fn lag1(v: &[f64]) -> f64 {
    let fin: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if fin.len() < 3 {
        return f64::NAN;
    }
    let mean = fin.iter().sum::<f64>() / fin.len() as f64;
    let den: f64 = fin.iter().map(|x| (x - mean) * (x - mean)).sum();
    if den <= 0.0 {
        return f64::NAN;
    }
    let num: f64 = v
        .windows(2)
        .filter(|w| w[0].is_finite() && w[1].is_finite())
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum();
    num / den
}

/// The effective number of dates of a series with lag-1 autocorrelation `rho`
/// (clamped to 0 to 0.99): `T (1 - rho) / (1 + rho)`.
pub fn effective_dates(t: usize, rho: f64) -> f64 {
    let r = if rho.is_finite() { rho.clamp(0.0, 0.99) } else { 0.0 };
    t as f64 * (1.0 - r) / (1.0 + r)
}

/// North et al.'s (1982) rule of thumb: the sampling error of each eigenvalue,
/// `delta_k = lambda_k sqrt(2 / n*_k)`, and whether mode k is separable from
/// its neighbours: `lambda_(k-1) - delta_(k-1) > lambda_k + delta_k` and
/// `lambda_k - delta_k > lambda_(k+1) + delta_(k+1)`. The first mode has no
/// upper neighbour; the next of the last mode is `next_eigen` (with the last
/// mode's `n*`).
pub fn north(eigen: &[f64], n_star: &[f64], next_eigen: f64) -> Vec<(f64, bool)> {
    assert_eq!(eigen.len(), n_star.len());
    let delta: Vec<f64> = eigen.iter().zip(n_star).map(|(l, n)| l * (2.0 / n).sqrt()).collect();
    let k = eigen.len();
    (0..k)
        .map(|i| {
            let upper = i == 0 || eigen[i - 1] - delta[i - 1] > eigen[i] + delta[i];
            let (ln, dn) = if i + 1 < k {
                (eigen[i + 1], delta[i + 1])
            } else {
                (next_eigen, next_eigen * (2.0 / n_star[i]).sqrt())
            };
            let lower = eigen[i] - delta[i] > ln + dn;
            (delta[i], upper && lower)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Varimax

/// `a R` for `a` (`rows x m`, column-major) and `r` (`m x m`, column-major).
pub fn rotate(a: &[f64], rows: usize, m: usize, r: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows * m];
    if rows > 0 && m > 0 {
        matmul(
            MatMut::from_column_major_slice_mut(&mut out, rows, m),
            Accum::Replace,
            MatRef::from_column_major_slice(a, rows, m),
            MatRef::from_column_major_slice(r, m, m),
            1.0,
            Par::Seq,
        );
    }
    out
}

/// The rotation of Kaiser's (1958) varimax by its SVD iteration, on `a`
/// (`ns x m`, column-major): `R` and the iterations used.
fn varimax_fit(a: &[f64], ns: usize, m: usize, tol: f64, max_iter: usize) -> (Vec<f64>, usize) {
    let mut r = vec![0.0; m * m];
    (0..m).for_each(|i| r[i * m + i] = 1.0);
    if m < 2 || ns == 0 {
        return (r, 0);
    }
    let am = MatRef::from_column_major_slice(a, ns, m);
    let mut d_old = 0.0;
    let mut iters = 0;
    for it in 1..=max_iter {
        iters = it;
        let lam = rotate(a, ns, m, &r);
        let mut w = vec![0.0; ns * m];
        for k in 0..m {
            let col = &lam[k * ns..(k + 1) * ns];
            let c = col.iter().map(|v| v * v).sum::<f64>() / ns as f64;
            for (o, &v) in w[k * ns..(k + 1) * ns].iter_mut().zip(col) {
                *o = v * v * v - v * c;
            }
        }
        let mut bm = Mat::<f64>::zeros(m, m);
        matmul(
            bm.as_mut(),
            Accum::Replace,
            am.transpose(),
            MatRef::from_column_major_slice(&w, ns, m),
            1.0,
            Par::Seq,
        );
        let d = parallel::pool(1).install(|| {
            let svd = bm.thin_svd().expect("the SVD of the varimax matrix did not converge");
            let mut rm = Mat::<f64>::zeros(m, m);
            matmul(rm.as_mut(), Accum::Replace, svd.U(), svd.V().transpose(), 1.0, Par::Seq);
            for c in 0..m {
                for i in 0..m {
                    r[c * m + i] = rm[(i, c)];
                }
            }
            svd.S().column_vector().iter().sum::<f64>()
        });
        if d < d_old * (1.0 + tol) {
            break;
        }
        d_old = d;
    }
    (r, iters)
}

/// Rotates the loadings `l` (`n x m`, column-major: `l[k * n + j]`, the
/// weighted `V S / sqrt(T - 1)`) in place by the varimax criterion, with
/// Kaiser's normalization (rows divided by their communality, multiplied back)
/// when `kaiser`; stops when the sum of the singular values grows by less than
/// `tol` (relative) or after `max_iter` iterations. Returns the rotation `R`
/// (`m x m`, column-major; the rotated PCs are `z R`) and the iterations. For
/// more than 2e6 rows, `R` is fitted on a seeded sample of about 1e6 rows and
/// applied to all.
pub fn varimax(l: &mut [f64], n: usize, m: usize, kaiser: bool, tol: f64, max_iter: usize) -> (Vec<f64>, usize) {
    assert_eq!(l.len(), n * m);
    let h: Vec<f64> = (0..n)
        .map(|j| {
            if kaiser {
                let c = (0..m).map(|k| l[k * n + j] * l[k * n + j]).sum::<f64>().sqrt();
                if c > 0.0 {
                    c
                } else {
                    1.0
                }
            } else {
                1.0
            }
        })
        .collect();
    let mut norm = vec![0.0; n * m];
    for k in 0..m {
        for j in 0..n {
            norm[k * n + j] = l[k * n + j] / h[j];
        }
    }
    const SAMPLE: usize = 1_000_000;
    let (r, iters) = if n > 2_000_000 {
        let rows: Vec<usize> = (0..n)
            .filter(|&j| splitmix64(j as u64 ^ 0xA5A5_5A5A) % (n as u64) < SAMPLE as u64)
            .collect();
        let ns = rows.len();
        let mut sample = vec![0.0; ns * m];
        for k in 0..m {
            for (i, &j) in rows.iter().enumerate() {
                sample[k * ns + i] = norm[k * n + j];
            }
        }
        varimax_fit(&sample, ns, m, tol, max_iter)
    } else {
        varimax_fit(&norm, n, m, tol, max_iter)
    };
    let rot = rotate(&norm, n, m, &r);
    for k in 0..m {
        for j in 0..n {
            l[k * n + j] = rot[k * n + j] * h[j];
        }
    }
    (r, iters)
}

// ---------------------------------------------------------------------------
// Gappy projection

/// The coefficients of the pixels of one date `x_t` (NaN where missing) on the
/// basis `eofs_w` (`N x m`, column-major, the maps in the same weighting as
/// `x_t`), by least squares over the valid pixels (Everson & Sirovich 1995):
/// for orthonormal maps `V` these are `U_k S_k` of that date. NaN when fewer
/// valid pixels than modes. The normal equations are solved through their
/// eigendecomposition (directions with a null eigenvalue are dropped).
pub fn project_gappy(eofs_w: &[f64], m: usize, x_t: &[f64]) -> Vec<f64> {
    let n = x_t.len();
    assert_eq!(eofs_w.len(), n * m);
    let valid: Vec<usize> = (0..n)
        .filter(|&j| x_t[j].is_finite() && (0..m).all(|k| eofs_w[k * n + j].is_finite()))
        .collect();
    if valid.len() < m || m == 0 {
        return vec![f64::NAN; m];
    }
    let mut g = vec![0.0; m * m];
    let mut b = vec![0.0; m];
    for a in 0..m {
        let ca = &eofs_w[a * n..(a + 1) * n];
        b[a] = valid.iter().map(|&j| ca[j] * x_t[j]).sum();
        for c in 0..=a {
            let cc = &eofs_w[c * n..(c + 1) * n];
            let v: f64 = valid.iter().map(|&j| ca[j] * cc[j]).sum();
            g[c * m + a] = v;
            g[a * m + c] = v;
        }
    }
    let Ok((vals, vecs)) = eig_desc(&g, m) else {
        return vec![f64::NAN; m];
    };
    let cut = vals[0].max(0.0) * 1e-12;
    let mut out = vec![0.0; m];
    for (k, &lam) in vals.iter().enumerate() {
        if lam > cut {
            let e = &vecs[k * m..(k + 1) * m];
            let proj: f64 = e.iter().zip(&b).map(|(a, b)| a * b).sum::<f64>() / lam;
            for (o, &ev) in out.iter_mut().zip(e) {
                *o += proj * ev;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dense: the blocks of a cube in memory

/// A (T, N) time-major cube in memory as [`Blocks`]: the pixels valid at least
/// `min_valid` of the dates are kept (the rest have no maps); then the dates
/// valid at least `min_date_valid` of the kept pixels; each kept column is
/// re-centred over the kept dates, scaled by the square root of its weight, and
/// its remaining NaN set to 0. The pixel index of the blocks is that of the
/// kept pixels (`kept_pixels` gives the cube's).
pub struct Dense<'a> {
    x: &'a [f64],
    n_all: usize,
    pixels: Vec<usize>,
    dates: Vec<usize>,
    mean: Vec<f64>,
    sqrt_w: Vec<f64>,
    block: usize,
}

impl<'a> Dense<'a> {
    /// `x` is the already anomalized cube (`x[t * n + j]`), `weights` (mean 1)
    /// per pixel or none. `Err` for a wrong shape or a negative weight, or if
    /// no pixel or fewer than 3 dates are kept.
    pub fn new(
        x: &'a [f64],
        t: usize,
        n: usize,
        weights: Option<&[f64]>,
        min_valid: f64,
        min_date_valid: f64,
    ) -> Result<Dense<'a>, String> {
        if x.len() != t * n {
            return Err(format!("the cube has {} values, not {t} x {n}", x.len()));
        }
        if let Some(w) = weights {
            if w.len() != n {
                return Err("one weight per pixel is needed".into());
            }
            if w.iter().any(|v| !(*v >= 0.0 && v.is_finite())) {
                return Err("the weights must be finite and not negative".into());
            }
        }
        let mut count = vec![0usize; n];
        for row in x.chunks(n.max(1)) {
            for (c, v) in count.iter_mut().zip(row) {
                *c += v.is_finite() as usize;
            }
        }
        let pixels: Vec<usize> = (0..n)
            .filter(|&j| count[j] > 0 && count[j] as f64 >= min_valid * t as f64)
            .collect();
        if pixels.is_empty() {
            return Err("no pixel has enough valid dates".into());
        }
        let dates: Vec<usize> = (0..t)
            .filter(|&i| {
                let row = &x[i * n..(i + 1) * n];
                let v = pixels.iter().filter(|&&j| row[j].is_finite()).count();
                v > 0 && v as f64 >= min_date_valid * pixels.len() as f64
            })
            .collect();
        if dates.len() < 3 {
            return Err("fewer than 3 dates are kept".into());
        }
        let mut sum = vec![0.0; pixels.len()];
        let mut cnt = vec![0usize; pixels.len()];
        for &i in &dates {
            let row = &x[i * n..(i + 1) * n];
            for (p, &j) in pixels.iter().enumerate() {
                if row[j].is_finite() {
                    sum[p] += row[j];
                    cnt[p] += 1;
                }
            }
        }
        let mean = sum
            .iter()
            .zip(&cnt)
            .map(|(s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();
        let sqrt_w = pixels.iter().map(|&j| weights.map_or(1.0, |w| w[j].sqrt())).collect();
        Ok(Dense {
            x,
            n_all: n,
            pixels,
            dates,
            mean,
            sqrt_w,
            block: BLOCK_PIXELS,
        })
    }

    /// Another block size (pixels), for tests and memory.
    pub fn with_block(mut self, pixels: usize) -> Dense<'a> {
        self.block = pixels.max(1);
        self
    }

    /// The cube's pixel index of each kept pixel.
    pub fn kept_pixels(&self) -> &[usize] {
        &self.pixels
    }

    /// The cube's date index of each kept date.
    pub fn kept_dates(&self) -> &[usize] {
        &self.dates
    }

    /// The square roots of the weights of the kept pixels.
    pub fn sqrt_weights_all(&self) -> &[f64] {
        &self.sqrt_w
    }

    /// The weighted, centred values of date `t` (the cube's index, a dropped date
    /// too) over the kept pixels, NaN where missing: the input of [`project_gappy`].
    pub fn weighted_row(&self, t: usize, out: &mut Vec<f64>) {
        let row = &self.x[t * self.n_all..(t + 1) * self.n_all];
        out.clear();
        out.extend(
            self.pixels
                .iter()
                .enumerate()
                .map(|(p, &j)| (row[j] - self.mean[p]) * self.sqrt_w[p]),
        );
    }
}

impl Blocks for Dense<'_> {
    fn dates(&self) -> usize {
        self.dates.len()
    }

    fn block_count(&self) -> usize {
        self.pixels.len().div_ceil(self.block)
    }

    fn block(&self, b: usize, out: &mut Vec<f64>) -> Range<usize> {
        let start = b * self.block;
        let end = (start + self.block).min(self.pixels.len());
        let tk = self.dates.len();
        out.clear();
        out.resize(tk * (end - start), 0.0);
        for (i, &t) in self.dates.iter().enumerate() {
            let row = &self.x[t * self.n_all..(t + 1) * self.n_all];
            for p in start..end {
                let v = row[self.pixels[p]];
                if v.is_finite() {
                    out[(p - start) * tk + i] = (v - self.mean[p]) * self.sqrt_w[p];
                }
            }
        }
        start..end
    }

    fn sqrt_weights(&self, range: Range<usize>, out: &mut Vec<f64>) {
        out.clear();
        out.extend_from_slice(&self.sqrt_w[range]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64, i: u64) -> f64 {
        gauss(seed, i, 7)
    }

    /// Orthonormalizes the columns (`rows x m`) by Gram-Schmidt, after removing
    /// their means.
    fn gram_schmidt(cols: &mut [Vec<f64>], center: bool) {
        for i in 0..cols.len() {
            if center {
                let m = cols[i].iter().sum::<f64>() / cols[i].len() as f64;
                cols[i].iter_mut().for_each(|v| *v -= m);
            }
            for j in 0..i {
                let d: f64 = cols[i].iter().zip(&cols[j]).map(|(a, b)| a * b).sum();
                let cj = cols[j].clone();
                cols[i].iter_mut().zip(&cj).for_each(|(a, b)| *a -= d * b);
            }
            let nrm = cols[i].iter().map(|v| v * v).sum::<f64>().sqrt();
            cols[i].iter_mut().for_each(|v| *v /= nrm);
        }
    }

    const T: usize = 120;
    const NX: usize = 25;
    const NY: usize = 40;
    const N: usize = NX * NY;

    struct Truth {
        x: Vec<f64>,
        u: Vec<Vec<f64>>,
        v: Vec<Vec<f64>>,
        s: [f64; 3],
    }

    /// X = sum s_k u_k v_k^T + 0.01 noise: sinusoids of 29, 47 and 61 months, a
    /// dipole, a monopole and a checker.
    fn make(s: [f64; 3]) -> Truth {
        let mut u: Vec<Vec<f64>> = [29.0, 47.0, 61.0]
            .iter()
            .map(|p| {
                (0..T)
                    .map(|t| (2.0 * std::f64::consts::PI * t as f64 / p + 0.3).sin())
                    .collect()
            })
            .collect();
        gram_schmidt(&mut u, true);
        let at = |i: usize, j: usize| i * NY + j;
        let mut v = vec![vec![0.0; N]; 3];
        for i in 0..NX {
            for j in 0..NY {
                v[0][at(i, j)] = if i < NX / 2 { 1.0 } else { -1.0 };
                v[1][at(i, j)] = (-(((i as f64 - 12.0) / 8.0).powi(2) + ((j as f64 - 20.0) / 14.0).powi(2))).exp();
                v[2][at(i, j)] = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        // the dipole sums to 0, the others are only roughly orthogonal to it
        gram_schmidt(&mut v, false);
        // unit entries (rms), so that the singular values are s sqrt(N) and the noise is small against them
        v.iter_mut()
            .for_each(|c| c.iter_mut().for_each(|a| *a *= (N as f64).sqrt()));
        let mut x = vec![0.0; T * N];
        for t in 0..T {
            for j in 0..N {
                let mut a = 0.01 * rng(11, (t * N + j) as u64);
                for k in 0..3 {
                    a += s[k] * u[k][t] * v[k][j];
                }
                x[t * N + j] = a;
            }
        }
        Truth { x, u, v, s }
    }

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let d: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        d / (na * nb)
    }

    fn run(d: &Dense, modes: usize, solver: Solver, n_jobs: i32) -> Decomposition {
        let s = Settings { modes, solver, seed: 1 };
        decompose(d, &s, n_jobs, &|_| {}).unwrap()
    }

    /// V of all the kept pixels (`n x modes`, column-major).
    fn all_loadings(d: &Dense, dec: &Decomposition) -> Vec<f64> {
        let k = dec.modes();
        let n = d.kept_pixels().len();
        let mut v = vec![0.0; n * k];
        let mut buf = Vec::new();
        for b in 0..d.block_count() {
            let r = d.block(b, &mut buf);
            let mut out = vec![0.0; r.len() * k];
            loadings(dec, &buf, r.len(), &mut out);
            for m in 0..k {
                v[m * n + r.start..m * n + r.end].copy_from_slice(&out[m * r.len()..(m + 1) * r.len()]);
            }
        }
        v
    }

    #[test]
    fn recovers_the_modes_and_the_explained_fractions() {
        let tr = make([10.0, 5.0, 2.0]);
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(150);
        let mut dec = run(&d, 3, Solver::Auto, 2);
        assert_eq!(dec.solver_used, Solver::Exact);
        fix_signs(&mut dec);
        let v = all_loadings(&d, &dec);
        for k in 0..3 {
            let cu = cosine(&dec.u[k * T..(k + 1) * T], &tr.u[k]).abs();
            let cv = cosine(&v[k * N..(k + 1) * N], &tr.v[k]).abs();
            assert!(cu > 0.999 && cv > 0.999, "mode {k}: {cu} {cv}");
        }
        let total_ref: f64 = (0..T * N)
            .map(|i| {
                let (t, j) = (i / N, i % N);
                let m: f64 = (0..T).map(|q| tr.x[q * N + j]).sum::<f64>() / T as f64;
                (tr.x[t * N + j] - m).powi(2)
            })
            .sum();
        assert!((dec.total - total_ref).abs() < 1e-8 * total_ref);
        let ex = dec.explained();
        for (k, f) in ex.iter().enumerate().take(3) {
            let want = tr.s[k] * tr.s[k] * N as f64 / total_ref;
            assert!((f - want).abs() < 1e-3, "mode {k}: {f} vs {want}");
        }
    }

    #[test]
    fn randomized_matches_the_exact_solver() {
        let tr = make([10.0, 5.0, 2.0]);
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(150);
        let ex = run(&d, 3, Solver::Exact, 2);
        let rnd = run(
            &d,
            3,
            Solver::Randomized {
                oversampling: 10,
                power_iterations: 2,
            },
            2,
        );
        for k in 0..3 {
            let rel = (ex.singular[k] - rnd.singular[k]).abs() / ex.singular[k];
            assert!(rel < 1e-8, "mode {k}: {rel}");
            let c = cosine(&ex.u[k * T..(k + 1) * T], &rnd.u[k * T..(k + 1) * T]).abs();
            assert!(c > 1.0 - 1e-10, "mode {k}: {c}");
        }
        assert!((ex.total - rnd.total).abs() < 1e-9 * ex.total);
        for (a, b) in ex.weighted_sum.iter().zip(&rnd.weighted_sum) {
            assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()));
        }
    }

    fn bits(d: &Decomposition) -> Vec<u64> {
        d.singular
            .iter()
            .chain(&d.u)
            .chain(&d.weighted_sum)
            .chain([&d.total, &d.next_singular])
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn bitwise_equal_for_any_thread_count() {
        let tr = make([10.0, 5.0, 2.0]);
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(40);
        for solver in [
            Solver::Exact,
            Solver::Randomized {
                oversampling: 10,
                power_iterations: 2,
            },
        ] {
            let one = bits(&run(&d, 5, solver, 1));
            for jobs in [2, 4] {
                assert_eq!(one, bits(&run(&d, 5, solver, jobs)), "{solver:?} with {jobs} jobs");
            }
        }
    }

    #[test]
    fn another_block_size_changes_the_results_a_little() {
        let tr = make([10.0, 5.0, 2.0]);
        let a = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(1000);
        let b = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(97);
        let da = run(&a, 5, Solver::Exact, 1);
        let db = run(&b, 5, Solver::Exact, 1);
        for k in 0..5 {
            assert!((da.singular[k] - db.singular[k]).abs() <= 1e-12 * da.singular[0]);
            let c = cosine(&da.u[k * T..(k + 1) * T], &db.u[k * T..(k + 1) * T]).abs();
            assert!(c > 1.0 - 1e-12, "mode {k}: {c}");
        }
    }

    #[test]
    fn weighted_decomposition_is_the_svd_of_the_scaled_matrix() {
        let tr = make([10.0, 5.0, 2.0]);
        let w: Vec<f64> = (0..N).map(|j| 0.5 + (j % 7) as f64 / 7.0).collect();
        let mean_w = w.iter().sum::<f64>() / N as f64;
        let w: Vec<f64> = w.iter().map(|v| v / mean_w).collect();
        let d = Dense::new(&tr.x, T, N, Some(&w), 0.8, 0.5).unwrap().with_block(200);
        let dec = run(&d, 4, Solver::Exact, 2);
        let a = Mat::from_fn(T, N, |t, j| {
            let m: f64 = (0..T).map(|q| tr.x[q * N + j]).sum::<f64>() / T as f64;
            (tr.x[t * N + j] - m) * w[j].sqrt()
        });
        let svd = a.thin_svd().unwrap();
        let s = svd.S().column_vector();
        for k in 0..4 {
            assert!((dec.singular[k] - s[k]).abs() < 1e-10 * s[0], "mode {k}");
            let c: f64 = (0..T).map(|t| dec.u[k * T + t] * svd.U()[(t, k)]).sum();
            assert!(c.abs() > 1.0 - 1e-10, "mode {k}: {c}");
        }
        // the weighted mean anomaly: sum_j w_j (x_jt - mean_j)
        for t in 0..T {
            let want: f64 = (0..N).map(|j| w[j].sqrt() * a[(t, j)]).sum();
            assert!((dec.weighted_sum[t] - want).abs() < 1e-8 * (1.0 + want.abs()));
        }
        // V from loadings is orthonormal and reconstructs the matrix
        let v = all_loadings(&d, &dec);
        for k in 0..4 {
            let nrm: f64 = v[k * N..(k + 1) * N].iter().map(|x| x * x).sum();
            assert!((nrm - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn sign_rule() {
        let tr = make([10.0, 5.0, 2.0]);
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(300);
        let mut dec = run(&d, 3, Solver::Exact, 1);
        // flip them any way: the rule gives the same result
        let mut flipped = dec.clone();
        for v in flipped.u[T..2 * T].iter_mut() {
            *v = -*v;
        }
        fix_signs(&mut dec);
        fix_signs(&mut flipped);
        assert_eq!(dec.u, flipped.u);
        // the monopole (mode 2 of the weights' sort, whichever it is) correlates
        // positively with the mean anomaly: U_k . m > 0 except for the dipole
        let norm_m = dec.weighted_sum.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut positive = 0;
        for k in 0..3 {
            let dot: f64 = (0..T).map(|t| dec.u[k * T + t] * dec.weighted_sum[t]).sum();
            if dot.abs() >= 1e-6 * norm_m {
                assert!(dot > 0.0, "mode {k}");
                positive += 1;
            }
        }
        assert!(positive >= 1);
        // a pure dipole (no mean anomaly): the largest value of the PC is positive
        let mut x = vec![0.0; T * N];
        for t in 0..T {
            for j in 0..N {
                x[t * N + j] = tr.u[1][t] * tr.v[0][j] * 3.0;
            }
        }
        let d = Dense::new(&x, T, N, None, 0.8, 0.5).unwrap();
        let mut dec = run(&d, 1, Solver::Exact, 1);
        for sgn in [1.0, -1.0] {
            let mut e = dec.clone();
            e.u.iter_mut().for_each(|v| *v *= sgn);
            fix_signs(&mut e);
            let big = e.u.iter().fold(0.0f64, |m, &v| if v.abs() > m.abs() { v } else { m });
            assert!(big > 0.0);
        }
        fix_signs(&mut dec);
    }

    #[test]
    fn north_separability() {
        let eig = |s: &[f64]| s.iter().map(|v| v * v / (T as f64 - 1.0)).collect::<Vec<_>>();
        let n_star = vec![T as f64; 3];
        // s1 = s2: neither is separable from the other, the third is
        let mixed = eig(&[10.0, 10.0, 2.0]);
        let r = north(&mixed, &n_star, 0.4 * 0.4 / (T as f64 - 1.0));
        assert!(!r[0].1 && !r[1].1 && r[2].1, "{r:?}");
        assert!((r[0].0 - mixed[0] * (2.0 / T as f64).sqrt()).abs() < 1e-15);
        // (10, 5, 2): all separable
        let sep = eig(&[10.0, 5.0, 2.0]);
        let r = north(&sep, &n_star, 0.4 * 0.4 / (T as f64 - 1.0));
        assert!(r.iter().all(|x| x.1), "{r:?}");
        // the same from a decomposition (the next singular value of the noise)
        let tr = make([10.0, 5.0, 2.0]);
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap();
        let dec = run(&d, 3, Solver::Exact, 1);
        let r = north(
            &dec.eigenvalues(),
            &n_star,
            dec.next_singular.powi(2) / (T as f64 - 1.0),
        );
        assert!(r.iter().all(|x| x.1), "{r:?}");
        let tr = make([10.0, 10.0, 2.0]);
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap();
        let dec = run(&d, 3, Solver::Exact, 1);
        let r = north(
            &dec.eigenvalues(),
            &n_star,
            dec.next_singular.powi(2) / (T as f64 - 1.0),
        );
        assert!(!r[0].1 && !r[1].1 && r[2].1, "{r:?}");
    }

    #[test]
    fn autocorrelation_and_effective_dates() {
        // AR(1) with phi = 0.7
        let mut v = vec![0.0; 4000];
        for i in 1..v.len() {
            v[i] = 0.7 * v[i - 1] + rng(3, i as u64);
        }
        assert!((lag1(&v) - 0.7).abs() < 0.05);
        assert!((effective_dates(100, 0.5) - 100.0 / 3.0).abs() < 1e-12);
        assert_eq!(effective_dates(100, -0.3), 100.0);
        assert!(lag1(&[1.0, 1.0, 1.0, 1.0]).is_nan());
        assert!((effective_dates(100, 2.0) - 100.0 * 0.01 / 1.99).abs() < 1e-12);
    }

    #[test]
    fn varimax_recovers_two_patches() {
        let n = 200;
        let p0: Vec<f64> = (0..n)
            .map(|j| {
                if j < 100 {
                    1.0 + 0.3 * (j as f64).sin()
                } else {
                    0.05 * (7.0 * j as f64).sin()
                }
            })
            .collect();
        let p1: Vec<f64> = (0..n)
            .map(|j| {
                if j >= 100 {
                    1.0 + 0.3 * (j as f64).cos()
                } else {
                    0.05 * (5.0 * j as f64).cos()
                }
            })
            .collect();
        let (c, s) = (30f64.to_radians().cos(), 30f64.to_radians().sin());
        let mut l = vec![0.0; n * 2];
        for j in 0..n {
            l[j] = c * p0[j] - s * p1[j];
            l[n + j] = s * p0[j] + c * p1[j];
        }
        for kaiser in [true, false] {
            let mut m = l.clone();
            let (r, it) = varimax(&mut m, n, 2, kaiser, 1e-8, 1000);
            assert!(it > 0 && it <= 1000);
            // each rotated column is one patch (up to sign and order)
            let a = cosine(&m[..n], &p0).abs().max(cosine(&m[..n], &p1).abs());
            let b = cosine(&m[n..], &p0).abs().max(cosine(&m[n..], &p1).abs());
            assert!(a > 0.999 && b > 0.999, "kaiser {kaiser}: {a} {b}");
            // R is a rotation: orthogonal, and L R is the rotated loadings
            let rr = r[0] * r[0] + r[1] * r[1];
            assert!((rr - 1.0).abs() < 1e-12 && (r[0] * r[2] + r[1] * r[3]).abs() < 1e-12);
            if !kaiser {
                let rot = rotate(&l, n, 2, &r);
                assert!(rot.iter().zip(&m).all(|(a, b)| (a - b).abs() < 1e-12));
            }
        }
    }

    #[test]
    fn gappy_projection_recovers_a_dropped_date() {
        let mut tr = make([10.0, 5.0, 2.0]);
        let drop = 17;
        let truth_row: Vec<f64> = tr.x[drop * N..(drop + 1) * N].to_vec();
        // 60% of the pixels missing at that date: it is dropped from the fit
        for j in 0..N {
            if j % 5 < 3 {
                tr.x[drop * N + j] = f64::NAN;
            }
        }
        let d = Dense::new(&tr.x, T, N, None, 0.8, 0.5).unwrap().with_block(300);
        assert_eq!(d.kept_dates().len(), T - 1);
        assert!(!d.kept_dates().contains(&drop));
        let mut dec = run(&d, 3, Solver::Exact, 2);
        fix_signs(&mut dec);
        let v = all_loadings(&d, &dec);
        let mut row = Vec::new();
        d.weighted_row(drop, &mut row);
        assert!(row.iter().any(|x| x.is_nan()));
        let c = project_gappy(&v, 3, &row);
        for k in 0..3 {
            // the true score of that date on the recovered mode: the dropped
            // date's row (all pixels) against the mode's map, centred
            // (the pixels' means are over the kept dates, which exclude it)
            let want: f64 = (0..N)
                .map(|j| {
                    let mean = (0..T).filter(|&q| q != drop).map(|q| tr.x[q * N + j]).sum::<f64>() / (T - 1) as f64;
                    v[k * N + j] * (truth_row[j] - mean)
                })
                .sum();
            assert!((c[k] - want).abs() < 0.08, "mode {k}: {} vs {want}", c[k]);
            // and it is the true s u v on the true mode, up to sign and the
            // centring over the other dates (the signal's mean there is -u/(T-1))
            let truth = tr.s[k] * (N as f64).sqrt() * tr.u[k][drop] * (1.0 + 1.0 / (T - 1) as f64);
            assert!((c[k].abs() - truth.abs()).abs() < 0.3, "mode {k}: {} vs {truth}", c[k]);
        }
        // too few pixels: NaN
        let few = vec![f64::NAN; N];
        assert!(project_gappy(&v, 3, &few).iter().all(|x| x.is_nan()));
    }

    #[test]
    fn dense_filters_pixels_and_dates() {
        let (t, n) = (10, 6);
        let mut x: Vec<f64> = (0..t * n).map(|i| (i % 5) as f64).collect();
        // pixel 2: valid on 5 of 10 dates (dropped at min_valid 0.8)
        for i in 0..5 {
            x[i * n + 2] = f64::NAN;
        }
        // date 3: valid on 2 of the 5 kept pixels (dropped at 0.5)
        for j in [0, 1, 4] {
            x[3 * n + j] = f64::NAN;
        }
        let d = Dense::new(&x, t, n, None, 0.8, 0.5).unwrap();
        assert_eq!(d.kept_pixels(), &[0, 1, 3, 4, 5]);
        assert_eq!(d.kept_dates(), &[0, 1, 2, 4, 5, 6, 7, 8, 9]);
        assert_eq!(Blocks::dates(&d), 9);
        let mut buf = Vec::new();
        let r = d.block(0, &mut buf);
        assert_eq!(r, 0..5);
        // each column has mean zero over the kept dates
        for col in buf.chunks(9) {
            assert!(col.iter().sum::<f64>().abs() < 1e-12);
        }
        assert!(Dense::new(&x, t, n, None, 1.1, 0.5).is_err());
    }

    /// Timing of a decomposition on a large synthetic cube: `cargo test --release
    /// eof::tests::timing -- --ignored --nocapture`.
    struct Synth {
        t: usize,
        n: usize,
    }
    impl Blocks for Synth {
        fn dates(&self) -> usize {
            self.t
        }
        fn block_count(&self) -> usize {
            self.n.div_ceil(BLOCK_PIXELS)
        }
        fn block(&self, b: usize, out: &mut Vec<f64>) -> Range<usize> {
            let (s, e) = (b * BLOCK_PIXELS, ((b + 1) * BLOCK_PIXELS).min(self.n));
            out.clear();
            out.resize(self.t * (e - s), 0.0);
            for (jj, col) in out.chunks_mut(self.t).enumerate() {
                let j = (s + jj) as u64;
                let a = splitmix64(j) as f64 / u64::MAX as f64;
                let c = splitmix64(j ^ 99) as f64 / u64::MAX as f64;
                for (t, v) in col.iter_mut().enumerate() {
                    let t = t as f64;
                    *v = 10.0 * a * (t * 0.26).sin()
                        + 5.0 * c * (t * 0.11).cos()
                        + 0.1 * (a - c) * (t * 0.5).sin()
                        + 0.05 * (splitmix64(j.wrapping_mul(1315423911) ^ t as u64) as f64 / u64::MAX as f64 - 0.5);
                }
            }
            s..e
        }
    }

    #[test]
    #[ignore]
    fn timing() {
        let x = Synth { t: 240, n: 1_000_000 };
        for solver in [
            Solver::Exact,
            Solver::Randomized {
                oversampling: 10,
                power_iterations: 2,
            },
        ] {
            let st = std::time::Instant::now();
            let s = Settings {
                modes: 10,
                solver,
                seed: 1,
            };
            let d = decompose(&x, &s, 4, &|_| {}).unwrap();
            println!("{solver:?}: {:?}, s = {:?}", st.elapsed(), &d.singular[..3]);
        }
    }
}
