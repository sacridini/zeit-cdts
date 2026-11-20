//! iMAD, the iteratively re-weighted multivariate alteration detection, and the relative radiometric
//! normalization from its invariant pixels.
//!
//! Written from the papers (clean room: no implementation of the method was read): Nielsen, Conradsen and Simpson
//! (1998, RSE 64:1-19) for the MAD transformation, Nielsen (2007, IEEE TIP 16:463-478) for the iterations and the
//! regularization, Canty and Nielsen (2008, RSE 112:1025-1036) for the chi-square standardization and the
//! normalization, Canty, Nielsen and Schmidt (2004, RSE 91:441-451) for the orthogonal regression and its tests.
//!
//! The statistics are float64 and are gathered in fixed blocks of `CHUNK` samples merged in block order (Chan et
//! al.'s pairwise update of the weighted co-moments), so the bits of every result are independent of the number of
//! threads.

use crate::dist::{chi2_sf, f_upper, t_two_sided};
use crate::gsl_symmv::eigen_symmv_desc;
use crate::parallel::{pool, threads, Shared};
use crate::sample64::{present, Sample64};
use rayon::prelude::*;

/// Samples per block of the statistics (fixed: it must not depend on the threads).
const CHUNK: usize = 4096;

/// Calls `f(block)` for the blocks `0..n` on the threads of `n_jobs`, and returns the results in block order.
fn map_blocks<R: Send>(n: usize, n_jobs: i32, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    let t = threads(n_jobs).min(n.max(1));
    if t <= 1 {
        (0..n).map(f).collect()
    } else {
        pool(t).install(|| (0..n).into_par_iter().map(f).collect())
    }
}

// ---------------------------------------------------------------------------
// Weighted moments

/// The weighted moments of `d`-vectors: the sum of the weights, the weighted mean and the weighted co-moment
/// matrix `sum w (z - mean)(z - mean)^T` (`d x d`, row-major), mergeable by the pairwise update.
#[derive(Clone, Debug)]
pub struct Moments {
    pub w: f64,
    pub mean: Vec<f64>,
    pub comoment: Vec<f64>,
}

impl Moments {
    pub fn new(d: usize) -> Moments {
        Moments {
            w: 0.0,
            mean: vec![0.0; d],
            comoment: vec![0.0; d * d],
        }
    }

    /// Adds the moments of another set of samples (Chan, Golub and LeVeque's pairwise update).
    pub fn merge(&mut self, o: &Moments) {
        if o.w <= 0.0 {
            return;
        }
        if self.w <= 0.0 {
            *self = o.clone();
            return;
        }
        let d = self.mean.len();
        let w = self.w + o.w;
        let f = self.w * o.w / w;
        let delta: Vec<f64> = (0..d).map(|i| o.mean[i] - self.mean[i]).collect();
        for r in 0..d {
            for c in 0..d {
                self.comoment[r * d + c] += o.comoment[r * d + c] + delta[r] * delta[c] * f;
            }
        }
        for i in 0..d {
            self.mean[i] += delta[i] * o.w / w;
        }
        self.w = w;
    }

    /// The covariance matrix with the normalization `1 / sum(w)`.
    pub fn cov(&self) -> Vec<f64> {
        self.comoment.iter().map(|c| c / self.w).collect()
    }
}

/// The moments of the rows `rows` (`d` values each) of `z`, row `i` with weight `w[i]`; two passes, the mean first.
fn block_moments(z: &[f64], d: usize, w: &[f64]) -> Moments {
    let mut m = Moments::new(d);
    let total: f64 = w.iter().sum();
    if !(total > 0.0) {
        return m;
    }
    for (row, &wi) in z.chunks_exact(d).zip(w) {
        for j in 0..d {
            m.mean[j] += wi * row[j];
        }
    }
    for v in &mut m.mean {
        *v /= total;
    }
    let mut dev = vec![0.0; d];
    for (row, &wi) in z.chunks_exact(d).zip(w) {
        for j in 0..d {
            dev[j] = row[j] - m.mean[j];
        }
        for r in 0..d {
            let f = wi * dev[r];
            for c in r..d {
                m.comoment[r * d + c] += f * dev[c];
            }
        }
    }
    for r in 0..d {
        for c in 0..r {
            m.comoment[r * d + c] = m.comoment[c * d + r];
        }
    }
    m.w = total;
    m
}

// ---------------------------------------------------------------------------
// Small dense algebra (row-major)

/// The lower Cholesky factor of the symmetric positive definite `a` (`n x n`), or None when a pivot is not
/// positive (relative to its diagonal element).
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= l[j * n + k] * l[j * n + k];
        }
        if !(s > 1e-12 * a[j * n + j].abs()) || !s.is_finite() {
            return None;
        }
        let d = s.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Some(l)
}

/// Solves `L x = b` in place (`L` lower).
fn solve_lower(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// Solves `L^T x = b` in place (`L` lower).
fn solve_lower_t(l: &[f64], n: usize, b: &mut [f64]) {
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= l[k * n + i] * b[k];
        }
        b[i] = s / l[i * n + i];
    }
}

/// `A^-1 B` for the Cholesky factor `l` of `A`, all `n x n`.
fn chol_solve_mat(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    let mut col = vec![0.0; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = b[r * n + c];
        }
        solve_lower(l, n, &mut col);
        solve_lower_t(l, n, &mut col);
        for r in 0..n {
            out[r * n + c] = col[r];
        }
    }
    out
}

fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let f = a[i * n + k];
            for j in 0..n {
                out[i * n + j] += f * b[k * n + j];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Canonical correlation

/// The penalty of the regularized canonical correlation (Nielsen 2007).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Penalty {
    /// `I`: the length of the coefficient vectors.
    Ridge,
    /// `D^T D`, `D` the second differences: the curvature of the coefficients as functions of the band's position.
    Curvature,
}

/// The penalty matrix of `n` bands.
fn omega(n: usize, penalty: Penalty) -> Vec<f64> {
    let mut o = vec![0.0; n * n];
    match penalty {
        Penalty::Ridge => (0..n).for_each(|i| o[i * n + i] = 1.0),
        Penalty::Curvature => {
            for r in 0..n.saturating_sub(2) {
                let d = [(r, 1.0), (r + 1, -2.0), (r + 2, 1.0)];
                for &(i, a) in &d {
                    for &(j, b) in &d {
                        o[i * n + j] += a * b;
                    }
                }
            }
        }
    }
    o
}

/// The canonical correlation analysis of the two dates.
#[derive(Clone, Debug)]
pub struct Cca {
    /// The canonical correlations, descending.
    pub rho: Vec<f64>,
    /// The coefficients of date 1 (`n x n`, row-major: row = band, column = canonical pair).
    pub a: Vec<f64>,
    /// The coefficients of date 2, the same layout.
    pub b: Vec<f64>,
    /// The weighted means, date 1 then date 2 (`2n`).
    pub mean: Vec<f64>,
    /// The variances of the MAD variates (MAD 1 = the pair of the smallest correlation, so the largest variance).
    pub var_mad: Vec<f64>,
    /// Whether some pair had `1 - rho < 1e-12` (the two dates are identical in some combination of bands).
    pub identical: bool,
}

/// `sxy^T` of the `n x n` block `(rows 0..n, columns n..2n)` of the `2n x 2n` matrix, etc.: the blocks of `s`.
fn blocks(s: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let d = 2 * n;
    let mut xx = vec![0.0; n * n];
    let mut xy = vec![0.0; n * n];
    let mut yy = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            xx[r * n + c] = s[r * d + c];
            xy[r * n + c] = s[r * d + n + c];
            yy[r * n + c] = s[(n + r) * d + n + c];
        }
    }
    (xx, xy, yy)
}

fn transpose_sq(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            t[c * n + r] = a[r * n + c];
        }
    }
    t
}

/// One side of the coupled eigenproblems `Sxy Syy^-1 Syx a = rho^2 Sxx a`: `rho^2` descending, the vectors `a`
/// (columns, `a^T Sxx a = 1`) and `Syy^-1 Syx` (`n x n`).
fn one_side(
    sxx: &[f64],
    sxy: &[f64],
    syy: &[f64],
    n: usize,
    bad: &str,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    let lx = cholesky(sxx, n).ok_or_else(|| {
        format!("the bands of date {bad} are collinear or constant over the valid pixels: drop a band or set lam > 0")
    })?;
    let ly = cholesky(syy, n).ok_or_else(|| {
        format!(
            "the bands of date {} are collinear or constant over the valid pixels: drop a band or set lam > 0",
            if bad == "1" { "2" } else { "1" }
        )
    })?;
    let syx = transpose_sq(sxy, n);
    let m = chol_solve_mat(&ly, n, &syx); // Syy^-1 Syx
    let t1 = matmul(sxy, &m, n);
    // K = Lx^-1 T1 Lx^-T
    let mut t2 = vec![0.0; n * n];
    let mut col = vec![0.0; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = t1[r * n + c];
        }
        solve_lower(&lx, n, &mut col);
        for r in 0..n {
            t2[r * n + c] = col[r];
        }
    }
    // K^T = Lx^-1 T2^T
    let t2t = transpose_sq(&t2, n);
    let mut k = vec![0.0; n * n];
    for c in 0..n {
        for r in 0..n {
            col[r] = t2t[r * n + c];
        }
        solve_lower(&lx, n, &mut col);
        for r in 0..n {
            k[r * n + c] = col[r];
        }
    }
    for r in 0..n {
        for c in 0..r {
            let v = 0.5 * (k[r * n + c] + k[c * n + r]);
            k[r * n + c] = v;
            k[c * n + r] = v;
        }
    }
    let (lam, u) = eigen_symmv_desc(&mut k, n);
    let mut a = vec![0.0; n * n];
    for c in 0..n {
        for r in 0..n {
            col[r] = u[r * n + c];
        }
        solve_lower_t(&lx, n, &mut col);
        for r in 0..n {
            a[r * n + c] = col[r];
        }
    }
    Ok((lam, a, m))
}

/// The canonical correlations and variates of the weighted moments `m` of `[x; y]` (`2n` values), with the
/// penalty `lam * Omega` added to both covariance matrices of the constraints (`lam = 0`: none).
pub fn cca(m: &Moments, n: usize, lam: f64, penalty: Penalty) -> Result<Cca, String> {
    assert_eq!(m.mean.len(), 2 * n);
    let s = m.cov();
    let (sxx0, sxy, syy0) = blocks(&s, n);
    let (mut sxx, mut syy) = (sxx0.clone(), syy0.clone());
    if lam > 0.0 {
        let o = omega(n, penalty);
        for i in 0..n * n {
            sxx[i] += lam * o[i];
            syy[i] += lam * o[i];
        }
    }
    let (lam2, a, my) = one_side(&sxx, &sxy, &syy, n, "1")?;
    let rho: Vec<f64> = lam2.iter().map(|&l| l.clamp(0.0, 1.0).sqrt()).collect();
    let mut a = a;
    let mut b = vec![0.0; n * n];
    // second side, only when a correlation is too small to divide by
    let second = if rho.iter().any(|&r| r < 1e-12) {
        let sxy_t = transpose_sq(&sxy, n);
        Some(one_side(&syy, &sxy_t, &sxx, n, "2")?)
    } else {
        None
    };
    for c in 0..n {
        if rho[c] >= 1e-12 {
            for r in 0..n {
                let mut v = 0.0;
                for k in 0..n {
                    v += my[r * n + k] * a[k * n + c];
                }
                b[r * n + c] = v / rho[c];
            }
        } else if let Some((_, b2, _)) = &second {
            for r in 0..n {
                b[r * n + c] = b2[r * n + c];
            }
        }
    }
    // sign: the sum of the correlations of U_i with the bands of date 1 positive
    for c in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let mut sa = 0.0;
            for k in 0..n {
                sa += sxx0[j * n + k] * a[k * n + c];
            }
            sum += sa / sxx0[j * n + j].sqrt();
        }
        if sum < 0.0 {
            for r in 0..n {
                a[r * n + c] = -a[r * n + c];
                b[r * n + c] = -b[r * n + c];
            }
        }
    }
    // variances of the MAD variates, MAD 1 = the last (smallest) correlation
    let mut identical = false;
    let mut var_mad = vec![0.0; n];
    for i in 0..n {
        let k = n - 1 - i;
        let v = if lam > 0.0 {
            // the constraint is penalized: the variance is that of the data
            let mut q = 0.0;
            for r in 0..n {
                for c in 0..n {
                    q += a[r * n + k] * sxx0[r * n + c] * a[c * n + k] + b[r * n + k] * syy0[r * n + c] * b[c * n + k]
                        - 2.0 * a[r * n + k] * sxy[r * n + c] * b[c * n + k];
                }
            }
            q
        } else {
            2.0 * (1.0 - rho[k])
        };
        if v < 2e-12 {
            identical = true;
            var_mad[i] = 2e-12;
        } else {
            var_mad[i] = v;
        }
    }
    Ok(Cca {
        rho,
        a,
        b,
        mean: m.mean.clone(),
        var_mad,
        identical,
    })
}

// ---------------------------------------------------------------------------
// The transformation and the iterations

/// The fixed transformation of a fit: what is applied to every pixel.
#[derive(Clone, Debug)]
pub struct Transform {
    pub n: usize,
    /// The means, date 1 then date 2 (`2n`).
    pub mean: Vec<f64>,
    /// The coefficients (`n x n`, row = band, column = canonical pair, descending correlation).
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    /// The standard deviations the MAD variates are divided by, MAD 1 first.
    pub sd: Vec<f64>,
}

impl Transform {
    /// The MAD variates of the pixel (`x`, `y`: `n` bands each) into `mad` (MAD 1 first); returns the chi-square.
    #[inline]
    pub fn mad(&self, x: &[f64], y: &[f64], mad: &mut [f64]) -> f64 {
        let n = self.n;
        let mut chi2 = 0.0;
        for i in 0..n {
            let k = n - 1 - i;
            let (mut u, mut v) = (0.0, 0.0);
            for j in 0..n {
                u += self.a[j * n + k] * (x[j] - self.mean[j]);
                v += self.b[j * n + k] * (y[j] - self.mean[n + j]);
            }
            mad[i] = u - v;
            let s = mad[i] / self.sd[i];
            chi2 += s * s;
        }
        chi2
    }

    /// The canonical variates `U_1..U_n` and `V_1..V_n` (descending correlation).
    pub fn canonical(&self, x: &[f64], y: &[f64], u: &mut [f64], v: &mut [f64]) {
        let n = self.n;
        for k in 0..n {
            let (mut su, mut sv) = (0.0, 0.0);
            for j in 0..n {
                su += self.a[j * n + k] * (x[j] - self.mean[j]);
                sv += self.b[j * n + k] * (y[j] - self.mean[n + j]);
            }
            u[k] = su;
            v[k] = sv;
        }
    }

    /// The probability of no change of the pixel `z = [x; y]`, and its chi-square.
    #[inline]
    pub fn pnc(&self, z: &[f64], scratch: &mut [f64]) -> (f64, f64) {
        let n = self.n;
        let chi2 = self.mad(&z[..n], &z[n..], scratch);
        (chi2_sf(chi2, n as f64), chi2)
    }
}

/// The options of the iterations.
#[derive(Clone, Copy, Debug)]
pub struct ImadOptions {
    pub max_iter: usize,
    /// Stops when the largest change of a canonical correlation is below this (`0`: runs `max_iter` iterations).
    pub tol: f64,
    pub lam: f64,
    pub penalty: Penalty,
    /// The MAD variates' standard deviation: `true` the one the current weights give (`sqrt(2 (1 - rho))`, Canty
    /// and Nielsen 2008), `false` the unweighted standard deviation over all the pixels (Nielsen 2007).
    pub weighted_sd: bool,
}

impl Default for ImadOptions {
    fn default() -> Self {
        ImadOptions {
            max_iter: 50,
            tol: 1e-3,
            lam: 0.0,
            penalty: Penalty::Ridge,
            weighted_sd: true,
        }
    }
}

/// The result of the iterations.
#[derive(Clone, Debug)]
pub struct ImadFit {
    pub transform: Transform,
    pub cca: Cca,
    /// The canonical correlations of every iteration (descending).
    pub rho_history: Vec<Vec<f64>>,
    /// The largest change of a correlation at each iteration (NaN at the first).
    pub max_delta: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// The weighted covariance matrix of `[x; y]` of the last iteration (`2n x 2n`).
    pub cov: Vec<f64>,
    /// The number of pixels the statistics were computed from.
    pub n_pixels: usize,
    pub warnings: Vec<String>,
}

/// The unweighted standard deviations of the MAD variates of the transform `t` over all the rows of `z`.
fn unweighted_sd(z: &[f64], n: usize, t: &Transform, n_jobs: i32) -> Vec<f64> {
    let rows = z.len() / (2 * n);
    let parts = map_blocks(rows.div_ceil(CHUNK), n_jobs, |c| {
        let r0 = c * CHUNK;
        let r1 = ((c + 1) * CHUNK).min(rows);
        let mut mad = vec![0.0; (r1 - r0) * n];
        for (i, row) in z[r0 * 2 * n..r1 * 2 * n].chunks_exact(2 * n).enumerate() {
            t.mad(&row[..n], &row[n..], &mut mad[i * n..(i + 1) * n]);
        }
        block_moments(&mad, n, &vec![1.0; r1 - r0])
    });
    let mut m = Moments::new(n);
    for p in &parts {
        m.merge(p);
    }
    (0..n).map(|i| (m.comoment[i * n + i] / m.w).sqrt()).collect()
}

/// The weighted moments of the rows of `z` with the probabilities of no change of `prev` as weights (all 1
/// without one).
fn weighted_moments(z: &[f64], n: usize, prev: Option<&Transform>, n_jobs: i32) -> Moments {
    let d = 2 * n;
    let rows = z.len() / d;
    let parts = map_blocks(rows.div_ceil(CHUNK), n_jobs, |c| {
        let r0 = c * CHUNK;
        let r1 = ((c + 1) * CHUNK).min(rows);
        let block = &z[r0 * d..r1 * d];
        let w: Vec<f64> = match prev {
            None => vec![1.0; r1 - r0],
            Some(t) => {
                let mut s = vec![0.0; n];
                block.chunks_exact(d).map(|row| t.pnc(row, &mut s).0).collect()
            }
        };
        block_moments(block, d, &w)
    });
    let mut m = Moments::new(d);
    for p in &parts {
        m.merge(p);
    }
    m
}

/// The iterations on the valid pixels `z` (`pixel x 2n`, row-major, float64: the `n` bands of date 1 then the `n`
/// of date 2). Iteration 1 is the original MAD; `max_iter = 1` stops there.
pub fn imad_fit(z: &[f64], n: usize, opts: &ImadOptions, n_jobs: i32) -> Result<ImadFit, String> {
    let d = 2 * n;
    if n == 0 || z.len() % d != 0 {
        return Err("z must be (pixels, 2 * bands)".into());
    }
    let rows = z.len() / d;
    if rows < 2 * d + 2 {
        return Err(format!("too few valid pixels ({rows}) for {n} bands"));
    }
    let mut prev: Option<Transform> = None;
    let mut history: Vec<Vec<f64>> = Vec::new();
    let mut delta: Vec<f64> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut converged = false;
    let mut last: Option<(Transform, Cca, Vec<f64>)> = None;
    for it in 0..opts.max_iter.max(1) {
        let m = weighted_moments(z, n, prev.as_ref(), n_jobs);
        if !(m.w > 0.0) {
            return Err("every pixel has a zero probability of no change".into());
        }
        let c = cca(&m, n, opts.lam, opts.penalty)?;
        if c.identical && !warnings.iter().any(|w| w.contains("identical")) {
            warnings.push("date 1 and date 2 have (almost) identical bands; drop them".into());
        }
        let dlt = match history.last() {
            Some(p) => c.rho.iter().zip(p).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max),
            None => f64::NAN,
        };
        history.push(c.rho.clone());
        delta.push(dlt);
        let mut t = Transform {
            n,
            mean: c.mean.clone(),
            a: c.a.clone(),
            b: c.b.clone(),
            sd: c.var_mad.iter().map(|v| v.sqrt()).collect(),
        };
        let cov = m.cov();
        let stop = it > 0 && dlt < opts.tol;
        if !opts.weighted_sd {
            t.sd = vec![1.0; n];
            t.sd = unweighted_sd(z, n, &t, n_jobs);
        }
        prev = Some(t.clone());
        last = Some((t, c, cov));
        if stop {
            converged = true;
            break;
        }
    }
    if !converged && opts.tol > 0.0 && opts.max_iter > 1 {
        warnings.push(format!("iMAD did not converge in {} iterations", history.len()));
    }
    let (transform, cca, cov) = last.expect("one iteration");
    Ok(ImadFit {
        transform,
        cca,
        iterations: history.len(),
        rho_history: history,
        max_delta: delta,
        converged,
        cov,
        n_pixels: rows,
        warnings,
    })
}

/// The correlations of the MAD variates with the bands of the dates under the final weights: `(date 1, date 2)`,
/// each `n x n` with row `i` = MAD `i`, column `j` = band `j`.
pub fn loadings(fit: &ImadFit) -> (Vec<f64>, Vec<f64>) {
    let n = fit.transform.n;
    let d = 2 * n;
    let (t, s) = (&fit.transform, &fit.cov);
    let mut lx = vec![0.0; n * n];
    let mut ly = vec![0.0; n * n];
    for i in 0..n {
        let k = n - 1 - i;
        for j in 0..n {
            let (mut cx, mut cy) = (0.0, 0.0);
            for l in 0..n {
                // cov(MAD_i, x_j) = (Sxx a - Sxy b)_j ; cov(MAD_i, y_j) = (Syx a - Syy b)_j
                cx += s[j * d + l] * t.a[l * n + k] - s[j * d + n + l] * t.b[l * n + k];
                cy += s[(n + j) * d + l] * t.a[l * n + k] - s[(n + j) * d + n + l] * t.b[l * n + k];
            }
            // MAD_i = U - V has the variance of the correlation of the pair (not the divisor `sd`, which can be
            // the unweighted one): the loadings are those of the weighted analysis
            let sd_w = fit.cca.var_mad[i].sqrt();
            lx[i * n + j] = cx / (sd_w * s[j * d + j].sqrt());
            ly[i * n + j] = cy / (sd_w * s[(n + j) * d + n + j].sqrt());
        }
    }
    (lx, ly)
}

/// The MAD variates, the chi-square, the probability of no change and optionally the canonical variates of two
/// dates. `x` and `y` are `(band, pixel)` (`n` bands, `npix` pixels each); a pixel with a value NaN, infinite or
/// among `missing` in any band of either date is NaN in every output. `out_mad` is `(n, npix)`, `out_cv`
/// `(2n, npix)` (U then V). Every value is written.
#[allow(clippy::too_many_arguments)]
pub fn imad_apply<T: Sample64>(
    x: &[T],
    y: &[T],
    n: usize,
    npix: usize,
    t: &Transform,
    missing: &[f64],
    out_mad: &mut [f32],
    out_chi2: &mut [f32],
    out_pnc: &mut [f32],
    out_cv: Option<&mut [f32]>,
    n_jobs: i32,
) {
    assert_eq!(x.len(), n * npix);
    assert_eq!(y.len(), n * npix);
    assert_eq!(out_mad.len(), n * npix);
    assert_eq!(out_chi2.len(), npix);
    assert_eq!(out_pnc.len(), npix);
    if let Some(cv) = &out_cv {
        assert_eq!(cv.len(), 2 * n * npix);
    }
    let (sm, sc, sp) = (Shared::new(out_mad), Shared::new(out_chi2), Shared::new(out_pnc));
    let scv = out_cv.map(Shared::new);
    let job = |c: usize| {
        let p0 = c * CHUNK;
        let p1 = ((c + 1) * CHUNK).min(npix);
        let mut xv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        let mut mad = vec![0.0; n];
        let mut u = vec![0.0; n];
        let mut v = vec![0.0; n];
        for p in p0..p1 {
            let mut ok = true;
            for j in 0..n {
                xv[j] = x[j * npix + p].to_f64();
                yv[j] = y[j * npix + p].to_f64();
                ok &= present(xv[j], missing) && present(yv[j], missing);
            }
            // SAFETY: every job writes its own pixels of each plane.
            unsafe {
                if ok {
                    let chi2 = t.mad(&xv, &yv, &mut mad);
                    sc.slice_mut(p..p + 1)[0] = chi2 as f32;
                    sp.slice_mut(p..p + 1)[0] = chi2_sf(chi2, n as f64) as f32;
                    for i in 0..n {
                        sm.slice_mut(i * npix + p..i * npix + p + 1)[0] = mad[i] as f32;
                    }
                    if let Some(cv) = &scv {
                        t.canonical(&xv, &yv, &mut u, &mut v);
                        for k in 0..n {
                            cv.slice_mut(k * npix + p..k * npix + p + 1)[0] = u[k] as f32;
                            cv.slice_mut((n + k) * npix + p..(n + k) * npix + p + 1)[0] = v[k] as f32;
                        }
                    }
                } else {
                    sc.slice_mut(p..p + 1)[0] = f32::NAN;
                    sp.slice_mut(p..p + 1)[0] = f32::NAN;
                    for i in 0..n {
                        sm.slice_mut(i * npix + p..i * npix + p + 1)[0] = f32::NAN;
                    }
                    if let Some(cv) = &scv {
                        for k in 0..2 * n {
                            cv.slice_mut(k * npix + p..k * npix + p + 1)[0] = f32::NAN;
                        }
                    }
                }
            }
        }
    };
    let nchunks = npix.div_ceil(CHUNK);
    let t_n = threads(n_jobs).min(nchunks.max(1));
    if t_n <= 1 {
        (0..nchunks).for_each(job);
    } else {
        pool(t_n).install(|| (0..nchunks).into_par_iter().for_each(job));
    }
}

/// The probabilities of no change of the rows of `z` (`pixel x 2n`).
pub fn imad_pnc(z: &[f64], n: usize, t: &Transform, n_jobs: i32) -> Vec<f64> {
    let d = 2 * n;
    let rows = z.len() / d;
    let parts = map_blocks(rows.div_ceil(CHUNK), n_jobs, |c| {
        let r0 = c * CHUNK;
        let r1 = ((c + 1) * CHUNK).min(rows);
        let mut s = vec![0.0; n];
        z[r0 * d..r1 * d]
            .chunks_exact(d)
            .map(|row| t.pnc(row, &mut s).0)
            .collect::<Vec<f64>>()
    });
    parts.concat()
}

// ---------------------------------------------------------------------------
// Radiometric normalization

/// The orthogonal regression of `v` on `u` (equal error variances in both): `v = a + b u`.
#[derive(Clone, Copy, Debug)]
pub struct OrthoFit {
    pub a: f64,
    pub b: f64,
    pub se_a: f64,
    pub se_b: f64,
    pub r: f64,
    pub rmse: f64,
    pub n: usize,
}

/// Canty, Nielsen and Schmidt (2004), appendix A.2: the linear functional relationship with equal error
/// variances. `u` is the x axis (the target date), `v` the y axis (the reference).
pub fn orthogonal_regression(u: &[f64], v: &[f64]) -> Result<OrthoFit, String> {
    let n = u.len();
    if n != v.len() || n < 3 {
        return Err("orthogonal regression needs at least 3 pairs".into());
    }
    let nf = n as f64;
    let (mut su, mut sv) = (0.0, 0.0);
    for i in 0..n {
        su += u[i];
        sv += v[i];
    }
    let (mu, mv) = (su / nf, sv / nf);
    let (mut suu, mut svv, mut suv) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (du, dv) = (u[i] - mu, v[i] - mv);
        suu += du * du;
        svv += dv * dv;
        suv += du * dv;
    }
    suu /= nf;
    svv /= nf;
    suv /= nf;
    if !(suv > 0.0) {
        return Err("reference and target are not positively related on the invariant pixels".into());
    }
    let b = (svv - suu + ((svv - suu).powi(2) + 4.0 * suv * suv).sqrt()) / (2.0 * suv);
    let a = mv - b * mu;
    let sigma2 = nf / ((nf - 2.0) * (1.0 + b * b)) * (svv - 2.0 * b * suv + b * b * suu).max(0.0);
    let tau = sigma2 * b / ((1.0 + b * b) * suv);
    let c = sigma2 * b * (1.0 + b * b) / (nf * suv);
    let var_b = c * (1.0 + tau);
    let var_a = c * (mu * mu * (1.0 + tau) + suv / b);
    Ok(OrthoFit {
        a,
        b,
        se_a: var_a.max(0.0).sqrt(),
        se_b: var_b.max(0.0).sqrt(),
        r: suv / (suu * svv).sqrt(),
        rmse: sigma2.sqrt(),
        n,
    })
}

/// The tests on the hold-out pixels of one band.
#[derive(Clone, Copy, Debug)]
pub struct HoldoutTest {
    pub mean_target: f64,
    pub mean_ref: f64,
    pub mean_norm: f64,
    pub t: f64,
    pub p_t: f64,
    pub var_target: f64,
    pub var_ref: f64,
    pub var_norm: f64,
    pub f: f64,
    pub p_f: f64,
    pub m: usize,
}

fn mean_var(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0);
    (m, v)
}

/// The paired t-test of the means of the reference and the normalized target and the F-test of their variances
/// on the hold-out pixels (the samples are paired: the F-test is approximate).
pub fn holdout_test(target: &[f64], reference: &[f64], fit: &OrthoFit) -> HoldoutTest {
    let m = target.len();
    let norm: Vec<f64> = target.iter().map(|&u| fit.a + fit.b * u).collect();
    let d: Vec<f64> = reference.iter().zip(&norm).map(|(r, n)| r - n).collect();
    let (md, vd) = mean_var(&d);
    let (mt, vt) = mean_var(target);
    let (mr, vr) = mean_var(reference);
    let (mn, vn) = mean_var(&norm);
    let t = md / (vd / m as f64).sqrt();
    let p_t = if vd == 0.0 && md == 0.0 {
        1.0
    } else {
        t_two_sided(t, m as f64 - 1.0)
    };
    let f = vr / vn;
    let up = f_upper(f, m as f64 - 1.0, m as f64 - 1.0);
    HoldoutTest {
        mean_target: mt,
        mean_ref: mr,
        mean_norm: mn,
        t,
        p_t,
        var_target: vt,
        var_ref: vr,
        var_norm: vn,
        f,
        p_f: (2.0 * up.min(1.0 - up)).clamp(0.0, 1.0),
        m,
    }
}

/// The fit of one band and its tests.
#[derive(Clone, Copy, Debug)]
pub struct BandFit {
    pub fit: OrthoFit,
    pub test: Option<HoldoutTest>,
    pub n_train: usize,
    pub n_test: usize,
}

/// The normalization of the target (date 2, columns `n..2n` of `z`) to the reference (date 1): the invariant
/// pixels are the rows of `z` whose probability of no change under `t` exceeds `threshold`; every
/// `holdout_every`-th of them in row order (the 0-based `r` with `r % every == every - 1`) is kept for the tests
/// (`0`: none). Returns the fit of each band and the number of invariant pixels.
pub fn radcal_fit(
    z: &[f64],
    n: usize,
    t: &Transform,
    threshold: f64,
    holdout_every: usize,
    n_jobs: i32,
) -> Result<(Vec<BandFit>, usize), String> {
    let d = 2 * n;
    let pnc = imad_pnc(z, n, t, n_jobs);
    let mut train: Vec<usize> = Vec::new();
    let mut test: Vec<usize> = Vec::new();
    let mut r = 0usize;
    for (i, &p) in pnc.iter().enumerate() {
        if p > threshold {
            if holdout_every > 0 && r % holdout_every == holdout_every - 1 {
                test.push(i);
            } else {
                train.push(i);
            }
            r += 1;
        }
    }
    if train.len() < 30 {
        return Err(format!(
            "only {} invariant pixels for the fit (at least 30 are needed): lower the threshold or give more pixels",
            train.len()
        ));
    }
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let u: Vec<f64> = train.iter().map(|&i| z[i * d + n + j]).collect();
        let v: Vec<f64> = train.iter().map(|&i| z[i * d + j]).collect();
        let fit = orthogonal_regression(&u, &v).map_err(|e| format!("band {}: {e}", j + 1))?;
        let ht = if test.len() >= 3 {
            let tu: Vec<f64> = test.iter().map(|&i| z[i * d + n + j]).collect();
            let tv: Vec<f64> = test.iter().map(|&i| z[i * d + j]).collect();
            Some(holdout_test(&tu, &tv, &fit))
        } else {
            None
        };
        out.push(BandFit {
            fit,
            test: ht,
            n_train: train.len(),
            n_test: test.len(),
        });
    }
    Ok((out, r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample64::rng::Rng;

    /// `pixels x 2n`: date 2 = A date 1 + c + noise, `block` pixels at the start changed in band 2.
    fn pair(rng: &mut Rng, rows: usize, n: usize, sigma: f64, changed: usize) -> Vec<f64> {
        let a: Vec<f64> = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    0.8 + 0.4 * rng.uniform()
                } else {
                    0.1 * (rng.uniform() - 0.5)
                }
            })
            .collect();
        let c: Vec<f64> = (0..n).map(|_| 0.04 * (rng.uniform() - 0.5)).collect();
        let mut z = vec![0.0; rows * 2 * n];
        for p in 0..rows {
            let base: Vec<f64> = (0..n).map(|j| 0.15 + 0.1 * rng.normal() + 0.05 * j as f64).collect();
            for j in 0..n {
                z[p * 2 * n + j] = base[j];
            }
            for j in 0..n {
                let mut v = c[j] + sigma * rng.normal();
                for k in 0..n {
                    v += a[j * n + k] * base[k];
                }
                if p < changed && (j == 2 || (j == 4 && n > 4)) {
                    // a spread of change magnitudes, from 5 to 20 sigma, up or down
                    v += sigma * (5.0 + 15.0 * rng.uniform()) * if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
                }
                z[p * 2 * n + n + j] = v;
            }
        }
        z
    }

    fn quick(z: &[f64], n: usize, lam: f64, iters: usize) -> ImadFit {
        let o = ImadOptions {
            max_iter: iters,
            tol: 1e-9,
            lam,
            ..Default::default()
        };
        imad_fit(z, n, &o, 1).unwrap()
    }

    #[test]
    fn cca_algebra() {
        let n = 4;
        let d = 2 * n;
        let mut rng = Rng::new(3);
        // a random positive definite covariance: G G^T
        let g: Vec<f64> = (0..d * d).map(|_| rng.normal()).collect();
        let mut s = vec![0.0; d * d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    s[i * d + j] += g[i * d + k] * g[j * d + k];
                }
            }
        }
        let mut m = Moments::new(d);
        m.w = 1.0;
        m.comoment = s.clone();
        let c = cca(&m, n, 0.0, Penalty::Ridge).unwrap();
        let (xx, xy, yy) = blocks(&s, n);
        let q = |p: &[f64], l: &[f64], r: &[f64], i: usize, j: usize| {
            let mut v = 0.0;
            for a in 0..n {
                for b in 0..n {
                    v += l[a * n + i] * p[a * n + b] * r[b * n + j];
                }
            }
            v
        };
        for i in 0..n {
            for j in 0..n {
                let delta = if i == j { 1.0 } else { 0.0 };
                assert!((q(&xx, &c.a, &c.a, i, j) - delta).abs() < 1e-10);
                assert!((q(&yy, &c.b, &c.b, i, j) - delta).abs() < 1e-10);
                assert!((q(&xy, &c.a, &c.b, i, j) - c.rho[i] * delta).abs() < 1e-10);
            }
        }
        assert!(c.rho.windows(2).all(|w| w[0] >= w[1]) && c.rho.iter().all(|&r| (0.0..=1.0).contains(&r)));
        // MAD variances 2 (1 - rho), MAD 1 the largest
        assert!(c.var_mad.windows(2).all(|w| w[0] >= w[1]));
        for i in 0..n {
            assert!((c.var_mad[i] - 2.0 * (1.0 - c.rho[n - 1 - i])).abs() < 1e-12);
        }
    }

    #[test]
    fn invariant_to_linear_transforms_of_either_date() {
        let n = 3;
        let mut rng = Rng::new(11);
        let z = pair(&mut rng, 20_000, n, 0.01, 2000);
        let base = quick(&z, n, 0.0, 12);
        // random nonsingular matrices, some diagonal with negative entries
        let mats: Vec<Vec<f64>> = vec![
            vec![2.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 3.0],
            vec![1.0, 0.3, -0.2, 0.1, 0.9, 0.4, -0.3, 0.2, 1.2],
        ];
        for (which, m) in mats.iter().enumerate() {
            for date in 0..2 {
                let mut z2 = z.clone();
                for p in 0..z.len() / (2 * n) {
                    for j in 0..n {
                        let mut v = 0.0;
                        for k in 0..n {
                            v += m[j * n + k] * z[p * 2 * n + date * n + k];
                        }
                        z2[p * 2 * n + date * n + j] = v + 0.5 * j as f64;
                    }
                }
                let f = quick(&z2, n, 0.0, 12);
                assert_eq!(f.iterations, base.iterations);
                for (r1, r2) in base.rho_history.iter().zip(&f.rho_history) {
                    for (a, b) in r1.iter().zip(r2) {
                        assert!((a - b).abs() < 1e-9, "matrix {which} date {date}: {a} {b}");
                    }
                }
                let (p1, p2) = (imad_pnc(&z, n, &base.transform, 1), imad_pnc(&z2, n, &f.transform, 1));
                for (a, b) in p1.iter().zip(&p2) {
                    assert!((a - b).abs() <= 1e-7 * a.abs().max(1e-3), "pnc {a} {b}");
                }
            }
        }
        // a ridge penalty breaks the invariance
        let scaled: Vec<f64> = z
            .chunks_exact(2 * n)
            .flat_map(|r| vec![r[0] * 100.0, r[1], r[2], r[3], r[4], r[5]])
            .collect();
        let (a, b) = (quick(&z, n, 0.01, 6), quick(&scaled, n, 0.01, 6));
        assert!((a.cca.rho[0] - b.cca.rho[0]).abs() > 1e-6 || (a.cca.rho[2] - b.cca.rho[2]).abs() > 1e-6);
    }

    #[test]
    fn detects_the_changed_pixels_and_iterates() {
        let n = 6;
        let mut rng = Rng::new(5);
        let rows = 60_000;
        let changed = 24_000; // 40 % of the image
        let z = pair(&mut rng, rows, n, 0.01, changed);
        let opts = |max_iter: usize, weighted_sd: bool| ImadOptions {
            max_iter,
            tol: 1e-6,
            weighted_sd,
            ..Default::default()
        };
        let rate = |f: &ImadFit| {
            let p = imad_pnc(&z, n, &f.transform, 1);
            let det = p[..changed].iter().filter(|&&v| v < 0.01).count() as f64 / changed as f64;
            let fa = p[changed..].iter().filter(|&&v| v < 0.01).count() as f64 / (rows - changed) as f64;
            (det, fa)
        };
        // the weighted standardization of Canty and Nielsen (2008): iterating finds the changed pixels
        let first = imad_fit(&z, n, &opts(1, true), 1).unwrap();
        let done = imad_fit(&z, n, &opts(50, true), 1).unwrap();
        assert!(done.iterations <= 50, "{} iterations", done.iterations);
        let ((d1, _), (d2, _)) = (rate(&first), rate(&done));
        assert!(d2 > 0.95 && d2 > d1, "detection {d1} -> {d2}");
        // Nielsen's (2007) unweighted standard deviation keeps the false alarms at the level
        let all = imad_fit(&z, n, &opts(50, false), 1).unwrap();
        assert!(all.converged);
        let (_, fa) = rate(&all);
        assert!(fa < 0.02, "false alarms {fa}");
    }

    #[test]
    fn threads_do_not_change_the_bits() {
        let n = 3;
        let mut rng = Rng::new(21);
        let z = pair(&mut rng, 30_000, n, 0.01, 3000);
        let o = ImadOptions {
            max_iter: 8,
            tol: 0.0,
            ..Default::default()
        };
        let a = imad_fit(&z, n, &o, 1).unwrap();
        let b = imad_fit(&z, n, &o, 4).unwrap();
        assert_eq!(a.rho_history, b.rho_history);
        assert_eq!(a.transform.a, b.transform.a);
    }

    #[test]
    fn collinear_bands_are_an_error_until_regularized() {
        let n = 3;
        let mut rng = Rng::new(8);
        let mut z = pair(&mut rng, 5000, n, 0.01, 0);
        for p in 0..5000 {
            z[p * 6 + 1] = z[p * 6]; // band 2 of date 1 duplicates band 1
        }
        let o = ImadOptions {
            max_iter: 3,
            ..Default::default()
        };
        let e = imad_fit(&z, n, &o, 1).unwrap_err();
        assert!(e.contains("collinear"), "{e}");
        let o = ImadOptions {
            max_iter: 3,
            lam: 1e-6 * 0.01,
            ..Default::default()
        };
        assert!(imad_fit(&z, n, &o, 1).is_ok());
    }

    #[test]
    fn orthogonal_regression_is_exact_and_symmetric() {
        let u: Vec<f64> = (0..50)
            .map(|i| (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.1)
            .collect();
        let v: Vec<f64> = u.iter().map(|&x| 2.0 * x + 1.0).collect();
        let f = orthogonal_regression(&u, &v).unwrap();
        assert!((f.b - 2.0).abs() < 1e-12 && (f.a - 1.0).abs() < 1e-12);
        assert!(f.se_a < 1e-6 && f.se_b < 1e-6 && f.rmse < 1e-6);
        let g = orthogonal_regression(&v, &u).unwrap();
        assert!((g.b - 1.0 / f.b).abs() < 1e-12);
        // noisy data: swapping gives 1/b as well
        let mut rng = Rng::new(2);
        let u: Vec<f64> = (0..400).map(|_| rng.normal()).collect();
        let v: Vec<f64> = u.iter().map(|&x| 0.7 * x + 0.2 + 0.1 * rng.normal()).collect();
        let (f, g) = (
            orthogonal_regression(&u, &v).unwrap(),
            orthogonal_regression(&v, &u).unwrap(),
        );
        assert!((g.b * f.b - 1.0).abs() < 1e-12);
        let neg: Vec<f64> = v.iter().map(|x| -x).collect();
        assert!(orthogonal_regression(&u, &neg).is_err());
    }

    #[test]
    fn radcal_recovers_the_inverse_transform() {
        let n = 3;
        let mut rng = Rng::new(17);
        let rows = 40_000;
        let sigma = 0.01;
        let gain = [1.3, 0.8, 1.1];
        let off = [0.02, -0.03, 0.01];
        let changed = 6000;
        let mut z = vec![0.0; rows * 2 * n];
        for p in 0..rows {
            for j in 0..n {
                let s = 0.2 + 0.1 * rng.normal() + 0.03 * j as f64;
                z[p * 2 * n + j] = s + sigma * rng.normal();
                let mut t = gain[j] * s + off[j] + sigma * rng.normal();
                if p < changed {
                    t += 0.2 * (j as f64 - 1.0);
                }
                z[p * 2 * n + n + j] = t;
            }
        }
        let fit = imad_fit(
            &z,
            n,
            &ImadOptions {
                tol: 1e-6,
                weighted_sd: false,
                ..Default::default()
            },
            1,
        )
        .unwrap();
        let (bands, invariant) = radcal_fit(&z, n, &fit.transform, 0.95, 3, 1).unwrap();
        assert!(invariant > 1_000, "{invariant}");
        for j in 0..n {
            let f = &bands[j].fit;
            let (tb, ta) = (1.0 / gain[j], -off[j] / gain[j]);
            assert!(
                (f.b - tb).abs() < 3.0 * f.se_b,
                "band {j}: b {} vs {tb} (se {})",
                f.b,
                f.se_b
            );
            assert!(
                (f.a - ta).abs() < 3.0 * f.se_a,
                "band {j}: a {} vs {ta} (se {})",
                f.a,
                f.se_a
            );
            assert!(f.rmse < 3.0 * sigma * 1.5);
            let t = bands[j].test.unwrap();
            assert!(t.p_t > 0.001, "t-test {}", t.p_t);
        }
        assert_eq!(bands[0].n_train + bands[0].n_test, invariant);
        assert!(bands[0].n_test.abs_diff(invariant / 3) <= 1);
        // a deliberately wrong gain is rejected by the test
        let wrong = OrthoFit {
            b: bands[0].fit.b * 1.1,
            ..bands[0].fit
        };
        let tu: Vec<f64> = (0..500).map(|p| z[p * 2 * n + n + 0]).collect();
        let tv: Vec<f64> = (0..500).map(|p| z[p * 2 * n]).collect();
        assert!(holdout_test(&tu, &tv, &wrong).p_t < 1e-6);
    }

    #[test]
    fn apply_marks_the_invalid_pixels() {
        let n = 2;
        let mut rng = Rng::new(4);
        let z = pair(&mut rng, 6000, n, 0.01, 0);
        let fit = quick(&z, n, 0.0, 3);
        let npix = 6000;
        let mut x = vec![0.0f32; n * npix];
        let mut y = vec![0.0f32; n * npix];
        for p in 0..npix {
            for j in 0..n {
                x[j * npix + p] = z[p * 4 + j] as f32;
                y[j * npix + p] = z[p * 4 + n + j] as f32;
            }
        }
        y[npix + 17] = f32::NAN;
        x[3] = -9999.0;
        let mut mad = vec![0.0f32; n * npix];
        let mut chi2 = vec![0.0f32; npix];
        let mut pnc = vec![0.0f32; npix];
        let mut cv = vec![0.0f32; 2 * n * npix];
        imad_apply(
            &x,
            &y,
            n,
            npix,
            &fit.transform,
            &[-9999.0],
            &mut mad,
            &mut chi2,
            &mut pnc,
            Some(&mut cv),
            2,
        );
        for p in [3usize, 17] {
            assert!(chi2[p].is_nan() && pnc[p].is_nan() && mad[p].is_nan() && mad[npix + p].is_nan() && cv[p].is_nan());
        }
        assert_eq!(chi2.iter().filter(|v| v.is_nan()).count(), 2);
        // the f32 output matches the f64 transform
        let (p, c) = fit.transform.pnc(&z[8 * 4..9 * 4], &mut [0.0; 2]);
        assert!((pnc[8] as f64 - p).abs() < 1e-5 && (chi2[8] as f64 - c).abs() < 1e-3 * c.max(1.0));
    }
}
