//! A linear model of a series with AR(1) errors, fitted by REML: remotePARTS's
//! `fitAR` (Ives et al. 2021, "Statistical inference for trends in
//! spatiotemporal data", Remote Sensing of Environment 266, 112678,
//! doi:10.1016/j.rse.2021.112678), the step of PARTS on each pixel: its trend,
//! with a t-test that accounts for the autocorrelation of its errors.
//!
//! remotePARTS builds the n x n precision matrix of the series at each step of
//! the optimizer over rho. Here the series is transformed as Prais-Winsten's
//! (the AR(1)'s precision is tridiagonal), so the cross-products of the
//! transformed design and series are a quadratic in rho^gap of sums taken once
//! (per gap between valid dates): a step costs O(q^3), whatever the length of
//! the series. The design is first replaced by the Q of its QR and the series
//! by its least-squares residuals, which changes neither the residuals nor the
//! optimum and keeps those sums free of cancellation (a series in kelvin).

use crate::dist::t_two_sided;
use crate::optimize::fmin;
use crate::parallel;

/// How missing values (NaN) enter the AR(1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gaps {
    /// The likelihood of the observations at their dates: rho^k between two
    /// observations k dates apart.
    Exact,
    /// The observations as consecutive, as remotePARTS's `fitAR` (its formula
    /// interface drops the missing values before the fit).
    Ignore,
}

/// The fit of one series.
#[derive(Clone, Debug, PartialEq)]
pub struct ArFit {
    /// One per column of the design.
    pub coefficients: Vec<f64>,
    pub se: Vec<f64>,
    /// Signed (remotePARTS gives |t|).
    pub t: Vec<f64>,
    /// Two-sided, on `df` degrees of freedom.
    pub p: Vec<f64>,
    pub rho: f64,
    /// The residual variance (of the innovations).
    pub mse: f64,
    /// remotePARTS's `logLik`.
    pub loglik: f64,
    /// Valid observations less columns.
    pub df: usize,
    /// y - X beta at each date, NaN where y is missing.
    pub residuals: Vec<f64>,
}

/// The sums of the pairs of consecutive observations `delta` dates apart, over
/// u = (row of Q, residual): lower triangles of (q + 1)^2 matrices.
#[derive(Clone, Debug, Default)]
struct Gap {
    delta: usize,
    count: usize,
    /// sum of u_k u_k'
    a: Vec<f64>,
    /// sum of u_k u_{k-1}' + u_{k-1} u_k'
    b: Vec<f64>,
    /// sum of u_{k-1} u_{k-1}'
    c: Vec<f64>,
}

/// Buffers of one thread's fits, and the last fit's statistics.
struct Scratch {
    q: usize,
    /// the valid dates
    idx: Vec<usize>,
    /// Q of the design's thin QR on the valid dates (n x q, by column)
    qm: Vec<f64>,
    /// its R (q x q, by column)
    r: Vec<f64>,
    /// Q'y
    qty: Vec<f64>,
    /// the least-squares residuals, and their sum of squares
    res: Vec<f64>,
    ssq_res: f64,
    /// a pair of consecutive rows of u = (Q, residuals)
    pair: Vec<f64>,
    gaps: Vec<Gap>,
    n_gaps: usize,
    /// u_0 u_0'
    u0: Vec<f64>,
    /// the transformed cross-products at a rho, and their Cholesky factor
    m: Vec<f64>,
    l: Vec<f64>,
    /// log det(X'X)
    logdet_xtx: f64,
    df: f64,
    // the last fit's statistics
    beta: Vec<f64>,
    se: Vec<f64>,
    rho: f64,
    mse: f64,
    loglik: f64,
    work: Vec<f64>,
}

impl Scratch {
    fn new(n_time: usize, q: usize) -> Scratch {
        let m = q + 1;
        Scratch {
            q,
            idx: Vec::with_capacity(n_time),
            qm: vec![0.0; n_time * q],
            r: vec![0.0; q * q],
            qty: vec![0.0; q],
            res: vec![0.0; n_time],
            ssq_res: 0.0,
            pair: vec![0.0; 2 * m],
            gaps: Vec::new(),
            n_gaps: 0,
            u0: vec![0.0; m * m],
            m: vec![0.0; m * m],
            l: vec![0.0; m * m],
            logdet_xtx: 0.0,
            df: 0.0,
            beta: vec![0.0; q],
            se: vec![0.0; q],
            rho: f64::NAN,
            mse: f64::NAN,
            loglik: f64::NAN,
            work: vec![0.0; 2 * q * q + q],
        }
    }

    /// Fits `y` (NaN missing) on the design `x` (`y.len()` x q, by column):
    /// false when there are fewer than `min_n` valid observations or the
    /// design has no full rank on them.
    fn fit(&mut self, y: &[f64], x: &[f64], gaps: Gaps, min_n: usize) -> bool {
        let q = self.q;
        let n_time = y.len();
        self.idx.clear();
        self.idx.extend((0..n_time).filter(|&t| !y[t].is_nan()));
        let n = self.idx.len();
        if n < min_n.max(q + 2) {
            return false;
        }
        if !self.qr(x, n_time) {
            return false;
        }
        // Q'y and the least-squares residuals, twice for their orthogonality to Q
        let res = &mut self.res[..n];
        for (r, &t) in res.iter_mut().zip(&self.idx) {
            *r = y[t];
        }
        let ssq_y = dot(res, res);
        self.qty.fill(0.0);
        for _ in 0..2 {
            for j in 0..q {
                let col = &self.qm[j * n..(j + 1) * n];
                let d = dot(col, res);
                self.qty[j] += d;
                for (r, c) in res.iter_mut().zip(col) {
                    *r -= d * c;
                }
            }
        }
        self.ssq_res = dot(res, res);
        self.df = (n - q) as f64;
        if self.ssq_res <= 1e-24 * ssq_y {
            // the design fits the series exactly: least squares, no test
            solve_upper(&self.r, q, &self.qty, &mut self.beta);
            self.se.fill(0.0);
            self.rho = f64::NAN;
            self.mse = 0.0;
            self.loglik = f64::NAN;
            return true;
        }
        self.sums(gaps);
        // optim(method = "Brent") on [-1, 1] at its default reltol
        let rho = fmin(-1.0, 1.0, |rho| self.objective(rho), f64::EPSILON.sqrt());
        let f = self.objective(rho);
        if f.is_nan() {
            return false;
        }
        self.statistics(rho, f);
        true
    }

    /// The thin QR of the design's valid rows (classical Gram-Schmidt run
    /// twice): false when a column depends on the others.
    fn qr(&mut self, x: &[f64], n_time: usize) -> bool {
        let q = self.q;
        let n = self.idx.len();
        self.r.fill(0.0);
        self.logdet_xtx = 0.0;
        for j in 0..q {
            for (k, &t) in self.idx.iter().enumerate() {
                self.qm[j * n + k] = x[j * n_time + t];
            }
            let (done, rest) = self.qm.split_at_mut(j * n);
            let qj = &mut rest[..n];
            let norm0 = dot(qj, qj).sqrt();
            for _ in 0..2 {
                for i in 0..j {
                    let qi = &done[i * n..(i + 1) * n];
                    let d = dot(qi, qj);
                    self.r[j * q + i] += d;
                    for (b, a) in qj.iter_mut().zip(qi) {
                        *b -= d * a;
                    }
                }
            }
            let rjj = dot(qj, qj).sqrt();
            if !(rjj > 1e-10 * norm0) {
                return false;
            }
            for v in qj.iter_mut() {
                *v /= rjj;
            }
            self.r[j * q + j] = rjj;
            self.logdet_xtx += 2.0 * rjj.ln();
        }
        true
    }

    /// Column `j` of u = (Q, residuals) on the `n` valid dates.
    fn col<'a>(qm: &'a [f64], res: &'a [f64], q: usize, j: usize, n: usize) -> &'a [f64] {
        if j < q {
            &qm[j * n..(j + 1) * n]
        } else {
            &res[..n]
        }
    }

    /// The sums over pairs of consecutive observations, per gap. Those of the
    /// commonest gap come from the sums over all pairs less those of the other
    /// gaps, taken pair by pair: u'u is diag(I, the residuals' sum of squares)
    /// (Q is orthonormal and the residuals orthogonal to it), and the lagged
    /// products are m^2 dot products of the columns.
    fn sums(&mut self, gaps: Gaps) {
        let m = self.q + 1;
        let n = self.idx.len();
        let delta = |idx: &[usize], k: usize| match gaps {
            Gaps::Exact => idx[k] - idx[k - 1],
            Gaps::Ignore => 1,
        };
        self.n_gaps = 0;
        for k in 1..n {
            let d = delta(&self.idx, k);
            match self.gaps[..self.n_gaps].iter_mut().find(|g| g.delta == d) {
                Some(g) => g.count += 1,
                None => {
                    if self.n_gaps == self.gaps.len() {
                        self.gaps.push(Gap::default());
                    }
                    let g = &mut self.gaps[self.n_gaps];
                    g.delta = d;
                    g.count = 1;
                    for v in [&mut g.a, &mut g.b, &mut g.c] {
                        v.clear();
                        v.resize(m * m, 0.0);
                    }
                    self.n_gaps += 1;
                }
            }
        }
        let main = (0..self.n_gaps).max_by_key(|&g| self.gaps[g].count).unwrap();
        let main_delta = self.gaps[main].delta;
        // the pairs of the other gaps
        if self.n_gaps > 1 {
            let mut pair = std::mem::take(&mut self.pair);
            for k in 1..n {
                let d = delta(&self.idx, k);
                if d == main_delta {
                    continue;
                }
                for j in 0..m {
                    let c = Self::col(&self.qm, &self.res, self.q, j, n);
                    pair[j] = c[k - 1];
                    pair[m + j] = c[k];
                }
                let (prev, cur) = pair.split_at(m);
                let g = self.gaps[..self.n_gaps].iter_mut().find(|g| g.delta == d).unwrap();
                for i in 0..m {
                    for j in 0..=i {
                        g.a[i * m + j] += cur[i] * cur[j];
                        g.b[i * m + j] += cur[i] * prev[j] + prev[i] * cur[j];
                        g.c[i * m + j] += prev[i] * prev[j];
                    }
                }
            }
            self.pair = pair;
        }
        // the commonest gap's: all pairs less the others
        let mut a = std::mem::take(&mut self.gaps[main].a);
        let mut b = std::mem::take(&mut self.gaps[main].b);
        let mut c = std::mem::take(&mut self.gaps[main].c);
        for i in 0..m {
            let ci = Self::col(&self.qm, &self.res, self.q, i, n);
            let (first_i, last_i) = (ci[0], ci[n - 1]);
            for j in 0..=i {
                let cj = Self::col(&self.qm, &self.res, self.q, j, n);
                let (first_j, last_j) = (cj[0], cj[n - 1]);
                let s = if i != j {
                    0.0
                } else if i < self.q {
                    1.0
                } else {
                    self.ssq_res
                };
                let k = i * m + j;
                self.u0[k] = first_i * first_j;
                a[k] = s - first_i * first_j;
                // u_k u_{k-1}' + u_{k-1} u_k' over k >= 1
                b[k] = dot(&ci[1..], &cj[..n - 1]) + dot(&cj[1..], &ci[..n - 1]);
                c[k] = s - last_i * last_j;
                for g in self.gaps[..self.n_gaps].iter().filter(|g| g.delta != main_delta) {
                    a[k] -= g.a[k];
                    b[k] -= g.b[k];
                    c[k] -= g.c[k];
                }
            }
        }
        let g = &mut self.gaps[main];
        (g.a, g.b, g.c) = (a, b, c);
    }

    /// The transformed cross-products at `rho` into `m`, and their LDL'
    /// factor into `l` (D on the diagonal, L's unit diagonal implied): 1 -
    /// rho^2 and the gaps' part of log det V, or NaN where the cross-products
    /// are not positive definite.
    fn factor(&mut self, rho: f64) -> (f64, f64) {
        let m = self.q + 1;
        let one_m_r2 = (1.0 - rho) * (1.0 + rho);
        if !(one_m_r2 > 0.0) {
            return (f64::NAN, f64::NAN);
        }
        // the first observation's row of the transform: sqrt(1 - rho^2) u_0
        for i in 0..m {
            for j in 0..=i {
                self.m[i * m + j] = one_m_r2 * self.u0[i * m + j];
            }
        }
        let mut logdet_gaps = 0.0;
        let ln_r2 = 2.0 * rho.abs().ln();
        for g in &self.gaps[..self.n_gaps] {
            // the k-th row: (u_k - rho^d u_{k-1}) / sqrt(1 + rho^2 + ... + rho^(2(d-1)))
            let (rd, inv_g) = if g.delta == 1 {
                (rho, 1.0)
            } else {
                let sum = -(g.delta as f64 * ln_r2).exp_m1() / one_m_r2;
                logdet_gaps += g.count as f64 * sum.ln();
                (rho.powi(g.delta as i32), 1.0 / sum)
            };
            let rd2 = rd * rd;
            for i in 0..m {
                for j in 0..=i {
                    let k = i * m + j;
                    self.m[k] += inv_g * (g.a[k] - rd * g.b[k] + rd2 * g.c[k]);
                }
            }
        }
        for j in 0..m {
            let mut d = self.m[j * m + j];
            for k in 0..j {
                d -= self.l[j * m + k] * self.l[j * m + k] * self.l[k * m + k];
            }
            if !(d > 0.0) {
                return (f64::NAN, f64::NAN);
            }
            self.l[j * m + j] = d;
            for i in j + 1..m {
                let mut s = self.m[i * m + j];
                for k in 0..j {
                    s -= self.l[i * m + k] * self.l[j * m + k] * self.l[k * m + k];
                }
                self.l[i * m + j] = s / d;
            }
        }
        (one_m_r2, logdet_gaps)
    }

    /// remotePARTS's restricted log-likelihood (AR_fun with logLik.only), the
    /// value minimized: 0.5 (df log s2 + log det V + log det X'V^-1X + df),
    /// where log det V = log det(gaps) - log(1 - rho^2), and D's last pivot is
    /// the sum of squares and the others' product det X'V^-1X / det X'X.
    fn objective(&mut self, rho: f64) -> f64 {
        let (one_m_r2, logdet_gaps) = self.factor(rho);
        if one_m_r2.is_nan() {
            return f64::NAN;
        }
        let (q, m) = (self.q, self.q + 1);
        let mut det = 1.0 / one_m_r2;
        for i in 0..q {
            det *= self.l[i * m + i];
        }
        let sse = self.l[q * m + q];
        let df = self.df;
        0.5 * (df * (sse / df).ln() + det.ln() + logdet_gaps + self.logdet_xtx + df)
    }

    /// The coefficients, their standard errors, the residual variance and the
    /// log-likelihood at `rho`, whose objective is `f` (and factor in `l`).
    fn statistics(&mut self, rho: f64, f: f64) {
        let (q, m) = (self.q, self.q + 1);
        let sse = self.l[q * m + q];
        let s2 = sse / self.df;
        // LDL' to Cholesky's L D^1/2
        for j in 0..m {
            let sd = self.l[j * m + j].sqrt();
            self.l[j * m + j] = sd;
            for i in j + 1..m {
                self.l[i * m + j] *= sd;
            }
        }
        let (g, rest) = self.work.split_at_mut(q * q);
        let (w, b) = rest.split_at_mut(q * q);
        // the coefficients on Q: Q'y and the GLS of the residuals on Q, L_xx' d = (row q of L)
        for i in (0..q).rev() {
            let mut s = self.l[q * m + i];
            for (k, bk) in b.iter().enumerate().skip(i + 1) {
                s -= self.l[k * m + i] * bk;
            }
            b[i] = s / self.l[i * m + i];
        }
        for (b, qty) in b.iter_mut().zip(&self.qty) {
            *b += qty;
        }
        solve_upper(&self.r, q, b, &mut self.beta);
        // Cov(beta) = s2 W W', W = R^-1 L_xx^-T: G = L_xx^-T, then W = R^-1 G
        g.fill(0.0);
        for c in 0..q {
            for i in (0..=c).rev() {
                let mut s = if i == c { 1.0 } else { 0.0 };
                for k in i + 1..=c {
                    s -= self.l[k * m + i] * g[c * q + k];
                }
                g[c * q + i] = s / self.l[i * m + i];
            }
        }
        w.fill(0.0);
        for c in 0..q {
            for i in (0..=c).rev() {
                let mut s = g[c * q + i];
                for k in i + 1..=c {
                    s -= self.r[k * q + i] * w[c * q + k];
                }
                w[c * q + i] = s / self.r[i * q + i];
            }
        }
        for i in 0..q {
            let ww: f64 = (i..q).map(|c| w[c * q + i] * w[c * q + i]).sum();
            self.se[i] = (s2 * ww).sqrt();
        }
        self.rho = rho;
        self.mse = s2;
        self.loglik = 0.5 * self.df * (2.0 * std::f64::consts::PI).ln() + self.logdet_xtx - f;
    }
}

/// The dot product of `a` and `b` in four sums, which the compiler keeps in
/// vector registers.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut s = [0.0f64; 4];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in ca.iter().zip(cb) {
        for l in 0..4 {
            s[l] += x[l] * y[l];
        }
    }
    let mut t = (s[0] + s[2]) + (s[1] + s[3]);
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

/// x = U^-1 b for U upper triangular (q x q, by column).
fn solve_upper(u: &[f64], q: usize, b: &[f64], x: &mut [f64]) {
    for i in (0..q).rev() {
        let mut s = b[i];
        for k in i + 1..q {
            s -= u[k * q + i] * x[k];
        }
        x[i] = s / u[i * q + i];
    }
}

/// Fits the series `y` (NaN missing) on the design `x` (`y.len()` x `q`, by
/// column: `x[t + y.len() * j]`). `[1, t]` is remotePARTS's default `y ~ t`.
/// None with fewer than q + 2 valid observations, or a design without full
/// rank on them. A series the design fits exactly has its least-squares
/// coefficients, standard errors 0, and NaN for the rest.
pub fn fit(y: &[f64], x: &[f64], q: usize, gaps: Gaps) -> Option<ArFit> {
    assert_eq!(x.len(), y.len() * q, "x must be len(y) x q");
    let mut sc = Scratch::new(y.len(), q);
    if !sc.fit(y, x, gaps, 0) {
        return None;
    }
    let n_time = y.len();
    let beta = sc.beta.clone();
    let residuals = (0..n_time)
        .map(|t| {
            if y[t].is_nan() {
                f64::NAN
            } else {
                y[t] - (0..q).map(|j| x[j * n_time + t] * beta[j]).sum::<f64>()
            }
        })
        .collect();
    let t: Vec<f64> = beta
        .iter()
        .zip(&sc.se)
        .map(|(b, s)| if sc.mse > 0.0 { b / s } else { f64::NAN })
        .collect();
    let p = t.iter().map(|&t| t_two_sided(t, sc.df)).collect();
    Some(ArFit {
        coefficients: beta,
        se: sc.se.clone(),
        t,
        p,
        rho: sc.rho,
        mse: sc.mse,
        loglik: sc.loglik,
        df: sc.df as usize,
        residuals,
    })
}

/// The metrics of `batch` per pixel for a design of `q` columns.
pub fn n_metrics(q: usize) -> usize {
    4 * q + 3
}

/// Fits every row of `values` (`n_pixels` x `n_time`, row-major) on the same
/// design `x` (`n_time` x `q`, by column): `n_metrics(q)` x `n_pixels` values,
/// the coefficients (q), their standard errors (q), t (q), p (q), rho, the
/// residual variance and the log-likelihood; NaN for pixels with fewer than
/// `min_valid` (at least q + 2) valid observations.
#[allow(clippy::too_many_arguments)]
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    x: &[f64],
    q: usize,
    gaps: Gaps,
    min_valid: usize,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    assert_eq!(x.len(), n_time * q, "x must be n_time x q");
    let width = n_metrics(q);
    let mut per_pixel = vec![f64::NAN; n_pixels * width];
    parallel::for_each_row(
        &mut per_pixel,
        width,
        n_jobs,
        || Scratch::new(n_time, q),
        |sc, p, out| {
            if !sc.fit(&values[p * n_time..(p + 1) * n_time], x, gaps, min_valid) {
                return;
            }
            for j in 0..q {
                let (b, s) = (sc.beta[j], sc.se[j]);
                let t = if sc.mse > 0.0 { b / s } else { f64::NAN };
                out[j] = b;
                out[q + j] = s;
                out[2 * q + j] = t;
                out[3 * q + j] = t_two_sided(t, sc.df);
            }
            out[4 * q] = sc.rho;
            out[4 * q + 1] = sc.mse;
            out[4 * q + 2] = sc.loglik;
        },
    );
    parallel::transpose(&per_pixel, n_pixels, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trend_design(n: usize) -> Vec<f64> {
        let mut x = vec![1.0; 2 * n];
        for t in 0..n {
            x[n + t] = (t + 1) as f64;
        }
        x
    }

    #[test]
    fn exact_line_has_no_test() {
        let y: Vec<f64> = (1..=10).map(|t| 2.0 + 0.5 * t as f64).collect();
        let f = fit(&y, &trend_design(10), 2, Gaps::Exact).unwrap();
        assert!((f.coefficients[0] - 2.0).abs() < 1e-12 && (f.coefficients[1] - 0.5).abs() < 1e-12);
        assert_eq!(f.se, vec![0.0, 0.0]);
        assert!(f.rho.is_nan() && f.p[1].is_nan());
        // a constant series
        let f = fit(&[3.0; 6], &trend_design(6), 2, Gaps::Exact).unwrap();
        assert!(f.coefficients[1].abs() < 1e-12);
    }

    #[test]
    fn too_few_observations_or_a_rank_deficient_design() {
        let y = [1.0, f64::NAN, 2.0, f64::NAN, f64::NAN, 1.5];
        assert!(fit(&y, &trend_design(6), 2, Gaps::Exact).is_none());
        let x = [1.0; 12];
        assert!(fit(&[1.0, 2.0, 1.5, 3.0, 2.5, 4.0], &x, 2, Gaps::Exact).is_none());
    }

    #[test]
    fn gaps_of_one_date_are_the_series_without_gaps() {
        // both treatments agree when nothing is missing
        let y: Vec<f64> = (0..40)
            .map(|k| 0.6 + 0.003 * k as f64 + 0.05 * ((k * 37 % 11) as f64 / 11.0 - 0.5))
            .collect();
        let a = fit(&y, &trend_design(40), 2, Gaps::Exact).unwrap();
        let b = fit(&y, &trend_design(40), 2, Gaps::Ignore).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn batch_matches_single_series_on_any_threads() {
        let n_time = 30;
        let values: Vec<f64> = (0..200 * n_time)
            .map(|k| {
                if k % 17 == 3 {
                    f64::NAN
                } else {
                    ((k * 7919) % 101) as f64 / 10.0 + (k % n_time) as f64 * 0.05
                }
            })
            .collect();
        let x = trend_design(n_time);
        for n_jobs in [1, 4] {
            let out = batch(&values, 200, n_time, &x, 2, Gaps::Exact, 5, n_jobs);
            for p in 0..200 {
                let f = fit(&values[p * n_time..(p + 1) * n_time], &x, 2, Gaps::Exact).unwrap();
                assert_eq!(out[p].to_bits(), f.coefficients[0].to_bits());
                assert_eq!(out[200 + p].to_bits(), f.coefficients[1].to_bits());
                assert_eq!(out[7 * 200 + p].to_bits(), f.p[1].to_bits());
                assert_eq!(out[8 * 200 + p].to_bits(), f.rho.to_bits());
            }
        }
    }
}
