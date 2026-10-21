//! Eigen 3.4's unsupported LevenbergMarquardt (the MINPACK lmdif port of
//! unsupported/Eigen/src/NonLinearOptimization: minimize, lmpar2, qrsolv) with
//! NumericalDiff's forward differences, and the norms it takes (stableNorm,
//! blueNorm), with the same operations in the same order as the C++ build.

// The loops index rows and columns as Eigen's do.
#![allow(clippy::needless_range_loop)]

use super::qr::{gemv_col, gemv_row, solve_upper, ColPivQr, Mat, Order};
use super::{max_by, std_max, std_min, sum_by};

/// `stableNorm()` of a vector (Eigen/src/Core/StableNorm.h): one block of at
/// most 4096 coefficients scaled by its largest absolute value.
pub fn stable_norm(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 1 {
        return v[0].abs();
    }
    let (mut scale, mut inv_scale, mut ssq) = (0.0f64, 1.0f64, 0.0f64);
    for block in v.chunks(4096) {
        let max = max_by(block.len(), |i| block[i].abs());
        if max > scale {
            ssq *= {
                let r = scale / max;
                r * r
            };
            let tmp = 1.0 / max;
            if tmp > f64::MAX {
                inv_scale = f64::MAX;
                scale = 1.0 / inv_scale;
            } else if max > f64::MAX {
                inv_scale = 1.0;
                scale = max;
            } else {
                scale = max;
                inv_scale = tmp;
            }
        } else if max.is_nan() {
            scale = max;
        }
        if scale > 0.0 {
            ssq += sum_by(block.len(), |i| {
                let x = block[i] * inv_scale;
                x * x
            });
        }
    }
    scale * ssq.sqrt()
}

/// `blueNorm()` (Blue's algorithm, one coefficient after the other).
pub fn blue_norm(v: &[f64]) -> f64 {
    // radix 2, 53 digits, exponents -1021 to 1024: exact powers of 2
    let b1 = 2f64.powi(-511);
    let b2 = 2f64.powi(486);
    let s1m = 2f64.powi(511);
    let s2m = 2f64.powi(-538);
    let relerr = 2f64.powi(-26);
    let ab2 = b2 / v.len() as f64;
    let (mut asml, mut amed, mut abig) = (0.0f64, 0.0f64, 0.0f64);
    for &x in v {
        let ax = x.abs();
        if ax > ab2 {
            let s = ax * s2m;
            abig += s * s;
        } else if ax < b1 {
            let s = ax * s1m;
            asml += s * s;
        } else {
            amed += ax * ax;
        }
    }
    if amed.is_nan() {
        return amed;
    }
    if abig > 0.0 {
        abig = abig.sqrt();
        if abig > f64::MAX {
            return abig;
        }
        if amed > 0.0 {
            abig /= s2m;
            amed = amed.sqrt();
        } else {
            return abig / s2m;
        }
    } else if asml > 0.0 {
        if amed > 0.0 {
            abig = amed.sqrt();
            amed = asml.sqrt() / s1m;
        } else {
            return asml.sqrt() / s1m;
        }
    } else {
        return amed.sqrt();
    }
    asml = std_min(abig, amed);
    abig = std_max(abig, amed);
    if asml <= abig * relerr {
        abig
    } else {
        let r = asml / abig;
        abig * (1.0 + r * r).sqrt()
    }
}

/// LevenbergMarquardtSpace::Status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    ImproperInputParameters = 0,
    RelativeReductionTooSmall = 1,
    RelativeErrorTooSmall = 2,
    RelativeErrorAndReductionTooSmall = 3,
    CosinusTooSmall = 4,
    TooManyFunctionEvaluation = 5,
    FtolTooSmall = 6,
    XtolTooSmall = 7,
    GtolTooSmall = 8,
}

/// The residuals to minimize: `values()` of them at `x`.
pub trait Functor {
    fn values(&self) -> usize;
    fn eval(&self, x: &[f64], fvec: &mut [f64]);

    /// The forward differences of NumericalDiff at `x`, where the residuals
    /// are `fvec`, into `jac` (column-major). A functor may compute them its
    /// own way, to the same bits.
    fn jacobian(&self, x: &[f64], fvec: &[f64], eps: f64, jac: &mut [f64]) {
        forward_differences(self, x, fvec, eps, jac);
    }
}

/// NumericalDiff::df (forward): column j is (f(x + h e_j) - f(x)) / h, h =
/// eps |x_j| (eps when 0). f(x) is `fvec`: NumericalDiff evaluates it again,
/// to the same values.
pub fn forward_differences<F: Functor + ?Sized>(f: &F, x0: &[f64], fvec: &[f64], eps: f64, jac: &mut [f64]) {
    let m = fvec.len();
    let mut x = x0.to_vec();
    let mut val2 = vec![0.0; m];
    for j in 0..x0.len() {
        let mut h = eps * x[j].abs();
        if h == 0.0 {
            h = eps;
        }
        x[j] += h;
        f.eval(&x, &mut val2);
        x[j] = x0[j];
        for (c, (&b, &a)) in jac[j * m..(j + 1) * m].iter_mut().zip(val2.iter().zip(fvec)) {
            *c = (b - a) / h;
        }
    }
}

/// LevenbergMarquardt::Parameters.
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub factor: f64,
    pub maxfev: usize,
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    pub epsfcn: f64,
}

impl Default for Parameters {
    fn default() -> Parameters {
        let sqrt_eps = f64::EPSILON.sqrt();
        Parameters {
            factor: 100.0,
            maxfev: 400,
            ftol: sqrt_eps,
            xtol: sqrt_eps,
            gtol: 0.0,
            epsfcn: 0.0,
        }
    }
}

/// `LevenbergMarquardt<NumericalDiff<F>>(f).minimize(x)` with `params`: `x`
/// moved to the minimum found.
pub fn minimize<F: Functor>(f: &F, x: &mut [f64], params: &Parameters) -> Status {
    let n = x.len();
    let m = f.values();
    if n == 0
        || m < n
        || params.ftol < 0.0
        || params.xtol < 0.0
        || params.gtol < 0.0
        || params.maxfev == 0
        || params.factor <= 0.0
    {
        return Status::ImproperInputParameters;
    }
    let mut lm = Lm {
        n,
        m,
        fvec: vec![0.0; m],
        diag: vec![0.0; n],
        nfev: 1,
        iter: 1,
        fnorm: 0.0,
        par: 0.0,
        delta: 0.0,
        xnorm: 0.0,
        temp: 0.0,
        wa4: vec![0.0; m],
    };
    f.eval(x, &mut lm.fvec);
    lm.fnorm = stable_norm(&lm.fvec);
    loop {
        if let Some(status) = lm.step(f, x, params) {
            return status;
        }
    }
}

struct Lm {
    n: usize,
    m: usize,
    fvec: Vec<f64>,
    diag: Vec<f64>,
    nfev: usize,
    iter: usize,
    fnorm: f64,
    par: f64,
    delta: f64,
    xnorm: f64,
    // a member of the C++ class: kept from one step to the next
    temp: f64,
    // scratch
    wa4: Vec<f64>,
}

impl Lm {
    /// NumericalDiff::df, forward differences: the Jacobian, column-major.
    fn jacobian<F: Functor>(&mut self, f: &F, x0: &[f64], params: &Parameters) -> Mat {
        let (n, m) = (self.n, self.m);
        let eps = std_max(params.epsfcn, f64::EPSILON).sqrt();
        let mut jac = vec![0.0; m * n];
        f.jacobian(x0, &self.fvec, eps, &mut jac);
        self.nfev += n + 1;
        Mat::from_data(m, n, Order::Col, jac)
    }

    /// minimizeOneStep: `Some(status)` when it stops.
    fn step<F: Functor>(&mut self, f: &F, x: &mut [f64], params: &Parameters) -> Option<Status> {
        let (n, m) = (self.n, self.m);
        let fjac = self.jacobian(f, x, params);

        let mut wa2: Vec<f64> = (0..n).map(|j| blue_norm(&fjac.data[j * m..(j + 1) * m])).collect();
        let qr = ColPivQr::new(fjac);
        let r = qr.matrix_qr();
        let perm = qr.perm();

        if self.iter == 1 {
            for j in 0..n {
                self.diag[j] = if wa2[j] == 0.0 { 1.0 } else { wa2[j] };
            }
            let dx: Vec<f64> = (0..n).map(|j| self.diag[j] * x[j]).collect();
            self.xnorm = stable_norm(&dx);
            self.delta = params.factor * self.xnorm;
            if self.delta == 0.0 {
                self.delta = params.factor;
            }
        }

        self.wa4.copy_from_slice(&self.fvec);
        qr.apply_qt(&mut self.wa4);
        let qtf = self.wa4[..n].to_vec();

        let mut gnorm = 0.0;
        if self.fnorm != 0.0 {
            for j in 0..n {
                let l = perm[j];
                if wa2[l] != 0.0 {
                    let col = &r.data[j * m..j * m + j + 1];
                    let fnorm = self.fnorm;
                    let s = sum_by(j + 1, |i| col[i] * (qtf[i] / fnorm));
                    gnorm = std_max(gnorm, (s / wa2[l]).abs());
                }
            }
        }
        if gnorm <= params.gtol {
            return Some(Status::CosinusTooSmall);
        }

        cwise_max(&mut self.diag, &wa2);

        let mut wa1 = vec![0.0; n];
        let mut wa3 = vec![0.0; m];
        loop {
            lmpar2(&qr, &self.diag, &qtf, self.delta, &mut self.par, &mut wa1);

            for v in wa1.iter_mut() {
                *v = -*v;
            }
            for j in 0..n {
                wa2[j] = x[j] + wa1[j];
            }
            let dp: Vec<f64> = (0..n).map(|j| self.diag[j] * wa1[j]).collect();
            let pnorm = stable_norm(&dp);

            if self.iter == 1 {
                self.delta = std_min(self.delta, pnorm);
            }

            f.eval(&wa2, &mut self.wa4);
            self.nfev += 1;
            let fnorm1 = stable_norm(&self.wa4);

            let mut actred = -1.0;
            if 0.1 * fnorm1 < self.fnorm {
                let q = fnorm1 / self.fnorm;
                actred = 1.0 - q * q;
            }

            // R * (P^-1 wa1): the triangular product, R's rows below n zero
            let v: Vec<f64> = (0..n).map(|i| wa1[perm[i]]).collect();
            upper_mat_vec(r, n, &v, &mut wa3);
            let temp1 = {
                let q = stable_norm(&wa3) / self.fnorm;
                q * q
            };
            let temp2 = {
                let q = self.par.sqrt() * pnorm / self.fnorm;
                q * q
            };
            let prered = temp1 + temp2 / 0.5;
            let dirder = -(temp1 + temp2);

            let mut ratio = 0.0;
            if prered != 0.0 {
                ratio = actred / prered;
            }

            if ratio <= 0.25 {
                if actred >= 0.0 {
                    self.temp = 0.5;
                }
                if actred < 0.0 {
                    self.temp = 0.5 * dirder / (dirder + 0.5 * actred);
                }
                if 0.1 * fnorm1 >= self.fnorm || self.temp < 0.1 {
                    self.temp = 0.1;
                }
                self.delta = self.temp * std_min(self.delta, pnorm / 0.1);
                self.par /= self.temp;
            } else if !(self.par != 0.0 && ratio < 0.75) {
                self.delta = pnorm / 0.5;
                self.par *= 0.5;
            }

            if ratio >= 1e-4 {
                x.copy_from_slice(&wa2);
                for j in 0..n {
                    wa2[j] = self.diag[j] * x[j];
                }
                self.fvec.copy_from_slice(&self.wa4);
                self.xnorm = stable_norm(&wa2);
                self.fnorm = fnorm1;
                self.iter += 1;
            }

            let (ftol, xtol) = (params.ftol, params.xtol);
            if actred.abs() <= ftol && prered <= ftol && 0.5 * ratio <= 1.0 && self.delta <= xtol * self.xnorm {
                return Some(Status::RelativeErrorAndReductionTooSmall);
            }
            if actred.abs() <= ftol && prered <= ftol && 0.5 * ratio <= 1.0 {
                return Some(Status::RelativeReductionTooSmall);
            }
            if self.delta <= xtol * self.xnorm {
                return Some(Status::RelativeErrorTooSmall);
            }
            if self.nfev >= params.maxfev {
                return Some(Status::TooManyFunctionEvaluation);
            }
            let eps = f64::EPSILON;
            if actred.abs() <= eps && prered <= eps && 0.5 * ratio <= 1.0 {
                return Some(Status::FtolTooSmall);
            }
            if self.delta <= eps * self.xnorm {
                return Some(Status::XtolTooSmall);
            }
            if gnorm <= eps {
                return Some(Status::GtolTooSmall);
            }
            if !(ratio < 1e-4) {
                return None;
            }
        }
    }
}

/// `a = a.cwiseMax(b)`: packets for the even part (pmax), std::max for an odd
/// last coefficient.
fn cwise_max(a: &mut [f64], b: &[f64]) {
    let even = a.len() / 2 * 2;
    for i in 0..even {
        a[i] = super::pmax(a[i], b[i]);
    }
    for i in even..a.len() {
        a[i] = std_max(a[i], b[i]);
    }
}

/// `fjac.triangularView<Upper>() * v` into `res` (all `m` rows; those below
/// `n` stay 0), as Eigen's column-major triangular matrix-vector product:
/// panels of 8 columns, their triangle by axpys, the rows above by the
/// matrix-vector kernel.
fn upper_mat_vec(r: &Mat, n: usize, v: &[f64], res: &mut [f64]) {
    res.iter_mut().for_each(|c| *c = 0.0);
    const PANEL: usize = 8;
    let mut pi = 0;
    while pi < n {
        let width = PANEL.min(n - pi);
        for k in 0..width {
            let i = pi + k;
            let a = v[i];
            for t in 0..k + 1 {
                res[pi + t] += a * r.get(pi + t, i);
            }
        }
        if pi > 0 {
            for row in 0..pi {
                let c = gemv_col(row, pi, width, |j| r.get(row, pi + j), |j| v[pi + j]);
                res[row] += c;
            }
        }
        pi += width;
    }
}

/// The lower-triangular solve `A.triangularView<Lower>().solveInPlace(rhs)`
/// of a row-major `n` x `n` matrix `a(i, j)`: panels of 8 rows, the columns
/// before by the row-major matrix-vector kernel.
fn solve_lower_row_major(n: usize, a: impl Fn(usize, usize) -> f64, rhs: &mut [f64]) {
    const PANEL: usize = 8;
    let mut pi = 0;
    while pi < n {
        let width = (n - pi).min(PANEL);
        if pi > 0 {
            for row in pi..pi + width {
                let cc = gemv_row(pi, |j| a(row, j), |j| rhs[j]);
                rhs[row] += -cc;
            }
        }
        for k in 0..width {
            let i = pi + k;
            if k > 0 {
                rhs[i] -= sum_by(k, |t| a(i, pi + t) * rhs[pi + t]);
            }
            if rhs[i] != 0.0 {
                rhs[i] /= a(i, i);
            }
        }
        pi += width;
    }
}

/// The upper-triangular solve of a row-major `n` x `n` matrix `a(i, j)`.
fn solve_upper_row_major(n: usize, a: impl Fn(usize, usize) -> f64, rhs: &mut [f64]) {
    const PANEL: usize = 8;
    let mut pi = n;
    while pi > 0 {
        let width = pi.min(PANEL);
        let r = n - pi;
        let start = pi - width;
        if r > 0 {
            for row in start..pi {
                let cc = gemv_row(r, |j| a(row, pi + j), |j| rhs[pi + j]);
                rhs[row] += -cc;
            }
        }
        for k in 0..width {
            let i = pi - k - 1;
            let s = i + 1;
            if k > 0 {
                rhs[i] -= sum_by(k, |t| a(i, s + t) * rhs[s + t]);
            }
            if rhs[i] != 0.0 {
                rhs[i] /= a(i, i);
            }
        }
        pi -= width;
    }
}

/// JacobiRotation::makeGivens(p, q): (c, s).
fn make_givens(p: f64, q: f64) -> (f64, f64) {
    if q == 0.0 {
        (if p < 0.0 { -1.0 } else { 1.0 }, 0.0)
    } else if p == 0.0 {
        (0.0, if q < 0.0 { 1.0 } else { -1.0 })
    } else if p.abs() > q.abs() {
        let t = q / p;
        let mut u = (1.0 + t * t).sqrt();
        if p < 0.0 {
            u = -u;
        }
        let c = 1.0 / u;
        (c, -t * c)
    } else {
        let t = p / q;
        let mut u = (1.0 + t * t).sqrt();
        if q < 0.0 {
            u = -u;
        }
        let s = -1.0 / u;
        (-t * s, s)
    }
}

/// internal::qrsolv on `s` (m x n, column-major: R above the diagonal, its
/// strictly lower part the workspace).
fn qrsolv(s: &mut Mat, ipvt: &[usize], diag: &[f64], qtb: &[f64], x: &mut [f64], sdiag: &mut [f64]) {
    let n = s.cols;
    for j in 0..n {
        x[j] = s.get(j, j);
    }
    let mut wa = qtb.to_vec();
    for j in 0..n {
        for i in j + 1..n {
            let v = s.get(j, i);
            s.set(i, j, v);
        }
    }

    for j in 0..n {
        let l = ipvt[j];
        if diag[l] == 0.0 {
            break;
        }
        sdiag[j..].iter_mut().for_each(|v| *v = 0.0);
        sdiag[j] = diag[l];
        let mut qtbpj = 0.0;
        for k in j..n {
            let (c, sn) = make_givens(-s.get(k, k), sdiag[k]);
            let skk = c * s.get(k, k) + sn * sdiag[k];
            s.set(k, k, skk);
            let temp = c * wa[k] + sn * qtbpj;
            qtbpj = -sn * wa[k] + c * qtbpj;
            wa[k] = temp;
            for i in k + 1..n {
                let sik = s.get(i, k);
                let temp = c * sik + sn * sdiag[i];
                sdiag[i] = -sn * sik + c * sdiag[i];
                s.set(i, k, temp);
            }
        }
    }

    let mut nsing = 0;
    while nsing < n && sdiag[nsing] != 0.0 {
        nsing += 1;
    }
    wa[nsing..].iter_mut().for_each(|v| *v = 0.0);
    {
        let s: &Mat = s;
        solve_upper_row_major(nsing, |i, j| s.get(j, i), &mut wa[..nsing]);
    }

    for j in 0..n {
        sdiag[j] = s.get(j, j);
        s.set(j, j, x[j]);
    }
    for j in 0..n {
        x[ipvt[j]] = wa[j];
    }
}

/// internal::lmpar2: the Levenberg-Marquardt parameter `par` and the step `x`.
fn lmpar2(qr: &ColPivQr, diag: &[f64], qtb: &[f64], delta: f64, par: &mut f64, x: &mut [f64]) {
    let dwarf = f64::MIN_POSITIVE;
    let r = qr.matrix_qr();
    let perm = qr.perm();
    let n = r.cols;

    let rank = qr.rank();
    let mut wa1 = qtb.to_vec();
    wa1[rank..].iter_mut().for_each(|v| *v = 0.0);
    solve_upper(r, rank, &mut wa1[..rank]);
    for i in 0..n {
        x[perm[i]] = wa1[i];
    }

    let mut iter = 0;
    let mut wa2: Vec<f64> = (0..n).map(|j| diag[j] * x[j]).collect();
    let mut dxnorm = blue_norm(&wa2);
    let mut fp = dxnorm - delta;
    if fp <= 0.1 * delta {
        *par = 0.0;
        return;
    }

    let mut parl = 0.0;
    if rank == n {
        for i in 0..n {
            let l = perm[i];
            wa1[i] = diag[l] * wa2[l] / dxnorm;
        }
        solve_lower_row_major(n, |i, j| r.get(j, i), &mut wa1);
        let temp = blue_norm(&wa1);
        parl = fp / delta / temp / temp;
    }

    for j in 0..n {
        let col = &r.data[j * r.rows..j * r.rows + j + 1];
        wa1[j] = sum_by(j + 1, |i| col[i] * qtb[i]) / diag[perm[j]];
    }
    let gnorm = stable_norm(&wa1);
    let mut paru = gnorm / delta;
    if paru == 0.0 {
        paru = dwarf / std_min(delta, 0.1);
    }

    *par = std_max(*par, parl);
    *par = std_min(*par, paru);
    if *par == 0.0 {
        *par = gnorm / dxnorm;
    }

    let mut s = r.clone();
    let mut sdiag = vec![0.0; n];
    loop {
        iter += 1;
        if *par == 0.0 {
            *par = std_max(dwarf, 0.001 * paru);
        }
        let sp = par.sqrt();
        for j in 0..n {
            wa1[j] = sp * diag[j];
        }
        qrsolv(&mut s, perm, &wa1, qtb, x, &mut sdiag);

        for j in 0..n {
            wa2[j] = diag[j] * x[j];
        }
        dxnorm = blue_norm(&wa2);
        let temp = fp;
        fp = dxnorm - delta;
        if fp.abs() <= 0.1 * delta || (parl == 0.0 && fp <= temp && temp < 0.0) || iter == 10 {
            break;
        }

        for i in 0..n {
            let l = perm[i];
            wa1[i] = diag[l] * (wa2[l] / dxnorm);
        }
        for j in 0..n {
            wa1[j] /= sdiag[j];
            let temp = wa1[j];
            for i in j + 1..n {
                wa1[i] -= s.get(i, j) * temp;
            }
        }
        let temp = blue_norm(&wa1);
        let parc = fp / delta / temp / temp;

        if fp > 0.0 {
            parl = std_max(parl, *par);
        }
        if fp < 0.0 {
            paru = std_min(paru, *par);
        }
        *par = std_max(parl, *par + parc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// y = a exp(b t), noiseless: the fit recovers a and b.
    struct Exp {
        t: Vec<f64>,
        y: Vec<f64>,
    }

    impl Functor for Exp {
        fn values(&self) -> usize {
            self.t.len()
        }
        fn eval(&self, x: &[f64], fvec: &mut [f64]) {
            for i in 0..self.t.len() {
                fvec[i] = x[0] * (x[1] * self.t[i]).exp() - self.y[i];
            }
        }
    }

    #[test]
    fn fits_an_exponential() {
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = t.iter().map(|&t| 2.5 * (-1.3 * t).exp()).collect();
        let f = Exp { t, y };
        let mut x = vec![1.0, 0.0];
        let status = minimize(&f, &mut x, &Parameters::default());
        assert_ne!(status, Status::ImproperInputParameters);
        assert!(
            (x[0] - 2.5).abs() < 1e-6 && (x[1] + 1.3).abs() < 1e-6,
            "{x:?} {status:?}"
        );
    }

    #[test]
    fn norms() {
        let v = [3.0, 4.0, 0.0, 12.0];
        assert!((stable_norm(&v) - 13.0).abs() < 1e-14);
        assert!((blue_norm(&v) - 13.0).abs() < 1e-14);
        assert_eq!(stable_norm(&[-2.0]), 2.0);
        assert!(stable_norm(&[1.0, f64::NAN]).is_nan());
    }
}
