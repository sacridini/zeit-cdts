//! Weighted, bounded least-squares fit of the seasonal curves, and the
//! two-pass fit with a weight update between the passes. Implemented from
//! the specification of the problem: a Levenberg-Marquardt (own, small, no
//! allocation per iteration) on the residuals `sqrt(w) (y - f)`, the box bounds by
//! the transformation `p = lo + (hi - lo) sin^2(theta)` (Zeit's choice), the
//! Jacobian analytic.

use super::curves::{
    evaluate_curve, gradient, guess_and_bounds, value, CurveType, Guess, MAX_PARAMS,
};
use super::smooth::wtsm;

/// Suggested `max_fev` for one start of the fit of `kind`: `200 (npar + 1)`.
pub fn default_max_fev(kind: CurveType) -> usize {
    200 * (kind.npar() + 1)
}

struct Problem {
    kind: CurveType,
    tau: Vec<f64>,
    y: Vec<f64>,
    sw: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Indices of the parameters that are fitted (the others sit at `lower`).
    free: Vec<usize>,
}

impl Problem {
    fn to_params(&self, theta: &[f64], p: &mut [f64]) {
        p.copy_from_slice(&self.lower);
        for (j, &k) in self.free.iter().enumerate() {
            let s = theta[j].sin();
            p[k] = self.lower[k] + (self.upper[k] - self.lower[k]) * s * s;
        }
    }

    fn to_theta(&self, p: &[f64]) -> Vec<f64> {
        self.free
            .iter()
            .map(|&k| {
                let q = ((p[k] - self.lower[k]) / (self.upper[k] - self.lower[k])).clamp(0.0, 1.0);
                q.sqrt().asin()
            })
            .collect()
    }
}

/// Largest number of free parameters.
const MAXN: usize = MAX_PARAMS;

impl Problem {
    /// The weighted residuals `sqrt(w) (y - f)` at `theta` into `r`.
    fn residuals(&self, theta: &[f64], r: &mut [f64]) {
        let mut p = [0.0; MAX_PARAMS];
        let p = &mut p[..self.lower.len()];
        self.to_params(theta, p);
        for (i, out) in r.iter_mut().enumerate() {
            *out = self.sw[i] * (self.y[i] - value(self.kind, p, self.tau[i]));
        }
    }

    /// The residuals' Jacobian in `theta` (row-major, `m` rows of `n`), analytic.
    fn jacobian(&self, theta: &[f64], jac: &mut [f64]) {
        let n = self.free.len();
        let mut p = [0.0; MAX_PARAMS];
        let p = &mut p[..self.lower.len()];
        self.to_params(theta, p);
        let mut g = [0.0; MAX_PARAMS];
        let mut scale = [0.0; MAXN];
        for (j, &k) in self.free.iter().enumerate() {
            scale[j] = (self.upper[k] - self.lower[k]) * (2.0 * theta[j]).sin();
        }
        for i in 0..self.tau.len() {
            gradient(self.kind, p, self.tau[i], &mut g);
            let row = &mut jac[i * n..(i + 1) * n];
            for (j, &k) in self.free.iter().enumerate() {
                row[j] = -self.sw[i] * g[k] * scale[j];
            }
        }
    }
}

/// Solves `a x = b` for the symmetric positive definite `a` (`n x n`,
/// row-major, destroyed) by Cholesky; `false` if it is not positive definite.
fn cholesky_solve(a: &mut [f64; MAXN * MAXN], b: &mut [f64; MAXN], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if !(d > 0.0) || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / d;
        }
    }
    for i in 0..n {
        let mut v = b[i];
        for k in 0..i {
            v -= a[i * n + k] * b[k];
        }
        b[i] = v / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut v = b[i];
        for k in i + 1..n {
            v -= a[k * n + i] * b[k];
        }
        b[i] = v / a[i * n + i];
    }
    true
}

/// Relative tolerance of the decrease of the SSE and of the step (Zeit's
/// choice: the 1e-10 of MINPACK's usual setting only makes the iterations
/// crawl along the flat directions of the longer curves for nothing; 1e-6
/// moves the parameters by far less than the data can tell).
const FTOL: f64 = 1e-6;
const XTOL: f64 = 1e-6;

/// A start whose converged, sane fit has a weighted RMS residual of at most
/// this fraction of the observed amplitude ends the search for starts (Zeit's
/// choice: the other starts only matter when the first one lands in a poor
/// local minimum, and then the residual is large).
const GOOD_RMS: f64 = 0.25;

/// Levenberg-Marquardt (Marquardt's scaling, Nielsen's update of the damping)
/// on the residuals of `pb`, from `theta`, in place. `max_fev` is the budget
/// in the usual accounting of MINPACK: a residual is one evaluation, a
/// Jacobian `n + 1`. Returns whether it stopped by tolerance (not by the
/// budget) and the final SSE.
fn levenberg_marquardt(pb: &Problem, theta: &mut [f64], max_fev: usize) -> (bool, f64) {
    let (m, n) = (pb.tau.len(), theta.len());
    let mut r = vec![0.0; m];
    let mut r_new = vec![0.0; m];
    let mut jac = vec![0.0; m * n];
    pb.residuals(theta, &mut r);
    let mut f: f64 = r.iter().map(|x| x * x).sum();
    let mut nfev = 1;
    if !f.is_finite() {
        return (false, f);
    }
    let mut a = [0.0; MAXN * MAXN];
    let mut g = [0.0; MAXN];
    let mut mu = 0.0;
    let mut nu = 2.0;
    let mut need_jac = true;
    let mut trial = [0.0; MAXN];
    loop {
        if need_jac {
            pb.jacobian(theta, &mut jac);
            nfev += n + 1;
            a[..n * n].fill(0.0);
            g[..n].fill(0.0);
            for i in 0..m {
                let row = &jac[i * n..(i + 1) * n];
                for j in 0..n {
                    g[j] += row[j] * r[i];
                    for k in 0..=j {
                        a[j * n + k] += row[j] * row[k];
                    }
                }
            }
            for j in 0..n {
                for k in j + 1..n {
                    a[j * n + k] = a[k * n + j];
                }
            }
            if mu == 0.0 {
                mu = 1e-3 * (0..n).map(|j| a[j * n + j]).fold(0.0, f64::max);
            }
            need_jac = false;
            if f <= 1e-30 || (0..n).all(|j| g[j].abs() <= 1e-12 * f.sqrt()) {
                return (true, f);
            }
        }
        if nfev >= max_fev {
            return (false, f);
        }
        // (A + mu D) h = -g, D the diagonal of A (Marquardt)
        let mut system = [0.0; MAXN * MAXN];
        let mut h = [0.0; MAXN];
        system[..n * n].copy_from_slice(&a[..n * n]);
        let dmax = (0..n).map(|j| a[j * n + j]).fold(0.0, f64::max);
        for j in 0..n {
            system[j * n + j] += mu * a[j * n + j].max(1e-12 * dmax).max(1e-300);
            h[j] = -g[j];
        }
        if !cholesky_solve(&mut system, &mut h, n) {
            mu = if mu > 0.0 { mu * nu } else { 1e-3 };
            nu *= 2.0;
            if mu > 1e18 {
                return (true, f);
            }
            continue;
        }
        let (hn, tn) = (
            h[..n].iter().map(|x| x * x).sum::<f64>().sqrt(),
            theta.iter().map(|x| x * x).sum::<f64>().sqrt(),
        );
        if hn <= XTOL * (tn + XTOL) {
            return (true, f);
        }
        for j in 0..n {
            trial[j] = theta[j] + h[j];
        }
        pb.residuals(&trial[..n], &mut r_new);
        nfev += 1;
        let f_new: f64 = r_new.iter().map(|x| x * x).sum();
        // predicted decrease of the model, 0.5 h' (mu D h - g), against the actual one
        let mut pred = 0.0;
        for j in 0..n {
            pred += h[j] * (mu * a[j * n + j].max(1e-12 * dmax).max(1e-300) * h[j] - g[j]);
        }
        let rho = if f_new.is_finite() && pred > 0.0 { (f - f_new) / pred } else { -1.0 };
        if rho > 0.0 {
            let reduction = f - f_new;
            theta.copy_from_slice(&trial[..n]);
            std::mem::swap(&mut r, &mut r_new);
            f = f_new;
            mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
            nu = 2.0;
            need_jac = true;
            if reduction <= FTOL * f && rho > 0.25 {
                return (true, f);
            }
        } else {
            mu *= nu;
            nu *= 2.0;
            if mu > 1e18 {
                return (true, f);
            }
        }
    }
}

/// One Levenberg-Marquardt run from `p0`: the parameters, the weighted SSE and
/// whether it converged; `None` if the result is not finite.
fn fit_start(pb: &Problem, p0: &[f64], max_fev: usize) -> Option<(Vec<f64>, f64, bool)> {
    let mut theta = pb.to_theta(p0);
    let (converged, sse) = levenberg_marquardt(pb, &mut theta, max_fev.max(1));
    let mut p = vec![0.0; pb.lower.len()];
    pb.to_params(&theta, &mut p);
    if !sse.is_finite() || p.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some((p, sse, converged))
}

/// The sanity checks after a fit (section 2): finite parameters, ordered
/// transitions, a positive plateau for Elmore and Klosterman, and a fitted
/// amplitude of at least a fifth of the observed one.
fn is_sane(kind: CurveType, p: &[f64], amplitude: f64, span: f64) -> bool {
    if p.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let ordered = match kind {
        CurveType::Beck | CurveType::Dl | CurveType::Zhang | CurveType::Elmore => p[2] < p[4],
        CurveType::Gu => p[3] < p[5],
        CurveType::Klos => p[7] < p[8],
        CurveType::Ag => p[0] >= 0.0 && p[0] <= span,
    };
    if !ordered {
        return false;
    }
    let n = span.floor() as usize;
    let grid = (0..=n).map(|j| j as f64).chain(std::iter::once(span));
    let (mut fmin, mut fmax) = (f64::INFINITY, f64::NEG_INFINITY);
    for tau in grid {
        let f = value(kind, p, tau);
        if !f.is_finite() {
            return false;
        }
        fmin = fmin.min(f);
        fmax = fmax.max(f);
        match kind {
            CurveType::Elmore if p[1] - p[6] * tau <= 0.0 => return false,
            CurveType::Klos if (p[1] * tau + p[3]) * tau + p[4] <= 0.0 => return false,
            _ => {}
        }
    }
    fmax - fmin >= 0.2 * amplitude
}

/// Fits `kind` to the points with `w > 0` and finite `y` by weighted least
/// squares within the bounds of `guess_and_bounds`. Times are absolute; the
/// parameters come out with origin `t[0]` (positions in days since `t[0]`).
///
/// If `params` is empty or has a NaN the guess of section 1.8 is computed and
/// the starts are tried in order, keeping the lowest SSE (the first on a tie),
/// and stopping at the first that converges to a sane curve close to the data
/// (`GOOD_RMS`); otherwise `params` is the only start. `max_fev` is the budget
/// of function evaluations of one start (see `default_max_fev`). A start that
/// spends its budget still counts, if its curve is sane: the curves with many
/// parameters creep along flat directions long after the fit is useful.
/// Returns `false` (and leaves `params` as it was) if the segment cannot be
/// fitted or no start gives a sane curve; the caller then writes NaN.
pub fn fit_curve(t: &[f64], y: &[f64], w: &[f64], params: &mut Vec<f64>, kind: CurveType, max_fev: usize) -> bool {
    let n = t.len();
    if n == 0 || y.len() != n || w.len() != n {
        return false;
    }
    let origin = t[0];
    let tau_all: Vec<f64> = t.iter().map(|&x| x - origin).collect();
    let Some(Guess { starts, lower, upper, amplitude, span }) = guess_and_bounds(kind, &tau_all, y, w) else {
        return false;
    };
    let np = kind.npar();
    let valid: Vec<usize> = (0..n).filter(|&i| w[i] > 0.0 && y[i].is_finite()).collect();
    let mut free = Vec::with_capacity(np);
    for k in 0..np {
        if upper[k] > lower[k] {
            free.push(k);
        }
    }
    let pb = Problem {
        kind,
        tau: valid.iter().map(|&i| tau_all[i]).collect(),
        y: valid.iter().map(|&i| y[i]).collect(),
        sw: valid.iter().map(|&i| w[i].sqrt()).collect(),
        lower: lower.clone(),
        upper: upper.clone(),
        free,
    };
    if pb.tau.len() < pb.free.len() {
        return false;
    }
    let given = params.len() == np && params.iter().all(|v| v.is_finite());
    let starts: Vec<Vec<f64>> = if given {
        let mut s = params.clone();
        for k in 0..np {
            s[k] = s[k].clamp(lower[k], upper[k]);
        }
        vec![s]
    } else {
        starts
    };
    // Starts are tried in order; once one gives a converged, sane curve that
    // follows the data closely the others are not needed.
    let mut fits: Vec<(Vec<f64>, f64)> = Vec::with_capacity(starts.len());
    let wsum: f64 = pb.sw.iter().map(|x| x * x).sum();
    for s in &starts {
        let Some((p, sse, converged)) = fit_start(&pb, s, max_fev) else { continue };
        let good = converged && is_sane(kind, &p, amplitude, span) && sse <= wsum * (GOOD_RMS * amplitude).powi(2);
        fits.push((p, sse));
        if good {
            break;
        }
    }
    // stable sort: the first start wins a tie
    fits.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (p, _) in fits {
        if is_sane(kind, &p, amplitude, span) {
            *params = p;
            return true;
        }
    }
    false
}

/// Result of `fit_two_pass`.
#[derive(Clone, Debug)]
pub struct TwoPass {
    /// The weights of the pass that produced the final parameters (`w1` when
    /// the second pass succeeded, the input weights otherwise).
    pub weights: Vec<f64>,
    pub second_pass_ok: bool,
}

/// The fit in two passes: fit with `w`, update the weights from the fitted
/// curve with `wtsm` (iteration 1, `nptperyear`, `wfact`), fit again from the
/// first result. If the second pass fails the first result stays. `None` if
/// the first pass fails. `params` has the result, origin `t[0]`.
#[allow(clippy::too_many_arguments)]
pub fn fit_two_pass(
    t: &[f64],
    y: &[f64],
    w: &[f64],
    params: &mut Vec<f64>,
    kind: CurveType,
    max_fev: usize,
    nptperyear: usize,
    wfact: f64,
) -> Option<TwoPass> {
    let mut p1 = params.clone();
    if !fit_curve(t, y, w, &mut p1, kind, max_fev) {
        return None;
    }
    let tau: Vec<f64> = t.iter().map(|&x| x - t[0]).collect();
    let yfit1 = evaluate_curve(kind, &p1, &tau);
    let w1 = wtsm(y, &yfit1, w, 1, nptperyear, wfact);
    let mut p2 = p1.clone();
    if w1.len() == t.len() && fit_curve(t, y, &w1, &mut p2, kind, max_fev) {
        *params = p2;
        Some(TwoPass { weights: w1, second_pass_ok: true })
    } else {
        *params = p1;
        Some(TwoPass { weights: w.to_vec(), second_pass_ok: false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenology::curves::{evaluate_curve_at, guess_and_bounds};
    use crate::phenology::metrics::compute_gof;
    use crate::phenology::testutil::Rng;

    fn times(step: usize) -> Vec<f64> {
        (0..).map(|i| 1.0 + (i * step) as f64).take_while(|&t| t <= 365.0).collect()
    }

    const BECK_P: [f64; 6] = [0.1, 0.7, 50.0, 0.1, 250.0, 0.1];

    /// Truth parameters, positions in days since the first time of the series.
    fn truth(kind: CurveType) -> Vec<f64> {
        match kind {
            CurveType::Beck | CurveType::Dl | CurveType::Zhang => BECK_P.to_vec(),
            CurveType::Elmore => vec![0.1, 0.6, 50.0, 0.1, 250.0, 0.1, 0.0003],
            CurveType::Gu => vec![0.1, 0.6, 0.6, 50.0, 0.1, 250.0, 0.1, 0.7, 1.4],
            CurveType::Ag => vec![150.0, 0.1, 0.7, 90.0, 2.0, 110.0, 3.0],
            CurveType::Klos => vec![0.0, 0.0, 0.1, 0.0, 0.6, 0.1, 0.1, 50.0, 250.0, 1.0, 1.0],
        }
    }

    fn synth(kind: CurveType, step: usize) -> (Vec<f64>, Vec<f64>) {
        let t = times(step);
        let y = evaluate_curve_at(kind, &truth(kind), &t, t[0]);
        (t, y)
    }

    fn rmse_of(kind: CurveType, t: &[f64], y: &[f64], p: &[f64]) -> f64 {
        let yf = evaluate_curve_at(kind, p, t, t[0]);
        compute_gof(y, &yf, &vec![1.0; t.len()]).1
    }

    #[test]
    fn noiseless_recovery_every_curve() {
        for kind in [CurveType::Beck, CurveType::Dl, CurveType::Elmore, CurveType::Gu, CurveType::Zhang, CurveType::Ag, CurveType::Klos] {
            for step in [8, 16, 1] {
                let (t, y) = synth(kind, step);
                let w = vec![1.0; t.len()];
                let mut p = Vec::new();
                assert!(fit_curve(&t, &y, &w, &mut p, kind, default_max_fev(kind)), "{kind:?} step {step} failed");
                let rmse = rmse_of(kind, &t, &y, &p);
                let amp = 0.6;
                match kind {
                    CurveType::Klos => assert!(rmse < 1e-3 * amp, "{kind:?} step {step}: rmse {rmse}"),
                    _ => assert!(rmse < 1e-4 * amp, "{kind:?} step {step}: rmse {rmse}"),
                }
                if kind == CurveType::Beck || kind == CurveType::Dl || kind == CurveType::Zhang {
                    let want = BECK_P;
                    for k in 0..6 {
                        assert!(
                            (p[k] - want[k]).abs() <= 1e-4 * want[k].abs().max(1.0),
                            "{kind:?} step {step} param {k}: {} vs {}",
                            p[k],
                            want[k]
                        );
                    }
                }
            }
        }
    }

    /// Klosterman's curve with the 50% crossings of its two sigmoids at
    /// `sos` and `eos` (plateau 0.6 over a base of 0.2), sampled every 16
    /// days for a year.
    fn klos_series(sos: f64, eos: f64, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let t: Vec<f64> = (0..23).map(|i| 1.0 + 16.0 * i as f64).collect();
        let p = [0.0, 0.0, 0.2, 0.0, 0.6, 0.08, 0.06, sos, eos, 1.0, 1.0];
        let mut rng = Rng::new(seed);
        let y = evaluate_curve_at(CurveType::Klos, &p, &t, t[0]).iter().map(|v| v + noise * rng.normal()).collect();
        (t, y)
    }

    /// Where the fitted curve crosses 50% of its range on the way up, in days
    /// since `t[0]`.
    fn rise_crossing(kind: CurveType, p: &[f64], span: f64) -> f64 {
        let tau: Vec<f64> = (0..=span as usize).map(|d| d as f64).collect();
        let f = evaluate_curve(kind, p, &tau);
        let (lo, hi) = (f.iter().cloned().fold(f64::INFINITY, f64::min), f.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        let level = lo + 0.5 * (hi - lo);
        let j = (0..f.len() - 1).find(|&j| f[j] < level && f[j + 1] >= level).expect("a rise");
        tau[j] + (level - f[j]) / (f[j + 1] - f[j])
    }

    #[test]
    fn klosterman_fits_its_own_clean_and_slightly_noisy_series() {
        let kind = CurveType::Klos;
        for (sos, eos) in [(120.0, 280.0), (60.0, 200.0), (150.0, 330.0)] {
            for (noise, seed) in [(0.0, 1), (0.005, 2), (0.01, 3)] {
                let (t, y) = klos_series(sos, eos, noise, seed);
                let w = vec![1.0; t.len()];
                let mut p = Vec::new();
                let r = fit_two_pass(&t, &y, &w, &mut p, kind, default_max_fev(kind), 23, 0.5);
                assert!(r.is_some(), "sos {sos} eos {eos} noise {noise}: no fit");
                // the 50% crossings of the sigmoids sit at n1 and n2 (up to the plateau's slope)
                let sos_fit = rise_crossing(kind, &p, t[t.len() - 1] - t[0]) + t[0];
                assert!((sos_fit - (sos + 1.0)).abs() < 5.0, "sos {sos} eos {eos} noise {noise}: rise at {sos_fit}");
            }
        }
    }

    #[test]
    fn noiseless_recovery_from_every_automatic_start() {
        let kind = CurveType::Beck;
        let (t, y) = synth(kind, 8);
        let w = vec![1.0; t.len()];
        let tau: Vec<f64> = t.iter().map(|x| x - t[0]).collect();
        let g = guess_and_bounds(kind, &tau, &y, &w).unwrap();
        assert_eq!(g.starts.len(), 4);
        for (i, s) in g.starts.iter().enumerate() {
            let mut p = s.clone();
            assert!(fit_curve(&t, &y, &w, &mut p, kind, default_max_fev(kind)), "start {i}");
            let want = BECK_P;
            for k in 0..6 {
                assert!((p[k] - want[k]).abs() <= 1e-4 * want[k].abs().max(1.0), "start {i} param {k}: {} vs {}", p[k], want[k]);
            }
        }
    }

    #[test]
    fn one_curve_fits_data_of_another() {
        let flex = [CurveType::Beck, CurveType::Gu, CurveType::Zhang, CurveType::Elmore, CurveType::Dl];
        for from in [CurveType::Gu, CurveType::Beck, CurveType::Elmore] {
            let (t, y) = synth(from, 8);
            let w = vec![1.0; t.len()];
            for kind in flex {
                let mut p = Vec::new();
                assert!(fit_curve(&t, &y, &w, &mut p, kind, default_max_fev(kind)), "{from:?} -> {kind:?}");
                let rmse = rmse_of(kind, &t, &y, &p);
                assert!(rmse < 0.02 * 0.6, "{from:?} -> {kind:?}: rmse {rmse}");
            }
        }
    }

    #[test]
    fn noisy_recovery_beck_200_repetitions() {
        let kind = CurveType::Beck;
        let t = times(8);
        let clean = evaluate_curve_at(kind, &BECK_P, &t, t[0]);
        let mut rng = Rng::new(2024);
        let n_rep = 200;
        let (mut s_sos, mut s_eos, mut s_rsp, mut s_mx, mut s_mn) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut fails = 0;
        let w = vec![1.0; t.len()];
        for _ in 0..n_rep {
            let y: Vec<f64> = clean.iter().map(|&c| c + 0.02 * rng.normal()).collect();
            let mut p = Vec::new();
            if !fit_curve(&t, &y, &w, &mut p, kind, default_max_fev(kind)) {
                fails += 1;
                continue;
            }
            s_sos += p[2];
            s_eos += p[4];
            s_rsp += p[3];
            s_mx += p[1];
            s_mn += p[0];
        }
        assert_eq!(fails, 0, "fits failed");
        let m = n_rep as f64;
        assert!((s_sos / m - 50.0).abs() < 2.0, "sos {}", s_sos / m);
        assert!((s_eos / m - 250.0).abs() < 2.0, "eos {}", s_eos / m);
        assert!((s_rsp / m - 0.1).abs() < 0.015, "rsp {}", s_rsp / m);
        assert!((s_mx / m - 0.7).abs() < 0.02, "mx {}", s_mx / m);
        assert!((s_mn / m - 0.1).abs() < 0.02, "mn {}", s_mn / m);
    }

    #[test]
    fn zero_weights_and_outliers_exercise_the_second_pass() {
        let kind = CurveType::Beck;
        let t = times(8);
        let clean = evaluate_curve_at(kind, &BECK_P, &t, t[0]);
        let mut rng = Rng::new(5);
        let mut y: Vec<f64> = clean.iter().map(|&c| c + 0.01 * rng.normal()).collect();
        let mut w = vec![1.0; t.len()];
        for i in (0..t.len()).step_by(10) {
            w[i] = 0.0;
            y[i] = f64::NAN;
        }
        // clouds: a few points far below the curve
        for i in [9usize, 20, 31] {
            y[i] -= 0.4;
        }
        let mut p1 = Vec::new();
        assert!(fit_curve(&t, &y, &w, &mut p1, kind, default_max_fev(kind)));
        let mut p = Vec::new();
        let r = fit_two_pass(&t, &y, &w, &mut p, kind, default_max_fev(kind), 46, 0.5).expect("two pass");
        assert!(r.second_pass_ok);
        assert_eq!(r.weights.len(), t.len());
        assert!(r.weights.iter().all(|&x| x >= 0.0));
        // the pass that down-weights the clouds is closer to the clean curve
        let err = |q: &[f64]| {
            let f = evaluate_curve_at(kind, q, &t, t[0]);
            (0..t.len()).map(|i| (f[i] - clean[i]).powi(2)).sum::<f64>()
        };
        assert!(err(&p) <= err(&p1) + 1e-12, "{} vs {}", err(&p), err(&p1));
    }

    #[test]
    fn failures_are_reported() {
        let t = times(8);
        let w = vec![1.0; t.len()];
        let kind = CurveType::Beck;
        let mut p = Vec::new();
        assert!(!fit_curve(&t, &vec![0.5; t.len()], &w, &mut p, kind, 1000), "constant");
        let y = evaluate_curve_at(kind, &BECK_P, &t, t[0]);
        assert!(!fit_curve(&t, &y, &vec![0.0; t.len()], &mut p, kind, 1000), "all weights zero");
        let mut w4 = vec![0.0; t.len()];
        for k in 0..4 {
            w4[k * 11] = 1.0;
        }
        assert!(!fit_curve(&t, &y, &w4, &mut p, kind, 1000), "4 valid points");
        let mut p2 = Vec::new();
        assert!(!fit_curve(&t, &y, &w[..3], &mut p2, kind, 1000), "length mismatch");
    }
}
