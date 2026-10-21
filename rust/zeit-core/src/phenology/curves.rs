//! The seasonal curves and their fit: a port of src/phenology_curves.cpp. The
//! parameters are fitted in an unbounded space (p = lb + (ub - lb)(sin u + 1)/2)
//! by Eigen's Levenberg-Marquardt with forward-difference Jacobians.
//!
//! Each curve is two terms (its exponentials and powers) and their
//! combination, in the C++'s operations. A column of the Jacobian moves one
//! parameter, which most terms do not depend on: those are kept from the
//! point the differences start from (the same values, computed once).

// The loops index the series as the C++'s do.
#![allow(clippy::needless_range_loop)]

use crate::eigen::lm::{self, Functor, Parameters, Status};
use crate::eigen::{max_coeff, min_coeff, std_max, std_min};

/// The curves, numbered as zeit._core.phenology.CurveType.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveType {
    Beck = 0,
    Elmore = 1,
    Gu = 2,
    Klos = 3,
    Zhang = 4,
    Ag = 5,
    Dl = 6,
}

impl CurveType {
    pub fn from_int(v: i64) -> Option<CurveType> {
        Some(match v {
            0 => CurveType::Beck,
            1 => CurveType::Elmore,
            2 => CurveType::Gu,
            3 => CurveType::Klos,
            4 => CurveType::Zhang,
            5 => CurveType::Ag,
            6 => CurveType::Dl,
            _ => return None,
        })
    }

    /// The number of parameters.
    pub fn n_params(self) -> usize {
        match self {
            CurveType::Beck | CurveType::Dl => 6,
            CurveType::Elmore | CurveType::Zhang | CurveType::Ag => 7,
            CurveType::Gu => 9,
            CurveType::Klos => 13,
        }
    }
}

#[inline]
fn transform_to_bounded(p: f64, lb: f64, ub: f64) -> f64 {
    lb + (ub - lb) * 0.5 * (p.sin() + 1.0)
}

#[inline]
fn transform_to_unbounded(bounded: f64, lb: f64, ub: f64) -> f64 {
    let mut val = 2.0 * (bounded - lb) / (ub - lb) - 1.0;
    val = std_max(-1.0, std_min(1.0, val));
    val.asin()
}

/// A curve as two terms and their combination: `predict` is the C++'s
/// predict_* operation for operation.
trait Curve {
    /// The terms each parameter enters: 0 neither, 1 the first, 2 the
    /// second, 3 both.
    const DEPS: &'static [u8];
    fn term1(t: f64, p: &[f64]) -> f64;
    fn term2(t: f64, p: &[f64]) -> f64;
    fn combine(t: f64, p: &[f64], a: f64, b: f64) -> f64;

    #[inline(always)]
    fn predict(t: f64, p: &[f64]) -> f64 {
        Self::combine(t, p, Self::term1(t, p), Self::term2(t, p))
    }
}

/// mn + (mx - mn) (1 / (1 + e^(-rsp (t - sos))) + 1 / (1 + e^(rau (t - eos))) - 1)
struct Beck;
impl Curve for Beck {
    const DEPS: &'static [u8] = &[0, 0, 1, 1, 2, 2];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        1.0 / (1.0 + (-p[3] * (t - p[2])).exp())
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        1.0 / (1.0 + (p[5] * (t - p[4])).exp())
    }
    #[inline(always)]
    fn combine(_t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        p[0] + (p[1] - p[0]) * (a + b - 1.0)
    }
}

/// mn + (mx - m7 t) (1 / (1 + e^(-rsp (t - sos))) - 1 / (1 + e^(-rau (t - eos))))
struct Elmore;
impl Curve for Elmore {
    const DEPS: &'static [u8] = &[0, 0, 1, 1, 2, 2, 0];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        1.0 / (1.0 + (-p[3] * (t - p[2])).exp())
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        1.0 / (1.0 + (-p[5] * (t - p[4])).exp())
    }
    #[inline(always)]
    fn combine(t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        p[0] + (p[1] - p[6] * t) * (a - b)
    }
}

/// y0 + a1 / base1^c1 - a2 / base2^c2, base = max(1e-8, 1 + e^(-r (t - s)))
struct Gu;
impl Curve for Gu {
    const DEPS: &'static [u8] = &[0, 0, 0, 1, 1, 2, 2, 1, 2];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        std_max(1e-8, 1.0 + (-p[4] * (t - p[3])).exp()).powf(p[7])
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        std_max(1e-8, 1.0 + (-p[6] * (t - p[5])).exp()).powf(p[8])
    }
    #[inline(always)]
    fn combine(_t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        p[0] + (p[1] / a) - (p[2] / b)
    }
}

/// (a1 t + b1) + (a2 t^2 + b2 t + c) (1 / base1^v1 - 1 / base2^v2),
/// base = max(1e-8, 1 + q e^(-B (t - m)))
struct Klos;
impl Curve for Klos {
    const DEPS: &'static [u8] = &[0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        std_max(1e-8, 1.0 + p[9] * (-p[5] * (t - p[7])).exp()).powf(p[11])
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        std_max(1e-8, 1.0 + p[10] * (-p[6] * (t - p[8])).exp()).powf(p[12])
    }
    #[inline(always)]
    fn combine(t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        (p[0] * t + p[2]) + (p[1] * t * t + p[3] * t + p[4]) * (1.0 / a - 1.0 / b)
    }
}

/// mn + (mx - mn) / (1 + e^(-rsp (t - sos))) up to t0, mn + (mx - mn) / (1 +
/// e^(rau (t - eos))) after; each term only on its side of t0
struct Zhang;
impl Curve for Zhang {
    const DEPS: &'static [u8] = &[3, 0, 0, 1, 1, 2, 2];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        if t <= p[0] {
            1.0 + (-p[4] * (t - p[3])).exp()
        } else {
            0.0
        }
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        if t <= p[0] {
            0.0
        } else {
            1.0 + (p[6] * (t - p[5])).exp()
        }
    }
    #[inline(always)]
    fn combine(t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        if t <= p[0] {
            p[1] + (p[2] - p[1]) / a
        } else {
            p[1] + (p[2] - p[1]) / b
        }
    }
}

/// mn + (mx - mn) e^(-max(0, (t0 - t) rsp)^a3) up to t0, mn + (mx - mn)
/// e^(-max(0, (t - t0) rau)^a5) after
struct Ag;
impl Curve for Ag {
    const DEPS: &'static [u8] = &[3, 0, 0, 1, 1, 2, 2];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        if t <= p[0] {
            (-std_max(0.0, (p[0] - t) * p[3]).powf(p[4])).exp()
        } else {
            0.0
        }
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        if t <= p[0] {
            0.0
        } else {
            (-std_max(0.0, (t - p[0]) * p[5]).powf(p[6])).exp()
        }
    }
    #[inline(always)]
    fn combine(t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        if t <= p[0] {
            p[1] + (p[2] - p[1]) * a
        } else {
            p[1] + (p[2] - p[1]) * b
        }
    }
}

/// mn + (mx - mn) (1 / (1 + e^(-rsp (t - sos))) - 1 / (1 + e^(-rau (t - eos))))
struct Dl;
impl Curve for Dl {
    const DEPS: &'static [u8] = &[0, 0, 1, 1, 2, 2];
    #[inline(always)]
    fn term1(t: f64, p: &[f64]) -> f64 {
        1.0 / (1.0 + (-p[3] * (t - p[2])).exp())
    }
    #[inline(always)]
    fn term2(t: f64, p: &[f64]) -> f64 {
        1.0 / (1.0 + (-p[5] * (t - p[4])).exp())
    }
    #[inline(always)]
    fn combine(_t: f64, p: &[f64], a: f64, b: f64) -> f64 {
        p[0] + (p[1] - p[0]) * (a - b)
    }
}

/// Runs `$body` with `$c` the curve type of `$kind`.
macro_rules! with_curve {
    ($kind:expr, $c:ident => $body:expr) => {
        match $kind {
            CurveType::Beck => {
                type $c = Beck;
                $body
            }
            CurveType::Elmore => {
                type $c = Elmore;
                $body
            }
            CurveType::Gu => {
                type $c = Gu;
                $body
            }
            CurveType::Klos => {
                type $c = Klos;
                $body
            }
            CurveType::Zhang => {
                type $c = Zhang;
                $body
            }
            CurveType::Ag => {
                type $c = Ag;
                $body
            }
            CurveType::Dl => {
                type $c = Dl;
                $body
            }
        }
    };
}

/// The curve of parameters `params` at `t`.
pub fn evaluate_curve(kind: CurveType, params: &[f64], t: &[f64]) -> Vec<f64> {
    with_curve!(kind, C => t.iter().map(|&ti| C::predict(ti, params)).collect())
}

/// The weighted residuals of a curve over the segment, parameters unbounded.
struct Residuals<'a> {
    kind: CurveType,
    t: &'a [f64],
    y: &'a [f64],
    /// sqrt(w), computed once
    sqrt_w: Vec<f64>,
    lb: &'a [f64],
    ub: &'a [f64],
}

impl Residuals<'_> {
    /// The bounded parameters of `x`.
    fn bounded(&self, x: &[f64]) -> [f64; 13] {
        let mut p = [0.0; 13];
        for i in 0..x.len() {
            p[i] = transform_to_bounded(x[i], self.lb[i], self.ub[i]);
        }
        p
    }

    fn eval_curve<C: Curve>(&self, x: &[f64], fvec: &mut [f64]) {
        let p = self.bounded(x);
        for i in 0..self.t.len() {
            fvec[i] = (C::predict(self.t[i], &p) - self.y[i]) * self.sqrt_w[i];
        }
    }

    /// NumericalDiff's forward differences, the terms a column's parameter
    /// does not enter kept from `x0`.
    fn jacobian_curve<C: Curve>(&self, x0: &[f64], fvec: &[f64], eps: f64, jac: &mut [f64]) {
        let (m, n) = (self.t.len(), x0.len());
        let (t, y, sw) = (self.t, self.y, &self.sqrt_w);
        let p = self.bounded(x0);
        let t1: Vec<f64> = t.iter().map(|&ti| C::term1(ti, &p)).collect();
        let t2: Vec<f64> = t.iter().map(|&ti| C::term2(ti, &p)).collect();
        for j in 0..n {
            let mut h = eps * x0[j].abs();
            if h == 0.0 {
                h = eps;
            }
            let mut pj = p;
            pj[j] = transform_to_bounded(x0[j] + h, self.lb[j], self.ub[j]);
            let col = &mut jac[j * m..(j + 1) * m];
            let dep = C::DEPS[j];
            for i in 0..m {
                let a = if dep & 1 != 0 { C::term1(t[i], &pj) } else { t1[i] };
                let b = if dep & 2 != 0 { C::term2(t[i], &pj) } else { t2[i] };
                let val2 = (C::combine(t[i], &pj, a, b) - y[i]) * sw[i];
                col[i] = (val2 - fvec[i]) / h;
            }
        }
    }
}

impl Functor for Residuals<'_> {
    fn values(&self) -> usize {
        self.t.len()
    }

    fn eval(&self, x: &[f64], fvec: &mut [f64]) {
        with_curve!(self.kind, C => self.eval_curve::<C>(x, fvec))
    }

    fn jacobian(&self, x: &[f64], fvec: &[f64], eps: f64, jac: &mut [f64]) {
        with_curve!(self.kind, C => self.jacobian_curve::<C>(x, fvec, eps, jac))
    }
}

/// optimize_functor: Levenberg-Marquardt from `params` (bounded), which it
/// overwrites; true when it converged.
#[allow(clippy::too_many_arguments)]
fn optimize(
    kind: CurveType,
    t: &[f64],
    y: &[f64],
    w: &[f64],
    params: &mut [f64],
    lb: &[f64],
    ub: &[f64],
    max_fev: usize,
) -> bool {
    let functor = Residuals {
        kind,
        t,
        y,
        sqrt_w: w.iter().map(|v| v.sqrt()).collect(),
        lb,
        ub,
    };
    let settings = Parameters {
        maxfev: max_fev,
        ftol: 1e-6,
        xtol: 1e-6,
        ..Parameters::default()
    };
    let mut x: Vec<f64> = (0..params.len())
        .map(|i| transform_to_unbounded(params[i], lb[i], ub[i]))
        .collect();
    let info = lm::minimize(&functor, &mut x, &settings);
    for i in 0..params.len() {
        params[i] = transform_to_bounded(x[i], lb[i], ub[i]);
    }
    matches!(
        info,
        Status::RelativeErrorTooSmall
            | Status::RelativeErrorAndReductionTooSmall
            | Status::RelativeReductionTooSmall
            | Status::CosinusTooSmall
    )
}

/// fit_curve: the bounds from the segment, the starting point from them when
/// `params` is all 1 or all 0 (or not of the curve's size), then the fit.
pub fn fit_curve(t: &[f64], y: &[f64], w: &[f64], params: &mut Vec<f64>, kind: CurveType, max_fev: usize) -> bool {
    if t.len() != y.len() || t.is_empty() {
        return false;
    }
    let t_min = min_coeff(t);
    let t_max = max_coeff(t);
    let y_min = min_coeff(y);
    let y_max = max_coeff(y);
    let y_amp = std_max(0.01, y_max - y_min);

    let mut half = (t_max - t_min) / 2.0;
    if half <= 0.0 {
        half = 100.0;
    }
    let k = 4.0 / half * 2.67;
    let r_min = k / 1.5;
    let r_max = k * 6.0;
    let d_t = half / 4.0;

    let np = kind.n_params();
    if params.len() != np {
        *params = vec![0.0; np];
    }
    let (lb, ub): (Vec<f64>, Vec<f64>) = match kind {
        CurveType::Beck => (
            vec![y_min - 0.5 * y_amp, y_min, t_min, r_min, t_min, r_min],
            vec![y_min + 0.5 * y_amp, y_max + 0.5 * y_amp, t_max, r_max, t_max, r_max],
        ),
        CurveType::Elmore => (
            vec![y_min - 0.5 * y_amp, y_min, t_min, r_min, t_min, r_min, -0.1],
            vec![
                y_min + 0.5 * y_amp,
                y_max + 0.5 * y_amp,
                t_max,
                r_max,
                t_max,
                r_max,
                0.1,
            ],
        ),
        CurveType::Gu => (
            vec![
                y_min - 0.5 * y_amp,
                0.0,
                0.0,
                t_min - 30.0,
                0.0,
                t_min - 30.0,
                0.0,
                0.1,
                0.1,
            ],
            vec![
                y_min + 0.5 * y_amp,
                5.0 * y_amp,
                5.0 * y_amp,
                t_max + 30.0,
                50.0,
                t_max + 30.0,
                50.0,
                10.0,
                10.0,
            ],
        ),
        CurveType::Klos => (
            vec![
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                y_min - 1.0,
                0.0,
                0.0,
                t_min - 30.0,
                t_min - 30.0,
                -5.0,
                -5.0,
                0.1,
                0.1,
            ],
            vec![
                1.0,
                1.0,
                1.0,
                1.0,
                y_max + 1.0,
                50.0,
                50.0,
                t_max + 30.0,
                t_max + 30.0,
                5.0,
                5.0,
                10.0,
                10.0,
            ],
        ),
        CurveType::Zhang => (
            vec![t_min, y_min - 0.5 * y_amp, y_min, t_min - 30.0, 0.0, t_min - 30.0, 0.0],
            vec![
                t_max,
                y_min + 0.5 * y_amp,
                y_max + 0.5 * y_amp,
                t_max + 30.0,
                50.0,
                t_max + 30.0,
                50.0,
            ],
        ),
        CurveType::Ag => (
            vec![t_min, y_min - 0.5 * y_amp, y_min, 0.0, 0.1, 0.0, 0.1],
            vec![t_max, y_min + 0.5 * y_amp, y_max + 0.5 * y_amp, 50.0, 10.0, 50.0, 10.0],
        ),
        CurveType::Dl => (
            vec![y_min - 0.5 * y_amp, y_min, t_min - 30.0, 0.0, t_min - 30.0, 0.0],
            vec![
                y_min + 0.5 * y_amp,
                y_max + 0.5 * y_amp,
                t_max + 30.0,
                50.0,
                t_max + 30.0,
                50.0,
            ],
        ),
    };

    if params.iter().all(|&v| v == 1.0) || params.iter().all(|&v| v == 0.0) {
        for i in 0..np {
            params[i] = (lb[i] + ub[i]) / 2.0;
        }
        let sos = t_min + (t_max - t_min) * 0.25;
        let eos = t_min + (t_max - t_min) * 0.75;
        let mid = t_min + (t_max - t_min) * 0.5;
        let set = |params: &mut Vec<f64>, values: &[(usize, f64)]| {
            for &(i, v) in values {
                params[i] = v;
            }
        };
        match kind {
            CurveType::Beck => set(params, &[(2, sos), (4, eos), (3, k), (5, k)]),
            CurveType::Elmore => set(params, &[(2, sos), (4, eos), (3, k), (5, k), (6, 0.0)]),
            CurveType::Gu => set(
                params,
                &[
                    (0, y_min),
                    (1, y_amp),
                    (2, y_amp),
                    (3, sos),
                    (4, k),
                    (5, eos),
                    (6, k),
                    (7, 1.0),
                    (8, 1.0),
                ],
            ),
            CurveType::Klos => set(
                params,
                &[
                    (0, 0.0),
                    (1, 0.0),
                    (2, y_min),
                    (3, 0.0),
                    (4, y_amp),
                    (5, k),
                    (6, k),
                    (7, sos),
                    (8, eos),
                    (9, 1.0),
                    (10, 1.0),
                    (11, 1.0),
                    (12, 1.0),
                ],
            ),
            CurveType::Zhang => set(
                params,
                &[(0, mid), (1, y_min), (2, y_max), (3, sos), (4, k), (5, eos), (6, k)],
            ),
            CurveType::Ag => set(
                params,
                &[
                    (0, mid),
                    (1, y_min),
                    (2, y_max),
                    (3, 1.0 / d_t),
                    (4, 1.0),
                    (5, 1.0 / d_t),
                    (6, 1.0),
                ],
            ),
            CurveType::Dl => set(params, &[(0, y_min), (1, y_max), (2, sos), (3, k), (4, eos), (5, k)]),
        }
    }
    optimize(kind, t, y, w, params, &lb, &ub, max_fev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_double_logistic() {
        let t: Vec<f64> = (0..23).map(|i| 1.0 + 16.0 * i as f64).collect();
        let truth = [0.2, 0.8, 120.0, 0.08, 280.0, 0.06];
        let y = evaluate_curve(CurveType::Beck, &truth, &t);
        let w = vec![1.0; t.len()];
        let mut params = vec![1.0; 6];
        assert!(fit_curve(&t, &y, &w, &mut params, CurveType::Beck, 2000));
        let fit = evaluate_curve(CurveType::Beck, &params, &t);
        let err = fit.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        // ftol and xtol of 1e-6 (relative) stop it early, the C++ too: the
        // timing is found, the plateau 0.15 low
        assert!(err < 0.2, "{params:?}");
        assert!(
            (params[2] - 120.0).abs() < 10.0 && (params[4] - 280.0).abs() < 10.0,
            "{params:?}"
        );
    }

    /// The Jacobian with kept terms is the forward differences of `eval`.
    #[test]
    fn jacobian_is_numericaldiffs() {
        let t: Vec<f64> = (0..30).map(|i| 1.0 + 12.0 * i as f64).collect();
        let y: Vec<f64> = t.iter().map(|&t| 0.3 + 0.4 * (t / 120.0).sin().abs()).collect();
        let w: Vec<f64> = (0..30).map(|i| 0.5 + (i % 3) as f64 * 0.25).collect();
        for code in 0..7 {
            let kind = CurveType::from_int(code).unwrap();
            let n = kind.n_params();
            let lb: Vec<f64> = (0..n).map(|i| -1.0 - i as f64).collect();
            let ub: Vec<f64> = (0..n).map(|i| 200.0 + 30.0 * i as f64).collect();
            let r = Residuals {
                kind,
                t: &t,
                y: &y,
                sqrt_w: w.iter().map(|v| v.sqrt()).collect(),
                lb: &lb,
                ub: &ub,
            };
            let x: Vec<f64> = (0..n).map(|i| 0.3 * i as f64 - 1.0).collect();
            let mut fvec = vec![0.0; t.len()];
            r.eval(&x, &mut fvec);
            let eps = f64::EPSILON.sqrt();
            let mut fast = vec![0.0; t.len() * n];
            r.jacobian(&x, &fvec, eps, &mut fast);
            let mut plain = vec![0.0; t.len() * n];
            lm::forward_differences(&r, &x, &fvec, eps, &mut plain);
            for (a, b) in fast.iter().zip(&plain) {
                assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "{kind:?}");
            }
        }
    }
}
