//! The seven seasonal curves of the phenology fit, their analytic Jacobians and
//! slopes, and the initial guess and bounds taken from a season's data.
//! Implemented from the papers (Beck et al. 2006; Elmore et al. 2012; Gu et al.
//! 2009; Klosterman et al. 2014; Zhang et al. 2003; Jonsson & Eklundh 2002,
//! 2004; Fisher et al. 2006 for the product form of `Dl`), without third-party
//! code. Choices that are Zeit's own are marked "Zeit's choice".
//!
//! Every position parameter (`sos`, `eos`, `t0`, `n1`, `n2`) is in tau = days
//! since the first time of the segment; the callers subtract that origin.
//!
//! Parameter layouts:
//! - Beck, Dl, Zhang: `[mn, mx, sos, rsp, eos, rau]`.
//! - Elmore: `[mn, mx, sos, rsp, eos, rau, m7]` (`mx` is the amplitude at tau = 0).
//! - Gu: `[y0, a1, a2, sos, rsp, eos, rau, c1, c2]`.
//! - Klos: `[a1, a2, b1, b2, c, h1, h2, n1, n2, v1, v2]` (Zeit's choice: the
//!   `q` of the paper are fixed to 1, they are redundant with `n`).
//! - Ag: `[t0, mn, mx, wL, pL, wR, pR]` (widths in days, Zeit's choice).

/// The curves, with the numbering of the Python layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
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
    pub fn from_int(v: i64) -> Option<Self> {
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

    /// Number of parameters of the curve.
    pub fn npar(self) -> usize {
        match self {
            CurveType::Beck | CurveType::Zhang | CurveType::Dl => 6,
            CurveType::Elmore | CurveType::Ag => 7,
            CurveType::Gu => 9,
            CurveType::Klos => 11,
        }
    }
}

/// Largest number of parameters of any curve.
pub const MAX_PARAMS: usize = 11;

/// Logistic function, evaluated by branch so it never overflows.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-700.0, 700.0);
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln(1 + e^z)` without overflow.
#[inline]
fn softplus(z: f64) -> f64 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// The junction of the two branches of Zhang's curve: where
/// `sigma(rsp (tau - sos)) = sigma(-rau (tau - eos))`.
#[inline]
fn zhang_junction(p: &[f64]) -> f64 {
    (p[3] * p[2] + p[5] * p[4]) / (p[3] + p[5])
}

/// `S = (1 + e^{-h (tau - n)})^{-v}` of Klosterman's curve with the pieces of
/// its derivatives: `(S, ln s, sigma(x))`, `x = -h (tau - n)`, `s = 1 + e^x`.
#[inline]
fn klos_sigmoid(h: f64, n: f64, v: f64, tau: f64) -> (f64, f64, f64) {
    let x = -h * (tau - n);
    let ln_s = softplus(x);
    ((-v * ln_s).exp(), ln_s, sigmoid(x))
}

/// The Asymmetric Gaussian's pieces at `tau`: `(B, z, E, left)`.
#[inline]
fn ag_parts(p: &[f64], tau: f64) -> (f64, f64, f64, bool) {
    let left = tau < p[0];
    let (w, pw) = if left { (p[3], p[4]) } else { (p[5], p[6]) };
    let z = (tau - p[0]).abs() / w;
    let e = (-z.powf(pw)).exp();
    (p[2] - p[1], z, e, left)
}

/// The curve at `tau` (days since the segment's first time).
pub fn value(kind: CurveType, p: &[f64], tau: f64) -> f64 {
    match kind {
        CurveType::Beck => {
            let u = sigmoid(p[3] * (tau - p[2]));
            let v = sigmoid(p[5] * (tau - p[4]));
            p[0] + (p[1] - p[0]) * (u - v)
        }
        CurveType::Dl => {
            let u = sigmoid(p[3] * (tau - p[2]));
            let w = sigmoid(-p[5] * (tau - p[4]));
            p[0] + (p[1] - p[0]) * u * w
        }
        CurveType::Elmore => {
            let u = sigmoid(p[3] * (tau - p[2]));
            let v = sigmoid(p[5] * (tau - p[4]));
            p[0] + (p[1] - p[6] * tau) * (u - v)
        }
        CurveType::Gu => {
            let ln_u = -softplus(-p[4] * (tau - p[3]));
            let ln_v = -softplus(-p[6] * (tau - p[5]));
            p[0] + p[1] * (p[7] * ln_u).exp() - p[2] * (p[8] * ln_v).exp()
        }
        CurveType::Klos => {
            let (s1, _, _) = klos_sigmoid(p[5], p[7], p[9], tau);
            let (s2, _, _) = klos_sigmoid(p[6], p[8], p[10], tau);
            let m = (p[1] * tau + p[3]) * tau + p[4];
            (p[0] * tau + p[2]) + m * (s1 - s2)
        }
        CurveType::Zhang => {
            let b = p[1] - p[0];
            if tau <= zhang_junction(p) {
                p[0] + b * sigmoid(p[3] * (tau - p[2]))
            } else {
                p[0] + b * sigmoid(-p[5] * (tau - p[4]))
            }
        }
        CurveType::Ag => {
            let (b, _, e, _) = ag_parts(p, tau);
            p[1] + b * e
        }
    }
}

/// The curve at each of the `tau`.
pub fn evaluate_curve(kind: CurveType, params: &[f64], tau: &[f64]) -> Vec<f64> {
    tau.iter().map(|&x| value(kind, params, x)).collect()
}

/// The curve at absolute times `t` for parameters fitted with origin `origin`.
pub fn evaluate_curve_at(kind: CurveType, params: &[f64], t: &[f64], origin: f64) -> Vec<f64> {
    t.iter().map(|&x| value(kind, params, x - origin)).collect()
}

/// The partial derivatives of the curve with respect to the parameters at
/// `tau`, into `out[..npar]`.
pub fn gradient(kind: CurveType, p: &[f64], tau: f64, out: &mut [f64]) {
    match kind {
        CurveType::Beck => {
            let (x1, x2) = (p[3] * (tau - p[2]), p[5] * (tau - p[4]));
            let (u, v) = (sigmoid(x1), sigmoid(x2));
            let (uu, vv) = (u * (1.0 - u), v * (1.0 - v));
            let b = p[1] - p[0];
            let a = u - v;
            out[0] = 1.0 - a;
            out[1] = a;
            out[2] = -b * p[3] * uu;
            out[3] = b * (tau - p[2]) * uu;
            out[4] = b * p[5] * vv;
            out[5] = -b * (tau - p[4]) * vv;
        }
        CurveType::Dl => {
            let (x1, x2) = (p[3] * (tau - p[2]), -p[5] * (tau - p[4]));
            let (u, w) = (sigmoid(x1), sigmoid(x2));
            let (uu, ww) = (u * (1.0 - u), w * (1.0 - w));
            let b = p[1] - p[0];
            out[0] = 1.0 - u * w;
            out[1] = u * w;
            out[2] = -b * p[3] * uu * w;
            out[3] = b * (tau - p[2]) * uu * w;
            out[4] = b * u * p[5] * ww;
            out[5] = -b * u * (tau - p[4]) * ww;
        }
        CurveType::Elmore => {
            let (x1, x2) = (p[3] * (tau - p[2]), p[5] * (tau - p[4]));
            let (u, v) = (sigmoid(x1), sigmoid(x2));
            let (uu, vv) = (u * (1.0 - u), v * (1.0 - v));
            let g = p[1] - p[6] * tau;
            let a = u - v;
            out[0] = 1.0;
            out[1] = a;
            out[2] = -g * p[3] * uu;
            out[3] = g * (tau - p[2]) * uu;
            out[4] = g * p[5] * vv;
            out[5] = -g * (tau - p[4]) * vv;
            out[6] = -tau * a;
        }
        CurveType::Gu => {
            let (x1, x2) = (p[4] * (tau - p[3]), p[6] * (tau - p[5]));
            let (ln_u, ln_v) = (-softplus(-x1), -softplus(-x2));
            let (big_u, big_v) = ((p[7] * ln_u).exp(), (p[8] * ln_v).exp());
            let (ub, vb) = (1.0 - ln_u.exp(), 1.0 - ln_v.exp());
            out[0] = 1.0;
            out[1] = big_u;
            out[2] = -big_v;
            out[3] = -p[1] * p[7] * p[4] * big_u * ub;
            out[4] = p[1] * p[7] * (tau - p[3]) * big_u * ub;
            out[5] = p[2] * p[8] * p[6] * big_v * vb;
            out[6] = -p[2] * p[8] * (tau - p[5]) * big_v * vb;
            out[7] = p[1] * big_u * ln_u;
            out[8] = -p[2] * big_v * ln_v;
        }
        CurveType::Klos => {
            let (s1, ls1, g1) = klos_sigmoid(p[5], p[7], p[9], tau);
            let (s2, ls2, g2) = klos_sigmoid(p[6], p[8], p[10], tau);
            let d = s1 - s2;
            let m = (p[1] * tau + p[3]) * tau + p[4];
            out[0] = tau;
            out[1] = tau * tau * d;
            out[2] = 1.0;
            out[3] = tau * d;
            out[4] = d;
            out[5] = m * p[9] * (tau - p[7]) * s1 * g1;
            out[6] = -m * p[10] * (tau - p[8]) * s2 * g2;
            out[7] = -m * p[9] * p[5] * s1 * g1;
            out[8] = m * p[10] * p[6] * s2 * g2;
            out[9] = -m * s1 * ls1;
            out[10] = m * s2 * ls2;
        }
        CurveType::Zhang => {
            let b = p[1] - p[0];
            out[..6].fill(0.0);
            if tau <= zhang_junction(p) {
                let x = p[3] * (tau - p[2]);
                let u = sigmoid(x);
                let uu = u * (1.0 - u);
                out[0] = 1.0 - u;
                out[1] = u;
                out[2] = -b * p[3] * uu;
                out[3] = b * (tau - p[2]) * uu;
            } else {
                let x = -p[5] * (tau - p[4]);
                let w = sigmoid(x);
                let ww = w * (1.0 - w);
                out[0] = 1.0 - w;
                out[1] = w;
                out[4] = b * p[5] * ww;
                out[5] = -b * (tau - p[4]) * ww;
            }
        }
        CurveType::Ag => {
            let (b, z, e, left) = ag_parts(p, tau);
            let (w, pw) = if left { (p[3], p[4]) } else { (p[5], p[6]) };
            out[..7].fill(0.0);
            out[1] = 1.0 - e;
            out[2] = e;
            let zp = z.powf(pw);
            let (dw, dp) = (b * e * pw * zp / w, if z > 0.0 { -b * e * zp * z.ln() } else { 0.0 });
            let dt0 = if z > 0.0 {
                let mag = b * e * pw * z.powf(pw - 1.0) / w;
                if left { -mag } else { mag }
            } else {
                0.0
            };
            out[0] = dt0;
            if left {
                out[3] = dw;
                out[4] = dp;
            } else {
                out[5] = dw;
                out[6] = dp;
            }
        }
    }
}

/// The first derivative of the curve in `tau`, analytic (units of y per day).
pub fn slope(kind: CurveType, p: &[f64], tau: f64) -> f64 {
    match kind {
        CurveType::Beck => {
            let (x1, x2) = (p[3] * (tau - p[2]), p[5] * (tau - p[4]));
            let (u, v) = (sigmoid(x1), sigmoid(x2));
            (p[1] - p[0]) * (p[3] * u * (1.0 - u) - p[5] * v * (1.0 - v))
        }
        CurveType::Dl => {
            let (x1, x2) = (p[3] * (tau - p[2]), -p[5] * (tau - p[4]));
            let (u, w) = (sigmoid(x1), sigmoid(x2));
            let (du, dw) = (p[3] * u * (1.0 - u), -p[5] * w * (1.0 - w));
            (p[1] - p[0]) * (du * w + u * dw)
        }
        CurveType::Elmore => {
            let (x1, x2) = (p[3] * (tau - p[2]), p[5] * (tau - p[4]));
            let (u, v) = (sigmoid(x1), sigmoid(x2));
            let g = p[1] - p[6] * tau;
            -p[6] * (u - v) + g * (p[3] * u * (1.0 - u) - p[5] * v * (1.0 - v))
        }
        CurveType::Gu => {
            let (x1, x2) = (p[4] * (tau - p[3]), p[6] * (tau - p[5]));
            let (ln_u, ln_v) = (-softplus(-x1), -softplus(-x2));
            let (big_u, big_v) = ((p[7] * ln_u).exp(), (p[8] * ln_v).exp());
            p[1] * p[7] * p[4] * big_u * (1.0 - ln_u.exp()) - p[2] * p[8] * p[6] * big_v * (1.0 - ln_v.exp())
        }
        CurveType::Klos => {
            let (s1, _, g1) = klos_sigmoid(p[5], p[7], p[9], tau);
            let (s2, _, g2) = klos_sigmoid(p[6], p[8], p[10], tau);
            let d = s1 - s2;
            let m = (p[1] * tau + p[3]) * tau + p[4];
            let dm = 2.0 * p[1] * tau + p[3];
            p[0] + dm * d + m * (p[9] * p[5] * s1 * g1 - p[10] * p[6] * s2 * g2)
        }
        CurveType::Zhang => {
            let b = p[1] - p[0];
            if tau <= zhang_junction(p) {
                let x = p[3] * (tau - p[2]);
                { let u = sigmoid(x); b * p[3] * u * (1.0 - u) }
            } else {
                let x = -p[5] * (tau - p[4]);
                { let u = sigmoid(x); -b * p[5] * u * (1.0 - u) }
            }
        }
        CurveType::Ag => {
            let (b, z, e, left) = ag_parts(p, tau);
            if z <= 0.0 {
                return 0.0;
            }
            let (w, pw) = if left { (p[3], p[4]) } else { (p[5], p[6]) };
            let mag = b * e * pw * z.powf(pw - 1.0) / w;
            if left { mag } else { -mag }
        }
    }
}

/// The second derivative, by a central difference of the analytic slope with
/// a step of 0.01 day (Zeit's choice: it spares the second-order algebra of
/// Gu, Klosterman and the Gaussian; the error is far below what the metrics
/// can see).
pub fn second_derivative(kind: CurveType, p: &[f64], tau: f64) -> f64 {
    let h = 0.01;
    (slope(kind, p, tau + h) - slope(kind, p, tau - h)) / (2.0 * h)
}

/// Summary of a season's valid points, the base of the guess and the bounds.
#[derive(Clone, Debug)]
pub struct SeasonStats {
    pub n_valid: usize,
    /// Robust minimum and maximum of y (weighted percentiles 2 and 98 when
    /// there are 20 valid points or more, the extremes otherwise).
    pub lo: f64,
    pub hi: f64,
    /// `hi - lo`.
    pub a: f64,
    /// Time of the maximum of the 3-point running median.
    pub tp: f64,
    /// Median step between valid points.
    pub dt: f64,
    /// Span of the segment, `tau` of its last point.
    pub span: f64,
    pub sos0: f64,
    pub eos0: f64,
}

fn weighted_quantile(sorted: &[(f64, f64)], total: f64, q: f64) -> f64 {
    let target = q * total;
    let mut cum = 0.0;
    for &(v, w) in sorted {
        cum += w;
        if cum >= target {
            return v;
        }
    }
    sorted[sorted.len() - 1].0
}

/// The statistics of section 1.8 of the specification; `None` for a constant
/// series or fewer than 3 valid points. `tau` is increasing.
pub fn season_stats(tau: &[f64], y: &[f64], w: &[f64]) -> Option<SeasonStats> {
    let idx: Vec<usize> = (0..tau.len()).filter(|&i| w[i] > 0.0 && y[i].is_finite() && tau[i].is_finite()).collect();
    let n = idx.len();
    if n < 3 || tau.len() < 2 {
        return None;
    }
    let span = tau[tau.len() - 1] - tau[0];
    if !(span > 0.0) {
        return None;
    }
    let (lo, hi) = if n >= 20 {
        let mut s: Vec<(f64, f64)> = idx.iter().map(|&i| (y[i], w[i])).collect();
        s.sort_by(|a, b| a.0.total_cmp(&b.0));
        let total: f64 = s.iter().map(|x| x.1).sum();
        (weighted_quantile(&s, total, 0.02), weighted_quantile(&s, total, 0.98))
    } else {
        let lo = idx.iter().map(|&i| y[i]).fold(f64::INFINITY, f64::min);
        let hi = idx.iter().map(|&i| y[i]).fold(f64::NEG_INFINITY, f64::max);
        (lo, hi)
    };
    let a = hi - lo;
    if !(a > 1e-9 * hi.abs().max(1.0)) {
        return None;
    }
    let tv: Vec<f64> = idx.iter().map(|&i| tau[i] - tau[0]).collect();
    let yv: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
    let mut steps: Vec<f64> = tv.windows(2).map(|s| s[1] - s[0]).collect();
    steps.sort_by(|a, b| a.total_cmp(b));
    let dt = steps[steps.len() / 2];
    if !(dt > 0.0) {
        return None;
    }
    // 3-point running median, the ends kept
    let med3 = |a: f64, b: f64, c: f64| a.max(b).min(a.min(b).max(c));
    let sm: Vec<f64> = (0..n)
        .map(|i| if i == 0 || i == n - 1 { yv[i] } else { med3(yv[i - 1], yv[i], yv[i + 1]) })
        .collect();
    let jp = (0..n).fold(0, |b, i| if sm[i] > sm[b] { i } else { b });
    let tp = tv[jp];
    let level = lo + 0.5 * a;
    let mut sos0 = f64::NAN;
    for i in 0..jp {
        if sm[i] < level && level <= sm[i + 1] {
            sos0 = tv[i] + (level - sm[i]) / (sm[i + 1] - sm[i]) * (tv[i + 1] - tv[i]);
            break;
        }
    }
    if !sos0.is_finite() {
        sos0 = 0.25 * span;
    }
    let mut eos0 = f64::NAN;
    for i in (jp..n - 1).rev() {
        if sm[i] >= level && level > sm[i + 1] {
            eos0 = tv[i] + (sm[i] - level) / (sm[i] - sm[i + 1]) * (tv[i + 1] - tv[i]);
            break;
        }
    }
    if !eos0.is_finite() {
        eos0 = 0.75 * span;
    }
    if !(sos0 < tp && tp < eos0) {
        sos0 = (tp - 0.25 * span).max(0.0);
        eos0 = (tp + 0.25 * span).min(span);
    }
    Some(SeasonStats { n_valid: n, lo, hi, a, tp, dt, span, sos0, eos0 })
}

/// Starting points (in order of attempt) and box bounds of a fit.
#[derive(Clone, Debug)]
pub struct Guess {
    pub starts: Vec<Vec<f64>>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    /// The observed amplitude `a`, for the sanity check after the fit.
    pub amplitude: f64,
    pub span: f64,
}

/// Initial guesses and bounds from the season's data (section 1.8: Zeit's
/// choices, from the valid points of the segment). `None` when the segment
/// cannot be fitted: a constant series or fewer than `npar + 2` valid points.
/// Starts: the base guess, then the transitions moved 10% of the span inward,
/// then (plateau curves only) half and double rate. Parameters at a bound are
/// pushed 1% of the interval inward (the sine transform has no gradient there).
pub fn guess_and_bounds(kind: CurveType, tau: &[f64], y: &[f64], w: &[f64]) -> Option<Guess> {
    let st = season_stats(tau, y, w)?;
    if st.n_valid < kind.npar() + 2 {
        return None;
    }
    let (lo, hi, a, tp, dt, l) = (st.lo, st.hi, st.a, st.tp, st.dt, st.span);
    let (rl, rh) = (3.0 / l, 2.2 / dt);
    if !(rh > rl) {
        return None;
    }
    let r0 = (25.0 / l).clamp(rl, rh);
    let (sos0, eos0) = (st.sos0, st.eos0);
    let (mn_lo, mn_hi) = (lo - 0.25 * a, lo + 0.5 * a);
    let (mx_lo, mx_hi) = (lo + 0.5 * a, hi + 0.5 * a);
    let (sos_lo, sos_hi) = (-0.1 * l, tp);
    let (eos_lo, eos_hi) = (tp, 1.1 * l);
    let shift = 0.1 * l;
    let (lower, upper, mut starts): (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) = match kind {
        CurveType::Beck | CurveType::Dl | CurveType::Zhang => {
            let base = vec![lo, hi, sos0, r0, eos0, r0];
            (
                vec![mn_lo, mx_lo, sos_lo, rl, eos_lo, rl],
                vec![mn_hi, mx_hi, sos_hi, rh, eos_hi, rh],
                plateau_starts(&base, 2, shift, &[3, 5]),
            )
        }
        CurveType::Elmore => {
            let base = vec![lo, a, sos0, r0, eos0, r0, 0.0];
            let m = 0.5 * a / l;
            // Zeit's choice: `mx` is an amplitude here, not a level
            (
                vec![mn_lo, 0.5 * a, sos_lo, rl, eos_lo, rl, -m],
                vec![mn_hi, 1.75 * a, sos_hi, rh, eos_hi, rh, m],
                plateau_starts(&base, 2, shift, &[3, 5]),
            )
        }
        CurveType::Gu => {
            let base = vec![lo, a, a, sos0, r0, eos0, r0, 1.0, 1.0];
            (
                vec![mn_lo, 0.25 * a, 0.25 * a, sos_lo, rl, eos_lo, rl, 0.25, 0.25],
                vec![mn_hi, 1.5 * a, 1.5 * a, sos_hi, rh, eos_hi, rh, 4.0, 4.0],
                plateau_starts(&base, 3, shift, &[4, 6]),
            )
        }
        CurveType::Klos => {
            let base = vec![0.0, 0.0, lo, 0.0, a, r0, r0, sos0, eos0, 1.0, 1.0];
            let mut inward = base.clone();
            inward[7] += shift;
            inward[8] -= shift;
            (
                vec![-0.5 * a / l, -a / (l * l), mn_lo, -a / l, 0.3 * a, rl, rl, sos_lo, eos_lo, 0.2, 0.2],
                vec![0.5 * a / l, a / (l * l), mn_hi, a / l, 1.5 * a, rh, rh, sos_hi, eos_hi, 5.0, 5.0],
                vec![base, inward],
            )
        }
        CurveType::Ag => {
            let t0 = tp.clamp(0.05 * l, 0.95 * l);
            let wl = ((t0 - 0.0).max(3.0 * dt)) / 2.0;
            let wr = ((l - t0).max(3.0 * dt)) / 2.0;
            let base = vec![t0, lo, hi, wl, 2.0, wr, 2.0];
            let narrow = vec![t0, lo, hi, 0.5 * wl, 3.0, 0.5 * wr, 3.0];
            (
                vec![0.05 * l, mn_lo, mx_lo, 1.5 * dt, 1.2, 1.5 * dt, 1.2],
                vec![0.95 * l, mn_hi, mx_hi, 2.0 * l, 12.0, 2.0 * l, 12.0],
                vec![base, narrow],
            )
        }
    };
    for s in starts.iter_mut() {
        for k in 0..s.len() {
            s[k] = push_inside(s[k], lower[k], upper[k]);
        }
    }
    Some(Guess { starts, lower, upper, amplitude: a, span: l })
}

/// Base start, the transitions (`sos` at `pos`, `eos` at `pos + 2`) moved
/// inward, and the rates (`rate_idx`) halved and doubled.
fn plateau_starts(base: &[f64], pos: usize, shift: f64, rate_idx: &[usize]) -> Vec<Vec<f64>> {
    let mut inward = base.to_vec();
    inward[pos] += shift;
    inward[pos + 2] -= shift;
    let (mut slow, mut fast) = (base.to_vec(), base.to_vec());
    for &k in rate_idx {
        slow[k] *= 0.5;
        fast[k] *= 2.0;
    }
    vec![base.to_vec(), inward, slow, fast]
}

fn push_inside(x: f64, lo: f64, hi: f64) -> f64 {
    let m = 0.01 * (hi - lo);
    x.clamp(lo + m, hi - m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenology::testutil::Rng;

    const ALL: [CurveType; 7] = [
        CurveType::Beck,
        CurveType::Elmore,
        CurveType::Gu,
        CurveType::Klos,
        CurveType::Zhang,
        CurveType::Ag,
        CurveType::Dl,
    ];

    /// A guess's bounds on a typical season, to draw random parameters in.
    fn bounds_of(kind: CurveType) -> Guess {
        let tau: Vec<f64> = (0..46).map(|i| i as f64 * 8.0).collect();
        let y: Vec<f64> = tau
            .iter()
            .map(|&t| 0.1 + 0.6 * (sigmoid(0.1 * (t - 50.0)) - sigmoid(0.1 * (t - 250.0))))
            .collect();
        guess_and_bounds(kind, &tau, &y, &vec![1.0; 46]).expect("guess")
    }

    #[test]
    fn from_int_round_trip() {
        for (i, k) in ALL.iter().enumerate() {
            assert_eq!(CurveType::from_int(*k as i64), Some(*k));
            assert!(k.npar() >= 6, "{i}");
        }
        assert_eq!(CurveType::from_int(7), None);
        assert_eq!(CurveType::from_int(-1), None);
    }

    #[test]
    fn jacobian_and_slope_match_finite_differences() {
        let mut rng = Rng::new(7);
        for kind in ALL {
            let g = bounds_of(kind);
            let np = kind.npar();
            for _ in 0..20 {
                let p: Vec<f64> =
                    (0..np).map(|k| g.lower[k] + (0.05 + 0.9 * rng.uniform()) * (g.upper[k] - g.lower[k])).collect();
                let tau = 3.0 + 360.0 * rng.uniform();
                let mut grad = vec![0.0; np];
                gradient(kind, &p, tau, &mut grad);
                for k in 0..np {
                    let range = g.upper[k] - g.lower[k];
                    let h = 1e-6 * range;
                    let (mut pp, mut pm) = (p.clone(), p.clone());
                    pp[k] += h;
                    pm[k] -= h;
                    let fd = (value(kind, &pp, tau) - value(kind, &pm, tau)) / (2.0 * h);
                    // compared in units of the parameter's range: a change over
                    // the whole range below 1e-3 in y is of no consequence
                    let scale = (fd.abs().max(grad[k].abs()) * range).max(1e-3);
                    assert!((fd - grad[k]).abs() * range <= 1e-5 * scale, "{kind:?} param {k}: analytic {} fd {fd}", grad[k]);
                }
                let h = 1e-4;
                let fd = (value(kind, &p, tau + h) - value(kind, &p, tau - h)) / (2.0 * h);
                let an = slope(kind, &p, tau);
                assert!((fd - an).abs() <= 1e-6 * fd.abs().max(an.abs()).max(1e-3), "{kind:?} slope {an} vs {fd}");
            }
        }
    }

    #[test]
    fn beck_and_dl_agree_when_transitions_are_apart() {
        let p = [0.1, 0.7, 50.0, 0.1, 250.0, 0.1];
        for t in [0.0, 50.0, 150.0, 250.0, 364.0] {
            let b = value(CurveType::Beck, &p, t);
            let d = value(CurveType::Dl, &p, t);
            assert!((b - d).abs() < 1e-4, "{t}: {b} {d}");
        }
        // but they are different curves where the transitions overlap
        let q = [0.0, 1.0, 100.0, 0.05, 130.0, 0.05];
        assert!((value(CurveType::Beck, &q, 115.0) - value(CurveType::Dl, &q, 115.0)).abs() > 0.01);
    }

    #[test]
    fn guess_fails_on_constant_series_and_short_ones() {
        let tau: Vec<f64> = (0..46).map(|i| i as f64 * 8.0).collect();
        assert!(guess_and_bounds(CurveType::Beck, &tau, &vec![0.4; 46], &vec![1.0; 46]).is_none());
        let y: Vec<f64> = tau.iter().map(|&t| (t / 100.0).sin()).collect();
        let mut w = vec![0.0; 46];
        for k in 0..4 {
            w[k * 10] = 1.0;
        }
        assert!(guess_and_bounds(CurveType::Beck, &tau, &y, &w).is_none());
        assert!(guess_and_bounds(CurveType::Beck, &tau, &y, &vec![0.0; 46]).is_none());
    }

    #[test]
    fn guess_is_inside_the_bounds() {
        for kind in ALL {
            let g = bounds_of(kind);
            for s in &g.starts {
                assert_eq!(s.len(), kind.npar());
                for (k, &x) in s.iter().enumerate() {
                    assert!(x > g.lower[k] && x < g.upper[k], "{kind:?} {k}: {x} not in ({}, {})", g.lower[k], g.upper[k]);
                }
            }
        }
    }
}
