//! The 19 phenology metrics of a fitted season and its goodness of fit.
//! Implemented from the papers (White et al. 1997 and Kong et al. 2022 for
//! the thresholds and the derivative; Gu et al. 2009 for the tangents;
//! Zhang et al. 2003 for the curvature; Klosterman et al. 2014), without
//! third-party code. Choices that are Zeit's own are marked.
//!
//! Dates come out in the unit of `t_segment` (the caller passes days of the
//! year, or days since a fixed origin); `LOS` is in days. A metric that does
//! not exist (the season is cut by the segment, a tangent is degenerate...)
//! is NaN.
//!
//! The curve is evaluated on a grid of whole days, `ceil(t_segment[0])` to
//! `t_segment[last]`. The first derivative is analytic; the second one, and
//! the rate of change of the curvature, are central differences on a grid of
//! a quarter of a day (Zeit's choice, see `curves::second_derivative`).

use super::curves::{slope, value, CurveType};

/// Number of metrics returned by `extract_metrics`.
pub const N_METRICS: usize = 19;

/// Names of the metrics, in the order of `extract_metrics`.
pub const METRIC_NAMES: [&str; N_METRICS] = [
    "TRS2.sos", "TRS2.eos", "TRS5.sos", "TRS5.eos", "TRS6.sos", "TRS6.eos", "DER.sos", "DER.pos", "DER.eos", "UD",
    "SD", "DD", "RD", "Greenup", "Maturity", "Senescence", "Dormancy", "LOS", "POP",
];

/// Thresholds of TRS2, TRS5 and TRS6 (Zeit's choice: 20%, 50%, 60%).
const THRESHOLDS: [f64; 3] = [0.2, 0.5, 0.6];

/// Index of the first extreme (maximum if `max`) of `v[lo..=hi]`, ties to the first.
fn arg_ext(v: &[f64], lo: usize, hi: usize, max: bool) -> usize {
    let mut best = lo;
    for j in lo + 1..=hi {
        let better = if max { v[j] > v[best] } else { v[j] < v[best] };
        if better {
            best = j;
        }
    }
    best
}

/// Fractional offset of the extreme of the parabola through `v[j-1..=j+1]`.
fn parabola_offset(v: &[f64], j: usize) -> f64 {
    let (a, b, c) = (v[j - 1], v[j], v[j + 1]);
    let den = a - 2.0 * b + c;
    if den.abs() < 1e-300 || !den.is_finite() {
        return 0.0;
    }
    (0.5 * (a - c) / den).clamp(-1.0, 1.0)
}

/// The 19 metrics of a curve of `kind` with `params` (origin `t_segment[0]`)
/// over the segment `t_segment`: TRS2.sos, TRS2.eos, TRS5.sos, TRS5.eos,
/// TRS6.sos, TRS6.eos, DER.sos, DER.pos, DER.eos, UD, SD, DD, RD, Greenup,
/// Maturity, Senescence, Dormancy, LOS, POP.
///
/// - TRS: first rising and last falling crossing of 20/50/60% of the curve's
///   range (one baseline, the curve's minimum); NaN if the curve starts (ends)
///   above the threshold. Linear interpolation on the daily grid.
/// - DER: SOS and EOS at the steepest rise and fall (refined by a parabola),
///   POS where the slope goes from + to - (the one of highest curve if several).
/// - UD, SD, DD, RD (Gu et al. 2009, Zeit's reading): the tangent at the
///   steepest rise meets the left baseline (minimum of the rising half, UD)
///   and the maximum (SD); the tangent at the steepest fall meets the maximum
///   (DD) and the right baseline (minimum of the falling half, RD).
/// - Greenup, Maturity, Senescence, Dormancy (Zhang et al. 2003): extremes of
///   the rate of change of the curvature `K = f''/(1+f'^2)^(3/2)` of the curve
///   scaled to [0, 1] (Zeit's choice), between the inflection points and the
///   peak. NaN if an extreme falls on an end of its search window.
/// - LOS = TRS5.eos - TRS5.sos; POP = date of the maximum of the curve (the
///   daily grid's, refined within a day).
pub fn extract_metrics(params: &[f64], kind: CurveType, t_segment: &[f64]) -> [f64; N_METRICS] {
    let mut out = [f64::NAN; N_METRICS];
    if params.len() != kind.npar() || params.iter().any(|v| !v.is_finite()) || t_segment.len() < 2 {
        return out;
    }
    let t0 = t_segment[0];
    let t_end = t_segment[t_segment.len() - 1];
    if !(t_end - t0 >= 8.0) {
        return out;
    }
    let span = t_end - t0;
    let first = t0.ceil();
    let n = (t_end.floor() - first) as usize + 1;
    if n < 8 {
        return out;
    }
    let tau_of = |j: usize| first + j as f64 - t0;
    let abs_of = |x: f64| first + x; // x: fractional grid index
    let f: Vec<f64> = (0..n).map(|j| value(kind, params, tau_of(j))).collect();
    if f.iter().any(|v| !v.is_finite()) {
        return out;
    }
    let d1: Vec<f64> = (0..n).map(|j| slope(kind, params, tau_of(j))).collect();
    let jp = arg_ext(&f, 0, n - 1, true);
    let (fmax, fmin) = (f[jp], f.iter().cloned().fold(f64::INFINITY, f64::min));
    if fmax - fmin <= 1e-12 * fmax.abs().max(1.0) {
        return out;
    }

    // thresholds
    let r: Vec<f64> = f.iter().map(|&v| (v - fmin) / (fmax - fmin)).collect();
    for (i, &th) in THRESHOLDS.iter().enumerate() {
        if r[0] < th {
            for j in 0..jp {
                if r[j] < th && th <= r[j + 1] {
                    out[2 * i] = abs_of(j as f64 + (th - r[j]) / (r[j + 1] - r[j]));
                    break;
                }
            }
        }
        if r[n - 1] < th {
            for j in (jp..n - 1).rev() {
                if r[j] >= th && th > r[j + 1] {
                    out[2 * i + 1] = abs_of(j as f64 + (r[j] - th) / (r[j] - r[j + 1]));
                    break;
                }
            }
        }
    }

    // derivative
    let js = arg_ext(&d1, 0, jp, true);
    let jd = arg_ext(&d1, jp, n - 1, false);
    let js_ok = js > 0 && js < n - 1 && d1[js] > 0.0;
    let jd_ok = jd > 0 && jd < n - 1 && d1[jd] < 0.0;
    if js_ok {
        out[6] = abs_of(js as f64 + parabola_offset(&d1, js));
    }
    if jd_ok {
        out[8] = abs_of(jd as f64 + parabola_offset(&d1, jd));
    }
    if js_ok && jd_ok && js < jd {
        let mut best: Option<(f64, f64)> = None; // (curve value, position)
        for j in js..jd {
            if d1[j] > 0.0 && d1[j + 1] <= 0.0 {
                let x = j as f64 + d1[j] / (d1[j] - d1[j + 1]);
                let fx = f[j].max(f[j + 1]);
                if best.is_none_or(|(bf, _)| fx > bf) {
                    best = Some((fx, x));
                }
            }
        }
        if let Some((_, x)) = best {
            out[7] = abs_of(x);
        }
    }

    // Gu's tangents
    let in_range = |x: f64| x >= -span && x <= 2.0 * span;
    if js_ok && d1[js].abs() >= 1e-12 {
        let fmin_left = f[..=jp].iter().cloned().fold(f64::INFINITY, f64::min);
        let ud = tau_of(js) - (f[js] - fmin_left) / d1[js];
        let sd = tau_of(js) + (fmax - f[js]) / d1[js];
        if in_range(ud) {
            out[9] = t0 + ud;
        }
        if in_range(sd) {
            out[10] = t0 + sd;
        }
    }
    if jd_ok && d1[jd].abs() >= 1e-12 {
        let fmin_right = f[jp..].iter().cloned().fold(f64::INFINITY, f64::min);
        let dd = tau_of(jd) + (fmax - f[jd]) / d1[jd];
        let rd = tau_of(jd) + (fmin_right - f[jd]) / d1[jd];
        if in_range(dd) {
            out[11] = t0 + dd;
        }
        if in_range(rd) {
            out[12] = t0 + rd;
        }
    }

    // Zhang's curvature
    let zh = zhang_dates(params, kind, first - t0, n, jp, fmax - fmin, &d1);
    for (o, v) in out[13..17].iter_mut().zip(zh) {
        *o = t0 + v;
    }

    out[17] = out[3] - out[2];
    if jp > 0 && jp < n - 1 {
        // golden-section search for the maximum within a day of the grid's
        // (Zeit's choice: a parabola through three days is biased on a peak
        // whose two sides differ)
        let (mut a, mut b) = (tau_of(jp) - 1.0, tau_of(jp) + 1.0);
        let gr = 0.5 * (5f64.sqrt() - 1.0);
        let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
        let (mut fc, mut fd) = (value(kind, params, c), value(kind, params, d));
        for _ in 0..40 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = value(kind, params, c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = value(kind, params, d);
            }
        }
        out[18] = t0 + 0.5 * (a + b);
    }
    out
}

/// Greenup, Maturity, Senescence and Dormancy by the rate of change of the
/// curvature of the curve scaled to [0, 1], on a grid of a quarter of a day.
fn zhang_dates(params: &[f64], kind: CurveType, tau_first: f64, n: usize, jp: usize, range: f64, d1: &[f64]) -> [f64; 4] {
    let mut out = [f64::NAN; 4];
    let step = 0.25;
    let m = 4 * (n - 1) + 1;
    let kp_idx = 4 * jp;
    let s_of = |k: usize| tau_first + step * k as f64;
    // the slope at the whole days is the daily grid's `d1`
    let g: Vec<f64> = (0..m).map(|k| if k % 4 == 0 { d1[k / 4] } else { slope(kind, params, s_of(k)) } / range).collect();
    let kappa: Vec<f64> = (0..m)
        .map(|k| {
            let (lo, hi) = (k.saturating_sub(1), (k + 1).min(m - 1));
            let d2 = (g[hi] - g[lo]) / (step * (hi - lo) as f64);
            {
                let q = 1.0 + g[k] * g[k];
                d2 / (q * q.sqrt())
            }
        })
        .collect();
    let kd: Vec<f64> = (0..m)
        .map(|k| {
            let (lo, hi) = (k.saturating_sub(1), (k + 1).min(m - 1));
            (kappa[hi] - kappa[lo]) / (step * (hi - lo) as f64)
        })
        .collect();
    if kd.iter().any(|v| !v.is_finite()) {
        return out;
    }
    let ji = arg_ext(&g, 0, kp_idx, true);
    let jk = arg_ext(&g, kp_idx, m - 1, false);
    // (window start, window end, maximum?, output index)
    let windows = [(0, ji, true, 0), (ji, kp_idx, true, 1), (kp_idx, jk, false, 2), (jk, m - 1, false, 3)];
    for (lo, hi, max, o) in windows {
        if hi <= lo + 1 {
            continue;
        }
        let k = arg_ext(&kd, lo, hi, max);
        if k == lo || k == hi {
            continue;
        }
        out[o] = s_of(k) + step * parabola_offset(&kd, k);
    }
    // times relative to t_segment[0]: the caller adds the origin
    out
}

/// Weighted R^2 (the NSE form, `1 - SSres / SStot`, which can be negative) and
/// weighted RMSE of `yfit` against `y` over the points with `w > 0` and finite
/// `y` and `yfit`. NaN if no weight, or `SStot` is 0 (R^2 only).
pub fn compute_gof(y: &[f64], yfit: &[f64], w: &[f64]) -> (f64, f64) {
    let (mut sw, mut swy) = (0.0, 0.0);
    for i in 0..y.len().min(yfit.len()).min(w.len()) {
        if w[i] > 0.0 && y[i].is_finite() && yfit[i].is_finite() {
            sw += w[i];
            swy += w[i] * y[i];
        }
    }
    if !(sw > 0.0) {
        return (f64::NAN, f64::NAN);
    }
    let ybar = swy / sw;
    let (mut ssr, mut sst) = (0.0, 0.0);
    for i in 0..y.len().min(yfit.len()).min(w.len()) {
        if w[i] > 0.0 && y[i].is_finite() && yfit[i].is_finite() {
            ssr += w[i] * (y[i] - yfit[i]).powi(2);
            sst += w[i] * (y[i] - ybar).powi(2);
        }
    }
    let rmse = (ssr / sw).sqrt();
    let r2 = if sst > 0.0 { 1.0 - ssr / sst } else { f64::NAN };
    (r2, rmse)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BECK_P: [f64; 6] = [0.1, 0.7, 50.0, 0.1, 250.0, 0.1];

    fn seg(t0: f64, t1: f64) -> Vec<f64> {
        let mut v = Vec::new();
        let mut t = t0;
        while t <= t1 {
            v.push(t);
            t += 8.0;
        }
        v
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
    }

    #[test]
    fn closed_forms_of_a_double_logistic() {
        // positions are days since t_segment[0] = 0, so the dates equal tau
        let t = seg(0.0, 364.0);
        let m = extract_metrics(&BECK_P, CurveType::Beck, &t);
        close(m[0], 36.137, 0.05, "TRS2 sos");
        close(m[1], 263.863, 0.05, "TRS2 eos");
        close(m[2], 50.0, 0.05, "TRS5 sos");
        close(m[3], 250.0, 0.05, "TRS5 eos");
        close(m[4], 54.055, 0.1, "TRS6 sos");
        close(m[5], 245.945, 0.1, "TRS6 eos");
        close(m[6], 50.0, 0.2, "DER sos");
        close(m[7], 150.0, 0.5, "DER pos");
        close(m[8], 250.0, 0.2, "DER eos");
        close(m[9], 30.0, 0.3, "UD");
        close(m[10], 70.0, 0.3, "SD");
        close(m[11], 230.0, 0.3, "DD");
        close(m[12], 270.0, 0.3, "RD");
        close(m[13], 27.08, 0.5, "Greenup");
        close(m[14], 72.92, 0.5, "Maturity");
        close(m[15], 227.08, 0.5, "Senescence");
        close(m[16], 272.92, 0.5, "Dormancy");
        close(m[17], 200.0, 0.1, "LOS");
        assert!(m[18] >= 100.0 && m[18] <= 200.0, "POP {}", m[18]);
    }

    #[test]
    fn dates_follow_the_origin_of_the_segment() {
        // the same curve, the segment starting at day 1000: dates shift by 1000
        let t = seg(1000.0, 1364.0);
        let m = extract_metrics(&BECK_P, CurveType::Beck, &t);
        close(m[2], 1050.0, 0.05, "TRS5 sos");
        close(m[3], 1250.0, 0.05, "TRS5 eos");
        close(m[9], 1030.0, 0.3, "UD");
        // a segment starting between two days: the grid starts at the next day
        let t = seg(0.5, 364.5);
        let m = extract_metrics(&BECK_P, CurveType::Beck, &t);
        close(m[2], 50.5, 0.05, "TRS5 sos (half-day origin)");
    }

    #[test]
    fn asymmetric_gaussian_peak_and_thresholds() {
        let p = [150.0, 0.0, 0.6, 90.0, 2.0, 110.0, 3.0];
        let t = seg(0.0, 364.0);
        let m = extract_metrics(&p, CurveType::Ag, &t);
        close(m[18], 150.0, 0.3, "POP");
        close(m[2], 150.0 - 90.0 * 2f64.ln().powf(0.5), 0.3, "TRS5 sos");
        close(m[3], 150.0 + 110.0 * 2f64.ln().powf(1.0 / 3.0), 0.3, "TRS5 eos");
        close(m[7], 150.0, 0.5, "DER pos");
    }

    #[test]
    fn product_logistic_has_the_same_dates_when_apart() {
        let t = seg(0.0, 364.0);
        let mb = extract_metrics(&BECK_P, CurveType::Beck, &t);
        let md = extract_metrics(&BECK_P, CurveType::Dl, &t);
        for k in 0..N_METRICS {
            close(mb[k], md[k], 0.1, METRIC_NAMES[k]);
        }
    }

    #[test]
    fn missing_metrics_are_nan() {
        let t = seg(0.0, 364.0);
        // a segment too short
        assert!(extract_metrics(&BECK_P, CurveType::Beck, &seg(0.0, 7.0)).iter().all(|v| v.is_nan()));
        // NaN parameters, wrong length
        let mut bad = BECK_P;
        bad[3] = f64::NAN;
        assert!(extract_metrics(&bad, CurveType::Beck, &t).iter().all(|v| v.is_nan()));
        assert!(extract_metrics(&BECK_P[..5], CurveType::Beck, &t).iter().all(|v| v.is_nan()));
        // a flat curve (mn = mx)
        let flat = [0.5, 0.5, 50.0, 0.1, 250.0, 0.1];
        assert!(extract_metrics(&flat, CurveType::Beck, &t).iter().all(|v| v.is_nan()));
        // starts high: sos = -40 puts r(0) at 0.98 for a 0.1 rate: no rising crossing at all
        let high = [0.1, 0.7, -40.0, 0.1, 250.0, 0.1];
        let m = extract_metrics(&high, CurveType::Beck, &t);
        assert!(m[0].is_nan() && m[2].is_nan() && m[4].is_nan(), "{m:?}");
        assert!(m[3].is_finite());
        // r(0) = 0.3: TRS2 sos is NaN, TRS5 and TRS6 are finite. sos with
        // sigma(0.1 (0 - sos)) = 0.3 -> sos = 8.47 ... but the rise is cut: use a wider curve
        let cut = [0.0, 1.0, -(0.3f64 / 0.7).ln() / 0.1, 0.1, 250.0, 0.1];
        let m = extract_metrics(&cut, CurveType::Beck, &t);
        assert!(m[0].is_nan(), "TRS2 sos {}", m[0]);
        assert!(m[2].is_finite() && m[4].is_finite(), "{m:?}");
        // the season cut at its end: the curve is still high at the last day
        let late = [0.1, 0.7, 50.0, 0.1, 500.0, 0.1];
        let m = extract_metrics(&late, CurveType::Beck, &t);
        assert!(m[1].is_nan() && m[3].is_nan() && m[5].is_nan(), "{m:?}");
        assert!(m[17].is_nan());
        assert!(m[2].is_finite());
    }

    #[test]
    fn gof_forms() {
        let y = [1.0, 2.0, 3.0, 4.0, f64::NAN];
        let yf = [1.0, 2.0, 3.0, 4.0, 9.0];
        let w = [1.0, 1.0, 1.0, 1.0, 1.0];
        let (r2, rmse) = compute_gof(&y, &yf, &w);
        close(r2, 1.0, 1e-15, "r2");
        close(rmse, 0.0, 1e-15, "rmse");
        // NSE: mean prediction gives 0, worse than the mean is negative
        let (r2, _) = compute_gof(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0], &[1.0; 3]);
        close(r2, 0.0, 1e-15, "mean predictor");
        let (r2, rmse) = compute_gof(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], &[1.0; 3]);
        close(r2, -3.0, 1e-12, "reversed");
        close(rmse, (8.0f64 / 3.0).sqrt(), 1e-12, "rmse");
        // weights: a zero-weight point does not count
        let (r2, rmse) = compute_gof(&[1.0, 2.0, 3.0, 100.0], &[1.0, 2.0, 3.0, 0.0], &[1.0, 1.0, 1.0, 0.0]);
        close(r2, 1.0, 1e-15, "w=0 r2");
        close(rmse, 0.0, 1e-15, "w=0 rmse");
        let (r2, rmse) = compute_gof(&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]);
        assert!(r2.is_nan());
        close(rmse, 0.0, 0.0, "constant rmse");
        let (r2, rmse) = compute_gof(&[1.0], &[1.0], &[0.0]);
        assert!(r2.is_nan() && rmse.is_nan());
    }
}
