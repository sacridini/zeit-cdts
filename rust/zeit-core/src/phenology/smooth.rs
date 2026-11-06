//! Smoothing of vegetation-index series: weighted Whittaker with upper-envelope
//! iterations, the weight update `wtsm`, and HANTS.
//!
//! Implemented from the papers (Eilers 2003; Kong et al. 2019, 2022; Jonsson &
//! Eklundh 2002, 2004; Roerink et al. 2000; Zhou et al. 2015), without any
//! third-party code. Where a paper does not fix a number or a rule, the rule is
//! Zeit's own and is marked "Zeit's rule" below.

// The dense linear algebra and the tests index several arrays at once.
#![allow(clippy::needless_range_loop)]

use crate::whittaker;

/// Smallest weight a down-weighted observation gets (Zeit's rule): it keeps the
/// system non-singular and lets the curve still feel the observation.
pub const W_MIN: f64 = 0.05;

const YEAR: f64 = 365.25;

fn is_valid(y: f64, w: f64) -> bool {
    y.is_finite() && w.is_finite() && w > 0.0
}

/// (min, max) of the valid observations, None when there are none.
fn valid_range(y: &[f64], w: &[f64]) -> Option<(f64, f64)> {
    let mut r: Option<(f64, f64)> = None;
    for i in 0..y.len() {
        if is_valid(y[i], w[i]) {
            r = Some(match r {
                None => (y[i], y[i]),
                Some((lo, hi)) => (lo.min(y[i]), hi.max(y[i])),
            });
        }
    }
    r
}

fn count_valid(y: &[f64], w: &[f64]) -> usize {
    (0..y.len()).filter(|&i| is_valid(y[i], w[i])).count()
}

/// True when the valid observations are all (nearly) the same value.
fn is_constant(y: &[f64], w: &[f64]) -> bool {
    let Some((lo, hi)) = valid_range(y, w) else { return false };
    let (mut s, mut n) = (0.0, 0.0);
    for i in 0..y.len() {
        if is_valid(y[i], w[i]) {
            s += y[i];
            n += 1.0;
        }
    }
    hi - lo <= 1e-12 * (s / n).abs().max(1.0)
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Order of the penalty (Eilers 2003; the Whittaker solver here is d = 2).
const D: usize = 2;

/// Whittaker smoother of `y` with weights `w` at positions `x` (Eilers 2003).
/// None with fewer than `d + 2 = 4` valid observations or on a failed
/// factorisation; a constant series gives the constant. `lambda <= 0` is taken
/// as 1e-9 (Zeit's rule).
fn smooth_x(x: &[f64], y: &[f64], w: &[f64], lambda: f64) -> Option<Vec<f64>> {
    if y.len() != w.len() || y.len() != x.len() || count_valid(y, w) < D + 2 {
        return None;
    }
    if is_constant(y, w) {
        let c = valid_range(y, w)?.0;
        return Some(vec![c; y.len()]);
    }
    let lambda = if lambda.is_finite() && lambda > 0.0 { lambda } else { 1e-9 };
    let z = whittaker::whittaker(x, y, w, lambda).ok()?;
    z.iter().all(|v| v.is_finite()).then_some(z)
}

/// Whittaker smoother on the observation index (the dates are taken as even).
/// The weights `w` are in [0, 1]; NaN in `y` or `w <= 0` marks a gap, which the
/// penalty fills. None when fewer than 4 observations are valid.
pub fn whittaker_weighted(y: &[f64], w: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let x: Vec<f64> = (0..y.len()).map(|i| i as f64).collect();
    smooth_x(&x, y, w, lambda)
}

/// `lambda` scaled to the number of observations per year, so the cut-off
/// frequency in cycles per year stays where `lambda` was tuned for 23 per year
/// (a d = 2 filter responds as 1 / (1 + lambda (2 sin(w/2))^4)). Zeit's rule,
/// optional.
pub fn scale_lambda(lambda: f64, nptperyear: usize) -> f64 {
    lambda * (nptperyear.max(1) as f64 / 23.0).powi(4)
}

/// Dates mapped onto a regular grid of step `dt` (Zeit's rule for uneven dates).
struct Grid {
    /// Grid cell of each observation.
    cell: Vec<usize>,
    len: usize,
}

/// The regular grid of the median step when the dates are uneven (largest step
/// above 1.5 times the median); None when the index can be used as it is, when
/// the dates do not increase, or when the grid would be absurdly long.
fn regular_grid(t: &[f64]) -> Option<Grid> {
    let n = t.len();
    if n < 2 {
        return None;
    }
    let mut steps: Vec<f64> = t.windows(2).map(|p| p[1] - p[0]).collect();
    if steps.iter().any(|s| !(*s > 0.0) || !s.is_finite()) {
        return None;
    }
    let max = steps.iter().cloned().fold(0.0, f64::max);
    let dt = median(&mut steps);
    if max <= 1.5 * dt {
        return None;
    }
    let len = ((t[n - 1] - t[0]) / dt).round() as usize + 1;
    if len > 50 * n {
        return None;
    }
    let cell = t.iter().map(|v| (((v - t[0]) / dt).round() as usize).min(len - 1)).collect();
    Some(Grid { cell, len })
}

fn increasing(t: &[f64]) -> bool {
    t.windows(2).all(|p| p[1] > p[0]) && t.iter().all(|v| v.is_finite())
}

/// Whittaker smoother at the dates `t`. Even dates (largest step within 1.5
/// times the median) are smoothed on the index; uneven ones are put on a
/// regular grid of the median step with weight 0 in the holes, smoothed, and
/// read back at the original dates (Zeit's rule). None when `t` does not
/// increase.
pub fn whittaker_dates(t: &[f64], y: &[f64], w: &[f64], lambda: f64) -> Option<Vec<f64>> {
    if t.len() != y.len() || !increasing(t) {
        return None;
    }
    let Some(g) = regular_grid(t) else { return whittaker_weighted(y, w, lambda) };
    let (gy, gw) = to_grid(&g, y, w);
    let x: Vec<f64> = (0..g.len).map(|i| i as f64).collect();
    let gz = smooth_x(&x, &gy, &gw, lambda)?;
    Some(g.cell.iter().map(|&c| gz[c]).collect())
}

/// `y` and `w` on the grid; where two observations share a cell the one with
/// the larger weight stays.
fn to_grid(g: &Grid, y: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut gy = vec![f64::NAN; g.len];
    let mut gw = vec![0.0; g.len];
    for i in 0..y.len() {
        if is_valid(y[i], w[i]) && w[i] > gw[g.cell[i]] {
            gy[g.cell[i]] = y[i];
            gw[g.cell[i]] = w[i];
        }
    }
    (gy, gw)
}

/// Asymmetric bisquare update of the weights after a fit `z` (Kong et al. 2019
/// in spirit; the formula is Zeit's rule). Clouds and shadows only lower a
/// vegetation index, so a point on or above the curve keeps its original
/// weight `w0`; a point below gets `w0 (1 - u^2)^2` with
/// `u = |r| / (4.685 s)` and `s` the robust scale (1.4826 times the median) of
/// the negative residuals, `W_MIN` at the least, never above `w0`.
pub fn bisquare_update(y: &[f64], z: &[f64], w0: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut neg: Vec<f64> = (0..n)
        .filter(|&i| is_valid(y[i], w0[i]) && z[i].is_finite() && y[i] - z[i] < 0.0)
        .map(|i| (y[i] - z[i]).abs())
        .collect();
    let amp = valid_range(y, w0).map_or(0.0, |(lo, hi)| hi - lo);
    let s = (1.4826 * median(&mut neg)).max(1e-12 * amp).max(f64::MIN_POSITIVE);
    (0..n)
        .map(|i| {
            if !is_valid(y[i], w0[i]) || !z[i].is_finite() {
                return 0.0;
            }
            let r = y[i] - z[i];
            if r >= 0.0 {
                return w0[i];
            }
            let u = r.abs() / (4.685 * s);
            let w = if u < 1.0 { w0[i] * (1.0 - u * u).powi(2) } else { W_MIN };
            w.max(W_MIN).min(w0[i])
        })
        .collect()
}

/// The result of `whittaker_envelope`.
#[derive(Clone, Debug)]
pub struct EnvelopeFit {
    /// The smoothed series (at every date, also in the gaps).
    pub z: Vec<f64>,
    /// The weights of the last pass (0 for the gaps).
    pub weights: Vec<f64>,
    /// Passes done after the first fit.
    pub iterations: usize,
}

/// Whittaker with upper-envelope iterations (Kong et al. 2019, "wWHIT"): fit,
/// lower the weights of the points below the curve (`bisquare_update`), fit
/// again, `iter` times at most (3 by default in Zeit; the papers say a few).
/// The weights start at `w0` (the QC) and never rise above it. It stops early
/// when the curve moves less than 1e-4 of the amplitude. `t` are the dates
/// (None: the index); see `whittaker_dates` for uneven dates. `iter = 0` is the
/// plain weighted fit. None when the fit is impossible (fewer than 4 valid
/// observations, dates not increasing).
pub fn whittaker_envelope(
    t: Option<&[f64]>,
    y: &[f64],
    w0: &[f64],
    lambda: f64,
    iter: usize,
) -> Option<EnvelopeFit> {
    let n = y.len();
    if w0.len() != n || t.is_some_and(|t| t.len() != n || !increasing(t)) {
        return None;
    }
    if count_valid(y, w0) < D + 2 {
        return None;
    }
    let grid = t.and_then(regular_grid);
    let (gy, gw0, x) = match &grid {
        Some(g) => {
            let (a, b) = to_grid(g, y, w0);
            (a, b, (0..g.len).map(|i| i as f64).collect::<Vec<f64>>())
        }
        None => (y.to_vec(), w0.to_vec(), (0..n).map(|i| i as f64).collect()),
    };
    let amp = valid_range(&gy, &gw0).map_or(0.0, |(lo, hi)| hi - lo);
    let mut w = gw0.clone();
    let mut z = smooth_x(&x, &gy, &w, lambda)?;
    let mut done = 0;
    for _ in 0..iter {
        w = bisquare_update(&gy, &z, &gw0);
        let znew = smooth_x(&x, &gy, &w, lambda)?;
        let delta = z.iter().zip(&znew).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        z = znew;
        done += 1;
        if delta < 1e-4 * amp {
            break;
        }
    }
    Some(match grid {
        Some(g) => EnvelopeFit {
            z: g.cell.iter().map(|&c| z[c]).collect(),
            weights: g.cell.iter().map(|&c| w[c]).collect(),
            iterations: done,
        },
        None => EnvelopeFit { z, weights: w, iterations: done },
    })
}

/// Marks the interpolated values that deserve little trust (Zeit's rule): the
/// points inside a run of more than `nptperyear / 2` consecutive observations
/// without data (NaN or weight <= 0).
pub fn low_confidence_mask(y: &[f64], w: &[f64], nptperyear: usize) -> Vec<bool> {
    let n = y.len();
    let mut mask = vec![false; n];
    let limit = nptperyear / 2;
    let mut i = 0;
    while i < n {
        if is_valid(y[i], w[i]) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !is_valid(y[j], w[j]) {
            j += 1;
        }
        if j - i > limit {
            mask[i..j].fill(true);
        }
        i = j;
    }
    mask
}

/// Weight update of TIMESAT's kind (Jonsson & Eklundh 2002, 2004: points below
/// the fitted curve lose weight, the others keep it). The rule is Zeit's own:
/// with `r = yfit - y` (positive below the curve) and `m` the largest `|r|` of
/// the valid points in a centred window of `nptperyear` observations,
/// `d = clip(r / m, -1, 1)`; a point on or above the curve keeps `w`, one
/// below gets `w (1 - d)^p`, `p = 1 + 3 wfact iter`, so the point farthest below
/// goes to `W_MIN` and `p` grows with the iteration (the "adaptation strength").
/// `iter` is the current iteration, 1-based; `wfact` in (0, 1] (0.5 by
/// default). The result is in [W_MIN, w] for valid points (or `w` if it is
/// already below `W_MIN`), 0 for NaN in `y` and for `w <= 0`.
pub fn wtsm(y: &[f64], yfit: &[f64], w: &[f64], iter: usize, nptperyear: usize, wfact: f64) -> Vec<f64> {
    let n = y.len();
    let p = 1.0 + 3.0 * wfact * iter.max(1) as f64;
    let half = (nptperyear.max(1)) / 2;
    let ok = |i: usize| is_valid(y[i], w[i]) && yfit[i].is_finite();
    (0..n)
        .map(|i| {
            if !y[i].is_finite() || !(w[i] > 0.0) {
                return 0.0;
            }
            if !ok(i) {
                return w[i];
            }
            let (lo, hi) = (i.saturating_sub(half), (i + half).min(n - 1));
            let m = (lo..=hi).filter(|&j| ok(j)).map(|j| (yfit[j] - y[j]).abs()).fold(0.0, f64::max);
            if !(m > 0.0) {
                return w[i];
            }
            let d = ((yfit[i] - y[i]) / m).clamp(-1.0, 1.0);
            if d <= 0.0 {
                return w[i];
            }
            ((w[i] * (1.0 - d).powf(p)).max(W_MIN)).min(w[i])
        })
        .collect()
}

/// Options of `hants_with`.
#[derive(Clone, Debug)]
pub struct HantsOptions {
    /// Frequencies in cycles per `period`.
    pub frequencies: Vec<f64>,
    /// The base period in the unit of the dates (365.25 days).
    pub period: f64,
    /// Fit error tolerance `fet`, in the unit of `y`: a point farther below the
    /// curve than this is suppressed. None or not positive: 0.1 of the
    /// amplitude of the valid observations (Zeit's rule).
    pub threshold: Option<f64>,
    /// Degree of over-determinedness (Roerink et al. 2000): the points kept
    /// number at least the coefficients plus this (5).
    pub dod: usize,
    /// Zhou et al. 2015 style: instead of removing a point, multiply its weight
    /// by `W_MIN` (Zeit's rule when initial weights are given).
    pub soft: bool,
    /// Ridge added to the normal equations (Zhou et al. 2015), relative to
    /// their diagonal.
    pub ridge: f64,
}

impl Default for HantsOptions {
    fn default() -> Self {
        HantsOptions {
            frequencies: vec![1.0, 2.0, 3.0],
            period: YEAR,
            threshold: None,
            dod: 5,
            soft: false,
            ridge: 1e-6,
        }
    }
}

/// HANTS (Roerink et al. 2000) with `num_frequencies` harmonics of one year
/// (1, 2, .. cycles per year), suppressing the points below the curve. `t` in
/// days. `threshold` is the fit error tolerance in the unit of `y` (not
/// positive: 0.1 of the amplitude). With `initial_weights` (the QC) the fit is
/// weighted and the suppressed points are down-weighted by `W_MIN` rather than
/// removed (Zhou et al. 2015); without them it is the classic removal. The
/// curve at all the dates, also where `y` is NaN; all NaN with fewer than
/// `2 nf + 2` valid observations or dates that do not increase.
pub fn hants(
    y: &[f64],
    t: &[f64],
    num_frequencies: usize,
    threshold: f64,
    initial_weights: Option<&[f64]>,
) -> Vec<f64> {
    let opts = HantsOptions {
        frequencies: (1..=num_frequencies).map(|k| k as f64).collect(),
        threshold: Some(threshold),
        soft: initial_weights.is_some(),
        ..HantsOptions::default()
    };
    hants_with(y, t, &opts, initial_weights)
}

/// `hants` with every option. Frequencies above the Nyquist limit of the median
/// step are dropped.
pub fn hants_with(y: &[f64], t: &[f64], opts: &HantsOptions, initial_weights: Option<&[f64]>) -> Vec<f64> {
    let n = y.len();
    let nan = vec![f64::NAN; n];
    if t.len() != n || !increasing(t) || initial_weights.is_some_and(|w| w.len() != n) {
        return nan;
    }
    let w0: Vec<f64> = (0..n)
        .map(|i| {
            let w = initial_weights.map_or(1.0, |w| w[i]);
            if is_valid(y[i], w) { w } else { 0.0 }
        })
        .collect();
    let Some((lo, hi)) = valid_range(y, &w0) else { return nan };
    let mut steps: Vec<f64> = t.windows(2).map(|p| p[1] - p[0]).collect();
    let nyquist = if steps.is_empty() { f64::INFINITY } else { 0.5 * opts.period / median(&mut steps) };
    let freqs: Vec<f64> = opts.frequencies.iter().cloned().filter(|f| *f > 0.0 && *f <= nyquist).collect();
    let ncoef = 1 + 2 * freqs.len();
    let nvalid = w0.iter().filter(|w| **w > 0.0).count();
    if nvalid < ncoef + 1 {
        return nan;
    }
    if is_constant(y, &w0) {
        return vec![lo; n];
    }
    let fet = match opts.threshold {
        Some(v) if v.is_finite() && v > 0.0 => v,
        _ => 0.1 * (hi - lo),
    };
    let basis = |ti: f64| -> Vec<f64> {
        let x = (ti - t[0]) / opts.period;
        let mut b = Vec::with_capacity(ncoef);
        b.push(1.0);
        for f in &freqs {
            let a = 2.0 * std::f64::consts::PI * f * x;
            b.push(a.cos());
            b.push(a.sin());
        }
        b
    };
    let rows: Vec<Vec<f64>> = t.iter().map(|&ti| basis(ti)).collect();
    let mut wcur = w0.clone();
    let mut active: Vec<bool> = w0.iter().map(|w| *w > 0.0).collect();
    let mut nactive = nvalid;
    let mut coef = vec![0.0; ncoef];
    for _ in 0..=n {
        match solve_fit(&rows, y, &wcur, opts.ridge) {
            Some(c) => coef = c,
            None => break,
        }
        let mut worst: Option<(usize, f64)> = None;
        for i in 0..n {
            if !active[i] {
                continue;
            }
            let e = y[i] - dot(&rows[i], &coef);
            if e < -fet && worst.is_none_or(|(_, we)| e < we) {
                worst = Some((i, e));
            }
        }
        let Some((i, _)) = worst else { break };
        if nactive - 1 < ncoef + opts.dod {
            break;
        }
        if opts.soft {
            wcur[i] *= W_MIN;
        } else {
            wcur[i] = 0.0;
        }
        active[i] = false;
        nactive -= 1;
    }
    rows.iter().map(|r| dot(r, &coef)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Weighted least squares by the normal equations with a small ridge and a
/// dense Cholesky factorisation (the systems are at most a dozen wide).
fn solve_fit(rows: &[Vec<f64>], y: &[f64], w: &[f64], ridge: f64) -> Option<Vec<f64>> {
    let m = rows[0].len();
    let mut a = vec![vec![0.0; m]; m];
    let mut b = vec![0.0; m];
    for i in 0..rows.len() {
        if !(w[i] > 0.0) || !y[i].is_finite() {
            continue;
        }
        for p in 0..m {
            b[p] += w[i] * rows[i][p] * y[i];
            for q in 0..=p {
                a[p][q] += w[i] * rows[i][p] * rows[i][q];
            }
        }
    }
    for p in 0..m {
        a[p][p] *= 1.0 + ridge;
        a[p][p] += 1e-12;
    }
    let mut l = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if !(s > 1e-14) {
                    return None;
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut u = vec![0.0; m];
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * u[k];
        }
        u[i] = s / l[i][i];
    }
    let mut c = vec![0.0; m];
    for i in (0..m).rev() {
        let mut s = u[i];
        for k in i + 1..m {
            s -= l[k][i] * c[k];
        }
        c[i] = s / l[i][i];
    }
    Some(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenology::testutil::{dates16, Rng};

    const TAU: f64 = std::f64::consts::TAU;

    fn truth(t: f64) -> f64 {
        0.5 + 0.3 * (TAU * (t - 80.0) / YEAR).sin()
    }

    /// Three years of 16-day composites with noise and `cloud` fraction of
    /// points lowered by U(0.1, 0.4); returns (t, y, truth, is_cloud).
    fn series(seed: u64, cloud: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
        let t = dates16(3);
        let mut rng = Rng::new(seed);
        let tr: Vec<f64> = t.iter().map(|&d| truth(d)).collect();
        let mut y: Vec<f64> = tr.iter().map(|v| v + 0.02 * rng.normal()).collect();
        let mut is_cloud = vec![false; y.len()];
        for i in 0..y.len() {
            if rng.uniform() < cloud {
                y[i] -= 0.1 + 0.3 * rng.uniform();
                is_cloud[i] = true;
            }
        }
        (t, y, tr, is_cloud)
    }

    fn rmse(a: &[f64], b: &[f64], keep: &[bool]) -> f64 {
        let (mut s, mut n) = (0.0, 0.0);
        for i in 0..a.len() {
            if keep[i] {
                s += (a[i] - b[i]).powi(2);
                n += 1.0;
            }
        }
        (s / n).sqrt()
    }

    #[test]
    fn envelope_beats_one_step_on_clouds() {
        let (t, y, tr, cloud) = series(7, 0.15);
        let clean: Vec<bool> = cloud.iter().map(|c| !c).collect();
        let w = vec![1.0; y.len()];
        let one = whittaker_weighted(&y, &w, 15.0).unwrap();
        let env = whittaker_envelope(Some(&t), &y, &w, 15.0, 3).unwrap();
        let (e1, e3) = (rmse(&one, &tr, &clean), rmse(&env.z, &tr, &clean));
        assert!(e3 <= 0.04, "envelope rmse {e3}");
        assert!(e3 < e1, "envelope {e3} vs one step {e1}");
        assert!(env.weights.iter().all(|w| (0.0..=1.0).contains(w)));
    }

    #[test]
    fn wtsm_lowers_the_clouds_only() {
        let (_, y, _, cloud) = series(11, 0.15);
        let w0 = vec![1.0; y.len()];
        let z = whittaker_weighted(&y, &w0, 15.0).unwrap();
        // The fit of an envelope, so the clouds sit well below it.
        let env = whittaker_envelope(None, &y, &w0, 15.0, 3).unwrap();
        let w = wtsm(&y, &env.z, &w0, 2, 23, 0.5);
        let _ = z;
        for i in 0..y.len() {
            assert!((0.0..=1.0).contains(&w[i]));
            if env.z[i] <= y[i] {
                assert_eq!(w[i], w0[i], "above the curve keeps its weight");
            }
            if cloud[i] && env.z[i] - y[i] > 0.15 {
                assert!(w[i] <= 0.3, "cloud {i} kept {}", w[i]);
            }
        }
        assert!(w.iter().any(|&v| v < 0.3));
    }

    #[test]
    fn wtsm_edges() {
        let y = [1.0, f64::NAN, 0.5, 0.9];
        let fit = [1.0, 1.0, 1.0, 1.0];
        let w = wtsm(&y, &fit, &[1.0, 1.0, 0.0, 1.0], 1, 4, 0.5);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        assert!(w[3] >= W_MIN && w[3] <= 1.0);
        // all on the curve: unchanged
        assert_eq!(wtsm(&[1.0; 5], &[1.0; 5], &[0.7; 5], 2, 5, 0.5), vec![0.7; 5]);
    }

    #[test]
    fn hants_clean_and_clouded() {
        let t = dates16(3);
        let tr: Vec<f64> = t.iter().map(|&d| truth(d)).collect();
        let mut rng = Rng::new(3);
        let clean: Vec<f64> = tr.iter().map(|v| v + 0.02 * rng.normal()).collect();
        let fit = hants(&clean, &t, 3, 0.1 * 0.6, None);
        let err = fit.iter().zip(&tr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(err < 0.03, "clean max error {err}");

        let (t, y, tr, _) = series(5, 0.15);
        let fet = 0.1 * 0.6;
        let with = hants(&y, &t, 3, fet, None);
        let without = hants(&y, &t, 3, 1e9, None);
        let maxerr = |f: &[f64]| f.iter().zip(&tr).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(maxerr(&with) <= 0.6 * maxerr(&without), "{} vs {}", maxerr(&with), maxerr(&without));
        // weighted version: the same idea, down-weighting instead of removing
        let w0 = vec![1.0; y.len()];
        let soft = hants(&y, &t, 3, fet, Some(&w0));
        assert!(maxerr(&soft) <= 0.6 * maxerr(&without));
    }

    #[test]
    fn hants_special_cases() {
        let t = dates16(1);
        assert!(hants(&[f64::NAN; 23], &t, 3, 0.1, None).iter().all(|v| v.is_nan()));
        assert!(hants(&[1.0, 2.0, 3.0], &[0.0, 1.0, 2.0], 3, 0.1, None).iter().all(|v| v.is_nan()));
        let c = hants(&[0.4; 23], &t, 3, 0.1, None);
        assert!(c.iter().all(|v| (v - 0.4).abs() < 1e-12));
        // dates not increasing
        assert!(hants(&[1.0; 4], &[0.0, 2.0, 1.0, 3.0], 1, 0.1, None).iter().all(|v| v.is_nan()));
        // a gap in y still yields the curve everywhere
        let mut y: Vec<f64> = t.iter().map(|&d| truth(d)).collect();
        y[5] = f64::NAN;
        assert!(hants(&y, &t, 2, 0.05, None).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn special_cases_of_whittaker() {
        assert!(whittaker_weighted(&[1.0, 2.0], &[1.0, 1.0], 10.0).is_none());
        assert!(whittaker_weighted(&[f64::NAN; 30], &[1.0; 30], 10.0).is_none());
        assert!(whittaker_weighted(&[1.0; 30], &[0.0; 30], 10.0).is_none());
        let c = whittaker_weighted(&[0.3; 30], &[1.0; 30], 10.0).unwrap();
        assert!(c.iter().all(|v| (v - 0.3).abs() < 1e-12));
        // lambda <= 0 does not panic and follows the data
        let y: Vec<f64> = (0..30).map(|i| (i as f64 * 0.7).sin()).collect();
        let z = whittaker_weighted(&y, &[1.0; 30], 0.0).unwrap();
        assert!(z.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-6));
        // dates not increasing
        assert!(whittaker_dates(&[0.0, 2.0, 1.0, 3.0, 4.0], &[1.0; 5], &[1.0; 5], 1.0).is_none());
        assert!(whittaker_envelope(Some(&[0.0, 2.0, 1.0, 3.0, 4.0]), &[1.0; 5], &[1.0; 5], 1.0, 3).is_none());
        // mixed weights: w <= 0 behaves like NaN
        let mut y: Vec<f64> = (0..40).map(|i| 0.5 + 0.2 * (i as f64 * 0.3).sin()).collect();
        let mut w = vec![1.0; 40];
        w[10] = 0.0;
        y[10] = 100.0;
        let z = whittaker_weighted(&y, &w, 5.0).unwrap();
        assert!(z[10] < 1.0);
    }

    #[test]
    fn long_gap_mask() {
        let mut y = vec![1.0; 60];
        for v in &mut y[10..30] {
            *v = f64::NAN;
        }
        for v in &mut y[40..45] {
            *v = f64::NAN;
        }
        let m = low_confidence_mask(&y, &[1.0; 60], 23);
        assert!(m[10..30].iter().all(|v| *v));
        assert!(!m[40..45].iter().any(|v| *v));
        let t: Vec<f64> = (0..60).map(|i| i as f64 * 16.0).collect();
        let z = whittaker_dates(&t, &y, &[1.0; 60], 15.0).unwrap();
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn uneven_dates_use_the_regular_grid() {
        // Dates with holes of three steps, from a smooth curve.
        let t: Vec<f64> = (0..69).filter(|i| i % 11 < 7).map(|i| i as f64 * 16.0).collect();
        let y: Vec<f64> = t.iter().map(|&d| truth(d)).collect();
        let w = vec![1.0; y.len()];
        // (a small lambda, so the bias on the sinusoid's peaks stays small)
        let z = whittaker_dates(&t, &y, &w, 1.0).unwrap();
        let worst = z.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(worst < 0.02, "{worst}");
        let env = whittaker_envelope(Some(&t), &y, &w, 1.0, 3).unwrap();
        assert_eq!(env.z.len(), y.len());
        // the envelope pushes the curve up, where it is below the data
        // (without noise the robust scale is tiny, so the curve rides the top)
        let we = env.z.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(we < 0.08, "{we}");
    }

    /// Solves (W + lambda D'D) z = W y by dense Gauss elimination.
    fn dense(y: &[f64], w: &[f64], lambda: f64) -> Vec<f64> {
        let n = y.len();
        let mut a = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            a[i][i] += w[i];
            a[i][n] = w[i] * y[i];
        }
        for k in 0..n - 2 {
            let d = [1.0, -2.0, 1.0];
            for p in 0..3 {
                for q in 0..3 {
                    a[k + p][k + q] += lambda * d[p] * d[q];
                }
            }
        }
        for c in 0..n {
            let piv = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
            a.swap(c, piv);
            for r in c + 1..n {
                let f = a[r][c] / a[c][c];
                for k in c..=n {
                    a[r][k] -= f * a[c][k];
                }
            }
        }
        let mut z = vec![0.0; n];
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|k| a[i][k] * z[k]).sum();
            z[i] = (a[i][n] - s) / a[i][i];
        }
        z
    }

    #[test]
    fn matches_dense_algebra_and_limits() {
        let mut rng = Rng::new(21);
        let y: Vec<f64> = (0..30).map(|i| 0.5 + 0.3 * (i as f64 * 0.4).sin() + 0.05 * rng.normal()).collect();
        let w: Vec<f64> = (0..30).map(|i| if i % 7 == 3 { 0.0 } else { 0.2 + 0.8 * rng.uniform() }).collect();
        for lambda in [0.5, 15.0, 400.0] {
            let z = whittaker_weighted(&y, &w, lambda).unwrap();
            let d = dense(&y, &w, lambda);
            assert!(z.iter().zip(&d).all(|(a, b)| (a - b).abs() < 1e-8), "lambda {lambda}");
        }
        // huge lambda: the weighted least-squares line
        let w1 = vec![1.0; 30];
        let z = whittaker_weighted(&y, &w1, 1e8).unwrap();
        let (mx, my) = (14.5, y.iter().sum::<f64>() / 30.0);
        let slope = (0..30).map(|i| (i as f64 - mx) * (y[i] - my)).sum::<f64>()
            / (0..30).map(|i| (i as f64 - mx).powi(2)).sum::<f64>();
        for i in 0..30 {
            assert!((z[i] - (my + slope * (i as f64 - mx))).abs() < 1e-4);
        }
        // tiny lambda: the data
        let z = whittaker_weighted(&y, &w1, 1e-9).unwrap();
        assert!(z.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn scaling_of_lambda() {
        assert_eq!(scale_lambda(15.0, 23), 15.0);
        assert!(scale_lambda(15.0, 46) > 15.0 * 15.0);
    }
}
