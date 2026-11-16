//! Lagged correlation of time series with significance that accounts for their
//! autocorrelation, and the false discovery rate.
//!
//! - Pairs: the dates where both series are valid (pairwise complete); at lag
//!   `k` the pairs are `(x_t, y_{t-k})`, so a positive lag means that `y`
//!   leads. Fewer than `min_pairs` pairs give NaN.
//! - Pearson on the pairs, or Spearman: average ranks (ties) of the pairs, then
//!   Pearson on the ranks.
//! - Effective sample size (Bretherton et al. 1999, eq. 31):
//!   `n_eff = n (1 - rx ry) / (1 + rx ry)`, with `rx`, `ry` the lag-1
//!   autocorrelations of the series and `rx ry` clamped to `[0, 0.99]`.
//! - p-value: two-sided Student t with `n_eff - 2` degrees of freedom (not
//!   necessarily an integer).
//! - Best lag, with a Sidak correction for the number of lags tried.
//! - False discovery rate: Benjamini and Hochberg (1995), as used for fields
//!   by Wilks (2016).

use crate::anomaly::{civil_from_days, days_from_civil, days_in_month};
use crate::dist::t_two_sided;
use crate::mann_kendall::rankdata;
use crate::parallel;
use rayon::prelude::*;

/// The correlation coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Pearson,
    Spearman,
}

/// Which lag is the best.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Best {
    /// The largest |r|.
    Absolute,
    /// The largest r.
    Positive,
    /// The smallest r.
    Negative,
}

/// How the dates of `y` are matched to the (shifted) dates of `x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Match {
    /// The same day.
    Exact,
    /// The nearest date, within this many days.
    Nearest(f64),
    /// A date of the same calendar month (the nearest one).
    SameMonth,
    /// A date of the same year (the nearest one).
    SameYear,
}

/// The shift of `x`'s dates before they are matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lag {
    /// `k` observations of `x` back: the date `x_days[t - k]`.
    Steps(i32),
    /// The same day `k` months earlier, clamped to the end of a shorter month.
    /// With `Match::Exact` the nearest date within half the median step of
    /// `y` (at least 8 days) is taken.
    Months(i32),
}

/// The date `k` months before `d`, the day clamped to the month's length.
fn months_back(d: i64, k: i32) -> i64 {
    let (y, m, day) = civil_from_days(d);
    let total = i64::from(y) * 12 + i64::from(m) - 1 - i64::from(k);
    let (ny, nm) = (total.div_euclid(12) as i32, total.rem_euclid(12) as u32 + 1);
    days_from_civil(ny, nm, day.min(days_in_month(ny, nm)))
}

/// For each date of `x`, the index of the date of `y` it is paired with at
/// this lag, or -1. `y_days` must be ascending.
pub fn match_index(x_days: &[i64], y_days: &[i64], m: Match, lag: Lag) -> Vec<i32> {
    let ny = y_days.len();
    let mut tol_months = 0.0;
    if let (Lag::Months(_), Match::Exact) = (lag, m) {
        let mut steps: Vec<f64> = y_days.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
        steps.sort_by(|a, b| a.total_cmp(b));
        let med = if steps.is_empty() { 0.0 } else { steps[steps.len() / 2] };
        tol_months = (med / 2.0).max(8.0);
    }
    (0..x_days.len())
        .map(|t| {
            let target = match lag {
                Lag::Steps(k) => {
                    let s = t as i64 - i64::from(k);
                    if s < 0 || s >= x_days.len() as i64 {
                        return -1;
                    }
                    x_days[s as usize]
                }
                Lag::Months(k) => months_back(x_days[t], k),
            };
            let (ty, tm, _) = civil_from_days(target);
            let ok = |i: usize| -> bool {
                let yd = y_days[i];
                match m {
                    Match::Exact => {
                        if matches!(lag, Lag::Months(_)) {
                            ((yd - target) as f64).abs() <= tol_months
                        } else {
                            yd == target
                        }
                    }
                    Match::Nearest(tol) => ((yd - target) as f64).abs() <= tol,
                    Match::SameMonth => civil_from_days(yd).0 == ty && civil_from_days(yd).1 == tm,
                    Match::SameYear => civil_from_days(yd).0 == ty,
                }
            };
            // the nearest date below and above; months and years are contiguous in a sorted list, so
            // if the nearest neighbour is outside, so is everything beyond it
            let hi = y_days.partition_point(|&d| d < target);
            let below = if hi > 0 { Some(hi - 1) } else { None };
            let above = if hi < ny { Some(hi) } else { None };
            let mut best: Option<usize> = None;
            for c in [below, above].into_iter().flatten() {
                if ok(c) {
                    best = match best {
                        Some(b) if (y_days[b] - target).abs() <= (y_days[c] - target).abs() => Some(b),
                        _ => Some(c),
                    };
                }
            }
            best.map_or(-1, |b| b as i32)
        })
        .collect()
}

/// A correlation and its significance.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub r: f64,
    /// The pairs.
    pub n: usize,
    /// The effective sample size (`n` if not asked).
    pub n_eff: f64,
    pub p: f64,
}

/// Pearson's r of two equal-length slices without NaN (two passes); NaN if a
/// variance is zero.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        let (da, db) = (a - mx, b - my);
        sxy += da * db;
        sxx += da * da;
        syy += db * db;
    }
    if !(sxx > 0.0 && syy > 0.0) {
        return f64::NAN;
    }
    (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)
}

/// The lag-1 autocorrelation of a series over consecutive steps that are both
/// valid (Pearson's r of those pairs); 0 with fewer than 3 such pairs or no
/// variance.
fn lag1(x: &[f64]) -> f64 {
    let (mut n, mut sa, mut sb) = (0usize, 0.0, 0.0);
    for w in x.windows(2) {
        if w[0].is_finite() && w[1].is_finite() {
            n += 1;
            sa += w[0];
            sb += w[1];
        }
    }
    if n < 3 {
        return 0.0;
    }
    let (ma, mb) = (sa / n as f64, sb / n as f64);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for w in x.windows(2) {
        if w[0].is_finite() && w[1].is_finite() {
            let (da, db) = (w[0] - ma, w[1] - mb);
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
    }
    if saa > 0.0 && sbb > 0.0 {
        sab / (saa * sbb).sqrt()
    } else {
        0.0
    }
}

/// `n (1 - rx ry) / (1 + rx ry)` with `rx ry` clamped to `[0, 0.99]` (a NaN
/// counts as 0).
pub fn effective_n(n: f64, rx: f64, ry: f64) -> f64 {
    let prod = rx * ry;
    let prod = if prod.is_nan() { 0.0 } else { prod.clamp(0.0, 0.99) };
    n * (1.0 - prod) / (1.0 + prod)
}

/// The two-sided p-value of a correlation `r` with `n_eff` effective pairs:
/// Student t with `n_eff - 2` degrees of freedom. NaN for `n_eff <= 2`.
pub fn p_value(r: f64, n_eff: f64) -> f64 {
    if r.is_nan() || !(n_eff > 2.0) {
        return f64::NAN;
    }
    if r.abs() >= 1.0 {
        return 0.0;
    }
    let df = n_eff - 2.0;
    t_two_sided(r * (df / (1.0 - r * r)).sqrt(), df)
}

#[derive(Default)]
struct Scratch {
    px: Vec<f64>,
    py: Vec<f64>,
    idx: Vec<usize>,
    rk: Vec<f64>,
    rx: Vec<f64>,
    ry: Vec<f64>,
}

fn correlate_with(x: &[f64], y: &[f64], method: Method, effective: bool, min_pairs: usize, sc: &mut Scratch) -> Pair {
    sc.px.clear();
    sc.py.clear();
    for (a, b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            sc.px.push(*a);
            sc.py.push(*b);
        }
    }
    let n = sc.px.len();
    if n < min_pairs.max(3) {
        return Pair {
            r: f64::NAN,
            n,
            n_eff: f64::NAN,
            p: f64::NAN,
        };
    }
    let r = match method {
        Method::Pearson => pearson(&sc.px, &sc.py),
        Method::Spearman => {
            rankdata(&sc.px, &mut sc.idx, &mut sc.rk);
            sc.rx.clear();
            sc.rx.extend_from_slice(&sc.rk);
            rankdata(&sc.py, &mut sc.idx, &mut sc.rk);
            sc.ry.clear();
            sc.ry.extend_from_slice(&sc.rk);
            pearson(&sc.rx, &sc.ry)
        }
    };
    let n_eff = if effective {
        effective_n(n as f64, lag1(x), lag1(y))
    } else {
        n as f64
    };
    Pair {
        r,
        n,
        n_eff,
        p: p_value(r, n_eff),
    }
}

/// The correlation of two equal-length series (pairwise complete), with its
/// effective sample size and p-value.
pub fn correlate(x: &[f64], y: &[f64], method: Method, effective: bool, min_pairs: usize) -> Pair {
    assert_eq!(x.len(), y.len(), "the series must be aligned");
    correlate_with(x, y, method, effective, min_pairs, &mut Scratch::default())
}

/// Benjamini and Hochberg's (1995) false discovery rate procedure over the
/// valid (not NaN) p-values: the adjusted values `q` (NaN where `p` is NaN) and
/// the threshold `p*`, the largest `p_(i) <= i alpha / M`, so that the pixels
/// with `p <= p*` are significant at the rate `alpha` (0 if there is none).
pub fn fdr_bh(p: &[f64], alpha: f64) -> (Vec<f64>, f64) {
    let mut order: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let m = order.len();
    let mut q = vec![f64::NAN; p.len()];
    let mut p_star = 0.0;
    let mut running = f64::INFINITY;
    for (rank0, &i) in order.iter().enumerate().rev() {
        let rank = (rank0 + 1) as f64;
        if p_star == 0.0 && p[i] <= rank / m as f64 * alpha {
            p_star = p[i];
        }
        running = running.min(m as f64 / rank * p[i]);
        q[i] = running.min(1.0);
    }
    (q, p_star)
}

/// The series that drives the correlation.
#[derive(Clone, Copy)]
pub enum YRef<'a> {
    /// A cube of `ty` dates x the pixels of `x` (time-major).
    Cube(&'a [f64], usize),
    /// One series for every pixel.
    Series(&'a [f64]),
}

#[derive(Clone, Copy, Debug)]
pub struct LagSettings {
    pub method: Method,
    pub min_pairs: usize,
    pub effective_n: bool,
    pub best: Best,
    /// Adjust the best lag's p for the number of lags tried (Sidak).
    pub sidak: bool,
}

impl Default for LagSettings {
    fn default() -> Self {
        LagSettings {
            method: Method::Pearson,
            min_pairs: 10,
            effective_n: true,
            best: Best::Absolute,
            sidak: true,
        }
    }
}

/// The result of `lagged`: `r`, `p`, `n`, `n_eff` are lags x pixels (lag-major),
/// the best lag per pixel (-1 where no lag has a correlation).
pub struct Lagged {
    pub lags: usize,
    pub r: Vec<f64>,
    pub p: Vec<f64>,
    pub n: Vec<u32>,
    pub n_eff: Vec<f64>,
    pub best_lag: Vec<i16>,
    pub best_r: Vec<f64>,
    /// The p of the best lag, Sidak-adjusted if asked.
    pub best_p: Vec<f64>,
}

/// Pixels of a chunk of `lagged`.
const CHUNK: usize = 256;

struct Chunk {
    start: usize,
    len: usize,
    r: Vec<f64>,
    p: Vec<f64>,
    n: Vec<u32>,
    n_eff: Vec<f64>,
    best_lag: Vec<i16>,
    best_r: Vec<f64>,
    best_p: Vec<f64>,
}

/// The lagged correlation of every pixel of `x` (`t` dates x `n` pixels,
/// time-major) with `y`. `index[l][t]` is the row of `y` paired with row `t` of
/// `x` at lag `l` (-1 for none), as `match_index` gives; there are `index.len()`
/// lags. Every pixel is independent.
pub fn lagged(x: &[f64], t: usize, n: usize, y: YRef, index: &[Vec<i32>], s: &LagSettings, n_jobs: i32) -> Lagged {
    assert_eq!(x.len(), t * n, "x must be dates x pixels");
    for ix in index {
        assert_eq!(ix.len(), t, "one index for each date of x");
    }
    let cube = matches!(y, YRef::Cube(..));
    let (y_data, y_rows) = match y {
        YRef::Cube(d, ty) => {
            assert_eq!(d.len(), ty * n, "y must be dates x pixels");
            (d, ty)
        }
        YRef::Series(d) => (d, d.len()),
    };
    let nl = index.len();
    let starts: Vec<usize> = (0..n).step_by(CHUNK).collect();
    let chunks: Vec<Chunk> = parallel::pool(parallel::threads(n_jobs)).install(|| {
        starts
            .par_iter()
            .map(|&start| {
                let len = CHUNK.min(n - start);
                let mut c = Chunk {
                    start,
                    len,
                    r: vec![f64::NAN; nl * len],
                    p: vec![f64::NAN; nl * len],
                    n: vec![0; nl * len],
                    n_eff: vec![f64::NAN; nl * len],
                    best_lag: vec![-1; len],
                    best_r: vec![f64::NAN; len],
                    best_p: vec![f64::NAN; len],
                };
                let mut sc = Scratch::default();
                let mut xs = vec![0.0; t];
                let mut ya = vec![0.0; t];
                for jj in 0..len {
                    let j = start + jj;
                    for i in 0..t {
                        xs[i] = x[i * n + j];
                    }
                    let mut best: Option<(usize, f64)> = None;
                    for (l, ix) in index.iter().enumerate() {
                        for i in 0..t {
                            let row = ix[i];
                            ya[i] = if row >= 0 && (row as usize) < y_rows {
                                y_data[if cube { row as usize * n + j } else { row as usize }]
                            } else {
                                f64::NAN
                            };
                        }
                        let pr = correlate_with(&xs, &ya, s.method, s.effective_n, s.min_pairs, &mut sc);
                        let at = l * len + jj;
                        c.r[at] = pr.r;
                        c.p[at] = pr.p;
                        c.n[at] = pr.n as u32;
                        c.n_eff[at] = pr.n_eff;
                        if pr.r.is_finite() {
                            let key = match s.best {
                                Best::Absolute => pr.r.abs(),
                                Best::Positive => pr.r,
                                Best::Negative => -pr.r,
                            };
                            if best.is_none_or(|(_, k)| key > k) {
                                best = Some((l, key));
                            }
                        }
                    }
                    if let Some((l, _)) = best {
                        let at = l * len + jj;
                        c.best_lag[jj] = l as i16;
                        c.best_r[jj] = c.r[at];
                        let p = c.p[at];
                        c.best_p[jj] = if s.sidak && p.is_finite() {
                            -(nl as f64 * (-p).ln_1p()).exp_m1()
                        } else {
                            p
                        };
                    }
                }
                c
            })
            .collect()
    });
    let mut out = Lagged {
        lags: nl,
        r: vec![f64::NAN; nl * n],
        p: vec![f64::NAN; nl * n],
        n: vec![0; nl * n],
        n_eff: vec![f64::NAN; nl * n],
        best_lag: vec![-1; n],
        best_r: vec![f64::NAN; n],
        best_p: vec![f64::NAN; n],
    };
    for c in chunks {
        for l in 0..nl {
            let (dst, src) = (l * n + c.start, l * c.len);
            out.r[dst..dst + c.len].copy_from_slice(&c.r[src..src + c.len]);
            out.p[dst..dst + c.len].copy_from_slice(&c.p[src..src + c.len]);
            out.n[dst..dst + c.len].copy_from_slice(&c.n[src..src + c.len]);
            out.n_eff[dst..dst + c.len].copy_from_slice(&c.n_eff[src..src + c.len]);
        }
        out.best_lag[c.start..c.start + c.len].copy_from_slice(&c.best_lag);
        out.best_r[c.start..c.start + c.len].copy_from_slice(&c.best_r);
        out.best_p[c.start..c.start + c.len].copy_from_slice(&c.best_p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let (u, v) = (self.next().max(1e-300), self.next());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    fn d(y: i32, m: u32, day: u32) -> i64 {
        days_from_civil(y, m, day)
    }

    #[test]
    fn pearson_matches_the_closed_form() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 1.0, 4.0, 3.0, 5.0];
        let p = correlate(&x, &y, Method::Pearson, false, 3);
        assert!((p.r - 0.8).abs() < 1e-15);
        assert_eq!(p.n, 5);
        assert_eq!(p.n_eff, 5.0);
        // 3 df have a closed form: p = 1 - (2/pi) (atan(u) + u / (1 + u^2)), u = t / sqrt(3)
        let t = 0.8 * (3.0f64 / 0.36).sqrt();
        let u = t / 3.0f64.sqrt();
        let want = 1.0 - 2.0 / std::f64::consts::PI * (u.atan() + u / (1.0 + u * u));
        assert!((p.p - want).abs() < 1e-12, "{} {}", p.p, want);
        // fewer pairs than min_pairs
        let q = correlate(&x, &y, Method::Pearson, false, 6);
        assert!(q.r.is_nan() && q.p.is_nan() && q.n == 5);
        // a constant series has no correlation
        assert!(correlate(&x, &[1.0; 5], Method::Pearson, false, 3).r.is_nan());
    }

    #[test]
    fn spearman_with_ties_and_gaps_matches_brute_force() {
        let mut rng = Rng(7);
        let n = 80;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                if i % 9 == 4 {
                    f64::NAN
                } else {
                    (rng.normal() * 2.0).round()
                }
            })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 13 == 5 {
                    f64::NAN
                } else {
                    (rng.normal() * 2.0).round() + 0.3 * i as f64 / 20.0
                }
            })
            .collect();
        let y: Vec<f64> = y.iter().map(|v| v.round()).collect();
        let (px, py): (Vec<f64>, Vec<f64>) = x
            .iter()
            .zip(&y)
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(a, b)| (*a, *b))
            .unzip();
        // average rank by counting
        let rank = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .map(|a| {
                    let less = v.iter().filter(|b| *b < a).count() as f64;
                    let eq = v.iter().filter(|b| *b == a).count() as f64;
                    less + (eq + 1.0) / 2.0
                })
                .collect()
        };
        let (rx, ry) = (rank(&px), rank(&py));
        let m = rx.len() as f64;
        let (mx, my) = (rx.iter().sum::<f64>() / m, ry.iter().sum::<f64>() / m);
        let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = rx.iter().map(|a| (a - mx) * (a - mx)).sum();
        let syy: f64 = ry.iter().map(|a| (a - my) * (a - my)).sum();
        let want = sxy / (sxx * syy).sqrt();
        let got = correlate(&x, &y, Method::Spearman, false, 10);
        assert_eq!(got.n, px.len());
        assert!((got.r - want).abs() < 1e-12, "{} {}", got.r, want);
    }

    #[test]
    fn effective_n_follows_bretherton() {
        assert!((effective_n(100.0, 0.5, 0.5) - 60.0).abs() < 1e-12);
        assert_eq!(effective_n(100.0, -0.5, 0.5), 100.0);
        assert_eq!(effective_n(100.0, f64::NAN, 0.5), 100.0);
        assert!((effective_n(100.0, 1.0, 1.0) - 100.0 * 0.01 / 1.99).abs() < 1e-12);
        // an autocorrelated pair has fewer effective pairs
        let mut rng = Rng(3);
        let mut a = vec![0.0; 400];
        let mut b = vec![0.0; 400];
        for i in 1..400 {
            a[i] = 0.8 * a[i - 1] + rng.normal();
            b[i] = 0.8 * b[i - 1] + rng.normal();
        }
        let p = correlate(&a, &b, Method::Pearson, true, 10);
        assert!(p.n_eff < 0.5 * p.n as f64 && p.n_eff > 2.0, "{}", p.n_eff);
        let q = correlate(&a, &b, Method::Pearson, false, 10);
        assert_eq!(q.n_eff, 400.0);
        assert!(p.p >= q.p);
    }

    #[test]
    fn p_value_matches_known_t_values() {
        // t = 2 with 10 df: p = 0.07338803 (scipy.stats.t.sf(2, 10) * 2)
        let t: f64 = 2.0;
        let r = t / (t * t + 10.0).sqrt();
        assert!((p_value(r, 12.0) - 0.073_388).abs() < 2e-6);
        // the critical t of 28 df at 5 %
        let t: f64 = 2.048_407_1;
        assert!((p_value(t / (t * t + 28.0).sqrt(), 30.0) - 0.05).abs() < 1e-6);
        // 1 df is Cauchy: p = 1 - 2 atan(t) / pi
        let t: f64 = 1.0;
        assert!((p_value(t / (t * t + 1.0).sqrt(), 3.0) - 0.5).abs() < 1e-12);
        // a non-integer df lies between its neighbours
        let r = 0.4;
        let (a, b, c) = (p_value(r, 12.0), p_value(r, 12.5), p_value(r, 13.0));
        assert!(a > b && b > c);
        assert!(p_value(r, 2.0).is_nan() && p_value(r, f64::NAN).is_nan());
        assert_eq!(p_value(1.0, 20.0), 0.0);
        assert_eq!(p_value(-1.0, 20.0), 0.0);
    }

    #[test]
    fn benjamini_hochberg_1995_example() {
        let p = [
            0.0201, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000, 0.0001, 0.0004, 0.0019, 0.0095,
            0.0278, 0.0298,
        ];
        let (q, star) = fdr_bh(&p, 0.05);
        // the four smallest are rejected; Bonferroni (0.05 / 15) rejects three
        assert_eq!(star, 0.0095);
        let sig: Vec<bool> = p.iter().map(|v| *v <= star).collect();
        assert_eq!(sig.iter().filter(|s| **s).count(), 4);
        // adjusted values: q <= alpha exactly for those
        for (qi, pi) in q.iter().zip(&p) {
            assert_eq!(*qi <= 0.05, *pi <= star);
        }
        assert!((q[9] - 0.0015).abs() < 1e-12); // 0.0001 * 15 / 1
        assert!((q[12] - 0.0095 * 15.0 / 4.0).abs() < 1e-12);
        assert_eq!(q[8], 1.0);
        // NaN stays NaN and does not count
        let mut with_nan = p.to_vec();
        with_nan.push(f64::NAN);
        let (q2, star2) = fdr_bh(&with_nan, 0.05);
        assert!(q2[15].is_nan() && star2 == 0.0095);
        // nothing significant
        let (_, none) = fdr_bh(&[0.5, 0.6, 0.9], 0.1);
        assert_eq!(none, 0.0);
    }

    /// An AR(1) series of unit variance.
    fn ar1(rng: &mut Rng, n: usize, phi: f64) -> Vec<f64> {
        let mut y = vec![0.0; n];
        y[0] = rng.normal();
        for i in 1..n {
            y[i] = phi * y[i - 1] + (1.0 - phi * phi).sqrt() * rng.normal();
        }
        y
    }

    #[test]
    fn lagged_finds_the_lag_of_a_driver() {
        let mut rng = Rng(11);
        let tt = 3000;
        let y = ar1(&mut rng, tt, 0.7);
        let days: Vec<i64> = (0..tt as i64).map(|i| i * 16).collect();
        // two pixels: x_t = 0.8 y_{t-2} + 0.5 e, and a pure-noise pixel
        let mut x = vec![0.0; tt * 2];
        for i in 0..tt {
            x[i * 2] = if i >= 2 {
                0.8 * y[i - 2] + 0.5 * rng.normal()
            } else {
                f64::NAN
            };
            x[i * 2 + 1] = rng.normal();
        }
        let index: Vec<Vec<i32>> = (0..5)
            .map(|k| match_index(&days, &days, Match::Exact, Lag::Steps(k)))
            .collect();
        let res = lagged(&x, tt, 2, YRef::Series(&y), &index, &LagSettings::default(), 2);
        assert_eq!(res.best_lag[0], 2);
        let want = 0.8 / (0.64f64 + 0.25).sqrt();
        assert!((res.best_r[0] - want).abs() < 0.03, "{} {}", res.best_r[0], want);
        assert!((res.r[2 * 2] - res.best_r[0]).abs() < 1e-15);
        assert!(res.best_p[0] < 1e-6);
        assert_eq!(res.n[2 * 2], (tt - 2) as u32);
        // the other lags are weaker
        assert!(res.r[0].abs() < res.best_r[0].abs() && res.r[4 * 2].abs() < res.best_r[0].abs());
        // the Sidak p of the noise pixel is not smaller than its own p
        let b = res.best_lag[1] as usize;
        assert!(res.best_p[1] >= res.p[b * 2 + 1] - 1e-15);
        // Positive/Negative pick by sign
        let neg = LagSettings {
            best: Best::Negative,
            sidak: false,
            ..Default::default()
        };
        let rn = lagged(&x, tt, 2, YRef::Series(&y), &index, &neg, 1);
        let lowest = (0..5).map(|l| rn.r[l * 2]).fold(f64::INFINITY, f64::min);
        assert_eq!(rn.best_r[0], lowest);
        assert_eq!(rn.best_p[0], rn.p[rn.best_lag[0] as usize * 2]);
        // the same with a cube for y
        let mut ycube = vec![0.0; tt * 2];
        for i in 0..tt {
            ycube[i * 2] = y[i];
            ycube[i * 2 + 1] = y[i];
        }
        let rc = lagged(&x, tt, 2, YRef::Cube(&ycube, tt), &index, &LagSettings::default(), 3);
        assert_eq!(rc.best_lag, res.best_lag);
        assert_eq!(rc.r, res.r);
    }

    #[test]
    fn null_pixels_give_few_false_positives() {
        let mut rng = Rng(5);
        let (tt, n) = (60, 10_000);
        let days: Vec<i64> = (0..tt as i64).map(|i| i * 30).collect();
        let y: Vec<f64> = (0..tt).map(|_| rng.normal()).collect();
        let x: Vec<f64> = (0..tt * n).map(|_| rng.normal()).collect();
        let index: Vec<Vec<i32>> = (0..1)
            .map(|k| match_index(&days, &days, Match::Exact, Lag::Steps(k)))
            .collect();
        let res = lagged(&x, tt, n, YRef::Series(&y), &index, &LagSettings::default(), 4);
        let raw = res.p.iter().filter(|p| **p <= 0.05).count() as f64 / n as f64;
        assert!((0.02..0.08).contains(&raw), "{raw}");
        let (_, star) = fdr_bh(&res.best_p, 0.10);
        let fp = res.best_p.iter().filter(|p| **p <= star).count() as f64 / n as f64;
        assert!(fp <= 0.10, "{fp}");
    }

    #[test]
    fn match_index_exact_and_steps() {
        let x = [0, 10, 20];
        let y = [10, 20, 30];
        assert_eq!(match_index(&x, &y, Match::Exact, Lag::Steps(0)), vec![-1, 0, 1]);
        assert_eq!(match_index(&x, &y, Match::Exact, Lag::Steps(1)), vec![-1, -1, 0]);
        assert_eq!(match_index(&x, &y, Match::Exact, Lag::Steps(-1)), vec![0, 1, -1]);
        assert_eq!(match_index(&x, &y, Match::Exact, Lag::Steps(5)), vec![-1, -1, -1]);
    }

    #[test]
    fn match_index_nearest() {
        let x = [0, 16, 32];
        let y = [1, 18, 30];
        assert_eq!(match_index(&x, &y, Match::Nearest(3.0), Lag::Steps(0)), vec![0, 1, 2]);
        assert_eq!(match_index(&x, &y, Match::Nearest(1.5), Lag::Steps(0)), vec![0, -1, -1]);
        // x_{t-1} = 16 is nearest to 18
        assert_eq!(match_index(&x, &y, Match::Nearest(3.0), Lag::Steps(1)), vec![-1, 0, 1]);
        // the earlier date wins a tie
        assert_eq!(
            match_index(&[10], &[8, 12], Match::Nearest(5.0), Lag::Steps(0)),
            vec![0]
        );
    }

    #[test]
    fn match_index_same_month_across_year_end_and_missing_months() {
        let x = [
            d(2020, 12, 20),
            d(2021, 1, 15),
            d(2021, 2, 20),
            d(2021, 3, 31),
            d(2021, 5, 2),
        ];
        // monthly y without February 2021 and with two dates in December 2020
        let y = [
            d(2020, 11, 1),
            d(2020, 12, 1),
            d(2020, 12, 28),
            d(2021, 1, 1),
            d(2021, 3, 1),
            d(2021, 4, 1),
        ];
        assert_eq!(
            match_index(&x, &y, Match::SameMonth, Lag::Months(0)),
            vec![2, 3, -1, 4, -1]
        );
        // one month back: Dec 2020 -> Nov 2020, Jan -> Dec 2020 (Dec 28 is nearer to Dec 15 than Dec 1),
        // Feb -> Jan, Mar 31 -> Feb 28 (missing), May 2 -> Apr 2
        assert_eq!(
            match_index(&x, &y, Match::SameMonth, Lag::Months(1)),
            vec![0, 2, 3, -1, 5]
        );
        // a year back from Jan 2021 is Jan 2020, absent
        assert_eq!(match_index(&x[1..2], &y, Match::SameMonth, Lag::Months(12)), vec![-1]);
        // a month ahead (negative lag) across the year end
        assert_eq!(match_index(&x[..1], &y, Match::SameMonth, Lag::Months(-1)), vec![3]);
    }

    #[test]
    fn match_index_months_clamp_and_tolerance() {
        // March 31 a month back is February 28; from February 29, 2024 a year back is February 28, 2023
        assert_eq!(months_back(d(2021, 3, 31), 1), d(2021, 2, 28));
        assert_eq!(months_back(d(2020, 3, 31), 1), d(2020, 2, 29));
        assert_eq!(months_back(d(2024, 2, 29), 12), d(2023, 2, 28));
        assert_eq!(months_back(d(2021, 1, 31), 2), d(2020, 11, 30));
        assert_eq!(months_back(d(2021, 1, 15), 13), d(2019, 12, 15));
        // exact dates with a month lag: the nearest date within half the median step (at least 8 days)
        let y = [d(2021, 2, 1), d(2021, 3, 1), d(2021, 4, 1)];
        let x = [d(2021, 3, 31), d(2021, 5, 10), d(2021, 7, 1)];
        // the tolerance is 15.5 days (half the median step of 31). Mar 31 -> Feb 28: Mar 1 is the nearest
        // (1 day); May 10 -> Apr 10: Apr 1 (9 days); Jul 1 -> Jun 1: Apr 1 is too far
        assert_eq!(match_index(&x, &y, Match::Exact, Lag::Months(1)), vec![1, 2, -1]);
        // a reachable one: Apr 1 a month back is Mar 1, exactly
        assert_eq!(match_index(&[d(2021, 4, 1)], &y, Match::Exact, Lag::Months(1)), vec![1]);
    }

    #[test]
    fn match_index_same_year() {
        let x = [d(2020, 6, 1), d(2021, 6, 1), d(2022, 6, 1)];
        let y = [d(2020, 1, 1), d(2020, 12, 1), d(2022, 3, 1)];
        // the nearest date of the same year
        assert_eq!(match_index(&x, &y, Match::SameYear, Lag::Steps(0)), vec![0, -1, 2]);
        // one step back: the year of the previous x
        assert_eq!(match_index(&x, &y, Match::SameYear, Lag::Steps(1)), vec![-1, 0, -1]);
    }
}
