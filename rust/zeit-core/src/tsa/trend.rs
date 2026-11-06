//! The linear trend of a series and its greatest change (CAT), per pixel: the
//! regression on the position of each valid step, its significance by a t-test,
//! and the split of the series at the step of greatest loss into the parts before
//! and after.
//!
//! Ported from FORCE (Framework for Operational Radiometric Correction for
//! Environmental monitoring), Copyright (C) 2013-2025 David Frantz,
//! GPL-3.0-or-later, https://github.com/davidfrantz/force, src/modules:
//! higher-level/trend-hl.c and cross-level/stats-cl.c (covar_recurrence,
//! slope_significant, tscore).
//!
//! The regression is the one-pass covariance recurrence of Pébay (2008, SANDIA
//! report SAND2008-6212), in FORCE's order. Unlike FORCE, which keeps int16 and
//! truncates, the work is in f64 and the results stay f64 (R-squared in 0-1, not
//! times 10000). Differences from FORCE, all deliberate:
//!
//! * The critical t of the significance test is the exact quantile of Student's
//!   t (by bisection on the regularized incomplete beta function, once for each
//!   degree of freedom), where FORCE iterates Hill's (1970) approximation to a
//!   tolerance of 1e-4; the slope is called significant exactly where the
//!   t-value passes the exact critical value.
//! * With the penalty, FORCE compares the penalized change of a candidate with
//!   the *unpenalized* change of the best one so far (`change_real`), which
//!   mixes two scales: a dip that the series recovers from, penalized to a
//!   fraction of its size, can then win against a later, real loss that is
//!   smaller than the dip's raw size. Here the penalized changes are compared
//!   with each other, and the unpenalized change of the winner is what is
//!   written (as FORCE does).

use crate::dist::t_two_sided;
use crate::parallel;

/// Values of a trend: mean, offset, slope, pct_gain, abs_gain, rsq, sig, rmse,
/// mae, maxe, num, len.
pub const TREND_LEN: usize = 12;
/// Values of a change: change, loss, change_index, then the trend values of the
/// total series, of the part before the change and of the part after it.
pub const CAT_LEN: usize = 3 + 3 * TREND_LEN;

/// The tail of the significance test of the slope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// Significantly negative: t <= the critical t (negative).
    Left,
    /// Either: |t| >= the critical t.
    Two,
    /// Significantly positive: t >= the critical t.
    Right,
}

impl Tail {
    /// The tail of a name: `left`, `two` (also `both`) or `right`.
    pub fn parse(name: &str) -> Result<Tail, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "left" => Ok(Tail::Left),
            "two" | "both" | "two-sided" => Ok(Tail::Two),
            "right" => Ok(Tail::Right),
            other => Err(format!("unknown tail {other:?}: use \"two\", \"left\" or \"right\"")),
        }
    }
}

/// The c >= 0 with P(|T| >= c) = `q` for Student's t with `df` degrees of
/// freedom, q in (0, 1].
fn two_sided_quantile(q: f64, df: f64) -> f64 {
    if q >= 1.0 {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while t_two_sided(hi, df) > q && hi < 1e300 {
        lo = hi;
        hi *= 2.0;
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if t_two_sided(mid, df) > q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The critical t of the test at significance level `alpha` with `df` degrees
/// of freedom: the positive one for `Two` (reject when |t| >= it) and `Right`,
/// the negative one for `Left` (as FORCE's `tscore`, with the exact quantile);
/// for a level above 0.5 the one-sided critical value changes sign.
pub fn critical_t(alpha: f64, df: usize, tail: Tail) -> f64 {
    let df = df as f64;
    let right = |a: f64| {
        if a <= 0.5 {
            two_sided_quantile(2.0 * a, df)
        } else {
            -two_sided_quantile(2.0 * (1.0 - a), df)
        }
    };
    match tail {
        Tail::Two => two_sided_quantile(alpha, df),
        Tail::Right => right(alpha),
        Tail::Left => -right(alpha),
    }
}

/// What the test needs: the critical t for each degree of freedom (the index),
/// made once for all the series.
struct Test {
    tail: Tail,
    crit: Vec<f64>,
}

impl Test {
    fn new(n: usize, conf: f64, tail: Tail) -> Result<Test, String> {
        if !(conf > 0.0 && conf < 1.0) {
            return Err(format!("conf must be between 0 and 1, got {conf}"));
        }
        let alpha = 1.0 - conf;
        // k valid steps of n give df = k - 2, from 1 to n - 2
        let crit = (0..n.saturating_sub(1))
            .map(|df| if df == 0 { f64::NAN } else { critical_t(alpha, df, tail) })
            .collect();
        Ok(Test { tail, crit })
    }

    /// -1 (significantly negative), 0 or 1 (significantly positive).
    fn sig(&self, k: usize, t: f64) -> f64 {
        let c = self.crit[k - 2];
        let (neg, pos) = match self.tail {
            Tail::Left => (t <= c, false),
            Tail::Right => (false, t >= c),
            Tail::Two => (t <= -c, t >= c),
        };
        if neg {
            -1.0
        } else if pos {
            1.0
        } else {
            0.0
        }
    }
}

/// Pébay's one-pass estimates of the means, the sums of squares and the
/// co-moment of x and y.
struct Moments {
    k: f64,
    mx: f64,
    my: f64,
    vx: f64,
    vy: f64,
    cv: f64,
}

impl Moments {
    /// Over the valid steps of `y` in `lo..hi`, x being the step.
    fn of(y: &[f64], lo: usize, hi: usize) -> Moments {
        let mut m = Moments {
            k: 0.0,
            mx: 0.0,
            my: 0.0,
            vx: 0.0,
            vy: 0.0,
            cv: 0.0,
        };
        for (f, &v) in y.iter().enumerate().take(hi).skip(lo) {
            if v.is_nan() {
                continue;
            }
            let x = f as f64;
            m.k += 1.0;
            if m.k == 1.0 {
                m.mx = x;
                m.my = v;
            } else {
                let (oldmx, oldmy) = (m.mx, m.my);
                m.mx = oldmx + (x - oldmx) / m.k;
                m.my = oldmy + (v - oldmy) / m.k;
                m.vx += (x - oldmx) * (x - m.mx);
                m.vy += (v - oldmy) * (v - m.my);
                m.cv += (m.k - 1.0) / m.k * (x - oldmx) * (v - oldmy);
            }
        }
        m
    }
}

/// The twelve values of the regression over the valid steps of `y` in
/// `lo..hi`, `len` steps long: mean, offset (the fit at step `lo`), slope,
/// pct_gain (slope x len / offset x 100 where the offset is positive, else 0),
/// abs_gain (slope x len), rsq, sig, rmse, mae, maxe, num, len. A part without
/// valid steps is all NaN; one with fewer than 3 has its mean, num and len only
/// (`whole`: nothing, as for the trend of a whole series).
fn part(y: &[f64], lo: usize, hi: usize, len: usize, test: &Test, whole: bool, out: &mut [f64]) {
    out.fill(f64::NAN);
    let m = Moments::of(y, lo, hi);
    if m.k < 1.0 || (whole && m.k < 3.0) {
        return;
    }
    out[0] = m.my;
    out[10] = m.k;
    out[11] = len as f64;
    if m.k < 3.0 {
        return;
    }
    let cv = m.cv / (m.k - 1.0);
    let vx = m.vx / (m.k - 1.0);
    let vy = m.vy / (m.k - 1.0);
    let slope = cv / vx;
    let off = m.my - slope * m.mx;
    let rsq = cv * cv / (vx * vy);
    let (mut ssqe, mut sae, mut sxsq, mut maxe) = (0.0, 0.0, 0.0, f64::NEG_INFINITY);
    for (f, &v) in y.iter().enumerate().take(hi).skip(lo) {
        if v.is_nan() {
            continue;
        }
        let x = f as f64;
        let e = v - (off + slope * x);
        ssqe += e * e;
        sae += e.abs();
        sxsq += (x - m.mx) * (x - m.mx);
        if e.abs() > maxe {
            maxe = e.abs();
        }
    }
    // the standard error of the slope and the t-value
    let seb = (1.0 / (m.k - 2.0) * ssqe).sqrt() / sxsq.sqrt();
    let k = m.k as usize;
    // the fit at the start of the part
    let yhat = off + slope * lo as f64;
    out[1] = yhat;
    out[2] = slope;
    out[3] = if yhat > 0.0 { slope * len as f64 / yhat * 100.0 } else { 0.0 };
    out[4] = slope * len as f64;
    out[5] = rsq;
    out[6] = test.sig(k, slope / seb);
    out[7] = (ssqe / m.k).sqrt();
    out[8] = sae / m.k;
    out[9] = maxe;
}

/// The trend of every series of `values` (pixels, steps; NaN marks a missing
/// step), x being the step (0..steps): (pixels, 12), flat, with the values of
/// `TREND_LEN`. A series with fewer than 3 valid steps is all NaN. `conf` is
/// the confidence level of the test of the slope (0.95: significance level
/// 0.05), `tail` its tail.
pub fn trend_batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    conf: f64,
    tail: Tail,
    n_jobs: i32,
) -> Result<Vec<f64>, String> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    let test = Test::new(n_time, conf, tail)?;
    let mut out = vec![f64::NAN; n_pixels * TREND_LEN];
    parallel::for_each_row(&mut out, TREND_LEN, n_jobs, || (), |_, p, row| {
        part(&values[p * n_time..(p + 1) * n_time], 0, n_time, n_time, &test, true, row);
    });
    Ok(out)
}

/// The greatest change of one series: the step `f` and its unpenalized change
/// (the previous valid value less this one, positive for a loss), or None.
fn greatest_change(y: &[f64], penalty: bool) -> Option<(usize, f64)> {
    let n = y.len();
    let mut best: Option<(usize, f64, f64)> = None; // step, penalized change, change
    let mut pre: Option<usize> = None;
    for f in 0..n {
        if y[f].is_nan() {
            continue;
        }
        let Some(f_pre) = pre.replace(f) else { continue };
        let real = y[f_pre] - y[f];
        let mut now = real;
        if penalty && real > 0.0 {
            // the loss is weighted by what the series has lost by the next valid step
            let Some(f_post) = (f + 1..n).find(|&g| !y[g].is_nan()) else { continue };
            now *= (y[f_pre] - y[f_post]) / 1e4;
        }
        // strictly greater: a tie keeps the first
        if best.is_none_or(|(_, b, _)| now > b) {
            best = Some((f, now, real));
        }
    }
    best.map(|(f, _, real)| (f, real))
}

/// The greatest change of every series of `values` (pixels, steps), and the
/// trend of the whole series, of the part before it and of the part after it:
/// (pixels, 39), flat: change (the greatest loss: the previous valid value less
/// the one at the change, positive for a loss), loss (the change as a percentage
/// of the total series' offset, 0 unless that is positive), change_index (the
/// step of the change), then the twelve values of `TREND_LEN` for the total
/// series (steps 0..n), before (0..change_index - 1) and after (change_index..n).
/// A series with fewer than 2 valid steps, or with `penalty` and no loss that a
/// valid step follows, is all NaN. With `penalty`, a loss is weighted by what
/// the series has lost by the next valid step (FORCE's CHANGE_PENALTY, whose
/// factor of 1e4 is that of values scaled by 10000).
pub fn cat_batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    penalty: bool,
    conf: f64,
    tail: Tail,
    n_jobs: i32,
) -> Result<Vec<f64>, String> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    let test = Test::new(n_time, conf, tail)?;
    let mut out = vec![f64::NAN; n_pixels * CAT_LEN];
    parallel::for_each_row(&mut out, CAT_LEN, n_jobs, || (), |_, p, row| {
        let y = &values[p * n_time..(p + 1) * n_time];
        let Some((f, change)) = greatest_change(y, penalty) else { return };
        row[0] = change;
        row[2] = f as f64;
        part(y, 0, n_time, n_time, &test, false, &mut row[3..3 + TREND_LEN]);
        let before = f - 1;
        part(y, 0, before, before, &test, false, &mut row[3 + TREND_LEN..3 + 2 * TREND_LEN]);
        part(y, f, n_time, n_time - f, &test, false, &mut row[3 + 2 * TREND_LEN..]);
        let offset = row[4];
        row[1] = if offset > 0.0 { change / offset * 100.0 } else { 0.0 };
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAN: f64 = f64::NAN;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= tol * (1.0 + b.abs())
    }

    fn trend_of(y: &[f64], conf: f64, tail: Tail) -> Vec<f64> {
        trend_batch(y, 1, y.len(), conf, tail, 1).unwrap()
    }

    fn cat_of(y: &[f64], penalty: bool) -> Vec<f64> {
        cat_batch(y, 1, y.len(), penalty, 0.95, Tail::Two, 1).unwrap()
    }

    /// The regression by the textbook sums, for the comparison.
    fn textbook(y: &[f64]) -> (f64, f64, f64, f64) {
        let pts: Vec<(f64, f64)> = y.iter().enumerate().filter(|p| !p.1.is_nan()).map(|(i, &v)| (i as f64, v)).collect();
        let k = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / k;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / k;
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let syy: f64 = pts.iter().map(|p| (p.1 - my).powi(2)).sum();
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let slope = sxy / sxx;
        let off = my - slope * mx;
        let sse: f64 = pts.iter().map(|p| (p.1 - off - slope * p.0).powi(2)).sum();
        (slope, off, sxy * sxy / (sxx * syy), (sse / (k - 2.0) / sxx).sqrt())
    }

    #[test]
    fn critical_values_of_students_t() {
        // two-sided 5 %: the table's 12.7062047, 2.2281389, 2.0422725
        assert!(close(critical_t(0.05, 1, Tail::Two), 12.7062047362, 1e-9));
        assert!(close(critical_t(0.05, 10, Tail::Two), 2.2281388520, 1e-9));
        assert!(close(critical_t(0.05, 30, Tail::Two), 2.0422724563, 1e-9));
        // one-sided 5 % (df 5): 2.0150484; the left one is its mirror
        assert!(close(critical_t(0.05, 5, Tail::Right), 2.0150483733, 1e-9));
        assert!(close(critical_t(0.05, 5, Tail::Left), -2.0150483733, 1e-9));
        // two-sided 1 % (df 3): 5.8409093; 50 %: the median of |T|
        assert!(close(critical_t(0.01, 3, Tail::Two), 5.8409093097, 1e-9));
        assert!(close(critical_t(0.5, 4, Tail::Right), 0.0, 1e-9));
        // a level above one half flips the one-sided critical value
        assert!(critical_t(0.9, 7, Tail::Right) < 0.0);
        assert!(critical_t(0.9, 7, Tail::Left) > 0.0);
    }

    #[test]
    fn trend_of_a_perfect_line() {
        let y: Vec<f64> = (0..6).map(|i| 2.0 + 3.0 * i as f64).collect();
        let r = trend_of(&y, 0.95, Tail::Two);
        let want = [9.5, 2.0, 3.0, 3.0 * 6.0 / 2.0 * 100.0, 18.0, 1.0, 1.0, 0.0, 0.0, 0.0, 6.0, 6.0];
        for (g, w) in r.iter().zip(want) {
            assert!(close(*g, w, 1e-9), "{r:?}");
        }
    }

    #[test]
    fn trend_against_the_textbook_regression() {
        let y = [0.31, 0.35, 0.29, 0.42, 0.40, 0.47, 0.44, 0.52, 0.49, 0.55];
        let r = trend_of(&y, 0.95, Tail::Two);
        let (slope, off, rsq, seb) = textbook(&y);
        assert!(close(r[1], off, 1e-12) && close(r[2], slope, 1e-12) && close(r[5], rsq, 1e-12));
        assert!(close(r[0], y.iter().sum::<f64>() / 10.0, 1e-12));
        let resid: Vec<f64> = y.iter().enumerate().map(|(i, v)| v - off - slope * i as f64).collect();
        assert!(close(r[7], (resid.iter().map(|e| e * e).sum::<f64>() / 10.0).sqrt(), 1e-12));
        assert!(close(r[8], resid.iter().map(|e| e.abs()).sum::<f64>() / 10.0, 1e-12));
        assert!(close(r[9], resid.iter().fold(0.0_f64, |a, e| a.max(e.abs())), 1e-12));
        assert_eq!((r[10], r[11]), (10.0, 10.0));
        // the t-value against the critical one at df 8
        let t = slope / seb;
        let tc = critical_t(0.05, 8, Tail::Two);
        assert_eq!(r[6], if t >= tc { 1.0 } else { 0.0 });
        assert_eq!(r[6], 1.0, "t = {t}");
        // a stricter level does not reject
        let strict = trend_of(&y, 1.0 - 1e-9, Tail::Two);
        assert_eq!(strict[6], 0.0);
        // the left tail never calls a rise significant
        assert_eq!(trend_of(&y, 0.95, Tail::Left)[6], 0.0);
        let down: Vec<f64> = y.iter().map(|v| -v).collect();
        assert_eq!(trend_of(&down, 0.95, Tail::Left)[6], -1.0);
        assert_eq!(trend_of(&down, 0.95, Tail::Two)[6], -1.0);
        assert_eq!(trend_of(&down, 0.95, Tail::Right)[6], 0.0);
    }

    #[test]
    fn trend_with_gaps_and_too_few_values() {
        let y = [1.0, NAN, 3.0, NAN, 5.5, 6.0];
        let r = trend_of(&y, 0.95, Tail::Two);
        let (slope, off, _, _) = textbook(&y);
        assert!(close(r[2], slope, 1e-12) && close(r[1], off, 1e-12));
        assert_eq!((r[10], r[11]), (4.0, 6.0));
        assert!(close(r[4], slope * 6.0, 1e-12), "the gain is over the whole length");
        // fewer than 3 valid steps: nothing at all
        let r = trend_of(&[1.0, NAN, 2.0, NAN], 0.95, Tail::Two);
        assert!(r.iter().all(|v| v.is_nan()), "{r:?}");
        // a constant series: slope 0, no rsq (0/0), not significant
        let r = trend_of(&[4.0, 4.0, 4.0, 4.0], 0.95, Tail::Two);
        assert_eq!((r[2], r[6], r[7]), (0.0, 0.0, 0.0));
        assert!(r[5].is_nan());
        assert!(trend_batch(&y, 1, 6, 1.5, Tail::Two, 1).is_err());
    }

    #[test]
    fn trend_of_many_pixels_in_parallel() {
        let (n, p) = (12, 50);
        let values: Vec<f64> = (0..p * n).map(|i| ((i * 7919) % 101) as f64 + (i % n) as f64 * 0.3).collect();
        let one = trend_batch(&values, p, n, 0.95, Tail::Two, 1).unwrap();
        let many = trend_batch(&values, p, n, 0.95, Tail::Two, 4).unwrap();
        assert_eq!(one.len(), p * TREND_LEN);
        assert!(one.iter().zip(&many).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
    }

    #[test]
    fn change_in_the_middle() {
        let y = [10.0, 10.0, 10.0, 4.0, 4.0, 4.0];
        let r = cat_of(&y, false);
        assert_eq!((r[0], r[2]), (6.0, 3.0));
        // total: the line over the whole series
        let (slope, off, _, _) = textbook(&y);
        let total = &r[3..15];
        assert!(close(total[1], off, 1e-12) && close(total[2], slope, 1e-12));
        assert_eq!((total[10], total[11]), (6.0, 6.0));
        assert!(close(r[1], 6.0 / off * 100.0, 1e-12), "loss is relative to the total offset");
        // before is [0, f_change - 1) = [0, 2): two steps, mean, num and len only
        let before = &r[15..27];
        assert_eq!((before[0], before[10], before[11]), (10.0, 2.0, 2.0));
        assert!(before[1..10].iter().all(|v| v.is_nan()));
        // after is [3, 6): flat at 4, offset at its start, no slope
        let after = &r[27..39];
        assert_eq!((after[0], after[1], after[2], after[3], after[4]), (4.0, 4.0, 0.0, 0.0, 0.0));
        assert_eq!((after[10], after[11]), (3.0, 3.0));
        assert_eq!(after[6], 0.0);
        assert!(after[5].is_nan(), "a constant part has no rsq");
    }

    #[test]
    fn part_offset_is_the_fit_at_its_start() {
        let y = [9.0, 8.0, 7.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let r = cat_of(&y, false);
        assert_eq!(r[2], 4.0);
        // before [0, 3): 9, 8, 7; after [4, 9): 1..5, so its offset (at step 4) is 1
        let before = &r[15..27];
        assert!(close(before[2], -1.0, 1e-12) && close(before[1], 9.0, 1e-12));
        assert!(close(before[3], -1.0 * 3.0 / 9.0 * 100.0, 1e-12));
        assert!(close(before[4], -3.0, 1e-12));
        let after = &r[27..39];
        assert!(close(after[2], 1.0, 1e-12) && close(after[1], 1.0, 1e-12));
        assert!(close(after[3], 5.0 / 1.0 * 100.0, 1e-12));
        assert_eq!((after[10], after[11]), (5.0, 5.0));
        assert_eq!(after[6], 1.0);
    }

    #[test]
    fn only_increases_keep_the_smallest_rise() {
        let r = cat_of(&[1.0, 2.0, 4.0, 8.0], false);
        assert_eq!((r[0], r[2]), (-1.0, 1.0));
        // before: [0, 0) is empty
        assert!(r[15..27].iter().all(|v| v.is_nan()));
        assert_eq!(r[27 + 11], 3.0);
    }

    #[test]
    fn a_tie_keeps_the_first() {
        let r = cat_of(&[5.0, 3.0, 1.0], false);
        assert_eq!((r[0], r[2]), (2.0, 1.0));
    }

    #[test]
    fn gaps_are_skipped_in_the_change() {
        // 5 -> (gap) -> 3: the change at step 2 is against step 0
        let r = cat_of(&[5.0, NAN, 3.0, NAN, 3.5], false);
        assert_eq!((r[0], r[2]), (2.0, 2.0));
        assert_eq!(r[3 + 10], 3.0, "total has 3 valid steps");
        assert_eq!(r[3 + 11], 5.0, "and a length of all the steps");
        // fewer than two valid steps: no change
        assert!(cat_of(&[NAN, 2.0, NAN], false).iter().all(|v| v.is_nan()));
        assert!(cat_of(&[NAN, NAN], false).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn a_part_with_fewer_than_three_valid_steps() {
        // after = [3, 5): two valid steps, so only mean, num and len
        let r = cat_of(&[9.0, 8.0, 7.0, 1.0, 2.0], false);
        assert_eq!(r[2], 3.0);
        let after = &r[27..39];
        assert_eq!((after[0], after[10], after[11]), (1.5, 2.0, 2.0));
        assert!(after[1..10].iter().all(|v| v.is_nan()));
        // a total with two valid steps: change, no loss (the total has no offset)
        let r = cat_of(&[5.0, NAN, 3.0, NAN], false);
        assert_eq!(r[0], 2.0);
        assert_eq!(r[1], 0.0);
        assert_eq!((r[3], r[13], r[14]), (4.0, 2.0, 4.0));
        assert!(r[4].is_nan());
    }

    #[test]
    fn loss_is_zero_without_a_positive_offset() {
        let r = cat_of(&[-1.0, -2.0, -6.0, -7.0, -8.0], false);
        assert_eq!(r[0], 4.0);
        assert!(r[4] < 0.0);
        assert_eq!(r[1], 0.0);
    }

    #[test]
    fn the_penalty_weighs_a_loss_by_what_is_lost_by_the_next_step() {
        // a dip of 7000 that recovers (8000, 1000, 7000) and a lasting loss of 5000 (7000, 2000, 0)
        let y = [8000.0, 1000.0, 7000.0, 2000.0, 0.0];
        // without the penalty, the dip is the greatest change
        let plain = cat_of(&y, false);
        assert_eq!((plain[0], plain[2]), (7000.0, 1.0));
        // penalized: the dip counts 7000 x (8000 - 7000) / 1e4 = 700, the lasting loss
        // 5000 x (7000 - 0) / 1e4 = 3500; the written change is the real one (5000)
        let pen = cat_of(&y, true);
        assert_eq!((pen[0], pen[2]), (5000.0, 3.0));
        // FORCE compares 3500 with the dip's raw 7000 and keeps the dip: the literal
        // behavior, which this port does not reproduce
        let force = literal_force_change(&y);
        assert_eq!(force, Some((1, 7000.0)));
    }

    /// FORCE's loop as written (the penalized candidate against the best one's real change).
    fn literal_force_change(y: &[f64]) -> Option<(usize, f64)> {
        let mut change = -32768.0;
        let mut found = None;
        for f in 1..y.len() {
            if y[f].is_nan() {
                continue;
            }
            let Some(f_pre) = (0..f).rev().find(|&g| !y[g].is_nan()) else { continue };
            let real = y[f_pre] - y[f];
            let mut now = real;
            if now > 0.0 {
                let Some(f_post) = (f + 1..y.len()).find(|&g| !y[g].is_nan()) else { continue };
                now *= (y[f_pre] - y[f_post]) / 1e4;
            }
            if now > change {
                change = real;
                found = Some((f, real));
            }
        }
        found
    }

    #[test]
    fn the_penalty_skips_a_loss_without_a_later_step() {
        // the last step's loss has no next valid step: only the earlier, smaller one counts
        let y = [10000.0, 9000.0, 9500.0, 0.0];
        let pen = cat_of(&y, true);
        assert_eq!((pen[0], pen[2]), (1000.0, 1.0));
        let plain = cat_of(&y, false);
        assert_eq!((plain[0], plain[2]), (9500.0, 3.0));
        // all of them skipped (a single loss, at the end): no change
        assert!(cat_of(&[5.0, 1.0], true).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn cat_of_many_pixels_in_parallel() {
        let (n, p) = (10, 40);
        let values: Vec<f64> = (0..p * n).map(|i| ((i * 7919) % 97) as f64 * 100.0).collect();
        let one = cat_batch(&values, p, n, true, 0.9, Tail::Right, 1).unwrap();
        let many = cat_batch(&values, p, n, true, 0.9, Tail::Right, 4).unwrap();
        assert_eq!(one.len(), p * CAT_LEN);
        assert!(one.iter().zip(&many).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
        assert!(cat_batch(&values, p, n, true, 0.0, Tail::Right, 1).is_err());
    }
}
