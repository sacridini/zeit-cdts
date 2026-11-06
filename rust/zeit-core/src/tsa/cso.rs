//! Clear-sky observations (CSO): per pixel and window of time, the number of valid
//! observations and the statistics of the intervals between them (the longest
//! gap without an observation, the mean revisit time...).
//!
//! Ported from FORCE (Framework for Operational Radiometric Correction for
//! Environmental monitoring), Copyright (C) 2013-2025 David Frantz,
//! GPL-3.0-or-later, https://github.com/davidfrantz/force, src/modules:
//! higher-level/cso-hl.c.
//!
//! The intervals of a window `[start, end)` are, as in FORCE: from the start to the
//! first valid observation, between consecutive valid observations, and from the
//! last one to the end. An observation on the same day as the one before it (or on
//! the start of the window) makes no interval. `count` is the number of valid
//! observations of the window, those on the same day included. A window without a
//! valid observation has one interval, as long as the window, and a count of 0.
//!
//! Differences from FORCE, all deliberate. FORCE writes the intervals' statistics
//! with the number of observations `n` where the number of intervals `k` belongs:
//! its standard deviation divides by `n - 1`, its skewness and kurtosis by `n`,
//! its quantiles are taken over the first `n` of the `k` intervals (which, as
//! `k = n + 1` when the last observation is before the window's end, leaves the
//! interval to the end out), and with observations on the same day (`n > k`) they
//! read memory left over from other windows. Here every statistic but `count` is
//! that of the same `k` intervals: the standard deviation has `k - 1` degrees of
//! freedom (NaN for one interval), the skewness and kurtosis are those of `stats` (population moments,
//! excess kurtosis; NaN for constant intervals), the quantiles are R-7 over all of
//! them. Without duplicates in the same day FORCE and Zeit agree on `count`, `min`,
//! `max`, `range` and `mean`, and differ on the rest where `k != n`. The float
//! results are not rounded. In a window without a valid observation (`n = 0`) FORCE
//! writes 0 for the standard deviation, skewness, kurtosis and quantiles (no
//! observations to divide by; GSL's quantile of no values); here the single
//! interval gives NaN for the first three and its own length for the quantiles.
//!
//! FORCE counts the windows while `year < last year || month < last month`, so with
//! a range that ends in December the last window is lost; the window list is the
//! caller's here.

use super::stats::{Metric, Plan};
use crate::parallel;

/// The buffers of one thread.
#[derive(Default)]
struct Work {
    gaps: Vec<f64>,
    sorted: Vec<f64>,
    q: Vec<f64>,
    tmp: Vec<f64>,
}

/// The CSO metrics of every series of `values` (pixels, time) for each window
/// `[start, end)` of `windows` (in the unit of `t`, days): (pixels, metrics,
/// windows), flat.
///
/// `t` (one per time step) must not decrease; a step is a valid observation where
/// its value is finite. `metrics` are those of `stats::Metric`; `count` is the
/// valid observations, any other the statistic of the intervals between them (see
/// the module's notes). Steps outside every window are ignored; windows may
/// overlap, and are found by bisection of `t`.
pub fn cso_batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    t: &[f64],
    windows: &[(f64, f64)],
    metrics: &[Metric],
    n_jobs: i32,
) -> Result<Vec<f64>, String> {
    if values.len() != n_pixels * n_time {
        return Err("values must be n_pixels x n_time".into());
    }
    if t.len() != n_time {
        return Err("one time per step".into());
    }
    if t.iter().any(|x| x.is_nan()) || t.windows(2).any(|w| w[1] < w[0]) {
        return Err("the times must not decrease".into());
    }
    if windows.iter().any(|&(a, b)| a.is_nan() || b.is_nan()) {
        return Err("a window has a missing limit".into());
    }
    // The steps of each window: the first with t >= start up to the first with t >= end.
    let span: Vec<(usize, usize)> = windows
        .iter()
        .map(|&(a, b)| {
            let lo = t.partition_point(|&x| x < a);
            let hi = t.partition_point(|&x| x < b).max(lo);
            (lo, hi)
        })
        .collect();
    let plan = Plan::new(metrics, 1.0);
    let m = plan.n_metrics();
    let nw = windows.len();
    let count_at: Vec<usize> = metrics
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == Metric::Count)
        .map(|(i, _)| i)
        .collect();
    let mut out = vec![f64::NAN; n_pixels * m * nw];
    parallel::for_each_row(&mut out, m * nw, n_jobs, Work::default, |w, p, row| {
        let y = &values[p * n_time..(p + 1) * n_time];
        let Work { gaps, sorted, q, tmp } = w;
        tmp.resize(m, f64::NAN);
        for (j, (&(start, end), &(lo, hi))) in windows.iter().zip(&span).enumerate() {
            gaps.clear();
            let (mut n, mut left) = (0usize, start);
            for s in lo..hi {
                if !y[s].is_finite() {
                    continue;
                }
                n += 1;
                if t[s] > left {
                    gaps.push(t[s] - left);
                    left = t[s];
                }
            }
            if end > left {
                gaps.push(end - left);
            }
            plan.eval(gaps, sorted, q, tmp);
            for &c in &count_at {
                tmp[c] = n as f64;
            }
            for (k, &v) in tmp.iter().enumerate() {
                row[k * nw + j] = v;
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NAMES: [&str; 11] = [
        "count", "min", "max", "range", "mean", "std", "skew", "kurtosis", "median", "p25", "iqr",
    ];

    fn metrics() -> Vec<Metric> {
        Metric::parse_all(&NAMES).unwrap()
    }

    fn one(y: &[f64], t: &[f64], window: (f64, f64)) -> Vec<f64> {
        cso_batch(y, 1, y.len(), t, &[window], &metrics(), 1).unwrap()
    }

    fn close(got: &[f64], want: &[f64]) {
        for (k, (a, b)) in got.iter().zip(want).enumerate() {
            let ok = (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * (1.0 + b.abs());
            assert!(ok, "{}: {a} vs {b}", NAMES[k]);
        }
    }

    /// The statistics of the intervals by their definitions (count apart).
    fn direct(g: &[f64]) -> Vec<f64> {
        let k = g.len() as f64;
        let mean = g.iter().sum::<f64>() / k;
        let mom = |e: i32| g.iter().map(|v| (v - mean).powi(e)).sum::<f64>();
        let sigma = (mom(2) / k).sqrt();
        let mut s = g.to_vec();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| {
            let i = p * (k - 1.0);
            let lo = i.floor() as usize;
            let hi = (lo + 1).min(s.len() - 1);
            s[lo] + (s[hi] - s[lo]) * (i - lo as f64)
        };
        vec![
            0.0,
            s[0],
            s[s.len() - 1],
            s[s.len() - 1] - s[0],
            mean,
            if k > 1.0 { (mom(2) / (k - 1.0)).sqrt() } else { f64::NAN },
            if sigma > 0.0 { mom(3) / k / sigma.powi(3) } else { f64::NAN },
            if sigma > 0.0 { mom(4) / k / sigma.powi(4) - 3.0 } else { f64::NAN },
            q(0.5),
            q(0.25),
            q(0.75) - q(0.25),
        ]
    }

    #[test]
    fn intervals_by_hand() {
        // window [0, 12), observations on 2, 5, 9: the intervals 2, 3, 4 and 3 to the end
        let got = one(&[1.0; 3], &[2.0, 5.0, 9.0], (0.0, 12.0));
        let mut want = direct(&[2.0, 3.0, 4.0, 3.0]);
        want[0] = 3.0;
        close(&got, &want);
        assert_eq!((got[1], got[2], got[3], got[4]), (2.0, 4.0, 2.0, 3.0));
    }

    #[test]
    fn an_empty_window_is_one_interval_the_length_of_the_window() {
        let got = one(&[1.0; 3], &[2.0, 5.0, 9.0], (20.0, 30.0));
        assert_eq!((got[0], got[1], got[2], got[3], got[4]), (0.0, 10.0, 10.0, 0.0, 10.0));
        assert!(got[5].is_nan() && got[6].is_nan() && got[7].is_nan());
        assert_eq!((got[8], got[9], got[10]), (10.0, 10.0, 0.0));
        // the same where every observation of the window is missing
        let got = one(&[f64::NAN, f64::NAN, 1.0], &[2.0, 5.0, 9.0], (0.0, 8.0));
        assert_eq!((got[0], got[1], got[2], got[4]), (0.0, 8.0, 8.0, 8.0));
    }

    #[test]
    fn same_day_observations_count_but_make_no_interval() {
        // one on the start of the window and 5 twice: 4 observations, the intervals 5, 4 and 3
        let got = one(&[1.0; 4], &[0.0, 5.0, 5.0, 9.0], (0.0, 12.0));
        let mut want = direct(&[5.0, 4.0, 3.0]);
        want[0] = 4.0;
        close(&got, &want);
    }

    #[test]
    fn nan_is_not_an_observation_and_the_window_is_half_open() {
        let t = [1.0, 3.0, 4.0, 6.0, 8.0, 10.0];
        let y = [1.0, f64::NAN, 2.0, f64::INFINITY, 3.0, 4.0];
        // [3, 8): the NaN at 3, 4 valid, the infinite one at 6 not, 8 belongs to the next window
        let got = one(&y, &t, (3.0, 8.0));
        let mut want = direct(&[1.0, 4.0]);
        want[0] = 1.0;
        close(&got, &want);
        // [8, 11): 8 (on the start: no interval) and 10; the intervals 2 and 1
        let got = one(&y, &t, (8.0, 11.0));
        let mut want = direct(&[2.0, 1.0]);
        want[0] = 2.0;
        close(&got, &want);
    }

    #[test]
    fn windows_pixels_and_threads() {
        let (np, nt) = (23, 40);
        let t: Vec<f64> = (0..nt).map(|i| (i * 3 + i / 7) as f64).collect();
        let mut s = 7u64;
        let v: Vec<f64> = (0..np * nt)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                if (s >> 33) % 3 == 0 { f64::NAN } else { 1.0 }
            })
            .collect();
        let windows = [(0.0, 31.0), (31.0, 60.0), (60.0, 61.0), (500.0, 600.0), (10.0, 90.0)];
        let m = metrics();
        let a = cso_batch(&v, np, nt, &t, &windows, &m, 1).unwrap();
        for jobs in [3, -1] {
            let b = cso_batch(&v, np, nt, &t, &windows, &m, jobs).unwrap();
            assert!(a.iter().zip(&b).all(|(x, y)| x == y || (x.is_nan() && y.is_nan())));
        }
        // against the definition, by a plain loop on the steps of each window
        for p in [0, 5, 22] {
            for (j, &(a0, a1)) in windows.iter().enumerate() {
                let mut gaps = Vec::new();
                let (mut n, mut left) = (0.0, a0);
                for i in 0..nt {
                    if t[i] >= a0 && t[i] < a1 && v[p * nt + i].is_finite() {
                        n += 1.0;
                        if t[i] > left {
                            gaps.push(t[i] - left);
                            left = t[i];
                        }
                    }
                }
                gaps.push(a1 - left);
                let mut want = direct(&gaps);
                want[0] = n;
                let got: Vec<f64> = (0..m.len()).map(|k| a[p * m.len() * windows.len() + k * windows.len() + j]).collect();
                close(&got, &want);
            }
        }
    }

    #[test]
    fn bad_input_is_refused() {
        let m = metrics();
        assert!(cso_batch(&[1.0; 3], 1, 3, &[0.0, 2.0, 1.0], &[(0.0, 5.0)], &m, 1).is_err());
        assert!(cso_batch(&[1.0; 3], 1, 3, &[0.0, 1.0], &[(0.0, 5.0)], &m, 1).is_err());
        assert!(cso_batch(&[1.0; 3], 1, 3, &[0.0, 1.0, 2.0], &[(f64::NAN, 5.0)], &m, 1).is_err());
    }
}
