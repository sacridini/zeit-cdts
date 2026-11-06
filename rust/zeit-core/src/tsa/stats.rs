//! Statistics of a series (min, max, range, IQR, mean, standard deviation,
//! skewness, kurtosis, count, quantiles), of its folds by period, and the
//! standardization of a series.
//!
//! Ported from FORCE (Framework for Operational Radiometric Correction for
//! Environmental monitoring), Copyright (C) 2013-2025 David Frantz,
//! GPL-3.0-or-later, https://github.com/davidfrantz/force, src/modules:
//! higher-level/stm-hl.c, higher-level/fold-hl.c, higher-level/standardize-hl.c
//! and cross-level/stats-cl.c.
//!
//! The moments are Pébay's one-pass recurrences (2008, SANDIA report
//! SAND2008-6212), in the order FORCE runs them; the quantiles are the
//! linear-interpolation estimator R-7 (GSL's `gsl_stats_quantile_from_sorted_data`).
//! Unlike FORCE, which keeps int16 and truncates, the work is in f64 and the
//! results stay f64; NaN marks a missing observation, and a series without
//! one gives NaN (0 for the count).

use super::Sample;
use crate::parallel;

/// A statistic of a series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    Min,
    Max,
    Range,
    /// Q75 - Q25 (FORCE's fold; its STM truncates both first).
    Iqr,
    Mean,
    /// Standard deviation with `ddof` degrees of freedom less than the count.
    Std,
    /// M3 / sigma^3, sigma the population standard deviation (FORCE's SKW).
    Skew,
    /// Excess kurtosis: M4 / sigma^4 - 3, sigma the population standard deviation
    /// (FORCE's KRT, 3 subtracted).
    Kurtosis,
    /// The valid observations.
    Count,
    /// The quantile of probability `p` in [0, 1].
    Quantile(f64),
}

impl Metric {
    /// The metric of a name: `min`, `max`, `range`, `iqr`, `mean`, `std`, `skew`,
    /// `kurtosis`, `count`, `median` or `pNN` (an integer percentile 0-100).
    pub fn parse(name: &str) -> Result<Metric, String> {
        let key = name.trim().to_ascii_lowercase();
        Ok(match key.as_str() {
            "min" => Metric::Min,
            "max" => Metric::Max,
            "range" | "rng" => Metric::Range,
            "iqr" => Metric::Iqr,
            "mean" | "avg" => Metric::Mean,
            "std" => Metric::Std,
            "skew" | "skw" => Metric::Skew,
            "kurtosis" | "krt" => Metric::Kurtosis,
            "count" | "num" => Metric::Count,
            "median" => Metric::Quantile(0.5),
            _ => match key.strip_prefix('p').and_then(|d| d.parse::<u32>().ok()) {
                Some(p) if p <= 100 => Metric::Quantile(p as f64 / 100.0),
                _ => return Err(format!("unknown metric {name:?}")),
            },
        })
    }

    /// The metrics of names.
    pub fn parse_all<S: AsRef<str>>(names: &[S]) -> Result<Vec<Metric>, String> {
        names.iter().map(|n| Metric::parse(n.as_ref())).collect()
    }
}

/// The standardization of a series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Standardize {
    /// x - mean.
    Center,
    /// (x - mean) / std.
    Normalize,
}

impl Standardize {
    pub fn parse(name: &str) -> Result<Standardize, String> {
        match name.trim().to_ascii_lowercase().as_str() {
            "center" => Ok(Standardize::Center),
            "normalize" => Ok(Standardize::Normalize),
            _ => Err(format!("unknown method {name:?}: use \"center\" or \"normalize\"")),
        }
    }
}

// ---- Pébay's recurrences (cross-level/stats-cl.c) --------------------------

/// The running mean and central moments M2, M3, M4 (sums of powers of the
/// deviations from the mean) of a series.
#[derive(Clone, Copy, Default)]
struct Moments {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl Moments {
    /// `kurt_recurrence`: one more value.
    #[inline]
    fn push(&mut self, x: f64) {
        self.n += 1.0;
        let n = self.n;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let tmp = delta * delta_n * (n - 1.0);
        self.mean += delta_n;
        self.m4 += tmp * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2 - 4.0 * delta_n * self.m3;
        self.m3 += tmp * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += tmp;
    }

    fn std(&self, ddof: f64) -> f64 {
        let d = self.n - ddof;
        if d > 0.0 {
            (self.m2 / d).sqrt()
        } else {
            f64::NAN
        }
    }

    /// `skewness`: M3 / (n sigma^3), sigma^2 = M2 / n.
    fn skew(&self) -> f64 {
        if self.m2 > 0.0 {
            let sigma = (self.m2 / self.n).sqrt();
            self.m3 / (self.n * sigma * sigma * sigma)
        } else {
            f64::NAN
        }
    }

    /// `kurtosis` less 3.
    fn kurtosis(&self) -> f64 {
        if self.m2 > 0.0 {
            let var = self.m2 / self.n;
            self.m4 / (self.n * var * var) - 3.0
        } else {
            f64::NAN
        }
    }
}

// ---- quantiles -----------------------------------------------------------

/// With this many distinct quantiles or more, the values are sorted once
/// instead of selecting each; for fewer than `SHORT` values already from two
/// (measured with `quantile_cost`: a selection costs about 1.4 ns per value
/// and quantile on series of 100 to 2000 values, a sort 1.5 to 7 ns per value
/// whatever the count; short series sort in under 0.1 us).
const SORT_FROM: usize = 4;
const SHORT: usize = 48;

/// The R-7 quantile of probability `p` of sorted values (as
/// `gsl_stats_quantile_from_sorted_data`).
fn quantile_sorted(s: &[f64], p: f64) -> f64 {
    let n = s.len();
    let index = p * (n - 1) as f64;
    let lhs = index.floor() as usize;
    let delta = index - lhs as f64;
    if lhs + 1 >= n {
        s[lhs.min(n - 1)]
    } else {
        (1.0 - delta) * s[lhs] + delta * s[lhs + 1]
    }
}

/// The same value, by selection: reorders `s`, sorts nothing. The R-7 quantile
/// needs the `lhs`-th and the next value of the sorted order: the first is what
/// `select_nth_unstable` places, the second the least of what follows it.
fn quantile_select(s: &mut [f64], p: f64) -> f64 {
    let n = s.len();
    let index = p * (n - 1) as f64;
    let lhs = index.floor() as usize;
    let delta = index - lhs as f64;
    let (_, &mut lo, right) = s.select_nth_unstable_by(lhs.min(n - 1), f64::total_cmp);
    if lhs + 1 >= n {
        return lo;
    }
    let hi = right.iter().copied().fold(f64::INFINITY, f64::min);
    (1.0 - delta) * lo + delta * hi
}

// ---- the metrics of one series ---------------------------------------------

/// The metrics of a series, planned once for a batch.
pub struct Plan {
    metrics: Vec<Metric>,
    ddof: f64,
    /// The distinct probabilities the metrics need, ascending.
    probs: Vec<f64>,
    moments: bool,
}

/// The buffers of one thread.
#[derive(Default)]
pub struct Work {
    /// The valid values of the series (or of the fold), in time order.
    vals: Vec<f64>,
    /// The values for the quantiles, reordered.
    sorted: Vec<f64>,
    /// The quantiles, one for each of the plan's probabilities.
    q: Vec<f64>,
    /// The metrics of one fold.
    tmp: Vec<f64>,
}

impl Plan {
    pub fn new(metrics: &[Metric], ddof: f64) -> Plan {
        let mut probs: Vec<f64> = Vec::new();
        let mut want = |p: f64| {
            if !probs.contains(&p) {
                probs.push(p);
            }
        };
        for m in metrics {
            match *m {
                Metric::Quantile(p) => want(p),
                Metric::Iqr => {
                    want(0.25);
                    want(0.75);
                }
                _ => {}
            }
        }
        probs.sort_by(f64::total_cmp);
        let moments = metrics
            .iter()
            .any(|m| matches!(m, Metric::Mean | Metric::Std | Metric::Skew | Metric::Kurtosis));
        Plan {
            metrics: metrics.to_vec(),
            ddof,
            probs,
            moments,
        }
    }

    pub fn n_metrics(&self) -> usize {
        self.metrics.len()
    }

    fn prob(&self, p: f64) -> usize {
        self.probs.iter().position(|&q| q == p).expect("a planned probability")
    }

    /// The metrics of `vals` (no NaN) into `out` (one per metric).
    pub(super) fn eval(&self, vals: &[f64], sorted: &mut Vec<f64>, q: &mut Vec<f64>, out: &mut [f64]) {
        let n = vals.len();
        if n == 0 {
            for (o, m) in out.iter_mut().zip(&self.metrics) {
                *o = if *m == Metric::Count { 0.0 } else { f64::NAN };
            }
            return;
        }
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut mo = Moments::default();
        for &x in vals {
            min = min.min(x);
            max = max.max(x);
            if self.moments {
                mo.push(x);
            }
        }
        if !self.probs.is_empty() {
            q.clear();
            sorted.clear();
            sorted.extend_from_slice(vals);
            if self.probs.len() >= SORT_FROM || (n < SHORT && self.probs.len() >= 2) {
                sorted.sort_unstable_by(f64::total_cmp);
                q.extend(self.probs.iter().map(|&p| quantile_sorted(sorted, p)));
            } else {
                for &p in &self.probs {
                    q.push(quantile_select(sorted, p));
                }
            }
        }
        for (o, m) in out.iter_mut().zip(&self.metrics) {
            *o = match *m {
                Metric::Min => min,
                Metric::Max => max,
                Metric::Range => max - min,
                Metric::Iqr => q[self.prob(0.75)] - q[self.prob(0.25)],
                Metric::Mean => mo.mean,
                Metric::Std => mo.std(self.ddof),
                Metric::Skew => mo.skew(),
                Metric::Kurtosis => mo.kurtosis(),
                Metric::Count => n as f64,
                Metric::Quantile(p) => q[self.prob(p)],
            };
        }
    }

    /// The metrics of the valid values of `y` into `out`.
    fn series<T: Sample>(&self, y: &[T], w: &mut Work, out: &mut [f64]) {
        let Work { vals, sorted, q, .. } = w;
        let n = gather(vals, y.iter().copied());
        self.eval(&vals[..n], sorted, q, out);
    }

    /// Whether the metrics are the mean and the count alone, which the lanes of
    /// `fold_mean` compute.
    fn mean_count_only(&self) -> bool {
        self.metrics.iter().all(|m| matches!(m, Metric::Mean | Metric::Count))
    }
}

/// The valid values of `it` into the front of `vals` (which only grows), their
/// number returned. Every value is stored and the position advances on a valid
/// one: no branch on the gaps, which are as unpredictable as the clouds.
#[inline]
fn gather<T: Sample, I: ExactSizeIterator<Item = T>>(vals: &mut Vec<f64>, it: I) -> usize {
    if vals.len() < it.len() {
        vals.resize(it.len(), 0.0);
    }
    let mut k = 0;
    for x in it {
        let x = x.to_f64();
        vals[k] = x;
        k += usize::from(!x.is_nan());
    }
    k
}

/// Pixels the mean of a fold runs at once: the recurrence of the mean is a chain
/// of a subtraction, a division and an addition (about 20 cycles a value), and
/// the series of different pixels are independent chains the processor overlaps.
const LANES: usize = 4;

/// The mean and the count of every group of the steps of `LANES` series at
/// once (`y` has the series; a block with fewer pixels repeats its first),
/// into `rows` (the rows of the pixels, `m * n_groups` each), as `Plan::eval`
/// gives them, and by the same recurrence.
fn fold_mean<T: Sample>(
    plan: &Plan,
    y: &[&[T]; LANES],
    steps: &[usize],
    starts: &[usize],
    rows: &mut [f64],
    width: usize,
) {
    let n_groups = starts.len() - 1;
    for g in 0..n_groups {
        let mut n = [0.0f64; LANES];
        let mut mean = [0.0f64; LANES];
        for &t in &steps[starts[g]..starts[g + 1]] {
            for l in 0..LANES {
                let x = y[l][t].to_f64();
                let ok = !x.is_nan();
                let nn = n[l] + 1.0;
                let step = (x - mean[l]) / nn;
                mean[l] = if ok { mean[l] + step } else { mean[l] };
                n[l] = if ok { nn } else { n[l] };
            }
        }
        for (l, row) in rows.chunks_mut(width).enumerate() {
            for (k, m) in plan.metrics.iter().enumerate() {
                row[k * n_groups + g] = match m {
                    Metric::Count => n[l],
                    _ if n[l] == 0.0 => f64::NAN,
                    _ => mean[l],
                };
            }
        }
    }
}

/// The metrics of the valid values of one series (a test and
/// single-series helper): one value per metric.
pub fn series_metrics(y: &[f64], metrics: &[Metric], ddof: f64) -> Vec<f64> {
    let plan = Plan::new(metrics, ddof);
    let mut out = vec![f64::NAN; metrics.len()];
    plan.series(y, &mut Work::default(), &mut out);
    out
}

/// The metrics of every series of `values` (pixels, time), over the whole time
/// axis (FORCE's STM): (pixels, metrics), flat.
pub fn stm_batch<T: Sample>(
    values: &[T],
    n_pixels: usize,
    n_time: usize,
    metrics: &[Metric],
    ddof: f64,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    let plan = Plan::new(metrics, ddof);
    let m = plan.n_metrics();
    let mut out = parallel::uninit_f64(n_pixels * m); // every row is written
    parallel::for_each_row(&mut out, m, n_jobs, Work::default, |w, p, row| {
        plan.series(&values[p * n_time..(p + 1) * n_time], w, row);
    });
    out
}

/// The metrics of every series of `values` (pixels, time) for each group of
/// time steps (FORCE's fold: by year, quarter, month, week or day of the year):
/// (pixels, metrics, groups), flat. `group[t]` is the group of time step `t`, in
/// 0..n_groups, or -1 to leave the step out; a group without a valid
/// observation gives NaN (0 for the count).
#[allow(clippy::too_many_arguments)]
pub fn fold_batch<T: Sample>(
    values: &[T],
    n_pixels: usize,
    n_time: usize,
    group: &[i32],
    n_groups: usize,
    metrics: &[Metric],
    ddof: f64,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    assert_eq!(group.len(), n_time, "one group per time step");
    // The steps of each group, indexed once (a counting sort of the steps).
    let mut starts = vec![0usize; n_groups + 1];
    for &g in group {
        if g >= 0 {
            assert!((g as usize) < n_groups, "group {g} out of 0..{n_groups}");
            starts[g as usize + 1] += 1;
        }
    }
    for g in 0..n_groups {
        starts[g + 1] += starts[g];
    }
    let mut fill = starts.clone();
    let mut steps = vec![0usize; starts[n_groups]];
    for (t, &g) in group.iter().enumerate() {
        if g >= 0 {
            steps[fill[g as usize]] = t;
            fill[g as usize] += 1;
        }
    }
    let plan = Plan::new(metrics, ddof);
    let m = plan.n_metrics();
    let width = m * n_groups;
    if n_pixels == 0 || width == 0 {
        return vec![f64::NAN; n_pixels * width];
    }
    let mut out = parallel::uninit_f64(n_pixels * width); // every row is written
    if plan.mean_count_only() {
        // blocks of LANES pixels (the last block repeats its first pixel to fill the lanes)
        parallel::for_each_row(&mut out, width * LANES, n_jobs, || (), |_, b, rows| {
            let first = b * LANES;
            let series = |l: usize| {
                let p = if first + l < n_pixels { first + l } else { first };
                &values[p * n_time..(p + 1) * n_time]
            };
            let y = [series(0), series(1), series(2), series(3)];
            fold_mean(&plan, &y, &steps, &starts, rows, width);
        });
        return out;
    }
    parallel::for_each_row(&mut out, width, n_jobs, Work::default, |w, p, row| {
        let y = &values[p * n_time..(p + 1) * n_time];
        let Work { vals, sorted, q, tmp } = w;
        tmp.resize(m, f64::NAN);
        for g in 0..n_groups {
            let n = gather(vals, steps[starts[g]..starts[g + 1]].iter().map(|&t| y[t]));
            plan.eval(&vals[..n], sorted, q, tmp);
            for (k, &v) in tmp.iter().enumerate() {
                row[k * n_groups + g] = v;
            }
        }
    });
    out
}

/// Standardizes one series in place: the mean and the variance over the valid
/// values by `var_recurrence`; fewer than 2 valid values leave it as it is.
fn standardize_series(y: &mut [f64], method: Standardize, ddof: f64) {
    let (mut k, mut avg, mut var) = (0.0f64, 0.0f64, 0.0f64);
    for &x in y.iter() {
        if x.is_nan() {
            continue;
        }
        k += 1.0;
        if k == 1.0 {
            avg = x;
        } else {
            let old = avg;
            avg = old + (x - old) / k;
            var += (x - old) * (x - avg);
        }
    }
    if k < 2.0 {
        return;
    }
    let std = match method {
        Standardize::Center => 1.0,
        Standardize::Normalize => {
            let d = k - ddof;
            if d > 0.0 {
                (var / d).sqrt()
            } else {
                f64::NAN
            }
        }
    };
    for x in y.iter_mut() {
        if !x.is_nan() {
            *x = (*x - avg) / std;
        }
    }
}

/// Every series of `values` (pixels, time) standardized along time (FORCE's
/// standardize): centered on its mean, or also divided by its standard
/// deviation. A series with fewer than 2 valid values stays as it is; a
/// constant one gives NaN when normalized. (pixels, time), flat.
pub fn standardize_batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    method: Standardize,
    ddof: f64,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    let mut out = values.to_vec();
    parallel::for_each_row(&mut out, n_time, n_jobs, || (), |_, _, row| {
        standardize_series(row, method, ddof);
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [&str; 12] = [
        "min", "max", "range", "iqr", "mean", "std", "skew", "kurtosis", "count", "median", "p10", "p99",
    ];

    fn metrics() -> Vec<Metric> {
        Metric::parse_all(&ALL).unwrap()
    }

    /// A series with values in any order, ties and negatives.
    fn series(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 2000) as f64 / 10.0 - 80.0
            })
            .collect()
    }

    /// The statistics by their definitions: two passes, a full sort.
    fn direct(y: &[f64], ddof: f64) -> Vec<f64> {
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let mom = |k: i32| y.iter().map(|v| (v - mean).powi(k)).sum::<f64>();
        let (m2, m3, m4) = (mom(2), mom(3), mom(4));
        let sigma = (m2 / n).sqrt();
        let mut s = y.to_vec();
        s.sort_by(f64::total_cmp);
        let q = |p: f64| {
            let i = p * (n - 1.0);
            let lo = i.floor() as usize;
            let hi = (lo + 1).min(s.len() - 1);
            s[lo] + (s[hi] - s[lo]) * (i - lo as f64)
        };
        vec![
            s[0],
            s[s.len() - 1],
            s[s.len() - 1] - s[0],
            q(0.75) - q(0.25),
            mean,
            (m2 / (n - ddof)).sqrt(),
            m3 / n / sigma.powi(3),
            m4 / n / sigma.powi(4) - 3.0,
            n,
            q(0.5),
            q(0.10),
            q(0.99),
        ]
    }

    fn close(a: &[f64], b: &[f64], what: &str) {
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            let ok = (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * (1.0 + y.abs());
            assert!(ok, "{what} metric {}: {x} vs {y}", ALL[k]);
        }
    }

    #[test]
    fn metrics_match_the_definitions() {
        for n in [2usize, 3, 4, 7, 16, 101, 500] {
            for ddof in [0.0, 1.0] {
                let y = series(n, n as u64);
                close(&series_metrics(&y, &metrics(), ddof), &direct(&y, ddof), &format!("n={n} ddof={ddof}"));
            }
        }
    }

    #[test]
    fn nan_is_dropped_and_an_empty_series_is_nan() {
        let y = vec![f64::NAN, 3.0, f64::NAN, 1.0, 2.0, 10.0];
        close(&series_metrics(&y, &metrics(), 1.0), &direct(&[3.0, 1.0, 2.0, 10.0], 1.0), "with gaps");
        let e = series_metrics(&[f64::NAN; 4], &metrics(), 1.0);
        for (v, name) in e.iter().zip(ALL) {
            assert!(if name == "count" { *v == 0.0 } else { v.is_nan() }, "{name}: {v}");
        }
        assert_eq!(series_metrics(&[], &metrics(), 1.0)[8], 0.0);
    }

    #[test]
    fn one_value_and_constant_series() {
        let m = metrics();
        let one = series_metrics(&[4.0], &m, 1.0);
        // min max range iqr mean, then the std with ddof 1 (n - 1 = 0), skew, kurtosis undefined
        assert_eq!((one[0], one[1], one[2], one[3], one[4]), (4.0, 4.0, 0.0, 0.0, 4.0));
        assert!(one[5].is_nan() && one[6].is_nan() && one[7].is_nan());
        assert_eq!((one[8], one[9], one[10], one[11]), (1.0, 4.0, 4.0, 4.0));
        assert_eq!(series_metrics(&[4.0], &m, 0.0)[5], 0.0);
        let c = series_metrics(&[2.5; 6], &m, 1.0);
        assert_eq!((c[5], c[2]), (0.0, 0.0));
        assert!(c[6].is_nan() && c[7].is_nan());
    }

    #[test]
    fn selection_and_sorting_give_the_same_quantiles() {
        let y = series(257, 9);
        let probs = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0];
        let all: Vec<Metric> = probs.iter().map(|&p| Metric::Quantile(p)).collect();
        let together = series_metrics(&y, &all, 1.0); // sorted once
        for (k, &p) in probs.iter().enumerate() {
            let alone = series_metrics(&y, &[Metric::Quantile(p)], 1.0); // selected
            assert_eq!(alone[0], together[k], "p = {p}");
        }
    }

    #[test]
    fn names_parse() {
        assert_eq!(Metric::parse("P05"), Ok(Metric::Quantile(0.05)));
        assert_eq!(Metric::parse("median"), Ok(Metric::Quantile(0.5)));
        assert_eq!(Metric::parse("kurtosis"), Ok(Metric::Kurtosis));
        assert!(Metric::parse("p101").is_err());
        assert!(Metric::parse("variance").is_err());
        assert_eq!(Standardize::parse("Center"), Ok(Standardize::Center));
        assert!(Standardize::parse("zscore").is_err());
    }

    #[test]
    fn stm_batch_runs_every_pixel_on_any_threads() {
        let (np, nt) = (37, 23);
        let mut v = series(np * nt, 3);
        v[5] = f64::NAN;
        v[nt * 4..nt * 5].fill(f64::NAN); // a pixel without observations
        let m = metrics();
        let one = stm_batch(&v, np, nt, &m, 1.0, 1);
        for n_jobs in [3, -1] {
            let got = stm_batch(&v, np, nt, &m, 1.0, n_jobs);
            assert_eq!(one.len(), np * m.len());
            assert!(one.iter().zip(&got).all(|(a, b)| a == b || (a.is_nan() && b.is_nan())));
        }
        for p in [0, 4, 36] {
            close(&one[p * m.len()..(p + 1) * m.len()], &series_metrics(&v[p * nt..(p + 1) * nt], &m, 1.0), "pixel");
        }
        assert_eq!(one[4 * m.len() + 8], 0.0);
    }

    #[test]
    fn fold_batch_groups_the_steps() {
        let (np, nt) = (3, 12);
        let v = series(np * nt, 5);
        // steps 0..4 in group 2, 4..8 in group 0, 8..10 left out, 10 and 11 in group 1; group 3 empty
        let group = [2, 2, 2, 2, 0, 0, 0, 0, -1, -1, 1, 1];
        let m = Metric::parse_all(&["mean", "count", "p25"]).unwrap();
        let out = fold_batch(&v, np, nt, &group, 4, &m, 1.0, 2);
        assert_eq!(out.len(), np * 3 * 4);
        for p in 0..np {
            let y = &v[p * nt..(p + 1) * nt];
            for (g, steps) in [(0usize, 4..8), (1, 10..12), (2, 0..4)] {
                let want = series_metrics(&y[steps], &m, 1.0);
                for k in 0..3 {
                    assert_eq!(out[p * 12 + k * 4 + g], want[k], "pixel {p} group {g} metric {k}");
                }
            }
            assert_eq!(out[p * 12 + 4 + 3], 0.0); // count of the empty group
            assert!(out[p * 12 + 3].is_nan() && out[p * 12 + 8 + 3].is_nan());
        }
    }

    #[test]
    fn fold_of_the_mean_by_lanes_is_the_recurrence_bit_for_bit() {
        // 11 pixels (not a multiple of the lanes), gaps, a pixel without observations, an empty group, f32 too
        let (np, nt) = (11, 40);
        let mut v = series(np * nt, 8);
        for i in (0..v.len()).step_by(3) {
            v[i] = f64::NAN;
        }
        v[nt * 6..nt * 7].fill(f64::NAN);
        let group: Vec<i32> = (0..nt as i32).map(|t| if t % 7 == 6 { -1 } else { [0, 0, 1, 1, 1, 3][t as usize % 6] }).collect();
        for names in [vec!["mean", "count"], vec!["count"], vec!["mean"], vec!["count", "mean", "count"]] {
            let m = Metric::parse_all(&names).unwrap();
            for jobs in [1, 3] {
                let got = fold_batch(&v, np, nt, &group, 4, &m, 1.0, jobs);
                let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                let got32 = fold_batch(&v32, np, nt, &group, 4, &m, 1.0, jobs);
                for p in 0..np {
                    for g in 0..4 {
                        let y: Vec<f64> = (0..nt).filter(|&t| group[t] == g as i32).map(|t| v[p * nt + t]).collect();
                        let y32: Vec<f64> = y.iter().map(|&x| f64::from(x as f32)).collect();
                        let (want, want32) = (series_metrics(&y, &m, 1.0), series_metrics(&y32, &m, 1.0));
                        for k in 0..m.len() {
                            let i = (p * m.len() + k) * 4 + g;
                            assert!(got[i] == want[k] || (got[i].is_nan() && want[k].is_nan()), "{names:?} px {p} g {g}");
                            assert!(got32[i] == want32[k] || (got32[i].is_nan() && want32[k].is_nan()), "f32 {names:?} px {p} g {g}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn standardize_centers_and_normalizes() {
        let y = vec![1.0, f64::NAN, 4.0, 7.0, 2.0];
        let valid = [1.0, 4.0, 7.0, 2.0];
        let mean = valid.iter().sum::<f64>() / 4.0;
        let std = (valid.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 3.0).sqrt();
        let c = standardize_batch(&y, 1, 5, Standardize::Center, 1.0, 1);
        let z = standardize_batch(&y, 1, 5, Standardize::Normalize, 1.0, 1);
        assert!(c[1].is_nan() && z[1].is_nan());
        for (i, v) in [(0, 1.0),(2, 4.0), (3, 7.0), (4, 2.0)] {
            assert!((c[i] - (v - mean)).abs() < 1e-12);
            assert!((z[i] - (v - mean) / std).abs() < 1e-12);
        }
        // fewer than two valid values: as they are; a constant series normalizes to NaN
        let w = [f64::NAN, 5.0, f64::NAN, 3.0, 3.0, 3.0];
        let o = standardize_batch(&w, 2, 3, Standardize::Normalize, 1.0, 2);
        assert!(o[0].is_nan() && o[1] == 5.0 && o[2].is_nan());
        assert!(o[3..].iter().all(|v| v.is_nan()));
        let o = standardize_batch(&w, 2, 3, Standardize::Center, 1.0, 2);
        assert_eq!(&o[3..], &[0.0, 0.0, 0.0]);
    }

    /// Selection against a full sort, for the choice of `SORT_FROM`:
    /// `cargo test -p zeit-core --release quantile_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn quantile_cost() {
        use std::time::Instant;
        for n in [24usize, 114, 365, 2000] {
            let y = series(n, 1);
            for k in 1..=4usize {
                let probs: Vec<f64> = (0..k).map(|i| (i as f64 + 1.0) / (k as f64 + 1.0)).collect();
                let reps = 2_000_000 / n;
                let mut buf = Vec::new();
                let mut sink = 0.0;
                let t = Instant::now();
                for _ in 0..reps {
                    buf.clear();
                    buf.extend_from_slice(&y);
                    for &p in &probs {
                        sink += quantile_select(&mut buf, p);
                    }
                }
                let select = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
                let t = Instant::now();
                for _ in 0..reps {
                    buf.clear();
                    buf.extend_from_slice(&y);
                    buf.sort_unstable_by(f64::total_cmp);
                    for &p in &probs {
                        sink += quantile_sorted(&buf, p);
                    }
                }
                let sort = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
                println!("n={n:5} quantiles={k}: select {select:8.2} us   sort {sort:8.2} us   ({sink:.0})");
            }
        }
    }
}
