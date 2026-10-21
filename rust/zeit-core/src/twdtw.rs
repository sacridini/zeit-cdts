//! Time-Weighted Dynamic Time Warping (Maus et al. 2016): a port of
//! src/twdtw.cpp, same operations in the same order (the multi-band distance
//! as Eigen's `norm()` sums it), so both give the same bits.
//!
//! One difference that only saves time: the C++ builds the table of the
//! logistic time weight (2001 `exp`) for every series; here a batch builds it
//! once.

use crate::eigen;
use crate::parallel;

/// The parameters of `fit_twdtw` and `fit_twdtw_batch`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// The logistic time weight: alpha / (1 + exp(-beta (dt - gamma))).
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    /// Sakoe-Chiba band: the most time elapsed between matched dates.
    pub max_time_warp: i32,
    /// Open-ended DTW: the pattern may match any part of the series.
    pub subsequence_matching: bool,
    /// Length of the time cycle in date units (366 for days of the year): the
    /// time elapsed between two dates goes around it, as R's twdtw. 0: the
    /// plain difference of the dates.
    pub cycle_length: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            alpha: 0.1,
            beta: 0.05,
            gamma: 50.0,
            max_time_warp: 365,
            subsequence_matching: false,
            cycle_length: 0.0,
        }
    }
}

/// Time elapsed between two dates, around the cycle when there is one (twdtw's ellapsed).
#[inline]
pub fn elapsed_time(a: i32, b: i32, cycle_length: f64) -> i32 {
    let mut r = if a > b { a.wrapping_sub(b) } else { b.wrapping_sub(a) };
    if cycle_length > 0.0 && r as f64 > cycle_length / 2.0 {
        let wrapped = cycle_length - r as f64;
        r = (if wrapped < 0.0 { -wrapped } else { wrapped }) as i32;
    }
    r
}

/// The time weight of every elapsed time up to 2000, as the C++'s TWDTW_LUT.
pub struct TimeWeight {
    lut: Vec<f64>,
    cycle_length: f64,
}

impl TimeWeight {
    const MAX_DIFF: usize = 2000;

    pub fn new(p: &Params) -> TimeWeight {
        let lut = (0..=Self::MAX_DIFF)
            .map(|i| p.alpha / (1.0 + (-p.beta * (i as f64 - p.gamma)).exp()))
            .collect();
        TimeWeight {
            lut,
            cycle_length: p.cycle_length,
        }
    }

    #[inline]
    fn get(&self, t1: i32, t2: i32) -> f64 {
        self.at(elapsed_time(t1, t2, self.cycle_length))
    }

    /// The weight of an elapsed time.
    #[inline]
    fn at(&self, dt: i32) -> f64 {
        match usize::try_from(dt) {
            Ok(dt) if dt < self.lut.len() => self.lut[dt],
            _ => self.lut[self.lut.len() - 1],
        }
    }
}

/// The distance between the observations of two dates: |a - b| for one band,
/// the Euclidean norm of a - b for several.
#[inline]
fn spatial_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() == 1 {
        (a[0] - b[0]).abs()
    } else {
        eigen::distance(a, b)
    }
}

#[inline]
fn min3(a: f64, b: f64, c: f64) -> f64 {
    // std::min({a, b, c}): the first of the smallest
    let mut m = a;
    if b < m {
        m = b;
    }
    if c < m {
        m = c;
    }
    m
}

/// A series (or a pattern): `values` holds `num_bands` values per date.
#[derive(Clone, Copy)]
pub struct Series<'a> {
    pub values: &'a [f64],
    pub dates: &'a [i32],
    pub num_bands: usize,
}

impl<'a> Series<'a> {
    #[inline]
    fn at(&self, i: usize) -> &'a [f64] {
        &self.values[i * self.num_bands..(i + 1) * self.num_bands]
    }
}

/// The two rows of the cost matrix a distance needs, reused between series.
#[derive(Default)]
pub struct Rows {
    prev: Vec<f64>,
    curr: Vec<f64>,
}

/// The TWDTW distance of `ts` to `pattern`: infinite when the cost of a whole
/// row exceeds `abort_threshold` (early abandoning) or nothing matches.
pub fn distance(
    ts: Series,
    pattern: Series,
    params: &Params,
    weight: &TimeWeight,
    abort_threshold: f64,
    rows: &mut Rows,
) -> f64 {
    let n = ts.dates.len();
    let m = pattern.dates.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
    let Rows { prev, curr } = rows;
    prev.clear();
    prev.resize(m + 1, f64::INFINITY);
    curr.clear();
    curr.resize(m + 1, f64::INFINITY);
    prev[0] = 0.0;

    let mut final_min = f64::INFINITY;
    for i in 1..=n {
        curr[0] = if params.subsequence_matching {
            0.0
        } else {
            f64::INFINITY
        };
        let mut min_in_row = f64::INFINITY;
        let (ti, vi) = (ts.dates[i - 1], ts.at(i - 1));
        // the elapsed time once per cell (the C++ takes it again for the weight:
        // the same value), and the one-band distance without the generic path
        let (cycle, max_warp) = (params.cycle_length, params.max_time_warp);
        let (p_row, c_row) = (&prev[..=m], &mut curr[..=m]);
        if ts.num_bands == 1 {
            let (v0, pv) = (vi[0], &pattern.values[..m]);
            for j in 1..=m {
                let dt = elapsed_time(ti, pattern.dates[j - 1], cycle);
                if dt > max_warp {
                    c_row[j] = f64::INFINITY;
                    continue;
                }
                let cvalue = (v0 - pv[j - 1]).abs() + weight.at(dt);
                let c = cvalue + min3(p_row[j - 1], p_row[j], c_row[j - 1]);
                c_row[j] = c;
                if c < min_in_row {
                    min_in_row = c;
                }
            }
        } else {
            for j in 1..=m {
                let dt = elapsed_time(ti, pattern.dates[j - 1], cycle);
                if dt > max_warp {
                    c_row[j] = f64::INFINITY;
                    continue;
                }
                let cvalue = spatial_distance(vi, pattern.at(j - 1)) + weight.at(dt);
                let c = cvalue + min3(p_row[j - 1], p_row[j], c_row[j - 1]);
                c_row[j] = c;
                if c < min_in_row {
                    min_in_row = c;
                }
            }
        }
        if min_in_row > abort_threshold {
            return f64::INFINITY;
        }
        if params.subsequence_matching && curr[m] < final_min {
            final_min = curr[m];
        }
        std::mem::swap(prev, curr);
    }
    if params.subsequence_matching {
        final_min
    } else {
        prev[m]
    }
}

/// The TWDTW distance and the warping path: (series index, pattern index)
/// pairs in order; no path when the distance is infinite.
pub fn distance_and_path(
    ts: Series,
    pattern: Series,
    params: &Params,
    weight: &TimeWeight,
    abort_threshold: f64,
) -> (f64, Vec<(i32, i32)>) {
    let n = ts.dates.len();
    let m = pattern.dates.len();
    if n == 0 || m == 0 {
        return (f64::INFINITY, Vec::new());
    }
    let w = m + 1;
    let mut d = vec![f64::INFINITY; (n + 1) * w];
    if params.subsequence_matching {
        for i in 0..=n {
            d[i * w] = 0.0;
        }
    } else {
        d[0] = 0.0;
    }

    for i in 1..=n {
        let mut min_in_row = f64::INFINITY;
        let (ti, vi) = (ts.dates[i - 1], ts.at(i - 1));
        for j in 1..=m {
            let tj = pattern.dates[j - 1];
            if elapsed_time(ti, tj, params.cycle_length) > params.max_time_warp {
                continue;
            }
            let cvalue = spatial_distance(vi, pattern.at(j - 1)) + weight.get(ti, tj);
            let v = cvalue + min3(d[(i - 1) * w + j - 1], d[(i - 1) * w + j], d[i * w + j - 1]);
            d[i * w + j] = v;
            if v < min_in_row {
                min_in_row = v;
            }
        }
        if min_in_row > abort_threshold {
            return (f64::INFINITY, Vec::new());
        }
    }

    let mut best_i = n;
    let dist = if params.subsequence_matching {
        let mut min_val = f64::INFINITY;
        for i in 1..=n {
            if d[i * w + m] < min_val {
                min_val = d[i * w + m];
                best_i = i;
            }
        }
        min_val
    } else {
        d[n * w + m]
    };

    let mut path = Vec::new();
    if dist != f64::INFINITY {
        let (mut i, mut j) = (best_i, m);
        while i > 0 && j > 0 {
            path.push((i as i32 - 1, j as i32 - 1));
            let diag = d[(i - 1) * w + j - 1];
            let up = d[(i - 1) * w + j];
            let left = d[i * w + j - 1];
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= diag && up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.reverse();
    }
    (dist, path)
}

/// The scratch buffers of one thread of a batch.
#[derive(Default)]
struct Scratch {
    kept: Vec<f64>,
    kept_dates: Vec<i32>,
    rows: Rows,
}

/// The distance of every series of `values` (`n_pixels` x T dates x
/// `num_bands`, row-major; `dates` the T dates) to `pattern`, on the threads of
/// `n_jobs`. Dates with a missing value (NaN in any band) leave a series, as
/// R's twdtw drops incomplete cases; NaN for a series with none left. With a
/// finite `abort_threshold` (and no subsequence matching), LB_Keogh skips the
/// series whose lower bound exceeds it (infinite distance).
#[allow(clippy::too_many_arguments)]
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    dates: &[i32],
    num_bands: usize,
    pattern: Series,
    params: &Params,
    abort_threshold: f64,
    n_jobs: i32,
) -> Vec<f64> {
    let t_len = dates.len();
    let len = t_len * num_bands;
    assert_eq!(values.len(), n_pixels * len, "values must be pixels x dates x bands");
    let weight = TimeWeight::new(params);

    // LB_Keogh envelope of the pattern at every date of the series
    let use_lb = abort_threshold < f64::INFINITY && !params.subsequence_matching;
    let mut upper = vec![f64::NEG_INFINITY; len];
    let mut lower = vec![f64::INFINITY; len];
    let mut valid_window = vec![false; t_len];
    if use_lb {
        for t in 0..t_len {
            for p in 0..pattern.dates.len() {
                if elapsed_time(dates[t], pattern.dates[p], params.cycle_length) <= params.max_time_warp {
                    valid_window[t] = true;
                    for b in 0..num_bands {
                        let val = pattern.values[p * num_bands + b];
                        upper[t * num_bands + b] = upper[t * num_bands + b].max(val);
                        lower[t * num_bands + b] = lower[t * num_bands + b].min(val);
                    }
                }
            }
        }
    }

    let mut out = vec![0.0; n_pixels];
    parallel::for_each_row(&mut out, 1, n_jobs, Scratch::default, |sc, p, out| {
        let ts = &values[p * len..(p + 1) * len];
        let complete = !ts.iter().any(|v| v.is_nan());
        let series = if complete {
            Series {
                values: ts,
                dates,
                num_bands,
            }
        } else {
            sc.kept.clear();
            sc.kept_dates.clear();
            for (t, obs) in ts.chunks_exact(num_bands).enumerate() {
                if obs.iter().all(|v| !v.is_nan()) {
                    sc.kept.extend_from_slice(obs);
                    sc.kept_dates.push(dates[t]);
                }
            }
            if sc.kept_dates.is_empty() {
                out[0] = f64::NAN;
                return;
            }
            Series {
                values: &sc.kept,
                dates: &sc.kept_dates,
                num_bands,
            }
        };

        if use_lb && complete {
            let mut lb = 0.0;
            for t in 0..t_len {
                if !valid_window[t] {
                    lb = f64::INFINITY;
                    break;
                }
                let spatial = if num_bands == 1 {
                    let v = ts[t];
                    if v > upper[t] {
                        v - upper[t]
                    } else if v < lower[t] {
                        lower[t] - v
                    } else {
                        0.0
                    }
                } else {
                    let mut sq = 0.0;
                    for b in 0..num_bands {
                        let (v, u, l) = (
                            ts[t * num_bands + b],
                            upper[t * num_bands + b],
                            lower[t * num_bands + b],
                        );
                        if v > u {
                            sq += (v - u) * (v - u);
                        } else if v < l {
                            sq += (l - v) * (l - v);
                        }
                    }
                    sq.sqrt()
                };
                lb += spatial;
                if lb > abort_threshold {
                    break;
                }
            }
            if lb > abort_threshold {
                out[0] = f64::INFINITY; // pruned by LB_Keogh
                return;
            }
        }

        out[0] = distance(series, pattern, params, &weight, abort_threshold, &mut sc.rows);
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series<'a>(values: &'a [f64], dates: &'a [i32]) -> Series<'a> {
        Series {
            values,
            dates,
            num_bands: 1,
        }
    }

    #[test]
    fn identical_series_cost_only_the_time_weight() {
        let dates: Vec<i32> = (0..10).map(|i| i * 16).collect();
        let v: Vec<f64> = (0..10).map(|i| (i as f64).sin()).collect();
        let p = Params::default();
        let w = TimeWeight::new(&p);
        let d = distance(
            series(&v, &dates),
            series(&v, &dates),
            &p,
            &w,
            f64::INFINITY,
            &mut Rows::default(),
        );
        let per_date = p.alpha / (1.0 + (-p.beta * (0.0 - p.gamma)).exp());
        assert!((d - 10.0 * per_date).abs() < 1e-12);
        let (dp, path) = distance_and_path(series(&v, &dates), series(&v, &dates), &p, &w, f64::INFINITY);
        assert_eq!(dp.to_bits(), d.to_bits());
        assert_eq!(path, (0..10).map(|i| (i, i)).collect::<Vec<_>>());
    }

    #[test]
    fn elapsed_time_goes_around_the_cycle() {
        assert_eq!(elapsed_time(10, 360, 366.0), 16);
        assert_eq!(elapsed_time(10, 360, 0.0), 350);
        assert_eq!(elapsed_time(100, 50, 366.0), 50);
    }

    #[test]
    fn batch_drops_missing_dates_and_matches_single_distances() {
        let dates: Vec<i32> = (0..20).map(|i| i * 16).collect();
        let pat_dates: Vec<i32> = (0..10).map(|i| i * 32).collect();
        let pat: Vec<f64> = (0..10).map(|i| (i as f64 / 2.0).sin()).collect();
        let mut values: Vec<f64> = (0..50 * 20)
            .map(|k| ((k % 20) as f64 / 4.0).sin() + (k / 20) as f64 * 0.01)
            .collect();
        values[3 * 20 + 5] = f64::NAN;
        values[4 * 20..5 * 20].fill(f64::NAN);
        let p = Params::default();
        let pattern = series(&pat, &pat_dates);
        for n_jobs in [1, 3] {
            let out = batch(&values, 50, &dates, 1, pattern, &p, f64::INFINITY, n_jobs);
            assert!(out[4].is_nan());
            let w = TimeWeight::new(&p);
            let d0 = distance(
                series(&values[..20], &dates),
                pattern,
                &p,
                &w,
                f64::INFINITY,
                &mut Rows::default(),
            );
            assert_eq!(out[0].to_bits(), d0.to_bits());
            assert!(out[3].is_finite());
        }
    }
}
