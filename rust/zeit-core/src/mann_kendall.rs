//! Mann-Kendall trend test family and Theil-Sen slope: a port of
//! src/mann_kendall.cpp, itself a port of pymannkendall (Hussain & Mahmud,
//! 2019, JOSS, doi:10.21105/joss.01556). Same operations in the same order as
//! the C++, so both give the same bits.

use crate::cmath::erfc;
use crate::parallel;

/// The test of `fit_mann_kendall_batch`'s `method`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Classic Mann-Kendall (Mann 1945, Kendall 1975).
    Original = 0,
    /// Hamed & Rao (1998) autocorrelation-corrected variance.
    HamedRao = 1,
    /// Yue & Wang (2004) autocorrelation-corrected variance.
    YueWang = 2,
    /// Hirsch & Slack (1984): pools per-season Mann-Kendall scores.
    Seasonal = 3,
}

impl Method {
    /// The method of an integer code. Like the C++ (a `static_cast` of the
    /// code), an unknown code runs the original test.
    pub fn from_code(code: i32) -> Method {
        match code {
            1 => Method::HamedRao,
            2 => Method::YueWang,
            3 => Method::Seasonal,
            _ => Method::Original,
        }
    }
}

/// The result of the test on one series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MkResult {
    /// -1 decreasing, 0 no trend, 1 increasing.
    pub trend: i32,
    /// Whether the trend is significant at alpha.
    pub h: bool,
    pub p: f64,
    pub z: f64,
    pub tau: f64,
    pub s: f64,
    pub var_s: f64,
    /// Per time step (index spacing); per full `period` cycle for the seasonal test.
    pub slope: f64,
    pub intercept: f64,
}

impl Default for MkResult {
    fn default() -> Self {
        MkResult {
            trend: 0,
            h: false,
            p: f64::NAN,
            z: f64::NAN,
            tau: f64::NAN,
            s: f64::NAN,
            var_s: f64::NAN,
            slope: f64::NAN,
            intercept: f64::NAN,
        }
    }
}

/// The values of the batch's output, per pixel, in this order.
pub const N_METRICS: usize = 9;

/// Scratch buffers of one thread, sized once for the series length: the
/// helpers write into them instead of allocating per pixel (see MKScratch in
/// the C++ for why that matters for short series).
pub struct Scratch {
    x: Vec<f64>,
    sorted_x: Vec<f64>,
    tie_sizes: Vec<f64>,
    pairwise: Vec<f64>,
    valid_vals: Vec<f64>,
    valid_idx: Vec<f64>,
    x_detrend: Vec<f64>,
    rank_idx: Vec<usize>,
    rank_out: Vec<f64>,
    acf_centered: Vec<f64>,
    acf_out: Vec<f64>,
    seasonal_cols: Vec<Vec<f64>>,
    seasonal_pooled: Vec<f64>,
}

impl Scratch {
    pub fn new(n_time: usize, period: i32) -> Scratch {
        let pairs = if n_time > 1 { n_time * (n_time - 1) / 2 } else { 0 };
        let p = period.max(1) as usize;
        let rows = n_time.div_ceil(p);
        Scratch {
            x: Vec::with_capacity(n_time),
            sorted_x: Vec::with_capacity(n_time),
            tie_sizes: Vec::with_capacity(n_time),
            pairwise: Vec::with_capacity(pairs),
            valid_vals: Vec::with_capacity(n_time),
            valid_idx: Vec::with_capacity(n_time),
            x_detrend: Vec::with_capacity(n_time),
            rank_idx: Vec::with_capacity(n_time),
            rank_out: Vec::with_capacity(n_time),
            acf_centered: Vec::with_capacity(n_time),
            acf_out: Vec::with_capacity(n_time + 1),
            seasonal_cols: (0..p).map(|_| Vec::with_capacity(rows)).collect(),
            seasonal_pooled: Vec::with_capacity(pairs),
        }
    }
}

// ---- small numeric helpers ----------------------------------------------

/// The median, reordering `v`: the value `std::nth_element` gives.
fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (lower, &mut hi, _) = v.select_nth_unstable_by(mid, f64::total_cmp);
    if n % 2 == 1 {
        return hi;
    }
    // the largest of the lower half: nth_element(begin, begin + mid - 1, begin + mid)
    let lo = lower
        .iter()
        .copied()
        .fold(lower[0], |a, b| if b.total_cmp(&a).is_gt() { b } else { a });
    0.5 * (lo + hi)
}

fn drop_nan(x: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend(x.iter().copied().filter(|v| !v.is_nan()));
}

fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// The standard normal CDF, exact through erfc.
fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / 2.0f64.sqrt())
}

/// The inverse standard normal CDF: Acklam's rational approximation, refined
/// with one Newton step against the erfc-based CDF (~1e-15). The constants are
/// written as in the C++ (they round to the same doubles).
#[allow(clippy::excessive_precision)]
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    let mut x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };

    let e = norm_cdf(x) - p;
    let u = e / norm_pdf(x);
    x -= u / (1.0 + x * u / 2.0);
    x
}

// ---- Mann-Kendall building blocks (pymannkendall's private functions) ---

/// __mk_score: the sum of the signs of all i < j pairs.
fn mk_score(x: &[f64]) -> f64 {
    let n = x.len();
    let mut s = 0.0;
    for i in 0..n.saturating_sub(1) {
        for j in i + 1..n {
            if x[j] > x[i] {
                s += 1.0;
            } else if x[j] < x[i] {
                s -= 1.0;
            }
        }
    }
    s
}

/// __variance_s, with the tie correction.
fn variance_s(x: &[f64], sorted_x: &mut Vec<f64>, tie_sizes: &mut Vec<f64>) -> f64 {
    let n = x.len();
    sorted_x.clear();
    sorted_x.extend_from_slice(x);
    sorted_x.sort_unstable_by(f64::total_cmp);

    tie_sizes.clear();
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && sorted_x[j] == sorted_x[i] {
            j += 1;
        }
        tie_sizes.push((j - i) as f64);
        i = j;
    }

    let nd = n as f64;
    if tie_sizes.len() == n {
        return nd * (nd - 1.0) * (2.0 * nd + 5.0) / 18.0;
    }
    let mut tie_sum = 0.0;
    for &t in tie_sizes.iter() {
        tie_sum += t * (t - 1.0) * (2.0 * t + 5.0);
    }
    (nd * (nd - 1.0) * (2.0 * nd + 5.0) - tie_sum) / 18.0
}

/// No guard on var_s, as pymannkendall: a negative one (possible after the
/// Hamed-Rao correction, which is not clamped) gives NaN, as np.sqrt does.
fn z_score(s: f64, var_s: f64) -> f64 {
    if s > 0.0 {
        (s - 1.0) / var_s.sqrt()
    } else if s < 0.0 {
        (s + 1.0) / var_s.sqrt()
    } else {
        0.0
    }
}

fn p_value(z: f64, alpha: f64) -> (f64, bool, i32) {
    let p = 2.0 * (1.0 - norm_cdf(z.abs()));
    let h = z.abs() > norm_ppf(1.0 - alpha / 2.0);
    let trend = if z < 0.0 && h {
        -1
    } else if z > 0.0 && h {
        1
    } else {
        0
    };
    (p, h, trend)
}

/// __acf up to `nlags` (inclusive): np.correlate(y, y, 'full')[n-1:] / n,
/// normalized by the lag-0 autocovariance.
fn acf(x: &[f64], nlags: i64, centered: &mut Vec<f64>, result: &mut Vec<f64>) {
    let n = x.len();
    let nlags = nlags.min(n as i64 - 1).max(0) as usize;

    let mut sum = 0.0;
    for &v in x {
        sum += v;
    }
    let mean = sum / n as f64;
    centered.clear();
    centered.extend(x.iter().map(|&v| v - mean));

    let mut acov0 = 0.0;
    for &c in centered.iter() {
        acov0 += c * c;
    }
    acov0 /= n as f64;

    result.clear();
    result.resize(nlags + 1, 0.0);
    for lag in 0..=nlags {
        let mut s = 0.0;
        for i in 0..n - lag {
            s += centered[i] * centered[i + lag];
        }
        s /= n as f64;
        result[lag] = if acov0 != 0.0 { s / acov0 } else { s };
    }
}

/// scipy.stats.rankdata(method='average').
fn rankdata(x: &[f64], idx: &mut Vec<usize>, ranks: &mut Vec<f64>) {
    let n = x.len();
    idx.clear();
    idx.extend(0..n);
    idx.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));

    ranks.clear();
    ranks.resize(n, 0.0);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-indexed average rank
        for &k in &idx[i..j] {
            ranks[k] = avg_rank;
        }
        i = j;
    }
}

/// sens_slope on the original (NaN-gapped) series: the positional spacing
/// (j - i) as the time step, pairs touching a NaN skipped (pymannkendall's
/// sens_slope on x_old), so gaps do not collapse the time axis.
fn sens_slope_full(
    x_old: &[f64],
    pairwise: &mut Vec<f64>,
    valid_vals: &mut Vec<f64>,
    valid_idx: &mut Vec<f64>,
) -> (f64, f64) {
    let n = x_old.len();
    pairwise.clear();
    for i in 0..n.saturating_sub(1) {
        if x_old[i].is_nan() {
            continue;
        }
        for j in i + 1..n {
            if x_old[j].is_nan() {
                continue;
            }
            pairwise.push((x_old[j] - x_old[i]) / (j - i) as f64);
        }
    }
    let slope = median(pairwise);

    valid_vals.clear();
    valid_idx.clear();
    for (i, &v) in x_old.iter().enumerate() {
        if !v.is_nan() {
            valid_vals.push(v);
            valid_idx.push(i as f64);
        }
    }
    let intercept = median(valid_vals) - median(valid_idx) * slope;
    (slope, intercept)
}

/// The original, Hamed-Rao and Yue-Wang tests (not the seasonal one).
fn mk_test_impl(y_old: &[f64], method: Method, alpha: f64, lag: i32, sc: &mut Scratch) -> MkResult {
    drop_nan(y_old, &mut sc.x);
    let n = sc.x.len();
    if n < 4 {
        return MkResult::default();
    }
    let x = &sc.x;

    let s = mk_score(x);
    let mut var_s = variance_s(x, &mut sc.sorted_x, &mut sc.tie_sizes);
    let tau = s / (0.5 * n as f64 * (n - 1) as f64);

    let (slope, intercept) = sens_slope_full(y_old, &mut sc.pairwise, &mut sc.valid_vals, &mut sc.valid_idx);

    if method == Method::HamedRao || method == Method::YueWang {
        let l = if lag < 0 { n as i64 } else { lag as i64 + 1 };
        let l = l.min(n as i64).max(1);

        sc.x_detrend.clear();
        sc.x_detrend
            .extend(x.iter().enumerate().map(|(i, &v)| v - (i + 1) as f64 * slope));

        let interval = norm_ppf(1.0 - alpha / 2.0) / (n as f64).sqrt();
        let mut sni = 0.0;
        let nf = n as f64;

        if method == Method::HamedRao {
            rankdata(&sc.x_detrend, &mut sc.rank_idx, &mut sc.rank_out);
            acf(&sc.rank_out, l - 1, &mut sc.acf_centered, &mut sc.acf_out);
            for i in 1..l as usize {
                let r = sc.acf_out[i];
                if !(r <= interval && r >= -interval) {
                    sni += (n - i) as f64 * (n - i - 1) as f64 * (n - i - 2) as f64 * r;
                }
            }
            // No abs() on sni: pymannkendall >= 1.4.3's formula.
            let n_ns = 1.0 + (2.0 / (nf * (n - 1) as f64 * (n - 2) as f64)) * sni;
            var_s *= n_ns;
        } else {
            acf(&sc.x_detrend, l - 1, &mut sc.acf_centered, &mut sc.acf_out);
            for i in 1..l as usize {
                sni += (1.0 - i as f64 / nf) * sc.acf_out[i];
            }
            let n_ns = 1.0 + 2.0 * sni;
            var_s *= n_ns;
        }
    }

    let z = z_score(s, var_s);
    let (p, h, trend) = p_value(z, alpha);
    MkResult {
        trend,
        h,
        p,
        z,
        tau,
        s,
        var_s,
        slope,
        intercept,
    }
}

/// The series in `period` season columns, the tail padded with NaN
/// (pymannkendall's seasonal_test reshape).
fn reshape_seasonal(x_old: &[f64], period: usize, cols: &mut [Vec<f64>]) {
    let rows = x_old.len().div_ceil(period);
    for col in cols.iter_mut() {
        col.clear();
        col.resize(rows, f64::NAN);
    }
    for (i, &v) in x_old.iter().enumerate() {
        cols[i % period][i / period] = v;
    }
}

/// seasonal_test: multivariate_test over the season columns, with
/// seasonal_sens_slope for the slope and intercept (pymannkendall).
fn seasonal_mk_test(y_old: &[f64], period: i32, alpha: f64, sc: &mut Scratch) -> MkResult {
    let period = if period < 2 { 1 } else { period as usize };

    reshape_seasonal(y_old, period, &mut sc.seasonal_cols);

    let (mut s, mut var_s, mut denom) = (0.0, 0.0, 0.0);
    for c in 0..period {
        drop_nan(&sc.seasonal_cols[c], &mut sc.x);
        let nc = sc.x.len();
        if nc < 2 {
            continue;
        }
        s += mk_score(&sc.x);
        var_s += variance_s(&sc.x, &mut sc.sorted_x, &mut sc.tie_sizes);
        denom += 0.5 * nc as f64 * (nc - 1) as f64;
    }
    if denom <= 0.0 {
        return MkResult::default();
    }

    let tau = s / denom;
    let z = z_score(s, var_s);
    let (p, h, trend) = p_value(z, alpha);

    // Seasonal Sen's slope: the pairwise slopes of every column pooled, over
    // the row gap (in units of one full period).
    sc.seasonal_pooled.clear();
    for col in &sc.seasonal_cols[..period] {
        let rows = col.len();
        for i in 0..rows.saturating_sub(1) {
            if col[i].is_nan() {
                continue;
            }
            for j in i + 1..rows {
                if col[j].is_nan() {
                    continue;
                }
                sc.seasonal_pooled.push((col[j] - col[i]) / (j - i) as f64);
            }
        }
    }
    let slope = median(&mut sc.seasonal_pooled);

    sc.valid_vals.clear();
    sc.valid_idx.clear();
    for (i, &v) in y_old.iter().enumerate() {
        if !v.is_nan() {
            sc.valid_vals.push(v);
            sc.valid_idx.push(i as f64);
        }
    }
    let intercept = median(&mut sc.valid_vals) - (median(&mut sc.valid_idx) / period as f64) * slope;

    MkResult {
        trend,
        h,
        p,
        z,
        tau,
        s,
        var_s,
        slope,
        intercept,
    }
}

/// The test on one series `y` (NaN skipped, pymannkendall's
/// missing_values_analysis(method='skip')). `lag` < 0 is pymannkendall's None
/// (the whole series); `period` is only used by the seasonal test.
pub fn mk_test(y: &[f64], method: Method, alpha: f64, lag: i32, period: i32) -> MkResult {
    let mut sc = Scratch::new(y.len(), period);
    mk_test_with(y, method, alpha, lag, period, &mut sc)
}

fn mk_test_with(y: &[f64], method: Method, alpha: f64, lag: i32, period: i32, sc: &mut Scratch) -> MkResult {
    if method == Method::Seasonal {
        seasonal_mk_test(y, period, alpha, sc)
    } else {
        mk_test_impl(y, method, alpha, lag, sc)
    }
}

/// The test on every row of `values` (`n_pixels` x `n_time`, row-major):
/// `N_METRICS` x `n_pixels` values (trend, h, p, z, tau, s, var_s, slope,
/// intercept), NaN for pixels with fewer than `min_valid` observations.
#[allow(clippy::too_many_arguments)]
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    n_time: usize,
    method: Method,
    alpha: f64,
    lag: i32,
    period: i32,
    min_valid: i32,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(values.len(), n_pixels * n_time, "values must be n_pixels x n_time");
    let mut per_pixel = vec![f64::NAN; n_pixels * N_METRICS];
    parallel::for_each_row(
        &mut per_pixel,
        N_METRICS,
        n_jobs,
        || Scratch::new(n_time, period),
        |sc, p, out| {
            let y = &values[p * n_time..(p + 1) * n_time];
            let valid = y.iter().filter(|v| !v.is_nan()).count() as i64;
            if valid < min_valid as i64 {
                return;
            }
            let r = mk_test_with(y, method, alpha, lag, period, sc);
            out.copy_from_slice(&[
                r.trend as f64,
                if r.h { 1.0 } else { 0.0 },
                r.p,
                r.z,
                r.tau,
                r.s,
                r.var_s,
                r.slope,
                r.intercept,
            ]);
        },
    );
    parallel::transpose(&per_pixel, n_pixels, N_METRICS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_and_even() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
        assert!(median(&mut []).is_nan());
    }

    #[test]
    fn increasing_series_has_a_significant_trend() {
        let y: Vec<f64> = (0..20)
            .map(|i| i as f64 + if i % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let r = mk_test(&y, Method::Original, 0.05, -1, 1);
        assert_eq!(r.trend, 1);
        assert!(r.h);
        assert!(r.p < 1e-6);
        assert!((r.slope - 1.0).abs() < 0.1);
    }

    #[test]
    fn short_series_give_nan() {
        let r = mk_test(&[1.0, f64::NAN, 2.0, 3.0], Method::HamedRao, 0.05, -1, 1);
        assert!(r.p.is_nan() && r.trend == 0);
    }

    #[test]
    fn batch_matches_single_series_on_any_threads() {
        let n_time = 30;
        let values: Vec<f64> = (0..200 * n_time)
            .map(|k| ((k * 7919) % 101) as f64 / 10.0 + (k % n_time) as f64 * 0.05)
            .collect();
        let single: Vec<MkResult> = values
            .chunks(n_time)
            .map(|y| mk_test(y, Method::HamedRao, 0.05, -1, 1))
            .collect();
        for n_jobs in [1, 4] {
            let out = batch(&values, 200, n_time, Method::HamedRao, 0.05, -1, 1, 4, n_jobs);
            for (p, r) in single.iter().enumerate() {
                assert_eq!(out[2 * 200 + p].to_bits(), r.p.to_bits());
                assert_eq!(out[7 * 200 + p].to_bits(), r.slope.to_bits());
            }
        }
    }
}
