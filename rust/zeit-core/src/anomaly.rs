//! Anomalies of a time series: what remains after the seasonal cycle (and
//! optionally a linear trend) is removed, pixel by pixel over its valid
//! values. Used by the temporal EOF and the lagged correlation.
//!
//! - `Mean`: the pixel's mean (of the base years, if given).
//! - `Monthly`: the climatology of each calendar month, the mean of the pixel's
//!   values of that month over the base years; a month with fewer than
//!   `min_per_month` values gives NaN to its dates.
//! - `Harmonic`: an ordinary least squares fit of a constant and `order`
//!   sine/cosine pairs of the decimal year (plus a trend if `detrend`).
//!
//! `detrend` removes a straight line in the decimal year (for `Mean` and
//! `Monthly` fitted together with the seasonal part), `standardize`
//! divides by the standard deviation of the anomalies. A NaN is a missing value.

use crate::parallel;
use rayon::prelude::*;

/// How the seasonal part of a series is taken out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Seasonal {
    Mean,
    Monthly { min_per_month: usize },
    Harmonic { order: usize },
}

/// What `anomalies` does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnomalySettings {
    pub seasonal: Seasonal,
    /// The years (inclusive) that define the climatology (`Mean`, `Monthly`) or
    /// the harmonic fit; None for all of them. The trend is fitted on all dates.
    pub base: Option<(i32, i32)>,
    pub detrend: bool,
    pub standardize: bool,
}

impl Default for AnomalySettings {
    fn default() -> Self {
        AnomalySettings {
            seasonal: Seasonal::Monthly { min_per_month: 2 },
            base: None,
            detrend: false,
            standardize: false,
        }
    }
}

/// The civil date (year, month 1..=12, day 1..=31) of a day count since
/// 1970-01-01 (Hinnant's `civil_from_days`).
pub fn civil_from_days(d: i64) -> (i32, u32, u32) {
    let z = d + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((y + i64::from(month <= 2)) as i32, month, day)
}

/// The day count since 1970-01-01 of a civil date (Hinnant's `days_from_civil`).
pub fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = i64::from(year) - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = i64::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The days of a month.
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}

/// The calendar of each date: year, month and decimal year.
struct Dates {
    year: Vec<i32>,
    month: Vec<usize>,
    /// The year plus the fraction of the year gone at the start of the day.
    tau: Vec<f64>,
}

impl Dates {
    fn new(days: &[i64]) -> Dates {
        let mut year = Vec::with_capacity(days.len());
        let mut month = Vec::with_capacity(days.len());
        let mut tau = Vec::with_capacity(days.len());
        for &d in days {
            let (y, m, _) = civil_from_days(d);
            let start = days_from_civil(y, 1, 1);
            let len = days_from_civil(y + 1, 1, 1) - start;
            year.push(y);
            month.push(m as usize - 1);
            tau.push(f64::from(y) + (d - start) as f64 / len as f64);
        }
        Dates { year, month, tau }
    }
}

/// Solves the symmetric system `a x = b` (`p` x `p`, row-major) by Gaussian
/// elimination with partial pivoting; false if it is singular.
fn solve(a: &mut [f64], b: &mut [f64], p: usize) -> bool {
    for c in 0..p {
        let mut piv = c;
        for r in c + 1..p {
            if a[r * p + c].abs() > a[piv * p + c].abs() {
                piv = r;
            }
        }
        let scale = (0..p).map(|r| a[r * p + c].abs()).fold(0.0, f64::max);
        if !(a[piv * p + c].abs() > 1e-12 * scale.max(f64::MIN_POSITIVE)) {
            return false;
        }
        if piv != c {
            for k in 0..p {
                a.swap(c * p + k, piv * p + k);
            }
            b.swap(c, piv);
        }
        for r in c + 1..p {
            let f = a[r * p + c] / a[c * p + c];
            if f != 0.0 {
                for k in c..p {
                    a[r * p + k] -= f * a[c * p + k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    for c in (0..p).rev() {
        let mut s = b[c];
        for k in c + 1..p {
            s -= a[c * p + k] * b[k];
        }
        b[c] = s / a[c * p + c];
    }
    true
}

fn in_base(s: &AnomalySettings, year: i32) -> bool {
    s.base.is_none_or(|(a, b)| year >= a && year <= b)
}

/// The harmonic design row of a date: 1, cos/sin of 2 pi k tau, and the
/// centred trend.
fn design_row(row: &mut [f64], order: usize, tau: f64, year: i32, tau_ref: f64, trend: bool) {
    row[0] = 1.0;
    let frac = tau - f64::from(year);
    for k in 1..=order {
        let ang = 2.0 * std::f64::consts::PI * k as f64 * frac;
        row[2 * k - 1] = ang.cos();
        row[2 * k] = ang.sin();
    }
    if trend {
        row[2 * order + 1] = tau - tau_ref;
    }
}

/// The fitted harmonic curve (with its trend if asked) at every date, NaN if
/// the fit is not possible.
fn harmonic_baseline(values: &[f64], dt: &Dates, s: &AnomalySettings, order: usize) -> Vec<f64> {
    let t = values.len();
    let p = 2 * order + 1 + usize::from(s.detrend);
    let fit: Vec<usize> = (0..t)
        .filter(|&i| values[i].is_finite() && in_base(s, dt.year[i]))
        .collect();
    if fit.len() < 2 * order + 2 || fit.is_empty() {
        return vec![f64::NAN; t];
    }
    let tau_ref = fit.iter().map(|&i| dt.tau[i]).sum::<f64>() / fit.len() as f64;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];
    let mut row = vec![0.0; p];
    for &i in &fit {
        design_row(&mut row, order, dt.tau[i], dt.year[i], tau_ref, s.detrend);
        for a in 0..p {
            atb[a] += row[a] * values[i];
            for b in a..p {
                ata[a * p + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            ata[a * p + b] = ata[b * p + a];
        }
    }
    if !solve(&mut ata, &mut atb, p) {
        return vec![f64::NAN; t];
    }
    (0..t)
        .map(|i| {
            design_row(&mut row, order, dt.tau[i], dt.year[i], tau_ref, s.detrend);
            row.iter().zip(&atb).map(|(r, c)| r * c).sum()
        })
        .collect()
}

/// The seasonal part at every date (`Mean`, `Monthly`).
fn seasonal_baseline(values: &[f64], dt: &Dates, s: &AnomalySettings) -> Vec<f64> {
    let t = values.len();
    match s.seasonal {
        Seasonal::Mean => {
            let (mut sum, mut n) = (0.0, 0usize);
            for (&v, &y) in values.iter().zip(&dt.year) {
                if v.is_finite() && in_base(s, y) {
                    sum += v;
                    n += 1;
                }
            }
            vec![if n > 0 { sum / n as f64 } else { f64::NAN }; t]
        }
        Seasonal::Monthly { min_per_month } => {
            let mut sum = [0.0; 12];
            let mut cnt = [0usize; 12];
            for i in 0..t {
                if values[i].is_finite() && in_base(s, dt.year[i]) {
                    sum[dt.month[i]] += values[i];
                    cnt[dt.month[i]] += 1;
                }
            }
            let clim: Vec<f64> = (0..12)
                .map(|m| {
                    if cnt[m] > 0 && cnt[m] >= min_per_month {
                        sum[m] / cnt[m] as f64
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            (0..t).map(|i| clim[dt.month[i]]).collect()
        }
        Seasonal::Harmonic { .. } => unreachable!(),
    }
}

/// What is removed from every date: the seasonal part and the trend.
fn baseline_with(values: &[f64], dt: &Dates, s: &AnomalySettings) -> Vec<f64> {
    if let Seasonal::Harmonic { order } = s.seasonal {
        return harmonic_baseline(values, dt, s, order);
    }
    let t = values.len();
    let mut base = seasonal_baseline(values, dt, s);
    if !s.detrend {
        return base;
    }
    // The seasonal part and the straight line are fitted together by backfitting: the line through what the
    // seasonal part leaves, then the seasonal part of what the line leaves, until the slope settles (the
    // seasonal part alone would absorb part of the trend).
    let used: Vec<usize> = (0..t)
        .filter(|&i| values[i].is_finite() && base[i].is_finite())
        .collect();
    let n = used.len();
    if n < 3 {
        return vec![f64::NAN; t];
    }
    let tm = used.iter().map(|&i| dt.tau[i]).sum::<f64>() / n as f64;
    let sxx: f64 = used.iter().map(|&i| (dt.tau[i] - tm) * (dt.tau[i] - tm)).sum();
    if !(sxx > 0.0) {
        return vec![f64::NAN; t];
    }
    let mut slope = 0.0;
    let mut shifted = vec![0.0; t];
    for _ in 0..50 {
        let rm = used.iter().map(|&i| values[i] - base[i]).sum::<f64>() / n as f64;
        let sxy: f64 = used
            .iter()
            .map(|&i| (dt.tau[i] - tm) * (values[i] - base[i] - rm))
            .sum();
        let new_slope = sxy / sxx;
        let done = (new_slope - slope).abs() <= 1e-14 * new_slope.abs();
        slope = new_slope;
        for i in 0..t {
            shifted[i] = values[i] - slope * (dt.tau[i] - tm);
        }
        base = seasonal_baseline(&shifted, dt, s);
        if done {
            break;
        }
    }
    for (b, &tau) in base.iter_mut().zip(&dt.tau) {
        *b += slope * (tau - tm);
    }
    base
}

fn anomalies_with(values: &mut [f64], dt: &Dates, s: &AnomalySettings) {
    let base = baseline_with(values, dt, s);
    for (v, b) in values.iter_mut().zip(&base) {
        // a missing value or a missing baseline (NaN) gives NaN
        *v -= *b;
    }
    if s.standardize {
        let valid: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let n = valid.len();
        let sd = if n >= 2 {
            let m = valid.iter().sum::<f64>() / n as f64;
            (valid.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1) as f64).sqrt()
        } else {
            f64::NAN
        };
        for v in values.iter_mut() {
            *v = if sd > 0.0 { *v / sd } else { f64::NAN };
        }
    }
}

/// Replaces a series (dates in `days`, NaN missing) by its anomalies.
pub fn anomalies(values: &mut [f64], days: &[i64], s: &AnomalySettings) {
    assert_eq!(values.len(), days.len(), "one day for each value");
    anomalies_with(values, &Dates::new(days), s);
}

/// What `anomalies` removes (the seasonal part and the trend, not the
/// standardization), at every date, for a chart overlay; NaN where there is no
/// fit.
pub fn baseline(values: &[f64], days: &[i64], s: &AnomalySettings) -> Vec<f64> {
    assert_eq!(values.len(), days.len(), "one day for each value");
    baseline_with(values, &Dates::new(days), s)
}

/// Pixels of a block of `anomalies_cube`.
const BLOCK: usize = 4096;

/// `anomalies` of every pixel of a cube of `t` dates x `n` pixels (time-major).
pub fn anomalies_cube(values: &mut [f64], t: usize, n: usize, days: &[i64], s: &AnomalySettings, n_jobs: i32) {
    assert_eq!(values.len(), t * n, "values must be dates x pixels");
    assert_eq!(days.len(), t, "one day for each date");
    if t == 0 || n == 0 {
        return;
    }
    let dt = Dates::new(days);
    parallel::pool(parallel::threads(n_jobs)).install(|| {
        for start in (0..n).step_by(BLOCK) {
            let end = (start + BLOCK).min(n);
            let cols: Vec<Vec<f64>> = {
                let src: &[f64] = values;
                (start..end)
                    .into_par_iter()
                    .map(|j| {
                        let mut col: Vec<f64> = (0..t).map(|i| src[i * n + j]).collect();
                        anomalies_with(&mut col, &dt, s);
                        col
                    })
                    .collect()
            };
            for (k, col) in cols.iter().enumerate() {
                for (i, v) in col.iter().enumerate() {
                    values[i * n + start + k] = *v;
                }
            }
        }
    });
}

/// The mean of the valid values of every calendar month of a cube of `t` dates
/// x `n` pixels (time-major): the months present in `days`, ascending, as
/// `year * 12 + month - 1`, and their means (months x pixels, NaN if a pixel
/// has no valid value in the month).
pub fn monthly_means(values: &[f64], t: usize, n: usize, days: &[i64]) -> (Vec<i64>, Vec<f64>) {
    assert_eq!(values.len(), t * n, "values must be dates x pixels");
    assert_eq!(days.len(), t, "one day for each date");
    let key: Vec<i64> = days
        .iter()
        .map(|&d| {
            let (y, m, _) = civil_from_days(d);
            i64::from(y) * 12 + i64::from(m) - 1
        })
        .collect();
    let mut months = key.clone();
    months.sort_unstable();
    months.dedup();
    let mut out = vec![f64::NAN; months.len() * n];
    let mut count = vec![0u32; n];
    for (mi, &mk) in months.iter().enumerate() {
        count.iter_mut().for_each(|c| *c = 0);
        let row = &mut out[mi * n..(mi + 1) * n];
        for i in (0..t).filter(|&i| key[i] == mk) {
            for j in 0..n {
                let v = values[i * n + j];
                if v.is_finite() {
                    row[j] = if count[j] == 0 { v } else { row[j] + v };
                    count[j] += 1;
                }
            }
        }
        for j in 0..n {
            if count[j] > 1 {
                row[j] /= f64::from(count[j]);
            }
        }
    }
    (months, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle(m: usize) -> f64 {
        10.0 + 3.0 * ((m % 12) as f64).sin() + (m % 12) as f64 * 0.1
    }

    /// Monthly dates (the 15th) of the years 2000.. and their month numbers.
    fn monthly_days(years: i32) -> Vec<i64> {
        let mut d = Vec::new();
        for y in 2000..2000 + years {
            for m in 1..=12 {
                d.push(days_from_civil(y, m, 15));
            }
        }
        d
    }

    #[test]
    fn civil_dates_round_trip() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(59), (1970, 3, 1));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
        assert_eq!(days_from_civil(2000, 3, 1) - days_from_civil(2000, 2, 28), 2);
        for d in (-40_000..40_000).step_by(37) {
            let (y, m, dd) = civil_from_days(d);
            assert_eq!(days_from_civil(y, m, dd), d);
        }
    }

    #[test]
    fn pure_cycle_is_zero_under_monthly() {
        let days = monthly_days(6);
        let mut v: Vec<f64> = (0..days.len()).map(cycle).collect();
        anomalies(&mut v, &days, &AnomalySettings::default());
        assert!(v.iter().all(|a| a.abs() < 1e-12));
    }

    #[test]
    fn pure_cycle_is_zero_under_harmonic() {
        let days: Vec<i64> = (0..6 * 365).step_by(8).map(|d| d as i64 + 10_957).collect();
        let dt = Dates::new(&days);
        let mut v: Vec<f64> = dt
            .tau
            .iter()
            .zip(&dt.year)
            .map(|(&t, &y)| {
                let f = 2.0 * std::f64::consts::PI * (t - f64::from(y));
                5.0 + 2.0 * f.cos() - 1.5 * f.sin() + 0.7 * (2.0 * f).cos() + 0.3 * (2.0 * f).sin()
            })
            .collect();
        let s = AnomalySettings {
            seasonal: Seasonal::Harmonic { order: 2 },
            ..Default::default()
        };
        anomalies(&mut v, &days, &s);
        assert!(
            v.iter().all(|a| a.abs() < 1e-8),
            "{:?}",
            v.iter().cloned().fold(0.0, f64::max)
        );
        // a first-order fit leaves the second harmonic
        let mut w: Vec<f64> = dt
            .tau
            .iter()
            .zip(&dt.year)
            .map(|(&t, &y)| (4.0 * std::f64::consts::PI * (t - f64::from(y))).cos())
            .collect();
        let s1 = AnomalySettings {
            seasonal: Seasonal::Harmonic { order: 1 },
            ..Default::default()
        };
        anomalies(&mut w, &days, &s1);
        assert!(w.iter().any(|a| a.abs() > 0.5));
    }

    #[test]
    fn harmonic_needs_enough_values() {
        let days = monthly_days(1);
        let mut v: Vec<f64> = (0..12).map(cycle).collect();
        for x in v.iter_mut().skip(5) {
            *x = f64::NAN;
        }
        // 5 valid values < 2 * 2 + 2
        let s = AnomalySettings {
            seasonal: Seasonal::Harmonic { order: 2 },
            ..Default::default()
        };
        anomalies(&mut v, &days, &s);
        assert!(v.iter().all(|a| a.is_nan()));
    }

    #[test]
    fn gaps_stay_gaps_and_short_months_are_nan() {
        let days = monthly_days(4);
        let mut v: Vec<f64> = (0..days.len()).map(cycle).collect();
        v[3] = f64::NAN; // April of the first year
        let keep = v.clone();
        // every January but one is missing: fewer than 2 values
        for y in 1..4 {
            v[y * 12] = f64::NAN;
        }
        anomalies(&mut v, &days, &AnomalySettings::default());
        assert!(v[3].is_nan());
        // January: one value left, so NaN
        assert!(v[0].is_nan());
        // the other months are still zero
        for i in 0..v.len() {
            if i % 12 != 0 && !keep[i].is_nan() {
                assert!(v[i].abs() < 1e-12, "{i}");
            }
        }
        // min_per_month = 1 keeps January
        let mut w: Vec<f64> = (0..days.len()).map(cycle).collect();
        for y in 1..4 {
            w[y * 12] = f64::NAN;
        }
        let s = AnomalySettings {
            seasonal: Seasonal::Monthly { min_per_month: 1 },
            ..Default::default()
        };
        anomalies(&mut w, &days, &s);
        assert!(w[0].abs() < 1e-12);
    }

    #[test]
    fn base_years_define_the_climatology() {
        let days = monthly_days(6);
        // the cycle plus a step of 4 from 2003 on
        let mut v: Vec<f64> = (0..days.len())
            .map(|i| cycle(i) + if i >= 36 { 4.0 } else { 0.0 })
            .collect();
        let s = AnomalySettings {
            base: Some((2000, 2002)),
            ..Default::default()
        };
        anomalies(&mut v, &days, &s);
        assert!(v[..36].iter().all(|a| a.abs() < 1e-12));
        assert!(v[36..].iter().all(|a| (a - 4.0).abs() < 1e-12));
        let mut w: Vec<f64> = (0..days.len())
            .map(|i| cycle(i) + if i >= 36 { 4.0 } else { 0.0 })
            .collect();
        anomalies(&mut w, &days, &AnomalySettings::default());
        assert!((w[0] + 2.0).abs() < 1e-12 && (w[40] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn detrend_removes_a_line() {
        let days = monthly_days(8);
        let dt = Dates::new(&days);
        let line = |i: usize| 0.5 * (dt.tau[i] - 2000.0) + 3.0;
        for seasonal in [
            Seasonal::Mean,
            Seasonal::Monthly { min_per_month: 2 },
            Seasonal::Harmonic { order: 2 },
        ] {
            let mut v: Vec<f64> = (0..days.len()).map(line).collect();
            let s = AnomalySettings {
                seasonal,
                detrend: true,
                ..Default::default()
            };
            anomalies(&mut v, &days, &s);
            assert!(v.iter().all(|a| a.abs() < 1e-9), "{seasonal:?}");
        }
        // without detrend the line stays
        let mut v: Vec<f64> = (0..days.len()).map(line).collect();
        let s = AnomalySettings {
            seasonal: Seasonal::Mean,
            ..Default::default()
        };
        anomalies(&mut v, &days, &s);
        assert!(v.iter().any(|a| a.abs() > 1.0));
        assert!(v.iter().sum::<f64>().abs() < 1e-9);
    }

    #[test]
    fn standardize_gives_unit_sd_and_baseline_is_the_removed_part() {
        let days = monthly_days(5);
        let x: Vec<f64> = (0..days.len())
            .map(|i| cycle(i) + ((i * 7919) % 13) as f64 * 0.3)
            .collect();
        let s = AnomalySettings {
            standardize: true,
            ..Default::default()
        };
        let mut v = x.clone();
        anomalies(&mut v, &days, &s);
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0)).sqrt();
        assert!((sd - 1.0).abs() < 1e-12);
        let plain = AnomalySettings::default();
        let b = baseline(&x, &days, &plain);
        let mut a = x.clone();
        anomalies(&mut a, &days, &plain);
        for i in 0..x.len() {
            assert!((x[i] - b[i] - a[i]).abs() < 1e-12);
        }
        // a constant pixel cannot be standardized
        let mut c = vec![2.0; days.len()];
        anomalies(&mut c, &days, &s);
        assert!(c.iter().all(|a| a.is_nan()));
    }

    #[test]
    fn cube_matches_series_for_any_jobs() {
        let days = monthly_days(5);
        let (t, n) = (days.len(), 7);
        let mut cube = vec![0.0; t * n];
        for i in 0..t {
            for j in 0..n {
                cube[i * n + j] = if (i + j) % 11 == 0 {
                    f64::NAN
                } else {
                    cycle(i) * (j + 1) as f64 + (i * j) as f64 * 0.01
                };
            }
        }
        let s = AnomalySettings {
            detrend: true,
            standardize: true,
            ..Default::default()
        };
        let mut a = cube.clone();
        anomalies_cube(&mut a, t, n, &days, &s, 1);
        let mut b = cube.clone();
        anomalies_cube(&mut b, t, n, &days, &s, 3);
        for j in 0..n {
            let mut col: Vec<f64> = (0..t).map(|i| cube[i * n + j]).collect();
            anomalies(&mut col, &days, &s);
            for i in 0..t {
                assert_eq!(col[i].to_bits(), a[i * n + j].to_bits());
                assert_eq!(col[i].to_bits(), b[i * n + j].to_bits());
            }
        }
    }

    #[test]
    fn monthly_means_aggregate_a_cube() {
        // 16-day dates over Dec 2020 .. Feb 2021, two pixels
        let days: Vec<i64> = vec![
            days_from_civil(2020, 12, 5),
            days_from_civil(2020, 12, 21),
            days_from_civil(2021, 1, 6),
            days_from_civil(2021, 2, 7),
            days_from_civil(2021, 2, 23),
        ];
        let v = vec![1.0, 10.0, 3.0, f64::NAN, 5.0, f64::NAN, 7.0, 8.0, 9.0, f64::NAN];
        let (months, m) = monthly_means(&v, 5, 2, &days);
        assert_eq!(months, vec![2020 * 12 + 11, 2021 * 12, 2021 * 12 + 1]);
        assert_eq!(m[0], 2.0);
        assert_eq!(m[1], 10.0);
        assert_eq!(m[2], 5.0);
        assert!(m[3].is_nan());
        assert_eq!(m[4], 8.0);
        assert_eq!(m[5], 8.0);
    }
}
