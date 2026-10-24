//! EWMACD: Exponentially Weighted Moving Average Change Detection (Brooks et al. 2014), the
//! continuous monitoring of an irregular series (every Landsat or Sentinel-2 observation, no
//! compositing) against a harmonic model of its training years, with quality control charts.
//!
//! Per pixel, on the dates from `train_start` on (values that are not finite or not above `low`
//! are missing):
//!
//! 1. a harmonic regression (a constant and `harmonics` sine and cosine pairs of the day of the
//!    year, `2π doy / 365`) on the training dates (years before `train_end`), refitted without
//!    the dates whose residual is beyond `xbar_train` standard deviations (a Shewhart X-bar chart:
//!    clouds and shadows of the training years);
//! 2. the residuals of every date from that model, and a second X-bar screen with the standard
//!    deviation `s0` of the training residuals: the training dates beyond `xbar_train s0` and the
//!    later dates beyond `xbar_monitor s0` (20 by default: only gross anomalies, not the changes)
//!    are left out of the chart; `sigma`, the standard deviation of the training residuals kept,
//!    sets its control limits;
//! 3. the EWMA chart of the residuals kept, `z_1 = r_1`, `z_j = (1 - λ) z_(j-1) + λ r_j`, and its
//!    control limits `CL_j = sigmas · sigma · sqrt(λ / (2 - λ) · (1 - (1 - λ)^(2j)))`;
//! 4. the flags: `z_j / CL_j` truncated toward zero (with `rounding`; without, the ratio itself
//!    where it is beyond ±1, else 0), negative for a loss of the index (a disturbance) and
//!    positive for a gain, ±1 the weakest signal;
//! 5. persistence: a run of flags of one sign (zeros included) shorter than `persistence` dates
//!    takes the value of the last date of a run that lasted (0 before the first): a signal must
//!    last to count, and a brief return inside the limits does not end one;
//! 6. the dates left out (missing, screened) carry the flag of the date before them; the first
//!    date's is 0 when it was left out.
//!
//! This is Brooks' EWMACD v1.3.0 script (the paper's; also Earth Engine's
//! `ee.Algorithms.TemporalSegmentation.Ewmacd`), whose flags it gives on series without missing
//! training dates when `low` is below every value and `persistence` is 1. It differs from the
//! script where the script slips: values not above `low` are missing from the start (the script
//! fits them in its first regression and its first standard deviation, as its v1.9 no longer
//! does); the training refit uses the values of the dates it keeps when training dates are
//! missing (the script's indices shift); and in step 5 the script means "the previous sustained
//! state" but reads it at the wrong date.
//!
//! From the flags of the monitoring dates (years from `train_end` on), `Events`: the first
//! disturbance (the first negative flag), its severity (its most negative flag until it
//! recovers), the recovery (the first flag back at 0 or above after it), the first gain, and the
//! state at the last date.

use crate::parallel;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// first year of the series looked at (dates before it are left out: NaN flags)
    pub train_start: i32,
    /// first year after the training period (years before it train the model)
    pub train_end: i32,
    /// sine and cosine pairs of the harmonic model
    pub harmonics: usize,
    /// X-bar limit of the training dates, in standard deviations
    pub xbar_train: f64,
    /// X-bar limit of the dates after them (gross anomalies only)
    pub xbar_monitor: f64,
    /// weight of each new residual in the EWMA, in (0, 1]
    pub lambda: f64,
    /// control limits in standard deviations of the EWMA
    pub sigmas: f64,
    /// flags truncated to integers (multiples of the control limit)
    pub rounding: bool,
    /// dates a signal must last
    pub persistence: usize,
    /// values not above it are missing (no vegetation, fill values)
    pub low: f64,
}

impl Params {
    /// Brooks et al. (2014)'s defaults: 2 harmonics, X-bar limits 1.5 and 20, λ = 0.3, 3 sigmas,
    /// rounding, persistence 3 (as Earth Engine's), no lower threshold.
    pub fn new(train_start: i32, train_end: i32) -> Params {
        Params {
            train_start,
            train_end,
            harmonics: 2,
            xbar_train: 1.5,
            xbar_monitor: 20.0,
            lambda: 0.3,
            sigmas: 3.0,
            rounding: true,
            persistence: 3,
            low: f64::NEG_INFINITY,
        }
    }

    /// The harmonic model's coefficients: a constant, the sines, the cosines.
    pub fn n_coefficients(&self) -> usize {
        1 + 2 * self.harmonics
    }
}

/// The state at the last date.
pub const STABLE: f64 = 0.0;
pub const DISTURBED: f64 = 1.0;
pub const RECOVERED: f64 = 2.0;
pub const GAIN: f64 = 3.0;

/// What the monitoring dates' flags tell (NaN where they do not happen).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Events {
    /// time (as given) of the first negative flag after the training years
    pub break_time: f64,
    /// the most negative flag from it until the recovery
    pub severity: f64,
    /// time of the first flag back at 0 or above after the break
    pub recovery_time: f64,
    /// time of the first positive flag after the training years
    pub gain_time: f64,
    /// STABLE, DISTURBED, RECOVERED or GAIN at the last date
    pub state: f64,
}

impl Default for Events {
    fn default() -> Events {
        let nan = f64::NAN;
        Events {
            break_time: nan,
            severity: nan,
            recovery_time: nan,
            gain_time: nan,
            state: nan,
        }
    }
}

/// One pixel's chart.
#[derive(Clone, Debug, Default)]
pub struct Ewmacd {
    /// per date: the flag (NaN before `train_start`, or everywhere without a model)
    pub flags: Vec<f64>,
    /// per date: the EWMA of the residuals (NaN at the dates left out of the chart)
    pub ewma: Vec<f64>,
    /// per date: the control limit of the EWMA there (NaN where left out)
    pub limit: Vec<f64>,
    /// per date: the harmonic model (NaN before `train_start`)
    pub fitted: Vec<f64>,
    /// per date: kept in the chart
    pub charted: Vec<bool>,
    /// the model's coefficients: a constant, the sines, the cosines (empty without a model)
    pub beta: Vec<f64>,
    /// the standard deviation of the kept training residuals (the control limits' scale)
    pub sigma: f64,
    /// the model's RMSE and R² on the training dates it was fitted on
    pub rmse: f64,
    pub r2: f64,
    pub events: Events,
}

/// The row of the harmonic model at day of the year `doy`.
fn row(doy: f64, k: usize, out: &mut [f64]) {
    let th = doy * 2.0 * PI / 365.0;
    out[0] = 1.0;
    for j in 0..k {
        let a = (j + 1) as f64 * th;
        out[1 + j] = a.sin();
        out[1 + k + j] = a.cos();
    }
}

/// Least squares of the rows `idx` of `x` (n x p, row-major) on `v`, by the normal equations
/// (Cholesky); None when they are singular.
fn ols(x: &[f64], p: usize, v: &[f64], idx: &[usize]) -> Option<Vec<f64>> {
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    for &i in idx {
        let r = &x[i * p..(i + 1) * p];
        for c in 0..p {
            b[c] += r[c] * v[i];
            for d in 0..=c {
                a[c * p + d] += r[c] * r[d];
            }
        }
    }
    let scale = (0..p).map(|c| a[c * p + c]).fold(0.0, f64::max);
    for c in 0..p {
        for d in 0..=c {
            let mut s = a[c * p + d];
            for e in 0..d {
                s -= a[c * p + e] * a[d * p + e];
            }
            if c == d {
                if s <= 1e-12 * scale {
                    return None;
                }
                a[c * p + c] = s.sqrt();
            } else {
                a[c * p + d] = s / a[d * p + d];
            }
        }
    }
    for c in 0..p {
        let s: f64 = (0..c).map(|e| a[c * p + e] * b[e]).sum();
        b[c] = (b[c] - s) / a[c * p + c];
    }
    for c in (0..p).rev() {
        let s: f64 = (c + 1..p).map(|e| a[e * p + c] * b[e]).sum();
        b[c] = (b[c] - s) / a[c * p + c];
    }
    Some(b)
}

/// The sample standard deviation (n - 1) of `r` at `idx`, NaN under 2 values.
fn sd(r: &[f64], idx: impl Iterator<Item = usize> + Clone) -> f64 {
    let n = idx.clone().count();
    if n < 2 {
        return f64::NAN;
    }
    let mean = idx.clone().map(|i| r[i]).sum::<f64>() / n as f64;
    (idx.map(|i| (r[i] - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
}

/// Step 5: runs shorter than `persistence` take the last sustained value.
fn persist(f: &mut [f64], persistence: usize) {
    if persistence <= 1 {
        return;
    }
    let sign = |v: f64| {
        if v > 0.0 {
            1
        } else if v < 0.0 {
            -1
        } else {
            0
        }
    };
    let mut last = 0.0;
    let mut i = 0;
    while i < f.len() {
        let s = sign(f[i]);
        let mut j = i;
        while j < f.len() && sign(f[j]) == s {
            j += 1;
        }
        if j - i >= persistence {
            last = f[j - 1];
        } else {
            f[i..j].fill(last);
        }
        i = j;
    }
}

/// The events of `flags` at the dates from `first` on, `times` their times.
fn events(flags: &[f64], times: &[f64], first: usize) -> Events {
    let mut e = Events::default();
    if first >= flags.len() || flags[first..].iter().any(|v| v.is_nan()) {
        return e;
    }
    for i in first..flags.len() {
        let f = flags[i];
        if f > 0.0 && e.gain_time.is_nan() {
            e.gain_time = times[i];
        }
        if e.break_time.is_nan() {
            if f < 0.0 {
                e.break_time = times[i];
                e.severity = f;
            }
        } else if e.recovery_time.is_nan() {
            if f >= 0.0 {
                e.recovery_time = times[i];
            } else {
                e.severity = e.severity.min(f);
            }
        }
    }
    let last = flags[flags.len() - 1];
    e.state = if last < 0.0 {
        DISTURBED
    } else if e.break_time.is_finite() {
        RECOVERED
    } else if last > 0.0 {
        GAIN
    } else {
        STABLE
    };
    e
}

/// EWMACD of one pixel: the calendar `years`, days of the year `doys` and `times` (any increasing
/// time, as the events are to be given: decimal years) of its dates, in order.
pub fn ewmacd(years: &[i32], doys: &[f64], times: &[f64], values: &[f64], p: &Params) -> Ewmacd {
    let n = values.len();
    assert!(
        years.len() == n && doys.len() == n && times.len() == n,
        "one year, day and time per value"
    );
    let k = p.harmonics;
    let np = p.n_coefficients();
    let nan = f64::NAN;
    let mut out = Ewmacd {
        flags: vec![nan; n],
        ewma: vec![nan; n],
        limit: vec![nan; n],
        fitted: vec![nan; n],
        charted: vec![false; n],
        sigma: nan,
        rmse: nan,
        r2: nan,
        ..Default::default()
    };
    let inside = |i: usize| years[i] >= p.train_start;
    let ok: Vec<bool> = (0..n)
        .map(|i| inside(i) && values[i].is_finite() && values[i] > p.low)
        .collect();
    let train: Vec<usize> = (0..n).filter(|&i| ok[i] && years[i] < p.train_end).collect();
    if train.len() <= np || !(p.lambda > 0.0 && p.lambda <= 1.0) {
        return out;
    }
    let mut x = vec![0.0; n * np];
    for i in 0..n {
        row(doys[i], k, &mut x[i * np..(i + 1) * np]);
    }
    let predict = |beta: &[f64], i: usize| {
        x[i * np..(i + 1) * np]
            .iter()
            .zip(beta)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };

    // 1. the model, refitted without the training dates beyond xbar_train
    let Some(b0) = ols(&x, np, values, &train) else {
        return out;
    };
    let r0: Vec<f64> = (0..n)
        .map(|i| if ok[i] { values[i] - predict(&b0, i) } else { nan })
        .collect();
    let s = sd(&r0, train.iter().copied());
    let keep: Vec<usize> = train
        .iter()
        .copied()
        .filter(|&i| r0[i].abs() <= p.xbar_train * s)
        .collect();
    if keep.len() <= np {
        return out;
    }
    let Some(beta) = ols(&x, np, values, &keep) else {
        return out;
    };
    for i in (0..n).filter(|&i| inside(i)) {
        out.fitted[i] = predict(&beta, i);
    }
    let r: Vec<f64> = (0..n)
        .map(|i| if ok[i] { values[i] - out.fitted[i] } else { nan })
        .collect();
    let ss_res: f64 = keep.iter().map(|&i| r[i] * r[i]).sum();
    let mean = keep.iter().map(|&i| values[i]).sum::<f64>() / keep.len() as f64;
    let ss_tot: f64 = keep.iter().map(|&i| (values[i] - mean).powi(2)).sum();
    out.rmse = (ss_res / keep.len() as f64).sqrt();
    out.r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { nan };
    out.beta = beta;

    // 2. the second screen and the chart's standard deviation
    let s0 = sd(&r, train.iter().copied());
    let used: Vec<usize> = (0..n)
        .filter(|&i| {
            let lim = if years[i] < p.train_end {
                p.xbar_train
            } else {
                p.xbar_monitor
            };
            ok[i] && r[i].abs() < lim * s0
        })
        .collect();
    let sigma = sd(&r, used.iter().copied().filter(|&i| years[i] < p.train_end));
    out.sigma = sigma;
    if !(sigma > 0.0) || used.is_empty() {
        return out;
    }

    // 3-4. the EWMA chart and its flags
    let lam = p.lambda;
    let mut f = Vec::with_capacity(used.len());
    let mut z = 0.0;
    for (j, &i) in used.iter().enumerate() {
        z = if j == 0 { r[i] } else { (1.0 - lam) * z + lam * r[i] };
        let cl = p.sigmas * sigma * (lam / (2.0 - lam) * (1.0 - (1.0 - lam).powi(2 * (j as i32 + 1)))).sqrt();
        out.ewma[i] = z;
        out.limit[i] = cl;
        out.charted[i] = true;
        let ratio = z / cl;
        f.push(if p.rounding {
            ratio.abs().floor().copysign(ratio)
        } else if ratio.abs() >= 1.0 {
            ratio
        } else {
            0.0
        });
    }
    // 5. persistence
    if f.len() > 3 {
        persist(&mut f, p.persistence);
    }
    // 6. the dates left out carry the flag before them
    for (j, &i) in used.iter().enumerate() {
        out.flags[i] = f[j] + 0.0;
    }
    let Some(first) = (0..n).find(|&i| inside(i)) else {
        return out;
    };
    if out.flags[first].is_nan() {
        out.flags[first] = 0.0;
    }
    for i in first + 1..n {
        if out.flags[i].is_nan() {
            out.flags[i] = out.flags[i - 1];
        }
    }
    let monitor = (0..n)
        .find(|&i| years[i] >= p.train_end.max(p.train_start))
        .unwrap_or(n);
    out.events = events(&out.flags, times, monitor);
    out
}

/// The per-pixel maps of `batch`, in its order; the model's coefficients follow them.
pub const NAMES: [&str; 8] = [
    "break_time",
    "severity",
    "recovery_time",
    "gain_time",
    "state",
    "sigma",
    "rmse",
    "r2",
];

/// `ewmacd` of every pixel of `values` (pixels x dates): the flags (pixels x dates) and the maps
/// (NAMES then the coefficients, each a row of pixels).
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    years: &[i32],
    doys: &[f64],
    times: &[f64],
    p: &Params,
    n_jobs: i32,
) -> (Vec<f64>, Vec<f64>) {
    let t = years.len();
    assert_eq!(values.len(), n_pixels * t, "values must be n_pixels x dates");
    let nm = NAMES.len() + p.n_coefficients();
    let width = t + nm;
    let mut rows = vec![f64::NAN; n_pixels * width];
    parallel::for_each_row(
        &mut rows,
        width,
        n_jobs,
        || (),
        |_, i, out| {
            let e = ewmacd(years, doys, times, &values[i * t..(i + 1) * t], p);
            out[..t].copy_from_slice(&e.flags);
            let ev = e.events;
            out[t..t + NAMES.len()].copy_from_slice(&[
                ev.break_time,
                ev.severity,
                ev.recovery_time,
                ev.gain_time,
                ev.state,
                e.sigma,
                e.rmse,
                e.r2,
            ]);
            if !e.beta.is_empty() {
                out[t + NAMES.len()..].copy_from_slice(&e.beta);
            }
        },
    );
    let mut flags = Vec::with_capacity(n_pixels * t);
    let mut maps = Vec::with_capacity(n_pixels * nm);
    for r in rows.chunks_exact(width) {
        flags.extend_from_slice(&r[..t]);
        maps.extend_from_slice(&r[t..]);
    }
    (flags, parallel::transpose(&maps, n_pixels, nm))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seasonal series of `per_year` dates a year from 2005 to 2012, `noise` from a fixed
    /// sequence, with `change(t)` added.
    fn series(per_year: usize, noise: f64, change: impl Fn(f64) -> f64) -> (Vec<i32>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut years, mut doys, mut times, mut values) = (vec![], vec![], vec![], vec![]);
        let mut seed = 7u64;
        for y in 2005..2013 {
            for k in 0..per_year {
                let doy = (5 + k * 360 / per_year) as f64;
                let t = f64::from(y) + doy / 365.0;
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                let th = 2.0 * PI * doy / 365.0;
                years.push(y);
                doys.push(doy);
                times.push(t);
                values
                    .push(500.0 + 120.0 * th.sin() - 60.0 * th.cos() + 30.0 * (2.0 * th).sin() + noise * u + change(t));
            }
        }
        (years, doys, times, values)
    }

    #[test]
    fn the_model_of_a_clean_series() {
        let (y, d, t, v) = series(12, 0.0, |_| 0.0);
        let e = ewmacd(&y, &d, &t, &v, &Params::new(2005, 2009));
        let want = [500.0, 120.0, 30.0, -60.0, 0.0];
        for (b, w) in e.beta.iter().zip(want) {
            assert!((b - w).abs() < 1e-8, "{:?}", e.beta);
        }
        assert!(e.r2 > 0.999999999);
    }

    #[test]
    fn a_clearcut_breaks_and_recovers() {
        // 300 lost in mid 2009, regrown over two years
        let (y, d, t, mut v) = series(12, 20.0, |t| {
            if t >= 2009.5 {
                -300.0 * (1.0 - (t - 2009.5) / 2.0).max(0.0)
            } else {
                0.0
            }
        });
        // a cloud in the training years and one after: screened, they leave no signal
        v[20] -= 400.0;
        v[70] -= 400.0;
        let e = ewmacd(&y, &d, &t, &v, &Params::new(2005, 2009));
        assert!(!e.charted[20] && !e.charted[70] || e.flags[70] <= 0.0);
        let ev = e.events;
        assert!((2009.5..2009.7).contains(&ev.break_time), "{ev:?}");
        assert!(ev.severity <= -5.0, "{ev:?}");
        assert!((2010.5..2012.5).contains(&ev.recovery_time), "{ev:?}");
        assert_eq!(ev.state, RECOVERED);
        // the training years are quiet, save the chart's warm-up
        assert!(e.flags[12..48].iter().all(|&f| f == 0.0), "{:?}", &e.flags[..48]);
    }

    #[test]
    fn a_thinning_stays_disturbed_and_growth_is_a_gain() {
        let (y, d, t, v) = series(12, 20.0, |t| if t >= 2010.0 { -40.0 } else { 0.0 });
        let e = ewmacd(&y, &d, &t, &v, &Params::new(2005, 2009));
        assert_eq!(e.events.state, DISTURBED, "{:?}", e.events);
        assert!(e.events.recovery_time.is_nan());
        let (y, d, t, v) = series(12, 20.0, |t| if t >= 2010.0 { 40.0 } else { 0.0 });
        let e = ewmacd(&y, &d, &t, &v, &Params::new(2005, 2009));
        assert_eq!(e.events.state, GAIN, "{:?}", e.events);
        assert!((2010.0..2010.5).contains(&e.events.gain_time));
    }

    #[test]
    fn persistence_keeps_the_sustained_state() {
        let mut f = vec![0.0, -1.0, 0.0, 0.0, -2.0, -3.0, -3.0, 0.0, -2.0, -2.0, -1.0, 1.0];
        persist(&mut f, 3);
        assert_eq!(
            f,
            vec![0.0, 0.0, 0.0, 0.0, -2.0, -3.0, -3.0, -3.0, -2.0, -2.0, -1.0, -1.0]
        );
    }

    #[test]
    fn missing_dates_carry_the_flag_and_too_few_give_nan() {
        let (y, d, t, mut v) = series(12, 20.0, |t| if t >= 2010.0 { -300.0 } else { 0.0 });
        v[70] = f64::NAN;
        v[71] = -5.0;
        let mut p = Params::new(2005, 2009);
        p.low = 0.0;
        let e = ewmacd(&y, &d, &t, &v, &p);
        assert!(!e.charted[70] && !e.charted[71]);
        assert_eq!(e.flags[70], e.flags[69]);
        assert_eq!(e.flags[71], e.flags[69]);
        let e = ewmacd(&y[..5], &d[..5], &t[..5], &v[..5], &p);
        assert!(e.flags.iter().all(|f| f.is_nan()) && e.events.state.is_nan());
        // before train_start: NaN
        let e = ewmacd(&y, &d, &t, &v, &Params::new(2006, 2009));
        assert!(e.flags[..12].iter().all(|f| f.is_nan()) && e.flags[12] == 0.0);
    }

    #[test]
    fn batch_is_the_pixels() {
        let (y, d, t, a) = series(10, 20.0, |t| if t >= 2010.0 { -60.0 } else { 0.0 });
        let (_, _, _, b) = series(10, 20.0, |_| 0.0);
        let p = Params::new(2005, 2009);
        let values: Vec<f64> = a.iter().chain(&b).copied().collect();
        let (flags, maps) = batch(&values, 2, &y, &d, &t, &p, 1);
        let ea = ewmacd(&y, &d, &t, &a, &p);
        assert_eq!(&flags[..a.len()], &ea.flags[..]);
        assert_eq!(maps.len(), 2 * (NAMES.len() + 5));
        assert!(ea.events.break_time >= 2010.0);
        assert_eq!(maps[0], ea.events.break_time);
        assert_eq!(maps[2 * 4 + 1], STABLE);
        assert_eq!(maps[2 * NAMES.len()], ea.beta[0]);
    }
}
