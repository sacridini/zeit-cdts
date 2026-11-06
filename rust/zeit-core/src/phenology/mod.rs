//! Vegetation phenology (clean-room implementation, written from the papers).
//!
//! The pieces: `smooth` (Whittaker with upper-envelope iterations, HANTS and
//! the weight update), `seasons` (the split into growing seasons), `curves` and
//! `fit` (the seven seasonal curves and their weighted fit) and `metrics` (the
//! 19 transition metrics and the goodness of fit). This file assembles them
//! into the batch function of the Python layer, `fit_phenology_batch`:
//!
//! 1. weights: `w` in [0, 1] (the QC; 1 without), 0 where the value is not
//!    finite or the weight is not positive;
//! 2. smoothing: HANTS when `apply_hants`, else Whittaker with upper-envelope
//!    iterations when `apply_whittaker`, else the observations as they are;
//! 3. the split of the smoothed series into growing seasons;
//! 4. for each season, on one segment of the series (start to end, both
//!    included): the two-pass fit of the curve to the observations and the
//!    extraction of the metrics from the fitted curve; R2 and RMSE of the fit
//!    with the weights of its final pass.
//!
//! The dates are in days (the unit of the dates the Python layer passes: days
//! since January 1st of the first year); the metrics come out in the same unit.

pub mod curves;
pub mod fit;
pub mod metrics;
pub mod seasons;
pub mod smooth;

#[cfg(test)]
pub(crate) mod testutil;

pub use curves::CurveType;
pub use seasons::{split_growing_seasons, Season, SeasonSettings};

use crate::parallel;

/// Columns of the output: the 19 metrics of `metrics::extract_metrics`, then
/// R2 and RMSE.
pub const N_METRICS: usize = metrics::N_METRICS + 2;

/// `wfact` of the weight update between the two passes of the fit.
const WFACT: f64 = 0.5;

/// Iterations of the upper-envelope Whittaker.
const ENVELOPE_ITERATIONS: usize = 3;

/// The settings of the batch function (the arguments of the Python layer).
#[derive(Clone, Debug)]
pub struct Settings {
    /// The curve, as `CurveType::from_int` reads it.
    pub curve_type: i64,
    /// Seasons kept for each pixel (the slots of the output). With more found,
    /// the first ones in time (whose fit succeeds) stay.
    pub max_seasons: usize,
    pub whittaker_lambda: f64,
    pub apply_whittaker: bool,
    pub apply_hants: bool,
    pub hants_frequencies: i32,
    /// The error tolerance of HANTS as a fraction of the pixel's amplitude
    /// (not positive: the module's own, 0.1).
    pub hants_threshold: f64,
    /// Shortest season, in days of the dates.
    pub min_season_length: i32,
    /// Smallest season amplitude (peak minus the lower valley of the smoothed
    /// series), in the unit of the values.
    pub min_amplitude: f64,
    /// Series whose smoothed amplitude is below this have no seasons.
    pub min_pixel_amplitude: f64,
    pub rtrough_max: f64,
    pub r_min_filter: f64,
    pub season_retry: bool,
}

impl Default for Settings {
    /// The defaults of `zeit.phenology`.
    fn default() -> Self {
        Settings {
            curve_type: 0,
            max_seasons: 2,
            whittaker_lambda: 10.0,
            apply_whittaker: true,
            apply_hants: false,
            hants_frequencies: 3,
            hants_threshold: 0.1,
            min_season_length: 0,
            min_amplitude: 0.0,
            min_pixel_amplitude: 0.1,
            rtrough_max: 0.6,
            r_min_filter: 0.02,
            season_retry: true,
        }
    }
}

/// Median step between the dates, None when they do not increase (or are
/// fewer than two).
pub fn median_step(t: &[f64]) -> Option<f64> {
    if t.len() < 2 || !t.iter().all(|v| v.is_finite()) || !t.windows(2).all(|p| p[1] > p[0]) {
        return None;
    }
    let mut steps: Vec<f64> = t.windows(2).map(|p| p[1] - p[0]).collect();
    steps.sort_by(|a, b| a.total_cmp(b));
    let n = steps.len();
    Some(if n % 2 == 1 { steps[n / 2] } else { 0.5 * (steps[n / 2 - 1] + steps[n / 2]) })
}

/// Observations per year of dates `t` (days): 365.25 over the median step,
/// rounded, at least 1.
pub fn nptperyear(t: &[f64]) -> Option<usize> {
    median_step(t).map(|s| (365.25 / s).round().max(1.0) as usize)
}

/// The settings of the season split for the smoothed series `z` at the dates
/// `t`: the minimum length in days becomes a number of observations (a season of
/// `n` observations lasts `n - 1` steps), the minimum amplitude in the unit of
/// the values becomes a fraction of the amplitude of `z`. Every season found is
/// kept (no cap per year): `fit_pixel` fits them in time
/// order until `max_seasons` have a fit.
pub fn season_settings(settings: &Settings, t: &[f64], z: &[f64]) -> SeasonSettings {
    let step = median_step(t).unwrap_or(1.0);
    let min_season_length = if settings.min_season_length > 0 {
        (settings.min_season_length as f64 / step).ceil() as usize + 1
    } else {
        0
    };
    let (lo, hi) = z
        .iter()
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let amp = hi - lo;
    let min_amplitude = if amp > 0.0 { (settings.min_amplitude / amp).max(0.0) } else { 0.0 };
    SeasonSettings {
        min_season_length,
        min_amplitude,
        min_pixel_amplitude: settings.min_pixel_amplitude,
        rtrough_max: settings.rtrough_max,
        r_min_filter: settings.r_min_filter,
        season_retry: settings.season_retry,
        max_seasons: 0,
    }
}

/// The weights of the observations: finite value and positive weight (at most
/// 1), else 0.
fn clean_weights(y: &[f64], w: Option<&[f64]>) -> Vec<f64> {
    (0..y.len())
        .map(|i| {
            let wi = w.map_or(1.0, |w| w[i]);
            if y[i].is_finite() && wi.is_finite() && wi > 0.0 {
                wi.min(1.0)
            } else {
                0.0
            }
        })
        .collect()
}

/// The seasons of one pixel: the 19 metrics, then R2 and RMSE, in the order of
/// the dates, at most `settings.max_seasons`: the seasons whose fit fails are
/// left out and the next ones take their place.
fn fit_pixel(
    y: &[f64],
    t: &[f64],
    w: Option<&[f64]>,
    settings: &Settings,
    kind: CurveType,
    nptperyear: usize,
) -> Vec<[f64; N_METRICS]> {
    let w = clean_weights(y, w);
    let valid: Vec<f64> = (0..y.len()).filter(|&i| w[i] > 0.0).map(|i| y[i]).collect();
    if valid.len() < 4 {
        return Vec::new();
    }
    let amp = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - valid.iter().cloned().fold(f64::INFINITY, f64::min);
    // the smoothed series and the weights the fit starts from
    let (z, fit_w) = if settings.apply_hants {
        let threshold = if settings.hants_threshold > 0.0 { settings.hants_threshold * amp } else { 0.0 };
        let nf = settings.hants_frequencies.max(1) as usize;
        (smooth::hants(y, t, nf, threshold, Some(&w)), w.clone())
    } else if settings.apply_whittaker {
        match smooth::whittaker_envelope(Some(t), y, &w, settings.whittaker_lambda, ENVELOPE_ITERATIONS) {
            Some(e) => (e.z, e.weights),
            None => return Vec::new(),
        }
    } else {
        ((0..y.len()).map(|i| if w[i] > 0.0 { y[i] } else { f64::NAN }).collect(), w.clone())
    };
    let observed: Vec<bool> = w.iter().map(|&v| v > 0.0).collect();
    let ss = season_settings(settings, t, &z);
    let seasons = split_growing_seasons(&z, t, &ss, nptperyear, Some(&observed));
    let mut out = Vec::with_capacity(seasons.len());
    for s in seasons {
        if out.len() == settings.max_seasons {
            break;
        }
        let r = s.start..=s.end;
        let (ts, ys, ws) = (&t[r.clone()], &y[r.clone()], &fit_w[r]);
        let mut params = Vec::new();
        let Some(pass) = fit::fit_two_pass(ts, ys, ws, &mut params, kind, fit::default_max_fev(kind), nptperyear, WFACT)
        else {
            continue;
        };
        let m = metrics::extract_metrics(&params, kind, ts);
        let tau: Vec<f64> = ts.iter().map(|&x| x - ts[0]).collect();
        let yfit = curves::evaluate_curve(kind, &params, &tau);
        let (r2, rmse) = metrics::compute_gof(ys, &yfit, &pass.weights);
        let mut row = [f64::NAN; N_METRICS];
        row[..metrics::N_METRICS].copy_from_slice(&m);
        row[metrics::N_METRICS] = r2;
        row[metrics::N_METRICS + 1] = rmse;
        out.push(row);
    }
    out
}

/// Phenology of a batch of pixels: `values` is (pixels, time) row-major, `dates`
/// the days of the time axis, `weights` (same shape as `values`) the optional
/// reliability in [0, 1]. The result is (N_METRICS, pixels, max_seasons)
/// row-major, NaN where a pixel has no such season. At most `n_jobs` threads
/// (not positive: the processors zeit may use, see `parallel::threads`).
pub fn fit_phenology_batch(
    values: &[f64],
    dates: &[f64],
    n_pixels: usize,
    weights: Option<&[f64]>,
    settings: &Settings,
    n_jobs: i32,
) -> Vec<f64> {
    let n_time = dates.len();
    let slots = settings.max_seasons;
    let mut out = vec![f64::NAN; N_METRICS * n_pixels * slots];
    let (Some(kind), Some(npt)) = (CurveType::from_int(settings.curve_type), nptperyear(dates)) else {
        return out;
    };
    if n_pixels == 0 || slots == 0 || values.len() != n_pixels * n_time || weights.is_some_and(|w| w.len() != values.len()) {
        return out;
    }
    let width = slots * N_METRICS;
    let mut rows = vec![f64::NAN; n_pixels * width];
    parallel::for_each_row(&mut rows, width, n_jobs, || (), |_, p, row| {
        let r = p * n_time..(p + 1) * n_time;
        let w = weights.map(|w| &w[r.clone()]);
        for (s, m) in fit_pixel(&values[r], dates, w, settings, kind, npt).iter().enumerate() {
            row[s * N_METRICS..(s + 1) * N_METRICS].copy_from_slice(m);
        }
    });
    for p in 0..n_pixels {
        for s in 0..slots {
            for m in 0..N_METRICS {
                out[(m * n_pixels + p) * slots + s] = rows[p * width + s * N_METRICS + m];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::testutil::{dates16, Rng};
    use super::*;

    /// One season a year: a double logistic with 50% crossings at `sos` and
    /// `eos` (days of the year).
    fn season_series(t: &[f64], sos: f64, eos: f64, mn: f64, mx: f64) -> Vec<f64> {
        t.iter()
            .map(|&d| {
                let doy = d % 365.25;
                mn + (mx - mn) * (curves::sigmoid((doy - sos) / 10.0) - curves::sigmoid((doy - eos) / 10.0))
            })
            .collect()
    }

    fn settings(max_seasons: usize) -> Settings {
        Settings { max_seasons, min_season_length: 45, whittaker_lambda: 5.0, ..Settings::default() }
    }

    #[test]
    fn nptperyear_from_the_median_step() {
        assert_eq!(nptperyear(&dates16(2)), Some(23));
        assert_eq!(nptperyear(&[0.0, 1.0, 2.0]), Some(365));
        assert_eq!(nptperyear(&[0.0, 5.0, 5.0]), None);
        assert_eq!(nptperyear(&[0.0]), None);
    }

    #[test]
    fn season_settings_maps_units() {
        let t = dates16(5);
        let z: Vec<f64> = (0..t.len()).map(|i| if i % 2 == 0 { 0.2 } else { 0.7 }).collect();
        let s = Settings { max_seasons: 5, min_season_length: 45, min_amplitude: 0.1, ..Settings::default() };
        let ss = season_settings(&s, &t, &z);
        assert_eq!(ss.min_season_length, 4); // ceil(45 / 16) + 1 observations
        assert!((ss.min_amplitude - 0.2).abs() < 1e-12);
        assert_eq!(ss.max_seasons, 0); // all of them: fit_pixel keeps the first ones
    }

    #[test]
    fn five_years_one_season_each() {
        let t = dates16(5);
        let mut rng = Rng::new(3);
        let y: Vec<f64> = season_series(&t, 110.0, 270.0, 0.2, 0.8).iter().map(|v| v + 0.02 * rng.normal()).collect();
        for curve in 0..7 {
            let set = Settings { curve_type: curve, ..settings(5) };
            let out = fit_phenology_batch(&y, &t, 1, None, &set, 1);
            assert_eq!(out.len(), N_METRICS * 5);
            let slot = |m: usize, s: usize| out[m * 5 + s];
            let n = (0..5).filter(|&s| slot(18, s).is_finite()).count();
            assert!(n >= 4, "curve {curve}: {n} seasons");
            for s in 0..n {
                let (sos, eos) = (slot(2, s) % 365.25, slot(3, s) % 365.25);
                assert!((sos - 110.0).abs() < 12.0, "curve {curve} season {s}: sos {sos}");
                assert!((eos - 270.0).abs() < 12.0, "curve {curve} season {s}: eos {eos}");
                assert!(slot(19, s) > 0.9 && slot(20, s) < 0.06, "curve {curve}: r2 {} rmse {}", slot(19, s), slot(20, s));
            }
        }
    }

    #[test]
    fn layout_nan_and_threads() {
        let t = dates16(3);
        let mut rng = Rng::new(9);
        let n_pixels = 7;
        let mut values = Vec::new();
        for p in 0..n_pixels {
            let v = if p == 3 { vec![0.4; t.len()] } else { season_series(&t, 100.0 + 5.0 * p as f64, 260.0, 0.2, 0.8) };
            values.extend(v.iter().map(|x| x + 0.01 * rng.normal()));
        }
        let set = settings(3);
        let one = fit_phenology_batch(&values, &t, n_pixels, None, &set, 1);
        let four = fit_phenology_batch(&values, &t, n_pixels, None, &set, 4);
        assert_eq!(one.len(), N_METRICS * n_pixels * 3);
        for (a, b) in one.iter().zip(&four) {
            assert!(a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()), "threads change the result");
        }
        let at = |m: usize, p: usize, s: usize| one[(m * n_pixels + p) * 3 + s];
        for s in 0..3 {
            assert!(at(18, 3, s).is_nan() && at(19, 3, s).is_nan(), "the constant pixel has no season");
        }
        assert!(at(18, 0, 0).is_finite() && at(18, 6, 2).is_finite());
        // pixel order: the start of the season moves with p
        assert!(at(2, 6, 0) > at(2, 0, 0));
    }

    #[test]
    fn nan_gaps_and_zero_weights_are_gaps() {
        let t = dates16(3);
        let clean = season_series(&t, 110.0, 270.0, 0.2, 0.8);
        let mut y = clean.clone();
        let mut w = vec![1.0; t.len()];
        for i in [4, 5, 31, 50] {
            y[i] = f64::NAN;
        }
        y[10] = 0.0; // a bad value with no weight
        w[10] = 0.0;
        let set = settings(3);
        let a = fit_phenology_batch(&y, &t, 1, Some(&w), &set, 1);
        let b = fit_phenology_batch(&clean, &t, 1, None, &set, 1);
        for s in 0..3 {
            let (x, z) = (a[2 * 3 + s], b[2 * 3 + s]);
            assert!(x.is_finite() && (x - z).abs() < 3.0, "season {s}: {x} vs {z}");
        }
    }

    #[test]
    fn flagged_clouds_do_not_move_the_dates() {
        let t = dates16(3);
        let clean = season_series(&t, 110.0, 270.0, 0.2, 0.8);
        let mut y = clean.clone();
        let mut w = vec![1.0; t.len()];
        for i in [8, 30, 53] {
            y[i] *= 0.3;
            w[i] = 0.05;
        }
        let set = settings(3);
        let a = fit_phenology_batch(&y, &t, 1, Some(&w), &set, 1);
        let b = fit_phenology_batch(&clean, &t, 1, None, &set, 1);
        for s in 0..3 {
            assert!((a[2 * 3 + s] - b[2 * 3 + s]).abs() < 6.0, "season {s}");
            assert!(a[19 * 3 + s] > 0.95, "r2 {}", a[19 * 3 + s]);
        }
    }

    #[test]
    fn invalid_inputs_give_nan() {
        let t = dates16(2);
        let y = season_series(&t, 110.0, 270.0, 0.2, 0.8);
        let bad_curve = Settings { curve_type: 9, ..settings(2) };
        assert!(fit_phenology_batch(&y, &t, 1, None, &bad_curve, 1).iter().all(|v| v.is_nan()));
        let mut dup = t.clone();
        dup[5] = dup[4];
        assert!(fit_phenology_batch(&y, &dup, 1, None, &settings(2), 1).iter().all(|v| v.is_nan()));
        assert!(fit_phenology_batch(&y[..10], &t, 1, None, &settings(2), 1).iter().all(|v| v.is_nan()));
        assert_eq!(fit_phenology_batch(&y, &t, 1, None, &settings(0), 1).len(), 0);
    }

    #[test]
    fn hants_path_and_cap_of_seasons() {
        let t = dates16(4);
        let y = season_series(&t, 110.0, 270.0, 0.2, 0.8);
        let set = Settings { apply_whittaker: false, apply_hants: true, ..settings(2) };
        let out = fit_phenology_batch(&y, &t, 1, None, &set, 1);
        let found = (0..2).filter(|&s| out[18 * 2 + s].is_finite()).count();
        assert_eq!(found, 2, "capped to the two slots of four seasons");
        assert!(out[2 * 2] < out[2 * 2 + 1], "in the order of the dates");
    }
}
