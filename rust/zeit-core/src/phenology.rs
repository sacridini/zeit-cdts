//! Land surface phenology: a port of src/phenology.cpp, same operations in
//! the same order, so the same bits. Per pixel: a Whittaker (or HANTS)
//! smoothing, the growing seasons of the smoothed series, then per season a
//! curve fitted to the raw values (twice, the weights updated by wTSM in
//! between) and its transition dates (thresholds, derivatives, Gu, Zhang).

// The loops index the series as the C++'s do.
#![allow(clippy::needless_range_loop)]

pub mod curves;
pub mod seasons;
pub mod smooth;

use crate::eigen::{max_coeff, min_coeff, sum_by};
use crate::parallel;
pub use curves::CurveType;
pub use seasons::{split_growing_seasons, Season};

/// The metrics of a season, in their order in the output: TRS2/5/6 SOS and
/// EOS, DER SOS/POS/EOS, Gu UD/SD/DD/RD, Zhang greenup/maturity/senescence/
/// dormancy, LOS, POP, R2, RMSE.
pub const N_METRICS: usize = 21;

/// extract_metrics: the curve on 365 points over the season, the dates of
/// metrics 0 to 18.
pub fn extract_metrics(params: &[f64], kind: CurveType, t_segment: &[f64]) -> [f64; 19] {
    let mut m = [f64::NAN; 19];
    if t_segment.is_empty() {
        return m;
    }
    let n_pts = 365usize;
    let t_min = min_coeff(t_segment);
    let mut t_max = max_coeff(t_segment);
    if t_max - t_min < 1.0 {
        t_max = t_min + 365.0;
    }
    let t_eval = lin_spaced(n_pts, t_min, t_max);
    let fvec = curves::evaluate_curve(kind, params, &t_eval);

    let min_val = min_coeff(&fvec);
    let max_val = max_coeff(&fvec);
    let amplitude = max_val - min_val;

    // maxCoeff(&index): the first of the largest
    let mut pop_idx = 0;
    for i in 1..n_pts {
        if fvec[i] > fvec[pop_idx] {
            pop_idx = i;
        }
    }
    let pop = t_eval[pop_idx];

    let deriv = |f: &[f64], i: usize| {
        if i == 0 {
            (f[i + 1] - f[i]) / (t_eval[i + 1] - t_eval[i])
        } else if i == n_pts - 1 {
            (f[i] - f[i - 1]) / (t_eval[i] - t_eval[i - 1])
        } else {
            (f[i + 1] - f[i - 1]) / (t_eval[i + 1] - t_eval[i - 1])
        }
    };
    let d1: Vec<f64> = (0..n_pts).map(|i| deriv(&fvec, i)).collect();
    let d2: Vec<f64> = (0..n_pts).map(|i| deriv(&d1, i)).collect();
    let k: Vec<f64> = (0..n_pts).map(|i| d2[i] / (1.0 + d1[i] * d1[i]).powf(1.5)).collect();

    // DER
    let mut max_d1 = -1e9;
    let mut der_sos_idx = 0;
    for i in 0..=pop_idx {
        if d1[i] > max_d1 {
            max_d1 = d1[i];
            der_sos_idx = i;
        }
    }
    let mut min_d1 = 1e9;
    let mut der_eos_idx = n_pts - 1;
    for i in pop_idx..n_pts {
        if d1[i] < min_d1 {
            min_d1 = d1[i];
            der_eos_idx = i;
        }
    }
    let der_sos = t_eval[der_sos_idx];
    let der_eos = t_eval[der_eos_idx];
    let mut los = der_eos - der_sos;

    // thresholds
    let find_trs = |thr: f64| -> (f64, f64) {
        let (mut sos, mut eos) = (f64::NAN, f64::NAN);
        for i in 0..pop_idx {
            if fvec[i] < thr && fvec[i + 1] >= thr {
                let w = (thr - fvec[i]) / (fvec[i + 1] - fvec[i]);
                sos = t_eval[i] + w * (t_eval[i + 1] - t_eval[i]);
                break;
            }
        }
        if sos.is_nan() {
            sos = t_eval[0];
        }
        for i in pop_idx..n_pts - 1 {
            if fvec[i] >= thr && fvec[i + 1] < thr {
                let w = (thr - fvec[i]) / (fvec[i + 1] - fvec[i]);
                eos = t_eval[i] + w * (t_eval[i + 1] - t_eval[i]);
                break;
            }
        }
        if eos.is_nan() {
            eos = t_eval[n_pts - 1];
        }
        (sos, eos)
    };
    let (trs2_sos, trs2_eos) = find_trs(min_val + 0.2 * amplitude);
    let (trs5_sos, trs5_eos) = find_trs(min_val + 0.5 * amplitude);
    let (trs6_sos, trs6_eos) = find_trs(min_val + 0.6 * amplitude);

    // the first index of the extreme of `v` over `range`
    let arg = |v: &[f64], range: std::ops::RangeInclusive<usize>, start: f64, init: usize, larger: bool| {
        let (mut best, mut idx) = (start, init);
        for i in range {
            if (larger && v[i] > best) || (!larger && v[i] < best) {
                best = v[i];
                idx = i;
            }
        }
        idx
    };

    // Gu
    let ud = arg(&d2, 0..=der_sos_idx, -1e9, 0, true);
    let sd = arg(&d2, der_sos_idx..=pop_idx, 1e9, 0, false);
    let dd = arg(&d2, pop_idx..=der_eos_idx, 1e9, n_pts - 1, false);
    let rd = arg(&d2, der_eos_idx..=n_pts - 1, -1e9, n_pts - 1, true);
    // Zhang
    let greenup = arg(&k, 0..=der_sos_idx, -1e9, 0, true);
    let maturity = arg(&k, der_sos_idx..=pop_idx, 1e9, 0, false);
    let senescence = arg(&k, pop_idx..=der_eos_idx, 1e9, n_pts - 1, false);
    let dormancy = arg(&k, der_eos_idx..=n_pts - 1, -1e9, n_pts - 1, true);

    if los.is_nan() || los <= 0.0 {
        los = f64::NAN;
    }
    m = [
        trs2_sos,
        trs2_eos,
        trs5_sos,
        trs5_eos,
        trs6_sos,
        trs6_eos,
        der_sos,
        pop,
        der_eos,
        t_eval[ud],
        t_eval[sd],
        t_eval[dd],
        t_eval[rd],
        t_eval[greenup],
        t_eval[maturity],
        t_eval[senescence],
        t_eval[dormancy],
        los,
        pop,
    ];
    m
}

/// `VectorXd::LinSpaced(n, low, high)`.
fn lin_spaced(n: usize, low: f64, high: f64) -> Vec<f64> {
    let size1 = if n == 1 { 1 } else { n - 1 };
    let step = if n == 1 { 0.0 } else { (high - low) / (n - 1) as f64 };
    let flip = high.abs() < low.abs();
    (0..n)
        .map(|i| {
            if flip {
                if i == 0 {
                    low
                } else {
                    high - (size1 - i) as f64 * step
                }
            } else if i == size1 {
                high
            } else {
                low + i as f64 * step
            }
        })
        .collect()
}

/// compute_gof: weighted R2 and RMSE of the fit (phenofit's get_GOF).
pub fn compute_gof(y: &[f64], yfit: &[f64], w: &[f64]) -> (f64, f64) {
    let n = y.len();
    let wsum = sum_by(n, |i| w[i]);
    if wsum < 1e-9 || n < 2 {
        return (f64::NAN, f64::NAN);
    }
    let ybar_w = sum_by(n, |i| w[i] * y[i]) / wsum;
    let ss_res = sum_by(n, |i| {
        let d = y[i] - yfit[i];
        w[i] * (d * d)
    });
    let ss_tot = sum_by(n, |i| {
        let d = y[i] - ybar_w;
        w[i] * (d * d)
    });
    let rmse = (ss_res / wsum).sqrt();
    let r2 = if ss_tot > 1e-12 {
        1.0 - ss_res / ss_tot
    } else {
        f64::NAN
    };
    (r2, rmse)
}

/// The settings of `fit_phenology_batch` (the C++'s arguments).
#[derive(Clone, Debug)]
pub struct Settings {
    pub curve_type: i64,
    pub max_seasons: usize,
    pub whittaker_lambda: f64,
    pub apply_whittaker: bool,
    pub apply_hants: bool,
    pub hants_frequencies: i32,
    pub hants_threshold: f64,
    pub min_season_length: i32,
    pub min_amplitude: f64,
    pub min_pixel_amplitude: f64,
    pub rtrough_max: f64,
    pub r_min_filter: f64,
    pub season_retry: bool,
}

/// The metrics of one pixel into `out` (N_METRICS x max_seasons, NaN where none).
fn fit_pixel(
    y_raw: &[f64],
    w_pixel: Option<&[f64]>,
    dates: &[f64],
    s: &Settings,
    cache: &mut smooth::WhittakerCache,
    out: &mut [f64],
) {
    let w_raw: Option<Vec<f64>> = w_pixel.map(|w| {
        w.iter()
            .map(|&v| if v.is_nan() { 1.0 } else { v.clamp(0.0, 1.0) })
            .collect()
    });

    let mut valid = false;
    let (mut pixel_min, mut pixel_max) = (99999.0, -99999.0);
    for &v in y_raw {
        if !v.is_nan() && v != -9999.0 {
            valid = true;
            if v < pixel_min {
                pixel_min = v;
            }
            if v > pixel_max {
                pixel_max = v;
            }
        }
    }
    if !valid || pixel_max - pixel_min < s.min_pixel_amplitude {
        return;
    }

    let y_smooth = if s.apply_whittaker {
        match smooth::whittaker(y_raw, w_raw.as_deref(), s.whittaker_lambda, cache) {
            Some(z) => z,
            None => return,
        }
    } else if s.apply_hants {
        smooth::hants(y_raw, dates, s.hants_frequencies, s.hants_threshold, w_raw.as_deref())
    } else {
        y_raw.to_vec()
    };

    let found = split_growing_seasons(
        &y_smooth,
        dates,
        s.min_season_length,
        s.min_amplitude,
        s.rtrough_max,
        s.r_min_filter,
        s.season_retry,
    );
    let kind = CurveType::from_int(s.curve_type);
    let ms = s.max_seasons;
    let mut s_idx = 0;
    for season in found {
        if s_idx >= ms {
            break;
        }
        let (start, end) = (season.start, season.end);
        let len = end + 1 - start;
        if len < 4 {
            continue;
        }
        let t_seg = &dates[start..=end];
        let y_seg = &y_raw[start..=end];
        let Some(kind) = kind else {
            // an unknown curve: fit_curve fails
            continue;
        };
        let mut params = vec![1.0; kind.n_params()];
        let mut w_seg: Vec<f64> = match &w_raw {
            Some(w) => w[start..=end].to_vec(),
            None => vec![1.0; len],
        };
        let mut converged = false;
        let iters = 2;
        for iter in 1..=iters {
            converged = curves::fit_curve(t_seg, y_seg, &w_seg, &mut params, kind, 2000);
            if converged && iter < iters {
                let yfit = curves::evaluate_curve(kind, &params, t_seg);
                w_seg = smooth::wtsm(y_seg, &yfit, &w_seg, iter, 23, 0.5);
            }
        }
        if converged {
            let metrics = extract_metrics(&params, kind, t_seg);
            for (mi, &v) in metrics.iter().enumerate() {
                out[mi * ms + s_idx] = v;
            }
            let yfit = curves::evaluate_curve(kind, &params, t_seg);
            let (r2, rmse) = compute_gof(y_seg, &yfit, &w_seg);
            out[19 * ms + s_idx] = r2;
            out[20 * ms + s_idx] = rmse;
            s_idx += 1;
        }
    }
}

/// fit_phenology_batch: `values` (pixels x time, row-major), `dates` (time),
/// optional `weights` like `values`; (N_METRICS, pixels, max_seasons),
/// row-major, NaN where there is no season.
pub fn fit_phenology_batch(
    values: &[f64],
    dates: &[f64],
    n_pixels: usize,
    weights: Option<&[f64]>,
    settings: &Settings,
    n_jobs: i32,
) -> Vec<f64> {
    let n_time = dates.len();
    let ms = settings.max_seasons;
    let width = N_METRICS * ms;
    // per pixel, then laid out metric by metric
    let mut per_pixel = vec![f64::NAN; n_pixels * width];
    if n_time > 0 && width > 0 {
        parallel::for_each_row(
            &mut per_pixel,
            width,
            n_jobs,
            smooth::WhittakerCache::default,
            |cache, p, out| {
                let y = &values[p * n_time..(p + 1) * n_time];
                let w = weights.map(|w| &w[p * n_time..(p + 1) * n_time]);
                fit_pixel(y, w, dates, settings, cache, out);
            },
        );
    }
    let mut out = vec![f64::NAN; n_pixels * width];
    for p in 0..n_pixels {
        for m in 0..N_METRICS {
            let src = &per_pixel[p * width + m * ms..p * width + (m + 1) * ms];
            out[(m * n_pixels + p) * ms..(m * n_pixels + p + 1) * ms].copy_from_slice(src);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_years_of_beck() {
        let t: Vec<f64> = (0..46).map(|i| 1.0 + 16.0 * i as f64).collect();
        let truth = [0.2, 0.8, 120.0, 0.08, 280.0, 0.06];
        let mut y = vec![];
        for &d in &t {
            let doy = (d - 1.0) % 368.0 + 1.0;
            y.push(curves::evaluate_curve(CurveType::Beck, &truth, &[doy])[0]);
        }
        let settings = Settings {
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
        };
        let out = fit_phenology_batch(&y, &t, 1, None, &settings, 1);
        assert_eq!(out.len(), N_METRICS * 2);
        // at least one season with its R2
        assert!(out[19 * 2].is_finite(), "{out:?}");
    }
}
