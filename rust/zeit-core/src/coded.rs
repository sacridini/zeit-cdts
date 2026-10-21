//! Continuous Degradation Detection (CODED; Bullock, Woodcock & Olofsson 2020),
//! the change monitoring of one pixel's NDFI series as in CODED's paper version
//! (cdd_simple.py): a port of src/coded.cpp, same operations in the same order
//! (Eigen's column-major QR included), so the same bits. See coded.h for the
//! method and the layout of the output.

use crate::eigen::qr::{ColPivQr, Mat, Order};
use crate::parallel;

use std::f64::consts::PI;

/// The parameters of CODED (the C++'s coded::Params).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub train_start: f64,
    pub train_years: f64,
    pub consec: i32,
    pub thresh: f64,
    pub min_years: f64,
    pub min_obs: i32,
    pub loss_only: bool,
    pub max_events: i32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            train_start: 0.0,
            train_years: 3.0,
            consec: 3,
            thresh: 3.0,
            min_years: 3.0,
            min_obs: 6,
            loss_only: true,
            max_events: 3,
        }
    }
}

#[derive(Clone, Copy)]
struct Model {
    c: f64,
    a: f64,
    b: f64,
    rmse: f64,
    n: usize,
}

impl Model {
    const NONE: Model = Model {
        c: f64::NAN,
        a: f64::NAN,
        b: f64::NAN,
        rmse: f64::NAN,
        n: 0,
    };

    fn ok(&self) -> bool {
        self.n > 0
    }

    /// The model at date `i`.
    fn predict(&self, d: &Dates, i: usize) -> f64 {
        self.c + self.a * d.sin[i] + self.b * d.cos[i]
    }
}

/// The dates and their sin(2 pi t) and cos(2 pi t), computed once for all the
/// pixels of a batch (the same values the C++ computes again in every fit).
pub struct Dates<'a> {
    t: &'a [f64],
    sin: Vec<f64>,
    cos: Vec<f64>,
}

impl<'a> Dates<'a> {
    pub fn new(t: &'a [f64]) -> Dates<'a> {
        Dates {
            t,
            sin: t.iter().map(|&tt| (2.0 * PI * tt).sin()).collect(),
            cos: t.iter().map(|&tt| (2.0 * PI * tt).cos()).collect(),
        }
    }
}

/// OLS of y = c + a sin(2 pi t) + b cos(2 pi t) on the observations `idx` (finite y).
fn fit(d: &Dates, y: &[f64], idx: &[usize], min_obs: i32) -> Model {
    let use_: Vec<usize> = idx.iter().copied().filter(|&i| y[i].is_finite()).collect();
    if (use_.len() as i64) < min_obs.max(4) as i64 {
        return Model::NONE;
    }
    let m = use_.len();
    let mut data = Vec::with_capacity(3 * m);
    data.extend(std::iter::repeat_n(1.0, m));
    data.extend(use_.iter().map(|&i| d.sin[i]));
    data.extend(use_.iter().map(|&i| d.cos[i]));
    let x = Mat::from_data(m, 3, Order::Col, data);
    let yy: Vec<f64> = use_.iter().map(|&i| y[i]).collect();
    let beta = ColPivQr::new(x).solve(&yy);
    let mut sse = 0.0;
    for (&i, &yr) in use_.iter().zip(&yy) {
        // X.row(r).dot(beta): the expression has the row's dynamic size (the
        // left operand's), so Eigen sums the three terms one after the other
        let dot = 1.0 * beta[0] + d.sin[i] * beta[1] + d.cos[i] * beta[2];
        let res = yr - dot;
        sse += res * res;
    }
    Model {
        c: beta[0],
        a: beta[1],
        b: beta[2],
        rmse: (sse / m as f64).sqrt(),
        n: m,
    }
}

fn window(t: &[f64], from: f64, to: f64) -> Vec<usize> {
    (0..t.len()).filter(|&i| t[i] >= from && t[i] < to).collect()
}

fn put_model(out: &mut [f64], m: &Model) {
    out[0] = m.c;
    out[1] = m.a;
    out[2] = m.b;
}

/// The models of the features on the same observations as NDFI (its finite ones).
fn put_features(out: &mut [f64], d: &Dates, ndfi: &[f64], features: &[f64], n_features: usize, idx: &[usize]) {
    let n_time = d.t.len();
    let use_: Vec<usize> = idx.iter().copied().filter(|&i| ndfi[i].is_finite()).collect();
    for f in 0..n_features {
        let m = fit(d, &features[f * n_time..(f + 1) * n_time], &use_, 1);
        put_model(&mut out[3 * f..], &m);
    }
}

/// The length of a pixel's output.
pub fn coded_size(n_features: i64, max_events: i64) -> i64 {
    let header = 3 + 1 + 1 + 3 * n_features + 1;
    let event = 3 + 3 + 1 + 3 * n_features;
    header + max_events * event
}

/// CODED on one pixel: `t` the decimal years (sorted), `ndfi` one value per
/// date, `features` n_features series of one value per date.
pub fn coded_pixel(
    t: &[f64],
    ndfi: &[f64],
    features: &[f64],
    n_features: usize,
    p: &Params,
) -> Result<Vec<f64>, &'static str> {
    let size = coded_size(n_features as i64, p.max_events as i64).max(0) as usize;
    let mut out = vec![f64::NAN; size];
    coded_into(&Dates::new(t), ndfi, features, n_features, p, &mut out)?;
    Ok(out)
}

fn coded_into(
    d: &Dates,
    y: &[f64],
    feats: &[f64],
    n_features: usize,
    p: &Params,
    out: &mut [f64],
) -> Result<(), &'static str> {
    let t = d.t;
    let n = t.len();
    if y.len() != n || feats.len() != n_features * n {
        return Err("coded: ndfi must have one value per date, features n_features per date");
    }
    if (1..n).any(|i| !(t[i] >= t[i - 1])) {
        return Err("coded: the dates must be sorted");
    }
    out.fill(f64::NAN);
    let header = 3 + 1 + 1 + 3 * n_features + 1;
    let per_event = 3 + 3 + 1 + 3 * n_features;

    let train_end = p.train_start + p.train_years;
    let train = window(t, p.train_start, train_end);
    let mut model = fit(d, y, &train, p.min_obs);
    out[header - 1] = 0.0; // number of changes
    if !model.ok() {
        return Ok(());
    }
    put_model(out, &model);
    out[3] = model.rmse;
    out[4] = model.n as f64;
    put_features(&mut out[5..], d, y, feats, n_features, &train);

    let (mut events, mut run) = (0i32, 0i32);
    let (mut first, mut last_before, mut previous_valid): (Option<usize>, Option<usize>, Option<usize>) =
        (None, None, None);
    let mut sum = 0.0;
    let mut i = 0;
    while i < n && t[i] < train_end {
        if y[i].is_finite() {
            previous_valid = Some(i);
        }
        i += 1;
    }
    while i < n && events < p.max_events {
        if !y[i].is_finite() {
            i += 1;
            continue;
        }
        let res = y[i] - model.predict(d, i);
        let z = res / if model.rmse < 1e-12 { 1e-12 } else { model.rmse };
        let beyond = if p.loss_only { z < -p.thresh } else { z.abs() > p.thresh };
        if !beyond {
            run = 0;
            sum = 0.0;
            previous_valid = Some(i);
            i += 1;
            continue;
        }
        if run == 0 {
            first = Some(i);
            last_before = previous_valid;
        }
        run += 1;
        sum += res;
        previous_valid = Some(i);
        if run < p.consec {
            i += 1;
            continue;
        }

        // a change: date, the observation before it, magnitude
        let f = first.expect("a run has a first observation");
        let ev = header + events as usize * per_event;
        out[ev] = t[f];
        out[ev + 1] = last_before.map_or(f64::NAN, |b| t[b]);
        out[ev + 2] = sum / run as f64;
        events += 1;
        out[header - 1] = events as f64;
        // the land cover after it: a new model on the next min_years
        let resume = t[f] + p.min_years;
        let after = window(t, t[f], resume);
        let next = fit(d, y, &after, p.min_obs);
        if !next.ok() {
            break;
        }
        put_model(&mut out[ev + 3..], &next);
        out[ev + 6] = next.rmse;
        put_features(&mut out[ev + 7..], d, y, feats, n_features, &after);
        model = next;
        run = 0;
        sum = 0.0;
        first = None;
        previous_valid = None;
        while i + 1 < n && t[i + 1] < resume {
            i += 1;
            if y[i].is_finite() {
                previous_valid = Some(i);
            }
        }
        i += 1;
    }
    Ok(())
}

/// `coded_pixel` on every pixel: `ndfi` (pixels x T), `features`
/// (pixels x n_features x T), row-major: pixels x coded_size values.
pub fn coded_batch(
    t: &[f64],
    ndfi: &[f64],
    features: &[f64],
    pixels: usize,
    n_features: usize,
    p: &Params,
    n_jobs: i32,
) -> Result<Vec<f64>, &'static str> {
    let n_time = t.len();
    let size = coded_size(n_features as i64, p.max_events as i64).max(0) as usize;
    let mut out = vec![0.0; pixels * size];
    let error = std::sync::Mutex::new(None);
    let dates = Dates::new(t);
    parallel::for_each_row(
        &mut out,
        size,
        n_jobs,
        || (),
        |_, px, row| {
            let y = &ndfi[px * n_time..(px + 1) * n_time];
            let f = &features[px * n_features * n_time..(px + 1) * n_features * n_time];
            if let Err(e) = coded_into(&dates, y, f, n_features, p, row) {
                *error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
            }
        },
    );
    match error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_drop() {
        let t: Vec<f64> = (0..200).map(|i| 2000.0 + i as f64 * 0.05).collect();
        let y: Vec<f64> = t
            .iter()
            .map(|&tt| 0.85 + 0.03 * (2.0 * PI * tt).sin() - if tt >= 2006.0 { 0.5 } else { 0.0 })
            .collect();
        let p = Params {
            train_start: 2000.0,
            ..Params::default()
        };
        let out = coded_pixel(&t, &y, &[], 0, &p).unwrap();
        assert_eq!(out[5], 1.0); // one change (header: 3 + 1 + 1 + 0 + 1)
        assert!((out[6] - 2006.0).abs() < 1e-9);
        assert!(out[8] < -0.4);
    }

    #[test]
    fn unsorted_dates_are_an_error() {
        assert!(coded_pixel(&[2.0, 1.0], &[0.0, 0.0], &[], 0, &Params::default()).is_err());
    }
}
