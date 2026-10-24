//! S-CCD: Stochastic Continuous Change Detection (Ye, Rogan, Zhu & Eastman 2021, Remote Sensing
//! of Environment 252:112167), written from the paper (its preprint, EarthArXiv
//! 10.31223/osf.io/cqhsj) and the documented outputs of pyxccd, the authors' package, whose
//! source was not read (it is Apache-2.0; this file is GPL-2.0-only).
//!
//! S-CCD is CCDC/COLD (`ccdc.rs`) whose harmonic model, once initialised, is no longer refitted by
//! a lasso at every observation but updated by a Kalman filter: the trend and the seasons are
//! stochastic states that drift a little from one observation to the next. Per pixel:
//!
//! 1. the observations of CCDC: clear (Fmask 0 or 1) and inside the valid range; a pixel with
//!    less than a quarter of its observations clear has no model;
//! 2. initialisation as CCDC's (Tmask screening, the stability test of a 4-coefficient lasso,
//!    the look back, the segment before the first model), with 18 observations (3 x 6
//!    coefficients) over at least a year, as the paper; the stability test is COLD's (Zhu et
//!    al. 2020, Eq. 5: the slope plus the larger of the first and last residuals, which S-CCD
//!    inherits from COLD), not CCDC's sum of the three, which keeps many pixels of the Amazon
//!    from ever starting a model;
//! 3. the state-space model of each band, `y_t = Z_t a_t + e_t`, `a_(t+1) = a_t + n_t`: the
//!    coefficients of `a0 + c1 t + a1 cos wt + b1 sin wt + a2 cos 2wt + b2 sin 2wt` as the state.
//!    This is the paper's "trend + annual + semiannual" structural model, whose cycles rotate
//!    from day to day (its appendix): a cycle's two states are the coefficients `a`, `b` turned
//!    by the angle `wt`, so a rotation of the states and an isotropic noise on them are the same
//!    model as fixed axes and a noise on `a` and `b`. As pyxccd's six coefficients, the state
//!    also holds a slope, without process noise (the paper's trend is a random walk of the
//!    level, which its noise is here);
//! 4. its parameters (paper, 3.2): the initial state from a 6-coefficient lasso (lambda 20) of
//!    the initialisation window, `H` the square of its RMSE, the initial covariance `P = l I`
//!    with `l = (0.05 Z a)^2 / 3` (5% of the first modelled value, the uncertainty of Landsat
//!    surface reflectance, over the three observed states), and the process noise from a first
//!    filtering of the window without noise: the variance of the filtered level's steps
//!    `(m_n - m_(n-1)) / sqrt(t_n - t_(n-1))` for the level (per day), and for each cycle that
//!    variance times the mean absolute value of the cycle over the mean absolute level; the
//!    window is then filtered again with that noise;
//! 5. monitoring as CCDC's: the `conse` observations after the last one in the model, each
//!    standardized by its band's RMSE (no less than the median absolute difference of
//!    consecutive observations, CCDC's adjusted RMSE); a change when the smallest squared norm
//!    over the detection bands exceeds `chi2inv(prob_threshold, bands)` and the mean angle of
//!    the standardized vectors to their median vector is below `max_angle` (paper, 3.5: the
//!    "angular spread"); else the first observation is an outlier and left out when its norm
//!    exceeds `chi2inv(outlier_prob, bands)`, or enters the model (one Kalman update). The
//!    window is `conse` observations, whatever days they span: the paper lengthens it to 80
//!    days at least for dense series, which pyxccd does not (its breaks on 8-day series are
//!    those of 6 observations), nor this;
//! 6. the RMSE of a band is that of the model's one-step-ahead innovations (the filtered
//!    initialisation window's, then every observation added), kept recursively as a sum of
//!    squares per band, as pyxccd's model state (`rmse_sum`); the paper's temporally adjusted
//!    RMSE (the 24 or more observations nearest in the day of the year, from 6-day bins) was
//!    tried and left out: on the Rondonia stack it found 7% more breaks than this, further from
//!    pyxccd, for a point more of break dates in common;
//! 7. after a change the model starts again from the change, as CCDC's; at the end of the
//!    series the last `conse` observations are not in the model, and the near-real-time state
//!    says how many of the latest ones are anomalous.
//!
//! Dates are the Python ordinal days of `ccdc.rs`; the coefficients are on CCDC's time axis
//! (MATLAB datenum, ordinal + 366; c1 per day).

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::ccdc::{
    self, auto_tmask, cmax, date_order, median_rows, min_element, rows, Ctx, Fit, Models, Series, DATENUM_OFFSET,
    MINI_YRS, MIN_NUM_C, NUM_YRS, T_CLR,
};
use crate::parallel;
use crate::robustfit::median_of;

/// Coefficients of the model: a0, c1, a1, b1, a2, b2.
pub const NUM_C: usize = 6;
/// Observations of an initialisation window: 3 x 6 coefficients (the paper's 18).
const N_MIN: i32 = 18;
/// The uncertainty of the initial states: 5% of the modelled value (paper, 3.2).
const P0_FRACTION: f64 = 0.05;

/// The parameters of S-CCD.
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Consecutive anomalous observations that make a change (the peek window).
    pub conse: i32,
    /// Change probability: T_cg = chi2inv(prob_threshold, number of detection bands).
    pub prob_threshold: f64,
    /// Outlier probability of a single observation (the paper's 0.99999).
    pub outlier_prob: f64,
    /// Largest mean angle (degrees) of the peek window's change vectors to their median.
    pub max_angle: f64,
    /// Bands of the change test (0-based); empty: 1..=5 with 6 bands or more, else all.
    pub detection_bands: Vec<i32>,
    /// The two bands of Tmask (0-based); empty: green and SWIR1 (1, 4).
    pub tmask_bands: Vec<i32>,
    /// Valid range of the values (exclusive).
    pub valid_min: f64,
    pub valid_max: f64,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            conse: 6,
            prob_threshold: 0.99,
            outlier_prob: 0.99999,
            max_angle: 30.0,
            detection_bands: Vec::new(),
            tmask_bands: Vec::new(),
            valid_min: 0.0,
            valid_max: 10000.0,
        }
    }
}

impl Params {
    /// CCDC's parameters for the steps S-CCD shares with it.
    fn ccdc(&self) -> ccdc::Params {
        ccdc::Params {
            conseq_anom: self.conse,
            chi2_prob_threshold: self.prob_threshold,
            tmax_cg_prob_threshold: self.outlier_prob,
            num_c: NUM_C as i32,
            detection_bands: self.detection_bands.clone(),
            tmask_bands: self.tmask_bands.clone(),
            thermal_band: -1,
            valid_min: self.valid_min,
            valid_max: self.valid_max,
            ..ccdc::Params::default()
        }
    }
}

/// One model of a pixel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Segment {
    /// First observation of the model (Python ordinal day).
    pub t_start: i32,
    /// Last observation in the model.
    pub t_end: i32,
    /// First observation of the change, 0 if the model did not end in one.
    pub t_break: i32,
    /// Observations in the model.
    pub num_obs: i32,
    /// 1 for a change; for the open model, the share of the peek window that is anomalous.
    pub change_prob: f64,
    /// [band]: a0, c1, a1, b1, a2, b2 on CCDC's time axis (datenum), the state at `t_end`.
    pub coefs: Vec<[f64; NUM_C]>,
    /// [band]: RMSE of the model's innovations.
    pub rmse: Vec<f64>,
    /// [band]: median difference of the change's observations from the model, 0 without one.
    pub magnitude: Vec<f64>,
}

/// Near-real-time state of a pixel at the end of its series.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Monitor {
    /// 0 no model (no clear observation, or too few), 1 monitoring, 2 waiting for
    /// enough observations to start a model (before the first, or after a change).
    pub mode: i32,
    /// The latest observations, consecutive, that depart from the model (0..=conse).
    pub n_anomalies: i32,
    /// Date of the first of them (0 if none).
    pub anomaly_start: i32,
    /// The smallest standardized squared norm of them (the test statistic; NaN if none).
    pub anomaly_statistic: f64,
    /// [band]: their median difference from the model (0 if none).
    pub anomaly_magnitude: Vec<f64>,
    /// Standardized squared norm of the latest clear observation (NaN without a model).
    pub last_statistic: f64,
    /// Date of the latest clear observation (0 if none).
    pub last_date: i32,
    /// Clear observations waiting for a model (mode 2).
    pub n_queue: i32,
}

/// What became of each observation (the chart of a pixel).
pub mod status {
    /// Not clear, out of range or missing.
    pub const UNUSED: i32 = 0;
    /// In a model.
    pub const MODEL: i32 = 1;
    /// Left out: Tmask or the outlier test.
    pub const OUTLIER: i32 = 2;
    /// The first observation of a change.
    pub const BREAK: i32 = 3;
    /// After the last observation in the model: the final peek window.
    pub const PEEK: i32 = 4;
    /// Clear, but in no model (before one could start, or between a change and the next).
    pub const QUEUE: i32 = 5;
}

/// Per date (input order): what the chart of a pixel draws.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub status: Vec<i32>,
    /// The test statistic (standardized squared norm) when the observation was judged.
    pub statistic: Vec<f64>,
    /// [date][band]: the model's prediction before the observation was judged.
    pub predicted: Vec<f64>,
    /// [date][band]: the filtered states after the observation entered the model: the trend
    /// (a0 + c1 t), the annual and the semiannual cycles.
    pub trend: Vec<f64>,
    pub annual: Vec<f64>,
    pub semiannual: Vec<f64>,
}

impl Trace {
    fn new(n: usize, nb: usize) -> Trace {
        Trace {
            status: vec![status::UNUSED; n],
            statistic: vec![f64::NAN; n],
            predicted: vec![f64::NAN; n * nb],
            trend: vec![f64::NAN; n * nb],
            annual: vec![f64::NAN; n * nb],
            semiannual: vec![f64::NAN; n * nb],
        }
    }
}

// ---------------------------------------------------------------------------
// The Kalman filter of one band

#[derive(Clone, Debug)]
struct Kalman {
    /// level at t_ref, slope per year, a1, b1, a2, b2
    a: [f64; NUM_C],
    p: [[f64; NUM_C]; NUM_C],
    h: f64,
    /// process noise per day (diagonal)
    q: [f64; NUM_C],
    t_ref: f64,
    t_last: f64,
    /// sum of the squared innovations, and their count
    ss: f64,
    n: i32,
}

/// The observation row of datenum `t` (harmonics from CCDC's terms).
#[inline]
fn z_row(terms: &[f64; 8], t: f64, t_ref: f64) -> [f64; NUM_C] {
    [1.0, (t - t_ref) / NUM_YRS, terms[2], terms[3], terms[4], terms[5]]
}

#[inline]
fn dot(a: &[f64; NUM_C], b: &[f64; NUM_C]) -> f64 {
    let mut s = 0.0;
    for k in 0..NUM_C {
        s += a[k] * b[k];
    }
    s
}

impl Kalman {
    fn new(a: [f64; NUM_C], p0: f64, h: f64, t_ref: f64) -> Kalman {
        let mut p = [[0.0; NUM_C]; NUM_C];
        for k in 0..NUM_C {
            p[k][k] = p0;
        }
        Kalman {
            a,
            p,
            h,
            q: [0.0; NUM_C],
            t_ref,
            t_last: t_ref,
            ss: 0.0,
            n: 0,
        }
    }

    #[inline]
    fn predict(&self, z: &[f64; NUM_C]) -> f64 {
        dot(z, &self.a)
    }

    /// One observation: the state and its covariance carried to `t` (the noise of the days
    /// since the last), then updated with `y`. Returns the innovation.
    fn update(&mut self, z: &[f64; NUM_C], t: f64, y: f64) -> f64 {
        let dt = t - self.t_last;
        if dt > 0.0 {
            for k in 0..NUM_C {
                self.p[k][k] += self.q[k] * dt;
            }
            self.t_last = t;
        }
        let mut pz = [0.0; NUM_C];
        for i in 0..NUM_C {
            pz[i] = dot(&self.p[i], z);
        }
        let f = dot(z, &pz) + self.h;
        let v = y - self.predict(z);
        if f > 0.0 && f.is_finite() {
            for i in 0..NUM_C {
                self.a[i] += pz[i] * v / f;
            }
            for i in 0..NUM_C {
                for j in 0..NUM_C {
                    self.p[i][j] -= pz[i] * pz[j] / f;
                }
            }
        }
        self.ss += v * v;
        self.n += 1;
        v
    }

    fn rmse(&self) -> f64 {
        if self.n > 0 {
            (self.ss / self.n as f64).sqrt()
        } else {
            0.0
        }
    }

    /// The level at datenum `t`, the annual and semiannual cycles there.
    fn components(&self, terms: &[f64; 8], t: f64) -> (f64, f64, f64) {
        let a = &self.a;
        (
            a[0] + a[1] * (t - self.t_ref) / NUM_YRS,
            a[2] * terms[2] + a[3] * terms[3],
            a[4] * terms[4] + a[5] * terms[5],
        )
    }

    /// The state as CCDC's coefficients: a0 + c1 t + ... on the datenum axis.
    fn coefs(&self) -> [f64; NUM_C] {
        let c1 = self.a[1] / NUM_YRS;
        [
            self.a[0] - c1 * self.t_ref,
            c1,
            self.a[2],
            self.a[3],
            self.a[4],
            self.a[5],
        ]
    }
}

/// The models of every band from the initialisation window `win` (paper, 3.2).
fn init_models(s: &Series, win: &[usize], m: &mut Models, mut trace: Option<&mut Trace>) -> Fit<Vec<Kalman>> {
    let nb = m.nb;
    m.fit_all(s, win, NUM_C as i32, false)?;
    let t_ref = s.sdate[win[0]];
    let mut out = Vec::with_capacity(nb);
    let mut steps = Vec::with_capacity(win.len());
    for b in 0..nb {
        let cft = m.fit_cft[b];
        let c1 = cft[1];
        let a = [cft[0] + c1 * t_ref, c1 * NUM_YRS, cft[2], cft[3], cft[4], cft[5]];
        let h = m.rmse[b] * m.rmse[b];
        let h = if h.is_finite() && h > 1e-12 { h } else { 1e-12 };
        let z0 = z_row(&s.terms[win[0]], t_ref, t_ref);
        let first = dot(&z0, &a);
        let p0 = ((P0_FRACTION * first).powi(2) / 3.0).max(1e-12);

        // a first filtering without process noise: the steps of its level
        let mut kf = Kalman::new(a, p0, h, t_ref);
        steps.clear();
        let (mut sum_level, mut sum_annual, mut sum_semi) = (0.0, 0.0, 0.0);
        let mut prev: Option<(f64, f64)> = None;
        for &r in win {
            let t = s.sdate[r];
            let z = z_row(&s.terms[r], t, t_ref);
            kf.update(&z, t, s.line[r * nb + b]);
            let (level, annual, semi) = kf.components(&s.terms[r], t);
            sum_level += level.abs();
            sum_annual += annual.abs();
            sum_semi += semi.abs();
            // the level at t_ref: what a process noise would move (the slope is deterministic)
            let lv = kf.a[0];
            if let Some((t0, l0)) = prev {
                if t > t0 {
                    steps.push((lv - l0) / (t - t0).sqrt());
                }
            }
            prev = Some((t, lv));
        }
        let var = if steps.len() > 1 {
            let mean = steps.iter().sum::<f64>() / steps.len() as f64;
            steps.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (steps.len() - 1) as f64
        } else {
            0.0
        };
        let (q1, q2) = if sum_level > 0.0 {
            (var * sum_annual / sum_level, var * sum_semi / sum_level)
        } else {
            (0.0, 0.0)
        };

        // the window filtered again with that noise
        let mut kf = Kalman::new(a, p0, h, t_ref);
        kf.q = [var, 0.0, q1, q1, q2, q2];
        for &r in win {
            let t = s.sdate[r];
            let z = z_row(&s.terms[r], t, t_ref);
            if let Some(tr) = trace.as_deref_mut() {
                tr.predicted[r * nb + b] = kf.predict(&z);
            }
            kf.update(&z, t, s.line[r * nb + b]);
            if let Some(tr) = trace.as_deref_mut() {
                let (level, annual, semi) = kf.components(&s.terms[r], t);
                tr.trend[r * nb + b] = level;
                tr.annual[r * nb + b] = annual;
                tr.semiannual[r * nb + b] = semi;
            }
        }
        out.push(kf);
    }
    if let Some(tr) = trace {
        for &r in win {
            // the first observation after a change stays the change's
            if tr.status[r] != status::BREAK {
                tr.status[r] = status::MODEL;
            }
        }
    }
    Ok(out)
}

/// The peek window's test: per observation the differences from the models
/// ([obs][band]), the standardized squared norms; and the mean angle of the
/// standardized vectors to their median vector (degrees).
struct Peek {
    diff: Vec<Vec<f64>>,
    mag: Vec<f64>,
    scv: Vec<Vec<f64>>,
}

impl Peek {
    fn new() -> Peek {
        Peek {
            diff: Vec::new(),
            mag: Vec::new(),
            scv: Vec::new(),
        }
    }

    /// The observations `obs` (indices of the series) against the models `kf`.
    fn fill(&mut self, s: &Series, c: &Ctx, kf: &[Kalman], scale: &[f64], obs: &[usize]) {
        let nb = c.nb;
        let n = obs.len();
        self.diff.resize(n, Vec::new());
        self.scv.resize(n, Vec::new());
        self.mag.clear();
        for (k, &r) in obs.iter().enumerate() {
            let t = s.sdate[r];
            let diff = &mut self.diff[k];
            diff.clear();
            diff.resize(nb, 0.0);
            let scv = &mut self.scv[k];
            scv.clear();
            scv.resize(nb, 0.0);
            for b in 0..nb {
                let z = z_row(&s.terms[r], t, kf[b].t_ref);
                diff[b] = s.line[r * nb + b] - kf[b].predict(&z);
                if c.is_detect[b] {
                    scv[b] = diff[b] / scale[b];
                }
            }
            self.mag.push(c.vec_row(scv));
        }
        self.diff.truncate(n);
        self.scv.truncate(n);
    }

    /// Mean angle (degrees) between each standardized vector and their median vector, over
    /// the detection bands of observations k0..k1.
    fn angle(&self, c: &Ctx, k0: usize, k1: usize) -> f64 {
        let mut med = vec![0.0; c.nb];
        let mut col = Vec::with_capacity(k1 - k0);
        for &b in &c.b_detect {
            col.clear();
            col.extend(self.scv[k0..k1].iter().map(|v| v[b]));
            med[b] = median_of(&col);
        }
        let nm = c.b_detect.iter().map(|&b| med[b] * med[b]).sum::<f64>().sqrt();
        let mut total = 0.0;
        for v in &self.scv[k0..k1] {
            let nv = c.b_detect.iter().map(|&b| v[b] * v[b]).sum::<f64>().sqrt();
            let cos = if nm > 0.0 && nv > 0.0 {
                (c.b_detect.iter().map(|&b| v[b] * med[b]).sum::<f64>() / (nm * nv)).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            total += cos.acos().to_degrees();
        }
        total / (k1 - k0) as f64
    }
}

/// The standardizing RMSE of each band: the model's, no less than the adjusted RMSE.
fn scales(kf: &[Kalman], adj_rmse: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend(kf.iter().zip(adj_rmse).map(|(k, &a)| cmax(a, k.rmse())));
}

/// The segment of models `kf` that started at `t_start`.
fn segment_of(kf: &[Kalman], t_start: f64, t_end: f64) -> Segment {
    Segment {
        t_start: t_start as i32,
        t_end: t_end as i32,
        t_break: 0,
        num_obs: kf.first().map_or(0, |k| k.n),
        change_prob: 0.0,
        coefs: kf.iter().map(|k| k.coefs()).collect(),
        rmse: kf.iter().map(|k| k.rmse()).collect(),
        magnitude: vec![0.0; kf.len()],
    }
}

fn fit_pixel(s: &Series, c: &Ctx, p: &Params, mut trace: Option<&mut Trace>) -> Fit<(Vec<Segment>, Monitor)> {
    let num_t = s.sdate.len();
    let nb = c.nb;
    let mut segs: Vec<Segment> = Vec::new();
    let mut mon = Monitor {
        anomaly_statistic: f64::NAN,
        last_statistic: f64::NAN,
        anomaly_magnitude: vec![0.0; nb],
        ..Monitor::default()
    };

    let mut idclr = vec![false; num_t];
    let (mut n_clr, mut n_all) = (0i32, 0i32);
    for t in 0..num_t {
        let mut ok = true;
        for b in 0..nb {
            let v = s.line[t * nb + b];
            ok = ok && v > p.valid_min && v < p.valid_max;
        }
        idclr[t] = ok && s.qa[t] < 2;
        n_clr += (s.qa[t] < 2) as i32;
        n_all += (s.qa[t] < 255) as i32;
    }
    let mut d: Vec<usize> = (0..num_t).filter(|&t| idclr[t]).collect();
    if let Some(&last) = d.last() {
        mon.last_date = s.sdate[last] as i32;
    }
    if let Some(tr) = trace.as_deref_mut() {
        for &r in &d {
            tr.status[r] = status::QUEUE;
        }
    }
    if n_all == 0 || d.is_empty() || (n_clr as f64) / (n_all as f64) < T_CLR {
        return Ok((segs, mon));
    }

    let conse = c.conse;
    if conse < 1 {
        return Err("conse must be at least 1");
    }
    let x = |d: &[usize], i: i32| s.sdate[d[(i - 1) as usize]];
    let y = |d: &[usize], i: i32, b: usize| s.line[d[(i - 1) as usize] * nb + b];
    let pred4 = |d: &[usize], i: i32, cft: &[f64; 8]| {
        let tm = &s.terms[d[(i - 1) as usize]];
        let mut v = 0.0;
        for k in 0..8 {
            v += tm[k] * cft[k];
        }
        v
    };
    let first_at_or_after = |d: &[usize], t: f64| {
        for k in 1..=d.len() as i32 {
            if x(d, k) >= t {
                return k;
            }
        }
        d.len() as i32 + 1
    };

    let mut adj_rmse = vec![0.0; nb];
    let mut diffs = Vec::with_capacity(d.len());
    for b in 0..nb {
        diffs.clear();
        for i in 2..=d.len() as i32 {
            diffs.push((y(&d, i, b) - y(&d, i - 1, b)).abs());
        }
        adj_rmse[b] = if diffs.is_empty() { 0.0 } else { median_of(&diffs) };
    }

    let mut m = Models::new(nb);
    let mut i = N_MIN;
    let mut i_start = 1;
    let mut bl_train = false;
    let mut kf: Vec<Kalman> = Vec::new();
    let mut last_assim = 0; // the last observation (1-based in d) in the models
    let mut seg_start = 0.0;
    let mut n_fc = 0; // models made (the first one may look back to the start)
    let mut last_break = 0.0;
    let mut peek = Peek::new();
    let mut scale = Vec::with_capacity(nb);
    let mut obs = Vec::with_capacity(conse as usize);

    let mark = |tr: &mut Option<&mut Trace>, r: usize, st: i32| {
        if let Some(tr) = tr.as_deref_mut() {
            tr.status[r] = st;
        }
    };

    while i <= d.len() as i32 - conse {
        if !bl_train {
            let mut i_span = i - i_start + 1;
            let time_span = (x(&d, i) - x(&d, i_start)) / NUM_YRS;
            if i_span < N_MIN || time_span < MINI_YRS {
                i += 1;
                continue;
            }
            // Tmask over i_start .. i + conse
            let bl = auto_tmask(
                s,
                c,
                &d[(i_start - 1) as usize..(i + conse) as usize],
                (x(&d, i + conse) - x(&d, i_start)) / NUM_YRS,
                adj_rmse[c.b1],
                adj_rmse[c.b2],
            );
            let n_ids = bl.len() as i32;
            let mut rm_ids = Vec::new();
            i_span = 0;
            for k in 0..n_ids - conse {
                if bl[k as usize] {
                    rm_ids.push(i_start + k);
                } else {
                    i_span += 1;
                }
            }
            if i_span < N_MIN {
                i += 1;
                continue;
            }
            let mut cp = d.clone();
            for &id in rm_ids.iter().rev() {
                cp.remove((id - 1) as usize);
            }
            let i_rec = i;
            i = i_start + i_span - 1;
            if (x(&cp, i) - x(&cp, i_start)) / NUM_YRS < MINI_YRS {
                i = i_rec + 1;
                continue;
            }
            for &id in &rm_ids {
                mark(&mut trace, d[(id - 1) as usize], status::OUTLIER);
            }
            d = cp;
            // the stability test of a 4-coefficient model
            m.fit_all(s, rows(&d, i_start, i), MIN_NUM_C, true)?;
            let last = (i - i_start) as usize * nb;
            let mut vd = vec![0.0; nb];
            for &b in &c.b_detect {
                let mini_rmse = cmax(adj_rmse[b], m.rmse[b]);
                let v_start = m.rec_v_dif[b] / mini_rmse;
                let v_end = m.rec_v_dif[last + b] / mini_rmse;
                let v_slope = m.fit_cft[b][1] * (x(&d, i) - x(&d, i_start)) / mini_rmse;
                // COLD's test (Zhu et al. 2020, Eq. 5): the slope and the larger end residual
                vd[b] = v_slope.abs() + cmax(v_start.abs(), v_end.abs());
            }
            if c.vec_row(&vd) > c.t_cg {
                i_start += 1;
                i += 1;
                continue;
            }

            // look back, toward the start or the last change
            let i_break = if n_fc == 0 {
                1
            } else {
                first_at_or_after(&d, last_break)
            };
            let mut v_dif_mag: Vec<Vec<f64>> = Vec::new();
            if i_start > i_break {
                let mut i_ini = i_start - 1;
                while i_ini >= i_break {
                    let ini_conse = (i_start - i_break).min(conse);
                    let mut v_dif = vec![vec![0.0; nb]; ini_conse as usize];
                    v_dif_mag = v_dif.clone();
                    let mut vec_mag = vec![0.0; ini_conse as usize];
                    for ic in 1..=ini_conse {
                        let o = i_ini - ic + 1;
                        let r = (ic - 1) as usize;
                        for b in 0..nb {
                            v_dif_mag[r][b] = y(&d, o, b) - pred4(&d, o, &m.fit_cft[b]);
                            if c.is_detect[b] {
                                v_dif[r][b] = v_dif_mag[r][b] / cmax(adj_rmse[b], m.rmse[b]);
                            }
                        }
                        vec_mag[r] = c.vec_row(&v_dif[r]);
                    }
                    if min_element(&vec_mag) > c.t_cg {
                        break;
                    } else if vec_mag[0] > c.tmax_cg {
                        mark(&mut trace, d[(i_ini - 1) as usize], status::OUTLIER);
                        d.remove((i_ini - 1) as usize);
                        i -= 1;
                    }
                    i_start = i_ini;
                    i_ini -= 1;
                }
            }
            // the observations before the first model: a model of their own, ended by a change
            if n_fc == 0 && i_start - i_break >= conse {
                let win = rows(&d, i_break, i_start - 1);
                m.fit_all(s, win, MIN_NUM_C, false)?;
                let med = median_rows(&v_dif_mag, 0, v_dif_mag.len(), nb);
                segs.push(Segment {
                    t_start: x(&d, 1) as i32,
                    t_end: x(&d, i_start - 1) as i32,
                    t_break: x(&d, i_start) as i32,
                    num_obs: i_start - i_break,
                    change_prob: 1.0,
                    coefs: m
                        .fit_cft
                        .iter()
                        .map(|c8| [c8[0], c8[1], c8[2], c8[3], c8[4], c8[5]])
                        .collect(),
                    rmse: m.rmse.clone(),
                    magnitude: med.iter().map(|v| -v).collect(),
                });
                for &r in win {
                    mark(&mut trace, r, status::MODEL);
                }
            }

            // the state-space models
            let win = rows(&d, i_start, i);
            kf = init_models(s, win, &mut m, trace.as_deref_mut())?;
            seg_start = x(&d, i_start);
            last_assim = i;
            n_fc += 1;
            bl_train = true;
        }

        // bring the models up to observation i
        while last_assim < i {
            last_assim += 1;
            let r = d[(last_assim - 1) as usize];
            let t = s.sdate[r];
            for b in 0..nb {
                let z = z_row(&s.terms[r], t, kf[b].t_ref);
                kf[b].update(&z, t, s.line[r * nb + b]);
                if let Some(tr) = trace.as_deref_mut() {
                    let (level, annual, semi) = kf[b].components(&s.terms[r], t);
                    tr.trend[r * nb + b] = level;
                    tr.annual[r * nb + b] = annual;
                    tr.semiannual[r * nb + b] = semi;
                }
            }
            mark(&mut trace, r, status::MODEL);
        }

        // the peek window: observations i + 1 ..= i + conse
        obs.clear();
        obs.extend_from_slice(&d[i as usize..(i + conse) as usize]);
        scales(&kf, &adj_rmse, &mut scale);
        peek.fill(s, c, &kf, &scale, &obs);
        let first = obs[0];
        if let Some(tr) = trace.as_deref_mut() {
            tr.statistic[first] = peek.mag[0];
            for b in 0..nb {
                tr.predicted[first * nb + b] = s.line[first * nb + b] - peek.diff[0][b];
            }
        }
        if min_element(&peek.mag) > c.t_cg && peek.angle(c, 0, conse as usize) < p.max_angle {
            let mut seg = segment_of(&kf, seg_start, x(&d, i));
            seg.t_break = x(&d, i + 1) as i32;
            seg.change_prob = 1.0;
            seg.magnitude = median_rows(&peek.diff, 0, conse as usize, nb);
            segs.push(seg);
            mark(&mut trace, first, status::BREAK);
            last_break = x(&d, i + 1);
            i_start = i + 1;
            bl_train = false;
            kf.clear();
        } else if peek.mag[0] > c.tmax_cg {
            mark(&mut trace, first, status::OUTLIER);
            d.remove(i as usize);
            i -= 1;
        }
        i += 1;
    }

    let len = d.len() as i32;
    if bl_train {
        // the open model and the latest observations, after the last one in it
        obs.clear();
        obs.extend_from_slice(&d[last_assim as usize..]);
        let mut seg = segment_of(&kf, seg_start, x(&d, last_assim));
        if !obs.is_empty() {
            scales(&kf, &adj_rmse, &mut scale);
            peek.fill(s, c, &kf, &scale, &obs);
            let n = obs.len();
            let mut n_anom = 0;
            while n_anom < n && peek.mag[n - 1 - n_anom] > c.t_cg {
                n_anom += 1;
            }
            if n_anom < n {
                seg.t_end = s.sdate[obs[n - 1 - n_anom]] as i32;
            }
            seg.change_prob = (n_anom as f64 / conse as f64).min(1.0);
            mon.last_statistic = peek.mag[n - 1];
            mon.n_anomalies = n_anom as i32;
            if n_anom > 0 {
                mon.anomaly_start = s.sdate[obs[n - n_anom]] as i32;
                mon.anomaly_statistic = min_element(&peek.mag[n - n_anom..]);
                mon.anomaly_magnitude = median_rows(&peek.diff, n - n_anom, n, nb);
            }
            if let Some(tr) = trace {
                for (k, &r) in obs.iter().enumerate() {
                    tr.status[r] = status::PEEK;
                    tr.statistic[r] = peek.mag[k];
                    for b in 0..nb {
                        tr.predicted[r * nb + b] = s.line[r * nb + b] - peek.diff[k][b];
                    }
                }
            }
        }
        segs.push(seg);
        mon.mode = 1;
    } else {
        mon.mode = 2;
        mon.n_queue = (len - i_start + 1).max(0);
    }
    Ok((segs, mon))
}

/// fit_pixel on the series in date order, its dates back in the input convention.
fn fit_sorted(s: &Series, c: &Ctx, p: &Params, trace: Option<&mut Trace>) -> Fit<(Vec<Segment>, Monitor)> {
    let (mut segs, mut mon) = fit_pixel(s, c, p, trace)?;
    let back = |t: i32| if t > 0 { t - DATENUM_OFFSET } else { 0 };
    for seg in &mut segs {
        seg.t_start = back(seg.t_start);
        seg.t_end = back(seg.t_end);
        seg.t_break = back(seg.t_break);
    }
    mon.anomaly_start = back(mon.anomaly_start);
    mon.last_date = back(mon.last_date);
    Ok((segs, mon))
}

/// What S-CCD finds in one pixel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Result {
    pub segments: Vec<Segment>,
    pub monitor: Monitor,
    pub trace: Trace,
    /// The change threshold chi2inv(prob_threshold, detection bands), for the chart.
    pub threshold: f64,
}

/// S-CCD of one pixel. dates: Python ordinal days; bands: [band][time] (surface reflectance x
/// 10000, Blue, Green, Red, NIR, SWIR1, SWIR2); qa: Fmask codes (0 clear land, 1 water,
/// 2 shadow, 3 snow, 4 cloud, 255 no observation).
pub fn sccd(dates: &[i32], bands: &[Vec<f64>], qa: &[i32], params: &Params) -> Fit<Result> {
    if qa.len() != dates.len() {
        return Err("qa and dates must have the same length");
    }
    if bands.iter().any(|b| b.len() != dates.len()) {
        return Err("every band must have one value per date");
    }
    let nb = bands.len();
    if nb == 0 {
        return Err("no band");
    }
    let cp = params.ccdc();
    let c = Ctx::new(&cp, nb)?;
    let (order, sdate, terms) = date_order(dates);
    let mut line = vec![0.0; order.len() * nb];
    let mut q: Vec<i32> = Vec::with_capacity(order.len());
    for (k, &t) in order.iter().enumerate() {
        let mut nan = false;
        for b in 0..nb {
            line[k * nb + b] = bands[b][t];
            nan |= bands[b][t].is_nan();
        }
        q.push(if nan { 255 } else { qa[t] });
    }
    let s = Series {
        sdate: &sdate,
        terms: &terms,
        line: &line,
        qa: &q,
    };
    let mut tr = Trace::new(order.len(), nb);
    let (segments, monitor) = fit_sorted(&s, &c, params, Some(&mut tr))?;
    // the trace back in input order
    let mut trace = Trace::new(order.len(), nb);
    for (k, &t) in order.iter().enumerate() {
        trace.status[t] = tr.status[k];
        trace.statistic[t] = tr.statistic[k];
        for b in 0..nb {
            trace.predicted[t * nb + b] = tr.predicted[k * nb + b];
            trace.trend[t * nb + b] = tr.trend[k * nb + b];
            trace.annual[t * nb + b] = tr.annual[k * nb + b];
            trace.semiannual[t * nb + b] = tr.semiannual[k * nb + b];
        }
    }
    Ok(Result {
        segments,
        monitor,
        trace,
        threshold: c.t_cg,
    })
}

/// The change threshold of `params` with `nb` bands: chi2inv(prob_threshold, detection bands).
pub fn threshold(params: &Params, nb: usize) -> Fit<f64> {
    Ok(Ctx::new(&params.ccdc(), nb)?.t_cg)
}

/// Values per segment in the batch output: t_start, t_end, t_break, change_prob, num_obs,
/// then per band rmse, magnitude and the 6 coefficients.
pub fn segment_width(nb: usize) -> usize {
    5 + nb * (2 + NUM_C)
}

/// Values of the near-real-time state in the batch output: mode, n_anomalies, anomaly_start,
/// anomaly_statistic, last_statistic, last_date, n_queue, then per band anomaly_magnitude.
pub fn monitor_width(nb: usize) -> usize {
    7 + nb
}

/// S-CCD of every pixel of `values` ([pixels][bands][times]) with `qa` ([pixels][times]):
/// per pixel `max_segments` segments (`segment_width` values each), their count, then the
/// near-real-time state (`monitor_width` values). A date with NaN in any band has no
/// observation; a pixel without data, or whose fit fails, has no segment (mode 0).
pub fn sccd_batch(
    values: &[f64],
    qa: &[i32],
    dates: &[i32],
    pixels: usize,
    nb: usize,
    params: &Params,
    max_segments: usize,
    n_jobs: i32,
) -> Fit<Vec<f64>> {
    let times = dates.len();
    assert!(values.len() == pixels * nb * times && qa.len() == pixels * times);
    let cp = params.ccdc();
    let c = Ctx::new(&cp, nb)?;
    if params.conse < 1 {
        return Err("conse must be at least 1");
    }
    let (order, sdate, terms) = date_order(dates);
    let pps = segment_width(nb);
    let seg_width = max_segments * pps;
    let width = seg_width + 1 + monitor_width(nb);
    let mut out = vec![0.0; pixels * width];
    parallel::for_each_row(
        &mut out,
        width,
        n_jobs,
        || (vec![0.0; times * nb], vec![0i32; times]),
        |(line, q): &mut (Vec<f64>, Vec<i32>), px, row| {
            let v = &values[px * nb * times..(px + 1) * nb * times];
            let mut any_data = false;
            for (k, &t) in order.iter().enumerate() {
                q[k] = qa[px * times + t];
                let mut nan_here = false;
                for b in 0..nb {
                    let x = v[b * times + t];
                    line[k * nb + b] = x;
                    if x.is_nan() {
                        nan_here = true;
                    } else if x != 0.0 {
                        any_data = true;
                    }
                }
                if nan_here {
                    q[k] = 255;
                    line[k * nb..(k + 1) * nb].fill(0.0);
                }
            }
            let mon_out = &mut row[seg_width + 1..];
            mon_out[3] = f64::NAN;
            mon_out[4] = f64::NAN;
            if !any_data {
                return;
            }
            let s = Series {
                sdate: &sdate,
                terms: &terms,
                line,
                qa: q,
            };
            let Ok((segs, mon)) = fit_sorted(&s, &c, params, None) else {
                return;
            };
            let n_segs = segs.len().min(max_segments);
            for (i, seg) in segs.iter().take(n_segs).enumerate() {
                let o = &mut row[i * pps..(i + 1) * pps];
                o[0] = seg.t_start as f64;
                o[1] = seg.t_end as f64;
                o[2] = seg.t_break as f64;
                o[3] = seg.change_prob;
                o[4] = seg.num_obs as f64;
                let mut idx = 5;
                for b in 0..nb {
                    o[idx] = seg.rmse[b];
                    o[idx + 1] = seg.magnitude[b];
                    o[idx + 2..idx + 2 + NUM_C].copy_from_slice(&seg.coefs[b]);
                    idx += 2 + NUM_C;
                }
            }
            row[seg_width] = n_segs as f64;
            let o = &mut row[seg_width + 1..];
            o[0] = mon.mode as f64;
            o[1] = mon.n_anomalies as f64;
            o[2] = mon.anomaly_start as f64;
            o[3] = mon.anomaly_statistic;
            o[4] = mon.last_statistic;
            o[5] = mon.last_date as f64;
            o[6] = mon.n_queue as f64;
            o[7..7 + nb].copy_from_slice(&mon.anomaly_magnitude);
        },
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ccdc::W;

    /// 16-day series of 6 bands with a seasonal cycle, a little noise and a step at `at`.
    fn pixel(n: usize, at: Option<usize>) -> (Vec<i32>, Vec<Vec<f64>>, Vec<i32>) {
        let dates: Vec<i32> = (0..n as i32).map(|k| 730_120 + 16 * k).collect();
        let bands = (0..6)
            .map(|b| {
                dates
                    .iter()
                    .enumerate()
                    .map(|(k, &d)| {
                        let base = 800.0 + 300.0 * b as f64 + 200.0 * (W * (d as f64 + 366.0)).sin();
                        let shift = if at.is_some_and(|a| k >= a) { 1000.0 } else { 0.0 };
                        base + shift + ((k * 7919 + b * 104_729) % 61) as f64 - 30.0
                    })
                    .collect()
            })
            .collect();
        (dates, bands, vec![0; n])
    }

    #[test]
    fn finds_a_break() {
        let (dates, bands, qa) = pixel(230, Some(120));
        let r = sccd(&dates, &bands, &qa, &Params::default()).unwrap();
        let breaks: Vec<i32> = r.segments.iter().map(|s| s.t_break).filter(|&t| t > 0).collect();
        assert_eq!(breaks, vec![dates[120]], "{:?}", r.segments);
        assert_eq!(r.monitor.mode, 1);
        assert_eq!(r.trace.status[120], status::BREAK);
    }

    #[test]
    fn stable_series() {
        let (dates, bands, qa) = pixel(230, None);
        let r = sccd(&dates, &bands, &qa, &Params::default()).unwrap();
        assert_eq!(r.segments.len(), 1, "{:?}", r.segments);
        assert_eq!(r.segments[0].t_break, 0);
        assert_eq!(r.segments[0].num_obs, 230 - 6);
        assert_eq!(r.monitor.n_anomalies, 0);
        // the model follows the series
        for b in 0..6 {
            assert!(r.segments[0].rmse[b] < 40.0, "{:?}", r.segments[0].rmse);
        }
    }

    #[test]
    fn recent_anomalies() {
        // the change starts 4 observations before the end: not yet a break
        let (dates, bands, qa) = pixel(230, Some(226));
        let r = sccd(&dates, &bands, &qa, &Params::default()).unwrap();
        assert!(r.segments.iter().all(|s| s.t_break == 0));
        assert_eq!(r.monitor.mode, 1);
        assert_eq!(r.monitor.n_anomalies, 4);
        assert_eq!(r.monitor.anomaly_start, dates[226]);
        assert!(r.monitor.anomaly_magnitude.iter().all(|&m| m > 800.0));
    }

    #[test]
    fn too_short_or_cloudy() {
        let (dates, bands, qa) = pixel(20, None);
        let r = sccd(&dates, &bands, &qa, &Params::default()).unwrap();
        assert!(r.segments.is_empty());
        assert_eq!(r.monitor.mode, 2);
        assert_eq!(r.monitor.n_queue, 20);
        let (dates, bands, _) = pixel(100, None);
        let r = sccd(&dates, &bands, &[4; 100], &Params::default()).unwrap();
        assert!(r.segments.is_empty());
        assert_eq!(r.monitor.mode, 0);
    }
}
