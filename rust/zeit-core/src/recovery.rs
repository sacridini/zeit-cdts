//! Recovery after a disturbance, from an annual series: the spectral recovery metrics of the forest
//! literature (Kennedy et al. 2012; White et al. 2017; Frazier et al. 2018), on raw values or on a
//! fitted trajectory (LandTrendr's, whose vertices `trajectory` interpolates).
//!
//! The disturbance runs from `start`, the last year before it (LandTrendr's year of detection, the
//! event's `yod`), to `trough`, the year it bottoms out (`yod + duration`). With `t0` the trough:
//!
//! - `pre_val`: the mean of the `pre_years` years up to `start` (NBR_pre: the 2 years before);
//! - `dist_val`: the value at `t0`; `magnitude`: `pre_val - dist_val` (ΔNBR), positive for a loss;
//! - `post_val`: the best of years `t0 + horizon - 1` and `t0 + horizon` (the maximum of years 4
//!   and 5 for a loss, Frazier et al. 2018), robust to one noisy year;
//! - `ari` (absolute recovery, ΔNBR_regrowth): `post_val - dist_val`;
//! - `rri` (relative recovery indicator, Kennedy et al. 2012): `ari / magnitude`;
//! - `r80p` (ratio of the target, 80% by default): `post_val / (target * pre_val)`, 1 or more when
//!   recovered; with the magnitude basis, the share of the target fraction of the loss regained;
//! - `yryr` (year-on-year recovery): `(value at t0 + horizon - dist_val) / horizon`;
//! - `y2r` (years to recovery, White et al. 2017): the years from `t0` until the value first reaches
//!   the recovery level (`target * pre_val`, or `dist_val + target * magnitude`); 0 when it never
//!   fell below it, NaN when it has not reached it by the end of the series.
//!
//! For a gain (an index that rises with the disturbance) the signs turn: the magnitude, `ari` and
//! `yryr` are positive in the direction of recovery and `post_val` is the lower of the two years.

use crate::parallel;

/// The level a pixel must return to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    /// `target * pre_val` (White et al. 2017's 80% of the pre-disturbance value; an index that is
    /// positive and drops with the disturbance, such as NBR)
    PreValue,
    /// `dist_val + target * (pre_val - dist_val)`: the target fraction of the loss regained (any
    /// index, either direction)
    Magnitude,
}

impl Basis {
    pub fn parse(s: &str) -> Option<Basis> {
        match s {
            "pre_value" | "pre" => Some(Basis::PreValue),
            "magnitude" => Some(Basis::Magnitude),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// the fraction of the pre-disturbance level (or of the loss) that counts as recovered
    pub target: f64,
    /// years after the trough at which `post_val`, `ari`, `rri`, `r80p` and `yryr` are measured
    pub horizon: i32,
    /// years averaged for `pre_val`, up to the disturbance's start
    pub pre_years: i32,
    pub basis: Basis,
    /// the disturbance lowers the index (true) or raises it
    pub loss: bool,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            target: 0.8,
            horizon: 5,
            pre_years: 2,
            basis: Basis::PreValue,
            loss: true,
        }
    }
}

/// The metrics in `batch`'s order.
pub const NAMES: [&str; 9] = [
    "pre_val",
    "dist_val",
    "magnitude",
    "post_val",
    "ari",
    "rri",
    "r80p",
    "yryr",
    "y2r",
];
pub const N_METRICS: usize = NAMES.len();

/// One pixel's metrics (NaN where they cannot be measured).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Recovery {
    pub pre_val: f64,
    pub dist_val: f64,
    pub magnitude: f64,
    pub post_val: f64,
    pub ari: f64,
    pub rri: f64,
    pub r80p: f64,
    pub yryr: f64,
    pub y2r: f64,
}

impl Default for Recovery {
    fn default() -> Recovery {
        let nan = f64::NAN;
        Recovery {
            pre_val: nan,
            dist_val: nan,
            magnitude: nan,
            post_val: nan,
            ari: nan,
            rri: nan,
            r80p: nan,
            yryr: nan,
            y2r: nan,
        }
    }
}

impl Recovery {
    pub fn to_array(&self) -> [f64; N_METRICS] {
        [
            self.pre_val,
            self.dist_val,
            self.magnitude,
            self.post_val,
            self.ari,
            self.rri,
            self.r80p,
            self.yryr,
            self.y2r,
        ]
    }

    /// The recovery level of `p`'s basis (NaN without the pre and trough values).
    pub fn level(&self, p: &Params) -> f64 {
        match p.basis {
            Basis::PreValue => p.target * self.pre_val,
            Basis::Magnitude => self.dist_val + p.target * (self.pre_val - self.dist_val),
        }
    }
}

/// The metrics of the disturbance from `start` to `trough` (years) of a series with one value per year
/// (`years` increasing; a year may be missing, a value NaN).
pub fn metrics(years: &[i32], values: &[f64], start: i32, trough: i32, p: &Params) -> Recovery {
    let mut r = Recovery::default();
    if years.is_empty() || trough <= start || p.horizon < 1 {
        return r;
    }
    let at = |y: i32| match years.binary_search(&y) {
        Ok(i) => values[i],
        Err(_) => f64::NAN,
    };
    let s = if p.loss { 1.0 } else { -1.0 };

    let (mut sum, mut n) = (0.0, 0);
    for y in start - p.pre_years.max(1) + 1..=start {
        let v = at(y);
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    if n == 0 {
        return r;
    }
    r.pre_val = sum / n as f64;
    r.dist_val = at(trough);
    if !r.dist_val.is_finite() {
        return r;
    }
    r.magnitude = s * (r.pre_val - r.dist_val);

    // the series must reach the horizon for the metrics measured there
    let last = years[years.len() - 1];
    let end = trough + p.horizon;
    if end <= last {
        let best = ((trough + 1).max(end - 1)..=end)
            .map(at)
            .filter(|v| v.is_finite())
            .reduce(|a, b| if s * (b - a) > 0.0 { b } else { a });
        if let Some(post) = best {
            r.post_val = post;
            r.ari = s * (post - r.dist_val);
            if r.magnitude != 0.0 {
                r.rri = r.ari / r.magnitude;
            }
            let level = r.level(p);
            r.r80p = match p.basis {
                Basis::PreValue => post / level,
                Basis::Magnitude => (post - r.dist_val) / (level - r.dist_val),
            };
            if !r.r80p.is_finite() {
                r.r80p = f64::NAN;
            }
        }
        let v = at(end);
        if v.is_finite() {
            r.yryr = s * (v - r.dist_val) / f64::from(p.horizon);
        }
    }

    let level = r.level(p);
    let from = years.partition_point(|&y| y < trough);
    r.y2r = years[from..]
        .iter()
        .zip(&values[from..])
        .find(|&(_, &v)| v.is_finite() && s * (v - level) >= 0.0)
        .map_or(f64::NAN, |(&y, _)| f64::from(y - trough));
    r
}

/// The fitted trajectory of a segmentation at each of `years`: the line through its vertices (`count`
/// of `vertex_years` and `vertex_values`), NaN before the first vertex and after the last.
pub fn trajectory(vertex_years: &[f64], vertex_values: &[f64], count: usize, years: &[i32]) -> Vec<f64> {
    let count = count.min(vertex_years.len()).min(vertex_values.len());
    let (vx, vy) = (&vertex_years[..count], &vertex_values[..count]);
    years
        .iter()
        .map(|&y| {
            let y = f64::from(y);
            if count == 0 || y < vx[0] || y > vx[count - 1] {
                return f64::NAN;
            }
            let j = vx.partition_point(|&v| v <= y);
            if j == count {
                return vy[count - 1];
            }
            let (x0, x1) = (vx[j - 1], vx[j]);
            if x1 <= x0 {
                return vy[j - 1];
            }
            vy[j - 1] + (vy[j] - vy[j - 1]) * (y - x0) / (x1 - x0)
        })
        .collect()
}

/// `metrics` of every pixel of `values` (pixels x years), each with its disturbance's `starts` and
/// `troughs` (a start of 0, or a trough not after it: no disturbance, NaN): the metrics in `NAMES`'
/// order, (N_METRICS, pixels).
#[allow(clippy::too_many_arguments)]
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    years: &[i32],
    starts: &[i32],
    troughs: &[i32],
    p: &Params,
    n_jobs: i32,
) -> Vec<f64> {
    let t = years.len();
    assert_eq!(values.len(), n_pixels * t, "values must be n_pixels x years");
    assert!(
        starts.len() == n_pixels && troughs.len() == n_pixels,
        "one start and trough per pixel"
    );
    let mut per_pixel = vec![f64::NAN; n_pixels * N_METRICS];
    parallel::for_each_row(
        &mut per_pixel,
        N_METRICS,
        n_jobs,
        || (),
        |_, i, out| {
            if starts[i] == 0 || troughs[i] <= starts[i] {
                return;
            }
            let r = metrics(years, &values[i * t..(i + 1) * t], starts[i], troughs[i], p);
            out.copy_from_slice(&r.to_array());
        },
    );
    parallel::transpose(&per_pixel, n_pixels, N_METRICS)
}

/// `trajectory` of every pixel of a segmentation (vertices pixels x max_vertices, years then values;
/// `counts` per pixel): (pixels, years).
pub fn trajectory_batch(
    vertex_years: &[f64],
    vertex_values: &[f64],
    counts: &[i32],
    max_vertices: usize,
    years: &[i32],
    n_jobs: i32,
) -> Vec<f64> {
    let n_pixels = counts.len();
    let t = years.len();
    let mut out = vec![f64::NAN; n_pixels * t];
    parallel::for_each_row(
        &mut out,
        t,
        n_jobs,
        || (),
        |_, i, row| {
            let v = i * max_vertices..(i + 1) * max_vertices;
            let c = counts[i].max(0) as usize;
            row.copy_from_slice(&trajectory(&vertex_years[v.clone()], &vertex_values[v], c, years));
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    /// NBR-like: 0.6 before, a fire in 2005 down to 0.1, a linear regrowth of 0.05 a year.
    fn fire() -> (Vec<i32>, Vec<f64>) {
        let years: Vec<i32> = (2000..=2020).collect();
        let values = years
            .iter()
            .map(|&y| {
                if y < 2005 {
                    0.6
                } else {
                    (0.1 + 0.05 * f64::from(y - 2005)).min(0.6)
                }
            })
            .collect();
        (years, values)
    }

    #[test]
    fn fire_and_regrowth() {
        let (years, values) = fire();
        let r = metrics(&years, &values, 2004, 2005, &Params::default());
        assert!(close(r.pre_val, 0.6));
        assert!(close(r.dist_val, 0.1));
        assert!(close(r.magnitude, 0.5));
        // the best of 2009 and 2010
        assert!(close(r.post_val, 0.35));
        assert!(close(r.ari, 0.25));
        assert!(close(r.rri, 0.5));
        assert!(close(r.r80p, 0.35 / 0.48));
        assert!(close(r.yryr, 0.05));
        // 0.48 reached at 0.1 + 0.05 k >= 0.48: k = 8 (2013)
        assert_eq!(r.y2r, 8.0);
        // the magnitude basis: 0.1 + 0.8 * 0.5 = 0.5, reached at k = 8; r80p = 0.25 / 0.4
        let m = metrics(
            &years,
            &values,
            2004,
            2005,
            &Params {
                basis: Basis::Magnitude,
                ..Params::default()
            },
        );
        assert_eq!(m.y2r, 8.0);
        assert!(close(m.r80p, 0.25 / 0.4));
    }

    #[test]
    fn edges() {
        let (years, mut values) = fire();
        let p = Params::default();
        // not recovered by the end of the series
        let short = metrics(&years[..12], &values[..12], 2004, 2005, &p);
        assert!(short.y2r.is_nan() && short.post_val.is_finite());
        // the horizon beyond the series
        let late = metrics(&years, &values, 2017, 2018, &p);
        assert!(late.post_val.is_nan() && late.ari.is_nan() && late.yryr.is_nan());
        // a missing pre year: the mean of the other; a missing year 5: year 4 alone
        values[3] = f64::NAN;
        values[10] = f64::NAN;
        let r = metrics(&years, &values, 2004, 2005, &p);
        assert!(close(r.pre_val, 0.6));
        assert!(close(r.post_val, 0.3));
        assert!(r.yryr.is_nan());
        // no disturbance, no metrics
        assert!(metrics(&years, &values, 2005, 2005, &p).pre_val.is_nan());
        // never below the level: recovered at once
        let small: Vec<f64> = years.iter().map(|&y| if y == 2005 { 0.55 } else { 0.6 }).collect();
        assert_eq!(metrics(&years, &small, 2004, 2005, &p).y2r, 0.0);
    }

    #[test]
    fn gain_is_the_mirror_of_a_loss() {
        let (years, values) = fire();
        let p = Params {
            basis: Basis::Magnitude,
            ..Params::default()
        };
        let loss = metrics(&years, &values, 2004, 2005, &p);
        let flipped: Vec<f64> = values.iter().map(|v| -v).collect();
        let gain = metrics(&years, &flipped, 2004, 2005, &Params { loss: false, ..p });
        assert!(close(gain.magnitude, loss.magnitude));
        assert!(close(gain.ari, loss.ari));
        assert!(close(gain.rri, loss.rri));
        assert!(close(gain.r80p, loss.r80p));
        assert!(close(gain.yryr, loss.yryr));
        assert_eq!(gain.y2r, loss.y2r);
        assert!(close(gain.post_val, -loss.post_val));
    }

    #[test]
    fn trajectory_through_vertices() {
        let t = trajectory(
            &[2000.0, 2004.0, 2005.0, 2015.0, 0.0],
            &[0.6, 0.6, 0.1, 0.6, f64::NAN],
            4,
            &[1999, 2000, 2003, 2005, 2010, 2015, 2016],
        );
        assert!(t[0].is_nan() && t[6].is_nan());
        assert!(close(t[1], 0.6) && close(t[2], 0.6) && close(t[3], 0.1) && close(t[4], 0.35) && close(t[5], 0.6));
    }

    #[test]
    fn batch_is_metrics() {
        let (years, values) = fire();
        let t = years.len();
        let mut all = values.clone();
        all.extend(values.iter().map(|v| v * 2.0));
        all.extend(&values);
        let p = Params::default();
        let out = batch(&all, 3, &years, &[2004, 2004, 0], &[2005, 2005, 0], &p, 2);
        for i in 0..2 {
            let r = metrics(&years, &all[i * t..(i + 1) * t], 2004, 2005, &p).to_array();
            for k in 0..N_METRICS {
                assert!(out[k * 3 + i] == r[k] || (out[k * 3 + i].is_nan() && r[k].is_nan()));
            }
        }
        assert!((0..N_METRICS).all(|k| out[k * 3 + 2].is_nan()));
    }
}
