//! BEAST, the Bayesian estimator of abrupt change, seasonal change and trend
//! (Zhao et al. 2019, *Remote Sensing of Environment* 232: 111181): a series
//! decomposed into a piecewise harmonic season and a piecewise polynomial
//! trend whose changepoints, numbers of changepoints and orders are all
//! sampled by reversible-jump MCMC, the models averaged over the posterior
//! instead of one chosen. What comes out is the probability of a changepoint
//! at each date, of each number of changepoints, the season and the trend
//! with their credible intervals, the trend's slope and the season's
//! amplitude.
//!
//! A port of Rbeast's C code (Kaiguang Zhao, GPL >= 2), version 1.0.2, its
//! `beast()` on one series with the generic (`cputype = 'sse'`) kernels: the
//! same PCG32 stream from the same seed, the same Gaussian and gamma
//! samplers, the same single-precision arithmetic in the same order, so the
//! same chain and the same posterior, number for number. Where Rbeast's
//! `beast123()` carries the stream, the precisions and the noise variance from
//! one pixel to the next, each series here starts from the seed as `beast()`
//! does, so a pixel's result does not depend on the others or on the threads.
//! The sources: [`rng`] (the stream and samplers), [`kern`] (the vector and
//! Cholesky kernels), [`basis`] (the season and trend bases and the moves),
//! [`model`] (the sampler), [`cpt`] (the changepoints from the probability
//! curve) and [`config`] (the options as Rbeast settles them).
//!
//! An irregular series is first aggregated to a regular one at `dt`, as
//! `beast.irreg()` does with numeric times ([`TimeAxis`]).

// The loops follow the C's, index for index, so that the port can be read
// against it.
#![allow(
    clippy::needless_range_loop,
    clippy::if_same_then_else,
    clippy::explicit_counter_loop,
    clippy::let_and_return,
    clippy::manual_div_ceil
)]

pub mod basis;
pub mod config;
pub mod cpt;
pub mod kern;
pub mod model;
pub mod rng;

pub use config::{Config, Options, PrecType};
pub use model::{Component, Fit, Workspace};

use crate::parallel;
use std::sync::Mutex;

/// How the values of a series are put on the regular grid BEAST fits.
#[derive(Clone, Debug)]
pub struct TimeAxis {
    /// The number of regular dates, the first and the step.
    pub n: usize,
    pub start: f32,
    pub dt: f32,
    mode: Mode,
}

#[derive(Clone, Debug)]
enum Mode {
    /// Already regular and ordered: the values as they are.
    AsIs,
    /// Each regular date the mean of the values in `counts` consecutive
    /// sorted times, from `first`.
    Aggregate {
        sorted: Vec<usize>,
        counts: Vec<usize>,
        first: usize,
    },
}

/// `IsRegularTS`: whether the sorted times are evenly spaced, and the step
/// (their first difference if so, else their mean difference).
fn is_regular(t: &[f64]) -> (bool, f32) {
    let n = t.len();
    if n < 2 {
        return (true, 1.0);
    }
    let est = (t[1] - t[0]) as f32;
    let mean = ((t[n - 1] - t[0]) / (n - 1) as f64) as f32;
    let eps = (mean as f64 * 1e-3) as f32;
    for i in 2..n {
        let d = (t[i] - t[i - 1]) as f32;
        if (d - est).abs() > eps {
            return (false, mean);
        }
    }
    (true, est)
}

fn round_i32(x: f32) -> i32 {
    (x as f64).round() as i32
}

impl TimeAxis {
    /// A regular series of `n` values from `start` every `dt`.
    pub fn regular(n: usize, start: f32, dt: f32) -> TimeAxis {
        TimeAxis {
            n,
            start,
            dt,
            mode: Mode::AsIs,
        }
    }

    /// The axis of values at numeric `times` (NaN times are dropped), on a
    /// grid of step `dt` from `start` (NaN: chosen as Rbeast chooses them).
    /// When the times are irregular the step is, for dates (`dates`: decimal
    /// years), the nearest of a day, half a month, a month and a week, else
    /// the period over the whole steps the mean spacing fits in it.
    /// `TimeVec_SortCheckRegularOrder`, `TimeVec_UpdateUserStartDt` and
    /// `tsAggegrationPrepare`, but a daily series stays in decimal years.
    pub fn irregular(times: &[f64], start: f32, dt: f32, period: f32, dates: bool) -> Result<TimeAxis, String> {
        let mut idx: Vec<usize> = (0..times.len()).filter(|&i| !times[i].is_nan()).collect();
        let nbad = times.len() - idx.len();
        if idx.len() < 2 {
            return Err("BEAST needs at least 2 times that are not NaN".into());
        }
        idx.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());
        let ordered = nbad == 0 && idx.iter().enumerate().all(|(i, &j)| i == j);
        let sorted: Vec<f64> = idx.iter().map(|&i| times[i]).collect();
        let ngood = sorted.len();
        let (regular, data_dt) = is_regular(&sorted);
        let data_start = sorted[0];

        let mut d = dt;
        if d.is_nan() {
            d = data_dt;
            if !regular {
                let freq = (1.0 / data_dt as f64) as f32;
                if dates && d as f64 > 0.5 / 366.0 {
                    let dt1 = ((freq - 365.0).abs() as f64 / 365.0) as f32;
                    let dt2 = ((freq - 24.0).abs() as f64 / 24.0) as f32;
                    let dt3 = ((freq - 12.0).abs() as f64 / 12.0) as f32;
                    let dt4 = ((freq as f64 - 365.0 / 7.0).abs() / 52.0) as f32;
                    let min = dt1.min(dt2).min(dt3.min(dt4));
                    if dt1 == min {
                        d = (1.0 / 365.0) as f32;
                    }
                    if dt2 == min {
                        d = (1.0 / 24.0) as f32;
                    }
                    if dt3 == min {
                        d = (1.0 / 12.0) as f32;
                    }
                    if dt4 == min {
                        d = (1.0 / 52.0) as f32;
                    }
                } else if period > 0.0 {
                    let freq = (period / data_dt).ceil();
                    d = period / freq;
                }
            }
        }
        if !(d > 0.0) {
            return Err(format!("the time step must be positive, got {d}"));
        }
        let t_start = sorted[0] as f32;
        let t_end = sorted[ngood - 1] as f32;
        let mut s = start;
        if !s.is_nan() && s >= t_end {
            s = data_start as f32;
        }
        if s.is_nan() {
            let i0 = round_i32(t_start / d);
            let est = i0 as f32 * d;
            s = if regular && ((d - data_dt) as f64).abs() < d as f64 * 1e-3 {
                data_start as f32
            } else {
                est
            };
        }
        if ((s as f64 - data_start).abs()) > d as f64 * 1e-2
            && s < t_end
            && ((s - t_start) as f64).abs() > d as f64 * 200.0
        {
            s = round_i32(s / d) as f32 * d;
        }

        let close = (data_start - s as f64).abs() < d as f64 * 1e-2 && ((data_dt - d) as f64).abs() < d as f64 * 1e-3;
        if regular && close {
            if ordered {
                return Ok(TimeAxis {
                    n: ngood,
                    start: s,
                    dt: d,
                    mode: Mode::AsIs,
                });
            }
            return Ok(TimeAxis {
                n: ngood,
                start: s,
                dt: d,
                mode: Mode::Aggregate {
                    sorted: idx,
                    counts: vec![1; ngood],
                    first: 0,
                },
            });
        }
        let i0 = round_i32(s / d);
        let i1 = round_i32(sorted[ngood - 1] as f32 / d);
        let nnew = i1 - i0 + 1;
        if nnew < 2 {
            return Err(format!("the times span fewer than 2 steps of {d}"));
        }
        let mut upper = (((i0 as f32) * d) as f64 + 0.5 * d as f64) as f32;
        let mut it = 0usize;
        while it < ngood && sorted[it] < (upper - d) as f64 {
            it += 1;
        }
        let first = it;
        let mut counts = vec![0usize; nnew as usize];
        for c in counts.iter_mut() {
            let mut k = 0;
            while it < ngood && sorted[it] as f32 <= upper {
                k += 1;
                it += 1;
            }
            *c = k;
            upper += d;
        }
        Ok(TimeAxis {
            n: nnew as usize,
            start: s,
            dt: d,
            mode: Mode::Aggregate {
                sorted: idx,
                counts,
                first,
            },
        })
    }

    /// `tsAggegrationPerform`: the series on the regular grid, each date the
    /// mean of its values (NaN when it has none).
    pub fn apply(&self, values: &[f32], out: &mut [f32]) {
        match &self.mode {
            Mode::AsIs => out[..self.n].copy_from_slice(&values[..self.n]),
            Mode::Aggregate { sorted, counts, first } => {
                let mut idx = *first;
                for (o, &c) in out.iter_mut().zip(counts) {
                    let mut s = 0f32;
                    let mut num = 0i32;
                    for _ in 0..c {
                        let y = values[sorted[idx]];
                        idx += 1;
                        if !y.is_nan() {
                            s += y;
                            num += 1;
                        }
                    }
                    *o = if num == 0 { f32::NAN } else { s / num as f32 };
                }
            }
        }
    }
}

/// The outputs of a batch, each `len` values per series, laid side by side
/// (the last, `data`, the series on the grid).
#[derive(Clone, Debug)]
pub struct Layout {
    pub fields: Vec<(String, usize, Vec<usize>)>,
    pub width: usize,
}

impl Layout {
    pub fn new(cfg: &Config) -> Layout {
        let n = cfg.n;
        let mut fields = Vec::new();
        let mut off = 0usize;
        let mut add = |name: &str, shape: Vec<usize>| {
            let len: usize = shape.iter().product();
            fields.push((name.to_string(), off, shape));
            off += len;
        };
        for s in ["marg_lik", "sig2", "r2", "rmse"] {
            add(s, vec![]);
        }
        let mut comp = |p: &str, maxk: usize, extra: &[&str], ci_extra: Option<&str>| {
            for s in ["ncp", "ncp_mode", "ncp_median", "ncp_pct90", "ncp_pct10"] {
                add(&format!("{p}_{s}"), vec![]);
            }
            add(&format!("{p}_ncp_prob"), vec![maxk + 1]);
            add(&format!("{p}_cp_occ"), vec![n]);
            add(&format!("{p}_order"), vec![n]);
            add(p, vec![n]);
            add(&format!("{p}_sd"), vec![n]);
            if cfg.ci {
                add(&format!("{p}_ci"), vec![2, n]);
            }
            add(&format!("{p}_cp"), vec![maxk]);
            add(&format!("{p}_cp_prob"), vec![maxk]);
            add(&format!("{p}_cp_abrupt"), vec![maxk]);
            add(&format!("{p}_cp_ci"), vec![2, maxk]);
            for e in extra {
                add(e, vec![n]);
            }
            if let (true, Some(c)) = (cfg.ci, ci_extra) {
                add(c, vec![2, n]);
            }
        };
        comp(
            "trend",
            cfg.tp.max_knot as usize,
            &[
                "trend_slope",
                "trend_slope_sd",
                "trend_slope_pos_prob",
                "trend_slope_zero_prob",
            ],
            Some("trend_slope_ci"),
        );
        if cfg.season {
            comp(
                "season",
                cfg.sp.max_knot as usize,
                &["season_amp", "season_amp_sd"],
                None,
            );
        }
        add("data", vec![n]);
        Layout { fields, width: off }
    }

    fn write(&self, fit: Option<&Fit>, row: &mut [f32]) {
        let Some(f) = fit else {
            row.fill(f32::NAN);
            return;
        };
        let mut off = 0usize;
        let mut put = |v: &[f32]| {
            row[off..off + v.len()].copy_from_slice(v);
            off += v.len();
        };
        put(&[f.marg_lik, f.sig2, f.r2, f.rmse]);
        let mut comp = |c: &Component, trend: bool| {
            put(&[c.ncp, c.ncp_mode, c.ncp_median, c.ncp_pct90, c.ncp_pct10]);
            put(&c.ncp_pr);
            put(&c.cp_occ);
            put(&c.order);
            put(&c.y);
            put(&c.sd);
            put(&c.ci);
            put(&c.cp);
            put(&c.cp_pr);
            put(&c.cp_abrupt);
            put(&c.cp_ci);
            put(&c.extra);
            put(&c.extra_sd);
            if trend {
                put(&c.slope_pos_pr);
                put(&c.slope_zero_pr);
                put(&c.slope_ci);
            }
        };
        comp(&f.trend, true);
        if let Some(s) = &f.season {
            comp(s, false);
        }
        debug_assert_eq!(off + f.data.len(), self.width);
    }
}

/// Fits one series already on its regular grid.
pub fn fit(values: &[f32], opts: &Options) -> Result<Option<Fit>, String> {
    let cfg = Config::new(opts, values.len())?;
    let mut ws = Workspace::new(&cfg);
    Ok(ws.fit(values))
}

/// Fits each of `pixels` series of `n_raw` values (row-major), put on the
/// grid of `axis` first, on the threads of `n_jobs`; every output of a
/// series in a row of `layout.width` values (NaN for a series that could not
/// be fitted).
pub fn batch(
    values: &[f32],
    n_raw: usize,
    axis: &TimeAxis,
    opts: &Options,
    n_jobs: i32,
) -> Result<(Layout, Vec<f32>), String> {
    let mut o = opts.clone();
    o.start = axis.start;
    o.dt = axis.dt;
    let cfg = Config::new(&o, axis.n)?;
    let layout = Layout::new(&cfg);
    let pixels = values.len().checked_div(n_raw).unwrap_or(0);
    let mut out = vec![0f32; pixels * layout.width];
    // rayon calls `init` for every task it steals, not once per thread: the
    // workspaces (a megabyte of buffers zeroed each) are leased from a pool
    // and given back, so that there are never more of them than threads.
    let pool = Mutex::new(Vec::new());
    parallel::for_each_row(
        &mut out,
        layout.width,
        n_jobs,
        || Lease {
            item: Some(
                pool.lock()
                    .unwrap()
                    .pop()
                    .unwrap_or_else(|| (Workspace::new(&cfg), vec![0f32; cfg.n])),
            ),
            pool: &pool,
        },
        |lease, i, row| {
            let (ws, buf) = lease.item.as_mut().unwrap();
            axis.apply(&values[i * n_raw..(i + 1) * n_raw], buf);
            let f = ws.fit(buf);
            layout.write(f.as_ref(), row);
            // The series on the grid, also where it could not be fitted.
            row[layout.width - cfg.n..].copy_from_slice(buf);
        },
    );
    Ok((layout, out))
}

type Scratch = (Workspace, Vec<f32>);

/// A workspace taken from the pool of `batch`, put back when dropped.
struct Lease<'a> {
    item: Option<Scratch>,
    pool: &'a Mutex<Vec<Scratch>>,
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.pool.lock().unwrap().push(item);
        }
    }
}
