//! Polar phenology metrics: per pixel and year, the 39 metrics of the dates, lengths, values,
//! amplitudes, integrals and rates of the seasons of an interpolated series, found in polar
//! coordinates (the day of the year an angle, the value a radius).
//!
//! Ported from FORCE (Framework for Operational Radiometric Correction for Environmental
//! monitoring), Copyright (C) 2013-2025 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force, src/modules: higher-level/polar-hl.c, which implements
//! and extends Brooks, Lee, Pomara and Hargrove (2020), "Monitoring Broadscale Vegetational
//! Diversity and Change across North American Landscapes Using Land Surface Phenology", Forests
//! 11(6):606, doi 10.3390/f11060606.
//!
//! The method, as in FORCE. Each step `i` of the series, on the day of the year `doy_i` of a
//! calendar of 365 days, is the point at angle `r = doy_i / 365 * 2 pi` and radius the value
//! (negative values count as 0). The mean of all the points is the long-term mean vector `alpha`;
//! its diametric opposite `theta`, the day of least activity, is the start of the phenological
//! year. The seasons are the stretches from each start (the first step within one `step` after
//! `theta`, year after year); with `adaptive` each season's own mean vector moves its start (by
//! whole steps, towards the season's own `theta`). The values of each season are accumulated
//! and divided by the season's total: the fraction `cum`. Within a season:
//!
//! * LEFT: the first step with `cum > 0`; RIGHT: the step after the last with `cum == 1` (the
//!   first of the next season); START, MID, END: the first step with `cum >= start`, `mid`, `end`
//!   (SOS, MOS, EOS); PEAK: the first maximum of the value between START and END.
//! * the growing season, `start <= cum < end`, split into EARLY (`< mid`) and LATE: the mean vector
//!   of each (its angle a date, its length a value: DEV, DAV, DLV, VEV, VAV, VLV), the mean and
//!   deviation of the values of the growing season (VGM, VGV), the rates of rise and fall.
//! * the integrals above the green level `(START + END) / 2`, above and below the base level
//!   `(LEFT + RIGHT) / 2`, and of the rises and falls.
//!
//! The arithmetic is FORCE's where it decides a day or an event: the angle in f32 and the day of
//! the year truncated from it, the points, the sums and the accumulation in f32 and the
//! `cum == 1` test exact, the integrals in f64. The sines and cosines are those of f64 of that f32
//! angle. The series is read as it is (f32 or f64, converted to f32); FORCE reads 16-bit
//! integers, so on the same series the events and dates agree and the values agree to the
//! truncation FORCE applies.
//!
//! Differences from FORCE, all deliberate.
//!
//! * The dates (DEM, DLM, DPS, DSS, DMS, DES, DEV, DAV, DLV, DPY) count days from January 1st of
//!   the band's year (`y_out`, see below); FORCE counts them from January 1st of the first year of
//!   the series. A trend of a date along the years, as `zeit.linear_trend`, then needs no
//!   correction (FORCE subtracts 365 from the slope and 365 (n - 1) / 2 from the mean:
//!   `in_ce`). A date of a start in the previous year is negative, one in the next year past 365.
//!   The lengths (LTS, LGS, LGV) and DPV are differences, the same as FORCE's.
//! * Where FORCE converts something undefined to an integer (a NaN to short, in practice
//!   -32768 or 0), the result is NaN: a season without steps or whose total is 0 (all its
//!   metrics), a window with no step (the metrics of its mean vector, DEV, VEV, DAV, VAV, DLV, VLV,
//!   LGV, and the mean rate RAR, RAF), VGM with no step and VGV with fewer than 2, DPS and the
//!   events when no step has a value above 0 in the growing season. The maximum rates (RMR, RMF)
//!   are 0 for no rise or fall, as FORCE's. Nothing overflows a 16-bit integer, as the output is
//!   float and not truncated.
//! * DPV, the shift of the start of the year of each season, is 0 without `adaptive`; FORCE
//!   writes the date of the long-term start, negated (it subtracts a start never found).
//! * FORCE's STANDARDIZE_POL does nothing, and does not exist here.
//! * A series that does not start before the long-term start of its first year (it starts in the
//!   middle of the year, after the start of the phenological year, or a step grid steps over a
//!   start): FORCE looks for the start of year 0 for ever and finds no season at all, so every
//!   metric of the pixel is a NaN written as 0. Here a start with no step in its window (one day
//!   wider than FORCE's, as the days are truncated) is skipped, a season is numbered by the year its
//!   start is in, and the seasons whose start is in the series are measured; the others (the first,
//!   when the series begins after its start) are NaN. Where FORCE finds its seasons the result is
//!   FORCE's.
//!
//! The band of a season `s` is `y_out = s` if the long-term start is before day 182, `s + 1`
//! otherwise (the season ends in the year `y_out`): even with `adaptive`, from the long-term
//! start. With `ny` years there are `ny - 1` seasons, in the bands 0..ny-1 (`theta < 182`: the
//! last band is left NaN) or 1..ny (the first one is NaN).
//!
//! The days are in a calendar of 365 days, 29 February sharing 60 with 1 March (`ce = year * 365
//! + doy`, counted from the first year's start); the caller makes them. Years and days of the
//! year are those of `ce`: `doy` in 1..=365.

use super::Sample;
use crate::parallel;
use std::f64::consts::PI;

/// The number of metrics.
pub const N_METRICS: usize = 39;

/// The metrics, in FORCE's order: dates, lengths, values, amplitudes, the year's start, integrals and
/// rates.
pub const NAMES: [&str; N_METRICS] = [
    "DEM", "DLM", "DPS", "DSS", "DMS", "DES", "DEV", "DAV", "DLV", "LTS", "LGS", "LGV", "VEM", "VLM", "VPS", "VSS",
    "VMS", "VES", "VEV", "VAV", "VLV", "VBL", "VGA", "VSA", "VPA", "VGM", "VGV", "DPY", "DPV", "IST", "IBL", "IBT",
    "IGS", "IRR", "IFR", "RAR", "RAF", "RMR", "RMF",
];

// The index of each metric in `NAMES`.
const DEM: usize = 0;
const DLM: usize = 1;
const DPS: usize = 2;
const DSS: usize = 3;
const DMS: usize = 4;
const DES: usize = 5;
const DEV: usize = 6;
const DAV: usize = 7;
const DLV: usize = 8;
const LTS: usize = 9;
const LGS: usize = 10;
const LGV: usize = 11;
const VEM: usize = 12;
const VLM: usize = 13;
const VPS: usize = 14;
const VSS: usize = 15;
const VMS: usize = 16;
const VES: usize = 17;
const VEV: usize = 18;
const VAV: usize = 19;
const VLV: usize = 20;
const VBL: usize = 21;
const VGA: usize = 22;
const VSA: usize = 23;
const VPA: usize = 24;
const VGM: usize = 25;
const VGV: usize = 26;
const DPY: usize = 27;
const DPV: usize = 28;
const IST: usize = 29;
const IBL: usize = 30;
const IBT: usize = 31;
const IGS: usize = 32;
const IRR: usize = 33;
const IFR: usize = 34;
const RAR: usize = 35;
const RAF: usize = 36;
const RMR: usize = 37;
const RMF: usize = 38;

/// The index in `NAMES` of a metric's name (any case).
pub fn parse(name: &str) -> Result<usize, String> {
    let up = name.trim().to_ascii_uppercase();
    NAMES
        .iter()
        .position(|n| *n == up)
        .ok_or_else(|| format!("unknown polar metric {name:?}: use one of {}", NAMES.join(", ")))
}

/// The indices of the names of `names`.
pub fn parse_all<S: AsRef<str>>(names: &[S]) -> Result<Vec<usize>, String> {
    names.iter().map(|n| parse(n.as_ref())).collect()
}

/// The tables that depend only on the step: the same for every pixel.
struct Grid {
    ni: usize,
    ny: usize,
    ns: usize,
    step: i32,
    /// The day of each step (counted from the first year), as the float FORCE interpolates on.
    ce: Vec<f32>,
    /// FORCE's `polar.ce`: the year times 365 and the day of the year truncated from the angle.
    pce: Vec<i32>,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

impl Grid {
    fn new(ce: &[i64], step: i32) -> Result<Grid, String> {
        let ni = ce.len();
        if ni < 2 {
            return Err("the series needs at least 2 steps".into());
        }
        if step < 1 {
            return Err("step must be a positive number of days".into());
        }
        if ce[0] < 1 || ce[0] > 365 {
            return Err("ce must start in the first year: 1..=365".into());
        }
        if ce.windows(2).any(|w| w[1] < w[0]) {
            return Err("ce must not decrease".into());
        }
        if ce[ni - 1] > 365 * 20_000 {
            return Err("ce is not a count of days from the first year".into());
        }
        let mut g = Grid {
            ni,
            ny: 0,
            ns: 0,
            step,
            ce: Vec::with_capacity(ni),
            pce: Vec::with_capacity(ni),
            cos: Vec::with_capacity(ni),
            sin: Vec::with_capacity(ni),
        };
        let mut year = 0;
        for &c in ce {
            let doy = (c - 1).rem_euclid(365) + 1;
            year = ((c - doy) / 365) as i32;
            let r = (doy as f64 / 365.0 * 2.0 * PI) as f32;
            let d = (f64::from(r) * 365.0 / (2.0 * PI)) as f32;
            g.ce.push(c as f32);
            g.pce.push(year * 365 + d as i32);
            g.cos.push(f64::from(r).cos());
            g.sin.push(f64::from(r).sin());
        }
        g.ny = year as usize + 1;
        if g.ny < 2 {
            return Err("the series needs at least 2 years (one season is a start to the next start)".into());
        }
        g.ns = g.ny - 1;
        Ok(g)
    }
}

/// What the metrics asked for need, so that the rest is not computed.
struct Plan {
    /// The slots of the output row each metric is written to (a metric may be asked for twice).
    slots: Vec<Vec<usize>>,
    recurrence: bool,
    integrals: bool,
    thresholds: [f32; 3],
    adaptive: bool,
}

impl Plan {
    fn new(metrics: &[usize], thresholds: [f32; 3], adaptive: bool) -> Plan {
        let mut slots = vec![Vec::new(); N_METRICS];
        for (j, &k) in metrics.iter().enumerate() {
            slots[k].push(j);
        }
        let any = |ks: &[usize]| ks.iter().any(|&k| !slots[k].is_empty());
        Plan {
            recurrence: any(&[VGM, VGV]),
            integrals: any(&[IST, IBL, IBT, IGS, IRR, IFR]),
            slots,
            thresholds,
            adaptive,
        }
    }
}

/// The buffers of one thread.
#[derive(Default)]
struct Work {
    ok: Vec<bool>,
    next: Vec<u32>,
    val: Vec<f32>,
    pcx: Vec<f32>,
    pcy: Vec<f32>,
    cum: Vec<f32>,
    season: Vec<i32>,
    diff: Vec<i32>,
    theta0: Vec<i32>,
    /// Per year, whether a step was found in the window after that year's long-term start.
    covered: Vec<bool>,
}

/// A step kept as an event of a season: FORCE's `timing`.
#[derive(Clone, Copy, Default)]
struct Event {
    val: f32,
    cum: f32,
    ce: i32,
}

/// The mean vector of a window: its length, its day and the day as a date of the season.
#[derive(Clone, Copy)]
struct Vector {
    val: f32,
    ce: i32,
    ok: bool,
}

/// FORCE's `polar_vector`: angle in (0, 2 pi], length and the truncated day, from a point.
fn polar_vector(x: f32, y: f32) -> (f32, f32, i32) {
    let mut r = f64::from(y).atan2(f64::from(x)) as f32;
    if r <= 0.0 {
        r = (f64::from(r) + 2.0 * PI) as f32;
    }
    let v = f64::from(x * x + y * y).sqrt() as f32;
    let doy = (f64::from(r) * 365.0 / (2.0 * PI)) as f32;
    (r, v, doy as i32)
}

/// The diametric opposite of an angle.
fn opposite(rad: f32) -> f32 {
    if f64::from(rad) < PI {
        (f64::from(rad) + PI) as f32
    } else {
        (f64::from(rad) - PI) as f32
    }
}

/// FORCE's `ce_from_polar_vector`: the day of a mean vector's angle after the start of the year.
fn vector_ce(s: i32, theta_doy: i32, doy: i32) -> i32 {
    if doy > theta_doy {
        s * 365 + doy
    } else {
        (s + 1) * 365 + doy
    }
}

fn series_pixel<T: Sample>(g: &Grid, plan: &Plan, w: &mut Work, series: &[T], row: &mut [f64]) {
    let (ni, ny, ns) = (g.ni, g.ny, g.ns);
    let step = g.step;
    let fstep = step as f32;
    let [t_start, t_mid, t_end] = plan.thresholds;

    // The steps with a value, and those without by linear interpolation in the day between the
    // nearest with one on each side (the value of the one side beyond the first and last).
    w.ok.clear();
    w.val.clear();
    let mut any = false;
    let mut gaps = false;
    for x in series {
        let x = x.to_f64();
        let ok = x.is_finite();
        any |= ok;
        gaps |= !ok;
        w.ok.push(ok);
        w.val.push(if ok { x as f32 } else { 0.0 });
    }
    if !any {
        return;
    }
    if gaps {
        w.next.clear();
        w.next.resize(ni, u32::MAX);
        let mut nxt = u32::MAX;
        for i in (0..ni).rev() {
            w.next[i] = nxt;
            if w.ok[i] {
                nxt = i as u32;
            }
        }
        let mut last: Option<usize> = None;
        for i in 0..ni {
            if w.ok[i] {
                last = Some(i);
                continue;
            }
            let right = (w.next[i] != u32::MAX).then(|| w.next[i] as usize);
            let ce = g.ce[i];
            w.val[i] = match (last, right) {
                (Some(l), Some(r)) => {
                    let den = g.ce[r] - g.ce[l];
                    if den > 0.0 {
                        (w.val[l] * (g.ce[r] - ce) + w.val[r] * (ce - g.ce[l])) / den
                    } else {
                        w.val[l]
                    }
                }
                (Some(l), None) => w.val[l],
                (None, Some(r)) => w.val[r],
                (None, None) => unreachable!(),
            };
        }
    }

    // The points, and the long-term mean vector.
    w.pcx.clear();
    w.pcy.clear();
    let (mut sx, mut sy) = (0f32, 0f32);
    for i in 0..ni {
        let v = if w.val[i] < 0.0 { 0.0 } else { w.val[i] };
        w.val[i] = v;
        let x = (f64::from(v) * g.cos[i]) as f32;
        let y = (f64::from(v) * g.sin[i]) as f32;
        w.pcx.push(x);
        w.pcy.push(y);
        sx += x;
        sy += y;
    }
    let (alpha_rad, _, _) = polar_vector(sx / ni as f32, sy / ni as f32);
    let theta_rad = opposite(alpha_rad);
    let theta_doy = (f64::from(theta_rad) * 365.0 / (2.0 * PI)) as i32;

    // The seasons from the long-term start. A season is numbered by the year its start is in; a start
    // with no step in its window (the series begins after it, or the step grid steps over it) is
    // skipped, where FORCE would look for that start forever and find no season at all. The window is
    // one day wider than FORCE's (`step`): the days are truncated, so two steps can be `step + 1`
    // apart and miss a window of `step`; where FORCE finds a start, the first step in its window is
    // the same.
    w.season.clear();
    w.covered.clear();
    w.covered.resize(ny + 2, false);
    {
        let (mut s, mut y) = (-1i32, 0i32);
        for i in 0..ni {
            while g.pce[i] > y * 365 + theta_doy + step + 1 {
                y += 1;
            }
            let tce = y * 365 + theta_doy;
            if g.pce[i] >= tce && g.pce[i] - tce <= step + 1 {
                s = y;
                w.covered[y as usize] = true;
                y += 1;
            }
            w.season.push(s);
        }
    }

    // Each season's own start, as the steps it moves.
    w.theta0.clear();
    w.theta0.resize(ns, i32::MIN);
    if plan.adaptive {
        let mut i0 = 0;
        for s in 0..ns {
            let (mut mx, mut my, mut n) = (0f32, 0f32, 0f32);
            for i in i0..ni {
                let si = w.season[i];
                if si < s as i32 {
                    continue;
                }
                if si > s as i32 {
                    i0 = i;
                    break;
                }
                mx += w.pcx[i];
                my += w.pcy[i];
                n += 1.0;
            }
            if n == 0.0 {
                continue; // a season without steps has no start of its own
            }
            let (a_rad, _, _) = polar_vector(mx / n, my / n);
            let t_rad = opposite(a_rad);
            let opp = (f64::from(a_rad) - PI) as f32;
            let t_doy = (f64::from(t_rad) * 365.0 / (2.0 * PI)) as i32;
            let year = if opp < 0.0 && a_rad - theta_rad >= 0.0 {
                s as i32 - 1
            } else if opp > 0.0 && a_rad - theta_rad < 0.0 {
                s as i32 + 1
            } else {
                s as i32
            };
            w.theta0[s] = year * 365 + t_doy;
        }
        w.diff.clear();
        w.diff.resize(ni, 0);
        let mut s = w.season[0];
        for i in 1..ni {
            if w.season[i] != s {
                s = w.season[i];
                if s as usize >= ns {
                    break;
                }
                let t0 = w.theta0[s as usize];
                if t0 == i32::MIN {
                    continue;
                }
                let ce_shift = t0 - (s * 365 + theta_doy);
                let i_shift = ((ce_shift.abs() as f32) / fstep).floor() as usize;
                if ce_shift > 0 {
                    for ii in 0..i_shift {
                        if i + ii >= ni {
                            break;
                        }
                        w.diff[i + ii] = -1;
                    }
                } else {
                    for ii in 0..i_shift {
                        if i < ii + 1 {
                            break;
                        }
                        // the season the step is in becomes this one (the previous, or none yet)
                        w.diff[i - ii - 1] = s - w.season[i - ii - 1];
                    }
                }
            }
        }
        for i in 0..ni {
            w.season[i] += w.diff[i];
        }
    }

    // The accumulated fraction of each season, in f32 and in order, as FORCE.
    w.cum.clear();
    w.cum.resize(ni, 0.0);
    w.cum[0] = w.val[0];
    let mut s = w.season[0];
    for i in 1..ni {
        if w.season[i] != s {
            w.cum[i] = w.val[i];
            s = w.season[i];
        } else {
            w.cum[i] = w.cum[i - 1] + w.val[i];
        }
    }
    let mut sum = w.cum[ni - 1];
    let mut s = w.season[ni - 1];
    for i in (0..ni).rev() {
        if w.season[i] != s {
            sum = w.cum[i];
            s = w.season[i];
        }
        w.cum[i] /= sum;
    }

    let val = &w.val;
    let cum = &w.cum;
    let season = &w.season;
    let pce = &g.pce;
    let mut i0 = 0usize;
    for s in 0..ns {
        let si = s as i32;
        let y_out = if theta_doy < 182 { s } else { s + 1 };
        let theta_ce = si * 365 + theta_doy;
        let off = y_out as i32 * 365;
        if !w.covered[s] {
            continue; // the series begins after this season's start: nothing to measure
        }

        let (mut left, mut right, mut start, mut mid, mut peak, mut end) = (
            Event::default(),
            Event::default(),
            Event::default(),
            Event::default(),
            Event::default(),
            Event::default(),
        );
        let (mut wx, mut wy) = ([0f32; 3], [0f32; 3]); // grow, early, late
        let mut nw = [0i32; 3];
        let (mut max_rate, mut sum_rate) = ([0f32; 2], [0f32; 2]); // early, late
        let (mut rec_mean, mut rec_m2) = (0f64, 0f64);

        for i in i0..ni {
            if season[i] < si {
                continue;
            }
            if season[i] > si {
                break;
            }
            let c = cum[i];
            let ev = Event {
                val: val[i],
                cum: c,
                ce: pce[i],
            };
            if c > 0.0 && left.cum == 0.0 {
                left = ev;
            }
            if c == 1.0 {
                right = if i + 1 < ni {
                    Event {
                        val: val[i + 1],
                        cum: cum[i + 1],
                        ce: pce[i + 1],
                    }
                } else {
                    ev
                };
            }
            if c >= t_start && start.cum == 0.0 {
                start = ev;
            }
            if c >= t_mid && mid.cum == 0.0 {
                mid = ev;
            }
            if c >= t_end && end.cum == 0.0 {
                end = ev;
            }
            if c >= t_start && c < t_end {
                if plan.recurrence {
                    let n = f64::from(nw[0] + 1);
                    let x = f64::from(val[i]);
                    let old = rec_mean;
                    rec_mean = old + (x - old) / n;
                    rec_m2 += (x - old) * (x - rec_mean);
                }
                nw[0] += 1;
                wx[0] += w.pcx[i];
                wy[0] += w.pcy[i];
                if val[i] > peak.val {
                    peak = ev;
                }
            }
            if c >= t_start && c < t_mid {
                wx[1] += w.pcx[i];
                wy[1] += w.pcy[i];
                if i > 0 && val[i] - val[i - 1] > 0.0 {
                    let rate = (val[i] - val[i - 1]) / fstep;
                    sum_rate[0] += rate;
                    if rate > max_rate[0] {
                        max_rate[0] = rate;
                    }
                }
                nw[1] += 1;
            }
            if c >= t_mid && c < t_end {
                wx[2] += w.pcx[i];
                wy[2] += w.pcy[i];
                if i > 0 && val[i] - val[i - 1] < 0.0 {
                    let rate = (val[i - 1] - val[i]) / fstep;
                    sum_rate[1] += rate;
                    if rate > max_rate[1] {
                        max_rate[1] = rate;
                    }
                }
                nw[2] += 1;
            }
        }

        // The scan of the integrals below also moves the start of the next season's scan.
        let first = i0;
        for i in i0..ni {
            if season[i] < si {
                continue;
            }
            if season[i] > si {
                i0 = i;
                break;
            }
        }
        // A season without steps, or without a total (every fraction NaN): undefined.
        if left.cum == 0.0 || start.cum == 0.0 || mid.cum == 0.0 || end.cum == 0.0 {
            continue;
        }

        let mean_vector = |k: usize| {
            let n = nw[k] as f32;
            let (_, v, doy) = polar_vector(wx[k] / n, wy[k] / n);
            Vector {
                val: v,
                ce: vector_ce(si, theta_doy, doy),
                ok: nw[k] > 0,
            }
        };
        let (grow, early, late) = (mean_vector(0), mean_vector(1), mean_vector(2));
        let mean_rate = [sum_rate[0] / nw[1] as f32, sum_rate[1] / nw[2] as f32];
        let green = ((f64::from(start.val + end.val)) / 2.0) as f32;
        let base = ((f64::from(left.val + right.val)) / 2.0) as f32;

        let mut integral = [0f64; 6]; // green, seasonal, latent, total, rising, falling
        if plan.integrals {
            for i in first..ni {
                if season[i] < si {
                    continue;
                }
                if season[i] > si {
                    break;
                }
                let c = cum[i];
                let v = val[i];
                if c >= t_start && c < t_end && v > green {
                    integral[0] += f64::from((v - green) * fstep);
                }
                if v > base {
                    integral[1] += f64::from((v - base) * fstep);
                    integral[2] += f64::from(base * fstep);
                } else {
                    integral[2] += f64::from(v * fstep);
                }
                integral[3] += f64::from(v * fstep);
                if i > 0 && v - val[i - 1] > 0.0 {
                    integral[4] += f64::from((v - val[i - 1]) * fstep);
                }
                if i > 0 && v - val[i - 1] < 0.0 {
                    integral[5] += f64::from((val[i - 1] - v) * fstep);
                }
            }
        }

        let d = |ce: i32| f64::from(ce - off);
        let nan_if = |ok: bool, x: f64| if ok { x } else { f64::NAN };
        let mut put = |k: usize, x: f64| {
            for &j in &plan.slots[k] {
                row[j * ny + y_out] = x;
            }
        };
        put(DEM, d(left.ce));
        put(DLM, d(right.ce));
        put(DPS, nan_if(peak.cum != 0.0, d(peak.ce)));
        put(DSS, d(start.ce));
        put(DMS, d(mid.ce));
        put(DES, d(end.ce));
        put(DEV, nan_if(early.ok, d(early.ce)));
        put(DAV, nan_if(grow.ok, d(grow.ce)));
        put(DLV, nan_if(late.ok, d(late.ce)));
        put(LTS, f64::from(right.ce - left.ce));
        put(LGS, f64::from(end.ce - start.ce));
        put(LGV, nan_if(early.ok && late.ok, f64::from(late.ce - early.ce)));
        put(VEM, f64::from(left.val));
        put(VLM, f64::from(right.val));
        put(VPS, f64::from(peak.val));
        put(VSS, f64::from(start.val));
        put(VMS, f64::from(mid.val));
        put(VES, f64::from(end.val));
        put(VEV, nan_if(early.ok, f64::from(early.val)));
        put(VAV, nan_if(grow.ok, f64::from(grow.val)));
        put(VLV, nan_if(late.ok, f64::from(late.val)));
        put(VBL, f64::from(base));
        put(VGA, f64::from(peak.val - green));
        put(VSA, f64::from(peak.val - base));
        put(VPA, f64::from(peak.val - mid.val));
        put(VGM, nan_if(nw[0] > 0, rec_mean));
        put(VGV, nan_if(nw[0] > 1, (rec_m2 / f64::from(nw[0] - 1)).sqrt()));
        put(DPY, d(theta_ce));
        put(
            DPV,
            if plan.adaptive && w.theta0[s] != i32::MIN {
                f64::from(w.theta0[s] - theta_ce)
            } else {
                0.0
            },
        );
        put(IST, integral[1] / 365.0);
        put(IBL, integral[2] / 365.0);
        put(IBT, integral[3] / 365.0);
        put(IGS, integral[0] / 365.0);
        put(IRR, integral[4] / 30.0);
        put(IFR, integral[5] / 30.0);
        put(RAR, nan_if(nw[1] > 0, f64::from(mean_rate[0] * 30.0)));
        put(RAF, nan_if(nw[2] > 0, f64::from(mean_rate[1] * 30.0)));
        put(RMR, f64::from(max_rate[0] * 30.0));
        put(RMF, f64::from(max_rate[1] * 30.0));
    }
}

/// The polar metrics of every series of `values` (pixels, steps): (pixels, metrics, years), flat,
/// and the number of years.
///
/// `ce` (one per step, not decreasing) is the day of each step in a calendar of 365 days counted
/// from the first year: `year * 365 + doy` with the first year 0 and `doy` in 1..=365. `step` is
/// the nominal interval in days (FORCE's INT_DAY). `start`, `mid`, `end` are the fractions of
/// the season's accumulated value that mark its start, middle and end (FORCE's 0.2, 0.5, 0.8),
/// each in (0, 1). `metrics` are indices into `NAMES` (see `parse_all`). A NaN or infinite value
/// is a missing step (filled by linear interpolation in the day, constant beyond the first and
/// last); a series without a value is NaN. The band of a season is where the module's notes put
/// it; a band without a season is NaN.
#[allow(clippy::too_many_arguments)]
pub fn polar_batch<T: Sample>(
    values: &[T],
    n_pixels: usize,
    n_time: usize,
    ce: &[i64],
    step: i32,
    start: f64,
    mid: f64,
    end: f64,
    adaptive: bool,
    metrics: &[usize],
    n_jobs: i32,
) -> Result<(Vec<f64>, usize), String> {
    if values.len() != n_pixels * n_time {
        return Err("values must be n_pixels x n_time".into());
    }
    if ce.len() != n_time {
        return Err("one day (ce) per step".into());
    }
    if metrics.is_empty() || metrics.iter().any(|&k| k >= N_METRICS) {
        return Err("metrics must be indices of the 39 polar metrics".into());
    }
    for (name, t) in [("start", start), ("mid", mid), ("end", end)] {
        if !(t > 0.0 && t < 1.0) {
            return Err(format!("{name} must be between 0 and 1, got {t}"));
        }
    }
    let g = Grid::new(ce, step)?;
    let plan = Plan::new(metrics, [start as f32, mid as f32, end as f32], adaptive);
    let ny = g.ny;
    let width = metrics.len() * ny;
    let mut out = parallel::uninit_f64(n_pixels * width);
    parallel::for_each_row(&mut out, width, n_jobs, Work::default, |w, p, row| {
        row.fill(f64::NAN);
        series_pixel(&g, &plan, w, &values[p * n_time..(p + 1) * n_time], row);
    });
    Ok((out, ny))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: i64 = 5;
    const YEARS: usize = 4;

    /// The days of a series of `YEARS` years sampled every 5 days (doy 5, 10, ... 365).
    fn days() -> Vec<i64> {
        (0..YEARS as i64)
            .flat_map(|y| (1..=73).map(move |k| y * 365 + STEP * k))
            .collect()
    }

    /// A series that is `f(year, doy)`.
    fn series(f: impl Fn(usize, i64) -> f64) -> Vec<f64> {
        days()
            .iter()
            .map(|&c| f(((c - 1) / 365) as usize, (c - 1) % 365 + 1))
            .collect()
    }

    fn all(v: &[f64], adaptive: bool) -> (Vec<f64>, usize) {
        let m: Vec<usize> = (0..N_METRICS).collect();
        polar_batch(v, 1, v.len(), &days(), STEP as i32, 0.2, 0.5, 0.8, adaptive, &m, 1).unwrap()
    }

    /// The metric `name` of band `y`.
    fn at(out: &[f64], ny: usize, name: &str, y: usize) -> f64 {
        out[parse(name).unwrap() * ny + y]
    }

    fn is(out: &[f64], ny: usize, name: &str, y: usize, want: f64) {
        let got = at(out, ny, name, y);
        assert!(
            (got - want).abs() <= 1e-6 * (1.0 + want.abs()),
            "{name} of band {y}: {got} vs {want}"
        );
    }

    fn same(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() < 1e-9
    }

    /// A box of 100 on the 19 days 100, 105 ... 190 of every year.
    fn box_series() -> Vec<f64> {
        series(|_, d| if (100..=190).contains(&d) { 100.0 } else { 0.0 })
    }

    #[test]
    fn a_box_by_hand() {
        // The box is symmetric about day 145, so alpha is at 145 and theta at 327.5 -> 327. The
        // seasons start on day 330 (the first step within 5 days of 327) and end on the day before
        // 330 of the next year; the band is the year they end in, band 0 is NaN. Fractions of the 19
        // steps: START (>= 0.2) the 4th, MID (>= 0.5) the 10th, END (>= 0.8) the 16th.
        let (o, ny) = all(&box_series(), false);
        assert_eq!(ny, 4);
        for k in 0..N_METRICS {
            assert!(o[k * ny].is_nan(), "{} of band 0", NAMES[k]);
        }
        for y in 1..4 {
            for (name, want) in [
                ("DEM", 100.0),
                ("DSS", 115.0),
                ("DMS", 145.0),
                ("DPS", 115.0),
                ("DES", 175.0),
                ("DLM", 330.0),
                ("LTS", 230.0),
                ("LGS", 60.0),
                ("DPY", 327.0 - 365.0),
                ("DPV", 0.0),
                ("VEM", 100.0),
                ("VSS", 100.0),
                ("VMS", 100.0),
                ("VPS", 100.0),
                ("VES", 100.0),
                ("VLM", 0.0),
                ("VBL", 50.0),
                ("VSA", 50.0),
                ("VGA", 0.0),
                ("VPA", 0.0),
                ("VGM", 100.0),
                ("VGV", 0.0),
                ("IST", 4750.0 / 365.0),
                ("IBL", 4750.0 / 365.0),
                ("IBT", 9500.0 / 365.0),
                ("IGS", 0.0),
                ("IRR", 500.0 / 30.0),
                ("IFR", 500.0 / 30.0),
                ("RAR", 0.0),
                ("RAF", 0.0),
                ("RMR", 0.0),
                ("RMF", 0.0),
            ] {
                is(&o, ny, name, y, want);
            }
            // The growing season is the 12 steps of days 115..170: its mean vector points to the
            // middle (142.5, the day truncated from the angle: 141 or 142), the early part (days
            // 115..140, middle 127.5) and the late part (145..170, 157.5) to theirs; the length of
            // the mean of 12 unit vectors 5 days apart is sin(12 d / 2) / (12 sin(d / 2)), d = 2 pi 5 / 365.
            let delta = 2.0 * PI * 5.0 / 365.0;
            let want = 100.0 * (6.0 * delta).sin() / (12.0 * (delta / 2.0).sin());
            assert!(
                (at(&o, ny, "VAV", y) - want).abs() < 0.01,
                "{} vs {want}",
                at(&o, ny, "VAV", y)
            );
            let want = 100.0 * (3.0 * delta).sin() / (6.0 * (delta / 2.0).sin());
            for name in ["VEV", "VLV"] {
                assert!((at(&o, ny, name, y) - want).abs() < 0.01, "{name}");
            }
            for (name, mid) in [("DAV", 142.5), ("DEV", 127.5), ("DLV", 157.5)] {
                assert!(
                    (at(&o, ny, name, y) - mid).abs() <= 1.0,
                    "{name} {}",
                    at(&o, ny, name, y)
                );
            }
            assert!((at(&o, ny, "LGV", y) - 30.0).abs() <= 1.0);
        }
    }

    #[test]
    fn a_triangle_by_hand_with_rates() {
        // 10 20 30 40 50 40 30 20 10 on days 100..140: the sum 250; cum 0.04 .12 .24 .4 .6 .76
        // .88 .96 1. START the 3rd (day 110), MID the 5th (120), END the 7th (130), PEAK 50 at 120.
        // The growing season is the 3rd to the 6th (30 40 50 40).
        let v = series(|_, d| {
            if (100..=140).contains(&d) {
                let k = (d - 100) / 5;
                10.0 * (1 + k.min(8 - k)) as f64
            } else {
                0.0
            }
        });
        let (o, ny) = all(&v, false);
        for y in 1..4 {
            for (name, want) in [
                ("DSS", 110.0),
                ("DMS", 120.0),
                ("DES", 130.0),
                ("DPS", 120.0),
                ("DEM", 100.0),
                ("VSS", 30.0),
                ("VMS", 50.0),
                ("VES", 30.0),
                ("VPS", 50.0),
                ("VEM", 10.0),
                ("VLM", 0.0),
                ("VBL", 5.0),
                ("VSA", 45.0),
                ("VGA", 20.0),
                ("VPA", 0.0),
                ("VGM", 40.0),
                ("VGV", (200.0f64 / 3.0).sqrt()),
                // EARLY (cum in [0.2, 0.5)): the 3rd and 4th steps, both rising 10 in 5 days: rate 2
                ("RAR", 60.0),
                ("RMR", 60.0),
                // LATE (cum in [0.5, 0.8)): the 5th (a rise, not counted) and the 6th (a fall of 10):
                // the rates add up to 2 over both steps (FORCE's mean rate: 1), the maximum is 2
                ("RAF", 30.0),
                ("RMF", 60.0),
                // green level (30 + 30) / 2 = 30: the values above it in the growing season
                ("IGS", (10.0 + 20.0 + 10.0) * 5.0 / 365.0),
                ("IRR", 50.0 * 5.0 / 30.0),
                ("IFR", 50.0 * 5.0 / 30.0),
                ("IBT", 250.0 * 5.0 / 365.0),
            ] {
                is(&o, ny, name, y, want);
            }
        }
    }

    #[test]
    fn adaptive_moves_the_start_of_the_year() {
        // Years 0, 1 and 3 have the box on days 100-190, year 2 on 115-205: the long-term mean vector
        // is 3.73 days after 145, theta 331.2 -> 331, the seasons start on day 335.
        // Season 1 (band 2) has its own theta0 of day 342 (box centre 160 + 182.5), 11 days after
        // 331: two whole steps of 5, so the first two steps of it (days 335, 340 of year 1) go to
        // season 0. Season 0's theta0 is 327, 4 days before: no step.
        let v = series(|y, d| {
            let (a, b) = if y == 2 { (115, 205) } else { (100, 190) };
            if (a..=b).contains(&d) {
                100.0
            } else {
                0.0
            }
        });
        let (fixed, ny) = all(&v, false);
        let (adapt, _) = all(&v, true);
        // the box itself does not depend on the start of the year
        for y in 1..4 {
            for name in ["DEM", "DSS", "DMS", "DES", "DPS", "VBL", "IST", "DPY"] {
                is(&adapt, ny, name, y, at(&fixed, ny, name, y));
            }
            is(&fixed, ny, "DPY", y, 331.0 - 365.0);
        }
        is(&fixed, ny, "DSS", 1, 115.0);
        is(&fixed, ny, "DSS", 2, 130.0);
        is(&fixed, ny, "DES", 2, 190.0);
        // fixed: each season ends where the next starts (day 335); adaptive: season 0 runs 10 days
        // longer (the next one starts on day 345)
        is(&fixed, ny, "DLM", 1, 335.0);
        is(&adapt, ny, "DLM", 1, 345.0);
        is(&fixed, ny, "LTS", 1, 335.0 - 100.0);
        is(&adapt, ny, "LTS", 1, 345.0 - 100.0);
        is(&adapt, ny, "DLM", 2, 335.0);
        // DPV: theta0 less theta: 327 - 331, 342 - 331, 327 - 331; 0 without adaptive
        is(&adapt, ny, "DPV", 1, -4.0);
        is(&adapt, ny, "DPV", 2, 11.0);
        is(&adapt, ny, "DPV", 3, -4.0);
        for y in 1..4 {
            is(&fixed, ny, "DPV", y, 0.0);
        }
    }

    #[test]
    fn a_series_that_starts_after_the_first_start_of_the_year() {
        // The box's seasons start on day 330 (window 327..332). A series from day 335 of the first
        // year has no step in the window of year 0: the season of year 0 is not covered (NaN), the
        // others are as in the whole series (FORCE finds no season at all there).
        let (want, ny) = all(&box_series(), false);
        let (c, v): (Vec<i64>, Vec<f64>) = days().into_iter().zip(box_series()).filter(|(c, _)| *c >= 335).unzip();
        let m: Vec<usize> = (0..N_METRICS).collect();
        let (got, ny2) = polar_batch(&v, 1, v.len(), &c, STEP as i32, 0.2, 0.5, 0.8, false, &m, 1).unwrap();
        assert_eq!(ny, ny2);
        for k in 0..N_METRICS {
            assert!(got[k * ny + 1].is_nan(), "{} of band 1", NAMES[k]);
        }
        for k in 0..N_METRICS {
            for y in 2..4 {
                let (a, b) = (got[k * ny + y], want[k * ny + y]);
                assert!(same(a, b), "{} of band {y}: {a} vs {b}", NAMES[k]);
            }
        }
    }

    #[test]
    fn missing_steps_are_interpolated_and_empty_pixels_are_nan() {
        let base = box_series();
        let (want, _) = all(&base, false);
        // gaps among the zeros, at both ends, and two inside the box (linear between two 100s)
        let mut gap = base.clone();
        for i in [0usize, 3, 4, 75, 76, 200, 201, base.len() - 1, 100] {
            gap[i] = f64::NAN;
        }
        gap[101] = f64::INFINITY;
        let (got, _) = all(&gap, false);
        for (a, b) in got.iter().zip(&want) {
            assert!(same(*a, *b), "{a} vs {b}");
        }
        let (o, _) = all(&vec![f64::NAN; base.len()], true);
        assert!(o.iter().all(|x| x.is_nan()));
        // a series of zeros: every season sums to 0 and is undefined
        let (o, _) = all(&vec![0.0; base.len()], true);
        assert!(o.iter().all(|x| x.is_nan()));
        // negative values count as 0
        let neg: Vec<f64> = base.iter().map(|&x| if x == 0.0 { -50.0 } else { x }).collect();
        let (o, _) = all(&neg, false);
        for (a, b) in o.iter().zip(&want) {
            assert!(same(*a, *b), "{a} vs {b}");
        }
    }

    #[test]
    fn pixels_threads_dtype_and_the_metrics_asked_for() {
        let np = 17;
        let nt = days().len();
        let mut s = 11u64;
        let mut v32 = Vec::new();
        for p in 0..np {
            let shift = (p % 5) as f64 * 9.0;
            let one = series(|y, d| {
                let c = 150.0 + shift + y as f64 * 3.0;
                (60.0 + 90.0 * (-((d as f64 - c) / 40.0).powi(2)).exp()).floor()
            });
            for (i, x) in one.into_iter().enumerate() {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                v32.push(if (s >> 33) % 17 == 0 && i % 7 == 0 {
                    f32::NAN
                } else {
                    x as f32
                });
            }
        }
        let v64: Vec<f64> = v32.iter().map(|&x| f64::from(x)).collect();
        let m: Vec<usize> = (0..N_METRICS).collect();
        let run = |jobs, metrics: &[usize]| {
            polar_batch(&v32, np, nt, &days(), 5, 0.2, 0.5, 0.8, true, metrics, jobs)
                .unwrap()
                .0
        };
        let a = run(1, &m);
        for jobs in [3, -1] {
            let b = run(jobs, &m);
            assert!(a.iter().zip(&b).all(|(x, y)| same(*x, *y)));
        }
        let c = polar_batch(&v64, np, nt, &days(), 5, 0.2, 0.5, 0.8, true, &m, 2)
            .unwrap()
            .0;
        assert!(a.iter().zip(&c).all(|(x, y)| same(*x, *y)));
        // only some, in the order given, one twice
        let some = [parse("IGS").unwrap(), parse("DSS").unwrap(), parse("IGS").unwrap()];
        let d = run(1, &some);
        let ny = 4;
        for p in 0..np {
            for (j, &k) in some.iter().enumerate() {
                for y in 0..ny {
                    assert!(same(d[(p * 3 + j) * ny + y], a[(p * N_METRICS + k) * ny + y]));
                }
            }
        }
        // the seasons were found: a good part of the output is not NaN
        assert!(a.iter().filter(|x| x.is_finite()).count() > a.len() / 4);
    }

    #[test]
    fn names_and_bad_input() {
        assert_eq!(parse("vss").unwrap(), VSS);
        assert_eq!(parse(" RMF ").unwrap(), RMF);
        assert_eq!(NAMES[DPV], "DPV");
        assert_eq!(NAMES[IFR], "IFR");
        assert!(parse("XYZ").is_err());
        let v = box_series();
        let m = [VSS];
        let run = |ce: &[i64], step, t: (f64, f64, f64)| {
            polar_batch(&v, 1, v.len(), ce, step, t.0, t.1, t.2, true, &m, 1).map(|_| ())
        };
        let ok = days();
        assert!(run(&ok, 5, (0.2, 0.5, 0.8)).is_ok());
        assert!(run(&ok[..10], 5, (0.2, 0.5, 0.8)).is_err());
        assert!(run(&ok, 0, (0.2, 0.5, 0.8)).is_err());
        assert!(run(&ok, 5, (0.0, 0.5, 0.8)).is_err());
        assert!(run(&ok, 5, (0.2, 0.5, 1.0)).is_err());
        let mut back = ok.clone();
        back.swap(3, 4);
        assert!(run(&back, 5, (0.2, 0.5, 0.8)).is_err());
        // a single year has no season
        let one: Vec<i64> = (1..=73).map(|k| k * 5).collect();
        assert!(polar_batch(&v[..73], 1, 73, &one, 5, 0.2, 0.5, 0.8, true, &m, 1).is_err());
        assert!(polar_batch(&v, 1, v.len(), &ok, 5, 0.2, 0.5, 0.8, true, &[], 1).is_err());
    }
}
