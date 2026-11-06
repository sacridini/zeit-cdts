//! Division of a smoothed series into growing seasons (valley - peak - valley).
//!
//! The idea of two steps (a rough smooth good enough to find the extremes, then
//! the seasons built from them) follows Kong et al. 2022; the rules below are
//! Zeit's own (marked "Zeit's rule" where a number is a project choice), written
//! from the published description, without any third-party code.

/// A growing season, as indices into the series: `start < peak < end`. The end of
/// a season may be the start of the next one (the valley they share).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Season {
    pub start: usize,
    pub peak: usize,
    pub end: usize,
}

/// The fields of the settings that the division uses.
#[derive(Clone, Debug)]
pub struct SeasonSettings {
    /// Shortest season, `end - start + 1` observations.
    pub min_season_length: usize,
    /// Smallest season amplitude (peak minus the lower valley), as a fraction of
    /// the pixel's amplitude.
    pub min_amplitude: f64,
    /// Series with a smoothed amplitude below this have no seasons.
    pub min_pixel_amplitude: f64,
    /// A season whose start valley rises above the minimum by more than this
    /// fraction of the amplitude is dropped.
    pub rtrough_max: f64,
    /// Smallest swing between neighbouring extremes, as a fraction of the
    /// amplitude; smaller ones are merged away.
    pub r_min_filter: f64,
    /// Try once more with relaxed thresholds when nothing was found.
    pub season_retry: bool,
    /// Seasons kept per year (the largest by amplitude); 0 keeps all.
    pub max_seasons: usize,
}

impl Default for SeasonSettings {
    fn default() -> Self {
        SeasonSettings {
            min_season_length: 0,
            min_amplitude: 0.1,
            min_pixel_amplitude: 0.1,
            rtrough_max: 0.6,
            r_min_filter: 0.1,
            season_retry: true,
            max_seasons: 2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Ext {
    idx: usize,
    peak: bool,
}

/// Local extremes of `z`; a plateau counts once, at its middle. The first and
/// last values are never extremes here (the series ends are handled by the
/// head and tail rules).
fn local_extrema(z: &[f64]) -> Vec<Ext> {
    let n = z.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut s = 0;
    for i in 1..=n {
        if i == n || z[i] != z[s] {
            runs.push((s, i - 1));
            s = i;
        }
    }
    let mut out = Vec::new();
    for k in 1..runs.len().saturating_sub(1) {
        let v = z[runs[k].0];
        let (a, b) = (z[runs[k - 1].0], z[runs[k + 1].0]);
        let idx = (runs[k].0 + runs[k].1) / 2;
        if a < v && b < v {
            out.push(Ext { idx, peak: true });
        } else if a > v && b > v {
            out.push(Ext { idx, peak: false });
        }
    }
    out
}

/// Whether `a` is more extreme than `b` (the same kind).
fn more_extreme(z: &[f64], a: Ext, b: Ext) -> bool {
    if a.peak { z[a.idx] > z[b.idx] } else { z[a.idx] < z[b.idx] }
}

/// Makes the sequence alternate: of two neighbours of one kind the more extreme
/// stays (the earlier on a tie).
fn alternate(ext: &mut Vec<Ext>, z: &[f64]) {
    let mut out: Vec<Ext> = Vec::with_capacity(ext.len());
    for &e in ext.iter() {
        match out.last_mut() {
            Some(last) if last.peak == e.peak => {
                if more_extreme(z, e, *last) {
                    *last = e;
                }
            }
            _ => out.push(e),
        }
    }
    *ext = out;
}

/// Removes the swings smaller than `min_diff` (the smallest first: both of the
/// pair go and their neighbours merge) and merges extremes of one kind closer
/// than `min_dist` observations (the less extreme goes with the extreme between).
fn simplify(ext: &mut Vec<Ext>, z: &[f64], min_diff: f64, min_dist: f64) {
    loop {
        alternate(ext, z);
        let mut best: Option<(usize, f64)> = None;
        for k in 0..ext.len().saturating_sub(1) {
            let d = (z[ext[k].idx] - z[ext[k + 1].idx]).abs();
            if d < min_diff && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        if let Some((k, _)) = best {
            ext.drain(k..k + 2);
            continue;
        }
        let mut merged = false;
        for k in 0..ext.len().saturating_sub(2) {
            if ((ext[k + 2].idx - ext[k].idx) as f64) < min_dist {
                // k and k + 2 are of one kind; drop the middle and the less extreme
                let drop = if more_extreme(z, ext[k], ext[k + 2]) { k + 2 } else { k };
                let (a, b) = if drop == k { (k, k + 1) } else { (k + 1, k + 2) };
                ext.drain(a..=b);
                merged = true;
                break;
            }
        }
        if !merged {
            return;
        }
    }
}

/// Moves each extreme that lies in a run of more than `limit` unobserved points
/// to the nearest observed border of the run (Zeit's rule).
fn move_out_of_gaps(ext: &mut Vec<Ext>, observed: &[bool], limit: usize) {
    let n = observed.len();
    for e in ext.iter_mut() {
        if observed[e.idx] {
            continue;
        }
        let mut a = e.idx;
        while a > 0 && !observed[a - 1] {
            a -= 1;
        }
        let mut b = e.idx;
        while b + 1 < n && !observed[b + 1] {
            b += 1;
        }
        if b - a < limit {
            continue;
        }
        let before = a.checked_sub(1);
        let after = (b + 1 < n).then_some(b + 1);
        e.idx = match (before, after) {
            (Some(p), Some(q)) => {
                if e.idx - p <= q - e.idx { p } else { q }
            }
            (Some(p), None) => p,
            (None, Some(q)) => q,
            (None, None) => e.idx,
        };
    }
    ext.sort_by_key(|e| e.idx);
    ext.dedup_by_key(|e| e.idx);
}

struct Ctx<'a> {
    z: &'a [f64],
    zmin: f64,
    amp: f64,
    npt: usize,
    min_len: usize,
    min_amp: f64,
    rtrough: f64,
}

struct Cand {
    season: Season,
    amplitude: f64,
    ok: bool,
    /// Position of the peak in the extremes.
    k: usize,
}

/// The seasons the extremes make, each checked against the rules of step 4 and
/// the head and tail rules of step 5.
fn candidates(ext: &[Ext], c: &Ctx) -> Vec<Cand> {
    let n = c.z.len();
    let near_valley = 0.25 * c.amp; // Zeit's rule: an end this close to the minimum may start a season
    let half_rise = c.npt as f64 / 12.0; // half of nptperyear / 6
    let mut out = Vec::new();
    for (k, e) in ext.iter().enumerate() {
        if !e.peak {
            continue;
        }
        let mut ok = true;
        let start = if k > 0 {
            ext[k - 1].idx
        } else {
            if c.z[0] - c.zmin > near_valley || ((e.idx) as f64) < half_rise {
                ok = false;
            }
            0
        };
        let end = if k + 1 < ext.len() {
            ext[k + 1].idx
        } else {
            if c.z[n - 1] - c.zmin > near_valley || ((n - 1 - e.idx) as f64) < half_rise {
                ok = false;
            }
            n - 1
        };
        let amplitude = c.z[e.idx] - c.z[start].min(c.z[end]);
        // the length is not checked here: a season too short is dropped at the
        // end, not merged with its neighbours (merging seasons that are each too
        // short would make one of several years)
        if amplitude < c.min_amp * c.amp
            || c.z[start] - c.zmin > c.rtrough * c.amp
        {
            ok = false;
        }
        out.push(Cand { season: Season { start, peak: e.idx, end }, amplitude, ok, k });
    }
    out
}

/// One attempt of steps 1 to 5 with the given thresholds.
fn attempt(z: &[f64], t: &[f64], observed: Option<&[bool]>, nptperyear: usize, c: &Ctx, rmin: f64, max_seasons: usize) -> Vec<Season> {
    let mut ext = local_extrema(z);
    if let Some(obs) = observed {
        move_out_of_gaps(&mut ext, obs, nptperyear / 2);
    }
    simplify(&mut ext, z, rmin * c.amp, nptperyear as f64 / 6.0);
    // Drop the invalid seasons one at a time, the weakest first: their peak goes
    // and the valleys on both sides merge (the lower stays).
    loop {
        let cands = candidates(&ext, c);
        let bad = cands.iter().filter(|s| !s.ok).min_by(|a, b| a.amplitude.total_cmp(&b.amplitude));
        let Some(bad) = bad else {
            let mut seasons: Vec<(Season, f64)> = cands
                .iter()
                .filter(|s| s.season.end - s.season.start + 1 >= c.min_len)
                .map(|s| (s.season, s.amplitude))
                .collect();
            let years = ((t[t.len() - 1] - t[0]) / 365.25).ceil().max(1.0) as usize;
            if max_seasons > 0 && seasons.len() > max_seasons * years {
                seasons.sort_by(|a, b| b.1.total_cmp(&a.1));
                seasons.truncate(max_seasons * years);
                seasons.sort_by_key(|s| s.0.peak);
            }
            return seasons.into_iter().map(|s| s.0).collect();
        };
        ext.remove(bad.k);
        alternate(&mut ext, z);
    }
}

/// The growing seasons of the smoothed series `z` at the dates `t` (days).
/// `observed` marks the points that had data (None: all); a peak or valley that
/// falls in a gap longer than `nptperyear / 2` is moved to the border of the
/// data. Steps: extremes of `z`, alternation, removal of swings smaller than
/// `r_min_filter` of the amplitude, building of valley - peak - valley seasons,
/// dropping of the short, weak or high-valley ones (a series that starts or ends
/// near its minimum may give a truncated season at the head or tail), and the
/// cap of `max_seasons` per year. With nothing found and `season_retry`, once
/// more with `r_min_filter` and `min_amplitude` halved and `rtrough_max` times
/// 1.3 (at most 1). Empty when the amplitude is below `min_pixel_amplitude`,
/// when the series is constant, when fewer than 3 points are finite, or when
/// the dates do not increase.
pub fn split_growing_seasons(
    z: &[f64],
    t: &[f64],
    settings: &SeasonSettings,
    nptperyear: usize,
    observed: Option<&[bool]>,
) -> Vec<Season> {
    let n = z.len();
    if n < 3 || t.len() != n || observed.is_some_and(|o| o.len() != n) {
        return Vec::new();
    }
    if !t.iter().all(|v| v.is_finite()) || !t.windows(2).all(|p| p[1] > p[0]) {
        return Vec::new();
    }
    let z = fill_gaps(z);
    let Some(z) = z else { return Vec::new() };
    let zmin = z.iter().cloned().fold(f64::INFINITY, f64::min);
    let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let amp = zmax - zmin;
    let mean = z.iter().sum::<f64>() / n as f64;
    if amp <= 1e-12 * mean.abs().max(1.0) || amp < settings.min_pixel_amplitude {
        return Vec::new();
    }
    let npt = nptperyear.max(1);
    let mut ctx = Ctx {
        z: &z,
        zmin,
        amp,
        npt,
        min_len: settings.min_season_length,
        min_amp: settings.min_amplitude,
        rtrough: settings.rtrough_max,
    };
    let found = attempt(&z, t, observed, npt, &ctx, settings.r_min_filter, settings.max_seasons);
    if !found.is_empty() || !settings.season_retry {
        return found;
    }
    ctx.min_amp = settings.min_amplitude / 2.0;
    ctx.rtrough = (settings.rtrough_max * 1.3).min(1.0);
    attempt(&z, t, observed, npt, &ctx, settings.r_min_filter / 2.0, settings.max_seasons)
}

/// `z` with the non-finite values filled by linear interpolation (the ends by
/// the nearest value); None when fewer than 3 are finite.
fn fill_gaps(z: &[f64]) -> Option<Vec<f64>> {
    let good: Vec<usize> = (0..z.len()).filter(|&i| z[i].is_finite()).collect();
    if good.len() < 3 {
        return None;
    }
    let mut out = z.to_vec();
    for i in 0..z.len() {
        if z[i].is_finite() {
            continue;
        }
        let p = good.partition_point(|&g| g < i);
        out[i] = match (p.checked_sub(1).map(|k| good[k]), good.get(p).copied()) {
            (Some(a), Some(b)) => z[a] + (z[b] - z[a]) * (i - a) as f64 / (b - a) as f64,
            (Some(a), None) => z[a],
            (None, Some(b)) => z[b],
            (None, None) => unreachable!(),
        };
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenology::smooth::whittaker_weighted;
    use crate::phenology::testutil::{dates16, Rng};

    const TAU: f64 = std::f64::consts::TAU;

    fn noisy(years: usize, seed: u64, sigma: f64, f: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
        let t = dates16(years);
        let mut rng = Rng::new(seed);
        let y = t.iter().map(|&d| f(d) + sigma * rng.normal()).collect();
        (t, y)
    }

    fn smooth(y: &[f64]) -> Vec<f64> {
        whittaker_weighted(y, &vec![1.0; y.len()], 15.0).unwrap()
    }

    fn annual(d: f64) -> f64 {
        0.5 + 0.3 * (TAU * (d - 80.0) / 365.25).sin()
    }

    #[test]
    fn one_season_per_year() {
        let (t, y) = noisy(3, 1, 0.02, annual);
        let s = split_growing_seasons(&smooth(&y), &t, &SeasonSettings::default(), 23, None);
        assert!((2..=4).contains(&s.len()), "{s:?}");
        for x in &s {
            assert!(x.start < x.peak && x.peak < x.end);
        }
        // peaks of the sinusoid: 80 + 91.3 + 365.25 k days
        for x in &s {
            let day = t[x.peak];
            let k = ((day - 171.3) / 365.25).round();
            assert!((day - (171.3 + 365.25 * k)).abs() <= 24.0, "peak at day {day}");
        }
    }

    #[test]
    fn two_seasons_per_year() {
        let f = |d: f64| 0.5 + 0.25 * (2.0 * TAU * (d - 80.0) / 365.25).sin();
        let (t, y) = noisy(3, 2, 0.02, f);
        let s = split_growing_seasons(&smooth(&y), &t, &SeasonSettings::default(), 23, None);
        assert!((5..=6).contains(&s.len()), "{s:?}");
        for x in &s {
            let day = t[x.peak];
            let k = ((day - 125.6) / 182.625).round();
            assert!((day - (125.6 + 182.625 * k)).abs() <= 16.0, "peak at day {day}");
        }
        // the shared valley convention: an end may equal the next start
        for p in s.windows(2) {
            assert!(p[0].end <= p[1].start);
        }
    }

    /// The main wave plus a narrow bump of 0.08 at each trough, which makes a
    /// weak second season with amplitude about 0.05.
    fn with_weak_season(d: f64) -> f64 {
        let trough = (d - 80.0 + 91.3) / 365.25;
        let tc = (trough.round() * 365.25) - 91.3 + 80.0;
        annual(d) + 0.08 * (-((d - tc) / 15.0).powi(2) / 2.0).exp()
    }

    #[test]
    fn weak_season_dropped_by_min_amplitude() {
        let t = dates16(3);
        let z: Vec<f64> = t.iter().map(|&d| with_weak_season(d)).collect();
        let strict = SeasonSettings { min_amplitude: 0.2, ..SeasonSettings::default() };
        let one = split_growing_seasons(&z, &t, &strict, 23, None);
        let loose = SeasonSettings { min_amplitude: 0.05, r_min_filter: 0.02, ..SeasonSettings::default() };
        let two = split_growing_seasons(&z, &t, &loose, 23, None);
        assert!(two.len() > one.len(), "{one:?} vs {two:?}");
        assert!((2..=4).contains(&one.len()), "{one:?}");
        assert!((4..=7).contains(&two.len()), "{two:?}");
    }

    #[test]
    fn no_seasonality_no_seasons() {
        let mut rng = Rng::new(9);
        let t = dates16(3);
        let y: Vec<f64> = t.iter().map(|_| 0.5 + 0.01 * rng.normal()).collect();
        let s = SeasonSettings { min_pixel_amplitude: 0.05, ..SeasonSettings::default() };
        assert!(split_growing_seasons(&smooth(&y), &t, &s, 23, None).is_empty());
        let drift: Vec<f64> = (0..69).map(|i| 0.3 + 0.004 * i as f64).collect();
        assert!(split_growing_seasons(&drift, &t, &s, 23, None).is_empty());
    }

    #[test]
    fn retry_relaxes_the_thresholds() {
        // A season's amplitude cannot exceed the pixel's, so min_amplitude 1.5
        // drops them all; halved by the retry (0.75) they stay.
        let t = dates16(3);
        let z: Vec<f64> = t.iter().map(|&d| 0.5 + 0.06 * (TAU * (d - 80.0) / 365.25).sin()).collect();
        let base = SeasonSettings {
            min_pixel_amplitude: 0.05,
            min_amplitude: 1.5,
            season_retry: false,
            ..SeasonSettings::default()
        };
        assert!(split_growing_seasons(&z, &t, &base, 23, None).is_empty());
        let retry = SeasonSettings { season_retry: true, ..base };
        assert!(!split_growing_seasons(&z, &t, &retry, 23, None).is_empty());
    }

    #[test]
    fn special_cases() {
        let t = dates16(3);
        let s = SeasonSettings::default();
        assert!(split_growing_seasons(&[], &[], &s, 23, None).is_empty());
        assert!(split_growing_seasons(&[1.0, 2.0], &[0.0, 1.0], &s, 23, None).is_empty());
        assert!(split_growing_seasons(&vec![f64::NAN; 69], &t, &s, 23, None).is_empty());
        assert!(split_growing_seasons(&vec![0.4; 69], &t, &s, 23, None).is_empty());
        // dates not increasing
        let z: Vec<f64> = t.iter().map(|&d| annual(d)).collect();
        let mut bad = t.clone();
        bad.swap(10, 11);
        assert!(split_growing_seasons(&z, &bad, &s, 23, None).is_empty());
        // length mismatch
        assert!(split_growing_seasons(&z, &t[..60], &s, 23, None).is_empty());
        // a few NaN in z are interpolated
        let mut zn = z.clone();
        zn[30] = f64::NAN;
        assert!(!split_growing_seasons(&zn, &t, &s, 23, None).is_empty());
    }

    #[test]
    fn min_season_length_and_cap() {
        let t = dates16(3);
        let z: Vec<f64> = t.iter().map(|&d| annual(d)).collect();
        let long = SeasonSettings { min_season_length: 60, ..SeasonSettings::default() };
        assert!(split_growing_seasons(&z, &t, &long, 23, None).is_empty());
        // at most max_seasons per year
        let f = |d: f64| 0.5 + 0.25 * (3.0 * TAU * (d - 80.0) / 365.25).sin();
        let z3: Vec<f64> = t.iter().map(|&d| f(d)).collect();
        let cap = SeasonSettings { max_seasons: 1, ..SeasonSettings::default() };
        assert!(split_growing_seasons(&z3, &t, &cap, 23, None).len() <= 3);
    }

    #[test]
    fn peak_not_inside_a_long_gap() {
        let t = dates16(3);
        let mut z: Vec<f64> = t.iter().map(|&d| annual(d)).collect();
        let mut obs = vec![true; 69];
        // the second year's peak (about index 33) lies in a gap of 16 points
        for i in 26..42 {
            obs[i] = false;
            z[i] = 0.5 + 0.3 * (TAU * (t[i] - 80.0) / 365.25).sin();
        }
        let s = split_growing_seasons(&z, &t, &SeasonSettings::default(), 23, Some(&obs));
        assert!(!s.is_empty());
        assert!(s.iter().all(|x| obs[x.peak]), "{s:?}");
    }
}
