//! Best available pixel (BAP) composites: of a period's dates, each pixel takes the observation with the highest
//! score, the score a weighted sum of what makes an observation good, in the manner of the parametric compositing
//! of Griffiths et al. (2013, IEEE JSTARS 6(5)) and White et al. (2014, Can. J. Remote Sens. 40(3)):
//!
//! - its day of the year near a target day: `exp(-0.5 (d / sigma)^2)`, `d` the days between them around the year;
//! - its distance to the nearest cloud or shadow of its date: 0 at `cloud_min` pixels or nearer, 1 at `cloud_max`
//!   or farther, linear between (the exact Euclidean distance, Felzenszwalb and Huttenlocher's transform);
//! - its sensor (a score of the date's, e.g. 0.5 for Landsat 7 after its scan line corrector failed);
//! - its scene's cover (a score of the date's, e.g. 1 - the scene's cloud cover).
//!
//! Only the quality of the observations enters the score, not their values: the dates that win are known before a
//! band is read, so a composite reads the bands of the winners only.
//!
//! Each pixel of each date is one of three states: no data (outside the scene, a fill, a defective pixel: neither a
//! candidate nor a cloud), cloud (a cloud or its shadow: not a candidate, and the others are scored by their
//! distance to it) or clear (a candidate).

use rayon::prelude::*;

use crate::parallel;

/// No observation: outside the scene, a fill, a defective or saturated pixel.
pub const NO_DATA: u8 = 0;
/// A cloud, a cloud's shadow, cirrus: not a candidate; the clear pixels near it score less.
pub const CLOUD: u8 = 1;
/// A clear observation: a candidate.
pub const CLEAR: u8 = 2;

/// The score's settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// the target day of the year (1 to 366)
    pub target_doy: f64,
    /// the days at which the day's score falls to exp(-0.5) (about 0.61)
    pub doy_sigma: f64,
    /// pixels: the distance to a cloud at which the cloud's score starts to rise from 0
    pub cloud_min: f64,
    /// pixels: the distance to a cloud from which the cloud's score is 1 (a date without clouds scores 1)
    pub cloud_max: f64,
    pub w_doy: f64,
    pub w_cloud: f64,
    pub w_sensor: f64,
    pub w_cover: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            target_doy: 196.0,
            doy_sigma: 30.0,
            cloud_min: 1.0,
            cloud_max: 50.0,
            w_doy: 0.3,
            w_cloud: 0.5,
            w_sensor: 0.1,
            w_cover: 0.1,
        }
    }
}

impl Options {
    /// The settings make sense: a positive sigma, cloud_min < cloud_max, weights not negative and not all 0.
    pub fn check(&self) -> Result<(), String> {
        if !(self.target_doy >= 1.0 && self.target_doy <= 366.0) {
            return Err(format!(
                "target_doy {} is not a day of the year (1 to 366)",
                self.target_doy
            ));
        }
        if !(self.doy_sigma > 0.0) {
            return Err("doy_sigma must be > 0".into());
        }
        if !(self.cloud_min >= 0.0 && self.cloud_max > self.cloud_min) {
            return Err(format!(
                "cloud distances {} to {}: 0 <= min < max",
                self.cloud_min, self.cloud_max
            ));
        }
        let w = [self.w_doy, self.w_cloud, self.w_sensor, self.w_cover];
        if w.iter().any(|&w| !(w >= 0.0)) || w.iter().sum::<f64>() <= 0.0 {
            return Err("the weights must be >= 0, one at least > 0".into());
        }
        Ok(())
    }

    /// The day's score of day of the year `doy`.
    pub fn doy_score(&self, doy: f64) -> f64 {
        let d = (doy - self.target_doy).abs() % 365.25;
        let d = d.min(365.25 - d);
        (-0.5 * (d / self.doy_sigma).powi(2)).exp()
    }

    /// The cloud's score of a distance (pixels) to the nearest cloud.
    pub fn cloud_score(&self, dist: f64) -> f64 {
        ((dist - self.cloud_min) / (self.cloud_max - self.cloud_min)).clamp(0.0, 1.0)
    }

    /// The pixels around a window whose clouds can change a score in it: the cloud distance's cap.
    pub fn margin(&self) -> usize {
        self.cloud_max.ceil() as usize
    }
}

/// What is known of a date beyond its pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Date {
    /// its day of the year (1 to 366)
    pub doy: f64,
    /// its sensor's score (0 to 1)
    pub sensor: f64,
    /// its scene's score (0 to 1), e.g. 1 - cloud cover
    pub cover: f64,
}

/// 1D squared distance transform of `f` (Felzenszwalb and Huttenlocher 2012, "Distance transforms of sampled
/// functions"): `d[q] = min_p (q - p)^2 + f[p]`.
fn dt_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let cross =
        |q: usize, p: usize| ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * (q as f64 - p as f64));
    for q in 1..n {
        // z[0] is -inf: the loop stops at k = 0 at the latest
        let mut s = cross(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = cross(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, dq) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        *dq = (q as f64 - p as f64).powi(2) + f[p];
    }
}

/// A squared distance larger than any of a window (60000 pixels a side: 7.2e9), small enough for the squares of the
/// pixels' places to keep their digits beside it: what "no cloud" stands for in the transform.
const FAR: f64 = 1e12;

/// The Euclidean distance (pixels) of each pixel of a `rows` x `cols` window to the nearest pixel where `is_cloud`
/// holds, at most `cap` (`cap` where none is that near); 0 on the clouds. As scipy.ndimage.distance_transform_edt of
/// the mask of the non-clouds.
pub fn distance_to(is_cloud: impl Fn(usize) -> bool, rows: usize, cols: usize, cap: f64) -> Vec<f32> {
    let n = rows * cols;
    let mut g = vec![FAR; n];
    let mut any = false;
    for (i, g) in g.iter_mut().enumerate() {
        if is_cloud(i) {
            *g = 0.0;
            any = true;
        }
    }
    if !any {
        return vec![cap as f32; n];
    }
    let m = rows.max(cols);
    let (mut f, mut d, mut v, mut z) = (vec![0.0; m], vec![0.0; m], vec![0usize; m], vec![0.0; m + 1]);
    // down the columns, then along the rows
    for x in 0..cols {
        for y in 0..rows {
            f[y] = g[y * cols + x];
        }
        dt_1d(&f[..rows], &mut d[..rows], &mut v, &mut z);
        for y in 0..rows {
            g[y * cols + x] = d[y].min(FAR);
        }
    }
    let cap2 = cap * cap;
    let mut out = vec![0f32; n];
    for y in 0..rows {
        let row = &g[y * cols..(y + 1) * cols];
        dt_1d(row, &mut d[..cols], &mut v, &mut z);
        for x in 0..cols {
            out[y * cols + x] = if d[x] >= cap2 { cap as f32 } else { d[x].sqrt() as f32 };
        }
    }
    out
}

/// The scores of one date's pixels (`states`, rows x cols): NaN where not clear.
pub fn date_scores(states: &[u8], rows: usize, cols: usize, date: &Date, opts: &Options) -> Vec<f32> {
    assert_eq!(states.len(), rows * cols, "states must be rows x cols");
    let fixed = opts.w_doy * opts.doy_score(date.doy) + opts.w_sensor * date.sensor + opts.w_cover * date.cover;
    let need_dist = opts.w_cloud > 0.0;
    let dist = if need_dist {
        distance_to(|i| states[i] == CLOUD, rows, cols, opts.cloud_max)
    } else {
        Vec::new()
    };
    states
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            if s != CLEAR {
                return f32::NAN;
            }
            let c = if need_dist {
                opts.w_cloud * opts.cloud_score(f64::from(dist[i]))
            } else {
                0.0
            };
            (fixed + c) as f32
        })
        .collect()
}

/// Each pixel's best date.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Best {
    /// the date (index into the dates) of the highest score, -1 where no date is clear
    pub date: Vec<i32>,
    /// its score, NaN where none
    pub score: Vec<f32>,
    /// the clear dates of the pixel
    pub clear: Vec<u16>,
}

impl Best {
    fn empty(n: usize) -> Best {
        Best {
            date: vec![-1; n],
            score: vec![f32::NAN; n],
            clear: vec![0; n],
        }
    }

    /// Takes date `t`'s scores where they beat this one's (the earlier date of equal scores).
    fn take(&mut self, t: i32, scores: &[f32]) {
        for (i, &s) in scores.iter().enumerate() {
            if s.is_nan() {
                continue;
            }
            self.clear[i] += 1;
            let (b, bs) = (self.date[i], self.score[i]);
            if b < 0 || s > bs || (s == bs && t < b) {
                self.date[i] = t;
                self.score[i] = s;
            }
        }
    }

    fn merge(mut self, o: Best) -> Best {
        for i in 0..self.date.len() {
            self.clear[i] += o.clear[i];
            let (t, s) = (o.date[i], o.score[i]);
            if t < 0 {
                continue;
            }
            let (b, bs) = (self.date[i], self.score[i]);
            if b < 0 || s > bs || (s == bs && t < b) {
                self.date[i] = t;
                self.score[i] = s;
            }
        }
        self
    }

    /// The dates that win somewhere, in order.
    pub fn winners(&self, n_dates: usize) -> Vec<usize> {
        let mut won = vec![false; n_dates];
        for &t in &self.date {
            if t >= 0 {
                won[t as usize] = true;
            }
        }
        (0..n_dates).filter(|&t| won[t]).collect()
    }
}

/// The best date of every pixel of a window: `states` dates x rows x cols (NO_DATA, CLOUD, CLEAR), `dates` what is
/// known of each. The dates on the threads of `n_jobs`; the same result on any number of them.
pub fn best(
    states: &[u8],
    rows: usize,
    cols: usize,
    dates: &[Date],
    opts: &Options,
    n_jobs: i32,
) -> Result<Best, String> {
    opts.check()?;
    let n = rows * cols;
    if states.len() != dates.len() * n {
        return Err(format!(
            "states hold {} values, not dates x rows x cols ({} x {rows} x {cols})",
            states.len(),
            dates.len()
        ));
    }
    if dates.len() > u16::MAX as usize {
        return Err(format!("{} dates: {} at most", dates.len(), u16::MAX));
    }
    let one = |t: usize| date_scores(&states[t * n..(t + 1) * n], rows, cols, &dates[t], opts);
    let threads = parallel::threads(n_jobs);
    if threads <= 1 || dates.len() <= 1 {
        let mut b = Best::empty(n);
        for t in 0..dates.len() {
            b.take(t as i32, &one(t));
        }
        return Ok(b);
    }
    Ok(parallel::pool(threads).install(|| {
        (0..dates.len())
            .into_par_iter()
            .fold(
                || Best::empty(n),
                |mut b, t| {
                    b.take(t as i32, &one(t));
                    b
                },
            )
            .reduce(|| Best::empty(n), Best::merge)
    }))
}

/// Of `values` (dates x bands x pixels, the pixels of a window), each pixel's bands at its best date (`best.date`):
/// bands x pixels, NaN where none.
pub fn pick(values: &[f32], bands: usize, best: &Best) -> Vec<f32> {
    let n = best.date.len();
    assert!(
        values.len().is_multiple_of((bands * n).max(1)),
        "values must be dates x bands x pixels"
    );
    let mut out = vec![f32::NAN; bands * n];
    for (i, &t) in best.date.iter().enumerate() {
        if t < 0 {
            continue;
        }
        let t = t as usize;
        for b in 0..bands {
            out[b * n + i] = values[(t * bands + b) * n + i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distance by brute force.
    fn brute(cloud: &[bool], rows: usize, cols: usize, cap: f64) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let (y, x) = ((i / cols) as f64, (i % cols) as f64);
                let mut d = f64::INFINITY;
                for (j, &c) in cloud.iter().enumerate() {
                    if c {
                        let (yy, xx) = ((j / cols) as f64, (j % cols) as f64);
                        d = d.min(((y - yy).powi(2) + (x - xx).powi(2)).sqrt());
                    }
                }
                d.min(cap) as f32
            })
            .collect()
    }

    #[test]
    fn distances_are_exact() {
        let (rows, cols) = (37, 53);
        let mut s = 12345u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for density in [2u64, 20, 200, 2000] {
            let cloud: Vec<bool> = (0..rows * cols).map(|_| rnd() % density == 0).collect();
            let got = distance_to(|i| cloud[i], rows, cols, 30.0);
            let want = brute(&cloud, rows, cols, 30.0);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-4, "{g} != {w} (density 1/{density})");
            }
        }
        // no cloud: the cap everywhere
        assert!(distance_to(|_| false, 4, 5, 7.0).iter().all(|&d| d == 7.0));
        // one row, one column
        let d = distance_to(|i| i == 3, 1, 8, 100.0);
        assert_eq!(d, vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        let d = distance_to(|i| i == 0, 5, 1, 100.0);
        assert_eq!(d, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn scores() {
        let o = Options {
            target_doy: 200.0,
            doy_sigma: 20.0,
            cloud_min: 2.0,
            cloud_max: 12.0,
            ..Default::default()
        };
        assert_eq!(o.doy_score(200.0), 1.0);
        assert!((o.doy_score(220.0) - (-0.5f64).exp()).abs() < 1e-12);
        // around the year: day 4.75 is 10 days after day 360 (a year of 365.25 days), as day 350 is before it
        let o2 = Options {
            target_doy: 360.0,
            ..o.clone()
        };
        assert!((o2.doy_score(4.75) - o2.doy_score(350.0)).abs() < 1e-12);
        assert_eq!(o.cloud_score(1.0), 0.0);
        assert_eq!(o.cloud_score(7.0), 0.5);
        assert_eq!(o.cloud_score(50.0), 1.0);
        assert!(Options {
            cloud_max: 1.0,
            cloud_min: 1.0,
            ..Default::default()
        }
        .check()
        .is_err());
        assert!(Options {
            w_doy: 0.0,
            w_cloud: 0.0,
            w_sensor: 0.0,
            w_cover: 0.0,
            ..Default::default()
        }
        .check()
        .is_err());
    }

    #[test]
    fn the_best_date_wins() {
        // 1 x 6 pixels, 3 dates
        let (rows, cols) = (1, 6);
        let c = CLEAR;
        #[rustfmt::skip]
        let states = [
            c, c, c, c, c, c,             // date 0: clear, far from the target day
            CLOUD, c, c, c, NO_DATA, c,   // date 1: on the target day, a cloud at pixel 0
            NO_DATA, NO_DATA, c, c, c, c, // date 2: on the target day, a worse sensor
        ];
        let o = Options {
            target_doy: 200.0,
            doy_sigma: 10.0,
            cloud_min: 0.0,
            cloud_max: 4.0,
            ..Default::default()
        };
        let dates = [
            Date {
                doy: 150.0,
                sensor: 1.0,
                cover: 1.0,
            },
            Date {
                doy: 200.0,
                sensor: 1.0,
                cover: 1.0,
            },
            Date {
                doy: 200.0,
                sensor: 0.5,
                cover: 1.0,
            },
        ];
        let b1 = best(&states, rows, cols, &dates, &o, 1).unwrap();
        // date 0 scores 0.5 + 0.1 + 0.1 = 0.7 (its day ~0), date 2 0.3 + 0.5 + 0.05 + 0.1 = 0.95, date 1 0.8 + 0.5
        // d / 4 at d pixels from its cloud: pixel 1 (d 1, 0.625) goes to date 0 (date 2 has none there), pixels 2 and
        // 3 (0.75, 0.875) to date 2, pixel 4 (no data at date 1) to date 2, pixel 5 (d 4, 1.0) to date 1
        assert_eq!(b1.date, vec![0, 0, 2, 2, 2, 1]);
        assert_eq!(b1.clear, vec![1, 2, 3, 3, 2, 3]);
        assert!((b1.score[1] - 0.7).abs() < 1e-5);
        assert!((b1.score[3] - 0.95).abs() < 1e-6);
        assert!((b1.score[5] - 1.0).abs() < 1e-6);
        // the same on several threads
        let b4 = best(&states, rows, cols, &dates, &o, 4).unwrap();
        assert_eq!(b1, b4);
        assert_eq!(b1.winners(3), vec![0, 1, 2]);
        // the values of the winners
        let values: Vec<f32> = (0..3 * 2 * 6).map(|i| i as f32).collect();
        let p = pick(&values, 2, &b1);
        // date t band b pixel i at (t * 2 + b) * 6 + i
        assert_eq!(&p[..6], &[0.0, 1.0, 26.0, 27.0, 28.0, 17.0]);
        assert_eq!(&p[6..], &[6.0, 7.0, 32.0, 33.0, 34.0, 23.0]);
        // nothing clear
        let none = best(&[NO_DATA, CLOUD], 1, 2, &dates[..1], &o, 1).unwrap();
        assert_eq!(none.date, vec![-1, -1]);
        assert!(none.score.iter().all(|s| s.is_nan()));
    }

    #[test]
    fn ties_go_to_the_earlier_date_on_any_threads() {
        let (rows, cols) = (8, 8);
        let n = rows * cols;
        let dates: Vec<Date> = (0..40)
            .map(|_| Date {
                doy: 196.0,
                sensor: 1.0,
                cover: 1.0,
            })
            .collect();
        let states = vec![CLEAR; dates.len() * n];
        for j in [1, 2, 3, 8] {
            let b = best(&states, rows, cols, &dates, &Options::default(), j).unwrap();
            assert!(b.date.iter().all(|&t| t == 0));
            assert!(b.clear.iter().all(|&c| c == 40));
        }
    }
}
