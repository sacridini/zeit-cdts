//! Spectral adjustment of Landsat 4, 5, 7, 8, 9 and MODIS Terra and Aqua surface reflectance to
//! the spectral response of Sentinel-2A, predicting the red edge bands and the broad NIR that
//! those sensors do not have: ten bands out of six.
//!
//! Ported from FORCE (Framework for Operational Radiometric Correction for Environmental
//! monitoring), Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force, src/modules: higher-level/spec-adjust-hl.c
//! (`spectral_predict`), which implements a simplification of Scheffler, Frantz and Segl (2020),
//! "Spectral harmonization and red edge prediction of Landsat-8 to Sentinel-2 using land cover
//! optimized multivariate regressors", Remote Sensing of Environment 241:111723,
//! doi 10.1016/j.rse.2020.111723. The tables are in specadjust_tables.rs, generated from the same
//! file (tools/gen_specadjust.py).
//!
//! The method. Each sensor has 50 clusters of spectra (found by the authors on a library of
//! spectra, each sensor on its own) and a global one, the 51st, each with a linear regression
//! (six source bands plus an intercept) for each of the ten Sentinel-2A bands. A pixel (BLUE,
//! GREEN, RED, NIR, SWIR1, SWIR2, reflectance x 10000) gets, for each cluster, the spectral
//! angle (SAM) to the cluster's centre and the weight `1 - SAM / 15 degrees` (`0.00001` beyond
//! 15 degrees). The clusters used are the closest ones, at most 10, in the order of their
//! weights, while the weight is above `1 - 4/15` (an angle under 4 degrees), or above `1 - 12/15`
//! (12 degrees) and the global cluster's. The prediction is the weighted average of their
//! regressions. When fewer than 10 clusters qualify, the global cluster fills the next slot,
//! with its own weight: it is always one of the members unless 10 clusters were close.
//!
//! The arithmetic is FORCE's, in f64: the angles are the literal radian constants of FORCE
//! (0.0698132, 0.2094395, 0.2617995, not 4, 12, 15 times pi / 180), the cosine is
//! `xy / sqrt(xx * yy)` (the product first), the sums run in the band order and the weighted
//! sums in the order the clusters were picked. FORCE writes `(short)` of the result, a truncation
//! towards zero: the output here is not truncated (see `force_int16`), so that `force_int16` of
//! it is FORCE's value but for the pixels where the libm's `acos` of the platform differs by an
//! ulp from glibc's.
//!
//! Differences from FORCE, deliberate.
//!
//! * FORCE's first search for the closest cluster skips the cluster of index 9
//!   (`s != _SPECHOMO_N_SIM_-1`, a confusion of the number of clusters kept with the index of the
//!   global one, 50) and lets the global cluster be the first pick; the searches for the next ones
//!   (`s < N_CLS - 1`) exclude the global cluster and include 9. Here the global cluster is
//!   excluded from every search and cluster 9 takes part in all of them, which is what the second
//!   search does and what the paper does. `spectral_adjust_batch_as(.., Quirk::Force)` is the
//!   literal rule, for comparing (the pixels that differ are those where cluster 9 or the global
//!   cluster is the closest).
//! * FORCE never looks at the nodata value (a mask outside decides which pixels are adjusted) and
//!   truncates to a 16-bit integer, with a wrap on overflow. Here a pixel with a NaN band is NaN
//!   in all ten, and the output is float and neither truncated nor clipped.
//!
//! Speed, with the same results: the squared norm of each centre is computed once, the centres
//! and the regressions are laid out cluster by cluster, the `acos` is skipped for the clusters
//! beyond 15 degrees (their cosine is below that of 15 degrees by more than any error of `acos`,
//! and their weight is `0.00001` whatever the angle) and the clusters picked are found in one
//! pass over the few above the lowest threshold, not by ten sweeps of 50.

use crate::parallel::{for_each_row, uninit_f64};
use crate::specadjust_tables::{CENTER, COEFS};
use crate::tsa::Sample;
use std::sync::OnceLock;

/// The sensors that can be adjusted, in the order of FORCE's tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sensor {
    Lnd04,
    Lnd05,
    Lnd07,
    Lnd08,
    Lnd09,
    Mod01,
    Mod02,
}

impl Sensor {
    /// Every sensor, in the order of the tables.
    pub const ALL: [Sensor; 7] = [
        Sensor::Lnd04,
        Sensor::Lnd05,
        Sensor::Lnd07,
        Sensor::Lnd08,
        Sensor::Lnd09,
        Sensor::Mod01,
        Sensor::Mod02,
    ];

    /// FORCE's name of the sensor.
    pub fn name(self) -> &'static str {
        ["LND04", "LND05", "LND07", "LND08", "LND09", "MOD01", "MOD02"][self as usize]
    }

    /// The sensor of a FORCE name (`LND08`...), in any case.
    pub fn parse(name: &str) -> Result<Sensor, String> {
        let name = name.trim().to_ascii_uppercase();
        Sensor::ALL.into_iter().find(|s| s.name() == name).ok_or_else(|| {
            format!(
                "unknown sensor {name:?}: use LND04, LND05, LND07, LND08, LND09, MOD01 (Terra) or MOD02 (Aqua)"
            )
        })
    }
}

/// Which closest-cluster search to use: see the module's differences from FORCE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quirk {
    /// The global cluster is never the first pick and cluster 9 competes (the default).
    Fixed,
    /// FORCE's literal first search: cluster 9 skipped, the global cluster a candidate.
    Force,
}

/// Source bands, destination bands, regression terms and clusters of the tables.
pub const N_SRC: usize = 6;
pub const N_DST: usize = 10;
const N_COF: usize = 7;
/// Clusters of a sensor, the last one the global.
pub const N_CLS: usize = 51;
const GLOBAL: usize = N_CLS - 1;
/// Most clusters in the average.
const N_SIM: usize = 10;

const GOOD_SAM: f64 = 0.0698132;
const MEDI_SAM: f64 = 0.2094395;
const POOR_SAM: f64 = 0.2617995;
const MIN_WEIGHT: f64 = 1.0 - GOOD_SAM / POOR_SAM;
const MED_WEIGHT: f64 = 1.0 - MEDI_SAM / POOR_SAM;
/// The weight of a cluster beyond `POOR_SAM`.
const WEAK: f64 = 0.00001;
/// Cosines below this (that of `POOR_SAM` less a margin far above the error of `acos`, and
/// above the slope of `acos` there times it) are beyond `POOR_SAM`.
const MARGIN: f64 = 1e-9;

/// FORCE's `(short)` of a prediction: truncation towards zero, and a wrap in 16 bits where it
/// overflows (what the C compiler does on x86-64).
#[inline]
pub fn force_int16(v: f64) -> i16 {
    (v as i32) as i16
}

/// The tables of a sensor, in the order the per-pixel loops read them.
struct Prepared {
    /// `[cluster][source band]`
    centre: [[f64; N_SRC]; N_CLS],
    /// Squared norm of each centre, as FORCE sums it per pixel.
    yy: [f64; N_CLS],
    /// `[cluster][destination band][coefficient]`, the last coefficient the intercept.
    coef: [[[f64; N_COF]; N_DST]; N_CLS],
}

fn prepared(sensor: Sensor) -> &'static Prepared {
    static TABLES: OnceLock<Vec<Prepared>> = OnceLock::new();
    &TABLES.get_or_init(|| {
        (0..Sensor::ALL.len())
            .map(|sid| {
                let mut p = Prepared {
                    centre: [[0.0; N_SRC]; N_CLS],
                    yy: [0.0; N_CLS],
                    coef: [[[0.0; N_COF]; N_DST]; N_CLS],
                };
                for s in 0..N_CLS {
                    for b in 0..N_SRC {
                        p.centre[s][b] = f64::from(CENTER[sid][b][s]);
                    }
                    let mut yy = 0.0;
                    for b in 0..N_SRC {
                        yy += p.centre[s][b] * p.centre[s][b];
                    }
                    p.yy[s] = yy;
                    for d in 0..N_DST {
                        for k in 0..N_COF {
                            p.coef[s][d][k] = COEFS[sid][d][k][s];
                        }
                    }
                }
                p
            })
            .collect()
    })[sensor as usize]
}

/// The ten Sentinel-2A bands (BLUE GREEN RED REDEDGE1 REDEDGE2 REDEDGE3 BROADNIR NIR SWIR1 SWIR2,
/// that is B2 B3 B4 B5 B6 B7 B8 B8A B11 B12) of one pixel `x` (BLUE GREEN RED NIR SWIR1 SWIR2).
#[inline]
fn predict(p: &Prepared, cos_far: f64, force: bool, x: &[f64; N_SRC], out: &mut [f64]) {
    let mut xx = 0.0;
    for b in 0..N_SRC {
        xx += x[b] * x[b];
    }
    // the weight of each cluster, and the first closest (strict >, the lowest index wins)
    let mut w = [0.0f64; N_CLS];
    let (mut first_w, mut first_c) = (-1.0f64, 0usize);
    for s in 0..N_CLS {
        let c = &p.centre[s];
        let mut xy = 0.0;
        for b in 0..N_SRC {
            xy += x[b] * c[b];
        }
        let yy = p.yy[s];
        let weight = if xx > 0.0 && yy > 0.0 {
            let cosine = xy / (xx * yy).sqrt();
            if cosine < cos_far {
                WEAK
            } else {
                let sam = cosine.acos();
                if sam > POOR_SAM {
                    WEAK
                } else {
                    1.0 - sam / POOR_SAM
                }
            }
        } else {
            WEAK
        };
        w[s] = weight;
        let searched = if force { s != N_SIM - 1 } else { s != GLOBAL };
        if searched && weight > first_w {
            first_w = weight;
            first_c = s;
        }
    }
    let passes = |max: f64, global: f64| max > MIN_WEIGHT || (max > MED_WEIGHT && max > global);

    // The picks: slot c is the closest cluster left while it passes, else the global one (which
    // then fills the slot, with its weight, and ends the search).
    let mut pick = [(GLOBAL, 0.0f64); N_SIM];
    let mut n = 0;
    let mut global = w[GLOBAL];
    if passes(first_w, global) {
        pick[0] = (first_c, first_w);
        n = 1;
        if first_c == GLOBAL {
            global = 0.0;
        }
        // The next picks are the clusters of weight above MED_WEIGHT (the lowest threshold: none
        // below can pass), by weight, the lowest index first among equals.
        let mut cand = [(0.0f64, 0usize); GLOBAL];
        let mut nc = 0;
        for s in 0..GLOBAL {
            if s != first_c && w[s] > MED_WEIGHT {
                cand[nc] = (w[s], s);
                nc += 1;
            }
        }
        cand[..nc].sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        while n < N_SIM && n - 1 < nc && passes(cand[n - 1].0, global) {
            pick[n] = (cand[n - 1].1, cand[n - 1].0);
            n += 1;
        }
    }
    let used = if n < N_SIM {
        pick[n] = (GLOBAL, global);
        n + 1
    } else {
        N_SIM
    };

    let mut wpred = [0.0f64; N_DST];
    let mut wsum = 0.0;
    for &(s, weight) in &pick[..used] {
        let coef = &p.coef[s];
        for d in 0..N_DST {
            let k = &coef[d];
            let mut pred = 0.0;
            for b in 0..N_SRC {
                pred += k[b] * x[b];
            }
            pred += k[N_SRC];
            wpred[d] += weight * pred;
        }
        wsum += weight;
    }
    for d in 0..N_DST {
        out[d] = wpred[d] / wsum;
    }
}

/// Pixels per task of the batch.
const BLOCK: usize = 256;

/// The Sentinel-2A bands of every pixel of `src` (pixels x 6, BLUE GREEN RED NIR SWIR1 SWIR2,
/// reflectance x 10000) of `sensor`, into `out` (pixels x 10: BLUE GREEN RED REDEDGE1 REDEDGE2
/// REDEDGE3 BROADNIR NIR SWIR1 SWIR2, x 10000, not truncated). A pixel with a NaN band is NaN in
/// all ten. See the module for the method and its one difference from FORCE.
pub fn spectral_adjust_batch<T: Sample>(src: &[T], sensor: Sensor, out: &mut [f64], n_jobs: i32) {
    spectral_adjust_batch_as(src, sensor, out, n_jobs, Quirk::Fixed)
}

/// `spectral_adjust_batch` with the choice of FORCE's literal first search (`Quirk::Force`).
pub fn spectral_adjust_batch_as<T: Sample>(src: &[T], sensor: Sensor, out: &mut [f64], n_jobs: i32, quirk: Quirk) {
    assert_eq!(src.len() % N_SRC, 0, "src is pixels x 6");
    assert_eq!(out.len(), src.len() / N_SRC * N_DST, "out is pixels x 10");
    let p = prepared(sensor);
    let cos_far = POOR_SAM.cos() - MARGIN;
    let force = quirk == Quirk::Force;
    for_each_row(
        out,
        BLOCK * N_DST,
        n_jobs,
        || (),
        |_, block, rows| {
            let first = block * BLOCK;
            for (i, row) in rows.chunks_mut(N_DST).enumerate() {
                let s = &src[(first + i) * N_SRC..(first + i + 1) * N_SRC];
                let mut x = [0.0f64; N_SRC];
                let mut missing = false;
                for b in 0..N_SRC {
                    x[b] = s[b].to_f64();
                    missing |= x[b].is_nan();
                }
                if missing {
                    row.fill(f64::NAN);
                } else {
                    predict(p, cos_far, force, &x, row);
                }
            }
        },
    );
}

/// `spectral_adjust_batch` returning a new vector.
pub fn spectral_adjust<T: Sample>(src: &[T], sensor: Sensor, n_jobs: i32) -> Vec<f64> {
    let mut out = uninit_f64(src.len() / N_SRC * N_DST);
    spectral_adjust_batch(src, sensor, &mut out, n_jobs);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// FORCE's `spectral_predict` for one pixel, literally: 51 `acos`, the sweeps, and the
    /// `s != 9` of the first one when `force`, else `s != 50`.
    fn literal(sid: usize, x: &[f64; 6], force: bool) -> [f64; 10] {
        let mut weight = [0.0f64; 51];
        let mut max_weight = -1.0f64;
        let mut max_cluster = 0usize;
        for s in 0..51 {
            let (mut xy, mut xx, mut yy) = (0.0f64, 0.0f64, 0.0f64);
            for b in 0..6 {
                let c = f64::from(CENTER[sid][b][s]);
                xy += x[b] * c;
                xx += x[b] * x[b];
                yy += c * c;
            }
            let sam = if xx > 0.0 && yy > 0.0 { (xy / (xx * yy).sqrt()).acos() } else { 1.0 };
            weight[s] = if sam > POOR_SAM { 0.00001 } else { 1.0 - sam / POOR_SAM };
            let searched = if force { s != 9 } else { s != 50 };
            if weight[s] > max_weight && searched {
                max_weight = weight[s];
                max_cluster = s;
            }
        }
        let mut cluster_select = [0usize; 10];
        let mut weight_select = [0.0f64; 10];
        let mut c = 0;
        while c < 10 {
            cluster_select[c] = 50;
            weight_select[c] = weight[50];
            if max_weight > MIN_WEIGHT || (max_weight > MED_WEIGHT && max_weight > weight[50]) {
                cluster_select[c] = max_cluster;
                weight_select[c] = max_weight;
                weight[max_cluster] = 0.0;
            } else {
                break;
            }
            max_weight = -1.0;
            for s in 0..50 {
                if weight[s] > max_weight {
                    max_weight = weight[s];
                    max_cluster = s;
                }
            }
            c += 1;
        }
        let n_cluster = (c + 1).min(10);
        let mut wpred = [0.0f64; 10];
        let mut wsum = 0.0;
        for c in 0..n_cluster {
            let s = cluster_select[c];
            for b in 0..10 {
                let mut pred = 0.0;
                for b_ in 0..6 {
                    pred += COEFS[sid][b][b_][s] * x[b_];
                }
                pred += COEFS[sid][b][6][s];
                wpred[b] += weight_select[c] * pred;
            }
            wsum += weight_select[c];
        }
        let mut out = [0.0; 10];
        for b in 0..10 {
            out[b] = wpred[b] / wsum;
        }
        out
    }

    /// Integer pixels of every kind: a cluster's centre scaled and jittered, a mix of two
    /// centres, and uniform spectra, so that the paths of the selection are all taken.
    fn pixels(sid: usize, n: usize, seed: u64) -> Vec<f64> {
        let mut r = Rng(seed);
        let mut v = Vec::with_capacity(n * 6);
        for i in 0..n {
            let a = (r.next() * 51.0) as usize;
            let b = (r.next() * 51.0) as usize;
            let t = r.next();
            let kind = i % 4;
            for k in 0..6 {
                let (ca, cb) = (f64::from(CENTER[sid][k][a]), f64::from(CENTER[sid][k][b]));
                let x = match kind {
                    0 => ca * (0.6 + 0.8 * r.next()) + (r.next() - 0.5) * 60.0,
                    1 => t * ca + (1.0 - t) * cb + (r.next() - 0.5) * 200.0,
                    2 => ca + (r.next() - 0.5) * 8.0,
                    _ => r.next() * 12000.0 - 500.0,
                };
                v.push(x.floor());
            }
        }
        v
    }

    #[test]
    fn the_tables_are_forces() {
        let n: usize = CENTER.iter().flatten().flatten().count();
        assert_eq!(n, 7 * 6 * 51);
        assert_eq!(COEFS.iter().flatten().flatten().flatten().count(), 7 * 10 * 7 * 51);
        let sum: i64 = CENTER.iter().flatten().flatten().map(|&v| i64::from(v)).sum();
        assert_eq!(sum, 4_682_611);
        let weighted: i64 = CENTER
            .iter()
            .flatten()
            .flatten()
            .enumerate()
            .map(|(i, &v)| i64::from(v) * (1 + i as i64 % 97))
            .sum();
        assert_eq!(weighted, 232_027_352);
        let csum: f64 = COEFS.iter().flatten().flatten().flatten().sum();
        assert!((csum - -16895.676508712648).abs() < 1e-6, "{csum}");
        // MOD02 has no SWIR1: fifty -1 and the 0 of the C initializer, coefficients 0
        assert!(CENTER[6][4][..50].iter().all(|&v| v == -1));
        assert_eq!(CENTER[6][4][50], 0);
        assert!(COEFS[6].iter().all(|d| d[4].iter().all(|&v| v == 0.0)));
        // LND09 is LND08
        assert!(CENTER[3] == CENTER[4]);
        assert!(COEFS[3] == COEFS[4]);
        assert!(CENTER[2] != CENTER[3] && CENTER[5] != CENTER[6]);
        // the global cluster of LND04
        let g: Vec<i16> = (0..6).map(|b| CENTER[0][b][50]).collect();
        assert_eq!(g, [749, 1036, 1099, 2393, 2255, 1633]);
    }

    #[test]
    fn parses_sensors() {
        assert_eq!(Sensor::parse(" lnd08 "), Ok(Sensor::Lnd08));
        assert_eq!(Sensor::parse("MOD02"), Ok(Sensor::Mod02));
        assert!(Sensor::parse("SEN2A").is_err());
        for s in Sensor::ALL {
            assert_eq!(Sensor::parse(s.name()), Ok(s));
        }
    }

    #[test]
    fn the_fast_path_is_forces_arithmetic_bit_for_bit() {
        for (sid, sensor) in Sensor::ALL.into_iter().enumerate() {
            let src = pixels(sid, 3000, 7 + sid as u64);
            for (quirk, force) in [(Quirk::Force, true), (Quirk::Fixed, false)] {
                let mut out = vec![0.0; 3000 * 10];
                spectral_adjust_batch_as(&src, sensor, &mut out, 1, quirk);
                for (i, (x, o)) in src.chunks(6).zip(out.chunks(10)).enumerate() {
                    let x: [f64; 6] = x.try_into().unwrap();
                    let want = literal(sid, &x, force);
                    for b in 0..10 {
                        assert!(
                            want[b].to_bits() == o[b].to_bits() || (want[b].is_nan() && o[b].is_nan()),
                            "{sensor:?} {quirk:?} pixel {i} {x:?} band {b}: {} vs {}",
                            o[b],
                            want[b]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cluster_9_and_the_global_one_are_the_deliberate_difference() {
        let sid = 3;
        let centre = |s: usize| -> [f64; 6] { std::array::from_fn(|b| f64::from(CENTER[sid][b][s])) };
        let run = |x: [f64; 6], quirk| {
            let mut o = [0.0; 10];
            spectral_adjust_batch_as(&x, Sensor::Lnd08, &mut o, 1, quirk);
            o
        };
        // a spectrum that is exactly cluster 3's centre: both pick it first, and agree
        let x = centre(3);
        assert_eq!(run(x, Quirk::Fixed), run(x, Quirk::Force));
        // exactly cluster 9's: FORCE skips it as the first pick (here 18, 4.4 degrees away, comes
        // first, and 9 second); with two members the sums are the same in either order, so the
        // difference shows where three or more members, or the fallback, make the order or the
        // set change, as in the pixels below
        // over many pixels the two differ only where 9 or the global cluster is the best match
        let src = pixels(sid, 4000, 99);
        let (mut a, mut b) = (vec![0.0; 40000], vec![0.0; 40000]);
        spectral_adjust_batch_as(&src, Sensor::Lnd08, &mut a, 1, Quirk::Fixed);
        spectral_adjust_batch_as(&src, Sensor::Lnd08, &mut b, 1, Quirk::Force);
        let mut differ = 0;
        for (i, x) in src.chunks(6).enumerate() {
            if a[i * 10..i * 10 + 10] != b[i * 10..i * 10 + 10] {
                differ += 1;
                let x: [f64; 6] = x.try_into().unwrap();
                let w = |s: usize| {
                    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
                    for k in 0..6 {
                        let c = f64::from(CENTER[sid][k][s]);
                        xy += x[k] * c;
                        xx += x[k] * x[k];
                        yy += c * c;
                    }
                    let sam = if xx > 0.0 && yy > 0.0 { (xy / (xx * yy).sqrt()).acos() } else { 1.0 };
                    if sam > POOR_SAM { 0.00001 } else { 1.0 - sam / POOR_SAM }
                };
                // the first pick of the fixed search: the strict maximum over 0..50, lowest index
                let mut best = 0;
                for s in 1..50 {
                    if w(s) > w(best) {
                        best = s;
                    }
                }
                assert!(best == 9 || w(50) > w(best), "pixel {i}: the first pick is the same, {best}");
            }
        }
        eprintln!("{differ} of 4000 pixels differ between the two first searches");
        assert!(differ > 10, "the pixels hardly exercise the difference: {differ}");
    }

    #[test]
    fn edge_pixels() {
        let run = |sensor, x: &[f64]| {
            let mut o = vec![0.0; x.len() / 6 * 10];
            spectral_adjust_batch(x, sensor, &mut o, 1);
            o
        };
        // all zero: no angle, every weight 0.00001, the global regression at 0 (its intercepts)
        let o = run(Sensor::Lnd08, &[0.0; 6]);
        for b in 0..10 {
            let want = COEFS[3][b][6][50];
            assert!((o[b] - want).abs() <= 1e-12 * want.abs().max(1.0), "{b}: {} {want}", o[b]);
        }
        // a NaN band: ten NaN, and the next pixel is not affected
        let o = run(Sensor::Lnd08, &[100.0, 200.0, f64::NAN, 3000.0, 2000.0, 1000.0, 100.0, 200.0, 300.0, 3000.0, 2000.0, 1000.0]);
        assert!(o[..10].iter().all(|v| v.is_nan()) && o[10..].iter().all(|v| v.is_finite()));
        // negatives, and values above the scale, are not special
        let o = run(Sensor::Lnd05, &[-50.0, -30.0, -20.0, -10.0, -5.0, -1.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0, 30000.0]);
        assert!(o.iter().all(|v| v.is_finite()));
        // MOD02's SWIR1 enters the angle through its -1
        let o = run(Sensor::Mod02, &[300.0, 500.0, 400.0, 3000.0, 2500.0, 1500.0]);
        assert!(o.iter().all(|v| v.is_finite()));
        assert_eq!(force_int16(-0.9), 0);
        assert_eq!(force_int16(-1.9), -1);
        assert_eq!(force_int16(1.9), 1);
        assert_eq!(force_int16(40000.0), 40000i32 as i16);
    }

    #[test]
    fn f32_input_and_any_threads_give_the_same() {
        let src = pixels(2, 3000, 5);
        let src32: Vec<f32> = src.iter().map(|&v| v as f32).collect();
        let a = spectral_adjust(&src, Sensor::Lnd07, 1);
        let b = spectral_adjust(&src32, Sensor::Lnd07, 1);
        let c = spectral_adjust(&src, Sensor::Lnd07, 3);
        assert!(a == b && a == c);
    }
}
