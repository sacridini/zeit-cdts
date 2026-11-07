//! Local co-registration: the shift of an image to a reference at the points of a grid, and the filters that keep
//! the points fit to correct it with.
//!
//! A port of the `COREG_LOCAL` of AROSICS 1.14.0 (Automated and Robust Open-Source Image Co-Registration Software,
//! `arosics/CoReg_local.py`, `arosics/Tie_Point_Grid.py`; Scheffler, Hollstein, Hofmann and Bochow 2017, "AROSICS:
//! An automated and robust open-source image co-registration software for multi-sensor satellite data", Remote
//! Sensing 9(7):676),
//!
//!   Copyright (C) 2017-2026 Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz.de) and the GFZ Helmholtz Centre
//!   for Geosciences, Potsdam, Germany (https://www.gfz.de/). Licensed under the Apache License, Version 2.0
//!   (https://www.apache.org/licenses/LICENSE-2.0), which has no NOTICE file; distributed "AS IS", without
//!   warranties or conditions of any kind. This file is a translation to Rust of the parts of AROSICS named
//!   below, with changes (valid-pixel masks instead of footprint polygons), under GPL-3.0-or-later as the rest of
//!   zeit (Apache-2.0 code may enter it).
//!
//! What is ported:
//!
//! * the tie point grid (`Tie_Point_Grid._get_imXY__mapXY_points`, the grid itself is made by the caller as AROSICS'
//!   `np.arange(0, size + grid_res, grid_res)` of the target's pixel corners): a point is kept where it lies on
//!   data of both images (AROSICS: within the overlap polygon and off the bad-data masks);
//! * the COREG of each point with the window centred on it (`Tie_Point_Grid._get_spatial_shifts`; the matching
//!   itself is `coreg::coregister_with`), the per-point outputs (`ABS_SHIFT`, `ANGLE`, `SSIM_IMPROVED`...);
//! * the filters in AROSICS' order (`Tie_Point_Refiner.run_filtering`): reliability (level 1), SSIM (level 2) and
//!   RANSAC (level 3), whose model is scikit-image's `ransac` of an `AffineTransform` with 6 samples, up to 2000
//!   trials and the dynamic stop of `stop_probability` 1, and whose threshold is searched as AROSICS does
//!   (`_RANSAC_outlier_detection`), in map units.
//!
//! scikit-image's RANSAC draws its samples from numpy's `default_rng(seed)` (PCG64 seeded by a `SeedSequence`,
//! `Generator.choice(n, 6, replace=False)`: Floyd's algorithm and a shuffle, with Lemire's bounded integers); the
//! generator is reproduced here, so that with the same seed the samples, and with them the flagged points, are the
//! same as AROSICS' (checked against numpy's output in the tests). The affine model is estimated as scikit-image's
//! `ProjectiveTransform._estimate` does it: the points centred and scaled by their root mean square, the right
//! singular vector of the smallest singular value (here by the eigenvectors of A'A with Jacobi's method).
//!
//! Where it differs from AROSICS: the points' validity is read on the pixel masks; the time limit of the threshold
//! search (`rs_timeout`, 20 s of wall clock) is not kept, as it cannot be reproduced; with 5 or 6 points left for
//! RANSAC, where AROSICS' result is an error or an empty one, no point is flagged; a pair of images is matched
//! point by point on one thread each by `rayon`, where AROSICS starts a process for each; and the flags of RANSAC
//! are put on the points they belong to, where AROSICS 1.14.0 puts them on the wrong ones when the reliability or SSIM
//! filter flagged a point before (a pandas index misalignment in `_RANSAC_outlier_detection`), which
//! `LocalParams::rs_arosics_quirk` reproduces to compare with it.

use crate::coreg::{coregister_with, Image, Outcome, Overlap, Params};
use crate::parallel;
use rayon::prelude::*;
use rustfft::FftPlanner;

// --- numpy's default_rng ---------------------------------------------------------------------------------

/// numpy's `Generator(PCG64(SeedSequence(seed)))`, as far as `choice(n, k, replace=False)` needs.
pub struct Pcg64 {
    state: u128,
    inc: u128,
    half: Option<u32>,
}

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// `SeedSequence(seed).generate_state(4, uint64)`.
fn seed_sequence_state(seed: u64) -> [u64; 4] {
    const INIT_A: u32 = 0x43b0_d7e5;
    const MULT_A: u32 = 0x931e_8875;
    const INIT_B: u32 = 0x8b51_f9dd;
    const MULT_B: u32 = 0x58f3_8ded;
    const MIX_L: u32 = 0xca01_f9dd;
    const MIX_R: u32 = 0x4973_f715;
    let entropy: Vec<u32> = if seed >> 32 == 0 {
        vec![seed as u32]
    } else {
        vec![seed as u32, (seed >> 32) as u32]
    };
    let mut hash_const = INIT_A;
    let mut hashmix = |value: u32| -> u32 {
        let mut v = value ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_A);
        v = v.wrapping_mul(hash_const);
        v ^ (v >> 16)
    };
    let mix = |x: u32, y: u32| -> u32 {
        let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
        r ^ (r >> 16)
    };
    let mut pool = [0u32; 4];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0));
    }
    for i_src in 0..4 {
        for i_dst in 0..4 {
            if i_src != i_dst {
                let h = hashmix(pool[i_src]);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }
    for &e in entropy.iter().skip(4) {
        for p in pool.iter_mut() {
            let h = hashmix(e);
            *p = mix(*p, h);
        }
    }
    let mut hash_const = INIT_B;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut d = pool[i % 4] ^ hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        d = d.wrapping_mul(hash_const);
        *w = d ^ (d >> 16);
    }
    let mut out = [0u64; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = words[2 * i] as u64 | (words[2 * i + 1] as u64) << 32;
    }
    out
}

impl Pcg64 {
    /// `np.random.default_rng(seed)`.
    pub fn new(seed: u64) -> Pcg64 {
        let w = seed_sequence_state(seed);
        let initstate = (w[0] as u128) << 64 | w[1] as u128;
        let initseq = (w[2] as u128) << 64 | w[3] as u128;
        let mut g = Pcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            half: None,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        (((s >> 64) as u64) ^ (s as u64)).rotate_right((s >> 122) as u32)
    }

    /// The low half of a 64-bit draw, then the high half of the same one (numpy keeps the other half).
    fn next_u32(&mut self) -> u32 {
        if let Some(h) = self.half.take() {
            return h;
        }
        let v = self.next_u64();
        self.half = Some((v >> 32) as u32);
        v as u32
    }

    /// An integer in `0..=max` (numpy's `random_bounded_uint64(0, max)` with Lemire's method).
    fn bounded(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        if max <= 0xFFFF_FFFF {
            if max == 0xFFFF_FFFF {
                return self.next_u32() as u64;
            }
            let excl = max + 1;
            let mut m = self.next_u32() as u64 * excl;
            let mut leftover = m & 0xFFFF_FFFF;
            if leftover < excl {
                let threshold = (0xFFFF_FFFFu64 - max) % excl;
                while leftover < threshold {
                    m = self.next_u32() as u64 * excl;
                    leftover = m & 0xFFFF_FFFF;
                }
            }
            return m >> 32;
        }
        if max == u64::MAX {
            return self.next_u64();
        }
        let excl = max + 1;
        let mut m = self.next_u64() as u128 * excl as u128;
        let mut leftover = m as u64;
        if leftover < excl {
            let threshold = (u64::MAX - max) % excl;
            while leftover < threshold {
                m = self.next_u64() as u128 * excl as u128;
                leftover = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// `rng.choice(n, size, replace=False)`: Floyd's algorithm, then the shuffle.
    pub fn choice(&mut self, n: usize, size: usize) -> Vec<usize> {
        assert!(size <= n);
        let mut idx = vec![0usize; size];
        let mut seen: Vec<usize> = Vec::with_capacity(size);
        for j in n - size..n {
            let val = self.bounded(j as u64) as usize;
            let pick = if seen.contains(&val) { j } else { val };
            seen.push(pick);
            idx[j - (n - size)] = pick;
        }
        for i in (1..size).rev() {
            let j = self.bounded(i as u64) as usize;
            idx.swap(i, j);
        }
        idx
    }
}

// --- scikit-image's ransac of an AffineTransform ------------------------------------------------------

/// The eigenvector of the smallest eigenvalue of a symmetric 7 x 7 matrix (cyclic Jacobi).
#[allow(clippy::needless_range_loop)]
fn smallest_eigenvector(mut a: [[f64; 7]; 7]) -> [f64; 7] {
    let mut v = [[0.0f64; 7]; 7];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..100 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for i in 0..7 {
            diag += a[i][i] * a[i][i];
            for j in i + 1..7 {
                off += a[i][j] * a[i][j];
            }
        }
        if off <= 1e-32 * (diag + off) || off == 0.0 {
            break;
        }
        for p in 0..6 {
            for q in p + 1..7 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta >= 0.0 { 1.0 } else { -1.0 } / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
                for k in 0..7 {
                    let (x, y) = (a[p][k], a[q][k]);
                    a[p][k] = c * x - s * y;
                    a[q][k] = s * x + c * y;
                }
                for row in v.iter_mut() {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
            }
        }
    }
    let mut m = 0;
    for i in 1..7 {
        if a[i][i] < a[m][m] {
            m = i;
        }
    }
    let mut out = [0.0; 7];
    for (k, o) in out.iter_mut().enumerate() {
        *o = v[k][m];
    }
    out
}

/// An affine map `src -> dst`: `dst = cd + sd * (r . (src - cs) / ss)`, with `r` its two rows of the normalised
/// coordinates.
#[derive(Clone, Copy, Debug)]
struct Affine {
    cs: [f64; 2],
    ss: f64,
    cd: [f64; 2],
    sd: f64,
    r0: [f64; 3],
    r1: [f64; 3],
}

impl Affine {
    fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        let (x, y) = ((p[0] - self.cs[0]) / self.ss, (p[1] - self.cs[1]) / self.ss);
        [
            self.cd[0] + self.sd * (self.r0[0] * x + self.r0[1] * y + self.r0[2]),
            self.cd[1] + self.sd * (self.r1[0] * x + self.r1[1] * y + self.r1[2]),
        ]
    }
}

/// The centre and the root mean square scale of points (`_center_and_normalize_points`); None when all are equal.
fn center_scale(pts: &[[f64; 2]]) -> Option<([f64; 2], f64)> {
    let n = pts.len() as f64;
    let c = [
        pts.iter().map(|p| p[0]).sum::<f64>() / n,
        pts.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let ms = pts
        .iter()
        .map(|p| (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2))
        .sum::<f64>()
        / (2.0 * n);
    let s = ms.sqrt();
    if s == 0.0 || !s.is_finite() {
        None
    } else {
        Some((c, s))
    }
}

/// scikit-image's `AffineTransform.from_estimate(src, dst)`; None where it fails.
fn estimate_affine(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Option<Affine> {
    let (cs, ss) = center_scale(src)?;
    let (cd, sd) = center_scale(dst)?;
    let mut m = [[0.0f64; 7]; 7];
    let mut add = |row: [f64; 7]| {
        for i in 0..7 {
            for j in 0..7 {
                m[i][j] += row[i] * row[j];
            }
        }
    };
    for (s, d) in src.iter().zip(dst) {
        let (x, y) = ((s[0] - cs[0]) / ss, (s[1] - cs[1]) / ss);
        let (xx, yy) = ((d[0] - cd[0]) / sd, (d[1] - cd[1]) / sd);
        add([x, y, 1.0, 0.0, 0.0, 0.0, xx]);
        add([0.0, 0.0, 0.0, x, y, 1.0, yy]);
    }
    let v = smallest_eigenvector(m);
    // numpy.isclose(v[-1], 0)
    if v[6].abs() <= 1e-8 || !v[6].is_finite() {
        return None;
    }
    let k = -1.0 / v[6];
    Some(Affine {
        cs,
        ss,
        cd,
        sd,
        r0: [v[0] * k, v[1] * k, v[2] * k],
        r1: [v[3] * k, v[4] * k, v[5] * k],
    })
}

const MIN_SAMPLES: usize = 6;
const MAX_TRIALS: usize = 2000;

/// scikit-image's `_dynamic_max_trials` with `probability` 1.
fn dynamic_max_trials(n_inliers: usize, n_samples: usize) -> f64 {
    if n_inliers == 0 {
        return f64::INFINITY;
    }
    let eps = f64::EPSILON; // np.spacing(1)
    let ratio = n_inliers as f64 / n_samples as f64;
    let nom = 0.0f64.clamp(eps, 1.0 - eps);
    let denom = (1.0 - ratio.powf(MIN_SAMPLES as f64)).clamp(eps, 1.0 - eps);
    (nom.ln() / denom.ln()).ceil()
}

/// scikit-image's `ransac((src, dst), AffineTransform, min_samples=6, residual_threshold=th, max_trials=2000,
/// stop_sample_num, stop_residuals_sum, rng=seed)`: the inlier mask, None when no model had an inlier.
fn ransac_affine(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    th: f64,
    stop_sample_num: f64,
    stop_residuals_sum: f64,
    rng: &mut Pcg64,
) -> Option<Vec<bool>> {
    let n = src.len();
    let mut best_num = 0usize;
    let mut best_sum = f64::INFINITY;
    let mut best: Vec<bool> = Vec::new();
    let mut max_trials = MAX_TRIALS as f64;
    let mut spl = rng.choice(n, MIN_SAMPLES);
    let mut trials = 0.0f64;
    let mut res = vec![0.0f64; n];
    while trials < max_trials {
        trials += 1.0;
        let samples = spl;
        spl = rng.choice(n, MIN_SAMPLES);
        let s: Vec<[f64; 2]> = samples.iter().map(|&i| src[i]).collect();
        let d: Vec<[f64; 2]> = samples.iter().map(|&i| dst[i]).collect();
        let Some(model) = estimate_affine(&s, &d) else {
            continue;
        };
        let mut sum = 0.0;
        let mut count = 0usize;
        for i in 0..n {
            let q = model.apply(src[i]);
            let r = ((q[0] - dst[i][0]).powi(2) + (q[1] - dst[i][1]).powi(2)).sqrt();
            res[i] = r;
            sum += r * r;
            if r < th {
                count += 1;
            }
        }
        if count > best_num || (count == best_num && sum < best_sum) {
            best_num = count;
            best_sum = sum;
            best = res.iter().map(|&r| r < th).collect();
            max_trials = max_trials.min(dynamic_max_trials(count, n));
            if best_num as f64 >= stop_sample_num || best_sum <= stop_residuals_sum {
                break;
            }
        }
    }
    if best.iter().any(|&b| b) {
        Some(best)
    } else {
        None
    }
}

/// The outliers among `src -> dst` by AROSICS' threshold search around scikit-image's RANSAC
/// (`Tie_Point_Refiner._RANSAC_outlier_detection`); all false where RANSAC cannot run (6 points or fewer) or finds
/// no model.
fn ransac_outliers(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    max_outlier: f64,
    tolerance: f64,
    max_iter: usize,
    seed: Option<u64>,
) -> Vec<bool> {
    let n = src.len();
    if n <= MIN_SAMPLES {
        return vec![false; n];
    }
    let seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    });
    let min_inlier = 100.0 - max_outlier;
    let stop_sample_num = ((min_inlier - tolerance) / 100.0 * n as f64) as i64 as f64;
    let stop_residuals_sum = ((max_outlier - tolerance) / 100.0 * n as f64) as i64 as f64;
    let ideal = min_inlier * n as f64 / 100.0;
    let mut th = 5.0f64;
    let mut checked: Vec<f64> = Vec::new();
    let mut th_sub = 2.0f64;
    let mut count_iter = 0usize;
    let mut count_inliers = 0usize;
    let mut inliers: Option<Vec<bool>>;
    loop {
        if !checked.is_empty() {
            let too_strict = (count_inliers as f64) < ideal;
            let mut th_new = 0.0f64;
            while th_new <= 0.0 {
                th_new = if too_strict { th + th_sub } else { th - th_sub };
                if th_new <= 0.0 {
                    th_sub /= 2.0;
                }
            }
            let already = checked.contains(&th_new);
            if already {
                th_sub /= 2.0;
            }
            th = if !already {
                th_new
            } else if too_strict {
                th + th_sub
            } else {
                th - th_sub
            };
        }
        // each call of skimage's ransac seeds its own generator
        let mut rng = Pcg64::new(seed);
        inliers = ransac_affine(src, dst, th, stop_sample_num, stop_residuals_sum, &mut rng);
        count_inliers = inliers.as_ref().map_or(0, |v| v.iter().filter(|&&b| b).count());
        let pct = count_inliers as f64 / n as f64 * 100.0;
        if !checked.contains(&th) {
            checked.push(th);
        }
        if min_inlier - tolerance < pct && pct < min_inlier + tolerance {
            break;
        }
        if count_iter > max_iter {
            break;
        }
        count_iter += 1;
    }
    match inliers {
        Some(v) => v.iter().map(|&b| !b).collect(),
        None => vec![false; n],
    }
}

// --- the grid of tie points -----------------------------------------------------------------------------

/// The geotransform of the target image: the corner of pixel `(c, r)` is `(x0 + c * dx, y0 + r * dy)`, `dy`
/// negative on a north-up grid.
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub x0: f64,
    pub dx: f64,
    pub y0: f64,
    pub dy: f64,
}

#[derive(Clone, Debug)]
pub struct LocalParams {
    /// The matching of each point: `window_size`, `max_shift`, `max_iter`, `integer_range`, `nodata` and `fill`
    /// (`window_pos` is the point's).
    pub coreg: Params,
    /// AROSICS' `tieP_filter_level` (3): 0 none, 1 reliability, 2 plus SSIM, 3 plus RANSAC.
    pub filter_level: u8,
    /// AROSICS' `min_reliability` (60).
    pub min_reliability: f64,
    /// AROSICS' `rs_max_outlier` (10, percent) and `rs_tolerance` (2.5, percent).
    pub rs_max_outlier: f64,
    pub rs_tolerance: f64,
    /// AROSICS' `rs_max_iter` (15), the tries of the threshold search.
    pub rs_max_iter: usize,
    /// AROSICS' `rs_random_state` (0); None: from the clock.
    pub rs_seed: Option<u64>,
    /// Reproduce a bug of AROSICS 1.14.0: when the reliability or SSIM filters have flagged points, the flags of
    /// RANSAC (one for each point it got) are assigned to the rows of the table by their labels instead of their
    /// positions (`inGDF['outliers'] = outliers` in `_RANSAC_outlier_detection`), so they land on other points. Off
    /// by default; on, the result is AROSICS' to the point.
    pub rs_arosics_quirk: bool,
}

impl Default for LocalParams {
    fn default() -> LocalParams {
        LocalParams {
            coreg: Params::default(),
            filter_level: 3,
            min_reliability: 60.0,
            rs_max_outlier: 10.0,
            rs_tolerance: 2.5,
            rs_max_iter: 15,
            rs_seed: Some(0),
            rs_arosics_quirk: false,
        }
    }
}

/// One tie point.
#[derive(Clone, Debug)]
pub struct TiePoint {
    /// The point's position in the list given.
    pub index: usize,
    /// The point in map coordinates (`X_MAP`, `Y_MAP`).
    pub x_map: f64,
    pub y_map: f64,
    /// The shift in pixels and map units of the target, as AROSICS': NaN where the match failed.
    pub x_shift_px: f64,
    pub y_shift_px: f64,
    pub x_shift_map: f64,
    pub y_shift_map: f64,
    /// `ABS_SHIFT`: the length of the shift in map units; `ANGLE`: its direction in degrees from north.
    pub abs_shift: f64,
    pub angle: f64,
    /// The similarity of the window did not decrease (`SSIM_IMPROVED`).
    pub ssim_improved: bool,
    pub outcome: Outcome,
    /// The flags of the filters: reliability, SSIM, RANSAC (`L1_OUTLIER` to `L3_OUTLIER`), and any of them.
    pub l1: bool,
    pub l2: bool,
    pub l3: bool,
    pub outlier: bool,
}

/// AROSICS' `angle_to_north`: the angle of `(x, y)` to north, clockwise, in degrees.
fn angle_to_north(x: f64, y: f64) -> f64 {
    (y.atan2(x) - std::f64::consts::FRAC_PI_2)
        .to_degrees()
        .rem_euclid(360.0)
        .abs()
}

/// The tie points of `points` (pixel coordinates in the target's image, `(x, y)`, the corner of its first pixel
/// is `(0, 0)`) kept where they are on data of both images, in the order given, each matched with `p.coreg`'s
/// window centred on it and then filtered. `reference` and `target` are the overlap of the two images cut to the
/// same grid, which starts `offset` pixels into the target's image. Points run on `n_jobs` threads.
pub fn tie_points(
    reference: &Image,
    target: &Image,
    points: &[(f64, f64)],
    offset: (f64, f64),
    geo: &Geo,
    p: &LocalParams,
    n_jobs: i32,
) -> Vec<TiePoint> {
    assert!(
        reference.ny == target.ny && reference.nx == target.nx,
        "the images must have the same shape"
    );
    let ov = Overlap::new(reference, target);
    let kept: Vec<usize> = (0..points.len())
        .filter(|&i| ov.contains(points[i].0 - offset.0, points[i].1 - offset.1))
        .collect();
    let one = |planner: &mut FftPlanner<f32>, i: usize| -> TiePoint {
        let (x, y) = points[i];
        let mut pp = p.coreg.clone();
        pp.window_pos = Some((x - offset.0, y - offset.1));
        let o = coregister_with(&ov, reference, target, &pp, planner);
        let x_map = geo.x0 + x * geo.dx;
        let y_map = geo.y0 - y * geo.dy.abs();
        let mut t = TiePoint {
            index: i,
            x_map,
            y_map,
            x_shift_px: f64::NAN,
            y_shift_px: f64::NAN,
            x_shift_map: f64::NAN,
            y_shift_map: f64::NAN,
            abs_shift: f64::NAN,
            angle: f64::NAN,
            ssim_improved: false,
            outcome: o,
            l1: false,
            l2: false,
            l3: false,
            outlier: false,
        };
        if t.outcome.success {
            let (sx, sy) = (t.outcome.x_shift, t.outcome.y_shift);
            t.x_shift_px = sx;
            t.y_shift_px = sy;
            // imXY2mapXY of the shift, minus the origin
            t.x_shift_map = (geo.x0 + sx * geo.dx + sy * 0.0) - geo.x0;
            t.y_shift_map = (geo.y0 + sx * 0.0 + sy * geo.dy) - geo.y0;
            t.abs_shift = (t.x_shift_map * t.x_shift_map + t.y_shift_map * t.y_shift_map).sqrt();
            t.angle = angle_to_north(sx, sy);
            t.ssim_improved = t.outcome.ssim_before <= t.outcome.ssim_after;
        }
        t
    };
    let threads = parallel::threads(n_jobs).max(1).min(kept.len().max(1));
    let mut tps: Vec<TiePoint> = if threads > 1 {
        parallel::pool(threads).install(|| {
            kept.par_iter()
                .map_init(FftPlanner::<f32>::new, |planner, &i| one(planner, i))
                .collect()
        })
    } else {
        let mut planner = FftPlanner::<f32>::new();
        kept.iter().map(|&i| one(&mut planner, i)).collect()
    };
    refine(&mut tps, p);
    tps
}

/// `Tie_Point_Refiner.run_filtering` on the points whose match succeeded; the flags of the failed stay false.
pub fn refine(tps: &mut [TiePoint], p: &LocalParams) {
    let ok: Vec<usize> = (0..tps.len()).filter(|&i| tps[i].outcome.success).collect();
    if ok.is_empty() || p.filter_level == 0 {
        return;
    }
    for &i in &ok {
        let t = &mut tps[i];
        let rel = t.outcome.reliability;
        t.l1 = rel.is_nan() || rel < p.min_reliability;
        if p.filter_level > 1 {
            t.l2 = !t.ssim_improved;
        }
    }
    if p.filter_level > 2 {
        let input: Vec<usize> = ok.iter().copied().filter(|&i| !(tps[i].l1 || tps[i].l2)).collect();
        if input.len() > 4 {
            let src: Vec<[f64; 2]> = input.iter().map(|&i| [tps[i].x_map, tps[i].y_map]).collect();
            let dst: Vec<[f64; 2]> = input
                .iter()
                .map(|&i| [tps[i].x_map + tps[i].x_shift_map, tps[i].y_map + tps[i].y_shift_map])
                .collect();
            let out = ransac_outliers(&src, &dst, p.rs_max_outlier, p.rs_tolerance, p.rs_max_iter, p.rs_seed);
            // AROSICS' misassignment applies only when some points were left out of RANSAC: the flag of the k-th
            // point it got goes to the row labelled k (the position in the table of the points that matched), and
            // to none if that row is past the last flag
            let quirk = p.rs_arosics_quirk && input.len() < ok.len();
            for (k, &i) in input.iter().enumerate() {
                tps[i].l3 = if quirk {
                    out.get(i).copied().unwrap_or(false)
                } else {
                    out[k]
                };
            }
        }
    }
    for &i in &ok {
        let t = &mut tps[i];
        t.outlier = t.l1 || t.l2 || t.l3;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coreg::tests::Scene;

    #[test]
    fn rng_is_numpys() {
        // numpy 2: default_rng(0).bit_generator.random_raw(4)
        let mut g = Pcg64::new(0);
        let raw: Vec<u64> = (0..4).map(|_| g.next_u64()).collect();
        assert_eq!(
            raw,
            [
                11749869230777074271,
                4976686463289251617,
                755828109848996024,
                304881062738325533
            ]
        );
        // a seed of more than 32 bits
        let mut g = Pcg64::new(12345678901234);
        assert_eq!(g.next_u64(), 7496343037614434180);
        assert_eq!(g.next_u64(), 483370428031952295);
        // rng.choice(n, 6, replace=False), three draws in a row
        for (n, want) in [
            (
                7usize,
                [[4usize, 2, 3, 5, 0, 1], [5, 2, 1, 6, 3, 4], [0, 3, 1, 6, 4, 5]],
            ),
            (
                40,
                [
                    [10, 22, 18, 12, 1, 29],
                    [28, 18, 31, 25, 22, 36],
                    [0, 14, 23, 1, 32, 21],
                ],
            ),
            (
                123,
                [
                    [32, 75, 61, 37, 5, 100],
                    [88, 59, 107, 77, 72, 117],
                    [0, 47, 79, 4, 103, 67],
                ],
            ),
            (
                20000,
                [
                    [5395, 12736, 10221, 6156, 819, 17008],
                    [14589, 10070, 18250, 12645, 12130, 19412],
                    [54, 7881, 13414, 671, 17146, 11085],
                ],
            ),
        ] {
            let mut g = Pcg64::new(0);
            for w in want {
                assert_eq!(g.choice(n, 6), w, "n = {n}");
            }
        }
    }

    #[test]
    fn affine_estimate_is_exact_on_an_affine_map() {
        let f = |p: [f64; 2]| [1.3 * p[0] - 0.2 * p[1] + 5.0, 0.1 * p[0] + 0.9 * p[1] - 3.0];
        let src: Vec<[f64; 2]> = (0..9)
            .map(|k| {
                [
                    500000.0 + (k * 37 % 11) as f64 * 100.0,
                    4600000.0 - (k * 53 % 7) as f64 * 130.0,
                ]
            })
            .collect();
        let dst: Vec<[f64; 2]> = src.iter().map(|&p| f(p)).collect();
        let m = estimate_affine(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let q = m.apply(*s);
            assert!((q[0] - d[0]).abs() < 1e-6 && (q[1] - d[1]).abs() < 1e-6, "{q:?} {d:?}");
        }
        // all points equal: no model
        assert!(estimate_affine(&[[1.0, 1.0]; 6], &[[2.0, 2.0]; 6]).is_none());
    }

    /// A grid of points with the shift of an affine field and noise, with some wrong shifts planted.
    fn field(n_side: usize, bad: &[usize]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let mut src = Vec::new();
        let mut dst = Vec::new();
        for r in 0..n_side {
            for c in 0..n_side {
                let p = [500000.0 + c as f64 * 6000.0, 4600000.0 - r as f64 * 6000.0];
                let k = r * n_side + c;
                let noise = ((k * 7919) % 13) as f64 / 13.0 - 0.5;
                let mut s = [
                    30.0 + 0.0005 * (p[0] - 500000.0) + noise * 0.3,
                    -15.0 + 0.0003 * (4600000.0 - p[1]),
                ];
                if bad.contains(&k) {
                    s = [s[0] + 90.0, s[1] - 60.0];
                }
                src.push(p);
                dst.push([p[0] + s[0], p[1] + s[1]]);
            }
        }
        (src, dst)
    }

    #[test]
    fn ransac_flags_the_planted_outliers() {
        let bad = [3usize, 17, 40, 41];
        let (src, dst) = field(9, &bad);
        // 81 points, 10 % expected outliers: the search settles on a threshold that flags the 4 wrong ones
        let out = ransac_outliers(&src, &dst, 10.0, 2.5, 15, Some(0));
        for (k, &o) in out.iter().enumerate() {
            if bad.contains(&k) {
                assert!(o, "point {k} not flagged");
            }
        }
        let n_out = out.iter().filter(|&&o| o).count();
        assert!((4..=12).contains(&n_out), "{n_out} flagged");
        // the same seed gives the same flags
        assert_eq!(out, ransac_outliers(&src, &dst, 10.0, 2.5, 15, Some(0)));
        // 6 points or fewer: RANSAC does not run
        assert!(ransac_outliers(&src[..6], &dst[..6], 10.0, 2.5, 15, Some(0))
            .iter()
            .all(|&o| !o));
    }

    fn point(i: usize, x: f64, y: f64, sx: f64, sy: f64, rel: f64, before: f64, after: f64) -> TiePoint {
        let mut o = Outcome::failed(crate::coreg::Failure::NoMatch);
        o.success = true;
        o.failure = None;
        o.reliability = rel;
        o.ssim_before = before;
        o.ssim_after = after;
        TiePoint {
            index: i,
            x_map: x,
            y_map: y,
            x_shift_px: sx,
            y_shift_px: sy,
            x_shift_map: sx * 30.0,
            y_shift_map: -sy * 30.0,
            abs_shift: 0.0,
            angle: 0.0,
            ssim_improved: before <= after,
            outcome: o,
            l1: false,
            l2: false,
            l3: false,
            outlier: false,
        }
    }

    #[test]
    fn filters_run_in_order() {
        let mut tps: Vec<TiePoint> = (0..30)
            .map(|i| {
                point(
                    i,
                    500000.0 + (i % 6) as f64 * 3000.0,
                    4600000.0 - (i / 6) as f64 * 3000.0,
                    1.0,
                    0.5,
                    80.0,
                    0.4,
                    0.6,
                )
            })
            .collect();
        tps[2].outcome.reliability = 30.0; // L1
        tps[5].outcome.ssim_after = 0.1; // L2
        tps[5].ssim_improved = false;
        tps[9].outcome.reliability = 20.0;
        tps[9].outcome.ssim_after = 0.1; // both
        tps[9].ssim_improved = false;
        tps[11].x_shift_map += 25.0 * 30.0; // RANSAC
        tps[20].outcome = Outcome::failed(crate::coreg::Failure::NoPeak);
        refine(&mut tps, &LocalParams::default());
        assert!(tps[2].l1 && !tps[2].l2 && !tps[2].l3 && tps[2].outlier);
        assert!(!tps[5].l1 && tps[5].l2 && !tps[5].l3 && tps[5].outlier);
        assert!(tps[9].l1 && tps[9].l2 && !tps[9].l3);
        assert!(tps[11].l3 && tps[11].outlier, "RANSAC missed the wrong shift");
        assert!(!tps[20].outlier && !tps[20].l1, "a failed point has no flags");
        assert!(!tps[0].outlier);
        // AROSICS' misassignment of RANSAC's flags: the wrong shift is the 9th point RANSAC got (the 3 flagged before
        // it left out), and the flag lands on the row labelled 8
        let mut q: Vec<TiePoint> = tps
            .iter()
            .map(|t| TiePoint {
                l1: false,
                l2: false,
                l3: false,
                outlier: false,
                ..t.clone()
            })
            .collect();
        refine(
            &mut q,
            &LocalParams {
                rs_arosics_quirk: true,
                ..LocalParams::default()
            },
        );
        assert!(q[8].l3 && !q[11].l3, "{:?}", q.iter().map(|t| t.l3).collect::<Vec<_>>());
        assert!(q[2].l1 && q[5].l2, "the other filters are as they were");
        // level 1: only the reliability
        let mut t1: Vec<TiePoint> = tps
            .iter()
            .map(|t| TiePoint {
                l1: false,
                l2: false,
                l3: false,
                outlier: false,
                ..t.clone()
            })
            .collect();
        refine(
            &mut t1,
            &LocalParams {
                filter_level: 1,
                ..LocalParams::default()
            },
        );
        assert!(t1[2].l1 && !t1[5].l2 && !t1[11].l3 && !t1[5].outlier);
        // level 0: none
        let mut t0 = tps.clone();
        for t in t0.iter_mut() {
            t.l1 = false;
            t.l2 = false;
            t.l3 = false;
            t.outlier = false;
        }
        refine(
            &mut t0,
            &LocalParams {
                filter_level: 0,
                ..LocalParams::default()
            },
        );
        assert!(t0.iter().all(|t| !t.outlier));
    }

    /// A scene moved by a smooth field: the reference is the scene, the target at `p` is the scene at
    /// `p + d(p)`, so the shift to find at `p` is `d(p)`.
    fn warped_pair(ny: usize, nx: usize, field: &dyn Fn(f64, f64) -> (f64, f64)) -> (Vec<f32>, Vec<f32>) {
        let sc = Scene::new(21);
        let mut r = Vec::with_capacity(ny * nx);
        let mut t = Vec::with_capacity(ny * nx);
        for y in 0..ny {
            for x in 0..nx {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                r.push(sc.at(px, py));
                let (dx, dy) = field(px, py);
                t.push(sc.at(px + dx, py + dy));
            }
        }
        (r, t)
    }

    #[test]
    fn tie_points_recover_a_smooth_field() {
        let (ny, nx) = (600, 700);
        let field = |x: f64, y: f64| (1.2 + 0.002 * x + 0.8 * (y / 120.0).sin(), -0.7 + 0.001 * y);
        let (rd, td) = warped_pair(ny, nx, &field);
        let valid = vec![true; ny * nx];
        let r = Image {
            data: &rd,
            valid: &valid,
            ny,
            nx,
        };
        let t = Image {
            data: &td,
            valid: &valid,
            ny,
            nx,
        };
        let points: Vec<(f64, f64)> = (0..=ny.div_ceil(100))
            .flat_map(|j| (0..=nx.div_ceil(100)).map(move |i| (i as f64 * 100.0, j as f64 * 100.0)))
            .collect();
        let geo = Geo {
            x0: 500000.0,
            dx: 30.0,
            y0: 4600000.0,
            dy: -30.0,
        };
        let p = LocalParams {
            coreg: Params {
                window_size: 128,
                ..Params::default()
            },
            ..LocalParams::default()
        };
        let tps = tie_points(&r, &t, &points, (0.0, 0.0), &geo, &p, 2);
        // the points on the image: 0..=600 x 0..=700 in steps of 100 (the grid reaches the far edge)
        assert_eq!(tps.len(), 7 * 8);
        let mut good = 0;
        for tp in &tps {
            if !tp.outcome.success {
                continue;
            }
            let (x, y) = points[tp.index];
            let (dx, dy) = field(x, y);
            if (tp.x_shift_px - dx).abs() < 0.3 && (tp.y_shift_px - dy).abs() < 0.3 {
                good += 1;
            }
            assert_eq!(tp.x_map, 500000.0 + x * 30.0);
            assert_eq!(tp.y_map, 4600000.0 - y * 30.0);
            assert!((tp.x_shift_map - tp.x_shift_px * 30.0).abs() < 1e-6);
            assert!((tp.y_shift_map + tp.y_shift_px * 30.0).abs() < 1e-6);
        }
        assert!(good >= 40, "only {good} of {} points recovered the field", tps.len());
        // the same on one thread
        let one = tie_points(&r, &t, &points, (0.0, 0.0), &geo, &p, 1);
        assert_eq!(one.len(), tps.len());
        for (a, b) in one.iter().zip(&tps) {
            assert!(a.x_shift_px.to_bits() == b.x_shift_px.to_bits() && a.outlier == b.outlier);
        }
    }

    #[test]
    fn points_off_the_data_are_dropped() {
        let (ny, nx) = (300, 300);
        let (rd, td) = warped_pair(ny, nx, &|_, _| (1.0, 1.0));
        let vr = vec![true; ny * nx];
        // the target has no data on its left 150 columns
        let vt: Vec<bool> = (0..ny * nx).map(|k| k % nx >= 150).collect();
        let r = Image {
            data: &rd,
            valid: &vr,
            ny,
            nx,
        };
        let t = Image {
            data: &td,
            valid: &vt,
            ny,
            nx,
        };
        let points = [
            (10.0, 10.0),
            (149.0, 100.0),
            (200.0, 100.0),
            (300.0, 300.0),
            (400.0, 100.0),
            (-1.0, 5.0),
        ];
        let geo = Geo {
            x0: 0.0,
            dx: 10.0,
            y0: 0.0,
            dy: -10.0,
        };
        let tps = tie_points(&r, &t, &points, (0.0, 0.0), &geo, &LocalParams::default(), 1);
        let idx: Vec<usize> = tps.iter().map(|t| t.index).collect();
        assert_eq!(idx, vec![2, 3]);
    }
}
