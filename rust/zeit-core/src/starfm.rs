//! STARFM (Spatial and Temporal Adaptive Reflectance Fusion Model) as FORCE's streamlined `starfm()`, and FORCE's
//! third Sentinel-2 resolution merge, `resolution_merge_3` (`RES_MERGE = STARFM`): the six 20 m bands predicted at
//! 10 m from a pseudo-PAN (mean of green and red) and the broad NIR, with coarse/fine pairs made by degrading those
//! 10 m bands with an approximated point spread function.
//!
//! Ported from FORCE 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring,
//! Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force),
//! src/modules/lower-level/resmerge-ll.c (`starfm`, `starfm_candidate_t` and `resolution_merge_3`) and
//! src/modules/cross-level/imagefuns-cl.c (`gauss_kernel`), line by line: the same float types, order of operations,
//! neighbourhood and candidate selection and handling of the pixels that are off. The text of resmerge-ll.c about
//! the STARFM part follows verbatim:
//!
//! ```text
//! The STARFM implementation included in this file, is adopted from the
//! version provided provided by the United States Department of Agricul-
//! ture, Agricultural Research Service:
//! https://www.ars.usda.gov/research/software/download/?softwareid=432
//! STARFM Copyright (C) 2005-2014 Feng Gao
//! -------------------------------------------------------------------
//! STARFM License:
//! The package is now being released at no cost to the public for further
//! development. ARS is releasing the STARFM so that interested parties
//! can use the software tool for their own needs and purposes. ARS does
//! not foresee providing monetary or technical support to refine, adapt,
//! or use this software tool, and provides no warranty for its use for
//! any purpose. ARS does not reserve any rights or interests in the work
//! that may be performed by others to refine or adapt it. ARS does reser-
//! ve the right to continue its own refinement of the current version of
//! the software package at a later date, should program needs require it.
//! ```
//!
//! The method is Gao, Masek, Schwaller and Hall (2006), "On the Blending of the Landsat and MODIS Surface
//! Reflectance: Predicting Daily Landsat Surface Reflectance", IEEE TGRS 44:2207-2218.
//!
//! `starfm`: for each pixel that is not off, the candidates are the pixels of the (2r+1) x (2r+1) window, in each
//! of the base pairs, that are spectrally similar to the pixel in the fine image (the absolute difference at most
//! `2 std / nclass` of that pair) and whose coarse-to-coarse difference (to the coarse image at the prediction
//! date) or coarse-to-fine difference is under the smallest of the pixel's own plus the uncertainty. Each
//! candidate is weighted by the inverse of (|coarse - coarse at the prediction date| + 0.0001) x (|coarse - fine| +
//! 0.0001) x (1 + distance / r), or by 1 if the product of the first two is under the combined uncertainty, and
//! the prediction is the weighted mean of `fine - (coarse - coarse at the prediction date)`. Where there is no
//! candidate the coarse value is kept. A pixel that is off is not predicted (its prediction is 0) and is not a
//! candidate.
//!
//! `resolution_merge_3` (`starfm::resolution_merge_3`): the fine pair is `((green + red) / 2, bnir)`; the coarse
//! one is the fine one convolved with a 5 x 5 Gaussian (sigma 0.5, the off pixels left out and the weights
//! normalised) and then averaged over the aligned 2 x 2 blocks; each of the six bands is the coarse image at the
//! prediction date and `starfm` with `r = 1` the prediction.
//!
//! What FORCE does that is not right, and what `force` means here. `force = true` is FORCE's code: the values are
//! `short` (here floats that hold integers), the predictions are truncated to integers, the pixels that are off
//! are written as 0, and in the 2 x 2 averaging the off test reads the pixel numbered as the block (a counter that
//! is incremented once per block, not `i * nx + j`, so FORCE looks at a pixel of the first rows for every block,
//! whatever its position: an evident bug). With `force` false the block's first pixel is tested, the predictions
//! are not truncated and the pixels that are off keep their
//! values. In both modes the clouds and shadows of the QAI are not looked at (FORCE's `resolution_merge_3` does
//! not, unlike the other two methods), so `Scene::bad` is ignored.

use crate::parallel::for_each_row;
use crate::resmerge::{Scene, NT};

/// `starfm`'s parameters: FORCE's are `radius` 1 (from `resolution_merge_3`), `nclass` 40 and every uncertainty 0.002.
#[derive(Clone, Copy, Debug)]
pub struct StarfmParams {
    /// Prediction radius in pixels.
    pub radius: usize,
    /// Number of spectral classes: the similarity threshold is `2 std / nclass`.
    pub nclass: f32,
    /// Uncertainty of the coarse images (FORCE's `unc_coarse_1` and `unc_coarse_2`).
    pub unc_coarse: f32,
    /// Uncertainty of the fine images.
    pub unc_fine: f32,
}

impl StarfmParams {
    /// FORCE's values with the radius of `resolution_merge_3`.
    pub const FORCE: StarfmParams = StarfmParams {
        radius: 1,
        nclass: 40.0,
        unc_coarse: 0.002,
        unc_fine: 0.002,
    };
}

/// FORCE's `starfm_candidate_t`.
#[derive(Clone, Copy, Default)]
struct Candidate {
    dx: i32,
    dy: i32,
    coarse: f32,
    fine: f32,
    r_coarse: f32,
    r_fine: f32,
    c: f32,
}

/// FORCE's `starfm`: the fine image at the prediction date from `fine.len()` pairs of coarse and fine images and
/// the coarse image at that date.
///
/// * `off`: no data (FORCE's `get_off`); not predicted (0) and not used.
/// * `coarse`: the coarse images, `(pairs + 1)` of `ny * nx`: the base pairs, then the one at the prediction date.
/// * `fine`: the fine images of the pairs, `pairs` of `ny * nx` (FORCE has two).
///
/// Returns the prediction `ny * nx`, row-major. The pixels run in parallel (each is independent of the others).
pub fn starfm(
    nx: usize,
    ny: usize,
    off: &[bool],
    coarse: &[&[f32]],
    fine: &[&[f32]],
    par: &StarfmParams,
    n_jobs: i32,
) -> Vec<f32> {
    let npair = fine.len();
    assert!(npair >= 1 && coarse.len() == npair + 1, "starfm: the coarse images are the pairs and one more");
    let npix = nx * ny;
    assert!(off.len() == npix && coarse.iter().chain(fine).all(|a| a.len() == npix));
    let r = par.radius as isize;

    // uncertainty
    let (unc_c1, unc_c2, unc_f) = (par.unc_coarse, par.unc_coarse, par.unc_fine);
    let unc_r_coarse = f64::from(unc_c1 * unc_c1 + unc_c2 * unc_c2).sqrt() as f32;
    let unc_r_fine = f64::from(unc_c1 * unc_c1 + unc_f * unc_f).sqrt() as f32;
    let unc_all = f64::from(unc_r_coarse * unc_r_coarse + unc_r_fine * unc_r_fine).sqrt() as f32;

    // slice values for the spectral similarity
    let mut slice_value = vec![0.0f32; npair];
    for pair in 0..npair {
        let (mut sumx, mut sumx2, mut num) = (0.0f64, 0.0f64, 0.0f64);
        for p in 0..npix {
            if off[p] {
                continue;
            }
            let v = fine[pair][p];
            sumx += f64::from(v);
            sumx2 += f64::from(v * v);
            num += 1.0;
        }
        let std = (sumx2 / num - (sumx / num) * (sumx / num)).sqrt() as f32;
        slice_value[pair] = (f64::from(std) * 2.0 / f64::from(par.nclass)) as f32;
    }

    let cap = (2 * par.radius + 1) * (2 * par.radius + 1) * npair + 1;
    let mut pred = vec![0.0f32; npix];
    if npix == 0 {
        return pred;
    }
    for_each_row(
        &mut pred,
        nx,
        n_jobs,
        || vec![Candidate::default(); cap + 1],
        |can, i, row| {
            for j in 0..nx {
                let p = i * nx + j;
                if off[p] {
                    continue;
                }
                // the coarse observation at the prediction date is candidate 0
                can[0].dx = 0;
                can[0].dy = 0;
                can[0].coarse = coarse[npair][p];
                // the base pairs of the central pixel, and the smallest relative differences
                let mut min_r_coarse = 32767.0f32;
                let mut min_r_fine = 32767.0f32;
                for pair in 0..npair {
                    let r_coarse = coarse[pair][p] - coarse[npair][p];
                    let r_fine = coarse[pair][p] - fine[pair][p];
                    if r_coarse.abs() < min_r_coarse {
                        min_r_coarse = r_coarse.abs();
                    }
                    if r_fine.abs() < min_r_fine {
                        min_r_fine = r_fine.abs();
                    }
                }
                // the base observations of the window that look like the pixel (the central pixel included)
                let mut c = 1usize;
                for ii in -r..=r {
                    for jj in -r..=r {
                        let ni = i as isize + ii;
                        let nj = j as isize + jj;
                        if ni < 0 || ni >= ny as isize || nj < 0 || nj >= nx as isize {
                            continue;
                        }
                        let np = ni as usize * nx + nj as usize;
                        if off[np] {
                            continue;
                        }
                        for pair in 0..npair {
                            let diff = (fine[pair][np] - fine[pair][p]).abs();
                            if diff > slice_value[pair] {
                                continue;
                            }
                            let cand = &mut can[c];
                            cand.dx = jj as i32;
                            cand.dy = ii as i32;
                            cand.coarse = coarse[pair][np];
                            cand.fine = fine[pair][np];
                            cand.r_coarse = coarse[pair][np] - coarse[npair][np];
                            cand.r_fine = coarse[pair][np] - fine[pair][np];
                            if cand.r_coarse.abs() < (min_r_coarse + unc_r_coarse)
                                || cand.r_fine.abs() < (min_r_fine + unc_r_fine)
                            {
                                c += 1;
                            }
                        }
                    }
                }
                let ncan = c;
                if ncan > 1 {
                    let mut c_sum = 0.0f32;
                    for cand in &mut can[1..ncan] {
                        // distance to the central pixel
                        let d = f64::from(cand.dx * cand.dx + cand.dy * cand.dy).sqrt() as f32;
                        // temporal and spectral differences, with a small uncertainty to avoid a zero
                        let t = (f64::from(cand.r_coarse.abs()) + 0.0001) as f32;
                        let s = (f64::from(cand.r_fine.abs()) + 0.0001) as f32;
                        let ts = t * s;
                        if ts < unc_all {
                            cand.c = 1.0;
                        } else {
                            let dd = (1.0f64 + f64::from(d / r as f32)) as f32;
                            cand.c = (1.0f64 / f64::from(ts * dd)) as f32;
                        }
                        c_sum += cand.c;
                    }
                    let mut v = 0.0f32;
                    for cand in &can[1..ncan] {
                        let w = cand.c / c_sum;
                        v += w * (cand.fine - cand.r_coarse);
                    }
                    row[j] = v;
                } else {
                    // no suitable candidate: the coarse data
                    row[j] = can[0].coarse;
                }
            }
        },
    );
    pred
}

/// FORCE's `gauss_kernel(nk, sigma)`: `nk` x `nk` weights, row-major.
fn gauss_kernel(nk: usize, sigma: f32) -> Vec<f32> {
    let sig = sigma * sigma * 2.0;
    let half = -((nk as i32 - 1) / 2);
    let mut k = vec![0.0f32; nk * nk];
    for i in 0..nk {
        for j in 0..nk {
            let hx = (half + j as i32) as f32;
            let hy = (half + i as i32) as f32;
            let e = f64::from(-(hx * hx + hy * hy) / sig).exp();
            k[i * nk + j] = (1.0 / (std::f64::consts::PI * f64::from(sig)) * e) as f32;
        }
    }
    k
}

/// `(short)(v)` of C on x86: the int conversion truncated to 16 bits.
#[inline]
fn to_short(v: f32) -> f32 {
    (v as i32 as i16) as f32
}

/// FORCE's `resolution_merge_3`: the six bands at 10 m by STARFM on the pseudo-PAN and the broad NIR. `denom` is what
/// the values are divided by to be reflectance (10000 for FORCE's). Returns `(NT, ny * nx)`: the prediction at the
/// pixels that are not off (clouds and shadows are predicted too, as FORCE does), and with `force` 0 at the others
/// (FORCE writes `(short)0`), else their input.
pub fn resolution_merge_3(sc: &Scene, denom: f64, force: bool, n_jobs: i32) -> Vec<f32> {
    let (nx, ny) = (sc.nx, sc.ny);
    let npix = nx * ny;
    let mut out = sc.targets.to_vec();
    if npix == 0 {
        return out;
    }
    let off = sc.off;
    const NK: usize = 5;

    // fine pairs: (green + red) / 2 and bnir, as doubles stored to floats
    let mut fine0 = vec![0.0f32; npix];
    let mut fine1 = vec![0.0f32; npix];
    for p in 0..npix {
        fine0[p] = ((f64::from(sc.green[p]) / denom + f64::from(sc.red[p]) / denom) / 2.0) as f32;
        fine1[p] = (f64::from(sc.bnir[p]) / denom) as f32;
    }

    // convolve with the Gaussian kernel (the pixels that are off left out, the weights normalised)
    let kernel = gauss_kernel(NK, 0.5);
    let mut gauss = vec![0.0f32; npix * 2];
    for_each_row(
        &mut gauss,
        nx * 2,
        n_jobs,
        || (),
        |_, i, row| {
            for j in 0..nx {
                let p = i * nx + j;
                if off[p] {
                    continue;
                }
                let (mut s0, mut s1, mut num) = (0.0f64, 0.0f64, 0.0f64);
                for ii in 0..NK {
                    for jj in 0..NK {
                        let ni = ii as isize - ((NK as isize - 1) / 2) + i as isize;
                        let nj = jj as isize - ((NK as isize - 1) / 2) + j as isize;
                        if ni < 0 || ni >= ny as isize || nj < 0 || nj >= nx as isize {
                            continue;
                        }
                        let np = ni as usize * nx + nj as usize;
                        if off[np] {
                            continue;
                        }
                        let k = f64::from(kernel[ii * NK + jj]);
                        s0 += (f64::from(sc.green[np]) / denom + f64::from(sc.red[np]) / denom) / 2.0 * k;
                        s1 += f64::from(sc.bnir[np]) / denom * k;
                        num += k;
                    }
                }
                if num > 0.0 {
                    row[j * 2] = (s0 / num) as f32;
                    row[j * 2 + 1] = (s1 / num) as f32;
                }
            }
        },
    );
    let mut coarse0 = vec![0.0f32; npix];
    let mut coarse1 = vec![0.0f32; npix];
    for p in 0..npix {
        coarse0[p] = gauss[p * 2];
        coarse1[p] = gauss[p * 2 + 1];
    }
    drop(gauss);

    // reduce the resolution, supersample to the target one: the mean of each aligned 2 x 2 block
    let mut blk = 0usize;
    for i in (0..ny).step_by(2) {
        for j in (0..nx).step_by(2) {
            // FORCE tests the pixel numbered as the block (its `p` is incremented once per block)
            let ptest = if force { blk } else { i * nx + j };
            blk += 1;
            if off[ptest] {
                continue;
            }
            let (mut s0, mut s1, mut num) = (0.0f64, 0.0f64, 0.0f64);
            for ii in 0..2 {
                for jj in 0..2 {
                    let (ni, nj) = (i + ii, j + jj);
                    if ni >= ny || nj >= nx {
                        continue;
                    }
                    let np = ni * nx + nj;
                    if off[np] {
                        continue;
                    }
                    s0 += f64::from(coarse0[np]);
                    s1 += f64::from(coarse1[np]);
                    num += 1.0;
                }
            }
            if num > 0.0 {
                for ii in 0..2 {
                    for jj in 0..2 {
                        let (ni, nj) = (i + ii, j + jj);
                        if ni >= ny || nj >= nx {
                            continue;
                        }
                        let np = ni * nx + nj;
                        if off[np] {
                            continue;
                        }
                        coarse0[np] = (s0 / num) as f32;
                        coarse1[np] = (s1 / num) as f32;
                    }
                }
            }
        }
    }

    // predict at the fine resolution, band by band
    let mut at_date = vec![0.0f32; npix];
    for b in 0..NT {
        for p in 0..npix {
            at_date[p] = (f64::from(sc.targets[b * npix + p]) / denom) as f32;
        }
        let pred = starfm(
            nx,
            ny,
            off,
            &[&coarse0, &coarse1, &at_date],
            &[&fine0, &fine1],
            &StarfmParams::FORCE,
            n_jobs,
        );
        for p in 0..npix {
            out[b * npix + p] = if force {
                to_short(pred[p] * 10000.0)
            } else if off[p] {
                sc.targets[b * npix + p]
            } else {
                (f64::from(pred[p]) * denom) as f32
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: f32, n: usize) -> Vec<f32> {
        vec![v; n]
    }

    #[test]
    fn gauss_kernel_is_symmetric_and_peaks_at_the_centre() {
        let k = gauss_kernel(5, 0.5);
        assert_eq!(k[12], k.iter().cloned().fold(0.0, f32::max));
        for i in 0..5 {
            for j in 0..5 {
                assert_eq!(k[i * 5 + j], k[j * 5 + i]);
                assert_eq!(k[i * 5 + j], k[(4 - i) * 5 + (4 - j)]);
            }
        }
        // 1 / (pi * 0.5) * exp(0)
        assert!((f64::from(k[12]) - 1.0 / (std::f64::consts::PI * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn uniform_window_gives_fine_plus_the_temporal_change() {
        // both pairs: fine 0.2, coarse 0.3; the coarse image at the date is 0.35
        let (nx, ny) = (4, 3);
        let n = nx * ny;
        let (c1, c2, c3) = (flat(0.3, n), flat(0.3, n), flat(0.35, n));
        let (f1, f2) = (flat(0.2, n), flat(0.2, n));
        let off = vec![false; n];
        let out = starfm(nx, ny, &off, &[&c1, &c2, &c3], &[&f1, &f2], &StarfmParams::FORCE, 2);
        for v in out {
            assert!((v - 0.25).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn off_pixels_are_not_predicted_nor_candidates() {
        let (nx, ny) = (3, 3);
        let n = nx * ny;
        let mut c1 = flat(0.3, n);
        let mut f1 = flat(0.2, n);
        let c3 = flat(0.3, n);
        let mut off = vec![false; n];
        // an off pixel in the middle with absurd values
        off[4] = true;
        c1[4] = 99.0;
        f1[4] = -99.0;
        let out = starfm(nx, ny, &off, &[&c1, &c3], &[&f1], &StarfmParams::FORCE, 1);
        assert_eq!(out[4], 0.0);
        for (p, v) in out.iter().enumerate() {
            if p != 4 {
                assert!((v - 0.2).abs() < 1e-6, "{p}: {v}");
            }
        }
    }

    #[test]
    fn no_candidate_keeps_the_coarse_value() {
        // a coarse value that is NaN gives NaN relative differences: the central pixel is no candidate
        let c1 = [f32::NAN];
        let c2 = [f32::NAN];
        let c3 = [0.42f32];
        let f1 = [0.1f32];
        let f2 = [0.1f32];
        let out = starfm(1, 1, &[false], &[&c1, &c2, &c3], &[&f1, &f2], &StarfmParams::FORCE, 1);
        assert_eq!(out[0], 0.42);
    }

    #[test]
    fn the_weights_follow_the_temporal_and_spectral_differences() {
        // pair 1: coarse 0.30, fine 0.20; pair 2: coarse 0.50, fine 0.45; the coarse image of the date is 0.30
        let (c1, f1) = ([0.30f32], [0.20f32]);
        let (c2, f2) = ([0.50f32], [0.45f32]);
        let c3 = [0.30f32];
        let out = starfm(1, 1, &[false], &[&c1, &c2, &c3], &[&f1, &f2], &StarfmParams::FORCE, 1);
        // pair 1: T S = 0.0001 x 0.1001 is under the combined uncertainty (0.0049): weight 1, prediction 0.20;
        // pair 2: T = 0.2001, S = 0.0501, weight 1 / (T S), prediction 0.45 - 0.2 = 0.25
        let w2 = 1.0 / (0.2001f64 * 0.0501);
        let expected = (0.20 + w2 * 0.25) / (1.0 + w2);
        assert!((f64::from(out[0]) - expected).abs() < 1e-5, "{} {expected}", out[0]);
    }

    #[test]
    fn dissimilar_neighbours_are_left_out() {
        // a 1 x 5 row, fine 0.2 except the last at 0.9 (very different), whose coarse is far off too
        let (nx, ny) = (5, 1);
        let c1 = [0.3, 0.3, 0.3, 0.3, 5.0];
        let f1 = [0.2, 0.2, 0.2, 0.2, 0.9];
        let c3 = [0.3, 0.3, 0.3, 0.3, 5.0];
        let off = [false; 5];
        let out = starfm(nx, ny, &off, &[&c1, &c3], &[&f1], &StarfmParams { radius: 2, ..StarfmParams::FORCE }, 1);
        assert!((out[2] - 0.2).abs() < 1e-6, "{}", out[2]);
    }

    fn scene<'a>(
        nx: usize,
        ny: usize,
        g: &'a [f32],
        r: &'a [f32],
        n: &'a [f32],
        t: &'a [f32],
        off: &'a [bool],
        bad: &'a [bool],
    ) -> Scene<'a> {
        Scene { nx, ny, green: g, red: r, bnir: n, targets: t, off, bad }
    }

    fn merge_inputs(ny: usize, nx: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let npix = ny * nx;
        let mut g = vec![0.0f32; npix];
        let mut r = vec![0.0f32; npix];
        let mut n = vec![0.0f32; npix];
        let mut t = vec![0.0f32; NT * npix];
        for i in 0..ny {
            for j in 0..nx {
                let p = i * nx + j;
                let class = if j < nx / 2 { 0.0 } else { 1.0 };
                g[p] = (800.0 + 1500.0 * class + 3.0 * ((i * 7 + j * 3) % 5) as f64) as f32;
                r[p] = (700.0 + 1400.0 * class + 2.0 * ((i * 5 + j) % 4) as f64) as f32;
                n[p] = (3000.0 - 1500.0 * class + 4.0 * ((i + j * 3) % 6) as f64) as f32;
                for b in 0..NT {
                    // the 20 m bands: blocks of 2 x 2
                    let v = 1000.0 + 300.0 * b as f64 + 900.0 * class + 10.0 * (((i / 2) * 3 + (j / 2)) % 4) as f64;
                    t[b * npix + p] = v as f32;
                }
            }
        }
        (g, r, n, t)
    }

    #[test]
    fn merge3_force_gives_integers_and_zero_at_off_pixels() {
        let (ny, nx) = (12, 14);
        let npix = ny * nx;
        let (g, r, n, mut t) = merge_inputs(ny, nx);
        let mut off = vec![false; npix];
        off[3 * nx + 4] = true;
        for b in 0..NT {
            t[b * npix + 3 * nx + 4] = f32::NAN;
        }
        let bad = vec![false; npix];
        let sc = scene(nx, ny, &g, &r, &n, &t, &off, &bad);
        let forced = resolution_merge_3(&sc, 10000.0, true, 2);
        assert!(forced.iter().all(|v| *v == v.trunc()));
        assert_eq!(forced[3 * nx + 4], 0.0);
        let free = resolution_merge_3(&sc, 10000.0, false, 2);
        assert!(free[3 * nx + 4].is_nan());
        for b in 0..NT {
            for p in 0..npix {
                if !off[p] {
                    let (a, c) = (forced[b * npix + p], free[b * npix + p]);
                    assert!((a - c).abs() <= 1.0, "{a} {c}");
                }
            }
        }
        // with one thread, as with four
        let one = resolution_merge_3(&sc, 10000.0, true, 1);
        assert_eq!(one[..8], forced[..8]);
        assert!(one.iter().zip(&forced).all(|(a, b)| a == b));
    }

    #[test]
    fn merge3_ignores_the_cloud_flags_and_a_constant_scene_is_kept() {
        let (ny, nx) = (6, 6);
        let npix = ny * nx;
        let g = flat(900.0, npix);
        let r = flat(800.0, npix);
        let n = flat(2900.0, npix);
        let t = flat(1500.0, NT * npix);
        let off = vec![false; npix];
        let mut bad = vec![false; npix];
        bad[7] = true;
        let sc = scene(nx, ny, &g, &r, &n, &t, &off, &bad);
        let out = resolution_merge_3(&sc, 10000.0, false, 1);
        for v in out {
            assert!((v - 1500.0).abs() < 0.05, "{v}");
        }
    }
}
