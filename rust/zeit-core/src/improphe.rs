//! ImproPhe: sharpening a coarse field (or the bands of a coarser sensor) to the grid of a finer sensor, from
//! the pixels around each one that look alike in the fine data (Frantz, Stellmes, Roeder, Udelhoven, Mader and
//! Hill 2016, "Improving the Spatial Resolution of Land Surface Phenology by Fusing Medium- and Coarse-Resolution
//! Inputs", IEEE TGRS 54(7):4153-4164, doi 10.1109/TGRS.2016.2537929).
//!
//! Ported from FORCE 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring,
//! Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force), src/modules:
//! higher-level/improphe-hl.c (`improphe`, `average_season`, `standardize_float`, `focal_sd`, `rescale_weight`),
//! cross-level/pca-cl.c (`pca`), cross-level/stats-cl.c (the Welford recurrences) and the way
//! higher-level/cf-improphe-hl.c and l2-improphe-hl.c feed them. The eigenvectors of FORCE's mode are those of
//! GSL 2.7's `gsl_eigen_symmv` (gsl_symmv.rs).
//!
//! The method, as the code does it. Every input is on the grid of the fine sensor (the caller resamples the
//! coarse one by nearest neighbour, as FORCE's GDAL read does).
//!
//! 1. `average_season`: the fine scenes of a year (CF-ImproPhe) or of all the years (L2-ImproPhe) are averaged
//!    in seasonal windows of the day of the year (`dwin`), a window with no scene taking the mean of all scenes.
//! 2. `pca`: the windows' means, `nwin * nb` bands, are rotated and truncated to the components that hold 97.5%
//!    of the variance; `standardize` gives each component zero mean and unit variance; `focal_sd` is the texture
//!    of the fine data: the largest standard deviation of the components in a (2 ksd + 1)^2 window.
//! 3. `focal_sd` of each coarse band is its texture.
//! 4. For each pixel, `improphe_grid` looks at the pixels in a disc of radius `h`: those whose mean absolute
//!    difference of the components to the pixel is under one of 0.025, 0.05, 0.1, 0.2, 0.4 are neighbours, the
//!    pixel itself first. The cutoff used is the smallest with at least `mink` neighbours (0.5% of the disc).
//!    The prediction of each coarse band is the mean of that band at the neighbours of the cutoff, weighted by
//!    `1 / (SS * TT * UU)`: SS the spectral difference, TT the neighbour's fine texture and UU its coarse
//!    texture, each rescaled over the neighbours to `1 + exp(25 x - 7.5)`, x in [0, 1].
//!
//! Two modes (`force`). `force = true` is FORCE's arithmetic to the letter, quirks included: the upper end of
//! every seasonal window excluded (day 365 in no window of the usual `1 91 181 271 365`), the seasonal means
//! truncated towards zero to integers (FORCE keeps `short`), and the projection on the TRANSPOSE of GSL's
//! eigenvector matrix: the component `b` is the data times the ROW `b` of the matrix that holds the
//! eigenvectors in its columns (`gsl_blas_dgemm(NoTrans, Trans, GINP, evec)`), which is not the principal
//! component `b`, and the signs of GSL's eigenvectors matter to the L1 distances. `force = false` is the paper's:
//! the last window includes its upper end, the means are kept as floats and the projection is on the
//! eigenvectors (faer's self-adjoint eigendecomposition; any solver gives the same distances, a sign of an
//! eigenvector does not change an absolute difference).
//!
//! Where the code departs from FORCE in both modes, on purpose: a NaN is the nodata (FORCE compares with
//! -9999), the output is a float (FORCE writes `(short)`, a truncation), and a pixel whose coarse value is NaN
//! is NaN (FORCE leaves it unwritten, 0 in some files). Where FORCE's result is `short` truncation of the sums
//! the caller truncates.
//!
//! Speed, with the same results as the literal loop: two passes over the disc. The first finds the cutoff,
//! keeping only the neighbours (index, exact distance, class); the weights, with their `exp`, are computed on
//! those alone, in the order of FORCE's scan (rows from the top, columns from the left). The distances of a row
//! of the disc are summed band by band over contiguous memory, in the same order per pixel, and the row extents
//! of the disc are computed once with the float test FORCE uses.

use crate::gsl_symmv;
use crate::parallel::{pool, threads, Shared};
use faer::{Mat, Side};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The cutoffs of the mean absolute difference, FORCE's `float Sthr[5]` promoted to double.
fn cutoffs() -> [f64; 5] {
    [0.025f32, 0.05, 0.1, 0.2, 0.4].map(f64::from)
}

/// `f(&mut state, i)` for `i` in `0..n` on the threads of `n_jobs`, one `state` per task.
fn parallel_for<S, I, F>(n: usize, n_jobs: i32, init: I, f: F)
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) + Sync + Send,
{
    let t = threads(n_jobs).min(n.max(1));
    if t <= 1 {
        let mut s = init();
        for i in 0..n {
            f(&mut s, i);
        }
    } else {
        pool(t).install(|| {
            (0..n)
                .into_par_iter()
                .with_min_len(1)
                .for_each_init(&init, |s, i| f(s, i));
        });
    }
}

/// The minimum number of neighbours that makes a prediction, `mink` of FORCE: 0.5% of the disc of radius `h`,
/// from the square's `nk = (2 h + 1)^2`: `(int)((nk * pi / 4) * 0.5 / 100)`.
pub fn min_neighbors(h: usize) -> usize {
    let nk = (2 * h + 1) * (2 * h + 1);
    ((nk as f64 * std::f64::consts::PI / 4.0) * 0.5 / 100.0) as usize
}

/// FORCE's `rescale_weight`: 1 for a zero or a degenerate range, else `1 + exp(25 (w - mn) / (mx - mn) - 7.5)`.
pub fn rescale_weight(weight: f64, minweight: f64, maxweight: f64) -> f64 {
    if weight == 0.0 || minweight == maxweight {
        1.0
    } else {
        1.0 + (25.0 * ((weight - minweight) / (maxweight - minweight)) - 7.5).exp()
    }
}

// ---------------------------------------------------------------------------------------------------------------
// seasonal means

/// The seasonal window of a scene of `year` and day of the year `doy` (a 365-day calendar, 29 February on 60),
/// the first of `dwin.len() - 1` windows that takes it, as `average_season` of FORCE tests them: a window
/// `dwin[w] < dwin[w + 1]` takes `dwin[w] <= doy < dwin[w + 1]` of the year `ywin`; one that crosses the year
/// (`dwin[w] > dwin[w + 1]`) takes the days from `dwin[w]` to 365 of the year before `ywin` and the days before
/// `dwin[w + 1]` of `ywin` (tested first), or those of `ywin` and of the year after (tested second). `ywin =
/// None` is a window of any year (FORCE's -1). With `force` false the last window includes its upper end.
pub fn season_window(doy: i32, year: i32, dwin: &[i32], ywin: Option<i32>, force: bool) -> Option<usize> {
    let nwin = dwin.len().saturating_sub(1);
    let any = ywin.is_none();
    let y = ywin.unwrap_or(0);
    for w in 0..nwin {
        let (lo, hi) = (dwin[w], dwin[w + 1]);
        if lo < hi {
            let upper = if !force && w + 1 == nwin { doy <= hi } else { doy < hi };
            if doy >= lo && upper && (year == y || any) {
                return Some(w);
            }
        }
        if lo > hi {
            if (doy >= lo && doy <= 365 && (year == y - 1 || any)) || (doy < hi && (year == y || any)) {
                return Some(w);
            }
            if (doy >= lo && doy <= 365 && (year == y || any)) || (doy < hi && (year == y + 1 || any)) {
                return Some(w);
            }
        }
    }
    None
}

/// Pixels per job of the seasonal means.
const SEASON_TILE: usize = 512;

/// FORCE's `average_season`: the mean of the scenes of `x` (`nt` scenes, `nb` bands, `npix` pixels, C order) in
/// each seasonal window of `dwin`, into `out` (`(nwin * nb, npix)`, window major: row `w * nb + b`; every value
/// written). A scene counts at a pixel when all its bands are finite. A window with none takes the mean of all
/// the scenes counted, a pixel with none is NaN in all the rows. With `force` the means are truncated towards
/// zero (FORCE's `(short)`), the upper end of the last window is excluded and `year`, `doy` are as FORCE's
/// calendar; see `season_window`. Returns the number of pixels with any scene.
#[allow(clippy::too_many_arguments)]
pub fn average_season(
    x: &[f32],
    nt: usize,
    nb: usize,
    npix: usize,
    doy: &[i32],
    year: &[i32],
    dwin: &[i32],
    ywin: Option<i32>,
    force: bool,
    out: &mut [f32],
    n_jobs: i32,
) -> usize {
    assert!(dwin.len() >= 2, "at least two window limits");
    let nwin = dwin.len() - 1;
    assert_eq!(x.len(), nt * nb * npix);
    assert_eq!(doy.len(), nt);
    assert_eq!(year.len(), nt);
    assert_eq!(out.len(), nwin * nb * npix);
    if npix == 0 {
        return 0;
    }
    let win: Vec<Option<usize>> = (0..nt)
        .map(|t| season_window(doy[t], year[t], dwin, ywin, force))
        .collect();
    let n_valid = AtomicUsize::new(0);
    let shared = Shared::new(out);
    let tiles = npix.div_ceil(SEASON_TILE);
    let tt = SEASON_TILE;
    parallel_for(
        tiles,
        n_jobs,
        || (vec![0f64; (nwin + 1) * nb * tt], vec![0u32; (nwin + 1) * tt]),
        |(sum, k), tile| {
            let p0 = tile * tt;
            let n = (npix - p0).min(tt);
            sum.iter_mut().for_each(|v| *v = 0.0);
            k.iter_mut().for_each(|v| *v = 0);
            for t in 0..nt {
                let Some(w) = win[t] else { continue };
                // a scene counts at the pixels where all its bands are finite
                let mut valid = [false; SEASON_TILE];
                valid[..n].iter_mut().for_each(|v| *v = true);
                for b in 0..nb {
                    for (v, &e) in valid[..n].iter_mut().zip(&x[(t * nb + b) * npix + p0..][..n]) {
                        *v &= e.is_finite();
                    }
                }
                for p in 0..n {
                    let c = u32::from(valid[p]);
                    k[w * tt + p] += c;
                    k[nwin * tt + p] += c;
                }
                for b in 0..nb {
                    let plane = &x[(t * nb + b) * npix + p0..][..n];
                    let (sw, sa) = {
                        let (head, tail) = sum.split_at_mut(nwin * nb * tt);
                        (&mut head[(w * nb + b) * tt..][..n], &mut tail[b * tt..][..n])
                    };
                    for p in 0..n {
                        // adding 0.0 leaves a sum as it is
                        let v = if valid[p] { f64::from(plane[p]) } else { 0.0 };
                        sw[p] += v;
                        sa[p] += v;
                    }
                }
            }
            let mut any = 0;
            for w in 0..nwin {
                for b in 0..nb {
                    // SAFETY: each tile writes its own pixels of every row.
                    let row = unsafe { shared.slice_mut(((w * nb + b) * npix + p0)..((w * nb + b) * npix + p0 + n)) };
                    for p in 0..n {
                        row[p] = if k[nwin * tt + p] == 0 {
                            f32::NAN
                        } else {
                            // the window's mean, or the overall mean when the window has no scene
                            let (s, c) = if k[w * tt + p] > 0 {
                                (sum[(w * nb + b) * tt + p], k[w * tt + p])
                            } else {
                                (sum[(nwin * nb + b) * tt + p], k[nwin * tt + p])
                            };
                            let m = s / f64::from(c);
                            if force {
                                (m as i16) as f32
                            } else {
                                m as f32
                            }
                        };
                    }
                }
            }
            for p in 0..n {
                if k[nwin * tt + p] > 0 {
                    any += 1;
                }
            }
            n_valid.fetch_add(any, Ordering::Relaxed);
        },
    );
    n_valid.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------------------------------------------
// principal components

/// The principal components of the seasonal means.
pub struct Pca {
    /// Number of components kept.
    pub npc: usize,
    /// Pixels with all the bands finite.
    pub n_valid: usize,
    /// The components `(npc, npix)`, NaN where a band is not finite.
    pub pcs: Vec<f32>,
    /// All the eigenvalues of the covariance matrix, descending.
    pub eigenvalues: Vec<f64>,
}

/// Valid pixels: every band finite.
fn finite_everywhere(x: &[f32], nb: usize, npix: usize) -> Vec<bool> {
    let mut ok = vec![true; npix];
    for b in 0..nb {
        for (o, v) in ok.iter_mut().zip(&x[b * npix..(b + 1) * npix]) {
            *o &= v.is_finite();
        }
    }
    ok
}

/// FORCE's `pca`: the PCs of `x` `(nb, npix)`, from the covariance matrix of the pixels with all the bands finite
/// (bands centred on their means, `X^T X / (nk - 1)`, each sum in the order of the pixels, in double), keeping
/// the first components whose cumulative variance, in float arithmetic as FORCE's, exceeds `min_var` (all of
/// them if `min_var >= 1`). With `force` the eigenvectors are GSL's and the projection is FORCE's transposed
/// one (see the module); else faer's and the projection is on the eigenvectors. Fewer than two valid pixels:
/// no components.
pub fn pca(x: &[f32], nb: usize, npix: usize, min_var: f32, force: bool, n_jobs: i32) -> Pca {
    assert_eq!(x.len(), nb * npix);
    let ok = finite_everywhere(x, nb, npix);
    let valid: Vec<u32> = (0..npix).filter(|&p| ok[p]).map(|p| p as u32).collect();
    let nk = valid.len();
    if nk < 2 || nb == 0 {
        return Pca {
            npc: 0,
            n_valid: nk,
            pcs: Vec::new(),
            eigenvalues: Vec::new(),
        };
    }

    // the means, band by band, over the valid pixels in order
    let mut mean = vec![0f64; nb];
    {
        let sh = Shared::new(&mut mean);
        parallel_for(
            nb,
            n_jobs,
            || (),
            |_, b| {
                let plane = &x[b * npix..(b + 1) * npix];
                let mut s = 0.0f64;
                for &p in &valid {
                    s += f64::from(plane[p as usize]);
                }
                // SAFETY: one writer per band.
                unsafe { sh.slice_mut(b..b + 1)[0] = s / nk as f64 };
            },
        );
    }

    // X^T X, the lower triangle: a sum over the pixels in order for each pair, streamed in chunks of pixels
    const CHUNK: usize = 2048;
    let mut cov = vec![0f64; nb * nb];
    {
        let sh = Shared::new(&mut cov);
        parallel_for(
            nb,
            n_jobs,
            || vec![0f64; nb * CHUNK],
            |buf, rev| {
                let i = nb - 1 - rev; // the long rows first
                let mut acc = vec![0f64; i + 1];
                for chunk in valid.chunks(CHUNK) {
                    let n = chunk.len();
                    for b in 0..=i {
                        let plane = &x[b * npix..(b + 1) * npix];
                        let m = mean[b];
                        for (o, &p) in buf[b * CHUNK..b * CHUNK + n].iter_mut().zip(chunk) {
                            *o = f64::from(plane[p as usize]) - m;
                        }
                    }
                    let bi = &buf[i * CHUNK..i * CHUNK + n];
                    let mut j = 0;
                    while j + 4 <= i + 1 {
                        let (b0, b1, b2, b3) = (
                            &buf[j * CHUNK..j * CHUNK + n],
                            &buf[(j + 1) * CHUNK..(j + 1) * CHUNK + n],
                            &buf[(j + 2) * CHUNK..(j + 2) * CHUNK + n],
                            &buf[(j + 3) * CHUNK..(j + 3) * CHUNK + n],
                        );
                        let (mut a0, mut a1, mut a2, mut a3) = (acc[j], acc[j + 1], acc[j + 2], acc[j + 3]);
                        for k in 0..n {
                            let v = bi[k];
                            a0 += v * b0[k];
                            a1 += v * b1[k];
                            a2 += v * b2[k];
                            a3 += v * b3[k];
                        }
                        acc[j] = a0;
                        acc[j + 1] = a1;
                        acc[j + 2] = a2;
                        acc[j + 3] = a3;
                        j += 4;
                    }
                    while j <= i {
                        let bj = &buf[j * CHUNK..j * CHUNK + n];
                        let mut a = acc[j];
                        for k in 0..n {
                            a += bi[k] * bj[k];
                        }
                        acc[j] = a;
                        j += 1;
                    }
                }
                // SAFETY: one writer per row.
                let row = unsafe { sh.slice_mut(i * nb..i * nb + i + 1) };
                row.copy_from_slice(&acc);
            },
        );
    }
    let scale = 1.0 / (nk - 1) as f64;
    for i in 0..nb {
        for j in 0..=i {
            cov[i * nb + j] *= scale;
            cov[j * nb + i] = cov[i * nb + j];
        }
    }

    // eigenvalues descending, eigenvectors in the columns of the row-major matrix `evec`
    let (eval, evec) = if force {
        gsl_symmv::eigen_symmv_desc(&mut cov, nb)
    } else {
        let m = Mat::from_fn(nb, nb, |i, j| cov[i * nb + j]);
        let e = m
            .self_adjoint_eigen(Side::Lower)
            .expect("eigendecomposition did not converge");
        let s = e.S().column_vector();
        let u = e.U();
        let mut eval = vec![0f64; nb];
        let mut evec = vec![0f64; nb * nb];
        for c in 0..nb {
            let src = nb - 1 - c; // faer's are ascending
            eval[c] = s[src];
            for r in 0..nb {
                evec[r * nb + c] = u[(r, src)];
            }
        }
        (eval, evec)
    };

    // how many components: in float, as FORCE
    let mut numcomp = nb;
    if min_var < 1.0 {
        let mut total = 0f32;
        let mut cum = 0f32;
        for &e in &eval {
            total = (f64::from(total) + e) as f32;
        }
        for (b, &e) in eval.iter().enumerate() {
            cum = (f64::from(cum) + e) as f32;
            let pct = cum / total;
            if pct > min_var {
                numcomp = b + 1;
                break;
            }
        }
    }

    // component b = sum_j centred_j * proj[b][j]: FORCE uses the row b of evec, the paper the column b
    let mut proj = vec![0f64; numcomp * nb];
    for b in 0..numcomp {
        for j in 0..nb {
            proj[b * nb + j] = if force { evec[b * nb + j] } else { evec[j * nb + b] };
        }
    }
    let mut pcs = vec![0f32; numcomp * npix];
    {
        const TILE: usize = 1024;
        let sh = Shared::new(&mut pcs);
        parallel_for(
            npix.div_ceil(TILE),
            n_jobs,
            || (vec![0f64; nb * TILE], vec![0f64; TILE]),
            |(cen, acc), tile| {
                let p0 = tile * TILE;
                let n = (npix - p0).min(TILE);
                for j in 0..nb {
                    let plane = &x[j * npix + p0..j * npix + p0 + n];
                    for (o, &v) in cen[j * TILE..j * TILE + n].iter_mut().zip(plane) {
                        *o = f64::from(v) - mean[j];
                    }
                }
                for b in 0..numcomp {
                    acc[..n].iter_mut().for_each(|a| *a = 0.0);
                    for j in 0..nb {
                        let w = proj[b * nb + j];
                        for (a, &c) in acc[..n].iter_mut().zip(&cen[j * TILE..j * TILE + n]) {
                            *a += c * w;
                        }
                    }
                    // SAFETY: each tile writes its own pixels of every component.
                    let row = unsafe { sh.slice_mut(b * npix + p0..b * npix + p0 + n) };
                    for p in 0..n {
                        row[p] = if ok[p0 + p] { acc[p] as f32 } else { f32::NAN };
                    }
                }
            },
        );
    }
    Pca {
        npc: numcomp,
        n_valid: nk,
        pcs,
        eigenvalues: eval,
    }
}

/// FORCE's `standardize_float` on each plane of `pcs` `(npc, npix)`: mean and sample variance by the Welford
/// recurrence in double over the non-NaN values, then `(x - (float)mean) / (float)sd` in float arithmetic.
pub fn standardize(pcs: &mut [f32], npix: usize, n_jobs: i32) {
    if npix == 0 {
        return;
    }
    let t = threads(n_jobs);
    let work = |plane: &mut [f32]| {
        let (mut avg, mut var, mut k) = (0.0f64, 0.0f64, 0.0f64);
        for &v in plane.iter() {
            if v.is_nan() {
                continue;
            }
            k += 1.0;
            let dat = f64::from(v);
            if k == 1.0 {
                avg = dat;
            } else {
                let (oldmx, oldvx) = (avg, var);
                avg = oldmx + (dat - oldmx) / k;
                var = oldvx + (dat - oldmx) * (dat - avg);
            }
        }
        let sd = (var / (k - 1.0)).sqrt();
        let (a, s) = (avg as f32, sd as f32);
        for v in plane.iter_mut() {
            if !v.is_nan() {
                *v = (*v - a) / s;
            }
        }
    };
    if t <= 1 {
        pcs.chunks_mut(npix).for_each(work);
    } else {
        pool(t).install(|| pcs.par_chunks_mut(npix).for_each(work));
    }
}

/// FORCE's `focal_sd` over the planes `bands` of `data` `(nplanes, ny, nx)`: the standard deviation of the
/// non-NaN values in the square window of radius `k` (the pixel included, the image's edge clipped), sample
/// (n - 1) by Welford in double, scanned from the top row and left column; the largest over the bands, 0 where
/// a window has at most one value.
pub fn focal_sd(
    data: &[f32],
    ny: usize,
    nx: usize,
    k: usize,
    bands: std::ops::Range<usize>,
    out: &mut [f32],
    n_jobs: i32,
) {
    let npix = ny * nx;
    assert!(data.len() >= bands.end * npix);
    assert_eq!(out.len(), npix);
    crate::parallel::for_each_row(
        out,
        nx,
        n_jobs,
        || (),
        |_, i, row| {
            let (i0, i1) = (i.saturating_sub(k), (i + k).min(ny - 1));
            for (j, o) in row.iter_mut().enumerate() {
                let (j0, j1) = (j.saturating_sub(k), (j + k).min(nx - 1));
                let mut sd_max = 0.0f64;
                for b in bands.clone() {
                    let plane = &data[b * npix..(b + 1) * npix];
                    let (mut avg, mut var, mut n) = (0.0f64, 0.0f64, 0.0f64);
                    for ni in i0..=i1 {
                        for &v in &plane[ni * nx + j0..=ni * nx + j1] {
                            if v.is_nan() {
                                continue;
                            }
                            let dat = f64::from(v);
                            n += 1.0;
                            if n == 1.0 {
                                avg = dat;
                            } else {
                                let (oldmx, oldvx) = (avg, var);
                                avg = oldmx + (dat - oldmx) / n;
                                var = oldvx + (dat - oldmx) * (dat - avg);
                            }
                        }
                    }
                    let sd = (var / (n - 1.0)).sqrt();
                    if sd > sd_max {
                        sd_max = sd;
                    }
                }
                *o = sd_max as f32;
            }
        },
    );
}

// ---------------------------------------------------------------------------------------------------------------
// the kernel

/// The half widths of the rows of the disc of radius `h`: for the row at distance `a`, the largest column
/// offset with FORCE's test `(float)sqrt(a^2 + b^2) <= h`.
pub fn disc_extents(h: usize) -> Vec<usize> {
    let inside = |a: usize, b: usize| f64::from((((a * a + b * b) as f64).sqrt()) as f32) <= h as f64;
    (0..=h)
        .map(|a| {
            let mut b = (((h * h - a * a) as f64).sqrt()) as usize;
            while !inside(a, b) {
                b -= 1;
            }
            while inside(a, b + 1) {
                b += 1;
            }
            b
        })
        .collect()
}

/// What the kernel reads: all planes `(planes, ny, nx)`.
pub struct Grid<'a> {
    /// The standardized components `(npc, ny * nx)`, NaN where invalid.
    pub hr: &'a [f32],
    pub npc: usize,
    /// The fine texture `(ny * nx)`.
    pub hr_tex: &'a [f32],
    /// The coarse bands `(nb_mr, ny * nx)`, NaN where missing.
    pub mr: &'a [f32],
    /// Their texture `(nb_mr, ny * nx)`.
    pub mr_tex: &'a [f32],
    pub nb_mr: usize,
    pub ny: usize,
    pub nx: usize,
    /// Radius of the disc in pixels.
    pub h: usize,
    /// Minimum number of neighbours.
    pub mink: usize,
}

struct Scratch {
    srow: Vec<f64>,
    kp: Vec<u32>,
    ks: Vec<f64>,
    kc: Vec<u8>,
    num: Vec<f64>,
    den: Vec<f64>,
    umin: Vec<f64>,
    umax: Vec<f64>,
    rowout: Vec<f32>,
    center: Vec<f32>,
}

/// FORCE's `improphe` for every pixel: `out` `(nb_mr, ny * nx)`, every value written; NaN where a band of `mr`
/// is NaN at the pixel. Where a pixel has no neighbour (its fine components are NaN, or none passes), the
/// coarse value.
pub fn improphe_grid(g: &Grid, out: &mut [f32], n_jobs: i32) {
    let (ny, nx, nb, npc) = (g.ny, g.nx, g.nb_mr, g.npc);
    let npix = ny * nx;
    assert!(npc > 0 && nb > 0);
    assert_eq!(g.hr.len(), npc * npix);
    assert_eq!(g.hr_tex.len(), npix);
    assert_eq!(g.mr.len(), nb * npix);
    assert_eq!(g.mr_tex.len(), nb * npix);
    assert_eq!(out.len(), nb * npix);
    if npix == 0 {
        return;
    }
    let h = g.h;
    let jext = disc_extents(h);
    // a pixel can be a neighbour when its first component and all the coarse bands are valid
    let mut ok = finite_everywhere(g.mr, nb, npix);
    for (o, v) in ok.iter_mut().zip(&g.hr[..npix]) {
        *o &= v.is_finite();
    }
    let thr = cutoffs();
    let npcf = npc as f32 as f64; // FORCE divides by (float)nb_hr
    let reject = thr[4] * npcf * (1.0 + 1e-9); // a sum above this has S >= the last cutoff whatever the rounding
    let shared = Shared::new(out);
    let width = 2 * h + 1;

    parallel_for(
        ny,
        n_jobs,
        || Scratch {
            srow: vec![0.0; width],
            kp: Vec::new(),
            ks: Vec::new(),
            kc: Vec::new(),
            num: vec![0.0; nb],
            den: vec![0.0; nb],
            umin: vec![0.0; nb],
            umax: vec![0.0; nb],
            rowout: vec![0.0; nb * nx],
            center: vec![0.0; npc],
        },
        |s, i| {
            for j in 0..nx {
                let p = i * nx + j;
                if (0..nb).any(|b| g.mr[b * npix + p].is_nan()) {
                    for b in 0..nb {
                        s.rowout[b * nx + j] = f32::NAN;
                    }
                    continue;
                }
                let copy_mr = |s: &mut Scratch| {
                    for b in 0..nb {
                        s.rowout[b * nx + j] = g.mr[b * npix + p];
                    }
                };
                if !g.hr[p].is_finite() {
                    copy_mr(s);
                    continue;
                }
                s.kp.clear();
                s.ks.clear();
                s.kc.clear();
                let mut sn = [0usize; 5];
                for b in 0..npc {
                    s.center[b] = g.hr[b * npix + p];
                }
                let center = std::mem::take(&mut s.center);
                // pass 1: the neighbours and their classes, in FORCE's scan order
                let (i_lo, i_hi) = (i.saturating_sub(h), (i + h).min(ny - 1));
                for ni in i_lo..=i_hi {
                    let a = ni.abs_diff(i);
                    let hw = jext[a];
                    let j0 = j.saturating_sub(hw);
                    let j1 = (j + hw).min(nx - 1);
                    let len = j1 - j0 + 1;
                    let base = ni * nx + j0;
                    let srow = &mut s.srow[..len];
                    for (b, &c) in center.iter().enumerate() {
                        let plane = &g.hr[b * npix + base..b * npix + base + len];
                        if b == 0 {
                            for (o, &v) in srow.iter_mut().zip(plane) {
                                *o = f64::from((c - v).abs());
                            }
                        } else {
                            for (o, &v) in srow.iter_mut().zip(plane) {
                                *o += f64::from((c - v).abs());
                            }
                        }
                    }
                    for q in 0..len {
                        let np = base + q;
                        let sum = srow[q];
                        if !ok[np] || !(sum <= reject) {
                            continue;
                        }
                        let sv = sum / npcf;
                        let Some(sc) = thr.iter().position(|&t| sv < t) else {
                            continue;
                        };
                        s.kp.push(np as u32);
                        s.ks.push(sv);
                        s.kc.push(sc as u8);
                        for n in &mut sn[sc..] {
                            *n += 1;
                        }
                    }
                }
                s.center = center;
                let wn = s.kp.len();
                if wn == 0 {
                    copy_mr(s);
                    continue;
                }
                let sc = sn.iter().position(|&n| n >= g.mink).unwrap_or(4) as u8;

                // the ranges of the positive values over the neighbours of the cutoff or lower
                let (mut smin, mut smax) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut tmin, mut tmax) = (f64::INFINITY, f64::NEG_INFINITY);
                s.umin.iter_mut().for_each(|v| *v = f64::INFINITY);
                s.umax.iter_mut().for_each(|v| *v = f64::NEG_INFINITY);
                for k in 0..wn {
                    if s.kc[k] > sc {
                        continue;
                    }
                    let np = s.kp[k] as usize;
                    let sv = s.ks[k];
                    if sv > 0.0 {
                        smax = smax.max(sv);
                        smin = smin.min(sv);
                    }
                    let t = f64::from(g.hr_tex[np]);
                    if t > 0.0 {
                        tmax = tmax.max(t);
                        tmin = tmin.min(t);
                    }
                    for b in 0..nb {
                        let u = f64::from(g.mr_tex[b * npix + np]);
                        if u > 0.0 {
                            s.umax[b] = s.umax[b].max(u);
                            s.umin[b] = s.umin[b].min(u);
                        }
                    }
                }

                // pass 2: the weighted mean, in the scan order
                s.num.iter_mut().for_each(|v| *v = 0.0);
                s.den.iter_mut().for_each(|v| *v = 0.0);
                for k in 0..wn {
                    if s.kc[k] > sc {
                        continue;
                    }
                    let np = s.kp[k] as usize;
                    let ss = rescale_weight(s.ks[k], smin, smax);
                    let tt = rescale_weight(f64::from(g.hr_tex[np]), tmin, tmax);
                    for b in 0..nb {
                        let uu = rescale_weight(f64::from(g.mr_tex[b * npix + np]), s.umin[b], s.umax[b]);
                        let w = 1.0 / (ss * tt * uu);
                        s.num[b] += w * f64::from(g.mr[b * npix + np]);
                        s.den[b] += w;
                    }
                }
                for b in 0..nb {
                    s.rowout[b * nx + j] = (s.num[b] / s.den[b]) as f32;
                }
            }
            for b in 0..nb {
                // SAFETY: each row is written by one task.
                let dst = unsafe { shared.slice_mut(b * npix + i * nx..b * npix + (i + 1) * nx) };
                dst.copy_from_slice(&s.rowout[b * nx..(b + 1) * nx]);
            }
        },
    );
}

// ---------------------------------------------------------------------------------------------------------------
// the two variants

/// Settings shared by both variants.
#[derive(Clone, Debug)]
pub struct Params {
    /// Limits of the seasonal windows (days of the year), `nwin + 1` of them.
    pub dwin: Vec<i32>,
    /// Radius of the disc in pixels (`KERNEL_SIZE / resolution`).
    pub h: usize,
    /// Radius of the texture window in pixels (`KERNEL_TEXT / resolution`).
    pub ksd: usize,
    /// Fraction of the variance the components keep.
    pub min_var: f32,
    /// FORCE's seasonal means: truncated to integers, every window's upper limit excluded (see the module).
    pub force: bool,
    /// FORCE's components: the projection on the rows of GSL's eigenvectors (see `pca`).
    pub force_pca: bool,
}

/// The fine data ready for the kernel: the standardized components and their texture.
pub struct Prepared {
    pub npc: usize,
    pub pcs: Vec<f32>,
    pub tex: Vec<f32>,
    pub n_valid: usize,
    pub eigenvalues: Vec<f64>,
}

/// The fine scenes `x` `(nt, nb, ny * nx)` of the year `ywin` (`None`: all of them) to components and texture:
/// the seasonal means, the PCA, the standardization and the texture of the components. `None` when no pixel has
/// a scene (or fewer than two have all the windows: no components).
#[allow(clippy::too_many_arguments)]
pub fn prepare_fine(
    x: &[f32],
    nt: usize,
    nb: usize,
    ny: usize,
    nx: usize,
    doy: &[i32],
    year: &[i32],
    ywin: Option<i32>,
    p: &Params,
    n_jobs: i32,
) -> Option<Prepared> {
    let npix = ny * nx;
    let nwin = p.dwin.len() - 1;
    let mut seasonal = vec![0f32; nwin * nb * npix];
    let n = average_season(
        x,
        nt,
        nb,
        npix,
        doy,
        year,
        &p.dwin,
        ywin,
        p.force,
        &mut seasonal,
        n_jobs,
    );
    if n == 0 {
        return None;
    }
    let mut pc = pca(&seasonal, nwin * nb, npix, p.min_var, p.force_pca, n_jobs);
    drop(seasonal);
    if pc.npc == 0 {
        return None;
    }
    standardize(&mut pc.pcs, npix, n_jobs);
    let mut tex = vec![0f32; npix];
    focal_sd(&pc.pcs, ny, nx, p.ksd, 0..pc.npc, &mut tex, n_jobs);
    Some(Prepared {
        npc: pc.npc,
        pcs: pc.pcs,
        tex,
        n_valid: pc.n_valid,
        eigenvalues: pc.eigenvalues,
    })
}

/// The prediction of the coarse bands `mr` `(nb_mr, ny * nx)` from the prepared fine data: their texture, then
/// the kernel; `out` as `improphe_grid`'s.
pub fn predict(
    prep: &Prepared,
    mr: &[f32],
    nb_mr: usize,
    ny: usize,
    nx: usize,
    p: &Params,
    out: &mut [f32],
    n_jobs: i32,
) {
    predict_planes(&prep.pcs, prep.npc, &prep.tex, mr, nb_mr, ny, nx, p, out, n_jobs);
}

/// `predict` from the planes themselves: the standardized components `pcs` `(npc, ny * nx)` and the fine
/// texture `tex`.
#[allow(clippy::too_many_arguments)]
pub fn predict_planes(
    pcs: &[f32],
    npc: usize,
    tex: &[f32],
    mr: &[f32],
    nb_mr: usize,
    ny: usize,
    nx: usize,
    p: &Params,
    out: &mut [f32],
    n_jobs: i32,
) {
    let npix = ny * nx;
    let mut mr_tex = vec![0f32; nb_mr * npix];
    for b in 0..nb_mr {
        focal_sd(
            mr,
            ny,
            nx,
            p.ksd,
            b..b + 1,
            &mut mr_tex[b * npix..(b + 1) * npix],
            n_jobs,
        );
    }
    let g = Grid {
        hr: pcs,
        npc,
        hr_tex: tex,
        mr,
        mr_tex: &mr_tex,
        nb_mr,
        ny,
        nx,
        h: p.h,
        mink: min_neighbors(p.h),
    };
    improphe_grid(&g, out, n_jobs);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn mink_values() {
        assert_eq!(min_neighbors(5), 0);
        assert_eq!(min_neighbors(10), 1);
        assert_eq!(min_neighbors(20), 6);
        assert_eq!(min_neighbors(83), 109);
        assert_eq!(min_neighbors(250), 985);
    }

    #[test]
    fn rescale_cases() {
        assert_eq!(rescale_weight(0.0, 0.1, 0.5), 1.0);
        assert_eq!(rescale_weight(0.3, 0.3, 0.3), 1.0);
        // the minimum: 1 + exp(-7.5); the maximum: 1 + exp(17.5)
        assert!((rescale_weight(0.1, 0.1, 0.5) - (1.0 + (-7.5f64).exp())).abs() < 1e-15);
        assert!((rescale_weight(0.5, 0.1, 0.5) - (1.0 + 17.5f64.exp())).abs() < 1e-6);
        assert!((rescale_weight(0.3, 0.1, 0.5) - (1.0 + 5.0f64.exp())).abs() < 1e-12);
    }

    #[test]
    fn disc_rows_match_the_float_test() {
        for h in [1usize, 2, 5, 8, 10, 33, 83, 250] {
            let ext = disc_extents(h);
            for a in 0..=h {
                for b in 0..=h + 1 {
                    let want = f64::from(((a * a + b * b) as f64).sqrt() as f32) <= h as f64;
                    assert_eq!(b <= ext[a], want, "h {h} a {a} b {b}");
                }
            }
        }
        // the disc of radius 2 has 13 pixels
        let e = disc_extents(2);
        assert_eq!(e, vec![2, 1, 0]);
        let n: usize = e
            .iter()
            .enumerate()
            .map(|(a, &b)| (if a == 0 { 1 } else { 2 }) * (2 * b + 1))
            .sum();
        assert_eq!(n, 13);
    }

    #[test]
    fn season_windows_and_day_365() {
        let dwin = [1, 91, 181, 271, 365];
        // upper end exclusive in FORCE: 90 in window 0, 91 in window 1, 364 in window 3, 365 nowhere
        assert_eq!(season_window(1, 2020, &dwin, Some(2020), true), Some(0));
        assert_eq!(season_window(90, 2020, &dwin, Some(2020), true), Some(0));
        assert_eq!(season_window(91, 2020, &dwin, Some(2020), true), Some(1));
        assert_eq!(season_window(364, 2020, &dwin, Some(2020), true), Some(3));
        assert_eq!(season_window(365, 2020, &dwin, Some(2020), true), None);
        // not FORCE: the last window includes day 365
        assert_eq!(season_window(365, 2020, &dwin, Some(2020), false), Some(3));
        assert_eq!(season_window(364, 2020, &dwin, Some(2020), false), Some(3));
        // other years are out, `None` takes any
        assert_eq!(season_window(100, 2019, &dwin, Some(2020), true), None);
        assert_eq!(season_window(100, 2019, &dwin, None, true), Some(1));
    }

    #[test]
    fn year_crossing_windows() {
        // winter window 330..60 crosses the year, then 60..330
        let dwin = [330, 60, 330];
        // ywin 2020: Dec 2019 (day 340 of 2019) and Jan 2020 (day 10) are window 0; also Dec 2020 by the second test
        assert_eq!(season_window(340, 2019, &dwin, Some(2020), true), Some(0));
        assert_eq!(season_window(10, 2020, &dwin, Some(2020), true), Some(0));
        assert_eq!(season_window(340, 2020, &dwin, Some(2020), true), Some(0));
        assert_eq!(season_window(10, 2021, &dwin, Some(2020), true), Some(0));
        assert_eq!(season_window(10, 2018, &dwin, Some(2020), true), None);
        // 60..330 as the second window: lo > hi is false for (60, 330), so a normal one
        assert_eq!(season_window(100, 2020, &dwin, Some(2020), true), Some(1));
        assert_eq!(season_window(365, 2020, &dwin, Some(2020), true), Some(0));
        // any year
        assert_eq!(season_window(340, 1999, &dwin, None, true), Some(0));
    }

    #[test]
    fn seasonal_means_and_the_empty_window() {
        // 1 band, 2 pixels, 4 scenes in 2020: days 10, 100, 100, 200; windows 1 91 181 271 365
        let (nt, nb, npix) = (4, 1, 2);
        let x = [10.0f32, 20.0, 31.0, 40.0, 32.0, 41.0, 100.0, f32::NAN];
        let doy = [10, 100, 100, 200];
        let year = [2020; 4];
        let dwin = [1, 91, 181, 271, 365];
        let mut out = vec![0f32; 4 * npix];
        let n = average_season(&x, nt, nb, npix, &doy, &year, &dwin, Some(2020), true, &mut out, 1);
        assert_eq!(n, 2);
        // pixel 0: window 0 = 10; window 1 = (31 + 32) / 2 = 31.5 -> 31; window 2 = 100; window 3 empty = overall (173/4 = 43.25 -> 43)
        assert_eq!([out[0], out[2], out[4], out[6]], [10.0, 31.0, 100.0, 43.0]);
        // not FORCE: floats
        let mut f = vec![0f32; 4 * npix];
        average_season(&x, nt, nb, npix, &doy, &year, &dwin, Some(2020), false, &mut f, 3);
        assert_eq!([f[0], f[2], f[4], f[6]], [10.0, 31.5, 100.0, 43.25]);
        // pixel 1: the last scene is NaN, window 2 is empty and takes the mean of the three: 101/3 = 33.67
        assert_eq!(out[1], 20.0);
        assert_eq!(out[5], 33.0);
        // another year: no scene at all, NaN everywhere
        let n = average_season(&x, nt, nb, npix, &doy, &year, &dwin, Some(2019), true, &mut out, 1);
        assert_eq!(n, 0);
        assert!(out.iter().all(|v| v.is_nan()));
    }

    /// The standard deviation of the values the plain way.
    fn sd_direct(v: &[f64]) -> f64 {
        if v.len() <= 1 {
            return 0.0;
        }
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    }

    #[test]
    fn focal_sd_against_a_direct_computation() {
        let (ny, nx, k) = (13, 17, 3);
        let mut s = 7u64;
        let mut data: Vec<f32> = (0..2 * ny * nx).map(|_| (lcg(&mut s) * 10.0) as f32).collect();
        data[5] = f32::NAN;
        data[ny * nx + 40] = f32::NAN;
        // a window with one value only
        for j in 0..nx {
            for i in 0..ny {
                if (i, j) != (0, 0) && i <= k && j <= k {
                    data[i * nx + j] = f32::NAN;
                    data[ny * nx + i * nx + j] = f32::NAN;
                }
            }
        }
        for (bands, nbands) in [(0..2, 2usize), (1..2, 1)] {
            let mut out = vec![0f32; ny * nx];
            focal_sd(&data, ny, nx, k, bands.clone(), &mut out, 2);
            for i in 0..ny {
                for j in 0..nx {
                    let mut best = 0.0f64;
                    for b in bands.clone() {
                        let mut v = Vec::new();
                        for ni in i.saturating_sub(k)..=(i + k).min(ny - 1) {
                            for nj in j.saturating_sub(k)..=(j + k).min(nx - 1) {
                                let x = data[b * ny * nx + ni * nx + nj];
                                if !x.is_nan() {
                                    v.push(f64::from(x));
                                }
                            }
                        }
                        best = best.max(sd_direct(&v));
                    }
                    let got = f64::from(out[i * nx + j]);
                    assert!(
                        (got - best).abs() < 1e-5 * best.max(1.0),
                        "({i},{j}) {got} vs {best}, {nbands}"
                    );
                }
            }
        }
        // the pixel with a single value in its window has texture 0
        let mut out = vec![1f32; ny * nx];
        focal_sd(&data, ny, nx, k, 0..1, &mut out, 1);
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn standardize_gives_mean_0_and_sd_1() {
        let mut s = 3u64;
        let npix = 5000;
        let mut a: Vec<f32> = (0..2 * npix).map(|_| (lcg(&mut s) * 100.0 + 50.0) as f32).collect();
        a[10] = f32::NAN;
        standardize(&mut a, npix, 2);
        assert!(a[10].is_nan());
        for b in 0..2 {
            let v: Vec<f64> = a[b * npix..(b + 1) * npix]
                .iter()
                .filter(|x| !x.is_nan())
                .map(|&x| f64::from(x))
                .collect();
            let m = v.iter().sum::<f64>() / v.len() as f64;
            assert!(m.abs() < 1e-4, "{m}");
            assert!((sd_direct(&v) - 1.0).abs() < 1e-5);
        }
    }

    /// A correlated 5-band image: two latent factors plus a little noise, with a few NaN pixels.
    fn latent(npix: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let nb = 5;
        let mut x = vec![0f32; nb * npix];
        let load = [[1.0, 0.5], [0.8, -0.4], [0.2, 1.0], [-0.6, 0.3], [0.1, 0.1]];
        for p in 0..npix {
            let (f1, f2) = (lcg(&mut s) * 3000.0, lcg(&mut s) * 800.0);
            for b in 0..nb {
                x[b * npix + p] = (1000.0 + f1 * load[b][0] + f2 * load[b][1] + lcg(&mut s) * 20.0).trunc() as f32;
            }
        }
        x[3] = f32::NAN;
        x[2 * npix + 77] = f32::NAN;
        x
    }

    #[test]
    fn pca_modes_and_the_transposed_projection() {
        let (nb, npix) = (5, 4000);
        let x = latent(npix, 11);
        let force = pca(&x, nb, npix, 0.975, true, 3);
        let paper = pca(&x, nb, npix, 0.975, false, 3);
        assert_eq!(force.n_valid, npix - 2);
        assert_eq!(force.npc, paper.npc);
        assert!(force.npc >= 2 && force.npc <= nb);
        for (a, b) in force.eigenvalues.iter().zip(&paper.eigenvalues) {
            assert!((a - b).abs() < 1e-8 * a.abs().max(1.0));
        }
        assert!(force.pcs[3].is_nan() && force.pcs[77].is_nan() && force.pcs[78].is_finite());
        assert!(paper.pcs[3].is_nan());
        // the paper's components are uncorrelated with variances equal to the eigenvalues
        let ok: Vec<usize> = (0..npix).filter(|&p| paper.pcs[p].is_finite()).collect();
        let var = |c: usize| {
            let v: Vec<f64> = ok.iter().map(|&p| f64::from(paper.pcs[c * npix + p])).collect();
            sd_direct(&v).powi(2)
        };
        for c in 0..paper.npc {
            assert!((var(c) / paper.eigenvalues[c] - 1.0).abs() < 1e-4, "component {c}");
        }
        let cov01: f64 = ok
            .iter()
            .map(|&p| f64::from(paper.pcs[p]) * f64::from(paper.pcs[npix + p]))
            .sum::<f64>()
            / (ok.len() - 1) as f64;
        assert!(cov01.abs() < 1e-3 * paper.eigenvalues[0], "{cov01}");
        // FORCE's: component b = sum_j centred_j * evec[b][j], from the GSL eigenvectors of the same covariance
        let nk = ok.len();
        let mean: Vec<f64> = (0..nb)
            .map(|b| ok.iter().map(|&p| f64::from(x[b * npix + p])).sum::<f64>() / nk as f64)
            .collect();
        let mut cov = vec![0f64; nb * nb];
        for i in 0..nb {
            for j in 0..nb {
                let mut t = 0.0;
                for &p in &ok {
                    t += (f64::from(x[i * npix + p]) - mean[i]) * (f64::from(x[j * npix + p]) - mean[j]);
                }
                cov[i * nb + j] = t * (1.0 / (nk - 1) as f64);
            }
        }
        let (_, evec) = gsl_symmv::eigen_symmv_desc(&mut cov, nb);
        for c in 0..force.npc {
            for &p in ok.iter().step_by(97) {
                let want: f64 = (0..nb)
                    .map(|j| (f64::from(x[j * npix + p]) - mean[j]) * evec[c * nb + j])
                    .sum();
                let got = f64::from(force.pcs[c * npix + p]);
                assert!(
                    (got - want).abs() < 1e-3 * want.abs().max(1.0),
                    "pc {c} pixel {p}: {got} vs {want}"
                );
            }
        }
        // and it is not the paper's component unless the eigenvector matrix is symmetric
        let differs = (0..force.npc).any(|c| {
            ok.iter().take(50).any(|&p| {
                (force.pcs[c * npix + p] - paper.pcs[c * npix + p]).abs()
                    > 1e-3 * paper.pcs[c * npix + p].abs().max(1.0)
            })
        });
        assert!(differs);
        // min_var 1 keeps all the components
        assert_eq!(pca(&x, nb, npix, 1.0, false, 1).npc, nb);
        // the thread count does not change a bit
        let again = pca(&x, nb, npix, 0.975, true, 1);
        assert!(again
            .pcs
            .iter()
            .zip(&force.pcs)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// The kernel as FORCE writes it, for the tests: one pass over the square, the distance table of floats,
    /// every neighbour kept.
    fn naive(g: &Grid, i: usize, j: usize, out: &mut [f32]) {
        let (nx, ny, nb, npc, h) = (g.nx as i64, g.ny as i64, g.nb_mr, g.npc, g.h as i64);
        let npix = (nx * ny) as usize;
        let p = (i as i64 * nx + j as i64) as usize;
        for b in 0..nb {
            if g.mr[b * npix + p].is_nan() {
                for o in out.iter_mut() {
                    *o = f32::NAN;
                }
                return;
            }
        }
        let sthr = [0.025f32, 0.05, 0.1, 0.2, 0.4];
        let (mut sclass, mut srec, mut trec) = (vec![], vec![], vec![]);
        let (mut mrec, mut urec) = (vec![vec![]; nb], vec![vec![]; nb]);
        let mut srange = [[f64::MAX; 5], [f64::MIN; 5]];
        let mut trange = [[f64::MAX; 5], [f64::MIN; 5]];
        let mut urange = vec![[[f64::MAX; 5], [f64::MIN; 5]]; nb];
        let mut sn = [0usize; 5];
        for ii in -h..=h {
            for jj in -h..=h {
                let (ni, nj) = (i as i64 + ii, j as i64 + jj);
                if ni < 0 || nj < 0 || ni >= ny || nj >= nx {
                    continue;
                }
                let np = (nx * ni + nj) as usize;
                let d = ((ii * ii + jj * jj) as f64).sqrt() as f32;
                if f64::from(d) > h as f64 {
                    continue;
                }
                if g.hr[np].is_nan() {
                    continue;
                }
                if (0..nb).any(|b| g.mr[b * npix + np].is_nan()) {
                    continue;
                }
                let mut sum = 0.0f64;
                for b in 0..npc {
                    sum += f64::from((g.hr[b * npix + p] - g.hr[b * npix + np]).abs());
                }
                let s = sum / f64::from(npc as f32);
                let Some(sc) = (0..5).find(|&k| s < f64::from(sthr[k])) else {
                    continue;
                };
                sclass.push(sc);
                srec.push(s);
                if s > 0.0 {
                    for k in sc..5 {
                        srange[1][k] = srange[1][k].max(s);
                        srange[0][k] = srange[0][k].min(s);
                    }
                }
                for b in 0..nb {
                    mrec[b].push(f64::from(g.mr[b * npix + np]));
                }
                let t = f64::from(g.hr_tex[np]);
                trec.push(t);
                if t > 0.0 {
                    for k in sc..5 {
                        trange[1][k] = trange[1][k].max(t);
                        trange[0][k] = trange[0][k].min(t);
                    }
                }
                for b in 0..nb {
                    let u = f64::from(g.mr_tex[b * npix + np]);
                    urec[b].push(u);
                    if u > 0.0 {
                        for k in sc..5 {
                            urange[b][1][k] = urange[b][1][k].max(u);
                            urange[b][0][k] = urange[b][0][k].min(u);
                        }
                    }
                }
                for n in &mut sn[sc..] {
                    *n += 1;
                }
            }
        }
        if sclass.is_empty() {
            for b in 0..nb {
                out[b] = g.mr[b * npix + p];
            }
            return;
        }
        let sc = (0..5).find(|&k| sn[k] >= g.mink).unwrap_or(4);
        let (mut num, mut den) = (vec![0f64; nb], vec![0f64; nb]);
        for k in 0..sclass.len() {
            if sclass[k] > sc {
                continue;
            }
            let ss = rescale_weight(srec[k], srange[0][sc], srange[1][sc]);
            let tt = rescale_weight(trec[k], trange[0][sc], trange[1][sc]);
            for b in 0..nb {
                let uu = rescale_weight(urec[b][k], urange[b][0][sc], urange[b][1][sc]);
                let w = 1.0 / (ss * tt * uu);
                num[b] += w * mrec[b][k];
                den[b] += w;
            }
        }
        for b in 0..nb {
            out[b] = (num[b] / den[b]) as f32;
        }
    }

    /// A 40 x 40 scene: four classes in blobs with a seasonal-looking pair of components, a coarse field that is
    /// the class mean on 5 x 5 blocks (so it varies inside the classes' borders), and a few holes.
    fn scene(ny: usize, nx: usize, seed: u64) -> (Vec<f32>, usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u8>) {
        let mut s = seed;
        let npix = ny * nx;
        let centers = [(8.0, 8.0), (10.0, 30.0), (30.0, 12.0), (31.0, 31.0)];
        let class: Vec<u8> = (0..npix)
            .map(|p| {
                let (i, j) = ((p / nx) as f64, (p % nx) as f64);
                (0..4)
                    .min_by(|&a, &b| {
                        let da = (i - centers[a].0).powi(2) + (j - centers[a].1).powi(2);
                        let db = (i - centers[b].0).powi(2) + (j - centers[b].1).powi(2);
                        da.partial_cmp(&db).unwrap()
                    })
                    .unwrap() as u8
            })
            .collect();
        let npc = 3;
        let proto = [[-1.0, 0.2, 0.0], [1.0, 0.3, 0.1], [0.1, -1.0, 0.4], [0.0, 0.9, -1.2]];
        let mut hr = vec![0f32; npc * npix];
        for p in 0..npix {
            for b in 0..npc {
                hr[b * npix + p] = (proto[class[p] as usize][b] + 0.02 * (lcg(&mut s) - 0.5)) as f32;
            }
        }
        let value = [10.0, 40.0, 25.0, 70.0];
        let nb_mr = 2;
        let mut mr = vec![0f32; nb_mr * npix];
        for p in 0..npix {
            // blocks of 5 x 5: the value of the class of the block's first pixel
            let (i, j) = (p / nx, p % nx);
            let q = (i / 5 * 5) * nx + (j / 5 * 5);
            for b in 0..nb_mr {
                mr[b * npix + p] = (value[class[q] as usize] * (1.0 + b as f64 * 0.5) + (lcg(&mut s) * 0.1)) as f32;
            }
        }
        mr[7] = f32::NAN; // a hole in band 0 only
        hr[npix + 9 * nx + 9] = f32::NAN; // fine pixels with no data
        hr[9 * nx + 9] = f32::NAN;
        hr[2 * npix + 9 * nx + 9] = f32::NAN;
        let hr_tex: Vec<f32> = (0..npix).map(|_| (lcg(&mut s) * 0.5) as f32).collect();
        let mut mr_tex = vec![0f32; nb_mr * npix];
        for b in 0..nb_mr {
            focal_sd(&mr, ny, nx, 2, b..b + 1, &mut mr_tex[b * npix..(b + 1) * npix], 1);
        }
        (hr, npc, hr_tex, mr, mr_tex, class)
    }

    #[test]
    fn the_kernel_is_the_literal_loop_bit_for_bit() {
        let (ny, nx) = (40, 40);
        let (hr, npc, hr_tex, mr, mr_tex, _) = scene(ny, nx, 5);
        for h in [5usize, 8, 13] {
            let g = Grid {
                hr: &hr,
                npc,
                hr_tex: &hr_tex,
                mr: &mr,
                mr_tex: &mr_tex,
                nb_mr: 2,
                ny,
                nx,
                h,
                mink: min_neighbors(h),
            };
            for n_jobs in [1, 3] {
                let mut out = vec![0f32; 2 * ny * nx];
                improphe_grid(&g, &mut out, n_jobs);
                let mut want = [0f32; 2];
                let npix = ny * nx;
                for i in 0..ny {
                    for j in 0..nx {
                        naive(&g, i, j, &mut want);
                        for b in 0..2 {
                            let got = out[b * npix + i * nx + j];
                            assert!(
                                got.to_bits() == want[b].to_bits() || (got.is_nan() && want[b].is_nan()),
                                "h {h} pixel ({i},{j}) band {b}: {got} vs {}",
                                want[b]
                            );
                        }
                    }
                }
            }
        }
        // a larger mink exercises the higher cutoffs
        let g = Grid {
            hr: &hr,
            npc,
            hr_tex: &hr_tex,
            mr: &mr,
            mr_tex: &mr_tex,
            nb_mr: 2,
            ny,
            nx,
            h: 8,
            mink: 400,
        };
        let mut out = vec![0f32; 2 * ny * nx];
        improphe_grid(&g, &mut out, 2);
        let mut want = [0f32; 2];
        for i in (0..ny).step_by(3) {
            for j in (0..nx).step_by(3) {
                naive(&g, i, j, &mut want);
                for b in 0..2 {
                    let got = out[b * ny * nx + i * nx + j];
                    assert!(got.to_bits() == want[b].to_bits() || (got.is_nan() && want[b].is_nan()));
                }
            }
        }
    }

    #[test]
    fn nan_handling_in_the_kernel() {
        let (ny, nx) = (40, 40);
        let (hr, npc, hr_tex, mr, mr_tex, _) = scene(ny, nx, 5);
        let g = Grid {
            hr: &hr,
            npc,
            hr_tex: &hr_tex,
            mr: &mr,
            mr_tex: &mr_tex,
            nb_mr: 2,
            ny,
            nx,
            h: 8,
            mink: min_neighbors(8),
        };
        let mut out = vec![0f32; 2 * ny * nx];
        improphe_grid(&g, &mut out, 1);
        let npix = ny * nx;
        // the hole of band 0 at pixel 7 makes both bands NaN there, and pixel 7 is no neighbour of others
        assert!(out[7].is_nan() && out[npix + 7].is_nan());
        // a fine pixel with no data takes the coarse value
        let p = 9 * nx + 9;
        assert_eq!(out[p], mr[p]);
        assert_eq!(out[npix + p], mr[npix + p]);
        // and everywhere else the output is a number
        assert_eq!(out.iter().filter(|v| v.is_nan()).count(), 2);
    }

    #[test]
    fn recovers_a_field_that_is_a_function_of_the_classes() {
        let (ny, nx) = (40, 40);
        let (hr, npc, hr_tex, mr, mr_tex, class) = scene(ny, nx, 21);
        let value = [10.0f64, 40.0, 25.0, 70.0];
        let g = Grid {
            hr: &hr,
            npc,
            hr_tex: &hr_tex,
            mr: &mr,
            mr_tex: &mr_tex,
            nb_mr: 2,
            ny,
            nx,
            h: 8,
            mink: min_neighbors(8),
        };
        let mut out = vec![0f32; 2 * ny * nx];
        improphe_grid(&g, &mut out, 1);
        let npix = ny * nx;
        let (mut e_imp, mut e_in, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for p in 0..npix {
            if !mr[p].is_finite() || p == 9 * nx + 9 {
                continue;
            }
            let truth = value[class[p] as usize];
            e_imp += (f64::from(out[p]) - truth).powi(2);
            e_in += (f64::from(mr[p]) - truth).powi(2);
            n += 1.0;
        }
        assert!(
            (e_imp / n).sqrt() < 0.7 * (e_in / n).sqrt(),
            "{} vs {}",
            (e_imp / n).sqrt(),
            (e_in / n).sqrt()
        );
    }

    #[test]
    fn the_two_variants_on_a_small_cube() {
        let (ny, nx) = (24, 24);
        let npix = ny * nx;
        let mut s = 31u64;
        // 12 fine scenes in 2 years, 3 bands, a class structure on the left and right halves
        let nt = 12;
        let nb = 3;
        let mut x = vec![0f32; nt * nb * npix];
        let mut doy = vec![];
        let mut year = vec![];
        for t in 0..nt {
            doy.push(20 + 55 * (t as i32 % 6));
            year.push(2020 + t as i32 / 6);
            for b in 0..nb {
                for p in 0..npix {
                    let left = (p % nx) < nx / 2;
                    let base = if left { 1000.0 } else { 3000.0 } + 300.0 * b as f64 + 100.0 * (t % 6) as f64;
                    x[(t * nb + b) * npix + p] = (base + lcg(&mut s) * 30.0) as f32;
                }
            }
        }
        let params = Params {
            dwin: vec![1, 91, 181, 271, 365],
            h: 6,
            ksd: 2,
            min_var: 0.975,
            force: false,
            force_pca: false,
        };
        let mr: Vec<f32> = (0..npix).map(|p| if (p % nx) < nx / 2 { 10.0 } else { 90.0 }).collect();
        for force in [false, true] {
            let p = Params {
                force,
                force_pca: force,
                ..params.clone()
            };
            let prep = prepare_fine(&x, nt, nb, ny, nx, &doy, &year, Some(2020), &p, 2).expect("data");
            assert!(prep.npc >= 1);
            let mut out = vec![0f32; npix];
            predict(&prep, &mr, 1, ny, nx, &p, &mut out, 2);
            assert!(out.iter().all(|v| v.is_finite()));
            // a field that is constant on each half stays constant
            for q in 0..npix {
                assert!((out[q] - mr[q]).abs() < 1e-3, "force {force}, pixel {q}: {}", out[q]);
            }
            // a year without scenes: nothing to prepare
            assert!(prepare_fine(&x, nt, nb, ny, nx, &doy, &year, Some(1999), &p, 1).is_none());
            // the all-years mean (L2)
            assert!(prepare_fine(&x, nt, nb, ny, nx, &doy, &year, None, &p, 1).is_some());
        }
    }
}
