//! Sentinel-2 parallax cloud test (the Cloud Displacement Index) and the buffer of a binary mask.
//!
//! Ported from FORCE 3.11.00 (Framework for Operational Radiometric Correction for Environmental monitoring,
//! Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later, https://github.com/davidfrantz/force):
//! `cloud_parallax` of src/modules/lower-level/cloud-ll.c (Frantz's own step, the Cloud Displacement Index of
//! Frantz, Haß, Uhl, Stoffels and Hill 2018, *Remote Sensing of Environment* 215:471-481), with `gauss_kernel`,
//! `buffer_` (the midpoint circle), `covar_recurrence`, `variance` and the queue of the cross-level code.
//! `cloud-ll.c` as a whole is derived from Zhu's Fmask, but none of Fmask's parts (the potential cloud pixels, the
//! land and water probabilities, the shadow matching) is here: they are the inputs `pcp`, `var` and `bright`.
//!
//! The test, on the 20 m grid (blocks of 2 x 2 pixels of the 10 m one):
//! 1. the broad NIR (B08) is smoothed by a 5 x 5 Gaussian kernel (sigma 0.5) at 10 m, off pixels left out;
//! 2. per block, with at least one potential cloud pixel and the block's first pixel not off: the means over the
//!    pixels that are not off of the red edge 3 (B07), the NIR (B8A), the smoothed broad NIR and the variability
//!    probability give the ratios `B07 / B8A` and `smoothed B08 / B8A`; blocks without a potential cloud pixel
//!    have the ratios 0 and are left out of everything that follows;
//! 3. the CDI of a block is `(var_x - var_y) / (var_x + var_y)`, the variances of the two ratios over the 7 x 7
//!    blocks around it (more than 2 of them), a high displaced object making the broad NIR ratio vary more than
//!    the red edge 3 one, which gives a negative CDI; and the minimum of the variability over the 15 x 15 blocks;
//! 4. clouds are the blocks with a CDI of at most -0.5, or a CDI below 0 and a minimum variability above 0.8,
//!    eroded by one block (a 3 x 3 neighbourhood in which every block has a CDI and either a CDI of at most -0.5
//!    or a minimum variability of at least 0.8);
//! 5. region growing: the clouds grow into 8-connected neighbours with a CDI in [-1, -0.25);
//! 6. back to 10 m (each block is four pixels); saturated pixels are added; those that are not bright are
//!    removed.
//!
//! The values are FORCE's `short` (reflectance times 10000), its `float` and `double` where it has them, and the
//! order of its sums, so that the CDI is the same to the last bit. FORCE's code walks `ny` and `nx` in steps of 2
//! and indexes the 20 m grid by `i / 2`, which is only right for even sizes; here the last row or column of an odd
//! image is not part of any block (it is read by the smoothing and is never cloud, unless saturated and bright).

use crate::parallel::for_each_row;
use std::collections::VecDeque;

/// The CDI of a block that has none (FORCE's `_FORCE_NO_DATA_`).
pub const NODATA: f32 = -9999.0;
const NK: usize = 5;
const SIGMA: f32 = 0.5;

/// The scene on the 10 m grid, row-major, in FORCE's units (reflectance times 10000).
pub struct Scene<'a> {
    pub nx: usize,
    pub ny: usize,
    /// Red edge 3 (B07).
    pub re3: &'a [i16],
    /// Broad NIR (B08).
    pub bnir: &'a [i16],
    /// NIR (B8A).
    pub nir: &'a [i16],
    /// Variability probability times 10000.
    pub var: &'a [i16],
    /// Potential cloud pixels.
    pub pcp: &'a [bool],
    /// No data (FORCE's `get_off`).
    pub off: &'a [bool],
}

/// `gauss_kernel` of FORCE: `nk * nk` weights, row-major.
pub fn gauss_kernel(nk: usize, sigma: f32) -> Vec<f32> {
    let sig = sigma * sigma * 2.0;
    let mut k = vec![0.0f32; nk * nk];
    for i in 0..nk {
        for j in 0..nk {
            let hx = (-((nk as i32 - 1) / 2) + j as i32) as f32;
            let hy = (-((nk as i32 - 1) / 2) + i as i32) as f32;
            let e = -(hx * hx + hy * hy) / sig;
            k[i * nk + j] = (1.0 / (std::f64::consts::PI * sig as f64) * (e as f64).exp()) as f32;
        }
    }
    k
}

/// The broad NIR smoothed at the 10 m pixel `(i, j)`, off pixels left out of the weighted mean.
fn smooth(sc: &Scene, kernel: &[f32], i: usize, j: usize) -> f32 {
    let (nx, ny) = (sc.nx as isize, sc.ny as isize);
    let h = (NK as isize - 1) / 2;
    let (mut sum, mut num) = (0.0f64, 0.0f64);
    for ii in 0..NK {
        for jj in 0..NK {
            let ni = i as isize - h + ii as isize;
            let nj = j as isize - h + jj as isize;
            if ni < 0 || ni >= ny || nj < 0 || nj >= nx {
                continue;
            }
            let np = ni as usize * sc.nx + nj as usize;
            if sc.off[np] {
                continue;
            }
            let k = kernel[ii * NK + jj];
            sum += (sc.bnir[np] as f32 * k) as f64;
            num += k as f64;
        }
    }
    (sum / num) as f32
}

/// Per block of the 20 m grid, `[B07 / B8A, smoothed B08 / B8A, variability]`; zeros where the block is not used.
fn ratios(sc: &Scene, n_jobs: i32) -> Vec<[f32; 3]> {
    let (nx_, ny_) = (sc.nx / 2, sc.ny / 2);
    let kernel = gauss_kernel(NK, SIGMA);
    let mut out = vec![[0.0f32; 3]; nx_ * ny_];
    for_each_row(
        &mut out,
        nx_,
        n_jobs,
        || (),
        |_, i_, row| {
            let i = i_ * 2;
            for (j_, cell) in row.iter_mut().enumerate() {
                let j = j_ * 2;
                if sc.off[i * sc.nx + j] {
                    continue;
                }
                let (mut num, mut pcp) = (0.0f64, false);
                for ii in 0..2 {
                    for jj in 0..2 {
                        let np = (i + ii) * sc.nx + j + jj;
                        if sc.off[np] {
                            continue;
                        }
                        num += 1.0;
                        if sc.pcp[np] {
                            pcp = true;
                        }
                    }
                }
                if !(num > 0.0 && pcp) {
                    continue;
                }
                let (mut s_re3, mut s_nir, mut s_bnir, mut s_var) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for ii in 0..2 {
                    for jj in 0..2 {
                        let np = (i + ii) * sc.nx + j + jj;
                        if sc.off[np] {
                            continue;
                        }
                        s_bnir += smooth(sc, &kernel, i + ii, j + jj) as f64;
                        s_nir += sc.nir[np] as f64;
                        s_re3 += sc.re3[np] as f64;
                        s_var += sc.var[np] as f64 / 10000.0;
                    }
                }
                *cell = [
                    ((s_re3 / num) / (s_nir / num)) as f32,
                    ((s_bnir / num) / (s_nir / num)) as f32,
                    (s_var / num) as f32,
                ];
            }
        },
    );
    out
}

/// The CDI and the minimum variability of each block, `(cdi, minvar)` on the 20 m grid.
fn cdi_minvar(sc: &Scene, n_jobs: i32) -> Vec<[f32; 2]> {
    let (nx_, ny_) = (sc.nx / 2, sc.ny / 2);
    let ra = ratios(sc, n_jobs);
    let mut out = vec![[NODATA, 0.0f32]; nx_ * ny_];
    for_each_row(
        &mut out,
        nx_,
        n_jobs,
        || (),
        |_, i_, row| {
            for (j_, cell) in row.iter_mut().enumerate() {
                if ra[i_ * nx_ + j_][0] == 0.0 {
                    continue;
                }
                let (mut mx, mut my, mut vx, mut vy, mut k) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for ii in -3isize..=3 {
                    for jj in -3isize..=3 {
                        let ni = i_ as isize + ii;
                        let nj = j_ as isize + jj;
                        if ni < 0 || ni >= ny_ as isize || nj < 0 || nj >= nx_ as isize {
                            continue;
                        }
                        let r = &ra[ni as usize * nx_ + nj as usize];
                        if r[0] == 0.0 {
                            continue;
                        }
                        k += 1.0;
                        if k == 1.0 {
                            mx = r[0] as f64;
                            my = r[1] as f64;
                        } else {
                            // covar_recurrence (Pébay 2008), without the covariance, which nothing reads
                            let (x, y) = (r[0] as f64, r[1] as f64);
                            let (omx, omy) = (mx, my);
                            mx = omx + (x - omx) / k;
                            my = omy + (y - omy) / k;
                            vx += (x - omx) * (x - mx);
                            vy += (y - omy) * (y - my);
                        }
                    }
                }
                if k > 2.0 {
                    let varx = vx / (k - 1.0);
                    let vary = vy / (k - 1.0);
                    cell[0] = ((varx - vary) / (varx + vary)) as f32;
                }
                let mut min_var = i32::MAX as f64;
                for ii in -7isize..=7 {
                    for jj in -7isize..=7 {
                        let ni = i_ as isize + ii;
                        let nj = j_ as isize + jj;
                        if ni < 0 || ni >= ny_ as isize || nj < 0 || nj >= nx_ as isize {
                            continue;
                        }
                        let np = ni as usize * nx_ + nj as usize;
                        if ra[np][0] == 0.0 {
                            continue;
                        }
                        if (ra[np][2] as f64) < min_var {
                            min_var = ra[np][2] as f64;
                        }
                    }
                }
                if min_var != i32::MAX as f64 {
                    cell[1] = min_var as f32;
                }
            }
        },
    );
    out
}

/// The Cloud Displacement Index on the 20 m grid `(ny / 2, nx / 2)`, row-major; `NODATA` for a block that has
/// none (no potential cloud pixel in it, or three blocks or fewer in its 7 x 7 window).
pub fn displacement_index(sc: &Scene, n_jobs: i32) -> Vec<f32> {
    cdi_minvar(sc, n_jobs).into_iter().map(|c| c[0]).collect()
}

/// Clouds on the 20 m grid from the CDI and the minimum variability: the separation of clouds from built-up and
/// the two passes of region growing.
fn grow_clouds(cm: &[[f32; 2]], nx_: usize, ny_: usize, n_jobs: i32) -> Vec<bool> {
    let nodata = NODATA as f64;
    let mut clouds = vec![false; nx_ * ny_];
    for_each_row(
        &mut clouds,
        nx_,
        n_jobs,
        || (),
        |_, i_, row| {
            for (j_, cloud) in row.iter_mut().enumerate() {
                let [cdi, var3] = cm[i_ * nx_ + j_];
                let (cdi, var3) = (cdi as f64, var3 as f64);
                if cdi == nodata {
                    continue;
                }
                if cdi <= -0.5 || (var3 > 0.8 && cdi < 0.0) {
                    let mut valid = true;
                    for ii in -1isize..=1 {
                        for jj in -1isize..=1 {
                            let ni = i_ as isize + ii;
                            let nj = j_ as isize + jj;
                            if ni < 0 || ni >= ny_ as isize || nj < 0 || nj >= nx_ as isize {
                                continue;
                            }
                            let [c, v] = cm[ni as usize * nx_ + nj as usize];
                            let (c, v) = (c as f64, v as f64);
                            if c == nodata || (c > -0.5 && v < 0.8) {
                                valid = false;
                            }
                        }
                    }
                    if valid {
                        *cloud = true;
                    }
                }
            }
        },
    );

    let growable = |clouds: &[bool], np: usize| {
        let c = cm[np][0] as f64;
        !clouds[np] && (-1.0..-0.25).contains(&c)
    };
    // region growing 1: the cloud blocks with a growable neighbour
    let mut fifo: VecDeque<usize> = VecDeque::new();
    for i_ in 0..ny_ {
        for j_ in 0..nx_ {
            let p = i_ * nx_ + j_;
            if !clouds[p] {
                continue;
            }
            'nb: for ii in -1isize..=1 {
                for jj in -1isize..=1 {
                    if ii == 0 && jj == 0 {
                        continue;
                    }
                    let ni = i_ as isize + ii;
                    let nj = j_ as isize + jj;
                    if ni < 0 || ni >= ny_ as isize || nj < 0 || nj >= nx_ as isize {
                        continue;
                    }
                    if growable(&clouds, ni as usize * nx_ + nj as usize) {
                        fifo.push_back(p);
                        break 'nb;
                    }
                }
            }
        }
    }
    // region growing 2: the connected blocks with a fairly low CDI become clouds
    while let Some(p) = fifo.pop_front() {
        let (i_, j_) = ((p / nx_) as isize, (p % nx_) as isize);
        for ii in -1isize..=1 {
            for jj in -1isize..=1 {
                if ii == 0 && jj == 0 {
                    continue;
                }
                let (ni, nj) = (i_ + ii, j_ + jj);
                if ni < 0 || ni >= ny_ as isize || nj < 0 || nj >= nx_ as isize {
                    continue;
                }
                let np = ni as usize * nx_ + nj as usize;
                if growable(&clouds, np) {
                    clouds[np] = true;
                    fifo.push_back(np);
                }
            }
        }
    }
    clouds
}

/// The cloud mask at 10 m after the parallax test, and the CDI `(ny / 2, nx / 2)`. `bright` is FORCE's bright
/// pixels (the mean of blue, green and red above 0.15): a cloud pixel that is not bright is removed; `saturated`
/// pixels are added before that.
pub fn parallax_clouds(sc: &Scene, bright: &[bool], saturated: &[bool], n_jobs: i32) -> (Vec<bool>, Vec<f32>) {
    let (nx_, ny_) = (sc.nx / 2, sc.ny / 2);
    let cm = cdi_minvar(sc, n_jobs);
    let clouds = grow_clouds(&cm, nx_, ny_, n_jobs);
    let mut cld = vec![false; sc.nx * sc.ny];
    for i_ in 0..ny_ {
        for j_ in 0..nx_ {
            if clouds[i_ * nx_ + j_] {
                for i in i_ * 2..=i_ * 2 + 1 {
                    for j in j_ * 2..=j_ * 2 + 1 {
                        cld[i * sc.nx + j] = true;
                    }
                }
            }
        }
    }
    for p in 0..cld.len() {
        if saturated[p] {
            cld[p] = true;
        }
        if cld[p] && !bright[p] {
            cld[p] = false;
        }
    }
    (cld, cm.into_iter().map(|c| c[0]).collect())
}

// ---------------------------------------------------------------------------------------------------------------
// the buffer

/// The disc of radius `r` of FORCE's midpoint circle: `cover[dy][dx]` for `0 <= dy, dx <= r` is whether the pixel at
/// that offset is drawn around a pixel (the other quadrants by symmetry), as `buffer_` of FORCE's `imagefuns-cl.c`
/// draws it: at each step `(x, y)` the columns `+-x` over the rows `-y..=y` and the columns `+-y` over `-x..=x`.
fn disc(r: usize) -> Vec<Vec<bool>> {
    let mut cover = vec![vec![false; r + 1]; r + 1];
    let (mut d, mut x, mut y) = (3 - r as i64, 0i64, r as i64);
    while x <= y {
        for row in cover.iter_mut().take(y as usize + 1) {
            row[x as usize] = true;
        }
        for row in cover.iter_mut().take(x as usize + 1) {
            row[y as usize] = true;
        }
        if d < 0 {
            d += 4 * x + 6;
        } else {
            d += 4 * (x - y) + 10;
            y -= 1;
        }
        x += 1;
    }
    cover
}

/// Buffers the true pixels of `image` `(ny, nx)` by `r` pixels with FORCE's midpoint circle (`buffer_`). A radius
/// of 0 returns the image.
///
/// `force = true` is FORCE's code to the letter, which only buffers the pixels at least `r` from the borders and
/// then replaces the image by what was drawn: the true pixels within `r` of an edge are lost (and so is what they
/// would have drawn). With `force = false` every true pixel is buffered, drawn up to the image's edge.
pub fn buffer(image: &[bool], nx: usize, ny: usize, r: usize, force: bool, n_jobs: i32) -> Vec<bool> {
    if r == 0 || nx == 0 || ny == 0 {
        return image.to_vec();
    }
    let cover = disc(r);
    // per row offset, the runs [a, b] of column offsets drawn (symmetric: [a, b] and [-b, -a])
    let runs: Vec<Vec<(usize, usize)>> = cover
        .iter()
        .map(|row| {
            let mut v: Vec<(usize, usize)> = Vec::new();
            for (dx, &c) in row.iter().enumerate() {
                if c {
                    match v.last_mut() {
                        Some(l) if l.1 + 1 == dx => l.1 = dx,
                        _ => v.push((dx, dx)),
                    }
                }
            }
            v
        })
        .collect();
    let mut src = image.to_vec();
    if force {
        for i in 0..ny {
            for j in 0..nx {
                if i < r || i + r >= ny || j < r || j + r >= nx {
                    src[i * nx + j] = false;
                }
            }
        }
    }
    let has_any: Vec<bool> = src.chunks(nx).map(|row| row.iter().any(|&v| v)).collect();
    let mut out = vec![false; nx * ny];
    const NONE: u32 = u32::MAX;
    let init = || (vec![NONE; nx], vec![NONE; nx]);
    for_each_row(&mut out, nx, n_jobs, init, |(next, prev), i, row| {
        let r = r as isize;
        for dy in -r..=r {
            let si = i as isize + dy;
            if si < 0 || si >= ny as isize || !has_any[si as usize] {
                continue;
            }
            let s = &src[si as usize * nx..(si as usize + 1) * nx];
            let mut n = NONE;
            for j in (0..nx).rev() {
                if s[j] {
                    n = j as u32;
                }
                next[j] = n;
            }
            let mut p = NONE;
            for j in 0..nx {
                if s[j] {
                    p = j as u32;
                }
                prev[j] = p;
            }
            for &(a, b) in &runs[dy.unsigned_abs()] {
                for (j, hit) in row.iter_mut().enumerate() {
                    if *hit {
                        continue;
                    }
                    // a source in [j + a, j + b] or in [j - b, j - a]
                    if j + a < nx {
                        let q = next[j + a];
                        if q != NONE && q as usize <= j + b {
                            *hit = true;
                            continue;
                        }
                    }
                    if j >= a {
                        let q = prev[j - a];
                        if q != NONE && q as usize + b >= j {
                            *hit = true;
                        }
                    }
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FORCE's `buffer_`, as written (scatter, midpoint circle, borders skipped).
    fn buffer_force(image: &[bool], nx: usize, ny: usize, r: usize) -> Vec<bool> {
        if r == 0 {
            return image.to_vec();
        }
        let mut tmp = vec![false; nx * ny];
        for i in r..ny - r {
            for j in r..nx - r {
                if !image[nx * i + j] {
                    continue;
                }
                let (mut d, mut x, mut y) = (3 - r as i64, 0i64, r as i64);
                while x <= y {
                    for k in i as i64 - y..=i as i64 + y {
                        tmp[nx * k as usize + j + x as usize] = true;
                        tmp[nx * k as usize + j - x as usize] = true;
                    }
                    for k in i as i64 - x..=i as i64 + x {
                        tmp[nx * k as usize + j + y as usize] = true;
                        tmp[nx * k as usize + j - y as usize] = true;
                    }
                    if d < 0 {
                        d += 4 * x + 6;
                    } else {
                        d += 4 * (x - y) + 10;
                        y -= 1;
                    }
                    x += 1;
                }
            }
        }
        tmp
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[test]
    fn gauss_weights() {
        let k = gauss_kernel(5, 0.5);
        // sig = 0.5: k(0,0) = 1 / (pi * 0.5), k(1,0) = k(0,0) * exp(-2)
        assert!((k[12] as f64 - 1.0 / (std::f64::consts::PI * 0.5)).abs() < 1e-6);
        assert!((k[13] as f64 / k[12] as f64 - (-2.0f64).exp()).abs() < 1e-6);
        assert_eq!(k[11], k[13]);
        assert_eq!(k[7], k[17]);
    }

    #[test]
    fn buffer_matches_force_scatter() {
        let (nx, ny) = (37, 29);
        let mut seed = 7;
        for &r in &[1usize, 2, 3, 5, 8] {
            let img: Vec<bool> = (0..nx * ny).map(|_| lcg(&mut seed) % 40 == 0).collect();
            let want = buffer_force(&img, nx, ny, r);
            for jobs in [1, 3] {
                assert_eq!(buffer(&img, nx, ny, r, true, jobs), want, "r={r} jobs={jobs}");
            }
        }
    }

    #[test]
    fn buffer_not_force_keeps_edges() {
        let (nx, ny) = (20, 20);
        let mut img = vec![false; nx * ny];
        img[0] = true; // a corner pixel
        let f = buffer(&img, nx, ny, 3, true, 1);
        assert!(f.iter().all(|&v| !v), "FORCE loses it");
        let b = buffer(&img, nx, ny, 3, false, 1);
        // the disc of radius 3 clipped to the corner; (0, 0) itself is in
        assert!(b[0] && b[3] && b[3 * nx] && !b[4]);
        // away from the edge the two are the same
        let mut img = vec![false; nx * ny];
        img[10 * nx + 10] = true;
        assert_eq!(buffer(&img, nx, ny, 3, false, 1), buffer(&img, nx, ny, 3, true, 1));
        assert_eq!(buffer(&img, nx, ny, 0, true, 1), img);
    }

    #[test]
    fn buffer_radius_and_shape() {
        let (nx, ny) = (21, 21);
        let mut img = vec![false; nx * ny];
        img[10 * nx + 10] = true;
        let out = buffer(&img, nx, ny, 5, true, 1);
        // the midpoint disc: the axis points at distance 5 are in, (4, 4) (distance 5.66) is out
        assert!(out[10 * nx + 15] && out[10 * nx + 5] && out[15 * nx + 10] && out[5 * nx + 10]);
        assert!(!out[10 * nx + 16] && !out[14 * nx + 14]);
        assert!(out[13 * nx + 13]); // (3, 3), distance 4.2
    }

    /// A scene `(ny, nx)` with a flat surface: re3 = 3000, nir = 3000, bnir = 3000 everywhere.
    fn flat(ny: usize, nx: usize) -> (Vec<i16>, Vec<i16>, Vec<i16>, Vec<i16>, Vec<bool>, Vec<bool>) {
        let n = ny * nx;
        (
            vec![3000; n],
            vec![3000; n],
            vec![3000; n],
            vec![5000; n],
            vec![true; n],
            vec![false; n],
        )
    }

    #[test]
    fn displaced_object_has_negative_cdi() {
        let (ny, nx) = (60, 60);
        let (mut re3, mut bnir, mut nir, var, pcp, off) = flat(ny, nx);
        // a high bright object: B07 and B8A see it at the same place, B08 6 pixels (60 m) to the right
        for i in 20..40 {
            for j in 20..40 {
                re3[i * nx + j] = 5000;
                nir[i * nx + j] = 5000;
                bnir[i * nx + j + 6] = 5000;
            }
        }
        let sc = Scene {
            nx,
            ny,
            re3: &re3,
            bnir: &bnir,
            nir: &nir,
            var: &var,
            pcp: &pcp,
            off: &off,
        };
        let cdi = displacement_index(&sc, 1);
        // at its left edge only the B08 ratio varies: CDI = -1
        assert_eq!(cdi[10 * 30 + 10], -1.0);
        let bright = vec![true; nx * ny];
        let sat = vec![false; nx * ny];
        let (cld, cdi2) = parallax_clouds(&sc, &bright, &sat, 1);
        assert!(cdi.iter().zip(&cdi2).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(cld[28 * nx + 22] && cld[29 * nx + 23]);
        assert!(!cld[0] && !cld[(ny - 1) * nx + nx - 1]);
        // not bright: removed
        let (cld, _) = parallax_clouds(&sc, &vec![false; nx * ny], &sat, 1);
        assert!(cld.iter().all(|&c| !c));
        // a static bright object (all three bands see it where it is, the red edge ratio changing more): CDI > 0
        let (mut re3, mut bnir, mut nir, ..) = flat(ny, nx);
        for i in 20..40 {
            for j in 20..40 {
                re3[i * nx + j] = 4000;
                nir[i * nx + j] = 5000;
                bnir[i * nx + j] = 5500;
            }
        }
        let sc = Scene {
            nx,
            ny,
            re3: &re3,
            bnir: &bnir,
            nir: &nir,
            var: &var,
            pcp: &pcp,
            off: &off,
        };
        let cdi = displacement_index(&sc, 1);
        assert!(cdi[10 * 30 + 10] > 0.3, "{}", cdi[10 * 30 + 10]);
        let (cld, _) = parallax_clouds(&sc, &bright, &sat, 1);
        assert!(cld.iter().all(|&c| !c));
    }

    #[test]
    fn nodata_where_no_pcp_and_off() {
        let (ny, nx) = (40, 40);
        let (re3, bnir, nir, var, mut pcp, mut off) = flat(ny, nx);
        for p in pcp.iter_mut() {
            *p = false;
        }
        off[0] = true;
        let sc = Scene {
            nx,
            ny,
            re3: &re3,
            bnir: &bnir,
            nir: &nir,
            var: &var,
            pcp: &pcp,
            off: &off,
        };
        assert!(displacement_index(&sc, 1).iter().all(|&c| c == NODATA));
        let bright = vec![true; nx * ny];
        let sat = vec![false; nx * ny];
        let (cld, _) = parallax_clouds(&sc, &bright, &sat, 1);
        assert!(cld.iter().all(|&c| !c));
        // saturated pixels are added and then removed unless bright
        let mut sat = sat;
        sat[5] = true;
        sat[6] = true;
        let mut bright = bright;
        bright[6] = false;
        let (cld, _) = parallax_clouds(&sc, &bright, &sat, 1);
        assert!(cld[5] && !cld[6]);
        assert_eq!(cld.iter().filter(|&&c| c).count(), 1);
    }

    #[test]
    fn threads_do_not_change_the_result() {
        let (ny, nx) = (80, 90);
        let (mut re3, mut bnir, nir, var, pcp, off) = flat(ny, nx);
        let mut seed = 11;
        for p in 0..nx * ny {
            re3[p] = 2500 + (lcg(&mut seed) % 2000) as i16;
            bnir[p] = 2500 + (lcg(&mut seed) % 2000) as i16;
        }
        let sc = Scene {
            nx,
            ny,
            re3: &re3,
            bnir: &bnir,
            nir: &nir,
            var: &var,
            pcp: &pcp,
            off: &off,
        };
        let bright = vec![true; nx * ny];
        let sat = vec![false; nx * ny];
        let (c1, d1) = parallax_clouds(&sc, &bright, &sat, 1);
        let (c3, d3) = parallax_clouds(&sc, &bright, &sat, 3);
        assert_eq!(c1, c3);
        assert!(d1.iter().zip(&d3).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    #[test]
    fn odd_size_runs() {
        let (ny, nx) = (31, 33);
        let (re3, bnir, nir, var, pcp, off) = flat(ny, nx);
        let sc = Scene {
            nx,
            ny,
            re3: &re3,
            bnir: &bnir,
            nir: &nir,
            var: &var,
            pcp: &pcp,
            off: &off,
        };
        assert_eq!(displacement_index(&sc, 1).len(), 15 * 16);
    }
}
