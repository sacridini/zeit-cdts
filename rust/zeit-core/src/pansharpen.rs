//! Pansharpening: multispectral bands on the grid of a finer panchromatic band, from the published formulations. For any
//! sensor: any number of bands, any real `ratio` of the cell sizes (not necessarily whole), no band names or
//! sensor tables in here.
//!
//! The bands `MS_k` (k = 1..K) are already on the pan's grid (resampled by the caller; `zeit.pansharpen` does it
//! with the warp's cubic kernel) and `ratio` is the pan's cells' number per ms cell on a side. No third-party code
//! was read or used. The papers were not at hand when this was written: the formulas below are the methods as
//! recalled from the literature, checked by their properties, by Wald's protocol and, for the Brovey, by GDAL's output:
//!
//! * `Brovey`: the weighted Brovey transform (Gillespie, Kahle and Walker 1987, Remote Sensing of Environment
//!   22(3):343-365, for the ratio form): `F_k = MS_k P / I`, `I = sum_k w_k MS_k`. With `weights` given, they are
//!   normalised to sum 1 (so `I` is their mean and P must be in the units of MS). Without, the pan's spectral
//!   response being unknown, they are fitted: the least squares regression of the reduced pan on the reduced
//!   bands through the origin with the weights kept not negative (a band whose weight would be negative is
//!   dropped and the fit repeated), so `I` approximates the pan in the pan's own units. That differs from the
//!   equal weights in that a gain between P and MS is absorbed, and bands that the pan does not see (NIR, SWIR)
//!   get a weight near 0. Where `I` is not positive, `F_k = MS_k`. Every band is multiplied by `P / I`, those
//!   outside the pan's range included (the ratio form has no way to tell them apart): use a component
//!   substitution method for them.
//! * `Gsa`: Gram-Schmidt adaptive (Aiazzi, Baronti and Selva 2007, IEEE TGRS 45(10):3230-3239). The Gram-Schmidt
//!   transform has the intensity as its first component and the bands orthogonalised against it, and in the
//!   component-substitution form it is, as recalled (not checked against the paper's numbered equations; notation of
//!   Vivone et al. 2015, IEEE GRSM 3(1):67-81):
//!       `F_k = MS_k + g_k (P' - I)`,  `g_k = cov(MS_k, I) / var(I)`,  `P' = (P - mean P) sd(I) / sd(P) + mean I`,
//!   `I = b + sum_k w_k MS_k` the intensity, whose weights are "adaptive": the least squares regression of the pan
//!   reduced to the ms resolution on the ms bands, with an intercept,  `P_lr = b + sum_k w_k MS_lr_k`. A band
//!   outside the pan's range gets a weight near 0 and a gain `g_k` that follows its covariance with `I`: little
//!   detail goes in. Which bands enter the regression can be chosen (`select`); the others are still injected,
//!   by their own `g_k`.
//! * `Hpf`: high-pass filter fusion (Schowengerdt 1980, Remote Sensing: Models and Methods for Image Processing,
//!   for the idea; Gangkofner, Pradhan and Holcomb 2008, PE&RS 74(9):1107-1118, for the kernel and the
//!   modulation): `F_k = MS_k + gain m_k (P - LP(P))`,  `m_k = sd(MS_k) / sd(LP(P))`, `LP` the box (moving
//!   average) filter of side `kernel` (odd). It has no intensity: every band gets the detail in proportion to its
//!   own spread, whether the pan covers it or not (`gain_regression` gives each band the slope of its regression
//!   on the low-passed pan instead, so a band the pan does not explain gets less).
//! * `Pca`: principal components substitution (Chavez, Sides and Anderson 1991, PE&RS 57(3):295-303; Shettigara
//!   1992, PE&RS 58(5):561-567): `PC = V' (MS - mean)`, the first component is replaced by the pan matched to its
//!   mean (0) and variance and the inverse transform is `F = mean + V PC'`, which is the component-substitution
//!   form above with `I = PC_1`, `w = v_1` and `g = v_1`. With `select`, the components are those of the chosen
//!   bands and the others are injected with `g_k = cov(MS_k, I) / var(I)`. Needs 2 bands at least (the others: 1).
//!
//! What the papers do not fix, and was decided here:
//!
//! * The pan is reduced to the ms resolution (the regression of GSA and fitted Brovey) by the box filter of side
//!   `2 floor(ratio / 2) + 1` sampled every `round(ratio)` pixels, and the bands the same (they are interpolated
//!   already, so the filter only matches the pan's); the papers' low-pass filter is matched to the sensor's
//!   modulation transfer function, which is not known here. A sample enters only if the pan and every band have
//!   data at the pixel sampled. The grids' corners need not coincide: the samples are then off the cells' centres
//!   by up to half a pixel, which the filter's width hides.
//! * Pixels with NaN in the pan or in any band are NaN in the output, and are left out of every statistic (the
//!   regression, the means and variances, the principal components).
//! * HPF: the paper tabulates the kernel by ratio and that table was not available here (my recollection of it,
//!   `2 ratio + 1`, did not hold up: that box passes scales that the bands already resolve and double-counts
//!   them). The default kernel is the box that matches the bands' cell, the odd side `2 floor(ratio / 2) + 1`
//!   (3 for ratios 2 and 3, 5 for 4, 9 for 8; a ratio that is not whole is rounded down to the odd side), which
//!   gave the best ERGAS in Wald's protocol on a WPM window (ratio 4: kernel 5-7, against 9 and 13 worse) and on a
//!   Landsat 8 window (ratio 2: kernel 3, against 5 and 7 worse); `kernel` sets it. The `gain` is zeit's
//!   regression slope per band (`gain_regression`) by default and Gangkofner's modulation when not: the modulation
//!   did not beat the cubic resampling in ERGAS on the synthetic scenes or on the Landsat window. Gangkofner's final
//!   linear stretch of the result to the means and variances of the ms bands is not applied (the high-pass adds no
//!   mean; the stretch would shrink the detail again). The window is cut at the image border, and pan pixels
//!   with NaN are left out of it.
//! * Brovey with weights and HPF read the units of the data (they are not invariant to a gain of P); GSA, PCA
//!   and the fitted Brovey are invariant to a linear change of P.
//! * PCA: the first eigenvector is oriented to have a non-negative covariance with the pan (its sign is arbitrary).
//! * A constant pan, or a constant intensity, injects nothing (`F = MS`).
//!
//! Statistics are over the whole scene, in `f64`, in chunks of rows of a size that does not depend on the
//! threads: the output does not depend on `n_jobs`. Memory: the ms bands and the output, and one extra plane for
//! the box filters.

use crate::parallel::{pool, threads, Shared};
use rayon::prelude::*;

/// The fusion method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Brovey,
    Gsa,
    Hpf,
    Pca,
}

impl Method {
    pub fn parse(name: &str) -> Option<Method> {
        match name {
            "brovey" => Some(Method::Brovey),
            "gsa" => Some(Method::Gsa),
            "hpf" => Some(Method::Hpf),
            "pca" => Some(Method::Pca),
            _ => None,
        }
    }
}

/// What the methods read besides the data.
#[derive(Clone, Debug)]
pub struct Options {
    /// Pan cells per ms cell on a side: any real >= 1 (the ratio of the cell sizes, not necessarily whole).
    pub ratio: f64,
    /// Brovey: the weights of the intensity (one per band, normalised to sum 1). Empty: fitted (see above).
    pub weights: Vec<f64>,
    /// The bands (indices) that enter the estimate of the intensity of Brovey (fitted), GSA and PCA; empty: all.
    pub select: Vec<usize>,
    /// HPF: the box filter's side (odd); 0: `2 floor(ratio / 2) + 1`.
    pub kernel: usize,
    /// HPF: the global weight of the injected detail.
    pub gain: f64,
    /// HPF: instead of Gangkofner's `sd(MS_k) / sd(LP)`, the least squares slope `cov(MS_k, LP) / var(LP)` of each
    /// band on the low-passed pan (zeit's choice, not the paper's: the same for a band that is a linear function of
    /// the pan, smaller for a band that the pan does not explain).
    pub gain_regression: bool,
    pub n_jobs: i32,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            ratio: 1.0,
            weights: Vec::new(),
            select: Vec::new(),
            kernel: 0,
            gain: 1.0,
            gain_regression: false,
            n_jobs: -1,
        }
    }
}

/// The fused bands `(K, ny, nx)` and what the method estimated.
#[derive(Clone, Debug, Default)]
pub struct Fusion {
    pub data: Vec<f32>,
    /// Brovey: the weights used. GSA: the weights then the intercept. PCA: the first eigenvector then the
    /// first eigenvalue. HPF: the kernel side. Empty otherwise.
    pub weights: Vec<f64>,
    /// GSA and PCA: the injection gains `g_k`. HPF: `gain m_k`. Brovey: empty.
    pub gains: Vec<f64>,
}

/// The odd side of the box filter that reduces the pan (and the bands) to the ms resolution, and the step
/// between the samples of the regression: `2 floor(ratio / 2) + 1` and `round(ratio)`.
fn reduction(ratio: f64) -> (usize, usize) {
    ((2.0 * (ratio / 2.0).floor() + 1.0) as usize, (ratio.round() as usize).max(1))
}

/// Rows of a chunk of the statistics and of the box filter: fixed, so that the sums do not depend on the threads.
const CHUNK: usize = 64;

/// `f(i)` for `i` in `0..n` on the threads of `n_jobs`.
fn par_for<F: Fn(usize) + Sync + Send>(n: usize, n_jobs: i32, f: F) {
    let t = threads(n_jobs);
    if t <= 1 {
        (0..n).for_each(f);
    } else {
        pool(t).install(|| (0..n).into_par_iter().for_each(f));
    }
}

/// `f(c)` for the chunks `c` in `0..n`, in order.
fn par_map<R: Send, F: Fn(usize) -> R + Sync + Send>(n: usize, n_jobs: i32, f: F) -> Vec<R> {
    let t = threads(n_jobs);
    if t <= 1 {
        (0..n).map(f).collect()
    } else {
        pool(t).install(|| (0..n).into_par_iter().map(f).collect())
    }
}

/// Means and covariance (divided by n) of the columns over the pixels where all of them are finite.
struct Moments {
    n: u64,
    mean: Vec<f64>,
    /// `m` x `m`, row-major.
    cov: Vec<f64>,
}

fn moments(cols: &[&[f32]], ny: usize, nx: usize, n_jobs: i32) -> Moments {
    let m = cols.len();
    let nchunks = ny.div_ceil(CHUNK);
    let rows = |c: usize| c * CHUNK..((c + 1) * CHUNK).min(ny);
    let finite = |p: usize| cols.iter().all(|col| col[p].is_finite());
    let sums = par_map(nchunks, n_jobs, |c| {
        let mut s = vec![0.0f64; m];
        let mut n = 0u64;
        for p in rows(c).start * nx..rows(c).end * nx {
            if finite(p) {
                n += 1;
                for (a, col) in s.iter_mut().zip(cols) {
                    *a += col[p] as f64;
                }
            }
        }
        (n, s)
    });
    let n: u64 = sums.iter().map(|s| s.0).sum();
    let mut mean = vec![0.0f64; m];
    for (_, s) in &sums {
        for (a, b) in mean.iter_mut().zip(s) {
            *a += b;
        }
    }
    if n > 0 {
        mean.iter_mut().for_each(|a| *a /= n as f64);
    }
    let parts = par_map(nchunks, n_jobs, |c| {
        let mut s = vec![0.0f64; m * m];
        let mut d = vec![0.0f64; m];
        for p in rows(c).start * nx..rows(c).end * nx {
            if finite(p) {
                for j in 0..m {
                    d[j] = cols[j][p] as f64 - mean[j];
                }
                for j in 0..m {
                    for k in j..m {
                        s[j * m + k] += d[j] * d[k];
                    }
                }
            }
        }
        s
    });
    let mut cov = vec![0.0f64; m * m];
    for s in &parts {
        for (a, b) in cov.iter_mut().zip(s) {
            *a += b;
        }
    }
    for j in 0..m {
        for k in j..m {
            let v = if n > 0 { cov[j * m + k] / n as f64 } else { 0.0 };
            cov[j * m + k] = v;
            cov[k * m + j] = v;
        }
    }
    Moments { n, mean, cov }
}

/// The eigenvalues (descending) and eigenvectors (column j of the row-major `n` x `n` result) of the symmetric
/// matrix `a`, by cyclic Jacobi rotations.
fn sym_eig(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j] * a[i * n + j])
            .sum();
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].partial_cmp(&a[i * n + i]).unwrap_or(std::cmp::Ordering::Equal));
    let vals = order.iter().map(|&i| a[i * n + i]).collect();
    let mut vecs = vec![0.0f64; n * n];
    for (j, &src) in order.iter().enumerate() {
        for i in 0..n {
            vecs[i * n + j] = v[i * n + src];
        }
    }
    (vals, vecs)
}

/// `pinv(a) y` for a symmetric positive semi-definite `a` (eigenvalues under 1e-10 of the largest dropped:
/// a constant or collinear band does not break the regression).
fn solve_sym(a: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let (vals, vecs) = sym_eig(a, n);
    let tol = 1e-10 * vals.first().copied().unwrap_or(0.0).max(0.0);
    let mut x = vec![0.0f64; n];
    for (j, &lam) in vals.iter().enumerate() {
        if lam > tol && lam > 0.0 {
            let proj: f64 = (0..n).map(|i| vecs[i * n + j] * y[i]).sum::<f64>() / lam;
            for i in 0..n {
                x[i] += proj * vecs[i * n + j];
            }
        }
    }
    x
}

/// The box filter of side `k` (odd) of `x`, the window cut at the border and the non-finite values left out
/// (the mean of the finite ones; NaN if there is none), in O(1) a pixel.
fn box_filter(x: &[f32], ny: usize, nx: usize, k: usize, n_jobs: i32) -> Vec<f32> {
    let h = k / 2;
    let mut out = vec![0.0f32; x.len()];
    let sh = Shared::new(&mut out);
    par_for(ny.div_ceil(CHUNK), n_jobs, |c| {
        let a = c * CHUNK;
        let b = ((c + 1) * CHUNK).min(ny);
        // rows a-h .. b+h, clipped, horizontally summed
        let lo = a.saturating_sub(h);
        let hi = (b + h).min(ny);
        let nr = hi - lo;
        let mut hs = vec![0.0f64; nr * nx];
        let mut hc = vec![0u32; nr * nx];
        for r in 0..nr {
            let row = &x[(lo + r) * nx..(lo + r + 1) * nx];
            let (s, n) = (&mut hs[r * nx..(r + 1) * nx], &mut hc[r * nx..(r + 1) * nx]);
            let (mut sum, mut cnt) = (0.0f64, 0u32);
            // window of column j: j-h ..= j+h, clipped
            for j in 0..h.min(nx) {
                if row[j].is_finite() {
                    sum += row[j] as f64;
                    cnt += 1;
                }
            }
            for j in 0..nx {
                if j + h < nx && row[j + h].is_finite() {
                    sum += row[j + h] as f64;
                    cnt += 1;
                }
                if j > h && row[j - h - 1].is_finite() {
                    sum -= row[j - h - 1] as f64;
                    cnt -= 1;
                }
                s[j] = sum;
                n[j] = cnt;
            }
        }
        // vertical sliding sums over the rows a-h ..= a+h, then moving down
        let mut vs = vec![0.0f64; nx];
        let mut vc = vec![0u32; nx];
        let first_lo = a.saturating_sub(h);
        let first_hi = (a + h).min(ny - 1);
        for r in first_lo..=first_hi {
            let rr = r - lo;
            for j in 0..nx {
                vs[j] += hs[rr * nx + j];
                vc[j] += hc[rr * nx + j];
            }
        }
        for i in a..b {
            if i > a {
                if i + h < ny {
                    let rr = i + h - lo;
                    for j in 0..nx {
                        vs[j] += hs[rr * nx + j];
                        vc[j] += hc[rr * nx + j];
                    }
                }
                if i > h {
                    let rr = i - h - 1 - lo;
                    for j in 0..nx {
                        vs[j] -= hs[rr * nx + j];
                        vc[j] -= hc[rr * nx + j];
                    }
                }
            }
            // SAFETY: the chunks write disjoint rows.
            let dst = unsafe { sh.slice_mut(i * nx..(i + 1) * nx) };
            for j in 0..nx {
                dst[j] = if vc[j] > 0 { (vs[j] / vc[j] as f64) as f32 } else { f32::NAN };
            }
        }
    });
    out
}

/// The columns reduced to the ms resolution: each filtered by the box of side `k` (odd; the identity for 1) and
/// sampled every `step` pixels, `(by, bx)` samples; NaN in all of them where any column is not finite at the
/// pixel sampled (a window that touches a gap averages the pixels around it that have data).
fn reduced(cols: &[&[f32]], ny: usize, nx: usize, k: usize, step: usize, n_jobs: i32) -> (Vec<Vec<f32>>, usize, usize) {
    let (by, bx) = (ny / step, nx / step);
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(cols.len());
    let pos = |m: usize| m * step + step / 2;
    for col in cols {
        let filtered = if k > 1 { Some(box_filter(col, ny, nx, k, n_jobs)) } else { None };
        let src: &[f32] = filtered.as_deref().unwrap_or(col);
        let mut v = vec![f32::NAN; by * bx];
        let sh = Shared::new(&mut v);
        par_for(by, n_jobs, |bi| {
            // SAFETY: each sample row is written by one task.
            let row = unsafe { sh.slice_mut(bi * bx..(bi + 1) * bx) };
            for (bj, r) in row.iter_mut().enumerate() {
                *r = src[pos(bi) * nx + pos(bj)];
            }
        });
        out.push(v);
    }
    // joint validity: the pixel sampled has data in every column
    let mut ok = vec![true; by * bx];
    for col in cols {
        for bi in 0..by {
            for bj in 0..bx {
                if !col[pos(bi) * nx + pos(bj)].is_finite() {
                    ok[bi * bx + bj] = false;
                }
            }
        }
    }
    for v in out.iter_mut() {
        for (x, o) in v.iter_mut().zip(&ok) {
            if !o {
                *x = f32::NAN;
            }
        }
    }
    (out, by, bx)
}

/// `F_k = MS_k + g_k (P' - I)` with `I = b + sum w_k MS_k` and `P'` the pan matched to the mean and standard
/// deviation of `I` (`P' = I` when either is constant).
struct Cs<'a> {
    w: &'a [f64],
    b: f64,
    g: &'a [f64],
    mu_p: f64,
    sd_p: f64,
    mu_i: f64,
    sd_i: f64,
}

fn run_rows<F: Fn(usize, &[f32], f32, &mut [f64]) -> bool + Sync + Send>(
    ms: &[f32],
    pan: &[f32],
    nb: usize,
    ny: usize,
    nx: usize,
    n_jobs: i32,
    f: F,
) -> Vec<f32> {
    let npix = ny * nx;
    let mut out = vec![0.0f32; nb * npix];
    let sh = Shared::new(&mut out);
    par_for(ny, n_jobs, |i| {
        let mut px = vec![0.0f32; nb];
        let mut res = vec![0.0f64; nb];
        // SAFETY: each row of each band is written by one task.
        let mut rows: Vec<&mut [f32]> = (0..nb).map(|k| unsafe { sh.slice_mut(k * npix + i * nx..k * npix + (i + 1) * nx) }).collect();
        for j in 0..nx {
            let p = i * nx + j;
            let mut ok = pan[p].is_finite();
            for k in 0..nb {
                px[k] = ms[k * npix + p];
                ok &= px[k].is_finite();
            }
            if ok && f(p, &px, pan[p], &mut res) {
                for k in 0..nb {
                    rows[k][j] = res[k] as f32;
                }
            } else {
                for k in 0..nb {
                    rows[k][j] = f32::NAN;
                }
            }
        }
    });
    out
}

fn inject(ms: &[f32], pan: &[f32], nb: usize, ny: usize, nx: usize, n_jobs: i32, cs: &Cs) -> Vec<f32> {
    run_rows(ms, pan, nb, ny, nx, n_jobs, |_, px, p, res| {
        let mut i = cs.b;
        for k in 0..nb {
            i += cs.w[k] * px[k] as f64;
        }
        let pm = if cs.sd_p > 0.0 && cs.sd_i > 0.0 {
            (p as f64 - cs.mu_p) * cs.sd_i / cs.sd_p + cs.mu_i
        } else {
            i
        };
        let d = pm - i;
        for k in 0..nb {
            res[k] = px[k] as f64 + cs.g[k] * d;
        }
        true
    })
}

/// `F_k = MS_k P / I` with `I = sum w_k MS_k`; `F_k = MS_k` where `I` is not positive.
fn brovey(ms: &[f32], pan: &[f32], nb: usize, ny: usize, nx: usize, n_jobs: i32, w: &[f64]) -> Vec<f32> {
    run_rows(ms, pan, nb, ny, nx, n_jobs, |_, px, p, res| {
        let i: f64 = (0..nb).map(|k| w[k] * px[k] as f64).sum();
        for k in 0..nb {
            res[k] = if i > 0.0 { px[k] as f64 * p as f64 / i } else { px[k] as f64 };
        }
        true
    })
}

/// Fuses `ms` `(nb, ny, nx)` (the bands on the pan's grid) with `pan` `(ny, nx)`.
pub fn pansharpen(
    ms: &[f32],
    pan: &[f32],
    nb: usize,
    ny: usize,
    nx: usize,
    method: Method,
    opt: &Options,
) -> Result<Fusion, String> {
    let npix = ny * nx;
    if nb == 0 || npix == 0 {
        return Err("pansharpen needs at least one band and one pixel".into());
    }
    if ms.len() != nb * npix || pan.len() != npix {
        return Err(format!(
            "ms has {} values and pan {}, for {nb} bands of {ny} x {nx}",
            ms.len(),
            pan.len()
        ));
    }
    if !(opt.ratio >= 1.0) || !opt.ratio.is_finite() {
        return Err(format!("ratio must be a number of at least 1, got {}", opt.ratio));
    }
    let nj = opt.n_jobs;
    let bands: Vec<&[f32]> = (0..nb).map(|k| &ms[k * npix..(k + 1) * npix]).collect();
    let sel: Vec<usize> = if opt.select.is_empty() { (0..nb).collect() } else { opt.select.clone() };
    {
        let mut seen = vec![false; nb];
        for &i in &sel {
            if i >= nb || seen[i] {
                return Err(format!("the bands of the intensity must be distinct indices below {nb}, got {sel:?}"));
            }
            seen[i] = true;
        }
    }
    let ns = sel.len();
    match method {
        Method::Brovey if !opt.weights.is_empty() => {
            let mut w = opt.weights.clone();
            if w.len() != nb {
                return Err(format!("weights has {} values for {nb} bands", w.len()));
            }
            if w.iter().any(|x| !x.is_finite() || *x < 0.0) || w.iter().sum::<f64>() <= 0.0 {
                return Err("weights must be finite, not negative and not all zero".into());
            }
            let s: f64 = w.iter().sum();
            w.iter_mut().for_each(|x| *x /= s);
            let data = brovey(ms, pan, nb, ny, nx, nj, &w);
            Ok(Fusion { data, weights: w, gains: Vec::new() })
        }
        Method::Brovey | Method::Gsa | Method::Pca => {
            if method == Method::Pca && ns < 2 {
                return Err("pca needs at least 2 bands in the intensity".into());
            }
            let mut cols = bands.clone();
            cols.push(pan);
            let m = moments(&cols, ny, nx, nj);
            if m.n < 2 {
                return Err("fewer than 2 pixels have data in the pan and in every band".into());
            }
            let stride = nb + 1;
            let c: Vec<f64> = (0..nb).flat_map(|j| (0..nb).map(move |k| (j, k))).map(|(j, k)| m.cov[j * stride + k]).collect();
            let mut w = vec![0.0f64; nb];
            let (b, mu_i);
            let mut weights;
            if method == Method::Pca {
                let csel: Vec<f64> = sel.iter().flat_map(|&j| sel.iter().map(move |&k| (j, k))).map(|(j, k)| c[j * nb + k]).collect();
                let (vals, vecs) = sym_eig(&csel, ns);
                for (a, &j) in sel.iter().enumerate() {
                    w[j] = vecs[a * ns];
                }
                // an eigenvector's sign is arbitrary: the component must follow the pan that replaces it
                if (0..nb).map(|k| w[k] * m.cov[k * stride + nb]).sum::<f64>() < 0.0 {
                    w.iter_mut().for_each(|x| *x = -*x);
                }
                b = -(0..nb).map(|k| w[k] * m.mean[k]).sum::<f64>();
                weights = w.clone();
                weights.push(vals[0]);
                mu_i = 0.0;
            } else {
                // the regression of the reduced pan on the reduced bands
                let (k, step) = reduction(opt.ratio);
                let (red, by, bx) = reduced(&cols, ny, nx, k, step, nj);
                let rc: Vec<&[f32]> = red.iter().map(|v| v.as_slice()).collect();
                let rm = moments(&rc, by, bx, nj);
                if rm.n < (ns + 2) as u64 {
                    return Err(format!(
                        "only {} samples of the pan reduced to the resolution of the bands have data, too few to fit the intensity of {ns} bands",
                        rm.n
                    ));
                }
                if method == Method::Gsa {
                    let cxx: Vec<f64> = sel.iter().flat_map(|&j| sel.iter().map(move |&k| (j, k))).map(|(j, k)| rm.cov[j * stride + k]).collect();
                    let cxy: Vec<f64> = sel.iter().map(|&j| rm.cov[j * stride + nb]).collect();
                    let x = solve_sym(&cxx, &cxy, ns);
                    for (a, &j) in sel.iter().enumerate() {
                        w[j] = x[a];
                    }
                    b = rm.mean[nb] - (0..nb).map(|k| w[k] * rm.mean[k]).sum::<f64>();
                    weights = w.clone();
                    weights.push(b);
                } else {
                    // Brovey: through the origin (I = sum w_k MS_k), weights not negative: bands that the pan
                    // does not see drop out
                    let mut active = sel.clone();
                    loop {
                        let na = active.len();
                        let g: Vec<f64> = active.iter().flat_map(|&j| active.iter().map(move |&k| (j, k))).map(|(j, k)| rm.cov[j * stride + k] + rm.mean[j] * rm.mean[k]).collect();
                        let h: Vec<f64> = active.iter().map(|&j| rm.cov[j * stride + nb] + rm.mean[j] * rm.mean[nb]).collect();
                        let x = solve_sym(&g, &h, na);
                        match x.iter().enumerate().filter(|(_, v)| **v < 0.0).min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
                            Some((worst, _)) if na > 1 => {
                                active.remove(worst);
                            }
                            _ => {
                                for (a, &j) in active.iter().enumerate() {
                                    w[j] = x[a].max(0.0);
                                }
                                break;
                            }
                        }
                    }
                    let data = brovey(ms, pan, nb, ny, nx, nj, &w);
                    return Ok(Fusion { data, weights: w, gains: Vec::new() });
                }
                mu_i = b + (0..nb).map(|k| w[k] * m.mean[k]).sum::<f64>();
            }
            // cov(MS_k, I) = (C w)_k, var(I) = w' C w
            let cw: Vec<f64> = (0..nb).map(|j| (0..nb).map(|k| c[j * nb + k] * w[k]).sum()).collect();
            let var_i: f64 = (0..nb).map(|k| w[k] * cw[k]).sum();
            let sd_i = var_i.max(0.0).sqrt();
            let g: Vec<f64> = if var_i > 0.0 { cw.iter().map(|x| x / var_i).collect() } else { vec![0.0; nb] };
            let cs = Cs {
                w: &w,
                b,
                g: &g,
                mu_p: m.mean[nb],
                sd_p: m.cov[nb * stride + nb].max(0.0).sqrt(),
                mu_i,
                sd_i,
            };
            let data = inject(ms, pan, nb, ny, nx, nj, &cs);
            Ok(Fusion { data, weights, gains: g })
        }
        Method::Hpf => {
            let k = if opt.kernel == 0 { reduction(opt.ratio).0 } else { opt.kernel };
            if k % 2 == 0 {
                return Err(format!("kernel must be odd, got {k}"));
            }
            if !opt.gain.is_finite() {
                return Err("gain must be finite".into());
            }
            let lp = box_filter(pan, ny, nx, k, nj);
            let mut cols = bands.clone();
            cols.push(&lp);
            cols.push(pan);
            let m = moments(&cols, ny, nx, nj);
            if m.n < 2 {
                return Err("fewer than 2 pixels have data in the pan and in every band".into());
            }
            let sd_lp = m.cov[nb * (nb + 2) + nb].max(0.0).sqrt();
            let gains: Vec<f64> = (0..nb)
                .map(|j| {
                    let sd = m.cov[j * (nb + 2) + j].max(0.0).sqrt();
                    if sd_lp <= 0.0 {
                        0.0
                    } else if opt.gain_regression {
                        opt.gain * m.cov[j * (nb + 2) + nb] / (sd_lp * sd_lp)
                    } else {
                        opt.gain * sd / sd_lp
                    }
                })
                .collect();
            let data = run_rows(ms, pan, nb, ny, nx, nj, |p, px, pv, res| {
                let hp = pv as f64 - lp[p] as f64;
                for kb in 0..nb {
                    res[kb] = px[kb] as f64 + gains[kb] * hp;
                }
                true
            });
            Ok(Fusion { data, weights: vec![k as f64], gains })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth scene of `nb` bands on `ny` x `nx` (a multiple of `r`), the pan a mix of them plus fine detail.
    fn scene(nb: usize, ny: usize, nx: usize) -> (Vec<f32>, Vec<f32>) {
        let npix = ny * nx;
        let mut ms = vec![0.0f32; nb * npix];
        let mut pan = vec![0.0f32; npix];
        for i in 0..ny {
            for j in 0..nx {
                let p = i * nx + j;
                let base = (i as f64 * 0.21).sin() + (j as f64 * 0.17).cos() + 0.5 * ((i + 2 * j) as f64 * 0.05).sin();
                let mut mix = 0.0;
                for k in 0..nb {
                    let v = 300.0 + 100.0 * (k as f64 + 1.0) + 60.0 * base * (1.0 + 0.3 * k as f64)
                        + 20.0 * ((i * (k + 3)) as f64 * 0.13).cos();
                    ms[k * npix + p] = v as f32;
                    mix += v / nb as f64;
                }
                pan[p] = (mix + 15.0 * (((i * 7 + j * 3) % 11) as f64 - 5.0)) as f32;
            }
        }
        (ms, pan)
    }

    #[test]
    fn pca_component_follows_the_pan() {
        let (ny, nx, nb) = (40usize, 40usize, 3usize);
        let npix = ny * nx;
        let mut ms = vec![0.0f32; nb * npix];
        let mut pan = vec![0.0f32; npix];
        for p in 0..npix {
            let b = (p as f64 * 0.37).sin() * 50.0;
            for k in 0..nb {
                ms[k * npix + p] = (200.0 + (k as f64 + 1.0) * b) as f32;
            }
            pan[p] = (1000.0 - b) as f32;
        }
        let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Pca, &opt(4.0, 2)).unwrap();
        assert!(f.weights[..nb].iter().all(|w| *w < 0.0), "the bands rise as the pan falls: {:?}", f.weights);
    }

    fn opt(r: f64, n_jobs: i32) -> Options {
        Options { ratio: r, n_jobs, ..Options::default() }
    }

    #[test]
    fn box_filter_equals_the_naive_mean() {
        let (ny, nx) = (150, 37);
        let mut x: Vec<f32> = (0..ny * nx).map(|p| ((p * 31) % 97) as f32).collect();
        x[5 * nx + 5] = f32::NAN;
        x[100 * nx + 36] = f32::NAN;
        for k in [1usize, 3, 9, 13] {
            let got = box_filter(&x, ny, nx, k, 3);
            let h = k / 2;
            for i in 0..ny {
                for j in 0..nx {
                    let (mut s, mut n) = (0.0f64, 0);
                    for a in i.saturating_sub(h)..=(i + h).min(ny - 1) {
                        for b in j.saturating_sub(h)..=(j + h).min(nx - 1) {
                            let v = x[a * nx + b];
                            if v.is_finite() {
                                s += v as f64;
                                n += 1;
                            }
                        }
                    }
                    let want = (s / n as f64) as f32;
                    assert!((got[i * nx + j].is_nan() && want.is_nan()) || (got[i * nx + j] - want).abs() < 1e-3, "k={k} at ({i},{j}): {} vs {want}", got[i * nx + j]);
                }
            }
        }
    }

    #[test]
    fn jacobi_reproduces_a_symmetric_matrix() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let (vals, v) = sym_eig(&a, 3);
        assert!(vals[0] >= vals[1] && vals[1] >= vals[2]);
        for i in 0..3 {
            for j in 0..3 {
                let r: f64 = (0..3).map(|k| v[i * 3 + k] * vals[k] * v[j * 3 + k]).sum();
                assert!((r - a[i * 3 + j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn constants_stay_constants() {
        let (nb, ny, nx) = (3, 16, 16);
        let ms: Vec<f32> = (0..nb * ny * nx).map(|p| 100.0 * (1 + p / (ny * nx)) as f32).collect();
        let pan = vec![250.0f32; ny * nx];
        for m in [Method::Brovey, Method::Gsa, Method::Hpf, Method::Pca] {
            let o = if m == Method::Brovey { Options { weights: vec![1.0; nb], ..opt(4.0, 2) } } else { opt(4.0, 2) };
            let f = pansharpen(&ms, &pan, nb, ny, nx, m, &o).unwrap();
            for (a, b) in f.data.iter().zip(&ms) {
                match m {
                    // pan 250 over a mean intensity of 200: the ratio form scales
                    Method::Brovey => assert!((a / b - 1.25).abs() < 1e-5),
                    _ => assert!((a - b).abs() < 1e-3, "{m:?}: {a} vs {b}"),
                }
            }
        }
    }

    #[test]
    fn pan_equal_to_the_intensity_changes_nothing() {
        let (nb, ny, nx, rf) = (4, 32, 40, 4.0);
        let (ms, _) = scene(nb, ny, nx);
        let npix = ny * nx;
        // GSA: the pan is an affine function of the bands (so is the regression's intensity)
        let w = [0.1, 0.4, 0.3, 0.2];
        let pan: Vec<f32> = (0..npix)
            .map(|p| (7.0 + 2.0 * (0..nb).map(|k| w[k] * ms[k * npix + p] as f64).sum::<f64>()) as f32)
            .collect();
        let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Gsa, &opt(rf, 2)).unwrap();
        for (a, b) in f.data.iter().zip(&ms) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
        for k in 0..nb {
            assert!((f.weights[k] - 2.0 * w[k]).abs() < 1e-3, "{:?}", f.weights);
        }
        // Brovey: pan equal to the weighted intensity
        let pan: Vec<f32> = (0..npix).map(|p| (0..nb).map(|k| ms[k * npix + p] / nb as f32).sum()).collect();
        let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Brovey, &opt(rf, 1)).unwrap();
        for (a, b) in f.data.iter().zip(&ms) {
            assert!((a / b - 1.0).abs() < 1e-5);
        }
        // HPF: a pan that its box filter does not change
        let flat = vec![321.0f32; npix];
        let f = pansharpen(&ms, &flat, nb, ny, nx, Method::Hpf, &opt(rf, 2)).unwrap();
        for (a, b) in f.data.iter().zip(&ms) {
            assert!((a - b).abs() < 1e-3);
        }
        // PCA: the pan equal to the first component
        let mut cols: Vec<&[f32]> = (0..nb).map(|k| &ms[k * npix..(k + 1) * npix]).collect();
        let m = moments(&cols, ny, nx, 1);
        cols.clear();
        let c: Vec<f64> = m.cov.clone();
        let (_, v) = sym_eig(&c, nb);
        let pan: Vec<f32> = (0..npix)
            .map(|p| (500.0 + 3.0 * (0..nb).map(|k| v[k * nb] * (ms[k * npix + p] as f64 - m.mean[k])).sum::<f64>()) as f32)
            .collect();
        let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Pca, &opt(rf, 2)).unwrap();
        for (a, b) in f.data.iter().zip(&ms) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn nan_pixels_are_nan_and_out_of_the_statistics() {
        let (nb, ny, nx, rf) = (3, 32, 32, 4.0);
        let (mut ms, mut pan) = scene(nb, ny, nx);
        let clean: Vec<_> = [Method::Brovey, Method::Gsa, Method::Hpf, Method::Pca]
            .iter()
            .map(|&m| pansharpen(&ms, &pan, nb, ny, nx, m, &opt(rf, 2)).unwrap())
            .collect();
        let npix = ny * nx;
        pan[3 * nx + 4] = f32::NAN;
        ms[npix + 20 * nx + 20] = f32::NAN;
        for (m, c) in [Method::Brovey, Method::Gsa, Method::Hpf, Method::Pca].iter().zip(&clean) {
            let f = pansharpen(&ms, &pan, nb, ny, nx, *m, &opt(rf, 2)).unwrap();
            for k in 0..nb {
                assert!(f.data[k * npix + 3 * nx + 4].is_nan() && f.data[k * npix + 20 * nx + 20].is_nan());
            }
            assert_eq!(f.data.iter().filter(|v| v.is_nan()).count(), 2 * nb, "{m:?}");
            // far from the gaps the statistics barely move
            let d = (f.data[10 * nx + 10] - c.data[10 * nx + 10]).abs();
            assert!(d < 10.0, "{m:?}: {d}");
        }
    }

    #[test]
    fn the_output_does_not_depend_on_the_threads() {
        let (nb, ny, nx, rf) = (4, 150, 130, 4.0);
        let (ms, pan) = scene(nb, ny, nx);
        for m in [Method::Brovey, Method::Gsa, Method::Hpf, Method::Pca] {
            let a = pansharpen(&ms, &pan, nb, ny, nx, m, &opt(rf, 1)).unwrap();
            for t in [2, 5] {
                let b = pansharpen(&ms, &pan, nb, ny, nx, m, &opt(rf, t)).unwrap();
                assert_eq!(a.data, b.data, "{m:?} with {t} threads");
            }
        }
    }

    #[test]
    fn any_ratio_fits_the_intensity_and_the_fitted_brovey_drops_the_bands_the_pan_does_not_see() {
        let (nb, ny, nx) = (4, 48, 52);
        let npix = ny * nx;
        let (ms, _) = scene(nb, ny, nx);
        // a pan that sees only the first two bands, in its own units
        let pan: Vec<f32> = (0..npix).map(|p| (7.0 + 1.5 * (ms[p] + ms[npix + p]) as f64) as f32).collect();
        for rf in [1.5, 2.0, 2.5, 3.0, 4.33, 8.0] {
            let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Gsa, &opt(rf, 2)).unwrap();
            for (a, b) in f.data.iter().zip(&ms) {
                assert!((a - b).abs() < 0.1, "ratio {rf}: {a} vs {b}");
            }
            assert!((f.weights[0] - 1.5).abs() < 1e-2 && (f.weights[1] - 1.5).abs() < 1e-2, "{:?}", f.weights);
            assert!(f.weights[2].abs() < 1e-2 && f.weights[3].abs() < 1e-2, "{:?}", f.weights);
            // through the origin there is no intercept: 7 is not absorbed, but the weights stay near
            let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Brovey, &opt(rf, 2)).unwrap();
            assert!(f.weights[2] < 0.05 && f.weights[3] < 0.05 && f.weights[0] > 1.0, "{:?}", f.weights);
        }
        // the intensity from the first two bands only
        let o = Options { select: vec![0, 1], ..opt(4.0, 2) };
        let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Gsa, &o).unwrap();
        assert!(f.weights[2] == 0.0 && f.weights[3] == 0.0 && (f.weights[0] - 1.5).abs() < 1e-2);
        let f = pansharpen(&ms, &pan, nb, ny, nx, Method::Pca, &o).unwrap();
        assert!(f.weights[2] == 0.0 && f.weights[3] == 0.0);
        let bad = Options { select: vec![0, 0], ..opt(4.0, 2) };
        assert!(pansharpen(&ms, &pan, nb, ny, nx, Method::Gsa, &bad).is_err());
        let bad = Options { select: vec![9], ..opt(4.0, 2) };
        assert!(pansharpen(&ms, &pan, nb, ny, nx, Method::Gsa, &bad).is_err());
    }

    #[test]
    fn one_band_and_many_bands() {
        for nb in [1usize, 3, 8] {
            let (ms, pan) = scene(nb, 32, 32);
            for m in [Method::Brovey, Method::Gsa, Method::Hpf] {
                let f = pansharpen(&ms, &pan, nb, 32, 32, m, &opt(3.0, 2)).unwrap();
                assert!(f.data.iter().all(|v| v.is_finite()), "{m:?} {nb}");
            }
            assert_eq!(pansharpen(&ms, &pan, nb, 32, 32, Method::Pca, &opt(3.0, 2)).is_ok(), nb > 1);
        }
    }

    #[test]
    fn errors() {
        let (ms, pan) = scene(3, 8, 8);
        assert!(pansharpen(&ms, &pan, 3, 8, 8, Method::Pca, &Options { ratio: 0.5, ..opt(1.0, 1) }).is_err());
        assert!(pansharpen(&ms, &pan[1..], 3, 8, 8, Method::Gsa, &opt(2.0, 1)).is_err());
        let bad = Options { weights: vec![1.0, 2.0], ..opt(2.0, 1) };
        assert!(pansharpen(&ms, &pan, 3, 8, 8, Method::Brovey, &bad).is_err());
        let even = Options { kernel: 4, ..opt(2.0, 1) };
        assert!(pansharpen(&ms, &pan, 3, 8, 8, Method::Hpf, &even).is_err());
    }
}
