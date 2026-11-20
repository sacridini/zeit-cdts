//! Spatiotemporal fusion of a fine and a coarse series: STARFM (Gao et al. 2006) and ESTARFM (Zhu et al. 2010).
//!
//! Given fine images (Landsat, HLS, Sentinel-2) on a few dates that have a coarse image (MODIS) of the same date, and
//! the coarse series on all its dates, the methods predict the fine image on every coarse date. This module is
//! written from the two papers (and the open restatements of them listed in the user documentation), not from any
//! reference code. `zeit_core::starfm` is FORCE's variant of STARFM for the Sentinel-2 20 -> 10 m merge and is not
//! this one.
//!
//! Layout: [`plan`] chooses the pairs and the dates, [`starfm`] and [`estarfm`] predict, [`reference`] holds the
//! naive per-pixel formulas the tests compare with.
//!
//! The fine images are given over a data extent (`nx * ny`, the output region plus its halo); the coarse series
//! stays on its own grid and a [`CoarseGeom`] says where each fine pixel falls on it. Everything is float32 in and
//! out, accumulated in f64, NaN = missing, and the order of every sum is fixed so that the result does not depend on
//! the tiling or the threads.

#![allow(clippy::needless_range_loop)] // the loops index several parallel planes by band

pub mod estarfm;
pub mod plan;
pub mod reference;
pub mod starfm;
#[cfg(test)]
mod tests;

pub use plan::{plan, DateRule, Method, PairMatch, PairsMode, Plan, Target};

use crate::parallel::{pool, threads};
use rayon::prelude::*;

/// Fine images: `(pair, band, pixel)` over the data extent (`nx * ny`, row-major), NaN = missing.
pub struct Fine<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nb: usize,
    pub values: &'a [f32],
}

impl Fine<'_> {
    pub fn npix(&self) -> usize {
        self.nx * self.ny
    }
    pub fn npairs(&self) -> usize {
        self.values.len() / (self.nb * self.npix()).max(1)
    }
    /// The image of `pair` and `band`.
    pub fn plane(&self, pair: usize, band: usize) -> &[f32] {
        let n = self.npix();
        let o = (pair * self.nb + band) * n;
        &self.values[o..o + n]
    }
}

/// Where the fine pixels fall on the coarse grid.
pub enum CoarseGeom<'a> {
    /// Fine pixel centre `(x + 0.5, y + 0.5)` of the data extent -> coarse pixel coordinates:
    /// `u = a0 + a1 (x + 0.5) + a2 (y + 0.5)`, `v = a3 + a4 (x + 0.5) + a5 (y + 0.5)`.
    Affine([f64; 6]),
    /// `(u, v)` per fine pixel of the data extent, row-major (another CRS, computed by the caller).
    Lookup { u: &'a [f32], v: &'a [f32] },
}

/// Coarse images on their own grid: `(date, band, cell)`, NaN = missing.
pub struct Coarse<'a> {
    pub ncx: usize,
    pub ncy: usize,
    pub nb: usize,
    pub values: &'a [f32],
    pub geom: CoarseGeom<'a>,
}

impl Coarse<'_> {
    pub fn ncell(&self) -> usize {
        self.ncx * self.ncy
    }
    pub fn ndates(&self) -> usize {
        self.values.len() / (self.nb * self.ncell()).max(1)
    }
    pub fn plane(&self, date: usize, band: usize) -> &[f32] {
        let n = self.ncell();
        let o = (date * self.nb + band) * n;
        &self.values[o..o + n]
    }
    /// The continuous coarse coordinates of the centre of fine pixel `(x, y)` of the data extent.
    #[inline]
    pub fn uv(&self, x: usize, y: usize, nx: usize) -> (f64, f64) {
        match &self.geom {
            CoarseGeom::Affine(a) => {
                let (fx, fy) = (x as f64 + 0.5, y as f64 + 0.5);
                (a[0] + a[1] * fx + a[2] * fy, a[3] + a[4] * fx + a[5] * fy)
            }
            CoarseGeom::Lookup { u, v } => (f64::from(u[y * nx + x]), f64::from(v[y * nx + x])),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resampling {
    Nearest,
    Bilinear,
}

/// Where the standard deviation of the similarity threshold is taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigmaScope {
    Image,
    Window,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fallback {
    Missing,
    Coarse,
}

/// The output region inside the data extent.
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub x0: usize,
    pub y0: usize,
    pub nx: usize,
    pub ny: usize,
}

/// What both methods share.
#[derive(Clone, Copy, Debug)]
pub struct Common {
    /// Half window in fine pixels.
    pub radius_px: usize,
    pub sigma: SigmaScope,
    pub resampling: Resampling,
    pub fallback: Fallback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    Direct,
    Log,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Both,
    Either,
    None,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pure {
    Epsilon,
    Tiers,
    Cells,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VScope {
    Window,
    Cell,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinglePair {
    Fallback,
    Missing,
}

#[derive(Clone, Copy, Debug)]
pub struct StarfmParams {
    pub common: Common,
    /// `m` of the similarity threshold `2 sigma / m`.
    pub classes: f64,
    pub fine_unc: f64,
    pub coarse_unc: f64,
    /// `A` of the spatial distance `1 + d / A`, in fine pixels.
    pub spatial_scale_px: f64,
    pub weighting: Weighting,
    pub filter: Filter,
    pub pure: Pure,
    pub zero_tol: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct EstarfmParams {
    pub common: Common,
    pub classes: f64,
    pub v_scope: VScope,
    pub alpha: f64,
    pub single_pair: SinglePair,
}

/// `eps` of the direct weights: 1 DN at scale 10000.
pub const EPS: f64 = 1e-4;
/// The scale that turns reflectance into DN in the logarithmic weights.
pub const LOG_B: f64 = 10000.0;

/// `source` codes of the quality output.
pub const SRC_OBSERVED: u16 = 0;
pub const SRC_TWO_PAIRS: u16 = 1;
pub const SRC_ONE_PAIR: u16 = 2;
pub const SRC_COARSE: u16 = 3;
pub const SRC_V_ONE: u16 = 4;
pub const SRC_MISSING: u16 = 255;

impl StarfmParams {
    pub fn new(radius_px: usize) -> Self {
        StarfmParams {
            common: Common {
                radius_px,
                sigma: SigmaScope::Image,
                resampling: Resampling::Bilinear,
                fallback: Fallback::Missing,
            },
            classes: 40.0,
            fine_unc: 0.002,
            coarse_unc: 0.005,
            spatial_scale_px: radius_px as f64,
            weighting: Weighting::Direct,
            filter: Filter::Both,
            pure: Pure::Epsilon,
            zero_tol: 0.5e-4,
        }
    }
}

impl EstarfmParams {
    pub fn new(radius_px: usize) -> Self {
        EstarfmParams {
            common: Common {
                radius_px,
                sigma: SigmaScope::Image,
                resampling: Resampling::Bilinear,
                fallback: Fallback::Missing,
            },
            classes: 4.0,
            v_scope: VScope::Window,
            alpha: 0.05,
            single_pair: SinglePair::Fallback,
        }
    }
}

/// Pixels of halo a tile needs around the output region: the window, plus for ESTARFM's per-cell coefficients one
/// coarse cell and a pixel.
pub fn required_halo(method: Method, radius_px: usize, cell_px: f64, v_scope: VScope) -> usize {
    match (method, v_scope) {
        (Method::Estarfm, VScope::Cell) => radius_px + cell_px.ceil() as usize + 1,
        _ => radius_px,
    }
}

/// Population standard deviation per `(pair, band)` over the valid pixels of `region`; NaN when none.
pub fn image_sigma(fine: &Fine, region: &Region, n_jobs: i32) -> Vec<f64> {
    let (np, nb) = (fine.npairs(), fine.nb);
    let one = |i: usize| -> f64 {
        let (k, b) = (i / nb, i % nb);
        let plane = fine.plane(k, b);
        let (mut n, mut mean, mut m2) = (0.0f64, 0.0f64, 0.0f64);
        for y in region.y0..region.y0 + region.ny {
            for &v in &plane[y * fine.nx + region.x0..y * fine.nx + region.x0 + region.nx] {
                if v.is_finite() {
                    n += 1.0;
                    let d = f64::from(v) - mean;
                    mean += d / n;
                    m2 += d * (f64::from(v) - mean);
                }
            }
        }
        if n > 0.0 {
            (m2 / n).sqrt()
        } else {
            f64::NAN
        }
    };
    let t = threads(n_jobs).min((np * nb).max(1));
    if t <= 1 {
        (0..np * nb).map(one).collect()
    } else {
        pool(t).install(|| (0..np * nb).into_par_iter().map(one).collect())
    }
}

/// The coarse cell containing each fine pixel centre of the data extent, `-1` outside the coarse grid.
pub fn cell_map(coarse: &Coarse, nx: usize, ny: usize) -> Vec<i32> {
    let mut out = vec![-1i32; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let (u, v) = coarse.uv(x, y, nx);
            let (i, j) = (u.floor(), v.floor());
            if i >= 0.0 && j >= 0.0 && i < coarse.ncx as f64 && j < coarse.ncy as f64 {
                out[y * nx + x] = (j as usize * coarse.ncx + i as usize) as i32;
            }
        }
    }
    out
}

/// One coarse band on the fine pixels of the data extent. Bilinear: the four cells around `(u - 0.5, v - 0.5)`,
/// missing cells dropped and the weights renormalised, NaN when the valid weight is below 0.5.
pub fn resample_band(coarse: &Coarse, date: usize, band: usize, nx: usize, ny: usize, how: Resampling) -> Vec<f32> {
    let plane = coarse.plane(date, band);
    let (ncx, ncy) = (coarse.ncx as i64, coarse.ncy as i64);
    let mut out = vec![f32::NAN; nx * ny];
    for y in 0..ny {
        for x in 0..nx {
            let (u, v) = coarse.uv(x, y, nx);
            out[y * nx + x] = match how {
                Resampling::Nearest => {
                    let (i, j) = (u.floor() as i64, v.floor() as i64);
                    if i >= 0 && j >= 0 && i < ncx && j < ncy {
                        plane[(j * ncx + i) as usize]
                    } else {
                        f32::NAN
                    }
                }
                Resampling::Bilinear => bilinear(plane, ncx, ncy, u, v),
            };
        }
    }
    out
}

/// The bilinear value at continuous coarse coordinates `(u, v)`. A point outside the coarse grid is NaN; the cells
/// beyond the edge of a point inside it are the edge cells (replicated), so that a fine pixel in the outer half cell
/// keeps its value. Missing cells are dropped and the weights renormalised, NaN when the valid weight is below 0.5.
pub fn bilinear(plane: &[f32], ncx: i64, ncy: i64, u: f64, v: f64) -> f32 {
    if !(u >= 0.0 && v >= 0.0 && u < ncx as f64 && v < ncy as f64) {
        return f32::NAN;
    }
    let (a, b) = (u - 0.5, v - 0.5);
    let (i0, j0) = (a.floor(), b.floor());
    let (fx, fy) = (a - i0, b - j0);
    let (i0, j0) = (i0 as i64, j0 as i64);
    let (mut sw, mut sv) = (0.0f64, 0.0f64);
    for (dj, wy) in [(0i64, 1.0 - fy), (1, fy)] {
        for (di, wx) in [(0i64, 1.0 - fx), (1, fx)] {
            let w = wx * wy;
            if w > 0.0 {
                let (i, j) = ((i0 + di).clamp(0, ncx - 1), (j0 + dj).clamp(0, ncy - 1));
                let c = plane[(j * ncx + i) as usize];
                if c.is_finite() {
                    sw += w;
                    sv += w * f64::from(c);
                }
            }
        }
    }
    if sw >= 0.5 {
        (sv / sw) as f32
    } else {
        f32::NAN
    }
}

/// All bands of a coarse date on the fine pixels: `(band, pixel)`.
pub fn resample_date(coarse: &Coarse, date: usize, nx: usize, ny: usize, how: Resampling) -> Vec<f32> {
    let mut out = Vec::with_capacity(coarse.nb * nx * ny);
    for b in 0..coarse.nb {
        out.extend(resample_band(coarse, date, b, nx, ny, how));
    }
    out
}

/// `f(row)` for `row` in `0..ny` on the threads of `n_jobs`, in row order.
pub(crate) fn rows<T: Send, F: Fn(usize) -> T + Sync + Send>(ny: usize, n_jobs: i32, f: F) -> Vec<T> {
    let t = threads(n_jobs).min(ny.max(1));
    if t <= 1 {
        (0..ny).map(f).collect()
    } else {
        pool(t).install(|| (0..ny).into_par_iter().with_min_len(1).map(&f).collect())
    }
}

/// The window of pixel `(x, y)` clipped to the image: `(x0, x1, y0, y1)`, ends exclusive.
#[inline]
pub(crate) fn window(x: usize, y: usize, r: usize, nx: usize, ny: usize) -> (usize, usize, usize, usize) {
    (
        x.saturating_sub(r),
        (x + r + 1).min(nx),
        y.saturating_sub(r),
        (y + r + 1).min(ny),
    )
}

pub(crate) fn check_inputs(fine: &Fine, coarse: &Coarse, plan: &Plan, region: &Region) -> Result<(), String> {
    if fine.nb != coarse.nb {
        return Err(format!("fine has {} bands and coarse {}", fine.nb, coarse.nb));
    }
    if fine.values.len() != fine.npairs() * fine.nb * fine.npix() {
        return Err("fine values do not fill (pair, band, y, x)".into());
    }
    if coarse.values.len() != coarse.ndates() * coarse.nb * coarse.ncell() {
        return Err("coarse values do not fill (date, band, cell)".into());
    }
    if let CoarseGeom::Lookup { u, v } = &coarse.geom {
        if u.len() != fine.npix() || v.len() != fine.npix() {
            return Err("coarse lookup must have one (u, v) per fine pixel".into());
        }
    }
    if region.x0 + region.nx > fine.nx || region.y0 + region.ny > fine.ny {
        return Err("region outside the data extent".into());
    }
    for t in &plan.targets {
        if t.coarse >= coarse.ndates() || t.pairs.iter().any(|&(f, c)| f >= fine.npairs() || c >= coarse.ndates()) {
            return Err("plan names an image that is not there".into());
        }
        if t.observed.is_some_and(|f| f >= fine.npairs()) {
            return Err("plan names an observed image that is not there".into());
        }
    }
    Ok(())
}
