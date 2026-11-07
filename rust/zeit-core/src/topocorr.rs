//! Topographic correction of reflectance: slope and aspect of a DEM, the illumination angle of every pixel, and
//! the C-correction and SCS+C that flatten the reflectance the relief modulates.
//!
//! Written from the published formulas (they are textbook): Horn (1981, Proc. IEEE 69:14-47) for slope and
//! aspect; the cosine of the illumination angle `cos i = cos sz cos s + sin sz sin s cos(saz - a)`; the C
//! correction of Teillet, Guindon & Goodenough (1982, Can. J. Remote Sens. 8:84-106), `rho_H = rho (cos sz + c)
//! / (cos i + c)` with `c = b / m` from the regression `rho = b + m cos i`; and the SCS+C of Soenen, Peddle &
//! Coburn (2005, IEEE TGRS 43:2148-2159), `rho_H = rho (cos s cos sz + c) / (cos i + c)`.
//!
//! Guards (a pixel that cannot be corrected keeps its value, a band that cannot be fitted is copied):
//! - a DEM cell with a non-finite value among its eight neighbours (Horn's kernel does not use the centre
//!   cell) has no slope (NaN), so its pixels are left as they are; the cells on the DEM's edge replicate the
//!   nearest cell for the missing neighbours and take the difference over the single cell it spans (a one-sided
//!   Horn gradient), so they have a slope as accurate as a plane's;
//! - a flat cell has aspect 0 (the term it multiplies is zero);
//! - a band of a date with fewer than `min_sample` sample pixels, a `cos i` with no variance, or a slope of the
//!   regression `m <= 0` (or a non-finite `c`) is not corrected;
//! - a pixel with `cos i + c < DEN_MIN`, a non-positive numerator, a missing sun angle (a zenith outside
//!   0-90 degrees or NaN) or a NaN reflectance is not corrected.

use crate::parallel::{pool, threads, Shared};
use rayon::prelude::*;

/// Pixels per job.
const TILE: usize = 16384;

/// `cos i + c` below this leaves the pixel uncorrected: the ratio of two small numbers is noise.
pub const DEN_MIN: f64 = 0.01;

/// The correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Teillet et al. (1982): `rho (cos sz + c) / (cos i + c)`.
    C,
    /// Soenen et al. (2005): `rho (cos s cos sz + c) / (cos i + c)`.
    ScsC,
}

/// The regression of a band of a date and what the correction did to it.
#[derive(Clone, Copy, Debug)]
pub struct Fit {
    /// `b / m`; NaN if the band was not fitted (and so copied).
    pub c: f64,
    /// Slope of `rho = b + m cos i` (also when it is <= 0 and the band is not corrected); NaN if no regression
    /// could be made.
    pub slope: f64,
    /// Intercept `b`; NaN if no regression could be made.
    pub intercept: f64,
    /// Sample pixels of the regression.
    pub n: usize,
    /// Pixels with a finite reflectance left as they were by the guards (not counting a band not fitted).
    pub skipped: usize,
}

fn run_jobs<R: Send>(t: usize, jobs: usize, f: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    if t <= 1 {
        (0..jobs).map(f).collect()
    } else {
        pool(t).install(|| (0..jobs).into_par_iter().map(f).collect())
    }
}

/// Slope and aspect, in radians, of the DEM `z` (`ny` rows of `nx`, C order) by Horn's 3x3 kernel. `dx` is the
/// change of the easting from a column to the next and `dy` the change of the northing from a row to the next
/// (negative for a north-up raster), both in the unit of `z`. The slope is the angle of the surface from the
/// horizontal, `atan(|grad z|)`; the aspect is the direction the surface faces (downslope), clockwise from
/// north, in `[0, 2 pi)`, 0 where the surface is flat. See the module's notes for NaN and the edges.
pub fn slope_aspect(z: &[f64], ny: usize, nx: usize, dx: f64, dy: f64, n_jobs: i32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(z.len(), ny * nx);
    assert!(
        dx != 0.0 && dy != 0.0 && dx.is_finite() && dy.is_finite(),
        "pixel size must be non-zero"
    );
    let mut slope = vec![f32::NAN; ny * nx];
    let mut aspect = vec![f32::NAN; ny * nx];
    if ny * nx == 0 {
        return (slope, aspect);
    }
    let asp = Shared::new(&mut aspect);
    let row = |r: usize, s: &mut [f32]| {
        // SAFETY: each row is written by one job.
        let a = unsafe { asp.slice_mut(r * nx..(r + 1) * nx) };
        let (ra, rb) = (r.saturating_sub(1), (r + 1).min(ny - 1));
        let up = &z[ra * nx..][..nx];
        let mid = &z[r * nx..][..nx];
        let dn = &z[rb * nx..][..nx];
        for c in 0..nx {
            let l = c.saturating_sub(1);
            let rr = (c + 1).min(nx - 1);
            let (za, zb, zc) = (up[l], up[c], up[rr]);
            let (zd, zf) = (mid[l], mid[rr]);
            let (zg, zh, zi) = (dn[l], dn[c], dn[rr]);
            // `rr - l` columns (`rb - ra` rows) apart: 2 inside, 1 on the edge, where the replicated neighbour
            // makes the difference a one-sided one over a single cell (Horn's /8 becomes /4 per cell of span)
            let (sx, sy) = ((rr - l) as f64, (rb - ra) as f64);
            let dzdx = ((zc + 2.0 * zf + zi) - (za + 2.0 * zd + zg)) / (4.0 * sx * dx);
            let dzdn = ((zg + 2.0 * zh + zi) - (za + 2.0 * zb + zc)) / (4.0 * sy * dy);
            if !(dzdx.is_finite() && dzdn.is_finite()) {
                s[c] = f32::NAN;
                a[c] = f32::NAN;
                continue;
            }
            s[c] = dzdx.hypot(dzdn).atan() as f32;
            a[c] = if dzdx == 0.0 && dzdn == 0.0 {
                0.0
            } else {
                let v = (-dzdx).atan2(-dzdn);
                (if v < 0.0 { v + std::f64::consts::TAU } else { v }) as f32
            };
        }
    };
    let t = threads(n_jobs).min(ny);
    if t <= 1 {
        for (r, s) in slope.chunks_mut(nx).enumerate() {
            row(r, s);
        }
    } else {
        pool(t).install(|| slope.par_chunks_mut(nx).enumerate().for_each(|(r, s)| row(r, s)));
    }
    (slope, aspect)
}

/// What the threads share.
struct Plan<'a> {
    x: &'a [f32],
    nb: usize,
    npix: usize,
    /// `cos s`, `sin s`, `cos a`, `sin a` of every pixel.
    cs: Vec<f32>,
    ss: Vec<f32>,
    ca: Vec<f32>,
    sa: Vec<f32>,
    /// Sun: `(time, ns)` zenith and azimuth in radians, `ns` 1 or `npix`.
    zen: Vec<f32>,
    azi: Vec<f32>,
    ns: usize,
    sample: Option<(&'a [u8], usize)>,
    min_slope: f32,
    method: Method,
}

impl Plan<'_> {
    /// `cos i` and `cos sz` of date `k`, pixels `p0..p1` (NaN where the geometry is missing).
    fn illumination(&self, k: usize, p0: usize, p1: usize, cosi: &mut [f32], cosz: &mut [f32]) {
        for (i, p) in (p0..p1).enumerate() {
            let q = if self.ns == 1 { 0 } else { p };
            let (z, a) = (self.zen[k * self.ns + q], self.azi[k * self.ns + q]);
            let (cz, sz) = (z.cos(), z.sin());
            let valid = (0.0..std::f32::consts::FRAC_PI_2).contains(&z) && a.is_finite();
            let cosaz = a.cos() * self.ca[p] + a.sin() * self.sa[p];
            cosi[i] = if valid {
                cz * self.cs[p] + sz * self.ss[p] * cosaz
            } else {
                f32::NAN
            };
            cosz[i] = if valid { cz } else { f32::NAN };
        }
    }

    fn in_sample(&self, k: usize, p: usize) -> bool {
        match self.sample {
            Some((m, nts)) => m[(if nts == 1 { 0 } else { k }) * self.npix + p] != 0,
            None => self.ss[p] >= self.min_slope.sin(),
        }
    }
}

/// `n`, `sum x`, `sum y`, `sum x^2`, `sum xy` of a tile, in float64.
#[derive(Clone, Copy, Default)]
struct Sums([f64; 5]);

/// Corrects the reflectance `x` (`nt` dates, `nb` bands, `npix` pixels, C order) into `out`, the same shape:
/// `slope` and `aspect` (radians, `npix` values, from `slope_aspect`); the sun's `zenith` and `azimuth` in
/// degrees (azimuth clockwise from north), `(nt, ns)` with `ns` 1 (one sun for the scene of a date) or `npix`
/// (per pixel); `sample`, if given, is `(mask, nts)` with `nts` 1 or `nt` rows of `npix` bytes (nonzero: the pixel
/// takes part in the regression), else the pixels with `slope >= min_slope` (radians) do. A pixel also needs a
/// finite reflectance and `cos i`. Returns the regression of every (date, band), in that order.
#[allow(clippy::too_many_arguments)]
pub fn topo_correct(
    x: &[f32],
    nt: usize,
    nb: usize,
    npix: usize,
    slope: &[f32],
    aspect: &[f32],
    zenith: &[f32],
    azimuth: &[f32],
    ns: usize,
    sample: Option<(&[u8], usize)>,
    method: Method,
    min_slope: f32,
    min_sample: usize,
    out: &mut [f32],
    n_jobs: i32,
) -> Vec<Fit> {
    assert_eq!(x.len(), nt * nb * npix);
    assert_eq!(out.len(), x.len());
    assert!(slope.len() == npix && aspect.len() == npix);
    assert!(ns == 1 || ns == npix);
    assert!(zenith.len() == nt * ns && azimuth.len() == nt * ns);
    if let Some((m, nts)) = sample {
        assert!(nts == 1 || nts == nt);
        assert_eq!(m.len(), nts * npix);
    }
    let nan_fit = Fit {
        c: f64::NAN,
        slope: f64::NAN,
        intercept: f64::NAN,
        n: 0,
        skipped: 0,
    };
    if nt * nb * npix == 0 {
        return vec![nan_fit; nt * nb];
    }
    let t = threads(n_jobs);
    let map = |v: &[f32], f: fn(f32) -> f32| -> Vec<f32> {
        if t <= 1 {
            v.iter().map(|&u| f(u)).collect()
        } else {
            pool(t).install(|| v.par_iter().map(|&u| f(u)).collect())
        }
    };
    let plan = Plan {
        x,
        nb,
        npix,
        cs: map(slope, f32::cos),
        ss: map(slope, f32::sin),
        ca: map(aspect, f32::cos),
        sa: map(aspect, f32::sin),
        zen: zenith.iter().map(|d| d.to_radians()).collect(),
        azi: azimuth.iter().map(|d| d.to_radians()).collect(),
        ns,
        sample,
        min_slope,
        method,
    };
    let per = npix.div_ceil(TILE);

    // 1. cos i and cos sz of every date and pixel
    let mut cosi = vec![0f32; nt * npix];
    let mut cosz = vec![0f32; nt * npix];
    {
        let (ci, cz) = (Shared::new(&mut cosi), Shared::new(&mut cosz));
        run_jobs(t.min(nt * per), nt * per, |j| {
            let (k, tl) = (j / per, j % per);
            let (p0, p1) = (tl * TILE, ((tl + 1) * TILE).min(npix));
            // SAFETY: each job writes its own date and pixels.
            let (a, b) = unsafe {
                (
                    ci.slice_mut(k * npix + p0..k * npix + p1),
                    cz.slice_mut(k * npix + p0..k * npix + p1),
                )
            };
            plan.illumination(k, p0, p1, a, b);
        });
    }

    // 2. the sums of the regression of every (date, band, tile)
    let jobs = nt * nb * per;
    let sums = run_jobs(t.min(jobs), jobs, |j| {
        let (kb, tl) = (j / per, j % per);
        let k = kb / plan.nb;
        let (p0, p1) = (tl * TILE, ((tl + 1) * TILE).min(npix));
        let rho = &plan.x[kb * npix + p0..kb * npix + p1];
        let ci = &cosi[k * npix + p0..k * npix + p1];
        let mut s = [0f64; 5];
        for (i, (&y, &c)) in rho.iter().zip(ci).enumerate() {
            if y.is_finite() && c.is_finite() && plan.in_sample(k, p0 + i) {
                let (c, y) = (f64::from(c), f64::from(y));
                s[0] += 1.0;
                s[1] += c;
                s[2] += y;
                s[3] += c * c;
                s[4] += c * y;
            }
        }
        Sums(s)
    });

    // the regression of every (date, band)
    let mut fits = vec![nan_fit; nt * nb];
    for (kb, fit) in fits.iter_mut().enumerate() {
        let mut s = [0f64; 5];
        for tl in 0..per {
            for (a, b) in s.iter_mut().zip(sums[kb * per + tl].0) {
                *a += b;
            }
        }
        let n = s[0];
        fit.n = n as usize;
        if fit.n < min_sample.max(3) {
            continue;
        }
        let (mx, my) = (s[1] / n, s[2] / n);
        let sxx = s[3] - n * mx * mx;
        let sxy = s[4] - n * mx * my;
        if !(sxx > 1e-12 * n) {
            continue;
        }
        let m = sxy / sxx;
        let b = my - m * mx;
        let c = b / m;
        fit.slope = m;
        fit.intercept = b;
        if m > 0.0 && c.is_finite() {
            fit.c = c;
        }
    }

    // 3. the correction
    let skipped = {
        let o = Shared::new(out);
        run_jobs(t.min(jobs), jobs, |j| {
            let (kb, tl) = (j / per, j % per);
            let k = kb / plan.nb;
            let (p0, p1) = (tl * TILE, ((tl + 1) * TILE).min(npix));
            let src = &plan.x[kb * npix + p0..kb * npix + p1];
            // SAFETY: each job writes its own band of its date and pixels.
            let dst = unsafe { o.slice_mut(kb * npix + p0..kb * npix + p1) };
            dst.copy_from_slice(src);
            let c = fits[kb].c;
            if !c.is_finite() {
                return 0usize;
            }
            let ci = &cosi[k * npix + p0..k * npix + p1];
            let cz = &cosz[k * npix + p0..k * npix + p1];
            let mut skip = 0;
            for i in 0..src.len() {
                let y = f64::from(src[i]);
                if !y.is_finite() {
                    continue;
                }
                let den = f64::from(ci[i]) + c;
                let num = match plan.method {
                    Method::C => f64::from(cz[i]) + c,
                    Method::ScsC => f64::from(plan.cs[p0 + i]) * f64::from(cz[i]) + c,
                };
                // a NaN anywhere fails these comparisons
                if den >= DEN_MIN && num > 0.0 {
                    dst[i] = (y * num / den) as f32;
                } else {
                    skip += 1;
                }
            }
            skip
        })
    };
    for (fit, s) in fits.iter_mut().zip(skipped) {
        fit.skipped = s;
    }
    fits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_slope_and_aspect() {
        // z rises 0.5 m per metre to the east: slope atan(0.5), faces west (270 degrees)
        let (ny, nx, d) = (6, 7, 30.0);
        let z: Vec<f64> = (0..ny * nx).map(|i| 0.5 * d * (i % nx) as f64).collect();
        let (s, a) = slope_aspect(&z, ny, nx, d, -d, 1);
        for i in 0..ny * nx {
            // the edge columns too: the gradient there is one-sided, over one cell
            assert!((f64::from(s[i]) - 0.5f64.atan()).abs() < 1e-6);
            assert!((f64::from(a[i]).to_degrees() - 270.0).abs() < 1e-4);
        }
        // rises to the north (the first row is the north): faces south (180)
        let z: Vec<f64> = (0..ny * nx).map(|i| 0.25 * d * (ny - 1 - i / nx) as f64).collect();
        let (s, a) = slope_aspect(&z, ny, nx, d, -d, 2);
        let i = 2 * nx + 3;
        assert!((f64::from(s[i]) - 0.25f64.atan()).abs() < 1e-6);
        assert!((f64::from(a[i]).to_degrees() - 180.0).abs() < 1e-4);
        // a flat cell: aspect 0
        let (s, a) = slope_aspect(&vec![5.0; ny * nx], ny, nx, d, -d, 1);
        assert!(s.iter().all(|&v| v == 0.0) && a.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn nan_dem_cell_poisons_its_window() {
        let (ny, nx) = (5, 5);
        let mut z = vec![1.0; ny * nx];
        z[2 * nx + 2] = f64::NAN;
        let (s, _) = slope_aspect(&z, ny, nx, 10.0, -10.0, 1);
        for r in 0..ny {
            for c in 0..nx {
                // Horn's kernel does not use the centre cell: its own slope is defined
                let near = (r as i32 - 2).abs() <= 1 && (c as i32 - 2).abs() <= 1 && (r, c) != (2, 2);
                assert_eq!(s[r * nx + c].is_nan(), near, "{r},{c}");
            }
        }
    }

    /// A sloped scene: facing every direction, rho = rho0 cos i / cos sz.
    fn scene(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, f64, f64) {
        let mut slope = vec![0f32; n];
        let mut aspect = vec![0f32; n];
        for p in 0..n {
            slope[p] = (8.0 + 25.0 * ((p * 7919) % 1000) as f32 / 1000.0).to_radians();
            aspect[p] = (((p * 104729) % 3600) as f32 / 10.0).to_radians();
        }
        (vec![0f32; n], slope, aspect, 40f64, 150f64)
    }

    #[test]
    fn lambertian_scene_with_a_background_is_flattened() {
        // rho = b + kk cos i, the C-correction's own model, with c = b / kk = 0.5: every pixel goes to
        // kk (cos sz + c)
        let n = 40000;
        let (_, slope, aspect, sz, saz) = scene(n);
        let (zen, azi) = (vec![sz as f32], vec![saz as f32]);
        let kk = 0.3f64;
        let c_true = 0.5;
        let b = kk * c_true;
        let mut cosi = vec![0f32; n];
        let mut cosz = vec![0f32; n];
        let plan_check = |out: &mut Vec<f32>, x: &Vec<f32>, method: Method| {
            let fits = topo_correct(
                x, 1, 1, n, &slope, &aspect, &zen, &azi, 1, None, method, 0.0, 10, out, 2,
            );
            fits[0]
        };
        // cos i by hand
        for p in 0..n {
            let (s, a) = (f64::from(slope[p]), f64::from(aspect[p]));
            cosz[p] = sz.to_radians().cos() as f32;
            cosi[p] = (sz.to_radians().cos() * s.cos() + sz.to_radians().sin() * s.sin() * (saz.to_radians() - a).cos())
                as f32;
        }
        let x: Vec<f32> = cosi.iter().map(|&c| (b + kk * f64::from(c)) as f32).collect();
        let mut out = vec![0f32; n];
        let fit = plan_check(&mut out, &x, Method::C);
        assert!((fit.c - c_true).abs() < 1e-3, "{}", fit.c);
        assert!((fit.slope - kk).abs() < 1e-4);
        // every pixel goes to b + kk cos sz
        let want = b + kk * sz.to_radians().cos();
        for &v in &out {
            assert!((f64::from(v) - want).abs() < 2e-4, "{v} {want}");
        }
        // the SCS+C: cos s varies, the result is rho (cos s cos sz + c) / (cos i + c)
        let fit = plan_check(&mut out, &x, Method::ScsC);
        assert!((fit.c - c_true).abs() < 1e-3);
        for p in (0..n).step_by(97) {
            let want = f64::from(x[p]) * (f64::from(slope[p]).cos() * sz.to_radians().cos() + fit.c)
                / (f64::from(cosi[p]) + fit.c);
            assert!((f64::from(out[p]) - want).abs() < 1e-5);
        }
    }

    #[test]
    fn guards_leave_things_alone() {
        let n = 2000;
        let (_, slope, aspect, sz, saz) = scene(n);
        let (zen, azi) = (vec![sz as f32], vec![saz as f32]);
        // anti-correlated reflectance: m < 0, not corrected
        let mut cosi = vec![0f64; n];
        for p in 0..n {
            let (s, a) = (f64::from(slope[p]), f64::from(aspect[p]));
            cosi[p] = sz.to_radians().cos() * s.cos() + sz.to_radians().sin() * s.sin() * (saz.to_radians() - a).cos();
        }
        let x: Vec<f32> = cosi.iter().map(|&c| (0.5 - 0.2 * c) as f32).collect();
        let mut out = vec![0f32; n];
        let fits = topo_correct(
            &x,
            1,
            1,
            n,
            &slope,
            &aspect,
            &zen,
            &azi,
            1,
            None,
            Method::C,
            0.0,
            10,
            &mut out,
            1,
        );
        assert!(fits[0].c.is_nan() && fits[0].slope < 0.0);
        assert_eq!(out, x);
        // too few sample pixels
        let y: Vec<f32> = cosi.iter().map(|&c| (0.1 + 0.2 * c) as f32).collect();
        let fits = topo_correct(
            &y,
            1,
            1,
            n,
            &slope,
            &aspect,
            &zen,
            &azi,
            1,
            None,
            Method::C,
            0.0,
            n + 1,
            &mut out,
            1,
        );
        assert!(fits[0].c.is_nan() && out == y);
        // a mask that keeps 5 pixels
        let mut mask = vec![0u8; n];
        mask[..5].fill(1);
        let fits = topo_correct(
            &y,
            1,
            1,
            n,
            &slope,
            &aspect,
            &zen,
            &azi,
            1,
            Some((&mask, 1)),
            Method::C,
            0.0,
            10,
            &mut out,
            1,
        );
        assert!(fits[0].c.is_nan() && fits[0].n == 5);
        // NaN reflectance stays NaN, doesn't enter the fit; NaN slope: pixel unchanged
        let mut y2 = y.clone();
        y2[3] = f32::NAN;
        let mut sl = slope.clone();
        sl[7] = f32::NAN;
        let fits = topo_correct(
            &y2,
            1,
            1,
            n,
            &sl,
            &aspect,
            &zen,
            &azi,
            1,
            None,
            Method::C,
            0.0,
            10,
            &mut out,
            1,
        );
        assert!(out[3].is_nan() && out[7] == y2[7] && fits[0].n == n - 2);
        assert!((fits[0].c - 0.1 / 0.2 * 1.0).abs() < 1e-3);
        // a sun below the horizon at pixel 0: unchanged
        let mut zen_px = vec![sz as f32; n];
        zen_px[0] = 95.0;
        let azi_px = vec![saz as f32; n];
        let fits = topo_correct(
            &y,
            1,
            1,
            n,
            &slope,
            &aspect,
            &zen_px,
            &azi_px,
            n,
            None,
            Method::C,
            0.0,
            10,
            &mut out,
            1,
        );
        assert!(out[0] == y[0] && fits[0].n == n - 1);
    }

    #[test]
    fn threads_and_tiles_agree() {
        let n = TILE * 2 + 5;
        let (nt, nb) = (2, 3);
        let (_, slope, aspect, _, _) = scene(n);
        let zen: Vec<f32> = vec![35.0, 50.0];
        let azi: Vec<f32> = vec![120.0, 160.0];
        let x: Vec<f32> = (0..nt * nb * n)
            .map(|i| 0.05 + ((i * 7919) % 1000) as f32 / 3000.0)
            .collect();
        let mut a = vec![0f32; x.len()];
        let mut b = vec![0f32; x.len()];
        let args = |out: &mut Vec<f32>, j: i32| {
            topo_correct(
                &x,
                nt,
                nb,
                n,
                &slope,
                &aspect,
                &zen,
                &azi,
                1,
                None,
                Method::ScsC,
                0.05,
                10,
                out,
                j,
            )
        };
        let fa = args(&mut a, 1);
        let fb = args(&mut b, 3);
        assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
        assert!(fa.iter().zip(&fb).all(|(p, q)| p.c.to_bits() == q.c.to_bits()));
    }
}
