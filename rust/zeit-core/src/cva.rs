//! Change vector analysis (CVA) of two dates: the magnitude and the direction of the vector of band differences,
//! and a threshold on the magnitude (fixed, Otsu's, or the minimum-error one of a two-class Gaussian mixture
//! fitted by EM).
//!
//! Written from the papers: Malila (1980) for the vector, Bovolo, Marchesi and Bruzzone (2012, IEEE TGRS
//! 50:2196-2212) for the compressed direction (C2VA), Bovolo and Bruzzone (2007, IEEE TGRS 45:218-236) for the
//! polar form, Otsu (1979) and Bruzzone and Prieto (2000, IEEE TGRS 38:1171-1182) for the thresholds.

use crate::parallel::{pool, threads, Shared};
use crate::sample64::{present, Sample64};
use rayon::prelude::*;

/// Pixels per job.
const CHUNK: usize = 4096;

/// How the direction of the change vector is described. The band numbers are positions in the list of bands the
/// vector is made of.
#[derive(Clone, Debug)]
pub enum Direction {
    /// The angle (degrees, 0 to 180) between the vector and the diagonal of uniform change (Bovolo et al. 2012).
    C2va,
    /// The angle (degrees, 0 to 360) of the vector in the plane of two bands, and `n_sectors` equal sectors from 0
    /// degrees (Bovolo and Bruzzone 2007).
    Polar { b1: usize, b2: usize, n_sectors: usize },
    /// The sectors of the signs of the changes of red, NIR and SWIR1: `1 + [d_red > 0] + 2 [d_nir > 0] + 4
    /// [d_swir1 > 0]`, 1 to 8. The direction plane holds the sector.
    Sign { red: usize, nir: usize, swir1: usize },
}

/// The EM fit of the two classes (no change, change) of the magnitudes.
#[derive(Clone, Copy, Debug)]
pub struct EmFit {
    pub p: [f64; 2],
    pub mu: [f64; 2],
    pub sigma: [f64; 2],
    pub iterations: usize,
    pub converged: bool,
}

/// The percentile `q` (0 to 1) of `values`: the element at `floor(q (N - 1))` of a sorted copy.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).floor() as usize]
}

/// Otsu's threshold of `values` in a histogram of `nbins` equal bins over `[0, upper]` (the larger values in the
/// last bin): the upper edge of the first bin that maximizes the between-class variance, and that variance.
pub fn otsu(values: &[f64], nbins: usize, upper: f64) -> (f64, f64) {
    let w = upper / nbins as f64;
    let mut hist = vec![0.0f64; nbins];
    for &v in values {
        let b = if w > 0.0 { ((v / w) as usize).min(nbins - 1) } else { 0 };
        hist[b] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let mean_all: f64 = hist
        .iter()
        .enumerate()
        .map(|(i, h)| h / total * (i as f64 + 0.5) * w)
        .sum();
    let (mut w0, mut m0) = (0.0, 0.0);
    let (mut best, mut best_k) = (-1.0, 0);
    for k in 0..nbins - 1 {
        let p = hist[k] / total;
        w0 += p;
        m0 += p * (k as f64 + 0.5) * w;
        let w1 = 1.0 - w0;
        if w0 <= 0.0 || w1 <= 1e-15 {
            continue;
        }
        let (mu0, mu1) = (m0 / w0, (mean_all - m0) / w1);
        let sb = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if sb > best {
            best = sb;
            best_k = k;
        }
    }
    ((best_k + 1) as f64 * w, best.max(0.0))
}

/// The root between the two means of `P_n N(T; mu_n, s_n) = P_c N(T; mu_c, s_c)` (Bayes' minimum-error threshold
/// of two Gaussian classes), or None if there is none.
pub fn bayes_threshold(p: [f64; 2], mu: [f64; 2], sigma: [f64; 2]) -> Option<f64> {
    let (pn, pc) = (p[0], p[1]);
    let (mn, mc) = (mu[0], mu[1]);
    let (vn, vc) = (sigma[0] * sigma[0], sigma[1] * sigma[1]);
    let a = vn - vc;
    let b = 2.0 * (mn * vc - mc * vn);
    let c = vn * mc * mc - vc * mn * mn + 2.0 * vn * vc * (pn * sigma[1] / (pc * sigma[0])).ln();
    let (lo, hi) = (mn.min(mc), mn.max(mc));
    let inside = |t: f64| t > lo && t < hi;
    if a.abs() < 1e-12 * vn.max(vc) {
        // equal variances: linear
        let t = -c / b;
        return if b != 0.0 && inside(t) { Some(t) } else { None };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    // the numerically stable pair of roots
    let q = -0.5 * (b + b.signum().max(0.0).mul_add(2.0, -1.0) * s);
    let (r1, r2) = (q / a, c / q);
    [r1, r2].into_iter().find(|&t| inside(t))
}

fn log_normal(x: f64, mu: f64, var: f64) -> f64 {
    -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + (x - mu) * (x - mu) / var)
}

/// The two-class Gaussian mixture of `values` by EM and the minimum-error threshold between its classes
/// (Bruzzone and Prieto 2000, the version without context), seeded by `{x < M (1 - init)}` and
/// `{x > M (1 + init)}` with `M` half the sum of the minimum and the 99.9th percentile (the Otsu split when a seed
/// has fewer than 10 values). If the mixture has no threshold between its means, Otsu's. Returns the threshold
/// and the fit (`iterations` 0 and `converged` false when the fallback was used for the threshold only).
pub fn em_threshold(values: &[f64], init: f64, max_iter: usize, tol: f64) -> Result<(f64, EmFit), String> {
    let n = values.len();
    if n < 20 {
        return Err("too few valid pixels for the EM threshold".into());
    }
    let nf = n as f64;
    let rmax = percentile(values, 0.999);
    let rmin = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = values.iter().sum::<f64>() / nf;
    let var_all = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / nf;
    if !(var_all > 0.0) {
        return Err("every magnitude is the same: there is nothing to threshold".into());
    }
    let floor = 1e-12 * var_all;
    let md = 0.5 * (rmin + rmax);
    let mut seed_n: Vec<f64> = values.iter().cloned().filter(|&v| v < md * (1.0 - init)).collect();
    let mut seed_c: Vec<f64> = values.iter().cloned().filter(|&v| v > md * (1.0 + init)).collect();
    let (ot, _) = otsu(values, 1024, rmax);
    if seed_n.len() < 10 || seed_c.len() < 10 {
        seed_n = values.iter().cloned().filter(|&v| v < ot).collect();
        seed_c = values.iter().cloned().filter(|&v| v >= ot).collect();
        if seed_n.len() < 2 || seed_c.len() < 2 {
            return Err("the magnitudes have one class only".into());
        }
    }
    let stats = |s: &[f64]| {
        let m = s.iter().sum::<f64>() / s.len() as f64;
        let v = s.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / s.len() as f64;
        (m, v.max(floor))
    };
    let (m_n, v_n) = stats(&seed_n);
    let (m_c, v_c) = stats(&seed_c);
    let tot = (seed_n.len() + seed_c.len()) as f64;
    let mut p = [seed_n.len() as f64 / tot, seed_c.len() as f64 / tot];
    let mut mu = [m_n, m_c];
    let mut var = [v_n, v_c];
    let mut prev = f64::NEG_INFINITY;
    let (mut iterations, mut converged) = (0, false);
    let mut g = [0.0f64; 2];
    for it in 0..max_iter {
        iterations = it + 1;
        let (mut s0, mut s1, mut x0, mut x1, mut ll) = (0.0, 0.0, 0.0, 0.0, 0.0);
        // E-step and the sums of the M-step
        for &x in values {
            let l0 = p[0].ln() + log_normal(x, mu[0], var[0]);
            let l1 = p[1].ln() + log_normal(x, mu[1], var[1]);
            let mx = l0.max(l1);
            let tot = mx + ((l0 - mx).exp() + (l1 - mx).exp()).ln();
            ll += tot;
            g[0] = (l0 - tot).exp();
            g[1] = (l1 - tot).exp();
            s0 += g[0];
            s1 += g[1];
            x0 += g[0] * x;
            x1 += g[1] * x;
        }
        ll /= nf;
        let new_mu = [x0 / s0, x1 / s1];
        // the variances about the new means, in a second pass
        let (mut v0, mut v1) = (0.0, 0.0);
        for &x in values {
            let l0 = p[0].ln() + log_normal(x, mu[0], var[0]);
            let l1 = p[1].ln() + log_normal(x, mu[1], var[1]);
            let mx = l0.max(l1);
            let tot = mx + ((l0 - mx).exp() + (l1 - mx).exp()).ln();
            v0 += (l0 - tot).exp() * (x - new_mu[0]) * (x - new_mu[0]);
            v1 += (l1 - tot).exp() * (x - new_mu[1]) * (x - new_mu[1]);
        }
        p = [s0 / nf, s1 / nf];
        mu = new_mu;
        var = [(v0 / s0).max(floor), (v1 / s1).max(floor)];
        if (ll - prev).abs() < tol {
            converged = true;
            break;
        }
        prev = ll;
    }
    let (mut p, mut mu, mut sigma) = (p, mu, [var[0].sqrt(), var[1].sqrt()]);
    if mu[0] > mu[1] {
        p.swap(0, 1);
        mu.swap(0, 1);
        sigma.swap(0, 1);
    }
    let fit = EmFit {
        p,
        mu,
        sigma,
        iterations,
        converged,
    };
    match bayes_threshold(p, mu, sigma) {
        Some(t) => Ok((t, fit)),
        None => Ok((ot, fit)),
    }
}

/// The magnitudes of the valid pixels of the lattice sample (every `stride`-th row and column from the first) and
/// the scale of each band. `x1` and `x2` are `(band, y, x)`; `bands` are the bands the vector is made of; with
/// `scale_std` the scale of a band is 1 over the standard deviation of its differences over the valid sample
/// pixels (0 when it has none), else 1.
#[allow(clippy::too_many_arguments)]
pub fn cva_sample<T: Sample64>(
    x1: &[T],
    x2: &[T],
    nb: usize,
    width: usize,
    height: usize,
    bands: &[usize],
    scale_std: bool,
    stride: usize,
    missing: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let npix = width * height;
    assert_eq!(x1.len(), nb * npix);
    assert_eq!(x2.len(), nb * npix);
    let nbands = bands.len();
    let mut d: Vec<f64> = Vec::new();
    let stride = stride.max(1);
    for r in (0..height).step_by(stride) {
        for c in (0..width).step_by(stride) {
            let p = r * width + c;
            let mut v = [0.0f64; 64];
            let mut ok = true;
            for (k, &b) in bands.iter().enumerate() {
                let (a, z) = (x1[b * npix + p].to_f64(), x2[b * npix + p].to_f64());
                ok &= present(a, missing) && present(z, missing);
                v[k] = z - a;
            }
            if ok {
                d.extend_from_slice(&v[..nbands]);
            }
        }
    }
    let rows = d.len() / nbands.max(1);
    let mut scale = vec![1.0; nbands];
    if scale_std {
        for k in 0..nbands {
            let m = (0..rows).map(|i| d[i * nbands + k]).sum::<f64>() / rows.max(1) as f64;
            let var = (0..rows).map(|i| (d[i * nbands + k] - m).powi(2)).sum::<f64>() / rows.max(1) as f64;
            scale[k] = if var > 0.0 { 1.0 / var.sqrt() } else { 0.0 };
        }
    }
    let rho = (0..rows)
        .map(|i| {
            (0..nbands)
                .map(|k| (d[i * nbands + k] * scale[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    (rho, scale)
}

/// What `cva_batch` does with each pixel.
pub struct CvaOptions<'a> {
    /// The bands of the vector.
    pub bands: &'a [usize],
    /// The scale of each of those bands.
    pub scale: &'a [f64],
    pub direction: Direction,
    /// A pixel changes when its magnitude exceeds it.
    pub threshold: f64,
    /// For `C2va`: the edges (degrees, ascending, the first 0 and the last 180) of the sectors of the angle.
    pub c2va_edges: Option<&'a [f64]>,
}

/// The magnitude, the direction, the change (1 or 0) and the sector (0 where nothing changed; NaN where the
/// direction has no sectors) of every pixel of two dates (`x1`, `x2`: `(band, pixel)`, `nb` bands, `npix`
/// pixels). A pixel with a band of the vector missing at either date (NaN, infinite or among `missing`) is NaN in
/// every plane. A difference of exactly 0 is not above 0 for the signs. Every value of the planes is written.
#[allow(clippy::too_many_arguments)]
pub fn cva_batch<T: Sample64>(
    x1: &[T],
    x2: &[T],
    nb: usize,
    npix: usize,
    o: &CvaOptions,
    missing: &[f64],
    out_mag: &mut [f32],
    out_dir: &mut [f32],
    out_change: &mut [f32],
    out_sector: &mut [f32],
    n_jobs: i32,
) {
    assert_eq!(x1.len(), nb * npix);
    assert_eq!(x2.len(), nb * npix);
    assert!(out_mag.len() == npix && out_dir.len() == npix && out_change.len() == npix && out_sector.len() == npix);
    let nbands = o.bands.len();
    assert!(nbands <= 64 && o.scale.len() == nbands);
    let (sm, sd, sc, ss) = (
        Shared::new(out_mag),
        Shared::new(out_dir),
        Shared::new(out_change),
        Shared::new(out_sector),
    );
    let job = |c: usize| {
        let p0 = c * CHUNK;
        let p1 = ((c + 1) * CHUNK).min(npix);
        // SAFETY: each job writes its own pixels of the planes.
        let (mag, dir, chg, sec) = unsafe {
            (
                sm.slice_mut(p0..p1),
                sd.slice_mut(p0..p1),
                sc.slice_mut(p0..p1),
                ss.slice_mut(p0..p1),
            )
        };
        let mut d = [0.0f64; 64];
        for p in p0..p1 {
            let i = p - p0;
            let mut ok = true;
            for (k, &b) in o.bands.iter().enumerate() {
                let (a, z) = (x1[b * npix + p].to_f64(), x2[b * npix + p].to_f64());
                ok &= present(a, missing) && present(z, missing);
                d[k] = (z - a) * o.scale[k];
            }
            if !ok {
                mag[i] = f32::NAN;
                dir[i] = f32::NAN;
                chg[i] = f32::NAN;
                sec[i] = f32::NAN;
                continue;
            }
            let rho = d[..nbands].iter().map(|v| v * v).sum::<f64>().sqrt();
            let changed = rho > o.threshold;
            mag[i] = rho as f32;
            chg[i] = if changed { 1.0 } else { 0.0 };
            let (angle, sector) = match &o.direction {
                Direction::C2va => {
                    if rho > 0.0 {
                        let s: f64 = d[..nbands].iter().sum();
                        let a = (s / ((nbands as f64).sqrt() * rho))
                            .clamp(-1.0, 1.0)
                            .acos()
                            .to_degrees();
                        let sector = match o.c2va_edges {
                            Some(e) => {
                                let k = e[1..e.len() - 1].iter().filter(|&&edge| a >= edge).count();
                                (k + 1) as f64
                            }
                            None => f64::NAN,
                        };
                        (a, sector)
                    } else {
                        (f64::NAN, if o.c2va_edges.is_some() { 1.0 } else { f64::NAN })
                    }
                }
                Direction::Polar { b1, b2, n_sectors } => {
                    let mut t = d[*b2].atan2(d[*b1]).to_degrees();
                    if t < 0.0 {
                        t += 360.0;
                    }
                    if t >= 360.0 {
                        t = 0.0;
                    }
                    (
                        t,
                        ((t / (360.0 / *n_sectors as f64)).floor() + 1.0).min(*n_sectors as f64),
                    )
                }
                Direction::Sign { red, nir, swir1 } => {
                    let code = (d[*red] > 0.0) as u32 + 2 * (d[*nir] > 0.0) as u32 + 4 * (d[*swir1] > 0.0) as u32;
                    ((code + 1) as f64, (code + 1) as f64)
                }
            };
            dir[i] = angle as f32;
            sec[i] = if changed {
                sector as f32
            } else if sector.is_nan() {
                f32::NAN
            } else {
                0.0
            };
        }
    };
    let nchunks = npix.div_ceil(CHUNK);
    let t = threads(n_jobs).min(nchunks.max(1));
    if t <= 1 {
        (0..nchunks).for_each(job);
    } else {
        pool(t).install(|| (0..nchunks).into_par_iter().for_each(job));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample64::rng::Rng;

    fn run(x1: &[f64], x2: &[f64], nb: usize, o: &CvaOptions, jobs: i32) -> [Vec<f32>; 4] {
        let npix = x1.len() / nb;
        let mut out = [vec![0f32; npix], vec![0f32; npix], vec![0f32; npix], vec![0f32; npix]];
        let [a, b, c, d] = &mut out;
        cva_batch(x1, x2, nb, npix, o, &[], a, b, c, d, jobs);
        out
    }

    #[test]
    fn magnitude_and_angles() {
        let mut rng = Rng::new(1);
        let nb = 3;
        let npix = 5000;
        let x1: Vec<f64> = (0..nb * npix).map(|_| rng.uniform()).collect();
        let x2: Vec<f64> = (0..nb * npix).map(|_| rng.uniform()).collect();
        let bands = [0, 1, 2];
        let scale = [1.0; 3];
        let o = CvaOptions {
            bands: &bands,
            scale: &scale,
            direction: Direction::C2va,
            threshold: 0.5,
            c2va_edges: None,
        };
        let [mag, dir, chg, _] = run(&x1, &x2, nb, &o, 1);
        let o2 = CvaOptions {
            direction: Direction::Polar {
                b1: 0,
                b2: 1,
                n_sectors: 4,
            },
            ..o
        };
        let [_, pol, _, sec] = run(&x1, &x2, nb, &o2, 1);
        for p in 0..npix {
            let d: Vec<f64> = (0..nb).map(|b| x2[b * npix + p] - x1[b * npix + p]).collect();
            let rho = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            let alpha = (d.iter().sum::<f64>() / (3f64.sqrt() * rho)).acos().to_degrees();
            assert!((mag[p] as f64 - rho).abs() < 1e-6);
            assert!((dir[p] as f64 - alpha).abs() < 1e-4);
            assert_eq!(chg[p], if rho > 0.5 { 1.0 } else { 0.0 });
            let mut th = d[1].atan2(d[0]).to_degrees();
            if th < 0.0 {
                th += 360.0;
            }
            assert!((pol[p] as f64 - th).abs() < 1e-4);
            if rho > 0.5 {
                assert_eq!(sec[p] as f64, (th / 90.0).floor() + 1.0);
            } else {
                assert_eq!(sec[p], 0.0);
            }
        }
        // the special directions
        let mk = |d: [f64; 3]| {
            let x1 = [0.0, 0.0, 0.0];
            let x2 = d;
            run(&x1, &x2, 3, &o, 1)[1][0] as f64
        };
        assert!(mk([0.2, 0.2, 0.2]).abs() < 1e-3);
        assert!((mk([-0.2, -0.2, -0.2]) - 180.0).abs() < 1e-3);
        assert!((mk([0.3, 0.0, 0.0]) - (1.0f64 / 3f64.sqrt()).acos().to_degrees()).abs() < 1e-3);
        assert!((mk([0.1, -0.1, 0.0]) - 90.0).abs() < 1e-3);
    }

    #[test]
    fn known_changes_by_sign_sectors_and_thresholds() {
        let (w, h) = (200, 200);
        let nb = 6; // red, nir, swir1 and three others
        let npix = w * h;
        let mut rng = Rng::new(7);
        let mut x1 = vec![0.0; nb * npix];
        let mut x2 = vec![0.0; nb * npix];
        for b in 0..nb {
            for p in 0..npix {
                let base = 0.02 + 0.28 * rng.uniform();
                x1[b * npix + p] = base;
                x2[b * npix + p] = base + 0.003 * rng.normal();
            }
        }
        let inb = |p: usize, r0: usize, c0: usize| {
            let (r, c) = (p / w, p % w);
            r >= r0 && r < r0 + 40 && c >= c0 && c < c0 + 40
        };
        for p in 0..npix {
            if inb(p, 10, 10) {
                x2[p] += 0.05; // red
                x2[npix + p] -= 0.15; // nir
                x2[2 * npix + p] += 0.10; // swir1
            }
            if inb(p, 10, 100) {
                for b in 0..nb {
                    x2[b * npix + p] -= 0.08;
                }
            }
            if inb(p, 100, 10) {
                x2[4 * npix + p] += 0.1;
            }
        }
        let bands: Vec<usize> = (0..nb).collect();
        let (rho, scale) = cva_sample(&x1, &x2, nb, w, h, &bands, false, 1, &[]);
        assert_eq!(rho.len(), npix);
        for method in ["otsu", "em"] {
            let t = if method == "otsu" {
                otsu(&rho, 1024, percentile(&rho, 0.999)).0
            } else {
                em_threshold(&rho, 0.2, 500, 1e-9).unwrap().0
            };
            let o = CvaOptions {
                bands: &bands,
                scale: &scale,
                direction: Direction::Sign {
                    red: 0,
                    nir: 1,
                    swir1: 2,
                },
                threshold: t,
                c2va_edges: None,
            };
            let [_, _, chg, sec] = run(&x1, &x2, nb, &o, 2);
            let share = |r0, c0, want: Option<f32>| {
                let mut hit = 0;
                for p in 0..npix {
                    if inb(p, r0, c0) && chg[p] == 1.0 && want.map_or(true, |s| sec[p] == s) {
                        hit += 1;
                    }
                }
                hit as f64 / 1600.0
            };
            assert!(share(10, 10, Some(6.0)) >= 0.99, "{method}: A");
            assert!(share(10, 100, Some(1.0)) >= 0.99, "{method}: B");
            assert!(share(100, 10, None) >= 0.99, "{method}: C");
            let fa = (0..npix)
                .filter(|&p| !inb(p, 10, 10) && !inb(p, 10, 100) && !inb(p, 100, 10) && chg[p] == 1.0)
                .count() as f64
                / (npix - 4800) as f64;
            assert!(fa < 0.005, "{method}: false alarms {fa}");
        }
    }

    #[test]
    fn em_recovers_a_known_mixture() {
        let mut rng = Rng::new(9);
        let v: Vec<f64> = (0..100_000)
            .map(|_| {
                if rng.uniform() < 0.8 {
                    1.0 + 0.2 * rng.normal()
                } else {
                    4.0 + 0.5 * rng.normal()
                }
            })
            .collect();
        let (t, fit) = em_threshold(&v, 0.2, 500, 1e-9).unwrap();
        assert!(fit.converged);
        let rel = |a: f64, b: f64| ((a - b) / b).abs();
        assert!(rel(fit.p[0], 0.8) < 0.02 && rel(fit.p[1], 0.2) < 0.02);
        assert!(rel(fit.mu[0], 1.0) < 0.02 && rel(fit.mu[1], 4.0) < 0.02);
        assert!(rel(fit.sigma[0], 0.2) < 0.02 && rel(fit.sigma[1], 0.5) < 0.02);
        let truth = bayes_threshold([0.8, 0.2], [1.0, 4.0], [0.2, 0.5]).unwrap();
        assert!(rel(t, truth) < 0.01, "{t} vs {truth}");
        // the Bayes root satisfies the equation
        let f = |x: f64, p: f64, m: f64, s: f64| p * (-(x - m).powi(2) / (2.0 * s * s)).exp() / s;
        assert!((f(truth, 0.8, 1.0, 0.2) - f(truth, 0.2, 4.0, 0.5)).abs() < 1e-9);
        // equal variances: the linear root
        let t = bayes_threshold([0.7, 0.3], [1.0, 3.0], [0.4, 0.4]).unwrap();
        assert!((t - (2.0 + 0.16 * (0.7f64 / 0.3).ln() / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn otsu_falls_in_the_gap() {
        let mut rng = Rng::new(3);
        let v: Vec<f64> = (0..20_000)
            .map(|_| {
                if rng.uniform() < 0.6 {
                    1.0 + 0.5 * rng.uniform()
                } else {
                    3.0 + 0.5 * rng.uniform()
                }
            })
            .collect();
        let (t, sb) = otsu(&v, 1024, percentile(&v, 0.999));
        assert!(t > 1.5 && t < 3.0, "{t}");
        assert!(sb > 0.0);
    }

    #[test]
    fn a_missing_value_blanks_only_its_pixel_and_threads_do_not_matter() {
        let mut rng = Rng::new(5);
        let (nb, npix) = (3, 20_000);
        let x1: Vec<f64> = (0..nb * npix).map(|_| rng.uniform()).collect();
        let mut x2: Vec<f64> = (0..nb * npix).map(|_| rng.uniform()).collect();
        x2[npix + 77] = f64::NAN;
        x2[2 * npix + 99] = -9999.0;
        let bands = [0, 1, 2];
        let scale = [1.0; 3];
        let o = CvaOptions {
            bands: &bands,
            scale: &scale,
            direction: Direction::C2va,
            threshold: 0.7,
            c2va_edges: Some(&[0.0, 60.0, 120.0, 180.0]),
        };
        let mut a = [vec![0f32; npix], vec![0f32; npix], vec![0f32; npix], vec![0f32; npix]];
        {
            let [m, d, c, s] = &mut a;
            cva_batch(&x1, &x2, nb, npix, &o, &[-9999.0], m, d, c, s, 1);
        }
        let mut b = a.clone();
        {
            let [m, d, c, s] = &mut b;
            cva_batch(&x1, &x2, nb, npix, &o, &[-9999.0], m, d, c, s, 4);
        }
        for plane in 0..4 {
            assert_eq!(
                a[plane].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b[plane].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(a[plane].iter().filter(|v| v.is_nan()).count(), 2, "plane {plane}");
            assert!(a[plane][77].is_nan() && a[plane][99].is_nan());
        }
        assert!(a[3].iter().all(|&s| s.is_nan() || (0.0..=3.0).contains(&s)));
    }
}
