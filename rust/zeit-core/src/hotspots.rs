//! Emerging hot spot analysis (ESRI's Space Time Pattern Mining; sfdep's `emerging_hotspot_analysis`): the
//! Getis-Ord Gi* z-score of every pixel at every date, its neighbourhood the pixels within a radius at that date and
//! the `lag` dates before (each date's neighbourhood sums from row prefix sums: O(radius) a pixel), and on each
//! pixel's series of Gi* the Mann-Kendall trend test; each pixel then classified as ESRI's documentation defines the
//! seventeen patterns (new, consecutive, intensifying, persistent, diminishing, sporadic, oscillating, historical
//! hot or cold spot, or no pattern).
//!
//! Gi* (Getis & Ord 1992; Ord & Getis 1995) with binary weights, the pixel among its neighbours:
//! z = (sum of the m neighbours' values - mean m) / (sd sqrt((n m - m^2) / (n - 1))), the mean, the standard
//! deviation (of the population) and n those of the date's pixels (as sfdep: its neighbours at earlier dates are
//! measured against the date's own) or of the whole cube. Missing values are left out of the sums and the counts.

use rayon::prelude::*;

use crate::cmath::erfc;
use crate::mann_kendall::{self, Method};
use crate::parallel;

/// The pattern codes (ESRI's): hot spots positive, cold spots negative.
pub const NEW: i8 = 1;
pub const CONSECUTIVE: i8 = 2;
pub const INTENSIFYING: i8 = 3;
pub const PERSISTENT: i8 = 4;
pub const DIMINISHING: i8 = 5;
pub const SPORADIC: i8 = 6;
pub const OSCILLATING: i8 = 7;
pub const HISTORICAL: i8 = 8;
pub const NO_PATTERN: i8 = 0;
/// Too few dates with a Gi*.
pub const NONE: i8 = i8::MIN;

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// the neighbourhood's radius in pixels (1 and a square: the 8 neighbours, queen contiguity)
    pub radius: usize,
    /// a disc (the pixels whose centres are within the radius) or a square
    pub circle: bool,
    /// the dates before each date in its neighbourhood (sfdep's `k`, ESRI's neighbourhood time step)
    pub lag: usize,
    /// the mean and deviation of the whole cube instead of each date's
    pub cube_reference: bool,
    /// the significance of the Gi* and of its trend
    pub alpha: f64,
    /// the trend test on the Gi* series
    pub method: Method,
}

/// The analysis: Gi* (dates x pixels) and per pixel the pattern, the Mann-Kendall test of its Gi* (z, p, tau) and
/// how many dates it was a significant hot or cold spot.
pub struct Emerging {
    pub gi: Vec<f64>,
    pub p: Vec<f64>,
    pub category: Vec<i8>,
    pub trend_z: Vec<f64>,
    pub trend_p: Vec<f64>,
    pub tau: Vec<f64>,
    pub n_hot: Vec<u16>,
    pub n_cold: Vec<u16>,
}

/// Each row's half-width at row offset dy.
fn half_widths(radius: usize, circle: bool) -> Vec<usize> {
    let r = radius as i64;
    (-r..=r)
        .map(|dy| {
            if circle {
                ((r * r - dy * dy) as f64 + 1e-9).sqrt().floor() as usize
            } else {
                radius
            }
        })
        .collect()
}

/// A date's neighbourhood sums of its values less `center` and of its valid pixels.
fn window_sums(
    plane: &[f64],
    ny: usize,
    nx: usize,
    center: f64,
    widths: &[usize],
    sums: &mut [f64],
    counts: &mut [f64],
) {
    // row prefix sums of the centred values and of the valid pixels
    let w1 = nx + 1;
    let mut ps = vec![0.0; ny * w1];
    let mut pc = vec![0.0; ny * w1];
    for y in 0..ny {
        for x in 0..nx {
            let v = plane[y * nx + x];
            let (s, c) = if v.is_nan() { (0.0, 0.0) } else { (v - center, 1.0) };
            ps[y * w1 + x + 1] = ps[y * w1 + x] + s;
            pc[y * w1 + x + 1] = pc[y * w1 + x] + c;
        }
    }
    let r = widths.len() / 2;
    sums.par_chunks_mut(nx)
        .zip(counts.par_chunks_mut(nx))
        .enumerate()
        .for_each(|(y, (srow, crow))| {
            for x in 0..nx {
                let (mut s, mut c) = (0.0, 0.0);
                for (k, &hw) in widths.iter().enumerate() {
                    let yy = y as i64 + k as i64 - r as i64;
                    if yy < 0 || yy >= ny as i64 {
                        continue;
                    }
                    let row = yy as usize * w1;
                    let (a, b) = (x.saturating_sub(hw), (x + hw + 1).min(nx));
                    s += ps[row + b] - ps[row + a];
                    c += pc[row + b] - pc[row + a];
                }
                srow[x] = s;
                crow[x] = c;
            }
        });
}

/// The mean, the population standard deviation and the count of the finite values.
fn moments(v: &[f64]) -> (f64, f64, f64) {
    let (mut n, mut s) = (0.0, 0.0);
    for &x in v.iter().filter(|x| !x.is_nan()) {
        n += 1.0;
        s += x;
    }
    if n == 0.0 {
        return (f64::NAN, f64::NAN, 0.0);
    }
    let mean = s / n;
    let ss: f64 = v.iter().filter(|x| !x.is_nan()).map(|&x| (x - mean) * (x - mean)).sum();
    (mean, (ss / n).sqrt(), n)
}

/// The Gi* z-scores of a cube `values` (dates x rows x columns, NaN missing): NaN where the value is missing.
#[allow(clippy::too_many_arguments)]
pub fn gi_star(
    values: &[f64],
    nt: usize,
    ny: usize,
    nx: usize,
    radius: usize,
    circle: bool,
    lag: usize,
    cube_reference: bool,
    n_jobs: i32,
) -> Vec<f64> {
    let np = ny * nx;
    assert_eq!(values.len(), nt * np, "values must be dates x rows x columns");
    let widths = half_widths(radius, circle);
    let per_date: Vec<(f64, f64, f64)> = (0..nt).map(|t| moments(&values[t * np..(t + 1) * np])).collect();
    let whole = moments(values);
    let mut gi = vec![f64::NAN; nt * np];
    parallel::pool(parallel::threads(n_jobs)).install(|| {
        // the last lag + 1 dates' sums
        let mut ring: Vec<(Vec<f64>, Vec<f64>)> = (0..=lag).map(|_| (vec![0.0; np], vec![0.0; np])).collect();
        for t in 0..nt {
            let (c_t, _, _) = per_date[t];
            let slot = t % (lag + 1);
            {
                let (s, c) = &mut ring[slot];
                let center = if c_t.is_nan() { 0.0 } else { c_t };
                window_sums(&values[t * np..(t + 1) * np], ny, nx, center, &widths, s, c);
            }
            let (mean, sd, n) = if cube_reference { whole } else { per_date[t] };
            if n < 2.0 || !(sd > 0.0) {
                continue;
            }
            let dates: Vec<(usize, f64)> = (t.saturating_sub(lag)..=t)
                .map(|tau| {
                    let c = per_date[tau].0;
                    (tau % (lag + 1), if c.is_nan() { 0.0 } else { c })
                })
                .collect();
            let plane = &values[t * np..(t + 1) * np];
            let ring = &ring;
            gi[t * np..(t + 1) * np]
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, out)| {
                    if plane[i].is_nan() {
                        return;
                    }
                    // sum of (x - mean) over the neighbours: each date's centred sums, plus (its centre - mean) a pixel
                    let (mut num, mut m) = (0.0, 0.0);
                    for &(slot, c) in &dates {
                        let (s, k) = (ring[slot].0[i], ring[slot].1[i]);
                        num += s + (c - mean) * k;
                        m += k;
                    }
                    let var = (n * m - m * m) / (n - 1.0);
                    if var > 0.0 {
                        *out = num / (sd * var.sqrt());
                    }
                });
        }
    });
    gi
}

/// The pattern of a pixel's Gi* series (its dates in order, NaN skipped) and the trend of its Gi* (`trend` the
/// sign, `significant` at alpha): ESRI's definitions, a hot spot a date whose Gi* is positive and significant.
pub fn classify(gi: &[f64], alpha: f64, trend: i32, significant: bool) -> i8 {
    let z: Vec<f64> = gi.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = z.len();
    if n < 3 {
        return NONE;
    }
    let sig = |v: f64| erfc(v.abs() / std::f64::consts::SQRT_2) < alpha;
    let hot: Vec<bool> = z.iter().map(|&v| v > 0.0 && sig(v)).collect();
    let cold: Vec<bool> = z.iter().map(|&v| v < 0.0 && sig(v)).collect();
    let kind = |h: &[bool], o: &[bool], sign: i32| -> Option<i8> {
        let count = h.iter().filter(|&&b| b).count();
        let mostly = count * 10 >= 9 * n;
        if !h[n - 1] {
            return mostly.then_some(HISTORICAL);
        }
        if count == 1 {
            return Some(NEW);
        }
        if mostly {
            return Some(match (significant, trend * sign) {
                (true, 1) => INTENSIFYING,
                (true, -1) => DIMINISHING,
                _ => PERSISTENT,
            });
        }
        let run = h.iter().rev().take_while(|&&b| b).count();
        if run == count {
            // run >= 2: count > 1
            return Some(CONSECUTIVE);
        }
        Some(if o[..n - 1].iter().any(|&b| b) {
            OSCILLATING
        } else {
            SPORADIC
        })
    };
    kind(&hot, &cold, 1)
        .or_else(|| kind(&cold, &hot, -1).map(|c| -c))
        .unwrap_or(NO_PATTERN)
}

/// The emerging hot spot analysis of a cube (dates x rows x columns).
pub fn emerging(values: &[f64], nt: usize, ny: usize, nx: usize, p: &Params, n_jobs: i32) -> Emerging {
    let np = ny * nx;
    let gi = gi_star(values, nt, ny, nx, p.radius, p.circle, p.lag, p.cube_reference, n_jobs);
    let pv: Vec<f64> = gi.iter().map(|&z| erfc(z.abs() / std::f64::consts::SQRT_2)).collect();
    // pixels x dates for the trend test
    let series = parallel::transpose(&gi, nt, np);
    let mk = mann_kendall::batch(&series, np, nt, p.method, p.alpha, -1, 1, 3, n_jobs);
    let (trend, h, trend_p, trend_z, tau) = (
        &mk[0..np],
        &mk[np..2 * np],
        &mk[2 * np..3 * np],
        &mk[3 * np..4 * np],
        &mk[4 * np..5 * np],
    );
    let mut category = vec![NONE; np];
    let mut n_hot = vec![0u16; np];
    let mut n_cold = vec![0u16; np];
    parallel::pool(parallel::threads(n_jobs)).install(|| {
        category
            .par_iter_mut()
            .zip(n_hot.par_iter_mut())
            .zip(n_cold.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((cat, hot), cold))| {
                let s = &series[i * nt..(i + 1) * nt];
                let tr = if trend[i].is_nan() { 0 } else { trend[i] as i32 };
                *cat = classify(s, p.alpha, tr, h[i] == 1.0);
                for &z in s.iter().filter(|z| !z.is_nan()) {
                    if erfc(z.abs() / std::f64::consts::SQRT_2) < p.alpha {
                        if z > 0.0 {
                            *hot += 1;
                        } else if z < 0.0 {
                            *cold += 1;
                        }
                    }
                }
            });
    });
    Emerging {
        gi,
        p: pv,
        category,
        trend_z: trend_z.to_vec(),
        trend_p: trend_p.to_vec(),
        tau: tau.to_vec(),
        n_hot,
        n_cold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gi* by its definition, pixel by pixel.
    #[allow(clippy::too_many_arguments)]
    fn brute(v: &[f64], nt: usize, ny: usize, nx: usize, r: i64, circle: bool, lag: usize, cube: bool) -> Vec<f64> {
        let np = ny * nx;
        let mut out = vec![f64::NAN; nt * np];
        for t in 0..nt {
            let pool: Vec<f64> = if cube {
                v.to_vec()
            } else {
                v[t * np..(t + 1) * np].to_vec()
            };
            let (mean, sd, n) = moments(&pool);
            for y in 0..ny as i64 {
                for x in 0..nx as i64 {
                    if v[t * np + (y as usize) * nx + x as usize].is_nan() {
                        continue;
                    }
                    let (mut s, mut m) = (0.0, 0.0);
                    for tau in t.saturating_sub(lag)..=t {
                        for yy in (y - r).max(0)..=(y + r).min(ny as i64 - 1) {
                            for xx in (x - r).max(0)..=(x + r).min(nx as i64 - 1) {
                                let (dy, dx) = (yy - y, xx - x);
                                if circle && dx * dx + dy * dy > r * r {
                                    continue;
                                }
                                let w = v[tau * np + yy as usize * nx + xx as usize];
                                if !w.is_nan() {
                                    s += w;
                                    m += 1.0;
                                }
                            }
                        }
                    }
                    out[t * np + y as usize * nx + x as usize] =
                        (s - mean * m) / (sd * ((n * m - m * m) / (n - 1.0)).sqrt());
                }
            }
        }
        out
    }

    #[test]
    fn gi_star_by_definition() {
        let (nt, ny, nx) = (5, 9, 11);
        let mut v: Vec<f64> = (0..nt * ny * nx)
            .map(|i| ((i * 7919) % 101) as f64 / 10.0 + 1000.0)
            .collect();
        v[17] = f64::NAN;
        v[ny * nx + 40] = f64::NAN;
        for &(r, circle, lag, cube) in &[
            (1, false, 0, false),
            (1, false, 1, false),
            (2, true, 2, false),
            (3, true, 1, true),
            (2, false, 0, true),
        ] {
            let a = gi_star(&v, nt, ny, nx, r, circle, lag, cube, 2);
            let b = brute(&v, nt, ny, nx, r as i64, circle, lag, cube);
            for (x, y) in a.iter().zip(&b) {
                assert!(
                    x.is_nan() == y.is_nan() && (x.is_nan() || (x - y).abs() < 1e-9),
                    "{x} {y} {r} {lag}"
                );
            }
        }
    }

    #[test]
    fn patterns_as_esri_defines_them() {
        let (h, c, o) = (3.0, -3.0, 0.1);
        let cases: &[(&[f64], i32, bool, i8)] = &[
            (&[o, o, o, o, o, o, o, o, o, h], 0, false, NEW),
            (&[c, o, o, o, o, o, o, o, o, h], 0, false, NEW),
            (&[o, o, o, o, o, o, o, o, h, h], 1, true, CONSECUTIVE),
            (&[h, h, h, h, h, h, h, h, h, h], 1, true, INTENSIFYING),
            (&[h, h, h, h, h, h, h, h, h, h], -1, true, DIMINISHING),
            (&[h, h, h, h, o, h, h, h, h, h], 1, false, PERSISTENT),
            (&[h, h, h, h, h, h, h, h, h, o], -1, true, HISTORICAL),
            (&[o, h, o, o, h, o, o, o, o, h], 0, false, SPORADIC),
            (&[o, h, o, c, o, o, o, o, h, h], 0, false, OSCILLATING),
            (&[o, o, o, o, o, o, o, o, o, c], 0, false, -NEW),
            (&[o, c, o, o, c, o, h, o, o, c], 0, false, -OSCILLATING),
            (&[c, c, c, c, c, c, c, c, c, c], -1, true, -INTENSIFYING),
            (&[o, h, o, o, h, o, c, o, o, o], 0, false, NO_PATTERN),
            (&[f64::NAN, o, f64::NAN, f64::NAN, f64::NAN], 0, false, NONE),
        ];
        for (k, &(z, trend, sig, want)) in cases.iter().enumerate() {
            assert_eq!(classify(z, 0.05, trend, sig), want, "case {k}");
        }
    }

    #[test]
    fn an_emerging_spot_is_found() {
        // a patch rising over the dates in noise: a hot spot that intensifies or emerges
        let (nt, ny, nx) = (12, 20, 20);
        let mut v = vec![0.0; nt * ny * nx];
        let mut s: u64 = 1;
        for t in 0..nt {
            for y in 0..ny {
                for x in 0..nx {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let noise = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    let patch = if (5..9).contains(&y) && (5..9).contains(&x) {
                        t as f64 * 0.4
                    } else {
                        0.0
                    };
                    v[t * ny * nx + y * nx + x] = noise + patch;
                }
            }
        }
        let p = Params {
            radius: 1,
            circle: false,
            lag: 1,
            cube_reference: false,
            alpha: 0.05,
            method: Method::Original,
        };
        let e = emerging(&v, nt, ny, nx, &p, 2);
        let c = e.category[6 * nx + 6];
        assert!([NEW, CONSECUTIVE, INTENSIFYING].contains(&c), "{c}");
        assert!(e.trend_z[6 * nx + 6] > 2.0 && e.n_hot[6 * nx + 6] >= 2);
        let far = e.category[16 * nx + 16];
        assert!(![INTENSIFYING, PERSISTENT, CONSECUTIVE].contains(&far), "{far}");
    }
}
