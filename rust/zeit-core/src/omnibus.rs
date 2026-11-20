//! The omnibus change test for a Sentinel-1 series of diagonal (dual-pol VV and VH) or single-pol covariance
//! matrices, and its factorization into the dates of change.
//!
//! Written from the papers (clean room: no implementation of the method was read): Conradsen, Nielsen and Skriver
//! (2016, IEEE TGRS 54:3007-3024) for the likelihood-ratio test, its factorization and the sequential procedure,
//! Canty, Nielsen, Conradsen and Skriver (2020, Remote Sensing 12:46) for the series, and Conradsen, Skriver and
//! Nielsen (2024, IEEE JSTARS 17:16359-16376) for the improved approximations of the distribution of the test
//! statistic for block diagonal matrices, which for the diagonal case here (blocks of size 1) reduce to the
//! parameters of `q_params` and `rj_params`. The direction of a change is the Loewner order of the difference of the
//! means (Nielsen, Skriver and Conradsen 2019).
//!
//! The input is linear power (sigma0 or gamma0), `n` the equivalent number of looks. The statistics are scale
//! invariant, so a per-pixel factor constant over time (gamma0 against sigma0) changes nothing.

use crate::dist::chi2_sf;
use crate::parallel::{pool, threads, Shared};
use crate::sample64::{present, Sample64};
use rayon::prelude::*;

/// Pixels per job.
const CHUNK: usize = 1024;

/// The planes `omnibus_batch` writes.
pub const PLANES: [&str; 8] = [
    "first_change",
    "last_change",
    "n_changes",
    "first_direction",
    "last_direction",
    "greatest_change",
    "p_change",
    "n_dates",
];

/// Which approximation gives the p-values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approx {
    /// The improved one of Conradsen et al. (2024): `P(-2 rho ln Q <= z)` is a chi-square of `f` degrees of freedom
    /// corrected by a term of `f + 4` degrees.
    Improved,
    /// Wilks' chi-square of `-2 ln Q` (`rho = 1`, `omega2 = 0`).
    Chi2,
}

#[derive(Clone, Copy, Debug)]
pub struct OmnibusOptions {
    /// The equivalent number of looks.
    pub enl: f64,
    /// The significance level of each test.
    pub alpha: f64,
    pub approx: Approx,
    /// The number of bands: 1 (single-pol) or 2 (dual-pol diagonal).
    pub p: usize,
}

/// The parameters of the test of equality of all `k` dates: degrees of freedom `f`, `rho` and `omega2`.
pub fn q_params(p: usize, n: f64, k: usize) -> (f64, f64, f64) {
    let kf = k as f64;
    let f = (kf - 1.0) * p as f64;
    let rho = 1.0 - (1.0 / (6.0 * (kf - 1.0))) * (kf / n - 1.0 / (n * kf));
    let omega2 = -(f / 4.0) * (1.0 - 1.0 / rho).powi(2);
    (f, rho, omega2)
}

/// The parameters of the test of date `j` against the `j - 1` before it: `g`, `rho_j` and `omega2_j`.
pub fn rj_params(p: usize, n: f64, j: usize) -> (f64, f64, f64) {
    let jf = j as f64;
    let g = p as f64;
    let rho = 1.0 - (1.0 / (6.0 * n)) * (1.0 + 1.0 / (jf * (jf - 1.0)));
    let omega2 = -(g / 4.0) * (1.0 - 1.0 / rho).powi(2);
    (g, rho, omega2)
}

/// The probability of a value above `z`: `Q_f(z) - omega2 (Q_f(z) - Q_{f+4}(z))`, `Q_d` the chi-square tail of `d`
/// degrees of freedom, in [0, 1].
pub fn pvalue(z: f64, f: f64, omega2: f64) -> f64 {
    let qf = chi2_sf(z, f);
    if omega2 == 0.0 {
        return qf;
    }
    (qf - omega2 * (qf - chi2_sf(z, f + 4.0))).clamp(0.0, 1.0)
}

/// `ln Q` of a series of `k` dates (`series`: `k x p`, row per date, linear power): `n sum_b [k ln k + sum_i ln C_ib
/// - k ln sum_i C_ib]`.
pub fn ln_q(series: &[f64], p: usize, n: f64, k: usize) -> f64 {
    let kf = k as f64;
    let mut total = 0.0;
    for b in 0..p {
        let (mut s, mut l) = (0.0, 0.0);
        for i in 0..k {
            s += series[i * p + b];
            l += series[i * p + b].ln();
        }
        total += kf * kf.ln() + l - kf * s.ln();
    }
    n * total
}

/// `ln R_j` for the sum `sum` of the `j - 1` first dates and the next date `c` (each `p` values).
pub fn ln_r(sum: &[f64], c: &[f64], p: usize, n: f64, j: usize) -> f64 {
    let jf = j as f64;
    let head = jf * jf.ln() - (jf - 1.0) * (jf - 1.0).ln();
    let mut total = 0.0;
    for b in 0..p {
        total += head + (jf - 1.0) * sum[b].ln() + c[b].ln() - jf * (sum[b] + c[b]).ln();
    }
    n * total
}

/// The parameters for every number of dates up to `kmax`, computed once.
pub struct Tables {
    q: Vec<(f64, f64, f64)>,
    r: Vec<(f64, f64, f64)>,
}

impl Tables {
    pub fn new(p: usize, n: f64, kmax: usize, approx: Approx) -> Tables {
        let fix = |t: (f64, f64, f64)| if approx == Approx::Chi2 { (t.0, 1.0, 0.0) } else { t };
        let mut q = vec![(0.0, 1.0, 0.0); kmax + 1];
        let mut r = vec![(0.0, 1.0, 0.0); kmax + 1];
        for m in 2..=kmax {
            q[m] = fix(q_params(p, n, m));
            r[m] = fix(rj_params(p, n, m));
        }
        Tables { q, r }
    }
}

/// What the test found for a pixel.
#[derive(Clone, Debug, Default)]
pub struct PixelChanges {
    pub n_dates: u32,
    /// The p-value of the test over the whole valid series.
    pub p_change: f64,
    /// `(index in the series, direction, -2 rho ln R_j, p-value)` of each detected change, in time order.
    pub changes: Vec<(u32, u8, f64, f64)>,
}

/// The sequential procedure on one pixel: `series` is `k x p` (the valid dates only, linear power, positive).
/// `scratch` holds the suffix sums (it is resized as needed).
pub fn omnibus_pixel(
    series: &[f64],
    k: usize,
    o: &OmnibusOptions,
    tables: &Tables,
    scratch: &mut Vec<f64>,
    out: &mut PixelChanges,
) {
    let p = o.p;
    out.n_dates = k as u32;
    out.p_change = f64::NAN;
    out.changes.clear();
    if k < 2 {
        return;
    }
    // suffix sums of C and ln C per band: row l holds the sums over the dates l..k
    scratch.clear();
    scratch.resize((k + 1) * 2 * p, 0.0);
    for l in (0..k).rev() {
        for b in 0..p {
            let (s, e) = (scratch[(l + 1) * 2 * p + b], scratch[(l + 1) * 2 * p + p + b]);
            scratch[l * 2 * p + b] = s + series[l * p + b];
            scratch[l * 2 * p + p + b] = e + series[l * p + b].ln();
        }
    }
    let mut sum = [0.0f64; 2];
    let mut l0 = 0usize;
    while k - l0 >= 2 {
        let m = k - l0;
        let mf = m as f64;
        let mut lnq = 0.0;
        for b in 0..p {
            lnq += mf * mf.ln() + scratch[l0 * 2 * p + p + b] - mf * scratch[l0 * 2 * p + b].ln();
        }
        let lnq = o.enl * lnq;
        let (f, rho, w2) = tables.q[m];
        let z = (-2.0 * rho * lnq).max(0.0);
        let pq = pvalue(z, f, w2);
        if l0 == 0 {
            out.p_change = pq;
        }
        if !(pq < o.alpha) {
            break;
        }
        // the first date j of the subsequence whose test rejects
        sum[..p].copy_from_slice(&series[l0 * p..l0 * p + p]);
        let mut found = false;
        for j in 2..=m {
            let t = l0 + j - 1;
            let c = &series[t * p..t * p + p];
            let lnr = ln_r(&sum[..p], c, p, o.enl, j);
            let (g, rho_j, w2_j) = tables.r[j];
            let zj = (-2.0 * rho_j * lnr).max(0.0);
            let pj = pvalue(zj, g, w2_j);
            if pj < o.alpha {
                let (mut up, mut down) = (true, true);
                for b in 0..p {
                    let d = c[b] - sum[b] / (j as f64 - 1.0);
                    up &= d > 0.0;
                    down &= d < 0.0;
                }
                let dir = if up {
                    1
                } else if down {
                    2
                } else {
                    3
                };
                out.changes.push((t as u32, dir, zj, pj));
                l0 = t;
                found = true;
                break;
            }
            for b in 0..p {
                sum[b] += c[b];
            }
        }
        if !found {
            break;
        }
    }
}

/// The planes of `PLANES` (and optionally the direction of the change at each date) for every pixel of a series.
/// `x` is `(date, band, pixel)` linear power with `nb` bands and `npix` pixels, `bands` the indices of VV (and VH);
/// a (pixel, date) with a band missing (NaN, infinite, among `missing`) or not strictly positive is skipped, the
/// pixel using its own valid dates. The dates in the planes are indices into the dates of `x` (as f32); a pixel with
/// fewer than two valid dates is NaN in every plane. `out` is `(8, npix)`; `out_per_date` `(k, npix)`: the direction
/// code at the date of each change, 0 elsewhere, NaN where that date is not valid.
#[allow(clippy::too_many_arguments)]
pub fn omnibus_batch<T: Sample64>(
    x: &[T],
    nb: usize,
    npix: usize,
    k: usize,
    bands: &[usize],
    o: &OmnibusOptions,
    missing: &[f64],
    out: &mut [f32],
    out_per_date: Option<&mut [f32]>,
    n_jobs: i32,
) {
    assert_eq!(o.p, bands.len());
    assert!(o.p == 1 || o.p == 2, "one or two bands");
    assert_eq!(x.len(), k * nb * npix);
    assert_eq!(out.len(), PLANES.len() * npix);
    if let Some(d) = &out_per_date {
        assert_eq!(d.len(), k * npix);
    }
    let tables = Tables::new(o.p, o.enl, k.max(2), o.approx);
    let shared = Shared::new(out);
    let per_date = out_per_date.map(Shared::new);
    let p = o.p;
    let job = |c: usize| {
        let p0 = c * CHUNK;
        let p1 = ((c + 1) * CHUNK).min(npix);
        let mut series = vec![0.0f64; k * p];
        let mut idx = vec![0u32; k];
        let mut scratch = Vec::new();
        let mut pc = PixelChanges::default();
        for pix in p0..p1 {
            let mut n = 0usize;
            for d in 0..k {
                let mut ok = true;
                for (b, &band) in bands.iter().enumerate() {
                    let v = x[(d * nb + band) * npix + pix].to_f64();
                    ok &= present(v, missing) && v > 0.0;
                    series[n * p + b] = v;
                }
                if ok {
                    idx[n] = d as u32;
                    n += 1;
                }
            }
            omnibus_pixel(&series, n, o, &tables, &mut scratch, &mut pc);
            // SAFETY: each job writes its own pixels of each plane.
            let put = |plane: usize, v: f32| unsafe {
                shared.slice_mut(plane * npix + pix..plane * npix + pix + 1)[0] = v;
            };
            if n < 2 {
                for plane in 0..PLANES.len() {
                    put(plane, f32::NAN);
                }
                if let Some(pd) = &per_date {
                    for d in 0..k {
                        unsafe { pd.slice_mut(d * npix + pix..d * npix + pix + 1)[0] = f32::NAN };
                    }
                }
                continue;
            }
            let nc = pc.changes.len();
            put(7, n as f32);
            put(6, pc.p_change as f32);
            put(2, nc as f32);
            if nc == 0 {
                put(0, f32::NAN);
                put(1, f32::NAN);
                put(3, 0.0);
                put(4, 0.0);
                put(5, f32::NAN);
            } else {
                let first = pc.changes[0];
                let last = pc.changes[nc - 1];
                let mut best = pc.changes[0];
                for ch in &pc.changes {
                    if ch.2 > best.2 {
                        best = *ch;
                    }
                }
                put(0, idx[first.0 as usize] as f32);
                put(1, idx[last.0 as usize] as f32);
                put(3, first.1 as f32);
                put(4, last.1 as f32);
                put(5, idx[best.0 as usize] as f32);
            }
            if let Some(pd) = &per_date {
                for d in 0..k {
                    unsafe { pd.slice_mut(d * npix + pix..d * npix + pix + 1)[0] = f32::NAN };
                }
                for q in 0..n {
                    unsafe { pd.slice_mut(idx[q] as usize * npix + pix..idx[q] as usize * npix + pix + 1)[0] = 0.0 };
                }
                for ch in &pc.changes {
                    unsafe {
                        pd.slice_mut(
                            idx[ch.0 as usize] as usize * npix + pix..idx[ch.0 as usize] as usize * npix + pix + 1,
                        )[0] = ch.1 as f32
                    };
                }
            }
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

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn the_parameters_of_the_papers() {
        // Conradsen et al. (2024), Frankfurt: p = 2, n = 12, k = 15
        let (f, rho, w2) = q_params(2, 12.0, 15);
        assert_eq!(f, 28.0);
        assert!(
            close(rho, 0.985_185_185_185_185_2, 1e-12) && close(w2, -0.001_582_904_629_99, 1e-9),
            "{rho} {w2}"
        );
        let (g, rho, w2) = rj_params(2, 12.0, 10);
        assert_eq!(g, 2.0);
        assert!(
            close(rho, 0.985_956_790_123_456_8, 1e-12) && close(w2, -0.000_101_434_808_286, 1e-9),
            "{rho} {w2}"
        );
        // n = 4.4, k = 10
        let (_, rho, w2) = q_params(2, 4.4, 10);
        assert!(
            close(rho, 0.958_333_333_333_333, 1e-12) && close(w2, -0.008_506_616_257_08, 1e-9),
            "{w2}"
        );
        let (f1, rho1, w21) = q_params(1, 4.4, 10);
        assert_eq!(f1, 9.0);
        assert!(close(rho1, rho, 1e-14) && close(w21, -0.004_253_308_128_54, 1e-9));
        // n = 4.4, j = 2
        let (_, rho, w2) = rj_params(2, 4.4, 2);
        assert!(
            close(rho, 0.943_181_818_181_818, 1e-12) && close(w2, -0.001_814_486_863_11, 1e-9),
            "{w2}"
        );
        let (g1, _, w21) = rj_params(1, 4.4, 2);
        assert_eq!(g1, 1.0);
        assert!(close(w21, -0.000_907_243_431_557, 1e-9));
        // the chi-square mode
        let t = Tables::new(2, 4.4, 5, Approx::Chi2);
        assert_eq!((t.q[3].1, t.q[3].2, t.r[3].1, t.r[3].2), (1.0, 0.0, 1.0, 0.0));
        // the p-value reduces to the chi-square tail without the correction
        assert_eq!(pvalue(7.0, 4.0, 0.0), chi2_sf(7.0, 4.0));
    }

    fn random_series(rng: &mut Rng, k: usize, p: usize) -> Vec<f64> {
        (0..k * p).map(|_| 0.05 + rng.gamma(2.0)).collect()
    }

    #[test]
    fn ln_q_is_the_sum_of_the_ln_r() {
        let mut rng = Rng::new(1);
        for (k, p) in [(2usize, 1usize), (5, 2), (12, 2), (9, 1)] {
            let s = random_series(&mut rng, k, p);
            let n = 4.4;
            let mut sum: Vec<f64> = s[..p].to_vec();
            let mut total = 0.0;
            for j in 2..=k {
                let c = &s[(j - 1) * p..j * p];
                total += ln_r(&sum, c, p, n, j);
                for b in 0..p {
                    sum[b] += c[b];
                }
            }
            let q = ln_q(&s, p, n, k);
            assert!(close(total, q, 1e-10), "k {k} p {p}: {total} vs {q}");
            assert!(q <= 1e-12);
        }
    }

    fn pixel(series: &[f64], k: usize, p: usize, approx: Approx, alpha: f64) -> PixelChanges {
        let o = OmnibusOptions {
            enl: 4.4,
            alpha,
            approx,
            p,
        };
        let t = Tables::new(p, 4.4, k.max(2), approx);
        let mut out = PixelChanges::default();
        omnibus_pixel(series, k, &o, &t, &mut Vec::new(), &mut out);
        out
    }

    #[test]
    fn scale_invariance() {
        let mut rng = Rng::new(2);
        for _ in 0..200 {
            let s = random_series(&mut rng, 10, 2);
            let a = pixel(&s, 10, 2, Approx::Improved, 0.01);
            let (c0, c1) = (rng.uniform() * 50.0 + 0.01, rng.uniform() * 50.0 + 0.01);
            let scaled: Vec<f64> = s.chunks(2).flat_map(|r| [r[0] * c0, r[1] * c1]).collect();
            let b = pixel(&scaled, 10, 2, Approx::Improved, 0.01);
            assert!((a.p_change - b.p_change).abs() < 1e-9 * a.p_change.max(1e-12) + 1e-14);
            assert_eq!(a.changes.len(), b.changes.len());
            for (u, v) in a.changes.iter().zip(&b.changes) {
                assert_eq!((u.0, u.1), (v.0, v.1));
            }
        }
    }

    /// Series under no change: a mean per band, gamma speckle of `n` looks.
    fn null_series(rng: &mut Rng, k: usize, p: usize, n: f64) -> Vec<f64> {
        let mu: Vec<f64> = (0..p).map(|_| (rng.normal() * 0.8).exp() * 0.1).collect();
        (0..k * p).map(|i| mu[i % p] * rng.gamma(n) / n).collect()
    }

    #[test]
    fn false_alarm_rate_under_no_change() {
        let (k, n) = (10usize, 4.4);
        let pixels = 200_000usize;
        for p in [2usize, 1] {
            let mut rng = Rng::new(10 + p as u64);
            for alpha in [0.01, 0.05] {
                let bound = 4.0 * (alpha * (1.0 - alpha) / pixels as f64).sqrt();
                let (mut rej_q, mut any) = (0usize, 0usize);
                let mut rej_r = vec![0usize; k + 1];
                let (mut rej_q_chi, mut rej_r_chi) = (0usize, vec![0usize; k + 1]);
                let o = OmnibusOptions {
                    enl: n,
                    alpha,
                    approx: Approx::Improved,
                    p,
                };
                let (ti, tc) = (
                    Tables::new(p, n, k, Approx::Improved),
                    Tables::new(p, n, k, Approx::Chi2),
                );
                let mut out = PixelChanges::default();
                let mut scratch = Vec::new();
                for _ in 0..pixels {
                    let s = null_series(&mut rng, k, p, n);
                    omnibus_pixel(&s, k, &o, &ti, &mut scratch, &mut out);
                    rej_q += (out.p_change < alpha) as usize;
                    any += !out.changes.is_empty() as usize;
                    // every R_j on its own (they are independent and uniform under no change)
                    let mut sum: Vec<f64> = s[..p].to_vec();
                    for j in 2..=k {
                        let c = &s[(j - 1) * p..j * p];
                        let lnr = ln_r(&sum, c, p, n, j);
                        let (g, rho, w2) = ti.r[j];
                        rej_r[j] += (pvalue((-2.0 * rho * lnr).max(0.0), g, w2) < alpha) as usize;
                        let (g, rho, w2) = tc.r[j];
                        rej_r_chi[j] += (pvalue((-2.0 * rho * lnr).max(0.0), g, w2) < alpha) as usize;
                        for b in 0..p {
                            sum[b] += c[b];
                        }
                    }
                    let (f, rho, w2) = tc.q[k];
                    rej_q_chi += (pvalue((-2.0 * rho * ln_q(&s, p, n, k)).max(0.0), f, w2) < alpha) as usize;
                }
                let rate = |c: usize| c as f64 / pixels as f64;
                assert!(
                    (rate(rej_q) - alpha).abs() < bound,
                    "p {p} alpha {alpha}: Q rate {}",
                    rate(rej_q)
                );
                assert!(
                    rate(any) < alpha + bound,
                    "p {p} alpha {alpha}: any change {}",
                    rate(any)
                );
                for j in 2..=k {
                    assert!(
                        (rate(rej_r[j]) - alpha).abs() < bound,
                        "p {p} alpha {alpha} j {j}: {}",
                        rate(rej_r[j])
                    );
                }
                // reported, not asserted: the plain chi-square is anti-conservative at few looks
                eprintln!(
                    "p {p} alpha {alpha}: improved Q {:.5}, chi2 Q {:.5}; chi2 R_j {:?}",
                    rate(rej_q),
                    rate(rej_q_chi),
                    (2..=k)
                        .map(|j| (rate(rej_r_chi[j]) * 1e4).round() / 1e4)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    /// k dates, p = 2, with the factors of each date for each band applied to a speckled background.
    fn series_with(rng: &mut Rng, factors: &[[f64; 2]], n: f64) -> Vec<f64> {
        let mu = [0.12 * (rng.normal() * 0.3).exp(), 0.03 * (rng.normal() * 0.3).exp()];
        factors
            .iter()
            .flat_map(|f| [mu[0] * f[0] * rng.gamma(n) / n, mu[1] * f[1] * rng.gamma(n) / n])
            .collect()
    }

    #[test]
    fn known_changes_and_their_direction() {
        let mut rng = Rng::new(7);
        let k = 12;
        let trials = 400;
        // each of the tests of the dates before the change has its own 1 % of false alarms, so about 5 % of the
        // pixels report a first change too early
        let (mut a_ok, mut b_ok, mut c_ok, mut bg_changed) = (0, 0, 0, 0);
        for _ in 0..trials {
            // A: x 0.1 from date 6, B: x 8 at date 3 back at date 8, C: VV x 8 and VH x 0.1 from 5
            // (a single new date against the mean of those before has limited power: x 0.3 is found at its date
            // in about half the pixels and a date or two later in the others)
            let fa: Vec<[f64; 2]> = (0..k).map(|d| if d >= 6 { [0.1, 0.1] } else { [1.0, 1.0] }).collect();
            let fb: Vec<[f64; 2]> = (0..k)
                .map(|d| if (3..8).contains(&d) { [8.0, 8.0] } else { [1.0, 1.0] })
                .collect();
            let fc: Vec<[f64; 2]> = (0..k).map(|d| if d >= 5 { [8.0, 0.1] } else { [1.0, 1.0] }).collect();
            let a = pixel(&series_with(&mut rng, &fa, 4.4), k, 2, Approx::Improved, 0.01);
            a_ok += (a.changes.len() == 1 && a.changes[0].0 == 6 && a.changes[0].1 == 2) as usize;
            let b = pixel(&series_with(&mut rng, &fb, 4.4), k, 2, Approx::Improved, 0.01);
            b_ok += (b.changes.len() == 2
                && b.changes[0].0 == 3
                && b.changes[0].1 == 1
                && b.changes[1].0 == 8
                && b.changes[1].1 == 2) as usize;
            let c = pixel(&series_with(&mut rng, &fc, 4.4), k, 2, Approx::Improved, 0.01);
            c_ok += (!c.changes.is_empty() && c.changes[0].0 == 5 && c.changes[0].1 == 3) as usize;
            let fz: Vec<[f64; 2]> = vec![[1.0, 1.0]; k];
            bg_changed += !pixel(&series_with(&mut rng, &fz, 4.4), k, 2, Approx::Improved, 0.01)
                .changes
                .is_empty() as usize;
        }
        let share = |c: usize| c as f64 / trials as f64;
        assert!(share(a_ok) >= 0.90, "A {}", share(a_ok));
        assert!(share(b_ok) >= 0.80, "B {}", share(b_ok));
        assert!(share(c_ok) >= 0.85, "C {}", share(c_ok));
        assert!(
            share(bg_changed) <= 1.5 * 0.01 + 0.005,
            "background {}",
            share(bg_changed)
        );
    }

    #[test]
    fn batch_missing_dates_single_pol_and_threads() {
        let (k, nb, npix) = (12usize, 3usize, 2000usize);
        let mut rng = Rng::new(3);
        // bands: VV, VH, an unused band; the first 1000 pixels darken from date 6
        let mut x = vec![0.0f32; k * nb * npix];
        for pix in 0..npix {
            let mu = [0.12, 0.03];
            for d in 0..k {
                let f = if pix < 1000 && d >= 6 { 0.1 } else { 1.0 };
                for b in 0..2 {
                    x[(d * nb + b) * npix + pix] = (mu[b] * f * rng.gamma(4.4) / 4.4) as f32;
                }
                x[(d * nb + 2) * npix + pix] = f32::NAN;
            }
        }
        // pixel 0 misses date 6 (zero), pixel 1 is NaN in date 2 band 1, pixel 2 has one valid date only
        x[(6 * nb) * npix] = 0.0;
        x[(2 * nb + 1) * npix + 1] = f32::NAN;
        for d in 1..k {
            x[(d * nb) * npix + 2] = f32::NAN;
        }
        let o = OmnibusOptions {
            enl: 4.4,
            alpha: 0.01,
            approx: Approx::Improved,
            p: 2,
        };
        let run = |jobs: i32| {
            let mut out = vec![0f32; PLANES.len() * npix];
            let mut per = vec![0f32; k * npix];
            omnibus_batch(&x, nb, npix, k, &[0, 1], &o, &[], &mut out, Some(&mut per), jobs);
            (out, per)
        };
        let (a, pa) = run(1);
        let (b, pb) = run(4);
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            pa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            pb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // pixel 2: a single valid date: NaN everywhere
        assert!((0..PLANES.len()).all(|pl| a[pl * npix + 2].is_nan()));
        // pixel 0 lost date 6: the change is reported at the next valid date, 7
        let first = a[0 * npix];
        assert!(first == 6.0 || first == 7.0, "pixel 0 first change {first}");
        assert_eq!(a[7 * npix], 11.0, "valid dates of pixel 0");
        assert_eq!(a[7 * npix + 5], 12.0);
        let darkened = (3..1000)
            .filter(|&p| a[p] == 6.0 && a[3 * npix + p] == 2.0 && a[2 * npix + p] == 1.0)
            .count();
        assert!(darkened > 800, "{darkened} of 997");
        assert_eq!(pa[6 * npix + 5], 2.0);
        assert!(pa[6 * npix].is_nan());
        // single-pol on the same data (VV only) runs with p = 1
        let o1 = OmnibusOptions { p: 1, ..o };
        let mut out = vec![0f32; PLANES.len() * npix];
        omnibus_batch(&x, nb, npix, k, &[0], &o1, &[], &mut out, None, 2);
        let found = (3..1000).filter(|&p| out[p] == 6.0).count();
        assert!(found > 700, "{found}");
    }
}
