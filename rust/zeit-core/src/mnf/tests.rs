//! The tests of the MNF, on images generated here: smooth latent fields mixed into the bands, plus white noise.

use super::*;
use std::f64::consts::PI;

struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        ((splitmix64(self.0) >> 11) as f64) / 9007199254740992.0
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (1.0 - self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos()
    }
}

/// The signal's standard deviation in units of the noise's. The spec asks for about 20 with periods of 400 to 1000
/// pixels, but at 20 no field of 128 pixels is smooth enough for its own bound (half the variance of the
/// differences under 2% of the noise's): 4 with periods of 350 to 700 pixels meets it with independent fields.
const SIGNAL_STD: f64 = 4.0;

struct Synth {
    b: usize,
    h: usize,
    w: usize,
    r: usize,
    /// planar `b x h x w`
    x: Vec<f64>,
    /// the signal, same layout
    s: Vec<f64>,
    /// `b x r`
    m: Vec<f64>,
    sigma: Vec<f64>,
    /// `r x h x w`
    f: Vec<f64>,
}

fn synth(h: usize, w: usize, b: usize, r: usize, hetero: bool, seed: u64) -> Synth {
    let mut rng = Rng(seed);
    let n = h * w;
    let mut f = vec![0.0; r * n];
    for l in 0..r {
        let waves: Vec<(f64, f64, f64, f64)> = (0..4)
            .map(|_| (0.5 + rng.u(), PI * rng.u(), 350.0 + 350.0 * rng.u(), 2.0 * PI * rng.u()))
            .collect();
        for y in 0..h {
            for x in 0..w {
                f[l * n + y * w + x] = waves
                    .iter()
                    .map(|&(c, th, t, ph)| c * (2.0 * PI * (y as f64 * th.cos() + x as f64 * th.sin()) / t + ph).sin())
                    .sum();
            }
        }
        let col = &mut f[l * n..][..n];
        let mu = col.iter().sum::<f64>() / n as f64;
        let sd = (col.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1) as f64).sqrt();
        col.iter_mut().for_each(|v| *v = (*v - mu) / sd);
    }
    let sigma: Vec<f64> = (0..b)
        .map(|i| {
            if hetero {
                0.5 * 10f64.powf(i as f64 / (b - 1) as f64)
            } else {
                1.0
            }
        })
        .collect();
    let mut m: Vec<f64> = (0..b * r).map(|_| rng.normal()).collect();
    let mut s = vec![0.0; b * n];
    for j in 0..b {
        for p in 0..n {
            s[j * n + p] = (0..r).map(|l| m[j * r + l] * f[l * n + p]).sum();
        }
        let col = &s[j * n..][..n];
        let mu = col.iter().sum::<f64>() / n as f64;
        let sd = (col.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (n - 1) as f64).sqrt();
        let k = SIGNAL_STD * sigma[j] / sd;
        (0..r).for_each(|l| m[j * r + l] *= k);
        s[j * n..][..n].iter_mut().for_each(|v| *v *= k);
    }
    let mut x = s.clone();
    for j in 0..b {
        for p in 0..n {
            x[j * n + p] += sigma[j] * rng.normal();
        }
    }
    Synth {
        b,
        h,
        w,
        r,
        x,
        s,
        m,
        sigma,
        f,
    }
}

fn fit_with(x: &[f64], nt: usize, b: usize, h: usize, w: usize, noise: Noise, n_jobs: i32) -> MnfTransform {
    let opts = StatsOptions::new(noise);
    mnf_fit(&mnf_stats_batch(x, nt, b, h, w, &opts, n_jobs).unwrap()).unwrap()
}

fn fit(sy: &Synth, noise: Noise) -> MnfTransform {
    fit_with(&sy.x, 1, sy.b, sy.h, sy.w, noise, 2)
}

fn forward_all(x: &[f64], b: usize, npix: usize, tr: &MnfTransform) -> Vec<f32> {
    let mut out = vec![0f32; tr.m * npix * (x.len() / (b * npix))];
    mnf_forward_batch(x, b, npix, tr, tr.m, Scale::Noise, &[], None, &mut out, 2);
    out
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn std_of(v: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = v.clone().count() as f64;
    let mu = v.clone().sum::<f64>() / n;
    (v.map(|x| (x - mu) * (x - mu)).sum::<f64>() / (n - 1.0)).sqrt()
}

/// The spans of the columns of `a` and `b` (`n x k` row-major): the largest principal angle, degrees.
fn max_angle(a: &[f64], b: &[f64], n: usize, k: usize) -> f64 {
    let ortho = |m: &[f64]| -> Vec<f64> {
        let mut q = m.to_vec();
        for c in 0..k {
            for p in 0..c {
                let d: f64 = (0..n).map(|i| q[i * k + c] * q[i * k + p]).sum();
                (0..n).for_each(|i| q[i * k + c] -= d * q[i * k + p]);
            }
            let nrm = (0..n).map(|i| q[i * k + c] * q[i * k + c]).sum::<f64>().sqrt();
            (0..n).for_each(|i| q[i * k + c] /= nrm);
        }
        q
    };
    let (qa, qb) = (ortho(a), ortho(b));
    let c = mat_mul(&transpose(&qa, n, k), k, n, &qb, k);
    let g = mat_mul(&c, k, k, &transpose(&c, k, k), k);
    let (vals, _) = sym_eig_desc(&g, k).unwrap();
    vals[k - 1].clamp(0.0, 1.0).sqrt().acos().to_degrees()
}

/// The covariance of the fields and `1 + eig(Sn^{-1/2} M Sf M' Sn^{-1/2})`, the population eigenvalues.
fn population(sy: &Synth) -> Vec<f64> {
    let (b, r, n) = (sy.b, sy.r, sy.h * sy.w);
    let mut sf = vec![0.0; r * r];
    for p in 0..r {
        for q in 0..r {
            let (fp, fq) = (&sy.f[p * n..][..n], &sy.f[q * n..][..n]);
            let (mp, mq) = (fp.iter().sum::<f64>() / n as f64, fq.iter().sum::<f64>() / n as f64);
            sf[p * r + q] = fp.iter().zip(fq).map(|(a, c)| (a - mp) * (c - mq)).sum::<f64>() / (n - 1) as f64;
        }
    }
    let mw: Vec<f64> = (0..b * r).map(|i| sy.m[i] / sy.sigma[i / r]).collect();
    let g = mat_mul(&mat_mul(&mw, b, r, &sf, r), b, r, &transpose(&mw, b, r), b);
    sym_eig_desc(&g, b).unwrap().0.iter().map(|v| 1.0 + v).collect()
}

fn sample_cov(x: &[f64], b: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mean: Vec<f64> = (0..b).map(|j| x[j * n..][..n].iter().sum::<f64>() / n as f64).collect();
    let mut cov = vec![0.0; b * b];
    for i in 0..b {
        for j in 0..b {
            cov[i * b + j] = (0..n)
                .map(|p| (x[i * n + p] - mean[i]) * (x[j * n + p] - mean[j]))
                .sum::<f64>()
                / (n - 1) as f64;
        }
    }
    (mean, cov)
}

// ---------------------------------------------------------------------------

fn random_spd(n: usize, rng: &mut Rng, shift: f64) -> Vec<f64> {
    let g: Vec<f64> = (0..n * 2 * n).map(|_| rng.normal()).collect();
    let mut a = mat_mul(&g, n, 2 * n, &transpose(&g, n, 2 * n), n);
    a.iter_mut().for_each(|v| *v /= (2 * n) as f64);
    (0..n).for_each(|i| a[i * n + i] += shift);
    a
}

fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            l[i * n + j] = if i == j {
                (a[i * n + i] - s).sqrt()
            } else {
                (a[i * n + j] - s) / l[j * n + j]
            };
        }
    }
    l
}

/// `L^{-1} b` for the lower `l` (`n x n`) and `b` (`n x c`).
fn solve_lower(l: &[f64], b: &[f64], n: usize, c: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    for col in 0..c {
        for i in 0..n {
            let s: f64 = (0..i).map(|k| l[i * n + k] * x[k * c + col]).sum();
            x[i * c + col] = (x[i * c + col] - s) / l[i * n + i];
        }
    }
    x
}

/// `L^{-T} b`.
fn solve_upper_t(l: &[f64], b: &[f64], n: usize, c: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    for col in 0..c {
        for i in (0..n).rev() {
            let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k * c + col]).sum();
            x[i * c + col] = (x[i * c + col] - s) / l[i * n + i];
        }
    }
    x
}

#[test]
fn algebra_given_covariances() {
    let nb = 8;
    let mut rng = Rng(11);
    let scale: Vec<f64> = (0..nb).map(|_| 0.1 + 100.0 * rng.u()).collect();
    let mut sigma = random_spd(nb, &mut rng, 0.5);
    let mut sn = random_spd(nb, &mut rng, 0.2);
    for i in 0..nb {
        for j in 0..nb {
            sigma[i * nb + j] *= scale[i] * scale[j];
            sn[i * nb + j] *= scale[i] * scale[j] * 0.2;
        }
    }
    let tr = mnf_fit_cov(&vec![1.0; nb], &sigma, &sn, nb).unwrap();
    assert_eq!((tr.m, tr.noise_floored), (nb, 0));
    let a = transpose(&tr.forward, nb, nb);
    let at = &tr.forward;
    let ans = mat_mul(&mat_mul(at, nb, nb, &sn, nb), nb, nb, &a, nb);
    let eye: Vec<f64> = (0..nb * nb).map(|k| if k / nb == k % nb { 1.0 } else { 0.0 }).collect();
    assert!(max_abs_diff(&ans, &eye) < 1e-10, "A' Sn A");
    let asa = mat_mul(&mat_mul(at, nb, nb, &sigma, nb), nb, nb, &a, nb);
    let lmax = tr.eigenvalues[0];
    for i in 0..nb {
        for j in 0..nb {
            let want = if i == j { tr.eigenvalues[i] } else { 0.0 };
            assert!(
                (asa[i * nb + j] - want).abs() <= 1e-9 * if i == j { want.abs() } else { lmax },
                "A' S A"
            );
        }
    }
    let atp = mat_mul(at, nb, nb, &tr.inverse, nb);
    assert!(max_abs_diff(&atp, &eye) < 1e-10, "A' P");
    // the Cholesky route (Roger 1994): K = L^-1 S L^-T, a = L^-T v
    let l = cholesky(&sn, nb);
    let half = solve_lower(&l, &sigma, nb, nb);
    let k = transpose(&solve_lower(&l, &transpose(&half, nb, nb), nb, nb), nb, nb);
    let ks: Vec<f64> = (0..nb * nb).map(|q| 0.5 * (k[q] + k[(q % nb) * nb + q / nb])).collect();
    let (vals, v) = sym_eig_desc(&ks, nb).unwrap();
    let aref = solve_upper_t(&l, &v, nb, nb);
    for i in 0..nb {
        assert!((vals[i] - tr.eigenvalues[i]).abs() <= 1e-10 * vals[i], "eigenvalue {i}");
        let ours: Vec<f64> = (0..nb).map(|j| a[j * nb + i]).collect();
        let theirs: Vec<f64> = (0..nb).map(|j| aref[j * nb + i]).collect();
        let sgn = if ours.iter().zip(&theirs).map(|(p, q)| p * q).sum::<f64>() < 0.0 {
            -1.0
        } else {
            1.0
        };
        let diff = ours
            .iter()
            .zip(&theirs)
            .fold(0.0f64, |m, (p, q)| m.max((p - sgn * q).abs()));
        assert!(diff <= 1e-8 * max_abs(&ours), "column {i}: {diff}");
    }
}

#[test]
fn the_signal_is_smooth_enough_for_the_difference_estimator() {
    // the bias bound of the spec: half the variance of the signal's differences under 2% of the noise's
    for hetero in [false, true] {
        let sy = synth(128, 128, 12, 3, hetero, 5);
        for j in 0..sy.b {
            for (dy, dx) in [(0usize, 1usize), (1, 0)] {
                let mut d = Vec::new();
                for y in 0..sy.h - dy {
                    for x in 0..sy.w - dx {
                        d.push(sy.s[j * sy.h * sy.w + y * sy.w + x] - sy.s[j * sy.h * sy.w + (y + dy) * sy.w + x + dx]);
                    }
                }
                let half = std_of(d.iter().copied()).powi(2) / 2.0;
                assert!(half < 0.02 * sy.sigma[j] * sy.sigma[j], "band {j}: {half}");
            }
        }
    }
}

#[test]
fn eigenvalues_are_one_plus_snr() {
    for (hetero, seed) in [(false, 1), (true, 2)] {
        let sy = synth(128, 128, 12, 3, hetero, seed);
        let pop = population(&sy);
        let tr = fit(&sy, Noise::DiffHv);
        for i in 0..sy.r {
            assert!(
                (tr.eigenvalues[i] / pop[i] - 1.0).abs() < 0.05,
                "diff_hv {i}: {} vs {}",
                tr.eigenvalues[i],
                pop[i]
            );
        }
        for i in sy.r..tr.m {
            assert!(
                (tr.eigenvalues[i] - 1.0).abs() <= 0.15,
                "noise {i}: {}",
                tr.eigenvalues[i]
            );
        }
        let mut given = vec![0.0; 144];
        (0..12).for_each(|j| given[j * 12 + j] = sy.sigma[j] * sy.sigma[j]);
        let tg = fit(&sy, Noise::Given(given));
        for i in 0..sy.r {
            assert!(
                (tg.eigenvalues[i] / pop[i] - 1.0).abs() < 0.02,
                "given {i}: {} vs {}",
                tg.eigenvalues[i],
                pop[i]
            );
        }
    }
}

#[test]
fn the_signal_subspace_is_recovered() {
    for (hetero, seed) in [(false, 3), (true, 4)] {
        let sy = synth(128, 128, 12, 3, hetero, seed);
        let tr = fit(&sy, Noise::DiffHv);
        // P = Sn A: its first r columns span the mixing
        let p: Vec<f64> = (0..sy.b * sy.r)
            .map(|q| tr.inverse[(q / sy.r) * tr.m + q % sy.r])
            .collect();
        let angle = max_angle(&p, &sy.m, sy.b, sy.r);
        assert!(angle < 2.0, "{angle} degrees");
    }
}

#[test]
fn denoising_factor_and_limits() {
    for (hetero, seed) in [(false, 6), (true, 7)] {
        let sy = synth(128, 128, 12, 3, hetero, seed);
        let (b, npix) = (sy.b, sy.h * sy.w);
        let tr = fit(&sy, Noise::DiffHv);
        let mut given = vec![0.0; 144];
        (0..12).for_each(|j| given[j * 12 + j] = sy.sigma[j] * sy.sigma[j]);
        let tg = fit(&sy, Noise::Given(given));
        // prediction: sqrt(trace(Pi Sn) / trace(Sn)), Pi the projector on span(Sn^-1/2 M)
        let mw: Vec<f64> = (0..b * sy.r).map(|i| sy.m[i] / sy.sigma[i / sy.r]).collect();
        let g = mat_mul(&transpose(&mw, b, sy.r), sy.r, b, &mw, sy.r);
        let (_, v) = sym_eig_desc(&g, sy.r).unwrap();
        let q = mat_mul(&mw, b, sy.r, &v, sy.r); // orthogonal columns
        let mut tr_pi_sn = 0.0;
        for c in 0..sy.r {
            let nrm: f64 = (0..b).map(|j| q[j * sy.r + c] * q[j * sy.r + c]).sum();
            tr_pi_sn += (0..b)
                .map(|j| q[j * sy.r + c] * q[j * sy.r + c] * sy.sigma[j] * sy.sigma[j])
                .sum::<f64>()
                / nrm;
        }
        let want = (tr_pi_sn / sy.sigma.iter().map(|s| s * s).sum::<f64>()).sqrt();
        for t in [&tr, &tg] {
            let mut out = vec![0f32; b * npix];
            mnf_denoise_batch(&sy.x, b, npix, t, sy.r, &[], None, &mut out, 2);
            let rmse = |a: &mut dyn Iterator<Item = f64>| (a.map(|v| v * v).sum::<f64>() / (b * npix) as f64).sqrt();
            let e_hat = rmse(&mut (0..b * npix).map(|i| f64::from(out[i]) - sy.s[i]));
            let e_z = rmse(&mut (0..b * npix).map(|i| sy.x[i] - sy.s[i]));
            let ratio = e_hat / e_z;
            assert!((ratio / want - 1.0).abs() < 0.10, "ratio {ratio} vs {want}");
            if !hetero {
                assert!((want - 0.5).abs() < 1e-9, "sqrt(r/B)");
            }
            // k = m is the input, k = 0 the mean
            let mut full = vec![0f32; b * npix];
            mnf_denoise_batch(&sy.x, b, npix, t, t.m, &[], None, &mut full, 2);
            for j in 0..b {
                let sd = std_of(sy.x[j * npix..][..npix].iter().copied());
                let worst = (0..npix).fold(0.0f64, |m, p| {
                    m.max((f64::from(full[j * npix + p]) - sy.x[j * npix + p]).abs())
                });
                assert!(worst <= 1e-5 * sd, "band {j}: {worst}");
            }
            let mut zero = vec![0f32; b * npix];
            mnf_denoise_batch(&sy.x, b, npix, t, 0, &[], None, &mut zero, 2);
            for j in 0..b {
                assert!(zero[j * npix..][..npix].iter().all(|&v| v == t.mean[j] as f32));
            }
            // snr_min picks the components
            let k = t.eigenvalues.iter().filter(|&&l| l - 1.0 >= 5.0).count();
            assert_eq!(t.n_with_snr(5.0), k);
            assert_eq!(t.n_with_snr(-1.0), t.m);
        }
        assert_eq!(tr.n_with_snr(1.0), sy.r, "the signal components have an SNR above 1");
    }
}

#[test]
fn scale_invariance_where_pca_has_none() {
    let sy = synth(128, 128, 12, 3, true, 8);
    let (b, npix) = (sy.b, sy.h * sy.w);
    let tr = fit(&sy, Noise::DiffHv);
    let y0 = forward_all(&sy.x, b, npix, &tr);
    let change = |f: &dyn Fn(usize, f64) -> f64| -> Vec<f64> {
        let mut x = sy.x.clone();
        for j in 0..b {
            for p in 0..npix {
                x[j * npix + p] = f(j, x[j * npix + p]);
            }
        }
        x
    };
    let cases: Vec<(Vec<f64>, f64, f64)> = vec![
        (
            change(&|j, v| match j {
                3 => v * 1024.0,
                5 => v / 1024.0,
                7 => v + 500.0,
                _ => v,
            }),
            1e-10,
            1e-6,
        ),
        (change(&|j, v| if j == 3 { v * 1000.0 } else { v }), 1e-6, 1e-5),
    ];
    for (x2, tol_l, tol_y) in &cases {
        let t2 = fit_with(x2, 1, b, sy.h, sy.w, Noise::DiffHv, 2);
        for i in 0..tr.m {
            assert!(
                (t2.eigenvalues[i] / tr.eigenvalues[i] - 1.0).abs() <= *tol_l,
                "eigenvalue {i}"
            );
        }
        let y2 = forward_all(x2, b, npix, &t2);
        for i in 0..tr.m {
            let sd = std_of(y0[i * npix..][..npix].iter().map(|&v| f64::from(v)));
            let worst = (0..npix).fold(0.0f64, |m, p| {
                m.max((f64::from(y2[i * npix + p]) - f64::from(y0[i * npix + p])).abs())
            });
            assert!(worst <= tol_y * sd, "component {i}: {worst} of {sd}");
        }
    }
    // the sign rule: the loadings a_j scale back to the originals, signs and all
    let t2 = fit_with(&cases[0].0, 1, b, sy.h, sy.w, Noise::DiffHv, 2);
    let k = [1.0, 1.0, 1.0, 1024.0, 1.0, 1.0 / 1024.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    for i in 0..tr.m {
        let back: Vec<f64> = (0..b).map(|j| t2.forward[i * b + j] * k[j]).collect();
        let dot: f64 = back.iter().zip(&tr.forward[i * b..][..b]).map(|(p, q)| p * q).sum();
        assert!(dot > 0.0, "sign of component {i}");
    }
    // the largest |correlation| of each component is positive
    for i in 0..tr.m {
        let col: Vec<f64> = (0..b).map(|j| tr.correlation[j * tr.m + i]).collect();
        let mx = max_abs(&col);
        let j = col.iter().position(|v| v.abs() >= mx * (1.0 - 1e-12)).unwrap();
        assert!(col[j] > 0.0, "component {i}");
    }
    // PCA of the covariance is not invariant
    let (_, c0) = sample_cov(&sy.x, b, npix);
    let (_, c1) = sample_cov(&cases[0].0, b, npix);
    let (_, v0) = sym_eig_desc(&c0, b).unwrap();
    let (_, v1) = sym_eig_desc(&c1, b).unwrap();
    let cosine: f64 = (0..b).map(|j| v0[j * b] * v1[j * b]).sum::<f64>().abs();
    assert!(
        cosine.clamp(0.0, 1.0).acos() > 0.1,
        "the first principal component should change"
    );
}

#[test]
fn a_constant_band_and_a_duplicate_band() {
    let sy = synth(128, 128, 12, 3, false, 9);
    let (b, npix) = (12, 128 * 128);
    let without = |drop: usize| -> Vec<f64> {
        (0..b)
            .filter(|&j| j != drop)
            .flat_map(|j| sy.x[j * npix..][..npix].to_vec())
            .collect()
    };
    // constant
    let mut x = sy.x.clone();
    x[4 * npix..5 * npix].iter_mut().for_each(|v| *v = 42.0);
    let tr = fit_with(&x, 1, b, 128, 128, Noise::DiffHv, 2);
    assert_eq!(tr.dropped, vec![(4, Dropped::Constant)]);
    assert_eq!(tr.m, 11);
    let mut x11 = x.clone();
    x11.drain(4 * npix..5 * npix);
    let t11 = fit_with(&x11, 1, 11, 128, 128, Noise::DiffHv, 2);
    for i in 0..11 {
        assert!((tr.eigenvalues[i] / t11.eigenvalues[i] - 1.0).abs() < 1e-9);
    }
    for k in [0, 3, 11] {
        let mut out = vec![0f32; b * npix];
        mnf_denoise_batch(&x, b, npix, &tr, k, &[], None, &mut out, 2);
        assert!(out[4 * npix..5 * npix].iter().all(|&v| v == 42.0), "k = {k}");
    }
    // duplicate: band 9 = 2 x band 2
    let mut x = sy.x.clone();
    for p in 0..npix {
        x[9 * npix + p] = 2.0 * x[2 * npix + p];
    }
    let td = fit_with(&x, 1, b, 128, 128, Noise::DiffHv, 2);
    let degenerate = (0..td.m).filter(|&i| td.is_degenerate(i)).count();
    assert_eq!(degenerate, 1);
    assert!(td.is_degenerate(td.m - 1) && td.floored[td.m - 1]);
    assert!(td.noise_floored >= 1);
    let t11 = fit_with(&without(9), 1, 11, 128, 128, Noise::DiffHv, 2);
    for i in 0..11 {
        assert!(
            (td.eigenvalues[i] / t11.eigenvalues[i] - 1.0).abs() < 1e-6,
            "eigenvalue {i}"
        );
    }
}

/// The statistics by the definition: the pixels and pairs the test enumerates.
struct Naive {
    n: usize,
    mean: Vec<f64>,
    cov: Vec<f64>,
    m: Vec<usize>,
    noise: Vec<f64>,
}

fn naive(
    x: &[f64],
    b: usize,
    h: usize,
    w: usize,
    valid: &dyn Fn(usize, usize) -> bool,
    inside: &dyn Fn(usize, usize) -> bool,
    dirs: &[(usize, isize)],
) -> Naive {
    let n = h * w;
    let px: Vec<(usize, usize)> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (y, x)))
        .filter(|&(y, x)| valid(y, x))
        .collect();
    let col = |j: usize, (y, xx): (usize, usize)| x[j * n + y * w + xx];
    let mean: Vec<f64> = (0..b)
        .map(|j| px.iter().map(|&p| col(j, p)).sum::<f64>() / px.len() as f64)
        .collect();
    let mut cov = vec![0.0; b * b];
    for i in 0..b {
        for j in 0..b {
            cov[i * b + j] = px
                .iter()
                .map(|&p| (col(i, p) - mean[i]) * (col(j, p) - mean[j]))
                .sum::<f64>()
                / (px.len() - 1) as f64;
        }
    }
    let mut noise = vec![0.0; b * b];
    let mut m = Vec::new();
    let mut total = 0;
    for &(dy, dx) in dirs {
        let pairs: Vec<((usize, usize), (usize, usize))> = px
            .iter()
            .filter_map(|&(y, xx)| {
                let (qy, qx) = (y + dy, xx as isize + dx);
                if qy >= h || qx < 0 || qx as usize >= w {
                    return None;
                }
                let q = (qy, qx as usize);
                (valid(q.0, q.1) && inside(y, xx) && inside(q.0, q.1)).then_some(((y, xx), q))
            })
            .collect();
        let dm: Vec<f64> = (0..b)
            .map(|j| pairs.iter().map(|&(p, q)| col(j, p) - col(j, q)).sum::<f64>() / pairs.len() as f64)
            .collect();
        for i in 0..b {
            for j in 0..b {
                noise[i * b + j] += pairs
                    .iter()
                    .map(|&(p, q)| (col(i, p) - col(i, q) - dm[i]) * (col(j, p) - col(j, q) - dm[j]))
                    .sum::<f64>();
            }
        }
        total += pairs.len();
        m.push(pairs.len());
    }
    noise.iter_mut().for_each(|v| *v /= 2.0 * (total - dirs.len()) as f64);
    Naive {
        n: px.len(),
        mean,
        cov,
        m,
        noise,
    }
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    max_abs_diff(a, b) <= tol * max_abs(b)
}

#[test]
fn missing_values_follow_the_definition() {
    let sy = synth(128, 128, 12, 3, false, 10);
    let (b, h, w) = (12, 128, 128);
    let npix = h * w;
    let mut x = sy.x.clone();
    let mut rng = Rng(77);
    for p in 0..npix {
        if rng.u() < 0.05 {
            x[6 * npix + p] = f64::NAN;
        }
    }
    for j in 0..b {
        for y in 40..60 {
            for xx in 70..90 {
                x[j * npix + y * w + xx] = -9999.0;
            }
        }
    }
    let mut mask = vec![1u8; npix];
    for y in 0..h {
        mask[y * w + 100] = 0;
    }
    let ok = |y: usize, xx: usize| -> bool {
        mask[y * w + xx] != 0
            && (0..b).all(|j| x[j * npix + y * w + xx].is_finite() && x[j * npix + y * w + xx] != -9999.0)
    };
    let mut opts = StatsOptions::new(Noise::DiffHv);
    opts.missing = &[-9999.0];
    opts.mask = Some(&mask);
    let stats = mnf_stats_batch(&x, 1, b, h, w, &opts, 2).unwrap();
    let want = naive(&x, b, h, w, &ok, &|_, _| true, &[(0, 1), (1, 0)]);
    assert_eq!(stats.n as usize, want.n);
    assert_eq!(stats.pairs.iter().map(|d| d.m as usize).collect::<Vec<_>>(), want.m);
    let tr = mnf_fit(&stats).unwrap();
    assert!(close(&tr.mean, &want.mean, 1e-12));
    assert!(close(&tr.cov, &want.cov, 1e-12));
    assert!(close(&tr.noise_cov, &want.noise, 1e-12));
    let mut out = vec![0f32; tr.m * npix];
    mnf_forward_batch(
        &x,
        b,
        npix,
        &tr,
        tr.m,
        Scale::Noise,
        &[-9999.0],
        Some(&mask),
        &mut out,
        2,
    );
    for p in 0..npix {
        let valid = ok(p / w, p % w);
        for i in 0..tr.m {
            assert_eq!(out[i * npix + p].is_nan(), !valid, "pixel {p}");
        }
    }
}

#[test]
fn blocks_halo_and_threads() {
    let sy = synth(128, 128, 12, 3, false, 12);
    let (b, h, w) = (12, 128, 128);
    let opts = StatsOptions::new(Noise::DiffHv);
    let whole = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 128, 1).unwrap();
    // by hand, blocks of 7 rows with their halo row
    let mut merged = MnfStats::new(b, &opts.noise);
    for r0 in (0..h).step_by(7) {
        let rows = 7.min(h - r0);
        let blk: Vec<f64> = (0..b)
            .flat_map(|j| sy.x[(j * h + r0) * w..][..rows * w].to_vec())
            .collect();
        let halo: Option<Vec<f64>> = (r0 + rows < h).then(|| {
            (0..b)
                .flat_map(|j| sy.x[(j * h + r0 + rows) * w..][..w].to_vec())
                .collect()
        });
        merged.add_block(&blk, halo.as_deref(), 0, r0, rows, h, w, &opts);
    }
    assert_eq!(merged.n, whole.n);
    for (a, c) in merged.pairs.iter().zip(&whole.pairs) {
        assert_eq!(a.m, c.m);
        assert!(close(&a.m2, &c.m2, 1e-12));
    }
    assert!(close(&merged.m2, &whole.m2, 1e-12));
    let scale = (max_abs(&whole.m2) / whole.n as f64).sqrt();
    assert!(max_abs_diff(&merged.mean, &whole.mean) <= 1e-12 * scale, "means");
    // the diagonal pairs and the lower-left ones cross blocks too
    for noise in [Noise::DiffDiag, Noise::DiffAntidiag] {
        let o = StatsOptions::new(noise);
        let a = mnf_stats_blocks(&sy.x, 1, b, h, w, &o, 7, 1).unwrap();
        let c = mnf_stats_blocks(&sy.x, 1, b, h, w, &o, 128, 1).unwrap();
        assert_eq!(a.pairs[0].m, c.pairs[0].m);
        assert!(close(&a.pairs[0].m2, &c.pairs[0].m2, 1e-12));
    }
    // threads and runs: bit-identical
    let t1 = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 7, 1).unwrap();
    let t8 = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 7, 8).unwrap();
    let t8b = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 7, 8).unwrap();
    assert_eq!(t1, t8);
    assert_eq!(t8, t8b);
    // the transforms too
    let tr = mnf_fit(&t1).unwrap();
    let npix = h * w;
    let (mut o1, mut o8) = (vec![0f32; tr.m * npix], vec![0f32; tr.m * npix]);
    mnf_forward_batch(&sy.x, b, npix, &tr, tr.m, Scale::Noise, &[], None, &mut o1, 1);
    mnf_forward_batch(&sy.x, b, npix, &tr, tr.m, Scale::Noise, &[], None, &mut o8, 8);
    assert!(o1.iter().zip(&o8).all(|(a, c)| a.to_bits() == c.to_bits()));
}

#[test]
fn subsampling_is_deterministic_and_independent_of_the_blocks() {
    let sy = synth(128, 128, 12, 3, false, 13);
    let (b, h, w) = (12, 128, 128);
    let mut opts = StatsOptions::new(Noise::DiffHv);
    opts.sample_fraction = 0.25;
    opts.seed = 1;
    let a = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 128, 2).unwrap();
    let again = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 128, 2).unwrap();
    assert_eq!(a, again);
    let small = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 9, 2).unwrap();
    assert_eq!(a.n, small.n);
    assert_eq!(a.pairs[0].m, small.pairs[0].m);
    assert_eq!(a.pairs[1].m, small.pairs[1].m);
    assert!((a.n as f64 / (h * w) as f64 - 0.25).abs() < 0.02);
    opts.seed = 2;
    let other = mnf_stats_blocks(&sy.x, 1, b, h, w, &opts, 128, 2).unwrap();
    assert!(other.n != a.n || other.mean != a.mean);
    // the eigenvalues are still those of the population (a looser bound: a quarter of the pixels)
    opts.seed = 1;
    let tr = mnf_fit(&a).unwrap();
    let pop = population(&sy);
    for i in 0..sy.r {
        assert!(
            (tr.eigenvalues[i] / pop[i] - 1.0).abs() < 0.08,
            "{i}: {} vs {}",
            tr.eigenvalues[i],
            pop[i]
        );
    }
    for i in sy.r..tr.m {
        assert!((tr.eigenvalues[i] - 1.0).abs() <= 0.3);
    }
}

#[test]
fn the_maf_identity() {
    // with the horizontal differences, 1 - Var(Delta Y_i) / (2 Var Y_i) = 1 - 1/lambda_i exactly
    let sy = synth(128, 128, 12, 3, true, 14);
    let (b, h, w) = (12, 128, 128);
    let npix = h * w;
    let tr = fit(&sy, Noise::DiffH);
    let y = |i: usize, p: usize| -> f64 {
        (0..b)
            .map(|j| tr.forward[i * b + j] * (sy.x[j * npix + p] - tr.mean[j]))
            .sum()
    };
    for i in 0..tr.m {
        let all: Vec<f64> = (0..npix).map(|p| y(i, p)).collect();
        let var = std_of(all.iter().copied()).powi(2);
        let d: Vec<f64> = (0..h)
            .flat_map(|r| (0..w - 1).map(move |c| (r, c)))
            .map(|(r, c)| all[r * w + c] - all[r * w + c + 1])
            .collect();
        let vd = std_of(d.iter().copied()).powi(2);
        let lhs = 1.0 - vd / (2.0 * var);
        let rhs = 1.0 - 1.0 / tr.eigenvalues[i];
        assert!((lhs - rhs).abs() < 1e-9, "component {i}: {lhs} vs {rhs}");
    }
}

#[test]
fn a_transform_applies_to_another_date_and_equals_the_stats_route() {
    let a = synth(128, 128, 12, 3, false, 15);
    let c = synth(128, 128, 12, 3, false, 16);
    let (b, npix) = (12, 128 * 128);
    let tr = fit(&a, Noise::DiffHv);
    let mut out = vec![0f32; 4 * npix];
    mnf_forward_batch(&c.x, b, npix, &tr, 4, Scale::Noise, &[], None, &mut out, 2);
    for p in (0..npix).step_by(97) {
        for i in 0..4 {
            let hand: f64 = tr
                .kept
                .iter()
                .map(|&j| tr.forward[i * b + j] * (c.x[j * npix + p] - tr.mean[j]))
                .sum();
            assert_eq!(out[i * npix + p], hand as f32);
        }
    }
    // the unit scale divides by the square root of the eigenvalue
    let mut unit = vec![0f32; 4 * npix];
    mnf_forward_batch(&c.x, b, npix, &tr, 4, Scale::Unit, &[], None, &mut unit, 2);
    for i in 0..4 {
        let want = f64::from(out[i * npix + 5]) / tr.eigenvalues[i].sqrt();
        assert!((f64::from(unit[i * npix + 5]) - want).abs() < 1e-5 * want.abs().max(1.0));
    }
    // inverse of the components = the denoised bands
    let mut y = vec![0f32; 3 * npix];
    mnf_forward_batch(&c.x, b, npix, &tr, 3, Scale::Noise, &[], None, &mut y, 2);
    let (mut inv, mut den) = (vec![0f32; b * npix], vec![0f32; b * npix]);
    mnf_inverse_batch(&y, 3, npix, &tr, 3, Scale::Noise, &mut inv, 2);
    mnf_denoise_batch(&c.x, b, npix, &tr, 3, &[], None, &mut den, 2);
    let sd = std_of(c.x[..npix].iter().copied());
    assert!((0..b * npix).all(|q| (f64::from(inv[q]) - f64::from(den[q])).abs() < 1e-4 * sd));
    // fitting the same covariances gives the same transform
    let t2 = mnf_fit_cov(&tr.mean, &tr.cov, &tr.noise_cov, b).unwrap();
    assert_eq!(t2.eigenvalues, tr.eigenvalues);
    assert_eq!(t2.forward, tr.forward);
    assert_eq!(t2.inverse, tr.inverse);
}

#[test]
fn a_noise_region_and_its_pairs() {
    let sy = synth(128, 128, 12, 3, false, 17);
    let (b, h, w) = (12, 128, 128);
    let region: Vec<u8> = (0..h * w)
        .map(|p| u8::from((30..70).contains(&(p / w)) && (20..60).contains(&(p % w))))
        .collect();
    let mut opts = StatsOptions::new(Noise::DiffHv);
    opts.noise_region = Some(&region);
    let s = mnf_stats_batch(&sy.x, 1, b, h, w, &opts, 2).unwrap();
    assert_eq!(
        s.n as usize,
        h * w,
        "the signal statistics are those of the whole image"
    );
    assert_eq!(s.pairs[0].m, 40 * 39);
    assert_eq!(s.pairs[1].m, 39 * 40);
    let inside = |y: usize, x: usize| (30..70).contains(&y) && (20..60).contains(&x);
    let want = naive(&sy.x, b, h, w, &|_, _| true, &inside, &[(0, 1), (1, 0)]);
    let tr = mnf_fit(&s).unwrap();
    assert!(close(&tr.noise_cov, &want.noise, 1e-12));
    // a region too small
    let tiny: Vec<u8> = (0..h * w)
        .map(|p| u8::from((30..33).contains(&(p / w)) && (20..23).contains(&(p % w))))
        .collect();
    opts.noise_region = Some(&tiny);
    let s = mnf_stats_batch(&sy.x, 1, b, h, w, &opts, 2).unwrap();
    match mnf_fit(&s) {
        Err(e @ MnfError::TooFewPairs { .. }) => {
            assert_eq!(
                e,
                MnfError::TooFewPairs {
                    direction: "h",
                    m: 6,
                    need: 13
                }
            );
            let text = e.to_string();
            assert!(text.contains('6') && text.contains("13"), "{text}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn errors() {
    let sy = synth(128, 128, 12, 3, false, 18);
    let opts = StatsOptions::new(Noise::DiffHv);
    // fewer pixels than bands + 1
    let few: Vec<f64> = (0..12).flat_map(|j| sy.x[j * 128 * 128..][..9].to_vec()).collect();
    let s = mnf_stats_batch(&few, 1, 12, 3, 3, &opts, 1).unwrap();
    assert!(matches!(mnf_fit(&s), Err(MnfError::TooFewPixels { n: 9, need: 13 })));
    // no valid pixel
    let nan = vec![f64::NAN; 12 * 16 * 16];
    let s = mnf_stats_batch(&nan, 1, 12, 16, 16, &opts, 1).unwrap();
    assert_eq!(mnf_fit(&s).unwrap_err(), MnfError::NoValidPixel);
    // every band constant
    let flat = vec![3.0; 12 * 16 * 16];
    let s = mnf_stats_batch(&flat, 1, 12, 16, 16, &opts, 1).unwrap();
    assert_eq!(mnf_fit(&s).unwrap_err(), MnfError::AllBandsConstant);
    // a noise covariance that is not symmetric, not 12 x 12, or has a negative variance
    let mut bad = vec![0.0; 144];
    (0..12).for_each(|j| bad[j * 12 + j] = 1.0);
    bad[1] = 0.5;
    let given = |g: Vec<f64>| mnf_stats_batch(&sy.x, 1, 12, 128, 128, &StatsOptions::new(Noise::Given(g)), 1);
    assert!(matches!(given(bad.clone()), Err(MnfError::BadNoiseCov(_))));
    assert!(matches!(given(vec![1.0; 100]), Err(MnfError::BadNoiseCov(_))));
    bad[1] = 0.0;
    bad[0] = -1.0;
    assert!(matches!(given(bad), Err(MnfError::BadNoiseCov(_))));
    assert!(matches!(
        mnf_fit_cov(&[0.0; 12], &[1.0; 144], &[f64::NAN; 144], 12),
        Err(MnfError::BadNoiseCov(_))
    ));
    // no noise at all
    let zero = vec![0.0; 144];
    let g = StatsOptions::new(Noise::Given(zero));
    assert_eq!(
        mnf_fit(&mnf_stats_batch(&sy.x, 1, 12, 128, 128, &g, 1).unwrap()).unwrap_err(),
        MnfError::NoNoise
    );
    // a bad subsample fraction
    let mut o = StatsOptions::new(Noise::DiffH);
    o.sample_fraction = 0.0;
    assert!(matches!(
        mnf_stats_batch(&sy.x, 1, 12, 128, 128, &o, 1),
        Err(MnfError::BadInput(_))
    ));
}

#[test]
fn a_band_with_no_valid_value_is_left_out() {
    let sy = synth(64, 64, 6, 2, false, 19);
    let (b, npix) = (6, 64 * 64);
    let mut x = sy.x.clone();
    x[2 * npix..3 * npix].iter_mut().for_each(|v| *v = f64::NAN);
    let opts = StatsOptions::new(Noise::DiffHv);
    let s = mnf_stats_batch(&x, 1, b, 64, 64, &opts, 1).unwrap();
    assert_eq!(s.band_valid[2], 0);
    assert_eq!(s.n as usize, npix);
    let tr = mnf_fit(&s).unwrap();
    assert_eq!(tr.dropped, vec![(2, Dropped::Empty)]);
    assert_eq!(tr.m, 5);
    let mut out = vec![0f32; b * npix];
    mnf_denoise_batch(&x, b, npix, &tr, 2, &[], None, &mut out, 1);
    assert!(out[2 * npix..3 * npix].iter().all(|v| v.is_nan()));
    assert!(out[..npix].iter().all(|v| v.is_finite()));
}

#[test]
fn a_series_pools_dates_without_crossing_them() {
    let dates: Vec<Synth> = (0..3).map(|t| synth(64, 64, 12, 3, false, 20 + t)).collect();
    let (b, h, w) = (12, 64, 64);
    let x: Vec<f64> = dates.iter().flat_map(|d| d.x.clone()).collect();
    let opts = StatsOptions::new(Noise::DiffHv);
    let pooled = mnf_stats_batch(&x, 3, b, h, w, &opts, 2).unwrap();
    assert_eq!(pooled.n as usize, 3 * h * w);
    assert_eq!(pooled.pairs[0].m as usize, 3 * h * (w - 1));
    assert_eq!(pooled.pairs[1].m as usize, 3 * (h - 1) * w);
    // = the dates' statistics merged in order
    let mut merged = MnfStats::new(b, &opts.noise);
    for (t, d) in dates.iter().enumerate() {
        let mut o = StatsOptions::new(Noise::DiffHv);
        o.date0 = t;
        merged.merge(&mnf_stats_batch(&d.x, 1, b, h, w, &o, 1).unwrap());
    }
    assert_eq!(merged.n, pooled.n);
    assert!(close(&merged.m2, &pooled.m2, 1e-12));
    assert!(close(&merged.pairs[1].m2, &pooled.pairs[1].m2, 1e-12));
    // one transform, applied to the three dates
    let tr = mnf_fit(&pooled).unwrap();
    let mut out = vec![0f32; 3 * 2 * h * w];
    mnf_forward_batch(&x, b, h * w, &tr, 2, Scale::Noise, &[], None, &mut out, 2);
    assert_eq!(out.len(), 3 * 2 * h * w);
    // each date fitted alone is its own transform
    let alone = fit_with(&dates[1].x, 1, b, h, w, Noise::DiffHv, 1);
    assert_ne!(alone.eigenvalues, tr.eigenvalues);
    // sampling is keyed by date, so an offset date differs from date 0 when subsampled
    let mut o = StatsOptions::new(Noise::DiffHv);
    o.sample_fraction = 0.5;
    let s0 = mnf_stats_batch(&dates[0].x, 1, b, h, w, &o, 1).unwrap();
    o.date0 = 1;
    let s1 = mnf_stats_batch(&dates[0].x, 1, b, h, w, &o, 1).unwrap();
    assert_ne!(s0.n, s1.n);
}
