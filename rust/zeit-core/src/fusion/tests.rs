//! Tests of the fusion module, on data generated here.

use super::estarfm::{estarfm, pearson, similar_weights, slope, temporal_weights};
use super::reference::{estarfm_pixel, starfm_pixel};
use super::starfm::starfm;
use super::*;

const NB: usize = 3;

/// A deterministic uniform number in [0, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / (1u64 << 31) as f64
    }
    fn normal(&mut self) -> f64 {
        let (a, b) = (self.next().max(1e-12), self.next());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

/// Class of each pixel: 0 soil, 1 water (disc), 2 vegetation (rectangle), 3 vegetation 2 (disc).
fn classes(nx: usize, ny: usize) -> Vec<u8> {
    let mut c = vec![0u8; nx * ny];
    let s = nx as f64 / 96.0;
    for y in 0..ny {
        for x in 0..nx {
            let (fx, fy) = (x as f64 / s, y as f64 / s);
            c[y * nx + x] = if (fx - 30.0).powi(2) + (fy - 30.0).powi(2) < 14.0f64.powi(2) {
                1
            } else if (50.0..90.0).contains(&fx) && (10.0..60.0).contains(&fy) {
                2
            } else if (fx - 30.0).powi(2) + (fy - 70.0).powi(2) < 18.0f64.powi(2) {
                3
            } else {
                0
            };
        }
    }
    c
}

const CLASS_VALUES: [[f64; NB]; 4] = [
    [0.10, 0.18, 0.25],
    [0.03, 0.02, 0.01],
    [0.06, 0.05, 0.35],
    [0.08, 0.07, 0.28],
];

/// A fine image `(band, pixel)` from the class map, with fixed texture noise.
fn scene(nx: usize, ny: usize, seed: u64, noise: f64) -> Vec<f32> {
    let cls = classes(nx, ny);
    let mut rng = Lcg(seed);
    let mut out = vec![0f32; NB * nx * ny];
    for b in 0..NB {
        for i in 0..nx * ny {
            out[b * nx * ny + i] = (CLASS_VALUES[cls[i] as usize][b] + noise * rng.normal()) as f32;
        }
    }
    out
}

/// Block mean of `(band, pixel)` over `k x k` blocks, `(band, cell)`; NaN-aware (NaN only if all NaN).
fn block_mean(img: &[f32], nx: usize, ny: usize, k: usize) -> Vec<f32> {
    let (cx, cy) = (nx / k, ny / k);
    let mut out = vec![f32::NAN; NB * cx * cy];
    for b in 0..NB {
        for j in 0..cy {
            for i in 0..cx {
                let (mut s, mut n) = (0.0f64, 0.0f64);
                for y in j * k..(j + 1) * k {
                    for x in i * k..(i + 1) * k {
                        let v = img[b * nx * ny + y * nx + x];
                        if v.is_finite() {
                            s += f64::from(v);
                            n += 1.0;
                        }
                    }
                }
                if n > 0.0 {
                    out[b * cx * cy + j * cx + i] = (s / n) as f32;
                }
            }
        }
    }
    out
}

fn cat(parts: &[&[f32]]) -> Vec<f32> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn shifted(img: &[f32], d: f32) -> Vec<f32> {
    img.iter().map(|v| v + d).collect()
}

fn affine(k: usize, ox: f64, oy: f64) -> [f64; 6] {
    let s = 1.0 / k as f64;
    [ox * s, s, 0.0, oy * s, 0.0, s]
}

fn plan_for(pairs: &[(usize, usize)], coarse: usize) -> Plan {
    Plan {
        targets: vec![Target {
            day: 0,
            coarse,
            pairs: pairs.to_vec(),
            observed: None,
            extrapolated: false,
        }],
        ..Default::default()
    }
}

fn whole(nx: usize, ny: usize) -> Region {
    Region { x0: 0, y0: 0, nx, ny }
}

fn max_err(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
        .fold(0.0, f64::max)
}

struct Opts {
    resampling: Resampling,
    weighting: Weighting,
    filter: Filter,
    pure: Pure,
}

fn all_opts() -> Vec<Opts> {
    let mut v = Vec::new();
    for resampling in [Resampling::Nearest, Resampling::Bilinear] {
        for weighting in [Weighting::Direct, Weighting::Log] {
            for filter in [Filter::Both, Filter::Either, Filter::None] {
                for pure in [Pure::Epsilon, Pure::Tiers, Pure::Cells] {
                    v.push(Opts {
                        resampling,
                        weighting,
                        filter,
                        pure,
                    });
                }
            }
        }
    }
    v
}

fn sp(r: usize, o: &Opts) -> StarfmParams {
    let mut p = StarfmParams::new(r);
    p.common.resampling = o.resampling;
    p.weighting = o.weighting;
    p.filter = o.filter;
    p.pure = o.pure;
    p
}

fn run_starfm(
    fine: &[f32],
    coarse: &[f32],
    nx: usize,
    ny: usize,
    cells: (usize, usize),
    geom: CoarseGeom,
    plan: &Plan,
    p: &StarfmParams,
    region: &Region,
    n_jobs: i32,
) -> (Vec<f32>, Vec<u16>) {
    let f = Fine {
        nx,
        ny,
        nb: NB,
        values: fine,
    };
    let c = Coarse {
        ncx: cells.0,
        ncy: cells.1,
        nb: NB,
        values: coarse,
        geom,
    };
    let sigma = image_sigma(&f, &whole(nx, ny), 1);
    let nt = plan.targets.len();
    let mut out = vec![0f32; nt * NB * region.nx * region.ny];
    let mut q = vec![0u16; nt * 2 * region.nx * region.ny];
    starfm(&f, &c, plan, &sigma, p, region, &mut out, Some(&mut q), n_jobs).unwrap();
    (out, q)
}

fn run_estarfm(
    fine: &[f32],
    coarse: &[f32],
    nx: usize,
    ny: usize,
    cells: (usize, usize),
    geom: CoarseGeom,
    plan: &Plan,
    p: &EstarfmParams,
    region: &Region,
    n_jobs: i32,
) -> (Vec<f32>, Vec<u16>) {
    let f = Fine {
        nx,
        ny,
        nb: NB,
        values: fine,
    };
    let c = Coarse {
        ncx: cells.0,
        ncy: cells.1,
        nb: NB,
        values: coarse,
        geom,
    };
    let sigma = image_sigma(&f, &whole(nx, ny), 1);
    let nt = plan.targets.len();
    let mut out = vec![0f32; nt * NB * region.nx * region.ny];
    let mut q = vec![0u16; nt * 2 * region.nx * region.ny];
    estarfm(&f, &c, plan, &sigma, p, region, &mut out, Some(&mut q), n_jobs).unwrap();
    (out, q)
}

#[test]
fn starfm_homogeneous_change_is_exact() {
    let (nx, ny) = (96, 96);
    let f1 = scene(nx, ny, 1, 0.0);
    let f2 = shifted(&f1, 0.04);
    let f3 = shifted(&f1, 0.08);
    let (c1, c2, c3) = (
        block_mean(&f1, nx, ny, 16),
        block_mean(&f2, nx, ny, 16),
        block_mean(&f3, nx, ny, 16),
    );
    let fine = cat(&[&f1, &f3]);
    let coarse = cat(&[&c1, &c2, &c3]);
    for o in all_opts() {
        let p = sp(8, &o);
        // One pair, predict t2 from t1.
        let (out, q) = run_starfm(
            &fine,
            &coarse,
            nx,
            ny,
            (6, 6),
            CoarseGeom::Affine(affine(16, 0.0, 0.0)),
            &plan_for(&[(0, 0)], 1),
            &p,
            &whole(nx, ny),
            1,
        );
        assert!(max_err(&out, &f2) <= 1e-6, "one pair {:?} {:?}", o.resampling, o.pure);
        assert!(q[nx * ny..].iter().all(|&s| s == SRC_TWO_PAIRS));
        // Two pairs.
        let (out, _) = run_starfm(
            &fine,
            &coarse,
            nx,
            ny,
            (6, 6),
            CoarseGeom::Affine(affine(16, 0.0, 0.0)),
            &plan_for(&[(0, 0), (1, 2)], 1),
            &p,
            &whole(nx, ny),
            1,
        );
        assert!(max_err(&out, &f2) <= 1e-6, "two pairs {:?} {:?}", o.resampling, o.pure);
    }
}

#[test]
fn estarfm_homogeneous_change_is_exact() {
    let (nx, ny) = (96, 96);
    let f1 = scene(nx, ny, 2, 0.004);
    let f2 = shifted(&f1, 0.04);
    let f3 = shifted(&f1, 0.08);
    let (c1, c2, c3) = (
        block_mean(&f1, nx, ny, 16),
        block_mean(&f2, nx, ny, 16),
        block_mean(&f3, nx, ny, 16),
    );
    let fine = cat(&[&f1, &f3]);
    let coarse = cat(&[&c1, &c2, &c3]);
    for how in [Resampling::Nearest, Resampling::Bilinear] {
        for vs in [VScope::Window, VScope::Cell] {
            let mut p = EstarfmParams::new(8);
            p.common.resampling = how;
            p.v_scope = vs;
            let (out, _) = run_estarfm(
                &fine,
                &coarse,
                nx,
                ny,
                (6, 6),
                CoarseGeom::Affine(affine(16, 0.0, 0.0)),
                &plan_for(&[(0, 0), (1, 2)], 1),
                &p,
                &whole(nx, ny),
                1,
            );
            assert!(max_err(&out, &f2) <= 1e-5, "{how:?} {vs:?}: {}", max_err(&out, &f2));
        }
    }
}

#[test]
fn regression_helpers() {
    // A perfect line of slope 1.3.
    let (mut n, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..10 {
        let x = 0.1 + 0.03 * i as f64;
        let y = 1.3 * x + 0.02;
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
    }
    let (v, built) = slope(n, sx, sy, sxx, sxy, syy, 0.05);
    assert!(built && (v - 1.3).abs() < 1e-9);
    // Unrelated points: not significant, so 1.
    let mut rng = Lcg(5);
    let pts: Vec<(f64, f64)> = (0..12).map(|_| (rng.next(), rng.next())).collect();
    let s = |f: &dyn Fn(&(f64, f64)) -> f64| pts.iter().map(f).sum::<f64>();
    let (v, built) = slope(
        12.0,
        s(&|p| p.0),
        s(&|p| p.1),
        s(&|p| p.0 * p.0),
        s(&|p| p.0 * p.1),
        s(&|p| p.1 * p.1),
        1e-6,
    );
    assert!(!built && v == 1.0);
    // Two points per similar pixel and fewer than three data points: 1.
    assert_eq!(slope(2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05), (1.0, false));
    // Closed form: one class, coarse values c_i varying by cell, a change Delta in half of the cell's pixels:
    // slope = (Delta^2 / 4) / (var(c) + Delta^2 / 4), where the points are (c, f) and (c + Delta / 2, f + Delta).
    // Here fine = c exactly (x = f) on date 1 and x = c + Delta / 2, y = c + Delta on date 2.
    let delta = 0.1f64;
    let cs: Vec<f64> = (0..20).map(|i| 0.1 + 0.02 * i as f64).collect();
    let (mut n, mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &c in &cs {
        for (x, y) in [(c, c), (c + delta / 2.0, c + delta)] {
            n += 1.0;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
        }
    }
    let (v, _) = slope(n, sx, sy, sxx, sxy, syy, 1.0);
    let mean = cs.iter().sum::<f64>() / cs.len() as f64;
    let var = cs.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / cs.len() as f64;
    let _ = var;
    // The regression is of y on x over 2n points: slope = cov / var with both dates pooled.
    let (xs, ys): (Vec<f64>, Vec<f64>) = cs.iter().flat_map(|&c| [(c, c), (c + delta / 2.0, c + delta)]).unzip();
    let (mx, my) = (xs.iter().sum::<f64>() / 40.0, ys.iter().sum::<f64>() / 40.0);
    let cov: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let vx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    assert!((v - cov / vx).abs() < 1e-9);
    assert!(v > 1.0, "a change makes the slope exceed 1: {v}");
}

#[test]
fn weight_helpers() {
    assert_eq!(temporal_weights(0.0, 0.0), (0.5, 0.5));
    assert_eq!(temporal_weights(0.0, 2.0), (1.0, 0.0));
    let (a, b) = temporal_weights(1.0, 3.0);
    assert!((a - 0.75).abs() < 1e-12 && (b - 0.25).abs() < 1e-12);
    // Perfect correlation: weights 1 / P on those.
    let w = similar_weights(&[1.0, 0.2, 1.0], &[1.0, 1.0, 2.0]);
    assert_eq!(w, vec![0.5, 0.0, 0.5]);
    let w = similar_weights(&[0.5, 0.0], &[1.0, 1.0]);
    assert!((w[0] - 2.0 / 3.0).abs() < 1e-12);
    assert!((pearson(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-12);
    assert_eq!(pearson(&[1.0, 1.0], &[1.0, 2.0]), 0.0);
}

/// A textured two-pair scene with a different change in each class, and its coarse images.
struct Scene {
    nx: usize,
    ny: usize,
    fine: Vec<f32>,
    coarse: Vec<f32>,
    ncx: usize,
}

fn changing_scene(nx: usize, ny: usize, seed: u64) -> Scene {
    let f1 = scene(nx, ny, seed, 0.01);
    let cls = classes(nx, ny);
    let mut rng = Lcg(seed + 100);
    let grow = |scale: f64, f: &[f32]| -> Vec<f32> {
        let mut r = Lcg(seed + 7);
        f.iter()
            .enumerate()
            .map(|(i, &v)| {
                let c = cls[i % (nx * ny)] as usize;
                let d = [0.0, 0.0, 0.12, 0.05][c];
                v + (scale * d + 0.003 * r.normal()) as f32
            })
            .collect()
    };
    let f2 = grow(0.6, &f1);
    let f3 = grow(1.0, &f1);
    let k = 8;
    let mut noisy = |img: &[f32]| -> Vec<f32> {
        block_mean(img, nx, ny, k)
            .iter()
            .map(|v| v * 1.02 + 0.004 + 0.002 * rng.normal() as f32)
            .collect()
    };
    let c1 = noisy(&f1);
    let c2 = noisy(&f2);
    let c3 = noisy(&f3);
    Scene {
        nx,
        ny,
        fine: cat(&[&f1, &f3]),
        coarse: cat(&[&c1, &c2, &c3]),
        ncx: nx / k,
    }
}

fn sample_pixels(nx: usize, ny: usize, n: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = Lcg(seed);
    let mut v = vec![(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1), (nx / 2, ny / 2)];
    for _ in 0..n {
        v.push(((rng.next() * nx as f64) as usize, (rng.next() * ny as f64) as usize));
    }
    v
}

#[test]
fn starfm_matches_the_reference() {
    let s = changing_scene(48, 48, 3);
    let mut fine = s.fine.clone();
    // A cloud hole in pair 1 and some NaN in the coarse image of pair 2.
    for b in 0..NB {
        for y in 10..18 {
            for x in 20..30 {
                fine[b * 48 * 48 + y * 48 + x] = f32::NAN;
            }
        }
    }
    let mut coarse = s.coarse.clone();
    coarse[2 * NB * 36 + 7] = f32::NAN;
    let plan = plan_for(&[(0, 0), (1, 2)], 1);
    let pixels = sample_pixels(48, 48, 25, 11);
    for (i, o) in all_opts().iter().enumerate() {
        for sigma in [SigmaScope::Image, SigmaScope::Window] {
            if sigma == SigmaScope::Window && i % 3 != 0 {
                continue;
            }
            let mut p = sp(5, o);
            p.common.sigma = sigma;
            p.common.fallback = Fallback::Coarse;
            p.classes = 6.0;
            p.spatial_scale_px = 3.5;
            let (out, q) = run_starfm(
                &fine,
                &coarse,
                48,
                48,
                (6, 6),
                CoarseGeom::Affine(affine(8, 0.0, 0.0)),
                &plan,
                &p,
                &whole(48, 48),
                1,
            );
            let f = Fine {
                nx: 48,
                ny: 48,
                nb: NB,
                values: &fine,
            };
            let c = Coarse {
                ncx: 6,
                ncy: 6,
                nb: NB,
                values: &coarse,
                geom: CoarseGeom::Affine(affine(8, 0.0, 0.0)),
            };
            let sig = image_sigma(&f, &whole(48, 48), 1);
            for &(x, y) in &pixels {
                let mut nmin = u32::MAX;
                let mut src = 0;
                for b in 0..NB {
                    let (v, n, sr) = starfm_pixel(&f, &c, &plan.targets[0], b, x, y, &sig, &p);
                    let got = f64::from(out[b * 48 * 48 + y * 48 + x]);
                    assert!(
                        (v.is_nan() && got.is_nan()) || (v - got).abs() <= 1e-6,
                        "{o_i} {x},{y},{b}: {v} vs {got}",
                        o_i = i
                    );
                    nmin = nmin.min(n);
                    if sr != SRC_MISSING {
                        src = src.max(sr);
                    }
                }
                assert_eq!(u32::from(q[y * 48 + x]), nmin);
                assert_eq!(q[48 * 48 + y * 48 + x], src);
            }
        }
    }
}

#[test]
fn estarfm_matches_the_reference() {
    let s = changing_scene(48, 48, 4);
    let mut fine = s.fine.clone();
    for b in 0..NB {
        for y in 10..18 {
            for x in 20..30 {
                fine[b * 48 * 48 + y * 48 + x] = f32::NAN; // pair 1 hole: single-pair fallback
            }
        }
        for y in 30..34 {
            for x in 30..36 {
                fine[(NB + b) * 48 * 48 + y * 48 + x] = f32::NAN;
            }
        }
    }
    let plan = plan_for(&[(0, 0), (1, 2)], 1);
    let pixels = sample_pixels(48, 48, 25, 12);
    for how in [Resampling::Nearest, Resampling::Bilinear] {
        for vs in [VScope::Window, VScope::Cell] {
            for sigma in [SigmaScope::Image, SigmaScope::Window] {
                let mut p = EstarfmParams::new(5);
                p.common.resampling = how;
                p.common.sigma = sigma;
                p.v_scope = vs;
                p.classes = 3.0;
                let (out, q) = run_estarfm(
                    &fine,
                    &s.coarse,
                    48,
                    48,
                    (6, 6),
                    CoarseGeom::Affine(affine(8, 0.3, -0.2)),
                    &plan,
                    &p,
                    &whole(48, 48),
                    1,
                );
                let f = Fine {
                    nx: 48,
                    ny: 48,
                    nb: NB,
                    values: &fine,
                };
                let c = Coarse {
                    ncx: 6,
                    ncy: 6,
                    nb: NB,
                    values: &s.coarse,
                    geom: CoarseGeom::Affine(affine(8, 0.3, -0.2)),
                };
                let sig = image_sigma(&f, &whole(48, 48), 1);
                for &(x, y) in &pixels {
                    let (v, n, src) = estarfm_pixel(&f, &c, &plan.targets[0], x, y, &sig, &p);
                    for b in 0..NB {
                        let got = f64::from(out[b * 48 * 48 + y * 48 + x]);
                        assert!(
                            (v[b].is_nan() && got.is_nan()) || (v[b] - got).abs() <= 1e-6,
                            "{how:?} {vs:?} {sigma:?} {x},{y},{b}: {} vs {got}",
                            v[b]
                        );
                    }
                    assert_eq!(u32::from(q[y * 48 + x]), n);
                    assert_eq!(q[48 * 48 + y * 48 + x], src);
                }
            }
        }
    }
}

#[test]
fn cloud_hole_and_coarse_hole() {
    let (nx, ny) = (96, 96);
    let f1 = scene(nx, ny, 1, 0.0);
    let f2 = shifted(&f1, 0.04);
    let f3 = shifted(&f1, 0.08);
    let (c1, c2, c3) = (
        block_mean(&f1, nx, ny, 16),
        block_mean(&f2, nx, ny, 16),
        block_mean(&f3, nx, ny, 16),
    );
    let mut f1h = f1.clone();
    for b in 0..NB {
        for y in 40..52 {
            for x in 40..52 {
                f1h[b * nx * ny + y * nx + x] = f32::NAN;
            }
        }
    }
    let fine = cat(&[&f1h, &f3]);
    let coarse = cat(&[&c1, &c2, &c3]);
    let geom = || CoarseGeom::Affine(affine(16, 0.0, 0.0));
    let mut p = StarfmParams::new(8);
    // Two pairs: the hole is predicted from pair 2, exactly.
    let (out, q) = run_starfm(
        &fine,
        &coarse,
        nx,
        ny,
        (6, 6),
        geom(),
        &plan_for(&[(0, 0), (1, 2)], 1),
        &p,
        &whole(nx, ny),
        1,
    );
    assert!(max_err(&out, &f2) <= 1e-6);
    assert_eq!(q[nx * ny + 45 * nx + 45], SRC_ONE_PAIR);
    assert_eq!(q[nx * ny + 5 * nx + 5], SRC_TWO_PAIRS);
    // One pair: the hole is missing, or the coarse value with the fallback.
    let (out, q) = run_starfm(
        &fine,
        &coarse,
        nx,
        ny,
        (6, 6),
        geom(),
        &plan_for(&[(0, 0)], 1),
        &p,
        &whole(nx, ny),
        1,
    );
    assert!(out[45 * nx + 45].is_nan() && q[nx * ny + 45 * nx + 45] == SRC_MISSING);
    p.common.fallback = Fallback::Coarse;
    p.common.resampling = Resampling::Nearest;
    let (out, q) = run_starfm(
        &fine,
        &coarse,
        nx,
        ny,
        (6, 6),
        geom(),
        &plan_for(&[(0, 0)], 1),
        &p,
        &whole(nx, ny),
        1,
    );
    let expect = c2[2 * 6 + 2]; // band 0, cell (2, 2) of the coarse image of t2
    assert!(
        (out[45 * nx + 45] - expect).abs() < 1e-7,
        "{} vs {expect}",
        out[45 * nx + 45]
    );
    assert_eq!(q[nx * ny + 45 * nx + 45], SRC_COARSE);
    // ESTARFM: single-pair fallback, exact in the homogeneous case.
    let e = EstarfmParams::new(8);
    let (out, q) = run_estarfm(
        &fine,
        &coarse,
        nx,
        ny,
        (6, 6),
        geom(),
        &plan_for(&[(0, 0), (1, 2)], 1),
        &e,
        &whole(nx, ny),
        1,
    );
    assert!(max_err(&out, &f2) <= 1e-5);
    assert_eq!(q[nx * ny + 45 * nx + 45], SRC_ONE_PAIR);
    // A coarse hole at the predicted date: the cell's pixels are missing under nearest.
    let mut coarse2 = coarse.clone();
    for b in 0..NB {
        coarse2[(NB + b) * 36 + 2 * 6 + 2] = f32::NAN;
    }
    let mut p = StarfmParams::new(8);
    p.common.resampling = Resampling::Nearest;
    let (out, _) = run_starfm(
        &fine,
        &coarse2,
        nx,
        ny,
        (6, 6),
        geom(),
        &plan_for(&[(0, 0), (1, 2)], 1),
        &p,
        &whole(nx, ny),
        1,
    );
    assert!(out[34 * nx + 34].is_nan(), "pixel in the missing cell");
    assert!(
        (out[10 * nx + 10] - f2[10 * nx + 10]).abs() < 1e-6,
        "neighbours are unaffected"
    );
    // Bilinear: the missing cell's pixels use the other cells, renormalised.
    let c = Coarse {
        ncx: 6,
        ncy: 6,
        nb: NB,
        values: &coarse2,
        geom: geom(),
    };
    let m = resample_band(&c, 1, 0, nx, ny, Resampling::Bilinear);
    // Pixel (33, 33): centre at u = v = 33.5/16 = 2.09375; the bilinear cells around 1.59375 are 1 and 2 with
    // weights 0.40625 and 0.59375 in each axis; cell (2, 2) is missing, so the other three renormalise.
    let (w00, w10, w01) = (0.40625 * 0.40625, 0.59375 * 0.40625, 0.40625 * 0.59375);
    let v = |i: usize, j: usize| f64::from(coarse2[NB * 36 + j * 6 + i]);
    let expect = (w00 * v(1, 1) + w10 * v(2, 1) + w01 * v(1, 2)) / (w00 + w10 + w01);
    assert!((f64::from(m[33 * nx + 33]) - expect).abs() < 1e-6);
}

fn sub_extent(fine: &[f32], npairs: usize, nx: usize, ny: usize, x0: usize, y0: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(npairs * NB * w * h);
    for k in 0..npairs * NB {
        for y in y0..y0 + h {
            out.extend_from_slice(&fine[k * nx * ny + y * nx + x0..k * nx * ny + y * nx + x0 + w]);
        }
    }
    out
}

#[test]
fn tiles_and_threads_do_not_change_the_result() {
    let s = changing_scene(64, 64, 5);
    let (nx, ny) = (s.nx, s.ny);
    let plan = plan_for(&[(0, 0), (1, 2)], 1);
    let mut sp_ = StarfmParams::new(5);
    sp_.common.sigma = SigmaScope::Window;
    sp_.classes = 8.0;
    let mut ep = EstarfmParams::new(5);
    ep.v_scope = VScope::Cell;
    ep.classes = 3.0;
    let geom = |y0: usize, x0: usize| {
        let mut a = affine(8, 0.0, 0.0);
        a[0] += a[1] * x0 as f64;
        a[3] += a[5] * y0 as f64;
        CoarseGeom::Affine(a)
    };
    let sig = image_sigma(
        &Fine {
            nx,
            ny,
            nb: NB,
            values: &s.fine,
        },
        &whole(nx, ny),
        1,
    );
    let (ref_s, qs) = run_starfm(
        &s.fine,
        &s.coarse,
        nx,
        ny,
        (8, 8),
        geom(0, 0),
        &plan,
        &sp_,
        &whole(nx, ny),
        1,
    );
    let (ref_e, qe) = run_estarfm(
        &s.fine,
        &s.coarse,
        nx,
        ny,
        (8, 8),
        geom(0, 0),
        &plan,
        &ep,
        &whole(nx, ny),
        1,
    );
    for jobs in [3, 8] {
        let (a, qa) = run_starfm(
            &s.fine,
            &s.coarse,
            nx,
            ny,
            (8, 8),
            geom(0, 0),
            &plan,
            &sp_,
            &whole(nx, ny),
            jobs,
        );
        assert!(a == ref_s || a.iter().zip(&ref_s).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(qa, qs);
        let (a, qa) = run_estarfm(
            &s.fine,
            &s.coarse,
            nx,
            ny,
            (8, 8),
            geom(0, 0),
            &plan,
            &ep,
            &whole(nx, ny),
            jobs,
        );
        assert!(a.iter().zip(&ref_e).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(qa, qe);
    }
    // Strips of rows with the halo; the coarse series is the whole one (a lookup of the strip's geometry).
    let halo_s = required_halo(Method::Starfm, 5, 8.0, VScope::Window);
    let halo_e = required_halo(Method::Estarfm, 5, 8.0, VScope::Cell);
    for rows_per in [7usize, 16, 33] {
        for (est, halo) in [(false, halo_s), (true, halo_e)] {
            let mut y = 0;
            while y < ny {
                let h = rows_per.min(ny - y);
                let (ey0, ey1) = (y.saturating_sub(halo), (y + h + halo).min(ny));
                let fine = sub_extent(&s.fine, 2, nx, ny, 0, ey0, nx, ey1 - ey0);
                let region = Region {
                    x0: 0,
                    y0: y - ey0,
                    nx,
                    ny: h,
                };
                let f = Fine {
                    nx,
                    ny: ey1 - ey0,
                    nb: NB,
                    values: &fine,
                };
                let c = Coarse {
                    ncx: 8,
                    ncy: 8,
                    nb: NB,
                    values: &s.coarse,
                    geom: geom(ey0, 0),
                };
                let mut out = vec![0f32; NB * nx * h];
                if est {
                    estarfm(&f, &c, &plan, &sig, &ep, &region, &mut out, None, 2).unwrap();
                } else {
                    starfm(&f, &c, &plan, &sig, &sp_, &region, &mut out, None, 2).unwrap();
                }
                let whole_out = if est { &ref_e } else { &ref_s };
                for b in 0..NB {
                    for r in 0..h {
                        let got = &out[(b * h + r) * nx..(b * h + r + 1) * nx];
                        let want = &whole_out[(b * ny + y + r) * nx..(b * ny + y + r + 1) * nx];
                        assert!(
                            got.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()),
                            "strip {rows_per} est={est} row {}",
                            y + r
                        );
                    }
                }
                y += h;
            }
        }
    }
    // A 3 x 3 tiling of columns and rows (STARFM).
    let halo = halo_s;
    for ty in 0..3 {
        for tx in 0..3 {
            let (x, y, w, h): (usize, usize, usize, usize) = (
                tx * 22,
                ty * 22,
                if tx == 2 { 20 } else { 22 },
                if ty == 2 { 20 } else { 22 },
            );
            let (ex0, ex1) = (x.saturating_sub(halo), (x + w + halo).min(nx));
            let (ey0, ey1) = (y.saturating_sub(halo), (y + h + halo).min(ny));
            let fine = sub_extent(&s.fine, 2, nx, ny, ex0, ey0, ex1 - ex0, ey1 - ey0);
            let f = Fine {
                nx: ex1 - ex0,
                ny: ey1 - ey0,
                nb: NB,
                values: &fine,
            };
            let c = Coarse {
                ncx: 8,
                ncy: 8,
                nb: NB,
                values: &s.coarse,
                geom: geom(ey0, ex0),
            };
            let region = Region {
                x0: x - ex0,
                y0: y - ey0,
                nx: w,
                ny: h,
            };
            let mut out = vec![0f32; NB * w * h];
            starfm(&f, &c, &plan, &sig, &sp_, &region, &mut out, None, 2).unwrap();
            for b in 0..NB {
                for r in 0..h {
                    for cc in 0..w {
                        assert_eq!(
                            out[(b * h + r) * w + cc].to_bits(),
                            ref_s[(b * ny + y + r) * nx + x + cc].to_bits()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lookup_geometry_equals_affine() {
    let s = changing_scene(48, 48, 6);
    let plan = plan_for(&[(0, 0), (1, 2)], 1);
    let a = affine(8, 0.25, 0.4);
    let (mut u, mut v) = (vec![0f32; 48 * 48], vec![0f32; 48 * 48]);
    for y in 0..48 {
        for x in 0..48 {
            u[y * 48 + x] = (a[0] + a[1] * (x as f64 + 0.5) + a[2] * (y as f64 + 0.5)) as f32;
            v[y * 48 + x] = (a[3] + a[4] * (x as f64 + 0.5) + a[5] * (y as f64 + 0.5)) as f32;
        }
    }
    let mut p = StarfmParams::new(4);
    p.common.resampling = Resampling::Nearest;
    let (a1, _) = run_starfm(
        &s.fine,
        &s.coarse,
        48,
        48,
        (6, 6),
        CoarseGeom::Affine(a),
        &plan,
        &p,
        &whole(48, 48),
        1,
    );
    let (a2, _) = run_starfm(
        &s.fine,
        &s.coarse,
        48,
        48,
        (6, 6),
        CoarseGeom::Lookup { u: &u, v: &v },
        &plan,
        &p,
        &whole(48, 48),
        1,
    );
    // Pixel centres on a cell edge may fall either side in f32; allow a few.
    let diff = a1.iter().zip(&a2).filter(|(x, y)| (**x - **y).abs() > 1e-6).count();
    assert!(diff * 100 < a1.len(), "{diff} of {} differ", a1.len());
}

#[test]
fn parameters_behave() {
    let s = changing_scene(48, 48, 7);
    let plan = plan_for(&[(0, 0), (1, 2)], 1);
    // A huge number of classes keeps only equal-valued neighbours: the centre.
    let mut p = StarfmParams::new(5);
    p.classes = 1e12;
    let (_, q) = run_starfm(
        &s.fine,
        &s.coarse,
        48,
        48,
        (6, 6),
        CoarseGeom::Affine(affine(8, 0.0, 0.0)),
        &plan,
        &p,
        &whole(48, 48),
        1,
    );
    // Two pairs, one centre each: n_similar = 2 (the minimum over the bands).
    assert!(q[..48 * 48].iter().all(|&n| n == 2), "{:?}", &q[..8]);
    // A huge spatial scale makes D = 1: same as the oracle with that scale.
    let mut p = StarfmParams::new(5);
    p.spatial_scale_px = 1e12;
    let f = Fine {
        nx: 48,
        ny: 48,
        nb: NB,
        values: &s.fine,
    };
    let c = Coarse {
        ncx: 6,
        ncy: 6,
        nb: NB,
        values: &s.coarse,
        geom: CoarseGeom::Affine(affine(8, 0.0, 0.0)),
    };
    let sig = image_sigma(&f, &whole(48, 48), 1);
    let (out, _) = run_starfm(
        &s.fine,
        &s.coarse,
        48,
        48,
        (6, 6),
        CoarseGeom::Affine(affine(8, 0.0, 0.0)),
        &plan,
        &p,
        &whole(48, 48),
        1,
    );
    let (v, _, _) = starfm_pixel(&f, &c, &plan.targets[0], 1, 20, 20, &sig, &p);
    assert!((v - f64::from(out[48 * 48 + 20 * 48 + 20])).abs() < 1e-6);
    // sigma of the image and of the window differ where the local variance differs.
    let mut a = StarfmParams::new(5);
    a.classes = 6.0;
    let mut b = a;
    b.common.sigma = SigmaScope::Window;
    let g = || CoarseGeom::Affine(affine(8, 0.0, 0.0));
    let (oa, _) = run_starfm(&s.fine, &s.coarse, 48, 48, (6, 6), g(), &plan, &a, &whole(48, 48), 1);
    let (ob, _) = run_starfm(&s.fine, &s.coarse, 48, 48, (6, 6), g(), &plan, &b, &whole(48, 48), 1);
    assert!(max_err(&oa, &ob) > 1e-4);
    // ESTARFM with the coarse image identical to the fine one (one cell per pixel): R_i = 1 and weights 1 / P.
    let (nx, ny) = (24, 24);
    let f1 = scene(nx, ny, 9, 0.01);
    let f2 = shifted(&f1, 0.03);
    let f3 = shifted(&f1, 0.06);
    let fine = cat(&[&f1, &f3]);
    let coarse = cat(&[&f1, &f2, &f3]);
    let (_, q) = run_estarfm(
        &fine,
        &coarse,
        nx,
        ny,
        (nx, ny),
        CoarseGeom::Affine([0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        &plan_for(&[(0, 0), (1, 2)], 1),
        &EstarfmParams::new(4),
        &whole(nx, ny),
        1,
    );
    assert!(q[..nx * ny].iter().all(|&n| n >= 1));
}

#[test]
fn image_edges_are_exact() {
    // The image edge and corners: the clipped windows give the exact homogeneous answer.
    let (nx, ny) = (40, 40);
    let f1 = scene(nx, ny, 3, 0.0);
    let f2 = shifted(&f1, 0.02);
    let (c1, c2) = (block_mean(&f1, nx, ny, 8), block_mean(&f2, nx, ny, 8));
    let fine = cat(&[&f1]);
    let coarse = cat(&[&c1, &c2]);
    let p = StarfmParams::new(6);
    let (out, _) = run_starfm(
        &fine,
        &coarse,
        nx,
        ny,
        (5, 5),
        CoarseGeom::Affine(affine(8, 0.0, 0.0)),
        &plan_for(&[(0, 0)], 1),
        &p,
        &whole(nx, ny),
        2,
    );
    assert!(max_err(&out, &f2) <= 1e-6);
}

// ---------------------------------------------------------------------------------------------------------------
// the plan

fn rule() -> DateRule {
    DateRule::default()
}

#[test]
fn plan_bracket_and_one_sided() {
    // Daily coarse series; fine on days 10, 20, 30.
    let coarse: Vec<i64> = (0..40).collect();
    let fine = [10, 20, 30];
    let valid = [1.0, 1.0, 1.0];
    let mut r = rule();
    r.include_fine_dates = false;
    let pl = plan(&fine, &valid, &coarse, None, &r, Method::Starfm).unwrap();
    // Output days 10..=30; day 15 is bracketed by the pairs of days 10 and 20.
    let t15 = pl.targets.iter().find(|t| t.day == 15).unwrap();
    assert_eq!(t15.pairs, vec![(0, 10), (1, 20)]);
    // Day 10 is a pair date: one pair.
    assert_eq!(pl.targets.iter().find(|t| t.day == 10).unwrap().pairs, vec![(0, 10)]);
    assert_eq!(pl.targets.len(), 21);
    // ESTARFM: bracket, and two pairs on one side flagged as extrapolated.
    let pl = plan(&fine, &valid, &coarse, None, &r, Method::Estarfm).unwrap();
    assert_eq!(
        pl.targets.iter().find(|t| t.day == 15).unwrap().pairs,
        vec![(0, 10), (1, 20)]
    );
    assert!(!pl.targets.iter().find(|t| t.day == 15).unwrap().extrapolated);
    let mut r2 = r.clone();
    r2.start = Some(0);
    r2.end = Some(39);
    let pl = plan(&fine, &valid, &coarse, None, &r2, Method::Estarfm).unwrap();
    let t5 = pl.targets.iter().find(|t| t.day == 5).unwrap();
    assert_eq!(t5.pairs, vec![(0, 10), (1, 20)]);
    assert!(t5.extrapolated);
    let t35 = pl.targets.iter().find(|t| t.day == 35).unwrap();
    assert_eq!(t35.pairs, vec![(1, 20), (2, 30)]);
    // STARFM nearest: ties go to the earlier date.
    let mut r3 = r.clone();
    r3.pairs = PairsMode::Nearest;
    let pl = plan(&fine, &valid, &coarse, None, &r3, Method::Starfm).unwrap();
    assert_eq!(pl.targets.iter().find(|t| t.day == 15).unwrap().pairs, vec![(0, 10)]);
    assert_eq!(pl.targets.iter().find(|t| t.day == 16).unwrap().pairs, vec![(1, 20)]);
}

#[test]
fn plan_observed_exclude_gap_and_windows() {
    let coarse: Vec<i64> = (0..40).collect();
    let fine = [10, 20, 30];
    let valid = [1.0, 0.1, 1.0];
    let r = rule();
    // The cloudy date (10% valid) is not a pair but is output as observed with the others as pairs.
    let pl = plan(&fine, &valid, &coarse, None, &r, Method::Starfm).unwrap();
    assert_eq!(pl.unpaired.len(), 1);
    let t20 = pl.targets.iter().find(|t| t.day == 20).unwrap();
    assert_eq!(t20.observed, Some(1));
    assert_eq!(t20.pairs, vec![(0, 10), (2, 30)]);
    // keep_observed excludes the date's own pair.
    let valid = [1.0; 3];
    let pl = plan(&fine, &valid, &coarse, None, &r, Method::Starfm).unwrap();
    let t20 = pl.targets.iter().find(|t| t.day == 20).unwrap();
    assert_eq!(t20.observed, Some(1));
    assert_eq!(t20.pairs, vec![(0, 10), (2, 30)]);
    // exclude_dates: never a pair; max_gap drops far pairs and skips a date left with none.
    let mut r2 = rule();
    r2.exclude = vec![20];
    r2.include_fine_dates = false;
    r2.max_gap = Some(6);
    let pl = plan(&fine, &valid, &coarse, None, &r2, Method::Starfm).unwrap();
    assert!(pl.targets.iter().all(|t| t.pairs.iter().all(|p| p.0 != 1)));
    assert!(pl.targets.iter().find(|t| t.day == 20).is_none());
    assert!(pl.skipped.iter().any(|s| s.0 == 20));
    // 8-day composites: a fine date inside a window is matched to it; daily tolerance 0.
    let cd: Vec<i64> = (0..10).map(|j| j * 8).collect();
    let win: Vec<(i64, i64)> = cd.iter().map(|&d| (d, d + 7)).collect();
    let pl = plan(&[27, 70], &[1.0, 1.0], &cd, Some(&win), &rule(), Method::Starfm).unwrap();
    assert_eq!(pl.targets.iter().find(|t| t.observed == Some(0)).unwrap().coarse, 3);
    assert_eq!(pl.targets.iter().find(|t| t.observed == Some(1)).unwrap().coarse, 8);
    let mut r4 = rule();
    r4.tolerance = Some(0);
    r4.include_fine_dates = false;
    let pl = plan(&[10], &[1.0], &coarse, None, &r4, Method::Starfm).unwrap();
    assert!(pl.targets.iter().any(|t| t.day == 10));
    let e = plan(&[10], &[1.0], &[0, 3, 6], None, &r4, Method::Starfm);
    assert!(e.is_err());
}

// ---------------------------------------------------------------------------------------------------------------
// leave one date out

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Truth of class `c` band `b` at day `t`: bands green, red, NIR.
fn truth(c: usize, b: usize, t: f64) -> f64 {
    let season = |on: f64, off: f64, k: f64| logistic(k * (t - on)) * (1.0 - logistic(k * (t - off)));
    match c {
        2 => [
            0.05 + 0.04 * season(110.0, 270.0, 0.07),
            0.20 - 0.15 * season(110.0, 270.0, 0.07),
            0.15 + 0.30 * season(110.0, 270.0, 0.07),
        ][b],
        3 => [
            0.06 + 0.03 * season(150.0, 300.0, 0.05),
            0.22 - 0.16 * season(150.0, 300.0, 0.05),
            0.18 + 0.22 * season(150.0, 300.0, 0.05),
        ][b],
        _ => CLASS_VALUES[c][b],
    }
}

#[test]
fn leave_one_date_out_beats_the_baselines() {
    let (nx, ny) = (96, 96);
    let cls = classes(nx, ny);
    let tex = scene(nx, ny, 21, 0.004);
    let tex0: Vec<f32> = tex
        .iter()
        .enumerate()
        .map(|(i, v)| v - CLASS_VALUES[cls[i % (nx * ny)] as usize][i / (nx * ny)] as f32)
        .collect();
    let image = |t: f64| -> Vec<f32> {
        (0..NB * nx * ny)
            .map(|i| (truth(cls[i % (nx * ny)] as usize, i / (nx * ny), t) as f32) + tex0[i])
            .collect()
    };
    let fine_days: Vec<i64> = vec![24, 88, 152, 208, 272, 336];
    let coarse_days: Vec<i64> = (0..46).map(|j| j * 8).collect();
    let fines: Vec<Vec<f32>> = fine_days.iter().map(|&d| image(d as f64)).collect();
    let mut rng = Lcg(33);
    let coarse: Vec<f32> = coarse_days
        .iter()
        .flat_map(|&d| {
            let c = block_mean(&image(d as f64), nx, ny, 16);
            c.iter()
                .map(|v| v * 1.03 + 0.005 + 0.003 * rng.normal() as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    let fine: Vec<f32> = fines.iter().flat_map(|f| f.iter().copied()).collect();
    let mut r = rule();
    r.exclude = vec![152];
    r.dates = Some(vec![152]);
    r.include_fine_dates = false;
    let pl = plan(&fine_days, &[1.0; 6], &coarse_days, None, &r, Method::Starfm).unwrap();
    assert_eq!(pl.targets[0].pairs, vec![(1, 11), (3, 26)]);
    let pe = plan(&fine_days, &[1.0; 6], &coarse_days, None, &r, Method::Estarfm).unwrap();
    let truth_img = &fines[2];
    let rmse = |a: &[f32]| -> f64 {
        let n = a.len();
        (a.iter()
            .zip(truth_img.iter())
            .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt()
    };
    // Baselines: the nearest fine date (208) and the linear interpolation in time of 88 and 208.
    let nearest = &fines[3];
    let w = (152.0 - 88.0) / (208.0 - 88.0);
    let lin: Vec<f32> = fines[1]
        .iter()
        .zip(&fines[3])
        .map(|(a, b)| a * (1.0 - w as f32) + b * w as f32)
        .collect();
    let (rn, rl) = (rmse(nearest), rmse(&lin));
    let geom = || CoarseGeom::Affine(affine(16, 0.0, 0.0));
    let mut p = StarfmParams::new(10);
    p.classes = 4.0;
    let (sf, _) = run_starfm(&fine, &coarse, nx, ny, (6, 6), geom(), &pl, &p, &whole(nx, ny), 2);
    let mut e = EstarfmParams::new(10);
    e.classes = 4.0;
    let (es, _) = run_estarfm(&fine, &coarse, nx, ny, (6, 6), geom(), &pe, &e, &whole(nx, ny), 2);
    let (rs, re) = (rmse(&sf), rmse(&es));
    println!("LOO rmse: nearest {rn:.5} linear {rl:.5} starfm {rs:.5} estarfm {re:.5}");
    assert!(rs < 0.8 * rn.min(rl), "starfm {rs} vs baselines {rn} {rl}");
    assert!(re < 0.8 * rn.min(rl), "estarfm {re} vs baselines {rn} {rl}");
}
