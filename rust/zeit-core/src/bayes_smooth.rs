//! Bayesian smoothing of classification probabilities.
//!
//! An implementation of the method of Camara et al. (2024, Remote Sensing
//! 16(23):4572), written from the paper's and the sits book's description of the
//! model, without third-party code. Each class' probability is taken to logits,
//! `x = ln(p / (1 - p))`, one class at a time. The logit of the truth is
//! `N(m, s2)`, with `m` and `s2` the mean and variance of the logits of the
//! `neigh_fraction` highest ones of the class in the pixel's square window
//! (so that a pixel at the edge of a class is not blurred by the other side), and
//! the observed logit is `N(truth, smoothness)`. The posterior mean is
//! `(s2 * x + smoothness * m) / (s2 + smoothness)`, taken back by the logistic
//! function. The classes are not renormalized.
//!
//! Choices the description leaves open (see [`Options`]), decided by comparing
//! the outputs with those of the R package bayesEO on a synthetic cube
//! (tests/bayes_smooth_ref.rs) and with the outputs of sits 1.5.4 (the fixture
//! tests/data/smooth_probs_sits.json of the Python tests, windows 5, 7 and 9,
//! fractions 0.5, 0.3 and 1, with missing pixels): the window is mirrored at the
//! image's edges repeating the edge pixel (`-1` is `0`); the probabilities are cut at `EPS` and
//! `1 - EPS` before the logit; the variance divides by `t - 1` of the `t` logits
//! used; `t = ceil(neigh_fraction * n)` of the `n` valid pixels of the window
//! (81 * 0.5 = 40.5 gives 41), at least 2. A pixel with a NaN in any class is NaN
//! in all of them and does not count as a neighbour; a pixel alone in its window
//! keeps its values.
//!
//! Layout `(class, row, column)`, index `(k * nrows + r) * ncols + c`.

use crate::parallel;

/// How many of the window's pixels, `fraction * n`, are used as the neighbourhood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CountRule {
    Floor,
    Round,
    Ceil,
}

/// The choices the method's description leaves open.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Probabilities are cut to `[eps, 1 - eps]` before the logit.
    pub eps: f64,
    /// The neighbourhood's variance divides by `t - 1` (else by `t`).
    pub sample_variance: bool,
    /// How `fraction * n` becomes a number of neighbours.
    pub count_rule: CountRule,
    /// Mirror the window at the edges without repeating the edge pixel (else
    /// repeating it).
    pub reflect101: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options { eps: EPS, sample_variance: true, count_rule: CountRule::Ceil, reflect101: false }
    }
}

/// The cut of probabilities before the logit.
pub const EPS: f64 = 1e-4;

/// A variance under this is zero.
const TINY: f64 = 1e-12;

/// Index `i` (possibly outside `0..n`) mirrored into `0..n`.
fn mirror(i: isize, n: usize, reflect101: bool) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    if reflect101 {
        let period = 2 * (n - 1);
        let m = i.rem_euclid(period);
        (if m >= n { period - m } else { m }) as usize
    } else {
        let period = 2 * n;
        let m = i.rem_euclid(period);
        (if m >= n { period - 1 - m } else { m }) as usize
    }
}

/// Smooths the probabilities `probs` (class, row, column) with the default
/// [`Options`]: `smoothness` is the logit variance of each class, `window` the
/// odd side of the window, `neigh_fraction` the fraction of its pixels used.
/// NaN in any class makes a pixel NaN. At most `n_jobs` threads (`n_jobs <= 0`
/// is the default of the crate).
#[allow(clippy::too_many_arguments)]
pub fn smooth_probs(
    probs: &[f64],
    n_classes: usize,
    nrows: usize,
    ncols: usize,
    window: usize,
    smoothness: &[f64],
    neigh_fraction: f64,
    n_jobs: i32,
) -> Vec<f64> {
    smooth_probs_with(
        probs,
        n_classes,
        nrows,
        ncols,
        window,
        smoothness,
        neigh_fraction,
        n_jobs,
        &Options::default(),
    )
}

/// [`smooth_probs`] with the open choices given.
#[allow(clippy::too_many_arguments)]
pub fn smooth_probs_with(
    probs: &[f64],
    n_classes: usize,
    nrows: usize,
    ncols: usize,
    window: usize,
    smoothness: &[f64],
    neigh_fraction: f64,
    n_jobs: i32,
    opts: &Options,
) -> Vec<f64> {
    let n_pix = nrows * ncols;
    assert_eq!(probs.len(), n_classes * n_pix, "probs must be n_classes x nrows x ncols");
    assert_eq!(smoothness.len(), n_classes, "one smoothness per class");
    if n_pix == 0 || n_classes == 0 {
        return vec![f64::NAN; probs.len()];
    }
    let half = (window / 2) as isize;

    // The logits and the pixels with every class present.
    let valid: Vec<bool> = (0..n_pix).map(|i| (0..n_classes).all(|k| !probs[k * n_pix + i].is_nan())).collect();
    let logits: Vec<f64> = probs
        .iter()
        .map(|&p| {
            if p.is_nan() {
                f64::NAN
            } else {
                let q = p.clamp(opts.eps, 1.0 - opts.eps);
                (q / (1.0 - q)).ln()
            }
        })
        .collect();

    // The window's mirrored rows and columns, per row and per column.
    let w = 2 * half as usize + 1;
    let row_idx: Vec<usize> = (0..nrows)
        .flat_map(|r| (-half..=half).map(move |d| mirror(r as isize + d, nrows, opts.reflect101)))
        .collect();
    let col_idx: Vec<usize> = (0..ncols)
        .flat_map(|c| (-half..=half).map(move |d| mirror(c as isize + d, ncols, opts.reflect101)))
        .collect();

    // One output row at a time, as (column, class).
    let mut by_row = vec![f64::NAN; n_pix * n_classes];
    parallel::for_each_row(
        &mut by_row,
        ncols * n_classes,
        n_jobs,
        || (Vec::<usize>::with_capacity(w * w), Vec::<f64>::with_capacity(w * w)),
        |(pos, vals), r, row| {
            for c in 0..ncols {
                let i = r * ncols + c;
                let out = &mut row[c * n_classes..(c + 1) * n_classes];
                if !valid[i] {
                    continue; // NaN already
                }
                pos.clear();
                for &rr in &row_idx[r * w..(r + 1) * w] {
                    for &cc in &col_idx[c * w..(c + 1) * w] {
                        let j = rr * ncols + cc;
                        if valid[j] {
                            pos.push(j);
                        }
                    }
                }
                let n = pos.len();
                if n <= 1 {
                    for (k, o) in out.iter_mut().enumerate() {
                        *o = probs[k * n_pix + i];
                    }
                    continue;
                }
                let t = neighbours(neigh_fraction, n, opts);
                for (k, o) in out.iter_mut().enumerate() {
                    vals.clear();
                    vals.extend(pos.iter().map(|&j| logits[k * n_pix + j]));
                    let (m, s2) = top_mean_var(vals, t, opts.sample_variance);
                    let sg = smoothness[k];
                    let x = logits[k * n_pix + i];
                    let s2 = if s2 < TINY { 0.0 } else { s2 };
                    let mu = if s2 + sg <= 0.0 { x } else { (s2 * x + sg * m) / (s2 + sg) };
                    *o = 1.0 / (1.0 + (-mu).exp());
                }
            }
        },
    );

    let mut out = vec![0.0; probs.len()];
    for i in 0..n_pix {
        for k in 0..n_classes {
            out[k * n_pix + i] = by_row[i * n_classes + k];
        }
    }
    out
}

/// The number of neighbours used out of the `n >= 2` valid pixels of a window.
fn neighbours(fraction: f64, n: usize, opts: &Options) -> usize {
    let x = fraction * n as f64;
    let t = match opts.count_rule {
        CountRule::Floor => x.floor(),
        CountRule::Round => x.round(),
        CountRule::Ceil => x.ceil(),
    } as usize;
    let least = if opts.sample_variance { 2 } else { 1 };
    t.clamp(least, n)
}

/// Mean and variance of the `t` highest of `v` (reordered). The variance divides
/// by `t - 1` if `sample`, with 0 for a single value.
fn top_mean_var(v: &mut [f64], t: usize, sample: bool) -> (f64, f64) {
    if t < v.len() {
        v.select_nth_unstable_by(t - 1, |a, b| b.total_cmp(a));
    }
    let top = &v[..t];
    let mean = top.iter().sum::<f64>() / t as f64;
    let ss: f64 = top.iter().map(|x| (x - mean) * (x - mean)).sum();
    let div = if sample { t as f64 - 1.0 } else { t as f64 };
    (mean, if div > 0.0 { ss / div } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward version of the method for one pixel, written for the
    /// tests: sorts the window's logits, uses `Options::default()`.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        probs: &[f64],
        nk: usize,
        nr: usize,
        nc: usize,
        window: usize,
        smooth: &[f64],
        frac: f64,
        r: usize,
        c: usize,
    ) -> Vec<f64> {
        let n_pix = nr * nc;
        let h = (window / 2) as isize;
        let valid = |j: usize| (0..nk).all(|k| !probs[k * n_pix + j].is_nan());
        let mut pos = Vec::new();
        for dr in -h..=h {
            for dc in -h..=h {
                let rr = mirror(r as isize + dr, nr, false);
                let cc = mirror(c as isize + dc, nc, false);
                if valid(rr * nc + cc) {
                    pos.push(rr * nc + cc);
                }
            }
        }
        let i = r * nc + c;
        if pos.len() == 1 {
            return (0..nk).map(|k| probs[k * n_pix + i]).collect();
        }
        let t = ((frac * pos.len() as f64).ceil() as usize).max(2).min(pos.len());
        (0..nk)
            .map(|k| {
                let lg = |j: usize| {
                    let q = probs[k * n_pix + j].clamp(EPS, 1.0 - EPS);
                    (q / (1.0 - q)).ln()
                };
                let mut v: Vec<f64> = pos.iter().map(|&j| lg(j)).collect();
                v.sort_by(|a, b| b.total_cmp(a));
                let top = &v[..t];
                let m = top.iter().sum::<f64>() / t as f64;
                let s2 = top.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (t as f64 - 1.0);
                let s2 = if s2 < TINY { 0.0 } else { s2 };
                let mu = if s2 + smooth[k] <= 0.0 { lg(i) } else { (s2 * lg(i) + smooth[k] * m) / (s2 + smooth[k]) };
                1.0 / (1.0 + (-mu).exp())
            })
            .collect()
    }

    /// A cube from a per-class value function of (row, column).
    fn cube(nk: usize, nr: usize, nc: usize, f: impl Fn(usize, usize, usize) -> f64) -> Vec<f64> {
        let mut v = vec![0.0; nk * nr * nc];
        for k in 0..nk {
            for r in 0..nr {
                for c in 0..nc {
                    v[(k * nr + r) * nc + c] = f(k, r, c);
                }
            }
        }
        v
    }

    fn argmax(v: &[f64], nk: usize, n_pix: usize, i: usize) -> usize {
        (0..nk).max_by(|&a, &b| v[a * n_pix + i].total_cmp(&v[b * n_pix + i])).unwrap()
    }

    /// A pseudo-random value in [0, 1) from an integer.
    fn hash(mut x: u64) -> f64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        x ^= x >> 29;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 32;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn mirror_reflect101_and_repeat() {
        assert_eq!(mirror(-1, 5, true), 1);
        assert_eq!(mirror(-2, 5, true), 2);
        assert_eq!(mirror(5, 5, true), 3);
        assert_eq!(mirror(6, 5, true), 2);
        assert_eq!(mirror(-1, 5, false), 0);
        assert_eq!(mirror(5, 5, false), 4);
        // Smaller than the window: back and forth.
        assert_eq!(mirror(-4, 2, true), 0);
        assert_eq!(mirror(3, 2, true), 1);
        assert_eq!(mirror(7, 1, true), 0);
        for i in -30..30 {
            assert!(mirror(i, 3, true) < 3 && mirror(i, 3, false) < 3);
        }
    }

    #[test]
    fn constant_image_is_unchanged() {
        let vals = [0.7, 0.2, 0.1];
        let p = cube(3, 12, 15, |k, _, _| vals[k]);
        for (window, frac, s) in [(9, 0.5, 20.0), (3, 1.0, 1.0), (21, 0.1, 200.0)] {
            let out = smooth_probs(&p, 3, 12, 15, window, &[s; 3], frac, 2);
            for (a, b) in out.iter().zip(&p) {
                assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn isolated_noisy_pixel_returns_to_the_background() {
        let (nr, nc) = (21, 21);
        let build = |ripple: f64| {
            cube(2, nr, nc, |k, r, c| {
                let jitter = ripple * if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                let p0 = if r == 10 && c == 10 { 0.4 } else { 0.9 + jitter };
                if k == 0 {
                    p0
                } else {
                    1.0 - p0
                }
            })
        };
        for ripple in [0.0, 0.01] {
            let p = build(ripple);
            let out = smooth_probs(&p, 2, nr, nc, 9, &[20.0, 20.0], 0.5, 1);
            let i = 10 * nc + 10;
            assert_eq!(argmax(&out, 2, nr * nc, i), 0, "ripple {ripple}");
            assert!(out[i] > 0.5, "ripple {ripple}: {}", out[i]);
            // Between the original and the background.
            assert!(out[i] > p[i] && out[i] < 0.9 + ripple + 1e-9,"ripple {ripple}: {}", out[i]);
            // Far from the noise: unchanged.
            for (r, c) in [(0, 0), (20, 20), (2, 18), (18, 3)] {
                let j = r * nc + c;
                let fresh = naive(&p, 2, nr, nc, 9, &[20.0; 2], 0.5, r, c);
                assert!((out[j] - fresh[0]).abs() < 1e-9);
                if ripple == 0.0 {
                    assert!((out[j] - p[j]).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn smoothness_limits() {
        let (nk, nr, nc) = (3, 11, 13);
        let p = cube(nk, nr, nc, |k, r, c| 0.05 + 0.9 * hash((k * 1000 + r * 40 + c) as u64));
        // No smoothness: the input back (the logit's cut aside).
        for s in [0.0, 1e-12] {
            let out = smooth_probs(&p, nk, nr, nc, 5, &[s; 3], 0.5, 1);
            for (a, b) in out.iter().zip(&p) {
                assert!((a - b).abs() < 1e-6, "{s}: {a} vs {b}");
            }
        }
        // Huge smoothness: the logistic of the mean of the highest logits.
        let out = smooth_probs(&p, nk, nr, nc, 3, &[1e12; 3], 0.5, 1);
        let n_pix = nr * nc;
        for r in 0..nr {
            for c in 0..nc {
                let mut pos = Vec::new();
                for dr in -1isize..=1 {
                    for dc in -1isize..=1 {
                        pos.push(mirror(r as isize + dr, nr, false) * nc + mirror(c as isize + dc, nc, false));
                    }
                }
                for k in 0..nk {
                    let mut v: Vec<f64> = pos.iter().map(|&j| (p[k * n_pix + j] / (1.0 - p[k * n_pix + j])).ln()).collect();
                    v.sort_by(|a, b| b.total_cmp(a));
                    let t = 5; // ceil(0.5 * 9)
                    let m = v[..t].iter().sum::<f64>() / t as f64;
                    let want = 1.0 / (1.0 + (-m).exp());
                    assert!((out[k * n_pix + r * nc + c] - want).abs() < 1e-9);
                }
            }
        }
    }

    #[test]
    fn corner_pixel_by_hand() {
        // 5 x 5, 2 classes, the first column is class 1, the rest class 0.
        let (nr, nc) = (5, 5);
        let p = cube(2, nr, nc, |k, _, c| {
            let in_first = c == 0;
            if (k == 1) == in_first {
                0.8
            } else {
                0.2
            }
        });
        let out = smooth_probs(&p, 2, nr, nc, 3, &[20.0, 20.0], 0.5, 1);
        // Corner (0, 0), window 3 with the edge repeated: rows {0, 0, 1}, columns
        // {0, 0, 1}, so 6 of the 9 values are in column 0 (the corner counted 4
        // times). The logits are +-ln 4 (class 1 at column 0 is +ln 4).
        let l = 4f64.ln();
        // Class 1: 6 values of +l, 3 of -l; t = ceil(4.5) = 5, all +l: the
        // variance is 0, so the result is the mean, 0.8.
        assert!((out[nr * nc] - 0.8).abs() < 1e-9, "{}", out[nr * nc]);
        // Class 0: 6 values of -l (column 0), 3 of +l; the 5 highest are 3 x +l
        // and 2 x -l; the pixel's own logit is -l.
        let m = l / 5.0;
        let s2 = (3.0 * (l - m).powi(2) + 2.0 * (-l - m).powi(2)) / 4.0;
        let mu = (s2 * -l + 20.0 * m) / (s2 + 20.0);
        let want0 = 1.0 / (1.0 + (-mu).exp());
        assert!((out[0] - want0).abs() < 1e-12, "{} vs {want0}", out[0]);
        // The naive version agrees at every pixel.
        for r in 0..nr {
            for c in 0..nc {
                let w = naive(&p, 2, nr, nc, 3, &[20.0, 20.0], 0.5, r, c);
                for k in 0..2 {
                    assert!((out[(k * nr + r) * nc + c] - w[k]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn images_smaller_than_the_window() {
        let one = smooth_probs(&[0.3, 0.7], 2, 1, 1, 9, &[20.0, 20.0], 0.5, 1);
        // The window of a 1 x 1 image is the pixel mirrored onto itself: its own
        // mean, back from the logit.
        assert!((one[0] - 0.3).abs() < 1e-12 && (one[1] - 0.7).abs() < 1e-12, "{one:?}");
        let p = cube(2, 2, 3, |k, r, c| if k == 0 { 0.2 + 0.1 * (r + c) as f64 } else { 0.5 });
        let out = smooth_probs(&p, 2, 2, 3, 9, &[20.0, 20.0], 0.5, 1);
        assert_eq!(out.len(), 12);
        assert!(out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)));
        for r in 0..2 {
            for c in 0..3 {
                let w = naive(&p, 2, 2, 3, 9, &[20.0; 2], 0.5, r, c);
                for k in 0..2 {
                    assert!((out[(k * 2 + r) * 3 + c] - w[k]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn mirrored_image_gives_the_mirrored_output() {
        let (nk, nr, nc) = (3, 9, 14);
        let p = cube(nk, nr, nc, |k, r, c| 0.02 + 0.96 * hash((k * 5000 + r * 50 + c) as u64));
        let flip = |v: &[f64]| cube(nk, nr, nc, |k, r, c| v[(k * nr + r) * nc + (nc - 1 - c)]);
        let a = smooth_probs(&flip(&p), nk, nr, nc, 5, &[10.0, 20.0, 40.0], 0.6, 1);
        let b = flip(&smooth_probs(&p, nk, nr, nc, 5, &[10.0, 20.0, 40.0], 0.6, 1));
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn nan_pixels() {
        let (nk, nr, nc) = (2, 9, 9);
        let base = cube(nk, nr, nc, |k, r, c| {
            let a = 0.1 + 0.8 * hash((r * 9 + c) as u64);
            if k == 0 {
                a
            } else {
                1.0 - a
            }
        });
        // A NaN in one class only.
        let mut p = base.clone();
        p[4 * nc + 4] = f64::NAN;
        let out = smooth_probs(&p, nk, nr, nc, 5, &[20.0; 2], 0.5, 1);
        let n_pix = nr * nc;
        assert!(out[4 * nc + 4].is_nan() && out[n_pix + 4 * nc + 4].is_nan());
        for r in 0..nr {
            for c in 0..nc {
                if (r, c) != (4, 4) {
                    let w = naive(&p, nk, nr, nc, 5, &[20.0; 2], 0.5, r, c);
                    for k in 0..nk {
                        assert!((out[k * n_pix + r * nc + c] - w[k]).abs() < 1e-12);
                        assert!(!out[k * n_pix + r * nc + c].is_nan());
                    }
                }
            }
        }
        // A pixel alone among NaN keeps its values.
        let mut q = vec![f64::NAN; nk * n_pix];
        q[3 * nc + 3] = 0.3;
        q[n_pix + 3 * nc + 3] = 0.6;
        let out = smooth_probs(&q, nk, nr, nc, 5, &[20.0; 2], 0.5, 1);
        assert_eq!(out[3 * nc + 3], 0.3);
        assert_eq!(out[n_pix + 3 * nc + 3], 0.6);
        assert_eq!(out.iter().filter(|v| !v.is_nan()).count(), 2);
        // All NaN.
        let all = vec![f64::NAN; nk * n_pix];
        assert!(smooth_probs(&all, nk, nr, nc, 5, &[20.0; 2], 0.5, 2).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn same_output_for_any_number_of_threads_and_layout() {
        // Not square, 3 classes.
        let (nk, nr, nc) = (3, 7, 11);
        let p = cube(nk, nr, nc, |k, r, c| 0.01 + 0.98 * hash((k * 777 + r * 31 + c) as u64));
        let one = smooth_probs(&p, nk, nr, nc, 5, &[10.0, 20.0, 40.0], 0.5, 1);
        for jobs in [2, 8] {
            let many = smooth_probs(&p, nk, nr, nc, 5, &[10.0, 20.0, 40.0], 0.5, jobs);
            assert_eq!(one.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), many.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        // Layout (class, row, column): a pixel against the naive version.
        let n_pix = nr * nc;
        for (r, c) in [(0, 10), (6, 0), (3, 5)] {
            let w = naive(&p, nk, nr, nc, 5, &[10.0, 20.0, 40.0], 0.5, r, c);
            for k in 0..nk {
                assert!((one[k * n_pix + r * nc + c] - w[k]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invariants_and_homogeneous_regions() {
        let (nk, nr, nc) = (3, 30, 30);
        // Class 0 in the left half, class 2 in the right, with noise.
        let p = cube(nk, nr, nc, |k, r, c| {
            let dom = if c < 15 { 0 } else { 2 };
            let noise = 0.1 * hash((k * 9000 + r * 100 + c) as u64);
            if k == dom {
                0.7 + noise
            } else {
                0.15 + noise / 2.0
            }
        });
        let out = smooth_probs(&p, nk, nr, nc, 7, &[20.0; 3], 0.5, 2);
        assert!(out.iter().all(|v| (0.0..=1.0).contains(v)));
        for r in 0..nr {
            for c in 0..nc {
                if !(13..17).contains(&c) {
                    let dom = if c < 15 { 0 } else { 2 };
                    assert_eq!(argmax(&out, nk, nr * nc, r * nc + c), dom);
                }
            }
        }
    }
}
