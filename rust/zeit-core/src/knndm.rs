//! The nearest-neighbour distances of kNNDM (k-fold nearest neighbour distance matching, Linnenbrink et al. 2024,
//! doi:10.5194/gmd-17-5897-2024): from each prediction point to the nearest training point (Gij), from each training
//! point to the nearest one of another fold (Gj*, the distances a cross-validation sees), and the Wasserstein
//! statistic W between two of them, the integral of the absolute difference of their empirical distribution
//! functions. kNNDM keeps the folds whose Gj* is nearest Gij. The points are searched in the k-d tree of `aoa` up to
//! 9 dimensions (coordinates, or a few predictors), by brute force beyond.

use crate::aoa::{distance2, KdTree, KD_MAX_P, LEAF};
use crate::parallel;

/// The Euclidean distance from each of the `m` points `query` (m x p, row-major) to the nearest of the `n` points
/// `train` (n x p).
pub fn nearest(train: &[f64], n: usize, p: usize, query: &[f64], m: usize, n_jobs: i32) -> Vec<f64> {
    assert!(
        train.len() == n * p && query.len() == m * p,
        "train must be n x p and query m x p"
    );
    let mut out = vec![f64::NAN; m];
    if n == 0 {
        return out;
    }
    let tree = (p <= KD_MAX_P && n > LEAF).then(|| KdTree::new(train, n, p));
    parallel::for_each_row(
        &mut out,
        1,
        n_jobs,
        || vec![0.0; p],
        |off, i, o| {
            let x = &query[i * p..(i + 1) * p];
            let mut best = f64::INFINITY;
            match &tree {
                Some(t) => {
                    off.fill(0.0);
                    t.nearest(x, 0, 0.0, off, &mut best);
                }
                None => {
                    for j in 0..n {
                        best = best.min(distance2(x, &train[j * p..(j + 1) * p]));
                    }
                }
            }
            o[0] = best.sqrt();
        },
    );
    out
}

/// The Euclidean distance from each of the `n` points (n x p) to the nearest point of another fold (`folds`, one per
/// point; a negative fold: in none, neither measured nor a neighbour); NaN without one.
pub fn nearest_other(points: &[f64], n: usize, p: usize, folds: &[i64], n_jobs: i32) -> Vec<f64> {
    assert!(
        points.len() == n * p && folds.len() == n,
        "points must be n x p, with a fold each"
    );
    let mut out = vec![f64::NAN; n];
    let tree = (p <= KD_MAX_P && n > LEAF).then(|| KdTree::new(points, n, p));
    parallel::for_each_row(
        &mut out,
        1,
        n_jobs,
        || vec![0.0; p],
        |off, i, o| {
            let fi = folds[i];
            if fi < 0 {
                return;
            }
            let x = &points[i * p..(i + 1) * p];
            let keep = |j: usize| folds[j] >= 0 && folds[j] != fi;
            let mut best = f64::INFINITY;
            match &tree {
                Some(t) => {
                    off.fill(0.0);
                    t.nearest_where(x, 0, 0.0, off, &mut best, &keep);
                }
                None => {
                    for j in (0..n).filter(|&j| keep(j)) {
                        best = best.min(distance2(x, &points[j * p..(j + 1) * p]));
                    }
                }
            }
            if best.is_finite() {
                o[0] = best.sqrt();
            }
        },
    );
    out
}

/// W = ∫ |F_a(t) - F_b(t)| dt of the empirical distribution functions of `a` and `b` (NaN left out); NaN if either
/// is empty.
pub fn wasserstein(a: &[f64], b: &[f64]) -> f64 {
    let sorted = |v: &[f64]| {
        let mut s: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
        s.sort_by(f64::total_cmp);
        s
    };
    let (a, b) = (sorted(a), sorted(b));
    if a.is_empty() || b.is_empty() {
        return f64::NAN;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    // walk the values of both in order: between two consecutive ones, both functions are constant
    let (mut i, mut j, mut w) = (0, 0, 0.0);
    let mut last = a[0].min(b[0]);
    while i < a.len() || j < b.len() {
        let next = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) => x.min(y),
            (Some(&x), None) => x,
            (None, Some(&y)) => y,
            (None, None) => unreachable!(),
        };
        w += (i as f64 / na - j as f64 / nb).abs() * (next - last);
        while i < a.len() && a[i] == next {
            i += 1;
        }
        while j < b.len() && b[j] == next {
            j += 1;
        }
        last = next;
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect()
    }

    #[test]
    fn nearest_of_another_fold_is_the_brute_search() {
        for (n, p) in [(2000, 2), (300, 12)] {
            let pts: Vec<f64> = random(n * p, 3).iter().map(|v| v * 100.0).collect();
            let r = random(n, 5);
            let folds: Vec<i64> = (0..n)
                .map(|i| if i % 97 == 0 { -1 } else { (r[i] * 4.0) as i64 })
                .collect();
            let got = nearest_other(&pts, n, p, &folds, 4);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&nearest_other(&pts, n, p, &folds, 1)));
            for i in 0..n {
                let mut best = f64::INFINITY;
                for j in 0..n {
                    if folds[i] >= 0 && folds[j] >= 0 && folds[j] != folds[i] {
                        best = best.min(distance2(&pts[i * p..(i + 1) * p], &pts[j * p..(j + 1) * p]));
                    }
                }
                let want = if best.is_finite() { best.sqrt() } else { f64::NAN };
                assert!(
                    got[i] == want || (got[i].is_nan() && want.is_nan()),
                    "{i}: {} {want}",
                    got[i]
                );
            }
            // each point its own fold: the leave-one-out distance, the plain nearest other point
            let own: Vec<i64> = (0..n as i64).collect();
            let loo = nearest_other(&pts, n, p, &own, 2);
            let q: Vec<f64> = pts.iter().map(|v| v + 0.5).collect();
            let near = nearest(&pts, n, p, &q, n, 2);
            assert!(loo.iter().all(|v| v.is_finite() && *v > 0.0) && near.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn wasserstein_by_hand() {
        // a shifted sample: W is the shift
        let a = [1.0, 2.0, 3.0];
        let b = [3.0, 4.0, 5.0];
        assert!((wasserstein(&a, &b) - 2.0).abs() < 1e-15);
        assert_eq!(wasserstein(&a, &a), 0.0);
        // samples of different sizes: F_a = 0 below 0, 1/2 to 2, 1 beyond; F_b = 0 below 1, 1 beyond
        assert!((wasserstein(&[0.0, 2.0], &[1.0, f64::NAN]) - 1.0).abs() < 1e-15);
        assert!(wasserstein(&[], &a).is_nan());
    }
}
