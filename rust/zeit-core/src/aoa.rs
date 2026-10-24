//! The area of applicability of a model (Meyer & Pebesma 2021), as CAST's `trainDI` and `aoa` (CAST is GPL >= 2):
//! the predictors scaled by the training data's means and standard deviations and weighted (by their importance),
//! each training point's dissimilarity index (DI) is its Euclidean distance to the nearest training point of another
//! cross-validation fold over the mean distance between the training points, and the threshold the upper whisker of
//! those (the 75% quantile plus 1.5 IQR, at most their largest). A new point's DI is its distance to the nearest
//! training point over the same mean; it is in the area of applicability when its DI is not above the threshold. Its
//! local point density (LPD) is how many training points are that near.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rayon::prelude::*;

use crate::parallel;

/// What `trainDI` keeps of the training data.
#[derive(Clone, Debug)]
pub struct TrainDi {
    pub n: usize,
    pub p: usize,
    /// the scaling (each predictor's mean and standard deviation) and the weights
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
    pub weight: Vec<f64>,
    /// the training data scaled and weighted (n x p)
    pub train: Vec<f64>,
    /// the mean distance between training points (CAST's trainDist_avrgmean)
    pub mean_distance: f64,
    /// each training point's DI (NaN without a fold, or no point of another)
    pub train_di: Vec<f64>,
    pub threshold: f64,
}

/// R's quantile of type 7 of sorted values.
fn quantile7(sorted: &[f64], q: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * q;
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

fn distance2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// CAST's trainDI of the training data `train` (n points x p predictors, no missing value) with the predictors'
/// `weight`s (negative ones taken as 0) and, optionally, each point's cross-validation fold (`folds`, a negative
/// fold: the point in none, its DI missing); without folds, the nearest of every other point.
pub fn train_di(
    train: &[f64],
    n: usize,
    p: usize,
    weight: &[f64],
    folds: Option<&[i64]>,
    n_jobs: i32,
) -> Result<TrainDi, String> {
    if train.len() != n * p {
        return Err(format!(
            "train has {} values for {n} points of {p} predictors",
            train.len()
        ));
    }
    if weight.len() != p {
        return Err(format!("{} weights for {p} predictors", weight.len()));
    }
    if n < 2 {
        return Err("at least two training points are needed".into());
    }
    if train.iter().any(|v| !v.is_finite()) {
        return Err("the training data have missing values".into());
    }
    if let Some(f) = folds.filter(|f| f.len() != n) {
        return Err(format!("{} folds for {n} training points", f.len()));
    }
    let weight: Vec<f64> = weight.iter().map(|&w| if w > 0.0 { w } else { 0.0 }).collect();
    if weight.iter().all(|&w| w == 0.0) {
        return Err("all weights are <= 0, hence no predictor is used".into());
    }
    let mut center = vec![0.0; p];
    let mut scale = vec![1.0; p];
    for j in 0..p {
        let m = (0..n).map(|i| train[i * p + j]).sum::<f64>() / n as f64;
        let s = ((0..n).map(|i| (train[i * p + j] - m).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
        center[j] = m;
        if s > 0.0 && s.is_finite() {
            scale[j] = s;
        } else if weight[j] > 0.0 {
            return Err(format!("predictor {j} has no variance in the training data"));
        }
    }
    let scaled: Vec<f64> = (0..n * p)
        .map(|k| (train[k] - center[k % p]) / scale[k % p] * weight[k % p])
        .collect();
    // each point's mean distance to the others, and to the nearest of another fold
    let mut rows = vec![f64::NAN; 2 * n];
    parallel::for_each_row(
        &mut rows,
        2,
        n_jobs,
        || (),
        |_, i, out| {
            let a = &scaled[i * p..(i + 1) * p];
            let (mut sum, mut min) = (0.0, f64::INFINITY);
            for j in (0..n).filter(|&j| j != i) {
                let d = distance2(a, &scaled[j * p..(j + 1) * p]).sqrt();
                sum += d;
                let other = folds.is_none_or(|f| f[j] >= 0 && f[j] != f[i]);
                if other && d < min {
                    min = d;
                }
            }
            out[0] = sum / (n - 1) as f64;
            out[1] = if folds.is_some_and(|f| f[i] < 0) || min.is_infinite() {
                f64::NAN
            } else {
                min
            };
        },
    );
    let mean_distance = (0..n).map(|i| rows[2 * i]).sum::<f64>() / n as f64;
    let train_di: Vec<f64> = (0..n).map(|i| rows[2 * i + 1] / mean_distance).collect();
    let mut sorted: Vec<f64> = train_di.iter().copied().filter(|v| !v.is_nan()).collect();
    if sorted.is_empty() {
        return Err("no training point has a neighbour in another fold".into());
    }
    sorted.sort_by(f64::total_cmp);
    let (q25, q75) = (quantile7(&sorted, 0.25), quantile7(&sorted, 0.75));
    let threshold = (q75 + 1.5 * (q75 - q25)).min(sorted[sorted.len() - 1]);
    Ok(TrainDi {
        n,
        p,
        center,
        scale,
        weight,
        train: scaled,
        mean_distance,
        train_di,
        threshold,
    })
}

/// Pixels a block of the new data's distances.
const BLOCK: usize = 256;

/// The DI of each new point of `data` (pixels x p, a missing value: NaN) and, with `lpd`, its local point density
/// (how many training points have a DI to it below the threshold; else empty).
pub fn di(data: &[f64], n_pixels: usize, td: &TrainDi, lpd: bool, n_jobs: i32) -> (Vec<f64>, Vec<f64>) {
    let (n, p) = (td.n, td.p);
    assert_eq!(data.len(), n_pixels * p, "data must be pixels x predictors");
    let train = MatRef::from_row_major_slice(&td.train, n, p);
    let norms: Vec<f64> = (0..n)
        .map(|j| td.train[j * p..(j + 1) * p].iter().map(|v| v * v).sum())
        .collect();
    let top = norms.iter().copied().fold(0.0, f64::max);
    let reach = td.threshold * td.mean_distance;
    let mut di = vec![f64::NAN; n_pixels];
    let mut lp = vec![f64::NAN; if lpd { n_pixels } else { 0 }];
    let work = |k0: usize, out: &mut [f64], out_lpd: &mut [f64]| {
        let m = out.len();
        let mut a = vec![0.0; m * p];
        let ok: Vec<bool> = (0..m)
            .map(|r| {
                let x = &data[(k0 + r) * p..(k0 + r + 1) * p];
                let row = &mut a[r * p..(r + 1) * p];
                for j in 0..p {
                    row[j] = (x[j] - td.center[j]) / td.scale[j] * td.weight[j];
                }
                // a predictor of weight 0 may be missing
                x.iter().zip(&td.weight).all(|(v, w)| v.is_finite() || *w == 0.0)
            })
            .collect();
        for (r, &good) in ok.iter().enumerate() {
            if !good {
                a[r * p..(r + 1) * p].fill(0.0);
            } else {
                for v in &mut a[r * p..(r + 1) * p] {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }
            }
        }
        // |a - b|² = |a|² + |b|² - 2 a.b, then the nearest candidates measured exactly
        let mut g = Mat::<f64>::zeros(m, n);
        matmul(
            g.as_mut(),
            Accum::Replace,
            MatRef::from_row_major_slice(&a, m, p),
            train.transpose(),
            1.0,
            Par::Seq,
        );
        for r in (0..m).filter(|&r| ok[r]) {
            let x = &a[r * p..(r + 1) * p];
            let an: f64 = x.iter().map(|v| v * v).sum();
            let tol = 1e-9 * (an + top) + 1e-300;
            let approx = |j: usize| an + norms[j] - 2.0 * g[(r, j)];
            let best = (0..n).map(approx).fold(f64::INFINITY, f64::min);
            let mut min = f64::INFINITY;
            for j in (0..n).filter(|&j| approx(j) <= best + tol) {
                min = min.min(distance2(x, &td.train[j * p..(j + 1) * p]));
            }
            out[r] = min.sqrt() / td.mean_distance;
            if !out_lpd.is_empty() {
                let r2 = reach * reach;
                // CAST: knnDI < threshold, strictly
                out_lpd[r] = (0..n)
                    .filter(|&j| {
                        let d = approx(j);
                        if (d - r2).abs() <= tol {
                            distance2(x, &td.train[j * p..(j + 1) * p]).sqrt() / td.mean_distance < td.threshold
                        } else {
                            d < r2
                        }
                    })
                    .count() as f64;
            }
        }
    };
    let threads = parallel::threads(n_jobs);
    let blocks: Vec<(usize, &mut [f64])> = di.chunks_mut(BLOCK).enumerate().map(|(b, c)| (b * BLOCK, c)).collect();
    let mut lp_blocks: Vec<&mut [f64]> = if lpd {
        lp.chunks_mut(BLOCK).collect()
    } else {
        (0..blocks.len()).map(|_| &mut [][..]).collect()
    };
    let jobs: Vec<_> = blocks.into_iter().zip(lp_blocks.drain(..)).collect();
    if threads == 1 {
        for ((k0, out), out_lpd) in jobs {
            work(k0, out, out_lpd);
        }
    } else {
        parallel::pool(threads).install(|| {
            jobs.into_par_iter()
                .for_each(|((k0, out), out_lpd)| work(k0, out, out_lpd));
        });
    }
    (di, lp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Points on a line: 0, 1, 2, 3, 10 (one predictor).
    fn line() -> TrainDi {
        train_di(&[0.0, 1.0, 2.0, 3.0, 10.0], 5, 1, &[1.0], None, 1).unwrap()
    }

    #[test]
    fn train_di_by_hand() {
        let td = line();
        // scaled by the sd, so the distances' ratios are the raw ones'
        let raw = [
            [1.0, 2.0, 3.0, 10.0],
            [1.0, 1.0, 2.0, 9.0],
            [2.0, 1.0, 1.0, 8.0],
            [3.0, 2.0, 1.0, 7.0],
            [10.0, 9.0, 8.0, 7.0],
        ];
        let mean_raw = raw.iter().map(|r| r.iter().sum::<f64>() / 4.0).sum::<f64>() / 5.0;
        let mins = [1.0, 1.0, 1.0, 1.0, 7.0];
        for i in 0..5 {
            assert!((td.train_di[i] - mins[i] / mean_raw).abs() < 1e-12);
        }
        // the whisker (1/m + 1.5 * 0) is below the largest: the threshold
        assert!((td.threshold - 1.0 / mean_raw).abs() < 1e-12);
        let (d, l) = di(&[1.5, 10.0, f64::NAN, -4.0], 4, &td, true, 1);
        assert!((d[0] - 0.5 / mean_raw).abs() < 1e-12 && d[1] == 0.0 && d[2].is_nan());
        assert!((d[3] - 4.0 / mean_raw).abs() < 1e-12);
        // within the threshold's distance (1): 1 and 2 of 1.5; 10 of 10
        assert_eq!((l[0], l[1], l[3]), (2.0, 1.0, 0.0));
    }

    #[test]
    fn folds_and_weights() {
        // two clusters, each its fold: a point's nearest of the other fold is far
        let t = [0.0, 0.0, 0.1, 0.0, 5.0, 5.0, 5.1, 5.0];
        let td = train_di(&t, 4, 2, &[1.0, 1.0], Some(&[1, 1, 2, 2]), 1).unwrap();
        assert!(td.train_di.iter().all(|&v| v > 1.0));
        let td = train_di(&t, 4, 2, &[1.0, 1.0], Some(&[1, 2, 1, -1]), 1).unwrap();
        assert!(td.train_di[3].is_nan() && td.train_di[0] < 0.1);
        // a predictor of weight 0 is left out, even without variance
        let t = [0.0, 7.0, 1.0, 7.0, 3.0, 7.0];
        let td = train_di(&t, 3, 2, &[2.0, 0.0], None, 1).unwrap();
        let (d, _) = di(&[1.0, f64::NAN], 1, &td, false, 1);
        assert!(d[0] == 0.0, "{d:?}");
        assert!(train_di(&t, 3, 2, &[1.0, 1.0], None, 1).is_err());
    }

    #[test]
    fn blocks_threads_and_exactness() {
        // many training points and pixels: the GEMM's distances refined to the exact ones
        let (n, p, m) = (300, 7, 1000);
        let mut s = 1u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 1000.0
        };
        let t: Vec<f64> = (0..n * p).map(|_| rnd()).collect();
        let mut x: Vec<f64> = (0..m * p).map(|_| rnd()).collect();
        x[..p].copy_from_slice(&t[5 * p..6 * p]);
        let td = train_di(&t, n, p, &[1.0, 2.0, 0.5, 1.0, 1.0, 3.0, 1.0], None, 4).unwrap();
        let (d1, l1) = di(&x, m, &td, true, 1);
        let (d4, l4) = di(&x, m, &td, true, 4);
        assert_eq!((&d1, &l1), (&d4, &l4));
        assert_eq!(d1[0], 0.0);
        for k in [1, 77, 999] {
            let a: Vec<f64> = (0..p)
                .map(|j| (x[k * p + j] - td.center[j]) / td.scale[j] * td.weight[j])
                .collect();
            let min = (0..n)
                .map(|j| distance2(&a, &td.train[j * p..(j + 1) * p]))
                .fold(f64::INFINITY, f64::min);
            assert_eq!(d1[k], min.sqrt() / td.mean_distance);
        }
    }
}
