//! The area of applicability of a model (Meyer & Pebesma 2021), as CAST's `trainDI` and `aoa` (CAST is GPL >= 2):
//! the predictors scaled by the training data's means and standard deviations and weighted (by their importance),
//! each training point's dissimilarity index (DI) is its Euclidean distance to the nearest training point of another
//! cross-validation fold over the mean distance between the training points, and the threshold the upper whisker of
//! those (the 75% quantile plus 1.5 IQR, at most their largest). A new point's DI is its distance to the nearest
//! training point over the same mean; it is in the area of applicability when its DI is not above the threshold. Its
//! local point density (LPD) is how many training points are that near. The nearest training points are found in a
//! k-d tree up to 9 predictors, from a matrix product beyond; the distances measured exactly either way.
//!
//! The Mahalanobis distance (CAST's `method = "MD"`) is the Euclidean one once the scaled data are whitened by the
//! training data's covariance (x -> L^-1 x, L its Cholesky factor): the training points are whitened once and every
//! new point as it is scaled, so the search is the same. The weights then change nothing but which predictors are
//! used: the distance does not change with the predictors' scale.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rayon::prelude::*;

use crate::parallel;

/// How the distance between two points is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Mahalanobis,
}

/// Training points left out as neighbours of each training point besides those of its fold (the points near it in
/// a buffered leave-one-out, CAST's `CVtrain`), by row: point i's are `indices[indptr[i]..indptr[i + 1]]`.
#[derive(Clone, Copy, Debug)]
pub struct Exclude<'a> {
    pub indptr: &'a [usize],
    pub indices: &'a [usize],
}

/// What `trainDI` keeps of the training data.
#[derive(Clone, Debug)]
pub struct TrainDi {
    pub n: usize,
    pub p: usize,
    /// the scaling (each predictor's mean and standard deviation) and the weights
    pub center: Vec<f64>,
    pub scale: Vec<f64>,
    pub weight: Vec<f64>,
    /// for the Mahalanobis distance, what whitens a scaled and weighted point (p x p, row-major, lower triangular)
    pub whiten: Option<Vec<f64>>,
    /// the training data scaled and weighted, and whitened for the Mahalanobis distance (n x p)
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

pub(crate) fn distance2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// L^-1 (p x p, row-major) of the Cholesky factor L of the covariance of the points `x` (n x p), the predictors of
/// weight 0 (all 0 in x) left as they are.
fn whitening(x: &[f64], n: usize, p: usize, weight: &[f64]) -> Result<Vec<f64>, String> {
    let mean: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| x[i * p + j]).sum::<f64>() / n as f64)
        .collect();
    let mut s = vec![0.0; p * p];
    for a in 0..p {
        for b in 0..=a {
            let c = if weight[a] == 0.0 || weight[b] == 0.0 {
                if a == b {
                    1.0
                } else {
                    0.0
                }
            } else {
                (0..n)
                    .map(|i| (x[i * p + a] - mean[a]) * (x[i * p + b] - mean[b]))
                    .sum::<f64>()
                    / (n - 1) as f64
            };
            s[a * p + b] = c;
            s[b * p + a] = c;
        }
    }
    // Cholesky, s = L L'
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let d = s[j * p + j] - (0..j).map(|k| l[j * p + k] * l[j * p + k]).sum::<f64>();
        if d <= 1e-10 * s[j * p + j] {
            return Err(format!(
                "predictor {j} is a linear combination of the others in the training data: the Mahalanobis \
                 distance needs a covariance of full rank (leave it out, or use the Euclidean distance)"
            ));
        }
        l[j * p + j] = d.sqrt();
        for i in j + 1..p {
            let v = s[i * p + j] - (0..j).map(|k| l[i * p + k] * l[j * p + k]).sum::<f64>();
            l[i * p + j] = v / l[j * p + j];
        }
    }
    // its inverse, column by column (forward substitution of the identity)
    let mut w = vec![0.0; p * p];
    for c in 0..p {
        for i in c..p {
            let rhs = if i == c { 1.0 } else { 0.0 };
            let v = rhs - (c..i).map(|k| l[i * p + k] * w[k * p + c]).sum::<f64>();
            w[i * p + c] = v / l[i * p + i];
        }
    }
    Ok(w)
}

/// `row` whitened in place by `w` (lower triangular): from the last predictor up, as each needs only the earlier
/// ones.
fn apply_whiten(w: &[f64], row: &mut [f64]) {
    let p = row.len();
    for i in (0..p).rev() {
        row[i] = (0..=i).map(|k| w[i * p + k] * row[k]).sum();
    }
}

/// CAST's trainDI of the training data `train` (n points x p predictors, no missing value) with the predictors'
/// `weight`s (negative ones taken as 0) and, optionally, each point's cross-validation fold (`folds`, a negative
/// fold: the point in none, its DI missing); without folds, the nearest of every other point. `exclude`: the points
/// each one's DI leaves out besides those of its fold.
#[allow(clippy::too_many_arguments)]
pub fn train_di(
    train: &[f64],
    n: usize,
    p: usize,
    weight: &[f64],
    folds: Option<&[i64]>,
    exclude: Option<Exclude>,
    distance: Distance,
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
    // each point's left-out neighbours, sorted
    let exclude: Option<Vec<Vec<usize>>> = match exclude {
        None => None,
        Some(e) => {
            if e.indptr.len() != n + 1 || e.indptr.windows(2).any(|w| w[0] > w[1]) || e.indptr[n] > e.indices.len() {
                return Err(format!("exclude needs {} increasing offsets into its indices", n + 1));
            }
            if e.indices.iter().any(|&j| j >= n) {
                return Err(format!("exclude names a point beyond the {n} training points"));
            }
            Some(
                (0..n)
                    .map(|i| {
                        let mut v = e.indices[e.indptr[i]..e.indptr[i + 1]].to_vec();
                        v.sort_unstable();
                        v
                    })
                    .collect(),
            )
        }
    };
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
    let mut scaled: Vec<f64> = (0..n * p)
        .map(|k| (train[k] - center[k % p]) / scale[k % p] * weight[k % p])
        .collect();
    let whiten = match distance {
        Distance::Euclidean => None,
        Distance::Mahalanobis => {
            let w = whitening(&scaled, n, p, &weight)?;
            for row in scaled.chunks_mut(p) {
                apply_whiten(&w, row);
            }
            Some(w)
        }
    };
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
                let other = folds.is_none_or(|f| f[j] >= 0 && f[j] != f[i])
                    && exclude.as_ref().is_none_or(|e| e[i].binary_search(&j).is_err());
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
        return Err("no training point has a neighbour in another fold (or beyond its buffer)".into());
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
        whiten,
        train: scaled,
        mean_distance,
        train_di,
        threshold,
    })
}

/// Pixels a block of the new data's distances.
const BLOCK: usize = 256;

/// Up to this many predictors, the nearest training points are searched in a k-d tree (as CAST's FNN); beyond, the
/// distances to all of them come from a matrix product.
pub(crate) const KD_MAX_P: usize = 9;

/// Training points a leaf of the k-d tree.
pub(crate) const LEAF: usize = 12;

/// A k-d tree of the training points: each node splits its points at the median of their widest predictor.
pub(crate) struct KdTree<'a> {
    pts: &'a [f64],
    p: usize,
    /// the points' indices, each node's a range of them
    idx: Vec<usize>,
    nodes: Vec<KdNode>,
}

struct KdNode {
    lo: usize,
    hi: usize,
    dim: usize,
    split: f64,
    /// None: a leaf
    children: Option<(usize, usize)>,
}

impl<'a> KdTree<'a> {
    pub(crate) fn new(pts: &'a [f64], n: usize, p: usize) -> Self {
        let mut t = KdTree {
            pts,
            p,
            idx: (0..n).collect(),
            nodes: Vec::new(),
        };
        t.build(0, n);
        t
    }

    fn build(&mut self, lo: usize, hi: usize) -> usize {
        let id = self.nodes.len();
        self.nodes.push(KdNode {
            lo,
            hi,
            dim: 0,
            split: 0.0,
            children: None,
        });
        if hi - lo > LEAF {
            let (pts, p) = (self.pts, self.p);
            let spread = |j: usize| {
                let (mn, mx) = self.idx[lo..hi]
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &i| {
                        (a.min(pts[i * p + j]), b.max(pts[i * p + j]))
                    });
                mx - mn
            };
            let dim = (1..p).fold(0, |best, j| if spread(j) > spread(best) { j } else { best });
            // the points before the median at most its value, those after at least
            let mid = (lo + hi) / 2;
            self.idx[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| pts[a * p + dim].total_cmp(&pts[b * p + dim]));
            let split = pts[self.idx[mid] * p + dim];
            let left = self.build(lo, mid);
            let right = self.build(mid, hi);
            let node = &mut self.nodes[id];
            (node.dim, node.split, node.children) = (dim, split, Some((left, right)));
        }
        id
    }

    fn point(&self, i: usize) -> &[f64] {
        &self.pts[i * self.p..(i + 1) * self.p]
    }

    /// Lowers `best` to the squared distance from x to the nearest point under `node` if nearer: the exact
    /// distance2, a subtree left out when the distance to its box
    /// is above `best` (Arya & Mount's incremental distance: `rd` the squared distance to the node's box, `off` its
    /// part along each predictor).
    pub(crate) fn nearest(&self, x: &[f64], node: usize, rd: f64, off: &mut [f64], best: &mut f64) {
        let nd = &self.nodes[node];
        match nd.children {
            None => {
                for &i in &self.idx[nd.lo..nd.hi] {
                    *best = best.min(distance2(x, self.point(i)));
                }
            }
            Some((left, right)) => {
                let diff = x[nd.dim] - nd.split;
                let (near, far) = if diff < 0.0 { (left, right) } else { (right, left) };
                self.nearest(x, near, rd, off, best);
                let old = off[nd.dim];
                let rd_far = rd - old * old + diff * diff;
                // a margin for rounding: a point as near as the best is never left out
                if rd_far * (1.0 - 1e-12) <= *best {
                    off[nd.dim] = diff;
                    self.nearest(x, far, rd_far, off, best);
                    off[nd.dim] = old;
                }
            }
        }
    }

    /// As `nearest`, among the points `keep` takes (the points of another fold, for kNNDM).
    pub(crate) fn nearest_where(
        &self,
        x: &[f64],
        node: usize,
        rd: f64,
        off: &mut [f64],
        best: &mut f64,
        keep: &dyn Fn(usize) -> bool,
    ) {
        let nd = &self.nodes[node];
        match nd.children {
            None => {
                for &i in self.idx[nd.lo..nd.hi].iter().filter(|&&i| keep(i)) {
                    *best = best.min(distance2(x, self.point(i)));
                }
            }
            Some((left, right)) => {
                let diff = x[nd.dim] - nd.split;
                let (near, far) = if diff < 0.0 { (left, right) } else { (right, left) };
                self.nearest_where(x, near, rd, off, best, keep);
                let old = off[nd.dim];
                let rd_far = rd - old * old + diff * diff;
                if rd_far * (1.0 - 1e-12) <= *best {
                    off[nd.dim] = diff;
                    self.nearest_where(x, far, rd_far, off, best, keep);
                    off[nd.dim] = old;
                }
            }
        }
    }

    /// The points under `node` whose squared distance to x is `inside`, those beyond `reach2` (with a margin)
    /// left out.
    fn count(&self, x: &[f64], node: usize, reach2: f64, inside: &dyn Fn(f64) -> bool) -> usize {
        let nd = &self.nodes[node];
        match nd.children {
            None => self.idx[nd.lo..nd.hi]
                .iter()
                .filter(|&&i| inside(distance2(x, self.point(i))))
                .count(),
            Some((left, right)) => {
                let diff = x[nd.dim] - nd.split;
                let (near, far) = if diff < 0.0 { (left, right) } else { (right, left) };
                let mut c = self.count(x, near, reach2, inside);
                if diff * diff <= reach2 * (1.0 + 1e-6) {
                    c += self.count(x, far, reach2, inside);
                }
                c
            }
        }
    }
}

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
    let tree = (p <= KD_MAX_P && n > LEAF).then(|| KdTree::new(&td.train, n, p));
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
                if let Some(w) = &td.whiten {
                    // a missing predictor of weight 0 stays out: its row and column of w are the identity's
                    for v in row.iter_mut().filter(|v| !v.is_finite()) {
                        *v = 0.0;
                    }
                    apply_whiten(w, row);
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
        if let Some(tree) = &tree {
            let mut off = vec![0.0; p];
            for r in (0..m).filter(|&r| ok[r]) {
                let x = &a[r * p..(r + 1) * p];
                let mut best = f64::INFINITY;
                off.fill(0.0);
                tree.nearest(x, 0, 0.0, &mut off, &mut best);
                out[r] = best.sqrt() / td.mean_distance;
                if !out_lpd.is_empty() {
                    // CAST: knnDI < threshold, strictly
                    let inside = |d2: f64| d2.sqrt() / td.mean_distance < td.threshold;
                    out_lpd[r] = tree.count(x, 0, reach * reach, &inside) as f64;
                }
            }
            return;
        }
        // |a - b|² = |a|² + |b|² - 2 a.b, then the nearest candidates measured exactly
        // n x m, column-major: each pixel's products with the training points contiguous
        let mut g = Mat::<f64>::zeros(n, m);
        matmul(
            g.as_mut(),
            Accum::Replace,
            train,
            MatRef::from_row_major_slice(&a, m, p).transpose(),
            1.0,
            Par::Seq,
        );
        for r in (0..m).filter(|&r| ok[r]) {
            let x = &a[r * p..(r + 1) * p];
            let an: f64 = x.iter().map(|v| v * v).sum();
            let tol = 1e-9 * (an + top) + 1e-300;
            let col = g.col(r).try_as_col_major().unwrap().as_slice();
            let approx = |j: usize| an + norms[j] - 2.0 * col[j];
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
        train_di(
            &[0.0, 1.0, 2.0, 3.0, 10.0],
            5,
            1,
            &[1.0],
            None,
            None,
            Distance::Euclidean,
            1,
        )
        .unwrap()
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
        for (got, min) in td.train_di.iter().zip(mins) {
            assert!((got - min / mean_raw).abs() < 1e-12);
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
        let td = train_di(&t, 4, 2, &[1.0, 1.0], Some(&[1, 1, 2, 2]), None, Distance::Euclidean, 1).unwrap();
        assert!(td.train_di.iter().all(|&v| v > 1.0));
        let td = train_di(
            &t,
            4,
            2,
            &[1.0, 1.0],
            Some(&[1, 2, 1, -1]),
            None,
            Distance::Euclidean,
            1,
        )
        .unwrap();
        assert!(td.train_di[3].is_nan() && td.train_di[0] < 0.1);
        // a predictor of weight 0 is left out, even without variance
        let t = [0.0, 7.0, 1.0, 7.0, 3.0, 7.0];
        let td = train_di(&t, 3, 2, &[2.0, 0.0], None, None, Distance::Euclidean, 1).unwrap();
        let (d, _) = di(&[1.0, f64::NAN], 1, &td, false, 1);
        assert!(d[0] == 0.0, "{d:?}");
        assert!(train_di(&t, 3, 2, &[1.0, 1.0], None, None, Distance::Euclidean, 1).is_err());
    }

    #[test]
    fn tree_is_the_brute_search() {
        // 5 predictors (the k-d tree) and the same points padded with 10 of weight 0 (the matrix product)
        let (n, p, m) = (500, 5, 2000);
        let mut s = 7u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let t: Vec<f64> = (0..n * p).map(|k| rnd() * 10.0 + (k / p % 4) as f64 * 20.0).collect();
        // half the new points near training points, half anywhere
        let x: Vec<f64> = (0..m * p)
            .map(|k| {
                if k / p % 2 == 0 {
                    t[(k / p / 2 % n) * p + k % p] + rnd()
                } else {
                    rnd() * 80.0
                }
            })
            .collect();
        let pad = |v: &[f64], rows: usize| -> Vec<f64> {
            (0..rows)
                .flat_map(|i| {
                    v[i * p..(i + 1) * p]
                        .iter()
                        .copied()
                        .chain((0..10).map(move |j| j as f64 + i as f64))
                })
                .collect()
        };
        let w: Vec<f64> = [1.0, 2.0, 0.5, 1.0, 3.0].into_iter().chain([0.0; 10]).collect();
        let td = train_di(&t, n, p, &w[..p], None, None, Distance::Euclidean, 1).unwrap();
        let tdp = train_di(&pad(&t, n), n, p + 10, &w, None, None, Distance::Euclidean, 1).unwrap();
        assert_eq!(td.mean_distance, tdp.mean_distance);
        let (d, l) = di(&x, m, &td, true, 1);
        let (dp, lp) = di(&pad(&x, m), m, &tdp, true, 1);
        assert_eq!((&d, &l), (&dp, &lp));
        assert!(l.iter().any(|&v| v > 0.0) && l.contains(&0.0));
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
        let td = train_di(
            &t,
            n,
            p,
            &[1.0, 2.0, 0.5, 1.0, 1.0, 3.0, 1.0],
            None,
            None,
            Distance::Euclidean,
            4,
        )
        .unwrap();
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

    /// d² = (a - b)' S⁻¹ (a - b) of the scaled points, S their covariance, solved by Gaussian elimination.
    fn mahalanobis2(a: &[f64], b: &[f64], s: &[f64], p: usize) -> f64 {
        let mut m: Vec<f64> = s.to_vec();
        let mut v: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        let diff = v.clone();
        for c in 0..p {
            for r in c + 1..p {
                let f = m[r * p + c] / m[c * p + c];
                for k in c..p {
                    m[r * p + k] -= f * m[c * p + k];
                }
                v[r] -= f * v[c];
            }
        }
        for c in (0..p).rev() {
            v[c] = (v[c] - (c + 1..p).map(|k| m[c * p + k] * v[k]).sum::<f64>()) / m[c * p + c];
        }
        diff.iter().zip(&v).map(|(d, x)| d * x).sum()
    }

    #[test]
    fn mahalanobis_is_the_whitened_euclidean() {
        let (n, p, m) = (80, 3, 50);
        let mut s = 3u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        // correlated predictors of different scales
        let mut t = vec![0.0; n * p];
        for i in 0..n {
            let (a, b, c) = (rnd(), rnd(), rnd());
            t[i * p] = a;
            t[i * p + 1] = 0.8 * a + 0.3 * b;
            t[i * p + 2] = 5.0 * c;
        }
        let x: Vec<f64> = (0..m * p).map(|_| 2.0 * rnd()).collect();
        let td = train_di(&t, n, p, &[1.0, 1.0, 1.0], None, None, Distance::Mahalanobis, 1).unwrap();
        // by the definition, on the data scaled
        let z = |v: &[f64]| -> Vec<f64> { (0..p).map(|j| (v[j] - td.center[j]) / td.scale[j]).collect() };
        let zt: Vec<Vec<f64>> = (0..n).map(|i| z(&t[i * p..(i + 1) * p])).collect();
        let mut cov = vec![0.0; p * p];
        for a in 0..p {
            for b in 0..p {
                cov[a * p + b] = zt.iter().map(|r| r[a] * r[b]).sum::<f64>() / (n - 1) as f64;
            }
        }
        let mean: f64 = (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| mahalanobis2(&zt[i], &zt[j], &cov, p).sqrt())
                    .sum::<f64>()
            })
            .sum::<f64>()
            / (n * (n - 1)) as f64;
        assert!((td.mean_distance - mean).abs() < 1e-12 * mean);
        let (d, _) = di(&x, m, &td, false, 1);
        for k in 0..m {
            let zx = z(&x[k * p..(k + 1) * p]);
            let min = zt
                .iter()
                .map(|r| mahalanobis2(&zx, r, &cov, p))
                .fold(f64::INFINITY, f64::min);
            assert!(
                (d[k] - min.sqrt() / mean).abs() < 1e-10,
                "{k}: {} {}",
                d[k],
                min.sqrt() / mean
            );
        }
        // the weights change nothing: the distance does not change with the predictors' scale
        let tw = train_di(&t, n, p, &[2.0, 0.5, 3.0], None, None, Distance::Mahalanobis, 1).unwrap();
        let (dw, _) = di(&x, m, &tw, false, 1);
        assert!(d.iter().zip(&dw).all(|(a, b)| (a - b).abs() < 1e-10));
        assert!((td.threshold - tw.threshold).abs() < 1e-10);
        // the same point twice in another predictor: the covariance has not full rank
        let twin: Vec<f64> = (0..n).flat_map(|i| [t[i * p], t[i * p + 1], 2.0 * t[i * p]]).collect();
        assert!(train_di(&twin, n, p, &[1.0; 3], None, None, Distance::Mahalanobis, 1).is_err());
    }

    #[test]
    fn excluded_neighbours_are_left_out() {
        // points on a line 0..5: without its neighbours within 1, a point's nearest is 2 away
        let t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut indptr = vec![0];
        let mut indices = vec![];
        for i in 0..6usize {
            indices.extend((0..6).filter(|&j| j != i && (i as f64 - j as f64).abs() <= 1.0));
            indptr.push(indices.len());
        }
        let ex = Exclude {
            indptr: &indptr,
            indices: &indices,
        };
        // each point its own fold: a buffered leave-one-out
        let folds: Vec<i64> = (0..6).collect();
        let plain = train_di(&t, 6, 1, &[1.0], Some(&folds), None, Distance::Euclidean, 1).unwrap();
        let buffered = train_di(&t, 6, 1, &[1.0], Some(&folds), Some(ex), Distance::Euclidean, 1).unwrap();
        for (a, b) in plain.train_di.iter().zip(&buffered.train_di) {
            assert!((b / a - 2.0).abs() < 1e-12);
        }
        let bad = Exclude {
            indptr: &indptr[..3],
            indices: &indices,
        };
        assert!(train_di(&t, 6, 1, &[1.0], None, Some(bad), Distance::Euclidean, 1).is_err());
    }
}
