//! Post-processing of classifications: the Bayesian smoothing of class probabilities of sits
//! (sits_smooth, Camara et al.; from sits' smooth_bayes.cpp, GPL-2), a majority filter, a minimum
//! mapping unit that merges small patches into the neighbouring class (a sieve), and the removal of
//! short excursions from the class series of pixels (A B A -> A A A). Missing values (NaN) stay
//! missing and take no part.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::parallel;

/// `i` mirrored into `0..size` as sits' locus_neigh: the edge pixel repeated (-1 -> 0, -2 -> 1,
/// size -> size - 1).
fn mirror(mut i: isize, size: isize) -> usize {
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= size {
            i = 2 * size - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// sits' Bayesian smoothing of the probabilities `probs` (class, row, col) of `n_classes` classes:
/// each class's logit against the others, log(p / (sum - p)), is pulled towards the mean of the
/// highest `neigh_fraction` of the logits of its `window` x `window` neighbourhood (mirrored at the
/// edges), as much as that neighbourhood varies little against `smoothness` (one per class):
/// w = s² / (s² + smoothness), x' = w x + (1 - w) m; a neighbourhood that hardly varies (s² < 1e-4)
/// gives its mean. The probabilities are given back from the logits (they no longer sum to one,
/// as sits'). A pixel missing in any class stays missing (sits gives it its neighbours' mean), and
/// a pixel alone in its window keeps its value.
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
    let n = nrows * ncols;
    assert_eq!(probs.len(), n_classes * n, "probs must be classes x rows x cols");
    assert_eq!(smoothness.len(), n_classes, "one smoothness per class");
    let mut logits = vec![f64::NAN; n_classes * n];
    let clamp = |v: f64| {
        if v == 1.0 {
            0.999999
        } else if v == 0.0 {
            1e-6
        } else {
            v
        }
    };
    for k in 0..n {
        let sum: f64 = (0..n_classes).map(|c| clamp(probs[c * n + k])).sum();
        for c in 0..n_classes {
            let v = clamp(probs[c * n + k]);
            logits[c * n + k] = (v / (sum - v)).ln();
        }
    }
    let leg = (window / 2) as isize;
    let (rows, cols) = (nrows as isize, ncols as isize);
    let mut out = vec![f64::NAN; n_classes * n];
    parallel::for_each_row(
        &mut out,
        ncols,
        n_jobs,
        || Vec::with_capacity(window * window),
        |neigh: &mut Vec<f64>, r, row| {
            let (c, i) = (r / nrows, (r % nrows) as isize);
            let band = &logits[c * n..(c + 1) * n];
            for (j, o) in row.iter_mut().enumerate() {
                let x0 = band[i as usize * ncols + j];
                if x0.is_nan() {
                    continue;
                }
                neigh.clear();
                for wi in -leg..=leg {
                    let ii = mirror(i + wi, rows) * ncols;
                    for wj in -leg..=leg {
                        let v = band[ii + mirror(j as isize + wj, cols)];
                        if !v.is_nan() {
                            neigh.push(v);
                        }
                    }
                }
                let k = ((neigh_fraction * neigh.len() as f64).ceil() as usize).clamp(1, neigh.len());
                if k < neigh.len() {
                    neigh.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
                }
                let high = &neigh[..k];
                let v = if k < 2 {
                    x0
                } else {
                    let m0 = high.iter().sum::<f64>() / k as f64;
                    let s0 = high.iter().map(|h| (h - m0) * (h - m0)).sum::<f64>() / (k - 1) as f64;
                    if s0 < 1e-4 {
                        m0
                    } else {
                        let w = s0 / (s0 + smoothness[c]);
                        w * x0 + (1.0 - w) * m0
                    }
                };
                *o = v.exp() / (v.exp() + 1.0);
            }
        },
    );
    out
}

/// The classes of a map as indices into its sorted distinct values (u32::MAX: missing).
fn class_index(classes: &[f64]) -> (Vec<f64>, Vec<u32>) {
    let mut values: Vec<f64> = classes.iter().copied().filter(|v| !v.is_nan()).collect();
    values.sort_by(f64::total_cmp);
    values.dedup();
    let index = classes
        .iter()
        .map(|v| {
            if v.is_nan() {
                u32::MAX
            } else {
                values.binary_search_by(|x| x.total_cmp(v)).unwrap() as u32
            }
        })
        .collect();
    (values, index)
}

/// Majority filter of a class map (rows x cols): each pixel takes the most frequent class of its
/// `window` x `window` neighbourhood (cut at the edges), its own class when it is among the most
/// frequent, else the smallest of them; `iterations` times, or until nothing changes. Missing
/// pixels stay missing and do not vote.
pub fn majority(
    classes: &[f64],
    nrows: usize,
    ncols: usize,
    window: usize,
    iterations: usize,
    n_jobs: i32,
) -> Vec<f64> {
    assert_eq!(classes.len(), nrows * ncols, "classes must be rows x cols");
    let (values, mut index) = class_index(classes);
    let nc = values.len();
    let leg = window / 2;
    for _ in 0..iterations {
        let mut next = vec![u32::MAX; nrows * ncols];
        parallel::for_each_row(
            &mut next,
            ncols,
            n_jobs,
            || vec![0u32; nc],
            |hist: &mut Vec<u32>, i, row| {
                hist.iter_mut().for_each(|h| *h = 0);
                let (r0, r1) = (i.saturating_sub(leg), (i + leg).min(nrows - 1));
                let column = |hist: &mut Vec<u32>, j: usize, add: bool| {
                    for r in r0..=r1 {
                        let c = index[r * ncols + j];
                        if c != u32::MAX {
                            if add {
                                hist[c as usize] += 1;
                            } else {
                                hist[c as usize] -= 1;
                            }
                        }
                    }
                };
                for j in 0..=leg.min(ncols - 1) {
                    column(hist, j, true);
                }
                for j in 0..ncols {
                    if j > 0 {
                        if j > leg {
                            column(hist, j - leg - 1, false);
                        }
                        if j + leg < ncols {
                            column(hist, j + leg, true);
                        }
                    }
                    let own = index[i * ncols + j];
                    if own == u32::MAX {
                        continue;
                    }
                    let mut best = own;
                    let mut top = hist[own as usize];
                    if nc <= window * window {
                        for (c, &h) in hist.iter().enumerate() {
                            if h > top {
                                (best, top) = (c as u32, h);
                            }
                        }
                    } else {
                        for r in r0..=r1 {
                            for jj in j.saturating_sub(leg)..=(j + leg).min(ncols - 1) {
                                let c = index[r * ncols + jj];
                                if c != u32::MAX {
                                    let h = hist[c as usize];
                                    if h > top || (h == top && best != own && c < best) {
                                        (best, top) = (c, h);
                                    }
                                }
                            }
                        }
                    }
                    row[j] = best;
                }
            },
        );
        let changed = next != index;
        index = next;
        if !changed {
            break;
        }
    }
    index
        .iter()
        .map(|&c| if c == u32::MAX { f64::NAN } else { values[c as usize] })
        .collect()
}

/// The neighbours of pixel `p` after it (right, down, and the two diagonals below with
/// `eight`), each pair of pixels once.
fn forward_neighbours(p: usize, nrows: usize, ncols: usize, eight: bool, mut f: impl FnMut(usize)) {
    let (i, j) = (p / ncols, p % ncols);
    if j + 1 < ncols {
        f(p + 1);
    }
    if i + 1 < nrows {
        f(p + ncols);
        if eight {
            if j > 0 {
                f(p + ncols - 1);
            }
            if j + 1 < ncols {
                f(p + ncols + 1);
            }
        }
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Minimum mapping unit by a sieve: the patches (pixels of one class connected 4- or, with
/// `eight`, 8-wise) of fewer than `min_size` pixels are merged, the smallest first, into the
/// neighbouring patch with which they share the longest border (on a tie, the largest, then the
/// smallest class), taking its class; a merged patch that joins others of its class grows into one.
/// A patch with no neighbour (surrounded by missing pixels) stays. Missing pixels stay missing.
pub fn sieve(classes: &[f64], nrows: usize, ncols: usize, min_size: usize, eight: bool) -> Vec<f64> {
    let n = nrows * ncols;
    assert_eq!(classes.len(), n, "classes must be rows x cols");
    // the patches
    let mut label = vec![u32::MAX; n];
    let mut class = vec![];
    let mut size: Vec<usize> = vec![];
    let mut stack = vec![];
    for start in 0..n {
        if label[start] != u32::MAX || classes[start].is_nan() {
            continue;
        }
        let id = class.len() as u32;
        let c = classes[start];
        label[start] = id;
        stack.push(start);
        let mut count = 0;
        while let Some(p) = stack.pop() {
            count += 1;
            let (i, j) = ((p / ncols) as isize, (p % ncols) as isize);
            for (di, dj) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
                if !eight && di != 0 && dj != 0 {
                    continue;
                }
                let (ii, jj) = (i + di, j + dj);
                if ii < 0 || jj < 0 || ii >= nrows as isize || jj >= ncols as isize {
                    continue;
                }
                let q = ii as usize * ncols + jj as usize;
                if label[q] == u32::MAX && classes[q] == c {
                    label[q] = id;
                    stack.push(q);
                }
            }
        }
        class.push(c);
        size.push(count);
    }
    let m = class.len();
    let small: Vec<bool> = size.iter().map(|&s| s < min_size).collect();
    if !small.iter().any(|&s| s) {
        return classes.to_vec();
    }
    // the borders of the small patches: (neighbour, pixel pairs)
    let mut border: Vec<Vec<(u32, u32)>> = vec![vec![]; m];
    for p in 0..n {
        let a = label[p];
        if a == u32::MAX {
            continue;
        }
        forward_neighbours(p, nrows, ncols, eight, |q| {
            let b = label[q];
            if b != u32::MAX && b != a {
                if small[a as usize] {
                    border[a as usize].push((b, 1));
                }
                if small[b as usize] {
                    border[b as usize].push((a, 1));
                }
            }
        });
    }
    let mut parent: Vec<u32> = (0..m as u32).collect();
    let mut heap: BinaryHeap<Reverse<(usize, u32)>> = (0..m as u32)
        .filter(|&x| small[x as usize])
        .map(|x| Reverse((size[x as usize], x)))
        .collect();
    let mut counts: HashMap<u32, u32> = HashMap::new();
    while let Some(Reverse((s, x))) = heap.pop() {
        if parent[x as usize] != x || size[x as usize] != s || s >= min_size {
            continue;
        }
        // the patches around x now, and the border shared with each
        counts.clear();
        let entries = std::mem::take(&mut border[x as usize]);
        for &(b, k) in &entries {
            let r = find(&mut parent, b);
            if r != x {
                *counts.entry(r).or_insert(0) += k;
            }
        }
        let mut around: Vec<(u32, u32)> = counts.iter().map(|(&r, &k)| (r, k)).collect();
        around.sort_unstable();
        let Some(&(to, _)) = around.iter().max_by(|a, b| {
            a.1.cmp(&b.1)
                .then(size[a.0 as usize].cmp(&size[b.0 as usize]))
                .then(class[b.0 as usize].total_cmp(&class[a.0 as usize]))
                .then(b.0.cmp(&a.0))
        }) else {
            continue;
        };
        // x joins `to`, and so do the patches of `to`'s class around x
        let joined: Vec<u32> = std::iter::once(x)
            .chain(
                around
                    .iter()
                    .filter(|&&(r, _)| r != to && class[r as usize] == class[to as usize])
                    .map(|&(r, _)| r),
            )
            .collect();
        for &y in &joined {
            parent[y as usize] = to;
            size[to as usize] += size[y as usize];
            if y == x {
                border[to as usize].extend(around.iter().copied());
            } else {
                let b = std::mem::take(&mut border[y as usize]);
                border[to as usize].extend(b);
            }
        }
        if size[to as usize] < min_size {
            heap.push(Reverse((size[to as usize], to)));
        }
    }
    (0..n)
        .map(|p| {
            if label[p] == u32::MAX {
                f64::NAN
            } else {
                class[find(&mut parent, label[p]) as usize]
            }
        })
        .collect()
}

/// Removes the short excursions of class series: in each pixel's series (pixels x dates) of
/// valid dates, a run of at most `max_length` dates of one class between two runs of another,
/// the same before and after (A B A), takes that class (A A A). The runs are those of the series
/// given (A B C B A with max_length 1: only C, between two B, changes). Missing dates stay
/// missing and do not break a run.
pub fn desawtooth(classes: &[f64], n_pixels: usize, n_dates: usize, max_length: usize, n_jobs: i32) -> Vec<f64> {
    assert_eq!(classes.len(), n_pixels * n_dates, "classes must be pixels x dates");
    let mut out = classes.to_vec();
    parallel::for_each_row(
        &mut out,
        n_dates,
        n_jobs,
        || (Vec::new(), Vec::new()),
        |(valid, runs): &mut (Vec<usize>, Vec<(usize, usize, f64)>), _, row| {
            valid.clear();
            valid.extend((0..row.len()).filter(|&t| !row[t].is_nan()));
            runs.clear();
            for (k, &t) in valid.iter().enumerate() {
                match runs.last_mut() {
                    Some(last) if last.2 == row[t] => last.1 = k + 1,
                    _ => runs.push((k, k + 1, row[t])),
                }
            }
            for r in 1..runs.len().saturating_sub(1) {
                let (before, (a, b, _), after) = (runs[r - 1].2, runs[r], runs[r + 1].2);
                if b - a <= max_length && before == after {
                    for &t in &valid[a..b] {
                        row[t] = before;
                    }
                }
            }
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mirror_repeats_the_edge() {
        let got: Vec<usize> = (-3..8).map(|i| mirror(i, 5)).collect();
        assert_eq!(got, vec![2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2]);
        assert_eq!(mirror(-3, 1), 0);
    }

    #[test]
    fn smoothing_pulls_an_odd_pixel_to_its_neighbours() {
        // two classes, a 7 x 7 map of class 0 at 0.9 but its centre at 0.2
        let (r, c) = (7, 7);
        let mut p0 = vec![0.9; r * c];
        p0[3 * c + 3] = 0.2;
        p0[c + 5] = 0.85;
        let p1: Vec<f64> = p0.iter().map(|v| 1.0 - v).collect();
        let probs = [p0.clone(), p1].concat();
        let s = smooth_probs(&probs, 2, r, c, 5, &[20.0, 20.0], 0.5, 1);
        assert!(
            s[3 * c + 3] > 0.5,
            "the centre takes its neighbours' class: {}",
            s[3 * c + 3]
        );
        // a uniform neighbourhood (s² < 1e-4) gives its mean
        assert!((s[0] - 0.9).abs() < 1e-9);
        // a missing pixel stays missing, and is left out of its neighbours'
        let mut probs2 = probs.clone();
        probs2[0] = f64::NAN;
        let s2 = smooth_probs(&probs2, 2, r, c, 5, &[20.0, 20.0], 1.0, 2);
        assert!(s2[0].is_nan() && s2[r * c].is_nan() && s2[1].is_finite());
    }

    #[test]
    fn majority_removes_salt_and_pepper() {
        let mut m = vec![1.0; 25];
        m[12] = 9.0;
        m[0] = f64::NAN;
        let f = majority(&m, 5, 5, 3, 1, 1);
        assert_eq!(f[12], 1.0);
        assert!(f[0].is_nan());
        // a tie keeps the pixel's class; else the smallest class of the most frequent
        let row = [1.0, 2.0, 2.0, 1.0];
        assert_eq!(majority(&row, 1, 4, 3, 1, 1), vec![1.0, 2.0, 2.0, 1.0]);
        let row = [2.0, 2.0, 9.0, 3.0, 3.0];
        assert_eq!(majority(&row, 1, 5, 5, 1, 1)[2], 2.0);
        // many classes (more than the window's pixels): a column of 1s among distinct classes
        let many: Vec<f64> = (0..100)
            .map(|v| if v % 10 == 5 { 1.0 } else { 1000.0 + v as f64 })
            .collect();
        let f = majority(&many, 10, 10, 3, 1, 1);
        assert_eq!((f[4], f[14], f[0]), (1.0, 1.0, 1000.0));
    }

    #[test]
    fn sieve_merges_small_patches_into_their_neighbour() {
        // 6 x 6: class 1 on the left, 2 on the right; a pixel of 3 inside the 1s, two of 4 on the border
        #[rustfmt::skip]
        let m = vec![
            1., 1., 1., 2., 2., 2.,
            1., 3., 1., 2., 2., 2.,
            1., 1., 1., 4., 2., 2.,
            1., 1., 1., 4., 2., 2.,
            1., 1., 1., 2., 2., 2.,
            1., 1., 1., 2., 2., f64::NAN,
        ];
        let s = sieve(&m, 6, 6, 3, false);
        assert_eq!(s[7], 1.0);
        // the two 4s share 4 pixel sides with the 2s and 2 with the 1s
        assert_eq!((s[15], s[21]), (2.0, 2.0));
        assert!(s[35].is_nan());
        assert_eq!(s.iter().filter(|v| **v == 2.0).count(), 17);
        // a small patch alone stays; min_size 1 changes nothing
        assert_eq!(sieve(&[5.0, f64::NAN], 1, 2, 10, true)[0], 5.0);
        assert_eq!(sieve(&m, 6, 6, 1, true)[7], 3.0);
    }

    #[test]
    fn sieve_joins_the_patches_it_connects() {
        // two patches of class 2 split by two pixels of 3: merged into the first, the 3s join the
        // two into one patch of 10 that stays with min_size 5
        #[rustfmt::skip]
        let m = vec![
            2., 2., 3., 2., 2.,
            2., 2., 3., 2., 2.,
            1., 1., 1., 1., 1.,
        ];
        let s = sieve(&m, 3, 5, 5, false);
        assert_eq!(&s[..10], &[2.0; 10]);
        assert_eq!(&s[10..], &[1.0; 5]);
    }

    #[test]
    fn desawtooth_flattens_excursions() {
        let n = f64::NAN;
        let series = [
            1., 2., 1., 1., 3., 3., 1., //
            1., 2., n, 1., 4., 2., 4., //
            5., 6., 7., 6., 5., 5., 5.,
        ];
        let out = desawtooth(&series, 3, 7, 1, 1);
        assert_eq!(&out[..7], &[1., 1., 1., 1., 3., 3., 1.]);
        assert_eq!(out[7..9], [1., 1.]);
        assert!(out[9].is_nan());
        assert_eq!(&out[10..14], &[1., 4., 4., 4.]);
        assert_eq!(&out[14..], &[5., 6., 6., 6., 5., 5., 5.]);
        let out2 = desawtooth(&series, 3, 7, 2, 1);
        assert_eq!(&out2[..7], &[1.; 7]);
    }
}
