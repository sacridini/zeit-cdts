//! The Monte Carlo of the stochastic watershed (phase 24b): many marker-based watersheds of one
//! gradient image from random markers, counting for each pixel in how many of them it is on a
//! boundary. One realisation is the loop body of the MappingDemo notebook: the points are coloured
//! on their k-nearest-neighbour graph (pyamg's `vertex_coloring`, method `MIS`), written into an
//! integer image as colour + 1, dilated by a disk (grey dilation: the larger label wins where
//! discs overlap), flooded by the watershed inside the mask and the boundaries of the result
//! (`find_boundaries(connectivity=4)`, thick) are added to the counts. Each step is the 24a
//! building block that equals scikit-image's, so the counts equal the notebook's given the same
//! markers; the markers themselves are drawn by the caller (numpy) and only thinned and coloured here.

use rayon::prelude::*;

use super::label::find_boundaries;
use super::morph::{dilation, Border, Footprint};
use super::points::trim_close_points;
use super::watershed::watershed;

/// The k nearest *other* points of every point as a CSR graph `(indptr, indices)`, the same
/// neighbour sets as scikit-learn's `kneighbors_graph(points, k)` (mode "connectivity", directed:
/// row i holds the neighbours of i, nearest first, ties by index). Brute force, O(n^2): meant for
/// the ~1000 markers of a realisation. If `k` > n - 1 the n - 1 others are taken (scikit-learn
/// raises).
pub fn knn_graph(x: &[f64], y: &[f64], k: usize) -> Result<(Vec<u32>, Vec<u32>), String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    let n = x.len();
    if n > u32::MAX as usize / 2 {
        return Err("too many points".into());
    }
    let k = k.min(n.saturating_sub(1));
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::with_capacity(n * k);
    indptr.push(0u32);
    let mut cand: Vec<(f64, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        cand.clear();
        for j in 0..n {
            if j != i {
                let (dx, dy) = (x[i] - x[j], y[i] - y[j]);
                cand.push((dx * dx + dy * dy, j as u32));
            }
        }
        let by = |a: &(f64, u32), b: &(f64, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        if k < cand.len() {
            cand.select_nth_unstable_by(k, by);
            cand.truncate(k);
        }
        cand.sort_unstable_by(by);
        indices.extend(cand.iter().map(|c| c.1));
        indptr.push(indices.len() as u32);
    }
    Ok((indptr, indices))
}

/// pyamg's `vertex_coloring(G, method="MIS")` of a CSR graph, exactly: colour K is the greedy
/// maximal independent set, in vertex order, of the vertices left after the colours before it.
/// The graph is read as given (not symmetrised: row i marks only the vertices it lists, so a
/// directed kNN graph is coloured as pyamg does, and may leave an edge j -> i between equal
/// colours); the diagonal is ignored. Colours start at 0.
pub fn vertex_coloring_mis(n: usize, indptr: &[u32], indices: &[u32]) -> Result<Vec<i32>, String> {
    if indptr.len() != n + 1 || indptr.first() != Some(&0) || indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err("indptr must have n + 1 non-decreasing entries starting at 0".into());
    }
    if indptr[n] as usize != indices.len() || indices.iter().any(|&j| j as usize >= n) {
        return Err("indices must match indptr and be below n".into());
    }
    let mut x = vec![-1i32; n];
    let mut done = 0usize;
    let mut k = 0i32;
    while done < n {
        let (active, c, f) = (-1 - k, k, -2 - k);
        for i in 0..n {
            if x[i] != active {
                continue;
            }
            x[i] = c;
            done += 1;
            for &j in &indices[indptr[i] as usize..indptr[i + 1] as usize] {
                if x[j as usize] == active {
                    x[j as usize] = f;
                }
            }
        }
        k += 1;
    }
    Ok(x)
}

/// How the markers of a set are labelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Labelling {
    /// The notebook's: pyamg's colouring of the `n_neighbors`-nearest-neighbour graph, plus 1.
    Knn(usize),
    /// Every marker its own label, 1..=k.
    Unique,
}

/// The markers of one realisation: flat pixel indices and labels, in the order they are written
/// into the marker image (a pixel hit twice keeps the last label, as numpy's fancy assignment).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MarkerSet {
    pub pixels: Vec<u32>,
    pub labels: Vec<i32>,
}

/// A marker set from float points (x = row, y = col, in pixels): thinned by the notebook's
/// `trim_close_points` with `min_distance` (> 0; 0 keeps every point; pairs sorted by (i, j)),
/// each point sent to the pixel `int(x), int(y)` (truncation, as `astype(int)`; a point outside the
/// `rows` x `cols` image is an error) and labelled.
pub fn marker_set(
    x: &[f64],
    y: &[f64],
    rows: usize,
    cols: usize,
    min_distance: f64,
    labelling: Labelling,
) -> Result<MarkerSet, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    if rows.checked_mul(cols).is_none_or(|n| n > u32::MAX as usize) {
        return Err("the image has more than 2^32 pixels".into());
    }
    let (xs, ys): (Vec<f64>, Vec<f64>) = if min_distance > 0.0 {
        let keep = trim_close_points(x, y, min_distance)?;
        (
            keep.iter().map(|&i| x[i as usize]).collect(),
            keep.iter().map(|&i| y[i as usize]).collect(),
        )
    } else {
        (x.to_vec(), y.to_vec())
    };
    let mut pixels = Vec::with_capacity(xs.len());
    for (&a, &b) in xs.iter().zip(&ys) {
        let (r, c) = (a.trunc(), b.trunc());
        if !(r >= 0.0 && c >= 0.0 && r < rows as f64 && c < cols as f64) {
            return Err(format!("the point ({a}, {b}) is outside the {rows} x {cols} image"));
        }
        pixels.push((r as usize * cols + c as usize) as u32);
    }
    let labels = match labelling {
        Labelling::Knn(k) => {
            let (indptr, indices) = knn_graph(&xs, &ys, k)?;
            vertex_coloring_mis(xs.len(), &indptr, &indices)?
                .into_iter()
                .map(|c| c + 1)
                .collect()
        }
        Labelling::Unique => (1..=xs.len() as i32).collect(),
    };
    Ok(MarkerSet { pixels, labels })
}

/// One realisation: adds 1 to `counts` where the watershed of `gradient` from `set` (dilated by
/// a disk of `disk_radius`) has a boundary.
fn realisation(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    set: &MarkerSet,
    disk: &Footprint,
    counts: &mut [u32],
) -> Result<(), String> {
    let mut img = vec![0.0f64; rows * cols];
    for (&p, &l) in set.pixels.iter().zip(&set.labels) {
        img[p as usize] = l as f64;
    }
    let dilated = dilation(&img, rows, cols, disk, Border::parse("reflect", 0.0)?)?;
    let markers: Vec<i32> = dilated.iter().map(|&v| v as i32).collect();
    let seg = watershed(gradient, &markers, rows, cols, 1, mask)?;
    for (c, b) in counts.iter_mut().zip(find_boundaries(&seg, rows, cols, 4)?) {
        *c += b as u32;
    }
    Ok(())
}

/// Checks and runs `count` realisations whose marker sets `make(i)` builds on the worker that runs
/// them: each worker folds into its own counts, which are summed at the end.
#[allow(clippy::too_many_arguments)]
fn run_realisations<F>(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    count: usize,
    make: F,
    disk_radius: usize,
    threads: usize,
) -> Result<Vec<u32>, String>
where
    F: Fn(usize) -> Result<MarkerSet, String> + Sync,
{
    let n = rows * cols;
    if gradient.len() != n || mask.is_some_and(|m| m.len() != n) {
        return Err("gradient and mask must have rows * cols values".into());
    }
    let disk = Footprint::disk(disk_radius);
    let pool = crate::parallel::pool(threads.max(1));
    pool.install(|| {
        (0..count)
            .into_par_iter()
            .try_fold(
                || vec![0u32; n],
                |mut acc, i| {
                    let set = make(i)?;
                    if set.pixels.len() != set.labels.len() || set.pixels.iter().any(|&p| p as usize >= n) {
                        return Err("a marker set has pixels out of the image or labels of another length".into());
                    }
                    realisation(gradient, rows, cols, mask, &set, &disk, &mut acc)?;
                    Ok::<_, String>(acc)
                },
            )
            .try_reduce(
                || vec![0u32; n],
                |mut a, b| {
                    for (u, v) in a.iter_mut().zip(&b) {
                        *u += v;
                    }
                    Ok(a)
                },
            )
    })
}

/// For each pixel of the `rows` x `cols` `gradient`, the number of the realisations (one per
/// marker set) in which it is on a boundary. `mask` (None: all valid) limits the watersheds. The
/// realisations run on `threads` threads, each folding into its own counts, summed at the end; the
/// segmentations are not kept, and the result does not depend on `threads`.
pub fn monte_carlo_counts(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    sets: &[MarkerSet],
    disk_radius: usize,
    threads: usize,
) -> Result<Vec<u32>, String> {
    run_realisations(
        gradient,
        rows,
        cols,
        mask,
        sets.len(),
        |i| Ok(sets[i].clone()),
        disk_radius,
        threads,
    )
}

/// [`monte_carlo_counts`] from float points (one `(x, y)` pair of slices per realisation): each
/// worker thins, colours and floods its own sets (see [`marker_set`]), so the sets are never all in
/// memory at once.
#[allow(clippy::too_many_arguments)]
pub fn monte_carlo_counts_points(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    points: &[(&[f64], &[f64])],
    min_distance: f64,
    labelling: Labelling,
    disk_radius: usize,
    threads: usize,
) -> Result<Vec<u32>, String> {
    run_realisations(
        gradient,
        rows,
        cols,
        mask,
        points.len(),
        |i| marker_set(points[i].0, points[i].1, rows, cols, min_distance, labelling),
        disk_radius,
        threads,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knn_of_a_line() {
        let x = [0.0, 1.0, 3.0, 7.0];
        let y = [0.0; 4];
        let (p, i) = knn_graph(&x, &y, 2).unwrap();
        assert_eq!(p, vec![0, 2, 4, 6, 8]);
        assert_eq!(i, vec![1, 2, 0, 2, 1, 0, 2, 1]);
        // k beyond n - 1 takes all the others
        let (p, _) = knn_graph(&x, &y, 10).unwrap();
        assert_eq!(p, vec![0, 3, 6, 9, 12]);
    }

    #[test]
    fn coloring_of_a_path_and_a_clique() {
        // path 0-1-2-3 (symmetric): MIS {0, 2} then {1, 3}
        let (indptr, indices) = (vec![0, 1, 3, 5, 6], vec![1, 0, 2, 1, 3, 2]);
        assert_eq!(vertex_coloring_mis(4, &indptr, &indices).unwrap(), vec![0, 1, 0, 1]);
        // triangle with a self loop on 0
        let (indptr, indices) = (vec![0, 3, 5, 7], vec![0, 1, 2, 0, 2, 0, 1]);
        assert_eq!(vertex_coloring_mis(3, &indptr, &indices).unwrap(), vec![0, 1, 2]);
        // no edges: one colour
        assert_eq!(vertex_coloring_mis(3, &[0, 0, 0, 0], &[]).unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn counts_do_not_depend_on_threads() {
        let (rows, cols) = (20usize, 24usize);
        let grad: Vec<f64> = (0..rows * cols).map(|i| ((i * 37) % 11) as f64 / 11.0).collect();
        let sets: Vec<MarkerSet> = (0..6)
            .map(|s| {
                let x: Vec<f64> = (0..30).map(|i| ((i * 7 + s * 3) % 20) as f64 + 0.5).collect();
                let y: Vec<f64> = (0..30).map(|i| ((i * 11 + s * 5) % 24) as f64 + 0.25).collect();
                marker_set(&x, &y, rows, cols, 2.0, Labelling::Knn(5)).unwrap()
            })
            .collect();
        let one = monte_carlo_counts(&grad, rows, cols, None, &sets, 1, 1).unwrap();
        let three = monte_carlo_counts(&grad, rows, cols, None, &sets, 1, 3).unwrap();
        assert_eq!(one, three);
        assert!(one.iter().all(|&c| c <= 6));
    }
}
