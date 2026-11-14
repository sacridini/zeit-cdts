//! The exact boundary probability of the stochastic watershed (phase 24b) and a Monte Carlo of the
//! same marker model. The method is the one of Meyer & Stawiaski 2010 (the watershed as a minimum
//! spanning forest cut) and Malmberg & Luengo Hendriks 2014 (Pattern Recognition Letters 47:80, the
//! exact probability from one pass over the sorted edges); the formulas here are derived in the
//! comments below, not copied from any implementation.
//!
//! Model: the graph holds the mask pixels, its edges join 4-neighbours inside the mask and weigh
//! `max(g[p], g[q])`. `N` markers are drawn uniformly, one pixel each, with replacement. The
//! watershed cut of the minimum spanning forest cuts an edge when both its Kruskal components
//! hold a marker. For an edge that joins components of `a` and `b` pixels out of `n`, by
//! inclusion-exclusion
//!
//! `P(e) = 1 - (1-a/n)^N - (1-b/n)^N + (1-(a+b)/n)^N`.
//!
//! An edge that closes a cycle takes the maximum of `P` over the path between its two ends in the
//! minimum spanning forest: the cut forest merges only along forest edges, so the ends are in
//! different basins iff a forest edge on that path is cut, and the probability of that union of
//! events is at least the largest of them. A ridge edge thus picks up the `P` of the saddle on its
//! path. It is a lower bound of the true cut probability (exact for forest edges). The pixel
//! probability is the maximum of `P` over its edges.
//!
//! Edge numbering (fixed, independent of threads): the horizontal edge between `(r, c)` and
//! `(r, c+1)` is `r * (cols - 1) + c`; the vertical one between `(r, c)` and `(r+1, c)` is
//! `rows * (cols - 1) + r * cols + c`. Edges are sorted by `(weight, id)`.

use rayon::prelude::*;

/// The edges of the graph sorted by (weight, id), and the number of horizontal ones.
fn sorted_edges(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    threads: usize,
) -> Result<(Vec<u32>, usize), String> {
    let n = rows * cols;
    if gradient.len() != n || mask.is_some_and(|m| m.len() != n) {
        return Err("gradient and mask must have rows * cols values".into());
    }
    if n >= (u32::MAX / 2) as usize {
        return Err("the image has too many pixels".into());
    }
    let inside = |p: usize| mask.is_none_or(|m| m[p]);
    if (0..n).any(|p| inside(p) && gradient[p].is_nan()) {
        return Err("the gradient has NaN inside the mask".into());
    }
    let nh = rows * cols.saturating_sub(1);
    let nv = rows.saturating_sub(1) * cols;
    let mut keys: Vec<(f64, u32)> = Vec::with_capacity(nh + nv);
    for r in 0..rows {
        for c in 0..cols.saturating_sub(1) {
            let (p, q) = (r * cols + c, r * cols + c + 1);
            if inside(p) && inside(q) {
                keys.push((gradient[p].max(gradient[q]), (r * (cols - 1) + c) as u32));
            }
        }
    }
    for r in 0..rows.saturating_sub(1) {
        for c in 0..cols {
            let (p, q) = (r * cols + c, (r + 1) * cols + c);
            if inside(p) && inside(q) {
                keys.push((gradient[p].max(gradient[q]), (nh + r * cols + c) as u32));
            }
        }
    }
    let pool = crate::parallel::pool(threads.max(1));
    pool.install(|| keys.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))));
    Ok((keys.into_iter().map(|k| k.1).collect(), nh))
}

/// The two pixels an edge joins.
#[inline]
fn endpoints(id: usize, nh: usize, cols: usize) -> (usize, usize) {
    if id < nh {
        let p = id / (cols - 1) * cols + id % (cols - 1);
        (p, p + 1)
    } else {
        let p = id - nh;
        (p, p + cols)
    }
}

#[inline]
fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let g = parent[parent[x as usize] as usize];
        parent[x as usize] = g;
        x = g;
    }
    x
}

/// The root of `x` by climbing, without compressing the path.
#[inline]
fn root(parent: &[u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        x = parent[x as usize];
    }
    x
}

/// `P(e)` for components of `a` and `b` pixels out of `n` and `markers` markers, stable for tiny
/// `a / n` and large `markers`: with `A = (1-a/n)^N` and `B = (1-b/n)^N`,
/// `P = (1-A)(1-B) + A*B*expm1(N * ln(1 - ab/((n-a)(n-b))))` (the last term is the ratio
/// `(1-(a+b)/n)^N / (AB)` minus 1; for `a + b = n` its argument is -1, giving -inf and -1).
fn edge_probability(a: f64, b: f64, n: f64, markers: f64) -> f64 {
    let la = markers * (-a / n).ln_1p();
    let lb = markers * (-b / n).ln_1p();
    let (big_a, big_b) = (la.exp(), lb.exp());
    let cross = (markers * (-a * b / ((n - a) * (n - b))).ln_1p()).exp_m1();
    let p = (-la.exp_m1()) * (-lb.exp_m1()) + big_a * big_b * cross;
    p.clamp(0.0, 1.0)
}

/// `(pixel, edges_h, edges_v)` of [`exact_boundary_probability`].
pub type Probabilities = (Vec<f64>, Vec<f64>, Vec<f64>);

/// The exact boundary probability: `(pixel, edges_h, edges_v)` with `pixel` (`rows * cols`, 0
/// outside the mask) the maximum of `P` over the edges of each pixel, `edges_h` (`rows * (cols - 1)`)
/// and `edges_v` (`(rows - 1) * cols`) the per-edge `P` (0 for edges not inside the mask). `n_markers`
/// is real and >= 0. NaN in the gradient inside the mask is an error (outside it is ignored).
pub fn exact_boundary_probability(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    n_markers: f64,
    threads: usize,
) -> Result<Probabilities, String> {
    if n_markers.is_nan() || n_markers < 0.0 {
        return Err("the number of markers must be >= 0".into());
    }
    let (order, nh) = sorted_edges(gradient, rows, cols, mask, threads)?;
    let n = rows * cols;
    let nv = rows.saturating_sub(1) * cols;
    let mut pixel = vec![0.0f64; n];
    let mut prob = vec![0.0f64; nh + nv];
    let n_mask = mask.map_or(n, |m| m.iter().filter(|&&v| v).count()) as f64;
    if n_markers > 0.0 {
        // Pass 1: Kruskal. A spanning-forest edge gets its own P; an edge that closes a cycle is
        // left for pass 2.
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut size = vec![1u32; n];
        let mut tree: Vec<(f64, u32)> = Vec::new();
        let mut cycles: Vec<u32> = Vec::new();
        for &e in &order {
            let (p, q) = endpoints(e as usize, nh, cols);
            let (ra, rb) = (find(&mut parent, p as u32), find(&mut parent, q as u32));
            if ra == rb {
                cycles.push(e);
                continue;
            }
            let (sa, sb) = (size[ra as usize], size[rb as usize]);
            let v = edge_probability(sa as f64, sb as f64, n_mask, n_markers);
            prob[e as usize] = v;
            tree.push((v, e));
            let (big, small) = if sa >= sb { (ra, rb) } else { (rb, ra) };
            parent[small as usize] = big;
            size[big as usize] = sa + sb;
        }
        // Pass 2: in the cut forest merges happen only on spanning-forest edges, so p and q (the
        // ends of a cycle edge) are apart iff some tree edge on their tree path is cut, and
        // P(cut) >= the maximum of P over that path. The maximum over the unique tree path is the
        // minimax value: relink the tree edges in order of P (ties by id) with union by size and no
        // path compression; the value is the P of the latest link on the path between p and q in
        // that link forest (links get later going up, so climb the node whose link is earlier
        // until the two meet).
        let pool = crate::parallel::pool(threads.max(1));
        pool.install(|| tree.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))));
        for (i, x) in parent.iter_mut().enumerate() {
            *x = i as u32;
        }
        size.fill(1);
        let mut time = vec![u32::MAX; n];
        let mut link_p = vec![0.0f64; n];
        for (k, &(v, e)) in tree.iter().enumerate() {
            let (p, q) = endpoints(e as usize, nh, cols);
            let (ra, rb) = (root(&parent, p as u32), root(&parent, q as u32));
            let (sa, sb) = (size[ra as usize], size[rb as usize]);
            let (big, small) = if sa >= sb { (ra, rb) } else { (rb, ra) };
            parent[small as usize] = big;
            size[big as usize] = sa + sb;
            time[small as usize] = k as u32;
            link_p[small as usize] = v;
        }
        let values: Vec<f64> = pool.install(|| {
            cycles
                .par_iter()
                .map(|&e| {
                    let (p, q) = endpoints(e as usize, nh, cols);
                    let (mut x, mut y, mut last) = (p, q, 0.0);
                    while x != y {
                        if time[x] < time[y] {
                            last = link_p[x];
                            x = parent[x] as usize;
                        } else {
                            last = link_p[y];
                            y = parent[y] as usize;
                        }
                    }
                    last
                })
                .collect()
        });
        for (&e, &v) in cycles.iter().zip(&values) {
            prob[e as usize] = v;
        }
        for &e in &order {
            let (p, q) = endpoints(e as usize, nh, cols);
            let v = prob[e as usize];
            for x in [p, q] {
                if v > pixel[x] {
                    pixel[x] = v;
                }
            }
        }
    }
    let edges_v = prob.split_off(nh);
    Ok((pixel, prob, edges_v))
}

/// Per-worker state of the Monte Carlo.
struct Worker {
    counts: Vec<u32>,
    parent: Vec<u32>,
    size: Vec<u32>,
    marked: Vec<bool>,
}

/// The Monte Carlo of the same marker model: for each realisation (a slice of marker pixels, flat
/// indices inside the mask, repeats allowed) Kruskal runs in the same edge order and an edge whose
/// two components both hold a marker is cut (not merged; marked = OR on merge); returns the number
/// of realisations that cut each edge, `(edges_h, edges_v)` as in [`exact_boundary_probability`].
/// An edge of the minimum spanning tree is cut with probability exactly its `P(e)` (every piece
/// next to an earlier cut is marked); any other edge between two separated marked pieces is cut
/// too, with a probability at least the maximum of `P` over the tree path between its ends, which
/// is what the exact function gives it. Realisations run on `threads`
/// threads folding into their own counts; the sums do not depend on `threads`.
pub fn msf_cut_counts(
    gradient: &[f64],
    rows: usize,
    cols: usize,
    mask: Option<&[bool]>,
    marker_pixels: &[&[u32]],
    threads: usize,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let n = rows * cols;
    let (order, nh) = sorted_edges(gradient, rows, cols, mask, threads)?;
    let nv = rows.saturating_sub(1) * cols;
    for s in marker_pixels {
        if s.iter()
            .any(|&p| p as usize >= n || mask.is_some_and(|m| !m[p as usize]))
        {
            return Err("a marker pixel is outside the image or the mask".into());
        }
    }
    let new = || Worker {
        counts: vec![0u32; nh + nv],
        parent: vec![0u32; n],
        size: vec![0u32; n],
        marked: vec![false; n],
    };
    let pool = crate::parallel::pool(threads.max(1));
    let total = pool.install(|| {
        marker_pixels
            .par_iter()
            .fold(new, |mut w, set| {
                for i in 0..n {
                    w.parent[i] = i as u32;
                }
                w.size.fill(1);
                w.marked.fill(false);
                for &p in *set {
                    w.marked[p as usize] = true;
                }
                for &e in &order {
                    let (p, q) = endpoints(e as usize, nh, cols);
                    let (ra, rb) = (find(&mut w.parent, p as u32), find(&mut w.parent, q as u32));
                    if ra == rb {
                        continue;
                    }
                    let (ma, mb) = (w.marked[ra as usize], w.marked[rb as usize]);
                    // both sides hold a marker: the edge is cut and the sides stay apart (a
                    // later edge between them is cut too)
                    if ma && mb {
                        w.counts[e as usize] += 1;
                        continue;
                    }
                    let (sa, sb) = (w.size[ra as usize], w.size[rb as usize]);
                    let (big, small) = if sa >= sb { (ra, rb) } else { (rb, ra) };
                    w.parent[small as usize] = big;
                    w.size[big as usize] = sa + sb;
                    w.marked[big as usize] = ma || mb;
                }
                w
            })
            .map(|w| w.counts)
            .reduce(
                || vec![0u32; nh + nv],
                |mut a, b| {
                    for (u, v) in a.iter_mut().zip(&b) {
                        *u += v;
                    }
                    a
                },
            )
    });
    let mut h = total;
    let v = h.split_off(nh);
    Ok((h, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_by_hand() {
        // three pixels, e0 (weight 1) joins {0},{1}, e1 (weight 2) joins {0,1},{2}
        let g = [0.0, 1.0, 2.0];
        let (px, h, v) = exact_boundary_probability(&g, 1, 3, None, 2.0, 1).unwrap();
        assert!(v.is_empty());
        assert!((h[0] - 2.0 / 9.0).abs() < 1e-15 && (h[1] - 4.0 / 9.0).abs() < 1e-15);
        assert!((px[0] - 2.0 / 9.0).abs() < 1e-15);
        assert!((px[1] - 4.0 / 9.0).abs() < 1e-15 && (px[2] - 4.0 / 9.0).abs() < 1e-15);
        // one marker never makes a boundary
        let (px, h, _) = exact_boundary_probability(&g, 1, 3, None, 1.0, 1).unwrap();
        assert!(h.iter().all(|&p| p.abs() < 1e-15) && px.iter().all(|&p| p.abs() < 1e-15));
    }

    #[test]
    fn no_markers_no_boundary() {
        let g: Vec<f64> = (0..20).map(|i| ((i * 7) % 5) as f64).collect();
        let (px, h, v) = exact_boundary_probability(&g, 4, 5, None, 0.0, 1).unwrap();
        assert!(px.iter().chain(&h).chain(&v).all(|&p| p == 0.0));
        assert_eq!((h.len(), v.len()), (16, 15));
    }

    #[test]
    fn two_disconnected_parts() {
        // 1 x 5, the middle pixel outside the mask: parts {0,1} and {3,4}, n = 4
        let g = [0.0, 1.0, 9.0, 0.0, 2.0];
        let mask = [true, true, false, true, true];
        let (px, h, _) = exact_boundary_probability(&g, 1, 5, Some(&mask), 2.0, 1).unwrap();
        assert_eq!(h[1], 0.0);
        assert_eq!(h[2], 0.0);
        // each edge joins two single pixels of n = 4: P = 1 - 2 (3/4)^2 + (1/2)^2 = 1/8
        assert!((h[0] - 0.125).abs() < 1e-15 && (h[3] - 0.125).abs() < 1e-15);
        assert_eq!(px[2], 0.0);
        // the Monte Carlo cuts only inside the parts
        let sets: Vec<Vec<u32>> = (0..4u32)
            .flat_map(|i| (0..4u32).map(move |j| vec![[0, 1, 3, 4][i as usize], [0, 1, 3, 4][j as usize]]))
            .collect();
        let refs: Vec<&[u32]> = sets.iter().map(|s| s.as_slice()).collect();
        let (ch, _) = msf_cut_counts(&g, 1, 5, Some(&mask), &refs, 1).unwrap();
        assert_eq!(ch, vec![2, 0, 0, 2]); // exactly 16 * 1/8 over all ordered pairs
    }

    #[test]
    fn nan_is_an_error_inside_the_mask_only() {
        let g = [0.0, f64::NAN, 1.0];
        assert!(exact_boundary_probability(&g, 1, 3, None, 2.0, 1).is_err());
        let mask = [true, false, true];
        assert!(exact_boundary_probability(&g, 1, 3, Some(&mask), 2.0, 1).is_ok());
    }

    #[test]
    fn counts_do_not_depend_on_threads() {
        let (rows, cols) = (12usize, 15usize);
        let g: Vec<f64> = (0..rows * cols).map(|i| ((i * 37) % 11) as f64 / 11.0).collect();
        let sets: Vec<Vec<u32>> = (0..30u32)
            .map(|s| (0..5u32).map(|i| (s * 31 + i * 17) % (rows * cols) as u32).collect())
            .collect();
        let refs: Vec<&[u32]> = sets.iter().map(|s| s.as_slice()).collect();
        let one = msf_cut_counts(&g, rows, cols, None, &refs, 1).unwrap();
        let three = msf_cut_counts(&g, rows, cols, None, &refs, 3).unwrap();
        assert_eq!(one, three);
        let a = exact_boundary_probability(&g, rows, cols, None, 5.0, 1).unwrap();
        let b = exact_boundary_probability(&g, rows, cols, None, 5.0, 3).unwrap();
        assert_eq!(a, b);
    }
}
