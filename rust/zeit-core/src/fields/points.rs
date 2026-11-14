//! The notebook's `trim_close_points`: a greedy thinning of random marker points so that the kept
//! ones are farther than `distance` from each other.
//!
//! The notebook walks `KDTree.query_pairs(distance)`, a Python set whose iteration order is arbitrary,
//! and for each pair (i, j), i < j, with neither point removed yet, removes j. The kept set depends
//! strongly on that order, so the greedy step takes the pairs in the order given
//! ([`trim_close_points_pairs`], to reproduce a run of the notebook from its own pairs), and
//! [`close_pairs`] finds them, sorted by (i, j): Zeit's documented default, which keeps the
//! lowest-numbered point of each cluster.

/// The indices (ascending) kept by the notebook's greedy loop over `pairs` in the given order.
pub fn trim_close_points_pairs(n: usize, pairs: &[(u32, u32)]) -> Vec<u32> {
    let mut removed = vec![false; n];
    for &(i, j) in pairs {
        if removed[i as usize] || removed[j as usize] {
            continue;
        }
        removed[j as usize] = true;
    }
    (0..n as u32).filter(|&i| !removed[i as usize]).collect()
}

/// The pairs (i, j), i < j, of points at Euclidean distance <= `distance` (as
/// `KDTree.query_pairs`), sorted by (i, j). Found with a uniform grid of cell `distance`.
pub fn close_pairs(x: &[f64], y: &[f64], distance: f64) -> Result<Vec<(u32, u32)>, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    if x.len() > u32::MAX as usize {
        return Err("more than 2^32 points".into());
    }
    if !(distance > 0.0) || !distance.is_finite() {
        return Err("distance must be positive and finite".into());
    }
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let (x0, y0) = (
        x.iter().copied().fold(f64::INFINITY, f64::min),
        y.iter().copied().fold(f64::INFINITY, f64::min),
    );
    let cell = |v: f64, v0: f64| ((v - v0) / distance).floor() as i64;
    let mut order: Vec<(i64, i64, u32)> = (0..n).map(|i| (cell(x[i], x0), cell(y[i], y0), i as u32)).collect();
    order.sort_unstable();
    // first slot of each occupied cell
    let mut cells: std::collections::HashMap<(i64, i64), (usize, usize)> = std::collections::HashMap::new();
    for (k, &(cx, cy, _)) in order.iter().enumerate() {
        cells.entry((cx, cy)).and_modify(|e| e.1 = k + 1).or_insert((k, k + 1));
    }
    let d2 = distance * distance;
    let mut pairs = Vec::new();
    for &(cx, cy, i) in &order {
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(&(a, b)) = cells.get(&(cx + dx, cy + dy)) else {
                    continue;
                };
                for &(_, _, j) in &order[a..b] {
                    if j > i {
                        let (ex, ey) = (x[i as usize] - x[j as usize], y[i as usize] - y[j as usize]);
                        if ex * ex + ey * ey <= d2 {
                            pairs.push((i, j));
                        }
                    }
                }
            }
        }
    }
    pairs.sort_unstable();
    Ok(pairs)
}

/// The kept indices of the greedy thinning with the pairs in sorted order.
pub fn trim_close_points(x: &[f64], y: &[f64], distance: f64) -> Result<Vec<u32>, String> {
    let pairs = close_pairs(x, y, distance)?;
    Ok(trim_close_points_pairs(x.len(), &pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_found_sorted() {
        let x = [0.0, 3.0, 1.0, 10.0];
        let y = [0.0, 0.0, 0.0, 0.0];
        assert_eq!(close_pairs(&x, &y, 2.0).unwrap(), vec![(0, 2), (1, 2)]);
        // exactly at the distance counts
        assert_eq!(close_pairs(&[0.0, 2.0], &[0.0, 0.0], 2.0).unwrap(), vec![(0, 1)]);
    }

    #[test]
    fn greedy_depends_on_pair_order() {
        // chain 0-1-2: (0,1) first removes 1, then (1,2) is skipped: keeps 0 and 2;
        // (1,2) first removes 2, then (0,1) removes 1: keeps only 0
        assert_eq!(trim_close_points_pairs(3, &[(0, 1), (1, 2)]), vec![0, 2]);
        assert_eq!(trim_close_points_pairs(3, &[(1, 2), (0, 1)]), vec![0]);
    }

    #[test]
    fn trimmed_points_are_far_apart() {
        let n = 400;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 211) as f64 * 0.5).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 104729) % 197) as f64 * 0.5).collect();
        let keep = trim_close_points(&x, &y, 8.0).unwrap();
        for (a, &i) in keep.iter().enumerate() {
            for &j in &keep[a + 1..] {
                let (dx, dy) = (x[i as usize] - x[j as usize], y[i as usize] - y[j as usize]);
                assert!(dx * dx + dy * dy > 64.0);
            }
        }
    }
}
