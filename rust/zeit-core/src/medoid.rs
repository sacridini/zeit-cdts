//! The medoid composite over time (src/utils.cpp's compute_medoid), same
//! operations in the same order: per pixel, the band-wise median of the valid
//! observations, then the observation closest to it (Eigen's `squaredNorm`
//! order of additions).

use crate::{eigen, parallel};

/// The median of `v` (reordered), as the C++: the upper middle value, or the
/// mean of the two middle ones written (max of the lower half + upper) / 2.
fn median(v: &mut [f64]) -> f64 {
    let (n, even) = (v.len() / 2, v.len().is_multiple_of(2));
    let (lower, &mut hi, _) = v.select_nth_unstable_by(n, f64::total_cmp);
    if even {
        // std::max_element: the first of the largest
        let lo = lower.iter().copied().fold(lower[0], |m, x| if m < x { x } else { m });
        (lo + hi) / 2.0
    } else {
        hi
    }
}

/// The medoid of every pixel of `values` (`pixels` x T x B, row-major): B
/// values per pixel. `no_data` (or NaN) marks a missing value; a pixel with no
/// complete observation gives `no_data`. On every processor, as the C++'s
/// OpenMP default.
pub fn medoid(values: &[f64], pixels: usize, times: usize, bands: usize, no_data: f64) -> Vec<f64> {
    assert_eq!(
        values.len(),
        pixels * times * bands,
        "values must be pixels x time x bands"
    );
    let mut out = vec![0.0; pixels * bands];
    let missing = |v: f64| v == no_data || v.is_nan();
    parallel::for_each_row(
        &mut out,
        bands,
        parallel::all_processors(),
        || (Vec::with_capacity(times), vec![0.0; bands]),
        |(valid, median_vec): &mut (Vec<f64>, Vec<f64>), p, out| {
            let px = &values[p * times * bands..(p + 1) * times * bands];
            let mut has_valid = false;
            for b in 0..bands {
                valid.clear();
                valid.extend((0..times).map(|t| px[t * bands + b]).filter(|&v| !missing(v)));
                median_vec[b] = if valid.is_empty() {
                    no_data
                } else {
                    has_valid = true;
                    median(valid)
                };
            }
            if !has_valid {
                out.fill(no_data);
                return;
            }
            let mut min_dist = f64::MAX;
            let mut best = None;
            for (t, obs) in px.chunks_exact(bands).enumerate() {
                if obs.iter().any(|&v| missing(v)) {
                    continue;
                }
                let dist = eigen::squared_distance(obs, median_vec);
                if dist < min_dist {
                    min_dist = dist;
                    best = Some(t);
                }
            }
            match best {
                Some(t) => out.copy_from_slice(&px[t * bands..(t + 1) * bands]),
                None => out.fill(no_data),
            }
        },
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_observation_nearest_the_median() {
        // one pixel, 3 dates, 2 bands; the middle date is the medoid
        let v = [0.0, 0.0, 1.0, 1.1, 5.0, 5.0];
        assert_eq!(medoid(&v, 1, 3, 2, -9999.0), vec![1.0, 1.1]);
        let empty = [f64::NAN; 6];
        assert_eq!(medoid(&empty, 1, 3, 2, -9999.0), vec![-9999.0; 2]);
    }
}
