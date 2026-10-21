//! The Whittaker smoother (Eilers 2003): a port of src/whittaker.cpp, same
//! operations in the same order (so the same bits). It minimises
//! sum w (y - z)^2 + lambda sum (second differences of z)^2, with divided
//! differences for uneven x, scaled by the median step so that an even series
//! gets the classic [1, -2, 1] penalty (and lambda keeps its usual meaning).

use crate::parallel;

/// The error of a series whose positions do not increase.
pub const NOT_INCREASING: &str = "whittaker: x must increase";

/// The smoothed series at positions `x` (`w` the weights). NaN observations
/// weigh 0, so the curve goes through the gaps; all NaN with fewer than two
/// weighted observations.
pub fn whittaker(x: &[f64], y: &[f64], w: &[f64], lambda: f64) -> Result<Vec<f64>, &'static str> {
    let mut z = vec![f64::NAN; y.len()];
    smooth_into(x, y, w, lambda, &mut z)?;
    Ok(z)
}

/// `whittaker` into `z` (as long as `y`).
pub fn smooth_into(x: &[f64], y: &[f64], w: &[f64], lambda: f64, z: &mut [f64]) -> Result<(), &'static str> {
    let n = y.len();
    z.fill(f64::NAN);
    let mut wy = vec![0.0; n];
    let mut ww = vec![0.0; n];
    let mut weighted = 0;
    for i in 0..n {
        let ok = y[i].is_finite() && w[i].is_finite() && w[i] > 0.0;
        ww[i] = if ok { w[i] } else { 0.0 };
        wy[i] = if ok { w[i] * y[i] } else { 0.0 };
        weighted += ok as usize;
    }
    if weighted < 2 {
        return Ok(());
    }
    if n < 3 || lambda <= 0.0 {
        // nothing to penalise: the observations themselves
        for i in 0..n {
            z[i] = if ww[i] > 0.0 { y[i] } else { f64::NAN };
        }
        return Ok(());
    }

    // The median step, so that an even series has the classic penalty.
    let steps: Vec<f64> = x.windows(2).map(|p| p[1] - p[0]).collect();
    let mut sorted = steps.clone();
    let mid = sorted.len() / 2;
    let (_, &mut median, _) = sorted.select_nth_unstable_by(mid, f64::total_cmp);
    let h0 = if median > 0.0 { median } else { 1.0 };

    // A = W + lambda D'D: symmetric, two diagonals above the main one.
    let mut a0 = ww;
    let mut a1 = vec![0.0; n];
    let mut a2 = vec![0.0; n];
    for k in 0..n - 2 {
        let (h1, h2) = (steps[k] / h0, steps[k + 1] / h0);
        if !(h1 > 0.0) || !(h2 > 0.0) {
            return Err(NOT_INCREASING);
        }
        let c = [2.0 / (h1 * (h1 + h2)), -2.0 / (h1 * h2), 2.0 / (h2 * (h1 + h2))];
        for p in 0..3 {
            a0[k + p] += lambda * c[p] * c[p];
            if p < 2 {
                a1[k + p] += lambda * c[p] * c[p + 1];
            }
        }
        a2[k] += lambda * c[0] * c[2];
    }

    // Banded Cholesky A = L L' (L: main diagonal d, below it l1, two below l2).
    let mut d = vec![0.0; n];
    let mut l1 = vec![0.0; n];
    let mut l2 = vec![0.0; n];
    for i in 0..n {
        if i >= 2 {
            l2[i] = a2[i - 2] / d[i - 2];
        }
        if i >= 1 {
            l1[i] = (a1[i - 1] - if i >= 2 { l2[i] * l1[i - 1] } else { 0.0 }) / d[i - 1];
        }
        let s = a0[i] - l1[i] * l1[i] - l2[i] * l2[i];
        if !(s > 0.0) {
            return Ok(()); // not positive definite (too few weighted observations)
        }
        d[i] = s.sqrt();
    }
    let mut u = vec![0.0; n]; // L u = W y
    for i in 0..n {
        let mut s = wy[i];
        if i >= 1 {
            s -= l1[i] * u[i - 1];
        }
        if i >= 2 {
            s -= l2[i] * u[i - 2];
        }
        u[i] = s / d[i];
    }
    for i in (0..n).rev() {
        // L' z = u
        let mut s = u[i];
        if i + 1 < n {
            s -= l1[i + 1] * z[i + 1];
        }
        if i + 2 < n {
            s -= l2[i + 2] * z[i + 2];
        }
        z[i] = s / d[i];
    }
    Ok(())
}

/// `whittaker` on every row of `values` (`n_pixels` x T, row-major) at the
/// positions `x`, with `weights` of the same shape or none (all 1). An error
/// when a series would raise one (x not increasing).
pub fn batch(
    values: &[f64],
    n_pixels: usize,
    x: &[f64],
    weights: Option<&[f64]>,
    lambda: f64,
    n_jobs: i32,
) -> Result<Vec<f64>, &'static str> {
    let t = x.len();
    assert_eq!(values.len(), n_pixels * t, "values must be pixels x time");
    let ones = vec![1.0; t];
    let failed = std::sync::atomic::AtomicBool::new(false);
    let mut out = vec![0.0; n_pixels * t];
    parallel::for_each_row(
        &mut out,
        t,
        n_jobs,
        || (),
        |_, p, z| {
            let w = weights.map_or(&ones[..], |w| &w[p * t..(p + 1) * t]);
            if smooth_into(x, &values[p * t..(p + 1) * t], w, lambda, z).is_err() {
                failed.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        },
    );
    if failed.into_inner() {
        return Err(NOT_INCREASING);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_stays_a_line() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 8.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 0.5 + 0.01 * v).collect();
        let z = whittaker(&x, &y, &[1.0; 20], 100.0).unwrap();
        assert!(z.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-9));
    }

    #[test]
    fn gaps_and_errors() {
        let x = [0.0, 1.0, 2.0, 3.0];
        let z = whittaker(&x, &[1.0, f64::NAN, f64::NAN, f64::NAN], &[1.0; 4], 10.0).unwrap();
        assert!(z.iter().all(|v| v.is_nan()));
        assert_eq!(
            whittaker(&[0.0, 2.0, 1.0, 3.0], &[1.0; 4], &[1.0; 4], 10.0),
            Err(NOT_INCREASING)
        );
        assert!(batch(&[1.0; 8], 2, &[0.0, 2.0, 1.0, 3.0], None, 10.0, 1).is_err());
    }
}
