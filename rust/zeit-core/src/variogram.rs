//! The range of a map's spatial autocorrelation, the size blockCV's `cv_spatial_autocor` gives the blocks of a
//! spatial cross-validation (Valavi et al. 2019, doi:10.1111/2041-210X.13107): the empirical semivariogram of its
//! values at sampled pixels, and an exponential model fitted to it.
//!
//! The semivariogram of the pairs of points in each band of distance is half their mean squared difference. The
//! model is γ(h) = c0 + c1 (1 - exp(-h / a)): for a fixed scale `a` it is linear in the nugget c0 and the partial sill
//! c1, so those come from weighted least squares (weights N_h / h², gstat's default fit), kept non-negative, and the
//! scale from a search over its logarithm. The range is the practical one, where the model reaches 95% of its sill
//! above the nugget: a ln 20 (about 3a).

use crate::parallel;

/// The empirical semivariogram: for each band of distance, its pairs' mean distance, semivariance and count (bands
/// without pairs left out).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Empirical {
    pub distance: Vec<f64>,
    pub gamma: Vec<f64>,
    pub pairs: Vec<f64>,
}

/// An exponential semivariogram.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exponential {
    pub nugget: f64,
    /// the partial sill (the sill above the nugget)
    pub sill: f64,
    /// a, in exp(-h / a)
    pub scale: f64,
    /// the practical range, a ln 20
    pub range: f64,
}

/// The semivariogram of the values `v` at the points (`x`, `y`), in `bins` equal bands of distance up to `cutoff`
/// (pairs farther left out).
pub fn empirical(x: &[f64], y: &[f64], v: &[f64], bins: usize, cutoff: f64, n_jobs: i32) -> Empirical {
    let n = v.len();
    assert!(x.len() == n && y.len() == n, "x, y and v must have the same length");
    if n < 2 || bins == 0 || !(cutoff > 0.0) {
        return Empirical::default();
    }
    // each point's sums over the points after it, per band: distance, squared difference, count
    let width = 3 * bins;
    let mut rows = vec![0.0; n * width];
    parallel::for_each_row(
        &mut rows,
        width,
        n_jobs,
        || (),
        |_, i, out| {
            for j in i + 1..n {
                let d = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
                if d >= cutoff {
                    continue;
                }
                let b = ((d / cutoff * bins as f64) as usize).min(bins - 1);
                out[3 * b] += d;
                out[3 * b + 1] += (v[i] - v[j]).powi(2);
                out[3 * b + 2] += 1.0;
            }
        },
    );
    let mut sums = vec![0.0; width];
    for row in rows.chunks(width) {
        for (s, r) in sums.iter_mut().zip(row) {
            *s += r;
        }
    }
    let mut e = Empirical::default();
    for b in 0..bins {
        let c = sums[3 * b + 2];
        if c > 0.0 {
            e.distance.push(sums[3 * b] / c);
            e.gamma.push(sums[3 * b + 1] / (2.0 * c));
            e.pairs.push(c);
        }
    }
    e
}

/// For a fixed scale, the nugget and partial sill of least weighted squared error, both >= 0, and that error.
fn linear_part(e: &Empirical, w: &[f64], scale: f64) -> (f64, f64, f64) {
    let f: Vec<f64> = e.distance.iter().map(|&h| 1.0 - (-h / scale).exp()).collect();
    let error = |c0: f64, c1: f64| -> f64 {
        e.gamma
            .iter()
            .zip(&f)
            .zip(w)
            .map(|((g, fi), wi)| wi * (g - c0 - c1 * fi).powi(2))
            .sum()
    };
    let (sw, sf, sff, sg, sfg) =
        e.gamma
            .iter()
            .zip(&f)
            .zip(w)
            .fold((0.0, 0.0, 0.0, 0.0, 0.0), |(sw, sf, sff, sg, sfg), ((g, fi), wi)| {
                (
                    sw + wi,
                    sf + wi * fi,
                    sff + wi * fi * fi,
                    sg + wi * g,
                    sfg + wi * fi * g,
                )
            });
    let det = sw * sff - sf * sf;
    let mut best = (0.0, 0.0, f64::INFINITY);
    let mut consider = |c0: f64, c1: f64| {
        if c0 >= 0.0 && c1 >= 0.0 && c0.is_finite() && c1.is_finite() {
            let err = error(c0, c1);
            if err < best.2 {
                best = (c0, c1, err);
            }
        }
    };
    if det > 0.0 {
        consider((sff * sg - sf * sfg) / det, (sw * sfg - sf * sg) / det);
    }
    // on the boundary: no nugget, or no structure
    if sff > 0.0 {
        consider(0.0, sfg / sff);
    }
    consider(sg / sw, 0.0);
    best
}

/// The exponential model fitted to `e`, or None without at least three bands of distance.
pub fn fit_exponential(e: &Empirical) -> Option<Exponential> {
    let k = e.gamma.len();
    if k < 3 {
        return None;
    }
    let w: Vec<f64> = e
        .distance
        .iter()
        .zip(&e.pairs)
        .map(|(&h, &c)| c / (h * h).max(1e-300))
        .collect();
    let (lo, hi) = (e.distance[0], e.distance[k - 1]);
    // the scale between a tenth of the first band and ten times the last: a grid of its log, then golden section
    let (a, b) = ((lo / 10.0).max(1e-12).ln(), (hi * 10.0).ln());
    let err = |t: f64| linear_part(e, &w, t.exp()).2;
    let steps = 200;
    let grid: Vec<f64> = (0..=steps).map(|i| a + (b - a) * i as f64 / steps as f64).collect();
    let best = (0..=steps).min_by(|&i, &j| err(grid[i]).total_cmp(&err(grid[j])))?;
    let (mut l, mut r) = (grid[best.saturating_sub(1)], grid[(best + 1).min(steps)]);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (r - g * (r - l), l + g * (r - l));
    for _ in 0..80 {
        if err(c) < err(d) {
            r = d;
        } else {
            l = c;
        }
        c = r - g * (r - l);
        d = l + g * (r - l);
    }
    let scale = ((l + r) / 2.0).exp();
    let (nugget, sill, _) = linear_part(e, &w, scale);
    Some(Exponential {
        nugget,
        sill,
        scale,
        range: scale * 20f64.ln(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_by_hand() {
        // three points on a line, 0, 1 and 3 apart: pairs at 1, 2 and 3
        let (x, y, v) = ([0.0, 1.0, 3.0], [0.0; 3], [0.0, 2.0, 1.0]);
        let e = empirical(&x, &y, &v, 3, 3.5, 1);
        assert_eq!(e.pairs, vec![1.0, 1.0, 1.0]);
        assert_eq!(e.distance, vec![1.0, 2.0, 3.0]);
        assert_eq!(e.gamma, vec![2.0, 0.5, 0.5]);
        // the last pair beyond the cutoff
        let e = empirical(&x, &y, &v, 2, 2.5, 4);
        assert_eq!(e.pairs, vec![1.0, 1.0]);
    }

    #[test]
    fn the_model_is_recovered() {
        // a semivariogram drawn from the model itself, its pairs growing with distance
        let truth = Exponential {
            nugget: 0.2,
            sill: 1.5,
            scale: 120.0,
            range: 120.0 * 20f64.ln(),
        };
        let distance: Vec<f64> = (1..=15).map(|i| i as f64 * 40.0).collect();
        let gamma = distance
            .iter()
            .map(|&h| truth.nugget + truth.sill * (1.0 - (-h / truth.scale).exp()))
            .collect();
        let pairs = distance.iter().map(|&h| 100.0 * h).collect();
        let fit = fit_exponential(&Empirical { distance, gamma, pairs }).unwrap();
        assert!((fit.scale / truth.scale - 1.0).abs() < 1e-6, "{fit:?}");
        assert!((fit.nugget - truth.nugget).abs() < 1e-6 && (fit.sill - truth.sill).abs() < 1e-6);
        assert!((fit.range / truth.range - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_field_of_known_range() {
        // white noise on a grid of 20-unit pixels, and its mean over 5 x 5 pixels (100 units): the mean is
        // correlated up to 100 units apart, and not beyond
        let (g, w) = (70usize, 5usize);
        let mut s = 9u64;
        let noise: Vec<f64> = (0..g * g)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let m = g - w + 1;
        let (mut x, mut y, mut smooth, mut white) = (vec![], vec![], vec![], vec![]);
        for i in 0..m {
            for j in 0..m {
                x.push(i as f64 * 20.0);
                y.push(j as f64 * 20.0);
                let sum: f64 = (0..w)
                    .flat_map(|a| (0..w).map(move |b| (a, b)))
                    .map(|(a, b)| noise[(i + a) * g + j + b])
                    .sum();
                smooth.push(sum / (w * w) as f64);
                white.push(noise[i * g + j]);
            }
        }
        let e = empirical(&x, &y, &smooth, 15, 400.0, 4);
        assert_eq!(e, empirical(&x, &y, &smooth, 15, 400.0, 1));
        let fit = fit_exponential(&e).unwrap();
        // an exponential reaches 95% of its sill only at about 3a: its practical range is beyond the 100 units
        assert!(
            fit.scale > 40.0 && fit.scale < 120.0 && fit.range > 100.0 && fit.range < 350.0,
            "{fit:?}"
        );
        // white noise: no structure beyond a pixel, all nugget
        let fit = fit_exponential(&empirical(&x, &y, &white, 15, 400.0, 4)).unwrap();
        assert!(fit.sill < 0.1 * fit.nugget || fit.range < 20.0, "{fit:?}");
    }
}
