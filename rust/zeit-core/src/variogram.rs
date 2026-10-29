//! The range of a map's spatial autocorrelation, the size blockCV's `cv_spatial_autocor` gives the blocks of a
//! spatial cross-validation (Valavi et al. 2019, doi:10.1111/2041-210X.13107): the empirical semivariogram of its
//! values at sampled pixels, and the model fitted to it that fits best, of the spherical, exponential and Gaussian
//! (as automap's `autofitVariogram`, which blockCV calls, picks among its models).
//!
//! The semivariogram of the pairs of points in each band of distance is half their mean squared difference. Each
//! model is γ(h) = c0 + c1 f(h / a): for a fixed scale `a` it is linear in the nugget c0 and the partial sill c1, so
//! those come from weighted least squares (weights N_h / h², gstat's default fit), kept non-negative, and the scale
//! from a search over its logarithm. The range is the distance beyond which the points are about independent: the
//! spherical's a (where it reaches its sill), the exponential's and the Gaussian's practical range (where they
//! reach 95% of it, a ln 20 and a √ln 20).

use crate::parallel;

/// The empirical semivariogram: for each band of distance, its pairs' mean distance, semivariance and count (bands
/// without pairs left out).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Empirical {
    pub distance: Vec<f64>,
    pub gamma: Vec<f64>,
    pub pairs: Vec<f64>,
}

/// A model of semivariogram, f(u) of u = h / a.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 1.5 u - 0.5 u³ up to u = 1, then 1
    Spherical,
    /// 1 - exp(-u)
    Exponential,
    /// 1 - exp(-u²)
    Gaussian,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Spherical, Kind::Exponential, Kind::Gaussian];

    fn f(self, u: f64) -> f64 {
        match self {
            Kind::Spherical if u >= 1.0 => 1.0,
            Kind::Spherical => 1.5 * u - 0.5 * u * u * u,
            Kind::Exponential => 1.0 - (-u).exp(),
            Kind::Gaussian => 1.0 - (-u * u).exp(),
        }
    }

    /// The range, from the scale.
    fn range(self, a: f64) -> f64 {
        match self {
            Kind::Spherical => a,
            Kind::Exponential => a * 20f64.ln(),
            Kind::Gaussian => a * 20f64.ln().sqrt(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Spherical => "spherical",
            Kind::Exponential => "exponential",
            Kind::Gaussian => "gaussian",
        }
    }
}

/// A fitted semivariogram.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    pub kind: Kind,
    pub nugget: f64,
    /// the partial sill (the sill above the nugget)
    pub sill: f64,
    /// a, in f(h / a)
    pub scale: f64,
    /// where the points are about independent (see the module)
    pub range: f64,
    /// the weighted sum of squared errors of the fit
    pub error: f64,
}

/// The bands of distance of automap's `autofitVariogram` (which blockCV calls): their upper ends at 2, 4, 6, 9, 12,
/// 15, 25, 35, 50, 65, 80 and 100% of `cutoff`, finer at short distances, where a short range shows.
pub fn automap_boundaries(cutoff: f64) -> Vec<f64> {
    [2.0, 4.0, 6.0, 9.0, 12.0, 15.0, 25.0, 35.0, 50.0, 65.0, 80.0, 100.0]
        .iter()
        .map(|f| f / 100.0 * cutoff)
        .collect()
}

/// The semivariogram of the values `v` at the points (`x`, `y`), in the bands of distance whose upper ends are
/// `boundaries` (increasing; pairs beyond the last left out).
pub fn empirical(x: &[f64], y: &[f64], v: &[f64], boundaries: &[f64], n_jobs: i32) -> Empirical {
    let n = v.len();
    assert!(x.len() == n && y.len() == n, "x, y and v must have the same length");
    let bins = boundaries.len();
    if n < 2 || bins == 0 || !(boundaries[0] > 0.0) || boundaries.windows(2).any(|w| !(w[1] > w[0])) {
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
                let b = boundaries.partition_point(|&e| e <= d);
                if b == bins {
                    continue;
                }
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
fn linear_part(e: &Empirical, w: &[f64], kind: Kind, scale: f64) -> (f64, f64, f64) {
    let f: Vec<f64> = e.distance.iter().map(|&h| kind.f(h / scale)).collect();
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

/// The model `kind` fitted to `e`, or None without at least three bands of distance.
pub fn fit(e: &Empirical, kind: Kind) -> Option<Model> {
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
    let err = |t: f64| linear_part(e, &w, kind, t.exp()).2;
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
    let (nugget, sill, error) = linear_part(e, &w, kind, scale);
    Some(Model {
        kind,
        nugget,
        sill,
        scale,
        range: kind.range(scale),
        error,
    })
}

/// Of the spherical, exponential and Gaussian models fitted to `e`, the one of least weighted squared error.
pub fn fit_best(e: &Empirical) -> Option<Model> {
    Kind::ALL
        .iter()
        .filter_map(|&k| fit(e, k))
        .min_by(|a, b| a.error.total_cmp(&b.error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_by_hand() {
        // three points on a line, 0, 1 and 3 apart: pairs at 1, 2 and 3
        let (x, y, v) = ([0.0, 1.0, 3.0], [0.0; 3], [0.0, 2.0, 1.0]);
        let e = empirical(&x, &y, &v, &[1.5, 2.5, 3.5], 1);
        assert_eq!(e.pairs, vec![1.0, 1.0, 1.0]);
        assert_eq!(e.distance, vec![1.0, 2.0, 3.0]);
        assert_eq!(e.gamma, vec![2.0, 0.5, 0.5]);
        // the last pair beyond the cutoff
        let e = empirical(&x, &y, &v, &[1.25, 2.5], 4);
        assert_eq!(e.pairs, vec![1.0, 1.0]);
    }

    #[test]
    fn the_model_is_recovered() {
        // a semivariogram drawn from each model itself, its pairs growing with distance: recovered, and chosen
        for kind in Kind::ALL {
            let (nugget, sill, scale) = (0.2, 1.5, 120.0);
            let distance: Vec<f64> = (1..=15).map(|i| i as f64 * 25.0).collect();
            let gamma = distance.iter().map(|&h| nugget + sill * kind.f(h / scale)).collect();
            let pairs = distance.iter().map(|&h| 100.0 * h).collect();
            let e = Empirical { distance, gamma, pairs };
            let m = fit(&e, kind).unwrap();
            assert!((m.scale / scale - 1.0).abs() < 1e-6, "{m:?}");
            assert!((m.nugget - nugget).abs() < 1e-6 && (m.sill - sill).abs() < 1e-6);
            assert!((m.range / kind.range(scale) - 1.0).abs() < 1e-6);
            assert_eq!(fit_best(&e).unwrap().kind, kind);
        }
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
        let bands = automap_boundaries(400.0);
        let e = empirical(&x, &y, &smooth, &bands, 4);
        assert_eq!(e, empirical(&x, &y, &smooth, &bands, 1));
        // the mean over a square is correlated as a tent, which reaches its sill: the best of the models with a range
        // near the 100 units (the spherical or the Gaussian, both flatter at the end than the exponential)
        let m = fit_best(&e).unwrap();
        assert!(
            m.kind != Kind::Exponential && m.range > 90.0 && m.range < 160.0,
            "{m:?}"
        );
        // an exponential reaches 95% of its sill only at about 3a: its practical range is beyond
        let m = fit(&e, Kind::Exponential).unwrap();
        assert!(
            m.scale > 40.0 && m.scale < 120.0 && m.range > 100.0 && m.range < 350.0,
            "{m:?}"
        );
        // white noise: no structure beyond a pixel, all nugget
        let fit = fit_best(&empirical(&x, &y, &white, &bands, 4)).unwrap();
        assert!(fit.sill < 0.1 * fit.nugget || fit.range < 20.0, "{fit:?}");
    }
}
