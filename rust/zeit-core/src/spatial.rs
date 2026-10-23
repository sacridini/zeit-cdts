//! The spatial correlation of PARTS (Ives et al. 2021): distances between
//! pixels, correlation functions of distance (remotePARTS's covar_exp and
//! covar_exppow), and the fit of one to the correlations between the
//! residual series of pixels (remotePARTS's fitCor).

use rayon::prelude::*;

use crate::parallel;

/// A correlation as a function of distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Correlation {
    /// exp(-d / range)
    Exponential { range: f64 },
    /// exp(-(d / range)^shape)
    ExponentialPower { range: f64, shape: f64 },
}

impl Correlation {
    pub fn at(&self, d: f64) -> f64 {
        match *self {
            Correlation::Exponential { range } => (-d / range).exp(),
            Correlation::ExponentialPower { range, shape } => (-(d / range).powf(shape)).exp(),
        }
    }

    fn params(&self) -> Vec<f64> {
        match *self {
            Correlation::Exponential { range } => vec![range],
            Correlation::ExponentialPower { range, shape } => vec![range, shape],
        }
    }

    fn with_params(&self, p: &[f64]) -> Correlation {
        match self {
            Correlation::Exponential { .. } => Correlation::Exponential { range: p[0] },
            Correlation::ExponentialPower { .. } => Correlation::ExponentialPower {
                range: p[0],
                shape: p[1],
            },
        }
    }

    /// The value at `d` and its derivatives in the logs of the parameters.
    fn value_and_gradient(&self, d: f64, grad: &mut [f64]) -> f64 {
        match *self {
            Correlation::Exponential { range } => {
                let u = d / range;
                let f = (-u).exp();
                grad[0] = f * u;
                f
            }
            Correlation::ExponentialPower { range, shape } => {
                let ratio = d / range;
                if ratio == 0.0 {
                    grad[0] = 0.0;
                    grad[1] = 0.0;
                    return 1.0;
                }
                let u = ratio.powf(shape);
                let f = (-u).exp();
                grad[0] = f * u * shape;
                grad[1] = -f * u * ratio.ln() * shape;
                f
            }
        }
    }
}

/// The Euclidean distances between the points `a` and `b` (|a| x |b|, by
/// column), in the coordinates' units; with `scaled`, divided by the largest
/// of them (remotePARTS's distm_scaled).
pub fn distances(ax: &[f64], ay: &[f64], bx: &[f64], by: &[f64], scaled: bool) -> Vec<f64> {
    let na = ax.len();
    let mut d = vec![0.0; na * bx.len()];
    for (col, (&x, &y)) in d.chunks_mut(na.max(1)).zip(bx.iter().zip(by)) {
        for ((v, &xa), &ya) in col.iter_mut().zip(ax).zip(ay) {
            let (dx, dy) = (xa - x, ya - y);
            *v = (dx * dx + dy * dy).sqrt();
        }
    }
    if scaled {
        let m = d.iter().copied().fold(0.0, f64::max);
        if m > 0.0 {
            for v in &mut d {
                *v /= m;
            }
        }
    }
    d
}

/// The correlations between the points `a` and `b` (|a| x |b|, by column).
pub fn correlations(ax: &[f64], ay: &[f64], bx: &[f64], by: &[f64], cor: Correlation, scaled: bool) -> Vec<f64> {
    let mut v = distances(ax, ay, bx, by, scaled);
    for d in &mut v {
        *d = cor.at(*d);
    }
    v
}

/// A correlation function fitted to the correlations of residual series.
#[derive(Clone, Debug, PartialEq)]
pub struct CorFit {
    pub cor: Correlation,
    /// The largest distance between the points (what scaled distances are
    /// divided by).
    pub max_distance: f64,
    /// The residual sum of squares of the correlations.
    pub sse: f64,
    pub pairs: usize,
}

/// Pearson's correlation of two series over the dates where both are valid.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (mut n, mut sa, mut sb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_nan() && !y.is_nan() {
            n += 1.0;
            sa += x;
            sb += y;
        }
    }
    if n < 2.0 {
        return f64::NAN;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        if !x.is_nan() && !y.is_nan() {
            let (u, v) = (x - ma, y - mb);
            sab += u * v;
            saa += u * u;
            sbb += v * v;
        }
    }
    sab / (saa * sbb).sqrt()
}

/// Fits `start`'s function, from its parameters, to the correlations between
/// the residual series `resids` (`x.len()` points x `n_time`, row-major; NaN
/// missing) of the points (`x`, `y`) as a function of their distance, by least
/// squares over every pair (remotePARTS's fitCor, which samples the points
/// first: give it the sample). None when it does not converge.
pub fn fit_correlation(
    resids: &[f64],
    n_time: usize,
    x: &[f64],
    y: &[f64],
    start: Correlation,
    scaled: bool,
    n_jobs: i32,
) -> Option<CorFit> {
    let n = x.len();
    assert_eq!(resids.len(), n * n_time, "resids must be len(x) x n_time");
    let d = distances(x, y, x, y, false);
    let max_distance = d.iter().copied().fold(0.0, f64::max);
    let scale = if scaled && max_distance > 0.0 {
        1.0 / max_distance
    } else {
        1.0
    };
    // the pairs of the upper triangle with the diagonal, as fitCor
    let threads = parallel::threads(n_jobs);
    let pairs: Vec<(f64, f64)> = parallel::pool(threads).install(|| {
        (0..n)
            .into_par_iter()
            .flat_map_iter(|j| {
                let rj = &resids[j * n_time..(j + 1) * n_time];
                let d = &d;
                (0..=j).filter_map(move |i| {
                    let c = pearson(&resids[i * n_time..(i + 1) * n_time], rj);
                    c.is_finite().then(|| (d[j * n + i] * scale, c))
                })
            })
            .collect()
    });
    let mut params: Vec<f64> = start.params().iter().map(|v| v.ln()).collect();
    let k = params.len();
    // the sum of squares, and J'J and J'r in the logs of the parameters
    let normal = |params: &[f64]| {
        let cor = start.with_params(&params.iter().map(|v| v.exp()).collect::<Vec<_>>());
        parallel::pool(threads).install(|| {
            pairs
                .par_chunks(4096)
                .map(|chunk| {
                    let (mut sse, mut jtj, mut jtr, mut g) = (0.0, [0.0; 4], [0.0; 2], [0.0; 2]);
                    for &(d, c) in chunk {
                        let f = cor.value_and_gradient(d, &mut g);
                        let r = c - f;
                        sse += r * r;
                        for a in 0..k {
                            jtr[a] += g[a] * r;
                            for b in 0..k {
                                jtj[a * 2 + b] += g[a] * g[b];
                            }
                        }
                    }
                    (sse, jtj, jtr)
                })
                .reduce(
                    || (0.0, [0.0; 4], [0.0; 2]),
                    |a, b| {
                        let mut jtj = a.1;
                        for (x, y) in jtj.iter_mut().zip(b.1) {
                            *x += y;
                        }
                        (a.0 + b.0, jtj, [a.2[0] + b.2[0], a.2[1] + b.2[1]])
                    },
                )
        })
    };
    // Levenberg-Marquardt
    let (mut sse, mut jtj, mut jtr) = normal(&params);
    let mut lambda = 1e-3;
    let mut converged = false;
    for _ in 0..500 {
        let mut a = jtj;
        for i in 0..k {
            a[i * 2 + i] *= 1.0 + lambda;
        }
        let step = if k == 1 {
            [jtr[0] / a[0], 0.0]
        } else {
            let det = a[0] * a[3] - a[1] * a[2];
            [
                (a[3] * jtr[0] - a[1] * jtr[1]) / det,
                (a[0] * jtr[1] - a[2] * jtr[0]) / det,
            ]
        };
        if !step[0].is_finite() || !step[1].is_finite() {
            return None;
        }
        let trial: Vec<f64> = (0..k).map(|i| params[i] + step[i]).collect();
        let (s, j, r) = normal(&trial);
        if s <= sse {
            let small = (0..k).all(|i| step[i].abs() < 1e-12 * (1.0 + params[i].abs()));
            let flat = sse - s <= 1e-15 * sse;
            params = trial;
            (sse, jtj, jtr) = (s, j, r);
            lambda = (lambda / 10.0).max(1e-12);
            if small || flat {
                converged = true;
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e12 {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return None;
    }
    let cor = start.with_params(&params.iter().map(|v| v.exp()).collect::<Vec<_>>());
    Some(CorFit {
        cor,
        max_distance,
        sse,
        pairs: pairs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distances_scaled_and_not() {
        let d = distances(&[0.0, 3.0], &[0.0, 4.0], &[0.0], &[0.0], false);
        assert_eq!(d, vec![0.0, 5.0]);
        let d = distances(
            &[0.0, 3.0, 6.0],
            &[0.0, 4.0, 8.0],
            &[0.0, 3.0, 6.0],
            &[0.0, 4.0, 8.0],
            true,
        );
        assert_eq!(d[2], 1.0);
        assert_eq!(d[1], 0.5);
    }

    #[test]
    fn gradients_in_the_logs_of_the_parameters() {
        let cor = Correlation::ExponentialPower {
            range: 0.25,
            shape: 1.4,
        };
        let mut g = [0.0; 2];
        let f = cor.value_and_gradient(0.1, &mut g);
        let h = 1e-6;
        let up = Correlation::ExponentialPower {
            range: 0.25 * (1.0 + h),
            shape: 1.4,
        }
        .at(0.1);
        assert!(((up - f) / h - g[0]).abs() < 1e-5);
        let up = Correlation::ExponentialPower {
            range: 0.25,
            shape: 1.4 * (1.0 + h),
        }
        .at(0.1);
        assert!(((up - f) / h - g[1]).abs() < 1e-5);
    }
}
