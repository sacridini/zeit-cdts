//! A monotone curve of a model's error against the dissimilarity index (CAST's `errorProfiles`, Meyer & Pebesma
//! 2021): a P-spline (cubic B-splines on equally spaced knots, a penalty on the second differences of their
//! coefficients, Eilers & Marx 1996) whose coefficients are kept non-decreasing (or non-increasing) by a large
//! penalty on the first differences that go the wrong way, iterated until the set of those is stable (Bollaerts,
//! Eilers & van Mechelen 2006, doi:10.1348/000711005X84293); monotone coefficients make a monotone curve. The weight
//! of the smoothness penalty is chosen by generalised cross-validation. Beyond the knots the curve goes on as a line
//! with its slope at the end. CAST fits a shape-constrained additive model (scam) instead: the curves are alike,
//! not equal.

use crate::parallel;

/// A fitted monotone curve.
#[derive(Clone, Debug, PartialEq)]
pub struct Monotone {
    /// the range of the knots (the data's x)
    pub lo: f64,
    pub hi: f64,
    /// the B-splines' coefficients
    pub coef: Vec<f64>,
    /// the smoothness penalty's weight chosen
    pub lambda: f64,
    /// effective degrees of freedom (the trace of the hat matrix)
    pub edf: f64,
}

/// The weight of the penalty on the differences going the wrong way.
const KAPPA: f64 = 1e8;

/// The values of the `k` cubic B-splines at `x` (inside [lo, hi]), on equally spaced knots.
fn basis(x: f64, lo: f64, hi: f64, k: usize, out: &mut [f64]) {
    let segments = k - 3;
    let h = (hi - lo) / segments as f64;
    let u = ((x - lo) / h).clamp(0.0, segments as f64);
    let seg = (u.floor() as usize).min(segments - 1);
    let t = u - seg as f64;
    out.fill(0.0);
    // the uniform cubic B-spline's four pieces on [seg, seg + 1)
    let t2 = t * t;
    let t3 = t2 * t;
    out[seg] = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    out[seg + 1] = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    out[seg + 2] = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    out[seg + 3] = t3 / 6.0;
}

impl Monotone {
    fn k(&self) -> usize {
        self.coef.len()
    }

    fn inside(&self, x: f64) -> f64 {
        let mut b = vec![0.0; self.k()];
        basis(x, self.lo, self.hi, self.k(), &mut b);
        b.iter().zip(&self.coef).map(|(a, c)| a * c).sum()
    }

    /// The slope at an end (`at_hi`) of the knots.
    fn end_slope(&self, at_hi: bool) -> f64 {
        let h = (self.hi - self.lo) / (self.k() - 3) as f64;
        // the derivative of a uniform cubic B-spline curve at a knot: (c[i+2] - c[i]) / (2h)
        let c = &self.coef;
        let n = c.len();
        if at_hi {
            (c[n - 1] - c[n - 3]) / (2.0 * h)
        } else {
            (c[2] - c[0]) / (2.0 * h)
        }
    }

    /// The curve at `x` (NaN at NaN).
    pub fn at(&self, x: f64) -> f64 {
        if x.is_nan() {
            f64::NAN
        } else if x > self.hi {
            self.inside(self.hi) + self.end_slope(true) * (x - self.hi)
        } else if x < self.lo {
            self.inside(self.lo) + self.end_slope(false) * (x - self.lo)
        } else {
            self.inside(x)
        }
    }

    /// The curve at each of `x`, in parallel.
    pub fn predict(&self, x: &[f64], n_jobs: i32) -> Vec<f64> {
        let mut out = vec![f64::NAN; x.len()];
        parallel::for_each_row(&mut out, 1, n_jobs, || (), |_, i, o| o[0] = self.at(x[i]));
        out
    }
}

/// Solves the symmetric positive definite system `a` x = `b` (k x k, row-major) by Cholesky; None if singular.
fn solve(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; k * k];
    for j in 0..k {
        let d = a[j * k + j] - (0..j).map(|m| l[j * k + m] * l[j * k + m]).sum::<f64>();
        if !(d > 0.0) {
            return None;
        }
        l[j * k + j] = d.sqrt();
        for i in j + 1..k {
            l[i * k + j] = (a[i * k + j] - (0..j).map(|m| l[i * k + m] * l[j * k + m]).sum::<f64>()) / l[j * k + j];
        }
    }
    let mut y = b.to_vec();
    for i in 0..k {
        y[i] = (y[i] - (0..i).map(|m| l[i * k + m] * y[m]).sum::<f64>()) / l[i * k + i];
    }
    for i in (0..k).rev() {
        y[i] = (y[i] - (i + 1..k).map(|m| l[m * k + i] * y[m]).sum::<f64>()) / l[i * k + i];
    }
    Some(y)
}

/// D'D of the differences of order `order` of k coefficients (k x k), with each first difference weighted by `v`
/// for order 1.
fn difference_penalty(k: usize, order: usize, v: Option<&[f64]>) -> Vec<f64> {
    let rows: Vec<Vec<f64>> = match order {
        1 => (0..k - 1)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if j == i {
                            -1.0
                        } else if j == i + 1 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect(),
        _ => (0..k - 2)
            .map(|i| {
                (0..k)
                    .map(|j| match j as isize - i as isize {
                        0 | 2 => 1.0,
                        1 => -2.0,
                        _ => 0.0,
                    })
                    .collect()
            })
            .collect(),
    };
    let mut p = vec![0.0; k * k];
    for (r, row) in rows.iter().enumerate() {
        let w = v.map_or(1.0, |v| v[r]);
        if w == 0.0 {
            continue;
        }
        for a in 0..k {
            for b in 0..k {
                p[a * k + b] += w * row[a] * row[b];
            }
        }
    }
    p
}

/// The monotone P-spline of `y` on `x` (`increasing`, or decreasing) with `k` B-splines (at least 4; fewer when
/// the data are few), its smoothness by generalised cross-validation; None without 4 distinct x.
pub fn fit_monotone(x: &[f64], y: &[f64], increasing: bool, k: usize) -> Option<Monotone> {
    let pts: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(&a, &b)| (a, b))
        .collect();
    let n = pts.len();
    let (lo, hi) = pts
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| (l.min(p.0), h.max(p.0)));
    let mut distinct: Vec<f64> = pts.iter().map(|p| p.0).collect();
    distinct.sort_by(f64::total_cmp);
    distinct.dedup();
    if distinct.len() < 4 || !(hi > lo) {
        return None;
    }
    let k = k.clamp(4, (distinct.len()).max(4));
    // B'B and B'y
    let mut btb = vec![0.0; k * k];
    let mut bty = vec![0.0; k];
    let mut rows = vec![0.0; n * k];
    for (i, &(xi, yi)) in pts.iter().enumerate() {
        let b = &mut rows[i * k..(i + 1) * k];
        basis(xi, lo, hi, k, b);
        for a in 0..k {
            bty[a] += b[a] * yi;
            for c in 0..k {
                btb[a * k + c] += b[a] * b[c];
            }
        }
    }
    let d2 = difference_penalty(k, 2, None);
    // lambda relative to the scale of B'B against the penalty's
    let scale = (0..k).map(|i| btb[i * k + i]).sum::<f64>() / (0..k).map(|i| d2[i * k + i]).sum::<f64>().max(1e-300);
    let sign = if increasing { 1.0 } else { -1.0 };
    let fit_at = |lambda: f64| -> Option<(Vec<f64>, f64, f64)> {
        let mut v = vec![0.0; k - 1];
        let mut coef = Vec::new();
        let mut system = vec![0.0; k * k];
        for _ in 0..100 {
            let d1 = difference_penalty(k, 1, Some(&v));
            for i in 0..k * k {
                system[i] = btb[i] + lambda * scale * d2[i] + KAPPA * scale * d1[i];
            }
            // a ridge too small to change the fit, for a system of fewer points than B-splines
            for i in 0..k {
                system[i * k + i] += 1e-10 * scale;
            }
            coef = solve(&system, &bty, k)?;
            let wrong: Vec<f64> = (0..k - 1)
                .map(|i| if sign * (coef[i + 1] - coef[i]) < 0.0 { 1.0 } else { 0.0 })
                .collect();
            if wrong == v {
                break;
            }
            v = wrong;
        }
        // the hat matrix's trace, tr(B (B'B + P)^-1 B') = tr((B'B + P)^-1 B'B), column by column
        let mut trace = 0.0;
        for c in 0..k {
            let col: Vec<f64> = (0..k).map(|r| btb[r * k + c]).collect();
            trace += solve(&system, &col, k)?[c];
        }
        let rss: f64 = (0..n)
            .map(|i| {
                let f: f64 = rows[i * k..(i + 1) * k].iter().zip(&coef).map(|(b, c)| b * c).sum();
                (pts[i].1 - f).powi(2)
            })
            .sum();
        let gcv = n as f64 * rss / (n as f64 - trace).max(1e-9).powi(2);
        Some((coef, trace, gcv))
    };
    let mut best: Option<(f64, Vec<f64>, f64, f64)> = None;
    for e in 0..=48 {
        let lambda = 10f64.powf(-6.0 + e as f64 * 0.25);
        if let Some((coef, edf, gcv)) = fit_at(lambda) {
            if best.as_ref().is_none_or(|b| gcv < b.3) {
                best = Some((lambda, coef, edf, gcv));
            }
        }
    }
    let (lambda, mut coef, edf, _) = best?;
    // the penalty leaves about 1/KAPPA of a difference going the wrong way: none, exactly
    for i in 1..k {
        coef[i] = if increasing {
            coef[i].max(coef[i - 1])
        } else {
            coef[i].min(coef[i - 1])
        };
    }
    Some(Monotone {
        lo,
        hi,
        coef,
        lambda,
        edf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    #[test]
    fn the_basis_sums_to_one_and_a_line_is_a_line() {
        let mut b = vec![0.0; 8];
        for i in 0..=50 {
            basis(i as f64 / 50.0, 0.0, 1.0, 8, &mut b);
            assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-14 && b.iter().all(|&v| v >= 0.0));
        }
        // a straight line is fitted exactly (the second differences of its coefficients are 0)
        let x: Vec<f64> = (0..40).map(|i| i as f64 / 4.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 + 0.5 * v).collect();
        let m = fit_monotone(&x, &y, true, 8).unwrap();
        for &v in &[0.0, 3.3, 9.75] {
            assert!((m.at(v) - (2.0 + 0.5 * v)).abs() < 1e-6, "{} {}", v, m.at(v));
        }
        // and goes on as the line beyond its data
        assert!((m.at(15.0) - 9.5).abs() < 1e-5 && (m.at(-2.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn monotone_however_noisy() {
        // a curve that rises and flattens, with noise that makes the raw data go down and up
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let e = noise(n, 4);
        let y: Vec<f64> = x
            .iter()
            .zip(&e)
            .map(|(v, e)| 1.0 - (-4.0 * v).exp() + 0.6 * e)
            .collect();
        let m = fit_monotone(&x, &y, true, 10).unwrap();
        let grid: Vec<f64> = (0..=400).map(|i| -0.2 + i as f64 / 300.0).collect();
        let f = m.predict(&grid, 2);
        assert!(f.windows(2).all(|w| w[1] >= w[0] - 1e-9), "not increasing");
        // near the truth where the data are
        for (&g, &v) in grid.iter().zip(&f).filter(|(g, _)| (0.05..0.95).contains(*g)) {
            assert!((v - (1.0 - (-4.0 * g).exp())).abs() < 0.15, "{g}: {v}");
        }
        // decreasing: the mirror image
        let yd: Vec<f64> = y.iter().map(|v| -v).collect();
        let md = fit_monotone(&x, &yd, false, 10).unwrap();
        let fd = md.predict(&grid, 1);
        assert!(fd.windows(2).all(|w| w[1] <= w[0] + 1e-9));
        assert!(f.iter().zip(&fd).all(|(a, b)| (a + b).abs() < 1e-6));
        // data going the other way: a flat line at their mean
        let flat = fit_monotone(&x, &yd, true, 10).unwrap();
        let ff = flat.predict(&x, 1);
        assert!(ff.windows(2).all(|w| w[1] >= w[0] - 1e-9));
        assert!(fit_monotone(&[1.0, 1.0, 2.0], &[1.0, 2.0, 3.0], true, 6).is_none());
    }
}
