//! Tail probabilities of Student's t and Fisher's F, for the p-values of
//! regressions (R's `pt` and `pf`), from the regularized incomplete beta
//! function (its continued fraction, DLMF 8.17.22).

use crate::cmath::lgamma;

/// The regularized incomplete beta function I_x(a, b), with y = 1 - x given
/// apart so that x near 1 keeps its precision.
pub fn incomplete_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() || !(a > 0.0) || !(b > 0.0) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    // x^a y^b / (a B(a, b))
    let front = (lgamma(a + b) - lgamma(a) - lgamma(b) + a * x.ln() + b * y.ln()).exp();
    // the continued fraction converges fast below the mean (a + 1) / (a + b + 2)
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_fraction(a, b, x) / a
    } else {
        1.0 - front * beta_fraction(b, a, y) / b
    }
}

/// 1 / (1 + d1 / (1 + d2 / (1 + ...))), the continued fraction of I_x(a, b)
/// (DLMF 8.17.22), by the modified Lentz method (Thompson & Barnett 1986).
fn beta_fraction(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let nonzero = |v: f64| if v.abs() < TINY { TINY } else { v };
    // d_{2m} = m (b - m) x / ((a + 2m - 1)(a + 2m)),
    // d_{2m+1} = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1))
    let d = |j: u32| {
        let m = f64::from(j / 2);
        if j.is_multiple_of(2) {
            m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        } else {
            -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        }
    };
    // the denominator g = 1 + d1 / (1 + d2 / ...), its convergents f_j = f_{j-1} C_j D_j
    let (mut g, mut c, mut den) = (1.0, 1.0, 0.0);
    for j in 1..20_000 {
        let dj = d(j);
        den = 1.0 / nonzero(1.0 + dj * den);
        c = nonzero(1.0 + dj / c);
        let delta = c * den;
        g *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    1.0 / g
}

/// P(|T| >= |t|) for Student's t with `df` degrees of freedom: R's
/// `2 * pt(abs(t), df, lower.tail = FALSE)`.
pub fn t_two_sided(t: f64, df: f64) -> f64 {
    if t.is_nan() || !(df > 0.0) {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    let t2 = t * t;
    incomplete_beta(df * 0.5, 0.5, df / (df + t2), t2 / (df + t2))
}

/// P(F >= f) for Fisher's F with `df1` and `df2` degrees of freedom: R's
/// `pf(f, df1, df2, lower.tail = FALSE)`.
pub fn f_upper(f: f64, df1: f64, df2: f64) -> f64 {
    if f.is_nan() || !(df1 > 0.0) || !(df2 > 0.0) {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if f.is_infinite() {
        return 0.0;
    }
    let s = df2 + df1 * f;
    incomplete_beta(df2 * 0.5, df1 * 0.5, df2 / s, df1 * f / s)
}

/// P(Q > q) for Q = sum of lambda_j X_j, X_j independent chi-squares of one
/// degree of freedom, the eigenvalues given with their multiplicities
/// (`(lambda, m)`): Imhof's (1961) formula, as CompQuadForm's `imhof` (h = 1,
/// delta = 0),
///
///   1/2 + 1/pi int_0^inf sin(theta(u)) / (u rho(u)) du,
///   theta(u) = 1/2 sum m atan(lambda u) - q u / 2,
///   rho(u) = prod (1 + lambda^2 u^2)^(m/4),
///
/// integrated by adaptive Gauss-Kronrod over pieces of half a period of the
/// oscillation sin(q u / 2) the integrand tends to. There the pieces
/// alternate in sign and shrink: the integral stops when the rest is below
/// Imhof's bound 1 / (k u^k prod |lambda|^(m/2)) (k = sum m / 2), or when two
/// pieces in a row are below 1e-11, as the mean of the last two partial sums
/// (the limit of an alternating series lies between them).
pub fn imhof_upper(q: f64, lambdas: &[(f64, f64)]) -> f64 {
    let lambdas: Vec<(f64, f64)> = lambdas.iter().copied().filter(|&(l, m)| l != 0.0 && m > 0.0).collect();
    if lambdas.is_empty() || q.is_nan() {
        return f64::NAN;
    }
    if q <= 0.0 && lambdas.iter().all(|&(l, _)| l > 0.0) {
        return 1.0;
    }
    let theta = |u: f64| 0.5 * lambdas.iter().map(|&(l, m)| m * (l * u).atan()).sum::<f64>() - 0.5 * q * u;
    let ln_rho = |u: f64| {
        lambdas
            .iter()
            .map(|&(l, m)| 0.25 * m * (l * l * u * u).ln_1p())
            .sum::<f64>()
    };
    let f = |u: f64| {
        if u == 0.0 {
            return 0.5 * (lambdas.iter().map(|&(l, m)| m * l).sum::<f64>() - q);
        }
        theta(u).sin() / (u * ln_rho(u).exp())
    };
    let k: f64 = lambdas.iter().map(|&(_, m)| 0.5 * m).sum();
    let ln_prod: f64 = lambdas.iter().map(|&(l, m)| 0.5 * m * l.abs().ln()).sum();
    let tail = |u: f64| (-(k * u.ln() + ln_prod) - k.ln()).exp();
    let smallest = lambdas.iter().map(|&(l, _)| l.abs()).fold(f64::INFINITY, f64::min);
    let largest = lambdas.iter().map(|&(l, _)| l.abs()).fold(0.0, f64::max);
    let h = if q.abs() > 0.0 {
        2.0 * std::f64::consts::PI / q.abs()
    } else {
        1.0 / largest
    };
    let (mut total, mut previous, mut small) = (0.0, 0.0, 0);
    for i in 0..2_000_000u32 {
        let u = f64::from(i) * h;
        let piece = gauss_kronrod(&f, u, u + h, 1e-15 * h, 10);
        previous = total;
        total += piece;
        if tail(u + h) < 1e-13 {
            return 0.5 + total / std::f64::consts::PI;
        }
        small = if piece.abs() < 1e-11 && u > 10.0 / smallest {
            small + 1
        } else {
            0
        };
        if small == 2 {
            break;
        }
    }
    0.5 + 0.5 * (total + previous) / std::f64::consts::PI
}

/// The integral of `f` on [a, b] by 15-point Gauss-Kronrod, halving the
/// interval while the 7-point Gauss rule differs by more than `tol`.
fn gauss_kronrod<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64, depth: u32) -> f64 {
    const XK: [f64; 8] = [
        0.991_455_371_120_812_6,
        0.949_107_912_342_758_5,
        0.864_864_423_359_769_1,
        0.741_531_185_599_394_4,
        0.586_087_235_467_691_1,
        0.405_845_151_377_397_2,
        0.207_784_955_007_898_5,
        0.0,
    ];
    const WK: [f64; 8] = [
        0.022_935_322_010_529_22,
        0.063_092_092_629_978_55,
        0.104_790_010_322_250_2,
        0.140_653_259_715_525_9,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.204_432_940_075_298_9,
        0.209_482_141_084_727_8,
    ];
    // the Gauss weights of the odd Kronrod nodes (1, 3, 5) and of the centre
    const WG: [f64; 4] = [
        0.129_484_966_168_869_7,
        0.279_705_391_489_276_7,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
    ];
    let (c, r) = (0.5 * (a + b), 0.5 * (b - a));
    let fc = f(c);
    let (mut k, mut g) = (WK[7] * fc, WG[3] * fc);
    for i in 0..7 {
        let s = f(c - r * XK[i]) + f(c + r * XK[i]);
        k += WK[i] * s;
        if i % 2 == 1 {
            g += WG[i / 2] * s;
        }
    }
    let (k, g) = (k * r, g * r);
    if (k - g).abs() <= tol.max(1e-10 * k.abs()) || depth == 0 {
        k
    } else {
        gauss_kronrod(f, a, c, tol / 2.0, depth - 1) + gauss_kronrod(f, c, b, tol / 2.0, depth - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn imhof_of_a_chi_square() {
        // four eigenvalues 1: a chi-square of 4 degrees of freedom, P(X > q) = exp(-q/2) (1 + q/2)
        for q in [0.5, 5.0, 20.0] {
            let want = (-q / 2.0f64).exp() * (1.0 + q / 2.0);
            assert!(close(imhof_upper(q, &[(1.0, 4.0)]), want, 1e-10), "{q}");
            assert!(close(imhof_upper(q, &[(1.0, 1.0), (1.0, 3.0)]), want, 1e-10));
        }
        // 2 X, X of 2 degrees of freedom: P(2X > q) = exp(-q/4)
        assert!(close(imhof_upper(3.0, &[(2.0, 2.0)]), (-0.75f64).exp(), 1e-10));
    }

    #[test]
    fn student_t_as_r() {
        // R: 2 * pt(c(0.5, 2.1, 7.3, 25), df, lower.tail = FALSE)
        assert!(close(t_two_sided(0.5, 3.0), 0.651_447_964_848_151, 1e-13));
        assert!(close(t_two_sided(-2.1, 12.0), 0.057_544_938_734_951_18, 1e-13));
        assert!(close(t_two_sided(7.3, 40.0), 7.191_696_977_890_648e-9, 1e-11));
        assert!(close(t_two_sided(25.0, 148.0), 5.4403490958740904e-55, 1e-10));
        assert_eq!(t_two_sided(0.0, 5.0), 1.0);
    }

    #[test]
    fn fisher_f_as_r() {
        // R: pf(c(0.7, 4.2, 30), df1, df2, lower.tail = FALSE)
        assert!(close(f_upper(0.7, 2.0, 30.0), 0.504_515_019_394_244_5, 1e-13));
        // large degrees of freedom: lgamma's rounding at 750 (~4200) is 1e-12 of the result
        assert!(close(f_upper(4.2, 3.0, 1497.0), 0.005_714_199_802_101_863, 1e-11));
        assert!(close(f_upper(30.0, 1.0, 10.0), 0.0002702957472546176, 1e-12));
    }
}
