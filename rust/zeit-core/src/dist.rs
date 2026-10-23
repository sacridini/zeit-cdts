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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
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
