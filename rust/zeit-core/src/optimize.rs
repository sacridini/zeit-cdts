//! One-dimensional minimization by Brent's fmin, as R's `optimize` (R's
//! src/library/stats/src/optimize.c), which remotePARTS's AR fits go through
//! (`optim(method = "Brent")` is `optimize`), and as remotePARTS's own copy for
//! the nugget of its GLS (src/optimize_nugget.cpp, from netlib's fmin.f). The
//! same steps in the same order take their paths to the minimum.

/// The minimum of `f` on [`lower`, `upper`] to within `tol` (R's
/// `optimize(f, lower = lower, upper = upper, tol = tol)$minimum`). Like R, a
/// non-finite value of `f` counts as the largest double.
pub fn fmin<F: FnMut(f64) -> f64>(lower: f64, upper: f64, mut f: F, tol: f64) -> f64 {
    let f = |x: f64| {
        let v = f(x);
        if v.is_finite() {
            v
        } else {
            f64::MAX
        }
    };
    brent(lower, upper, f, tol, false)
}

/// remotePARTS's copy of fmin (optimize_nugget_cpp's loop): its relative
/// precision is the square root of 2^-53 (it halves 1 until 1 + eps is 1),
/// its steps of tol1 take the sign of xm - x and of d (copysign: +0 is
/// positive), and the values of `f` are compared as they come.
pub fn fmin_netlib<F: FnMut(f64) -> f64>(lower: f64, upper: f64, f: F, tol: f64) -> f64 {
    brent(lower, upper, f, tol, true)
}

fn brent<F: FnMut(f64) -> f64>(lower: f64, upper: f64, mut f: F, tol: f64, netlib: bool) -> f64 {
    // the squared inverse of the golden ratio
    let c = (3.0 - 5f64.sqrt()) * 0.5;
    let eps = if netlib {
        (f64::EPSILON / 2.0).sqrt()
    } else {
        f64::EPSILON.sqrt()
    };

    let (mut a, mut b) = (lower, upper);
    let mut v = a + c * (b - a);
    let mut w = v;
    let mut x = v;
    let mut d = 0.0f64;
    let mut e = 0.0f64;
    let mut fx = f(x);
    let mut fv = fx;
    let mut fw = fx;
    let tol3 = tol / 3.0;

    loop {
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + tol3;
        let t2 = tol1 * 2.0;
        // the stopping criterion
        if (x - xm).abs() <= t2 - (b - a) * 0.5 {
            break;
        }
        let mut p = 0.0;
        let mut q = 0.0;
        let mut r = 0.0;
        if e.abs() > tol1 {
            // fit a parabola
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            // a golden-section step
            e = if x < xm { b - x } else { a - x };
            d = c * e;
        } else {
            // a parabolic-interpolation step
            d = p / q;
            let u = x + d;
            // f must not be evaluated too close to the bounds
            if u - a < t2 || b - u < t2 {
                d = if netlib {
                    tol1.copysign(xm - x)
                } else if x >= xm {
                    -tol1
                } else {
                    tol1
                };
            }
        }
        // f must not be evaluated too close to x
        let u = if d.abs() >= tol1 {
            x + d
        } else if netlib {
            x + tol1.copysign(d)
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_of_a_parabola_and_r_s_answers() {
        let x = fmin(-1.0, 1.0, |x| (x - 0.3) * (x - 0.3), 1e-10);
        assert!((x - 0.3).abs() < 1e-8);
        // R: optimize(function(x) cos(x) + x^2 / 10, c(0, 6), tol = 1e-8)$minimum
        let x = fmin(0.0, 6.0, |x: f64| x.cos() + x * x / 10.0, 1e-8);
        assert!((x - 2.5957390804504716).abs() < 1e-12, "{x}");
    }

    #[test]
    fn non_finite_values_are_the_largest() {
        // log goes to -inf at 0 and NaN below: the minimum of -log(x) - log(1 - x) inside
        let x = fmin(-1.0, 1.0, |x: f64| -(x.ln()) - (1.0 - x).ln(), 1e-9);
        assert!((x - 0.5).abs() < 1e-6, "{x}");
    }
}
