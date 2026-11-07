//! Brent's one-dimensional minimizer `gsl_min_fminimizer_brent` of GSL 2.7 (`min/brent.c`, `min/fsolver.c` for the
//! `gsl_min_fminimizer` around it, `min/convergence.c` for `gsl_min_test_interval`; Copyright (C) 1996-2000, 2007
//! Brian Gough, GPL-3.0-or-later), which FORCE's `aod_elev_dependency` (`aod-ll.c`) uses to find the scale height of
//! the aerosols.
//!
//! Ported operation by operation, in doubles, in the same order: the iterates equal GSL's bit for bit (no fused
//! multiply-add, the same literals: the golden ratio is GSL's `0.3819660`, not `(3 - sqrt(5)) / 2`). Where the C
//! would call GSL's error handler (a function value that is not finite, an interval that does not enclose a minimum)
//! this port returns an [`Err`], which is what the status code is when the handler is turned off.

/// `GSL_SQRT_DBL_EPSILON`.
const SQRT_DBL_EPSILON: f64 = 1.490_116_119_384_765_6e-8;
/// The golden section of GSL's brent.c, `0.3819660` (a literal, not the exact `(3 - sqrt(5)) / 2`).
const GOLDEN: f64 = 0.3819660;

/// What `gsl_min_fminimizer_set` and `_iterate` report instead of `GSL_SUCCESS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrentError {
    /// `GSL_EINVAL`: the interval is invalid, the start is not inside it, or the end points do not enclose a minimum.
    Inval,
    /// `GSL_EBADFUNC`: the function returned a value that is not finite.
    BadFunc,
}

/// The state of GSL's `gsl_min_fminimizer` of type `brent`.
#[derive(Clone, Debug)]
pub struct Brent {
    /// The current estimate of the minimum (`gsl_min_fminimizer_x_minimum`).
    pub x_minimum: f64,
    /// Its function value.
    pub f_minimum: f64,
    /// The lower end of the bracketing interval (`x_lower`).
    pub x_lower: f64,
    pub f_lower: f64,
    /// The upper end.
    pub x_upper: f64,
    pub f_upper: f64,
    d: f64,
    e: f64,
    v: f64,
    w: f64,
    f_v: f64,
    f_w: f64,
}

/// `SAFE_FUNC_CALL`: the function value, `Err` if it is not finite.
#[inline]
fn safe_call(f: &mut impl FnMut(f64) -> f64, x: f64) -> Result<f64, BrentError> {
    let y = f(x);
    if y.is_finite() {
        Ok(y)
    } else {
        Err(BrentError::BadFunc)
    }
}

impl Brent {
    /// `gsl_min_fminimizer_set`: the function at the three points (lower, upper, then the start, in this order), then
    /// `gsl_min_fminimizer_set_with_values` and `brent_init`.
    pub fn set(f: &mut impl FnMut(f64) -> f64, x_minimum: f64, x_lower: f64, x_upper: f64) -> Result<Self, BrentError> {
        let f_lower = safe_call(f, x_lower)?;
        let f_upper = safe_call(f, x_upper)?;
        let f_minimum = safe_call(f, x_minimum)?;
        Self::set_with_values(f, x_minimum, f_minimum, x_lower, f_lower, x_upper, f_upper)
    }

    /// `gsl_min_fminimizer_set_with_values`.
    pub fn set_with_values(
        f: &mut impl FnMut(f64) -> f64,
        x_minimum: f64,
        f_minimum: f64,
        x_lower: f64,
        f_lower: f64,
        x_upper: f64,
        f_upper: f64,
    ) -> Result<Self, BrentError> {
        if x_lower > x_upper {
            return Err(BrentError::Inval);
        }
        if x_minimum >= x_upper || x_minimum <= x_lower {
            return Err(BrentError::Inval);
        }
        if f_minimum >= f_lower || f_minimum >= f_upper {
            return Err(BrentError::Inval);
        }
        // brent_init
        let v = x_lower + GOLDEN * (x_upper - x_lower);
        let w = v;
        let f_vw = safe_call(f, v)?;
        Ok(Brent {
            x_minimum,
            f_minimum,
            x_lower,
            f_lower,
            x_upper,
            f_upper,
            d: 0.0,
            e: 0.0,
            v,
            w,
            f_v: f_vw,
            f_w: f_vw,
        })
    }

    /// `gsl_min_fminimizer_iterate` (`brent_iterate`): one step.
    pub fn iterate(&mut self, f: &mut impl FnMut(f64) -> f64) -> Result<(), BrentError> {
        let x_left = self.x_lower;
        let x_right = self.x_upper;

        let z = self.x_minimum;
        let mut d = self.e;
        let mut e = self.d;
        let v = self.v;
        let w = self.w;
        let f_v = self.f_v;
        let f_w = self.f_w;
        let f_z = self.f_minimum;

        let w_lower = z - x_left;
        let w_upper = x_right - z;

        let tolerance = SQRT_DBL_EPSILON * z.abs();

        let mut p = 0.0f64;
        let mut q = 0.0f64;
        let mut r = 0.0f64;

        let midpoint = 0.5 * (x_left + x_right);

        if e.abs() > tolerance {
            // fit parabola
            r = (z - w) * (f_z - f_v);
            q = (z - v) * (f_z - f_w);
            p = (z - v) * q - (z - w) * r;
            q = 2.0 * (q - r);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            r = e;
            e = d;
        }

        if p.abs() < (0.5 * q * r).abs() && p < q * w_lower && p < q * w_upper {
            let t2 = 2.0 * tolerance;

            d = p / q;
            let u1 = z + d;

            if (u1 - x_left) < t2 || (x_right - u1) < t2 {
                d = if z < midpoint { tolerance } else { -tolerance };
            }
        } else {
            e = if z < midpoint { x_right - z } else { -(z - x_left) };
            d = GOLDEN * e;
        }

        let u = if d.abs() >= tolerance { z + d } else { z + if d > 0.0 { tolerance } else { -tolerance } };

        self.e = e;
        self.d = d;

        let f_u = safe_call(f, u)?;

        if f_u <= f_z {
            if u < z {
                self.x_upper = z;
                self.f_upper = f_z;
            } else {
                self.x_lower = z;
                self.f_lower = f_z;
            }

            self.v = w;
            self.f_v = f_w;
            self.w = z;
            self.f_w = f_z;
            self.x_minimum = u;
            self.f_minimum = f_u;
            return Ok(());
        }
        if u < z {
            self.x_lower = u;
            self.f_lower = f_u;
        } else {
            self.x_upper = u;
            self.f_upper = f_u;
        }

        if f_u <= f_w || w == z {
            self.v = w;
            self.f_v = f_w;
            self.w = u;
            self.f_w = f_u;
        } else if f_u <= f_v || v == z || v == w {
            self.v = u;
            self.f_v = f_u;
        }
        Ok(())
    }
}

/// The result of `gsl_min_test_interval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalTest {
    /// `GSL_SUCCESS`: the interval is smaller than the tolerance.
    Success,
    /// `GSL_CONTINUE`.
    Continue,
}

/// `gsl_min_test_interval`: `Err` for a negative tolerance or `lower > upper` (`GSL_EBADTOL`, `GSL_EINVAL`).
pub fn test_interval(x_lower: f64, x_upper: f64, epsabs: f64, epsrel: f64) -> Result<IntervalTest, BrentError> {
    let (lower, upper) = (x_lower, x_upper);
    let (abs_lower, abs_upper) = (lower.abs(), upper.abs());
    if epsrel < 0.0 || epsabs < 0.0 || lower > upper {
        return Err(BrentError::Inval);
    }
    let min_abs = if (lower > 0.0 && upper > 0.0) || (lower < 0.0 && upper < 0.0) {
        // GSL_MIN_DBL(a, b) = (a < b ? a : b)
        if abs_lower < abs_upper {
            abs_lower
        } else {
            abs_upper
        }
    } else {
        0.0
    };
    let tolerance = epsabs + epsrel * min_abs;
    if (upper - lower).abs() < tolerance {
        Ok(IntervalTest::Success)
    } else {
        Ok(IntervalTest::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(f: &mut impl FnMut(f64) -> f64, x0: f64, lo: f64, hi: f64, max: usize) -> (Brent, usize) {
        let mut s = Brent::set(f, x0, lo, hi).unwrap();
        let mut it = 0;
        while it < max {
            it += 1;
            s.iterate(f).unwrap();
            if test_interval(s.x_lower, s.x_upper, 0.001, 0.0).unwrap() == IntervalTest::Success {
                break;
            }
        }
        (s, it)
    }

    #[test]
    fn a_parabola_is_found_in_a_few_steps() {
        let mut f = |x: f64| (x - 2.5) * (x - 2.5) + 1.0;
        let (s, it) = run(&mut f, 1.0, 0.0, 10.0, 100);
        assert!((s.x_minimum - 2.5).abs() < 1e-3, "{}", s.x_minimum);
        assert!(it <= 8, "{it}");
    }

    /// The iterates of GSL 2.7's `gsl_min_fminimizer_brent` (a C program against the real library): `(x_minimum,
    /// x_lower, x_upper)` after each iteration, as the bits of the doubles.
    fn check(f: &mut impl FnMut(f64) -> f64, x0: f64, lo: f64, hi: f64, want: &[(u64, u64, u64)], exact: bool) {
        let mut s = Brent::set(f, x0, lo, hi).unwrap();
        for (k, &(m, l, u)) in want.iter().enumerate() {
            s.iterate(f).unwrap();
            let got = [s.x_minimum, s.x_lower, s.x_upper];
            for (g, w) in got.iter().zip([m, l, u]) {
                let w = f64::from_bits(w);
                if exact {
                    assert_eq!(g.to_bits(), w.to_bits(), "iteration {}: {g} vs {w}", k + 1);
                } else {
                    // (cos of the C library may differ from this platform's in the last bit)
                    assert!((g - w).abs() < 1e-12, "iteration {}: {g} vs {w}", k + 1);
                }
            }
        }
    }

    #[test]
    fn the_iterates_of_gsl_for_a_kinked_polynomial_bit_for_bit() {
        // f(x) = (x-3)^2 (x+1)^2 / 100 + |x-2| on [-2, 6], start 1: only + - * and fabs, so the same on every platform
        let mut f = |x: f64| (x - 3.0) * (x - 3.0) * (x + 1.0) * (x + 1.0) * 0.01 + (x - 2.0).abs();
        let want: [(u64, u64, u64); 14] = [
        (0x40074754f3775b82, 0x3ff0000000000000, 0x4018000000000000),
        (0x4000bec4601c8b4c, 0x3ff0000000000000, 0x40074754f3775b82),
        (0x4000308fd5e9d495, 0x3ff0000000000000, 0x4000bec4601c8b4c),
        (0x4000308fd5e9d495, 0x3ffa1f7e36b848c7, 0x4000bec4601c8b4c),
        (0x4000308fd5e9d495, 0x3fff3365ffb2ab3c, 0x4000bec4601c8b4c),
        (0x4000189cc973a538, 0x3fff3365ffb2ab3c, 0x4000308fd5e9d495),
        (0x4000189cc973a538, 0x3fffd04595e29b15, 0x4000308fd5e9d495),
        (0x4000043906e1469a, 0x3fffd04595e29b15, 0x4000189cc973a538),
        (0x400001ddb5dda9df, 0x3fffd04595e29b15, 0x4000043906e1469a),
        (0x400001ddb5dda9df, 0x3ffff01376b100ef, 0x4000043906e1469a),
        (0x3fffff7890b7c687, 0x3ffff01376b100ef, 0x400001ddb5dda9df),
        (0x3fffff7890b7c687, 0x3ffff99732ce9f39, 0x400001ddb5dda9df),
        (0x3fffffc5de91ce9b, 0x3fffff7890b7c687, 0x400001ddb5dda9df),
        (0x3fffffc5de91ce9b, 0x3fffff7890b7c687, 0x400000a4817bd91b),
        ];
        check(&mut f, 1.0, -2.0, 6.0, &want, true);
    }

    #[test]
    fn the_iterates_of_gsl_for_the_cosine() {
        // the example of GSL's manual: the minimum of cos(x) on [0, 6] from 2, six iterations to the interval 0.001
        let mut f = |x: f64| x.cos();
        let want: [(u64, u64, u64); 6] = [
        (0x400c3910c2c5e2ce, 0x4000000000000000, 0x4018000000000000),
        (0x40096608e7dad65e, 0x4000000000000000, 0x400c3910c2c5e2ce),
        (0x400902fc30f8d87e, 0x4000000000000000, 0x40096608e7dad65e),
        (0x400921bd4a1189b7, 0x400902fc30f8d87e, 0x40096608e7dad65e),
        (0x400921fb875f7c3c, 0x400921bd4a1189b7, 0x40096608e7dad65e),
        (0x400921fb546b2b27, 0x400921bd4a1189b7, 0x400921fb875f7c3c),
        ];
        check(&mut f, 2.0, 0.0, 6.0, &want, false);
    }

    #[test]
    fn an_interval_without_a_minimum_is_refused_like_gsl() {
        // GSL returned GSL_EINVAL (4) for this one: the end points are lower than the start
        let mut f = |x: f64| (x - 1.7).abs().powf(1.5) + 0.1 * x + 0.05 * (5.0 * x).sin();
        assert_eq!(Brent::set(&mut f, 4.0, 0.1, 10.0).unwrap_err(), BrentError::Inval);
    }

    #[test]
    fn bad_intervals_and_functions_are_refused() {
        let mut f = |x: f64| x * x;
        assert_eq!(Brent::set(&mut f, 1.0, 2.0, 3.0).unwrap_err(), BrentError::Inval); // start outside
        assert_eq!(Brent::set(&mut f, 1.0, 3.0, 2.0).unwrap_err(), BrentError::Inval); // lower > upper
        assert_eq!(Brent::set(&mut f, 3.5, 3.0, 4.0).unwrap_err(), BrentError::Inval); // no enclosed minimum
        let mut g = |x: f64| if x > 1.0 { f64::NAN } else { x };
        assert_eq!(Brent::set(&mut g, 0.5, 0.0, 2.0).unwrap_err(), BrentError::BadFunc);
    }

    #[test]
    fn the_interval_test() {
        assert_eq!(test_interval(1.0, 1.0005, 0.001, 0.0), Ok(IntervalTest::Success));
        assert_eq!(test_interval(1.0, 1.01, 0.001, 0.0), Ok(IntervalTest::Continue));
        assert_eq!(test_interval(1.0, 1.01, 0.0, 0.1), Ok(IntervalTest::Success)); // relative: 0.1 * 1
        assert_eq!(test_interval(-1.0, 1.0, 0.0, 0.1), Ok(IntervalTest::Continue)); // straddles zero: min_abs 0
        assert!(test_interval(2.0, 1.0, 0.1, 0.0).is_err());
    }
}
