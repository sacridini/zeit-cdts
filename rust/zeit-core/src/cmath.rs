//! Functions of the C runtime's math library that Rust's standard library does
//! not have. They come from the same library the C++ core calls (the UCRT on
//! Windows, glibc's libm on Linux, libSystem on macOS), so both cores get the
//! same bits; a Rust implementation (the `libm` crate) would round differently.
//!
//! `f64::exp`, `ln`, `sqrt`, `atan` and `powf` already call that same library.

mod sys {
    extern "C" {
        pub fn erfc(x: f64) -> f64;
        #[cfg(not(unix))]
        pub fn lgamma(x: f64) -> f64;
        #[cfg(unix)]
        pub fn lgamma_r(x: f64, sign: *mut i32) -> f64;
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        pub fn sincos(x: f64, sin: *mut f64, cos: *mut f64);
    }
}

/// The complementary error function, `std::erfc`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    // SAFETY: a pure function of a double, present in every C runtime Rust links.
    unsafe { sys::erfc(x) }
}

/// The log of the absolute value of the gamma function, `std::lgamma`. On
/// Unix through lgamma_r, the same computation (glibc's lgamma is lgamma_r
/// into the global signgam), without that global's race between threads.
#[inline]
pub fn lgamma(x: f64) -> f64 {
    #[cfg(unix)]
    {
        let mut sign = 0;
        // SAFETY: a pure function of a double writing the sign into a local.
        unsafe { sys::lgamma_r(x, &mut sign) }
    }
    #[cfg(not(unix))]
    {
        // SAFETY: a pure function of a double, present in every C runtime Rust links.
        unsafe { sys::lgamma(x) }
    }
}

/// `(sin x, cos x)` where the C++ core takes a sine and a cosine of the same
/// argument in one function: GCC makes the two calls one call to glibc's
/// sincos, which differs from sin and cos by an ulp for some arguments in older
/// glibc (2.28, the manylinux_2_28 image). Elsewhere (MSVC; Apple's
/// __sincos_stret agrees with sin and cos) the two calls.
#[inline]
pub fn sin_cos(x: f64) -> (f64, f64) {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        let (mut s, mut c) = (0.0, 0.0);
        // SAFETY: a pure function of a double writing into two locals.
        unsafe { sys::sincos(x, &mut s, &mut c) };
        (s, c)
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        (x.sin(), x.cos())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn erfc_values() {
        assert_eq!(super::erfc(0.0), 1.0);
        assert!((super::erfc(1.0) - 0.157_299_207_050_285_13).abs() < 1e-16);
        assert_eq!(super::erfc(f64::INFINITY), 0.0);
    }

    #[test]
    fn sin_cos_values() {
        for x in [0.0, 0.5, -2.0, 12566.0, 1e6] {
            let (s, c) = super::sin_cos(x);
            assert!((s - x.sin()).abs() <= 1e-15 * x.abs().max(1.0));
            assert!((c - x.cos()).abs() <= 1e-15 * x.abs().max(1.0));
        }
    }

    #[test]
    fn lgamma_values() {
        assert_eq!(super::lgamma(1.0), 0.0);
        assert!((super::lgamma(10.0) - 362880f64.ln()).abs() < 1e-12);
    }
}
