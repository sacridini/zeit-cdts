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

#[cfg(test)]
mod tests {
    #[test]
    fn erfc_values() {
        assert_eq!(super::erfc(0.0), 1.0);
        assert!((super::erfc(1.0) - 0.157_299_207_050_285_13).abs() < 1e-16);
        assert_eq!(super::erfc(f64::INFINITY), 0.0);
    }

    #[test]
    fn lgamma_values() {
        assert_eq!(super::lgamma(1.0), 0.0);
        assert!((super::lgamma(10.0) - 362880f64.ln()).abs() < 1e-12);
    }
}
