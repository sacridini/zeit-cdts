//! Functions of the C runtime's math library that Rust's standard library does
//! not have. They come from the same library the C++ core calls (the UCRT on
//! Windows, glibc's libm on Linux, libSystem on macOS), so both cores get the
//! same bits; a Rust implementation (the `libm` crate) would round differently.
//!
//! `f64::exp`, `ln`, `sqrt` and `powf` already call that same library.

mod sys {
    extern "C" {
        pub fn erfc(x: f64) -> f64;
    }
}

/// The complementary error function, `std::erfc`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    // SAFETY: a pure function of a double, present in every C runtime Rust links.
    unsafe { sys::erfc(x) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn erfc_values() {
        assert_eq!(super::erfc(0.0), 1.0);
        assert!((super::erfc(1.0) - 0.157_299_207_050_285_13).abs() < 1e-16);
        assert_eq!(super::erfc(f64::INFINITY), 0.0);
    }
}
