//! NumPy arrays in and out, as pybind11's `array_t<double, c_style | forcecast>`
//! took them: any dtype converted to the one asked, any layout made C-contiguous
//! (without a copy when it already is).

use numpy::ndarray::{ArrayView, Dimension};
use std::borrow::Cow;

/// The values of `a` in C order, borrowed when `a` already is C-contiguous.
pub fn c_order<'a, T: Copy, D: Dimension>(a: &'a ArrayView<'a, T, D>) -> Cow<'a, [T]> {
    match a.as_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(a.iter().copied().collect()),
    }
}
