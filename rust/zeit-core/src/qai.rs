//! FORCE's Quality Assurance Information (QAI) bit field, the part the atmospheric correction's gas functions read and
//! write. Ported from FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force), `src/modules/cross-level/quality-cl.c` (`get_qai`, `set_qai`, `get_off`,
//! `get_shadow`, `get_water`, `set_vaporfill`) and `enum-cl.h` (the bit positions). One QAI value is an `i16`.
//!
//! Only the flags the ported stages test are here (`gas-ll.c`: off, shadow, water, vapour fill; `aod-ll.c`: also
//! cloud); the rest of the field (snow, aerosol, subzero, saturation, sun, illumination, slope) comes with the stages
//! that use them.

/// Bit of the off/on flag (1 = no data, outside the scene).
pub const BIT_OFF: u32 = 0;
/// Bit of the cloud shadow flag.
pub const BIT_SHD: u32 = 3;
/// Bit of the water flag.
pub const BIT_WTR: u32 = 5;
/// Bit of the water vapour fill flag (the water vapour of the pixel is the scene average).
pub const BIT_WVP: u32 = 14;

/// Bit of the first cloud flag bit (2 bits: 0 = none, 1 = opaque, 2 = unsure, 3 = cirrus).
pub const BIT_CLD: u32 = 1;

/// `get_qai(qai, index, p, 1)`: one flag bit of a QAI value.
#[inline]
fn bit(v: i16, index: u32) -> bool {
    (v >> index) & 1 != 0
}

/// `get_off`: the pixel is off (no data).
#[inline]
pub fn off(v: i16) -> bool {
    bit(v, BIT_OFF)
}

/// `get_cloud`: the two cloud bits of a QAI value, 0 (no cloud) to 3 (`get_qai(qai, _QAI_BIT_CLD_, p, 2)`).
#[inline]
pub fn cloud(v: i16) -> i8 {
    ((v >> BIT_CLD) & 3) as i8
}

/// `get_shadow`: the pixel is cloud shadow.
#[inline]
pub fn shadow(v: i16) -> bool {
    bit(v, BIT_SHD)
}

/// `get_water`: the pixel is water.
#[inline]
pub fn water(v: i16) -> bool {
    bit(v, BIT_WTR)
}

/// `get_vaporfill`: the water vapour of the pixel was filled with the scene average.
#[inline]
pub fn vaporfill(v: i16) -> bool {
    bit(v, BIT_WVP)
}

/// `set_vaporfill(qai, p, val)`: clear the water vapour fill bit, then set it to `val` (`set_qai`).
#[inline]
pub fn set_vaporfill(v: &mut i16, val: bool) {
    *v &= !(1i16 << BIT_WVP);
    *v |= (val as i16) << BIT_WVP;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_and_the_fill_flag() {
        assert!(off(1) && !off(2));
        assert!(shadow(1 << 3) && !shadow(1 << 2));
        assert_eq!((cloud(0), cloud(1 << 1), cloud(2 << 1), cloud(3 << 1), cloud(1 << 3)), (0, 1, 2, 3, 0));
        assert!(water(1 << 5) && !water(1 << 4));
        let mut v = 0b101i16;
        assert!(!vaporfill(v));
        set_vaporfill(&mut v, true);
        assert!(vaporfill(v) && v == 0b101 | (1 << 14));
        set_vaporfill(&mut v, false);
        assert_eq!(v, 0b101);
    }
}
