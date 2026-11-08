//! FORCE's Quality Assurance Information (QAI) bit field, the part the atmospheric correction's gas functions read and
//! write. Ported from FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force), `src/modules/cross-level/quality-cl.c` (`get_qai`, `set_qai`, `get_off`,
//! `get_shadow`, `get_water`, `set_vaporfill`) and `enum-cl.h` (the bit positions). One QAI value is an `i16`.
//!
//! Only the flags the ported stages test are here (`gas-ll.c`: off, shadow, water, vapour fill; `aod-ll.c`: also
//! cloud; `sunview-ll.c`: low sun; `atmo-ll.c`: aerosol and, in the surface reflectance, subzero and saturation; the
//! cloud distance of `cloud-ll.c` reads snow).

/// Bit of the off/on flag (1 = no data, outside the scene).
pub const BIT_OFF: u32 = 0;
/// Bit of the cloud shadow flag.
pub const BIT_SHD: u32 = 3;
/// Bit of the snow flag.
pub const BIT_SNW: u32 = 4;
/// Bit of the subzero reflectance flag.
pub const BIT_SUB: u32 = 8;
/// Bit of the saturated reflectance flag.
pub const BIT_SAT: u32 = 9;
/// Bit of the water flag.
pub const BIT_WTR: u32 = 5;
/// Bit of the aerosol flag (2 bits: 0 = good, 1 = interpolated, 2 = high AOD, 3 = fill value of the scene).
pub const BIT_AOD: u32 = 6;
/// Bit of the low sun angle flag.
pub const BIT_SUN: u32 = 10;
/// Bit of the illumination flag (2 bits: 0 = good, 1 = moderate, 2 = poor, 3 = deep shadow).
pub const BIT_ILL: u32 = 11;
/// Bit of the slope flag (the slope is above 2 degrees).
pub const BIT_SLP: u32 = 13;
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

/// `get_snow`: the pixel is snow.
#[inline]
pub fn snow(v: i16) -> bool {
    bit(v, BIT_SNW)
}

/// `get_subzero`: the surface reflectance of the pixel was below 0 in a band.
#[inline]
pub fn subzero(v: i16) -> bool {
    bit(v, BIT_SUB)
}

/// `get_saturation`: the surface reflectance of the pixel was above 1 in a band.
#[inline]
pub fn saturation(v: i16) -> bool {
    bit(v, BIT_SAT)
}

/// `get_water`: the pixel is water.
#[inline]
pub fn water(v: i16) -> bool {
    bit(v, BIT_WTR)
}

/// `get_aerosol`: the two aerosol bits of a QAI value, 0 to 3.
#[inline]
pub fn aerosol(v: i16) -> i8 {
    ((v >> BIT_AOD) & 3) as i8
}

/// `get_lowsun`: the sun is low (the zenith is above 75 degrees).
#[inline]
pub fn lowsun(v: i16) -> bool {
    bit(v, BIT_SUN)
}

/// `get_vaporfill`: the water vapour of the pixel was filled with the scene average.
#[inline]
pub fn vaporfill(v: i16) -> bool {
    bit(v, BIT_WVP)
}

/// `set_off(qai, p, val)`: clear the off flag, then set it to `val` (`set_qai`).
#[inline]
pub fn set_off(v: &mut i16, val: bool) {
    *v &= !(1i16 << BIT_OFF);
    *v |= (val as i16) << BIT_OFF;
}

/// `set_aerosol(qai, p, val)`: clear the two aerosol bits, then set them to `val` (0 to 3).
#[inline]
pub fn set_aerosol(v: &mut i16, val: i16) {
    *v &= !(3i16 << BIT_AOD);
    *v |= val << BIT_AOD;
}

/// `set_lowsun(qai, p, val)`: clear the low sun flag, then set it to `val`.
#[inline]
pub fn set_lowsun(v: &mut i16, val: bool) {
    *v &= !(1i16 << BIT_SUN);
    *v |= (val as i16) << BIT_SUN;
}

/// `set_subzero(qai, p, val)`: clear the subzero flag, then set it to `val`.
#[inline]
pub fn set_subzero(v: &mut i16, val: bool) {
    *v &= !(1i16 << BIT_SUB);
    *v |= (val as i16) << BIT_SUB;
}

/// `set_saturation(qai, p, val)`: clear the saturation flag, then set it to `val`.
#[inline]
pub fn set_saturation(v: &mut i16, val: bool) {
    *v &= !(1i16 << BIT_SAT);
    *v |= (val as i16) << BIT_SAT;
}

/// `set_slope(qai, p, val)`: clear the slope flag, then set it to `val`.
#[inline]
pub fn set_slope(v: &mut i16, val: bool) {
    *v &= !(1i16 << BIT_SLP);
    *v |= (val as i16) << BIT_SLP;
}

/// `set_illumination(qai, p, val)`: clear the two illumination bits, then set them to `val` (0 to 3).
#[inline]
pub fn set_illumination(v: &mut i16, val: i16) {
    *v &= !(3i16 << BIT_ILL);
    *v |= val << BIT_ILL;
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
        let mut a = 0b101i16 | (1 << 3);
        set_aerosol(&mut a, 2);
        set_lowsun(&mut a, true);
        assert_eq!((aerosol(a), lowsun(a), a & 0b1111), (2, true, 0b1101));
        set_aerosol(&mut a, 3);
        set_aerosol(&mut a, 1);
        set_lowsun(&mut a, false);
        assert_eq!((aerosol(a), lowsun(a)), (1, false));
        let mut s = 0b1i16 | (1 << 4);
        assert!(snow(s) && !subzero(s) && !saturation(s));
        set_subzero(&mut s, true);
        set_saturation(&mut s, true);
        assert_eq!((s, subzero(s), saturation(s)), (1 | (1 << 4) | (1 << 8) | (1 << 9), true, true));
        set_subzero(&mut s, false);
        assert!(!subzero(s) && saturation(s) && snow(s));
    }
}
