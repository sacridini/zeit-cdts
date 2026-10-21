//! The raster types of the warp kernel and the C++'s conversions between them
//! and double: `static_cast` as the C++ compilers emit it on the machine (on
//! x86-64 the truncating SSE2 conversion, whose out-of-range result is the
//! "integer indefinite" value; elsewhere a saturating one, as Rust's `as`),
//! ClampRoundAndAvoidNoData's rounding, AvoidNoData and
//! GDALWarpNoDataMasker's test.

/// `static_cast<int>(v)`.
#[inline(always)]
pub fn cvt_i32(v: f64) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_cvttsd_si32, _mm_set_sd};
        // SAFETY: SSE2 is part of x86-64.
        unsafe { _mm_cvttsd_si32(_mm_set_sd(v)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v as i32
    }
}

/// `static_cast<int64_t>(v)`.
#[inline(always)]
pub fn cvt_i64(v: f64) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_cvttsd_si64, _mm_set_sd};
        // SAFETY: SSE2 is part of x86-64.
        unsafe { _mm_cvttsd_si64(_mm_set_sd(v)) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v as i64
    }
}

/// `static_cast<uint64_t>(v)`: on x86-64, a signed conversion of v below 2^63
/// (NaN included), else of v - 2^63 with the top bit flipped, as GCC emits it;
/// MSVC's (/fpcvt:BC) gives 2^63 from 2^64 on and NaN.
#[inline(always)]
pub fn cvt_u64(v: f64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        const TWO63: f64 = 9_223_372_036_854_775_808.0;
        if !(v >= TWO63) {
            cvt_i64(v) as u64
        } else if cfg!(target_env = "msvc") && !(v < 2.0 * TWO63) {
            1u64 << 63
        } else {
            (cvt_i64(v - TWO63) as u64) ^ (1u64 << 63)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v as u64
    }
}

/// `std::min` and `std::max` of doubles: the first argument unless the second
/// compares strictly lower (higher), whatever NaNs and signed zeros there are.
#[inline(always)]
pub fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline(always)]
pub fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// floor and ceil of values in the range of int, as ints.
#[inline(always)]
pub fn ifloor(v: f64) -> i32 {
    let i = cvt_i32(v);
    if i as f64 > v {
        i.wrapping_sub(1)
    } else {
        i
    }
}

#[inline(always)]
pub fn iceil(v: f64) -> i32 {
    let i = cvt_i32(v);
    if (i as f64) < v {
        i.wrapping_add(1)
    } else {
        i
    }
}

/// A raster cell type: (u)int8/16/32/64, float32 or float64.
pub trait Raster: Copy + PartialEq + Default + Send + Sync + 'static {
    /// The 8 and 16-bit integers: NoData compared as an int, mode by histogram.
    const SMALL: bool;
    /// uint8 (GDAL's Byte).
    const BYTE: bool;
    const FLOAT32: bool;
    /// GWKNearest copies the cell; GWKRealCase (64-bit integers) goes through a double.
    const COPY: bool;
    /// The histogram of the mode: its bins and the bin of value 0.
    const BINS: usize;
    const BINS_OFFSET: i32;
    /// The lowest and highest values, as doubles.
    const LOWEST_F64: f64;
    const MAX_F64: f64;

    fn to_f64(self) -> f64;
    fn to_f32(self) -> f32;
    fn to_i32(self) -> i32;
    /// `static_cast<T>(v)`.
    fn from_f64(v: f64) -> Self;
    /// ClampRoundAndAvoidNoData's clamping and rounding.
    fn clamp_round(v: f64) -> Self;
    /// AvoidNoData: one step away from the destination NoData.
    fn avoid_nodata(self) -> Self;
    fn is_nan(self) -> bool;
}

macro_rules! integer {
    ($t:ty, small: $small:expr, copy: $copy:expr, bins: $bins:expr, offset: $offset:expr,
     from: $from:expr, round: $round:expr) => {
        impl Raster for $t {
            const SMALL: bool = $small;
            const BYTE: bool = $bins == 256 && $offset == 0;
            const FLOAT32: bool = false;
            const COPY: bool = $copy;
            const BINS: usize = $bins;
            const BINS_OFFSET: i32 = $offset;
            const LOWEST_F64: f64 = <$t>::MIN as f64;
            const MAX_F64: f64 = <$t>::MAX as f64;

            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
            #[inline(always)]
            fn to_f32(self) -> f32 {
                self as f32
            }
            #[inline(always)]
            fn to_i32(self) -> i32 {
                self as i32
            }
            #[inline(always)]
            fn from_f64(v: f64) -> $t {
                let from: fn(f64) -> $t = $from;
                from(v)
            }
            #[inline(always)]
            fn clamp_round(v: f64) -> $t {
                if v < Self::LOWEST_F64 {
                    return <$t>::MIN;
                }
                if v > Self::MAX_F64 {
                    return <$t>::MAX;
                }
                let round: fn(f64) -> $t = $round;
                round(v)
            }
            #[inline(always)]
            fn avoid_nodata(self) -> $t {
                if self == <$t>::MIN {
                    <$t>::MIN + 1
                } else {
                    self - 1
                }
            }
            #[inline(always)]
            fn is_nan(self) -> bool {
                false
            }
        }
    };
}

// static_cast<T>(double): on x86-64 a 32-bit conversion for the types up to
// 16 bits and int32, a 64-bit one for uint32 and int64; elsewhere saturating
// to 32 bits (signed or not) or 64.
#[cfg(target_arch = "x86_64")]
mod from {
    use super::{cvt_i32, cvt_i64, cvt_u64};
    pub fn i8(v: f64) -> i8 {
        cvt_i32(v) as i8
    }
    pub fn u8(v: f64) -> u8 {
        cvt_i32(v) as u8
    }
    pub fn i16(v: f64) -> i16 {
        cvt_i32(v) as i16
    }
    pub fn u16(v: f64) -> u16 {
        cvt_i32(v) as u16
    }
    pub fn i32(v: f64) -> i32 {
        cvt_i32(v)
    }
    pub fn u32(v: f64) -> u32 {
        cvt_i64(v) as u32
    }
    pub fn i64(v: f64) -> i64 {
        cvt_i64(v)
    }
    pub fn u64(v: f64) -> u64 {
        cvt_u64(v)
    }
}

#[cfg(not(target_arch = "x86_64"))]
mod from {
    pub fn i8(v: f64) -> i8 {
        v as i32 as i8
    }
    pub fn u8(v: f64) -> u8 {
        v as u32 as u8
    }
    pub fn i16(v: f64) -> i16 {
        v as i32 as i16
    }
    pub fn u16(v: f64) -> u16 {
        v as u32 as u16
    }
    pub fn i32(v: f64) -> i32 {
        v as i32
    }
    pub fn u32(v: f64) -> u32 {
        v as u32
    }
    pub fn i64(v: f64) -> i64 {
        v as i64
    }
    pub fn u64(v: f64) -> u64 {
        v as u64
    }
}

// The rounding: floor(v + 0.5) for the signed types (through an int up to 32
// bits), truncation of v + 0.5 for the unsigned ones.
integer!(u8, small: true, copy: true, bins: 256, offset: 0, from: from::u8, round: |v| from::u8(v + 0.5));
integer!(i8, small: true, copy: true, bins: 256, offset: 128, from: from::i8, round: |v| ifloor(v + 0.5) as i8);
integer!(u16, small: true, copy: true, bins: 65536, offset: 0, from: from::u16, round: |v| from::u16(v + 0.5));
integer!(i16, small: true, copy: true, bins: 65536, offset: 32768, from: from::i16,
         round: |v| ifloor(v + 0.5) as i16);
integer!(u32, small: false, copy: true, bins: 0, offset: 0, from: from::u32, round: |v| from::u32(v + 0.5));
integer!(i32, small: false, copy: true, bins: 0, offset: 0, from: from::i32, round: |v| ifloor(v + 0.5));
integer!(u64, small: false, copy: false, bins: 0, offset: 0, from: from::u64, round: |v| from::u64(v + 0.5));
integer!(i64, small: false, copy: false, bins: 0, offset: 0, from: from::i64,
         round: |v| from::i64((v + 0.5).floor()));

/// `std::nextafter(v, t)`.
#[inline]
fn next_toward_f64(v: f64, t: f64) -> f64 {
    if v.is_nan() || t.is_nan() {
        v + t
    } else if v == t {
        t
    } else if v < t {
        v.next_up()
    } else {
        v.next_down()
    }
}

#[inline]
fn next_toward_f32(v: f32, t: f32) -> f32 {
    if v.is_nan() || t.is_nan() {
        v + t
    } else if v == t {
        t
    } else if v < t {
        v.next_up()
    } else {
        v.next_down()
    }
}

impl Raster for f32 {
    const SMALL: bool = false;
    const BYTE: bool = false;
    const FLOAT32: bool = true;
    const COPY: bool = true;
    const BINS: usize = 0;
    const BINS_OFFSET: i32 = 0;
    const LOWEST_F64: f64 = f32::MIN as f64;
    const MAX_F64: f64 = f32::MAX as f64;

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        cvt_i32(self as f64)
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    #[inline(always)]
    fn clamp_round(v: f64) -> f32 {
        v as f32
    }
    #[inline]
    fn avoid_nodata(self) -> f32 {
        if self == f32::MAX {
            next_toward_f32(self, 0.0)
        } else {
            next_toward_f32(self, f32::MAX)
        }
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl Raster for f64 {
    const SMALL: bool = false;
    const BYTE: bool = false;
    const FLOAT32: bool = false;
    const COPY: bool = true;
    const BINS: usize = 0;
    const BINS_OFFSET: i32 = 0;
    const LOWEST_F64: f64 = f64::MIN;
    const MAX_F64: f64 = f64::MAX;

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        cvt_i32(self)
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn clamp_round(v: f64) -> f64 {
        v
    }
    #[inline]
    fn avoid_nodata(self) -> f64 {
        if self == f64::MAX {
            next_toward_f64(self, 0.0)
        } else {
            next_toward_f64(self, f64::MAX)
        }
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

/// GDAL's ARE_REAL_EQUAL (float epsilon for doubles too).
#[inline(always)]
fn real_equal_f32(a: f32, b: f32) -> bool {
    a == b || (a - b).abs() < f32::EPSILON * (a + b).abs() * 2.0
}

#[inline(always)]
fn real_equal_f64(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() < (f32::EPSILON as f64) * (a + b).abs() * 2.0
}

/// GDALWarpNoDataMasker: is a value the source NoData?
#[derive(Clone, Copy, Debug)]
pub struct NoDataTest {
    /// False when the NoData of an 8 or 16-bit type is out of its range.
    pub active: bool,
    nint: i32,
    fnd: f32,
    dnd: f64,
    nan: bool,
}

impl NoDataTest {
    pub fn new<T: Raster>(nodata: f64) -> NoDataTest {
        let mut t = NoDataTest {
            active: true,
            nint: 0,
            fnd: 0.0,
            dnd: 0.0,
            nan: false,
        };
        if T::SMALL {
            if !(nodata >= T::LOWEST_F64) || nodata > T::MAX_F64 + 0.000001 {
                t.active = false;
            } else {
                t.nint = cvt_i32((nodata + 0.000001).floor());
            }
        } else if T::FLOAT32 {
            t.fnd = nodata as f32;
            t.nan = t.fnd.is_nan();
        } else {
            t.dnd = nodata;
            t.nan = nodata.is_nan();
        }
        t
    }

    #[inline(always)]
    pub fn test<T: Raster>(&self, v: T) -> bool {
        if T::SMALL {
            v.to_i32() == self.nint
        } else if T::FLOAT32 {
            let v = v.to_f32();
            if self.nan {
                v.is_nan()
            } else {
                real_equal_f32(v, self.fnd)
            }
        } else {
            let dv = v.to_f64();
            if self.nan {
                dv.is_nan()
            } else {
                real_equal_f64(dv, self.dnd)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_as_the_cpp() {
        assert_eq!(ifloor(-0.5), -1);
        assert_eq!(ifloor(2.0), 2);
        assert_eq!(iceil(2.1), 3);
        assert_eq!(iceil(-2.1), -2);
        assert_eq!(u8::clamp_round(254.5), 255);
        assert_eq!(u8::clamp_round(-3.0), 0);
        assert_eq!(i16::clamp_round(-2.5), -2);
        assert_eq!(i64::clamp_round(-2.5), -2);
        assert_eq!(u32::clamp_round(1e12), u32::MAX);
        assert_eq!(cvt_u64(1.8e19), 18_000_000_000_000_000_000);
    }

    #[test]
    fn avoids_nodata() {
        assert_eq!(0u8.avoid_nodata(), 1);
        assert_eq!(7u8.avoid_nodata(), 6);
        assert_eq!(f64::MAX.avoid_nodata(), f64::MAX.next_down());
        assert_eq!(0.0f32.avoid_nodata(), f32::from_bits(1));
        assert_eq!(f64::INFINITY.avoid_nodata(), f64::MAX);
    }

    #[test]
    fn nodata_tests() {
        let t = NoDataTest::new::<u8>(300.0);
        assert!(!t.active);
        let t = NoDataTest::new::<u8>(7.0);
        assert!(t.test(7u8) && !t.test(8u8));
        let t = NoDataTest::new::<f32>(f64::NAN);
        assert!(t.test(f32::NAN) && !t.test(1.0f32));
        let t = NoDataTest::new::<f64>(-9999.0);
        assert!(t.test(-9999.0000001f64) && !t.test(-9998.0f64));
    }
}
