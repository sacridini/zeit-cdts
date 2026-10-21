//! Reductions in the order Eigen's vectorized code performs them, for the
//! modules whose C++ used Eigen: the same additions in the same order give the
//! same bits.
//!
//! The C++ core is built for the baseline instruction sets (SSE2 on x86-64,
//! NEON on arm64; no -march), where Eigen's packets hold 2 doubles. Its sum of a
//! dynamic-size vector expression (`.sum()`, `.squaredNorm()`, `.norm()`,
//! `.dot()` of Maps and expressions; Eigen/src/Core/Redux.h,
//! LinearVectorizedTraversal, no alignment peeling for expressions) runs two
//! packet accumulators over blocks of 4, adds them, adds a last packet, sums
//! the lanes, then adds the remaining coefficient one at a time.

pub mod ldlt;
pub mod lm;
pub mod qr;
pub mod sparse;

/// Doubles per packet in the C++ build (SSE2, NEON).
pub const PACKET: usize = 2;

/// Rows of the matrix-matrix kernel's register blocks (gebp_traits::mr): 4 on
/// SSE2 without FMA (16 registers / 2 / nr * 2), 6 where Eigen has a
/// single-instruction multiply-add (NEON).
#[cfg(target_arch = "aarch64")]
pub const MR: usize = 6;
#[cfg(not(target_arch = "aarch64"))]
pub const MR: usize = 4;

/// The size of the L1 data cache as Eigen 3.4 takes it (manage_caching_sizes):
/// queried with cpuid on x86 (leaf 4 on Intel, 0x80000005 on AMD; 32 KB when
/// unknown), 16 KB elsewhere (Eigen's default off x86 and PPC: there is no
/// query on ARM). Queried once, as Eigen does; on a hybrid processor it is the
/// cache of the core that ran the first product.
pub fn l1_cache_size() -> usize {
    static L1: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *L1.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let l1 = x86_l1();
            if l1 > 0 {
                return l1;
            }
            32 * 1024
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            16 * 1024
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn x86_l1() -> usize {
    use std::arch::x86_64::__cpuid_count;
    let id = __cpuid_count(0, 0);
    let vendor = |b: u32, d: u32, c: u32| id.ebx == b && id.edx == d && id.ecx == c;
    let amd = vendor(0x6874_7541, 0x6974_6e65, 0x444d_4163) || vendor(0x6944_4d41, 0x7465_6273, 0x2172_6574);
    if amd {
        let ext = __cpuid_count(0x8000_0000, 0);
        if ext.eax >= 0x8000_0006 {
            return ((__cpuid_count(0x8000_0005, 0).ecx >> 24) as usize) * 1024;
        }
        return 0;
    }
    if id.eax < 4 {
        return 0; // Eigen's leaf-2 descriptors: processors of before 2004
    }
    // Intel's deterministic cache parameters (Eigen's queryCacheSizes_intel_direct)
    let mut l1 = 0;
    for cache_id in 0..16 {
        let r = __cpuid_count(4, cache_id);
        let cache_type = r.eax & 0x0F;
        if cache_type == 0 {
            break;
        }
        if (cache_type == 1 || cache_type == 3) && ((r.eax & 0xE0) >> 5) == 1 {
            let ways = ((r.ebx & 0xFFC0_0000) >> 22) as usize;
            let partitions = ((r.ebx & 0x003F_F000) >> 12) as usize;
            let line = (r.ebx & 0x0000_0FFF) as usize;
            let sets = r.ecx as usize;
            l1 = (ways + 1) * (partitions + 1) * (line + 1) * (sets + 1);
        }
    }
    l1
}

/// The depth Eigen's matrix-matrix product takes at once (kc of
/// evaluateProductBlockingSizesHeuristic, one thread) for a product of depth
/// `k` into `m` x `n`: all of it but past what fits in L1 with the kernel's
/// panels; then blocks as even as possible, multiples of 8.
pub fn gemm_kc(k: usize, m: usize, n: usize) -> usize {
    if k.max(m).max(n) < 48 {
        return k;
    }
    const NR: usize = 4;
    let (k_div, k_sub) = (MR * 8 + NR * 8, MR * NR * 8);
    let l1 = l1_cache_size();
    let max_kc = ((l1.saturating_sub(k_sub) / k_div) & !7).max(1);
    if k > max_kc {
        if k.is_multiple_of(max_kc) {
            max_kc
        } else {
            max_kc - 8 * ((max_kc - 1 - k % max_kc) / (8 * (k / max_kc + 1)))
        }
    } else {
        k
    }
}

/// Eigen's pmadd on packets, `a * b + c`: fused on NEON (vfmaq_f64, one
/// rounding), a multiplication then an addition on SSE2 without FMA. Eigen's
/// scalar code (the leftover rows and columns of its kernels) never fuses.
#[inline(always)]
pub fn pmadd(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

/// Eigen's sum of the `n` coefficients `c(0)`, ..., `c(n - 1)` of an expression.
#[inline]
pub fn sum_by(n: usize, c: impl Fn(usize) -> f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let aligned = n / PACKET * PACKET;
    if aligned == 0 {
        // too small to vectorize: one coefficient
        return c(0);
    }
    let aligned2 = n / (2 * PACKET) * (2 * PACKET);
    let mut r0 = [c(0), c(1)];
    if aligned > PACKET {
        let mut r1 = [c(2), c(3)];
        let mut i = 2 * PACKET;
        while i < aligned2 {
            r0[0] += c(i);
            r0[1] += c(i + 1);
            r1[0] += c(i + 2);
            r1[1] += c(i + 3);
            i += 2 * PACKET;
        }
        r0[0] += r1[0];
        r0[1] += r1[1];
        if aligned > aligned2 {
            r0[0] += c(aligned2);
            r0[1] += c(aligned2 + 1);
        }
    }
    let mut res = r0[0] + r0[1];
    for i in aligned..n {
        res += c(i);
    }
    res
}

/// Eigen's sum of `f(a[i], b[i])` over two slices of the same length: `sum_by`
/// on slices, without bounds checks in the loop.
#[inline(always)]
pub fn sum_zip<A: Copy, B: Copy>(a: &[A], b: &[B], f: impl Fn(A, B) -> f64) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    if n == 0 {
        return 0.0;
    }
    if n < PACKET {
        return f(a[0], b[0]);
    }
    let aligned = n / PACKET * PACKET;
    let aligned2 = n / (2 * PACKET) * (2 * PACKET);
    let mut r0 = [f(a[0], b[0]), f(a[1], b[1])];
    if aligned > PACKET {
        let mut r1 = [f(a[2], b[2]), f(a[3], b[3])];
        for (ca, cb) in a[4..aligned2]
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b[4..aligned2].as_chunks::<4>().0)
        {
            r0[0] += f(ca[0], cb[0]);
            r0[1] += f(ca[1], cb[1]);
            r1[0] += f(ca[2], cb[2]);
            r1[1] += f(ca[3], cb[3]);
        }
        r0[0] += r1[0];
        r0[1] += r1[1];
        if aligned > aligned2 {
            r0[0] += f(a[aligned2], b[aligned2]);
            r0[1] += f(a[aligned2 + 1], b[aligned2 + 1]);
        }
    }
    let mut res = r0[0] + r0[1];
    for (&x, &y) in a[aligned..].iter().zip(&b[aligned..]) {
        res += f(x, y);
    }
    res
}

/// `std::min(a, b)`: `b` if `b < a`, else `a` (so `a` when either is NaN).
#[inline(always)]
pub fn std_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `std::max(a, b)`: `b` if `a < b`, else `a` (so `a` when either is NaN).
#[inline(always)]
pub fn std_max(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// Eigen's pmin of a lane of the accumulator `a` and of a new value `b`. SSE2:
/// `_mm_min_pd(b, a)`, `a` kept on ties and NaN, as std::min. NEON: vminq_f64,
/// NaN when either is, -0 below +0.
#[inline(always)]
pub fn pmin(a: f64, b: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else if a == b {
            if a.is_sign_negative() {
                a
            } else {
                b
            }
        } else if a < b {
            a
        } else {
            b
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if b < a {
            b
        } else {
            a
        }
    }
}

/// Eigen's pmax, as `pmin`: SSE2 `_mm_max_pd(b, a)`, NEON vmaxq_f64.
#[inline(always)]
pub fn pmax(a: f64, b: f64) -> f64 {
    #[cfg(target_arch = "aarch64")]
    {
        if a.is_nan() || b.is_nan() {
            f64::NAN
        } else if a == b {
            if a.is_sign_negative() {
                b
            } else {
                a
            }
        } else if a < b {
            b
        } else {
            a
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        if b > a {
            b
        } else {
            a
        }
    }
}

/// Eigen's reduction of `n` > 0 coefficients in the order of `sum_by`: `pop`
/// on the packet lanes, `fin` on the two lanes, `sop` on the rest. For
/// minCoeff and maxCoeff, `fin` is not the platform's predux_min (vpminq on
/// NEON, `_mm_min_sd` on SSE2) but Eigen's generic predux_helper with the
/// scalar std::min: on NEON a NaN in the second lane is dropped there.
#[inline]
fn redux(
    n: usize,
    c: impl Fn(usize) -> f64,
    pop: impl Fn(f64, f64) -> f64,
    fin: impl Fn(f64, f64) -> f64,
    sop: impl Fn(f64, f64) -> f64,
) -> f64 {
    let aligned = n / PACKET * PACKET;
    if aligned == 0 {
        return c(0);
    }
    let aligned2 = n / (2 * PACKET) * (2 * PACKET);
    let mut r0 = [c(0), c(1)];
    if aligned > PACKET {
        let mut r1 = [c(2), c(3)];
        let mut i = 2 * PACKET;
        while i < aligned2 {
            r0 = [pop(r0[0], c(i)), pop(r0[1], c(i + 1))];
            r1 = [pop(r1[0], c(i + 2)), pop(r1[1], c(i + 3))];
            i += 2 * PACKET;
        }
        r0 = [pop(r0[0], r1[0]), pop(r0[1], r1[1])];
        if aligned > aligned2 {
            r0 = [pop(r0[0], c(aligned2)), pop(r0[1], c(aligned2 + 1))];
        }
    }
    let mut res = fin(r0[0], r0[1]);
    for i in aligned..n {
        res = sop(res, c(i));
    }
    res
}

/// `minCoeff()` of a vector expression of `n` > 0 coefficients.
#[inline]
pub fn min_by(n: usize, c: impl Fn(usize) -> f64) -> f64 {
    redux(n, c, pmin, std_min, std_min)
}

/// `maxCoeff()` of a vector expression of `n` > 0 coefficients.
#[inline]
pub fn max_by(n: usize, c: impl Fn(usize) -> f64) -> f64 {
    redux(n, c, pmax, std_max, std_max)
}

/// `minCoeff()` of a vector.
#[inline]
pub fn min_coeff(v: &[f64]) -> f64 {
    min_by(v.len(), |i| v[i])
}

/// `maxCoeff()` of a vector.
#[inline]
pub fn max_coeff(v: &[f64]) -> f64 {
    max_by(v.len(), |i| v[i])
}

/// `(a - b).squaredNorm()`.
#[inline]
pub fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    sum_zip(a, b, |x, y| {
        let d = x - y;
        d * d
    })
}

/// `(a - b).norm()`.
#[inline]
pub fn distance(a: &[f64], b: &[f64]) -> f64 {
    squared_distance(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two lanes meet in std::min / std::max, on SSE2 and on NEON alike: a
    /// NaN left in the second lane is dropped, one in the first is kept.
    #[test]
    fn min_max_coeff_fold_the_lanes_with_std() {
        let nan = f64::NAN;
        assert_eq!(min_coeff(&[1.0, nan, 2.0, 3.0]), 1.0);
        assert_eq!(max_coeff(&[1.0, nan, 2.0, 3.0]), 2.0); // the 3 shared a lane with the NaN
        assert!(min_coeff(&[nan, 1.0, 2.0, 3.0]).is_nan());
        assert!(max_coeff(&[nan, 1.0, 2.0, 3.0]).is_nan());
    }

    /// The order spelled out for each length up to 9.
    #[test]
    fn order_of_the_additions() {
        let c = |i: usize| [1e16, 1.0, -1e16, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0][i];
        let expect = [
            0.0,
            c(0),
            c(0) + c(1),
            (c(0) + c(1)) + c(2),
            (c(0) + c(2)) + (c(1) + c(3)),
            ((c(0) + c(2)) + (c(1) + c(3))) + c(4),
            ((c(0) + c(2) + c(4)) + (c(1) + c(3) + c(5))),
            ((c(0) + c(2) + c(4)) + (c(1) + c(3) + c(5))) + c(6),
            ((c(0) + c(4)) + (c(2) + c(6))) + ((c(1) + c(5)) + (c(3) + c(7))),
            (((c(0) + c(4)) + (c(2) + c(6))) + ((c(1) + c(5)) + (c(3) + c(7)))) + c(8),
        ];
        let values: Vec<f64> = (0..9).map(c).collect();
        let zeros = [0.0; 9];
        for (n, e) in expect.iter().enumerate() {
            assert_eq!(sum_by(n, c).to_bits(), e.to_bits(), "n = {n}");
            assert_eq!(
                sum_zip(&values[..n], &zeros[..n], |a, b| a + b).to_bits(),
                e.to_bits(),
                "n = {n}"
            );
        }
    }
}
