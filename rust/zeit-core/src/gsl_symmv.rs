//! `gsl_eigen_symmv` followed by `gsl_eigen_symmv_sort(.., GSL_EIGEN_SORT_VAL_DESC)`: the eigenvalues and
//! eigenvectors of a real symmetric matrix as GSL 2.7 gives them, signs and last bits included.
//!
//! Ported from the GNU Scientific Library 2.7 (https://www.gnu.org/software/gsl/), GPL-3.0-or-later:
//!
//! * eigen/symmv.c, Copyright (C) 2001, 2007 Brian Gough
//! * eigen/qrstep.c, Copyright (C) 2007, 2010 Brian Gough
//! * eigen/sort.c, Copyright (C) 1996, 1997, 1998, 1999, 2000, 2006, 2007 Gerard Jungman, Patrick Alken
//! * linalg/symmtd.c, Copyright (C) 2001, 2007 Brian Gough, Copyright (C) 2019 Patrick Alken
//! * linalg/householder.c, Copyright (C) 1996, 1997, 1998, 1999, 2000, 2004, 2007, 2010 Gerard Jungman, Brian Gough
//! * the reference CBLAS of GSL (cblas/source_nrm2_r.h, source_dot_r.h, source_axpy_r.h, source_scal_r.h,
//!   source_symv.h, source_syr2.h, source_gemv_r.h, source_ger.h, Copyright (C) 1996, 1997, 1998, 1999, 2000 Gerard
//!   Jungman) that those functions call: `dnrm2`, `dscal`, `ddot`, `daxpy`, `dsymv`, `dsyr2`, `dgemv`, `dger`, for
//!   the row-major storage `gsl_matrix` has.
//!
//! This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
//! Public License as published by the Free Software Foundation, either version 3 of the License, or (at your
//! option) any later version.
//!
//! The method: Householder reduction of the lower triangle to a tridiagonal matrix, the orthogonal factor
//! formed explicitly, then the implicit-shift QR iteration (Golub and Van Loan, Matrix Computations, 3rd ed.,
//! section 8.3) with its Givens rotations applied to the factor. The operations are in GSL's order and plain
//! IEEE (no fused multiply-add), as GSL's reference CBLAS compiles for the baseline x86-64 and arm64: if FORCE
//! was linked to an optimized BLAS instead (OpenBLAS), the last bits of its eigenvectors differ from these.
//! The eigenvectors are the COLUMNS of the returned row-major matrix, as in `gsl_matrix`. `hypot` is correctly
//! rounded, as glibc's (see `hypot` below). Checked against GSL 2.7 itself linked to its reference CBLAS
//! (`benchmarks/vs_originals/gsl_stages.c`, `examples/gsl_stages.rs`): every stage of the reduction and the
//! eigenvectors equal to the last bit on the covariance matrices of the ImproPhe comparison.

const EPS: f64 = f64::EPSILON;

/// `hypot(x, y)` correctly rounded (but for a result within 2^-100 of a tie): the square roots of the double-double
/// sum of the exact squares, corrected once. GSL calls the C library's `hypot`, which on glibc 2.35 and later (Borges's
/// algorithm) is correctly rounded in practice, while Rust's `f64::hypot` is the platform's: the Windows C runtime's
/// differs from it in the last bit in about one call of ten, which a Householder reduction turns into last-bit
/// differences in every eigenvector. This one gives glibc's values wherever both are correctly rounded.
fn hypot(x: f64, y: f64) -> f64 {
    let (x, y) = (x.abs(), y.abs());
    if x.is_infinite() || y.is_infinite() {
        return f64::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    let (big, small) = if x >= y { (x, y) } else { (y, x) };
    if small == 0.0 {
        return big;
    }
    // scale by a power of two so that the larger is in [1, 2): the squares neither overflow nor underflow
    let mut pre = 1.0f64;
    let mut bg = big;
    if bg < f64::MIN_POSITIVE {
        pre = 18446744073709551616.0; // 2^64
        bg *= pre;
    }
    let e = ((bg.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let pow2 = |k: i32| f64::from_bits(((1023 + k) as u64) << 52);
    // two steps keep each factor a normal power of two even for the extreme exponents
    let (k1, k2) = (-e / 2, -e - (-e / 2));
    let (a, b) = (bg * pow2(k1) * pow2(k2), small * pre * pow2(k1) * pow2(k2));
    let (ah, al) = (a * a, a.mul_add(a, -(a * a)));
    let (bh, bl) = (b * b, b.mul_add(b, -(b * b)));
    // (sh, sl) = ah + bh + al + bl as a normalized double-double
    let sh0 = ah + bh;
    let t = sh0 - ah;
    let lo = (ah - (sh0 - t)) + (bh - t) + (al + bl);
    let sh = sh0 + lo;
    let sl = lo - (sh - sh0);
    let r0 = sh.sqrt();
    let corr = (-r0).mul_add(r0, sh) + sl;
    let r = r0 + corr / (2.0 * r0);
    r / pre * pow2(-k1) * pow2(-k2)
}

/// `cblas_dnrm2`: the scaled sum of squares of the reference BLAS.
fn dnrm2(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    if x.len() == 1 {
        return x[0].abs();
    }
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &v in x {
        if v != 0.0 {
            let ax = v.abs();
            if scale < ax {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// `gsl_linalg_householder_transform`: `v` becomes the Householder vector (`v[0]` holds beta, the unit first
/// element is implicit) and the coefficient tau is returned.
fn householder_transform(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 1 {
        return 0.0;
    }
    let xnorm = dnrm2(&v[1..]);
    if xnorm == 0.0 {
        return 0.0;
    }
    let alpha = v[0];
    let sign = if alpha >= 0.0 { 1.0 } else { -1.0 };
    let beta = -sign * hypot(alpha, xnorm);
    let tau = (beta - alpha) / beta;
    let s = alpha - beta;
    if s.abs() > f64::MIN_POSITIVE {
        let f = 1.0 / s;
        for x in &mut v[1..] {
            *x *= f;
        }
        v[0] = beta;
    } else {
        let f = EPS / s;
        for x in &mut v[1..] {
            *x *= f;
        }
        let f = 1.0 / EPS;
        for x in &mut v[1..] {
            *x *= f;
        }
        v[0] = beta;
    }
    tau
}

/// `gsl_linalg_symmtd_decomp`: the tridiagonal form of the lower triangle of `a` (n x n, row-major) in its
/// diagonal and first subdiagonal, the Householder vectors below, and `tau` (length n - 1).
fn symmtd_decomp(a: &mut [f64], n: usize, tau: &mut [f64]) {
    let mut v = vec![0.0f64; n];
    let mut x = vec![0.0f64; n];
    for i in 0..n - 2 {
        let m = n - i - 1;
        for k in 0..m {
            v[k] = a[(i + 1 + k) * n + i];
        }
        let tau_i = householder_transform(&mut v[..m]);
        if tau_i != 0.0 {
            let ei = v[0];
            v[0] = 1.0;
            let off = (i + 1) * n + (i + 1); // the m x m submatrix starts here, row stride n
            let mat = |r: usize, c: usize| off + r * n + c;
            // x = tau * A * v, `cblas_dsymv(RowMajor, Lower)`, beta = 0
            for e in &mut x[..m] {
                *e = 0.0;
            }
            for r in (0..m).rev() {
                let temp1 = tau_i * v[r];
                let mut temp2 = 0.0;
                x[r] += temp1 * a[mat(r, r)];
                for c in 0..r {
                    x[c] += temp1 * a[mat(r, c)];
                    temp2 += v[c] * a[mat(r, c)];
                }
                x[r] += tau_i * temp2;
            }
            // w = x - (1/2) tau (x' v) v
            let mut xv = 0.0;
            for k in 0..m {
                xv += x[k] * v[k];
            }
            let alpha = -0.5 * tau_i * xv;
            for k in 0..m {
                x[k] += alpha * v[k];
            }
            // A = A - v w' - w v', `cblas_dsyr2(RowMajor, Lower, -1.0, v, w)`
            for r in 0..m {
                let tmp1 = -1.0 * v[r];
                let tmp2 = -1.0 * x[r];
                for c in 0..=r {
                    a[mat(r, c)] += tmp1 * x[c] + tmp2 * v[c];
                }
            }
            v[0] = ei;
        }
        // the Householder vector stays in the column (below the subdiagonal)
        for k in 0..m {
            a[(i + 1 + k) * n + i] = v[k];
        }
        tau[i] = tau_i;
    }
}

/// `gsl_linalg_householder_left` on the m x m block of `q` at (`off`, `off`) (row stride `n`) with `h`.
fn householder_left(tau: f64, h: &[f64], q: &mut [f64], off: usize, n: usize, work: &mut [f64]) {
    if tau == 0.0 {
        return;
    }
    let m = h.len();
    // work := A^T h, `cblas_dgemv(RowMajor, Trans)`
    for w in &mut work[..m] {
        *w = 0.0;
    }
    for j in 0..m {
        let temp = 1.0 * h[j];
        if temp != 0.0 {
            for i in 0..m {
                work[i] += temp * q[off + j * n + i];
            }
        }
    }
    // A := A - tau h work^T, `cblas_dger(RowMajor)`
    let alpha = -tau;
    for i in 0..m {
        let tmp = alpha * h[i];
        for j in 0..m {
            q[off + i * n + j] += work[j] * tmp;
        }
    }
}

/// `gsl_linalg_symmtd_unpack`: the orthogonal factor `q` (row-major), the diagonal `d` and the subdiagonal `sd`.
fn symmtd_unpack(a: &[f64], n: usize, tau: &[f64], q: &mut [f64], d: &mut [f64], sd: &mut [f64]) {
    for e in q.iter_mut() {
        *e = 0.0;
    }
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    let mut work = vec![0.0f64; n];
    let mut h = vec![0.0f64; n];
    // for (i = N - 2; i-- > 0;)
    let mut i = n - 2;
    while i > 0 {
        i -= 1;
        let m = n - i - 1;
        for k in 0..m {
            h[k] = a[(i + 1 + k) * n + i];
        }
        h[0] = 1.0;
        householder_left(tau[i], &h[..m], q, (i + 1) * n + (i + 1), n, &mut work);
    }
    for i in 0..n {
        d[i] = a[i * n + i];
    }
    for i in 0..n - 1 {
        sd[i] = a[(i + 1) * n + i];
    }
}

/// `chop_small_elements`.
fn chop_small_elements(n: usize, d: &[f64], sd: &mut [f64]) {
    let mut d_i = d[0];
    for i in 0..n - 1 {
        let sd_i = sd[i];
        let d_ip1 = d[i + 1];
        if sd_i.abs() < EPS * (d_i.abs() + d_ip1.abs()) {
            sd[i] = 0.0;
        }
        d_i = d_ip1;
    }
}

/// `create_givens`: (cos, sin) taking (a, b) to (|v|, 0).
fn create_givens(a: f64, b: f64) -> (f64, f64) {
    if b == 0.0 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let t = -a / b;
        let s1 = 1.0 / (1.0 + t * t).sqrt();
        (s1 * t, s1)
    } else {
        let t = -b / a;
        let c1 = 1.0 / (1.0 + t * t).sqrt();
        (c1, c1 * t)
    }
}

/// `trailing_eigenvalue`: the Wilkinson shift.
fn trailing_eigenvalue(n: usize, d: &[f64], sd: &[f64]) -> f64 {
    let ta = d[n - 2];
    let tb = d[n - 1];
    let tab = sd[n - 2];
    let dt = (ta - tb) / 2.0;
    if dt > 0.0 {
        tb - tab * (tab / (dt + hypot(dt, tab)))
    } else if dt == 0.0 {
        tb - tab.abs()
    } else {
        tb + tab * (tab / ((-dt) + hypot(dt, tab)))
    }
}

/// `qrstep`: one implicit-shift QR step on the unreduced block (`d` of length n, `sd` of length n - 1), the
/// rotations into `gc` and `gs`.
fn qrstep(n: usize, d: &mut [f64], sd: &mut [f64], gc: &mut [f64], gs: &mut [f64]) {
    let mut mu = trailing_eigenvalue(n, d, sd);
    if EPS * mu.abs() > (d[0].abs() + sd[0].abs()) {
        mu = 0.0;
    }
    let mut x = d[0] - mu;
    let mut z = sd[0];
    let mut ak;
    let mut bk = 0.0f64;
    let mut zk = 0.0f64;
    let mut ap = d[0];
    let mut bp = sd[0];
    let mut aq = d[1];

    if n == 2 {
        let (c, s) = create_givens(x, z);
        gc[0] = c;
        gs[0] = s;
        let ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        let bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        let aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        ak = ap1;
        bk = bp1;
        ap = aq1;
        d[0] = ak;
        sd[0] = bk;
        d[1] = ap;
        return;
    }

    let mut bq = sd[1];
    let mut k = 0usize;
    while k < n - 1 {
        let (c, s) = create_givens(x, z);
        gc[k] = c;
        gs[k] = s;

        let bk1 = c * bk - s * zk;
        let ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        let bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        let zp1 = -s * bq;
        let aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        let bq1 = c * bq;

        ak = ap1;
        bk = bp1;
        zk = zp1;
        ap = aq1;
        bp = bq1;

        if k < n - 2 {
            aq = d[k + 2];
        }
        if k + 3 < n {
            bq = sd[k + 2];
        }
        d[k] = ak;
        if k > 0 {
            sd[k - 1] = bk1;
        }
        if k < n - 2 {
            sd[k + 1] = bp;
        }
        x = bk;
        z = zk;
        k += 1;
    }
    // k = n - 1
    d[k] = ap;
    sd[k - 1] = bk;
}

/// `gsl_eigen_symmv`: the eigenvalues (not sorted) and the eigenvectors, as the columns of the returned
/// row-major n x n matrix, of the symmetric `a` (n x n, row-major; only the lower triangle is read). `a` is
/// destroyed.
pub fn eigen_symmv(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(a.len(), n * n);
    assert!(n > 0, "matrix dimension must be positive integer");
    if n == 1 {
        return (vec![a[0]], vec![1.0]);
    }
    let mut d = vec![0.0f64; n];
    let mut sd = vec![0.0f64; n];
    let mut evec = vec![0.0f64; n * n];
    {
        let mut tau = vec![0.0f64; n - 1];
        symmtd_decomp(a, n, &mut tau);
        symmtd_unpack(a, n, &tau, &mut evec, &mut d, &mut sd);
    }
    chop_small_elements(n, &d, &mut sd);

    let mut gc = vec![0.0f64; n];
    let mut gs = vec![0.0f64; n];
    let mut b = n - 1;
    while b > 0 {
        if sd[b - 1] == 0.0 || sd[b - 1].is_nan() {
            b -= 1;
            continue;
        }
        // the largest unreduced block (a, b) starting from b and working backwards
        let mut lo = b - 1;
        while lo > 0 {
            if sd[lo - 1] == 0.0 {
                break;
            }
            lo -= 1;
        }
        let n_block = b - lo + 1;
        qrstep(n_block, &mut d[lo..], &mut sd[lo..], &mut gc, &mut gs);
        // Q <- Q G
        for i in 0..n_block - 1 {
            let (c, s) = (gc[i], gs[i]);
            for k in 0..n {
                let qki = evec[k * n + lo + i];
                let qkj = evec[k * n + lo + i + 1];
                evec[k * n + lo + i] = qki * c - qkj * s;
                evec[k * n + lo + i + 1] = qki * s + qkj * c;
            }
        }
        chop_small_elements(n, &d, &mut sd);
    }
    (d, evec)
}

/// The stages of the reduction of the symmetric `a` (row-major, lower triangle read): `(tau, d, sd, q)` after
/// `gsl_linalg_symmtd_decomp` and `_unpack`, to be compared with GSL's own (`examples/gsl_stages.rs`).
#[doc(hidden)]
pub fn symmtd_stages(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut tau = vec![0.0f64; n - 1];
    symmtd_decomp(&mut a, n, &mut tau);
    let (mut q, mut d, mut sd) = (vec![0.0f64; n * n], vec![0.0f64; n], vec![0.0f64; n - 1]);
    symmtd_unpack(&a, n, &tau, &mut q, &mut d, &mut sd);
    (tau, d, sd, q)
}

/// `gsl_eigen_symmv_sort(.., GSL_EIGEN_SORT_VAL_DESC)`: a selection sort, in place, swapping the columns of
/// `evec` with the eigenvalues.
pub fn eigen_symmv_sort_desc(eval: &mut [f64], evec: &mut [f64]) {
    let n = eval.len();
    assert_eq!(evec.len(), n * n);
    for i in 0..n.saturating_sub(1) {
        let mut k = i;
        let mut ek = eval[i];
        for j in i + 1..n {
            let ej = eval[j];
            if ej > ek {
                k = j;
                ek = ej;
            }
        }
        if k != i {
            eval.swap(i, k);
            for r in 0..n {
                evec.swap(r * n + i, r * n + k);
            }
        }
    }
}

/// `gsl_eigen_symmv` then `gsl_eigen_symmv_sort(VAL_DESC)`: eigenvalues in descending order and the matching
/// eigenvectors as columns of the row-major matrix.
pub fn eigen_symmv_desc(a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut eval, mut evec) = eigen_symmv(a, n);
    eigen_symmv_sort_desc(&mut eval, &mut evec);
    (eval, evec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Mat, Side};

    #[test]
    fn hypot_is_correctly_rounded() {
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(-3.0, 4.0), 5.0);
        assert_eq!(hypot(1.0, 1.0).to_bits(), 0x3FF6A09E667F3BCD); // sqrt(2), correctly rounded
        assert_eq!(hypot(0.0, 2.5), 2.5);
        assert_eq!(hypot(1e300, 1e300), 1.4142135623730951e300);
        assert_eq!(hypot(1e-300, 1e-300), 1.4142135623730951e-300);
        assert_eq!(hypot(1e-320, 0.0), 1e-320);
        assert!(hypot(f64::NAN, 1.0).is_nan());
        assert_eq!(hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
        // against the exact value: the square of the result is within half an ulp's worth of x^2 + y^2, checked in
        // 128-bit integers on values with a 40-bit mantissa (their squares and sum are exact)
        let mut s = 7u64;
        for _ in 0..20000 {
            let a = (lcg(&mut s) * (1u64 << 40) as f64).floor() + 1.0;
            let b = (lcg(&mut s) * (1u64 << 40) as f64).floor() + 1.0;
            let h = hypot(a, b);
            let (ai, bi) = (a as u128, b as u128);
            // exact, < 2^82
            let t = ai * ai + bi * bi;
            // h is correctly rounded iff neither neighbouring double is closer to the root of t
            let up = f64::from_bits(h.to_bits() + 1);
            let dn = f64::from_bits(h.to_bits() - 1);
            // exact test with big integers: v = m * 2^e, v^2 = m^2 * 2^(2e)
            let exact = |v: f64| -> (u128, i32) {
                let bits = v.to_bits();
                let m = ((bits & ((1u64 << 52) - 1)) | (1u64 << 52)) as u128;
                (m, ((bits >> 52) & 0x7ff) as i32 - 1075)
            };
            let sq_diff = |v: f64| -> f64 {
                // |v^2 - t| as f64, computed exactly: m^2 < 2^106, shifted to the scale of t
                let (m, e) = exact(v);
                let (m2, e2) = (m * m, 2 * e);
                let d: i128 = if e2 >= 0 {
                    (m2 << e2) as i128 - t as i128
                } else {
                    // t * 2^-e2 vs m2
                    (m2 as i128) - ((t << (-e2)) as i128)
                };
                (d.unsigned_abs() as f64) * if e2 >= 0 { 1.0 } else { 2f64.powi(e2) }
            };
            // v closer to the true root than h's neighbours: |v^2 - t| / (v + root) is monotone; a neighbour that is
            // strictly closer would have a strictly smaller |v - root|, i.e. (v^2 - t)^2 / (v + root)^2 smaller
            let root = (t as f64).sqrt();
            let err = |v: f64| sq_diff(v) / (v + root);
            assert!(
                err(h) <= err(up) * (1.0 + 1e-9) && err(h) <= err(dn) * (1.0 + 1e-9),
                "hypot({a}, {b}) = {h}"
            );
        }
    }

    fn lcg(s: &mut u64) -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// A symmetric positive definite matrix B^T B + eps I from a random B.
    fn spd(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        let b: Vec<f64> = (0..n * n).map(|_| lcg(&mut s) * 2.0 - 1.0).collect();
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut t = 0.0;
                for k in 0..n {
                    t += b[k * n + i] * b[k * n + j];
                }
                a[i * n + j] = t + if i == j { 0.01 } else { 0.0 };
            }
        }
        a
    }

    /// Checks A v = l v, orthonormality and the values against faer's.
    fn check(a: &[f64], n: usize) {
        let mut work = a.to_vec();
        let (eval, evec) = eigen_symmv_desc(&mut work, n);
        let scale = eval.iter().fold(0.0f64, |m, &e| m.max(e.abs())).max(1.0);
        for c in 0..n {
            for r in 0..n {
                let av: f64 = (0..n).map(|k| a[r * n + k] * evec[k * n + c]).sum();
                assert!(
                    (av - eval[c] * evec[r * n + c]).abs() < 1e-12 * scale,
                    "A v != l v at ({r},{c})"
                );
            }
        }
        for c1 in 0..n {
            for c2 in 0..n {
                let dot: f64 = (0..n).map(|k| evec[k * n + c1] * evec[k * n + c2]).sum();
                let want = if c1 == c2 { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "not orthonormal ({c1},{c2}): {dot}");
            }
        }
        for w in eval.windows(2) {
            assert!(w[0] >= w[1], "not descending");
        }
        let m = Mat::from_fn(n, n, |i, j| a[i * n + j]);
        let e = m.self_adjoint_eigen(Side::Lower).unwrap();
        let s = e.S().column_vector();
        for i in 0..n {
            let f = s[n - 1 - i];
            assert!(
                (f - eval[i]).abs() < 1e-10 * scale,
                "value {i}: faer {f}, gsl {}",
                eval[i]
            );
        }
    }

    #[test]
    fn a_3x3_matrix() {
        check(&[4.0, 1.0, 0.5, 1.0, 3.0, -0.25, 0.5, -0.25, 2.0], 3);
    }

    #[test]
    fn small_sizes() {
        let (e, v) = eigen_symmv_desc(&mut [5.0], 1);
        assert_eq!((e[0], v[0]), (5.0, 1.0));
        check(&[2.0, 1.0, 1.0, 2.0], 2);
        for n in [4usize, 5, 7, 12] {
            check(&spd(n, 99 + n as u64), n);
        }
    }

    #[test]
    fn a_40x40_random_spd_matrix() {
        check(&spd(40, 2024), 40);
    }

    #[test]
    fn repeated_eigenvalues_and_diagonal() {
        // identity, a diagonal with repeats, and a block with a repeated eigenvalue
        let n = 6;
        let mut id = vec![0.0; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        check(&id, n);
        let mut dg = vec![0.0; n * n];
        for (i, v) in [3.0, 1.0, 3.0, 2.0, 1.0, 3.0].iter().enumerate() {
            dg[i * n + i] = *v;
        }
        check(&dg, n);
        // 2 I + 1 1^T (n = 5): eigenvalues 7, 2, 2, 2, 2
        let n = 5;
        let a: Vec<f64> = (0..n * n).map(|i| if i / n == i % n { 3.0 } else { 1.0 }).collect();
        check(&a, n);
        let (e, _) = eigen_symmv_desc(&mut a.clone(), n);
        assert!((e[0] - 7.0).abs() < 1e-12 && e[1..].iter().all(|&x| (x - 2.0).abs() < 1e-12));
    }

    #[test]
    fn the_sort_swaps_columns_with_values() {
        let mut e = vec![1.0, 3.0, 2.0];
        // columns are (1,0,0), (0,1,0), (0,0,1) as a row-major identity
        let mut v = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        eigen_symmv_sort_desc(&mut e, &mut v);
        assert_eq!(e, vec![3.0, 2.0, 1.0]);
        // new columns: old 1, old 2, old 0
        assert_eq!(v, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn a_known_2x2_decomposition() {
        // [[2, 1], [1, 2]]: eigenvalues 3 and 1; the vectors are (1, 1) / sqrt 2 up to sign
        let (e, v) = eigen_symmv_desc(&mut [2.0, 1.0, 1.0, 2.0], 2);
        assert!((e[0] - 3.0).abs() < 1e-15 && (e[1] - 1.0).abs() < 1e-15);
        let r = 1.0 / 2.0f64.sqrt();
        assert!((v[0].abs() - r).abs() < 1e-15 && (v[2].abs() - r).abs() < 1e-15);
        assert!(v[0] * v[2] > 0.0, "the first vector is (1, 1) / sqrt 2 up to a sign");
    }
}
