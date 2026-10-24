//! Rbeast's arithmetic: its fast exp, log and square root (`abc_math.c`), the
//! generic (non-AVX) vector kernels of `abc_vec_generic.c` that its own BLAS is
//! made of, and the Cholesky updates and triangular solves of `abc_mat.c`, each
//! in single precision with the sums in the order the C writes them (and in
//! double where the C promotes).

#![allow(clippy::excessive_precision, clippy::approx_constant)]

/// `fastexp`.
#[inline]
pub fn fastexp(x: f32) -> f32 {
    let x = if x > 88.7 { 88.7 } else { x };
    let x = if x < -51.0 { -51.0 } else { x };
    let x = 1.442695040f32 * x;
    let z = x - (x as i32) as f32 + if x < 0.0 { 1.0f32 } else { 0.0 };
    let v = 8388608.0f32 * (x + 121.2740575f32 + 27.7280233f32 / (4.84252568f32 - z) - 1.49012907f32 * z);
    f32::from_bits(v as u32)
}

/// `fastsqrt`.
#[inline]
pub fn fastsqrt(x: f32) -> f32 {
    let mut i = x.to_bits();
    i = i.wrapping_sub(1 << 23);
    i >>= 1;
    i = i.wrapping_add(1 << 29);
    let mut v = f32::from_bits(i);
    v += x / v;
    v * 0.25f32 + x / v
}

/// `fastlog`.
#[inline]
pub fn fastlog(x: f32) -> f32 {
    let i = x.to_bits();
    let m = f32::from_bits((i & 0x007F_FFFF) | 0x3f00_0000);
    let v = i as f32 * (1.1920928955078125e-7f32 * 0.69314718f32);
    let a = v - 125.5233166734556f32 * 0.69314718f32;
    let inner = -0.472721975352920f32 + 0.078018528401178f32 * m;
    let b = (m as f64) * (-0.413356886671142f64 + (m * inner) as f64) * 0.69314718f32 as f64;
    let c = -0.787757784962750f32 * 0.69314718f32 / (0.1781810261970705f32 + m);
    ((a as f64 + b) + c as f64) as f32
}

/// `sum_log_diagv2`: the log of the product of the diagonal of a `k`-by-`k`
/// matrix, the product in double, its log taken before it over- or underflows.
pub fn sum_log_diag(u: &[f32], k: usize) -> f32 {
    let mut sumlog = 0f64;
    let mut cumprod = 1f64;
    for i in 0..k {
        let x = u[i * (k + 1)] as f64;
        let next = cumprod * x;
        if next > 1e-305 && next < 1e305 {
            cumprod = next;
        } else {
            sumlog += cumprod.ln();
            cumprod = x;
        }
    }
    sumlog += cumprod.ln();
    sumlog as f32
}

/// Four products summed left to right, the group of `gen_f32_dot`.
#[inline(always)]
fn quad(x: &[f32; 4], y: &[f32; 4]) -> f32 {
    x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
}

/// `gen_f32_dot`: four products summed, then added to the running sum.
#[inline]
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let (x, y) = (&x[..n], &y[..n]);
    let mut sum = 0f32;
    for (a, b) in x.as_chunks::<4>().0.iter().zip(y.as_chunks::<4>().0) {
        sum += quad(a, b);
    }
    let r = n & !3;
    for (a, b) in x[r..].iter().zip(&y[r..]) {
        sum += a * b;
    }
    sum
}

/// `gen_f32_dot2x2`: the four dots of `x1`, `x2` with `y1`, `y2`, each summed
/// as `dot` sums it (so the same values), side by side: `[x1 y1, x2 y1,
/// x1 y2, x2 y2]`.
#[inline]
fn dot2x2(x1: &[f32], x2: &[f32], y1: &[f32], y2: &[f32], n: usize) -> [f32; 4] {
    let (x1, x2, y1, y2) = (&x1[..n], &x2[..n], &y1[..n], &y2[..n]);
    let (mut s11, mut s21, mut s12, mut s22) = (0f32, 0f32, 0f32, 0f32);
    for (((a1, a2), b1), b2) in x1
        .as_chunks::<4>()
        .0
        .iter()
        .zip(x2.as_chunks::<4>().0)
        .zip(y1.as_chunks::<4>().0)
        .zip(y2.as_chunks::<4>().0)
    {
        s11 += quad(a1, b1);
        s21 += quad(a2, b1);
        s12 += quad(a1, b2);
        s22 += quad(a2, b2);
    }
    for i in n & !3..n {
        s11 += x1[i] * y1[i];
        s21 += x2[i] * y1[i];
        s12 += x1[i] * y2[i];
        s22 += x2[i] * y2[i];
    }
    [s11, s21, s12, s22]
}

/// `gen_f32_dot2x1`: the dots of `x1` and `x2` with `v`.
#[inline]
fn dot2x1(x1: &[f32], x2: &[f32], v: &[f32], n: usize) -> [f32; 2] {
    let (x1, x2, v) = (&x1[..n], &x2[..n], &v[..n]);
    let (mut s1, mut s2) = (0f32, 0f32);
    for ((a1, a2), b) in x1
        .as_chunks::<4>()
        .0
        .iter()
        .zip(x2.as_chunks::<4>().0)
        .zip(v.as_chunks::<4>().0)
    {
        s1 += quad(a1, b);
        s2 += quad(a2, b);
    }
    for i in n & !3..n {
        s1 += x1[i] * v[i];
        s2 += x2[i] * v[i];
    }
    [s1, s2]
}

/// `gen_f32_gemm_XtY2x2`: `out[c * ld + r] = dot(x_r, y_c)` for the `kx`
/// columns `x_r` of `x` (leading dimension `ldx`) and the `ky` columns of
/// `y`, each `len` long, two by two. Each value is the `dot` of its pair.
#[allow(clippy::too_many_arguments)]
pub fn cross(
    x: &[f32],
    ldx: usize,
    kx: usize,
    y: &[f32],
    ldy: usize,
    ky: usize,
    len: usize,
    out: &mut [f32],
    ld: usize,
) {
    let xc = |r: usize| &x[r * ldx..r * ldx + len];
    let yc = |c: usize| &y[c * ldy..c * ldy + len];
    let mut c = 0;
    while c + 1 < ky {
        let (y1, y2) = (yc(c), yc(c + 1));
        let mut r = 0;
        while r + 1 < kx {
            let [s11, s21, s12, s22] = dot2x2(xc(r), xc(r + 1), y1, y2, len);
            out[c * ld + r] = s11;
            out[c * ld + r + 1] = s21;
            out[(c + 1) * ld + r] = s12;
            out[(c + 1) * ld + r + 1] = s22;
            r += 2;
        }
        if r < kx {
            let [s1, s2] = dot2x1(y1, y2, xc(r), len);
            out[c * ld + r] = s1;
            out[(c + 1) * ld + r] = s2;
        }
        c += 2;
    }
    if c < ky {
        let y1 = yc(c);
        let mut r = 0;
        while r + 1 < kx {
            let [s1, s2] = dot2x1(xc(r), xc(r + 1), y1, len);
            out[c * ld + r] = s1;
            out[c * ld + r + 1] = s2;
            r += 2;
        }
        if r < kx {
            out[c * ld + r] = dot(xc(r), y1);
        }
    }
}

/// `gen_f32_sum`: four values summed in single precision, then added in double.
pub fn sum(x: &[f32]) -> f32 {
    let n = x.len();
    let regular = n & !3;
    let mut s = 0f64;
    let mut i = 0;
    while i < regular {
        s += (x[i] + x[i + 1] + x[i + 2] + x[i + 3]) as f64;
        i += 4;
    }
    while i < n {
        s += x[i] as f64;
        i += 1;
    }
    s as f32
}

/// `gen_f32_gemv_Xb`: `y = X b`, `X` `n` rows by `b.len()` columns of
/// leading dimension `ld`, the columns two at a time.
pub fn gemv(x: &[f32], ld: usize, n: usize, b: &[f32], y: &mut [f32]) {
    let k = b.len();
    y[..n].fill(0.0);
    let y = &mut y[..n];
    let mut col = 0;
    while col + 1 < k {
        let (a0, a1) = (b[col], b[col + 1]);
        let x1 = &x[col * ld..col * ld + n];
        let x2 = &x[(col + 1) * ld..(col + 1) * ld + n];
        for ((v, p), q) in y.iter_mut().zip(x1).zip(x2) {
            *v += p * a0 + q * a1;
        }
        col += 2;
    }
    if col < k {
        let c = b[col];
        let x1 = &x[col * ld..col * ld + n];
        for (v, p) in y.iter_mut().zip(x1) {
            *v += p * c;
        }
    }
}

/// `gen_f32_seq`: `x0, x0 + dx, ...` accumulated four at a time.
pub fn seq(p: &mut [f32], mut x0: f32, dx: f32) {
    let n = p.len();
    let regular = n & !3;
    let dx2 = dx + dx;
    let dx3 = dx2 + dx;
    let mut i = 0;
    while i < regular {
        p[i] = x0;
        p[i + 1] = x0 + dx;
        p[i + 2] = x0 + dx2;
        p[i + 3] = x0 + dx3;
        x0 = x0 + dx3 + dx;
        i += 4;
    }
    while i < n {
        p[i] = x0;
        x0 += dx;
        i += 1;
    }
}

/// `gen_f32_avgstd`: the mean and the standard deviation (n - 1), the sums in
/// double of single-precision groups of four.
pub fn avgstd(x: &[f32]) -> (f32, f32) {
    let n = x.len();
    let regular = n & !3;
    let (mut sx, mut sxx) = (0f64, 0f64);
    let mut i = 0;
    while i < regular {
        sx += (x[i] + x[i + 1] + x[i + 2] + x[i + 3]) as f64;
        sxx += (x[i] * x[i] + x[i + 1] * x[i + 1] + x[i + 2] * x[i + 2] + x[i + 3] * x[i + 3]) as f64;
        i += 4;
    }
    while i < n {
        sx += x[i] as f64;
        sxx += (x[i] * x[i]) as f64;
        i += 1;
    }
    let avg = sx / n as f64;
    let std = (((sxx - avg * sx) / (n as f64 - 1.0)) as f32).sqrt();
    (avg as f32, std)
}

/// `gen_f32_scale_inplace`: `x * gain + offset`.
#[inline]
pub fn scale(x: &mut [f32], gain: f32, offset: f32) {
    for v in x {
        *v = *v * gain + offset;
    }
}

/// `f32_normalize_inplace`.
pub fn normalize(x: &mut [f32]) {
    let (avg, std) = avgstd(x);
    let gain = 1.0 / std;
    let offset = -avg / std;
    scale(x, gain, offset);
}

/// `f32_normalize_x_factor_inplace`.
pub fn normalize_factor(x: &mut [f32], factor: f32) {
    let (avg, std) = avgstd(x);
    let gain = factor / std;
    let offset = -factor * avg / std;
    scale(x, gain, offset);
}

/// `chol_addCol_skipleadingzeros_prec(Vec)_invdiag`: columns `k0..=k1`
/// (1-based) of the Cholesky factor of `XtX + diag(prec)`, `k` by `k` with
/// the reciprocals on its diagonal, from the upper triangle of those columns
/// of `xtx` (leading dimension `k`); a column's leading zeros of `xtx` are
/// skipped. `prec` holds one value per column, or one for all.
pub fn chol_add_cols(xtx: &[f32], u: &mut [f32], prec: &[f32], k: usize, k0: usize, k1: usize) {
    for col in k0..=k1 {
        let au = &xtx[(col - 1) * k..(col - 1) * k + col];
        let ucol_start = (col - 1) * k;
        let mut first = 1;
        while au[first - 1] == 0.0 && first < col {
            u[ucol_start + first - 1] = 0.0;
            first += 1;
        }
        let mut big_sum = 0f64;
        for c in first..col {
            let ubase = (c - 1) * k;
            let mut s = 0f64;
            for row in first..c {
                s += (u[ubase + row - 1] * u[ucol_start + row - 1]) as f64;
            }
            let ukk_inv = u[ubase + c - 1] as f64;
            let cur = (au[c - 1] as f64 - s) * ukk_inv;
            u[ucol_start + c - 1] = cur as f32;
            big_sum += cur * cur;
        }
        let p = if prec.len() == 1 { prec[0] } else { prec[col - 1] };
        u[ucol_start + col - 1] = (1.0 / (((au[col - 1] + p) as f64) - big_sum).sqrt()) as f32;
    }
}

/// `solve_U_as_LU_invdiag_sqrmat`: `x` from `U'U x = y`, `U` `k` by `k` with
/// the reciprocals of its diagonal.
pub fn solve_lu_invdiag(u: &[f32], y: &[f32], x: &mut [f32], k: usize) {
    for col in 1..=k {
        let ucol = &u[(col - 1) * k..];
        let mut s = 0f32;
        for row in 1..col {
            s += ucol[row - 1] * x[row - 1];
        }
        x[col - 1] = (y[col - 1] - s) * ucol[col - 1];
    }
    solve_u_invdiag(u, x, k, k);
}

/// `solve_U_as_U_invdiag`: `x` from `U x = b` in place.
pub fn solve_u_invdiag(u: &[f32], x: &mut [f32], ldu: usize, k: usize) {
    for ncol in 1..=k {
        // U starts at the last column's row K - ncol.
        let row = k - ncol;
        let mut s = 0f32;
        let mut col = k;
        while col > k - ncol + 1 {
            s += u[(col - 1) * ldu + row] * x[col - 1];
            col -= 1;
        }
        x[row] = (x[row] - s) * u[(col - 1) * ldu + row];
    }
}

/// `f32_sumfilter`: the sums of `x` over a moving window of `win` values.
pub fn sumfilter(x: &[f32], y: &mut [f32], win: usize) {
    let n = x.len();
    if win <= 1 {
        y[..n].copy_from_slice(x);
        return;
    }
    let w_left = win / 2;
    let w_right = win - w_left - 1;
    let mut right = 0f32;
    let nadj = w_right.min(n - 1);
    for v in &x[1..=nadj] {
        right += *v;
    }
    let mut left_mid = 0f32;
    for i in 0..=w_left {
        left_mid += x[i];
        y[i] = left_mid + right;
        let new_right = if i + w_right + 1 >= n { 0.0 } else { x[i + w_right + 1] };
        let old_right = if i + 1 >= n { 0.0 } else { x[i + 1] };
        right += new_right - old_right;
    }
    let mut all = y[w_left];
    for i in w_left + 1..n.saturating_sub(w_right) {
        all += x[i + w_right] - x[i - 1 - w_left];
        y[i] = all;
    }
    for i in n.saturating_sub(w_right).max(w_left + 1)..n {
        all -= x[i - w_left - 1];
        y[i] = all;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_functions_are_close() {
        for &x in &[0.001f32, 0.3, 1.0, 2.5, 100.0, 12345.0] {
            assert!((fastlog(x) - x.ln()).abs() < 1e-3, "{x}");
            assert!((fastsqrt(x) - x.sqrt()).abs() / x.sqrt() < 1e-3, "{x}");
        }
        for &x in &[-20.0f32, -3.0, -0.5, 0.0, 0.7, 10.0] {
            assert!((fastexp(x) - x.exp()).abs() / x.exp() < 1e-3, "{x}");
        }
    }

    #[test]
    fn cholesky_solves() {
        // A = [[4, 2], [2, 3]] plus prec 1 on the diagonal: [[5, 2], [2, 4]].
        let xtx = [4.0f32, 2.0, 2.0, 3.0];
        let mut u = [0f32; 4];
        chol_add_cols(&xtx, &mut u, &[1.0], 2, 1, 2);
        let mut x = [0f32; 2];
        solve_lu_invdiag(&u, &[1.0, 2.0], &mut x, 2);
        // [[5, 2], [2, 4]] x = [1, 2] -> x = [0, 0.5].
        assert!(x[0].abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn cross_is_each_dot() {
        let x: Vec<f32> = (0..70).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.37).collect();
        let y: Vec<f32> = (0..90).map(|i| ((i * 13 % 7) as f32 - 3.0) * 1.3).collect();
        for (kx, ky, len) in [(3, 5, 13), (4, 4, 8), (1, 1, 3), (5, 3, 9), (2, 7, 11)] {
            let mut out = vec![0f32; ky * 9];
            cross(&x, len, kx, &y, len, ky, len, &mut out, 9);
            for c in 0..ky {
                for r in 0..kx {
                    let d = dot(&x[r * len..(r + 1) * len], &y[c * len..(c + 1) * len]);
                    assert_eq!(out[c * 9 + r].to_bits(), d.to_bits());
                }
            }
        }
    }

    #[test]
    fn sumfilter_window() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y = [0f32; 6];
        sumfilter(&x, &mut y, 3);
        assert_eq!(y, [3.0, 6.0, 9.0, 12.0, 15.0, 11.0]);
    }
}
