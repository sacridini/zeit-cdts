//! The MATLAB primitives of Fmask 4.7, with MATLAB's semantics. Written from the documented behaviour of the
//! functions (Image Processing and Statistics toolboxes; the strel disk after Adams 1993 and Jones and Soille 1996),
//! not from any code; checked against GNU Octave by `benchmarks/vs_originals/fmask4.py`. Used by the port of
//! Fmask 4.7 (https://github.com/GERSL/Fmask, commit e32f5bd, MIT License, Copyright (c) 2018 Global Environmental
//! Remote Sensing Lab).
//!
//! Images are row-major slices of `rows * cols`. Where the order matters, MATLAB's is by columns and it is kept
//! (labels of [`bwlabel`], pixel lists of [`regionprops`]). Everything is linear in the number of pixels (times the
//! number of rows of a structuring element, for [`erode`] and [`dilate`] with an arbitrary one), so that a
//! 5490 x 5490 Sentinel-2 20 m scene takes seconds. The rayon pool is the global one (`RAYON_NUM_THREADS`).

use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

// ---------------------------------------------------------------------------------------------------------------
// round
// ---------------------------------------------------------------------------------------------------------------

/// MATLAB's `round`: halves go away from zero (`round(4.5) = 5`, `round(-4.5) = -5`, `round(0.5) = 1`), unlike
/// the banker's rounding of numpy and Octave's `rint`-like tie handling in some builds. (Rust's `f64::round` is the
/// same rule; this function names it, so that the ports say where MATLAB's `round` is meant.)
#[inline]
pub fn round_half_away(x: f64) -> f64 {
    x.round()
}

// ---------------------------------------------------------------------------------------------------------------
// prctile
// ---------------------------------------------------------------------------------------------------------------

/// MATLAB's `prctile(values, p)` of a vector, for `p` in percent (0..100): the sorted values are the quantiles of
/// the probabilities `(i - 0.5) / n`, `i = 1..n`, between them linear interpolation, and below the first and above
/// the last the minimum and the maximum. So the position of `p` is `p / 100 * n + 0.5` (1-based). NaN is left out,
/// as MATLAB does; no value (or all NaN) gives NaN. The position and the two weights are in double (`p` and `n`
/// are doubles, and in single the position of a 30 million pixel scene has no fraction left), rounded to single
/// for the weights; the products and their sum are in single, the data's class. (Equal to GNU Octave's single
/// result in every value tested, the other orders of the operations are not.) This is NOT the type 7 of R and numpy (`aoa::quantile7`).
pub fn prctile(values: &[f32], p: f64) -> f32 {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 || p.is_nan() {
        return f32::NAN;
    }
    let pos = p / 100.0 * (n as f64) + 0.5;
    if !(pos > 1.0) {
        return v.iter().copied().fold(f32::INFINITY, f32::min);
    }
    if pos >= n as f64 {
        return v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    }
    let k = pos.floor();
    let frac = pos - k;
    let lo = k as usize - 1; // 0-based index of the value below; lo + 1 < n
    v.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let a = v[lo];
    if frac == 0.0 {
        return a;
    }
    let b = v[lo + 1..].iter().copied().fold(f32::INFINITY, f32::min);
    ((1.0 - frac) as f32) * a + (frac as f32) * b
}

/// [`prctile`] of double data (the same positions and weights, everything in double).
pub fn prctile_f64(values: &[f64], p: f64) -> f64 {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    let n = v.len();
    if n == 0 || p.is_nan() {
        return f64::NAN;
    }
    let pos = p / 100.0 * (n as f64) + 0.5;
    if !(pos > 1.0) {
        return v.iter().copied().fold(f64::INFINITY, f64::min);
    }
    if pos >= n as f64 {
        return v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    let k = pos.floor();
    let frac = pos - k;
    let lo = k as usize - 1;
    v.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let a = v[lo];
    if frac == 0.0 {
        return a;
    }
    let b = v[lo + 1..].iter().copied().fold(f64::INFINITY, f64::min);
    (1.0 - frac) * a + frac * b
}

// ---------------------------------------------------------------------------------------------------------------
// imresize
// ---------------------------------------------------------------------------------------------------------------

/// `imresize(img, 0.5, 'box')`, the 10 m to 20 m of Sentinel-2: the mean of every 2 x 2 block, in single (the
/// means of the pairs of rows, then of the pairs of columns: each is the exact half of a sum), a NaN making the
/// block NaN. `rows` and `cols` must be even. Returns the `rows / 2 x cols / 2` image.
pub fn resize_box_half(img: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert!(rows % 2 == 0 && cols % 2 == 0 && img.len() == rows * cols, "resize_box_half: even rows and cols");
    let (r2, c2) = (rows / 2, cols / 2);
    let mut out = vec![0f32; r2 * c2];
    out.par_chunks_mut(c2.max(1)).enumerate().for_each(|(i, o)| {
        let (a, b) = (&img[2 * i * cols..(2 * i + 1) * cols], &img[(2 * i + 1) * cols..(2 * i + 2) * cols]);
        for j in 0..c2 {
            let p = (a[2 * j] + b[2 * j]) * 0.5f32;
            let q = (a[2 * j + 1] + b[2 * j + 1]) * 0.5f32;
            o[j] = (p + q) * 0.5f32;
        }
    });
    out
}

/// `imresize(img, 3, 'nearest')`, the 60 m to 20 m of Sentinel-2: every pixel repeated 3 x 3. Returns the
/// `3 rows x 3 cols` image.
pub fn resize_replicate3<T: Copy + Send + Sync>(img: &[T], rows: usize, cols: usize) -> Vec<T> {
    assert_eq!(img.len(), rows * cols);
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let c3 = cols * 3;
    let mut out = Vec::with_capacity(rows * 3 * c3);
    for i in 0..rows {
        let start = out.len();
        for j in 0..cols {
            let v = img[i * cols + j];
            out.extend_from_slice(&[v, v, v]);
        }
        let row: Vec<T> = out[start..].to_vec();
        out.extend_from_slice(&row);
        out.extend_from_slice(&row);
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// stdfilt
// ---------------------------------------------------------------------------------------------------------------

/// Neumaier's compensated addition (the sliding sums run for thousands of steps).
#[inline(always)]
fn nadd(s: &mut f64, c: &mut f64, v: f64) {
    let t = *s + v;
    if s.abs() >= v.abs() {
        *c += (*s - t) + v;
    } else {
        *c += (v - t) + *s;
    }
    *s = t;
}

/// Symmetric padding: the index `i` (any integer) mapped into `0..n` by reflecting at the edges, the edge pixel
/// repeated (`abcd` is padded as `dcba|abcd|dcba`), periodically if the padding is longer than the image.
#[inline(always)]
fn mirror(i: i64, n: i64) -> usize {
    if i >= 0 && i < n {
        return i as usize;
    }
    let m = i.rem_euclid(2 * n);
    (if m >= n { 2 * n - 1 - m } else { m }) as usize
}

/// MATLAB's `stdfilt(img, true(k))`, `k` odd: the standard deviation of the `k x k` neighbourhood of every pixel,
/// with symmetric padding, `n = k * k` and the divisor `n - 1`, in double:
/// `sqrt(max(sum(x^2) / (n - 1) - sum(x)^2 / (n * (n - 1)), 0))` (the E[x^2] - E[x]^2 form, not Welford's, so that
/// a flat window gives 0 or a few 1e-4 of rounding noise, as MATLAB's convolutions do). The sums slide (compensated
/// sums, linear in the pixels, whatever `k`); a window with a NaN or an infinity gives NaN (an infinity would
/// poison the sliding sums for the rest of the line).
pub fn stdfilt(img: &[f32], rows: usize, cols: usize, k: usize) -> Vec<f64> {
    assert!(k % 2 == 1 && img.len() == rows * cols, "stdfilt: odd k");
    let mut out = vec![0f64; rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    let h = (k / 2) as i64;
    let (ni, nj) = (rows as i64, cols as i64);
    let nn = (k * k) as f64;
    // pass 1: the sums of x and x^2 and the NaN count along the rows
    let mut s1 = vec![0f64; rows * cols];
    let mut s2 = vec![0f64; rows * cols];
    let mut cn = vec![0u16; rows * cols];
    s1.par_chunks_mut(cols)
        .zip(s2.par_chunks_mut(cols))
        .zip(cn.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(i, ((a1, a2), ac))| {
            let row = &img[i * cols..(i + 1) * cols];
            let (mut sa, mut ca, mut sb, mut cb, mut c) = (0f64, 0f64, 0f64, 0f64, 0i32);
            let put = |sa: &mut f64, ca: &mut f64, sb: &mut f64, cb: &mut f64, c: &mut i32, j: i64, sign: f64| {
                let x = row[mirror(j, nj)];
                if !x.is_finite() {
                    *c += sign as i32;
                } else {
                    let x = x as f64;
                    nadd(sa, ca, sign * x);
                    nadd(sb, cb, sign * (x * x));
                }
            };
            for j in -h..=h {
                put(&mut sa, &mut ca, &mut sb, &mut cb, &mut c, j, 1.0);
            }
            for j in 0..cols {
                a1[j] = sa + ca;
                a2[j] = sb + cb;
                ac[j] = c as u16;
                let jj = j as i64;
                put(&mut sa, &mut ca, &mut sb, &mut cb, &mut c, jj + h + 1, 1.0);
                put(&mut sa, &mut ca, &mut sb, &mut cb, &mut c, jj - h, -1.0);
            }
        });
    // pass 2: along the columns, in bands of columns (cache friendly, in parallel)
    const BAND: usize = 256;
    let nb = cols.div_ceil(BAND);
    let bands: Vec<Vec<f64>> = (0..nb)
        .into_par_iter()
        .map(|b| {
            let (x0, x1) = (b * BAND, ((b + 1) * BAND).min(cols));
            let w = x1 - x0;
            let mut o = vec![0f64; rows * w];
            let (mut sa, mut ca) = (vec![0f64; w], vec![0f64; w]);
            let (mut sb, mut cb) = (vec![0f64; w], vec![0f64; w]);
            let mut c = vec![0i32; w];
            let add = |i: i64, sign: f64, sa: &mut [f64], ca: &mut [f64], sb: &mut [f64], cb: &mut [f64], c: &mut [i32]| {
                let r = mirror(i, ni) * cols + x0;
                for x in 0..w {
                    nadd(&mut sa[x], &mut ca[x], sign * s1[r + x]);
                    nadd(&mut sb[x], &mut cb[x], sign * s2[r + x]);
                    c[x] += sign as i32 * cn[r + x] as i32;
                }
            };
            for i in -h..=h {
                add(i, 1.0, &mut sa, &mut ca, &mut sb, &mut cb, &mut c);
            }
            for i in 0..rows {
                for x in 0..w {
                    o[i * w + x] = if c[x] > 0 {
                        f64::NAN
                    } else {
                        let (t1, t2) = (sa[x] + ca[x], sb[x] + cb[x]);
                        (t2 / (nn - 1.0) - (t1 * t1) / (nn * (nn - 1.0))).max(0.0).sqrt()
                    };
                }
                let ii = i as i64;
                add(ii + h + 1, 1.0, &mut sa, &mut ca, &mut sb, &mut cb, &mut c);
                add(ii - h, -1.0, &mut sa, &mut ca, &mut sb, &mut cb, &mut c);
            }
            o
        })
        .collect();
    for (b, o) in bands.iter().enumerate() {
        let (x0, x1) = (b * BAND, ((b + 1) * BAND).min(cols));
        let w = x1 - x0;
        for i in 0..rows {
            out[i * cols + x0..i * cols + x1].copy_from_slice(&o[i * w..(i + 1) * w]);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// strel
// ---------------------------------------------------------------------------------------------------------------

/// A flat structuring element (`strel`): an odd-sized mask in row-major order, the origin at its centre.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strel {
    pub rows: usize,
    pub cols: usize,
    pub mask: Vec<bool>,
}

impl Strel {
    pub fn new(rows: usize, cols: usize, mask: Vec<bool>) -> Strel {
        assert!(rows % 2 == 1 && cols % 2 == 1 && mask.len() == rows * cols, "Strel: odd sizes");
        Strel { rows, cols, mask }
    }

    /// `strel('square', w)` with `w` odd.
    pub fn square(w: usize) -> Strel {
        Strel::new(w, w, vec![true; w * w])
    }

    /// Number of neighbours (true cells).
    pub fn count(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }

    /// The offsets `(dy, dx)` of the true cells from the origin, in row-major order.
    pub fn offsets(&self) -> Vec<(i32, i32)> {
        let (hr, hc) = ((self.rows / 2) as i32, (self.cols / 2) as i32);
        let mut v = Vec::new();
        for y in 0..self.rows {
            for x in 0..self.cols {
                if self.mask[y * self.cols + x] {
                    v.push((y as i32 - hr, x as i32 - hc));
                }
            }
        }
        v
    }
}

/// The length of the horizontal and vertical lines that `strel('disk', r)` adds to the periodic lines, which reach
/// `ext` pixels from the origin in x: the line of `len` pixels reaches `(len - 1) / 2`, and the disk should reach
/// `r` in all, so `len = 2 * (r - ext) - 1`.
///
/// UNCONFIRMED: this is the one point of the disk that no public source I checked states for every radius. It gives
/// the published shapes for r = 5 (9 x 9, 69 neighbours) and r = 15 (29 x 29); for r = 10 it gives 19 x 19 with
/// 321 neighbours, not confirmed. If a reference disagrees, change this function only.
fn disk_line_len(r: i32, ext: i32) -> i32 {
    2 * (r - ext) - 1
}

/// `strel('disk', r)` (N = 4, MATLAB's default): for `r < 3` the exact disk `x^2 + y^2 <= r^2`; from 3 on the
/// approximation by periodic lines (Adams 1993; Jones and Soille 1996): the Minkowski sum of the four periodic
/// lines of the directions (1,0), (1,1), (0,1) and (-1,1), each `i * v` for `i` in `-rp..=rp` with
/// `rp = floor(k / |v|)` and `k = 2 r / (cot(pi/8) + 1 / sin(pi/8))`, and then, if `len >= 3`, of a horizontal and a
/// vertical line of [`disk_line_len`] pixels. The result is the whole element (the one MATLAB decomposes into
/// the sequence of its parts), r = 5 being 9 x 9 and r = 15 29 x 29.
pub fn strel_disk(r: usize) -> Strel {
    if r < 3 {
        let w = 2 * r + 1;
        let ri = r as i32;
        let mut m = vec![false; w * w];
        for y in 0..w {
            for x in 0..w {
                let (dy, dx) = (y as i32 - ri, x as i32 - ri);
                m[y * w + x] = dy * dy + dx * dx <= ri * ri;
            }
        }
        return Strel::new(w, w, m);
    }
    let ri = r as i32;
    let theta = std::f64::consts::PI / 8.0;
    let k = 2.0 * r as f64 / (1.0 / theta.tan() + 1.0 / theta.sin());
    let vs: [(i32, i32); 4] = [(1, 0), (1, 1), (0, 1), (-1, 1)];
    let rps: Vec<i32> = vs.iter().map(|&(a, b)| (k / ((a * a + b * b) as f64).sqrt()).floor() as i32).collect();
    // the grid: half size m, large enough for all the lines and the final lines (at most r - 1)
    let ext_max: i32 = vs.iter().zip(&rps).map(|(&(a, _), &p)| a.abs() * p).sum();
    let m = ext_max + ri;
    let w = (2 * m + 1) as usize;
    let mut grid = vec![false; w * w];
    grid[(m as usize) * w + m as usize] = true;
    // grid[y][x] with (y, x) from -m..=m; the Minkowski sum of the set with the points `pts`
    let sum = |grid: &Vec<bool>, pts: &[(i32, i32)]| -> Vec<bool> {
        let mut g = vec![false; w * w];
        for y in 0..w as i32 {
            for x in 0..w as i32 {
                if grid[(y as usize) * w + x as usize] {
                    for &(dy, dx) in pts {
                        let (ny, nx) = (y + dy, x + dx);
                        if ny >= 0 && ny < w as i32 && nx >= 0 && nx < w as i32 {
                            g[(ny as usize) * w + nx as usize] = true;
                        }
                    }
                }
            }
        }
        g
    };
    for (&(vx, vy), &rp) in vs.iter().zip(&rps) {
        let pts: Vec<(i32, i32)> = (-rp..=rp).map(|i| (i * vy, i * vx)).collect();
        grid = sum(&grid, &pts);
    }
    let mut ext = 0;
    for y in 0..w {
        for x in 0..w {
            if grid[y * w + x] {
                ext = ext.max((x as i32 - m).abs());
            }
        }
    }
    let len = disk_line_len(ri, ext);
    if len >= 3 {
        let hl = (len - 1) / 2;
        let hor: Vec<(i32, i32)> = (-hl..=hl).map(|i| (0, i)).collect();
        let ver: Vec<(i32, i32)> = (-hl..=hl).map(|i| (i, 0)).collect();
        grid = sum(&grid, &hor);
        grid = sum(&grid, &ver);
    }
    // crop to the bounding box (symmetric)
    let mut half = 0;
    for y in 0..w {
        for x in 0..w {
            if grid[y * w + x] {
                half = half.max((y as i32 - m).abs()).max((x as i32 - m).abs());
            }
        }
    }
    let sz = (2 * half + 1) as usize;
    let mut mask = vec![false; sz * sz];
    for y in 0..sz {
        for x in 0..sz {
            mask[y * sz + x] = grid[((y as i32 - half + m) as usize) * w + (x as i32 - half + m) as usize];
        }
    }
    Strel::new(sz, sz, mask)
}

// ---------------------------------------------------------------------------------------------------------------
// imerode, imdilate
// ---------------------------------------------------------------------------------------------------------------

#[inline(always)]
fn pick<T: Copy + PartialOrd>(erode: bool, a: T, b: T) -> T {
    if erode {
        if b < a {
            b
        } else {
            a
        }
    } else if b > a {
        b
    } else {
        a
    }
}

/// The min (erosion) or max (dilation) of each window `[x + a, x + b]` of every row (van Herk / Gil-Werman: three
/// operations per pixel whatever the width), outside the row `ident`.
fn row_filter<T: Copy + PartialOrd + Send + Sync>(img: &[T], rows: usize, cols: usize, a: i32, b: i32, erode: bool, ident: T) -> Vec<T> {
    let mut out = img.to_vec();
    if rows == 0 || cols == 0 {
        return out;
    }
    let w = (b - a + 1) as usize;
    let l = cols + w - 1;
    out.par_chunks_mut(cols).enumerate().for_each_init(
        || (vec![ident; l], vec![ident; l], vec![ident; l]),
        |(buf, g, hh), (i, o)| {
            let row = &img[i * cols..(i + 1) * cols];
            for t in 0..l {
                let j = t as i64 + a as i64;
                buf[t] = if j >= 0 && j < cols as i64 { row[j as usize] } else { ident };
            }
            for t in 0..l {
                g[t] = if t % w == 0 { buf[t] } else { pick(erode, g[t - 1], buf[t]) };
            }
            for t in (0..l).rev() {
                hh[t] = if t % w == w - 1 || t == l - 1 { buf[t] } else { pick(erode, hh[t + 1], buf[t]) };
            }
            for x in 0..cols {
                o[x] = pick(erode, hh[x], g[x + w - 1]);
            }
        },
    );
    out
}

fn transpose<T: Copy + Send + Sync>(img: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(rows * cols);
    if rows == 0 || cols == 0 {
        return out;
    }
    let fill = img[0];
    out.resize(rows * cols, fill);
    const B: usize = 32;
    // out is cols x rows: out[c][r] = img[r][c]; parallel over the rows of out in blocks
    out.par_chunks_mut(rows * B).enumerate().for_each(|(cb, chunk)| {
        let c0 = cb * B;
        let nc = chunk.len() / rows;
        for r0 in (0..rows).step_by(B) {
            for r in r0..(r0 + B).min(rows) {
                for dc in 0..nc {
                    chunk[dc * rows + r] = img[r * cols + c0 + dc];
                }
            }
        }
    });
    out
}

/// A rectangle `(2 hy + 1) x (2 hx + 1)` as two passes (rows, then columns), the outside `ident`.
fn morph_rect<T: Copy + PartialOrd + Send + Sync>(img: &[T], rows: usize, cols: usize, hy: i32, hx: i32, erode: bool, ident: T) -> Vec<T> {
    let m = row_filter(img, rows, cols, -hx, hx, erode, ident);
    if hy == 0 {
        return m;
    }
    let t = transpose(&m, rows, cols);
    let t = row_filter(&t, cols, rows, -hy, hy, erode, ident);
    transpose(&t, cols, rows)
}

/// The element as the groups of rows that share a run `[a, b]` of x offsets: `((a, b), [dy..])`. For the erosion
/// the offsets are the element's; for the dilation they are reflected (`imdilate` reflects the element about its
/// origin, the Minkowski sum).
fn runs(se: &Strel, erode: bool) -> Vec<((i32, i32), Vec<i32>)> {
    let (hr, hc) = ((se.rows / 2) as i32, (se.cols / 2) as i32);
    let mut groups: Vec<((i32, i32), Vec<i32>)> = Vec::new();
    for y in 0..se.rows {
        let mut x = 0;
        while x < se.cols {
            if !se.mask[y * se.cols + x] {
                x += 1;
                continue;
            }
            let x0 = x;
            while x < se.cols && se.mask[y * se.cols + x] {
                x += 1;
            }
            let (mut a, mut b, mut dy) = (x0 as i32 - hc, x as i32 - 1 - hc, y as i32 - hr);
            if !erode {
                (a, b, dy) = (-b, -a, -dy);
            }
            match groups.iter_mut().find(|g| g.0 == (a, b)) {
                Some(g) => g.1.push(dy),
                None => groups.push(((a, b), vec![dy])),
            }
        }
    }
    groups
}

fn morph<T: Copy + PartialOrd + Send + Sync>(img: &[T], rows: usize, cols: usize, se: &Strel, erode: bool, ident: T) -> Vec<T> {
    assert_eq!(img.len(), rows * cols);
    // a full rectangle goes separably
    if se.mask.iter().all(|&b| b) {
        return morph_rect(img, rows, cols, (se.rows / 2) as i32, (se.cols / 2) as i32, erode, ident);
    }
    let mut out = vec![ident; rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    for ((a, b), dys) in runs(se, erode) {
        let m = row_filter(img, rows, cols, a, b, erode, ident);
        for dy in dys {
            out.par_chunks_mut(cols).enumerate().for_each(|(y, o)| {
                let y2 = y as i64 + dy as i64;
                if y2 >= 0 && y2 < rows as i64 {
                    let r = &m[y2 as usize * cols..(y2 as usize + 1) * cols];
                    for x in 0..cols {
                        o[x] = pick(erode, o[x], r[x]);
                    }
                }
            });
        }
    }
    out
}

/// `imerode` of a binary image (0 or 1) by the flat element `se`: the pixel is 1 if the element placed on it
/// lies in 1s; outside the image counts as 1 (`+inf`), and the result is the element's as a whole (not
/// decomposed, whose borders would differ).
pub fn erode_bin(img: &[u8], rows: usize, cols: usize, se: &Strel) -> Vec<u8> {
    morph(img, rows, cols, se, true, 1u8)
}

/// `imdilate` of a binary image by `se` (reflected about its origin); outside the image counts as 0 (`-inf`).
pub fn dilate_bin(img: &[u8], rows: usize, cols: usize, se: &Strel) -> Vec<u8> {
    morph(img, rows, cols, se, false, 0u8)
}

/// `imerode` of a grayscale image: the minimum under the element; outside the image `+inf`. NaN is not supported.
pub fn erode_gray(img: &[f32], rows: usize, cols: usize, se: &Strel) -> Vec<f32> {
    morph(img, rows, cols, se, true, f32::INFINITY)
}

/// `imdilate` of a grayscale image: the maximum under the element (reflected); outside `-inf`. NaN is not supported.
pub fn dilate_gray(img: &[f32], rows: usize, cols: usize, se: &Strel) -> Vec<f32> {
    morph(img, rows, cols, se, false, f32::NEG_INFINITY)
}

/// `imerode(img, strel('square', 2 hw + 1))` of a binary image, in two passes (linear in the pixels whatever the
/// width). `w` is the odd side.
pub fn erode_bin_square(img: &[u8], rows: usize, cols: usize, w: usize) -> Vec<u8> {
    assert!(w % 2 == 1);
    morph_rect(img, rows, cols, (w / 2) as i32, (w / 2) as i32, true, 1u8)
}

/// `imdilate(img, strel('square', w))` of a binary image, `w` odd.
pub fn dilate_bin_square(img: &[u8], rows: usize, cols: usize, w: usize) -> Vec<u8> {
    assert!(w % 2 == 1);
    morph_rect(img, rows, cols, (w / 2) as i32, (w / 2) as i32, false, 0u8)
}

/// `imerode` of a grayscale image by a square of odd side `w`.
pub fn erode_gray_square(img: &[f32], rows: usize, cols: usize, w: usize) -> Vec<f32> {
    assert!(w % 2 == 1);
    morph_rect(img, rows, cols, (w / 2) as i32, (w / 2) as i32, true, f32::INFINITY)
}

/// `imdilate` of a grayscale image by a square of odd side `w`.
pub fn dilate_gray_square(img: &[f32], rows: usize, cols: usize, w: usize) -> Vec<f32> {
    assert!(w % 2 == 1);
    morph_rect(img, rows, cols, (w / 2) as i32, (w / 2) as i32, false, f32::NEG_INFINITY)
}

// ---------------------------------------------------------------------------------------------------------------
// bwlabel, bwareaopen, regionprops
// ---------------------------------------------------------------------------------------------------------------

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// `bwlabel(img, 8)` / `bwlabeln(img, 8)`: the 8-connected components of the nonzero pixels, numbered from 1 in the
/// order of their first pixel in a scan by COLUMNS (MATLAB's linear order), 0 for the background. Returns the
/// labels (row-major, like the image) and the number of components.
pub fn bwlabel(img: &[u8], rows: usize, cols: usize) -> (Vec<i32>, usize) {
    assert_eq!(img.len(), rows * cols);
    let mut out = vec![0i32; rows * cols];
    if rows == 0 || cols == 0 {
        return (out, 0);
    }
    // 8-connectivity is symmetric under the transpose: scan the transposed image by rows
    let t = transpose(img, rows, cols); // cols x rows
    let (tr, tc) = (cols, rows);
    let mut lbl = vec![0u32; tr * tc];
    let mut parent: Vec<u32> = vec![0];
    for y in 0..tr {
        for x in 0..tc {
            let p = y * tc + x;
            if t[p] == 0 {
                continue;
            }
            let mut cur = 0u32;
            let mut join = |l: u32, parent: &mut Vec<u32>| {
                if l == 0 {
                    return;
                }
                let r = find(parent, l);
                if cur == 0 {
                    cur = r;
                } else if r != cur {
                    let (lo, hi) = (cur.min(r), cur.max(r));
                    parent[hi as usize] = lo;
                    cur = lo;
                }
            };
            if x > 0 {
                join(lbl[p - 1], &mut parent);
            }
            if y > 0 {
                if x > 0 {
                    join(lbl[p - tc - 1], &mut parent);
                }
                join(lbl[p - tc], &mut parent);
                if x + 1 < tc {
                    join(lbl[p - tc + 1], &mut parent);
                }
            }
            if cur == 0 {
                cur = parent.len() as u32;
                parent.push(cur);
            }
            lbl[p] = cur;
        }
    }
    let mut ids = vec![0u32; parent.len()];
    let mut n = 0u32;
    for y in 0..tr {
        for x in 0..tc {
            let p = y * tc + x;
            if lbl[p] == 0 {
                continue;
            }
            let r = find(&mut parent, lbl[p]) as usize;
            if ids[r] == 0 {
                n += 1;
                ids[r] = n;
            }
            out[x * cols + y] = ids[r] as i32;
        }
    }
    (out, n as usize)
}

/// `bwareaopen(img, min, 8)`: removes the 8-connected components of fewer than `min` pixels.
pub fn bwareaopen(img: &[u8], rows: usize, cols: usize, min: usize) -> Vec<u8> {
    let (lab, n) = bwlabel(img, rows, cols);
    let mut area = vec![0usize; n + 1];
    lab.iter().for_each(|&l| area[l as usize] += 1);
    lab.iter().map(|&l| (l > 0 && area[l as usize] >= min) as u8).collect()
}

/// One component of [`regionprops`].
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// `Area`: the number of pixels.
    pub area: usize,
    /// `Centroid`: the mean `(x, y)` = (column, row), 1-based as MATLAB's.
    pub centroid: (f64, f64),
    /// `PixelList`: the `(x, y)` = (column, row), 1-based as MATLAB's, in column-major order (linear index order).
    pub pixel_list: Vec<(u32, u32)>,
}

/// `regionprops(labels, 'Area', 'Centroid', 'PixelList')` of a label image (as [`bwlabel`] returns, `n`
/// components): one [`Region`] per label `1..=n`.
pub fn regionprops(labels: &[i32], rows: usize, cols: usize, n: usize) -> Vec<Region> {
    assert_eq!(labels.len(), rows * cols);
    let mut regs: Vec<Region> = (0..n).map(|_| Region { area: 0, centroid: (0.0, 0.0), pixel_list: Vec::new() }).collect();
    for l in labels.iter().filter(|&&l| l > 0) {
        regs[*l as usize - 1].area += 1;
    }
    for r in regs.iter_mut() {
        r.pixel_list.reserve_exact(r.area);
    }
    for x in 0..cols {
        for y in 0..rows {
            let l = labels[y * cols + x];
            if l > 0 {
                regs[l as usize - 1].pixel_list.push((x as u32 + 1, y as u32 + 1));
            }
        }
    }
    for r in regs.iter_mut() {
        let (mut sx, mut sy) = (0f64, 0f64);
        for &(x, y) in &r.pixel_list {
            sx += x as f64;
            sy += y as f64;
        }
        r.centroid = (sx / r.area as f64, sy / r.area as f64);
    }
    regs
}

// ---------------------------------------------------------------------------------------------------------------
// imfill
// ---------------------------------------------------------------------------------------------------------------

/// The key of an f32 in the total order of its values, as an unsigned integer.
#[inline]
fn order_key(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

#[inline]
fn key_value(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 { k & 0x7FFF_FFFF } else { !k })
}

/// `imfill(img)` of a grayscale image (4-connected, MATLAB's default for 2-D): fills the regional minima that are
/// not connected to the border, i.e. the morphological reconstruction by erosion of the image from a marker that is
/// the image on the border and its maximum elsewhere. Each pixel gets the lowest level to which water can rise
/// before spilling over the border by a 4-connected path: unique, so the flood from the border by a priority queue
/// (with a plain stack for the pixels not above the level being flooded) gives it. NaN is not supported.
pub fn imfill_gray(img: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(img.len(), rows * cols);
    let mut out = img.to_vec();
    if rows < 3 || cols < 3 {
        return out;
    }
    let mut seen = vec![false; rows * cols];
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Reverse<u64>>, p: usize, v: f32| heap.push(Reverse(((order_key(v) as u64) << 32) | p as u64));
    for y in 0..rows {
        for x in 0..cols {
            if y == 0 || y == rows - 1 || x == 0 || x == cols - 1 {
                let p = y * cols + x;
                seen[p] = true;
                push(&mut heap, p, img[p]);
            }
        }
    }
    let mut stack: Vec<usize> = Vec::new();
    while let Some(Reverse(top)) = heap.pop() {
        let level = key_value((top >> 32) as u32);
        stack.push((top & 0xFFFF_FFFF) as usize);
        while let Some(p) = stack.pop() {
            let (y, x) = (p / cols, p % cols);
            let mut visit = |q: usize, stack: &mut Vec<usize>, heap: &mut BinaryHeap<Reverse<u64>>| {
                if seen[q] {
                    return;
                }
                seen[q] = true;
                if img[q] <= level {
                    out[q] = level;
                    stack.push(q);
                } else {
                    push(heap, q, img[q]);
                }
            };
            if y > 0 {
                visit(p - cols, &mut stack, &mut heap);
            }
            if y + 1 < rows {
                visit(p + cols, &mut stack, &mut heap);
            }
            if x > 0 {
                visit(p - 1, &mut stack, &mut heap);
            }
            if x + 1 < cols {
                visit(p + 1, &mut stack, &mut heap);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------
// imfilter
// ---------------------------------------------------------------------------------------------------------------

/// `imfilter(img, t)` with a 3 x 3 kernel `t` (row-major), single: the default correlation (the kernel is not
/// flipped) with zero padding and the output the size of the input. Each pixel is the sum of the nine products
/// in the order of the kernel's rows, accumulated in single. A NaN spreads to its 3 x 3 neighbourhood.
pub fn imfilter3(img: &[f32], rows: usize, cols: usize, t: &[f32; 9]) -> Vec<f32> {
    assert_eq!(img.len(), rows * cols);
    let mut out = vec![0f32; rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    out.par_chunks_mut(cols).enumerate().for_each(|(y, o)| {
        for x in 0..cols {
            let mut s = 0f32;
            for dy in 0..3usize {
                let yy = y as i64 + dy as i64 - 1;
                if yy < 0 || yy >= rows as i64 {
                    continue;
                }
                for dx in 0..3usize {
                    let xx = x as i64 + dx as i64 - 1;
                    if xx < 0 || xx >= cols as i64 {
                        continue;
                    }
                    s += t[dy * 3 + dx] * img[yy as usize * cols + xx as usize];
                }
            }
            o[x] = s;
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny xorshift generator for the tests' own data.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn round_ties_away() {
        assert_eq!(round_half_away(4.5), 5.0);
        assert_eq!(round_half_away(-4.5), -5.0);
        assert_eq!(round_half_away(2.5), 3.0);
        assert_eq!(round_half_away(0.5), 1.0);
        assert_eq!(round_half_away(-0.4), 0.0);
    }

    #[test]
    fn prctile_positions() {
        let v = [5.0f32, 1.0, 4.0, 2.0, 3.0];
        // position p/100*5+0.5
        assert_eq!(prctile(&v, 50.0), 3.0);
        assert_eq!(prctile(&v, 10.0), 1.0);
        assert_eq!(prctile(&v, 0.0), 1.0);
        assert_eq!(prctile(&v, 100.0), 5.0);
        assert_eq!(prctile(&v, 30.0), 2.0);
        assert!((prctile(&v, 25.0) - 1.75).abs() < 1e-6); // 0.25*5+0.5 = 1.75
        let w = [f32::NAN, 2.0, 1.0, f32::NAN];
        assert_eq!(prctile(&w, 50.0), 1.5);
        assert!(prctile(&[f32::NAN], 50.0).is_nan());
        assert_eq!(prctile(&[7.0], 99.0), 7.0);
    }

    #[test]
    fn resize() {
        let a: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        assert_eq!(resize_box_half(&a, 4, 4), vec![3.5, 5.5, 11.5, 13.5]);
        let mut b = a.clone();
        b[5] = f32::NAN;
        let r = resize_box_half(&b, 4, 4);
        assert!(r[0].is_nan() && r[1] == 5.5);
        let r3 = resize_replicate3(&[1, 2, 3, 4], 2, 2);
        assert_eq!(r3.len(), 36);
        assert_eq!(&r3[0..6], &[1, 1, 1, 2, 2, 2]);
        assert_eq!(&r3[12..18], &[1, 1, 1, 2, 2, 2]);
        assert_eq!(&r3[18..24], &[3, 3, 3, 4, 4, 4]);
    }

    fn brute_std(img: &[f32], rows: usize, cols: usize, k: usize) -> Vec<f64> {
        let h = (k / 2) as i64;
        let n = (k * k) as f64;
        let mut out = vec![0.0; rows * cols];
        for i in 0..rows as i64 {
            for j in 0..cols as i64 {
                let (mut s1, mut s2) = (0f64, 0f64);
                for a in -h..=h {
                    for b in -h..=h {
                        let x = img[mirror(i + a, rows as i64) * cols + mirror(j + b, cols as i64)] as f64;
                        s1 += x;
                        s2 += x * x;
                    }
                }
                out[(i as usize) * cols + j as usize] = (s2 / (n - 1.0) - s1 * s1 / (n * (n - 1.0))).max(0.0).sqrt();
            }
        }
        out
    }

    #[test]
    fn stdfilt_against_brute_force() {
        let mut r = Rng(88172645463325252);
        let (rows, cols) = (13, 17);
        let img: Vec<f32> = (0..rows * cols).map(|_| (r.unit() * 100.0) as f32).collect();
        for k in [1usize, 3, 5, 7, 31] {
            let a = stdfilt(&img, rows, cols, k);
            let b = brute_std(&img, rows, cols, k);
            for (x, y) in a.iter().zip(&b) {
                if k == 1 {
                    assert!(x.is_nan() || *x == 0.0 || y.is_nan());
                } else {
                    assert!((x - y).abs() < 1e-9 * (1.0 + y.abs()), "k={k}: {x} vs {y}");
                }
            }
        }
        let mut nanimg = img.clone();
        nanimg[5 * cols + 5] = f32::NAN;
        let s = stdfilt(&nanimg, rows, cols, 3);
        assert!(s[5 * cols + 5].is_nan() && s[4 * cols + 6].is_nan() && !s[7 * cols + 7].is_nan());
        // a constant window
        assert_eq!(stdfilt(&vec![3.0f32; 25], 5, 5, 3), vec![0.0; 25]);
    }

    #[test]
    fn disk_shapes() {
        // the exact disks
        assert_eq!(strel_disk(1).count(), 5);
        assert_eq!(strel_disk(2).count(), 13);
        // r = 3: k = 6/5.0273 = 1.193, rp = 1 (1,0), 0 (1,1), 1 (0,1), 0 (-1,1); ext = 1; len = 2 * 2 - 1 = 3
        let d3 = strel_disk(3);
        assert_eq!((d3.rows, d3.cols), (5, 5));
        // r = 5: 9 x 9, 69 neighbours (decomposition of 18 pixels)
        let d5 = strel_disk(5);
        assert_eq!((d5.rows, d5.cols, d5.count()), (9, 9, 69));
        assert_eq!(strel_disk(6).rows % 2, 1);
        // r = 10: 19 x 19, 321 neighbours (UNCONFIRMED, see disk_line_len)
        let d10 = strel_disk(10);
        assert_eq!((d10.rows, d10.cols, d10.count()), (19, 19, 321));
        // r = 15: 29 x 29
        let d15 = strel_disk(15);
        assert_eq!((d15.rows, d15.cols), (29, 29));
        // symmetric under flips and the transpose
        for d in [&d3, &d5, &d10, &d15, &strel_disk(6)] {
            let n = d.rows;
            for y in 0..n {
                for x in 0..n {
                    let v = d.mask[y * n + x];
                    assert_eq!(v, d.mask[(n - 1 - y) * n + x]);
                    assert_eq!(v, d.mask[y * n + n - 1 - x]);
                    assert_eq!(v, d.mask[x * n + y]);
                }
            }
        }
    }

    fn brute_morph<T: Copy + PartialOrd>(img: &[T], rows: usize, cols: usize, se: &Strel, erode: bool, ident: T) -> Vec<T> {
        let offs = se.offsets();
        let mut out = vec![ident; rows * cols];
        for y in 0..rows as i32 {
            for x in 0..cols as i32 {
                let mut acc = ident;
                for &(dy, dx) in &offs {
                    let (dy, dx) = if erode { (dy, dx) } else { (-dy, -dx) };
                    let (yy, xx) = (y + dy, x + dx);
                    if yy >= 0 && yy < rows as i32 && xx >= 0 && xx < cols as i32 {
                        acc = pick(erode, acc, img[yy as usize * cols + xx as usize]);
                    }
                }
                out[y as usize * cols + x as usize] = acc;
            }
        }
        out
    }

    #[test]
    fn morphology_against_brute_force() {
        let mut r = Rng(0x9E3779B97F4A7C15);
        let (rows, cols) = (23, 31);
        let bin: Vec<u8> = (0..rows * cols).map(|_| (r.unit() < 0.7) as u8).collect();
        let gray: Vec<f32> = (0..rows * cols).map(|_| (r.unit() * 50.0) as f32).collect();
        let odd: Vec<bool> = (0..5 * 7).map(|_| r.unit() < 0.5).collect(); // asymmetric
        let ses = [strel_disk(2), strel_disk(5), Strel::square(5), Strel::new(5, 7, odd), strel_disk(15)];
        for se in &ses {
            assert_eq!(erode_bin(&bin, rows, cols, se), brute_morph(&bin, rows, cols, se, true, 1u8));
            assert_eq!(dilate_bin(&bin, rows, cols, se), brute_morph(&bin, rows, cols, se, false, 0u8));
            assert_eq!(erode_gray(&gray, rows, cols, se), brute_morph(&gray, rows, cols, se, true, f32::INFINITY));
            assert_eq!(dilate_gray(&gray, rows, cols, se), brute_morph(&gray, rows, cols, se, false, f32::NEG_INFINITY));
        }
        for w in [1usize, 3, 7, 41] {
            let se = Strel::square(w);
            assert_eq!(erode_bin_square(&bin, rows, cols, w), brute_morph(&bin, rows, cols, &se, true, 1u8));
            assert_eq!(dilate_bin_square(&bin, rows, cols, w), brute_morph(&bin, rows, cols, &se, false, 0u8));
            assert_eq!(erode_gray_square(&gray, rows, cols, w), brute_morph(&gray, rows, cols, &se, true, f32::INFINITY));
            assert_eq!(dilate_gray_square(&gray, rows, cols, w), brute_morph(&gray, rows, cols, &se, false, f32::NEG_INFINITY));
        }
    }

    #[test]
    fn labels_by_columns() {
        // 1 0 0 1
        // 0 1 0 0
        // 0 0 0 1
        // 1 0 1 0
        let b = [1u8, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0];
        let (l, n) = bwlabel(&b, 4, 4);
        assert_eq!(n, 4);
        assert_eq!(l, vec![1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 0, 3, 2, 0, 3, 0]);
        let reg = regionprops(&l, 4, 4, n);
        assert_eq!(reg[0].area, 2);
        assert_eq!(reg[0].centroid, (1.5, 1.5));
        assert_eq!(reg[0].pixel_list, vec![(1, 1), (2, 2)]);
        // column order of a component spanning columns
        assert_eq!(reg[2].pixel_list, vec![(3, 4), (4, 3)]);
        let kept = bwareaopen(&b, 4, 4, 2);
        assert_eq!(kept, vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]);
    }

    #[test]
    fn labels_against_flood_fill() {
        let mut r = Rng(12345);
        let (rows, cols) = (40, 33);
        let img: Vec<u8> = (0..rows * cols).map(|_| (r.unit() < 0.4) as u8).collect();
        let (l, n) = bwlabel(&img, rows, cols);
        // reference: flood fill from the first pixel in a scan by columns
        let mut refl = vec![0i32; rows * cols];
        let mut k = 0;
        for x in 0..cols {
            for y in 0..rows {
                if img[y * cols + x] != 0 && refl[y * cols + x] == 0 {
                    k += 1;
                    let mut st = vec![(y, x)];
                    refl[y * cols + x] = k;
                    while let Some((cy, cx)) = st.pop() {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let (ny, nx) = (cy as i32 + dy, cx as i32 + dx);
                                if ny >= 0 && ny < rows as i32 && nx >= 0 && nx < cols as i32 {
                                    let q = ny as usize * cols + nx as usize;
                                    if img[q] != 0 && refl[q] == 0 {
                                        refl[q] = k;
                                        st.push((ny as usize, nx as usize));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(n, k as usize);
        assert_eq!(l, refl);
    }

    #[test]
    fn fill_holes() {
        let a: Vec<f32> = vec![
            5., 5., 5., 5., 5., //
            5., 1., 5., 2., 5., //
            5., 5., 1., 5., 5., //
            5., 3., 5., 1., 6., //
            5., 5., 5., 5., 5.,
        ];
        let fa = imfill_gray(&a, 5, 5);
        for p in [6, 8, 12, 16, 18] {
            assert_eq!(fa[p], 5.0);
        }
        assert_eq!(fa[19], 6.0); // the border is as it is
        let b: Vec<f32> = vec![
            9., 9., 9., 9., //
            9., 1., 2., 9., //
            9., 9., 9., 3., //
            9., 9., 0., 9.,
        ];
        let f = imfill_gray(&b, 4, 4);
        // the basin (1,1)-(1,2) is walled by 9s on its 4-neighbours (the 3 at (2,3) is only diagonal to it)
        assert_eq!(f[5], 9.0);
        assert_eq!(f[6], 9.0);
        assert_eq!(f[11], 3.0);
        assert_eq!(f[14], 0.0);
        // against an iterative reconstruction
        let mut r = Rng(777);
        let (rows, cols) = (20, 25);
        let img: Vec<f32> = (0..rows * cols).map(|_| (r.next() % 7) as f32).collect();
        let mut marker: Vec<f32> = img.iter().map(|_| 100.0).collect();
        for y in 0..rows {
            for x in 0..cols {
                if y == 0 || x == 0 || y == rows - 1 || x == cols - 1 {
                    marker[y * cols + x] = img[y * cols + x];
                }
            }
        }
        loop {
            let mut ch = false;
            for y in 0..rows {
                for x in 0..cols {
                    let p = y * cols + x;
                    let mut m = marker[p];
                    if y > 0 {
                        m = m.min(marker[p - cols]);
                    }
                    if y + 1 < rows {
                        m = m.min(marker[p + cols]);
                    }
                    if x > 0 {
                        m = m.min(marker[p - 1]);
                    }
                    if x + 1 < cols {
                        m = m.min(marker[p + 1]);
                    }
                    let m = m.max(img[p]);
                    if m < marker[p] {
                        marker[p] = m;
                        ch = true;
                    }
                }
            }
            if !ch {
                break;
            }
        }
        assert_eq!(imfill_gray(&img, rows, cols), marker);
    }

    #[test]
    fn filter3() {
        let img: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let t = [0., 0., 0., 0., 0., 0., 0., 1., 0.0f32]; // picks the pixel below (correlation)
        assert_eq!(imfilter3(&img, 3, 3, &t), vec![4., 5., 6., 7., 8., 9., 0., 0., 0.]);
        let ones = [1.0f32; 9];
        assert_eq!(imfilter3(&img, 3, 3, &ones)[0], 1. + 2. + 4. + 5.);
    }
}
