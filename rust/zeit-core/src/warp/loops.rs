//! The loops over the destination: GWKNearest / GWKRealCase (the methods that
//! sample at cell centres) and GWKAverageOrModeThread (those that take the
//! source cells a destination cell covers), rows on the threads of a pool.

use super::kernels::{bilinear4, cubic4, lanczos, resample, Kernel, Src, Work};
use super::raster::{cvt_i32, iceil, ifloor, Raster};
use super::{Coords, Spec};
use super::{AVERAGE, BILINEAR, CUBIC, CUBICSPLINE, LANCZOS, MAX, MED, MIN, MODE, NEAREST, Q1, Q3, RMS};
use crate::parallel;
use rayon::prelude::*;
use std::ops::Range;

const BAND_DENSITY_THRESHOLD: f64 = 0.0000000001;

/// The destination being written: GWKSetPixelValueReal and the NoData
/// avoidance of one or several bands. The threads write rows of their own.
#[derive(Clone, Copy)]
pub struct Dst<T> {
    pub data: *mut T,
    pub len: usize,
    pub plane: usize,
    pub bands: i32,
    pub nodata: f64,
}

// SAFETY: the rows of the destination go to one thread each, which alone reads
// and writes their cells.
unsafe impl<T: Send> Send for Dst<T> {}
unsafe impl<T: Send> Sync for Dst<T> {}

impl<T: Raster> Dst<T> {
    /// The address of cell `at` of band `band`: the loops take `band` below
    /// `bands` (0 with per_band) and `at` below the plane, `dst_h * dst_w`.
    #[inline(always)]
    fn ptr(&self, band: usize, at: usize) -> *mut T {
        let k = band * self.plane + at;
        debug_assert!(at < self.plane && k < self.len);
        // SAFETY: in bounds (warp() checks that dst holds bands planes).
        unsafe { self.data.add(k) }
    }

    #[inline(always)]
    fn get(&self, band: usize, at: usize) -> T {
        // SAFETY: a cell of a row of the calling thread's (see Dst).
        unsafe { *self.ptr(band, at) }
    }

    #[inline(always)]
    fn put(&self, band: usize, at: usize, v: T) {
        // SAFETY: a cell of a row of the calling thread's (see Dst).
        unsafe { *self.ptr(band, at) = v }
    }

    /// The cells of row jr without a value yet: NoData.
    fn clear_row(&self, fill: T, jr: usize, w: usize) {
        for b in 0..self.bands as usize {
            let k = b * self.plane + jr * w;
            assert!(k + w <= self.len);
            // SAFETY: in bounds, and the calling thread's row.
            unsafe { std::slice::from_raw_parts_mut(self.data.add(k), w) }.fill(fill);
        }
    }

    #[inline(always)]
    fn set(&self, band: usize, at: usize, density: f64, mut real: f64) {
        if density < 0.9999 {
            if density < 0.0001 {
                return;
            }
            // The destination cell has no value yet (density 0).
            let dst_real = self.get(band, at).to_f64();
            let dst_influence = (1.0 - density) * 0.0;
            real = (real * density + dst_real * dst_influence) / (density + dst_influence);
        }
        self.set_value(band, at, T::clamp_round(real));
    }

    #[inline(always)]
    fn set_value(&self, band: usize, at: usize, value: T) {
        let out = if self.bands == 1 && self.nodata == value.to_f64() {
            value.avoid_nodata()
        } else {
            value
        };
        self.put(band, at, out);
    }

    /// GWKAvoidNoDataMultiBand.
    fn avoid_multi(&self, at: usize) {
        if self.bands == 1 {
            return;
        }
        for b in 0..self.bands as usize {
            if self.nodata != self.get(b, at).to_f64() {
                return;
            }
        }
        for b in 0..self.bands as usize {
            self.put(b, at, self.get(b, at).avoid_nodata());
        }
    }
}

/// `f(state, rows)` on chunks of `chunk` consecutive rows of `0..n`, on
/// `threads` threads, with one `state` from `init` per task (OpenMP's
/// `schedule(dynamic, chunk)` with per-thread buffers).
fn rows<S, I, F>(n: usize, chunk: usize, threads: usize, init: I, f: F)
where
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, Range<usize>) + Sync + Send,
{
    if n == 0 {
        return;
    }
    if threads <= 1 {
        let mut state = init();
        f(&mut state, 0..n);
        return;
    }
    parallel::pool(threads).install(|| {
        (0..n.div_ceil(chunk))
            .into_par_iter()
            .for_each_init(&init, |state, c| f(state, c * chunk..n.min((c + 1) * chunk)));
    });
}

/// GWKOneSourceCornerFailsToReproject: a corner of the source does not go to
/// the destination (with the transformer only).
fn corner_fails<T: Raster>(s: &Src<T>, coords: &Coords) -> bool {
    if coords.transformer.is_none() {
        return false;
    }
    for iy in 0..=1 {
        for ix in 0..=1 {
            let mut x = [ix as f64 * s.w as f64];
            let mut y = [iy as f64 * s.h as f64];
            let mut ok = [0];
            coords.transform(false, &mut x, &mut y, &mut ok);
            if ok[0] == 0 {
                return true;
            }
        }
    }
    false
}

/// GWKAdjustSrcOffsetOnEdge: an invalid cell at the edge of the projection's
/// domain (a corner of it does not go to the destination) takes a valid
/// neighbour.
fn adjust_on_edge<T: Raster>(s: &Src<T>, coords: &Coords, isx: &mut i32, isy: &mut i32) -> bool {
    let mut ok = [0];
    let (mut x, mut y) = ([*isx as f64], [*isy as f64]);
    coords.transform(false, &mut x, &mut y, &mut ok);
    if ok[0] != 0 {
        x[0] = *isx as f64;
        y[0] = (*isy + 1) as f64;
        coords.transform(false, &mut x, &mut y, &mut ok);
    }
    if ok[0] != 0 {
        x[0] = (*isx + 1) as f64;
        y[0] = *isy as f64;
        coords.transform(false, &mut x, &mut y, &mut ok);
    }
    if ok[0] != 0 {
        return false;
    }
    if *isx + 1 < s.w && s.ok(*isy, *isx + 1) {
        *isx += 1;
        return true;
    }
    if *isy + 1 < s.h && s.ok(*isy + 1, *isx) {
        *isy += 1;
        return true;
    }
    if *isx > 0 && s.ok(*isy, *isx - 1) {
        *isx -= 1;
        return true;
    }
    if *isy > 0 && s.ok(*isy - 1, *isx) {
        *isy -= 1;
        return true;
    }
    false
}

/// The source and destination of band b, and the band's index in them: with
/// per_band (ps, pd: one source and destination of a band each), every band is
/// tested and written on its own.
#[inline(always)]
fn band<'s, 'a, T>(
    s: &'s Src<'a, T>,
    d: &'s Dst<T>,
    ps: Option<&'s [Src<'a, T>]>,
    pd: Option<&'s [Dst<T>]>,
    b: usize,
) -> (&'s Src<'a, T>, &'s Dst<T>, usize) {
    match (ps, pd) {
        (Some(ps), Some(pd)) => (&ps[b], &pd[b], 0),
        _ => (s, d, b),
    }
}

/// GWKNearest / GWKRealCase: the methods that sample at cell centres, one
/// loop per method.
#[allow(clippy::too_many_arguments)]
fn warp_points_loop<T: Raster, const M: i32>(
    s: &Src<T>,
    d: &Dst<T>,
    ps: Option<&[Src<T>]>,
    pd: Option<&[Dst<T>]>,
    coords: &Coords,
    sp: &Spec,
    k: &Kernel,
    threads: usize,
) {
    let one_cell = s.w == 1 || s.h == 1;
    let on_edge = s.masked && corner_fails(s, coords);
    let w = sp.dst_w as usize;
    let fill = T::from_f64(d.nodata);
    let (sw, sh) = (s.w as f64, s.h as f64);
    let init = || (vec![0.0; w], vec![0.0; w], Work::new(k));
    rows(sp.dst_h as usize, 4, threads, init, |(x, y, wk), range| {
        for jr in range {
            d.clear_row(fill, jr, w);
            coords.row(sp.dst_r0 as i64 + jr as i64, sp.dst_c0 as i64, x, y);
            for ir in 0..w {
                let (sx, sy) = (x[ir], y[ir]);
                // GWKCheckAndComputeSrcOffsets
                if !(sx >= 0.0) || !(sy >= 0.0) {
                    continue; // also NaN
                }
                if !(sx + 1e-10 <= sw) || !(sy + 1e-10 <= sh) {
                    continue; // also infinite
                }
                let mut isx = cvt_i32(sx + 1.0e-10);
                let mut isy = cvt_i32(sy + 1.0e-10);
                if isx == s.w {
                    isx -= 1;
                }
                if isy == s.h {
                    isy -= 1;
                }
                if !s.held(isy, isx) {
                    continue; // never with a window computed right
                }
                if ps.is_none()
                    && !s.valid_at(s.at(isy, isx))
                    && !(on_edge && adjust_on_edge(s, coords, &mut isx, &mut isy))
                {
                    continue;
                }

                let at = jr * w + ir;
                let k0 = s.at(isy, isx); // the source cell's index in the window, every band's
                let mut found = false;
                for b in 0..sp.bands as usize {
                    let (sb, db, bi) = band(s, d, ps, pd, b);
                    let (mut bx, mut by) = (isx, isy);
                    if ps.is_some() && !sb.valid_at(k0) && !(on_edge && adjust_on_edge(sb, coords, &mut bx, &mut by)) {
                        continue;
                    }
                    let (mut dens, mut value) = (0.0, 0.0);
                    if M == NEAREST || one_cell {
                        let kb = bi * sb.plane + if (bx, by) == (isx, isy) { k0 } else { sb.at(by, bx) };
                        if T::COPY && M == NEAREST {
                            db.set_value(bi, at, sb.data[kb]);
                            found = true;
                            continue;
                        }
                        value = sb.data[kb].to_f64();
                        dens = 1.0;
                    } else if M == BILINEAR {
                        if k.use4 {
                            bilinear4(sb, bi, sx, sy, &mut dens, &mut value);
                        } else {
                            resample::<T, M>(sb, k, wk, bi, sx, sy, &mut dens, &mut value);
                        }
                    } else if M == CUBIC {
                        if k.use4 {
                            cubic4(sb, bi, sx, sy, &mut dens, &mut value);
                        } else {
                            resample::<T, M>(sb, k, wk, bi, sx, sy, &mut dens, &mut value);
                        }
                    } else if M == LANCZOS {
                        lanczos(sb, k, wk, bi, sx, sy, &mut dens, &mut value);
                    } else {
                        resample::<T, M>(sb, k, wk, bi, sx, sy, &mut dens, &mut value);
                    }
                    if dens < BAND_DENSITY_THRESHOLD {
                        continue;
                    }
                    found = true;
                    db.set(bi, at, dens, value);
                }
                if found && ps.is_none() {
                    d.avoid_multi(at);
                }
            }
        }
    });
}

pub fn warp_points<T: Raster>(
    s: &Src<T>,
    d: &Dst<T>,
    ps: Option<&[Src<T>]>,
    pd: Option<&[Dst<T>]>,
    coords: &Coords,
    sp: &Spec,
    threads: usize,
) {
    let k = Kernel::new(sp.method, sp.xscale, sp.yscale);
    match sp.method {
        NEAREST => warp_points_loop::<T, NEAREST>(s, d, ps, pd, coords, sp, &k, threads),
        BILINEAR => warp_points_loop::<T, BILINEAR>(s, d, ps, pd, coords, sp, &k, threads),
        CUBIC => warp_points_loop::<T, CUBIC>(s, d, ps, pd, coords, sp, &k, threads),
        CUBICSPLINE => warp_points_loop::<T, CUBICSPLINE>(s, d, ps, pd, coords, sp, &k, threads),
        _ => warp_points_loop::<T, LANCZOS>(s, d, ps, pd, coords, sp, &k, threads),
    }
}

/// GWKAverageOrModeComputeSourceCoords: the source cells under destination
/// cell (i, j), from its corners; across the antimeridian (wrap) the columns
/// go on past the source's last one, back to its first.
struct Footprint {
    x_min: f64,
    y_min: f64,
    x_max: f64,
    y_max: f64,
    isx_min: i32,
    isy_min: i32,
    isx_max: i32,
    isy_max: i32,
    wrap: bool,
}

#[allow(clippy::too_many_arguments)]
fn footprint(
    mut x1: f64,
    mut y1: f64,
    mut x2: f64,
    mut y2: f64,
    w: i32,
    h: i32,
    x_margin: i32,
    y_margin: i32,
    xscale: f64,
    coords: &Coords,
    i: i64,
    j: i64,
) -> Option<Footprint> {
    let (xm, ym, wf, hf) = (x_margin as f64, y_margin as f64, w as f64, h as f64);
    if !(x1 >= -xm
        && x2 >= -xm
        && y1 >= -ym
        && y2 >= -ym
        && x1 - wf <= xm
        && x2 - wf <= xm
        && y1 - hf <= ym
        && y2 - hf <= ym)
    {
        return None;
    }
    if x1 > x2 {
        std::mem::swap(&mut x1, &mut x2);
    }
    let mut wrap = false;
    let threshold = 2.min(w / 10) as f64;
    if coords.transformer.is_some() && x1 * xscale < threshold && (wf - x2) * xscale < threshold {
        // The cell spans the whole source: across the antimeridian if its middle is left of it.
        let (mut x, mut y) = ([i as f64 + 0.5], [j as f64]);
        let mut ok = [0];
        coords.transform(true, &mut x, &mut y, &mut ok);
        if ok[0] != 0 && x[0] < x1 {
            wrap = true;
            std::mem::swap(&mut x1, &mut x2);
            x2 += wf;
        }
    }
    let (x_min, x_max) = (x1, x2);
    const EPSILON: f64 = 1e-10;
    if !(x_max > -EPSILON && x_min < wf + EPSILON) {
        return None;
    }
    // (the margins keep the coordinates in the range of int)
    let isx_min = ifloor(x_min + EPSILON).max(0);
    let mut isx_max = iceil(x_max - EPSILON);
    if !wrap {
        isx_max = isx_max.min(w);
    }
    if isx_min == isx_max && isx_max < w {
        isx_max += 1;
    }
    if y1 > y2 {
        std::mem::swap(&mut y1, &mut y2);
    }
    let (y_min, y_max) = (y1, y2);
    if !(y_max > -EPSILON && y_min < hf + EPSILON) {
        return None;
    }
    let isy_min = ifloor(y_min + EPSILON).max(0);
    let mut isy_max = iceil(y_max - EPSILON).min(h);
    if isy_min == isy_max && isy_max < h {
        isy_max += 1;
    }
    Some(Footprint {
        x_min,
        y_min,
        x_max,
        y_max,
        isx_min,
        isy_min,
        isx_max,
        isy_max,
        wrap,
    })
}

impl Footprint {
    /// COMPUTE_WEIGHT_Y.
    #[inline(always)]
    fn weight_y(&self, isy: i32) -> f64 {
        if isy == self.isy_min {
            if self.isy_min + 1 == self.isy_max {
                1.0
            } else {
                1.0 - (self.y_min - self.isy_min as f64)
            }
        } else if isy + 1 == self.isy_max {
            1.0 - (self.isy_max as f64 - self.y_max)
        } else {
            1.0
        }
    }

    /// COMPUTE_WEIGHT.
    #[inline(always)]
    fn weight(&self, isx: i32, weight_y: f64) -> f64 {
        if isx == self.isx_min {
            if self.isx_min + 1 == self.isx_max {
                weight_y
            } else {
                weight_y * (1.0 - (self.x_min - self.isx_min as f64))
            }
        } else if isx + 1 == self.isx_max {
            weight_y * (1.0 - (self.isx_max as f64 - self.x_max))
        } else {
            weight_y
        }
    }
}

/// A thread's buffers of the area methods.
struct AreaWork<T> {
    x0: Vec<f64>,
    y0: Vec<f64>,
    x1: Vec<f64>,
    y1: Vec<f64>,
    counts: Vec<f32>,
    touched: Vec<i32>,
    mode_vals: Vec<T>,
    mode_counts: Vec<f32>,
    values: Vec<f64>,
    /// The row whose lower corners x1, y1 hold.
    last: i64,
}

/// GWKAverageOrModeThread and GWKModeRealType: the methods that take the
/// source cells a destination cell covers (per_band: as warp_points_loop).
fn warp_areas_loop<T: Raster, const M: i32>(
    s: &Src<T>,
    d: &Dst<T>,
    ps: Option<&[Src<T>]>,
    pd: Option<&[Dst<T>]>,
    coords: &Coords,
    sp: &Spec,
    threads: usize,
) {
    let x_margin = 2 * 1.max(cvt_i32((1.0 / sp.xscale).ceil()));
    let y_margin = 2 * 1.max(cvt_i32((1.0 / sp.yscale).ceil()));
    let quant: f32 = match M {
        MED => 0.5,
        Q1 => 0.25,
        Q3 => 0.75,
        _ => 0.0,
    };
    let w = sp.dst_w as usize;
    let fill = T::from_f64(d.nodata);
    let init = || AreaWork::<T> {
        x0: vec![0.0; w + 1],
        y0: vec![0.0; w + 1],
        x1: vec![0.0; w + 1],
        y1: vec![0.0; w + 1],
        counts: vec![0.0; if M == MODE && T::SMALL { T::BINS } else { 0 }],
        touched: Vec::new(),
        mode_vals: Vec::new(),
        mode_counts: Vec::new(),
        values: Vec::new(),
        last: -2,
    };
    rows(sp.dst_h as usize, 8, threads, init, |wk, range| {
        for jr in range {
            d.clear_row(fill, jr, w);
            let j = sp.dst_r0 as i64 + jr as i64;
            if jr as i64 == wk.last + 1 {
                // its upper corners: the lower ones of the row before
                std::mem::swap(&mut wk.x0, &mut wk.x1);
                std::mem::swap(&mut wk.y0, &mut wk.y1);
            } else {
                coords.row(j, sp.dst_c0 as i64, &mut wk.x0, &mut wk.y0);
            }
            coords.row(j + 1, sp.dst_c0 as i64, &mut wk.x1, &mut wk.y1);
            wk.last = jr as i64;
            for ir in 0..w {
                let Some(f) = footprint(
                    wk.x0[ir],
                    wk.y0[ir],
                    wk.x1[ir + 1],
                    wk.y1[ir + 1],
                    s.w,
                    s.h,
                    x_margin,
                    y_margin,
                    sp.xscale,
                    coords,
                    sp.dst_c0 as i64 + ir as i64,
                    j,
                ) else {
                    continue;
                };
                let at = jr * w + ir;
                let mut found = false;
                let held = f.isx_min < f.isx_max
                    && f.isy_min < f.isy_max
                    && s.held(f.isy_min, f.isx_min)
                    && s.held(f.isy_max - 1, f.isx_max - 1);
                for b in 0..sp.bands as usize {
                    let (sb, db, bi) = band(s, d, ps, pd, b);
                    let valid = |r: i32, c: i32| if held { sb.valid_at(sb.at(r, c)) } else { sb.ok(r, c) };
                    // the source column of footprint column c
                    let col = |c: i32| if f.wrap { c % s.w } else { c };
                    if M == AVERAGE {
                        let (mut total_weight, mut value_real) = (0.0, 0.0);
                        for r in f.isy_min..f.isy_max {
                            let weight_y = f.weight_y(r);
                            for c in f.isx_min..f.isx_max {
                                let cc = col(c);
                                if !valid(r, cc) {
                                    continue;
                                }
                                let weight = f.weight(c, weight_y);
                                if weight > 0.0 {
                                    total_weight += weight;
                                    value_real += (weight / total_weight) * (sb.val(bi, r, cc) - value_real);
                                }
                            }
                        }
                        if total_weight > 0.0 {
                            found = true;
                            db.set(bi, at, 1.0, value_real);
                        }
                    } else if M == RMS {
                        let (mut total_real, mut total_weight) = (0.0, 0.0);
                        for r in f.isy_min..f.isy_max {
                            let weight_y = f.weight_y(r);
                            for c in f.isx_min..f.isx_max {
                                let cc = col(c);
                                if !valid(r, cc) {
                                    continue;
                                }
                                let v = sb.val(bi, r, cc);
                                let weight = f.weight(c, weight_y);
                                total_weight += weight;
                                total_real += v * v * weight;
                            }
                        }
                        if total_weight > 0.0 {
                            found = true;
                            db.set(bi, at, 1.0, (total_real / total_weight).sqrt());
                        }
                    } else if M == MODE && T::SMALL {
                        let mut max_count = 0.0f32;
                        let mut mode = -1;
                        let mut has_source_values = false;
                        for r in f.isy_min..f.isy_max {
                            let weight_y = f.weight_y(r);
                            for c in f.isx_min..f.isx_max {
                                let cc = col(c);
                                if !valid(r, cc) {
                                    continue;
                                }
                                has_source_values = true;
                                let v = sb.raw(bi, r, cc).to_i32();
                                let bin = v + T::BINS_OFFSET;
                                let weight = f.weight(c, weight_y);
                                let count = &mut wk.counts[bin as usize];
                                if *count == 0.0 {
                                    wk.touched.push(bin);
                                }
                                *count += weight as f32;
                                if *count > max_count {
                                    // ties: the first (GWKTS_First)
                                    mode = v;
                                    max_count = *count;
                                }
                            }
                        }
                        for &t in &wk.touched {
                            wk.counts[t as usize] = 0.0;
                        }
                        wk.touched.clear();
                        if has_source_values {
                            found = true;
                            db.set(bi, at, 1.0, mode as f64);
                        }
                    } else if M == MODE {
                        // GWKModeRealType: values in order of appearance.
                        wk.mode_vals.clear();
                        wk.mode_counts.clear();
                        let mut mode_index: isize = -1;
                        for r in f.isy_min..f.isy_max {
                            let weight_y = f.weight_y(r);
                            for c in f.isx_min..f.isx_max {
                                let cc = col(c);
                                if !valid(r, cc) {
                                    continue;
                                }
                                let v = sb.raw(bi, r, cc);
                                let weight = f.weight(c, weight_y);
                                let mut i = 0;
                                while i < wk.mode_vals.len() {
                                    let m = wk.mode_vals[i];
                                    // IsSame: NaN is NaN
                                    if m == v || (m.is_nan() && v.is_nan()) {
                                        wk.mode_counts[i] += weight as f32;
                                        if wk.mode_counts[i] > wk.mode_counts[mode_index as usize] {
                                            mode_index = i as isize;
                                        }
                                        break;
                                    }
                                    i += 1;
                                }
                                if i == wk.mode_vals.len() {
                                    wk.mode_vals.push(v);
                                    wk.mode_counts.push(weight as f32);
                                    if mode_index < 0 {
                                        mode_index = i as isize;
                                    }
                                }
                            }
                        }
                        if mode_index != -1 {
                            found = true;
                            db.set_value(bi, at, wk.mode_vals[mode_index as usize]);
                        }
                    } else if M == MAX || M == MIN {
                        let mut found_valid = false;
                        let mut total_real = if M == MAX { f64::MIN } else { f64::MAX };
                        for r in f.isy_min..f.isy_max {
                            for c in f.isx_min..f.isx_max {
                                let cc = col(c);
                                if !valid(r, cc) {
                                    continue;
                                }
                                let v = sb.val(bi, r, cc);
                                found_valid = true;
                                if if M == MAX { total_real < v } else { total_real > v } {
                                    total_real = v;
                                }
                            }
                        }
                        if found_valid {
                            found = true;
                            db.set(bi, at, 1.0, total_real);
                        }
                    } else {
                        // med, q1, q3
                        wk.values.clear();
                        for r in f.isy_min..f.isy_max {
                            for c in f.isx_min..f.isx_max {
                                let cc = col(c);
                                if valid(r, cc) {
                                    wk.values.push(sb.val(bi, r, cc));
                                }
                            }
                        }
                        if !wk.values.is_empty() {
                            wk.values.sort_unstable_by(f64::total_cmp);
                            let pos = quant * wk.values.len() as f32 - 1.0;
                            let quant_idx = cvt_i32(pos.ceil() as f64);
                            found = true;
                            db.set(bi, at, 1.0, wk.values[quant_idx as usize]);
                        }
                    }
                }
                if found && ps.is_none() {
                    d.avoid_multi(at);
                }
            }
        }
    });
}

pub fn warp_areas<T: Raster>(
    s: &Src<T>,
    d: &Dst<T>,
    ps: Option<&[Src<T>]>,
    pd: Option<&[Dst<T>]>,
    coords: &Coords,
    sp: &Spec,
    threads: usize,
) {
    match sp.method {
        AVERAGE => warp_areas_loop::<T, AVERAGE>(s, d, ps, pd, coords, sp, threads),
        RMS => warp_areas_loop::<T, RMS>(s, d, ps, pd, coords, sp, threads),
        MODE => warp_areas_loop::<T, MODE>(s, d, ps, pd, coords, sp, threads),
        MAX => warp_areas_loop::<T, MAX>(s, d, ps, pd, coords, sp, threads),
        MIN => warp_areas_loop::<T, MIN>(s, d, ps, pd, coords, sp, threads),
        MED => warp_areas_loop::<T, MED>(s, d, ps, pd, coords, sp, threads),
        Q1 => warp_areas_loop::<T, Q1>(s, d, ps, pd, coords, sp, threads),
        _ => warp_areas_loop::<T, Q3>(s, d, ps, pd, coords, sp, threads),
    }
}
