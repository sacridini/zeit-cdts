//! Global co-registration of an image to a reference: the shift of a target image that best aligns it with a
//! reference on the same pixel grid, from the phase correlation of one matching window.
//!
//! A port of the global `COREG` of AROSICS 1.14.0 (Automated and Robust Open-Source Image Co-Registration
//! Software, `arosics/CoReg.py`; Scheffler, Hollstein, Hofmann and Bochow 2017, "AROSICS: An automated and robust
//! open-source image co-registration software for multi-sensor satellite data", Remote Sensing 9(7):676),
//!
//!   Copyright (C) 2017-2026 Daniel Scheffler (GFZ Potsdam, daniel.scheffler@gfz.de) and the GFZ Helmholtz Centre
//!   for Geosciences, Potsdam, Germany (https://www.gfz.de/). Licensed under the Apache License, Version 2.0
//!   (https://www.apache.org/licenses/LICENSE-2.0), which has no NOTICE file; distributed "AS IS", without
//!   warranties or conditions of any kind. This file is a translation to Rust of the parts of AROSICS named
//!   below, with changes (the window choice on pixel masks instead of footprint polygons, the FFT of the
//!   `rustfft` crate), under GPL-3.0-or-later as the rest of zeit (Apache-2.0 code may enter it).
//!
//! What is ported, in AROSICS' order of operations:
//!
//! * the matching window (`_get_opt_winpos_winsize`, `_get_clip_window_properties`): centred on the centroid of the
//!   overlap of the two images' valid data (or a given position), `window_size` pixels square, clipped to the
//!   overlap, moved to the pixel grid by taking the floor of its edges (`move_shapelyPoly_to_image_grid`, `NW`),
//!   shrunk while the window of the other image, which AROSICS keeps as the unrounded box of the full size until it
//!   has to be replaced, is not within the overlap; an odd count of rows or columns loses the last one
//!   (`_get_image_windows_to_match`);
//! * the shifted cross power spectrum (`_calc_shifted_cross_power_spectrum`): the largest power of 2 not above the
//!   window (the same on both sides), centred, single-precision FFTs, `F0 conj(F1) / (|F0| |F1| + eps)` with
//!   `eps = max|F1| 1e-15`, the inverse FFT's magnitude, `fftshift`;
//! * the integer shift (`_get_peakpos`, `_calc_integer_shifts`) and its validation by shifting the reference's
//!   window virtually by it (`_get_grossly_deshifted_images`, `_validate_integer_shifts`), repeated up to
//!   `max_iter` times as AROSICS does;
//! * the subpixel shift from the side maxima of the peak (`_find_side_maximum`, `_calc_subpixel_shifts`), the
//!   `max_shift` test, the reliability (`_calc_shift_reliability`) and the structural similarity of the window
//!   before and after the shift (`_validate_ssim_improvement` with scikit-image's `structural_similarity` at its
//!   defaults: a 7 x 7 uniform window, K1 0.01, K2 0.03, sample covariance, on the windows scaled to 0..1).
//!
//! Shifts are as AROSICS': in pixels of the target, `x` to the right and `y` down, the amount the target has to be
//! moved to fit the reference. A target pixel `(r, c)` of the aligned image is the target's at `(r - y, c - x)`.
//!
//! Where it differs from AROSICS (which works on footprint polygons and GDAL): the overlap is the intersection of
//! the two valid-data masks (holes are not filled; AROSICS fills those of a footprint), its centroid is the mean
//! of its pixels' centres, a centroid outside it is replaced by the valid pixel nearest to it (AROSICS takes
//! shapely's representative point), the branch of AROSICS that shrinks the window from a one-unit box (which only
//! runs on rounding noise there) is left out, the two images must be on the same grid (AROSICS resamples the
//! other image's window with GDAL when they are not; they are then equal windows on equal grids anyway), and the
//! shifted window of the similarity is GDAL's cubic of 4 x 4 cells (Keys' with a = -0.5, bilinear where it does
//! not fit or a cell is invalid, as `GWKCubicResample4Sample`).

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

const EPS: f64 = 1e-9;

/// The reasons a shift is not found (AROSICS raises these errors; with `ignore_errors` it flags the failure).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// No pixel is valid in both images.
    NoOverlap,
    /// The overlap covers at most 16 x 16 pixels.
    SmallOverlap,
    /// The window position given is not on valid data of both images.
    WindowPosition,
    /// The window has fewer than 16 pixels on a side, or is too small for the FFT.
    WindowTooSmall,
    /// The window of the other image is larger than the overlap and cannot shrink to fit.
    WindowShrink,
    /// No integer shift passed the validation in `max_iter` tries.
    NoMatch,
    /// The shift is above `max_shift`.
    ShiftTooLarge,
    /// The correlation has no usable peak (not finite, or at its border).
    NoPeak,
}

impl Failure {
    pub fn as_str(self) -> &'static str {
        match self {
            Failure::NoOverlap => "no valid data in common",
            Failure::SmallOverlap => "overlap of at most 16 x 16 pixels",
            Failure::WindowPosition => "window position outside the valid data of both images",
            Failure::WindowTooSmall => "matching window too small",
            Failure::WindowShrink => "window larger than the overlap and cannot shrink",
            Failure::NoMatch => "no match found in the window",
            Failure::ShiftTooLarge => "shift above max_shift",
            Failure::NoPeak => "no usable correlation peak",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    /// AROSICS' `ws`: the matching window's side, in pixels (256).
    pub window_size: usize,
    /// AROSICS' `wp`: the window's centre `(x, y)` in pixel coordinates of the images (the corner of the first
    /// pixel is `(0, 0)`); None: the centroid of the overlap.
    pub window_pos: Option<(f64, f64)>,
    /// AROSICS' `max_shift`, in pixels (5).
    pub max_shift: f64,
    /// AROSICS' `max_iter` (5).
    pub max_iter: usize,
    /// The range of the target's integer type, when it is one: the shifted window is rounded and clamped to it
    /// before the similarity, as GDAL writes it into an image of that type.
    pub integer_range: Option<(f64, f64)>,
    /// The target's NoData value: GDAL's AvoidNoData moves a rounded value that equals it one step (down, or up
    /// from the type's minimum).
    pub nodata: Option<f64>,
    /// The value of the window's cells where the shifted target has no data.
    pub fill: f32,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            window_size: 256,
            window_pos: None,
            max_shift: 5.0,
            max_iter: 5,
            integer_range: None,
            nodata: None,
            fill: 0.0,
        }
    }
}

/// One image: `ny * nx` values, row-major, and whether each is data.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    pub data: &'a [f32],
    pub valid: &'a [bool],
    pub ny: usize,
    pub nx: usize,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub success: bool,
    pub failure: Option<Failure>,
    /// The total shift in pixels (integer + subpixel); NaN when not found.
    pub x_shift: f64,
    pub y_shift: f64,
    pub x_int: i64,
    pub y_int: i64,
    pub x_sub: f64,
    pub y_sub: f64,
    /// AROSICS' confidence of the peak, 0-100 (NaN when not found).
    pub reliability: f64,
    /// Structural similarity of the window before and after the shift (NaN when not found).
    pub ssim_before: f64,
    pub ssim_after: f64,
    /// The matching window `[x0, y0, x1, y1)` in pixels (zeros when it could not be placed).
    pub window: [usize; 4],
    /// The side of the FFT window.
    pub fft_size: usize,
    /// Validations of the integer shift it took.
    pub iterations: usize,
}

impl Outcome {
    fn failed(f: Failure) -> Outcome {
        Outcome {
            success: false,
            failure: Some(f),
            x_shift: f64::NAN,
            y_shift: f64::NAN,
            x_int: 0,
            y_int: 0,
            x_sub: f64::NAN,
            y_sub: f64::NAN,
            reliability: f64::NAN,
            ssim_before: f64::NAN,
            ssim_after: f64::NAN,
            window: [0; 4],
            fft_size: 0,
            iterations: 0,
        }
    }
}

// --- the overlap and the window ----------------------------------------------------------------

/// The pixels valid in both images, with the counts of the others in a summed-area table.
struct Overlap {
    nx: usize,
    ny: usize,
    ok: Vec<bool>,
    inv: Vec<u32>,
    /// `[x0, y0, x1, y1)` of the valid pixels.
    bbox: [i64; 4],
    count: usize,
    sum_x: f64,
    sum_y: f64,
}

impl Overlap {
    fn new(a: &Image, b: &Image) -> Overlap {
        let (ny, nx) = (a.ny, a.nx);
        let ok: Vec<bool> = a.valid.iter().zip(b.valid).map(|(&p, &q)| p && q).collect();
        let w = nx + 1;
        let mut inv = vec![0u32; (ny + 1) * w];
        let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
        let (mut count, mut sx, mut sy) = (0usize, 0.0f64, 0.0f64);
        for r in 0..ny {
            let mut run = 0u32;
            let mut row_n = 0usize;
            let mut row_sx = 0.0f64;
            for c in 0..nx {
                if ok[r * nx + c] {
                    row_n += 1;
                    row_sx += c as f64 + 0.5;
                    x0 = x0.min(c as i64);
                    x1 = x1.max(c as i64 + 1);
                } else {
                    run += 1;
                }
                inv[(r + 1) * w + c + 1] = inv[r * w + c + 1] + run;
            }
            if row_n > 0 {
                y0 = y0.min(r as i64);
                y1 = y1.max(r as i64 + 1);
                count += row_n;
                sx += row_sx;
                sy += row_n as f64 * (r as f64 + 0.5);
            }
        }
        Overlap {
            nx,
            ny,
            ok,
            inv,
            bbox: [x0, y0, x1, y1],
            count,
            sum_x: sx,
            sum_y: sy,
        }
    }

    /// The box `[x0, x1) x [y0, y1)` is inside the image and all of it valid.
    fn box_valid(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> bool {
        if x1 <= x0 || y1 <= y0 || x0 < 0 || y0 < 0 || x1 > self.nx as i64 || y1 > self.ny as i64 {
            return false;
        }
        let w = self.nx + 1;
        let g = |r: i64, c: i64| self.inv[r as usize * w + c as usize] as i64;
        g(y1, x1) - g(y0, x1) - g(y1, x0) + g(y0, x0) == 0
    }

    /// The same for a box with fractional edges: every pixel it touches must be valid.
    fn box_valid_f(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
        self.box_valid(
            (x0 + EPS).floor() as i64,
            (y0 + EPS).floor() as i64,
            (x1 - EPS).ceil() as i64,
            (y1 - EPS).ceil() as i64,
        )
    }
}

/// The matching window `[x0, y0, x1, y1)` (AROSICS' `matchBox`), whole pixels.
fn match_box(ov: &Overlap, p: &Params) -> Result<[i64; 4], Failure> {
    if ov.count == 0 {
        return Err(Failure::NoOverlap);
    }
    // AROSICS: at least 16 * 16 pixels of overlap.
    if ov.count <= 16 * 16 {
        return Err(Failure::SmallOverlap);
    }
    let (nx, ny) = (ov.nx as i64, ov.ny as i64);
    let (wpx, wpy) = match p.window_pos {
        Some((x, y)) => {
            let (c, r) = (x.floor() as i64, y.floor() as i64);
            if c < 0 || r < 0 || c >= nx || r >= ny || !ov.ok[r as usize * ov.nx + c as usize] {
                return Err(Failure::WindowPosition);
            }
            (x, y)
        }
        None => {
            let (x, y) = (ov.sum_x / ov.count as f64, ov.sum_y / ov.count as f64);
            let (c, r) = (x.floor() as i64, y.floor() as i64);
            if c >= 0 && r >= 0 && c < nx && r < ny && ov.ok[r as usize * ov.nx + c as usize] {
                (x, y)
            } else {
                // The centroid is outside the overlap: its nearest valid pixel (AROSICS: shapely's representative
                // point).
                let mut best = (f64::INFINITY, x, y);
                for r in 0..ov.ny {
                    for c in 0..ov.nx {
                        if ov.ok[r * ov.nx + c] {
                            let (px, py) = (c as f64 + 0.5, r as f64 + 0.5);
                            let d = (px - x).powi(2) + (py - y).powi(2);
                            if d < best.0 {
                                best = (d, px, py);
                            }
                        }
                    }
                }
                (best.1, best.2)
            }
        }
    };
    let half = p.window_size as f64 / 2.0;
    let (wx0, wx1, wy0, wy1) = (wpx - half, wpx + half, wpy - half, wpy + half);

    // The window clipped to the overlap: the bounds of the valid pixels it touches.
    let rx0 = ((wx0 + EPS).floor() as i64).max(0);
    let rx1 = ((wx1 - EPS).ceil() as i64).min(nx);
    let ry0 = ((wy0 + EPS).floor() as i64).max(0);
    let ry1 = ((wy1 - EPS).ceil() as i64).min(ny);
    let (mut minc, mut maxc, mut minr, mut maxr) = (i64::MAX, i64::MIN, i64::MAX, i64::MIN);
    for r in ry0..ry1 {
        for c in rx0..rx1 {
            if ov.ok[r as usize * ov.nx + c as usize] {
                minc = minc.min(c);
                maxc = maxc.max(c);
                minr = minr.min(r);
                maxr = maxr.max(r);
            }
        }
    }
    if minc > maxc {
        return Err(Failure::WindowPosition);
    }
    let mx0 = wx0.max(minc as f64);
    let mx1 = wx1.min((maxc + 1) as f64);
    let my0 = wy0.max(minr as f64);
    let my1 = wy1.min((maxr + 1) as f64);
    // To the pixel grid, towards the north west: the floor of every edge.
    let fl = |v: f64| (v + EPS).floor() as i64;
    let (mut x0, mut y0, mut x1, mut y1) = (fl(mx0), fl(my0), fl(mx1), fl(my1));
    let ob = ov.bbox;

    // Moved out of the overlap: smaller by one pixel a side in the dimension that is larger than it.
    if !ov.box_valid(x0, y0, x1, y1) {
        let xl = x0 < ob[0] || x1 > ob[2];
        let yl = y0 < ob[1] || y1 > ob[3];
        if xl {
            x0 += 1;
            x1 -= 1;
        }
        if yl {
            y0 += 1;
            y1 -= 1;
        }
    }

    // The other image's window: the unrounded box of the full size, until it is replaced by the smallest box of
    // its grid that holds the matching window (the same box, the grids being equal).
    let mut other: Option<[f64; 4]> = Some([wx0, wy0, wx1, wy1]);
    if let Some(o) = other {
        let inside =
            x0 as f64 >= o[0] - EPS && y0 as f64 >= o[1] - EPS && x1 as f64 <= o[2] + EPS && y1 as f64 <= o[3] + EPS;
        if !inside {
            other = None;
        }
    }
    for _ in 0..10_000 {
        let within = match other {
            Some(o) => ov.box_valid_f(o[0], o[1], o[2], o[3]),
            None => ov.box_valid(x0, y0, x1, y1),
        };
        if within {
            break;
        }
        let (b, prev_area) = match other {
            Some(o) => (
                [
                    o[0].round_ties_even() as i64,
                    o[1].round_ties_even() as i64,
                    o[2].round_ties_even() as i64,
                    o[3].round_ties_even() as i64,
                ],
                (o[2] - o[0]) * (o[3] - o[1]),
            ),
            None => ([x0, y0, x1, y1], ((x1 - x0) * (y1 - y0)) as f64),
        };
        if b[0] < ob[0] || b[2] > ob[2] {
            x0 += 1;
            x1 -= 1;
        }
        if b[1] < ob[1] || b[3] > ob[3] {
            y0 += 1;
            y1 -= 1;
        }
        other = None;
        if x1 <= x0 || y1 <= y0 {
            return Err(Failure::WindowTooSmall);
        }
        if (prev_area - ((x1 - x0) * (y1 - y0)) as f64).abs() < EPS {
            return Err(Failure::WindowShrink);
        }
    }
    if x1 - x0 < 16 || y1 - y0 < 16 {
        return Err(Failure::WindowTooSmall);
    }
    Ok([x0, y0, x1, y1])
}

// --- the cross power spectrum -------------------------------------------------------------------

/// A window of one image.
#[derive(Clone)]
struct Win {
    d: Vec<f32>,
    h: usize,
    w: usize,
}

impl Win {
    /// `im[y0:y1, x0:x1]` as numpy takes it (the bounds clamped to the window).
    fn clip(&self, y0: i64, y1: i64, x0: i64, x1: i64) -> Win {
        let cl = |v: i64, n: usize| v.clamp(0, n as i64) as usize;
        let (y0, y1, x0, x1) = (cl(y0, self.h), cl(y1, self.h), cl(x0, self.w), cl(x1, self.w));
        let (h, w) = (y1.saturating_sub(y0), x1.saturating_sub(x0));
        let mut d = Vec::with_capacity(h * w);
        for r in y0..y0 + h {
            d.extend_from_slice(&self.d[r * self.w + x0..r * self.w + x0 + w]);
        }
        Win { d, h, w }
    }
}

/// The shifted cross power spectrum (peak at `n / 2`), a square of side `n`.
struct Scps {
    d: Vec<f32>,
    n: usize,
}

/// `_shrink_winsize_to_binarySize`: the largest power of 2 in 8..=8192 not above each dimension; `(rows, cols)`.
fn binary_size(h: usize, w: usize) -> Option<(usize, usize)> {
    let big = |v: usize| (3..=13).map(|i| 1usize << i).filter(|&s| s <= v).max();
    Some((big(h)?, big(w)?))
}

/// In-place 2-D FFT of an `h x w` row-major array, forward or inverse (not scaled).
fn fft2(buf: &mut [Complex<f32>], h: usize, w: usize, inverse: bool, planner: &mut FftPlanner<f32>) {
    let rows = if inverse {
        planner.plan_fft_inverse(w)
    } else {
        planner.plan_fft_forward(w)
    };
    for row in buf.chunks_exact_mut(w) {
        rows.process(row);
    }
    let cols = if inverse {
        planner.plan_fft_inverse(h)
    } else {
        planner.plan_fft_forward(h)
    };
    let mut t = vec![Complex::new(0.0f32, 0.0); h * w];
    for r in 0..h {
        for c in 0..w {
            t[c * h + r] = buf[r * w + c];
        }
    }
    for col in t.chunks_exact_mut(h) {
        cols.process(col);
    }
    for r in 0..h {
        for c in 0..w {
            buf[r * w + c] = t[c * h + r];
        }
    }
}

/// `_calc_shifted_cross_power_spectrum` (binary window sizes, quadratic window: AROSICS' defaults).
fn calc_scps(im0: &Win, im1: &Win, planner: &mut FftPlanner<f32>) -> Option<Scps> {
    debug_assert!(im0.h == im1.h && im0.w == im1.w);
    let (ty, tx) = binary_size(im0.h, im0.w)?;
    let n = ty.min(tx);
    // int(center - n / 2) .. int(center + n / 2), centre = shape / 2
    let (cy, cx) = (im0.h as f64 / 2.0, im0.w as f64 / 2.0);
    let (ymin, xmin) = ((cy - n as f64 / 2.0) as usize, (cx - n as f64 / 2.0) as usize);
    let crop = |im: &Win| -> Vec<Complex<f32>> {
        let mut v = Vec::with_capacity(n * n);
        for r in ymin..ymin + n {
            for c in xmin..xmin + n {
                v.push(Complex::new(im.d[r * im.w + c], 0.0));
            }
        }
        v
    };
    let mut f0 = crop(im0);
    let mut f1 = crop(im1);
    fft2(&mut f0, n, n, false, planner);
    fft2(&mut f1, n, n, false, planner);
    let hyp = |z: Complex<f32>| z.re.hypot(z.im);
    let mut amax = f32::NEG_INFINITY;
    for &z in &f1 {
        let a = hyp(z);
        if a > amax || a.is_nan() {
            amax = a;
        }
    }
    let eps = amax * 1e-15f32;
    let mut t: Vec<Complex<f32>> = f0
        .iter()
        .zip(&f1)
        .map(|(&a, &b)| {
            // numpy's complex64: a * conj(b), divided by the real denominator as a complex number (Smith's
            // algorithm, which for a zero imaginary part multiplies by the reciprocal).
            let num = Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im);
            let den = hyp(a) * hyp(b) + eps;
            if den == 0.0 {
                Complex::new(num.re / den, num.im / den)
            } else {
                let scl = 1.0 / den;
                Complex::new(num.re * scl, num.im * scl)
            }
        })
        .collect();
    fft2(&mut t, n, n, true, planner);
    let scale = 1.0 / (n * n) as f32;
    let mut d = vec![0.0f32; n * n];
    let half = n / 2;
    for r in 0..n {
        for c in 0..n {
            let z = t[r * n + c];
            let m = (z.re * scale).hypot(z.im * scale);
            d[((r + half) % n) * n + (c + half) % n] = m;
        }
    }
    Some(Scps { d, n })
}

/// numpy's `argmax` of a 2-D array: the first maximum, the first NaN if there is one; `(row, column)`.
fn peak(s: &Scps) -> (usize, usize) {
    let mut best = 0usize;
    for (i, &v) in s.d.iter().enumerate() {
        if v.is_nan() {
            best = i;
            break;
        }
        if v > s.d[best] {
            best = i;
        }
    }
    (best / s.n, best % s.n)
}

/// `_get_shifts_from_peakpos`: `(x, y)`.
fn shifts_from_peak(s: &Scps) -> (i64, i64) {
    let (r, c) = peak(s);
    (c as i64 - (s.n / 2) as i64, r as i64 - (s.n / 2) as i64)
}

/// `_get_grossly_deshifted_images`: the reference's window moved by the integer shift (virtually) and the
/// target's window of the same size at the centre.
fn grossly_deshifted(im0: &Win, im1: &Win, xs: i64, ys: i64) -> (Win, Win) {
    let (cy, cx) = (im0.h as f64 / 2.0 + ys as f64, im0.w as f64 / 2.0 + xs as f64);
    let wx = 2.0 * cx.min(im0.w as f64 - cx);
    let wy = 2.0 * cy.min(im0.h as f64 - cy);
    let side = wx.min(wy); // force_quadratic_win
    let clip = |im: &Win, cy: f64, cx: f64, wy: f64, wx: f64| {
        // numpy's int() truncates towards zero
        im.clip(
            (cy - wy / 2.0) as i64,
            (cy + wy / 2.0) as i64,
            (cx - wx / 2.0) as i64,
            (cx + wx / 2.0) as i64,
        )
    };
    let g0 = clip(im0, cy, cx, side, side);
    let c1 = clip(im1, im1.h as f64 / 2.0, im1.w as f64 / 2.0, g0.h as f64, g0.w as f64);
    (g0, c1)
}

/// `_validate_integer_shifts`: (valid, the shift found, its spectrum).
fn validate_integer(
    im0: &Win,
    im1: &Win,
    xs: i64,
    ys: i64,
    planner: &mut FftPlanner<f32>,
) -> (bool, Option<(i64, i64)>, Option<Scps>) {
    if (xs, ys) == (0, 0) {
        return (true, Some((0, 0)), None);
    }
    let (g0, c1) = grossly_deshifted(im0, im1, xs, ys);
    if g0.h == 0 || g0.w == 0 || c1.h != g0.h || c1.w != g0.w {
        return (false, None, None);
    }
    match calc_scps(&g0, &c1, planner) {
        None => (false, None, None),
        Some(s) => {
            if s.n < 3 {
                return (false, None, None);
            }
            let (x, y) = shifts_from_peak(&s);
            if (x, y) == (0, 0) {
                (true, Some((0, 0)), Some(s))
            } else {
                (false, Some((x, y)), Some(s))
            }
        }
    }
}

/// `_calc_subpixel_shifts`: from the side maxima of the peak, in single precision as AROSICS' arrays are.
fn subpixel(s: &Scps) -> Option<(f32, f32)> {
    let (r, c) = peak(s);
    if r == 0 || c == 0 || r + 1 >= s.n || c + 1 >= s.n {
        return None;
    }
    let at = |r: usize, c: usize| s.d[r * s.n + c];
    let mut mx = f32::NEG_INFINITY;
    for &v in &s.d {
        if v > mx || v.is_nan() {
            mx = v;
            if v.is_nan() {
                break;
            }
        }
    }
    let (left, right) = (at(r, c - 1), at(r, c + 1));
    let (above, below) = (at(r - 1, c), at(r + 1, c));
    let (vx, fx) = (left.max(right), if left > right { -1.0f32 } else { 1.0 });
    let (vy, fy) = (above.max(below), if above > below { -1.0f32 } else { 1.0 });
    Some(((fx * vx) / (mx + vx), (fy * vy) / (mx + vy)))
}

/// `_calc_shift_reliability`: 100 - (mean + 2 sd of the spectrum without the 3 x 3 around its peak) / (mean of
/// the 3 x 3) * 100, clipped to 0..100.
fn reliability(s: &Scps) -> f64 {
    let (r, c) = peak(s);
    let (mut sum_peak, mut n_peak) = (0.0f64, 0usize);
    let (mut sum, mut sum2, mut n) = (0.0f64, 0.0f64, 0usize);
    for rr in 0..s.n {
        for cc in 0..s.n {
            let v = s.d[rr * s.n + cc] as f64;
            if rr + 1 >= r && rr <= r + 1 && cc + 1 >= c && cc <= c + 1 {
                sum_peak += v;
                n_peak += 1;
            } else {
                sum += v;
                sum2 += v * v;
                n += 1;
            }
        }
    }
    let power_at_peak = sum_peak / n_peak as f64;
    let mean = sum / n as f64;
    let sd = (sum2 / n as f64 - mean * mean).max(0.0).sqrt();
    let confid = 100.0 - ((mean + 2.0 * sd) / power_at_peak) * 100.0;
    if confid > 100.0 {
        100.0
    } else if confid < 0.0 || confid.is_nan() {
        0.0
    } else {
        confid
    }
}

// --- structural similarity ----------------------------------------------------------------------

/// `_compute_ssim`: the windows scaled to 0..1 by their values' range over the cells valid in both, then
/// scikit-image's `structural_similarity` (7 x 7 uniform window, data range 1, K1 0.01, K2 0.03, sample
/// covariance), averaged over the cells whose window is whole.
fn ssim(a: &[f32], b: &[f32], ok: &[bool], h: usize, w: usize) -> f64 {
    let norm = |v: &[f32]| -> Option<Vec<f64>> {
        let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
        for (&x, &o) in v.iter().zip(ok) {
            if o {
                mn = mn.min(x);
                mx = mx.max(x);
            }
        }
        if mn > mx {
            return None;
        }
        if mx == mn {
            mx += 1e-5; // avoids the zero division
        }
        Some(v.iter().map(|&x| ((x - mn) / (mx - mn)) as f64).collect())
    };
    let (Some(x), Some(y)) = (norm(a), norm(b)) else {
        return f64::NAN;
    };
    const WIN: usize = 7;
    if h < WIN || w < WIN {
        return f64::NAN;
    }
    let np = (WIN * WIN) as f64;
    let cov_norm = np / (np - 1.0);
    let (c1, c2) = ((0.01f64).powi(2), (0.03f64).powi(2));
    // box sums over the 7 x 7 windows: horizontal then vertical
    let boxsum = |v: &dyn Fn(usize) -> f64| -> Vec<f64> {
        let ow = w - WIN + 1;
        let mut tmp = vec![0.0f64; h * ow];
        for r in 0..h {
            for c in 0..ow {
                let mut s = 0.0;
                for k in 0..WIN {
                    s += v(r * w + c + k);
                }
                tmp[r * ow + c] = s;
            }
        }
        let oh = h - WIN + 1;
        let mut out = vec![0.0f64; oh * ow];
        for r in 0..oh {
            for c in 0..ow {
                let mut s = 0.0;
                for k in 0..WIN {
                    s += tmp[(r + k) * ow + c];
                }
                out[r * ow + c] = s / np;
            }
        }
        out
    };
    let ux = boxsum(&|i| x[i]);
    let uy = boxsum(&|i| y[i]);
    let uxx = boxsum(&|i| x[i] * x[i]);
    let uyy = boxsum(&|i| y[i] * y[i]);
    let uxy = boxsum(&|i| x[i] * y[i]);
    let mut total = 0.0f64;
    for k in 0..ux.len() {
        let vx = cov_norm * (uxx[k] - ux[k] * ux[k]);
        let vy = cov_norm * (uyy[k] - uy[k] * uy[k]);
        let vxy = cov_norm * (uxy[k] - ux[k] * uy[k]);
        let a1 = 2.0 * ux[k] * uy[k] + c1;
        let a2 = 2.0 * vxy + c2;
        let b1 = ux[k] * ux[k] + uy[k] * uy[k] + c1;
        let b2 = vx + vy + c2;
        total += (a1 * a2) / (b1 * b2);
    }
    total / ux.len() as f64
}

/// GDAL's cubic weights (`GWKCubicComputeWeights`).
fn cubic_weights(x: f64) -> [f64; 4] {
    let half_x = 0.5 * x;
    let three_x = 3.0 * x;
    let half_x2 = half_x * x;
    [
        half_x * (-1.0 + x * (2.0 - x)),
        1.0 + half_x2 * (-5.0 + three_x),
        half_x * (1.0 + x * (4.0 - three_x)),
        half_x2 * (-1.0 + x),
    ]
}

/// The image at `(x, y)`, pixel coordinates (the corner of the first pixel is the origin), by GDAL's cubic
/// resampling: the 4 x 4 cells when all are inside and valid, else the bilinear one of the valid cells among the
/// 2 x 2 around (`GWKCubicResample4Sample`, `GWKBilinearResample4Sample`); None outside the image or without
/// valid cells.
fn sample_cubic(im: &Image, x: f64, y: f64) -> Option<f64> {
    let (nx, ny) = (im.nx as i64, im.ny as i64);
    if !(x >= 0.0 && y >= 0.0 && x < nx as f64 && y < ny as f64) {
        return None;
    }
    let at = |r: i64, c: i64| (r as usize) * im.nx + c as usize;
    let isx = (x - 0.5).floor() as i64;
    let isy = (y - 0.5).floor() as i64;
    if isx > 0 && isx + 2 < nx && isy > 0 && isy + 2 < ny {
        let mut all = true;
        'chk: for r in isy - 1..=isy + 2 {
            for c in isx - 1..=isx + 2 {
                if !im.valid[at(r, c)] {
                    all = false;
                    break 'chk;
                }
            }
        }
        if all {
            let wx = cubic_weights(x - 0.5 - isx as f64);
            let wy = cubic_weights(y - 0.5 - isy as f64);
            let mut rows = [0.0f64; 4];
            for (i, row) in rows.iter_mut().enumerate() {
                let r = isy - 1 + i as i64;
                let mut s = 0.0;
                for (k, w) in wx.iter().enumerate() {
                    s += w * im.data[at(r, isx - 1 + k as i64)] as f64;
                }
                *row = s;
            }
            return Some(wy[0] * rows[0] + wy[1] * rows[1] + wy[2] * rows[2] + wy[3] * rows[3]);
        }
    }
    // bilinear
    let (mut bx, mut by) = (isx, isy);
    let mut ratio_x = 1.5 - (x - bx as f64);
    let mut ratio_y = 1.5 - (y - by as f64);
    if bx == -1 {
        bx = 0;
        ratio_x = 1.0;
    }
    if by == -1 {
        by = 0;
        ratio_y = 1.0;
    }
    let m = [
        ratio_x * ratio_y,
        (1.0 - ratio_x) * ratio_y,
        ratio_x * (1.0 - ratio_y),
        (1.0 - ratio_x) * (1.0 - ratio_y),
    ];
    let (mut acc, mut div) = (0.0f64, 0.0f64);
    for (q, &mq) in m.iter().enumerate() {
        let (r, c) = (by + (q as i64) / 2, bx + (q as i64) % 2);
        if r >= ny || c >= nx || !im.valid[at(r, c)] {
            continue;
        }
        div += mq;
        acc += im.data[at(r, c)] as f64 * mq;
    }
    if div < 0.00001 {
        None
    } else {
        Some(acc / div)
    }
}

// --- the whole ------------------------------------------------------------------------------------

/// The shift of `target` to `reference` (the images on the same grid): AROSICS' `COREG.calculate_spatial_shifts`.
pub fn coregister(reference: &Image, target: &Image, p: &Params) -> Outcome {
    assert!(
        reference.ny == target.ny && reference.nx == target.nx,
        "the images must have the same shape"
    );
    assert!(
        reference.data.len() == reference.ny * reference.nx
            && reference.valid.len() == reference.data.len()
            && target.data.len() == reference.data.len()
            && target.valid.len() == reference.data.len(),
        "data and valid must have ny * nx values"
    );
    let ov = Overlap::new(reference, target);
    let bx = match match_box(&ov, p) {
        Ok(b) => b,
        Err(f) => return Outcome::failed(f),
    };
    // Odd rows or columns lose the last one.
    let (mut h, mut w) = ((bx[3] - bx[1]) as usize, (bx[2] - bx[0]) as usize);
    h -= h % 2;
    w -= w % 2;
    let (x0, y0) = (bx[0] as usize, bx[1] as usize);
    let cut = |im: &Image| -> (Win, Vec<bool>) {
        let mut d = Vec::with_capacity(h * w);
        let mut ok = Vec::with_capacity(h * w);
        for r in y0..y0 + h {
            d.extend_from_slice(&im.data[r * im.nx + x0..r * im.nx + x0 + w]);
            ok.extend_from_slice(&im.valid[r * im.nx + x0..r * im.nx + x0 + w]);
        }
        (Win { d, h, w }, ok)
    };
    let (im0, ok0) = cut(reference);
    let (im1, ok1) = cut(target);
    let window = [x0, y0, x0 + w, y0 + h];
    let fail = |f: Failure, it: usize| {
        let mut o = Outcome::failed(f);
        o.window = window;
        o.iterations = it;
        o
    };

    let mut planner = FftPlanner::<f32>::new();
    let Some(scps0) = calc_scps(&im0, &im1, &mut planner) else {
        return fail(Failure::WindowTooSmall, 0);
    };
    let (mut x_int, mut y_int) = shifts_from_peak(&scps0);
    let mut iterations = 1usize;
    let scps: Scps = if (x_int, y_int) == (0, 0) {
        scps0
    } else {
        let (mut valid, mut found, mut sc) = validate_integer(&im0, &im1, x_int, y_int, &mut planner);
        let mut count = 1usize;
        while !valid {
            count += 1;
            if count > p.max_iter {
                return fail(Failure::NoMatch, count - 1);
            }
            let Some((xv, yv)) = found else {
                return fail(Failure::WindowTooSmall, count - 1);
            };
            (valid, found, sc) = validate_integer(&im0, &im1, xv, yv, &mut planner);
            if valid {
                // AROSICS overwrites the integer shift with the one the validation returns, which is (0, 0)
                // for a valid one.
                (x_int, y_int) = (0, 0);
            }
        }
        iterations = count;
        match sc {
            Some(s) => s,
            None => return fail(Failure::NoPeak, iterations),
        }
    };

    // the side of the spectrum the shift is read from (AROSICS' `fft_win_size_YX` after its last correlation)
    let fft_size = scps.n;
    let Some((xs, ys)) = subpixel(&scps) else {
        return fail(Failure::NoPeak, iterations);
    };
    let (xs, ys) = (xs as f64, ys as f64);
    let (xt, yt) = (x_int as f64 + xs, y_int as f64 + ys);
    if !(xt.abs().max(yt.abs()) <= p.max_shift) {
        let mut o = fail(Failure::ShiftTooLarge, iterations);
        o.fft_size = fft_size;
        o.x_shift = xt;
        o.y_shift = yt;
        return o;
    }

    let rel = reliability(&scps);
    let before = ssim(&im0.d, &im1.d, &both(&ok0, &ok1), h, w);
    // The target at the shifted position, on the window's cells (GDAL warps the target with its origin moved by
    // the shift onto the window's grid).
    let mut dd = vec![p.fill; h * w];
    let mut okd = vec![false; h * w];
    for r in 0..h {
        for c in 0..w {
            let sx = (x0 + c) as f64 + 0.5 - xt;
            let sy = (y0 + r) as f64 + 0.5 - yt;
            if let Some(v) = sample_cubic(target, sx, sy) {
                let v = match p.integer_range {
                    Some((lo, hi)) => {
                        let r = (v + 0.5).floor().clamp(lo, hi);
                        match p.nodata {
                            Some(nd) if r == nd => {
                                if nd == lo {
                                    nd + 1.0
                                } else {
                                    nd - 1.0
                                }
                            }
                            _ => r,
                        }
                    }
                    None => v,
                };
                dd[r * w + c] = v as f32;
                okd[r * w + c] = true;
            }
        }
    }
    let after = ssim(&im0.d, &dd, &both(&ok0, &okd), h, w);
    Outcome {
        success: true,
        failure: None,
        x_shift: xt,
        y_shift: yt,
        x_int,
        y_int,
        x_sub: xs,
        y_sub: ys,
        reliability: rel,
        ssim_before: before,
        ssim_after: after,
        window,
        fft_size,
        iterations,
    }
}

fn both(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(&p, &q)| p && q).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured scene: value noise of 5 octaves (lattices of 2 to 32 pixels, cubic Hermite between the nodes),
    /// continuous, so that a scene moved by a fraction of a pixel is exact; `at(x, y)` in pixel coordinates.
    struct Scene(u64);

    impl Scene {
        fn new(seed: u64) -> Scene {
            Scene(seed)
        }
        fn node(&self, octave: u64, ix: i64, iy: i64) -> f64 {
            let mut h = self.0 ^ (octave.wrapping_mul(0x9E3779B97F4A7C15));
            h ^= (ix as u64).wrapping_mul(0xBF58476D1CE4E5B9);
            h = h.rotate_left(27).wrapping_mul(0x94D049BB133111EB);
            h ^= (iy as u64).wrapping_mul(0xD6E8FEB86659FD93);
            h ^= h >> 31;
            h = h.wrapping_mul(0x9E3779B97F4A7C15);
            h ^= h >> 29;
            (h >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }
        fn at(&self, x: f64, y: f64) -> f32 {
            let mut v = 0.0;
            for o in 0..5u64 {
                let period = (2u64 << o) as f64;
                let (u, w) = (x / period, y / period);
                let (iu, iw) = (u.floor(), w.floor());
                let (fu, fw) = (u - iu, w - iw);
                let (su, sw) = (fu * fu * (3.0 - 2.0 * fu), fw * fw * (3.0 - 2.0 * fw));
                let n = |dx: i64, dy: i64| self.node(o, iu as i64 + dx, iw as i64 + dy);
                let top = n(0, 0) * (1.0 - su) + n(1, 0) * su;
                let bot = n(0, 1) * (1.0 - su) + n(1, 1) * su;
                v += (top * (1.0 - sw) + bot * sw) * period.sqrt();
            }
            (3000.0 + 300.0 * v) as f32
        }
        /// The scene at pixel centres, moved by `(dx, dy)` pixels (content to the right and down for positive).
        fn image(&self, ny: usize, nx: usize, dx: f64, dy: f64) -> Vec<f32> {
            let mut v = Vec::with_capacity(ny * nx);
            for r in 0..ny {
                for c in 0..nx {
                    v.push(self.at(c as f64 + 0.5 - dx, r as f64 + 0.5 - dy));
                }
            }
            v
        }
    }

    fn run(ref_d: &[f32], tgt_d: &[f32], vr: &[bool], vt: &[bool], ny: usize, nx: usize, p: &Params) -> Outcome {
        coregister(
            &Image {
                data: ref_d,
                valid: vr,
                ny,
                nx,
            },
            &Image {
                data: tgt_d,
                valid: vt,
                ny,
                nx,
            },
            p,
        )
    }

    #[test]
    fn integer_and_subpixel_shifts_are_recovered() {
        let (ny, nx) = (400, 500);
        let sc = Scene::new(1);
        let all = vec![true; ny * nx];
        let reference = sc.image(ny, nx, 0.0, 0.0);
        for &(dx, dy, tol) in &[(3.0, -2.0, 0.12), (0.4, 1.7, 0.15), (-4.0, 4.0, 0.12), (0.0, 0.0, 0.1)] {
            // the target is the reference moved by (-dx, -dy): to align it, move it by (dx, dy)
            let target = sc.image(ny, nx, -dx, -dy);
            let o = run(&reference, &target, &all, &all, ny, nx, &Params::default());
            assert!(o.success, "({dx}, {dy}): {:?}", o.failure);
            assert!(
                (o.x_shift - dx).abs() < tol && (o.y_shift - dy).abs() < tol,
                "({dx}, {dy}) found ({}, {})",
                o.x_shift,
                o.y_shift
            );
            assert!(o.reliability > 20.0, "reliability {}", o.reliability);
            assert!(
                o.ssim_after >= o.ssim_before - 1e-9,
                "ssim {} -> {}",
                o.ssim_before,
                o.ssim_after
            );
            // the spectrum is that of the window shifted by the integer part: 250 pixels give 128
            assert_eq!(o.fft_size, if (dx, dy) == (0.0, 0.0) { 256 } else { 128 });
        }
    }

    #[test]
    fn integer_part_is_the_integer_shift() {
        let (ny, nx) = (300, 300);
        let sc = Scene::new(2);
        let all = vec![true; ny * nx];
        let o = run(
            &sc.image(ny, nx, 0.0, 0.0),
            &sc.image(ny, nx, -3.0, 2.0),
            &all,
            &all,
            ny,
            nx,
            &Params::default(),
        );
        assert!(o.success);
        assert_eq!((o.x_int, o.y_int), (3, -2));
        assert!(o.x_sub.abs() < 0.1 && o.y_sub.abs() < 0.1);
    }

    #[test]
    fn nodata_margins_are_left_out_of_the_window() {
        let (ny, nx) = (400, 400);
        let sc = Scene::new(3);
        let reference = sc.image(ny, nx, 0.0, 0.0);
        let target = sc.image(ny, nx, -2.0, -1.0);
        // the target has no data on its left 150 columns and bottom 100 rows
        let vt: Vec<bool> = (0..ny * nx).map(|k| k % nx >= 150 && k / nx < 300).collect();
        let vr = vec![true; ny * nx];
        let o = run(&reference, &target, &vr, &vt, ny, nx, &Params::default());
        assert!(o.success, "{:?}", o.failure);
        assert!(
            (o.x_shift - 2.0).abs() < 0.12 && (o.y_shift - 1.0).abs() < 0.12,
            "({}, {})",
            o.x_shift,
            o.y_shift
        );
        let [x0, y0, x1, y1] = o.window;
        assert!(x0 >= 150 && y1 <= 300 && x1 <= nx && y0 < y1, "window {:?}", o.window);
    }

    #[test]
    fn window_of_a_small_image_is_one_pixel_smaller_a_side() {
        // 200 x 200 < the 256 window: AROSICS' other window (the unrounded 256 box) is not within the overlap, so
        // the matching window loses a pixel on each side before the FFT takes 128 of its 198.
        let n = 200;
        let sc = Scene::new(4);
        let all = vec![true; n * n];
        let im = sc.image(n, n, 0.0, 0.0);
        let o = run(&im, &im, &all, &all, n, n, &Params::default());
        assert!(o.success);
        assert_eq!(o.window, [1, 1, 199, 199]);
        assert_eq!(o.fft_size, 128);
        assert!(o.x_shift.abs() < 1e-3 && o.y_shift.abs() < 1e-3);
    }

    #[test]
    fn a_featureless_scene_fails() {
        let (ny, nx) = (300, 300);
        let flat = vec![100.0f32; ny * nx];
        let all = vec![true; ny * nx];
        let o = run(&flat, &flat, &all, &all, ny, nx, &Params::default());
        assert!(!o.success, "found ({}, {})", o.x_shift, o.y_shift);
    }

    #[test]
    fn errors() {
        let (ny, nx) = (100, 100);
        let sc = Scene::new(5);
        let im = sc.image(ny, nx, 0.0, 0.0);
        let all = vec![true; ny * nx];
        let none = vec![false; ny * nx];
        assert_eq!(
            run(&im, &im, &all, &none, ny, nx, &Params::default()).failure,
            Some(Failure::NoOverlap)
        );
        let tiny: Vec<bool> = (0..ny * nx).map(|k| k % nx < 2).collect();
        assert_eq!(
            run(&im, &im, &all, &tiny, ny, nx, &Params::default()).failure,
            Some(Failure::SmallOverlap)
        );
        let p = Params {
            window_pos: Some((-5.0, 50.0)),
            ..Params::default()
        };
        assert_eq!(
            run(&im, &im, &all, &all, ny, nx, &p).failure,
            Some(Failure::WindowPosition)
        );
        // above max_shift
        let big = sc.image(ny, nx, -8.0, 0.0);
        let p = Params {
            window_size: 64,
            ..Params::default()
        };
        let o = run(&im, &big, &all, &all, ny, nx, &p);
        assert!(
            !o.success && o.failure == Some(Failure::ShiftTooLarge),
            "{:?}",
            o.failure
        );
    }

    #[test]
    fn pieces() {
        assert_eq!(binary_size(198, 300), Some((128, 256)));
        assert_eq!(binary_size(7, 300), None);
        // numpy's argmax: the first maximum, the first NaN
        let s = Scps {
            d: vec![1.0, 5.0, 5.0, f32::NAN, 9.0, f32::NAN, 0.0, 0.0, 0.0],
            n: 3,
        };
        assert_eq!(peak(&s), (1, 0));
        let s = Scps {
            d: vec![1.0, 5.0, 5.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0],
            n: 3,
        };
        assert_eq!(peak(&s), (0, 1));
        // the cubic at a pixel centre is the pixel
        let d: Vec<f32> = (0..64).map(|k| (k * k % 17) as f32).collect();
        let v = vec![true; 64];
        let im = Image {
            data: &d,
            valid: &v,
            ny: 8,
            nx: 8,
        };
        assert!((sample_cubic(&im, 3.5, 4.5).unwrap() - d[4 * 8 + 3] as f64).abs() < 1e-12);
        assert!(sample_cubic(&im, 8.0, 1.0).is_none());
        // identical windows have a similarity of 1
        let a: Vec<f32> = (0..400).map(|k| ((k * 7919) % 101) as f32).collect();
        let ok = vec![true; 400];
        assert!((ssim(&a, &a, &ok, 20, 20) - 1.0).abs() < 1e-12);
    }
}
