//! The warp kernel of load_raster(..., like=): a port of src/warp.cpp, itself
//! GDAL 3.12's alg/gdalwarpkernel.cpp (MIT, Frank Warmerdam, Even Rouault and
//! others) on the code paths GDAL takes for a destination with a NoData value
//! and UNIFIED_SRC_NODATA, with the same operations in the same order, so the
//! same bits as both.
//!
//! GDAL's resampling runs on the source coordinates of a fixed set of
//! destination points (exact ones, or interpolated in a lattice anchored at the
//! destination's origin), so that any region of the destination gets the same
//! values whoever computes it. Coordinates are the whole source's (the window
//! given is only used to index the cells it holds). With `per_band`, every band
//! has a validity mask of its own (the dates of a time series have clouds of
//! their own) instead of GDAL's unified one.

use crate::parallel;
use std::cell::RefCell;
use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;

mod kernels;
mod loops;
mod raster;

pub use raster::Raster;
use raster::{cmax, cmin, NoDataTest};

// GDAL's GDALResampleAlg numbers.
pub const NEAREST: i32 = 0;
pub const BILINEAR: i32 = 1;
pub const CUBIC: i32 = 2;
pub const CUBICSPLINE: i32 = 3;
pub const LANCZOS: i32 = 4;
pub const AVERAGE: i32 = 5;
pub const MODE: i32 = 6;
pub const MAX: i32 = 8;
pub const MIN: i32 = 9;
pub const MED: i32 = 10;
pub const Q1: i32 = 11;
pub const Q3: i32 = 12;
pub const RMS: i32 = 14;

/// A method of GDAL's this kernel has.
pub fn valid_method(method: i32) -> bool {
    (NEAREST..=MODE).contains(&method) || (MAX..=Q3).contains(&method) || method == RMS
}

/// True for the methods that take the source cells a destination cell covers
/// (their samples are the cells' corners, not their centres).
pub fn by_area(method: i32) -> bool {
    method >= AVERAGE
}

/// Samples a side of a lattice cell.
pub const CELL: i64 = 16;

/// GDAL's GDALTransformerFunc (GDALGenImgProjTransform): pixel coordinates of
/// the destination to the source's (dst_to_src) or back, in place; success[k]
/// false where the transformation fails.
pub type TransformerFunc = unsafe extern "C" fn(
    arg: *mut c_void,
    dst_to_src: c_int,
    n: c_int,
    x: *mut f64,
    y: *mut f64,
    z: *mut f64,
    success: *mut c_int,
) -> c_int;

/// The lattice's interpolation between the nodes around a sample, at fractions
/// fa (along x) and fb (along y) of the way: along y on the left and right
/// edges, then along x.
#[inline(always)]
pub fn lattice_interp(p00: f64, p10: f64, p01: f64, p11: f64, fa: f64, fb: f64) -> f64 {
    let left = p00 * (1.0 - fb) + p01 * fb;
    let right = p10 * (1.0 - fb) + p11 * fb;
    left * (1.0 - fa) + right * fa
}

// The kinds of coordinates.
pub const AFFINE: i32 = 0;
pub const LATTICE: i32 = 1;
pub const DENSE: i32 = 2;
pub const EXACT: i32 = 3;

/// The source pixel coordinates (column, row; 0 at the source's top-left
/// corner) of the destination samples. Sample (i, j) is the destination point
/// (i + offset, j + offset) in destination pixel coordinates: the cell centres
/// (offset 0.5) or the cell corners (offset 0, one more sample a side).
#[derive(Clone, Copy, Default)]
pub struct Coords<'a> {
    pub kind: i32,
    pub offset: f64,
    /// AFFINE (the same CRS): the destination geotransform and the inverse of
    /// the source's, applied as GDAL's GDALGenImgProjTransform does.
    pub gt: [f64; 6],
    pub inv: [f64; 6],
    /// LATTICE: cells of CELL x CELL samples, the cells_x x cells_y from cell
    /// (cell_x0, cell_y0) on. Cell (cx, cy), number
    /// c = (cy - cell_y0) * cells_x + cx - cell_x0, has its nodes every
    /// CELL >> level[c] samples: at level 0 the corners, in grid0
    /// ((cells_y + 1) x (cells_x + 1) x, y pairs, every CELL samples), else an
    /// (n + 1) x (n + 1) block of x, y pairs at nodes[2 * node_off[c]..]. The
    /// samples between nodes are interpolated (lattice_interp).
    pub cells_x: i64,
    pub cells_y: i64,
    pub cell_x0: i64,
    pub cell_y0: i64,
    pub level: &'a [i8],
    pub grid0: &'a [f64],
    pub node_off: &'a [i64],
    pub nodes: &'a [f64],
    /// DENSE: x, y pairs of every sample of rows dense_row0.., columns
    /// dense_col0.. (dense_stride samples a row).
    pub dense: &'a [f64],
    pub dense_row0: i64,
    pub dense_col0: i64,
    pub dense_stride: i64,
    /// The exact transformation (GDAL's transformer, one argument per thread):
    /// the samples of EXACT coordinates, and the points GDAL transforms on
    /// demand (near the edge of a projection's domain, across the
    /// antimeridian); without it those checks are left out.
    pub transformer: Option<TransformerFunc>,
    pub transformer_args: &'a [usize],
}

thread_local! {
    static Z: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static OK: RefCell<Vec<c_int>> = const { RefCell::new(Vec::new()) };
}

impl Coords<'_> {
    /// Transforms the points in place; false where it fails (or no
    /// transformer). Each thread of the pool calls the transformer with an
    /// argument of its own (the calling thread with the first).
    pub fn transform(&self, dst_to_src: bool, x: &mut [f64], y: &mut [f64], ok: &mut [c_int]) {
        let n = x.len();
        let f = match self.transformer {
            Some(f) if !self.transformer_args.is_empty() => f,
            _ => {
                ok[..n].fill(0);
                return;
            }
        };
        let t = parallel::thread_index();
        let arg = self.transformer_args[t % self.transformer_args.len()];
        assert!(y.len() == n && ok.len() >= n);
        Z.with(|z| {
            let mut z = z.borrow_mut();
            z.clear();
            z.resize(n, 0.0);
            // SAFETY: GDAL's transformer reads and writes n values of x, y, z and
            // success; its argument is this thread's alone.
            unsafe {
                f(
                    arg as *mut c_void,
                    c_int::from(dst_to_src),
                    n as c_int,
                    x.as_mut_ptr(),
                    y.as_mut_ptr(),
                    z.as_mut_ptr(),
                    ok.as_mut_ptr(),
                );
            }
        });
    }

    /// x, y of the samples (i0 .. i0 + x.len() - 1, j).
    pub fn row(&self, j: i64, i0: i64, x: &mut [f64], y: &mut [f64]) {
        let n = x.len();
        let y = &mut y[..n];
        match self.kind {
            AFFINE => {
                // GDALGenImgProjTransform without reprojection.
                let (gt, inv) = (&self.gt, &self.inv);
                let v = j as f64 + self.offset;
                for k in 0..n {
                    let u = (i0 + k as i64) as f64 + self.offset;
                    let gx = gt[0] + u * gt[1] + v * gt[2];
                    let gy = gt[3] + u * gt[4] + v * gt[5];
                    x[k] = inv[0] + gx * inv[1] + gy * inv[2];
                    y[k] = inv[3] + gx * inv[4] + gy * inv[5];
                }
            }
            EXACT => {
                // GWKRealCase / GWKAverageOrModeComputeLineCoords: the samples of a row at once.
                let v = j as f64 + self.offset;
                for k in 0..n {
                    x[k] = (i0 + k as i64) as f64 + self.offset;
                    y[k] = v;
                }
                OK.with(|ok| {
                    let mut ok = ok.borrow_mut();
                    ok.resize(n, 0);
                    self.transform(true, x, y, &mut ok);
                    for k in 0..n {
                        if ok[k] == 0 {
                            x[k] = f64::INFINITY;
                            y[k] = f64::INFINITY;
                        }
                    }
                });
            }
            DENSE => {
                let start = 2 * ((j - self.dense_row0) * self.dense_stride + (i0 - self.dense_col0)) as usize;
                let p = &self.dense[start..start + 2 * n];
                for k in 0..n {
                    x[k] = p[2 * k];
                    y[k] = p[2 * k + 1];
                }
            }
            _ => self.lattice_row(j, i0, x, y),
        }
    }

    /// LATTICE: the samples of the row a sub-cell at a time; every sample's
    /// value depends only on its sub-cell's nodes and its fractions.
    fn lattice_row(&self, j: i64, i0: i64, x: &mut [f64], y: &mut [f64]) {
        let n = x.len();
        let cell = CELL as i32;
        let cy = j / CELL;
        let lj = (j - cy * CELL) as i32;
        let mut k = 0usize;
        while k < n {
            let i = i0 + k as i64;
            let cx = i / CELL;
            let mut li = (i - cx * CELL) as i32;
            let c = ((cy - self.cell_y0) * self.cells_x + (cx - self.cell_x0)) as usize;
            let lev = self.level[c] as i32;
            let step = cell >> lev;
            let (pts, base, stride) = if lev == 0 {
                let at = (cy - self.cell_y0) * (self.cells_x + 1) + (cx - self.cell_x0);
                (self.grid0, 2 * at as usize, (self.cells_x + 1) as usize)
            } else {
                (self.nodes, 2 * self.node_off[c] as usize, (cell / step + 1) as usize)
            };
            let inv_step = 1.0 / step as f64; // exact: steps are powers of 2
            let b = lj / step;
            let fbn = lj - b * step;
            let fb = fbn as f64 * inv_step;
            let end = n.min(k + (cell - li) as usize);
            while k < end {
                let a = li / step;
                let o00 = base + 2 * (b as usize * stride + a as usize);
                let (o10, o01) = (o00 + 2, o00 + 2 * stride);
                let o11 = o01 + 2;
                let sub_end = end.min(k + (step - (li - a * step)) as usize);
                let lx = pts[o00] * (1.0 - fb) + pts[o01] * fb;
                let rx = pts[o10] * (1.0 - fb) + pts[o11] * fb;
                let ly = pts[o00 + 1] * (1.0 - fb) + pts[o01 + 1] * fb;
                let ry = pts[o10 + 1] * (1.0 - fb) + pts[o11 + 1] * fb;
                while k < sub_end {
                    let fan = li - a * step;
                    if fan == 0 && fbn == 0 {
                        // a node: exact, whatever its neighbours
                        x[k] = pts[o00];
                        y[k] = pts[o00 + 1];
                    } else {
                        let fa = fan as f64 * inv_step;
                        x[k] = lx * (1.0 - fa) + rx * fa;
                        y[k] = ly * (1.0 - fa) + ry * fa;
                    }
                    k += 1;
                    li += 1;
                }
            }
        }
    }
}

/// What to warp: the source, its window, the method and the destination region.
#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub method: i32,
    pub bands: i32,
    /// The whole source, and the window of it that `src` holds.
    pub src_h: i32,
    pub src_w: i32,
    pub win_r0: i32,
    pub win_c0: i32,
    pub win_h: i32,
    pub win_w: i32,
    /// GDAL's dfXScale / dfYScale (destination cells per source cell).
    pub xscale: f64,
    pub yscale: f64,
    /// Source NoData (GDAL's validity mask: a cell is valid when any band is
    /// not NoData; no mask without a NoData value).
    pub has_src_nodata: bool,
    /// Every band on its own (its own validity mask and NoData avoidance), as
    /// GDAL warping each band alone; the coordinates are still computed once.
    pub per_band: bool,
    pub src_nodata: f64,
    pub dst_nodata: f64,
    /// The destination region computed: rows dst_r0.., columns dst_c0.. of
    /// the whole destination.
    pub dst_r0: i32,
    pub dst_c0: i32,
    pub dst_h: i32,
    pub dst_w: i32,
}

/// Warps src ((bands, win_h, win_w), row-major) into dst ((bands, dst_h,
/// dst_w)), the cells without a value set to dst_nodata, rows on `threads`
/// threads. With a transformer, `coords.transformer_args` holds one argument
/// per thread at least.
pub fn warp<T: Raster>(src: &[T], dst: &mut [T], coords: &Coords, sp: &Spec, threads: usize) {
    // SAFETY: MaybeUninit<T> has T's layout, and warp_uninit writes only Ts.
    let dst = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut MaybeUninit<T>, dst.len()) };
    warp_uninit(src, dst, coords, sp, threads);
}

/// `warp` into memory that holds no value yet (a new array, which C++
/// leaves uninitialized too): every cell of dst is written, its rows first
/// filled with dst_nodata by the thread that computes them.
pub fn warp_uninit<T: Raster>(src: &[T], dst: &mut [MaybeUninit<T>], coords: &Coords, sp: &Spec, threads: usize) {
    let plane = sp.win_h as usize * sp.win_w as usize;
    let dst_plane = sp.dst_h as usize * sp.dst_w as usize;
    let bands = sp.bands as usize;
    assert!(src.len() == bands * plane && dst.len() == bands * dst_plane);
    let threads = threads.max(1);

    let nodata = NoDataTest::new::<T>(sp.src_nodata);
    let mut s = kernels::Src {
        data: src,
        plane,
        h: sp.src_h,
        w: sp.src_w,
        r0: sp.win_r0,
        c0: sp.win_c0,
        wh: sp.win_h,
        ww: sp.win_w,
        masked: sp.has_src_nodata && nodata.active,
        valid: None,
        nodata,
    };
    // GDAL's unified validity mask (a cell is valid when any band is not
    // NoData). GDAL drops it when its window happens to hold no NoData; here
    // it is kept whenever there is a NoData value, so that every window gives
    // the same values.
    let mut valid = Vec::new();
    if s.masked && bands > 1 && !sp.per_band {
        valid = vec![1u8; plane];
        let fill = |i0: usize, out: &mut [u8]| {
            for (k, v) in out.iter_mut().enumerate() {
                let i = i0 + k;
                let any = (0..bands).any(|b| !nodata.test(src[b * plane + i]));
                *v = u8::from(any);
            }
        };
        // OpenMP's schedule(static): a block of cells a thread.
        let chunk = plane.div_ceil(threads).max(1);
        let shared = parallel::Shared::new(&mut valid);
        parallel::region(threads, |t| {
            let (a, b) = ((t * chunk).min(plane), ((t + 1) * chunk).min(plane));
            // SAFETY: the threads' blocks are disjoint.
            fill(a, unsafe { shared.slice_mut(a..b) });
        });
    }
    if !valid.is_empty() {
        s.valid = Some(&valid);
    }

    let d = loops::Dst {
        data: dst.as_mut_ptr() as *mut T,
        len: dst.len(),
        plane: dst_plane,
        bands: sp.bands,
        nodata: sp.dst_nodata,
    };
    // per_band: a source and a destination of one band each, the band's
    // values testing its own validity (GDAL warping the band alone).
    let (mut ps, mut pd) = (Vec::new(), Vec::new());
    if sp.per_band && bands > 1 {
        for b in 0..bands {
            let mut sb = s;
            sb.data = &src[b * plane..(b + 1) * plane];
            ps.push(sb);
            let mut db = d;
            // SAFETY: within dst, whose band b this is.
            db.data = unsafe { d.data.add(b * dst_plane) };
            db.len = dst_plane;
            db.bands = 1;
            pd.push(db);
        }
    }
    let (pps, ppd) = if ps.is_empty() {
        (None, None)
    } else {
        (Some(&ps[..]), Some(&pd[..]))
    };
    if by_area(sp.method) {
        loops::warp_areas(&s, &d, pps, ppd, coords, sp, threads);
    } else {
        loops::warp_points(&s, &d, pps, ppd, coords, sp, threads);
    }
}

/// The checks of lattice cells: g holds n grids of (2 my + 1) x (2 mx + 1)
/// x, y pairs, exact coordinates every half step of a cell (its nodes at even
/// indices). For every cell (n x my x mx): err, the largest |dx| + |dy| of
/// lattice_interp at the cell's centre and the middles of its top and left
/// edges (the other edges are its neighbours'); bad, a node or check not
/// finite; outside, the cell well outside the source (src_h x src_w), its
/// errors included; bbox, xmin, xmax, ymin, ymax of its finite corner nodes
/// (NaN when none).
#[allow(clippy::too_many_arguments)]
pub fn lattice_checks(
    g: &[f64],
    n: usize,
    my: usize,
    mx: usize,
    src_h: i32,
    src_w: i32,
    err: &mut [f64],
    bad: &mut [bool],
    outside: &mut [bool],
    bbox: &mut [f64],
    threads: usize,
) {
    let (sx, sy) = (2 * mx + 1, 2 * my + 1);
    let cells = n * my * mx;
    assert!(g.len() == n * sy * sx * 2 && err.len() == cells && bad.len() == cells);
    assert!(outside.len() == cells && bbox.len() == 4 * cells);
    let check = |c: usize, err: &mut f64, bad: &mut bool, outside: &mut bool, bbox: &mut [f64]| {
        let q = c / (my * mx);
        let (b, a) = ((c / mx) % my, c % mx);
        let grid = &g[q * sy * sx * 2..(q + 1) * sy * sx * 2];
        let pt = |r: usize, s: usize| {
            let k = 2 * ((2 * b + r) * sx + 2 * a + s);
            (grid[k], grid[k + 1])
        };
        let (p00, p10, p01, p11) = (pt(0, 0), pt(0, 2), pt(2, 0), pt(2, 2));
        let checks = [pt(1, 1), pt(0, 1), pt(1, 0)];
        const FRACTIONS: [(f64, f64); 3] = [(0.5, 0.5), (0.5, 0.0), (0.0, 0.5)];
        let mut e = 0.0;
        let mut finite = true;
        for k in 0..3 {
            let (fa, fb) = FRACTIONS[k];
            let ex = (lattice_interp(p00.0, p10.0, p01.0, p11.0, fa, fb) - checks[k].0).abs();
            let ey = (lattice_interp(p00.1, p10.1, p01.1, p11.1, fa, fb) - checks[k].1).abs();
            let ek = ex + ey;
            if !ek.is_finite() {
                finite = false;
            } else {
                e = cmax(e, ek);
            }
        }
        let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) =
            (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
        for r in 0..3 {
            for s in 0..3 {
                let p = pt(r, s);
                if !p.0.is_finite() || !p.1.is_finite() {
                    finite = false;
                    continue;
                }
                lo_x = cmin(lo_x, p.0);
                hi_x = cmax(hi_x, p.0);
                lo_y = cmin(lo_y, p.1);
                hi_y = cmax(hi_y, p.1);
            }
        }
        *err = if finite { e } else { f64::NAN };
        *bad = !finite;
        let margin = 2.0 * e + 4.0;
        *outside = finite
            && (hi_x < -margin || lo_x > src_w as f64 + margin || hi_y < -margin || lo_y > src_h as f64 + margin);
        bbox[0] = f64::INFINITY;
        bbox[2] = f64::INFINITY;
        bbox[1] = f64::NEG_INFINITY;
        bbox[3] = f64::NEG_INFINITY;
        let mut any = false;
        for p in [p00, p10, p01, p11] {
            if !p.0.is_finite() || !p.1.is_finite() {
                continue;
            }
            any = true;
            bbox[0] = cmin(bbox[0], p.0);
            bbox[1] = cmax(bbox[1], p.0);
            bbox[2] = cmin(bbox[2], p.1);
            bbox[3] = cmax(bbox[3], p.1);
        }
        if !any {
            bbox.fill(f64::NAN);
        }
    };
    // OpenMP's schedule(static): a block of cells a thread.
    let threads = threads.max(1);
    let chunk = cells.div_ceil(threads).max(1);
    let (err, bad, outside, bbox) = (
        parallel::Shared::new(err),
        parallel::Shared::new(bad),
        parallel::Shared::new(outside),
        parallel::Shared::new(bbox),
    );
    parallel::region(threads, |t| {
        let (a, b) = ((t * chunk).min(cells), ((t + 1) * chunk).min(cells));
        // SAFETY: the threads' blocks are disjoint.
        let (err, bad, outside, bbox) = unsafe {
            (
                err.slice_mut(a..b),
                bad.slice_mut(a..b),
                outside.slice_mut(a..b),
                bbox.slice_mut(4 * a..4 * b),
            )
        };
        for (c, (((e, bd), o), bx)) in err
            .iter_mut()
            .zip(bad.iter_mut())
            .zip(outside.iter_mut())
            .zip(bbox.chunks_mut(4))
            .enumerate()
        {
            check(a + c, e, bd, o, bx);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(method: i32, bands: i32, h: i32, w: i32) -> Spec {
        Spec {
            method,
            bands,
            src_h: h,
            src_w: w,
            win_r0: 0,
            win_c0: 0,
            win_h: h,
            win_w: w,
            xscale: 1.0,
            yscale: 1.0,
            has_src_nodata: false,
            per_band: false,
            src_nodata: 0.0,
            dst_nodata: 0.0,
            dst_r0: 0,
            dst_c0: 0,
            dst_h: h,
            dst_w: w,
        }
    }

    fn identity() -> Coords<'static> {
        Coords {
            kind: AFFINE,
            offset: 0.5,
            gt: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            inv: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        }
    }

    #[test]
    fn the_identity_copies_with_every_method() {
        let src: Vec<f64> = (0..64).map(|v| v as f64 + 1.0).collect();
        for method in [NEAREST, BILINEAR, AVERAGE, MODE, MAX, MIN, MED, Q1, Q3, RMS] {
            let mut dst = vec![0.0; 64];
            let offset = if by_area(method) { 0.0 } else { 0.5 };
            let c = Coords { offset, ..identity() };
            warp(&src, &mut dst, &c, &spec(method, 1, 8, 8), 2);
            assert_eq!(dst, src, "method {method}");
        }
    }

    #[test]
    fn nodata_moves_off_the_destination_nodata() {
        let src = vec![5u8, 0, 7, 9];
        let mut dst = vec![1u8; 4];
        let mut sp = spec(NEAREST, 1, 2, 2);
        sp.has_src_nodata = true;
        sp.src_nodata = 7.0;
        sp.dst_nodata = 5.0;
        warp(&src, &mut dst, &identity(), &sp, 1);
        assert_eq!(dst, vec![4, 0, 5, 9]);
    }

    #[test]
    fn lattice_rows_interpolate_between_nodes() {
        // one cell at level 0: corners only, x = column, y = 2 row
        let grid0 = [0.0, 0.0, 16.0, 0.0, 0.0, 32.0, 16.0, 32.0];
        let level = [0i8];
        let node_off = [0i64];
        let c = Coords {
            kind: LATTICE,
            cells_x: 1,
            cells_y: 1,
            level: &level,
            grid0: &grid0,
            node_off: &node_off,
            ..Default::default()
        };
        let (mut x, mut y) = (vec![0.0; 16], vec![0.0; 16]);
        c.row(3, 0, &mut x, &mut y);
        for k in 0..16 {
            assert_eq!(x[k], k as f64);
            assert_eq!(y[k], 6.0);
        }
    }

    #[test]
    fn lattice_checks_of_an_affine_grid() {
        let (my, mx) = (1, 2);
        let (sy, sx) = (2 * my + 1, 2 * mx + 1);
        let mut g = vec![0.0; sy * sx * 2];
        for r in 0..sy {
            for s in 0..sx {
                g[2 * (r * sx + s)] = s as f64 * 8.0;
                g[2 * (r * sx + s) + 1] = r as f64 * 8.0;
            }
        }
        let (mut err, mut bad, mut out, mut bbox) = (vec![0.0; 2], vec![true; 2], vec![true; 2], vec![0.0; 8]);
        lattice_checks(&g, 1, my, mx, 100, 100, &mut err, &mut bad, &mut out, &mut bbox, 2);
        assert_eq!(err, vec![0.0, 0.0]);
        assert_eq!(bad, vec![false, false]);
        assert_eq!(out, vec![false, false]);
        assert_eq!(&bbox[4..], &[16.0, 32.0, 0.0, 16.0]);
    }
}
